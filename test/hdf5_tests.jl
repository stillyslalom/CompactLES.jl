# HDF5 extension tests. Included by runtests.jl when HDF5 is loadable, and
# runnable under mpiexec for the decomposition-independent restart, which is the
# property the shared-file checkpoint exists for and which no serial run can
# demonstrate.
#
#   julia --project=. test/hdf5_tests.jl                  # serial
#   mpiexec -n 4 julia --project=. test/hdf5_tests.jl     # decomposed
#
# HDF5 is a weak dependency, so it is not loadable from the package environment
# alone. Run these from an environment that has both, or through `Pkg.test`.

if !@isdefined(CL)
    using MPI
    MPI.Init(threadlevel=:funneled)
    using CompactLES
    using Test
    const CL = CompactLES
end
using HDF5

@testset "HDF5 extension: shared-file checkpoint" begin
    comm = MPI.COMM_WORLD
    np = MPI.Comm_size(comm)
    rank = MPI.Comm_rank(comm)
    per3h = ntuple(_ -> (PeriodicBC(), PeriodicBC()), 3)

    @test hdf5_available()
    # Which backend is in use is a property of the libhdf5 binary, not of the
    # run. Both produce the same file; only the cost differs.
    @test hdf5_parallel() isa Bool

    # A block description is independent of Decomp, so that a refinement patch
    # can be written by the same path later.
    region = BlockRegion((4, 0, 2), (8, 16, 4))
    @test CL.region_ranges(region) == (5:12, 1:16, 3:6)

    eos = IdealMixture([IdealSpecies{Float64}("a", 1.0, 1.4),
                        IdealSpecies{Float64}("b", 2.0, 1.6)])
    # 72 along the split dimension keeps 9 points per rank at np = 8, which the
    # C8 filter closure requires; the transverse dimensions stay undivided.
    mk() = begin
        s = Solver(bcs=per3h, n_global=(72, 16, 16), L_domain=(1.0, 1.0, 1.0),
                   eos=eos, art=ArtParams(enabled=false), dims=(np, 1, 1))
        Q = allocate_state(s)
        initialize!(s, Q, (x, y, z) -> Prim(Y=(0.3, 0.7), u=(x, 2y, 3z),
                                            p=1 + x, rho=1 + y + 2z))
        s, Q
    end

    dir = rank == 0 ? mktempdir() : ""
    dir = MPI.bcast(dir, comm; root=0)
    stem = joinpath(dir, "state")

    s1, Q1 = mk()
    s1.t = 1.25
    s1.step = 17
    save_checkpoint_hdf5(s1, Q1, stem)
    MPI.Barrier(comm)
    @test isfile(stem * ".h5")

    # One file, whatever the rank count: this is the whole point of the shared
    # write. The per-rank checkpoint writes np of them.
    if rank == 0
        @test length(filter(endswith(".h5"), readdir(dir))) == 1
        h5open(stem * ".h5", "r") do file
            @test size(file["state/Q"]) == (72, 16, 16, 6)
            @test read(file["meta/t"]) == 1.25
            @test read(file["meta/step"]) == 17
            @test Int.(read(file["meta/n_global"])) == [72, 16, 16]
            @test read(file["meta/component_names"])[1] == "rho_a"
            # Fixed-length rather than variable-length, which parallel HDF5
            # refuses to write. A serialized-backend workstation writes a VL
            # string happily, so nothing else here would catch the regression.
            @test !HDF5.API.h5t_is_variable_str(
                HDF5.datatype(file["meta/component_names"]))
        end
    end
    MPI.Barrier(comm)

    # Round trip onto the same decomposition, bit for bit.
    s2, Q2 = mk()
    fill!(Q2, 0.0)
    s2.t = 0.0
    s2.step = 0
    load_checkpoint_hdf5!(s2, Q2, stem)
    @test s2.t == 1.25
    @test s2.step == 17
    d = 0.0
    for c in 1:s2.equations.n_cons, k in 1:s2.decomp.n_local[3],
        j in 1:s2.decomp.n_local[2], i in 1:s2.decomp.n_local[1]
        I = gidx(s2, i, j, k)
        d = max(d, abs(Q2[I, c] - Q1[I, c]))
    end
    @test MPI.Allreduce(d, max, comm) == 0.0

    # The state is stored in global index space, so a mismatched grid is
    # detected rather than silently misread.
    s3 = Solver(bcs=per3h, n_global=(72, 16, 12), L_domain=(1.0, 1.0, 1.0),
                eos=eos, art=ArtParams(enabled=false), dims=(np, 1, 1))
    Q3 = allocate_state(s3)
    @test_throws ErrorException load_checkpoint_hdf5!(s3, Q3, stem)

    # Everything the state's interpretation depends on is in the header. Each
    # solver below reads the file cleanly if its own field is not checked: the
    # array has the right shape in every case and means something else. The
    # expected message is asserted so that each case exercises the check it is
    # there for rather than tripping an earlier one.
    reject(s, msg) = begin
        Qx = allocate_state(s)
        @test_throws msg load_checkpoint_hdf5!(s, Qx, stem)
    end

    # A different species set of the same size: same n_cons, same extent, same
    # metric. Only the component names separate them.
    other = IdealMixture([IdealSpecies{Float64}("c", 1.0, 1.4),
                          IdealSpecies{Float64}("d", 2.0, 1.6)])
    reject(Solver(bcs=per3h, n_global=(72, 16, 16), L_domain=(1.0, 1.0, 1.0),
                  eos=other, art=ArtParams(enabled=false), dims=(np, 1, 1)),
           "conserved component mismatch")

    # A different species count, which also moves n_cons.
    reject(Solver(bcs=per3h, n_global=(72, 16, 16), L_domain=(1.0, 1.0, 1.0),
                  eos=IdealMixture([IdealSpecies{Float64}("a", 1.0, 1.4)]),
                  art=ArtParams(enabled=false), dims=(np, 1, 1)),
           "conserved layout mismatch")

    # The same grid dimensions over a longer domain, which n_global cannot see
    # and the coordinates can.
    reject(Solver(bcs=per3h, n_global=(72, 16, 16), L_domain=(2.0, 1.0, 1.0),
                  eos=eos, art=ArtParams(enabled=false), dims=(np, 1, 1)),
           "grid coordinate mismatch")

    # The same grid shifted, likewise invisible to every other field.
    reject(Solver(bcs=per3h, n_global=(72, 16, 16), L_domain=(1.0, 1.0, 1.0),
                  origin=(0.5, 0.0, 0.0), eos=eos,
                  art=ArtParams(enabled=false), dims=(np, 1, 1)),
           "grid coordinate mismatch")

    # A stretched dimension against the uniform grid it was written on. The two
    # agree on extent, origin, species and metric and differ only point by
    # point, which is why the coordinates are stored rather than the extent.
    # A stretched dimension must be non-periodic, so this pair needs a
    # checkpoint of its own rather than the periodic one above.
    wall1 = ((SlipWallBC(), SlipWallBC()), per3h[2], per3h[3])
    mkw(st) = begin
        s = Solver(bcs=wall1, n_global=(72, 16, 16), L_domain=(1.0, 1.0, 1.0),
                   stretch=(st, nothing, nothing), eos=eos,
                   art=ArtParams(enabled=false), dims=(np, 1, 1))
        s, allocate_state(s)
    end
    su, Qu = mkw(nothing)
    initialize!(su, Qu, (x, y, z) -> Prim(Y=(0.3, 0.7), p=1.0, rho=1.0))
    uniform_stem = joinpath(dir, "uniform1")
    save_checkpoint_hdf5(su, Qu, uniform_stem)
    MPI.Barrier(comm)
    ss, Qs = mkw(sine_cluster(0.0, 1.0, 0.5, 0.3))
    @test_throws "grid coordinate mismatch" load_checkpoint_hdf5!(ss, Qs,
                                                                 uniform_stem)
    # ...and the same file loads onto the grid it was written on.
    su2, Qu2 = mkw(nothing)
    load_checkpoint_hdf5!(su2, Qu2, uniform_stem)
    @test su2.step == su.step

    # A different metric on the same grid dimensions: the state array is the
    # same shape and every momentum component means something else.
    reject(Solver(bcs=((SlipWallBC(), SlipWallBC()), per3h[2], per3h[3]),
                  n_global=(72, 16, 16), L_domain=(1.0, 2π, 1.0),
                  origin=(0.5, 0.0, 0.0), metric=CylindricalMetric(),
                  eos=eos, art=ArtParams(enabled=false), dims=(np, 1, 1)),
           "metric mismatch")

    MPI.Barrier(comm)
    rank == 0 && rm(dir; recursive=true)
    MPI.Barrier(comm)
end

# The property the per-rank checkpoint cannot provide: a file written under one
# decomposition restored under another. Run the writer under mpiexec and the
# reader here, against the analytic initial condition rather than against a
# retained array, since the two runs do not share memory.
@testset "HDF5 extension: restart across a different rank count" begin
    comm = MPI.COMM_WORLD
    np = MPI.Comm_size(comm)
    rank = MPI.Comm_rank(comm)
    per3h = ntuple(_ -> (PeriodicBC(), PeriodicBC()), 3)
    ic = (x, y, z) -> Prim(u=(sin(2π * x), 0.3y, 0.0), p=1 + 0.1cos(2π * y),
                           rho=1 + 0.2z)

    dir = rank == 0 ? mktempdir() : ""
    dir = MPI.bcast(dir, comm; root=0)
    stem = joinpath(dir, "across")

    # Split along x to write and along y to read: a genuinely different block
    # decomposition of the same global array. Both dimensions are 72 so that
    # every rank keeps at least the 9 points the C8 filter closure needs.
    build(dims) = begin
        s = Solver(bcs=per3h, n_global=(72, 72, 12), L_domain=(1.0, 1.0, 1.0),
                   art=ArtParams(enabled=false), dims=dims)
        Q = allocate_state(s)
        initialize!(s, Q, ic)
        s, Q
    end
    sw, Qw = build((np, 1, 1))
    sw.t = 0.5
    sw.step = 9
    save_checkpoint_hdf5(sw, Qw, stem)
    MPI.Barrier(comm)

    # Read it back on a DIFFERENT process grid over the same communicator. At
    # np > 1 this is a genuinely different block decomposition of the same
    # global array; at np == 1 it degenerates to the same one, and the check
    # still verifies the global-index round trip.
    sr = Solver(bcs=per3h, n_global=(72, 72, 12), L_domain=(1.0, 1.0, 1.0),
                art=ArtParams(enabled=false), dims=(1, np, 1))
    Qr = allocate_state(sr)
    load_checkpoint_hdf5!(sr, Qr, stem)
    @test sr.t == 0.5
    @test sr.step == 9

    # Against the analytic IC on the reading decomposition: if the hyperslab
    # offsets were wrong the values would land on the wrong coordinates, which a
    # comparison against a retained array on the writing layout would miss.
    ref = allocate_state(sr)
    initialize!(sr, ref, ic)
    d = 0.0
    for c in 1:sr.equations.n_cons, k in 1:sr.decomp.n_local[3],
        j in 1:sr.decomp.n_local[2], i in 1:sr.decomp.n_local[1]
        I = gidx(sr, i, j, k)
        d = max(d, abs(Qr[I, c] - ref[I, c]))
    end
    @test MPI.Allreduce(d, max, comm) == 0.0

    MPI.Barrier(comm)
    rank == 0 && rm(dir; recursive=true)
    MPI.Barrier(comm)
end

@testset "HDF5 extension: field dump and XDMF3 sidecar" begin
    comm = MPI.COMM_WORLD
    np = MPI.Comm_size(comm)
    rank = MPI.Comm_rank(comm)
    per3h = ntuple(_ -> (PeriodicBC(), PeriodicBC()), 3)

    dir = rank == 0 ? mktempdir() : ""
    dir = MPI.bcast(dir, comm; root=0)

    nx, ny, nz = 72, 16, 12
    rho_of = (x, y, z) -> 1 + x + 100y + 10000z
    s = Solver(bcs=per3h, n_global=(nx, ny, nz), L_domain=(1.0, 1.0, 1.0),
               art=ArtParams(enabled=false), dims=(np, 1, 1))
    Q = allocate_state(s)
    initialize!(s, Q, (x, y, z) -> Prim(u=(x, 10y, 100z), p=1.0,
                                        rho=rho_of(x, y, z)))
    s.t = 0.75
    save_hdf5(s, Q, joinpath(dir, "dump"); fields=(:rho, :velocity))
    MPI.Barrier(comm)

    if rank == 0
        # One file per frame at any rank count, plus the sidecar a reader opens.
        @test isfile(joinpath(dir, "dump.h5"))
        @test isfile(joinpath(dir, "dump.xmf"))
        @test length(filter(endswith(".h5"), readdir(dir))) == 1

        h5open(joinpath(dir, "dump.h5")) do h
            @test size(h["fields/rho"]) == (nx, ny, nz)
            @test size(h["fields/velocity"]) == (3, nx, ny, nz)
            @test size(h["grid/x"]) == (nx,)
            @test read(h["meta/t"]) == 0.75

            # Every rank's hyperslab landed where the global index says it
            # should. A misplaced block would still be self-consistent within
            # its own piece, so the comparison is against the analytic field
            # over the whole global array.
            rho = read(h["fields/rho"])
            vel = read(h["fields/velocity"])
            er, ev = 0.0, 0.0
            for k in 1:nz, j in 1:ny, i in 1:nx
                x = global_xcoord(s, 1, i)
                y = global_xcoord(s, 2, j)
                z = global_xcoord(s, 3, k)
                er = max(er, abs(rho[i, j, k] - Float32(rho_of(x, y, z))))
                ev = max(ev, abs(vel[1, i, j, k] - Float32(x)),
                             abs(vel[2, i, j, k] - Float32(10y)),
                             abs(vel[3, i, j, k] - Float32(100z)))
            end
            @test er < 1e-2                     # Float32 payload at magnitude 1e4
            @test ev < 1e-4

            # The sidecar declares the on-disk dimension order, which is the
            # reverse of the Julia one. If these disagree the reader transposes
            # the field and nothing reports an error.
            for (name, want) in (("fields/rho", UInt64[nz, ny, nx]),
                                 ("fields/velocity", UInt64[nz, ny, nx, 3]))
                dims, _ = HDF5.API.h5s_get_simple_extent_dims(HDF5.dataspace(h[name]))
                @test dims == want
            end
        end

        xmf = read(joinpath(dir, "dump.xmf"), String)
        @test occursin("TopologyType=\"3DRectMesh\" Dimensions=\"$nz $ny $nx\"", xmf)
        @test occursin("GeometryType=\"VXVYVZ\"", xmf)
        @test occursin("<Time Value=\"0.75\"/>", xmf)
        @test occursin("Dimensions=\"$nz $ny $nx\" NumberType=\"Float\" " *
                       "Precision=\"4\" Format=\"HDF\">dump.h5:/fields/rho", xmf)
        @test occursin("AttributeType=\"Vector\"", xmf)
        @test occursin("Dimensions=\"$nz $ny $nx 3\"", xmf)
        # Relative to the sidecar's own directory, so the pair stays portable.
        @test !occursin(dir, xmf)
    end
    MPI.Barrier(comm)

    # Subsampling reduces the declared grid, and the sidecar follows it.
    save_hdf5(s, Q, joinpath(dir, "coarse"); fields=(:rho,), stride=(2, 2, 1))
    MPI.Barrier(comm)
    if rank == 0
        h5open(joinpath(dir, "coarse.h5")) do h
            @test size(h["fields/rho"]) == (36, 8, 12)
            @test size(h["grid/x"]) == (36,)
            # The retained stations are the odd global ones, not a re-spacing.
            @test read(h["grid/x"]) ≈ [global_xcoord(s, 1, i) for i in 1:2:nx]
        end
        @test occursin("Dimensions=\"12 8 36\"",
                       read(joinpath(dir, "coarse.xmf"), String))
    end
    MPI.Barrier(comm)

    # A resolved angle writes an explicit position per point and a curvilinear
    # topology, matching the rule the VTK path uses.
    cyl = Solver(bcs=((SlipWallBC(), SlipWallBC()), per3h[2], per3h[3]),
                 n_global=(12, 72, 10), L_domain=(1.0, 2π, 1.0),
                 origin=(0.5, 0.0, 0.0), metric=CylindricalMetric(),
                 art=ArtParams(enabled=false), dims=(1, np, 1))
    Qc = allocate_state(cyl)
    initialize!(cyl, Qc, (r, θ, z) -> Prim(u=(0.0, 0.5, 0.0), p=1.0, rho=1.0))
    save_hdf5(cyl, Qc, joinpath(dir, "annulus"); fields=(:rho, :velocity))
    MPI.Barrier(comm)
    if rank == 0
        h5open(joinpath(dir, "annulus.h5")) do h
            @test size(h["grid/points"]) == (3, 12, 72, 10)
            @test !haskey(h, "grid/x")
            pts = read(h["grid/points"])
            radii = [hypot(pts[1, i, j, k], pts[2, i, j, k])
                     for i in 1:12, j in 1:72, k in 1:10]
            @test minimum(radii) ≈ 0.5 rtol = 1e-12
            @test maximum(radii) ≈ 1.5 rtol = 1e-12
            # Velocity is rotated into the frame the positions are written in.
            vel = read(h["fields/velocity"])
            radial = 0.0
            for k in 1:10, j in 1:72, i in 1:12
                x, y = pts[1, i, j, k], pts[2, i, j, k]
                radial = max(radial, abs(vel[1, i, j, k] * x +
                                         vel[2, i, j, k] * y) / hypot(x, y))
            end
            @test radial < 1e-6
        end
        xmf = read(joinpath(dir, "annulus.xmf"), String)
        @test occursin("TopologyType=\"3DSMesh\"", xmf)
        @test occursin("GeometryType=\"XYZ\"", xmf)
        @test occursin("annulus.h5:/grid/points", xmf)
    end

    MPI.Barrier(comm)
    rank == 0 && rm(dir; recursive=true)
    MPI.Barrier(comm)
end

@testset "HDF5 extension: slicing" begin
    comm = MPI.COMM_WORLD
    np = MPI.Comm_size(comm)
    rank = MPI.Comm_rank(comm)
    per3h = ntuple(_ -> (PeriodicBC(), PeriodicBC()), 3)

    dir = rank == 0 ? mktempdir() : ""
    dir = MPI.bcast(dir, comm; root=0)

    nx, ny, nz = 72, 16, 12
    rho_of = (x, y, z) -> 1 + x + 100y + 10000z
    s = Solver(bcs=per3h, n_global=(nx, ny, nz), L_domain=(1.0, 1.0, 1.0),
               art=ArtParams(enabled=false), dims=(np, 1, 1))
    Q = allocate_state(s)
    initialize!(s, Q, (x, y, z) -> Prim(u=(0.1, 0, 0), p=1.0, rho=rho_of(x, y, z)))

    # Across the split dimension, so most ranks hold no part of the plane and
    # write no hyperslab; the dataset is still the full plane.
    gx = nx ÷ 2 + 1
    save_hdf5(s, Q, joinpath(dir, "sx"); fields=(:rho,), slice=(1, gx))
    MPI.Barrier(comm)
    if rank == 0
        h5open(joinpath(dir, "sx.h5")) do h
            @test size(h["fields/rho"]) == (1, ny, nz)
            @test size(h["grid/x"]) == (1,)
            @test read(h["grid/x"]) ≈ [global_xcoord(s, 1, gx)]
            rho = read(h["fields/rho"])
            e = 0.0
            for k in 1:nz, j in 1:ny
                want = rho_of(global_xcoord(s, 1, gx), global_xcoord(s, 2, j),
                              global_xcoord(s, 3, k))
                e = max(e, abs(rho[1, j, k] - Float32(want)))
            end
            @test e < 1e-2                        # Float32 at magnitude 1e4
        end
        @test occursin("Dimensions=\"$nz $ny 1\"",
                       read(joinpath(dir, "sx.xmf"), String))
    end
    MPI.Barrier(comm)

    # Across an undivided dimension, so every rank contributes part of the plane.
    gz = 5
    save_hdf5(s, Q, joinpath(dir, "sz"); fields=(:rho,), slice=(3, gz))
    MPI.Barrier(comm)
    if rank == 0
        h5open(joinpath(dir, "sz.h5")) do h
            @test size(h["fields/rho"]) == (nx, ny, 1)
            rho = read(h["fields/rho"])
            e = 0.0
            for j in 1:ny, i in 1:nx
                want = rho_of(global_xcoord(s, 1, i), global_xcoord(s, 2, j),
                              global_xcoord(s, 3, gz))
                e = max(e, abs(rho[i, j, 1] - Float32(want)))
            end
            @test e < 1e-2
        end
    end
    MPI.Barrier(comm)

    rank == 0 && rm(dir; recursive=true)
    MPI.Barrier(comm)
end
