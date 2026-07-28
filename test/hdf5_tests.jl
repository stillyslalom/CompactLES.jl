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
