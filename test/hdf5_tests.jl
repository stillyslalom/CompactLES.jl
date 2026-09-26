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

    # A block description is independent of Decomp, allowing a refinement patch
    # to use the same write path later.
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
            # The state's element type, recorded and used: written as Float64
            # here, and a Float32 solver would write a Float32 dataset that a
            # Float64 one is refused rather than allowed to widen.
            @test eltype(file["state/Q"]) == Float64
            @test read(file["meta/eltype"])[1] == "Float64"
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
    # expected message is asserted to ensure each case exercises its intended
    # check, not an earlier one.
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

    # The mutable run state. (t, step) alone leaves behind a retry's reduced
    # CFL, the step history the growth cap and `filter_weight` read, and a
    # boundary face that has already switched — which on a switch that changes
    # the collective pattern is a deadlock on resume, not a wrong answer.
    mkstate(bcs) = begin
        s = Solver(bcs=bcs, n_global=(72, 16, 16), L_domain=(1.0, 1.0, 1.0),
                   eos=eos, art=ArtParams(enabled=false), dims=(np, 1, 1))
        Q = allocate_state(s)
        initialize!(s, Q, (x, y, z) -> Prim(Y=(0.3, 0.7), p=1.0, rho=1.0))
        s, Q
    end
    written_face = SwitchableBC(SlipWallBC(), ExtrapolationBC())
    sst, Qst = mkstate(((written_face, SlipWallBC()), per3h[2], per3h[3]))
    sst.cfl = 0.125
    sst.dt_prev = 1.5e-4
    sst.rate_prev = 987.5
    sst.filter_rate_prev = (12.5, 250.0, 0.0)
    switch!(written_face)
    state_stem = joinpath(dir, "runstate")
    save_checkpoint_hdf5(sst, Qst, state_stem)
    MPI.Barrier(comm)

    read_face = SwitchableBC(SlipWallBC(), ExtrapolationBC())
    sback, Qback = mkstate(((read_face, SlipWallBC()), per3h[2], per3h[3]))
    load_checkpoint_hdf5!(sback, Qback, state_stem)
    @test sback.cfl == 0.125
    @test sback.dt_prev == 1.5e-4
    @test sback.rate_prev == 987.5
    @test sback.filter_rate_prev == (12.5, 250.0, 0.0)
    @test switched(read_face)

    # A face the file describes as switchable where this solver has a plain
    # condition: the boundary would silently differ for the rest of the run.
    # The configuration record refuses it first; allowing the boundary
    # change leaves the switch record in force.
    splain, Qplain = mkstate(((SlipWallBC(), SlipWallBC()), per3h[2], per3h[3]))
    @test_throws "configuration mismatch" load_checkpoint_hdf5!(splain, Qplain,
                                                                state_stem)
    @test_throws "boundary mismatch" load_checkpoint_hdf5!(splain, Qplain,
                                                           state_stem;
                                                           allow=(:boundaries,))

    # The configuration record: the same species names over another gamma are
    # refused on every rank, since every rank reads the one record; a numerics
    # change is refused unless allowed. The record's groups are summarized by
    # digest in `config/digests`.
    lean = IdealMixture([IdealSpecies{Float64}("a", 1.0, 1.4),
                         IdealSpecies{Float64}("b", 2.0, 1.5)])
    mkrec(; kw...) = Solver(bcs=per3h, n_global=(72, 16, 16), L_domain=(1.0, 1.0, 1.0),
                            art=ArtParams(enabled=false), dims=(np, 1, 1); kw...)
    refused = try
        load_checkpoint_hdf5!(mkrec(eos=lean), allocate_state(mkrec(eos=lean)), stem)
        0
    catch e
        occursin("eos.sp[2].gamma", sprint(showerror, e)) ? 1 : 0
    end
    @test MPI.Allreduce(refused, +, comm) == np
    c8 = mkrec(eos=eos, deriv=lele_d1_8())
    @test_throws "allow = (:numerics,)" load_checkpoint_hdf5!(c8, allocate_state(c8),
                                                               stem)
    Qc8 = allocate_state(c8)
    load_checkpoint_hdf5!(c8, Qc8, stem; allow=(:numerics,))
    @test c8.step == 17
    if rank == 0
        h5open(stem * ".h5", "r") do file
            @test read(file["meta/format"]) == 6
            digests = String.(read(file["config/digests"]))
            @test startswith(digests[1], "thermodynamics fnv1a64 ")
            @test "eos.sp[2].gamma" in String.(read(file["config/paths"]))
        end
    end
    MPI.Barrier(comm)

    # A file of the previous format has no record and loads with a warning,
    # nothing compared: this format's file without `config`, format word 5.
    legacy_stem = joinpath(dir, "legacy")
    if rank == 0
        cp(stem * ".h5", legacy_stem * ".h5")
        h5open(legacy_stem * ".h5", "r+") do file
            delete_object(file, "config")
            delete_object(file, "meta/format")
            file["meta/format"] = 5
        end
    end
    MPI.Barrier(comm)
    old = mkrec(eos=lean)
    Qold = allocate_state(old)
    if rank == 0
        @test_logs (:warn, r"no configuration record") match_mode=:any begin
            load_checkpoint_hdf5!(old, Qold, legacy_stem)
        end
    else
        load_checkpoint_hdf5!(old, Qold, legacy_stem)
    end
    @test old.step == 17

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
    # decomposition of the same global array. Both dimensions are 72, keeping at
    # least the 9 points the C8 filter closure needs on every rank.
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

# The coefficient record: with the artificial properties on, a restart from
# the shared file continues the uninterrupted run bit for bit on the same
# rank count, because the arrays the next step is sized from are restored
# with the state. The per-rank path is pinned the same way in io_tests.jl.
@testset "HDF5 extension: restart continues the run bit for bit" begin
    comm = MPI.COMM_WORLD
    np = MPI.Comm_size(comm)
    rank = MPI.Comm_rank(comm)
    per3h = ntuple(_ -> (PeriodicBC(), PeriodicBC()), 3)
    wall = (SlipWallBC(), SlipWallBC())
    mk(; kw...) = Solver(n_global=(144, 1, 1), L_domain=(1.0, 1.0, 1.0),
                         bcs=(wall, per3h[2], per3h[3]), cfl=0.4,
                         dims=(np, 1, 1); kw...)
    ic(x, y, z) = x < 0.5 ? Prim(u=(0, 0, 0), p=1.0, rho=1.0) :
                            Prim(u=(0, 0, 0), p=0.1, rho=0.125)
    dir = rank == 0 ? mktempdir() : ""
    dir = MPI.bcast(dir, comm; root=0)
    stem = joinpath(dir, "sod")

    s = mk()
    Q = allocate_state(s)
    initialize!(s, Q, ic)
    run!(s, Q; tfinal=1.0, nmax=20)
    save_checkpoint_hdf5(s, Q, stem)
    MPI.Barrier(comm)
    run!(s, Q; tfinal=1.0, nmax=40)
    if rank == 0
        h5open(stem * ".h5", "r") do file
            @test size(file["state/art"]) == (144, 1, 1, 4)
            @test read(file["meta/n_art"]) == 4
            @test read(file["meta/n_levels"]) == 1
        end
    end
    MPI.Barrier(comm)

    r = mk()
    Qr = allocate_state(r)
    load_checkpoint_hdf5!(r, Qr, stem)
    @test r.step == 20
    run!(r, Qr; tfinal=1.0, nmax=40)
    @test r.t == s.t
    inner = CL.interior(s.decomp)
    d = maximum(abs.(parent(Qr)[inner, :] .- parent(Q)[inner, :]))
    d = max(d, maximum(abs.(r.beta_art[inner] .- s.beta_art[inner])))
    @test MPI.Allreduce(d, max, comm) == 0.0

    off = mk(art=ArtParams(enabled=false))
    @test_throws "artificial-property mismatch" load_checkpoint_hdf5!(
        off, allocate_state(off), stem)

    MPI.Barrier(comm)
    rank == 0 && rm(dir; recursive=true)
    MPI.Barrier(comm)
end

# The hierarchy checkpoint. On the rank count that wrote the file the stored
# ownership is restored and the regridded run continues bit for bit; the
# layout tables are the readable form of what the per-rank checkpoint stores.
@testset "HDF5 extension: hierarchy checkpoint on the same rank count" begin
    comm = MPI.COMM_WORLD
    np = MPI.Comm_size(comm)
    rank = MPI.Comm_rank(comm)
    per3h = ntuple(_ -> (PeriodicBC(), PeriodicBC()), 3)
    wall = (SlipWallBC(), SlipWallBC())
    ic(x, y, z) = x < 0.5 ? Prim(u=(0, 0, 0), p=1.0, rho=1.0) :
                            Prim(u=(0, 0, 0), p=0.1, rho=0.125)
    # 201 root nodes keep 25 per rank at np = 8; a tile of 8 is 25 fine nodes.
    mk() = Solver(n_global=(201, 1, 1), L_domain=(1.0, 1.0, 1.0),
                  bcs=(wall, per3h[2], per3h[3]), cfl=0.2, subcycle=true,
                  regrid_interval=5, tile=8, dims=(np, 1, 1),
                  refine=BlockRegion((85, 0, 0), (31, 1, 1)))
    dir = rank == 0 ? mktempdir() : ""
    dir = MPI.bcast(dir, comm; root=0)
    stem = joinpath(dir, "tiled")

    s = mk()
    states = allocate_state(s)
    initialize!(s, states, ic)
    run!(s, states; tfinal=1.0, nmax=23)
    spec = getfield(s, :regrid)
    regs = level_regions(s, 1)
    created = copy(spec.created)
    owners = copy(s.levels[2].owners)
    checks = spec.checks
    save_checkpoint_hdf5(s, states, stem)
    MPI.Barrier(comm)
    run!(s, states; tfinal=1.0, nmax=46)
    if rank == 0
        h5open(stem * ".h5", "r") do file
            @test read(file["hierarchy/np"]) == np
            @test read(file["hierarchy/n_levels"]) == 1
            R = read(file["hierarchy/level1/regions"])
            @test size(R) == (6, length(regs))
            @test [BlockRegion(Tuple(Int.(R[1:3, k])), Tuple(Int.(R[4:6, k])))
                   for k in axes(R, 2)] == regs
            @test read(file["hierarchy/regrid/tile"]) == 8
            @test read(file["hierarchy/regrid/checks"]) == checks
            @test size(file["levels/1/tiles/1/Q"]) == (25, 1, 1, 5)
            @test size(file["levels/1/tiles/1/art"]) == (25, 1, 1, 4)
        end
    end
    MPI.Barrier(comm)

    r = mk()
    sr = allocate_state(r)
    load_checkpoint_hdf5!(r, sr, stem)
    @test r.step == 23
    @test level_regions(r, 1) == regs
    @test r.levels[2].owners == owners
    @test getfield(r, :regrid).created == created
    run!(r, sr; tfinal=1.0, nmax=46)
    @test r.t == s.t
    @test level_regions(r, 1) == level_regions(s, 1)
    d = 0.0
    for i in eachindex(states)
        inner = CL.interior(s.patches[i].decomp)
        d = max(d, maximum(abs.(parent(sr[i])[inner, :] .- parent(states[i])[inner, :])))
    end
    @test MPI.Allreduce(d, max, comm) == 0.0

    MPI.Barrier(comm)
    rank == 0 && rm(dir; recursive=true)
    MPI.Barrier(comm)
end

# A tiled, regridded run that starts with no tiles: the file written before
# the first tag records a level with no tables, the one written after records
# the tiles, and a solver set up unrefined continues from either bit for bit.
@testset "HDF5 extension: hierarchy checkpoint of an unrefined start" begin
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    per = (PeriodicBC(), PeriodicBC())
    ramp(t) = clamp((t - 0.04) / 0.01, 0.0, 1.0)
    inflow = DirichletBC((x, y, z, t) -> (w = ramp(t);
        Prim(rho=1 + 0.8621w, u=(0.8216w, 0.0, 0.0), p=1 + 1.4583w)))
    prob = Problem(eos=IdealSpecies("gas"; gamma=1.4, R=1.0),
                   domain=((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)),
                   bcs=((inflow, SlipWallBC()), per, per),
                   ic=(x, y, z) -> Prim(rho=1.0, u=(0.0, 0.0, 0.0), p=1.0))
    num = Numerics(n_global=(121, 1, 1), cfl=0.3,
                   amr=AMR(initial=:sensor, tile=8, regrid_interval=5,
                           tag_buffer=2, subcycle=true))
    dir = rank == 0 ? mktempdir() : ""
    dir = MPI.bcast(dir, comm; root=0)
    s, states = setup(prob, num)
    run!(s, states; tfinal=1.0, nmax=15)
    @test isempty(level_regions(s, 1))
    save_checkpoint_hdf5(s, states, joinpath(dir, "before"))
    run!(s, states; tfinal=1.0, nmax=50)
    regs50 = level_regions(s, 1)
    @test !isempty(regs50)
    save_checkpoint_hdf5(s, states, joinpath(dir, "after"))
    run!(s, states; tfinal=1.0, nmax=80)
    MPI.Barrier(comm)
    for (stem, want) in (("before", BlockRegion[]), ("after", regs50))
        r, rs = setup(prob, num)
        load_checkpoint_hdf5!(r, rs, joinpath(dir, stem))
        @test level_regions(r, 1) == want
        run!(r, rs; tfinal=1.0, nmax=80)
        @test r.t == s.t && level_regions(r, 1) == level_regions(s, 1)
        d = length(rs) == length(states) ? 0.0 : Inf
        if isfinite(d)
            for i in eachindex(states)
                inner = CL.interior(s.patches[i].decomp)
                d = max(d, maximum(abs.(parent(rs[i])[inner, :] .-
                                        parent(states[i])[inner, :])))
            end
        end
        @test MPI.Allreduce(d, max, comm) == 0.0
    end
    MPI.Barrier(comm)
    rank == 0 && rm(dir; recursive=true)
    MPI.Barrier(comm)
end

# The property the hierarchy checkpoint exists for: a file written on one
# rank count restored onto another, the level's tiles partitioned afresh. The
# writer runs on the first half of the ranks and the reader on all of them,
# built with a different initial region so that the restart rebuilds the
# level from the recorded twelve tiles; the continued wave error agrees with
# the writer's continuation to round-off, the tier a different decomposition
# of the same tiles holds. The smooth wave tags nothing, so the tiles are held
# by their lifetime; without it the first check would remove them all.
@testset "HDF5 extension: hierarchy restart across a different rank count" begin
    comm = MPI.COMM_WORLD
    np = MPI.Comm_size(comm)
    rank = MPI.Comm_rank(comm)
    per3h = ntuple(_ -> (PeriodicBC(), PeriodicBC()), 3)
    u0 = 0.5
    wave(x, y, z) = Prim(u=(u0, 0, 0), p=1.0, rho=1.0 + 0.2 * sin(x))
    mk(refine, sub) = Solver(n_global=(192, 1, 1), L_domain=(2π, 1.0, 1.0),
                             bcs=per3h, art=ArtParams(enabled=false),
                             filter_interval=0, subcycle=true, tile=8,
                             regrid_interval=5, tile_lifetime=100,
                             refine=refine, comm=sub)
    function wave_error(solver, states)
        e = 0.0
        for (ps, Q) in CL.eachpatch(solver, states)
            for i in 1:ps.decomp.n_local[1]
                I = gidx(ps, i, 1, 1)
                e = max(e, abs(Q[I, 1] - (1.0 + 0.2 * sin(xcoord(ps, 1, i) -
                                                          u0 * solver.t))))
            end
        end
        return MPI.Allreduce(e, max, solver.comm)
    end
    dir = rank == 0 ? mktempdir() : ""
    dir = MPI.bcast(dir, comm; root=0)
    stem = joinpath(dir, "twelve")

    half = max(1, np ÷ 2)
    inside = rank < half
    sub = MPI.Comm_split(comm, inside ? 0 : nothing, rank)
    e_sub, t_sub = 0.0, 0.0
    if inside
        sw = mk(BlockRegion((40, 0, 0), (96, 1, 1)), sub)
        states = allocate_state(sw)
        initialize!(sw, states, wave)
        run!(sw, states; tfinal=1.0, nmax=10)
        @test length(level_regions(sw, 1)) == 12
        save_checkpoint_hdf5(sw, states, stem)
        run!(sw, states; tfinal=1.0, nmax=20)
        e_sub, t_sub = wave_error(sw, states), sw.t
    end
    MPI.Barrier(comm)
    e_sub = MPI.bcast(e_sub, comm; root=0)
    t_sub = MPI.bcast(t_sub, comm; root=0)

    sr = mk(BlockRegion((56, 0, 0), (48, 1, 1)), comm)
    @test length(level_regions(sr, 1)) == 6
    sr_states = allocate_state(sr)
    load_checkpoint_hdf5!(sr, sr_states, stem)
    @test sr.step == 10
    @test length(level_regions(sr, 1)) == 12
    @test length(sr_states) == length(sr.patches)
    run!(sr, sr_states; tfinal=1.0, nmax=20)
    @test abs(sr.t - t_sub) < 1e-14
    @test abs(wave_error(sr, sr_states) - e_sub) < 1e-12

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

# A minimal well-formedness check for the XDMF this package writes: elements,
# double-quoted attributes and text, with the declaration and DOCTYPE skipped.
# It throws on an unbalanced or unrecognized construct, so a truncated or
# malformed collection fails the parse rather than an assertion further on.
mutable struct XNode
    name::String
    attrs::Dict{String,String}
    children::Vector{XNode}
    text::String
end

function parse_xml(s::AbstractString)
    s = replace(s, r"<\?.*?\?>"s => "", r"<!DOCTYPE[^>]*>" => "")
    root = XNode("#document", Dict{String,String}(), XNode[], "")
    stack = [root]
    token = r"\G(?:<(/?)([A-Za-z_][\w.-]*)((?:\s+[\w.:-]+=\"[^\"<]*\")*)\s*(/?)>|([^<]+))"
    pos = 1
    while pos <= ncodeunits(s)
        m = match(token, s, pos)
        m === nothing && error("parse_xml: unrecognized markup at byte $pos")
        pos += ncodeunits(m.match)
        if m.captures[5] !== nothing
            txt = strip(m.captures[5])
            isempty(txt) || (stack[end].text *= txt)
        elseif m.captures[1] == "/"
            stack[end].name == m.captures[2] ||
                error("parse_xml: </$(m.captures[2])> closes <$(stack[end].name)>")
            pop!(stack)
        else
            attrs = Dict(String(a.captures[1]) => String(a.captures[2])
                         for a in eachmatch(r"([\w.:-]+)=\"([^\"]*)\"", m.captures[3]))
            node = XNode(m.captures[2], attrs, XNode[], "")
            push!(stack[end].children, node)
            m.captures[4] == "/" || push!(stack, node)
        end
    end
    length(stack) == 1 || error("parse_xml: <$(stack[end].name)> is not closed")
    length(root.children) == 1 || error("parse_xml: more than one root element")
    return root.children[1]
end

xml_equal(a::XNode, b::XNode) =
    a.name == b.name && a.attrs == b.attrs && a.text == b.text &&
    length(a.children) == length(b.children) &&
    all(xml_equal(x, y) for (x, y) in zip(a.children, b.children))

function xml_find(node::XNode, name::AbstractString, out=XNode[])
    node.name == name && push!(out, node)
    foreach(c -> xml_find(c, name, out), node.children)
    return out
end

# The uniform grids of a temporal collection, after checking its structure.
function collection_grids(path)
    doc = parse_xml(read(path, String))
    @test doc.name == "Xdmf"
    domain = only(doc.children)
    series = only(domain.children)
    @test series.attrs["GridType"] == "Collection"
    @test series.attrs["CollectionType"] == "Temporal"
    return series.children
end

# Every DataItem of `grid` names an HDF5 file relative to `dir` and a dataset in
# it whose shape is the declared Dimensions, which XDMF lists row-major.
function datasets_match(grid::XNode, dir)
    ok = true
    for item in xml_find(grid, "DataItem")
        file, path = split(item.text, ":")
        ok &= !isabspath(file) && isfile(joinpath(dir, file))
        ok || return false
        dims = reverse(parse.(Int, split(item.attrs["Dimensions"])))
        h5open(joinpath(dir, file), "r") do h
            ok &= haskey(h, path) && collect(size(h[path])) == dims
        end
    end
    return ok
end

grid_time(grid::XNode) = parse(Float64, only(xml_find(grid, "Time")).attrs["Value"])

@testset "HDF5 extension: FieldWriter time series and XDMF collection" begin
    comm = MPI.COMM_WORLD
    np = MPI.Comm_size(comm)
    rank = MPI.Comm_rank(comm)
    per3h = ntuple(_ -> (PeriodicBC(), PeriodicBC()), 3)
    eos = IdealMixture([IdealSpecies{Float64}("a", 1.0, 1.4),
                        IdealSpecies{Float64}("b", 2.0, 1.6)])
    mk() = begin
        s = Solver(bcs=per3h, n_global=(72, 16, 12), L_domain=(1.0, 1.0, 1.0),
                   eos=eos, art=ArtParams(enabled=false), dims=(np, 1, 1))
        Q = allocate_state(s)
        initialize!(s, Q, (x, y, z) -> Prim(Y=(0.3 + 0.2sin(2π * x), 0.7 - 0.2sin(2π * x)),
                                            u=(0.2, 0, 0), p=1.0, rho=1.0))
        s, Q
    end

    dir = rank == 0 ? mktempdir() : ""
    dir = MPI.bcast(dir, comm; root=0)
    out = joinpath(dir, "series")
    prefix = joinpath(out, "field")

    # An EveryTime schedule stopped between instants, as a run ended by the
    # scheduler is, and a second writer on an AtTime list in the same run.
    s, Q = mk()
    writer = FieldWriter(prefix; format=:hdf5, fields=(:rho, :velocity, :Y))
    listed = FieldWriter(joinpath(out, "listed"); format=:hdf5, fields=(:p,),
                         stride=(2, 2, 1))
    run!(s, Q; tfinal=0.025,
         callback=(Callback(EveryTime(0.01), writer),
                   Callback(AtTime([0.005, 0.015]), listed)))
    @test writer.index == 3
    @test writer.times ≈ [0.0, 0.01, 0.02] rtol = 1e-14
    @test listed.times ≈ [0.005, 0.015] rtol = 1e-14
    save_checkpoint_hdf5(s, Q, joinpath(dir, "ckpt"))
    MPI.Barrier(comm)
    if rank == 0
        grids = collection_grids(prefix * ".xmf")
        @test length(grids) == 3
        @test grid_time.(grids) ≈ [0.0, 0.01, 0.02] rtol = 1e-14
        @test all(datasets_match(g, out) for g in grids)
        # The collection repeats each frame's own sidecar grid, so a frame read
        # on its own and read through the collection are the same description:
        # the species expansion and the vector component count included.
        for (m, g) in enumerate(grids)
            frame = parse_xml(read(CL.frame_prefix(writer, m - 1) * ".xmf", String))
            @test xml_equal(g, only(xml_find(frame, "Grid")))
        end
        names = [a.attrs["Name"] for a in xml_find(grids[1], "Attribute")]
        @test names == ["rho", "velocity", "Y1", "Y2"]
        vel = only(filter(a -> a.attrs["Name"] == "velocity",
                          xml_find(grids[1], "Attribute")))
        @test vel.attrs["AttributeType"] == "Vector"
        @test only(xml_find(vel, "DataItem")).attrs["Dimensions"] == "12 16 72 3"
        h5open(CL.frame_prefix(writer, 2) * ".h5", "r") do h
            @test read(h["meta/t"]) ≈ 0.02 rtol = 1e-14
            @test size(h["fields/Y2"]) == (72, 16, 12)
        end
        # Relative paths only, so the directory can be moved.
        @test !occursin(dir, read(prefix * ".xmf", String))

        lgrids = collection_grids(joinpath(out, "listed.xmf"))
        @test grid_time.(lgrids) ≈ [0.005, 0.015] rtol = 1e-14
        @test all(datasets_match(g, out) for g in lgrids)
        @test only(xml_find(lgrids[1], "Topology")).attrs["Dimensions"] == "12 8 36"
    end
    MPI.Barrier(comm)

    # An interruption during a collection rewrite leaves a partial temporary
    # file beside the complete collection of the previous frame. The next frame
    # replaces both.
    if rank == 0
        write(prefix * ".xmf.tmp", "<?xml version=\"1.0\" ?>\n<Xdmf Version=\"3.0\">\n <Dom")
        @test_throws ErrorException parse_xml(read(prefix * ".xmf.tmp", String))
        @test length(collection_grids(prefix * ".xmf")) == 3
    end
    MPI.Barrier(comm)

    # Restart from the checkpoint and continue the frame sequence. The frame
    # counter is not in the checkpoint, so it is supplied as start_index; the
    # restarted collection lists the frames this writer wrote, as the .pvd of a
    # restarted VTK writer does, and the earlier frames stay readable on their own.
    s2, Q2 = mk()
    load_checkpoint_hdf5!(s2, Q2, joinpath(dir, "ckpt"))
    @test s2.t ≈ 0.025 rtol = 1e-14
    resumed = FieldWriter(prefix; format=:hdf5, fields=(:rho, :velocity, :Y),
                          start_index=writer.index)
    run!(s2, Q2; tfinal=0.04, callback=Callback(EveryTime(0.01), resumed))
    @test resumed.index == 5
    @test resumed.times ≈ [0.03, 0.04] rtol = 1e-14
    MPI.Barrier(comm)
    if rank == 0
        @test !isfile(prefix * ".xmf.tmp")
        grids = collection_grids(prefix * ".xmf")
        @test grid_time.(grids) ≈ [0.03, 0.04] rtol = 1e-14
        @test all(datasets_match(g, out) for g in grids)
        refs = unique(String(first(split(item.text, ":")))
                      for g in grids for item in xml_find(g, "DataItem"))
        @test refs == ["field_0003.h5", "field_0004.h5"]
        for m in 0:2
            frame = parse_xml(read(CL.frame_prefix(writer, m) * ".xmf", String))
            @test datasets_match(only(xml_find(frame, "Grid")), out)
        end
    end
    MPI.Barrier(comm)

    # Stride and slice as the VTK writer takes them. The slice crosses the
    # split dimension, so at np > 1 most ranks hold no part of the plane and
    # write an empty selection.
    gx = 72 ÷ 2 + 1
    sliced = FieldWriter(joinpath(out, "sliced"); format=:hdf5, fields=(:rho, :velocity),
                         slice=(1, gx))
    sliced(s2, Q2)
    MPI.Barrier(comm)
    if rank == 0
        grids = collection_grids(joinpath(out, "sliced.xmf"))
        @test only(xml_find(grids[1], "Topology")).attrs["Dimensions"] == "12 16 1"
        @test all(datasets_match(g, out) for g in grids)
        h5open(joinpath(out, "sliced_0000.h5"), "r") do h
            @test size(h["fields/velocity"]) == (3, 1, 16, 12)
            @test read(h["grid/x"]) ≈ [global_xcoord(s2, 1, gx)]
            # Every point of the plane was written: a skipped block reads as
            # the dataset's fill value, zero, and the density is near one.
            @test all(>(0.5), read(h["fields/rho"]))
        end
    end
    MPI.Barrier(comm)

    # Rank 0 owns the collection: a writer holding `collection = true` on every
    # rank but rank 0 writes none, and one holding it on rank 0 alone writes it.
    others = FieldWriter(joinpath(out, "others"); format=:hdf5, fields=(:rho,),
                         collection=rank != 0)
    others(s2, Q2)
    root = FieldWriter(joinpath(out, "root"); format=:hdf5, fields=(:rho,),
                       collection=rank == 0)
    root(s2, Q2)
    MPI.Barrier(comm)
    if rank == 0
        @test !isfile(joinpath(out, "others.xmf"))
        @test isfile(joinpath(out, "others_0000.xmf"))
        @test length(collection_grids(joinpath(out, "root.xmf"))) == 1
    end
    MPI.Barrier(comm)

    # A refined solver's state vector has no shared-file form.
    wall = (SlipWallBC(), SlipWallBC())
    amr = Solver(n_global=(201, 1, 1), L_domain=(1.0, 1.0, 1.0),
                 bcs=(wall, per3h[2], per3h[3]), tile=8, dims=(np, 1, 1),
                 refine=BlockRegion((85, 0, 0), (31, 1, 1)))
    states = allocate_state(amr)
    initialize!(amr, states, (x, y, z) -> Prim(u=(0, 0, 0), p=1.0, rho=1.0))
    @test_throws "patch layout" FieldWriter(joinpath(out, "amr"); format=:hdf5)(amr, states)
    @test_throws ArgumentError FieldWriter(prefix; format=:netcdf)

    MPI.Barrier(comm)
    rank == 0 && rm(dir; recursive=true)
    MPI.Barrier(comm)
end
