# What the I/O paths carry beyond the state array: the checkpoint's element type
# and mutable run state, the boundary-switch record, the frame numbering of a
# restarted FieldWriter, and the byte order the VTK headers declare. Included by
# runtests.jl, and runnable on its own:
#
#   julia --project=. test/io_tests.jl
#
# Serial. The decomposition-dependent half of the per-rank checkpoint is covered
# by test/mpi_tests.jl and the shared-file path by test/hdf5_tests.jl, both of
# which check the same header fields through their own readers.

if !@isdefined(CL)
    using MPI
    MPI.Init(threadlevel=:funneled)
    using CompactLES
    using Test
    const CL = CompactLES
end

const io_per = (PeriodicBC(), PeriodicBC())

# Dimension 1 is closed to hold a SwitchableBC; the other two are
# periodic, which keeps the grid small enough to build several of these.
io_solver(lo, hi; kw...) =
    Solver(n_global=(12, 12, 12), L_domain=(1.0, 1.0, 1.0),
           bcs=((lo, hi), io_per, io_per), art=ArtParams(enabled=false), kw...)

io_state(solver) = begin
    Q = allocate_state(solver)
    initialize!(solver, Q, (x, y, z) -> Prim(u=(sin(2π * y), 0, 0),
                                             p=1 + 0.1cos(2π * z), rho=1 + 0.2x))
    Q
end

@testset "checkpoint carries the mutable run state" begin
    dir = mktempdir()
    # Under a directory that does not exist yet, which is the writer's job to
    # create and was previously only FieldWriter's.
    stem = joinpath(dir, "frames", "state")

    written = SwitchableBC(SlipWallBC(), ExtrapolationBC())
    s = io_solver(written, SlipWallBC())
    Q = io_state(s)
    s.t = 0.37
    s.step = 42
    s.cfl = 0.125            # what a StepControl retry leaves behind
    s.dt_prev = 1.5e-4
    s.rate_prev = 987.5
    switch!(written)
    save_checkpoint(s, Q, stem)
    @test isfile(stem * ".r0000.ckpt")

    read_back = SwitchableBC(SlipWallBC(), ExtrapolationBC())
    s2 = io_solver(read_back, SlipWallBC())
    Q2 = allocate_state(s2)
    load_checkpoint!(s2, Q2, stem)
    @test s2.t == 0.37
    @test s2.step == 42
    @test s2.cfl == 0.125
    @test s2.dt_prev == 1.5e-4
    @test s2.rate_prev == 987.5
    # Resuming with this cleared would run the remainder of the calculation
    # under the pre-switch boundary condition, silently.
    @test switched(read_back)
    @test all(Q2[gidx(s2, i, j, k), c] == Q[gidx(s, i, j, k), c]
              for c in 1:5, i in 1:12, j in 1:12, k in 1:12)

    # A face the file describes as switchable and this solver does not, and the
    # reverse: both change what the boundary does for the rest of the run.
    @test_throws "boundary mismatch" load_checkpoint!(
        io_solver(SlipWallBC(), SlipWallBC()),
        allocate_state(s2), stem)

    plain_stem = joinpath(dir, "plain")
    sp = io_solver(SlipWallBC(), SlipWallBC())
    save_checkpoint(sp, io_state(sp), plain_stem)
    @test_throws "boundary mismatch" load_checkpoint!(
        io_solver(SwitchableBC(SlipWallBC(), ExtrapolationBC()), SlipWallBC()),
        allocate_state(sp), plain_stem)

    # switch! is one-way, so a checkpoint written before a face switched cannot
    # be restored onto a solver whose face already has.
    ahead = SwitchableBC(SlipWallBC(), ExtrapolationBC())
    sa = io_solver(ahead, SlipWallBC())
    save_checkpoint(sa, io_state(sa), joinpath(dir, "before"))
    switch!(ahead)
    @test_throws "already switched" load_checkpoint!(sa, allocate_state(sa),
                                                     joinpath(dir, "before"))

    rm(dir; recursive=true)
end

@testset "restart continues the run bit for bit" begin
    # A Sod tube with the artificial properties on: the coefficients the last
    # right-hand side left on the patch size the first step after the
    # restart, so a checkpoint without them would take a different one.
    wall = (SlipWallBC(), SlipWallBC())
    mk(; kw...) = Solver(n_global=(96, 1, 1), L_domain=(1.0, 1.0, 1.0),
                         bcs=(wall, io_per, io_per), cfl=0.4; kw...)
    ic(x, y, z) = x < 0.5 ? Prim(u=(0, 0, 0), p=1.0, rho=1.0) :
                            Prim(u=(0, 0, 0), p=0.1, rho=0.125)
    dir = mktempdir()
    stem = joinpath(dir, "sod")
    s = mk()
    Q = allocate_state(s)
    initialize!(s, Q, ic)
    run!(s, Q; tfinal=1.0, nmax=20)
    save_checkpoint(s, Q, stem)
    @test maximum(s.beta_art) > 0          # the record is not trivially zero
    run!(s, Q; tfinal=1.0, nmax=40)

    r = mk()
    Qr = allocate_state(r)
    load_checkpoint!(r, Qr, stem)
    @test r.step == 20
    run!(r, Qr; tfinal=1.0, nmax=40)
    @test r.step == 40
    @test r.t == s.t
    @test r.dt_prev == s.dt_prev
    inner = CL.interior(s.decomp)
    @test parent(Qr)[inner, :] == parent(Q)[inner, :]
    @test r.mu_art[inner] == s.mu_art[inner]
    @test r.beta_art[inner] == s.beta_art[inner]

    # The primitives are current on return, so a callback may read them.
    r2 = mk()
    Q2 = allocate_state(r2)
    load_checkpoint!(r2, Q2, stem)
    @test all(r2.rho[I] == mixture_density(r2, Q2, I) for I in inner)

    # A solver that does not compute the coefficients is refused: it would
    # ignore a record the writer's next step depended on, and the reverse
    # would start from zeros the writer did not have.
    off = mk(art=ArtParams(enabled=false))
    @test_throws "artificial-property mismatch" load_checkpoint!(
        off, allocate_state(off), stem)
    save_checkpoint(off, allocate_state(off), joinpath(dir, "off"))
    @test_throws "artificial-property mismatch" load_checkpoint!(
        mk(), allocate_state(s), joinpath(dir, "off"))
    rm(dir; recursive=true)
end

@testset "hierarchy checkpoint: layout, tag history and bitwise continuation" begin
    # A regridded Sod run checkpointed between checks and continued from a
    # solver built with the initial region: the restart rebuilds the recorded
    # tile set, restores the tag history, and the continued states agree bit
    # for bit with the uninterrupted run, the regrids after the restart
    # included.
    wall = (SlipWallBC(), SlipWallBC())
    ic(x, y, z) = x < 0.5 ? Prim(u=(0, 0, 0), p=1.0, rho=1.0) :
                            Prim(u=(0, 0, 0), p=0.1, rho=0.125)
    initial = BlockRegion((85, 0, 0), (31, 1, 1))
    mk(; kw...) = Solver(n_global=(201, 1, 1), L_domain=(1.0, 1.0, 1.0),
                         bcs=(wall, io_per, io_per), cfl=0.2, subcycle=true,
                         regrid_interval=5, refine=initial; kw...)
    same(a, b, decomp) = (inner = CL.interior(decomp);
                          parent(a)[inner, :] == parent(b)[inner, :])
    dir = mktempdir()
    for (label, tile) in (("tiled", 8), ("box", 0))
        stem = joinpath(dir, label)
        s = mk(tile=tile)
        states = allocate_state(s)
        initialize!(s, states, ic)
        run!(s, states; tfinal=1.0, nmax=23)
        spec = getfield(s, :regrid)
        regs = level_regions(s, 1)
        created = copy(spec.created)
        checks, last_step, t_save = spec.checks, spec.last_step, s.t
        @test regs != level_regions(mk(tile=tile), 1)   # the layout has moved
        save_checkpoint(s, states, stem)
        run!(s, states; tfinal=1.0, nmax=130)
        @test level_regions(s, 1) != regs                 # ...and moves again

        r = mk(tile=tile)
        sr = allocate_state(r)
        load_checkpoint!(r, sr, stem)
        rspec = getfield(r, :regrid)
        @test r.step == 23 && r.t == t_save
        @test level_regions(r, 1) == regs
        @test length(sr) == length(r.patches) == length(regs) + 1
        @test rspec.created == created
        @test rspec.checks == checks && rspec.last_step == last_step
        run!(r, sr; tfinal=1.0, nmax=130)
        @test r.t == s.t && r.step == s.step == 130
        @test level_regions(r, 1) == level_regions(s, 1)
        @test getfield(r, :regrid).created == spec.created
        @test all(same(sr[i], states[i], s.patches[i].decomp)
                  for i in eachindex(states))
        @test all(r.patches[i].beta_art == s.patches[i].beta_art
                  for i in eachindex(states))
    end

    # A file of one hierarchy refused by a solver of another: unrefined, a
    # different lattice, and a static tiled level with no RegridSpec to
    # rebuild through.
    stem = joinpath(dir, "tiled")
    plain = Solver(n_global=(201, 1, 1), L_domain=(1.0, 1.0, 1.0),
                   bcs=(wall, io_per, io_per))
    @test_throws "refinement mismatch" load_checkpoint!(plain, allocate_state(plain),
                                                        stem)
    other = mk(tile=4)
    @test_throws "tile mismatch" load_checkpoint!(other, allocate_state(other), stem)
    static = Solver(n_global=(201, 1, 1), L_domain=(1.0, 1.0, 1.0),
                    bcs=(wall, io_per, io_per), refine=initial, tile=8)
    @test_throws "cannot rebuild" load_checkpoint!(static, allocate_state(static),
                                                   stem)

    # A static three-level hierarchy round-trips onto the layout it was
    # built with and continues bit for bit; another level-2 region is refused.
    per3 = ntuple(_ -> io_per, 3)
    r1 = BlockRegion((40, 0, 0), (16, 1, 1))
    e1 = 3 * 16 - 2
    r2 = BlockRegion((3 * 40 + e1 ÷ 4, 0, 0), (e1 ÷ 2, 1, 1))
    mk3(regions) = Solver(n_global=(96, 1, 1), L_domain=(2π, 1.0, 1.0), bcs=per3,
                          refine=regions, art=ArtParams(enabled=false))
    wave(x, y, z) = Prim(u=(0.5, 0, 0), p=1.0, rho=1 + 0.2sin(x))
    s3 = mk3([r1, r2])
    st3 = allocate_state(s3)
    initialize!(s3, st3, wave)
    run!(s3, st3; tfinal=1.0, nmax=5)
    save_checkpoint(s3, st3, joinpath(dir, "three"))
    run!(s3, st3; tfinal=1.0, nmax=10)
    q3 = mk3([r1, r2])
    sq3 = allocate_state(q3)
    load_checkpoint!(q3, sq3, joinpath(dir, "three"))
    @test nlevels(q3) == 3 && q3.step == 5
    run!(q3, sq3; tfinal=1.0, nmax=10)
    @test q3.t == s3.t
    @test all(same(sq3[i], st3[i], s3.patches[i].decomp) for i in eachindex(st3))
    wrong = mk3([r1, BlockRegion((3 * 40 + e1 ÷ 4 + 2, 0, 0), (e1 ÷ 2 - 4, 1, 1))])
    @test_throws "level 2 layout mismatch" load_checkpoint!(
        wrong, allocate_state(wrong), joinpath(dir, "three"))
    rm(dir; recursive=true)
end

@testset "checkpoint round trip in Float32" begin
    T = Float32
    dir = mktempdir()
    stem = joinpath(dir, "f32")
    mk32() = Solver(n_global=(12, 12, 12), L_domain=(one(T), one(T), one(T)),
                    bcs=ntuple(_ -> io_per, 3), transport=Transport{T}(),
                    art=ArtParams{T}(enabled=false), deriv=lele_d1_6(T),
                    filt=compact_filter(T(0.45), T))
    s = mk32()
    Q = io_state(s)
    @test parent(Q) isa Array{T,4}
    s.t = T(0.25)
    s.step = 5
    save_checkpoint(s, Q, stem)

    s2 = mk32()
    Q2 = allocate_state(s2)
    load_checkpoint!(s2, Q2, stem)
    @test s2.step == 5
    @test all(Q2[gidx(s2, i, j, k), c] == Q[gidx(s, i, j, k), c]
              for c in 1:5, i in 1:12, j in 1:12, k in 1:12)

    # The payload is raw binary, so reading a Float32 block at Float64 does not
    # produce wrong numbers, it runs off the end of the file. The element type
    # is stored to make that a message instead.
    s64 = Solver(n_global=(12, 12, 12), L_domain=(1.0, 1.0, 1.0),
                 bcs=ntuple(_ -> io_per, 3), art=ArtParams(enabled=false))
    @test_throws "element type mismatch" load_checkpoint!(s64,
                                                          allocate_state(s64),
                                                          stem)
    rm(dir; recursive=true)
end

@testset "FieldWriter start_index continues a frame sequence" begin
    dir = mktempdir()
    s = io_solver(SlipWallBC(), SlipWallBC())
    Q = io_state(s)
    # Also a prefix whose directory does not exist, which the writer creates.
    writer = FieldWriter(joinpath(dir, "out", "field"); fields=(:rho,),
                         start_index=12)
    writer(s, Q)
    s.t = 0.1
    writer(s, Q)
    @test isfile(joinpath(dir, "out", "field_0012.pvtr"))
    @test isfile(joinpath(dir, "out", "field_0013.pvtr"))
    @test !isfile(joinpath(dir, "out", "field_0000.pvtr"))

    pvd = read(joinpath(dir, "out", "field.pvd"), String)
    @test occursin("field_0012.pvtr", pvd)
    @test occursin("field_0013.pvtr", pvd)
    @test !occursin("field_0000", pvd)

    # The default writer is unchanged by the keyword.
    plain = FieldWriter(joinpath(dir, "plain"); fields=(:rho,))
    plain(s, Q)
    @test isfile(joinpath(dir, "plain_0000.pvtr"))
    rm(dir; recursive=true)
end

@testset "multiblock VTK: one piece per patch, covered nodes blanked" begin
    # Four 37-node tiles of a 2-D level under a 48 x 48 root: five pieces
    # under one index, the root carrying the ghost array that blanks the
    # nodes the tiles cover entirely and the tiles carrying none.
    solver = Solver(n_global=(48, 48, 1), L_domain=(2π, 2π, 1.0),
                    bcs=(io_per, io_per, io_per), tile=12,
                    refine=BlockRegion((12, 12, 0), (24, 24, 1)),
                    art=ArtParams(enabled=false))
    states = allocate_state(solver)
    initialize!(solver, states, (x, y, z) ->
        Prim(u=(0.4, 0.3, 0), p=1.0, rho=1 + 0.1 * sin(x) * sin(y)))
    dir = mktempdir()
    save_vtk(solver, states, joinpath(dir, "amr"); fields=(:rho, :velocity))
    vtm = read(joinpath(dir, "amr.vtm"), String)
    @test count("<DataSet", vtm) == 5
    @test count("<Block", vtm) == 2
    @test length(filter(endswith(".vtr"), readdir(dir))) == 5
    root_piece = read(joinpath(dir, "amr.L0.T0001.r0000.vtr"), String)
    @test occursin("Name=\"vtkGhostType\"", root_piece)
    @test occursin("Name=\"rho\"", root_piece) && occursin("Name=\"velocity\"", root_piece)
    tile_piece = read(joinpath(dir, "amr.L1.T0001.r0000.vtr"), String)
    @test !occursin("vtkGhostType", tile_piece)
    @test occursin("WholeExtent=\"0 36 0 36 0 0\"", tile_piece)
    # The ghost array hides exactly the fully covered root nodes: the
    # interior of the 24 x 24 region, 23 x 23 nodes, and not its faces.
    root = PatchSolver(solver, solver.patches[1])
    ranges = CL._output_ranges(root, (1, 1, 1), nothing)
    ghost = CL._vtk_ghost_points(root, ranges)
    @test count(==(CL.VTK_HIDDENPOINT), ghost) == 23 * 23
    @test count(==(0xff), root.covered) == 23 * 23
    @test container_extension(solver) == ".vtm"

    # A FieldWriter on the state vector numbers .vtm frames into its .pvd.
    writer = FieldWriter(joinpath(dir, "series"); fields=(:rho,))
    writer(solver, states)
    @test isfile(joinpath(dir, "series_0000.vtm"))
    @test occursin("series_0000.vtm", read(joinpath(dir, "series.pvd"), String))

    # A slice names a root plane: through the refined region it reaches the
    # tiles the plane meets (root node 20 is inside the first tile column
    # only), outside it the root alone, and beyond the domain it is refused.
    save_vtk(solver, states, joinpath(dir, "cut"); fields=(:rho,), slice=(1, 20))
    @test count("<DataSet", read(joinpath(dir, "cut.vtm"), String)) == 3
    save_vtk(solver, states, joinpath(dir, "edge"); fields=(:rho,), slice=(1, 5))
    @test count("<DataSet", read(joinpath(dir, "edge.vtm"), String)) == 1
    @test_throws ArgumentError save_vtk(solver, states, joinpath(dir, "bad");
                                        fields=(:rho,), slice=(1, 49))
    # A stride acts in each patch's own node space.
    save_vtk(solver, states, joinpath(dir, "coarse"); fields=(:rho,), stride=2)
    @test occursin("WholeExtent=\"0 18 0 18 0 0\"",
                   read(joinpath(dir, "coarse.L1.T0001.r0000.vtr"), String))
    rm(dir; recursive=true)
end

@testset "VTK headers declare the byte order actually written" begin
    # The appended blocks go out in native order, so the declaration follows
    # ENDIAN_BOM rather than asserting little-endian.
    @test CL.VTK_BYTE_ORDER ==
          (Base.ENDIAN_BOM == 0x04030201 ? "LittleEndian" : "BigEndian")
    dir = mktempdir()
    s = io_solver(SlipWallBC(), SlipWallBC())
    Q = io_state(s)
    save_vtk(s, Q, joinpath(dir, "dump"); fields=(:rho,))
    for name in ("dump.r0000.vtr", "dump.pvtr")
        @test occursin("byte_order=\"$(CL.VTK_BYTE_ORDER)\"",
                       read(joinpath(dir, name), String))
    end
    writer = FieldWriter(joinpath(dir, "series"); fields=(:rho,))
    writer(s, Q)
    @test occursin("byte_order=\"$(CL.VTK_BYTE_ORDER)\"",
                   read(joinpath(dir, "series.pvd"), String))
    rm(dir; recursive=true)
end

@testset "halo exchange refuses a foreign element type" begin
    # The staging buffers are Vector{T} for a Decomp{T}. A field of another type
    # used to be converted into and back out of them element by element, losing
    # precision on the way out and costing a pass over the slab each way.
    s = io_solver(SlipWallBC(), SlipWallBC())
    decomp = s.decomp
    wrong = zeros(Float32, size(CL.field(decomp)))
    @test_throws MethodError CL.exchange_halos!(wrong, decomp)
    @test_throws MethodError CL.exchange_dim!(wrong, decomp, 1)
    @test_throws "Vector{Float64}" CL.exchange_dim_batch!([wrong], decomp, 1)
    # ...and the decomposition's own type still goes through.
    @test CL.exchange_halos!(CL.field(decomp), decomp) isa Array{Float64,3}
end
