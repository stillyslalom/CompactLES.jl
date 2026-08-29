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
