# Makie extension tests. Included by runtests.jl when a Makie backend is
# loadable, and runnable under mpiexec for the decomposition-independent
# profile `line_profile` produces.
#
#   julia --project=docs test/makie_tests.jl               # serial
#   mpiexec -n 4 julia --project=docs test/makie_tests.jl  # decomposed
#
# A Makie backend is a weak dependency, so it is not loadable from the package
# environment alone. Run these from an environment that has both — the docs
# environment carries CairoMakie.

if !@isdefined(CL)
    using MPI
    MPI.Init(threadlevel=:funneled)
    using CompactLES
    using Test
    const CL = CompactLES
end
using CompactLES: padded_index, makie_available
using CairoMakie

# The old `density_line`: rank-local sampling of mixture_density along dim 1 at
# (i, 1, 1). Kept here as the reference the new API must reproduce.
function density_line_local(solver, Q)
    n = solver.decomp.n_local[1]
    [mixture_density(solver, Q, padded_index(solver, i, 1, 1)) for i in 1:n]
end

@testset "Makie extension: extraction API" begin
    comm = MPI.COMM_WORLD
    np = MPI.Comm_size(comm)
    rank = MPI.Comm_rank(comm)

    @test makie_available()

    # A 1-D acoustic-style setup: density varies along x, other dims collapsed.
    gamma = 1.4
    prob = Problem(name="viz", eos=IdealSpecies("gas"; R=1.0, gamma=gamma),
                   domain=((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)),
                   bcs=ntuple(_ -> (PeriodicBC(), PeriodicBC()), 3),
                   ic=(x, y, z) -> begin
                       p = 1 + 0.1 * exp(-((x - 0.4) / 0.05)^2)
                       Prim(p=p, rho=p^(1 / gamma))
                   end)
    num = Numerics(n_global=(96, 1, 1), art=ArtificialProperties(enabled=false),
                   filter=nothing, execution=Execution(dims=(np, 1, 1)))
    solver, Q = setup(prob, num)

    # line_profile gathers globally. Its value equals the concatenation of the
    # per-rank local samples in rank order, so on one rank it is that rank's
    # samples and on many it is the whole line, which no rank-local sampler
    # produces.
    coord, value = line_profile(solver, Q, :rho)
    @test length(coord) == 96
    @test length(value) == 96
    @test issorted(coord)

    local_rho = density_line_local(solver, Q)
    counts = MPI.Allgather(Cint(length(local_rho)), comm)
    gathered = Vector{Float64}(undef, sum(counts))
    MPI.Allgatherv!(local_rho, MPI.VBuffer(gathered, counts), comm)
    # Transverse dims are collapsed, so the plane average is the point value:
    # the gathered per-rank line equals the global profile bit for bit.
    @test value == gathered

    # The profile is identical on every rank (a collective reduction, not a
    # rank-local read). Compare against rank 0's copy.
    ref = MPI.bcast(value, comm; root=0)
    @test value == ref
end

@testset "Makie extension: slices and revolution" begin
    comm = MPI.COMM_WORLD
    np = MPI.Comm_size(comm)

    # A 2-D Cartesian field, decomposed along x when ranks allow it. 48 along x
    # keeps ≥ 9 points per rank up to np = 4, which the C8 filter closure needs.
    dims = np == 1 ? (1, 1, 1) : (np, 1, 1)
    prob = Problem(name="slice", eos=IdealSpecies("gas"; R=1.0, gamma=1.4),
                   domain=((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)),
                   bcs=ntuple(_ -> (PeriodicBC(), PeriodicBC()), 3),
                   ic=(x, y, z) -> Prim(p=1.0,
                                        rho=1.0 + 0.5sin(2pi * x) * cos(2pi * y)))
    num = Numerics(n_global=(48, 16, 1), art=ArtificialProperties(enabled=false),
                   filter=nothing, execution=Execution(dims=dims))
    solver, Q = setup(prob, num)

    slice = field_slice(solver, Q, :rho; normal=3, index=1)
    if MPI.Comm_rank(comm) == 0
        x1, x2, vals = slice
        @test size(vals) == (48, 16)
        @test length(x1) == 48 && length(x2) == 16
        # The analytic field is recovered on the gathered global plane.
        @test isapprox(vals[6, 4], 1.0 + 0.5sin(2pi * x1[6]) * cos(2pi * x2[4]);
                       atol=1e-12)
        X, Y, grid = cartesian_slice(solver, (1, 2), x1, x2, vals; n=32)
        @test size(grid) == (32, 32)
        @test all(isfinite, grid)           # Cartesian slice fills the raster
    else
        @test slice === nothing             # a slice is a rank-0 gather
    end

    # revolve_profile is the collapsed-radial view: a monotone bump becomes a
    # disk whose finite fraction is the inscribed area π/4.
    r = collect(range(0.02, 1.0; length=40))
    v = exp.(-((r .- 0.5) ./ 0.1) .^ 2)
    axis, disk = revolve_profile(r, v; n=120)
    @test size(disk) == (120, 120)
    @test isapprox(count(isfinite, disk) / length(disk), pi / 4; atol=0.03)
end

@testset "Makie extension: refined state vector" begin
    # The composite forms: `profileplot` and `fieldheatmap` take the state
    # vector of a refined run and draw the root-grid profile and plane.
    prob = Problem(name="refined", eos=IdealSpecies("gas"; R=1.0, gamma=1.4),
                   domain=((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)),
                   bcs=ntuple(_ -> (PeriodicBC(), PeriodicBC()), 3),
                   ic=(x, y, z) -> Prim(p=1.0, rho=1.0 + 0.5sin(2pi * x) * cos(2pi * y)))
    num = Numerics(n_global=(36, 24, 1), art=ArtificialProperties(enabled=false),
                   filter=nothing, amr=AMR(initial=BlockRegion((10, 6, 0), (10, 8, 1))))
    solver, states = setup(prob, num)
    @test states isa Vector
    fig, ax, plt = profileplot(solver, states, :rho)
    @test plt isa Makie.Lines
    slice = field_slice(solver, states, :rho)
    heat = fieldheatmap(solver, states, :rho)
    # The plane y = x2[10] crosses the refined region (root nodes 7..14 in y);
    # y = x2[3] misses it. Each patch's plane is at its own spacing.
    exact(x, y) = 1.0 + 0.5sin(2pi * x) * cos(2pi * y)
    crossing = field_snapshot(solver, states; fields=(:rho,), normal=2, index=10)
    missing_plane = field_snapshot(solver, states; fields=(:rho,), normal=2, index=3)
    @test_throws ArgumentError field_snapshot(solver, states; normal=2, index=25)
    if MPI.Comm_rank(MPI.COMM_WORLD) == 0
        x1, x2, vals = slice
        @test size(vals) == (36, 24)
        @test maximum(abs(vals[i, j] - exact(x1[i], x2[j]))
                      for i in 1:36, j in 1:24) < 1e-12
        # One heatmap per level, root first, sharing one color range; the root
        # leaves the nodes the refined patch covers transparent.
        @test heat[3] isa Vector && length(heat[3]) == 2
        @test all(p -> p isa Makie.Heatmap, heat[3])
        @test heat[3][1].colorrange[] == heat[3][2].colorrange[]
        @test count(isnan, heat[3][1][3][]) > 0
        @test !any(isnan, heat[3][2][3][])

        @test length(missing_plane) == 1 && size(only(missing_plane)) == (36, 1, 1)
        root, fine = crossing
        @test size(root) == (36, 1, 1) && root.offset == (0, 9, 0)
        @test size(fine) == (28, 1, 1) && fine.offset == (30, 27, 0)
        @test fine.coords[2][1] ≈ x2[10]
        @test maximum(abs(fine[:rho][i] - exact(fine.coords[1][i], fine.coords[2][1]))
                      for i in 1:28) < 1e-12

        # The mesh lines are the snapshots' own coordinates, bit for bit.
        whole = field_snapshot(solver, states; fields=(:rho,))
        blocks = CL._mesh_blocks(solver, 3, 1)
        @test [b.level for b in blocks] == [s.level for s in whole] == [0, 1]
        @test all(b.x1 == s.coords[1] && b.x2 == s.coords[2]
                  for (b, s) in zip(blocks, whole))
    else
        @test slice === nothing && heat === nothing
        @test crossing === nothing && missing_plane === nothing
        field_snapshot(solver, states; fields=(:rho,))
    end

    # A mesh plot reads the layout rank 0 holds and communicates nothing.
    if MPI.Comm_rank(MPI.COMM_WORLD) == 0
        fig, ax, plots = meshplot(solver)
        @test fig isa Makie.Figure
        @test length(plots) == 4 && all(p -> p isa Makie.Lines, plots)
        @test length(meshplot!(ax, solver; grid=false, outlines=1:1)) == 1
        @test_throws ArgumentError meshplot!(ax, solver; normal=1)
        # Composes with the refined heatmap in one axis.
        @test meshplot!(heat[2], solver; color=[:white, :cyan]) isa Vector
    else
        @test meshplot(solver) === nothing
    end

    # A level of several tiles is one heatmap over the box enclosing them,
    # transparent between tiles, with each tile node at its own value.
    tiled = Numerics(num; amr=AMR(initial=BlockRegion((4, 4, 0), (26, 14, 1)), tile=6))
    tsolver, tstates = setup(prob, tiled)
    theat = fieldheatmap(tsolver, tstates, :rho)
    # Makie converts a heatmap's coordinates, so the drawn blocks are read
    # from the extension before they reach it.
    ext = Base.get_extension(CompactLES, :CompactLESMakieExt)
    tdata = ext._heatmap_blocks(tsolver, tstates, :rho, 3, 1, 1)
    if MPI.Comm_rank(MPI.COMM_WORLD) == 0
        @test length(level_regions(tsolver, 1)) > 1
        @test length(theat[3]) == length(tdata.blocks) == 2
        x1f, x2f, gridf = tdata.blocks[2].x1, tdata.blocks[2].x2, tdata.blocks[2].values
        # The mesh shows the nodes the heatmap draws, level by level.
        @test [count(!, lev.shown) for lev in CL._mesh_levels(tsolver, 3, 1)] ==
              [count(isnan, blk.values) for blk in tdata.blocks]
        @test maximum(abs(gridf[i, j] - exact(x1f[i], x2f[j]))
                      for i in axes(gridf, 1), j in axes(gridf, 2)
                      if !isnan(gridf[i, j])) < 1e-12
        tile_nodes = Set((x, y) for b in CL._mesh_blocks(tsolver, 3, 1) if b.level == 1
                         for x in b.x1, y in b.x2)
        @test count(!isnan, gridf) == length(tile_nodes)
    else
        @test theat === nothing
    end
end

# Building a plot is a rank-0 / serial concern: `profileplot` returns a replicated
# figure and `fieldheatmap` gathers to rank 0. The curvilinear solver here also
# resolves θ, which does not decompose cleanly at small extents, so this runs
# only in serial.
if MPI.Comm_size(MPI.COMM_WORLD) == 1
    @testset "Makie extension: plot objects" begin
        prob = Problem(name="plot", eos=IdealSpecies("gas"; R=1.0, gamma=1.4),
                       metric=CylindricalMetric(),
                       domain=((0.0, 1.0), (0.0, 2pi), (0.0, 1.0)),
                       bcs=((AxisBC(), SlipWallBC()),
                            (PeriodicBC(), PeriodicBC()),
                            (PeriodicBC(), PeriodicBC())),
                       ic=(r, th, z) -> Prim(p=1.0, rho=1.0 + 0.2cos(th) * r))
        num = Numerics(n_global=(24, 16, 1), art=ArtificialProperties(enabled=false),
                       filter=nothing)
        solver, Q = setup(prob, num)

        fig, ax, plt = profileplot(solver, Q, :rho)
        @test fig isa Makie.Figure
        @test plt isa Makie.Lines

        fig2, ax2, plt2 = fieldheatmap(solver, Q, :rho; normal=3, index=1)
        @test fig2 isa Makie.Figure
        @test plt2 isa Makie.Heatmap

        # The disk is filled through the axis: the raster point nearest the
        # center takes the mean of the first nodes on either side of it.
        X, Y, disk = cartesian_slice(solver, (1, 2),
                                     field_slice(solver, Q, :rho; normal=3, index=1)...)
        r1 = CL.global_xcoord(solver, 1, 1)
        inner = [disk[i, j] for i in eachindex(X), j in eachindex(Y) if hypot(X[i], Y[j]) < r1]
        @test !isempty(inner) && all(isfinite, inner)
        @test isapprox(disk[argmin(abs.(X)), argmin(abs.(Y))], 1.0; atol=0.01)

        # The polar mesh closes around the angle and passes through the axis:
        # every radius is a full circle through the wrapped node, and the
        # outline is the outer circle alone.
        blk = only(CL._mesh_blocks(solver, 3, 1))
        @test blk.closed && blk.axis && length(blk.x2) == 17
        @test blk.x2[end] ≈ blk.x2[1] + 2pi
        plots = meshplot!(ax2, solver)
        @test length(plots) == 2
        outline = plots[2][1][]
        radii = [hypot(p...) for p in outline if !isnan(p[1])]
        @test all(r -> isapprox(r, blk.x1[end]; rtol=1e-5), radii)
        # The cells around the first nodes are wedges meeting at the origin, and
        # the cells of the wall nodes end on the wall.
        cells = plots[1][1][]
        @test minimum(hypot(p...) for p in cells if !isnan(p[1])) < 1e-12
        lev = only(CL._mesh_levels(solver, 3, 1))
        @test lev.e1[1] == 0 && lev.e1[end] == 1.0
        @test lev.e2[end] ≈ lev.e2[1] + 2pi && all(lev.shown)

        # A spherical meridian (normal = 3 → (r, θ) plane) resamples onto x–z with
        # the pole vertical, not x–y. Slicing the 3-D position by (a, b) collapsed
        # the plane onto Y = 0 and rastered to all-NaN; guard that the meridian
        # fills a physical half-disk (≈ π/4 of the bounding box).
        sph = Problem(name="meridian", eos=IdealSpecies("gas"; R=1.0, gamma=1.4),
                      metric=SphericalMetric(),
                      domain=((0.0, 1.0), (0.0, pi), (0.0, 2pi)),
                      bcs=((OriginBC(), SlipWallBC()),
                           (PoleBC(), PoleBC()),
                           (PeriodicBC(), PeriodicBC())),
                      ic=(r, th, ph) -> Prim(p=1.0, rho=1.0 + exp(-(r / 0.25)^2)))
        snum = Numerics(n_global=(24, 16, 12), art=ArtificialProperties(enabled=false),
                        filter=nothing)
        ssolver, sQ = setup(sph, snum)
        x1, x2, vals = field_slice(ssolver, sQ, :rho; normal=3, index=1)
        X, Y, grid = cartesian_slice(ssolver, (1, 2), x1, x2, vals; n=100)
        finite = count(isfinite, grid) / length(grid)
        @test finite > 0.5                       # not the degenerate all-NaN case
        @test isapprox(finite, pi / 4; atol=0.06)
    end
end

println("Makie extension tests complete")
