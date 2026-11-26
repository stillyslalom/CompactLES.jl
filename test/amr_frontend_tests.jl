using Test
using CompactLES
using CompactLES: padded_index

const AMR_TEST_BCS = (PeriodicBC(), PeriodicBC(), PeriodicBC())

struct MovingAMRWindow
    center::Float64
end
(window::MovingAMRWindow)(x, y, z, t) = abs(x - (window.center + 0.1t)) < 0.04

function amr_test_problem(ic=(x, y, z, h) -> Prim(p=1.0, rho=1.0, u=(0.0, 0.0, 0.0)))
    return Problem(domain=((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)),
                   bcs=AMR_TEST_BCS, ic=ic)
end

@testset "AMR frontend chooses an initial region from physical coordinates" begin
    # The predicate is independent of root node numbering. The selected box
    # follows the physical window within a few coarse cells at both grids.
    for n in (48, 96)
        selector = MovingAMRWindow(0.65)
        solver, states = setup(amr_test_problem(),
            Numerics(n_global=(n, 1, 1), filter=nothing,
                     art=ArtificialProperties(enabled=false),
                     amr=AMR(initial=selector, tag_buffer=2, regrid_interval=0)))
        region = refined_region(solver)
        center = (region.offset[1] + region.extent[1] / 2) / n
        @test abs(center - 0.65) < 6 / n
        @test nlevels(solver) == 2
        @test getfield(solver, :regrid) === nothing
        @test 1 <= length(states.patches) <= 2  # fine patch exists only on its owner ranks
    end
end

@testset "AMR frontend evaluates IC with actual fine spacing" begin
    ic = (x, y, z, h) -> Prim(p=1.0, rho=1.0 + h, u=(0.0, 0.0, 0.0))
    solver, states = setup(amr_test_problem(ic),
        Numerics(n_global=(96, 1, 1), filter=nothing,
                 art=ArtificialProperties(enabled=false),
                 amr=AMR(initial=(x, y, z, t) -> abs(x - 0.7) < 0.04)))
    root = CompactLES.PatchSolver(solver, getfield(solver, :patches)[1])
    @test states.patches[1][padded_index(root, 1, 1, 1), 1] ≈ 1 + root.h[1]
    if length(states.patches) > 1
        fine = CompactLES.PatchSolver(solver, getfield(solver, :patches)[2])
        @test states.patches[2][padded_index(fine, 1, 1, 1), 1] ≈ 1 + fine.h[1]
        @test fine.h[1] ≈ root.h[1] / 3
    end

    # The temporary center patch used to plan tags must never evaluate the
    # user's fine-grid IC. Only the final off-center region has fine nodes.
    guarded = (x, y, z, h) -> begin
        h < 1 / 200 && abs(x - 0.5) < 0.08 &&
            error("IC evaluated on the temporary AMR seed")
        Prim(p=1.0, rho=1.0 + h, u=(0.0, 0.0, 0.0))
    end
    selected, _ = setup(amr_test_problem(guarded),
        Numerics(n_global=(96, 1, 1), filter=nothing,
                 amr=AMR(initial=MovingAMRWindow(0.72),
                         tag_threshold=Inf)))
    @test refined_region(selected).offset[1] > 48
end

@testset "AMR frontend sensor and empty-selection behavior" begin
    uniform = amr_test_problem()
    @test_throws ArgumentError setup(uniform,
        Numerics(n_global=(64, 1, 1), filter=nothing,
                 amr=AMR(initial=:sensor)))
    @test_throws ArgumentError setup(uniform,
        Numerics(n_global=(64, 1, 1), filter=nothing,
                 amr=AMR(initial=(x, y, z, t) -> false)))
    # A tiled, regridded run has an empty form and starts in it.
    unrefined, Qu = setup(uniform,
        Numerics(n_global=(64, 1, 1), filter=nothing,
                 amr=AMR(initial=:sensor, tile=8)))
    @test nlevels(unrefined) == 2 && isempty(level_regions(unrefined, 1))
    @test length(Qu.patches) == 1
    @test_throws ArgumentError setup(uniform,
        Numerics(n_global=(64, 1, 1), filter=nothing,
                 amr=AMR(initial=:sensor, tile=8, regrid_interval=0)))
    invalid = amr_test_problem((x, y, z, h) ->
        Prim(p=-1.0, rho=1.0, u=(0.0, 0.0, 0.0)))
    @test_throws SolverFailure setup(invalid,
        Numerics(n_global=(64, 1, 1), filter=nothing,
                 amr=AMR(initial=:sensor)))

    jump = amr_test_problem((x, y, z, h) ->
        Prim(p=1.0, rho=x < 0.7 ? 1.0 : 1.3, u=(0.0, 0.0, 0.0)))
    solver, _ = setup(jump,
        Numerics(n_global=(96, 1, 1), filter=nothing,
                 amr=AMR(initial=:sensor, tag_buffer=2)))
    region = refined_region(solver)
    @test region.offset[1] < 70 < region.offset[1] + region.extent[1]

    shear = amr_test_problem((x, y, z, h) ->
        Prim(p=1.0, rho=1.0, u=(x < 0.7 ? 0.0 : 1.0, 0.0, 0.0)))
    sensed, _ = setup(shear,
        Numerics(n_global=(96, 1, 1), filter=nothing,
                 amr=AMR(initial=:sensor, tag_threshold=Inf,
                         tag_sensor_threshold=1e-8, tag_buffer=2)))
    @test nlevels(sensed) == 2
end

@testset "AMR frontend preserves explicit layout and follows time predicate" begin
    region = BlockRegion((40, 0, 0), (16, 1, 1))
    prob = amr_test_problem((x, y, z, h) ->
        Prim(p=1.0, rho=1.0 + 0.1sin(2pi * x), u=(0.0, 0.0, 0.0)))
    common = (n_global=(96, 1, 1), filter=nothing,
              art=ArtificialProperties(enabled=false))
    grouped, _ = setup(prob, Numerics(; common..., amr=AMR(initial=region)))
    @test refined_region(grouped) == region

    moving = (x, y, z, t) -> abs(x - (0.45 + 0.2t)) < 0.035
    solver, states = setup(prob, Numerics(; common...,
        amr=AMR(initial=moving, regrid_interval=1,
                tag_threshold=Inf, tag_buffer=1)))
    first_region = refined_region(solver)
    solver.t = 1.0
    getfield(solver, :regrid).checks += 1
    @test CompactLES.regrid!(solver, states, Workspace(states), nothing)
    @test refined_region(solver).offset[1] > first_region.offset[1]
end

@testset "AMR frontend tiled bootstrap drops seed layout" begin
    solver, states = setup(amr_test_problem(),
        Numerics(n_global=(96, 1, 1), filter=nothing,
                 art=ArtificialProperties(enabled=false),
                 amr=AMR(initial=MovingAMRWindow(0.72),
                         tag_threshold=Inf, tag_buffer=1, tile=8,
                         tile_lifetime=10)))
    regions = level_regions(solver, 1)
    @test !isempty(regions)
    @test all(r -> r.offset[1] > 48, regions)
    @test 1 <= length(states.patches) <= 1 + length(regions)

    bump = amr_test_problem((x, y, z, h) ->
        Prim(p=1.0, rho=1.0 + 0.3exp(-((x - 0.72) / 0.025)^2),
             u=(0.0, 0.0, 0.0)))
    sensed, _ = setup(bump,
        Numerics(n_global=(96, 1, 1), filter=nothing,
                 amr=AMR(initial=:sensor, tag_buffer=1, tile=8)))
    @test all(r -> r.offset[1] > 48, level_regions(sensed, 1))
end

@testset "AMR frontend refines a jump at rest under sensor tags" begin
    # A density step at rest over 0.6 < x < 0.8 (nodes 58 to 77): the
    # artificial coefficients of a state at rest stay under the sensor
    # threshold, so the initial layout is tagged by the density criterion at
    # its default threshold, and the given threshold holds from then on. The
    # box, which has no empty form, is placed there too.
    jump(x, x0) = 0.5 * (1 + tanh((x - x0) / 0.01))
    step(x, a, b) = jump(x, a) - jump(x, b)
    rest = amr_test_problem((x, y, z, h) ->
        Prim(p=1.0, rho=1.0 + 3step(x, 0.6, 0.8), u=(0.0, 0.0, 0.0)))
    sensed = (initial=:sensor, tag_threshold=Inf, tag_sensor_threshold=0.5)
    covers(regions, g) = any(r -> r.offset[1] < g <= r.offset[1] + r.extent[1], regions)
    for tile in (8, 0)
        solver, _ = setup(rest, Numerics(n_global=(96, 1, 1), filter=nothing,
                                         amr=AMR(; sensed..., tile)))
        regions = level_regions(solver, 1)
        @test covers(regions, 58) && covers(regions, 77)
        @test getfield(solver, :regrid).threshold == Inf
    end
    # Beside it, a slab moving at u = -1 over 0.2 < x < 0.4 (nodes 20 to
    # 39), whose trailing edge at node 39 the sensor tags at setup (the
    # leading edge falls under the threshold); the step at rest is then left
    # to the given criteria.
    both = amr_test_problem((x, y, z, h) ->
        Prim(p=1.0, rho=1.0 + 3step(x, 0.6, 0.8), u=(-step(x, 0.2, 0.4), 0.0, 0.0)))
    solver, _ = setup(both, Numerics(n_global=(96, 1, 1), filter=nothing,
                                     amr=AMR(; sensed..., tile=8)))
    regions = level_regions(solver, 1)
    @test covers(regions, 20) || covers(regions, 39)
    @test !covers(regions, 58) && !covers(regions, 77)
end

@testset "AMR frontend widens a given start to the tagged cells" begin
    # A density bump at x = 0.72 (node 70) beside a given region over nodes
    # 41 to 56, which leaves the bump and its tag buffer outside it. A run
    # that regrids takes the two together as its first layout, as one box and
    # as tiles, with a warning, and fills it from the initial condition at
    # each patch's own spacing. A fixed layout stays as given, and so does a
    # region of a regridded run that holds the buffered bump.
    profile(x) = 1.0 + 0.3exp(-((x - 0.72) / 0.025)^2)
    bump = amr_test_problem((x, y, z, h) -> Prim(p=1.0, rho=profile(x) + h,
                                                 u=(0.0, 0.0, 0.0)))
    region = BlockRegion((40, 0, 0), (16, 1, 1))
    common = (n_global=(96, 1, 1), filter=nothing,
              art=ArtificialProperties(enabled=false))
    covers(regions, g) = any(r -> r.offset[1] < g <= r.offset[1] + r.extent[1], regions)
    fixed, _ = setup(bump, Numerics(; common..., amr=AMR(initial=region)))
    @test refined_region(fixed) == region
    for tile in (0, 8)
        num = Numerics(; common..., amr=AMR(initial=region, regrid_interval=5,
                                            tag_buffer=2, tile=tile))
        solver, states = @test_logs (:warn, r"were widened") match_mode=:any setup(bump, num)
        regions = level_regions(solver, 1)
        @test covers(regions, 41) && covers(regions, 56) && covers(regions, 70)
        tile == 0 && @test length(regions) == 1
        spec = getfield(solver, :regrid)
        @test spec.checks == 0 && all(==(0), values(spec.created[1]))
        for (p, Q) in zip(getfield(solver, :patches), states.patches)
            ps = CompactLES.PatchSolver(solver, p)
            @test Q[padded_index(ps, 1, 1, 1), 1] ≈
                  profile(CompactLES.xcoord(ps, 1, 1)) + ps.h[1]
        end
    end
    holding = BlockRegion((56, 0, 0), (28, 1, 1))
    kept, _ = @test_logs min_level=Base.CoreLogging.Warn setup(bump,
        Numerics(; common..., amr=AMR(initial=holding, regrid_interval=5, tag_buffer=2)))
    @test refined_region(kept) == holding
end

@testset "AMR frontend rejects ambiguous or invalid configuration" begin
    prob = amr_test_problem()
    region = BlockRegion((40, 0, 0), (16, 1, 1))
    base = (n_global=(64, 1, 1), filter=nothing)
    # A refinement setting is a field of `AMR`, not a keyword of `Numerics`.
    for flat in ((; refine=region), (; tag_buffer=5))
        @test_throws ArgumentError Numerics(; base..., amr=AMR(initial=region), flat...)
    end
    @test_throws ArgumentError setup(prob,
        Numerics(; base..., amr=AMR(initial=:unknown)))
    @test_throws ArgumentError setup(prob,
        Numerics(; base..., amr=AMR(initial=BlockRegion[])))
    @test_throws ArgumentError setup(prob,
        Numerics(; base..., amr=AMR(initial=(x, y, z, t) -> 1)))
end
