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
            Numerics(n_global=(n, 1, 1), filter_interval=0,
                     art=ArtificialProperties(enabled=false),
                     amr=AMR(initial=selector, tag_buffer=2, regrid_interval=0)))
        region = refined_region(solver)
        center = (region.offset[1] + region.extent[1] / 2) / n
        @test abs(center - 0.65) < 6 / n
        @test nlevels(solver) == 2
        @test getfield(solver, :regrid) === nothing
        @test 1 <= length(states) <= 2  # fine patch exists only on its owner ranks
    end
end

@testset "AMR frontend evaluates IC with actual fine spacing" begin
    ic = (x, y, z, h) -> Prim(p=1.0, rho=1.0 + h, u=(0.0, 0.0, 0.0))
    solver, states = setup(amr_test_problem(ic),
        Numerics(n_global=(96, 1, 1), filter_interval=0,
                 art=ArtificialProperties(enabled=false),
                 amr=AMR(initial=(x, y, z, t) -> abs(x - 0.7) < 0.04)))
    root = CompactLES.PatchSolver(solver, getfield(solver, :patches)[1])
    @test states[1][padded_index(root, 1, 1, 1), 1] ≈ 1 + root.h[1]
    if length(states) > 1
        fine = CompactLES.PatchSolver(solver, getfield(solver, :patches)[2])
        @test states[2][padded_index(fine, 1, 1, 1), 1] ≈ 1 + fine.h[1]
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
        Numerics(n_global=(96, 1, 1), filter_interval=0,
                 amr=AMR(initial=MovingAMRWindow(0.72),
                         tag_threshold=Inf)))
    @test refined_region(selected).offset[1] > 48
end

@testset "AMR frontend sensor and empty-selection behavior" begin
    uniform = amr_test_problem()
    @test_throws ArgumentError setup(uniform,
        Numerics(n_global=(64, 1, 1), filter_interval=0,
                 amr=AMR(initial=:sensor)))
    @test_throws ArgumentError setup(uniform,
        Numerics(n_global=(64, 1, 1), filter_interval=0,
                 amr=AMR(initial=(x, y, z, t) -> false)))
    # A tiled, regridded run has an empty form and starts in it.
    unrefined, Qu = setup(uniform,
        Numerics(n_global=(64, 1, 1), filter_interval=0,
                 amr=AMR(initial=:sensor, tile=8)))
    @test nlevels(unrefined) == 2 && isempty(level_regions(unrefined, 1))
    @test length(Qu) == 1
    @test_throws ArgumentError setup(uniform,
        Numerics(n_global=(64, 1, 1), filter_interval=0,
                 amr=AMR(initial=:sensor, tile=8, regrid_interval=0)))
    invalid = amr_test_problem((x, y, z, h) ->
        Prim(p=-1.0, rho=1.0, u=(0.0, 0.0, 0.0)))
    @test_throws SolverFailure setup(invalid,
        Numerics(n_global=(64, 1, 1), filter_interval=0,
                 amr=AMR(initial=:sensor)))

    jump = amr_test_problem((x, y, z, h) ->
        Prim(p=1.0, rho=x < 0.7 ? 1.0 : 1.3, u=(0.0, 0.0, 0.0)))
    solver, _ = setup(jump,
        Numerics(n_global=(96, 1, 1), filter_interval=0,
                 amr=AMR(initial=:sensor, tag_buffer=2)))
    region = refined_region(solver)
    @test region.offset[1] < 70 < region.offset[1] + region.extent[1]

    shear = amr_test_problem((x, y, z, h) ->
        Prim(p=1.0, rho=1.0, u=(x < 0.7 ? 0.0 : 1.0, 0.0, 0.0)))
    sensed, _ = setup(shear,
        Numerics(n_global=(96, 1, 1), filter_interval=0,
                 amr=AMR(initial=:sensor, tag_threshold=Inf,
                         tag_sensor_threshold=1e-8, tag_buffer=2)))
    @test nlevels(sensed) == 2
end

@testset "AMR frontend preserves explicit layout and follows time predicate" begin
    region = BlockRegion((40, 0, 0), (16, 1, 1))
    prob = amr_test_problem((x, y, z, h) ->
        Prim(p=1.0, rho=1.0 + 0.1sin(2pi * x), u=(0.0, 0.0, 0.0)))
    common = (n_global=(96, 1, 1), filter_interval=0,
              art=ArtificialProperties(enabled=false))
    legacy, Qlegacy = setup(prob, Numerics(; common..., refine=region))
    grouped, Qgrouped = setup(prob, Numerics(; common..., amr=AMR(initial=region)))
    @test refined_region(grouped) == refined_region(legacy)
    @test all(parent(Qgrouped[i]) == parent(Qlegacy[i]) for i in eachindex(Qlegacy))

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
        Numerics(n_global=(96, 1, 1), filter_interval=0,
                 art=ArtificialProperties(enabled=false),
                 amr=AMR(initial=MovingAMRWindow(0.72),
                         tag_threshold=Inf, tag_buffer=1, tile=8,
                         tile_lifetime=10)))
    regions = level_regions(solver, 1)
    @test !isempty(regions)
    @test all(r -> r.offset[1] > 48, regions)
    @test 1 <= length(states) <= 1 + length(regions)

    bump = amr_test_problem((x, y, z, h) ->
        Prim(p=1.0, rho=1.0 + 0.3exp(-((x - 0.72) / 0.025)^2),
             u=(0.0, 0.0, 0.0)))
    sensed, _ = setup(bump,
        Numerics(n_global=(96, 1, 1), filter_interval=0,
                 amr=AMR(initial=:sensor, tag_buffer=1, tile=8)))
    @test all(r -> r.offset[1] > 48, level_regions(sensed, 1))
end

@testset "AMR frontend rejects ambiguous or invalid configuration" begin
    prob = amr_test_problem()
    region = BlockRegion((40, 0, 0), (16, 1, 1))
    base = (n_global=(64, 1, 1), filter_interval=0)
    @test_throws ArgumentError setup(prob,
        Numerics(; base..., amr=AMR(initial=region), refine=region))
    @test_throws ArgumentError setup(prob,
        Numerics(; base..., amr=AMR(initial=region), tag_buffer=5))
    @test_throws ArgumentError setup(prob,
        Numerics(; base..., amr=AMR(initial=:unknown)))
    @test_throws ArgumentError setup(prob,
        Numerics(; base..., amr=AMR(initial=BlockRegion[])))
    @test_throws ArgumentError setup(prob,
        Numerics(; base..., amr=AMR(initial=(x, y, z, t) -> 1)))
end
