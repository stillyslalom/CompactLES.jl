# The serial suite's tests of the frontend: knobs, endpoint conversion, the Numerics
# groups and the displays. Included by test/serial_suite.jl, which loads the package
# and defines the imports and helpers used here; not runnable on its own.

@testset "with_knob restores the switches it sets" begin
    # The value of the block comes back, and each switch returns to what it
    # held, on a normal exit and on a throw.
    @test CL.with_knob(() -> (CL.FORCE_KA[], CL.TILE_SLOTS[]),
                       CL.FORCE_KA => true, CL.TILE_SLOTS => 3) == (true, 3)
    @test (CL.FORCE_KA[], CL.TILE_SLOTS[]) == (false, 0)
    @test_throws ErrorException CL.with_knob(() -> error("thrown"), CL.NODE_SIZE => 4)
    @test CL.NODE_SIZE[] == 0
    @test CL.with_knob(() -> 7) == 7
end

@testset "Problem preserves endpoint conversion" begin
    prob = Problem(domain=((0, 1), (0, pi), (0, 2pi)), bcs=per3,
                   ic=(x, y, z) -> Prim(p=1.0, rho=1.0))
    @test prob.domain === ((0.0, 1.0), (0.0, Float64(pi)), (0.0, 2pi))
    @test prob.ic(0.0, 0.0, 0.0).p == 1.0
end

@testset "Numerics groups, shorthands and copies" begin
    num = Numerics(n_global=(16, 1, 1))
    @test num.filter.interval == 1 && num.filter.cfl == 0.35
    @test num.patch_interfaces.flux === :ghost && num.execution.dims === nothing
    @test Numerics(n_global=(16, 1, 1), filter=nothing).filter.interval == 0
    shorthand = Numerics(n_global=(16, 1, 1), filter=pyranda_filter(),
                         patch_interfaces=:closure)
    @test shorthand.filter.scheme isa typeof(pyranda_filter())
    @test shorthand.filter.interval == 1
    @test shorthand.patch_interfaces.flux === :closure
    copied = Numerics(num; cfl=0.3, filter=StateFilter(num.filter; cfl=0.0))
    @test copied.cfl == 0.3 && copied.filter.cfl == 0.0 && copied.n_global == num.n_global
    @test num.cfl == 0.5
    @test_throws ArgumentError Numerics(n_global=(16, 1, 1), nonexistent=1)
    @test_throws ArgumentError Numerics(n_global=(16, 1, 1), filter=1.0)
    @test_throws ArgumentError StateFilter(StateFilter(); nonexistent=1)

    # A group's field is not a keyword of `Numerics`.
    @test_throws ArgumentError Numerics(n_global=(16, 1, 1), filter_interval=0)
    @test_throws ArgumentError Numerics(n_global=(16, 1, 1), refine=nothing)

    # A copy keeps the element type of a parametric group and still validates.
    art = ArtificialProperties(ArtificialProperties{Float32}(); C_D=3.0)
    @test art.C_D === 3.0f0
    @test StepControl(StepControl(retries=4); validity=:permissive).retries == 4
    @test_throws ArgumentError StepControl(StepControl(); retries=-1)
    @test AMR(AMR(tile=12); subcycle=true).tile == 12

    # `storage` names the backend and precision by an element or array type.
    for (storage, precision) in ((Float32, Float32), (Array, nothing),
                                 (Array{Float32}, Float32), (Array{Float64}, Float64))
        ex = Execution(; storage)
        @test ex.backend isa CompactLES.CPUBackend && ex.precision === precision
    end
    @test Execution(Execution(dims=(1, 1, 1)); storage=Float32).precision === Float32
    @test Execution(Execution(dims=(1, 1, 1)); storage=Float32).dims == (1, 1, 1)
    device = DeviceBackend(CompactLES.KernelAbstractions.CPU())
    @test Execution(storage=device).backend === device
    @test Execution(storage=device => Float32).precision === Float32
    @test_throws ArgumentError Execution(storage=device => Int)
    @test_throws ArgumentError Execution(Execution(); precision=Float32)
    @test_throws ArgumentError Execution(storage=Array{Int})
    @test_throws ArgumentError Execution(storage=AbstractArray)
    @test_throws ArgumentError Execution(storage=:gpu)

    # The groups reach the solver under its flat names.
    prob = Problem(domain=((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)), bcs=per3,
                   ic=(x, y, z) -> Prim(p=1.0, rho=1.0 + 0.1sin(2π * x)))
    solver, _ = setup(prob, Numerics(n_global=(32, 1, 1),
                                     filter=StateFilter(compact_filter(0.49);
                                                        interval=2, cfl=0.0)))
    @test solver.filter_interval == 2 && solver.filter_cfl == 0
    @test solver.control.retries == 0

    # A spherical origin takes retries unless the deck gives its own control.
    sphere = Problem(metric=SphericalMetric(),
                     domain=((0.0, 1.0), (π / 2, π / 2 + 1), (0.0, 1.0)),
                     bcs=((OriginBC(), SlipWallBC()), per3[2], per3[3]),
                     ic=(r, θ, φ) -> Prim(p=1.0 + 0.1exp(-(r / 0.3)^2), rho=1.0))
    origin, _ = setup(sphere, Numerics(n_global=(32, 1, 1)))
    @test origin.control.retries == 4
    explicit, _ = setup(sphere, Numerics(n_global=(32, 1, 1), control=StepControl()))
    @test explicit.control.retries == 0

    # A preset is a keyword set; keywords after it override it, and merge
    # combines two, the later winning.
    @test num.filter.scheme.alpha == 0.47
    resolved = Numerics(Presets.resolved(); n_global=(16, 1, 1), cfl=0.4)
    @test resolved.filter.scheme.alpha == 0.49 && resolved.cfl == 0.4
    shocked = Numerics(Presets.refined_shock(); n_global=(16, 1, 1))
    @test shocked.filter.scheme.alpha == 0.45 && shocked.patch_interfaces.flux === :closure
    cold = Numerics(merge(Presets.resolved(), Presets.converging(cold_ambient=true));
                    n_global=(16, 1, 1), filter=nothing)
    @test cold.control.retries == 4 && cold.control.validity === :permissive
    @test cold.filter.interval == 0
    walls = Numerics(Presets.smooth_walls(); n_global=(16, 1, 1)).deriv
    @test repr(walls.closures) == repr(lele_d1_6(closures=:brady_livescu).closures)
    @test repr(walls.closures) != repr(lele_d1_6().closures)
    @test occursin("filter: off",
                   sprint(show, MIME("text/plain"), Numerics(num; filter=nothing)))
end

@testset "concise frontend displays" begin
    prob = Problem(name="display test",
                   domain=((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)), bcs=per3,
                   ic=(x, y, z) -> Prim(u=(x, y, z), p=1.0, rho=1.0))
    num = Numerics(n_global=(12, 12, 12), art=ArtificialProperties(enabled=false))
    solver, Q = setup(prob, num)

    @test Q isa CompactLES.SimulationState
    @test parent(Q) isa Array{Float64,4}
    @test size(Q) == (20, 20, 20, 5)
    Q[1, 1, 1, 1] = 2.0
    @test parent(Q)[1, 1, 1, 1] == 2.0
    @test parent(view(Q, :, :, :, 1)) === parent(Q)
    @test copy(Q) isa CompactLES.SimulationState
    @test zero(Q) isa CompactLES.SimulationState

    q_display = sprint(show, MIME("text/plain"), Q)
    pair_display = sprint(show, MIME("text/plain"), (solver, Q))
    @test q_display == "SimulationState{Float64} 20×20×20×5"
    @test length(pair_display) < 200
    @test occursin("Solver(grid=12 × 12 × 12", pair_display)
    @test occursin(q_display, pair_display)
    @test run!(solver, Q; tfinal=1.0, nmax=0) === Q

    solver_display = sprint(show, MIME("text/plain"), solver)
    problem_display = sprint(show, MIME("text/plain"), prob)
    numerics_display = sprint(show, MIME("text/plain"), num)
    @test occursin("5 conserved variables", solver_display)
    @test occursin(prob.name, problem_display)
    @test !occursin("#", problem_display)       # do not dump the IC closure type
    @test occursin("artificial properties: disabled", numerics_display)
    @test all(length.((solver_display, problem_display, numerics_display)) .< 500)

    # EOS objects: the NASA-9 records must not dump their fit coefficients.
    nasa = Nasa9Mixture(["He", "CO2"])
    nasa_display = sprint(show, MIME("text/plain"), nasa)
    @test occursin("He", nasa_display) && occursin("CO2", nasa_display)
    @test occursin("T_guess = 300.0", nasa_display)
    @test !occursin("Nasa9Interval", nasa_display)
    @test length(nasa_display) < 200
    @test length(sprint(show, nasa)) < 80
    @test length(sprint(show, MIME("text/plain"), nasa.sp)) < 300
    @test length(sprint(show, MIME("text/plain"), nasa.sp[1].intervals)) < 300
    @test sprint(show, IdealSpecies("gas"; R=1.0, gamma=1.4)) ==
          "IdealSpecies(\"gas\"; R=1.0, gamma=1.4)"
    ideal_display = sprint(show, MIME("text/plain"), IdealMixture(["He", "CO2"]))
    @test occursin("gamma", ideal_display) && length(ideal_display) < 200
    @test sprint(show, StiffenedGas()) ==
          "StiffenedGas(gamma=4.4, p_inf=6.0e8, cv=1816.0, name=\"liquid\")"
end
