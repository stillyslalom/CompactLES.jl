# Two contracts of the outer run loop.
#
# The first is that observation does not disturb the calculation. The
# artificial coefficient arrays outlive the step that filled them: `max_rate`
# reads them for the diffusive part of the next CFL rate, and the sensor tag
# criterion reads them at a regrid check. A field dump, a profile, or a
# dissipation measurement asks `compute_artificial!` for the same arrays, so
# without the preservation in `preserving_artificial` a run whose diagnostics
# are called between steps takes different steps from one whose are not. The
# sets below run each case twice, observing at every step boundary in one of
# the two, and compare the step sequence, the final state, and the regrid
# decisions bit for bit.
#
# The second is that every accepted step advances the clock. `solver.t` carries
# the solver's element type, so an endpoint or a scheduled instant it cannot
# represent has to be resolved before it is compared against, and a step below
# the spacing of the floating-point grid at `t` has to be diagnosed rather than
# repeated to `nmax`. Every run here is bounded by `nmax`, so a regression
# fails the count instead of hanging.
#
# Serial. Included by runtests.jl, and runnable on its own:
#
#   julia --project=. test/runloop_tests.jl

if !@isdefined(CL)
    using MPI
    MPI.Init(threadlevel=:funneled)
    using CompactLES
    using Test
    const CL = CompactLES
end

const rl_per = (PeriodicBC(), PeriodicBC())

# A shock tube one cell thick in the transverse directions, with the artificial
# properties live so the coefficient arrays carry something to preserve.
function rl_tube(; N=64, cfl=0.15, backend=CPUBackend(), kw...)
    h = 1.0 / (N - 1)
    wall = (SlipWallBC(), SlipWallBC())
    solver = Solver(n_global=(N, 1, 1), L_domain=(1.0, h, h),
                    bcs=(wall, rl_per, rl_per), cfl=cfl, backend=backend,
                    filter_interval=1; kw...)
    Q = allocate_state(solver)
    initialize!(solver, Q, (x, y, z) ->
        x < 0.5 ? Prim(u=(0, 0, 0), p=1.0, rho=1.0) :
                  Prim(u=(0, 0, 0), p=0.1, rho=0.125))
    return solver, Q
end

# `steps` steps of `build()`, calling `observe!(solver, state)` after each one
# and recording the step taken and the level-1 regions in force. `tfinal` is
# out of reach, so `nmax` sets the length.
function rl_history(build, observe!; steps, refined=false)
    solver, state = build()
    dts = Float64[]
    regions = Vector{Vector{CL.BlockRegion}}()
    cb = Callback(EveryStep(1), function (s, q)
        push!(dts, Float64(s.dt_prev))
        refined && push!(regions, CL.level_regions(s, 1))
        observe!(s, q)
        return false
    end)
    run!(solver, state; tfinal=1e6, nmax=steps, callback=cb)
    return (solver=solver, state=state, dts=dts, regions=regions)
end

rl_states(Q) = [parent(Q)]
rl_states(states::Vector{<:CL.ConservedState}) = [parent(q) for q in states]

# The comparison every observation set below makes: same steps, same clock,
# same state, same refinement history.
function rl_identical(a, b)
    @test a.dts == b.dts
    @test a.solver.t == b.solver.t
    @test a.solver.step == b.solver.step
    @test rl_states(a.state) == rl_states(b.state)
    @test a.regions == b.regions
    return nothing
end

@testset "observation between steps leaves the trajectory" begin
    dir = mktempdir()
    steps = 6
    quiet = (s, q) -> nothing
    function watch(s, q)
        field_array(s, q, :beta_art)
        line_profile(s, q, :mu_art)
        dissipation_rate(s, q)
        save_vtk(s, q, joinpath(dir, "watch"); fields=(:rho, :mu_art, :sensor))
        return nothing
    end
    base = rl_history(rl_tube, quiet; steps=steps)
    seen = rl_history(rl_tube, watch; steps=steps)
    rl_identical(base, seen)
    # The case has to be one where the coefficients matter at all, or the
    # comparison above passes on a solver that never fills them.
    @test maximum(base.solver.beta_art) > 0
    # And one where the preservation is what makes it pass: recomputing the
    # coefficients from the state at the step boundary gives a different
    # diffusive rate, which is the defect this set covers.
    solver, Q = rl_tube()
    run!(solver, Q; tfinal=1e6, nmax=1)
    dt_kept = CL.compute_dt(solver, Q)
    CL.compute_primitives_and_gradients!(solver, Q)
    CL.compute_artificial!(solver, Q)
    @test CL.compute_dt(solver, Q) != dt_kept
end

@testset "observation leaves the trajectory on device storage" begin
    steps = 5
    cpu_ka = CL.KernelAbstractions.CPU()
    build() = rl_tube(; backend=DeviceBackend(cpu_ka))
    quiet = (s, q) -> nothing
    watch = (s, q) -> (field_array(s, q, :kappa_art); dissipation_rate(s, q);
                       nothing)
    CL.FORCE_KA[] = true
    CL.FORCE_DEVICE_EXCHANGE[] = true
    base, seen = try
        (rl_history(build, quiet; steps=steps),
         rl_history(build, watch; steps=steps))
    finally
        CL.FORCE_KA[] = false
        CL.FORCE_DEVICE_EXCHANGE[] = false
    end
    rl_identical(base, seen)
end

@testset "observation leaves a patched trajectory" begin
    dir = mktempdir()
    steps = 5
    function build()
        h = 1.0 / 95
        solver = Solver(n_global=(96, 1, 1), L_domain=(1.0, h, h),
                        bcs=(rl_per, rl_per, rl_per), cfl=0.2,
                        patch_grid=(2, 1, 1), filter_interval=1)
        states = allocate_state(solver)
        initialize!(solver, states, (x, y, z) ->
            Prim(u=(0.2, 0, 0), p=1.0 + 0.5exp(-200 * (x - 0.5)^2),
                 rho=1.0 + 0.5exp(-200 * (x - 0.5)^2)))
        return solver, states
    end
    quiet = (s, q) -> nothing
    watch = (s, q) -> (dissipation_rate(s, q);
                       save_vtk(s, q, joinpath(dir, "patched");
                                fields=(:rho, :beta_art)); nothing)
    rl_identical(rl_history(build, quiet; steps=steps),
                 rl_history(build, watch; steps=steps))
end

# The artificial diffusivity number `_tag_sensor_point!` thresholds, over the
# root patch of a refined solver.
function rl_sensor_number(solver)
    root = CL.PatchSolver(solver, solver.patches[1])
    q = 0.0
    for i in 1:root.decomp.n_local[1]
        I = gidx(root, i, 1, 1)
        ν = (root.mu_art[I] + root.beta_art[I]) / root.rho[I] +
            root.kappa_art[I] / (root.rho[I] * root.cp_mix[I]) +
            maximum(D[I] for D in root.D_art)
        q = max(q, ν / (root.c[I] * root.h[1]))
    end
    return q
end

@testset "observation leaves a regridded trajectory" begin
    steps = 8
    # The δ⁴ criterion is parked, so the sensor criterion alone decides the
    # box and the tag reads exactly the fields an observational call rebuilds.
    function build(threshold)
        solver = Solver(n_global=(120, 1, 1), L_domain=(1.0, 1.0, 1.0),
                        bcs=((SlipWallBC(), SlipWallBC()), rl_per, rl_per),
                        cfl=0.2, refine=CL.BlockRegion((50, 0, 0), (31, 1, 1)),
                        regrid_interval=2, tag_threshold=1e6,
                        tag_sensor_threshold=threshold, tag_buffer=2)
        states = allocate_state(solver)
        initialize!(solver, states, (x, y, z) ->
            x < 0.5 ? Prim(u=(0, 0, 0), p=1.0, rho=1.0) :
                      Prim(u=(0, 0, 0), p=0.1, rho=0.125))
        return solver, states
    end
    # Half the number the shock reaches, measured on a run of its own, which
    # puts the threshold where the criterion tags the shock and little else.
    cal, cq = build(0.0)
    run!(cal, cq; tfinal=1e6, nmax=4)
    threshold = rl_sensor_number(cal) / 2
    @test threshold > 0
    quiet = (s, q) -> nothing
    watch = (s, q) -> (dissipation_rate(s, q); nothing)
    base = rl_history(() -> build(threshold), quiet; steps=steps, refined=true)
    seen = rl_history(() -> build(threshold), watch; steps=steps, refined=true)
    rl_identical(base, seen)
    # The criterion has to have moved the region, or the histories above agree
    # on a decision nothing exercised.
    @test length(unique(base.regions)) > 1
end

@testset "sensor tagging reads the state it is given" begin
    solver = Solver(n_global=(120, 1, 1), L_domain=(1.0, 1.0, 1.0),
                    bcs=((SlipWallBC(), SlipWallBC()), rl_per, rl_per),
                    cfl=0.2, refine=CL.BlockRegion((50, 0, 0), (31, 1, 1)),
                    regrid_interval=1000, tag_threshold=1e6,
                    tag_sensor_threshold=1e-3, tag_buffer=2)
    states = allocate_state(solver)
    initialize!(solver, states, (x, y, z) ->
        x < 0.5 ? Prim(u=(0, 0, 0), p=1.0, rho=1.0) :
                  Prim(u=(0, 0, 0), p=0.1, rho=0.125))
    run!(solver, states; tfinal=1e6, nmax=6)
    # The primitives the criterion reads are the ones the previous step's last
    # stage left, which are not those of the state under test. The sweep
    # refreshes them, which is what makes the decision a function of that state.
    root = CL.PatchSolver(solver, solver.patches[1])
    stale = copy(root.rho)
    region = CL.tagged_region(solver, states[1])
    @test region !== nothing
    @test root.rho != stale
    # A refresh from the state under test, and an observational call that
    # performs one, therefore leave the decision alone.
    refresh_primitives!(solver, states)
    @test CL.tagged_region(solver, states[1]) == region
    dissipation_rate(solver, states)
    @test CL.tagged_region(solver, states[1]) == region
end

# --- The clock ---------------------------------------------------------------

# A uniform periodic box in either precision: the step is the acoustic one and
# stays constant, so the count to any endpoint is known ahead of the run.
function rl_box(::Type{T}; cfl=0.5, kw...) where {T}
    solver = Solver(n_global=(24, 12, 12), L_domain=(one(T), one(T), one(T)),
                    bcs=ntuple(_ -> rl_per, 3),
                    eos=IdealSpecies(T, "gas"; R=one(T), gamma=T(1.4)),
                    transport=Transport{T}(), art=ArtParams{T}(enabled=false),
                    deriv=lele_d1_6(T), filt=compact_filter(T(0.45), T),
                    cfl=T(cfl), filter_interval=0; kw...)
    Q = allocate_state(solver)
    initialize!(solver, Q, (x, y, z) -> Prim(u=(0, 0, 0), p=1, rho=1))
    return solver, Q
end

@testset "endpoints are reached in the solver's own precision" begin
    # 0.7 is below its Float32 neighbour and 0.3 is above it, so the two cover
    # an endpoint the clock rounds down to and one it rounds up to. Before the
    # conversion the first stalled: the clock reached 0.699999988079071 and
    # every remaining step took the 1.1920929e-8 remainder, which the clock
    # cannot add, until nmax stopped the run.
    for tfinal in (0.7, 0.3)
        solver, Q = rl_box(Float32)
        run!(solver, Q; tfinal=tfinal, nmax=300)
        @test solver.step < 300
        @test solver.t == Float32(tfinal)
        @test abs(Float64(solver.t) - tfinal) <= eps(Float32(tfinal))
        # A second call with the same endpoint takes no step at all.
        n = solver.step
        run!(solver, Q; tfinal=tfinal, nmax=300)
        @test solver.step == n
    end
    solver, Q = rl_box(Float64)
    run!(solver, Q; tfinal=0.7, nmax=300)
    @test solver.step < 300
    @test solver.t == 0.7
end

@testset "a step that cannot advance the clock is diagnosed" begin
    # A restart at a time whose floating-point spacing exceeds the timestep.
    # The run has no way forward and says so, rather than turning over its
    # remaining steps at no advance.
    for (T, t0) in ((Float64, 1e300), (Float32, 1f30))
        solver, Q = rl_box(T)
        solver.t = T(t0)
        err = try
            run!(solver, Q; tfinal=2 * Float64(t0), nmax=20)
            nothing
        catch e
            e
        end
        @test err isa CL.SolverFailure
        @test err.reason === :no_progress
        @test solver.step == 0
    end
end

@testset "scheduled instants land in either precision" begin
    for T in (Float64, Float32)
        hits = Float64[]
        solver, Q = rl_box(T)
        cb = (Callback(AtTime([0.1, 0.25]), (s, q) -> (push!(hits, s.t); false)),
              Callback(EveryTime(0.05), (s, q) -> false))
        run!(solver, Q; tfinal=0.3, nmax=300, callback=cb)
        @test solver.step < 300
        @test solver.t == T(0.3)
        @test length(hits) == 2
        # Landed, not merely passed: each instant is within the clock's own
        # resolution of the scheduled time.
        @test hits[1] ≈ 0.1 atol = 8eps(T(0.1))
        @test hits[2] ≈ 0.25 atol = 8eps(T(0.25))
    end
end

@testset "subcycled runs reach the endpoint" begin
    solver = Solver(n_global=(96, 1, 1), L_domain=(1.0, 1.0, 1.0),
                    bcs=((SlipWallBC(), SlipWallBC()), rl_per, rl_per),
                    cfl=0.3, subcycle=true,
                    refine=CL.BlockRegion((40, 0, 0), (17, 1, 1)))
    states = allocate_state(solver)
    initialize!(solver, states, (x, y, z) ->
        Prim(u=(0, 0, 0), p=1.0 + 0.2exp(-200 * (x - 0.5)^2), rho=1.0))
    fired = Ref(0)
    cb = Callback(EveryTime(0.01), (s, q) -> (fired[] += 1; false))
    run!(solver, states; tfinal=0.03, nmax=400, callback=cb)
    @test solver.step < 400
    @test solver.t == 0.03
    @test fired[] == 3
end
