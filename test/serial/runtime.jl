# The serial suite's tests of initial states, parameter checks, callbacks, phase
# changes, checkpoints and smoke runs. Included by test/serial_suite.jl, which loads
# the package and defines the imports and helpers used here; not runnable on its
# own.

@testset "conserved_from_prim / nspecies round trip" begin
    eos = IdealMixture([IdealSpecies{Float64}("a", 1.0, 1.4),
                        IdealSpecies{Float64}("b", 0.2, 1.09)])
    @test nspecies(eos) == 2
    @test nspecies(IdealMixture(IdealSpecies("gas"; gamma=1.4, R=1.0))) == 1
    pr = Prim(u=(0.3, -0.1, 0.2), p=0.8, T_ion=1.7, Y=(0.35, 0.65))
    q = conserved_from_prim(eos, pr)
    ρ = q[1] + q[2]
    @test q[1] / ρ ≈ 0.35 atol = 1e-12
    @test q[3] / ρ ≈ 0.3 atol = 1e-12          # ρu / ρ
    Rm = 0.35 * eos.Rk[1] + 0.65 * eos.Rk[2]
    @test ρ * Rm * 1.7 ≈ 0.8 atol = 1e-12      # p = ρ R_m T_ion

    # Any two of (p, rho, T_ion) determine the state; all three spellings of the
    # same point must give the same conserved vector.
    ρ0 = 0.8 / (Rm * 1.7)
    from_rho_T = conserved_from_prim(eos, Prim(u=(0.3, -0.1, 0.2), rho=ρ0,
                                              T_ion=1.7, Y=(0.35, 0.65)))
    from_p_rho = conserved_from_prim(eos, Prim(u=(0.3, -0.1, 0.2), p=0.8,
                                              rho=ρ0, Y=(0.35, 0.65)))
    @test all(isapprox.(from_rho_T, q; rtol=1e-14))
    @test all(isapprox.(from_p_rho, q; rtol=1e-14))
    # The other two EOS models take the same three spellings. Each one's thermal
    # relation supplies the pressure that (rho, T_ion) implies.
    liquid = StiffenedGas(gamma=4.4, p_inf=6.0e8, cv=1816.0)
    p_liquid = 1000.0 * CL.gas_constant(liquid) * 300.0 - liquid.p_inf
    ideal = IdealSpecies("gas"; gamma=1.4, R=287.0)
    for (E, p_of) in ((liquid, p_liquid), (ideal, 1000.0 * 287.0 * 300.0))
        @test all(isapprox.(conserved_from_prim(E, Prim(rho=1000.0, T_ion=300.0)),
                            conserved_from_prim(E, Prim(p=p_of, T_ion=300.0));
                            rtol=1e-12))
    end
    # Exactly two, no more and no fewer.
    @test_throws ArgumentError Prim(p=1.0)
    @test_throws ArgumentError Prim(rho=1.0)
    @test_throws ArgumentError Prim(p=1.0, rho=1.0, T_ion=1.0)
end

@testset "parameter ranges: early errors naming the parameter" begin
    # Each check raises an ArgumentError whose message names the parameter.
    function names(f, name)
        err = try f(); nothing catch e; e end
        return err isa ArgumentError && occursin(name, err.msg)
    end
    wall = (SlipWallBC(), SlipWallBC())
    per = (PeriodicBC(), PeriodicBC())
    mk(; cfl=0.5, filter_interval=1, filter_cfl=0.35, L_domain=(1.0, 1.0, 1.0),
         n_global=(16, 12, 1), transport=ConstantTransport(), art=ArtificialProperties()) =
        build_solver(Problem(domain=ntuple(d -> (0.0, L_domain[d]), 3), bcs=(wall, per, per),
                             transport=transport, ic=nothing),
                     Numerics(n_global=n_global, cfl=cfl, art=art,
                              filter=StateFilter(interval=filter_interval, cfl=filter_cfl)))
    for (kw, name) in [((; predict=-1.0), "predict"), ((; max_growth=0.5), "max_growth"),
                       ((; dt_min=NaN), "dt_min"), ((; dt_min_ratio=1.0), "dt_min_ratio"),
                       ((; retries=-1), "retries"), ((; cfl_backoff=1.0), "cfl_backoff"),
                       ((; cfl_backoff=0.0), "cfl_backoff")]
        @test names(() -> StepControl(; kw...), name)
    end
    @test names(() -> compact_filter(0.5), "alphaf")
    @test names(() -> compact_filter(-0.6), "alphaf")
    @test names(() -> StiffenedGas(gamma=1.0), "gamma")
    @test names(() -> StiffenedGas(p_inf=-1.0), "p_inf")
    @test names(() -> StiffenedGas(cv=0.0), "cv")
    @test names(() -> Prim(p=Inf, rho=1.0), "finite")
    @test names(() -> Prim(p=1.0, rho=-1.0), "rho")
    @test names(() -> Prim(p=1.0, T_ion=-1.0), "T_ion")
    @test names(() -> Prim(p=1.0, rho=1.0, u=(NaN, 0, 0)), "u")
    @test names(() -> Prim(p=1.0, rho=1.0, Y=(1.1, -0.1)), "nonnegative")
    @test names(() -> Prim(p=1.0, rho=1.0, Y=(0.5, 0.6)), "sum to 1")
    @test names(() -> AtTime([0.1, NaN]), "finite")
    @test names(() -> EveryTime(Inf), "interval")
    @test names(() -> mk(cfl=0.0), "cfl")
    @test names(() -> mk(filter_interval=-1), "filter_interval")
    @test names(() -> mk(filter_cfl=-0.1), "filter_cfl")
    @test names(() -> mk(L_domain=(1.0, Inf, 1.0)), "domain")
    @test names(() -> mk(n_global=(16, 0, 1)), "n_global")
    @test names(() -> mk(transport=ConstantTransport(mu0=NaN)), "mu0")
    @test names(() -> mk(transport=ConstantTransport(Pr=0.0)), "Pr")
    @test names(() -> mk(transport=ConstantTransport(Sc=-1.0)), "Sc")
    @test names(() -> mk(art=ArtificialProperties(C_beta=-1.0)), "C_beta")
    @test names(() -> mk(art=ArtificialProperties(Y_tolerance=NaN)), "Y_tolerance")
    # A grid below the scheme minimum is reported with the binding scheme and
    # the extent required before any plan is built: the C8 filter at a wall.
    err = try mk(n_global=(8, 12, 1)); nothing catch e; e end
    @test err isa ArgumentError && occursin("n_global[1]", err.msg) &&
          occursin("at least 9", err.msg) && occursin("filter", err.msg)
    @test names(() -> mk(n_global=(16, 6, 1)), "n_global[2]")
    prob = Problem(domain=((0.0, 1.0), (0.0, Inf), (0.0, 1.0)), bcs=(wall, per, per),
                   ic=(x, y, z) -> Prim(p=1.0, rho=1.0))
    @test names(() -> setup(prob, Numerics(n_global=(16, 12, 1))), "domain")
end

@testset "callbacks: triggers, dt landing, composition, termination" begin
    wall3 = ((SlipWallBC(), SlipWallBC()), (PeriodicBC(), PeriodicBC()),
             (PeriodicBC(), PeriodicBC()))
    mkrun() = begin
        solver = build_solver(
            Problem(domain=((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)), bcs=wall3, ic=nothing),
            Numerics(n_global=(16, 12, 12), art=ArtificialProperties(enabled=false),
                     cfl=0.4))
        Q = allocate_state(solver)
        initialize!(solver, Q, (x, y, z) -> Prim(u=(0.2, 0, 0),
                                                 p=1 + 0.1exp(-40(x - 0.5)^2), rho=1.0))
        solver, Q
    end

    # AtTime must be landed on exactly, not overshot: dt is clipped inside run!,
    # which is the whole reason the trigger cannot be written by a caller.
    solver, Q = mkrun()
    hits = Float64[]
    targets = [0.02, 0.05, 0.11]
    run!(solver, Q; tfinal=0.2,
         callback=Callback(AtTime(targets), (s, _) -> (push!(hits, s.t); nothing)))
    @test length(hits) == 3
    @test hits ≈ targets rtol = 1e-14
    @test solver.t ≈ 0.2 rtol = 1e-14

    # Unsorted input is ordered, and a scalar is accepted.
    @test AtTime([0.3, 0.1, 0.2]).times == [0.1, 0.2, 0.3]
    @test AtTime(0.5).times == [0.5]

    # A time already behind the solver must not drive dt to zero and stall.
    solver, Q = mkrun()
    late = Float64[]
    run!(solver, Q; tfinal=0.01, nmax=20,
         callback=Callback(AtTime(-1.0), (s, _) -> (push!(late, s.t); nothing)))
    @test length(late) == 1 && solver.step > 0

    # EveryStep, and a bare function still runs every step with its return
    # value ignored — returning true must not stop the run.
    solver, Q = mkrun()
    steps = Int[]
    every = 0
    run!(solver, Q; tfinal=1e9, nmax=10,
         callback=(Callback(EveryStep(3), (s, _) -> (push!(steps, s.step); nothing)),
                   (s, _) -> (every += 1; true)))
    @test steps == [3, 6, 9]
    @test every == 10 && solver.step == 10

    # WhenState fires once by default and repeatedly with once=false.
    for (once, want) in ((true, 1), (false, 3))
        solver, Q = mkrun()
        fires = 0
        run!(solver, Q; tfinal=1e9, nmax=5,
             callback=Callback(WhenState((s, _) -> s.step >= 3; once=once),
                               (_, _) -> (fires += 1; nothing)))
        @test fires == want
    end

    # An effect returning true ends the run after that step.
    solver, Q = mkrun()
    run!(solver, Q; tfinal=1e9, nmax=1000,
         callback=Callback(WhenState((s, _) -> s.step >= 4), (_, _) -> true))
    @test solver.step == 4

    # BeforeWallLimit: a deadline already inside the margin fires at the first
    # check, `check_every` steps in, and once; one far off never fires.
    for (every, want) in ((1, 1), (3, 3))
        solver, Q = mkrun()
        fires = Int[]
        trigger = BeforeWallLimit(walltime=0.0, check_every=every, signals=())
        run!(solver, Q; tfinal=1e9, nmax=6,
             callback=Callback(trigger, (s, _) -> (push!(fires, s.step); nothing)))
        @test fires == [want]
        @test trigger.source === :walltime && isfinite(trigger.deadline)
    end
    solver, Q = mkrun()
    far = BeforeWallLimit(deadline=time() + 1e6, signals=())
    run!(solver, Q; tfinal=1e9, nmax=4,
         callback=Callback(far, (_, _) -> true))
    @test solver.step == 4 && !far.done
    # A signal's flag fires it on the next check whatever the deadline, and the
    # effect's `true` ends the run there.
    solver, Q = mkrun()
    raise = (s, _) -> (s.step == 2 && CL._wall_signal_callback(C_NULL, Cint(10)); nothing)
    run!(solver, Q; tfinal=1e9, nmax=6,
         callback=(raise, Callback(far, (_, _) -> true)))
    CL.WALL_SIGNAL[] = 0
    @test solver.step == 2 && far.fired_step == 2
    # A real signal, in a process that leaves signals to the program, which
    # the test runner does not.
    if !Sys.iswindows()
        script = """
            using CompactLES
            trigger = BeforeWallLimit(signals=(:USR1,))
            ccall(:kill, Cint, (Cint, Cint), getpid(), $(CL._signal_number(:USR1)))
            for _ in 1:100
                Base.process_events()
                CompactLES.WALL_SIGNAL[] == 0 || break
                sleep(0.01)
            end
            print(trigger.signals == [$(CL._signal_number(:USR1))], " ",
                  CompactLES.WALL_SIGNAL[])
            """
        cmd = `$(Base.julia_cmd()) --handle-signals=no -t 1 --startup-file=no
               --project=$(Base.active_project()) -e $script`
        @test read(cmd, String) == "true $(CL._signal_number(:USR1))"
    end
    # A rollback to before the step it fired on re-arms it; one after does not.
    CL.rewind!(far, 0.0, 3)
    @test far.done
    CL.rewind!(far, 0.0, 1)
    @test !far.done && far.fired_step == -1

    @test_throws ArgumentError BeforeWallLimit(deadline=1.0, walltime=1.0)
    @test_throws ArgumentError BeforeWallLimit(margin=-1)
    @test_throws ArgumentError BeforeWallLimit(signals=(:NOPE,))
    @test_throws ArgumentError BeforeWallLimit(check_every=0)

    # The scheduler's remaining time, in each of the forms Slurm prints.
    @test CL._slurm_seconds("1-02:03:04") == 93784.0
    @test CL._slurm_seconds("2-03") == 2 * 86400.0 + 3 * 3600
    @test CL._slurm_seconds("02:03:04\n") == 7384.0
    @test CL._slurm_seconds("29:58") == 1798.0
    @test CL._slurm_seconds("45") == 2700.0
    @test CL._slurm_seconds("UNLIMITED") === nothing
    @test CL._slurm_seconds(nothing) === nothing
    @test CL._job_deadline(Dict("SLURM_JOB_END_TIME" => "1900000000")) ==
          (1.9e9, :slurm_env)
    @test CL._job_deadline(Dict{String,String}()) == (Inf, :none)
    @test CL._command_output(`$(Base.julia_cmd()) --startup-file=no -e "print(42)"`) ==
          "42"
    @test CL._command_output(`a-command-that-does-not-exist`) === nothing
end

@testset "EveryTime: evenly spaced instants, restart anchoring, soft landing" begin
    wall3 = ((SlipWallBC(), SlipWallBC()), (PeriodicBC(), PeriodicBC()),
             (PeriodicBC(), PeriodicBC()))
    mkrun() = begin
        # cfl 0.244 keeps the step this test was written at (0.4 under the
        # summed acoustic rate): the sliver below needs dt below the 0.01 interval.
        solver = build_solver(
            Problem(domain=((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)), bcs=wall3, ic=nothing),
            Numerics(n_global=(16, 12, 12), art=ArtificialProperties(enabled=false),
                     cfl=0.244))
        Q = allocate_state(solver)
        initialize!(solver, Q, (x, y, z) -> Prim(u=(0.2, 0, 0),
                                                 p=1 + 0.1exp(-40(x - 0.5)^2), rho=1.0))
        solver, Q
    end

    # Every instant is landed on exactly, and none is skipped; the instant at
    # the initial time fires before the first step.
    solver, Q = mkrun()
    hits = Float64[]
    run!(solver, Q; tfinal=0.05,
         callback=Callback(EveryTime(0.01), (s, _) -> (push!(hits, s.t); nothing)))
    @test hits ≈ collect(0.0:0.01:0.05) rtol = 1e-14
    @test solver.step > 0 && count(iszero, hits) == 1

    # AtTime fires at a listed initial time too, and only once.
    solver, Q = mkrun()
    hits = Float64[]
    run!(solver, Q; tfinal=0.02,
         callback=Callback(AtTime([0.0, 0.01]), (s, _) -> (push!(hits, s.t); nothing)))
    @test hits ≈ [0.0, 0.01] rtol = 1e-14

    # Anchored to solver.t on first use, not to zero: a restarted run picks up at
    # the next instant rather than replaying the schedule.
    solver, Q = mkrun()
    solver.t = 0.055
    hits = Float64[]
    run!(solver, Q; tfinal=0.08,
         callback=Callback(EveryTime(0.01), (s, _) -> (push!(hits, s.t); nothing)))
    @test hits ≈ [0.06, 0.07, 0.08] rtol = 1e-14

    # `start` offsets the schedule, and instants before it are not visited.
    solver, Q = mkrun()
    hits = Float64[]
    run!(solver, Q; tfinal=0.05,
         callback=Callback(EveryTime(0.02; start=0.015),
                           (s, _) -> (push!(hits, s.t); nothing)))
    @test hits ≈ [0.015, 0.035] rtol = 1e-14

    @test_throws ArgumentError EveryTime(0.0)
    @test_throws ArgumentError StepControl(landing_steps=0)

    # The soft landing is the purpose of `landing_steps`. Under a hard clip the
    # same schedule leaves one very small step before every instant, so compare
    # the spread of accepted steps.
    spread(control) = begin
        s, q = mkrun()
        dts = Float64[]
        run!(s, q; tfinal=0.05, control=control,
             callback=(Callback(EveryTime(0.01), (_, _) -> nothing),
                       (x, _) -> push!(dts, x.dt_prev)))
        minimum(dts) / maximum(dts)
    end
    @test spread(StepControl(landing_steps=1)) < 0.35     # sliver: 2.2e-3 vs 7.8e-3
    @test spread(StepControl()) > 0.99                    # split evenly instead

    # An instant BEYOND tfinal must not be landed on. Instants are computed as
    # `start + n*interval`, so the third one here is `3 * 0.05`, which is the next
    # float above the 0.15 literal — the schedule ends one ULP past the endpoint.
    # The soft landing used to aim at it anyway, and since `dt` was already
    # clipped to `tfinal - t` the gap stayed a shade above `dt` and `ceil(gap/dt)`
    # stayed at 2, halving the step against a target the run never reaches. The
    # run still finished and every instant still fired, so the only symptom was
    # step count: 48 steps against 9 in the case this was found in.
    #
    # Measured against the same run with no trigger rather than a fixed number,
    # since the step count is a property of the CFL here and not of the landing.
    @test 3 * 0.05 > 0.15                          # the ULP the case rests on
    nsteps(cb) = begin
        s, q = mkrun()
        run!(s, q; tfinal=0.15, callback=cb)
        s.step
    end
    bare = nsteps(nothing)
    @test nsteps(Callback(EveryTime(0.05), (_, _) -> nothing)) <= bare + 4
end

@testset "rewind!: a rollback re-arms the instants it abandoned" begin
    # Unit behaviour first. AtTime rewinds to the first instant ahead of the
    # restored time, EveryTime re-anchors, and a step-counting trigger has no
    # schedule to move and takes the no-op default.
    trigger = AtTime([0.1, 0.2, 0.3])
    trigger.next = 3
    CL.rewind!(trigger, 0.15, 7)
    @test trigger.next == 2
    CL.rewind!(trigger, 0.35, 9)
    @test trigger.next == 4                              # all behind: none left
    CL.rewind!(trigger, 0.0, 0)
    @test trigger.next == 1

    et = EveryTime(0.01)
    solver = build_solver(
        Problem(domain=((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)), bcs=per3, ic=nothing),
        Numerics(n_global=(16, 12, 12), art=ArtificialProperties(enabled=false)))
    solver.t = 0.045
    @test CL.next_time(et, solver) ≈ 0.05
    CL.rewind!(et, 0.021, 3)
    solver.t = 0.021
    @test CL.next_time(et, solver) ≈ 0.03
    @test CL.rewind!(EveryStep(4), 0.1, 3) === nothing

    # Then the wiring, against a rollback placed where it can be reasoned about.
    # Physical failures occur in the startup transient, before a savepoint has
    # advanced past step 0, which demonstrates nothing here. Corrupting the
    # state once at a chosen time and letting the positivity check find it
    # places the rollback across instants that have already fired.
    walls = ((SlipWallBC(), SlipWallBC()), (PeriodicBC(), PeriodicBC()),
             (PeriodicBC(), PeriodicBC()))
    s2 = build_solver(
        Problem(domain=((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)), bcs=walls, ic=nothing),
        Numerics(n_global=(16, 12, 12), art=ArtificialProperties(enabled=false), cfl=0.4))
    Q2 = allocate_state(s2)
    initialize!(s2, Q2, (x, y, z) -> Prim(u=(0.2, 0, 0),
                                          p=1 + 0.1exp(-40(x - 0.5)^2), rho=1.0))
    instants = collect(0.005:0.005:0.06)
    seen = Float64[]
    # A rollback re-arms a WhenState that fired after the savepoint, so the
    # one-shot fault is held in the condition, not in the trigger.
    spoiled = Ref(false)
    spoil = Callback(WhenState((x, _) -> x.t >= 0.037 && !spoiled[]),
                     (x, q) -> (spoiled[] = true; q[padded_index(x, 3, 3, 3), 1] = -1.0;
                                nothing))
    run!(s2, Q2; tfinal=0.06, nmax=5000,
         control=StepControl(retries=2, savepoint_interval=2),
         callback=(Callback(AtTime(instants), (x, _) -> (push!(seen, x.t); nothing)),
                   spoil))
    @test s2.cfl < 0.4                                    # it did roll back
    @test s2.t ≈ 0.06 rtol = 1e-12                        # and then finished
    # Re-arming has a positive signature: an instant crossed on the abandoned
    # trajectory is visited a second time on the replacement trajectory. Without
    # the rewind it is visited once and never revisited, and the `unique` check
    # below would fall short.
    @test length(seen) > length(instants)
    @test sort(unique(round.(seen, digits=10))) ≈ instants rtol = 1e-9
end

@testset "state queries and refresh_primitives! in a callback" begin
    eos = IdealMixture([IdealSpecies{Float64}("a", 1.0, 1.4),
                        IdealSpecies{Float64}("b", 2.0, 1.6)])
    solver = build_solver(
        Problem(domain=((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)), bcs=per3, eos=eos, ic=nothing),
        Numerics(n_global=(16, 12, 12), art=ArtificialProperties(enabled=false), cfl=0.4))
    Q = allocate_state(solver)
    initialize!(solver, Q, (x, y, z) -> Prim(Y=(0.3, 0.7), u=(0.2, -0.1, 0.05),
                                             p=1 + 0.1exp(-40(x - 0.5)^2), rho=1.0))
    refresh_primitives!(solver, Q)

    # The layout-free spellings agree with the primitives they stand in for, and
    # do not assume a species count.
    I = padded_index(solver, 5, 4, 3)
    @test mixture_density(solver, Q, I) ≈ solver.rho[I] rtol = 1e-14
    @test all(velocity(solver, Q, I) .≈ (solver.u[I], solver.v[I], solver.w[I]))
    @test mass_fraction(solver, Q, I, 2) ≈ solver.Y[2][I] rtol = 1e-14
    @test total_energy(solver, Q, I) == Q[I, solver.equations.i_energy]
    @test mixture_density(solver, Q, I) ≈ Q[I, 1] + Q[I, 2] rtol = 1e-14

    # In serial every rank holds every edge, so the `nothing` branch is
    # exercised only under MPI.
    @test boundary_plane(solver, 2, 1) == CL.wallplane(solver.decomp, 2, 1)

    # interior_index is the documented conversion between the two index bases:
    # boundary_plane yields padded indices, xcoord takes interior ones.
    @test interior_index(solver, I) == (5, 4, 3)
    @test padded_index(solver, interior_index(solver, I)...) == I
    for J in boundary_plane(solver, 1, 2)
        i, j, k = interior_index(solver, J)
        @test padded_index(solver, i, j, k) == J
        @test i == solver.decomp.n_local[1]              # the high-x face
        @test xcoord(solver, 1, i) ==
              global_xcoord(solver, 1, solver.decomp.offset[1] + i)
    end

    # The contract stated by refresh_primitives!: inside a callback the
    # primitives belong to the fifth RK stage's INPUT state, so they predate the
    # final stage update and the filter pass. The assertion is that refreshing
    # changes the answer, and that the refreshed value is the one Q holds.
    stale = Ref(0.0); fresh = Ref(0.0); fromQ = Ref(0.0)
    run!(solver, Q; tfinal=1e9, nmax=6, callback=(s, q) -> begin
        s.step == 6 || return
        stale[] = s.rho[I]
        fromQ[] = mixture_density(s, q, I)
        refresh_primitives!(s, q)
        fresh[] = s.rho[I]
    end)
    @test isapprox(fresh[], fromQ[]; rtol=1e-14)         # refreshed agrees with Q
    @test !isapprox(stale[], fromQ[]; rtol=1e-12)        # unrefreshed does not
end

@testset "phase change: opening the upstream face lets the reflected wave leave" begin
    # A shock/interface interaction in a translating frame, with the upstream
    # boundary changing from inflow to outflow, by a phase change, when the
    # interface-reflected wave reaches it. A finiteness check says nothing
    # about the question this case asks: whether the opened boundary lets the
    # wave out.
    #
    #        shocked driven | driven |    heavy
    #      ---------------->|        |
    #        (region 2)     ^shock   ^interface
    #      0               0.20     0.40              1
    #
    # The state initialized upstream of the shock is the SHOCKED driven gas —
    # region 2 of the normal-shock relations, not a driver — so the incident
    # shock is already formed and the run starts at the physics of interest.
    # Sustaining that state makes the upstream boundary an inflow
    # (u2 + U > 0 into the domain) for as long as the incident shock is being
    # driven. NSCBCInflowBC is the condition that holds it: it replaces every
    # incoming characteristic with a relaxation toward (u, T_ion, Y), the
    # appropriate condition while region 2 is uniform there.
    #
    # It becomes inappropriate when the interface-reflected shock arrives.
    # Behind that shock the state is no longer region 2, so an inflow condition
    # still relaxing toward region-2 targets is over-constraining the boundary,
    # and the mismatch radiates back inward onto the interface. The remedy is
    # to stop imposing and start absorbing: the second phase takes
    # NSCBCOutflowBC there.
    #
    # The frame translation U is chosen so the SHOCKED interface is nearly at
    # rest (it drifts ~0.04 once the shock has passed it), the standard frame for a
    # shock/interface calculation: it keeps the interface in the box, and it
    # gives both faces a mean flow so nothing is being tested at the degenerate
    # u = 0 point where NSCBC's Mach-dependent terms drop out.
    γ, RL, RH, MS, U = 1.4, 1.0, 0.25, 1.5, -0.5
    N, XS, XI, TEND = 160, 0.20, 0.40, 1.0
    Hx = 1 / (N - 1)

    # Region 1 (unshocked driven gas) and the normal-shock relations for the
    # region 2 that sits behind a Mach MS shock running into it.
    p1 = ρ1 = T1 = 1.0
    c1 = sqrt(γ)
    p2 = p1 * (2γ * MS^2 - (γ - 1)) / (γ + 1)
    ρ2 = ρ1 * ((γ + 1) * MS^2) / ((γ - 1) * MS^2 + 2)
    u2 = c1 * 2 / (γ + 1) * (MS^2 - 1) / MS
    T2 = p2 / (ρ2 * RL)
    eos2 = IdealMixture([IdealSpecies{Float64}("light", RL, γ),
                         IdealSpecies{Float64}("heavy", RH, γ)])

    # `xlo < 0` extends the domain upstream at the same spacing, which is how
    # the reference below is built. The heavy gas is at the driven gas's p and
    # T_ion, so RH sets the density ratio: Atwood 0.6 here.
    build(xbc; xlo=0.0) = begin
        n = round(Int, (1.0 - xlo) / Hx) + 1
        δ = 2Hx
        setup(Problem(eos=eos2, transport=ConstantTransport(mu0=0.0),
                      domain=((xlo, xlo + (n - 1) * Hx), (0.0, Hx), (0.0, Hx)),
                      bcs=(xbc, per3[2], per3[3]),
                      ic=(x, y, z) -> begin
                          s = tanh_blend(x, XS, δ)     # 0 shocked, 1 unshocked
                          θ = tanh_blend(x, XI, δ)     # 0 light,   1 heavy
                          Prim(Y=(1 - θ, θ), u=((1 - s) * u2 + U, 0.0, 0.0),
                               p=(1 - s) * p2 + s * p1,
                               T_ion=(1 - s) * T2 + s * T1)
                      end),
              Numerics(n_global=(n, 1, 1), art=ArtificialProperties(enabled=true),
                       cfl=0.4,
                       # A shocked binary interface ends beyond the
                       # mass-fraction band, five of 160 points here, as the
                       # shock/SF6 validation case does. The boundary condition
                       # is what this tests, so the state is reported rather
                       # than rejected.
                       control=StepControl(validity=:permissive)))
    end

    inflow() = NSCBCInflowBC(u=(u2 + U, 0.0, 0.0), T_ion=T2, Y=[1.0, 0.0])
    # sigma small on purpose. The pressure this boundary will face once the
    # reflected shock has passed is the post-reflected-shock pressure (~3.23),
    # which is the answer to the interaction and not something the test may
    # assume; relaxing hard toward `pinf` would impose a pressure known to be
    # wrong and drag the driven section with it. A weak relaxation is the
    # honest reading of "let the wave out and do not hold a pressure I do not
    # know" — measured: sigma 0.25 costs a factor 3.5 in the agreement below.
    outflow() = NSCBCOutflowBC(pinf=p2, sigma=0.05)
    # Nothing reaches the downstream end within TEND (the transmitted shock
    # gets to x ~ 0.8), so a Dirichlet on the unshocked heavy state is exact
    # there and keeps this test about the upstream boundary alone.
    downstream() = DirichletBC((x, y, z, t) -> Prim(Y=(0.0, 1.0),
                                                    u=(U, 0.0, 0.0), p=p1,
                                                    T_ion=T1))

    # "The reflected shock has reached the upstream plane." A rank-local verdict
    # on the plane this rank owns, left to WhenState to reduce, as the
    # WhenState docstring describes: a run that ends for a phase change must
    # never be ended by an unreduced rank-local test.
    arrived(solver, Q) = begin
        plane = CL.wallplane(solver.decomp, 1, 1)
        plane === nothing && return false
        any(I -> solver.p[I] > 1.05 * p2, plane)
    end

    xline(s, Q) = begin
        CL.exchange_state!(Q, s.decomp)
        CL.primitives!(s, Q)
        nx = s.decomp.n_local[1]
        ([xcoord(s, 1, i) for i in 1:nx],
         [s.p[padded_index(s, i, 1, 1)] for i in 1:nx],
         [s.u[padded_index(s, i, 1, 1)] for i in 1:nx],
         [s.Y[2][padded_index(s, i, 1, 1)] for i in 1:nx])
    end
    # Interface position, interpolated across the cell where Y_heavy crosses
    # 1/2 — a sub-cell measure, since a re-shock moves the interface by a
    # fraction of a cell over this run.
    ipos(x, Y) = begin
        i = findfirst(>=(0.5), Y)
        x[i-1] + (0.5 - Y[i-1]) / (Y[i] - Y[i-1]) * (x[i] - x[i-1])
    end

    # (1) Opened on arrival: the first phase ends on the step the wave
    #     arrives, and the second continues from it under the outflow.
    s_sw, Q_sw = build((inflow(), downstream()))
    run!(s_sw, Q_sw; tfinal=TEND, nmax=100_000,
         callback=Callback(WhenState(arrived), Returns(true)))
    fired_step = Ref(s_sw.step)
    Q_fired = Ref{Any}(copy(Q_sw))
    @test 0 < fired_step[] && s_sw.t < TEND    # the first phase ended early
    s_sw, Q_sw = setup(s_sw, Q_sw; bcs=((outflow(), downstream()), per3[2], per3[3]))
    run!(s_sw, Q_sw; tfinal=TEND, nmax=100_000)
    @test s_sw.t == TEND

    # (2) Inflow held throughout: the control, and the same run bit-for-bit up
    #     to the change. WhenState does not clip dt, so the step sequences agree
    #     and this is an equality, not an approximation.
    s_h, Q_h = build((inflow(), downstream()))
    run!(s_h, Q_h; tfinal=TEND, nmax=fired_step[])
    @test s_h.step == fired_step[]
    @test Q_h == Q_fired[]
    run!(s_h, Q_h; tfinal=TEND, nmax=100_000)

    # (3) The reference: the same problem with the upstream boundary one domain
    #     length further away, so the reflected shock never reaches it and no
    #     boundary treatment is relevant. This makes the comparison a
    #     measurement rather than a difference of two guesses — a non-reflecting
    #     condition is only ever tested against a domain big enough not to need
    #     one. Same spacing, so the grids coincide on x >= 0 to round-off.
    s_r, Q_r = build((inflow(), downstream()); xlo=-1.0)
    run!(s_r, Q_r; tfinal=TEND, nmax=100_000)

    xr, pr, ur, Yr = xline(s_r, Q_r)
    xh, ph, uh, Yh = xline(s_h, Q_h)
    xs, ps, us, Ys = xline(s_sw, Q_sw)
    keep = [i for i in eachindex(xr) if xr[i] >= -1e-9]
    @test maximum(abs, xr[keep] .- xh) < 1e-12

    win = [i for i in eachindex(xh) if 0.05 <= xh[i] <= 0.95]
    errp(p) = maximum(abs.(p[win] .- pr[keep][win]))
    erru(u) = maximum(abs.(u[win] .- ur[keep][win]))
    ref_i = ipos(xr, Yr)

    # The measurement. Held, the inflow condition keeps imposing region 2 after
    # the reflected shock has arrived and the error radiates back over the
    # interface; opened, the wave leaves. Measured against the reference:
    # pressure 0.104 held vs 0.011 opened, velocity 0.0279 vs 0.0031,
    # interface displacement 0.0024 vs 0.00016 (about a fortieth of a cell).
    # Guards are ratios plus loose absolute bounds: this is a nonlinear run
    # through the artificial-property sensor, so it is not bit-reproducible and
    # only the order of magnitude is being asserted.
    @test errp(ps) < 0.04
    @test errp(ph) > 0.08
    @test errp(ph) > 3 * errp(ps)
    @test erru(us) < 0.008
    @test erru(uh) > 0.02
    @test erru(uh) > 3 * erru(us)
    @test abs(ipos(xs, Ys) - ref_i) < 0.0015
    @test abs(ipos(xh, Yh) - ref_i) > 2.5 * abs(ipos(xs, Ys) - ref_i)
end

@testset "checkpoint round trip" begin
    solver = mkslv(n_global=(12, 12, 12))
    Q = allocate_state(solver)
    initialize!(solver, Q, (x, y, z) -> Prim(u=(sin(x), 0, 0), p=1 + 0.1cos(y), rho=1.0))
    solver.t = 0.37; solver.step = 42
    save_checkpoint(solver, Q, "test_ckpt")
    Q2 = allocate_state(solver); solver.t = 0.0; solver.step = 0
    load_checkpoint!(solver, Q2, "test_ckpt")
    @test solver.t == 0.37 && solver.step == 42
    @test all(Q2[padded_index(solver, i, j, k), c] == Q[padded_index(solver, i, j, k), c]
              for c in 1:5, i in 1:12, j in 1:12, k in 1:12)
    foreach(rm, filter(startswith("test_ckpt"), readdir()))
end

@testset "checkpoint header rejects a solver it does not describe" begin
    # Everything the payload's interpretation depends on is in the header. Each
    # solver below reads the block cleanly if its own field is not checked: the
    # extents agree in every case and the state means something else. The
    # expected message is asserted to ensure each case exercises its intended
    # check, not an earlier one.
    mk(; eos=IdealSpecies("gas"; R=1.0, gamma=1.4), L_domain=(2π, 2π, 2π),
         origin=(0.0, 0.0, 0.0), bcs=per3, metric=CartesianMetric()) =
        build_solver(Problem(domain=ntuple(d -> (origin[d], origin[d] + L_domain[d]), 3),
                             eos=eos, bcs=bcs, metric=metric, ic=nothing),
                     Numerics(n_global=(12, 12, 12), art=ArtificialProperties(enabled=false)))
    eos2 = IdealMixture([IdealSpecies{Float64}("a", 1.0, 1.4),
                         IdealSpecies{Float64}("b", 2.0, 1.6)])
    base = mk(eos=eos2)
    Qb = allocate_state(base)
    initialize!(base, Qb, (x, y, z) -> Prim(Y=(0.3, 0.7), p=1.0, rho=1.0))
    save_checkpoint(base, Qb, "test_ckpt_header")
    reject(solver, msg) =
        @test_throws msg load_checkpoint!(solver, allocate_state(solver),
                                          "test_ckpt_header")

    # A different species set of the same size: same conserved count, same grid,
    # same metric. Only the component names separate them.
    reject(mk(eos=IdealMixture([IdealSpecies{Float64}("c", 1.0, 1.4),
                                IdealSpecies{Float64}("d", 2.0, 1.6)])),
           "conserved component mismatch")
    # A different species count, which also moves the conserved count.
    reject(mk(eos=IdealMixture([IdealSpecies{Float64}("a", 1.0, 1.4)])),
           "conserved layout mismatch")
    # The same grid dimensions over a longer domain, which the extents cannot
    # see and the coordinates can.
    reject(mk(eos=eos2, L_domain=(4π, 2π, 2π)), "grid coordinate mismatch")
    # The same grid shifted, likewise invisible to every other field.
    reject(mk(eos=eos2, origin=(0.5, 0.0, 0.0)), "grid coordinate mismatch")
    # A different metric on the same grid dimensions: the block is the same
    # shape and every momentum component means something else.
    reject(mk(eos=eos2, bcs=((SlipWallBC(), SlipWallBC()), per3[2], per3[3]),
              origin=(0.5, 0.0, 0.0), metric=CylindricalMetric()),
           "metric mismatch")

    # A file without the checkpoint magic is refused by it.
    open("test_ckpt_header_old.r0000.ckpt", "w") do io
        write(io, UInt64(0x434c4553_434b5054))
    end
    @test_throws "not a CompactLES checkpoint" load_checkpoint!(
        base, Qb, "test_ckpt_header_old")

    # A stretched dimension against the uniform grid it was written on. The two
    # agree on extent, origin, species and metric and differ only point by
    # point, which is why the coordinates are stored rather than the extent. A
    # stretched dimension must be non-periodic, so this pair needs a checkpoint
    # of its own, and it is paired with a reload onto the grid that wrote it so
    # that the check is shown to discriminate rather than to refuse everything.
    mkw(st) = build_solver(
        Problem(domain=((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)),
                bcs=((SlipWallBC(), SlipWallBC()), per3[2], per3[3]), ic=nothing),
        Numerics(n_global=(12, 12, 12), stretch=(st, nothing, nothing),
                 art=ArtificialProperties(enabled=false)))
    uniform = mkw(nothing)
    Qu = allocate_state(uniform)
    initialize!(uniform, Qu, (x, y, z) -> Prim(p=1.0, rho=1.0))
    uniform.step = 7
    save_checkpoint(uniform, Qu, "test_ckpt_uniform")
    stretched = mkw(sine_cluster(0.0, 1.0, 0.5, 0.3))
    @test_throws "grid coordinate mismatch" load_checkpoint!(
        stretched, allocate_state(stretched), "test_ckpt_uniform")
    reload = mkw(nothing)
    load_checkpoint!(reload, allocate_state(reload), "test_ckpt_uniform")
    @test reload.step == 7

    foreach(rm, filter(startswith("test_ckpt"), readdir()))
end

@testset "smoke: three RK steps of every headline configuration" begin
    for build in (
        () -> setup(Problem(domain=((0.0, 2π), (0.0, 2π), (0.0, 2π)), bcs=per3,
                            ic=(x, y, z) -> Prim(u=(0.1sin(x), 0, 0), p=1.0, rho=1.0)),
                    Numerics(n_global=(16, 16, 16))),
        () -> setup(Problem(domain=((0.0, 2π), (0.0, 2π), (0.0, 2π)), bcs=per3,
                            ic=(x, y, z) -> Prim(u=(0.1sin(x), 0, 0), p=1.0, rho=1.0)),
                    Numerics(n_global=(16, 16, 16), deriv=lele_d1_10())),
        () -> setup(Problem(domain=((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)),
                            metric=CylindricalMetric(),
                            bcs=((AxisBC(), SlipWallBC()), per3[2], per3[3]),
                            ic=(r, θ, z) -> Prim(u=(0, 0, 0), p=1 + exp(-40(r - 0.4)^2), rho=1.0)),
                    Numerics(n_global=(48, 1, 1))),
    )
        solver, Q = build()
        run!(solver, Q; tfinal=1e9, nmax=3)
        bad = any(!isfinite(Q[padded_index(solver, i, j, k), c])
                  for c in 1:solver.equations.n_cons, i in 1:solver.decomp.n_local[1],
                      j in 1:solver.decomp.n_local[2], k in 1:solver.decomp.n_local[3])
        @test !bad
    end
end

@testset "Sod shock tube: profile matches exact Riemann solution" begin
    # Classic single-gas Sod (γ = 1.4) on [0,1], diaphragm at x = 0.5:
    #   left (ρ,u,p) = (1, 0, 1), right (ρ,u,p) = (0.125, 0, 0.1).
    # Integrate to t = 0.2 in 1-D (collapsed transverse dims) with artificial
    # fluid properties capturing the shock, and compare the ρ/u/p line profile
    # against the analytic Riemann solution sampled at each node. Errors are
    # L1 over the profile — shock/contact smearing over a few cells is expected,
    # so the tolerance is looser than the smooth-operator tests but still tight
    # enough that a wrong wave speed, plateau, or star state fails it.
    γ = 1.4
    ρL, uL, pL = 1.0, 0.0, 1.0
    ρR, uR, pR = 0.125, 0.0, 0.1
    x0 = 0.5; tfin = 0.2
    Nx = 400; Lx = 1.0; hx = Lx / (Nx - 1); δ = 2hx
    prob = Problem(name="Sod", eos=IdealSpecies("gas"; gamma=γ, R=1.0),
                   transport=ConstantTransport(mu0=0.0),
                   domain=((0.0, Lx), (0.0, hx), (0.0, hx)),
                   bcs=((SlipWallBC(), SlipWallBC()), per3[2], per3[3]),
                   ic=(x, y, z) -> begin
                       θ = tanh_blend(x, x0, δ)
                       Prim(rho=(1 - θ) * ρL + θ * ρR,
                            u=((1 - θ) * uL + θ * uR, 0.0, 0.0),
                            p=(1 - θ) * pL + θ * pR)
                   end)
    solver, Q = setup(prob,
                      Numerics(n_global=(Nx, 1, 1), art=ArtificialProperties(enabled=true),
                                cfl=0.4))
    run!(solver, Q; tfinal=tfin, nmax=100_000)
    CL.exchange_state!(Q, solver.decomp)
    CL.primitives!(solver, Q)

    pstar, ustar, cL, cR = exact_riemann_star(ρL, uL, pL, ρR, uR, pR, γ)
    # Star state is the canonical Sod result (Toro): p* ≈ 0.30313, u* ≈ 0.92745.
    @test pstar ≈ 0.30313 atol = 1e-4
    @test ustar ≈ 0.92745 atol = 1e-4

    nx = solver.decomp.n_local[1]
    eρ = eu = ep = 0.0
    for i in 1:nx
        I = padded_index(solver, i, 1, 1); x = xcoord(solver, 1, i)
        r, u, p = exact_riemann_sample((x - x0) / tfin, ρL, uL, pL, ρR, uR, pR,
                                       γ, pstar, ustar, cL, cR)
        eρ += abs(solver.rho[I] - r); eu += abs(solver.u[I] - u); ep += abs(solver.p[I] - p)
    end
    eρ /= nx; eu /= nx; ep /= nx
    # Guards at roughly twice the measured error, as in test/validation.jl. The
    # run integrates a nonlinear sensor for hundreds of steps, so the fourth
    # significant figure moves under an arithmetic reassociation; a moved third
    # digit is real.
    @test eρ < 6.0e-3    # measured 2.901e-3
    @test eu < 9.5e-3    # measured 4.622e-3
    @test ep < 5.0e-3    # measured 2.458e-3
end
