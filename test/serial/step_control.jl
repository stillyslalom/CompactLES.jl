# The serial suite's tests of the step size, step control, the positivity failsafe
# and state validity. Included by test/serial_suite.jl, which loads the package and
# defines the imports and helpers used here; not runnable on its own.

@testset "dt_report agrees with compute_dt and names the limiter" begin
    solver = mkslv(n_global=(16, 16, 16), transport=ConstantTransport(mu0=1e-3))
    Q = allocate_state(solver)
    initialize!(solver, Q, (x, y, z) -> Prim(u=(0.2sin(x), 0, 0), p=1 + 0.1cos(y), rho=1.0))
    r = dt_report(solver, Q)
    @test r.dt ≈ compute_dt(solver, Q) rtol = 1e-12
    @test r.kind in (:acoustic, :diffusive, :curvature)
    @test r.dim in 1:3
    @test all(1 .<= r.index .<= 16)
end

@testset "curvature_rate: collapsed angular dims restrict dt" begin
    # The spherical branch of curvature_rate had no coverage at all, and the
    # whole point of the term is that a COLLAPSED angular dimension carries a
    # stiff geometric source the advective CFL loop never sees. Swirl must
    # therefore shorten dt even though nothing varies in θ or φ.
    mk(metric, uang) = begin
        solver = build_solver(
            Problem(domain=((0.5, 0.5 + 1.0), (π / 2, π / 2 + 1.0), (0.0, 0.0 + 1.0)),
                    metric=metric, bcs=((SlipWallBC(), SlipWallBC()), per3[2], per3[3]),
                    ic=nothing),
            Numerics(n_global=(64, 1, 1), art=ArtificialProperties(enabled=false)))
        Q = allocate_state(solver)
        initialize!(solver, Q, (r, θ, φ) -> Prim(u=(0.0, uang, uang), p=1.0, rho=1.0))
        solver, Q
    end
    for metric in (CylindricalMetric(), SphericalMetric())
        s0, Q0 = mk(metric, 0.0)
        s1, Q1 = mk(metric, 2.0)
        @test compute_dt(s1, Q1) < compute_dt(s0, Q0)
        @test CL.curvature_rate(s1, metric, padded_index(s1, 5, 1, 1), (0.0, 2.0, 2.0)) > 0
        @test CL.curvature_rate(s0, metric, padded_index(s0, 5, 1, 1), (0.0, 0.0, 0.0)) == 0
    end
    # Cartesian has no curvature term at all
    sc = mkslv(n_global=(16, 16, 16))
    @test CL.curvature_rate(sc, CartesianMetric(), padded_index(sc, 2, 2, 2),
                            (1.0, 1.0, 1.0)) == 0
end

@testset "StepControl: floors, positivity, and the default no-op" begin
    # The floor logic is pure, so it is checked directly rather than by
    # constructing a run that fails in each of five ways.
    c = StepControl()
    ok(dt, ρ; seen=1.0, ctl=c) = CL.check_step(ctl, dt, ρ, seen, 1, 0.0, 0.5)
    @test ok(1e-3, 1.0) === nothing
    @test ok(NaN, 1.0).reason === :nonfinite
    @test ok(Inf, 1.0).reason === :nonfinite
    @test ok(1e-3, -1e-9).reason === :negative_density
    @test ok(1e-3, 0.0).reason === :negative_density
    @test ok(1e-50, 1.0).reason === :planck            # below the Planck failsafe
    @test ok(1e-40, 1.0).reason === :dt_collapse       # above it, but collapsed
    # The Planck floor is unconditional: it fires even with every user floor off.
    bare = StepControl(dt_min=0.0, dt_min_ratio=0.0)
    @test ok(1e-50, 1.0; ctl=bare).reason === :planck
    @test ok(1e-40, 1.0; ctl=bare) === nothing         # relative floor disabled
    # An explicit absolute floor.
    @test ok(1e-9, 1.0; ctl=StepControl(dt_min=1e-6)).reason === :dt_min
    # Ordering: a non-finite dt is reported as such, not as a floor breach.
    @test ok(NaN, -1.0).reason === :nonfinite
    # Message carries the state that localizes the failure.
    e = ok(1e-50, 1.0)
    msg = sprint(showerror, e)
    @test occursin("SolverFailure(:planck)", msg)
    @test occursin("StepControl(retries", msg)

    # With prediction off (the default) the chosen step is exactly compute_dt,
    # so adding all of this changed nothing for a healthy run.
    solver = mkslv(n_global=(16, 16, 16))
    Q = allocate_state(solver)
    initialize!(solver, Q, (x, y, z) -> Prim(u=(0.2sin(x), 0, 0), p=1.0, rho=1.0))
    rate, ρmin = max_rate(solver, Q)
    @test ρmin ≈ 1.0 atol = 1e-12
    @test CL.predicted_dt(solver, StepControl(), rate) == compute_dt(solver, Q)
    # Prediction only ever shortens the step, and only when the rate is rising.
    solver.rate_prev = 0.5 * rate
    @test CL.predicted_dt(solver, StepControl(predict=1.0), rate) < compute_dt(solver, Q)
    solver.rate_prev = 2.0 * rate              # falling rate: no extrapolation
    @test CL.predicted_dt(solver, StepControl(predict=1.0), rate) == compute_dt(solver, Q)
    # Growth capping is relative to the previous accepted step.
    solver.rate_prev = 0.0
    solver.dt_prev = 1e-9
    @test CL.predicted_dt(solver, StepControl(max_growth=1.5), rate) ≈ 1.5e-9 rtol = 1e-14
end

@testset "positivity failsafe: floors, scope, and what each conserves" begin
    @test_throws ArgumentError StepControl(floor_ratio=1.0)
    @test_throws ArgumentError StepControl(floor_ratio=-1e-9)
    @test_throws ArgumentError StepControl(floor_scope=:everything)
    @test StepControl().floor_ratio == 0.0                 # off by default
    @test StepControl().floor_scope === :representable

    eos = IdealMixture([IdealSpecies{Float64}("a", 1.0, 1.4),
                        IdealSpecies{Float64}("b", 2.0, 1.6)])
    solver = mkslv(n_global=(12, 12, 12), eos=eos)
    Q = allocate_state(solver)
    initialize!(solver, Q, (x, y, z) -> Prim(rho=2.0, u=(0.5, 0.0, 0.0), p=1.0,
                                             Y=(0.25, 0.75)))
    m1, m2, m3 = solver.equations.i_mom
    ie = solver.equations.i_energy
    # The reference state is uniform, so the derived floors are exactly the
    # ratio times the values it carries: e = p / ((γ_m − 1) ρ) with the
    # mass-averaged mixture constants.
    ρ0 = 2.0
    e0 = let Rm = 0.25 * 1.0 + 0.75 * 2.0, cvm = 0.25 * 1.0 / 0.4 + 0.75 * 2.0 / 0.6
        1.0 / (ρ0 * Rm) * cvm
    end
    rho_floor, e_floor = CL.positivity_floors(solver, Q, StepControl(floor_ratio=1e-6))
    @test rho_floor ≈ 1e-6 * ρ0 rtol = 1e-12
    @test e_floor ≈ 1e-6 * e0 rtol = 1e-12
    # Disabled by default, and disabled again when the state supplies no scale.
    @test CL.positivity_floors(solver, Q, StepControl()) == (0.0, 0.0)
    saved = Q[padded_index(solver, 2, 2, 2), 1]
    Q[padded_index(solver, 2, 2, 2), 1] = -3.0
    @test CL.positivity_floors(solver, Q, StepControl(floor_ratio=1e-6)) == (0.0, 0.0)
    Q[padded_index(solver, 2, 2, 2), 1] = saved

    floor!(Qx, scope) = CL.apply_positivity_floor!(solver, Qx, rho_floor, e_floor,
                                                   scope)
    # A healthy state is untouched under either scope, and reports nothing.
    for scope in (:representable, :internal_energy)
        Q1 = copy(Q)
        t = floor!(Q1, scope)
        @test (t.cells, t.low_energy) == (0, 0)
        @test Q1 == Q
    end

    # 1. A negative partial density is clipped and the mixture density is left
    #    exactly where it was, so the repair adds no mass at all.
    I = padded_index(solver, 3, 3, 3)
    Q2 = copy(Q)
    ρ_before = mixture_density(solver, Q2, I)
    Q2[I, 1] = -0.4
    Q2[I, 2] = ρ_before + 0.4
    t = floor!(Q2, :representable)
    @test t.cells == 1
    @test t.mass == 0.0
    @test Q2[I, 1] == 0.0
    @test mixture_density(solver, Q2, I) ≈ ρ_before rtol = 1e-15

    #    An undershoot inside the species band is one the validation accepts,
    #    so the clip leaves it and counts nothing; a zero band clips it.
    Qb = copy(Q)
    Qb[I, 1] = -0.02 * ρ_before
    Qb[I, 2] = 1.02 * ρ_before
    Qb0 = copy(Qb)
    t = floor!(Qb, :representable)
    @test t.cells == 0 && all(iszero, t.species)
    @test Qb == Qb0
    t = CL.apply_positivity_floor!(solver, Qb, rho_floor, e_floor, :representable;
                                   species_band=0.0)
    @test t.cells == 1
    @test Qb[I, 1] == 0.0

    #    With three species an excess of one can be shared between partners
    #    that each stay inside the band. The validation counts it on the upper
    #    side, and the clip returns the point to [0, 1].
    eos3 = IdealMixture([IdealSpecies{Float64}("a", 1.0, 1.4),
                         IdealSpecies{Float64}("b", 2.0, 1.6),
                         IdealSpecies{Float64}("c", 3.0, 1.3)])
    s3 = mkslv(n_global=(12, 12, 12), eos=eos3)
    Q3s = allocate_state(s3)
    initialize!(s3, Q3s, (x, y, z) -> Prim(rho=2.0, p=1.0, Y=(0.5, 0.25, 0.25)))
    floors3 = CL.positivity_floors(s3, Q3s, StepControl(floor_ratio=1e-6))
    J = padded_index(s3, 3, 3, 3)
    for (Y, counted) in (((1.08, -0.04, -0.04), true), ((1.03, -0.015, -0.015), false))
        Qx = copy(Q3s)
        foreach(sp -> Qx[J, sp] = 2.0 * Y[sp], 1:3)
        @test state_report(s3, Qx).negative_species == counted
        t = CL.apply_positivity_floor!(s3, Qx, floors3..., :representable)
        @test t.cells == counted
        if counted
            @test (Qx[J, 2], Qx[J, 3]) == (0.0, 0.0)
            @test Qx[J, 1] ≈ 2.0 rtol = 1e-15
            @test state_valid(state_report(s3, Qx))
        else
            @test Qx[J, 2] == 2.0 * Y[2]
        end
    end

    # 2. A mixture density below the floor cannot be repaired conservatively, so
    #    the added mass is reported.
    Q3 = copy(Q)
    Q3[I, 1] = 1e-30
    Q3[I, 2] = 1e-30
    t = floor!(Q3, :representable)
    @test t.cells == 1
    @test mixture_density(solver, Q3, I) ≈ rho_floor rtol = 1e-12
    @test t.mass > 0

    # 3. Internal energy below the floor: counted under both scopes, repaired
    #    only under :internal_energy. This is the case the Noh validation run
    #    carries for its whole duration while still reaching the exact plateau.
    Q4 = copy(Q)
    ke = 0.5 * (Q4[I, m1]^2 + Q4[I, m2]^2 + Q4[I, m3]^2) / ρ0
    Q4[I, ie] = ke * 0.9                       # e = −0.1 ke / ρ, E still positive
    Q5 = copy(Q4)
    t = floor!(Q4, :representable)
    @test (t.cells, t.low_energy) == (0, 1)
    @test Q4 == Q5                              # counted, and nothing else
    E_before = Q5[I, ie]
    p_before = sqrt(Q5[I, m1]^2 + Q5[I, m2]^2 + Q5[I, m3]^2)
    t = floor!(Q5, :internal_energy)
    @test (t.cells, t.low_energy) == (1, 1)
    @test Q5[I, ie] == E_before                 # total energy conserved exactly
    @test t.energy == 0.0
    ke5 = 0.5 * (Q5[I, m1]^2 + Q5[I, m2]^2 + Q5[I, m3]^2) / ρ0
    @test (Q5[I, ie] - ke5) / ρ0 ≈ e_floor rtol = 1e-10
    @test t.momentum > 0
    @test sqrt(Q5[I, m1]^2 + Q5[I, m2]^2 + Q5[I, m3]^2) < p_before

    # 4. A negative total energy density is unrepresentable in any frame, so
    #    both scopes repair it, and the branch that does conserves momentum.
    for scope in (:representable, :internal_energy)
        Q6 = copy(Q)
        Q6[I, ie] = -1.0
        p_before = (Q6[I, m1], Q6[I, m2], Q6[I, m3])
        t = floor!(Q6, scope)
        @test (t.cells, t.low_energy) == (1, 1)
        @test (Q6[I, m1], Q6[I, m2], Q6[I, m3]) == p_before
        @test t.momentum == 0.0
        @test t.energy > 0
        ke6 = 0.5 * sum(x -> x^2, p_before) / ρ0
        @test (Q6[I, ie] - ke6) / ρ0 ≈ e_floor rtol = 1e-10
    end
end

@testset "state validity: the report and what each policy mode does" begin
    @test StepControl().validity === :strict
    @test_throws ArgumentError StepControl(validity=:lenient)
    # :repair has nothing to repair with unless the failsafe floors exist.
    @test_throws ArgumentError StepControl(validity=:repair)
    @test StepControl(validity=:repair, floor_ratio=1e-6).validity === :repair

    eos = IdealMixture([IdealSpecies{Float64}("a", 1.0, 1.4),
                        IdealSpecies{Float64}("b", 2.0, 1.6)])
    solver = mkslv(n_global=(12, 12, 12), eos=eos)
    Q = allocate_state(solver)
    initialize!(solver, Q, (x, y, z) -> Prim(rho=2.0, u=(0.5, 0.0, 0.0), p=1.0,
                                             Y=(0.25, 0.75)))
    m1, m2, m3 = solver.equations.i_mom
    ie = solver.equations.i_energy
    I = padded_index(solver, 3, 3, 3)

    clean = state_report(solver, Q)
    @test state_valid(clean)
    @test clean.points == 12^3
    @test clean.rho_min ≈ 2.0 rtol = 1e-14
    @test clean.nonfinite == 0 && clean.negative_density == 0
    @test occursin("StateReport(1728 points;", sprint(show, clean))

    # Each rejected kind is counted on its own, and a point that cannot carry a
    # density or an energy is not put to the EOS.
    poison(f) = (Qx = copy(Q); f(Qx); state_report(solver, Qx))
    r = poison(Qx -> Qx[I, ie] = NaN)
    @test (r.nonfinite, r.negative_density, r.inadmissible) == (1, 0, 0)
    @test isfinite(r.rho_min)                   # the NaN point is skipped whole
    r = poison(Qx -> (Qx[I, 1] = -3.0; Qx[I, 2] = 0.0))
    @test (r.nonfinite, r.negative_density, r.negative_species) == (0, 1, 0)
    r = poison(Qx -> (Qx[I, 1] = -0.2; Qx[I, 2] = 2.2))
    @test (r.negative_density, r.negative_species) == (0, 1)
    @test state_valid(r) == false
    # A mass fraction of -0.01 is the excursion a species interface carries at
    # any resolution, inside the default band; a zero band rejects it.
    @test StepControl().species_band == 0.05
    @test_throws ArgumentError StepControl(species_band=1.0)
    @test_throws ArgumentError StepControl(species_band=-0.01)
    Qs = copy(Q); Qs[I, 1] = -0.02; Qs[I, 2] = 2.02
    @test state_valid(state_report(solver, Qs))
    @test state_report(solver, Qs; species_band=0.0).negative_species == 1
    @test CL.check_validity(StepControl(species_band=0.0),
                            state_report(solver, Qs; species_band=0.0), "s",
                            1, 0.0, 1e-3, 0.4) isa SolverFailure
    @test validate_state!(solver, Qs; warn=false).negative_species == 0
    @test_throws SolverFailure validate_state!(solver, Qs;
                                               control=StepControl(species_band=0.0))
    # Internal energy below zero is inadmissible for a calorically perfect gas,
    # where e = cv T, and is reported as such rather than by a hardcoded test.
    r = poison(Qx -> Qx[I, ie] = 0.5 * Qx[I, m1]^2 / 2.0 - 1.0)
    @test (r.inadmissible, r.negative_density) == (1, 0)
    @test r.e_min < 0

    # The verdict is the policy, and it is taken from the reduced counts.
    bad = poison(Qx -> Qx[I, 1] = -3.0)
    for mode in (:strict, :repair)
        control = mode === :strict ? StepControl() :
                  StepControl(validity=:repair, floor_ratio=1e-6)
        f = CL.check_validity(control, bad, "the probe state", 7, 0.5, 1e-3, 0.4)
        @test f isa SolverFailure
        @test f.reason === :invalid_state
        @test (f.step, f.t) == (7, 0.5)
        @test occursin("negative density", sprint(showerror, f))
        @test occursin("validity = :permissive", sprint(showerror, f))
    end
    @test CL.check_validity(StepControl(validity=:permissive), bad, "s",
                            1, 0.0, 1e-3, 0.4) === nothing
    @test CL.check_validity(StepControl(), clean, "s", 1, 0.0, 1e-3, 0.4) ===
          nothing

    # validate_state! applies it. Permissive accepts and reports; strict raises.
    Qbad = copy(Q)
    Qbad[I, 1] = -3.0
    @test_throws SolverFailure validate_state!(solver, Qbad)
    permissive = StepControl(validity=:permissive)
    @test (@test_logs (:warn,) match_mode=:any validate_state!(solver, Qbad;
              control=permissive)).negative_density == 1
    @test Qbad[I, 1] == -3.0                    # accepted, never touched
    @test state_valid(validate_state!(solver, Q))

    # Each scope keeps its own postcondition where it acts, which is what makes
    # the residual rejection below a contract rather than a defect.
    rho_floor, e_floor = CL.positivity_floors(solver, Q,
                                              StepControl(floor_ratio=1e-6))
    floor!(Qx, scope) = CL.apply_positivity_floor!(solver, Qx, rho_floor,
                                                   e_floor, scope)
    dens(Qx, J) = sum(Qx[J, sp] for sp in 1:2)
    kin(Qx, J) = (Qx[J, m1]^2 + Qx[J, m2]^2 + Qx[J, m3]^2) / (2 * dens(Qx, J))
    spec(Qx, J) = (Qx[J, ie] - kin(Qx, J)) / dens(Qx, J)

    # Raising a density toward the floor at fixed momentum and total energy
    # increases the internal energy DENSITY, so the substitution never deepens
    # the deficit it leaves behind.
    Qm = copy(Q)
    Qm[I, 1] = rho_floor * 0.3
    Qm[I, 2] = 0.0
    rhoe_before = Qm[I, ie] - kin(Qm, I)
    mom_before = (Qm[I, m1], Qm[I, m2], Qm[I, m3])
    E_before = Qm[I, ie]
    floor!(Qm, :representable)
    @test Qm[I, ie] - kin(Qm, I) > rhoe_before
    @test (Qm[I, m1], Qm[I, m2], Qm[I, m3]) == mom_before
    @test Qm[I, ie] == E_before

    # Where :representable does act on the energy, it lands exactly on the
    # floor and leaves the momentum alone.
    Qr = copy(Q)
    Qr[I, ie] = dens(Qr, I) * e_floor * 0.1     # E < rho * e_floor
    mom_before = (Qr[I, m1], Qr[I, m2], Qr[I, m3])
    floor!(Qr, :representable)
    @test spec(Qr, I) >= e_floor * (1 - 1e-12)
    @test (Qr[I, m1], Qr[I, m2], Qr[I, m3]) == mom_before

    # :internal_energy carries the stronger postcondition on a point
    # :representable declines: the internal energy reaches the floor, the total
    # energy is conserved exactly, and the momentum it removed is tallied.
    Qd = copy(Q)
    Qd[I, ie] = kin(Qd, I) + dens(Qd, I) * e_floor * 0.1
    E_before = Qd[I, ie]
    t_shallow = floor!(copy(Qd), :representable)
    @test t_shallow.low_energy == 1 && t_shallow.cells == 0   # counted, not fixed
    t_deep = floor!(Qd, :internal_energy)
    @test spec(Qd, I) >= e_floor * (1 - 1e-12)
    @test Qd[I, ie] == E_before
    @test t_deep.momentum > 0

    # Repair mode substitutes, reports what it substituted, and then rejects
    # what the substitution did not fix. The two scopes differ in what they
    # promise, and this pins the difference. A nonpositive density carries no
    # recoverable internal energy, so the floor substitutes one; the original
    # momentum at that minimal density leaves a large specific kinetic energy,
    # and the point's total energy still clears rho*e_floor, which is the
    # bound :representable acts on. It therefore counts the point and leaves
    # it, exactly as documented, and strict validation rejects what remains.
    # The repair does not deepen the deficit: raising rho at fixed momentum and
    # total energy increases E - |m|^2/(2rho) monotonically.
    repair = StepControl(validity=:repair, floor_ratio=1e-6)
    floors = CL.positivity_floors(solver, Q, repair)
    @test floors[1] > 0
    Qrep = copy(Q)
    Qrep[I, 1] = -3.0
    Qrep[I, 2] = 0.0
    tally_before = solver.floor_tally.cells
    @test_throws SolverFailure validate_state!(solver, Qrep; control=repair,
                                               floors=floors, warn=false)
    @test solver.floor_tally.cells == tally_before + 1
    @test mixture_density(solver, Qrep, I) ≈ floors[1] rtol = 1e-12
    # Every species change the repairs make is tallied: the clip moves mass
    # between species and the density floor adds `mass`.
    @test sum(solver.floor_tally.species) ≈ solver.floor_tally.mass rtol = 1e-12

    # The scope that converts kinetic energy back into internal energy does
    # fix it, and the state is then accepted.
    deep = StepControl(validity=:repair, floor_ratio=1e-6,
                       floor_scope=:internal_energy)
    Qdeep = copy(Q)
    Qdeep[I, 1] = -3.0
    Qdeep[I, 2] = 0.0
    rep = @test_logs (:warn,) match_mode=:any validate_state!(solver, Qdeep;
              control=deep, floors=floors)
    @test state_valid(rep)
    @test mixture_density(solver, Qdeep, I) ≈ floors[1] rtol = 1e-12
    @test solver.floor_tally.momentum > 0
end

@testset "state validity: a repair before the step renews the prepared state" begin
    # run! measures the rate, which refreshes the primitives the first stage
    # reuses, and then validates. A repair there rewrites points after that
    # refresh, so the step must start from the repaired state as if it had
    # been handed that state: the run below is compared with one whose
    # callback applies the same repair itself.
    eos = IdealMixture([IdealSpecies{Float64}("a", 1.0, 1.4),
                        IdealSpecies{Float64}("b", 2.0, 1.6)])
    control = StepControl(validity=:repair, floor_ratio=1e-6, validity_interval=1)
    ic(x, y, z) = Prim(rho=2.0 + 0.1sin(x), u=(0.5, 0.0, 0.0), p=1.0 + 0.1cos(y),
                       Y=(0.25, 0.75))
    fresh() = begin
        s = mkslv(n_global=(12, 12, 12), eos=eos)
        Qs = allocate_state(s)
        initialize!(s, Qs, ic)
        (s, Qs)
    end
    s0, Q0 = fresh()
    floors = CL.positivity_floors(s0, Q0, control)
    # After the first step, move a fifth of one point's mixture mass from the
    # first species onto the second, outside the species band.
    function poison!(s, Qs)
        s.step == 1 || return false
        I = padded_index(s, 3, 3, 3)
        rho = Qs[I, 1] + Qs[I, 2]
        Qs[I, 1] = -0.2rho
        Qs[I, 2] = 1.2rho
        return false
    end
    repaired!(s, Qs) = (poison!(s, Qs);
                        s.step == 1 && CL.apply_positivity_floor!(s, Qs, floors...,
                                                                  control.floor_scope);
                        false)
    sa, Qa = fresh()
    @test_logs (:warn, r"repaired 1 cell") match_mode=:any run!(sa, Qa;
        tfinal=1.0, nmax=2, callback=poison!, control=control)
    sb, Qb = fresh()
    run!(sb, Qb; tfinal=1.0, nmax=2, callback=repaired!, control=control)
    @test sa.floor_tally.cells == sb.floor_tally.cells + 1
    @test sa.dt_prev == sb.dt_prev
    @test maximum(abs, parent(Qa) .- parent(Qb)) == 0
end

@testset "state validity: initial and returned states are checked" begin
    # An initial condition outside the EOS domain is rejected by setup, where
    # the previous behaviour was to integrate it.
    build(p, control) = begin
        prob = Problem(domain=((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)), bcs=per3,
                       ic=(x, y, z) -> Prim(rho=1.0, p=p))
        setup(prob, Numerics(n_global=(12, 12, 12), control=control))
    end
    @test_throws SolverFailure build(0.0, StepControl())
    solver, Q = build(0.0, StepControl(validity=:permissive))
    @test state_report(solver, Q).inadmissible == 12^3
    solver, Q = build(1.0, StepControl())
    @test state_valid(state_report(solver, Q))

    # The state a run returns is validated on the way out, at each of the three
    # ways a run ends. The step checks read the state ENTERING a step, so
    # without this an nmax, tfinal or callback exit returned its last result
    # uninspected.
    sink!(s, Qs) = (Qs[padded_index(s, 2, 2, 2), 1] = -1.0; false)
    fresh(; control=StepControl()) = begin
        s = mkslv(n_global=(16, 12, 12), control=control)
        Qs = allocate_state(s)
        initialize!(s, Qs, (x, y, z) -> Prim(rho=1.0, p=1.0))
        (s, Qs)
    end
    s, Qs = fresh()
    @test_throws SolverFailure run!(s, Qs; tfinal=1.0, nmax=1, callback=sink!)
    # A callback that ends the run is the third exit, and is checked too.
    stop!(s, Qs) = (Qs[padded_index(s, 2, 2, 2), 1] = -1.0; true)
    s, Qs = fresh()
    @test_throws SolverFailure run!(s, Qs; tfinal=1.0, nmax=99, callback=stop!)
    # Permissive returns it and says so.
    s, Qs = fresh(control=StepControl(validity=:permissive))
    @test_logs (:warn,) match_mode=:any run!(s, Qs; tfinal=1.0, nmax=1,
                                             callback=sink!)
    @test s.step == 1
    @test mixture_density(s, Qs, padded_index(s, 2, 2, 2)) < 0
    @test state_valid(state_report(s, Qs)) == false

    # The rejection reaches the retry path rather than raising past it: with a
    # savepoint and retries available the run rolls back and lowers the CFL,
    # exactly as a step check does. The sink fires only on the first attempt,
    # so the replacement trajectory returns a valid state.
    s, Qs = fresh(control=StepControl(retries=2, savepoint_interval=1))
    fired = Ref(0)
    once!(sv, Qv) = (fired[] += 1; fired[] == 1 &&
                     (Qv[padded_index(sv, 2, 2, 2), 1] = -1.0); false)
    cfl0 = s.cfl
    @test_logs (:warn,) match_mode=:any run!(s, Qs; tfinal=1.0, nmax=2,
                                             callback=once!)
    @test s.cfl < cfl0                          # rolled back, not raised past
    @test state_valid(state_report(s, Qs))

    s, Qs = fresh()
    @test_throws SolverFailure run!(s, Qs; tfinal=1.0, nmax=1,
                                    callback=(sink!, state_guard(s, Qs)))

    # The guard reports under its own control; the solver's has to agree that
    # the state may be returned, since run! validates the result as well.
    s, Qs = fresh(control=StepControl(validity=:permissive))
    guard = CL.StateGuard(s, Qs; control=StepControl(validity=:permissive))
    @test_logs (:warn,) match_mode=:any run!(s, Qs; tfinal=1.0, nmax=1,
        callback=(sink!, Callback(EveryStep(), guard)))
    @test (guard.checks, guard.rejected) == (1, 1)
    @test mixture_density(s, Qs, padded_index(s, 2, 2, 2)) < 0   # accepted, not repaired

    s, Qs = fresh()
    repair = StepControl(validity=:repair, floor_ratio=1e-6)
    guard = CL.StateGuard(s, Qs; control=repair)
    run!(s, Qs; tfinal=1.0, nmax=1, callback=(sink!, Callback(EveryStep(), guard)))
    @test guard.rejected == 0                   # the repair left nothing to reject
    @test s.floor_tally.cells == 1
    @test mixture_density(s, Qs, padded_index(s, 2, 2, 2)) ≈ 1e-6 rtol = 1e-12
end

@testset "run!: failure is raised, and recoverable with retries" begin
    # Noh, three behaviours:
    #
    #   Planar Noh from cfl = 0.9 completes without a retry. Its first step
    #   is sized from the coefficients the initial data produces, because
    #   run! evaluates the right-hand side once before that step; sized on
    #   the acoustic rate alone the same run lost positivity above cfl 0.25,
    #   and the rollback that appeared to recover it rested on the failed
    #   trajectory's coefficients throttling the retry's first step.
    #
    #   Spherical Noh from the warm start at cfl = 0.9 fails abruptly in the
    #   origin's excursion near t = 0.39, representative of a guessed CFL,
    #   and must fail loudly rather than grind.
    #
    #   The same run from cfl = 1.8 with retries rolls back past the
    #   excursion with a halved CFL, twice, and recovers the correct plateau.
    #
    # All arms run the unrelaxed filter (filter_cfl = 0), the configuration
    # they were characterized under.
    γ = 5 / 3; p0 = 1e-4; tfin = 0.6
    build(ν, cfl, control; N, t0) = begin
        metric = ν == 1 ? CartesianMetric() : SphericalMetric()
        lobc = ν == 1 ? SlipWallBC() : OriginBC()
        dom2 = ν == 1 ? (0.0, 1 / N) : (π / 2, π / 2 + 1)
        dom3 = ν == 1 ? (0.0, 1 / N) : (0.0, 1.0)
        inflow = DirichletBC((x, y, z, t) -> begin
            ρ, u, _ = noh_exact(x, isfinite(t) ? t + t0 : t0, ν, γ)
            Prim(rho=ρ, u=(u, 0.0, 0.0), p=p0)
        end)
        w = 4 / N
        ic = (x, y, z) -> begin
            t0 <= 0 && return Prim(rho=1.0, u=(-1.0, 0.0, 0.0), p=p0)
            ρin, _, pin = noh_exact(0.0, t0, ν, γ)
            ρout, _, _ = noh_exact(x, t0, ν, γ)
            θ = tanh_blend(x, (γ - 1) / 2 * t0, w)
            Prim(rho=(1 - θ) * ρin + θ * ρout, u=(-θ, 0.0, 0.0),
                 p=(1 - θ) * pin + θ * p0)
        end
        prob = Problem(eos=IdealSpecies("gas"; gamma=γ, R=1.0),
                       transport=ConstantTransport(mu0=0.0),
                       metric=metric, domain=((0.0, 1.0), dom2, dom3),
                       bcs=((lobc, inflow), per3[2], per3[3]), ic=ic)
        setup(prob, Numerics(n_global=(N, 1, 1), art=ArtificialProperties(enabled=true),
                             cfl=cfl, control=control, filter=StateFilter(; cfl=0.0)))
    end
    # Post-shock plateau, sampled between the wall-heating layer and the shock
    # at x = (γ−1)t/2 = 0.2 — a window that straddles the shock would average
    # the answer with the undisturbed inflow and pass for the wrong reason.
    plateau(s, Q, N) = begin
        CL.exchange_state!(Q, s.decomp); CL.primitives!(s, Q)
        core = [i for i in 1:N if 0.06 <= xcoord(s, 1, i) <= 0.14]
        sum(s.rho[padded_index(s, i, 1, 1)] for i in core) / length(core)
    end

    # Permissive on the returned state: a Noh run ends with a handful of
    # negative-internal-energy cells ahead of the front at any CFL, so a
    # strict exit check would reject a correct trajectory.
    s1, Q1 = build(1, 0.9, StepControl(validity=:permissive); N=400, t0=0.0)
    run!(s1, Q1; tfinal=tfin, nmax=20_000)
    @test s1.t ≈ tfin rtol = 1e-9
    @test s1.cfl == 0.9                        # no retry was needed
    @test plateau(s1, Q1, 400) ≈ 4.0 rtol = 0.05  # exact Noh plateau for nu = 1

    s2, Q2 = build(3, 0.9, StepControl(); N=256, t0=0.3)
    err = nothing
    try
        run!(s2, Q2; tfinal=tfin - 0.3, nmax=20_000)
    catch e
        err = e
    end
    @test err isa SolverFailure
    @test err.reason in (:negative_density, :dt_collapse)
    @test s2.step < 20_000                     # it stopped early, it did not grind

    s3, Q3 = build(3, 1.8, StepControl(retries=5, savepoint_interval=20,
                                       validity=:permissive); N=256, t0=0.3)
    run!(s3, Q3; tfinal=tfin - 0.3, nmax=20_000)
    @test s3.t ≈ tfin - 0.3 rtol = 1e-9
    @test s3.cfl < 1.8                         # it backed off, and says by how much
    @test plateau(s3, Q3, 256) ≈ 64.0 rtol = 0.05  # exact Noh plateau for nu = 3
    # The backoff compounds: successive retries must keep halving, not keep
    # re-applying one factor to the same starting CFL.
    @test s3.cfl <= 1.8 * 0.25 + 1e-12
end
