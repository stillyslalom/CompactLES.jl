# The serial suite's tests of the NASA-9 and CEA thermodynamics and the stiffened
# gas. Included by test/serial_suite.jl, which loads the package and defines the
# imports and helpers used here; not runnable on its own.

@testset "NASA-9 mixture reduces exactly to the ideal mixture" begin
    @test_throws ArgumentError IdealMixture(IdealSpecies{Float64}[])
    @test_throws ArgumentError Nasa9Mixture(Nasa9Species{Float64}[])
    # The strongest available check on the polynomial machinery: with only the
    # constant term a3 populated, cp is temperature-independent and every
    # quantity — including the Newton inversion of e(T) — must reproduce the
    # closed-form ideal-gas answer to round-off. A transcription error in the
    # enthalpy integral, the cv, or the sound speed shows up here. The
    # temperatures are nondimensional and lie below the record's 200 K edge, so
    # the polynomial is evaluated there by stating `:polynomial`.
    γ, R = 1.4, 1.0
    cp = γ * R / (γ - 1)
    ideal = IdealSpecies("gas"; gamma=γ, R=R)
    poly = Nasa9Mixture([nasa9_constant_cp("gas", R, cp)]; extrapolate=:polynomial)
    @test nspecies(poly) == 1
    for T in (0.3, 1.0, 7.5, 300.0)
        @test CL.species_cp(poly, 1, T) ≈ cp rtol = 1e-14
        @test CL.species_enthalpy(poly, 1, T) ≈ cp * T rtol = 1e-14
        @test CL.species_energy(poly, 1, T) ≈ (cp - R) * T rtol = 1e-14
    end
    pr = Prim(u=(0.3, -0.1, 0.2), p=0.8, T_ion=1.7)
    qi = conserved_from_prim(ideal, pr)
    qp = conserved_from_prim(poly, pr)
    @test all(qi .≈ qp)
    # ... and through the full primitives path, which is where the Newton
    # inversion actually runs.
    for eos in (ideal, poly)
        solver = mkslv(n_global=(12, 12, 12), eos=eos)
        Q = allocate_state(solver)
        initialize!(solver, Q, (x, y, z) -> Prim(u=(0.3, 0, 0), p=0.8, T_ion=1.7))
        CL.exchange_state!(Q, solver.decomp)
        CL.primitives!(solver, Q)
        I = padded_index(solver, 3, 4, 5)
        @test solver.p[I] ≈ 0.8 rtol = 1e-12
        @test solver.T_ion[I] ≈ 1.7 rtol = 1e-12
        @test solver.c[I] ≈ sqrt(γ * R * 1.7) rtol = 1e-12
        @test solver.cp_mix[I] ≈ cp rtol = 1e-12
    end
end

@testset "NASA-9: thermodynamic consistency of a varying cp" begin
    # A genuinely temperature-dependent coefficient set, checked against the
    # two identities that must hold whatever the coefficients are:
    # dh/dT = cp, and the Newton inversion is the inverse of e(T).
    R = 287.0
    sp = Nasa9Species{Float64}(name="fake", R=R,
                               a=(1.2e4, -50.0, 3.6, 6.0e-4, -1.0e-7, 1.0e-11,
                                  -4.0e-16), b1=-1.0e3)
    eos = Nasa9Mixture([sp, nasa9_constant_cp("inert", 200.0, 900.0)];
                       T_guess=500.0)
    for T in (250.0, 800.0, 2500.0)
        δ = 1e-4 * T
        dh = (CL.species_enthalpy(eos, 1, T + δ) -
              CL.species_enthalpy(eos, 1, T - δ)) / 2δ
        @test dh ≈ CL.species_cp(eos, 1, T) rtol = 1e-7
    end
    # cp really does vary — otherwise the identity above is vacuous — and cv
    # stays positive across the range, which the Newton solve relies on.
    @test CL.species_cp(eos, 1, 2500.0) / CL.species_cp(eos, 1, 300.0) > 1.2
    @test all(CL.species_cp(eos, 1, T) > R for T in 250.0:50.0:2500.0)
    # e(T) round trip through the Newton solve, at three compositions.
    for Y in ((1.0, 0.0), (0.5, 0.5), (0.2, 0.8))
        for T in (250.0, 800.0, 2500.0)
            e = sum(Y[k] * CL.species_energy(eos, k, T) for k in 1:2)
            @test CL.mixture_temperature(eos, e, k -> Y[k]) ≈ T rtol = 1e-12
        end
    end
    # And end to end: a state initialized from (p, T_ion) recovers both.
    solver = mkslv(n_global=(12, 12, 12), eos=eos)
    Q = allocate_state(solver)
    initialize!(solver, Q, (x, y, z) -> Prim(Y=(0.35, 0.65), u=(120.0, 0, 0),
                                             p=2.5e5, T_ion=1400.0))
    CL.exchange_state!(Q, solver.decomp)
    CL.primitives!(solver, Q)
    I = padded_index(solver, 3, 4, 5)
    @test solver.p[I] ≈ 2.5e5 rtol = 1e-10
    @test solver.T_ion[I] ≈ 1400.0 rtol = 1e-10
    @test solver.Y[2][I] ≈ 0.65 rtol = 1e-12
    # A step must stay finite: the Newton solve runs inside every RK stage.
    run!(solver, Q; tfinal=1e9, nmax=3)
    @test all(isfinite, Q)
end

@testset "NASA-9: what the temperature recovery reports about itself" begin
    # Every check runs in both precisions, since the criterion is written from
    # eps of the coefficient type and a Float32 mechanism must behave the same
    # way rather than merely not crash.
    for T in (Float64, Float32)
        # A recovery inside the fitted range succeeds and says so.
        mix = Nasa9Mixture(T, read_nasa9(["He", "CO2"]))
        Y = (T(0.3), T(0.7))
        for Tref in T.((500.0, 2000.0, 4000.0, 8000.0))
            e = sum(Y[k] * CL.species_energy(mix, k, Tref) for k in 1:2)
            Trec, status = CL.mixture_temperature_status(mix, e, k -> Y[k])
            @test status == CL.TEMPERATURE_OK
            @test Trec ≈ Tref rtol = 256eps(T)
            @test CL.mixture_temperature(mix, e, k -> Y[k]) === Trec
        end
        # An interval join is approached from both sides. The fits are only
        # continuous at a join to the tolerance the reader enforces, so the
        # inversion is accurate there to that continuity and not to eps.
        for Tjoin in T.((1000.0, 6000.0)), δ in T.((-1e-3, 1e-3))
            Tref = Tjoin * (1 + δ)
            e = sum(Y[k] * CL.species_energy(mix, k, Tref) for k in 1:2)
            Trec, status = CL.mixture_temperature_status(mix, e, k -> Y[k])
            @test status == CL.TEMPERATURE_OK
            @test Trec ≈ Tref rtol = 1e-4
        end
        # He is fitted from 300 K up, so an ambient state below that is an
        # extrapolation. It recovers accurately and is reported as extrapolated.
        e_cold = sum(Y[k] * CL.species_energy(mix, k, T(250)) for k in 1:2)
        Tcold, status = CL.mixture_temperature_status(mix, e_cold, k -> Y[k])
        @test status & CL.TEMPERATURE_OUT_OF_RANGE != 0
        @test status & CL.TEMPERATURE_NOT_CONVERGED == 0
        @test Tcold ≈ 250 rtol = 1e-5
        # Temperature extremes: no root in the search range, diagnosed, finite,
        # and returned rather than raised, because this runs per point.
        for e_extreme in T.((-1e12, 1e30))
            Text, status = CL.mixture_temperature_status(mix, e_extreme, k -> Y[k])
            @test status & CL.TEMPERATURE_NOT_CONVERGED != 0
            @test isfinite(Text) && Text > 0
        end
        # An invalid composition leaves the mixture cv nonpositive, so there is
        # no bracket and no unique root to converge to.
        _, status = CL.mixture_temperature_status(mix, T(1e5), _ -> zero(T))
        @test status & CL.TEMPERATURE_NO_BRACKET != 0
        @test status & CL.TEMPERATURE_NOT_CONVERGED != 0
        # The failure reaches the state validation as the EOS's own verdict.
        @test state_admissibility(mix, T(1), T(1e5), _ -> zero(T), 2) &
              CL.STATE_UNRECOVERABLE != 0
        @test state_admissibility(mix, T(1), e_cold, k -> Y[k], 2) &
              CL.STATE_EXTRAPOLATED != 0
        # Extrapolating is not by itself a rejection: the polynomial and
        # tangent policies say the extension is acceptable, so the point is
        # reported and carried. `:missing` says the fit is undefined there and
        # marks the same point inadmissible, which is what strict validation
        # rejects.
        for policy in (:polynomial, :linear)
            m = Nasa9Mixture(["He", "CO2"]; extrapolate=policy)
            f = state_admissibility(m, T(1), e_cold, k -> Y[k], 2)
            @test f & CL.STATE_EXTRAPOLATED != 0
            @test f & CL.STATE_INADMISSIBLE == 0
        end
        gone = Nasa9Mixture(["He", "CO2"]; extrapolate=:missing)
        f = state_admissibility(gone, T(1), e_cold, k -> Y[k], 2)
        @test f & CL.STATE_EXTRAPOLATED != 0
        @test f & CL.STATE_INADMISSIBLE != 0
        # Inside the range the setting changes nothing.
        e_in = sum(Y[k] * CL.species_energy(gone, k, T(1000)) for k in 1:2)
        @test state_admissibility(gone, T(1), e_in, k -> Y[k], 2) == CL.STATE_OK
        @test state_admissibility(mix, T(1),
                                  sum(Y[k] * CL.species_energy(mix, k, T(1000))
                                      for k in 1:2), k -> Y[k], 2) == CL.STATE_OK
    end

    # The extrapolation policy is a choice, and it is inert inside the range.
    poly = Nasa9Mixture(["CO2"]; extrapolate=:polynomial)
    lin = Nasa9Mixture(["CO2"]; extrapolate=:linear)
    @test_throws ArgumentError Nasa9Mixture(["CO2"]; extrapolate=:constant)
    @test poly.extrapolate === :polynomial
    @test Nasa9Mixture(["CO2"]).extrapolate === :linear
    @test Nasa9Mixture(Float32, ["CO2"]).extrapolate === :linear
    @test Nasa9Mixture(read_nasa9(["CO2"])).extrapolate === :linear
    # `:missing` evaluates as the tangent extension does, so a step in progress
    # completes and the offending state can be read back; the difference is the
    # verdict it carries, not the number it returns.
    gone1 = Nasa9Mixture(["CO2"]; extrapolate=:missing)
    for Tq in (250.0, 1000.0, 6000.0, 20000.0)
        @test CL.species_cp(gone1, 1, Tq) === CL.species_cp(lin, 1, Tq)
    end
    for Tq in (250.0, 1000.0, 6000.0, 20000.0)
        @test CL.species_cp(poly, 1, Tq) === CL.species_cp(lin, 1, Tq)
        @test CL.species_enthalpy(poly, 1, Tq) === CL.species_enthalpy(lin, 1, Tq)
    end
    # Outside it, the degree-four fit is not a model of anything: run past
    # 20000 K its energy is not monotone in T, and the inversion finds another
    # temperature of the same energy, thousands of kelvin from the one the
    # energy was formed at, while the tangent extension recovers that one. Both
    # report that the answer came from outside the data.
    e_hot = CL.species_energy(poly, 1, 30000.0)
    Tpoly, spoly = CL.mixture_temperature_status(poly, e_hot, _ -> 1.0)
    Tlin, slin = CL.mixture_temperature_status(lin, CL.species_energy(lin, 1,
                                                                     30000.0),
                                               _ -> 1.0)
    @test spoly & CL.TEMPERATURE_OUT_OF_RANGE != 0
    @test slin & CL.TEMPERATURE_OUT_OF_RANGE != 0
    @test abs(Tpoly - 30000.0) > 1000
    @test Tlin ≈ 30000.0 rtol = 1e-12
    @test CL.species_cp(lin, 1, 40000.0) === CL.species_cp(lin, 1, 20000.0)
    # A hot state inside the range recovers under `:polynomial` as well. Its seed lands
    # past 20000 K, where the polynomial CO2 fit has a negative cv; an iterate
    # there would end the search with no bracket and a temperature twice the
    # root's. The iterate is held on the edge of the fitted range instead.
    for T in (Float64, Float32)
        co2 = Nasa9Mixture(T, ["CO2"]; extrapolate=:polynomial)
        for Tref in T.((13000, 15000, 17500, 19900))
            e = CL.species_energy(co2, 1, Tref)
            Trec, status = CL.mixture_temperature_status(co2, e, _ -> one(T))
            @test status == CL.TEMPERATURE_OK
            @test Trec ≈ Tref rtol = 1e-4
        end
    end

    # The inversion and its status run per point inside the primitives pass, so
    # neither the status nor the bracket may allocate. Measured on the pass
    # itself rather than on a loop over the routine, since that is the path and
    # a hand-written loop measures its own closure instead.
    # The transverse directions are collapsed so the pass runs serially at any
    # thread count: a threaded region allocates per region per thread, which
    # would measure the launcher rather than the inversion.
    mix = Nasa9Mixture(["He", "CO2"])
    solver = mkslv(n_global=(12, 1, 1), eos=mix)
    Q = allocate_state(solver)
    initialize!(solver, Q, (x, y, z) -> Prim(Y=(0.3, 0.7), p=1e5, T_ion=1000.0))
    # The patch routine itself: the `Solver` method builds the adapter first.
    ps = CL.sole_patch(solver)
    CL.primitives!(ps, Q)
    @test (@allocated CL.primitives!(ps, Q)) == 0
    @test solver.T_ion[padded_index(solver, 3, 1, 1)] ≈ 1000.0 rtol = 1e-10
end

@testset "NASA-9: leaving the fitted range ends a run only under :missing" begin
    # The per-point verdicts above, reached through run!, which validates the
    # state entering the call under the solver's policy. He is fitted from
    # 300 K, so a 250 K state is an extrapolation at every point.
    ic = (x, y, z) -> Prim(Y=(0.3, 0.7), p=1e5, T_ion=250.0)
    build(policy; kw...) = begin
        s = mkslv(n_global=(12, 1, 1),
                  eos=Nasa9Mixture(["He", "CO2"]; extrapolate=policy); kw...)
        Qs = allocate_state(s)
        initialize!(s, Qs, ic)
        (s, Qs)
    end
    for policy in (:polynomial, :linear)
        s, Qs = build(policy)
        run!(s, Qs; tfinal=1.0, nmax=1)
        r = state_report(s, Qs)
        @test s.step == 1
        @test (r.extrapolated, r.inadmissible) == (12, 0)
        @test state_valid(r)
    end
    s, Qs = build(:missing)
    err = try run!(s, Qs; tfinal=1.0, nmax=1); nothing catch e; e end
    @test err isa SolverFailure && err.reason === :invalid_state
    @test s.step == 0
    s, Qs = build(:missing; control=StepControl(validity=:permissive))
    @test_logs (:warn,) match_mode=:any run!(s, Qs; tfinal=1.0, nmax=1)
    r = state_report(s, Qs)
    @test s.step == 1
    @test (r.extrapolated, r.inadmissible) == (12, 12)
    @test state_valid(r) == false
end

@testset "NASA CEA reader: intervals, molar mass, and energy reference" begin
    he, co2 = read_nasa9(["He", "CO2"])
    @test (he.name, co2.name) == ("He", "CO2")
    @test co2.R ≈ 8.31446261815324 / 44.0095e-3 rtol = 1e-14
    @test length(co2.intervals) == 3
    @test [(item.Tmin, item.Tmax) for item in co2.intervals] ==
          [(200.0, 1000.0), (1000.0, 6000.0), (6000.0, 20000.0)]

    sensible = Nasa9Mixture([co2])
    @test CL.species_enthalpy(sensible, 1, 298.15) ≈ 0.0 atol = 1e-6
    @test CL.species_cp(sensible, 1, 300.0) ≈ 845.7241586606974 rtol = 1e-13
    for i in 1:2
        T_join = co2.intervals[i].Tmax
        left, right = co2.intervals[i], co2.intervals[i + 1]
        @test CL._nasa9_cp_over_R(left, T_join) ≈
              CL._nasa9_cp_over_R(right, T_join) rtol = 5e-7
        @test CL._nasa9_h_over_R(left, T_join) ≈
              CL._nasa9_h_over_R(right, T_join) rtol = 5e-7
    end

    co2_formation = read_nasa9("CO2"; reference=:formation)
    formation = Nasa9Mixture([co2_formation])
    h298_molar = CL.species_enthalpy(formation, 1, 298.15) * 44.0095e-3
    @test h298_molar ≈ -393510.0 atol = 5.0
    @test CL.species_cp(formation, 1, 300.0) == CL.species_cp(sensible, 1, 300.0)
    for eos in (sensible, formation), T in (220.0, 300.0, 999.0, 1001.0,
                                                    1500.0, 5999.0, 6001.0, 10000.0)
        e = CL.species_energy(eos, 1, T)
        @test CL.mixture_temperature(eos, e, _ -> 1.0) ≈ T rtol = 2e-13
    end

    Y = (0.3, 0.7)
    for reference in (:sensible, :formation)
        mixture = Nasa9Mixture(read_nasa9(["He", "CO2"]; reference))
        for T in (300.0, 1400.0, 7000.0)
            e = sum(Y[k] * CL.species_energy(mixture, k, T) for k in 1:2)
            @test CL.mixture_temperature(mixture, e, k -> Y[k]) ≈ T rtol = 2e-13
        end
    end
    # The late Air record exercises the CEA file's product/reactant separators.
    @test read_nasa9("Air").name == "Air"
    @test CL.species_names(Nasa9Mixture(read_nasa9(["CO2", "He"]))) == ["CO2", "He"]
    @test_throws ArgumentError read_nasa9("CO2"; reference=:unknown)

    # Zero-interval records list a heat of formation and no fit, and are three
    # lines, not two. Miscounting them desynchronizes the scan against
    # the file without a local failure, so pin both the record that has to be
    # rejected and a real species that only parses if the skip is right.
    @test_throws "no temperature intervals" read_nasa9("n-Butanol")
    @test length(read_nasa9("Jet-A(g)").intervals) == 2
    # Assert the message because a desynchronized scan also throws ArgumentError
    # after misreading a coefficient line as a species header.
    @test_throws "NASA CEA species not found" read_nasa9("not-a-CEA-species")

    # The two CEA records whose interval joins are loosest; the tolerance is
    # sized to admit them and still reject a mistranscribed coefficient.
    @test all(nspecies(Nasa9Mixture(read_nasa9([name]))) == 1
              for name in ("ALOCL", "SnCL2"))
    good = Nasa9Interval{Float64}(200.0, 1000.0,
                                  (0.0, 0.0, 3.5, 0.0, 0.0, 0.0, 0.0), 0.0)
    bad = Nasa9Interval{Float64}(1000.0, 6000.0,
                                 (0.0, 0.0, 4.5, 0.0, 0.0, 0.0, 0.0), 0.0)
    @test_throws ArgumentError Nasa9Species{Float64}("broken", 287.0, [good, bad])
end

@testset "StiffenedGas: perfect-gas limit, and a real liquid" begin
    # p_inf = 0 must reproduce a perfect gas exactly, which pins the algebra;
    # then a water-like parameter set exercises the nonzero-p_inf branch.
    γ = 1.4; R = 1.0; cv = R / (γ - 1)
    sg = StiffenedGas(gamma=γ, p_inf=0.0, cv=cv, name="gas")
    @test nspecies(sg) == 1
    pr = Prim(u=(0.3, -0.1, 0.2), p=0.8, T_ion=1.7)
    @test all(conserved_from_prim(IdealSpecies("gas"; gamma=γ, R=R), pr) .≈
              conserved_from_prim(sg, pr))
    solver = mkslv(n_global=(12, 12, 12), eos=sg)
    Q = allocate_state(solver)
    initialize!(solver, Q, (x, y, z) -> pr)
    CL.exchange_state!(Q, solver.decomp); CL.primitives!(solver, Q)
    I = padded_index(solver, 3, 4, 5)
    @test solver.p[I] ≈ 0.8 rtol = 1e-12
    @test solver.T_ion[I] ≈ 1.7 rtol = 1e-12
    @test solver.c[I] ≈ sqrt(γ * R * 1.7) rtol = 1e-12
    @test CL.eos_phi(sg, 1.0, 0.8, 1.7, γ * cv) ≈ cv / R rtol = 1e-13

    # Water: γ = 4.4, p∞ = 6e8 Pa. The point of the model is that the sound
    # speed is set by p∞, not by p, so it stays near 1500 m/s at 1 atm where a
    # perfect gas would give a few hundred.
    water = StiffenedGas(gamma=4.4, p_inf=6.0e8, cv=1816.0, name="water")
    s2 = build_solver(
        Problem(domain=((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)), bcs=per3, eos=water,
                ic=nothing),
        Numerics(n_global=(12, 12, 12), art=ArtificialProperties(enabled=false)))
    Q2 = allocate_state(s2)
    initialize!(s2, Q2, (x, y, z) -> Prim(u=(0, 0, 0), p=101325.0, rho=1000.0))
    CL.exchange_state!(Q2, s2.decomp); CL.primitives!(s2, Q2)
    J = padded_index(s2, 3, 4, 5)
    @test s2.p[J] ≈ 101325.0 rtol = 1e-9
    @test 1400 < s2.c[J] < 1700                      # c = sqrt(γ(p+p∞)/ρ)
    @test s2.c[J] ≈ sqrt(4.4 * (101325.0 + 6.0e8) / 1000.0) rtol = 1e-12
    # Uniform state ⇒ zero RHS, the same freestream statement made for every
    # other configuration in this suite.
    apply_bcs!(s2, Q2)
    dQ2 = zero(Q2)
    compute_rhs!(s2, Q2, dQ2)
    @test maximum(abs, dQ2) < 1e-8 * 101325.0
    # An acoustic pulse stays finite and does not leave the stiffened branch.
    initialize!(s2, Q2, (x, y, z) ->
        Prim(u=(0, 0, 0), p=101325.0 * (1 + 0.01sin(2π * x)), rho=1000.0))
    run!(s2, Q2; tfinal=1e9, nmax=5)
    CL.primitives!(s2, Q2)
    @test all(isfinite, Q2)
    @test minimum(s2.rho[padded_index(s2, i, j, k)] for i in 1:12, j in 1:12, k in 1:12) > 0
end
