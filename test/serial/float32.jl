# The serial suite's tests of single precision: the frontend, the operators, the
# EOS, refinement and the step. Included by test/serial_suite.jl, which loads the
# package and defines the imports and helpers used here; not runnable on its own.

@testset "Float32 frontend and full step" begin
    T = Float32
    prob = Problem(name="Float32 smoke", eos=IdealSpecies(T, "gas"; R=one(T), gamma=T(1.4)),
                   transport=ConstantTransport{T}(),
                   domain=((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)), bcs=per3,
                   ic=(x, y, z) -> Prim(rho=1 + 0.01sin(2π * x), T_ion=1,
                                        u=(0.1, 0.0, 0.0)))
    num = Numerics(n_global=(12, 12, 12), deriv=lele_d1_6(T),
                   filter=compact_filter(T(0.45), T), art=ArtificialProperties{T}(),
                   cfl=0.2)
    solver, Q = setup(prob, num)

    @test prob.transport isa ConstantTransport{T}
    @test num.art isa ArtificialProperties{T}
    @test parent(Q) isa Array{T,4}
    @test solver.deriv_plans[1] isa DirPlan{T}
    @test CL.positive_floor(T) > zero(T)
    @test CL.positive_floor(Float64) == 1e-300

    dt = compute_dt(solver, Q)
    @test dt isa T
    dQ = zero(Q)
    compute_rhs!(solver, Q, dQ)
    @test all(isfinite, parent(dQ))
    @test maximum(abs, parent(dQ)) > zero(T)

    # Compile every converted pointwise body through KernelAbstractions in
    # Float32 as well as through the ordinary threaded CPU route.
    dQ_ka = zero(Q)
    CL.with_knob(CL.FORCE_KA => true) do
        compute_rhs!(solver, Q, dQ_ka)
    end
    @test parent(dQ_ka) == parent(dQ)

    run!(solver, Q; tfinal=T(1e-4), nmax=1)
    @test solver.step == 1
    @test solver.t isa T
    @test all(isfinite, parent(Q))

    # Retries keep a savepoint, whose clock is Float64 in either precision.
    run!(solver, Q; tfinal=T(1e-3), nmax=3, control=StepControl(retries=2))
    @test solver.step > 1 && solver.t isa T && all(isfinite, parent(Q))
end

@testset "Float32 operator accuracy to the roundoff floor" begin
    T = Float32
    errs = T[]
    for N in (12, 18, 24)
        solver = build_solver(
            Problem(domain=((0.0, T(2π)), (0.0, T(2π)), (0.0, T(2π))), bcs=per3,
                    transport=ConstantTransport{T}(),
                    eos=IdealSpecies(T, "gas"; R=1.0, gamma=1.4), ic=nothing),
            Numerics(n_global=(N, 12, 12), art=ArtificialProperties{T}(enabled=false),
                     deriv=lele_d1_6(T), filter=compact_filter(T(0.45), T)))
        f = CL.field(solver.decomp)
        df = similar(f)
        fillf!(solver, f, (x, y, z) -> sin(x))
        exchange_halos!(f, solver.decomp)
        CL.deriv_along!(df, f, solver, 1, 1)
        CL._scale_grad!(df, solver, 1)
        push!(errs, ferr(solver, df, (x, y, z) -> cos(x)))
    end
    # C6 improves rapidly until truncation error meets Float32 roundoff around
    # 1e-6; requiring formal sixth order beyond that would test the format,
    # not the scheme.
    @test errs[2] < errs[1] / 4
    @test maximum(errs[2:3]) < T(3e-6)
end

@testset "Float32 built-in EOS and closed boundary matrix" begin
    T = Float32
    typed_num(; enabled=false) =
        (transport=ConstantTransport{T}(), art=ArtificialProperties{T}(enabled=enabled),
         deriv=lele_d1_6(T), filt=compact_filter(T(0.45), T))

    # A NASA-9 fit is tabulated in kelvin over a declared interval, so the
    # states here are physical: `nasa9_constant_cp` carries the CEA default
    # range of 200 K to 6000 K, and a nondimensional temperature would be an
    # argument outside it rather than a state the solver cannot carry.
    nasa = Nasa9Mixture([CL.nasa9_constant_cp(T, "gas", T(1), T(3.5))])
    tn = typed_num()
    ns = build_solver(
        Problem(domain=((0.0, T(2π)), (0.0, T(2π)), (0.0, T(2π))), bcs=per3, eos=nasa,
                transport=tn.transport, ic=nothing),
        Numerics(n_global=(12, 12, 12), art=tn.art, deriv=tn.deriv,
                 filter=StateFilter(tn.filt, interval=0)))
    NQ = allocate_state(ns)
    initialize!(ns, NQ, (x, y, z) -> Prim(u=(0.1, 0, 0), p=1, T_ion=300))
    CL.exchange_state!(NQ, ns.decomp)
    CL.primitives!(ns, NQ)
    I = padded_index(ns, 3, 4, 5)
    @test ns.p[I] ≈ T(1) rtol=T(2e-6)
    # The inversion's own criterion is 32 eps(Float32) relative, so the
    # recovered temperature is checked at that scale and not tighter.
    @test ns.T_ion[I] ≈ T(300) rtol=T(1e-5)
    @test state_valid(state_report(ns, NQ))
    run!(ns, NQ; tfinal=T(1e-4), nmax=1)
    @test all(isfinite, parent(NQ))

    sg = StiffenedGas{T}(gamma=T(1.4), p_inf=zero(T), cv=T(2.5),
                         name="gas")
    wall = (SlipWallBC(), SlipWallBC())
    ss = build_solver(
        Problem(domain=((0.0, one(T)), (0.0, one(T)), (0.0, one(T))),
                bcs=(wall, per3[2], per3[3]), eos=sg, transport=tn.transport, ic=nothing),
        Numerics(n_global=(24, 1, 1), art=tn.art, deriv=tn.deriv,
                 filter=StateFilter(tn.filt, interval=0)))
    SQ = allocate_state(ss)
    initialize!(ss, SQ, (x, y, z) ->
        Prim(u=(0.1sin(T(π) * x), 0, 0), p=1, rho=1))
    apply_bcs!(ss, SQ)
    m1 = ss.equations.i_mom[1]
    @test SQ[padded_index(ss, 1, 1, 1), m1] == zero(T)
    @test SQ[padded_index(ss, 24, 1, 1), m1] == zero(T)
    SdQ = zero(SQ)
    compute_rhs!(ss, SQ, SdQ)
    @test all(isfinite, parent(SdQ))
    run!(ss, SQ; tfinal=T(1e-4), nmax=1)
    @test all(isfinite, parent(SQ))
end

@testset "Float32 static two-level AMR step" begin
    T = Float32
    solver = build_solver(
        Problem(domain=((0.0, T(2π)), (0.0, one(T)), (0.0, one(T))), bcs=per3,
                transport=ConstantTransport{T}(), eos=IdealSpecies(T, "gas"; R=1.0, gamma=1.4),
               ic=nothing),
        Numerics(n_global=(48, 1, 1), art=ArtificialProperties{T}(enabled=false),
                 deriv=lele_d1_6(T), cfl=T(0.2),
                 filter=StateFilter(compact_filter(T(0.45), T), interval=0),
                 amr=AMR(initial=BlockRegion((20, 0, 0), (8, 1, 1)), regrid_interval=0)))
    states = allocate_state(solver)
    initialize!(solver, states, (x, y, z) ->
        Prim(u=(0.5, 0, 0), p=1, rho=1 + 0.1sin(x)))
    @test all(Q -> eltype(Q) === T, states.patches)
    @test compute_dt(solver, states) isa T
    run!(solver, states; tfinal=T(0.01), nmax=10)
    @test solver.t isa T
    @test all(Q -> all(isfinite, parent(Q)), states.patches)
end

@testset "Float32 varying-cp mixture and open/viscous boundaries" begin
    T = Float32
    # Every component at T, so the solver adopts it.
    transport = ConstantTransport{T}()
    typed = (art=ArtificialProperties{T}(enabled=false), deriv=lele_d1_6(T),
             filter=StateFilter(compact_filter(T(0.45), T); interval=0))

    # Both fits are tabulated over the CEA default range of 200 K to 6000 K and
    # the states below are physical, so the mixture is evaluated where its data
    # is defined. The first species has cp/R = 3.5 + 1e-3 T, which exercises the
    # iterative temperature recovery rather than its constant-cp limit.
    varying = Nasa9Species{T}(
        name="varying", R=one(T),
        a=(zero(T), zero(T), T(3.5), T(1e-3), zero(T), zero(T), zero(T)))
    inert = CL.nasa9_constant_cp(T, "inert", T(0.7), T(2.8))
    eos = Nasa9Mixture([varying, inert])
    @test CL.species_cp(eos, 1, T(600)) > CL.species_cp(eos, 1, T(300))
    ns = build_solver(Problem(domain=((0.0, T(2π)), (0.0, T(2π)), (0.0, T(2π))),
                              bcs=per3, eos=eos, transport=transport, ic=nothing),
                      Numerics(; n_global=(12, 12, 12), typed...))
    NQ = allocate_state(ns)
    initialize!(ns, NQ, (x, y, z) ->
        Prim(u=(0.1, 0, 0), p=1, T_ion=300, Y=(0.3, 0.7)))
    CL.exchange_state!(NQ, ns.decomp)
    CL.primitives!(ns, NQ)
    I = padded_index(ns, 3, 4, 5)
    @test ns.p[I] ≈ T(1) rtol=T(2e-6)
    @test ns.T_ion[I] ≈ T(300) rtol=T(1e-5)
    @test ns.Y[1][I] ≈ T(0.3) rtol=T(2e-6)
    @test state_valid(state_report(ns, NQ))
    run!(ns, NQ; tfinal=T(1e-4), nmax=1)
    @test all(isfinite, parent(NQ))

    per = (PeriodicBC(), PeriodicBC())
    uin = (T(0.2), zero(T), zero(T))
    sb = build_solver(Problem(domain=((0.0, one(T)), (0.0, one(T)), (0.0, one(T))),
                              bcs=((NSCBCInflowBC(u=uin, T_ion=one(T), Y=T[1]),
                                    NSCBCOutflowBC(pinf=one(T))), per, per),
                              eos=IdealSpecies(T, "gas"; R=one(T), gamma=T(1.4)),
                              transport=transport, ic=nothing),
                      Numerics(; n_global=(32, 1, 1), typed...))
    BQ = allocate_state(sb)
    initialize!(sb, BQ, (x, y, z) -> Prim(u=uin, p=1, T_ion=1))
    apply_bcs!(sb, BQ)
    BdQ = zero(BQ)
    compute_rhs!(sb, BQ, BdQ)
    matched = maximum(abs(BdQ[padded_index(sb, i, 1, 1), c])
                      for i in (1, 32), c in 1:sb.equations.n_cons)
    @test matched < T(5e-5)
    @test sb.bcs[1][1] isa NSCBCInflowBC{T}
    @test sb.bcs[1][2] isa NSCBCOutflowBC{T}
    run!(sb, BQ; tfinal=T(1e-4), nmax=1)
    @test all(isfinite, parent(BQ))

    Twall = T(1.2)
    sw = build_solver(Problem(domain=((0.0, one(T)), (0.0, one(T)), (0.0, one(T))),
                              bcs=((NoSlipWallBC(Twall=Twall),
                                    NoSlipWallBC(Twall=Twall)), per, per),
                              eos=IdealSpecies(T, "gas"; R=one(T), gamma=T(1.4)),
                              transport=transport, ic=nothing),
                      Numerics(; n_global=(24, 1, 1), typed...))
    WQ = allocate_state(sw)
    initialize!(sw, WQ, (x, y, z) ->
        Prim(u=(0.3, -0.2, 0.1), p=1, T_ion=1))
    apply_bcs!(sw, WQ)
    CL.exchange_state!(WQ, sw.decomp)
    CL.primitives!(sw, WQ)
    for i in (1, 24)
        Iw = padded_index(sw, i, 1, 1)
        @test sw.u[Iw] == sw.v[Iw] == sw.w[Iw] == zero(T)
        @test sw.T_ion[Iw] == Twall
    end
    @test sw.bcs[1][1] isa NoSlipWallBC{T}
end

@testset "Float32 resolved fold and moving subcycled level" begin
    T = Float32
    per = (PeriodicBC(), PeriodicBC())
    # Every component at T, so the solver adopts it.
    transport = ConstantTransport{T}()
    typed = (art=ArtificialProperties{T}(enabled=false), deriv=lele_d1_6(T),
             filter=StateFilter(compact_filter(T(0.45), T); interval=0))

    sf = build_solver(Problem(domain=((0.0, one(T)), (0.0, T(2π)), (0.0, one(T))),
                              metric=CylindricalMetric(),
                              bcs=((AxisBC(), SlipWallBC()), per, per), transport=transport,
                              eos=IdealSpecies(T, "gas"; R=1.0, gamma=1.4),
                              ic=nothing),
                      Numerics(; n_global=(32, 16, 1), typed...))
    f = CL.field(sf.decomp)
    df = similar(f)
    fillf!(sf, f, (r, θ, z) -> r * cos(θ) * exp(-T(4) * r^2))
    CL.exchange_halos!(f, sf.decomp)
    CL.deriv_along!(df, f, sf, 1, 1)
    CL._scale_grad!(df, sf, 1)
    @test ferr(sf, df, (r, θ, z) ->
        cos(θ) * (one(T) - T(8) * r^2) * exp(-T(4) * r^2)) < T(6e-5)

    wall = (SlipWallBC(), SlipWallBC())
    sr = build_solver(
        Problem(domain=((0.0, one(T)), (0.0, one(T)), (0.0, one(T))), bcs=(wall, per, per),
                transport=ConstantTransport{T}(), eos=IdealSpecies(T, "gas"; R=1.0, gamma=1.4),
               ic=nothing),
        Numerics(n_global=(96, 1, 1), cfl=T(0.1), art=ArtificialProperties{T}(),
                 deriv=lele_d1_6(T), filter=compact_filter(T(0.45), T),
                 amr=AMR(initial=BlockRegion((34, 0, 0), (28, 1, 1)), subcycle=true,
                         regrid_interval=1)))
    states = allocate_state(sr)
    initialize!(sr, states, (x, y, z) ->
        x < T(0.5) ? Prim(u=(0, 0, 0), p=1, rho=1) :
                     Prim(u=(0, 0, 0), p=0.1, rho=0.125))
    initial_offset = CL.refined_region(sr).offset[1]
    run!(sr, states; tfinal=one(T), nmax=2)
    @test CL.refined_region(sr).offset[1] != initial_offset
    @test sr.t isa T
    @test all(Q -> eltype(Q) === T && all(isfinite, parent(Q)), states.patches)
    for (ps, Q) in CL.eachpatch(sr, states)
        @test minimum(Q[padded_index(ps, i, 1, 1), 1]
                      for i in 1:ps.decomp.n_local[1]) > T(0.05)
    end
end

include(joinpath(@__DIR__, "..", "float32_validation.jl"))
