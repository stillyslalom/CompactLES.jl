# The serial suite's tests of the right-hand side, the equation set, sources and the
# boundary conditions. Included by test/serial_suite.jl, which loads the package and
# defines the imports and helpers used here; not runnable on its own.

@testset "freestream preservation (uniform state ⇒ dQ ≈ 0)" begin
    # Uniform ρ, p, u = 0 must give zero RHS in every metric, with stretch,
    # and with folds: divergence/source/geometry consistency in one number.
    cases = [
        (; n_global=(24, 12, 12), L_domain=(1.0, 1.0, 1.0), metric=CartesianMetric(),
           bcs=per3, kw=(;)),
        (; n_global=(24, 12, 12), L_domain=(1.0, 1.0, 1.0), metric=CartesianMetric(),
           bcs=per3, kw=(; stretch=(sine_cluster(0.0, 1.0, 0.5, 0.4),
                                    nothing, nothing),
                         bcs=((SlipWallBC(), SlipWallBC()), per3[2], per3[3]))),
        (; n_global=(24, 12, 12), L_domain=(1.0, 2π, 0.5), metric=CylindricalMetric(),
           bcs=((SlipWallBC(), SlipWallBC()), per3[2], per3[3]),
           kw=(; origin=(0.2, 0.0, 0.0))),
        (; n_global=(32, 1, 12), L_domain=(1.0, 1.0, 0.5), metric=CylindricalMetric(),
           bcs=((AxisBC(), SlipWallBC()), per3[2], per3[3]), kw=(;)),
        (; n_global=(24, 16, 1), L_domain=(1.0, 2π, 1.0), metric=CylindricalMetric(),
           bcs=((AxisBC(), SlipWallBC()), per3[2], per3[3]), kw=(;)),
        (; n_global=(24, 12, 12), L_domain=(1.0, π, 2π), metric=SphericalMetric(),
           bcs=((OriginBC(), SlipWallBC()), (PoleBC(), PoleBC()), per3[3]),
           kw=(;)),
        # A stretched annulus with θ resolved, where the radial momentum takes
        # its pressure as a gradient because the compact derivative of a
        # nonlinear r(ξ) would not balance the source p/r.
        (; n_global=(16, 12, 1), L_domain=(1.0, 2π, 1.0), metric=CylindricalMetric(),
           bcs=((SlipWallBC(), SlipWallBC()), per3[2], per3[3]),
           kw=(; origin=(0.2, 0.0, 0.0),
                 stretch=(sine_cluster(0.2, 1.2, 0.5, 0.4), nothing, nothing))),
        # r collapsed at r = 1: no radial divergence is taken, so the radial
        # momentum's source carries Π_θθ, and on the sphere Π_φφ, without the
        # pressure. Setup warns once that the rest of the radial divergence's
        # area term is omitted.
        (; n_global=(1, 12, 1), L_domain=(1.0, 2π, 1.0), metric=CylindricalMetric(),
           bcs=per3, kw=(; origin=(1.0, 0.0, 0.0))),
        (; n_global=(1, 12, 1), L_domain=(1.0, π/2, 1.0), metric=SphericalMetric(),
           bcs=(per3[1], (SlipWallBC(), SlipWallBC()), per3[3]),
           kw=(; origin=(1.0, π/4, 0.0))),
    ]
    for cs in cases
        o = merge((; origin=(0.0, 0.0, 0.0), stretch=(nothing, nothing, nothing),
                   bcs=cs.bcs), cs.kw)
        domain = ntuple(d -> (o.origin[d], o.origin[d] + cs.L_domain[d]), 3)
        build() = build_solver(Problem(; domain, metric=cs.metric, bcs=o.bcs, ic=nothing),
                               Numerics(; n_global=cs.n_global, stretch=o.stretch,
                                        art=ArtificialProperties(enabled=false)))
        solver = cs.n_global[1] == 1 ?
                 (@test_logs (:warn, r"r is collapsed") build()) : build()
        Q = allocate_state(solver)
        initialize!(solver, Q, (x, y, z) -> Prim(u=(0, 0, 0), p=1.0, rho=1.0))
        apply_bcs!(solver, Q)
        dQ = zero(Q)
        compute_rhs!(solver, Q, dQ)
        m = maximum(abs(dQ[padded_index(solver, i, j, k), c])
                    for c in 1:solver.equations.n_cons, i in 1:solver.decomp.n_local[1],
                        j in 1:solver.decomp.n_local[2], k in 1:solver.decomp.n_local[3])
        @test m < 1e-8
    end
end

@testset "EOS: conserved ↔ primitive round trip (two species)" begin
    eos = IdealMixture([IdealSpecies{Float64}("a", 1.0, 1.4),
                        IdealSpecies{Float64}("b", 0.2, 1.09)])
    solver = mkslv(n_global=(12, 12, 12), eos=eos)
    Q = allocate_state(solver)
    pr = Prim(u=(0.3, -0.1, 0.2), p=0.8, T_ion=1.7, Y=(0.35, 0.65))
    initialize!(solver, Q, (x, y, z) -> pr)
    CL.exchange_state!(Q, solver.decomp)
    CL.primitives!(solver, Q)
    I = padded_index(solver, 3, 4, 5)
    @test solver.p[I] ≈ 0.8 atol = 1e-12
    @test solver.T_ion[I] ≈ 1.7 atol = 1e-12
    @test solver.u[I] ≈ 0.3 atol = 1e-12
    @test solver.Y[2][I] ≈ 0.65 atol = 1e-12
end

@testset "EquationSet owns the conserved layout" begin
    eos = IdealMixture([IdealSpecies{Float64}("a", 1.0, 1.4),
                        IdealSpecies{Float64}("b", 0.2, 1.09)])
    solver = mkslv(n_global=(12, 12, 12), eos=eos)
    @test solver.equations isa NavierStokes1T
    @test solver.equations.n_species == 2
    @test solver.equations.n_cons == 6
    @test solver.equations.i_mom == (3, 4, 5)
    @test solver.equations.i_energy == 6
    @test solver.equations.component_names ==
          ["rho_a", "rho_b", "rho_u1", "rho_u2", "rho_u3", "rho_E"]
    @test fieldtype(typeof(solver), :eos) === typeof(solver.eos)
    @test fieldtype(fieldtype(typeof(solver), :grid), :metric) === typeof(solver.metric)
    # `folds` is per-patch state; the concreteness contract moved with it.
    @test fieldtype(eltype(fieldtype(typeof(solver), :patches)), :folds) ===
          typeof(solver.folds)
end

@testset "tuple source terms add momentum and energy work" begin
    force = ConstantBodyForce((1.0, 2.0, -3.0))
    solver = mkslv(n_global=(12, 12, 12), sources=(force,))
    Q = allocate_state(solver)
    initialize!(solver, Q, (x, y, z) ->
        Prim(u=(0.5, 0.25, -0.1), p=1.0, rho=2.0))
    dQ = zero(Q)
    compute_rhs!(solver, Q, dQ)
    I = padded_index(solver, 3, 4, 5)
    @test dQ[I, solver.equations.i_mom[1]] ≈ 2.0 atol = 1e-10
    @test dQ[I, solver.equations.i_mom[2]] ≈ 4.0 atol = 1e-10
    @test dQ[I, solver.equations.i_mom[3]] ≈ -6.0 atol = 1e-10
    @test dQ[I, solver.equations.i_energy] ≈ 2.6 atol = 1e-10
    @test typeof(solver.sources) === Tuple{typeof(force)}
end

@testset "Workspace reproduces explicit stage arrays" begin
    solver1 = mkslv(n_global=(12, 12, 12))
    solver2 = mkslv(n_global=(12, 12, 12))
    Q1 = allocate_state(solver1)
    Q2 = allocate_state(solver2)
    ic = (x, y, z) -> Prim(u=(0.1sin(x), -0.1cos(y), 0.05sin(z)),
                            p=1 + 0.02cos(x), rho=1 + 0.03sin(y))
    initialize!(solver1, Q1, ic)
    initialize!(solver2, Q2, ic)
    dQ = zero(Q1)
    du = zero(Q1)
    workspace = Workspace(Q2)
    step!(solver1, Q1, dQ, du, 1e-4)
    step!(solver2, Q2, workspace, 1e-4)
    @test Q1 == Q2
    @test dQ == workspace.dQ
    @test du == workspace.du
end

@testset "conservation: periodic RHS integrates to zero" begin
    solver = mkslv(n_global=(16, 16, 16), transport=ConstantTransport(mu0=1e-3))
    Q = allocate_state(solver)
    initialize!(solver, Q, (x, y, z) ->
        Prim(u=(0.1sin(x)cos(y), -0.1cos(x)sin(y), 0.05sin(z)),
             p=1 + 0.05cos(x)cos(z), rho=1 + 0.1sin(y)))
    dQ = zero(Q)
    compute_rhs!(solver, Q, dQ)
    for c in 1:solver.equations.n_cons
        tot = sum(dQ[padded_index(solver, i, j, k), c] for i in 1:16, j in 1:16, k in 1:16)
        @test abs(tot) < 1e-8 * 16^3
    end
end

@testset "NSCBC outflow: matched uniform stream ⇒ no correction" begin
    inflow = DirichletBC((x, y, z, t) -> Prim(u=(0.3, 0, 0), p=1.0, rho=1.0))
    solver = build_solver(Problem(domain=((0.0, 1.0), (0.0, 0.4), (0.0, 0.4)),
                                  bcs=((inflow, NSCBCOutflowBC(pinf=1.0)), per3[2], per3[3]),
                                  ic=nothing),
                          Numerics(n_global=(32, 12, 12),
                                   art=ArtificialProperties(enabled=false)))
    Q = allocate_state(solver)
    initialize!(solver, Q, (x, y, z) -> Prim(u=(0.3, 0, 0), p=1.0, rho=1.0))
    apply_bcs!(solver, Q)
    dQ = zero(Q)
    compute_rhs!(solver, Q, dQ)
    nx = solver.decomp.n_local[1]
    m = maximum(abs(dQ[padded_index(solver, nx, j, k), c])
                for c in 1:solver.equations.n_cons, j in 1:12, k in 1:12)
    @test m < 1e-8
end

@testset "NSCBC outflow: a radial face relaxes at the curvature rate" begin
    # Gas at rest at the uniform pressure p∞ + δ: every term of the right-hand
    # side vanishes but the face's relaxation, ΔL = K δ, which enters the mass
    # as −ΔL/(2c²). On the outer radial face of r ∈ [1/2, 3/2], K is the
    # larger of σc/L and n_c c/(2r), n_c = 1 on a cylinder and 2 on a sphere;
    # on a Cartesian face it is σc/L.
    δ = 1e-3
    c = sqrt(1.4 * (1 + δ))
    for (metric, origin, nc) in ((CartesianMetric(), (0.5, 0.0, 0.0), 0),
                                 (CylindricalMetric(), (0.5, 0.0, 0.0), 1),
                                 (SphericalMetric(), (0.5, π / 2, 0.0), 2)),
        sigma in (0.0, 0.25, 4.0)
        solver = build_solver(Problem(domain=ntuple(d -> (origin[d], origin[d] + 1.0), 3),
                                      metric=metric,
                                      bcs=((SlipWallBC(),
                                            NSCBCOutflowBC(pinf=1.0, sigma=sigma)),
                                           per3[2], per3[3]),
                                      ic=nothing),
                              Numerics(n_global=(24, 1, 1),
                                       art=ArtificialProperties(enabled=false)))
        Q = allocate_state(solver)
        initialize!(solver, Q, (x, y, z) -> Prim(rho=1.0, p=1.0 + δ))
        dQ = zero(Q)
        compute_rhs!(solver, Q, dQ)
        nx = solver.decomp.n_local[1]
        K = max(sigma * c, nc * c / 3)
        face = dQ[padded_index(solver, nx, 1, 1), 1]
        @test isapprox(face, -K * δ / (2c^2); rtol=1e-8, atol=1e-14)
        @test maximum(abs(dQ[padded_index(solver, i, 1, 1), 1]) for i in 2:nx-1) < 1e-12
    end
end

# The Oscillating sphere tutorial's dipole on its radial grid with half its
# polar nodes, at the default relaxation: the plane form of the outflow
# reflects it by 1/(2kR), and with the curvature term the reflection is the
# second-order remainder 1/(2(kR)²) and the discretization's share.
include(joinpath(@__DIR__, "..", "sphere_dipole.jl"))

@testset "NSCBC outflow: a spherical wave leaves a radial face" begin
    m = sphere_dipole_reflection(2.1; n=128)
    plane = 1 / (2m.kR)
    @info "spherical outflow reflection" m.reflection plane m.amplitude
    @test m.reflection < plane / 10
    @test abs(m.amplitude - 1) < 0.03
end

@testset "NoSlipWallBC: adiabatic zeroes velocity, isothermal sets T_ion" begin
    for Twall in (NaN, 2.5)
        solver = build_solver(Problem(domain=((0.0, 1.0), (0.0, 0.4), (0.0, 0.4)),
                                      bcs=((NoSlipWallBC(Twall=Twall),
                                            NoSlipWallBC(Twall=Twall)), per3[2], per3[3]),
                                      ic=nothing),
                              Numerics(n_global=(24, 12, 12),
                                       art=ArtificialProperties(enabled=false)))
        Q = allocate_state(solver)
        initialize!(solver, Q, (x, y, z) -> Prim(u=(0.3, -0.2, 0.1), p=1.0, T_ion=1.0))
        apply_bcs!(solver, Q)
        CL.exchange_state!(Q, solver.decomp)
        CL.primitives!(solver, Q)
        nx = solver.decomp.n_local[1]
        uw = maximum(abs(solver.u[padded_index(solver, i, j, k)]) + abs(solver.v[padded_index(solver, i, j, k)]) +
                     abs(solver.w[padded_index(solver, i, j, k)])
                     for i in (1, nx), j in 1:12, k in 1:12)   # both walls
        @test uw < 1e-12
        if !isnan(Twall)                       # isothermal wall holds Twall
            e = maximum(abs(solver.T_ion[padded_index(solver, i, j, k)] - Twall)
                        for i in (1, nx), j in 1:12, k in 1:12)
            @test e < 1e-12
        end
    end
end

@testset "ExtrapolationBC: copies the adjacent interior plane" begin
    solver = build_solver(
        Problem(domain=((0.0, 1.0), (0.0, 0.4), (0.0, 0.4)),
                bcs=((ExtrapolationBC(), ExtrapolationBC()), per3[2], per3[3]), ic=nothing),
        Numerics(n_global=(24, 12, 12), art=ArtificialProperties(enabled=false)))
    Q = allocate_state(solver)
    initialize!(solver, Q, (x, y, z) -> Prim(u=(0.2 + x, 0, 0), p=1 + 0.1x, rho=1 + 0.3x))
    apply_bcs!(solver, Q)
    nx = solver.decomp.n_local[1]
    d = 0.0
    for c in 1:solver.equations.n_cons, j in 1:12, k in 1:12
        d = max(d, abs(Q[padded_index(solver, 1, j, k), c]  - Q[padded_index(solver, 2, j, k), c]))
        d = max(d, abs(Q[padded_index(solver, nx, j, k), c] - Q[padded_index(solver, nx-1, j, k), c]))
    end
    @test d == 0.0
    # a uniform state must be untouched by the extrapolation
    Q2 = allocate_state(solver)
    initialize!(solver, Q2, (x, y, z) -> Prim(u=(0.2, 0, 0), p=1.0, rho=1.0))
    ref = copy(Q2)
    apply_bcs!(solver, Q2)
    @test Q2 == ref
end

@testset "NSCBC inflow: matched uniform stream ⇒ no correction" begin
    # Mirrors the outflow test. Nothing else in the suite compiles this
    # method, so without it the inflow path, the transverse terms included,
    # is never executed.
    uin = (0.3, 0.0, 0.0)
    solver = build_solver(Problem(domain=((0.0, 1.0), (0.0, 0.4), (0.0, 0.4)),
                                  bcs=((NSCBCInflowBC(u=uin, T_ion=1.0),
                                        NSCBCOutflowBC(pinf=1.0)), per3[2], per3[3]),
                                  eos=IdealSpecies("gas"; gamma=1.4, R=1.0), ic=nothing),
                          Numerics(n_global=(32, 12, 12),
                                   art=ArtificialProperties(enabled=false)))
    Q = allocate_state(solver)
    initialize!(solver, Q, (x, y, z) -> Prim(u=uin, p=1.0, T_ion=1.0))
    apply_bcs!(solver, Q)
    dQ = zero(Q)
    compute_rhs!(solver, Q, dQ)
    m = maximum(abs(dQ[padded_index(solver, 1, j, k), c])
                for c in 1:solver.equations.n_cons, j in 1:12, k in 1:12)
    @test m < 1e-8
    # and a mismatched stream must produce a non-trivial correction
    Q2 = allocate_state(solver)
    initialize!(solver, Q2, (x, y, z) -> Prim(u=(0.15, 0, 0), p=1.0, T_ion=1.0))
    apply_bcs!(solver, Q2)
    dQ2 = zero(Q2)
    compute_rhs!(solver, Q2, dQ2)
    m2 = maximum(abs(dQ2[padded_index(solver, 1, j, k), c])
                 for c in 1:solver.equations.n_cons, j in 1:12, k in 1:12)
    @test m2 > 1e-3
end

@testset "NSCBC inflow: transverse terms freeze the incoming characteristics" begin
    # With the relaxation rates at zero and beta_t = 1, every imposed
    # incoming amplitude is −𝒯, so at the plane the rates of the incoming
    # characteristic variables, p ± ρc u_n for the incoming acoustic wave,
    # c²ρ − p, u_t and Y_k, vanish up to the truncation gap between the
    # conservative divergence in dQ and the characteristic form; the outgoing
    # one stays of order one. The transverse pressure gradient's share of 𝒯_t
    # carries the Mach number instead (src/physics/nscbc.jl), so the u_t rate measured
    # is the remainder after the −(1 − M) ∇_t p/ρ that share leaves, with ∇_t p
    # the solver's own derivative. Under beta_t = 0 the same rates are of
    # order one, and doubling the transverse resolution shrinks the gap at the
    # sixth order (measured ×70; ×16 is asserted). Both faces, two species
    # of one γ so that ρe = p/(γ − 1) gives ∂p/∂t from dQ.
    γ = 1.4
    eos = IdealMixture([IdealSpecies{Float64}("a", 1.0, γ),
                        IdealSpecies{Float64}("b", 0.5, γ)])
    ic(sgn) = (x, y, z) -> begin
        θ = 0.5 + 0.2 * sin(2π * y) * cos(4π * z)
        Prim(u=(sgn * (0.3 + 0.05 * sin(2π * y) + 0.02 * cos(4π * z)),
                0.1 * cos(2π * y), 0.05 * sin(4π * z) * cos(2π * y)),
             p=1 + 0.05 * sin(2π * y + 1) + 0.02 * cos(4π * z),
             T_ion=1 + 0.1 * cos(2π * y) * sin(4π * z), Y=(θ, 1 - θ))
    end
    function rhs(side, beta_t, n; eta=0.0, target=nothing)
        sgn = side == 1 ? 1.0 : -1.0
        bc = NSCBCInflowBC(u=(0.3sgn, 0.0, 0.0), T_ion=1.0, Y=[0.6, 0.4],
                           eta_u=eta, eta_T=eta, eta_t=eta, eta_Y=eta,
                           beta_t=beta_t, target=target)
        out = NSCBCOutflowBC(pinf=1.0)
        solver = build_solver(
            Problem(domain=((0.0, 1.0), (0.0, 1.0), (0.0, 0.5)),
                    bcs=(side == 1 ? (bc, out) : (out, bc), per3[2], per3[3]), eos=eos,
                    transport=ConstantTransport(mu0=0.0), ic=nothing),
            Numerics(n_global=n, art=ArtificialProperties(enabled=false)))
        Q = allocate_state(solver)
        initialize!(solver, Q, ic(sgn))
        apply_bcs!(solver, Q)
        dQ = zero(Q)
        compute_rhs!(solver, Q, dQ)
        return solver, dQ
    end
    # Plane maxima of the incoming characteristic rates and of the outgoing.
    function rates(solver, dQ, side; pressure=true)
        dpy, dpz = solver.tmp_a, solver.tmp_b
        CompactLES.deriv_along!(dpy, solver.p, solver, 2, 1)
        CompactLES.deriv_along!(dpz, solver.p, solver, 3, 1)
        m = solver.equations.i_mom
        ie = solver.equations.i_energy
        nx, ny, nz = solver.decomp.n_local
        i = side == 1 ? 1 : nx
        sgn = side == 1 ? 1.0 : -1.0
        ac_in = ac_out = en = tv = ty = 0.0
        for k in 1:nz, j in 1:ny
            I = padded_index(solver, i, j, k)
            ρ = solver.rho[I]; c = solver.c[I]
            u = (solver.u[I], solver.v[I], solver.w[I])
            ρt = dQ[I, 1] + dQ[I, 2]
            ut = ntuple(a -> (dQ[I, m[a]] - u[a] * ρt) / ρ, 3)
            pt = (γ - 1) * (dQ[I, ie] - sum(abs2, u) / 2 * ρt - ρ * sum(u .* ut))
            ac_in = max(ac_in, abs(pt + sgn * ρ * c * ut[1]))
            ac_out = max(ac_out, abs(pt - sgn * ρ * c * ut[1]))
            en = max(en, abs(c^2 * ρt - pt))
            gy, gz = solver.inv_h[2][I] * dpy[I], solver.inv_h[3][I] * dpz[I]
            share = pressure ? (1 - abs(u[1]) / c) / ρ : 0.0
            tv = max(tv, abs(ut[2] + share * gy), abs(ut[3] + share * gz))
            ty = max(ty, abs((dQ[I, 1] - solver.Y[1][I] * ρt) / ρ))
        end
        return (ac_in, en, tv, ty), ac_out
    end
    for side in (1, 2)
        full, out = rates(rhs(side, 1.0, (24, 24, 16))..., side)
        lodi, _ = rates(rhs(side, 0.0, (24, 24, 16))..., side; pressure=false)
        fine, _ = rates(rhs(side, 1.0, (24, 48, 32))..., side)
        @test out > 1.0
        @test all(lodi .> (1.0, 0.05, 0.1, 0.05))
        @test all(full ./ lodi .< 1e-2)
        @test all(full ./ fine .> 16)
    end
    # A pointwise target returning the constants reproduces the constant
    # body bitwise, relaxation on; the weight changes the answer.
    _, d1 = rhs(1, 1.0, (24, 24, 16); eta=0.28)
    _, d2 = rhs(1, 1.0, (24, 24, 16); eta=0.28,
                target=(x, y, z, t) -> Prim(u=(0.3, 0.0, 0.0), T_ion=1.0,
                                            p=1.0, Y=(0.6, 0.4)))
    _, d3 = rhs(1, 0.0, (24, 24, 16); eta=0.28)
    @test parent(d1) == parent(d2)
    @test parent(d1) != parent(d3)
end

@testset "NSCBC inflow: the r-z face freezes the incoming wave beside the axis" begin
    # The top face of an r-z grid with a radial velocity proportional to r
    # beside the axis, relaxation off. The transverse divergence there is
    # ∂u_r/∂r + u_r/r, the second term from the collapsed θ; with it in 𝒯
    # the incoming acoustic rate p − ρc u_z and the u_r rate fall with the
    # resolution as on a Cartesian face (×19 and ×12 from 24 to 48 nodes).
    # Without it the acoustic rate stays at half its beta_t = 0 value. The u_r
    # rate is measured after the −(1 − M) ∂p/∂r/ρ that the pressure share of
    # 𝒯_t leaves, as on the Cartesian face above.
    γ = 1.4
    eos = IdealMixture([IdealSpecies{Float64}("a", 1.0, γ),
                        IdealSpecies{Float64}("b", 0.5, γ)])
    ic(r, θ, z) = begin
        s = 0.5 + 0.2 * cos(2π * r^2) * sin(2π * z)
        Prim(u=(0.1 * r * (1 - r^2) * (1 + 0.5 * cos(2π * z)), 0.0,
                -(0.3 + 0.05 * cos(π * r^2) + 0.02 * sin(2π * z))),
             p=1 + 0.05 * cos(2π * r^2 + 1) + 0.02 * cos(2π * z),
             T_ion=1 + 0.1 * cos(π * r^2) * sin(2π * z), Y=(s, 1 - s))
    end
    function face_rates(beta_t, n; pressure=true)
        bc = NSCBCInflowBC(u=(0.0, 0.0, -0.3), T_ion=1.0, Y=[0.6, 0.4], eta_u=0.0,
                           eta_T=0.0, eta_t=0.0, eta_Y=0.0, beta_t=beta_t)
        solver = build_solver(Problem(domain=((0.0, 1.0), (0.0, 2π), (0.0, 1.0)),
                                      metric=CylindricalMetric(),
                                      bcs=((AxisBC(), SlipWallBC()), per3[2],
                                      (NSCBCOutflowBC(pinf=1.0), bc)),
                                      eos=eos, transport=ConstantTransport(mu0=0.0),
                                      ic=nothing),
                              Numerics(n_global=(n, 1, n),
                                       art=ArtificialProperties(enabled=false)))
        Q = allocate_state(solver)
        initialize!(solver, Q, ic)
        apply_bcs!(solver, Q)
        dQ = zero(Q)
        compute_rhs!(solver, Q, dQ)
        m = solver.equations.i_mom
        ie = solver.equations.i_energy
        dpr = solver.tmp_a
        CompactLES.deriv_along!(dpr, solver.p, solver, 1, 1)
        acoustic = radial = 0.0
        for i in 1:n
            I = padded_index(solver, i, 1, n)
            ρ = solver.rho[I]; c = solver.c[I]
            u = (solver.u[I], solver.v[I], solver.w[I])
            ρt = dQ[I, 1] + dQ[I, 2]
            ut = ntuple(a -> (dQ[I, m[a]] - u[a] * ρt) / ρ, 3)
            pt = (γ - 1) * (dQ[I, ie] - sum(abs2, u) / 2 * ρt - ρ * sum(u .* ut))
            acoustic = max(acoustic, abs(pt - ρ * c * ut[3]))
            share = pressure ? (1 - abs(u[3]) / c) / ρ : 0.0
            radial = max(radial, abs(ut[1] + share * solver.inv_h[1][I] * dpr[I]))
        end
        return acoustic, radial
    end
    full = face_rates(1.0, 24)
    fine = face_rates(1.0, 48)
    lodi = face_rates(0.0, 24; pressure=false)
    @test all(lodi .> (0.1, 0.1))
    @test all(full ./ lodi .< 1e-2)
    @test all(full ./ fine .> 8)
end

@testset "NSCBC outflow: the r-z face freezes the incoming wave beside the axis" begin
    # The inflow test above with the stream reversed, leaving through the top
    # face under sigma = 0 and beta_t = 1, where the imposed amplitude is −𝒯
    # and the incoming acoustic rate p − ρc u_z vanishes up to the truncation
    # gap. With the collapsed θ's u_r/r in 𝒯 the rate falls with the
    # resolution (measured ×14 from 24 to 48 nodes); without it the rate stays
    # at half its beta_t = 0 value.
    γ = 1.4
    ic(r, θ, z) = Prim(u=(0.1 * r * (1 - r^2) * (1 + 0.5 * cos(2π * z)), 0.0,
                          0.3 + 0.05 * cos(π * r^2) + 0.02 * sin(2π * z)),
                       p=1 + 0.05 * cos(2π * r^2 + 1) + 0.02 * cos(2π * z),
                       T_ion=1 + 0.1 * cos(π * r^2) * sin(2π * z))
    function face_rate(beta_t, n)
        solver = build_solver(Problem(domain=((0.0, 1.0), (0.0, 2π), (0.0, 1.0)),
                                      metric=CylindricalMetric(),
                                      bcs=((AxisBC(), SlipWallBC()), per3[2],
                                      (NSCBCInflowBC(u=(0.0, 0.0, 0.3), T_ion=1.0),
                                       NSCBCOutflowBC(pinf=1.0, sigma=0.0, beta_t=beta_t))),
                                      eos=IdealSpecies("gas"; gamma=γ, R=1.0),
                                      transport=ConstantTransport(mu0=0.0), ic=nothing),
                              Numerics(n_global=(n, 1, n),
                                       art=ArtificialProperties(enabled=false)))
        Q = allocate_state(solver)
        initialize!(solver, Q, ic)
        apply_bcs!(solver, Q)
        dQ = zero(Q)
        compute_rhs!(solver, Q, dQ)
        m = solver.equations.i_mom
        ie = solver.equations.i_energy
        acoustic = 0.0
        for i in 1:n
            I = padded_index(solver, i, 1, n)
            ρ = solver.rho[I]; c = solver.c[I]
            u = (solver.u[I], solver.v[I], solver.w[I])
            ρt = dQ[I, 1]
            ut = ntuple(a -> (dQ[I, m[a]] - u[a] * ρt) / ρ, 3)
            pt = (γ - 1) * (dQ[I, ie] - sum(abs2, u) / 2 * ρt - ρ * sum(u .* ut))
            acoustic = max(acoustic, abs(pt - ρ * c * ut[3]))
        end
        return acoustic
    end
    full = face_rate(1.0, 24)
    lodi = face_rate(0.0, 24)
    @test lodi > 0.1
    @test full / lodi < 1e-2
    @test full / face_rate(1.0, 48) > 8
end

@testset "validate_bc: NSCBC restrictions are setup errors" begin
    # Both restrictions are setup errors rather than documented caveats: left
    # unchecked, an angular face would fail nothing, since the wave analysis
    # drops the curvature terms carried by grad_u[d,d], and the run would
    # complete with a wrong answer.
    two = IdealMixture([IdealSpecies{Float64}("a", 1.0, 1.4),
                        IdealSpecies{Float64}("b", 2.0, 1.6)])
    wall = (SlipWallBC(), SlipWallBC())
    cyl(θbc) = build_solver(
        Problem(domain=((1.0, 1.0 + 1.0), (0.0, 0.0 + 1.0), (0.0, 0.0 + 1.0)),
                bcs=(wall, θbc, per3[3]), metric=CylindricalMetric(), ic=nothing),
        Numerics(n_global=(12, 12, 12), art=ArtificialProperties(enabled=false)))
    # r is a length under CylindricalMetric, θ is not.
    @test cyl(wall) isa Solver
    @test_throws ErrorException cyl((NSCBCOutflowBC(pinf=1.0), SlipWallBC()))
    @test_throws ErrorException cyl((SlipWallBC(),
                                     NSCBCInflowBC(u=(0.0, 0.1, 0.0), T_ion=1.0)))
    # Composition length, checked once at setup rather than on every RHS call.
    cart(Y) = build_solver(Problem(domain=((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)),
                                   bcs=((NSCBCInflowBC(u=(0.3, 0.0, 0.0), T_ion=1.0, Y=Y),
                                    NSCBCOutflowBC(pinf=1.0)), per3[2], per3[3]),
                                   eos=two, ic=nothing),
                           Numerics(n_global=(12, 12, 12),
                                    art=ArtificialProperties(enabled=false)))
    @test cart([0.4, 0.6]) isa Solver
    @test_throws ErrorException cart([1.0])
    @test_throws ErrorException cart([0.2, 0.3, 0.5])
end
