# Serial regressions for the wall flux contracts: the no-slip wall's
# impermeable, noncatalytic, adiabatic-or-isothermal contract and the slip
# wall's symmetry plane. This file is both included by test/runtests.jl and
# directly runnable.

if !isdefined(@__MODULE__, :CL)
    using MPI
    MPI.Initialized() || MPI.Init(threadlevel=:funneled)
    using CompactLES
    using Test
    const CL = CompactLES
end

import KernelAbstractions

_wf_per() = (PeriodicBC(), PeriodicBC())

function _wf_prepare!(solver, Q; kappa_art=nothing, D_art=nothing)
    apply_bcs!(solver, Q)
    CL.compute_primitives_and_gradients!(solver, Q)
    CL.compute_artificial!(solver, Q)
    if kappa_art !== nothing
        fill!(solver.kappa_art, kappa_art)
    end
    if D_art !== nothing
        foreach(a -> fill!(a, D_art), solver.D_art)
    end
    for d in 1:3
        solver.decomp.active[d] || continue
        CL.deriv_scaled_along!(solver.grad_T_ion[d], solver.T_ion, solver, d, 1)
        for sp in 1:solver.equations.n_species
            CL.deriv_scaled_along!(solver.grad_Y[d, sp], solver.Y[sp], solver, d, 1)
        end
    end
    CL._shared_species_diffusivity(solver) && CL._bulk_gradients!(solver, Q)
    CL.assemble_fluxes!(solver, Q)
    return solver
end

function _wf_correct!(solver, Q)
    for d in 1:3, side in 1:2
        solver.decomp.active[d] || continue
        CL.correct_flux!(solver.bcs[d][side], solver, Q, d, side)
    end
    return solver
end

function _wf_max_plane(a, plane)
    maximum(abs(a[I]) for I in plane; init=zero(eltype(a)))
end

"""
A wall that enforces no state and corrects no flux. On a state whose velocity
already vanishes at the wall it leaves exactly the uncorrected divergence, so
it is the reference the imposed contracts are measured against.
"""
struct UncorrectedWallBC <: CL.BoundaryCondition end

@testset "wall flux: an imposed wall flux reaches compact boundary rows" begin
    # This is deliberately not a compatible insulated solution. Its constant
    # conductive flux survives at both walls under the uncorrected reference
    # and its divergence there is zero. Imposing zero endpoint flux must also
    # move coupled compact rows, rather than patching only the endpoint
    # energy RHS.
    T = Float64
    eos = IdealMixture(IdealSpecies(T, "gas"; R=one(T), gamma=T(1.4)))
    transport = Transport{T}(mu0=T(0.01), Pr=T(0.8), Sc=T(0.7))
    function audit(bc)
        sol = Solver(n_global=(33, 1, 1), L_domain=(one(T), one(T), one(T)),
                     bcs=((bc, bc), _wf_per(), _wf_per()), eos=eos,
                     transport=transport,
                     art=ArtParams{T}(enabled=false), deriv=lele_d1_6(T),
                     filt=compact_filter(T(0.45), T))
        Q = allocate_state(sol)
        initialize!(sol, Q, (x, y, z) -> Prim(rho=one(T), p=one(T) + T(0.1)*x,
                                               u=(zero(T), zero(T), zero(T))))
        rhs = zero(Q)
        apply_bcs!(sol, Q)
        compute_rhs!(sol, Q, rhs)
        return sol, rhs
    end
    sr, dQr = audit(UncorrectedWallBC())
    ie = sr.equations.i_energy
    @test abs(sr.flux[1, ie][first(CL.wallplane(sr.decomp, 1, 1))]) > T(1e-3)
    for bc in (NoSlipWallBC(), SlipWallBC())
        s, dQ = audit(bc)
        for side in 1:2
            plane = CL.wallplane(s.decomp, 1, side)
            @test all(I -> all(s.flux[1, sp][I] == 0
                               for sp in 1:s.equations.n_species), plane)
            @test all(I -> s.flux[1, ie][I] == 0, plane)
        end
        # i=2 is not a wall node. Its change proves the corrected boundary
        # flux entered the compact divergence solve rather than an
        # endpoint-only fix.
        I2 = gidx(s, 2, 1, 1)
        @test abs(dQ[I2, ie] - dQr[I2, ie]) > T(1e-5)
        @test isfinite(dQ[I2, ie])
    end
end

@testset "no-slip wall flux: faces, corners, transport channels and EOS" begin
    ideal(T) = IdealMixture((IdealSpecies(T, "a"; R=T(1), gamma=T(1.4)),
                             IdealSpecies(T, "b"; R=T(0.7), gamma=T(1.3))))
    nasa(T) = Nasa9Mixture((CL.nasa9_constant_cp(T, "a", T(1), T(3.5)),
                            CL.nasa9_constant_cp(T, "b", T(0.7), T(3.2))))
    cases = ((T, d, channel, d == 2 ? nasa : ideal)
             for T in (Float32, Float64) for d in 1:3
             for channel in (:fickian, :bulk, :partial_density))
    for (T, d, channel, eosfn) in cases
        eos = eosfn(T)
        Tw = T(315)
        walls = (NoSlipWallBC(Twall=Tw), NoSlipWallBC(Twall=Tw))
        bcs = (walls, walls, walls)
        s = Solver(n_global=(12, 12, 12), L_domain=(one(T), one(T), one(T)),
                   bcs=bcs, eos=eos,
                   transport=Transport{T}(mu0=T(0.02), Pr=T(0.75), Sc=T(0.6)),
                   art=ArtParams{T}(enabled=true, species_flux=channel),
                   deriv=lele_d1_6(T), filt=compact_filter(T(0.45), T))
        Q = allocate_state(s)
        initialize!(s, Q, (x, y, z) -> begin
            q = (x, y, z)[d]
            Y1 = T(0.35) + T(0.08)*q
            Prim(Y=(Y1, one(T)-Y1), rho=T(1.1) + T(0.05)*q,
                 T_ion=T(300) + T(30)*q,
                 u=(T(0.2)+T(0.03)*x, T(-0.1)+T(0.02)*y, T(0.15)-T(0.01)*z))
        end)
        kap = T(0.007)
        _wf_prepare!(s, Q; kappa_art=kap, D_art=T(0.011))
        moms = s.equations.i_mom
        before = Dict((side, m) => copy(s.flux[d, m]) for side in 1:2 for m in moms)
        _wf_correct!(s, Q)
        tol = T === Float32 ? T(2e-5) : T(2e-12)
        for side in 1:2
            plane = CL.wallplane(s.decomp, d, side)
            @test plane !== nothing
            @test all(sp -> all(I -> s.flux[d, sp][I] == zero(T), plane),
                      1:s.equations.n_species)
            heaterr = maximum(abs(s.flux[d, s.equations.i_energy][I] -
                (-(s.transport.mu0 * s.cp_mix[I] / s.transport.Pr +
                   s.kappa_art[I]) * s.grad_T_ion[d][I])) for I in plane)
            heatscale = maximum(abs(s.flux[d, s.equations.i_energy][I])
                                for I in plane; init=one(T))
            @test heaterr <= tol * max(one(T), heatscale)
            @test all(m -> all(I -> s.flux[d, m][I] == before[(side, m)][I], plane), moms)
        end
        # All six faces are physical walls, including their intersections.
        @test all(isfinite, s.flux[d, s.equations.i_energy])
        for side in 1:2
            CL.correct_flux!(NoSlipWallBC(), s, Q, d, side)
            plane = CL.wallplane(s.decomp, d, side)
            @test all(I -> s.flux[d, s.equations.i_energy][I] == 0, plane)
        end
    end
end

@testset "slip wall flux: faces, corners, transport channels and EOS" begin
    ideal(T) = IdealMixture((IdealSpecies(T, "a"; R=T(1), gamma=T(1.4)),
                             IdealSpecies(T, "b"; R=T(0.7), gamma=T(1.3))))
    nasa(T) = Nasa9Mixture((CL.nasa9_constant_cp(T, "a", T(1), T(3.5)),
                            CL.nasa9_constant_cp(T, "b", T(0.7), T(3.2))))
    cases = ((T, d, channel, d == 2 ? nasa : ideal)
             for T in (Float32, Float64) for d in 1:3
             for channel in (:fickian, :bulk, :partial_density))
    for (T, d, channel, eosfn) in cases
        walls = (SlipWallBC(), SlipWallBC())
        # Every face is a slip wall, so their intersections are exercised too.
        s = Solver(n_global=(12, 12, 12), L_domain=(one(T), one(T), one(T)),
                   bcs=(walls, walls, walls), eos=eosfn(T),
                   transport=Transport{T}(mu0=T(0.02), Pr=T(0.75), Sc=T(0.6)),
                   art=ArtParams{T}(enabled=true, species_flux=channel),
                   deriv=lele_d1_6(T), filt=compact_filter(T(0.45), T))
        Q = allocate_state(s)
        initialize!(s, Q, (x, y, z) -> begin
            q = (x, y, z)[d]
            Y1 = T(0.35) + T(0.08)*q
            Prim(Y=(Y1, one(T)-Y1), rho=T(1.1) + T(0.05)*q,
                 T_ion=T(300) + T(30)*q,
                 u=(T(0.2)+T(0.03)*x, T(-0.1)+T(0.02)*y, T(0.15)-T(0.01)*z))
        end)
        _wf_prepare!(s, Q; kappa_art=T(0.007), D_art=T(0.011))
        moms = s.equations.i_mom
        normal = moms[d]
        before = Dict((side, m) => copy(s.flux[d, m]) for side in 1:2 for m in moms)
        _wf_correct!(s, Q)
        ie = s.equations.i_energy
        for side in 1:2
            plane = CL.wallplane(s.decomp, d, side)
            @test plane !== nothing
            @test all(sp -> all(I -> s.flux[d, sp][I] == zero(T), plane),
                      1:s.equations.n_species)
            @test all(I -> s.flux[d, ie][I] == zero(T), plane)
            for m in moms
                if m == normal
                    # Wall traction: present before the hook and unchanged.
                    @test _wf_max_plane(before[(side, m)], plane) > zero(T)
                    @test all(I -> s.flux[d, m][I] == before[(side, m)][I], plane)
                else
                    @test all(I -> s.flux[d, m][I] == zero(T), plane)
                end
            end
        end
        @test all(isfinite, s.flux[d, ie])
    end
end

@testset "slip wall flux: the symmetry plane reproduces its periodic mirror" begin
    # A wall run whose data are even in rho, p and the tangential velocity
    # and odd in the normal velocity is the restriction of a periodic run on
    # the doubled domain. Under physical viscosity an uncorrected wall leaves
    # a conductive heat flux across the symmetry plane and a shear traction
    # on it; the contract removes both, and the difference from the mirror
    # falls at the closure's order. The evolution orders are measured in
    # test/convergence.jl.
    mu, tf = 0.005, 0.2
    prof(x) = (1 + 0.05cospi(x), 0.05sinpi(x), 0.05cospi(x))
    function line(N, bc, L, bcs)
        s = Solver(n_global=(N, 1, 1), L_domain=(L, 1.0, 1.0), bcs=bcs,
                   transport=Transport(mu0=mu, Pr=0.7), art=ArtParams(enabled=false),
                   deriv=lele_d1_6(), filter_interval=0, cfl=0.25)
        Q = allocate_state(s)
        initialize!(s, Q, (x, y, z) -> begin
            rho, u, v = prof(x)
            Prim(rho=rho, u=(u, v, 0.0), p=rho^1.4)
        end)
        run!(s, Q; tfinal=tf, nmax=2000)
        s, Q
    end
    errs = Float64[]
    for N in (49, 97)
        wall = (SlipWallBC(), SlipWallBC())
        s, Q = line(N, wall, 1.0, (wall, _wf_per(), _wf_per()))
        m, Qm = line(2(N - 1), PeriodicBC(), 2.0,
                     (_wf_per(), _wf_per(), _wf_per()))
        push!(errs, maximum(abs(Q[gidx(s, i, 1, 1), 1] - Qm[gidx(m, i, 1, 1), 1])
                            for i in 1:N))
        ie = s.equations.i_energy
        for side in 1:2, I in CL.wallplane(s.decomp, 1, side)
            @test s.flux[1, ie][I] == 0
            @test s.flux[1, s.equations.i_mom[2]][I] == 0
            @test s.flux[1, 1][I] == 0
        end
    end
    @info "Viscous slip wall against its mirror" errs
    # 4.75e-7 and 3.00e-8 measured, a factor of 15.8 for a halving of h.
    # Without the flux contract the pair reads 2.43e-6 and 1.50e-6, a factor
    # of 1.6.
    @test errs[2] < errs[1] / 8
    @test errs[2] < 5e-8
end

@testset "wall flux: SwitchableBC forwards its active condition" begin
    # An isothermal no-slip wall keeps the conductive heat flux and a slip
    # wall removes it, so the two contracts are distinguishable at the same
    # wall node.
    T = Float64
    bc = SwitchableBC(NoSlipWallBC(Twall=1.0), SlipWallBC())
    s = Solver(n_global=(17, 1, 1), L_domain=(1.0, 1.0, 1.0),
               bcs=((bc, NoSlipWallBC()), _wf_per(), _wf_per()),
               transport=Transport(mu0=0.01), art=ArtParams(enabled=false))
    Q = allocate_state(s)
    initialize!(s, Q, (x, y, z) -> Prim(rho=1.0, T_ion=1.0 + 0.2x))
    _wf_prepare!(s, Q)
    I = first(CL.wallplane(s.decomp, 1, 1))
    ie = s.equations.i_energy
    heat = -(s.transport.mu0 * s.cp_mix[I] / s.transport.Pr) * s.grad_T_ion[1][I]
    @test abs(heat) > 1e-4
    CL.correct_flux!(bc, s, Q, 1, 1)
    @test s.flux[1, ie][I] ≈ heat rtol = 1e-12
    switch!(bc)
    CL.correct_flux!(bc, s, Q, 1, 1)
    @test s.flux[1, ie][I] == 0
end

@testset "no-slip wall flux: compatible insulated manufactured fields" begin
    T = Float64
    mu, Pr, Sc = T(0.015), T(0.8), T(0.6)
    eos = IdealMixture((IdealSpecies(T, "a"; R=one(T), gamma=T(1.4)),
                        IdealSpecies(T, "b"; R=one(T), gamma=T(1.4))))
    walls = (NoSlipWallBC(), NoSlipWallBC())
    s = Solver(n_global=(65, 1, 1), L_domain=(one(T), one(T), one(T)),
               bcs=(walls, _wf_per(), _wf_per()), eos=eos,
               transport=Transport{T}(mu0=mu, Pr=Pr, Sc=Sc),
               art=ArtParams{T}(enabled=false), deriv=lele_d1_6(T))
    Q = allocate_state(s)
    initialize!(s, Q, (x, y, z) -> begin
        Y1 = T(0.5) + T(0.1)*cospi(T(2)*x)
        Prim(Y=(Y1, one(T)-Y1), rho=one(T),
             T_ion=one(T) + T(0.08)*cospi(T(2)*x))
    end)
    dQ = zero(Q)
    apply_bcs!(s, Q)
    compute_rhs!(s, Q, dQ)
    ie = s.equations.i_energy
    for side in 1:2
        plane = CL.wallplane(s.decomp, 1, side)
        @test _wf_max_plane(s.flux[1, 1], plane) == 0
        @test _wf_max_plane(s.flux[1, 2], plane) == 0
        @test _wf_max_plane(s.flux[1, ie], plane) == 0
    end
    # Compare only the interior accuracy here. The endpoint enforcement and
    # whole-domain quadrature are reported by separate assertions below.
    kcond = mu * eos.cpk[1] / Pr
    errsY = T[]; errsE = T[]
    for i in 7:59
        x = xcoord(s, 1, i)
        I = gidx(s, i, 1, 1)
        push!(errsY, abs(dQ[I, 1] - (mu/Sc) * (-T(0.4)*T(pi)^2*cospi(T(2)*x))))
        push!(errsE, abs(dQ[I, ie] - kcond * (-T(0.32)*T(pi)^2*cospi(T(2)*x))))
    end
    # Measured 1.32e-5 (species) and 2.77e-5 (energy) at N=65 under the
    # `:neutral3` rows (3.69e-6 and 7.76e-6 under `:cascade3`). The
    # deliberately wider 4e-5 guard detects a changed closure without claiming
    # that this instantaneous check establishes an evolution order.
    @test maximum(errsY) < 4e-5
    @test maximum(errsE) < 4e-5

    # A node-centered trapezoid is not the compact operator's conservation
    # norm. Keep this whole-domain defect visible, but distinct from exact
    # pointwise wall flux enforcement.
    trap(c) = sum((i == 1 || i == 65 ? 0.5 : 1.0) * dQ[gidx(s, i, 1, 1), c]
                  for i in 1:65) / 64
    @test abs(trap(1)) < 2e-5
    @test abs(trap(ie)) < 2e-5
end

# Balance the analytic pressure gradient, leaving the energy equation as pure
# insulated conduction: rho=1, u=0, T=1+A exp(-kappa/cv * k^2*t) cos(k*x).
# This adds an analytic force; it does not overwrite a computed RHS component.
struct WallConductionBalance
    amplitude::Float64
    diffusivity::Float64
end

function CL.add_source!(source::WallConductionBalance, solver, dQ, Q, t)
    decay = exp(-source.diffusivity * 4pi^2 * t)
    for i in 1:solver.decomp.n_local[1]
        x = xcoord(solver, 1, i)
        I = gidx(solver, i, 1, 1)
        dQ[I, solver.equations.i_mom[1]] +=
            -2pi * source.amplitude * decay * sinpi(2x)
    end
    return dQ
end

@testset "no-slip wall flux: compatible insulated conduction evolution" begin
    mu, Pr, amp, tf = 0.015, 0.8, 0.08, 0.002
    eos = IdealMixture(IdealSpecies("gas"; R=1.0, gamma=1.4))
    diffusivity = mu * eos.cpk[1] / (Pr * (eos.cpk[1] - eos.Rk[1]))
    errors = Float64[]
    defects = Float64[]
    for n in (33, 65)
        s = Solver(n_global=(n, 1, 1), L_domain=(1.0, 1.0, 1.0), eos=eos,
                   bcs=((NoSlipWallBC(), NoSlipWallBC()), _wf_per(), _wf_per()),
                   sources=(WallConductionBalance(amp, diffusivity),),
                   transport=Transport(mu0=mu, Pr=Pr),
                   art=ArtParams(enabled=false), filter_interval=0, cfl=0.2)
        Q = allocate_state(s)
        initialize!(s, Q, (x, y, z) -> Prim(rho=1.0, T_ion=1+amp*cospi(2x)))
        ie = s.equations.i_energy
        energy() = sum((i in (1, n) ? 0.5 : 1.0) * Q[gidx(s, i, 1, 1), ie]
                       for i in 1:n) / (n-1)
        E0 = energy()
        run!(s, Q; tfinal=tf, nmax=100)
        @test s.t == tf
        CL.refresh_primitives!(s, Q)
        decay = exp(-diffusivity * 4pi^2 * tf)
        push!(errors, maximum(abs(s.T_ion[gidx(s, i, 1, 1)] -
                         (1+amp*decay*cospi(2xcoord(s, 1, i)))) for i in 1:n))
        push!(defects, abs(energy()-E0))
        compute_rhs!(s, Q, zero(Q))
        @test all(side -> all(I -> s.flux[1, ie][I] == 0,
                               CL.wallplane(s.decomp, 1, side)), 1:2)
    end
    @info "Insulated conduction evolution" errors defects
    # 8.0e-7 and 1.85e-8 at the finer resolution under the `:neutral3` rows
    # (below 5e-7 and 1e-8 under `:cascade3`).
    @test errors[2] < 1.2e-6
    @test errors[2] < errors[1] / 4
    @test defects[2] < 3e-8
end

@testset "no-slip wall flux: compatible species-diffusion evolution" begin
    # Identical species thermodynamics make this a scalar manufactured
    # diffusion problem at exactly uniform rho, p and T. Its cosine has zero
    # wall derivative, so no momentum source is needed and the exact solution
    # is Y1 = 1/2 + A exp(-D k^2 t) cos(kx).
    T = Float64
    mu, Sc = T(0.012), T(0.75)
    eos = IdealMixture((IdealSpecies(T, "a"; R=one(T), gamma=T(1.4)),
                        IdealSpecies(T, "b"; R=one(T), gamma=T(1.4))))
    s = Solver(n_global=(65, 1, 1), L_domain=(one(T), one(T), one(T)),
               bcs=((NoSlipWallBC(), NoSlipWallBC()), _wf_per(), _wf_per()), eos=eos,
               transport=Transport{T}(mu0=mu, Pr=T(0.8), Sc=Sc),
               art=ArtParams{T}(enabled=false), filter_interval=0, cfl=T(0.2))
    Q = allocate_state(s)
    amp, tf = T(0.1), T(0.002)
    initialize!(s, Q, (x, y, z) -> begin
        y1 = T(0.5) + amp*cospi(T(2)*x)
        Prim(Y=(y1, one(T)-y1), rho=one(T), p=one(T))
    end)
    run!(s, Q; tfinal=tf, nmax=100)
    @test s.t == tf
    decay = exp(-(mu/Sc) * T(4pi^2) * tf)
    err = maximum(abs(Q[gidx(s, i, 1, 1), 1] -
                      (T(0.5) + amp*decay*cospi(T(2)*xcoord(s, 1, i))))
                  for i in 1:65)
    # N=65 measures 6.51e-7 in the full-domain max norm (including walls)
    # under the `:neutral3` rows, 2.445e-7 under `:cascade3`.
    @test err < 1e-6
    CL.refresh_primitives!(s, Q)
    @test maximum(abs(s.p[gidx(s, i, 1, 1)] - one(T)) for i in 1:65) < 2e-12
    @test maximum(abs(s.T_ion[gidx(s, i, 1, 1)] - one(T)) for i in 1:65) < 2e-12
end

@testset "no-slip wall flux: isothermal heat exchange and energy balance" begin
    # A sine temperature perturbation vanishes at both prescribed-temperature
    # walls. The initial conductive boundary fluxes have opposite signs. The
    # short evolution's volume-energy change is compared with their trapezoidal
    # time integral; the spatial quadrature defect is stated separately.
    T = Float64
    Tw, mu, Pr = one(T), T(0.01), T(0.8)
    eos = IdealMixture(IdealSpecies(T, "gas"; R=one(T), gamma=T(1.4)))
    s = Solver(n_global=(65, 1, 1), L_domain=(one(T), one(T), one(T)),
               bcs=((NoSlipWallBC(Twall=Tw), NoSlipWallBC(Twall=Tw)),
                    _wf_per(), _wf_per()), eos=eos,
               transport=Transport{T}(mu0=mu, Pr=Pr, Sc=T(0.7)),
               art=ArtParams{T}(enabled=false), filter_interval=0, cfl=T(0.1))
    Q = allocate_state(s)
    initialize!(s, Q, (x, y, z) ->
        Prim(rho=one(T), T_ion=Tw + T(0.05)*sinpi(x)))
    ie = s.equations.i_energy
    weights = [i == 1 || i == 65 ? T(0.5) : one(T) for i in 1:65]
    energy(A) = sum(weights[i] * A[gidx(s, i, 1, 1), ie] for i in 1:65) / T(64)
    function heatflux!()
        apply_bcs!(s, Q)
        dQ = zero(Q)
        compute_rhs!(s, Q, dQ)
        Il = first(CL.wallplane(s.decomp, 1, 1))
        Ih = first(CL.wallplane(s.decomp, 1, 2))
        return s.flux[1, ie][Il], s.flux[1, ie][Ih],
               sum(weights[i] * dQ[gidx(s, i, 1, 1), ie] for i in 1:65) / T(64)
    end
    E0 = energy(Q)
    Fl0, Fh0, rhsint = heatflux!()
    @test Fl0 < 0 < Fh0
    boundary_rate0 = Fl0 - Fh0
    # This is the discrete whole-domain quadrature defect, not wall leakage.
    @test abs(rhsint - boundary_rate0) < 4e-6
    tf = T(2e-5)
    run!(s, Q; tfinal=tf, nmax=20)
    @test s.t == tf
    Fl1, Fh1, _ = heatflux!()
    measured_rate = (energy(Q) - E0) / tf
    boundary_rate = ((Fl0 - Fh0) + (Fl1 - Fh1)) / 2
    @info "Isothermal energy budget" boundary_rate0 rhsint measured_rate boundary_rate
    @test abs(measured_rate - boundary_rate) < 3e-6
    @test all(side -> all(I -> abs(s.T_ion[I]-Tw) < 1e-14,
                           CL.wallplane(s.decomp, 1, side)), 1:2)
end

@testset "wall flux: nonsingular curved metrics and stiffened gas" begin
    cases = ((CylindricalMetric(), (1.0, 2pi, 0.5), (0.2, 0.0, 0.0),
              IdealMixture(IdealSpecies("gas"; R=1.0, gamma=1.4))),
             (SphericalMetric(), (1.0, 1.0, 2pi), (0.3, 0.4, 0.0),
              StiffenedGas(gamma=1.4, p_inf=0.0, cv=2.5)))
    for (metric, extent, origin, eos) in cases, bc in (NoSlipWallBC(), SlipWallBC())
        walls = (bc, bc)
        s = Solver(n_global=(12, 12, 12), L_domain=extent,
                   bcs=(walls, _wf_per(), _wf_per()), metric=metric, origin=origin,
                   eos=eos, transport=Transport(mu0=0.01),
                   art=ArtParams(enabled=false))
        Q = allocate_state(s)
        initialize!(s, Q, (x, y, z) -> Prim(rho=1.0, p=1.0 + 0.03x))
        apply_bcs!(s, Q)
        compute_rhs!(s, Q, zero(Q))
        ie = s.equations.i_energy
        @test all(side -> all(I -> s.flux[1, ie][I] == 0,
                               CL.wallplane(s.decomp, 1, side)), 1:2)
    end
end

@testset "wall flux: threaded and KA-CPU pointwise equality" begin
    for backend in (CPUBackend(), DeviceBackend(KernelAbstractions.CPU())),
        wall in (NoSlipWallBC(Twall=1.1), SlipWallBC())
        s = Solver(n_global=(12, 12, 12), L_domain=(1.0, 1.0, 1.0), backend=backend,
                   bcs=((wall, wall), _wf_per(), _wf_per()),
                   transport=Transport(mu0=0.01),
                   art=ArtParams(enabled=false))
        Q = allocate_state(s)
        initialize!(s, Q, (x, y, z) -> Prim(rho=1.0, T_ion=0.9 + 0.3x,
                                             u=(0.1, -0.2, 0.05)))
        old = CL.FORCE_KA[]
        try
            CL.FORCE_KA[] = false
            _wf_prepare!(s, Q); _wf_correct!(s, Q)
            threaded = [copy(s.flux[1, c]) for c in 1:s.equations.n_cons]
            CL.FORCE_KA[] = true
            _wf_prepare!(s, Q); _wf_correct!(s, Q)
            @test all(c -> s.flux[1, c] == threaded[c], 1:s.equations.n_cons)
        finally
            CL.FORCE_KA[] = old
        end
    end
end

@testset "no-slip wall flux: filtering remains a separate operation" begin
    s = Solver(n_global=(33, 1, 1), L_domain=(1.0, 1.0, 1.0),
               bcs=((NoSlipWallBC(), NoSlipWallBC()), _wf_per(), _wf_per()),
               eos=IdealMixture((IdealSpecies("a"; R=1.0, gamma=1.4),
                                 IdealSpecies("b"; R=1.0, gamma=1.4))),
               transport=Transport(mu0=0.01), art=ArtParams(enabled=false),
               filt=compact_filter(0.35))
    Q = allocate_state(s)
    initialize!(s, Q, (x, y, z) -> Prim(rho=1.0 + 0.02cospi(4x),
        Y=(0.5+0.1cospi(20x), 0.5-0.1cospi(20x)), p=1.0+0.1cospi(20x)))
    Qbefore = copy(Q)
    filter_state!(s, Q)
    @test maximum(abs(Q[I, 1]-Qbefore[I, 1]) for I in CL.interior(s.decomp)) > 1e-4
    budget_change(c) = sum((i in (1, 33) ? 0.5 : 1.0) *
        (Q[gidx(s, i, 1, 1), c]-Qbefore[gidx(s, i, 1, 1), c]) for i in 1:33) / 32
    changes = (species=budget_change(1), energy=budget_change(s.equations.i_energy))
    @info "Filter-only budget changes" changes
    _wf_prepare!(s, Q)
    _wf_correct!(s, Q)
    ie = s.equations.i_energy
    @test all(side -> all(I -> all(c -> s.flux[1, c][I] == 0, (1, 2, ie)),
                           CL.wallplane(s.decomp, 1, side)), 1:2)
end
