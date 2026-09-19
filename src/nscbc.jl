# Navier–Stokes characteristic boundary conditions (Poinsot & Lele 1992):
# subsonic non-reflecting outflow with pressure relaxation, extended to the
# multicomponent conserved layout.
#
# Implementation: LODI wave-amplitude correction of the RHS, not a hard
# state enforcement. The interior scheme produces one-sided normal
# derivatives at boundary planes (compact closure rows), so the conventionally
# computed dQ contains the physically evaluated incoming acoustic wave. The
# correction replaces only that wave: with L computed from one-sided ∂p and
# ∂u_n, the imposed value is L* = K (p − p∞), K = σ (1 − M²) c / L_ref, and
# the difference ΔL = L* − L propagates analytically to the primitive
# time-derivative deltas
#
#   Δ(∂ρ/∂t) = −ΔL / (2c²),   Δ(∂p/∂t) = −ΔL / 2,
#   Δ(∂u_n/∂t) = ±ΔL / (2ρc)   (+ at the high face where L₁ is incoming,
#                                − at the low face where L₅ is incoming),
#
# with mass fractions and transverse velocities untouched. The three are −Δd₁,
# −Δd₂ and −Δd₃ for the LODI deltas listed with the inflow below, and it is Δd₃
# that the code forms, as sgn·ΔL/(2ρc) with sgn = −1 at the high face. These
# map onto the conserved components (species scale with Y_k, energy through
# φ = cv_m / R_m = ∂(ρe)/∂p |_{ρ,Y}, ideal mixtures). Supersonic-outflow
# points (M ≥ 1) receive no correction: all waves leave the domain. Viscous
# terms are left as computed, which is the classical NSCBC approximation. Both
# corrections carry the Yoo & Im transverse terms under a damping weight
# `beta_t`: the outflow on its one incoming wave (the derivation is at its
# point loop), the inflow on every incoming wave (the derivation is at the
# head of its section).
#
# Restricted to faces whose normal has unit scale factor (Cartesian faces,
# cylindrical r/z faces, spherical r faces); on angular faces the stored
# grad_u[d,d] carries curvature contributions and the wave analysis would need
# the metric terms. `validate_bc` at the foot of this file enforces that at
# setup.

"""
    NSCBCOutflowBC(; pinf, sigma=0.25, Lref=0.0, beta_t=-1.0)

Subsonic characteristic outflow. The single incoming acoustic amplitude is
relaxed toward far-field pressure `pinf`; pressure is not imposed pointwise.
`sigma` sets relaxation strength, `Lref <= 0` selects the domain length normal
to the face, and negative `beta_t` selects local-Mach transverse coupling.

Supersonic outflow points receive no correction. The LODI formulation covers
Cartesian faces and radial or axial curvilinear faces, whose normal metric scale
factor is one; [`setup`](@ref) rejects this condition on an angular face.
"""
struct NSCBCOutflowBC{T<:AbstractFloat} <: BoundaryCondition
    pinf::T                  # far-field pressure target
    sigma::T                 # relaxation strength (0.2–0.6 typical)
    Lref::T                  # relaxation length; ≤ 0 → domain length in d
    beta_t::T                # transverse-term coupling (Yoo & Im): fraction
                             # of the transverse contribution carried by the
                             # imposed wave; 0 → plain LODI, 1 → full
                             # accounting, negative → local Mach number
                             # (the Yoo–Im recommendation)
end

function NSCBCOutflowBC(; pinf::Real, sigma::Real=0.25, Lref::Real=0.0,
                        beta_t::Real=-1.0)
    T = typeof(float(pinf))
    return NSCBCOutflowBC{T}(T(pinf), T(sigma), T(Lref), T(beta_t))
end

# No hard state enforcement; the physics enters through the RHS correction.
enforce!(::NSCBCOutflowBC, Q, solver, d, side) = nothing

function correct_rhs!(bc::NSCBCOutflowBC, solver, Q, dQ, d::Int, side::Int)
    # Run distributed solves before the boundary-plane ownership check.
    # `deriv_along!` is collective over the dimension's sub-communicator, so
    # every rank must reach it, including ranks that own none of this plane.
    # Testing `wallplane === nothing` before these calls deadlocks as soon as
    # the boundary-normal dimension is decomposed, i.e. the normal way to run
    # an NSCBC problem, splitting along the flow direction.

    # One-sided coordinate derivative of p along d (full-field call; the
    # closure rows make the boundary values one-sided), scaled to physical.
    deriv_along!(solver.tmp_a, solver.p, solver, d, 1)

    # Transverse pressure gradients for the Yoo & Im correction, only along
    # active transverse dimensions (they vanish in 1-D and collapsed dims).
    # `active` is a global property, so every rank agrees on these branches.
    #
    # `tmp_b` and `sensor_sp` are borrowed here. This is safe because
    # `compute_rhs!` calls the boundary corrections
    # after add_metric_sources!, by which point compute_artificial! has
    # consumed both sensors into mu_art/beta_art/kappa_art/D_art and the flux
    # divergence has finished with tmp_a/tmp_b. See the invariant recorded in
    # the `compute_artificial!` docstring. `sensor` itself is not
    # reused: io.jl exposes it as the `:sensor` output field, so clobbering it
    # would make a post-step dump of the sensor show a pressure derivative.
    t1, t2 = d == 1 ? (2, 3) : d == 2 ? (1, 3) : (1, 2)
    act1 = solver.decomp.active[t1]
    act2 = solver.decomp.active[t2]
    act1 && deriv_along!(solver.tmp_b, solver.p, solver, t1, 1)
    act2 && deriv_along!(solver.sensor_sp, solver.p, solver, t2, 1)

    # Only now may ranks that own no piece of this face drop out.
    plane = wallplane(solver.decomp, d, side)
    plane === nothing && return nothing

    T = eltype(Q)
    Lref = bc.Lref > 0 ? T(bc.Lref) : solver.L_domain[d]
    m = solver.equations.i_mom
    ft = solver.field_tuples
    # Scalars ride in tuples: a splatted kernel-argument tuple longer than 32
    # elements lowers through the dynamic apply and is an InvalidIRError on
    # device (measured on the first NSCBC device run). The bc carries its own
    # element type, fixed by the type of `pinf` at construction, so it is
    # converted to the state's here: an unconverted Float64 field promotes the
    # whole LODI algebra below to Float64 in a Float32 solver.
    plane_pointwise!(_nscbc_outflow_point!, Q, plane,
                     dQ, solver.eos, solver.rho, solver.u, solver.v, solver.w,
                     solver.p, solver.c, solver.T_ion, solver.cp_mix, ft.Y,
                     ft.grad_u, solver.tmp_a, solver.tmp_b, solver.sensor_sp,
                     solver.inv_h[d], solver.inv_h[t1], solver.inv_h[t2],
                     (T(bc.pinf), T(bc.sigma), T(bc.beta_t)), Lref,
                     side == 2, (act1, act2), (d, t1, t2), m,
                     solver.equations.i_energy, solver.equations.n_species)
    return nothing
end

@inline function _nscbc_outflow_point!(dQ, eos, rho, u, v, w, p_a, c_a, T_a,
                                       cp_a, Y, grad_u, dp_n, dp_t1, dp_t2,
                                       ih_d, ih_t1, ih_t2, bcp, Lref, hiface,
                                       acts, dts, m, i_energy, n_species,
                                       o1, o2, o3, i, j, k)
    @inbounds begin
        pinf, sigma, beta_t = bcp
        act1, act2 = acts
        d, t1, t2 = dts
        I = CartesianIndex(i + o1, j + o2, k + o3)
        T = eltype(rho)
        ρ = rho[I]
        c = c_a[I]
        p = p_a[I]
        uv = (u[I], v[I], w[I])
        un = uv[d]
        Ma = abs(un) / c
        Ma < 1 || return nothing      # supersonic: all waves outgoing
        sgn = hiface ? -one(T) : one(T) # sign of Δd3 (see header)
        dpn = ih_d[I] * dp_n[I]
        dun = grad_u[d, d][I]
        Lcomp = hiface ? (un - c) * (dpn - ρ * c * dun) :
                         (un + c) * (dpn + ρ * c * dun)
        # Transverse contribution to the incoming characteristic (Yoo & Im
        # 2007). Exact transparency to obliquely incident waves requires the
        # incoming amplitude to carry −𝒯_in (`transverse_in` below, which is a
        # wave amplitude and not a temperature), where (derived from the p and
        # u_n equations projected on the incoming characteristic; ρc² = γp)
        #   𝒯_in = u_t·∇_t p + ρc² ∇_t·u_t ∓ ρc u_t·∇_t u_n
        # with − at a high face (incoming L₁) and + at a low face (L₅),
        # i.e. a coefficient of sgn·ρc in the code's sign convention.
        # β_t blends between plain LODI (0) and full accounting (1); the
        # local Mach number is the recommended damping.
        transverse_in = zero(T)
        if act1
            ut = uv[t1]
            transverse_in += ut * ih_t1[I] * dp_t1[I] +
                  ρ * c * c * grad_u[t1, t1][I] + sgn * ρ * c * ut * grad_u[t1, d][I]
        end
        if act2
            ut = uv[t2]
            transverse_in += ut * ih_t2[I] * dp_t2[I] +
                  ρ * c * c * grad_u[t2, t2][I] + sgn * ρ * c * ut * grad_u[t2, d][I]
        end
        βt = beta_t < 0 ? Ma : beta_t
        K = sigma * (1 - Ma * Ma) * c / Lref
        ΔL = K * (p - pinf) - βt * transverse_in - Lcomp
        Δd1 = ΔL / (2 * c * c)
        Δd2 = ΔL / 2
        Δd3 = sgn * ΔL / (2 * ρ * c)
        # φ = ∂(ρe)/∂p |_{ρ,Y}, from the EOS, not from ideal-gas algebra
        # (physics.jl). For an ideal mixture this is cv_m/R_m as before.
        φ = eos_phi(eos, ρ, p, T_a[I], cp_a[I])
        ke = (uv[1]*uv[1] + uv[2]*uv[2] + uv[3]*uv[3]) / T(2)
        for kk in 1:n_species
            dQ[I, kk] -= Y[kk][I] * Δd1
        end
        for a in 1:3
            dQ[I, m[a]] -= uv[a] * Δd1 + (a == d ? ρ * Δd3 : zero(T))
        end
        dQ[I, i_energy] -= ke * Δd1 + φ * Δd2 + ρ * un * Δd3
    end
    return nothing
end

# ---------------------------------------------------------------------------
# Subsonic characteristic inflow: relax normal velocity, transverse velocity,
# temperature, and composition toward targets by replacing every incoming
# wave amplitude, leaving the single outgoing acoustic wave as computed.
#
# LODI relations used (derived from the primitive time derivatives
# ∂ρ/∂t = −d₁, ∂p/∂t = −d₂, ∂u_n/∂t = −d₃, ∂u_t/∂t = −(d₄, d₅),
# ∂Y_k/∂t = −L_{s,k}, with d₁ = [L₂ + ½(L₅+L₁)]/c², d₂ = ½(L₅+L₁),
# d₃ = (L₅−L₁)/2ρc):
#
#   normal velocity:  ∂u_n/∂t gains −ΔL₅/2ρc (low face) or +ΔL₁/2ρc (high),
#     so L₅* =  η_u ρc²(1−M²)/L (u_n − u∞)   drives u_n → u∞ at a low face,
#        L₁* = −η_u ρc²(1−M²)/L (u_n − u∞)   at a high face;
#   temperature: ∂T_ion/∂t|_{L₂} = +T_ion L₂ / ρc² (entropy changes ρ, not p),
#     so L₂* = η_T ρc³ (T_ion,∞ − T_ion) / (L T_ion);
#   transverse:  ∂u_t/∂t = −L₃,₄, so L₃,₄* = η_t (c/L)(u_t − u_t∞);
#   species:     ∂Y_k/∂t = −L_{s,k}, so L_{s,k}* = η_Y (c/L)(Y_k − Y∞_k).
#
# Transverse terms (Yoo & Im 2007). With the transverse convective terms
# kept, each primitive equation reads ∂φ/∂t = −(d_i + 𝒯_φ), and the
# combination that isolates one incoming amplitude carries the matching
# combination of 𝒯 terms, the transverse contribution of that characteristic:
#
#   acoustic:   ∂(p ± ρc u_n)/∂t = −(L₅,₁ + 𝒯_ac),
#               𝒯_ac = u_t·∇_t p + ρc² ∇_t·u_t ± ρc u_t·∇_t u_n
#               (+ for L₅ at a low face, − for L₁ at a high face);
#   entropy:    ∂(c²ρ − p)/∂t = −(L₂ + 𝒯_en),  𝒯_en = u_t·(c² ∇_t ρ − ∇_t p)
#               (the velocity divergence cancels);
#   transverse: ∂u_t/∂t = −(L₃,₄ + 𝒯_t),  𝒯_t = u_t·∇_t u_t + ∇_t p / ρ;
#   species:    ∂Y_k/∂t = −(L_{s,k} + 𝒯_Y),  𝒯_Y = u_t·∇_t Y_k.
#
# A relaxation imposed on the LODI amplitude alone leaves the rate of the
# relaxed quantity at −(relaxation + 𝒯), so a steady transverse flow at the
# face (a diverging stream, a vortex crossing the plane) holds it off its
# target by 𝒯 over the relaxation rate. Each imposed amplitude therefore
# carries −β_t·𝒯, as the outflow's does: L* = relaxation − β_t 𝒯, with
# β_t = 1 the full accounting, under which the incoming characteristic
# variables follow their targets through the transverse flow, and 0 the plain
# LODI form. The curvature terms of a cylindrical or spherical face are not
# part of 𝒯 (they stay in dQ as computed, like the viscous terms): this is
# the Cartesian form of the transverse terms, applied on a face whose normal
# scale factor is one.
#
# The deltas relative to the physically computed amplitudes then use the
# outflow mapping to conserved components, with the species terms
# entering the energy through ρe = p·φ(Y), ∂φ/∂Y_k = (cv_k R_m − R_k cv_m)/R_m².
# Supersonic points are skipped (full-state DirichletBC is the right tool
# there). Targets are constant or a pointwise stage-time function (`target`).

# Parametric on the target-callback type: a `Union{Nothing,Function}` field is
# abstract, so `bc.target(...)` was a runtime dispatch whose `Prim` result
# inferred as `Any`; since uT/TT/YT are read from it, every downstream
# expression in the point loop went dynamic too. `F` is `Nothing` when no
# target is supplied, which the `bc.target !== nothing` test then resolves at
# compile time.
"""
    NSCBCInflowBC(; u, T_ion, Y=[1.0], eta_u=0.28, eta_T=0.28,
                  eta_t=0.28, eta_Y=0.28, Lref=0.0, beta_t=1.0, target=nothing)

Subsonic characteristic inflow. Incoming acoustic, entropy, transverse, and
species waves relax toward target velocity `u`, temperature `T_ion`, and mass
fractions `Y`; the outgoing acoustic wave remains determined by the interior.

`beta_t` weights the transverse terms (Yoo & Im 2007) that every imposed wave
carries: 1 is the full accounting, under which the relaxed quantities follow
their targets through a transverse flow at the face, 0 the plain LODI
relaxation, and a negative value the local Mach number, the outflow's
damping.

`target` may be a stage-time function `(x, y, z, t) -> Prim` overriding the
constant targets pointwise. Its state must contain temperature and the full
composition. `Lref <= 0` selects the domain length normal to the face. A
structure the inflow is to admit, a vortex or a turbulent inflow, enters
only when the relaxation time `Lref / (eta c)` is short against its passage
time, which the default rates do not give; the error of its imposition then
falls as `1 / eta`.

Use `DirichletBC`, not NSCBC inflow, for a supersonic boundary or one whose
full state is to be forced. As for the outflow, the formulation covers only faces
whose normal metric scale factor is one, and [`setup`](@ref) rejects an angular
face. `Y` is checked against the EOS species count there as well.
"""
struct NSCBCInflowBC{T<:AbstractFloat,F} <: BoundaryCondition
    u::NTuple{3,T}                  # target velocity (coordinate-aligned)
    T_ion::T                        # target temperature
    Y::Vector{T}                    # target composition
    eta_u::T
    eta_T::T
    eta_t::T
    eta_Y::T
    Lref::T                         # ≤ 0 → domain length in d
    beta_t::T                       # transverse-term weight, as the outflow's
    target::F
    # Optional (x₁, x₂, x₃, t) -> Prim overriding the constant targets per
    # point at the RK stage time (the Prim must carry T_ion and the full Y).
end


function NSCBCInflowBC(; u, T_ion::Real, Y=[1.0], eta_u::Real=0.28,
                       eta_T::Real=0.28, eta_t::Real=0.28,
                       eta_Y::Real=0.28, Lref::Real=0.0, beta_t::Real=1.0,
                       target=nothing)
    T = typeof(float(T_ion))
    uT = ntuple(i -> T(u[i]), 3)
    YT = T.(collect(Y))
    return NSCBCInflowBC{T,typeof(target)}(uT, T(T_ion), YT, T(eta_u),
                                           T(eta_T), T(eta_t), T(eta_Y),
                                           T(Lref), T(beta_t), target)
end

enforce!(::NSCBCInflowBC, Q, solver, d, side) = nothing

function correct_rhs!(bc::NSCBCInflowBC, solver, Q, dQ, d::Int, side::Int)
    # Run these distributed solves before the boundary-plane ownership check.
    # Every rank must call them before any rank returns early; see the
    # outflow method above.
    # One-sided coordinate derivatives of p and ρ along d.
    deriv_along!(solver.tmp_a, solver.p, solver, d, 1)
    deriv_along!(solver.tmp_b, solver.rho, solver, d, 1)
    # Coordinate derivatives of p and ρ along the active transverse
    # dimensions, for the transverse terms, and only when those are weighted
    # at all. `beta_t` and `active` are setup constants, identical on every
    # rank, so no collective sits below a rank-dependent branch; a cleared
    # flag also tells the point body not to read the array, which is
    # unwritten scratch then.
    #
    # `sensor_sp` and the three `grad_T_ion` arrays are borrowed. The outflow
    # method records why the sensor scratch is dead by this point of
    # `compute_rhs!`; the temperature gradients were consumed by
    # `assemble_fluxes!` and the wall flux hooks, both ahead of the
    # divergence, and nothing reads them again before the next evaluation
    # rebuilds them (`max_rate` writes its own directional rates into them
    # first). `sensor` is not touched, for the reason given there.
    t1, t2 = d == 1 ? (2, 3) : d == 2 ? (1, 3) : (1, 2)   # transverse dims
    transverse = bc.beta_t != 0
    act1 = transverse && solver.decomp.active[t1]
    act2 = transverse && solver.decomp.active[t2]
    dp_t1, dr_t1 = solver.sensor_sp, solver.grad_T_ion[1]
    dp_t2, dr_t2 = solver.grad_T_ion[2], solver.grad_T_ion[3]
    act1 && deriv_along!(dp_t1, solver.p, solver, t1, 1)
    act1 && deriv_along!(dr_t1, solver.rho, solver, t1, 1)
    act2 && deriv_along!(dp_t2, solver.p, solver, t2, 1)
    act2 && deriv_along!(dr_t2, solver.rho, solver, t2, 1)

    plane = wallplane(solver.decomp, d, side)
    plane === nothing && return nothing

    T = eltype(Q)
    Lref = bc.Lref > 0 ? T(bc.Lref) : solver.L_domain[d]
    # The bc carries its own element type, fixed by the type of `T_ion` at
    # construction, and `Prim` targets are Float64 by definition. Both are
    # converted to the state's type before any arithmetic: an unconverted
    # Float64 target or relaxation rate promotes the whole LODI algebra below
    # to Float64 in a Float32 solver.
    #
    # The scalars ride in tuples. With the six derivative arrays of the
    # transverse terms, a flat argument list would pass the 32-element splat
    # limit recorded at the outflow launch.
    coef = (T(bc.eta_u), T(bc.eta_T), T(bc.eta_t), T(bc.eta_Y), T(bc.beta_t),
            Lref)
    flags = (side == 1, act1, act2)
    dts = (d, t1, t2)
    eq = (solver.equations.i_mom, solver.equations.i_energy,
          solver.equations.n_species)
    if bc.target === nothing
        # Constant targets: one launchable plane body, with the composition
        # tuple and field collections materialized at launch.
        ft = solver.field_tuples
        YT = map(T, (bc.Y...,))   # splat first: `T.(bc.Y)` would allocate
        plane_pointwise!(_nscbc_inflow_point!, Q, plane,
                         dQ, solver.eos, solver.rho, solver.u, solver.v,
                         solver.w, solver.p, solver.c, solver.T_ion,
                         solver.cp_mix, ft.Y, ft.grad_u, ft.grad_Y,
                         solver.tmp_a, solver.tmp_b, dp_t1, dr_t1, dp_t2, dr_t2,
                         solver.inv_h[d], solver.inv_h[t1], solver.inv_h[t2],
                         (map(T, bc.u), T(bc.T_ion)), YT, coef, flags, dts, eq)
        return nothing
    end
    # A pointwise `target` is an arbitrary host closure; the loop stays on the
    # host, which a device-resident patch cannot serve yet. The per-point
    # algebra is the launched body's, `_nscbc_inflow_apply!`, with the
    # targets evaluated per point ahead of it.
    _cpu_storage(Q) ||
        error("NSCBCInflowBC with a pointwise target is host-only; use " *
              "constant targets on a DeviceBackend")
    tnow = solver.tstage
    @inbounds for I in plane
        i1, i2, i3 = interior_index(solver, I)
        pr = bc.target(xcoord(solver, 1, i1), xcoord(solver, 2, i2),
                       xcoord(solver, 3, i3), tnow)
        isnan(pr.T_ion) && error("NSCBCInflowBC target must specify T_ion")
        targets = (map(T, pr.u), T(pr.T_ion))
        YT = map(T, pr.Y)
        _nscbc_inflow_apply!(dQ, solver.eos, solver.rho, solver.u, solver.v,
                             solver.w, solver.p, solver.c, solver.T_ion,
                             solver.cp_mix, solver.Y, solver.grad_u,
                             solver.grad_Y, solver.tmp_a, solver.tmp_b,
                             dp_t1, dr_t1, dp_t2, dr_t2, solver.inv_h[d],
                             solver.inv_h[t1], solver.inv_h[t2], targets, YT,
                             coef, flags, dts, eq, I)
    end
    return nothing
end

@inline function _nscbc_inflow_point!(dQ, eos, rho, u, v, w, p_a, c_a, T_a,
                                      cp_a, Y, grad_u, grad_Y, dp_n, dr_n,
                                      dp_t1, dr_t1, dp_t2, dr_t2, ih_d, ih_t1,
                                      ih_t2, targets, YT, coef, flags, dts, eq,
                                      o1, o2, o3, i, j, k)
    I = CartesianIndex(i + o1, j + o2, k + o3)
    _nscbc_inflow_apply!(dQ, eos, rho, u, v, w, p_a, c_a, T_a, cp_a, Y, grad_u,
                         grad_Y, dp_n, dr_n, dp_t1, dr_t1, dp_t2, dr_t2, ih_d,
                         ih_t1, ih_t2, targets, YT, coef, flags, dts, eq, I)
    return nothing
end

# The inflow correction at one padded index `I`: the launched body and the
# host loop of a pointwise target share it. `targets` is `(u∞, T∞)` and `YT`
# the target composition tuple; `coef` is `(η_u, η_T, η_t, η_Y, β_t, L_ref)`,
# `flags` is `(lowface, act1, act2)`, `dts` is `(d, t1, t2)`, and `eq` is
# `(i_mom, i_energy, n_species)`. `dp_*`/`dr_*` are coordinate derivatives of
# p and ρ along the face normal and the two transverse dimensions, read only
# where the matching `act` flag is set.
@inline function _nscbc_inflow_apply!(dQ, eos, rho, u, v, w, p_a, c_a, T_a,
                                      cp_a, Y, grad_u, grad_Y, dp_n, dr_n,
                                      dp_t1, dr_t1, dp_t2, dr_t2, ih_d, ih_t1,
                                      ih_t2, targets, YT, coef, flags, dts, eq,
                                      I)
    @inbounds begin
        uT, TT = targets
        eta_u, eta_T, eta_t, eta_Y, beta_t, Lref = coef
        lowface, act1, act2 = flags
        d, t1, t2 = dts
        m, i_energy, n_species = eq
        T = eltype(rho)
        ρ = rho[I]
        c = c_a[I]
        p = p_a[I]
        Tp = T_a[I]
        uv = (u[I], v[I], w[I])
        un = uv[d]
        Ma = abs(un) / c
        Ma < 1 || return nothing
        ih = ih_d[I]
        dpn = ih * dp_n[I]
        drn = ih * dr_n[I]
        dun = grad_u[d, d][I]
        K = c / Lref
        # Physically computed amplitudes.
        L1c = (un - c) * (dpn - ρ * c * dun)
        L5c = (un + c) * (dpn + ρ * c * dun)
        L2c = un * (c * c * drn - dpn)
        # Transverse contributions of the incoming characteristics, derived
        # at the head of this section. `sgn` is the sign of the ρc u_t·∇_t u_n
        # term of the acoustic one: + for L₅ at a low face, − for L₁ at a
        # high face, the outflow's convention.
        sgn = lowface ? one(T) : -one(T)
        tr_ac = zero(T)
        tr_en = zero(T)
        tr_t1 = zero(T)
        tr_t2 = zero(T)
        if act1
            ut = uv[t1]
            dpt = ih_t1[I] * dp_t1[I]
            drt = ih_t1[I] * dr_t1[I]
            tr_ac += ut * dpt + ρ * c * c * grad_u[t1, t1][I] +
                     sgn * ρ * c * ut * grad_u[t1, d][I]
            tr_en += ut * (c * c * drt - dpt)
            tr_t1 += ut * grad_u[t1, t1][I] + dpt / ρ
            tr_t2 += ut * grad_u[t1, t2][I]
        end
        if act2
            ut = uv[t2]
            dpt = ih_t2[I] * dp_t2[I]
            drt = ih_t2[I] * dr_t2[I]
            tr_ac += ut * dpt + ρ * c * c * grad_u[t2, t2][I] +
                     sgn * ρ * c * ut * grad_u[t2, d][I]
            tr_en += ut * (c * c * drt - dpt)
            tr_t1 += ut * grad_u[t2, t1][I]
            tr_t2 += ut * grad_u[t2, t2][I] + dpt / ρ
        end
        βt = beta_t < 0 ? Ma : beta_t
        # Imposed incoming amplitudes (outgoing one kept as computed), each
        # the relaxation less the weighted transverse contribution.
        rel_ac = eta_u * ρ * c * c * (1 - Ma * Ma) / Lref * (un - uT[d])
        ΔL1 = lowface ? zero(T) : (-rel_ac - βt * tr_ac - L1c)
        ΔL5 = lowface ? (rel_ac - βt * tr_ac - L5c) : zero(T)
        ΔL2 = eta_T * ρ * c^3 * (TT - Tp) / (Lref * Tp) - βt * tr_en - L2c
        Δd1 = ΔL2 / (c * c) + (ΔL5 + ΔL1) / (2 * c * c)
        Δd2 = (ΔL5 + ΔL1) / 2
        Δd3 = (ΔL5 - ΔL1) / (2 * ρ * c)
        Δd4 = eta_t * K * (uv[t1] - uT[t1]) - βt * tr_t1 - un * grad_u[d, t1][I]
        Δd5 = eta_t * K * (uv[t2] - uT[t2]) - βt * tr_t2 - un * grad_u[d, t2][I]
        # Mixture quantities for the energy mapping, through the EOS contract.
        cpm = cp_a[I]
        φ = eos_phi(eos, ρ, p, Tp, cpm)
        ke = (uv[1]*uv[1] + uv[2]*uv[2] + uv[3]*uv[3]) / T(2)
        ΣφY = zero(T)
        for kk in 1:n_species
            tr_Y = zero(T)
            act1 && (tr_Y += uv[t1] * grad_Y[t1, kk][I])
            act2 && (tr_Y += uv[t2] * grad_Y[t2, kk][I])
            ΔLs = eta_Y * K * (Y[kk][I] - YT[kk]) - βt * tr_Y -
                  un * grad_Y[d, kk][I]
            dQ[I, kk] -= Y[kk][I] * Δd1 + ρ * ΔLs
            ΣφY += eos_dphi_dY(eos, kk, ρ, p, Tp, cpm) * ΔLs
        end
        for a in 1:3
            extra = a == d ? ρ * Δd3 : a == t1 ? ρ * Δd4 : ρ * Δd5
            dQ[I, m[a]] -= uv[a] * Δd1 + extra
        end
        dQ[I, i_energy] -= ke * Δd1 + φ * Δd2 + p * ΣφY +
                     ρ * (un * Δd3 + uv[t1] * Δd4 + uv[t2] * Δd5)
    end
    return nothing
end

# ---------------------------------------------------------------------------
# Setup-time validation of both conditions.
#
# The angular-face restriction is enforced here rather than left to the header
# above and the two docstrings: without the check an NSCBC on a cylindrical θ
# face or a spherical θ/φ face would run to completion on a wave analysis
# missing the metric terms that the stored grad_u[d,d] carries there. Nothing
# would fail and the answer would be wrong, so it is a setup error rather than
# a documented caveat.

function _validate_nscbc_face(metric, d::Int, what::String)
    unit_scalefactor(metric, d) && return nothing
    error("$what on dimension $d: the LODI formulation requires a face whose " *
          "normal metric scale factor is one (a Cartesian face, or a " *
          "cylindrical r/z or spherical r face). Dimension $d is angular under " *
          "$(nameof(typeof(metric))), where the wave analysis would need the " *
          "curvature terms carried by grad_u[$d,$d].")
end

validate_bc(::NSCBCOutflowBC, metric, eos, d::Int, side::Int) =
    _validate_nscbc_face(metric, d, "NSCBCOutflowBC")

function validate_bc(bc::NSCBCInflowBC, metric, eos, d::Int, side::Int)
    _validate_nscbc_face(metric, d, "NSCBCInflowBC")
    # Checked here, not in correct_rhs!, where it ran once per RHS call
    # per rank for the life of the run to catch a setup mistake.
    length(bc.Y) == nspecies(eos) ||
        error("NSCBCInflowBC: target composition has $(length(bc.Y)) entries; " *
              "the EOS has $(nspecies(eos)) species")
    return nothing
end
