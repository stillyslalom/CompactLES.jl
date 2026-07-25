# Solver container and conservative multicomponent Navier–Stokes RHS in
# orthogonal curvilinear coordinates, with collapsed (1-D/2-D) dimensions and
# a regularized cylindrical axis.
#
# Collapsed dimensions (n_global[d] == 1) carry no derivatives, filters, halos,
# or exchanges — but keep their velocity component and all metric source
# terms, which is exactly what axisymmetric (r, z) flow with optional swirl
# requires. Coordinate singularities (cylindrical axis, spherical origin and
# poles — axisymmetric or fully resolved) are regularized
# by a half-offset grid, r_i = (i − ½)h, plus parity conditions: halos below
# the axis are mirror-filled with per-field signs (even scalars/u_z, odd
# u_r/u_θ), interior stencils run all the way to the first node, and the
# implicit LHS coupling to the ghost unknown is folded analytically onto the
# diagonal (per solution parity σg: derivatives flip field parity, filters
# preserve it). No node sits at r = 0 and no scale factor vanishes.

mutable struct Solver{T,Eq<:EquationSet,E<:EOS,M<:Metric,St,Fo,BC,DP,FP,Src}
    decomp::Decomp
    equations::Eq
    eos::E
    transport::Transport{T}
    art::ArtParams{T}
    metric::M
    stretch::St
    folds::Fo                                  # coordinate-singularity folds
    sources::Src
    L_domain::NTuple{3,T}
    origin::NTuple{3,T}
    coord_shift::NTuple{3,T}                # half-cell offset (axis grids)
    h::NTuple{3,T}
    bcs::BC
    deriv_plans::DP
    filter_plans::FP
    pairbuf::Array{T,3}                     # paired-fold combo scratch
    pairout::Array{T,3}                     # paired-fold second-parity result
    cfl::T
    filter_interval::Int
    # primitives (full padded arrays)
    rho::Array{T,3}; u::Array{T,3}; v::Array{T,3}; w::Array{T,3}
    p::Array{T,3};  T_ion::Array{T,3}; c::Array{T,3}; cp_mix::Array{T,3}
    Y::Vector{Array{T,3}}
    # gradients
    grad_u::Matrix{Array{T,3}}              # grad_u[d, j] = physical (∇u)_{dj}
    grad_T_ion::NTuple{3,Array{T,3}}
    grad_Y::Matrix{Array{T,3}}              # grad_Y[d, k]
    # artificial properties and scratch
    mu_art::Array{T,3}; beta_art::Array{T,3}; kappa_art::Array{T,3}
    D_art::Vector{Array{T,3}}
    strain_mag::Array{T,3}; sensor::Array{T,3}; sensor_sp::Array{T,3}
    tmp_a::Array{T,3}; tmp_b::Array{T,3}
    # geometry
    inv_J::Array{T,3}
    area_d::NTuple{3,Array{T,3}}
    inv_h::NTuple{3,Array{T,3}}
    inv_r::Array{T,3}
    cot_over_r::Array{T,3}
    # fluxes flux[d, c]
    flux::Matrix{Array{T,3}}
    t::T
    tstage::T
    step::Int
end

"""
    Solver(; n_global, L_domain, bcs, kwargs...)

See earlier keywords, plus:
- Collapsed dimensions: set `n_global[d] = 1` with `(PeriodicBC(), PeriodicBC())`
  for that dimension; e.g. axisymmetric cylindrical is `n_global = (Nr, 1, Nz)`.
- Cylindrical axis: `bcs[1] = (AxisBC(), <outer bc>)` with
  `metric = CylindricalMetric()`, `origin[1] = 0`, θ collapsed
  (`n_global[2] = 1`), dimension 1 unstretched. The grid is half-offset in r.
"""
function Solver(; n_global::NTuple{3,Int}, L_domain, bcs,
                eos::EOS=single_species(),
                equations=nothing,
                transport::Transport{T}=Transport(),
                art::ArtParams=ArtParams(),
                metric::Metric=CartesianMetric(),
                stretch::NTuple{3,Union{Nothing,Stretch}}=(nothing, nothing, nothing),
                sources=(),
                origin=(0.0, 0.0, 0.0),
                deriv::AbstractCompactScheme=lele_d1_6(),
                filt::AbstractCompactScheme=compact_filter(0.45),
                cfl::Real=0.5, filter_interval::Int=1,
                dims=nothing, n_halo::Int=4) where {T}
    for d in 1:3
        isperiodic(bcs[d][1]) == isperiodic(bcs[d][2]) ||
            error("dimension $d mixes periodic and non-periodic conditions")
        n_global[d] > 1 || (isperiodic(bcs[d][1]) ||
            error("collapsed dimension $d must use (PeriodicBC(), PeriodicBC())"))
    end
    # ---- Coordinate-singularity folds -----------------------------------
    axis   = bcs[1][1] isa AxisBC
    orig1  = bcs[1][1] isa OriginBC
    pole_l = bcs[2][1] isa PoleBC
    pole_h = bcs[2][2] isa PoleBC
    pole_l == pole_h || error("PoleBC must be applied at both ends of θ")
    poles = pole_l
    if axis
        metric isa CylindricalMetric || error("AxisBC requires CylindricalMetric")
        stretch[1] === nothing ||
            error("folded dimensions cannot be stretched")
        abs(Float64(origin[1])) < 1e-14 || error("AxisBC requires origin[1] = 0")
        n_global[2] == 1 || iseven(n_global[2]) ||
            error("resolved-θ AxisBC requires an even θ point count over 2π")
    end
    if orig1
        metric isa SphericalMetric || error("OriginBC requires SphericalMetric")
        stretch[1] === nothing || error("folded dimensions cannot be stretched")
        abs(Float64(origin[1])) < 1e-14 || error("OriginBC requires origin[1] = 0")
        n_global[3] == 1 || iseven(n_global[3]) ||
            error("resolved-φ OriginBC requires an even φ point count over 2π")
        θsum = 2 * Float64(origin[2]) + Float64(L_domain[2])
        n_global[2] == 1 || isapprox(θsum, π; atol=1e-10) ||
            error("OriginBC requires a θ range symmetric about π/2")
    end
    if poles
        metric isa SphericalMetric || error("PoleBC requires SphericalMetric")
        stretch[2] === nothing || error("folded dimensions cannot be stretched")
        abs(Float64(origin[2])) < 1e-14 && isapprox(Float64(L_domain[2]), π; atol=1e-10) ||
            error("PoleBC requires the θ domain (0, π)")
        n_global[3] == 1 || iseven(n_global[3]) ||
            error("resolved-φ PoleBC requires an even φ point count over 2π")
    end
    periodic = ntuple(d -> n_global[d] > 1 ? isperiodic(bcs[d][1]) : true, 3)
    (axis || orig1) && (periodic = (false, periodic[2], periodic[3]))
    poles && (periodic = (periodic[1], false, periodic[3]))
    for d in 1:3
        stretch[d] === nothing || !periodic[d] ||
            error("dimension $d: stretched dimensions must be non-periodic")
    end
    decomp = Decomp(n_global, periodic; dims=dims, n_halo=n_halo)
    Lt = ntuple(d -> T(L_domain[d]), 3)
    # Grid spacing: computational ξ ∈ [0,1] for stretched dims; half-offset
    # r ∈ (0, R] for axis grids (h = R/(N − ½), r₁ = h/2); standard otherwise.
    # Half-offset grids on folded dimensions: fold at the low end only
    # (r: axis/origin) gives h = L/(N − ½); folds at both ends (θ poles)
    # give h = L/N; either way the first node sits at h/2.
    fold_lo_dim = ntuple(d -> (d == 1 && (axis || orig1)) || (d == 2 && poles), 3)
    fold_hi_dim = ntuple(d -> d == 2 && poles, 3)
    h = ntuple(3) do d
        decomp.active[d] || return one(T)
        stretch[d] === nothing || return one(T) / (n_global[d] - 1)
        fold_lo_dim[d] && fold_hi_dim[d] && return Lt[d] / n_global[d]
        fold_lo_dim[d] && return Lt[d] / (n_global[d] - T(0.5))
        periodic[d] ? Lt[d] / n_global[d] : Lt[d] / (n_global[d] - 1)
    end
    coord_shift = ntuple(d -> fold_lo_dim[d] ? h[d] / 2 : zero(T), 3)
    mkd(sch, d; kw...) = plan_direction(decomp, sch, d, h[d]; kw...)
    equations = equations === nothing ? NavierStokes1T(eos) : equations
    equations isa EquationSet || error("equations must be an EquationSet")
    equations.n_species == nspecies(eos) ||
        error("equation set carries $(equations.n_species) species; " *
              "EOS has $(nspecies(eos))")
    n_species = equations.n_species
    n_cons = equations.n_cons
    f() = field(decomp)

    # Fold specs. sigvel/sigflux derivations live in folds.jl and the README.
    function pairspec(pdim, revdim)
        # Degenerate mappings on collapsed dims are identities.
        shift_needed = pdim != 0 && decomp.active[pdim]
        rev_needed = revdim != 0 && decomp.active[revdim]
        (!shift_needed && !rev_needed) && return nothing   # self-paired
        shift_local = !shift_needed || decomp.dims[pdim] == 1
        rev_local = !rev_needed || decomp.dims[revdim] == 1
        if shift_needed
            shift_local || (iseven(decomp.dims[pdim]) &&
                            n_global[pdim] % decomp.dims[pdim] == 0) ||
                error("pairing dim $pdim must be on one rank or split into an " *
                      "even number of uniform blocks")
            shift_local && (iseven(decomp.n_local[pdim]) ||
                error("pairing dim $pdim local extent must be even"))
        end
        if rev_needed && !rev_local
            n_global[revdim] % decomp.dims[revdim] == 0 ||
                error("reversed dim $revdim must split into uniform blocks")
            iseven(decomp.dims[revdim]) ||
                error("reversed dim $revdim needs an even rank count")
        end
        crd = collect(Int, decomp.coords)
        shift_needed && !shift_local &&
            (crd[pdim] = mod(crd[pdim] + decomp.dims[pdim] ÷ 2, decomp.dims[pdim]))
        rev_needed && !rev_local &&
            (crd[revdim] = decomp.dims[revdim] - 1 - crd[revdim])
        partner = Int(MPI.Cart_rank(decomp.comm, Cint.(crd)))
        loc = shift_local && rev_local
        keep_e = if shift_needed && !shift_local
            decomp.coords[pdim] < decomp.dims[pdim] ÷ 2
        elseif rev_needed && !rev_local
            decomp.coords[revdim] < decomp.dims[revdim] ÷ 2
        else
            true
        end
        PairSpec(shift_needed ? pdim : 0, rev_needed ? revdim : 0,
                 shift_local, rev_local, loc, partner, keep_e,
                 loc ? zeros(0, 0, 0) : field(decomp))
    end
    function foldspec(d, lo, hi, pdim, revdim, sigvel, sigflux)
        dp = (mkd(deriv, d; lo_fold=(lo ? 1 : nothing), hi_fold=(hi ? 1 : nothing)),
              mkd(deriv, d; lo_fold=(lo ? -1 : nothing), hi_fold=(hi ? -1 : nothing)))
        fp = (mkd(filt, d; lo_fold=(lo ? 1 : nothing), hi_fold=(hi ? 1 : nothing)),
              mkd(filt, d; lo_fold=(lo ? -1 : nothing), hi_fold=(hi ? -1 : nothing)))
        FoldSpec(d, lo, hi, pairspec(pdim, revdim), sigvel, sigflux, dp, fp)
    end
    folds = (nothing, nothing, nothing)
    if axis
        sv = (-1, -1, 1)
        folds = (foldspec(1, true, false, 2, 0, sv,
                          flux_parities(equations, sv, 1, -1)),
                 folds[2], folds[3])           # A₁ = r is odd
    elseif orig1
        sv = (-1, 1, -1)
        folds = (foldspec(1, true, false, 3, 2, sv,
                          flux_parities(equations, sv, 1, 1)),
                 folds[2], folds[3])           # A₁ = r² sinθ is even
    end
    if poles
        sv = (1, -1, -1)
        # Fold along θ: flux parities use σ(u_θ); A₂ = r sinθ is odd in θ.
        sf2 = flux_parities(equations, sv, 2, -1)
        folds = (folds[1], foldspec(2, true, true, 3, 0, sv, sf2), folds[3])
    end
    haspair = any(fold -> fold !== nothing && fold.pair !== nothing, folds)
    deriv_plans = ntuple(d -> decomp.active[d] && folds[d] === nothing ?
                         mkd(deriv, d) : nothing, 3)
    filter_plans = ntuple(d -> decomp.active[d] && folds[d] === nothing ?
                         mkd(filt, d) : nothing, 3)
    orig = ntuple(d -> stretch[d] === nothing ? T(origin[d]) : zero(T), 3)
    bcs_t = ntuple(d -> (bcs[d][1], bcs[d][2]), 3)
    solver = Solver{T,typeof(equations),typeof(eos),typeof(metric),
                    typeof(stretch),typeof(folds),typeof(bcs_t),
                    typeof(deriv_plans),typeof(filter_plans),typeof(sources)}(
                  decomp, equations, eos, transport, art, metric, stretch, folds, sources,
                  Lt, orig, coord_shift, h,
                  bcs_t,
                  deriv_plans, filter_plans,
                  any(fold -> fold !== nothing, folds) ? field(decomp) : zeros(T, 0, 0, 0),
                  any(fold -> fold !== nothing, folds) ? field(decomp) : zeros(T, 0, 0, 0),
                  T(cfl), filter_interval,
                  f(), f(), f(), f(), f(), f(), f(), f(),
                  [f() for _ in 1:n_species],
                  [f() for _ in 1:3, _ in 1:3],
                  (f(), f(), f()),
                  [f() for _ in 1:3, _ in 1:n_species],
                  f(), f(), f(),
                  [f() for _ in 1:n_species],
                  f(), f(), f(), f(), f(),
                  f(), (f(), f(), f()), (f(), f(), f()), f(), f(),
                  [f() for _ in 1:3, _ in 1:n_cons],
                  zero(T), zero(T), 0)
    init_geometry!(solver)
    return solver
end

"Physical coordinate of local interior index i along dimension d."
function xcoord(solver::Solver, d::Int, i::Int)
    ξ = solver.origin[d] + solver.coord_shift[d] +
        (solver.decomp.offset[d] + i - 1) * solver.h[d]
    st = solver.stretch[d]
    return st === nothing ? ξ : st.x(ξ)
end

"Halo-offset CartesianIndex of local interior point (i, j, k)."
function gidx(solver::Solver, i::Int, j::Int, k::Int)
    pad = solver.decomp.n_halo_d
    return CartesianIndex(i + pad[1], j + pad[2], k + pad[3])
end

allocate_state(solver::Solver) = allocate_state(solver.decomp, solver.equations.n_cons)

# --- Operator routing through folds ----------------------------------------

"""
    deriv_along!(out, f, solver, d, σf)

Compact derivative of `f` along active dimension `d`; `σf` is the field's
antipodal sign for the fold on `d` (ignored when there is none). Caller
ensures current halos.
"""
function deriv_along!(out, f, solver::Solver, d::Int, σf::Int)
    fold = solver.folds[d]
    if fold === nothing
        apply_along!(out, solver.deriv_plans[d], f, solver.decomp)
    else
        fold_apply!(out, f, solver, fold, σf; isfilter=false)
    end
    return out
end

"Compact filter of `f` along dimension `d` with antipodal sign `σf`."
function filt_along!(out, f, solver::Solver, d::Int, σf::Int)
    fold = solver.folds[d]
    if fold === nothing
        apply_along!(out, solver.filter_plans[d], f, solver.decomp)
    else
        fold_apply!(out, f, solver, fold, σf; isfilter=true)
    end
    return out
end

# Scale a raw coordinate-derivative field by 1/h_d pointwise (full array).
function _scale_grad!(g, solver, d)
    ih = solver.inv_h[d]
    @threaded length(g) for idx in eachindex(g)
        @inbounds g[idx] *= ih[idx]
    end
    return g
end

# Antipodal signs of velocity and conserved components for the fold (if any)
# on dimension d; scalars, partial densities, and energy are always +1.
vel_parity(solver::Solver, d::Int, j::Int) =
    solver.folds[d] === nothing ? 1 : solver.folds[d].sigvel[j]
cons_parity(solver::Solver, d::Int, c::Int) =
    solver.folds[d] === nothing ? 1 :
    conserved_parity(solver.equations, solver.folds[d].sigvel, c)

assemble_fluxes!(solver::Solver, Q) = _assemble_fluxes!(solver, solver.eos, Q)

function _assemble_fluxes!(solver::Solver{T}, eos, Q) where {T}
    decomp = solver.decomp
    o1, o2, o3 = decomp.n_halo_d
    nx, ny, nz = decomp.n_local
    tr = solver.transport
    mu0 = tr.mu0
    n_species = solver.equations.n_species
    i_energy = solver.equations.i_energy
    m1, m2, m3 = solver.equations.i_mom
    grad_u = solver.grad_u
    gT = solver.grad_T_ion
    gY = solver.grad_Y
    flux = solver.flux
    @threaded nx*ny*nz for k in 1:nz
        @inbounds for j in 1:ny, i in 1:nx
            I = CartesianIndex(i + o1, j + o2, k + o3)
            ρ = solver.rho[I]
            uv = (solver.u[I], solver.v[I], solver.w[I])
            p = solver.p[I]
            Tp = solver.T_ion[I]
            E = Q[I, i_energy]
            μ = mu0 + solver.mu_art[I]
            β = solver.beta_art[I]
            κ = mu0 * solver.cp_mix[I] / tr.Pr + solver.kappa_art[I]
            D0 = mu0 / (tr.Sc * ρ)               # molecular part of each D_k
            divu = grad_u[1, 1][I] + grad_u[2, 2][I] + grad_u[3, 3][I]
            τ11 = μ * (2*grad_u[1,1][I] - (2/3) * divu) + β * divu
            τ22 = μ * (2*grad_u[2,2][I] - (2/3) * divu) + β * divu
            τ33 = μ * (2*grad_u[3,3][I] - (2/3) * divu) + β * divu
            τ12 = μ * (grad_u[1,2][I] + grad_u[2,1][I])
            τ13 = μ * (grad_u[1,3][I] + grad_u[3,1][I])
            τ23 = μ * (grad_u[2,3][I] + grad_u[3,2][I])
            τ = ((τ11, τ12, τ13), (τ12, τ22, τ23), (τ13, τ23, τ33))
            for d in 1:3
                ud = uv[d]
                τd = τ[d]
                # Per-species diffusion with a correction velocity:
                # J_k = −ρ D_k ∇Y_k + ρ Y_k V_c, V_c = Σ_j D_j ∇Y_j,
                # which enforces Σ_k J_k = 0 exactly since ΣY_k = 1.
                Vc = zero(T)
                for sp in 1:n_species
                    Vc += (D0 + solver.D_art[sp][I]) * gY[d, sp][I]
                end
                hdiff = zero(T)              # Σ_k h_k J_{k,d}
                for sp in 1:n_species
                    Dk = D0 + solver.D_art[sp][I]
                    Jkd = ρ * (-Dk * gY[d, sp][I] + solver.Y[sp][I] * Vc)
                    flux[d, sp][I] = ρ * solver.Y[sp][I] * ud + Jkd
                    hdiff += species_enthalpy(eos, sp, Tp) * Jkd
                end
                flux[d, m1][I] = ρ * ud * uv[1] + (d == 1 ? p : zero(T)) - τd[1]
                flux[d, m2][I] = ρ * ud * uv[2] + (d == 2 ? p : zero(T)) - τd[2]
                flux[d, m3][I] = ρ * ud * uv[3] + (d == 3 ? p : zero(T)) - τd[3]
                flux[d, i_energy][I] = (E + p) * ud -
                               (uv[1]*τd[1] + uv[2]*τd[2] + uv[3]*τd[3]) -
                               κ * gT[d][I] + hdiff
            end
        end
    end
    return solver
end

"""
    compute_rhs!(solver, Q, dQ)

Evaluate dQ/dt into the interior of `dQ` from the conserved state `Q`
(boundary conditions should already be enforced on `Q`). Collapsed dimensions
contribute no derivatives; the axis dimension routes through parity-folded
plans with mirror-filled halos.
"""
function compute_rhs!(solver::Solver, Q, dQ)
    decomp = solver.decomp
    exchange_state!(Q, decomp)
    primitives!(solver, Q)
    vel = (solver.u, solver.v, solver.w)
    for jj in 1:3, d in 1:3
        if decomp.active[d]
            deriv_along!(solver.grad_u[d, jj], vel[jj], solver, d, vel_parity(solver, d, jj))
            _scale_grad!(solver.grad_u[d, jj], solver, d)   # 1/h_d incl. stretching Jacobian
        else
            fill!(solver.grad_u[d, jj], 0)
        end
    end
    metric_correct_gradients!(solver, solver.metric)   # additive curvature terms
    compute_artificial!(solver, Q)
    for d in 1:3
        decomp.active[d] || continue
        deriv_along!(solver.grad_T_ion[d], solver.T_ion, solver, d, 1)
        _scale_grad!(solver.grad_T_ion[d], solver, d)
        for sp in 1:solver.equations.n_species
            deriv_along!(solver.grad_Y[d, sp], solver.Y[sp], solver, d, 1)
            _scale_grad!(solver.grad_Y[d, sp], solver, d)
        end
    end
    assemble_fluxes!(solver, Q)
    for d in 1:3
        exchange_dim_batch!(view(solver.flux, d, :), decomp, d)
    end
    o1, o2, o3 = decomp.n_halo_d
    nx, ny, nz = decomp.n_local
    # On an unstretched Cartesian grid every scale factor is exactly 1, so
    # A_d ≡ 1 and inv_J ≡ 1: the A_d·F_d product below is a full-array copy
    # whose result equals its input, and the inv_J multiply is a no-op. Skipping
    # both removes three array streams per (component, dimension) — 45 of them
    # for a 5-component 3-D RHS, in what the phase budget shows is the single
    # largest phase. Curved or stretched grids take the general path unchanged.
    unitgeom = solver.metric isa CartesianMetric && all(isnothing, solver.stretch)
    for c in 1:solver.equations.n_cons
        @threaded nx*ny*nz for k in 1:nz
            @inbounds for j in 1:ny, i in 1:nx
                dQ[i+o1, j+o2, k+o3, c] = 0
            end
        end
        for d in 1:3
            decomp.active[d] || continue
            Fdc = solver.flux[d, c]
            σ = solver.folds[d] === nothing ? 1 : solver.folds[d].sigflux[c]
            if unitgeom
                deriv_along!(solver.tmp_a, Fdc, solver, d, σ)
                @threaded nx*ny*nz for k in 1:nz
                    @inbounds for j in 1:ny, i in 1:nx
                        dQ[i+o1, j+o2, k+o3, c] -= solver.tmp_a[i+o1, j+o2, k+o3]
                    end
                end
            else
                # tmp_b = A_d F_d over the full array; A_d is odd in r for the
                # cylindrical axis (A₁ = r), flipping the flux parity.
                Ad = solver.area_d[d]
                @threaded length(solver.tmp_b) for idx in eachindex(solver.tmp_b)
                    @inbounds solver.tmp_b[idx] = Ad[idx] * Fdc[idx]
                end
                deriv_along!(solver.tmp_a, solver.tmp_b, solver, d, σ)
                @threaded nx*ny*nz for k in 1:nz
                    @inbounds for j in 1:ny, i in 1:nx
                        I = CartesianIndex(i + o1, j + o2, k + o3)
                        dQ[I, c] -= solver.inv_J[I] * solver.tmp_a[I]
                    end
                end
            end
        end
    end
    add_metric_sources!(solver, dQ, Q, solver.metric)
    for d in 1:3, side in 1:2
        decomp.active[d] || continue
        correct_rhs!(solver.bcs[d][side], solver, Q, dQ, d, side)
    end
    add_sources!(solver, dQ, Q, solver.tstage)
    return dQ
end


