# Solver container and conservative multicomponent Navier–Stokes RHS in
# orthogonal curvilinear coordinates, with collapsed (1-D/2-D) dimensions and
# a regularized cylindrical axis.
#
# Collapsed dimensions (nglob[d] == 1) carry no derivatives, filters, halos,
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

mutable struct Solver{T}
    dec::Decomp
    eos::EOS
    transport::Transport{T}
    art::ArtParams{T}
    metric::Metric
    stretch::NTuple{3,Union{Nothing,Stretch}}
    folds::NTuple{3,Union{Nothing,FoldSpec}}   # coordinate-singularity folds
    ns::Int
    ncons::Int
    mom::NTuple{3,Int}
    ie::Int
    Ldom::NTuple{3,T}
    origin::NTuple{3,T}
    cshift::NTuple{3,T}                     # half-cell offset (axis grids)
    h::NTuple{3,T}
    bcs::NTuple{3,Tuple{BoundaryCondition,BoundaryCondition}}
    dplans::NTuple{3,Union{Nothing,AbstractDirPlan}}
    fplans::NTuple{3,Union{Nothing,AbstractDirPlan}}
    pairbuf::Array{T,3}                     # paired-fold combo scratch
    pairout::Array{T,3}                     # paired-fold second-parity result
    cfl::T
    filter_interval::Int
    # primitives (full padded arrays)
    rho::Array{T,3}; u::Array{T,3}; v::Array{T,3}; w::Array{T,3}
    p::Array{T,3};  Tt::Array{T,3}; c::Array{T,3}; cpm::Array{T,3}
    Ys::Vector{Array{T,3}}
    # gradients
    G::Matrix{Array{T,3}}            # G[d, j] = physical (∇u)_{dj}
    gradT::NTuple{3,Array{T,3}}
    gradY::Matrix{Array{T,3}}        # gradY[d, k]
    # artificial properties and scratch
    mua::Array{T,3}; betaa::Array{T,3}; kappaa::Array{T,3}
    Dart::Vector{Array{T,3}}
    Smag::Array{T,3}; sens::Array{T,3}; sacc::Array{T,3}
    tmpA::Array{T,3}; tmpB::Array{T,3}
    # geometry
    Jinv::Array{T,3}
    Adim::NTuple{3,Array{T,3}}
    invh::NTuple{3,Array{T,3}}
    rinv::Array{T,3}
    cotr::Array{T,3}
    # fluxes FF[d, c]
    FF::Matrix{Array{T,3}}
    t::T
    tstage::T
    step::Int
end

"""
    Solver(; nglob, Ldom, bcs, kwargs...)

See earlier keywords, plus:
- Collapsed dimensions: set `nglob[d] = 1` with `(PeriodicBC(), PeriodicBC())`
  for that dimension; e.g. axisymmetric cylindrical is `nglob = (Nr, 1, Nz)`.
- Cylindrical axis: `bcs[1] = (AxisBC(), <outer bc>)` with
  `metric = CylindricalMetric()`, `origin[1] = 0`, θ collapsed
  (`nglob[2] = 1`), dimension 1 unstretched. The grid is half-offset in r.
"""
function Solver(; nglob::NTuple{3,Int}, Ldom, bcs,
                eos::EOS=single_species(),
                transport::Transport{T}=Transport(),
                art::ArtParams=ArtParams(),
                metric::Metric=CartesianMetric(),
                stretch::NTuple{3,Union{Nothing,Stretch}}=(nothing, nothing, nothing),
                origin=(0.0, 0.0, 0.0),
                deriv::AbstractCompactScheme=lele_d1_6(),
                filt::AbstractCompactScheme=compact_filter(0.45),
                cfl::Real=0.5, filter_interval::Int=1,
                dims=nothing, H::Int=4) where {T}
    for d in 1:3
        isperiodic(bcs[d][1]) == isperiodic(bcs[d][2]) ||
            error("dimension $d mixes periodic and non-periodic conditions")
        nglob[d] > 1 || (isperiodic(bcs[d][1]) ||
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
        nglob[2] == 1 || iseven(nglob[2]) ||
            error("resolved-θ AxisBC requires an even θ point count over 2π")
    end
    if orig1
        metric isa SphericalMetric || error("OriginBC requires SphericalMetric")
        stretch[1] === nothing || error("folded dimensions cannot be stretched")
        abs(Float64(origin[1])) < 1e-14 || error("OriginBC requires origin[1] = 0")
        nglob[3] == 1 || iseven(nglob[3]) ||
            error("resolved-φ OriginBC requires an even φ point count over 2π")
        θsum = 2 * Float64(origin[2]) + Float64(Ldom[2])
        nglob[2] == 1 || isapprox(θsum, π; atol=1e-10) ||
            error("OriginBC requires a θ range symmetric about π/2")
    end
    if poles
        metric isa SphericalMetric || error("PoleBC requires SphericalMetric")
        stretch[2] === nothing || error("folded dimensions cannot be stretched")
        abs(Float64(origin[2])) < 1e-14 && isapprox(Float64(Ldom[2]), π; atol=1e-10) ||
            error("PoleBC requires the θ domain (0, π)")
        nglob[3] == 1 || iseven(nglob[3]) ||
            error("resolved-φ PoleBC requires an even φ point count over 2π")
    end
    periodic = ntuple(d -> nglob[d] > 1 ? isperiodic(bcs[d][1]) : true, 3)
    (axis || orig1) && (periodic = (false, periodic[2], periodic[3]))
    poles && (periodic = (periodic[1], false, periodic[3]))
    for d in 1:3
        stretch[d] === nothing || !periodic[d] ||
            error("dimension $d: stretched dimensions must be non-periodic")
    end
    dec = Decomp(nglob, periodic; dims=dims, H=H)
    Lt = ntuple(d -> T(Ldom[d]), 3)
    # Grid spacing: computational s ∈ [0,1] for stretched dims; half-offset
    # r ∈ (0, R] for axis grids (h = R/(N − ½), r₁ = h/2); standard otherwise.
    # Half-offset grids on folded dimensions: fold at the low end only
    # (r: axis/origin) gives h = L/(N − ½); folds at both ends (θ poles)
    # give h = L/N; either way the first node sits at h/2.
    fold_lo_dim = ntuple(d -> (d == 1 && (axis || orig1)) || (d == 2 && poles), 3)
    fold_hi_dim = ntuple(d -> d == 2 && poles, 3)
    h = ntuple(3) do d
        dec.active[d] || return one(T)
        stretch[d] === nothing || return one(T) / (nglob[d] - 1)
        fold_lo_dim[d] && fold_hi_dim[d] && return Lt[d] / nglob[d]
        fold_lo_dim[d] && return Lt[d] / (nglob[d] - T(0.5))
        periodic[d] ? Lt[d] / nglob[d] : Lt[d] / (nglob[d] - 1)
    end
    cshift = ntuple(d -> fold_lo_dim[d] ? h[d] / 2 : zero(T), 3)
    mkd(sch, d; kw...) = plan_direction(dec, sch, d, h[d]; kw...)
    ns = nspecies(eos)
    ncons = ns + 4
    mom = (ns + 1, ns + 2, ns + 3)
    f() = field(dec)

    # Fold specs. sigvel/sigflux derivations live in folds.jl and the README.
    function pairspec(pdim, revdim)
        # Degenerate mappings on collapsed dims are identities.
        shift_needed = pdim != 0 && dec.active[pdim]
        rev_needed = revdim != 0 && dec.active[revdim]
        (!shift_needed && !rev_needed) && return nothing   # self-paired
        shift_local = !shift_needed || dec.dims[pdim] == 1
        rev_local = !rev_needed || dec.dims[revdim] == 1
        if shift_needed
            shift_local || (iseven(dec.dims[pdim]) &&
                            nglob[pdim] % dec.dims[pdim] == 0) ||
                error("pairing dim $pdim must be on one rank or split into an " *
                      "even number of uniform blocks")
            shift_local && (iseven(dec.nloc[pdim]) ||
                error("pairing dim $pdim local extent must be even"))
        end
        if rev_needed && !rev_local
            nglob[revdim] % dec.dims[revdim] == 0 ||
                error("reversed dim $revdim must split into uniform blocks")
            iseven(dec.dims[revdim]) ||
                error("reversed dim $revdim needs an even rank count")
        end
        crd = collect(Int, dec.coords)
        shift_needed && !shift_local &&
            (crd[pdim] = mod(crd[pdim] + dec.dims[pdim] ÷ 2, dec.dims[pdim]))
        rev_needed && !rev_local &&
            (crd[revdim] = dec.dims[revdim] - 1 - crd[revdim])
        partner = Int(MPI.Cart_rank(dec.comm, Cint.(crd)))
        loc = shift_local && rev_local
        keep_e = if shift_needed && !shift_local
            dec.coords[pdim] < dec.dims[pdim] ÷ 2
        elseif rev_needed && !rev_local
            dec.coords[revdim] < dec.dims[revdim] ÷ 2
        else
            true
        end
        PairSpec(shift_needed ? pdim : 0, rev_needed ? revdim : 0,
                 shift_local, rev_local, loc, partner, keep_e,
                 loc ? zeros(0, 0, 0) : field(dec))
    end
    function foldspec(d, lo, hi, pdim, revdim, sigvel, sigflux)
        dp = (mkd(deriv, d; lo_fold=(lo ? 1 : nothing), hi_fold=(hi ? 1 : nothing)),
              mkd(deriv, d; lo_fold=(lo ? -1 : nothing), hi_fold=(hi ? -1 : nothing)))
        fp = (mkd(filt, d; lo_fold=(lo ? 1 : nothing), hi_fold=(hi ? 1 : nothing)),
              mkd(filt, d; lo_fold=(lo ? -1 : nothing), hi_fold=(hi ? -1 : nothing)))
        FoldSpec(d, lo, hi, pairspec(pdim, revdim), sigvel, sigflux, dp, fp)
    end
    sigflux_of(sv, Aσ) = Int[(c <= ns ? sv[1] :
                              c == ns + 4 ? sv[1] :
                              sv[1] * sv[c - ns]) * Aσ for c in 1:(ns+4)]
    folds = (nothing, nothing, nothing)
    if axis
        sv = (-1, -1, 1)
        folds = (foldspec(1, true, false, 2, 0, sv, sigflux_of(sv, -1)),
                 folds[2], folds[3])           # A₁ = r is odd
    elseif orig1
        sv = (-1, 1, -1)
        folds = (foldspec(1, true, false, 3, 2, sv, sigflux_of(sv, 1)),
                 folds[2], folds[3])           # A₁ = r² sinθ is even
    end
    if poles
        sv = (1, -1, -1)
        # Fold along θ: flux parities use σ(u_θ); A₂ = r sinθ is odd in θ.
        sf2 = Int[(c <= ns ? sv[2] : c == ns + 4 ? sv[2] :
                   sv[2] * sv[c - ns]) * -1 for c in 1:(ns+4)]
        folds = (folds[1], foldspec(2, true, true, 3, 0, sv, sf2), folds[3])
    end
    haspair = any(fs -> fs !== nothing && fs.pair !== nothing, folds)
    dplans = ntuple(d -> dec.active[d] && folds[d] === nothing ?
                         mkd(deriv, d) : nothing, 3)
    fplans = ntuple(d -> dec.active[d] && folds[d] === nothing ?
                         mkd(filt, d) : nothing, 3)
    orig = ntuple(d -> stretch[d] === nothing ? T(origin[d]) : zero(T), 3)
    s = Solver{T}(dec, eos, transport, art, metric, stretch, folds,
                  ns, ncons, mom, ns + 4,
                  Lt, orig, cshift, h,
                  ntuple(d -> (bcs[d][1], bcs[d][2]), 3),
                  dplans, fplans,
                  any(fs -> fs !== nothing, folds) ? field(dec) : zeros(T, 0, 0, 0),
                  any(fs -> fs !== nothing, folds) ? field(dec) : zeros(T, 0, 0, 0),
                  T(cfl), filter_interval,
                  f(), f(), f(), f(), f(), f(), f(), f(),
                  [f() for _ in 1:ns],
                  [f() for _ in 1:3, _ in 1:3],
                  (f(), f(), f()),
                  [f() for _ in 1:3, _ in 1:ns],
                  f(), f(), f(),
                  [f() for _ in 1:ns],
                  f(), f(), f(), f(), f(),
                  f(), (f(), f(), f()), (f(), f(), f()), f(), f(),
                  [f() for _ in 1:3, _ in 1:ncons],
                  zero(T), zero(T), 0)
    init_geometry!(s)
    return s
end

"Physical coordinate of local interior index i along dimension d."
function xcoord(s::Solver, d::Int, i::Int)
    ξ = s.origin[d] + s.cshift[d] + (s.dec.off[d] + i - 1) * s.h[d]
    st = s.stretch[d]
    return st === nothing ? ξ : st.x(ξ)
end

"Halo-offset CartesianIndex of local interior point (i, j, k)."
gidx(s::Solver, i::Int, j::Int, k::Int) =
    CartesianIndex(i + s.dec.Hd[1], j + s.dec.Hd[2], k + s.dec.Hd[3])

allocate_state(s::Solver) = allocate_state(s.dec, s.ncons)

# --- Operator routing through folds ----------------------------------------

"""
    deriv_along!(out, f, s, d, σf)

Compact derivative of `f` along active dimension `d`; `σf` is the field's
antipodal sign for the fold on `d` (ignored when there is none). Caller
ensures current halos.
"""
function deriv_along!(out, f, s::Solver, d::Int, σf::Int)
    fs = s.folds[d]
    if fs === nothing
        apply_along!(out, s.dplans[d], f, s.dec)
    else
        fold_apply!(out, f, s, fs, σf; isfilter=false)
    end
    return out
end

"Compact filter of `f` along dimension `d` with antipodal sign `σf`."
function filt_along!(out, f, s::Solver, d::Int, σf::Int)
    fs = s.folds[d]
    if fs === nothing
        apply_along!(out, s.fplans[d], f, s.dec)
    else
        fold_apply!(out, f, s, fs, σf; isfilter=true)
    end
    return out
end

# Scale a raw coordinate-derivative field by 1/h_d pointwise (full array).
function _scale_grad!(g, s, d)
    ih = s.invh[d]
    @threaded length(g) for idx in eachindex(g)
        @inbounds g[idx] *= ih[idx]
    end
    return g
end

# Antipodal signs of velocity and conserved components for the fold (if any)
# on dimension d; scalars, partial densities, and energy are always +1.
vel_parity(s::Solver, d::Int, j::Int) =
    s.folds[d] === nothing ? 1 : s.folds[d].sigvel[j]
cons_parity(s::Solver, d::Int, c::Int) =
    (s.folds[d] === nothing || c <= s.ns || c == s.ie) ? 1 :
    s.folds[d].sigvel[c - s.ns]

assemble_fluxes!(s::Solver, Q) = _assemble_fluxes!(s, s.eos, Q)

function _assemble_fluxes!(s::Solver{T}, eos, Q) where {T}
    dec = s.dec
    o1, o2, o3 = dec.Hd
    nx, ny, nz = dec.nloc
    tr = s.transport
    mu0 = tr.mu0
    ns = s.ns
    ie = s.ie
    G = s.G
    gT = s.gradT
    gY = s.gradY
    FF = s.FF
    @threaded nx*ny*nz for k in 1:nz
        @inbounds for j in 1:ny, i in 1:nx
            I = CartesianIndex(i + o1, j + o2, k + o3)
            ρ = s.rho[I]
            uv = (s.u[I], s.v[I], s.w[I])
            p = s.p[I]
            Tp = s.Tt[I]
            E = Q[I, ie]
            μ = mu0 + s.mua[I]
            β = s.betaa[I]
            κ = mu0 * s.cpm[I] / tr.Pr + s.kappaa[I]
            D0 = mu0 / (tr.Sc * ρ)               # molecular part of each D_k
            divu = G[1, 1][I] + G[2, 2][I] + G[3, 3][I]
            τ11 = μ * (2G[1,1][I] - (2/3) * divu) + β * divu
            τ22 = μ * (2G[2,2][I] - (2/3) * divu) + β * divu
            τ33 = μ * (2G[3,3][I] - (2/3) * divu) + β * divu
            τ12 = μ * (G[1,2][I] + G[2,1][I])
            τ13 = μ * (G[1,3][I] + G[3,1][I])
            τ23 = μ * (G[2,3][I] + G[3,2][I])
            τ = ((τ11, τ12, τ13), (τ12, τ22, τ23), (τ13, τ23, τ33))
            for d in 1:3
                ud = uv[d]
                τd = τ[d]
                # Per-species diffusion with a correction velocity:
                # J_k = −ρ D_k ∇Y_k + ρ Y_k V_c, V_c = Σ_j D_j ∇Y_j,
                # which enforces Σ_k J_k = 0 exactly since ΣY_k = 1.
                Vc = zero(T)
                for sp in 1:ns
                    Vc += (D0 + s.Dart[sp][I]) * gY[d, sp][I]
                end
                hdiff = zero(T)              # Σ_k h_k J_{k,d}
                for sp in 1:ns
                    Dk = D0 + s.Dart[sp][I]
                    Jkd = ρ * (-Dk * gY[d, sp][I] + s.Ys[sp][I] * Vc)
                    FF[d, sp][I] = ρ * s.Ys[sp][I] * ud + Jkd
                    hdiff += species_enthalpy(eos, sp, Tp) * Jkd
                end
                FF[d, s.mom[1]][I] = ρ * ud * uv[1] + (d == 1 ? p : zero(T)) - τd[1]
                FF[d, s.mom[2]][I] = ρ * ud * uv[2] + (d == 2 ? p : zero(T)) - τd[2]
                FF[d, s.mom[3]][I] = ρ * ud * uv[3] + (d == 3 ? p : zero(T)) - τd[3]
                FF[d, ie][I] = (E + p) * ud -
                               (uv[1]*τd[1] + uv[2]*τd[2] + uv[3]*τd[3]) -
                               κ * gT[d][I] + hdiff
            end
        end
    end
    return s
end

"""
    compute_rhs!(s, Q, dQ)

Evaluate dQ/dt into the interior of `dQ` from the conserved state `Q`
(boundary conditions should already be enforced on `Q`). Collapsed dimensions
contribute no derivatives; the axis dimension routes through parity-folded
plans with mirror-filled halos.
"""
function compute_rhs!(s::Solver, Q, dQ)
    dec = s.dec
    exchange_state!(Q, dec)
    primitives!(s, Q)
    vel = (s.u, s.v, s.w)
    for jj in 1:3, d in 1:3
        if dec.active[d]
            deriv_along!(s.G[d, jj], vel[jj], s, d, vel_parity(s, d, jj))
            _scale_grad!(s.G[d, jj], s, d)   # 1/h_d incl. stretching Jacobian
        else
            fill!(s.G[d, jj], 0)
        end
    end
    metric_correct_gradients!(s, s.metric)   # additive curvature terms
    compute_artificial!(s, Q)
    for d in 1:3
        dec.active[d] || continue
        deriv_along!(s.gradT[d], s.Tt, s, d, 1)
        _scale_grad!(s.gradT[d], s, d)
        for sp in 1:s.ns
            deriv_along!(s.gradY[d, sp], s.Ys[sp], s, d, 1)
            _scale_grad!(s.gradY[d, sp], s, d)
        end
    end
    assemble_fluxes!(s, Q)
    for d in 1:3
        exchange_dim_batch!(view(s.FF, d, :), dec, d)
    end
    o1, o2, o3 = dec.Hd
    nx, ny, nz = dec.nloc
    for c in 1:s.ncons
        @threaded nx*ny*nz for k in 1:nz
            @inbounds for j in 1:ny, i in 1:nx
                dQ[i+o1, j+o2, k+o3, c] = 0
            end
        end
        for d in 1:3
            dec.active[d] || continue
            # tmpB = A_d F_d over the full array; A_d is odd in r for the
            # cylindrical axis (A₁ = r), flipping the flux parity.
            Ad = s.Adim[d]
            Fdc = s.FF[d, c]
            @threaded length(s.tmpB) for idx in eachindex(s.tmpB)
                @inbounds s.tmpB[idx] = Ad[idx] * Fdc[idx]
            end
            σ = s.folds[d] === nothing ? 1 : s.folds[d].sigflux[c]
            deriv_along!(s.tmpA, s.tmpB, s, d, σ)
            @threaded nx*ny*nz for k in 1:nz
                @inbounds for j in 1:ny, i in 1:nx
                    I = CartesianIndex(i + o1, j + o2, k + o3)
                    dQ[I, c] -= s.Jinv[I] * s.tmpA[I]
                end
            end
        end
    end
    add_metric_sources!(s, dQ, Q, s.metric)
    for d in 1:3, side in 1:2
        dec.active[d] || continue
        correct_rhs!(s.bcs[d][side], s, Q, dQ, d, side)
    end
    return dQ
end


