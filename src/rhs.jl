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

mutable struct Solver{T,Eq<:EquationSet,E<:EOS,M<:Metric,St,Fo,BC,DP,FP,SP,RP,Src}
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
    smooth_plans::SP                        # sensor smoother (see ArtParams.smoother)
    ring_plans::RP                          # sensor detector (see ArtParams.detector)
    pairbuf::Array{T,3}                     # paired-fold combo scratch
    pairout::Array{T,3}                     # paired-fold second-parity result
    cfl::T
    filter_interval::Int
    filter_cfl::T                           # 0 = unrelaxed; see filter_weight
    control::StepControl                    # timestep floors, prediction, retry
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
    ring_buf::Array{T,3}                    # d8 detector output, empty otherwise
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
    dt_prev::T                              # last accepted step, for growth capping
    rate_prev::T                            # last CFL rate, for the predictor
    # Wall-clock accounting, filled in by `run!`. Rank-local and deliberately so:
    # a reduction here would be a collective on every step, paid by every run
    # whether or not anything reads it. Callers wanting load imbalance reduce
    # these themselves at their own (much lower) reporting frequency — see
    # `ProgressLog`. Seconds, and Float64 regardless of T.
    wall_step::Float64                      # last completed step, excl. callbacks
    wall_total::Float64                     # cumulative over the run
    # What the positivity failsafe has seen and repaired. Global rather than
    # rank-local, unlike the wall-clock fields above: the repair reduces its own
    # tally on every step it runs, and it runs only when
    # `StepControl.floor_ratio` is set, so a run that leaves it off pays nothing.
    floor_tally::FloorTally
end

"""
    Solver(; n_global, L_domain, bcs, kwargs...)

Backend constructor: allocate the distributed solver, plan every compact
operator, and fill the geometry. [`setup`](@ref) is the normal entry point and
calls this after splitting a [`Problem`](@ref) from a [`Numerics`](@ref);
construct a `Solver` directly when there is no `Problem` to build from, as the
test and benchmark suites do. It allocates no conserved state, so pair it with
[`allocate_state`](@ref) and [`initialize!`](@ref).

# Required keywords

- `n_global`: global point count per direction. A count of one collapses that
  direction, as described under [`Numerics`](@ref).
- `L_domain`: three domain *extents*, one per direction. Note the difference from
  `Problem.domain`, which gives endpoints; here `origin` carries the low corner.
- `bcs`: three `(low, high)` pairs of [`BoundaryCondition`](@ref) objects.

# Optional keywords

`eos`, `transport`, `metric`, and `sources` take their defaults and meaning from
[`Problem`](@ref); `art`, `deriv`, `filt`, `cfl`, `control`, `filter_interval`,
`filter_cfl`, `dims`, `n_halo`, and `stretch` from [`Numerics`](@ref). The two with no
`Problem`/`Numerics` counterpart are:

- `origin`: low corner of the domain, one value per direction. Default
  `(0.0, 0.0, 0.0)`. A folded direction requires its origin at zero. A stretched
  direction ignores this entry: its computational coordinate runs over [0, 1] and
  the [`Stretch`](@ref) mapping carries both endpoints.
- `equations`: the [`EquationSet`](@ref) owning the conserved layout. The default
  builds [`NavierStokes1T`](@ref) from `eos`; a supplied set must agree with
  `eos` on the species count.

# Geometry selected by the boundary conditions

Collapsed dimensions and coordinate singularities are not keywords: setup
recognizes them from `bcs` and validates the rest of the configuration against
them.

- Collapsed dimensions: set `n_global[d] = 1` with `(PeriodicBC(), PeriodicBC())`
  for that dimension; e.g. axisymmetric cylindrical is `n_global = (Nr, 1, Nz)`.
- Cylindrical axis: `bcs[1] = (AxisBC(), <outer bc>)` with
  `metric = CylindricalMetric()`, `origin[1] = 0`, and dimension 1 unstretched.
  θ may be collapsed (axisymmetric) or resolved over 2π with an even point
  count. The grid is half-offset in r, so no node sits at r = 0.
- Spherical origin and poles: [`OriginBC`](@ref) at the low end of r and
  [`PoleBC`](@ref) at *both* ends of θ, with `metric = SphericalMetric()`. The
  origin additionally requires a θ range symmetric about π/2, and the poles the
  full range (0, π). Either may be combined with the other. φ may be collapsed
  or resolved over 2π with an even point count, as θ may be at the axis.
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
                cfl::Real=0.5, filter_interval::Int=1, filter_cfl::Real=0.0,
                control::StepControl=StepControl(),
                dims=nothing, n_halo::Int=4) where {T}
    for d in 1:3
        isperiodic(bcs[d][1]) == isperiodic(bcs[d][2]) ||
            error("dimension $d mixes periodic and non-periodic conditions")
        n_global[d] > 1 || (isperiodic(bcs[d][1]) ||
            error("collapsed dimension $d must use (PeriodicBC(), PeriodicBC())"))
    end
    art.mu_sensor in (:strain, :velocity) ||
        error("art.mu_sensor must be :strain or :velocity, got :$(art.mu_sensor)")
    art.beta_sensor in (:strain, :gated_strain, :dilatation, :ungated_dilatation) ||
        error("art.beta_sensor must be :strain, :gated_strain, :dilatation or " *
              ":ungated_dilatation, got :$(art.beta_sensor)")
    art.reduction in (:sum, :max) ||
        error("art.reduction must be :sum or :max, got :$(art.reduction)")
    art.smoother in (:compact, :gaussian) ||
        error("art.smoother must be :compact or :gaussian, got :$(art.smoother)")
    art.detector in (:delta4, :d8) ||
        error("art.detector must be :delta4 or :d8, got :$(art.detector)")
    # Per-condition restrictions on geometry and EOS agreement (boundary.jl).
    for d in 1:3, side in 1:2
        validate_bc(bcs[d][side], metric, eos, d, side)
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
    # The sensor smoother stands in for Cook's Gaussian test filter. `:compact`
    # reuses `filt`, which is what shipped before the option existed and keeps
    # every plan identical; `:gaussian` is the explicit nine-point stencil the
    # reference implementation uses, and carries no line solve.
    smoo = art.smoother === :gaussian ? gaussian_filter(T) : filt
    # The sensor detector. `:delta4` is the explicit undivided fourth
    # difference applied inside `delta4_sum!`, which needs no plan at all;
    # `:d8` is the pentadiagonal compact eighth derivative and needs one per
    # dimension, and one pair per fold, exactly as the smoother does.
    ring = compact_d8(T)
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
        # `:compact` aliases the filter plans rather than duplicating them.
        # That is exactly what `smooth!` did before it had an operator of its
        # own, so the default path keeps both its answer and its footprint.
        sp = art.smoother === :compact ? fp :
             (mkd(smoo, d; lo_fold=(lo ? 1 : nothing), hi_fold=(hi ? 1 : nothing)),
              mkd(smoo, d; lo_fold=(lo ? -1 : nothing), hi_fold=(hi ? -1 : nothing)))
        rp = art.detector === :delta4 ? (nothing, nothing) :
             (mkd(ring, d; lo_fold=(lo ? 1 : nothing), hi_fold=(hi ? 1 : nothing)),
              mkd(ring, d; lo_fold=(lo ? -1 : nothing), hi_fold=(hi ? -1 : nothing)))
        FoldSpec(d, lo, hi, pairspec(pdim, revdim), sigvel, sigflux, dp, fp, sp, rp)
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
    smooth_plans = art.smoother === :compact ? filter_plans :
                   ntuple(d -> decomp.active[d] && folds[d] === nothing ?
                          mkd(smoo, d) : nothing, 3)
    # `nothing` rather than a tuple of nothings, and the difference matters:
    # `detect_sum!` dispatches on this field's type to decide which detector
    # runs, so under `:delta4` the whole d8 path — `ring_sum!`, `ring_along!`,
    # and the `apply_along!` call taking a possibly-absent plan — is not
    # reachable from inference and costs the default configuration nothing.
    # A tuple would not serve: a fully folded run carries no plans here even
    # under `:d8`, since those live on the FoldSpec.
    ring_plans = art.detector === :delta4 ? nothing :
                 ntuple(d -> decomp.active[d] && folds[d] === nothing ?
                        mkd(ring, d) : nothing, 3)
    orig = ntuple(d -> stretch[d] === nothing ? T(origin[d]) : zero(T), 3)
    bcs_t = ntuple(d -> (bcs[d][1], bcs[d][2]), 3)
    solver = Solver{T,typeof(equations),typeof(eos),typeof(metric),
                    typeof(stretch),typeof(folds),typeof(bcs_t),
                    typeof(deriv_plans),typeof(filter_plans),typeof(smooth_plans),
                    typeof(ring_plans),typeof(sources)}(
                  decomp, equations, eos, transport, art, metric, stretch, folds, sources,
                  Lt, orig, coord_shift, h,
                  bcs_t,
                  deriv_plans, filter_plans, smooth_plans, ring_plans,
                  any(fold -> fold !== nothing, folds) ? field(decomp) : zeros(T, 0, 0, 0),
                  any(fold -> fold !== nothing, folds) ? field(decomp) : zeros(T, 0, 0, 0),
                  T(cfl), filter_interval, T(filter_cfl), control,
                  f(), f(), f(), f(), f(), f(), f(), f(),
                  [f() for _ in 1:n_species],
                  [f() for _ in 1:3, _ in 1:3],
                  (f(), f(), f()),
                  [f() for _ in 1:3, _ in 1:n_species],
                  f(), f(), f(),
                  [f() for _ in 1:n_species],
                  f(), f(), f(), f(), f(),
                  art.detector === :delta4 ? zeros(T, 0, 0, 0) : f(),
                  f(), (f(), f(), f()), (f(), f(), f()), f(), f(),
                  [f() for _ in 1:3, _ in 1:n_cons],
                  zero(T), zero(T), 0, zero(T), zero(T), 0.0, 0.0, FloorTally())
    init_geometry!(solver)
    return solver
end

"""
    xcoord(solver, d, i)

Physical coordinate of rank-local, one-based interior index `i` in direction
`d`. The index does not include halo padding. This is equivalent to
`global_xcoord(solver, d, solver.decomp.offset[d] + i)`.
"""
xcoord(solver::Solver, d::Int, i::Int) =
    global_xcoord(solver, d, solver.decomp.offset[d] + i)

"""
    global_xcoord(solver, d, g)

Physical coordinate of global, one-based index `g` in direction `d`, including
any half-cell fold offset and [`Stretch`](@ref) mapping. Every rank returns the
same value for the same `(d, g)`. Use this form when assembling a global
coordinate vector and [`xcoord`](@ref) for a local interior index.
"""
function global_xcoord(solver::Solver, d::Int, g::Int)
    ξ = solver.origin[d] + solver.coord_shift[d] + (g - 1) * solver.h[d]
    st = solver.stretch[d]
    return st === nothing ? ξ : st.x(ξ)
end

"""
    gidx(solver, i, j, k) -> CartesianIndex

Convert rank-local, one-based interior indices to a `CartesianIndex` for the
halo-padded state and solver fields. Collapsed directions have zero padding.
The result is rank-local; it does not locate a global point owned by another
rank.
"""
function gidx(solver::Solver, i::Int, j::Int, k::Int)
    pad = solver.decomp.n_halo_d
    return CartesianIndex(i + pad[1], j + pad[2], k + pad[3])
end

"""
    interior_index(solver, I) -> (i, j, k)

Rank-local, one-based interior indices of the padded `CartesianIndex` `I`; the
inverse of [`gidx`](@ref). Use it wherever a padded index has to be handed to
something that takes interior ones. [`boundary_plane`](@ref) yields padded
indices, while [`xcoord`](@ref) expects an interior index:

```julia
for I in boundary_plane(solver, 1, 1)
    i, j, k = interior_index(solver, I)
    x1 = xcoord(solver, 1, i)
end
```

Indices of halo cells come back outside `1:n_local[d]`, and are meaningful only
as an offset from this rank's block.
"""
@inline function interior_index(solver::Solver, I::CartesianIndex{3})
    pad = solver.decomp.n_halo_d
    return (I[1] - pad[1], I[2] - pad[2], I[3] - pad[3])
end

# --- Reading the in-flight conserved state -----------------------------------
#
# `Q` is a 4-D array in the layout defined by the equation set, so a caller
# wanting the density writes `Q[I, 1] + Q[I, 2]` and has silently hardcoded a
# two-species run. The functions below are the layout-independent equivalents,
# for callback conditions, custom diagnostics, and any other code reading `Q`
# between steps rather than reading the primitives, which are stale there; see
# `refresh_primitives!`.
#
# All of them index the PADDED arrays, following the convention of `gidx` and
# `boundary_plane`, and all are rank-local: they report nothing about points
# this rank does not hold. A predicate built from them is reduced by
# `WhenState`, or must be reduced by the caller.

"""
    mixture_density(solver, Q, I) -> ρ

Mixture density at padded index `I`, the sum of the partial densities. Prefer
this method to explicit component indices so that the expression remains valid
when the species count changes.
"""
@inline function mixture_density(solver::Solver, Q, I)
    ρ = zero(eltype(Q))
    @inbounds for sp in 1:solver.equations.n_species
        ρ += Q[I, sp]
    end
    return ρ
end

"""
    velocity(solver, Q, I) -> (u, v, w)

Physical, coordinate-aligned velocity components at padded index `I`, recovered
from the momenta and the mixture density.
"""
@inline function velocity(solver::Solver, Q, I)
    ri = one(eltype(Q)) / mixture_density(solver, Q, I)
    m1, m2, m3 = solver.equations.i_mom
    @inbounds return (Q[I, m1] * ri, Q[I, m2] * ri, Q[I, m3] * ri)
end

"Total energy per unit volume at padded index `I`."
@inline total_energy(solver::Solver, Q, I) =
    @inbounds Q[I, solver.equations.i_energy]

"Mass fraction of species `sp` at padded index `I`."
@inline mass_fraction(solver::Solver, Q, I, sp::Int) =
    @inbounds Q[I, sp] / mixture_density(solver, Q, I)

"""
    boundary_plane(solver, d, side) -> CartesianIndices or nothing

Padded indices of this rank's plane on the global `side` (1 low, 2 high) of
dimension `d`, or `nothing` when this rank does not own that edge. Ownership
along `d` is the only test applied, so a periodic or collapsed dimension yields
a plane as well. The plane spans the full local extent of the other two
dimensions and is one point thick in `d`. For example, a boundary diagnostic may
be written as

```julia
plane = boundary_plane(solver, 1, 1)
plane === nothing && return -Inf
return maximum(I -> mixture_density(solver, Q, I), plane)
```

The result is rank-local. [`WhenState`](@ref) reduces a predicate constructed
from it; other uses require an explicit reduction. An unreduced value is valid
globally only in a serial calculation.

These are padded indices. Convert one with [`interior_index`](@ref) before
passing it to [`xcoord`](@ref) or anything else expecting an interior index.
"""
boundary_plane(solver::Solver, d::Int, side::Int) = wallplane(solver.decomp, d, side)

allocate_state(solver::Solver) = allocate_state(solver.decomp, solver.equations.n_cons)

# --- Operator routing through folds ----------------------------------------

"""
    deriv_along!(out, f, solver, d, σf)

Compact derivative of `f` along active dimension `d`; `σf` is the field's
antipodal sign for the fold on `d` (ignored when there is none). The caller
ensures current rank-boundary halos of `f`. Only the interior of `out` is
written.

This is a distributed line solve along `d`, so it is collective over that
dimension's sub-communicator and every rank must call it, including ranks
holding no part of a fold. At a self-paired fold `f` is also written:
`fold_fill!` mirrors its halos beyond the folded end before the sweep. A paired
fold leaves `f` untouched, running the even/odd butterfly through
`solver.pairbuf` instead.
"""
function deriv_along!(out, f, solver::Solver, d::Int, σf::Int)
    fold = solver.folds[d]
    if fold === nothing
        apply_along!(out, solver.deriv_plans[d], f, solver.decomp)
    else
        fold_apply!(out, f, solver, fold, σf, Val(:deriv))
    end
    return out
end

"""Compact filter of `f` along dimension `d` with antipodal sign `σf`. Collective,
with the same halo and fold contract as `deriv_along!`."""
function filt_along!(out, f, solver::Solver, d::Int, σf::Int)
    fold = solver.folds[d]
    if fold === nothing
        apply_along!(out, solver.filter_plans[d], f, solver.decomp)
    else
        fold_apply!(out, f, solver, fold, σf, Val(:filter))
    end
    return out
end

"""
    smooth_along!(out, f, solver, d, σf)

Sensor smoother of `f` along dimension `d` with antipodal sign `σf`. This is
the Cook test filter, selected by `ArtParams.smoother`, and is a distinct
operator from `filt_along!`: the two coincide only under
`ArtParams(smoother = :compact)`, which aliases the filter plans rather than
planning an operator of its own; the default `:gaussian` plans the explicit
nine-point stencil of [`gaussian_filter`](@ref). Only the artificial-property
sensors go through here, by way of `smooth!`.

Collective, as `deriv_along!` is.
"""
function smooth_along!(out, f, solver::Solver, d::Int, σf::Int)
    fold = solver.folds[d]
    if fold === nothing
        apply_along!(out, solver.smooth_plans[d], f, solver.decomp)
    else
        fold_apply!(out, f, solver, fold, σf, Val(:smooth))
    end
    return out
end

"""
    ring_along!(out, f, solver, d, σf)

Compact eighth derivative of `f` along dimension `d` with antipodal sign `σf`,
the ringing detector selected by `ArtParams(detector = :d8)`. Only `ring_sum!`
calls this, and only under that setting: `solver.ring_plans` is `nothing`
otherwise, which keeps this function off the default configuration's inference
path. Indexing that field under `:delta4` would throw. See `detect_sum!`.

Collective, with the same halo and fold contract as `deriv_along!`.
"""
function ring_along!(out, f, solver::Solver, d::Int, σf::Int)
    fold = solver.folds[d]
    if fold === nothing
        apply_along!(out, solver.ring_plans[d], f, solver.decomp)
    else
        fold_apply!(out, f, solver, fold, σf, Val(:ring))
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
    @threaded nx*ny*nz for jk in outer_indices(ny, nz)
        j, k = Tuple(jk)
        @inbounds for i in 1:nx
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
    refresh_primitives!(solver, Q)

Update the primitive fields on `solver` — `rho`, `u`, `v`, `w`, `p`, `T_ion`,
`c`, `cp_mix`, and `Y` — from `Q` by exchanging rank-boundary halos and calling
`primitives!`. This operation is collective and idempotent.

During `compute_rhs!`, including source terms and boundary corrections, the
primitive fields correspond to the state being evaluated. In a `run!` callback,
they instead correspond to the input state of the fifth RK stage and predate the
final stage update and subsequent [`filter_state!`](@ref) pass.

A callback must therefore call this function before reading the primitive
fields. `save_vtk`, `dissipation_rate`, and the mixing diagnostics perform this
update internally. The conserved state `Q` is current in a callback;
`mixture_density` and related functions read it without depending on the
conserved layout.
"""
refresh_primitives!(solver::Solver, Q) =
    (exchange_state!(Q, solver.decomp); primitives!(solver, Q); solver)

"""
    compute_primitives_and_gradients!(solver, Q, primitives_current=false)

Refresh halos, primitives, and the physical-component velocity gradients from
`Q`. Sharing this sequence between the RHS and the diagnostics gives both the
same parity and curvature-correction routing, and gradients taken from the
current state.

`solver.grad_u[d, j]` is overwritten with the physical component (∇u)_{dj},
zeroed on a collapsed dimension, and the metric curvature terms are added on
top. Collective: every rank must call it, since each active dimension is a
distributed line solve.

Pass `primitives_current = true` when [`refresh_primitives!`](@ref) has already
run on this exact `Q` and only the gradients are wanted; the caller is then
responsible for the claim that nothing has touched `Q` since.
"""
function compute_primitives_and_gradients!(solver::Solver, Q,
                                           primitives_current::Bool=false)
    decomp = solver.decomp
    primitives_current || refresh_primitives!(solver, Q)
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
    return solver
end

"""
    compute_rhs!(solver, Q, dQ, primitives_current=false)

Evaluate dQ/dt into the interior of `dQ` from the conserved state `Q`
(boundary conditions should already be enforced on `Q`). Collapsed dimensions
contribute no derivatives; the axis dimension routes through parity-folded
plans with mirror-filled halos. The interior of `dQ` is zeroed first, so it is
overwritten rather than accumulated into.

This is collective: it exchanges halos, runs a distributed line solve per active
dimension per field, and calls `correct_rhs!` for every boundary condition,
which under NSCBC carries collectives of its own. Every rank must call it at the
same point in the step.

The primitives, the gradients, the artificial coefficients, `strain_mag`,
`flux`, and the scratch fields `tmp_a`, `tmp_b`, `sensor` and `sensor_sp` on
`solver` are all overwritten; so are `pairbuf` and `pairout` wherever a paired
fold exists, and `ring_buf` under `detector = :d8`. The docstring of
`compute_artificial!` records which of the sensor scratch fields are dead on
return and may therefore be borrowed by a later phase of the same call.

A trailing `primitives_current = true` skips the opening halo exchange and
primitives pass, and is valid only when the caller has just performed both on
this same `Q`. See [`compute_primitives_and_gradients!`](@ref); [`step!`](@ref)
passes it for the first RK stage of a `prepared` step, where [`max_rate`](@ref)
has already done the work. It is positional rather than a keyword so that
`bench/audit.jl` can still reach the body with `code_typed`, which sees only the
forwarding method of a function with keywords.
"""
function compute_rhs!(solver::Solver, Q, dQ, primitives_current::Bool=false)
    decomp = solver.decomp
    compute_primitives_and_gradients!(solver, Q, primitives_current)
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
        @threaded nx*ny*nz for jk in outer_indices(ny, nz)
            j, k = Tuple(jk)
            @inbounds for i in 1:nx
                dQ[i+o1, j+o2, k+o3, c] = 0
            end
        end
        for d in 1:3
            decomp.active[d] || continue
            Fdc = solver.flux[d, c]
            σ = solver.folds[d] === nothing ? 1 : solver.folds[d].sigflux[c]
            if unitgeom
                deriv_along!(solver.tmp_a, Fdc, solver, d, σ)
                @threaded nx*ny*nz for jk in outer_indices(ny, nz)
                    j, k = Tuple(jk)
                    @inbounds for i in 1:nx
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
                @threaded nx*ny*nz for jk in outer_indices(ny, nz)
                    j, k = Tuple(jk)
                    @inbounds for i in 1:nx
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


