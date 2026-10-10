# Solver construction. Nothing here runs on the step path except through a
# regrid or a restart, which rebuild refined patches with the builders below.
#
# `Solver(; ...)` proceeds in these phases, in this order:
#
# 1. Precision. The keyword method resolves the element type T from the
#    components passed explicitly (`_resolve_precision`, precision.jl),
#    converts them to T and fills in the defaults, then calls `_Solver`.
# 2. Validation. `_validate_configuration` checks the keyword ranges, the
#    transport and artificial-property settings and each boundary condition
#    against the metric and EOS. The rest of the validation in `_Solver`
#    (coordinate folds, symmetry planes, patch layout, refinement nesting,
#    interface rows, device residency) is interleaved with the quantities it
#    derives: the periodicity, the spacing `h`, the half-cell `coord_shift`,
#    the patch slabs and their faces, and the refined regions.
# 3. Same-level patches. A `patch_grid` of more than one patch leaves
#    `_Solver` here for `_build_patched_solver`, which builds one
#    decomposition, plan set and array set per patch and the interface
#    exchange records.
# 4. Root patch. The `Decomp`, the replicated extent check
#    (`check_block_extents`), the fold specs, the operator plans, the RHS
#    workspace pool and the root `Patch`.
# 5. Hierarchy. For each refined level: the tile owners and the level and
#    tile communicators (levels.jl), the patches this rank holds
#    (`_build_level_patches`, through `_build_fine_patch` on the host and
#    `_build_tile_stack` for a stacked device level), the level transfers
#    (`build_level_transfer`), the level exchange records, and the
#    `RegridSpec` when regridding is on.
# 6. Assembly. The concretely typed `Solver`, the geometry of every patch
#    (`init_geometry!`) and each parent patch's covered mask.
#
# Device residency is not a phase of its own: every array and plan is
# allocated through the backend (`field`, `backend_plan`) as it is built.

# Which physical faces the sensor operators close on the node-centred mirror,
# per dimension and side: those whose condition answers the detector's
# `sensor_mirror` hook (boundary.jl) with `true`.
_sensor_wall_faces(bcs) = ntuple(d -> (sensor_mirror(bcs[d][1]) === true,
                                       sensor_mirror(bcs[d][2]) === true), 3)

# The wall rows one sensor operator takes at one face, or `nothing` where it
# keeps the scheme's own. `use` gates the hook to the two operators whose rows
# fold onto the half-offset mirror: the `:gaussian` smoother and the `:d8`
# detector. The `:compact` smoother keeps the state filter's plans and its own
# one-sided rows, which reference the boundary node itself and carry no
# half-cell shift.
_sensor_wall_rows(scheme, wall::Bool, σ::Int, use::Bool) =
    (use && wall) ? wall_closures(scheme, σ) : nothing

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
[`Problem`](@ref); `art`, `deriv`, `cfl`, `control`, `polar_truncation`,
`implicit`, `n_halo` and `stretch` from [`Numerics`](@ref); `filt`, `filter_interval`,
`filter_cfl` and `filter_weighting` from [`StateFilter`](@ref) (its `scheme`,
`interval`, `cfl` and `weighting`); `interface_flux`, `interface_rhs` and
`interface_divergence` from [`PatchInterfaces`](@ref); and `dims`, `comm`,
`backend`, `precision` and `patch_grid` from [`Execution`](@ref). Without
`precision`, the element type is the one shared by the components passed
explicitly (`eos`, `transport`, `art`,
`deriv`, `filt`, `interface_divergence`), or `Float64` when none is passed.
Components left out are built at that type, and components of different
types raise an `ArgumentError`. The two keywords with no
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
- Without these folds, setup rejects a grid with a node at r = 0 or, under
  `SphericalMetric`, at θ = 0 or π. A collapsed θ sits at the low end of its
  range.
"""
function Solver(; n_global::NTuple{3,Int}, L_domain, bcs,
                precision::Union{Nothing,Type}=nothing,
                eos=nothing,
                transport::Union{Nothing,AbstractTransport}=nothing,
                art::Union{Nothing,ArtificialProperties}=nothing,
                deriv::Union{Nothing,AbstractCompactScheme}=nothing,
                filt::Union{Nothing,AbstractCompactScheme}=nothing,
                interface_divergence::Union{Nothing,AbstractCompactScheme}=nothing,
                kwargs...)
    eos === nothing || (eos = _as_eos(eos))
    T = _resolve_precision(precision, (; eos, transport, art, deriv, filt,
                                        interface_divergence))
    cv(x, default) = x === nothing ? default : _to_precision(T, x)
    return _Solver(T; n_global, L_domain, bcs,
                   eos=cv(eos, _default_ideal_mixture(T)),
                   transport=cv(transport, ConstantTransport{T}()),
                   art=cv(art, ArtificialProperties{T}()),
                   deriv=cv(deriv, lele_d1_6(T)),
                   filt=cv(filt, compact_filter(0.47, T)),
                   interface_divergence=cv(interface_divergence, nothing),
                   kwargs...)
end

# The checks of phase 2 that derive nothing: keyword ranges, the transport
# and artificial-property settings, and each boundary condition against the
# metric and EOS.
function _validate_configuration(transport, eos, art, bcs, metric, n_global,
                                 L_domain, origin, cfl, filter_interval,
                                 filter_cfl)
    validate_transport(transport, eos)
    validate_art(art, eos)
    all(>=(1), n_global) ||
        throw(ArgumentError("n_global must be at least 1 in every direction " *
                            "(1 collapses one), got $n_global"))
    all(L -> isfinite(L) && L > 0, L_domain) ||
        throw(ArgumentError("L_domain must hold three finite positive extents, " *
                            "got $L_domain"))
    all(isfinite, origin) ||
        throw(ArgumentError("origin must be finite, got $origin"))
    isfinite(cfl) && cfl > 0 ||
        throw(ArgumentError("cfl must be finite and positive, got $cfl"))
    filter_interval >= 0 ||
        throw(ArgumentError("filter_interval must be >= 0 (0 disables the state " *
                            "filter), got $filter_interval"))
    isfinite(filter_cfl) && filter_cfl >= 0 ||
        throw(ArgumentError("filter_cfl must be finite and >= 0 (0 disables the " *
                            "relaxation), got $filter_cfl"))
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
    art.detector in (:delta4, :d8, :species_d8) ||
        error("art.detector must be :delta4, :d8 or :species_d8, " *
              "got :$(art.detector)")
    art.species_flux in (:fickian, :bulk, :partial_density) ||
        error("art.species_flux must be :fickian, :bulk or :partial_density, " *
              "got :$(art.species_flux)")
    # Per-condition restrictions on geometry and EOS agreement (boundary.jl).
    for d in 1:3, side in 1:2
        validate_bc(bcs[d][side], metric, eos, d, side)
    end
    return nothing
end

# The configurations the implicit conduction covers: one patch on host
# storage, and ends whose temperature is even through a mirror plane or a
# fold, or periodic, which is what the staggered stage imposes. An isothermal
# wall, an inflow or outflow, and an extrapolated end carry other conditions
# on the temperature.
function _validate_implicit(bcs, patch_grid, refine, max_levels, backend)
    backend isa CPUBackend ||
        throw(ArgumentError("implicit conduction runs on the host backend only"))
    prod(patch_grid) == 1 && refine === nothing && something(max_levels, 1) == 1 ||
        throw(ArgumentError("implicit conduction runs on a single patch without " *
                            "refinement"))
    for d in 1:3, side in 1:2
        bc = bcs[d][side]
        ok = bc isa Union{PeriodicBC,SlipWallBC,SymmetryPlaneBC,AxisBC,OriginBC,PoleBC} ||
             (bc isa NoSlipWallBC && isnan(bc.Twall))
        ok || throw(ArgumentError(
            "implicit conduction supports periodic ends, adiabatic walls, symmetry " *
            "planes and coordinate folds; face $d/$side carries $(typeof(bc).name.name)"))
    end
    return nothing
end

# Reject a grid node on a coordinate singularity: r = 0 under either curvilinear
# metric, and sin θ = 0 under SphericalMetric. The volume Jacobian (r, or
# r² sin θ) vanishes there, so the first step divides by zero. The folds place
# the nodes half a cell off the singularity, so a folded end never trips this;
# an unfolded end or a collapsed dimension placed on it does. The node
# coordinates follow `global_xcoord`, in the solver's element type, and the
# tolerances absorb that type's representation of π and of the spacing.
function _check_singular_nodes(::Type{T}, metric, n_global, L_domain, origin,
                               stretch, coord_shift, h, angle_tol) where {T}
    metric isa Union{CylindricalMetric,SphericalMetric} || return nothing
    function first_node(on_singularity, d)
        ξ0 = stretch[d] === nothing ? T(origin[d]) : zero(T)
        for g in 1:n_global[d]
            ξ = ξ0 + coord_shift[d] + (g - 1) * h[d]
            x = stretch[d] === nothing ? ξ : stretch[d].x(ξ)
            on_singularity(Float64(x)) && return g
        end
        return nothing
    end
    r_tol = angle_tol * max(abs(Float64(origin[1])), Float64(L_domain[1]))
    g = first_node(r -> abs(r) <= r_tol, 1)
    if g !== nothing
        spherical = metric isa SphericalMetric
        throw(ArgumentError(
            "$(spherical ? "SphericalMetric" : "CylindricalMetric"): radial node " *
            "$g lies on the $(spherical ? "origin" : "axis") r = 0, where the " *
            "volume Jacobian vanishes; close that end with " *
            "$(spherical ? "OriginBC" : "AxisBC"), which offsets the nodes half " *
            "a cell, or start the radial domain above r = 0"))
    end
    metric isa SphericalMetric || return nothing
    g = first_node(θ -> abs(θ - π * round(θ / π)) <= angle_tol, 2)
    g === nothing && return nothing
    n_global[2] == 1 &&
        throw(ArgumentError(
            "SphericalMetric: the collapsed θ node lies on a pole (sin θ = 0), " *
            "where the volume Jacobian vanishes; a collapsed θ sits at the low " *
            "end of its domain, so start that domain away from 0 and π " *
            "(at π/2, for example)"))
    throw(ArgumentError(
        "SphericalMetric: θ node $g lies on a pole (sin θ = 0), where the " *
        "volume Jacobian vanishes; fold both θ ends with PoleBC over (0, π), " *
        "which offsets the nodes half a cell, or keep every θ node strictly " *
        "between 0 and π"))
end

function _Solver(::Type{T}; n_global::NTuple{3,Int}, L_domain, bcs,
                eos::EOS,
                equations=nothing,
                transport::AbstractTransport{T},
                art::ArtificialProperties{T},
                metric::Metric=CartesianMetric(),
                stretch::NTuple{3,Union{Nothing,Stretch}}=(nothing, nothing, nothing),
                sources=(),
                origin=(0.0, 0.0, 0.0),
                deriv::AbstractCompactScheme,
                filt::AbstractCompactScheme,
                cfl::Real=0.5, filter_interval::Int=1, filter_cfl::Real=0.35,
                filter_weighting::Symbol=:none,
                polar_truncation::Real=0,
                control::StepControl=StepControl(),
                dims=nothing, n_halo::Int=4,
                comm::MPI.Comm=MPI.COMM_WORLD,
                patch_grid::NTuple{3,Int}=(1, 1, 1),
                backend::AbstractBackend=CPUBackend(),
                interface_rhs::Symbol=:extended,
                interface_divergence::Union{Nothing,AbstractCompactScheme},
                interface_flux::Symbol=:ghost,
                refine::Union{Nothing,BlockRegion,Vector{BlockRegion}}=nothing,
                level_restriction::Symbol=:inject,
                level_interpolation_order::Union{Nothing,Int}=nothing,
                subcycle::Bool=false,
                regrid_interval::Int=0,
                tag_threshold::Real=0.02,
                tag_buffer::Int=4,
                tag_sensor_threshold::Real=0,
                tag_gradient_threshold::Real=0,
                tag_vorticity_threshold::Real=0,
                tag_predicate=nothing,
                untag_ratio::Real=2,
                tile_lifetime::Int=1,
                tile::Int=0,
                rebalance::Real=0,
                rebalance_persist::Int=2,
                max_levels::Union{Nothing,Int}=nothing,
                level_boundaries::Bool=true,
                implicit::Union{Nothing,ImplicitConduction}=nothing,
                positivity_limiter::Bool=false) where {T}
    # One compiled body for every configuration of a precision. Specialized,
    # this body was compiled again for each combination of the types of its
    # arguments, the user's tuple of boundary conditions among them, at
    # 12k LLVM instructions a time. It runs once per solver, assembles
    # objects and calls the routines that build the grid-sized arrays, which
    # are compiled for the concrete types they are called with.
    @nospecialize
    bcs = _face_conditions(bcs)
    _validate_configuration(transport, eos, art, bcs, metric, n_global, L_domain,
                            origin, cfl, filter_interval, filter_cfl)
    if positivity_limiter
        nlev_limited = something(max_levels, refine === nothing ? 1 :
                                 refine isa BlockRegion ? 2 : length(refine) + 1)
        # The limiter's face form holds on lines the closure rows close at an
        # interface (`_validate_positivity`), so the interfaces take them
        # before any plan is built.
        if interface_flux === :ghost && (prod(patch_grid) > 1 || nlev_limited > 1)
            MPI.Comm_rank(comm) == 0 &&
                @warn "positivity_limiter: the limiter takes interface lines closed by " *
                      "the closure rows, so the patch and level interfaces close with " *
                      "them (PatchInterfaces(flux = :closure)) in place of :ghost; " *
                      "this can lower the order of accuracy at an interface."
            interface_flux = :closure
        end
        _validate_positivity(bcs, metric, stretch, patch_grid, nlev_limited,
                             backend, eos,
                             implicit, equations === nothing ? NavierStokes1T(eos) :
                                       equations,
                             deriv, filt, filter_weighting, n_global, n_halo,
                             L_domain, T; interface_flux, interface_rhs,
                             interface_divergence, tile, level_restriction)
    end
    if implicit !== nothing
        _validate_implicit(bcs, patch_grid, refine, max_levels, backend)
        # The explicit half carries the molecular transport without its
        # conductivity; the implicit half reads the conductivity from the
        # wrapped model (imex.jl).
        transport = WithoutConduction(transport)
    end
    # ---- Coordinate-singularity folds -----------------------------------
    axis   = bcs[1][1] isa AxisBC
    orig1  = bcs[1][1] isa OriginBC
    pole_l = bcs[2][1] isa PoleBC
    pole_h = bcs[2][2] isa PoleBC
    pole_l == pole_h || error("PoleBC must be applied at both ends of θ")
    poles = pole_l
    # A domain endpoint supplied in the solver's storage type can only
    # represent π to that type's precision. Configuration checks should reject
    # a different angular range, not the Float32 representation of
    # the requested one.
    angle_tol = max(1e-10, 8 * Float64(eps(T)) * π)
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
        n_global[2] == 1 || isapprox(θsum, π; atol=angle_tol) ||
            error("OriginBC requires a θ range symmetric about π/2")
    end
    if poles
        metric isa SphericalMetric || error("PoleBC requires SphericalMetric")
        stretch[2] === nothing || error("folded dimensions cannot be stretched")
        abs(Float64(origin[2])) < 1e-14 &&
            isapprox(Float64(L_domain[2]), π; atol=angle_tol) ||
            error("PoleBC requires the θ domain (0, π)")
        n_global[3] == 1 || iseven(n_global[3]) ||
            error("resolved-φ PoleBC requires an even φ point count over 2π")
    end
    # ---- Face-centred symmetry planes -----------------------------------
    # A reflecting plane half a cell outside an end, folded by the same
    # machinery on any dimension and either end. The geometry restriction is
    # `validate_bc`'s; what is left here is the interaction with the
    # coordinate folds, the stretch and the grid spacing.
    symplane = ntuple(d -> (bcs[d][1] isa SymmetryPlaneBC,
                            bcs[d][2] isa SymmetryPlaneBC), 3)
    for d in 1:3
        (symplane[d][1] || symplane[d][2]) || continue
        stretch[d] === nothing ||
            error("folded dimensions cannot be stretched")
        # One FoldSpec per dimension carries one sigvel, and a coordinate
        # fold's is not the plane's.
        ((d == 1 && (axis || orig1)) || (d == 2 && poles)) &&
            error("dimension $d cannot carry both SymmetryPlaneBC and a " *
                  "coordinate fold (AxisBC, OriginBC, PoleBC)")
    end
    periodic = ntuple(d -> n_global[d] > 1 ? isperiodic(bcs[d][1]) : true, 3)
    (axis || orig1) && (periodic = (false, periodic[2], periodic[3]))
    poles && (periodic = (periodic[1], false, periodic[3]))
    periodic = ntuple(d -> periodic[d] && !(symplane[d][1] || symplane[d][2]), 3)
    for d in 1:3
        stretch[d] === nothing || !periodic[d] ||
            error("dimension $d: stretched dimensions must be non-periodic")
    end
    Lt = ntuple(d -> T(L_domain[d]), 3)
    active_g = ntuple(d -> n_global[d] > 1, 3)
    # Grid spacing: computational ξ ∈ [0,1] for stretched dims; half-offset
    # r ∈ (0, R] for axis grids (h = R/(N − ½), r₁ = h/2); standard otherwise.
    # Half-offset grids on folded dimensions: each folded end moves the
    # physical edge half a cell past the last node, so the line carries half
    # a cell less per folded end. Folding the low end alone (r: axis/origin,
    # a symmetry plane there) gives h = L/(N − ½) with node 1 at h/2; both
    # ends (θ poles, a plane at each end) give h = L/N; the high end alone
    # gives h = L/(N − ½) with node 1 on the physical edge.
    fold_lo_dim = ntuple(d -> (d == 1 && (axis || orig1)) || (d == 2 && poles) ||
                              symplane[d][1], 3)
    fold_hi_dim = ntuple(d -> (d == 2 && poles) || symplane[d][2], 3)
    h = ntuple(3) do d
        active_g[d] || return one(T)
        stretch[d] === nothing || return one(T) / (n_global[d] - 1)
        fold_lo_dim[d] && fold_hi_dim[d] && return Lt[d] / n_global[d]
        (fold_lo_dim[d] || fold_hi_dim[d]) && return Lt[d] / (n_global[d] - T(0.5))
        periodic[d] ? Lt[d] / n_global[d] : Lt[d] / (n_global[d] - 1)
    end
    coord_shift = ntuple(d -> fold_lo_dim[d] ? h[d] / 2 : zero(T), 3)
    _check_singular_nodes(T, metric, n_global, L_domain, origin, stretch,
                          coord_shift, h, angle_tol)
    filter_weighting in (:none, :volume) ||
        error("filter_weighting must be :none or :volume, got :$filter_weighting")
    # --- Patch layout ----------------------------------------------------
    npatch = prod(patch_grid)
    regions = patch_slabs(n_global, periodic, patch_grid)
    if npatch > 1
        (axis || orig1 || poles) &&
            error("patch decomposition across a coordinate fold is not " *
                  "supported; a folded run takes a single patch")
        any(any, symplane) &&
            error("patch decomposition across a SymmetryPlaneBC is not " *
                  "supported; a patched run takes SlipWallBC at that face")
        filt isa CompactScheme ||
            error("patch interfaces carry closure variants for a tridiagonal " *
                  "filter only")
        art.detector !== :d8 ||
            error("patch interfaces support the :delta4 and :species_d8 " *
                  "detectors only; the :d8 detector on every sensor takes a " *
                  "single patch")
        dims === nothing ||
            error("an explicit process grid cannot combine with patch_grid; " *
                  "each patch derives its own")
        interface_rhs in (:extended, :onesided) ||
            error("interface_rhs must be :extended or :onesided, " *
                  "got :$interface_rhs")
        ds = findfirst(>(1), patch_grid)
        stretch[ds] === nothing ||
            error("the patched dimension cannot be stretched")
        for r in regions
            r.extent[ds] >= max(9, n_halo + 2) ||
                error("patch extent $(r.extent[ds]) along dim $ds is below " *
                      "the scheme minimum; use fewer patches or more points")
        end
    end
    # --- Static refinement (levels.jl) -----------------------------------
    # `refines[ℓ]` is level ℓ's region in level ℓ−1's node space.
    refines = refine === nothing ? BlockRegion[] :
              refine isa BlockRegion ? [refine] : refine
    # Along a periodic dimension a region may start anywhere and run past the
    # seam; it is stored with its offset in [0, P), P the period of its
    # parent's node space (`_level_period`), which leaves any other region as
    # it was given.
    refines = [_canonical(rg, _level_period(n_global, periodic, ℓ - 1))
               for (ℓ, rg) in enumerate(refines)]
    # The hierarchy's depth, the root included. Levels beyond those `refine`
    # gives start with no tiles and are created by the regrid.
    nlev = something(max_levels, length(refines) + 1)
    nlev >= length(refines) + 1 ||
        error("max_levels = $nlev holds fewer levels than the $(length(refines)) " *
              "refined region(s) of refine and the root")
    if nlev > length(refines) + 1
        regrid_interval > 0 && tile > 0 ||
            error("max_levels = $nlev asks for levels that refine does not give; " *
                  "they start with no tiles, which requires tile > 0 and " *
                  "regrid_interval > 0")
    end
    if nlev == 1
        subcycle &&
            error("subcycle requires a refined region (the refine keyword)")
        regrid_interval == 0 ||
            error("regrid_interval requires a refined region (the refine " *
                  "keyword supplies the initial one)")
    end
    # --- Interface flux ---------------------------------------------------
    # Without a patch or level interface `interface_flux` selects nothing, and
    # the solver records `:closure`, so a single-patch run is the same whichever
    # is passed. The configurations the ghost fluxes do not support are
    # checked below, once the refinement checks have run.
    interface_flux in (:closure, :ghost) ||
        throw(ArgumentError("interface_flux must be :closure or :ghost, got " *
                            ":$interface_flux"))
    npatch > 1 || nlev > 1 || (interface_flux = :closure)
    # The level interpolation order follows the interface flux actually taken.
    level_interpolation_order =
        something(level_interpolation_order,
                  default_interpolation_order(deriv, interface_flux))
    schemes = SchemeSettings(deriv, filt, interface_divergence, interface_rhs,
                             level_interpolation_order, level_restriction)
    # --- Azimuthal mode truncation (modes.jl) ----------------------------
    # The ring projection is a Fourier series in θ over the whole circle on
    # host storage, and the limit table assumes a uniform Δr.
    isfinite(polar_truncation) && (polar_truncation == 0 || polar_truncation >= 1) ||
        throw(ArgumentError("polar_truncation must be 0 (off) or a margin of at " *
                            "least 1, got $polar_truncation"))
    if polar_truncation > 0
        metric isa CylindricalMetric ||
            error("polar_truncation applies to CylindricalMetric; the " *
                  "spherical form is not implemented")
        n_global[2] > 1 && periodic[2] &&
            isapprox(Float64(L_domain[2]), 2π; atol=angle_tol) ||
            error("polar_truncation requires θ resolved and periodic over 2π")
        stretch[1] === nothing ||
            error("polar_truncation requires an unstretched radial dimension")
        npatch == 1 && nlev == 1 ||
            error("polar_truncation takes a single patch without refinement")
        backend isa CPUBackend ||
            error("polar_truncation runs on the host backend only")
    end
    regrid_interval >= 0 || error("regrid_interval must be non-negative")
    tag_buffer >= 0 || error("tag_buffer must be non-negative")
    tag_threshold > 0 || error("tag_threshold must be positive")
    for (name, value) in ((:tag_sensor_threshold, tag_sensor_threshold),
                          (:tag_gradient_threshold, tag_gradient_threshold),
                          (:tag_vorticity_threshold, tag_vorticity_threshold))
        value >= 0 || error("$name must be non-negative (0 leaves it off)")
        value == 0 || regrid_interval > 0 ||
            error("$name is a regrid tag criterion; it requires regrid_interval > 0")
    end
    tag_predicate === nothing || regrid_interval > 0 ||
        error("tag_predicate is a regrid tag criterion; it requires " *
              "regrid_interval > 0")
    tag_predicate === nothing || tag_predicate isa Function ||
        error("tag_predicate must be a function (patch, I) -> Bool or nothing")
    # A ratio of one holds nothing below the tag threshold, which is the
    # hysteresis-free rule; below one the hold band would sit above the tag.
    untag_ratio >= 1 || error("untag_ratio must be at least 1 (1 disables the hold band)")
    tile_lifetime >= 1 || error("tile_lifetime must be at least 1 regrid check")
    # A lattice cell of `tile` parent nodes is a patch of tile + 1 nodes,
    # 3·tile + 1 fine nodes; the four-node minimum gives tile ≥ 3.
    tile == 0 || tile >= 3 ||
        error("tile must be 0 (one patch per level) or at least 3 parent nodes")
    # max/mean is at least one, so a threshold below one is not a setting.
    rebalance == 0 || rebalance >= 1 ||
        error("rebalance must be 0 (off) or a max/mean threshold of at least 1")
    rebalance == 0 || (tile > 0 && regrid_interval > 0) ||
        error("rebalance repartitions a tiled level at the regrid cadence; " *
              "it requires tile > 0 and regrid_interval > 0")
    rebalance_persist >= 1 || error("rebalance_persist must be at least 1")
    if regrid_interval > 0 && nlev > 2
        tile > 0 ||
            error("regridding more than one refined level requires tile > 0; " *
                  "$(nlev - 1) refined levels were asked for")
        rebalance == 0 ||
            error("rebalance repartitions the refined level of a two-level " *
                  "hierarchy; it cannot combine with $(nlev - 1) regridded levels")
        backend isa DeviceBackend &&
            error("regridding more than one refined level runs on the host " *
                  "backend only")
    end
    if nlev > 1
        MPI.Initialized() || MPI.Init(threadlevel=:funneled)
        MPI.Comm_size(comm) == 1 || level_restriction === :inject ||
            error("level_restriction = :filter restricts through a " *
                  "whole-patch line solve and is serial-only; use :inject " *
                  "under MPI")
        npatch == 1 ||
            error("refine cannot combine with a same-level patch_grid yet")
        # Each tile evaluates its geometry at its own nodes, which serves the
        # axisymmetric cylindrical metric as it does the Cartesian one. A
        # resolved θ or the spherical metric would carry angular scale factors
        # into the level transfers and the ghost fluxes, which are not built.
        metric isa CartesianMetric || (metric isa CylindricalMetric && !active_g[2]) ||
            error("refinement requires CartesianMetric or CylindricalMetric with " *
                  "θ collapsed (n_global[2] = 1)")
        all(isnothing, stretch) ||
            error("refinement requires an unstretched grid")
        # The axis of an r-z run is a self-paired parity fold, which the first
        # refined level may reach as it does a symmetry plane (below). The
        # spherical origin and poles are rejected with their metric above.
        (orig1 || poles) &&
            error("refinement across a coordinate fold is forbidden")
        level_restriction in (:inject, :filter) ||
            error("level_restriction must be :inject or :filter, " *
                  "got :$level_restriction")
        # The Lagrange tables are built for even orders 2 to 10. Every stencil
        # fits the gathered box, whose extent along a refined dimension is at
        # least 4 + 2·LEVEL_BUFFER = 12 nodes; up to order 6 it is centered at
        # every shell node, at order 8 the outermost ghost layer's stencil
        # sits one node inward of centered, and at order 10 two nodes inward
        # for the outermost layer and one for the next.
        level_interpolation_order in (2, 4, 6, 8, 10) ||
            error("level_interpolation_order must be 2, 4, 6, 8 or 10, " *
                  "got $level_interpolation_order")
        # The level shell is read from the fine box at an offset of
        # 3·LEVEL_BUFFER nodes; a halo wider than that would index the box's
        # own zero halo silently (`_write_fine_shell!`, `_impose_shell!`).
        n_halo <= 3 * LEVEL_BUFFER ||
            error("refinement supports n_halo ≤ $(3 * LEVEL_BUFFER), got $n_halo")
        # Each region is nested by the margin inside the patches of the level
        # above it, in that level's node space (the root's is the grid), at
        # every face but one on the domain boundary, which a region may reach
        # where the root's condition there qualifies
        # (`_level_boundary_condition`). Below the first refined level a
        # region reaching a symmetry plane or the r-z axis starts, or ends, at
        # its parent's node half a parent spacing from the fold, outside the
        # lattice coincident with the root's (`_level_span`).
        margin = max(n_halo, LEVEL_BUFFER)
        parent_regions = [BlockRegion((0, 0, 0), n_global)]
        for (ℓ, rg) in enumerate(refines)
            span = _level_span(n_global, active_g, ℓ - 1, bcs)
            period = _level_period(n_global, periodic, ℓ - 1)
            eligible = _level_boundary_eligible(bcs, active_g, periodic)
            # Along a periodic dimension a region may run across the seam but
            # not around it: one patch keeps the margin clear of its own other
            # end, so its box and its covered window each meet a parent node
            # once, and a tiled region spans the lattice's ring at most.
            for d in 1:3
                period[d] > 0 || continue
                most = tile == 0 ? period[d] - 2 * margin : period[d] + 1
                rg.extent[d] <= most ||
                    error("level $ℓ region $rg spans $(rg.extent[d]) level-$(ℓ - 1) " *
                          "nodes along periodic dimension $d, whose period is " *
                          "$(period[d]) nodes; " *
                          (tile == 0 ? "one refined patch spans at most $most, " *
                                       "keeping $margin nodes clear at either end, " *
                                       "and a tiled level (tile > 0) may close the ring" :
                                       "a tiled region spans at most the ring, $most"))
            end
            for d in 1:3
                if active_g[d]
                    rg.extent[d] >= 4 ||
                        error("level $ℓ region needs at least 4 parent nodes " *
                              "along dimension $d (9 fine points for the C8 " *
                              "filter)")
                    for side in 1:2
                        at_face = side == 1 ? rg.offset[d] + 1 == first(span[d]) :
                                  rg.offset[d] + rg.extent[d] == last(span[d])
                        at_face && !periodic[d] && !eligible[d][side] &&
                            error("level $ℓ region $rg reaches the " *
                                  "$(side == 1 ? "low" : "high") domain face of " *
                                  "dimension $d, whose $(nameof(typeof(bcs[d][side]))) " *
                                  "a refined level cannot carry; a level reaches " *
                                  "SlipWallBC, NoSlipWallBC, NSCBCOutflowBC, " *
                                  "NSCBCInflowBC, SymmetryPlaneBC and AxisBC " *
                                  "faces only")
                    end
                else
                    rg.offset[d] == 0 && rg.extent[d] == 1 ||
                        error("level $ℓ region must span collapsed dimension " *
                              "$d with offset 0 and extent 1")
                end
            end
            bnd = _boundary_faces(rg, span, eligible)
            folded = _level_fold_faces(bnd, bcs)
            # The box stops at a wall face, where the interpolation takes
            # one-sided stencils of the order's width over the box.
            for d in 1:3
                active_g[d] || continue
                n = _box_extent(rg, _box_buffer(active_g, bnd, folded))[d]
                n >= level_interpolation_order ||
                    error("level $ℓ region $rg spans $n level-$(ℓ - 1) nodes " *
                          "along dimension $d with its buffer, fewer than the " *
                          "level interpolation order $level_interpolation_order; " *
                          "widen the region")
            end
            if !_covered_by(_buffered(rg, active_g, margin, bnd), parent_regions, period)
                p = only(parent_regions)
                ranges = join(("offset $(p.offset[d] + margin):" *
                               "$(p.offset[d] + p.extent[d] - margin - rg.extent[d]) " *
                               "along dimension $d" for d in 1:3 if active_g[d]), ", ")
                # A region meant to reach a fold below the first level starts
                # outside the coincident lattice; name the offset that does.
                folds = join(("offset $(first(span[d]) - 1) reaches the " *
                              "$(nameof(typeof(bcs[d][1]))) of dimension $d"
                              for d in 1:3
                              if active_g[d] && ℓ > 1 && _level_fold_condition(bcs[d][1])),
                             ", ")
                error("level $ℓ region $rg must be nested at least $margin " *
                      "level-$(ℓ - 1) nodes inside the level-$(ℓ - 1) patches' " *
                      "own (not imposed) nodes: with its extent that is $ranges, " *
                      "counted on the level-$(ℓ - 1) lattice over the whole domain" *
                      (isempty(folds) ? "" : "; $folds") * ". " *
                      "AMR(initial = [shape, ...]) takes the levels as shapes in " *
                      "physical coordinates instead")
            end
            # The next level reads this one's own nodes: a one-patch level's
            # parent-fed boundary planes are imposed data and are eroded (the
            # tiled cover is checked face by face at construction).
            parent_regions = [_erode(_fine_region(rg, active_g, folded),
                                     ntuple(d -> (!bnd[d][1], !bnd[d][2]), 3), active_g)]
        end
    end
    # --- Interface divergence rows ------------------------------------------
    # The source scheme only supplies the divergence's closure rows at patch
    # and level interface ends, so a run without an interface would ignore it.
    if interface_divergence !== nothing
        npatch > 1 || nlev > 1 ||
            error("interface_divergence selects the flux divergence rows at a " *
                  "patch or level interface; this run has neither (patch_grid, " *
                  "refine)")
        idiv_rows = interface_divergence_rows(deriv, interface_divergence)
        # A regrid builds refined patches mid-run, down to 10 fine nodes along
        # a dimension (4 parent nodes; a tile of t parent nodes gives 3t + 1),
        # and a patch closed at both ends by these rows needs 2 rows + 1.
        min_fine = tile > 0 ? 3 * tile + 1 : 10
        regrid_interval == 0 || 2 * length(idiv_rows) + 1 <= min_fine ||
            error("interface_divergence '$(interface_divergence.name)' closes a " *
                  "line with $(length(idiv_rows)) rows per end, more than a " *
                  "regridded patch of $min_fine fine nodes admits; raise tile")
    end
    # --- Ghost-flux divergence -----------------------------------------------
    # `:ghost` differences the inviscid flux through an interface end with
    # the gradient plans, whose interface rows exist only under `:extended`;
    # the ghost fluxes carry the area factors and the Jacobian of an
    # unstretched Cartesian or axisymmetric cylindrical grid, and the
    # curvature terms of the latter's velocity gradient, and no other
    # metric's. A configuration they do not support raises rather than
    # falling back, since `:ghost` is the default.
    if interface_flux === :ghost
        interface_rhs === :extended ||
            throw(ArgumentError(
                "interface_flux = :ghost (the default) reads the gradient plans' " *
                "interface rows, which exist under interface_rhs = :extended only; " *
                "pass interface_flux = :closure with interface_rhs = :$interface_rhs"))
        (metric isa CartesianMetric || (metric isa CylindricalMetric && !active_g[2])) &&
            all(isnothing, stretch) ||
            throw(ArgumentError(
                "interface_flux = :ghost (the default) requires an unstretched " *
                "CartesianMetric, or CylindricalMetric with θ collapsed, at a patch " *
                "or level interface; pass interface_flux = :closure on this grid"))
        # A coarse-fine face's molecular ghost flux recovers the temperature
        # gradient from the conserved ones through the internal energy, which
        # `_temperature_gradient` inverts for the built-in models only.
        nlev == 1 || !_ghost_viscous(interface_flux, transport) ||
            _ghost_gradient_eos(eos) ||
            throw(ArgumentError(
                "interface_flux = :ghost (the default) with molecular transport at " *
                "a refined level supports IdealMixture, Nasa9Mixture and " *
                "StiffenedGas; got $(typeof(eos)); use interface_flux = :closure " *
                "for this EOS"))
    end
    # --- Device residency -------------------------------------------------
    # A DeviceBackend supports a decomposed patch, patched, refined or
    # tiled: halos, fold pairs, the interface records and the level
    # transfer's gathers and writes stage through the backend, and the
    # transfer chain runs on it. The `:filter` restriction (a whole-patch
    # host line solve) stays barred.
    if backend isa DeviceBackend
        refine === nothing || level_restriction === :inject ||
            error("level_restriction = :filter is host-only; use :inject " *
                  "on a DeviceBackend")
    end
    ds_split = npatch > 1 ? findfirst(>(1), patch_grid) : 0
    faces_all = [ntuple(3) do d
        d != ds_split && return (0, 0)
        P = patch_grid[d]
        lo = pid > 1 ? pid - 1 : (periodic[d] ? P : 0)
        hi = pid < P ? pid + 1 : (periodic[d] ? 1 : 0)
        (lo, hi)
    end for pid in 1:npatch]
    # The sensor smoother stands in for Cook's Gaussian test filter. `:compact`
    # reuses `filt`, which was the smoother before the option existed and keeps
    # every plan identical; `:gaussian` is the explicit nine-point stencil the
    # public Pyranda implementation uses, and carries no line solve.
    smoo = art.smoother === :gaussian ? gaussian_filter(T) : filt
    # The sensor detector. `:delta4` is the explicit undivided fourth
    # difference applied inside `delta4_sum!`, which needs no plan at all;
    # `:d8` is the pentadiagonal compact eighth derivative and needs one per
    # dimension and one pair per fold, matching the smoother. `:species_d8`
    # builds the same plans wherever a species sensor exists, and none
    # otherwise (`_ring_detector`).
    ring = compact_d8(T)
    ring_det = _ring_detector(art, nspecies(eos))
    # Reflecting-wall closure rows for the two sensor operators. Both schemes
    # fold their overhanging weights onto the half-offset mirror, which is half
    # a cell out at a wall node; at such a face they take the node-centred rows
    # the `:delta4` detector's mirror already reads (`wall_closures`). The
    # smoother's input is even at a wall, every detector output passing through
    # an absolute value, so it needs one sign; the detector also runs on the
    # velocity components and takes both.
    wall_face = _sensor_wall_faces(bcs)
    swrow(d, side) = _sensor_wall_rows(smoo, wall_face[d][side], 1,
                                       art.smoother === :gaussian)
    rwrow(d, side, σw) = _sensor_wall_rows(ring, wall_face[d][side], σw,
                                           ring_det)
    equations = equations === nothing ? NavierStokes1T(eos) : equations
    equations isa EquationSet || error("equations must be an EquationSet")
    equations.n_species == nspecies(eos) ||
        error("equation set carries $(equations.n_species) species; " *
              "EOS has $(nspecies(eos))")
    n_species = equations.n_species
    n_cons = equations.n_cons
    if npatch > 1
        solver = _build_patched_solver(T, n_global, periodic, regions, faces_all,
                                       patch_grid, bcs, eos, equations, transport,
                                       art, metric, stretch, sources, origin, Lt,
                                       coord_shift, h, deriv, filt, smoo, cfl,
                                       filter_interval, filter_cfl, filter_weighting,
                                       control, n_halo, comm, backend, interface_rhs,
                                       n_cons, n_species; interface_divergence,
                                       interface_flux, schemes)
        # One limiter per patch this rank holds, in the order of its patches.
        if positivity_limiter
            held = getfield(solver, :patches)
            solver.positivity = PatchLimiters(Any[PositivityLimiter(PatchSolver(solver, p))
                                                  for p in held], Any[p for p in held])
        end
        return solver
    end
    decomp = Decomp{T}(n_global, periodic; dims=dims, n_halo=n_halo, comm=comm)
    truncation = mode_truncation(T, polar_truncation, decomp,
                                 T(origin[1]) + coord_shift[1], h[1], n_global[2],
                                 n_cons, equations.i_mom[1]:equations.i_mom[2])
    # The per-rank extent check in `plan_direction` would raise on some ranks
    # only when the blocks differ in size; this one is replicated.
    check_block_extents(n_global, decomp.dims,
                        ntuple(d -> !periodic[d] && !fold_lo_dim[d], 3),
                        ntuple(d -> !periodic[d] && !fold_hi_dim[d], 3),
                        d -> (
        (deriv, nothing, nothing), (filt, nothing, nothing),
        (art.smoother === :gaussian ? ((smoo, swrow(d, 1), swrow(d, 2)),) : ())...,
        (ring_det ? ((ring, rwrow(d, 1, 1), rwrow(d, 2, 1)),) : ())...))
    mkd(sch, d; kw...) =
        backend_plan(backend, plan_direction(decomp, sch, d, h[d]; kw...))
    f() = field(backend, decomp)

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
        partner = cart_rank(decomp, crd)
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
                 loc ? empty_field(backend, T) : field(backend, decomp))
    end
    function foldspec(d, lo, hi, pdim, revdim, sigvel, sigflux)
        dp = (mkd(deriv, d; lo_fold=(lo ? 1 : nothing), hi_fold=(hi ? 1 : nothing)),
              mkd(deriv, d; lo_fold=(lo ? -1 : nothing), hi_fold=(hi ? -1 : nothing)))
        fp = (mkd(filt, d; lo_fold=(lo ? 1 : nothing), hi_fold=(hi ? 1 : nothing)),
              mkd(filt, d; lo_fold=(lo ? -1 : nothing), hi_fold=(hi ? -1 : nothing)))
        # A fold's far end may be a physical wall, as the outer end of a radial
        # line is, and takes the same wall rows an unfolded end does. The
        # folded end itself is open and takes none.
        sw(side, folded) = folded ? nothing : swrow(d, side)
        rw(side, folded, σw) = folded ? nothing : rwrow(d, side, σw)
        # `:compact` aliases the filter plans to avoid duplicating them.
        # This matches `smooth!` before it had an operator of its
        # own, so the default path keeps both its answer and its footprint.
        sp = art.smoother === :compact ? fp :
             (mkd(smoo, d; lo_fold=(lo ? 1 : nothing), hi_fold=(hi ? 1 : nothing),
                  lo_closures=sw(1, lo), hi_closures=sw(2, hi)),
              mkd(smoo, d; lo_fold=(lo ? -1 : nothing), hi_fold=(hi ? -1 : nothing),
                  lo_closures=sw(1, lo), hi_closures=sw(2, hi)))
        ringfold(σg, σw) =
            mkd(ring, d; lo_fold=(lo ? σg : nothing), hi_fold=(hi ? σg : nothing),
                lo_closures=rw(1, lo, σw), hi_closures=rw(2, hi, σw))
        # One plan per wall sign, the pair aliased to a single plan where the
        # far end is no wall, so a fold without one carries one plan per ghost
        # parity rather than two.
        farwall = (!lo && wall_face[d][1]) || (!hi && wall_face[d][2])
        function rpair(σg)
            even = ringfold(σg, 1)
            farwall ? (even, ringfold(σg, -1)) : (even, even)
        end
        rp = ring_det ? (rpair(1), rpair(-1)) :
             ((nothing, nothing), (nothing, nothing))
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
    # A symmetry plane is self-paired (each line continues into itself, so no
    # pairing dimension), the normal velocity is the only odd component, and
    # the area factor is independent of the folded coordinate, hence even.
    for d in 1:3
        (symplane[d][1] || symplane[d][2]) || continue
        sv = ntuple(j -> j == d ? -1 : 1, 3)
        fs = foldspec(d, symplane[d][1], symplane[d][2], 0, 0, sv,
                      flux_parities(equations, sv, d, 1))
        folds = (d == 1 ? fs : folds[1], d == 2 ? fs : folds[2],
                 d == 3 ? fs : folds[3])
    end
    deriv_plans = ntuple(d -> decomp.active[d] && folds[d] === nothing ?
                         mkd(deriv, d) : nothing, 3)
    filter_plans = ntuple(d -> decomp.active[d] && folds[d] === nothing ?
                         mkd(filt, d) : nothing, 3)
    smooth_plans = art.smoother === :compact ? filter_plans :
                   ntuple(d -> decomp.active[d] && folds[d] === nothing ?
                          mkd(smoo, d; lo_closures=swrow(d, 1),
                              hi_closures=swrow(d, 2)) : nothing, 3)
    # `nothing`, not a tuple of nothings: `detect_sum!` dispatches on this
    # field's type to decide which detector runs, so where no sensor takes
    # `:d8` (`_ring_detector`) the whole d8 path (`ring_sum!`, `ring_along!`,
    # and the `apply_along!` call taking a possibly-absent plan) is not
    # reachable from inference and costs such a configuration nothing.
    # A tuple would not serve: a fully folded run carries no plans here even
    # under `:d8`, since those live on the FoldSpec.
    #
    # Each dimension's entry is a pair indexed by the sign of the field across
    # a reflecting wall, aliased to one plan where neither face of the
    # dimension is a wall, so a dimension without one carries a single plan.
    function ringpair(d)
        (decomp.active[d] && folds[d] === nothing) || return nothing
        even = mkd(ring, d; lo_closures=rwrow(d, 1, 1), hi_closures=rwrow(d, 2, 1))
        (wall_face[d][1] || wall_face[d][2]) || return (even, even)
        (even, mkd(ring, d; lo_closures=rwrow(d, 1, -1),
                   hi_closures=rwrow(d, 2, -1)))
    end
    ring_plans = ring_det ? ntuple(ringpair, 3) : nothing
    paired_fold = any(fold -> fold !== nothing && fold.pair !== nothing, folds)
    orig = ntuple(d -> stretch[d] === nothing ? T(origin[d]) : zero(T), 3)
    bcs_t = ntuple(d -> (bcs[d][1], bcs[d][2]), 3)
    # One RHS scratch pool per rank, seeded with the root patch's set and
    # handed to every refined patch below (patches.jl).
    ws_pool = rhs_workspace_pool(backend, T)
    ws_root = rhs_workspace!(ws_pool, backend, decomp, n_species, n_cons,
                             ring_det,
                             _grad_Q_columns(art, n_species, n_cons), nlev > 1)
    patch = Patch(1, 0, regions[1], comm, decomp, h,
                  ntuple(d -> (0, 0), 3), bcs_t, folds,
                  deriv_plans, deriv_plans, filter_plans, smooth_plans, ring_plans,
                  # Only a paired fold's butterfly reads these; a self-paired
                  # fold (the axisymmetric axis, a symmetry plane) needs
                  # neither, so it carries no padded field of its own.
                  paired_fold ? f() : empty_field(backend, T),
                  paired_fold ? f() : empty_field(backend, T),
                  f(), f(), f(), f(), f(), f(), f(), f(),
                  [f() for _ in 1:n_species],
                  f(), f(), f(),
                  [f() for _ in 1:n_species],
                  f(), (f(), f(), f()), (f(), f(), f()), f(), f(), f(),
                  ws_root, _covered_mask(decomp),
                  nlev > 1 ? f() : empty_field(backend, T),
                  _empty_level_scratch(empty_field(backend, T)),
                  ntuple(_ -> similar(empty_field(backend, T), T, 0, 0, 0, 0), 3))
    if nlev == 1
        patches = [patch]
        solver = Solver{T,typeof(equations),typeof(eos),typeof(transport),typeof(metric),
                        typeof(stretch),typeof(sources),typeof(patch)}(
                      equations, eos, transport, art, metric, stretch, sources,
                      Lt, orig, coord_shift, h,
                      T(cfl), filter_interval, T(filter_cfl), filter_weighting, control,
                      n_global, patches, regions, decomp.comm,
                      GhostRecord{T}[], GhostRecord{T}[], PlaneRecord{T}[],
                      [Level{T}(0, root_level_comm(comm), [1],
                                LevelTransfer{T}[])], false, nothing,
                      zero(T), zero(T), 0, zero(T), zero(T),
                  ntuple(_ -> zero(T), 3), 0.0, 0.0, 0.0, 0.0, FloorTally(),
                  interface_flux, schemes, truncation, nothing, nothing, nothing)
        init_geometry!(solver)
        # The conduction stage reads the metric, so it is built last.
        implicit === nothing || (solver.implicit = ImexIntegrator(solver, implicit))
        positivity_limiter && (solver.positivity = PositivityLimiter(solver))
        return solver
    end
    # --- Refined patches and their couplings (levels.jl) ------------------
    # Level ℓ covers `refines[ℓ]` of level ℓ − 1: one patch over the region
    # exactly with `tile = 0`, the lattice tiles meeting it otherwise.
    fines = Patch{T}[]
    root_lc = root_level_comm(comm)
    levels = [Level{T}(0, root_lc, [1], LevelTransfer{T}[])]
    parent_regions = [BlockRegion((0, 0, 0), n_global)]
    parent_local = [1]        # solver index of each parent tile; 0 if not held
    # The parent nodes a child may read: the root's are all its own; a
    # refined parent's parent-fed planes are imposed data and are eroded.
    parent_valid = parent_regions
    parent_h = h
    margin = max(n_halo, LEVEL_BUFFER)
    parent_lc = root_lc
    level_tiles = Vector{BlockRegion}[]     # every level's tiles, on every rank
    for (ℓ, rg) in enumerate(refines)
        extent = _level_extent(n_global, active_g, ℓ - 1)
        span = _level_span(n_global, active_g, ℓ - 1, bcs)
        period = _level_period(n_global, periodic, ℓ - 1)
        eligible = _level_boundary_eligible(bcs, active_g, periodic)
        if tile == 0
            tregions = [rg]
        else
            # Clip the lattice to the parent patches' bounding box less the
            # margin, or to the box itself at a domain face `rg` reaches; a
            # tile that then still leaves the union is refused. Along a
            # periodic dimension where `rg` reaches past that clip, into the
            # margin band of the seam or across it, the lattice wraps instead:
            # its last cell ends on the seam, and the cells on either side of
            # the seam are neighbors.
            rb = _boundary_faces(rg, span, eligible)
            plo = ntuple(d -> minimum(r.offset[d] for r in parent_regions), 3)
            phi = ntuple(d -> maximum(r.offset[d] + r.extent[d] for r in parent_regions), 3)
            wrap = ntuple(d -> period[d] > 0 &&
                               (rg.offset[d] - margin < plo[d] ||
                                rg.offset[d] + rg.extent[d] + margin > phi[d]) ?
                               period[d] : 0, 3)
            lo = ntuple(d -> wrap[d] > 0 ? 1 : plo[d] + 1 + (rb[d][1] ? 0 : margin), 3)
            hi = ntuple(d -> wrap[d] > 0 ? wrap[d] + 1 :
                             phi[d] - (rb[d][2] ? 0 : margin), 3)
            tregions = _level_tiles(rg, active_g, tile, lo, hi,
                                    ntuple(d -> wrap[d] > 0 ? wrap[d] + 1 : extent[d], 3),
                                    wrap)
            isempty(tregions) &&
                error("level $ℓ region admits no tile of edge $tile inside " *
                      "the nesting margin")
        end
        push!(level_tiles, tregions)
        faces = _tile_faces(tregions, period)
        boundaries = [_boundary_faces(tr, span, eligible) for tr in tregions]
        folded_all = [_level_fold_faces(b, bcs) for b in boundaries]
        for (tr, bnd) in zip(tregions, boundaries)
            _covered_by(_buffered(tr, active_g, margin, bnd), parent_valid, period) ||
                error("level $ℓ tile $tr must be nested at least $margin " *
                      "level-$(ℓ - 1) nodes inside the level-$(ℓ - 1) patches' " *
                      "own (not imposed) nodes")
        end
        # The level's geometry is derived, not read off built patches, so a
        # rank outside the level's subset carries the same node spaces into
        # the next level's nesting checks as its owners do.
        fine_regions = [_fine_region(tr, active_g, fo)
                        for (tr, fo) in zip(tregions, folded_all)]
        imposed_all = [ntuple(d -> (f[d][1] == 0 && !b[d][1], f[d][2] == 0 && !b[d][2]), 3)
                       for (f, b) in zip(faces, boundaries)]
        next_h = ntuple(d -> active_g[d] ? parent_h[d] / 3 : parent_h[d], 3)
        local_of = zeros(Int, length(tregions))   # solver index of each tile
        if parent_lc.owned
            # This level is owned by the first np ranks of the parent's
            # subset, np the union of the tile owner ranges `_tile_owners`
            # lays out, and this rank holds the tiles of its own range.
            # `split_level_comm` and `split_tile_comm` are collective over
            # the parent's and the level's communicators, and the ranges
            # depend only on the tile geometry, which every rank holds, so
            # all ranks pass the same inputs with no communication beyond
            # the splits themselves.
            owners, np_level = _tile_owners(tregions, active_g, parent_lc.size)
            lc = split_level_comm(parent_lc, np_level)
            group = lc.owned ? split_tile_comm(lc, owners) : absent_tile_group()
            held = [ti for ti in eachindex(tregions) if owners[ti] == group.ranks]
            id0 = 1 + length(fines)
            built, stacks = _build_level_patches(T, tregions, held, faces, active_g,
                                                 parent_h, n_halo, group.comm, deriv,
                                                 filt, smoo, art.smoother,
                                                 interface_rhs, backend, ws_pool,
                                                 n_species, n_cons,
                                                 _grad_Q_columns(art, n_species, n_cons),
                                                 id0, ℓ, tile; interface_divergence,
                                                 ghost_viscous=
                                                     _ghost_viscous(interface_flux,
                                                                    transport),
                                                 ring=_ring_detector(art, n_species),
                                                 boundaries, bcs, root_folds=folds,
                                                 n_sensed=_sensed_field_count(art, tile,
                                                                              n_species))
            append!(fines, built)
            indices = [id0 + k for k in eachindex(held)]
            for (k, ti) in enumerate(held)
                local_of[ti] = id0 + k
            end
            decomp_at(li) = li == 0 ? nothing : li == 1 ? decomp :
                            fines[li - 1].decomp
            transfers = LevelTransfer{T}[]
            for (ti, tr) in enumerate(tregions)
                pids = _parents_of(tr, active_g, collect(eachindex(parent_regions)),
                                   parent_regions, period)
                push!(transfers, build_level_transfer(
                    T, tr, active_g, n_halo, parent_regions[pids],
                    parent_local[pids], local_of[ti], level_restriction, n_cons,
                    subcycle, decomp_at(local_of[ti]), parent_lc.comm,
                    length(owners[ti]), faces[ti];
                    interpolation_order=level_interpolation_order,
                    gradient_deriv=_ghost_viscous(interface_flux, transport) ?
                                   deriv : nothing,
                    parent_h=parent_h, boundary=boundaries[ti],
                    folded=folded_all[ti], period))
            end
            coupling = build_level_coupling(T, parent_lc.comm, transfers,
                                            parent_regions,
                                            map(decomp_at, parent_local),
                                            map(decomp_at, local_of))
            if lc.owned
                # A record's partner is a rank number in the communicator the
                # exchange runs over, here the level's own; that numbering
                # may differ from a tile's Cartesian communicator's.
                records = _level_records(T, lc.comm, fine_regions, held, indices,
                                         [fines[li - 1].decomp for li in indices],
                                         n_cons,
                                         _level_period(n_global, periodic, ℓ))
                push!(levels, Level{T}(ℓ, lc, owners, group, held, indices,
                                       transfers, records; stacks, coupling))
            else
                # The level's transfers and coupling are still held: the
                # parent's side of the box and restriction exchanges runs on
                # every rank of the parent's communicator.
                push!(levels, Level{T}(ℓ, lc, owners, group, held, indices,
                                       transfers; stacks, coupling))
            end
            parent_lc = lc
        else
            # Outside the parent's subset, and so outside this level and every
            # level below it: no patch, no transfer, no communicator.
            push!(levels, Level{T}(ℓ, absent_level_comm(), Int[],
                                   LevelTransfer{T}[]))
        end
        parent_local = local_of
        parent_regions = fine_regions
        parent_valid = [_erode(r, imp, active_g)
                        for (r, imp) in zip(fine_regions, imposed_all)]
        parent_h = next_h
    end
    # The levels `refine` does not give, up to `max_levels`, start with no
    # tiles: no owners, no transfers and an absent communicator on every rank,
    # the form a regridded level takes when its last tile is dropped.
    for ℓ in (length(refines) + 1):(nlev - 1)
        push!(levels, Level{T}(ℓ, absent_level_comm(), UnitRange{Int}[],
                               absent_tile_group(), Int[], Int[], LevelTransfer{T}[]))
        push!(level_tiles, BlockRegion[])
    end
    # A `Vector{Patch}`: the element type is fixed for the life of the solver
    # and a regrid may hand this rank tiles later, whose types differ from the
    # root's (the boundary-condition tuple, and on a device backend the view
    # storage). A typed vector, not a splat: `[patch, fines...]` compiles a
    # `promote_typeof` specialization per distinct patch count, 0.5 s on the
    # CPU backend and 1.7 s on the device backend each.
    patches = Patch[patch]
    append!(patches, fines)
    # The tag sweep's scratch spans the root's padded block, whose extent
    # is fixed for the life of the solver (a regrid moves the refined level
    # only).
    regrid = regrid_interval == 0 ? nothing :
             RegridSpec{T}(regrid_interval, T(tag_threshold), tag_buffer,
                           margin, n_halo, interface_rhs,
                           deriv, filt, smoo, backend, tile, 0,
                           Float64(rebalance), rebalance_persist, 0, 1.0,
                           0.0, 0.0, 0.0,
                           T(tag_sensor_threshold), T(tag_gradient_threshold),
                           T(tag_vorticity_threshold), tag_predicate,
                           zeros(Int8, ntuple(d -> decomp.n_local[d] +
                                                   2 * decomp.n_halo_d[d], 3)),
                           T(untag_ratio), tile_lifetime, 0,
                           Dict(r => 0 for r in level_tiles[1]),
                           interface_divergence,
                           level_interpolation_order, level_restriction,
                           [Dict(r => 0 for r in regions)
                            for regions in level_tiles[2:end]], level_boundaries)
    solver = Solver{T,typeof(equations),typeof(eos),typeof(transport),typeof(metric),
                    typeof(stretch),typeof(sources),eltype(patches)}(
                  equations, eos, transport, art, metric, stretch, sources,
                  Lt, orig, coord_shift, h,
                  T(cfl), filter_interval, T(filter_cfl), filter_weighting, control,
                  n_global, patches, regions, decomp.comm,
                  GhostRecord{T}[], GhostRecord{T}[], PlaneRecord{T}[],
                  levels, subcycle, regrid,
                  zero(T), zero(T), 0, zero(T), zero(T),
                  ntuple(_ -> zero(T), 3), 0.0, 0.0, 0.0, 0.0, FloorTally(),
                  interface_flux, schemes, ModeTruncation{T}(), nothing, nothing,
                  nothing)
    for p in getfield(solver, :patches)
        init_geometry!(PatchSolver(solver, p))
    end
    # The patches' limiters are built as `run!` starts, and again for the
    # patches a regrid replaces (`_follow_patches!`).
    positivity_limiter &&
        (solver.positivity = PatchLimiters(Any[], Any[],
                                           _limiter_least_box(deriv, filt, interface_rhs,
                                                              interface_divergence, n_halo,
                                                              T)))
    # Each held patch's covered mask from the regions of the level below,
    # which every rank of the patch's level holds (`_fill_covered!`).
    for ℓ in 1:length(levels)-1
        child_regions = [lt.region for lt in levels[ℓ + 1].transfers]
        for li in levels[ℓ].patches
            _fill_covered!(patches[li], child_regions,
                           _level_period(n_global, periodic, ℓ - 1))
        end
    end
    return solver
end

# Refined patch construction over the region `refine` (in the parent level's
# node space; `h` is the parent level's spacing), shared between the `Solver`
# constructor and `regrid!`. It takes the schemes, not plans off an existing
# patch, because a regrid changes the extents and every plan must be
# rebuilt. `ws_pool` carries the rank's RHS scratch sets: a tile of an extent
# already held reuses one, and only an unlike extent allocates.
function _build_fine_patch(::Type{T}, refine::BlockRegion,
                           active_g::NTuple{3,Bool}, h::NTuple{3,T},
                           n_halo::Int, comm::MPI.Comm, deriv, filt, smoo,
                           smoother::Symbol,
                           interface_rhs::Symbol, backend::AbstractBackend,
                           ws_pool::AbstractVector,
                           n_species::Int, n_cons::Int, grad_Q_columns::Int, id::Int,
                           level::Int,
                           faces::NTuple{3,NTuple{2,Int}}=ntuple(d -> (0, 0), 3);
                           interface_divergence=nothing,
                           ghost_viscous::Bool=false,
                           ring::Bool=false,
                           boundary::NTuple{3,NTuple{2,Bool}}=_NO_BOUNDARY,
                           bcs=nothing, root_folds=nothing,
                           n_sensed::Int=0, slot::Int=1) where {T}
    folded = bcs === nothing ? _NO_BOUNDARY : _level_fold_faces(boundary, bcs)
    region_f, decomp_f, hf = _fine_decomp(T, refine, active_g, h, n_halo, comm, folded)
    fbcs = _fine_bcs(active_g, faces, boundary, bcs)
    plans = _fine_plans(decomp_f, hf, deriv, filt, smoo, interface_rhs, backend;
                        interface_divergence, ring, boundary,
                        wall_faces=_sensor_wall_faces(fbcs),
                        gaussian=smoother === :gaussian, folded, root_folds)
    g() = field(backend, decomp_f)
    empty3 = empty_field(backend, T)
    # `ring` adds the `:d8` ringing buffer; `grad_Q_columns` selects the conserved
    # gradients of the shared-D_b species channels, which a refined patch
    # differences as the root does. A refined patch may be a parent itself,
    # so it takes the derivative mask's scratch as a refined root does.
    # `slot`, the tile's place among its level's tiles on this rank, spreads
    # a level evaluated concurrently over several sets (`rhs_workspace!`).
    ws = rhs_workspace!(ws_pool, backend, decomp_f, n_species, n_cons,
                        ring, grad_Q_columns, true, slot)
    # A tile of a level whose artificial coefficients are computed level-wide
    # keeps the velocity gradients of that pass for its right-hand side. A
    # stacked tile has them in its own block of the stack's set already.
    n_sensed > 0 && (ws = _own_gradients(ws, g))
    scratch = _level_scratch(empty3, refine, active_g, n_halo, n_cons,
                             MPI.Comm_size(comm), MPI.Comm_rank(comm);
                             gradient_deriv=ghost_viscous ? deriv : nothing,
                             backend, fine_decomp=decomp_f, hf, boundary, folded)
    # Every face of a refined patch but one on the domain boundary is an
    # interface end, a coarse-fine or a same-level one, so each dimension
    # with such an end takes a ghost-flux array.
    gflux = _ghost_flux_arrays(() -> parent(allocate_state(backend, decomp_f, n_cons)),
                               similar(empty3, T, 0, 0, 0, 0),
                               ghost_viscous || GHOST_FLUX_REMAINDER[] ?
                                   _interface_dims(active_g, boundary) :
                                   (false, false, false))
    return _assemble_patch(id, level, region_f, comm, decomp_f, hf, faces,
                           fbcs, plans, empty3,
                           _patch_arrays(g, n_species, n_sensed), ws,
                           _covered_mask(decomp_f),
                           scratch, gflux)
end

# The dimensions of a refined patch with an interface end: active, and not
# closed by the domain boundary at both faces.
_interface_dims(active_g::NTuple{3,Bool}, boundary::NTuple{3,NTuple{2,Bool}}) =
    ntuple(d -> active_g[d] && !(boundary[d][1] && boundary[d][2]), 3)

# A patch's `ghost_flux` arrays: `g4()` in each dimension of `dims`, the
# zero-extent `empty4` in the others.
_ghost_flux_arrays(g4::F, empty4, dims::NTuple{3,Bool}) where {F} =
    ntuple(d -> dims[d] ? g4() : empty4, 3)

# The refined region's node space, decomposition and spacing: parent-level
# node g is refined-level node 3(g − 1) + 1, and a folded face adds the node
# beyond (`_fine_region`).
function _fine_decomp(::Type{T}, refine::BlockRegion, active_g::NTuple{3,Bool},
                      h::NTuple{3,T}, n_halo::Int, comm::MPI.Comm,
                      folded::NTuple{3,NTuple{2,Bool}}=_NO_BOUNDARY) where {T}
    hf = ntuple(d -> active_g[d] ? h[d] / 3 : h[d], 3)
    region_f = _fine_region(refine, active_g, folded)
    pper_f = ntuple(d -> !active_g[d], 3)
    np_f = (MPI.Initialized() || MPI.Init(threadlevel=:funneled);
            MPI.Comm_size(comm))
    decomp_f = Decomp{T}(region_f.extent, pper_f;
                         dims=_amr_dims(region_f.extent,
                                        ntuple(d -> region_f.extent[d] > 1, 3),
                                        np_f),
                         n_halo=n_halo, comm=comm)
    return region_f, decomp_f, hf
end

# A refined patch's plans on `backend`: gradient, divergence, filter,
# smoother and, where `ring` (`_ring_detector`), the detector. `ntiles` and `stride`
# plan the batched device solve of a stacked level's spanning patch
# (lines_device.jl); the default is one patch's plans.
function _fine_plans(decomp_f::Decomp{T}, hf, deriv, filt, smoo,
                     interface_rhs::Symbol, backend::AbstractBackend;
                     ntiles::Int=1, stride::Int=0, interface_divergence=nothing,
                     ring::Bool=false,
                     boundary::NTuple{3,NTuple{2,Bool}}=_NO_BOUNDARY,
                     wall_faces::NTuple{3,NTuple{2,Bool}}=_NO_BOUNDARY,
                     gaussian::Bool=true,
                     folded::NTuple{3,NTuple{2,Bool}}=_NO_BOUNDARY,
                     root_folds=nothing) where {T}
    mkf(sch, d; kw...) =
        backend_plan(backend, plan_direction(decomp_f, sch, d, hf[d]; kw...,
                                             lines_factor=ntiles); ntiles, stride)
    ext_f = interface_rhs === :extended
    icd = ext_f ? interface_closures(deriv) : nothing
    icf = ext_f ? interface_closures(filt) : nothing
    ivd = ext_f || interface_divergence !== nothing ?
          interface_divergence_rows(deriv, interface_divergence) : nothing
    # Every face of a refined patch but one on the domain boundary closes
    # with the interface rows and reads ghosts; the boundary condition
    # (`_fine_bcs`) only records where they come from. A face on the domain
    # boundary keeps the scheme's own rows (`nothing` here), as the root's
    # face there does. The divergence takes the one-sided interface rows
    # instead (`interface_divergence_rows`), since a flux array has no
    # ghosts. A source scheme selects those rows under either
    # `interface_rhs`; without one, `:onesided` keeps the gradient plans' own
    # rows for the divergence. A dimension closed by the boundary at both
    # faces has no interface end, and its divergence plans are its gradient
    # plans, as the root's are.
    at(d, side, rows) = boundary[d][side] ? nothing : rows
    iface = _interface_dims(ntuple(d -> decomp_f.active[d], 3), boundary)
    dplans_f = ntuple(d -> decomp_f.active[d] ?
        mkf(deriv, d; lo_closures=at(d, 1, icd), hi_closures=at(d, 2, icd)) :
        nothing, 3)
    vplans_f = ntuple(d -> !decomp_f.active[d] ? nothing :
        (ivd !== nothing && iface[d] ?
         mkf(deriv, d; lo_closures=at(d, 1, ivd), hi_closures=at(d, 2, ivd)) :
         dplans_f[d]), 3)
    fplans_f = ntuple(d -> decomp_f.active[d] ?
        mkf(filt, d; lo_closures=at(d, 1, icf), hi_closures=at(d, 2, icf)) :
        nothing, 3)
    # The sensor smoother's input is built per patch and its coarse-fine
    # ghosts are never filled, so its plans keep the standard closures even
    # under `smoother = :compact`, as the same-level patch path does below.
    # Aliasing `fplans_f` here would read four ghost layers of allocation
    # zeros at every coarse-fine face through the C8 interior rows the
    # interface closures leave in place.
    #
    # A face on the domain boundary that reflects (`wall_faces`) takes the
    # node-centred wall rows of the `:gaussian` smoother, as the root's does
    # (`_sensor_wall_rows`); every other face keeps the scheme's rows.
    sw(d, side) = _sensor_wall_rows(smoo, wall_faces[d][side], 1, gaussian)
    # A tiled level's artificial-property pass fills the ghost layers of the
    # smoother's input at every interface face (`_level_artificial!`): the
    # neighbor's sensor at a shared face, the half-offset mirror at a
    # coarse-fine one. Its plans read them through the smoother's interface
    # rows, none for the `:gaussian` stencil and the identity row of the
    # compact filter, and keep the rows above at a face on the domain
    # boundary. Every other caller takes the plans above (`InterfaceSmoothPlans`).
    isr = interface_closures(smoo)
    swg(d, side) = boundary[d][side] ? sw(d, side) : isr
    function splan(d)
        decomp_f.active[d] || return nothing
        closed = mkf(smoo, d; lo_closures=sw(d, 1), hi_closures=sw(d, 2))
        iface[d] || return InterfaceSmoothPlans(closed, closed)
        return InterfaceSmoothPlans(mkf(smoo, d; lo_closures=swg(d, 1),
                                        hi_closures=swg(d, 2)), closed)
    end
    splans_f = ntuple(splan, 3)
    # The d8 detector reads the interface ghosts of a field recovered over the
    # padded extent through rows of its own (`_ring_interface_rows`), and
    # closes on the scheme's own rows for a field without them. `nothing`
    # where no sensor takes `:d8`, as on the root, so `detect_sum!`
    # dispatches alike. A reflecting face on the domain boundary takes the
    # wall rows of the field's sign there in both plans, one pair per sign
    # where the dimension has such a face.
    rplans_f = nothing
    if ring
        d8 = compact_d8(T)
        rrows = _ring_interface_rows(T)
        wr(d, side, σw, rows) = boundary[d][side] ?
            _sensor_wall_rows(d8, wall_faces[d][side], σw, true) : rows
        function rplans(d)
            decomp_f.active[d] || return nothing
            plan(σw, rows) = mkf(d8, d; lo_closures=wr(d, 1, σw, rows),
                                 hi_closures=wr(d, 2, σw, rows))
            ghost = plan(1, rrows)
            closed = plan(1, nothing)
            any(wall_faces[d]) || return InterfaceRingPlans((ghost, ghost),
                                                            (closed, closed))
            return InterfaceRingPlans((ghost, plan(-1, rrows)),
                                      (closed, plan(-1, nothing)))
        end
        rplans_f = ntuple(rplans, 3)
    end
    any(any, folded) || return (deriv=dplans_f, div=vplans_f, filter=fplans_f,
                                smooth=splans_f, ring=rplans_f,
                                folds=(nothing, nothing, nothing))
    # A dimension with a folded face routes every operator through its fold,
    # as the root's does, and holds no plan of its own. The fold's plans fold
    # the folded end onto the diagonal per ghost parity and close the other
    # end as above, with its interface, wall or scheme rows, so the divergence
    # keeps a pair of its own where that end is an interface (`nothing`
    # elsewhere, as on the root). The signs are the root fold's on the same
    # dimension.
    root_folds === nothing &&
        error("a refined patch on a fold needs the root's folds")
    fp(σ, d) = (lo_fold=folded[d][1] ? σ : nothing, hi_fold=folded[d][2] ? σ : nothing)
    pairof(sch, d, lo, hi) = (mkf(sch, d; fp(1, d)..., lo_closures=lo, hi_closures=hi),
                              mkf(sch, d; fp(-1, d)..., lo_closures=lo, hi_closures=hi))
    function foldspec(d)
        (decomp_f.active[d] && any(folded[d])) || return nothing
        rf = root_folds[d]
        dp = pairof(deriv, d, at(d, 1, icd), at(d, 2, icd))
        vp = ivd !== nothing && iface[d] ? pairof(deriv, d, at(d, 1, ivd), at(d, 2, ivd)) :
             nothing
        fq = pairof(filt, d, at(d, 1, icf), at(d, 2, icf))
        sw_f(side) = folded[d][side] ? nothing : sw(d, side)
        swg_f(side) = folded[d][side] ? nothing : swg(d, side)
        sp = FoldRingPlans(pairof(smoo, d, swg_f(1), swg_f(2)),
                           pairof(smoo, d, sw_f(1), sw_f(2)))
        rp = ((nothing, nothing), (nothing, nothing))
        if ring
            d8 = compact_d8(T)
            rrows = _ring_interface_rows(T)
            wrf(side, σw, rows) = folded[d][side] ? nothing :
                boundary[d][side] ? _sensor_wall_rows(d8, wall_faces[d][side], σw, true) :
                rows
            rfold(σg, σw, rows) = mkf(d8, d; fp(σg, d)..., lo_closures=wrf(1, σw, rows),
                                      hi_closures=wrf(2, σw, rows))
            function rpair(σg, rows)
                even = rfold(σg, 1, rows)
                any(wall_faces[d]) ? (even, rfold(σg, -1, rows)) : (even, even)
            end
            rp = FoldRingPlans((rpair(1, rrows), rpair(-1, rrows)),
                               (rpair(1, nothing), rpair(-1, nothing)))
        end
        return FoldSpec(d, folded[d][1], folded[d][2], nothing, rf.sigvel, rf.sigflux,
                        dp, fq, sp, rp, vp)
    end
    folds = ntuple(foldspec, 3)
    keep(plans) = ntuple(d -> folds[d] === nothing ? plans[d] : nothing, 3)
    return (deriv=keep(dplans_f), div=keep(vplans_f), filter=keep(fplans_f),
            smooth=keep(splans_f), ring=rplans_f === nothing ? nothing : keep(rplans_f),
            folds=folds)
end

# The persistent arrays of a patch from an allocator `g()`, by name, in the
# order the `Patch` constructor takes them.
function _patch_arrays(g::F, n_species::Int, n_sensed::Int=0) where {F}
    Y = [g() for _ in 1:n_species]
    return (rho=g(), u=g(), v=g(), w=g(), p=g(), T_ion=g(), c=g(), cp_mix=g(),
            Y=Y, mu_art=g(), beta_art=g(), kappa_art=g(),
            D_art=[g() for _ in 1:n_species], inv_J=g(), area_d=(g(), g(), g()),
            inv_h=(g(), g(), g()), inv_r=g(), cot_over_r=g(), cot_over_r_gcl=g(),
            overwritten=g(),
            sensed_fields=append!(empty(Y), (g() for _ in 1:n_sensed)))
end

# The refined `Patch` from its parts, with no pair buffers (`empty` stands in
# for both): a refined patch's folds, those of a symmetry plane or the r-z
# axis, are self-paired.
_assemble_patch(id::Int, level::Int, region, comm, decomp, hf, faces, bcs, plans,
                empty, a, ws, covered, scratch, gflux) =
    Patch(id, level, region, comm, decomp, hf, faces, bcs,
          plans.folds, plans.deriv, plans.div, plans.filter,
          plans.smooth, plans.ring, empty, empty,
          a.rho, a.u, a.v, a.w, a.p, a.T_ion, a.c, a.cp_mix, a.Y,
          a.mu_art, a.beta_art, a.kappa_art, a.D_art,
          a.inv_J, a.area_d, a.inv_h, a.inv_r, a.cot_over_r, a.cot_over_r_gcl,
          ws, covered, a.overwritten, scratch, gflux, a.sensed_fields)

# --- Stacked tiles of a device level ------------------------------------------
#
# On a device backend a tiled level's tiles of equal padded extent share one
# allocation per field, the tiles' blocks laid along the third padded
# dimension at a fixed stride (`StackedArray`, pointwise.jl), so that the
# right-hand side, the stage update and the filter launch once per level and
# the compact solves batch every tile's lines behind one interface fence
# (reference/AMR_GPU.md, launch policy). A spanning `Patch` holds the stacked
# arrays, the first tile's decomposition and communicator (every tile of the
# stack has the same extent, process grid and rank range, which the
# construction checks) and the batched device plans; each tile's `Patch`
# holds views of the same arrays and its own unbatched plans, so a tile
# evaluated alone (a diagnostic's gradients) reads and writes the slots the
# batched evaluation does. The stacked storage is rebuilt with the tiles at
# every regrid.

"Whether a tiled level's tiles take stacked storage on `backend`."
_stacked_level(backend::AbstractBackend, tile::Int) =
    backend isa DeviceBackend && tile > 0

# Build this rank's tiles `held` (indices into `tregions`) of one level, ids
# `id0 + 1, id0 + 2, ...` in `held` order: through `_build_fine_patch` on the
# host backend, and as stacked storage on a device backend, one stack per
# padded extent. Returns the patches in `held` order and the stacks, whose
# members are the patches' solver indices.
function _build_level_patches(::Type{T}, tregions::Vector{BlockRegion},
                              held::Vector{Int}, faces, active_g::NTuple{3,Bool},
                              h::NTuple{3,T}, n_halo::Int, comm::MPI.Comm,
                              deriv, filt, smoo, smoother::Symbol,
                              interface_rhs::Symbol, backend::AbstractBackend,
                              ws_pool::AbstractVector, n_species::Int, n_cons::Int,
                              grad_Q_columns::Int, id0::Int, level::Int, tile::Int;
                              interface_divergence=nothing,
                              ghost_viscous::Bool=false,
                              ring::Bool=false,
                              boundaries=fill(_NO_BOUNDARY, length(tregions)),
                              bcs=nothing, root_folds=nothing,
                              n_sensed::Int=0) where {T}
    patches = Patch[]
    stacks = TileStack[]
    if !_stacked_level(backend, tile)
        for (k, ti) in enumerate(held)
            push!(patches, _build_fine_patch(T, tregions[ti], active_g, h, n_halo,
                                             comm, deriv, filt, smoo, smoother,
                                             interface_rhs, backend, ws_pool,
                                             n_species, n_cons, grad_Q_columns, id0 + k,
                                             level, faces[ti]; interface_divergence,
                                             ghost_viscous, ring,
                                             boundary=boundaries[ti], bcs,
                                             root_folds, n_sensed, slot=k))
        end
        return patches, stacks
    end
    resize!(patches, length(held))
    # One stack per padded extent and set of boundary faces, the tiles of each
    # in `held` order: the batched plans close every member's lines alike.
    groups = [(fine_extent(tregions[ti], active_g), boundaries[ti]) for ti in held]
    for key in unique(groups)
        ks = [k for k in eachindex(held) if groups[k] == key]
        members = [id0 + k for k in ks]
        span, tiles = _build_tile_stack(T, [tregions[held[k]] for k in ks],
                                        [faces[held[k]] for k in ks], members,
                                        active_g, h, n_halo, comm, deriv, filt, smoo,
                                        interface_rhs, backend, n_species, n_cons,
                                        grad_Q_columns, level; interface_divergence,
                                        ghost_viscous, ring, boundary=key[2], bcs,
                                        smoother, root_folds, n_sensed)
        for (slot, k) in enumerate(ks)
            patches[k] = tiles[slot]
        end
        push!(stacks, TileStack(span, members))
    end
    return patches, stacks
end

# One stack: the spanning patch and its member tiles, in slot order.
function _build_tile_stack(::Type{T}, tregions::Vector{BlockRegion}, faces,
                           ids::Vector{Int}, active_g::NTuple{3,Bool},
                           h::NTuple{3,T}, n_halo::Int, comm::MPI.Comm,
                           deriv, filt, smoo, interface_rhs::Symbol,
                           backend::DeviceBackend, n_species::Int, n_cons::Int,
                           grad_Q_columns::Int, level::Int;
                           interface_divergence=nothing,
                           ghost_viscous::Bool=false,
                           ring::Bool=false,
                           boundary::NTuple{3,NTuple{2,Bool}}=_NO_BOUNDARY,
                           bcs=nothing, smoother::Symbol=:gaussian,
                           root_folds=nothing, n_sensed::Int=0) where {T}
    ntiles = length(tregions)
    # The members share their boundary faces and so fold alike: at a symmetry
    # plane or the r-z axis the spanning patch holds the fold with batched
    # plans, and each member holds it with its own, as a host tile does.
    folded = bcs === nothing ? _NO_BOUNDARY : _level_fold_faces(boundary, bcs)
    region1, decomp1, hf = _fine_decomp(T, tregions[1], active_g, h, n_halo, comm,
                                        folded)
    npad = padded_extent(decomp1)
    stride = npad[3]
    wall_faces = _sensor_wall_faces(_fine_bcs(active_g, faces[1], boundary, bcs))
    span_plans = _fine_plans(decomp1, hf, deriv, filt, smoo, interface_rhs, backend;
                             ntiles, stride, interface_divergence, ring, boundary,
                             wall_faces, gaussian=smoother === :gaussian, folded,
                             root_folds)
    empty_raw = empty_field(backend, T)
    stacked() = StackedArray(KernelAbstractions.zeros(backend.ka, T, npad[1], npad[2],
                                                      ntiles * stride),
                             ntiles, stride)
    empty_s = StackedArray(empty_raw, ntiles, stride)
    arrays = _patch_arrays(stacked, n_species, n_sensed)
    ws_span = _rhs_workspace(stacked, empty_s, n_species, n_cons,
                             ring, grad_Q_columns; active=decomp1.active)
    empty4 = similar(empty_raw, T, 0, 0, 0, 0)
    gflux_span = _ghost_flux_arrays(
        () -> StackedArray(KernelAbstractions.zeros(backend.ka, T, npad[1], npad[2],
                                                    ntiles * stride, n_cons),
                           ntiles, stride),
        StackedArray(empty4, ntiles, stride),
        ghost_viscous || GHOST_FLUX_REMAINDER[] ? _interface_dims(active_g, boundary) :
                                                  (false, false, false))
    # The spanning patch: id 0 (it is not in `solver.patches`), the first
    # tile's region, faces and boundary conditions (the batched phases read
    # only their kind: the members share the boundary faces, every other
    # face closes with the interface rows, and the markers enforce nothing),
    # no covered mask, no scratch.
    span = _assemble_patch(0, level, region1, comm, decomp1, hf, faces[1],
                           _fine_bcs(active_g, faces[1], boundary, bcs),
                           span_plans, empty_s,
                           arrays, ws_span, zeros(UInt8, 0, 0, 0),
                           _empty_level_scratch(empty_raw), gflux_span)
    tiles = Patch[]
    for (slot, refine) in enumerate(tregions)
        region_t, decomp_t, _ = slot == 1 ? (region1, decomp1, hf) :
                                _fine_decomp(T, refine, active_g, h, n_halo, comm,
                                             folded)
        # Every member sits in the span's slots the way the first tile does:
        # same extent, same process grid, same block of it on this rank.
        (padded_extent(decomp_t) == npad && decomp_t.dims == decomp1.dims &&
         decomp_t.coords == decomp1.coords && decomp_t.offset == decomp1.offset) ||
            error("tile $refine cannot join the stack of $(tregions[1]): the " *
                  "two tiles decompose differently on this rank")
        kr = ((slot - 1) * stride + 1):(slot * stride)
        v(a) = view(a, :, :, kr)
        tile_arrays = (rho=v(arrays.rho), u=v(arrays.u), v=v(arrays.v),
                       w=v(arrays.w), p=v(arrays.p), T_ion=v(arrays.T_ion),
                       c=v(arrays.c), cp_mix=v(arrays.cp_mix), Y=map(v, arrays.Y),
                       mu_art=v(arrays.mu_art), beta_art=v(arrays.beta_art),
                       kappa_art=v(arrays.kappa_art), D_art=map(v, arrays.D_art),
                       inv_J=v(arrays.inv_J), area_d=map(v, arrays.area_d),
                       inv_h=map(v, arrays.inv_h), inv_r=v(arrays.inv_r),
                       cot_over_r=v(arrays.cot_over_r),
                       cot_over_r_gcl=v(arrays.cot_over_r_gcl),
                       overwritten=v(arrays.overwritten),
                       # Typed as `Y`'s views also when there is none.
                       sensed_fields=append!(empty(map(v, arrays.Y)),
                                             (v(a) for a in arrays.sensed_fields)))
        plans_t = _fine_plans(decomp_t, hf, deriv, filt, smoo, interface_rhs, backend;
                              interface_divergence, ring, boundary, wall_faces,
                              gaussian=smoother === :gaussian, folded, root_folds)
        scratch = _level_scratch(empty_raw, refine, active_g, n_halo, n_cons,
                                 MPI.Comm_size(comm), MPI.Comm_rank(comm);
                                 gradient_deriv=ghost_viscous ? deriv : nothing,
                                 backend, fine_decomp=decomp_t, hf, boundary, folded)
        push!(tiles, _assemble_patch(ids[slot], level, region_t, comm, decomp_t, hf,
                                     faces[slot],
                                     _fine_bcs(active_g, faces[slot], boundary, bcs),
                                     plans_t, view(empty_raw, :, :, 1:0),
                                     tile_arrays, _view_workspace(ws_span, kr),
                                     _covered_mask(decomp_t), scratch,
                                     map(a -> size(a, 3) == 0 ?
                                              view(parent(a), :, :, 1:0, :) :
                                              view(parent(a), :, :, kr, :),
                                         gflux_span)))
    end
    return span, tiles
end

# A refined patch's face conditions: the root's condition `bcs[d][side]` at a
# face on the domain boundary, the same object; an interface marker elsewhere.
_fine_bcs(active_g::NTuple{3,Bool}, faces::NTuple{3,NTuple{2,Int}},
          boundary::NTuple{3,NTuple{2,Bool}}=_NO_BOUNDARY, bcs=nothing) =
    ntuple(d -> !active_g[d] ? (PeriodicBC(), PeriodicBC()) :
                ntuple(side -> _fine_bc(faces[d][side], boundary[d][side],
                                        bcs === nothing ? nothing : bcs[d][side]), 2), 3)

_fine_bc(face::Int, on_boundary::Bool, bc) =
    on_boundary ? bc : face == 0 ? CoarseFineBC() : InterfaceBC(face)

# A patch with new id, faces and boundary conditions sharing every array and
# plan of `p`: what a regrid hands a surviving tile whose neighbors changed.
# `comm` is the tile group's new communicator over the same ranks: the regrid
# frees the group `p.comm` belongs to, and `MPI.free` nulls that handle.
function _repatch(p::Patch, id::Int, faces::NTuple{3,NTuple{2,Int}}, bcs,
                  comm::MPI.Comm)
    # `field_tuples` is derived, and the junction captures are rebuilt for
    # the new layout, their deferral clear.
    names = filter(f -> f ∉ (:field_tuples, :reflux_captures, :reflux_deferred),
                   fieldnames(typeof(p)))
    args = map(names) do f
        f === :id ? id : f === :faces ? faces : f === :bcs ? bcs :
        f === :comm ? comm : getfield(p, f)
    end
    return Patch(args...)
end

# Multi-patch construction: the rank set is partitioned over the patch slabs,
# each patch builds its own decomposition, plans and arrays over its own
# communicator, and the interface exchange records are derived from one
# world Allgather. Folds, a banded filter, the :d8 detector on every sensor,
# and an explicit process grid are rejected by the caller before this runs.
function _build_patched_solver(::Type{T}, n_global, periodic, regions, faces_all,
                               patch_grid, bcs, eos, equations, transport, art,
                               metric, stretch, sources, origin, Lt, coord_shift,
                               h, deriv, filt, smoo, cfl, filter_interval,
                               filter_cfl, filter_weighting, control, n_halo,
                               comm, backend,
                               interface_rhs, n_cons, n_species;
                               interface_divergence=nothing,
                               interface_flux::Symbol=:closure,
                               schemes::SchemeSettings) where {T}
    MPI.Initialized() || MPI.Init(threadlevel=:funneled)
    world = comm
    np = MPI.Comm_size(world)
    npatch = length(regions)
    if np == 1
        my_pids = collect(1:npatch)
        pcomm = world
    else
        counts = patch_rank_counts(regions, np)
        myrank = MPI.Comm_rank(world)
        color = searchsortedfirst(cumsum(counts), myrank + 1)
        pcomm = MPI.Comm_split(world, color, myrank)
        my_pids = [color]
    end
    ext = interface_rhs === :extended
    icd = ext ? interface_closures(deriv) : nothing
    icf = ext ? interface_closures(filt) : nothing
    ivd = ext || interface_divergence !== nothing ?
          interface_divergence_rows(deriv, interface_divergence) : nothing
    nofold = (nothing, nothing, nothing)
    ring = compact_d8(T)
    ring_det = _ring_detector(art, n_species)
    # One RHS scratch pool for the rank's patches. A partitioned run gives each
    # rank one patch; a serial one holds every slab, and equal-extent slabs
    # then share a single set (patches.jl).
    ws_pool = rhs_workspace_pool(backend, T)
    patches = map(my_pids) do pid
        region = regions[pid]
        faces = faces_all[pid]
        pper = ntuple(d -> patch_grid[d] > 1 ? false : periodic[d], 3)
        dcp = Decomp{T}(region.extent, pper; n_halo=n_halo, comm=pcomm)
        pbcs = ntuple(d -> (faces[d][1] == 0 ? bcs[d][1] : InterfaceBC(faces[d][1]),
                            faces[d][2] == 0 ? bcs[d][2] : InterfaceBC(faces[d][2])), 3)
        mk(sch, d; kw...) =
            backend_plan(backend, plan_direction(dcp, sch, d, h[d]; kw...))
        locl(rows, d) = faces[d][1] == 0 ? nothing : rows
        hicl(rows, d) = faces[d][2] == 0 ? nothing : rows
        dplans = ntuple(d -> dcp.active[d] ?
            mk(deriv, d; lo_closures=locl(icd, d), hi_closures=hicl(icd, d)) :
            nothing, 3)
        # The flux divergence keeps one-sided closures at an interface (ghost
        # fluxes are unavailable; see patches.jl), the scheme's own rows or
        # the cascade's for the neutral set (`interface_divergence_rows`), or
        # a source scheme's rows under either `interface_rhs`, so it takes
        # separate plans wherever an interface end takes rows of its own; a
        # physical end keeps the scheme's rows on both.
        vplans = ntuple(d -> !dcp.active[d] ? nothing :
            (ivd !== nothing && (faces[d][1] != 0 || faces[d][2] != 0) ?
             mk(deriv, d; lo_closures=locl(ivd, d), hi_closures=hicl(ivd, d)) :
             dplans[d]), 3)
        fplans = ntuple(d -> dcp.active[d] ?
            mk(filt, d; lo_closures=locl(icf, d), hi_closures=hicl(icf, d)) :
            nothing, 3)
        # The sensor smoother's input is built per patch, so its interface
        # ghosts carry no data and its plans keep the standard closures even
        # under `smoother = :compact`. A face carrying a physical wall takes
        # the node-centred rows, as the single-patch path does.
        wface = _sensor_wall_faces(pbcs)
        swp(d, side) = _sensor_wall_rows(smoo, wface[d][side], 1,
                                         art.smoother === :gaussian)
        splans = ntuple(d -> dcp.active[d] ?
            mk(smoo, d; lo_closures=swp(d, 1), hi_closures=swp(d, 2)) : nothing, 3)
        # Under `:species_d8` the species sensors take the `:d8` detector (a
        # patched run is rejected under `:d8` on every sensor). It reads the
        # interface ghosts of the mass and mole fractions through rows of its
        # own (`_ring_interface_rows`), as a refined patch does, and closes a
        # wall face on the even node-centred rows. No other field reaches
        # these plans (`detect_sum!`), and those fractions carry valid ghosts
        # and are even at a wall, so one plan serves both wall signs.
        rwp(d, side) = faces[d][side] != 0 ? _ring_interface_rows(T) :
                       _sensor_wall_rows(ring, wface[d][side], 1, true)
        function rpair(d)
            dcp.active[d] || return nothing
            plan = mk(ring, d; lo_closures=rwp(d, 1), hi_closures=rwp(d, 2))
            return (plan, plan)
        end
        rplans = ring_det ? ntuple(rpair, 3) : nothing
        g() = field(backend, dcp)
        empty3 = empty_field(backend, T)
        # The `:d8` ringing buffer exists where the species plans above do;
        # the bulk channel's conserved gradients go through the same
        # interface plans as `grad_Y`.
        ws = rhs_workspace!(ws_pool, backend, dcp, n_species, n_cons, ring_det,
                            _grad_Q_columns(art, n_species, n_cons))
        Patch(pid, 0, region, pcomm, dcp, h, faces, pbcs, nofold,
              dplans, vplans, fplans, splans, rplans,
              empty3, empty3,
              g(), g(), g(), g(), g(), g(), g(), g(),
              [g() for _ in 1:n_species],
              g(), g(), g(),
              [g() for _ in 1:n_species],
              g(), (g(), g(), g()), (g(), g(), g()), g(), g(), g(),
              ws, _covered_mask(dcp), empty3, _empty_level_scratch(empty3),
              _ghost_flux_arrays(() -> parent(allocate_state(backend, dcp, n_cons)),
                                 similar(empty3, T, 0, 0, 0, 0),
                                 ntuple(d -> (_ghost_viscous(interface_flux, transport) ||
                                              _ghost_remainder(interface_flux, art)) &&
                                             dcp.active[d] &&
                                             (pbcs[d][1] isa InterfaceBC ||
                                              pbcs[d][2] isa InterfaceBC), 3)))
    end
    ghost_sends, ghost_recvs, plane_pairs = build_interface_records(
        T, world, regions, faces_all, my_pids, [p.decomp for p in patches], n_cons)
    orig = ntuple(d -> stretch[d] === nothing ? T(origin[d]) : zero(T), 3)
    solver = Solver{T,typeof(equations),typeof(eos),typeof(transport),typeof(metric),
                    typeof(stretch),typeof(sources),eltype(patches)}(
                  equations, eos, transport, art, metric, stretch, sources,
                  Lt, orig, coord_shift, h,
                  T(cfl), filter_interval, T(filter_cfl), filter_weighting, control,
                  n_global, patches, regions, world,
                  ghost_sends, ghost_recvs, plane_pairs,
                  [Level{T}(0, root_level_comm(world),
                            collect(eachindex(patches)), LevelTransfer{T}[])],
                  false, nothing,
                  zero(T), zero(T), 0, zero(T), zero(T),
                  ntuple(_ -> zero(T), 3), 0.0, 0.0, 0.0, 0.0, FloorTally(),
                  interface_flux, schemes, ModeTruncation{T}(), nothing, nothing,
                  nothing)
    for p in getfield(solver, :patches)
        init_geometry!(PatchSolver(solver, p))
    end
    return solver
end
