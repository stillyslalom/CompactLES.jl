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
# per dimension and side. `planned_sensor_mirror` (boundary.jl) is the
# setup-time form of the detector's `sensor_mirror` hook.
_sensor_wall_faces(bcs) = ntuple(d -> (planned_sensor_mirror(bcs[d][1]),
                                       planned_sensor_mirror(bcs[d][2])), 3)

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
[`Problem`](@ref); `art`, `deriv`, `filt`, `cfl`, `control`, `filter_interval`,
`filter_cfl`, `filter_weighting`, `dims`, `n_halo`, `stretch`, and `precision`
from [`Numerics`](@ref). Without `precision`, the element type is the one
shared by the components passed explicitly (`eos`, `transport`, `art`,
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
"""
function Solver(; n_global::NTuple{3,Int}, L_domain, bcs,
                precision::Union{Nothing,Type}=nothing,
                eos=nothing,
                transport::Union{Nothing,AbstractTransport}=nothing,
                art::Union{Nothing,ArtParams}=nothing,
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
                   transport=cv(transport, Transport{T}()),
                   art=cv(art, ArtParams{T}()),
                   deriv=cv(deriv, lele_d1_6(T)),
                   filt=cv(filt, compact_filter(0.45, T)),
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
    validate_art(art)
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
    art.detector in (:delta4, :d8) ||
        error("art.detector must be :delta4 or :d8, got :$(art.detector)")
    art.species_flux in (:fickian, :bulk, :partial_density) ||
        error("art.species_flux must be :fickian, :bulk or :partial_density, " *
              "got :$(art.species_flux)")
    # Per-condition restrictions on geometry and EOS agreement (boundary.jl).
    for d in 1:3, side in 1:2
        validate_bc(bcs[d][side], metric, eos, d, side)
    end
    return nothing
end

function _Solver(::Type{T}; n_global::NTuple{3,Int}, L_domain, bcs,
                eos::EOS,
                equations=nothing,
                transport::AbstractTransport{T},
                art::ArtParams{T},
                metric::Metric=CartesianMetric(),
                stretch::NTuple{3,Union{Nothing,Stretch}}=(nothing, nothing, nothing),
                sources=(),
                origin=(0.0, 0.0, 0.0),
                deriv::AbstractCompactScheme,
                filt::AbstractCompactScheme,
                cfl::Real=0.5, filter_interval::Int=1, filter_cfl::Real=0.35,
                filter_weighting::Symbol=:none,
                control::StepControl=StepControl(),
                dims=nothing, n_halo::Int=4,
                comm::MPI.Comm=MPI.COMM_WORLD,
                patch_grid::NTuple{3,Int}=(1, 1, 1),
                backend::AbstractBackend=CPUBackend(),
                interface_rhs::Symbol=:extended,
                interface_divergence::Union{Nothing,AbstractCompactScheme},
                interface_flux::Symbol=:closure,
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
                rebalance_persist::Int=2) where {T}
    bcs = _face_conditions(bcs)
    level_interpolation_order =
        something(level_interpolation_order,
                  default_interpolation_order(deriv, interface_flux))
    schemes = SchemeSettings(deriv, filt, interface_divergence, interface_rhs,
                             level_interpolation_order, level_restriction)
    _validate_configuration(transport, eos, art, bcs, metric, n_global, L_domain,
                            origin, cfl, filter_interval, filter_cfl)
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
        art.detector === :delta4 ||
            error("patch interfaces support the :delta4 detector only; the " *
                  ":d8 detector's banded scheme takes a single patch")
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
    if isempty(refines)
        subcycle &&
            error("subcycle requires a refined region (the refine keyword)")
        regrid_interval == 0 ||
            error("regrid_interval requires a refined region (the refine " *
                  "keyword supplies the initial one)")
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
    regrid_interval == 0 || length(refines) <= 1 ||
        error("regridding is implemented for a two-level hierarchy; " *
              "$(length(refines)) refined levels were given")
    if !isempty(refines)
        MPI.Initialized() || MPI.Init(threadlevel=:funneled)
        MPI.Comm_size(comm) == 1 || level_restriction === :inject ||
            error("level_restriction = :filter restricts through a " *
                  "whole-patch line solve and is serial-only; use :inject " *
                  "under MPI")
        npatch == 1 ||
            error("refine cannot combine with a same-level patch_grid yet")
        metric isa CartesianMetric ||
            error("refinement requires CartesianMetric in this stage")
        all(isnothing, stretch) ||
            error("refinement requires an unstretched grid")
        (axis || orig1 || poles) &&
            error("refinement across a coordinate fold is forbidden")
        any(any, symplane) &&
            error("refinement across a SymmetryPlaneBC is forbidden; a " *
                  "refined run takes SlipWallBC at that face")
        filt isa CompactScheme ||
            error("the coarse-fine boundary carries closure variants for a " *
                  "tridiagonal filter only")
        art.detector === :delta4 ||
            error("refinement supports the :delta4 detector only")
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
        # above it, in that level's node space (the root's is the grid).
        margin = max(n_halo, LEVEL_BUFFER)
        parent_regions = [BlockRegion((0, 0, 0), n_global)]
        for (ℓ, rg) in enumerate(refines)
            for d in 1:3
                if active_g[d]
                    rg.extent[d] >= 4 ||
                        error("level $ℓ region needs at least 4 parent nodes " *
                              "along dimension $d (9 fine points for the C8 " *
                              "filter)")
                else
                    rg.offset[d] == 0 && rg.extent[d] == 1 ||
                        error("level $ℓ region must span collapsed dimension " *
                              "$d with offset 0 and extent 1")
                end
            end
            if !_covered_by(_buffered(rg, active_g, margin), parent_regions)
                p = only(parent_regions)
                ranges = join(("offset $(p.offset[d] + margin):" *
                               "$(p.offset[d] + p.extent[d] - margin - rg.extent[d]) " *
                               "along dimension $d" for d in 1:3 if active_g[d]), ", ")
                error("level $ℓ region $rg must be nested at least $margin " *
                      "level-$(ℓ - 1) nodes inside the level-$(ℓ - 1) patches' " *
                      "own (not imposed) nodes: with its extent that is $ranges, " *
                      "counted on the level-$(ℓ - 1) lattice over the whole domain. " *
                      "AMR(initial = [shape, ...]) takes the levels as shapes in " *
                      "physical coordinates instead")
            end
            # The next level reads this one's own nodes: a one-patch level's
            # boundary planes are imposed data and are eroded (the tiled
            # cover is checked face by face at construction).
            parent_regions = [_erode(BlockRegion(
                ntuple(d -> active_g[d] ? 3 * rg.offset[d] : 0, 3),
                fine_extent(rg, active_g)), ntuple(d -> (true, true), 3), active_g)]
        end
    end
    # --- Interface divergence rows ------------------------------------------
    # The source scheme only supplies the divergence's closure rows at patch
    # and level interface ends, so a run without an interface would ignore it.
    if interface_divergence !== nothing
        npatch > 1 || !isempty(refines) ||
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
    # the ghost fluxes carry no area or Jacobian factors, so the geometry
    # must be unit.
    interface_flux in (:closure, :ghost) ||
        error("interface_flux must be :closure or :ghost, got :$interface_flux")
    if interface_flux === :ghost
        npatch > 1 || !isempty(refines) ||
            error("interface_flux = :ghost differences through a patch or level " *
                  "interface; this run has neither (patch_grid, refine)")
        interface_rhs === :extended ||
            error("interface_flux = :ghost reads the gradient plans' interface " *
                  "rows, which exist under interface_rhs = :extended only")
        metric isa CartesianMetric && all(isnothing, stretch) ||
            error("interface_flux = :ghost requires an unstretched CartesianMetric")
        # A coarse-fine face's molecular ghost flux recovers the temperature
        # gradient from the conserved ones through the internal energy.
        isempty(refines) || !_ghost_viscous(interface_flux, transport) ||
            _ghost_gradient_eos(eos) ||
            error("interface_flux = :ghost with molecular transport at a refined " *
                  "level supports IdealMixture, Nasa9Mixture and StiffenedGas; " *
                  "got $(typeof(eos))")
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
        eos isa Nasa9Mixture &&
            error("Nasa9Mixture has no device coefficient mirror yet; use " *
                  "IdealMixture or StiffenedGas on a DeviceBackend")
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
    # dimension and one pair per fold, matching the smoother.
    ring = compact_d8(T)
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
                                           art.detector !== :delta4)
    equations = equations === nothing ? NavierStokes1T(eos) : equations
    equations isa EquationSet || error("equations must be an EquationSet")
    equations.n_species == nspecies(eos) ||
        error("equation set carries $(equations.n_species) species; " *
              "EOS has $(nspecies(eos))")
    n_species = equations.n_species
    n_cons = equations.n_cons
    if npatch > 1
        return _build_patched_solver(T, n_global, periodic, regions, faces_all,
                                     patch_grid, bcs, eos, equations, transport,
                                     art, metric, stretch, sources, origin, Lt,
                                     coord_shift, h, deriv, filt, smoo, cfl,
                                     filter_interval, filter_cfl, filter_weighting,
                                     control, n_halo, comm, backend, interface_rhs,
                                     n_cons, n_species; interface_divergence,
                                     interface_flux, schemes)
    end
    decomp = Decomp{T}(n_global, periodic; dims=dims, n_halo=n_halo, comm=comm)
    # The per-rank extent check in `plan_direction` would raise on some ranks
    # only when the blocks differ in size; this one is replicated.
    check_block_extents(n_global, decomp.dims,
                        ntuple(d -> !periodic[d] && !fold_lo_dim[d], 3),
                        ntuple(d -> !periodic[d] && !fold_hi_dim[d], 3),
                        d -> (
        (deriv, nothing, nothing), (filt, nothing, nothing),
        (art.smoother === :gaussian ? ((smoo, swrow(d, 1), swrow(d, 2)),) : ())...,
        (art.detector === :delta4 ? () :
         ((ring, rwrow(d, 1, 1), rwrow(d, 2, 1)),))...))
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
        rp = art.detector === :delta4 ? ((nothing, nothing), (nothing, nothing)) :
             (rpair(1), rpair(-1))
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
    # field's type to decide which detector runs, so under `:delta4` the whole
    # d8 path (`ring_sum!`, `ring_along!`, and the `apply_along!` call taking a
    # possibly-absent plan) is not reachable from inference and costs the
    # default configuration nothing.
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
    ring_plans = art.detector === :delta4 ? nothing : ntuple(ringpair, 3)
    paired_fold = any(fold -> fold !== nothing && fold.pair !== nothing, folds)
    orig = ntuple(d -> stretch[d] === nothing ? T(origin[d]) : zero(T), 3)
    bcs_t = ntuple(d -> (bcs[d][1], bcs[d][2]), 3)
    # One RHS scratch pool per rank, seeded with the root patch's set and
    # handed to every refined patch below (patches.jl).
    ws_pool = rhs_workspace_pool(backend, T)
    ws_root = rhs_workspace!(ws_pool, backend, decomp, n_species, n_cons,
                             art.detector !== :delta4,
                             _shared_species_diffusivity(art, n_species))
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
                  _empty_level_scratch(empty_field(backend, T)),
                  ntuple(_ -> similar(empty_field(backend, T), T, 0, 0, 0, 0), 3))
    if isempty(refines)
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
                  interface_flux, schemes)
        init_geometry!(solver)
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
    for (ℓ, rg) in enumerate(refines)
        if tile == 0
            tregions = [rg]
        else
            # Clip the lattice to the parent patches' bounding box less the
            # margin; a tile that then still leaves the union is refused.
            lo = ntuple(d -> minimum(r.offset[d] for r in parent_regions) +
                             1 + margin, 3)
            hi = ntuple(d -> maximum(r.offset[d] + r.extent[d]
                                     for r in parent_regions) - margin, 3)
            tregions = _level_tiles(rg, active_g, tile, lo, hi)
            isempty(tregions) &&
                error("level $ℓ region admits no tile of edge $tile inside " *
                      "the nesting margin")
        end
        faces = _tile_faces(tregions)
        for tr in tregions
            _covered_by(_buffered(tr, active_g, margin), parent_valid) ||
                error("level $ℓ tile $tr must be nested at least $margin " *
                      "level-$(ℓ - 1) nodes inside the level-$(ℓ - 1) patches' " *
                      "own (not imposed) nodes")
        end
        # The level's geometry is derived, not read off built patches, so a
        # rank outside the level's subset carries the same node spaces into
        # the next level's nesting checks as its owners do.
        fine_regions = [BlockRegion(
            ntuple(d -> active_g[d] ? 3 * tr.offset[d] : 0, 3),
            fine_extent(tr, active_g)) for tr in tregions]
        imposed_all = [ntuple(d -> (f[d][1] == 0, f[d][2] == 0), 3)
                       for f in faces]
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
                                                 _shared_species_diffusivity(art, n_species),
                                                 id0, ℓ, tile; interface_divergence,
                                                 ghost_viscous=
                                                     _ghost_viscous(interface_flux,
                                                                    transport))
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
                                   parent_regions)
                push!(transfers, build_level_transfer(
                    T, tr, active_g, n_halo, parent_regions[pids],
                    parent_local[pids],
                    Union{Nothing,Decomp{T}}[decomp_at(parent_local[p])
                                             for p in pids],
                    local_of[ti], level_restriction, n_cons, subcycle,
                    decomp_at(local_of[ti]), parent_lc.comm,
                    length(owners[ti]), faces[ti];
                    interpolation_order=level_interpolation_order,
                    gradient_deriv=_ghost_viscous(interface_flux, transport) ?
                                   deriv : nothing,
                    parent_h=parent_h))
            end
            if lc.owned
                # A record's partner is a rank number in the communicator the
                # exchange runs over, here the level's own; that numbering
                # may differ from a tile's Cartesian communicator's.
                records = _level_records(T, lc.comm, fine_regions, held, indices,
                                         [fines[li - 1].decomp for li in indices],
                                         n_cons)
                push!(levels, Level{T}(ℓ, lc, owners, group, held, indices,
                                       transfers, records; stacks))
            else
                # The level's transfers are still held: their box gathers and
                # restriction write-back run on the parent's communicator.
                push!(levels, Level{T}(ℓ, lc, owners, group, held, indices,
                                       transfers; stacks))
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
                           Dict(lt.region => 0 for lt in levels[2].transfers),
                           interface_divergence,
                           level_interpolation_order, level_restriction)
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
                  interface_flux, schemes)
    for p in getfield(solver, :patches)
        init_geometry!(PatchSolver(solver, p))
    end
    # Each held patch's covered mask from the regions of the level below,
    # which every rank of the patch's level holds (`_fill_covered!`).
    for ℓ in 1:length(levels)-1
        child_regions = [lt.region for lt in levels[ℓ + 1].transfers]
        for li in levels[ℓ].patches
            _fill_covered!(patches[li], child_regions)
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
                           n_species::Int, n_cons::Int, bulk::Bool, id::Int,
                           level::Int,
                           faces::NTuple{3,NTuple{2,Int}}=ntuple(d -> (0, 0), 3);
                           interface_divergence=nothing,
                           ghost_viscous::Bool=false) where {T}
    region_f, decomp_f, hf = _fine_decomp(T, refine, active_g, h, n_halo, comm)
    plans = _fine_plans(decomp_f, hf, deriv, filt, smoo, interface_rhs, backend;
                        interface_divergence)
    g() = field(backend, decomp_f)
    empty3 = empty_field(backend, T)
    # Refinement takes the `:delta4` detector (rejected otherwise at setup), so
    # no refined patch carries the `:d8` ringing buffer; `bulk` selects the
    # conserved gradients of the shared-D_b species channels, which a refined
    # patch differences as the root does.
    ws = rhs_workspace!(ws_pool, backend, decomp_f, n_species, n_cons, false,
                        bulk)
    scratch = _level_scratch(empty3, refine, active_g, n_halo, n_cons,
                             MPI.Comm_size(comm), MPI.Comm_rank(comm))
    # Every face of a refined patch is an interface end, a coarse-fine or a
    # same-level one, so each active dimension takes a ghost-flux array.
    gflux = _ghost_flux_arrays(() -> parent(allocate_state(backend, decomp_f, n_cons)),
                               similar(empty3, T, 0, 0, 0, 0),
                               ghost_viscous ? active_g : (false, false, false))
    return _assemble_patch(id, level, region_f, comm, decomp_f, hf, faces,
                           _fine_bcs(active_g, faces), plans, empty3,
                           _patch_arrays(g, n_species), ws, _covered_mask(decomp_f),
                           scratch, gflux)
end

# A patch's `ghost_flux` arrays: `g4()` in each dimension of `dims`, the
# zero-extent `empty4` in the others.
_ghost_flux_arrays(g4::F, empty4, dims::NTuple{3,Bool}) where {F} =
    ntuple(d -> dims[d] ? g4() : empty4, 3)

# The refined region's node space, decomposition and spacing: parent-level
# node g is refined-level node 3(g − 1) + 1.
function _fine_decomp(::Type{T}, refine::BlockRegion, active_g::NTuple{3,Bool},
                      h::NTuple{3,T}, n_halo::Int, comm::MPI.Comm) where {T}
    hf = ntuple(d -> active_g[d] ? h[d] / 3 : h[d], 3)
    region_f = BlockRegion(ntuple(d -> active_g[d] ? 3 * refine.offset[d] : 0, 3),
                           fine_extent(refine, active_g))
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

# A refined patch's plans on `backend`: gradient, divergence, filter and
# smoother. `ntiles` and `stride` plan the batched device solve of a stacked
# level's spanning patch (lines_device.jl); the default is one patch's plans.
function _fine_plans(decomp_f::Decomp, hf, deriv, filt, smoo, interface_rhs::Symbol,
                     backend::AbstractBackend; ntiles::Int=1, stride::Int=0,
                     interface_divergence=nothing)
    mkf(sch, d; kw...) =
        backend_plan(backend, plan_direction(decomp_f, sch, d, hf[d]; kw...,
                                             lines_factor=ntiles); ntiles, stride)
    ext_f = interface_rhs === :extended
    icd = ext_f ? interface_closures(deriv) : nothing
    icf = ext_f ? interface_closures(filt) : nothing
    ivd = ext_f || interface_divergence !== nothing ?
          interface_divergence_rows(deriv, interface_divergence) : nothing
    # Every face of a refined patch closes with the interface rows and reads
    # ghosts; the boundary condition (`_fine_bcs`) only records where they
    # come from. The divergence takes the one-sided interface rows instead
    # (`interface_divergence_rows`), since a flux array has no ghosts. A
    # source scheme selects those rows under either `interface_rhs`; without
    # one, `:onesided` keeps the gradient plans' own rows for the divergence.
    dplans_f = ntuple(d -> decomp_f.active[d] ?
        mkf(deriv, d; lo_closures=icd, hi_closures=icd) : nothing, 3)
    vplans_f = ntuple(d -> !decomp_f.active[d] ? nothing :
        (ivd !== nothing ? mkf(deriv, d; lo_closures=ivd, hi_closures=ivd) :
         dplans_f[d]), 3)
    fplans_f = ntuple(d -> decomp_f.active[d] ?
        mkf(filt, d; lo_closures=icf, hi_closures=icf) : nothing, 3)
    # The sensor smoother's input is built per patch and its coarse-fine
    # ghosts are never filled, so its plans keep the standard closures even
    # under `smoother = :compact`, as the same-level patch path does below.
    # Aliasing `fplans_f` here would read four ghost layers of allocation
    # zeros at every coarse-fine face through the C8 interior rows the
    # interface closures leave in place.
    #
    # No wall rows either: every face of a refined patch is a coarse-fine or
    # interface end (`_fine_bcs`), so none of them reflects.
    splans_f = ntuple(d -> decomp_f.active[d] ? mkf(smoo, d) : nothing, 3)
    return (deriv=dplans_f, div=vplans_f, filter=fplans_f, smooth=splans_f)
end

# The persistent arrays of a patch from an allocator `g()`, by name, in the
# order the `Patch` constructor takes them.
_patch_arrays(g::F, n_species::Int) where {F} =
    (rho=g(), u=g(), v=g(), w=g(), p=g(), T_ion=g(), c=g(), cp_mix=g(),
     Y=[g() for _ in 1:n_species], mu_art=g(), beta_art=g(), kappa_art=g(),
     D_art=[g() for _ in 1:n_species], inv_J=g(), area_d=(g(), g(), g()),
     inv_h=(g(), g(), g()), inv_r=g(), cot_over_r=g(), cot_over_r_gcl=g())

# The refined `Patch` from its parts; fold-free, with no ring plans and no
# pair buffers (`empty` stands in for both).
_assemble_patch(id::Int, level::Int, region, comm, decomp, hf, faces, bcs, plans,
                empty, a, ws, covered, scratch, gflux) =
    Patch(id, level, region, comm, decomp, hf, faces, bcs,
          (nothing, nothing, nothing), plans.deriv, plans.div, plans.filter,
          plans.smooth, nothing, empty, empty,
          a.rho, a.u, a.v, a.w, a.p, a.T_ion, a.c, a.cp_mix, a.Y,
          a.mu_art, a.beta_art, a.kappa_art, a.D_art,
          a.inv_J, a.area_d, a.inv_h, a.inv_r, a.cot_over_r, a.cot_over_r_gcl,
          ws, covered, scratch, gflux)

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
                              bulk::Bool, id0::Int, level::Int, tile::Int;
                              interface_divergence=nothing,
                              ghost_viscous::Bool=false) where {T}
    patches = Patch[]
    stacks = TileStack[]
    if !_stacked_level(backend, tile)
        for (k, ti) in enumerate(held)
            push!(patches, _build_fine_patch(T, tregions[ti], active_g, h, n_halo,
                                             comm, deriv, filt, smoo, smoother,
                                             interface_rhs, backend, ws_pool,
                                             n_species, n_cons, bulk, id0 + k,
                                             level, faces[ti]; interface_divergence,
                                             ghost_viscous))
        end
        return patches, stacks
    end
    resize!(patches, length(held))
    # One stack per padded extent, the tiles of each in `held` order.
    extents = [fine_extent(tregions[ti], active_g) for ti in held]
    for ext in unique(extents)
        ks = [k for k in eachindex(held) if extents[k] == ext]
        members = [id0 + k for k in ks]
        span, tiles = _build_tile_stack(T, [tregions[held[k]] for k in ks],
                                        [faces[held[k]] for k in ks], members,
                                        active_g, h, n_halo, comm, deriv, filt, smoo,
                                        interface_rhs, backend, n_species, n_cons,
                                        bulk, level; interface_divergence,
                                        ghost_viscous)
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
                           bulk::Bool, level::Int; interface_divergence=nothing,
                           ghost_viscous::Bool=false) where {T}
    ntiles = length(tregions)
    region1, decomp1, hf = _fine_decomp(T, tregions[1], active_g, h, n_halo, comm)
    npad = padded_extent(decomp1)
    stride = npad[3]
    span_plans = _fine_plans(decomp1, hf, deriv, filt, smoo, interface_rhs, backend;
                             ntiles, stride, interface_divergence)
    empty_raw = empty_field(backend, T)
    stacked() = StackedArray(KernelAbstractions.zeros(backend.ka, T, npad[1], npad[2],
                                                      ntiles * stride),
                             ntiles, stride)
    empty_s = StackedArray(empty_raw, ntiles, stride)
    arrays = _patch_arrays(stacked, n_species)
    ws_span = _rhs_workspace(stacked, empty_s, n_species, n_cons, false, bulk)
    empty4 = similar(empty_raw, T, 0, 0, 0, 0)
    gflux_span = _ghost_flux_arrays(
        () -> StackedArray(KernelAbstractions.zeros(backend.ka, T, npad[1], npad[2],
                                                    ntiles * stride, n_cons),
                           ntiles, stride),
        StackedArray(empty4, ntiles, stride),
        ghost_viscous ? active_g : (false, false, false))
    # The spanning patch: id 0 (it is not in `solver.patches`), the first
    # tile's region, faces and boundary conditions (nothing in the batched
    # phases reads them: every refined face closes with the interface rows,
    # and the conditions enforce nothing), no covered mask, no scratch.
    span = _assemble_patch(0, level, region1, comm, decomp1, hf, faces[1],
                           _fine_bcs(active_g, faces[1]), span_plans, empty_s,
                           arrays, ws_span, zeros(UInt8, 0, 0, 0),
                           _empty_level_scratch(empty_raw), gflux_span)
    tiles = Patch[]
    for (slot, refine) in enumerate(tregions)
        region_t, decomp_t, _ = slot == 1 ? (region1, decomp1, hf) :
                                _fine_decomp(T, refine, active_g, h, n_halo, comm)
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
                       cot_over_r_gcl=v(arrays.cot_over_r_gcl))
        plans_t = _fine_plans(decomp_t, hf, deriv, filt, smoo, interface_rhs, backend;
                              interface_divergence)
        scratch = _level_scratch(empty_raw, refine, active_g, n_halo, n_cons,
                                 MPI.Comm_size(comm), MPI.Comm_rank(comm))
        push!(tiles, _assemble_patch(ids[slot], level, region_t, comm, decomp_t, hf,
                                     faces[slot], _fine_bcs(active_g, faces[slot]),
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

_fine_bcs(active_g::NTuple{3,Bool}, faces::NTuple{3,NTuple{2,Int}}) =
    ntuple(d -> !active_g[d] ? (PeriodicBC(), PeriodicBC()) :
                (faces[d][1] == 0 ? CoarseFineBC() : InterfaceBC(faces[d][1]),
                 faces[d][2] == 0 ? CoarseFineBC() : InterfaceBC(faces[d][2])), 3)

# A patch with new id, faces and boundary conditions sharing every array and
# plan of `p`: what a regrid hands a surviving tile whose neighbors changed.
function _repatch(p::Patch, id::Int, faces::NTuple{3,NTuple{2,Int}}, bcs)
    names = fieldnames(typeof(p))[1:end-1]     # field_tuples is derived
    args = map(names) do f
        f === :id ? id : f === :faces ? faces : f === :bcs ? bcs : getfield(p, f)
    end
    return Patch(args...)
end

# Multi-patch construction: the rank set is partitioned over the patch slabs,
# each patch builds its own decomposition, plans and arrays over its own
# communicator, and the interface exchange records are derived from one
# world Allgather. Folds, a banded filter, the :d8 detector, and an explicit
# process grid are rejected by the caller before this runs.
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
        # the node-centred rows, as the single-patch path does; a patched run
        # is rejected under `detector = :d8`, so no ring plans arise here.
        wface = _sensor_wall_faces(pbcs)
        swp(d, side) = _sensor_wall_rows(smoo, wface[d][side], 1,
                                         art.smoother === :gaussian)
        splans = ntuple(d -> dcp.active[d] ?
            mk(smoo, d; lo_closures=swp(d, 1), hi_closures=swp(d, 2)) : nothing, 3)
        g() = field(backend, dcp)
        empty3 = empty_field(backend, T)
        # A patched run takes the `:delta4` detector, rejected otherwise at
        # setup, so no patch carries the `:d8` ringing buffer; the bulk
        # channel's conserved gradients go through the same interface plans
        # as `grad_Y`.
        ws = rhs_workspace!(ws_pool, backend, dcp, n_species, n_cons, false,
                            _shared_species_diffusivity(art, n_species))
        Patch(pid, 0, region, pcomm, dcp, h, faces, pbcs, nofold,
              dplans, vplans, fplans, splans, nothing,
              empty3, empty3,
              g(), g(), g(), g(), g(), g(), g(), g(),
              [g() for _ in 1:n_species],
              g(), g(), g(),
              [g() for _ in 1:n_species],
              g(), (g(), g(), g()), (g(), g(), g()), g(), g(), g(),
              ws, _covered_mask(dcp), _empty_level_scratch(empty3),
              _ghost_flux_arrays(() -> parent(allocate_state(backend, dcp, n_cons)),
                                 similar(empty3, T, 0, 0, 0, 0),
                                 ntuple(d -> _ghost_viscous(interface_flux, transport) &&
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
                  interface_flux, schemes)
    for p in getfield(solver, :patches)
        init_geometry!(PatchSolver(solver, p))
    end
    return solver
end
