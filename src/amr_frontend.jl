"""
    AMR(; initial = :sensor, regrid_interval = nothing, tile = 0, ...)

Refinement configuration for [`Numerics`](@ref). `initial` selects what is
refined at setup:

- `:sensor`: the cells the tag criteria select on the initial state, followed
  as they move.
- a predicate `(x, y, z, t) -> Bool`: the nodes where it holds, re-evaluated at
  every regrid check, so a region can move on a prescribed path.
- a [`Shape`](@ref) or a predicate `(x, y, z) -> Bool`: a fixed region.
- a vector of nested shapes, one per level, finest last:
  `[Box((0.2, 0, 0), (0.6, 1, 1)), Sphere((0.4, 0.5, 0.5), 0.05)]` refines
  the box once and the sphere twice. Each shape is covered by the nodes of its
  level's parent.
- a [`BlockRegion`](@ref) or a vector of nested ones, in the node lattice of
  each level's parent, for a layout given exactly.

A refined level always lies at least `max(n_halo, 4)` of its parent's nodes
inside its parent (the root's boundaries included, periodic or not), which is
the room the coarse–fine transfer needs. A region asked for closer to a
boundary is reduced to fit, with a warning naming the level.

`regrid_interval` defaults to 0 (a fixed region) for a shape, a region or a
predicate of position alone, and for `:sensor` or a time-dependent predicate
to the number of steps a feature moving at the CFL limit takes to cross half
of `tag_buffer`, so a followed feature cannot leave its refined region between
checks. `tile = 0` covers the tagged cells with one box, the cheaper cover of a
single compact feature; a positive edge covers them with lattice tiles instead,
so that separated features (a shock and a distant interface) refine
separately rather than as one bounding box, at a per-tile cost that makes
small edges expensive in three dimensions. At setup, sensor and predicate
selection require at least one tagged node.

`tag_threshold`, the density criterion, defaults to `0.02` except under a
predicate, where it defaults to `Inf` so that the predicate alone selects the
refined region; give it explicitly to combine the two.
`tag_sensor_threshold` reads the artificial coefficients, so it requires
`ArtParams(enabled = true)`.

Regions use root node indices; the ratio between successive levels is three.
The other keywords match the established refinement controls of [`Numerics`](@ref).
The current regrid implementation permits one refined level. An explicit
vector of nested regions is static.
"""
Base.@kwdef struct AMR
    initial::Any = :sensor
    level_restriction::Symbol = :inject
    level_interpolation_order::Union{Nothing,Int} = nothing
    subcycle::Bool = false
    regrid_interval::Union{Nothing,Int} = nothing
    tag_threshold::Union{Nothing,Float64} = nothing
    tag_buffer::Int = 4
    tag_sensor_threshold::Float64 = 0.0
    tag_gradient_threshold::Float64 = 0.0
    tag_vorticity_threshold::Float64 = 0.0
    untag_ratio::Float64 = 2.0
    tile_lifetime::Int = 1
    tile::Int = 0
    rebalance::Float64 = 0.0
    rebalance_persist::Int = 2
end

# A predicate of position alone describes a fixed region; it is called with the
# time dropped.
struct _StaticPredicate{F}
    f::F
end
(p::_StaticPredicate)(x, y, z, t) = p.f(x, y, z)

_amr_predicate(f) = applicable(f, 0.0, 0.0, 0.0, 0.0) ? f :
                    applicable(f, 0.0, 0.0, 0.0) ? _StaticPredicate(f) : nothing

_amr_tag_threshold(amr::AMR) =
    amr.tag_threshold !== nothing ? amr.tag_threshold :
    _amr_callable(amr.initial) ? Inf : 0.02

function _amr_physical_tag(predicate::F) where {F}
    return function (patch, I)
        i, j, k = interior_index(patch, I)
        result = predicate(xcoord(patch, 1, i), xcoord(patch, 2, j),
                           xcoord(patch, 3, k), patch.t)
        result isa Bool || throw(ArgumentError("AMR initial predicate must return Bool"))
        return result
    end
end

function _amr_keywords(amr::AMR; refine=amr.initial, bootstrap::Bool=false)
    interval = bootstrap ? max(1, amr.regrid_interval) : amr.regrid_interval
    predicate = _amr_callable(amr.initial) ?
                _amr_physical_tag(_amr_predicate(amr.initial)) : nothing
    return (; refine, level_restriction=amr.level_restriction,
            level_interpolation_order=amr.level_interpolation_order,
            subcycle=amr.subcycle, regrid_interval=interval,
            tag_threshold=_amr_tag_threshold(amr), tag_buffer=amr.tag_buffer,
            tag_sensor_threshold=amr.tag_sensor_threshold,
            tag_gradient_threshold=amr.tag_gradient_threshold,
            tag_vorticity_threshold=amr.tag_vorticity_threshold,
            tag_predicate=predicate, untag_ratio=amr.untag_ratio,
            tile_lifetime=amr.tile_lifetime, tile=amr.tile,
            rebalance=amr.rebalance, rebalance_persist=amr.rebalance_persist)
end

_amr_callable(initial) = !(initial isa Symbol || initial isa BlockRegion ||
                           initial isa AbstractVector || initial isa Shape)

# Whether the selection follows something that moves: the sensor, or a
# predicate of time.
_amr_moving(initial) = initial === :sensor ||
    (_amr_callable(initial) && applicable(initial, 0.0, 0.0, 0.0, 0.0))

# The default that depends on the run: the regrid interval from the CFL (a
# feature moves at most about `cfl` root cells per root step, so it crosses
# half the buffer in `tag_buffer / (2 cfl)` steps).
function _resolve_amr(amr::AMR, num)
    interval = amr.regrid_interval !== nothing ? amr.regrid_interval :
               _amr_moving(amr.initial) ?
               max(1, floor(Int, amr.tag_buffer / (2 * num.cfl))) : 0
    fields = (f => getfield(amr, f) for f in fieldnames(AMR))
    return AMR(; fields..., regrid_interval=interval)
end

# The scope of refinement, checked before anything is built, in the terms of
# the `AMR` a user wrote rather than of the solver keywords it becomes.
function _check_amr_scope(prob, num)
    fail(msg) = throw(ArgumentError("AMR: " * msg))
    num.patch_grid == (1, 1, 1) || fail("cannot be combined with a patch_grid")
    prob.metric isa CartesianMetric ||
        fail("requires CartesianMetric; cylindrical and spherical runs cannot " *
             "refine yet")
    all(isnothing, num.stretch) ||
        fail("requires a uniform grid; set stretch = nothing in every direction")
    any(pair -> any(bc -> bc isa SymmetryPlaneBC, pair), prob.bcs) &&
        fail("cannot refine a run with a SymmetryPlaneBC; use SlipWallBC at that face")
    num.filt isa CompactScheme ||
        fail("requires a tridiagonal filter such as compact_filter(0.45); " *
             "$(typeof(num.filt).name.name) is not one")
    num.art.detector === :delta4 ||
        fail("requires ArtParams(detector = :delta4)")
    return nothing
end

# The node spacing of the root along each dimension, as the solver sets it.
_root_spacing(prob, num) = ntuple(3) do d
    L = prob.domain[d][2] - prob.domain[d][1]
    n = num.n_global[d]
    n == 1 ? L : isperiodic(prob.bcs[d][1]) ? L / n : L / (n - 1)
end

# Nested regions covering nested shapes. Each shape is sampled on its parent's
# lattice, over the parent's own nodes; a node within one cell diagonal of the
# shape counts, so the region brackets the shape rather than falling inside
# it, and a shape thinner than the spacing is still covered.
function _shape_regions(shapes, prob, num)
    n_global = num.n_global
    active = ntuple(d -> n_global[d] > 1, 3)
    margin = max(num.n_halo, LEVEL_BUFFER)
    origin = ntuple(d -> prob.domain[d][1], 3)
    h = _root_spacing(prob, num)
    plo = (0, 0, 0)
    phi = ntuple(d -> n_global[d] - 1, 3)
    regions = BlockRegion[]
    for (ℓ, shape) in enumerate(shapes)
        shape isa Shape ||
            throw(ArgumentError("AMR: nested initial regions are all shapes or all " *
                                "BlockRegions"))
        reach = sqrt(sum(h[d]^2 for d in 1:3 if active[d]))
        lo = [typemax(Int), typemax(Int), typemax(Int)]
        hi = [typemin(Int), typemin(Int), typemin(Int)]
        for k in plo[3]:phi[3], j in plo[2]:phi[2], i in plo[1]:phi[1]
            x = (origin[1] + i * h[1], origin[2] + j * h[2], origin[3] + k * h[3])
            signed_distance(shape, x, active) <= reach || continue
            for (d, n) in enumerate((i, j, k))
                lo[d] = min(lo[d], n)
                hi[d] = max(hi[d], n)
            end
        end
        lo[1] == typemax(Int) &&
            throw(ArgumentError("AMR: the level-$ℓ shape contains no node of " *
                                "level $(ℓ - 1)" *
                                (ℓ > 1 ? " inside the level-$(ℓ - 1) region" : "")))
        clipped = false
        offset = zeros(Int, 3)
        extent = ones(Int, 3)
        for d in 1:3
            active[d] || continue
            a, b = plo[d] + margin, phi[d] - margin
            b - a + 1 >= 4 ||
                throw(ArgumentError("AMR: level $(ℓ - 1) is too small along " *
                                    "dimension $d to hold a refined level"))
            l, u = max(lo[d], a), min(hi[d], b)
            clipped |= l > lo[d] || u < hi[d]
            l > u && ((l, u) = lo[d] > b ? (b, b) : (a, a))
            while u - l + 1 < 4
                u < b ? (u += 1) : (l -= 1)
            end
            offset[d], extent[d] = l, u - l + 1
        end
        clipped && MPI.Comm_rank(num.comm) == 0 &&
            @warn "AMR: the level-$ℓ shape reaches within $margin level-$(ℓ - 1) " *
                  "nodes of " * (ℓ == 1 ? "the domain boundary" : "its parent's edge") *
                  ", which a refined level cannot; it is refined only up to that margin."
        region = BlockRegion(Tuple(offset), Tuple(extent))
        push!(regions, region)
        # The next level nests inside this one's own nodes: the refined
        # lattice triples the spacing count and its boundary planes are
        # imposed from the parent.
        plo = ntuple(d -> active[d] ? 3 * offset[d] + 1 : 0, 3)
        phi = ntuple(d -> active[d] ? 3 * (offset[d] + extent[d] - 1) - 1 : 0, 3)
        h = ntuple(d -> active[d] ? h[d] / 3 : h[d], 3)
    end
    return regions
end

# A legal, four-node coarse box solely for constructing the tagging machinery.
# Its fine nodes remain blank until tagging chooses the actual initial layout.
function _amr_seed_region(n_global::NTuple{3,Int}, n_halo::Int)
    margin = max(n_halo, LEVEL_BUFFER)
    offext = ntuple(3) do d
        n = n_global[d]
        n == 1 && return (0, 1)
        n >= 2 * margin + 4 ||
            throw(ArgumentError("AMR initial tagging needs at least $(2 * margin + 4) " *
                                "root nodes in active dimension $d"))
        lo = max(margin + 1, (n - 4) ÷ 2 + 1)
        (lo - 1, 4)
    end
    return BlockRegion(ntuple(d -> offext[d][1], 3),
                       ntuple(d -> offext[d][2], 3))
end

function _setup_amr(prob, num, amr::AMR)
    _check_amr_scope(prob, num)
    amr = _resolve_amr(amr, num)
    if amr.initial isa Shape || (amr.initial isa AbstractVector && !isempty(amr.initial) &&
                                 all(s -> s isa Shape, amr.initial))
        shapes = amr.initial isa Shape ? [amr.initial] : collect(amr.initial)
        if amr.regrid_interval > 0
            length(shapes) == 1 ||
                throw(ArgumentError("AMR: regridding moves one refined level; " *
                                    "give one shape, or regrid_interval = 0 for " *
                                    "a fixed nested hierarchy"))
            # A regridded run keeps the shape as a tag criterion, united with
            # the others, through the predicate path.
            active = ntuple(d -> num.n_global[d] > 1, 3)
            shape = only(shapes)
            inside = (x, y, z) -> signed_distance(shape, (x, y, z), active) <= 0
            fields = (f => getfield(amr, f) for f in fieldnames(AMR))
            amr = AMR(; fields..., initial=inside)
        else
            fields = (f => getfield(amr, f) for f in fieldnames(AMR))
            amr = AMR(; fields..., initial=_shape_regions(shapes, prob, num))
        end
    end
    amr.initial === :sensor ||
        amr.initial isa BlockRegion || amr.initial isa Vector{BlockRegion} ||
        (_amr_callable(amr.initial) && _amr_predicate(amr.initial) !== nothing) ||
        throw(ArgumentError("AMR initial must be :sensor, a predicate " *
                            "(x, y, z, t) -> Bool or (x, y, z) -> Bool, a " *
                            "BlockRegion, or a Vector{BlockRegion}"))
    amr.tag_sensor_threshold > 0 && !num.art.enabled &&
        throw(ArgumentError("AMR tag_sensor_threshold reads the artificial " *
                            "coefficients, which ArtParams(enabled = false) " *
                            "leaves at zero; enable them or tag with another " *
                            "criterion"))
    if amr.initial isa BlockRegion || amr.initial isa Vector{BlockRegion}
        amr.initial isa Vector{BlockRegion} && isempty(amr.initial) &&
            throw(ArgumentError("AMR initial region vector cannot be empty"))
        return _setup_with_amr_keywords(prob, num, _amr_keywords(amr))
    end
    seed = _amr_seed_region(num.n_global, num.n_halo)
    solver, states = _setup_with_amr_keywords(
        prob, num, _amr_keywords(amr; refine=seed, bootstrap=true);
        seed_only=true)
    workspace = Workspace(states)
    if amr.tag_sensor_threshold > 0 && solver.art.enabled
        root = PatchSolver(solver, first(getfield(solver, :patches)))
        apply_bcs!(root, states[1])
        compute_rhs!(root, states[1], workspace.dQ[1])
    end
    candidate = tagged_region(solver, states[1])
    candidate === nothing &&
        throw(ArgumentError("AMR initial selection tagged no root nodes; " *
                            "supply a BlockRegion or lower the tag threshold"))
    spec = getfield(solver, :regrid)
    if candidate != seed || amr.tile > 0
        # The temporary seed is not a user-selected tile. Bypass its normal
        # minimum lifetime while preserving the requested lifetime for the
        # actual initial layout created by this check.
        spec.checks = max(spec.checks + 1, spec.lifetime)
        # The temporary fine state is blank. Layout construction reuses the
        # regrid machinery without its RHS priming; the final state is filled
        # directly from the IC below rather than interpolated seed values.
        _regrid_impl!(solver, states, workspace, nothing)
    end
    initialize!(solver, states, prob.ic)
    validate_state!(solver, states; control=num.control,
                    stage="the AMR initial state")
    # Bootstrap is layout construction, not a completed regrid check. Start
    # the user's hysteresis clock at zero for every actual initial tile.
    spec.checks = 0
    for region in keys(spec.created)
        spec.created[region] = 0
    end
    amr.regrid_interval == 0 && setfield!(solver, :regrid, nothing)
    return solver, states
end
