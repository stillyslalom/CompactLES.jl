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

A refined level lies at least `max(n_halo, 4)` of its parent's nodes inside
its parent, which is the room the coarse–fine transfer needs. By default this
holds at the root's boundaries too, periodic or not, and a region asked for
closer to a boundary is reduced to fit, with a warning naming the level. With
`level_boundaries = true`, a shape or a tagged feature is refined up to a wall
or NSCBC face instead, which the level then carries at its own spacing (see
the keyword below).

`regrid_interval` defaults to 0 (a fixed region) for a shape, a region or a
predicate of position alone, and for `:sensor` or a time-dependent predicate
to the number of steps a feature moving at the CFL limit takes to cross half
of `tag_buffer`, so a followed feature cannot leave its refined region between
checks. `tile = 0` covers the tagged cells with one box, the cheaper cover of a
single compact feature; a positive edge covers them with lattice tiles instead,
so that separated features (a shock and a distant interface) refine
separately rather than as one bounding box, at a per-tile cost that makes
small edges expensive in three dimensions. At setup, sensor and predicate
selection require at least one tagged node, except in a tiled run that
regrids: there an initial state that tags nothing starts with no refined
tiles, the first regrid check that tags creates them, and a check at which
nothing tags or holds removes every tile past `tile_lifetime`.

`tag_threshold`, the density criterion, defaults to `0.02` except under a
predicate, where it defaults to `Inf` so that the predicate alone selects the
refined region; give it explicitly to combine the two.
`tag_sensor_threshold` reads the artificial coefficients, so it requires
`ArtificialProperties(enabled = true)`.

Regions use root node indices; the ratio between successive levels is three.
An explicit vector of nested regions is static unless `regrid_interval` and
`tile` are given, when it is the initial layout. `AMR(base; keywords...)`
copies `base` with the given keywords replaced.

# Keywords

- `level_restriction`: `:inject` (default) writes the fine coincident-node
  values onto the covered region of the parent; `:filter` applies the
  invertible transfer pair's anti-alias filter first.
- `level_interpolation_order`: 2, 4, 6, 8 or 10; by default two above the
  interior order of `deriv`, at most 10, so 8 for [`lele_d1_6`](@ref) and
  10 for [`lele_d1_8`](@ref) and [`lele_d1_10`](@ref), and the interior
  order itself under `PatchInterfaces(flux = :closure)`. The order of the
  Lagrange interpolation from the parent that fills a refined level's ghost
  ring and boundary planes at every stage, and a newly refined region at a
  regrid. Under the closure rows, order 8 with C6 lowers the error with
  viscosity, the filter, a multidimensional level or an interface divergence
  scheme; 2 is the only monotone choice. A checkpoint records it with the
  numerics.
- `subcycle`: `false` (default) advances every level at the global dt;
  `true` selects the Berger–Oliger step, three steps of a third of the
  parent's step on each refined level, recursively, with Hermite boundary
  forcing.
- `regrid_interval`: `0` keeps the layout static; a positive `K` retags the
  parent level every `K` steps. The default is described above.
- `max_levels` (default: one more than the regions `initial` gives): the
  number of levels, the root included. With `regrid_interval` and `tile`,
  every refined level regrids, each tagged on the level above it and nested
  in its tiles, and a level that `initial` does not give starts with no
  tiles. A regridded level with children buffers its tags by enough parent
  nodes for its children to nest. Rebalancing is not available with more
  than one regridded level.
- `tag_threshold` and `tag_buffer` (default `4`): the tagging threshold on
  the relative undivided fourth difference of the mixture density, and the
  coarse-cell buffer added around tagged cells. The tag is the union of this
  criterion with the three below and the predicate, each evaluated per point
  over the parent level's state; every one of those is off by default.
- `tag_sensor_threshold` (default `0`, off): a threshold on the artificial
  diffusivity number ((μ\\* + β\\*)/ρ + κ\\*/(ρ c_p) + max_k D\\*_k) / (c h),
  the artificial diffusivity of the last right-hand-side evaluation in units
  of the acoustic cell diffusivity, read from the coefficient arrays the
  scheme itself wrote. It measures where the scheme is regularizing an
  under-resolved feature; a captured Sod shock reads about 2 under the
  default `C_beta`.
- `tag_gradient_threshold` (default `0`, off): a threshold on the
  mass-fraction change per cell, max_k |δY_k| over the centered difference
  of one cell, for mixing layers. Dimensionless; 0.05 tags an interface
  resolved over about ten cells.
- `tag_vorticity_threshold` (default `0`, off): a threshold on the vorticity
  magnitude |∇ × u| from centered differences, in the run's units of
  inverse time.
- `untag_ratio` (default `2`) and `tile_lifetime` (default `1`): the
  derefinement hysteresis. A node above a criterion's threshold divided by
  `untag_ratio` holds an existing tile (the current box, with `tile = 0`)
  without calling for a new one, so a tile at the edge of a feature does
  not flicker as the feature crosses the threshold; `1` disables the hold
  band. A tile is not dropped before `tile_lifetime` regrid checks have
  passed since its creation.
- `tile`: `0` (default) covers each refined region with one patch; a
  positive edge (in parent nodes, at least 3) covers it with the tiles of a
  global lattice of that edge instead, abutting tiles sharing their
  interface plane and coupled as root slabs are. Regridding then moves
  tiles in and out of the set, a surviving tile never changing its region,
  and the set may become empty.
- `level_boundaries` (default `false`): at `true`, shapes and tags may place
  a level on a domain face carrying `SlipWallBC`, `NoSlipWallBC`,
  `NSCBCOutflowBC` or `NSCBCInflowBC`, and a static shape's first level on
  a `SymmetryPlaneBC` or the `AxisBC` of an r-z run, on the host backend
  under `:inject` restriction. The level carries the face's condition at its
  own spacing, so a feature at a wall or an open face is refined up to the
  face. Periodic seams and other faces keep the margin, and so do a symmetry
  plane and the axis under regridding. With `tile`, a face keeps it also
  when the tile next to the face's tile would come within the margin of
  the domain: an edge below `max(n_halo, 4)`, or a partial last lattice
  cell at the high face spanning fewer parent cells than that. At `false`
  every face keeps the margin.
- `rebalance` (default `0`, off) and `rebalance_persist` (default `2`): a
  threshold on the ratio of the largest to the mean per-rank busy time over
  a regrid interval, above which a tiled level is repartitioned on those
  measurements once the ratio has exceeded it at that many consecutive
  regrid checks. Requires `tile` and regridding. Off, a surviving tile keeps
  its owner ranks across every regrid.
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
    max_levels::Union{Nothing,Int} = nothing
    level_boundaries::Bool = false
end

AMR(base::AMR; kw...) = _with(base, kw)

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
            rebalance=amr.rebalance, rebalance_persist=amr.rebalance_persist,
            max_levels=amr.max_levels,
            level_boundaries=amr.level_boundaries && interval > 0)
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
    return AMR(amr; regrid_interval=interval)
end

# The scope of refinement, checked before anything is built, in the terms of
# the `AMR` a user wrote rather than of the solver keywords it becomes.
function _check_amr_scope(prob, num)
    fail(msg) = throw(ArgumentError("AMR: " * msg))
    num.execution.patch_grid == (1, 1, 1) ||
        fail("cannot be combined with a patch_grid")
    prob.metric isa CartesianMetric ||
        (prob.metric isa CylindricalMetric && num.n_global[2] == 1) ||
        fail("requires CartesianMetric or CylindricalMetric with θ collapsed " *
             "(n_global[2] = 1); spherical and resolved-θ runs cannot refine yet")
    all(isnothing, num.stretch) ||
        fail("requires a uniform grid; set stretch = nothing in every direction")
    return nothing
end

# The node spacing of the root along each dimension, as the solver sets it:
# an axis moves the first radial node half a cell off r = 0, so the line
# carries half a cell less, and a symmetry plane lies half a cell beyond the
# node nearest it.
_root_spacing(prob, num) = ntuple(3) do d
    L = prob.domain[d][2] - prob.domain[d][1]
    n = num.n_global[d]
    planes = count(bc -> bc isa SymmetryPlaneBC, prob.bcs[d])
    n == 1 ? L : planes > 0 ? L / (n - planes / 2) :
    isperiodic(prob.bcs[d][1]) ? L / n :
    _root_axis(prob, d) ? L / (n - 0.5) : L / (n - 1)
end

# Whether dimension `d` of the root is the radius of an axis, whose first
# node lies half a spacing from the domain's low end.
_root_axis(prob, d::Int) = d == 1 && prob.bcs[1][1] isa AxisBC

# The coordinate of the root's first node along each dimension: half a cell
# inside an axis or a symmetry plane at the low end.
_root_first_node(prob, h) =
    ntuple(d -> prob.domain[d][1] +
                (_root_axis(prob, d) || prob.bcs[d][1] isa SymmetryPlaneBC ?
                 h[d] / 2 : 0.0), 3)

# Nested regions covering nested shapes. Each shape is sampled on its parent's
# lattice, over the parent's own nodes; a node within one cell diagonal of the
# shape counts, so the region brackets the shape rather than falling inside
# it, and a shape thinner than the spacing is still covered. A region stays
# the nesting margin inside its parent's own nodes, except at a domain face
# `_shape_faces` lets it reach.
function _shape_regions(shapes, prob, num, amr::AMR)
    n_global = num.n_global
    active = ntuple(d -> n_global[d] > 1, 3)
    margin = max(num.n_halo, LEVEL_BUFFER)
    order = something(amr.level_interpolation_order,
                      default_interpolation_order(num.deriv, num.patch_interfaces.flux))
    h = _root_spacing(prob, num)
    origin = _root_first_node(prob, h)
    plo = (0, 0, 0)
    phi = ntuple(d -> n_global[d] - 1, 3)
    # The last node of the parent level's node space along each dimension.
    top = phi
    reach = _shape_faces(prob, num, amr)
    regions = BlockRegion[]
    for (ℓ, shape) in enumerate(shapes)
        shape isa Shape ||
            throw(ArgumentError("AMR: nested initial regions are all shapes or all " *
                                "BlockRegions"))
        # A fold (a symmetry plane, the axis) is reached by the first
        # refined level only.
        ℓ == 1 || (reach = ntuple(d -> ntuple(side ->
            reach[d][side] && !_level_fold_condition(prob.bcs[d][side]), 2), 3))
        amr.tile > 0 && (reach = _lattice_reach(reach, top .+ 1, active, amr.tile,
                                                margin, order))
        reach_lo = ntuple(d -> active[d] && reach[d][1], 3)
        reach_hi = ntuple(d -> active[d] && reach[d][2], 3)
        span = sqrt(sum(h[d]^2 for d in 1:3 if active[d]))
        lo = [typemax(Int), typemax(Int), typemax(Int)]
        hi = [typemin(Int), typemin(Int), typemin(Int)]
        for k in plo[3]:phi[3], j in plo[2]:phi[2], i in plo[1]:phi[1]
            x = (origin[1] + i * h[1], origin[2] + j * h[2], origin[3] + k * h[3])
            signed_distance(shape, x, active) <= span || continue
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
            a = plo[d] + (reach_lo[d] ? 0 : margin)
            b = phi[d] - (reach_hi[d] ? 0 : margin)
            b - a + 1 >= 4 ||
                throw(ArgumentError("AMR: level $(ℓ - 1) is too small along " *
                                    "dimension $d to hold a refined level"))
            l, u = max(lo[d], a), min(hi[d], b)
            clipped |= l > lo[d] || u < hi[d]
            l > u && ((l, u) = lo[d] > b ? (b, b) : (a, a))
            # An end inside the margin band of a face the region may reach
            # moves onto the face, where it needs no room for a box; at a
            # face without a box buffer beyond it the region spans the
            # interpolation order (a fold keeps its buffer). The
            # widening repeats the move.
            while true
                reach_lo[d] && l < plo[d] + margin && (l = plo[d])
                reach_hi[d] && u > phi[d] - margin && (u = phi[d])
                need = _placement_extent(order,
                                         l == 0 && reach_lo[d] &&
                                         !_level_fold_condition(prob.bcs[d][1]),
                                         u == top[d] && reach_hi[d] &&
                                         !_level_fold_condition(prob.bcs[d][2]))
                (u - l + 1 >= need || u - l + 1 >= b - a + 1) && break
                u < b ? (u += 1) : (l -= 1)
            end
            offset[d], extent[d] = l, u - l + 1
        end
        clipped && MPI.Comm_rank(num.execution.comm) == 0 &&
            @warn "AMR: the level-$ℓ shape reaches within $margin level-$(ℓ - 1) " *
                  "nodes of " * (ℓ == 1 ? "the domain boundary" : "its parent's edge") *
                  ", which a refined level cannot; it is refined only up to that margin."
        region = BlockRegion(Tuple(offset), Tuple(extent))
        push!(regions, region)
        # The next level nests inside this one's own nodes: the refined
        # lattice triples the spacing count and its boundary planes are
        # imposed from the parent, except on a domain face, where the plane
        # is the level's own and the next level may reach it too.
        at_lo = ntuple(d -> reach_lo[d] && offset[d] == 0, 3)
        at_hi = ntuple(d -> reach_hi[d] && offset[d] + extent[d] - 1 == top[d], 3)
        plo = ntuple(d -> active[d] ? 3 * offset[d] + (at_lo[d] ? 0 : 1) : 0, 3)
        phi = ntuple(d -> active[d] ?
                          3 * (offset[d] + extent[d] - 1) - (at_hi[d] ? 0 : 1) : 0, 3)
        top = ntuple(d -> 3 * top[d], 3)
        reach = ntuple(d -> (at_lo[d], at_hi[d]), 3)
        h = ntuple(d -> active[d] ? h[d] / 3 : h[d], 3)
    end
    return regions
end

# The domain faces a shape's level may reach: none, unless `level_boundaries`
# is set, and then the faces whose condition a refined level carries
# (`_level_boundary_condition`), a fold (a symmetry plane, the r-z axis) only
# where setup admits a level on one (the host backend with `:inject`
# restriction; the shapes are static here).
function _shape_faces(prob, num, amr::AMR)
    amr.level_boundaries || return _NO_BOUNDARY
    active = ntuple(d -> num.n_global[d] > 1, 3)
    folds = !(num.execution.backend isa DeviceBackend) && amr.level_restriction === :inject
    return ntuple(d -> ntuple(side -> begin
        bc = prob.bcs[d][side]
        active[d] && !isperiodic(bc) && _level_boundary_condition(bc) &&
            (folds || !_level_fold_condition(bc))
    end, 2), 3)
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
            amr = AMR(amr; initial=inside)
        else
            amr = AMR(amr; initial=_shape_regions(shapes, prob, num, amr))
        end
    end
    amr.initial === :sensor ||
        amr.initial isa BlockRegion || amr.initial isa Vector{BlockRegion} ||
        (_amr_callable(amr.initial) && _amr_predicate(amr.initial) !== nothing) ||
        throw(ArgumentError("AMR initial must be :sensor, a predicate " *
                            "(x, y, z, t) -> Bool or (x, y, z) -> Bool, a " *
                            "BlockRegion, or a Vector{BlockRegion}"))
    amr.max_levels === nothing || amr.max_levels <= 2 ||
        amr.tile > 0 && amr.regrid_interval > 0 ||
        throw(ArgumentError("AMR: max_levels = $(amr.max_levels) regrids " *
                            "$(amr.max_levels - 1) refined levels, which requires " *
                            "tile > 0 and regridding"))
    amr.tag_sensor_threshold > 0 && !num.art.enabled &&
        throw(ArgumentError("AMR tag_sensor_threshold reads the artificial " *
                            "coefficients, which ArtificialProperties(enabled = false) " *
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
    # A regridded tiled level may hold no tiles, so a start with nothing tagged
    # begins unrefined and the first regrid check that tags creates its tiles.
    # The box has no empty form.
    candidate === nothing && !(amr.tile > 0 && amr.regrid_interval > 0) &&
        throw(ArgumentError("AMR initial selection tagged no root nodes; " *
                            "supply a BlockRegion, lower the tag threshold, or " *
                            "give tile > 0 with regridding to start unrefined"))
    spec = getfield(solver, :regrid)
    if candidate != seed || amr.tile > 0
        # The temporary seed is not a user-selected tile. Bypass its normal
        # minimum lifetime while preserving the requested lifetime for the
        # actual initial layout created by this check.
        spec.checks = max(spec.checks + 1, spec.lifetime)
        # The temporary fine state is blank. Layout construction reuses the
        # regrid machinery without its RHS priming; the final state is filled
        # directly from the IC below rather than interpolated seed values.
        # A deeper level is tagged on its parent, whose seed state is filled
        # from the root for the first pass; each later pass tags on the
        # initial condition itself.
        deep = nlevels(solver) > 2
        deep && _fill_levels_from_parents!(solver, states)
        _regrid_impl!(solver, states, workspace, nothing; bootstrap=true)
        for _ in 3:nlevels(solver)
            initialize!(solver, states, prob.ic)
            spec.checks += spec.lifetime
            _regrid_impl!(solver, states, workspace, nothing; bootstrap=true)
        end
    end
    initialize!(solver, states, prob.ic)
    validate_state!(solver, states; control=num.control,
                    stage="the AMR initial state")
    # Bootstrap is layout construction, not a completed regrid check. Start
    # the user's hysteresis clock at zero for every actual initial tile.
    spec.checks = 0
    for record in (spec.created, spec.deep_created...), region in keys(record)
        record[region] = 0
    end
    amr.regrid_interval == 0 && setfield!(solver, :regrid, nothing)
    return solver, states
end
