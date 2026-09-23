"""
    AMR(; initial=:sensor, regrid_interval=0, ...)

Refinement configuration for [`Numerics`](@ref). `initial` is `:sensor`, a
[`BlockRegion`](@ref), a vector of nested regions, or a physical-coordinate
predicate `(x, y, z, t) -> Bool`. The predicate is evaluated on root nodes at
setup and again at each regrid check when `regrid_interval > 0`. `:sensor`
uses the enabled tag criteria on the initialized coarse state. At setup,
sensor and predicate selection require at least one tagged node; use an
explicit region for a uniform initial state.

Regions use root node indices; the ratio between successive levels is three.
The other keywords match the established refinement controls of [`Numerics`](@ref).
The current regrid implementation permits one refined level. An explicit
vector of nested regions is static.
"""
Base.@kwdef struct AMR
    initial::Any = :sensor
    level_restriction::Symbol = :inject
    level_interpolation_order::Int = 6
    subcycle::Bool = false
    regrid_interval::Int = 0
    tag_threshold::Float64 = 0.02
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
    predicate = _amr_callable(amr.initial) ? _amr_physical_tag(amr.initial) : nothing
    return (; refine, level_restriction=amr.level_restriction,
            level_interpolation_order=amr.level_interpolation_order,
            subcycle=amr.subcycle, regrid_interval=interval,
            tag_threshold=amr.tag_threshold, tag_buffer=amr.tag_buffer,
            tag_sensor_threshold=amr.tag_sensor_threshold,
            tag_gradient_threshold=amr.tag_gradient_threshold,
            tag_vorticity_threshold=amr.tag_vorticity_threshold,
            tag_predicate=predicate, untag_ratio=amr.untag_ratio,
            tile_lifetime=amr.tile_lifetime, tile=amr.tile,
            rebalance=amr.rebalance, rebalance_persist=amr.rebalance_persist)
end

_amr_callable(initial) = !(initial isa Symbol || initial isa BlockRegion ||
                           initial isa AbstractVector)

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
    amr.initial === :sensor ||
        amr.initial isa BlockRegion || amr.initial isa Vector{BlockRegion} ||
        (_amr_callable(amr.initial) &&
         applicable(amr.initial, 0.0, 0.0, 0.0, 0.0)) ||
        throw(ArgumentError("AMR initial must be :sensor, a physical predicate, " *
                            "BlockRegion, or Vector{BlockRegion}"))
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
