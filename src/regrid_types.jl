# The records of a regridding run that `Solver` holds: the lattice cost
# model and the regrid configuration. The algorithms are in the regrid_*.jl
# files.

"""
    LatticeCosts

The cost model a regrid chooses each tiled level's lattice by
(`_choose_lattice`, `src/regrid_lattice.jl`), and the timers of the run that feed it.
A tile of `V` fine nodes with interface ends along `g` dimensions costs
`s (a V + b V g + k)` seconds per root step, where `s` is its level's steps
per root step, `a` the cost per node, `b` the ghost-flux pass per node and
interface dimension, and `k` the cost per tile; each off-rank tile face and
each tile's coupling to its parent costs `m` in the level's exchanges, or
`m_x` where the message leaves the node; and a regrid check costs `c` per
tile its busiest rank holds. `ready` says whether the coefficients are set: measured ones
after an interval the model has folded in, fixed ones from the start.
`nodes` holds the node of each rank of the run's communicator, numbered from
0 in the order of their lowest ranks (`_node_map`), empty until the first
regrid check reads it.

The timers are rank-local and run only while `timing` is set (a decomposed
run on host storage whose lattice is measured): `tile_time` and
`ghost_time` hold, per patch index, the nanoseconds of each tile's phases
since the last check and of its ghost-flux solves among them, `exchange`
per level (index + 1) the nanoseconds in its exchange windows, `step_mark`
the step of the last check, `check_wall` the wall of the last check,
`check_compile` its compilation and `check_tiles` the most tiles a rank
held after it, and `compile_mark` Julia's compile-time counter at the last
check, which the run turns on once it measures (`compile_counted`).
`stats` holds the exponentially weighted sums the coefficients are fitted
from and `roots` and `walls` the recent root seconds and the wall and tiles
of the recent checks, on rank 0, which fits and broadcasts.
"""
mutable struct LatticeCosts
    timing::Bool
    tile_time::Vector{UInt64}
    ghost_time::Vector{UInt64}
    exchange::Vector{UInt64}
    step_mark::Int
    check_wall::Float64
    check_compile::Float64
    check_tiles::Int
    compile_counted::Bool            # Julia's compile-time counter turned on
    compile_mark::UInt64             # its reading at the last check
    stats::Vector{Float64}
    roots::Vector{Float64}           # the root's seconds, recent checks
    walls::Vector{Float64}           # wall and tiles of recent checks, pairs
    ready::Bool
    a::Float64
    b::Float64
    k::Float64
    m::Float64
    c::Float64
    root::Float64                    # the root's measured seconds per root step
    m_x::Float64                     # per message leaving the node
    nodes::Vector{Int}               # node of each rank of the run
    # Per level, the lattice the measured model last preferred to the
    # level's own and the consecutive checks it has (`_persistent_lattice!`).
    favored::Dict{Int,Tuple{NTuple{3,Int},Int}}
end

LatticeCosts() = LatticeCosts(false, UInt64[], UInt64[], UInt64[], 0, 0.0, 0.0, 0, false,
                              zero(UInt64), Float64[], Float64[], Float64[], false, 1.0,
                              0.0, 0.0, 0.0, 0.0, 0.0, 0.0, Int[],
                              Dict{Int,Tuple{NTuple{3,Int},Int}}())

"""
    RegridSpec

Configuration and rebuild inputs for tagging-driven regridding
(the `src/regrid_*.jl` files): the regrid cadence in coarse steps, the tag criteria, the
buffer of coarse cells added around tagged cells, the nesting margin, and
everything a fine-patch rebuild needs that the `Solver` does not itself
retain: the schemes, including the `interface_divergence` source, the
halo width, the interface treatment, the level interpolation order, the
level restriction and the backend.
`last_step` records the step of the most recent regrid check so a run
resumed on the same solver keeps the cadence. Constructed by the
[`Solver`](@ref) constructor's `regrid_interval` keyword; consumed by
`regrid!`.

The tag is the union of the criteria `_tag_sweep!` evaluates over the
parent level's state: `threshold` on the relative undivided fourth
difference of the mixture density, always on; `sensor_threshold` on the
artificial diffusivity number, from coefficients a check computes on that
state with `mask_child = false`, which keeps the sensors a parent otherwise
drops at its overwritten nodes (`_tag_coefficients!`); `gradient_threshold`
on the mass-fraction change per cell; `vorticity_threshold` on the vorticity
magnitude; and `predicate`, a user closure over the parent patch. A zero
threshold or a `nothing` predicate leaves that criterion off. `tags` is the
sweep's host scratch over the root's padded extent.

Derefinement carries hysteresis. A node above a criterion's threshold tags;
one above the threshold divided by `untag_ratio` holds, which keeps an
existing tile (or the current box) but calls for no new one. A tile is not
dropped before `lifetime` regrid checks have passed since it was created,
whatever the tags say. `checks` counts the regrid checks so far and
`created` records, per current tile region, the check at which the tile was
created (0 at setup): the tag history, derived from the reduced tag flags
so that every rank holds the same record, and state that survives a regrid.
`created` is level 1's record and `deep_created[ℓ - 1]` that of level ℓ ≥ 2.

`boundaries` (the `level_boundaries` keyword, on by default) lets the tag
clamp and the lattice clip place a level on a domain face whose condition a
level carries in the run's configuration (a fold only on the host backend
under `:inject`) and across a periodic seam; any other face keeps the
nesting margin, as every face does with it off (`_placement_faces`,
`_placement_period`).

`subdivide` (the `subdivide_lattice` keyword, on by default) lets a level of
a decomposed run take a finer lattice than the configured `tile`
(`_level_chain`); off, every level keeps the configured lattice, so a
decomposed run tiles its levels as a serial one does and the two can be
compared to round-off.

The rebalance fields drive the repartition of a tiled level on measured
load: `rebalance` is the threshold on the ratio of the largest to the mean
per-rank busy time over the last interval (0 leaves ownership stored as it
is), and `persist` the number of consecutive checks the ratio must exceed
it before the level is repartitioned, so tile flicker at the tag boundary
does not move state every interval. `streak` counts those checks,
`imbalance` holds the last measured ratio, and `wall_mark`, `wait_mark` and
`wall_regrid` are the marks the interval's busy time is taken against
(`_rebalance_due!`).
"""
mutable struct RegridSpec{T}
    interval::Int
    threshold::T
    buffer::Int
    margin::Int
    n_halo::Int
    interface_rhs::Symbol
    deriv::Union{CompactScheme{T},BandedCompactScheme{T}}
    filt::Union{CompactScheme{T},BandedCompactScheme{T}}
    smoo::Union{CompactScheme{T},BandedCompactScheme{T}}
    backend::AbstractBackend
    tile::NTuple{3,Int}              # lattice edge per dimension; 0 = one box
    last_step::Int
    rebalance::Float64               # max/mean busy-time threshold; 0 = off
    persist::Int                     # consecutive checks above it before moving
    streak::Int
    imbalance::Float64               # last measured max/mean
    wall_mark::Float64               # solver.wall_total at the last check
    wait_mark::Float64               # solver.wait_total at the last check
    wall_regrid::Float64             # this rank's regrid work since, excluded
    sensor_threshold::T              # artificial diffusivity number; 0 = off
    gradient_threshold::T            # mass-fraction change per cell; 0 = off
    vorticity_threshold::T           # vorticity magnitude; 0 = off
    predicate::Union{Nothing,Function} # (patch, I) -> Bool over the parent
    tags::Array{Int8,3}              # sweep scratch, root padded extent
    untag_ratio::T                   # hold above threshold / untag_ratio
    lifetime::Int                    # minimum tile age in regrid checks
    checks::Int                      # regrid checks so far
    created::Dict{BlockRegion,Int}   # per current tile: the check it was
                                     # created at (0 at setup)
    interface_divergence::Union{Nothing,CompactScheme{T},BandedCompactScheme{T}}
    interpolation_order::Int         # Lagrange order of a rebuilt transfer
    restriction::Symbol              # level_restriction of a rebuilt transfer
    deep_created::Vector{Dict{BlockRegion,Int}} # `created` of levels 2, 3, ...
    boundaries::Bool                 # tags may place a level on a domain face
    edges::Vector{NTuple{3,Int}}     # per refined level, its lattice's edges
    costs::LatticeCosts              # the lattice cost model and its timers
    subdivide::Bool                  # a level may take a finer lattice than `tile`
end

RegridSpec{T}(interval, threshold, buffer, margin, n_halo, interface_rhs,
              deriv, filt, smoo, backend, tile, last_step,
              rebalance=0.0, persist=1) where {T} =
    RegridSpec{T}(interval, threshold, buffer, margin, n_halo, interface_rhs,
                  deriv, filt, smoo, backend, _tile_edges(tile), last_step,
                  rebalance, persist, 0, 1.0, 0.0, 0.0, 0.0,
                  zero(T), zero(T), zero(T), nothing, zeros(Int8, 0, 0, 0),
                  T(2), 1, 0, Dict{BlockRegion,Int}(), nothing, 6, :inject,
                  Dict{BlockRegion,Int}[], true, NTuple{3,Int}[], LatticeCosts(),
                  true)

# The creation record of refined level `ℓ` (`created` for level 1).
_created(spec::RegridSpec, ℓ::Int) = ℓ == 1 ? spec.created : spec.deep_created[ℓ - 1]

# The lattice edges refined level `ℓ` is tiled on: the configured `tile`
# until the regrid subdivides it (`_choose_lattice`).
_level_edges(spec::RegridSpec, ℓ::Int) =
    ℓ <= length(spec.edges) ? spec.edges[ℓ] : spec.tile

function _set_level_edges!(spec::RegridSpec, ℓ::Int, edges::NTuple{3,Int})
    while length(spec.edges) < ℓ
        push!(spec.edges, spec.tile)
    end
    spec.edges[ℓ] = edges
    return spec
end
