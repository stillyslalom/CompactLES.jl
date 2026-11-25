# The tiled regrid's lattice: the cells a tagged set selects and the choice
# of each level's lattice edge by the measured cost model (`LatticeCosts`,
# regrid_types.jl). The header of regrid_tagging.jl describes the regrid.

# --- Tiled regridding -------------------------------------------------------
#
# On a tiled level the tagged set selects lattice cells, not a box: a cell is
# wanted when any tagged coarse node, grown by the buffer, meets it. Because
# the lattice is global, a wanted tile that already exists keeps its region,
# and when its owner range survives too it keeps its arrays and its state
# untouched (only its neighbor faces and transfer are rebuilt); a newly
# wanted tile is built and initialized by interpolation of the coarse
# state; an unwanted one is dropped, its last restriction already on the
# coarse level. Between distinct lattice cells no carry-over arises: they
# overlap in a shared plane at most, and that plane takes the neighbor's
# values at the first averaging after the regrid.
#
# Ownership is stored, not recomputed: `Level.owners` is the authority, a
# surviving tile keeps its range there, and a fresh tile is placed among the
# ranks the survivors leave free or on the least loaded group
# (`_place_tiles`). The partition is recomputed by volume when the stored
# placement leaves the most loaded rank past `PLACEMENT_GAIN` times what
# a fresh partition would give it (`_balanced_owners`), and by a
# rebalance, when the per-rank busy time the run measures over the last
# interval is imbalanced past `RegridSpec.rebalance` at `RegridSpec.persist`
# consecutive checks (`_rebalance_due!`); the weights are then the measured
# per-tile costs (`_measured_weights`), so the interface factor is measured
# inside the job rather than calibrated, and the volume rule is not applied,
# since it would undo them. A surviving tile whose owner range a repartition
# moved is rebuilt on its new owners and takes its evolved solution back by
# point-to-point migration from the old owners' blocks (`_migrate_tile!`),
# one tile at a time at the regrid cadence, with no replica of the tile on
# any rank.

# Levels over the cells of each lattice of `chain` (the maximum over the
# buffered nodes meeting each cell: `TAG_MARK`, `TAG_HOLD` or 0), reduced so
# every rank derives the same sets, and whether a node at the tag level lies
# outside the feasible interval `[lo, hi]`, reduced with them. Along a
# dimension of nonzero `wrap` the lattice runs over [1, P + 1], its last
# cell ending on the seam, and a buffered node near the seam meets the cells
# on both sides of it (`_wrapped_parts`); the same number of cells covers
# it, (N − 1) ÷ tile + 1. Returns the flags and the cell counts per lattice.
function _tag_tiles(solver::Solver, Qc, spec::RegridSpec, lo::NTuple{3,Int},
                    hi::NTuple{3,Int}, wrap::NTuple{3,Int}=(0, 0, 0),
                    chain::Vector{NTuple{3,Int}}=[spec.tile])
    n_global = solver.n_global
    active = ntuple(d -> n_global[d] > 1, 3)
    Ks = [ntuple(d -> active[d] ? (n_global[d] - 1) ÷ a[d] + 1 : 1, 3) for a in chain]
    sizes = [prod(K) for K in Ks]
    starts = cumsum([0; sizes[1:end-1]])
    # Every lattice's flags and the band flag share one buffer and one
    # reduction.
    buf = zeros(Int8, sum(sizes) + 1)
    flagsets = [reshape(view(buf, s .+ (1:n)), K) for (s, n, K) in zip(starts, sizes, Ks)]
    b = spec.buffer
    _tag_sweep!(solver, Qc) do g1, g2, g3, level
        g = (g1, g2, g3)
        level == TAG_MARK && any(d -> active[d] && !(lo[d] <= g[d] <= hi[d]), 1:3) &&
            (buf[end] = one(Int8))
        for (a, K, flags) in zip(chain, Ks, flagsets)
            spans = ntuple(d -> active[d] ?
                                _wrapped_cells(g[d] - b, g[d] + b, a[d], K[d], wrap[d]) :
                                (0:0), 3)
            for k3 in spans[3], k2 in spans[2], k1 in spans[1]
                flags[k1 + 1, k2 + 1, k3 + 1] = max(flags[k1 + 1, k2 + 1, k3 + 1], level)
            end
        end
    end
    red = MPI.Allreduce(buf, max, getfield(solver, :comm))
    return [reshape(red[s .+ (1:n)], K) for (s, n, K) in zip(starts, sizes, Ks)], Ks,
           red[end] > 0
end

# --- Choosing the lattice ---------------------------------------------------
#
# A tiled level's tile is held whole by one rank (`_tile_owners`), so a level
# with fewer tiles than its parent has ranks leaves ranks idle on it, and a
# finer lattice spreads it wider at the price of more tiles: more interface
# faces, whose ghost-flux passes and exchanges grow with the tile count, and
# more tiles to rebuild at each regrid check. The regrid tiles each level on
# the lattice of a chain that a cost model predicts to give the shortest
# step (`_choose_lattice`): the configured edges first, then each lattice
# halving the largest edge of the one before that stays at least
# `MIN_TILE_EDGE` parent nodes and keeps every domain face the configured
# lattice reaches (`_edge_chain`). Every lattice of the chain is tagged in
# the same sweep and reduced together, and the wanted tiles of each are
# counted by the one rule (`_lattice_wanted`). The model (`LatticeCosts`)
# prices each lattice's wanted tiles under the placement the regrid would
# make, at the most loaded rank, and adds the rebuild cost of its tiles over
# the regrid interval. The levels below are priced with it, on the ranks the
# lattice gives the level, since a level of whole tiles spans as many ranks
# as it has tiles and its children are placed within them; the root and the
# levels above enter as the share of the step they take, so a level that
# does little of the work, and leaves its children their ranks, never gains
# enough to change. A level keeps its lattice unless another is predicted to
# make the step `PLACEMENT_GAIN` times shorter, the change's own rebuild of
# every tile included. A change of lattice replaces
# every tile: each new tile is filled from its parent as a fresh tile is,
# then takes the evolved solution of the old tiles it overlaps
# (`_carry_tiles!`), every node an old tile held as its own solution. On one
# rank the chain is the configured lattice alone.
#
# The model's coefficients are measured on the run itself: per-tile timers
# around the tile phases the step already runs (`_charge_tile!`), timers on
# the level exchanges (`_charge_exchange!`) and the wall of each regrid
# check, gathered on rank 0 at the check, fitted there and broadcast, so
# every rank takes the same lattice (`_harvest_lattice_costs!`). Until an
# interval has been measured the configured lattice holds. Because they are
# timings, the choice of lattice, and with it the layout and the solution at
# the level of the truncation error, can differ from one run to the next. A
# level therefore changes lattice under measured coefficients only once the
# model has preferred the same lattice at `LATTICE_PERSIST` consecutive
# checks (`_persistent_lattice!`), so the load of the machine during one
# interval does not set the layout, and `FIXED_LATTICE_COSTS` fixes the
# coefficients for a reproducible one.

# The least lattice edge, in parent nodes, the subdivision reaches: the
# edge at which a lattice still reaches a domain face (`_lattice_reach`
# needs an edge of at least the nesting margin, four nodes) and a tile
# carries 13 fine nodes, past the filter's nine-node minimum.
const MIN_TILE_EDGE = 4

# The chain of lattices a level of `np` ranks may take, coarsest first, over
# a parent node space of `N` nodes: `tile`, then each lattice halving the
# largest active edge that halves to at least `least` parent nodes without
# changing the faces `_lattice_reach` gives along that dimension, while the
# lattice holds at most `budget` cells (`_lattice_budget`).
function _edge_chain(tile::NTuple{3,Int}, active::NTuple{3,Bool}, N::NTuple{3,Int},
                     faces, bare, margin::Int, order::Int, least::Int, np::Int,
                     budget::Int=typemax(Int))
    chain = [tile]
    np > 1 || return chain
    reach = _lattice_reach(faces, N, active, tile, margin, order, bare)
    cells(e) = prod(active[d] ? (N[d] - 1) ÷ e[d] + 1 : 1 for d in 1:3)
    e = tile
    while true
        best = 0
        for d in 1:3
            active[d] && e[d] ÷ 2 >= least || continue
            cand = Base.setindex(e, e[d] ÷ 2, d)
            _lattice_reach(faces, N, active, cand, margin, order, bare)[d] == reach[d] ||
                continue
            (best == 0 || e[d] > e[best]) && (best = d)
        end
        best == 0 && return chain
        e = Base.setindex(e, e[best] ÷ 2, best)
        cells(e) <= budget || return chain
        push!(chain, e)
    end
end

# The most cells a lattice of the chain holds over a run of `np` ranks.
# Every lattice of the chain is tagged at every check, its cell flags reduced
# over the run (level 1) or gathered (deeper levels), and its wanted tiles
# priced, so the finest lattices set the cost of the decision; at a
# cluster's rank count the halvings down to `MIN_TILE_EDGE` reach tens of
# millions of cells. A lattice past 256 cells per rank is left out: the one
# before it holds at least 128 per rank, so a level covering a thirtieth of
# the node space already has four tiles per rank there, and whole tiles over
# a level with several per rank leave it within a fraction of a tile of
# balance. The floor keeps the chain of a run on a few nodes, and of every
# run on a workstation, whole.
_lattice_budget(np::Int) = max(1 << 20, 256 * np)

# The chain of level ℓ's lattices under the solver's placement rules, over
# its parent's node space of `N` nodes (the lattice coincident with the
# root's), for a level of `np` ranks.
function _level_chain(solver, spec::RegridSpec, N::NTuple{3,Int},
                      active::NTuple{3,Bool}, np::Int)
    spec.subdivide || return [spec.tile]
    faces = _placement_faces(solver, spec)
    least = max(MIN_TILE_EDGE, _limiter_least_extent(solver) - 1)
    return _edge_chain(spec.tile, active, N, faces, _unbuffered_faces(solver, faces),
                       spec.margin, solver.schemes.level_interpolation_order, least, np,
                       _lattice_budget(np))
end

# Whether two regions share more than a plane, over the periodic images of
# `b` under `period`: at least two nodes in common along every active
# dimension.
function _overlaps(a::BlockRegion, b::BlockRegion, active::NTuple{3,Bool},
                   period::NTuple{3,Int})
    meets(c) = all(d -> !active[d] ||
                        min(a.offset[d] + a.extent[d], c.offset[d] + c.extent[d]) -
                        max(a.offset[d], c.offset[d]) >= 2, 1:3)
    # Without a period the one image is `b`, and the image list is not built.
    period == (0, 0, 0) && return meets(b)
    return any(σ -> meets(_shifted(b, σ)), _images(period))
end

# Regions binned by node: bucket `b` along each dimension holds the regions
# with a node in `b grain + 0 .. b grain + grain − 1`, `grain` the largest
# extent among them, so a region lies in at most two buckets per dimension.
# Two regions sharing a node share a bucket, so the regions of the buckets a
# query meets, over its periodic images, hold every region it overlaps or
# covers a node of. A level of a cluster run holds tens of thousands of tiles,
# and the tests of the regrid against the old tiles (`_tile_wanted`,
# `_inherited_check`, `_covered_by`) would otherwise compare every tile with
# every old one.
struct _RegionIndex
    grain::NTuple{3,Int}
    period::NTuple{3,Int}
    buckets::Dict{NTuple{3,Int},Vector{Int}}
end

_bucket_span(r::BlockRegion, grain::NTuple{3,Int}, d::Int) =
    fld(r.offset[d] + 1, grain[d]):fld(r.offset[d] + r.extent[d], grain[d])

function _RegionIndex(regions::AbstractVector{BlockRegion}, period::NTuple{3,Int})
    grain = ntuple(d -> maximum((r.extent[d] for r in regions); init=1), 3)
    buckets = Dict{NTuple{3,Int},Vector{Int}}()
    for (i, r) in enumerate(regions)
        s = ntuple(d -> _bucket_span(r, grain, d), 3)
        for b3 in s[3], b2 in s[2], b1 in s[1]
            push!(get!(Vector{Int}, buckets, (b1, b2, b3)), i)
        end
    end
    return _RegionIndex(grain, period, buckets)
end

# The indices of the indexed regions that may meet `t` or an image of it,
# ascending.
function _candidates(index::_RegionIndex, t::BlockRegion)
    out = Int[]
    for σ in _images(index.period)
        r = _shifted(t, σ)
        s = ntuple(d -> _bucket_span(r, index.grain, d), 3)
        for b3 in s[3], b2 in s[2], b1 in s[1]
            v = get(index.buckets, (b1, b2, b3), nothing)
            v === nothing || append!(out, v)
        end
    end
    return sort!(unique!(out))
end

# The creation check a tile of a new lattice inherits: the latest among the
# old tiles it overlaps, so a subdivided tile keeps its lifetime, or
# `checks` where it overlaps none. `old_regions` may be any subset of the
# old tiles holding every one `r` overlaps (`_candidates`).
function _inherited_check(r::BlockRegion, old_regions::AbstractVector{BlockRegion},
                          created::Dict{BlockRegion,Int}, checks::Int,
                          active::NTuple{3,Bool}, period::NTuple{3,Int})
    over = [get(created, o, 0) for o in old_regions if _overlaps(r, o, active, period)]
    return isempty(over) ? checks : maximum(over)
end

# The old tiles of a level as the wanted test reads them: the tiles, the
# position of each, their index and which of them are younger than the
# lifetime, and whether the lattice under test is the current one.
struct _OldTiles
    regions::Vector{BlockRegion}
    position::Dict{BlockRegion,Int}
    index::_RegionIndex
    young::Vector{Bool}
    any_young::Bool
    current::Bool
    active::NTuple{3,Bool}
    period::NTuple{3,Int}
end

function _OldTiles(old_regions::Vector{BlockRegion}, created::Dict{BlockRegion,Int},
                   spec::RegridSpec, current::Bool, active::NTuple{3,Bool},
                   period::NTuple{3,Int})
    young = [spec.checks - get(created, r, 0) < spec.lifetime for r in old_regions]
    position = Dict(r => i for (i, r) in enumerate(old_regions))
    return _OldTiles(old_regions, position, _RegionIndex(old_regions, period),
                     young, any(young), current, active, period)
end

# Whether tile `t` of a lattice is wanted at cell level `f`: at the tag
# level, or at the hold level or younger than the lifetime where the
# current layout already refines it. On the current lattice that is the
# tile itself; on another, an old tile overlapping it, which is how the
# hysteresis carries across a change of lattice.
function _tile_wanted(t::BlockRegion, f::Int8, old::_OldTiles)
    f == TAG_MARK && return true
    if old.current
        i = get(old.position, t, 0)
        i == 0 && return false
        return f == TAG_HOLD || old.young[i]
    end
    f == TAG_HOLD || old.any_young || return false
    for i in _candidates(old.index, t)
        _overlaps(t, old.regions[i], old.active, old.period) || continue
        (f == TAG_HOLD || old.young[i]) && return true
    end
    return false
end

# The wanted tiles of level 1 on lattice `edges`, from its reduced cell
# flags, in lattice order. A cell below the tag level is examined only where
# an old tile can make it wanted: on the current lattice its own cell, on
# another a cell within one cell of an old tile.
function _lattice_wanted(flags, K::NTuple{3,Int}, edges::NTuple{3,Int},
                         active::NTuple{3,Bool}, lo::NTuple{3,Int}, hi::NTuple{3,Int},
                         spec::RegridSpec, old_regions::Vector{BlockRegion},
                         current::Bool, period::NTuple{3,Int})
    wanted = BlockRegion[]
    old = _OldTiles(old_regions, _created(spec, 1), spec, current, active, period)
    near = falses(K)
    for o in old_regions, σ in _images(period)
        r = _shifted(o, σ)
        s = ntuple(d -> active[d] ?
                        (max(fld(r.offset[d] - 1, edges[d]) - 1, 0):
                         min(fld(r.offset[d] + r.extent[d] - 1, edges[d]) + 1, K[d] - 1)) :
                        (0:0), 3)
        for k3 in s[3], k2 in s[2], k1 in s[1]
            near[k1 + 1, k2 + 1, k3 + 1] = true
        end
    end
    for k3 in 0:K[3]-1, k2 in 0:K[2]-1, k1 in 0:K[1]-1
        f = flags[k1 + 1, k2 + 1, k3 + 1]
        (f == TAG_MARK || near[k1 + 1, k2 + 1, k3 + 1]) || continue
        t = _lattice_tile((k1, k2, k3), active, edges, lo, hi)
        t === nothing && continue
        _tile_wanted(t, f, old) && push!(wanted, t)
    end
    return wanted
end

# Test and bench toggle: `(a, b, k, m, c)` or `(a, b, k, m, c, m_x)` fixes
# the coefficients of the lattice cost model (`LatticeCosts`) and turns its
# timers off, so the lattice a level takes depends on the layout alone and a
# run repeats its layout; `nothing`, the default, measures them. Without
# `m_x` a message leaving the node costs `m` as one within it does.
# `(1, 0, 0, 0, 0)` prices a lattice by the fine nodes of its most loaded
# rank.
const FIXED_LATTICE_COSTS =
    Ref{Union{Nothing,NTuple{5,Float64},NTuple{6,Float64}}}(nothing)

"""
    _node_map(comm) -> Vector{Int}

The node of each rank of `comm`, numbered from 0 in the order of each
node's lowest rank: the ranks sharing memory with it
(`MPI.COMM_TYPE_SHARED`), or the groups of `NODE_SIZE` consecutive ranks
when that toggle is set. Collective over `comm` (a split and an Allgather of
one integer per rank), and the same on every rank.
"""
function _node_map(comm::MPI.Comm)
    np = MPI.Comm_size(comm)
    NODE_SIZE[] > 0 && return [r ÷ NODE_SIZE[] for r in 0:(np - 1)]
    np == 1 && return [0]
    me = MPI.Comm_rank(comm)
    shared = MPI.Comm_split_type(comm, MPI.COMM_TYPE_SHARED, me)
    # The lowest rank of this rank's node: the shared communicator's first,
    # since the split keeps the order of `comm`.
    lead = MPI.Allreduce(me, min, shared)
    MPI.free(shared)
    leads = MPI.Allgather(lead, comm)
    order = sort!(unique(leads))
    return [searchsortedfirst(order, l) - 1 for l in leads]
end

# The coefficients a device level takes. A device phase returns as soon as
# its kernels are launched, so the tile timers would read the launch and not
# the work, and synchronizing per tile would stall the stream the stacked
# tiles share; the device path prices a lattice by the fine nodes of its
# most loaded rank instead, its tiles launching per stack rather than per
# tile.
const DEVICE_LATTICE_COSTS = (1.0, 0.0, 0.0, 0.0, 0.0)

# The weight the statistics of earlier intervals keep at each check: the
# model follows a change of regime within ten or so checks, the noise of one
# interval does not by itself change a lattice, and the tile volumes of an
# earlier lattice keep the per-tile cost apart from the per-node one for a
# while after a change.
const LATTICE_MEMORY = 0.85

# The least spread of the tile volumes, as the relative determinant of the
# volume-and-count normal equations (about the squared coefficient of
# variation), at which the per-tile cost `k` is fitted apart from the
# per-node cost `a`. Tiles of one lattice share one volume but at the domain
# edges; below this spread the fit keeps the last `k` it told apart (0
# before any), so the model does not forget the per-tile cost and flip back
# to a lattice it has just left.
const LATTICE_SPREAD = 0.01

# Whether the model measures its coefficients on this run: a decomposed
# tiled run on host storage, the subdivision on and no fixed coefficients.
# Setup constants and toggles, the same on every rank.
_lattice_measured(solver, spec::RegridSpec) =
    spec.subdivide && FIXED_LATTICE_COSTS[] === nothing && _tiled(spec.tile) &&
    spec.backend isa CPUBackend && MPI.Comm_size(getfield(solver, :comm)) > 1

# The steps level ℓ takes per root step.
_level_substeps(solver, ℓ::Int) = getfield(solver, :subcycle) ? 3.0^ℓ : 1.0

# The timers zeroed and sized to the current patch vector and levels, at the
# end of a regrid check, so the next check reads one interval's work.
function _reset_lattice_timers!(solver, spec::RegridSpec)
    costs = spec.costs
    costs.timing = _lattice_measured(solver, spec)
    costs.compile_mark = _counted_compile_ns(costs)
    n = length(getfield(solver, :patches))
    for v in (costs.tile_time, costs.ghost_time)
        resize!(v, n)
        fill!(v, zero(UInt64))
    end
    resize!(costs.exchange, length(getfield(solver, :levels)))
    fill!(costs.exchange, zero(UInt64))
    costs.step_mark = solver.step
    return costs
end

# The close of a regrid check: its wall, the seconds it spent compiling and
# the most refined tiles a rank holds after it, which the next check's
# harvest fits the rebuild cost from, and the timers reset.
function _close_lattice_check!(solver, spec::RegridSpec, wall::Float64,
                               compiled::Float64=0.0)
    costs = spec.costs
    costs.check_wall = wall
    costs.check_compile = compiled
    levels = getfield(solver, :levels)
    costs.check_tiles = _most_held(reduce(vcat, (lev.owners for lev in levels[2:end]);
                                          init=UnitRange{Int}[]))
    _reset_lattice_timers!(solver, spec)
    return costs
end

# The most tiles any rank holds under `owners`, a rank of each tile's range
# counting the tile: the tiles a regrid check rebuilds on its busiest rank,
# which sets the check's wall, since every rank builds its own tiles at once.
function _most_held(owners::Vector{UnitRange{Int}})
    held = Dict{Int,Int}()
    for o in owners, r in o
        held[r] = get(held, r, 0) + 1
    end
    return maximum(values(held); init=0)
end

# Per rank of `np`, the messages a layout's tiles send in the level's
# exchanges: one coupling with the parent per tile, and one per face a tile
# shares with a tile held by another rank. `owners` are whole-tile ranges.
_rank_messages(regions::Vector{BlockRegion}, owners::Vector{UnitRange{Int}}, faces,
               np::Int) = first(_rank_messages(regions, owners, faces, np, Int[]))

# The same split by `nodes`, the node of each rank (`_node_map`): per rank,
# the messages to a rank of its own node and those leaving it. A tile's
# coupling with its parent counts as leaving the node whenever the run spans
# more than one: the parent's blocks lie wherever its decomposition put them,
# which for the root is anywhere in the run.
function _rank_messages(regions::Vector{BlockRegion}, owners::Vector{UnitRange{Int}},
                        faces, np::Int, nodes::AbstractVector{Int})
    within = zeros(Int, np)
    across = zeros(Int, np)
    spread = !_one_node(nodes, length(nodes))
    node(r) = spread ? nodes[r + 1] : 0
    for t in eachindex(regions)
        r = first(owners[t])
        spread ? (across[r + 1] += 1) : (within[r + 1] += 1)
        for d in 1:3, side in 1:2
            q = faces[t][d][side]
            (q != 0 && first(owners[q]) != r) || continue
            node(first(owners[q])) == node(r) ? (within[r + 1] += 1) : (across[r + 1] += 1)
        end
    end
    return within, across
end

# The predicted seconds per root step of a level whose tiles `regions` are
# held as `owners` by `np` ranks, at the most loaded rank, `s` steps per root
# step. `boundaries` and `folds` are the tiles' domain faces and folded faces,
# `period` the parent node space's, and `nodes` the node of each rank.
function _layout_cost(costs::LatticeCosts, regions::Vector{BlockRegion},
                      boundaries, folds, owners::Vector{UnitRange{Int}},
                      active::NTuple{3,Bool}, period::NTuple{3,Int}, s::Float64, np::Int,
                      nodes::AbstractVector{Int}=costs.nodes)
    isempty(regions) && return 0.0
    faces = _tile_faces(regions, period)
    within, across = _rank_messages(regions, owners, faces, np, nodes)
    load = costs.m * s .* within
    any(>(0), across) && (load .+= costs.m_x * s .* across)
    for t in eachindex(regions)
        V = Float64(prod(fine_extent(regions[t], active, folds[t])))
        g = count(_interface_dims(active, boundaries[t]))
        load[first(owners[t]) + 1] += s * (costs.a * V + costs.b * V * g + costs.k)
    end
    return maximum(load)
end

# The placement the regrid would give a level of `regions` over `np` ranks
# from scratch: whole tiles by volume (`_fresh_partition`), a node's tiles
# kept together under `nodes`.
_predicted_owners(regions::Vector{BlockRegion}, active::NTuple{3,Bool}, np::Int,
                  nodes::AbstractVector{Int}=Int[]) =
    first(length(regions) > np ? _knapsack_owners(regions, active, np; nodes) :
                                 _tile_owners(regions, active, np))

# The predicted seconds per root step of refined level ℓ tiled as `regions`
# over `np` ranks, its rebuilds at the regrid checks included, and the most
# tiles a rank holds under that placement.
function _lattice_level_cost(solver, spec::RegridSpec, ℓ::Int,
                             regions::Vector{BlockRegion}, np::Int)
    isempty(regions) && return 0.0, 0
    active = ntuple(d -> solver.n_global[d] > 1, 3)
    boundaries = _region_boundaries(solver, regions, ℓ)
    folds = _region_folds(solver, boundaries)
    costs = spec.costs
    owners = _predicted_owners(regions, active, np, costs.nodes)
    held = _most_held(owners)
    return _layout_cost(costs, regions, boundaries, folds, owners, active,
                        _level_period(solver, ℓ - 1), _level_substeps(solver, ℓ), np) +
           costs.c * held / spec.interval, held
end

# The ranks a level of `n` whole tiles spans within a parent of `np`: one per
# tile up to `np` (`_tile_owners`); its children are placed within them.
_level_rank_count(n::Int, np::Int) = n == 0 ? np : min(n, np)

# The checks over which a change of lattice repays its rebuild of every tile
# of the level, beyond what the rebuilds at the checks already cost.
const LATTICE_HORIZON = 10

# The predicted seconds per root step of a refined level as it is laid out.
function _current_level_cost(solver, spec::RegridSpec, lev::Level)
    isempty(lev.transfers) && return 0.0
    active = ntuple(d -> solver.n_global[d] > 1, 3)
    regions = [lt.region for lt in lev.transfers]
    np = maximum(last, lev.owners) + 1
    costs = spec.costs
    return _layout_cost(costs, regions, [lt.boundary for lt in lev.transfers],
                        [lt.folded for lt in lev.transfers], lev.owners, active,
                        _level_period(solver, lev.index - 1),
                        _level_substeps(solver, lev.index), np) +
           costs.c * _most_held(lev.owners) / spec.interval
end

"""
    _cheapest_lattice(level, rest, current, fresh) -> Int

The lattice of a chain a level takes, from the predicted seconds per root
step of the level, with the levels below it, on each lattice (`level`,
coarsest first) and of the rest of the step (`rest`): the lattice of the
shortest step, the coarsest among equals, unless the level holds tiles on
lattice `current` and the step on it is less than `PLACEMENT_GAIN` times
the shortest, which keeps the current one. `current` is `nothing` when the
level's lattice is not in the chain, and `fresh` marks a level holding no
tile; either takes the shortest outright. Pure and deterministic, so every
rank deciding from the same costs agrees.
"""
function _cheapest_lattice(level::Vector{Float64}, rest::Float64,
                           current::Union{Nothing,Int}, fresh::Bool)
    best = argmin(level)
    (fresh || current === nothing) && return best
    rest + level[current] >= PLACEMENT_GAIN * (rest + level[best]) && return best
    return current
end

"""
    _choose_lattice(solver, spec, ℓ, chain, wanted_sets, current, np, fresh) -> Int

The index into `chain` of the lattice refined level ℓ takes, from the
wanted tiles of each lattice (`wanted_sets`) over the `np` ranks of its
parent: each priced by the cost model (`LatticeCosts`) under the placement
the regrid would make, with the levels below it, against the rest of the
step (the root and the levels above as they are laid out), by
`_cheapest_lattice`. A level of whole tiles spans as many ranks as it has
tiles, at most its parent's, and the levels below it are placed within
those (`_level_rank_count`), so a lattice too coarse to give its level a
rank per rank of its parent slows its children as well. A lattice other
than `current` also pays its first rebuild, every tile at once, over
`LATTICE_HORIZON` checks. Before the model is ready the level keeps
`current`, or takes the configured lattice. Reads only values every rank
deciding holds alike: the reduced tile sets, the stored layouts and the
broadcast model.
"""
function _choose_lattice(solver, spec::RegridSpec, ℓ::Int, chain::Vector{NTuple{3,Int}},
                         wanted_sets::Vector{Vector{BlockRegion}},
                         current::Union{Nothing,Int}, np::Int, fresh::Bool)
    length(chain) == 1 && return 1
    costs = spec.costs
    costs.ready || return something(current, 1)
    levels = getfield(solver, :levels)
    rest = costs.root
    for lev in levels[2:end]
        lev.index < ℓ && (rest += _current_level_cost(solver, spec, lev))
    end
    deeper = [[lt.region for lt in lev.transfers] for lev in levels[2:end]
              if lev.index > ℓ]
    level = zeros(length(chain))
    # A deeper level's price depends on its rank count alone, which most
    # lattices of a cluster run share (a tile per rank or more).
    priced = Dict{Tuple{Int,Int},Float64}()
    for (i, w) in enumerate(wanted_sets)
        level[i], held = _lattice_level_cost(solver, spec, ℓ, w, np)
        ranks = _level_rank_count(length(w), np)
        for (j, regions) in enumerate(deeper)
            level[i] += get!(() -> first(_lattice_level_cost(solver, spec, ℓ + j, regions,
                                                             ranks)),
                             priced, (j, ranks))
            ranks = _level_rank_count(length(regions), ranks)
        end
        i == current ||
            (level[i] += costs.c * held / (spec.interval * LATTICE_HORIZON))
    end
    cheapest = _cheapest_lattice(level, rest, current, fresh)
    pick = _persistent_lattice!(costs, ℓ, chain, cheapest, current, fresh)
    LATTICE_TRACE[] && MPI.Comm_rank(getfield(solver, :comm)) == 0 &&
        _trace_lattice(solver, spec, ℓ, chain, wanted_sets, level, rest, current, pick)
    return pick
end

# The consecutive regrid checks at which a measured model must prefer one
# lattice to the level's own before the level takes it. The coefficients are
# timings, and on the eight-rank blobs of bench/amrwin.jl the predicted gain
# of a finer lattice sat within a few percent of `PLACEMENT_GAIN` at the
# first check after the model was ready, so the load of the machine during
# one interval decided the layout of the run.
const LATTICE_PERSIST = 3

"""
    _persistent_lattice!(costs, ℓ, chain, cheapest, current, fresh) -> Int

The lattice level ℓ takes given the one the model prices `cheapest`
(`_cheapest_lattice`): `cheapest` itself under fixed coefficients, for a
level holding no tile and for one whose lattice is not in the chain, whose
choice does not depend on a timing or cannot wait. Under measured
coefficients a level on `current` changes only once the model has preferred
the same other lattice at `LATTICE_PERSIST` consecutive checks, counted in
`costs.favored`; a check that prefers `current`, or a third lattice, starts
the count again. The count advances alike on every rank deciding, since
each reads the same broadcast model at every check.
"""
function _persistent_lattice!(costs::LatticeCosts, ℓ::Int, chain::Vector{NTuple{3,Int}},
                              cheapest::Int, current::Union{Nothing,Int}, fresh::Bool)
    (costs.timing && !fresh && current !== nothing) || return cheapest
    if cheapest == current
        delete!(costs.favored, ℓ)
        return cheapest
    end
    edges, count = get(costs.favored, ℓ, (chain[cheapest], 0))
    count = edges == chain[cheapest] ? count + 1 : 1
    if count >= LATTICE_PERSIST
        delete!(costs.favored, ℓ)
        return cheapest
    end
    costs.favored[ℓ] = (chain[cheapest], count)
    return current
end

@noinline function _trace_lattice(solver, spec, ℓ, chain, wanted_sets, level, rest,
                                  current, pick)
    c = spec.costs
    println("lattice: step ", solver.step, " level ", ℓ, " model a=", c.a, " b=", c.b,
            " k=", c.k, " m=", c.m, " m_x=", c.m_x, " c=", c.c, " root=", c.root,
            " rest=", rest)
    for (i, e) in enumerate(chain)
        println("  ", e, i == current ? " current" : "", i == pick ? " pick" : "",
                " tiles ", length(wanted_sets[i]), " level s/step ", level[i])
    end
    return nothing
end

# The fixed coefficients, from `FIXED_LATTICE_COSTS` or for a device level,
# with the root priced as the model prices a tile: its largest block's nodes.
function _fix_lattice_costs!(solver, spec::RegridSpec)
    costs = spec.costs
    fixed = something(FIXED_LATTICE_COSTS[], DEVICE_LATTICE_COSTS)
    a, b, k, m, c = fixed
    root = getfield(solver, :patches)[1].decomp
    n_global = solver.n_global
    block = prod(cld(n_global[d], root.dims[d]) for d in 1:3)
    costs.a, costs.b, costs.k, costs.m, costs.c = a, b, k, m, c
    costs.m_x = length(fixed) == 6 ? fixed[6] : m
    costs.root = a * block
    costs.ready = true
    return costs
end

# The number of recent checks whose walls give the rebuild cost and whose
# median gives the root's seconds.
const LATTICE_WINDOW = 9

# A rank that spent more than this share of an interval's work compiling
# (Julia's cumulative compile time: the first steps, a tile of a new face
# combination, the first change of lattice) gives no observation for it, and
# the root's and the exchanges' seconds, which every rank's compilation
# delays, are not taken from an interval in which the longest compilation of
# any rank passed this share of the work of the most loaded rank. The second
# test is against the step rather than each rank's own work, since a rank
# holding little of the work compiling for a moment delays the others by that
# moment alone, and over thousands of ranks some rank compiles in nearly every
# interval. A check's wall is taken less the compilation inside it.
const LATTICE_COMPILE = 0.05

# A tile whose seconds per node exceed the median of its harvest by this
# factor is left out of the fit: one rank's interruption, or a compilation
# below the share above.
const LATTICE_OUTLIER = 3.0

# Fields per tile of a rank's harvest record after its level entries:
# level, fine nodes, interface dimensions, seconds per root step, and those
# of the ghost-flux solves.
const _TILE_FIELDS = 5

# Nanoseconds this process has spent compiling, counted while the lattice
# timers run (`_counted_compile_ns` turns Julia's counter on).
_compile_ns() = Base.cumulative_compile_time_ns()[1]

# The same, turning the counter on the first time the model measures, before
# the first check reads it: the counter is process-wide and counted, so a
# `@time` elsewhere leaves it on.
function _counted_compile_ns(costs::LatticeCosts)
    if costs.timing && !costs.compile_counted
        Base.cumulative_compile_timing(true)
        costs.compile_counted = true
    end
    return _compile_ns()
end

"""
    _harvest_lattice_costs!(solver, spec)

Fold the interval since the last regrid check into the lattice cost model
and refit it. Each rank records, per level, the seconds of its tiles and of
its exchanges per root step, the seconds it spent compiling, and, per tile
it holds, the tile's volume, interface dimensions and seconds; rank 0
gathers them and fits `y = s (a V + k)` to each tile's seconds less its
ghost-flux solves by non-negative least squares over exponentially weighted
sums (`LATTICE_MEMORY`; `k = 0` where the tile volumes do not spread,
`LATTICE_SPREAD`), `b` as the ratio of the ghost-flux seconds to `s V g`,
and `m` and `m_x` from the exchange seconds of each level's most loaded
rank, which waits least, against the messages it sends within its node and
out of it (`_message_fit`). The root's seconds are the median of its
most loaded rank's over the last `LATTICE_WINDOW` checks, and `c` the summed
wall of those checks, less the longest compilation of any rank in each,
over the most tiles a rank held after each (`_most_held`). A rank that
compiled during the interval gives no observation (`LATTICE_COMPILE`), and
a tile far above its harvest's median per node is left out
(`LATTICE_OUTLIER`). The
coefficients are broadcast, so every rank holds the same model. Collective
over the run's communicator: at the first check every rank reads the node
map (`_node_map`); after that, with fixed coefficients it sets them and
communicates nothing.
"""
function _harvest_lattice_costs!(solver, spec::RegridSpec)
    costs = spec.costs
    comm = getfield(solver, :comm)
    isempty(costs.nodes) && (costs.nodes = _node_map(comm))
    _lattice_measured(solver, spec) || return _fix_lattice_costs!(solver, spec)
    levels = getfield(solver, :levels)
    nlev = length(levels)
    steps = solver.step - costs.step_mark
    record = zeros(2 * nlev + 3)
    record[2nlev + 1] = costs.check_wall
    record[2nlev + 3] = costs.check_compile
    if steps > 0
        record[2nlev + 2] = (_compile_ns() - costs.compile_mark) / 1e9 / steps
        times = costs.tile_time
        ghost = costs.ghost_time
        for (i, lev) in enumerate(levels)
            for (k, pi) in enumerate(lev.patches)
                pi <= length(times) || continue
                y = times[pi] / 1e9 / steps
                record[2i - 1] += y
                lev.index == 0 && continue
                lt = lev.transfers[lev.tiles[k]]
                append!(record, (lev.index, prod(fine_extent(lt.region, lt.active, lt.folded)),
                                 count(_interface_dims(lt.active, lt.boundary)), y,
                                 ghost[pi] / 1e9 / steps))
            end
            i <= length(costs.exchange) && (record[2i] = costs.exchange[i] / 1e9 / steps)
        end
    end
    gathered = _gather_records(record, comm)
    model = zeros(8)
    if MPI.Comm_rank(comm) == 0
        steps > 0 && _fold_lattice_costs!(solver, spec, gathered, nlev)
        model .= _fit_lattice_costs(costs)
        LATTICE_TRACE[] && println("lattice harvest: step ", solver.step, " steps ", steps,
                                   " last check wall ",
                                   maximum(r[2nlev + 1] for r in gathered), " tiles ",
                                   costs.check_tiles, " compiled at most ",
                                   maximum(r[2nlev + 2] for r in gathered), " model ",
                                   model)
    end
    MPI.Bcast!(model, comm; root=0)
    costs.ready = model[1] > 0
    costs.a, costs.b, costs.k, costs.m, costs.c, costs.root, costs.m_x = model[2:8]
    return costs
end

# Every rank's record on rank 0, in rank order, and nothing elsewhere: the
# lengths gathered, then the values in one Gatherv. An object gather would
# serialize every record and deserialize them one by one on rank 0, a cost
# in the rank count that a cluster run pays at every check.
function _gather_records(record::Vector{T}, comm::MPI.Comm) where {T}
    counts = MPI.Gather(Cint(length(record)), comm; root=0)
    if MPI.Comm_rank(comm) == 0
        flat = Vector{T}(undef, sum(counts))
        MPI.Gatherv!(record, MPI.VBuffer(flat, counts), comm; root=0)
        ends = cumsum(counts)
        return [flat[(e - c + 1):e] for (e, c) in zip(ends, counts)]
    end
    MPI.Gatherv!(record, nothing, comm; root=0)
    return Vector{T}[]
end

# The weighted sums, on rank 0: the tile fit's five (Σx₁², Σx₁x₂, Σx₂²,
# Σx₁y, Σx₂y for x = (s V, s)), the ghost-flux seconds and their Σ s V g,
# the tiles counted, and the exchange fit's five over the most loaded ranks
# (`_message_sums!`).
const _STAT_SIZE = 13

function _fold_lattice_costs!(solver, spec::RegridSpec, records, nlev::Int)
    costs = spec.costs
    levels = getfield(solver, :levels)
    np = length(records)
    work(rec) = sum(@view rec[1:2nlev])
    clean = [rec[2nlev + 2] <= LATTICE_COMPILE * work(rec) for rec in records]
    steady = maximum(rec[2nlev + 2] for rec in records) <=
             LATTICE_COMPILE * maximum(work(rec) for rec in records)
    # Every tile's observation from a rank that did not compile, with its
    # seconds per node and step.
    tiles = NTuple{5,Float64}[]
    for (rec, ok) in zip(records, clean), at in (2nlev + 4):_TILE_FIELDS:length(rec)
        ok && push!(tiles, ntuple(j -> rec[at + j - 1], 5))
    end
    perstep(t) = (t[4] - t[5]) / (_level_substeps(solver, Int(t[1])) * t[2])
    new = zeros(_STAT_SIZE)
    if !isempty(tiles)
        typical = _median(perstep.(tiles))
        for t in tiles
            perstep(t) <= LATTICE_OUTLIER * typical || continue
            s = _level_substeps(solver, Int(t[1]))
            x1, x2, y = s * t[2], s, t[4] - t[5]
            new[1:5] .+= (x1 * x1, x1 * x2, x2 * x2, x1 * y, x2 * y)
            new[6] += t[5]
            new[7] += s * t[2] * t[3]
            new[8] += 1
        end
    end
    if steady
        for i in 2:nlev
            lev = levels[i]
            isempty(lev.transfers) && continue
            busiest = argmax([rec[2i - 1] for rec in records])
            regions, owners = _measured_layout(lev)
            faces = _tile_faces(regions, _level_period(solver, lev.index - 1))
            within, across = _rank_messages(regions, owners, faces,
                                            max(np, maximum(last, owners) + 1),
                                            costs.nodes)
            s = _level_substeps(solver, lev.index)
            _message_sums!(new, s * within[busiest], s * across[busiest],
                           records[busiest][2i])
        end
        _window!(costs.roots, maximum(rec[1] for rec in records))
    end
    stats = costs.stats
    isempty(stats) && append!(stats, zeros(_STAT_SIZE))
    stats .= LATTICE_MEMORY .* stats .+ new
    # A rank waiting at the check's collectives for one that compiles waits
    # out that compilation, so the slowest compilation comes off the wall.
    wall = maximum(rec[2nlev + 1] for rec in records) -
           maximum(rec[2nlev + 3] for rec in records)
    if wall > 0 && costs.check_tiles > 0
        push!(costs.walls, wall, costs.check_tiles)
        length(costs.walls) > 2LATTICE_WINDOW && deleteat!(costs.walls, 1:2)
    end
    return costs
end

# The median of the values of `v`.
function _median(v)
    x = sort!(collect(Float64, v))
    n = length(x)
    return isodd(n) ? x[(n + 1) ÷ 2] : (x[n ÷ 2] + x[n ÷ 2 + 1]) / 2
end

# Append `x` to the window of recent values, dropping the oldest past
# `LATTICE_WINDOW`.
function _window!(v::Vector{Float64}, x::Float64)
    push!(v, x)
    length(v) > LATTICE_WINDOW && popfirst!(v)
    return v
end

# The layout of each refined level as rank 0 holds it: regions and owners.
_measured_layout(lev::Level) = ([lt.region for lt in lev.transfers], lev.owners)

# The wall of a check per tile its busiest rank holds after it, over the
# recent checks (`walls` holds the pairs of wall and tiles): their summed
# wall over their summed tiles, since the rebuilds that cost a check most
# come at some checks only. The ranks rebuild their own tiles at once, so a
# check's wall follows the tiles of one rank rather than the level's count,
# and a lattice that spreads a level over more ranks rebuilds no more tiles
# per rank. The part of a check no lattice changes (the tag sweep, the
# coefficient priming) is counted per tile too, which prices a finer
# lattice's rebuilds high.
function _rebuild_cost(walls::Vector{Float64})
    tiles = sum(@view walls[2:2:end]; init=0.0)
    return tiles > 0 ? sum(@view walls[1:2:end]) / tiles : 0.0
end

# The exchange observation of one level's most loaded rank folded into the
# sums `new[9:13]`: `y` seconds against `x₁` messages within its node and
# `x₂` out of it, each per root step, weighted by 1/(x₁ + x₂) since the
# seconds of many messages vary as their count does. The sums are Σx₁y/x,
# Σx₁²/x, Σx₁x₂/x, Σx₂²/x and Σx₂y/x, x = x₁ + x₂; with every message on one
# node they are the seconds and the messages summed.
function _message_sums!(new::Vector{Float64}, x1::Float64, x2::Float64, y::Float64)
    x1 + x2 > 0 || return new
    if x2 == 0
        new[9] += y
        new[10] += x1
    elseif x1 == 0
        new[12] += x2
        new[13] += y
    else
        x = x1 + x2
        new[9] += x1 * y / x
        new[10] += x1 * x1 / x
        new[11] += x1 * x2 / x
        new[12] += x2 * x2 / x
        new[13] += x2 * y / x
    end
    return new
end

"""
    _message_fit(t1, s11, s12, s22, t2, spread) -> (m, m_x)

The seconds per message within a node and out of it from the weighted sums
of `_message_sums!`: the weighted least squares of both where the two
counts vary apart (the relative determinant above `spread`) and both come
out non-negative, and otherwise one coefficient for every message, which is
what the observations identify when the messages never leave the node or
leave it in a fixed proportion. On one node it is the exchange seconds over
the messages, summed.
"""
function _message_fit(t1::Float64, s11::Float64, s12::Float64, s22::Float64,
                      t2::Float64, spread::Float64)
    s22 > 0 || (m = s11 > 0 ? t1 / s11 : 0.0; return (m, m))
    s11 > 0 || (m = t2 / s22; return (m, m))
    det = s11 * s22 - s12 * s12
    if det > spread * s11 * s22
        m = (s22 * t1 - s12 * t2) / det
        m_x = (s11 * t2 - s12 * t1) / det
        m >= 0 && m_x >= 0 && return (m, m_x)
    end
    m = max((t1 + t2) / (s11 + 2s12 + s22), 0.0)
    return (m, m)
end

# The model `[ready, a, b, k, m, c, root, m_x]` from the weighted sums and the
# windows (rank 0).
function _fit_lattice_costs(costs::LatticeCosts)
    st = costs.stats
    (isempty(st) || st[8] <= 0) && return [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    a, k = _nonnegative_fit(st[1], st[2], st[3], st[4], st[5], LATTICE_SPREAD, costs.k)
    # A noisy interval can put the fit on its boundary at a = 0, every tile's
    # seconds a constant whatever its nodes, and a model without a cost per
    # node prices a lattice by its faces and messages alone. A tile's seconds
    # always carry its nodes, so the fit then takes them alone, through the
    # origin. Without a positive one (no tile measured any time) the model
    # is not ready and every level keeps its lattice.
    a > 0 || st[1] <= 0 || ((a, k) = (max(st[4] / st[1], 0.0), 0.0))
    b = st[7] > 0 ? st[6] / st[7] : 0.0
    m, m_x = _message_fit(st[9], st[10], st[11], st[12], st[13], LATTICE_SPREAD)
    c = _rebuild_cost(costs.walls)
    root = isempty(costs.roots) ? 0.0 : _median(costs.roots)
    return [a > 0 ? 1.0 : 0.0, a, b, k, m, c, root, m_x]
end

"""
    _nonnegative_fit(sxx, sx1, s11, sxy, s1y, spread, β0=0) -> (α, β)

The least-squares `y ≈ α x + β`, both non-negative, from the sums
`Σx²`, `Σx`, `Σ1`, `Σxy`, `Σy` (each possibly weighted alike). Where `x`
does not spread (the relative determinant `1 − (Σx)² / (Σx² Σ1)` below
`spread`) the intercept is not told apart from the slope, and the slope is
fitted with the intercept held at `β0`.
"""
function _nonnegative_fit(sxx::Float64, sx1::Float64, s11::Float64, sxy::Float64,
                          s1y::Float64, spread::Float64, β0::Float64=0.0)
    sxx > 0 || return (0.0, s11 > 0 ? max(s1y / s11, 0.0) : 0.0)
    origin = (max(sxy / sxx, 0.0), 0.0)
    det = sxx * s11 - sx1 * sx1
    s11 > 0 && det > spread * sxx * s11 ||
        return (max((sxy - β0 * sx1) / sxx, 0.0), β0)
    α = (s11 * sxy - sx1 * s1y) / det
    β = (sxx * s1y - sx1 * sxy) / det
    α >= 0 && β >= 0 && return (α, β)
    # On the boundary: the better of the two one-parameter fits.
    level = (0.0, max(s1y / s11, 0.0))
    residual((p, q)) = p * p * sxx + 2p * q * sx1 + q * q * s11 - 2p * sxy - 2q * s1y
    return residual(origin) <= residual(level) ? origin : level
end
