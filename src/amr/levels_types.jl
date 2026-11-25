# Level refinement: a hierarchy of nested patches, each level one patch over
# a refined region of its parent, the coupling between a patch and its
# parent, and the machinery that distributes that coupling. Design rationale
# and measurements: reference/AMR_GPU.md.
#
# Each refined region is given in its parent level's node space and is
# covered by one patch at refinement ratio 3, node-centered: a region of m
# parent nodes per refined dimension carries 3m − 2 fine nodes, parent node
# a + k − 1 coinciding with fine node 3k − 2. By default every level advances
# every RK stage with the same dt (the global minimum, which the shared
# `max_rate` reduction supplies once the patches join `solver.patches`), so
# no temporal interpolation arises anywhere. Under `subcycle = true` each
# level instead takes three steps of dt/3 per step of its parent, recursively
# (`_advance_level!`, timestep_rk.jl), and the temporal interpolation this
# needs is the Hermite box at the end of levels_transfer.jl. Under `regrid_interval`
# the refined level of a two-level hierarchy moves as one box or as tiles,
# and a tiled hierarchy of more refined levels, up to `max_levels`, is
# regridded level by level on the host backend (the regrid_*.jl files).
#
# A patch and its parent are coupled on two schedules, both by default
# through the point-sample halves of the transfer machinery (transfer.jl):
#
#   - After every RK stage update, `prolong_level_ghosts!` interpolates the
#     coarse state (at `level_interpolation_order`, by default the derivative
#     operator's interior order, `default_interpolation_order` in
#     transfer.jl) over a box extending `LEVEL_BUFFER` coarse nodes
#     beyond the refined region and overwrites the fine patch's ghost ring and
#     its boundary-plane nodes from the result. Including the plane nodes lets
#     the coarse solution force the fine solve's boundary as a `DirichletBC`
#     face would, removing the drift mode
#     between the levels without an averaging step.
#
#   - After every completed step (post-filter), `restrict_level!` copies the
#     fine field's coincident-node values onto the covered coarse region,
#     holding `RESTRICT_MARGIN` coarse nodes back from the boundary (see that
#     constant).
#     Between restrictions the coarse operator advances the covered nodes,
#     so on a fixed grid the coupling carries a term first order in the step
#     and proportional to the difference between the coarse and fine
#     right-hand sides there, O(dt H^p), which a fixed cfl places above the
#     spatial order. Restricting before every stage of the global step
#     removes the term but moves the error against the exact solution at
#     the default cfl, and on a Sod crossing, by under 2%, at seven
#     restrictions per step instead of one, so the schedule stays per step
#     (bench/archive/restrictcost.jl). Under subcycling the parent's step precedes
#     its children's, so there is no stage to restrict at; there most of a
#     refined level's time error is fourth order at the root step, the
#     integrator's and the Hermite shell's reconstruction of the parent.
#
# The invertible filter pair itself (`prolong!`/`restrict!`, deconvolution
# against Gaussian filtering) is not the default coupling, on a
# measurement: the pair's contract is that prolongation input is samples of
# the filtered field, while the live coarse solution is point samples of the
# field itself. Deconvolving point samples "sharpens" data that was never
# smoothed and filtering on the way down attenuates resolved content, both
# O(h²) against the solution: the entropy-wave gate measured order 1.3–1.7
# and errors three decades above the interpolation/injection coupling's,
# which measures order ≈ 3.5 (the one-sided divergence closures binding, as
# at a same-level patch interface). `level_restriction = :filter` keeps the
# filtered path selectable; its anti-alias smoothing is the tool to reach
# for if injection restriction proves positivity-limited on captured shocks.
# Regridding likewise initializes new fine regions by interpolation, since
# freshly covered coarse data are point samples too (regrid_swap.jl).
#
# Multi-dimensional transfer is a tensor product realized as a chain: each
# active dimension is refined in turn, so a chain of K = (number of active
# dimensions) `TransferPlan`s connects the coarse box to the fully fine box
# through K − 1 intermediate grids, each stage carrying its own scratch array.
#
# Scope, enforced by `Solver`: the Cartesian or axisymmetric (θ-collapsed)
# cylindrical metric, no stretching, no fold on a refined level but a symmetry
# plane or the r-z axis, which is folded on every level reaching it (see
# "Levels on the domain boundary" in levels_coupling.jl), and no same-level patch
# decomposition alongside refinement. The transfers move the conserved
# variables themselves on either metric; each tile evaluates its own geometry
# at its nodes. The fine patch's line solves close at the coarse–fine
# boundary with the same-level interface rows (extended-data gradients and
# filters). Under `interface_flux = :ghost`, the default, the whole flux is
# differenced through the face with the gradient rows, the ghost values of
# all but its inviscid part extrapolated from the tile's interior; with the
# artificial properties off, the molecular ghost values come from the
# gradient ring and the one-sided divergence rows (`div_plans`) take any
# remainder. Under `:closure` those rows close the whole flux. A
# pentadiagonal scheme, the C10 derivative or a banded filter, takes two such
# rows per end (kernels_banded.jl). The `:d8` detector closes there with rows
# of its own that read the imposed ghosts (`_ring_interface_rows`).
#
# Distribution: each level is owned by a rank subset of its parent's, a
# contiguous prefix of the parent level's communicator (`LevelComm`), and
# within it every tile is owned by its own contiguous rank range
# (`_tile_owners`, `TileGroup`): the tiles are ordered along a Morton curve,
# and each tile of a tiled level is held whole by one rank, the k-th tile on
# the curve by rank k − 1 when the ranks outnumber the tiles and the curve
# cut into runs of about equal weight otherwise. The one patch of a box level
# (`tile = 0`) is instead decomposed over as many ranks as it admits under
# the 9-point scheme minimum. A rank belongs to one group, the
# ranks sharing its range, and holds exactly the tiles of that group, so
# `Solver.patches` is the root followed by this rank's tiles of each level;
# `Level.tiles` records which tile each of those is. When every tile spans
# the whole level, the case of every uniform run, every serial run, and every
# one-tile level, no communicator is split beyond the level's and no code
# path changes. Level-local collectives run on the tile's own communicator
# (the compact line solves and halo exchanges beneath it, the shell ring of
# `_impose_shell!`) or, for the same-level records, point-to-point on the
# level's. The rate reduction is a single Allreduce over the whole run at the
# step boundary: each rank reduces the maximum over the patches it holds, and
# the maximum is exact and order-independent, so the grouping is immaterial.
# Cross-level coupling moves point to point over the parent level's
# communicator, which contains the child's (`LevelCoupling`; the section
# "Point-to-point level coupling" in levels_coupling.jl). The parent ranks holding a tile's
# buffered box send their part of it to the tile's owners
# (`_exchange_boxes!`), each owner receiving the components whose chains it
# runs, and the tile's owners send the coincident samples of `:inject`
# restriction to the parent ranks owning the covered nodes
# (`_exchange_restriction!`). Both sides derive the same pieces from two
# block tables Allgathered once per level build (`build_level_coupling`).
# Every rank of the parent's subset enters both exchanges and posts messages
# only to the peers of its own pieces, so a rank's coupling traffic and memory
# follow the tiles it holds and the parent blocks beneath them.
#
# The interpolation chains distribute by conserved component over a tile's
# ranks, each rank running the serial chain (COMM_SELF Decomps) for its own
# components, and the thin shell ring is shared over the tile's communicator;
# see the section comment above `_impose_shell!` (levels_transfer.jl) for the measurement that
# forced that split. No halo machinery enters the transfer, and the serial
# path is the same code with one-rank communicators. `gather_region!`, the
# replicated gather, remains for the regrid carries.
# `level_restriction = :filter` remains serial-only, since its restriction is
# a whole-patch line solve.

"Coarse nodes of prolongation buffer beyond the refined region, per side."
const LEVEL_BUFFER = 4

# Region nodes per side of a parent-fed face that the restriction does not
# write back (`_restrict_window`, which `build_level_coupling` reads for the
# injection pieces and `_write_covered_region!` for `:filter`). A face shared
# with a same-level tile, or one carrying the root's own condition, is written
# to the face. Under `level_restriction = :filter` the margin breaks an
# amplifying loop: the restriction's filter closes at the ends of the fine
# patch with rows that read the boundary plane the shell imposes from the
# parent, and the coarse nodes written from them feed the next shell. At a
# margin of 0 the error of the two-level entropy wave doubles every step, and
# the wave and the subcycled Sod crossing fail within twenty steps; at a
# margin of 1 the error is flat. Under injection no margin from 0 to 3 shows a
# loop. A margin of 1 lowers the wave's error and the Sod crossing's noise but
# raises its mass drift, and a margin of 0 raises that drift by one to three
# orders, so the margin is 2. The parent nodes the derivative mask may take
# (`_fill_child_deep!`) and the overwritten nodes (`_fill_overwritten!`) are
# formed from the covered mask, not from this constant: the first set starts at
# the first node a margin of 2 writes, the second `LEVEL_BUFFER + 1` nodes
# inside the face.
const RESTRICT_MARGIN = 2

# --- Level ownership ----------------------------------------------------------

"""
    LevelComm

The rank subset owning one level of the hierarchy: the communicator spanning
it, whether the calling rank belongs to it, whether that communicator was
split for this level (and is freed when the level is dropped), and its size.
A level whose subset is its parent's whole set holds the parent's
communicator itself, so no split exists in an unrefined run, a serial run,
or a refined run large enough for every rank.

The subset is a contiguous prefix of the parent level's communicator, which
makes the subsets nested: a rank outside level ℓ is outside every level below
it. Within the subset each tile is owned by its own rank range, a
[`TileGroup`](@ref); the subset's size is the union of those ranges
([`_tile_owners`](@ref)).
"""
struct LevelComm
    comm::MPI.Comm      # MPI_COMM_NULL on a rank outside the subset
    owned::Bool
    scoped::Bool        # split for this level; freed by `free_level_comm!`
    size::Int
end

"The root level's ownership: every rank of `comm`, unsplit."
root_level_comm(comm::MPI.Comm) =
    LevelComm(comm, true, false, MPI.Initialized() ? MPI.Comm_size(comm) : 1)

"The ownership entry for a level this rank holds no part of: not owned, no
communicator, size zero. A rank outside the parent's subset carries it so
that `solver.levels` has the same length everywhere."
absent_level_comm() = LevelComm(MPI.COMM_NULL, false, false, 0)

"""
    split_level_comm(parent, np) -> LevelComm

The [`LevelComm`](@ref) of a level owned by the first `np` ranks of `parent`.
Returns `parent` unsplit when `np` is its whole size, so the common case adds
no communicator, and `absent_level_comm()` when `np` is zero, the
ownership of a level with no tiles. Collective over `parent.comm` otherwise;
a rank outside `parent` must not call it, and every rank inside must pass the
same `np`.
"""
function split_level_comm(parent::LevelComm, np::Int)
    np == 0 && return absent_level_comm()
    np == parent.size && return LevelComm(parent.comm, true, false, np)
    key = MPI.Comm_rank(parent.comm)
    inside = key < np
    sub = MPI.Comm_split(parent.comm, inside ? 0 : nothing, key)
    return LevelComm(sub, inside, true, np)
end

"""
    free_level_comm!(level_comm)

Free a level communicator that [`split_level_comm`](@ref) created, for the
reason `free_communicators!` records: MPI frees nothing until garbage
collection finalizes the handle, and a regrid that resizes a level's subset
discards one per call. A no-op on an unsplit subset and on a rank outside it.
`MPI_Comm_free` is collective, so every owner must reach this together; in
the regrid paths that holds because the tile set is reduced before any rank
rebuilds.
"""
free_level_comm!(lc::LevelComm) =
    (lc.scoped && lc.owned && MPI.free(lc.comm); nothing)

# --- Tile ownership -----------------------------------------------------------

"""
    TileGroup

This rank's tile group on one level: the communicator over the group's
ranks, on which every tile of the group is decomposed, the group's rank
range in the level's communicator, and whether the communicator was split
for the group (and is freed with it, `free_tile_group!`). A rank
holding no tile of the level carries a null group.
"""
struct TileGroup
    comm::MPI.Comm
    ranks::UnitRange{Int}
    scoped::Bool
end

"The group of a rank with no tile of the level."
absent_tile_group() = TileGroup(MPI.COMM_NULL, 0:-1, false)

# Morton (Z-order) key of a lattice point: the bits of the three coordinates
# interleaved, so that points close on the curve are close in space. Twenty-one
# bits per coordinate cover any node offset a grid here can hold.
function _morton(c::NTuple{3,Int})
    key = 0
    for b in 0:20, i in 1:3
        key |= ((c[i] >> b) & 1) << (3 * b + i - 1)
    end
    return key
end

# The tiles of a level in space-filling-curve order: a permutation of
# `eachindex(regions)` by the Morton key of each region's offset, so that a
# contiguous run of it is a compact set of tiles. The tiles keep their lattice
# (raster) order for everything else, whose contiguous runs are strips.
_sfc_order(regions::Vector{BlockRegion}) = sortperm([_morton(r.offset) for r in regions])

"""
    _rank_counts(weights, caps, np; admits) -> Vector{Int}

Ranks per tile when the ranks are at least as many as the tiles: proportional
to `weights` by largest remainder, at least one each and at most `caps[t]`,
the largest count tile `t` admits under the scheme minimum, summing to `np`
unless no tile can move to its next admitted count without passing its cap
or the rank total, when the remaining ranks hold nothing on the level. A
count below the cap need not be admitted itself (a 25 × 25 fine tile takes
two ranks or four, never three), so every count is one for which
`admits(t, count)` holds, and a tile grows from one admitted count to the
next. Deterministic, so every rank computes the same
partition without communication.
"""
function _rank_counts(weights::AbstractVector{<:Real}, caps::Vector{Int}, np::Int;
                      admits=(t, count) -> true)
    n = length(weights)
    np >= n || error("$np ranks cannot each take a tile of $n")
    total = sum(weights)
    share = [np * w / total for w in weights]
    # One rank always fits, so the step down to an admitted count ends.
    admitted_below(t, count) = something(findlast(c -> admits(t, c), 1:count), 1)
    counts = [admitted_below(t, clamp(floor(Int, share[t]), 1, caps[t])) for t in 1:n]
    # The floor of a share below one rounds up to the one rank a tile must
    # have, which can overshoot; the largest counts give the excess back.
    while sum(counts) > np
        t = argmax(counts)
        counts[t] = admitted_below(t, counts[t] - 1)
    end
    while sum(counts) < np
        best = 0
        bestgap = -Inf
        bestnext = 0
        for t in 1:n
            next = findfirst(c -> admits(t, c), (counts[t] + 1):caps[t])
            next === nothing && continue
            next += counts[t]
            sum(counts) - counts[t] + next <= np || continue
            gap = share[t] - counts[t]
            gap > bestgap && (best = t; bestgap = gap; bestnext = next)
        end
        best == 0 && break
        counts[best] = bestnext
    end
    return counts
end

"""
    _tile_owners(regions, active, np; weights, whole=true) -> (owners, np_level)

The owner rank range of every tile of a level, as 0-based ranges in the
level's communicator, and the level's rank count, from the tile geometry
alone (parent-level node space), so every rank computes the same answer.
Each tile of a tiled level is held whole by one rank: with at least as many
ranks as tiles, tile `k` along the space-filling curve takes rank `k − 1` and
the ranks past the last tile hold nothing on the level; with more tiles than
ranks the curve is cut into `np` runs of about equal weight. Under
`whole = false`, the one patch of a box level (`tile = 0`), each tile
instead takes its own contiguous range when the ranks are at least as many
(`_rank_counts`), laid out along the curve. Ranges never straddle: a rank
belongs to one group, the tiles sharing its range, and one communicator per
group serves all of them. The weight is the tile's fine volume unless
`weights` supplies one per tile, as a rebalance does from the per-rank step
wall it measured (`_measured_weights`); a caller passing weights must pass
the same on every rank. A box level reduces to `_level_ranks`.
"""
function _tile_owners(regions::Vector{BlockRegion}, active::NTuple{3,Bool},
                      np::Int;
                      weights::AbstractVector{<:Real}=
                          [prod(fine_extent(r, active)) for r in regions],
                      whole::Bool=true)
    n = length(regions)
    length(weights) == n || error("one weight per tile: $(length(weights)) for $n")
    order = _sfc_order(regions)
    owners = Vector{UnitRange{Int}}(undef, n)
    if np >= n && whole
        # A tile split over several ranks pays a reduced-system exchange in
        # every line solve along a split dimension and an exchange per shell
        # fill, latency that a tile of a few tens of thousands of nodes does
        # not amortize: on a refined Richtmyer–Meshkov level of four tiles
        # over eight ranks every phase, the right-hand side included, spent
        # about half its time in MPI. A rank holding no tile idles on the
        # level instead.
        for (k, t) in enumerate(order)
            owners[t] = (k - 1):(k - 1)
        end
        return owners, n
    end
    if np >= n
        # A lattice level's tiles share a handful of extents, and the cap
        # depends on the extent alone.
        cap = Dict{NTuple{3,Int},Int}()
        caps = [get!(() -> _level_ranks([r], active, np), cap, r.extent)
                for r in regions]
        seen = Dict{Tuple{NTuple{3,Int},Int},Bool}()
        admits(t, count) = get!(() -> _admits(regions[t], active, count), seen,
                                (regions[t].extent, count))
        counts = _rank_counts(weights, caps, np; admits)
        at = 0
        for t in order
            owners[t] = at:(at + counts[t] - 1)
            at += counts[t]
        end
        return owners, at
    end
    total = sum(weights)
    before = 0
    for t in order
        mid = before + weights[t] / 2
        r = min(floor(Int, np * mid / total), np - 1)
        owners[t] = r:r
        before += weights[t]
    end
    return owners, np
end

"""
    _place_tiles(regions, active, np, old_regions, old_owners) -> (owners, np_level)

The owner ranges of a regridded level under stored ownership: a tile whose
region survives from `old_regions` keeps the range `old_owners` recorded for
it, and a fresh tile is placed by the rule [`_tile_owners`](@ref) applies to
a whole level, restricted to the ranks the survivors leave free. The free
ranks are dealt to the fresh tiles as if they were contiguous, and a range
that would straddle a gap between free ranks is cut at the gap, so a group
stays a contiguous rank range, and down to a count the tile admits. When
the fresh tiles outnumber the free ranks, each joins an existing group or
takes a free rank alone, whichever leaves the least fine volume per rank
(`_join_groups!`), and the level is partitioned afresh when no group admits
the tile. With no survivor the level is partitioned afresh. The level's rank
count is one past the highest rank any range uses, so a rank inside the
level may hold no tile of it after a departure. Every input is identical on
every rank (the wanted set is reduced, and the previous owners are held by
every rank of the parent's subset), so the answer is too.
"""
function _place_tiles(regions::Vector{BlockRegion}, active::NTuple{3,Bool},
                      np::Int, old_regions::Vector{BlockRegion},
                      old_owners::Vector{UnitRange{Int}})
    n = length(regions)
    old_of = Dict(r => i for (i, r) in enumerate(old_regions))
    owners = Vector{UnitRange{Int}}(undef, n)
    fresh = Int[]
    survivors = Int[]
    for (t, r) in enumerate(regions)
        if haskey(old_of, r)
            owners[t] = old_owners[old_of[r]]
            push!(survivors, t)
        else
            push!(fresh, t)
        end
    end
    isempty(survivors) && return _tile_owners(regions, active, np)
    if !isempty(fresh)
        taken = falses(np)
        for t in survivors, r in owners[t]
            taken[r + 1] = true
        end
        free = [r for r in 0:(np - 1) if !taken[r + 1]]
        if length(free) < length(fresh)
            _join_groups!(owners, regions, active, survivors, fresh, free) ||
                return _tile_owners(regions, active, np)
        else
            virtual, _ = _tile_owners(regions[fresh], active, length(free))
            for (k, t) in enumerate(fresh)
                v = virtual[k]
                lo = free[first(v) + 1]
                hi = lo
                for i in (first(v) + 1):last(v)
                    free[i + 1] == hi + 1 || break
                    hi += 1
                end
                # A cut range can fall on a count the tile does not admit.
                while hi > lo && !_admits(regions[t], active, hi - lo + 1)
                    hi -= 1
                end
                owners[t] = lo:hi
            end
        end
    end
    return owners, maximum(last, owners) + 1
end

# Fine volume of a tile, the per-rank cost `_tile_owners` weighs by default.
_tile_volume(r::BlockRegion, active::NTuple{3,Bool}) = Float64(prod(fine_extent(r, active)))

# Place the `fresh` tiles one at a time, largest first, each in the group
# that carries the least fine volume per rank once it holds the tile: a group
# of `survivors` or a single rank of `free`. Equal loads go to the group
# holding the tile nearest on the space-filling curve, so a front entering
# a balanced level stays together where it can. A fresh tile joins only a
# group whose rank count admits it, since a group's ranks decompose each of
# its tiles (a tile clipped at the margin can admit fewer ranks than a
# lattice cell). Fresh tiles cluster at the moving features, so a rule of
# proximity alone would stack a front onto one group while ranks vacated by
# departures sit idle. Writes `owners[fresh]`; false when some tile fits no
# group, for the caller to partition afresh.
function _join_groups!(owners::Vector{UnitRange{Int}}, regions::Vector{BlockRegion},
                       active::NTuple{3,Bool}, survivors::Vector{Int},
                       fresh::Vector{Int}, free::Vector{Int})
    keys = [_morton(r.offset) for r in regions]
    groups = unique([owners[s] for s in survivors])
    append!(groups, [r:r for r in free])
    index = Dict(g => i for (i, g) in enumerate(groups))
    load = zeros(length(groups))
    members = [Int[] for _ in groups]
    for s in survivors
        i = index[owners[s]]
        load[i] += _tile_volume(regions[s], active) / length(owners[s])
        push!(members[i], s)
    end
    # The answer depends on the extent and the count alone, and a level holds
    # a handful of each.
    seen = Dict{Tuple{NTuple{3,Int},Int},Bool}()
    admits(r, g) = get!(() -> _admits(r, active, g), seen, (r.extent, g))
    order = sort(fresh; by=t -> (-_tile_volume(regions[t], active), keys[t]))
    # The groups' rank counts, a handful of distinct ones, and each group's.
    counts = unique(length.(groups))
    count_of = [findfirst(==(length(g)), counts) for g in groups]
    for t in order
        vol = _tile_volume(regions[t], active)
        best = 0
        bestkey = (Inf, typemax(Int))
        fits = [admits(regions[t], c) for c in counts]
        # Only a group at the least resulting load can take the tile, so the
        # distance along the curve, a scan of the group's tiles, is taken for
        # those.
        least = Inf
        for (i, g) in enumerate(groups)
            fits[count_of[i]] && (least = min(least, load[i] + vol / length(g)))
        end
        for (i, g) in enumerate(groups)
            load[i] + vol / length(g) == least || continue
            fits[count_of[i]] || continue
            near = isempty(members[i]) ? typemax(Int) :
                   minimum(abs(keys[t] - keys[m]) for m in members[i])
            key = (load[i] + vol / length(g), near)
            key < bestkey && (best = i; bestkey = key)
        end
        best == 0 && return false
        owners[t] = groups[best]
        load[best] = bestkey[1]
        push!(members[best], t)
    end
    return true
end

# The fine volume of the most loaded rank among `np` under `owners`, each
# tile's volume divided evenly over its range.
function _peak_load(regions::Vector{BlockRegion}, active::NTuple{3,Bool},
                    owners::Vector{UnitRange{Int}}, np::Int)
    load = zeros(np)
    for (r, rg) in zip(regions, owners), k in rg
        load[k + 1] += _tile_volume(r, active) / length(rg)
    end
    return maximum(load; init=0.0)
end

# Whole-tile owners of a level with more tiles than ranks, one rank per tile:
# largest tile first onto the least loaded rank (among equals the rank
# holding the tile nearest on the space-filling curve, then the lowest), then
# moves of one tile, or swaps of two, between the most and the least loaded
# rank while they lower the larger of the two loads. With a few tiles per
# rank, cutting the curve into runs of equal weight (`_tile_owners`) can leave
# the heaviest rank up to a tile above the mean, where an assignment free of
# the curve's order pairs large tiles with small ones. The cost is locality:
# tiles adjacent in space land on different ranks, and their shared planes
# are exchanged between ranks rather than within one.
#
# Across nodes (`nodes`, the node of each rank, `_node_map`) that cost is an
# off-node message for every shared plane, so a run spanning several nodes
# first cuts the curve into one run per node, of weight in proportion to the
# node's ranks, and places each run on its own node's ranks by the
# assignment above: the planes the scattering exposes stay within a node,
# and only the planes between runs leave it. A node's run is balanced to
# within a tile over its ranks, so the cut costs the balance half a tile over
# the node's ranks at most. With every rank on one node, or `nodes` empty,
# the level is placed as a whole.
function _knapsack_owners(regions::Vector{BlockRegion}, active::NTuple{3,Bool},
                          np::Int; nodes::AbstractVector{Int}=Int[])
    _one_node(nodes, np) && return _flat_knapsack(regions, active, np)
    order, ranks = _node_ranks(nodes, np)
    vol = [_tile_volume(r, active) for r in regions]
    total = sum(vol)
    # The cumulative weight at the end of each node's run.
    ends = cumsum([length(ranks[g]) for g in order]) .* (total / np)
    runs = [Int[] for _ in order]
    before = 0.0
    j = 1
    for t in _sfc_order(regions)
        mid = before + vol[t] / 2
        while j < length(order) && mid > ends[j]
            j += 1
        end
        push!(runs[j], t)
        before += vol[t]
    end
    owners = Vector{UnitRange{Int}}(undef, length(regions))
    for (j, g) in enumerate(order)
        isempty(runs[j]) && continue
        local_owners, _ = _flat_knapsack(regions[runs[j]], active, length(ranks[g]))
        for (i, t) in enumerate(runs[j])
            r = ranks[g][first(local_owners[i]) + 1]
            owners[t] = r:r
        end
    end
    return owners, np
end

# Whether the first `np` ranks share one node under `nodes` (empty: one node).
_one_node(nodes::AbstractVector{Int}, np::Int) =
    isempty(nodes) || length(nodes) < np || all(==(nodes[1]), view(nodes, 1:np))

# The nodes of the first `np` ranks in the order of their lowest rank, and
# the ranks (0-based) of each.
function _node_ranks(nodes::AbstractVector{Int}, np::Int)
    order = Int[]
    ranks = Dict{Int,Vector{Int}}()
    for r in 0:(np - 1)
        g = nodes[r + 1]
        haskey(ranks, g) || push!(order, g)
        push!(get!(Vector{Int}, ranks, g), r)
    end
    return order, ranks
end

function _flat_knapsack(regions::Vector{BlockRegion}, active::NTuple{3,Bool}, np::Int)
    n = length(regions)
    keys = [_morton(r.offset) for r in regions]
    vol = [_tile_volume(r, active) for r in regions]
    load = zeros(np)
    held = [Int[] for _ in 1:np]
    for t in sort(collect(1:n); by=t -> (-vol[t], keys[t]))
        best = 0
        bestkey = (Inf, typemax(Int), 0)
        # Only a rank at the least load can take the tile, so the distance
        # along the curve, a scan of the rank's tiles, is taken for those.
        least = minimum(load)
        for r in 1:np
            load[r] == least || continue
            near = isempty(held[r]) ? typemax(Int) :
                   minimum(abs(keys[t] - keys[m]) for m in held[r])
            key = (load[r], near, r)
            key < bestkey && (best = r; bestkey = key)
        end
        load[best] += vol[t]
        push!(held[best], t)
    end
    for _ in 1:n * np
        h, l = argmax(load), argmin(load)
        best = (load[h], 0, 0)
        for t in held[h]
            moved = max(load[h] - vol[t], load[l] + vol[t])
            moved < best[1] && (best = (moved, t, 0))
            for u in held[l]
                vol[u] < vol[t] || continue
                swapped = max(load[h] - vol[t] + vol[u], load[l] + vol[t] - vol[u])
                swapped < best[1] && (best = (swapped, t, u))
            end
        end
        _, t, u = best
        t == 0 && break
        filter!(!=(t), held[h])
        push!(held[l], t)
        load[h] -= vol[t]
        load[l] += vol[t]
        if u != 0
            filter!(!=(u), held[l])
            push!(held[h], u)
            load[l] -= vol[u]
            load[h] += vol[u]
        end
    end
    owners = Vector{UnitRange{Int}}(undef, n)
    for r in 1:np, t in held[r]
        owners[t] = (r - 1):(r - 1)
    end
    return owners, np
end

# The ranks of a whole-tile partition `fresh` renumbered to keep surviving
# tiles where they are: the knapsack's ranks are interchangeable, so each of
# its rank sets goes to the rank already holding the largest volume of its
# surviving tiles under `owners` (`survived` marks them), greedily from the
# largest such volume, and the sets holding no survivor take the ranks left
# in order. Every rank's load is unchanged, and fewer tiles move. Across
# nodes (`nodes`) a rank set is renumbered within its node only, since the
# placement of the runs on the nodes is not interchangeable. The pairs of
# rank sets and ranks are held sparsely: at a cluster's rank count a table of
# every pair would be the size of the rank count squared.
function _keep_survivors(fresh::Vector{UnitRange{Int}}, regions::Vector{BlockRegion},
                         active::NTuple{3,Bool}, np::Int,
                         owners::Vector{UnitRange{Int}}, survived::AbstractVector{Bool};
                         nodes::AbstractVector{Int}=Int[])
    one = _one_node(nodes, np)
    node(r) = one ? 0 : nodes[r]
    held = Dict{Tuple{Int,Int},Float64}()   # (fresh rank, current rank) => volume
    for t in eachindex(regions)
        survived[t] || continue
        for r in owners[t]
            a, b = first(fresh[t]) + 1, r + 1
            # A current rank past the level, or on another node, keeps nothing.
            (b <= np && node(a) == node(b)) || continue
            held[(a, b)] = get(held, (a, b), 0.0) +
                           _tile_volume(regions[t], active) / length(owners[t])
        end
    end
    pairs = [k for (k, v) in held if v > 0]
    sort!(pairs; by=((a, b),) -> (-held[(a, b)], a, b))
    to = zeros(Int, np)
    taken = falses(np)
    for (a, b) in pairs
        (to[a] == 0 && !taken[b]) || continue
        to[a] = b
        taken[b] = true
    end
    rest = Dict{Int,Vector{Int}}()
    for b in 1:np
        taken[b] || push!(get!(Vector{Int}, rest, node(b)), b)
    end
    for a in 1:np
        to[a] == 0 && (to[a] = popfirst!(rest[node(a)]))
    end
    return [(to[first(rg) + 1] - 1):(to[first(rg) + 1] - 1) for rg in fresh]
end

# A level's partition from scratch by fine volume: whole tiles by
# `_knapsack_owners` when the tiles outnumber the ranks, its ranks numbered
# to keep the survivors of `owners` in place, and each tile on a rank range
# of its own by `_tile_owners` otherwise.
function _fresh_partition(regions::Vector{BlockRegion}, active::NTuple{3,Bool}, np::Int,
                          owners::Vector{UnitRange{Int}}, survived::AbstractVector{Bool};
                          nodes::AbstractVector{Int}=Int[])
    length(regions) > np || return _tile_owners(regions, active, np)
    fresh, np_fresh = _knapsack_owners(regions, active, np; nodes)
    return _keep_survivors(fresh, regions, active, np, owners, survived; nodes),
           np_fresh
end

# The gain in the most loaded rank's fine volume, stored placement over a
# fresh partition, at which a regrid adopts the fresh partition
# (`_balanced_owners`). Under subcycling each level advances in turn and its
# slowest rank sets the pace of its substeps, so the excess over the fresh
# partition is wall time lost on every rank. Adopting it moves surviving
# tiles, at the cost of rebuilding them and carrying their state, and a tile
# decomposed over several ranks then computes to round-off on its new
# decomposition rather than bit for bit. A tenth is the threshold WarpX
# applies to the same decision; a smaller predicted gain is comparable to the
# cost differences between tiles of equal volume (the share of interface and
# boundary work) that the volume model leaves out.
const PLACEMENT_GAIN = 1.1

"""
    _balanced_owners(regions, active, np, owners, np_level, survived) -> (owners, np_level)

The placement `owners` (from `_place_tiles`) unless the level's most loaded
rank carries more than `PLACEMENT_GAIN` times the fine volume it would under
a fresh partition (`_fresh_partition`), in which case the fresh partition.
The comparison is with the partition the tiles admit rather than with the
mean, so a level whose tiles cannot be spread evenly (fewer tiles than ranks
at the scheme minimum, or a few tiles per rank) is not repartitioned at
every regrid for an imbalance no partition removes. Depends on the reduced
tile set and the stored owners alone, so every rank takes the same decision
without a measurement. `survived` marks the tiles whose region and owners
survive from the last layout, which a whole-tile partition keeps in place
where its loads allow. `nodes` is the node of each rank (`_node_map`), which
the fresh partition keeps a node's tiles together by.
"""
function _balanced_owners(regions::Vector{BlockRegion}, active::NTuple{3,Bool},
                          np::Int, owners::Vector{UnitRange{Int}}, np_level::Int,
                          survived::AbstractVector{Bool}=falses(length(regions));
                          nodes::AbstractVector{Int}=Int[])
    isempty(regions) && return owners, np_level
    fresh, np_fresh = _fresh_partition(regions, active, np, owners, survived; nodes)
    _peak_load(regions, active, owners, np) >
        PLACEMENT_GAIN * _peak_load(regions, active, fresh, np) &&
        return fresh, np_fresh
    return owners, np_level
end

"""
    _measured_weights(regions, active, old_regions, old_owners, busy) -> Vector{Float64}

Per-tile weights for a rebalance, from `busy`, each rank's measured step
wall over the last regrid interval less its time inside the run-wide
collectives, indexed by rank of the level's communicator. A tile of the old
level costs the busy time of its owner ranks, summed, times its share by
fine volume of the tiles those ranks held (a group's tiles all share its
range); a tile in `regions` that survives from `old_regions` takes that
cost, and a fresh one takes its volume at the mean measured cost per fine
node. The interface factor the design anticipates, small tiles costing more
than their volume, is therefore measured by the run itself rather than
calibrated. Falls back to the volumes when nothing was measured. `busy` must
be the same vector on every rank.
"""
function _measured_weights(regions::Vector{BlockRegion}, active::NTuple{3,Bool},
                           old_regions::Vector{BlockRegion},
                           old_owners::Vector{UnitRange{Int}},
                           busy::Vector{Float64})
    old_vol = [Float64(prod(fine_extent(r, active))) for r in old_regions]
    group_vol = Dict{UnitRange{Int},Float64}()
    for (t, rg) in enumerate(old_owners)
        group_vol[rg] = get(group_vol, rg, 0.0) + old_vol[t]
    end
    cost = [sum(busy[r + 1] for r in old_owners[t]) *
            old_vol[t] / group_vol[old_owners[t]] for t in eachindex(old_regions)]
    vol = [Float64(prod(fine_extent(r, active))) for r in regions]
    total_cost = sum(cost; init=0.0)
    total_cost > 0 || return vol
    per_node = total_cost / sum(old_vol)
    old_of = Dict(r => i for (i, r) in enumerate(old_regions))
    return [haskey(old_of, r) ? cost[old_of[r]] : vol[t] * per_node
            for (t, r) in enumerate(regions)]
end

# The owner range containing rank `me`, or `nothing`.
function _group_of(owners::Vector{UnitRange{Int}}, me::Int)
    for rg in owners
        me in rg && return rg
    end
    return nothing
end

"""
    split_tile_comm(lc, owners) -> TileGroup

This rank's [`TileGroup`](@ref) on a level owned by `lc` with tile owner
ranges `owners`. Returns the level's communicator unsplit when every tile
spans the whole level, so a one-tile level (and every existing configuration)
creates no communicator here. Collective over `lc.comm`; every owner of the
level must call it, and a rank outside must not.
"""
function split_tile_comm(lc::LevelComm, owners::Vector{UnitRange{Int}})
    whole = 0:(lc.size - 1)
    all(==(whole), owners) && return TileGroup(lc.comm, whole, false)
    me = MPI.Comm_rank(lc.comm)
    mine = _group_of(owners, me)
    sub = MPI.Comm_split(lc.comm, mine === nothing ? nothing : first(mine), me)
    return mine === nothing ? absent_tile_group() : TileGroup(sub, mine, true)
end

"""
    free_tile_group!(group)

Free the communicator `split_tile_comm` created, when it did; the
Cartesian communicators of the tiles built on it are independent and outlive
it. Collective over the group's ranks, which all drop the level's tile set at
the same regrid.
"""
free_tile_group!(g::TileGroup) = (g.scoped && MPI.free(g.comm); nothing)

"""
    GatherBuffers{T}()

Send and receive staging for one of the coupling's collectives, resized on
first use and retained for the life of the [`LevelTransfer`](@ref). Every
call site's message sizes are fixed by the region and the rank set, so the
resize is a no-op after the first call.
"""
struct GatherBuffers{T}
    send::Vector{T}
    recv::Vector{T}
end

GatherBuffers{T}() where {T} = GatherBuffers{T}(T[], T[])

@inline _fit!(buf::Vector, n::Int) = (length(buf) == n || resize!(buf, n); buf)

"""
    ShellRing

Geometry and staging of the fine shell ring: the slab ranges, the writers'
`(lo, hi, offset)` table, the ring length, the replicated ring itself, its
multilinear counterpart for the admissible fallback (`_shell_fallback`), the
per-rank Allgatherv counts, the ring's own buffers, and, per stage of the
interpolation chain, the boxes of that stage the shell slots depend on
(`_chain_boxes`). All of it follows from the refined region, the fine
decomposition and the faces the parent feeds, which do not change over a
[`LevelTransfer`](@ref)'s life, so it is built once at setup. A subcycled
step imposes the shell about twenty times, so the geometry is not rebuilt
per call. The slabs are those of the parent-fed faces. `memo` records the
Hermite fraction and parent step whose shell the host ring holds, so that a
second imposition at the same fraction writes it again
(`_shell_memo_hit`).
"""
struct ShellRing{T}
    slabs::Vector{NTuple{3,UnitRange{Int}}}
    table::Vector{Tuple{NTuple{3,Int},NTuple{3,Int},Int}}
    len::Int
    ring::Matrix{T}
    linear::Matrix{T}
    counts::Vector{Int}
    buffers::GatherBuffers{T}
    boxes::Vector{Vector{NTuple{3,UnitRange{Int}}}} # per chain stage 1 .. K
    memo::Vector{Float64}            # held, θ, dt of the ring's Hermite fill
end

"""
    ShellGradients

The conserved-variable gradients a refined patch's ghost fluxes read at its
coarse-fine faces under `interface_flux = :ghost` with molecular transport:
the compact derivative plans along each active dimension over the fine box
the interpolation chain produces (`nothing` elsewhere), a box-sized scratch
field, and the gradient ring, one column per conserved component and
dimension (`3(c − 1) + j` holds ∂Q_c/∂x_j), laid out as the shell ring is
and holding data where the ghost fluxes read it, on the ghost layers of the
parent-fed faces.
`_impose_shell!` refreshes the ring with the shell itself, so the two
describe the same interpolated state, on a patch with a coarse-fine face
that reads it (`_reads_gradient_ring`): with the artificial properties off,
or with the remainder of the flux left to the divergence plans. With the
artificial properties on, the default, the extrapolated remainder carries
the molecular flux on those ghost layers, and no transfer is built with the
ring (`_gradient_ring_built`). A device patch holds the backend's
form of the plans, the scratch field and the ring in its `LevelScratch`
instead, and the host fields here go unused on it. `decomps[d]` is the
decomposition the plan along `d` applies over (`_box_gradient_plans`).
"""
struct ShellGradients{T}
    plans::Vector{Any}
    decomps::Vector{Any}
    tmp::Array{T,3}
    gring::Matrix{T}
end

# The host plans of the gradient ring over the fine box `boxf`: `deriv`'s
# box-gradient form along each active dimension at the fine spacing `hf`,
# `nothing` along a collapsed one, and the decompositions they apply over.
#
# The ghost fluxes read the gradient ring only on the ghost layers of the
# faces flagged in `faces` (the parent-fed ones; every face by default),
# over the interior transverse range, which lie within `pad` fine nodes of
# the patch, while the box extends 3·LEVEL_BUFFER beyond it. The plan along
# `d` therefore solves only the lines through the bounding box of those
# nodes: full length along `d`, trimmed transversally. Each line is filled
# and solved on its own, so the values read are the full box's bit for bit;
# the rest of the ring's gradient columns carry no data. On a 37-node patch
# with every face read, the full box holds a third more lines in 2-D and
# nearly twice as many in 3-D; on a tile with one parent-fed face, the lines
# parallel to that face reduce to its ghost layers. A plan's decomposition is
# the trimmed box with its transverse halo pads widened by the trim, which
# places line (j, k) of the plan on the box's line through the same nodes.
# The device scratch keeps every face, since a tile kept at a regrid keeps
# its scratch while its faces change.
function _box_gradient_plans(boxf::Decomp{T}, deriv, active::NTuple{3,Bool}, hf,
                             pad::NTuple{3,Int},
                             faces::NTuple{3,NTuple{2,Bool}}=ntuple(d -> (true, true), 3),
                             buffer=_box_buffer(active, _NO_BOUNDARY),
                             folded::NTuple{3,NTuple{2,Bool}}=_NO_BOUNDARY) where {T}
    shift = _box_shift(buffer, folded)
    beyond = ntuple(e -> 3 * buffer[e][2] - _fold_lead(folded, e, 2), 3)
    all(e -> !faces[e][1] || shift[e] >= pad[e], 1:3) &&
        all(e -> !faces[e][2] || beyond[e] >= pad[e], 1:3) ||
        error("a ring wider than the box buffer: pad $pad")
    Nf = ntuple(e -> boxf.n_global[e] - shift[e] - beyond[e], 3)
    read_faces = [(d, side) for d in 1:3 if active[d] for side in 1:2 if faces[d][side]]
    isempty(read_faces) && (read_faces = [(d, side) for d in 1:3 if active[d] for side in 1:2])
    plans = Any[]
    decomps = Any[]
    for d in 1:3
        if !active[d]
            push!(plans, nothing)
            push!(decomps, nothing)
            continue
        end
        # The patch-index range of the lines along `d` in each transverse
        # dimension: a face's ghost layers along its own dimension, the
        # interior along the others.
        lo = ntuple(e -> e == d || !active[e] ? 1 :
                    minimum(f -> f[1] != e ? 1 : f[2] == 1 ? 1 - pad[e] : Nf[e] + 1,
                            read_faces), 3)
        hi = ntuple(e -> e == d || !active[e] ? Nf[e] :
                    maximum(f -> f[1] != e ? Nf[e] : f[2] == 1 ? 0 : Nf[e] + pad[e],
                            read_faces), 3)
        ext = ntuple(e -> e == d ? boxf.n_global[e] : hi[e] - lo[e] + 1, 3)
        dc = Decomp{T}(ext, boxf.periodic; dims=(1, 1, 1), n_halo=boxf.n_halo,
                       comm=MPI.COMM_SELF)
        dc.active == boxf.active || error("a trimmed gradient box lost a dimension")
        push!(plans, plan_direction(dc, _box_gradient_scheme(deriv), d, T(hf[d])))
        halo = ntuple(e -> dc.n_halo_d[e] + (e == d ? 0 : lo[e] + shift[e] - 1), 3)
        push!(decomps, Decomp{T}(dc.comm, dc.dims, dc.coords, dc.periodic, dc.n_global,
                                 dc.n_local, dc.offset, dc.n_halo, dc.active, halo,
                                 dc.neighbors, dc.sub, dc.sub_rank, dc.sub_size,
                                 dc.owns_communicators, dc.send_buf, dc.recv_buf,
                                 dc.line_buffers))
    end
    return plans, decomps
end

# The derivative operator the gradient ring is taken with: `deriv`'s interior
# with explicit one-sided rows of seventh order on eight points at the box
# ends, one per closure row the interior needs. The ghost layers sit eight to
# eleven fine nodes inside the box, where a closure row's error has decayed
# by the interior rows' factor per node (2 − √3 for C6) but not vanished:
# under the third-order rows of the operator's default set it entered the
# molecular divergence at second order in the spacing, a floor of 5e-11 on
# the viscous standing wave at N = 192. A derivative taken once, and never
# stepped, has no stability constraint on its closure rows.
function _box_gradient_scheme(deriv::CompactScheme{T}) where {T}
    rows = [ClosureRow{T}((zero(T), one(T), zero(T)), _one_sided_d1(T, r, 8))
            for r in 1:nclosure(deriv)]
    return CompactScheme{T}(deriv.name * ", box gradient", deriv.alpha, deriv.a0,
                            deriv.coeffs, deriv.symmetric, rows)
end
function _box_gradient_scheme(deriv::BandedCompactScheme{T}) where {T}
    rows = [ClosureRow{T}((zero(T), one(T), zero(T)), _one_sided_d1(T, r, 8))
            for r in 1:nclosure(deriv)]
    return BandedCompactScheme{T}(deriv.name * ", box gradient", deriv.q, deriv.lhs,
                                  deriv.a0, deriv.coeffs, deriv.symmetric,
                                  _banded_closure_rows(rows))
end
_box_gradient_scheme(deriv) = deriv

# The first-derivative weights at node `r` of the nodes 1..npts, from the
# Lagrange interpolant through them, exact rationals, undivided.
function _one_sided_d1(::Type{T}, r::Int, npts::Int) where {T}
    x = collect(1:npts) .// 1
    w = zeros(Rational{BigInt}, npts)
    for j in 1:npts
        # d/dx of the j-th Lagrange basis polynomial at x[r].
        s = 0 // 1
        for m in 1:npts
            m == j && continue
            p = 1 // (x[j] - x[m])
            for q in 1:npts
                (q == j || q == m) && continue
                p *= (x[r] - x[q]) // (x[j] - x[q])
            end
            s += p
        end
        w[j] = s
    end
    return T.(w)
end

"""
    LevelTransfer

Bound form of the coupling between one refined patch and its parent: the
refined region (in the parent patch's node space), the prolongation chain
over the buffered box, and the restriction chain over the fine patch's
extent, each a sequence of [`TransferPlan`](@ref)s refining one active
dimension at a time with per-stage scratch. Constructed at setup by the
[`Solver`](@ref) constructor's `refine` keyword, one per refined patch, and
held on the patch's [`Level`](@ref); consumed by `prolong_level_ghosts!`
and `restrict_level!`. Every rank of the parent level's subset holds one per
tile of the level; the chains and the box storage exist on the tile's own
ranks only.
"""
struct LevelTransfer{T}
    region::BlockRegion              # refined region, parent-level node space
    active::NTuple{3,Bool}           # resolved dimensions of the refined patch;
                                     # held here rather than read off the fine
                                     # decomposition, which a parent-only rank
                                     # does not have
    coarse_regions::Vector{BlockRegion} # parent-level patches meeting the
                                     # buffered box, in that level's node space
    coarse_local::Vector{Int}        # their `solver.patches` indices on this
                                     # rank, 0 where this rank holds no piece
    fine_index::Int                  # index of the refined patch in
                                     # solver.patches, 0 on a rank holding no
                                     # piece of it
    parent_comm::MPI.Comm            # the parent level's communicator, on
                                     # which the box and restriction gathers run
    imposed::NTuple{3,NTuple{2,Bool}} # per face: shell imposed from the parent
                                     # (false where a same-level neighbor
                                     # supplies the face through the records,
                                     # or where the face lies on the domain
                                     # boundary)
    boundary::NTuple{3,NTuple{2,Bool}} # per face: on a non-periodic domain
                                     # boundary, carrying the root's condition
                                     # there; the box takes no buffer beyond it
                                     # unless the face is folded
    folded::NTuple{3,NTuple{2,Bool}} # per face: a boundary face carrying a
                                     # fold (a symmetry plane or the r-z
                                     # axis), where the tile
                                     # takes one fine node beyond the
                                     # coincident lattice and the box its
                                     # mirror image (`_fold_lead`)
    period::NTuple{3,Int}            # the parent level's node-space period
                                     # along a periodic dimension, 0 along
                                     # the others (`_level_period`): a box
                                     # or a covered window across the seam
                                     # meets the parent in a periodic image
    restriction::Symbol             # :inject (coincident-node copy, default)
                                     # or :filter (the invertible transfer pair)
    active_dims::Vector{Int}
    pdecomps::Vector{Decomp{T}}      # prolongation chain, stage 0 (coarse box) .. K
    pplans::Vector{TransferPlan{T}}
    pstage::Vector{Array{T,3}}
    rdecomps::Vector{Decomp{T}}      # restriction chain, stage 0 (region) .. K
    rplans::Vector{TransferPlan{T}}
    rstage::Vector{Array{T,3}}       # scratch for stages 0 .. K-1
    # Subcycling storage: the coarse solution on the buffered box at
    # the two ends of the current coarse step, values and RHS rates, per
    # conserved component: the data of the cubic Hermite interpolant that
    # supplies the fine shell at fine stage times. Shaped like pstage[1] with a
    # trailing component index; empty when the solver does not subcycle.
    box_Q0::Array{T,4}               # coarse box state at t^n
    box_dQ0::Array{T,4}              # coarse box RHS at t^n
    box_Q1::Array{T,4}               # coarse box state at t^n + dt
    box_dQ1::Array{T,4}              # coarse box RHS at t^n + dt
    # Distribution: the chain, the box and the Hermite storage exist on the
    # refined patch's owners only, each of which receives the buffered box
    # of its own components from the parent ranks holding it through the
    # level's point-to-point exchange (`LevelCoupling`); a rank of the
    # parent's subset holding no piece of the patch keeps the light fields
    # above, which its sends and its covered-node receives read, and empty
    # arrays here.
    box_gather::Array{T,4}           # the parent state over the box
    restricted::Array{T,4}           # the filtered samples of `:filter`
                                     # restriction over the region; empty
                                     # under `:inject`
    shell::ShellRing{T}
    gradients::Union{Nothing,ShellGradients{T}}  # the ghost fluxes' gradient
                                     # ring; `nothing` unless the solver
                                     # differences molecular fluxes through
                                     # ghost fluxes (`_ghost_viscous`)
end
