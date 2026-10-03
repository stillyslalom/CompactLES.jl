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
# (`_advance_level!`, timestep.jl), and the temporal interpolation this
# needs is the Hermite box at the end of this file; a two-level region can
# also move under `regrid_interval` (src/regrid.jl).
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
#     constant for the measured amplifying loop the margin breaks).
#     Between restrictions the coarse operator advances the covered nodes,
#     so on a fixed grid the coupling carries a term first order in the step
#     and proportional to the difference between the coarse and fine
#     right-hand sides there, O(dt H^p), which a fixed cfl places above the
#     spatial order. Restricting before every stage of the global step
#     removes the term but moves the error against the exact solution at
#     the default cfl, and on a Sod crossing, by under 2%, at seven
#     restrictions per step instead of one, so the schedule stays per step
#     (bench/restrictcost.jl). Under subcycling the parent's step precedes
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
# freshly covered coarse data are point samples too (regrid.jl).
#
# Multi-dimensional transfer is a tensor product realized as a chain: each
# active dimension is refined in turn, so a chain of K = (number of active
# dimensions) `TransferPlan`s connects the coarse box to the fully fine box
# through K − 1 intermediate grids, each stage carrying its own scratch array.
#
# Scope, enforced by `Solver`: the Cartesian or axisymmetric (θ-collapsed)
# cylindrical metric, no stretching, no fold on a refined level but a symmetry
# plane or the r-z axis where the first level reaches it (see "Levels on the
# domain boundary" below), and no same-level patch decomposition alongside
# refinement. The transfers move the
# conserved variables themselves on either metric; each tile evaluates its own
# geometry at its nodes. The fine patch's
# line solves close at the coarse–fine boundary with the same-level interface
# rows (extended-data gradients and filters, one-sided divergence); a
# pentadiagonal scheme, the C10 derivative or a banded filter, takes two such
# rows per end (kernels_banded.jl). The `:d8` detector closes there with rows
# of its own that read the imposed ghosts (`_ring_interface_rows`).
#
# Distribution: each level is owned by a rank subset of its parent's, a
# contiguous prefix of the parent level's communicator (`LevelComm`), and
# within it every tile is decomposed over its own contiguous rank range
# (`_tile_owners`, `TileGroup`): the tiles are ordered along a Morton curve
# and the ranks dealt out along it by tile volume, at least one and at most
# the count the tile admits under the 9-point scheme minimum per tile when
# the ranks outnumber the tiles, one rank per tile with the curve cut into
# runs of about equal weight otherwise. A rank belongs to one group, the
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
# Cross-level coupling runs on the parent level's communicator, which
# contains the child's: the buffered-box gathers read parent state that a
# child's owner need not hold, and the covered nodes of the restriction
# write-back lie on parent ranks outside the child's subset, so every rank
# owning the parent level enters both, contributing nothing for a parent
# tile it does not hold.
#
# The coupling data are replicated: every rank gathers the buffered coarse box
# (and, for restriction, the coincident-node samples of the fine patch) with
# one Allgatherv and writes only the shell or covered nodes it owns. The
# interpolation chains distribute by conserved component, with each rank
# running the serial chain (COMM_SELF Decomps) for its own components and
# sharing only the thin shell ring; see the section comment above
# `_impose_shell!` for the measurement that forced that split. No halo
# machinery enters the transfer, and the serial path is the same code with
# one-rank communicators. The costs that grow with this choice are the
# gather volume (region-sized messages, several per step) and one replicated
# region-sized array per rank; a rank-partitioned transfer is the recorded
# follow-up if a measured case outgrows them. `level_restriction = :filter`
# remains serial-only, since its restriction is a whole-patch line solve.

"Coarse nodes of prolongation buffer beyond the refined region, per side."
const LEVEL_BUFFER = 4

# Coarse nodes per side of the covered region excluded from the restriction
# write-back. Restricting all the way to the coarse-fine boundary closes an
# amplifying loop: the fine solution is least accurate at its imposed
# boundary, the restriction segment's one-sided closure rows read exactly
# those values, and the polluted coarse boundary nodes feed the next fine
# shell. The measured gain is ≈ 2 per step on the entropy-wave test, against
# a flat error with the write-back held off the boundary. Two coarse nodes clear
# both the closure rows' footprint and the imposed plane's neighborhood.
const RESTRICT_MARGIN = 2

# The CFL number `max_rate` holds a parent's overwritten nodes to where the
# solver's own is smaller (`Patch.overwritten`, `_fill_overwritten!` in
# patches.jl). It stays below the strong-shock ceiling: on the converging
# shock at the r-z axis of bench/amrwin.jl, covered nodes at the axis left
# out of the rate ran away within one parent step at the focus, and held to
# 1.0 they still did, where held to 0.9 or 0.75 they did not.
const OVERWRITTEN_CFL = 0.75

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
_sfc_order(regions::Vector{BlockRegion}) =
    sortperm(regions; by=r -> _morton(r.offset))

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
    _tile_owners(regions, active, np; weights) -> (owners, np_level)

The owner rank range of every tile of a level, as 0-based ranges in the
level's communicator, and the level's rank count, from the tile geometry
alone (parent-level node space), so every rank computes the same answer.
With at least as many ranks as tiles each tile takes its own contiguous range
(`_rank_counts`), the ranges laid out along the space-filling curve so
that neighboring tiles sit on neighboring ranks; with more tiles than ranks
each tile takes one rank, the curve cut into `np` runs of about equal weight.
Ranges never straddle: a rank belongs to one group, the tiles sharing its
range, and one communicator per group serves all of them. The weight is the
tile's fine volume unless `weights` supplies one per tile, as a rebalance
does from the per-rank step wall it measured (`_measured_weights`); a caller
passing weights must pass the same on every rank. A level of one tile
reduces to `_level_ranks`.
"""
function _tile_owners(regions::Vector{BlockRegion}, active::NTuple{3,Bool},
                      np::Int;
                      weights::AbstractVector{<:Real}=
                          [prod(fine_extent(r, active)) for r in regions])
    n = length(regions)
    length(weights) == n || error("one weight per tile: $(length(weights)) for $n")
    order = _sfc_order(regions)
    owners = Vector{UnitRange{Int}}(undef, n)
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
stays a contiguous rank range, and down to a count the tile admits; when the survivors leave no rank free, a
fresh tile joins the group of the survivor nearest it on the space-filling
curve among the groups whose rank count admits the tile, and the level is
partitioned afresh when none does. With no survivor the level is partitioned
afresh. The level's rank
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
        if isempty(free)
            keys = [_morton(r.offset) for r in regions]
            # A group's ranges all decompose each of its tiles, so a fresh
            # tile joins only a group whose rank count admits it (a tile
            # clipped at the margin can admit fewer ranks than a lattice
            # cell); with no such group the level is partitioned afresh. The
            # answer depends on the extent and the count alone, and a level
            # holds a handful of each.
            seen = Dict{Tuple{NTuple{3,Int},Int},Bool}()
            admits(r, g) = get!(() -> _admits(r, active, g), seen, (r.extent, g))
            for t in fresh
                fit = [s for s in survivors if admits(regions[t], length(owners[s]))]
                isempty(fit) && return _tile_owners(regions, active, np)
                nearest = fit[argmin([abs(keys[t] - keys[s]) for s in fit])]
                owners[t] = owners[nearest]
            end
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
per call.
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
describe the same interpolated state. A device patch holds the backend's
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
                                 dc.owns_communicators, dc.send_buf, dc.recv_buf))
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

# --- Point-to-point level coupling --------------------------------------------
#
# The coupling between a level and its parent moves data between two rank
# sets that do not coincide: a tile's buffered box lies on whichever parent
# ranks own those parent nodes, and its covered nodes are written back
# there. A collective over the parent's whole subset per tile would deliver
# every tile's box to every rank, so the per-rank traffic, the memory and
# the number of collectives would all grow with the level's tile count.
# Instead each message goes from the rank that holds the data to the rank
# that needs it: a parent block owner sends the part of each tile's box it
# holds to that tile's owners, each receiving the components its share of
# the interpolation chains runs (`_impose_shell!`), and a tile owner sends
# the coincident samples of its block to the parent ranks owning them. The
# pieces a rank sends to one peer travel as one message, so one exchange
# posts one send and one receive per peer. Both sides derive the same
# pieces, in the same order, from two block tables Allgathered once per
# level build, and nothing but the payload moves at run time. The data moved
# are copies, and each receive applies its pieces in tile order, then parent
# order, so a node that two tiles' covered windows share takes the later
# tile's value at every rank count, as a restriction taken tile by tile
# leaves it.

# Tags of the two exchanges. Each runs to completion before anything else is
# posted on its communicator, and the per-pair message index the same-level
# records use starts above both.
const _BOX_TAG = 1600
const _RESTRICT_TAG = 1700

"""
    OwnedBlock

One entry of a coupling block table: a rank of the parent level's
communicator, its owned interior block of a patch (in the patch's node
space), and its position `share` in that patch's own communicator of size
`shares`, which fixes the components its share of the interpolation chains
runs (`share + 1 : shares : n_cons`).
"""
struct OwnedBlock
    rank::Int
    block::BlockRegion
    share::Int
    shares::Int
end

"""
    CouplingPiece

One block of one message of a level's coupling exchange: the peer rank in the
parent level's communicator, the tile, the position among the tile's
`coarse_regions` of the parent patch the piece lies in, the periodic image
of that patch it lies in (`image`, an index into `_images` of the transfer's
`period`; 1 is the patch itself), the padded index ranges this rank reads (a
send) or writes (a receive), and the components `first:step:n_cons` it
carries. A restriction piece's ranges on the fine side step by three along
the refined dimensions, selecting the coincident nodes.
"""
struct CouplingPiece
    peer::Int
    tile::Int
    part::Int
    image::Int
    ranges::NTuple{3,StepRange{Int,Int}}
    first::Int
    step::Int
end

"""
    LevelCoupling{T}

This rank's part of the point-to-point traffic between one refined level and
its parent, over the parent level's communicator `comm`: the block tables
(per parent patch and per tile, every rank holding a block), the pieces of
the box exchange (parent state to the tiles' owners, `box_sends` sorted by
peer and `box_recvs` by tile) and of the restriction exchange (the tiles'
coincident samples to the parent's owners), and the retained per-peer
buffers. Built by `build_level_coupling` whenever the level's tiles are
built, and held on the [`Level`](@ref); run by `_exchange_boxes!` and
`_exchange_restriction!`. Every quantity scales with the pieces this rank
sends or receives, apart from the two tables, whose entries number the blocks
of the level and its parent.
"""
struct LevelCoupling{T}
    comm::MPI.Comm
    parent_blocks::Vector{Vector{OwnedBlock}}   # per parent patch
    fine_blocks::Vector{Vector{OwnedBlock}}     # per tile
    box_sends::Vector{CouplingPiece}
    box_recvs::Vector{CouplingPiece}
    restrict_sends::Vector{CouplingPiece}
    restrict_recvs::Vector{CouplingPiece}
    sendbufs::Dict{Int,Vector{T}}
    recvbufs::Dict{Int,Vector{T}}
end

LevelCoupling{T}() where {T} =
    LevelCoupling{T}(MPI.COMM_NULL, Vector{OwnedBlock}[], Vector{OwnedBlock}[],
                     CouplingPiece[], CouplingPiece[], CouplingPiece[], CouplingPiece[],
                     Dict{Int,Vector{T}}(), Dict{Int,Vector{T}}())

# Every rank's rows of `width` integers, Allgathered over `comm`, as
# `(rank, row)` pairs in rank order; a serial communicator returns its own.
function _allgather_rows(mine::Vector{Int64}, width::Int, comm::MPI.Comm)
    np = MPI.Comm_size(comm)
    counts = np == 1 ? Cint[length(mine)] : MPI.Allgather(Cint(length(mine)), comm)
    flat = np == 1 ? mine : Vector{Int64}(undef, sum(counts))
    np == 1 || MPI.Allgatherv!(mine, MPI.VBuffer(flat, counts), comm)
    rows = Tuple{Int,Vector{Int}}[]
    at = 0
    for r in 0:np-1, _ in 1:(counts[r + 1] ÷ width)
        push!(rows, (r, Int.(flat[at .+ (1:width)])))
        at += width
    end
    return rows
end

# The nodes of `b` as ranges, shifted by `off`.
_block_nodes(b::BlockRegion, off::NTuple{3,Int}=(0, 0, 0)) =
    ntuple(d -> (off[d] + b.offset[d] + 1):(off[d] + b.offset[d] + b.extent[d]), 3)

const _NO_BOUNDARY = ((false, false), (false, false), (false, false))

# The buffer of a transfer's box per face, in parent nodes: `LEVEL_BUFFER`
# along an active dimension, none at a face on the domain boundary, where
# the parent holds no node beyond the region and the interpolation takes
# its one-sided stencils instead. A folded boundary face keeps the buffer,
# filled with the mirror image of the parent's nodes (`_mirror_folded_box!`).
_box_buffer(active::NTuple{3,Bool}, boundary::NTuple{3,NTuple{2,Bool}},
            folded::NTuple{3,NTuple{2,Bool}}=_NO_BOUNDARY) =
    ntuple(d -> active[d] ? (boundary[d][1] && !folded[d][1] ? 0 : LEVEL_BUFFER,
                             boundary[d][2] && !folded[d][2] ? 0 : LEVEL_BUFFER) :
                            (0, 0), 3)
_box_buffer(lt::LevelTransfer) = _box_buffer(lt.active, lt.boundary, lt.folded)

# The fine box node of fine patch node g is g + shift: three fine nodes per
# parent node of the low buffer, less the node a tile takes beyond the
# coincident lattice at a folded low face.
_box_shift(buffer, folded::NTuple{3,NTuple{2,Bool}}=_NO_BOUNDARY) =
    ntuple(d -> 3 * buffer[d][1] - _fold_lead(folded, d, 1), 3)
_box_shift(lt::LevelTransfer) = _box_shift(_box_buffer(lt), lt.folded)

# The box's extent in parent nodes.
_box_extent(region::BlockRegion, buffer) =
    ntuple(d -> region.extent[d] + buffer[d][1] + buffer[d][2], 3)

# The buffered box of a transfer as parent-level node ranges.
function _box_nodes(lt::LevelTransfer)
    b = _box_buffer(lt)
    lo = ntuple(d -> lt.region.offset[d] + 1 - b[d][1], 3)
    hi = ntuple(d -> lt.region.offset[d] + lt.region.extent[d] + b[d][2], 3)
    return ntuple(d -> lo[d]:hi[d], 3)
end

# The region-local coarse nodes the restriction writes: the whole region,
# less `RESTRICT_MARGIN` at a parent-fed face, since the fine solution is
# not imposed data at a face shared with a same-level tile.
function _restrict_window(lt::LevelTransfer)
    ext = lt.region.extent
    return ntuple(3) do d
        lt.active[d] || return 1:ext[d]
        lo = lt.imposed[d][1] ? 1 + RESTRICT_MARGIN : 1
        hi = lt.imposed[d][2] ? ext[d] - RESTRICT_MARGIN : ext[d]
        lo:hi
    end
end

# The region-local coarse nodes whose coincident fine node lies in the fine
# block `fb`: fine node f ↔ coarse node (f − 1) ÷ 3 + 1 along a refined
# dimension, when f ≡ 1 (mod 3), f counted on the coincident lattice, which
# starts `lead[d]` nodes into the patch (`_fold_lead`).
function _coincident(fb::BlockRegion, active::NTuple{3,Bool},
                     lead::NTuple{3,Int}=(0, 0, 0))
    return ntuple(3) do d
        lo = fb.offset[d] + 1 - lead[d]
        hi = fb.offset[d] + fb.extent[d] - lead[d]
        active[d] ? ((cld(lo - 1, 3) + 1):(fld(hi - 1, 3) + 1)) : (lo:hi)
    end
end

_isect3(a, b) = ntuple(d -> max(first(a[d]), first(b[d])):min(last(a[d]), last(b[d])), 3)

# Padded index ranges, unit step, of node ranges `r` on a block at `off`
# with pad `pad`: node n at n − off + pad.
_padded_steps(r, off, pad) =
    ntuple(d -> (first(r[d]) - off[d] + pad[d]):1:(last(r[d]) - off[d] + pad[d]), 3)

"""
    build_level_coupling(T, comm, transfers, parent_regions, parent_decomps,
                         fine_decomps) -> LevelCoupling{T}

This rank's [`LevelCoupling`](@ref) of the level whose tiles `transfers`
couple to the parent patches `parent_regions` (the parent level's node
space). `parent_decomps[p]` and `fine_decomps[t]` are this rank's
decompositions of parent patch `p` and tile `t`, `nothing` where it holds no
block. Collective over `comm`, the parent level's communicator: two
Allgathervs, of the parent's blocks and of the tiles' blocks, from which every
rank derives the pieces it sends and receives without further communication.
"""
function build_level_coupling(::Type{T}, comm::MPI.Comm,
                              transfers::Vector{LevelTransfer{T}},
                              parent_regions::Vector{BlockRegion},
                              parent_decomps::AbstractVector,
                              fine_decomps::AbstractVector) where {T}
    pmine = Int64[]
    for (p, dc) in enumerate(parent_decomps)
        dc === nothing && continue
        append!(pmine, Int64[p, dc.offset..., dc.n_local...])
    end
    parent_blocks = [OwnedBlock[] for _ in parent_regions]
    for (r, e) in _allgather_rows(pmine, 7, comm)
        push!(parent_blocks[e[1]],
              OwnedBlock(r, BlockRegion((e[2], e[3], e[4]), (e[5], e[6], e[7])), 0, 1))
    end
    fmine = Int64[]
    for (t, dc) in enumerate(fine_decomps)
        dc === nothing && continue
        append!(fmine, Int64[t, MPI.Comm_rank(dc.comm), MPI.Comm_size(dc.comm),
                             dc.offset..., dc.n_local...])
    end
    fine_blocks = [OwnedBlock[] for _ in transfers]
    for (r, e) in _allgather_rows(fmine, 9, comm)
        push!(fine_blocks[e[1]],
              OwnedBlock(r, BlockRegion((e[4], e[5], e[6]), (e[7], e[8], e[9])),
                         e[2], e[3]))
    end
    pid = Dict(r => p for (p, r) in enumerate(parent_regions))
    box_sends = CouplingPiece[]
    box_recvs = CouplingPiece[]
    restrict_sends = CouplingPiece[]
    restrict_recvs = CouplingPiece[]
    for (t, lt) in enumerate(transfers)
        active = lt.active
        box = _box_nodes(lt)
        win = _restrict_window(lt)
        roff = lt.region.offset
        fdc = fine_decomps[t]
        sample = ntuple(d -> active[d] ? 3 : 1, 3)
        lead = ntuple(d -> _fold_lead(lt.folded, d, 1), 3)
        # The box array: parent-level node n at n − box offset + pad.
        boxoff = ntuple(d -> first(box[d]) - 1, 3)
        padb = fdc === nothing ? (0, 0, 0) : lt.pdecomps[1].n_halo_d
        # Along a periodic dimension a box or a window across the seam meets a
        # parent patch in one of its images, the patch shifted by a period;
        # each image a range meets is a piece of its own. Only the patch
        # itself (image 1) is met by a region clear of the seam.
        images = _images(lt.period)
        for (k, creg) in enumerate(lt.coarse_regions), (v, σ) in enumerate(images)
            p = pid[creg]
            pdc = parent_decomps[p]
            # The image's origin in the parent level's node space.
            coff = creg.offset .+ σ
            # Region-local coarse node m ↔ parent patch node m + roff − coff.
            to_patch = ntuple(d -> roff[d] - coff[d], 3)
            if pdc !== nothing
                pad = pdc.n_halo_d
                mine = _block_nodes(BlockRegion(pdc.offset, pdc.n_local), coff)
                # Box: the part of the box this rank holds, to every owner of the tile.
                nodes = _isect3(box, mine)
                if !any(isempty, nodes)
                    lr = _padded_steps(nodes, coff .+ pdc.offset, pad)
                    for fb in fine_blocks[t]
                        push!(box_sends, CouplingPiece(fb.rank, t, k, v, lr, fb.share + 1,
                                                       fb.shares))
                    end
                end
                # Restriction: the covered nodes of this block, from every
                # fine block holding their coincident samples.
                mreg = _isect3(win, _block_nodes(BlockRegion(pdc.offset, pdc.n_local),
                                                 (.-to_patch)))
                for fb in fine_blocks[t]
                    m = _isect3(mreg, _coincident(fb.block, active, lead))
                    any(isempty, m) && continue
                    lr = _padded_steps(m, pdc.offset .- to_patch, pad)
                    push!(restrict_recvs, CouplingPiece(fb.rank, t, k, v, lr, 1, 1))
                end
            end
            fdc === nothing && continue
            share = MPI.Comm_rank(fdc.comm)
            shares = MPI.Comm_size(fdc.comm)
            padf = fdc.n_halo_d
            coin = _isect3(win, _coincident(BlockRegion(fdc.offset, fdc.n_local), active,
                                            lead))
            for pb in parent_blocks[p]
                nodes = _isect3(box, _block_nodes(pb.block, coff))
                if !any(isempty, nodes)
                    lr = _padded_steps(nodes, boxoff, padb)
                    push!(box_recvs, CouplingPiece(pb.rank, t, k, v, lr, share + 1,
                                                   shares))
                end
                m = _isect3(coin, _block_nodes(pb.block, (.-to_patch)))
                any(isempty, m) && continue
                # Coincident fine node s(m − 1) + 1 + lead on this rank's block.
                lr = ntuple(3) do d
                    s = sample[d]
                    lo = s * (first(m[d]) - 1) + 1 + lead[d] - fdc.offset[d] + padf[d]
                    hi = s * (last(m[d]) - 1) + 1 + lead[d] - fdc.offset[d] + padf[d]
                    lo:s:hi
                end
                push!(restrict_sends, CouplingPiece(pb.rank, t, k, v, lr, 1, 1))
            end
        end
    end
    # A sender packs its pieces for one peer in tile order and the receiver
    # reads them in the same order; the receives apply in tile order.
    sort!(box_sends; by=q -> (q.peer, q.tile, q.part, q.image))
    sort!(restrict_sends; by=q -> (q.peer, q.tile, q.part, q.image))
    sort!(box_recvs; by=q -> (q.tile, q.part, q.image, q.peer))
    sort!(restrict_recvs; by=q -> (q.tile, q.part, q.image, q.peer))
    return LevelCoupling{T}(comm, parent_blocks, fine_blocks, box_sends, box_recvs,
                            restrict_sends, restrict_recvs,
                            Dict{Int,Vector{T}}(), Dict{Int,Vector{T}}())
end

# The components a piece carries: its receiver's share, or every one.
@inline _piece_comps(q::CouplingPiece, n_cons::Int, all_comps::Bool) =
    all_comps ? (1:1:n_cons) : (q.first:q.step:n_cons)

@inline _piece_length(q::CouplingPiece, n_cons::Int, all_comps::Bool) =
    length(_piece_comps(q, n_cons, all_comps)) * prod(length.(q.ranges))

# Pack the block of `A` a piece names into `buf` after linear index `at`
# (components slowest, then k, j, i) and return the new end. Device storage
# stages through a contiguous device array, as the halo exchange does.
function _pack_piece!(buf::Vector, at::Int, A, r, comps)
    if !_device_path(A)
        @inbounds for c in comps, k in r[3], j in r[2], i in r[1]
            at += 1
            buf[at] = A[i, j, k, c]
        end
        return at
    end
    v = view(parent(A), r[1], r[2], r[3], comps)
    n = length(v)
    dsend = _device_send_stage(parent(A), n)
    reshape(view(dsend, 1:n), size(v)) .= v
    _tracked_copy!(buf, at + 1, dsend, 1, n)
    return at + n
end

function _unpack_piece!(A, buf::Vector, at::Int, r, comps)
    if !_device_path(A)
        @inbounds for c in comps, k in r[3], j in r[2], i in r[1]
            at += 1
            A[i, j, k, c] = buf[at]
        end
        return at
    end
    v = view(parent(A), r[1], r[2], r[3], comps)
    n = length(v)
    drecv = _device_send_stage(parent(A), n)
    _tracked_copy!(drecv, 1, buf, at + 1, n)
    v .= reshape(view(drecv, 1:n), size(v))
    return at + n
end

# One exchange: every piece of `sends` whose tile `tiles` selects (all of
# them for `nothing`) packed per peer from `src(piece)`, one message per peer,
# and every selected piece of `recvs` written into `dst(piece)` in the order
# of `recvs`. A piece to or from this rank itself moves through its buffer
# without MPI. Entered by every rank of `cp.comm`; a rank with no piece
# posts nothing and returns.
function _run_coupling!(cp::LevelCoupling{T}, sends::Vector{CouplingPiece},
                        recvs::Vector{CouplingPiece}, n_cons::Int, all_comps::Bool,
                        tiles, tag::Int, src::F, dst::G) where {T,F,G}
    (isempty(sends) && isempty(recvs)) && return nothing
    comm = cp.comm
    me = MPI.Comm_rank(comm)
    # An empty piece moves nothing: a tile spread over more ranks than there
    # are conserved components leaves some of its ranks no share of the
    # chains, hence no box. Both sides drop it, so a message is posted only
    # where both hold data for it; a receive posted for a message its sender
    # skips would wait forever.
    want(q) = (tiles === nothing || tiles[q.tile]) &&
              _piece_length(q, n_cons, all_comps) > 0
    rsize = Dict{Int,Int}()
    for q in recvs
        want(q) || continue
        rsize[q.peer] = get(rsize, q.peer, 0) + _piece_length(q, n_cons, all_comps)
    end
    ssize = Dict{Int,Int}()
    for q in sends
        want(q) || continue
        ssize[q.peer] = get(ssize, q.peer, 0) + _piece_length(q, n_cons, all_comps)
    end
    reqs = MPI.Request[]
    for (peer, n) in rsize
        buf = _fit!(get!(() -> T[], cp.recvbufs, peer), n)
        peer == me || push!(reqs, MPI.Irecv!(buf, comm; source=peer, tag=tag))
    end
    # `sends` is sorted by peer, so each peer's pieces are one run of it.
    i = 1
    while i <= length(sends)
        peer = sends[i].peer
        j = i
        while j <= length(sends) && sends[j].peer == peer
            j += 1
        end
        n = get(ssize, peer, 0)
        if n > 0
            # The pieces to oneself are packed straight into the receive buffer.
            buf = _fit!(peer == me ? cp.recvbufs[me] :
                                     get!(() -> T[], cp.sendbufs, peer), n)
            at = 0
            for k in i:(j - 1)
                q = sends[k]
                want(q) || continue
                at = _pack_piece!(buf, at, src(q), q.ranges,
                                  _piece_comps(q, n_cons, all_comps))
            end
            peer == me || push!(reqs, MPI.Isend(buf, comm; dest=peer, tag=tag))
        end
        i = j
    end
    MPI.Waitall(reqs)
    offset = Dict{Int,Int}()
    for q in recvs
        want(q) || continue
        offset[q.peer] = _unpack_piece!(dst(q), cp.recvbufs[q.peer],
                                        get(offset, q.peer, 0), q.ranges,
                                        _piece_comps(q, n_cons, all_comps))
    end
    return nothing
end

"""
    TileStack

The stacked storage of some of a device level's tiles held on this rank:
`patch` is the spanning patch, whose arrays are `StackedArray`s over the
tiles' blocks and whose plans are the batched device plans, and `members`
are the `solver.patches` indices of the tiles in slot order, each holding
views of the same arrays. The step drivers
evaluate the right-hand side, the stage update and the filter once per stack
through `PatchSolver(solver, stack.patch)` on the state the members' views
share (`_stack_state`); everything else on the level stays per tile.
A level's tiles group into one stack per padded extent (a lattice cell
clipped at the domain edge differs from the rest), and a regrid rebuilds the
stacks with the tiles.
"""
struct TileStack
    patch::Patch
    members::Vector{Int}
end

"""
    _stack_state(stack, states) -> ConservedState

The conserved state of a whole `TileStack`: the array every member's state in
`states` is a view of, wrapped as a `StackedArray` so a launch on it runs
every tile. The members are checked to be the views `allocate_state` hands
out, in slot order; a state vector assembled any other way is refused rather
than advanced tile by tile. A stack of one tile accepts any state of the
stack's extent: its view spans the whole stacked array, which a GPU array
package returns as a contiguous array rather than a `SubArray`.
"""
function _stack_state(stack::TileStack, states)
    field = stack.patch.rho
    ntiles, stride = field.ntiles, field.stride
    first_view = parent(states[stack.members[1]])
    if !(first_view isa SubArray)
        ntiles == 1 && size(first_view, 3) == stride || _unstacked_state_error()
        return ConservedState(StackedArray(first_view, ntiles, stride))
    end
    raw = parent(first_view)
    for (slot, li) in enumerate(stack.members)
        v = parent(states[li])
        (v isa SubArray && parent(v) === raw &&
         parentindices(v)[3] == ((slot - 1) * stride + 1):(slot * stride)) ||
            _unstacked_state_error()
    end
    return ConservedState(StackedArray(raw, ntiles, stride))
end

@noinline _unstacked_state_error() =
    error("the states of a stacked level's tiles must be the views of one " *
          "stacked array that allocate_state returns; a state vector assembled " *
          "from separate arrays cannot be advanced on this level")

"""
    Level

One level of the refinement hierarchy: its index (0 is the root), the rank
subset owning it ([`LevelComm`](@ref)), the owner rank range of each of its
tiles and this rank's [`TileGroup`](@ref), the indices into `solver.patches`
of the tiles this rank holds and which tiles they are, the
[`LevelTransfer`](@ref) coupling each tile to its parent (empty at the root),
and the same-level interface records among the level's tiles (the root's
live on the `Solver`). `solver.levels` holds them in ascending order, and
every level below the root is nested inside the one above it. Below the root
a level is a set of tiles on a lattice of edge `tile` parent nodes (one patch
over the whole region when `tile` is 0), abutting tiles sharing their
interface plane as root slabs do.

`patches`, `tiles` and the records are this rank's own and are empty on a
rank holding no tile of the level; `owners` and `transfers` are held by every
rank of the parent level's subset, since the parent's side of the coupling
exchange (`coupling`, a [`LevelCoupling`](@ref)) reads them there, and are
indexed by tile. `owners` is the
authority across regrids: a surviving tile keeps its range there until a
rebalance moves it (`_place_tiles`, `src/regrid.jl`). A regridded tiled
level may hold no tiles; `owners` and `transfers` are then empty and every
rank carries an absent `LevelComm`.
`patches[i]` is tile `tiles[i]`; `transfers[t].fine_index` is the
`solver.patches` index of tile `t` on this rank, or 0.
"""
struct Level{T}
    index::Int
    level_comm::LevelComm
    owners::Vector{UnitRange{Int}}   # per tile: its rank range in level_comm
    group::TileGroup                 # this rank's group
    tiles::Vector{Int}               # tile of each held patch
    patches::Vector{Int}
    transfers::Vector{LevelTransfer{T}}
    # Same-level records per dimension, applied in dimension order: the
    # records of dimension d span the transverse dimensions before d over
    # their padded ranges, so edges and corners are reached, and a node
    # shared by 2^k tiles ends at the mean of all its copies. See
    # `_sync_level_records!`.
    ghost_sends::NTuple{3,Vector{GhostRecord{T}}}
    ghost_recvs::NTuple{3,Vector{GhostRecord{T}}}
    plane_pairs::NTuple{3,Vector{PlaneRecord{T}}}
    phases::NTuple{3,Bool}           # some tile has a neighbor along d
    # The stacked storage of this rank's tiles on a device backend
    # (`TileStack`), one per padded extent; empty on the host backend, at the
    # root and on a rank holding no tile.
    stacks::Vector{TileStack}
    # This rank's part of the point-to-point traffic between the level and
    # its parent (`LevelCoupling`); empty at the root and on a rank outside
    # the parent's subset.
    coupling::LevelCoupling{T}
end

# The root level, or one this rank holds no tile of: the whole subset is one
# group and there are no records.
Level{T}(index::Int, lc::LevelComm, patches::Vector{Int},
         transfers::Vector{LevelTransfer{T}}) where {T} =
    Level{T}(index, lc, [0:(lc.size - 1)],
             lc.owned ? TileGroup(lc.comm, 0:(lc.size - 1), false) :
                        absent_tile_group(),
             collect(eachindex(patches)), patches, transfers)

# A level with no records (a rank outside it, or one tile).
Level{T}(index::Int, lc::LevelComm, owners::Vector{UnitRange{Int}},
         group::TileGroup, tiles::Vector{Int}, patches::Vector{Int},
         transfers::Vector{LevelTransfer{T}};
         stacks::Vector{TileStack}=TileStack[],
         coupling::LevelCoupling{T}=LevelCoupling{T}()) where {T} =
    Level{T}(index, lc, owners, group, tiles, patches, transfers,
             ntuple(_ -> GhostRecord{T}[], 3), ntuple(_ -> GhostRecord{T}[], 3),
             ntuple(_ -> PlaneRecord{T}[], 3), (false, false, false), stacks,
             coupling)

# A level with the records `_level_records` returns.
Level{T}(index::Int, lc::LevelComm, owners::Vector{UnitRange{Int}},
         group::TileGroup, tiles::Vector{Int}, patches::Vector{Int},
         transfers::Vector{LevelTransfer{T}}, records::Tuple;
         stacks::Vector{TileStack}=TileStack[],
         coupling::LevelCoupling{T}=LevelCoupling{T}()) where {T} =
    Level{T}(index, lc, owners, group, tiles, patches, transfers, records...,
             stacks, coupling)

"""
    _exchange_boxes!(solver, srcs, lev, select, all_comps, tiles=nothing)

Deliver the buffered box of each tile of `lev` selected by `tiles` to that
tile's owners, from `srcs` (the solver's state vector, or its right-hand
sides for the Hermite endpoints), into `select(transfer)`: each owner's own
components, or every component under `all_comps`. Entered by every rank of
the parent level's subset, whatever it holds.
"""
function _exchange_boxes!(solver, srcs, lev::Level, select::F, all_comps::Bool,
                          tiles=nothing) where {F}
    cp = lev.coupling
    tr = lev.transfers
    _run_coupling!(cp, cp.box_sends, cp.box_recvs, solver.equations.n_cons, all_comps,
                   tiles, _BOX_TAG, q -> srcs[tr[q.tile].coarse_local[q.part]],
                   q -> select(tr[q.tile]))
    _mirror_folded_box!(solver, lev, select)
    return srcs
end

"""
    _exchange_restriction!(solver, states, lev, tiles=nothing)

Write the coincident samples of each tile of `lev` selected by `tiles` onto
the covered parent nodes, holding `RESTRICT_MARGIN` off a parent-fed face,
from the tiles' owners to the parent ranks owning those nodes. Entered by
every rank of the parent level's subset.
"""
function _exchange_restriction!(solver, states, lev::Level, tiles=nothing)
    cp = lev.coupling
    tr = lev.transfers
    _run_coupling!(cp, cp.restrict_sends, cp.restrict_recvs, solver.equations.n_cons,
                   true, tiles, _RESTRICT_TAG, q -> states[tr[q.tile].fine_index],
                   q -> states[tr[q.tile].coarse_local[q.part]])
    return states
end

"Number of levels in the hierarchy, the root included."
nlevels(solver) = length(getfield(solver, :levels))

"""
    refined_region(solver, level=1) -> BlockRegion

The refined region of the sole patch on `level`, in the parent level's node
space. Errors on a level holding several patches; see
[`level_regions`](@ref) for those.
"""
function refined_region(solver, level::Int=1)
    lev = getfield(solver, :levels)[level + 1]
    length(lev.transfers) == 1 ||
        error("level $level holds $(length(lev.transfers)) patches; " *
              "refined_region needs one")
    return lev.transfers[1].region
end

"""
    level_regions(solver, level) -> Vector{BlockRegion}

The regions of the patches on `level`, in the parent level's node space, in
patch order. Empty on a rank outside the rank subset of the parent level,
which holds no part of that level; rank 0 holds every level's regions.
"""
level_regions(solver, level::Int) =
    [lt.region for lt in getfield(solver, :levels)[level + 1].transfers]

# --- Tile lattice -------------------------------------------------------------
#
# A refined level is tiled on a lattice of edge `tile` parent nodes anchored
# at parent node 0: lattice cell k spans parent nodes k·tile + 1 .. (k+1)·tile
# + 1, so abutting tiles share their interface plane (as root slabs do) and
# every tile face is either fully shared with one neighbor or fully fed from
# the parent. Tiles at the domain edge are clipped to the nesting margin and
# dropped when clipping leaves them below the four-node minimum. The lattice
# is global so that a regrid moves tiles in and out of the set without ever
# changing a surviving tile's region.

# Lattice cell indices meeting the node interval [lo, hi]. A node on a
# lattice plane belongs to both cells it separates, but an interval of two
# or more nodes needs a cell only where it reaches past the plane, so its
# end nodes count toward the cell on the inner side alone.
function _tile_span(lo::Int, hi::Int, tile::Int)
    hi > lo && return (max(lo - 1, 0) ÷ tile):((hi - 2) ÷ tile)
    return (max(lo - 2, 0) ÷ tile):(max(lo - 1, 0) ÷ tile)
end

# The tile of lattice cell `k` clipped to the feasible node interval
# `[lo, hi]`, or `nothing` when clipping leaves fewer than four nodes. `last`
# is the last node of the lattice, the parent's node coincident with the
# root's last; a feasible interval reaching past either end of the lattice,
# as it does at a fold below the first refined level (`_level_span`),
# extends the first and the last cell to its ends instead of adding a cell.
function _lattice_tile(k::NTuple{3,Int}, active::NTuple{3,Bool}, tile::Int,
                       lo::NTuple{3,Int}, hi::NTuple{3,Int}, last::NTuple{3,Int}=hi)
    any(d -> active[d] && k[d] * tile + 1 >= last[d], 1:3) && return nothing
    offext = ntuple(3) do d
        active[d] || return (0, 1)
        a = k[d] == 0 ? lo[d] : max(k[d] * tile + 1, lo[d])
        top = k[d] * tile + tile + 1
        b = top >= last[d] ? hi[d] : min(top, hi[d])
        (a - 1, b - a + 1)
    end
    any(d -> active[d] && offext[d][2] < 4, 1:3) && return nothing
    return BlockRegion(ntuple(d -> offext[d][1], 3), ntuple(d -> offext[d][2], 3))
end

# Tiles covering `box` (parent-level node space), in lattice order. Along a
# dimension of nonzero `wrap`, the period of a periodic dimension whose seam
# the lattice crosses, the box may run past the seam and the cells it meets
# are those of its pieces on either side (`_wrapped_parts`), on the lattice
# over [1, P + 1] whose last cell ends on the seam.
function _level_tiles(box::BlockRegion, active::NTuple{3,Bool}, tile::Int,
                      lo::NTuple{3,Int}, hi::NTuple{3,Int}, last::NTuple{3,Int}=hi,
                      wrap::NTuple{3,Int}=(0, 0, 0))
    spans = ntuple(3) do d
        active[d] || return [0]
        parts = _wrapped_parts(box.offset[d] + 1, box.offset[d] + box.extent[d],
                               wrap[d])
        sort!(unique!(reduce(vcat, [collect(_tile_span(a, b, tile)) for (a, b) in parts])))
    end
    tiles = BlockRegion[]
    for k3 in spans[3], k2 in spans[2], k1 in spans[1]
        t = _lattice_tile((k1, k2, k3), active, tile, lo, hi, last)
        t === nothing || push!(tiles, t)
    end
    return tiles
end

# Face table of a tile set: neighbor patch index (1-based within the set)
# per face, 0 where none. Two tiles abut along `d` when one's high plane is
# the other's low plane and their transverse extents coincide, which the
# lattice guarantees whenever they touch at all. Along a periodic dimension
# of `period` the planes are compared modulo the period, so the tiles on
# either side of the seam abut there.
function _tile_faces(regions::Vector{BlockRegion}, period::NTuple{3,Int}=(0, 0, 0))
    n = length(regions)
    same_node(a, b, d) = period[d] == 0 ? a == b : mod(a - b, period[d]) == 0
    faces = Vector{NTuple{3,NTuple{2,Int}}}(undef, n)
    for p in 1:n
        rp = regions[p]
        faces[p] = ntuple(3) do d
            lo = 0
            hi = 0
            for q in 1:n
                q == p && continue
                rq = regions[q]
                same = all(e -> e == d || (same_node(rq.offset[e], rp.offset[e], e) &&
                                           rq.extent[e] == rp.extent[e]), 1:3)
                same || continue
                same_node(rq.offset[d] + rq.extent[d] - 1, rp.offset[d], d) && (lo = q)
                same_node(rp.offset[d] + rp.extent[d] - 1, rq.offset[d], d) && (hi = q)
            end
            (lo, hi)
        end
    end
    return faces
end

# The parent-level patches (solver indices) whose regions meet the buffered
# box of `region`, from the parent level's transfers (the root's patches
# from `patch_regions`).
function _parents_of(region::BlockRegion, active::NTuple{3,Bool},
                     parent_indices::Vector{Int}, parent_regions::Vector{BlockRegion},
                     period::NTuple{3,Int}=(0, 0, 0))
    box = _buffered(region, active, LEVEL_BUFFER)
    hits = Int[]
    for (i, r0) in zip(parent_indices, parent_regions)
        any(_images(period)) do σ
            r = _shifted(r0, σ)
            !any(d -> box.offset[d] + box.extent[d] <= r.offset[d] ||
                      r.offset[d] + r.extent[d] <= box.offset[d], 1:3)
        end || continue
        push!(hits, i)
    end
    return hits
end

# Same-level records among the tiles of a refined level, built through the
# root machinery on tile ids (positions in `regions`, every tile of the
# level) and shifted onto solver patch indices, one record set per dimension
# for the phased sync. `tiles` are the tiles this rank holds, `patch_indices`
# their solver indices and `decomps` their decompositions, all aligned.
# Collective over `comm`, the level's own communicator (one Allgatherv per
# dimension of every rank's blocks of the tiles it holds; a rank outside the
# subset must not call). Each record's `partner` is therefore a rank number
# in that communicator, the one `_sync_level_records!` exchanges over, and a
# plane record's `partner_pid` stays the partner's tile id, which
# `_seed_planes!` reads. `period` is the level's own node-space period
# (`_level_period`), across whose seam two tiles abut. Returns the tuple the
# `Level{T}` constructor takes.
function _level_records(::Type{T}, comm::MPI.Comm, regions::Vector{BlockRegion},
                        tiles::Vector{Int}, patch_indices::Vector{Int}, decomps,
                        n_cons::Int, period::NTuple{3,Int}=(0, 0, 0)) where {T}
    sends = ntuple(_ -> GhostRecord{T}[], 3)
    recvs = ntuple(_ -> GhostRecord{T}[], 3)
    planes = ntuple(_ -> PlaneRecord{T}[], 3)
    length(regions) > 1 || return sends, recvs, planes, (false, false, false)
    faces = _tile_faces(regions, period)
    phases = ntuple(d -> any(f -> f[d] != (0, 0), faces), 3)
    shift(r::GhostRecord{T}) =
        GhostRecord{T}(patch_indices[r.patch], r.partner,
                       r.partner_patch == 0 ? 0 : patch_indices[r.partner_patch],
                       r.tag, r.mine, r.theirs, r.buf)
    shift(r::PlaneRecord{T}) =
        PlaneRecord{T}(patch_indices[r.patch], r.partner,
                       r.partner_patch == 0 ? 0 : patch_indices[r.partner_patch],
                       r.partner_pid,
                       r.tag, r.sendtag, r.mine, r.theirs, r.buf, r.sbuf)
    for d in 1:3
        phases[d] || continue
        faces_d = [ntuple(e -> e == d ? f[e] : (0, 0), 3) for f in faces]
        s, r, p = build_interface_records(T, comm, regions, faces_d, tiles,
                                          decomps, n_cons;
                                          padded_transverse=ntuple(e -> e < d, 3))
        append!(sends[d], map(shift, s))
        append!(recvs[d], map(shift, r))
        append!(planes[d], map(shift, p))
    end
    return sends, recvs, planes, phases
end

# `region` shrunk by one node on each face flagged in `imposed`: the nodes
# of a refined patch that are its own solution rather than imposed data.
_erode(region::BlockRegion, imposed::NTuple{3,NTuple{2,Bool}},
       active::NTuple{3,Bool}) =
    BlockRegion(ntuple(d -> region.offset[d] + (active[d] && imposed[d][1]), 3),
                ntuple(d -> region.extent[d] - (active[d] && imposed[d][1]) -
                            (active[d] && imposed[d][2]), 3))

# Whether the fine shell slot at patch-global node `g` (padded slots included)
# is imposed from the parent: outside the strict interior along some active
# dimension whose face on that side is parent-fed, and not beyond a face on
# the domain boundary (`boundary`), where the box holds no data and the slot
# is the patch's own ghost, as at the root. With every face imposed this is
# the complement of the strict interior [2, N − 1]^3.
@inline function _in_shell(g1, g2, g3, Nf, active, imposed, boundary)
    @inbounds begin
        ((g1 >= 1 || !boundary[1][1]) && (g1 <= Nf[1] || !boundary[1][2]) &&
         (g2 >= 1 || !boundary[2][1]) && (g2 <= Nf[2] || !boundary[2][2]) &&
         (g3 >= 1 || !boundary[3][1]) && (g3 <= Nf[3] || !boundary[3][2])) &&
        ((active[1] && ((g1 <= 1 && imposed[1][1]) || (g1 >= Nf[1] && imposed[1][2]))) ||
         (active[2] && ((g2 <= 1 && imposed[2][1]) || (g2 >= Nf[2] && imposed[2][2]))) ||
         (active[3] && ((g3 <= 1 && imposed[3][1]) || (g3 >= Nf[3] && imposed[3][2]))))
    end
end

# Same-level consistency of one refined level, one dimension at a time:
# shared-plane averaging and ghost refill through the records of dimension
# d, whose transverse ranges are padded along the dimensions already done,
# then each patch's own halo exchange so the rank halos carry the result
# into the next phase. Pairwise averaging in one flat pass is not
# idempotent where four (2-D) or eight (3-D) tiles share a node, since a
# later pair reads a value an earlier pair changed; phased over dimensions
# every copy ends at the mean of all of them, and the phase-d ghost strips,
# read over the padded transverse ranges, carry the earlier phases' ghosts
# into the edge and corner ghosts. The trailing per-patch halo exchange is
# the caller's, as for the root.
function _sync_level_records!(solver, states, lev::Level)
    any(lev.phases) || return states
    patches = getfield(solver, :patches)
    comm = lev.level_comm.comm
    last = findlast(lev.phases)
    # Counted as waiting, not work, for the rebalance measure: a rank with
    # fewer tiles than its neighbor's owner blocks here until that rank
    # reaches the same phase, and the work of the exchange itself is a few
    # planes.
    t0 = time_ns()
    for d in 1:last
        lev.phases[d] || continue
        _average_planes!(solver, states, comm, lev.plane_pairs[d])
        _exchange_ghosts!(solver, states, comm, lev.ghost_sends[d],
                          lev.ghost_recvs[d])
        d == last && break
        for pi in lev.patches
            exchange_state!(states[pi], patches[pi].decomp)
        end
    end
    _wait!(solver, t0)
    return states
end

# One-way seeding of the planes fresh tiles share with surviving ones after
# a regrid: a fresh tile takes the survivor's plane (its evolved value)
# instead of averaging its interpolated one into it; two fresh tiles, or
# two survivors, take the mean. Phased as the sync is. `fresh` is indexed by
# tile; a record names its own side by solver patch index and its partner
# by tile id.
function _seed_planes!(solver, states, lev::Level, fresh::AbstractVector{Bool})
    tile_of = Dict(zip(lev.patches, lev.tiles))
    weight(pl) = fresh[tile_of[pl.patch]] == fresh[pl.partner_pid] ? 0.5 :
                 fresh[tile_of[pl.patch]] ? 0.0 : 1.0
    patches = getfield(solver, :patches)
    for d in 1:3
        lev.phases[d] || continue
        _combine_planes!(solver, states, lev.level_comm.comm,
                         lev.plane_pairs[d], weight)
        for pi in lev.patches
            exchange_state!(states[pi], patches[pi].decomp)
        end
    end
    return states
end

# Level sync inside the subcycled driver: records, then each patch's own
# halo exchange, skipped entirely for a one-tile level (its RHS evaluation
# exchanges halos itself). Entered by every owner of the level, a rank
# holding one tile or none included: the records are point-to-point and a
# tile's partner may sit on another rank.
function _sync_level!(solver, states, lev::Level)
    any(lev.phases) || return states
    _sync_level_records!(solver, states, lev)
    patches = getfield(solver, :patches)
    for pi in lev.patches
        exchange_state!(states[pi], patches[pi].decomp)
    end
    return states
end

# Decomp chain refining `dims_to_refine` one at a time, starting from the
# coarse extents. Every stage runs replicated per rank (COMM_SELF, so the
# chain operators are the serial ones whatever the rank count) and is
# non-periodic along active dimensions; collapsed dimensions stay collapsed.
# `folded` marks the faces of a restriction chain's fine patch on a fold,
# where the fine extent carries the node beyond the coincident lattice and
# each stage's restriction folds (`plan_transfer`); an interpolation chain
# runs on the coincident lattice of a mirror-filled box and takes none.
function _refine_chain(::Type{T}, ext0::NTuple{3,Int}, active::NTuple{3,Bool},
                       dims_to_refine::Vector{Int}, n_halo::Int,
                       interp_order::Int,
                       folded::NTuple{3,NTuple{2,Bool}}=_NO_BOUNDARY) where {T}
    pper = ntuple(d -> !active[d], 3)
    exts = _chain_extents(ext0, dims_to_refine, folded)
    decomps = [Decomp{T}(e, pper; dims=(1, 1, 1), n_halo=n_halo,
                         comm=MPI.COMM_SELF) for e in exts]
    plans = [plan_transfer(decomps[k+1], decomps[k], dims_to_refine[k], T;
                           interp_order=interp_order,
                           lo_fold=folded[dims_to_refine[k]][1],
                           hi_fold=folded[dims_to_refine[k]][2])
             for k in eachindex(dims_to_refine)]
    stages = [zeros(T, _padded_extent(e, active, n_halo)) for e in exts]
    return decomps, plans, stages
end

# The unpadded extents of a chain's stages 0 .. K from the stage-0 extent,
# refining one dimension per stage, each folded face adding its node
# (`_fold_lead`), and a stage's padded array size.
function _chain_extents(ext0::NTuple{3,Int}, dims_to_refine::Vector{Int},
                        folded::NTuple{3,NTuple{2,Bool}}=_NO_BOUNDARY)
    exts = [ext0]
    for dk in dims_to_refine
        prev = exts[end]
        lead = _fold_lead(folded, dk, 1) + _fold_lead(folded, dk, 2)
        push!(exts, ntuple(d -> d == dk ? 3 * prev[d] - 2 + lead : prev[d], 3))
    end
    return exts
end

_padded_extent(e::NTuple{3,Int}, active::NTuple{3,Bool}, n_halo::Int) =
    ntuple(d -> e[d] + 2 * (active[d] ? n_halo : 0), 3)

# --- Device scratch of a level transfer ---------------------------------------
#
# A device-resident refined patch builds its shell on the backend: the
# gathered coarse box uploads once per imposition (or, under subcycling, the
# four Hermite boxes once per parent step and the blend runs as a kernel),
# the tensor-product Lagrange chain runs as one kernel per stage over this
# rank's components, and the shell ring packs on the device ahead of its
# Allgatherv. Under `interface_flux = :ghost` with molecular transport the
# conserved gradients of the ring are taken there too, through device plans
# on the chain's fine box, and the gradient ring stays on the device for the
# right-hand side that reads it. The scratch below is the storage that
# takes: it lives on the fine `Patch` (typed by the patch's array type, so
# `LevelTransfer`, `Level` and `Solver` keep their types) and is empty on the
# host backend, whose chain runs on the transfer's own host stages.

"""
    LevelScratch

Device storage of one refined patch's level transfer: the four Hermite
boxes (uploaded by `save_level_boxes!`, which fills this rank's own
components), the interpolation
chain's stages 0 .. K over this rank's own components (the
component-distributed chain of `_impose_shell!`) and, under
`interface_flux = :ghost` with molecular transport, the backend's form of
the `ShellGradients` plans with their decompositions, its box-sized scratch
field and its gradient ring. Empty on the host backend and on a patch without a parent.
"""
struct LevelScratch{A4<:AbstractArray,A3<:AbstractArray,A2<:AbstractArray}
    Q0::A4
    dQ0::A4
    Q1::A4
    dQ1::A4
    stages::Vector{A4}
    gplans::Vector{Any}
    gdecomps::Vector{Any}
    gtmp::A3
    gring::A2
end

# The empty scratch of a host patch or a root patch, typed by the backend's
# array type through a field of it.
function _empty_level_scratch(f::AbstractArray{T,3}) where {T}
    e() = similar(f, T, 0, 0, 0, 0)
    return LevelScratch(e(), e(), e(), e(), typeof(e())[], Any[], Any[],
                        similar(f, T, 0, 0, 0), similar(f, T, 0, 0))
end

# The scratch of a refined patch over `region` (parent node space) on the
# backend `f` belongs to: empty on host storage. `np` and `me` are the
# tile's communicator size and this rank's position, which fix the
# components this rank's chain runs (`(me+1):np:n_cons`). A
# `gradient_deriv` adds the gradient ring's storage on `backend`, whatever
# the patch's faces: a tile kept at a regrid keeps its scratch while its
# faces change. `fine_decomp` and `hf` are the patch's decomposition and
# spacing; `folded` marks its boundary faces on a fold, where the box keeps
# its buffer, as `build_level_transfer` builds the host chain.
function _level_scratch(f::AbstractArray{T,3}, region::BlockRegion,
                        active::NTuple{3,Bool}, n_halo::Int, n_cons::Int,
                        np::Int, me::Int; gradient_deriv=nothing,
                        backend::AbstractBackend=CPUBackend(),
                        fine_decomp::Union{Nothing,Decomp}=nothing,
                        hf=nothing,
                        boundary::NTuple{3,NTuple{2,Bool}}=_NO_BOUNDARY,
                        folded::NTuple{3,NTuple{2,Bool}}=_NO_BOUNDARY) where {T}
    _device_path(f) || return _empty_level_scratch(f)
    dims = [d for d in 1:3 if active[d]]
    buffer = _box_buffer(active, boundary, folded)
    boxext = _box_extent(region, buffer)
    exts = _chain_extents(boxext, dims)
    n_owned = length((me+1):np:n_cons)
    box = _padded_extent(exts[1], active, n_halo)
    stages = [similar(f, T, _padded_extent(e, active, n_halo)..., n_owned)
              for e in exts]
    gplans = Any[]
    gdecomps = Any[]
    gtmp = similar(f, T, 0, 0, 0)
    gring = similar(f, T, 0, 0)
    if gradient_deriv !== nothing
        # The chain's final box, as `_refine_chain` builds it.
        boxf = Decomp{T}(exts[end], ntuple(d -> !active[d], 3); dims=(1, 1, 1),
                         n_halo=n_halo, comm=MPI.COMM_SELF)
        # Every face but a boundary one, whose ghost layers lie outside the box.
        hplans, gdecomps = _box_gradient_plans(boxf, gradient_deriv, active, hf,
                                               fine_decomp.n_halo_d,
                                               ntuple(d -> (!boundary[d][1],
                                                            !boundary[d][2]), 3),
                                               buffer, folded)
        gplans = Any[p === nothing ? nothing : backend_plan(backend, p) for p in hplans]
        gtmp = fill!(similar(f, T, _padded_extent(exts[end], active, n_halo)), 0)
        _, ringlen = _slab_table(_ring_slabs(region,
                                             ntuple(d -> fine_decomp.active[d], 3),
                                             fine_decomp.n_halo_d, boundary, folded))
        gring = fill!(similar(f, T, ringlen, 3 * n_cons), 0)
    end
    return LevelScratch(similar(f, T, box..., n_cons), similar(f, T, box..., n_cons),
                        similar(f, T, box..., n_cons), similar(f, T, box..., n_cons),
                        stages, gplans, gdecomps, gtmp, gring)
end

# Stage-0 sources of an imposition: the gathered box as it is, or the cubic
# Hermite blend of the stored boxes at fraction `θ` of a parent step `dt`.
struct BoxFill end
struct HermiteFill{T}
    θ::T
    dt::T
end

_fill_stage0!(::BoxFill, dst, lt::LevelTransfer, c::Int) =
    (dst .= view(lt.box_gather, :, :, :, c); dst)
_fill_stage0!(hf::HermiteFill, dst, lt::LevelTransfer, c::Int) =
    _hermite_box!(dst, lt, c, hf.θ, hf.dt)

# Device forms over this rank's components `owned`: the box's components
# upload in one contiguous copy; the Hermite blend reads the uploaded boxes.
function _fill_stage0_dev!(::BoxFill, stage0, scratch::LevelScratch,
                           lt::LevelTransfer, owned)
    h = Array(view(lt.box_gather, :, :, :, owned))
    if stage0 isa SubArray
        # A partial slice of the stage (the regrid fill's last chunk):
        # upload contiguously, then assign into the view.
        d = similar(parent(stage0), size(h))
        copyto!(d, h)
        stage0 .= d
    else
        copyto!(stage0, h)
    end
    return stage0
end

function _fill_stage0_dev!(hf::HermiteFill, stage0, scratch::LevelScratch,
                           lt::LevelTransfer, owned)
    T = eltype(stage0)
    θT = T(hf.θ)
    dtT = T(hf.dt)
    oneT = one(θT)
    h = ((oneT + T(2) * θT) * (oneT - θT)^2, θT * (oneT - θT)^2,
         θT^2 * (T(3) - T(2) * θT), θT^2 * (θT - oneT))
    box = lt.pdecomps[1]
    pad = box.n_halo_d
    nb = box.n_local
    n_owned = length(owned)
    pointwise!(_hermite_point!, stage0, nb[1], nb[2], nb[3] * n_owned,
               stage0, scratch.Q0, scratch.dQ0, scratch.Q1, scratch.dQ1,
               h, dtT, pad, nb[3], first(owned), step(owned))
    return stage0
end

# One node of one owned component of the Hermite blend; `kb` folds the
# third index and the component slot. The arithmetic is `_hermite_box!`'s.
@inline function _hermite_point!(dst, Q0, dQ0, Q1, dQ1, h, dtT, pad, n3,
                                 c1, cstep, i, j, kb)
    b, kk = divrem(kb - 1, n3)
    c = c1 + b * cstep
    @inbounds begin
        I = CartesianIndex(i + pad[1], j + pad[2], kk + 1 + pad[3])
        dst[I, b + 1] = h[1] * Q0[I, c] + h[3] * Q1[I, c] +
                        dtT * (h[2] * dQ0[I, c] + h[4] * dQ1[I, c])
    end
    return nothing
end

# The device interpolation of one chain stage over `n_owned` components:
# the `_inject_interpolate!` arithmetic of transfer.jl as a per-point body,
# one point per coarse node, its injection and its interval's two sub-nodes.
# A `box` of interior node ranges of the fine stage limits the launch to the
# coarse nodes whose intervals hold its nodes along the refined dimension and
# to its ranges along the other two; each node written takes the value the
# whole launch gives it.
function _interpolate_dev!(tmp, plan::TransferPlan{T}, coarse, n_owned::Int,
                           box=nothing) where {T}
    D = plan.dim
    padf = plan.fine.n_halo_d
    padc = plan.coarse.n_halo_d
    nc = plan.coarse.n_local[D]
    periodic = plan.coarse.periodic[D]
    p = plan.interp_order
    nintervals = periodic ? nc : nc - 1
    o1, o2 = D == 1 ? (2, 3) : D == 2 ? (1, 3) : (1, 2)
    r1 = box === nothing ? (1:plan.coarse.n_local[o1]) : box[o1]
    r2 = box === nothing ? (1:plan.coarse.n_local[o2]) : box[o2]
    rm = box === nothing ? (1:nc) :
                           (((first(box[D]) - 1) ÷ 3 + 1):((last(box[D]) - 1) ÷ 3 + 1))
    n2 = length(r2)
    W = (plan.weights...,)
    pointwise!(_interp_point!, tmp, length(rm), length(r1), n2 * n_owned,
               tmp, coarse, W, p, nc, periodic, nintervals, padf, padc, n2, D,
               (first(rm) - 1, first(r1) - 1, first(r2) - 1))
    return tmp
end

# The smallest box holding every box of `boxes`.
_bounding_box(boxes) =
    ntuple(d -> minimum(b -> first(b[d]), boxes):maximum(b -> last(b[d]), boxes), 3)

# Line coordinate `i` and orthogonal coordinates `j < k` along dimension
# `D`, as `_gidx` (operators.jl) maps them, with `D` a plain integer: a
# `Val` would be a type argument through the launcher's Vararg.
@inline function _gidx_dim(D::Int, i, j, k, pad)
    D == 1 && return CartesianIndex(i + pad[1], j + pad[2], k + pad[3])
    D == 2 && return CartesianIndex(j + pad[1], i + pad[2], k + pad[3])
    return CartesianIndex(j + pad[1], k + pad[2], i + pad[3])
end

@inline function _interp_point!(tmp, coarse, W, p, nc, periodic, nintervals,
                                padf, padc, n2, D, start, mi, ji, kb)
    b, kk = divrem(kb - 1, n2)
    m = start[1] + mi
    j = start[2] + ji
    k = start[3] + kk + 1
    b += 1
    half = p ÷ 2
    @inbounds begin
        tmp[_gidx_dim(D, 3m - 2, j, k, padf), b] = coarse[_gidx_dim(D, m, j, k, padc), b]
        if m <= nintervals
            js = periodic ? m - (half - 1) : clamp(m - (half - 1), 1, nc - p + 1)
            r = m - js
            for sub in 1:2
                acc = zero(eltype(tmp))
                for jj in 1:p
                    # W[jj, sub, r + 1] of the (p, 2, p − 1) table, column-major.
                    acc += W[jj + p * ((sub - 1) + 2r)] *
                           coarse[_gidx_dim(D, js + jj - 1, j, k, padc), b]
                end
                tmp[_gidx_dim(D, 3m - 2 + sub, j, k, padf), b] = acc
            end
        end
    end
    return nothing
end

# One ring entry of one owned component: the slab holding ring offset `at`
# (the slabs' offset ranges partition the ring, so exactly one matches)
# gives the shell node, read from the chain's final stage.
@inline function _ring_pack_point!(send, stage, table, shift, padb, ringlen,
                                   at, b, _k)
    @inbounds for (lo, hi, base) in table
        n1 = hi[1] - lo[1] + 1
        n2 = hi[2] - lo[2] + 1
        n3 = hi[3] - lo[3] + 1
        if base < at <= base + n1 * n2 * n3
            r = at - base - 1
            g1 = lo[1] + r % n1
            r ÷= n1
            g2 = lo[2] + r % n2
            g3 = lo[3] + r ÷ n2
            send[(b - 1) * ringlen + at] =
                stage[g1 + shift[1] + padb[1], g2 + shift[2] + padb[2],
                      g3 + shift[3] + padb[3], b]
        end
    end
    return nothing
end

# Upload the just-gathered Hermite endpoint boxes to the fine patch's
# scratch; a host patch, or a rank holding no piece of the fine patch, has
# nothing to upload.
function _upload_hermite!(lt::LevelTransfer, patches, at_end::Bool)
    lt.fine_index == 0 && return nothing
    scratch = patches[lt.fine_index].level_scratch
    isempty(scratch.stages) && return nothing
    copyto!(at_end ? scratch.Q1 : scratch.Q0, at_end ? lt.box_Q1 : lt.box_Q0)
    copyto!(at_end ? scratch.dQ1 : scratch.dQ0, at_end ? lt.box_dQ1 : lt.box_dQ0)
    return nothing
end

# Free the chain decompositions a discarded transfer owns. Both chains are
# built by `_refine_chain` on `COMM_SELF`, so these frees are rank-local, and
# no-ops today: a one-rank `COMM_SELF` decomposition borrows the communicator
# and owns nothing. The call stays so that the ownership rule lives in
# `Decomp` alone.
# The parent and fine patch decompositions are not stored on the transfer
# (only their `_owned_blocks` tables are) and are owned by their patches.
function free_transfer_decomps!(lt::LevelTransfer)
    for decomp in lt.pdecomps
        free_communicators!(decomp)
    end
    for decomp in lt.rdecomps
        free_communicators!(decomp)
    end
    return nothing
end

"""
    _owned_blocks(decomp, comm) -> Vector{BlockRegion}

One entry per rank of `comm`: that rank's owned interior block of the patch
decomposed as `decomp`, in the patch's node space and in `comm`'s rank
order. `decomp`
may be `nothing`, passed by a rank of `comm` that holds no block of the
patch: such a rank contributes a zero-extent region, which every consumer
intersects away, and receives the table like any other rank. Collective over
`comm` (one Allgather); `comm` must contain every rank that owns a block.
"""
function _owned_blocks(decomp::Union{Nothing,Decomp}, comm::MPI.Comm)
    np = MPI.Comm_size(comm)
    mine = decomp === nothing ? Int64[0, 0, 0, 0, 0, 0] :
           Int64[decomp.offset..., decomp.n_local...]
    flat = MPI.Allgather(mine, comm)
    return [BlockRegion((Int(flat[6r+1]), Int(flat[6r+2]), Int(flat[6r+3])),
                        (Int(flat[6r+4]), Int(flat[6r+5]), Int(flat[6r+6])))
            for r in 0:np-1]
end

"""
    build_level_transfer(T, region, active, n_halo, coarse_regions,
                         coarse_local, fine_index, restriction, n_cons,
                         subcycle, fine_decomp, parent_comm, np_tile, faces;
                         interpolation_order=6)

The [`LevelTransfer`](@ref) coupling one refined patch to its parents. Built
on every rank of the parent level's subset, the child's owners and the ranks
outside it alike, because the parent's side of the coupling exchange reads
the region, the faces and the local indices there. `coarse_local` is this
rank's solver index of each parent patch in `coarse_regions`, 0 for one it
holds no piece of; `fine_index` and `fine_decomp` are likewise 0 and
`nothing` on a rank holding no piece of the refined patch, where everything
only the patch's owners read (the chains, the box and Hermite storage, the
shell ring's staging) comes out empty. `parent_comm` is the parent level's
communicator, and `np_tile` the size of the refined patch's own
communicator, over which the shell ring distributes its components.
`boundary` flags the faces on the domain boundary, which take neither a box
buffer nor a shell. Rank-local: the traffic is planned level by level by
`build_level_coupling`.
"""
function build_level_transfer(::Type{T}, region::BlockRegion,
                              active::NTuple{3,Bool}, n_halo::Int,
                              coarse_regions::Vector{BlockRegion},
                              coarse_local::Vector{Int},
                              fine_index::Int, restriction::Symbol,
                              n_cons::Int, subcycle::Bool,
                              fine_decomp::Union{Nothing,Decomp{T}},
                              parent_comm::MPI.Comm, np_tile::Int,
                              faces::NTuple{3,NTuple{2,Int}}=ntuple(d -> (0, 0), 3);
                              interpolation_order::Int=6,
                              gradient_deriv=nothing,
                              parent_h=nothing,
                              boundary::NTuple{3,NTuple{2,Bool}}=_NO_BOUNDARY,
                              folded::NTuple{3,NTuple{2,Bool}}=_NO_BOUNDARY,
                              period::NTuple{3,Int}=(0, 0, 0)) where {T}
    imposed = ntuple(d -> (faces[d][1] == 0 && !boundary[d][1],
                           faces[d][2] == 0 && !boundary[d][2]), 3)
    dims_to_refine = [d for d in 1:3 if active[d]]
    buffer = _box_buffer(active, boundary, folded)
    boxext = _box_extent(region, buffer)
    held = fine_decomp !== nothing
    pdecomps, pplans, pstage = held ?
        _refine_chain(T, boxext, active, dims_to_refine, n_halo, interpolation_order) :
        (Decomp{T}[], TransferPlan{T}[], Array{T,3}[])
    # The restriction chain never interpolates (that half of each TransferPlan
    # goes unused), so it is built at interpolation order 2, which admits the
    # smallest legal regions. Only `:filter` restriction applies it; `:inject`
    # is realized directly as the coincident-node exchange. At a folded face
    # the chain's fine end is the patch's, the node beyond the lattice
    # included, and its filter folds there.
    rdecomps, rplans, rstage = held && restriction === :filter ?
        _refine_chain(T, region.extent, active, dims_to_refine, n_halo, 2, folded) :
        (Decomp{T}[], TransferPlan{T}[], Array{T,3}[])
    boxsize = held ? (size(pstage[1])..., n_cons) : (0, 0, 0, 0)
    hermite = subcycle ? boxsize : (0, 0, 0, 0)
    # The ring geometry follows from the region and the halo width alone; only
    # the patch's owners run `_impose_shell!`, so it is built there only.
    if fine_decomp === nothing
        slabs = NTuple{3,UnitRange{Int}}[]
        table, ringlen = _slab_table(slabs)
        shell = ShellRing{T}(slabs, table, ringlen, Matrix{T}(undef, 0, n_cons),
                             Matrix{T}(undef, 0, n_cons), Int[], GatherBuffers{T}(),
                             Vector{NTuple{3,UnitRange{Int}}}[])
    else
        slabs = _ring_slabs(region, ntuple(d -> fine_decomp.active[d], 3),
                            fine_decomp.n_halo_d, boundary, folded)
        table, ringlen = _slab_table(slabs)
        # Component-distributed chains: rank r owns components r+1, r+1+npf, ...
        ring_counts = [ringlen * length((r+1):np_tile:n_cons)
                       for r in 0:np_tile-1]
        fine_active = ntuple(d -> fine_decomp.active[d], 3)
        _check_ring_cover(table, region, fine_active, fine_decomp.n_halo_d, imposed,
                          boundary, folded)
        boxes = _chain_boxes(_ring_slabs(region, fine_active, fine_decomp.n_halo_d,
                                         boundary, folded, imposed),
                             _box_shift(buffer, folded),
                             _chain_extents(boxext, dims_to_refine), dims_to_refine,
                             interpolation_order)
        shell = ShellRing{T}(slabs, table, ringlen,
                             Matrix{T}(undef, ringlen, n_cons),
                             Matrix{T}(undef, ringlen, n_cons), ring_counts,
                             GatherBuffers{T}(), boxes)
    end
    # The gradients are taken on the chain's fine box, whose ends lie
    # 3·LEVEL_BUFFER fine nodes beyond the patch, with the derivative
    # operator's own closure rows there; only the owners of the refined patch
    # run the chain.
    gradients = nothing
    if gradient_deriv !== nothing && fine_decomp !== nothing
        plans, gdecomps = _box_gradient_plans(pdecomps[end], gradient_deriv, active,
                                              ntuple(d -> T(parent_h[d]) / 3, 3),
                                              fine_decomp.n_halo_d, imposed, buffer,
                                              folded)
        gradients = ShellGradients{T}(plans, gdecomps, zeros(T, size(pstage[end])),
                                      zeros(T, shell.len, 3 * n_cons))
    end
    return LevelTransfer{T}(region, active, coarse_regions, coarse_local,
                            fine_index, parent_comm, imposed, boundary, folded,
                            period, restriction, dims_to_refine,
                            pdecomps, pplans, pstage,
                            rdecomps, rplans, rstage,
                            zeros(T, hermite), zeros(T, hermite),
                            zeros(T, hermite), zeros(T, hermite),
                            zeros(T, boxsize),
                            zeros(T, held && restriction === :filter ?
                                     (region.extent..., n_cons) : (0, 0, 0, 0)),
                            shell, gradients)
end

# --- Replicated-region gathers ----------------------------------------------
#
# Every rank of a communicator assembles a whole node region: each
# contributes the intersection of its owned block with the region through
# one Allgatherv and unpacks every rank's contribution into its own replica.
# The per-step coupling moves point to point instead (`LevelCoupling`); the
# replica serves the regrid cadence only, where one region is carried whole
# (the box regrid's surviving state, the deep regrid's moved tile, the
# migration audit's reference).

# One-way counterpart of `_device_stage` (halo.jl): the gather packs and
# copies out, never in, so only the send half is allocated. `similar` keeps
# the result type inferable from the field, as the two-buffer form does.
@inline _device_send_stage(f, need::Int) = similar(f, eltype(f), need)

# Intersection of a node region with an owned block, as node ranges.
function _region_isect(region::NTuple{3,UnitRange{Int}}, block::BlockRegion)
    return ntuple(d -> max(first(region[d]), block.offset[d] + 1):
                       min(last(region[d]), block.offset[d] + block.extent[d]), 3)
end

"""
    gather_region!(dst, region_ranges, dst_off, dst_pad, Q, decomp, blocks,
                   sample=1; buffers=GatherBuffers{T}())
    gather_region!(dst, region_ranges, dst_off, dst_pad, Q, comm, blocks,
                   src_off, src_pad, sample=1; buffers=GatherBuffers{T}())

Assemble the node region `region_ranges` (in the patch's node space) of the
distributed field `Q` (padded, 4-D) into the replicated array `dst` on every
rank: node `n` lands at `dst[n - dst_off + dst_pad, ..., c]`, where `dst_off`
maps node space onto `dst`'s unpadded box. With `sample = s`, only nodes
`n ≡ 1 (mod s)` along active dimensions participate and land at
`dst[(n-1) ÷ s + 1 ...]`, the coincident-node form the `:inject` restriction
uses (`s = 3` per refined dimension).

The first form gathers over the field's own decomposition and is collective
over `decomp.comm`. The second names the communicator explicitly, with
`blocks` in its rank order and `src_off`/`src_pad` describing this rank's own
block; a rank of `comm` holding no block of the field passes `nothing` for `Q`
and contributes nothing. The regrid carries use this second form: the
ranks receiving a carried tile need not be the ones holding it.

`buffers` supplies the MPI staging; a caller at the regrid cadence can let
the default allocate. The per-step coupling does not replicate its regions
and moves point to point instead ([`LevelCoupling`](@ref)).
"""
gather_region!(dst::AbstractArray{T,4},
               region_ranges::NTuple{3,UnitRange{Int}},
               dst_off::NTuple{3,Int}, dst_pad::NTuple{3,Int},
               Q, decomp::Decomp{T}, blocks::Vector{BlockRegion},
               sample::NTuple{3,Int}=(1, 1, 1);
               buffers::GatherBuffers{T}=GatherBuffers{T}()) where {T} =
    gather_region!(dst, region_ranges, dst_off, dst_pad, Q, decomp.comm, blocks,
                   decomp.offset, decomp.n_halo_d, sample; buffers=buffers)

function gather_region!(dst::AbstractArray{T,4},
                        region_ranges::NTuple{3,UnitRange{Int}},
                        dst_off::NTuple{3,Int}, dst_pad::NTuple{3,Int},
                        Q, comm::MPI.Comm, blocks::Vector{BlockRegion},
                        src_off::NTuple{3,Int}, src_pad::NTuple{3,Int},
                        sample::NTuple{3,Int}=(1, 1, 1);
                        buffers::GatherBuffers{T}=GatherBuffers{T}()) where {T}
    np = MPI.Comm_size(comm)
    me = MPI.Comm_rank(comm)
    n_cons = size(dst, 4)
    pad = src_pad
    # A sampled gather keeps nodes n with (n − 1) % sample == 0.
    keep(r, d) = first(r) + mod(sample[d] - mod(first(r) - 1, sample[d]),
                                sample[d]):sample[d]:last(r)
    isects = [ntuple(d -> keep(_region_isect(region_ranges, blocks[r+1])[d], d), 3)
              for r in 0:np-1]
    counts = [n_cons * prod(max(length(ir[d]), 0) for d in 1:3)
              for ir in isects]
    sendbuf = _fit!(buffers.send, counts[me+1])
    mine = isects[me+1]
    if counts[me+1] == 0
        # Nothing of this rank's own is wanted, `Q` may be absent, and the
        # packs below would read it: the Allgatherv still runs, with an empty
        # contribution.
    elseif !_device_path(Q)
        idx = 1
        @inbounds for c in 1:n_cons, k in mine[3], j in mine[2], i in mine[1]
            sendbuf[idx] = Q[i - src_off[1] + pad[1],
                             j - src_off[2] + pad[2],
                             k - src_off[3] + pad[3], c]
            idx += 1
        end
    else
        # Device storage packs by broadcast into a contiguous stage (the
        # same strided-to-contiguous move the halo staging makes, with the
        # sampled ranges expressed as strided views), and one contiguous
        # device-to-host copy fills the MPI buffer. Column-major broadcast
        # order matches the scalar pack's (c slowest, i fastest).
        lr = ntuple(d -> (first(mine[d]) - src_off[d] + pad[d]):sample[d]:
                         (last(mine[d]) - src_off[d] + pad[d]), 3)
        v = view(parent(Q), lr[1], lr[2], lr[3], 1:n_cons)
        dsend = _device_send_stage(parent(Q), counts[me+1])
        reshape(view(dsend, 1:counts[me+1]), size(v)) .= v
        _tracked_copy!(sendbuf, 1, dsend, 1, counts[me+1])
    end
    recvbuf = _fit!(buffers.recv, sum(counts))
    MPI.Allgatherv!(sendbuf, MPI.VBuffer(recvbuf, counts), comm)
    pos = 0
    for r in 0:np-1
        ir = isects[r+1]
        @inbounds for c in 1:n_cons, k in ir[3], j in ir[2], i in ir[1]
            pos += 1
            dst[(i - 1) ÷ sample[1] + 1 - dst_off[1] + dst_pad[1],
                (j - 1) ÷ sample[2] + 1 - dst_off[2] + dst_pad[2],
                (k - 1) ÷ sample[3] + 1 - dst_off[3] + dst_pad[3], c] =
                recvbuf[pos]
        end
    end
    return dst
end

"Fine extent of a refined region: 3m − 2 nodes per active dimension."
fine_extent(region::BlockRegion, active::NTuple{3,Bool}) =
    ntuple(d -> active[d] ? 3 * region.extent[d] - 2 : region.extent[d], 3)

# The same with the node a folded face adds beyond the coincident lattice
# (`_fold_lead`), and the fine patch of a region in the refined level's node
# space: parent node g is fine node 3(g − 1) + 1, the folded low face's
# extra node fine node 3(g − 1).
fine_extent(region::BlockRegion, active::NTuple{3,Bool},
            folded::NTuple{3,NTuple{2,Bool}}) =
    ntuple(d -> fine_extent(region, active)[d] + _fold_lead(folded, d, 1) +
                _fold_lead(folded, d, 2), 3)
_fine_region(region::BlockRegion, active::NTuple{3,Bool},
             folded::NTuple{3,NTuple{2,Bool}}=_NO_BOUNDARY) =
    BlockRegion(ntuple(d -> active[d] ? 3 * region.offset[d] - _fold_lead(folded, d, 1) :
                            0, 3),
                fine_extent(region, active, folded))
_fine_region(lt::LevelTransfer) = _fine_region(lt.region, lt.active, lt.folded)

# `region` grown by `margin` nodes per side along active dimensions, except
# at a face on the domain boundary (`boundary`), which a level reaches
# without a margin.
_buffered(region::BlockRegion, active::NTuple{3,Bool}, margin::Int,
          boundary::NTuple{3,NTuple{2,Bool}}=_NO_BOUNDARY) =
    BlockRegion(ntuple(d -> region.offset[d] -
                            (active[d] && !boundary[d][1] ? margin : 0), 3),
                ntuple(d -> region.extent[d] +
                            (active[d] && !boundary[d][1] ? margin : 0) +
                            (active[d] && !boundary[d][2] ? margin : 0), 3))

# --- Levels on the domain boundary ---------------------------------------------
#
# A refined region may reach a non-periodic face of the domain. The tile's
# face there is not fed from the parent: it carries the root's own condition
# at the fine spacing (`_fine_bcs`), closes its line solves with that
# condition's rows as the root does (`_fine_plans`), and is left out of the
# shell (`_in_shell`), the ring (`_ring_slabs`) and the restriction margin
# (`_restrict_window`), since its boundary plane is the tile's own solution.
# The buffered box stops at the face (`_box_buffer`), where the Lagrange
# interpolation takes its one-sided stencils. The nesting margin applies to
# the parent-fed faces only (`_buffered`). The faces a level may reach carry
# a condition in `_level_boundary_condition`; a periodic seam and any other
# condition keep the margin.
#
# A symmetry plane lies half a root cell outside the root's first node, so
# at ratio 3 the fine nodes nearest it sit at h/6 and h/2: the tile takes one
# fine node beyond the coincident lattice there (`_fold_lead`), which puts
# its own first node half a fine cell from the plane, and carries the root's
# face-centred fold at the fine spacing (`_fine_plans`). The box keeps its
# buffer across the plane, filled with the parity mirror of the parent's
# nodes (`_mirror_folded_box!`), so the Lagrange chain stays centred there
# and interpolates the extra node. The setup places a level on the plane, and
# so does the regrid under `level_boundaries`, carrying the extra node through
# the box carry, the tile migration and the restart. A level below the first
# does the same one level down: its parent's node nearest the plane lies half
# a parent spacing from it, outside the lattice coincident with the root's,
# so the parent's node space extends beyond that lattice (`_level_span`) and
# a region reaching the plane starts at a negative offset in it, −1 for the
# second level; the tile again takes one node beyond its parent's lattice.
#
# The axis of a θ-collapsed r-z run is the same fold with other parities.
# Each radial line continues into itself through r = 0, so the fold is
# self-paired, and the root's node nearest the axis lies at h/2, which puts
# the tile's nodes at h/6 + (j − 1)h/3, those of a uniform axis run at h/3.
# The radial velocity, the swirl and the area factor A₁ = r are odd there;
# the root fold's `sigvel` and `sigflux` carry these signs to the tile's fold
# and to the box mirror. The metric needs nothing beyond the per-tile
# geometry: `init_geometry!` evaluates r, 1/r and the face areas at the
# tile's own nodes. A uniform state stays uniform: the radial momentum takes
# its pressure term as ∂p/∂r, which every row gives as zero on a constant,
# and the rest of its flux vanishes at rest.

"Whether a refined level may reach a domain face carrying `bc`."
_level_boundary_condition(bc) =
    bc isa Union{SlipWallBC,NoSlipWallBC,SymmetryPlaneBC,AxisBC,NSCBCOutflowBC,
                 NSCBCInflowBC}

# Whether a tile reaching a face carrying `bc` folds there: a symmetry plane,
# and the axis of a θ-collapsed run, the only axis a refined run carries.
_level_fold_condition(bc) = bc isa Union{SymmetryPlaneBC,AxisBC}

# Per dimension and side, whether a level may reach that domain face: a
# non-periodic active dimension whose root condition there qualifies.
_level_boundary_eligible(bcs, active::NTuple{3,Bool}, periodic::NTuple{3,Bool}) =
    ntuple(d -> ntuple(s -> active[d] && !periodic[d] &&
                            _level_boundary_condition(bcs[d][s]), 2), 3)

# The boundary faces among `boundary` whose root condition is a fold.
_level_fold_faces(boundary::NTuple{3,NTuple{2,Bool}}, bcs) =
    ntuple(d -> ntuple(s -> boundary[d][s] && _level_fold_condition(bcs[d][s]), 2), 3)

# The fine nodes a tile takes beyond its coincident lattice at face `side` of
# dimension `d`: one at a folded face, none elsewhere.
@inline _fold_lead(folded::NTuple{3,NTuple{2,Bool}}, d::Int, side::Int) =
    folded[d][side] ? 1 : 0

# Fill the part of a delivered box beyond a folded face with the parity mirror
# of the parent's nodes inside it, for every transfer of `lev` whose tile this
# rank holds: with the parent's node space spanning `lo:hi` along the
# dimension (`_level_span`), parent node lo − n reads node lo + n − 1 at a low
# fold and hi + n reads hi + 1 − n at a high one, each component signed by
# its parity across the fold, a plane's or the axis's, which are the root
# fold's at every level. `A` is `select(transfer)`, padded like the chain's
# stage 0. A tile at a corner of two folds takes the dimensions in turn, the
# second mirroring what the first filled.
function _mirror_folded_box!(solver, lev::Level, select::F) where {F}
    root = getfield(solver, :patches)[1]
    any(lt -> lt.fine_index != 0 && any(any, lt.folded), lev.transfers) || return lev
    span = _level_span(solver.n_global, ntuple(d -> solver.n_global[d] > 1, 3),
                       lev.index - 1, root.bcs)
    for lt in lev.transfers
        lt.fine_index == 0 && continue
        any(any, lt.folded) || continue
        A = select(lt)
        isempty(A) && continue
        box = _box_nodes(lt)
        pad = lt.pdecomps[1].n_halo_d
        for d in 1:3, side in 1:2
            lt.folded[d][side] || continue
            fold = root.folds[d]
            lo, hi = first(span[d]), last(span[d])
            at(n) = n - first(box[d]) + 1 + pad[d]
            for c in axes(A, 4)
                σ = conserved_parity(solver.equations, fold.sigvel, c)
                for m in 1:LEVEL_BUFFER
                    dst, src = side == 1 ? (at(lo - m), at(lo + m - 1)) :
                                           (at(hi + m), at(hi + 1 - m))
                    sl(i) = ntuple(e -> e == d ? (i:i) : axes(A, e), 3)
                    view(A, sl(dst)..., c) .= σ .* view(A, sl(src)..., c)
                end
            end
        end
    end
    return lev
end

# The node count of level ℓ's node space along each dimension, the root's
# being `n_global`; along a periodic dimension no face lies on the boundary
# and the count is not used.
_level_extent(n_global::NTuple{3,Int}, active::NTuple{3,Bool}, ℓ::Int) =
    ntuple(d -> active[d] ? 3^ℓ * (n_global[d] - 1) + 1 : 1, 3)

# The nodes of level ℓ's node space along each dimension, the root's being
# 1:n_global. Level-ℓ node n lies n − 1 level spacings from the root's first
# node, so the nodes coincident with the root's run from 1 to 3^ℓ(N − 1) + 1
# (`_level_extent`). At a face carrying a fold (`_level_fold_condition`) the
# root's first node lies half a root spacing from the fold, and the space
# extends by (3^ℓ − 1)/2 nodes to the node half a level spacing from it: a
# region reaching the fold starts, or ends, there. At the first refined level
# that is the one node `_fold_lead` counts, node 0 at a low fold; at the
# second, level-1 node 0 is level-2 node −2 and the space starts at −3.
function _level_span(n_global::NTuple{3,Int}, active::NTuple{3,Bool}, ℓ::Int, bcs)
    ext = (3^ℓ - 1) ÷ 2
    return ntuple(3) do d
        active[d] || return 1:1
        lo = _level_fold_condition(bcs[d][1]) ? 1 - ext : 1
        hi = 3^ℓ * (n_global[d] - 1) + 1 + (_level_fold_condition(bcs[d][2]) ? ext : 0)
        lo:hi
    end
end

# --- Levels across a periodic seam ---------------------------------------------
#
# Along a periodic dimension of N root nodes, root node N + 1 is node 1, and
# level ℓ's node space repeats every 3^ℓ N nodes (parent node g is fine node
# 3(g − 1) + 1, so g + N is fine node 3(g − 1) + 1 + 3N). A region there may
# extend past the last node and wrap: its offset lies in [0, P) and its
# nodes past P are the nodes a period lower. Every relation between a level
# and its parent (the box pieces and the covered samples of the coupling,
# the parents under a box, the covered mask, the nesting test) and between
# two tiles of a level (the shared plane of a same-level interface) is taken
# over the periodic images of the parent's or the partner's region, the
# region shifted by −P, 0 and +P, so the patches themselves keep contiguous
# node spaces: a tile across the seam is one block whose coordinates run on
# past the domain's face, and the seam is nothing but where its parent's
# data come from. A region clear of the seam meets only the unshifted image,
# so nothing moves for it.

# The period of level ℓ's node space along each dimension: 3^ℓ N along an
# active periodic dimension of the root, 0 along the others. `periodic` is
# the root decomposition's, which marks collapsed dimensions periodic too.
_level_period(n_global::NTuple{3,Int}, periodic::NTuple{3,Bool}, ℓ::Int) =
    ntuple(d -> n_global[d] > 1 && periodic[d] ? 3^ℓ * n_global[d] : 0, 3)
_level_period(solver, ℓ::Int) =
    _level_period(solver.n_global, getfield(solver, :patches)[1].decomp.periodic, ℓ)

# The shifts of a region's periodic images under `period`: the region itself
# first, then ∓P along each periodic dimension, their combinations included.
# Built from vectors of one type whatever the period, so the result infers
# concretely (the ledger reads it inside the step).
function _images(period::NTuple{3,Int})
    shifts(P) = P == 0 ? [0] : [0, -P, P]
    out = NTuple{3,Int}[]
    for s3 in shifts(period[3]), s2 in shifts(period[2]), s1 in shifts(period[1])
        push!(out, (s1, s2, s3))
    end
    return out
end

# `region` shifted by `σ`.
_shifted(region::BlockRegion, σ::NTuple{3,Int}) =
    BlockRegion(region.offset .+ σ, region.extent)

# `region` with its offset along each periodic dimension brought into
# [0, P), the form every level stores.
_canonical(region::BlockRegion, period::NTuple{3,Int}) =
    BlockRegion(ntuple(d -> period[d] == 0 ? region.offset[d] :
                            mod(region.offset[d], period[d]), 3), region.extent)

# The node intervals of [lo, hi] inside [1, P + 1] along a periodic
# dimension of period P, for an interval that may run past either end of it
# but spans less than a period: the interval itself, or its two pieces on
# either side of the seam, one ending on node P + 1 and one starting on node 1
# (the same node, the plane both share). A piece that would hold that node
# alone is dropped: an interval ending on a lattice plane meets the cell on
# its inner side only (`_tile_span`).
function _wrapped_parts(lo::Int, hi::Int, P::Int)
    P == 0 && return [(lo, hi)]
    lo < 1 && return hi > 1 ? [(lo + P, P + 1), (1, hi)] :
                     hi == 1 ? [(lo + P, P + 1)] : [(lo + P, hi + P)]
    hi > P + 1 && return lo < P + 1 ? [(lo, P + 1), (1, hi - P)] :
                         lo == P + 1 ? [(1, hi - P)] : [(lo - P, hi - P)]
    return [(lo, hi)]
end
# The faces of `region` (in a level's node space of `extent` nodes, or over
# the node ranges `span`) that lie on a domain face a level may reach
# (`eligible`).
_boundary_faces(region::BlockRegion, extent::NTuple{3,Int}, eligible) =
    _boundary_faces(region, ntuple(d -> 1:extent[d], 3), eligible)
_boundary_faces(region::BlockRegion, span::NTuple{3,UnitRange{Int}}, eligible) =
    ntuple(d -> (eligible[d][1] && region.offset[d] + 1 == first(span[d]),
                 eligible[d][2] && region.offset[d] + region.extent[d] == last(span[d])), 3)

# The boundary faces of each of `regions`, level-ℓ tiles given in level
# ℓ − 1's node space, from the root patch's conditions: the form the regrid
# and the restart take, which hold the solver rather than its keywords.
function _region_boundaries(solver, regions::AbstractVector{BlockRegion}, ℓ::Int)
    root = getfield(solver, :patches)[1]
    n_global = solver.n_global
    active = ntuple(d -> n_global[d] > 1, 3)
    eligible = _level_boundary_eligible(root.bcs, active, root.decomp.periodic)
    span = _level_span(n_global, active, ℓ - 1, root.bcs)
    return [_boundary_faces(r, span, eligible) for r in regions]
end

# The folded faces among each of `boundaries`, from the root patch's
# conditions: what a tile built by a regrid or a restart takes beyond its
# coincident lattice (`_fold_lead`).
_region_folds(solver, boundaries::AbstractVector) =
    [_level_fold_faces(b, getfield(solver, :patches)[1].bcs) for b in boundaries]

# The node count of a level-ℓ tile over `region` (level ℓ − 1's node space),
# with the node each of its folded faces adds.
_tile_fine_extent(solver, region::BlockRegion, ℓ::Int) =
    fine_extent(region, ntuple(d -> solver.n_global[d] > 1, 3),
                only(_region_folds(solver, _region_boundaries(solver, [region], ℓ))))

# Whether every node of `region` lies in some member of `regions` (a union
# of boxes, not one box: the test is by node, and regions are small), or in
# a periodic image of one under `period`.
function _covered_by(region::BlockRegion, regions::Vector{BlockRegion},
                     period::NTuple{3,Int}=(0, 0, 0))
    images = [_shifted(r, σ) for r in regions for σ in _images(period)]
    inside(g, r) = all(d -> r.offset[d] < g[d] <= r.offset[d] + r.extent[d], 1:3)
    for k in 1:region.extent[3], j in 1:region.extent[2], i in 1:region.extent[1]
        g = region.offset .+ (i, j, k)
        any(r -> inside(g, r), images) || return false
    end
    return true
end

"""
    _amr_dims(extent, active, np) -> NTuple{3,Int}

Process grid for the fine patch: a factorization of `np` over the active
dimensions whose smallest local block stays at or above the C8 filter's
9-point minimum, preferring the factorization with the largest smallest
block. `MPI.Dims_create` cannot be used here because it lacks the
scheme minimum, and a regrid that picked an infeasible grid would kill a run
mid-flight; this errors with the actual numbers instead, at setup or at the
regrid that shrank the region.

Refined-level construction sizes its rank subset first
([`_level_ranks`](@ref)), so from that path only an `np` the search below
admits reaches this method.
"""
function _amr_dims(extent::NTuple{3,Int}, active::NTuple{3,Bool}, np::Int)
    dims = _amr_dims_or_nothing(extent, active, np)
    dims === nothing &&
        error("no process grid over $np rank(s) gives every rank ≥ 9 fine " *
              "points per split dimension of a $extent fine patch; use " *
              "fewer ranks or a larger refined region")
    return dims
end

# The search itself, returning `nothing` where no factorization qualifies:
# `_amr_dims` raises on that, `_level_ranks` reads it as "this rank count does
# not fit" and tries a smaller one.
function _amr_dims_or_nothing(extent::NTuple{3,Int}, active::NTuple{3,Bool},
                              np::Int)
    best = (0, 0, 0)
    bestmin = -1
    for p1 in 1:np
        np % p1 == 0 || continue
        for p2 in 1:(np ÷ p1)
            (np ÷ p1) % p2 == 0 || continue
            p3 = np ÷ (p1 * p2)
            dims = (p1, p2, p3)
            ok = true
            small = typemax(Int)
            for d in 1:3
                if !active[d]
                    dims[d] == 1 || (ok = false)
                else
                    blk = extent[d] ÷ dims[d]
                    blk >= 9 || (ok = false)
                    dims[d] > 1 && (small = min(small, blk))
                end
            end
            ok || continue
            small == typemax(Int) && (small = minimum(
                extent[d] for d in 1:3 if active[d]; init=extent[1]))
            if small > bestmin
                bestmin = small
                best = dims
            end
        end
    end
    bestmin < 0 && return nothing
    return best
end

"""
    _level_ranks(regions, active, np) -> Int

The rank count of a refined level: the largest count at or below `np` for
which every one of its `regions` (parent-level node space) admits a process
grid under [`_amr_dims`](@ref)'s 9-point minimum. The level's owners are the
first that many ranks of the parent level's communicator
([`split_level_comm`](@ref)).

The search runs downward and stops at the first count that fits, so a level
large enough for the whole rank set returns `np` itself. One rank always
fits: a region spans at least four parent nodes, hence ten fine ones.
Feasibility is not monotone in the rank count (27 fine nodes fit three ranks
and not four), so the search is a linear scan rather than a bisection. It
starts at the largest count every region can admit, the least over the
regions of the product over the active dimensions of ⌊extent/9⌋, since a
split dimension must give every rank nine nodes: `_tile_owners` sizes every
tile of a level this way, and a scan from `np` would cost the rank count
squared per tile at a cluster's rank count.
"""
function _level_ranks(regions::Vector{BlockRegion}, active::NTuple{3,Bool},
                      np::Int)
    admitted = minimum(regions; init=np) do r
        ext = fine_extent(r, active)
        prod(ext[d] > 1 ? max(ext[d] ÷ 9, 1) : 1 for d in 1:3)
    end
    for p in min(np, admitted):-1:1
        all(r -> _admits(r, active, p), regions) && return p
    end
    return 1
end

# Whether a refined patch over `region` (parent-level node space) admits a
# process grid of `np` ranks under the 9-point minimum. The extent is the
# coincident lattice's; a folded face adds a node, which keeps every block
# at least as large, so a count admitted here is admitted by the folded tile.
function _admits(region::BlockRegion, active::NTuple{3,Bool}, np::Int)
    ext = fine_extent(region, active)
    return _amr_dims_or_nothing(ext, ntuple(d -> ext[d] > 1, 3), np) !== nothing
end

# --- Prolongation: coarse state → fine ghost ring and boundary planes -------
#
# The buffered box lies inside the union of the parent patches by the
# nesting margin, so every value a tile's box needs exists on some parent
# rank; `_exchange_boxes!` delivers it.

function _write_fine_shell!(fine_Q, c::Int, box_field, lt::LevelTransfer,
                            df::Decomp, boxf::Decomp, shell_only::Bool=true)
    # Fine patch node g ↔ fine box node g + 3·LEVEL_BUFFER (active dims). The
    # shell is every padded slot whose patch-global index lies outside the
    # strict interior [2, N−1] along an active dimension whose face on that
    # side is parent-fed (`lt.imposed`): the ghost ring plus the boundary
    # planes, both imposed from the prolonged parent state; a face shared
    # with a same-level tile is left to the level's records. A decomposed
    # rank writes only its own padded slots, so an interior rank writes
    # nothing under `shell_only` and its rank-boundary halos keep the
    # exchanged neighbor values. `shell_only = false` writes every slot
    # instead: the whole-patch initialization a regrid performs on a freshly
    # created fine region.
    padf = df.n_halo_d
    padb = boxf.n_halo_d
    nf = df.n_local
    off = df.offset
    Nf = fine_extent(lt.region, ntuple(d -> df.active[d], 3), lt.folded)
    shift = _box_shift(lt)
    active = (df.active[1], df.active[2], df.active[3])
    imposed = lt.imposed
    boundary = lt.boundary
    if !_device_path(fine_Q)
        r = ntuple(d -> (1 - padf[d]):(nf[d] + padf[d]), 3)
        @inbounds for k in r[3], j in r[2], i in r[1]
            g1, g2, g3 = i + off[1], j + off[2], k + off[3]
            shell_only && !_in_shell(g1, g2, g3, Nf, active, imposed, boundary) &&
                continue
            fine_Q[i + padf[1], j + padf[2], k + padf[3], c] =
                box_field[g1 + shift[1] + padb[1], g2 + shift[2] + padb[2],
                          g3 + shift[3] + padb[3]]
        end
        return fine_Q
    end
    # Device patch: a host box (the chain product of a host-only path)
    # uploads once; the device chain's stage is read in place.
    dev_box = if _cpu_storage(box_field)
        d = similar(parent(fine_Q), size(box_field))
        copyto!(d, box_field)
        d
    else
        box_field
    end
    pointwise!(_fine_shell_point!, fine_Q,
               nf[1] + 2 * padf[1], nf[2] + 2 * padf[2], nf[3] + 2 * padf[3],
               fine_Q, dev_box, c, off, padf, padb, shift, Nf,
               active, imposed, boundary, shell_only)
    return fine_Q
end

@inline function _fine_shell_point!(fine_Q, box_field, c, off, padf, padb,
                                    shift, Nf, active, imposed, boundary,
                                    shell_only, i, j, k)
    @inbounds begin
        g1 = i - padf[1] + off[1]
        g2 = j - padf[2] + off[2]
        g3 = k - padf[3] + off[3]
        if !shell_only || _in_shell(g1, g2, g3, Nf, active, imposed, boundary)
            fine_Q[i, j, k, c] =
                box_field[g1 + shift[1] + padb[1], g2 + shift[2] + padb[2],
                          g3 + shift[3] + padb[3]]
        end
    end
    return nothing
end

# --- Component-distributed shell imposition ---------------------------------
#
# The interpolation chain, not the physics, is the expensive half of the
# coupling: a subcycled step imposes the shell ~20 times (five coarse stages
# plus every fine stage's Hermite shell), each a K-stage tensor-product
# interpolation over the whole buffered box per conserved component, and a
# replicated chain repeats all of it on every rank. Measured on the 3-D cost
# case at np = 8, that put the composite at 85% of the uniform-fine wall.
# The chains therefore distribute by component: rank r runs the chain only
# for components c with (c − 1) mod np == r, packs the thin shell ring of its
# results (2 slabs of thickness pad+1 per active dimension, full transverse
# extent, in patch-padded node space), and one Allgatherv replicates the
# rings; every rank then writes its own shell slots from the ring. The chain
# work per rank drops by ~min(np, n_cons)×, and the collective carries the
# ring, not the box. Values are bit-identical to the replicated form: the
# same chain output moves through a pack/unpack without recomputation.

# Ring slabs in patch-padded node space, ascending dimension order, low side
# then high per active dimension. Slabs overlap at corners; both copies of a
# corner value come from the same chain output, so the first match wins. A
# face on the domain boundary (`boundary`) has no slab, and the other slabs
# stop at its plane, since the box holds nothing beyond it. `faces` selects
# the faces whose slabs are returned, every one by default.
function _ring_slabs(region::BlockRegion, active::NTuple{3,Bool},
                     pad::NTuple{3,Int},
                     boundary::NTuple{3,NTuple{2,Bool}}=_NO_BOUNDARY,
                     folded::NTuple{3,NTuple{2,Bool}}=_NO_BOUNDARY,
                     faces::NTuple{3,NTuple{2,Bool}}=((true, true), (true, true),
                                                      (true, true)))
    Nf = fine_extent(region, active, folded)
    full = ntuple(d -> (boundary[d][1] ? 1 : 1 - pad[d]):
                       (boundary[d][2] ? Nf[d] : Nf[d] + pad[d]), 3)
    slabs = NTuple{3,UnitRange{Int}}[]
    for d in 1:3
        active[d] || continue
        boundary[d][1] || !faces[d][1] ||
            push!(slabs, ntuple(q -> q == d ? ((1 - pad[d]):1) : full[q], 3))
        boundary[d][2] || !faces[d][2] ||
            push!(slabs, ntuple(q -> q == d ? (Nf[d]:(Nf[d] + pad[d])) : full[q], 3))
    end
    return slabs
end

# The boxes of each stage of the interpolation chain that the shell slots
# depend on, as interior node ranges of that stage, `boxes[k]` for the output
# of plan `k`. A shell slot lies in the slab of a parent-fed face (`_in_shell`),
# so the chain's last stage is needed over those slabs alone, `slabs`, carried
# into the box by `shift`; each earlier stage over the coarse nodes the
# Lagrange stencils of the next one read, which along the dimension plan `k`
# refines are the injected node of a coincident fine node and the `order`
# nodes of an intermediate one's stencil (`_inject_interpolate!`). `exts` are
# the stages' extents (`_chain_extents`) and `dims` the dimension each plan
# refines. A stage whose boxes add up to no fewer nodes than it holds is
# taken whole.
function _chain_boxes(slabs, shift::NTuple{3,Int}, exts, dims::Vector{Int},
                      order::Int)
    K = length(dims)
    boxes = [NTuple{3,UnitRange{Int}}[] for _ in 1:K]
    whole(k) = ntuple(d -> 1:exts[k+1][d], 3)
    inside(b, k) = all(d -> 1 <= first(b[d]) && last(b[d]) <= exts[k+1][d], 1:3)
    for s in slabs
        b = ntuple(d -> (first(s[d]) + shift[d]):(last(s[d]) + shift[d]), 3)
        for k in K:-1:1
            # The box holds every shell slot; a slab reaching beyond it would
            # break that invariant, and the whole chain is taken instead.
            inside(b, k) || return [[whole(k)] for k in 1:K]
            push!(boxes[k], b)
            D = dims[k]
            nc = exts[k][D]
            lo, hi = typemax(Int), typemin(Int)
            for n in b[D]
                m = (n - 1) ÷ 3 + 1
                if (n - 1) % 3 == 0
                    lo, hi = min(lo, m), max(hi, m)
                else
                    js = clamp(m - (order ÷ 2 - 1), 1, nc - order + 1)
                    lo, hi = min(lo, js), max(hi, js + order - 1)
                end
            end
            b = ntuple(d -> d == D ? (lo:hi) : b[d], 3)
        end
    end
    for k in 1:K
        sum(b -> prod(length.(b)), boxes[k]; init=0) >= prod(length.(whole(k))) &&
            (boxes[k] = [whole(k)])
    end
    return boxes
end

# Isbits slab table for the writers: (lo, hi, zero-based ring offset) per slab.
function _slab_table(slabs)
    table = Tuple{NTuple{3,Int},NTuple{3,Int},Int}[]
    base = 0
    for s in slabs
        push!(table, (first.(s), last.(s), base))
        base += prod(length.(s))
    end
    return table, base
end

# One-based ring offset of shell node (g1, g2, g3), or 0 when no slab holds
# it. Invariant: the slabs of `_ring_slabs` cover every padded slot whose
# patch-global index lies outside the strict interior, matching the shell test
# both writers apply. A shell slot therefore always matches. Both writers
# visit the slabs' nodes rather than the patch's, so a slot outside every slab
# would be skipped, not misread; `_check_ring_cover` raises at setup if the
# slab set and the shell test have fallen out of step.
@inline function _ring_offset(table, g1, g2, g3)
    for (lo, hi, base) in table
        if lo[1] <= g1 <= hi[1] && lo[2] <= g2 <= hi[2] && lo[3] <= g3 <= hi[3]
            n1 = hi[1] - lo[1] + 1
            n2 = hi[2] - lo[2] + 1
            return base + 1 + (g1 - lo[1]) +
                   n1 * ((g2 - lo[2]) + n2 * (g3 - lo[3]))
        end
    end
    return 0
end

# Whether some active face of the transfer's patch takes its shell from the
# parent.
_imposes_shell(lt::LevelTransfer) = any(d -> lt.active[d] && any(lt.imposed[d]), 1:3)

@noinline _ring_miss(g1, g2, g3) =
    error("fine shell node ($g1, $g2, $g3) lies in no ring slab; the slab " *
          "set and the shell test disagree")

# Raise unless every shell slot of the fine patch over `region`, with halo
# `pad`, lies in a slab of `table`: the invariant both shell writers rest on.
# One pass over the padded patch when a transfer is built.
function _check_ring_cover(table, region::BlockRegion, active::NTuple{3,Bool},
                           pad::NTuple{3,Int}, imposed, boundary, folded)
    Nf = fine_extent(region, active, folded)
    r = ntuple(d -> (1 - pad[d]):(Nf[d] + pad[d]), 3)
    for g3 in r[3], g2 in r[2], g1 in r[1]
        _in_shell(g1, g2, g3, Nf, active, imposed, boundary) || continue
        _ring_offset(table, g1, g2, g3) == 0 && _ring_miss(g1, g2, g3)
    end
    return nothing
end

# Shared driver: run the chain for this rank's components with `fill`
# (a `BoxFill` or a `HermiteFill`) supplying stage 0, replicate the shell
# rings, and impose each rank's own shell slots. Collective over the fine
# communicator. A device patch runs the chain on its `LevelScratch`; the
# host patch on the transfer's host stages.
#
# Under `lt.gradients` each component's ring is followed by the rings of its
# derivatives along the three dimensions (zero along a collapsed one), taken
# on the fine box the chain produced, and the unpack fills the gradient ring
# beside the shell ring. A device patch takes the derivatives through its
# scratch's device plans and packs and unpacks both rings on the device, so
# the box stays there; only the Allgatherv of a decomposed patch stages the
# packed rings through the host. A tile with no parent-fed face reads no
# gradient ring and takes none.
function _impose_shell!(solver, states, lt::LevelTransfer, fill)
    # A tile whose every face is shared with a same-level tile or lies on the
    # domain boundary has no shell slot to write (`_in_shell`), and no reader
    # of the ring, so the chain is skipped. `imposed` is a property of the
    # transfer, the same on every rank of the tile's communicator, so the
    # ranks skip the ring gather together.
    _imposes_shell(lt) || return states
    lt.gradients === nothing || return _impose_shell_gradients!(solver, states, lt, fill)
    patches = getfield(solver, :patches)
    fine = patches[lt.fine_index]
    Qf = states[lt.fine_index]
    fdcp = fine.decomp
    comm = fdcp.comm
    np = MPI.Comm_size(comm)
    me = MPI.Comm_rank(comm)
    n_cons = solver.equations.n_cons
    K = length(lt.pplans)
    shell = lt.shell
    slabs = shell.slabs
    table = shell.table
    ringlen = shell.len
    ring = shell.ring
    padb = lt.pdecomps[K+1].n_halo_d
    shift = _box_shift(lt)
    # This rank's components, as a range, not a filtered vector: the
    # ascending order is the one the ring unpack below assumes.
    owned = (me+1):np:n_cons
    n_owned = length(owned)
    # The admissible fallback (`_shell_fallback`). On one rank every component
    # is at hand, and a failed ring entry reads the parent's boxes directly; a
    # decomposed patch's ranks hold their own components only, so each sends
    # its multilinear rings after its chain rings, and the gathered rings are
    # checked after the unpack.
    fallback = _shell_fallback(solver)
    gathered = fallback && np > 1
    sendbuf = _fit!(shell.buffers.send, (gathered ? 2 : 1) * ringlen * n_owned)
    if !_device_path(Qf)
        pos = 0
        for c in owned
            _fill_stage0!(fill, lt.pstage[1], lt, c)
            for k in 1:K
                # Interpolation, not deconvolution: the coarse solution is
                # point samples, and `prolong!`'s deconvolution is exact only
                # on data a `restrict!` produced (see the `interpolate!`
                # docstring). Only the boxes the shell slots read are filled;
                # the ring entries packed from the rest of the last stage
                # are stale and reach no slot.
                interpolate!(lt.pstage[k+1], lt.pplans[k], lt.pstage[k],
                             shell.boxes[k])
            end
            bf = lt.pstage[K+1]
            @inbounds for s in slabs, g3 in s[3], g2 in s[2], g1 in s[1]
                pos += 1
                sendbuf[pos] = bf[g1 + shift[1] + padb[1], g2 + shift[2] + padb[2],
                                  g3 + shift[3] + padb[3]]
            end
        end
        if np == 1
            # Every component is this rank's, in order, so the send buffer
            # holds the ring column by column.
            copyto!(ring, 1, sendbuf, 1, ringlen * n_cons)
            fallback && _shell_fallback!(ring, _linear_sources(fill, lt, owned), lt, fdcp,
                                         solver)
            _write_shell_from_ring!(Qf, ring, table, lt, fdcp, n_cons)
            return states
        end
        gathered && _shell_linear_ring!(sendbuf, ringlen * n_owned, ringlen, parent(Qf),
                                        _linear_sources(fill, lt, owned), lt, fdcp,
                                        n_owned)
    else
        scratch = fine.level_scratch
        stages = scratch.stages
        _fill_stage0_dev!(fill, stages[1], scratch, lt, owned)
        for k in 1:K
            # One launch per stage, over the bounding box of the stage's
            # boxes, rather than one per box.
            _interpolate_dev!(stages[k+1], lt.pplans[k], stages[k], n_owned,
                              _bounding_box(shell.boxes[k]))
        end
        dsend = _device_send_stage(parent(Qf), ringlen * n_owned)
        pointwise!(_ring_pack_point!, parent(Qf), ringlen, n_owned, 1,
                   dsend, stages[K+1], (table...,), shift, padb, ringlen)
        if np == 1
            # The packed stage is the ring itself, column by column, and
            # stays on the device for the shell write.
            dring = reshape(dsend, ringlen, n_cons)
            fallback && _shell_fallback!(dring, _linear_sources_dev(fill, scratch, lt, owned),
                                         lt, fdcp, solver)
            _write_shell_from_ring!(Qf, dring, table, lt, fdcp, n_cons)
            return states
        end
        _tracked_copy!(sendbuf, 1, dsend, 1, ringlen * n_owned)
        if gathered
            dlinear = _device_send_stage(parent(Qf), ringlen * n_owned)
            _shell_linear_ring!(dlinear, 0, ringlen, parent(Qf),
                                _linear_sources_dev(fill, scratch, lt, owned), lt, fdcp,
                                n_owned)
            _tracked_copy!(sendbuf, ringlen * n_owned + 1, dlinear, 1, ringlen * n_owned)
        end
    end
    counts = gathered ? 2 .* shell.counts : shell.counts
    recv = _fit!(shell.buffers.recv, sum(counts))
    MPI.Allgatherv!(sendbuf, MPI.VBuffer(recv, counts), comm)
    at = 0
    for r in 0:np-1
        for c in (r+1):np:n_cons
            copyto!(view(ring, :, c), view(recv, at+1:at+ringlen))
            at += ringlen
        end
        gathered || continue
        for c in (r+1):np:n_cons
            copyto!(view(shell.linear, :, c), view(recv, at+1:at+ringlen))
            at += ringlen
        end
    end
    gathered && _select_admissible!(ring, shell.linear, lt, fdcp, solver)
    _write_shell_from_ring!(Qf, ring, table, lt, fdcp, n_cons)
    return states
end

# --- Admissible shell fallback ----------------------------------------------
#
# The Lagrange chain and, under subcycling, the cubic Hermite blend in time
# overshoot across a shock, so they can set a shell node to ρ ≤ 0 or ρe ≤ 0
# between admissible parent nodes. Such a node takes instead the multilinear
# interpolant of the parent nodes bracketing it, blended linearly in time
# between the two stored endpoint boxes under subcycling. Its weights are
# nonnegative and sum to one, so the node is a convex combination of parent
# states. The shell interpolates the conserved variables (ρY_k, m, E), in
# which the set ρ > 0, ρe = E − |m|²/(2ρ) > 0 is convex: |m|²/ρ is jointly
# convex for ρ > 0, so ρe is concave and its positive superlevel set convex.
# The node is therefore admissible whenever its parent nodes are. A node the
# chain leaves admissible keeps the chain's value bit for bit, and so does one
# whose multilinear state is not admissible either (a parent state outside the
# admissible set, which no interpolant repairs). The test is
# the positivity limiter's: ρ and ρe, not the partial densities, whose
# interface undershoots lie inside the species band. Only the ring entries a
# shell slot reads are tested (`_in_shell`); the rest are stale where the
# chain fills only the boxes the slots need.

# Whether the shell takes the fallback: an ideal-gas mixture under the
# single-temperature equations, whose admissible set is the convex one
# above. The caloric inversion of a tabulated or NASA-9 model has no such
# closed form, and its shell keeps the chain's values.
_shell_fallback(solver) =
    solver.eos isa IdealMixture &&
    solver.equations.n_cons == solver.equations.n_species + 4

# The sources of the multilinear values over this rank's components: the two
# boxes blended in time at weight `w`, and the slot of component `owned[b]`
# in them, `c1 + (b − 1) cstep`. The gathered box stands for both ends of an
# imposition at one time; the Hermite endpoints are the parent's states at
# the two ends of its step.
_linear_sources(::BoxFill, lt::LevelTransfer{T}, owned) where {T} =
    (lt.box_gather, lt.box_gather, zero(T), first(owned), step(owned))
_linear_sources(hf::HermiteFill, lt::LevelTransfer{T}, owned) where {T} =
    (lt.box_Q0, lt.box_Q1, T(hf.θ), first(owned), step(owned))
# A device patch's: the uploaded box is stage 0 of its chain, its components
# in slot order; the Hermite endpoints are the uploaded boxes.
_linear_sources_dev(::BoxFill, scratch::LevelScratch, lt::LevelTransfer{T},
                    owned) where {T} =
    (scratch.stages[1], scratch.stages[1], zero(T), 1, 1)
_linear_sources_dev(hf::HermiteFill, scratch::LevelScratch, lt::LevelTransfer{T},
                    owned) where {T} =
    (scratch.Q0, scratch.Q1, T(hf.θ), first(owned), step(owned))

# The shell geometry the fallback bodies take over storage like `route`:
# the slab table, the box shift, the patch's extent and resolved dimensions
# and its imposed and boundary faces, as `_write_shell_from_ring!` takes
# them; and the stage-0 box's halo pad and extent.
function _fallback_geometry(lt::LevelTransfer, df::Decomp, route)
    active = (df.active[1], df.active[2], df.active[3])
    geometry = (_body_table(route, lt.shell.table), _box_shift(lt),
                fine_extent(lt.region, active, lt.folded), active, lt.imposed,
                lt.boundary)
    box = lt.pdecomps[1]
    return geometry, (box.n_halo_d, box.n_local)
end

# The slab table as a body reads it: the vector itself on host storage, and
# on a device, whose kernels take no vector, an isbits tuple. A splatted
# vector has no static length, so the launch on host storage, which runs at
# every shell imposition, would otherwise be dispatched at run time.
_body_table(::Array, table) = table
_body_table(route, table) = (table...,)

# On one rank: every shell entry of `ring` whose chain state is not
# admissible takes its multilinear state from the sources `src`, which hold
# every component.
function _shell_fallback!(ring, src, lt::LevelTransfer, df::Decomp, solver)
    src0, src1, w, c1, cstep = src
    geometry, box = _fallback_geometry(lt, df, ring)
    pointwise!(_shell_fallback_point!, ring, size(ring, 1), 1, 1,
               ring, src0, src1, w, c1, cstep, geometry, box, _limiter_layout(solver))
    return ring
end

@inline function _shell_fallback_point!(ring, src0, src1, w, c1, cstep, geometry, box,
                                        lay, at, _j, _k)
    @inbounds begin
        table, shift, Nf, active, imposed, boundary = geometry
        g = _ring_node(table, at)
        if _in_shell(g[1], g[2], g[3], Nf, active, imposed, boundary) &
           !_ring_admissible(ring, at, lay)
            n = (g[1] + shift[1], g[2] + shift[2], g[3] + shift[3])
            if _linear_admissible(src0, src1, w, n, active, box, c1, cstep, lay)
                for c in 1:lay[6]
                    ring[at, c] = _linear_value(src0, src1, w, n, active, box,
                                                c1 + (c - 1) * cstep)
                end
            end
        end
    end
    return nothing
end

# On a decomposed patch: the multilinear rings of this rank's `n_owned`
# components into `dst`, the b-th after linear index `base + (b − 1) stride`
# in the chain ring's order, at its shell entries.
function _shell_linear_ring!(dst, base::Int, stride::Int, route, src, lt::LevelTransfer,
                             df::Decomp, n_owned::Int)
    src0, src1, w, c1, cstep = src
    geometry, box = _fallback_geometry(lt, df, route)
    pointwise!(_shell_linear_point!, route, lt.shell.len, n_owned, 1,
               dst, src0, src1, w, c1, cstep, geometry, box, base, stride)
    return dst
end

@inline function _shell_linear_point!(dst, src0, src1, w, c1, cstep, geometry, box,
                                      base, stride, at, b, _k)
    @inbounds begin
        table, shift, Nf, active, imposed, boundary = geometry
        g = _ring_node(table, at)
        if _in_shell(g[1], g[2], g[3], Nf, active, imposed, boundary)
            n = (g[1] + shift[1], g[2] + shift[2], g[3] + shift[3])
            dst[base + (b - 1) * stride + at] =
                _linear_value(src0, src1, w, n, active, box, c1 + (b - 1) * cstep)
        end
    end
    return nothing
end

# On a decomposed patch, after the gather: every shell entry of `ring` whose
# chain state is not admissible takes its gathered multilinear state from
# `linear`; both hold every component, one column each.
function _select_admissible!(ring, linear, lt::LevelTransfer, df::Decomp, solver)
    geometry, _ = _fallback_geometry(lt, df, ring)
    pointwise!(_shell_select_point!, ring, size(ring, 1), 1, 1,
               ring, linear, geometry, _limiter_layout(solver))
    return ring
end

@inline function _shell_select_point!(ring, linear, geometry, lay, at, _j, _k)
    @inbounds begin
        table, _, Nf, active, imposed, boundary = geometry
        g = _ring_node(table, at)
        if _in_shell(g[1], g[2], g[3], Nf, active, imposed, boundary) &
           !_ring_admissible(ring, at, lay) & _ring_admissible(linear, at, lay)
            for c in 1:lay[6]
                ring[at, c] = linear[at, c]
            end
        end
    end
    return nothing
end

# The shell node (g1, g2, g3) of ring entry `at`: the slab decode of
# `_ring_pack_point!`. The slabs' offset ranges partition the ring.
@inline function _ring_node(table, at)
    g = (0, 0, 0)
    @inbounds for (lo, hi, sbase) in table
        n1 = hi[1] - lo[1] + 1
        n2 = hi[2] - lo[2] + 1
        n3 = hi[3] - lo[3] + 1
        if sbase < at <= sbase + n1 * n2 * n3
            r = at - sbase - 1
            g = (lo[1] + r % n1, lo[2] + (r ÷ n1) % n2, lo[3] + r ÷ (n1 * n2))
        end
    end
    return g
end

# ρ > 0 and ρe > 0 at ring entry `at`, the limiter's `_limiter_positive`
# written with `&` for a device body.
@inline function _ring_admissible(ring, at, lay)
    ns, m1, m2, m3, ie, _ = lay
    @inbounds begin
        ρ = zero(eltype(ring))
        for sp in 1:ns
            ρ += ring[at, sp]
        end
        mx, my, mz = ring[at, m1], ring[at, m2], ring[at, m3]
        return (ρ > 0) & (2 * ρ * ring[at, ie] - (mx * mx + my * my + mz * mz) > 0)
    end
end

# Whether the multilinear state at node `n` of the chain's final box is
# admissible, the test of `_ring_admissible`.
@inline function _linear_admissible(src0, src1, w, n, active, box, c1, cstep, lay)
    ns, m1, m2, m3, ie, _ = lay
    value(c) = _linear_value(src0, src1, w, n, active, box, c1 + (c - 1) * cstep)
    ρ = zero(eltype(src0))
    for sp in 1:ns
        ρ += value(sp)
    end
    mx, my, mz = value(m1), value(m2), value(m3)
    return (ρ > 0) & (2 * ρ * value(ie) - (mx * mx + my * my + mz * mz) > 0)
end

# The multilinear value of component slot `c` at node `n` of the chain's
# final box. Along a refined dimension the node lies at fraction f/3 of the
# parent interval from stage-0 node m = (n − 1) ÷ 3 + 1, f = (n − 1) mod 3,
# and the tensor product of the weights (3 − f)/3 and f/3 over the
# bracketing parent nodes gives the value, at each end of the time blend. A
# coincident node takes its parent node alone; the zero-weight corner is read
# at a clamped index, so no read leaves the box.
@inline function _linear_value(src0, src1, w, n, active, box, c)
    pad0, ext0 = box
    T = eltype(src0)
    m1, f1 = _linear_bracket(n[1], active[1])
    m2, f2 = _linear_bracket(n[2], active[2])
    m3, f3 = _linear_bracket(n[3], active[3])
    v0 = zero(T)
    v1 = zero(T)
    @inbounds for s3 in 0:1, s2 in 0:1, s1 in 0:1
        wt = _linear_weight(T, f1, s1) * _linear_weight(T, f2, s2) *
             _linear_weight(T, f3, s3)
        I = CartesianIndex(clamp(m1 + s1, 1, ext0[1]) + pad0[1],
                           clamp(m2 + s2, 1, ext0[2]) + pad0[2],
                           clamp(m3 + s3, 1, ext0[3]) + pad0[3])
        v0 += wt * src0[I, c]
        v1 += wt * src1[I, c]
    end
    return (one(T) - w) * v0 + w * v1
end

# The parent node at or below final-box node `n` and the node's offset from
# it in fine spacings, along a refined dimension; a collapsed one is
# unrefined.
@inline function _linear_bracket(n, refined)
    m = (n - 1) ÷ 3 + 1
    return (ifelse(refined, m, n), ifelse(refined, n - 1 - 3 * (m - 1), 0))
end

@inline _linear_weight(::Type{T}, f, s) where {T} = T(ifelse(s == 0, 3 - f, f)) / T(3)

function _impose_shell_gradients!(solver, states, lt::LevelTransfer, fill)
    patches = getfield(solver, :patches)
    fine = patches[lt.fine_index]
    Qf = states[lt.fine_index]
    fdcp = fine.decomp
    comm = fdcp.comm
    np = MPI.Comm_size(comm)
    me = MPI.Comm_rank(comm)
    n_cons = solver.equations.n_cons
    K = length(lt.pplans)
    shell = lt.shell
    slabs = shell.slabs
    ringlen = shell.len
    g = lt.gradients
    boxf = lt.pdecomps[K+1]
    padb = boxf.n_halo_d
    shift = _box_shift(lt)
    owned = (me+1):np:n_cons
    n_owned = length(owned)
    # Under the fallback on a decomposed patch each component's rings are its
    # value ring, its three gradient rings and its multilinear ring.
    fallback = _shell_fallback(solver)
    gathered = fallback && np > 1
    per = gathered ? 5 : 4
    sendbuf = _fit!(shell.buffers.send, per * ringlen * n_owned)
    if _device_path(Qf)
        _impose_shell_gradients_dev!(solver, fine, Qf, lt, fill, owned, sendbuf, shift,
                                     n_cons, fallback)
        return states
    end
    bf = lt.pstage[K+1]
    ring = shell.ring
    gring = g.gring
    # On one rank every component is this rank's, in order, and each ring is
    # packed straight into its column of `ring` or `gring`; a decomposed patch
    # packs into the send buffer for the gather.
    direct = np == 1
    pos = 0
    for c in owned
        _fill_stage0!(fill, lt.pstage[1], lt, c)
        for k in 1:K
            interpolate!(lt.pstage[k+1], lt.pplans[k], lt.pstage[k])
        end
        for j in 0:3
            dst, at = direct ? (j == 0 ? ring : gring,
                                (j == 0 ? c - 1 : 3 * (c - 1) + j - 1) * ringlen) :
                               (sendbuf, pos)
            src = bf
            if j > 0
                plan = g.plans[j]
                if plan === nothing
                    fill!(view(dst, at+1:at+ringlen), 0)
                    pos += ringlen
                    continue
                end
                apply_along!(g.tmp, plan, bf, g.decomps[j])
                src = g.tmp
            end
            _pack_ring!(dst, at, src, slabs, shift, padb)
            pos += ringlen
        end
        gathered && (pos += ringlen)
    end
    if direct
        fallback && _shell_fallback!(ring, _linear_sources(fill, lt, owned), lt, fdcp,
                                     solver)
        _write_shell_from_ring!(Qf, ring, shell.table, lt, fdcp, n_cons)
        return states
    end
    gathered && _shell_linear_ring!(sendbuf, 4 * ringlen, 5 * ringlen, parent(Qf),
                                    _linear_sources(fill, lt, owned), lt, fdcp, n_owned)
    counts = per .* shell.counts
    recv = _fit!(shell.buffers.recv, sum(counts))
    MPI.Allgatherv!(sendbuf, MPI.VBuffer(recv, counts), comm)
    at = 0
    for r in 0:np-1, c in (r+1):np:n_cons
        copyto!(view(ring, :, c), view(recv, at+1:at+ringlen))
        at += ringlen
        for j in 1:3
            copyto!(view(gring, :, 3 * (c - 1) + j), view(recv, at+1:at+ringlen))
            at += ringlen
        end
        gathered || continue
        copyto!(view(shell.linear, :, c), view(recv, at+1:at+ringlen))
        at += ringlen
    end
    gathered && _select_admissible!(ring, shell.linear, lt, fdcp, solver)
    _write_shell_from_ring!(Qf, ring, shell.table, lt, fdcp, n_cons)
    return states
end

# One ring of `src` over the chain's final box into `dst` after linear index
# `at`, in slab order.
function _pack_ring!(dst, at::Int, src, slabs, shift, padb)
    @inbounds for s in slabs, g3 in s[3], g2 in s[2], g1 in s[1]
        at += 1
        dst[at] = src[g1 + shift[1] + padb[1], g2 + shift[2] + padb[2],
                      g3 + shift[3] + padb[3]]
    end
    return dst
end

# The device form over the fine patch's scratch: the chain, the derivatives
# through the scratch's plans, and a pack of each owned component's value and
# gradient rings, and under the fallback on a decomposed patch its
# multilinear ring, into one device buffer in the order the host path packs
# them. The unpack writes the shell ring, the scratch's gradient ring and the
# multilinear ring from that buffer, or, on a decomposed patch, from the
# gathered rings, which stage through the host for the Allgatherv: one
# download of this rank's rings and one upload of every rank's.
function _impose_shell_gradients_dev!(solver, fine, Qf, lt::LevelTransfer, fill, owned,
                                      sendbuf, shift, n_cons::Int, fallback::Bool)
    fdcp = fine.decomp
    np = MPI.Comm_size(fdcp.comm)
    K = length(lt.pplans)
    shell = lt.shell
    ringlen = shell.len
    table = (shell.table...,)
    boxf = lt.pdecomps[K+1]
    padb = boxf.n_halo_d
    scratch = fine.level_scratch
    stages = scratch.stages
    n_owned = length(owned)
    _fill_stage0_dev!(fill, stages[1], scratch, lt, owned)
    for k in 1:K
        _interpolate_dev!(stages[k+1], lt.pplans[k], stages[k], n_owned)
    end
    route = parent(Qf)
    gathered = fallback && np > 1
    per = gathered ? 5 : 4
    dsend = _device_send_stage(route, per * ringlen * n_owned)
    for b in 1:n_owned
        bf = view(stages[K+1], :, :, :, b)
        for j in 0:3
            base = (per * (b - 1) + j) * ringlen
            if j == 0
                pointwise!(_ring_pack_field_point!, route, ringlen, 1, 1,
                           dsend, bf, table, shift, padb, base)
                continue
            end
            plan = scratch.gplans[j]
            if plan === nothing
                fill!(view(dsend, (base + 1):(base + ringlen)), 0)
                continue
            end
            apply_along!(scratch.gtmp, plan, bf, scratch.gdecomps[j])
            pointwise!(_ring_pack_field_point!, route, ringlen, 1, 1,
                       dsend, scratch.gtmp, table, shift, padb, base)
        end
    end
    gathered && _shell_linear_ring!(dsend, 4 * ringlen, 5 * ringlen, route,
                                    _linear_sources_dev(fill, scratch, lt, owned), lt,
                                    fdcp, n_owned)
    recv = dsend
    if np > 1
        counts = per .* shell.counts
        total = sum(counts)
        _tracked_copy!(sendbuf, 1, dsend, 1, per * ringlen * n_owned)
        hrecv = _fit!(shell.buffers.recv, total)
        MPI.Allgatherv!(sendbuf, MPI.VBuffer(hrecv, counts), fdcp.comm)
        recv = _device_send_stage(route, total)
        _tracked_copy!(recv, 1, hrecv, 1, total)
    end
    ring = similar(route, ringlen, n_cons)
    linear = gathered ? similar(route, ringlen, n_cons) : ring
    pointwise!(_ring_unpack_point!, route, ringlen, per * n_cons, 1,
               ring, scratch.gring, linear, recv, ringlen, np, n_cons, per)
    if gathered
        _select_admissible!(ring, linear, lt, fdcp, solver)
    elseif fallback
        _shell_fallback!(ring, _linear_sources_dev(fill, scratch, lt, owned), lt, fdcp,
                         solver)
    end
    _write_shell_from_ring!(Qf, ring, shell.table, lt, fdcp, n_cons)
    return Qf
end

# One ring entry of a field `src` over the chain's final box, written at
# offset `base` of the send buffer; the slab decode is `_ring_pack_point!`'s.
@inline function _ring_pack_field_point!(send, src, table, shift, padb, base,
                                         at, _j, _k)
    @inbounds for (lo, hi, sbase) in table
        n1 = hi[1] - lo[1] + 1
        n2 = hi[2] - lo[2] + 1
        n3 = hi[3] - lo[3] + 1
        if sbase < at <= sbase + n1 * n2 * n3
            r = at - sbase - 1
            g1 = lo[1] + r % n1
            r ÷= n1
            g2 = lo[2] + r % n2
            g3 = lo[3] + r ÷ n2
            send[base + at] = src[g1 + shift[1] + padb[1], g2 + shift[2] + padb[2],
                                  g3 + shift[3] + padb[3]]
        end
    end
    return nothing
end

# One entry of the gathered rings: block `p − 1` holds, in the ranks'
# order, the value ring (`j = 0`), the gradient ring along `j` (1 to 3) or,
# with `per` = 5 rings to a component, the multilinear ring (`j = 4`) of the
# component that position of the order names, rank r's components being
# (r+1):np:n_cons.
@inline function _ring_unpack_point!(ring, gring, linear, recv, ringlen, np, n_cons,
                                     per, at, p, _k)
    @inbounds begin
        blk, j = divrem(p - 1, per)
        c = 0
        for r in 0:np-1
            nr = r < n_cons ? (n_cons - r - 1) ÷ np + 1 : 0
            if blk < nr
                c = r + 1 + blk * np
                break
            end
            blk -= nr
        end
        v = recv[(p - 1) * ringlen + at]
        if j == 0
            ring[at, c] = v
        elseif j == 4
            linear[at, c] = v
        else
            gring[at, 3 * (c - 1) + j] = v
        end
    end
    return nothing
end

function _write_shell_from_ring!(Qf, ring, table, lt::LevelTransfer,
                                 df::Decomp, n_cons::Int)
    padf = df.n_halo_d
    nf = df.n_local
    off = df.offset
    Nf = fine_extent(lt.region, ntuple(d -> df.active[d], 3), lt.folded)
    active = (df.active[1], df.active[2], df.active[3])
    imposed = lt.imposed
    boundary = lt.boundary
    if !_device_path(Qf)
        # Every shell slot lies in a slab (`_check_ring_cover`), so the slabs'
        # nodes inside this rank's padded block are visited instead of the
        # block; a slot two slabs share is written twice with one value.
        lo_g = ntuple(d -> 1 - padf[d] + off[d], 3)
        hi_g = ntuple(d -> nf[d] + padf[d] + off[d], 3)
        @inbounds for (lo, hi, _) in table
            r = ntuple(d -> max(lo[d], lo_g[d]):min(hi[d], hi_g[d]), 3)
            for g3 in r[3], g2 in r[2], g1 in r[1]
                _in_shell(g1, g2, g3, Nf, active, imposed, boundary) || continue
                at = _ring_offset(table, g1, g2, g3)
                i, j, k = g1 - off[1] + padf[1], g2 - off[2] + padf[2],
                          g3 - off[3] + padf[3]
                for c in 1:n_cons
                    Qf[i, j, k, c] = ring[at, c]
                end
            end
        end
        return Qf
    end
    # Device patch: the ring (thin) uploads, unless the chain packed it on
    # the device already, and one kernel over the ring's entries writes every
    # component of every shell slot, far less traffic than the whole box.
    dev_ring = if _cpu_storage(ring)
        r = similar(parent(Qf), size(ring))
        copyto!(r, ring)
        r
    else
        ring
    end
    ringlen = size(ring, 1)
    pointwise!(_shell_ring_point!, Qf, ringlen, 1, 1,
               Qf, dev_ring, (table...,), off, padf, nf, Nf, active, imposed,
               boundary, n_cons)
    return Qf
end

# One ring entry: the slab decode of `_ring_pack_field_point!` gives its
# node, which is written, from the entry the host writer reads for it, where
# it lies in this rank's padded block and in the shell. A slot that two slabs
# share is written by both entries with one value.
@inline function _shell_ring_point!(Qf, ring, table, off, padf, nf, Nf, active,
                                    imposed, boundary, n_cons, at, _j, _k)
    @inbounds for (lo, hi, sbase) in table
        n1 = hi[1] - lo[1] + 1
        n2 = hi[2] - lo[2] + 1
        n3 = hi[3] - lo[3] + 1
        if sbase < at <= sbase + n1 * n2 * n3
            r = at - sbase - 1
            g1 = lo[1] + r % n1
            r ÷= n1
            g2 = lo[2] + r % n2
            g3 = lo[3] + r ÷ n2
            i = g1 - off[1] + padf[1]
            j = g2 - off[2] + padf[2]
            k = g3 - off[3] + padf[3]
            if 1 <= i <= nf[1] + 2 * padf[1] && 1 <= j <= nf[2] + 2 * padf[2] &&
               1 <= k <= nf[3] + 2 * padf[3] &&
               _in_shell(g1, g2, g3, Nf, active, imposed, boundary)
                first_at = _ring_offset(table, g1, g2, g3)
                for c in 1:n_cons
                    Qf[i, j, k, c] = ring[first_at, c]
                end
            end
        end
    end
    return nothing
end

"""
    prolong_level_ghosts!(solver, states)

Impose every refined patch's ghost ring and boundary-plane nodes from the
Lagrange interpolation (`level_interpolation_order`) of the current state of
its parent over the buffered box, per conserved component, and return
`states`. For an ideal-gas mixture, a shell node the interpolation leaves
with ρ ≤ 0 or ρe ≤ 0 takes instead the multilinear interpolant of the parent
nodes around it, which is admissible wherever they are; every other node
keeps the Lagrange value. Levels are visited from the root down, so a patch
two levels deep reads a parent whose own shell has just been imposed. Runs
after every RK stage update and inside the pre-step synchronization; a solver
without refinement returns immediately. Communicates in two rank sets per
level, so the loop is written over levels rather than over one flat
transfer list: the box exchange reads the parent state and is entered by
every rank owning the parent, point to point from
the parent ranks holding each box to the owners of its tile; the chains and
the ring Allgatherv of `_impose_shell!` run over each tile's own
communicator, which only its owners enter. The name "prolong" refers to the
operation's role; the operator is `interpolate!`, per the header note on why
the deconvolving `prolong!` is not used here.
"""
function prolong_level_ghosts!(solver, states)
    levels = getfield(solver, :levels)
    for ℓ in 2:length(levels)
        # Outside the parent's subset there is no parent state to send and,
        # the subsets being nested, no piece of this level or any below it.
        levels[ℓ-1].level_comm.owned || continue
        lev = levels[ℓ]
        t0 = time_ns()
        _exchange_boxes!(solver, states, lev, lt -> lt.box_gather, false)
        _wait!(solver, t0)
        for lt in lev.transfers
            # The imposition is collective over the tile's own communicator,
            # which its holders alone enter.
            lt.fine_index == 0 || _impose_shell!(solver, states, lt, BoxFill())
            # The imposed shell replaces the halo values the previous exchange
            # left wherever the two overlap (the edge-owning ranks' outer
            # halos); interior rank-boundary halos keep their exchanged values,
            # so the composite fine state is self-consistent without a further
            # exchange.
        end
    end
    return states
end

# --- Restriction: fine state → covered coarse region ------------------------

# Write the restricted values of `:filter` restriction (`src4`, region-shaped,
# unpadded, on the one rank of a serial run) onto the parent-level nodes
# inside the covered region, holding `RESTRICT_MARGIN` region nodes back from
# a parent-fed face (`_restrict_window`), one parent patch at a time.
function _write_covered_region!(src4, lt::LevelTransfer, states, patches)
    win = _restrict_window(lt)
    for li in lt.coarse_local, σ in _images(lt.period)
        li == 0 && continue
        # A window across a periodic seam lands partly on the parent's image.
        _write_covered_patch!(states[li], src4, win, lt.region.offset .- σ, patches[li])
    end
    return src4
end

# The part of the window this rank owns of one parent patch. `src4[i, j, k]`
# is region-local node (i, j, k) ↔ parent-level node off + (i, j, k) ↔ that
# patch's local node minus its region offset ↔ padded index minus the rank's
# block offset plus the pad.
# `parent_patch`, not `parent`: the device branch below calls `Base.parent` on
# the conserved state, which a parameter of that name shadows.
function _write_covered_patch!(coarse_Q, src4, win, off, parent_patch::Patch)
    dp = parent_patch.decomp
    padc = dp.n_halo_d
    poff = parent_patch.region.offset
    r = ntuple(3) do d
        # Region-local nodes intersected with this rank's owned patch nodes.
        lo = max(first(win[d]), dp.offset[d] + 1 + poff[d] - off[d])
        hi = min(last(win[d]), dp.offset[d] + dp.n_local[d] + poff[d] - off[d])
        lo:hi
    end
    any(isempty, r) && return coarse_Q
    sh = ntuple(d -> off[d] - poff[d] - dp.offset[d] + padc[d], 3)
    if !_device_path(coarse_Q)
        @inbounds for c in 1:size(src4, 4), k in r[3], j in r[2], i in r[1]
            coarse_Q[i + sh[1], j + sh[2], k + sh[3], c] = src4[i, j, k, c]
        end
        return coarse_Q
    end
    # Device parent patch: the covered write is a rectangular block copy, so
    # the replicated window uploads once and a broadcast assigns it.
    w = view(src4, r[1], r[2], r[3], :)
    dev_win = similar(parent(coarse_Q), size(w))
    copyto!(dev_win, Array(w))
    lr = ntuple(d -> (first(r[d]) + sh[d]):(last(r[d]) + sh[d]), 3)
    view(parent(coarse_Q), lr[1], lr[2], lr[3], 1:size(src4, 4)) .= dev_win
    return coarse_Q
end

# Test hook: the number of `restrict_level!` calls in this process, from
# which a test reads the restrictions a run takes per step.
const RESTRICTION_COUNT = Ref(0)

"""
    restrict_level!(solver, states)

Restrict every refined patch's state onto the covered region of its parent,
per conserved component, finest level first, and return `states`. Under the
default `:inject` mode the coincident-node values move directly from the
tiles' owners to the parent ranks owning the covered nodes, point to point
([`LevelCoupling`](@ref)), with no `TransferPlan` per dimension; under
`:filter` the invertible pair's Gaussian filter runs over the fine patch's
extent before subsampling, which is a whole-patch line solve and therefore
still serial-only (guarded at setup). Either way the write-back stops
`RESTRICT_MARGIN` coarse nodes short of a parent-fed face. Runs once per
completed step, after the state filter; a solver without refinement returns
immediately. Every rank owning the parent enters, those outside the refined
level's own subset included: they hold covered nodes but no fine state.
"""
function restrict_level!(solver, states)
    RESTRICTION_COUNT[] += 1
    levels = getfield(solver, :levels)
    for ℓ in length(levels):-1:2
        levels[ℓ-1].level_comm.owned || continue
        _restrict_tiles!(solver, states, levels[ℓ])
        # The written parent nodes can sit beside a parent-level tile
        # interface; refresh that level's records before it is read again
        # (by its own restriction outward, or by the shells below it).
        ℓ > 2 && _sync_level!(solver, states, levels[ℓ - 1])
    end
    return states
end

# Restrict the tiles of `lev` that `tiles` selects (all of them for
# `nothing`) onto the parent. Entered by every rank owning the parent.
function _restrict_tiles!(solver, states, lev::Level, tiles=nothing)
    isempty(lev.transfers) && return states
    if lev.transfers[1].restriction === :filter
        for (t, lt) in enumerate(lev.transfers)
            (tiles === nothing || tiles[t]) && _restrict_filtered!(solver, states, lt)
        end
        return states
    end
    t0 = time_ns()
    _exchange_restriction!(solver, states, lev, tiles)
    _wait!(solver, t0)
    return states
end

# `:filter` restriction, serial only (rejected at setup under MPI), so the
# one rank holds the refined patch and `lt.fine_index` is set. A stage along
# a folded dimension filters component `c` with its parity across the fold,
# the root fold's on that dimension, as the patch's own fold does.
function _restrict_filtered!(solver, states, lt::LevelTransfer)
    patches = getfield(solver, :patches)
    root_folds = patches[1].folds
    Qf = states[lt.fine_index]
    K = length(lt.rplans)
    padb = lt.rdecomps[1].n_halo_d
    ext = lt.region.extent
    for c in 1:solver.equations.n_cons
        src = view(Qf, :, :, :, c)
        for k in K:-1:1
            dst = lt.rstage[k]
            input = k == K ? src : lt.rstage[k+1]
            plan = lt.rplans[k]
            σ = any(plan.folds) ?
                conserved_parity(solver.equations, root_folds[plan.dim].sigvel, c) : 1
            restrict!(dst, plan, input, σ)
        end
        # The chain's output is padded region-shaped scratch; the write-back
        # takes the unpadded region form (serial-only, so the copy is one
        # small array per component).
        view(lt.restricted, :, :, :, c) .=
            view(lt.rstage[1], (1 + padb[1]):(ext[1] + padb[1]),
                 (1 + padb[2]):(ext[2] + padb[2]),
                 (1 + padb[3]):(ext[3] + padb[3]))
    end
    _write_covered_region!(lt.restricted, lt, states, patches)
    return states
end

"""
    sync_levels!(solver, states)

Bring the levels to mutual consistency: restrict each refined state onto the
covered region of its parent, finest first, then re-impose every shell from
the (updated) parent states, root first. This is the pre-step form; within a
step only the prolongation half runs, since restriction is a per-step
operation in the coupling schedule.
"""
function sync_levels!(solver, states)
    restrict_level!(solver, states)
    prolong_level_ghosts!(solver, states)
    return states
end

# --- Subcycling support: the Hermite box ------------------------------------
#
# Under subcycling (three fine steps of dt/3 per parent step, Berger–Oliger
# order: parent first, children after), a refined patch's shell needs parent
# values at stage times between the parent's t^n and t^{n+1}. The parent
# solution over its step is reconstructed on the buffered box by cubic
# Hermite interpolation from its endpoint values and endpoint RHS rates,
# O(dt⁴), matching the integrator's order; LSRK54 has no free dense output
# and this is the standard substitute. The t^n data falls out of the parent
# step's first stage; the t^{n+1} data costs one extra parent RHS evaluation
# per parent step, taken before the children's substeps so it samples the
# parent trajectory, not the restricted composite (the restriction write-back
# would perturb the box values read here). At three or more levels the
# extra RHS recurs on every level that has children, once per substep of
# that level.

"""
    save_level_boxes!(solver, lev, states, dQs, at_end)

Deliver the parent state and its RHS over the buffered prolongation box of
every tile of `lev` into that tile's Hermite storage on its owners: the `t^n`
slots when `at_end` is false, the `t^n + dt` slots when true, each owner
receiving the components its share of the chains runs. `states` and `dQs`
are the solver's full vectors. Two box exchanges, entered by every rank
owning the parent level whether or not it owns a tile; the Hermite shell
evaluation at every fine stage is then communication-free.
"""
function save_level_boxes!(solver, lev::Level, states, dQs, at_end::Bool)
    _exchange_boxes!(solver, states, lev, lt -> at_end ? lt.box_Q1 : lt.box_Q0, false)
    _exchange_boxes!(solver, dQs, lev, lt -> at_end ? lt.box_dQ1 : lt.box_dQ0, false)
    patches = getfield(solver, :patches)
    for lt in lev.transfers
        _upload_hermite!(lt, patches, at_end)
    end
    return lev
end

# Cubic Hermite blend of the stored box data at fraction θ ∈ [0, 1] of the
# coarse step, written into `dst` (the chain's stage-0 scratch) for component
# `c`. `dt` is the coarse step, which scales the stored rates.
function _hermite_box!(dst, lt::LevelTransfer, c::Int, θ, dt)
    θT = eltype(dst)(θ)
    dtT = eltype(dst)(dt)
    oneT = one(θT)
    h00 = (oneT + eltype(dst)(2) * θT) * (oneT - θT)^2
    h10 = θT * (oneT - θT)^2
    h01 = θT^2 * (eltype(dst)(3) - eltype(dst)(2) * θT)
    h11 = θT^2 * (θT - oneT)
    box = lt.pdecomps[1]
    pad = box.n_halo_d
    nb = box.n_local
    Q0, dQ0 = lt.box_Q0, lt.box_dQ0
    Q1, dQ1 = lt.box_Q1, lt.box_dQ1
    @inbounds for k in 1:nb[3], j in 1:nb[2], i in 1:nb[1]
        I = CartesianIndex(i + pad[1], j + pad[2], k + pad[3])
        dst[I] = h00 * Q0[I, c] + h01 * Q1[I, c] +
                 dtT * (h10 * dQ0[I, c] + h11 * dQ1[I, c])
    end
    return dst
end

"""
    RegridSpec

Configuration and rebuild inputs for tagging-driven regridding
(`src/regrid.jl`): the regrid cadence in coarse steps, the tag criteria, the
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
artificial diffusivity number of the last right-hand side; `gradient_threshold`
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
    tile::Int                        # lattice edge; 0 = one box over the tags
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
end

RegridSpec{T}(interval, threshold, buffer, margin, n_halo, interface_rhs,
              deriv, filt, smoo, backend, tile, last_step,
              rebalance=0.0, persist=1) where {T} =
    RegridSpec{T}(interval, threshold, buffer, margin, n_halo, interface_rhs,
                  deriv, filt, smoo, backend, tile, last_step,
                  rebalance, persist, 0, 1.0, 0.0, 0.0, 0.0,
                  zero(T), zero(T), zero(T), nothing, zeros(Int8, 0, 0, 0),
                  T(2), 1, 0, Dict{BlockRegion,Int}(), nothing, 6, :inject,
                  Dict{BlockRegion,Int}[], true)

# The creation record of refined level `ℓ` (`created` for level 1).
_created(spec::RegridSpec, ℓ::Int) = ℓ == 1 ? spec.created : spec.deep_created[ℓ - 1]

"""
    hermite_level_shell!(solver, states, lt, θ, dt)

Impose the shell (ghost ring plus boundary planes) of the refined patch of
`lt` from the cubic Hermite reconstruction of its parent's solution at
fraction `θ` of the parent step of size `dt`, through the same
interpolation chain [`prolong_level_ghosts!`](@ref) uses. Requires both
endpoint slots filled by [`save_level_boxes!`](@ref); at `θ = 0` the result is
exactly the parent's `t^n` state and the imposition reduces to the
unsubcycled one. A node the reconstruction leaves inadmissible takes the
multilinear interpolant in space of the linear blend in time of the two
endpoint states, as [`prolong_level_ghosts!`](@ref) describes.
"""
function hermite_level_shell!(solver, states, lt::LevelTransfer, θ, dt)
    _impose_shell!(solver, states, lt, HermiteFill(θ, dt))
    return states
end
