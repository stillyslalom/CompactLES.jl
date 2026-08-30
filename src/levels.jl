# Level refinement: a hierarchy of nested patches, each level one patch over
# a refined region of its parent, the coupling between a patch and its
# parent, and the machinery that distributes that coupling. Design rationale
# and measurements: reference/AMR_GPU.md.
#
# Each refined region is given in its PARENT level's node space and is
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
# through the POINT-SAMPLE halves of the transfer machinery (transfer.jl):
#
#   - After every RK stage update, `prolong_level_ghosts!` interpolates the
#     coarse state (order 6) over a box extending `LEVEL_BUFFER` coarse nodes
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
#
# The invertible filter pair itself (`prolong!`/`restrict!`, deconvolution
# against Gaussian filtering) is not the default coupling, on a
# measurement: the pair's contract is that prolongation input is samples of
# the FILTERED field, while the live coarse solution is point samples of the
# field itself. Deconvolving point samples "sharpens" data that was never
# smoothed and filtering on the way down attenuates resolved content, both
# O(h²) against the solution: the entropy-wave gate measured order 1.3–1.7
# and errors three decades above the interpolation/injection coupling's,
# which measures order ≈ 3.5 (the one-sided divergence closures binding, as
# at a same-level patch interface). `level_restriction = :filter` keeps the
# filtered path selectable; its anti-alias smoothing is the tool to reach
# for if injection restriction proves positivity-limited on captured shocks.
# Regridding likewise initializes NEW fine regions by interpolation, since
# freshly covered coarse data are point samples too (regrid.jl).
#
# Multi-dimensional transfer is a tensor product realized as a chain: each
# active dimension is refined in turn, so a chain of K = (number of active
# dimensions) `TransferPlan`s connects the coarse box to the fully fine box
# through K − 1 intermediate grids, each stage carrying its own scratch array.
#
# Scope, enforced by `Solver`: Cartesian metric, no stretching, no folds,
# tridiagonal schemes, the `:delta4` detector, and no same-level patch
# decomposition alongside refinement. The fine patch's line solves close at
# the coarse–fine boundary with the same-level interface rows (extended-data
# gradients and filters, one-sided divergence).
#
# Distribution: both levels decompose over the whole rank set. The
# coupling DATA is replicated: every rank gathers the buffered coarse box
# (and, for restriction, the coincident-node samples of the fine patch) with
# one Allgatherv and writes only the shell or covered nodes it owns. The
# interpolation CHAINS distribute by conserved component, each rank
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
`(lo, hi, offset)` table, the ring length, the replicated ring itself, the
per-rank Allgatherv counts, and the ring's own buffers. All of it follows
from the refined region and the fine decomposition, which do not change over
a [`LevelTransfer`](@ref)'s life, so it is built once at setup. A subcycled
step imposes the shell about twenty times, so the geometry is not rebuilt
per call.
"""
struct ShellRing{T}
    slabs::Vector{NTuple{3,UnitRange{Int}}}
    table::Vector{Tuple{NTuple{3,Int},NTuple{3,Int},Int}}
    len::Int
    ring::Matrix{T}
    counts::Vector{Int}
    buffers::GatherBuffers{T}
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
and `restrict_level!`.
"""
struct LevelTransfer{T}
    region::BlockRegion              # refined region, parent LEVEL node space
    coarse_indices::Vector{Int}      # solver.patches indices of the parent-level
                                     # patches meeting the buffered box
    fine_index::Int                  # index of the refined patch in solver.patches
    imposed::NTuple{3,NTuple{2,Bool}} # per face: shell imposed from the parent
                                     # (false where a same-level neighbor
                                     # supplies the face through the records)
    restriction::Symbol              # :inject (coincident-node copy, default)
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
    # Distribution: the level transfer runs replicated; every rank
    # gathers the (small) coupling regions, runs the identical interpolation
    # chain, and writes only what it owns, so no consistency question arises
    # and the chain needs no halo exchanges of its own. These tables record
    # every rank's owned block of each patch, in that patch's node space and
    # in the rank order of the patch's Cartesian communicator; one table per
    # parent patch, aligned with `coarse_indices`.
    coarse_blocks::Vector{Vector{BlockRegion}}
    fine_blocks::Vector{BlockRegion}
    box_gather::Array{T,4}           # replicated box state, all components
    restricted::Array{T,4}           # replicated coincident-sample of the fine
                                     # patch over the region, all components
    # Retained staging for the three collectives that run every step.
    box_buffers::GatherBuffers{T}
    restrict_buffers::GatherBuffers{T}
    shell::ShellRing{T}
end

"""
    Level

One level of the refinement hierarchy: its index (0 is the root), the
indices into `solver.patches` of the patches at this level, and, aligned
with those, the [`LevelTransfer`](@ref) coupling each patch to its parent
(empty at the root), plus the same-level interface records among the
level's patches (the root's live on the `Solver`). `solver.levels` holds
them in ascending order, and every level below the root is nested inside
the one above it. Below the root a level is a set of tiles on a lattice of
edge `tile` parent nodes (one patch over the whole region when `tile` is
0), abutting tiles sharing their interface plane as root slabs do.
"""
struct Level{T}
    index::Int
    patches::Vector{Int}
    transfers::Vector{LevelTransfer{T}}
    ghost_sends::Vector{GhostRecord{T}}
    ghost_recvs::Vector{GhostRecord{T}}
    plane_pairs::Vector{PlaneRecord{T}}
end

Level{T}(index::Int, patches::Vector{Int},
         transfers::Vector{LevelTransfer{T}}) where {T} =
    Level{T}(index, patches, transfers, GhostRecord{T}[], GhostRecord{T}[],
             PlaneRecord{T}[])

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
patch order.
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
# `[lo, hi]`, or `nothing` when clipping leaves fewer than four nodes.
function _lattice_tile(k::NTuple{3,Int}, active::NTuple{3,Bool}, tile::Int,
                       lo::NTuple{3,Int}, hi::NTuple{3,Int})
    offext = ntuple(3) do d
        active[d] || return (0, 1)
        a = max(k[d] * tile + 1, lo[d])
        b = min(k[d] * tile + tile + 1, hi[d])
        (a - 1, b - a + 1)
    end
    any(d -> active[d] && offext[d][2] < 4, 1:3) && return nothing
    return BlockRegion(ntuple(d -> offext[d][1], 3), ntuple(d -> offext[d][2], 3))
end

# Tiles covering `box` (parent-level node space), in lattice order.
function _level_tiles(box::BlockRegion, active::NTuple{3,Bool}, tile::Int,
                      lo::NTuple{3,Int}, hi::NTuple{3,Int})
    spans = ntuple(d -> active[d] ?
                   _tile_span(box.offset[d] + 1, box.offset[d] + box.extent[d], tile) :
                   (0:0), 3)
    tiles = BlockRegion[]
    for k3 in spans[3], k2 in spans[2], k1 in spans[1]
        t = _lattice_tile((k1, k2, k3), active, tile, lo, hi)
        t === nothing || push!(tiles, t)
    end
    return tiles
end

# Face table of a tile set: neighbor patch index (1-based within the set)
# per face, 0 where none. Two tiles abut along `d` when one's high plane is
# the other's low plane and their transverse extents coincide, which the
# lattice guarantees whenever they touch at all.
function _tile_faces(regions::Vector{BlockRegion})
    n = length(regions)
    faces = Vector{NTuple{3,NTuple{2,Int}}}(undef, n)
    for p in 1:n
        rp = regions[p]
        faces[p] = ntuple(3) do d
            lo = 0
            hi = 0
            for q in 1:n
                q == p && continue
                rq = regions[q]
                same = all(e -> e == d || (rq.offset[e] == rp.offset[e] &&
                                           rq.extent[e] == rp.extent[e]), 1:3)
                same || continue
                rq.offset[d] + rq.extent[d] - 1 == rp.offset[d] && (lo = q)
                rp.offset[d] + rp.extent[d] - 1 == rq.offset[d] && (hi = q)
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
                     parent_indices::Vector{Int}, parent_regions::Vector{BlockRegion})
    box = _buffered(region, active, LEVEL_BUFFER)
    hits = Int[]
    for (i, r) in zip(parent_indices, parent_regions)
        any(d -> box.offset[d] + box.extent[d] <= r.offset[d] ||
                 r.offset[d] + r.extent[d] <= box.offset[d], 1:3) && continue
        push!(hits, i)
    end
    return hits
end

# Same-level records among the patches of a refined level, built through the
# root machinery on level-local ids and shifted onto solver patch indices.
# Collective over `comm` (one Allgather of every rank's blocks of every
# tile: each rank holds a block of every tile of a refined level).
function _level_records(::Type{T}, comm::MPI.Comm, regions::Vector{BlockRegion},
                        patch_indices::Vector{Int}, decomps, n_cons::Int) where {T}
    length(regions) > 1 || return GhostRecord{T}[], GhostRecord{T}[], PlaneRecord{T}[]
    faces = _tile_faces(regions)
    sends, recvs, planes = build_interface_records(T, comm, regions, faces,
                                                   collect(1:length(regions)),
                                                   decomps, n_cons)
    shift(r::GhostRecord{T}) =
        GhostRecord{T}(patch_indices[r.patch], r.partner,
                       r.partner_patch == 0 ? 0 : patch_indices[r.partner_patch],
                       r.tag, r.mine, r.theirs, r.buf)
    shift(r::PlaneRecord{T}) =
        PlaneRecord{T}(patch_indices[r.patch], r.partner,
                       r.partner_patch == 0 ? 0 : patch_indices[r.partner_patch],
                       r.tag, r.sendtag, r.mine, r.theirs, r.buf, r.sbuf)
    return map(shift, sends), map(shift, recvs), map(shift, planes)
end

# Whether the fine shell slot at patch-global node `g` (padded slots included)
# is imposed from the parent: outside the strict interior along some active
# dimension whose face on that side is parent-fed. With every face imposed
# this is the complement of the strict interior [2, N − 1]^3.
@inline function _in_shell(g1, g2, g3, Nf, active, imposed)
    @inbounds begin
        (active[1] && ((g1 <= 1 && imposed[1][1]) || (g1 >= Nf[1] && imposed[1][2]))) ||
        (active[2] && ((g2 <= 1 && imposed[2][1]) || (g2 >= Nf[2] && imposed[2][2]))) ||
        (active[3] && ((g3 <= 1 && imposed[3][1]) || (g3 >= Nf[3] && imposed[3][2])))
    end
end

# Same-level consistency of one refined level: shared-plane averaging and
# ghost refill through the level's records. The per-patch halo exchange that
# carries the strips into corners is the caller's, as for the root.
function _sync_level_records!(solver, states, lev::Level)
    isempty(lev.plane_pairs) && isempty(lev.ghost_sends) &&
        isempty(lev.ghost_recvs) && return states
    _average_planes!(solver, states, lev.plane_pairs)
    _exchange_ghosts!(solver, states, lev.ghost_sends, lev.ghost_recvs)
    return states
end

# Level sync inside the subcycled driver: records, then each patch's own
# halo exchange, skipped entirely for a one-tile level (its RHS evaluation
# exchanges halos itself).
function _sync_level!(solver, states, lev::Level)
    length(lev.patches) > 1 || return states
    _sync_level_records!(solver, states, lev)
    patches = getfield(solver, :patches)
    for pi in lev.patches
        exchange_state!(states[pi], patches[pi].decomp)
    end
    return states
end

# The transfers of every level below the root, root-adjacent first (the
# order shell imposition runs in; restriction runs it reversed).
_all_transfers(solver) =
    (lt for lev in getfield(solver, :levels) for lt in lev.transfers)

# Decomp chain refining `dims_to_refine` one at a time, starting from the
# coarse extents. Every stage runs replicated per rank (COMM_SELF, so the
# chain operators are the serial ones whatever the rank count) and is
# non-periodic along active dimensions; collapsed dimensions stay collapsed.
function _refine_chain(::Type{T}, ext0::NTuple{3,Int}, active::NTuple{3,Bool},
                       dims_to_refine::Vector{Int}, n_halo::Int,
                       interp_order::Int) where {T}
    pper = ntuple(d -> !active[d], 3)
    exts = [ext0]
    for dk in dims_to_refine
        prev = exts[end]
        push!(exts, ntuple(d -> d == dk ? 3 * prev[d] - 2 : prev[d], 3))
    end
    decomps = [Decomp{T}(e, pper; dims=(1, 1, 1), n_halo=n_halo,
                         comm=MPI.COMM_SELF) for e in exts]
    plans = [plan_transfer(decomps[k+1], decomps[k], dims_to_refine[k], T;
                           interp_order=interp_order)
             for k in eachindex(dims_to_refine)]
    stages = [zeros(T, ntuple(d -> e[d] + 2 * (active[d] ? n_halo : 0), 3))
              for e in exts]
    return decomps, plans, stages
end

"Every rank's owned interior block of `decomp`, in that decomp's node space
and communicator rank order. Collective over `decomp.comm` (one Allgather)."
function _owned_blocks(decomp::Decomp)
    np = MPI.Comm_size(decomp.comm)
    mine = Int64[decomp.offset..., decomp.n_local...]
    flat = MPI.Allgather(mine, decomp.comm)
    return [BlockRegion((Int(flat[6r+1]), Int(flat[6r+2]), Int(flat[6r+3])),
                        (Int(flat[6r+4]), Int(flat[6r+5]), Int(flat[6r+6])))
            for r in 0:np-1]
end

function build_level_transfer(::Type{T}, region::BlockRegion,
                              active::NTuple{3,Bool}, n_halo::Int,
                              coarse_indices::Vector{Int}, fine_index::Int,
                              restriction::Symbol,
                              n_cons::Int, subcycle::Bool,
                              coarse_decomps::Vector{<:Decomp{T}},
                              fine_decomp::Decomp{T},
                              faces::NTuple{3,NTuple{2,Int}}=ntuple(d -> (0, 0), 3)
                              ) where {T}
    imposed = ntuple(d -> (faces[d][1] == 0, faces[d][2] == 0), 3)
    dims_to_refine = [d for d in 1:3 if active[d]]
    boxext = ntuple(d -> active[d] ? region.extent[d] + 2 * LEVEL_BUFFER :
                                     region.extent[d], 3)
    pdecomps, pplans, pstage = _refine_chain(T, boxext, active, dims_to_refine,
                                             n_halo, 6)
    # The restriction chain never interpolates (that half of each TransferPlan
    # goes unused), so it is built at interpolation order 2, which admits the
    # smallest legal regions. Only `:filter` restriction applies it; `:inject`
    # is realized directly as the coincident-node gather below.
    rdecomps, rplans, rstage = _refine_chain(T, region.extent, active,
                                             dims_to_refine, n_halo, 2)
    boxsize = subcycle ? (size(pstage[1])..., n_cons) : (0, 0, 0, 0)
    slabs = _ring_slabs(region, fine_decomp)
    table, ringlen = _slab_table(slabs)
    npf = MPI.Comm_size(fine_decomp.comm)
    # Component-distributed chains: rank r owns components r+1, r+1+npf, ...
    ring_counts = [ringlen * length((r+1):npf:n_cons) for r in 0:npf-1]
    shell = ShellRing{T}(slabs, table, ringlen,
                         Matrix{T}(undef, ringlen, n_cons), ring_counts,
                         GatherBuffers{T}())
    return LevelTransfer{T}(region, coarse_indices, fine_index, imposed,
                            restriction, dims_to_refine,
                            pdecomps, pplans, pstage,
                            rdecomps, rplans, rstage,
                            zeros(T, boxsize), zeros(T, boxsize),
                            zeros(T, boxsize), zeros(T, boxsize),
                            [_owned_blocks(dc) for dc in coarse_decomps],
                            _owned_blocks(fine_decomp),
                            zeros(T, size(pstage[1])..., n_cons),
                            zeros(T, region.extent..., n_cons),
                            GatherBuffers{T}(), GatherBuffers{T}(), shell)
end

# --- Replicated-region gathers ----------------------------------------------
#
# Every rank assembles the full coupling region: each contributes the
# intersection of its owned block with the requested node region through one
# Allgatherv, and unpacks every rank's contribution into its own replica.
# The regions are small by construction (the refined region plus its buffer),
# so replication is the right first distribution: no halo
# machinery inside the transfer chains, no consistency questions, and the
# serial path is the same code with a one-rank communicator. The cost grows
# with region volume times rank count in message total, not per-rank memory;
# a rank-partitioned transfer is the recorded follow-up if a measured case
# outgrows it.

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

Assemble the node region `region_ranges` (in `decomp`'s node space) of the
distributed field `Q` (padded, 4-D) into the replicated array `dst` on every
rank: node `n` lands at `dst[n - dst_off + dst_pad, ..., c]`, where `dst_off`
maps node space onto `dst`'s unpadded box. With `sample = s`, only nodes
`n ≡ 1 (mod s)` along active dimensions participate and land at
`dst[(n-1) ÷ s + 1 ...]`, the coincident-node form the `:inject` restriction
uses (`s = 3` per refined dimension). Collective over `decomp.comm`.

`buffers` supplies the MPI staging. The per-step call sites pass the
[`GatherBuffers`](@ref) retained on the [`LevelTransfer`](@ref); a caller
running at the regrid cadence can let the default allocate.
"""
function gather_region!(dst::AbstractArray{T,4},
                        region_ranges::NTuple{3,UnitRange{Int}},
                        dst_off::NTuple{3,Int}, dst_pad::NTuple{3,Int},
                        Q, decomp::Decomp{T}, blocks::Vector{BlockRegion},
                        sample::NTuple{3,Int}=(1, 1, 1);
                        buffers::GatherBuffers{T}=GatherBuffers{T}()) where {T}
    comm = decomp.comm
    np = MPI.Comm_size(comm)
    me = MPI.Comm_rank(comm)
    n_cons = size(dst, 4)
    pad = decomp.n_halo_d
    # A sampled gather keeps nodes n with (n − 1) % sample == 0.
    keep(r, d) = first(r) + mod(sample[d] - mod(first(r) - 1, sample[d]),
                                sample[d]):sample[d]:last(r)
    isects = [ntuple(d -> keep(_region_isect(region_ranges, blocks[r+1])[d], d), 3)
              for r in 0:np-1]
    counts = [n_cons * prod(max(length(ir[d]), 0) for d in 1:3)
              for ir in isects]
    sendbuf = _fit!(buffers.send, counts[me+1])
    mine = isects[me+1]
    if _cpu_storage(Q)
        idx = 1
        @inbounds for c in 1:n_cons, k in mine[3], j in mine[2], i in mine[1]
            sendbuf[idx] = Q[i - decomp.offset[1] + pad[1],
                             j - decomp.offset[2] + pad[2],
                             k - decomp.offset[3] + pad[3], c]
            idx += 1
        end
    elseif counts[me+1] > 0
        # Device storage packs by broadcast into a contiguous stage (the
        # same strided-to-contiguous move the halo staging makes, with the
        # sampled ranges expressed as strided views), and one contiguous
        # device-to-host copy fills the MPI buffer. Column-major broadcast
        # order matches the scalar pack's (c slowest, i fastest).
        lr = ntuple(d -> (first(mine[d]) - decomp.offset[d] + pad[d]):sample[d]:
                         (last(mine[d]) - decomp.offset[d] + pad[d]), 3)
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

# `region` grown by `margin` nodes per side along active dimensions.
_buffered(region::BlockRegion, active::NTuple{3,Bool}, margin::Int) =
    BlockRegion(ntuple(d -> region.offset[d] - (active[d] ? margin : 0), 3),
                ntuple(d -> region.extent[d] + (active[d] ? 2 * margin : 0), 3))

# Whether every node of `region` lies in some member of `regions` (a union
# of boxes, not one box: the test is by node, and regions are small).
function _covered_by(region::BlockRegion, regions::Vector{BlockRegion})
    inside(g, r) = all(d -> r.offset[d] < g[d] <= r.offset[d] + r.extent[d], 1:3)
    for k in 1:region.extent[3], j in 1:region.extent[2], i in 1:region.extent[1]
        g = region.offset .+ (i, j, k)
        any(r -> inside(g, r), regions) || return false
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
"""
function _amr_dims(extent::NTuple{3,Int}, active::NTuple{3,Bool}, np::Int)
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
    bestmin < 0 &&
        error("no process grid over $np rank(s) gives every rank ≥ 9 fine " *
              "points per split dimension of a $extent fine patch; use " *
              "fewer ranks or a larger refined region")
    return best
end

# --- Prolongation: coarse state → fine ghost ring and boundary planes -------

# The buffered box as parent-level node ranges, and its node-space offset
# (box node 1 sits at node offset + 1). The box lies inside the union of the
# parent patches by the nesting margin, so every gathered value exists.
_box_offset(lt::LevelTransfer, active::NTuple{3,Bool}) =
    ntuple(d -> lt.region.offset[d] - (active[d] ? LEVEL_BUFFER : 0), 3)

function _box_ranges(lt::LevelTransfer, active::NTuple{3,Bool})
    off = _box_offset(lt, active)
    box = lt.pdecomps[1]
    return ntuple(d -> (off[d] + 1):(off[d] + box.n_local[d]), 3)
end

# Parent-level node ranges of `ranges` that fall in `region`, as that patch's
# own (local) node ranges: patch node = level node − region offset.
function _patch_local(ranges::NTuple{3,UnitRange{Int}}, region::BlockRegion)
    return ntuple(d -> (max(first(ranges[d]), region.offset[d] + 1) -
                        region.offset[d]):
                       (min(last(ranges[d]), region.offset[d] + region.extent[d]) -
                        region.offset[d]), 3)
end

# Gather the parent box, all components, into `dst4` (shaped like the chain's
# stage 0 with a trailing component index) on every rank: one gather per
# parent patch meeting the box, each placing its part by parent-level node.
# Collective over every parent patch's communicator, in `coarse_indices`
# order. `states` and `patches` are the solver's full vectors.
function _gather_box!(dst4, lt::LevelTransfer, states, patches)
    fine = patches[lt.fine_index]
    active = fine.decomp.active
    box = _box_ranges(lt, active)
    boxoff = _box_offset(lt, active)
    pad = lt.pdecomps[1].n_halo_d
    for (k, ci) in enumerate(lt.coarse_indices)
        parent = patches[ci]
        local_r = _patch_local(box, parent.region)
        any(isempty, local_r) && continue
        gather_region!(dst4, local_r, boxoff .- parent.region.offset, pad,
                       states[ci], parent.decomp, lt.coarse_blocks[k];
                       buffers=lt.box_buffers)
    end
    return dst4
end

function _write_fine_shell!(fine_Q, c::Int, box_field, lt::LevelTransfer,
                            df::Decomp, boxf::Decomp, shell_only::Bool=true)
    # Fine patch node g ↔ fine box node g + 3·LEVEL_BUFFER (active dims). The
    # shell is every padded slot whose PATCH-GLOBAL index lies outside the
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
    Nf = fine_extent(lt.region, ntuple(d -> df.active[d], 3))
    shift = ntuple(d -> df.active[d] ? 3 * LEVEL_BUFFER : 0, 3)
    active = (df.active[1], df.active[2], df.active[3])
    imposed = lt.imposed
    if _cpu_storage(fine_Q)
        r = ntuple(d -> (1 - padf[d]):(nf[d] + padf[d]), 3)
        @inbounds for k in r[3], j in r[2], i in r[1]
            g1, g2, g3 = i + off[1], j + off[2], k + off[3]
            shell_only && !_in_shell(g1, g2, g3, Nf, active, imposed) && continue
            fine_Q[i + padf[1], j + padf[2], k + padf[3], c] =
                box_field[g1 + shift[1] + padb[1], g2 + shift[2] + padb[2],
                          g3 + shift[3] + padb[3]]
        end
        return fine_Q
    end
    # Device patch: the interpolated box (a host chain product) uploads once
    # per component and a kernel imposes the shell. Batching the components
    # into one upload per stage would cut that traffic; this is the
    # correctness form.
    dev_box = similar(parent(fine_Q), size(box_field))
    copyto!(dev_box, box_field)
    pointwise!(_fine_shell_point!, fine_Q,
               nf[1] + 2 * padf[1], nf[2] + 2 * padf[2], nf[3] + 2 * padf[3],
               fine_Q, dev_box, c, off, padf, padb, shift, Nf,
               active, imposed, shell_only)
    return fine_Q
end

@inline function _fine_shell_point!(fine_Q, box_field, c, off, padf, padb,
                                    shift, Nf, active, imposed, shell_only,
                                    i, j, k)
    @inbounds begin
        g1 = i - padf[1] + off[1]
        g2 = j - padf[2] + off[2]
        g3 = k - padf[3] + off[3]
        if !shell_only || _in_shell(g1, g2, g3, Nf, active, imposed)
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
# The chains therefore distribute BY COMPONENT: rank r runs the chain only
# for components c with (c − 1) mod np == r, packs the thin shell ring of its
# results (2 slabs of thickness pad+1 per active dimension, full transverse
# extent, in patch-padded node space), and one Allgatherv replicates the
# rings; every rank then writes its own shell slots from the ring. The chain
# work per rank drops by ~min(np, n_cons)×, and the collective carries the
# ring, not the box. Values are bit-identical to the replicated form: the
# same chain output moves through a pack/unpack without recomputation.

# Ring slabs in patch-padded node space, ascending dimension order, low side
# then high per active dimension. Slabs overlap at corners; both copies of a
# corner value come from the same chain output, so the first match wins.
function _ring_slabs(region::BlockRegion, df::Decomp)
    Nf = fine_extent(region, ntuple(d -> df.active[d], 3))
    pad = df.n_halo_d
    full = ntuple(d -> (1 - pad[d]):(Nf[d] + pad[d]), 3)
    slabs = NTuple{3,UnitRange{Int}}[]
    for d in 1:3
        df.active[d] || continue
        push!(slabs, ntuple(q -> q == d ? ((1 - pad[d]):1) : full[q], 3))
        push!(slabs, ntuple(q -> q == d ? (Nf[d]:(Nf[d] + pad[d])) : full[q], 3))
    end
    return slabs
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
# both writers apply before calling here. A shell slot therefore always matches.
# A 0 return indicates that the slab set and the shell test
# have fallen out of step; `_write_shell_from_ring!` raises before indexing the
# ring at 0 under its `@inbounds`. The device body cannot raise,
# so the host path carries the check for both, running the identical
# arithmetic on the identical table.
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

@noinline _ring_miss(g1, g2, g3) =
    error("fine shell node ($g1, $g2, $g3) lies in no ring slab; the slab " *
          "set and the shell test disagree")

# Shared driver: run the chain for this rank's components with `fill0!(dst, c)`
# supplying stage 0, replicate the shell rings, and impose each rank's own
# shell slots. Collective over the fine communicator.
function _impose_shell!(solver, states, lt::LevelTransfer, fill0!::F) where {F}
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
    shift = ntuple(d -> fdcp.active[d] ? 3 * LEVEL_BUFFER : 0, 3)
    # This rank's components, as a range, not a filtered vector: the
    # ascending order is the one the ring unpack below assumes.
    owned = (me+1):np:n_cons
    sendbuf = _fit!(shell.buffers.send, ringlen * length(owned))
    pos = 0
    for c in owned
        fill0!(lt.pstage[1], c)
        for k in 1:K
            # Interpolation, not deconvolution: the coarse solution is point
            # samples, and `prolong!`'s deconvolution is exact only on data a
            # `restrict!` produced (see the `interpolate!` docstring).
            interpolate!(lt.pstage[k+1], lt.pplans[k], lt.pstage[k])
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
        _write_shell_from_ring!(Qf, ring, table, lt, fdcp, n_cons)
        return states
    end
    counts = shell.counts
    recv = _fit!(shell.buffers.recv, sum(counts))
    MPI.Allgatherv!(sendbuf, MPI.VBuffer(recv, counts), comm)
    at = 0
    for r in 0:np-1, c in (r+1):np:n_cons
        copyto!(view(ring, :, c), view(recv, at+1:at+ringlen))
        at += ringlen
    end
    _write_shell_from_ring!(Qf, ring, table, lt, fdcp, n_cons)
    return states
end

function _write_shell_from_ring!(Qf, ring, table, lt::LevelTransfer,
                                 df::Decomp, n_cons::Int)
    padf = df.n_halo_d
    nf = df.n_local
    off = df.offset
    Nf = fine_extent(lt.region, ntuple(d -> df.active[d], 3))
    active = (df.active[1], df.active[2], df.active[3])
    imposed = lt.imposed
    if _cpu_storage(Qf)
        r = ntuple(d -> (1 - padf[d]):(nf[d] + padf[d]), 3)
        @inbounds for k in r[3], j in r[2], i in r[1]
            g1, g2, g3 = i + off[1], j + off[2], k + off[3]
            _in_shell(g1, g2, g3, Nf, active, imposed) || continue
            at = _ring_offset(table, g1, g2, g3)
            at == 0 && _ring_miss(g1, g2, g3)
            for c in 1:n_cons
                Qf[i + padf[1], j + padf[2], k + padf[3], c] = ring[at, c]
            end
        end
        return Qf
    end
    # Device patch: the ring (thin) uploads and one kernel writes every
    # component of every shell slot, far less traffic than the whole box.
    dev_ring = similar(parent(Qf), size(ring))
    copyto!(dev_ring, ring)
    pointwise!(_shell_ring_point!, Qf,
               nf[1] + 2 * padf[1], nf[2] + 2 * padf[2], nf[3] + 2 * padf[3],
               Qf, dev_ring, (table...,), off, padf, Nf, active, imposed, n_cons)
    return Qf
end

@inline function _shell_ring_point!(Qf, ring, table, off, padf, Nf, active,
                                    imposed, n_cons, i, j, k)
    @inbounds begin
        g1 = i - padf[1] + off[1]
        g2 = j - padf[2] + off[2]
        g3 = k - padf[3] + off[3]
        if _in_shell(g1, g2, g3, Nf, active, imposed)
            at = _ring_offset(table, g1, g2, g3)
            for c in 1:n_cons
                Qf[i, j, k, c] = ring[at, c]
            end
        end
    end
    return nothing
end

"""
    prolong_level_ghosts!(solver, states)

Impose every refined patch's ghost ring and boundary-plane nodes from the
order-6 interpolation of the current state of its parent over the buffered
box, per conserved component, and return `states`. Levels are visited from
the root down, so a patch two levels deep reads a parent whose own shell has
just been imposed. Runs after every RK stage update and inside the pre-step
synchronization; a solver without refinement returns immediately.
Collective: one replicated box gather, the component-distributed chains, and
the ring Allgatherv of `_impose_shell!` per transfer. The name "prolong"
refers to the operation's role; the operator is `interpolate!`, per the
header note on why the deconvolving `prolong!` is not used here.
"""
function prolong_level_ghosts!(solver, states)
    for lt in _all_transfers(solver)
        _prolong_ghosts!(solver, states, lt)
    end
    return states
end

function _prolong_ghosts!(solver, states, lt::LevelTransfer)
    _gather_box!(lt.box_gather, lt, states, getfield(solver, :patches))
    _impose_shell!(solver, states, lt,
                   (dst, c) -> dst .= view(lt.box_gather, :, :, :, c))
    # The imposed shell replaces the halo values the previous exchange left
    # wherever the two overlap (the edge-owning ranks' outer halos); interior
    # rank-boundary halos keep their exchanged values, so the composite fine
    # state is self-consistent without a further exchange.
    return states
end

# --- Restriction: fine state → covered coarse region ------------------------

# Write the replicated restricted values (`src4`, region-shaped, unpadded)
# onto the parent-level nodes this rank owns inside the covered region,
# holding `RESTRICT_MARGIN` region nodes back from the boundary along active
# dims, one parent patch at a time.
function _write_covered_region!(src4, lt::LevelTransfer, states, patches)
    fine = patches[lt.fine_index]
    active = fine.decomp.active
    off = lt.region.offset
    ext = lt.region.extent
    # The written window in region-local nodes: the margin applies at a
    # parent-fed face only, since at a face shared with a same-level tile
    # the fine solution is not imposed data.
    win = ntuple(3) do d
        active[d] || return 1:ext[d]
        lo = lt.imposed[d][1] ? 1 + RESTRICT_MARGIN : 1
        hi = lt.imposed[d][2] ? ext[d] - RESTRICT_MARGIN : ext[d]
        lo:hi
    end
    for ci in lt.coarse_indices
        _write_covered_patch!(states[ci], src4, win, off, patches[ci])
    end
    return src4
end

# The part of the window this rank owns of one parent patch. `src4[i, j, k]`
# is region-local node (i, j, k) ↔ parent-level node off + (i, j, k) ↔ that
# patch's local node minus its region offset ↔ padded index minus the rank's
# block offset plus the pad.
function _write_covered_patch!(coarse_Q, src4, win, off, parent::Patch)
    dp = parent.decomp
    padc = dp.n_halo_d
    poff = parent.region.offset
    r = ntuple(3) do d
        # Region-local nodes intersected with this rank's owned patch nodes.
        lo = max(first(win[d]), dp.offset[d] + 1 + poff[d] - off[d])
        hi = min(last(win[d]), dp.offset[d] + dp.n_local[d] + poff[d] - off[d])
        lo:hi
    end
    any(isempty, r) && return coarse_Q
    sh = ntuple(d -> off[d] - poff[d] - dp.offset[d] + padc[d], 3)
    if _cpu_storage(coarse_Q)
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

"""
    restrict_level!(solver, states)

Restrict every refined patch's state onto the covered region of its parent,
per conserved component, finest level first, and return `states`. Under the
default `:inject` mode the coincident-node values gather directly (a sampled
[`gather_region!`](@ref), which subsamples the fine lattice in one collective
and avoids one `TransferPlan` per dimension); under `:filter` the invertible
pair's Gaussian filter runs over the fine patch's extent before subsampling,
which is a whole-patch line solve and therefore still serial-only (guarded at
setup). Either way the write-back stops `RESTRICT_MARGIN` coarse nodes short
of the coarse-fine boundary. Runs once per completed step, after the state
filter; a solver without refinement returns immediately. Collective under
`:inject`.
"""
function restrict_level!(solver, states)
    levels = getfield(solver, :levels)
    for ℓ in length(levels):-1:2, lt in levels[ℓ].transfers
        _restrict_patch!(solver, states, lt)
    end
    return states
end

function _restrict_patch!(solver, states, lt::LevelTransfer)
    patches = getfield(solver, :patches)
    fine = patches[lt.fine_index]
    Qf = states[lt.fine_index]
    if lt.restriction === :filter
        K = length(lt.rplans)
        padb = lt.rdecomps[1].n_halo_d
        ext = lt.region.extent
        for c in 1:solver.equations.n_cons
            src = view(Qf, :, :, :, c)
            for k in K:-1:1
                dst = lt.rstage[k]
                input = k == K ? src : lt.rstage[k+1]
                restrict!(dst, lt.rplans[k], input)
            end
            # The chain's output is padded region-shaped scratch; the shared
            # write-back takes the unpadded region form (serial-only, so the
            # copy is one small array per component).
            view(lt.restricted, :, :, :, c) .=
                view(lt.rstage[1], (1 + padb[1]):(ext[1] + padb[1]),
                     (1 + padb[2]):(ext[2] + padb[2]),
                     (1 + padb[3]):(ext[3] + padb[3]))
        end
        _write_covered_region!(lt.restricted, lt, states, patches)
        return states
    end
    dfine = fine.decomp
    fr = ntuple(d -> 1:(dfine.n_global[d]), 3)
    sample = ntuple(d -> dfine.active[d] ? 3 : 1, 3)
    gather_region!(lt.restricted, fr, (0, 0, 0), (0, 0, 0), Qf, dfine,
                   lt.fine_blocks, sample; buffers=lt.restrict_buffers)
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
# values at its stage times BETWEEN the parent's t^n and t^{n+1}. The parent
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
    save_level_box!(lt, patches, states, dQs, at_end)

Gather the parent state and its RHS over the buffered prolongation box into
the [`LevelTransfer`](@ref)'s Hermite storage: the `t^n` slots when `at_end`
is false, the `t^n + dt` slots when true. `patches`, `states` and `dQs` are
the solver's full vectors. Collective over the parent patches'
communicators (two replicated-box gathers); the box data is then identical
on every rank, so the Hermite shell evaluation is communication-free at
every fine stage.
"""
function save_level_box!(lt::LevelTransfer, patches, states, dQs, at_end::Bool)
    boxQ = at_end ? lt.box_Q1 : lt.box_Q0
    boxdQ = at_end ? lt.box_dQ1 : lt.box_dQ0
    _gather_box!(boxQ, lt, states, patches)
    _gather_box!(boxdQ, lt, dQs, patches)
    return lt
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
(`src/regrid.jl`): the regrid cadence in coarse steps, the tagging threshold on
the relative undivided fourth difference of the mixture density, the buffer of
coarse cells added around tagged cells, the nesting margin, and everything a
fine-patch rebuild needs that the `Solver` does not itself retain: the
schemes, halo width, interface treatment, and backend. `last_step` records the
step of the most recent regrid check so a run resumed on the same solver keeps
the cadence. Constructed by the [`Solver`](@ref) constructor's
`regrid_interval` keyword; consumed by `regrid!`.
"""
mutable struct RegridSpec{T}
    interval::Int
    threshold::T
    buffer::Int
    margin::Int
    n_halo::Int
    interface_rhs::Symbol
    deriv::CompactScheme{T}
    filt::CompactScheme{T}
    smoo::CompactScheme{T}
    backend::AbstractBackend
    tile::Int                        # lattice edge; 0 = one box over the tags
    last_step::Int
end

"""
    hermite_level_shell!(solver, states, lt, θ, dt)

Impose the shell (ghost ring plus boundary planes) of the refined patch of
`lt` from the cubic Hermite reconstruction of its parent's solution at
fraction `θ` of the parent step of size `dt`, through the same order-6
interpolation chain [`prolong_level_ghosts!`](@ref) uses. Requires both
endpoint slots filled by [`save_level_box!`](@ref); at `θ = 0` the result is
exactly the parent's `t^n` state and the imposition reduces to the
unsubcycled one.
"""
function hermite_level_shell!(solver, states, lt::LevelTransfer, θ, dt)
    _impose_shell!(solver, states, lt,
                   (dst, c) -> _hermite_box!(dst, lt, c, θ, dt))
    return states
end
