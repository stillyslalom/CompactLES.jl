# The level hierarchy and its distribution: the point-to-point coupling
# between a level and its parent, the `Level` record, the tile lattice, the
# device scratch of a level transfer, the replicated gathers, and the
# placement of levels on a domain boundary and across a periodic seam. The
# header of levels_types.jl describes the design.

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

# Every face kept: the carry of a tile whose region survives copies all of
# its nodes, so that a tile moved to new owners continues as one left in
# place does (`_migrate_tile!`, `_swap_level!`).
const _EVERY_FACE = ((true, true), (true, true), (true, true))

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
block, in a `Vector{Union{Nothing,Decomp{T}}}` whatever the rank holds: a
vector typed by its values (all `nothing`, all decompositions, or a mix)
would compile this function once more on each rank meeting another of those
types. Collective over `comm`, the parent level's communicator: two
Allgathervs, of the parent's blocks and of the tiles' blocks, from which every
rank derives the pieces it sends and receives without further communication.
"""
function build_level_coupling(::Type{T}, comm::MPI.Comm,
                              transfers::Vector{LevelTransfer{T}},
                              parent_regions::Vector{BlockRegion},
                              parent_decomps::Vector{Union{Nothing,Decomp{T}}},
                              fine_decomps::Vector{Union{Nothing,Decomp{T}}}) where {T}
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
and the same-level interface records among the level's tiles (at the root,
among the slabs of a `patch_grid`). `solver.levels` holds them in ascending order, and
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
rebalance moves it (`_place_tiles`, `src/amr/levels_types.jl`). A regridded tiled
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
# A refined level is tiled on a lattice of edge `tile[d]` parent nodes along
# dimension d, anchored at parent node 0: lattice cell k spans parent nodes
# k·tile + 1 .. (k+1)·tile + 1 along each dimension, so abutting tiles share
# their interface plane (as root slabs do) and every tile face is either
# fully shared with one neighbor or fully fed from the parent. Tiles at the
# domain edge are clipped to the nesting margin and dropped when clipping
# leaves them below the four-node minimum. The lattice is global so that a
# regrid moves tiles in and out of the set without ever changing a surviving
# tile's region; a regrid that changes the lattice (`_choose_lattice`,
# regrid_swap.jl) replaces every tile at once.

# The lattice edge per dimension of a `tile` keyword: one integer for every
# dimension, or a tuple of one to three edges, x first, a dimension it omits
# taking its last entry. Zero along every dimension is one patch per level.
_tile_edges(tile::Integer) = (Int(tile), Int(tile), Int(tile))
_tile_edges(tile::NTuple{3,Int}) = tile
function _tile_edges(tile::Tuple)
    (1 <= length(tile) <= 3 && all(e -> e isa Integer, tile)) ||
        throw(ArgumentError("tile must be an integer or a tuple of one to three " *
                            "integers, x first; got $tile"))
    return ntuple(d -> Int(tile[min(d, length(tile))]), 3)
end
_tile_edges(tile) =
    throw(ArgumentError("tile must be an integer or a tuple of one to three " *
                        "integers, x first; got $tile"))

# Whether a `tile` setting tiles its levels rather than covering each with
# one box.
_tiled(tile) = any(>(0), _tile_edges(tile))

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
function _lattice_tile(k::NTuple{3,Int}, active::NTuple{3,Bool}, tile::NTuple{3,Int},
                       lo::NTuple{3,Int}, hi::NTuple{3,Int}, last::NTuple{3,Int}=hi)
    any(d -> active[d] && k[d] * tile[d] + 1 >= last[d], 1:3) && return nothing
    offext = ntuple(3) do d
        active[d] || return (0, 1)
        a = k[d] == 0 ? lo[d] : max(k[d] * tile[d] + 1, lo[d])
        top = k[d] * tile[d] + tile[d] + 1
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
function _level_tiles(box::BlockRegion, active::NTuple{3,Bool}, tile::NTuple{3,Int},
                      lo::NTuple{3,Int}, hi::NTuple{3,Int}, last::NTuple{3,Int}=hi,
                      wrap::NTuple{3,Int}=(0, 0, 0))
    spans = ntuple(3) do d
        active[d] || return [0]
        parts = _wrapped_parts(box.offset[d] + 1, box.offset[d] + box.extent[d],
                               wrap[d])
        sort!(unique!(reduce(vcat, [collect(_tile_span(a, b, tile[d]))
                                    for (a, b) in parts])))
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
# either side of the seam abut there. Where several tiles qualify the one
# last in `regions` is taken. The tiles are found through a table keyed on
# the plane and the transverse offsets and extents, so the cost is linear in
# the tile count: a level of a cluster run holds tens of thousands of tiles,
# and the model of the regrid prices several lattices of them per check.
function _tile_faces(regions::Vector{BlockRegion}, period::NTuple{3,Int}=(0, 0, 0))
    n = length(regions)
    node(a, d) = period[d] == 0 ? a : mod(a, period[d])
    # The key of a tile's face along `d` on the plane `plane`: the plane and
    # the transverse offsets and extents, the dimension along `d` zeroed.
    key(r, d, plane) = (d, ntuple(e -> e == d ? node(plane, d) : node(r.offset[e], e), 3),
                        ntuple(e -> e == d ? 0 : r.extent[e], 3))
    # Per key, the last two tiles with a face there, latest first: a tile
    # holds one face per key, so the latest other than a given tile is one of
    # the two.
    low = Dict{Tuple{Int,NTuple{3,Int},NTuple{3,Int}},NTuple{2,Int}}()
    high = Dict{Tuple{Int,NTuple{3,Int},NTuple{3,Int}},NTuple{2,Int}}()
    sizehint!(low, 3n)
    sizehint!(high, 3n)
    for q in 1:n, d in 1:3
        r = regions[q]
        for (table, plane) in ((low, r.offset[d]), (high, r.offset[d] + r.extent[d] - 1))
            k = key(r, d, plane)
            table[k] = (q, first(get(table, k, (0, 0))))
        end
    end
    # The latest tile of a key's pair other than `p`, 0 where none.
    other(pair, p) = pair[1] == p ? pair[2] : pair[1]
    faces = Vector{NTuple{3,NTuple{2,Int}}}(undef, n)
    for p in 1:n
        rp = regions[p]
        faces[p] = ntuple(3) do d
            lo = other(get(high, key(rp, d, rp.offset[d]), (0, 0)), p)
            hi = other(get(low, key(rp, d, rp.offset[d] + rp.extent[d] - 1), (0, 0)), p)
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
# their solver indices and `decomps` their decompositions, all aligned; the
# decompositions come typed, since a rank holding none would otherwise pass
# a `Vector{Any}` and compile this anew when it drops its last tile.
# Collective over `comm`, the level's own communicator (one Allgatherv per
# dimension of every rank's blocks of the tiles it holds; a rank outside the
# subset must not call). Each record's `partner` is therefore a rank number
# in that communicator, the one `_sync_level_records!` exchanges over, and a
# plane record's `partner_pid` stays the partner's tile id, which
# `_seed_planes!` reads. `period` is the level's own node-space period
# (`_level_period`), across whose seam two tiles abut. Returns the tuple the
# `Level{T}` constructor takes.
function _level_records(::Type{T}, comm::MPI.Comm, regions::Vector{BlockRegion},
                        tiles::Vector{Int}, patch_indices::Vector{Int},
                        decomps::Vector{Decomp{T}}, n_cons::Int,
                        period::NTuple{3,Int}=(0, 0, 0)) where {T}
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
            _exchange_patch_state!(states, pi, _cold(patches[pi]))
        end
    end
    _charge_exchange!(solver, lev, t0)
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
            _exchange_patch_state!(states, pi, _cold(patches[pi]))
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
        _exchange_patch_state!(states, pi, _cold(patches[pi]))
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
# Allgatherv. Where a coarse-fine face reads the gradient ring
# (`ShellGradients`), the conserved gradients of the ring are taken there
# too, through device plans on the chain's fine box, and the gradient ring
# stays on the device for the right-hand side that reads it. The scratch
# below is the storage that takes: it lives on the fine `Patch` (typed by the
# patch's array type, so `LevelTransfer`, `Level` and `Solver` keep their
# types) and is empty on the host backend, whose chain runs on the transfer's
# own host stages.

"""
    LevelScratch

Device storage of one refined patch's level transfer: the four Hermite
boxes (uploaded by `save_level_boxes!`, which fills this rank's own
components), the interpolation
chain's stages 0 .. K over this rank's own components (the
component-distributed chain of `_impose_shell!`) and, where the gradient
ring is built (`_gradient_ring_built`), the backend's form of
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
    b, kk = _unchecked_div(kb - 1, n3), _unchecked_rem(kb - 1, n3)
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
    b, kk = _unchecked_div(kb - 1, n2), _unchecked_rem(kb - 1, n2)
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
            g1 = lo[1] + _unchecked_rem(r, n1)
            r = _unchecked_div(r, n1)
            g2 = lo[2] + _unchecked_rem(r, n2)
            g3 = lo[3] + _unchecked_div(r, n2)
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
                             Vector{NTuple{3,UnitRange{Int}}}[], zeros(3))
    else
        # The slabs of the parent-fed faces, which hold every shell slot: a
        # face shared with a same-level tile takes its ghosts from the level's
        # records, and the chain does not fill the box beyond it.
        slabs = _ring_slabs(region, ntuple(d -> fine_decomp.active[d], 3),
                            fine_decomp.n_halo_d, boundary, folded, imposed)
        table, ringlen = _slab_table(slabs)
        # Component-distributed chains: rank r owns components r+1, r+1+npf, ...
        ring_counts = [ringlen * length((r+1):np_tile:n_cons)
                       for r in 0:np_tile-1]
        fine_active = ntuple(d -> fine_decomp.active[d], 3)
        _check_ring_cover(table, region, fine_active, fine_decomp.n_halo_d, imposed,
                          boundary, folded)
        boxes = _chain_boxes(slabs, _box_shift(buffer, folded),
                             _chain_extents(boxext, dims_to_refine), dims_to_refine,
                             interpolation_order)
        shell = ShellRing{T}(slabs, table, ringlen,
                             Matrix{T}(undef, ringlen, n_cons),
                             Matrix{T}(undef, ringlen, n_cons), ring_counts,
                             GatherBuffers{T}(), boxes, zeros(3))
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
    equations = solver.equations
    for lt in lev.transfers
        lt.fine_index == 0 && continue
        any(any, lt.folded) || continue
        A = select(lt)
        isempty(A) && continue
        box = _box_nodes(lt)
        pad = lt.pdecomps[1].n_halo_d
        for d in 1:3, side in 1:2
            lt.folded[d][side] || continue
            # Padded index of parent node n along d.
            shift = 1 + pad[d] - first(box[d])
            _mirror_box_layers!(A, equations, root.folds[d].sigvel, d, side,
                                first(span[d]) + shift, last(span[d]) + shift)
        end
    end
    return lev
end

# The `LEVEL_BUFFER` layers of `A` beyond the fold along `d`, each the signed
# copy of its mirror node, for every component. `lo` and `hi` are the padded
# indices of the span's first and last parent node. A function barrier with
# explicit loops: the slab views of a broadcast, `OneTo` across and a
# `UnitRange` along `d`, have a type that depends on `d`, so each copy was a
# dynamic dispatch.
function _mirror_box_layers!(A::Array{T,4}, equations, sigvel, d::Int, side::Int,
                             lo::Int, hi::Int) where {T}
    n1, n2, n3 = size(A, 1), size(A, 2), size(A, 3)
    for c in axes(A, 4)
        σ = T(conserved_parity(equations, sigvel, c))
        for m in 1:LEVEL_BUFFER
            dst, src = side == 1 ? (lo - m, lo + m - 1) : (hi + m, hi + 1 - m)
            if d == 1
                for k in 1:n3, j in 1:n2
                    A[dst, j, k, c] = σ * A[src, j, k, c]
                end
            elseif d == 2
                for k in 1:n3, i in 1:n1
                    A[i, dst, k, c] = σ * A[i, src, k, c]
                end
            else
                for j in 1:n2, i in 1:n1
                    A[i, j, dst, c] = σ * A[i, j, src, c]
                end
            end
        end
    end
    return A
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
# of boxes, not one box), or in a periodic image of one under `period`.
function _covered_by(region::BlockRegion, regions::Vector{BlockRegion},
                     period::NTuple{3,Int}=(0, 0, 0))
    inside(g, r) = all(d -> r.offset[d] < g[d] <= r.offset[d] + r.extent[d], 1:3)
    # Only the images meeting the region can cover a node of it, and one
    # holding the whole region settles the test without the node scan.
    meets(r) = all(d -> r.offset[d] < region.offset[d] + region.extent[d] &&
                        region.offset[d] < r.offset[d] + r.extent[d], 1:3)
    images = period == (0, 0, 0) ? filter(meets, regions) :
             [s for r in regions for σ in _images(period) for s in (_shifted(r, σ),)
              if meets(s)]
    any(r -> inside(region.offset .+ 1, r) &&
             inside(region.offset .+ region.extent, r), images) && return true
    # Along each dimension the region's nodes split into runs between the
    # planes where an image starts or ends; every node of a cell of runs lies
    # in the same images, so the first node of each cell stands for it.
    cuts = ntuple(3) do d
        lo, hi = region.offset[d] + 1, region.offset[d] + region.extent[d]
        c = [lo]
        for r in images, e in (r.offset[d] + 1, r.offset[d] + r.extent[d] + 1)
            lo < e <= hi && push!(c, e)
        end
        sort!(unique!(c))
    end
    for k in cuts[3], j in cuts[2], i in cuts[1]
        any(r -> inside((i, j, k), r), images) || return false
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
