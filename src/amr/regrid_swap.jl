# The swap of a regrid: the carry and the migration of tile data, the tiled
# regrid of one level, and the level-by-level regrid of a deeper hierarchy.
# The header of regrid_tagging.jl describes the regrid.

const _CARRY_TAG = 1550

"""
    _carry_tiles!(T, comm, active, period, old_fine, old_own, old_owners,
                  new_fine, new_owners, targets, old_data, new_data)

After a change of lattice, the evolved solution of a level's old tiles onto
the new tiles `targets` marks: every node of a new tile that an old tile
holds as its own solution (`old_own`, its fine region less its parent-fed
planes) takes the old tile's value, over the periodic images of the old
tile under the level's fine node-space `period`. The regions are fine
regions in the level's own node space, and every tile is held whole by the
first rank of its owner range. One message per overlapping pair, old owner
to new owner on `comm`, whose rank numbers the owner ranges use; every rank
derives the same message list from the layouts, which every rank passes
alike, and posts only its own. `old_data[i]` and `new_data[j]` are this
rank's state and decomposition of old tile `i` and new tile `j`, absent
where it holds none. The sends complete before this returns.
"""
function _carry_tiles!(::Type{T}, comm::MPI.Comm, active::NTuple{3,Bool},
                       period::NTuple{3,Int}, old_fine::Vector{BlockRegion},
                       old_own::Vector{BlockRegion}, old_owners::Vector{UnitRange{Int}},
                       new_fine::Vector{BlockRegion}, new_owners::Vector{UnitRange{Int}},
                       targets::AbstractVector{Bool}, old_data::AbstractDict,
                       new_data::AbstractDict) where {T}
    me = MPI.Comm_rank(comm)
    reqs = MPI.Request[]
    recvs = Tuple{Int,NTuple{3,UnitRange{Int}},Vector{T}}[]
    sends = Vector{T}[]
    for j in eachindex(new_fine), i in eachindex(old_fine)
        targets[j] || continue
        src, dst = first(old_owners[i]), first(new_owners[j])
        (me == src || me == dst) || continue
        nf = new_fine[j]
        for σ in _images(period)
            o = _shifted(old_own[i], σ)
            g = ntuple(d -> (max(o.offset[d], nf.offset[d]) + 1):
                            min(o.offset[d] + o.extent[d], nf.offset[d] + nf.extent[d]), 3)
            any(isempty, g) && continue
            rnew = ntuple(d -> g[d] .- nf.offset[d], 3)
            rold = ntuple(d -> g[d] .- (old_fine[i].offset[d] + σ[d]), 3)
            if me == src && me == dst
                Q_old, d_old = old_data[i]
                Q_new, d_new = new_data[j]
                lo, ln = _padded3(rold, d_old), _padded3(rnew, d_new)
                if _cpu_storage(Q_new) && _cpu_storage(Q_old)
                    _copy_block!(Q_new, ln, Q_old, lo)
                else
                    view(parent(Q_new), ln[1], ln[2], ln[3], :) .=
                        view(parent(Q_old), lo[1], lo[2], lo[3], :)
                end
            elseif me == dst
                Q_new, _ = new_data[j]
                buf = Vector{T}(undef, size(Q_new, 4) * prod(length.(rnew)))
                push!(reqs, MPI.Irecv!(buf, comm; source=src, tag=_CARRY_TAG))
                push!(recvs, (j, rnew, buf))
            else
                Q_old, d_old = old_data[i]
                buf = _pack_message(T, Q_old, _padded3(rold, d_old))
                push!(sends, buf)
                push!(reqs, MPI.Isend(buf, comm; dest=dst, tag=_CARRY_TAG))
            end
        end
    end
    MPI.Waitall(reqs)
    for (j, rnew, buf) in recvs
        Q_new, d_new = new_data[j]
        _unpack_message!(Q_new, buf, _padded3(rnew, d_new))
    end
    return nothing
end

# This rank's state and decomposition of each tile it holds, by tile index,
# from `local_of` (each tile's patch index, 0 where the rank holds none): the
# new tiles' side of `_carry_tiles!`, typed as the old tiles' side is, so
# the one-level regrid and the hierarchy's share one specialization of it.
function _held_tile_data(::Type{T}, states::Vector{S}, patches,
                         local_of::Vector{Int}) where {T,S}
    data = Dict{Int,Tuple{S,Decomp{T}}}()
    for (ti, li) in enumerate(local_of)
        li == 0 || (data[ti] = (states[li], patches[li].decomp))
    end
    return data
end

# One tile's whole state, replicated on every rank of the parent level's
# communicator. The box regrid carries its moved region through this (one
# region, whose old and new decompositions differ in offset as well as in
# rank set); a tiled level migrates a moved tile block to block instead
# (`_migrate_tile!`), and this gather is its audit reference under
# `MIGRATION_AUDIT`. Collective over that communicator; a rank with no block
# of the tile contributes nothing.
function _gather_tile(::Type{T}, region::BlockRegion, active::NTuple{3,Bool},
                      n_cons::Int, lt::LevelTransfer, solver, states) where {T}
    held = lt.fine_index != 0
    return _gather_replicated(T, region, active, n_cons,
                              held ? (states[lt.fine_index],
                                      getfield(solver, :patches)[lt.fine_index].decomp) :
                                     nothing, lt.parent_comm, lt.folded)
end

# The periodic image of `old` (under `period`) that overlaps `new` the most,
# `old` itself when none does: the placement of a box that moved across a
# periodic seam, against which its surviving nodes are carried.
function _nearest_image(old::BlockRegion, new::BlockRegion, period::NTuple{3,Int})
    overlap(r) = prod(max(0, min(r.offset[d] + r.extent[d], new.offset[d] + new.extent[d]) -
                          max(r.offset[d], new.offset[d])) for d in 1:3)
    best = old
    for σ in _images(period)
        r = _shifted(old, σ)
        overlap(r) > overlap(best) && (best = r)
    end
    return best
end

# The faces on the domain boundary for both of two regions, whose planes a
# carry or a migration copies (`_carry_over!`, `_migrate_tile!`).
_shared_boundary(a::NTuple{3,NTuple{2,Bool}}, b::NTuple{3,NTuple{2,Bool}}) =
    ntuple(d -> (a[d][1] && b[d][1], a[d][2] && b[d][2]), 3)

# A tile's block table over `comm` for `_migrate_tile!`: every rank's owned
# block of it, from `decomp` on the ranks holding one (one Allgather; the
# share fields are unused there). Collective over `comm`.
_block_table(decomp::Union{Nothing,Decomp}, comm::MPI.Comm) =
    [OwnedBlock(r - 1, b, 0, 1) for (r, b) in enumerate(_owned_blocks(decomp, comm))
     if prod(b.extent) > 0]

# Tag of the migration messages. They run on the root communicator, on
# which nothing else is in flight during a regrid, and one tile's messages
# complete before the next tile's are posted, so one tag serves every tile.
const _MIGRATE_TAG = 1500

# Padded local ranges of the patch-node ranges `r` on the block of `decomp`.
_padded3(r::NTuple{3,UnitRange{Int}}, decomp::Decomp) =
    ntuple(d -> _padded(r[d], decomp.offset[d], decomp.n_halo_d[d]), 3)

# One message of the migration: the padded ranges `lr` of `Q`, packed into
# a fresh host buffer. Device storage packs by broadcast into a device stage
# and copies once, as `gather_region!` does.
function _pack_message(::Type{T}, Q, lr::NTuple{3,UnitRange{Int}}) where {T}
    n = size(Q, 4) * prod(length.(lr))
    buf = Vector{T}(undef, n)
    if _cpu_storage(Q)
        _pack!(buf, Q, lr)
    else
        v = view(parent(Q), lr[1], lr[2], lr[3], 1:size(Q, 4))
        dsend = _device_send_stage(parent(Q), n)
        reshape(view(dsend, 1:n), size(v)) .= v
        _tracked_copy!(buf, 1, dsend, 1, n)
    end
    return buf
end

function _unpack_message!(Q, buf::Vector, lr::NTuple{3,UnitRange{Int}})
    _cpu_storage(Q) && return _unpack!(Q, buf, lr)
    v = view(parent(Q), lr[1], lr[2], lr[3], 1:size(Q, 4))
    dev = similar(parent(Q), size(v))
    copyto!(dev, reshape(buf, size(v)))
    v .= dev
    return Q
end

# Point-to-point migration of one surviving tile's solution from its old
# owners' blocks to its new owners'. `old_blocks` and `new_blocks` are the
# tile's block tables (the old and the new level's `LevelCoupling`), each
# rank named in `comm`, so every rank derives the identical message list,
# old block ∩ new block ∩ interior per rank pair, and posts only the sends
# and receives of its own; a rank with neither posts nothing, and no
# collective runs. The interior holds one node off both boundary planes
# along every active dimension but those of the faces in `keep`, the rule
# `_carry_over!` applies. A tile whose region survives keeps every face
# (`_EVERY_FACE`): its planes shared with neighbors are solution, its
# imposed planes are re-imposed by the next shell fill on a tile left in
# place as on a moved one, and with the whole tile carried the moved tile
# continues bit for bit as the tile left in place would. `Q_old`/`d_old`
# are this rank's old block of the tile and `Q_new`/`d_new` its new one,
# `nothing` where it holds none. The sends complete before this returns, so
# the caller may free the old decomposition afterwards.
#
# `ghosts` also carries the ghost layers beyond each kept face of the tile.
# The hierarchy regrid sweeps a level's tiles for the next level's tags
# before any exchange, and the sweep reads those layers at shared and
# parent-fed faces (`_tag_sweep!`). A tile left in place holds its
# neighbor's nodes or the shell there; a moved tile that carries them sweeps
# to the same tags, whichever ranks hold it.
function _migrate_tile!(::Type{T}, Q_new, d_new, Q_old, d_old, Nf::NTuple{3,Int},
                        active::NTuple{3,Bool}, old_blocks::Vector{OwnedBlock},
                        new_blocks::Vector{OwnedBlock}, comm::MPI.Comm,
                        keep::NTuple{3,NTuple{2,Bool}}=_NO_BOUNDARY;
                        ghosts::Bool=false) where {T}
    me = MPI.Comm_rank(comm)
    dref = d_new === nothing ? d_old : d_new
    g = ntuple(d -> ghosts && dref !== nothing && active[d] ? dref.n_halo_d[d] : 0, 3)
    interior = ntuple(d -> active[d] ?
                      ((keep[d][1] ? 1 - g[d] : 2):(Nf[d] + (keep[d][2] ? g[d] : -1))) :
                      (1:1), 3)
    # A block on a face of the tile reaches into the ghost layers beyond it.
    nodes(b::BlockRegion) =
        ntuple(d -> (b.offset[d] + 1 - (b.offset[d] == 0 ? g[d] : 0)):
                    (b.offset[d] + b.extent[d] +
                     (b.offset[d] + b.extent[d] == Nf[d] ? g[d] : 0)), 3)
    piece(a::BlockRegion, b::BlockRegion) =
        ntuple(d -> _isect(_isect(nodes(a)[d], nodes(b)[d]), interior[d]), 3)
    wanted(r) = !any(isempty, r)
    mine(blocks) = (i = findfirst(e -> e.rank == me, blocks);
                    i === nothing ? nothing : blocks[i].block)
    reqs = MPI.Request[]
    recvs = Tuple{NTuple{3,UnitRange{Int}},Vector{T}}[]
    if d_new !== nothing
        for a in old_blocks
            a.rank == me && continue
            r = piece(a.block, mine(new_blocks))
            wanted(r) || continue
            buf = Vector{T}(undef, size(Q_new, 4) * prod(length.(r)))
            push!(reqs, MPI.Irecv!(buf, comm; source=a.rank, tag=_MIGRATE_TAG))
            push!(recvs, (r, buf))
        end
    end
    sends = Vector{T}[]
    if d_old !== nothing
        for nb in new_blocks
            b = nb.rank
            r = piece(mine(old_blocks), nb.block)
            wanted(r) || continue
            if b == me
                # A node this rank keeps moves between its own two blocks.
                lo, ln = _padded3(r, d_old), _padded3(r, d_new)
                if _cpu_storage(Q_new)
                    _copy_block!(Q_new, ln, Q_old, lo)
                else
                    view(parent(Q_new), ln[1], ln[2], ln[3], :) .=
                        view(parent(Q_old), lo[1], lo[2], lo[3], :)
                end
            else
                buf = _pack_message(T, Q_old, _padded3(r, d_old))
                push!(sends, buf)
                push!(reqs, MPI.Isend(buf, comm; dest=b, tag=_MIGRATE_TAG))
            end
        end
    end
    MPI.Waitall(reqs)
    for (r, buf) in recvs
        _unpack_message!(Q_new, buf, _padded3(r, d_new))
    end
    return Q_new
end

# Whether this regrid check repartitions the level on measured load, and
# the measurement when it does: each rank's busy time over the interval
# since the last check (its step wall less its time inside the run-wide
# collectives, less its own regrid work), Allgathered so every rank holds
# the same vector. The ratio of the largest to the mean must exceed
# `spec.rebalance` at `spec.persist` consecutive checks. Collective over the
# root communicator whenever rebalancing is on, and every rank takes the
# same decision from the same reduced data; a rank-local decision here
# would be a deadlock, since the ranks would then split different
# communicators. Returns `nothing` when nothing is due.
function _rebalance_due!(solver::Solver, spec::RegridSpec)
    spec.rebalance > 0 || return nothing
    busy = (solver.wall_total - spec.wall_mark) -
           (solver.wait_total - spec.wait_mark) - spec.wall_regrid
    spec.wall_mark = solver.wall_total
    spec.wait_mark = solver.wait_total
    spec.wall_regrid = 0.0
    walls = MPI.Allgather(max(busy, 0.0), getfield(solver, :comm))
    mean = sum(walls) / length(walls)
    spec.imbalance = mean > 0 ? maximum(walls) / mean : 1.0
    spec.streak = spec.imbalance > spec.rebalance ? spec.streak + 1 : 0
    spec.streak >= spec.persist || return nothing
    spec.streak = 0
    return walls
end

# A kept tile's arrays, old storage to new: its state and every persistent
# field a `Patch` carries (the primitives, the artificial coefficients the
# next `max_rate` reads, the geometry). Element copies, so the tile continues
# bit for bit; the stacked device level takes this in place of `_repatch`'s
# sharing, since its new stack is a new allocation.
function _copy_tile!(pnew::Patch, Qnew, pold::Patch, Qold)
    _assign!(parent(Qnew), parent(Qold))
    for name in (:rho, :u, :v, :w, :p, :T_ion, :c, :cp_mix, :mu_art, :beta_art,
                 :kappa_art, :inv_J, :inv_r, :cot_over_r)
        _assign!(getfield(pnew, name), getfield(pold, name))
    end
    _assign!(pnew.amr.overwritten, pold.amr.overwritten)
    for name in (:Y, :D_art, :area_d, :inv_h)
        for (a, b) in zip(getfield(pnew, name), getfield(pold, name))
            _assign!(a, b)
        end
    end
    copyto!(pnew.amr.covered, pold.amr.covered)
    return pnew
end

function _regrid_tiles!(solver::Solver{T}, states::Vector{<:ConservedState},
                        workspace::Workspace, save) where {T}
    spec = getfield(solver, :regrid)
    schemes = getfield(solver, :schemes)
    levels = getfield(solver, :levels)
    patches = getfield(solver, :patches)
    lev = levels[2]
    n_global = solver.n_global
    active = ntuple(d -> n_global[d] > 1, 3)
    # The lattice clip: the nesting margin inside every face but those the
    # lattice reaches under `level_boundaries` (`_lattice_reach`), and none
    # across a periodic seam, where the lattice wraps (`_placement_period`).
    reach = _placed_reach(solver, spec, n_global, active)
    wrap = _placement_period(solver, spec, 0)
    lo, hi = _wrap_feasible(_feasible_nodes(n_global, active, spec.margin, reach)..., wrap)
    root_lc = levels[1].level_comm
    chain = _level_chain(solver, spec, n_global, active, root_lc.size)
    current = findfirst(==(_level_edges(spec, 1)), chain)
    flagsets, Ks, banded = _tag_tiles(solver, states[1], spec, lo, hi, wrap, chain)
    banded && _warn_margin_band(solver, spec.margin, 1)
    busy = _rebalance_due!(solver, spec)
    _harvest_lattice_costs!(solver, spec)
    old_regions = [lt.region for lt in lev.transfers]
    # With no tile and no tag there is nothing to build or move; the flags
    # are reduced and the tile set is held by every rank, so this return is
    # uniform.
    isempty(old_regions) && all(f -> all(==(0), f), flagsets) && return false
    # A cell at the tag level is wanted; one at the hold level, or younger
    # than the lifetime, is wanted where its tile exists (the hysteresis).
    # `created` is derived from the reduced flags, so every rank holds it,
    # and so is the choice of lattice.
    period = _level_period(solver, 0)
    # Under the positivity limiter a tile too short for its face form, over
    # a partial cell at a domain face or at the margin, is not created
    # (`_limiter_tile_ok`), and the parent covers that cell.
    wanted_sets = [filter!(t -> _limiter_tile_ok(solver, t, 1),
                           _lattice_wanted(flagsets[i], Ks[i], chain[i], active, lo, hi,
                                           spec, old_regions, i == current, period))
                   for i in eachindex(chain)]
    pick = _choose_lattice(solver, spec, 1, chain, wanted_sets, current, root_lc.size,
                           isempty(old_regions))
    wanted = wanted_sets[pick]
    rechopped = pick != current && !isempty(old_regions)
    _set_level_edges!(spec, 1, chain[pick])
    # An empty wanted set removes every tile: the level stays in the
    # hierarchy with no tiles, and the next tag builds its first again.
    wanted != old_regions || busy !== nothing || return false
    old_set, wanted_set = Set(old_regions), Set(wanted)
    old_bins = _RegionIndex(old_regions, period)
    created = _created(spec, 1)
    inherited = Dict(r => _inherited_check(r, view(old_regions, _candidates(old_bins, r)),
                                           created, spec.checks, active, period)
                     for r in wanted if rechopped && !(r in old_set))
    for r in old_regions
        r in wanted_set || delete!(created, r)
    end
    for r in wanted
        haskey(created, r) || (created[r] = get(inherited, r, spec.checks))
    end
    old_lc = lev.level_comm
    # Old and new tile of a region: the transfers are held by every rank of
    # the parent's subset, the patches only by a tile's owners.
    old_of = Dict(r => i for (i, r) in enumerate(old_regions))
    new_of = Dict(r => i for (i, r) in enumerate(wanted))
    n_cons = solver.equations.n_cons
    # Ownership is stored: a surviving tile keeps its range and a fresh one
    # is placed among the ranks left free or on the least loaded group, so
    # the tile set changing around a survivor does not move it unless the
    # placement has drifted from a balanced one. A rebalance repartitions
    # the whole level instead, weighted by the costs the last interval
    # measured. Either way the inputs are the reduced tile flags and the
    # owners every rank holds, so every rank derives the same assignment. A
    # tile stays in place, arrays and state included, when both its region
    # and its range survive; otherwise it is rebuilt, and one whose region
    # survived takes its solution back below. The level's rank count moves when a fresh
    # tile takes ranks beyond the old level or a departure vacates its top
    # ranks; a survivor's Cartesian communicator is independent of the
    # level communicator the resize replaces, so it survives the resize.
    if busy === nothing || isempty(wanted)
        owners, np_new = _place_tiles(wanted, active, root_lc.size,
                                      old_regions, lev.owners)
        # Under a measured rebalance the measured weights govern the
        # partition; a repartition by volume would undo theirs.
        spec.rebalance > 0 ||
            ((owners, np_new) = _balanced_owners(wanted, active, root_lc.size,
                                                 owners, np_new,
                                                 [haskey(old_of, r) for r in wanted];
                                                 nodes=spec.costs.nodes))
    else
        weights = _measured_weights(wanted, active, old_regions, lev.owners, busy)
        owners, np_new = _tile_owners(wanted, active, root_lc.size; weights)
    end
    resized = np_new != old_lc.size
    kept = [haskey(old_of, r) && lev.owners[old_of[r]] == owners[ti]
            for (ti, r) in enumerate(wanted)]
    # A rebalance that moved nothing, with the tile set unchanged, is no
    # regrid; the decision is reduced, so every rank returns here together.
    wanted == old_regions && all(kept) && return false
    # A tile about to leave restricts once more: the post-step restriction
    # preceded the positivity repair, and nothing else writes it back.
    departing = [!haskey(new_of, r) for r in old_regions]
    any(departing) && _restrict_tiles!(solver, states, lev, departing)
    # The audit reference for each moved tile, gathered replicated over the
    # root communicator while the old decompositions are still live; empty
    # unless the test hook is on.
    carried = Dict{BlockRegion,Array{T,4}}()
    if MIGRATION_AUDIT[]
        for (ti, r) in enumerate(wanted)
            (kept[ti] || !haskey(old_of, r)) && continue
            carried[r] = _gather_tile(T, r, active, n_cons,
                                      lev.transfers[old_of[r]], solver, states)
        end
    end
    # Decompositions of the old tiles this rank holds that do not stay in
    # place, collected before the patch vector is overwritten and freed at
    # the end, after the migration has read them; a surviving tile keeps its
    # decomposition through `_repatch` and must not appear here. A moved
    # tile's old state and decomposition on this rank are held past the swap
    # by tile, for the sends the migration posts from them.
    # On a stacked device level (construction.jl, stacked tiles) every tile is rebuilt
    # into the new stacks, a kept tile included, which then copies its arrays
    # across below and drops its old decomposition like the rest.
    stacked = _stacked_level(spec.backend, spec.tile)
    dropped_decomps = Decomp{T}[]
    old_piece = Dict{Int,Tuple{eltype(states),Decomp{T}}}()
    # Under a change of lattice every old tile this rank holds is a source
    # of the carry, by old tile index.
    carry_src = Dict{Int,Tuple{eltype(states),Decomp{T}}}()
    for (li, ti_old) in zip(lev.patches, lev.tiles)
        rechopped && (carry_src[ti_old] = (states[li], patches[li].decomp))
        r = old_regions[ti_old]
        keeps = haskey(new_of, r) && kept[new_of[r]]
        keeps && !stacked && continue
        push!(dropped_decomps, patches[li].decomp)
        haskey(new_of, r) && !keeps &&
            (old_piece[new_of[r]] = (states[li], patches[li].decomp))
    end
    # Every owner of the old level drops its tile group here, and the level
    # communicator goes with it where the rank count changed; both are
    # collective, and the tile flags are Allreduced, so every rank takes the
    # same path. Surviving tiles' Cartesian communicators are independent
    # of the group they were split from.
    free_tile_group!(lev.group)
    resized && free_level_comm!(old_lc)
    new_lc = resized ? split_level_comm(root_lc, np_new) : old_lc
    group = new_lc.owned ? split_tile_comm(new_lc, owners) : absent_tile_group()
    held = [ti for ti in eachindex(wanted) if owners[ti] == group.ranks]
    root = patches[1]
    period = _level_period(solver, 0)
    faces = _tile_faces(wanted, period)
    boundaries = _region_boundaries(solver, wanted, 1)
    folds = _region_folds(solver, boundaries)
    # The rank's scratch sets before the swap, departing tiles included: a
    # fresh tile of the lattice edge reuses one rather than allocating, and the
    # pool grows only for an extent nothing on the rank already carries (an
    # edge-clipped cell). Whatever no surviving or new patch takes is dropped
    # with the old patch vector.
    ws_pool = [p.rhs_workspace for p in patches]
    old_local = Dict(zip(lev.tiles, lev.patches))
    new_patches = Any[]
    new_states = similar(states, 0)
    new_dQ = similar(workspace.dQ, 0)
    new_du = similar(workspace.du, 0)
    local_of = zeros(Int, length(wanted))
    indices = [k + 1 for k in eachindex(held)]
    for (k, ti) in enumerate(held)
        local_of[ti] = k + 1
    end
    stacks = TileStack[]
    if stacked
        built, stacks = _build_level_patches(T, wanted, held, faces, active, root.h,
                                             spec.n_halo, group.comm, schemes.deriv,
                                             schemes.filt, spec.smoo,
                                             solver.art.smoother, schemes.interface_rhs,
                                             spec.backend, ws_pool,
                                             solver.equations.n_species, n_cons,
                                             _grad_Q_columns(solver),
                                             1, 1, spec.tile;
                                             interface_divergence=
                                                 schemes.interface_divergence,
                                             gradient_ring=_gradient_ring_built(solver),
                                             ghost_fluxes=_ghost_fluxes_built(solver),
                                             ring=_ring_detector(solver),
                                             boundaries, bcs=root.bcs,
                                             root_folds=root.folds,
                                             n_sensed=_sensed_field_count(solver,
                                                                          spec.tile))
        append!(new_patches, built)
        resize!(new_states, length(held))
        resize!(new_dQ, length(held))
        resize!(new_du, length(held))
        for st in stacks
            _stacked_states!(new_states, st, n_cons, -1)
            _stacked_states!(new_dQ, st, n_cons, -1)
            _stacked_states!(new_du, st, n_cons, -1)
        end
        # A kept tile takes its evolved arrays into its new slots: the state,
        # and the persistent fields the next step reads before its first
        # right-hand side (the artificial coefficients through `max_rate`,
        # the geometry). The stage arrays start from zero, as a fresh tile's
        # do; the first stage reads neither.
        for (k, ti) in enumerate(held)
            kept[ti] || continue
            oi = old_local[old_of[wanted[ti]]]
            _copy_tile!(new_patches[k], new_states[k], patches[oi], states[oi])
        end
    else
        for (k, ti) in enumerate(held)
            tr = wanted[ti]
            idx = k + 1
            bcs = _fine_bcs(active, faces[ti], boundaries[ti], root.bcs)
            if kept[ti]
                oi = old_local[old_of[tr]]
                p = _repatch(patches[oi], idx, faces[ti], bcs, group.comm,
                             _survivor_plans(patches[oi], faces[ti], bcs, boundaries[ti],
                                             root.bcs, root.folds, spec, schemes,
                                             solver.art.smoother, _ring_detector(solver)))
                push!(new_states, states[oi])
                push!(new_dQ, workspace.dQ[oi])
                push!(new_du, workspace.du[oi])
            else
                p = _build_fine_patch(T, tr, active, root.h, spec.n_halo,
                                      group.comm, schemes.deriv, schemes.filt,
                                      spec.smoo, solver.art.smoother,
                                      schemes.interface_rhs, spec.backend, ws_pool,
                                      solver.equations.n_species, n_cons,
                                      _grad_Q_columns(solver),
                                      idx, 1, faces[ti];
                                      interface_divergence=schemes.interface_divergence,
                                      gradient_ring=_gradient_ring_built(solver),
                                      ghost_fluxes=_ghost_fluxes_built(solver),
                                      ring=_ring_detector(solver),
                                      boundary=boundaries[ti], bcs=root.bcs,
                                      root_folds=root.folds,
                                      n_sensed=_sensed_field_count(solver,
                                                                   spec.tile),
                                      slot=k)
                Q = _state_like(p.rho, n_cons)
                push!(new_states, Q)
                push!(new_dQ, zero(Q))
                push!(new_du, zero(Q))
            end
            push!(new_patches, p)
        end
    end
    fine_decomp(ti) = local_of[ti] == 0 ? nothing : new_patches[local_of[ti] - 1].decomp
    transfers = LevelTransfer{T}[build_level_transfer(
        T, tr, active, spec.n_halo, [root.region], [1], local_of[ti],
        schemes.level_restriction, n_cons, getfield(solver, :subcycle), fine_decomp(ti),
        root_lc.comm, length(owners[ti]), faces[ti];
        interpolation_order=schemes.level_interpolation_order,
        gradient_deriv=_gradient_ring_built(solver) ? schemes.deriv : nothing,
        parent_h=root.h, boundary=boundaries[ti], folded=folds[ti], period)
        for (ti, tr) in enumerate(wanted)]
    coupling = build_level_coupling(T, root_lc.comm, transfers, [root.region],
                                    Union{Nothing,Decomp{T}}[root.decomp],
                                    Union{Nothing,Decomp{T}}[fine_decomp(ti)
                                                             for ti in eachindex(wanted)])
    _resize_level_patches!(solver, states, workspace, 1 + length(held))
    for (k, p) in enumerate(new_patches)
        patches[k + 1] = p
        states[k + 1] = new_states[k]
        workspace.dQ[k + 1] = new_dQ[k]
        workspace.du[k + 1] = new_du[k]
    end
    if new_lc.owned
        # The records are built on the tiles' fine regions (fine node space),
        # not on the lattice cells `wanted` holds, which are the parent's.
        fine_regions = [_fine_region(tr, active, fo) for (tr, fo) in zip(wanted, folds)]
        records = _level_records(T, new_lc.comm, fine_regions, held, indices,
                                 Decomp{T}[p.decomp for p in new_patches], n_cons,
                                 _level_period(solver, 1))
        levels[2] = Level{T}(1, new_lc, owners, group, held, indices, transfers,
                             records; stacks, coupling)
    else
        levels[2] = Level{T}(1, new_lc, owners, group, held, indices, transfers;
                             stacks, coupling)
    end
    _fill_covered!(root, wanted, period)
    fresh = [ti for ti in eachindex(wanted) if !kept[ti]]
    is_fresh = falses(length(wanted))
    is_fresh[fresh] .= true
    # The tiles whose region is new; a moved survivor is rebuilt as a fresh
    # tile is but carries its whole state, planes included.
    is_new = [!haskey(old_of, r) for r in wanted]
    for ti in fresh
        li = local_of[ti]
        li == 0 || init_geometry!(PatchSolver(solver, patches[li]))
    end
    # The fresh tiles' boxes move in one exchange, entered by every rank.
    isempty(fresh) || _fill_tiles_from_parent!(solver, states, levels[2], is_fresh)
    for ti in fresh
        li = local_of[ti]
        r = wanted[ti]
        haskey(old_of, r) || continue
        # A rebuilt tile whose region survived takes its evolved state
        # back over the interpolated initialization, block to block from
        # its old owners. Both block tables name ranks of the root
        # communicator and every rank holds both, so every rank posts from
        # the same message list.
        Nf = fine_extent(r, active, folds[ti])
        Q_old, d_old = get(old_piece, ti, (nothing, nothing))
        Q_new = li == 0 ? nothing : states[li]
        d_new = li == 0 ? nothing : patches[li].decomp
        audit = li != 0 && haskey(carried, r)
        Qref = audit ? ConservedState(copy(parent(states[li]))) : nothing
        _migrate_tile!(T, Q_new, d_new, Q_old, d_old, Nf, active,
                       lev.coupling.fine_blocks[old_of[r]], coupling.fine_blocks[ti],
                       root_lc.comm, _EVERY_FACE)
        if audit
            _carry_over!(Qref, d_new, r, carried[r], Nf, r, active, n_cons,
                         _EVERY_FACE, folds[ti], folds[ti])
            differ = count(Array(parent(Qref)) .!= Array(parent(states[li])))
            res = MIGRATION_AUDIT_RESULT[]
            MIGRATION_AUDIT_RESULT[] = (tiles=res.tiles + 1,
                                        mismatches=res.mismatches + differ)
        end
    end
    # Under a change of lattice a new tile takes the evolved solution of the
    # old tiles it overlaps over its interpolated initialization.
    if rechopped
        new_data = _held_tile_data(T, states, patches, local_of)
        _carry_tiles!(T, root_lc.comm, active, _level_period(solver, 1),
                      [_fine_region(lt) for lt in lev.transfers],
                      [_erode(_fine_region(lt), lt.imposed, active) for lt in lev.transfers],
                      lev.owners,
                      [_fine_region(tr, active, fo) for (tr, fo) in zip(wanted, folds)],
                      owners, is_new, carry_src, new_data)
    end
    # A new tile's plane shared with a survivor takes the survivor's
    # evolved values rather than averaging its interpolation into them.
    any(is_new) && new_lc.owned && _seed_planes!(solver, states, levels[2], is_new)
    # Every old transfer was replaced above, surviving tiles included, and a
    # departing or rebuilt tile's decomposition has no further reader, the
    # migration's sends having completed inside `_migrate_tile!`. Free
    # their communicators now: left to garbage collection they accumulate at
    # the regrid cadence and exhaust MPI's context-id budget (2048 per process
    # under MPICH). The tile flags are Allreduced, so every rank frees the
    # same communicators.
    for old_lt in lev.transfers
        free_transfer_decomps!(old_lt)
    end
    for decomp in dropped_decomps
        free_communicators!(decomp)
    end
    return true
end

# --- Regridding several levels ----------------------------------------------
#
# A hierarchy of more than one refined level regrids top-down at each check.
# Level ℓ is tagged on level ℓ − 1 after that level has been regridded, by
# the level-1 rule on the lattice of level ℓ − 1's node space; a wanted cell
# must also lie, with the nesting margin, inside the own (not imposed) nodes
# of level ℓ − 1's tiles, and a cell that does not is left out. A level whose
# parent was rebuilt is rebuilt with it, since its transfers, its patch
# indices and its rank subset follow the parent's tiles, and a surviving tile
# keeps its arrays and its state throughout.
#
# The decision of each level is taken on rank 0 and broadcast. Rank 0 belongs
# to every level that holds a tile (the subsets are prefixes of the rank
# list), so it holds every parent's layout, while a rank outside a parent's
# subset holds none; the broadcast gives every rank the same wanted set, owner
# ranges and creation record, which is what keeps the communicator splits and
# frees collective.
#
# Before the first level is tagged the whole hierarchy is restricted, finest
# first, so that a tile about to leave has written its state onto its parent;
# this is the one-level rule's last restriction of a departing tile, taken
# while every old communicator is still live. A surviving tile whose owner
# range no longer fits its parent's shrunken subset moves: its state is
# migrated block to block over the run's communicator (`_migrate_tile!`),
# which every rank enters since its old owners may lie outside the parent's
# new subset. A survivor also moves when the level is partitioned afresh by
# volume (`_balanced_owners`), each level on its own, since under
# subcycling the levels advance in turn. The measured rebalance is not
# offered here (the constructor refuses it): one busy time per rank cannot
# separate the levels' shares of it.

# The tag buffer of a level that may have children, per dimension: a child
# cell reaches `buffer + tile` of its own parent's nodes past a tagged node
# and needs `margin` more of them, plus one eroded plane, inside that
# parent's own nodes, which is a third as many nodes of this level. With the
# tag at the same place on both levels, this buffer lets the child nest. The
# configured edge bounds every lattice the child may take.
_nest_buffer(buffer::Int, tile::NTuple{3,Int}, margin::Int) =
    ntuple(d -> max(buffer, cld(buffer + tile[d] + margin + 1, 3) + 1), 3)

# The lattice cell of a tile region: its offset over the edge, which also
# holds for a tile clipped at the domain's margin and for the first cell
# extended to a fold (`_lattice_tile`).
_cell_of(r::BlockRegion, tile::NTuple{3,Int}, active::NTuple{3,Bool}) =
    ntuple(d -> active[d] ? max(r.offset[d], 0) ÷ tile[d] : 0, 3)

# The cells of a lattice of edges `tile` meeting a region: those of
# `_tile_span` over its node interval along each active dimension.
function _cells_of(r::BlockRegion, tile::NTuple{3,Int}, active::NTuple{3,Bool})
    spans = ntuple(d -> active[d] ?
                        _tile_span(r.offset[d] + 1, r.offset[d] + r.extent[d], tile[d]) :
                        (0:0), 3)
    return [(k1, k2, k3) for k3 in spans[3] for k2 in spans[2] for k1 in spans[1]]
end

# Fill every refined tile by interpolation of its parent, top-down: the
# initial layout of the AMR frontend, whose seed tile holds no state yet.
# Collective over each parent level's communicator.
function _fill_levels_from_parents!(solver::Solver, states)
    levels = getfield(solver, :levels)
    for ℓ in 2:length(levels)
        levels[ℓ - 1].level_comm.owned || continue
        _fill_tiles_from_parent!(solver, states, levels[ℓ])
    end
    return states
end

# The old layout a level's new tiles take their solution from under a change
# of lattice (`_carry_tiles!`): the old tiles' fine regions, their own nodes
# and their owner ranges.
struct LatticeCarry
    fine::Vector{BlockRegion}
    own::Vector{BlockRegion}
    owners::Vector{UnitRange{Int}}
end

# What rank 0 decides for one refined level at a regrid check or a restart,
# broadcast to every rank: whether the level changes, its lattice edges, the
# wanted tiles and their owner ranges, the level's new and old rank counts,
# each tile's old index (0 for a new region) and whether it stays in place,
# the creation record, and the carry of a change of lattice. Every outcome
# has this one type, so `_swap_level!`, which is specialized on it, compiles
# once per run; a decision whose type followed the outcome compiled it again
# at the first change of lattice.
struct LevelDecision
    changed::Bool
    edges::NTuple{3,Int}
    wanted::Vector{BlockRegion}
    owners::Vector{UnitRange{Int}}
    np_new::Int
    np_old::Int
    old_index::Vector{Int}
    kept::Vector{Bool}
    created::Dict{BlockRegion,Int}
    carry::Union{Nothing,LatticeCarry}
end

# A decision that leaves the level as it is, on the lattice of edges `edges`.
LevelDecision(edges::NTuple{3,Int}) =
    LevelDecision(false, edges, BlockRegion[], UnitRange{Int}[], 0, 0, Int[], Bool[],
                  Dict{BlockRegion,Int}(), nothing)

# The decision of rank 0 for level ℓ from the reduced cell flags of each
# lattice of `chain`: the lattice, the wanted tiles in lattice order, the
# owner ranges, and what survives. Under a change of lattice the decision
# also carries the old tiles' fine regions, own nodes and owners, which
# every rank needs for the carry and a rank outside the parent's old subset
# does not hold.
function _decide_level(solver::Solver, ℓ::Int, spec::RegridSpec,
                       flagsets::Vector{Dict{NTuple{3,Int},Int8}},
                       chain::Vector{NTuple{3,Int}}, active::NTuple{3,Bool},
                       N::NTuple{3,Int}, parent_changed::Bool)
    levels = getfield(solver, :levels)
    lev = levels[ℓ + 1]
    parent = levels[ℓ]
    margin = spec.margin
    old_regions = [lt.region for lt in lev.transfers]
    old_owners = lev.owners
    created = copy(_created(spec, ℓ))
    parent_valid = ℓ == 1 ? [BlockRegion((0, 0, 0), solver.n_global)] :
                   [_erode(_fine_region(lt), lt.imposed, active)
                    for lt in parent.transfers]
    parent_np = parent.level_comm.size
    reach = _placed_reach(solver, spec, N, active)
    span = _level_span(solver.n_global, active, ℓ - 1, getfield(solver, :patches)[1].bcs)
    wrap = _placement_period(solver, spec, ℓ - 1)
    period = _level_period(solver, ℓ - 1)
    lo, hi = _wrap_feasible(_feasible_nodes(N, active, margin, reach, span)..., wrap)
    current = findfirst(==(_level_edges(spec, ℓ)), chain)
    wanted_sets = Vector{BlockRegion}[]
    parent_bins = _RegionIndex(parent_valid, period)
    for (i, a) in enumerate(chain)
        wanted = BlockRegion[]
        push!(wanted_sets, wanted)
        isempty(parent_valid) && continue
        old = _OldTiles(old_regions, created, spec, i == current, active, period)
        flags = flagsets[i]
        cells = Set(keys(flags))
        foreach(r -> push!(cells, _cell_of(r, a, active)), old_regions)
        i == current || foreach(r -> union!(cells, _cells_of(r, a, active)), old_regions)
        for k in sort!(collect(cells); by=k -> (k[3], k[2], k[1]))
            # The lattice's last node: across a seam the period's P + 1.
            t = _lattice_tile(k, active, a, lo, hi,
                              ntuple(d -> wrap[d] > 0 ? wrap[d] + 1 : N[d], 3))
            (t === nothing || !_limiter_tile_ok(solver, t, ℓ)) && continue
            _tile_wanted(t, get(flags, k, zero(Int8)), old) || continue
            box = _buffered(t, active, margin, _boundary_faces(t, span, reach))
            _covered_by(box, parent_valid[_candidates(parent_bins, box)], period) || continue
            push!(wanted, t)
        end
    end
    pick = _choose_lattice(solver, spec, ℓ, chain, wanted_sets, current, parent_np,
                           isempty(old_regions))
    wanted = wanted_sets[pick]
    rechopped = pick != current && !isempty(old_regions)
    changed = (wanted != old_regions || parent_changed) &&
              !(isempty(wanted) && isempty(old_regions))
    changed || return LevelDecision(chain[pick])
    old_set, wanted_set = Set(old_regions), Set(wanted)
    old_bins = _RegionIndex(old_regions, period)
    inherited = Dict(r => _inherited_check(r, view(old_regions, _candidates(old_bins, r)),
                                           created, spec.checks, active, period)
                     for r in wanted if rechopped && !(r in old_set))
    for r in old_regions
        r in wanted_set || delete!(created, r)
    end
    for r in wanted
        haskey(created, r) || (created[r] = get(inherited, r, spec.checks))
    end
    # A survivor keeps its range where the parent's subset still holds it.
    fit = [last(o) < parent_np for o in old_owners]
    owners, np_new = _place_tiles(wanted, active, parent_np, old_regions[fit],
                                  old_owners[fit])
    survived = Set(old_regions[fit])
    owners, np_new = _balanced_owners(wanted, active, parent_np, owners, np_new,
                                      [r in survived for r in wanted];
                                      nodes=spec.costs.nodes)
    old_position = Dict(r => i for (i, r) in enumerate(old_regions))
    old_index = [get(old_position, r, 0) for r in wanted]
    kept = [old_index[ti] > 0 && old_owners[old_index[ti]] == owners[ti]
            for ti in eachindex(wanted)]
    carry = rechopped ?
            LatticeCarry([_fine_region(lt) for lt in lev.transfers],
                         [_erode(_fine_region(lt), lt.imposed, active)
                          for lt in lev.transfers],
                         copy(old_owners)) : nothing
    return LevelDecision(true, chain[pick], wanted, owners, np_new, lev.level_comm.size,
                         old_index, kept, created, carry)
end

# Tag level ℓ − 1 and decide level ℓ. The sweep runs on the tiles each rank
# holds (the root for ℓ = 1); the flagged lattice cells gather on rank 0,
# which decides, and the decision is broadcast. Collective over the run's
# communicator, whatever the rank holds.
function _level_decision(solver::Solver, states, ℓ::Int, spec::RegridSpec,
                         parent_changed::Bool)
    levels = getfield(solver, :levels)
    patches = getfield(solver, :patches)
    comm = getfield(solver, :comm)
    n_global = solver.n_global
    active = ntuple(d -> n_global[d] > 1, 3)
    b = ℓ < length(levels) - 1 ? _nest_buffer(spec.buffer, spec.tile, spec.margin) :
        ntuple(_ -> spec.buffer, 3)
    N = ntuple(d -> active[d] ? 3^(ℓ - 1) * (n_global[d] - 1) + 1 : 1, 3)
    # The lattices the level may take. The chain reads the rank count only
    # for whether it exceeds one, so the run's count gives every rank the
    # same chain; the parent subset's count, which the choice among them
    # reads, is rank 0's to apply.
    chain = _level_chain(solver, spec, N, active, MPI.Comm_size(comm))
    # Across a periodic seam the lattice runs over the period's P + 1 nodes.
    wrap = _placement_period(solver, spec, ℓ - 1)
    Ks = [ntuple(d -> active[d] ? ((wrap[d] > 0 ? wrap[d] + 1 : N[d]) - 1) ÷ a[d] + 1 : 1,
                 3) for a in chain]
    cellsets = [Dict{NTuple{3,Int},Int8}() for _ in chain]
    # A node at the tag level outside the feasible interval of the lattice
    # clip `_decide_level` applies: the margin band of a face no level reaches.
    reach = _placed_reach(solver, spec, N, active)
    lo, hi = _wrap_feasible(_feasible_nodes(N, active, spec.margin, reach,
                                            _level_span(n_global, active, ℓ - 1,
                                                        patches[1].bcs))..., wrap)
    banded = Ref(false)
    function mark!(g1, g2, g3, level)
        g = (g1, g2, g3)
        level == TAG_MARK && any(d -> active[d] && !(lo[d] <= g[d] <= hi[d]), 1:3) &&
            (banded[] = true)
        for (a, K, cells) in zip(chain, Ks, cellsets)
            spans = ntuple(d -> active[d] ?
                                _wrapped_cells(g[d] - b[d], g[d] + b[d], a[d], K[d],
                                               wrap[d]) : (0:0), 3)
            for k3 in spans[3], k2 in spans[2], k1 in spans[1]
                k = (k1, k2, k3)
                cells[k] = max(get(cells, k, zero(Int8)), level)
            end
        end
        return nothing
    end
    if ℓ == 1
        _tag_sweep!(mark!, solver, states[1])
    else
        for li in levels[ℓ].patches
            p = patches[li]
            _tag_sweep!(mark!, solver, p, states[li], zeros(Int8, size(p.amr.covered)),
                        false)
        end
    end
    # A record is the lattice's index in the chain, the cell and its level;
    # the band flag travels as a record of its own, lattice 0.
    packed = Int64[]
    for (i, cells) in enumerate(cellsets), (k, v) in cells
        append!(packed, (i, k[1], k[2], k[3], v))
    end
    banded[] && append!(packed, (0, 0, 0, 0, 1))
    gathered = _gather_records(packed, comm)
    decision = nothing
    if MPI.Comm_rank(comm) == 0
        flagsets = [Dict{NTuple{3,Int},Int8}() for _ in chain]
        band = false
        for buf in gathered, i in 1:5:length(buf)
            buf[i] == 0 && (band = true; continue)
            flags = flagsets[buf[i]]
            k = (Int(buf[i + 1]), Int(buf[i + 2]), Int(buf[i + 3]))
            flags[k] = max(get(flags, k, zero(Int8)), Int8(buf[i + 4]))
        end
        band && _warn_margin_band(solver, spec.margin, ℓ)
        decision = _decide_level(solver, ℓ, spec, flagsets, chain, active, N,
                                 parent_changed)
    end
    decision = MPI.bcast(decision, comm; root=0)::LevelDecision
    _set_level_edges!(spec, ℓ, decision.edges)
    if decision.changed
        created = _created(spec, ℓ)
        empty!(created)
        merge!(created, decision.created)
    end
    return decision
end

# One tile's state replicated on every rank of `comm`, from the blocks of it
# the ranks hold (`piece` is this rank's `(Q, decomp)` or `nothing`), with the
# node each of its `folded` faces adds. Collective over `comm`.
function _gather_replicated(::Type{T}, region::BlockRegion, active::NTuple{3,Bool},
                            n_cons::Int, piece, comm::MPI.Comm,
                            folded::NTuple{3,NTuple{2,Bool}}=_NO_BOUNDARY) where {T}
    Nf = fine_extent(region, active, folded)
    dst = zeros(T, Nf..., n_cons)
    Q, dp = piece === nothing ? (nothing, nothing) : piece
    gather_region!(dst, ntuple(d -> 1:Nf[d], 3), (0, 0, 0), (0, 0, 0), Q, comm,
                   _owned_blocks(dp, comm),
                   dp === nothing ? (0, 0, 0) : dp.offset,
                   dp === nothing ? (0, 0, 0) : dp.n_halo_d)
    return dst
end

# A level whose patches moved by `delta` places in the patch vector. Its
# transfers still name the old indices; the caller rebuilds the level before
# anything reads them.
_shifted_level(lev::Level{T}, delta::Int) where {T} = delta == 0 ? lev :
    Level{T}(lev.index, lev.level_comm, lev.owners, lev.group, lev.tiles,
             lev.patches .+ delta, lev.transfers, lev.ghost_sends, lev.ghost_recvs,
             lev.plane_pairs, lev.phases, lev.stacks, lev.coupling)

"""
Internal: rebuild refined level `ℓ` of a hierarchy to the tiles of the
`LevelDecision` `decision` (`wanted`, `owners`, `np_new`, `np_old`,
`old_index`, `kept`, `carry`, the same on every rank), after level ℓ − 1 has
taken its own new layout.

A kept tile (region and owner range unchanged) keeps its arrays and state
under a new id, faces and transfer; every other tile is built fresh. With
`init`, a fresh tile is filled by interpolation of its parent, a moved
survivor then takes its whole old state back by migration, and
the planes fresh tiles share with survivors are seeded from the survivors; a
restart passes `init = false` and reads the state afterwards. The patch
vector, `states` and, when given, the workspace's stage arrays are spliced
at the level's place, and the deeper levels' patch indices are shifted with
them (those levels are rebuilt next). `resplit` says whether level ℓ − 1's
communicator was split afresh in this pass; the return value says the same
of level ℓ's. Collective over the run's communicator.
"""
function _swap_level!(solver::Solver{T}, states::Vector{<:ConservedState},
                      workspace::Union{Nothing,Workspace}, ℓ::Int,
                      decision::LevelDecision, resplit::Bool, init::Bool) where {T}
    spec = getfield(solver, :regrid)
    schemes = getfield(solver, :schemes)
    levels = getfield(solver, :levels)
    patches = getfield(solver, :patches)
    comm = getfield(solver, :comm)
    lev = levels[ℓ + 1]
    parent = levels[ℓ]
    wanted = decision.wanted
    owners = decision.owners
    old_index = decision.old_index
    kept = decision.kept
    n_global = solver.n_global
    active = ntuple(d -> n_global[d] > 1, 3)
    n_cons = solver.equations.n_cons
    base = 1 + sum((length(levels[k].patches) for k in 2:ℓ); init=0)
    nold = length(lev.patches)
    lev.patches == collect((base + 1):(base + nold)) ||
        error("regrid: level $ℓ's patches are not the contiguous run after its parents'")
    old_local = Dict(zip(lev.tiles, lev.patches))
    # The moved survivors: this rank's old block of each, state and
    # decomposition, held past the swap for the migration's sends, and the
    # tile's old block table over the run's communicator, taken while the old
    # decompositions are live. The list is the broadcast decision's, so every
    # rank enters the same table gathers.
    moved = Dict{Int,Tuple{Any,Any,Vector{OwnedBlock}}}()
    boundaries = _region_boundaries(solver, wanted, ℓ)
    folds = _region_folds(solver, boundaries)
    for ti in eachindex(wanted)
        (old_index[ti] == 0 || kept[ti]) && continue
        li = get(old_local, old_index[ti], 0)
        Q_old, d_old = li == 0 ? (nothing, nothing) : (states[li], patches[li].decomp)
        moved[ti] = (Q_old, d_old, _block_table(d_old, comm))
    end
    kept_old = Set(old_index[ti] for ti in eachindex(wanted) if kept[ti])
    dropped = Decomp{T}[patches[li].decomp for (li, t) in zip(lev.patches, lev.tiles)
                        if !(t in kept_old)]
    # Under a change of lattice every old tile this rank holds is a source
    # of the carry, by old tile index.
    carry = decision.carry
    carry_src = Dict{Int,Tuple{eltype(states),Decomp{T}}}()
    if carry !== nothing
        for (li, t) in zip(lev.patches, lev.tiles)
            carry_src[t] = (states[li], patches[li].decomp)
        end
    end
    # The old group and, where the subset changes, the old level communicator
    # are freed by their members; the new ones are split over the parent's
    # new subset. `resized` is derived from broadcast values, so it agrees on
    # every rank.
    parent_lc = parent.level_comm
    old_lc = lev.level_comm
    free_tile_group!(lev.group)
    resized = resplit || decision.np_new != decision.np_old
    resized && free_level_comm!(old_lc)
    new_lc = !parent_lc.owned ? absent_level_comm() :
             resized ? split_level_comm(parent_lc, decision.np_new) : old_lc
    group = new_lc.owned ? split_tile_comm(new_lc, owners) : absent_tile_group()
    held = [ti for ti in eachindex(wanted) if owners[ti] == group.ranks]
    period = _level_period(solver, ℓ - 1)
    faces = _tile_faces(wanted, period)
    rbcs = patches[1].bcs
    ws_pool = [p.rhs_workspace for p in patches]
    ph = patches[1].h
    for _ in 2:ℓ
        ph = ntuple(d -> active[d] ? ph[d] / 3 : ph[d], 3)
    end
    new_patches = Patch[]
    new_states = similar(states, 0)
    new_dQ = similar(states, 0)
    new_du = similar(states, 0)
    local_of = zeros(Int, length(wanted))
    for (k, ti) in enumerate(held)
        idx = base + k
        local_of[ti] = idx
        if kept[ti]
            oi = old_local[old_index[ti]]
            fbcs = _fine_bcs(active, faces[ti], boundaries[ti], rbcs)
            push!(new_patches, _repatch(patches[oi], idx, faces[ti], fbcs, group.comm,
                                        _survivor_plans(patches[oi], faces[ti], fbcs,
                                                        boundaries[ti], rbcs,
                                                        patches[1].folds, spec,
                                                        schemes, solver.art.smoother,
                                                        _ring_detector(solver))))
            push!(new_states, states[oi])
            if workspace !== nothing
                push!(new_dQ, workspace.dQ[oi])
                push!(new_du, workspace.du[oi])
            end
        else
            p = _build_fine_patch(T, wanted[ti], active, ph, spec.n_halo, group.comm,
                                  schemes.deriv, schemes.filt, spec.smoo,
                                  solver.art.smoother, schemes.interface_rhs,
                                  spec.backend, ws_pool, solver.equations.n_species,
                                  n_cons, _grad_Q_columns(solver), idx, ℓ,
                                  faces[ti];
                                  interface_divergence=schemes.interface_divergence,
                                  gradient_ring=_gradient_ring_built(solver),
                                  ghost_fluxes=_ghost_fluxes_built(solver),
                                  ring=_ring_detector(solver),
                                  boundary=boundaries[ti], bcs=rbcs,
                                  root_folds=patches[1].folds,
                                  n_sensed=_sensed_field_count(solver, spec.tile),
                                  slot=k)
            Q = _state_like(p.rho, n_cons)
            push!(new_patches, p)
            push!(new_states, Q)
            if workspace !== nothing
                push!(new_dQ, zero(Q))
                push!(new_du, zero(Q))
            end
        end
    end
    # The transfers, on every rank of the parent's subset, against the
    # parent's current tiles (whose patch indices precede this level's and
    # are untouched by the splice below).
    transfers = LevelTransfer{T}[]
    coupling = LevelCoupling{T}()
    if parent_lc.owned
        pregions = ℓ == 1 ? [patches[1].region] :
                   [_fine_region(lt) for lt in parent.transfers]
        plocal = ℓ == 1 ? [1] : [lt.fine_index for lt in parent.transfers]
        pdecomp(li) = li == 0 ? nothing : patches[li].decomp
        fdecomp(ti) = local_of[ti] == 0 ? nothing : new_patches[local_of[ti] - base].decomp
        for (ti, tr) in enumerate(wanted)
            pids = _parents_of(tr, active, collect(eachindex(pregions)), pregions,
                               period)
            push!(transfers, build_level_transfer(
                T, tr, active, spec.n_halo, pregions[pids], plocal[pids], local_of[ti],
                schemes.level_restriction, n_cons, getfield(solver, :subcycle), fdecomp(ti),
                parent_lc.comm, length(owners[ti]), faces[ti];
                interpolation_order=schemes.level_interpolation_order,
                gradient_deriv=_gradient_ring_built(solver) ? schemes.deriv : nothing,
                parent_h=ph, boundary=boundaries[ti], folded=folds[ti], period))
        end
        pdecomps = Union{Nothing,Decomp{T}}[pdecomp(li) for li in plocal]
        fdecomps = Union{Nothing,Decomp{T}}[fdecomp(ti) for ti in eachindex(wanted)]
        coupling = build_level_coupling(T, parent_lc.comm, transfers, pregions, pdecomps,
                                        fdecomps)
    end
    rng = (base + 1):(base + nold)
    splice!(patches, rng, new_patches)
    splice!(states, rng, new_states)
    if workspace !== nothing
        splice!(workspace.dQ, rng, new_dQ)
        splice!(workspace.du, rng, new_du)
    end
    delta = length(new_patches) - nold
    for k in (ℓ + 2):length(levels)
        levels[k] = _shifted_level(levels[k], delta)
    end
    indices = [base + k for k in eachindex(held)]
    if new_lc.owned
        records = _level_records(T, new_lc.comm,
                                 [_fine_region(tr, active, fo)
                                  for (tr, fo) in zip(wanted, folds)], held,
                                 indices, Decomp{T}[p.decomp for p in new_patches],
                                 n_cons, _level_period(solver, ℓ))
        levels[ℓ + 1] = Level{T}(ℓ, new_lc, owners, group, held, indices, transfers,
                                 records; coupling)
    elseif parent_lc.owned
        levels[ℓ + 1] = Level{T}(ℓ, new_lc, owners, group, held, indices, transfers;
                                 coupling)
    else
        levels[ℓ + 1] = Level{T}(ℓ, absent_level_comm(), UnitRange{Int}[],
                                 absent_tile_group(), Int[], Int[],
                                 LevelTransfer{T}[])
    end
    for li in parent.patches
        _fill_covered!(patches[li], wanted, period)
    end
    fresh = [ti for ti in eachindex(wanted) if !kept[ti]]
    for ti in fresh
        local_of[ti] == 0 || init_geometry!(PatchSolver(solver, patches[local_of[ti]]))
    end
    if init && parent_lc.owned
        is_fresh = falses(length(wanted))
        is_fresh[fresh] .= true
        isempty(fresh) || _fill_tiles_from_parent!(solver, states, levels[ℓ + 1], is_fresh)
    end
    # A moved survivor takes its whole state back over the interpolated
    # initialization, block to block from its old owners. Every rank enters,
    # an old owner outside the parent's new subset included, since it holds
    # blocks to send; the block tables name ranks of the run's communicator.
    if init
        for ti in sort!(collect(keys(moved)))
            Q_old, d_old, old_table = moved[ti]
            li = local_of[ti]
            d_new = li == 0 ? nothing : patches[li].decomp
            _migrate_tile!(T, li == 0 ? nothing : states[li], d_new, Q_old, d_old,
                           fine_extent(wanted[ti], active, folds[ti]), active,
                           old_table, _block_table(d_new, comm), comm, _EVERY_FACE;
                           ghosts=true)
        end
    end
    # Under a change of lattice a new tile takes the evolved solution of the
    # old tiles it overlaps, over the run's communicator, whose rank numbers
    # the owner ranges of both layouts use.
    if init && carry !== nothing
        new_data = _held_tile_data(T, states, patches, local_of)
        _carry_tiles!(T, comm, active, _level_period(solver, ℓ), carry.fine, carry.own,
                      carry.owners,
                      [_fine_region(tr, active, fo) for (tr, fo) in zip(wanted, folds)],
                      owners, [old_index[ti] == 0 for ti in eachindex(wanted)],
                      carry_src, new_data)
    end
    if init && parent_lc.owned
        # A moved survivor carries its planes and seeds as a survivor.
        is_new = [old_index[ti] == 0 for ti in eachindex(wanted)]
        any(is_new) && new_lc.owned &&
            _seed_planes!(solver, states, levels[ℓ + 1], is_new)
    end
    # The old transfers' chains are on COMM_SELF; the dropped tiles'
    # decompositions are freed by their old owners in the order they held
    # them, the migration's reads being done.
    for lt in lev.transfers
        free_transfer_decomps!(lt)
    end
    for decomp in dropped
        free_communicators!(decomp)
    end
    return resized
end

"""
Internal: the regrid of a hierarchy of more than one refined level (see the
section comment above). Restricts the hierarchy, then tags, decides and,
where anything changed, rebuilds each level top-down. `bootstrap` skips the
restriction, for the AMR frontend's initial layout, whose refined state is
not yet the solution. Returns whether any level changed; the value is the
same on every rank.
"""
function _regrid_hierarchy!(solver::Solver, states::Vector{<:ConservedState},
                            workspace::Union{Nothing,Workspace}, save;
                            bootstrap::Bool=false)
    spec = getfield(solver, :regrid)
    levels = getfield(solver, :levels)
    _harvest_lattice_costs!(solver, spec)
    bootstrap || restrict_level!(solver, states)
    changed = false
    parent_changed = false
    resplit = false
    for ℓ in 1:length(levels)-1
        decision = _level_decision(solver, states, ℓ, spec, parent_changed)
        if decision.changed
            resplit = _swap_level!(solver, states, workspace, ℓ, decision, resplit,
                                   true)
            parent_changed = true
            changed = true
        else
            parent_changed = false
            resplit = false
        end
    end
    return changed
end

# The `run!` hook: cadence check, then `regrid!`. The single-array state form
# never regrids; a multi-patch solver without a RegridSpec returns at the
# first test. Returns whether `regrid!` ran, which is the same on every rank;
# a check that changes no level can still write the state (the hierarchy
# regrid's restriction, the coefficient priming's boundary conditions).
#
# Under a positive `validity_interval` the state the check leaves is swept
# before the savepoint is rebuilt from it. A new tile's state is interpolated
# from its parent and no step has produced it, so it is a state entering a
# step whatever the cadence says. A rejection raises rather than rolling
# back: the savepoint predates the layout and cannot be restored onto it.
_maybe_regrid!(solver::Solver, Q, workspace::Workspace, save,
               control=solver.control, floors=(0.0, 0.0)) = false
function _maybe_regrid!(solver::Solver, states::Vector{<:ConservedState},
                        workspace::Workspace, save,
                        control::StepControl=solver.control,
                        floors::Tuple{Float64,Float64}=(0.0, 0.0))
    spec = getfield(solver, :regrid)
    spec === nothing && return false
    solver.step - spec.last_step >= spec.interval || return false
    spec.last_step = solver.step
    spec.checks += 1
    # The regrid's own work is excluded from the busy time the next check
    # measures: its gathers and frees are charged to the waiting account as
    # they run, and the rest is charged here.
    t0 = time_ns()
    spec.costs.timing = _lattice_measured(solver, spec)
    compiled0 = _counted_compile_ns(spec.costs)
    wait0 = solver.wall_wait
    # `regrid!` with the savepoint refresh after the sweep.
    _tag_coefficients!(solver, states)
    changed = _regrid_impl!(solver, states, workspace, save)
    _regrid_prime!(solver, states, workspace, nothing, changed)
    control.validity_interval > 0 &&
        validate_state!(solver, states; control=control,
                        stage="the state after a regrid check", floors=floors)
    changed && _rebank!(solver, states, save)
    spec.wall_regrid += (time_ns() - t0) / 1e9 - (solver.wall_wait - wait0)
    _close_lattice_check!(solver, spec, (time_ns() - t0) / 1e9,
                          (_compile_ns() - compiled0) / 1e9)
    return true
end
