# Patch abstraction: per-patch state split out of the solver configuration
# (Stage 2 of reference/AMR_GPU.md, single level).
#
# A Patch is a logically rectangular block of uniform resolution carrying its
# own communicator, decomposition, operator plans, folds, and every field
# array, typed A <: AbstractArray{T,3} against the storage backend (GPU Stage
# G0). The Solver in rhs.jl keeps the physics configuration and the run clock,
# plus the vector of this rank's patches in global order. In the common case of
# one patch spanning all ranks, that patch's decomposition is built over
# MPI.COMM_WORLD exactly as the pre-patch solver's was, and `Base.getproperty`
# on Solver forwards the patch-owned names to the sole patch, so the whole
# existing surface (`solver.rho`, `solver.decomp`, ...) reads unchanged and the
# single-patch code path is the old code path.
#
# With several patches, the rank set is partitioned: MPI.Comm_split assigns
# each rank to exactly one patch's communicator, ranks in proportion to patch
# volume, except in a serial run where the one rank advances every patch in
# sequence. Patch interfaces are node-coincident — abutting patches share the
# interface plane — and are coupled by three mechanisms, all between stages and
# never inside a per-patch RHS evaluation (a shared rank advances its patches
# sequentially, so cross-patch communication inside compute_rhs! would wait on
# work that has not run yet):
#
#   1. `exchange_patch_ghosts!` copies each patch's interface ghost layers from
#      the abutting patch's interior. The ghosts carry conserved state; every
#      derived quantity in them (primitives, internal energy, mass fractions)
#      is recomputed locally by the full-padded-array passes that already
#      exist.
#   2. The line solves are closed at the face with rows whose left-hand side
#      couples no ghost unknown (`interface_closures` in kernels.jl); whether
#      their right-hand sides read the exchanged ghosts (`:extended`) or stay
#      one-sided (`:onesided`) is the Solver's `interface_rhs` setting. The
#      flux divergence always keeps the scheme's own one-sided closures: ghost
#      FLUXES would require a second communication phase inside the RHS, which
#      the sequential-patch constraint above forbids.
#   3. `average_shared_planes!` replaces both patches' copies of each shared
#      interface-plane node by their mean after every RK stage, removing the
#      drift mode between the two solves.
#
# The current layout generator tiles the domain into slabs along a single
# dimension (`patch_grid` with one entry > 1). That keeps every interface's
# transverse extent equal to the full domain width, so diagonal patch
# adjacency does not arise; general patch grids extend the same record
# machinery with corner exchanges and are future work, as is patching across a
# coordinate fold (constraint 4 of the plan; `Solver` rejects both).

"""
    Patch

Per-patch state of a [`Solver`](@ref): the patch's place in the global grid
(`region`, in global node index space), its communicator and [`Decomp`](@ref),
its operator plans and folds, and every field array, typed
`A <: AbstractArray{T,3}` by the storage backend. Constructed by `Solver`;
user code normally reaches these fields through the solver (which forwards
them for a single-patch run) rather than holding a `Patch` directly.
"""
struct Patch{T,A<:AbstractArray{T,3},Fo,BC,DP,VP,FP,SP,RP}
    id::Int
    level::Int
    region::BlockRegion                     # offset + extent, in this LEVEL's node
                                            # index space (level ℓ spacing h/3^ℓ,
                                            # so physical coordinates follow from
                                            # region.offset and h alone)
    comm::MPI.Comm                          # ranks owning pieces of this patch
    decomp::Decomp{T}                       # decomposition of the patch over comm
    h::NTuple{3,T}                          # this level's grid spacing (h/3 per
                                            # refinement level below the root)
    faces::NTuple{3,NTuple{2,Int}}          # 0 = physical/periodic; else neighbor patch id
    bcs::BC                                 # per-face conditions (InterfaceBC at interfaces)
    folds::Fo
    deriv_plans::DP
    div_plans::VP                           # divergence plans: deriv_plans unless an
                                            # interface end forces one-sided closures
    filter_plans::FP
    smooth_plans::SP
    ring_plans::RP
    pairbuf::A
    pairout::A
    # primitives (full padded arrays)
    rho::A; u::A; v::A; w::A
    p::A; T_ion::A; c::A; cp_mix::A
    Y::Vector{A}
    # gradients
    grad_u::Matrix{A}
    grad_T_ion::NTuple{3,A}
    grad_Y::Matrix{A}
    # artificial properties and scratch
    mu_art::A; beta_art::A; kappa_art::A
    D_art::Vector{A}
    strain_mag::A; sensor::A; sensor_sp::A
    tmp_a::A; tmp_b::A
    ring_buf::A
    # geometry
    inv_J::A
    area_d::NTuple{3,A}
    inv_h::NTuple{3,A}
    inv_r::A
    cot_over_r::A
    # fluxes flux[d, c]
    flux::Matrix{A}
end

# Property names owned by the patch rather than the solver configuration.
# `Base.getproperty(::Solver, name)` forwards these to the sole patch of a
# single-patch solver, and `PatchSolver` routes them per patch. The test is a
# `===` chain so that a literal property name constant-folds to a plain
# `getfield` at every call site.
@inline _is_patch_prop(n::Symbol) =
    n === :decomp || n === :bcs || n === :folds || n === :faces ||
    n === :region || n === :h || n === :deriv_plans || n === :div_plans ||
    n === :filter_plans || n === :smooth_plans || n === :ring_plans ||
    n === :pairbuf || n === :pairout ||
    n === :rho || n === :u || n === :v || n === :w ||
    n === :p || n === :T_ion || n === :c || n === :cp_mix || n === :Y ||
    n === :grad_u || n === :grad_T_ion || n === :grad_Y ||
    n === :mu_art || n === :beta_art || n === :kappa_art || n === :D_art ||
    n === :strain_mag || n === :sensor || n === :sensor_sp ||
    n === :tmp_a || n === :tmp_b || n === :ring_buf ||
    n === :inv_J || n === :area_d || n === :inv_h || n === :inv_r ||
    n === :cot_over_r || n === :flux

"""
    PatchSolver(solver, patch)

Access adapter binding one [`Patch`](@ref) to its solver configuration.
Property reads forward patch-owned names to the patch and everything else to
the solver, so every routine written against the solver's property surface
(`s.rho`, `s.decomp`, `s.equations`, ...) runs unchanged per patch. The
multi-patch drivers in timestep.jl construct these; a single-patch `Solver`
plays the role itself through its own property forwarding.
"""
struct PatchSolver{T,S,P<:Patch{T}}
    solver::S
    patch::P
    PatchSolver(solver, patch::Patch{T}) where {T} =
        new{T,typeof(solver),typeof(patch)}(solver, patch)
end

@inline function Base.getproperty(ps::PatchSolver, name::Symbol)
    _is_patch_prop(name) && return getfield(getfield(ps, :patch), name)
    name === :solver && return getfield(ps, :solver)
    name === :patch && return getfield(ps, :patch)
    return getproperty(getfield(ps, :solver), name)
end

# Solver clock updates written through a PatchSolver land on the solver.
@inline Base.setproperty!(ps::PatchSolver, name::Symbol, value) =
    setproperty!(getfield(ps, :solver), name, value)

# --- Patch layout -----------------------------------------------------------

"""
    patch_slabs(n_global, periodic, patch_grid) -> Vector{BlockRegion}

Tile the global node lattice into `prod(patch_grid)` slab patches. Exactly one
entry of `patch_grid` may exceed one (see the header). Abutting patches share
their interface plane node; along a non-periodic split dimension of N nodes,
P patches therefore carry N + P − 1 nodes in total, and along a periodic one
N + P, the last patch's high plane being the first patch's low plane. Regions
are returned in patch id order with global node offsets.
"""
function patch_slabs(n_global::NTuple{3,Int}, periodic::NTuple{3,Bool},
                     patch_grid::NTuple{3,Int})
    all(>=(1), patch_grid) || error("patch_grid entries must be positive")
    nsplit = count(>(1), patch_grid)
    nsplit <= 1 || error("patch_grid may split one dimension only " *
                         "(slab layout); got $patch_grid")
    npatch = prod(patch_grid)
    npatch <= 30 || error("patch_grid gives $npatch patches; the interface " *
                          "message tagging supports at most 30")
    npatch == 1 && return [BlockRegion((0, 0, 0), n_global)]
    ds = findfirst(>(1), patch_grid)
    P = patch_grid[ds]
    N = n_global[ds]
    # Interval boundaries in global node index space: patch j spans nodes
    # b[j] .. b[j+1] inclusive (the shared planes). Periodic wraps the last
    # boundary onto node N + 1 ≡ 1.
    span = periodic[ds] ? N : N - 1
    bounds = [1 + round(Int, span * (j / P)) for j in 0:P]
    length(unique(bounds)) == P + 1 ||
        error("n_global[$ds] = $N is too small for $P patches")
    regions = BlockRegion[]
    for j in 1:P
        lo, hi = bounds[j], bounds[j+1]
        offset = ntuple(d -> d == ds ? lo - 1 : 0, 3)
        extent = ntuple(d -> d == ds ? hi - lo + 1 : n_global[d], 3)
        push!(regions, BlockRegion(offset, extent))
    end
    return regions
end

"""
    patch_rank_counts(regions, np) -> Vector{Int}

Ranks assigned to each patch: proportional to patch volume by largest
remainder, at least one per patch, summing to `np`. Deterministic, so every
rank computes the same partition without communication.
"""
function patch_rank_counts(regions::Vector{BlockRegion}, np::Int)
    npatch = length(regions)
    np >= npatch || error("$(np) ranks cannot partition over $npatch patches; " *
                          "run with at least one rank per patch (or serially)")
    vol = [prod(r.extent) for r in regions]
    total = sum(vol)
    share = [np * v / total for v in vol]
    counts = max.(floor.(Int, share), 1)
    # Largest-remainder distribution of what floor+min left over, negative
    # corrections coming off the largest counts first.
    while sum(counts) < np
        i = argmax(share .- counts)
        counts[i] += 1
    end
    while sum(counts) > np
        order = sortperm(counts; rev=true)
        done = false
        for i in order
            counts[i] > 1 || continue
            counts[i] -= 1
            done = true
            break
        end
        done || error("cannot assign at least one rank per patch")
    end
    return counts
end

# --- Interface exchange records ---------------------------------------------
#
# Built once at setup from a world-Allgathered table of every rank's
# (patch id, block offset, block extent). Every record carries PADDED local
# index ranges into this rank's own patch arrays, plus the partner world rank
# and a deterministic tag both sides compute identically. `partner == myrank`
# marks the serial case, executed as a direct copy between two local patches.

struct GhostRecord{T}
    patch::Int                        # local patch index on this rank
    partner::Int                      # world rank (== my rank for a local copy)
    partner_patch::Int                # local patch index of the partner (local case)
    tag::Int
    mine::NTuple{3,UnitRange{Int}}    # padded ranges in my patch's arrays
    theirs::NTuple{3,UnitRange{Int}}  # padded ranges in the partner patch (local case)
    buf::Vector{T}
end

struct PlaneRecord{T}
    patch::Int
    partner::Int                      # world rank (== my rank for a local pairing)
    partner_patch::Int
    tag::Int                          # tag this side receives on
    sendtag::Int                      # tag the partner receives on
    mine::NTuple{3,UnitRange{Int}}
    theirs::NTuple{3,UnitRange{Int}}  # local case only
    buf::Vector{T}                    # receive buffer
    sbuf::Vector{T}                   # send buffer
end

# tag spaces: ghosts and plane averages must not collide, and a periodic
# two-slab layout pairs the same two patches across both faces, so the tag
# carries (source patch, dest patch, dimension, dest face side).
function _iface_tag(base::Int, npatch::Int, src::Int, dst::Int, ds::Int, side::Int)
    return base + (((src * (npatch + 1) + dst) * 3 + (ds - 1)) * 2 + (side - 1))
end

const _GHOST_TAG_BASE = 2000
const _PLANE_TAG_BASE = 16000

# Intersection of 1-based node ranges, empty allowed.
_isect(a::UnitRange{Int}, b::UnitRange{Int}) =
    max(first(a), first(b)):min(last(a), last(b))

# Padded local range of patch-node range `r` for a rank whose block starts at
# `off` with pad `pad`: node g ↦ padded index g - off + pad.
_padded(r::UnitRange{Int}, off::Int, pad::Int) = (first(r)-off+pad):(last(r)-off+pad)

function _pack!(buf::AbstractVector, Q, r::NTuple{3,UnitRange{Int}})
    idx = 1
    @inbounds for c in 1:size(Q, 4), k in r[3], j in r[2], i in r[1]
        buf[idx] = Q[i, j, k, c]
        idx += 1
    end
    return buf
end

function _unpack!(Q, buf::AbstractVector, r::NTuple{3,UnitRange{Int}})
    idx = 1
    @inbounds for c in 1:size(Q, 4), k in r[3], j in r[2], i in r[1]
        Q[i, j, k, c] = buf[idx]
        idx += 1
    end
    return Q
end

function _average_from!(Q, buf::AbstractVector, r::NTuple{3,UnitRange{Int}})
    idx = 1
    @inbounds for c in 1:size(Q, 4), k in r[3], j in r[2], i in r[1]
        Q[i, j, k, c] = 0.5 * (Q[i, j, k, c] + buf[idx])
        idx += 1
    end
    return Q
end

function _copy_block!(Qdst, rdst::NTuple{3,UnitRange{Int}},
                      Qsrc, rsrc::NTuple{3,UnitRange{Int}})
    @inbounds for c in 1:size(Qdst, 4)
        for (kd, ks) in zip(rdst[3], rsrc[3]), (jd, js) in zip(rdst[2], rsrc[2])
            for (id, is) in zip(rdst[1], rsrc[1])
                Qdst[id, jd, kd, c] = Qsrc[is, js, ks, c]
            end
        end
    end
    return Qdst
end

function _average_blocks!(Qa, ra::NTuple{3,UnitRange{Int}},
                          Qb, rb::NTuple{3,UnitRange{Int}})
    @inbounds for c in 1:size(Qa, 4)
        for (ka, kb) in zip(ra[3], rb[3]), (ja, jb) in zip(ra[2], rb[2])
            for (ia, ib) in zip(ra[1], rb[1])
                m = 0.5 * (Qa[ia, ja, ka, c] + Qb[ib, jb, kb, c])
                Qa[ia, ja, ka, c] = m
                Qb[ib, jb, kb, c] = m
            end
        end
    end
    return Qa
end

# --- Record construction ----------------------------------------------------

# One row per (world rank, patch) pair: who owns which block of which patch.
struct BlockEntry
    rank::Int
    pid::Int
    offset::NTuple{3,Int}     # patch-local node offset of the rank's block
    n_local::NTuple{3,Int}
end

# Every rank's block, known everywhere. With the rank set partitioned over
# patches each rank contributes its one block through a single Allgather; a
# serial run holds every patch already and communicates nothing.
function _block_table(comm::MPI.Comm, my_pids::Vector{Int}, my_decomps)
    np = MPI.Comm_size(comm)
    if np == 1
        return [BlockEntry(0, my_pids[i], my_decomps[i].offset, my_decomps[i].n_local)
                for i in eachindex(my_pids)]
    end
    length(my_pids) == 1 ||
        error("rank partitioning gives one patch per rank; got $(length(my_pids))")
    d = my_decomps[1]
    mine = Int64[my_pids[1], d.offset..., d.n_local...]
    flat = MPI.Allgather(mine, comm)
    table = BlockEntry[]
    for r in 0:np-1
        e = flat[7r+1:7r+7]
        push!(table, BlockEntry(r, Int(e[1]), (Int(e[2]), Int(e[3]), Int(e[4])),
                                (Int(e[5]), Int(e[6]), Int(e[7]))))
    end
    return table
end

"""
    build_interface_records(T, comm, regions, faces_all, my_pids, my_decomps,
                            n_cons) -> (ghost_sends, ghost_recvs, plane_pairs)

Construct the interface exchange records for this rank's patches: ghost sends,
ghost receives, and shared-plane averaging pairs, each carrying padded local
index ranges, the partner world rank, a deterministic tag, and its buffer.
Collective over `comm` (one Allgather of the per-rank blocks); every rank
derives every record it participates in from the same table, so senders and
receivers agree without negotiation.
"""
function build_interface_records(::Type{T}, comm::MPI.Comm,
                                 regions::Vector{BlockRegion}, faces_all,
                                 my_pids::Vector{Int}, my_decomps,
                                 n_cons::Int) where {T}
    npatch = length(regions)
    ghost_sends = GhostRecord{T}[]
    ghost_recvs = GhostRecord{T}[]
    plane_pairs = PlaneRecord{T}[]
    npatch == 1 && return ghost_sends, ghost_recvs, plane_pairs
    table = _block_table(comm, my_pids, my_decomps)
    me = MPI.Comm_rank(comm)
    local_of = Dict(p => i for (i, p) in enumerate(my_pids))

    # Padded ranges of patch-node ranges for a given block.
    padded3(rs, off, pad) = ntuple(d -> _padded(rs[d], off[d], pad[d]), 3)

    for (li, p) in enumerate(my_pids)
        dp = my_decomps[li]
        pad = dp.n_halo_d
        n_halo = dp.n_halo
        myint = ntuple(d -> dp.offset[d]+1:dp.offset[d]+dp.n_local[d], 3)
        for ds in 1:3, side in 1:2
            q = faces_all[p][ds][side]
            q == 0 && continue
            n_ext = regions[p].extent[ds]
            nq_ext = regions[q].extent[ds]
            otherside = side == 1 ? 2 : 1
            # This rank participates only if it owns the face plane.
            owns = side == 1 ? dp.offset[ds] == 0 :
                               dp.offset[ds] + dp.n_local[ds] == n_ext
            owns || continue
            # Map between my patch-node index m and the partner's g along ds:
            # hi face: m = n_ext + (g - 1); lo face: m = 1 - (nq_ext - g).
            tomine(g) = side == 2 ? n_ext + g - 1 : g - nq_ext + 1
            # My ghost layers correspond to partner interior nodes:
            src_q = side == 2 ? (2:n_halo+1) : (nq_ext-n_halo:nq_ext-1)
            # The partner's ghost layers correspond to my interior nodes:
            src_p = side == 2 ? (n_ext-n_halo:n_ext-1) : (2:n_halo+1)
            # Their ghost node for my source node m, in their patch-node space:
            totheirs(m) = side == 2 ? m - n_ext + 1 : m + nq_ext - 1
            plane_p = side == 2 ? n_ext : 1
            plane_q = side == 2 ? 1 : nq_ext
            rtag = _iface_tag(_GHOST_TAG_BASE, npatch, q, p, ds, side)
            stag = _iface_tag(_GHOST_TAG_BASE, npatch, p, q, ds, otherside)
            prtag = _iface_tag(_PLANE_TAG_BASE, npatch, q, p, ds, side)
            pstag = _iface_tag(_PLANE_TAG_BASE, npatch, p, q, ds, otherside)
            for e in table
                e.pid == q || continue
                eint = ntuple(d -> e.offset[d]+1:e.offset[d]+e.n_local[d], 3)
                tover = ntuple(d -> d == ds ? (1:1) : _isect(myint[d], eint[d]), 3)
                any(d -> d != ds && isempty(tover[d]), 1:3) && continue
                # Only the partner block that owns its own face plane holds the
                # ghost-source nodes (its extent is at least the scheme minimum,
                # which exceeds n_halo + 1) and carries interface ghosts of its
                # own to fill.
                owns_e = side == 1 ? e.offset[ds] + e.n_local[ds] == nq_ext :
                                     e.offset[ds] == 0
                # --- ghost receive: partner nodes src_q ∩ e's block → my ghosts
                dsq = owns_e ? _isect(src_q, eint[ds]) : (1:0)
                if !isempty(dsq) && e.rank != me
                    # (the e.rank == me case is realized as a local copy on the
                    # send side, so no receive record is built for it)
                    mine_r = ntuple(d -> d == ds ?
                        (tomine(first(dsq)):tomine(last(dsq))) : tover[d], 3)
                    minep = padded3(mine_r, dp.offset, pad)
                    buf = Vector{T}(undef, n_cons * prod(length.(minep)))
                    push!(ghost_recvs, GhostRecord{T}(li, e.rank, 0, rtag,
                          minep, minep, buf))
                end
                # --- ghost send: my nodes src_p ∩ my block, needed by e
                dsp = owns_e ? _isect(src_p, myint[ds]) : (1:0)
                if !isempty(dsp)
                    src = ntuple(d -> d == ds ? dsp : tover[d], 3)
                    srcp = padded3(src, dp.offset, pad)
                    if e.rank == me
                        lq = local_of[q]
                        dq = my_decomps[lq]
                        theirs_ds = totheirs(first(dsp)):totheirs(last(dsp))
                        dst = ntuple(d -> d == ds ? theirs_ds : tover[d], 3)
                        dstp = padded3(dst, dq.offset, dq.n_halo_d)
                        push!(ghost_sends, GhostRecord{T}(li, me, lq, stag,
                              srcp, dstp, Vector{T}()))
                    else
                        buf = Vector{T}(undef, n_cons * prod(length.(srcp)))
                        push!(ghost_sends, GhostRecord{T}(li, e.rank, 0, stag,
                              srcp, srcp, buf))
                    end
                end
                # --- shared plane: my node plane_p ↔ partner node plane_q
                if plane_q in eint[ds]
                    mine_r = ntuple(d -> d == ds ? (plane_p:plane_p) : tover[d], 3)
                    minep = padded3(mine_r, dp.offset, pad)
                    if e.rank == me
                        lq = local_of[q]
                        dq = my_decomps[lq]
                        theirs_r = ntuple(d -> d == ds ? (plane_q:plane_q) : tover[d], 3)
                        theirsp = padded3(theirs_r, dq.offset, dq.n_halo_d)
                        push!(plane_pairs, PlaneRecord{T}(li, me, lq, prtag, pstag,
                              minep, theirsp, Vector{T}(), Vector{T}()))
                    else
                        nvals = n_cons * prod(length.(minep))
                        push!(plane_pairs, PlaneRecord{T}(li, e.rank, 0, prtag, pstag,
                              minep, minep, Vector{T}(undef, nvals),
                              Vector{T}(undef, nvals)))
                    end
                end
            end
        end
    end
    return ghost_sends, ghost_recvs, plane_pairs
end

# --- Runtime exchange -------------------------------------------------------

"""
    exchange_patch_ghosts!(solver, states)

Fill every patch's interface ghost layers from the abutting patch's interior
values of the conserved state, and return `states` (the per-patch state vector
aligned with `solver.patches`). Ghost strips span each rank's own transverse
interior; the following within-patch [`exchange_state!`](@ref) carries them
into edge and corner halos, exactly as the sequential-dimension halo exchange
already fills corners. Collective over the ranks owning either side of any
interface; a run with one patch returns immediately.
"""
function exchange_patch_ghosts!(solver, states)
    recvs = solver.ghost_recvs
    sends = solver.ghost_sends
    (isempty(recvs) && isempty(sends)) && return states
    me = MPI.Comm_rank(solver.comm)
    reqs = MPI.Request[]
    for r in recvs
        r.partner == me && continue
        push!(reqs, MPI.Irecv!(r.buf, solver.comm; source=r.partner, tag=r.tag))
    end
    for s in sends
        if s.partner == me
            _copy_block!(states[s.partner_patch], s.theirs, states[s.patch], s.mine)
        else
            _pack!(s.buf, states[s.patch], s.mine)
            push!(reqs, MPI.Isend(s.buf, solver.comm; dest=s.partner, tag=s.tag))
        end
    end
    MPI.Waitall(reqs)
    for r in recvs
        r.partner == me && continue
        _unpack!(states[r.patch], r.buf, r.mine)
    end
    return states
end

"""
    average_shared_planes!(solver, states)

Make each shared interface-plane node consistent by replacing both patches'
copies with their mean, over each rank's transverse interior, and return
`states`. Both sides compute the identical mean of the identical pair, so the
result does not depend on which side is asked. Collective as
[`exchange_patch_ghosts!`](@ref) is; no-op with one patch.
"""
function average_shared_planes!(solver, states)
    planes = solver.plane_pairs
    isempty(planes) && return states
    me = MPI.Comm_rank(solver.comm)
    reqs = MPI.Request[]
    for pl in planes
        if pl.partner == me
            # A local pairing appears once from each side; averaging is
            # idempotent, so the second application is a no-op rather than a
            # double count.
            _average_blocks!(states[pl.patch], pl.mine,
                             states[pl.partner_patch], pl.theirs)
        else
            push!(reqs, MPI.Irecv!(pl.buf, solver.comm; source=pl.partner, tag=pl.tag))
        end
    end
    for pl in planes
        pl.partner == me && continue
        _pack!(pl.sbuf, states[pl.patch], pl.mine)
        # The send uses the RECEIVER's tag, computed at record build time.
        push!(reqs, MPI.Isend(pl.sbuf, solver.comm; dest=pl.partner, tag=pl.sendtag))
    end
    MPI.Waitall(reqs)
    for pl in planes
        pl.partner == me && continue
        _average_from!(states[pl.patch], pl.buf, pl.mine)
    end
    return states
end

"""
    sync_patches!(solver, states)

Bring the whole multi-patch state to mutual consistency: average the shared
interface planes, refill the interface ghost layers from the (averaged)
neighboring interiors, and exchange each patch's own rank-boundary halos so
edge and corner cells agree with both. Called by the multi-patch drivers after
every RK stage update and at the head of each `run!` iteration; a single-patch
solver never reaches it. Collective over `solver.comm`.
"""
function sync_patches!(solver, states)
    average_shared_planes!(solver, states)
    exchange_patch_ghosts!(solver, states)
    for (i, patch) in enumerate(solver.patches)
        exchange_state!(states[i], patch.decomp)
    end
    return states
end
