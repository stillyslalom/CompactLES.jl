# The transfers between a level and its parent: the prolongation of the
# coarse state to a tile's ghost ring and boundary planes, the shell
# imposition and its admissible fallback, the restriction, and the Hermite
# box of a subcycled run. The header of levels_types.jl describes the
# design.

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
# Where `ring` is true (`_fills_gradient_ring`), each component's ring is
# followed by the rings of its derivatives along the three dimensions (zero
# along a collapsed one), taken on the fine box the chain produced, and the
# unpack fills the gradient ring beside the shell ring. A device patch takes
# the derivatives through its scratch's device plans and packs and unpacks
# both rings on the device, so the box stays there; only the Allgatherv of a
# decomposed patch stages the packed rings through the host. Elsewhere the
# chain fills only the boxes the shell slots read, whose values are the full
# chain's bit for bit (`interpolate!`), so the shell does not depend on `ring`.
# This form for one transfer is `hermite_level_shell!`'s; the drivers call
# `_impose_shell_at!`.
_impose_shell!(solver, states, lt::LevelTransfer, fill) =
    _impose_shell_on!(solver, states, lt, fill,
                      _cold(getfield(solver, :patches)[lt.fine_index]))
_impose_shell_on!(solver, states, lt::LevelTransfer, fill, fine) =
    _impose_shell!(solver, states, lt, fill, fine.decomp,
                   _fills_gradient_ring(solver, fine, lt))

# `_impose_shell!` for `transfers[t]` on its tile `fine`, which the caller
# passes through `_cold`. This barrier, compiled for the tile's concrete type
# (`_unit_call`), reads the tile's decomposition unboxed, and the imposition,
# which takes the decomposition and not the tile, is compiled once for every
# tile type. Read from an abstractly typed tile, the decomposition is boxed,
# and so is the transfer at each dynamic call taking it; the transfer crosses
# this dynamic call as its vector and index for the same reason.
_impose_shell_at!(solver, states, transfers, t::Int, fill, fine) =
    _impose_shell!(solver, states, transfers[t], fill, fine.decomp,
                   _fills_gradient_ring(solver, fine, transfers[t]))

# Whether the imposition on the tile `fine` fills the gradient ring of `lt`:
# where the transfer carries one (`_gradient_ring_built`) and a coarse-fine
# face of the tile reads it (`_reads_gradient_ring`). The test is the
# reader's own, on setup constants of the tile and the solver-wide toggles,
# so every rank of the tile's communicator, which the fill's gather spans,
# reaches the same value.
_fills_gradient_ring(solver, fine, lt::LevelTransfer) =
    lt.gradients !== nothing && _reads_gradient_ring(PatchSolver(solver, fine))

# The shell of the tile held as patch `pi` of `lev`, the form `_foreach_tile`
# calls.
function _tile_shell!(solver, pi::Int, lev, states, fill)
    t = lev.tiles[findfirst(==(pi), lev.patches)]
    _impose_shell_at!(solver, states, lev.transfers, t, fill,
                      _cold(getfield(solver, :patches)[pi]))
    return nothing
end

# The size of the tile's communicator and this rank's place in it, read
# without an MPI call where one rank holds the tile, so that the tiles of a
# level may impose their shells concurrently (`_foreach_tile`).
function _shell_ranks(fdcp::Decomp)
    np = prod(fdcp.dims)
    return np, np == 1 ? 0 : MPI.Comm_rank(fdcp.comm)
end

function _impose_shell!(solver, states, lt::LevelTransfer, fill, fdcp::Decomp,
                        ring::Bool)
    # A tile whose every face is shared with a same-level tile or lies on the
    # domain boundary has no shell slot to write (`_in_shell`), and no reader
    # of the ring, so the chain is skipped. `imposed` is a property of the
    # transfer, the same on every rank of the tile's communicator, so the
    # ranks skip the ring gather together.
    _imposes_shell(lt) || return states
    shell = lt.shell
    if ring
        shell.memo[1] = 0
        return _impose_shell_gradients!(solver, states, lt, fill, fdcp)
    end
    Qf = states[lt.fine_index]
    comm = fdcp.comm
    np, me = _shell_ranks(fdcp)
    n_cons = solver.equations.n_cons
    K = length(lt.pplans)
    slabs = shell.slabs
    table = shell.table
    ringlen = shell.len
    ring = shell.ring
    host = !_device_path(Qf)
    # A repeated Hermite fraction (a substep's first stage after the previous
    # substep's end, the end again after the filter) reads the boxes the last
    # imposition read, so the ring it left is the ring this one would build.
    # The memo is the same on every rank of the tile, which skip the gather
    # together.
    if host && _shell_memo_hit(shell.memo, fill)
        _write_shell_from_ring!(Qf, ring, table, lt, fdcp, n_cons)
        return states
    end
    shell.memo[1] = 0
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
    if host
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
            _shell_memo!(shell.memo, fill)
            _write_shell_from_ring!(Qf, ring, table, lt, fdcp, n_cons)
            return states
        end
        gathered && _shell_linear_ring!(sendbuf, ringlen * n_owned, ringlen, parent(Qf),
                                        _linear_sources(fill, lt, owned), lt, fdcp,
                                        n_owned)
    else
        # Behind a barrier (`_cold`), which also keeps the tile's type out of
        # the host body.
        fine = _cold(getfield(solver, :patches)[lt.fine_index])
        _impose_shell_dev!(solver, Qf, lt, fill, fine, sendbuf, owned, fallback,
                           gathered) && return states
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
    host && _shell_memo!(shell.memo, fill)
    _write_shell_from_ring!(Qf, ring, table, lt, fdcp, n_cons)
    return states
end

# Whether the host ring holds the shell of the Hermite fill `fill`, and the
# record that it does. A gathered box is gathered anew at every imposition,
# so a `BoxFill` never matches; `save_level_boxes!` clears the record.
_shell_memo_hit(memo, ::BoxFill) = false
_shell_memo_hit(memo, fill::HermiteFill) =
    memo[1] == 1 && memo[2] == Float64(fill.θ) && memo[3] == Float64(fill.dt)
_shell_memo!(memo, ::BoxFill) = (memo[1] = 0; nothing)
function _shell_memo!(memo, fill::HermiteFill)
    memo[1] = 1
    memo[2] = Float64(fill.θ)
    memo[3] = Float64(fill.dt)
    return nothing
end

# The device branch of `_impose_shell!`: the chain and the ring pack on the
# device, then on one rank the shell write from the packed ring (`true`), and
# on a decomposed patch this rank's rings staged into `sendbuf` for the gather
# (`false`).
function _impose_shell_dev!(solver, Qf, lt::LevelTransfer, fill, fine, sendbuf, owned,
                            fallback::Bool, gathered::Bool)
    fdcp = fine.decomp
    np = MPI.Comm_size(fdcp.comm)
    n_cons = solver.equations.n_cons
    n_owned = length(owned)
    K = length(lt.pplans)
    shell = lt.shell
    table = shell.table
    ringlen = shell.len
    padb = lt.pdecomps[K+1].n_halo_d
    shift = _box_shift(lt)
    scratch = fine.amr.level_scratch
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
        return true
    end
    _tracked_copy!(sendbuf, 1, dsend, 1, ringlen * n_owned)
    if gathered
        dlinear = _device_send_stage(parent(Qf), ringlen * n_owned)
        _shell_linear_ring!(dlinear, 0, ringlen, parent(Qf),
                            _linear_sources_dev(fill, scratch, lt, owned), lt, fdcp,
                            n_owned)
        _tracked_copy!(sendbuf, ringlen * n_owned + 1, dlinear, 1, ringlen * n_owned)
    end
    return false
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

function _impose_shell_gradients!(solver, states, lt::LevelTransfer, fill, fdcp::Decomp)
    Qf = states[lt.fine_index]
    comm = fdcp.comm
    np, me = _shell_ranks(fdcp)
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
        # Behind a barrier, as in `_impose_shell!`.
        fine = _cold(getfield(solver, :patches)[lt.fine_index])
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
    scratch = fine.amr.level_scratch
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
            g1 = lo[1] + _unchecked_rem(r, n1)
            r = _unchecked_div(r, n1)
            g2 = lo[2] + _unchecked_rem(r, n2)
            g3 = lo[3] + _unchecked_div(r, n2)
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
        blk, j = _unchecked_div(p - 1, per), _unchecked_rem(p - 1, per)
        c = 0
        for r in 0:np-1
            nr = r < n_cons ? _unchecked_div(n_cons - r - 1, np) + 1 : 0
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
            g1 = lo[1] + _unchecked_rem(r, n1)
            r = _unchecked_div(r, n1)
            g2 = lo[2] + _unchecked_rem(r, n2)
            g3 = lo[3] + _unchecked_div(r, n2)
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
        _charge_exchange!(solver, lev, t0)
        _wait!(solver, t0)
        # The imposition is collective over the tile's own communicator,
        # which its holders alone enter, and reads and writes nothing of
        # another tile's, so the tiles take it concurrently where they may.
        # The imposed shell replaces the halo values the previous exchange
        # left wherever the two overlap (the edge-owning ranks' outer halos);
        # interior rank-boundary halos keep their exchanged values, so the
        # composite fine state is self-consistent without a further exchange.
        _foreach_tile(_tile_shell!, solver, lev, lev, states, BoxFill())
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
    _charge_exchange!(solver, lev, t0)
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
        # The ring no longer holds a shell of these boxes.
        lt.shell.memo[1] = 0
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

# The same for the transfer `transfers[t]`, the form the subcycled driver
# calls (`_impose_shell_at!`).
function hermite_level_shell!(solver, states, transfers::AbstractVector, t::Int, θ, dt)
    patches = getfield(solver, :patches)
    _impose_shell_at!(solver, states, transfers, t, HermiteFill(θ, dt),
                      _cold(patches[transfers[t].fine_index]))
    return states
end

# The same for the tile held as patch `pi` of `lev`, the form `_foreach_tile`
# calls.
function _tile_hermite_shell!(solver, pi::Int, lev, states, θ, dt)
    t = lev.tiles[findfirst(==(pi), lev.patches)]
    hermite_level_shell!(solver, states, lev.transfers, t, θ, dt)
    return nothing
end
