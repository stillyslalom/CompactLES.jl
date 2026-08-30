# Tagging and regridding for the two-level refinement.
#
# Every `RegridSpec.interval` coarse steps, `run!` retags the coarse level and
# rebuilds the level-1 patch when the tagged region moved. The tag is the
# undivided fourth difference of the mixture density, summed over the active
# dimensions and normalized by the local density, the same quantity the Cook
# δ⁴ sensors are built from. It is read from the conserved state directly, so
# tagging needs no primitives pass and can run at the head of the step loop.
# A coarse cell tags when that ratio exceeds `RegridSpec.threshold`; the tagged
# set is grown by `RegridSpec.buffer` coarse cells (measured: shock round-trip
# pollution decays ≈ 3.4× per point, bench/amr_transfer.jl, so a ~4-cell
# buffer drops interface pollution by two orders of magnitude) and its
# bounding box, clamped to the nesting margin, becomes the new refined region.
# The region is a single box, not a tile set, because the solver carries one
# level-1 patch and tile clustering only pays once several fine patches exist
# (reference/AMR_GPU.md, tiles and adjacency).
#
# The new fine state is initialized by the order-6 interpolation of the coarse
# state, the same operator as the live shell coupling and the right one here
# for the same reason: the coarse data are point samples, which the
# deconvolving `prolong!` would sharpen spuriously (see `interpolate!`).
# Surviving fine data on the overlap of the old and new regions is then copied
# across on the coincident fine lattice, holding one node off both patches'
# boundary planes (the old plane was imposed data; the new one is re-imposed by
# the next shell fill).
#
# When no cell tags, or the box did not move, the current region is kept: a
# feature fading below threshold leaves refinement where it last was instead
# of collapsing it, the conservative choice for a machinery whose purpose is
# robustness at captured features.

# One coarse cell's tag quantity: Σ_d |δ⁴_d ρ| / ρ, using the zeroth-order
# closed-edge clamp from `delta4_sum!`, with halos read across periodic
# wraps. `mark!(g1, g2, g3)` is called for every tagged cell of this rank's
# interior, in global node indices. Rank-local apart from the halo exchange.
function _tag_sweep!(mark!::F, solver::Solver, Qc) where {F}
    patches = getfield(solver, :patches)
    coarse = patches[1]
    dcp = coarse.decomp
    spec = getfield(solver, :regrid)
    exchange_state!(Qc, dcp)
    o = dcp.n_halo_d
    n = dcp.n_local
    n_species = solver.equations.n_species
    # A device-resident coarse patch downloads its block for the tag sweep (a
    # backend copy at the regrid cadence, not per step) and takes a host
    # scratch in place of tmp_a.
    Qc = _cpu_storage(Qc) ? Qc : Array(parent(Qc))
    # Mixture density over the interior extended two layers along each active
    # dimension: the δ⁴ taps reach that far, and beyond a closed edge the
    # clamped indexing below never reads it.
    rho = _cpu_storage(coarse.tmp_a) ? coarse.tmp_a :
          zeros(eltype(Qc), size(coarse.tmp_a))
    ext = ntuple(d -> dcp.active[d] ? (-1:n[d]+2) : (1:1), 3)
    @inbounds for k in ext[3], j in ext[2], i in ext[1]
        I = CartesianIndex(i + o[1], j + o[2], k + o[3])
        acc = zero(eltype(Qc))
        for sp in 1:n_species
            acc += Qc[I, sp]
        end
        rho[I] = acc
    end
    lomin = ntuple(d -> at_lo_edge(dcp, d) ? 1 : -1, 3)
    himax = ntuple(d -> at_hi_edge(dcp, d) ? n[d] : n[d] + 2, 3)
    thr = spec.threshold
    off = dcp.offset
    @inbounds for k in 1:n[3], j in 1:n[2], i in 1:n[1]
        I = CartesianIndex(i + o[1], j + o[2], k + o[3])
        s = zero(eltype(Qc))
        for d in 1:3
            dcp.active[d] || continue
            il = (d == 1 ? i : d == 2 ? j : k)
            e = CartesianIndex(ntuple(q -> q == d ? 1 : 0, 3))
            acc = zero(eltype(Qc))
            for m in -2:2
                ilm = clamp(il + m, lomin[d], himax[d])
                acc += D4[m + 3] * rho[I + (ilm - il) * e]
            end
            s += abs(acc)
        end
        s > thr * rho[I] && mark!(i + off[1], j + off[2], k + off[3])
    end
    return nothing
end

# Global bounding box of the tagged set, or `nothing`. Collective (one
# Allreduce), so every rank derives the identical box.
function _tag_bounds(solver::Solver, Qc)
    dcp = getfield(solver, :patches)[1].decomp
    lo = Ref((typemax(Int), typemax(Int), typemax(Int)))
    hi = Ref((0, 0, 0))
    _tag_sweep!(solver, Qc) do g1, g2, g3
        lo[] = min.(lo[], (g1, g2, g3))
        hi[] = max.(hi[], (g1, g2, g3))
    end
    # A rank with no tagged cell contributes typemax/0 sentinels that lose
    # every min/max.
    lo_g = ntuple(d -> hi[][d] == 0 ? typemax(Int) ÷ 2 : lo[][d], 3)
    hi_g = hi[]
    red = MPI.Allreduce(Int64[lo_g..., (-).(hi_g)...], min, dcp.comm)
    glo = ntuple(d -> Int(red[d]), 3)
    ghi = ntuple(d -> -Int(red[3 + d]), 3)
    ghi[1] == 0 && ghi[2] == 0 && ghi[3] == 0 && return nothing
    return (glo, ghi)
end

"""
    tagged_region(solver, Qc) -> Union{Nothing, BlockRegion}

The refined region the current coarse state calls for: the bounding box of
the tagged cells, buffered by `RegridSpec.buffer` coarse cells per side,
clamped to the nesting margin, and widened where necessary to the four-node
minimum extent. Returns `nothing` when no cell exceeds the threshold.
Collapsed dimensions keep offset 0 and extent 1. Collective over the coarse
communicator (one Allreduce of the tag bounds), so every rank returns the
identical region.
"""
function tagged_region(solver::Solver, Qc)
    spec = getfield(solver, :regrid)
    bounds = _tag_bounds(solver, Qc)
    bounds === nothing && return nothing
    tlo, thi = bounds
    margin = spec.margin
    n_global = solver.n_global
    offext = ntuple(3) do d
        n_global[d] > 1 || return (0, 1)
        lo = max(tlo[d] - spec.buffer, margin + 1)
        hi = min(thi[d] + spec.buffer, n_global[d] - margin)
        # The clamps can invert the interval when every tagged cell sits
        # inside the margin band; the widening below reopens it from
        # whichever end has room.
        hi < lo && (hi = lo)
        while hi - lo + 1 < 4
            hi < n_global[d] - margin ? (hi += 1) : (lo -= 1)
        end
        (lo - 1, hi - lo + 1)
    end
    return BlockRegion(ntuple(d -> offext[d][1], 3),
                       ntuple(d -> offext[d][2], 3))
end

# Whole-patch initialization of a fresh fine state: interpolate the parent
# state over the buffered box and write every slot, interior and shell alike.
function _fill_fine_from_coarse!(solver::Solver, states, lt::LevelTransfer)
    patches = getfield(solver, :patches)
    fine = patches[lt.fine_index]
    Qf = states[lt.fine_index]
    K = length(lt.pplans)
    _gather_box!(lt.box_gather, lt, states, patches)
    for c in 1:solver.equations.n_cons
        lt.pstage[1] .= view(lt.box_gather, :, :, :, c)
        for k in 1:K
            interpolate!(lt.pstage[k+1], lt.pplans[k], lt.pstage[k])
        end
        _write_fine_shell!(Qf, c, lt.pstage[K+1], lt, fine.decomp,
                           lt.pdecomps[K+1], false)
    end
    return states
end

# Copy surviving fine data from the old patch onto the new one over the
# coincident fine lattice of the region overlap, holding one node off both
# patches' boundary planes. Distributed: the old fine state gathers
# replicated over the old region (one transient region-sized array per rank,
# at the regrid cadence only), and each rank writes the overlap slots of its
# own new block. `old_gather` is that replica, indexed by old patch node.
function _carry_over!(Qf_new, dnew::Decomp, rnew::BlockRegion,
                      old_gather, Nf_old::NTuple{3,Int}, rold::BlockRegion,
                      active::NTuple{3,Bool}, n_cons::Int)
    # Root node g maps to fine-global 3(g − offset) − 2 on either patch, so
    # the old-to-new fine index shift is 3(offset_old − offset_new).
    shift = ntuple(d -> active[d] ? 3 * (rold.offset[d] - rnew.offset[d]) : 0, 3)
    Nf_new = ntuple(d -> active[d] ? 3 * rnew.extent[d] - 2 : rnew.extent[d], 3)
    rng = ntuple(3) do d
        active[d] || return 1:1
        glo = max(rold.offset[d], rnew.offset[d]) + 1
        ghi = min(rold.offset[d] + rold.extent[d], rnew.offset[d] + rnew.extent[d])
        f_lo = max(3 * (glo - rold.offset[d]) - 2, 2)
        f_hi = min(3 * (ghi - rold.offset[d]) - 2, Nf_old[d] - 1)
        # Window in NEW patch global nodes, then this rank's slice of it.
        w = max(f_lo + shift[d], 2):min(f_hi + shift[d], Nf_new[d] - 1)
        max(first(w), dnew.offset[d] + 1):min(last(w),
                                              dnew.offset[d] + dnew.n_local[d])
    end
    any(isempty, rng) && return Qf_new
    pn = dnew.n_halo_d
    off = dnew.offset
    if _cpu_storage(Qf_new)
        @inbounds for c in 1:n_cons, k in rng[3], j in rng[2], i in rng[1]
            Qf_new[i - off[1] + pn[1], j - off[2] + pn[2],
                   k - off[3] + pn[3], c] =
                old_gather[i - shift[1], j - shift[2], k - shift[3], c]
        end
        return Qf_new
    end
    # Device patch: the overlap window is one rectangular block, uploaded and
    # assigned by broadcast at the regrid cadence.
    win = view(old_gather, (rng[1] .- shift[1]), (rng[2] .- shift[2]),
               (rng[3] .- shift[3]), 1:n_cons)
    dev_win = similar(parent(Qf_new), size(win))
    copyto!(dev_win, Array(win))
    lr = ntuple(d -> (first(rng[d]) - off[d] + pn[d]):
                     (last(rng[d]) - off[d] + pn[d]), 3)
    view(parent(Qf_new), lr[1], lr[2], lr[3], 1:n_cons) .= dev_win
    return Qf_new
end

"""
    regrid!(solver, states, workspace, save) -> Bool

Retag the coarse level and, when the tagged region moved, rebuild the level-1
patch over it: a new [`Patch`](@ref) and [`LevelTransfer`](@ref) from the
schemes retained on [`RegridSpec`](@ref), a new fine state initialized by
order-6 interpolation of the coarse state with surviving fine data copied
across the region overlap, and fresh workspace arrays for the new extents.
Returns whether anything changed.

The retry savepoint, when one exists, is refreshed to the post-regrid state:
its snapshot has the old extents, and restoring across a layout change is not
meaningful. The state being banked completed its last step's health checks, so
this keeps the retry mechanism live at the cost of never rolling back past a
regrid.

The refresh observes the same suppression `run!` applies to its own
savepoint writes: after a rollback nothing is banked at or below the step
that failed (`Savepoint.guard`), so a regrid landing inside that window
cannot re-arm the retry loop that guard exists to break.
"""
function regrid!(solver::Solver{T}, states::Vector{<:ConservedState},
                 workspace::Workspace, save) where {T}
    spec = getfield(solver, :regrid)
    spec.tile > 0 && return _regrid_tiles!(solver, states, workspace, save)
    levels = getfield(solver, :levels)
    # A two-level hierarchy, enforced at setup: the root and one refined
    # patch, the sole transfer of level 1.
    lt = levels[2].transfers[1]
    newregion = tagged_region(solver, states[1])
    newregion === nothing && return false
    newregion == lt.region && return false
    patches = getfield(solver, :patches)
    fi = lt.fine_index
    oldfine = patches[fi]
    oldregion = lt.region
    Qf_old = states[fi]
    active_g = ntuple(d -> solver.n_global[d] > 1, 3)
    n_cons = solver.equations.n_cons
    # getfield: `h` is also a patch property name, and the property forwarding
    # refuses it on a multi-patch solver; the root spacing lives on the config.
    # Gather the surviving fine state BEFORE the patch swap, while the old
    # decomposition and block table are still live. Replicated over the old
    # region; freed when this call returns.
    dold = oldfine.decomp
    Nf_old = dold.n_global
    old_gather = zeros(T, Nf_old..., n_cons)
    gather_region!(old_gather, ntuple(d -> 1:Nf_old[d], 3), (0, 0, 0),
                   (0, 0, 0), Qf_old, dold, lt.fine_blocks)
    newfine = _build_fine_patch(T, newregion, active_g, getfield(solver, :h),
                                spec.n_halo, getfield(solver, :comm),
                                spec.deriv, spec.filt, spec.smoo,
                                solver.art.smoother, spec.interface_rhs,
                                spec.backend, solver.equations.n_species,
                                n_cons, fi, 1)
    newlt = build_level_transfer(T, newregion, active_g, spec.n_halo, [1], fi,
                                 lt.restriction, n_cons,
                                 getfield(solver, :subcycle),
                                 [patches[1].decomp], newfine.decomp)
    Qf_new = _state_like(newfine.rho, n_cons)
    patches[fi] = newfine
    levels[2] = Level{T}(1, [fi], [newlt])
    states[fi] = Qf_new
    workspace.dQ[fi] = zero(Qf_new)
    workspace.du[fi] = zero(Qf_new)
    init_geometry!(PatchSolver(solver, newfine))
    _fill_fine_from_coarse!(solver, states, newlt)
    _carry_over!(Qf_new, newfine.decomp, newregion, old_gather, Nf_old,
                 oldregion, active_g, n_cons)
    _rebank!(solver, states, save)
    return true
end

# Re-bank only where `run!` itself would: a regrid landing on a retried
# trajectory at or below the rollback guard would otherwise re-arm the
# rolled-back-to-the-failing-step loop that guard exists to break.
function _rebank!(solver, states, save)
    if save !== nothing && solver.step > save.guard
        save.Q = _snapshot(states)
        save.t = solver.t
        save.step = solver.step
    end
    return save
end

# --- Tiled regridding -------------------------------------------------------
#
# On a tiled level the tagged set selects lattice cells, not a box: a cell is
# wanted when any tagged coarse node, grown by the buffer, meets it. Because
# the lattice is global, a wanted tile that already exists keeps its region,
# its arrays and its state untouched (only its neighbor faces and transfer
# are rebuilt); a newly wanted tile is built and initialized by
# interpolation of the coarse state; an unwanted one is dropped, its last
# restriction already on the coarse level. No carry-over arises: distinct
# lattice cells overlap in a shared plane at most, and that plane takes the
# neighbor's values at the first averaging after the regrid.

# Flags over the lattice cells, reduced so every rank derives the same set.
function _tag_tiles(solver::Solver, Qc, spec::RegridSpec)
    n_global = solver.n_global
    active = ntuple(d -> n_global[d] > 1, 3)
    a = spec.tile
    K = ntuple(d -> active[d] ? (n_global[d] - 1) ÷ a + 1 : 1, 3)
    flags = zeros(Int8, K)
    b = spec.buffer
    _tag_sweep!(solver, Qc) do g1, g2, g3
        g = (g1, g2, g3)
        spans = ntuple(d -> active[d] ?
                       intersect(_tile_span(g[d] - b, g[d] + b, a), 0:K[d]-1) :
                       (0:0), 3)
        for k3 in spans[3], k2 in spans[2], k1 in spans[1]
            flags[k1 + 1, k2 + 1, k3 + 1] = one(Int8)
        end
    end
    return MPI.Allreduce(flags, max, getfield(solver, :comm)), K
end

function _regrid_tiles!(solver::Solver{T}, states::Vector{<:ConservedState},
                        workspace::Workspace, save) where {T}
    spec = getfield(solver, :regrid)
    levels = getfield(solver, :levels)
    patches = getfield(solver, :patches)
    lev = levels[2]
    n_global = solver.n_global
    active = ntuple(d -> n_global[d] > 1, 3)
    flags, K = _tag_tiles(solver, states[1], spec)
    any(!=(0), flags) || return false
    lo = ntuple(d -> 1 + spec.margin, 3)
    hi = ntuple(d -> n_global[d] - spec.margin, 3)
    wanted = BlockRegion[]
    for k3 in 0:K[3]-1, k2 in 0:K[2]-1, k1 in 0:K[1]-1
        flags[k1 + 1, k2 + 1, k3 + 1] == 0 && continue
        t = _lattice_tile((k1, k2, k3), active, spec.tile, lo, hi)
        t === nothing || push!(wanted, t)
    end
    isempty(wanted) && return false
    old_regions = [lt.region for lt in lev.transfers]
    wanted == old_regions && return false
    old_index = Dict(r => lev.patches[i] for (i, r) in enumerate(old_regions))
    # A tile about to leave restricts once more: the post-step restriction
    # preceded the positivity repair, and nothing else writes it back.
    for (i, r) in enumerate(old_regions)
        r in wanted || _restrict_patch!(solver, states, lev.transfers[i])
    end
    n_cons = solver.equations.n_cons
    root = patches[1]
    comm = getfield(solver, :comm)
    faces = _tile_faces(wanted)
    new_patches = Any[]
    new_states = similar(states, 0)
    new_dQ = similar(workspace.dQ, 0)
    new_du = similar(workspace.du, 0)
    transfers = LevelTransfer{T}[]
    indices = Int[]
    fresh = Int[]
    for (ti, tr) in enumerate(wanted)
        idx = ti + 1
        bcs = _fine_bcs(active, faces[ti])
        if haskey(old_index, tr)
            oi = old_index[tr]
            p = _repatch(patches[oi], idx, faces[ti], bcs)
            push!(new_states, states[oi])
            push!(new_dQ, workspace.dQ[oi])
            push!(new_du, workspace.du[oi])
        else
            p = _build_fine_patch(T, tr, active, root.h, spec.n_halo, comm,
                                  spec.deriv, spec.filt, spec.smoo,
                                  solver.art.smoother, spec.interface_rhs,
                                  spec.backend, solver.equations.n_species,
                                  n_cons, idx, 1, faces[ti])
            Q = _state_like(p.rho, n_cons)
            push!(new_states, Q)
            push!(new_dQ, zero(Q))
            push!(new_du, zero(Q))
            push!(fresh, ti)
        end
        push!(new_patches, p)
        push!(indices, idx)
        push!(transfers, build_level_transfer(T, tr, active, spec.n_halo, [1],
                                              idx, lev.transfers[1].restriction,
                                              n_cons, getfield(solver, :subcycle),
                                              [root.decomp], p.decomp, faces[ti]))
    end
    records = _level_records(T, comm, [p.region for p in new_patches], indices,
                             [p.decomp for p in new_patches], n_cons)
    resize!(patches, 1 + length(wanted))
    resize!(states, 1 + length(wanted))
    resize!(workspace.dQ, 1 + length(wanted))
    resize!(workspace.du, 1 + length(wanted))
    for ti in eachindex(wanted)
        patches[ti + 1] = new_patches[ti]
        states[ti + 1] = new_states[ti]
        workspace.dQ[ti + 1] = new_dQ[ti]
        workspace.du[ti + 1] = new_du[ti]
    end
    levels[2] = Level{T}(1, indices, transfers, records)
    for ti in fresh
        init_geometry!(PatchSolver(solver, patches[ti + 1]))
        _fill_fine_from_coarse!(solver, states, transfers[ti])
    end
    # A fresh tile's plane shared with a survivor takes the survivor's
    # evolved values rather than averaging its interpolation into them.
    if !isempty(fresh)
        is_fresh = falses(length(patches))
        for ti in fresh
            is_fresh[ti + 1] = true
        end
        _seed_planes!(solver, states, levels[2], is_fresh)
    end
    _rebank!(solver, states, save)
    return true
end

# The `run!` hook: cadence check, then `regrid!`. The single-array state form
# never regrids; a multi-patch solver without a RegridSpec returns at the
# first test.
_maybe_regrid!(solver::Solver, Q, workspace::Workspace, save) = nothing
function _maybe_regrid!(solver::Solver, states::Vector{<:ConservedState},
                        workspace::Workspace, save)
    spec = getfield(solver, :regrid)
    spec === nothing && return nothing
    solver.step - spec.last_step >= spec.interval || return nothing
    spec.last_step = solver.step
    regrid!(solver, states, workspace, save)
    return nothing
end
