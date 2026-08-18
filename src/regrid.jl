# Tagging and regridding for the two-level refinement (Stage 4 of
# reference/AMR_GPU.md).
#
# Every `RegridSpec.interval` coarse steps, `run!` retags the coarse level and
# rebuilds the level-1 patch when the tagged region moved. The tag is the
# undivided fourth difference of the mixture density, summed over the active
# dimensions and normalized by the local density — the same quantity the Cook
# δ⁴ sensors are built from, read from the conserved state directly so tagging
# needs no primitives pass and can run at the head of the step loop. A coarse
# cell tags when that ratio exceeds `RegridSpec.threshold`; the tagged set is
# grown by `RegridSpec.buffer` coarse cells (the Stage 1 measurement: shock
# round-trip pollution decays ≈ 3.4× per point, so a ~4-cell buffer drops
# interface pollution by two orders of magnitude) and its bounding box, clamped
# to the nesting margin, becomes the new refined region. A single box rather
# than a tile set is deliberate: the Stage 3 base carries one level-1 patch,
# and the plan's tile clustering only pays once several fine patches exist.
#
# The new fine state is initialized by the order-6 interpolation of the coarse
# state — the same operator as the live shell coupling, and the right one here
# for the same reason: the coarse data are point samples, which the
# deconvolving `prolong!` would sharpen spuriously (see `interpolate!`).
# Surviving fine data on the overlap of the old and new regions is then copied
# across on the coincident fine lattice, holding one node off both patches'
# boundary planes (the old plane was imposed data; the new one is re-imposed by
# the next shell fill).
#
# When no cell tags, or the box did not move, the current region is kept: a
# feature fading below threshold leaves refinement where it last was rather
# than collapsing it, which is the conservative choice for a machinery whose
# purpose is robustness at captured features.

# One coarse cell's tag quantity: Σ_d |δ⁴_d ρ| / ρ, indices clamped at closed
# edges exactly as `delta4_sum!` clamps (a zeroth-order extension), halos read
# across periodic wraps.
function _tag_bounds(solver::Solver, Qc)
    patches = getfield(solver, :patches)
    coarse = patches[1]
    dcp = coarse.decomp
    spec = getfield(solver, :regrid)
    exchange_state!(Qc, dcp)
    o = dcp.n_halo_d
    n = dcp.n_local
    n_species = solver.equations.n_species
    # Mixture density over the interior extended two layers along each active
    # dimension: the δ⁴ taps reach that far, and beyond a closed edge the
    # clamped indexing below never reads it.
    rho = coarse.tmp_a
    ext = ntuple(d -> dcp.active[d] ? (-1:n[d]+2) : (1:1), 3)
    @inbounds for k in ext[3], j in ext[2], i in ext[1]
        I = CartesianIndex(i + o[1], j + o[2], k + o[3])
        acc = 0.0
        for sp in 1:n_species
            acc += Qc[I, sp]
        end
        rho[I] = acc
    end
    lomin = ntuple(d -> at_lo_edge(dcp, d) ? 1 : -1, 3)
    himax = ntuple(d -> at_hi_edge(dcp, d) ? n[d] : n[d] + 2, 3)
    thr = spec.threshold
    lo = (typemax(Int), typemax(Int), typemax(Int))
    hi = (0, 0, 0)
    @inbounds for k in 1:n[3], j in 1:n[2], i in 1:n[1]
        I = CartesianIndex(i + o[1], j + o[2], k + o[3])
        s = 0.0
        for d in 1:3
            dcp.active[d] || continue
            il = (d == 1 ? i : d == 2 ? j : k)
            e = CartesianIndex(ntuple(q -> q == d ? 1 : 0, 3))
            acc = 0.0
            for m in -2:2
                ilm = clamp(il + m, lomin[d], himax[d])
                acc += D4[m + 3] * rho[I + (ilm - il) * e]
            end
            s += abs(acc)
        end
        if s > thr * rho[I]
            lo = min.(lo, (i, j, k))
            hi = max.(hi, (i, j, k))
        end
    end
    hi[1] == 0 && hi[2] == 0 && hi[3] == 0 && return nothing
    return (lo, hi)
end

"""
    tagged_region(solver, Qc) -> Union{Nothing, BlockRegion}

The refined region the current coarse state calls for: the bounding box of
the tagged cells, buffered by `RegridSpec.buffer` coarse cells per side,
clamped to the nesting margin, and widened where necessary to the four-node
minimum extent. Returns `nothing` when no cell exceeds the threshold.
Collapsed dimensions keep offset 0 and extent 1. Serial, like the level
machinery.
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

# Whole-patch initialization of a fresh fine state: interpolate the coarse
# state over the buffered box and write every slot, interior and shell alike.
function _fill_fine_from_coarse!(solver::Solver, states)
    lt = solver.level_transfer
    patches = getfield(solver, :patches)
    coarse = patches[1]
    fine = patches[lt.fine_index]
    Qc = states[1]
    Qf = states[lt.fine_index]
    K = length(lt.pplans)
    for c in 1:solver.equations.n_cons
        _copy_coarse_box!(lt.pstage[1], Qc, c, lt, coarse.decomp, lt.pdecomps[1])
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
# patches' boundary planes.
function _carry_over!(Qf_new, dnew::Decomp, rnew::BlockRegion,
                      Qf_old, dold::Decomp, rold::BlockRegion,
                      active::NTuple{3,Bool}, n_cons::Int)
    # Root node g maps to fine-local 3(g − offset) − 2 on either patch, so the
    # old-to-new fine index shift is 3(offset_old − offset_new).
    shift = ntuple(d -> active[d] ? 3 * (rold.offset[d] - rnew.offset[d]) : 0, 3)
    rng = ntuple(3) do d
        active[d] || return 1:1
        glo = max(rold.offset[d], rnew.offset[d]) + 1
        ghi = min(rold.offset[d] + rold.extent[d], rnew.offset[d] + rnew.extent[d])
        f_lo = max(3 * (glo - rold.offset[d]) - 2, 2)
        f_hi = min(3 * (ghi - rold.offset[d]) - 2, dold.n_local[d] - 1)
        max(f_lo + shift[d], 2):min(f_hi + shift[d], dnew.n_local[d] - 1)
    end
    any(isempty, rng) && return Qf_new
    pn = dnew.n_halo_d
    po = dold.n_halo_d
    @inbounds for c in 1:n_cons, k in rng[3], j in rng[2], i in rng[1]
        Qf_new[i + pn[1], j + pn[2], k + pn[3], c] =
            Qf_old[i - shift[1] + po[1], j - shift[2] + po[2],
                   k - shift[3] + po[3], c]
    end
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
"""
function regrid!(solver::Solver{T}, states::Vector{<:ConservedState},
                 workspace::Workspace, save) where {T}
    spec = getfield(solver, :regrid)
    lt = getfield(solver, :level_transfer)
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
    newfine = _build_fine_patch(T, newregion, active_g, getfield(solver, :h),
                                spec.n_halo,
                                spec.deriv, spec.filt, spec.smoo,
                                solver.art.smoother, spec.interface_rhs,
                                spec.backend, solver.equations.n_species,
                                n_cons)
    newlt = build_level_transfer(T, newregion, active_g, spec.n_halo, fi,
                                 lt.restriction, n_cons,
                                 getfield(solver, :subcycle))
    Qf_new = allocate_state(newfine.decomp, n_cons)
    patches[fi] = newfine
    solver.level_transfer = newlt
    states[fi] = Qf_new
    workspace.dQ[fi] = zero(Qf_new)
    workspace.du[fi] = zero(Qf_new)
    init_geometry!(PatchSolver(solver, newfine))
    _fill_fine_from_coarse!(solver, states)
    _carry_over!(Qf_new, newfine.decomp, newregion, Qf_old, oldfine.decomp,
                 oldregion, active_g, n_cons)
    if save !== nothing
        save.Q = _snapshot(states)
        save.t = solver.t
        save.step = solver.step
    end
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
