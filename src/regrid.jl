# Tagging and regridding for the two-level refinement.
#
# Every `RegridSpec.interval` coarse steps, `run!` retags the coarse level and
# rebuilds the level-1 patch when the tagged region moved. The tag is a union
# of criteria (the "Tag criteria" section below); the one always on is the
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

# --- Tag criteria -------------------------------------------------------------
#
# The tag is a union of criteria, each a per-point body over the parent
# level's state run through `pointwise!` on the host and writing a flag into
# the sweep scratch `RegridSpec.tags`. A body reads the conserved state and
# the persistent per-patch arrays only: the RHS scratch (`sensor` among it)
# lives on the pooled workspace and, at the head of the step loop, holds
# whichever patch of that padded extent last evaluated a right-hand side,
# so nothing here reads it. The artificial coefficients are per patch and
# outlive their evaluation (they feed the next step's `max_rate`), which is
# how the scheme's own sensor reaches the tag: μ*, β*, κ* and D* are
# C·ρ·max(sensor, 0) of the last evaluation, and their sum in units of the
# acoustic cell diffusivity c·h is the artificial diffusivity number the
# sensor criterion thresholds.
#
# Every body indexes its neighbors through the clamp `[lomin, himax]`: one
# halo layer (two for the δ⁴ taps) across a rank or periodic boundary, and
# the edge node itself at a closed edge, where the difference becomes
# one-sided. `active`, `o` (the halo pad) and the clamps are tuples so a body
# takes plain arrays and isbits scalars, the pointwise contract.
#
# A body writes one of two levels: 2 where its quantity exceeds the
# criterion's threshold (the node tags), 1 where it exceeds the threshold
# over `RegridSpec.untag_ratio` (the node holds: it keeps an existing tile
# but calls for no new one). The union over criteria is the maximum, so a
# body never lowers a level another body wrote.

const TAG_HOLD = Int8(1)
const TAG_MARK = Int8(2)

@inline _tag_unit(d::Int) = CartesianIndex(d == 1 ? 1 : 0, d == 2 ? 1 : 0, d == 3 ? 1 : 0)

@inline _tag_level(q, hi, lo) = q > hi ? TAG_MARK : q > lo ? TAG_HOLD : zero(Int8)

@inline function _raise_tag!(tags, I, level::Int8)
    @inbounds level > tags[I] && (tags[I] = level)
    return nothing
end

# Σ_d |δ⁴_d ρ| / ρ against the threshold pair, the zeroth-order closed-edge
# clamp of `delta4_sum!`.
@inline function _tag_delta4_point!(tags, rho, thr, thr_lo, lomin, himax,
                                    active, o, i, j, k)
    @inbounds begin
        I = CartesianIndex(i + o[1], j + o[2], k + o[3])
        s = zero(eltype(rho))
        for d in 1:3
            active[d] || continue
            il = (d == 1 ? i : d == 2 ? j : k)
            e = _tag_unit(d)
            acc = zero(eltype(rho))
            for m in -2:2
                ilm = clamp(il + m, lomin[d], himax[d])
                acc += D4[m + 3] * rho[I + (ilm - il) * e]
            end
            s += abs(acc)
        end
        _raise_tag!(tags, I, _tag_level(s, thr * rho[I], thr_lo * rho[I]))
    end
    return nothing
end

# max_k |δY_k| > threshold: the mass-fraction change per cell, the centered
# difference (Y⁺ − Y⁻)/(i⁺ − i⁻) along each active dimension in cell units,
# so the quantity is dimensionless and one-sided at a closed edge.
@inline function _tag_gradient_point!(tags, Q, rho, thr, thr_lo, n_species,
                                      lomin, himax, active, o, i, j, k)
    @inbounds begin
        I = CartesianIndex(i + o[1], j + o[2], k + o[3])
        q = zero(eltype(rho))
        for sp in 1:n_species
            s2 = zero(eltype(rho))
            for d in 1:3
                active[d] || continue
                il = (d == 1 ? i : d == 2 ? j : k)
                ip = min(il + 1, himax[d])
                im = max(il - 1, lomin[d])
                e = _tag_unit(d)
                Ip = I + (ip - il) * e
                Im = I + (im - il) * e
                dY = (Q[Ip, sp] / rho[Ip] - Q[Im, sp] / rho[Im]) / (ip - im)
                s2 += dY * dY
            end
            q = max(q, s2)
        end
        _raise_tag!(tags, I, _tag_level(q, thr * thr, thr_lo * thr_lo))
    end
    return nothing
end

# |∇ × u| > threshold from centered differences of the velocity
# components, u_c = Q[I, i_mom[c]] / ρ.
@inline function _tag_vorticity_point!(tags, Q, rho, thr, thr_lo, i_mom, h,
                                       lomin, himax, active, o, i, j, k)
    @inbounds begin
        I = CartesianIndex(i + o[1], j + o[2], k + o[3])
        w1 = w2 = w3 = zero(eltype(rho))
        for d in 1:3
            active[d] || continue
            il = (d == 1 ? i : d == 2 ? j : k)
            ip = min(il + 1, himax[d])
            im = max(il - 1, lomin[d])
            e = _tag_unit(d)
            Ip = I + (ip - il) * e
            Im = I + (im - il) * e
            inv = one(eltype(rho)) / ((ip - im) * h[d])
            du1 = (Q[Ip, i_mom[1]] / rho[Ip] - Q[Im, i_mom[1]] / rho[Im]) * inv
            du2 = (Q[Ip, i_mom[2]] / rho[Ip] - Q[Im, i_mom[2]] / rho[Im]) * inv
            du3 = (Q[Ip, i_mom[3]] / rho[Ip] - Q[Im, i_mom[3]] / rho[Im]) * inv
            if d == 1
                w2 -= du3
                w3 += du2
            elseif d == 2
                w3 -= du1
                w1 += du3
            else
                w1 -= du2
                w2 += du1
            end
        end
        _raise_tag!(tags, I, _tag_level(w1 * w1 + w2 * w2 + w3 * w3,
                                        thr * thr, thr_lo * thr_lo))
    end
    return nothing
end

# ((μ* + β*)/ρ + κ*/(ρ c_p) + max_k D*_k) / (c h_min) > threshold: the
# artificial diffusivity of the last right-hand side in units of the
# acoustic cell diffusivity. The primitives ρ, c and c_p are the same
# evaluation's, so the ratio is consistent; a node the positivity
# placeholder wrote (c = 0) never tags.
@inline function _tag_sensor_point!(tags, mu, beta, kappa, D, rho, c, cp, thr,
                                    thr_lo, hmin, n_species, o, i, j, k)
    @inbounds begin
        I = CartesianIndex(i + o[1], j + o[2], k + o[3])
        ci = c[I]
        ri = rho[I]
        cpi = cp[I]
        (ci > 0 && ri > 0 && cpi > 0) || return nothing
        nu = (mu[I] + beta[I]) / ri + kappa[I] / (ri * cpi)
        Dmax = zero(nu)
        for sp in 1:n_species
            Dmax = max(Dmax, D[sp][I])
        end
        _raise_tag!(tags, I, _tag_level(nu + Dmax, thr * ci * hmin,
                                        thr_lo * ci * hmin))
    end
    return nothing
end

# The user closure, a serial host loop behind a function barrier: the
# closure's type is the barrier's parameter, so the loop specializes on it
# and the call inside is direct.
function _tag_predicate!(tags, predicate::F, ps, n::NTuple{3,Int},
                         o::NTuple{3,Int}) where {F}
    @inbounds for k in 1:n[3], j in 1:n[2], i in 1:n[1]
        I = CartesianIndex(i + o[1], j + o[2], k + o[3])
        predicate(ps, I) && (tags[I] = TAG_MARK)
    end
    return nothing
end

# Host alias of a patch array: the array itself on the CPU backend, a
# download of a device one (a backend copy at the regrid cadence only).
_tag_host(x) = _cpu_storage(x) ? x : Array(x)

# Mixture density at one node of the tag sweep's extended block: the point
# `(i, j, k)` of the iteration box maps to the padded slot `base + (i, j, k)`.
@inline function _tag_rho_point!(rho, Qc, n_species, base, i, j, k)
    @inbounds begin
        I = CartesianIndex(i + base[1], j + base[2], k + base[3])
        acc = zero(eltype(rho))
        for sp in 1:n_species
            acc += Qc[I, sp]
        end
        rho[I] = acc
    end
    return nothing
end

# Tag sweep over this rank's interior of the parent (root) patch: every
# enabled criterion writes its levels into `spec.tags`, and `mark!(g1, g2,
# g3, level)` is then called for every flagged node, in global node indices
# with its level (`TAG_MARK` or `TAG_HOLD`). Rank-local apart from the halo
# exchange.
function _tag_sweep!(mark!::F, solver::Solver, Qc) where {F}
    patches = getfield(solver, :patches)
    coarse = patches[1]
    dcp = coarse.decomp
    spec = getfield(solver, :regrid)
    exchange_state!(Qc, dcp)
    o = dcp.n_halo_d
    n = dcp.n_local
    active = dcp.active
    n_species = solver.equations.n_species
    # A device-resident coarse patch evaluates the criteria on the device
    # and downloads the tag bytes alone; the predicate criterion is a user
    # closure over the patch, so with one set the block downloads and the
    # sweep runs on the host as a whole (a backend copy at the regrid
    # cadence, not per step).
    on_device = _device_path(Qc) && spec.predicate === nothing
    Qc = _cpu_storage(Qc) || on_device ? Qc : Array(parent(Qc))
    # Mixture density over the interior extended two layers along each active
    # dimension: the δ⁴ taps reach that far, and beyond a closed edge the
    # clamped indexing below never reads it.
    tmp_a = coarse.rhs_workspace.tmp_a
    rho = _cpu_storage(tmp_a) == _cpu_storage(Qc) ? tmp_a :
          zeros(eltype(Qc), size(tmp_a))
    ext = ntuple(d -> active[d] ? (-1:n[d]+2) : (1:1), 3)
    pointwise!(_tag_rho_point!, rho, length(ext[1]), length(ext[2]), length(ext[3]),
               rho, Qc, n_species, ntuple(d -> first(ext[d]) - 1 + o[d], 3))
    lomin = ntuple(d -> at_lo_edge(dcp, d) ? 1 : -1, 3)
    himax = ntuple(d -> at_hi_edge(dcp, d) ? n[d] : n[d] + 2, 3)
    tags = on_device ? similar(parent(Qc), Int8, size(spec.tags)) : spec.tags
    fill!(tags, zero(Int8))
    ratio = spec.untag_ratio
    pointwise!(_tag_delta4_point!, tags, n[1], n[2], n[3],
               tags, rho, spec.threshold, spec.threshold / ratio,
               lomin, himax, active, o)
    if spec.gradient_threshold > 0
        thr = spec.gradient_threshold
        pointwise!(_tag_gradient_point!, tags, n[1], n[2], n[3],
                   tags, Qc, rho, thr, thr / ratio, n_species,
                   lomin, himax, active, o)
    end
    if spec.vorticity_threshold > 0
        thr = spec.vorticity_threshold
        pointwise!(_tag_vorticity_point!, tags, n[1], n[2], n[3],
                   tags, Qc, rho, thr, thr / ratio,
                   solver.equations.i_mom, coarse.h, lomin, himax, active, o)
    end
    if spec.sensor_threshold > 0
        thr = spec.sensor_threshold
        hmin = minimum(coarse.h[d] for d in 1:3 if active[d])
        host(a) = on_device ? a : _tag_host(a)
        D_host = FieldVector([host(a) for a in coarse.D_art])
        pointwise!(_tag_sensor_point!, tags, n[1], n[2], n[3],
                   tags, host(coarse.mu_art), host(coarse.beta_art),
                   host(coarse.kappa_art), D_host, host(coarse.rho),
                   host(coarse.c), host(coarse.cp_mix),
                   thr, thr / ratio, hmin, n_species, o)
    end
    spec.predicate === nothing ||
        _tag_predicate!(tags, spec.predicate, PatchSolver(solver, coarse), n, o)
    if on_device
        copyto!(spec.tags, tags)
        tags = spec.tags
    end
    off = dcp.offset
    @inbounds for k in 1:n[3], j in 1:n[2], i in 1:n[1]
        I = CartesianIndex(i + o[1], j + o[2], k + o[3])
        tags[I] == 0 || mark!(i + off[1], j + off[2], k + off[3], tags[I])
    end
    return nothing
end

# Is global node `g` inside one of the current refined regions (parent
# node space)? The hold level of the hysteresis counts there only.
function _in_regions(regions::Vector{BlockRegion}, g::NTuple{3,Int})
    for r in regions
        all(d -> r.offset[d] < g[d] <= r.offset[d] + r.extent[d], 1:3) && return true
    end
    return false
end

# Global bounding box of the tagged set, or `nothing`. The globally reduced
# results are whether any rank has a `TAG_MARK` and the extrema of all marked
# nodes plus `TAG_HOLD` nodes inside current refined regions. They share one
# packed `Allreduce` on the root communicator. Without a global mark, the
# function returns `nothing` and retains the current box. Otherwise, held nodes
# may enlarge the box initiated by marked nodes, but cannot define or shrink a
# box by themselves. Every rank derives the same result.
function _tag_bounds(solver::Solver, Qc)
    dcp = getfield(solver, :patches)[1].decomp
    current = [lt.region for lt in getfield(solver, :levels)[2].transfers]
    lo = Ref((typemax(Int), typemax(Int), typemax(Int)))
    hi = Ref((0, 0, 0))
    marked = Ref(0)
    _tag_sweep!(solver, Qc) do g1, g2, g3, level
        level == TAG_MARK && (marked[] = 1)
        level == TAG_MARK || _in_regions(current, (g1, g2, g3)) || return nothing
        lo[] = min.(lo[], (g1, g2, g3))
        hi[] = max.(hi[], (g1, g2, g3))
        return nothing
    end
    # A rank with no tagged cell contributes typemax/0 sentinels that lose
    # every min/max; the mark flag rides along as a negated maximum.
    lo_g = ntuple(d -> hi[][d] == 0 ? typemax(Int) ÷ 2 : lo[][d], 3)
    hi_g = hi[]
    red = MPI.Allreduce(Int64[lo_g..., (-).(hi_g)..., -marked[]], min, dcp.comm)
    red[7] == 0 && return nothing
    glo = ntuple(d -> Int(red[d]), 3)
    ghi = ntuple(d -> -Int(red[3 + d]), 3)
    return (glo, ghi)
end

"""
    tagged_region(solver, Qc) -> Union{Nothing, BlockRegion}

The refined region the current coarse state calls for: the bounding box of
the tagged cells (those at a criterion's threshold, and those inside the
current refined regions at the hold threshold, `untag_ratio` below it),
buffered by `RegridSpec.buffer` coarse cells per side, clamped to the
nesting margin, and widened where necessary to the four-node minimum
extent. Returns `nothing` when no cell qualifies. Collapsed dimensions keep
offset 0 and extent 1. Collective over the coarse communicator (one
Allreduce of the tag bounds), so every rank returns the identical region.
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
        # Both ends clamp into the feasible interval, so a tagged set inside
        # the margin band collapses onto its inner edge rather than keeping
        # an end past it, and the widening below to the four-node minimum
        # then reopens the interval inward. Setup guarantees the feasible
        # interval holds at least four nodes.
        lo = clamp(tlo[d] - spec.buffer, margin + 1, n_global[d] - margin)
        hi = clamp(thi[d] + spec.buffer, margin + 1, n_global[d] - margin)
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
# The gather runs on the parent's communicator, so every rank owning the
# parent calls this; `owned` says whether this one also holds the fine patch
# and therefore has anything to write.
function _fill_fine_from_coarse!(solver::Solver, states, lt::LevelTransfer)
    patches = getfield(solver, :patches)
    _gather_box!(lt.box_gather, lt, states, patches)
    lt.fine_index == 0 && return states
    fine = patches[lt.fine_index]
    Qf = states[lt.fine_index]
    K = length(lt.pplans)
    n_cons = solver.equations.n_cons
    if !_device_path(Qf)
        for c in 1:n_cons
            lt.pstage[1] .= view(lt.box_gather, :, :, :, c)
            for k in 1:K
                interpolate!(lt.pstage[k+1], lt.pplans[k], lt.pstage[k])
            end
            _write_fine_shell!(Qf, c, lt.pstage[K+1], lt, fine.decomp,
                               lt.pdecomps[K+1], false)
        end
        return states
    end
    # Device patch: every component runs through the scratch's chain, whose
    # stages hold this rank's share of the components, in slices of that
    # width (the whole set at once on a one-rank tile).
    stages = fine.level_scratch.stages
    width = size(stages[1], 4)
    for lo in 1:width:n_cons
        comps = lo:min(lo + width - 1, n_cons)
        _fill_stage0_dev!(BoxFill(), view(stages[1], :, :, :, 1:length(comps)),
                          fine.level_scratch, lt, comps)
        for k in 1:K
            _interpolate_dev!(stages[k+1], lt.pplans[k], stages[k], length(comps))
        end
        for (b, c) in enumerate(comps)
            _write_fine_shell!(Qf, c, view(stages[K+1], :, :, :, b), lt,
                               fine.decomp, lt.pdecomps[K+1], false)
        end
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
    if !_device_path(Qf_new)
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
across the region overlap, and stage arrays for the new extents. The RHS
scratch is taken from the rank's pool, into which the departing patch's set
is returned first, so a region that moved without changing extent allocates
none.
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
    # The box is a tile of one: it is not rebuilt before its lifetime.
    # `checks` and `created` are derived from reduced data on every rank, so
    # this return, ahead of the collective sweep, is uniform.
    spec.checks - get(spec.created, lt.region, 0) >= spec.lifetime || return false
    newregion = tagged_region(solver, states[1])
    newregion === nothing && return false
    newregion == lt.region && return false
    delete!(spec.created, lt.region)
    spec.created[newregion] = spec.checks
    patches = getfield(solver, :patches)
    old_lc = levels[2].level_comm
    root_lc = levels[1].level_comm
    oldregion = lt.region
    active_g = ntuple(d -> solver.n_global[d] > 1, 3)
    n_cons = solver.equations.n_cons
    # The setup-time nesting rule, re-asserted here: a box reaching into
    # the margin band would read parent samples the gather never fills and
    # interpolate zeros into the fine state without any other symptom.
    _covered_by(_buffered(newregion, active_g, spec.margin), [patches[1].region]) ||
        error("regrid: the tagged region $newregion is not nested $(spec.margin) " *
              "root nodes inside the domain")
    # getfield: `h` is also a patch property name, and the property forwarding
    # refuses it on a multi-patch solver; the root spacing lives on the config.
    # Gather the surviving fine state BEFORE the patch swap, while the old
    # decomposition and block table are still live. Replicated over the old
    # region on every rank of the root communicator, since the level's next
    # owners may be ranks the old ones were not; freed when this call returns.
    Nf_old = fine_extent(oldregion, active_g)
    old_gather = _gather_tile(T, oldregion, active_g, n_cons, lt, solver, states)
    # Free the old level's decomposition, its tile group and, where the new
    # region's rank count differs, the level communicator itself, before the
    # replacements are built: MPI frees none of them until garbage collection
    # finalizes it, and a regrid discards one of each per call.
    # `tagged_region` is reduced, so every rank reaches these collective frees
    # together.
    lt.fine_index == 0 || free_communicators!(patches[lt.fine_index].decomp)
    free_tile_group!(levels[2].group)
    owners, np_new = _tile_owners([newregion], active_g, root_lc.size)
    resized = np_new != old_lc.size
    resized && free_level_comm!(old_lc)
    new_lc = resized ? split_level_comm(root_lc, np_new) : old_lc
    group = new_lc.owned ? split_tile_comm(new_lc, owners) : absent_tile_group()
    held = new_lc.owned && owners[1] == group.ranks
    fi = held ? 2 : 0
    # The rank's scratch sets, the departing patch's included: a new region of
    # the same extent takes it over instead of allocating beside it.
    ws_pool = [p.rhs_workspace for p in patches]
    newfine = held ?
        _build_fine_patch(T, newregion, active_g, getfield(solver, :h),
                          spec.n_halo, group.comm,
                          spec.deriv, spec.filt, spec.smoo,
                          solver.art.smoother, spec.interface_rhs,
                          spec.backend, ws_pool,
                          solver.equations.n_species, n_cons,
                          solver.art.species_flux === :bulk, fi, 1) : nothing
    newlt = build_level_transfer(T, newregion, active_g, spec.n_halo,
                                 [patches[1].region], [1],
                                 Union{Nothing,Decomp{T}}[patches[1].decomp],
                                 fi, lt.restriction, n_cons,
                                 getfield(solver, :subcycle),
                                 newfine === nothing ? nothing : newfine.decomp,
                                 root_lc.comm, length(owners[1]))
    _resize_level_patches!(solver, states, workspace, held ? 2 : 1)
    if held
        Qf_new = _state_like(newfine.rho, n_cons)
        patches[fi] = newfine
        states[fi] = Qf_new
        workspace.dQ[fi] = zero(Qf_new)
        workspace.du[fi] = zero(Qf_new)
    end
    levels[2] = Level{T}(1, new_lc, owners, group, held ? [1] : Int[],
                         held ? [fi] : Int[], [newlt])
    _fill_covered!(patches[1], [newregion])
    held && init_geometry!(PatchSolver(solver, newfine))
    _fill_fine_from_coarse!(solver, states, newlt)
    held && _carry_over!(states[fi], newfine.decomp, newregion, old_gather,
                         Nf_old, oldregion, active_g, n_cons)
    _rebank!(solver, states, save)
    # The old transfer's chains are on COMM_SELF, so this free is rank-local
    # and every holder of the transfer makes it.
    free_transfer_decomps!(lt)
    return true
end

# Grow or shrink this rank's patch vector (and the state vectors aligned with
# it) to `n` entries. The refined level's tiles are the trailing run of the
# vector (regridding is two-level), so the root at index 1 stays put and the
# caller overwrites every entry above it; the added entries are placeholders.
function _resize_level_patches!(solver::Solver, states, workspace, n::Int)
    patches = getfield(solver, :patches)
    length(patches) == n && return solver
    if n < length(patches)
        resize!(patches, n)
        resize!(states, n)
        resize!(workspace.dQ, n)
        resize!(workspace.du, n)
        return solver
    end
    for _ in (length(patches)+1):n
        push!(patches, patches[1])
        push!(states, states[1])
        push!(workspace.dQ, workspace.dQ[1])
        push!(workspace.du, workspace.du[1])
    end
    return solver
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
# ranks the survivors leave free (`_place_tiles`). The partition is
# recomputed only by a rebalance, when the per-rank busy time the run
# measures over the last interval is imbalanced past `RegridSpec.rebalance`
# at `RegridSpec.persist` consecutive checks (`_rebalance_due!`); the
# weights are then the measured per-tile costs (`_measured_weights`), so
# the interface factor is measured inside the job rather than calibrated. A
# surviving tile whose owner range a rebalance moved is rebuilt on its new
# owners and takes its evolved solution back by point-to-point migration
# from the old owners' blocks (`_migrate_tile!`), one tile at a time at the
# regrid cadence, with no replica of the tile on any rank.

# Levels over the lattice cells (the maximum over the buffered nodes meeting
# each cell: `TAG_MARK`, `TAG_HOLD` or 0), reduced so every rank derives the
# same set.
function _tag_tiles(solver::Solver, Qc, spec::RegridSpec)
    n_global = solver.n_global
    active = ntuple(d -> n_global[d] > 1, 3)
    a = spec.tile
    K = ntuple(d -> active[d] ? (n_global[d] - 1) ÷ a + 1 : 1, 3)
    flags = zeros(Int8, K)
    b = spec.buffer
    _tag_sweep!(solver, Qc) do g1, g2, g3, level
        g = (g1, g2, g3)
        spans = ntuple(d -> active[d] ?
                       intersect(_tile_span(g[d] - b, g[d] + b, a), 0:K[d]-1) :
                       (0:0), 3)
        for k3 in spans[3], k2 in spans[2], k1 in spans[1]
            flags[k1 + 1, k2 + 1, k3 + 1] = max(flags[k1 + 1, k2 + 1, k3 + 1], level)
        end
    end
    return MPI.Allreduce(flags, max, getfield(solver, :comm)), K
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
    Nf = fine_extent(region, active)
    dst = zeros(T, Nf..., n_cons)
    held = lt.fine_index != 0
    dp = held ? getfield(solver, :patches)[lt.fine_index].decomp : nothing
    gather_region!(dst, ntuple(d -> 1:Nf[d], 3), (0, 0, 0), (0, 0, 0),
                   held ? states[lt.fine_index] : nothing, lt.parent_comm,
                   lt.fine_blocks, dp === nothing ? (0, 0, 0) : dp.offset,
                   dp === nothing ? (0, 0, 0) : dp.n_halo_d)
    return dst
end

# Tag of the migration messages. They run on the root communicator, on
# which nothing else is in flight during a regrid, and one tile's messages
# complete before the next tile's are posted, so one tag serves every tile.
const _MIGRATE_TAG = 1500

# Test hook. With the flag on, `_regrid_tiles!` also carries each moved tile
# through the replicated gather and `_carry_over!`, and adds to the result
# the tiles it audited on this rank and the slots at which the migrated
# state differs from that reference; the MPI suite holds the count at zero.
const MIGRATION_AUDIT = Ref(false)
const MIGRATION_AUDIT_RESULT = Ref((tiles=0, mismatches=0))

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
# tile's block tables in the rank order of `comm` (the old and the new
# transfer's `fine_blocks`), so every rank derives the identical message
# list, old block ∩ new block ∩ interior per rank pair, and posts only the
# sends and receives of its own; a rank with neither posts nothing, and no
# collective runs. The interior holds one node off both boundary planes
# along every active dimension, the rule `_carry_over!` applies: the old
# planes were imposed data and the new ones are re-imposed by the next
# shell fill. `Q_old`/`d_old` are this rank's old block of the tile and
# `Q_new`/`d_new` its new one, `nothing` where it holds none. The sends
# complete before this returns, so the caller may free the old
# decomposition afterwards.
function _migrate_tile!(::Type{T}, Q_new, d_new, Q_old, d_old, Nf::NTuple{3,Int},
                        active::NTuple{3,Bool}, old_blocks::Vector{BlockRegion},
                        new_blocks::Vector{BlockRegion}, comm::MPI.Comm) where {T}
    me = MPI.Comm_rank(comm)
    np = MPI.Comm_size(comm)
    interior = ntuple(d -> active[d] ? (2:Nf[d]-1) : (1:1), 3)
    nodes(b::BlockRegion) = ntuple(d -> (b.offset[d] + 1):(b.offset[d] + b.extent[d]), 3)
    piece(a::BlockRegion, b::BlockRegion) =
        ntuple(d -> _isect(_isect(nodes(a)[d], nodes(b)[d]), interior[d]), 3)
    wanted(r) = !any(isempty, r)
    reqs = MPI.Request[]
    recvs = Tuple{NTuple{3,UnitRange{Int}},Vector{T}}[]
    if d_new !== nothing
        for a in 0:np-1
            a == me && continue
            r = piece(old_blocks[a+1], new_blocks[me+1])
            wanted(r) || continue
            buf = Vector{T}(undef, size(Q_new, 4) * prod(length.(r)))
            push!(reqs, MPI.Irecv!(buf, comm; source=a, tag=_MIGRATE_TAG))
            push!(recvs, (r, buf))
        end
    end
    sends = Vector{T}[]
    if d_old !== nothing
        for b in 0:np-1
            r = piece(old_blocks[me+1], new_blocks[b+1])
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
                 :kappa_art, :inv_J, :inv_r, :cot_over_r, :cot_over_r_gcl)
        _assign!(getfield(pnew, name), getfield(pold, name))
    end
    for name in (:Y, :D_art, :area_d, :inv_h)
        for (a, b) in zip(getfield(pnew, name), getfield(pold, name))
            _assign!(a, b)
        end
    end
    copyto!(pnew.covered, pold.covered)
    return pnew
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
    busy = _rebalance_due!(solver, spec)
    any(!=(0), flags) || return false
    lo = ntuple(d -> 1 + spec.margin, 3)
    hi = ntuple(d -> n_global[d] - spec.margin, 3)
    old_regions = [lt.region for lt in lev.transfers]
    # A cell at the tag level is wanted; one at the hold level, or younger
    # than the lifetime, is wanted where its tile exists (the hysteresis).
    # `created` is derived from the reduced flags, so every rank holds it.
    wanted = BlockRegion[]
    for k3 in 0:K[3]-1, k2 in 0:K[2]-1, k1 in 0:K[1]-1
        f = flags[k1 + 1, k2 + 1, k3 + 1]
        t = _lattice_tile((k1, k2, k3), active, spec.tile, lo, hi)
        t === nothing && continue
        exists = t in old_regions
        young = exists && spec.checks - get(spec.created, t, 0) < spec.lifetime
        (f == TAG_MARK || (exists && (f == TAG_HOLD || young))) && push!(wanted, t)
    end
    isempty(wanted) && return false
    wanted != old_regions || busy !== nothing || return false
    for r in old_regions
        r in wanted || delete!(spec.created, r)
    end
    for r in wanted
        haskey(spec.created, r) || (spec.created[r] = spec.checks)
    end
    old_lc = lev.level_comm
    root_lc = levels[1].level_comm
    # Old and new tile of a region: the transfers are held by every rank of
    # the parent's subset, the patches only by a tile's owners.
    old_of = Dict(r => i for (i, r) in enumerate(old_regions))
    new_of = Dict(r => i for (i, r) in enumerate(wanted))
    n_cons = solver.equations.n_cons
    # Ownership is stored: a surviving tile keeps its range and a fresh one
    # is placed among the ranks left free, so the tile set changing around
    # a survivor does not move it. A rebalance repartitions the whole level
    # instead, weighted by the costs the last interval measured. Either
    # way the inputs are the reduced tile flags and the owners every rank
    # holds, so every rank derives the same assignment. A tile stays in
    # place, arrays and state included, when both its region and its range
    # survive; otherwise it is rebuilt, and one whose region survived takes
    # its solution back below. The level's rank count moves when a fresh
    # tile takes ranks beyond the old level or a departure vacates its top
    # ranks; a survivor's Cartesian communicator is independent of the
    # level communicator the resize replaces, so it survives the resize.
    if busy === nothing
        owners, np_new = _place_tiles(wanted, active, root_lc.size,
                                      old_regions, lev.owners)
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
    for (i, r) in enumerate(old_regions)
        haskey(new_of, r) || _restrict_patch!(solver, states, lev.transfers[i], root_lc)
    end
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
    # On a stacked device level (rhs.jl, stacked tiles) every tile is rebuilt
    # into the new stacks, a kept tile included, which then copies its arrays
    # across below and drops its old decomposition like the rest.
    stacked = _stacked_level(spec.backend, spec.tile)
    dropped_decomps = Decomp{T}[]
    old_piece = Dict{Int,Tuple{eltype(states),Decomp{T}}}()
    for (li, ti_old) in zip(lev.patches, lev.tiles)
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
    faces = _tile_faces(wanted)
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
                                             spec.n_halo, group.comm, spec.deriv,
                                             spec.filt, spec.smoo,
                                             solver.art.smoother, spec.interface_rhs,
                                             spec.backend, ws_pool,
                                             solver.equations.n_species, n_cons,
                                             solver.art.species_flux === :bulk,
                                             1, 1, spec.tile)
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
            bcs = _fine_bcs(active, faces[ti])
            if kept[ti]
                oi = old_local[old_of[tr]]
                p = _repatch(patches[oi], idx, faces[ti], bcs)
                push!(new_states, states[oi])
                push!(new_dQ, workspace.dQ[oi])
                push!(new_du, workspace.du[oi])
            else
                p = _build_fine_patch(T, tr, active, root.h, spec.n_halo,
                                      group.comm, spec.deriv, spec.filt,
                                      spec.smoo, solver.art.smoother,
                                      spec.interface_rhs, spec.backend, ws_pool,
                                      solver.equations.n_species, n_cons,
                                      solver.art.species_flux === :bulk,
                                      idx, 1, faces[ti])
                Q = _state_like(p.rho, n_cons)
                push!(new_states, Q)
                push!(new_dQ, zero(Q))
                push!(new_du, zero(Q))
            end
            push!(new_patches, p)
        end
    end
    transfers = [build_level_transfer(
        T, tr, active, spec.n_halo, [root.region], [1],
        Union{Nothing,Decomp{T}}[root.decomp], local_of[ti],
        lev.transfers[1].restriction, n_cons, getfield(solver, :subcycle),
        local_of[ti] == 0 ? nothing : new_patches[local_of[ti] - 1].decomp,
        root_lc.comm, length(owners[ti]), faces[ti])
        for (ti, tr) in enumerate(wanted)]
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
        fine_regions = [BlockRegion(ntuple(d -> active[d] ? 3 * tr.offset[d] : 0, 3),
                                    fine_extent(tr, active)) for tr in wanted]
        records = _level_records(T, new_lc.comm, fine_regions, held, indices,
                                 [p.decomp for p in new_patches], n_cons)
        levels[2] = Level{T}(1, new_lc, owners, group, held, indices, transfers,
                             records; stacks)
    else
        levels[2] = Level{T}(1, new_lc, owners, group, held, indices, transfers;
                             stacks)
    end
    _fill_covered!(root, wanted)
    fresh = [ti for ti in eachindex(wanted) if !kept[ti]]
    for ti in fresh
        li = local_of[ti]
        li == 0 || init_geometry!(PatchSolver(solver, patches[li]))
        _fill_fine_from_coarse!(solver, states, transfers[ti])
        r = wanted[ti]
        haskey(old_of, r) || continue
        # A rebuilt tile whose region survived takes its evolved interior
        # back over the interpolated initialization, block to block from
        # its old owners. Both block tables are in the root communicator's
        # rank order, so every rank posts from the same message list; the
        # gather above is collective, so every rank reaches this together.
        Nf = fine_extent(r, active)
        Q_old, d_old = get(old_piece, ti, (nothing, nothing))
        Q_new = li == 0 ? nothing : states[li]
        d_new = li == 0 ? nothing : patches[li].decomp
        audit = li != 0 && haskey(carried, r)
        Qref = audit ? ConservedState(copy(parent(states[li]))) : nothing
        _migrate_tile!(T, Q_new, d_new, Q_old, d_old, Nf, active,
                       lev.transfers[old_of[r]].fine_blocks,
                       transfers[ti].fine_blocks, root_lc.comm)
        if audit
            _carry_over!(Qref, d_new, r, carried[r], Nf, r, active, n_cons)
            differ = count(Array(parent(Qref)) .!= Array(parent(states[li])))
            res = MIGRATION_AUDIT_RESULT[]
            MIGRATION_AUDIT_RESULT[] = (tiles=res.tiles + 1,
                                        mismatches=res.mismatches + differ)
        end
    end
    # A fresh tile's plane shared with a survivor takes the survivor's
    # evolved values rather than averaging its interpolation into them.
    if !isempty(fresh) && new_lc.owned
        is_fresh = falses(length(wanted))
        is_fresh[fresh] .= true
        _seed_planes!(solver, states, levels[2], is_fresh)
    end
    _rebank!(solver, states, save)
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
    spec.checks += 1
    # The regrid's own work is excluded from the busy time the next check
    # measures: its gathers and frees are charged to the waiting account as
    # they run, and the rest is charged here.
    t0 = time_ns()
    wait0 = solver.wall_wait
    regrid!(solver, states, workspace, save)
    spec.wall_regrid += (time_ns() - t0) / 1e9 - (solver.wall_wait - wait0)
    return nothing
end
