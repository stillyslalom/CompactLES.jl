# Tagging and regridding. The sections below through the tiled regrid move
# the one refined level of a two-level hierarchy; the last section regrids a
# deeper tiled hierarchy level by level with the same tag criteria.
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
# The new fine state is initialized by the Lagrange interpolation of the coarse
# state, the same operator as the live shell coupling and the right one here
# for the same reason: the coarse data are point samples, which the
# deconvolving `prolong!` would sharpen spuriously (see `interpolate!`).
# Surviving fine data on the overlap of the old and new regions is then copied
# across on the coincident fine lattice, holding one node off both patches'
# boundary planes (the old plane was imposed data; the new one is re-imposed by
# the next shell fill).
#
# When no cell tags, or the box did not move, the box keeps its current region:
# a feature fading below threshold leaves refinement where it last was instead
# of collapsing it, since the box has no empty form. A tiled level instead
# drops every tile past its lifetime when nothing tags or holds, and the next
# tag creates tiles again (`_regrid_tiles!`).

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

# Σ_d |δ⁴_d ρ| / ρ against the threshold pair, under the zeroth-order
# closed-edge clamp `delta4_sum!` applies where an edge reflects nothing.
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
_tag_sweep!(mark!::F, solver::Solver, Qc) where {F} =
    _tag_sweep!(mark!, solver, getfield(solver, :patches)[1], Qc,
                getfield(solver, :regrid).tags, true)

# The sweep over one patch `coarse` of a parent level with state `Qc`, into
# the scratch `tags` (the patch's padded extent), calling `mark!` in the
# parent level's global node indices. `closed` clamps the taps at the
# patch's own edges, the root's closed-edge rule; a refined tile is swept
# with its parent-fed and shared edges open, since its ghost layers there
# hold the imposed shell or the same-level neighbor's nodes at the head of a
# step, and its edges on the domain boundary closed.
function _tag_sweep!(mark!::F, solver::Solver, coarse::Patch, Qc, tags_scratch,
                     closed::Bool) where {F}
    dcp = coarse.decomp
    spec = getfield(solver, :regrid)
    exchange_state!(Qc, dcp)
    # The sensor criterion is the one that reads primitive fields (ρ, c and
    # c_p), and those arrays otherwise hold whatever the last Runge–Kutta stage
    # of the previous step left in them. Refreshing them from the state under
    # test makes the tag a function of that state alone, so a restart, and an
    # observational call between steps, leave the decision unchanged. The
    # artificial coefficients the same criterion reads are the previous step's
    # by design, as they are in `max_rate`.
    spec.sensor_threshold > 0 && primitives!(PatchSolver(solver, coarse), Qc)
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
    # A tile's face on the domain boundary carries the root's condition and
    # clamps as the root's edge does: nothing beyond it holds the tile's data.
    shut(d, side) = closed || _level_boundary_condition(coarse.bcs[d][side])
    lomin = ntuple(d -> shut(d, 1) && at_lo_edge(dcp, d) ? 1 : -1, 3)
    himax = ntuple(d -> shut(d, 2) && at_hi_edge(dcp, d) ? n[d] : n[d] + 2, 3)
    tags = on_device ? similar(parent(Qc), Int8, size(tags_scratch)) : tags_scratch
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
        copyto!(tags_scratch, tags)
        tags = tags_scratch
    end
    off = ntuple(d -> coarse.region.offset[d] + dcp.offset[d], 3)
    @inbounds for k in 1:n[3], j in 1:n[2], i in 1:n[1]
        I = CartesianIndex(i + o[1], j + o[2], k + o[3])
        tags[I] == 0 || mark!(i + off[1], j + off[2], k + off[3], tags[I])
    end
    return nothing
end

# Is global node `g` inside one of the current refined regions (parent
# node space), or one of their periodic images under `period`? The hold level
# of the hysteresis counts there only.
function _in_regions(regions::Vector{BlockRegion}, g::NTuple{3,Int},
                     period::NTuple{3,Int}=(0, 0, 0))
    for r0 in regions, σ in _images(period)
        r = _shifted(r0, σ)
        all(d -> r.offset[d] < g[d] <= r.offset[d] + r.extent[d], 1:3) && return true
    end
    return false
end

# The periods along which automatic placement crosses a periodic seam of
# level ℓ's node space: `_level_period` under `level_boundaries`, none
# with it off, every periodic face then keeping the nesting margin.
_placement_period(solver, spec::RegridSpec, ℓ::Int) =
    spec.boundaries ? _level_period(solver, ℓ) : (0, 0, 0)

# The feasible interval `[lo, hi]` of a lattice clip, opened along a
# dimension of nonzero `wrap` to the whole lattice over [1, P + 1].
_wrap_feasible(lo::NTuple{3,Int}, hi::NTuple{3,Int}, wrap::NTuple{3,Int}) =
    (ntuple(d -> wrap[d] > 0 ? 1 : lo[d], 3),
     ntuple(d -> wrap[d] > 0 ? wrap[d] + 1 : hi[d], 3))

# The lattice cells, among 0 .. K − 1, that the node interval [lo, hi]
# meets: `_tile_span` of the interval, or of its pieces on either side of the
# seam along a dimension of nonzero `wrap` (`_wrapped_parts`).
function _wrapped_cells(lo::Int, hi::Int, tile::Int, K::Int, wrap::Int)
    wrap == 0 && return intersect(_tile_span(lo, hi, tile), 0:K-1)
    cells = Int[]
    for (a, b) in _wrapped_parts(lo, hi, wrap)
        append!(cells, intersect(_tile_span(a, b, tile), 0:K-1))
    end
    return sort!(unique!(cells))
end

# The shortest arc of a period of `P` nodes holding every flagged node of
# `occupied` (one flag per node 1 .. P): the complement of the longest run of
# unflagged nodes, the run through the seam counted whole, the first such
# run taken on a tie. Returns `(lo, hi)` with lo in 1 .. P and hi possibly
# past P, the arc crossing the seam; `nothing` when no node is flagged.
function _seam_arc(occupied::AbstractVector)
    P = length(occupied)
    any(!iszero, occupied) || return nothing
    all(!iszero, occupied) && return (1, P)
    best_start, best_len = 0, -1
    for s in 1:P
        # A run starts at an unflagged node whose predecessor is flagged.
        (iszero(occupied[s]) && !iszero(occupied[mod1(s - 1, P)])) || continue
        len = 0
        while iszero(occupied[mod1(s + len, P)])
            len += 1
        end
        len > best_len && ((best_start, best_len) = (s, len))
    end
    lo = mod1(best_start + best_len, P)
    hi = best_start - 1
    hi < lo && (hi += P)
    return (lo, hi)
end

# Global bounding box of the tagged set, or `nothing`. The globally reduced
# results are whether any rank has a `TAG_MARK` and the extrema of all marked
# nodes plus `TAG_HOLD` nodes inside current refined regions. They share one
# packed `Allreduce` on the root communicator. Without a global mark, the
# function returns `nothing` and retains the current box. Otherwise, held nodes
# may enlarge the box initiated by marked nodes, but cannot define or shrink a
# box by themselves. Every rank derives the same result.
#
# Along a dimension of nonzero `wrap` (`_placement_period`) the bounds are the
# shortest arc of the period holding the counted nodes instead (`_seam_arc`),
# from their occupancy of that dimension's nodes, reduced in a second
# Allreduce, so a feature across the seam is bounded across it; `ghi` then
# lies past the period where the arc crosses the seam.
function _tag_bounds(solver::Solver, Qc, wrap::NTuple{3,Int}=(0, 0, 0))
    dcp = getfield(solver, :patches)[1].decomp
    current = [lt.region for lt in getfield(solver, :levels)[2].transfers]
    lo = Ref((typemax(Int), typemax(Int), typemax(Int)))
    hi = Ref((0, 0, 0))
    marked = Ref(0)
    seams = [d for d in 1:3 if wrap[d] > 0]
    occupied = [zeros(Int8, wrap[d]) for d in seams]
    _tag_sweep!(solver, Qc) do g1, g2, g3, level
        level == TAG_MARK && (marked[] = 1)
        level == TAG_MARK || _in_regions(current, (g1, g2, g3), wrap) || return nothing
        lo[] = min.(lo[], (g1, g2, g3))
        hi[] = max.(hi[], (g1, g2, g3))
        for (m, d) in enumerate(seams)
            occupied[m][(g1, g2, g3)[d]] = one(Int8)
        end
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
    isempty(seams) && return (glo, ghi)
    occ = MPI.Allreduce(reduce(vcat, occupied), max, dcp.comm)
    at = 0
    for (m, d) in enumerate(seams)
        a, b = _seam_arc(view(occ, at .+ (1:wrap[d])))
        glo = Base.setindex(glo, a, d)
        ghi = Base.setindex(ghi, b, d)
        at += wrap[d]
    end
    return (glo, ghi)
end

"""
    tagged_region(solver, Qc) -> Union{Nothing, BlockRegion}

The refined region the current coarse state calls for: the bounding box of
the tagged cells (those at a criterion's threshold, and those inside the
current refined regions at the hold threshold, `untag_ratio` below it),
buffered by `RegridSpec.buffer` coarse cells per side, clamped to the
nesting margin, and widened where necessary to the four-node minimum
extent. Under `level_boundaries`, the default, the clamp stops instead at a
domain face whose condition a level carries, and a region on such a face is
widened until its box spans the interpolation order; along a periodic
dimension the bounds are the shortest arc of the period holding the tagged
cells, and the region may cross the seam, spanning at most the period less
the margin at either end. Returns `nothing` when no cell qualifies.
Collapsed dimensions keep offset 0 and extent 1. Collective over the coarse
communicator (one Allreduce of the tag bounds, and one of their occupancy
along a periodic dimension under `level_boundaries`), so every rank returns
the identical region.
"""
function tagged_region(solver::Solver, Qc)
    spec = getfield(solver, :regrid)
    wrap = _placement_period(solver, spec, 0)
    bounds = _tag_bounds(solver, Qc, wrap)
    bounds === nothing && return nothing
    tlo, thi = bounds
    margin = spec.margin
    n_global = solver.n_global
    active = ntuple(d -> n_global[d] > 1, 3)
    reach = _placement_faces(solver, spec)
    bare = _unbuffered_faces(solver, reach)
    lo_feasible, hi_feasible = _feasible_nodes(n_global, active, margin, reach)
    # Four nodes, or more under the positivity limiter, whose face form needs
    # a fine line of a length (`_limiter_least_box`).
    least = _limiter_least_extent(solver)
    # Tagged nodes in the margin band of a face the level does not reach
    # cannot be refined; say so once rather than leave a feature at a
    # boundary silently coarse. The bounds are reduced, so every rank agrees.
    any(d -> active[d] && wrap[d] == 0 &&
             (tlo[d] < lo_feasible[d] || thi[d] > hi_feasible[d]), 1:3) &&
        _warn_margin_band(solver, margin, 1)
    offext = ntuple(3) do d
        active[d] || return (0, 1)
        if wrap[d] > 0
            # The arc, buffered, widened to the four-node minimum and capped
            # at the period less the margin at either end, about its centre;
            # the offset is brought into [0, P).
            P = wrap[d]
            lo = tlo[d] - spec.buffer
            hi = thi[d] + spec.buffer
            n = clamp(hi - lo + 1, least, P - 2 * margin)
            lo = fld(lo + hi - n + 1, 2)
            return (mod(lo - 1, P), n)
        end
        a, b = lo_feasible[d], hi_feasible[d]
        # Both ends clamp into the feasible interval, so a tagged set inside
        # the margin band collapses onto its inner edge rather than keeping
        # an end past it, and the widening below to the minimum extent then
        # reopens the interval inward. Setup guarantees the feasible
        # interval holds at least four nodes; a region on a domain face also
        # widens until its box spans the interpolation order.
        lo = clamp(tlo[d] - spec.buffer, a, b)
        hi = clamp(thi[d] + spec.buffer, a, b)
        # An end inside the margin band of a face the region may reach
        # moves onto the face, since a parent-fed face there would have no
        # room for its box; the widening repeats the move.
        while true
            reach[d][1] && lo <= margin && (lo = 1)
            reach[d][2] && hi > n_global[d] - margin && (hi = n_global[d])
            need = max(least, _placement_extent(spec.interpolation_order,
                                                bare[d][1] && lo == 1,
                                                bare[d][2] && hi == n_global[d]))
            (hi - lo + 1 >= need || hi - lo + 1 >= b - a + 1) && break
            hi < b ? (hi += 1) : (lo -= 1)
        end
        (lo - 1, hi - lo + 1)
    end
    return BlockRegion(ntuple(d -> offext[d][1], 3),
                       ntuple(d -> offext[d][2], 3))
end

# The domain faces automatic placement may reach, per dimension and side:
# under `level_boundaries` those whose root condition a level carries, the
# faces `_region_boundaries` gives, a fold (a symmetry plane, the axis)
# included; none otherwise, every face then keeping the nesting margin.
function _placement_faces(solver, spec::RegridSpec)
    spec.boundaries || return _NO_BOUNDARY
    root = getfield(solver, :patches)[1]
    active = ntuple(d -> solver.n_global[d] > 1, 3)
    return _level_boundary_eligible(root.bcs, active, root.decomp.periodic)
end

# The faces among `reach` on which a placed region's box takes no buffer: a
# fold keeps its buffer, the parity mirror of the parent's nodes.
function _unbuffered_faces(solver, reach)
    bcs = getfield(solver, :patches)[1].bcs
    return ntuple(d -> ntuple(s -> reach[d][s] && !_level_fold_condition(bcs[d][s]), 2),
                  3)
end

# The feasible node interval of a level's placement along each dimension of
# a node space of `N` nodes: the margin inside every face but those `reach`
# marks, which a region may touch, at the ends of `span` (the node space's
# own at a fold below the first refined level, `_level_span`).
_feasible_nodes(N::NTuple{3,Int}, active::NTuple{3,Bool}, margin::Int, reach,
                span=ntuple(d -> 1:N[d], 3)) =
    (ntuple(d -> !active[d] ? 1 : reach[d][1] ? first(span[d]) : 1 + margin, 3),
     ntuple(d -> !active[d] ? N[d] : reach[d][2] ? last(span[d]) : N[d] - margin, 3))

# The least extent, in parent nodes, of a placed region along one dimension:
# four nodes, and with a face on the domain boundary (`lo`, `hi`), where its
# box takes no buffer beyond the region, enough nodes for the box to span the
# interpolation order `order`.
_placement_extent(order::Int, lo::Bool, hi::Bool) =
    max(4, order - (lo ? 0 : LEVEL_BUFFER) - (hi ? 0 : LEVEL_BUFFER))

# The faces among `reach` that the tiles of a lattice of edge `tile` over a
# node space of `N` nodes can reach. The cell at a face must span the
# placement extent, and its neighbor, whose face toward the boundary is
# parent-fed, must stay the margin inside it: at the low face the neighbor's
# face lies `tile` nodes in, and at the high face the last cell holds the
# remainder `r` of the lattice, `tile` nodes when the lattice ends on the
# face. A face the lattice cannot reach keeps the margin. `bare` marks the
# faces among `reach` whose box takes no buffer (all of them by default; a
# fold keeps its buffer, `_unbuffered_faces`).
function _lattice_reach(reach, N::NTuple{3,Int}, active::NTuple{3,Bool},
                        tile::NTuple{3,Int}, margin::Int, order::Int, bare=reach)
    return ntuple(3) do d
        active[d] || return (false, false)
        r = (N[d] - 1) % tile[d]
        edge(s) = tile[d] >= margin &&
                  tile[d] + 1 >= _placement_extent(order, s == 1 && bare[d][1],
                                                   s == 2 && bare[d][2])
        last = r == 0 ? edge(2) :
               r >= margin && r + 1 >= _placement_extent(order, false, bare[d][2])
        (reach[d][1] && edge(1), reach[d][2] && last)
    end
end

# The faces a level's lattice, over its parent's node space of `N` nodes on
# the lattice coincident with the root's, reaches under the solver's
# placement rules.
function _placed_reach(solver, spec::RegridSpec, N::NTuple{3,Int},
                       active::NTuple{3,Bool})
    faces = _placement_faces(solver, spec)
    return _lattice_reach(faces, N, active, spec.tile, spec.margin,
                          spec.interpolation_order, _unbuffered_faces(solver, faces))
end

# The warning of a tagged node in the margin band of a face no level reaches,
# once per session, from rank 0; the callers' decisions are reduced, so the
# ranks agree.
function _warn_margin_band(solver, margin::Int, ℓ::Int)
    MPI.Comm_rank(getfield(solver, :comm)) == 0 || return nothing
    nodes = ℓ == 1 ? "root" : "level-$(ℓ - 1)"
    @warn "AMR: tagged cells lie within $margin $nodes nodes of a domain boundary " *
          "that a refined level does not reach; they stay at the parent " *
          "resolution. A level reaches a SlipWallBC, NoSlipWallBC, NSCBCOutflowBC " *
          "or NSCBCInflowBC face and crosses a periodic seam, and reaches a " *
          "SymmetryPlaneBC or the AxisBC of an r-z run; any other face keeps " *
          "the margin, as every face does under level_boundaries = false. A " *
          "tiled level reaches a face " *
          "only where the tile next to it stays the margin inside the domain." maxlog = 1
    return nothing
end

# Whole-patch initialization of fresh fine states: interpolate the parent
# state over each selected tile's buffered box and write every slot, interior
# and shell alike. `tiles` selects tiles of `lev` (all of them for
# `nothing`). The box exchange is entered by every rank owning the parent,
# and each owner of a selected tile then fills its block, every component.
function _fill_tiles_from_parent!(solver::Solver, states, lev::Level, tiles=nothing)
    _exchange_boxes!(solver, states, lev, lt -> lt.box_gather, true, tiles)
    for (t, lt) in enumerate(lev.transfers)
        (tiles === nothing || tiles[t]) && lt.fine_index != 0 &&
            _fill_tile_from_box!(solver, states, lt)
    end
    return states
end

function _fill_tile_from_box!(solver::Solver, states, lt::LevelTransfer)
    patches = getfield(solver, :patches)
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
# patches' boundary planes, except at a face `keep` marks, one on the domain
# boundary for both regions: there the plane is either patch's own solution
# and is copied, with the fine node a folded face adds beyond the coincident
# lattice (`folded_old`, `folded_new`, `_fold_lead`). Distributed: the old
# fine state gathers replicated over the old region (one transient
# region-sized array per rank, at the regrid cadence only), and each rank
# writes the overlap slots of its own new block. `old_gather` is that
# replica, indexed by old patch node.
function _carry_over!(Qf_new, dnew::Decomp, rnew::BlockRegion,
                      old_gather, Nf_old::NTuple{3,Int}, rold::BlockRegion,
                      active::NTuple{3,Bool}, n_cons::Int,
                      keep::NTuple{3,NTuple{2,Bool}}=_NO_BOUNDARY,
                      folded_old::NTuple{3,NTuple{2,Bool}}=_NO_BOUNDARY,
                      folded_new::NTuple{3,NTuple{2,Bool}}=_NO_BOUNDARY)
    # Parent node g maps to fine node 3(g − offset) − 2 + lead on either
    # patch, the lead being the node a folded low face adds, so the
    # old-to-new fine index shift is 3(offset_old − offset_new) plus the
    # difference of the leads.
    lead_old = ntuple(d -> _fold_lead(folded_old, d, 1), 3)
    shift = ntuple(d -> active[d] ? 3 * (rold.offset[d] - rnew.offset[d]) +
                                    _fold_lead(folded_new, d, 1) - lead_old[d] : 0, 3)
    Nf_new = fine_extent(rnew, active, folded_new)
    rng = ntuple(3) do d
        active[d] || return 1:1
        glo = max(rold.offset[d], rnew.offset[d]) + 1
        ghi = min(rold.offset[d] + rold.extent[d], rnew.offset[d] + rnew.extent[d])
        first_node = keep[d][1] ? 1 : 2
        last_off = keep[d][2] ? 0 : 1
        # A kept face copies its plane and, on a fold, the node beyond it.
        f_lo = keep[d][1] ? 1 : max(3 * (glo - rold.offset[d]) - 2 + lead_old[d], 2)
        f_hi = keep[d][2] ? Nf_old[d] :
               min(3 * (ghi - rold.offset[d]) - 2 + lead_old[d], Nf_old[d] - 1)
        # Window in NEW patch global nodes, then this rank's slice of it.
        w = max(f_lo + shift[d], first_node):min(f_hi + shift[d], Nf_new[d] - last_off)
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
Internal layout-rebuild implementation for [`regrid!`](@ref).

Retag the coarse level and, when the tagged region moved, rebuild the level-1
patch over it: a new [`Patch`](@ref) and [`LevelTransfer`](@ref) from the
schemes retained on [`RegridSpec`](@ref), a new fine state initialized by
Lagrange interpolation of the coarse state with surviving fine data copied
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
function _regrid_impl!(solver::Solver{T}, states::Vector{<:ConservedState},
                       workspace::Workspace, save; bootstrap::Bool=false) where {T}
    spec = getfield(solver, :regrid)
    levels = getfield(solver, :levels)
    _tiled(spec.tile) && length(levels) > 2 &&
        return _regrid_hierarchy!(solver, states, workspace, save; bootstrap)
    _tiled(spec.tile) && return _regrid_tiles!(solver, states, workspace, save)
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
    bnd = only(_region_boundaries(solver, [newregion], 1))
    folded = only(_region_folds(solver, [bnd]))
    period = _level_period(solver, 0)
    _covered_by(_buffered(newregion, active_g, spec.margin, bnd), [patches[1].region],
                period) ||
        error("regrid: the tagged region $newregion is not nested $(spec.margin) " *
              "root nodes inside the domain")
    # getfield: `h` is also a patch property name, and the property forwarding
    # refuses it on a multi-patch solver; the root spacing lives on the config.
    # Gather the surviving fine state BEFORE the patch swap, while the old
    # decomposition and block table are still live. Replicated over the old
    # region on every rank of the root communicator, since the level's next
    # owners may be ranks the old ones were not; freed when this call returns.
    Nf_old = fine_extent(oldregion, active_g, lt.folded)
    old_gather = _gather_tile(T, oldregion, active_g, n_cons, lt, solver, states)
    # Free the old level's decomposition, its tile group and, where the new
    # region's rank count differs, the level communicator itself, before the
    # replacements are built: MPI frees none of them until garbage collection
    # finalizes it, and a regrid discards one of each per call.
    # `tagged_region` is reduced, so every rank reaches these collective frees
    # together.
    lt.fine_index == 0 || free_communicators!(patches[lt.fine_index].decomp)
    free_tile_group!(levels[2].group)
    owners, np_new = _tile_owners([newregion], active_g, root_lc.size; whole=false)
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
                          _grad_Q_columns(solver), fi, 1;
                          interface_divergence=spec.interface_divergence,
                          ghost_viscous=_ghost_viscous(solver),
                          ring=_ring_detector(solver),
                          boundary=bnd, bcs=patches[1].bcs,
                          root_folds=patches[1].folds) : nothing
    newlt = build_level_transfer(T, newregion, active_g, spec.n_halo,
                                 [patches[1].region], [1],
                                 fi, lt.restriction, n_cons,
                                 getfield(solver, :subcycle),
                                 newfine === nothing ? nothing : newfine.decomp,
                                 root_lc.comm, length(owners[1]);
                                 interpolation_order=spec.interpolation_order,
                                 gradient_deriv=_ghost_viscous(solver) ?
                                                spec.deriv : nothing,
                                 parent_h=getfield(solver, :h), boundary=bnd, folded,
                                 period)
    coupling = build_level_coupling(T, root_lc.comm, [newlt], [patches[1].region],
                                    [patches[1].decomp],
                                    [newfine === nothing ? nothing : newfine.decomp])
    _resize_level_patches!(solver, states, workspace, held ? 2 : 1)
    if held
        Qf_new = _state_like(newfine.rho, n_cons)
        patches[fi] = newfine
        states[fi] = Qf_new
        workspace.dQ[fi] = zero(Qf_new)
        workspace.du[fi] = zero(Qf_new)
    end
    levels[2] = Level{T}(1, new_lc, owners, group, held ? [1] : Int[],
                         held ? [fi] : Int[], [newlt]; coupling)
    _fill_covered!(patches[1], [newregion], period)
    held && init_geometry!(PatchSolver(solver, newfine))
    _fill_tiles_from_parent!(solver, states, levels[2])
    held && _carry_over!(states[fi], newfine.decomp, newregion, old_gather,
                         Nf_old, _nearest_image(oldregion, newregion, period),
                         active_g, n_cons, _shared_boundary(lt.boundary, bnd), lt.folded,
                         folded)
    # The old transfer's chains are on COMM_SELF, so this free is rank-local
    # and every holder of the transfer makes it.
    free_transfer_decomps!(lt)
    return true
end

"""
    regrid!(solver, states, workspace, save) -> Bool

Retag and, when needed, rebuild the refined layout. Every call also refreshes
the artificial coefficients on the resulting hierarchy before the next root
CFL estimate; the refresh cadence is therefore every regrid check, independent
of whether tagging or load balancing changed the layout. The savepoint is
rebuilt only after an actual layout change. Returns whether it changed.
"""
function regrid!(solver::Solver, states::Vector{<:ConservedState},
                 workspace::Workspace, save)
    changed = _regrid_impl!(solver, states, workspace, save)
    # The coefficient refresh is a regrid-check cadence, not a layout-change
    # side effect: otherwise a timing-dependent rebalance decision changes the
    # root CFL sequence.  A savepoint still follows only a changed layout.
    _regrid_prime!(solver, states, workspace, save, changed)
    _close_lattice_check!(solver, getfield(solver, :regrid), 0.0)
    return changed
end

# Grow or shrink this rank's patch vector (and the state vectors aligned with
# it) to `n` entries. The refined level's tiles are the trailing run of the
# vector (a two-level hierarchy), so the root at index 1 stays put and the
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
        save.art = _art_snapshot(solver)
        save.t = solver.t
        save.step = solver.step
    end
    return save
end

# A regridded fine patch begins with blank artificial properties.  Rebuild them
# before either the next root CFL estimate or a retry bank can read the new
# layout; this is shared by direct `regrid!` and run!'s cadence hook. A check
# that kept the layout recomputes the coefficients alone
# (`_prime_level_coefficients!`): the rest of a right-hand side writes only
# what the next step's first evaluation overwrites before reading it.
function _regrid_prime!(solver, states, workspace, save, changed)
    if solver.art.enabled
        solver.tstage = solver.t
        _presync!(solver, states)
        changed ? _prime_rhs!(solver, states, workspace) :
                  _prime_level_coefficients!(solver, states)
    end
    changed && _rebank!(solver, states, save)
    return nothing
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
# ranks the survivors leave free or on the least loaded group
# (`_place_tiles`). The partition is recomputed by volume when the stored
# placement leaves the most loaded rank past `PLACEMENT_GAIN` times what
# a fresh partition would give it (`_balanced_owners`), and by a
# rebalance, when the per-rank busy time the run measures over the last
# interval is imbalanced past `RegridSpec.rebalance` at `RegridSpec.persist`
# consecutive checks (`_rebalance_due!`); the weights are then the measured
# per-tile costs (`_measured_weights`), so the interface factor is measured
# inside the job rather than calibrated, and the volume rule is not applied,
# since it would undo them. A surviving tile whose owner range a repartition
# moved is rebuilt on its new owners and takes its evolved solution back by
# point-to-point migration from the old owners' blocks (`_migrate_tile!`),
# one tile at a time at the regrid cadence, with no replica of the tile on
# any rank.

# Levels over the cells of each lattice of `chain` (the maximum over the
# buffered nodes meeting each cell: `TAG_MARK`, `TAG_HOLD` or 0), reduced so
# every rank derives the same sets, and whether a node at the tag level lies
# outside the feasible interval `[lo, hi]`, reduced with them. Along a
# dimension of nonzero `wrap` the lattice runs over [1, P + 1], its last
# cell ending on the seam, and a buffered node near the seam meets the cells
# on both sides of it (`_wrapped_parts`); the same number of cells covers
# it, (N − 1) ÷ tile + 1. Returns the flags and the cell counts per lattice.
function _tag_tiles(solver::Solver, Qc, spec::RegridSpec, lo::NTuple{3,Int},
                    hi::NTuple{3,Int}, wrap::NTuple{3,Int}=(0, 0, 0),
                    chain::Vector{NTuple{3,Int}}=[spec.tile])
    n_global = solver.n_global
    active = ntuple(d -> n_global[d] > 1, 3)
    Ks = [ntuple(d -> active[d] ? (n_global[d] - 1) ÷ a[d] + 1 : 1, 3) for a in chain]
    sizes = [prod(K) for K in Ks]
    starts = cumsum([0; sizes[1:end-1]])
    # Every lattice's flags and the band flag share one buffer and one
    # reduction.
    buf = zeros(Int8, sum(sizes) + 1)
    flagsets = [reshape(view(buf, s .+ (1:n)), K) for (s, n, K) in zip(starts, sizes, Ks)]
    b = spec.buffer
    _tag_sweep!(solver, Qc) do g1, g2, g3, level
        g = (g1, g2, g3)
        level == TAG_MARK && any(d -> active[d] && !(lo[d] <= g[d] <= hi[d]), 1:3) &&
            (buf[end] = one(Int8))
        for (a, K, flags) in zip(chain, Ks, flagsets)
            spans = ntuple(d -> active[d] ?
                                _wrapped_cells(g[d] - b, g[d] + b, a[d], K[d], wrap[d]) :
                                (0:0), 3)
            for k3 in spans[3], k2 in spans[2], k1 in spans[1]
                flags[k1 + 1, k2 + 1, k3 + 1] = max(flags[k1 + 1, k2 + 1, k3 + 1], level)
            end
        end
    end
    red = MPI.Allreduce(buf, max, getfield(solver, :comm))
    return [reshape(red[s .+ (1:n)], K) for (s, n, K) in zip(starts, sizes, Ks)], Ks,
           red[end] > 0
end

# --- Choosing the lattice ---------------------------------------------------
#
# A tiled level's tile is held whole by one rank (`_tile_owners`), so a level
# with fewer tiles than its parent has ranks leaves ranks idle on it, and a
# finer lattice spreads it wider at the price of more tiles: more interface
# faces, whose ghost-flux passes and exchanges grow with the tile count, and
# more tiles to rebuild at each regrid check. The regrid tiles each level on
# the lattice of a chain that a cost model predicts to give the shortest
# step (`_choose_lattice`): the configured edges first, then each lattice
# halving the largest edge of the one before that stays at least
# `MIN_TILE_EDGE` parent nodes and keeps every domain face the configured
# lattice reaches (`_edge_chain`). Every lattice of the chain is tagged in
# the same sweep and reduced together, and the wanted tiles of each are
# counted by the one rule (`_lattice_wanted`). The model (`LatticeCosts`)
# prices each lattice's wanted tiles under the placement the regrid would
# make, at the most loaded rank, and adds the rebuild cost of its tiles over
# the regrid interval. The levels below are priced with it, on the ranks the
# lattice gives the level, since a level of whole tiles spans as many ranks
# as it has tiles and its children are placed within them; the root and the
# levels above enter as the share of the step they take, so a level that
# does little of the work, and leaves its children their ranks, never gains
# enough to change. A level keeps its lattice unless another is predicted to
# make the step `PLACEMENT_GAIN` times shorter, the change's own rebuild of
# every tile included. A change of lattice replaces
# every tile: each new tile is filled from its parent as a fresh tile is,
# then takes the evolved solution of the old tiles it overlaps
# (`_carry_tiles!`), every node an old tile held as its own solution. On one
# rank the chain is the configured lattice alone.
#
# The model's coefficients are measured on the run itself: per-tile timers
# around the tile phases the step already runs (`_charge_tile!`), timers on
# the level exchanges (`_charge_exchange!`) and the wall of each regrid
# check, gathered on rank 0 at the check, fitted there and broadcast, so
# every rank takes the same lattice (`_harvest_lattice_costs!`). Until an
# interval has been measured the configured lattice holds. Because they are
# timings, the choice of lattice, and with it the layout and the solution at
# the level of the truncation error, can differ from one run to the next;
# `FIXED_LATTICE_COSTS` fixes the coefficients for a reproducible layout.

# The least lattice edge, in parent nodes, the subdivision reaches: the
# edge at which a lattice still reaches a domain face (`_lattice_reach`
# needs an edge of at least the nesting margin, four nodes) and a tile
# carries 13 fine nodes, past the filter's nine-node minimum.
const MIN_TILE_EDGE = 4

# The chain of lattices a level of `np` ranks may take, coarsest first, over
# a parent node space of `N` nodes: `tile`, then each lattice halving the
# largest active edge that halves to at least `least` parent nodes without
# changing the faces `_lattice_reach` gives along that dimension.
function _edge_chain(tile::NTuple{3,Int}, active::NTuple{3,Bool}, N::NTuple{3,Int},
                     faces, bare, margin::Int, order::Int, least::Int, np::Int)
    chain = [tile]
    np > 1 || return chain
    reach = _lattice_reach(faces, N, active, tile, margin, order, bare)
    e = tile
    while true
        best = 0
        for d in 1:3
            active[d] && e[d] ÷ 2 >= least || continue
            cand = Base.setindex(e, e[d] ÷ 2, d)
            _lattice_reach(faces, N, active, cand, margin, order, bare)[d] == reach[d] ||
                continue
            (best == 0 || e[d] > e[best]) && (best = d)
        end
        best == 0 && return chain
        e = Base.setindex(e, e[best] ÷ 2, best)
        push!(chain, e)
    end
end

# Test and bench toggle: off, every level keeps the configured lattice, so
# a decomposed run tiles its levels as a serial one does and the two can be
# compared to round-off.
const SUBDIVIDE_LATTICE = Ref(true)

# The chain of level ℓ's lattices under the solver's placement rules, over
# its parent's node space of `N` nodes (the lattice coincident with the
# root's), for a level of `np` ranks.
function _level_chain(solver, spec::RegridSpec, N::NTuple{3,Int},
                      active::NTuple{3,Bool}, np::Int)
    SUBDIVIDE_LATTICE[] || return [spec.tile]
    faces = _placement_faces(solver, spec)
    least = max(MIN_TILE_EDGE, _limiter_least_extent(solver) - 1)
    return _edge_chain(spec.tile, active, N, faces, _unbuffered_faces(solver, faces),
                       spec.margin, spec.interpolation_order, least, np)
end

# Whether two regions share more than a plane, over the periodic images of
# `b` under `period`: at least two nodes in common along every active
# dimension.
function _overlaps(a::BlockRegion, b::BlockRegion, active::NTuple{3,Bool},
                   period::NTuple{3,Int})
    any(_images(period)) do σ
        c = _shifted(b, σ)
        all(d -> !active[d] ||
                 min(a.offset[d] + a.extent[d], c.offset[d] + c.extent[d]) -
                 max(a.offset[d], c.offset[d]) >= 2, 1:3)
    end
end

# The creation check a tile of a new lattice inherits: the latest among the
# old tiles it overlaps, so a subdivided tile keeps its lifetime, or
# `checks` where it overlaps none.
function _inherited_check(r::BlockRegion, old_regions::Vector{BlockRegion},
                          created::Dict{BlockRegion,Int}, checks::Int,
                          active::NTuple{3,Bool}, period::NTuple{3,Int})
    over = [get(created, o, 0) for o in old_regions if _overlaps(r, o, active, period)]
    return isempty(over) ? checks : maximum(over)
end

# Whether tile `t` of a lattice is wanted at cell level `f`: at the tag
# level, or at the hold level or younger than the lifetime where the
# current layout already refines it. On the current lattice that is the
# tile itself; on another, an old tile overlapping it, which is how the
# hysteresis carries across a change of lattice.
function _tile_wanted(t::BlockRegion, f::Int8, spec::RegridSpec,
                      old_regions::Vector{BlockRegion}, created::Dict{BlockRegion,Int},
                      current::Bool, active::NTuple{3,Bool}, period::NTuple{3,Int})
    f == TAG_MARK && return true
    young(r) = spec.checks - get(created, r, 0) < spec.lifetime
    if current
        t in old_regions || return false
        return f == TAG_HOLD || young(t)
    end
    over = [o for o in old_regions if _overlaps(t, o, active, period)]
    isempty(over) && return false
    return f == TAG_HOLD || any(young, over)
end

# The wanted tiles of level 1 on lattice `edges`, from its reduced cell
# flags, in lattice order.
function _lattice_wanted(flags, K::NTuple{3,Int}, edges::NTuple{3,Int},
                         active::NTuple{3,Bool}, lo::NTuple{3,Int}, hi::NTuple{3,Int},
                         spec::RegridSpec, old_regions::Vector{BlockRegion},
                         current::Bool, period::NTuple{3,Int})
    wanted = BlockRegion[]
    for k3 in 0:K[3]-1, k2 in 0:K[2]-1, k1 in 0:K[1]-1
        t = _lattice_tile((k1, k2, k3), active, edges, lo, hi)
        t === nothing && continue
        _tile_wanted(t, flags[k1 + 1, k2 + 1, k3 + 1], spec, old_regions, spec.created,
                     current, active, period) && push!(wanted, t)
    end
    return wanted
end

# Test and bench toggle: `(a, b, k, m, c)` fixes the coefficients of the
# lattice cost model (`LatticeCosts`) and turns its timers off, so the
# lattice a level takes depends on the layout alone and a run repeats its
# layout; `nothing`, the default, measures them. `(1, 0, 0, 0, 0)` prices a
# lattice by the fine nodes of its most loaded rank.
const FIXED_LATTICE_COSTS = Ref{Union{Nothing,NTuple{5,Float64}}}(nothing)

# The coefficients a device level takes. A device phase returns as soon as
# its kernels are launched, so the tile timers would read the launch and not
# the work, and synchronizing per tile would stall the stream the stacked
# tiles share; the device path prices a lattice by the fine nodes of its
# most loaded rank instead, its tiles launching per stack rather than per
# tile.
const DEVICE_LATTICE_COSTS = (1.0, 0.0, 0.0, 0.0, 0.0)

# The weight the statistics of earlier intervals keep at each check: the
# model follows a change of regime within ten or so checks, the noise of one
# interval does not by itself change a lattice, and the tile volumes of an
# earlier lattice keep the per-tile cost apart from the per-node one for a
# while after a change.
const LATTICE_MEMORY = 0.85

# The least spread of the tile volumes, as the relative determinant of the
# volume-and-count normal equations (about the squared coefficient of
# variation), at which the per-tile cost `k` is fitted apart from the
# per-node cost `a`. Tiles of one lattice share one volume but at the domain
# edges; below this spread the fit keeps the last `k` it told apart (0
# before any), so the model does not forget the per-tile cost and flip back
# to a lattice it has just left.
const LATTICE_SPREAD = 0.01

# Whether the model measures its coefficients on this run: a decomposed
# tiled run on host storage, the subdivision on and no fixed coefficients.
# Setup constants and toggles, the same on every rank.
_lattice_measured(solver, spec::RegridSpec) =
    SUBDIVIDE_LATTICE[] && FIXED_LATTICE_COSTS[] === nothing && _tiled(spec.tile) &&
    spec.backend isa CPUBackend && MPI.Comm_size(getfield(solver, :comm)) > 1

# The steps level ℓ takes per root step.
_level_substeps(solver, ℓ::Int) = getfield(solver, :subcycle) ? 3.0^ℓ : 1.0

# The timers zeroed and sized to the current patch vector and levels, at the
# end of a regrid check, so the next check reads one interval's work.
function _reset_lattice_timers!(solver, spec::RegridSpec)
    costs = spec.costs
    costs.timing = _lattice_measured(solver, spec)
    costs.compile_mark = _counted_compile_ns(costs)
    n = length(getfield(solver, :patches))
    for v in (costs.tile_time, costs.ghost_time)
        resize!(v, n)
        fill!(v, zero(UInt64))
    end
    resize!(costs.exchange, length(getfield(solver, :levels)))
    fill!(costs.exchange, zero(UInt64))
    costs.step_mark = solver.step
    return costs
end

# The close of a regrid check: its wall, the seconds it spent compiling and
# the refined tiles it leaves, which the next check's harvest fits the
# rebuild cost from, and the timers reset.
function _close_lattice_check!(solver, spec::RegridSpec, wall::Float64,
                               compiled::Float64=0.0)
    costs = spec.costs
    costs.check_wall = wall
    costs.check_compile = compiled
    levels = getfield(solver, :levels)
    costs.check_tiles = sum((length(levels[ℓ].transfers) for ℓ in 2:length(levels));
                            init=0)
    _reset_lattice_timers!(solver, spec)
    return costs
end

# Per rank of `np`, the messages a layout's tiles send in the level's
# exchanges: one coupling with the parent per tile, and one per face a tile
# shares with a tile held by another rank. `owners` are whole-tile ranges.
function _rank_messages(regions::Vector{BlockRegion}, owners::Vector{UnitRange{Int}},
                        faces, np::Int)
    msgs = zeros(Int, np)
    for t in eachindex(regions)
        r = first(owners[t])
        msgs[r + 1] += 1
        for d in 1:3, side in 1:2
            q = faces[t][d][side]
            q != 0 && first(owners[q]) != r && (msgs[r + 1] += 1)
        end
    end
    return msgs
end

# The predicted seconds per root step of a level whose tiles `regions` are
# held as `owners` by `np` ranks, at the most loaded rank, `s` steps per root
# step. `boundaries` and `folds` are the tiles' domain faces and folded faces,
# `period` the parent node space's.
function _layout_cost(costs::LatticeCosts, regions::Vector{BlockRegion},
                      boundaries, folds, owners::Vector{UnitRange{Int}},
                      active::NTuple{3,Bool}, period::NTuple{3,Int}, s::Float64, np::Int)
    isempty(regions) && return 0.0
    faces = _tile_faces(regions, period)
    msgs = _rank_messages(regions, owners, faces, np)
    load = costs.m * s .* msgs
    for t in eachindex(regions)
        V = Float64(prod(fine_extent(regions[t], active, folds[t])))
        g = count(_interface_dims(active, boundaries[t]))
        load[first(owners[t]) + 1] += s * (costs.a * V + costs.b * V * g + costs.k)
    end
    return maximum(load)
end

# The placement the regrid would give a level of `regions` over `np` ranks
# from scratch: whole tiles by volume (`_fresh_partition`).
_predicted_owners(regions::Vector{BlockRegion}, active::NTuple{3,Bool}, np::Int) =
    first(length(regions) > np ? _knapsack_owners(regions, active, np) :
                                 _tile_owners(regions, active, np))

# The predicted seconds per root step of refined level ℓ tiled as `regions`
# over `np` ranks, its rebuilds at the regrid checks included.
function _lattice_level_cost(solver, spec::RegridSpec, ℓ::Int,
                             regions::Vector{BlockRegion}, np::Int)
    isempty(regions) && return 0.0
    active = ntuple(d -> solver.n_global[d] > 1, 3)
    boundaries = _region_boundaries(solver, regions, ℓ)
    folds = _region_folds(solver, boundaries)
    costs = spec.costs
    return _layout_cost(costs, regions, boundaries, folds,
                        _predicted_owners(regions, active, np), active,
                        _level_period(solver, ℓ - 1), _level_substeps(solver, ℓ), np) +
           costs.c * length(regions) / spec.interval
end

# The ranks a level of `n` whole tiles spans within a parent of `np`: one per
# tile up to `np` (`_tile_owners`); its children are placed within them.
_level_rank_count(n::Int, np::Int) = n == 0 ? np : min(n, np)

# The checks over which a change of lattice repays its rebuild of every tile
# of the level, beyond what the rebuilds at the checks already cost.
const LATTICE_HORIZON = 10

# The predicted seconds per root step of a refined level as it is laid out.
function _current_level_cost(solver, spec::RegridSpec, lev::Level)
    isempty(lev.transfers) && return 0.0
    active = ntuple(d -> solver.n_global[d] > 1, 3)
    regions = [lt.region for lt in lev.transfers]
    np = maximum(last, lev.owners) + 1
    costs = spec.costs
    return _layout_cost(costs, regions, [lt.boundary for lt in lev.transfers],
                        [lt.folded for lt in lev.transfers], lev.owners, active,
                        _level_period(solver, lev.index - 1),
                        _level_substeps(solver, lev.index), np) +
           costs.c * length(regions) / spec.interval
end

"""
    _cheapest_lattice(level, rest, current, fresh) -> Int

The lattice of a chain a level takes, from the predicted seconds per root
step of the level, with the levels below it, on each lattice (`level`,
coarsest first) and of the rest of the step (`rest`): the lattice of the
shortest step, the coarsest among equals, unless the level holds tiles on
lattice `current` and the step on it is less than `PLACEMENT_GAIN` times
the shortest, which keeps the current one. `current` is `nothing` when the
level's lattice is not in the chain, and `fresh` marks a level holding no
tile; either takes the shortest outright. Pure and deterministic, so every rank deciding from the same
costs agrees.
"""
function _cheapest_lattice(level::Vector{Float64}, rest::Float64,
                           current::Union{Nothing,Int}, fresh::Bool)
    best = argmin(level)
    (fresh || current === nothing) && return best
    rest + level[current] >= PLACEMENT_GAIN * (rest + level[best]) && return best
    return current
end

"""
    _choose_lattice(solver, spec, ℓ, chain, wanted_sets, current, np, fresh) -> Int

The index into `chain` of the lattice refined level ℓ takes, from the
wanted tiles of each lattice (`wanted_sets`) over the `np` ranks of its
parent: each priced by the cost model (`LatticeCosts`) under the placement
the regrid would make, with the levels below it, against the rest of the
step (the root and the levels above as they are laid out), by
`_cheapest_lattice`. A level of whole tiles spans as many ranks as it has
tiles, at most its parent's, and the levels below it are placed within
those (`_level_rank_count`), so a lattice too coarse to give its level a
rank per rank of its parent slows its children as well. A lattice other
than `current` also pays its first rebuild, every tile at once, over
`LATTICE_HORIZON` checks. Before the model is ready the level keeps
`current`, or takes the configured lattice. Reads only values every rank
deciding holds alike: the reduced tile sets, the stored layouts and the
broadcast model.
"""
function _choose_lattice(solver, spec::RegridSpec, ℓ::Int, chain::Vector{NTuple{3,Int}},
                         wanted_sets::Vector{Vector{BlockRegion}},
                         current::Union{Nothing,Int}, np::Int, fresh::Bool)
    length(chain) == 1 && return 1
    costs = spec.costs
    costs.ready || return something(current, 1)
    levels = getfield(solver, :levels)
    rest = costs.root
    for lev in levels[2:end]
        lev.index < ℓ && (rest += _current_level_cost(solver, spec, lev))
    end
    deeper = [[lt.region for lt in lev.transfers] for lev in levels[2:end]
              if lev.index > ℓ]
    level = zeros(length(chain))
    for (i, w) in enumerate(wanted_sets)
        level[i] = _lattice_level_cost(solver, spec, ℓ, w, np)
        ranks = _level_rank_count(length(w), np)
        for (j, regions) in enumerate(deeper)
            level[i] += _lattice_level_cost(solver, spec, ℓ + j, regions, ranks)
            ranks = _level_rank_count(length(regions), ranks)
        end
        i == current ||
            (level[i] += costs.c * length(w) / (spec.interval * LATTICE_HORIZON))
    end
    pick = _cheapest_lattice(level, rest, current, fresh)
    LATTICE_TRACE[] && MPI.Comm_rank(getfield(solver, :comm)) == 0 &&
        _trace_lattice(solver, spec, ℓ, chain, wanted_sets, level, rest, current, pick)
    return pick
end

# Bench toggle: rank 0 prints each lattice choice, the model and the
# predicted seconds per root step of every lattice.
const LATTICE_TRACE = Ref(false)

@noinline function _trace_lattice(solver, spec, ℓ, chain, wanted_sets, level, rest,
                                  current, pick)
    c = spec.costs
    println("lattice: step ", solver.step, " level ", ℓ, " model a=", c.a, " b=", c.b,
            " k=", c.k, " m=", c.m, " c=", c.c, " root=", c.root, " rest=", rest)
    for (i, e) in enumerate(chain)
        println("  ", e, i == current ? " current" : "", i == pick ? " pick" : "",
                " tiles ", length(wanted_sets[i]), " level s/step ", level[i])
    end
    return nothing
end

# The fixed coefficients, from `FIXED_LATTICE_COSTS` or for a device level,
# with the root priced as the model prices a tile: its largest block's nodes.
function _fix_lattice_costs!(solver, spec::RegridSpec)
    costs = spec.costs
    a, b, k, m, c = something(FIXED_LATTICE_COSTS[], DEVICE_LATTICE_COSTS)
    root = getfield(solver, :patches)[1].decomp
    n_global = solver.n_global
    block = prod(cld(n_global[d], root.dims[d]) for d in 1:3)
    costs.a, costs.b, costs.k, costs.m, costs.c = a, b, k, m, c
    costs.root = a * block
    costs.ready = true
    return costs
end

# The number of recent checks whose walls give the rebuild cost and whose
# median gives the root's seconds.
const LATTICE_WINDOW = 9

# A rank that spent more than this share of an interval's work compiling
# (Julia's cumulative compile time: the first steps, a tile of a new face
# combination, the first change of lattice) gives no observation for it, and
# the root's and the exchanges' seconds, which every rank's compilation
# delays, are not taken from that interval at all. A check's wall is taken
# less the compilation inside it.
const LATTICE_COMPILE = 0.05

# A tile whose seconds per node exceed the median of its harvest by this
# factor is left out of the fit: one rank's interruption, or a compilation
# below the share above.
const LATTICE_OUTLIER = 3.0

# Fields per tile of a rank's harvest record after its level entries:
# level, fine nodes, interface dimensions, seconds per root step, and those
# of the ghost-flux solves.
const _TILE_FIELDS = 5

# Nanoseconds this process has spent compiling, counted while the lattice
# timers run (`_counted_compile_ns` turns Julia's counter on).
_compile_ns() = Base.cumulative_compile_time_ns()[1]

# The same, turning the counter on the first time the model measures, before
# the first check reads it: the counter is process-wide and counted, so a
# `@time` elsewhere leaves it on.
function _counted_compile_ns(costs::LatticeCosts)
    if costs.timing && !costs.compile_counted
        Base.cumulative_compile_timing(true)
        costs.compile_counted = true
    end
    return _compile_ns()
end

"""
    _harvest_lattice_costs!(solver, spec)

Fold the interval since the last regrid check into the lattice cost model
and refit it. Each rank records, per level, the seconds of its tiles and of
its exchanges per root step, the seconds it spent compiling, and, per tile
it holds, the tile's volume, interface dimensions and seconds; rank 0
gathers them and fits `y = s (a V + k)` to each tile's seconds less its
ghost-flux solves by non-negative least squares over exponentially weighted
sums (`LATTICE_MEMORY`; `k = 0` where the tile volumes do not spread,
`LATTICE_SPREAD`), `b` as the ratio of the ghost-flux seconds to `s V g`,
and `m` as the exchange seconds of each level's most loaded rank, which
waits least, per message it sends. The root's seconds are the median of its
most loaded rank's over the last `LATTICE_WINDOW` checks, and `c` the summed
wall of those checks, less the longest compilation of any rank in each,
over the tiles they left. A rank that compiled during the interval gives no
observation (`LATTICE_COMPILE`), and a tile far above
its harvest's median per node is left out (`LATTICE_OUTLIER`). The
coefficients are broadcast, so every rank holds the same model. Collective
over the run's communicator when the model measures; with fixed
coefficients it sets them and communicates nothing.
"""
function _harvest_lattice_costs!(solver, spec::RegridSpec)
    _lattice_measured(solver, spec) || return _fix_lattice_costs!(solver, spec)
    costs = spec.costs
    comm = getfield(solver, :comm)
    levels = getfield(solver, :levels)
    nlev = length(levels)
    steps = solver.step - costs.step_mark
    record = zeros(2 * nlev + 3)
    record[2nlev + 1] = costs.check_wall
    record[2nlev + 3] = costs.check_compile
    if steps > 0
        record[2nlev + 2] = (_compile_ns() - costs.compile_mark) / 1e9 / steps
        times = costs.tile_time
        ghost = costs.ghost_time
        for (i, lev) in enumerate(levels)
            for (k, pi) in enumerate(lev.patches)
                pi <= length(times) || continue
                y = times[pi] / 1e9 / steps
                record[2i - 1] += y
                lev.index == 0 && continue
                lt = lev.transfers[lev.tiles[k]]
                append!(record, (lev.index, prod(fine_extent(lt.region, lt.active, lt.folded)),
                                 count(_interface_dims(lt.active, lt.boundary)), y,
                                 ghost[pi] / 1e9 / steps))
            end
            i <= length(costs.exchange) && (record[2i] = costs.exchange[i] / 1e9 / steps)
        end
    end
    gathered = MPI.gather(record, comm; root=0)
    model = zeros(7)
    if MPI.Comm_rank(comm) == 0
        steps > 0 && _fold_lattice_costs!(solver, spec, gathered, nlev)
        model .= _fit_lattice_costs(costs)
        LATTICE_TRACE[] && println("lattice harvest: step ", solver.step, " steps ", steps,
                                   " last check wall ",
                                   maximum(r[2nlev + 1] for r in gathered), " tiles ",
                                   costs.check_tiles, " compiled ",
                                   [r[2nlev + 2] for r in gathered], " model ", model)
    end
    MPI.Bcast!(model, comm; root=0)
    costs.ready = model[1] > 0
    costs.a, costs.b, costs.k, costs.m, costs.c, costs.root = model[2:7]
    return costs
end

# The weighted sums, on rank 0: the tile fit's five (Σx₁², Σx₁x₂, Σx₂²,
# Σx₁y, Σx₂y for x = (s V, s)), the ghost-flux seconds and their Σ s V g,
# the tiles counted, and the exchange seconds and messages of the most
# loaded ranks.
const _STAT_SIZE = 10

function _fold_lattice_costs!(solver, spec::RegridSpec, records, nlev::Int)
    costs = spec.costs
    levels = getfield(solver, :levels)
    np = length(records)
    work(rec) = sum(@view rec[1:2nlev])
    clean = [rec[2nlev + 2] <= LATTICE_COMPILE * work(rec) for rec in records]
    # Every tile's observation from a rank that did not compile, with its
    # seconds per node and step.
    tiles = NTuple{5,Float64}[]
    for (rec, ok) in zip(records, clean), at in (2nlev + 4):_TILE_FIELDS:length(rec)
        ok && push!(tiles, ntuple(j -> rec[at + j - 1], 5))
    end
    perstep(t) = (t[4] - t[5]) / (_level_substeps(solver, Int(t[1])) * t[2])
    new = zeros(_STAT_SIZE)
    if !isempty(tiles)
        typical = _median(perstep.(tiles))
        for t in tiles
            perstep(t) <= LATTICE_OUTLIER * typical || continue
            s = _level_substeps(solver, Int(t[1]))
            x1, x2, y = s * t[2], s, t[4] - t[5]
            new[1:5] .+= (x1 * x1, x1 * x2, x2 * x2, x1 * y, x2 * y)
            new[6] += t[5]
            new[7] += s * t[2] * t[3]
            new[8] += 1
        end
    end
    if all(clean)
        for i in 2:nlev
            lev = levels[i]
            isempty(lev.transfers) && continue
            busiest = argmax([rec[2i - 1] for rec in records])
            regions, owners = _measured_layout(lev)
            faces = _tile_faces(regions, _level_period(solver, lev.index - 1))
            msgs = _rank_messages(regions, owners, faces,
                                  max(np, maximum(last, owners) + 1))
            msgs[busiest] > 0 || continue
            new[9] += records[busiest][2i]
            new[10] += _level_substeps(solver, lev.index) * msgs[busiest]
        end
        _window!(costs.roots, maximum(rec[1] for rec in records))
    end
    stats = costs.stats
    isempty(stats) && append!(stats, zeros(_STAT_SIZE))
    stats .= LATTICE_MEMORY .* stats .+ new
    # A rank waiting at the check's collectives for one that compiles waits
    # out that compilation, so the slowest compilation comes off the wall.
    wall = maximum(rec[2nlev + 1] for rec in records) -
           maximum(rec[2nlev + 3] for rec in records)
    if wall > 0 && costs.check_tiles > 0
        push!(costs.walls, wall, costs.check_tiles)
        length(costs.walls) > 2LATTICE_WINDOW && deleteat!(costs.walls, 1:2)
    end
    return costs
end

# The median of the values of `v`.
function _median(v)
    x = sort!(collect(Float64, v))
    n = length(x)
    return isodd(n) ? x[(n + 1) ÷ 2] : (x[n ÷ 2] + x[n ÷ 2 + 1]) / 2
end

# Append `x` to the window of recent values, dropping the oldest past
# `LATTICE_WINDOW`.
function _window!(v::Vector{Float64}, x::Float64)
    push!(v, x)
    length(v) > LATTICE_WINDOW && popfirst!(v)
    return v
end

# The layout of each refined level as rank 0 holds it: regions and owners.
_measured_layout(lev::Level) = ([lt.region for lt in lev.transfers], lev.owners)

# The wall of a check per tile it leaves, over the recent checks (`walls`
# holds the pairs of wall and tiles): their summed wall over their summed
# tiles, since the rebuilds that cost a check most come at some checks only.
# The part of a check no lattice changes (the tag sweep, the coefficient
# priming) is counted per tile too, which prices a finer lattice's rebuilds
# high.
function _rebuild_cost(walls::Vector{Float64})
    tiles = sum(@view walls[2:2:end]; init=0.0)
    return tiles > 0 ? sum(@view walls[1:2:end]) / tiles : 0.0
end

# The model `[ready, a, b, k, m, c, root]` from the weighted sums and the
# windows (rank 0).
function _fit_lattice_costs(costs::LatticeCosts)
    st = costs.stats
    (isempty(st) || st[8] <= 0) && return [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    a, k = _nonnegative_fit(st[1], st[2], st[3], st[4], st[5], LATTICE_SPREAD, costs.k)
    b = st[7] > 0 ? st[6] / st[7] : 0.0
    m = st[10] > 0 ? st[9] / st[10] : 0.0
    c = _rebuild_cost(costs.walls)
    root = isempty(costs.roots) ? 0.0 : _median(costs.roots)
    return [1.0, a, b, k, m, c, root]
end

"""
    _nonnegative_fit(sxx, sx1, s11, sxy, s1y, spread, β0=0) -> (α, β)

The least-squares `y ≈ α x + β`, both non-negative, from the sums
`Σx²`, `Σx`, `Σ1`, `Σxy`, `Σy` (each possibly weighted alike). Where `x`
does not spread (the relative determinant `1 − (Σx)² / (Σx² Σ1)` below
`spread`) the intercept is not told apart from the slope, and the slope is
fitted with the intercept held at `β0`.
"""
function _nonnegative_fit(sxx::Float64, sx1::Float64, s11::Float64, sxy::Float64,
                          s1y::Float64, spread::Float64, β0::Float64=0.0)
    sxx > 0 || return (0.0, s11 > 0 ? max(s1y / s11, 0.0) : 0.0)
    origin = (max(sxy / sxx, 0.0), 0.0)
    det = sxx * s11 - sx1 * sx1
    s11 > 0 && det > spread * sxx * s11 ||
        return (max((sxy - β0 * sx1) / sxx, 0.0), β0)
    α = (s11 * sxy - sx1 * s1y) / det
    β = (sxx * s1y - sx1 * sxy) / det
    α >= 0 && β >= 0 && return (α, β)
    # On the boundary: the better of the two one-parameter fits.
    level = (0.0, max(s1y / s11, 0.0))
    residual((p, q)) = p * p * sxx + 2p * q * sx1 + q * q * s11 - 2p * sxy - 2q * s1y
    return residual(origin) <= residual(level) ? origin : level
end

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
function _migrate_tile!(::Type{T}, Q_new, d_new, Q_old, d_old, Nf::NTuple{3,Int},
                        active::NTuple{3,Bool}, old_blocks::Vector{OwnedBlock},
                        new_blocks::Vector{OwnedBlock}, comm::MPI.Comm,
                        keep::NTuple{3,NTuple{2,Bool}}=_NO_BOUNDARY) where {T}
    me = MPI.Comm_rank(comm)
    interior = ntuple(d -> active[d] ?
                      ((keep[d][1] ? 1 : 2):(Nf[d] - (keep[d][2] ? 0 : 1))) : (1:1), 3)
    nodes(b::BlockRegion) = ntuple(d -> (b.offset[d] + 1):(b.offset[d] + b.extent[d]), 3)
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
                 :kappa_art, :inv_J, :inv_r, :cot_over_r, :cot_over_r_gcl,
                 :overwritten)
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
    wanted_sets = [_lattice_wanted(flagsets[i], Ks[i], chain[i], active, lo, hi, spec,
                                   old_regions, i == current, period)
                   for i in eachindex(chain)]
    pick = _choose_lattice(solver, spec, 1, chain, wanted_sets, current, root_lc.size,
                           isempty(old_regions))
    wanted = wanted_sets[pick]
    rechopped = pick != current && !isempty(old_regions)
    _set_level_edges!(spec, 1, chain[pick])
    # An empty wanted set removes every tile: the level stays in the
    # hierarchy with no tiles, and the next tag builds its first again.
    wanted != old_regions || busy !== nothing || return false
    inherited = Dict(r => _inherited_check(r, old_regions, spec.created, spec.checks,
                                           active, period)
                     for r in wanted if rechopped && !(r in old_regions))
    for r in old_regions
        r in wanted || delete!(spec.created, r)
    end
    for r in wanted
        haskey(spec.created, r) || (spec.created[r] = get(inherited, r, spec.checks))
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
                                                 [haskey(old_of, r) for r in wanted]))
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
                                             spec.n_halo, group.comm, spec.deriv,
                                             spec.filt, spec.smoo,
                                             solver.art.smoother, spec.interface_rhs,
                                             spec.backend, ws_pool,
                                             solver.equations.n_species, n_cons,
                                             _grad_Q_columns(solver),
                                             1, 1, spec.tile;
                                             interface_divergence=
                                                 spec.interface_divergence,
                                             ghost_viscous=_ghost_viscous(solver),
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
                p = _repatch(patches[oi], idx, faces[ti], bcs, group.comm)
                push!(new_states, states[oi])
                push!(new_dQ, workspace.dQ[oi])
                push!(new_du, workspace.du[oi])
            else
                p = _build_fine_patch(T, tr, active, root.h, spec.n_halo,
                                      group.comm, spec.deriv, spec.filt,
                                      spec.smoo, solver.art.smoother,
                                      spec.interface_rhs, spec.backend, ws_pool,
                                      solver.equations.n_species, n_cons,
                                      _grad_Q_columns(solver),
                                      idx, 1, faces[ti];
                                      interface_divergence=spec.interface_divergence,
                                      ghost_viscous=_ghost_viscous(solver),
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
        spec.restriction, n_cons, getfield(solver, :subcycle), fine_decomp(ti),
        root_lc.comm, length(owners[ti]), faces[ti];
        interpolation_order=spec.interpolation_order,
        gradient_deriv=_ghost_viscous(solver) ? spec.deriv : nothing,
        parent_h=root.h, boundary=boundaries[ti], folded=folds[ti], period)
        for (ti, tr) in enumerate(wanted)]
    coupling = build_level_coupling(T, root_lc.comm, transfers, [root.region],
                                    [root.decomp], map(fine_decomp, eachindex(wanted)))
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
                                 [p.decomp for p in new_patches], n_cons,
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
        new_data = Dict(ti => (states[local_of[ti]], patches[local_of[ti]].decomp)
                        for ti in eachindex(wanted) if local_of[ti] != 0)
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
    for (i, a) in enumerate(chain)
        wanted = BlockRegion[]
        push!(wanted_sets, wanted)
        isempty(parent_valid) && continue
        flags = flagsets[i]
        cells = Set(keys(flags))
        foreach(r -> push!(cells, _cell_of(r, a, active)), old_regions)
        i == current || foreach(r -> union!(cells, _cells_of(r, a, active)), old_regions)
        for k in sort!(collect(cells); by=k -> (k[3], k[2], k[1]))
            # The lattice's last node: across a seam the period's P + 1.
            t = _lattice_tile(k, active, a, lo, hi,
                              ntuple(d -> wrap[d] > 0 ? wrap[d] + 1 : N[d], 3))
            t === nothing && continue
            _tile_wanted(t, get(flags, k, zero(Int8)), spec, old_regions, created,
                         i == current, active, period) || continue
            _covered_by(_buffered(t, active, margin, _boundary_faces(t, span, reach)),
                        parent_valid, period) || continue
            push!(wanted, t)
        end
    end
    pick = _choose_lattice(solver, spec, ℓ, chain, wanted_sets, current, parent_np,
                           isempty(old_regions))
    wanted = wanted_sets[pick]
    rechopped = pick != current && !isempty(old_regions)
    changed = (wanted != old_regions || parent_changed) &&
              !(isempty(wanted) && isempty(old_regions))
    changed || return (changed=false, edges=chain[pick])
    inherited = Dict(r => _inherited_check(r, old_regions, created, spec.checks, active,
                                           period)
                     for r in wanted if rechopped && !(r in old_regions))
    for r in old_regions
        r in wanted || delete!(created, r)
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
                                      [r in survived for r in wanted])
    old_index = [something(findfirst(==(r), old_regions), 0) for r in wanted]
    kept = [old_index[ti] > 0 && old_owners[old_index[ti]] == owners[ti]
            for ti in eachindex(wanted)]
    carry = rechopped ?
            (fine=[_fine_region(lt) for lt in lev.transfers],
             own=[_erode(_fine_region(lt), lt.imposed, active) for lt in lev.transfers],
             owners=copy(old_owners)) : nothing
    return (changed=true, edges=chain[pick], wanted=wanted, owners=owners,
            np_new=np_new, np_old=lev.level_comm.size, old_index=old_index, kept=kept,
            created=created, carry=carry)
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
            _tag_sweep!(mark!, solver, p, states[li], zeros(Int8, size(p.covered)),
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
    gathered = MPI.gather(packed, comm; root=0)
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
    decision = MPI.bcast(decision, comm; root=0)
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
Internal: rebuild refined level `ℓ` of a hierarchy to the tiles of
`decision` (`wanted`, `owners`, `np_new`, `np_old`, `old_index`, `kept`, the
same on every rank), after level ℓ − 1 has taken its own new layout.

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
                      workspace::Union{Nothing,Workspace}, ℓ::Int, decision,
                      resplit::Bool, init::Bool) where {T}
    spec = getfield(solver, :regrid)
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
    carry = get(decision, :carry, nothing)
    carry_src = Dict{Int,Tuple{eltype(states),Any}}()
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
            push!(new_patches, _repatch(patches[oi], idx, faces[ti],
                                        _fine_bcs(active, faces[ti], boundaries[ti],
                                                  rbcs), group.comm))
            push!(new_states, states[oi])
            if workspace !== nothing
                push!(new_dQ, workspace.dQ[oi])
                push!(new_du, workspace.du[oi])
            end
        else
            p = _build_fine_patch(T, wanted[ti], active, ph, spec.n_halo, group.comm,
                                  spec.deriv, spec.filt, spec.smoo,
                                  solver.art.smoother, spec.interface_rhs,
                                  spec.backend, ws_pool, solver.equations.n_species,
                                  n_cons, _grad_Q_columns(solver), idx, ℓ,
                                  faces[ti];
                                  interface_divergence=spec.interface_divergence,
                                  ghost_viscous=_ghost_viscous(solver),
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
                spec.restriction, n_cons, getfield(solver, :subcycle), fdecomp(ti),
                parent_lc.comm, length(owners[ti]), faces[ti];
                interpolation_order=spec.interpolation_order,
                gradient_deriv=_ghost_viscous(solver) ? spec.deriv : nothing,
                parent_h=ph, boundary=boundaries[ti], folded=folds[ti], period))
        end
        coupling = build_level_coupling(T, parent_lc.comm, transfers, pregions,
                                        map(pdecomp, plocal),
                                        map(fdecomp, eachindex(wanted)))
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
                                 indices, [p.decomp for p in new_patches], n_cons,
                                 _level_period(solver, ℓ))
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
                           old_table, _block_table(d_new, comm), comm, _EVERY_FACE)
        end
    end
    # Under a change of lattice a new tile takes the evolved solution of the
    # old tiles it overlaps, over the run's communicator, whose rank numbers
    # the owner ranges of both layouts use.
    if init && carry !== nothing
        new_data = Dict(ti => (states[local_of[ti]], patches[local_of[ti]].decomp)
                        for ti in eachindex(wanted) if local_of[ti] != 0)
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
