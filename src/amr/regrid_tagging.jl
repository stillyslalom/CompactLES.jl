# Tagging and regridding. This file and regrid_lattice.jl, with the tiled
# regrid in regrid_swap.jl, move the one refined level of a two-level
# hierarchy; the last section of regrid_swap.jl regrids a deeper tiled
# hierarchy level by level with the same tag criteria.
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
# pollution decays ≈ 3.4× per point, bench/archive/amr_transfer.jl, so a ~4-cell
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
# artificial diffusivity the patch holds in units of the acoustic cell
# diffusivity. At a regrid check the coefficients and the primitives ρ, c and
# c_p are computed from the state under test (`_tag_coefficients!`), so the
# ratio is consistent; a node the positivity placeholder wrote (c = 0) never
# tags.
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
    # artificial coefficients the same criterion reads are the ones the patch
    # holds: a regrid check computes them from the same state first
    # (`_tag_coefficients!`), and a direct call reads the last evaluation's.
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
Collapsed dimensions keep offset 0 and extent 1. The sensor criterion reads
the artificial coefficients the root holds, which a regrid check computes
from the state first (`_tag_coefficients!`). Collective over the coarse
communicator (one Allreduce of the tag bounds, and one of their occupancy
along a periodic dimension under `level_boundaries`), so every rank returns
the identical region.
"""
function tagged_region(solver::Solver, Qc)
    spec = getfield(solver, :regrid)
    schemes = getfield(solver, :schemes)
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
            need = max(least, _placement_extent(schemes.level_interpolation_order,
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

# The bounding box of two regions of the root's node space, each a legal
# refined box. Either end of the result is an end of one of them, so it keeps
# the margin or lies on the face that region reached. Along a dimension the
# placement wraps (`wrap`, its period) the image of `b` nearest `a` is taken,
# the span is capped at the period less the margin at either end, about its
# centre, as `tagged_region` caps it, and the offset is brought into [0, P).
function _region_union(a::BlockRegion, b::BlockRegion, wrap::NTuple{3,Int}, margin::Int)
    offext = ntuple(3) do d
        alo, ahi = a.offset[d], a.offset[d] + a.extent[d]
        blo, bhi = b.offset[d], b.offset[d] + b.extent[d]
        lo, hi = min(alo, blo), max(ahi, bhi)
        P = wrap[d]
        P == 0 && return (lo, hi - lo)
        for s in (-P, P)
            slo, shi = min(alo, blo + s), max(ahi, bhi + s)
            shi - slo < hi - lo && ((lo, hi) = (slo, shi))
        end
        n = min(hi - lo, P - 2 * margin)
        (mod(fld(lo + hi - n, 2), P), n)
    end
    return BlockRegion(ntuple(d -> offext[d][1], 3), ntuple(d -> offext[d][2], 3))
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
                          solver.schemes.level_interpolation_order,
                          _unbuffered_faces(solver, faces))
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
        _fill_fallback(solver) && _fill_fallback!(solver, Qf, lt, fine.decomp)
        return states
    end
    # Device patch: every component runs through the scratch's chain, whose
    # stages hold this rank's share of the components, in slices of that
    # width (the whole set at once on a one-rank tile).
    stages = fine.amr.level_scratch.stages
    width = size(stages[1], 4)
    for lo in 1:width:n_cons
        comps = lo:min(lo + width - 1, n_cons)
        _fill_stage0_dev!(BoxFill(), view(stages[1], :, :, :, 1:length(comps)),
                          fine.amr.level_scratch, lt, comps)
        for k in 1:K
            _interpolate_dev!(stages[k+1], lt.pplans[k], stages[k], length(comps))
        end
        for (b, c) in enumerate(comps)
            _write_fine_shell!(Qf, c, view(stages[K+1], :, :, :, b), lt,
                               fine.decomp, lt.pdecomps[K+1], false)
        end
    end
    _fill_fallback(solver) && _fill_fallback!(solver, Qf, lt, fine.decomp)
    return states
end

# The admissible fallback of a fill, the shell's (`_shell_fallback!`) over the
# patch's own nodes: a node the Lagrange chain leaves with ρ ≤ 0 or ρe ≤ 0
# takes the multilinear interpolant of the parent nodes bracketing it where
# that is admissible, as it is wherever they are. In cold gas moving fast, as
# ahead of the shock of Noh, ρe = E − |m|²/(2ρ) is a small difference of large
# terms, and the chain's errors in E, m and ρ, each interpolated on its own,
# can exceed it. The positivity limiter keeps a node admissible from an
# admissible base state and does not raise one that enters a stage below its
# bound, so such a node stays inadmissible for several steps. The halos and
# the shell are written again before the next step, the shell with its own
# fallback. Taken where the limiter runs, under either interface flux: the
# closure rows' fills leave such nodes as well at the same interpolation
# order. A run without the limiter keeps the chain's values.
_fill_fallback(solver) =
    getfield(solver, :positivity) !== nothing && _shell_fallback(solver)

function _fill_fallback!(solver, Qf, lt::LevelTransfer, df::Decomp)
    src = _limiter_upload(parent(Qf), lt.box_gather)
    place = (df.offset, df.n_halo_d, _box_shift(lt),
             (df.active[1], df.active[2], df.active[3]))
    box = (lt.pdecomps[1].n_halo_d, lt.pdecomps[1].n_local)
    n = df.n_local
    pointwise!(_fill_fallback_point!, Qf, n[1], n[2], n[3],
               Qf, src, zero(eltype(src)), place, box, _limiter_layout(solver))
    return Qf
end

@inline function _fill_fallback_point!(Q, src, w, place, box, lay, i, j, k)
    @inbounds begin
        off, pad, shift, active = place
        ns, m1, m2, m3, ie, nc = lay
        I = CartesianIndex(i + pad[1], j + pad[2], k + pad[3])
        ρ = zero(w)
        for sp in 1:ns
            ρ += Q[I, sp]
        end
        mx, my, mz = Q[I, m1], Q[I, m2], Q[I, m3]
        admissible = (ρ > 0) & (2 * ρ * Q[I, ie] - (mx * mx + my * my + mz * mz) > 0)
        if !admissible
            # The node of the chain's final box, as `_write_fine_shell!` reads it.
            n = (i + off[1] + shift[1], j + off[2] + shift[2], k + off[3] + shift[3])
            if _linear_admissible(src, src, w, n, active, box, 1, 1, lay)
                for c in 1:nc
                    Q[I, c] = _linear_value(src, src, w, n, active, box, c)
                end
            end
        end
    end
    return nothing
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
function _regrid_impl!(solver::Solver{T}, states::AbstractVector,
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
    spec.checks - get(_created(spec, 1), lt.region, 0) >= spec.lifetime || return false
    newregion = tagged_region(solver, states[1])
    newregion === nothing && return false
    newregion == lt.region && return false
    return _move_box!(solver, states, workspace, newregion)
end

# The refined box of a two-level hierarchy rebuilt over `newregion`, a region
# every rank holds alike: the body of the box regrid, which the setup of a
# run started from a given region calls as well (`_widen_given_start!`).
function _move_box!(solver::Solver{T}, states::AbstractVector,
                    workspace::Workspace, newregion::BlockRegion) where {T}
    spec = getfield(solver, :regrid)
    schemes = getfield(solver, :schemes)
    levels = getfield(solver, :levels)
    lt = levels[2].transfers[1]
    delete!(_created(spec, 1), lt.region)
    _created(spec, 1)[newregion] = spec.checks
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
        _build_fine_patch(T, newregion, active_g, _root_spacing(solver),
                          spec.n_halo, group.comm,
                          schemes.deriv, schemes.filt, spec.smoo,
                          solver.art.smoother, schemes.interface_rhs,
                          spec.backend, ws_pool,
                          solver.equations.n_species, n_cons,
                          _grad_Q_columns(solver), fi, 1;
                          interface_divergence=schemes.interface_divergence,
                          gradient_ring=_gradient_ring_built(solver),
                          ghost_fluxes=_ghost_fluxes_built(solver),
                          ring=_ring_detector(solver),
                          boundary=bnd, bcs=patches[1].bcs,
                          root_folds=patches[1].folds) : nothing
    newlt = build_level_transfer(T, newregion, active_g, spec.n_halo,
                                 [patches[1].region], [1],
                                 fi, lt.restriction, n_cons,
                                 getfield(solver, :subcycle),
                                 newfine === nothing ? nothing : newfine.decomp,
                                 root_lc.comm, length(owners[1]);
                                 interpolation_order=schemes.level_interpolation_order,
                                 gradient_deriv=_gradient_ring_built(solver) ?
                                                schemes.deriv : nothing,
                                 parent_h=_root_spacing(solver), boundary=bnd, folded,
                                 period)
    coupling = build_level_coupling(T, root_lc.comm, [newlt], [patches[1].region],
                                    Union{Nothing,Decomp{T}}[patches[1].decomp],
                                    Union{Nothing,Decomp{T}}[newfine === nothing ? nothing :
                                                             newfine.decomp])
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
function regrid!(solver::Solver, Q, workspace::Workspace, save, observer=nothing)
    states, workspace = _patch_states(Q), _patch_workspace(workspace)
    _tag_coefficients!(solver, states, observer)
    changed = _regrid_impl!(solver, states, workspace, save)
    # The coefficient refresh is a regrid-check cadence, not a layout-change
    # side effect: otherwise a timing-dependent rebalance decision changes the
    # root CFL sequence.  A savepoint still follows only a changed layout.
    _regrid_prime!(solver, states, workspace, save, changed, observer)
    _close_lattice_check!(solver, getfield(solver, :regrid), 0.0)
    return changed
end

# The artificial coefficients the sensor criterion tags from, computed from the
# state under test on every level before a check's sweeps, with every sensor
# kept: a parent's sensors are otherwise dropped at the nodes a child's
# restriction overwrites (`MASK_CHILD_SENSORS`, rhs.jl), where the criterion
# would then find nothing to hold the child with and a level tagged by it
# would leave at every check. The check's own priming restores the masked
# coefficients before the next step reads them (`_regrid_prime!`). A no-op
# with the criterion off; collective otherwise, the record alike on every
# rank.
function _tag_coefficients!(solver::Solver, states::AbstractVector, observer=nothing)
    spec = getfield(solver, :regrid)
    (spec.sensor_threshold > 0 && solver.art.enabled) || return nothing
    solver.tstage = solver.t
    _presync!(solver, states, true, observer)
    # Every patch's sensors kept whole (`_sensor_mask_patch`).
    _prime_level_coefficients!(solver, states, false, observer)
    return nothing
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
function _regrid_prime!(solver, states, workspace, save, changed, observer=nothing)
    if solver.art.enabled
        solver.tstage = solver.t
        _presync!(solver, states, true, observer)
        changed ? _prime_rhs!(solver, states, workspace, observer) :
                  _prime_level_coefficients!(solver, states, true, observer)
    end
    changed && _rebank!(solver, states, save)
    return nothing
end
