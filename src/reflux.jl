# Conservative coupling at coarse-fine faces.
#
# The conserved quadrature. A node-centred compact divergence is a difference
# of face fluxes at the half nodes, D f_i = (F̂_{i+½} − F̂_{i−½})/h, with
# F̂ the solution of the interior face relation
#
#     Σ_s l_s F̂_{j+½+s} = R_{j+½},   R_{j+½} = Σ_m c_m Σ_{l=1−m}^{m} f_{j+l},
#
# l the left-hand side's band (l_0 = 1) and c_m the undivided right-hand side
# of (B f)_i = Σ_m c_m (f_{i+m} − f_{i−m})/h. A sum of h D f over the nodes of
# a segment telescopes to F̂ at its two outer half nodes. The composite
# quadrature is built on that: a parent node counts with its whole cell unless
# a child covers the cell entirely, and a child node counts with its whole
# cell from the third node in from a parent-fed face, so the two grids meet at
# x_b + H/2, the half node beyond the parent's face node b and the half node
# between the child's second and third nodes. The child's first node (which
# the shell overwrites) and second, and the covered parent nodes (which
# restriction overwrites or which lie inside the child) take no weight, so
# neither the shell nor the restriction moves the integral.
#
# The face fluxes. The relation holds with one constant over any run of
# interior rows, and that constant is zero: the running sum of a closed line
# from its wall, under the weights its closure rows telescope with, satisfies
# the relation at every interior face to round-off for every closure set (the
# `:neutral3`, cascade and Brady–Livescu rows of C6 and C8, checked against
# `_line_operator`). F̂ at one face therefore follows locally from the
# derivative at the nodes beside it,
#
#     F̂_{j+½} = (R_{j+½} − Σ_{s≥1} l_s (C_s + C_{−s})) / Σ_s l_s,
#
# C_s the cumulative h D f from face j+½ to face j+½+s, and a sum of h D f
# from that face is the flux at another. The parent's flux is taken at the
# face beside its face node, the child's at its third half node in from the
# face or further, past the rows its interface closures replace. A filter
# pass is the same with its correction (B − A) q, a difference of an explicit
# face stencil (`_filter_face_stencil`); it changes Q and not the volume
# weighted Q the budget sums, so each side's face value is weighted by the
# cell volume of its counted node beside the junction, which leaves along a
# line of varying volume the pass's own defect, as on a uniform grid.
#
# The correction. Over a step the parent's counted nodes lose κ_c ∫F̂^c dt
# through the junction and the child's gain κ_f ∫F̂^f dt, κ the transverse
# measure of a line, and the parent is given the difference. That alone
# conserves the box rule exactly, but the box rule on two spacings is second
# order, and F̂^c − F̂^f is of order (H² − h²) f'' for smooth data, so the
# correction was a first-order source: the interface orders fell to 2 on a
# model problem. The conserved quadrature therefore carries a correction Ω of
# the box rule at each junction, six weights ω on the child's counted nodes
# beside it chosen so that its rate cancels F̂^c − F̂^f for smooth data
# (`_junction_omega`). The rate of Ω enters the registers from each
# divergence's increments at those nodes, and its change in a filter pass
# with the pass, so a source or a node term (the radial momentum's pressure
# gradient), which the budget integrates as it is, enters neither part.
#
# Where it applies. Along a 1-D line the correction keeps the interface's
# order. Across a 2-D or 3-D face the child's lines are summed into the
# parent line whose cell holds them, and the two grids' tangential
# quadratures differ by order H³ per line, more beside the r-z axis, which
# summed per line is again a first-order source (a 2-D entropy wave fell from
# sixth order to second). A junction line is therefore corrected only in a
# step in which the parent's density, within `GATE_REACH` nodes of the face,
# holds a feature its spacing does not resolve (the masks' fourth-difference
# test); smooth flow is left to the uncorrected coupling, whose drift
# converges with the spacing. Each connected run of flagged lines on a face
# takes its summed correction as one change over the run, so the tangential
# difference, which telescopes along the run, is not put on any one node.
#
# The correction's node is the parent's next node out from the face, not the
# face node, which the shell imposes on the child's boundary plane: there it
# fed back on itself, and a cold inflow into a tile along r grew without
# bound. A run takes the largest fraction of its change that keeps each node
# above half its density and internal energy per volume, and the rest is
# applied on later steps (`RefluxCarry`), so a strong shock leaving a tile
# into cold gas does not take the node ahead of it negative.
#
# Timing. A right-hand side collects each junction line's face fluxes in
# `stage` (`compute_rhs!` zeroes it, the divergence funnels add), the stage
# update carries them through the low-storage recurrence into `reg`, and a
# filter pass adds to `reg` directly; the Hermite endpoint's extra right-hand
# side takes no update and so enters nothing. After the child level's steps
# and the restriction that ends its parent's step, the junction lines are
# summed over the parent level's communicator and the correction applied.
# Every root step starts from empty registers and restores what a retried
# step held back, so a rollback, a regrid or a restart leaves nothing stale.
#
# Scope. Host storage, without the positivity limiter, which closes the
# interfaces with their closure rows and limits its own faces.

"""
Test and benchmark toggle: `false` leaves the coarse-fine faces uncorrected,
as before the conservative coupling. Read when the captures are built and at
the start of every root step.
"""
const REFLUX = Ref(true)

# --- Face relations -------------------------------------------------------------

# The left-hand-side band of a scheme, l_1 .. l_q.
_reflux_band(s::CompactScheme) = [Float64(s.alpha)]
_reflux_band(s::BandedCompactScheme) = Float64.(s.lhs)

# A derivative's explicit face stencil r_l, l = 1 − M .. M, at r[l + M]:
# R_{j+½} = Σ_l r_l f_{j+l}.
function _derivative_face_stencil(s)
    c = Float64.(s.coeffs)
    M = length(c)
    return [sum(c[m] for m in max(l, 1 - l):M; init=0.0) for l in 1-M:M]
end

# The coefficients of the h·D values at nodes (patch-global indices along the
# line) in the face flux at face J + ½, from the face relation at the anchor
# face j + ½ (`band`, its l_1 .. l_q) and the running sum between: a Dict
# from node to coefficient, and 1/Σ l, the anchor's weight on R.
function _anchor_coefficients(band::Vector{Float64}, j::Int, J::Int)
    q = length(band)
    rs = 1 / (1 + 2 * sum(band))
    coef = Dict{Int,Float64}()
    add(i, v) = (coef[i] = get(coef, i, 0.0) + v)
    for m in 1:q
        add(j + m, -rs * sum(band[m:q]))
    end
    for m in 0:q-1
        add(j - m, rs * sum(band[m+1:q]))
    end
    for m in j+1:J
        add(m, 1.0)
    end
    for m in J+1:j
        add(m, -1.0)
    end
    return coef, rs
end

# The parent's coefficients at a junction whose face node is node `gp` of a
# parent patch of `n` nodes along the normal, `side` the junction's side of
# the child: the coefficient dictionary, the anchor's weight on its stencil,
# and the anchor face's node (0 for none). A face node inside the patch takes
# the face relation beside it. On a plane the patch shares with a same-level
# neighbour the node counts half in each, so the patch whose counted nodes
# lie beyond it takes the relation at its last interior face and half the
# node's increment, and the patch on the child's side half the node's
# increment alone. A patch spanning a periodic dimension has no ends along it.
function _parent_coefficients(band::Vector{Float64}, gp::Int, n::Int, side::Int,
                              periodic::Bool=false)
    s = side == 1 ? 1 : -1
    lo_end, hi_end = periodic ? (false, false) : (gp == 1, gp == n)
    if !(lo_end || hi_end)
        j = side == 1 ? gp : gp - 1
        c, rs = _anchor_coefficients(band, j, j)
        return c, rs, j
    end
    # The patch holds the counted side when the node is its end toward it.
    counted = side == 1 ? hi_end : lo_end
    if counted
        j = side == 1 ? gp - 1 : gp
        c, rs = _anchor_coefficients(band, j, j)
        c[gp] = get(c, gp, 0.0) + s * 0.5
        return c, rs, j
    end
    return Dict(gp => s * 0.5), 0.0, 0
end

# --- The quadrature correction --------------------------------------------------

# Truncated power series in θ (coefficients of θ^0 .. θ^(K-1)).
const _SERIES_TERMS = 10
_series_sin(a) = [isodd(k) ? a^k * (-1)^((k - 1) ÷ 2) / factorial(k) : 0.0
                  for k in 0:_SERIES_TERMS-1]
_series_cos(a) = [iseven(k) ? a^k * (-1)^(k ÷ 2) / factorial(k) : 0.0
                  for k in 0:_SERIES_TERMS-1]
_series_mul(x, y) = [sum(x[i+1] * y[k-i+1] for i in 0:k) for k in 0:_SERIES_TERMS-1]
function _series_div(x, y)
    q = zeros(_SERIES_TERMS)
    for k in 0:_SERIES_TERMS-1
        q[k+1] = (x[k+1] - sum((q[i+1] * y[k-i+1] for i in 0:k-1); init=0.0)) / y[1]
    end
    return q
end
_series_shift(x) = [x[2:end]; 0.0]      # x / θ, for x[1] == 0

# The face flux's symbol relative to the point value at the face, σ(θ) =
# (k'(θ)/θ) (θ/2)/sin(θ/2), θ = κh, for the scheme's modified wavenumber
# k'(θ) = Σ_m 2c_m sin(mθ) / (1 + Σ_s 2l_s cos(sθ)): its even coefficients.
function _face_symbol(scheme)
    c = Float64.(scheme.coeffs)
    band = _reflux_band(scheme)
    num = zeros(_SERIES_TERMS)
    for (m, cm) in enumerate(c)
        num .+= 2cm .* _series_sin(m)
    end
    den = zeros(_SERIES_TERMS)
    den[1] = 1.0
    for (s, ls) in enumerate(band)
        den .+= 2ls .* _series_cos(s)
    end
    kp = _series_div(_series_shift(num), den)
    half = _series_shift(_series_sin(0.5))
    inv_half = _series_div([0.5; zeros(_SERIES_TERMS - 1)], half)
    return _series_mul(kp, inv_half)
end

"""
    _junction_omega(scheme, offsets) -> Vector{Float64}

The quadrature correction's weights at a junction, in units of the child's
spacing h, on child nodes at `offsets` (in h, signed) from the junction's
half node, for a 3:1 refinement under `scheme`. With them,
Σ_j ω_j h f'(x_j) equals F̂^c − F̂^f, the difference of the two grids' face
fluxes at the junction for smooth f, through the order the number of nodes
allows: (σ(3θ) − σ(θ)) f matched in its powers of θ up to θ^n.
"""
function _junction_omega(scheme, offsets::Vector{Float64})
    σ = _face_symbol(scheme)
    n = length(offsets)
    A = zeros(n, n)
    b = zeros(n)
    for p in 0:n-1
        q = p + 1
        for (k, o) in enumerate(offsets)
            # i^{p+1} o^p / p!, real part of the matched power
            A[p+1, k] = iseven(p) ? o^p / factorial(p) :
                        (-1)^(q ÷ 2) * o^p / factorial(p)
        end
        b[p+1] = iseven(q) ? σ[q+1] * (3.0^q - 1) : 0.0
    end
    return A \ b
end

# --- Construction ---------------------------------------------------------------

# The rows each kind of plan replaces at a refined patch's coarse-fine end,
# from the schemes the solver was built with: the gradient rows, the
# divergence rows and the filter rows.
function _replaced_rows(settings::SchemeSettings)
    ext = settings.interface_rhs === :extended
    nd = length(ext ? interface_closures(settings.deriv) : settings.deriv.closures)
    src = settings.interface_divergence
    nv = ext || src !== nothing ?
         length(interface_divergence_rows(settings.deriv, src)) : nd
    nf = length(ext ? interface_closures(settings.filt) : settings.filt.closures)
    return max(nd, nv), nf
end

# Whether the solver's configuration takes the conservative coupling.
function _reflux_supported(solver::Solver)
    REFLUX[] || return false
    nlevels(solver) > 1 || return false
    getfield(solver, :positivity) === nothing || return false
    getfield(solver, :implicit) === nothing || return false
    # A stacked device level advances its tiles per stack.
    any(lev -> !isempty(lev.stacks), getfield(solver, :levels)) && return false
    return all(p -> _cpu_storage(p.rho), getfield(solver, :patches))
end

# The junctions under parent level `ℓp` (1-based): per tile of the level
# below and per parent-fed face, its tile, normal, side, and the box of
# parent lines (1-based parent-level node indices, per transverse dimension
# in ascending order) that cross it. Replicated on every rank of the parent
# level's subset, which holds every transfer.
function _reflux_junctions(solver::Solver, ℓp::Int)
    levels = getfield(solver, :levels)
    out = NamedTuple{(:tile, :d, :side, :box),
                     Tuple{Int,Int,Int,NTuple{2,UnitRange{Int}}}}[]
    ℓp < length(levels) || return out
    for (t, lt) in enumerate(levels[ℓp+1].transfers)
        r = lt.region
        for d in 1:3, side in 1:2
            lt.active[d] && lt.imposed[d][side] || continue
            # The child's anchors and Ω band need ten nodes along d.
            3 * (r.extent[d] - 1) + 1 >= 10 || continue
            o = d == 1 ? (2, 3) : d == 2 ? (1, 3) : (1, 2)
            box = ntuple(2) do k
                e = o[k]
                lt.active[e] || return 1:1
                lo = r.offset[e] + 1 + (lt.imposed[e][1] ? 1 : 0)
                hi = r.offset[e] + r.extent[e] - (lt.imposed[e][2] ? 1 : 0)
                lo:hi
            end
            push!(out, (tile=t, d=d, side=side, box=box))
        end
    end
    return out
end

_reflux_lines(box) = length(box[1]) * length(box[2])

_float_type(::Solver{T}) where {T} = T

# Rebuild the captures if the layout has changed since they were built. The
# rebuild runs at setup and after a regrid and reads the schemes at their
# abstract types, so it sits behind `_cold`, out of the step's inference.
function _reflux_current!(solver::Solver)
    nlevels(solver) > 1 || return solver
    get(REFLUX_KEYS, solver, UInt(0)) == _reflux_key(solver) ||
        _build_reflux!(_cold(solver))
    return solver
end

# The key of the layout the captures were built for.
const REFLUX_KEYS = WeakKeyDict{Any,UInt}()

# Per solver and parent level: the part of each junction line's correction
# the positivity guard has held back (`_reflux_apply!`), replicated over the
# level's ranks, and its value at the start of the root step, which a step's
# retry restores.
mutable struct RefluxCarry
    step::Int
    carry::Dict{Int,Vector{Float64}}
    saved::Dict{Int,Vector{Float64}}
end
const REFLUX_CARRY = WeakKeyDict{Any,RefluxCarry}()
_reflux_carry(solver) = get!(() -> RefluxCarry(-1, Dict{Int,Vector{Float64}}(),
                                                Dict{Int,Vector{Float64}}()),
                             REFLUX_CARRY, solver)
_reflux_key(solver::Solver) =
    hash((REFLUX[], map(objectid, getfield(solver, :patches)),
          [lt.region for lev in getfield(solver, :levels) for lt in lev.transfers]))

"""
    _build_reflux!(solver)

Rebuild the junction captures of every held patch for the current layout,
empty where the configuration does not take the conservative coupling.
Rank-local.
"""
function _build_reflux!(@nospecialize(solver::Solver))
    # Setup, so compiled once rather than per solver type: every solver type
    # a run meets at a regrid or a diagnostic otherwise compiles its own.
    T = _float_type(solver)
    patches = getfield(solver, :patches)
    for p in patches
        empty!(p.reflux_captures)
    end
    REFLUX_KEYS[solver] = _reflux_key(solver)
    # A new layout starts with nothing held back; what a regrid replaces
    # with the old layout is the regrid's.
    delete!(REFLUX_CARRY, solver)
    _reflux_supported(solver) || return solver
    levels = getfield(solver, :levels)
    settings = solver.schemes
    deriv, filt = settings.deriv, settings.filt
    dband, fband = _reflux_band(deriv), _reflux_band(filt)
    drs, frs = _derivative_face_stencil(deriv), Float64.(_filter_face_stencil(filt, Float64))
    dM, fM = length(drs) ÷ 2, length(frs) ÷ 2
    (dM <= 4 && fM <= 4) || return solver
    nd, nf = _replaced_rows(settings)
    ja, jaf = max(2, nd), max(2, nf)
    lo_offsets = [g - 2.5 for g in 3:8]
    ω = _junction_omega(deriv, lo_offsets)
    n_cons = solver.equations.n_cons
    for ℓp in 1:length(levels)-1
        junctions = _reflux_junctions(solver, ℓp)
        for (jn, jc) in enumerate(junctions)
            lt = levels[ℓp+1].transfers[jc.tile]
            s = jc.side == 1 ? 1.0 : -1.0
            gb = jc.side == 1 ? lt.region.offset[jc.d] + 1 :
                                lt.region.offset[jc.d] + lt.region.extent[jc.d]
            # The parent's side, on each parent patch holding part of it, in
            # each periodic image of the tile the patch meets: across a seam
            # the face node and the lines lie in different images. A patch
            # spanning a periodic dimension wraps its stencil along it.
            P = lt.period
            o = jc.d == 1 ? (2, 3) : jc.d == 2 ? (1, 3) : (1, 2)
            for q in lt.coarse_local, σ in _images(P)
                q == 0 && continue
                p = patches[q]
                np = p.region.extent[jc.d]
                gp = gb + σ[jc.d] - p.region.offset[jc.d]
                1 <= gp <= np || continue
                wrap = P[jc.d] > 0 && np == P[jc.d] ? np : 0
                box = ntuple(k -> jc.box[k] .+ σ[o[k]], 2)
                dc, drsc, dj = _parent_coefficients(dband, gp, np, jc.side, wrap > 0)
                fc, frsc, fj = _parent_coefficients(fband, gp, np, jc.side, wrap > 0)
                cap = _reflux_capture(T, solver, p, ℓp, jn, false, jc.d, s, box,
                                      dc, dj, drs, drsc, fc, fj, frs, frsc, gp,
                                      Int[], Float64[], n_cons, gb, lt, wrap)
                cap === nothing || push!(p.reflux_captures, cap)
            end
            # The child's side.
            lt.fine_index == 0 && continue
            f = patches[lt.fine_index]
            n = f.region.extent[jc.d]
            n >= 2ja + 4 || error("reflux: a tile of $n nodes along $(jc.d) is " *
                                  "too short for its interface rows")
            if jc.side == 1
                dc, drsc = _anchor_coefficients(dband, ja, 2)
                fc, frsc = _anchor_coefficients(fband, jaf, 2)
                aj, af = ja, jaf
                band = collect(3:8)
                om = -ω
            else
                dc, drsc = _anchor_coefficients(dband, n - ja, n - 2)
                fc, frsc = _anchor_coefficients(fband, n - jaf, n - 2)
                aj, af = n - ja, n - jaf
                band = collect(n-2:-1:n-7)
                om = _junction_omega(deriv, [g - (n - 1.5) for g in band])
            end
            cap = _reflux_capture(T, solver, f, ℓp, jn, true, jc.d, -s, jc.box,
                                  dc, aj, drs, drsc, fc, af, frs, frsc, 0,
                                  band, om, n_cons, gb, lt)
            cap === nothing || push!(f.reflux_captures, cap)
        end
    end
    return solver
end

# One capture on patch `p`, or `nothing` where this rank holds none of its
# lines. Indices along `d` are patch-global (1-based) on entry; `wrap` is the
# patch's period along `d` when it spans a periodic dimension, 0 otherwise,
# and then every index along `d` is taken modulo it.
function _reflux_capture(::Type{T}, @nospecialize(solver), @nospecialize(p), ℓp, jn,
                         child, d, sgn, box, dc, dr, drs, drsc, fc, fr, frs, frsc,
                         gface, band, om, n_cons, gb, @nospecialize(lt),
                         wrap::Int=0) where {T}
    ps = PatchSolver(solver, p)
    decomp = p.decomp
    pad = decomp.n_halo_d
    nl = decomp.n_local
    off = decomp.offset
    o = d == 1 ? (2, 3) : d == 2 ? (1, 3) : (1, 2)
    local_of(g, e) = (i = g - off[e]; 1 <= i <= nl[e] ? i + pad[e] : 0)
    wrapd(g) = wrap > 0 ? mod1(g, wrap) : g
    # An anchor of 0 is none, but under `wrap` there is always one.
    anchor(g) = wrap > 0 ? local_of(wrapd(g), d) : local_of(g, d)
    # The transverse lines held here, as padded ranges.
    lines = ntuple(2) do k
        e = o[k]
        decomp.active[e] || return (pad[e] + 1):(pad[e] + 1)
        if child
            # The child's counted lines: in from a parent-fed transverse face.
            glo = lt.imposed[e][1] ? 3 : 1
            ghi = p.region.extent[e] - (lt.imposed[e][2] ? 2 : 0)
        else
            glo = box[k].start - p.region.offset[e]
            ghi = box[k].stop - p.region.offset[e]
        end
        ilo = max(glo - off[e], 1)
        ihi = min(ghi - off[e], nl[e])
        (ilo + pad[e]):(ihi + pad[e])
    end
    any(isempty, lines) && return nothing
    allnodes = sort(collect(union(keys(dc), keys(fc))))
    length(allnodes) <= 6 || error("reflux: more than six captured nodes")
    nodes = ntuple(k -> k <= length(allnodes) ? local_of(wrapd(allnodes[k]), d) : 0, 6)
    dcoef = ntuple(k -> T(k <= length(allnodes) ? get(dc, allnodes[k], 0.0) : 0.0), 6)
    fcoef = ntuple(k -> T(k <= length(allnodes) ? get(fc, allnodes[k], 0.0) : 0.0), 6)
    pad8(v) = ntuple(k -> T(k <= length(v) ? v[k] : 0.0), 8)
    nline = length(lines[1]) * length(lines[2])
    measure = Float64(cell_measure(ps)) / Float64(p.h[d])
    # A filter pass changes Q, not the volume-weighted Q the budget sums, so at
    # the junction its face value enters the budget weighted by the cell
    # volume of the counted node beside it: the parent's face node, the
    # child's third node. What remains along a line where the volume varies
    # is the pass's own defect, which the uniform grid carries too.
    jg = child ? band[1] : gface
    ja = jg - off[d] + pad[d]
    jin = 1 <= ja <= size(p.inv_J, d)
    kap = zeros(T, nline)
    kapf = zeros(T, nline)
    wb = zeros(T, nline)
    entry = zeros(Int, nline)
    # The correction's node: the parent's next node out from the face. The
    # face node itself is the child's boundary plane through the shell, and
    # a correction there fed back on itself: on a cold inflow into the tile
    # along r it grew without bound, where one node out it decays.
    bnode = child ? 0 : local_of(wrapd(gface - round(Int, sgn)), d)
    # The gate's nodes: the parent's within `GATE_REACH` + 2 of its face node
    # on either side, the reach of the density test's taps, covered ones
    # included: they hold the child's restricted solution, where a feature
    # leaving the child is first seen. Slot s of the junction's `GATE_WIDTH`
    # is the node gface - GATE_REACH - 3 + s on every patch and rank, taken
    # across the seam under `wrap`.
    window = Int[]
    wslots = Int[]
    if !child
        for s in 1:GATE_WIDTH
            g = wrapd(gface - GATE_REACH - 3 + s)
            1 <= g <= p.region.extent[d] || continue
            a = local_of(g, d)
            a == 0 && continue
            push!(window, a)
            push!(wslots, s)
        end
    end
    lo_box = (box[1].start, box[2].start)
    n_box = (length(box[1]), length(box[2]))
    for (l, (i1, i2)) in enumerate(Iterators.product(lines[1], lines[2]))
        idx = (i1, i2)
        w = measure
        for k in 1:2
            e = o[k]
            decomp.active[e] || continue
            w *= quad_weight(ps, e, idx[k] - pad[e])
        end
        # The parent-level node of each transverse index: its own, or the
        # parent node whose cell holds a child's.
        pt = ntuple(2) do k
            e = o[k]
            decomp.active[e] || return 1
            g = idx[k] - pad[e] + off[e]
            # A tile at a fold starts `_fold_lead` fine nodes before the
            # lattice coincident with its parent's.
            child ? lt.region.offset[e] + 1 + fld(g - _fold_lead(lt.folded, e, 1), 3) :
                    g + p.region.offset[e]
        end
        all(k -> 0 <= pt[k] - lo_box[k] < n_box[k], 1:2) || continue
        entry[l] = (pt[1] - lo_box[1]) + n_box[1] * (pt[2] - lo_box[2]) + 1
        # A parent line on the plane a tile shares with a same-level
        # neighbour is half this junction's and half the neighbour's, as the
        # tiles' own lines there are.
        share = 1.0
        if !child
            for k in 1:2
                e = o[k]
                decomp.active[e] || continue
                for (sd, at_end) in ((1, pt[k] == lo_box[k]),
                                     (2, pt[k] == lo_box[k] + n_box[k] - 1))
                    at_end && !lt.imposed[e][sd] && !lt.boundary[e][sd] && (share /= 2)
                end
            end
        end
        I = CartesianIndex(ntuple(e -> e == d ? (bnode > 0 ? bnode : pad[d] + 1) :
                                       e == o[1] ? i1 : i2, 3))
        ef = Float64(_edge_factor(ps, o[1], i1 - pad[o[1]], I)) *
             Float64(_edge_factor(ps, o[2], i2 - pad[o[2]], I))
        kap[l] = T(share * w * ef)
        if jin
            Ij = CartesianIndex(ntuple(e -> e == d ? ja : e == o[1] ? i1 : i2, 3))
            kapf[l] = T(share * w * ef * Float64(p.h[d]) / Float64(p.inv_J[Ij]))
        end
        bnode > 0 && (wb[l] = T(w * ef * quad_weight(ps, d, bnode - pad[d]) *
                                Float64(p.h[d]) / Float64(p.inv_J[I])))
    end
    bandp = ntuple(k -> k <= length(band) ? local_of(band[k], d) : 0, 6)
    omh = ntuple(k -> T(k <= length(om) ? om[k] * Float64(p.h[d]) : 0.0), 6)
    z() = zeros(T, nline, n_cons)
    return RefluxCapture{T,Matrix{T},Vector{T}}(
        ℓp, jn, child, d, T(sgn), nodes, dcoef, anchor(dr), pad8(drs),
        length(drs) ÷ 2, T(drsc), fcoef, anchor(fr), pad8(frs),
        length(frs) ÷ 2, T(frsc), T(p.h[d]), lines, kap, kapf, bnode, wb, bandp,
        omh, entry, window, wslots, zeros(Float64, nline, length(window)),
        zeros(T, nline, 12), z(), z(), z(), z())
end

# --- Capture hooks ----------------------------------------------------------------

_reflux_captures(solver::PatchSolver) = getfield(getfield(solver, :patch), :reflux_captures)
_reflux_captures(solver::Solver) = getfield(getfield(solver, :patches)[1], :reflux_captures)
_reflux_captures(solver) = ()

_reflux_raw(Q::ConservedState) = parent(Q)
_reflux_raw(Q) = Q

@inline _reflux_index(cap, a::Int, i1::Int, i2::Int) =
    cap.d == 1 ? CartesianIndex(a, i1, i2) :
    cap.d == 2 ? CartesianIndex(i1, a, i2) : CartesianIndex(i1, i2, a)

# A term the funnels subtract that is not a divergence: the radial
# momentum's pressure gradient along r, and the spherical θ-momentum's along
# θ, which take `inv_h[d]` for their factor. On the Cartesian grid that
# factor is one and the gradient is the divergence of the pressure.
_reflux_node_term(solver, d::Int, scale) =
    !(solver.metric isa CartesianMetric) && scale === solver.inv_h[d]

# Zero the stage fluxes of a patch, at the start of its right-hand side.
@noinline function _reflux_zero!(solver)
    for cap in _reflux_captures(solver)
        fill!(cap.stage, 0)
    end
    return nothing
end

# Inside `compute_rhs!`'s loop over components, each of which starts from a
# zeroed `dQ[:, c]` and takes only divergences and the radial pressure
# gradient, the child's rate of Ω is taken once per component from `dQ`
# itself (`_reflux_component!`), less the pressure gradient's, rather than
# from each divergence's increment: one read of Ω's nodes where every
# divergence along every dimension took two. A divergence outside the loop
# (the second phase of a level's right-hand side) takes its increment.
const REFLUX_DEFER = Ref(false)

@noinline function _reflux_defer!(solver, on::Bool)
    REFLUX_DEFER[] = on && !isempty(_reflux_captures(solver))
    return nothing
end

# The child's rate of Ω from component `c` of `dQ`, which the component's
# divergences have filled since it was zeroed.
@noinline function _reflux_component!(solver, dQ, c::Int)
    REFLUX_DEFER[] || return nothing
    q = _reflux_raw(dQ)
    inv_J = solver.inv_J
    @inbounds for cap in _reflux_captures(solver)
        cap.child || continue
        T = eltype(cap.stage)
        for (l, (i1, i2)) in enumerate(Iterators.product(cap.lines...))
            cap.entry[l] == 0 && continue
            Δ = zero(T)
            for k in 1:6
                b = cap.band[k]
                b == 0 && continue
                I = _reflux_index(cap, b, i1, i2)
                Δ += cap.omega[k] * q[I, c] / inv_J[I]
            end
            cap.stage[l, c] -= cap.kap[l] * Δ
        end
    end
    return nothing
end

# Before a divergence of component `c` along `d` enters `dQ`: the values at
# the captured nodes, and on the child's side at the nodes of Ω, whatever
# the dimension of the divergence, unless the component loop defers Ω and
# this is a divergence rather than the pressure gradient.
@noinline function _reflux_open!(solver, dQ, c::Int, d::Int, scale)
    caps = _reflux_captures(solver)
    isempty(caps) && return nothing
    node = _reflux_node_term(solver, d, scale)
    band = !REFLUX_DEFER[] || node
    q = _reflux_raw(dQ)
    @inbounds for cap in caps
        flux = cap.d == d && !node
        omega = cap.child && band
        (flux || omega) || continue
        for (l, (i1, i2)) in enumerate(Iterators.product(cap.lines...))
            for k in 1:6
                a = flux ? cap.nodes[k] : 0
                a == 0 || (cap.snap[l, k] = q[_reflux_index(cap, a, i1, i2), c])
                b = omega ? cap.band[k] : 0
                b == 0 || (cap.snap[l, 6 + k] = q[_reflux_index(cap, b, i1, i2), c])
            end
        end
    end
    return nothing
end

@inline _reflux_scale(::Nothing, I) = 1
@inline _reflux_scale(s::AbstractArray, I) = @inbounds s[I]

# After it: the face flux of each line, from the increments at the captured
# nodes and the explicit stencil of `f` at the anchor face, and on the
# child's side the divergence's rate of Ω, which leaves the stage. `scale`
# is the factor on the derivative (`inv_J`, `inv_h[d]`, or `nothing` for
# one). A term that is not a divergence enters neither: it is a node term of
# the budget, as a source is.
@noinline function _reflux_close!(solver, dQ, c::Int, f, d::Int, scale)
    caps = _reflux_captures(solver)
    isempty(caps) && return nothing
    node = _reflux_node_term(solver, d, scale)
    # Under the component loop's deferral the pressure gradient's increment
    # is in the `dQ` the component's Ω is taken from, so it is returned here.
    node && !REFLUX_DEFER[] && return nothing
    q = _reflux_raw(dQ)
    inv_J = solver.inv_J
    @inbounds for cap in caps
        T = eltype(cap.stage)
        if node
            cap.child || continue
            for (l, (i1, i2)) in enumerate(Iterators.product(cap.lines...))
                cap.entry[l] == 0 && continue
                Δ = zero(T)
                for k in 1:6
                    b = cap.band[k]
                    b == 0 && continue
                    I = _reflux_index(cap, b, i1, i2)
                    Δ += cap.omega[k] * (q[I, c] - cap.snap[l, 6 + k]) / inv_J[I]
                end
                cap.stage[l, c] += cap.kap[l] * Δ
            end
            continue
        end
        if cap.child && !REFLUX_DEFER[]
            for (l, (i1, i2)) in enumerate(Iterators.product(cap.lines...))
                cap.entry[l] == 0 && continue
                Δ = zero(T)
                for k in 1:6
                    b = cap.band[k]
                    b == 0 && continue
                    I = _reflux_index(cap, b, i1, i2)
                    Δ += cap.omega[k] * (q[I, c] - cap.snap[l, 6 + k]) / inv_J[I]
                end
                cap.stage[l, c] -= cap.kap[l] * Δ
            end
        end
        cap.d == d || continue
        for (l, (i1, i2)) in enumerate(Iterators.product(cap.lines...))
            cap.entry[l] == 0 && continue
            v = zero(T)
            for k in 1:6
                a = cap.nodes[k]
                (a == 0 || iszero(cap.dcoef[k])) && continue
                I = _reflux_index(cap, a, i1, i2)
                G = -(q[I, c] - cap.snap[l, k]) * cap.hd / _reflux_scale(scale, I)
                v += cap.dcoef[k] * G
            end
            if cap.dr > 0
                R = zero(T)
                for m in 1:2cap.dM
                    R += cap.drs[m] * f[_reflux_index(cap, cap.dr + m - cap.dM, i1, i2)]
                end
                v += cap.drscale * R
            end
            cap.stage[l, c] += cap.sgn * cap.kap[l] * v
        end
    end
    return nothing
end

# The stage update of patch `p`: the stage fluxes through the recurrence.
@noinline function _reflux_fold!(solver, p, dQ, A, B, dt)
    for cap in p.reflux_captures
        @. cap.du = A * cap.du + dt * cap.stage
        @. cap.reg += B * cap.du
    end
    return nothing
end

# One filter pass of component `c` along `d`: `q` before it, `filtered` the
# filtered values, `w` the blending weight; the pass's change at each line's
# junction enters the register directly.
@noinline function _reflux_filter!(solver, q, filtered, c::Int, d::Int, w)
    caps = _reflux_captures(solver)
    isempty(caps) && return nothing
    @inbounds for cap in caps
        T = eltype(cap.reg)
        p = _patch_of(solver)
        for (l, (i1, i2)) in enumerate(Iterators.product(cap.lines...))
            (cap.entry[l] == 0 || iszero(cap.kapf[l])) && continue
            # The pass's change of Ω, along any dimension.
            if cap.child
                Δ = zero(T)
                for k in 1:6
                    a = cap.band[k]
                    a == 0 && continue
                    I = _reflux_index(cap, a, i1, i2)
                    Δ += cap.omega[k] * (w * (filtered[I] - q[I])) / p.inv_J[I]
                end
                cap.reg[l, c] -= cap.kap[l] * Δ
            end
            cap.d == d || continue
            v = zero(T)
            for k in 1:6
                a = cap.nodes[k]
                (a == 0 || iszero(cap.fcoef[k])) && continue
                I = _reflux_index(cap, a, i1, i2)
                v += cap.fcoef[k] * (w * (filtered[I] - q[I]))
            end
            if cap.fr > 0
                ψ = zero(T)
                for m in 1:2cap.fM
                    ψ += cap.frs[m] * q[_reflux_index(cap, cap.fr + m - cap.fM, i1, i2)]
                end
                v += cap.frscale * w * ψ
            end
            # The pass adds +Ψ differences where a divergence subtracts.
            cap.reg[l, c] -= cap.sgn * cap.kapf[l] * v
        end
    end
    return nothing
end

# --- The gate -----------------------------------------------------------------------

# A junction line is corrected in a step when the density's undivided fourth
# difference along it exceeds `CHILD_MASK_THRESHOLD` times the density, the
# test of the parent's filter and derivative masks, at a parent node within
# `GATE_REACH` nodes of the face, at the step's start or its end. A feature
# the parent spacing does not resolve, which is what the masks drop and what
# crossing the face lost mass on, is then carried conservatively, and smooth
# flow is left as the uncorrected coupling advances it. A feature deep inside
# the child leaves the face to the uncorrected coupling too: on the cold
# inflow ahead of a Noh shock the tile holds, corrections there took the
# pre-shock density error from 2e-4 to 1e-2.
const GATE_REACH = 8

# The gate's nodes of a junction line: the tested ones and two either side.
const GATE_WIDTH = 2GATE_REACH + 5

"""
Benchmark toggle: `false` corrects every junction line in every step, the
form `bench/reflux.jl` measures the gate against.
"""
const REFLUX_GATED = Ref(true)

# The density at the gate's nodes a parent capture holds, per line. The test
# itself is taken on every rank from these values summed over the level's
# ranks, so that it reads no halo: at the step's start and after the
# restriction a decomposed parent's halos are not current.
function _reflux_density!(out::Matrix{Float64}, cap, Q, n_species::Int)
    q = _reflux_raw(Q)
    for (l, (i1, i2)) in enumerate(Iterators.product(cap.lines...))
        cap.entry[l] == 0 && continue
        for (k, a) in enumerate(cap.window)
            I = _reflux_index(cap, a, i1, i2)
            s = 0.0
            for sp in 1:n_species
                s += Float64(q[I, sp])
            end
            out[l, k] = s
        end
    end
    return out
end

# Whether the gate flags a junction line, from its slots in the reduced
# buffer at offset `o`: the summed densities at the step's start and end and
# the count of patches holding each node, two where same-level tiles share
# it. A node is tested where all five of its taps are held.
function _reflux_gated(buf, o::Int)
    W = GATE_WIDTH
    thr = CHILD_MASK_THRESHOLD[]
    for pass in 0:1, s in 3:W-2
        all(t -> buf[o + 2W + s + t] > 0, -2:2) || continue
        ρ(t) = buf[o + pass * W + s + t] / buf[o + 2W + s + t]
        r0 = ρ(0)
        δ4 = ρ(-2) - 4ρ(-1) + 6r0 - 4ρ(1) + ρ(2)
        abs(δ4) > thr * abs(r0) && return true
    end
    return false
end

# The connected runs of flagged lines in a junction's box of n1 × n2 lines,
# four-connected; `flagged[k]` for line k = i1 + n1 (i2 − 1).
function _flagged_runs(flagged::BitVector, n1::Int, n2::Int)
    runs = Vector{Vector{Int}}()
    seen = falses(length(flagged))
    for k0 in eachindex(flagged)
        (flagged[k0] && !seen[k0]) || continue
        run = Int[]
        stack = [k0]
        seen[k0] = true
        while !isempty(stack)
            k = pop!(stack)
            push!(run, k)
            i1, i2 = mod1(k, n1), (k - 1) ÷ n1 + 1
            for (j1, j2) in ((i1 - 1, i2), (i1 + 1, i2), (i1, i2 - 1), (i1, i2 + 1))
                (1 <= j1 <= n1 && 1 <= j2 <= n2) || continue
                kk = j1 + n1 * (j2 - 1)
                (flagged[kk] && !seen[kk]) || continue
                seen[kk] = true
                push!(stack, kk)
            end
        end
        push!(runs, run)
    end
    return runs
end

# --- Step boundaries --------------------------------------------------------------

# Ω of a child capture's lines over the state `Q`, per line and component.
function _reflux_omega!(out, cap, p, Q)
    fill!(out, 0)
    q = _reflux_raw(Q)
    n_cons = size(out, 2)
    for (l, (i1, i2)) in enumerate(Iterators.product(cap.lines...))
        cap.entry[l] == 0 && continue
        for k in 1:6
            a = cap.band[k]
            a == 0 && continue
            I = _reflux_index(cap, a, i1, i2)
            w = cap.kap[l] * cap.omega[k] / p.inv_J[I]
            for c in 1:n_cons
                out[l, c] += w * q[I, c]
            end
        end
    end
    return out
end

"""
    _reflux_begin_step!(solver, states)

At the start of a root step: rebuild the captures if the layout changed,
empty every register, and take Ω of the state the step starts from.
Rank-local.
"""
_reflux_begin_step!(solver, Q) = nothing
function _reflux_begin_step!(solver::Solver, states::Vector{<:ConservedState})
    _reflux_current!(solver)
    held = _reflux_carry(solver)
    # A retry of the step starts from what was held back before the attempt.
    if held.step == solver.step
        for (k, v) in held.saved
            copyto!(get!(() -> similar(v), held.carry, k), v)
        end
    else
        held.step = solver.step
        empty!(held.saved)
        for (k, v) in held.carry
            held.saved[k] = copy(v)
        end
    end
    ns = solver.equations.n_species
    for (pi, p) in enumerate(getfield(solver, :patches))
        _reflux_reset!(p.reflux_captures, states[pi], ns)
    end
    return nothing
end

# A patch's registers emptied and its gate densities taken (a function
# barrier, as the held patches differ in type).
function _reflux_reset!(caps, Q, ns)
    for cap in caps
        fill!(cap.du, 0)
        fill!(cap.reg, 0)
        _reflux_density!(cap.rho0, cap, Q, ns)
    end
    return nothing
end

"""
    _reflux_apply!(solver, states, ℓp)

Correct the face nodes of parent level `ℓp` (1-based) for the step its
junctions have carried: each junction line's registers and the change of Ω
summed over the level's communicator, divided by the face node's composite
weight, added to the parent's state, and the registers emptied. Collective
over the parent level's subset; a rank outside it returns.
"""
_reflux_apply!(solver, Q, ℓp::Int) = nothing
function _reflux_apply!(solver::Solver{T}, states::Vector{<:ConservedState},
                        ℓp::Int) where {T}
    levels = getfield(solver, :levels)
    ℓp < length(levels) || return nothing
    lev = levels[ℓp]
    lev.level_comm.owned || return nothing
    patches = getfield(solver, :patches)
    _reflux_supported(solver) || return nothing
    junctions = _reflux_junctions(solver, ℓp)
    isempty(junctions) && return nothing
    n_cons = solver.equations.n_cons
    ns = solver.equations.n_species
    # Per junction line: its components' registers, the correction node's
    # weight, and the gate's densities at the step's start and end with the
    # count of patches holding each node, summed over the level's ranks.
    W = GATE_WIDTH
    stride = n_cons + 1 + 3W
    base = cumsum([0; [_reflux_lines(j.box) for j in junctions]])
    buf = zeros(Float64, base[end] * stride)
    at(jn, line, k) = (base[jn] + line - 1) * stride + k
    for (pi, p) in enumerate(patches)
        _reflux_gather!(buf, p.reflux_captures, states[pi], ℓp, ns, n_cons, base, stride)
    end
    t0 = time_ns()
    MPI.Allreduce!(buf, +, lev.level_comm.comm)
    _wait!(solver, t0)
    held = _reflux_carry(solver)
    # One binding: the comprehension below captures it.
    carry = _carry_vector!(held, ℓp, base[end] * n_cons)
    cat(jn, k, c) = (base[jn] + k - 1) * n_cons + c
    # Each connected run of flagged lines, or of lines holding a correction
    # back, takes its summed registers as one change of state over the run's
    # correction nodes: on a line alone in its run this is the line's own
    # correction, and across a face it keeps the tangential difference of the
    # two grids' quadratures, which telescopes along the run, off any single
    # node.
    runs = Tuple{Int,Vector{Int},Float64}[]
    change = zeros(Float64, base[end] * n_cons)
    for (jn, jc) in enumerate(junctions)
        n1, n2 = length(jc.box[1]), length(jc.box[2])
        flagged = BitVector([!REFLUX_GATED[] || _reflux_gated(buf, at(jn, k, n_cons + 1)) ||
                             any(c -> carry[cat(jn, k, c)] != 0, 1:n_cons)
                             for k in 1:n1*n2])
        for run in _flagged_runs(flagged, n1, n2)
            Wr = sum(buf[at(jn, k, n_cons + 1)] for k in run)
            Wr > 0 || continue
            push!(runs, (jn, run, Wr))
            for c in 1:n_cons
                E = sum(buf[at(jn, k, c)] + carry[cat(jn, k, c)] for k in run)
                for k in run
                    change[cat(jn, k, c)] = E / Wr
                end
            end
        end
    end
    # The positivity guard: each run takes the largest fraction of its
    # change, of 1, 1/2, 1/4 .. 2⁻¹⁰ or none, that leaves every node it
    # corrects at more than half its density and internal energy per volume,
    # agreed over the level's ranks, and holds the rest back for the next
    # step. Over the steps the correction is the same; a strong shock leaving
    # a tile into cold gas otherwise took the node ahead of it negative.
    θs = ones(Float64, length(runs))
    run_of = Dict{Tuple{Int,Int},Int}()
    for (r, (jn, run, _)) in enumerate(runs), k in run
        run_of[(jn, k)] = r
    end
    eq = solver.equations
    for (pi, p) in enumerate(patches)
        _reflux_guard!(θs, run_of, p.reflux_captures, parent(states[pi]), eq, change,
                       base, n_cons, ℓp)
    end
    t1 = time_ns()
    MPI.Allreduce!(θs, min, lev.level_comm.comm)
    _wait!(solver, t1)
    fill!(carry, 0)
    for (r, (jn, run, _)) in enumerate(runs), k in run, c in 1:n_cons
        i = cat(jn, k, c)
        # What a line holds back is its share of the run's, by its weight.
        carry[i] = (1 - θs[r]) * change[i] * buf[at(jn, k, n_cons + 1)]
        change[i] *= θs[r]
    end
    for (pi, p) in enumerate(patches)
        _reflux_change!(p.reflux_captures, states[pi], change, base, n_cons, ℓp, ns)
    end
    return nothing
end

# The three per-patch passes of `_reflux_apply!`, each a function barrier,
# since the held patches differ in type. Each line's registers, its
# correction node's weight and the gate's densities into `buf`:
function _reflux_gather!(buf, caps, Q, ℓp, ns, n_cons, base, stride)
    W = GATE_WIDTH
    at(jn, line, k) = (base[jn] + line - 1) * stride + k
    rho1 = Matrix{Float64}(undef, 0, 0)
    for cap in caps
        cap.level == ℓp || continue
        if !cap.child
            size(rho1) == size(cap.rho0) || (rho1 = similar(cap.rho0))
            _reflux_density!(rho1, cap, Q, ns)
        end
        @inbounds for l in eachindex(cap.entry)
            e = cap.entry[l]
            e == 0 && continue
            for c in 1:n_cons
                buf[at(cap.junction, e, c)] += Float64(cap.reg[l, c])
            end
            cap.child && continue
            cap.bnode > 0 && (buf[at(cap.junction, e, n_cons + 1)] += Float64(cap.wb[l]))
            o = at(cap.junction, e, n_cons + 1)
            for (k, s) in enumerate(cap.wslots)
                buf[o + s] += cap.rho0[l, k]
                buf[o + W + s] += rho1[l, k]
                buf[o + 2W + s] += 1
            end
        end
    end
    return buf
end

# the positivity guard's fraction of each run at the nodes this patch holds:
function _reflux_guard!(θs, run_of, caps, q, eq, change, base, n_cons, ℓp)
    for cap in caps
        (cap.level == ℓp && !cap.child && cap.bnode > 0) || continue
        for (l, (i1, i2)) in enumerate(Iterators.product(cap.lines...))
            e = cap.entry[l]
            r = get(run_of, (cap.junction, e), 0)
            (e == 0 || r == 0) && continue
            I = _reflux_index(cap, cap.bnode, i1, i2)
            o = (base[cap.junction] + e - 1) * n_cons
            θs[r] = min(θs[r], _reflux_admissible_fraction(q, I, eq, change, o))
        end
    end
    return θs
end

# and the change itself, the registers emptied and the next step's densities.
function _reflux_change!(caps, Q, change, base, n_cons, ℓp, ns)
    q = parent(Q)
    for cap in caps
        cap.level == ℓp || continue
        T = eltype(cap.reg)
        if !cap.child && cap.bnode > 0
            @inbounds for (l, (i1, i2)) in enumerate(Iterators.product(cap.lines...))
                e = cap.entry[l]
                e == 0 && continue
                I = _reflux_index(cap, cap.bnode, i1, i2)
                o = (base[cap.junction] + e - 1) * n_cons
                for c in 1:n_cons
                    q[I, c] += T(change[o + c])
                end
            end
        end
        fill!(cap.du, 0)
        fill!(cap.reg, 0)
        cap.child || _reflux_density!(cap.rho0, cap, Q, ns)
    end
    return nothing
end

# Parent level `ℓp`'s held-back corrections, `n` long, emptied if the
# layout's line count changed.
function _carry_vector!(held::RefluxCarry, ℓp::Int, n::Int)
    v = get(held.carry, ℓp, nothing)
    (v === nothing || length(v) != n) && (v = held.carry[ℓp] = zeros(Float64, n))
    return v
end

# The largest of 1, 1/2, .., 2⁻¹⁰ and 0 by which the change `change[o + c]`
# at node `I` keeps its density and its internal energy per volume
# ρe = E − ½|m|²/ρ above half their values, or, where ρe is not positive,
# from falling.
function _reflux_admissible_fraction(q, I, eq, change, o)
    ns = eq.n_species
    m1, m2, m3 = eq.i_mom
    ie = eq.i_energy
    ρ0 = sum(Float64(q[I, sp]) for sp in 1:ns)
    mom0 = (Float64(q[I, m1]), Float64(q[I, m2]), Float64(q[I, m3]))
    E0 = Float64(q[I, ie])
    ρe0 = E0 - 0.5 * sum(abs2, mom0) / ρ0
    dρ = sum(change[o + sp] for sp in 1:ns)
    dm = (change[o + m1], change[o + m2], change[o + m3])
    dE = change[o + ie]
    θ = 1.0
    for _ in 0:10
        ρ = ρ0 + θ * dρ
        if ρ > 0.5ρ0
            ρe = E0 + θ * dE - 0.5 * sum(abs2, mom0 .+ θ .* dm) / ρ
            (ρe0 > 0 ? ρe > 0.5ρe0 : ρe >= ρe0) && return θ
        end
        θ /= 2
    end
    return 0.0
end

# --- The conserved quadrature -------------------------------------------------------

# Ω of the junctions a child patch holds, added to its budget channels
# (`_local_conserved_budget`); nothing on a patch without captures.
_junction_budget!(out, solver, Q) = out
function _junction_budget!(out, solver::PatchSolver, Q)
    p = getfield(solver, :patch)
    isempty(p.reflux_captures) && return out
    eq = solver.equations
    nsp = eq.n_species
    comps = Int[1:nsp; collect(eq.i_mom); eq.i_energy]
    for cap in p.reflux_captures
        cap.child || continue
        om = _reflux_omega!(similar(cap.om0), cap, p, Q)
        for l in eachindex(cap.entry), (b, c) in enumerate(comps)
            cap.entry[l] == 0 || (out[b] += Float64(om[l, c]))
        end
    end
    return out
end

# Ω of a scalar field over the junctions a child patch holds, the term the
# composite `volume_integral` adds to its cells; `f === nothing` takes the
# field one, for `domain_volume`, where Ω is nonzero only on a metric whose
# Jacobian varies across the face.
_junction_integral(solver, f) = 0.0
function _junction_integral(solver::PatchSolver, f)
    p = getfield(solver, :patch)
    acc = 0.0
    for cap in p.reflux_captures
        cap.child || continue
        for (l, (i1, i2)) in enumerate(Iterators.product(cap.lines...))
            cap.entry[l] == 0 && continue
            for k in 1:6
                a = cap.band[k]
                a == 0 && continue
                I = _reflux_index(cap, a, i1, i2)
                v = f === nothing ? 1.0 : Float64(f[I])
                acc += Float64(cap.kap[l]) * Float64(cap.omega[k]) * v /
                       Float64(p.inv_J[I])
            end
        end
    end
    return acc
end

# The composite budget in the conserved quadrature, which `_conserved_budget`
# takes on a composite solver.
_reflux_budget(solver::Solver, states) = _conserved_budget(solver, states)
