# The per-point bodies of the positivity limiter. The header of
# positivity_setup.jl describes the method.

# --- Per-point bodies --------------------------------------------------------

# The subtraction of a divergence from component `c` of dQ, as
# `_subtract_div_point!` and `_subtract_jac_div_point!` make it, and the
# register update r ← A r + dt div from the same read over the interior.
# RKA[1] = 0, and the first stage assigns, so a register left non-finite by an
# abandoned step is forgotten.
@inline function _limiter_subtract_point!(dQ, r, div, inv_J, c, A, dt, o1, o2, o3,
                                          i, j, k)
    @inbounds begin
        I = CartesianIndex(i + o1, j + o2, k + o3)
        x = div[I]
        if inv_J === nothing
            dQ[I, c] -= x
        else
            dQ[I, c] -= inv_J[I] * x
        end
        r[I] = ifelse(iszero(A), dt * x, A * r[I] + dt * x)
    end
    return nothing
end

# The face register at the global low face of each line, Φ ← A Φ + dt F̂:
# the point flux of node 1 on a closed line (the wall-corrected flux), and on
# a periodic line (`periodic`) the constant the interior face relation gives
# the running sum of this stage's divergence, measured at local face `j` of the
# rank holding the global low face; an open end takes the same measurement
# (`_limiter_open_face`), its end face at node 1. At a fold the mirrored face
# relation at face 0,
# Σ_{|s|≤q} l_s F̂_s = Σ_m c_m Σ_{l=1−m}^{m} f_l with F̂_{−s} = σf F̂_s and the
# halo holding the mirror f_{1−l} = σf f_l, gives zero for an odd flux and,
# for an even one, (Σ_m c_m Σ_l f_l − 2 Σ_s l_s S_s) / (1 + 2 Σ_s l_s).
@inline function _limiter_anchor_point!(anchor, F, div, W, A, dt, c, d, periodic,
                                        fold, σf, j, lhs, rhs, o1, o2, o3, a, b, _)
    @inbounds begin
        o = d == 1 ? o1 : d == 2 ? o2 : o3
        if fold
            T = eltype(anchor)
            value = zero(T)
            if σf > 0
                S = zero(T)
                acc = zero(T)
                sum_l = one(T)
                for s in eachindex(lhs)
                    S += W[s+o] * div[_line_node(d, s, a, b, o1, o2, o3)]
                    acc += 2 * lhs[s] * S
                    sum_l += 2 * lhs[s]
                end
                G = zero(T)
                for m in eachindex(rhs)
                    for l in 1-m:m
                        G += rhs[m] * F[_line_node(d, l, a, b, o1, o2, o3)]
                    end
                end
                value = (G - acc) / sum_l
            end
        elseif periodic
            q = length(lhs)
            T = eltype(anchor)
            S = zero(T)
            lo = j - q
            acc = zero(T)
            sum_l = one(T)
            for s in 1:q
                sum_l += 2 * lhs[s]
            end
            for p in 1:j+q
                S += W[p+o] * div[_line_node(d, p, a, b, o1, o2, o3)]
                if p >= lo
                    s = p - j
                    acc += (s == 0 ? one(T) : lhs[abs(s)]) * S
                end
            end
            # The face j − q may be face 0, whose running sum is zero.
            G = zero(T)
            for m in eachindex(rhs)
                for l in 1-m:m
                    G += rhs[m] * F[_line_node(d, j + l, a, b, o1, o2, o3)]
                end
            end
            value = (G - acc) / sum_l
        else
            value = F[_line_node(d, 1, a, b, o1, o2, o3)]
        end
        anchor[c, a, b] = ifelse(iszero(A), dt * value, A * anchor[c, a, b] + dt * value)
    end
    return nothing
end

# The face register at the high face of a line ending at a patch interface,
# Φ ← A Φ + dt F_n, the point flux of its last node, as the low face's is of
# node 1: the patch across the interface holds the same node as its node 1.
@inline function _limiter_end_anchor_point!(anchor, F, A, dt, c, d, n, o1, o2, o3,
                                            a, b, _)
    @inbounds begin
        value = F[_line_node(d, n, a, b, o1, o2, o3)]
        anchor[c, a, b] = ifelse(iszero(A), dt * value, A * anchor[c, a, b] + dt * value)
    end
    return nothing
end

# The stage's face value at the high face of a line ending at a patch
# interface: B_k times its register, in place of the running sum's end, which
# meets it to round-off. Both patches then take one value at the shared node's
# face, bit for bit where their point fluxes agree.
@inline function _limiter_end_face_point!(faces, anchor, scale, n_comp, n, d, o1, o2,
                                          o3, a, b, _)
    @inbounds begin
        n1, n2 = size(faces[1], 1), size(faces[1], 2)
        L = _line_linear(d, n, a, b, o1, o2, o3, n1, n2)
        for c in 1:n_comp
            faces[c][L] = scale * anchor[c, a, b]
        end
    end
    return nothing
end

# One line's local running sums into the face slots: slot p holds face p + ½,
# slot 0 (a halo node) the low face of the block, at zero. Mode 1 sums W r,
# mode 2 the filter correction −ω Δ with ω = W/h, and in mode 2 the filter's
# face relation is measured at local face `j` when j > 0:
# (Σ_s l_s S_{j+s} + w ψ_j) / Σ_s l_s. Where `volume` is set, a component
# marked in `mask` sums −ω J Δ instead. On a line this rank holds whole
# (`ends[1]`) the offsets are the line's own, the stage's face register
# `anchor` or the filter's measured constant, and the face values are
# finished here as `_limiter_offset_point!` finishes them on a decomposed one,
# scaled by `ends[3]`, the global low face's value at the high face of a
# periodic line (`ends[2]`): a stage's in the same sweep as the sums, a
# filter pass's after its measurement. A pass that dropped a child's residual
# at the nodes `cmask` marks (`_child_mask!`; `nothing` otherwise) measures the
# relation relative to the line's middle face, ψ_j less χ_j, the dropped
# residual summed from the middle face to face j, and leaves this rank's sums
# of it, over its part of the line and up to the middle face, in `dropped` and
# `dropped_mid` for the offsets of a decomposed line (`_dropped_rows`).
@inline function _limiter_scan_point!(faces, regs, Q, base, W, hd, totals, aux, anchor,
                                      n_cons, n, d, mode, j, lhs, psi, wf, volume, mask,
                                      inv_J, cmask, drop, dropped, dropped_mid, ends,
                                      o1, o2, o3, _, a, b)
    @inbounds begin
        o = d == 1 ? o1 : d == 2 ? o2 : o3
        T = eltype(totals)
        whole, wrap_high, scale = ends
        n1, n2 = size(inv_J, 1), size(inv_J, 2)
        sd = _line_stride(d, n1, n2)
        L0 = _line_linear(d, 0, a, b, o1, o2, o3, n1, n2)
        for c in 1:n_cons
            fc = faces[c]
            S = zero(T)
            weighted = volume && mask[c]
            if mode == 1
                rc = regs[c]
                if whole
                    off = anchor[c, a, b]
                    fc[L0] = scale * ((off + S) - zero(T))
                    for p in 1:n
                        L = L0 + p * sd
                        S += W[p+o] * rc[L]
                        fc[L] = scale * ((off + S) - zero(T))
                    end
                    wrap_high && (fc[L0+n*sd] = scale * (off - zero(T)))
                    totals[c, a, b] = S
                    continue
                end
                fc[L0] = S
                for p in 1:n
                    L = L0 + p * sd
                    S += W[p+o] * rc[L]
                    fc[L] = S
                end
            else
                bc = base[c]
                fc[L0] = S
                for p in 1:n
                    L = L0 + p * sd
                    if weighted
                        S -= (W[p+o] / hd) / inv_J[L] * (Q[L, c] - bc[L])
                    else
                        S -= (W[p+o] / hd) * (Q[L, c] - bc[L])
                    end
                    fc[L] = S
                end
            end
            totals[c, a, b] = S
            χ = zero(T)
            if mode == 2 && cmask !== nothing
                r, Md, row_lo, row_hi, mid = drop
                X = zero(T)
                Y = zero(T)
                bc = base[c]
                for p in row_lo:row_hi
                    L = L0 + p * sd
                    _fully_covered(cmask[L]) || continue
                    res = r[1] * bc[L]
                    for m in 1:Md
                        res += r[m+1] * (bc[L+m*sd] + bc[L-m*sd])
                    end
                    X += res
                    p <= j && (χ += res)
                    p <= mid && (Y += res)
                end
                dropped[c, a, b] = X
                dropped_mid[c, a, b] = Y
                # Relative to the line's middle face, where a line one rank
                # holds measures the relation: its χ is then zero.
                χ -= Y
            end
            if mode == 2 && j > 0
                q = length(lhs)
                Lj = L0 + j * sd
                acc = fc[Lj]
                sum_l = one(T)
                for s in 1:q
                    acc += lhs[s] * (fc[Lj-s*sd] + fc[Lj+s*sd])
                    sum_l += 2 * lhs[s]
                end
                M = length(psi) ÷ 2
                ψ = zero(T)
                bc = base[c]
                for l in 1-M:M
                    ψ += psi[l+M] * bc[Lj+l*sd]
                end
                aux[c, a, b] = cmask === nothing ? (acc + wf * ψ) / sum_l :
                                                   (acc + wf * (ψ - χ)) / sum_l
            end
            whole || continue
            # A filter pass: no offset, and the measured constant removed.
            off = zero(T)
            cstar = aux[c, a, b]
            for f in 0:n
                L = L0 + f * sd
                v = wrap_high && f == n ? off : off + fc[L]
                fc[L] = weighted ? scale * v : scale * (v - cstar)
            end
        end
    end
    return nothing
end

# Face f = f1 − 1 of a line: scale · ((O + S_f) − c*). On the last rank of a
# periodic line the high face is the global low face, whose value the first
# rank holds. A component the scan weighted by J starts from zero at the fold
# and takes no constant.
@inline function _limiter_offset_point!(faces, offset, wrap, cstar, scale, wrap_high,
                                        n, d, n_cons, volume, mask, o1, o2, o3, f1, a, b)
    @inbounds begin
        f = f1 - 1
        I = _line_node(d, f, a, b, o1, o2, o3)
        for c in 1:n_cons
            v = wrap_high && f == n ? wrap[c, a, b] : offset[c, a, b] + faces[c][I]
            faces[c][I] = volume && mask[c] ? scale * v : scale * (v - cstar[c, a, b])
        end
    end
    return nothing
end

# Mixture density, momentum and total energy of a state.
Base.@propagate_inbounds function _limiter_state(Q, I, lay)
    ns, m1, m2, m3, ie, _ = lay
    ρ = zero(eltype(Q[I, 1]))
    for sp in 1:ns
        ρ += Q[I, sp]
    end
    return (ρ, Q[I, m1], Q[I, m2], Q[I, m3], Q[I, ie])
end

Base.@propagate_inbounds function _limiter_face_state(faces, I, lay)
    ns, m1, m2, m3, ie, _ = lay
    ρ = zero(eltype(faces[1]))
    for sp in 1:ns
        ρ += faces[sp][I]
    end
    return (ρ, faces[m1][I], faces[m2][I], faces[m3][I], faces[ie][I])
end

# q + s G, componentwise over the five aggregated values.
@inline _limiter_axpy(q, s, G) = map((x, y) -> x + s * y, q, G)

@inline _limiter_internal(q) = q[5] - (q[2] * q[2] + q[3] * q[3] + q[4] * q[4]) / (2 * q[1])
# ρ > 0 and ρe > 0.
@inline _limiter_positive(q) =
    q[1] > 0 && 2 * q[1] * q[5] - (q[2] * q[2] + q[3] * q[3] + q[4] * q[4]) > 0
# Whether q holds a cell's bound ε (`_limiter_cell_bound`): ρ ≥ ε[1] and ρe at
# least ε[3], the bound less the tolerance where the floor sets it, the second
# multiplied through by 2ρ > 0.
@inline _limiter_admissible(q, ε) =
    q[1] >= ε[1] &&
    2 * q[1] * q[5] - (q[2] * q[2] + q[3] * q[3] + q[4] * q[4]) >= 2 * q[1] * ε[3]

# τ times the Lax–Friedrichs flux of the aggregated state along d between
# nodes L and R, whose wave speed is the larger of the two.
@inline function _limiter_lf_state(qL, qR, uL, uR, pL, pR, a, τ, d)
    fL = (qL[1] * uL, qL[2] * uL + (d == 1) * pL, qL[3] * uL + (d == 2) * pL,
          qL[4] * uL + (d == 3) * pL, (qL[5] + pL) * uL)
    fR = (qR[1] * uR, qR[2] * uR + (d == 1) * pR, qR[3] * uR + (d == 2) * pR,
          qR[4] * uR + (d == 3) * pR, (qR[5] + pR) * uR)
    return map((x, y, u, v) -> τ * ((x + y) / 2 - a * (v - u) / 2), fL, fR, qL, qR)
end

# The limit of one cell side: the largest θ for which q + s (θ G + (1 − θ) G_L)
# keeps ρ and ρe at or above ε, by linear interpolation in ρ and then in ρe
# along the shortened segment, sufficient since ρe is concave. Returns θ and
# whether the side is guaranteed. A half state holds the bound as
# `_limiter_admissible` tests it, but θ is interpolated to ε[2], the bound
# itself, so that the limited write rounds to within the tolerance of it; where
# q_L lies within the tolerance under the bound, θ is interpolated to q_L's own
# ρe.
@inline function _limiter_side(q, G, GL, s, ε)
    T = typeof(s)
    qH = _limiter_axpy(q, s, G)
    _limiter_admissible(qH, ε) && return (one(T), true)
    qL = _limiter_axpy(q, s, GL)
    _limiter_admissible(qL, ε) || return (zero(T), false)
    θ = qH[1] < ε[1] ? (qL[1] - ε[1]) / (qL[1] - qH[1]) : one(T)
    qθ = _limiter_axpy(qL, θ, map(-, qH, qL))
    eL = _limiter_internal(qL)
    e = _limiter_internal(qθ)
    target = max(min(ε[2], eL), ε[3])
    e < target && (θ *= (eL - target) / (eL - e))
    return (clamp(θ, zero(T), one(T)), true)
end

# Σ_d (|u_d| + c)/W_d at the node of linear index L and Cartesian index I, the
# cell's total first-order rate, from the inverse weights.
Base.@propagate_inbounds function _limiter_rate(uvw, c, inv_W, act, L, I)
    Λ = zero(eltype(c))
    for e in 1:3
        act[e] || continue
        Λ += (abs(uvw[e][L]) + c[L]) * inv_W[e][I[e]]
    end
    return Λ
end

# The nodes of a parent patch deep inside a child, which its restriction
# overwrites after every step (the second grade of `Patch.overwritten`), which
# the limiter does not hold. Held to their cell bounds stage after stage,
# while `max_rate` let the parent's step run there beyond the first-order
# bound, they fell to temperatures of 1e-12, at which κ* = ρc/T set the root
# step. Their neighbours' first-order fluxes do not read them where they are
# not admissible. Empty where no node is exempt.
_limiter_exempt(solver) = similar(solver.tmp_a, eltype(solver.tmp_a), 0, 0, 0)
_limiter_exempt(ps::PatchSolver) =
    isempty(ps.patch.amr.overwritten) ?
    similar(ps.tmp_a, eltype(ps.tmp_a), 0, 0, 0) : ps.patch.amr.overwritten

# A node is held unless a face condition overwrites it or it is exempt
# (`_limiter_exempt`).
Base.@propagate_inbounds _limiter_constrained(free, I) =
    !(free[1][I[1]] | free[2][I[2]] | free[3][I[3]]) &&
    (isempty(free[4]) || free[4][I] <= 1)

# The half-state scale of a cell along d: 2Λ/(|u_d| + c) at a stage, its rate
# over |u_d| + c along an area-weighted direction, Inf where that speed
# vanishes, and 2/ω = 2h/W in a filter pass.
Base.@propagate_inbounds function _limiter_scale(uvw, c, inv_W, act, rates, d, hd, mode,
                                                 areal, L, I)
    T = eltype(c)
    if mode == 1
        sp = abs(uvw[d][L]) + c[L]
        sp > 0 || return T(Inf)
        return areal ? rates[L] / sp : 2 * _limiter_rate(uvw, c, inv_W, act, L, I) / sp
    end
    return 2 * hd * inv_W[d][I[d]]
end

# θ at each face of a line of direction d into `theta`, the face f in place of
# node f, and the line's tallies into `counts` (see `_limiter_tally!`), an
# unguaranteed side flagged 1 for the low cell and 2 for the high one. A global
# closed end is not limited; a fold's face is, from node 1's side alone, its
# other cell node 1's mirror. Mode 1 is a stage (half states scaled by
# 2Λ/(|u_d| + c) and the first-order flux τ F_LF); mode 2 a filter pass (2/ω,
# toward zero). Each cell keeps its own bound (`_limiter_cell_bound`). Under
# `volume` a cell takes the components marked in `vmask` divided by its J: at a
# stage along an area-weighted direction (`areal`) every component, whose
# face values are A_d times a flux on a line of constant A_d = J, and in a
# filter pass on a radial grid the components weighted by J. An areal stage
# reads each cell's rate from `rates`, which the radial pass fills. A node's
# state, bound, half-state scale s (s_d = 2Λ/(|u_d| + c) at a stage, the low
# cell taking −s) and constraint serve both its faces: the sweep carries the
# high cell's to the next face.
@inline function _limiter_theta_point!(theta, counts, Q, faces, uvw, c, p, inv_W, free,
                                       act, d, n, closed, hd, τ, mode, bounds, lay,
                                       volume, vmask, inv_J, areal, rates,
                                       o1, o2, o3, _, a, b)
    @inbounds begin
        T = eltype(theta)
        ud = uvw[d]
        n1, n2 = size(theta, 1), size(theta, 2)
        sd = _line_stride(d, n1, n2)
        L0 = _line_linear(d, 0, a, b, o1, o2, o3, n1, n2)
        limited = 0
        shared = 0
        sides = 0
        I0 = _line_node(d, 0, a, b, o1, o2, o3)
        q_lo = _limiter_state(Q, L0, lay)
        ε_lo = _limiter_cell_bound(q_lo, bounds)
        s_lo = _limiter_scale(uvw, c, inv_W, act, rates, d, hd, mode, areal, L0, I0)
        free_lo = _limiter_constrained(free, I0)
        for f in 0:n
            Ll = L0 + f * sd
            Lr = Ll + sd
            Ir = _line_node(d, f + 1, a, b, o1, o2, o3)
            q_hi = _limiter_state(Q, Lr, lay)
            ε_hi = _limiter_cell_bound(q_hi, bounds)
            s_hi = _limiter_scale(uvw, c, inv_W, act, rates, d, hd, mode, areal, Lr, Ir)
            free_hi = _limiter_constrained(free, Ir)
            G = _limiter_face_state(faces, Ll, lay)
            flag = 0
            θ = one(T)
            lo_bnd = closed[1] && f == 0
            hi_bnd = closed[2] && f == n
            fold_face = closed[3] && f == 0
            # The low cell takes −s; the sign change is exact, so −s is the
            # low cell's own expression.
            sl = lo_bnd || (mode == 1 && areal && fold_face) ? zero(T) : -s_lo
            sr = hi_bnd ? zero(T) : s_hi
            if !(lo_bnd || hi_bnd)
                # The filter's row at a closed end is the identity, and its
                # correction leaves the end node alone; so does the limiter.
                # Under the conservative row at a same-level interface
                # (`_filter_ends`) the end node is held, in one part.
                wall_l = closed[1] && f == 1
                wall_r = closed[2] && f == n - 1
                cl = free_lo && !(mode == 2 && wall_l && !closed[4]) && !fold_face
                cr = free_hi && !(mode == 2 && wall_r && !closed[5])
                qr = q_hi
                # The state across a fold is node 1's mirror, its normal momentum
                # reversed.
                ql = fold_face ? _limiter_mirror(qr, d) : q_lo
                uL = fold_face ? -ud[Lr] : ud[Ll]
                cL = fold_face ? c[Lr] : c[Ll]
                pL = fold_face ? p[Lr] : p[Ll]
                εl = fold_face ? _limiter_cell_bound(ql, bounds) : ε_lo
                εr = ε_hi
                Gl = volume ? _limiter_volume(G, vmask, inv_J[Ll]) : G
                Gr = volume ? _limiter_volume(G, vmask, inv_J[Lr]) : G
                # At a stage, a cell on a closed end keeps its boundary face, which
                # is never limited, in the same part as its interior face: one part
                # of weight α_d, q + (G_boundary − G_interior)/(α_d W), in place of
                # two halves, so the wall's flux enters at its own size, not twice.
                # A filter pass's end node at a conservative interface row
                # takes its end face in the same part.
                bl = ql; br = qr
                Gbl = G; Gbr = G
                if ((mode == 1) | closed[4]) & wall_l
                    sl /= 2
                    Gb = _limiter_face_state(faces, L0, lay)
                    Gbl = volume ? _limiter_volume(Gb, vmask, inv_J[Ll]) : Gb
                    bl = _limiter_axpy(ql, -sl, Gbl)
                end
                if ((mode == 1) | closed[5]) & wall_r
                    sr /= 2
                    Gb = _limiter_face_state(faces, L0 + n * sd, lay)
                    Gbr = volume ? _limiter_volume(Gb, vmask, inv_J[Lr]) : Gb
                    br = _limiter_axpy(qr, -sr, Gbr)
                end
                need = (cl && !_limiter_admissible(_limiter_axpy(bl, sl, Gl), εl)) ||
                       (cr && !_limiter_admissible(_limiter_axpy(br, sr, Gr), εr))
                if need
                    GL = (zero(T), zero(T), zero(T), zero(T), zero(T))
                    a_face = zero(T)
                    if mode == 1
                        # A node the limiter does not hold whose state is not
                        # admissible (a shell node interpolated across a shock)
                        # takes the held cell's state in the first-order flux.
                        own_l = !free_lo && !fold_face && !_limiter_positive(ql)
                        own_r = !free_hi && !_limiter_positive(qr)
                        qa, ua, ca, pa = own_l ? (qr, ud[Lr], c[Lr], p[Lr]) :
                                                 (ql, uL, cL, pL)
                        qb, ub, cb, pb = own_r ? (ql, uL, cL, pL) :
                                                 (qr, ud[Lr], c[Lr], p[Lr])
                        a_face = max(abs(ua) + ca, abs(ub) + cb)
                        GL = _limiter_lf_state(qa, qb, ua, ub, pa, pb, a_face, τ, d)
                    end
                    # The end cell at a patch interface blends its end face with
                    # the face beside it, toward the point flux of its state.
                    # In a filter pass the end cell's two faces are blended
                    # toward zero by one θ, so the end face's first-order value
                    # is its own value, which the half state has taken out.
                    GLl = mode == 1 && wall_l && closed[4] ?
                          _limiter_end_flux(GL, Gbl, ql, uL, pL, τ, d) :
                          ifelse((mode == 2) & wall_l & closed[4], Gbl, GL)
                    GLr = mode == 1 && wall_r && closed[5] ?
                          _limiter_end_flux(GL, Gbr, qr, ud[Lr], p[Lr], τ, d) :
                          ifelse((mode == 2) & wall_r & closed[5], Gbr, GL)
                    # A side whose first-order half state is not admissible, or
                    # whose first-order bound τ a |s| ≤ 1 fails, is unguaranteed.
                    if cl
                        t, ok = _limiter_side(bl, Gl, GLl, sl, εl)
                        θ = min(θ, t)
                        (ok && τ * a_face * abs(sl) <= 1) || (flag |= 1)
                    end
                    if cr
                        t, ok = _limiter_side(br, Gr, GLr, sr, εr)
                        θ = min(θ, t)
                        (ok && τ * a_face * abs(sr) <= 1) || (flag |= 2)
                    end
                end
            end
            theta[Ll] = θ
            θ < 1 && ((f >= 1 || closed[3]) ? (limited += 1) : (shared += 1))
            f >= 1 && (flag & 1) != 0 && (sides += 1)
            f <= n - 1 && (flag & 2) != 0 && (sides += 1)
            q_lo, ε_lo, s_lo, free_lo = q_hi, ε_hi, s_hi, free_hi
        end
        # A patch's end face takes the θ of the face beside it: at a stage at
        # every same-level interface, in a filter pass where its row conserves.
        if closed[5]
            θe = theta[L0+(n-1)*sd]
            theta[L0+n*sd] = θe
            θe < 1 && (limited += 1)
        end
        if closed[4]
            θe = theta[L0+sd]
            theta[L0] = θe
            θe < 1 && (shared += 1)
        end
        counts[1, a, b] = limited
        counts[2, a, b] = shared
        counts[3, a, b] = sides
        counts[4, a, b] = 0
    end
    return nothing
end

# The first-order flux an end cell at a patch interface blends toward, as the
# face beside it sees it: that face's Lax–Friedrichs flux GL, less τ times the
# point flux of the cell's state f(q) at the end face, plus the end face's own
# value Gb, which the cell's half state has already taken out.
@inline _limiter_end_flux(GL, Gb, q, u, p, τ, d) =
    map((x, b, f) -> x + b - f, GL, Gb, _limiter_lf_state(q, q, u, u, p, p, zero(u), τ, d))

# A state reflected across a plane normal to d.
@inline _limiter_mirror(q, d) =
    (q[1], d == 1 ? -q[2] : q[2], d == 2 ? -q[3] : q[3], d == 3 ? -q[4] : q[4], q[5])

# The aggregated face values a cell divides by its J where `vmask` is set.
@inline _limiter_volume(G, vmask, inv_J) = map((g, m) -> m ? g * inv_J : g, G, vmask)

# τ times component `cc` of the Lax–Friedrichs flux along d between L and R.
Base.@propagate_inbounds function _limiter_lf_component(Q, uvw, c, p, cc, L, R, τ, d, lay)
    ns, m1, m2, m3, ie, _ = lay
    ud = uvw[d]
    uL, uR = ud[L], ud[R]
    a = max(abs(uL) + c[L], abs(uR) + c[R])
    qL, qR = Q[L, cc], Q[R, cc]
    fL = qL * uL
    fR = qR * uR
    if cc == ie
        fL += p[L] * uL
        fR += p[R] * uR
    elseif (cc == m1 && d == 1) || (cc == m2 && d == 2) || (cc == m3 && d == 3)
        fL += p[L]
        fR += p[R]
    end
    return τ * ((fL + fR) / 2 - a * (qR - qL) / 2)
end

# The corrections at node p of a line from its two faces, where either is
# limited: δ = (θ − 1)(G − G_L) per face, node change (δ_{p−½} − δ_{p+½})/W_p.
# Mode 1 adds it to dQ / (B dt) and takes it out of the register, r ← r −
# change / B, and a limited fold face enters the face register `anchor`; mode 2
# adds it to the state, with ω = W/h in place of W, and under `volume` with ω J
# for the components marked in `mask`. Along an `areal` direction the
# first-order face value is J G_L and the node change inv_J times the face form's.
@inline function _limiter_correct_point!(target, regs, faces, theta, Q, uvw, c, pr,
                                         W, d, hd, τ, mode, scales, lay, closed, n,
                                         volume, mask, inv_J, areal, anchors, free,
                                         o1, o2, o3, pp, a, b)
    @inbounds begin
        T = eltype(theta)
        # A filter pass leaves a closed end's node alone (see the face pass),
        # except at an end the conservative interface row closes
        # (`_filter_ends`).
        mode == 2 && ((closed[1] && pp == 1 && !closed[4]) ||
                      (closed[2] && pp == n && !closed[5])) &&
            return nothing
        n1, n2 = size(theta, 1), size(theta, 2)
        I = _line_linear(d, pp, a, b, o1, o2, o3, n1, n2)
        Im = I - _line_stride(d, n1, n2)
        Ip = I + _line_stride(d, n1, n2)
        θm = theta[Im]
        θp = theta[I]
        (θm < 1 || θp < 1) || return nothing
        o = d == 1 ? o1 : d == 2 ? o2 : o3
        Wp = mode == 1 ? W[pp+o] : W[pp+o] / hd
        n_cons = lay[6]
        inv_B, inv_Bdt = scales
        fold_face = closed[3] && pp == 1
        # A patch's end face takes the point flux of its node, and its
        # correction enters the face register at that end, as a fold's does.
        end_lo = closed[4] && pp == 1
        end_hi = closed[5] && pp == n
        # A node the limiter does not hold whose state is not admissible is
        # replaced by the other node of the face in the first-order flux, as
        # the face pass replaced it: a neighbour by this node, and this node
        # by the neighbour. Both nodes of a face then take one flux, and the
        # corrections telescope; a face whose two corrections differed would
        # leave their difference in the face register, carried by every later
        # stage's running sum to the far end of the line.
        own_m = mode == 1 && !fold_face && !end_lo &&
                !_limiter_constrained(free, _line_node(d, pp - 1, a, b, o1, o2, o3)) &&
                !_limiter_positive(_limiter_state(Q, Im, lay))
        own_p = mode == 1 && !end_hi &&
                !_limiter_constrained(free, _line_node(d, pp + 1, a, b, o1, o2, o3)) &&
                !_limiter_positive(_limiter_state(Q, Ip, lay))
        own = mode == 1 && !_limiter_constrained(free, _line_node(d, pp, a, b, o1, o2, o3)) &&
              !_limiter_positive(_limiter_state(Q, I, lay))
        Jm = end_lo || own_m ? I : Im
        Km = own && !end_lo ? Im : I
        Jp = end_hi || own_p ? I : Ip
        Kp = own && !end_hi ? Ip : I
        anchor, anchor_hi = anchors
        for cc in 1:n_cons
            change = zero(T)
            if θm < 1
                gl = mode == 2 ? zero(T) :
                     fold_face ? _limiter_lf_mirror(Q, uvw, c, pr, cc, I, τ, d, lay) :
                     _limiter_lf_component(Q, uvw, c, pr, cc, Jm, Km, τ, d, lay)
                areal && (gl /= inv_J[I])
                δ = (θm - 1) * (faces[cc][Im] - gl)
                change += δ
                mode == 1 && (fold_face || end_lo) && (anchor[cc, a, b] += δ * inv_B)
            end
            if θp < 1
                gl = mode == 1 ? _limiter_lf_component(Q, uvw, c, pr, cc, Kp, Jp, τ, d,
                                                       lay) : zero(T)
                areal && (gl /= inv_J[I])
                δp = (θp - 1) * (faces[cc][I] - gl)
                change -= δp
                mode == 1 && end_hi && (anchor_hi[cc, a, b] += δp * inv_B)
            end
            change /= Wp
            if mode == 1
                target[I, cc] += (areal ? inv_J[I] * change : change) * inv_Bdt
                regs[cc][I] -= change * inv_B
            else
                volume && mask[cc] && (change *= inv_J[I])
                target[I, cc] += change
            end
        end
    end
    return nothing
end

# τ times component `cc` of the Lax–Friedrichs flux along d between node I's
# mirror across a fold and node I.
Base.@propagate_inbounds function _limiter_lf_mirror(Q, uvw, c, p, cc, I, τ, d, lay)
    ns, m1, m2, m3, ie, _ = lay
    md = d == 1 ? m1 : d == 2 ? m2 : m3
    u = uvw[d][I]
    a = abs(u) + c[I]
    q = Q[I, cc]
    qL = cc == md ? -q : q
    fR = q * u
    fL = -qL * u
    if cc == ie
        fL -= p[I] * u
        fR += p[I] * u
    elseif cc == md
        fL += p[I]
        fR += p[I]
    end
    return τ * ((fL + fR) / 2 - a * (q - qL) / 2)
end
