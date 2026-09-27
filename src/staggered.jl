# Staggered compact operators for the implicit diffusion operator
# L = J⁻¹ G C K D_s of reference/IMPLICIT.md: the sixth-order compact
# derivative from nodes to midpoints D_s (Lele 1992, JCP 103, α = 9/62,
# a = 63/62, b = 17/62), the matching derivative G from midpoints back to
# nodes, and the sixth-order explicit interpolation of a coefficient from nodes
# to midpoints. None of it is reached from the right-hand side.
#
# Storage. Midpoint data live in an ordinary padded field: the midpoint
# between local nodes i and i+1 (global position x_i + h/2) is stored at local
# index i. On a periodic dimension the global line holds as many midpoints as
# nodes, the last one between node N and the wrapped node 1, so every index
# carries data and the halo exchange of a node field serves midpoint fields
# unchanged. On a closed dimension the line holds N - 1 midpoints and global
# index N carries none; the rank owning the high end writes zero there, and no
# operator reads it.
#
# Closed ends. The walls of a closed dimension lie on nodes 1 and N, and each
# is a mirror plane through its wall node: node data reflect as
# f(1 - m) = σ f(1 + m) and f(N + m) = σ f(N - m), midpoint data as
# g(1 - m) = σ g(m) and g(N - 1 + m) = σ g(N - m), with σ the parity of the
# operator's input. The fill reads a tap beyond the wall at its mirror, so no
# halo is written, and the left-hand side folds the coupling to the mirrored
# unknown onto the row, as the half-offset folds of `plan_direction` do. The
# interior rows then run to the wall. For an even temperature (σ = +1, zero
# wall flux: the adiabatic wall) the flux is odd and G takes σ = -1; an
# odd temperature is the isothermal wall written on T - T_wall. Under the
# mirror the adjoint identity of the periodic operators survives the wall:
# W_n G = -D_sᵀ W_m, with W_m = h on the midpoints and the trapezoidal W_n
# (h/2 on the wall nodes), so W_n L is symmetric negative semidefinite and
# telescopes. One-sided closure rows were measured instead and lose both
# (reference/CALIBRATION_APPENDIX.md, "Stiff diffusion"). The mirror is exact
# to the interior order only for data of its parity, and first order at the
# wall node otherwise.
#
# Folded ends. A symmetry plane or a coordinate singularity lies half a cell
# beyond the end node (folds.jl), which puts a midpoint on the plane: at local
# index 0 below node 1, and at index N above node N, where the storage above
# already has a slot. Node data reflect as f(1 - m) = σ f(m) and
# f(N + m) = σ f(N + 1 - m), midpoint data about the plane midpoint as
# g(-m) = σ g(m) and g(N + m) = σ g(N - m). A plane midpoint of odd parity is
# zero; one of even parity is an unknown of the line. At the high end it is
# the row of slot N. At the low end the row of midpoint 0,
# g(0) + 2α g(1) = r(0), is eliminated into the row of midpoint 1, whose
# diagonal becomes 1 - 2α² and whose right-hand side takes -α r(0), and g(0)
# is recovered after the solve into the halo slot 0, where the operators with
# a midpoint input read it. The exchange along the dimension leaves a
# physical edge's halo alone, so the slot survives it. With the plane weight
# h/2 on the plane midpoints and h on every node, the adjoint identity and
# everything that follows from it hold as at a wall, and the fold is exact to
# the interior order on data of its parity. A paired fold (the resolved axis,
# the spherical origin and poles) runs the even and odd combinations of the
# line and its antipodal partner through the two parity plans, as
# `fold_apply!` does, the plane midpoint included (`StaggeredPair`).
#
# Metric. On a curvilinear or stretched line the flux is C K D_s T, with C
# the face area over the spacing, J/h_d², evaluated analytically at the
# midpoints, and the divergence takes J⁻¹ at the nodes: J W_n L =
# -D_sᵀ W_m C K D_s, so J W_n L is symmetric and conserves with the node
# volumes J W_n. The mirror at a curved wall reflects the geometry as well
# as the data, which C, not even about the wall, does not satisfy, so there
# the wall node is first order whatever the parity of the data. At a fold
# whose area vanishes oddly (the cylindrical axis, the spherical poles) the
# flux C K D_s T is even across the plane where D_s T is odd. Its smooth
# continuation, which the explicit divergence also uses, is not the mirror
# adjoint to D_s: J W L is then symmetric only up to a defect confined to
# the rows near the fold and decaying with h, and conserves to second order
# there. The adjoint mirror would continue the flux as |r| K T', whose kink
# makes the first node inconsistent. The spherical origin's area is even,
# and there the two coincide.

const STAGGERED_ALPHA = 9 // 62
const STAGGERED_A = 63 // 62
const STAGGERED_B = 17 // 62

# Offsets of the first tap, and the taps, of each operator: the output at
# local index i reads the input at i + first, i + first + 1, ...
function _staggered_taps(::Type{T}, op::Symbol) where {T}
    a, b = T(STAGGERED_A), T(STAGGERED_B) / 3
    op === :to_mid && return -1, T[-b, -a, a, b]
    op === :to_node && return -2, T[-b, -a, a, b]
    op === :interpolate && return -2, T[3, -25, 150, 150, -25, 3] ./ 256
    throw(ArgumentError("staggered operator must be :to_mid, :to_node or " *
                        ":interpolate, not :$op"))
end

_input_is_mid(op::Symbol) = op === :to_node
_output_is_mid(op::Symbol) = op !== :to_node

"""
    StaggeredPlan

Directional plan of one staggered operator along one dimension of a
decomposition: `:to_mid` (the compact derivative from nodes to midpoints),
`:to_node` (from midpoints to nodes) or `:interpolate` (sixth-order explicit
interpolation from nodes to midpoints). Built by `plan_staggered` and applied
by `apply_along!`; the storage convention, the wall mirror and the folded
ends are described at the top of `src/staggered.jl`.
"""
struct StaggeredPlan{T}
    dim::Int
    n::Int                        # local extent along dim
    lines::Int
    tr::Bool                      # transposed (lines × n) layout for y/z sweeps
    op::Symbol
    parity::Int                   # parity of the input at a closed end
    first::Int                    # offset of the first interior tap
    w::Vector{T}                  # interior taps, prescaled by 1/h for derivatives
    edge::Vector{Int}             # per output row: 0 interior, else index below
    edge_index::Vector{Vector{Int}}   # mirrored input indices of an edge row
    edge_weight::Vector{Vector{T}}    # and their signed weights (empty: zero row)
    plane::Bool                   # writes the low plane midpoint into slot 0
    plane_index::Vector{Int}      # the plane row's mirrored input indices
    plane_weight::Vector{T}       # and weights: r(0)
    plane_coupling::T             # g(0) = r(0) - plane_coupling * g(1)
    plane_rhs::Matrix{T}          # r(0) per line, over the two other dimensions
    line_solver::LineSolver{T}
    B::Matrix{T}
end

# The local input index and sign a tap at `i` reads on a rank owning an end;
# `(i, 1)` where no mirror applies and a zero sign where the tap reads a
# plane midpoint of odd parity. `lo`/`hi` say which ends this rank owns,
# `fl`/`fh` whether they are folds, `σl`/`σh` the input parity there.
function _mirror_tap(i, n, lo, hi, fl, fh, mid, σl, σh)
    if lo
        if fl
            if mid
                i < 0 && return -i, σl
                i == 0 && return 0, (σl > 0 ? 1 : 0)
            else
                i < 1 && return 1 - i, σl
            end
        elseif i < 1
            return (mid ? 1 - i : 2 - i), σl
        end
    end
    if hi
        if fh
            i > n && return (mid ? 2n - i : 2n + 1 - i), σh
            mid && i == n && return n, (σh > 0 ? 1 : 0)
        elseif mid ? i >= n : i > n
            return (mid ? 2n - 1 - i : 2n - i), σh
        end
    end
    return i, 1
end

"""
    plan_staggered(decomp, op, dim, h; parity=1, lo_fold=nothing, hi_fold=nothing)

Build a `StaggeredPlan` for the staggered operator `op` (`:to_mid`,
`:to_node` or `:interpolate`) along dimension `dim` of `decomp`, with uniform
spacing `h`. `parity` is the parity (±1) of the operator's input at the
mirror planes of a closed dimension and is ignored on a periodic one.
`lo_fold` and `hi_fold` declare an end folded half a cell beyond its end node
(a symmetry plane or a coordinate singularity), each giving the input's
parity (±1) across that plane, as in `plan_direction`; an unfolded closed end
is a wall through its node.

A decomposed `dim` makes the factorization of the two derivatives
collective over the sub-communicator along `dim`, as for `plan_direction`;
the interpolation has an identity left-hand side and communicates nothing.
The taps reach three points, within the halo, and on a rank owning an end
every mirrored tap must land inside the block, which needs a local extent of
at least 4.
"""
function plan_staggered(decomp::Decomp{T}, op::Symbol, dim::Int, h::Real;
                        parity::Int=1, lo_fold::Union{Nothing,Int}=nothing,
                        hi_fold::Union{Nothing,Int}=nothing) where {T}
    decomp.active[dim] || throw(ArgumentError(
        "staggered operator along collapsed dimension $dim"))
    abs(parity) == 1 || throw(ArgumentError("parity must be ±1, not $parity"))
    for σ in (lo_fold, hi_fold)
        σ === nothing || abs(σ) == 1 ||
            throw(ArgumentError("fold parity must be ±1 or nothing, not $σ"))
    end
    decomp.periodic[dim] && (lo_fold !== nothing || hi_fold !== nothing) &&
        throw(ArgumentError("a periodic dimension carries no fold"))
    first, taps = _staggered_taps(T, op)
    reach = max(-first, first + length(taps) - 1)
    reach <= decomp.n_halo || error("staggered taps reach $reach points; halo " *
                                    "width is $(decomp.n_halo)")
    n = decomp.n_local[dim]
    lo = at_lo_edge(decomp, dim)
    hi = at_hi_edge(decomp, dim)
    fl = lo && lo_fold !== nothing
    fh = hi && hi_fold !== nothing
    σl = fl ? lo_fold : parity
    σh = fh ? hi_fold : parity
    n >= 4 || error("local extent $n along dim $dim too small for the " *
                    "staggered operators (need ≥ 4); use fewer ranks in this dimension")
    mid_in = _input_is_mid(op)
    mid_out = _output_is_mid(op)
    derivative = op !== :interpolate
    σl_out = derivative ? -σl : σl
    σh_out = derivative ? -σh : σh
    w = derivative ? taps ./ T(h) : taps
    α = derivative ? T(STAGGERED_ALPHA) : zero(T)

    # The input indices a row may read through the mirror: the plane midpoint
    # of a low fold sits in slot 0, and a wall's dummy midpoint slot is never
    # read.
    first_input = fl && mid_in && σl > 0 ? 0 : 1
    last_input = mid_in && hi && !fh ? n - 1 : n
    function row_taps(i)
        index = Int[]; weight = T[]
        for (m, wm) in enumerate(w)
            raw = i + first + m - 1
            j, s = _mirror_tap(raw, n, lo, hi, fl, fh, mid_in, σl, σh)
            s == 0 && continue
            (j == raw && s == 1) || first_input <= j <= last_input || error(
                "local extent $n along dim $dim too small for the staggered " *
                "mirror; use fewer ranks in this dimension")
            k = findfirst(==(j), index)
            if k === nothing
                push!(index, j); push!(weight, s * wm)
            else
                weight[k] += s * wm
            end
        end
        return index, weight
    end

    # Output rows whose taps cross an end, with the mirrored reads summed per
    # input index; the dummy midpoint slot at a high wall is a zero row, and
    # at a low fold the plane row is eliminated into the row of midpoint 1.
    edge = zeros(Int, n)
    edge_index = Vector{Int}[]
    edge_weight = Vector{T}[]
    schur = fl && mid_out && derivative
    for i in 1:n
        dummy = hi && !fh && mid_out && i == n
        crosses = false
        for m in eachindex(w)
            raw = i + first + m - 1
            crosses |= _mirror_tap(raw, n, lo, hi, fl, fh, mid_in, σl, σh) != (raw, 1)
        end
        (dummy || crosses || (schur && i == 1)) || continue
        index = Int[]; weight = T[]
        if !dummy
            index, weight = row_taps(i)
            if schur && i == 1
                for (j, wj) in zip(row_taps(0)...)
                    k = findfirst(==(j), index)
                    if k === nothing
                        push!(index, j); push!(weight, -α * wj)
                    else
                        weight[k] -= α * wj
                    end
                end
            end
        end
        push!(edge_index, index); push!(edge_weight, weight)
        edge[i] = length(edge_index)
    end
    plane = fl && mid_out
    plane_index, plane_weight = plane ? row_taps(0) : (Int[], T[])
    plane_coupling = plane ? (1 + σl_out) * α : zero(T)

    a = fill(α, n); b = fill(one(T), n); c = fill(α, n)
    if mid_out
        if fl
            # g(0) = r(0) - (1 + σ) α g(1), eliminated from the row of g(1).
            b[1] -= (1 + σl_out) * α^2
        elseif lo
            # Midpoint 1 couples to the mirrored midpoint 0 = σ · midpoint 1.
            b[1] += σl_out * α
        end
        if fh
            # The plane midpoint N couples to N + 1 = σ · midpoint N - 1.
            a[n] = (1 + σh_out) * α
        elseif hi
            # Midpoint N - 1 couples to midpoint N = σ · midpoint N - 1, and
            # the slot of midpoint N is a decoupled identity row.
            b[n-1] += σh_out * α; c[n-1] = zero(T)
            a[n] = zero(T); b[n] = one(T); c[n] = zero(T)
        end
    else
        # Node 1 couples to the mirrored node 0: σ · node 2 through a wall
        # node, σ · node 1 across a plane half a cell out; node N likewise.
        fl ? (b[1] += σl_out * α) : lo && (c[1] += σl_out * α)
        fh ? (b[n] += σh_out * α) : hi && (a[n] += σh_out * α)
    end
    aL = lo ? zero(T) : α
    cR = hi ? zero(T) : α
    lines = prod(decomp.n_local[k] for k in 1:3 if k != dim)
    line_solver = LineSolver(a, b, c, aL, cR, decomp.sub[dim], decomp.sub_size[dim],
                             decomp.sub_rank[dim], lines; periodic=decomp.periodic[dim],
                             explicit=!derivative)
    tr = dim > 1
    others = [decomp.n_local[k] for k in 1:3 if k != dim]
    plane_rhs = plane ? zeros(T, others[1], others[2]) : zeros(T, 0, 0)
    StaggeredPlan{T}(dim, n, lines, tr, op, parity, first, w, edge, edge_index,
                     edge_weight, plane, plane_index, plane_weight, plane_coupling,
                     plane_rhs, line_solver,
                     tr ? zeros(T, lines, n) : zeros(T, n, lines))
end

function _fill_staggered!(B::Matrix{T}, plan::StaggeredPlan{T}, f, decomp::Decomp,
                          ::Val{1}) where {T}
    pad = decomp.n_halo_d
    n = plan.n
    n1 = decomp.n_local[2]
    w = plan.w
    o = plan.first - 1
    @threaded plan.lines*n for l in 1:plan.lines
        kk, jj = divrem(l - 1, n1)
        j = jj + 1
        k = kk + 1
        @inbounds for i in 1:n
            e = plan.edge[i]
            acc = zero(T)
            if e == 0
                for m in eachindex(w)
                    acc += w[m] * f[i+o+m+pad[1], j+pad[2], k+pad[3]]
                end
            else
                index = plan.edge_index[e]; weight = plan.edge_weight[e]
                for m in eachindex(index)
                    acc += weight[m] * f[index[m]+pad[1], j+pad[2], k+pad[3]]
                end
            end
            B[i, l] = acc
        end
    end
    return B
end

function _fill_staggered!(B::Matrix{T}, plan::StaggeredPlan{T}, f, decomp::Decomp,
                          ::Val{D}) where {T,D}
    o1, o2, o3 = decomp.n_halo_d
    n = plan.n
    nx = decomp.n_local[1]
    nout = D == 2 ? decomp.n_local[3] : decomp.n_local[2]
    w = plan.w
    o = plan.first - 1
    @threaded nout*n*nx for jk in outer_indices(n, nout)
        jr, kk = Tuple(jk)
        @inbounds begin
            base = (kk - 1) * nx
            e = plan.edge[jr]
            if e == 0
                for i in 1:nx
                    acc = zero(T)
                    for m in eachindex(w)
                        acc += w[m] * (D == 2 ? f[i+o1, jr+o+m+o2, kk+o3] :
                                                f[i+o1, kk+o2, jr+o+m+o3])
                    end
                    B[base+i, jr] = acc
                end
            else
                index = plan.edge_index[e]; weight = plan.edge_weight[e]
                for i in 1:nx
                    acc = zero(T)
                    for m in eachindex(index)
                        acc += weight[m] * (D == 2 ? f[i+o1, index[m]+o2, kk+o3] :
                                                     f[i+o1, kk+o2, index[m]+o3])
                    end
                    B[base+i, jr] = acc
                end
            end
        end
    end
    return B
end

# The right-hand side r(0) of the low plane midpoint on every line, read
# before the scatter so that `out` may alias `f`, and the recovery of g(0)
# into slot 0 after it.
function _plane_rhs!(plan::StaggeredPlan{T}, f, decomp::Decomp, ::Val{D}) where {T,D}
    pad = decomp.n_halo_d
    index = plan.plane_index; weight = plan.plane_weight
    R = plan.plane_rhs
    @inbounds for k in axes(R, 2), j in axes(R, 1)
        acc = zero(T)
        for m in eachindex(index)
            acc += weight[m] * f[_gidx(Val(D), index[m], j, k, pad)]
        end
        R[j, k] = acc
    end
    return R
end

function _plane_write!(out, plan::StaggeredPlan, decomp::Decomp, ::Val{D}) where {D}
    pad = decomp.n_halo_d
    R = plan.plane_rhs
    coupling = plan.plane_coupling
    @inbounds for k in axes(R, 2), j in axes(R, 1)
        out[_gidx(Val(D), 0, j, k, pad)] =
            R[j, k] - coupling * out[_gidx(Val(D), 1, j, k, pad)]
    end
    return out
end

# The staggered operator of `plan` applied to `f` along `plan.dim`, into the
# interior of `out`, and at a low fold into the plane slot 0 of a midpoint
# output. `f` must have current rank-boundary halos along the dimension;
# beyond a wall or a fold the mirror is read from the interior, so a closed
# end's halo is never read, except slot 0 of a midpoint input at a low fold,
# which holds the plane midpoint. The derivatives' line solve is collective
# over the sub-communicator along a decomposed dimension. (A comment, not a
# docstring: the method would otherwise join the rendered `apply_along!`
# entry.)
function apply_along!(out, plan::StaggeredPlan, f, decomp::Decomp)
    d = plan.dim
    if d == 1
        plan.plane && _plane_rhs!(plan, f, decomp, Val(1))
        _fill_staggered!(plan.B, plan, f, decomp, Val(1))
        solve_lines!(plan.B, plan.line_solver)
        _scatter_lines!(out, plan.B, plan, decomp, Val(1))
        plan.plane && _plane_write!(out, plan, decomp, Val(1))
    elseif d == 2
        plan.plane && _plane_rhs!(plan, f, decomp, Val(2))
        _fill_staggered!(plan.B, plan, f, decomp, Val(2))
        solve_lines_t!(plan.B, plan.line_solver)
        _scatter_t!(out, plan.B, plan, decomp, Val(2))
        plan.plane && _plane_write!(out, plan, decomp, Val(2))
    else
        plan.plane && _plane_rhs!(plan, f, decomp, Val(3))
        _fill_staggered!(plan.B, plan, f, decomp, Val(3))
        solve_lines_t!(plan.B, plan.line_solver)
        _scatter_t!(out, plan.B, plan, decomp, Val(3))
        plan.plane && _plane_write!(out, plan, decomp, Val(3))
    end
    return out
end

"""
    StaggeredPair

A staggered operator across a paired fold (the resolved-θ cylindrical axis,
the spherical origin or poles): the two parity plans of the folded end, the
fold's `PairSpec`, and the antipodal sign `sigma` of the operator's input.
Applied by `apply_along!` as `fold_apply!` applies a compact operator: the
even and odd combinations of each line and its antipodal partner, the plan
of each parity, and the inverse butterfly, carried through the plane
midpoint of a midpoint field as well as the interior.
"""
struct StaggeredPair{T,P,A}
    plans::NTuple{2,StaggeredPlan{T}}   # input parity +1 (even), -1 (odd)
    pair::P
    sigma::Int
    layer_in::Bool                      # the input's plane slot 0 is carried
    layer_out::Bool                     # the output's plane slot 0 is carried
    work::A                             # the combinations
    second::A                           # the odd plan's output, partner on-rank
end

# The butterfly bodies of folds.jl over the interior along the fold
# dimension, extended by the plane slot 0 when `layer` is set: the pairing
# map acts on the other two dimensions only.
function _pair_extent(decomp::Decomp, d::Int, layer::Bool)
    n = ntuple(k -> decomp.n_local[k] + (layer && k == d ? 1 : 0), 3)
    o = ntuple(k -> decomp.n_halo_d[k] - (layer && k == d ? 1 : 0), 3)
    return n, o
end

function _stag_pair_forward!(w, f, sp::StaggeredPair, decomp::Decomp, d::Int)
    pair = sp.pair
    sf = eltype(f)(sp.sigma)
    (nx, ny, nz), (o1, o2, o3) = _pair_extent(decomp, d, sp.layer_in)
    if pair.local_pair
        sd = pair.pdim != 0 ? pair.pdim : pair.revdim
        half = decomp.n_local[sd] ÷ 2
        pointwise!(_pair_forward_local_point!, w, nx, ny, nz,
                   w, f, sf, sd, half, pair.pdim, pair.shift_local,
                   pair.revdim, decomp.n_local, decomp.n_halo_d, o1, o2, o3)
    else
        sendrecv_block!(f, pair.buf, decomp, d, pair.partner, 41)
        pointwise!(_pair_forward_remote_point!, w, nx, ny, nz,
                   w, f, pair.buf, sf, pair.keep_e, pair.pdim,
                   pair.shift_local, pair.revdim, decomp.n_local,
                   decomp.n_halo_d, o1, o2, o3)
    end
    return w
end

function _stag_pair_backward!(out, sp::StaggeredPair, decomp::Decomp, d::Int)
    pair = sp.pair
    sf = eltype(out)(sp.sigma)
    (nx, ny, nz), (o1, o2, o3) = _pair_extent(decomp, d, sp.layer_out)
    if pair.local_pair
        sd = pair.pdim != 0 ? pair.pdim : pair.revdim
        half = decomp.n_local[sd] ÷ 2
        pointwise!(_pair_select_point!, out, nx, ny, nz,
                   out, sp.second, sd, half, o1, o2, o3)
        pointwise!(_pair_backward_local_point!, out, nx, ny, nz,
                   out, sf, sd, half, pair.pdim, pair.shift_local, pair.revdim,
                   decomp.n_local, decomp.n_halo_d, o1, o2, o3)
    else
        sendrecv_block!(out, pair.buf, decomp, d, pair.partner, 42)
        pointwise!(_pair_backward_remote_point!, out, nx, ny, nz,
                   out, pair.buf, sf, pair.keep_e, pair.pdim, pair.shift_local,
                   pair.revdim, decomp.n_local, decomp.n_halo_d, o1, o2, o3)
    end
    return out
end

# The paired form of the staggered operator, into the interior of `out` (and
# its plane slot on a midpoint output). `f` is read, not modified, and needs
# no halo: the combinations are exchanged here. Every rank must call it: the
# line solves are collective along the fold dimension and an off-rank
# partner adds two pairwise exchanges.
function apply_along!(out, sp::StaggeredPair, f, decomp::Decomp)
    d = sp.plans[1].dim
    w = sp.work
    _stag_pair_forward!(w, f, sp, decomp, d)
    exchange_dim!(w, decomp, d)
    if sp.pair.local_pair
        apply_along!(out, sp.plans[1], w, decomp)
        apply_along!(sp.second, sp.plans[2], w, decomp)
    else
        apply_along!(out, sp.plans[sp.pair.keep_e ? 1 : 2], w, decomp)
    end
    _stag_pair_backward!(out, sp, decomp, d)
    return out
end

"""
    StaggeredDiffusion(decomp, dim, h; parity=1, lo_fold=nothing, hi_fold=nothing)
    StaggeredDiffusion(solver, dim; parity=1)

The conservative diffusion operator `J⁻¹ ∂(C κ ∂T)` along `dim` in its
staggered form `J⁻¹ G C K D_s`: the three plans, the two midpoint fields it
works in, and on a curvilinear or stretched line the face factor `C` (the
face area over the spacing at the midpoints) and `J⁻¹` at the nodes.
`parity` is the temperature's parity at the mirror planes of a closed
dimension (+1 the adiabatic wall); the flux takes the opposite parity and the
coefficient is even. Applied by `staggered_diffusion!`.

The first form is a uniform Cartesian line of spacing `h`, where `lo_fold`
and `hi_fold` give the temperature's parity across a symmetry plane half a
cell beyond an end, as in `plan_staggered`. The second takes the spacing,
the metric, any stretching and the folds of dimension `dim` from a
single-patch `solver` on host storage, for a scalar temperature; a paired
fold makes each plan a `StaggeredPair`.
"""
struct StaggeredDiffusion{P1,P2,P3,A,C,J}
    to_mid::P1
    to_node::P2
    interpolate::P3
    flux::A
    kappa_mid::A
    face::C               # J/h_d² at the midpoints, or nothing on a unit line
    inv_J::J              # 1/J at the nodes, or nothing on a unit line
end

function StaggeredDiffusion(decomp::Decomp, dim::Int, h::Real; parity::Int=1,
                            lo_fold::Union{Nothing,Int}=nothing,
                            hi_fold::Union{Nothing,Int}=nothing)
    neg(σ) = σ === nothing ? nothing : -σ
    even(σ) = σ === nothing ? nothing : 1
    StaggeredDiffusion(plan_staggered(decomp, :to_mid, dim, h; parity=parity,
                                      lo_fold=lo_fold, hi_fold=hi_fold),
                       plan_staggered(decomp, :to_node, dim, h; parity=-parity,
                                      lo_fold=neg(lo_fold), hi_fold=neg(hi_fold)),
                       plan_staggered(decomp, :interpolate, dim, h; parity=1,
                                      lo_fold=even(lo_fold), hi_fold=even(hi_fold)),
                       field(decomp), field(decomp), nothing, nothing)
end

function StaggeredDiffusion(solver, dim::Int; parity::Int=1)
    decomp = solver.decomp
    _cpu_storage(solver.inv_J) || throw(ArgumentError(
        "the staggered operators run on host storage only"))
    h = solver.h[dim]
    fold = solver.folds[dim]
    # The temperature is even across every fold and κ is even; the flux
    # C K D_s T takes the fold's sign of the energy flux, which carries the
    # parity of the area factor, as the explicit divergence does. Where the
    # area is odd (the axis, the poles) this continues the flux smoothly
    # through the singular set but is not the mirror adjoint to D_s, so
    # there J W L is not symmetric; the adjoint mirror continues r K T' as
    # |r| K T' and is inconsistent at the first node (header of this file).
    σq = fold === nothing ? 1 : fold.sigflux[solver.equations.i_energy]
    lo = fold !== nothing && fold.lo
    hi = fold !== nothing && fold.hi
    folded(σ) = (lo ? σ : nothing, hi ? σ : nothing)
    plan(op, p, σ) = plan_staggered(decomp, op, dim, h; parity=p,
                                    lo_fold=folded(σ)[1], hi_fold=folded(σ)[2])
    walls = ((:to_mid, parity), (:to_node, -parity), (:interpolate, 1))
    signs = (1, σq, 1)
    plans = if fold === nothing || fold.pair === nothing
        ntuple(i -> plan(walls[i][1], walls[i][2], signs[i]), 3)
    else
        # One scratch pair serves all three operators, applied in sequence.
        work = field(decomp); second = field(decomp)
        layer = lo && at_lo_edge(decomp, dim)
        ntuple(3) do i
            op, p = walls[i]
            StaggeredPair((plan(op, p, 1), plan(op, p, -1)), fold.pair, signs[i],
                          layer && _input_is_mid(op), layer && _output_is_mid(op),
                          work, second)
        end
    end
    unit = solver.metric isa CartesianMetric && all(isnothing, solver.stretch)
    face = unit ? nothing : _staggered_face(solver, dim)
    StaggeredDiffusion(plans..., field(decomp), field(decomp), face,
                       unit ? nothing : solver.inv_J)
end

# J/h_d² with coordinate `d` at the midpoints and the others at the nodes,
# evaluated analytically over the padded extent as `init_geometry!` does, so
# the plane midpoint of a fold on a coordinate singularity carries the
# vanishing area of the singular set.
function _staggered_face(solver, d::Int)
    face = field(solver.decomp)
    T = eltype(face)
    tiny = positive_floor(T)
    for k in axes(face, 3), j in axes(face, 2), i in axes(face, 1)
        x1, m1 = _phys_and_jac(solver, 1, i, d == 1)
        x2, m2 = _phys_and_jac(solver, 2, j, d == 2)
        x3, m3 = _phys_and_jac(solver, 3, k, d == 3)
        hs = scalefactors(solver.metric, x1, x2, x3) .* (m1, m2, m3)
        face[i, j, k] = hs[1] * hs[2] * hs[3] / max(hs[d], tiny)^2
    end
    return face
end

"""
    staggered_diffusion!(out, op, T_ion, kappa, decomp)

Write `J⁻¹ G C K D_s T_ion`, the staggered approximation of
`J⁻¹ ∂(C κ ∂T_ion)` along `op`'s dimension, into the interior of `out`, with
`K` the midpoint interpolation of `kappa`. Exchanges the halos of `T_ion`,
`kappa` and the midpoint flux along the dimension, so every rank of the
sub-communicator must call it (every rank of the run across a paired fold).
"""
function staggered_diffusion!(out, op::StaggeredDiffusion, T_ion, kappa, decomp::Decomp)
    d = _staggered_dim(op.to_mid)
    exchange_dim!(T_ion, decomp, d)
    exchange_dim!(kappa, decomp, d)
    apply_along!(op.flux, op.to_mid, T_ion, decomp)
    apply_along!(op.kappa_mid, op.interpolate, kappa, decomp)
    op.flux .*= op.kappa_mid
    op.face === nothing || (op.flux .*= op.face)
    exchange_dim!(op.flux, decomp, d)
    apply_along!(out, op.to_node, op.flux, decomp)
    op.inv_J === nothing || (out .*= op.inv_J)
    return out
end

_staggered_dim(plan::StaggeredPlan) = plan.dim
_staggered_dim(sp::StaggeredPair) = sp.plans[1].dim
