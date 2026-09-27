# Staggered compact operators for the implicit diffusion operator
# L = G K D_s of reference/IMPLICIT.md: the sixth-order compact derivative from
# nodes to midpoints D_s (Lele 1992, JCP 103, α = 9/62, a = 63/62,
# b = 17/62), the matching derivative G from midpoints back to nodes, and the
# sixth-order explicit interpolation of a coefficient from nodes to
# midpoints. None of it is reached from the right-hand side.
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
# The metric enters nowhere yet: the plans assume a uniform Cartesian line.
# A curvilinear line needs the face area over the spacing at the midpoints
# (area_d / h interpolated, or evaluated there) folded into K, and inv_J at
# the nodes applied to the divergence.

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
by `apply_along!`; the storage convention and the wall mirror are described
at the top of `src/staggered.jl`.
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
    line_solver::LineSolver{T}
    B::Matrix{T}
end

# The local input index and sign a tap at `i` reads on a rank owning a wall;
# `(i, 1)` where no mirror applies.
function _mirror_tap(i, n, lo, hi, mid, σ)
    if lo && i < 1
        return mid ? (1 - i, σ) : (2 - i, σ)
    elseif hi && (mid ? i >= n : i > n)
        return mid ? (2n - 1 - i, σ) : (2n - i, σ)
    end
    return i, 1
end

"""
    plan_staggered(decomp, op, dim, h; parity=1)

Build a `StaggeredPlan` for the staggered operator `op` (`:to_mid`,
`:to_node` or `:interpolate`) along dimension `dim` of `decomp`, with uniform
spacing `h`. `parity` is the parity (±1) of the operator's input at the
mirror planes of a closed dimension and is ignored on a periodic one.

A decomposed `dim` makes the factorization of the two derivatives
collective over the sub-communicator along `dim`, as for `plan_direction`;
the interpolation has an identity left-hand side and communicates nothing.
The taps reach three points, within the halo, and on a rank owning a wall
every mirrored tap must land inside the block, which needs a local extent of
at least 4.
"""
function plan_staggered(decomp::Decomp{T}, op::Symbol, dim::Int, h::Real;
                        parity::Int=1) where {T}
    decomp.active[dim] || throw(ArgumentError(
        "staggered operator along collapsed dimension $dim"))
    abs(parity) == 1 || throw(ArgumentError("parity must be ±1, not $parity"))
    first, taps = _staggered_taps(T, op)
    reach = max(-first, first + length(taps) - 1)
    reach <= decomp.n_halo || error("staggered taps reach $reach points; halo " *
                                    "width is $(decomp.n_halo)")
    n = decomp.n_local[dim]
    lo = at_lo_edge(decomp, dim)
    hi = at_hi_edge(decomp, dim)
    n >= 4 || error("local extent $n along dim $dim too small for the " *
                    "staggered operators (need ≥ 4); use fewer ranks in this dimension")
    mid_in = _input_is_mid(op)
    mid_out = _output_is_mid(op)
    derivative = op !== :interpolate
    σ_out = derivative ? -parity : parity
    w = derivative ? taps ./ T(h) : taps

    # Output rows whose taps cross a wall, with the mirrored reads summed per
    # input index; the dummy midpoint slot at the high wall is a zero row.
    edge = zeros(Int, n)
    edge_index = Vector{Int}[]
    edge_weight = Vector{T}[]
    last_input = mid_in && hi ? n - 1 : n
    for i in 1:n
        dummy = hi && mid_out && i == n
        crosses = false
        for (m, wm) in enumerate(w)
            j = i + first + m - 1
            crosses |= (lo && j < 1) || (hi && (mid_in ? j >= n : j > n))
        end
        (dummy || crosses) || continue
        index = Int[]; weight = T[]
        if !dummy
            for (m, wm) in enumerate(w)
                j, s = _mirror_tap(i + first + m - 1, n, lo, hi, mid_in, parity)
                1 <= j <= last_input || error(
                    "local extent $n along dim $dim too small for the staggered " *
                    "wall mirror; use fewer ranks in this dimension")
                k = findfirst(==(j), index)
                if k === nothing
                    push!(index, j); push!(weight, s * wm)
                else
                    weight[k] += s * wm
                end
            end
        end
        push!(edge_index, index); push!(edge_weight, weight)
        edge[i] = length(edge_index)
    end

    α = derivative ? T(STAGGERED_ALPHA) : zero(T)
    a = fill(α, n); b = fill(one(T), n); c = fill(α, n)
    if mid_out
        # Midpoint 1 couples to the mirrored midpoint 0 = σ_out · midpoint 1;
        # midpoint N - 1 to midpoint N = σ_out · midpoint N - 1, and the slot
        # of midpoint N is a decoupled identity row.
        lo && (b[1] += σ_out * α)
        if hi
            b[n-1] += σ_out * α; c[n-1] = zero(T)
            a[n] = zero(T); b[n] = one(T); c[n] = zero(T)
        end
    else
        # Node 1 couples to the mirrored node 0 = σ_out · node 2, node N to
        # node N + 1 = σ_out · node N - 1.
        lo && (c[1] += σ_out * α)
        hi && (a[n] += σ_out * α)
    end
    aL = lo ? zero(T) : α
    cR = hi ? zero(T) : α
    lines = prod(decomp.n_local[k] for k in 1:3 if k != dim)
    line_solver = LineSolver(a, b, c, aL, cR, decomp.sub[dim], decomp.sub_size[dim],
                             decomp.sub_rank[dim], lines; periodic=decomp.periodic[dim],
                             explicit=!derivative)
    tr = dim > 1
    StaggeredPlan{T}(dim, n, lines, tr, op, parity, first, w, edge, edge_index,
                     edge_weight, line_solver,
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

# The staggered operator of `plan` applied to `f` along `plan.dim`, into the
# interior of `out`. `f` must have current rank-boundary halos along the
# dimension; beyond a wall the mirror is read from the interior, so a closed
# end's halo is never read. The derivatives' line solve is collective over the
# sub-communicator along a decomposed dimension. (A comment, not a docstring:
# the method would otherwise join the rendered `apply_along!` entry.)
function apply_along!(out, plan::StaggeredPlan, f, decomp::Decomp)
    d = plan.dim
    if d == 1
        _fill_staggered!(plan.B, plan, f, decomp, Val(1))
        solve_lines!(plan.B, plan.line_solver)
        _scatter_lines!(out, plan.B, plan, decomp, Val(1))
    elseif d == 2
        _fill_staggered!(plan.B, plan, f, decomp, Val(2))
        solve_lines_t!(plan.B, plan.line_solver)
        _scatter_t!(out, plan.B, plan, decomp, Val(2))
    else
        _fill_staggered!(plan.B, plan, f, decomp, Val(3))
        solve_lines_t!(plan.B, plan.line_solver)
        _scatter_t!(out, plan.B, plan, decomp, Val(3))
    end
    return out
end

"""
    StaggeredDiffusion(decomp, dim, h; parity=1)

The conservative diffusion operator `∂(κ ∂T)` along `dim` in its staggered
form `G K D_s`: the three plans and the two midpoint fields it works in.
`parity` is the temperature's parity at the mirror planes of a closed
dimension (+1 the adiabatic wall); the flux takes the opposite parity and the
coefficient is even. Applied by `staggered_diffusion!`.
"""
struct StaggeredDiffusion{T,A}
    to_mid::StaggeredPlan{T}
    to_node::StaggeredPlan{T}
    interpolate::StaggeredPlan{T}
    flux::A
    kappa_mid::A
end

function StaggeredDiffusion(decomp::Decomp, dim::Int, h::Real; parity::Int=1)
    StaggeredDiffusion(plan_staggered(decomp, :to_mid, dim, h; parity=parity),
                       plan_staggered(decomp, :to_node, dim, h; parity=-parity),
                       plan_staggered(decomp, :interpolate, dim, h; parity=1),
                       field(decomp), field(decomp))
end

"""
    staggered_diffusion!(out, op, T_ion, kappa, decomp)

Write `G K D_s T_ion`, the staggered approximation of `∂(κ ∂T_ion)` along
`op`'s dimension, into the interior of `out`, with `K` the midpoint
interpolation of `kappa`. Exchanges the halos of `T_ion`, `kappa` and the
midpoint flux along the dimension, so every rank of the sub-communicator
must call it.
"""
function staggered_diffusion!(out, op::StaggeredDiffusion, T_ion, kappa, decomp::Decomp)
    d = op.to_mid.dim
    exchange_dim!(T_ion, decomp, d)
    exchange_dim!(kappa, decomp, d)
    apply_along!(op.flux, op.to_mid, T_ion, decomp)
    apply_along!(op.kappa_mid, op.interpolate, kappa, decomp)
    op.flux .*= op.kappa_mid
    exchange_dim!(op.flux, decomp, d)
    apply_along!(out, op.to_node, op.flux, decomp)
    return out
end
