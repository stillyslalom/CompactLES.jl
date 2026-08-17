# AMR level-transfer operators (reference/AMR_GPU.md, Stage 1).
#
# The 3:1 level transfer is an invertible compact filter pair transcribed from
# Miranda (LLNL/pyranda@b4e0afc, pyranda/parcop/stencils.f90, cfamrcf/cfamrfc,
# inside an `#if 0` block):
#
#   α f̄_{i-1} + f̄_i + α f̄_{i+1} = c f_{i-2} + b f_{i-1} + a f_i + b f_{i+1} + c f_{i+2}
#
# whose transfer function matches a Gaussian filter of width 3Δx, the coarse
# spacing. Both operators run on the FINE grid: restriction solves the
# tridiagonal [α, 1, α] side for f̄ given f (an ordinary compact filter
# application); prolongation solves the pentadiagonal [c, b, a, b, c] side for
# f given f̄ (deconvolution). The Gaussian symbol a + 2b cos k + 2c cos 2k is
# positive over [0, π] (0.0526 at k = π), so the pentadiagonal side is
# symmetric positive definite and factorizes without pivoting; its inverse
# amplifies the fine Nyquist by ≈ 20 (`bench/amr_transfer.jl`).
#
# Sampling convention (risk 1 of the plan, resolved here and pinned by the
# bench and the test gate): the grid is node-centered with refinement ratio 3,
# so coarse node m coincides with fine node 3m − 2 and a closed line has
# n_fine = 3 n_coarse − 2 (a periodic one n_fine = 3 n_coarse). Restriction is
# filter-then-subsample: apply the compact filter along the fine line, then
# take the coincident nodes. Prolongation is interpolate-then-deconvolve: the
# coarse values are samples of the filtered field f̄, so Lagrange interpolation
# on the coarse nodes fills f̄ at the two intermediate fine nodes of each
# coarse interval, and the deconvolution solve recovers f on the whole fine
# line. Because the interpolant reproduces its data at the coincident nodes and
# the filter pair round-trips to machine precision, restriction is a left
# inverse of prolongation: coarse → fine → coarse is exact to solver precision
# for any coarse data, with no refluxing machinery. Fine → coarse → fine loses
# the content above the coarse Nyquist to subsampling, as any 3:1 transfer
# must; on smooth data the error is set by the interpolation order.
#
# Boundary closures: the source tabulates four variants per end on a -1:2
# index. Variant 0 (one-sided) and variant 2 ("with extended boundary data:
# same as one-sided to maintain invertibility") are numerically identical, so
# one closure set serves both a physical edge and a patch-interface edge. The
# symmetric variants ±1 are built by lower_symm_weights/upper_symm_weights,
# the same ghost-folding algebra `plan_direction` performs for `lo_fold` /
# `hi_fold`, so a folded end needs no tabulated rows here. `plan_transfer`
# itself never builds fold plans: refinement across a fold's singular region
# is forbidden (constraint 4 of the plan), so a transfer dimension is closed
# or periodic. The source's row 3 per end is the interior stencil, so each
# scheme carries two genuine closure rows.

# Interior coefficients of the transfer pair. The zero-wavenumber gains agree
# to the last bit: a + 2b + 2c == 1 + 2α == 0.9356346489741322, so the pair
# preserves the mean exactly (conservation by unit DC gain, not refluxing).
const AMR_ALPHA = -0.0321826755129339
const AMR_A = 0.4451523642186118
const AMR_B = 0.2207614172195584
const AMR_C = 0.0244797251582018

# One-sided closure rows, applied to whichever side of the equation carries
# the Gaussian pentadiagonal: row 1 is explicit and sums to exactly 1.0; row
# 2's weights sum to 1 + 2α within one unit in the last place. The compact
# tridiagonal side keeps the identity on row 1 and its interior stencil from
# row 2 on. Source comment: "no filter (current) or bad filter at boundary
# causes ringing in the solution".
const AMR_EDGE_ROW1 = (1.375, -0.75, 0.375)
const AMR_EDGE_ROW2 = (0.3186803178523657, 0.2982740132694008, 0.3186803178523657)

"""
    amr_restriction_scheme(T=Float64)

Fine-to-coarse AMR transfer filter (Miranda's `cfamrfc`) as a
[`CompactScheme`](@ref): the tridiagonal `[α, 1, α]` left-hand side against
the pentadiagonal Gaussian right-hand side, with the tabulated one-sided
closure rows. Applying it along a fine line and taking every third node is
the level restriction; see [`plan_transfer`](@ref) for the bound form.
"""
function amr_restriction_scheme(::Type{T}=Float64) where {T}
    α = T(AMR_ALPHA)
    CompactScheme{T}("AMR 3:1 fine-to-coarse transfer filter", α, T(AMR_A),
        T[AMR_B, AMR_C], true,
        [ClosureRow{T}((zero(T), one(T), zero(T)), T[AMR_EDGE_ROW1...]),
         ClosureRow{T}((α, one(T), α), T[AMR_EDGE_ROW2...])])
end

"""
    amr_prolongation_scheme(T=Float64)

Coarse-to-fine AMR deconvolution (Miranda's `cfamrcf`) as a
[`BandedCompactScheme`](@ref) with `q = 2`: the pentadiagonal Gaussian
left-hand side against the tridiagonal `[α, 1, α]` right-hand side — the
same coefficients as [`amr_restriction_scheme`](@ref) with the two sides
swapped, which is what makes the pair invertible. Interior rows are divided
by the Gaussian center weight `a` so the left-hand-side diagonal is one, as
`BandedCompactScheme` requires; the closure rows enter verbatim, since
scaling a row of a linear system leaves its solution unchanged.
"""
function amr_prolongation_scheme(::Type{T}=Float64) where {T}
    a = T(AMR_A)
    α = T(AMR_ALPHA)
    BandedCompactScheme{T}("AMR 3:1 coarse-to-fine deconvolution", 2,
        T[T(AMR_B) / a, T(AMR_C) / a],
        inv(a),
        T[α / a],
        true,
        [BandedClosureRow{T}(T[0, 0, AMR_EDGE_ROW1...], T[1]),
         BandedClosureRow{T}(T[0, AMR_EDGE_ROW2..., 0], T[α, 1, α])])
end

"""
    amr_transfer_schemes(T=Float64)

The `(restriction, prolongation)` pair —
[`amr_restriction_scheme`](@ref) and [`amr_prolongation_scheme`](@ref).
"""
amr_transfer_schemes(::Type{T}=Float64) where {T} =
    (amr_restriction_scheme(T), amr_prolongation_scheme(T))

"""
    amr_interpolation_weights(T, order)

Lagrange interpolation weights filling the filtered field at the two
intermediate fine nodes of each coarse interval, evaluated at ξ = 1/3 and
ξ = 2/3 of the interval on a centered stencil of `order` coarse nodes
(shifted inward near a closed edge). Computed in exact rational arithmetic
and returned as `W[point, sub-node, class]`, where `class = r + 1` and `r`
is the offset of the interval's low node within the stencil; the centered
interior class is `r = order ÷ 2 - 1`. Each weight row sums to exactly one
before conversion to `T`.
"""
function amr_interpolation_weights(::Type{T}, order::Int) where {T}
    (iseven(order) && 2 <= order <= 8) ||
        error("interpolation order must be even and between 2 and 8, got $order")
    p = order
    W = Array{T,3}(undef, p, 2, p - 1)
    for r in 0:(p-2), (sub, ξ) in enumerate((1 // 3, 2 // 3))
        nodes = (0:(p-1)) .- r
        for j in 1:p
            w = 1 // 1
            for s in nodes
                s == nodes[j] && continue
                w *= (ξ - s) // (nodes[j] - s)
            end
            W[j, sub, r + 1] = T(w)
        end
    end
    return W
end

"""
    TransferPlan

Bound form of the 3:1 level transfer along one dimension: the restriction and
prolongation plans on the fine decomposition, the interpolation weight table,
and a fine-grid scratch field. Constructed by [`plan_transfer`](@ref) and
consumed by [`restrict!`](@ref) and [`prolong!`](@ref).
"""
struct TransferPlan{T}
    dim::Int
    fine::Decomp
    coarse::Decomp
    restrict_plan::DirPlan{T}
    prolong_plan::BandPlan{T}
    interp_order::Int
    weights::Array{T,3}    # (point, sub-node, edge class); amr_interpolation_weights
    tmp::Array{T,3}        # fine-grid scratch: the filtered / interpolated field
end

"""
    plan_transfer(fine, coarse, dim, T=Float64; interp_order=6)

Build a [`TransferPlan`](@ref) between a fine and a coarse [`Decomp`](@ref)
along dimension `dim`. The grids are node-centered at refinement ratio 3, so
coarse node `m` coincides with fine node `3m − 2`; a closed transfer
dimension requires `n_fine = 3 n_coarse − 2` globally and a periodic one
`n_fine = 3 n_coarse`. Every other dimension must agree between the two
decompositions rank by rank.

The transfer dimension must not be decomposed: distributing the subsampling
and interpolation requires the per-rank 3:1 alignment the patch machinery of
Stage 2/3 will own. Transverse dimensions may be decomposed freely, and the
compact solves themselves remain collective over them. Folds are excluded on
a transfer dimension (refinement across a fold's singular region is
forbidden); the schemes accept `lo_fold`/`hi_fold` through
[`plan_direction`](@ref) directly should a future stage lift that.

`interp_order` (even, 2–8, default 6) sets the Lagrange stencil width of the
coarse-to-fine interpolation and therefore the measured order of the
fine → coarse → fine round trip on smooth data.
"""
function plan_transfer(fine::Decomp, coarse::Decomp, dim::Int,
                       ::Type{T}=Float64; interp_order::Int=6) where {T}
    fine.active[dim] || error("transfer dimension $dim is collapsed")
    fine.active == coarse.active ||
        error("fine and coarse decompositions disagree on active dimensions")
    fine.periodic == coarse.periodic ||
        error("fine and coarse decompositions disagree on periodicity")
    fine.sub_size[dim] == 1 && coarse.sub_size[dim] == 1 ||
        error("the transfer dimension must not be decomposed; distributed " *
              "transfer arrives with the patch stages (reference/AMR_GPU.md)")
    for d in 1:3
        d == dim && continue
        fine.n_global[d] == coarse.n_global[d] &&
            fine.n_local[d] == coarse.n_local[d] &&
            fine.offset[d] == coarse.offset[d] ||
            error("transverse dimension $d does not match between fine and coarse")
    end
    nc = coarse.n_global[dim]
    nf_expected = fine.periodic[dim] ? 3nc : 3nc - 2
    fine.n_global[dim] == nf_expected ||
        error("fine extent $(fine.n_global[dim]) along dim $dim does not match " *
              "3:1 refinement of $nc coarse nodes (expected $nf_expected)")
    if fine.periodic[dim]
        interp_order ÷ 2 <= coarse.n_halo ||
            error("interpolation order $interp_order needs coarse halo ≥ " *
                  "$(interp_order ÷ 2)")
    else
        nc >= interp_order ||
            error("coarse extent $nc along dim $dim is below interpolation " *
                  "order $interp_order")
    end
    restrict_plan = plan_direction(fine, amr_restriction_scheme(T), dim, 1)
    prolong_plan = plan_direction(fine, amr_prolongation_scheme(T), dim, 1)
    weights = amr_interpolation_weights(T, interp_order)
    tmp = zeros(T, ntuple(d -> fine.n_local[d] + 2 * fine.n_halo_d[d], 3))
    TransferPlan{T}(dim, fine, coarse, restrict_plan, prolong_plan,
                    interp_order, weights, tmp)
end

function _subsample!(coarse_field, filtered, plan::TransferPlan{T},
                     ::Val{D}) where {T,D}
    padf = plan.fine.n_halo_d
    padc = plan.coarse.n_halo_d
    nc = plan.coarse.n_local[D]
    o1, o2 = _odims(Val(D))
    n1 = plan.coarse.n_local[o1]
    n2 = plan.coarse.n_local[o2]
    @threaded n1*n2*nc for jk in outer_indices(n1, n2)
        j, k = Tuple(jk)
        @inbounds for m in 1:nc
            coarse_field[_gidx(Val(D), m, j, k, padc)] =
                filtered[_gidx(Val(D), 3m - 2, j, k, padf)]
        end
    end
    return coarse_field
end

function _inject_interpolate!(tmp, coarse_field, plan::TransferPlan{T},
                              ::Val{D}) where {T,D}
    padf = plan.fine.n_halo_d
    padc = plan.coarse.n_halo_d
    nc = plan.coarse.n_local[D]
    periodic = plan.coarse.periodic[D]
    p = plan.interp_order
    half = p ÷ 2
    W = plan.weights
    nintervals = periodic ? nc : nc - 1
    o1, o2 = _odims(Val(D))
    n1 = plan.coarse.n_local[o1]
    n2 = plan.coarse.n_local[o2]
    @threaded 3*n1*n2*nc for jk in outer_indices(n1, n2)
        j, k = Tuple(jk)
        @inbounds begin
            for m in 1:nc
                tmp[_gidx(Val(D), 3m - 2, j, k, padf)] =
                    coarse_field[_gidx(Val(D), m, j, k, padc)]
            end
            for m in 1:nintervals
                js = periodic ? m - (half - 1) : clamp(m - (half - 1), 1, nc - p + 1)
                r = m - js
                for sub in 1:2
                    acc = zero(T)
                    for jj in 1:p
                        acc += W[jj, sub, r + 1] *
                               coarse_field[_gidx(Val(D), js + jj - 1, j, k, padc)]
                    end
                    tmp[_gidx(Val(D), 3m - 2 + sub, j, k, padf)] = acc
                end
            end
        end
    end
    return tmp
end

"""
    restrict!(coarse_field, plan, fine_field)

Level restriction along `plan.dim`: exchange the fine field's halos along that
dimension, apply the fine-to-coarse transfer filter, and copy the coincident
nodes into the interior of `coarse_field`, returning it. `plan.tmp` is scratch
and is overwritten. Collective over ranks sharing transverse decompositions,
like any compact solve.
"""
function restrict!(coarse_field, plan::TransferPlan, fine_field)
    exchange_dim!(fine_field, plan.fine, plan.dim)
    apply_along!(plan.tmp, plan.restrict_plan, fine_field, plan.fine)
    d = plan.dim
    if d == 1
        _subsample!(coarse_field, plan.tmp, plan, Val(1))
    elseif d == 2
        _subsample!(coarse_field, plan.tmp, plan, Val(2))
    else
        _subsample!(coarse_field, plan.tmp, plan, Val(3))
    end
    return coarse_field
end

"""
    prolong!(fine_field, plan, coarse_field)

Level prolongation along `plan.dim`: exchange the coarse field's halos along
that dimension, inject its values onto the coincident fine nodes, fill the
intermediate nodes by Lagrange interpolation of order `plan.interp_order`,
and deconvolve the result into the interior of `fine_field`, returning it.
`plan.tmp` is scratch and is overwritten. [`restrict!`](@ref) is a left
inverse of this map: restriction of the prolonged field reproduces
`coarse_field` to solver precision.
"""
function prolong!(fine_field, plan::TransferPlan, coarse_field)
    exchange_dim!(coarse_field, plan.coarse, plan.dim)
    d = plan.dim
    if d == 1
        _inject_interpolate!(plan.tmp, coarse_field, plan, Val(1))
    elseif d == 2
        _inject_interpolate!(plan.tmp, coarse_field, plan, Val(2))
    else
        _inject_interpolate!(plan.tmp, coarse_field, plan, Val(3))
    end
    exchange_dim!(plan.tmp, plan.fine, plan.dim)
    apply_along!(fine_field, plan.prolong_plan, plan.tmp, plan.fine)
    return fine_field
end
