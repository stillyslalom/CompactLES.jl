# Generalized coordinate-singularity folds.
#
# The axisymmetric cylindrical axis was regularized by parity: each radial
# line continues through r = 0 into itself, so ghost values are mirror-filled
# and the implicit ghost coupling folds onto the matrix diagonal. With the
# angular dimension resolved, a line no longer continues into itself but into
# its ANTIPODAL PARTNER:
#
#   cylindrical axis:   (−r, θ)      ≡ (r, θ + π)
#   spherical origin:   (−r, θ, φ)   ≡ (r, π − θ, φ + π)
#   spherical poles:    (−θ, φ)      ≡ (θ, φ + π)        (both ends of θ)
#
# The even/odd trick makes this the SAME problem as the axisymmetric fold.
# For a field f with pairing map M (partner slot) and vector sign σ (how the
# component transforms under the antipodal basis change), define on the
# canonical side
#
#   e(x) = ½[f(x) + σ f(Mx)],    o(x) = ½[f(x) − σ f(Mx)].
#
# e is even across the singular point along the folded dimension and o is
# odd, so each solves with the existing parity-folded plans (halo mirror fill
# plus the diagonal LHS fold, per solution parity). Reconstruction is the
# inverse butterfly: (op f)(x) = Re + Ro and (op f)(Mx) = σ (Re − Ro) — the
# operator's own parity behavior is already inside the folded closures, so
# the reconstruction sign is just σ, for derivatives and filters alike.
#
# Component signs σ (scalars are always +1):
#   cyl axis:    (u_r, u_θ, u_z) → (−1, −1, +1)
#   sph origin:  (u_r, u_θ, u_φ) → (−1, +1, −1)   [ê_θ is invariant at the
#                antipode; ê_r and ê_φ flip]
#   sph poles:   (u_r, u_θ, u_φ) → (+1, −1, −1)
#
# Parallel layout restrictions (validated at setup):
#   - the shift (pairing) dimension has an even point count spanning its full
#     period, and is either on one rank or split into an even number of
#     uniform blocks (partner rank = +P/2, same local slot);
#   - the reversed dimension of the spherical origin (θ) is either on one
#     rank or split into uniform blocks (partner rank reflects, local slot
#     reverses);
#   - folded dimensions use half-offset grids (no node on the singular set).
#
# Cost: when the partner block is on-rank, folded-dimension operators run
# both parity plans and select per half (2× that operator). Off-rank, each
# application costs two full-block pairwise exchanges and one parity plan.

"Pairing/partner description for one folded dimension."
struct PairSpec{T,A<:AbstractArray{T,3}}
    pdim::Int                 # shift-by-half-period dimension
    revdim::Int               # dimension index-reversed in the partner (0 = none)
    shift_local::Bool         # pdim shift resolved within this rank's block
    rev_local::Bool           # revdim reversal resolved within this block
    local_pair::Bool          # shift_local && rev_local: partner is this rank
    partner::Int              # partner rank in decomp.comm (when !local_pair)
    keep_e::Bool              # this rank carries the even combo (off-rank case)
    buf::A                    # partner-block scratch
end

"Fold description for one dimension: which ends fold, the pairing (nothing
for the axisymmetric self-paired case), component signs, and parity plans."
mutable struct FoldSpec{P,D,F,S}
    dim::Int
    lo::Bool
    hi::Bool
    pair::P
    sigvel::NTuple{3,Int}                 # antipodal signs of (u, v, w)
    sigflux::Vector{Int}                  # antipodal flux sign per conserved comp
    deriv_plans::D                         # derivative plans, σg = (+1, −1)
    filter_plans::F                        # filter plans,     σg = (+1, −1)
    smooth_plans::S                        # sensor-smoother plans, σg = (+1, −1)
end

fold_dplan(fold::FoldSpec, σg::Int) = fold.deriv_plans[σg > 0 ? 1 : 2]
fold_fplan(fold::FoldSpec, σg::Int) = fold.filter_plans[σg > 0 ? 1 : 2]
fold_splan(fold::FoldSpec, σg::Int) = fold.smooth_plans[σg > 0 ? 1 : 2]

# --- Halo mirror fill (generalizes the old axis_fill!) ----------------------

"""
    fold_fill!(f, decomp, d, lo, hi, σ)

Mirror-fill the halos of `f` beyond the folded end(solver) of dimension `d` with
sign `σ` (half-offset mirror: ghost layer j ↔ interior layer j). Only acts on
ranks owning the corresponding global edge.
"""
function fold_fill!(f, decomp::Decomp, d::Int, lo::Bool, hi::Bool, σ::Int)
    pad = decomp.n_halo_d[d]
    n = decomp.n_local[d]
    sgn = Float64(σ)
    if lo && decomp.sub_rank[d] == 0
        if d == 1
            @inbounds for k in axes(f, 3), j in axes(f, 2), q in 1:pad
                f[pad-q+1, j, k] = sgn * f[pad+q, j, k]
            end
        elseif d == 2
            @inbounds for k in axes(f, 3), i in axes(f, 1), q in 1:pad
                f[i, pad-q+1, k] = sgn * f[i, pad+q, k]
            end
        else
            @inbounds for j in axes(f, 2), i in axes(f, 1), q in 1:pad
                f[i, j, pad-q+1] = sgn * f[i, j, pad+q]
            end
        end
    end
    if hi && decomp.sub_rank[d] == decomp.sub_size[d] - 1
        if d == 1
            @inbounds for k in axes(f, 3), j in axes(f, 2), q in 1:pad
                f[pad+n+q, j, k] = sgn * f[pad+n-q+1, j, k]
            end
        elseif d == 2
            @inbounds for k in axes(f, 3), i in axes(f, 1), q in 1:pad
                f[i, pad+n+q, k] = sgn * f[i, pad+n-q+1, k]
            end
        else
            @inbounds for j in axes(f, 2), i in axes(f, 1), q in 1:pad
                f[i, j, pad+n+q] = sgn * f[i, j, pad+n-q+1]
            end
        end
    end
    return f
end

# --- Pairing transforms -----------------------------------------------------

# Partner slot map within local blocks: shift pdim by half (on-rank case) and
# reverse revdim. `I` and the returned index are full-array (halo-offset).
# The full pairing map M factors into a pdim half-shift and an optional
# revdim reversal; each component is applied within the local block when the
# corresponding dimension is undecomposed (locally resolved), and by rank
# pairing otherwise. This helper applies exactly the locally-resolved parts.
@inline function _pair_index(I::CartesianIndex{3}, decomp::Decomp, pair::PairSpec)
    i1, i2, i3 = Tuple(I)
    if pair.pdim != 0 && pair.shift_local
        pd = pair.pdim
        half = decomp.n_local[pd] ÷ 2
        Hp = decomp.n_halo_d[pd]
        v = (pd == 1 ? i1 : pd == 2 ? i2 : i3) - Hp
        v = v <= half ? v + half : v - half
        v += Hp
        i1 = pd == 1 ? v : i1; i2 = pd == 2 ? v : i2; i3 = pd == 3 ? v : i3
    end
    if pair.revdim != 0
        # The reversal g → n_global+1−g always flips the intra-block slot
        # i ↔ n_local+1−i. When revdim is split (rev_local == false) the partner
        # rank is the reflected one (chosen at setup) and the slot flip is
        # STILL required to index its block; when it is on one rank the flip is
        # the whole reversal. (The half-period shift differs: split blocks map
        # slot-for-slot, so that stays gated on shift_local above.)
        rd = pair.revdim
        Hr = decomp.n_halo_d[rd]
        v = decomp.n_local[rd] + 1 - ((rd == 1 ? i1 : rd == 2 ? i2 : i3) - Hr) + Hr
        i1 = rd == 1 ? v : i1; i2 = rd == 2 ? v : i2; i3 = rd == 3 ? v : i3
    end
    CartesianIndex(i1, i2, i3)
end

"""
    pair_forward!(w, f, solver, fold, σ)

Load the even/odd combination of `f` (interior only) into `w` for the fold of
`fold`, using field sign `σ`. On-rank pairing: lower pdim half of `w` holds e,
upper half holds o (canonically indexed on the lower half). Off-rank: `w`
holds this rank's designated combo after a pairwise block exchange.
"""
function pair_forward!(w, f, solver, fold::FoldSpec, σ::Int)
    decomp = solver.decomp
    pair = fold.pair
    sf = Float64(σ)
    if pair.local_pair
        # Butterfly in place: the lower half of the locally-active
        # mapping dimension holds e (canonically indexed), upper half o.
        sd = pair.pdim != 0 ? pair.pdim : pair.revdim
        half = decomp.n_local[sd] ÷ 2
        Hp = decomp.n_halo_d[sd]
        r1, r2, r3 = interior(decomp).indices
        @threaded prod(decomp.n_local) for i23 in outer_indices(r2, r3)
            i2, i3 = Tuple(i23)
            @inbounds for i1 in r1
                I = CartesianIndex(i1, i2, i3)
                v = (sd == 1 ? i1 : sd == 2 ? i2 : i3) - Hp
                if v <= half
                    Mx = _pair_index(I, decomp, pair)
                    a = f[I]
                    b = sf * f[Mx]
                    w[I] = 0.5 * (a + b)      # e at the canonical slot
                    w[Mx] = 0.5 * (a - b)     # o stored at the partner slot
                end
            end
        end
    else
        # Full-block pairwise exchange, then this rank's designated combo.
        # e-keeper: e(x) = ½[f(x) + σ buf(Mx)];
        # o-keeper (slots y = Mx): o(My) = ½[buf(My) − σ f(y)].
        MPI.Sendrecv!(f, pair.buf, decomp.comm; dest=pair.partner, source=pair.partner,
                      sendtag=41, recvtag=41)
        r1, r2, r3 = interior(decomp).indices
        @threaded prod(decomp.n_local) for i23 in outer_indices(r2, r3)
            i2, i3 = Tuple(i23)
            @inbounds for i1 in r1
                I = CartesianIndex(i1, i2, i3)
                Mx = _pair_index(I, decomp, pair)
                if pair.keep_e
                    w[I] = 0.5 * (f[I] + sf * pair.buf[Mx])
                else
                    w[I] = 0.5 * (pair.buf[Mx] - sf * f[I])
                end
            end
        end
    end
    return w
end

"""
    pair_backward!(out, solver, fold, σ)

Inverse butterfly applied in place to the operator result `out`:
final(x) = Re + Ro, final(Mx) = σ (Re − Ro). Off-rank: one pairwise exchange
of result blocks.
"""
function pair_backward!(out, solver, fold::FoldSpec, σ::Int)
    decomp = solver.decomp
    pair = fold.pair
    sf = Float64(σ)
    if pair.local_pair
        sd = pair.pdim != 0 ? pair.pdim : pair.revdim
        half = decomp.n_local[sd] ÷ 2
        Hp = decomp.n_halo_d[sd]
        r1, r2, r3 = interior(decomp).indices
        @threaded prod(decomp.n_local) for i23 in outer_indices(r2, r3)
            i2, i3 = Tuple(i23)
            @inbounds for i1 in r1
                I = CartesianIndex(i1, i2, i3)
                v = (sd == 1 ? i1 : sd == 2 ? i2 : i3) - Hp
                if v <= half
                    Mx = _pair_index(I, decomp, pair)
                    Re = out[I]
                    Ro = out[Mx]
                    out[I] = Re + Ro
                    out[Mx] = sf * (Re - Ro)
                end
            end
        end
    else
        MPI.Sendrecv!(out, pair.buf, decomp.comm; dest=pair.partner, source=pair.partner,
                      sendtag=42, recvtag=42)
        r1, r2, r3 = interior(decomp).indices
        @threaded prod(decomp.n_local) for i23 in outer_indices(r2, r3)
            i2, i3 = Tuple(i23)
            @inbounds for i1 in r1
                I = CartesianIndex(i1, i2, i3)
                Mx = _pair_index(I, decomp, pair)
                if pair.keep_e
                    out[I] = out[I] + pair.buf[Mx]         # Re + Ro
                else
                    out[I] = sf * (pair.buf[Mx] - out[I])  # σ(Re − Ro)
                end
            end
        end
    end
    return out
end

# --- Folded operator application -------------------------------------------

# The ghost parity σg is the field parity σ for a symmetric operator and its
# negation for an antisymmetric one; the role selects both that and which plan
# pair to draw from. `Val` keeps the choice static, since every call site fixes
# it. The smoother pair is built like the filter pair even when its scheme is
# explicit, where the two parities coincide numerically because an identity
# left-hand side has no ghost coupling to fold onto the diagonal. They are kept
# as separate plans rather than one shared twice, so that neither aliases the
# other's line scratch.
@inline _fold_plan(fold::FoldSpec, σ::Int, ::Val{:deriv})  = fold_dplan(fold, -σ)
@inline _fold_plan(fold::FoldSpec, σ::Int, ::Val{:filter}) = fold_fplan(fold, σ)
@inline _fold_plan(fold::FoldSpec, σ::Int, ::Val{:smooth}) = fold_splan(fold, σ)

"""
    fold_apply!(out, f, solver, fold, σ, role = Val(:deriv))

Apply the compact operator of `role` — `:deriv`, `:filter` or `:smooth` —
along the folded dimension of `fold` to field `f` (current halos required)
with antipodal sign `σ`. Self-paired (axisymmetric) folds reduce to mirror
fill plus one folded plan. Paired folds run the even/odd butterfly; the input
`f` is left untouched (the combo lives in scratch `s.pairbuf`).
"""
function fold_apply!(out, f, solver, fold::FoldSpec, σ::Int, role::Val=Val(:deriv))
    decomp = solver.decomp
    d = fold.dim
    if fold.pair === nothing
        fold_fill!(f, decomp, d, fold.lo, fold.hi, σ)
        apply_along!(out, _fold_plan(fold, σ, role), f, decomp)
        return out
    end
    pair = fold.pair
    w = solver.pairbuf
    pair_forward!(w, f, solver, fold, σ)
    exchange_dim_batch!([w], decomp, d)   # rank-boundary halos along the fold dim
    if pair.local_pair
        # Mixed parities per pdim half: run both plans and select. The even
        # combo e carries the field's antipodal sign σ across the r-mirror,
        # while the odd combo o is ALWAYS odd (o(−r,θ)=−o(r,θ) holds for either
        # σ, and for a θ-independent radial field the whole flux lands in o and
        # must fold as odd). So the e half folds with σ and the o half with −1.
        fold_fill!(w, decomp, d, fold.lo, fold.hi, σ)
        apply_along!(out, _fold_plan(fold, σ, role), w, decomp)
        fold_fill!(w, decomp, d, fold.lo, fold.hi, -1)
        apply_along!(solver.pairout, _fold_plan(fold, -1, role), w, decomp)
        sd = pair.pdim != 0 ? pair.pdim : pair.revdim
        half = decomp.n_local[sd] ÷ 2
        Hp = decomp.n_halo_d[sd]
        r1, r2, r3 = interior(decomp).indices
        @threaded prod(decomp.n_local) for i23 in outer_indices(r2, r3)
            i2, i3 = Tuple(i23)
            @inbounds for i1 in r1
                v = (sd == 1 ? i1 : sd == 2 ? i2 : i3) - Hp
                if v > half
                    I = CartesianIndex(i1, i2, i3)
                    out[I] = solver.pairout[I]
                end
            end
        end
    else
        # e combo mirrors with σ, o combo is ALWAYS odd (−1) — matching the
        # local branch above, where the o half uses fold_fill(−1)/plan(−1)
        # regardless of σ. (Writing −σ here silently works only for σ = +1.)
        σc = pair.keep_e ? σ : -1
        fold_fill!(w, decomp, d, fold.lo, fold.hi, σc)
        apply_along!(out, _fold_plan(fold, σc, role), w, decomp)
    end
    pair_backward!(out, solver, fold, σ)
    return out
end
