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
# inverse butterfly: (op f)(x) = Re + Ro and (op f)(Mx) = σ (Re − Ro). The
# operator's own parity behavior is contained in the folded closures, so the
# reconstruction sign is σ for derivatives and filters alike.
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
#     rank or split into an even number of uniform blocks (partner rank
#     reflects, local slot reverses);
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
mutable struct FoldSpec{P,D,F,S,R}
    dim::Int
    lo::Bool
    hi::Bool
    pair::P
    sigvel::NTuple{3,Int}                 # antipodal signs of (u, v, w)
    sigflux::Vector{Int}                  # antipodal flux sign per conserved comp
    deriv_plans::D                         # derivative plans, σg = (+1, −1)
    filter_plans::F                        # filter plans,     σg = (+1, −1)
    smooth_plans::S                        # sensor-smoother plans, σg = (+1, −1)
    ring_plans::R                          # d8 detector plans, σg = (+1, −1);
                                           # (nothing, nothing) unless :d8
end

fold_dplan(fold::FoldSpec, σg::Int) = fold.deriv_plans[σg > 0 ? 1 : 2]
fold_fplan(fold::FoldSpec, σg::Int) = fold.filter_plans[σg > 0 ? 1 : 2]
fold_splan(fold::FoldSpec, σg::Int) = fold.smooth_plans[σg > 0 ? 1 : 2]
fold_rplan(fold::FoldSpec, σg::Int) = fold.ring_plans[σg > 0 ? 1 : 2]

# --- Halo mirror fill -------------------------------------------------------

"""
    fold_fill!(f, decomp, d, lo, hi, σ)

Mirror-fill the halos of `f` beyond the folded ends of dimension `d` with sign
`σ` (half-offset mirror: ghost layer j ↔ interior layer j). `lo` and `hi` select
which ends of `d` carry a fold.

`f` is written in place on the ranks owning the corresponding global edge and
left unchanged on the others; it is returned either way. No communication takes
place here, so this is not a collective call.
"""
function fold_fill!(f, decomp::Decomp, d::Int, lo::Bool, hi::Bool, σ::Int)
    pad = decomp.n_halo_d[d]
    n = decomp.n_local[d]
    sgn = eltype(f)(σ)
    o1, o2 = d == 1 ? (2, 3) : d == 2 ? (1, 3) : (1, 2)
    no1, no2 = size(f, o1), size(f, o2)
    if lo && decomp.sub_rank[d] == 0
        pointwise!(_fold_fill_point!, f, pad, no1, no2, f, sgn, d, pad, n, true)
    end
    if hi && decomp.sub_rank[d] == decomp.sub_size[d] - 1
        pointwise!(_fold_fill_point!, f, pad, no1, no2, f, sgn, d, pad, n, false)
    end
    return f
end

# One mirrored ghost write: `i` indexes the halo layer 1:pad, `j`/`k` the two
# orthogonal FULL axes in ascending dimension order (matching `_odims`).
@inline function _fold_fill_point!(f, sgn, d, pad, n, lo, i, j, k)
    @inbounds if lo
        if d == 1
            f[pad-i+1, j, k] = sgn * f[pad+i, j, k]
        elseif d == 2
            f[j, pad-i+1, k] = sgn * f[j, pad+i, k]
        else
            f[j, k, pad-i+1] = sgn * f[j, k, pad+i]
        end
    else
        if d == 1
            f[pad+n+i, j, k] = sgn * f[pad+n-i+1, j, k]
        elseif d == 2
            f[j, pad+n+i, k] = sgn * f[j, pad+n-i+1, k]
        else
            f[j, k, pad+n+i] = sgn * f[j, k, pad+n-i+1]
        end
    end
    return nothing
end

# --- Pairing transforms -----------------------------------------------------

# Partner slot map within local blocks: shift pdim by half (on-rank case) and
# reverse revdim. `I` and the returned index are full-array (halo-offset).
# The full pairing map M factors into a pdim half-shift and an optional
# revdim reversal; each component is applied within the local block when the
# corresponding dimension is undecomposed (locally resolved), and by rank
# pairing otherwise. This helper applies exactly the locally-resolved parts.
@inline _pair_index(I::CartesianIndex{3}, decomp::Decomp, pair::PairSpec) =
    _pair_slot(I, pair.pdim, pair.shift_local, pair.revdim,
               decomp.n_local, decomp.n_halo_d)

# The launchable form of `_pair_index`: plain values in place of the Decomp
# and the PairSpec (whose scratch array a kernel argument may not carry).
@inline function _pair_slot(I::CartesianIndex{3}, pdim::Int, shift_local::Bool,
                            revdim::Int, n_local::NTuple{3,Int},
                            halo::NTuple{3,Int})
    i1, i2, i3 = Tuple(I)
    if pdim != 0 && shift_local
        half = n_local[pdim] ÷ 2
        Hp = halo[pdim]
        v = (pdim == 1 ? i1 : pdim == 2 ? i2 : i3) - Hp
        v = v <= half ? v + half : v - half
        v += Hp
        i1 = pdim == 1 ? v : i1; i2 = pdim == 2 ? v : i2; i3 = pdim == 3 ? v : i3
    end
    if revdim != 0
        # The reversal g → n_global+1−g always flips the intra-block slot
        # i ↔ n_local+1−i. When revdim is split (rev_local == false) the partner
        # rank is the reflected one (chosen at setup) and the slot flip is
        # still required to index its block; when it is on one rank the flip is
        # the whole reversal. (The half-period shift differs: split blocks map
        # slot-for-slot, and remains gated on shift_local above.)
        Hr = halo[revdim]
        v = n_local[revdim] + 1 -
            ((revdim == 1 ? i1 : revdim == 2 ? i2 : i3) - Hr) + Hr
        i1 = revdim == 1 ? v : i1; i2 = revdim == 2 ? v : i2
        i3 = revdim == 3 ? v : i3
    end
    CartesianIndex(i1, i2, i3)
end

"""
    pair_forward!(w, f, solver, fold, σ)

Load the even/odd combination of `f` under the pairing of `fold` into `w`,
using field sign `σ`. `f` is read but not modified; only the interior of `w` is
written, and `w` is returned.

When the partner block is on this rank, the lower half of the mapping dimension
(`pdim`, or `revdim` when the shift is degenerate) receives e at the canonical
slot and the upper half receives o at the partner slot. Otherwise each rank
keeps the single combo named by `keep_e` after a pairwise full-block
`MPI.Sendrecv!`, so both partners must reach this call.
"""
function pair_forward!(w, f, solver, fold::FoldSpec, σ::Int)
    decomp = solver.decomp
    pair = fold.pair
    sf = eltype(f)(σ)
    o1, o2, o3 = decomp.n_halo_d
    nx, ny, nz = decomp.n_local
    if pair.local_pair
        # Butterfly in place: the lower half of the locally-active
        # mapping dimension holds e (canonically indexed), upper half o.
        sd = pair.pdim != 0 ? pair.pdim : pair.revdim
        half = decomp.n_local[sd] ÷ 2
        pointwise!(_pair_forward_local_point!, w, nx, ny, nz,
                   w, f, sf, sd, half, pair.pdim, pair.shift_local,
                   pair.revdim, decomp.n_local, decomp.n_halo_d, o1, o2, o3)
    else
        # Full-block pairwise exchange, then this rank's designated combo.
        # e-keeper: e(x) = ½[f(x) + σ buf(Mx)];
        # o-keeper (slots y = Mx): o(My) = ½[buf(My) − σ f(y)].
        sendrecv_block!(f, pair.buf, decomp, fold.dim, pair.partner, 41)
        pointwise!(_pair_forward_remote_point!, w, nx, ny, nz,
                   w, f, pair.buf, sf, pair.keep_e, pair.pdim,
                   pair.shift_local, pair.revdim, decomp.n_local,
                   decomp.n_halo_d, o1, o2, o3)
    end
    return w
end

@inline function _pair_forward_local_point!(w, f, sf, sd, half, pdim,
                                            shift_local, revdim, n_local, halo,
                                            o1, o2, o3, i, j, k)
    @inbounds begin
        v = sd == 1 ? i : sd == 2 ? j : k
        if v <= half
            I = CartesianIndex(i + o1, j + o2, k + o3)
            Mx = _pair_slot(I, pdim, shift_local, revdim, n_local, halo)
            a = f[I]
            b = sf * f[Mx]
            w[I] = (a + b) / oftype(a, 2)  # e at the canonical slot
            w[Mx] = (a - b) / oftype(a, 2) # o at the partner slot
        end
    end
    return nothing
end

@inline function _pair_forward_remote_point!(w, f, buf, sf, keep_e, pdim,
                                             shift_local, revdim, n_local,
                                             halo, o1, o2, o3, i, j, k)
    @inbounds begin
        I = CartesianIndex(i + o1, j + o2, k + o3)
        Mx = _pair_slot(I, pdim, shift_local, revdim, n_local, halo)
        if keep_e
            w[I] = (f[I] + sf * buf[Mx]) / oftype(sf, 2)
        else
            w[I] = (buf[Mx] - sf * f[I]) / oftype(sf, 2)
        end
    end
    return nothing
end

"""
    pair_backward!(out, solver, fold, σ)

Inverse butterfly applied in place to the interior of the operator result
`out`: final(x) = Re + Ro, final(Mx) = σ (Re − Ro). Halo cells are left
untouched and `out` is returned.

When the partner block is off-rank the result blocks are exchanged through one
pairwise `MPI.Sendrecv!`, so both partners must reach this call.
"""
function pair_backward!(out, solver, fold::FoldSpec, σ::Int)
    decomp = solver.decomp
    pair = fold.pair
    sf = eltype(out)(σ)
    o1, o2, o3 = decomp.n_halo_d
    nx, ny, nz = decomp.n_local
    if pair.local_pair
        sd = pair.pdim != 0 ? pair.pdim : pair.revdim
        half = decomp.n_local[sd] ÷ 2
        pointwise!(_pair_backward_local_point!, out, nx, ny, nz,
                   out, sf, sd, half, pair.pdim, pair.shift_local, pair.revdim,
                   decomp.n_local, decomp.n_halo_d, o1, o2, o3)
    else
        sendrecv_block!(out, pair.buf, decomp, fold.dim, pair.partner, 42)
        pointwise!(_pair_backward_remote_point!, out, nx, ny, nz,
                   out, pair.buf, sf, pair.keep_e, pair.pdim, pair.shift_local,
                   pair.revdim, decomp.n_local, decomp.n_halo_d, o1, o2, o3)
    end
    return out
end

@inline function _pair_backward_local_point!(out, sf, sd, half, pdim,
                                             shift_local, revdim, n_local,
                                             halo, o1, o2, o3, i, j, k)
    @inbounds begin
        v = sd == 1 ? i : sd == 2 ? j : k
        if v <= half
            I = CartesianIndex(i + o1, j + o2, k + o3)
            Mx = _pair_slot(I, pdim, shift_local, revdim, n_local, halo)
            Re = out[I]
            Ro = out[Mx]
            out[I] = Re + Ro
            out[Mx] = sf * (Re - Ro)
        end
    end
    return nothing
end

@inline function _pair_backward_remote_point!(out, buf, sf, keep_e, pdim,
                                              shift_local, revdim, n_local,
                                              halo, o1, o2, o3, i, j, k)
    @inbounds begin
        I = CartesianIndex(i + o1, j + o2, k + o3)
        Mx = _pair_slot(I, pdim, shift_local, revdim, n_local, halo)
        if keep_e
            out[I] = out[I] + buf[Mx]         # Re + Ro
        else
            out[I] = sf * (buf[Mx] - out[I])  # σ(Re − Ro)
        end
    end
    return nothing
end

# --- Folded operator application -------------------------------------------

# The ghost parity σg is the field parity σ for a symmetric operator and its
# negation for an antisymmetric one; the role selects both that and which plan
# pair to draw from. `Val` keeps the choice static, since every call site fixes
# it. The smoother pair is built like the filter pair even when its scheme is
# explicit, where the two parities coincide numerically because an identity
# left-hand side has no ghost coupling to fold onto the diagonal. They are kept
# as two separate plans, not one shared twice, preventing either from aliasing the
# other's line scratch. The ring detector is an eighth derivative, which is an
# even one and so preserves parity: it is planned with the symmetric roles,
# not with `:deriv`.
@inline _fold_plan(fold::FoldSpec, σ::Int, ::Val{:deriv})  = fold_dplan(fold, -σ)
@inline _fold_plan(fold::FoldSpec, σ::Int, ::Val{:filter}) = fold_fplan(fold, σ)
@inline _fold_plan(fold::FoldSpec, σ::Int, ::Val{:smooth}) = fold_splan(fold, σ)
@inline _fold_plan(fold::FoldSpec, σ::Int, ::Val{:ring})   = fold_rplan(fold, σ)

"""
    fold_apply!(out, f, solver, fold, σ, role = Val(:deriv))

Apply the compact operator selected by `role`, one of `Val(:deriv)`,
`Val(:filter)`, `Val(:smooth)` and `Val(:ring)`, along the folded dimension of
`fold` to field `f` with antipodal sign `σ`. `f` must carry current
rank-boundary halos. The result is written to the interior of `out`, which is
returned.

A self-paired (axisymmetric) fold reduces to a mirror fill plus one folded
plan, and so writes the folded-end halos of `f` in place. A paired fold runs
the even/odd butterfly instead and leaves `f` unmodified, taking
`solver.pairbuf` as scratch for the combination and, when the partner block is
on-rank, `solver.pairout` for the second parity result. Neither may alias `out`.

The line solves are collective along the folded dimension and a paired fold
adds pairwise exchanges, so every rank must reach this call.
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
    exchange_dim!(w, decomp, d)   # rank-boundary halos along the fold dim
    if pair.local_pair
        # Mixed parities per pdim half: run both plans and select. The line
        # continued through the singular point reads f(−r,θ) = σ f(Mx), so
        # e(−r,θ) = ½[σ f(Mx) + σ² f(x)] = e(x) and o(−r,θ) = −o(x): the even
        # even combination has mirror parity +1, and the odd combination has
        # mirror parity -1 (the header of this file derives this invariant).
        # The field parity σ affects only the butterfly and reconstruction.
        # Folding e with σ instead, as an
        # earlier version did, is correct only where e vanishes, which every
        # axisymmetric field and the odd-pairing scalar of the test suite
        # satisfy; a uniform Cartesian velocity through the axis has e = f and
        # its radial derivative came out O(1/h) at the axis.
        fold_fill!(w, decomp, d, fold.lo, fold.hi, 1)
        apply_along!(out, _fold_plan(fold, 1, role), w, decomp)
        fold_fill!(w, decomp, d, fold.lo, fold.hi, -1)
        apply_along!(solver.pairout, _fold_plan(fold, -1, role), w, decomp)
        sd = pair.pdim != 0 ? pair.pdim : pair.revdim
        half = decomp.n_local[sd] ÷ 2
        o1, o2, o3 = decomp.n_halo_d
        nx, ny, nz = decomp.n_local
        pointwise!(_pair_select_point!, out, nx, ny, nz,
                   out, solver.pairout, sd, half, o1, o2, o3)
    else
        # The e combo mirrors even and the o combo odd, as in the local
        # branch above; σ plays no part in the mirror.
        σc = pair.keep_e ? 1 : -1
        fold_fill!(w, decomp, d, fold.lo, fold.hi, σc)
        apply_along!(out, _fold_plan(fold, σc, role), w, decomp)
    end
    pair_backward!(out, solver, fold, σ)
    return out
end

@inline function _pair_select_point!(out, pairout, sd, half, o1, o2, o3, i, j, k)
    @inbounds begin
        v = sd == 1 ? i : sd == 2 ? j : k
        if v > half
            I = CartesianIndex(i + o1, j + o2, k + o3)
            out[I] = pairout[I]
        end
    end
    return nothing
end
