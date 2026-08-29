# Compact-scheme kernels.
#
# A CompactScheme describes a tridiagonal-LHS compact operator
#
#   α g_{i-1} + g_i + α g_{i+1} = a0 f_i + Σ_m coeffs[m] (f_{i+m} ± f_{i-m}),
#
# with '−' (antisymmetric) for first derivatives, whose RHS is divided by h at
# plan time, and '+' (symmetric, plus the a0 center weight) for filters.
# Non-periodic edges are closed by explicit ClosureRows: row j at the
# low edge has LHS (sub, diag, super) on columns (j−1, j, j+1) and an RHS
# stencil applied to f[1:length(rhs)]. High-edge rows are mirrored
# automatically (LHS sub/super swapped; RHS reversed, negated for derivatives).
#
# Users supply their own schemes by constructing CompactScheme directly; the
# presets below cover the standard Lele sixth- and eighth-order tridiagonal
# interiors, a fourth-order Padé variant, the Gaitonde–Visbal eighth-order
# filter, and the explicit nine-point Gaussian test filter. Each preset takes
# the element type as an optional trailing argument, defaulting to Float64.

abstract type AbstractCompactScheme end

"""
    ClosureRow(lhs, rhs, first=1)

One low-edge closure row for a tridiagonal [`CompactScheme`](@ref). `lhs` is
the `(subdiagonal, diagonal, superdiagonal)` tuple and `rhs` contains weights
on the points `f[first], f[first+1], ...`, indices counted from the edge node
at 1. High-edge rows are mirrored automatically.

`first = 1` is the closed-boundary case: the stencil reads interior points
only, so nothing beyond the edge is touched and stale physical-edge halos are
never read. A `first <= 0` row reads ghost points beyond the edge and is valid
only where those ghosts carry data — a patch-interface end filled by
`exchange_patch_ghosts!` — which `plan_direction` selects through its
`lo_closures`/`hi_closures` keywords rather than through the scheme itself.
"""
struct ClosureRow{T}
    lhs::NTuple{3,T}   # (sub, diag, super); sub is ignored on row 1
    rhs::Vector{T}     # coefficients on f[first], f[first+1], ...
    first::Int         # index of the first rhs point (1 = the edge node)
end

ClosureRow{T}(lhs::NTuple{3,T}, rhs::Vector{T}) where {T} =
    ClosureRow{T}(lhs, rhs, 1)
ClosureRow(lhs, rhs) = ClosureRow(lhs, rhs, 1)

"""
    CompactScheme(name, alpha, a0, coeffs, symmetric, closures)

Tridiagonal compact operator definition. Antisymmetric schemes represent first
derivatives and have their right-hand side divided by the grid spacing when
planned; symmetric schemes represent filters and include the center coefficient
`a0`.

Use [`lele_d1_6`](@ref), [`pade_d1_4`](@ref), or [`compact_filter`](@ref) unless
defining a custom compact scheme.
"""
struct CompactScheme{T} <: AbstractCompactScheme
    name::String
    alpha::T           # interior LHS off-diagonal
    a0::T              # center RHS weight (filters); zero for derivatives
    coeffs::Vector{T}  # coeffs[m] weights (f_{i+m} ± f_{i-m}), m = 1..M
    symmetric::Bool    # true: filter (symmetric); false: derivative (antisymmetric)
    closures::Vector{ClosureRow{T}}
end

nclosure(scheme::CompactScheme) = length(scheme.closures)
halfwidth(scheme::CompactScheme) = length(scheme.coeffs)

# Closure rows for the tridiagonal derivative presets. Three sets are offered,
# selected by the `closures` keyword of `lele_d1_6` and `lele_d1_8`:
#
#   :cascade3      Lele's one-sided row 1 at α = 2 (third order), the centered
#                  Padé row 2 (fourth), then the C6 interior row (sixth) for a
#                  scheme reaching ±3. This is the reduced-order cascade of
#                  Carpenter, Gottlieb & Abarbanel (1993) and the default.
#   :cascade4      The same with row 1 at α = 3, Lele's fourth-order one-sided
#                  row. Gaitonde & Visbal run it under the compact filter.
#   :brady_livescu Brady & Livescu (Computers & Fluids 2019), scheme T6 or
#                  T8, set 1 of the companion Data in Brief databases
#                  (doi 10.1016/j.dib.2019.104086): every row one order below
#                  the interior, discretely conservative under the quadrature
#                  weights the paper tabulates, and stable on their long-time
#                  Euler tests without a filter. The tables below are
#                  evaluated from the published constraint files in 256-bit
#                  arithmetic and rounded once.
#
# The Brady–Livescu rows are far from diagonally dominant (row 1 of T6 has a
# superdiagonal of 6.74, row 2 of T8 one of 42.1), so a closed line's
# unpivoted Thomas factorization carries a condition number near 1e3–4e3
# against 16 for the cascades. That is three lost digits in Float64 and is
# the reason the cascade stays the default.

function cascade_closures(::Type{T}, nrows::Int, first_order::Int) where {T}
    row1 = first_order == 3 ?
        ClosureRow{T}((zero(T), one(T), T(2)), T[-5//2, 2, 1//2]) :
        ClosureRow{T}((zero(T), one(T), T(3)), T[-17//6, 3//2, 3//2, -1//6])
    rows = [row1,
            ClosureRow{T}((T(1//4), one(T), T(1//4)), T[-3//4, 0, 3//4]),
            ClosureRow{T}((T(1//3), one(T), T(1//3)),
                          T[-1//36, -7//9, 0, 7//9, 1//36])]
    rows[1:nrows]
end

# Brady–Livescu T6 set 1 (Table A.10 of the paper), four fifth-order rows.
const BRADY_LIVESCU_T6 = (
    ((0.0, 1.0, 6.736832494852786),
     [-3.6306998323038906, -2.298235202757185, 8.473664989705572,
      -3.4034991615194525, 0.9956108316175953, -0.1368416247426393]),
    ((0.4885251620537965, 1.0, 2.7185849538712983),
     [-1.1795365389959371, 0.0, -1.3488207948927484,
      3.3470021607172864, -0.9569693577017367, 0.1383245308731359]),
    ((-0.3891997445794, 1.0, -1.1117328224921332),
     [0.1648977096656178, -0.35630014899535, 0.0,
      1.0186221370820223, -0.9355996594392, 0.10837996168691]),
    ((-0.5719411698333021, 1.0, -0.11930391824998438),
     [-0.06789558773749765, 0.5757385576666458, -0.9286568616388837,
      0.0, 0.5137393810208425, -0.09292548931110695]))

# Brady–Livescu T8 set 1 (Table 5 of the paper), six seventh-order rows.
const BRADY_LIVESCU_T8 = (
    ((0.0, 1.0, 3.210113927329531),
     [-3.0514448467613615, 2.34533480537218, -0.8696582180114064,
      3.6413818483428386, -3.399810121117448, 1.7924145545028516,
      -0.5246438812007604, 0.0664258588731064]),
    ((2.2111047304323885, 1.0, 42.08319933908016),
     [-4.873954900119213, 0.0, -53.18177248515288, 93.43488742017814,
      -52.749832507183534, 22.564372980842755, -5.886555463761134,
      0.6928549551958666]),
    ((1.557633196122124, 1.0, 6.482610425055064),
     [-0.2604486511132088, -1.9436404252049067, 0.0, -3.8480689299024093,
      8.245332418591937, -2.7796746912747787, 0.6603673342280957,
      -0.0738670553247296]),
    ((-1.329341564172446, 1.0, -2.4655692736207433),
     [-0.05878600824425403, 0.7074851396321983, -0.29835322348447363, 0.0,
      1.4913923184051858, -2.222455418896595, 0.42400205770977817,
      -0.0432848651218399]),
    ((-1.2137038102472755, 1.0, 1.6607695424897684),
     [0.05619441751764611, -0.47611421984355806, 1.967293657743865,
      -2.297921866880886, 0.0, -0.7202751534930104, 1.8653374792388604,
      -0.4478236435362061, 0.0533093292532889]),
    ((-0.1776191023391514, 1.0, 0.22329248864693368),
     [-0.00535309375150073, 0.0495326225233603, -0.2157074945553578,
      0.6319138819385538, -1.1442347131478185, 0.0, 0.678311555099823,
      0.005414306071706973, 0.00012293582123288293]))

function derivative_closures(::Type{T}, closures::Symbol, nrows::Int, table) where {T}
    closures === :cascade3 && return cascade_closures(T, nrows, 3)
    closures === :cascade4 && return cascade_closures(T, nrows, 4)
    closures === :brady_livescu && return [
        ClosureRow{T}(T.(lhs), T.(rhs)) for (lhs, rhs) in table]
    error("unknown closure set $(repr(closures)); " *
          "use :cascade3, :cascade4 or :brady_livescu")
end

"""
    lele_d1_6(T=Float64; closures=:cascade3)

Sixth-order tridiagonal first derivative (Lele 1992): α = 1/3, a = 14/9,
b = 1/9. `closures` selects the rows applied at a closed edge:

- `:cascade3` (default): a third-order one-sided row 1 and a fourth-order
  centered Padé row 2, the usual reduced-order cascade.
- `:cascade4`: Lele's fourth-order one-sided row 1 (α = 3) over the same
  Padé row 2.
- `:brady_livescu`: the four fifth-order rows of Brady & Livescu (2019),
  scheme T6, conservative and stable without a filter on their tests but
  with a closed-line condition number near 1e3 (see the source comment).
  Under the default filter wall cascade these rows grow a wall mode
  wherever the artificial bulk viscosity is active at a slip wall, which
  ends every wall-bounded shock case; with
  `compact_filter(closures = :onesided)` they survive a captured shock at a
  wall but not a singular start there. In Float32 the closed line's
  conditioning floors the wall error near 1e-3, above the default
  cascade's, from N = 48 up. The runs are in `reference/CALIBRATION.md`.
"""
function lele_d1_6(::Type{T}=Float64; closures::Symbol=:cascade3) where {T}
    CompactScheme{T}("Lele C6 first derivative", T(1//3), zero(T),
        T[7//9, 1//36],   # a/2, b/4  (multiply (f_{i+1}−f_{i-1}), (f_{i+2}−f_{i-2}))
        false,
        derivative_closures(T, closures, 2, BRADY_LIVESCU_T6))
end

"""
    lele_d1_8(T=Float64; closures=:cascade3)

Eighth-order tridiagonal first derivative (Lele 1992, eq. 2.1 with a
seven-point right-hand side): α = 3/8, a = 25/16, b = 1/5, c = −1/80
(consistency: a + b + c = 1 + 2α). It keeps the tridiagonal line solve of
[`lele_d1_6`](@ref), and with it the multi-patch, device and decomposed paths
that the pentadiagonal [`lele_d1_10`](@ref) does not have, at two more
multiply-adds per point. The interior reaches ±3, so a closed edge takes three
rows under `:cascade3`/`:cascade4` (the C6 cascade plus the C6 interior row)
and the six seventh-order rows of Brady & Livescu's scheme T8 under
`:brady_livescu`. Requires `n_halo ≥ 3`. The T8 rows carry the slip-wall and
Float32 restrictions of the T6 ones (see `lele_d1_6`), fail under the default
filter wall cascade even on smooth data, and need 13 points along a dimension
closed at both ends.
"""
function lele_d1_8(::Type{T}=Float64; closures::Symbol=:cascade3) where {T}
    CompactScheme{T}("Lele C8 first derivative", T(3//8), zero(T),
        T[25//32, 1//20, -1//480],   # a/2, b/4, c/6
        false,
        derivative_closures(T, closures, 3, BRADY_LIVESCU_T8))
end

"""
    pade_d1_4()

Fourth-order Padé first derivative: α = 1/4, a = 3/2, with a third-order
one-sided closure on the first row (the interior stencil is valid from row 2).
"""
function pade_d1_4(::Type{T}=Float64) where {T}
    CompactScheme{T}("Padé C4 first derivative", T(1//4), zero(T),
        T[3//4], false,
        [ClosureRow{T}((zero(T), one(T), T(2)), T[-5//2, 2, 1//2])])
end

"""
    onesided_filter_row(af, i, M) -> Vector

Right-hand side of the one-sided compact filter row at point `i` (counted from
the edge node at 1) of Gaitonde & Visbal (2000), over the points `1:2M+1` with
the interior left-hand side `(af, 1, af)`. The `2M + 1` weights are fixed by
exactness on polynomials of degree `0:2M-1` and a zero response at the Nyquist
wavenumber, which is the interior filter's own construction: at `i = M + 1`
the system returns the centered stencil to round-off, and at `i = 2` it
reproduces the published `(1 + 254αf)/256, (31 + 2αf)/32, ...` row. Solved
here rather than tabulated so the rows follow `af` exactly.
"""
function onesided_filter_row(af::T, i::Int, M::Int) where {T}
    K = 2M + 1
    V = zeros(T, K, K); rhs = zeros(T, K)
    for p in 0:2M-1
        for n in 1:K
            V[p + 1, n] = T(n - i)^p
        end
        rhs[p + 1] = af * T(-1)^p + (p == 0 ? one(T) : zero(T)) + af
    end
    for n in 1:K
        V[K, n] = T(-1)^n
    end
    return V \ rhs
end

"""
    compact_filter(alphaf=0.45; closures=:cascade)

Eighth-order Gaitonde–Visbal compact filter. `alphaf ∈ (−0.5, 0.5)` sets the
strength (larger → weaker filtering). At a closed edge the first row is always
left unfiltered; `closures` selects rows 2–4:

- `:cascade` (default): centered compact filters of order 2, 4 and 6 with the
  same αf, the standard reduced-order boundary cascade. One filter pass of a
  smooth field is then second order in the maximum norm along the whole line,
  not only at the wall, because the compact solve carries the row-2 error
  inward.
- `:onesided`: the one-sided eighth-order rows of Gaitonde & Visbal (2000),
  derived at construction by `onesided_filter_row`. One pass is
  eighth order everywhere, and under repeated application the closed
  operator amplifies less than the cascade does (‖F¹⁰⁰‖₂ 1.05 against 1.14
  at αf = 0.45, N = 64). Its rows 2 and 3 do exceed unit gain at some
  wavenumbers taken alone (1.10 and 1.03 at αf = 0.45, worse at smaller αf),
  which the paper also notes; the measurements are in
  `reference/CALIBRATION.md`.
"""
function compact_filter(alphaf::Real=0.45, ::Type{T}=Float64;
                        closures::Symbol=:cascade) where {T}
    af = T(alphaf)
    a0 = (93 + 70af) / 128
    a1 = (7 + 18af) / 16
    a2 = (-7 + 14af) / 32
    a3 = (1 - 2af) / 16
    a4 = (-1 + 2af) / 128
    # Boundary closures: row 1 identity; rows 2–4 host centered compact
    # filters of order 2, 4, 6 (same αf), the standard reduced-order cascade.
    # Consistency: the RHS coefficients of each of these rows sum to 1 + 2αf.
    b2 = (T((1 + 2af) / 2), T((1 + 2af) / 2))                        # F2: a0, a1
    b4 = (T((5 + 6af) / 8), T((1 + 2af) / 2), T((-1 + 2af) / 8))     # F4
    b6 = (T((11 + 10af) / 16), T((15 + 34af) / 32),
          T((-3 + 6af) / 16), T((1 - 2af) / 32))                     # F6
    ctr(c) = [ [c[m + 1] / 2 for m in length(c)-1:-1:1]; c[1];
               [c[m + 1] / 2 for m in 1:length(c)-1] ]
    row1 = ClosureRow{T}((zero(T), one(T), zero(T)), T[1])
    cl = if closures === :cascade
        [row1,
         ClosureRow{T}((af, one(T), af), ctr(b2)),
         ClosureRow{T}((af, one(T), af), ctr(b4)),
         ClosureRow{T}((af, one(T), af), ctr(b6))]
    elseif closures === :onesided
        [row1; [ClosureRow{T}((af, one(T), af), onesided_filter_row(af, i, 4))
                for i in 2:4]]
    else
        error("unknown filter closure set $(repr(closures)); " *
              "use :cascade or :onesided")
    end
    CompactScheme{T}("Gaitonde–Visbal C8 filter", af, a0,
                     T[a1/2, a2/2, a3/2, a4/2], true, cl)
end

"""
    gaussian_filter()

Explicit nine-point Gaussian test filter, the smoother Cook's artificial
properties assume and the one Miranda applies as `gbar`
(`pyranda/parcop/stencils.f90`, `cgfs4`). The left-hand side is the identity,
so this is a `CompactScheme` only in the sense that it reuses the same fill,
fold and closure machinery; `plan_direction` detects the zero α and
skips the line solve and its interface reduction entirely.

The weights sum to exactly 1 over the common denominator 103680, so constants
are reproduced without relying on cancellation. Each of the four closure rows
is the interior stencil with its overhanging weights folded back onto the
half-offset mirror (ghost j ↔ interior j), which preserves that unit sum at a
closed edge; the same construction is what `fold_fill!` performs at a fold, so
the two edge treatments agree.

Contrast [`compact_filter`](@ref), which is a dealiasing filter for the
conserved state rather than a test filter: at αf = 0.45 it retains 99% of the
amplitude at four points per wavelength where this filter retains 19%. The
measurement is in `reference/CALIBRATION.md`.
"""
function gaussian_filter(::Type{T}=Float64) where {T}
    a = T(3565//10368); b = T(3091//12960); c = T(1997//25920)
    d = T(149//12960);  e = T(107//103680)
    lhs = (zero(T), one(T), zero(T))
    cl = [ClosureRow{T}(lhs, T[a+b, b+c, c+d, d+e, e]),
          ClosureRow{T}(lhs, T[b+c, a+d, b+e, c, d, e]),
          ClosureRow{T}(lhs, T[c+d, b+e, a, b, c, d, e]),
          ClosureRow{T}(lhs, T[d+e, c, b, a, b, c, d, e])]
    CompactScheme{T}("explicit 9-point Gaussian", zero(T), a, T[b, c, d, e], true, cl)
end

# --- Patch-interface closures ------------------------------------------------
#
# At a patch interface the ghost layers carry the abutting patch's data, but the
# ghost UNKNOWNS belong to that patch's solve, so a row's left-hand side must
# couple interior unknowns only (Miranda tabulates the extended-data transfer
# closures identically to the one-sided ones for this reason — "same as
# one-sided to maintain invertibility"). The right-hand side is free to read the
# copied ghost data. The rows below exploit that: only the edge row's LHS
# couples a ghost unknown in the interior scheme, so a single replacement row
# per end suffices, and every following row keeps the full interior stencil,
# its RHS reaching into ghosts the exchange has filled.

# Central explicit first-derivative weights of order 2m on offsets -m:m,
# exact rationals, undivided (plan_direction applies the 1/h scale).
function _central_d1_weights(::Type{T}, m::Int) where {T}
    m == 1 && return T[-1//2, 0, 1//2]
    m == 2 && return T[1//12, -8//12, 0, 8//12, -1//12]
    m == 3 && return T[-1//60, 9//60, -45//60, 0, 45//60, -9//60, 1//60]
    m == 4 && return T[3//840, -32//840, 168//840, -672//840, 0,
                       672//840, -168//840, 32//840, -3//840]
    error("central first-derivative weights tabulated for half-widths 1-4, got $m")
end

"""
    interface_closures(scheme) -> Vector{ClosureRow}

Closure rows for a patch-interface end of `scheme`, per the extended-data
convention above. For a derivative the single replacement row is an explicit
central difference of order `2(M+1)` (the interior formal order) whose stencil
reads `M+1` ghost points; for a filter it is the identity, leaving the shared
interface-plane node to the post-stage averaging; and a scheme that is explicit
throughout (`gaussian_filter`) needs no closure at all — the interior stencil
simply reads the exchanged ghosts. `plan_direction` verifies the ghost reach
against the halo width.
"""
function interface_closures(scheme::CompactScheme{T}) where {T}
    lhs = (zero(T), one(T), zero(T))
    if !scheme.symmetric
        m = halfwidth(scheme) + 1
        return [ClosureRow{T}(lhs, _central_d1_weights(T, m), 1 - m)]
    end
    # Explicit symmetric schemes close themselves: identity LHS everywhere and
    # an RHS that reads ghosts directly.
    iszero(scheme.alpha) && return ClosureRow{T}[]
    return [ClosureRow{T}(lhs, T[1], 1)]
end
