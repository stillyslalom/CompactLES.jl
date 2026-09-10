# Banded compact-scheme kernels (LHS half-bandwidth q ≥ 1), enabling
# pentadiagonal schemes up to 10th order:
#
#   Σ_{s=1}^{q} lhs[s] (g_{i-s} + g_{i+s}) + g_i
#       = a0 f_i + Σ_m coeffs[m] (f_{i+m} ± f_{i-m}),
#
# with '−' for first derivatives (RHS divided by h at plan time) and '+' for
# symmetric operators: filters, and the undivided even derivative that serves as
# the ringing detector.
#
# Closure rows near closed edges carry a full centered LHS
# vector of length 2q+1 (entries falling outside the domain must be zero) and
# an explicit RHS stencil counted from the edge; high-side rows are mirrored
# automatically (LHS reversed; RHS reversed and negated for derivatives).

"""
    BandedClosureRow(lhs, rhs, first=1)

One low-edge closure row for a [`BandedCompactScheme`](@ref). `lhs` is the full
centered band, of length `2q + 1` with the diagonal at index `q + 1` and entries
reaching outside the domain set to zero; `rhs` holds weights on `f[first]`,
`f[first+1]`, ..., where `first = 1` is the edge node and `first ≤ 0` a ghost
layer the row reads across a patch interface. High-edge rows are mirrored
automatically.
"""
struct BandedClosureRow{T}
    lhs::Vector{T}    # length 2q+1, centered on the diagonal
    rhs::Vector{T}    # coefficients on f[first], f[first+1], ...
    first::Int        # index of the first rhs point (1 = the edge node)
end

BandedClosureRow{T}(lhs::Vector{T}, rhs::Vector{T}) where {T} =
    BandedClosureRow{T}(lhs, rhs, 1)
BandedClosureRow(lhs, rhs) = BandedClosureRow(lhs, rhs, 1)

"""
    BandedCompactScheme(name, q, lhs, a0, coeffs, symmetric, closures)

Compact operator with left-hand-side half-bandwidth `q`. This is the
pentadiagonal-capable counterpart of [`CompactScheme`](@ref); the built-in
[`lele_d1_10`](@ref) is the usual entry point.
"""
struct BandedCompactScheme{T} <: AbstractCompactScheme
    name::String
    q::Int
    lhs::Vector{T}    # off-diagonal LHS coefficients, lhs[s] at distance s
    a0::T
    coeffs::Vector{T} # RHS half-stencil weights, m = 1..M
    symmetric::Bool
    closures::Vector{BandedClosureRow{T}}
end

nclosure(scheme::BandedCompactScheme) = length(scheme.closures)
halfwidth(scheme::BandedCompactScheme) = length(scheme.coeffs)

"""
    lele_d1_10()

Tenth-order pentadiagonal first derivative (Lele 1992): β = 1/20, α = 1/2,
a = 17/12, b = 101/150, c = 1/100 (consistency: a + b + c = 1 + 2α + 2β).
Three closure rows are needed at a closed edge, since the interior RHS reaches
±3: the C6 third-order one-sided row 1, the C6 fourth-order centered Padé row 2,
and the C6 tridiagonal interior row on row 3. This is the usual boundary
cascade, with local order reduction near walls. Requires halo width n_halo ≥ 3
(the default n_halo = 4 is sufficient); the two [`interface_closures`](@ref)
rows of a patch interface read four, so the default serves there too.
"""
function lele_d1_10(::Type{T}=Float64) where {T}
    BandedCompactScheme{T}("Lele C10 first derivative", 2,
        T[1//2, 1//20],                 # α, β
        zero(T),
        T[17//24, 101//600, 1//600],    # a/2, b/4, c/6
        false,
        [BandedClosureRow{T}(T[0, 0, 1, 2, 0],       T[-5//2, 2, 1//2]),
         BandedClosureRow{T}(T[0, 1//4, 1, 1//4, 0], T[-3//4, 0, 3//4]),
         BandedClosureRow{T}(T[0, 1//3, 1, 1//3, 0],
                             T[-1//36, -7//9, 0, 7//9, 1//36])])
end

"""
    compact_d8()

Pentadiagonal compact eighth derivative, undivided, used as the ringing
detector of [`ArtParams`](@ref) under `detector = :d8`. It is Pyranda's
[public `ring` operator](https://github.com/LLNL/pyranda)
(`pyranda/parcop/stencils.f90`, `c10d8`), whose interior rows are

    1.5 g_{i-2} + 14 g_{i-1} + 29 g_i + 14 g_{i+1} + 1.5 g_{i+2} = 60 δ⁸f_i,

with δ⁸ the undivided eighth difference (1, −8, 28, −56, 70, −56, 28, −8, 1).
The operator is symmetric, since an even derivative preserves parity, so it is
planned like a filter, not a derivative: no `1/h` scaling, and the high-edge
rows are mirrored, not negated.

# Normalization

The coefficients below are the reference ones divided by 29 (making the
left-hand-side diagonal one, as `BandedCompactScheme` requires) and by a
further 240, which sets the response to a grid-to-grid oscillation to 16, the
value the undivided δ⁴ detector it replaces produces there. The two detectors
therefore agree at the wavelength both are built to catch, and the `C_mu`,
`C_beta`, `C_kappa` and `C_D` calibrations carry over as starting points and are
not off by two orders of magnitude. Below the Nyquist the two responses
diverge: the ratio of the δ⁴ response to this one is 569 at eight points per
wavelength, 26 at four and 3.2 at 2.7.

# Closure rows

Four rows are needed at a closed edge, since the interior right-hand side
reaches ±4. They are the interior stencil with its overhanging weights folded
back onto the half-offset mirror (ghost j ↔ interior j), which is the
Pyranda's own `-1:2` boundary variant 0 and the same construction
[`gaussian_filter`](@ref) uses. Every row's weights sum to zero, so a constant
is annihilated exactly, without cancellation. Minimum local extent is 9
points per rank, matching [`compact_filter`](@ref).
"""
function compact_d8(::Type{T}=Float64) where {T}
    # Reference weights: ζ = 29, α = 14, β = 3/2; a = 4200, b = −3360,
    # c = 1680, d = −480, e = 60. Divided here by ζ (left) and by 240ζ = 6960
    # (right). Boundary rows carry Pyranda's own folded combinations,
    # e.g. row 1 has ζ + α on its diagonal and a + b on f₁.
    BandedCompactScheme{T}("compact eighth-derivative ring detector", 2,
        T[14//29, 3//58],                            # α/ζ, β/ζ
        T(35//58),                                   # a/240ζ
        T[-14//29, 7//29, -2//29, 1//116],           # b, c, d, e over 240ζ
        true,
        [BandedClosureRow{T}(T[0, 0, 43//29, 31//58, 3//58],
                             T[7//58, -7//29, 5//29, -7//116, 1//116]),
         BandedClosureRow{T}(T[0, 31//58, 1, 14//29, 3//58],
                             T[-7//29, 31//58, -55//116, 7//29, -2//29, 1//116]),
         BandedClosureRow{T}(T[3//58, 14//29, 1, 14//29, 3//58],
                             T[5//29, -55//116, 35//58, -14//29, 7//29, -2//29,
                               1//116]),
         BandedClosureRow{T}(T[3//58, 14//29, 1, 14//29, 3//58],
                             T[-7//116, 7//29, -14//29, 35//58, -14//29, 7//29,
                               -2//29, 1//116])])
end

"""
    pyranda_filter()

Pyranda's compact eighth-order dealiasing filter, the `c8ff8` stencil of
`pyranda/parcop/stencils.f90` ([LLNL/pyranda](https://github.com/LLNL/pyranda))
transcribed verbatim as a symmetric pentadiagonal [`BandedCompactScheme`](@ref):

    β F_{i±2} + α F_{i±1} + F_i = a f_i + b f_{i±1} + c f_{i±2} + d f_{i±3} + e f_{i±4}

with β = 0.16688, α = 0.66624, a = 0.99965, b = 0.66652, c = 0.16674,
d = 4·10⁻⁵ and e = −5·10⁻⁶, each `±` term standing for the sum of the two
neighbours. Pyranda names it the 9/10 filter: the transfer function
T(k) = (a + 2b cos k + 2c cos 2k + 2d cos 3k + 2e cos 4k) /
(1 + 2α cos k + 2β cos 2k) integrates to 9/10 of π over 0 ≤ k ≤ π. It is one
at k = 0 (the right-hand side sums to 1 + 2α + 2β) and zero at the Nyquist
wavenumber (a − 2b + 2c − 2d + 2e = 0), and it is still 0.988 at
k = 0.75π, where [`compact_filter`](@ref) at αf = 0.45 has fallen to 0.854.

At a closed edge the rows are Pyranda's "telescoping" set: row 1 the
identity, row 2 the tridiagonal filter α₂ = 0.4997 with right-hand side
(0.49985, 0.9997, 0.49985), and rows 3 and 4 the interior left-hand side
over right-hand sides (0.1668, 0.66656, 0.99952, ...) and
(4·10⁻⁵, 0.16672, 0.66652, 0.99968, ...). Each row's two sides sum equally,
so a constant passes exactly. Pyranda applies these rows where a boundary is
neither periodic nor symmetric; at a symmetric boundary it folds the interior
stencil across the mirror instead, which is not transcribed.

Pass it as `Numerics.filt` for a like-for-like comparison with Pyranda,
beside [`lele_d1_10`](@ref) for the derivative and `detector = :d8` for the
sensor.
"""
function pyranda_filter(::Type{T}=Float64) where {T}
    β, α = T(1.6688e-1), T(6.6624e-1)
    a, b, c, d, e = T(9.9965e-1), T(6.6652e-1), T(1.6674e-1), T(4.0e-5), T(-5.0e-6)
    interior = T[β, α, 1, α, β]
    BandedCompactScheme{T}("Pyranda c8ff8 filter", 2, T[α, β], a, T[b, c, d, e], true,
        [BandedClosureRow{T}(T[0, 0, 1, 0, 0], T[1]),
         BandedClosureRow{T}(T[0, 4.997e-1, 1, 4.997e-1, 0],
                             T[4.9985e-1, 9.997e-1, 4.9985e-1]),
         BandedClosureRow{T}(interior,
                             T[1.668e-1, 6.6656e-1, 9.9952e-1, 6.6656e-1, 1.668e-1]),
         BandedClosureRow{T}(interior,
                             T[4.0e-5, 1.6672e-1, 6.6652e-1, 9.9968e-1, 6.6652e-1,
                               1.6672e-1, 4.0e-5])])
end

"""
    interface_closures(scheme::BandedCompactScheme) -> Vector{BandedClosureRow}

The patch-interface closure rows of a banded scheme, per the extended-data
convention of the [`CompactScheme`](@ref) method: the first `q` rows from the
edge are the ones whose left-hand side would couple a ghost unknown, so each
is replaced by a row whose left-hand side couples interior unknowns only.
For a derivative the rows are compact and Taylor-matched to order
`2(M + q)`, the interior formal order, on a right-hand side of half-width
`M + q − 1` centered on the row, so row 1 reads `M + q − 1` ghost layers
(four for [`lele_d1_10`](@ref), the default halo): row 1 couples the
`q` unknowns inward of it, and row `j > 1` its `j − 1` neighbors on each
side plus the `q − j + 1` beyond on the interior side, symmetrically
where it can. An explicit central row of the same order would read one
more layer. `plan_direction` checks the reach against the halo width. For
a symmetric operator each row is the identity on its own node. Rows
`q + 1` onward keep the interior stencil, their right-hand sides reading
the exchanged ghosts.
"""
function interface_closures(scheme::BandedCompactScheme{T}) where {T}
    q = scheme.q
    if scheme.symmetric
        lhs = zeros(T, 2q + 1)
        lhs[q+1] = one(T)
        return [BandedClosureRow{T}(copy(lhs), T[1], j) for j in 1:q]
    end
    m = halfwidth(scheme) + q - 1
    rows = BandedClosureRow{T}[]
    for j in 1:q
        # Free left-hand-side coefficients: symmetric pairs at ±s for
        # s < j (both sides interior), single entries at +s for s ≥ j.
        pairs = collect(1:j-1)
        singles = collect(j:q)
        c_pairs, c_singles, w = _taylor_d1_row(pairs, singles, m)
        lhs = zeros(T, 2q + 1)
        lhs[q+1] = one(T)
        for (s, c) in zip(pairs, c_pairs)
            lhs[q+1+s] = T(c)
            lhs[q+1-s] = T(c)
        end
        for (s, c) in zip(singles, c_singles)
            lhs[q+1+s] = T(c)
        end
        push!(rows, BandedClosureRow{T}(lhs, T.(w), j - m))
    end
    return rows
end

# One compact first-derivative row by Taylor matching, in exact rationals:
# g'_0 + Σ_{s∈pairs} c_s (g'_{-s} + g'_{s}) + Σ_{s∈singles} c_s g'_{s}
#     = (1/h) Σ_{k=-m}^{m} w_k f_k,
# matched through order `length(pairs) + length(singles) + 2m`, the count of
# unknowns less one. The left-hand side of a node at offset s contributes
# s^(n−1)/(n−1)! to the coefficient of f^{(n)} h^{n−1}, the diagonal only at
# n = 1; the right-hand side Σ_k w_k k^n / n!.
function _taylor_d1_row(pairs::Vector{Int}, singles::Vector{Int}, m::Int)
    np_, ns = length(pairs), length(singles)
    nunk = np_ + ns + 2m + 1
    p = nunk - 1
    A = zeros(Rational{BigInt}, p + 1, nunk)
    b = zeros(Rational{BigInt}, p + 1)
    for n in 0:p
        for (i, k) in enumerate(-m:m)
            A[n+1, np_+ns+i] = big(k)^n // factorial(big(n))
        end
        n == 0 && continue
        for (i, s) in enumerate(pairs)
            A[n+1, i] = -(big(-s)^(n - 1) + big(s)^(n - 1)) // factorial(big(n - 1))
        end
        for (i, s) in enumerate(singles)
            A[n+1, np_+i] = -(big(s)^(n - 1)) // factorial(big(n - 1))
        end
        b[n+1] = n == 1 ? 1 // 1 : 0 // 1
    end
    x = A \ b
    return x[1:np_], x[np_+1:np_+ns], x[np_+ns+1:end]
end
