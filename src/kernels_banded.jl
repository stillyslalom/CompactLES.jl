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
    BandedClosureRow(lhs, rhs)

One low-edge closure row for a [`BandedCompactScheme`](@ref). `lhs` is the full
centered band, of length `2q + 1` with the diagonal at index `q + 1` and entries
reaching outside the domain set to zero; `rhs` holds weights on `f[1]`, `f[2]`,
... counted inward from the edge. High-edge rows are mirrored automatically.
"""
struct BandedClosureRow{T}
    lhs::Vector{T}    # length 2q+1, centered on the diagonal
    rhs::Vector{T}    # coefficients on f[1], f[2], ..., from the edge
end

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
(the default n_halo = 4 is sufficient).
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
detector of [`ArtParams`](@ref) under `detector = :d8`. It is Miranda's `ring`
operator (`pyranda/parcop/stencils.f90`, `c10d8`), whose interior rows are

    1.5 g_{i-2} + 14 g_{i-1} + 29 g_i + 14 g_{i+1} + 1.5 g_{i+2} = 60 δ⁸f_i,

with δ⁸ the undivided eighth difference (1, −8, 28, −56, 70, −56, 28, −8, 1).
The operator is symmetric, since an even derivative preserves parity, so it is
planned like a filter rather than a derivative: no `1/h` scaling, and mirrored
rather than negated at a high edge.

# Normalization

The coefficients below are the reference ones divided by 29 (making the
left-hand-side diagonal one, as `BandedCompactScheme` requires) and by a
further 240, which sets the response to a grid-to-grid oscillation to 16, the
value the undivided δ⁴ detector it replaces produces there. The two detectors
therefore agree at the wavelength both are built to catch, and the `C_mu`,
`C_beta`, `C_kappa` and `C_D` calibrations carry over as starting points
instead of being off by two orders of magnitude. The two responses differ below
the Nyquist; `reference/CALIBRATION.md` records them at 8, 4 and 2.7 points per
wavelength.

# Closure rows

Four rows are needed at a closed edge, since the interior right-hand side
reaches ±4. They are the interior stencil with its overhanging weights folded
back onto the half-offset mirror (ghost j ↔ interior j), which is the
reference's own `-1:2` boundary variant 0 and the same construction
[`gaussian_filter`](@ref) uses. Every row's weights sum to zero, so a constant
is annihilated exactly rather than by cancellation. Minimum local extent is 9
points per rank, matching [`compact_filter`](@ref).
"""
function compact_d8(::Type{T}=Float64) where {T}
    # Reference weights: ζ = 29, α = 14, β = 3/2; a = 4200, b = −3360,
    # c = 1680, d = −480, e = 60. Divided here by ζ (left) and by 240ζ = 6960
    # (right). Boundary rows carry the reference's own folded combinations,
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
