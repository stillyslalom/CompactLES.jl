# Banded compact-scheme kernels (LHS half-bandwidth q ≥ 1), enabling
# pentadiagonal schemes up to 10th order:
#
#   Σ_{s=1}^{q} lhs[s] (g_{i-s} + g_{i+s}) + g_i
#       = a0 f_i + Σ_m coeffs[m] (f_{i+m} ± f_{i-m}),
#
# with '−' for derivatives (RHS scaled by 1/h at plan time) and '+' for
# symmetric filters. Closure rows near closed edges carry a full centered LHS
# vector of length 2q+1 (entries falling outside the domain must be zero) and
# an explicit RHS stencil counted from the edge; high-side rows are mirrored
# automatically (LHS reversed; RHS reversed and negated for derivatives).

struct BandedClosureRow{T}
    lhs::Vector{T}    # length 2q+1, centered on the diagonal
    rhs::Vector{T}    # coefficients on f[1], f[2], ..., from the edge
end

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
Three closure rows are needed at closed edges (the interior RHS reaches ±3):
the C6 third/fourth-order one-sided rows on rows 1–2 and the C6 tridiagonal
interior row on row 3 — the usual boundary cascade, with local order reduction
near walls. Requires halo width n_halo ≥ 3 (default n_halo = 4 is fine).
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
