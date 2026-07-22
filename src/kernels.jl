# Compact-scheme kernels.
#
# A CompactScheme describes a tridiagonal-LHS compact operator
#
#   α g_{i-1} + g_i + α g_{i+1} = a0 f_i + Σ_m coeffs[m] (f_{i+m} ± f_{i-m}),
#
# with '−' (antisymmetric) for derivatives — the RHS is additionally scaled by
# 1/h at plan time — and '+' (symmetric, plus the a0 center weight) for
# filters. Non-periodic edges are closed by explicit ClosureRows: row j at the
# low edge has LHS (sub, diag, super) on columns (j−1, j, j+1) and an RHS
# stencil applied to f[1:length(rhs)]. High-edge rows are mirrored
# automatically (LHS sub/super swapped; RHS reversed, negated for derivatives).
#
# Users supply their own schemes by constructing CompactScheme directly; the
# presets below cover the standard Lele sixth-order interior, a fourth-order
# Padé variant, and the Gaitonde–Visbal eighth-order filter.

abstract type AbstractCompactScheme end

struct ClosureRow{T}
    lhs::NTuple{3,T}   # (sub, diag, super); sub is ignored on row 1
    rhs::Vector{T}     # coefficients on f[1], f[2], ..., counted from the edge
end

struct CompactScheme{T} <: AbstractCompactScheme
    name::String
    alpha::T           # interior LHS off-diagonal
    a0::T              # center RHS weight (filters); zero for derivatives
    coeffs::Vector{T}  # coeffs[m] weights (f_{i+m} ± f_{i-m}), m = 1..M
    symmetric::Bool    # true: filter (symmetric); false: derivative (antisymmetric)
    closures::Vector{ClosureRow{T}}
end

nclosure(s::CompactScheme) = length(s.closures)
halfwidth(s::CompactScheme) = length(s.coeffs)

"""
    lele_d1_6()

Sixth-order tridiagonal first derivative (Lele 1992): α = 1/3, a = 14/9,
b = 1/9, with third/fourth-order one-sided closures on the first two rows.
"""
function lele_d1_6(::Type{T}=Float64) where {T}
    CompactScheme{T}("Lele C6 first derivative", T(1//3), zero(T),
        T[7//9, 1//36],   # a/2, b/4  (multiply (f_{i+1}−f_{i-1}), (f_{i+2}−f_{i-2}))
        false,
        [ClosureRow{T}((zero(T), one(T), T(2)), T[-5//2, 2, 1//2]),
         ClosureRow{T}((T(1//4), one(T), T(1//4)), T[-3//4, 0, 3//4])])
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
    compact_filter(alphaf=0.45)

Eighth-order Gaitonde–Visbal compact filter. `alphaf ∈ (−0.5, 0.5)` sets the
strength (larger → weaker filtering). Near closed edges the first row is
left unfiltered and rows 2–4 apply centered compact filters of order 2, 4, 6
with the same αf — the standard reduced-order boundary cascade.
"""
function compact_filter(alphaf::Real=0.45, ::Type{T}=Float64) where {T}
    af = T(alphaf)
    a0 = (93 + 70af) / 128
    a1 = (7 + 18af) / 16
    a2 = (-7 + 14af) / 32
    a3 = (1 - 2af) / 16
    a4 = (-1 + 2af) / 128
    # Boundary closures: row 1 identity; rows 2–4 host centered compact
    # filters of order 2, 4, 6 (same αf), the standard reduced-order cascade.
    # Consistency: RHS coefficients of each row sum to 1 + 2αf.
    b2 = (T((1 + 2af) / 2), T((1 + 2af) / 2))                        # F2: a0, a1
    b4 = (T((5 + 6af) / 8), T((1 + 2af) / 2), T((-1 + 2af) / 8))     # F4
    b6 = (T((11 + 10af) / 16), T((15 + 34af) / 32),
          T((-3 + 6af) / 16), T((1 - 2af) / 32))                     # F6
    ctr(c) = [ [c[m + 1] / 2 for m in length(c)-1:-1:1]; c[1];
               [c[m + 1] / 2 for m in 1:length(c)-1] ]
    cl = [ClosureRow{T}((zero(T), one(T), zero(T)), T[1]),
          ClosureRow{T}((af, one(T), af), ctr(b2)),
          ClosureRow{T}((af, one(T), af), ctr(b4)),
          ClosureRow{T}((af, one(T), af), ctr(b6))]
    CompactScheme{T}("Gaitonde–Visbal C8 filter", af, a0,
                     T[a1/2, a2/2, a3/2, a4/2], true, cl)
end
