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
# presets below cover the standard Lele sixth-order interior, a fourth-order
# Padé variant, the Gaitonde–Visbal eighth-order filter, and the explicit
# nine-point Gaussian test filter. Each preset takes the element type as an
# optional trailing argument, defaulting to Float64.

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

"""
    lele_d1_6()

Sixth-order tridiagonal first derivative (Lele 1992): α = 1/3, a = 14/9,
b = 1/9. A closed edge takes two closure rows: a third-order one-sided row 1
and a fourth-order centered Padé row 2.
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
strength (larger → weaker filtering). At a closed edge the first row is left
unfiltered and rows 2–4 apply centered compact filters of order 2, 4 and 6 with
the same αf, the standard reduced-order boundary cascade.
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
    # Consistency: the RHS coefficients of each of these rows sum to 1 + 2αf.
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
