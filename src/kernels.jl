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
only where those ghosts carry data (a patch-interface end filled by
`exchange_patch_ghosts!`); `plan_direction` selects such rows through its
`lo_closures`/`hi_closures` keywords, not through the scheme itself.
"""
struct ClosureRow{T}
    lhs::NTuple{3,T}   # (sub, diag, super); sub is ignored on row 1
    rhs::Vector{T}     # coefficients on f[first], f[first+1], ...
    first::Int         # index of the first rhs point (1 = the edge node)
end

ClosureRow{T}(lhs::NTuple{3,T}, rhs::Vector{T}) where {T} =
    ClosureRow{T}(lhs, rhs, 1)
ClosureRow(lhs, rhs) = ClosureRow(lhs, rhs, 1)
Base.:(==)(a::ClosureRow, b::ClosureRow) =
    a.lhs == b.lhs && a.rhs == b.rhs && a.first == b.first

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

# Closure rows for the derivative presets. Four sets are offered, selected by
# the `closures` keyword of `lele_d1_6` and `lele_d1_8`; `lele_d1_10` takes
# the first two through the same tables, widened by a zero outer band.
#
#   :neutral3      The default of all three presets. The explicit third-order
#                  one-sided row 1 on four points and a fourth-order compact
#                  row 2 on five points with left-hand side (3/5, 1, 3/10); a
#                  three-row edge adds the C6 interior row as its sixth-order
#                  row 3. See the comment on NEUTRAL3_ROWS.
#   :cascade3      Lele's one-sided row 1 at α = 2 (third order), the centered
#                  Padé row 2 (fourth), then the C6 interior row (sixth) for a
#                  scheme reaching ±3. This is the reduced-order cascade of
#                  Carpenter, Gottlieb & Abarbanel (1993).
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

# The neutral rows in exact rationals. Row 1 is the explicit third-order
# one-sided difference (left-hand side (0, 1, 0)); row 2 is the fourth-order
# five-point row that the left-hand side (3/5, 1, 3/10) determines uniquely.
# Both belong to the families
#   g_1 + a g_2 = Σ_{k=1}^{4} w_k f_k         (third order, free a)
#   b g_1 + g_2 + c g_3 = Σ_{k=1}^{5} w_k f_k  (fourth order, free b, c)
# at (a, b, c) = (0, 3/5, 3/10). With the wall's normal velocity injected,
# the Euler equations linearized about a uniform state between slip walls
# have a purely imaginary spectrum on a two-parameter subset of (a, b, c); a
# member can be neutral at one line length and not at another, so this one
# was selected by a sweep over every N from 12 to 1200 (the neutral subset
# and the sweep are in reference/CALIBRATION_APPENDIX.md). No energy norm
# has been identified for it; the property is measured, not proved.
#
# A C8 edge takes three rows, and the same search over the three-row family
# that widens each cascade row by one point returns these two rows again:
# its neutral members also carry an explicit third-order row 1, a sixth-order
# row 3 is possible only along a one-parameter line through the C6 interior
# row, and both members that survive the sweep keep that interior row. The
# C8 set is therefore rows 1 and 2 here over the C6 interior row, which
# `cascade_closures(T, 3, 3)[3]` already supplies. The pentadiagonal C10
# edge, searched over the same three rows with their outer band entries
# free, returns the same set: see `_banded_closure_rows`.
const NEUTRAL3_ROWS = (
    ((0//1, 1//1, 0//1), [-11//6, 3//1, -3//2, 1//3]),
    ((3//5, 1//1, 3//10), [-59//40, 41//30, -3//10, 1//2, -11//120]))

function neutral_closures(::Type{T}, nrows::Int) where {T}
    2 <= nrows <= 3 ||
        error("the :neutral3 closure rows close a two-row (C6) or three-row " *
              "(C8, C10) edge; a $(nrows)-row edge takes :cascade3, :cascade4 " *
              "or :brady_livescu")
    rows = [ClosureRow{T}(T.(lhs), T.(rhs)) for (lhs, rhs) in NEUTRAL3_ROWS]
    nrows == 3 && push!(rows, cascade_closures(T, 3, 3)[3])
    rows
end

function derivative_closures(::Type{T}, closures::Symbol, nrows::Int, table) where {T}
    closures === :neutral3 && return neutral_closures(T, nrows)
    closures === :cascade3 && return cascade_closures(T, nrows, 3)
    closures === :cascade4 && return cascade_closures(T, nrows, 4)
    closures === :brady_livescu && return [
        ClosureRow{T}(T.(lhs), T.(rhs)) for (lhs, rhs) in table]
    error("unknown closure set $(repr(closures)); " *
          "use :neutral3, :cascade3, :cascade4 or :brady_livescu")
end

"""
    lele_d1_6(T=Float64; closures=:neutral3)

Sixth-order tridiagonal first derivative (Lele 1992): α = 1/3, a = 14/9,
b = 1/9. `closures` selects the rows applied at a closed edge:

- `:neutral3` (default): an explicit third-order row 1 on four points and a
  fourth-order compact row 2 on five points, with coefficients chosen so
  that the Euler step linearized about a uniform state between slip walls
  is neutral. A long inviscid run between slip walls or symmetry planes
  holds its round-off seed. Same orders as `:cascade3` with about 2.5 times
  its wall error constant and a better-conditioned closed line.
- `:cascade3`: the reduced-order cascade of Carpenter, Gottlieb & Abarbanel
  (a third-order one-sided row 1, the fourth-order Padé row 2). Linearly
  unstable at an inviscid slip wall: a uniform state grows a wall-normal
  velocity from round-off at 2.3 per unit time on a unit domain, visible
  after about thirty time units in Float64, which the F2 row of
  `compact_filter(closures = :cascade)` damps and the default one-sided
  filter rows do not. Select it to reproduce earlier results.
- `:cascade4`: Lele's fourth-order one-sided row 1 (α = 3) over the same
  Padé row 2. It carries an undamped mode at an inviscid wall that only the
  F2 row of `compact_filter(closures = :cascade)` damps; under the default
  one-sided filter rows it fails even a smooth pulse.
- `:brady_livescu`: the four fifth-order rows of Brady & Livescu (2019),
  scheme T6. Select it for a high-order wall on a resolved start: under
  the default one-sided filter rows it is the supported high-order wall
  configuration within measured limits, a wall whose initial state is
  resolved (not the singular start of cold planar Noh), the CFL numbers
  the default closure completes a case at, either precision, the block
  extents the filter already requires, serial or decomposed. Its wall
  solution is sixth order with the artificial properties off; with them
  on a viscous or shear wall keeps that order and an inviscid slip wall is
  limited by the strain sensor's cusp (`beta_sensor = :dilatation` removes
  it). Costs: a closed-line condition number near 1e3 (see the source
  comment), which in Float32 floors one derivative's wall error near 1e-3
  from N = 48 up, and a wall mode under `compact_filter(closures =
  :cascade)` wherever the artificial bulk viscosity is active at a slip
  wall.
"""
function lele_d1_6(::Type{T}=Float64; closures::Symbol=:neutral3) where {T}
    CompactScheme{T}("Lele C6 first derivative", T(1//3), zero(T),
        T[7//9, 1//36],   # a/2, b/4  (multiply (f_{i+1}−f_{i-1}), (f_{i+2}−f_{i-2}))
        false,
        derivative_closures(T, closures, 2, BRADY_LIVESCU_T6))
end

"""
    lele_d1_8(T=Float64; closures=:neutral3)

Eighth-order tridiagonal first derivative (Lele 1992, eq. 2.1 with a
seven-point right-hand side): α = 3/8, a = 25/16, b = 1/5, c = −1/80
(consistency: a + b + c = 1 + 2α). It keeps the tridiagonal line solve of
[`lele_d1_6`](@ref) and costs two more multiply-adds per point: a few percent
of a step in a decomposed run (4% at 64³ per rank on eight single-threaded
ranks, within the run-to-run spread) and 10% on one rank. The interior
reaches ±3, so a closed edge takes three rows and `n_halo ≥ 3` is required.

The eighth order is the order of a periodic dimension and of the interior of
a closed one. At a closed edge the first row is third order under `:neutral3`
and `:cascade3`, so one derivative on a wall-bounded line is third order in
the maximum norm and a wall-bounded evolution is fourth order at the wall, as
on C6; see [`lele_d1_6`](@ref). `closures` selects the rows at a closed edge:

- `:neutral3` (default): the two rows of `lele_d1_6(closures = :neutral3)`
  followed by the C6 interior row. The Euler step linearized about a uniform
  state between slip walls is neutral, as at a Dirichlet end and a viscous
  no-slip wall, and a long inviscid run between slip walls holds its
  round-off seed where `:cascade3` grows. Same orders as `:cascade3`, a
  larger error constant at the wall and in the interior, and a
  better-conditioned closed line.
- `:cascade3`: the C6 cascade rows followed by the C6 interior row. Linearly
  unstable at an inviscid slip wall, growing at 1.4 per unit time, which the
  F2 row of `compact_filter(closures = :cascade)` damps and the default
  one-sided filter rows do not. Select it to reproduce earlier results.
- `:cascade4`: the same with Lele's fourth-order one-sided first row (α = 3).
  It needs the cascade filter rows for the same reason.
- `:brady_livescu`: the six seventh-order rows of Brady & Livescu (2019),
  scheme T8. Not a supported wall configuration: the rows fail under the
  filter's cascade rows even on smooth data, and under the default one-sided
  rows they fail a smooth wall from `cfl = 1.25` where the periodic interior
  completes, the warm-started planar Noh wall from 0.9, and every start of
  that case but a well-resolved one. They need 13 points along a dimension
  closed at both ends (7 with one end closed) and remain usable on a periodic
  or interior block.
"""
function lele_d1_8(::Type{T}=Float64; closures::Symbol=:neutral3) where {T}
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
wavenumber, which is the interior filter's own construction. At `i = M + 1`,
solving the system reproduces the centered stencil to round-off. At `i = 2`,
it reproduces the published `(1 + 254αf)/256, (31 + 2αf)/32, ...` row. Solving
the rows here instead of tabulating them preserves their exact dependence on
`af`.
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
    compact_filter(alphaf=0.45; closures=:onesided)

Eighth-order Gaitonde–Visbal compact filter. `alphaf ∈ (−0.5, 0.5)` sets the
strength (larger → weaker filtering). At a closed edge the first row is always
left unfiltered; `closures` selects rows 2–4:

- `:onesided` (default): the one-sided eighth-order rows of Gaitonde & Visbal
  (2000), derived at construction by `onesided_filter_row`. One pass is
  eighth order everywhere, and under repeated application the closed
  operator amplifies less than the cascade does (‖F¹⁰⁰‖₂ 1.05 against 1.14
  at αf = 0.45, N = 64). Its rows 2 and 3 do exceed unit gain at some
  wavenumbers taken alone (1.10 and 1.03 at αf = 0.45, worse at smaller αf),
  which the paper also notes: on a reflection resolved over fewer than about
  ten cells the rows read two to three times the cascade's error, and above
  that resolution ten to a thousand times less. In a wall-bounded evolution
  the wall window converges at the derivative closure's own order, the
  planar Noh wall deficit is 10–18 points smaller, and the mass and energy a
  closed line's filter creates fall by two orders. These rows do not damp
  the cascade closures' inviscid slip-wall mode, which the `:cascade` set's
  F2 row removed exactly: under them it grows at half its unfiltered rate,
  and faster at a smaller `alphaf` (see `lele_d1_6`).
- `:cascade`: centered compact filters of order 2, 4 and 6 with the same αf,
  the standard reduced-order boundary cascade. One filter pass of a smooth
  field is then second order in the maximum norm along the whole line, not
  only at the wall, because solving
  the coupled compact system propagates the row-2 truncation error into
  interior solution entries, and a filtered wall calculation is second order
  at the wall whatever the derivative closure. `lele_d1_6(closures =
  :cascade4)` needs this row set: without the F2 row its inviscid wall mode
  is undamped and it fails even a smooth pulse.
"""
function compact_filter(alphaf::Real=0.45, ::Type{T}=Float64;
                        closures::Symbol=:onesided) where {T}
    # The weights are evaluated in at least Float64 and rounded once to T, so
    # a Float32 filter carries the rounded Float64 weights. Solving the
    # one-sided rows' Vandermonde systems in Float32 instead leaves errors of
    # tens of ulps.
    S = promote_type(Float64, T)
    as = S(alphaf)
    af = T(as)
    a0 = (93 + 70as) / 128
    a1 = (7 + 18as) / 16
    a2 = (-7 + 14as) / 32
    a3 = (1 - 2as) / 16
    a4 = (-1 + 2as) / 128
    # Boundary closures: row 1 identity; under `:cascade` rows 2–4 host
    # centered compact filters of order 2, 4, 6 (same αf), the standard
    # reduced-order cascade. Consistency: the RHS coefficients of each of
    # these rows sum to 1 + 2αf.
    b2 = ((1 + 2as) / 2, (1 + 2as) / 2)                              # F2: a0, a1
    b4 = ((5 + 6as) / 8, (1 + 2as) / 2, (-1 + 2as) / 8)              # F4
    b6 = ((11 + 10as) / 16, (15 + 34as) / 32,
          (-3 + 6as) / 16, (1 - 2as) / 32)                           # F6
    ctr(c) = T[ [c[m + 1] / 2 for m in length(c)-1:-1:1]; c[1];
                [c[m + 1] / 2 for m in 1:length(c)-1] ]
    row1 = ClosureRow{T}((zero(T), one(T), zero(T)), T[1])
    cl = if closures === :cascade
        [row1,
         ClosureRow{T}((af, one(T), af), ctr(b2)),
         ClosureRow{T}((af, one(T), af), ctr(b4)),
         ClosureRow{T}((af, one(T), af), ctr(b6))]
    elseif closures === :onesided
        [row1; [ClosureRow{T}((af, one(T), af), T.(onesided_filter_row(as, i, 4)))
                for i in 2:4]]
    else
        error("unknown filter closure set $(repr(closures)); " *
              "use :cascade or :onesided")
    end
    CompactScheme{T}("Gaitonde–Visbal C8 filter", af, T(a0),
                     T[a1/2, a2/2, a3/2, a4/2], true, cl)
end

"""
    gaussian_filter()

Explicit nine-point Gaussian test filter, the smoother Cook's artificial
properties assume and the one [Pyranda](https://github.com/LLNL/pyranda)
applies as `gbar`
(`pyranda/parcop/stencils.f90`, `cgfs4`). The left-hand side is the identity,
so this is a `CompactScheme` only in the sense that it reuses the same fill,
fold and closure machinery; `plan_direction` detects the zero α and
skips the line solve and its interface reduction entirely.

The weights sum to exactly 1 over the common denominator 103680, so constants
are reproduced without relying on cancellation. Each of the four closure rows
is the interior stencil with its overhanging weights folded back onto the
half-offset mirror (ghost j ↔ interior j), which preserves that unit sum at a
closed edge; `fold_fill!` uses the same construction at a fold, so
the two edge treatments agree.

That mirror is half a cell out at a node-centred wall. Where this filter
serves as the sensor smoother, a reflecting wall face therefore takes the
node-centred rows of [`wall_closures`](@ref) instead, which fold the same
stencil onto the boundary node.

Contrast [`compact_filter`](@ref), a dealiasing filter for the conserved
state and not a test filter: at αf = 0.45 it retains 99% of the
amplitude at four points per wavelength where this filter retains 19%.
"""
function gaussian_filter(::Type{T}=Float64) where {T}
    # The folded row sums are formed in at least Float64 and rounded once to
    # T, as in `compact_filter`.
    S = promote_type(Float64, T)
    a = S(3565//10368); b = S(3091//12960); c = S(1997//25920)
    d = S(149//12960);  e = S(107//103680)
    lhs = (zero(T), one(T), zero(T))
    cl = [ClosureRow{T}(lhs, T[a+b, b+c, c+d, d+e, e]),
          ClosureRow{T}(lhs, T[b+c, a+d, b+e, c, d, e]),
          ClosureRow{T}(lhs, T[c+d, b+e, a, b, c, d, e]),
          ClosureRow{T}(lhs, T[d+e, c, b, a, b, c, d, e])]
    CompactScheme{T}("explicit 9-point Gaussian", zero(T), T(a), T[b, c, d, e], true, cl)
end

# --- Patch-interface closures ------------------------------------------------
#
# At a patch interface the ghost layers carry the abutting patch's data, but the
# ghost unknowns belong to that patch's solve, so a row's left-hand side must
# couple interior unknowns only (Pyranda tabulates the extended-data transfer
# closures identically to the one-sided ones for this reason: "same as
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
throughout (`gaussian_filter`) needs no closure at all, since the interior
stencil reads the exchanged ghosts. `plan_direction` verifies the ghost reach
against the halo width. A [`BandedCompactScheme`](@ref) takes `q` compact
rows per end, one per left-hand-side band that would reach a ghost unknown;
see the method in `kernels_banded.jl`.
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

"""
    interface_divergence_closures(scheme) -> closure rows

Closure rows for the flux divergence at a patch-interface end under
`interface_rhs = :extended`, where a flux array carries no ghost data and the
divergence keeps one-sided rows (`div_along!`). A scheme's own rows are kept,
except the `:neutral3` sets of [`lele_d1_6`](@ref), [`lele_d1_8`](@ref) and
[`lele_d1_10`](@ref), which the `:cascade3` rows of the same width replace:
the neutral rows are selected for the wall's injected condition, which an
interface does not impose, and the cascade rows carry the smaller truncation
constants. The pentadiagonal method is in `kernels_banded.jl`.
"""
interface_divergence_closures(scheme::AbstractCompactScheme) = scheme.closures
function interface_divergence_closures(scheme::CompactScheme{T}) where {T}
    # `neutral_closures` is defined for two- and three-row edges only, so the
    # row count decides before the comparison is formed.
    nrows = nclosure(scheme)
    2 <= nrows <= 3 || return scheme.closures
    scheme.closures == neutral_closures(T, nrows) ?
        cascade_closures(T, nrows, 3) : scheme.closures
end

# Closure rows of the flux divergence at a patch or level interface end: those
# of `interface_divergence_closures(deriv)` when the `interface_divergence`
# source is `nothing`, the source scheme's own closure rows otherwise.
interface_divergence_rows(deriv::AbstractCompactScheme, ::Nothing) =
    interface_divergence_closures(deriv)
function interface_divergence_rows(deriv::AbstractCompactScheme,
                                   source::AbstractCompactScheme)
    # A closure row is derived against the interior rows it hands over to:
    # its truncation constants and the stability of the closed line both
    # assume that interior, so rows taken from a scheme with other interior
    # coefficients, or another element type, are not admitted.
    _same_interior(deriv, source) ||
        throw(ArgumentError("interface_divergence must carry the interior " *
                            "coefficients and element type of deriv; " *
                            "'$(source.name)' ($(typeof(source))) does not " *
                            "match '$(deriv.name)' ($(typeof(deriv)))"))
    source.symmetric &&
        throw(ArgumentError("interface_divergence must be a first-derivative " *
                            "scheme; '$(source.name)' is symmetric"))
    nclosure(source) >= 1 ||
        throw(ArgumentError("interface_divergence '$(source.name)' carries no " *
                            "closure rows"))
    return source.closures
end

_same_interior(a::AbstractCompactScheme, b::AbstractCompactScheme) = false
_same_interior(a::CompactScheme{T}, b::CompactScheme{T}) where {T} =
    a.alpha == b.alpha && a.a0 == b.a0 && a.coeffs == b.coeffs &&
    a.symmetric == b.symmetric

# --- Reflecting-wall closures ------------------------------------------------
#
# At a reflecting wall the solution continues past the boundary node as its own
# reflection about that node, f at 2−q scaled by the field's sign σ, which is
# the continuation `delta4_sum!` reads for the explicit detector. A symmetric
# operator needs no closure of its own there: a field of parity σ maps to a
# result of parity σ, so both sides of the row fold onto the same mirror and
# the interior stencil closes itself. The rows below are that fold.
#
# The mirror is node-centred, one node further out than the half-offset mirror
# (ghost j ↔ interior j) a coordinate fold takes and the built-in rows of
# `gaussian_filter` and `compact_d8` fold onto. Those rows are half a cell out
# at a wall, which is what these replace.

"""
    wall_closures(scheme, σ) -> Vector{ClosureRow}

Closure rows folding the interior stencil of a symmetric `scheme` onto the
node-centred mirror of a reflecting wall, for a field of parity `σ` across it
(`+1` even, `−1` odd). Row `j` carries the interior weights with every tap at
an index `q < 1` added onto `2 − q` with the sign `σ`, and the left-hand-side
unknown folded the same way, so `halfwidth(scheme)` rows close the edge.

The rows are built from the interior weights rather than tabulated, so a
filter's unit row sum and an even derivative's zero row sum are inherited
from the interior stencil at `σ = +1`. This is the sensor operators' wall
hook: the artificial-property smoother and the `:d8` detector take these rows
at a face [`sensor_mirror`](@ref) names, matching the mirror the `:delta4`
detector reads directly. An antisymmetric scheme has no such fold and raises
an error.
"""
function wall_closures(scheme::CompactScheme{T}, σ::Int) where {T}
    scheme.symmetric || error("wall closures need a symmetric scheme; " *
                              "'$(scheme.name)' is antisymmetric")
    s = T(σ)
    M = halfwidth(scheme)
    rows = ClosureRow{T}[]
    for j in 1:M
        rhs = zeros(T, j + M)
        rhs[j] += scheme.a0
        for m in 1:M
            rhs[j+m] += scheme.coeffs[m]
            q = j - m
            q >= 1 ? (rhs[q] += scheme.coeffs[m]) :
                     (rhs[2-q] += s * scheme.coeffs[m])
        end
        # Row 1's ghost unknown g₀ = σ g₂ moves onto the superdiagonal; every
        # row below it couples interior unknowns already.
        lhs = j == 1 ? (zero(T), one(T), (one(T) + s) * scheme.alpha) :
              (scheme.alpha, one(T), scheme.alpha)
        push!(rows, ClosureRow{T}(lhs, rhs))
    end
    return rows
end
