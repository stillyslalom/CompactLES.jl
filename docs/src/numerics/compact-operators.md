# Compact operators

CompactLES represents a directional compact operator as an implicit banded
relation between the output ``g`` and input ``f``. For the tridiagonal family,

```math
\alpha g_{i-1} + g_i + \alpha g_{i+1}
= a_0 f_i + \sum_{m=1}^{M} c_m
  \left(f_{i+m} \mathbin{\pm} f_{i-m}\right).
```

The minus sign gives an antisymmetric first derivative; the plus sign and
center coefficient ``a_0`` give a symmetric filter. The right-hand side of a
derivative is scaled by ``1/h`` when its directional plan is constructed.

## Built-in schemes

| Constructor | Interior scheme | Implicit band | Closed-edge behavior |
|:--|:--|:--|:--|
| [`pade_d1_4`](@ref) | fourth-order derivative | tridiagonal | third-order first row |
| [`lele_d1_6`](@ref) | sixth-order derivative | tridiagonal | third/fourth-order first two rows |
| [`lele_d1_10`](@ref) | tenth-order derivative | pentadiagonal | C6 closure cascade on the first three rows |
| [`compact_filter`](@ref) | eighth-order filter | tridiagonal | identity, then second/fourth/sixth-order rows |

The interior order is not the global order of a closed or folded domain.
Boundary closures and singularity folds are approximately third order in the
current convergence suite. Periodic domains expose the sixth- and tenth-order
interior behavior.

## Applying a scheme

[`Numerics`](@ref) accepts derivative and filter definitions directly:

```julia
num = Numerics(
    n_global = (128, 64, 64),
    deriv = lele_d1_10(),
    filt = compact_filter(0.45),
)
```

`setup` constructs one directional plan per active dimension. Each application
packs all grid lines, evaluates the explicit right-hand side, solves the
implicit system across every rank in that direction, and scatters the result.
The y and z paths use a transposed line layout so their inner memory access
remains contiguous.

The low-level [`apply_along!`](@ref) API requires current rank-boundary halos.
[`filter_field!`](@ref) performs its own exchange before each directional
filter pass.

## Local extent constraints

Every rank-local extent must be large enough for both the implicit closure rows
and the explicit stencil:

- C6 requires at least 5 points;
- C10 requires at least 7 points; and
- the C8 filter requires at least 9 points.

Because filtering is enabled by default, 9 points per rank in every decomposed
dimension is the practical minimum. A grid can therefore be valid for its
derivative scheme and still be too small for filtering.

## Custom schemes

[`CompactScheme`](@ref) and [`BandedCompactScheme`](@ref) are public so a
custom coefficient set can use the same planning and distributed solves.
Closure rows are specified at the low edge; the high edge is mirrored
automatically, including the sign change for derivatives. New coefficient
sets should add periodic and closed-domain convergence guards before being used
in production.
