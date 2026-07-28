# Spatial and temporal discretization

## Compact first derivatives

A compact finite difference defines derivative values implicitly. The default
tridiagonal family has the form

```math
\alpha g_{i-1} + g_i + \alpha g_{i+1}
= \frac{1}{h}\sum_{m=1}^{M} c_m(f_{i+m}-f_{i-m}),
```

where ``g`` approximates ``df/dx``. Solving the banded left-hand side couples
the complete grid line. Compared with an explicit stencil of similar width,
this implicit relation resolves a larger fraction of the representable
wavenumber range accurately.

Built-in derivatives are fourth-order Padé, sixth-order Lele C6, and
tenth-order pentadiagonal Lele C10. These orders describe a periodic interior.
One-sided boundary closures and coordinate folds reduce the measured global
order to approximately three in current tests.

## Conservative flux divergence

The solver first constructs physical fluxes at nodes. It then differentiates
area-weighted fluxes:

```math
\nabla\!\cdot\boldsymbol{F}
= \frac{1}{J}\sum_d \frac{\partial(A_d F_d)}{\partial \xi_d},
\qquad A_d=\frac{J}{h_d}.
```

This form preserves telescoping conservation on the structured grid and admits
one common derivative machinery for Cartesian and orthogonal-curvilinear
metrics.

## Boundary closures

Closed dimensions include their endpoints. The first rows use one-sided
closure relations, and the high-side rows are mirrored automatically. A
periodic dimension omits the duplicate endpoint and uses a cyclic compact
system.

Formal interior order should not be quoted as the accuracy of a wall-bounded
calculation. If the error is dominated by closure points, refinement exposes
the lower closure order. This is why the validation suite reports interior and
closed-domain convergence separately.

## Time integration

Time advancement uses the five-stage, fourth-order, low-storage
Carpenter--Kennedy Runge--Kutta scheme. Each stage:

1. applies hard boundary conditions;
2. evaluates primitive variables, gradients, regularization, fluxes, sources,
   and characteristic corrections; and
3. updates the two low-storage stage arrays.

Time-dependent Dirichlet and NSCBC targets are evaluated at stage time.
`Workspace` owns the reusable stage arrays.

## Explicit timestep

The CFL estimate contains acoustic, diffusive, and collapsed-coordinate
curvature rates. In schematic form,

```math
\Delta t = \frac{\mathrm{CFL}}
 {\sum_d (|u_d|+c)/\Delta x_d
  + C_D\nu/\Delta x_d^2
  + R_{\mathrm{curvature}}}.
```

Physical spacing includes mesh stretching and metric scale factors. A resolved
azimuthal direction therefore has spacing `r * Delta theta` near a cylindrical
axis; this can dominate the timestep even when radial spacing is moderate.
`dt_report` identifies the actual limiting term and location.

Artificial coefficients used by the estimate lag by one completed RHS
evaluation. `StepControl` mitigates a strong startup transient by rolling back
and retrying at lower CFL, but it does not make an intrinsically unstable
configuration valid.

## Resolution requirements

Rank-local extent must accommodate the explicit stencil and implicit closure:

| Operator | Minimum local extent |
|:--|--:|
| C6 derivative | 5 |
| C10 derivative | 7 |
| C8 compact filter | 9 |

Because the filter is enabled by default, nine is the normal practical
minimum. Accuracy ordinarily requires substantially more points than this
algebraic minimum.

## Error interpretation

Truncation error, time-integration error, filtering, artificial transport, and
physical-model error are separate contributions. A grid-refinement study can
measure discretization convergence only while model parameters and the
physical problem remain fixed. Shock solutions are not pointwise smooth, so
high formal order does not imply high-order convergence at the discontinuity.
