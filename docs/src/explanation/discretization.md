# Spatial and temporal discretization

The governing equations are continuous in space and time. CompactLES first
replaces their spatial derivatives by grid operators, producing the
method-of-lines system

```math
\frac{dQ_h}{dt}=R_h(Q_h,t),
```

and then advances this finite system explicitly. The compact spatial
derivative requires a banded solve along each grid line; that implicit spatial
relation should not be confused with an implicit time integrator.

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

## From state to spatial right-hand side

One evaluation of ``R_h(Q_h,t)`` performs the following operations:

1. enforce or fill the state required by boundary and fold conditions;
2. recover density, velocity, pressure, temperature, sound speed, and
   composition from the conserved state;
3. differentiate velocity, temperature, and mass fractions, including metric
   corrections to physical gradients;
4. construct molecular and artificial transport and assemble the complete
   species, momentum, and energy fluxes;
5. differentiate area-weighted fluxes and add geometric terms; and
6. apply right-hand-side boundary corrections and explicit sources at the
   Runge--Kutta stage time.

Collapsed dimensions contribute neither a derivative nor an operator solve.
They can still leave geometric terms in the equations, as the radial tutorial
demonstrates.

## Conservative flux divergence

For an orthogonal metric, the solver differentiates each complete physical
flux after multiplying it by the corresponding face-area factor:

```math
\nabla\!\cdot\boldsymbol{F}
= \frac{1}{J}\sum_d \frac{\partial(A_d F_d)}{\partial \xi_d},
\qquad A_d=\frac{J}{h_d}.
```

Here ``h_d`` are the metric scale factors, ``J=h_1h_2h_3`` is the volume
Jacobian, and ``A_d`` converts a flux normal to direction ``d`` into a flux per
computational area. Differentiating ``A_dF_d`` as one quantity is essential:
expanding the product into separately discretized terms would not generally
produce the same global balance.

### What “telescoping conservation” means

In one dimension, let ``D`` be the discrete derivative and ``w_i`` the
computational quadrature weights. A conservative summation identity has the
form

```math
\sum_i w_i(DG)_i=G_{\mathrm{high}}-G_{\mathrm{low}}.
```

This is the discrete analogue of integrating a derivative: contributions from
the interior cancel, leaving only physical boundary fluxes. On a periodic line
the two boundary values are the same, so the right-hand side is zero. Applying
that identity to ``G=A_dF_d`` gives the metric-weighted balance

```math
\sum_i w_iJ_i\frac{dQ_i}{dt}
=-\sum_d\sum_i w_iD_d(A_dF_d)_i=0
```

for a periodic domain without sources or boundary corrections. A decomposition
over MPI ranks does not add new physical boundaries: the distributed compact
solve represents the same global line, so rank-interface contributions do not
create or destroy the summed conserved quantity.

This is the scope verified by the serial and distributed periodic conservation
tests. It is not a blanket statement that every one-sided closure or every
operation in a timestep preserves an arbitrary discrete integral. Physical
boundary fluxes, characteristic corrections, and explicit sources change the
balance as specified by the problem. The compact state filter is applied after
the Runge--Kutta step and has its own conservation properties; it is not part
of the flux-divergence identity above.

## Boundary closures

Closed dimensions include their endpoints. The first rows use one-sided
closure relations, and the high-side rows are mirrored automatically. A
periodic dimension omits the duplicate endpoint and uses a cyclic compact
system. Coordinate singularities instead use the parity and antipodal folds
described in [Curvilinear coordinates](@ref).

The tridiagonal derivatives [`lele_d1_6`](@ref) and [`lele_d1_8`](@ref) offer
three closure sets through their `closures` keyword. The default `:cascade3`
is the reduced-order cascade of Carpenter, Gottlieb and Abarbanel (1993): a
third-order one-sided first row, a fourth-order Padé second row, and the
sixth-order interior row where a third is needed. `:cascade4` raises the first
row to Lele's fourth-order one-sided relation. `:brady_livescu` applies the
closure rows of Brady and Livescu (2019), one order below the interior on
every row, constructed to be discretely conservative and selected by
optimization on the Euler equations for stability without a filter. Their
rows are far from diagonally dominant, and the closed-line condition number
rises from about 16 to about 1e3, which costs three digits at the wall and is
the reason the cascade remains the default.

Formal interior order should not be quoted as the accuracy of a wall-bounded
calculation. If the error is dominated by closure points, refinement exposes
the lower closure order. This is why the validation suite reports interior and
closed-domain convergence separately, and reports the closed-domain order for
each closure set.

## Five-stage time advancement

Time advancement uses the five-stage, fourth-order Carpenter--Kennedy
low-storage Runge--Kutta scheme, abbreviated LSRK(5,4).
Starting from ``Q^{(0)}=Q^n``, its stage recurrence can be written

```math
G^{(s)} = a_sG^{(s-1)}
          +\Delta t\,R_h\!\left(Q^{(s-1)},t^n+c_s\Delta t\right),
\qquad
Q^{(s)} = Q^{(s-1)}+b_sG^{(s)},
```

for ``s=1,\ldots,5``. The first coefficient ``a_1=0`` resets the residual
register at the start of every step. Apart from the conserved state, the
implementation retains one right-hand-side array and one residual array rather
than storing all five stage states.

At each stage, `step!` sets the stage time, applies hard boundary conditions,
evaluates ``R_h``, and updates the low-storage state and residual. Time-dependent
Dirichlet and NSCBC targets and explicit sources therefore see the appropriate
stage time. After stage five, boundary conditions are applied once more to the
completed state.

The older shorthand “RK45” in implementation-oriented material means five
stages and fourth-order accuracy here. It does **not** mean an embedded
adaptive fourth/fifth-order pair: there is no temporal error estimate and no
accept/reject controller based on truncation error.

## Timestep selection and completed-step order

Before each full Runge--Kutta step, `run!` estimates a global stability rate
from the state entering the step. Schematically,

```math
\Delta t=\frac{\mathrm{CFL}}{\max_i \lambda_i},\qquad
\lambda_i=\sum_d\left(\frac{|u_d|+c}{\Delta x_d}
+\frac{2\nu}{\Delta x_d^{2}}\right)+R_{\mathrm{curvature}}.
```

Here ``i`` indexes grid points and ``d`` the resolved directions, ``c`` is the
mixture sound speed, ``\Delta x_d`` the physical spacing along ``d``, and ``\nu``
the sum of the kinematic momentum, thermal, and species diffusivities at that
point, each including its artificial contribution and the last taken over the
most diffusive species. The factor two is the diffusive safety constant, not one
of the `ArtParams` coefficients.
``R_{\mathrm{curvature}}`` is the geometric source rate that a collapsed angular
dimension would otherwise contribute unaccounted; it is zero in Cartesian
coordinates and for every resolved angular direction.

Physical spacing includes mesh stretching and metric scale factors. A resolved
azimuthal direction therefore has spacing ``r\Delta\theta`` near a cylindrical
axis; this can dominate the timestep even when radial spacing is moderate.
[`dt_report`](@ref) identifies the actual limiting term and location.

One accepted outer-loop iteration occurs in this order:

1. recover the acoustic, diffusive, and curvature stability rates and choose
   ``\Delta t`` from the CFL number;
2. shorten the step if necessary to land on `tfinal` or a scheduled output
   time;
3. perform all five Runge--Kutta stages;
4. advance the simulation clock and step counter;
5. filter every conserved component when `filter_interval` is due; and
6. invoke callbacks on the completed, optionally filtered state.

Artificial coefficients in the timestep estimate lag by one completed RHS
evaluation. `StepControl` can mitigate a strong startup transient by rolling
back to a saved state and retrying at lower CFL, but it does not make an
intrinsically unstable configuration valid. See
[Control and diagnose a run](@ref) for endpoint scheduling, retries, and
timestep diagnostics.

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
physical-model error are separate contributions. LSRK(5,4) is fourth-order in
time for a sufficiently smooth semi-discrete problem, but observing that order
requires spatial error and regularization error to be smaller. A refinement
study can measure discretization convergence only while model parameters and
the physical problem remain fixed. Shock solutions are not pointwise smooth,
so high formal order does not imply high-order convergence at a discontinuity.

See [Verification, validation, and calibration](@ref) for the tests that
separate these error sources and for the current observed spatial orders.
