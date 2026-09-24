# Choose numerics for accuracy per cost

The derivative operator, boundary closure rows, state filter and floating-point
precision all affect accuracy and cost. Choose them together: a higher-order
interior derivative helps on a periodic domain, but boundary errors can limit
its benefit on a bounded domain.

This page compares the supported choices and gives recipes for common problems.
For the numerical methods, see
[Spatial and temporal discretization](@ref) and
[Filtering and artificial properties](@ref); the measured orders are in
[Verification, validation, and calibration](@ref).

## Global spatial accuracy

The formal order of a compact derivative applies on a periodic domain.
In a non-periodic dimension, lower-order closure rows replace the interior
stencil near each end. The compact solve carries their truncation error inward,
so the boundaries can limit accuracy along the whole line. The order of a
single derivative and the order of an evolved solution are different
measurements:

| Configuration | Maximum-norm order | Measured |
|:--|--:|:--|
| periodic C6 / C8 / C10 | 6 / 8 / 10 | 6.01 / 8.00 / 10.04 |
| one derivative on a wall-bounded line, C6/8/10, default rows | 3 | 3.18 (C6, C8 and C10, the same errors) |
| wall-bounded evolution at the wall, default rows | 4 | 4.01 |
| wall-bounded evolution, C6 `:brady_livescu` | 6 | 5.73 |
| cylindrical axis, spherical origin (folds) | 3 | 3.76 / 2.97 |
| patch or level interface | 3 to 4 | 3.31 / 3.62 |
| patch or level interface, C6 `interface_divergence` `:brady_livescu`, Float64 | 5 to 6 | 5.8 to 7.0 / 5.1 to 6.0 |
| patch or level interface, C6 `interface_flux = :ghost`, level interpolation order 8, Float64 | 6 | 5.4 to 6.9 / 5.8 to 6.1 |

With default wall closures, the closed-line C8 and C10 studies have the same
wall and maximum-norm errors as C6, to the printed digits. Their benefit is
greater resolving power away from the ends: less error at a given number of
points per wavelength in the interior and in periodic dimensions. On a fully
periodic box, each operator delivers its formal order throughout the domain.

To raise the wall order, change the closure coefficients. Increasing the
interior derivative order alone does not help. The supported higher-order
wall configuration is C6 `:brady_livescu` with a resolved initial state,
described below.

## Built-in derivative operators

| Operator | Line solve | Step cost, 8 ranks at one thread | Minimum local extent | Choose it when |
|:--|:--|:--|--:|:--|
| [`lele_d1_6`](@ref) (default) | tridiagonal | reference | 5 | walls, folds or interfaces limit accuracy; also the usual choice for Float32 or device runs |
| [`lele_d1_8`](@ref) | tridiagonal, same solve as C6 | +4% | 5 | a periodic or mostly periodic problem needs more resolving power at little additional cost |
| [`lele_d1_10`](@ref) | pentadiagonal | +16% at 64³ per rank, +20% at 32³ | 7 | a periodic problem needs more resolving power and can afford roughly 20% more per step |

All three support serial, decomposed, multi-patch, refined and device runs.
The timings below use a single-species cube with the default filter and
artificial properties on a desktop with eight performance cores. The
eight-rank runs use one thread per rank; a serial timing is included for
comparison.

| Configuration, ns per point per step | C6 | C8 | C10 |
|:--|--:|--:|--:|
| periodic, 128³ over 8 ranks (64³ per rank) | 444 | 462 | 517 |
| slip walls in x, 128³ over 8 ranks | 447 | 459 | 518 |
| periodic, 64³ over 8 ranks (32³ per rank) | 503 | 521 | 605 |
| periodic, one rank, one thread, 64³ | 1354 | 1486 | 1749 |

The step-cost increase is modest because derivative solves account for only
20–25% of a right-hand-side evaluation. Halo exchanges, artificial-property
calculations and filtering of the conserved state have the same cost for all
three operators. For the derivative calculation alone, excluding halo
exchange, C8 is 10% slower than C6 and C10 is 29% slower.

Reducing the block from 64³ to 32³ points per rank increases communication's
share of the work. Cost per point rises by about an eighth, and the C10
penalty grows. These Windows MPI timings also vary by 3–4% between runs
because ranks are not pinned to CPU cores. That variation is as large as
the measured C8 step-cost increase.

The minimum local extent is the number of points each rank needs along a
decomposed dimension. The default state filter requires nine, exceeding
the requirement of every derivative operator in the table. With filtering
enabled, the global extent must therefore be at least nine times the number
of ranks along that dimension. See [Run in parallel](@ref).

## Closure rows at a closed edge

Each derivative operator's `closures` keyword selects the rows used at
non-periodic ends, such as walls, Dirichlet inflows and outflows.

| Closure rows | Row orders | Wall stability | Accuracy and conditioning relative to `:cascade3` | Availability |
|:--|:--|:--|:--|:--|
| `:neutral3` (default) | 3, 4, then the sixth-order interior row | neutral at slip walls, no-slip walls and Dirichlet ends, at every line length | 1.35× wall error and 3.4× interior error; better-conditioned line solve | C6, C8, C10 |
| `:cascade3` | 3, 4, 6 | grows at an inviscid slip wall from round-off, visible after about thirty time units in Float64 | reference | C6, C8, C10 |
| `:cascade4` | 4, 4, 6 | the same growth; needs `compact_filter(closures = :cascade)` | raises the wall derivative order by one, but requires a second-order filter along the line | C6, C8 |
| `:brady_livescu` | one below the interior on every row | neutral unfiltered; grows slowly under the default filter | sixth-order wall on C6, condition number near 1e3 | C6 supported; C8 unsupported at a wall |

Keep `:neutral3` as the general choice, including for long inviscid runs
between slip walls or symmetry planes. If you use cascade derivative rows
instead, their wall oscillation needs damping from
`compact_filter(closures = :cascade)` or physical viscosity.

For a higher-order wall, choose C6 `:brady_livescu` with a resolved initial
state: no discontinuity within about thirteen cells of the wall. Singular
cold starts remain outside this qualified configuration. A shock that
arrives later has already been spread by the artificial properties; the
measured error is then within a few tenths of a percent.

The Brady–Livescu rows lose about three digits of wall accuracy through
conditioning. In Float32, a single derivative's wall error reaches a floor
near 1e-3 at roughly fifty points and above, so use Float64 to benefit from
these rows. C8 `:brady_livescu` fails the smooth-wall study from CFL 1.25
and is not supported as a wall configuration.

Patch and level interfaces keep the cascade rows regardless of the selected
wall closures. An interface imposes no boundary condition, and the cascade
rows have smaller truncation-error constants there.

The `interface_divergence` keyword replaces the flux divergence's rows at
interface ends only, leaving walls and the gradient rows unchanged. Pass a
scheme with the same interior coefficients and element type as `deriv`, such
as `interface_divergence = lele_d1_6(closures = :brady_livescu)` beside the
default `deriv`; any other scheme is rejected at setup. On smooth Float64
flows, the Brady–Livescu rows lower the interface error by one to three
orders of magnitude and raise the measured order to about 6, or 5 for an
acoustic wave through a coarse–fine face. They cost nothing per step. They
are experimental, and are not recommended for:

- Float32 runs, whose interface error floor they do not lower;
- shocks crossing same-level patch planes, where they halve the minimum
  pressure behind the shock;
- runs that use `interface_rhs = :onesided` to survive a discontinuity on a
  shared plane, which then fail within two steps.

The `:cascade4` rows are unstable between a wall and an interface; do not
use them here.

The experimental `interface_flux = :ghost` removes the closure rows from the
inviscid part of the divergence. It evaluates the inviscid flux on the
interface ghost layers and differentiates it through the interface with the
interior stencil, so a same-level interface has the interior's error. At a
coarse-fine interface the ghost values are interpolated from the parent, and
sixth order in the solution requires `level_interpolation_order = 8`. The
viscous and artificial fluxes keep the closure rows, which
`interface_divergence` selects. This option also runs a discontinuity
placed on a shared patch plane, which fails under the closure rows. It
requires `interface_rhs = :extended` and an unstretched Cartesian grid,
and it adds 5 to 25% to the step time. Float32 runs do not benefit.

## Boundary conditions

Choose each face's condition from its physics, as described in
[Choose boundary conditions](@ref). The accuracy and cost implications are:

| Condition | Order at the face | Stability | Cost | Notes |
|:--|:--|:--|:--|:--|
| [`PeriodicBC`](@ref) | the interior operator's | unconditional | cheapest; cyclic solve | the only face that keeps the formal order |
| [`SlipWallBC`](@ref) | closure rows: 3, evolution 4 | neutral under `:neutral3` | none beyond the rows | an inviscid slip wall with the artificial properties on is limited by the strain sensor's cusp; `beta_sensor = :dilatation` removes that cap |
| [`SymmetryPlaneBC`](@ref) | the interior operator's | unconditional: the folded step is the periodic step restricted by parity | none; a single unrefined, unstretched patch | the slip wall half a cell outside the end node; nothing is injected and no closure row exists |
| [`NoSlipWallBC`](@ref) | closure rows: 3, evolution 4 | neutral, viscosity damps every closure's wall mode | the wall flux contract | adiabatic by default; a wall temperature makes it isothermal |
| [`DirichletBC`](@ref) | closure rows | neutral under every closure option | none | shock tubes and supersonic inflow |
| [`NSCBCInflowBC`](@ref), [`NSCBCOutflowBC`](@ref) | closure rows | depends on the relaxation scale | one to three compact solves per face per stage, six for an inflow carrying its transverse terms | faces with a unit scale factor only; both carry the transverse coupling of Yoo and Im, weighted by `beta_t` |
| folds ([`AxisBC`](@ref), [`OriginBC`](@ref), [`PoleBC`](@ref)) | 3 | neutral | the fold exchange | the spherical origin needs data resolved over three cells and a CFL of 0.3 on a converging shock |

When the problem's symmetry permits a periodic or folded dimension, use it
instead of adding a wall at the symmetry plane. This avoids the wall's
closure rows and improves accuracy and cost. A symmetry plane is itself a
fold: [`SymmetryPlaneBC`](@ref) keeps the interior order at the plane where
[`SlipWallBC`](@ref) pays the closure rows' third order, at the cost of a
grid whose end node sits half a cell inside the plane.

## Filter and artificial properties

By default, the compact filter removes grid-scale content from the conserved
state after every step. Artificial properties regularize shocks and
interfaces. They do not replace the filter: turbulent runs fail without
filtering even when artificial properties are enabled. Use
`filter_interval = 0` for diagnostics rather than to reduce production cost.

| Setting | Default | Alternative and when to use it |
|:--|:--|:--|
| filter closure rows | `:onesided`, eighth order along the whole closed line | `:cascade` (second order along the line): only with `:cascade4`, or to damp the cascade rows' wall mode |
| `filter_cfl` | 0.35, the reference CFL at which a pass is full strength | 0 applies every pass at full strength; dissipation per unit time then depends on the timestep |
| `smoother` | `:gaussian`, explicit | `:compact`, one pass of the state filter per sensor: a quarter of the right-hand side in the multicomponent case, and one sweep per species |
| `beta_sensor` | `:strain` | `:dilatation` on an inviscid wall or wherever the strain cusp costs order |
| `detector` | `:delta4` | `:d8` separates shocks from smooth flow only with a cusp-free sensor field, and lowers the spherical-origin CFL ceiling to 0.25 |
| `species_flux` | `:partial_density`, one diffusivity on the partial densities with the mass flux carried into momentum and energy | `:bulk` for a shocked interface at a density ratio of 100 or more, at about a tenth more per step; `:fickian`, Cook's per-species flux, is cheaper and moves the pressure at an interface of unequal molecular weight (see [Filtering and artificial properties](@ref)) |

Sensor smoothing is the largest single cost in the artificial-property
calculation. With the default smoother, it takes about a quarter of a
multicomponent right-hand-side evaluation and requires one sweep per
species. The species-smoothing machinery has no measured effect with two
species; its benefit appears at three or more.

## Precision and step size

Float32 uses half the memory of Float64 and is used by the device path.
With default closures, the Float32 wall-evolution error reaches a floor
near 3e-5, and a freestream at a wall holds to roundoff of about 2e-6.
At 96 points, a single derivative's wall error is about 9e-5 in either
precision. Keep the default closures in Float32: the Brady–Livescu rows
have a higher wall-derivative error floor, near 1e-3.

The default `cfl = 0.5` completes every case in the regression battery except
the converging strong shock at the spherical origin, whose limit is 0.3
at every tested resolution. `StepControl(retries = 4)` recovers that case
by rolling back and retrying at a lower CFL; the same mechanism handles
startup transients.

The three-dimensional CFL uses a Euclidean bound over the three directions.
For the same timestep, its value is smaller than a per-direction CFL
reported in the literature. See
[Control and diagnose a run](@ref) for retries and timestep diagnostics.

## Recipes

| Problem | Derivative | Closures | Filter | Sensors | Precision | Reason |
|:--|:--|:--|:--|:--|:--|:--|
| periodic turbulence box | C8 or C10 | not used | default | default | Float64, or Float32 on a device | every dimension retains the formal order; choose based on resolving power per point and step cost |
| wall-bounded channel or cavity | C6 | `:brady_livescu` for a sixth-order wall with a resolved start; `:neutral3` for a shocked start | default `:onesided` rows | `:dilatation` if the walls are inviscid | Float64 | wall closure coefficients determine the wall order |
| shock tube, Dirichlet ends | C6 | `:neutral3` | default | default | either | the ends are neutral and the shock sets the resolution |
| converging shock on a cylindrical axis or spherical origin | C6 | `:neutral3` | default | default; not `:d8` at the origin | Float64 | the fold is third order; at the origin start with resolved data, `cfl = 0.3` and retries |
| long inviscid run between symmetry planes | any | `:neutral3` | default | default | either | the neutral rows hold the round-off seed; the cascade rows do not |
