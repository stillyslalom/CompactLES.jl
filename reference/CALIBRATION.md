# Calibration: the defaults and how to tune them

This file states the calibrated defaults of the artificial-property and
compact-filter settings, what each setting does, and which one to change when
a run misbehaves. Every number in it summarizes a measurement in
[CALIBRATION_APPENDIX.md](CALIBRATION_APPENDIX.md), which carries the full
sweeps together with the dead ends and null results. Read the linked section
before moving a default or reopening a rejected explanation.

The one-dimensional results are reproduced by `bench/artcal.jl` on the cases
of `test/cases.jl` that `test/validation.jl` guards, and the Taylor–Green
results by `bench/tgv_energy.jl`; the appendix names the script behind every
other table.

## Contents

1. [The defaults](#the-defaults)
2. [Which setting to change](#which-setting-to-change)
3. [The battery and how to read it](#the-battery-and-how-to-read-it)
4. [The constants](#the-constants)
5. [The sensor options](#the-sensor-options)
6. [The compact filter](#the-compact-filter)
7. [CFL, step control and the symmetry cell](#cfl-step-control-and-the-symmetry-cell)
8. [Walls, folds and metrics](#walls-folds-and-metrics)
9. [The species channel](#the-species-channel)
10. [Open items](#open-items)

## The defaults

```
ArtParams(C_mu = 0.002, C_beta = 1.0, C_kappa = 0.01, C_D = 0.01, C_Y = 100,
          Y_tolerance = 1e-4, mu_sensor = :strain, beta_sensor = :strain,
          reduction = :sum, smoother = :gaussian, detector = :delta4,
          species_flux = :fickian)
Numerics(filt = compact_filter(0.45), filter_interval = 1, filter_cfl = 0.35,
         filter_weighting = :none, cfl = 0.5, control = StepControl())
```

| Setting | Default | Status | Basis |
|---|---|---|---|
| `C_beta` | 1.0 | keep | Accuracy optimum near 0.4; 1.0 maximizes the spherical-origin CFL ceiling and is the one value viable under both detectors ([appendix](CALIBRATION_APPENDIX.md#c_beta-the-shock-constant)). |
| `C_kappa` | 0.01 | keep | The wall-heating trough under the default smoother; zero loses spherical Noh under `:compact` ([appendix](CALIBRATION_APPENDIX.md#c_kappa-the-conductivity)). |
| `C_mu` | 0.002 | keep | Inert in one dimension; above 0.008 spherical Noh fails. Taylor–Green is consistent with it and cannot select it ([appendix](CALIBRATION_APPENDIX.md#c_mu-the-shear-viscosity)). |
| `C_D` | 0.01 | keep | The filter dominates interface broadening; a 64× sweep moves the width 22% ([appendix](CALIBRATION_APPENDIX.md#c_d-the-species-diffusivity)). |
| `C_Y` | 100 | keep | A shocked 2h interface rings to ±0.2 without the bound and ±0.013 with it ([appendix](CALIBRATION_APPENDIX.md#c_y-the-mass-fraction-bound)). |
| `Y_tolerance` | 1e-4 | keep | A dead band restoring the unbounded order on smooth profiles touching 0 or 1 ([appendix](CALIBRATION_APPENDIX.md#the-dead-band)). |
| `smoother` | `:gaussian` | keep | Raised the origin ceiling 0.15 → 0.4 in its sweep (0.3 under the current defaults), 29% cheaper, costs seven points of wall heating ([appendix](CALIBRATION_APPENDIX.md#the-sensor-smoother)). |
| `detector` | `:delta4` | provisional | `:d8` improves six of seven columns and halves the wall deficit at high CFL, but lowers the origin's ceiling 0.3 → 0.25 ([appendix](CALIBRATION_APPENDIX.md#the-ringing-detector)). |
| `mu_sensor`, `beta_sensor`, `reduction` | `:strain`, `:strain`, `:sum` | keep | The alternatives move no column past the fourth digit or cost a converging geometry ([appendix](CALIBRATION_APPENDIX.md#the-sensor-fields-and-the-compression-switch)). |
| `species_flux` | `:fickian` | provisional | `:bulk` is at least as good on every measured column; the constants are fitted on the Fickian channel ([appendix](CALIBRATION_APPENDIX.md#the-bulk-species-channel)). |
| `cfl` | 0.5 | keep | Use 0.3 or `StepControl(retries = 4)` for a converging shock at a spherical origin, whose ceiling is 0.3; walls and axes carry none ([appendix](CALIBRATION_APPENDIX.md#cfl-and-the-symmetry-cell-restriction)). |
| `compact_filter` α | 0.45 | too strong | 0.49 fits at 128³ and 256³ and clears the battery; the stability edge is at 0.49875 ([appendix](CALIBRATION_APPENDIX.md#the-default-decision)). |
| `filter_cfl` | 0.35 | keep | Makes the filter's dissipation a rate, invariant to the CFL, landing steps, retries and subcycling; clears the battery at its production CFL numbers ([appendix](CALIBRATION_APPENDIX.md#the-battery-under-relaxation), [invariances](CALIBRATION_APPENDIX.md#retries-and-subcycling-under-relaxation)). |
| `filter_interval` | 1 | keep | Redundant with α ([appendix](CALIBRATION_APPENDIX.md#cadence-and-alpha-are-one-axis)). |
| `filter_weighting` | `:none` | keep | The volume-weighted form conserves no better on a closed line, is 17× less conservative at an axis or a pole, and moves the Noh wall deficit in opposite directions at the axis and the origin ([appendix](CALIBRATION_APPENDIX.md#the-filter-on-non-uniform-volumes)). |

Two facts frame the table. Every constant was fitted under
`compact_filter(0.45)` applied at full strength every step, and the four
that could depend on it have been re-swept at α = 0.49 and under
`filter_cfl = 0.35` without moving. For a strong shock the CFL number
decides stability more than any constant in the list.

## Which setting to change

Each entry is a symptom, the setting that addresses it, and the measured
effect. The appendix link carries the sweep.

- **A converging shock loses positivity at the spherical origin early in
  the run.** Lower `cfl` to 0.3 or set `StepControl(retries = 4)`. The
  failure is an excursion of the origin cell landing at t ≈ 0.39 for Noh at
  every resolution, and rollback recovers it in about half the steps of a
  fixed `cfl = 0.15`. The ceiling under the defaults is 0.3; `detector = :d8`
  lowers it to 0.25. The planar wall and the cylindrical axis carry no
  ceiling: Noh completes from `cfl = 0.9` in both, their former ceilings of
  0.25 and 0.2 having been the first step of the run, sized before any
  artificial coefficient existed, which `run!` now primes. Do not lower
  `C_beta` below 0.5 for the origin, and do not weaken the filter past
  α = 0.49. The timestep predictor, the sensor's reach and magnitude, sensor
  blindness at the fold, the fold closure, the density proportionality of β\*
  and the per-step filter strength have each been measured and are not the
  cause ([where the restriction
  originates](CALIBRATION_APPENDIX.md#where-the-restriction-originates),
  [the first step](CALIBRATION_APPENDIX.md#the-first-step-of-a-run)).
- **A resolved or smooth solution is over-dissipated.** The compact filter is
  the sink, 37% of the Taylor–Green dissipation at 128³ and 12% at 256³,
  and the artificial properties are nearly inert there. Weaken the filter
  with `compact_filter(0.49)`, which fits at both resolutions. Under the
  default `filter_cfl = 0.35` the dissipation is a rate, invariant to the
  CFL, to shortened steps, to retries and to subcycling to the digits
  printed, so lowering the CFL or writing output more often does not change
  it. Under `filter_cfl = 0` each pass dissipates a fixed amount, so more
  steps mean more dissipation
  ([the compact filter](#the-compact-filter)). `filter_interval` is
  redundant with α. Raising `C_mu` does not help: on Taylor–Green at 64³
  removing μ\* improves every estimator
  ([the μ\* controls](CALIBRATION_APPENDIX.md#the-mu-controls-at-64)).
- **Mass fractions leave [0, 1] at a shocked interface.** The bound
  `C_Y = 100` holds a 2h interface to ±0.013 where it rings to ±0.2 without
  it. Resolve the interface over 4h for 0.25% or 8h for a clean profile.
  Resolution, closures, detector, sensor field, CFL and filter strength were
  each varied and move the excursion by nothing
  ([C_Y](CALIBRATION_APPENDIX.md#c_y-the-mass-fraction-bound)). At a density
  ratio of 100 the Fickian channel fails and `species_flux = :bulk` completes
  ([the bulk channel](CALIBRATION_APPENDIX.md#the-bulk-species-channel)).
  Run under `validity = :permissive` and bound the excursion in the guard.
- **Wall heating at a stagnation wall.** Planar Noh carries a 64% density
  deficit at the wall under the defaults, and most of it is the filter's
  second row. `compact_filter(closures = :onesided)` takes it to 27%;
  `lele_d1_6(closures = :cascade4)` takes it to 58% under the default
  filter; `detector = :d8` to 53%. The two closure choices are coupled:
  `:cascade4` with the cascade filter, `:cascade3` or C6 `:brady_livescu`
  with the one-sided filter. `C_kappa` does not help under the default
  smoother, and the deficit does not converge away with resolution
  ([walls](#walls-folds-and-metrics)).
- **A smooth wave train or a contact is over-damped.** `C_beta = 0.5` keeps
  0.7% more Shu–Osher amplitude and an 18% narrower contact than 1.0, at
  the cost of half the spherical origin's timestep. `detector = :d8` keeps
  1.1% more amplitude. A weaker filter helps the train monotonically.
- **The run ends on a state the EOS rejects.** Converging shocks carry six
  to eight cells of negative internal energy for the whole run and still
  reach the exact plateau to 0.07%; repairing those cells terminates the run
  within twenty steps. Use `validity = :permissive` and the default
  `floor_scope = :representable`, and bound the count in a guard
  ([the budget](CALIBRATION_APPENDIX.md#negative-internal-energy-in-completed-runs)).
- **The ambient is cold.** κ\* is written as ρc/T_ion and is not singular in
  practice: the sound speed vanishes with the temperature at a floored cell,
  and on planar Noh the κ\* rate stays an order of magnitude below the β\*
  rate, with the step count unchanged, from p₀ = 1e-2 down to 1e-8. What a
  cold ambient does change is the count of cells the EOS calls inadmissible,
  since the precursor's negative internal energy is a fixed absolute
  amplitude
  ([the cold ambient](CALIBRATION_APPENDIX.md#the-cold-state-limit)).
- **A spherical-origin run fails within tens of steps of a sharp start.**
  The origin fold needs initial data resolved over three cells or more and
  cannot take the singular t = 0 start of Noh; warm-start from a resolved
  profile. The cylindrical axis accepts both
  ([geometry limits](CALIBRATION_APPENDIX.md#geometry-limits)).
- **Results differ across process grids.** Stay on `beta_sensor = :strain`.
  The switched forms carry a discontinuous compression switch and reproduce
  to 2e-6 relative across decompositions against 1e-14 for the strain sensor.
- **Float32 or device runs.** Keep the default closures. The Brady–Livescu
  rows floor at 1e-3 in Float32, above the cascade's 1e-4.
- **A three-dimensional CFL number looks small next to the literature.**
  `max_rate` takes the acoustic rate as the Euclidean bound
  `c · sqrt(Σ 1/h_d²)`, the linear limit on an isotropic grid;
  [Pyranda](https://github.com/LLNL/pyranda) counts the sound speed once,
  optimistic by √3, so its CFL reads as √3 times this one on such a grid.
  Taylor–Green at 32³ is stable to nominal `cfl` 1.75 to 1.9, where the bound
  predicts 1.68 to 1.85. Until September 2026 the rate was summed over
  dimensions, conservative by √3, and a three-dimensional number recorded
  before then reads as `cfl_old / √3`; the battery is one-dimensional and
  unaffected
  ([the rate convention](CALIBRATION_APPENDIX.md#the-cfl-rate-is-normalized-differently)).

## The battery and how to read it

The one-dimensional cases live in `test/cases.jl`, shared by
`bench/artcal.jl` and `test/validation.jl`, so a swept constant is measured
on the run its guard is set from. ν is the Noh geometry index: 1 planar
(a slip wall), 2 the cylindrical axis fold, 3 the spherical origin fold, with
exact post-shock compression 4^ν at γ = 5/3.

The columns and what each responds to:

- **Noh plateau over exact** (4, 16, 64): the shock-thickness constants and
  the fold. Exact is 1.0000, the most precise measure in the set.
- **Wall deficit**: the density shortfall at the symmetry point, which is
  spurious entropy deposited where the flow stagnates.
- **Lax L1 and contact width**: error against the exact Riemann solution and
  the broadening the regularization adds.
- **Shu–Osher train amplitude**: the smooth structure the high-order scheme
  exists to preserve; it pulls against every dissipative constant.
- **Woodward–Colella peak**: survival at a 10⁵ pressure ratio.
- **NaN** is positivity loss, a real limit. **Inf** is the step cap, an
  artifact of the sweep's `nmax`.

Every table is reproducible to four significant figures. A run integrates
thousands of steps through a nonlinear sensor, so any arithmetic
reassociation moves the fourth digit; a moved third digit is a real change.
The cases are one-dimensional, so they reach `C_mu` only through the stress
trace and constrain its stability bound, not its accuracy.

## The constants

### `C_beta`, the shock constant

β\* regularizes the shock. At zero every strong-shock case loses positivity;
at 2 the planar wall fails at the production CFL. Between the limits the
accuracy measures are monotone: the contact broadens 0.0041 to 0.0074 and
the Shu–Osher train loses 2.8% over 0.25 to 4, and the spherical plateau
crosses exact between 0.25 and 0.5. Accuracy alone puts the optimum near
0.4; the default 1.0 buys robustness at the origin
([sweep](CALIBRATION_APPENDIX.md#c_beta-the-shock-constant)).

The constant trades the planar and cylindrical CFL ceilings against the
spherical one, and the default is the value that maximizes the spherical
ceiling ([the CFL ladder](CALIBRATION_APPENDIX.md#the-cfl-ladder)):

```
C_beta      nu = 1     nu = 2     nu = 3
0.25         1.0+       1.0        0.2
0.5          0.4        0.4        0.25
1.0    *     0.25       0.2        0.4
2.0          none       none       0.3
```

Under `:d8` the viable window moves to 1.0–4.0 and intersects the `:delta4`
window in the single value 1.0
([the refit](CALIBRATION_APPENDIX.md#the-c_beta-refit-under-d8)). Use 0.5
for interface-dominated work with moderate shocks. Values below 0.25 or
above 2 are not recommended.

### `C_kappa`, the conductivity

κ\* transports spuriously deposited entropy out of the stagnation cell.
Under the default smoother the wall-heating trough sits at the default, and
raising the constant makes it worse: +64% at 0.01, +68% at 0.04, +91% at
0.16. Lax L1 grows 4.8e-3 to 5.8e-3 across the range because the contact is
nearly isothermal. Zero completes under `:gaussian` and fails spherical Noh
under `:compact`. Retain 0.01
([sweep](CALIBRATION_APPENDIX.md#c_kappa-the-conductivity)). The
construction is written as ρc/T_ion; measured on Noh, it neither limits the
step nor grows as the ambient cools
([the cold ambient](CALIBRATION_APPENDIX.md#the-cold-state-limit)).

### `C_mu`, the shear viscosity

In one dimension μ\* is inert to four digits across a 64× sweep, and the
only bound the battery sets is that spherical Noh fails above 0.008. On
Taylor–Green at 128³ the peak carries a 6% one-signed residual the constant
does not control, so the peak cannot select it, and at 64³ removing μ\*
improves every history estimator. Retain 0.002 as consistent with the case,
not determined by it; a fit needs a case with an unresolved cascade
([Taylor–Green](CALIBRATION_APPENDIX.md#taylorgreen),
[controls](CALIBRATION_APPENDIX.md#the-mu-controls-at-64)). Two consequences
for anyone refitting it: the peak must be read through a window and the
truncated final step excluded
([the window](CALIBRATION_APPENDIX.md#read-the-rate-over-a-window)), and the
μ\* share of the sink depends on the CFL under `filter_cfl = 0`, so a refit
has to state its CFL.

### `C_D`, the species diffusivity

A passive interface more than doubles in width with D\* off, so the filter
is the broadening; a 64× sweep in `C_D` moves the width 22%. At two species
the per-species machinery is a measurable no-op. Retain 0.01, and consider
larger values only with three or more species
([sweep](CALIBRATION_APPENDIX.md#c_d-the-species-diffusivity)).

### `C_Y`, the mass-fraction bound

The species diffusivity carries a second term, the bound of Shankar, Kawai
and Lele, which is zero wherever 0 ≤ Y ≤ 1:

    D*_k = c · G[ max( C_D h |δ⁴Y_k| , C_Y h max(0, −Y_k, Y_k − 1) ) ]

Cook's D\* alone cannot hold an interface a shock has thinned. On a Mach 1.5
air/SF6 interface:

```
initial width   C_Y = 0             C_Y = 100
2h              -0.204 / +1.204     -0.013 / +1.013
4h              -0.023 / +1.023     -0.0026 / +1.0026
8h              clean               clean
```

The residual at 2h is the compact scheme's dispersion at a three-cell
contact. Sum and max combination, and one or two smoothing passes, are
indistinguishable; the unsmoothed bound triples the step count. The bound
stays in the diffusive timestep. `Y_tolerance = 1e-4` keeps it inert on a
smooth profile that touches 0 or 1, where it otherwise costs a convergence
order. Retain 100 and 1e-4
([sweep](CALIBRATION_APPENDIX.md#c_y-the-mass-fraction-bound),
[dead band](CALIBRATION_APPENDIX.md#the-dead-band)).

## The sensor options

### The smoother

`smoother = :gaussian` is Pyranda's nine-point explicit stencil;
`:compact` is one pass of the state filter, near the identity across the
resolved band and a distributed line solve per sensor. The Gaussian raises
the spherical-origin ceiling from 0.15 to 0.4 and the axis from 0.15 to
0.2, cuts the sensor phase 29%, and costs seven points of planar wall
heating that `C_kappa` cannot buy back. The mechanism is sensor
intermittency at the damaged cell: under a near-identity smoother the cell
alternates between spikes and troughs of β\*; the Gaussian leaves no trough
([smoother](CALIBRATION_APPENDIX.md#the-sensor-smoother),
[intermittency](CALIBRATION_APPENDIX.md#sensor-intermittency-at-the-damage-site)).

### The detector

`detector = :delta4` is Cook's undivided fourth difference; `:d8` is
Pyranda's compact eighth-derivative operator, normalized so both respond
equally to a two-point oscillation and diverge below it (569× at eight
points per wavelength). `:d8` improves six of seven battery columns, the
spherical plateau error 2.45% → 0.42% and the Shu–Osher train +1.1%, and
costs 3.2% of the Woodward–Colella peak and +19% on the right-hand side.

```
Noh ceiling               :delta4    :d8
nu = 1  planar wall          0.2      1.0+
nu = 2  cylindrical axis     0.2      1.0+
nu = 3  spherical origin     0.4      0.25
```

`:delta4` stays the default only because the general guidance for
converging shocks rests on the spherical case; a user who does not know which
geometry is ahead is better off under `:d8`. The selectivity is available to
κ\* and D\* and largely unavailable to μ\* and β\*, whose input |S| has cusps
at every resolution
([detector](CALIBRATION_APPENDIX.md#the-ringing-detector)).

### The sensor fields and the reduction

`mu_sensor = :velocity` (Pyranda's field) moves no battery column past the
fourth digit and costs +46% on the sensor phase; on Taylor–Green it raises
the μ\* share by a third, so it cannot be evaluated until `C_mu` is refitted
under it. `beta_sensor = :dilatation` loses both converging geometries at
every CFL through the fold cells, where Δ adds two same-signed components
that |S| partially cancels; `:ungated_dilatation` keeps the origin and loses
the axis; `:gated_strain` raises the axis ceiling and changes nothing
elsewhere. `reduction = :max` cuts the μ\* share 4.5% → 2.7% on Taylor–Green
and lands the peak time on the reference's. Retain `:strain`, `:strain`,
`:sum`
([fields](CALIBRATION_APPENDIX.md#the-sensor-fields-and-the-compression-switch)).

### Directional bulk viscosity

Measured and not adopted. One β\* per grid direction, in that direction's
normal stress with its own step limit, removes the aspect-ratio penalty
from the step (4.5× fewer steps at AR 4 and 15× at AR 16 on a planar Noh
run along the coarse direction) and fails on a curved front from AR 3 on:
with unequal coefficients the bulk force is not a gradient, and in a cold
irrotational pre-shock flow it makes vorticity until the flow cavitates.
Neither the sensor field nor the compression switch nor the ambient
pressure changes the outcome
([measurement](CALIBRATION_APPENDIX.md#directional-bulk-viscosity-on-anisotropic-grids)).
The scalar form's cost on such a grid is the step, and through the relaxed
filter the dissipation as well (open item 10).
The coefficient definitions and the package's starting values come from
[Cook (2007)](https://doi.org/10.1063/1.2728937), while Cook's later
[2009 formulation](https://doi.org/10.1063/1.3139305) supplies the dilatation
sensor. The smaller parameter set reported by
[Brill, Olson & Bokman (2025)](https://arxiv.org/abs/2503.12680) runs through the
battery under `bench/artcal.jl brill2025` and survives both converging geometries;
those later values belong to their own scaling and are not the package's
([Brill 2025 parameter set](CALIBRATION_APPENDIX.md#brill-2025-parameter-set)).

## The compact filter

The state filter is the stabilizer and the main energy sink of every run,
and it had never been calibrated. Its strength is set by α: a pass departs
from the identity in proportion to (1 − 2α), so 0.45 is a strong filter and
0.499 nearly the identity. Three facts govern its use.

**One axis.** The dissipation per unit time is set by
ε = (1 − 2α) · w / `filter_interval`, with w the relaxation weight below.
Points at different α and cadence collapse onto that axis on Taylor–Green,
and the converging Noh cases fail at the same ε in either formulation.
Fit α at `filter_interval = 1` and leave the cadence alone
([cadence](CALIBRATION_APPENDIX.md#cadence-and-alpha-are-one-axis),
[edge](CALIBRATION_APPENDIX.md#the-stability-edge)).

**The fit and the edge.** Against the vendored 512³ Taylor–Green reference,
α = 0.45 is too strong at 128³ and at 256³; α = 0.49 minimizes the history
misfit at 128³, and at 256³ the misfit keeps falling to 0.499 because the
dissipation range is resolved and nothing needs stabilizing. The upper
bound comes from the shock battery instead: spherical Noh loses positivity
between ε = 0.00125 and 0.0025, which is α = 0.49875 at full strength. The
256³ leg's best point sits on that edge. Every battery case completes
through α = 0.49 and no CFL ceiling moves
([128³](CALIBRATION_APPENDIX.md#the-alpha-sweep-at-128),
[256³](CALIBRATION_APPENDIX.md#the-256-confirmation),
[battery](CALIBRATION_APPENDIX.md#the-battery-under-alpha)).

```
                       rate misfit 128³   rate misfit 256³   battery
alphaf 0.45 (default)    5.67e-2            1.23e-2            completes
alphaf 0.49              2.53e-2            0.86e-2            completes
alphaf 0.499             2.91e-2            0.75e-2            completes (ε = 0.002)
```

**Per application or per unit time.** Under `filter_cfl = 0` each pass
replaces the state by its filtered image, so the dissipation is per
application: halving the CFL doubles it, retries and shortened output
steps add it, and a run at a low CFL is a more filtered run. A positive
`filter_cfl` relaxes each pass along direction d by
w_d = `filter_interval` · dt · r_d · √n / `filter_cfl`, capped at one, with
r_d the largest one-dimensional hyperbolic rate (|u_d| + c)/h_d of that
direction and n the number of active dimensions, and the dissipation per
unit time is then constant to six figures across a fourfold CFL change and
invariant to landing steps, to induced retries and to a subcycled refined
level. The rate is the hyperbolic one, not the maximum that sized the
step, so a diffusion-limited step, physical or artificial, filters no more
per unit time than an acoustic-limited one, and it is the direction's own,
so a fine spacing in one direction leaves the passes along the others
unchanged. On an isotropic grid at `cfl = 0.35` and `filter_cfl = 0.35`
a pass is at full strength up to the advective share of the rate; the
Taylor–Green fits at 32³ and 64³ with the artificial properties off moved
by 0.06% and 0.5% under the change, and with them on, where the step is
artificial-diffusion-limited for part of the run, they improved by 6% and
12%
([per application](CALIBRATION_APPENDIX.md#dissipation-per-application),
[relaxation](CALIBRATION_APPENDIX.md#the-relaxation-leg),
[invariances](CALIBRATION_APPENDIX.md#retries-and-subcycling-under-relaxation),
[the directional rate](CALIBRATION_APPENDIX.md#the-filter-relaxed-against-the-directional-acoustic-rate)).
Under relaxation a retry halves ε, so retries walk a case toward the edge
rather than away from it.

**The default.** `filter_cfl = 0.35` at α = 0.45 is the default since
September 2026, with `compact_filter(0.49)` the per-run selection for
resolved smooth turbulence. At the reference CFL nothing changed; below it
the strength falls with the CFL to a value the battery reads as its
α = 0.486 row, with a margin of seventeen times the edge at `cfl = 0.15` and
eight after one retry, where α = 0.49 relaxed would reach the edge on the
second. `C_beta`, `C_kappa`, `C_D` and `C_Y` do not move under it
([constants](CALIBRATION_APPENDIX.md#the-constants-under-a-weaker-filter),
[decision](CALIBRATION_APPENDIX.md#the-default-decision)). The pins in
`test/cases.jl` moved with `Numerics`, so the validation guards measure the
default configuration: the Woodward–Colella and Noh rows moved, and Lax,
Shu–Osher, Sedov and the interface case did not to the digits printed
([the battery under relaxation](CALIBRATION_APPENDIX.md#the-battery-under-relaxation)).
When the weight began reading the directional hyperbolic rate, every Noh
row moved again toward a weaker filter at the front, where the step is
diffusion-limited under C_β = 1, and the aligned case reads one profile at
every aspect ratio
([the directional rate](CALIBRATION_APPENDIX.md#the-filter-relaxed-against-the-directional-acoustic-rate)).

The filter is also the wall-order cap of every filtered run, second order
through its row-2 closure, and the closure rows are where it fails to
conserve: a few percent of the first rows' content per pass on any grid,
decaying inward at the tridiagonal root. The volume weighting of the
reference implementation does not change that and is not the default
([walls](#walls-folds-and-metrics)).

## CFL, step control and the symmetry cell

A converging strong shock at the spherical origin does not survive the
default `cfl = 0.5`. The restriction is an excursion of the origin cell, not
the shock front: the cell is quiescent until t ≈ 0.39 on Noh at every
resolution, then passes through an excursion in which β\* at the origin
reaches the line maximum, and the ceiling is whether the cell survives it
([the origin cell](CALIBRATION_APPENDIX.md#the-origin-cell-is-a-startup-transient)).
Scaling β\* by the smoothed density in place of the local one, so that the
evacuating cell does not suppress its own regularization, and the per-step
filter strength from unrelaxed to twice relaxed both leave the ceiling where
it is ([the first step](CALIBRATION_APPENDIX.md#the-first-step-of-a-run)).

The planar wall and the cylindrical axis have no ceiling. Their recorded
ones were the first step of the run: `max_rate` reads the artificial
coefficient arrays the previous evaluation left, and a fresh solver had
none, so the first step from Noh's u = −1 against the wall or axis was sized
on the acoustic rate alone. `run!` now evaluates the right-hand side once
before its first step, and both geometries complete from `cfl = 0.9` with
the plateau of the `cfl = 0.15` run.

Ceilings, the highest CFL reaching the end with a correct plateau, under
the priming:

```
                     nu = 1 wall   nu = 2 axis   nu = 3 origin
default              none to 0.9   none to 0.9   0.3
detector = :d8       none to 1.0   none to 1.0   0.25
```

The `:d8` row was measured without the priming and reproduces its earlier
record; under it the planar wall deficit falls from 51% at 0.2 to 23% at
1.0. The origin ladder samples 0.15, 0.2, 0.25, 0.3, 0.4 and 0.5 and reads
the same under the unrelaxed filter, the default and `filter_cfl = 0.7`; the
warm-started axis (t₀ = 0.3) completes through 0.5 with an exact shock
position.

`StepControl(retries = 4)` recovers spherical Noh from an initial
`cfl = 0.9` in about half the steps of a fixed `cfl = 0.15`, rolling back
twice through the excursion to 0.225, and leaves the accepted value in
`solver.cfl` ([recovery](CALIBRATION_APPENDIX.md#recovery-strategy)).

Recommendation: the default `cfl = 0.5` with `StepControl(retries = 4)`,
or `cfl = 0.3` for a converging shock at a spherical origin. Converging
shocks under the `:d8` ladder show cells where the axis fails below a CFL
rather than above one; that is the per-application filter and not a CFL
restriction
([a failure that gets worse](CALIBRATION_APPENDIX.md#a-failure-that-gets-worse-as-the-timestep-falls)).

## Walls, folds and metrics

**Wall closures.** The filter's cascade rows 2–4 leave a second-order error
along the whole line and deposit an O(h²) disturbance two cells from the
wall on every step; `compact_filter(closures = :onesided)` makes one pass
eighth order everywhere and takes the planar Noh wall deficit from 64% to
27%. The derivative closures `:brady_livescu` raise the smooth wall order
from 3.17 to 5.88 (C6) and 7.91 (C8) but fail at a shock-bounded wall under
the cascade filter through a β\*-driven wall mode; with the one-sided filter
C6 completes and C8 does not, and `:cascade4` needs the F2 row it removes.
The coupling rule: `:cascade4` with the cascade filter, `:cascade3` or C6
`:brady_livescu` with the one-sided filter. In Float32 the Brady–Livescu
rows floor at 1e-3. All of it stays optional because every constant and
guard is set under the cascade
([wall cascade](CALIBRATION_APPENDIX.md#the-filters-wall-cascade),
[wall closures](CALIBRATION_APPENDIX.md#wall-closures-under-the-artificial-properties)).

**Folds.** The axis and origin folds converge at sixth to seventh order and
are the most accurate region of the line; every global error in
`test/convergence.jl` is the outer wall's. The origin needs initial data
over three cells or more and cannot take the singular Noh start; why it is
less forgiving than the axis is open
([fold order](CALIBRATION_APPENDIX.md#fold-order-and-geometry-limits)).

**Grid convergence.** Lax L1 halves per doubling, interface width halves
per doubling, and wall heating does not converge away (60% → 56% over 8×),
which is the character of the Noh problem
([grid convergence](CALIBRATION_APPENDIX.md#grid-convergence)).

**Non-Cartesian metrics.** `filter_state!` filters the conserved components
unweighted, and that is the better of the two forms measured. A pass
conserves the volume integral of a component exactly when the transpose of
its line operator fixes the quadrature volumes, which a periodic line does
to round-off and a closed line does not: the cascade rows create or destroy
2–4% of the first three rows' content, the defect decays inward at 0.627 per
row at α = 0.45, the same on a uniform, a clustered, a cylindrical and a
spherical line, and the one-sided rows move it to rows 3–7. The
volume-weighted form of the reference, F(Jq)/F(J), behind
`filter_weighting = :volume`, leaves the wall rows as they are, conserves at
the spherical origin as the unweighted form does, and is 17× less
conservative at the cylindrical axis and the spherical poles, where the odd-parity
fold of the product does not have unit column sums. Both hold a uniform state
exactly on every metric. On Noh it lowers the axis wall deficit from 55% to
45% and raises the origin's from 28% to 46%; the filter's mass defect over a
whole run is 1e-3 on the planar case and below 2e-5 on the curved ones under
either form
([non-uniform volumes](CALIBRATION_APPENDIX.md#the-filter-on-non-uniform-volumes)).

## The species channel

`species_flux = :bulk` replaces the Fickian species flux by one diffusive
flux on every conserved variable, sensed on the maximum of mass and mole
fraction. It removes the Fickian enthalpy flux's pressure error at unequal
gas constants (1e-2 → 1e-10 on the Brill slab), completes the shocked
interface at density ratio 100 where the Fickian channel fails, and
reproduces the Fickian results to seven digits at equal molecular weights,
for four more gradient line solves per direction. It is not the default
because the constants are fitted on the Fickian channel and the
vortex-ring/SF6 case has not been run under it
([bulk channel](CALIBRATION_APPENDIX.md#the-bulk-species-channel)).

## Open items

In approximate priority order; each links to the measurements it rests on.

1. **Raise the CFL ceiling at the spherical origin.** The density
   proportionality of β\* and the per-step filter strength are measured
   and closed; the excursion itself is the remaining object
   ([origin cell](CALIBRATION_APPENDIX.md#the-origin-cell-is-a-startup-transient),
   [the first step](CALIBRATION_APPENDIX.md#the-first-step-of-a-run)).
2. **Refit `C_mu` on the history misfit** on a case with an unresolved
   cascade; the peak is unusable and 0.004 is withdrawn
   ([Taylor–Green](CALIBRATION_APPENDIX.md#taylorgreen)).
3. **Decide the detector**; waits on item 1
   ([recommendation](CALIBRATION_APPENDIX.md#recommendation)).
4. **Refit `C_mu` under `:gaussian`** and then evaluate
   `mu_sensor = :velocity`.
5. **Explain the spherical fold's intolerance of sharp data**
   ([geometry limits](CALIBRATION_APPENDIX.md#geometry-limits)).
6. **Make `filter_state!` conservative on non-Cartesian metrics.**
7. **Decide the filter wall rows**, a recalibration of the wall cases
   ([wall cascade](CALIBRATION_APPENDIX.md#the-filters-wall-cascade)).
8. **Put `delta4_sum!`'s even path on the half-offset mirror**; unmeasured
   ([the clamp](CALIBRATION_APPENDIX.md#the-fourth-difference-clamp-at-a-fold)).
9. **Decide `species_flux`** on the vortex-ring/SF6 case
   ([open](CALIBRATION_APPENDIX.md#open)).
