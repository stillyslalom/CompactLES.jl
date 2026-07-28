# Calibrating the artificial fluid properties

This study measures the effects and failure limits of the four `ArtParams`
constants. `bench/artcal.jl` reproduces the results using the cases defined in
`test/cases.jl` and guarded by `test/validation.jl`.

The results support the shipped defaults. For strong shocks, however, the CFL
number determines stability more strongly than any of the four constants.

```
ArtParams(C_mu = 0.002, C_beta = 1.0, C_kappa = 0.01, C_D = 0.01)
```

## Contents

1. [How to read the tables](#how-to-read-the-tables)
2. [C_beta — the shock constant](#c_beta--the-shock-constant)
3. [C_kappa — the conductivity](#c_kappa--the-conductivity)
4. [C_mu — the shear viscosity](#c_mu--the-shear-viscosity)
5. [C_D — the species diffusivity](#c_d--the-species-diffusivity)
6. [CFL, which dominates all four](#cfl-which-dominates-all-four)
7. [Grid convergence](#grid-convergence)
8. [Geometry limits](#geometry-limits)
9. [Recommendations](#recommendations)

## How to read the tables

All entries in the four constant tables are from one-dimensional calculations.
Because `C_mu` multiplies the shear component of the artificial stress, these
cases exercise it only through the stress trace. The `C_mu` table therefore
constrains stability but not shear accuracy. The cases fully exercise `C_beta`,
`C_kappa`, and `C_D`.

The [`C_mu` section](#c_mu--the-shear-viscosity) also contains a separate
three-dimensional Taylor–Green study with distinct estimator limitations.

Columns:

- **Noh plat/exact** — post-shock density plateau over its exact value
  (4, 16, 64 for ν = 1, 2, 3). The exact ratio is 1.0000, making this the most
  precise accuracy measure in the set.
- **deficit** — density shortfall at the symmetry point relative to the exact
  plateau. This is *wall heating*: spurious entropy deposited where the flow
  stagnates, showing up as too-hot, too-rarefied gas in the first cell.
- **Lax L1 / contact** — mean density error against the exact Riemann solution,
  and the 10–90% width of the contact discontinuity. The contact width is the
  broadening introduced by regularization.
- **Shu train amp** — peak-to-trough density in the post-shock entropy wave
  train. This is the quantity the tenth-order interior scheme exists to
  preserve, and it pulls against every constant that adds dissipation.
- **WC peak** — peak density in the Woodward–Colella collision, mostly a
  survival check at a 10⁵ pressure ratio.
- **NaN** — the calculation lost positivity or stalled. The diffusive rate in
  `compute_dt` increases until `dt` collapses. See the `NMAX` note in
  `test/cases.jl`. `StepControl` now raises a `SolverFailure` in `run!`, detecting
  the positivity loss approximately 150 steps before the timestep collapse.

## C_beta — the shock constant

```
C_beta    | Noh1 plat  deficit | Noh3 plat | Lax L1  contact | Shu train | WC peak
0         |     NaN       NaN  |      NaN  | 5.7e-3   0.0032 |    1.6638 |    NaN
0.25      |  0.9993       +54% |   1.0122  | 4.9e-3   0.0041 |    1.6406 | 6.4556
0.5       |  0.9992       +57% |   0.9953  | 4.9e-3   0.0045 |    1.6310 | 6.4702
1.0    *  |  0.9992       +58% |   0.9732  | 5.0e-3   0.0051 |    1.6185 | 6.5731
2.0       |     NaN       NaN  |   0.9516  | 5.2e-3   0.0062 |    1.6155 | 6.7221
4.0       |     NaN       NaN  |      NaN  | 6.2e-3   0.0074 |    1.5859 | 6.7711
```

The lower and upper ends of the sampled range fail for different reasons.

At `C_beta = 0` there is no shock regularization, and every strong-shock
case loses positivity: Noh in both geometries and Woodward–Colella. The Lax tube
survives because the compact filter alone is sufficient at this shock strength.
Its contact is the narrowest in the table (0.0032) because artificial bulk
viscosity does not broaden it, but the setting is not viable for stronger
shocks.

At the upper end, doubling `C_beta` to 2 improves the Woodward peak and produces
only small changes in the other measures, but Noh ν = 1 does not complete. The
upper bound is therefore a stability bound
rather than an accuracy one, and it moves with the CFL, which is what the
[CFL table](#cfl-which-dominates-all-four) shows. Tests described in that section
exclude timestep lag as the cause.

Between the two limits, the accuracy measures vary monotonically:

- Contact width grows roughly linearly, 0.0041 → 0.0074 over 0.25 → 4.
- The Shu–Osher wave train loses about 3.5% of its amplitude over the same
  range. Because the entropy waves are smooth, this reduction measures
  overdamping relevant to a mixing calculation.
- The Noh ν = 3 plateau *degrades* with increasing `C_beta` (1.0122 → 0.9516),
  crossing exact between 0.25 and 0.5.

Accuracy alone gives an optimum near `C_beta = 0.4`. The shipped value of 1.0
instead provides a robustness margin: Woodward–Colella and both Noh geometries
complete, at the cost of a 13% wider contact and 0.8% lower wave-train amplitude
relative to 0.5.

**Recommendation:** retain 1.0 as the default. Use 0.5 for interface-dominated
Richtmyer–Meshkov or Rayleigh–Taylor calculations with moderate shocks. Values
below 0.25 or above 2 are not recommended.

## C_kappa — the conductivity

```
C_kappa   | Noh1 plat  deficit | Noh3 plat | Lax L1  contact | WC peak
0         |  0.9998       +64% |      NaN  | 4.8e-3   0.0051 | 6.6847
0.0025    |  0.9997       +61% |   0.9728  | 4.9e-3   0.0051 | 6.6390
0.01   *  |  0.9992       +58% |   0.9732  | 5.0e-3   0.0051 | 6.5731
0.04      |  0.9980       +57% |   0.9748  | 5.3e-3   0.0052 | 6.4934
0.16      |  0.9986       +70% |   0.9796  | 5.8e-3   0.0052 | 6.3176
```

The table separates two effects of κ\*.

**Wall heating.** The deficit falls from 64% to 57% as `C_kappa` goes from 0 to
0.04, then rises to 70% at 0.16. Artificial conduction transports spuriously
deposited entropy out of the stagnation cell, but excessive conduction increases
the error. The broad minimum occurs near 0.04.

**Robustness.** Spherical Noh does not complete with `C_kappa = 0`. Although
`C_beta` regularizes the momentum equation, the energy equation therefore
requires separate regularization for a converging strong shock.

Lax L1 grows from 4.8e-3 to 5.8e-3 across the range, while the contact width
changes from 0.0051 to 0.0052 because κ\* diffuses temperature,
not composition, and the Lax contact is nearly isothermal.

**Recommendation:** retain 0.01 as a conservative default. Values of 0.02–0.04
reduce wall heating while increasing shock-tube L1 error by approximately 5%.
Zero is not recommended.

The κ\* formulation has an additional low-temperature limitation. It is built as
`C_kappa · (ρc/T_ion) · sensor`, which is singular as T_ion → 0. In a sufficiently
cold ambient (p₀ ≲ 10⁻³ at ρ₀ = 1) the 1/T factor drives the diffusive rate up,
`dt` collapses, and internal energy can become negative. The subsequent T_ion
clamp then produces an extremely large κ\*. Each case therefore uses a finite
ambient pressure. The scale is an EOS dispatch point
(`art_conductivity_scale`), so a tabular or condensed-matter model can supply
one that is finite at its own cold limit; the gas models still divide by the
temperature.

## C_mu — the shear viscosity

```
C_mu      | Noh1 plat  deficit | Noh3 plat | Lax L1  contact | Shu train
0         |  0.9992       +58% |   0.9731  | 5.0e-3   0.0051 |    1.6183
0.0005    |  0.9992       +58% |   0.9732  | 5.0e-3   0.0051 |    1.6182
0.002  *  |  0.9992       +58% |   0.9732  | 5.0e-3   0.0051 |    1.6185
0.008     |  0.9992       +58% |      NaN  | 5.0e-3   0.0052 |    1.6214
0.032     |  0.9992       +58% |      NaN  | 5.0e-3   0.0052 |    1.6206
```

In one dimension μ\* is nearly inert: every accuracy column is flat to four
digits across a 64× sweep, because the shear artificial viscosity only reaches
the solution through the trace of the stress, where β\* already dominates by a
factor of 500.

The table establishes that **`C_mu` above approximately 0.008 destabilizes the
spherical origin**, consistent with the upper-end stability limit observed for
`C_beta`. This is a stability ceiling; the table does not establish an accuracy
lower bound.

The one-dimensional battery does not constrain `C_mu` for damping under-resolved
shear in a turbulent mixing layer. That calibration requires a three-dimensional
case with resolved shear.

### Taylor–Green calibration

`bench/tgv_energy.jl` runs TGV at Re = 1600 and splits −dKE/dt by which mechanism
removes the energy: molecular stress, artificial shear μ\*, artificial bulk β\*,
and the residual, which is the compact filter plus numerical loss. At 32³ with
`filter_interval = 1` and the default `C_mu`, at the dissipation peak (t ≈ 6.3):

```
channel              share of -dKE/dt
molecular                    12%
artificial shear  μ*          5%
artificial bulk   β*        ~ 0%
compact filter (residual)    83%
```

Peak −dKE/dt is 1.46e-2 at t = 6.5 against the reference 1.2e-2 at t = 9 —
an early overprediction dominated by the filter contribution. Two findings
follow:

- **μ\* contributes to this case** — it reaches 28% of the *viscous* budget, so
  TGV exercises the shear-viscosity parameter absent from the one-dimensional
  cases. However, `C_mu` cannot be
  *fitted* against the dissipation curve while the filter owns 83% of the total
  sink; that would be fitting a coefficient to a residual. And the filter cannot
  be removed to isolate it: at 32³, `filter_interval = 4` diverges and
  `filter_interval = 0` fails outright with `SolverFailure(:negative_density)` at
  t = 5.32. **The compact filter is the primary stabilizer at this resolution,
  not an auxiliary smoother.** `C_mu` and the filter set the subgrid dissipation
  jointly and must be calibrated together. At 128³, the filter share decreases
  to 37%, permitting `C_mu` to be fitted while holding the filter fixed. The
  limitation therefore applies to the 32³ resolution rather than to TGV itself.
- β\* is four orders of magnitude smaller than μ\* despite `C_beta = 1.0`
  because dilatation is negligible at Ma 0.1. This agrees with the role of
  `C_beta` as a shock parameter.

### Results at 128³ on a cluster

The following results test whether the filter dominance measured at 32³ persists
at higher resolution. They use 128³, Re = 1600, t = 10, and 224 ranks over two
rzhound nodes; each configuration required approximately 20–25 minutes:

```
config              peak -dKE/dt        mol    mu*   filter   steps   wall
art ON,  filter 1   1.2044e-2 @ t≈9.06  60.4%  2.3%   37.3%   12739   1520 s
art OFF, filter 1   1.2153e-2 @ t≈9.00  62.5%  0      37.5%   10830   1063 s
art ON,  filter 0   SolverFailure(:negative_density) at t = 4.66, step 8515
```

Peaks are windowed rates; see [Read the rate over a
window](#read-the-rate-over-a-window-not-instantaneously). Windowing is required
when `art` is enabled. Channel shares are taken from the sampled table.

The filter share decreases from 87% at 16³ and 83% at 32³ to 37% at 128³. The
molecular contribution increases correspondingly from 12% to 60%.

The peak −dKE/dt is 1.2065e-2 at t ≈ 8.8, compared with the reference 1.2e-2 at
t = 9. The magnitude differs by less than 1%, and the time lies within one
sample interval. At 32³, the same case overpredicted dissipation by 22% and
peaked 2.5 time units early.

The filter remains required, and its stabilizing role is not proportional to
its share of the energy budget. At 128³ it supplies 37% of the sink, and
removing it kills the run *earlier* than at 32³ — t = 4.66 against 5.32. The
failure is a clean energy blow-up, not the dispersive-undershoot signature of
the shock cases: KE tracks the filtered run to within 0.1% until t ≈ 4.4, turns
upward, and triples before positivity goes, with `dt` collapsing to 2e-60. TGV is
unforced, so rising KE is unambiguously numerical. The filter owns essentially
100% of the grid-scale sink whatever its share of the total. Meanwhile
art-off/filter-on configuration reaches t = 10. The filter is therefore
necessary and sufficient for stability at this resolution, whereas the Cook
properties are neither. Whether this remains true at 256³ requires measurement.

**Refinement does not disentangle μ\* from the filter.** Their ratio is
essentially invariant across the 4× refinement:

```
32³ :  mu* 5.0%  / filter 83.0%  = 0.060
128³:  mu* 2.3%  / filter 37.3%  = 0.062
```

Both contributions decrease together. Refinement alone therefore does not
isolate `C_mu`: the μ\* contribution becomes small as the filter contribution
decreases.

The calibration instead holds the filter fixed and evaluates the dissipation
rate over a time window:

```
C_mu = 0       (art off)   1.2153e-2 @ t ≈ 9.00    +1.28% vs reference
C_mu = 0.0005              1.2108e-2 @ t ≈ 9.05    +0.90%
C_mu = 0.002   (default)   1.2044e-2 @ t ≈ 9.06    +0.37%
C_mu = 0.008               1.1950e-2 @ t ≈ 8.77    −0.42%
```

Across the 16-fold range, the peak varies monotonically and crosses the
reference value. The μ\* calculation increases wall time by 43% (21% per step from
`compute_artificial!`, plus 18% more steps). Channel shares at the peak scale as
expected: μ\* is 2.3% of the sink at `C_mu = 0.002` and 8.2% at 0.008, while
the filter share changes from 37.2% to 33.1%.

Interpolating the last interval gives 0.0048 linearly and 0.0038 in log — call
the optimum **`C_mu ≈ 0.004`**.

The uncertainty band includes the shipped default. The slope near the crossing
is −1.57e-2 per unit `C_mu`, and the
peak estimate carries ~0.4% uncertainty — at `C_mu = 0.002` the top two windowed
values are 0.011966 and 0.012044, 0.65% apart, so where the window falls relative
to the true maximum matters. That is ±0.002–0.003 in `C_mu`: the band runs
roughly 0.002–0.007 and 0.002 sits inside it.

The upper end of the band is excluded by the one-dimensional results: Noh ν = 3
does not complete at `C_mu = 0.008`. The intermediate value 0.004 has not been
tested between the 0.002 and 0.008 samples. A change to the default therefore
requires an additional `bench/artcal.jl` run at 0.004 to verify the spherical
origin case.

The existing data can narrow the band through a better peak estimator. Because
`kes` is recorded every step, a parabolic fit to the windowed maximum would
reduce the ±0.4% uncertainty associated with the discrete argmax.

### Read the rate over a window, not instantaneously

The rate must be evaluated over a window. The filter removes energy per
application, so the instantaneous rate is
`(filter loss)/dt + physical` and carries the full step-to-step `dt` jitter
divided into it. With `art` on, the sensor feeds `compute_dt` and `dt` swings
±12% step to step at 128³; against a filter supplying ~37% of the sink that
predicts ∓4.4% on the total, and ∓4% is exactly the scatter the one-step column
showed. With `art` disabled, `dt` and the rate vary by less than one percent;
the artifact therefore appeared only when the artificial properties were varied.

`C_mu` is being ranked on differences well under 1%. The one-step rate cannot
resolve that; the 501-step windowed rate can, and reduces the within-run scatter
to ~0.3%. `bench/tgv_energy.jl` now reports every rate windowed (`window=`), and
the numbers above are windowed. **Numbers quoted from output before that change
are contaminated for any `art`-on configuration.**

A boxcar over a curved peak reads ~0.2% low at this window width. That is
common-mode across configurations, so it cancels in the `C_mu` comparison and
only biases the absolute value against the external reference.

Two qualifications apply. Fitting one scalar with one parameter always succeeds, so an
optimum near 0.002 would show the default is *consistent* with TGV, not derived
from it; the stronger test is the whole −dKE/dt(t) curve, which these runs
already produce at ~37 points but which needs the van Rees reference curve
digitized to compare against. And `filter_interval` and the filter's α are
themselves uncalibrated, so any `C_mu` fitted this way is conditional on
`compact_filter(0.45)` applied every step.

**Endpoint treatment.** `run!` truncates the final step to end exactly at
`tfinal`, and the compact filter removes energy per *application* rather than per
unit time — so a short step still pays a full filter pass and −dKE/dt inflates.
At 128³ this produced a spurious 1.4226e-2 at t = 10.00 against the true
1.2065e-2 at t = 8.77, on both filtered configurations. `bench/tgv_energy.jl`
now excludes that step; output from before that fix reports the peak ~18% high.

**Recommendation: retain 0.002.** TGV at 128³ with resolved shear brackets the
reference across a 16-fold sweep and gives an optimum of
`C_mu ≈ 0.004 ± 0.003`, a band that contains the shipped value. The resolved
three-dimensional shear case constrains the coefficient to within a factor of
two and is consistent with 0.002.

Do not raise it past 0.008 — Noh ν = 3 does not complete there.

## C_D — the species diffusivity

Measured on a sharp binary interface advected at u = 1 for t = 0.5 on 256
points, so the initial 10–90% width is 2h = 0.0078:

```
C_D       | interface width
0         |  0.01785
0.0025    |  0.01794
0.01   *  |  0.01820
0.04      |  0.01922
0.16      |  0.02171
```

With D\* disabled, the interface still more than doubles in width. The compact
filter is therefore the dominant source of broadening for this passive
interface. Across a 64-fold sweep in `C_D`, the width changes by 22%.

`C_D` consequently has weak influence in this case. Interface width is more
sensitive to `filter_interval` and the filter parameter α; the default is
`compact_filter(0.45)`, and increasing α weakens the filter.

This also sharpens the note in `CLAUDE.md` about the per-species sensor being a
measurable no-op at `n_species == 2`: not only do `D*_1` and `D*_2` agree to
4.8e-16 there, the whole term is a minor contributor to what it controls.

**Recommendation:** retain 0.01. Consider larger values only for mixtures with
at least three species, where the correction velocity can alter the species
fluxes.

## CFL, which dominates all four

```
cfl       | Noh1 plat  deficit | Noh2 plat | Noh3 plat | WC peak
0.4       |     NaN       NaN  |      NaN  |      NaN  | 6.5762
0.3       |     NaN       NaN  |      NaN  |      NaN  | 6.5731
0.2       |  0.9993       +55% |      NaN  |      NaN  | 6.5597
0.15      |  0.9992       +58% |   0.9355  |   0.9732  | 6.5407
0.1       |  0.9995       +58% |   0.9345  |   0.9722  | 6.5138
```

No setting of the four constants stabilizes a converging strong shock at the
default `cfl = 0.5`, whereas every sampled setting works at 0.15. Accuracy at
0.15 and 0.1 is identical to three digits, so the smaller step changes wall time
without a measured accuracy benefit.

### Failure mechanism

An earlier version of this document asserted that the cause was the lag in
`compute_dt`, which builds its diffusive rate from the previous step's
artificial coefficients: at a forming shock β\* grows by an order of magnitude
in a few steps, so the step is chosen from coefficients that are already stale.
The lag is present but does not cause the observed failure.

Linear extrapolation of the rate was tested with
`StepControl(predict = n)` over several lookahead values. Noh at ν = 1, N = 400,
cfl = 0.3 fails at **step 175** with no lookahead, **179** with three steps of
it, and **200** with thirty. Capping the growth of `dt` (`max_growth = 1.05`)
moves it to 186. These changes delay but do not prevent the failure.

The trace identifies the preceding loss of density:

```
step   25  dt=2.1e-4  rate=1418   rho_min=0.951
step   75  dt=1.8e-4  rate=1633   rho_min=0.798
step  125  dt=1.8e-4  rate=1718   rho_min=0.398
step  175  dt=4.8e-5  rate=1.5e4  rho_min=0.252   <- and then negative
```

The pre-shock density is exactly 1 in the exact solution. It is 5% low by step
25 and 60% low by step 125, while `dt` and the rate remain nearly constant. A dispersive
undershoot at the shock reduces the state for 150 steps before the timestep
responds. Positivity is lost around step 175, and only then does the diffusive
rate increase rapidly and `dt` collapse. The underlying cause is insufficient
damping of the undershoot at that CFL, rather than the timestep collapse itself.
This is a spatial-regularization problem rather than a temporal one, and it is
why raising `C_beta` does not help either — the C_beta table above shows the
upper bound is itself a stability bound.

### Recovery strategy

Rollback retains a larger CFL after startup because the restriction occurs while
the shock forms. `StepControl(retries = 4)` recovers Noh from an initial
`cfl = 0.9`:

```
nu   start cfl   recovered cfl   steps   plateau/exact
1    0.9         0.45            1433    0.9989
2    0.9         0.45             915    0.9369
3    0.9         0.1125          1636    0.9729
```

The corresponding ν = 1 calculation requires 4485 steps at a fixed
`cfl = 0.15`. Recovery is approximately three times faster, and `solver.cfl`
records the accepted value for subsequent calculations.

Rollback recovers abrupt failures but not the gradual degradation observed at
cfl = 0.3, because the most recent savepoint is already nonphysical when the
check fires. The latter case requires a lower initial CFL.

**Recommendation:** use `cfl = 0.5` for smooth and moderately compressible flow,
`cfl = 0.3` with shocks, and `StepControl(retries = 4)` for automatic recovery.
Use `cfl = 0.15` for converging geometry when rollback is disabled.

## Grid convergence

```
N         | Noh1 plat  deficit | Lax L1  | mix width
128       |  0.9945       +60% | 1.3e-2  |  0.03581
256       |  0.9994       +59% | 7.1e-3  |  0.01820
512       |  0.9991       +58% | 3.9e-3  |  0.00931
1024      |  0.9992       +56% | 1.9e-3  |  0.00481
```

The refinement study exhibits three distinct behaviors:

- **Lax L1 halves per doubling**, giving first-order L1 convergence for the
  captured discontinuity regardless of interior order. The sixth- and
  tenth-order convergence lives in `test/convergence.jl`, on smooth fields.
- **Interface width halves per doubling** — the regularization follows the mesh
  rather than remaining at a fixed physical scale, as required for the Cook
  artificial properties to act as a subgrid model.
- **Wall heating does not converge away.** 60% → 56% over an 8× refinement. This
  is the known character of the Noh problem: the entropy error is deposited once,
  in the first cell at shock formation and remains there. Its spatial extent
  decreases with the cell size, so the integrated error vanishes
  while the pointwise error does not. Pointwise convergence of the wall-heating
  deficit is therefore not expected for this problem.

## Geometry limits

Construction of the validation battery identified two geometry limits.

**The spherical origin requires initial data resolved over ≳3 cells.** A blast
initialized as a top hat with a 1–2 cell transition loses positivity within
tens of steps; at 3 cells and wider it runs to completion. The cylindrical axis
accepts a 1-cell transition, and the same top hat completes in
Cartesian, so this is specific to the origin fold and its antipodal pairing —
where a discontinuity is being differentiated through a fold whose closure is
third order. `test/cases.jl` therefore initializes Sedov with a Gaussian deposit.

**The spherical origin is incompatible with the singular t = 0 start of Noh.** Every
CFL and every constant setting fails; the exact solution requires 64× compression
to appear at r = 0 instantaneously. A warm start from the exact solution at
t = 0.3 integrates to 0.6 and tests whether the solver can maintain the solution
through the origin without also testing the initialization singularity. The
cylindrical axis accepts the cold start at 16× compression.

## Recommendations

| Constant | Default | Keep? | Notes |
|---|---|---|---|
| `C_mu` | 0.002 | yes | TGV at 128³ gives an optimum of 0.004 ± 0.003, which contains 0.002; values above 0.008 are unstable in spherical Noh. |
| `C_beta` | 1.0 | yes | Accuracy optimum near 0.4; use 0.5 for interface-dominated work and avoid zero. |
| `C_kappa` | 0.01 | yes | 0.02–0.04 measurably reduces wall heating; avoid zero. |
| `C_D` | 0.01 | yes | The compact filter dominates interface broadening, so sensitivity to `C_D` is weak. |
| `cfl` | 0.5 | **no** | Use 0.3 with shocks and `StepControl(retries = 4)` for recovery. |

Remaining items, in approximate priority order:

1. Damp the dispersive undershoot that precedes the positivity loss. This is
   the observed CFL restriction and is a spatial-regularization problem: the
   sensor is not switching on early enough, or β\* is not reaching far enough
   ahead of the front. Neither a timestep predictor nor a larger `C_beta`
   touches it; both were measured.
2. Make the κ\* construction non-singular as T_ion → 0. It is an EOS dispatch
   point as of the Phase 1 work, so a tabular model can already supply its own;
   the gas models still divide by the temperature.
3. Narrow the `C_mu` band below ±0.003 by fitting a parabola to the windowed
   maximum and sampling the peak more densely. A subsequent comparison should
   digitize the van Rees −dKE/dt curve and fit the complete history rather than
   one scalar.
4. Calibrate the compact filter itself. It is necessary and sufficient for
   stability at 128³ while `filter_interval` and α have never been fitted to
   anything, which makes every `C_mu` number conditional on
   `compact_filter(0.45)` every step.
5. Determine why the spherical-origin fold is less tolerant of
   under-resolved data than the cylindrical axis fold.
