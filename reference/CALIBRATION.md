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
6. [The filter dissipates per application, not per unit time](#the-filter-dissipates-per-application-not-per-unit-time)
7. [The β\* sensor — strain, gated, or dilatation](#the-beta-sensor--strain-or-dilatation)
8. [CFL, which dominates all four](#cfl-which-dominates-all-four)
9. [The fold closure is not third order](#the-fold-closure-is-not-third-order)
10. [The origin cell is a startup transient](#the-origin-cell-is-a-startup-transient)
11. [Measured against the reference implementation](#measured-against-the-reference-implementation)
12. [Grid convergence](#grid-convergence)
13. [Geometry limits](#geometry-limits)
14. [Recommendations](#recommendations)

## How to read the tables

All entries in the four constant tables are from one-dimensional calculations.
Because `C_mu` multiplies the shear component of the artificial stress, these
cases exercise it only through the stress trace. The `C_mu` table therefore
constrains stability but not shear accuracy. The cases fully exercise `C_beta`,
`C_kappa`, and `C_D`.

The [`C_mu` section](#c_mu--the-shear-viscosity) also contains a separate
three-dimensional Taylor–Green study with distinct estimator limitations.

Throughout, **ν is the Noh geometry index**: ν = 1 planar (a slip wall), ν = 2
the cylindrical axis fold, ν = 3 the spherical origin fold. It is one more than
the exponent in the r^(ν−1) area weight, and the exact post-shock compression is
4^ν at γ = 5/3. The three cases are defined in `test/cases.jl`, and the labels
`Noh1`, `Noh2`, `Noh3` in the table headings below carry the same index.

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
0         |     NaN       NaN  |      NaN  | 5.7e-3   0.0032 |    1.6610 |    NaN
0.25      |  0.9995       +56% |   1.0098  | 4.8e-3   0.0041 |    1.6408 | 6.4151
0.5       |  0.9994       +61% |   0.9931  | 4.9e-3   0.0045 |    1.6297 | 6.4742
1.0    *  |  0.9993       +64% |   0.9751  | 5.0e-3   0.0053 |    1.6180 | 6.6049
2.0       |     NaN       NaN  |   0.9569  | 5.2e-3   0.0063 |    1.6055 | 6.6859
4.0       |     NaN       NaN  |   0.9408  | 5.4e-3   0.0074 |    1.5945 | 6.8366
```

Measured under the shipped `smoother = :gaussian`. The table previously held the
`:compact` measurement and was not re-run when that default changed in August
2026; the differences are in the third and fourth digits everywhere except the
ν = 1 deficit, which reads +64% rather than +58% because the Gaussian smoother
costs wall heating. Two entries changed qualitatively: `C_beta = 4` completes
spherical Noh under the Gaussian where it did not under `:compact`, and the
CFL ladder below is new.

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
- The Shu–Osher wave train loses about 2.8% of its amplitude over the same
  range. Because the entropy waves are smooth, this reduction measures
  overdamping relevant to a mixing calculation.
- The Noh ν = 3 plateau *degrades* with increasing `C_beta` (1.0098 → 0.9408),
  crossing exact between 0.25 and 0.5.

Accuracy alone gives an optimum near `C_beta = 0.4`. The shipped value of 1.0
instead provides a robustness margin: Woodward–Colella and both Noh geometries
complete, at the cost of an 18% wider contact and 0.7% lower wave-train amplitude
relative to 0.5.

### The CFL ladder per constant

The table above is read at `NOH_CFL = 0.15`, which hides the interaction that
matters most for converging geometry. Highest CFL reaching `t_final` with a
correct plateau, from `bench/artcal.jl beta`:

```
C_beta      nu = 1     nu = 2     nu = 3
0.25         1.0+       1.0        0.2
0.5          0.4        0.4        0.25
1.0    *     0.25       0.2        0.4
2.0          none       none       0.3
```

`C_beta` trades the planar and cylindrical ceilings against the spherical one,
monotonically across the sampled range. Lowering it to 0.25 removes the
restriction at the wall and the axis and costs the origin a factor of two;
raising it does the reverse. The shipped value of 1.0 is the setting that
maximizes the spherical ceiling, which is the geometry the general guidance for
converging shocks rests on.

This corrects a claim carried in the remainders list below since the CFL study.
A *larger* `C_beta` was measured against the ν = 1 restriction and does not move
it; a smaller one moves it from 0.25 to beyond 1.0. The earlier measurement was
not wrong, it was one-sided.

The trade is the same one `detector = :d8` makes, and in the same direction:
both act by reducing β\* where the field is smooth, and both help at the wall
and the axis while costing the origin. That is the reason the two are refitted
together [below](#the-c_beta-refit-under-d8).

**Recommendation:** retain 1.0 as the default. Use 0.5 for interface-dominated
Richtmyer–Meshkov or Rayleigh–Taylor calculations with moderate shocks, noting
that it costs the spherical origin half its timestep. Values below 0.25 or
above 2 are not recommended.

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

<a id="the-beta-sensor--strain-or-dilatation"></a>

## The filter dissipates per application, not per unit time

The compact filter supplies most of the energy sink at every resolution
measured here, and it has never been calibrated. This section covers one half
of that debt, the half that is a property of the formulation rather than of a
constant: the filter removes energy per *application*, so its dissipation is
not a rate and does not converge as `dt → 0` at fixed resolution. Fitting α and
the cadence against a reference dissipation history is the other half and still
requires cluster time.

### The measurement

A parallel shear layer, `u_x = 0.1 sin(4y)` at uniform ρ and p, is an exact
steady solution of the Euler equations and stays one discretely, since every
x-derivative of the field vanishes. Kinetic energy is then constant in time and
the filter is the only mechanism that can change it. `bench/filterrate.jl`,
N = 32 to t = 0.5, zero viscosity and artificial properties off:

```
cfl    steps   unrelaxed          relaxed (filter_cfl = 0.4)
0.4      73    4.092e-3  1.000    4.042e-3  1.000
0.2     145    8.107e-3  1.981    4.042e-3  1.000
0.1     289    1.608e-2  3.930    4.042e-3  1.000
```

Unrelaxed, the loss tracks the **step count** (73 : 145 : 289 = 1 : 1.99 : 3.96)
rather than the elapsed time. A calculation at half the CFL applies twice the
subgrid dissipation over the same physical interval. Relaxed, the loss is
constant to six significant figures across a fourfold change in timestep.

Getting a clean reading required the right case, and two obvious ones do not
work. A broadband field loses 64% of its kinetic energy within tens of steps
and then cannot lose more, collapsing the spread across a 4× CFL change to
1.2%. A velocity sine at uniform pressure is an acoustic oscillation trading
kinetic for internal energy hundreds of times faster than the filter acts.
Total energy shows nothing either: a symmetric filter on a periodic grid
conserves the discrete sum of every conserved variable exactly, so the filter
moves energy between the two reservoirs rather than removing it.

### What it costs on a real case

Taylor–Green at 32³, Re = 1600, artificial properties on, Gaussian smoother,
`bench/tgv_energy.jl` at the dissipation peak:

```
filter_cfl   cfl    peak -dKE/dt        filter    mu*     molecular
0 (shipped)  0.6    1.4216e-2 @ 6.58    82.2%     5.1%    12.6%
0 (shipped)  0.3    1.4297e-2 @ 6.61    85.0%     3.6%    11.4%
0.6          0.6    1.4216e-2 @ 6.58    82.2%     5.1%    12.6%
0.6          0.3    1.4373e-2 @ 6.44    82.1%     5.2%    12.7%
```

At the reference CFL the relaxed run reproduces the unrelaxed one to every
printed digit, which is the `w = 1` path and a check on the implementation.

**The timestep moves the attribution rather than the total.** Halving the
CFL changes the peak dissipation by 0.6%, but moves the filter's share of it
from 82.2% to 85.0% and the μ\* share from 5.1% to 3.6% — a 29% relative
change in the artificial-viscosity channel from a timestep change alone, with
`C_mu` held fixed. Under the relaxation both hold: 82.1% and 5.2%.

The total barely moves because the sinks compete rather than add. The cascade
rate is set at the large scales, and a filter that takes more at the grid scale
leaves less to reach the scales where μ\* and molecular dissipation act. This
is why filter dominance can be large and still leave the peak near the
reference value.

**The consequence for calibration is that `C_mu` is conditional on the CFL as
well as on the filter.** The μ\* share of the sink is the quantity `C_mu` is
fitted against, and it moves by 29% relative under a change that has nothing to
do with the physics. Any refit under the unrelaxed formulation has to state its
CFL to be reproducible. This is measured at 32³ only; the shares themselves are
strongly resolution dependent (the filter falls to 37% at 128³), and whether
the CFL sensitivity survives refinement has not been tested.

### The formulation

`filter_cfl` is the CFL at which one pass is applied at full strength. Below it
the state is relaxed toward the filtered image rather than replaced by it,

    Q ← (1 − w) Q + w F(Q),    w = filter_interval · dt · rate / filter_cfl

capped at one, which holds the dissipation per unit time fixed. Since
`compute_dt` sets `dt = cfl / rate`, the product `dt · rate` recovers the CFL
actually taken, including `StepControl` backoff and the shortening applied to
land on a callback instant. Reading it that way rather than reading
`solver.cfl` makes a shortened step filter proportionally less, which removes
the truncated-final-step artifact recorded against `bench/tgv_energy.jl`.

The default is `filter_cfl = 0`, the unrelaxed formulation, and it takes the
original code path exactly rather than a blend at `w = 1`. Every guarded number
in the suite was measured there and none of them moves.

## The β\* sensor — strain, gated, or dilatation

`ArtParams.beta_sensor` selects how β\* is built. μ\*, κ\* and D\* are unchanged
by all three settings. Throughout this section, S is the strain-rate tensor and
|S| its magnitude, δ⁴ the undivided fourth difference (1, −4, 6, −4, 1), h_d the
grid spacing along dimension d, Δ = ∇·u the dilatation, ω the vorticity vector,
H the Heaviside step, and ε a fixed regularizer at the literature value 1e-32.

- `:strain` (default) is the Cook original, Σ_d h_d²|δ⁴_d S|.
- `:gated_strain` keeps that sensor and multiplies it by the compression
  switch H(−Δ)·Δ²/(Δ² + |ω|² + ε). One pointwise pass, no line solves.
- `:dilatation` additionally rebuilds the sensor from Δ,
  the full form of Mani, Larsson and Moin (JCP 228, 2009). One further sensor
  smoothing pass per RHS evaluation.

The two halves of the literature refinement are separable, and separating them
is what the measurements below are for. The motivation was the hypothesis
recorded in [the CFL section](#cfl-which-dominates-all-four): that the
`cfl ≤ 0.15` restriction comes from a sensor that does not switch on early
enough at a forming shock.

### The CFL ceiling

```
sensor        cfl  | Noh1 plat/exact | Noh2 plat/exact | Noh3 plat/exact
strain       0.4   |             NaN |             NaN |             NaN
strain       0.3   |             NaN |             NaN |             NaN
strain       0.2   |          0.9993 |             NaN |             NaN
strain       0.15  |          0.9992 |          0.9355 |          0.9732
gated_strain 0.4   |             NaN |             NaN |             NaN
gated_strain 0.3   |             NaN |             NaN |             NaN
gated_strain 0.2   |          0.9992 |          0.9353 |             NaN
gated_strain 0.15  |          0.9992 |          0.9344 |          0.9722
dilatation   0.4   |             NaN |             NaN |             NaN
dilatation   0.3   |             NaN |             NaN |             NaN
dilatation   0.2   |          0.9990 |             NaN |             NaN
dilatation   0.15  |          0.9990 |             NaN |             NaN
```

**The switch moves the cylindrical ceiling; the sensor change does not move
anything.** `:gated_strain` completes cylindrical Noh at `cfl = 0.2`, where both
other settings fail, and the plateau it reaches there (0.9353) agrees with its
own value at 0.15 (0.9344), so the larger step is a genuine completion rather
than a run that merely avoided the positivity check. A finer ladder puts the new
ceiling between 0.2 and 0.25:

```
Noh2   cfl  | strain     gated_strain
       0.15 |    0.9355        0.9344
       0.2  |       NaN        0.9353
       0.25 |       NaN           NaN
       0.3  |       NaN           NaN
```

Planar and spherical Noh are unmoved: 0.2 completes and 0.3 does not under all
three settings for ν = 1, and ν = 3 needs 0.15 under both settings that reach it
at all. So one geometry of three gains a 33% larger timestep, and the general
`cfl ≤ 0.15` guidance for converging shocks stands, now governed by the
spherical case alone.

`:dilatation` loses cylindrical and spherical Noh at every CFL sampled. Since
`:gated_strain` keeps them and the two differ only in the sensor field, the
failure is attributable to the sensor change and not to the switch — which the
[fold analysis below](#why-the-dilatation-sensor-loses-the-converging-cases)
confirms directly.

### Accuracy at the shipped CFL

```
sensor       | Noh1 plat  deficit | Noh3 plat | Lax L1  contact | Shu train | WC peak
strain     * |  0.9992       +58% |   0.9732  | 5.0e-3   0.0051 |    1.6185 | 6.5731
gated_strain |  0.9992       +58% |   0.9722  | 5.0e-3   0.0052 |    1.6177 | 6.5850
dilatation   |  0.9990       +59% |      NaN  | 5.0e-3   0.0051 |    1.6226 | 6.5315
```

`:gated_strain` is a wash on accuracy: every column moves in the fourth digit
and the movements go both ways (Woodward's peak improves, the Noh ν = 3 plateau
and the Shu–Osher train degrade slightly). Its case rests on the cylindrical
ceiling, not on these numbers.

The Shu–Osher gain belongs to the sensor field, not to the switch. `:dilatation`
keeps 0.25% more wave-train amplitude than the default, while `:gated_strain`
keeps 0.05% *less* — so the improvement comes from measuring compression rather
than deformation, and gating alone does not produce it.

### What the switch removes

On a solenoidal Taylor–Green field at 32³, where β\* has nothing legitimate to
do, `:gated_strain` removes 99.4% of the summed β\* and leaves 71 of 32768
points above 1e-12. It does not remove the *maximum*, which stays at 0.42 of the
ungated value. Those surviving points are the cusps of |S|: the strain sensor is
a fourth difference, so it peaks exactly where |S| passes through zero with a
kink, and the switch degenerates at those same points because the vorticity
vanishes there too and ε is all that is left in the denominator. A relative ε
scaled to the local |S| was tried and reverted — it cannot help, because the
scale it would use vanishes with |S|. `:dilatation` has no such points, its β\*
falling to 1e-15 of the ungated maximum, because its sensor is built from ∇·u
and is therefore small wherever the flow is solenoidal.

<a id="why-the-dilatation-sensor-loses-the-converging-cases"></a>

### Why the dilatation sensor loses the converging cases

Raising `C_beta` does not recover them, which rules out the obvious explanation
that the switch (bounded by 1) simply scales β\* down — and `:gated_strain`,
which applies that same switch and keeps both geometries, rules it out a second
way. At `C_beta = 2, 4, 8` under the dilatation sensor, cylindrical and
spherical Noh still fail *and planar Noh joins them* — the same upper stability
bound the [`C_beta` table](#c_beta--the-shock-constant) shows for the strain
sensor.

The cause is at the coordinate fold, and it is visible at t = 0 before any
shock forms. Noh starts from u_r = −1 everywhere, for which the strain and
dilatation sensors are analytically identical away from the axis: Δ = −1/r and
|S| = 1/r, with zero vorticity, so the switch is exactly 1 and β\* agrees
bit-for-bit. At the first
few cells of the axis the discrete radial derivative of u_r is no longer zero,
and there the two forms diverge, because Δ = S_rr + S_θθ **adds** two
same-signed components where |S| = √(S_rr² + S_θθ²) partially cancels them.
Measured on cylindrical Noh at N = 256, t = 0:

```
i   r        |S|      div       beta* strain   beta* dilatation
1   0.00196  612.0    -847.8    1.746e-02      3.790e-02
2   0.00587  204.1     -57.9    1.466e-02      4.106e-02
3   0.00978  110.9    -145.1    3.771e-03      2.087e-02
6   0.02153   46.5     -44.1    0             8.089e-04
120 0.46771    2.14     -2.14   3.856e-12      3.856e-12
256 1.00000    1.00     -1.00   1.199e-07      1.199e-07
```

The two agree to the last digit well away from the axis and differ by factors
of 2 to 70 within the first several cells, where the cell measure is smallest.
The run loses positivity on the first step. This is a property of the fold
rather than of the shock-capturing, and it is why the option is not recommended
in converging geometry regardless of how the constants are set.

### Decomposition sensitivity

Neither compression-keyed setting reproduces to round-off when the process grid
changes. Summed over the domain, three different split axes agree to 2e-6
relative for `:gated_strain` and 2e-7 for `:dilatation`, against 1e-14 for the
strain sensor. `:gated_strain` is the worse of the two for the same reason its
maximum survives a solenoidal field: the sensor it gates is O(1) at the points
where the gate toggles. The cause is not a missing halo
exchange — the sensor fields themselves reproduce to 1e-14 — but the switch.
H(−Δ) is discontinuous at Δ = 0, and with ε at the literature value of 1e-32 the
ratio Δ²/(Δ² + |ω|² + ε) has not decayed by the time Δ reaches round-off, so a
point whose dilatation cancels to zero carries either no β\* or the full
C_β·ρ·sensor depending on the last bit. The affected points contribute
negligible β\*, but the property is real and `test/mpi_tests.jl` records it in
the tolerance it uses. Anything relying on bit-identical results across process
grids should stay on `:strain`.

**Recommendation:** retain `:strain` as the default, and reach for
`:gated_strain` in cylindrical converging geometry, where it buys a 33% larger
timestep for one pointwise pass and costs nothing measurable elsewhere. Whether
it should *become* the default is a separate question this study does not
settle: it changes guarded numbers in the fourth digit across the whole
validation battery, so the case for it needs a re-baseline and a second geometry
showing the same gain. The mechanism is now known:
[the CFL measurement](#where-the-restriction-originates) shows that the gate
relieves the axis cell rather than the shock. That narrows where a second
geometry could come from, since it would have to be one whose ceiling is also
set by a fold the gate can relieve, and the planar wall and spherical origin are
measured not to be. `:dilatation` is worth considering only for Cartesian,
shock-dominated work where the Shu–Osher amplitude gain matters and no
coordinate fold is present.

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

The density falls for 150 steps while `dt` and the rate remain nearly constant.
Positivity is lost around step 175, and only then does the diffusive rate
increase rapidly and `dt` collapse. The cause is therefore spatial rather than
temporal, which is also why raising `C_beta` does not help: the C_beta table
above shows its upper bound is itself a stability bound.

An earlier version of this section located the loss *ahead of the front*, as a
dispersive pre-shock undershoot that β\* fails to damp. That attribution was
wrong; the subsection below replaces it.

<a id="where-the-restriction-originates"></a>

### Where the restriction originates

`bench/nohprobe.jl` samples the state per step and reports, alongside the
density, the internal energy recovered from `Q`, the count of cells carrying
negative internal energy, and two measures of where β\* is: its value at the
worst-affected cell relative to its own maximum, and the furthest cell ahead of
the front carrying above a thousandth of that maximum. The three readings below
rule out insufficient β\* reach.

**β\* reach is not short.** Over a complete ν = 1 run at the shipped
`cfl = 0.15`, the reach holds at 14.2–14.8 cells ahead of the front while the
worst cell sits 3–5 cells ahead:

```
step    x_sh/h | rho_min   i | e/e0_min   i  n_e<0 | b*@e/b*max  reach/h
 500      9.81 | 0.98875  17 |   -469.4  12      7 |      0.166    14.19
1500     27.63 | 0.99008  35 |   -421.7  30      8 |      0.192    14.37
2500     45.47 | 0.98851  53 |   -465.4  48      7 |      0.161    14.53
3500     63.30 | 0.98720  71 |   -498.7  66      7 |      0.140    14.70
```

The margin is a factor of three to five, held for the whole run. Widening the
sensor stencil would therefore address a deficit that is not present.

**β\* is present where the damage is, and the damage persists.** The
`b*@e/b*max` column above stands at 0.14–0.21 throughout, so the affected cell
carries a sixth of the peak artificial bulk viscosity in the domain. At the
failing `cfl = 0.3` the same column reads 0.84–1.00 over the first 125 steps,
placing the worst cell at or near the β\* maximum itself. A failure occurring
under the largest β\* in the domain is not explained by insufficient β\*.

**The failure starts at the symmetry plane, not ahead of the front.** At ν = 1,
`cfl = 0.3` the first cell to degrade is the wall cell i = 1, whose internal
energy is negative by step 5, and the density hole that the trace above reports
at step 125 is at i = 3, between the wall and a front then at cell 4.4, rather
than ahead of it. The pre-shock field at that step is still within 1% of unity
from cell 11 outward. The ν = 3 origin fails the same way and more abruptly: one
step before the failure at step 172 the density minimum over the whole line is
still 1.92, and the origin cell then carries an outward u = +5.6 against an
inflow of −1, with e/e₀ = +2.6e5 against −1.6e4 in its neighbour, under a β\* of
5.7 where the front carries 0.06.

The restriction is therefore a symmetry-plane startup problem. β\* is already
saturated at the fold, so the surviving candidate is the fold closure rather
than the artificial properties. (The fold closure has since been measured and
is *not* third order — see
[the fold closure is not third order](#the-fold-closure-is-not-third-order) —
so this sentence records what was believed here, not a live candidate. The
startup character of the restriction survives that correction and is
strengthened by it.) This is consistent with `cfl ≤ 0.15` binding in
all three geometries, and with `StepControl(retries = 4)` recovering afterwards,
since the restriction applies while the shock forms at the symmetry point. It
also accounts for the one setting that moved a ceiling: `:gated_strain` at
ν = 2, `cfl = 0.2` migrates the worst-energy cell off the axis (i = 1 → i =
6…11) and reduces its magnitude about sixtyfold, from e/e₀ = −5.5e4 to
−90…−212, where the ungated sensor holds i = 1 at −1.1e4 to −1.2e4
indefinitely. Its gain is at the fold rather than at the shock.

The half of that conclusion exonerating the artificial properties does not
survive the smoother measurement. Changing the sensor smoother alone, with every
constant held, moves the ν = 3 ceiling from 0.15 to 0.4 and the ν = 2 ceiling
from 0.15 to 0.2. The artificial-property path does set the restriction at the
fold, through the *continuity* of β\* rather than its magnitude, which is what
the saturation argument above measured and correctly ruled out. The subsection
below carries the measurement. The fold closure has since been tested and
retired as a candidate.

### The sensor is intermittent at the damage site

The readings above establish that β\* is present at the damaged cell on average.
Sampling the ν = 3 origin every 25 steps rather than every 1000 shows that the
average conceals an alternation. Both smoothers at `cfl = 0.15`, where both runs
complete, reporting `b*@e/b*max`:

```
step        150    175    200    225    250    275    300    325    350    375    400
compact    0.393  0.004  0.302  0.007  0.305  0.009  0.300  0.011  0.306  0.007  0.213
gaussian   0.103  0.336  0.118  0.429  0.415  0.120  0.143  0.131  0.108  0.375  0.130
```

Under the shipped smoother the cell being damaged carries about 30% of the
domain maximum on one sample and under 1% on the next. Under the explicit
Gaussian of
[the reference comparison](#measured-against-the-reference-implementation) it
never falls below 0.100. `reach/h` is 6–11 cells for both, so reach is again not
the discriminating quantity, and `n_e<0` holds at 7–9 for both, so the
negative-internal-energy condition below is independent of this axis.

The same signature precedes the failure at `cfl = 0.3`, where `:compact` reads
0.002, 0.045, 0.251, 0.025, 0.270 before losing density at step 172 while
`:gaussian` completes 400 steps with a floor near 0.09.

What this indicates is sensor roughness rather than sensor magnitude. The
undivided δ⁴ applied to |S| produces a spiky field; the damaged cell drifts
outward over the run (i = 30 → 39) and, under a smoother that is close to the
identity across the resolved band, drifts alternately onto spikes and into
troughs. A nine-point Gaussian spreads each spike widely enough that no trough
remains to fall into.

**This neither overturns the ν = 1 readings above nor confirms them.** That
table samples every 1000 steps, which cannot resolve an alternation of this
period, so its steady 0.14–0.21 is consistent both with a genuinely steady
sensor and with the average of an alternation. Re-running the ν = 1 probe at
`every = 25` is the cheap way to settle it and has not been done.

The consequence for the eighth-derivative detector is a sequencing one. A
sharper high-pass produces narrower spikes, so on its own it would be expected
to worsen intermittency; the reference implementation pairs it with the Gaussian
and never with a near-identity filter. A measurement of the detector must
therefore be taken on top of the Gaussian smoother, or it will reject a
refinement for a reason belonging to the smoother.

### Negative internal energy in runs that pass

The `n_e<0` column above is not zero in any sampled configuration, including
every run that completes. Six to eight interior cells carry negative internal
energy, travelling with the front, for the entire duration of the shipped ν = 1
validation case. No diagnostic reports this: `primitives!` floors T_ion at
1e-300 wherever e ≤ 0, so p becomes ρ·R·1e-300 and the run continues, while the
positivity check in `max_rate` reads ρ, which stays positive throughout.

The quantity is ill-conditioned in this problem rather than merely inaccurate.
At the Noh ambient p₀ = 1e-4 the internal energy is 1.5e-4 while the kinetic
energy is 0.5, so e is recovered as a difference of terms that agree to within
0.03%, and a rounding-level error in either produces a sign error in e. Since
the κ\* sensor is built on e, this also bounds the artificial conductivity. It
is the concrete case for extending model debt 2, the positivity failsafe, to
cover internal energy rather than density alone.

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
For converging geometry with rollback disabled, the ceiling is now set by
`ArtParams.smoother`: under the default `:gaussian` it is 0.4 at the spherical
origin and 0.2 at the cylindrical axis, against 0.15 for both under `:compact`.
The tables in this section were measured under `:compact` and are retained as
the record of that setting.

## The fold closure is not third order

The candidate this document carried longest for the converging-shock ceiling was
the fold closure: third order against a sixth- or tenth-order interior, at the
one geometry the `:d8` detector costs anything. That premise does not hold,
and the number behind it belongs to the outer wall rather than to the fold.

`test/convergence.jl` reports a **global** max norm, and every one of its fold
studies closes the outer end with a `SlipWallBC` whose closure rows measure 3.17
on their own. Splitting that norm by region (`bench/foldorder.jl`, same fields
and resolutions) separates the two ends:

```
study                              fold(1:3)   mid    outer(3)   global argmax
C6, both ends walls (control)         3.23     3.19     3.17       i = n
cylindrical axis, odd  (u_r-like)     6.05     3.76     3.71       i = n
cylindrical axis, even (scalar)       7.01     2.99     3.00       i = n
spherical origin, even (scalar)       7.00     2.97     2.99       i = n
spherical origin, odd  (u_r-like)     6.07     3.86     3.81       i = n
```

**The global maximum sits at the outer wall in every fold study.** The fold's
own error converges at 6.05 to 7.01 and is three to five orders of magnitude
below the interior: at N = 96 the spherical origin carries 7.0e-12 against
1.9e-7 in the middle of the line. The fold is the most accurate region of the
line.

Every global error `test/convergence.jl` prints equals the outer-window norm
above to every digit printed: 7.707e-05, 1.953e-07, 1.992e-06 and 4.730e-06 at
the finest resolution of each study. The identification is therefore exact
rather than approximate, and the guarded numbers in that file are measurements
of the outer wall, taken through a norm insensitive to the fold.

The control establishes that the split is meaningful. With walls at both ends
the same window reports 3.23, so the instrument does detect a third-order
closure where one is present. The middle of the line converges at 3 as well, which is the compact
scheme's line-global coupling carrying the wall's closure error inward, not a
property of the fold.

Both parities were measured, including the odd one that
`test/convergence.jl` does not cover and that a converging calculation actually
differentiates at the origin.

### What this leaves

Retiring the closure removes the last standing candidate for the ceiling rather
than replacing it. All four mechanisms this document has proposed are now
measured wrong: the timestep predictor, the sensor reach, the fold closure, and
sensor blindness at the fold, which
[the origin-cell probe](#the-origin-cell-is-a-startup-transient) rules out.

## The origin cell is a startup transient

`bench/nohprobe.jl` now reports the symmetry cell on every line rather than only
when it is the worst cell. The argmin columns track the front for most of a
run, so a symmetry cell
degrading underneath them stays invisible until it overtakes the front, by which
point the run is a few steps from losing positivity.

Spherical Noh, N = 256, sampling `rho1/rho2` (the symmetry cell over its
neighbour) and β\* at the symmetry cell over the line maximum:

```
                  :d8, cfl 0.3 (fails)     :delta4, cfl 0.3 (survives)
step   rho1/rho2   b*1/max            step   rho1/rho2   b*1/max
  85     1.0021      0.001              80     0.9993      0.004
  90     0.9879      0.009             120     0.9808      0.172
 110     0.8930      0.043             140     0.7469      0.964
 115     1.0532      0.101             160     0.9652      1.000
 120     1.3089      0.304             180     0.9223      0.346
 125     0.2257      0.018             200     0.9530      0.010
 128     FAILED
```

Three readings follow, and the first two retire standing explanations.

**The symmetry cell is quiescent for most of the run.** Through step 85 it holds
`rho1/rho2` to within 0.2% of unity and carries β\* at a thousandth of the line
maximum. Whatever sets the ceiling does not act gradually from the start.

**The sensor is not blind at the fold.** During the excursion β\* at the origin
reaches the domain maximum, 1.000, in both surviving configurations. The
[detector account](#where-the-spherical-loss-comes-from) proposed that δ⁴'s poor
selectivity was supplying a background regularization at the fold, where the
field is smooth and even by construction and a selective detector correctly
returns almost nothing. The first half of that holds: β\* is negligible at the
fold while the field is smooth. The conclusion does not, because the field does
not stay smooth, and once it stops the sensor responds under either detector.

**Every configuration has the excursion; the ceiling is whether the cell
survives it.** `:delta4` at cfl 0.3 and `:d8` at cfl 0.25 both pass through it
and continue to `t_final`. `:d8` at cfl 0.3 enters it around forty steps earlier
and the cell evacuates within one sampling interval, `rho1/rho2` falling
1.3089 → 0.2257 with the internal energy reaching −6335 e₀.

### The excursion is physical, not grid-scale

Peak of β\*1/max under refinement, `:delta4` at cfl 0.3:

```
N      step   t          peak b*1/max
128      58   0.377-0.382    1.000
256     160   0.392-0.395    1.000
512     342   0.393-0.395    0.950
```

The step number scales with N while the time does not: the excursion lands at
t ≈ 0.394 at every resolution, converging as the grid refines, and its amplitude
relative to the line maximum weakens. It is a resolved feature of the warm start
at t₀ = 0.3 rather than a grid-scale artifact, which is consistent with
[the restriction being a startup one](#where-the-restriction-originates) and
with `StepControl(retries = 4)` beating a globally lowered CFL.

### Where the regularization goes at the moment of failure

β\* is proportional to the density by construction, `C_beta * rho * sensor`
(`src/artificial.jl:502`). At the failing step the symmetry cell has thinned to
0.23 of its neighbour, and β\* there has fallen 0.304 → 0.018 of the line
maximum while the cell is the worst in the domain. The step-126 profile puts
ρ = 38.1 at the origin against 136.7 and 162.1 at the next two cells, and the
velocity at the origin reversed to +1.19 against −0.51 in its neighbour.

The regularization at an evacuating cell is therefore suppressed by the
evacuation itself. This is Cook's
formulation working as specified rather than a defect, and it is a mechanism
consistent with the numbers rather than a demonstrated cause: establishing it
requires a β\* that does not vanish with the density, which has not been run.
The immediate consequence for the roadmap is that the origin ceiling looks like
a robustness problem at a symmetry cell under a startup transient, which is
[model debt 2](ROADMAP.md), not a discretization-order problem.

## Measured against the reference implementation

Miranda's Fortran kernels, carried by Pyranda in `pyranda/parcop/`, implement
the same Cook artificial-property method. Reading them against `artificial.jl`
identifies four differences that bear on the constants calibrated above. Two
have since been measured and their subsections record the outcome; the other
two record what the reference does and what is analytically established about
the gap, so that the investigation does not start from source archaeology a
second time.

### The ringing detector is an eighth derivative, not a fourth difference

`ring()` in `parcop/operators.f90` dispatches to `d8x/d8y/d8z`, a full compact
operator (`c10d8`, `parcop/stencils.f90`) with a pentadiagonal left-hand side,
a nine-point right-hand side, and its own symmetric and antisymmetric boundary
closures. `artificial.jl` uses the undivided fourth difference
δ⁴ = (1, −4, 6, −4, 1), following Cook (2007) literally.

`ArtParams.detector` now selects between them: `:delta4` is the default and
`:d8` is [`compact_d8`](../src/kernels_banded.jl), the reference operator
transcribed. Its interior rows are

```
1.5 g_{i-2} + 14 g_{i-1} + 29 g_i + 14 g_{i+1} + 1.5 g_{i+2} = 60 δ⁸f_i,
```

with δ⁸ the undivided eighth difference. An even derivative preserves parity,
so it is planned as a symmetric operator rather than as a derivative: no `1/h`
scaling, mirrored rather than negated at a high edge, and folded with the
filter-side parity at a coordinate singularity. It is the only symmetric
banded scheme in the package and therefore the only exercise of that
combination.

The reference's `-1:2` boundary variant 0 turns out to be the interior stencil
with its overhanging weights folded onto the **half-offset mirror**, ghost j ↔
interior j, which is the same construction `gaussian_filter` uses and the same
convention the fold machinery already implements. Four closure rows are
needed, since the right-hand side reaches ±4, and every row's weights sum to
zero, so a constant is annihilated in floating point rather than by
cancellation (measured 1.2e-16 through the closure rows).

#### Normalization, and why it is not the reference's

The coefficients are divided by ζ = 29 to put a unit diagonal on the
left-hand side, and by a further 240. The second factor sets the response to
a grid-to-grid oscillation to 16, which is what undivided δ⁴ gives there. Both
detectors then agree at the wavelength both exist to catch, and diverge only
below it:

```
k/pi     delta4     d8        ratio
0.25     3.43e-1    6.03e-4   569x
0.5      4.00e+0    1.54e-1    26x
0.75     1.17e+1    3.69e+0    3.2x
1.0      1.60e+1    1.60e+1     1x
```

Without that second factor `:d8` would produce sensors 240× larger at the
Nyquist and the four constants would have to be refit by two orders before
any case could be run at all. With it they transfer as starting points, which
is what makes the comparison below a one-variable one.

#### Results

Measured on top of `smoother = :gaussian`, which is a requirement rather than
a convenience: the failure the Gaussian fixes is
[β\* intermittency](#the-sensor-is-intermittent-at-the-damage-site) at a
symmetry cell, and a sharper high-pass makes narrower sensor spikes. Testing
`:d8` against the compact smoother would have rejected it for a defect
belonging to the smoother. `bench/artcal.jl detector`, shipped constants:

```
detector    Noh1 plat  deficit   Noh3 plat   Lax L1   contact   Shu train   WC peak
delta4*      0.9993     +64%      0.9751     5.0e-3   0.0053     1.6180     6.6049
d8           0.9997     +53%      0.9951     4.7e-3   0.0044     1.6360     6.3953
```

Six of the seven columns improve, several of them well beyond the fourth
digit that separated the β\* sensor variants. The ν = 3 plateau error falls
from 2.45% to 0.42%. Wall heating, the standing regression of the smoother
change, recovers eleven of the points it lost. The Shu–Osher wave train, which
is the column that pulls against every damping constant, gains 1.1%: the
detector declines to fire on structure the scheme resolves, which is precisely
what it was adopted for. Woodward–Colella peak density is the one regression,
6.6049 → 6.3953 at N = 400.

The CFL ceiling is where the result is largest, and where it is mixed. Highest
CFL reaching `t_final` with a correct plateau:

```
Noh geometry              :delta4    :d8
nu = 1  planar wall          0.2      1.0+
nu = 2  cylindrical axis     0.2      1.0+
nu = 3  spherical origin     0.4      0.25
```

`1.0+` is not a table edge: the plateau under `:d8` is flat to four digits
from 0.15 to 1.0 in both geometries (ν = 1 at 0.9997, ν = 2 at 0.9481–0.9500),
so the artificial-property restriction on those two is not raised, it is gone.
The spherical origin moves the other way, 0.4 to 0.25.

#### Where the spherical loss comes from

`bench/nohprobe.jl 3 detector=d8 cfl=0.3` puts the failure at the origin cell.
Through steps 50–100 the worst internal-energy cell travels with the front at
i = 30–32, carrying β\* at 15–36% of its own domain maximum. At step 125 the
worst cell is **i = 1**, the origin, at e/e₀ = −6335 with β\* at **1.8%** of
maximum, and the run loses density three steps later.

This is the mechanism already recorded under
[where the restriction originates](#where-the-restriction-originates), not a
new one: the restriction is a symmetry-plane startup problem, and `:d8` moves
its threshold rather than its character. The reading offered here was that δ⁴'s
poor selectivity was doing double duty, also supplying a background
regularization at the coordinate fold, where the field is smooth and even by
construction and a selective detector correctly returns almost nothing.

**Half of that reading is now measured wrong.** β\* is indeed negligible at the
fold while the field is smooth, but the symmetry cell does not fail while the
field is smooth: it fails during a startup excursion, and during that excursion
β\* at the origin reaches the *line maximum* under both detectors. The sensor is
not blind there. The fold closure the paragraph pointed at is not third order
either. Both corrections are in
[the origin cell is a startup transient](#the-origin-cell-is-a-startup-transient)
and [the fold closure is not third order](#the-fold-closure-is-not-third-order).

#### What the detector cannot help with

The selectivity is available to the κ\* and D\* channels and largely
unavailable to μ\* and β\*. On a pressure wave resolved over 64 points the two
detectors differ by 2.6e6 on κ\*, whose input is the internal energy. On a
velocity sine on the same grid they differ by a factor of 1.8 on β\*, whose
input is |S|. The absolute value is the reason: |S| has a cusp wherever the
strain passes through zero, a cusp is a grid-scale feature, and no detector
can decline to see one. This is the same geometry that
[defeats the Ducros switch](#what-the-switch-removes) on a solenoidal field,
observed from the other side.

It is also why the reference does not build the sensor this way. Miranda takes
μ\* from `ringV(u, v, w)` — the ring of each velocity *component* along each
direction, reduced by `MAX` over the nine combinations — and β\* from
`ring(∇·u)`. Neither input carries an absolute value. So the detector swap and
the sensor-field change are not independent improvements to be assessed
separately: this measurement is a lower bound on what the reference method
gains.

Both fields have since been measured, along with `MAX` against `Σ_d` over
directions and the ungated dilatation the reference actually ships, under
[the sensors read |S|](#the-sensors-read-s-not-the-velocity-and-the-dilatation).
In summary: the field determines how much of the detector's selectivity
survives; the μ\* field change is nonetheless a null result on every case in
the battery; and the β\* field change costs the cylindrical axis.

#### Cost, and why the default has not moved

On the two-species tube of `bench/phases.jl`, back to back on one machine:

```
detector   artificial   % of RHS   compute_rhs!   line solves
delta4      0.946 ms     23.5%       3.927 ms     24 + 0
d8          1.707 ms     34.6%       4.689 ms     24 + 8
```

Eight pentadiagonal solves per right-hand side, one per active dimension per
sensor, for +80% on the sensor phase and +19% on the whole evaluation. Against
a 10–20% run-to-run spread the phase figure is resolved and the total is
marginal.

`:delta4` remains the default. The battery says `:d8` is better and the
planar and cylindrical ceilings say so emphatically, but the general guidance
for converging shocks rests on the spherical case alone, and that is the case
`:d8` costs 40% of its timestep. The four constants are also still the δ⁴ fit;
a detector that changes the sensor's spatial support by this much has no claim
on them. Two things would settle it: `C_beta` refit under `:d8`, and an
account of the origin cell good enough to say whether the loss is the detector
or the third-order fold closure.

Both have since been done, and neither settles it the way this paragraph
expected. The [refit](#the-c_beta-refit-under-d8) retains `C_beta = 1.0` and
finds no value in 0.25–4 that recovers the origin. The
[account of the origin cell](#the-origin-cell-is-a-startup-transient) retires
the closure and the blind-sensor reading together: the closure is sixth to
seventh order at the fold, and β\* there reaches the line maximum during the
excursion that fails. What the two leave is a symmetry cell that evacuates
under a startup transient every configuration goes through, which is a
robustness question rather than a detector question. `:delta4` remains the
default because it survives that excursion at a larger timestep, not because
anything has been shown wrong with `:d8`.

#### The C_beta refit under `:d8`

`bench/artcal.jl beta` and `bench/artcal.jl beta detector=d8`, back to back on
one machine. The `:delta4` column is a fresh control rather than the
[C_beta table](#c_beta--the-shock-constant) above, and it reproduces the
`:delta4` row of the detector comparison in every column, so the two detectors
are separated by one variable.

```
                :delta4                                  :d8
C_beta   Noh1 deficit  Noh3 plat  contact  Shu     Noh1 deficit  Noh3 plat  contact  Shu
0.25        +56%        1.0098    0.0041  1.6408      +33%          NaN     0.0039  1.6489
0.5         +61%        0.9931    0.0045  1.6297      +44%          NaN     0.0041  1.6420
1.0   *     +64%        0.9751    0.0053  1.6180      +53%        0.9951    0.0044  1.6360
2.0         NaN           0.9569  0.0063  1.6055      +58%        0.9767    0.0046  1.6300
4.0         NaN           0.9408  0.0074  1.5945      +62%        0.9578    0.0051  1.6232
```

The viable window moves rather than the optimum inside it. Under `:delta4` the
window is bounded above by planar Noh, which fails at 2.0, and runs 0.25 to 1.0.
Under `:d8` it is bounded *below* by the two converging geometries, which fail
at 0.5 and at 0.25, and runs 1.0 to 4.0 without reaching an upper bound in the
sample. **The two windows intersect in the single value 1.0**, which is the
shipped one.

`:d8` also flattens the response to the constant on every smooth measure. Over
0.25 → 4 the contact broadens 31% under `:d8` against 80% under `:delta4`, the
Shu–Osher train loses 1.6% against 2.8%, and Lax L1 moves 4.7e-3 → 4.9e-3
against 4.8e-3 → 5.4e-3. Selectivity against resolved structure also makes the
resolved structure less sensitive to the magnitude of the constant, which is
the behaviour `:d8` was adopted for, observed here on the constant rather than
on the cases.

The CFL ladder answers the question the refit was run for:

```
             nu = 1    nu = 2                nu = 3
C_beta    :d4  :d8   :d4     :d8          :d4   :d8
0.25      1.0+ 1.0+  1.0     0.4 only     0.2   none
0.5       0.4  1.0+  0.4     >= 0.4       0.25  none
1.0  *    0.25 1.0+  0.2     1.0+         0.4   0.25
2.0       none 1.0+  none    0.4          0.3   0.3
```

No `C_beta` in the sample recovers the spherical origin under `:d8`. The best
available is 0.3 at `C_beta = 2`, still below the 0.4 that `:delta4` reaches at
the shipped constant, and it is bought at 2.33% ν = 3 plateau error against
0.49%, the ν = 2 ceiling falling from 1.0+ to 0.4, 0.4% of the Shu–Osher train
and five points of wall heating. **`C_beta = 1.0` is retained under `:d8`**, and
the detector decision falls entirely to the origin cell, as the cost section
above anticipated.

One further reading favours the detector and was not visible before the ladder
existed. At `C_beta = 1.0` the *worst* geometry improves under `:d8`, 0.2 to
0.25, because ν = 2 rises from 0.2 to beyond 1.0 while ν = 3 falls from 0.4 to
0.25. A user who does not know which geometry is ahead is better off under
`:d8`; the case against it is specifically the converging spherical one.

##### A failure that gets worse as the timestep falls

Two cells of the `:d8` ladder are not ceilings. At `C_beta = 0.25`, ν = 2
completes at cfl = 0.4 and fails at 1.0 and at everything from 0.3 down; at
`C_beta = 0.5` it completes at 1.0 and 0.4 and fails from 0.3 down. Both were
re-run against a probe separating the two ways `m_noh` returns NaN, and every
one of those cells is a genuine positivity loss rather than the step cap. The
sweep now prints the step cap as `Inf` so the distinction survives.

A CFL-type stability restriction cannot produce a failure that appears only
*below* a CFL. A per-*step* operation can, because a fixed physical interval
integrated at half the timestep applies it twice as many times. The sign of the
dependence is the evidence here; the accumulation itself has not been measured
step by step, which `bench/nohprobe.jl` could do.

The per-step operation in the loop is the compact filter, which runs once per
step at `filter_interval = 1` and is already recorded as removing energy per
application rather than per unit time, so that its effective dissipation depends
on the CFL number and does not converge as dt → 0 at fixed resolution. That is
[model debt 1](ROADMAP.md), arrived at from an unrelated measurement.

This was first written up as support for the fold closure, which
[has since been measured](#the-fold-closure-is-not-third-order) at sixth to
seventh order and retired. The observation stands; its attribution does not.

### The sensors read |S|, not the velocity and the dilatation

Cook builds μ\* and β\* from the strain magnitude |S| = sqrt(S_ij S_ij).
Miranda builds μ\* from `ringV(u, v, w)`, the ring of each velocity *component*
along each direction reduced by `MAX` over the nine pairs, and β\* from
`ring(∇·u)`. Neither reference field carries an absolute value.
`ArtParams.mu_sensor` (`:strain`, `:velocity`), `ArtParams.beta_sensor`
(`:ungated_dilatation` alongside the three already there) and
`ArtParams.reduction` (`:sum`, `:max`) now select between them. The weight is
h_d for a field carrying one velocity derivative fewer, against h_d² for |S|
and ∇·u, and the two coincide at the grid scale, so the four constants transfer
as starting points exactly as they do between detectors.

#### The field and the detector are not independent

`bench/artcal.jl response` puts a velocity sine of one wavelength on a periodic
64-point line and reads the peak coefficient back, with no time integration.
The comparison is against the detectors' own designed separation, a factor of
569 at eight points per wavelength, 26 at four, and 1 at the Nyquist:

```
k/pi   ppw  |  mu* from |S|:   d4       d8    ratio  |  mu* from u:    d4       d8     ratio
0.125  16.0 |             7.24e-5  4.20e-5   1.72    |            4.11e-6  4.18e-10  9.8e+03
0.250   8.0 |             3.31e-4  2.58e-4   1.28    |            4.71e-5  8.29e-08     569
0.500   4.0 |             2.44e-3  2.44e-3   1.00    |            3.93e-4  1.51e-05      26
0.750   2.7 |             8.33e-4  6.49e-4   1.28    |            1.60e-3  5.07e-04    3.16
1.000   2.0 |             0.00e+0  0.00e+0    ---    |            3.14e-3  3.14e-03       1
```

Applied to the velocity, the two detectors reproduce their designed ratios to
four figures at every wavelength. Applied to |S| they differ by a factor of 1.8
or less from 32 points per wavelength down to the Nyquist. The cause is the
absolute value: |S| has a cusp wherever the strain passes through zero, and a
cusp is grid-scale structure at any resolution, so both detectors receive
grid-scale content from a resolved field. β\* from ∇·u gives the right-hand
ratios to the digits printed, except in the last row. The 2.6e6 versus 1.8 gap
recorded under
[what the detector cannot help with](#what-the-detector-cannot-help-with) has
the same cause.

The last row is a property of every field obtained by differentiation. A
centered scheme has zero modified wavenumber at the Nyquist, so |S| and ∇·u
vanish identically for a two-point velocity wave and the sensors built from
them return zero there, whichever detector is used. Only the velocity-component
sensor responds, with the full undivided 16Ah. The calculation is stable in
spite of that, for the reason the C_mu section records at 128³: grid-scale
dissipation comes from the compact filter and not from the Cook properties.

#### Results on the battery

`bench/artcal.jl field`, shipped constants, both detectors, at the shipped CFL:

```
detector  mu*       beta*         Noh1 plat  deficit  Noh3 plat  Lax L1   contact  Shu train  WC peak
delta4*   strain*   strain*        0.9993     +64%      0.9751   5.0e-3   0.0053   1.6180     6.6049
delta4    velocity  strain         0.9993     +64%      0.9750   5.0e-3   0.0053   1.6180     6.6046
delta4    strain    ungated_dil    0.9993     +64%      0.9715   5.0e-3   0.0051   1.6181     6.5367
delta4    velocity  ungated_dil    0.9993     +64%      0.9714   5.0e-3   0.0051   1.6181     6.5361
d8        strain    strain         0.9997     +53%      0.9951   4.7e-3   0.0044   1.6360     6.3953
d8        velocity  strain         0.9997     +53%      0.9951   4.7e-3   0.0044   1.6368     6.3948
d8        strain    ungated_dil    0.9997     +51%      0.9976   4.6e-3   0.0041   1.6279     6.3005
d8        velocity  ungated_dil    0.9997     +51%      0.9975   4.6e-3   0.0041   1.6279     6.3002
```

**μ\* from the velocity components moves no column past the fourth digit.** The
largest movement in the table is the Shu–Osher train under `:d8`, 1.6360 →
1.6368, or 0.05%. Every case here is one-dimensional and C_mu is 0.002, so the
shear channel does very little in them; it is active in the Taylor–Green
measurement below.

**β\* from the dilatation improves four columns and degrades two.** Under `:d8`
the ν = 3 plateau error falls from 0.49% to 0.24%, ν = 1 wall heating from
+53% to +51%, the Lax contact sharpens 0.0044 → 0.0041 and its L1 falls
4.7e-3 → 4.6e-3, while the Shu–Osher train loses 0.5% (1.6360 → 1.6279) and
the Woodward–Colella peak 1.5% (6.3953 → 6.3005). Under `:delta4` the same
change gives up more and gains less: 0.4% of the ν = 3 plateau and 1.0% of the
Woodward peak, for the same contact gain.

#### The dilatation sensor loses the cylindrical axis

Highest CFL reaching `t_final` with a correct plateau:

```
                             nu = 1     nu = 2      nu = 3
detector  mu*       beta*     wall       axis       origin
delta4    strain    strain      0.2       0.2         0.4
delta4    velocity  strain      0.2       0.2         0.4
delta4    any       ungated_dil 0.2      none         0.2
d8        strain    strain      1.0+      1.0+        0.2
d8        velocity  strain      1.0+      1.0+        0.2
d8        any       ungated_dil 1.0+      0.2         0.2
```

The ungated form keeps the spherical origin and loses the cylindrical axis. No
CFL sampled reaches `t_final` at ν = 2 under `:delta4`, and under `:d8` the
axis ceiling falls from 1.0+ to 0.2. `:dilatation`, which is the same sensor
with the Ducros switch applied to it, loses both geometries. The spherical loss
is therefore attributable to the switch and the cylindrical loss to the sensor
field, a separation the earlier measurement could not make. The mechanism for
the cylindrical loss is unchanged from
[why the dilatation sensor loses the converging cases](#why-the-dilatation-sensor-loses-the-converging-cases):
Δ = S_rr + S_θθ adds two same-signed components in the first cells where |S|
takes their root-sum-square, so the sensor is 2 to 70 times larger where the
cell measure is smallest.

This bears on the detector result above. `:d8` removes the artificial-property
CFL restriction at the wall and the axis; pairing it with the reference's β\*
field restores the axis restriction.

#### The μ\* channel on Taylor–Green

No case in the battery exercises μ\*, so the field change for that channel is
measured where it carries a share of the sink. 64³, Re = 1600, to t = 10 under
the shipped smoother, dissipation split at the peak
(`bench/tgv_energy.jl 64 10.0 configs=on:1 smoother=gaussian mu_sensor=…`):

```
mu*        reduction   steps   t_peak   peak -dKE/dt   molecular   mu*    filter
strain*    sum*        5888     8.49      1.2459e-2      33.8%     4.5%   61.6%
velocity   sum         5884     8.49      1.2421e-2      33.6%     6.0%   60.4%
strain     max         5682     8.97      1.2496e-2      33.3%     2.7%   64.0%
```

β\* is 0.0% in all three, as in every Taylor–Green measurement here: at Ma 0.1
there is almost no dilatation for it to act on.

The velocity sensor raises the μ\* share by a third at the expense of the
filter's, and the directional maximum cuts it by 40%, which is what reducing
rather than summing three directions gives. Both are moves of one to two points
in a 4.5% channel, within a sink the filter dominates, so neither is
distinguishable from a rescaling of `C_mu`. Two further results are not
rescalings. The peak falls 0.3% under `:velocity`, toward the van Rees
reference of 1.2e-2, and the peak *time* under `:max` moves 8.49 → 8.97 against
a reference peak at t = 9, the other two settings being half a time unit early.

#### Cost

`bench/phases.jl` on the two-species tube, back to back on one machine:

```
setting                          artificial   % of RHS   compute_rhs!   line solves
strain / strain / delta4*         1.050 ms     26.0%       4.178 ms      24 + 0
mu_sensor = velocity              1.535 ms     33.7%       4.690 ms      24 + 0
beta_sensor = ungated_dilatation  1.287 ms     29.8%       4.349 ms      24 + 0
detector = d8                     1.733 ms     35.9%       4.798 ms      24 + 8
mu_sensor = velocity, d8          2.678 ms     47.1%       5.775 ms      24 + 14
```

The velocity sensor detects three fields where the strain sensor detects one,
which is +46% on the sensor phase under δ⁴. Paired with `:d8` it adds six more
pentadiagonal solves per right-hand side, and the sensor phase then costs more
than everything else in the evaluation combined. Against a 10–20% run-to-run
spread the phase figures are resolved and the totals are marginal.

#### The fourth-difference clamp at a fold

`delta4_sum!` extends the field past a closed edge by clamping the index, a
zeroth-order extension where the compact closures use the half-offset mirror.
The velocity sensor is the first sensor whose field is *odd* across a fold,
which is where the two extensions differ in order. For an even field the clamp
misplaces one δ⁴ tap by a term that the vanishing edge derivative makes O(h²);
for an odd field the edge derivative is the largest quantity there and the same
tap is wrong at O(h). The test case is u_r = r at the cylindrical axis, N = 32,
the regular behaviour of a radial velocity, which should produce no sensor at
all. The clamp gives μ\* = 1.2e-6 on the axis cell against 0 for the mirror,
or C_mu·ρ·h² of spurious viscosity on the cell where every converging case
fails. `delta4_sum!` therefore uses the mirror wherever the parity is −1 at a
folded edge. The two detectors then agree there, `:d8` reaching its own
half-offset closure through the fold plans and giving 7e-16 on the same field.

The even path is left on the clamp, so no shipped number moves. Its error is
two orders smaller in h, and changing it would move every guarded number in
`test/validation.jl`, a separate measurement that nobody has taken.

### The test filter is an explicit Gaussian, not a compact-filter pass

Cook's sensor smoothing is `gbar` in Miranda, which resolves to `cgfs4`
(`parcop/stencils.f90`): an explicit nine-point symmetric stencil with
`nol = 0` and `implicit_op = .false.`, weights 3565/10368, 3091/12960,
1997/25920, 149/12960 and 107/103680. Over the common denominator 103680 these
sum to exactly 1, so the filter preserves constants without relying on
cancellation, and at boundaries the overhanging weight is folded onto the
mirror point, which preserves the unit sum there as well. There is no linear
solve and no interface reduction; the operator needs a halo of four.

`smooth!` in `artificial.jl` substitutes one pass of `compact_filter(0.45)`.
The two are not interchangeable. Evaluating both transfer functions at
modified wavenumber k:

```
k/pi   Gaitonde-Visbal 0.45   Miranda 9-point Gaussian
0.125            1.000000                     0.902300
0.25             0.999972                     0.662818
0.375            0.999325                     0.396188
0.5              0.993750                     0.191821
0.625            0.965155                     0.073590
0.75             0.854020                     0.020747
0.875            0.491876                     0.003309
1.0              0.000000                     0.000000
```

As a Cook test filter the shipped smoother is close to the identity over the
whole resolved band, and its cost is a distributed line solve per active
dimension per sensor — `n_species` of them per RHS evaluation for the species
sensors. That cost is the ~31% of the multicomponent right-hand side recorded
in the known limitations. The explicit Gaussian would remove those solves and
their `Allgather`s while smoothing considerably more, so the two effects move
together rather than trading off.

`ArtParams.smoother` selects between them, and the converging cases answer the
question the transfer functions raise. Highest CFL reaching `t_final` with a
correct plateau, ladder extended until both settings fail:

```
Noh geometry              :compact   :gaussian
nu = 1  planar wall          0.2        0.2
nu = 2  cylindrical axis     0.15       0.2
nu = 3  spherical origin     0.15       0.4
```

Both fail at 0.5 everywhere, so 0.4 is measured rather than a table edge. The
origin, recorded under [Geometry limits](#geometry-limits) as the least
forgiving fold, becomes the most forgiving one. Cost moves the same way: on the
two-species tube of `bench/phases.jl` the `artificial` phase falls from 1.360 ms
to 0.971 ms, 31.8% to 24.8% of the right-hand side, against a run-to-run spread
of about 1.3% over three `:compact` readings.

The mechanism is
[sensor intermittency](#the-sensor-is-intermittent-at-the-damage-site), measured
rather than inferred. Accuracy is mixed and small by comparison, with wall
heating the one clear regression: the ν = 1 deficit moves +58% → +64%, while
plateaux and the Shu–Osher train move in the fourth digit.

Two refit diagnostics were run against that regression, and both are negative
in the useful sense of closing a question.

**κ\* cannot buy the wall heating back.** Sweeping `C_kappa` under each smoother,
ν = 1 wall deficit:

```
C_kappa      0      0.0025    0.01*     0.04     0.16
compact    +64%     +61%      +58%      +57%     +70%
gaussian   +65%     +64%      +64%      +68%     +91%
```

The lever inverts. Under `:compact` the trough is at 0.01–0.04, as recorded
under [C_kappa](#c_kappa--the-conductivity); under `:gaussian` the trough is
+64% and raising the constant makes wall heating worse, sooner and more steeply.
This follows from the same mechanism: the Gaussian widens the κ\* footprint and
lowers its peak, so added conductivity is spread across a region instead of
concentrated in the wall cell where the entropy error is deposited. The
regression is a property of the smoother, not a mis-set constant. Incidentally,
at `C_kappa = 0` the ν = 3 case fails under `:compact` and completes under
`:gaussian`, and the ν = 3 plateau is better under `:gaussian` across the whole
sweep, 0.9745–0.9822 against 0.9728–0.9796.

**Three dimensions are neutral.** Taylor–Green at 64³ to t = 10, dissipation
split at the peak:

```
smoother   t_peak   molecular   mu*     beta*   filter
compact     8.39      32.6%     4.0%     0.0%   63.5%
gaussian    8.45      33.8%     4.5%     0.0%   61.6%
```

The channel that `C_mu` controls moves from 4.0% to 4.5% of the sink and the
filter still dominates, so nothing here argues that `C_mu` needs refitting.
A full `C_mu` sweep under `:gaussian` has not been run; this is the weaker
statement that the 3-D budget does not shift enough to demand one.

### The conservative filter is normalized by a filtered cell volume

`filter` in `parcop/operators.f90` treats non-Cartesian coordinates by
filtering the volume-weighted field and dividing by a cell volume that has
been passed through **the same filter once at setup** (`CellVolG`, `CellVolS`
in `parcop/mesh.f90`). The filtered field then reproduces a constant exactly
on any non-uniform metric, and the integrated quantity is preserved. The
radial parity flips in the process, because the cylindrical cell volume is
proportional to r and therefore odd across the axis — the same algebra
`sigflux` encodes for the flux products in `rhs.jl`.

`filter_state!` filters the conserved components unweighted. On a uniform
Cartesian grid the two agree identically, which is why nothing in the
Taylor–Green numbers above would show it. On cylindrical, spherical or
stretched grids the filter is not conservative, and the filter supplies 37% of
the energy sink at 128³.

### The CFL rate is normalized differently

Miranda forms `Σ_d |u_d|/Δ_d` for advection and adds `|c|/min_d(Δ_d)` once for
the acoustic part, then takes the diffusive limits as separate minima with
their own coefficients (0.1 and 0.2). `max_rate` sums `(|u_d| + c)/h_d` over
active dimensions and folds the diffusive rate into the same sum.

The sound speed is therefore counted once there and once per active dimension
here. On an isotropic three-dimensional grid the rate computed here is up to
three times larger for the same state, so `cfl = 0.15` corresponds to a step
comparable to `cfl ≈ 0.4` under the reference convention. This matters for
comparing the numbers in this document against the literature; it does **not**
explain away [the ceiling](#cfl-which-dominates-all-four), because the cases
that establish it are one- and two-dimensional converging geometries, where
the two conventions largely agree.

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
Cartesian, so this is specific to the origin fold and its antipodal pairing.
The explanation offered here was that the fold's closure is third order, which
[is not the case](#the-fold-closure-is-not-third-order): the fold is sixth to
seventh order and the most accurate region of the line. The limit itself is
measured and stands; why the origin fold is less forgiving than the cylindrical
axis remains open, and is item 5 below. `test/cases.jl` therefore initializes
Sedov with a Gaussian deposit.

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
| `C_beta` | 1.0 | yes | Accuracy optimum near 0.4; use 0.5 for interface-dominated work and avoid zero. The constant trades the planar and cylindrical CFL ceilings against the spherical one, and 1.0 is the value that maximizes the spherical one. Refit under `:d8`, where the viable window moves from 0.25–1.0 up to 1.0–4.0 and intersects the `:delta4` window only at the shipped value. [Measurements](#the-c_beta-refit-under-d8). |
| `C_kappa` | 0.01 | yes | 0.02–0.04 measurably reduces wall heating; avoid zero. |
| `C_D` | 0.01 | yes | The compact filter dominates interface broadening, so sensitivity to `C_D` is weak. |
| `mu_sensor` | `:strain` | yes | `:velocity` is the reference field and recovers the detector's full designed selectivity, which `:strain` destroys through the cusps of \|S\|; it then moves no column of the battery past the fourth digit, because no case there gives the shear channel anything to do. On Taylor–Green it raises the μ\* share of the sink 4.5% → 6.0% for +46% on the sensor phase. [Measurements](#the-sensors-read-s-not-the-velocity-and-the-dilatation). |
| `beta_sensor` | `:strain` | yes | `:gated_strain` raises the cylindrical Noh CFL ceiling from 0.15 to 0.2 for one pointwise pass and is otherwise a wash; `:dilatation` buys 0.25% of the Shu–Osher wave train and loses both converging Noh geometries at the fold; `:ungated_dilatation`, the reference form, keeps the origin and still loses the axis, improves the ν = 3 plateau and the Lax contact, and costs 0.5% of the Shu–Osher train and 1.5% of the Woodward peak. |
| `reduction` | `:sum` | yes | `:max` is the reference's directional reduction. Identical in one dimension, so the whole battery is blind to it; on Taylor–Green it cuts the μ\* share 4.5% → 2.7% and moves the dissipation peak from t = 8.49 to 8.97 against a reference peak at t = 9. |
| `smoother` | `:gaussian` | yes | Changed from `:compact` in August 2026. Raises the spherical-origin CFL ceiling 0.15 → 0.4 and the cylindrical 0.15 → 0.2, cuts the sensor phase 29%, and improves every Noh plateau and pre-shock L1. Costs wall heating at ν = 1 (+58 → +64%) and ν = 2 (+41 → +56%), improves it at ν = 3 (+31 → +27%), and moves the two stored cases by 1–3% of their L1. `C_kappa` cannot recover the wall heating. |
| `detector` | `:delta4` | provisionally | `:d8` improves six of seven battery columns and removes the artificial-property CFL restriction outright at the planar wall and the cylindrical axis (0.2 → 1.0+), at 40% of the spherical-origin timestep (0.4 → 0.25) and +19% on the right-hand side. The `C_beta` refit is [done](#the-c_beta-refit-under-d8): 1.0 is retained, no value in 0.25–4 recovers the origin, and the worst geometry improves 0.2 → 0.25 under `:d8`. Held at `:delta4` on the origin cell alone. [Measurements](#the-ringing-detector-is-an-eighth-derivative-not-a-fourth-difference). |
| `cfl` | 0.5 | **no** | Use 0.3 with shocks and `StepControl(retries = 4)` for recovery. Converging geometry now tolerates 0.4 (spherical) and 0.2 (cylindrical) under the default smoother. Note that under `filter_cfl = 0` the CFL is not only a stability choice: it sets how much subgrid dissipation the filter supplies, and moves the μ\* share of the Taylor–Green sink by 29% relative across a factor of two. |
| `filter_cfl` | 0.0 | provisionally | Reference CFL for a full-strength filter pass. The default is the unrelaxed formulation, in which the filter dissipates per application and its subgrid dissipation does not converge as `dt → 0`. A positive value makes it a rate: filter loss is constant to six figures across a 4× CFL change, and the Taylor–Green channel split holds to 0.1 points where it otherwise moves 2.8. Held at 0 pending the α and cadence fit, which is the other half of the same debt and requires the 3-D campaign. [Measurements](#the-filter-dissipates-per-application-not-per-unit-time). |

Remaining items, in approximate priority order:

1. Raise the CFL ceiling at the symmetry plane. This is the observed
   restriction, and [the measurement above](#where-the-restriction-originates)
   places it at the wall, axis or origin cell rather than ahead of the front.
   A timestep predictor, a larger `C_beta`, and a dilatation-built β\* have all
   been measured against it and none moves the ceiling; gating the existing
   sensor on compression (`:gated_strain`) moves it for the cylindrical
   geometry only, 0.15 to 0.2, and does so by relieving the axis cell. A
   *smaller* `C_beta` does move it, which the one-sided earlier test missed:
   0.25 lifts ν = 1 past 1.0 and ν = 2 to 1.0 while cutting ν = 3 to 0.2, so
   the constant trades the ceilings against each other rather than raising them
   ([the ladder](#the-cfl-ladder-per-constant)). Widening
   the sensor stencil was the standing hypothesis and is now ruled out: β\*
   already reaches three to five times further ahead of the front than the
   damage extends, and sits at or near its own domain maximum on the cell that
   fails. **Largely addressed by `smoother = :gaussian`**, which moves the ν = 3
   ceiling 0.15 → 0.4 and the ν = 2 ceiling 0.15 → 0.2 while making the sensor
   phase cheaper; the mechanism is
   [sensor intermittency](#the-sensor-is-intermittent-at-the-damage-site) and the
   remaining work is the refit under item 6 below rather than a further search.
   ν = 1 is unmoved at 0.2. **The eighth-derivative detector has since been
   measured** and moves ν = 1 and ν = 2 from 0.2 to 1.0 or beyond, which is the
   restriction lifted rather than raised, while moving ν = 3 from 0.4 to 0.25.
   **The fold closure, which stood as the last candidate, has since been
   measured and retired.** It is sixth to seventh order at the fold and three to
   five orders of magnitude more accurate than the interior; the third-order
   number attributed to it belongs to the outer wall, through a global max norm
   ([the measurement](#the-fold-closure-is-not-third-order)). The companion
   reading, that a selective detector is blind at the fold, is retired by the
   same pass: β\* at the origin reaches the *line maximum* during the excursion
   that fails ([the probe](#the-origin-cell-is-a-startup-transient)).

   What the origin cell actually does is evacuate during a startup transient
   that lands at t ≈ 0.394 regardless of resolution and that every
   configuration passes through, surviving or not. No discretization-order
   candidate remains. The two live leads are the density proportionality of
   β\*, which suppresses the regularization of exactly the cell that is
   thinning, and the per-step compact filter, which the `C_beta` ladder
   implicates from another direction: under `:d8` at reduced `C_beta` the
   cylindrical axis fails *below* a CFL rather than above one
   ([the measurement](#a-failure-that-gets-worse-as-the-timestep-falls)), and a
   per-step operation is what produces that sign. Both leads point at item 4
   and at model debt 2 rather than at the numerics of the fold.
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
   `compact_filter(0.45)` every step. **The dt-consistency half of this is
   done**: the filter dissipates per application rather than per unit time, the
   effect is a factor of 3.93 across a 4× CFL change, and `filter_cfl` makes it
   a rate ([measurements](#the-filter-dissipates-per-application-not-per-unit-time)).
   What remains is fitting α and the cadence against a reference dissipation
   history, which requires the 3-D campaign. Note that the two are coupled: a fit
   taken under the unrelaxed formulation is only reproducible at the CFL it was
   taken at.
5. Determine why the spherical-origin fold is less tolerant of
   under-resolved data than the cylindrical axis fold.
6. Refit under the adopted smoother, and decide the detector.
   `smoother = :gaussian` is the shipped default as of August 2026; what it did
   not come with is a refit. `C_beta` has since been swept under it and under
   `:d8`, retaining 1.0 in both ([measurements](#the-c_beta-refit-under-d8)); a
   `C_mu` sweep under the Gaussian has still not been run, and `C_D` was not
   swept because the filter dominates interface broadening. The detector
   decision no longer waits on the constant. It waits only on the origin cell
   of item 1, since no `C_beta` in 0.25–4 recovers the spherical ceiling under
   `:d8`. The sensor
   fields are a third setting in the same class and the clearest case for it:
   `mu_sensor = :velocity` moves the μ\* share of the Taylor–Green sink by a
   third with `C_mu` held fixed, so what it is worth cannot be read off until
   the constant is refitted under it.
7. Make `filter_state!` conservative on non-Cartesian metrics, by the
   filtered-cell-volume normalization recorded in the same section. Uniform
   Cartesian results are unaffected by construction, so this is measurable
   against the converging cases alone.
8. Put `delta4_sum!`'s even path on the half-offset mirror as well. The clamp
   it uses instead is second-order-consistent for an even field, so this is
   expected to be small, but it is a spurious sensor at the fold cell where
   every converging case fails and the size of it has not been measured. The
   odd path already uses the mirror,
   [for a reason worth reading first](#the-fourth-difference-clamp-at-a-fold).
