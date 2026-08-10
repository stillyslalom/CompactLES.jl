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
6. [The β\* sensor — strain, gated, or dilatation](#the-beta-sensor--strain-or-dilatation)
7. [CFL, which dominates all four](#cfl-which-dominates-all-four)
8. [Measured against the reference implementation](#measured-against-the-reference-implementation)
9. [Grid convergence](#grid-convergence)
10. [Geometry limits](#geometry-limits)
11. [Recommendations](#recommendations)

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

<a id="the-beta-sensor--strain-or-dilatation"></a>

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
than the artificial properties. This is consistent with `cfl ≤ 0.15` binding in
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
below carries the measurement. The fold closure itself remains untested.

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
is the concrete case for extending item 3 of the model debts, the positivity
failsafe, to cover internal energy rather than density alone.

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

## Measured against the reference implementation

Miranda's Fortran kernels, carried by Pyranda in `pyranda/parcop/`, implement
the same Cook artificial-property method. Reading them against `artificial.jl`
identifies four differences that bear on the constants calibrated above. None
of the four has been measured here; this section records what the reference
does and what is analytically established about the gap, so that the
investigation does not start from source archaeology a second time.

### The ringing detector is an eighth derivative, not a fourth difference

`ring()` in `parcop/operators.f90` dispatches to `d8x/d8y/d8z`, a full compact
operator (`c10d8`, `parcop/stencils.f90`) with a pentadiagonal left-hand side,
a nine-point right-hand side, and its own symmetric and antisymmetric boundary
closures. `artificial.jl` instead uses the undivided fourth difference
δ⁴ = (1, −4, 6, −4, 1), following Cook (2007) literally.

The fields differ too. Miranda builds μ\* from `ringV(u, v, w)`, the ring of
each velocity *component* along each direction reduced by `MAX` over the nine
combinations, and β\* from `ring(∇·u)`. This code builds both from δ⁴|S|
summed over directions.

Two consequences follow analytically. An eighth derivative has negligible
response below the grid scale, whereas a fourth difference of |S| responds to
smooth curvature in the strain field, which is the behaviour
[the ceiling measurement](#where-the-restriction-originates) recorded when it
found β\* reaching three to five times further ahead of the front than the
damage extends. And `MAX` over directions does not accumulate with dimension
where `Σ_d` does.

Note also that Miranda's default β\* is `gbar(|ring(∇·u)| ρ)` with **no**
compression switch. The `:dilatation` variant rejected above is a dilatation
sensor *plus* the switch. Dilatation sensor without the switch, which is what
the reference actually ships, has not been tried.

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
| `beta_sensor` | `:strain` | yes | `:gated_strain` raises the cylindrical Noh CFL ceiling from 0.15 to 0.2 for one pointwise pass and is otherwise a wash; `:dilatation` buys 0.25% of the Shu–Osher wave train and loses both converging Noh geometries at the fold. |
| `smoother` | `:gaussian` | yes | Changed from `:compact` in August 2026. Raises the spherical-origin CFL ceiling 0.15 → 0.4 and the cylindrical 0.15 → 0.2, cuts the sensor phase 29%, and improves every Noh plateau and pre-shock L1. Costs wall heating at ν = 1 (+58 → +64%) and ν = 2 (+41 → +56%), improves it at ν = 3 (+31 → +27%), and moves the two stored cases by 1–3% of their L1. `C_kappa` cannot recover the wall heating. |
| `cfl` | 0.5 | **no** | Use 0.3 with shocks and `StepControl(retries = 4)` for recovery. Converging geometry now tolerates 0.4 (spherical) and 0.2 (cylindrical) under the default smoother. |

Remaining items, in approximate priority order:

1. Raise the CFL ceiling at the symmetry plane. This is the observed
   restriction, and [the measurement above](#where-the-restriction-originates)
   places it at the wall, axis or origin cell rather than ahead of the front.
   A timestep predictor, a larger `C_beta`, and a dilatation-built β\* have all
   been measured against it and none moves the ceiling; gating the existing
   sensor on compression (`:gated_strain`) moves it for the cylindrical
   geometry only, 0.15 to 0.2, and does so by relieving the axis cell. Widening
   the sensor stencil was the standing hypothesis and is now ruled out: β\*
   already reaches three to five times further ahead of the front than the
   damage extends, and sits at or near its own domain maximum on the cell that
   fails. **Largely addressed by `smoother = :gaussian`**, which moves the ν = 3
   ceiling 0.15 → 0.4 and the ν = 2 ceiling 0.15 → 0.2 while making the sensor
   phase cheaper; the mechanism is
   [sensor intermittency](#the-sensor-is-intermittent-at-the-damage-site) and the
   remaining work is the refit under item 6 below rather than a further search.
   ν = 1 is unmoved at 0.2, and the fold closure is still untested, that is,
   whether the third-order closure at the axis and origin sets the step size
   there. The eighth-derivative detector of
   [the reference comparison](#measured-against-the-reference-implementation)
   remains open and must be measured on top of the Gaussian, never against the
   compact smoother.
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
6. Decide whether `smoother = :gaussian` becomes the default. The measurements
   are in
   [the reference comparison](#measured-against-the-reference-implementation):
   it buys a 0.15 → 0.4 ceiling at the spherical origin, 0.15 → 0.2 at the
   cylindrical axis, and a 29% cheaper sensor phase, and it costs +57 → +64% of
   ν = 1 wall heating, which `C_kappa` has been measured unable to recover.
   Taylor–Green is neutral. What remains open is a `C_mu` sweep under the new
   smoother and whether the wall-heating cost is acceptable against the
   converging-geometry gain, which is a judgement about the intended workload
   rather than a measurement. `C_beta` and `C_D` were not swept: the former has
   a broad optimum and an upper bound that is a stability bound which has just
   moved favourably, the latter is documented as weakly sensitive because the
   filter dominates interface broadening.
7. Make `filter_state!` conservative on non-Cartesian metrics, by the
   filtered-cell-volume normalization recorded in the same section. Uniform
   Cartesian results are unaffected by construction, so this is measurable
   against the converging cases alone.
