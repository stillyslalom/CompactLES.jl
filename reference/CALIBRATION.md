# Calibrating the artificial fluid properties

This file records the measured behaviour of the artificial-property constants
and of the sensor, smoother, detector and filter settings around them, together
with the default each measurement supports.

`bench/artcal.jl` reproduces the one-dimensional results using the cases in
`test/cases.jl`, which `test/validation.jl` guards. `bench/tgv_energy.jl`
produces the Taylor–Green results, `bench/filterrate.jl` the filter-rate
results, `bench/nohprobe.jl` the per-step Noh diagnostics, `bench/phases.jl` the
phase costs, `bench/foldorder.jl` the region-split convergence orders, and
`bench/boundaryorder.jl` the wall-flux probe.

Current defaults:

```
ArtParams(C_mu = 0.002, C_beta = 1.0, C_kappa = 0.01, C_D = 0.01, C_Y = 100,
          mu_sensor = :strain, beta_sensor = :strain, reduction = :sum,
          smoother = :gaussian, detector = :delta4, species_flux = :fickian)
```

For strong shocks the CFL number determines stability more strongly than any
constant in that list.

## Contents

1. [Defaults and their basis](#defaults-and-their-basis)
2. [The battery](#the-battery)
3. [C_beta, the shock constant](#c_beta-the-shock-constant)
4. [C_kappa, the conductivity](#c_kappa-the-conductivity)
5. [C_mu, the shear viscosity](#c_mu-the-shear-viscosity)
6. [C_D, the species diffusivity](#c_d-the-species-diffusivity)
7. [C_Y, the mass-fraction bound](#c_y-the-mass-fraction-bound)
8. [The sensor smoother](#the-sensor-smoother)
9. [The ringing detector](#the-ringing-detector)
10. [The sensor fields and the compression switch](#the-sensor-fields-and-the-compression-switch)
11. [The compact filter](#the-compact-filter)
12. [CFL and the symmetry-cell restriction](#cfl-and-the-symmetry-cell-restriction)
13. [Wall closures under the artificial properties](#wall-closures-under-the-artificial-properties)
14. [Fold order and geometry limits](#fold-order-and-geometry-limits)
15. [Grid convergence](#grid-convergence)
16. [Miranda's set as a package](#mirandas-set-as-a-package)
17. [The bulk species channel](#the-bulk-species-channel)
18. [Remaining differences from the reference implementation](#remaining-differences-from-the-reference-implementation)
19. [The no-slip wall flux contract](#the-no-slip-wall-flux-contract)
20. [Open items](#open-items)

## Defaults and their basis

| Setting | Default | Keep? | Basis |
|---|---|---|---|
| `C_mu` | 0.002 | yes | [TGV at 128³](#c_mu-the-shear-viscosity) is consistent with it but does not select it: the peak carries a 6% residual the coefficient does not control. Above 0.008 spherical Noh fails. |
| `C_beta` | 1.0 | yes | [Accuracy optimum near 0.4](#c_beta-the-shock-constant); 1.0 maximizes the spherical-origin CFL ceiling and is the only value viable under both detectors. |
| `C_kappa` | 0.01 | yes | [0.02–0.04 lowers wall heating under `:compact`](#c_kappa-the-conductivity) but not under the default smoother; zero loses spherical Noh. |
| `C_D` | 0.01 | yes | [The compact filter dominates interface broadening](#c_d-the-species-diffusivity); a 64× sweep moves the width 22%. |
| `C_Y` | 100 | yes | [The mass-fraction bound](#c_y-the-mass-fraction-bound): a shocked 2h interface rings to ±0.2 without it and ±0.013 with it. |
| `Y_tolerance` | 1e-4 | yes | [A dead band](#the-dead-band) restoring the unbounded convergence order on smooth profiles that touch 0 or 1. |
| `mu_sensor` | `:strain` | yes | [`:velocity` is the reference field](#the-sensor-fields-and-the-compression-switch) and moves no battery column past the fourth digit, for +46% on the sensor phase. |
| `beta_sensor` | `:strain` | yes | [`:gated_strain`, `:dilatation` and `:ungated_dilatation`](#the-sensor-fields-and-the-compression-switch) each cost a converging geometry or gain nothing measurable. |
| `reduction` | `:sum` | yes | [`:max` is the reference reduction](#the-mu-channel-on-taylor-green); identical in one dimension, and on TGV it cuts the μ\* share 4.5% → 2.7%. |
| `smoother` | `:gaussian` | yes | [Raises the ν = 3 CFL ceiling 0.15 → 0.4 and ν = 2 0.15 → 0.2](#the-sensor-smoother), cuts the sensor phase 29%, costs wall heating. |
| `detector` | `:delta4` | provisionally | [`:d8` improves six of seven battery columns](#the-ringing-detector) and removes the ν = 1 and ν = 2 restrictions, at 40% of the ν = 3 timestep and +19% on the RHS. |
| `species_flux` | `:fickian` | provisionally | [The bulk channel](#the-bulk-species-channel) is at least as good on every measured column, but the four constants are fitted on the Fickian one. |
| `cfl` | 0.5 | **no** | [Use 0.3 with shocks](#cfl-and-the-symmetry-cell-restriction) and `StepControl(retries = 4)`. Converging geometry tolerates 0.4 (ν = 3) and 0.2 (ν = 2). |
| `filter_cfl` | 0.0 | provisionally | [The unrelaxed formulation](#the-compact-filter) dissipates per application. A positive value makes it a rate and [removes the CFL dependence](#the-relaxation-leg); held at 0 until the relaxed formulation has been run through the shock battery at its production CFL numbers. |
| `filter_interval` | 1 | yes | [Redundant with α](#cadence-and-alpha-are-one-axis): diluting the filter in time is equivalent to weakening the pass. Every constant above is conditional on `compact_filter(0.45)` applied every step. |
| `compact_filter` α | 0.45 | **no** | Too strong at 128³ and 256³. The [128³ fit](#the-alpha-sweep-at-128) gives 0.49, which [does not transfer to 256³](#the-256-confirmation); 0.49 completes the [shock battery](#the-battery-under-alpha). Not moved because every constant above was fitted under 0.45. |

## The battery

The one-dimensional cases are defined in `test/cases.jl` and shared with
`test/validation.jl`, so the calibration study and the regression guards cannot
drift apart.

**ν is the Noh geometry index**: ν = 1 planar (a slip wall), ν = 2 the
cylindrical axis fold, ν = 3 the spherical origin fold. It is one more than the
exponent in the r^(ν−1) area weight, and the exact post-shock compression is 4^ν
at γ = 5/3.

Columns:

- **Noh plat/exact**: post-shock density plateau over its exact value (4, 16, 64
  for ν = 1, 2, 3). The exact ratio is 1.0000, which makes this the most precise
  accuracy measure in the set.
- **deficit**: density shortfall at the symmetry point relative to the exact
  plateau. This is wall heating, spurious entropy deposited where the flow
  stagnates.
- **Lax L1 / contact**: mean density error against the exact Riemann solution,
  and the 10–90% width of the contact discontinuity, which measures the
  broadening the regularization introduces.
- **Shu train amp**: peak-to-trough density in the post-shock entropy wave train.
  This is the quantity the tenth-order interior scheme exists to preserve, and it
  pulls against every constant that adds dissipation.
- **WC peak**: peak density in the Woodward–Colella collision, a survival check
  at a 10⁵ pressure ratio.
- **NaN**: the calculation lost positivity or stalled. Losing positivity drives
  the diffusive rate in `compute_dt` up until `dt` collapses, so a bad
  configuration costs wall time rather than failing quickly; every case carries
  an `nmax`. `StepControl` raises `SolverFailure` in `run!` about 150 steps
  before the timestep collapse.

Every constant table is one-dimensional. Because `C_mu` multiplies the shear
component of the artificial stress, these cases reach it only through the stress
trace, so the `C_mu` table constrains stability and not shear accuracy; the
three-dimensional Taylor–Green study in that section carries the shear
measurement. `C_beta`, `C_kappa` and `C_D` are fully exercised.

Every grid here is uniform and Cartesian, or has its angular directions
collapsed, so the physical and computational spacings coincide. `h_d` in a
sensor expression is the local physical spacing along direction d.

## C_beta, the shock constant

Measured under the default smoother at `NOH_CFL = 0.15`:

```
C_beta    | Noh1 plat  deficit | Noh3 plat | Lax L1  contact | Shu train | WC peak
0         |     NaN       NaN  |      NaN  | 5.7e-3   0.0032 |    1.6610 |    NaN
0.25      |  0.9995       +56% |   1.0098  | 4.8e-3   0.0041 |    1.6408 | 6.4151
0.5       |  0.9994       +61% |   0.9931  | 4.9e-3   0.0045 |    1.6297 | 6.4742
1.0    *  |  0.9993       +64% |   0.9751  | 5.0e-3   0.0053 |    1.6180 | 6.6049
2.0       |     NaN       NaN  |   0.9569  | 5.2e-3   0.0063 |    1.6055 | 6.6859
4.0       |     NaN       NaN  |   0.9408  | 5.4e-3   0.0074 |    1.5945 | 6.8366
```

The two ends of the range fail for different reasons. At `C_beta = 0` there is no
shock regularization and every strong-shock case loses positivity: Noh in both
converging geometries and Woodward–Colella. The Lax tube survives because the
compact filter alone suffices at that shock strength, and its contact is the
narrowest in the table because artificial bulk viscosity does not broaden it. At
the upper end `C_beta = 2` improves the Woodward peak but planar Noh does not
complete, so the upper bound is a stability bound and moves with the CFL.

Between the limits the accuracy measures vary monotonically. Contact width grows
roughly linearly, 0.0041 to 0.0074 over 0.25 to 4. The Shu–Osher wave train loses
2.8% of its amplitude over the same range, which measures overdamping of smooth
entropy waves. The Noh ν = 3 plateau degrades from 1.0098 to 0.9408, crossing
exact between 0.25 and 0.5.

Accuracy alone gives an optimum near 0.4. The default of 1.0 buys a robustness
margin at the cost of an 18% wider contact and 0.7% lower wave-train amplitude
relative to 0.5.

### The CFL ladder

The table above is read at a fixed CFL, which hides the dominant interaction for
converging geometry. Highest CFL reaching `t_final` with a correct plateau,
from `bench/artcal.jl beta`:

```
C_beta      nu = 1     nu = 2     nu = 3
0.25         1.0+       1.0        0.2
0.5          0.4        0.4        0.25
1.0    *     0.25       0.2        0.4
2.0          none       none       0.3
```

`C_beta` trades the planar and cylindrical ceilings against the spherical one
monotonically. Lowering it to 0.25 removes the restriction at the wall and the
axis and costs the origin a factor of two; raising it does the reverse. The
default is the value that maximizes the spherical ceiling, which is the geometry
the general guidance for converging shocks rests on. `detector = :d8` makes the
same trade in the same direction, and the two are refitted together
[below](#the-c_beta-refit-under-d8).

**Recommendation:** retain 1.0. Use 0.5 for interface-dominated
Richtmyer–Meshkov or Rayleigh–Taylor work with moderate shocks, noting that it
costs the spherical origin half its timestep. Values below 0.25 or above 2 are
not recommended.

## C_kappa, the conductivity

Measured under `smoother = :compact`:

```
C_kappa   | Noh1 plat  deficit | Noh3 plat | Lax L1  contact | WC peak
0         |  0.9998       +64% |      NaN  | 4.8e-3   0.0051 | 6.6847
0.0025    |  0.9997       +61% |   0.9728  | 4.9e-3   0.0051 | 6.6390
0.01   *  |  0.9992       +58% |   0.9732  | 5.0e-3   0.0051 | 6.5731
0.04      |  0.9980       +57% |   0.9748  | 5.3e-3   0.0052 | 6.4934
0.16      |  0.9986       +70% |   0.9796  | 5.8e-3   0.0052 | 6.3176
```

Two effects separate. Wall heating falls from 64% to 57% as `C_kappa` goes from 0
to 0.04, then rises to 70% at 0.16: artificial conduction transports spuriously
deposited entropy out of the stagnation cell, and excessive conduction adds its
own error. Robustness is the second: spherical Noh does not complete at
`C_kappa = 0`, so regularizing the momentum equation through `C_beta` does not by
itself carry a converging strong shock.

Lax L1 grows from 4.8e-3 to 5.8e-3 across the range while the contact width moves
from 0.0051 to 0.0052, because κ\* diffuses temperature and the Lax contact is
nearly isothermal.

Under the default `:gaussian` smoother the wall-heating lever inverts and the
trough sits at the default value; the sweep is in [the smoother
section](#the-sensor-smoother).

**Recommendation:** retain 0.01. Zero is not recommended.

### The cold-state limit

κ\* is built as `C_kappa · (ρc/T_ion) · sensor`, which is singular as T_ion → 0.
In an ambient below p ≈ 1e-3 at ρ₀ = 1 the 1/T factor drives the diffusive rate
up, `dt` collapses, and internal energy can go negative; the subsequent T_ion
clamp then produces an extremely large κ\*. Every case therefore uses a finite
ambient pressure. `artificial_conductivity_scale` is an EOS dispatch point, so a
tabular or condensed-matter model can supply a scale that is finite at its own
cold limit. The gas models still divide by the temperature.

## C_mu, the shear viscosity

```
C_mu      | Noh1 plat  deficit | Noh3 plat | Lax L1  contact | Shu train
0         |  0.9992       +58% |   0.9731  | 5.0e-3   0.0051 |    1.6183
0.0005    |  0.9992       +58% |   0.9732  | 5.0e-3   0.0051 |    1.6182
0.002  *  |  0.9992       +58% |   0.9732  | 5.0e-3   0.0051 |    1.6185
0.008     |  0.9992       +58% |      NaN  | 5.0e-3   0.0052 |    1.6214
0.032     |  0.9992       +58% |      NaN  | 5.0e-3   0.0052 |    1.6206
```

In one dimension μ\* is nearly inert: every accuracy column is flat to four
digits across a 64× sweep, because the shear viscosity reaches the solution only
through the trace of the stress, where β\* dominates by a factor of 500. The
table establishes one bound, that `C_mu` above about 0.008 destabilizes the
spherical origin, and no accuracy lower bound.

### Taylor–Green

`bench/tgv_energy.jl` runs TGV at Re = 1600 and splits −dKE/dt by mechanism.
The table below is the split the script printed before September 2026:
molecular stress, artificial shear μ\*, artificial bulk β\*, and a residual,
−dKE/dt − (mol + μ\* + β\*), labelled `filter`. That residual contains the
pressure work and the numerical error as well as the filter's dissipation. The
script now measures the filter pass directly and prints the pressure work as a
separate channel ([the measured budget](#the-measured-budget)); the tables in
this file that predate the change are on the residual definition. Channel
shares at the dissipation peak, artificial properties on, `filter_interval = 1`.
Every row but 128³ uses the Gaussian smoother; that campaign predates its
adoption and ran under `:compact`:

```
resolution   peak -dKE/dt        vs reference   molecular   mu*    beta*   filter
32³          1.4216e-2 @ 6.58      +16.4%        12.6%     5.1%    ~0%     82.2%
64³          1.2459e-2 @ 8.49        —           33.8%     4.5%    0.0%    61.6%
128³         1.2044e-2 @ 9.06       −6.0%        60.4%     2.3%    0.0%    37.3%
256³         1.3043e-2 @ 8.84       +1.6%        86.4%     0.8%    0.0%    12.8%
```

The reference peak is 1.28575e-2 at t = 8.97, read from the tabulated 512³
pseudo-spectral solution now vendored at `data/spectral_Re1600_512.gdiag`
([provenance](../data/README.md#taylor-green-reference-solution)). Each
percentage above is against that solution seen through the same boxcar the run
was measured with, since the two estimators have to match
([the window](#read-the-rate-over-a-window)): 1.2210e-2 at 32³, 1.2809e-2 at
128³, 1.2844e-2 at 256³, the widths falling with the step count. The 64³ row's
step count was not recorded, so its window cannot be reconstructed. Against the
raw tabulated peak it is 3.1% low, and since the window always lowers the
reference, its windowed figure is above that: −3.1% is the most negative value
the row can take. β\* is negligible at every resolution because
dilatation is negligible at Ma 0.1, which agrees with the role of `C_beta` as a
shock parameter.

These comparisons were previously made against the rounded `1.2e-2 at t = 9`
quoted from the figure, which is 6.7% below the tabulated peak. The rounding
error is larger than most of the differences it was used to judge, and
correcting it reverses two of the earlier readings: 128³, previously described
as within 1% of the reference, is the worst of the resolved rows, and 256³ is
the best. The trend with resolution is not monotone, since 64³ is no worse than
−3.1% and 128³ is −6.0%. The rows also differ in smoother, backend and machine,
128³ having run under `:compact` on rzhound and 256³ under the Gaussian
smoother on MI300A, so a resolution effect cannot be separated from a
configuration effect here.

The 128³ runs used 224 ranks over two rzhound nodes at 20–25 minutes each. The
256³ run was the first production-scale run on the GPU target: 4 MI300A APUs,
`backend=amdgpu`, `flux run -N1 -n4` at `-t 1`, 24,490 steps in 3.69 h, carrying
the device stall cost documented in `reference/AMR_GPU.md`.

<a id="the-measured-budget"></a>

### The measured budget

`bench/tgv_energy.jl` measures five channels of −dKE/dt: molecular
dissipation, μ\*, β\*, the compact filter, and the reversible pressure work
∫p∇·u. The remainder is printed as `unattr`.

The filter column of every table above this section is a different quantity:
the residual −dKE/dt − (mol + μ\* + β\*), which contains the pressure work,
the aliasing and dispersion error and the time-integration error together with
the filter's dissipation. The two definitions are not comparable. A table that
reports μ\*/filter ratios or filter shares without an `unattr` column beside
them is on the residual definition.

The filter's dissipation cannot be measured on the state a callback sees,
because that state has already been filtered in the current step. A second pass
removes, at each wavenumber, the first pass's loss scaled by the square of the
transfer function, and so understates the loss most where the filter acts.
`filter_loss` therefore advances a copy of the state by one step and measures
the kinetic energy removed by a `filter_state!` pass on that copy, which is the
loss the run incurs at its next step.

At 32³ to t = 10 under `:compact`, at the peak:

```
             mol     mu*    beta*   filter   p*divu   unattr
measured    12.1%    4.7%    0.0%    84.1%     0.4%    -1.3%
residual    12.1%    4.7%    0.0%    83.1%       -        -
```

The residual was one point low here, so at 32³ it was an adequate proxy.
Whether it remains one at higher resolution depends on how the absorbed error
scales, which has not been measured. If the error stays near one point of
−dKE/dt while the filter's share falls to 12.8% at 256³, the correction is
about 1% of the channel at 32³ and about 8% at 256³. The 128³ and 256³ rows
have not been remeasured.

The −1.3% `unattr` means that the measured channels slightly over-account for
the loss. Two candidate causes are the trial step, which reads the filter's
loss on the next step's state rather than the current one, and the windowed
estimate of −dKE/dt itself. Neither has been isolated.

Two checks were made on the instrument. With the filter off at 16³ the budget
closes to 0.0%, with molecular dissipation at 100.8% and pressure work at
−0.8%. This fixes the sign convention: pressure work enters −dKE/dt with the
opposite sign to a dissipation, since it exchanges energy between the kinetic
and internal forms rather than removing it. The early negative residual that
the script's header had attributed to pressure-dilatation is now measured to be
exactly that.

A diagnostic on this case must restore the artificial coefficient arrays it
writes. `diss_split` and the filter probe both recompute the coefficients into
the solver's own storage, and `max_rate` sizes the next step from that storage
without recomputing, so an unbracketed sample perturbs the trajectory it is
measuring. The budget pass therefore brackets itself with `art_block` and
`set_art_block!`. The perturbation is below the reading at production times:
32³ to t = 10 agrees on the peak to five digits and on the kinetic-energy
misfit to 0.013% with and without the bracket, and the difference reaches 1.9%
only at 16³ to t = 1, where the peak is still rising.

**Refinement does not separate μ\* from the filter.** Their ratio is invariant
across the refinement. The 32³ point below is the earlier `:compact` run
(12% / 5% / ~0% / 83%, peak 1.46e-2 at t = 6.5), not the Gaussian row above:

```
32³ :  mu* 5.0%  / filter 83.0%  = 0.060
128³:  mu* 2.3%  / filter 37.3%  = 0.062
256³:  mu* 0.8%  / filter 12.8%  = 0.0625
```

The calibration therefore holds the filter fixed and sweeps `C_mu` at 128³:

```
C_mu = 0       (art off)   1.2153e-2 @ t ≈ 9.00    −5.0% vs reference
C_mu = 0.0005              1.2108e-2 @ t ≈ 9.05    −5.5%
C_mu = 0.002   (default)   1.2044e-2 @ t ≈ 9.06    −6.0%
C_mu = 0.008               1.1950e-2 @ t ≈ 8.77    −6.7%
```

The sweep is a `:compact` campaign, taken before the Gaussian smoother became
the default. The smoother sets how the sensor feeding μ\* is smoothed, so these
shares are not the shipped configuration's. The same point under the Gaussian
gives 1.2120e-2 at t = 8.89 with 61.1 / 2.4 / 36.4 against 60.4 / 2.3 / 37.3
([the α sweep](#the-alpha-sweep-at-128)), 0.63% apart in the peak, so the
conclusions below are unaffected.

**The 128³ peak does not determine `C_mu`.** Every value underpredicts the
reference peak and raising the constant moves further away, so the sweep has no
crossing. The whole 16-fold range spans 1.7% in the peak while the 128³ residual
is 6% and of one sign, three and a half times larger. The earlier reading of this
table located `C_mu ≈ 0.004` by interpolating a crossing that exists only against
the rounded `1.2e-2`, which happened to fall inside the swept band; the direction
of that fit was an artifact of the rounding, and `C_mu ≈ 0.004` should not be
carried forward as a candidate default on this evidence.

The table still establishes that the peak is monotone in `C_mu` and weakly
dependent on it, and that the residual it sits in belongs to the resolution and
configuration rather than to the coefficient, since the 256³ row overshoots by
1.6% on the same estimator. **Retain 0.002**, as a value consistent with the
case rather than one determined by it. The upper end of the band remains excluded by
the one-dimensional result that Noh ν = 3 does not complete at 0.008.

The μ\* calculation costs 43% of wall time at 128³, 21% per step from
`compute_artificial!` plus 18% more steps. Channel shares scale as expected: μ\*
is 2.3% of the sink at `C_mu = 0.002` and 8.2% at 0.008, while the filter share
moves 37.2% to 33.1%.

<a id="the-mu-controls-at-64"></a>

### Removing μ\* improves the fit at 64³

If μ\* acted as a subgrid closure on this case, removing it would degrade the
histories. Instead it improves them, at both filter strengths and on every
estimator. Three controls were run at 64³ with the Gaussian smoother,
`cfl = 0.6` and `filter_interval = 1`: the shipped configuration, `C_mu = 0`
with β\* and κ\* retained, and every artificial property off.

```
alpha = 0.499        peak vs ref   peak time   KE misfit   -dKE/dt misfit
full (C_mu 0.002)      +6.97%        -0.50      2.532e-2       1.366e-1
C_mu = 0               +6.39%        -0.50      1.412e-2       9.936e-2
art off                +5.32%        -0.42      1.176e-2       8.241e-2

alpha = 0.45
full (C_mu 0.002)      -1.57%        -0.45      4.137e-2       1.777e-1
C_mu = 0               -1.32%        +0.09      3.831e-2       1.646e-1
art off                -0.40%        -0.02      3.446e-2       1.593e-1
```

The ordering is monotone in all four columns and holds under the production
filter as well as the near-off one, so it is not an artifact of weak filtering.
μ\* carries 17.6% of the sink at α = 0.499 and 4.7% at α = 0.45, and the
histories are worse for it in both cases. On this case the best-fitting value
of `C_mu` is zero.

This does not show that 0.002 is wrong; it shows that the value cannot be
chosen on Taylor–Green. The case is close to resolved at 64³ and closer at
128³ and 256³, so what is measured is the excess dissipation a setting adds
rather than the subgrid dissipation it contributes. `C_mu` has to be fitted on
a case with an unresolved cascade, and the upper bound continues to come from
the one-dimensional battery of `bench/artcal.jl`.

The art-off row at α = 0.45 matches the windowed reference peak to −0.40% and
its time to −0.02, the closest agreement recorded on this case. The same row
carries the worst kinetic-energy misfit of its arm, 3.446e-2 against 1.176e-2
at α = 0.499. The two estimators disagree sharply at this resolution, and the
history is the more reliable of the two ([the estimator
note](#read-the-rate-over-a-window)).

The budget closes to between −3.0% and +1.1% across the α = 0.45 arm. Under the
near-off filter the deficit grows to −3.8%, −6.1% and −12.8% as the damping is
removed. Two explanations remain and the data does not separate them. A central
compact scheme without dealiasing produces energy at the grid scale, which the
filter then removes on top of the physical cascade, so the measured
dissipations exceed the observed loss. Alternatively, the windowed −dKE/dt is
compared against instantaneous channel values at a sample that may sit on the
curved flank of the peak rather than at it. A proportional bias in the filter
probe is excluded, since the deficit is largest where the filter's share is
smallest.

### The filter is the stabilizer

At 32³, `filter_interval = 4` diverges and `filter_interval = 0` fails with
`SolverFailure(:negative_density)` at t = 5.32. At 128³:

```
config              peak -dKE/dt        mol    mu*   filter   steps   wall
art ON,  filter 1   1.2044e-2 @ t≈9.06  60.4%  2.3%   37.3%   12739   1520 s
art OFF, filter 1   1.2153e-2 @ t≈9.00  62.5%  0      37.5%   10830   1063 s
art ON,  filter 0   SolverFailure(:negative_density) at t = 4.66, step 8515
```

Removing the filter at 128³ kills the run earlier than at 32³ despite the filter
supplying only 37% of the sink there, while the run with the artificial
properties disabled and the filter on reaches t = 10. The 128³ failure is a clean
energy blow-up rather than the dispersive-undershoot signature of the shock
cases: KE tracks the filtered run to within 0.1% until t ≈ 4.4, turns upward, and
triples before positivity goes, with `dt` collapsing to 2e-60. TGV is unforced,
so rising KE is unambiguously numerical. The filter supplies essentially all of
the grid-scale sink whatever its share of the total, and is therefore necessary and
sufficient for stability at this resolution while the Cook properties are
neither. Only the art-on leg was run at 256³, so this statement rests on 128³.

### Read the rate over a window

The filter removes energy per application, so the instantaneous rate is
`(filter loss)/dt + physical` and carries the full step-to-step `dt` jitter
divided into it. With the artificial properties on, the sensor feeds `compute_dt`
and `dt` swings ±12% step to step at 128³; against a filter supplying 37% of the
sink this predicts ∓4.4% on the total, matching the ∓4% scatter in the one-step
column. With them off, `dt` and the rate vary by under one percent.

`C_mu` is ranked on differences well under 1%, which the one-step rate cannot
resolve. The 501-step windowed rate reduces within-run scatter to about 0.3%.
`bench/tgv_energy.jl` reports every rate windowed (`window=`), and every number
here is windowed. A boxcar over a curved peak reads low: 0.10% at the 128³
half-width of 0.196 time units, 0.39% at 0.2, and 3.0% at 0.5. That is
common-mode across configurations run at the same window, so it cancels in a
`C_mu` comparison, but it does not cancel against the external reference, and at
32³ the same 501 steps span 0.8 time units and cost 5.2%. The script therefore
compares against the reference passed through the run's own window rather than
against the tabulated maximum, and prints both.

`run!` truncates the final step to land on `tfinal`, and a short step still pays a
full filter pass, so −dKE/dt inflates there. At 128³ that produced 1.4226e-2 at
t = 10.00 against the true 1.2065e-2 at t = 8.77. `bench/tgv_energy.jl` excludes
that step.

**Recommendation: retain 0.002.** Do not raise it past 0.008. The resolved
three-dimensional shear case is consistent with the default but does not
determine it, for the reason above: at 128³ the peak carries a 6% one-signed
residual that the coefficient does not control, so the 1.7% it does control
cannot be read off against an external value. The peak is therefore not usable
as an estimator at this resolution.

The history misfit is the replacement. `bench/tgv_energy.jl` reports the
relative L2 distance between a run's kinetic-energy and −dKE/dt histories and
the vendored reference's, over every step of the run, which is a curve fit
rather than one scalar fitted with one parameter. Ranking `C_mu` on it requires
a resolution whose own history error is below the effect size, and requires the
filter to be settled first, since the fit stays conditional on
`compact_filter(0.45)` applied every step and, under `filter_cfl = 0`, on the
CFL as well. Both belong to N1. The 64³ controls below show a further
difficulty: on this case the best-fitting `C_mu` on the history misfit is zero
([removing μ\*](#the-mu-controls-at-64)).

## C_D, the species diffusivity

A sharp binary interface advected at u = 1 for t = 0.5 on 256 points, initial
10–90% width 2h = 0.0078:

```
C_D       | interface width
0         |  0.01785
0.0025    |  0.01794
0.01   *  |  0.01820
0.04      |  0.01922
0.16      |  0.02171
```

With D\* disabled the interface still more than doubles in width, so the compact
filter is the dominant source of broadening for a passive interface. Across a
64-fold sweep in `C_D` the width changes by 22%. Interface width is more
sensitive to `filter_interval` and to α, where increasing α weakens the filter.

At `n_species == 2` the per-species sensor machinery is a measurable no-op
(`D*_1` and `D*_2` agree to 4.8e-16), and the whole term is a minor contributor
to what it controls.

**Recommendation:** retain 0.01. Consider larger values only for mixtures with at
least three species, where the correction velocity can alter the species fluxes.

## C_Y, the mass-fraction bound

A shocked species interface rings. Measured on a Mach 1.5 shock in air (γ = 1.4,
R = 1) running into a tanh interface with SF6 (γ = 1.09, density 5.04) on 400
points, Dirichlet ends, `cfl = 0.4`, default `ArtParams`:

```
initial interface | worst Y          | final Y range     | width (cells)
2h                | -0.204 / +1.204  | -0.009 / +1.098   | 6 → 3
4h                | -0.023 / +1.023  | -0.0004 / +1.021  | 12 → 4
8h                | clean            | clean             | 14
```

The shock compresses the interface by the density ratio across it, and the
ringing is a two-cell odd-even train on the light side set by the cells the
interface spans after compression. Nothing else varied moves it: N = 800 and 1600
at a fixed 2h interface, `closures = :onesided`, `beta_sensor = :dilatation`,
`detector = :d8`, `cfl = 0.2` and `compact_filter(0.3)` all leave the worst
excursion between −0.18 and −0.29. `C_D = 1`, a hundred times Cook, reaches
−0.008, and disabling the artificial properties gives −0.43. Cook's
D\* = C_D c h |δ⁴Y| peaks near 2e-5 in the train, a diffusive time across a cell
of order one against a shock crossing of 0.005, so it cannot hold an interface
the shock has thinned.

The species diffusivity therefore carries a second term, the bound of Shankar,
Kawai & Lele (Phys. Fluids 23, 024102, 2011, eq. A4), which is zero wherever
0 ≤ Y ≤ 1:

    D*_k = c · G[ max( C_D h |δ⁴Y_k| , C_Y h max(0, −Y_k, Y_k − 1) ) ]

with G the sensor smoother. Cook (Phys. Fluids 21, 055109, 2009, eq. 42) writes
the same bound as (|Y| − 1 + |1 − Y|), twice the max, scaled by Δ²/Δt with
C_Y = 50; Miranda today (Brill, Olson & Bokman, arXiv:2503.12680, 2025, eq. 24)
combines its ringing and bound terms by max at C_O,Y = 100 in the same scaling.
The two scalings differ by 1/CFL, so every published value sits near 100 in the
form above.

Measured on the 2h case at `C_D = 0.01`:

```
C_Y    combine  smoothing   | worst Y          | final Y range      | steps
0      –        –           | -0.204 / +1.204  | -0.0087 / +1.098   |  640
50     sum      separate    | -0.018 / +1.018  | -0.0012 / +1.001   |  644
100    sum      separate    | -0.013 / +1.013  | -0.0007 / +1.001   |  647
200    sum      separate    | -0.010 / +1.010  | -0.0004 / +1.001   |  655
100    max      separate    | -0.014 / +1.014  | -0.0007 / +1.001   |  647
100    max      one pass    | -0.013 / +1.013  | -0.0007 / +1.001   |  647
1000   sum      separate    | -0.072 / +1.072  | -0.0001 / +1.000   |  669
100    sum      none        | -0.147 / +1.147  | -0.0038 / +1.019   | 1955
```

Sum and max are indistinguishable, as are smoothing the two terms separately and
smoothing their max once, which is Cook's single filter of the bracket and costs
no line solve beyond the one the ringing sensor already pays. The unsmoothed term
is a grid-scale diffusivity that both helps less and, through `compute_dt`,
triples the step count. The residual 1% at `C_Y = 100` is the compact scheme's
dispersion at a three-cell contact; at a 4h initial interface it is 0.25%, and at
Mach 3 the term takes the worst excursion from −0.94 to −0.043.

The bound enters the diffusive rate in `compute_dt` like the rest of D\*.
Excluding it is stable up to `C_Y = 500` on this case and saves 1% of the steps at
100 (638 against 647); at 1000 it takes a square root of a negative pressure.
Cook's Δ²/Δt scaling fixes the bound's diffusion number by construction, so the
bound never reaches his timestep at all. Keeping it in the rate here is the same
choice made explicitly, and it stays.

On the species-advection case, which also rings (−0.064 / +1.041 at `C_Y = 0` on
256 points, −0.054 / +1.037 on 512), `C_Y = 100` brings the range to ±6e-4 and
moves the 10–90% width from 0.018187 to 0.018199. With two species the bound is
identical for both, so a shared diffusivity is bit-identical to the per-species
one here; its case is three or more species, where per-species diffusivities can
move bulk density at an equal-density interface (Brill et al., §4.3).

### The dead band

The term is not inert on a smooth profile that touches a bound. Uniform advection
of Y = (1 + cos 2πx)/2 for one period measures order 4.9 in L2 at `C_Y = 0` and
3.6 at 100, although the completed steps never leave [0, 1]: the excursion appears
only at Runge–Kutta stages 2–5, at 6e-6 on 64 points and falling 4× per doubling,
and the bound turns it into diffusivity. A dead band `Y_tolerance = 1e-4`
restores the `C_Y = 0` errors to the last digit and leaves the shock case
unchanged, worst −0.0135 against −0.0134. Evaluating the excursion at stage 1 only
does the same; the dead band stores nothing and is the form landed.

### Cost on a smooth profile

On two-species advection at uniform ρ, p and u with Y₁ = (1 + cos 2πx)/2 over one
period, the L2 error under the default D\* with its bound is 3.12e-6 at N = 32 and
1.15e-10 at N = 256 (order 4.84 → 4.97); with the species channel off altogether
(`C_D = C_Y = 0`) it is 7.85e-8 and 3.23e-12 (order 5.50 → 4.29), 40× smaller at
N = 32 and 35× at N = 256. The profile touches 0 and 1 and the dead band keeps the
bound inert there, so the factor belongs to the ringing sensor,
`C_D Δ|δ⁴Y|` responding to a resolved cosine. It is the price of a regularization
that cannot distinguish a smooth extremum from an incipient overshoot. The bulk
channel at equal molecular weights reproduces these errors to seven digits, so
this is a property of the bracket and not of the flux form.

### Supporting measurements on the surrounding formulation

Taken on a stationary two-gas interface at uniform p, T and u = 0 for t = 0.25,
to establish that the bound is not covering an error belonging elsewhere.

- The compact filter alone leaves p, u and T at 1e-14 and is the whole source of
  the ±1e-3 mass-fraction overshoot of a resting 2h interface. The uniform-(p, T)
  state is a linear subspace of the conserved variables for ideal gases of
  constant c_v, which a componentwise linear filter preserves.
- The artificial species flux with its enthalpy energy flux generates the
  interdiffusion velocity u ≈ −D\* ∇ln ρ (1.4e-4 at D\* ≈ 1e-6) and holds T
  uniform to 1e-6. The same flux carrying species internal energy instead, which
  Brill et al. argue for, leaves a secular temperature drift 50× larger, so the
  enthalpy form stays.
- Three species at equal density under the per-species diffusivities hold bulk
  density to 4e-14, so the correction velocity already supplies what Brill et al.
  obtain from a shared diffusivity, and the per-species form stays.

**Recommendation:** `C_Y = 100` with `Y_tolerance = 1e-4`, the defaults.

## The sensor smoother

Cook's sensor smoothing is `gbar` in Miranda, which resolves to `cgfs4`
(`parcop/stencils.f90`): an explicit nine-point symmetric stencil with `nol = 0`
and `implicit_op = .false.`, weights 3565/10368, 3091/12960, 1997/25920,
149/12960 and 107/103680. Over the common denominator 103680 these sum to exactly
1, so the filter preserves constants without relying on cancellation, and at
boundaries the overhanging weight is folded onto the mirror point, which
preserves the unit sum there. There is no linear solve and no interface
reduction; the operator needs a halo of four.

`ArtParams.smoother` selects between that Gaussian and one pass of
`compact_filter(0.45)`. Their transfer functions at modified wavenumber k:

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

As a Cook test filter the compact pass is close to the identity over the whole
resolved band, and it costs a distributed line solve per active dimension per
sensor, `n_species` of them per RHS evaluation for the species sensors.

Highest CFL reaching `t_final` with a correct plateau, ladder extended until both
settings fail:

```
Noh geometry              :compact   :gaussian
nu = 1  planar wall          0.2        0.2
nu = 2  cylindrical axis     0.15       0.2
nu = 3  spherical origin     0.15       0.4
```

Both fail at 0.5 everywhere, so 0.4 is a measured ceiling and not a table edge.
The origin, previously the least forgiving fold, becomes the most forgiving one.
Cost moves the same way: on the two-species tube of `bench/phases.jl` the
`artificial` phase falls from 1.360 ms to 0.971 ms, 31.8% to 24.8% of the
right-hand side, against a run-to-run spread of about 1.3% over three `:compact`
readings. The mechanism is [sensor
intermittency](#sensor-intermittency-at-the-damage-site).

Accuracy is mixed and small, with wall heating the one clear regression: the ν = 1
deficit moves +58% to +64%, while plateaux and the Shu–Osher train move in the
fourth digit. Two refit diagnostics were run against that regression, and both
are negative.

**κ\* cannot buy the wall heating back.** Sweeping `C_kappa` under each smoother,
ν = 1 wall deficit:

```
C_kappa      0      0.0025    0.01*     0.04     0.16
compact    +64%     +61%      +58%      +57%     +70%
gaussian   +65%     +64%      +64%      +68%     +91%
```

The lever inverts. Under `:compact` the trough is at 0.01–0.04; under `:gaussian`
the trough is the default, and raising the constant makes wall heating worse,
sooner and more steeply. The Gaussian widens the κ\* footprint and lowers its
peak, so added conductivity spreads across a region rather than concentrating in
the wall cell where the entropy error is deposited. Separately, at `C_kappa = 0`
the ν = 3 case fails under `:compact` and completes under `:gaussian`, and the
ν = 3 plateau is better under `:gaussian` across the whole sweep, 0.9745–0.9822
against 0.9728–0.9796.

**Three dimensions are neutral.** Taylor–Green at 64³ to t = 10, split at the
peak:

```
smoother   t_peak   molecular   mu*     beta*   filter
compact     8.39      32.6%     4.0%     0.0%   63.5%
gaussian    8.45      33.8%     4.5%     0.0%   61.6%
```

The channel `C_mu` controls moves from 4.0% to 4.5% and the filter still
dominates, so the 3-D budget does not shift enough to demand a `C_mu` refit. A
full `C_mu` sweep under `:gaussian` has not been run.

**Recommendation:** `:gaussian`, the default since August 2026. The wall-heating
regression is a property of the smoother and is largely recovered by
[`detector = :d8`](#the-ringing-detector) or by [the one-sided filter wall
rows](#the-filters-wall-cascade).

## The ringing detector

`ring()` in `parcop/operators.f90` dispatches to `d8x/d8y/d8z`, a full compact
operator (`c10d8`, `parcop/stencils.f90`) with a pentadiagonal left-hand side, a
nine-point right-hand side, and its own symmetric and antisymmetric boundary
closures. `artificial.jl` uses the undivided fourth difference
δ⁴ = (1, −4, 6, −4, 1), following Cook (2007) literally.

`ArtParams.detector` selects between them. `:delta4` is the default and `:d8` is
[`compact_d8`](../src/kernels_banded.jl), the reference operator transcribed. Its
interior rows are

```
1.5 g_{i-2} + 14 g_{i-1} + 29 g_i + 14 g_{i+1} + 1.5 g_{i+2} = 60 δ⁸f_i,
```

with δ⁸ the undivided eighth difference. It is planned as a symmetric operator
rather than a derivative, and its four closure rows fold the overhanging interior
weights onto the half-offset mirror; `reference/DESIGN.md` carries the
construction. Every closure row's weights sum to zero, so a constant is
annihilated without relying on cancellation, measured at 1.2e-16 through the
closure rows.

### Normalization

The coefficients are divided by ζ = 29 to put a unit diagonal on the left-hand
side, and by a further 240. The second factor sets the response to a
grid-to-grid oscillation to 16, the value undivided δ⁴ gives there, so both
detectors agree at the wavelength both exist to catch and diverge only below it:

```
k/pi     delta4     d8        ratio
0.25     3.43e-1    6.03e-4   569x
0.5      4.00e+0    1.54e-1    26x
0.75     1.17e+1    3.69e+0    3.2x
1.0      1.60e+1    1.60e+1     1x
```

Without that second factor `:d8` would produce sensors 240× larger at the Nyquist
and the four constants would need refitting by two orders before any case could
run. With it they transfer as starting points, which keeps the comparison a
one-variable one. This is not the reference's own normalization.

### Results

Measured on top of `smoother = :gaussian`, which is a requirement rather than a
convenience: the failure the Gaussian fixes is [β\*
intermittency](#sensor-intermittency-at-the-damage-site) at a symmetry cell, and
a sharper high-pass makes narrower sensor spikes, so testing `:d8` against the
compact smoother would reject it for a defect belonging to the smoother.
`bench/artcal.jl detector`, default constants:

```
detector    Noh1 plat  deficit   Noh3 plat   Lax L1   contact   Shu train   WC peak
delta4*      0.9993     +64%      0.9751     5.0e-3   0.0053     1.6180     6.6049
d8           0.9997     +53%      0.9951     4.7e-3   0.0044     1.6360     6.3953
```

Six of seven columns improve, several well beyond the fourth digit that separates
the β\* sensor variants. The ν = 3 plateau error falls from 2.45% to 0.42%. Wall
heating, the standing regression of the smoother change, recovers eleven of the
points it lost. The Shu–Osher wave train, the column that pulls against every
damping constant, gains 1.1%, which is the selectivity the detector was adopted
for. Woodward–Colella peak density is the one regression, 6.6049 to 6.3953 at
N = 400.

Highest CFL reaching `t_final` with a correct plateau:

```
Noh geometry              :delta4    :d8
nu = 1  planar wall          0.2      1.0+
nu = 2  cylindrical axis     0.2      1.0+
nu = 3  spherical origin     0.4      0.25
```

`1.0+` is not a table edge: the plateau under `:d8` is flat to four digits from
0.15 to 1.0 in both geometries (ν = 1 at 0.9997, ν = 2 at 0.9481–0.9500), so the
artificial-property restriction on those two is gone rather than raised. The
spherical origin moves the other way, 0.4 to 0.25. `bench/nohprobe.jl 3
detector=d8 cfl=0.3` puts that failure at the origin cell: through steps 50–100
the worst internal-energy cell travels with the front at i = 30–32 carrying β\* at
15–36% of its own domain maximum, then at step 125 the worst cell is i = 1 at
e/e₀ = −6335 with β\* at 1.8% of maximum, and the run loses density three steps
later. This is the symmetry-cell startup mechanism of [the CFL
section](#where-the-restriction-originates), with `:d8` moving its threshold and
not its character.

### Detector selectivity depends on the sensor field

The selectivity is available to the κ\* and D\* channels and largely unavailable
to μ\* and β\*. `bench/artcal.jl response` puts a velocity sine of one wavelength
on a periodic 64-point line and reads the peak coefficient back, with no time
integration, against the detectors' own designed separation of 569 at eight
points per wavelength, 26 at four, and 1 at the Nyquist:

```
k/pi   ppw  |  mu* from |S|:   d4       d8    ratio  |  mu* from u:    d4       d8     ratio
0.125  16.0 |             7.24e-5  4.20e-5   1.72    |            4.11e-6  4.18e-10  9.8e+03
0.250   8.0 |             3.31e-4  2.58e-4   1.28    |            4.71e-5  8.29e-08     569
0.500   4.0 |             2.44e-3  2.44e-3   1.00    |            3.93e-4  1.51e-05      26
0.750   2.7 |             8.33e-4  6.49e-4   1.28    |            1.60e-3  5.07e-04    3.16
1.000   2.0 |             0.00e+0  0.00e+0    ---    |            3.14e-3  3.14e-03       1
```

Applied to the velocity the two detectors reproduce their designed ratios to four
figures at every wavelength. Applied to |S| they differ by a factor of 1.8 or less
from 32 points per wavelength down to the Nyquist. The cause is the absolute
value: |S| has a cusp wherever the strain passes through zero, a cusp is
grid-scale structure at any resolution, and no detector is insensitive to one. β\*
from ∇·u gives the right-hand ratios except in the last row. On a pressure wave
resolved over 64 points the two detectors differ by 2.6e6 on κ\*, whose input is
the internal energy, against a factor of 1.8 on β\*.

The last row is a property of every field obtained by differentiation. A centered
scheme has zero modified wavenumber at the Nyquist, so |S| and ∇·u vanish
identically for a two-point velocity wave and the sensors built from them return
zero there under either detector. Only the velocity-component sensor responds,
with the full undivided 16Ah. The calculation is stable in spite of that for the
reason the `C_mu` section records: grid-scale dissipation comes from the compact
filter and not from the Cook properties.

### Detector cost

On the two-species tube of `bench/phases.jl`, back to back on one machine:

```
detector   artificial   % of RHS   compute_rhs!   line solves
delta4      0.946 ms     23.5%       3.927 ms     24 + 0
d8          1.707 ms     34.6%       4.689 ms     24 + 8
```

Eight pentadiagonal solves per right-hand side, one per active dimension per
sensor, for +80% on the sensor phase and +19% on the whole evaluation. Against a
10–20% run-to-run spread the phase figure is resolved and the total is marginal.

### The C_beta refit under `:d8`

`bench/artcal.jl beta` and `bench/artcal.jl beta detector=d8`, back to back on one
machine. The `:delta4` column is a fresh control that reproduces the `:delta4` row
of the detector comparison in every column, so the two detectors are separated by
one variable.

```
                :delta4                                  :d8
C_beta   Noh1 deficit  Noh3 plat  contact  Shu     Noh1 deficit  Noh3 plat  contact  Shu
0.25        +56%        1.0098    0.0041  1.6408      +33%          NaN     0.0039  1.6489
0.5         +61%        0.9931    0.0045  1.6297      +44%          NaN     0.0041  1.6420
1.0   *     +64%        0.9751    0.0053  1.6180      +53%        0.9951    0.0044  1.6360
2.0         NaN           0.9569  0.0063  1.6055      +58%        0.9767    0.0046  1.6300
4.0         NaN           0.9408  0.0074  1.5945      +62%        0.9578    0.0051  1.6232
```

The viable window moves and the optimum inside it does not. Under `:delta4` the
window is bounded above by planar Noh, which fails at 2.0, and runs 0.25 to 1.0.
Under `:d8` it is bounded below by the two converging geometries, which fail at
0.5 and at 0.25, and runs 1.0 to 4.0 without reaching an upper bound in the
sample. **The two windows intersect in the single value 1.0**, the default.

`:d8` also flattens the response to the constant on every smooth measure. Over
0.25 to 4 the contact broadens 31% under `:d8` against 80% under `:delta4`, the
Shu–Osher train loses 1.6% against 2.8%, and Lax L1 moves 4.7e-3 to 4.9e-3
against 4.8e-3 to 5.4e-3. Selectivity against resolved structure also makes the
resolved structure less sensitive to the magnitude of the constant.

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
the default constant, and it costs 2.33% ν = 3 plateau error against 0.49%, the
ν = 2 ceiling falling from 1.0+ to 0.4, 0.4% of the Shu–Osher train and five
points of wall heating. **`C_beta = 1.0` is retained under `:d8`**, and the
detector decision falls to the origin cell alone.

One reading favours the detector. At `C_beta = 1.0` the *worst* geometry improves
under `:d8`, 0.2 to 0.25, because ν = 2 rises from 0.2 to beyond 1.0 while ν = 3
falls from 0.4 to 0.25. A user who does not know which geometry is ahead is better
off under `:d8`; the case against it is specifically the converging spherical one.

### A failure that gets worse as the timestep falls

Two cells of the `:d8` ladder are not ceilings. At `C_beta = 0.25`, ν = 2
completes at cfl = 0.4 and fails at 1.0 and at everything from 0.3 down; at
`C_beta = 0.5` it completes at 1.0 and 0.4 and fails from 0.3 down. Both were
re-run against a probe separating the two ways `m_noh` returns NaN, and every one
of those cells is a positivity loss, not the step cap. The sweep prints the step
cap as `Inf` so the distinction survives.

A CFL-type stability restriction cannot produce a failure that appears only below
a CFL. A per-step operation can, because a fixed physical interval integrated at
half the timestep applies it twice as many times. The per-step operation in the
loop is the compact filter, which runs once per step at `filter_interval = 1` and
[removes energy per application](#the-compact-filter). The sign of the dependence
is the evidence; the accumulation itself has not been measured step by step,
which `bench/nohprobe.jl` could do.

### Recommendation

`:delta4` remains the default. The battery favours `:d8`, as do the planar and
cylindrical ceilings, but the general guidance for converging shocks rests on the
spherical case, where `:d8` costs 40% of the timestep. The four constants are
also the δ⁴ fit, and a detector that changes the sensor's spatial support by this
much has no claim on them; the `C_beta` refit above is the one that has been
done, and it retains the default. What remains is [the origin
cell](#the-origin-cell-is-a-startup-transient), which is a robustness question
about a symmetry cell under a startup transient rather than a detector question.
`:delta4` is kept because it survives that excursion at a larger timestep, not
because anything has been shown wrong with `:d8`.

## The sensor fields and the compression switch

Cook builds μ\* and β\* from the strain magnitude |S| = sqrt(S_ij S_ij). Miranda
builds μ\* from `ringV(u, v, w)`, the ring of each velocity component along each
direction reduced by `MAX` over the nine pairs, and β\* from `ring(∇·u)`. Neither
reference field carries an absolute value. `ArtParams.mu_sensor` (`:strain`,
`:velocity`), `ArtParams.beta_sensor` and `ArtParams.reduction` (`:sum`, `:max`)
select between them. The weight is h_d for a field carrying one velocity
derivative fewer, against h_d² for |S| and ∇·u, and the two coincide at the grid
scale, so the four constants transfer between sensor fields on the same basis as
between detectors.

`beta_sensor` has four settings. Throughout, S is the strain-rate tensor and |S|
its magnitude, δ⁴ the undivided fourth difference, Δ = ∇·u the dilatation, ω the
vorticity vector, H the Heaviside step, and ε a fixed regularizer at the
literature value 1e-32.

- `:strain` (default) is the Cook original, Σ_d h_d²|δ⁴_d S|.
- `:gated_strain` keeps that sensor and multiplies it by the compression switch
  H(−Δ)·Δ²/(Δ² + |ω|² + ε). One pointwise pass, no line solves.
- `:dilatation` additionally rebuilds the sensor from Δ, the full form of Mani,
  Larsson and Moin (JCP 228, 2009). One further sensor smoothing pass per RHS
  evaluation.
- `:ungated_dilatation` is that sensor without the switch, the reference form.

### Results on the battery

`bench/artcal.jl field`, default constants, both detectors, at the default CFL:

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
largest movement is the Shu–Osher train under `:d8`, 1.6360 to 1.6368, or 0.05%.
Every case here is one-dimensional at `C_mu = 0.002`, so the shear channel does
very little in them.

**β\* from the dilatation improves four columns and degrades two.** Under `:d8`
the ν = 3 plateau error falls from 0.49% to 0.24%, ν = 1 wall heating from +53% to
+51%, the Lax contact sharpens 0.0044 to 0.0041 and its L1 falls 4.7e-3 to
4.6e-3, while the Shu–Osher train loses 0.5% and the Woodward–Colella peak 1.5%.
Under `:delta4` the same change gives up more and gains less: 0.4% of the ν = 3
plateau and 1.0% of the Woodward peak for the same contact gain.

The gated and switched forms at the default CFL:

```
sensor       | Noh1 plat  deficit | Noh3 plat | Lax L1  contact | Shu train | WC peak
strain     * |  0.9992       +58% |   0.9732  | 5.0e-3   0.0051 |    1.6185 | 6.5731
gated_strain |  0.9992       +58% |   0.9722  | 5.0e-3   0.0052 |    1.6177 | 6.5850
dilatation   |  0.9990       +59% |      NaN  | 5.0e-3   0.0051 |    1.6226 | 6.5315
```

`:gated_strain` is a wash on accuracy: every column moves in the fourth digit and
the movements go both ways. The Shu–Osher gain belongs to the sensor field and not
to the switch, since `:dilatation` keeps 0.25% more wave-train amplitude than the
default while `:gated_strain` keeps 0.05% less.

### The CFL ceilings

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

The gated and switched forms were laddered separately, under `:delta4` and the
`:compact` smoother:

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

The switch moves the cylindrical ceiling and the sensor change moves nothing:
planar Noh completes at 0.2 and fails at 0.3 under all three settings, and ν = 3
needs 0.15 under both settings that reach it at all. `:dilatation` loses both
converging geometries at every CFL sampled.

The ν = 3 column of this sweep reads 0.2 for the two `:d8` strain rows where the
detector ladder reads 0.25, because the two sweeps sampled different CFL values;
0.25 is the finer reading.

The ungated dilatation keeps the spherical origin and loses the cylindrical axis;
`:dilatation`, the same sensor with the switch applied, loses both. The spherical
loss is therefore attributable to the switch and the cylindrical loss to the
sensor field. `:gated_strain`, which applies the switch to the strain sensor,
raises the cylindrical ceiling instead, from 0.15 to between 0.2 and 0.25 under
the `:compact` smoother the ladder below was taken with:

```
Noh2   cfl  | strain     gated_strain
       0.15 |    0.9355        0.9344
       0.2  |       NaN        0.9353
       0.25 |       NaN           NaN
```

The plateau `:gated_strain` reaches at 0.2 agrees with its own value at 0.15, so
the larger step is a completion and not a run that avoided the positivity check.
Planar and spherical Noh are unmoved by the switch.

### Why the dilatation sensor loses the axis

The cause is at the coordinate fold and is visible at t = 0 before any shock
forms. Noh starts from u_r = −1 everywhere, for which the strain and dilatation
sensors are analytically identical away from the axis: Δ = −1/r and |S| = 1/r
with zero vorticity, so the switch is exactly 1 and β\* agrees bit for bit. At the
first few cells of the axis the discrete radial derivative of u_r is no longer
zero, and there the two forms diverge, because Δ = S_rr + S_θθ adds two
same-signed components where |S| = √(S_rr² + S_θθ²) partially cancels them.
Cylindrical Noh at N = 256, t = 0:

```
i   r        |S|      div       beta* strain   beta* dilatation
1   0.00196  612.0    -847.8    1.746e-02      3.790e-02
2   0.00587  204.1     -57.9    1.466e-02      4.106e-02
3   0.00978  110.9    -145.1    3.771e-03      2.087e-02
6   0.02153   46.5     -44.1    0              8.089e-04
120 0.46771    2.14     -2.14   3.856e-12      3.856e-12
256 1.00000    1.00     -1.00   1.199e-07      1.199e-07
```

The two agree to the last digit well away from the axis and differ by factors of 2
to 70 within the first several cells, where the cell measure is smallest. The run
loses positivity on the first step. Raising `C_beta` does not recover it: at 2, 4
and 8 under the dilatation sensor both converging geometries still fail and planar
Noh joins them, which is the same upper stability bound the [`C_beta`
table](#c_beta-the-shock-constant) shows. This is a property of the fold rather
than of the shock capturing, so the option is not recommended in converging
geometry at any setting of the constants.

### Switch selectivity and decomposition sensitivity

On a solenoidal Taylor–Green field at 32³, where β\* has nothing legitimate to do,
`:gated_strain` removes 99.4% of the summed β\* and leaves 71 of 32768 points
above 1e-12. It does not remove the maximum, which stays at 0.42 of the ungated
value. Those surviving points are the cusps of |S|: the strain sensor is a fourth
difference, so it peaks where |S| passes through zero with a kink, and the switch
degenerates at those same points because the vorticity vanishes there too and ε is
all that remains in the denominator. A relative ε scaled to the local |S| was
tried and reverted, since the scale it would use vanishes with |S|. `:dilatation`
has no such points and its β\* falls to 1e-15 of the ungated maximum.

Neither compression-keyed setting reproduces to round-off when the process grid
changes. Summed over the domain, three different split axes agree to 2e-6 relative
for `:gated_strain` and 2e-7 for `:dilatation`, against 1e-14 for the strain
sensor. The cause is the switch and not a missing halo exchange, since the sensor
fields themselves reproduce to 1e-14. H(−Δ) is discontinuous at Δ = 0, and with ε
at 1e-32 the ratio Δ²/(Δ² + |ω|² + ε) has not decayed by the time Δ reaches
round-off, so a point whose dilatation cancels to zero carries either no β\* or
the full C_β·ρ·sensor depending on the last bit. The affected points contribute
negligible β\*, and `test/mpi_tests.jl` records the property in its tolerance.
Anything relying on bit-identical results across process grids should stay on
`:strain`.

<a id="the-mu-channel-on-taylor-green"></a>

### The μ\* channel on Taylor–Green

No case in the battery exercises μ\*, so the field change for that channel is
measured where it carries a share of the sink. 64³, Re = 1600, to t = 10 under the
default smoother, dissipation split at the peak:

```
mu*        reduction   steps   t_peak   peak -dKE/dt   molecular   mu*    filter
strain*    sum*        5888     8.49      1.2459e-2      33.8%     4.5%   61.6%
velocity   sum         5884     8.49      1.2421e-2      33.6%     6.0%   60.4%
strain     max         5682     8.97      1.2496e-2      33.3%     2.7%   64.0%
```

β\* is 0.0% in all three, as in every Taylor–Green measurement here. The velocity
sensor raises the μ\* share by a third at the expense of the filter's, and the
directional maximum cuts it by 40% because the maximum over three directions is
smaller than their sum. Both are moves of one to two points in a 4.5% channel
within a sink the filter dominates, so neither is distinguishable from a rescaling
of `C_mu`. One further result is not a rescaling: the peak time under `:max`
moves 8.49 to 8.97, landing on the reference peak time exactly, the other two
settings being half a time unit early. The 0.3% fall in the peak under
`:velocity` was read as a move toward the reference and is not one: all three
64³ peaks are below the tabulated 1.28575e-2, so the smallest of them is the
furthest away. Its direction was an artifact of the rounded reference.

### Sensor-field cost

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
than everything else in the evaluation combined.

### The fourth-difference clamp at a fold

`delta4_sum!` extends the field past a closed edge by clamping the index, a
zeroth-order extension where the compact closures use the half-offset mirror. The
velocity sensor is the first sensor whose field is odd across a fold, which is
where the two extensions differ in order. For an even field the clamp misplaces
one δ⁴ tap by a term that the vanishing edge derivative makes O(h²); for an odd
field the edge derivative is the largest quantity there and the same tap is wrong
at O(h). On u_r = r at the cylindrical axis, N = 32, the regular behaviour of a
radial velocity, which should produce no sensor at all, the clamp gives
μ\* = 1.2e-6 on the axis cell against 0 for the mirror, or C_mu·ρ·h² of spurious
viscosity on the cell where every converging case fails. `delta4_sum!` therefore
uses the mirror wherever the parity is −1 at a folded edge, and the two detectors
then agree there, `:d8` reaching its own half-offset closure through the fold
plans and giving 7e-16 on the same field.

The even path is left on the clamp, so no recorded number moves. Its error is two
orders smaller in h, and changing it would move every guarded number in
`test/validation.jl`.

### Recommendations

Retain `:strain`, `:strain` and `:sum`. Reach for `:gated_strain` in cylindrical
converging geometry, where it buys a 33% larger timestep for one pointwise pass
and costs nothing measurable elsewhere; whether it should become the default needs
a re-baseline of the fourth-digit changes across the battery and a second geometry
showing the same gain, which the [CFL
measurement](#where-the-restriction-originates) constrains, since the gate
relieves the axis cell and the planar wall and spherical origin are measured not
to respond. `:dilatation` suits Cartesian shock-dominated work where the
Shu–Osher amplitude gain is useful and no coordinate fold is present.
`mu_sensor = :velocity` cannot be evaluated on its merits until `C_mu` is refitted
under it, since it moves the μ\* share of the Taylor–Green sink by a third with
the constant held fixed.

## The compact filter

The compact filter supplies most of the energy sink at every resolution measured
here and it has never been calibrated. Two halves separate. The formulation half
is measured and delivered: the filter removes energy per application, so its
dissipation is not a rate and does not converge as `dt → 0` at fixed resolution,
and `filter_cfl` makes it a rate. The constant half, fitting α, the cadence and
the reference CFL against a reference dissipation history, was run at 128³ and
checked at 256³ ([the fit instrument](#the-fit-instrument) and the sections
following it). The cadence is redundant with α, the relaxed formulation removes
the CFL dependence, and the α that fits at 128³ does not transfer to 256³, so
the default has not moved.

### Dissipation per application

A parallel shear layer, `u_x = 0.1 sin(4y)` at uniform ρ and p, is an exact steady
solution of the Euler equations and stays one discretely, since every x-derivative
of the field vanishes. Kinetic energy is then constant in time and the filter is
the only mechanism that can change it. `bench/filterrate.jl`, N = 32 to t = 0.5,
zero viscosity and artificial properties off:

```
cfl    steps   unrelaxed          relaxed (filter_cfl = 0.4)
0.4      73    4.092e-3  1.000    4.042e-3  1.000
0.2     145    8.107e-3  1.981    4.042e-3  1.000
0.1     289    1.608e-2  3.930    4.042e-3  1.000
```

Unrelaxed, the loss tracks the step count (73 : 145 : 289 = 1 : 1.99 : 3.96) and
not the elapsed time, so a calculation at half the CFL applies twice the subgrid
dissipation over the same physical interval. Relaxed, the loss is constant to six
significant figures across a fourfold change in timestep.

Two obvious alternatives do not isolate the dependence and were rejected. A
broadband field loses 64% of its kinetic energy within tens of steps and then
cannot lose more, collapsing the spread across a 4× CFL change to 1.2%. A velocity
sine at uniform pressure is an acoustic oscillation trading kinetic for internal
energy hundreds of times faster than the filter acts. Total energy shows nothing
either: a symmetric filter on a periodic grid conserves the discrete sum of every
conserved variable exactly, so the filter moves energy between the two reservoirs
and removes none.

### The timestep moves the attribution, not the total

Taylor–Green at 32³, Re = 1600, artificial properties on, Gaussian smoother, at
the dissipation peak:

```
filter_cfl   cfl    peak -dKE/dt        filter    mu*     molecular
0 (default)  0.6    1.4216e-2 @ 6.58    82.2%     5.1%    12.6%
0 (default)  0.3    1.4297e-2 @ 6.61    85.0%     3.6%    11.4%
0.6          0.6    1.4216e-2 @ 6.58    82.2%     5.1%    12.6%
0.6          0.3    1.4373e-2 @ 6.44    82.1%     5.2%    12.7%
```

At the reference CFL the relaxed run reproduces the unrelaxed one to every printed
digit, which is the `w = 1` path and a check on the implementation. Halving the
CFL changes the peak dissipation by 0.6% but moves the filter's share from 82.2%
to 85.0% and the μ\* share from 5.1% to 3.6%, a 29% relative change in the
artificial-viscosity channel from a timestep change alone with `C_mu` held fixed.
Under the relaxation both hold, at 82.1% and 5.2%.

The total barely moves because the sinks compete for a fixed supply: the cascade
rate is set at the large scales, and a filter that takes more at the grid scale
leaves less to reach the scales where μ\* and molecular dissipation act. Filter
dominance can therefore be large and still leave the peak near the reference
value.

The consequence for calibration is that under `filter_cfl = 0`, `C_mu` is
conditional on the CFL as well as on the filter. The μ\* share of the sink is the
quantity `C_mu` is fitted against, and it moves by 29% relative under a change
that has nothing to do with the physics, so any refit under the unrelaxed
formulation has to state its CFL to be reproducible. This is measured at 32³ only.
The shares themselves are strongly resolution dependent, and whether the CFL
sensitivity survives refinement has not been tested.

### The relaxed formulation

`filter_cfl` is the CFL at which one pass is applied at full strength. Below it
the state is relaxed toward the filtered image rather than replaced by it,

    Q ← (1 − w) Q + w F(Q),    w = filter_interval · dt · rate / filter_cfl

capped at one, which holds the dissipation per unit time fixed. Since `compute_dt`
sets `dt = cfl / rate`, the product `dt · rate` recovers the CFL of the step as
taken, including `StepControl` backoff and the shortening applied to land on a
callback instant. Reading the product rather than `solver.cfl` makes a shortened
step filter proportionally less and removes the truncated-final-step artifact
recorded against `bench/tgv_energy.jl`.

The default `filter_cfl = 0` takes the original code path exactly, not a blend at
`w = 1`. Every guarded number in the suite was measured there and none of them
moves.

**Recommendation:** hold `filter_cfl = 0` until α and the cadence are fitted. The
two are coupled, since a fit taken under the unrelaxed formulation is only
reproducible at the CFL it was taken at, so they should be settled together.

### The fit instrument

`bench/tgv_energy.jl` takes `alphaf`, `filter_cfl` and `cfl` as comma-separated
lists and crosses them with `configs`, whose `filter_interval` field is the
cadence, so one invocation covers a grid of the three coupled filter settings.
Each point is scored by the relative L2 distance of its kinetic-energy and
−dKE/dt histories from the vendored reference over every step of the run, and
by the peak seen through the run's own window.

The score is a curve fit rather than a scalar. `C_mu` was fitted on one scalar
with one parameter, which always succeeds, and the correction above shows the
consequence ([Taylor–Green](#taylorgreen)). Both misfits are normalized by the
reference's own RMS over the steps compared, so each is dimensionless and falls
as the fit improves. The rate is compared only where the full window fits
inside the run, and the truncated final step is excluded from both, for the
reasons in [the estimator note](#read-the-rate-over-a-window).

A 32³ shakeout on the default configuration, Gaussian smoother, `cfl = 0.6`,
`filter_interval = 1`, artificial properties on:

```
alphaf   steps   peak -dKE/dt        vs window   KE misfit   -dKE/dt misfit   filter
0.40      3009   1.4039e-2 @ 6.62     +15.2%      1.532e-1     7.216e-1       85.1%
0.45      3078   1.4217e-2 @ 6.61     +16.4%      1.435e-1     7.408e-1       82.2%
0.49      3386   1.3797e-2 @ 6.71     +12.3%      1.245e-1     6.665e-1       74.7%
```

The α = 0.45 row is the check on the instrument: it returns 1.4217e-2 against
the recorded 1.4216e-2 with the recorded channel shares reproduced exactly, and
the peak time 0.03 away, one sample interval. α = 0.49 is first on all three
estimators; below it the orderings differ, with 0.40 second on the peak and
the dissipation misfit and 0.45 second on the kinetic-energy misfit. A
three-point sweep is therefore already enough to separate the estimators.

These are 32³ numbers and nothing follows from them about the default. The best
misfits are 0.12 and 0.67, so the run does not resemble the reference history
at this resolution, and the weakest filter wins because 82% of the sink is
filter. The fit belongs at 128³ or above, where the filter's share is 37% or
less. The table shows only that the axes move the score, that the score is
reproducible, and that the instrument reproduces the archive.

<a id="the-alpha-sweep-at-128"></a>

### The α sweep at 128³

Five values at 128³, 224 ranks over two rzhound nodes, `cfl = 0.6`,
`filter_interval = 1`, `filter_cfl = 0`, Gaussian smoother, `C_mu = 0.002`, about
twenty minutes each. The α = 0.486 row was measured afterwards, at the value
where the other four interpolate the peak crossing:

```
alphaf   steps   peak -dKE/dt        vs window   KE misfit   -dKE/dt misfit   filter
0.40     11504   1.1814e-2 @ 8.38      −7.76%     8.986e-3     8.903e-2        38.7%
0.45     11737   1.2120e-2 @ 8.89      −5.40%     7.019e-3     5.666e-2        36.4%
0.486    12145   1.2865e-2 @ 8.85      +0.39%     4.919e-3     2.723e-2        29.7%
0.49     12267   1.3004e-2 @ 8.85      +1.46%     4.887e-3     2.526e-2        27.4%
0.499    13240   1.3258e-2 @ 8.88      +3.37%     4.750e-3     2.910e-2        15.7%
```

Both misfits fall steeply from α = 0.40 to 0.49, by 46% and 72%, against 19% and
8% across the same three values at 32³, where the dissipation misfit was not even
monotone. The α signal grew under refinement rather than shrinking with the
filter's share of the sink, which is the opposite of what those shares predicted.
The 32³ misfit is dominated by a resolution error common to every α, at 0.12 to
0.15; at 128³ that floor is fifteen times smaller and α accounts for most of
what remains. The history fit is usable at this resolution and was not usable
at 32³.

**The dissipation misfit has a minimum at α = 0.49**, the first interior
optimum in this calibration. It is a property of this resolution rather than a
constant, and is absent at 256³
([the 256³ confirmation](#the-256-confirmation)). The natural axis is
`1 − 2α`, which sets the per-pass strength and takes the values 0.20, 0.10,
0.02 and 0.002 at the four original points. On that axis the kinetic-energy
misfit falls by 6.5e-3 per decade over the first interval, then 3.0e-3, then
0.14e-3: it has reached a floor that α cannot lower. The dissipation misfit
turns instead, 15% worse at α = 0.499 than at 0.49 after improving 55% over the
preceding interval. The 32³ sweep produced no such point: the weakest filter
scored best on every estimator there, a boundary fit at the least filtering the
run survives rather than an optimum.

**The peak crosses the reference between α = 0.45 and 0.49.** At 32³ α moved
the peak 3% and not monotonically; here it moves 12% over the four points,
monotonically, and changes sign. Interpolating in log(1 − 2α) places the
total-dissipation match at 1 − 2α ≈ 0.028, α ≈ 0.486, just below the misfit
minimum, so the two estimators agree on a band of roughly 0.485 to 0.49. The
peak overshoots by 3.37% at α = 0.499, and its response is flattening as well,
1.9 points per decade over the last interval against 9.8 over the one before.
Running α = 0.486 confirms the interpolation and separates the two estimators:
its peak is 0.39% high, the closest match in the sweep, but its misfits are
0.7% and 7.8% worse than those at α = 0.49, so the misfit minimum stays at 0.49
and the peak match sits at 0.486. None of this restores the peak as a fit
criterion on its own. The filter is a grid-scale sink whose dissipation is
numerical, so a setting that reproduces the total says nothing about where the
energy sits in wavenumber, which only the spectra show. The peak time carries
even less information: at α = 0.40 it sits on a 1.6% plateau from t = 8.15 to
8.79, so the location of the maximum is noise at this resolution.

The mechanism is visible in the channel split. Molecular, μ\* and filter shares at
the peak are 59.6 / 1.7 / 38.7 at α = 0.40, 61.1 / 2.4 / 36.4 at 0.45,
68.0 / 4.6 / 27.4 at 0.49 and 75.9 / 8.4 / 15.7 at 0.499. Across the first
tenfold reduction in `1 − 2α` the filter's dissipation falls only 22%, from
4.57e-3 to 3.56e-3, while molecular dissipation rises 25% and μ\* triples. A
weaker filter leaves more energy at the grid scale and acts on that larger
amplitude, so its dissipation responds far less than its coefficient. The
compensation is incomplete, and the total rises 10% here rather than holding as
it does at 32³ under a timestep change
([the timestep](#the-timestep-moves-the-attribution-not-the-total)).

Over the last decade the filter's dissipation falls 41%, and the resolved and
μ\* channels more than replace it: molecular dissipation rises 14% and μ\* by
86%, for a further 2% on the total. **The minimum is therefore a joint one.**
μ\* grows from 1.7% of the sink at α = 0.40 to 8.4% at 0.499, and from 2.8% to
11% of the molecular dissipation, so as the filter weakens the Cook viscosity
takes over the grid-scale sink and the overshoot at 0.499 is partly its. The
turnover is therefore the best α at `C_mu = 0.002` and not a property of the
filter alone. A `C_mu` fit is conditional on α in the same way as on the CFL,
so the two have to be closed together, which is the order N1 and N4 already
assume.

**The α = 0.45 row does not reproduce the archive 128³ configuration**,
although it matches it in resolution, rank count, CFL, cadence and `C_mu`:
1.2120e-2 at t = 8.89 against 1.2044e-2 at t = 9.06, 11,737 steps against
12,739, and less wall time per step. The archive campaign predates the adoption
of the Gaussian smoother by two weeks, so its 128³ rows are `:compact` runs. A
32³ A/B on the current code, otherwise identical, gives 3516 steps under
`:compact` against 3078 under the Gaussian, 14% more, the same sign and order
as the 8.5% at 128³. The per-step difference is consistent with
`plan_direction` skipping the Gaussian's line solve and its collective
interface stage, which costs more at 224 ranks than the 3.8% it costs serially,
although six weeks of package changes also separate the two runs. The same 32³
run reproduces the recorded shakeout row exactly, 3078 steps and 1.4217e-2 at
t = 6.61 with 12.6 / 5.1 / 82.2, so the instrument is not the difference. The
two 128³ configurations agree to 0.63% in the peak and 0.7 points in every
channel share.

### Spectra

`bench/tgv_energy.jl snapshots=<times>` writes an HDF5 checkpoint at each listed
instant and `bench/tgv_spectrum.jl` takes the shell-averaged kinetic-energy
spectrum from it offline. The solver computes no transform: there is no
distributed FFT and no new package dependency, and the postprocessor runs from
any project carrying HDF5 and FFTW. `run!` shortens a step to land exactly on
each instant, so snapshots are comparable across a sweep.

The two α runs above, at t = 9, with `E(k)` normalized so that its sum is the
volume-averaged kinetic energy:

```
alphaf   sum E(k)    E(k=8)     E(k=12)    E(k=16)    share above k = 8
0.40     5.930e-2    5.58e-4    1.18e-6    4.76e-10       0.39%
0.49     6.569e-2    2.88e-3    1.36e-4    1.21e-6        4.03%
```

The spectra separate the two settings by a factor of ten in the grid-scale band
and by three and a half decades at k = 16, where the histories separated them by
13% and the peak by 3%. At α = 0.40 the compensated spectrum `k^(5/3) E(k)` is
flat only to k ≈ 7 and then collapses; at 0.49 it holds to k ≈ 10. The stronger
filter empties the band above half the Nyquist wavenumber and takes the top of
the inertial range with it, and it has removed 11% more total energy by t = 9.

For this reason N1 includes spectra as well as histories. −dKE/dt is one number
per instant, and the sinks compete for a supply fixed at the large scales
([the timestep](#the-timestep-moves-the-attribution-not-the-total)), so two
filter settings can reach nearly the same total while distributing it very
differently in wavenumber. At 32³ the high-wavenumber share is the scalar that
distinguishes them and the history misfits do not. The two exchange roles at
128³ ([the spectra at 128³](#the-spectra-at-128)).

The sum of the spectrum reproduces the solver's own kinetic energy to 4e-4
relative at both settings, which is the Parseval check on the normalization; the
residual is the density fluctuation, since the spectrum is taken on velocity and
the solver's energy is density-weighted.


<a id="the-spectra-at-128"></a>

### The spectra at 128³

The snapshots of the sweep above at t = 9, against the reference's energy at that
instant, 8.6404e-2:

```
alphaf   sum E(k)     vs reference   share above k = 32
0.40     8.4654e-2       −2.02%            0.110%
0.45     8.5096e-2       −1.51%            0.267%
0.49     8.5428e-2       −1.13%            0.681%
0.499    8.5361e-2       −1.21%            1.326%
```

Every setting has lost too much energy by t = 9, and the deficit is smallest at
α = 0.49, a third estimator agreeing with the dissipation misfit's minimum and
with the peak crossing at α ≈ 0.486. The band above half the Nyquist
wavenumber, by contrast, grows monotonically by a factor of about 2.4 per step
in α while `1 − 2α` falls by factors of two, five and ten. Its content responds
far less than proportionally to the filter's strength, as the filter's
dissipation does, and shows no feature at the optimum.

Below k ≈ 15 the spectra agree to a few percent and are not consistently
ordered, which is realization scatter rather than an α effect. They separate
monotonically only from k ≈ 18 upward, by 15% at k = 18 and 40% at k = 23. At
32³ the same comparison separated two settings by a factor of ten at k = 8 and
by three and a half decades at k = 16, the Nyquist wavenumber there. At 128³
the filter leaves the resolved spectrum unchanged and sets the content of the
top quarter of the wavenumbers, a band holding under 1% of the kinetic energy.

**The excess dissipation is concentrated in transition rather than at the
peak.** Against the reference at the same instants, −dKE/dt is 11.8%, 9.7% and
6.5% high at t ≈ 4.4 for α = 0.40, 0.45 and 0.49; 6.8%, 5.3% and 3.5% high at
t ≈ 5.95; and 11.1%, 6.9% and 2.6% high at t ≈ 8. The run's own peak then
arrives early and turns over below the reference's. The peak deficit recorded
above is a shape difference rather than a shortage of dissipation: the run
dissipates too much and too early through transition, where a grid-scale sink
has the least to model, and reaches the reference's peak time with less energy
left to dissipate. The run figures are windowed and the reference ones
instantaneous, which biases the comparison by well under half a percent
([the window](#read-the-rate-over-a-window)).

At Re = 1600 and ε ≈ 1.28e-2 the Kolmogorov scale is η ≈ 1.18e-2, so `k_max η`
is 0.75 at 128³ and 1.5 at 256³. The dissipation range is not resolved at the
screening resolution and the numerical sink supplies part of it, so an α fitted
here is a subgrid tuning for this resolution rather than a constant. The 256³
confirmation shows that the fitted value does move
([the 256³ confirmation](#the-256-confirmation)).

**The tails carry no pile-up at any α, including the one the histories
reject.** Every tail steepens toward Nyquist rather than flattening: the decay
per wavenumber across k = 32 to 48 and then 48 to 64 is 0.69 then 0.61 at
α = 0.40, 0.75 then 0.63 at 0.45, 0.83 then 0.75 at 0.49, and 0.91 then 0.87
at 0.499. At Nyquist the four settings sit at 3.3e-11, 3.3e-10, 5.9e-8 and
2.5e-6, so α = 0.499 holds 43 times the grid-scale energy of the fitted value
and 75,000 times that of the strongest filter, and is still three and a half
orders below the spectral peak. The criterion the postprocessor was built
around is therefore one-sided at this resolution: the emptied band is present
at α = 0.40, and the pile-up is absent even at α = 0.499, where the
dissipation history has already turned. The history is the more sensitive
indicator here.

The share of the reference's dissipation that the resolved field carries by
itself also rises with α: molecular dissipation at the peak is 55%, 58%, 69%
and 78% of the reference total at α = 0.40, 0.45, 0.49 and 0.499. Even at the
weakest filter a fifth of the physical dissipation is not on the grid, as
`k_max η` = 0.75 implies. The link runs the other way as well, since molecular
dissipation is weighted by k²: the fitted α corresponds to a grid-scale energy
content of about 0.7% above half Nyquist, between the 0.11% the strongest
filter leaves and the 1.33% at α = 0.499.

**The high-wavenumber share is monotone in filter strength and has no feature
at the optimum.** At 32³ it was the discriminating instrument because the histories
were unusable there. At 128³ the roles reverse: the histories carry the
minimum, and the share is monotone in α with no feature at it, because the
filter's spectral footprint sits in a band holding under 1.5% of the energy.
The share is a bounding check, an emptied band below and a pile-up above; the
fit belongs to the history misfit at any resolution where the history is
meaningful.

<a id="the-battery-under-alpha"></a>

### The shock battery under α

`bench/artcal.jl filter` runs the one-dimensional battery at five filter
strengths, each case at its production settings: Noh at `cfl = 0.15`,
Lax and Shu–Osher at 0.4, Woodward–Colella at 0.3. The 0.486 row is the
peak-crossing value from the sweep above.

```
alphaf   Noh1 plat  deficit | Noh2 plat | Noh3 plat | Lax L1  | Shu amp | WC peak
0.40      0.9997     +61%   |  0.9365   |  0.9748   | 5.1e-3  | 1.6146  | 6.5846
0.45      0.9993     +64%   |  0.9367   |  0.9751   | 5.0e-3  | 1.6180  | 6.6050
0.486     0.9990     +61%   |  0.9376   |  0.9766   | 4.9e-3  | 1.6192  | 6.6093
0.49      0.9989     +59%   |  0.9377   |  0.9769   | 5.0e-3  | 1.6192  | 6.6251
0.499     0.9987     +49%   |  0.9372   |  0.9763   | 5.7e-3  | 1.6223  | 6.6887
```

Every case completes at every strength, the Woodward–Colella collision at a
10⁵ pressure ratio included, so the battery's robustness floor is met
throughout.

The columns whose reference value comes from outside the code agree with the
Taylor–Green fit. The Noh plateaus are exact at 4, 16 and 64, and both curved
geometries are closest at α = 0.49 and fall back at 0.499. The Lax L1 error
against the exact Riemann solution is flat from 0.45 to 0.49 and 14% worse at
0.499. The planar wall deficit falls from 64% to 49% as the filter weakens, the
same error that the one-sided closures reduce from the other side
([the wall cascade](#the-filters-wall-cascade)), and the Shu–Osher train
amplitude rises monotonically, since a weaker filter smears the train less. The
same bound as the 128³ history therefore follows from three one-dimensional
cases, none of which was used to fit it.

The two candidate values inside the band are not separable here. α = 0.486
matches 0.49 on the Shu–Osher amplitude to four digits and on the Noh plateaus
to the fourth, trailing it by 0.0003 on the spherical plateau and leading it by
2 points on the planar wall deficit. It is the best row in the table on Lax,
4.9e-3 against 5.0e-3 at both neighbours, the only column besides Noh with a
reference value independent of the code. The choice within 0.485 to 0.49
therefore rests on the Taylor–Green misfit, now measured at 0.486 in the sweep
above.

At the CFL ceiling the reading is less clean:

```
cfl   alphaf | Noh1 plat  deficit | Noh2 plat | Noh3 plat | WC peak
0.4   0.45   |    NaN      NaN    |   NaN     |   NaN     | 6.6106
0.4   0.486  |    NaN      NaN    |   NaN     |   NaN     | 6.6221
0.4   0.49   |    NaN      NaN    |   NaN     |   NaN     | 6.6368
0.3   0.45   |    NaN      NaN    |   NaN     |  0.9761   | 6.6050
0.3   0.486  |  1.0004    −208%   |   NaN     |  0.9774   | 6.6093
0.3   0.49   |  1.0000    −147%   |   NaN     |  0.9775   | 6.6251
0.2   0.45   |  0.9995     +59%   |  0.9375   |  0.9756   | 6.5947
0.2   0.486  |  0.9992     +54%   |  0.9381   |  0.9770   | 6.6094
0.2   0.49   |  0.9992     +52%   |  0.9382   |  0.9773   | 6.6065
```

Under the Gaussian smoother the planar wall and the cylindrical axis complete
at `cfl = 0.2` and fail at 0.3, matching the pair of ceilings recorded above,
while the spherical origin completes at 0.3 and fails at 0.4, where the
recorded ceiling is 0.4. That disagreement is a granularity or policy
difference and is not resolved here; it does not affect the α comparison,
which varies one setting inside one table.

α moves no ceiling. The single apparent exception is not a raised one: the
planar case at `cfl = 0.3` keeps positivity at α = 0.486 and 0.49 where
α = 0.45 loses it, but returns plateaus of 1.0004 and 1.0000 with wall deficits
of −208% and −147%, an excess where every healthy row carries a deficit, and
larger at the weaker of the two failures. That is a changed failure mode rather
than a working configuration, and it sets in somewhere between α = 0.45 and
0.486.

**α = 0.49 is not excluded by the battery, and the battery results do not
justify moving the default either.** Every constant in this file was fitted
under `compact_filter(0.45)` applied every step, so a change to the filter puts
`C_beta`, `C_kappa`, `C_D` and `C_Y` back in question. `bench/artcal.jl` sweeps
each of them against these same cases, and those sweeps have to be re-run at
the new value before the default moves.

<a id="cadence-and-alpha-are-one-axis"></a>

### Cadence and α are one axis

`filter_interval` dilutes the filter in time and α weakens each pass. Whether
the two have to be fitted jointly is one of the questions under N1. Six points
at 128³, the two α values crossed with `filter_interval` 1, 2 and 4, with the
other settings those of the sweep above:

```
interval  alphaf   steps   peak vs window   KE misfit   -dKE/dt misfit   filter
1         0.45     11737       −5.40%        7.019e-3     5.666e-2         36.4%
2         0.45     11938       −1.50%        5.447e-3     3.881e-2         33.2%
4         0.45     12191       +0.65%        4.913e-3     2.627e-2         29.0%
1         0.49     12267       +1.47%        4.887e-3     2.526e-2         27.4%
2         0.49     12521       +3.07%        4.911e-3     2.682e-2         24.4%
4         0.49     12821       +3.12%        4.825e-3     2.800e-2         20.2%
```

The rows are ordered by `(1 − 2α)/interval`, the per-pass strength divided by the
cadence, which takes the values 0.10, 0.05, 0.025, 0.02, 0.01 and 0.005 down the
table. On that axis the dissipation misfit is single-valued with its minimum
at 0.02, and the α sweep's own points, at 0.20, 0.10, 0.028, 0.02 and 0.002,
lie on the same curve with the same minimum. **Cadence and α are not
independent settings.** The two settings closest on the combined axis, α = 0.45 at
`interval = 4` and α = 0.49 at `interval = 1`, are 25% apart on it and agree to
0.5% in the kinetic-energy misfit, 4% in the dissipation misfit and 1.6 points
in the filter's share of the sink, which is the resolving power of the
instrument here ([the fit instrument](#the-fit-instrument)).

Diluting the filter in time is equivalent to weakening the pass, so
`filter_interval` is redundant with α for this fit. Fit α at `interval = 1`
and leave the cadence at 1. The equivalence is established on smooth turbulence
only. A strong pass applied every fourth step leaves grid-scale energy
unchecked for three steps, and whether that matters is a question for a
shocked case; the Taylor–Green histories do not resolve it.

<a id="the-relaxation-leg"></a>

### The relaxation leg

`filter_cfl` is the reference CFL at which one pass is full strength; below it
each pass is scaled by the running CFL, so that the filter's dissipation per unit
time stops depending on the timestep ([the relaxed
formulation](#the-relaxed-formulation)). Four points at 128³ and α = 0.45, `cfl`
0.3 and 0.6 crossed with `filter_cfl` 0 and 0.6, and deliberately without
snapshots: `run!` shortens a step to land on a snapshot instant, and a shortened
step pays a full pass under `filter_cfl = 0` and a scaled one otherwise, which
would bias this comparison in particular.

```
cfl   filter_cfl   steps   peak vs window   peak time   KE misfit   -dKE/dt misfit
0.3   0            23021       −8.04%          −0.68     8.867e-3      8.896e-2
0.6   0            11737       −5.41%          −0.05     7.019e-3      5.666e-2
0.3   0.6          23472       −5.20%          −0.08     7.013e-3      5.827e-2
0.6   0.6          11737       −5.41%          −0.05     7.019e-3      5.666e-2
```

The relaxed formulation removes the CFL dependence, as intended. Unrelaxed,
halving the CFL doubles the number of passes and costs 26% in the
kinetic-energy misfit, 57% in the dissipation misfit and 0.63 in the peak time,
which is 7% of the run. Relaxed, the two CFL numbers agree to 0.1% in the
kinetic-energy misfit and 2.8% in the dissipation misfit, and their peak times
agree to 0.03. The fourth row reproduces the second in every digit, step count
included, because `filter_cfl = 0.6` at `cfl = 0.6` is a weight of one; this
is the instrument's own check on the scaling.

**A lower CFL is currently a stronger filter.** Under the default
`filter_cfl = 0`, halving the timestep, the usual safe response to a marginal
configuration, doubles the numerical dissipation per unit time and moves the
answer further from the reference rather than closer, and nothing in the
output indicates it. Retries, subcycled levels and shortened output steps do
the same thing locally and intermittently. Relaxation removes the dependence,
and it is the property that would let an α fitted at one CFL be used at
another.

Switching it on is not free. The reference CFL fixes the absolute strength, so
`filter_cfl = 0.6` gives a case running at `cfl = 0.15` a quarter of the
filtering per unit time it receives today. The one-dimensional battery runs at
0.15 to 0.4 precisely because those cases are hard to stabilize, so
`bench/artcal.jl filter` has to clear the policy change at the production CFL
numbers before it can become a default. That sweep has not been run.

<a id="the-256-confirmation"></a>

### The 256³ confirmation

Three values at 256³, 896 ranks over eight rzhound nodes, otherwise at the
settings of the α sweep, 1.7 hours each at 65 Mpoint-steps/s:

```
alphaf   steps   peak -dKE/dt        vs window   KE misfit   -dKE/dt misfit   filter
0.45     23178   1.3019e-2 @ 8.86      +1.36%     1.191e-3     1.226e-2         11.8%
0.49     23923   1.2916e-2 @ 8.86      +0.55%     8.122e-4     8.618e-3          6.7%
0.499    24783   1.2838e-2 @ 8.88      −0.07%     5.161e-4     7.457e-3          2.3%
```

**The minimum does not transfer.** Both misfits fall monotonically through
α = 0.499, and the turn the 128³ sweep found there is absent: the dissipation
misfit is 13% better at 0.499 than at 0.49, where at 128³ it was 15% worse. The
peak reverses as well. At 128³ the strongest filter undershot the reference
peak by 7.76% and the weakest overshot by 3.37%; at 256³ every setting
overshoots, and the overshoot falls as the filter weakens, reaching −0.07% at
α = 0.499. The weakest filter is now first on all three estimators, the
boundary fit seen at 32³ and not at 128³.

The reason is visible in the channel split. At fixed α, refinement takes the filter's
share of the sink at the peak from 36.4% to 11.8% at α = 0.45, from 27.4% to
6.7% at 0.49 and from 15.7% to 2.3% at 0.499, and the μ\* share falls with it,
from 2.4% to 0.8% at α = 0.45. Molecular dissipation carries 87% to 96% of the
sink at 256³ against 61% to 76% at 128³. `k_max η` is 1.5 here and 0.75 there,
so the dissipation range is resolved and there is nothing left for a numerical
sink to supply. The best-scoring strength at 128³ is the one that best
replaces the missing dissipation range at that resolution, and at 256³ no
replacement is needed.

**α = 0.45 is too strong at both resolutions, and that conclusion transfers**:
0.49 beats it on every estimator at 128³ and at 256³, and the same ordering
holds independently on the one-dimensional battery ([the shock battery under
α](#the-battery-under-alpha)). The interior optimum does not transfer, so
**α = 0.49 is a 128³ subgrid tuning and not a fitted constant.** The 256³ rows
place no upper bound on α inside the range that was run. The strength below
which a resolved run stops being stabilized is open, and the 256³ leg says only
that it lies above 0.499.

The misfits fall by factors of six to nine and three to five under the
refinement at fixed α, so the resolution error common to every setting is still
falling and the 256³ numbers are not at a floor either. The peak still arrives
early by 0.10 at every α, unchanged from 128³, which is the shape difference
recorded above and not a filter effect.

<a id="the-filters-wall-cascade"></a>

### The filter's wall cascade

`compact_filter` leaves row 1 unfiltered and applies centered F2/F4/F6 rows at
rows 2–4. Measured as one pass |F f − f| on a smooth closed line, that is second
order along the whole line and not only at the wall, because the compact solve
carries the row-2 error inward (`test/convergence.jl`, 1.88 in the max norm, 2.21
in L2). The filter, not the derivative closure, is therefore the wall-order cap of
every filtered run, and its row-2 error is an O(h²) disturbance deposited two
cells from the wall on every step.

`compact_filter(closures = :onesided)` replaces rows 2–4 by the one-sided
eighth-order rows of Gaitonde and Visbal, derived at construction from polynomial
exactness through degree 7 plus a Nyquist zero, which is the interior stencil's
own construction. The derivation reproduces the centered stencil at the centered
point to 1e-16 and the published row 2 to 1e-14. One pass is then eighth order
everywhere (8.07 max norm, 8.75 L2). Rows 2 and 3 taken alone exceed unit gain at
some wavenumbers (1.10 and 1.03 at αf = 0.45; 1.39 and 1.32 at αf = 0), which the
paper notes, but the closed operator as a whole amplifies less under repeated
application than the cascade does: ‖F¹⁰⁰‖₂ is 1.05 against 1.14 at αf = 0.45 and
N = 64, and 1.42 against 1.35 only at αf = 0. Both keep every eigenvalue inside the
unit disk apart from the two exact ones at 1, the constant and the unfiltered end
rows.

The wall deficit of the default configuration is largely the filter's: the
one-sided rows take the planar Noh wall heating from 64% to 27% at N = 400 and 65%
to 38% at N = 800, with the plateau, shock position and Woodward–Colella profile
unchanged to three digits. The [wall-closure
section](#wall-closures-under-the-artificial-properties) carries the full table
and the coupling to the derivative closures.

The one-sided rows stay optional. Every constant in this file was calibrated under
the cascade, the `test/validation.jl` guards are set from it, and no periodic case
can tell the two apart, so switching the default is a recalibration of the wall
cases rather than a code change.

## CFL and the symmetry-cell restriction

```
cfl       | Noh1 plat  deficit | Noh2 plat | Noh3 plat | WC peak
0.4       |     NaN       NaN  |      NaN  |      NaN  | 6.5762
0.3       |     NaN       NaN  |      NaN  |      NaN  | 6.5731
0.2       |  0.9993       +55% |      NaN  |      NaN  | 6.5597
0.15      |  0.9992       +58% |   0.9355  |   0.9732  | 6.5407
0.1       |  0.9995       +58% |   0.9345  |   0.9722  | 6.5138
```

Measured under `smoother = :compact` and retained as the record of that setting.
No setting of the four constants stabilizes a converging strong shock at the
default `cfl = 0.5`, whereas every sampled setting works at 0.15, and accuracy at
0.15 and 0.1 is identical to three digits. Under the default `:gaussian` smoother
the ceilings are 0.4 at the spherical origin, 0.2 at the cylindrical axis and 0.2
at the planar wall.

<a id="where-the-restriction-originates"></a>

### Where the restriction originates

The restriction is a symmetry-plane startup problem: the wall, axis or origin
cell, not the shock front. Five explanations have been proposed and measured
wrong, and each is closed by a number below. Do not reopen them without new
evidence.

**Not the timestep predictor.** `compute_dt` builds its diffusive rate from the
previous step's artificial coefficients, so at a forming shock the step is chosen
from stale coefficients. Linear extrapolation with `StepControl(predict = n)`
moves Noh ν = 1, N = 400, cfl = 0.3 from failure at step 175 with no lookahead to
179 with three steps and 200 with thirty; capping growth at `max_growth = 1.05`
moves it to 186. These delay the failure and do not prevent it. The per-step trace
shows why:

```
step   25  dt=2.1e-4  rate=1418   rho_min=0.951
step   75  dt=1.8e-4  rate=1633   rho_min=0.798
step  125  dt=1.8e-4  rate=1718   rho_min=0.398
step  175  dt=4.8e-5  rate=1.5e4  rho_min=0.252   <- and then negative
```

Density falls for 150 steps while `dt` and the rate stay nearly constant, and only
after positivity is lost does the diffusive rate climb and `dt` collapse. The
cause is spatial, not temporal.

**Not insufficient β\* reach.** Over a complete ν = 1 run at `cfl = 0.15`,
`bench/nohprobe.jl` reports the furthest cell ahead of the front carrying above a
thousandth of the domain maximum β\*, alongside the worst-affected cell:

```
step    x_sh/h | rho_min   i | e/e0_min   i  n_e<0 | b*@e/b*max  reach/h
 500      9.81 | 0.98875  17 |   -469.4  12      7 |      0.166    14.19
1500     27.63 | 0.99008  35 |   -421.7  30      8 |      0.192    14.37
2500     45.47 | 0.98851  53 |   -465.4  48      7 |      0.161    14.53
3500     63.30 | 0.98720  71 |   -498.7  66      7 |      0.140    14.70
```

Reach holds at 14.2–14.8 cells while the worst cell sits 3–5 cells ahead of the
front, a margin of three to five held for the whole run. Widening the sensor
stencil would address a deficit that is not present.

**Not insufficient β\* magnitude.** The `b*@e/b*max` column above stands at
0.14–0.21 throughout, so the affected cell carries a sixth of the peak artificial
bulk viscosity in the domain. At the failing `cfl = 0.3` the same column reads
0.84–1.00 over the first 125 steps, placing the worst cell at or near the β\*
maximum itself.

**Not the fold closure.** The fold is sixth to seventh order and the most accurate
region of the line; see [fold order](#fold-order-and-geometry-limits).

**Not sensor blindness at the fold.** During the excursion that fails, β\* at the
origin reaches the line maximum under both detectors; see [the origin
cell](#the-origin-cell-is-a-startup-transient).

The failure starts at the symmetry plane. At ν = 1, `cfl = 0.3` the first cell to
degrade is the wall cell i = 1, whose internal energy is negative by step 5, and
the density hole the trace reports at step 125 is at i = 3, between the wall and a
front then at cell 4.4, with the pre-shock field still within 1% of unity from
cell 11 outward. The ν = 3 origin fails the same way and more abruptly: one step
before the failure at step 172 the density minimum over the whole line is still
1.92, and the origin cell then carries an outward u = +5.6 against an inflow of
−1, with e/e₀ = +2.6e5 against −1.6e4 in its neighbour, under a β\* of 5.7 where
the front carries 0.06. This is consistent with `StepControl(retries = 4)`
recovering afterwards, since the restriction applies while the shock forms at the
symmetry point.

The one setting that moves a ceiling acts there too. `:gated_strain` at ν = 2,
`cfl = 0.2` migrates the worst-energy cell off the axis (i = 1 to i = 6…11) and
reduces its magnitude about sixtyfold, from e/e₀ = −5.5e4 to −90…−212, where the
ungated sensor holds i = 1 at −1.1e4 to −1.2e4 indefinitely.

<a id="sensor-intermittency-at-the-damage-site"></a>

### Sensor intermittency at the damage site

The artificial-property path does set the restriction at the fold, through the
continuity of β\* rather than its magnitude. Sampling the ν = 3 origin at 25-step
intervals, where both smoothers complete at `cfl = 0.15`:

```
step        150    175    200    225    250    275    300    325    350    375    400
compact    0.393  0.004  0.302  0.007  0.305  0.009  0.300  0.011  0.306  0.007  0.213
gaussian   0.103  0.336  0.118  0.429  0.415  0.120  0.143  0.131  0.108  0.375  0.130
```

Under `:compact` the cell being damaged carries about 30% of the domain maximum on
one sample and under 1% on the next; under `:gaussian` it never falls below 0.100.
`reach/h` is 6–11 cells for both, so reach is again not the discriminating
quantity, and `n_e<0` holds at 7–9 for both. The same signature precedes the
failure at `cfl = 0.3`, where `:compact` reads 0.002, 0.045, 0.251, 0.025, 0.270
before losing density at step 172 while `:gaussian` completes 400 steps with a
floor near 0.09.

The discriminating quantity is sensor roughness. The undivided δ⁴ applied to |S|
produces a spiky field; the damaged cell drifts outward over the run (i = 30 to
39) and, under a smoother close to the identity across the resolved band, drifts
alternately onto spikes and into troughs. A nine-point Gaussian spreads each spike
widely enough that no trough remains to fall into.

Two consequences follow. A measurement of the detector must be taken on top of the
Gaussian smoother, since a sharper high-pass produces narrower spikes and on its
own would be expected to worsen intermittency; the reference implementation pairs
`:d8` with the Gaussian and never with a near-identity filter. And the ν = 1 probe
above samples every 1000 steps, which cannot resolve an alternation of this
period, so its steady 0.14–0.21 is consistent both with a steady sensor and with
the average of an alternation. Re-running that probe at `every = 25` would settle
it and has not been done.

<a id="the-origin-cell-is-a-startup-transient"></a>

### The origin cell is a startup transient

`bench/nohprobe.jl` reports the symmetry cell on every line, not only when it is
the worst cell, because the argmin columns track the front for most of a run and a
symmetry cell degrading underneath them stays invisible until it overtakes the
front. Spherical Noh, N = 256, sampling `rho1/rho2` (the symmetry cell over its
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

The symmetry cell is quiescent for most of the run: through step 85 it holds
`rho1/rho2` to within 0.2% of unity and carries β\* at a thousandth of the line
maximum, so whatever sets the ceiling does not act gradually from the start. The
sensor is not blind at the fold: during the excursion β\* at the origin reaches the
domain maximum, 1.000, in both surviving configurations. Every configuration has
the excursion, and the ceiling is whether the cell survives it. `:delta4` at
cfl 0.3 and `:d8` at cfl 0.25 both pass through and continue to `t_final`; `:d8`
at cfl 0.3 enters it about forty steps earlier and the cell evacuates within one
sampling interval, `rho1/rho2` falling 1.3089 to 0.2257 with the internal energy
reaching −6335 e₀.

The excursion is physical, not grid-scale. Peak of `b*1/max` under refinement,
`:delta4` at cfl 0.3:

```
N      step   t          peak b*1/max
128      58   0.377-0.382    1.000
256     160   0.392-0.395    1.000
512     342   0.393-0.395    0.950
```

The step number scales with N while the time does not: the excursion lands at
t ≈ 0.394 at every resolution and its amplitude relative to the line maximum
weakens. It is a resolved feature of the warm start at t₀ = 0.3.

At the moment of failure the regularization is suppressed by the evacuation
itself. β\* is proportional to density by construction, `C_beta * rho * sensor`
(`src/artificial.jl`). At the failing step the symmetry cell has thinned to 0.23 of
its neighbour and β\* there has fallen from 0.304 to 0.018 of the line maximum
while the cell is the worst in the domain; the step-126 profile puts ρ = 38.1 at
the origin against 136.7 and 162.1 at the next two cells, with the velocity at the
origin reversed to +1.19 against −0.51 in its neighbour. This is the specified
behaviour of Cook's formulation and not a defect in it. The mechanism is
consistent with the numbers but not demonstrated; establishing it requires a
β\* that does not vanish with the density, which has not been run.

### Negative internal energy in completed runs

The `n_e<0` column is not zero in any sampled configuration, including every run
that completes. Six to eight interior cells carry negative internal energy,
travelling with the front, for the entire duration of the ν = 1 validation case.
No diagnostic reports it by itself: `primitives!` floors T_ion at 1e-300 wherever
e ≤ 0, so p becomes ρ·R·1e-300 and the run continues, while the positivity check
in `max_rate` reads ρ, which stays positive.

The quantity is ill-conditioned in this problem. At the Noh ambient p₀ = 1e-4 the
internal energy is 1.5e-4 while the kinetic energy is 0.5, so e is recovered as a
difference of terms that agree to within 0.03% and a rounding-level error in
either produces a sign error. Since the κ\* sensor is built on e, this also bounds
the artificial conductivity.

The affected cells are not a rounding step below zero. Over the complete ν = 1
validation case, 3756 steps at cfl 0.15 returning a plateau of 3.9971 against the
exact 4:

```
cell-steps with e < 0        25179
cell-steps with E <= 0           0
cell-steps with rho <= 0         0
```

The worst cell reaches several hundred ambient internal energies below zero: the
sampled line at step 2000 carries seven such cells and a minimum of −428 e₀. Total
energy density and mixture density stay positive at every cell of every step, so
the state never leaves the set any frame can represent. The wall region runs as a
pressureless layer, `primitives!` maps the whole of it to T_ion = 1e-300, and the
calculation still reaches the plateau to within 0.07%.

**Repairing those cells is a percent-level intervention and terminates the run.**
Under `StepControl(floor_ratio = 1e-8, floor_scope = :internal_energy)` the
failsafe repairs five cells on step 1 and 256 cell-steps in all, adding 1.4007 of
mass and 9.3e29 of energy and removing 1.7e10 of momentum before the run fails at
step 19 with `:dt_collapse`. The repair damps the velocity at a cell whose total
energy is still positive and raises the total energy where there is no kinetic
energy left to convert; both are the same order of intervention, so the cost
belongs to the case and not to the choice of repair.

The default `floor_scope = :representable` follows: it repairs only what no frame
can represent and counts the rest, so on this case it repairs nothing and
reproduces the unfloored run exactly (3756 steps, plateau 3.9971, shock 0.2044,
pre-shock L1 5.99e-07) while reporting all 25179 cell-steps. Reproduce the tally
with

```
julia --project=. -t 1 bench/nohprobe.jl 1 cfl=0.15 nmax=5000 every=2000 \
      floor=1e-8 scope=representable
```

### Consequences for the state-validity policy

[State validation](DESIGN.md#state-validity-and-its-policy) puts admissibility to
the EOS, and a calorically perfect gas answers that a cell with e < 0 is outside
its domain. That is correct, and it is a property of 25179 cell-steps of a run
that reaches the right answer. The state this case ends on carries six such cells
of four hundred, at e = −0.051, so a strict check applied to the returned state
rejects a completed and correct Noh run. The shock/SF6 case ends with six of four
hundred points whose mass fraction is below −`Y_tolerance`, having first crossed
that band at step 32 of 646. Any per-step or returned-state check on these two
cases therefore has to run under `validity = :permissive`.

Reporting is not the same as accepting whatever appears. Both cases are guarded on
the state they end with as well as on their solution error: `test/validation.jl`
bounds the Noh cases at twelve inadmissible cells with `e_min > −1`, and the
shock/SF6 case at twelve points outside the mass-fraction band, alongside the
plateau, wall-deficit, shock-position and excursion guards.

The threshold is not confined to converging shocks. A binary interface spanning
about one cell overshoots the mass-fraction bound by 1.5e-2 within two steps, 150
times `Y_tolerance` and the same order as the shock/SF6 case's −0.0135, with the
artificial bound then pulling it back. Any under-resolved multi-species run
therefore ends on a state a strict check rejects, which is why the small
configurations in `src/precompile.jl` select `:permissive`.

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

The corresponding ν = 1 calculation requires 4485 steps at a fixed `cfl = 0.15`, so
recovery is approximately three times faster, and `solver.cfl` records the
accepted value for subsequent calculations. Rollback recovers abrupt failures but
not the gradual degradation at cfl = 0.3, because the most recent savepoint is
nonphysical by the time the check fires; that case requires a lower initial CFL.

**Recommendation:** `cfl = 0.5` for smooth and moderately compressible flow, 0.3
with shocks, and `StepControl(retries = 4)` for automatic recovery. For converging
geometry with rollback disabled the ceiling is set by `ArtParams.smoother`: under
the default `:gaussian` it is 0.4 at the spherical origin and 0.2 at the
cylindrical axis and the planar wall.

## Wall closures under the artificial properties

`lele_d1_6(closures = :brady_livescu)` and `lele_d1_8(closures = :brady_livescu)`
raise the smooth-field wall order from 3.17 to 5.88 and 7.91
(`test/convergence.jl`). Brady and Livescu state that their rows are not stable at
discontinuities. The two wall-bounded cases of `test/cases.jl` take `deriv` and
`filt` keywords to test that. With the default filter wall cascade and
`nmax = 20_000`:

```
case                closures          cfl     outcome
Woodward–Colella    :cascade3         0.3     L1 rho 3.252e-2, peak 6.6074 at 0.7785
                    :cascade4         0.3     L1 rho 3.250e-2, peak 6.6075 at 0.7785
                    :brady_livescu    0.3     negative density, step 854, t = 0.0059
                    :brady_livescu    0.15    negative density, step 677, t = 0.0023
                    :brady_livescu    0.075   negative density, step 1000, t = 0.0016
                    C8 :brady_livescu 0.3     negative density, step 43
                    C8 :brady_livescu 0.15    dt collapse, step 273
Noh planar          :cascade3         0.15    plateau 3.9971, wall deficit 64%
                    :cascade4         0.15    plateau 3.9983, wall deficit 58%
                    :brady_livescu    0.15    dt collapse, step 87
                    :brady_livescu    0.075   dt collapse, step 366
                    C8 :brady_livescu 0.15    negative density, step 1
```

Lowering the CFL moves the failure earlier in time, so this is not the startup
restriction of the [CFL section](#cfl-and-the-symmetry-cell-restriction) and
`StepControl(retries = 4)` would not recover it. `:cascade4` is an improvement at
no cost under the default filter: it takes the planar Noh wall deficit from 64% to
58% at an unchanged plateau and shock position and leaves Woodward–Colella
unchanged to four digits.

### The wall mode

The failure is a wall mode driven by the artificial bulk viscosity and not by the
discontinuity. A planar Noh warm-started from the exact solution at t = 0.3 has a
uniform ρ = 4, u = 0 plateau at the wall and nothing reaches the wall before the
run ends; under `:brady_livescu` the wall density still departs from 4 at t ≈ 0.06
and grows in a two-cell alternating pattern, doubling every ≈ 0.01 time units
(4.006 at t = 0.063, 4.087 at 0.076, 4.61 at 0.085) until it ends the run at
ρ_wall = 20.3 against 3.99 under `:cascade3`. Zeroing `C_kappa` and `C_mu`
together leaves that growth unchanged (ρ_wall = 19.8), so β\* is the driver;
zeroing `C_beta` loses the shock instead.

The rows are innocuous wherever β\* is not active at a wall. A 1% Gaussian
pressure pulse between two slip walls, artificial properties on or off and filter
on or off, runs three acoustic transits under C6 `:brady_livescu` with the same
wall density as `:cascade3` to four digits. The mirrored Noh problem on (−1, 1),
inflow at both ends and no wall, runs to completion under `:brady_livescu` with a
final profile identical to `:cascade3`'s, so a captured shock and a Dirichlet
inflow through the rows are both fine.

The scalar spectra do not predict the failure. The sets are separated by the wall
mode of the diffusion operator D(β D) that the bulk term assembles from the
first-derivative rows: at N = 64 and h = 1 the largest real eigenvalue of D² is
1.3e-6 for `:cascade3`, 1.7e-5 for `:cascade4`, 1.4e-4 for C6 `:brady_livescu` and
3.7e-3 for C8 `:brady_livescu` with the end rows free, and −2.5e-3 for all five
with both end rows injected. Only the momentum is injected at a slip wall, so the
density and energy rows see the free-end spectrum.

### Under the one-sided filter rows

The other ingredient is [the filter's wall cascade](#the-filters-wall-cascade),
whose row-2 error is an O(h²) disturbance deposited two cells from the wall on
every step. With that row replaced the mode does not appear:

```
case                closures          outcome
Woodward–Colella    :cascade3         L1 rho 3.259e-2, peak 6.6076 at 0.7785
                    :cascade4         negative density, step 2485, t = 0.019
                    :brady_livescu    L1 rho 3.253e-2, peak 6.6073 at 0.7785
                    C8 :brady_livescu L1 rho 3.265e-2, peak 6.6159 at 0.7785
Noh planar          :cascade3         plateau 3.9982, wall deficit 27%, shock 0.2025
                    :cascade4         plateau 3.65, wall density 89.6, unusable
                    :brady_livescu    dt collapse, step 275
                    C8 :brady_livescu negative density, step 1
Noh planar, N=800   :cascade3         plateau 3.9983, wall deficit 38% (cascade: 65%)
Noh warm t0=0.3     :cascade3         rho[1:4] 3.906 4.042 4.026 3.974
                    :brady_livescu    rho[1:4] 4.036 3.979 4.009 4.007 (cascade: 20.3 at the wall)
                    C8 :brady_livescu negative density, step 757, t = 0.126
smooth pulse        C8 :brady_livescu completes (cascade filter: negative density, step 36)
```

Three results follow. The wall deficit of the default configuration is largely the
filter's, 64% to 27% at N = 400 and 65% to 38% at N = 800, with plateau, shock
position and Woodward–Colella profile unchanged to three digits. C6
`:brady_livescu` becomes usable at a shock-bounded wall, since the wall mode is
gone from the warm-started Noh and Woodward–Colella completes; it still cannot
take the singular t = 0 start of cold Noh, where u jumps from −1 to 0 on the wall
row itself. And `:cascade4` depends on the F2 row: without it its negative-real-part
eigenvalues are no longer damped (−8.9e-3 at N = 32 under inflow injection, where
`:cascade3` and `:brady_livescu` are entirely in the right half-plane) and it fails
even the smooth pulse at t = 0.915.

**The two knobs are coupled:** `:cascade4` with the cascade filter, or `:cascade3`
or C6 `:brady_livescu` with the one-sided filter.

### Float32

The Brady–Livescu conditioning is the error in Float32. On the smooth closed-line
derivative of `test/convergence.jl`, wall error:

```
N     C6 BL f32  f64        C8 BL f32  f64        cascade3 f32  f64      cascade4 f32  f64
24    1.02e-3    1.21e-3    2.48e-3    2.03e-3    6.24e-3  6.26e-3        8.40e-4  8.35e-4
48    1.21e-3    1.94e-5    2.36e-3    2.46e-6    6.88e-4  6.71e-4        1.31e-4  5.22e-5
96    2.80e-3    3.48e-7    1.49e-3    3.52e-8    8.99e-5  7.71e-5        9.06e-5  3.15e-6
192   4.52e-3    7.42e-9    3.33e-3    2.92e-10   9.82e-5  9.22e-6        3.24e-4  1.92e-7
```

The Brady–Livescu sets floor between 1e-3 and 5e-3 absolute on a derivative of
magnitude 8, about four digits, and rise with N; the cascade floors near 1e-4.
From N = 48 up the default closure is the more accurate one in Float32, which is
the precision the device path runs at. `test/float32_validation.jl` pins this.

## Fold order and geometry limits

`test/convergence.jl` reports a global max norm, and every one of its fold studies
closes the outer end with a `SlipWallBC` whose closure rows measure 3.17 on their
own. Splitting that norm by region (`bench/foldorder.jl`, same fields and
resolutions) separates the two ends:

```
study                              fold(1:3)   mid    outer(3)   global argmax
C6, both ends walls (control)         3.23     3.19     3.17       i = n
cylindrical axis, odd  (u_r-like)     6.05     3.76     3.71       i = n
cylindrical axis, even (scalar)       7.01     2.99     3.00       i = n
spherical origin, even (scalar)       7.00     2.97     2.99       i = n
spherical origin, odd  (u_r-like)     6.07     3.86     3.81       i = n
```

**The global maximum sits at the outer wall in every fold study.** The fold's own
error converges at 6.05 to 7.01 and is three to five orders of magnitude below the
interior: at N = 96 the spherical origin carries 7.0e-12 against 1.9e-7 in the
middle of the line. The fold is the most accurate region of the line. Every global
error `test/convergence.jl` prints equals the outer-window norm to every digit
printed (7.707e-05, 1.953e-07, 1.992e-06 and 4.730e-06 at the finest resolution of
each study), so the guarded numbers in that file are measurements of the outer
wall taken through a norm insensitive to the fold.

The control establishes that the split is meaningful: with walls at both ends the
same window reports 3.23, so the instrument does detect a third-order closure
where one is present. The middle of the line converges at 3 as well, which is the
compact scheme's line-global coupling carrying the wall's closure error inward and
not a property of the fold. Both parities were measured, including the odd one
that `test/convergence.jl` does not cover and that a converging calculation
differentiates at the origin.

### Geometry limits

**The spherical origin requires initial data resolved over ≳3 cells.** A blast
initialized as a top hat with a 1–2 cell transition loses positivity within tens
of steps; at 3 cells and wider it runs to completion. The cylindrical axis accepts
a 1-cell transition and the same top hat completes in Cartesian, so this is
specific to the origin fold and its antipodal pairing. `test/cases.jl` therefore
initializes Sedov with a Gaussian deposit. Why the origin fold is less forgiving
than the cylindrical axis is open; the fold order above rules out the closure.

**The spherical origin is incompatible with the singular t = 0 start of Noh.**
Every CFL and every constant setting fails, since the exact solution requires 64×
compression to appear at r = 0 instantaneously. A warm start from the exact
solution at t = 0.3 integrates to 0.6 and tests whether the solver can maintain
the solution through the origin without also testing the initialization
singularity. The cylindrical axis accepts the cold start at 16× compression.

## Grid convergence

```
N         | Noh1 plat  deficit | Lax L1  | mix width
128       |  0.9945       +60% | 1.3e-2  |  0.03581
256       |  0.9994       +59% | 7.1e-3  |  0.01820
512       |  0.9991       +58% | 3.9e-3  |  0.00931
1024      |  0.9992       +56% | 1.9e-3  |  0.00481
```

Three behaviours separate.

- **Lax L1 halves per doubling**, giving first-order L1 convergence for the
  captured discontinuity regardless of interior order. The sixth- and tenth-order
  convergence lives in `test/convergence.jl`, on smooth fields.
- **Interface width halves per doubling**: the regularization follows the mesh and
  does not settle at a fixed physical scale, as required for the Cook artificial
  properties to act as a subgrid model.
- **Wall heating does not converge away**, 60% to 56% over an 8× refinement. This
  is the known character of the Noh problem: the entropy error is deposited once,
  in the first cell at shock formation, and remains there. Its spatial extent
  decreases with the cell size, so the integrated error vanishes while the
  pointwise error does not. Pointwise convergence of the wall-heating deficit is
  not expected for this problem.

## Miranda's set as a package

Miranda's current artificial-property set (Brill, Olson & Bokman 2025, eqs. 22–27)
is the eighth-derivative detector, the max over directions, μ\* from the velocity
components and β\* from the switched dilatation, at constants an order of magnitude
below Cook 2007's in a Δ²/Δt scaling that differs from the cΔ one here by about
1/CFL. Each choice exists as an `ArtParams` option and each is measured alone
above; `bench/artcal.jl miranda` runs the combination through the battery.
`sensors` is `detector = :d8, reduction = :max, mu_sensor = :velocity,
beta_sensor = :dilatation`; `x1` is C_mu = 2.5e-4, C_beta = 0.175,
C_kappa = 2.5e-3, C_D = 5e-4, their values in this scaling; `x4` is four times
that; C_Y = 100 throughout.

```
config             | Noh1 plat   def | Noh2 plat   def | Noh3 plat   def | Lax L1  | Shu tr | WC peak | mix wid | SI minY  wid
default            |    0.9993   +64% |    0.9367   +56% |    0.9751   +27% | 5.0e-03 | 1.6180 |  6.6050 | 0.01820 | -0.0135    4
sensors only       |    0.9997   +51% |    0.9477   +56% |    0.9929   +36% | 4.7e-03 | 1.6289 |  6.4269 | 0.01787 | -0.0176    4
sensors, x1        |    0.9996   +22% |    0.9650   +39% |       NaN  +NaN% | 5.1e-03 | 1.6490 |  6.3467 | 0.01787 | -0.0203    4
sensors, x4        |    0.9997   +48% |    0.9518   +52% |    1.0006   +37% | 4.7e-03 | 1.6321 |  6.3836 | 0.01787 | -0.0184    4
sensors, x1, Cb=1  |    0.9997   +52% |    0.9472   +57% |    0.9923   +36% | 4.7e-03 | 1.6312 |  6.4693 | 0.01787 | -0.0178    4
```

The combination survives both converging geometries at the default CFL, which
`:dilatation` alone does not; whether `:d8`'s own fold closure or the max reduction
is responsible is not separated here, and the CFL ladder was not run. The one
loss, spherical Noh at `x1`, is C_beta = 0.175 and not the dilatation switch:
C_beta = 1.0 with the other three constants at `x1` is indistinguishable from
`sensors only` in every column, so cutting C_mu, C_kappa and C_D to an eighth of
their defaults moves nothing at this battery and C_beta alone governs it,
consistent with the 1.0–4.0 window recorded for `:d8`.

Against the default, the sensors buy the Noh plateaus (ν = 3 from 0.9751 to
0.9929) and Lax, and cost 2.7% of the Woodward peak, 0.7% of the Shu–Osher train,
and 30% on the shocked interface's excursion (−0.0135 to −0.0176), which worsens on
every Miranda row. Mix width is 0.01787 on all four rows across a 20× range of
C_D.

The strongest refit candidate is `x4`, the only row with ν = 3 within 0.06%, at the
second-best wall heating, for 3.4% of the Woodward peak. A C_beta ladder in 0.2–0.7
under these sensors, with the Noh CFL ladder, would locate the spherical bound.
None of it is the default: the package's four constants are calibrated for Cook's
sensors and stay.

## The bulk species channel

`ArtParams.species_flux = :bulk` replaces the Fickian artificial species flux by
one diffusive flux F_q = −D_b ∇q on every conserved variable, with D_b built from
the Fickian channel's own bracket sensed on both the mass and the mole fraction of
every species (`reference/DESIGN.md`, "The species channel"). Everything below was
measured on the one-dimensional cases of `test/cases.jl` at the package's
constants, C_D = 0.01, C_Y = 100, tolerance 1e-4, the Gaussian smoother, D_b in
the timestep, at CFL 0.4. `bench/artcal.jl bulk` reruns the rows that carry two
species.

### What the Brill slab measures

Section 3.1 of Brill, Olson & Bokman (arXiv:2503.12680, 2025) advects a bubble of
density R in gas of density 1 at p = 1, T = 1 and u = 10 for ten periods through a
periodic unit square, with N_p points across the interface and the grid chosen so
that N_p Δ = 0.05, all gases γ = 1.4, and reports completion and the pressure
oscillation at the end: for their traditional Fickian formulation "at best around
1% of the pressure and at worst greater than 25%", stable at N_p = 7 for small R
and needing more points above R = 1000; for their diffused-density formulation
O(1e-6 to 1e-12) and stable at N_p = 7 for R ≥ 100. `brill_slab` in
`test/cases.jl` is the one-dimensional slab analogue with N = 20 N_p and reports
max |p − 1| at ten periods. Under the default channel it reproduces the paper's
picture, somewhat better than the paper reports it:

```
default channel; max|p-1| at ten periods, worst Y over the run, final rho_min, steps
R    | Np | N   | max|p-1| | worst Y | rho min | steps
10   |  7 | 140 | 7.30e-03 | -0.0180 | 0.9974  | 4041
10   | 14 | 280 | 1.44e-04 | -0.0022 | 0.9990  | 7911
100  |  7 | 140 | 1.67e-02 | -0.0821 | 0.9954  | 4207
100  | 14 | 280 | 7.95e-04 | -0.0052 | 0.9982  | 8005
1000 |  7 | 140 | FAIL at step 11 (rho 0.026, Y -0.33)
1000 | 14 | 280 | 7.97e-03 | -0.0141 | 0.9990  | 8350
```

The error is the Fickian channel's alone. At uniform u, p, T every linear operator
of the scheme preserves the uniform state to round-off, and the Fickian enthalpy
flux Σ_k h_k J_k is the one operator that does not at unequal gas constants:
switching that channel off (C_D = C_Y = 0) with nothing in its place takes every
completing row to 1e-12 to 1e-10 while losing two rows to instability (R = 100 at
N_p = 7 fails at step 150, R = 1000 at N_p = 14 at step 1256). Adding the bulk
flux beside the Fickian one moves nothing, so no term added beside the Fickian
flux can remove its pressure error.

### The sensor field

With the bulk flux in place of the Fickian channel, the field D_b is sensed on
decides what it does. X is the mole fraction, Y the mass fraction, XY the maximum
over both.

```
shock_interface at 2h (N = 400): worst Y / width cells / steps
channel           | 5.04              | 100               | 1000
Fickian (default) | -0.0135 / 4 / 646 | FAIL step 427     | FAIL step 454
none              | -0.2496 / 4 / 640 | FAIL step 415     | FAIL step 369
bulk, X           | -0.0304 / 3 / 640 | -0.5653 / 5 / 679 | FAIL step 361
bulk, Y           | -0.0123 / 3 / 644 | FAIL step 393     | FAIL step 484
bulk, XY          | -0.0122 / 3 / 644 | -0.0222 / 7 / 684 | FAIL step 486

brill_slab, ten periods: max|p-1| / worst Y / final rho_min / steps
channel           | R=100 Np=7                          | R=100 Np=14                         | R=1000 Np=14
Fickian (default) | 1.67e-02 / -0.0821 / 0.9954 / 4207  | 7.95e-04 / -0.0052 / 0.9982 / 8005  | 7.97e-03 / -0.0141 / 0.9990 / 8350
none              | FAIL step 150                       | 1.85e-10 / -1.0626 / 0.5192 / 8009  | FAIL step 1256
bulk, X           | 6.52e-11 / -1.1390 / 0.8573 / 4083  | 1.74e-10 / -0.1042 / 0.9268 / 7911  | 1.60e-09 / -70.597 / 0.2805 / 9699
bulk, Y           | 4.98e-11 / -0.0674 / 0.9940 / 4181  | 1.74e-10 / -0.0052 / 0.9985 / 7986  | 1.78e-09 / -0.0128 / 0.9988 / 8349
bulk, XY          | 5.38e-11 / -0.0674 / 0.9940 / 4201  | 1.94e-10 / -0.0052 / 0.9985 / 7986  | 1.45e-09 / -0.0128 / 0.9988 / 8348
```

R = 1000 at N_p = 7 fails on every row, at step 11 to 120, with the density
undershooting beside a jump of 1000 over seven cells before any diffusivity has
acted; R = 10 completes on every row and is not shown.

The two sensors do different jobs. The mole fraction is the volume fraction and
sits on the density jump, so a sensor on it carries the ratio-100 shocked
interface, which X and XY complete and Y and the Fickian channel do not. The mass
fraction on the light side of a ratio-R interface amplifies a volume-fraction
excursion by up to R (Y_l = (1 − V)/(1 + (R − 1)V) at equal γ; the −70.6 on the X
row is V = −1.0e-3), so a sensor on it is what bounds Y, and the mole-fraction
sensor is blind to that excursion. The maximum over both is the reference
implementation's own combination of mass- and volume-fraction detectors (their
eqs. 33–37), and it is at least as good as either and as the default in every
column: the default's Y bounds and ρ_min on the slab, the ratio-100 shocked
interface which the default fails, and pressure at round-off on advection. On the
shock case it does so with a 7-cell width at ratio 100 against the default's 4 at
ratio 5.04. The channel is sensed on XY; the single-field forms exist only in the
prototype.

### Costs and invariances

At equal molecular weights X ≡ Y, the bulk flux of ρY_k at uniform ρ is the
Fickian flux, and both channels' energy fluxes vanish, so `species_advection`
reproduces the default's 10–90% width (0.01819762) to eight digits with the two
mass-fraction profiles agreeing to 5e-14, and the smooth bounded profile of the
order study reproduces the default's L2 errors to seven digits (3.121191e-06 at
N = 32, 1.145817e-10 against 1.145815e-10 at N = 256, order 4.84 → 4.97).

The resting air/SF6 interface at uniform p and T, with the filter and the
artificial properties on, holds max |u| at 1.2e-14, |δp/p| at 3.8e-14 and |δT/T|
at 4.0e-14 over 298 steps while ρ relaxes by 2.0e-2; the default channel's
enthalpy flux drives u to 2.2e-4 and p to 1.3e-4 on the same case.

The channel costs n_cons gradient line solves per direction in place of the
Fickian flux's n_species, four more, plus one further detector and smoother pass
per species.

### Open

Not the default. The four constants are calibrated on the Fickian channel, and the
single-species battery is untouched by the option either way. The vortex-ring/SF6
run that motivated the mass-fraction bound is the case that should decide it, in
three dimensions and with the artificial viscosities active, and it has not been
run under the bulk channel. Two further questions: whether the unequal-γ contact
drift of 5e-3 with no shock is also the Fickian enthalpy flux, which the slab test
cannot see at equal γ and the resting-interface figures above suggest; and the
ratio-1000 failures, which are the transmitted shock's foot on the shock case and
the seven-cell density jump on the slab, neither a species-channel failure and
both untouched by every row above.

## Remaining differences from the reference implementation

Miranda's Fortran kernels, carried by Pyranda in `pyranda/parcop/`, implement the
same Cook artificial-property method. Reading them against `artificial.jl`
identifies four differences that bear on the constants calibrated here. Two are
measured and have their own sections, [the detector](#the-ringing-detector) and
[the sensor fields](#the-sensor-fields-and-the-compression-switch), along with
[the smoother](#the-sensor-smoother). The two below are recorded as read, with
what is analytically established about the gap, to keep a future investigation
from repeating the source archaeology.

### The conservative filter is normalized by a filtered cell volume

`filter` in `parcop/operators.f90` treats non-Cartesian coordinates by filtering
the volume-weighted field and dividing by a cell volume that has been passed
through the same filter once at setup (`CellVolG`, `CellVolS` in
`parcop/mesh.f90`). The filtered field then reproduces a constant exactly on any
non-uniform metric, and the integrated quantity is preserved. The radial parity
flips in the process, because the cylindrical cell volume is proportional to r and
therefore odd across the axis, the same algebra `sigflux` encodes for the flux
products in `rhs.jl`.

`filter_state!` filters the conserved components unweighted. On a uniform
Cartesian grid the two agree identically, so nothing in the Taylor–Green numbers
here would show it. On cylindrical, spherical or stretched grids the filter is not
conservative, and the filter supplies 37% of the energy sink at 128³.

### The CFL rate is normalized differently

Miranda forms `Σ_d |u_d|/Δ_d` for advection and adds `|c|/min_d(Δ_d)` once for the
acoustic part, then takes the diffusive limits as separate minima with their own
coefficients (0.1 and 0.2). `max_rate` sums `(|u_d| + c)/h_d` over active
dimensions and folds the diffusive rate into the same sum.

The sound speed is therefore counted once there and once per active dimension
here. On an isotropic three-dimensional grid the rate computed here is up to three
times larger for the same state, so `cfl = 0.15` corresponds to a step comparable
to `cfl ≈ 0.4` under the reference convention. Apply the factor of three when
comparing any CFL number in this file against the literature. It does not explain
away [the ceiling](#cfl-and-the-symmetry-cell-restriction), because the cases that
establish it are one- and two-dimensional converging geometries, where the two
conventions largely agree.

<a id="no-slip-wall-flux-contract-r5-september-2026"></a>

## The no-slip wall flux contract

The wall correction sets the assembled normal flux before halo exchange and
compact divergence. Each species flux is zero at an impermeable noncatalytic wall.
Adiabatic total-energy flux is zero; isothermal energy flux is
`-(mu0 * cp_mix / Pr + kappa_art) * grad_T_ion[d]`. This removes species enthalpy
transport and the normal `:bulk` species/energy component flux while retaining
pressure and viscous traction. It does not change the derivative closure or filter
coefficients.

`julia --project=. bench/boundaryorder.jl wall_only=true` repeats the original
incompatible linear-temperature probe. Its two energy fluxes change from
`[-0.005, -0.005]` to exactly `[0.0, 0.0]`. The regression also compares the
second compact RHS row with the uncorrected slip-wall case, so an endpoint-only
RHS patch cannot satisfy the test.

`test/wall_flux_tests.jl` separates compatible evolution from that imposition
probe. Measurements below use Julia 1.11.4, Float64, the default C6 closure, unit
length and density, ideal gas R = 1 and gamma = 1.4, and no filtering or
artificial transport unless stated otherwise. Integrations have bounded step
counts and assert the requested final time.

| Check | Measurement | Regression guard |
|---|---|---|
| Insulated conduction, N=33 / 65, temperature max error | 1.3972e-6 / 3.0304e-7 | fine error <5e-7 and reduction >4 |
| Same, absolute trapezoidal domain-energy drift | 4.0894e-8 / 4.3029e-9 | fine drift <1e-8 |
| Species cosine diffusion, N=65, mass-fraction max error | 2.4444e-7 | <3e-7 |
| Isothermal, initial integrated RHS minus boundary heat rate | 2.7576e-6 | absolute defect <4e-6 |
| Isothermal, evolved energy rate minus time-averaged boundary heat rate | 1.8716e-6 | absolute defect <3e-6 |

The insulated temperature is `1 + 0.08 exp(-alpha*4pi^2*t) cos(2pi*x)`, with
`mu0=0.015`, `Pr=0.8`, `alpha=mu0*cp/(Pr*cv)` and final time 0.002. An analytic
momentum source balances its pressure gradient, leaving conduction to evolve
through the computed energy RHS. The species case uses identical species
thermodynamics, `Y1=0.5+0.1 cos(2pi*x)`, `mu0=0.012`, `Sc=0.75` and final time
0.002; pressure and temperature stay uniform to roundoff. At N=65, the separate
instantaneous mixed temperature/composition probe measures interior RHS max errors
3.6932e-6 (species) and 7.7558e-6 (energy), both guarded at 1e-5.

The isothermal case starts at `T=1+0.05 sin(pi*x)`, with `Twall=1`, `mu0=0.01`,
`Pr=0.8`, N=65 and final time 2e-5. Its initial boundary heat rate is
-0.0137444830 and the integrated energy RHS is -0.0137417254. The measured
energy-change rate is -0.0137425903, versus -0.0137444619 from a trapezoidal time
integral of the two endpoint heat rates. The residual includes spatial quadrature
and hard temperature enforcement; it is not a residual normal species flux or an
assertion of exact discrete conservation.

A separate N=33 filter-only probe with alpha=0.35 and high-frequency cosine fields
changes the trapezoidal species-1 mass by +4.5253e-4 and total energy by
+1.1230e-3. Subsequent adiabatic wall energy/species fluxes are still exactly
zero. No global filter-conservation claim follows from this wall fix.

Coverage: artificial conductivity and species diffusivity are seeded independently
of the detector in the direct face tests, including both `:fickian` and `:bulk`;
both precisions, all six physical faces and their corners, ideal/NASA-9 and
stiffened-gas EOS, nonsingular cylindrical/spherical metrics, SwitchableBC, and
KernelAbstractions CPU execution. The 24-case `bench/wallflux.jl` matrix also
passes on the Radeon RX 6800 XT (AMDGPU on Windows), with zero maximum CPU/GPU
evolved-state difference for every precision, normal, thermal condition and
species channel. The hardware probe uses nonzero seeded artificial transport and
physical corners; run it from an environment carrying AMDGPU with
`backend=amdgpu`, or `backend=cpu` for the same assertions on KA CPU.

The before/after allocation and JET audits used Julia 1.11.4, one thread,
`OPENBLAS_NUM_THREADS=1` and `--compiled-modules=no` in both runs. The abstract
face dispatch costs a fixed 16 B per active face and nothing proportional to wall
area:

```
probe                                  before    after
48³ RHS                                 208 B     304 B
48³ five-stage step                    1552 B    2032 B
1-D axis RHS                            720 B     752 B
1-D axis step                          3728 B    3888 B
isothermal wall hook, 256/1024/4096       0 B       0 B
JET reports, RHS / step                  1 / 2     2 / 3
RHS non-concrete SSA count                 92       110
```

The RHS and step figures hold for single-species, two-species C10 and bulk
configurations, and the JET gain is the expected `correct_flux!` dispatch with
every other probe unchanged. These are fixed boundary-dispatch costs, not timings.

## Open items

In approximate priority order.

1. **Fit the compact filter.** It is necessary and sufficient for stability at
   128³, and `filter_interval` and α had never been fitted to anything, so
   every `C_mu` number is conditional on `compact_filter(0.45)` every step. The
   dt-consistency half is [done](#the-compact-filter): the filter dissipates
   per application, the effect is a factor of 3.93 across a 4× CFL change,
   `filter_cfl` makes it a rate, and the relaxed formulation is measured to
   remove the CFL dependence of the answer where the unrelaxed one costs 26% of
   the kinetic-energy misfit across a 2× CFL change
   ([the relaxation leg](#the-relaxation-leg)). The cadence is settled as
   redundant with α ([cadence and α are one axis](#cadence-and-alpha-are-one-axis)).
   α is not settled: the 128³ histories have an interior minimum at 0.49, and
   at 256³ it is absent, with the weakest filter tested scoring best
   ([the 256³ confirmation](#the-256-confirmation)), so the fitted value is a
   subgrid tuning at the screening resolution. Setting a default requires the
   strength below which a run stops being stabilized, which none of the legs
   measured. α = 0.45 is too strong at both resolutions. The default has not moved
   because every other constant here was fitted under it, and turning
   `filter_cfl` on would cut the one-dimensional battery's filtering fourfold
   at its production CFL numbers.
2. **Raise the CFL ceiling at the symmetry cell.** Every explanation proposed so
   far is measured and closed: the timestep predictor, the sensor reach,
   the sensor magnitude, sensor blindness at the fold, and the fold closure
   ([where the restriction originates](#where-the-restriction-originates)).
   `smoother = :gaussian` moved ν = 3 from 0.15 to 0.4 and ν = 2 from 0.15 to 0.2
   through [sensor intermittency](#sensor-intermittency-at-the-damage-site), and
   `detector = :d8` removes the restriction at ν = 1 and ν = 2 outright while
   costing ν = 3. What remains is a symmetry cell that evacuates during [a startup
   transient](#the-origin-cell-is-a-startup-transient) landing at t ≈ 0.394
   regardless of resolution. Two live leads: the density proportionality of β\*,
   which suppresses regularization as the cell thins, and the per-step compact
   filter, supported by the `:d8` ladder cells where the cylindrical axis fails
   [below a CFL rather than above one](#a-failure-that-gets-worse-as-the-timestep-falls).
3. **Refit `C_mu` against the history, not the peak.** The reference is
   vendored, and the peak is unusable as an estimator: at 128³ it carries a 6%
   one-signed residual the coefficient does not control
   ([Taylor–Green](#taylorgreen)). `bench/tgv_energy.jl` reports the relative
   L2 misfit of the kinetic-energy and −dKE/dt histories against the reference;
   rank `C_mu` on that, at a resolution whose own history error is below the
   effect size, and with the filter settled first. `C_mu = 0.004` is withdrawn
   as a candidate.
4. **Decide the detector.** `:d8` improves six of seven battery columns and the
   `C_beta` refit under it [retains 1.0](#the-c_beta-refit-under-d8), with no
   value in 0.25–4 recovering the spherical origin. The decision waits only on
   item 2.
5. **Refit `C_mu` under `:gaussian`.** The smoother changed in August 2026 without
   a refit. `C_beta` has been swept under it and under `:d8`, retaining 1.0 in
   both; `C_D` was not swept because the filter dominates interface broadening.
   `mu_sensor = :velocity` is in the same class and cannot be evaluated until
   `C_mu` is refitted under it, since it moves the μ\* share of the Taylor–Green
   sink by a third with the constant held fixed.
6. **Make the κ\* construction non-singular as T_ion → 0.** It is an EOS dispatch
   point, so a tabular model can supply its own; the gas models still divide by
   the temperature.
7. **Determine why the spherical-origin fold is less tolerant of under-resolved
   data than the cylindrical axis fold.** The closure explanation is
   [retired](#fold-order-and-geometry-limits) and no replacement exists.
8. **Make `filter_state!` conservative on non-Cartesian metrics**, by the
   filtered-cell-volume normalization the reference uses. Uniform Cartesian
   results are unaffected by construction, so this is measurable against the
   converging cases alone.
9. **Decide the filter wall rows.** `compact_filter(closures = :onesided)` takes
   the planar Noh wall deficit from 64% to 27% and is the wall-order cap of every
   filtered run, but every constant here was calibrated under the cascade and the
   `test/validation.jl` guards are set from it. Switching is a recalibration of
   the wall cases. The derivative and filter closures are coupled: `:cascade4`
   with the cascade filter, or `:cascade3` or C6 `:brady_livescu` with the
   one-sided filter.
10. **Put `delta4_sum!`'s even path on the half-offset mirror.** The clamp it uses
    is second-order-consistent for an even field, so this is expected to be small,
    but it is a spurious sensor at the fold cell where every converging case fails
    and the size of it has not been measured. The odd path already uses the
    mirror.
11. **Decide `species_flux`.** The bulk channel is at least as good as the Fickian
    one on every measured column and removes its pressure error at unequal gas
    constants, but the four constants are calibrated on the Fickian channel and
    the vortex-ring/SF6 case that should decide it has not been run.
