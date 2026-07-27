# Calibrating the artificial fluid properties

What the four `ArtParams` constants actually do, measured rather than inherited,
and where each one fails. Reproduce with `bench/artcal.jl`; the cases are the
ones `test/validation.jl` guards, from `test/cases.jl`.

The headline: **the shipped defaults are sound, but the constant that decides
whether a strong-shock run works at all is the CFL number, not any of them.**

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
8. [Geometry limits found on the way](#geometry-limits-found-on-the-way)
9. [Recommendations](#recommendations)

## How to read the tables

Every number in the four constant tables is one-dimensional. That bounds what
they can say: `C_mu` multiplies the *shear* part of the artificial stress, which
a 1-D case cannot exercise except through its trace, so the `C_mu` table
constrains stability and essentially nothing else. `C_beta`, `C_kappa` and `C_D`
are all fully exercised.

The exception is the 3-D Taylor–Green work inside the [`C_mu`
section](#c_mu--the-shear-viscosity), which exists precisely because the 1-D
tables cannot reach that constant. Read those subsections as a separate study
with its own estimator caveats, not as more rows of the table above them.

Columns:

- **Noh plat/exact** — post-shock density plateau over its exact value
  (4, 16, 64 for ν = 1, 2, 3). 1.0000 is right; this is the sharpest accuracy
  number in the set because the answer is a fixed integer multiple of ρ₀.
- **deficit** — density shortfall at the symmetry point relative to the exact
  plateau. This is *wall heating*: spurious entropy deposited where the flow
  stagnates, showing up as too-hot, too-rarefied gas in the first cell.
- **Lax L1 / contact** — mean density error against the exact Riemann solution,
  and the 10–90% width of the contact discontinuity. The contact width is the
  price of regularization.
- **Shu train amp** — peak-to-trough density in the post-shock entropy wave
  train. This is the quantity the tenth-order interior scheme exists to
  preserve, and it pulls against every constant that adds dissipation.
- **WC peak** — peak density in the Woodward–Colella collision, mostly a
  survival check at a 10⁵ pressure ratio.
- **NaN** — the run lost positivity or stalled. Neither is a crash: the diffusive
  rate in `compute_dt` climbs until `dt` collapses and the run grinds. See the
  `NMAX` note in `test/cases.jl`. As of the `StepControl` work this now raises
  a `SolverFailure` in `run!` instead, with the positivity loss caught roughly
  150 steps before the `dt` collapse it used to be diagnosed by.

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

**Both ends fail, for different reasons, and this is the whole calibration.**

At `C_beta = 0` there is no shock regularization at all and every strong-shock
case loses positivity — Noh in both geometries and Woodward–Colella. The Lax
tube survives because it is weak enough for the compact filter alone to hold,
and its contact is the sharpest in the table (0.0032) precisely because nothing
is smearing it. That is the shape of the trade-off in one row: **the sharpest
answer is the one that does not run.**

At the top end the failure is *not* an accuracy failure. Doubling `C_beta` to 2
gives a monotonically better Woodward peak and only slightly worse everything
else — but Noh ν = 1 stops completing. The upper bound is a stability bound
rather than an accuracy one, and it moves with the CFL, which is what the
[CFL table](#cfl-which-dominates-all-four) shows. What it is *not* is a
timestep-lag effect — that hypothesis was tested and rejected; see the mechanism
discussion in that section.

Between the two ends the accuracy trend is clean and monotone:

- Contact width grows roughly linearly, 0.0041 → 0.0074 over 0.25 → 4.
- The Shu–Osher wave train loses about 3.5% of its amplitude over the same
  range. The entropy waves are smooth, so this is pure over-damping, and it is
  the cost that matters for a mixing calculation.
- The Noh ν = 3 plateau *degrades* with increasing `C_beta` (1.0122 → 0.9516),
  crossing exact between 0.25 and 0.5.

So on accuracy alone the optimum is near `C_beta = 0.4`. The shipped 1.0 buys
robustness margin instead: it is the value at which Woodward–Colella and both
Noh geometries all complete with room to spare, and the accuracy it gives up
against 0.5 is a 13% wider contact and 0.8% of wave-train amplitude.

**Recommendation: keep 1.0 as the default. Drop to 0.5 for interface-dominated
work (Richtmyer–Meshkov, Rayleigh–Taylor), where contact sharpness is the
answer and the shocks are moderate. Do not go below 0.25 or above 2.**

## C_kappa — the conductivity

```
C_kappa   | Noh1 plat  deficit | Noh3 plat | Lax L1  contact | WC peak
0         |  0.9998       +64% |      NaN  | 4.8e-3   0.0051 | 6.6847
0.0025    |  0.9997       +61% |   0.9728  | 4.9e-3   0.0051 | 6.6390
0.01   *  |  0.9992       +58% |   0.9732  | 5.0e-3   0.0051 | 6.5731
0.04      |  0.9980       +57% |   0.9748  | 5.3e-3   0.0052 | 6.4934
0.16      |  0.9986       +70% |   0.9796  | 5.8e-3   0.0052 | 6.3176
```

κ\* is doing two jobs, and the table separates them.

**Wall heating.** The deficit falls from 64% to 57% as `C_kappa` goes from 0 to
0.04, then rises sharply to 70% at 0.16. That is the expected behaviour: the
artificial conduction is what carries the spuriously deposited entropy out of
the stagnation cell, so some is better than none, but past a point the
conduction itself becomes the error. The minimum is broad and sits near 0.04.

**Robustness.** `C_kappa = 0` will not run spherical Noh at all. Given that
`C_beta` alone regularizes the momentum equation, this says the energy equation
needs its own regularization at a converging strong shock — not a surprise, but
worth knowing before switching κ\* off to save the sensor pass.

The cost is ordinary: Lax L1 grows 4.8e-3 → 5.8e-3 across the range, and the
contact width barely moves (0.0051 → 0.0052) because κ\* diffuses temperature,
not composition, and the Lax contact is nearly isothermal.

**Recommendation: 0.01 is defensible and conservative; 0.02–0.04 measurably
reduces wall heating at ~5% more L1 error on a shock tube. Never 0.**

There is a caveat that matters more than any of these numbers. κ\* is built as
`C_kappa · (ρc/T_ion) · sensor`, which is singular as T_ion → 0. In a genuinely
cold ambient (p₀ ≲ 10⁻³ at ρ₀ = 1) the 1/T factor drives the diffusive rate up,
`dt` collapses, and internal energy can be driven negative — after which T_ion
clamps and κ\* becomes astronomical. This is why every case here carries a
finite ambient pressure. The scale is now an EOS dispatch point
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

What the table does say is that **`C_mu` above about 0.008 destabilizes the
spherical origin**, through the same lagged-diffusive-timestep mechanism as
`C_beta`. So this is a stability ceiling with no accuracy floor visible here.

The value of `C_mu` for its actual purpose — damping under-resolved shear in a
turbulent mixing layer — is not constrained by any case in this battery, and
should not be claimed to be. That needs a 3-D case with resolved shear.

### Why the Taylor–Green route is harder than it looks

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
over-dissipating and too early, and the excess is mostly filter. Two things
follow, and the first is the one that matters:

- **μ\* is no longer inert here** — it reaches 28% of the *viscous* budget, so
  TGV does fix the 1-D problem this section opens with. But `C_mu` cannot be
  *fitted* against the dissipation curve while the filter owns 83% of the total
  sink; that would be fitting a coefficient to a residual. And the filter cannot
  be removed to isolate it: at 32³, `filter_interval = 4` diverges and
  `filter_interval = 0` fails outright with `SolverFailure(:negative_density)` at
  t = 5.32. **The compact filter is the primary stabilizer at this resolution,
  not an auxiliary smoother.** `C_mu` and the filter set the subgrid dissipation
  jointly and have to be calibrated as a pair. (Read the next section before
  acting on this one: at 128³ the filter share collapses to 37% and `C_mu`
  *does* become fittable — by holding the filter fixed rather than removing it.
  The "cannot be fitted" here is a statement about 32³, not about the case.)
- β\* is four orders below μ\* despite `C_beta = 1.0`, because at Ma 0.1 there is
  almost no dilatation for it to act on. Consistent with the `C_beta` section:
  that constant is a shock constant and TGV has no shocks.

### At resolution: 128³ on a cluster

The caveat that used to close this section — that filter dominance was
established at 32³ and unestablished anywhere worth quoting — has been settled.
128³, Re = 1600, t = 10, 224 ranks over two rzhound nodes, ~20–25 min per
configuration:

```
config              peak -dKE/dt        mol    mu*   filter   steps   wall
art ON,  filter 1   1.2044e-2 @ t≈9.06  60.4%  2.3%   37.3%   12739   1520 s
art OFF, filter 1   1.2153e-2 @ t≈9.00  62.5%  0      37.5%   10830   1063 s
art ON,  filter 0   SolverFailure(:negative_density) at t = 4.66, step 8515
```

Peaks are windowed rates — see [Read the rate over a
window](#read-the-rate-over-a-window-not-instantaneously), which is not optional
for anything with `art` on. Channel shares are from the sampled table.

**Filter dominance does not survive resolution.** 87% (16³) → 83% (32³) → 37%
(128³). Two coarse points were not a trend, and the third breaks it. The
molecular term takes over correspondingly, 12% → 60%.

**The solver is close to right here, which is a result in its own right.** Peak
−dKE/dt of 1.2065e-2 at t ≈ 8.8 against the reference 1.2e-2 at t = 9 is under
1% on magnitude and within a sample stride on timing. At 32³ the same case
over-dissipated by 22% and peaked 2.5 time units early.

**But the filter is still required, and its stabilizing role is decoupled from
its share of the energy budget.** At 128³ it supplies 37% of the sink, and
removing it kills the run *earlier* than at 32³ — t = 4.66 against 5.32. The
failure is a clean energy blow-up, not the dispersive-undershoot signature of
the shock cases: KE tracks the filtered run to within 0.1% until t ≈ 4.4, turns
upward, and triples before positivity goes, with `dt` collapsing to 2e-60. TGV is
unforced, so rising KE is unambiguously numerical. The filter owns essentially
100% of the grid-scale sink whatever its share of the total. Meanwhile
art-off/filter-on runs cleanly to t = 10. **The filter is necessary and
sufficient at this resolution; the Cook properties are neither.** That inverts
the intended division of labour and it does not soften with refinement, so
expect the same at 256³.

**Refinement does not disentangle μ\* from the filter.** Their ratio is
essentially invariant across the 4× refinement:

```
32³ :  mu* 5.0%  / filter 83.0%  = 0.060
128³:  mu* 2.3%  / filter 37.3%  = 0.062
```

Both shrink together. So the obvious plan — refine until the filter releases its
grip, then fit `C_mu` against what is left — does not work: at the resolution
where the filter lets go, μ\* has gone inert for the same reason.

**What does work is holding the filter fixed rather than removing it**, and
reading the rate over a window rather than instantaneously (see the next
subsection — this matters more than it sounds):

```
C_mu = 0       (art off)   1.2153e-2 @ t ≈ 9.00    +1.28% vs reference
C_mu = 0.0005              1.2108e-2 @ t ≈ 9.05    +0.90%
C_mu = 0.002   (default)   1.2044e-2 @ t ≈ 9.06    +0.37%
C_mu = 0.008               1.1950e-2 @ t ≈ 8.77    −0.42%
```

**Monotone across a 16× span, and it crosses the reference.** μ\* moves the peak
in the right direction and costs 43% wall time to do it (21% per step from
`compute_artificial!`, plus 18% more steps). Channel shares at the peak scale as
they should: μ\* is 2.3% of the sink at `C_mu = 0.002` and 8.2% at 0.008, against
a filter share that barely moves (37.2% → 33.1%).

Interpolating the last interval gives 0.0048 linearly and 0.0038 in log — call
the optimum **`C_mu ≈ 0.004`**.

**The uncertainty band includes the shipped default, so this is not a reason to
change it.** The slope near the crossing is −1.57e-2 per unit `C_mu`, and the
peak estimate carries ~0.4% uncertainty — at `C_mu = 0.002` the top two windowed
values are 0.011966 and 0.012044, 0.65% apart, so where the window falls relative
to the true maximum matters. That is ±0.002–0.003 in `C_mu`: the band runs
roughly 0.002–0.007 and 0.002 sits inside it.

**And the top of the band is already excluded by another case.** The 1-D table
above has Noh ν = 3 as NaN at `C_mu = 0.008` — the spherical origin does not
complete there. So the value that best-fits TGV from above breaks a shipped case,
and 0.004 is untested between the 0.002 and 0.008 rows. Anyone moving the default
must run `bench/artcal.jl` at 0.004 first and confirm the origin survives; that
is a desktop job.

To narrow the band, no more compute is needed — only a better peak estimator.
`kes` is recorded every step, so fitting a parabola to the windowed hump and
reporting its vertex would beat taking the argmax, which is what the ±0.4% is.

### Read the rate over a window, not instantaneously

This was very nearly a wrong answer, so it is worth stating as a rule. The filter
removes energy per *application*, so the instantaneous rate is
`(filter loss)/dt + physical` and carries the full step-to-step `dt` jitter
divided into it. With `art` on, the sensor feeds `compute_dt` and `dt` swings
±12% step to step at 128³; against a filter supplying ~37% of the sink that
predicts ∓4.4% on the total, and ∓4% is exactly the scatter the one-step column
showed. With `art` off, `dt` is smooth to under a percent and so is the curve —
which is why the artifact hid until the artificial properties were varied.

`C_mu` is being ranked on differences well under 1%. The one-step rate cannot
resolve that; the 501-step windowed rate can, and reduces the within-run scatter
to ~0.3%. `bench/tgv_energy.jl` now reports every rate windowed (`window=`), and
the numbers above are windowed. **Numbers quoted from output before that change
are contaminated for any `art`-on configuration.**

A boxcar over a curved peak reads ~0.2% low at this window width. That is
common-mode across configurations, so it cancels in the `C_mu` comparison and
only biases the absolute value against the external reference.

Two caveats to carry. Fitting one scalar with one knob always succeeds, so an
optimum near 0.002 would show the default is *consistent* with TGV, not derived
from it; the stronger test is the whole −dKE/dt(t) curve, which these runs
already produce at ~37 points but which needs the van Rees reference curve
digitized to compare against. And `filter_interval` and the filter's α are
themselves uncalibrated, so any `C_mu` fitted this way is conditional on
`compact_filter(0.45)` applied every step.

**Reading the peak.** `run!` truncates the final step to land exactly on
`tfinal`, and the compact filter removes energy per *application* rather than per
unit time — so a short step still pays a full filter pass and −dKE/dt inflates.
At 128³ this produced a spurious 1.4226e-2 at t = 10.00 against the true
1.2065e-2 at t = 8.77, on both filtered configurations. `bench/tgv_energy.jl`
now excludes that step; output from before that fix reports the peak ~18% high.

**Recommendation: keep 0.002, and it is no longer unvalidated.** TGV at 128³ with
resolved shear brackets the reference across a 16× sweep and puts the optimum at
`C_mu ≈ 0.004 ± 0.003` — a band that contains the shipped value. That closes the
question this section opened with ("`C_mu` should not be claimed to be
constrained by any case in this battery; that needs a 3-D case with resolved
shear"). It is now constrained, to within a factor of two, and 0.002 is
consistent with it.

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

The interesting result is the first row. **With D\* switched off entirely the
interface still more than doubles in width**, so the dominant smearing agent on
a passive interface in this code is the compact filter, not the artificial
diffusivity. Across a 64× sweep in `C_D` the width moves by 22%.

Two consequences. First, `C_D` is a weak knob and there is no reason to tune it
finely. Second, if interface sharpness is the goal, `filter_interval` and the
filter's α parameter are the levers with real authority — `compact_filter(0.45)`
is the default and a higher α (weaker filter) is the first thing to try.

This also sharpens the note in `CLAUDE.md` about the per-species sensor being a
measurable no-op at `n_species == 2`: not only do `D*_1` and `D*_2` agree to
4.8e-16 there, the whole term is a minor contributor to what it controls.

**Recommendation: keep 0.01. Raise it only for genuine multi-species (≥ 3)
problems where the correction velocity has something to do.**

## CFL, which dominates all four

```
cfl       | Noh1 plat  deficit | Noh2 plat | Noh3 plat | WC peak
0.4       |     NaN       NaN  |      NaN  |      NaN  | 6.5762
0.3       |     NaN       NaN  |      NaN  |      NaN  | 6.5731
0.2       |  0.9993       +55% |      NaN  |      NaN  | 6.5597
0.15      |  0.9992       +58% |   0.9355  |   0.9732  | 6.5407
0.1       |  0.9995       +58% |   0.9345  |   0.9722  | 6.5138
```

This is the table to read first. **No setting of the four constants rescues a
converging strong shock at the default `cfl = 0.5`, and every one of them works
at 0.15.** The accuracy at 0.15 and 0.1 is identical to three digits, so the
cost of the smaller step is wall-clock and nothing else.

### The mechanism is not what it looks like

An earlier version of this document asserted that the cause was the lag in
`compute_dt`, which builds its diffusive rate from the previous step's
artificial coefficients: at a forming shock β\* grows by an order of magnitude
in a few steps, so the step is chosen from coefficients that are already stale.
That is a real property of the code, and it is not the cause.

It was tested by building the obvious fix — linear extrapolation of the rate,
`StepControl(predict = n)` — and sweeping the lookahead. Noh at ν = 1, N = 400,
cfl = 0.3 fails at **step 175** with no lookahead, **179** with three steps of
it, and **200** with thirty. Capping the growth of `dt` (`max_growth = 1.05`)
moves it to 186. None of that is a fix; it is noise on a failure that happens
regardless.

The trace shows what is actually going on:

```
step   25  dt=2.1e-4  rate=1418   rho_min=0.951
step   75  dt=1.8e-4  rate=1633   rho_min=0.798
step  125  dt=1.8e-4  rate=1718   rho_min=0.398
step  175  dt=4.8e-5  rate=1.5e4  rho_min=0.252   <- and then negative
```

The pre-shock density is exactly 1 in the exact solution. It is 5% low by step
25 and 60% low by step 125, while `dt` and the rate sit flat. **A dispersive
undershoot at the shock is eating the state for 150 steps before anything in
the timestep notices.** Positivity is lost around step 175, and only then does
the diffusive rate explode and `dt` collapse. The collapse is the symptom; the
disease is that the artificial viscosity is not damping the undershoot at that
CFL. That is a spatial-regularization problem, not a temporal one, and it is
why raising `C_beta` does not help either — the C_beta table above shows the
upper bound is itself a stability bound.

### What to do instead

Rolling back is strictly better than guessing a CFL, because **the restriction
is a startup restriction**: once the shock has formed cleanly the large step is
fine. `StepControl(retries = 4)` recovers Noh from a deliberately terrible
`cfl = 0.9`:

```
nu   start cfl   recovered cfl   steps   plateau/exact
1    0.9         0.45            1433    0.9989
2    0.9         0.45             915    0.9369
3    0.9         0.1125          1636    0.9729
```

Against 4485 steps at a globally fixed `cfl = 0.15` for the same ν = 1 answer.
Three times faster, no number to guess, and the CFL it settled on is reported
back in `solver.cfl` for the next run.

The limit of the mechanism is the flip side of the diagnosis above: rollback
recovers an *abrupt* failure, and the gradual degradation at cfl = 0.3 is not
recoverable, because by the time the check fires the last savepoint is already
damaged. There the answer is still a lower CFL.

**Recommendation: `cfl = 0.5` for smooth and moderately compressible flow;
`cfl = 0.3` with shocks; `StepControl(retries = 4)` always, so a bad guess costs
a warning instead of a wasted run. `cfl = 0.15` for converging geometry if you
would rather not rely on the rollback.**

## Grid convergence

```
N         | Noh1 plat  deficit | Lax L1  | mix width
128       |  0.9945       +60% | 1.3e-2  |  0.03581
256       |  0.9994       +59% | 7.1e-3  |  0.01820
512       |  0.9991       +58% | 3.9e-3  |  0.00931
1024      |  0.9992       +56% | 1.9e-3  |  0.00481
```

Three separate behaviours, all as they should be:

- **Lax L1 halves per doubling** — first order, which is what any scheme gives
  in L1 on a captured discontinuity regardless of interior order. The sixth- and
  tenth-order convergence lives in `test/convergence.jl`, on smooth fields.
- **Interface width halves per doubling** — the regularization follows the mesh
  rather than sitting at a fixed physical scale, which is the design intent of
  the whole Cook artificial-property approach and the reason it is usable as a
  subgrid model.
- **Wall heating does not converge away.** 60% → 56% over an 8× refinement. This
  is the known character of the Noh problem: the entropy error is deposited once,
  in the first cell, at shock formation, and stays there. What *does* converge is
  its extent — one cell of a shrinking mesh — so the integrated error vanishes
  while the pointwise error does not. A code that reported wall heating going to
  zero pointwise would be the suspicious one.

## Geometry limits found on the way

Two hard limits turned up building the battery. Both are documented here rather
than worked around in the solver.

**The spherical origin needs its initial data resolved over ≳3 cells.** A blast
initialized as a top hat with a 1–2 cell transition loses positivity within
tens of steps; at 3 cells and wider it runs to completion. The cylindrical axis
takes a 1-cell transition without complaint, and the same top hat runs fine in
Cartesian, so this is specific to the origin fold and its antipodal pairing —
where a discontinuity is being differentiated through a fold whose closure is
third order. `test/cases.jl` therefore initializes Sedov with a Gaussian deposit.

**The spherical origin will not take the singular t = 0 start of Noh.** Every
CFL and every constant setting fails; the exact solution requires 64× compression
to appear at r = 0 instantaneously. Warm-starting from the exact solution at
t = 0.3 and integrating to 0.6 runs cleanly and is arguably the better test of
the fold anyway, since it measures whether the solver can *maintain* the
solution through the origin rather than conflating that with an initialization
singularity. The cylindrical axis handles the cold start at 16× compression.

## Recommendations

| Constant | Default | Keep? | Notes |
|---|---|---|---|
| `C_mu` | 0.002 | yes | TGV at 128³ puts the optimum at 0.004 ± 0.003, which contains 0.002. Never above 0.008. |
| `C_beta` | 1.0 | yes | Accuracy optimum near 0.4; use 0.5 for interface-dominated work. Never 0. |
| `C_kappa` | 0.01 | yes | 0.02–0.04 measurably reduces wall heating. Never 0. |
| `C_D` | 0.01 | yes | Weak knob; the compact filter dominates interface thickness. |
| `cfl` | 0.5 | **no** | Use 0.3 with shocks. Prefer `StepControl(retries = 4)` to guessing. |

Open items this study leaves behind, in rough priority order:

1. Damp the dispersive undershoot that precedes the positivity loss. This is
   the real CFL restriction and it is a spatial-regularization question — the
   sensor is not switching on early enough, or β\* is not reaching far enough
   ahead of the front. Neither a timestep predictor nor a larger `C_beta`
   touches it; both were measured.
2. Make the κ\* construction non-singular as T_ion → 0. It is an EOS dispatch
   point as of the Phase 1 work, so a tabular model can already supply its own;
   the gas models still divide by the temperature.
3. Narrow the `C_mu` band below ±0.003. Needs no more compute — a parabola fit
   to the windowed hump instead of an argmax, plus denser sampling through the
   peak. The harder follow-up is digitizing the van Rees −dKE/dt curve so the
   fit is against the whole history rather than one scalar.
4. Calibrate the compact filter itself. It is necessary and sufficient for
   stability at 128³ while `filter_interval` and α have never been fitted to
   anything, which makes every `C_mu` number conditional on
   `compact_filter(0.45)` every step.
4. Understand why the spherical origin fold is so much less forgiving of
   under-resolved data than the cylindrical axis fold.
