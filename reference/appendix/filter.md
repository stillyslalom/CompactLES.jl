# Calibration appendix: Taylor-Green and the filter

This file holds the sections of the measurement record on the Taylor-Green energy
history and the compact filter's dissipation, on uniform and non-uniform volumes.
[CALIBRATION_APPENDIX.md](../CALIBRATION_APPENDIX.md) indexes the record and states its
conventions.

## Taylor-Green

```text
julia --project=. bench/tgv_energy.jl 128 10 configs=on:1 smoother=gaussian cfl=0.35 alphaf=0.45
julia --project=. bench/archive/tgv_spectrum.jl <checkpoint>
```

`bench/tgv_energy.jl` runs Taylor–Green at Re = 1600 and splits −dKE/dt by mechanism. The
reference is the tabulated 512³ pseudo-spectral solution vendored at
`data/spectral_Re1600_512.gdiag`
([provenance](../../data/README.md#taylor-green-reference-solution)). Its peak is 1.28575e-2
at t = 8.97, and every comparison passes that solution through the run's own boxcar.

**Read the rate over a window.** The instantaneous rate scatters by several times the
differences on which `C_mu` is ranked, and a window wide enough to remove the scatter reads
a curved peak low by an amount that grows with its width in time. The instantaneous rate is
`(filter loss)/dt + physical`, so it carries the full `dt` jitter. At 128³ with the
artificial properties on, `dt` swings ±12% step to step. Against the filter's 37% share of
the sink this predicts a one-step scatter of ∓4%, where `C_mu` is ranked on differences well
under a percent. The 501-step windowed rate holds within-run scatter near 0.3%. The bias of
the boxcar on the peak:

```
window                 width              reads low by
501 steps at 128³      its half-width     0.10%
                       0.5                3.0%
501 steps at 32³       0.8 time units     5.2%
```

The truncated final step, which pays a full filter pass at short `dt`, is excluded.

### The measured budget

The script measures five channels: molecular dissipation, μ\*, β\*, the compact filter and
the reversible pressure work ∫p∇·u, with the remainder printed as `unattr`. A `filter`
column in a table without `unattr` is instead the residual −dKE/dt − (mol + μ\* + β\*),
which also holds the pressure work, the aliasing, dispersion and time-integration error;
the two definitions are not comparable.

A callback sees an already filtered state, on which a second pass removes, at each
wavenumber, the first pass's loss scaled by the square of the transfer function.
`filter_loss` therefore advances a copy of the state by one step and measures what a
`filter_state!` pass removes from it. At 32³ to t = 10 under `:compact`, at the peak:

```
             mol     mu*    beta*   filter   p*divu   unattr
measured    12.1%    4.7%    0.0%    84.1%     0.4%    -1.3%
residual    12.1%    4.7%    0.0%    83.1%       -        -
```

The residual is one point low here, an adequate proxy at 32³. If the absorbed error held
near one point of −dKE/dt at every resolution, the correction would grow as the filter's
share falls, to about 8% of the channel at 256³; that is unmeasured. The negative `unattr`
means the channels slightly over-account for the loss; the trial step and the windowed
estimate are both candidates and neither has been isolated. With the filter off at 16³ the
budget closes, which fixes the sign convention: pressure work exchanges energy between the
kinetic and internal forms rather than removing it.

```
16³, filter off    mol 100.8%    p*divu −0.8%    unattr 0.0%
```

A diagnostic here must restore the artificial coefficient arrays it writes, since
`max_rate` sizes the next step from that storage; the budget pass brackets itself with
`art_block` and `set_art_block!`, and 32³ to t = 10 agrees on the peak to five digits either
way.

### Channel shares under refinement

Shares at the dissipation peak, artificial properties on, `filter_interval = 1`, residual
definition. Every row but 128³ uses the Gaussian smoother; 128³ ran under `:compact`:

```
resolution   peak -dKE/dt     vs reference   windowed ref.   molecular   mu*    beta*   filter   mu*/filter
32³          1.4216e-2 @ 6.58   +16.4%       1.2210e-2        12.6%      5.1%    ~0%    82.2%    0.060
64³          1.2459e-2 @ 8.49     —          —                33.8%      4.5%    0.0%   61.6%
128³         1.2044e-2 @ 9.06    −6.0%       1.2809e-2        60.4%      2.3%    0.0%   37.3%    0.062
256³         1.3043e-2 @ 8.84    +1.6%       1.2844e-2        86.4%      0.8%    0.0%   12.8%    0.0625
```

**Refinement does not separate μ\* from the filter**: their ratio is the same to within a few
percent at every resolution (the last column). The windowed reference rises toward the
tabulated peak as the window narrows with the step count. The 64³ row's step count was not
recorded, so its window cannot be reconstructed. Against the raw tabulated peak it is 3.1%
low, and since the window only lowers the reference, that is the most negative value the row
can take. β\* is negligible at every resolution, dilatation being negligible at Ma 0.1. The
trend is not monotone, and the rows differ in smoother, backend and machine, so resolution
and configuration cannot be separated:

```
run     smoother    machine
128³    :compact    224 rzhound ranks over two nodes, 20–25 minutes each
256³    :gaussian   4 MI300A APUs at -t 1, 24,490 steps in 3.69 h
```

### C_mu, the shear viscosity

In one dimension μ\* reaches the solution only through the trace of the stress, where β\*
dominates by a factor of 500. The battery sets one bound and no accuracy lower bound:

```
C_mu      | Noh1 plat  deficit | Noh3 plat | Lax L1  contact | Shu train
0         |  0.9992       +58% |   0.9731  | 5.0e-3   0.0051 |    1.6183
0.0005    |  0.9992       +58% |   0.9732  | 5.0e-3   0.0051 |    1.6182
0.002  *  |  0.9992       +58% |   0.9732  | 5.0e-3   0.0051 |    1.6185
0.008     |  0.9992       +58% |      NaN  | 5.0e-3   0.0052 |    1.6214
0.032     |  0.9992       +58% |      NaN  | 5.0e-3   0.0052 |    1.6206
```

`C_mu` above about 0.008 destabilizes the spherical origin. The three-dimensional sweep
holds the filter fixed at 128³ under `:compact`:

```
C_mu = 0       (art off)   1.2153e-2 @ t ≈ 9.00    −5.0% vs reference
C_mu = 0.0005              1.2108e-2 @ t ≈ 9.05    −5.5%
C_mu = 0.002   (default)   1.2044e-2 @ t ≈ 9.06    −6.0%
C_mu = 0.008               1.1950e-2 @ t ≈ 8.77    −6.7%
```

The same point under the Gaussian gives nearly the same shares and a peak 0.63% away, so the
conclusions are unaffected:

```
C_mu = 0.002 at 128³   molecular   mu*    filter
:compact               60.4%       2.3%   37.3%
:gaussian              61.1%       2.4%   36.4%
```

**The 128³ peak does not determine `C_mu`.** The sweep has no crossing: every value
underpredicts the reference peak and raising the constant moves further away. Across the
sixteenfold range the peak moves 1.7%, against a residual three and a half times larger and
of one sign. The 256³ row overshoots on the same estimator, so that residual belongs to the
resolution and configuration; the histories below decide what the peak cannot. At 128³ μ\*
costs 43% of wall time. It costs 21% per step in `compute_artificial!` and takes 18% more
steps.

**Removing μ\* improves the fit at 64³.** Three controls at 64³ with the Gaussian smoother,
`cfl = 0.35` and `filter_interval = 1`:

```
alpha = 0.499        peak vs ref   peak time   KE misfit   -dKE/dt misfit   mu* share
full (C_mu 0.002)      +6.97%        -0.50      2.532e-2       1.366e-1        17.6%
C_mu = 0               +6.39%        -0.50      1.412e-2       9.936e-2
art off                +5.32%        -0.42      1.176e-2       8.241e-2

alpha = 0.45
full (C_mu 0.002)      -1.57%        -0.45      4.137e-2       1.777e-1         4.7%
C_mu = 0               -1.32%        +0.09      3.831e-2       1.646e-1
art off                -0.40%        -0.02      3.446e-2       1.593e-1
```

The histories are worse with μ\* in both arms, whether it carries a large or a small share
of the sink, so the best-fitting `C_mu` here is zero. The budget does not close exactly, and
in the near-off filter's arm its deficit grows as the damping is removed:

```
budget closure at 64³
across the α = 0.45 arm                    −3.0% to +1.1%
α = 0.499: full, C_mu = 0, art off         −3.8%, −6.1%, −12.8%
```

The cause is either grid-scale energy from a central compact scheme without dealiasing or a
windowed −dKE/dt read against instantaneous channels on the curved flank of the peak; the
two are not separated. A proportional bias in the filter probe is excluded, the deficit being
largest where the filter's share is smallest.

**Removing μ\* improves the fit at 128³.** The same controls under the production defaults,
with the shares at the peak:

```
128³, Gaussian smoother, cfl = 0.5, filter_cfl = 0.35, filter_interval = 1;
one RX 6800 XT in Float64 through backend=amdgpu, 32 to 51 minutes a run
alpha = 0.45        steps  peak vs window  peak time  KE misfit  -dKE/dt misfit  mu*   filter
full (C_mu 0.002)    9009      -3.91%        -0.04    6.711e-3     4.686e-2     2.7%   36.6%
C_mu = 0             9036      -3.49%        -0.03    5.551e-3     4.128e-2     0      38.5%
art off              7848      -3.32%        -0.03    4.429e-3     3.652e-2     0      36.3%

alpha = 0.499
full (C_mu 0.002)   10847      +3.17%        -0.07    4.954e-3     2.876e-2     9.3%   15.3%
C_mu = 0            11116      +3.79%        -0.04    1.964e-3     2.052e-2     0      20.0%
```

Setting `C_mu = 0` lowers both misfits in both arms by more than their resolving power of
0.5% and 4% ([cadence and α](#cadence-and-alpha-are-one-axis)), while the weaker filter's
peak moves further above the windowed reference:

```
reduction at C_mu = 0    KE misfit   -dKE/dt misfit
alpha = 0.45             17%         12%
alpha = 0.499            60%         29%
```

The best-fitting `C_mu` at 128³ is zero under both filters. The case is not resolved there:
`k_max η` is 0.75 and a fifth of the physical dissipation lies off the grid
([spectra](#spectra)), so a subgrid viscosity has missing dissipation to supply, and removing
μ\* raises the filter's share of the sink in the weaker arm. No positive `C_mu` fits
Taylor–Green at Re = 1600 at either resolution, and the upper bound still comes from the
one-dimensional battery. The lowest misfits of the five are those of the weakest filter at
`C_mu = 0`, so the α = 0.49 fit, made at `C_mu = 0.002`, is conditional on that value.

**Where 0.002 comes from.** Cook never published a calibration of it. The value is the
recommendation of [Cook and Cabot (2005)](https://www.osti.gov/servlets/purl/15011510) for
the r = 4 form. Their own runs use the r = 2 form at other constants:

```
Cook and Cabot (2005)   form    C_μ      C_β
run                     r = 2   0.025    5
recommended             r = 4   0.002    1    "recommended values for the empirical
                                              coefficients with r = 4"
```

In [Cook (2007)](https://www.osti.gov/servlets/purl/902334) the five r = 4 constants "have
been found to work well for a wide variety of test problems". Neither paper fits `C_mu` on
Taylor–Green. Both run it at 64³ and compare enstrophy. The 2005 note compares against the
semi-analytic solution to t ≈ 3.5 and finds the model inert until vorticity reaches the grid
scale. The later paper compares against two implicit-LES schemes as a test of resolving
power. The evidence that μ\* supplies subgrid transfer is the 2005 note's second case, the
Kang, Chester and Meneveau active-grid experiment at Re_λ ≈ 720. On a 192³ grid the model
gives the Kolmogorov spectrum and the measured decay, and its absence corrupts the spectral
flux. That case is far more under-resolved than Taylor–Green at Re = 1600 and 128³.

The filter differs as well. The 2005 runs dealias by the two-thirds rule and filter nothing,
so μ\* is the only grid-scale sink. The later method adds a pentadiagonal eighth-order
filter, α = 0.66624, β = 0.16688, after every RK4 substep, designed to remove the top tenth
of the wavenumbers as sharply as possible; it is `pyranda_filter()`. Its transfer function
against the two arms above:

```
transfer function        0.8 k_max   0.9 k_max
pyranda_filter()         0.966       0.579
compact_filter(0.45)     0.754       0.371
α = 0.499                0.993       0.964
```

Under Cook's filter μ\* carries the transfer below about 0.85 k_max. Under the production
filter the filter carries 37% of the sink at 128³ and does that job itself, consistent with
the zero fit above. The weaker arm is the nearest to Cook's setting below 0.85 k_max but
leaves the band above it almost unfiltered, so neither arm reproduces the configuration 0.002
was chosen in.

### The filter is the stabilizer

At 32³, `filter_interval = 4` diverges and `filter_interval = 0` fails with
`SolverFailure(:negative_density)` at t = 5.32. At 128³:

```
config              peak -dKE/dt        mol    mu*   filter   steps   wall
art ON,  filter 1   1.2044e-2 @ t≈9.06  60.4%  2.3%   37.3%   12739   1520 s
art OFF, filter 1   1.2153e-2 @ t≈9.00  62.5%  0      37.5%   10830   1063 s
art ON,  filter 0   SolverFailure(:negative_density) at t = 4.66, step 8515
```

Without the filter the 128³ run fails earlier than the 32³ one, although the filter's share
of the sink is smaller there. With the artificial properties disabled and the filter on, the
run reaches t = 10. The 128³ failure is a clean energy blow-up, not the dispersive-undershoot
signature of the shock cases. KE tracks the filtered run to within 0.1% until t ≈ 4.4, then
turns upward and triples before positivity goes. As it does `dt` collapses to 2e-60, and
since Taylor–Green is unforced, rising KE is numerical. The filter is necessary and
sufficient for stability at 128³ and the Cook properties are neither; only the art-on leg
was run at 256³.

### The alpha sweep at 128 cubed

Five values at 128³, 224 ranks, `cfl = 0.35`, `filter_interval = 1`, `filter_cfl = 0`,
Gaussian smoother, `C_mu = 0.002`. The α = 0.486 row sits where the other four interpolate
the peak crossing:

```
alphaf   steps   peak -dKE/dt        vs window   KE misfit   -dKE/dt misfit   molecular   mu*    filter
0.40     11504   1.1814e-2 @ 8.38      −7.76%     8.986e-3     8.903e-2        59.6%      1.7%   38.7%
0.45     11737   1.2120e-2 @ 8.89      −5.40%     7.019e-3     5.666e-2        61.1%      2.4%   36.4%
0.486    12145   1.2865e-2 @ 8.85      +0.39%     4.919e-3     2.723e-2                          29.7%
0.49     12267   1.3004e-2 @ 8.85      +1.46%     4.887e-3     2.526e-2        68.0%      4.6%   27.4%
0.499    13240   1.3258e-2 @ 8.88      +3.37%     4.750e-3     2.910e-2        75.9%      8.4%   15.7%
```

At 128³ the misfits are far more sensitive to α than at 32³:

```
misfit fall from α = 0.40 to 0.49    KE misfit   -dKE/dt misfit
128³                                 46%         72%
32³                                  19%         8%, not monotone
```

The coarse grid's misfit is dominated by a resolution error common to every α, at 0.12 to
0.15. At 128³ that floor is fifteen times smaller, and α accounts for most of what remains.

**The dissipation misfit has a minimum at α = 0.49.** On the natural axis `1 − 2α`, the
per-pass strength, the kinetic-energy misfit approaches a floor that α cannot lower, while
the dissipation misfit turns:

```
interval of α     KE misfit fall per decade of 1 − 2α    -dKE/dt misfit
0.40 to 0.45      6.5e-3
0.45 to 0.49      3.0e-3                                  improves 55%
0.49 to 0.499     0.14e-3                                 15% worse
```

**The peak crosses the reference just below the misfit minimum.** Interpolating in
`log(1 − 2α)` places the total-dissipation match at α ≈ 0.486, the row added for it, where
both misfits are slightly worse than at the minimum. The peak is not a fit criterion on its
own, and the peak time carries less still, since the rate sits on a plateau around its peak:

```
α = 0.40    within 1.6% of the peak    t = 8.15 to 8.79
```

Across the first tenfold reduction in `1 − 2α` the filter's dissipation falls only 22% while
molecular dissipation rises 25% and μ\* triples: a weaker filter acts on a larger grid-scale
amplitude, so its dissipation responds far less than its coefficient. Over the last decade
it falls 41% and the resolved and μ\* channels more than replace it, so **the minimum is a
joint one**, the best α at `C_mu = 0.002` rather than a property of the filter alone.

The α = 0.45 row differs from the `:compact` 128³ rows above at matching resolution, rank
count, CFL, cadence and `C_mu`. An A/B on the coarse grid gives the same sign and order as
the 8.5% step-count gap at 128³, consistent with `plan_direction` skipping the Gaussian's
line solve and its collective interface stage:

```
32³ A/B      steps
:compact     3516
:gaussian    3078
```

### Spectra

`bench/archive/tgv_spectrum.jl` takes the shell-averaged kinetic-energy spectrum from a checkpoint
offline; the solver computes no transform. The spectrum sums to the solver's own kinetic
energy to 4e-4 relative, the Parseval check, the residual being the density fluctuation.

At 32³ the spectrum separates two settings far more sharply than the histories do. At t = 9
they differ by a factor of ten in the grid-scale band and by three and a half decades at
k = 16:

```
32³, t = 9                              α = 0.40    α = 0.49
E(k = 16)                               4.76e-10    1.21e-6
k^(5/3) E(k) flat to                    k ≈ 7       k ≈ 10
```

The histories separated them by 13% and the peak by 3%. At 128³, against the reference at
t = 9:

```
reference sum E(k) 8.6404e-2
alphaf   sum E(k)     vs reference   share above k = 32   E at Nyquist
0.40     8.4654e-2       −2.02%            0.110%           3.3e-11
0.45     8.5096e-2       −1.51%            0.267%           3.3e-10
0.49     8.5428e-2       −1.13%            0.681%           5.9e-8
0.499    8.5361e-2       −1.21%            1.326%           2.5e-6
```

Every setting has lost too much energy by t = 9, the deficit smallest at α = 0.49, a third
estimator agreeing with the misfit minimum and the peak crossing. The band above half the
Nyquist wavenumber grows by about 2.4 per step in α while `1 − 2α` falls by factors of two,
five and ten, so it shows no feature at the optimum. Below k ≈ 15 the spectra agree to a few
percent in no consistent order, and they separate monotonically only from k ≈ 18 upward.
The filter sets the top quarter of the wavenumbers, which hold under 1% of the kinetic
energy.

**The excess dissipation is concentrated in transition rather than at the peak.** Against
the reference at the same instants, −dKE/dt is high at both times and least so under the
weakest of the three filters:

```
−dKE/dt above the reference at the same instant
alphaf   t ≈ 4.4   t ≈ 8
0.40     11.8%     11.1%
0.45     9.7%      6.9%
0.49     6.5%      2.6%
```

The run's own peak then arrives early and turns over below the reference's, a shape
difference rather than a shortage.

**The tails carry no pile-up at any α, including the one the histories reject.** Every tail
steepens toward Nyquist (the last column of the table of sums), and α = 0.499 holds 43 times
the grid-scale energy of the fitted value while still sitting three and a half orders below
the spectral peak. The high-wavenumber share is therefore a one-sided bounding check, and the
fit belongs to the history misfit wherever the history is meaningful.

The dissipation range is not resolved at the screening resolution:

```
Re = 1600, ε ≈ 1.28e-2, Kolmogorov scale η ≈ 1.18e-2
resolution   k_max η
128³         0.75
256³         1.5
```

Molecular dissipation at the peak falls short of the reference total at every α, so even at
the weakest filter a fifth of the physical dissipation is off the grid:

```
alphaf                                                  0.40   0.45   0.49   0.499
molecular dissipation at the peak, of reference total   55%    58%    69%    78%
```

### The 256 cubed confirmation

Three values at 256³, otherwise at the settings of the α sweep:

```
896 ranks over eight rzhound nodes, 1.7 hours each at 65 Mpoint-steps/s
alphaf   steps   peak -dKE/dt        vs window   KE misfit   -dKE/dt misfit   filter
0.45     23178   1.3019e-2 @ 8.86      +1.36%     1.191e-3     1.226e-2         11.8%
0.49     23923   1.2916e-2 @ 8.86      +0.55%     8.122e-4     8.618e-3          6.7%
0.499    24783   1.2838e-2 @ 8.88      −0.07%     5.161e-4     7.457e-3          2.3%
```

**The minimum does not transfer.** Both misfits fall monotonically through the weakest
strength, and the turn seen at 128³ is absent. The peak reverses as well: every setting
overshoots, and the overshoot falls as the filter weakens, the boundary fit seen at 32³ and
not at 128³. The channel split gives the reason. Refinement cuts the filter's share of the
sink by a factor of three to seven at the two strengths both grids ran (the filter columns),
molecular dissipation carries 87% to 96%, and the dissipation range is resolved
([spectra](#spectra)), so nothing is left for a numerical sink. The best-scoring strength at
128³ is the one that best replaces the missing dissipation range there.

**α = 0.45 is too strong at both resolutions, and that conclusion transfers**: α = 0.49
scores better on every estimator at both, and the same ordering holds on the one-dimensional
battery. The interior optimum does not transfer, so **α = 0.49 is a subgrid tuning at 128³
and not a fitted constant.** No upper bound lies inside the range run at the finer
resolution; that comes from [the battery's stability edge](shocks.md#the-shock-battery). The misfits
fall by factors of six to nine and three to five under refinement at fixed α, so the 256³
numbers are not at a floor.

### The mu star channel on Taylor-Green

No case in the battery exercises μ\*, so the sensor-field change is measured on
Taylor–Green, where that channel carries a share of the sink:

```
64³, Re = 1600, to t = 10, default smoother, at the peak
mu*        reduction   steps   t_peak   peak -dKE/dt   molecular   mu*    filter
strain*    sum*        5888     8.49      1.2459e-2      33.8%     4.5%   61.6%
velocity   sum         5884     8.49      1.2421e-2      33.6%     6.0%   60.4%
strain     max         5682     8.97      1.2496e-2      33.3%     2.7%   64.0%
```

The velocity sensor raises the μ\* share by a third at the expense of the filter's, and the
directional maximum cuts it by 40%, the maximum over three directions being smaller than
their sum; both are moves of one to two points in a channel the filter dominates,
indistinguishable from a rescaling of `C_mu`. The peak time under `:max` is not: it lands
exactly on the reference peak time, the other two settings being half a time unit early.
All three 64³ peaks are below the tabulated reference, and β\* is 0.0% in all three.

## The filter's dissipation

```text
julia --project=. -t 1 bench/archive/filterrate.jl filter_cfl=0.4
```

A parallel shear layer, `u_x = 0.1 sin(4y)` at uniform ρ and p, is an exact steady solution
of the Euler equations and stays one discretely, since every x-derivative vanishes, so the
filter is the only mechanism that can change its kinetic energy. N = 32 to t = 0.5, zero
viscosity, artificial properties off. Two rejected alternatives: a broadband field loses 64%
of its kinetic energy within tens of steps and cannot lose more, which collapses the spread
across a fourfold CFL change to 1.2%; a velocity sine at uniform pressure is an acoustic
oscillation hundreds of times faster than the filter. Total energy shows nothing either, a
symmetric filter on a periodic grid conserving each discrete sum exactly.

### Dissipation per application

```
cfl    steps   unrelaxed          relaxed
0.4      42    2.357e-3  1.000    2.31554e-3
0.2      84    4.707e-3  1.997    2.31552e-3
0.1     168    9.386e-3  3.982    2.31552e-3
```

Unrelaxed, the loss tracks the step count, not the elapsed time: at half the CFL a
calculation applies twice the subgrid dissipation over the same interval. Relaxed, it is
constant to six figures across a fourfold change in timestep.

Shortened steps are the same dependence, so numerical dissipation depends on the output
schedule. `landing=0.037` schedules an `EveryTime` callback at an interval that does not
divide the step; unrelaxed, landing on thirteen instants adds twelve steps at `cfl = 0.4`
and the loss rises by 28.5% to 3.029e-3. Relaxed, landed and unlanded agree to six figures.

**Retries.** A steady case cannot provoke a retry, so `retry_at` has a callback set one
cell's density negative; the positivity check then raises `:negative_density` and `run!`
restores the savepoint, halves the CFL and discards the poisoned state.

```
       one retry after step 25                 two retries, after steps 25 and 45
cfl   steps  final  unrelaxed         relaxed  | steps  final  unrelaxed         relaxed
0.4     64   0.2    3.589e-3  1.000   2.31553e-3 |   88   0.1    4.930e-3  1.000   2.31553e-3
0.2    148   0.1    8.274e-3  2.306   2.31552e-3 |  256   0.05   1.426e-2  2.893   2.31552e-3
0.1    316   0.05   1.757e-2  4.896   2.31551e-3 |  591   0.025  3.259e-2  6.611   2.31551e-3
```

Unrelaxed, a retry raises the loss: the run at `cfl = 0.4` that retried once loses 52% more
than the one that did not, and at the lowest CFL two retries multiply it 3.5 times.
Relaxed, every entry is the unretried value, the first pass after the rollback reading the
halved `dt · rate` as taken.

**Subcycling.** A refined level filters at its own cadence and its pass reads the root
`dt · rate`, an upper bound on the level's own product. A level must nest four root nodes
inside the root on every side, so the refined region is a box, which costs `u_x(y)` its
exactness; the passive variant `u_z = 0.1 sin(4y)` with `u_x = u_y = 0` in a planar
`(32, 32, 1)` run stays exactly steady, so `planar=true component=3 refine=16` is the case,
a subcycled level-1 box of 16 root nodes about the centre. The planar run takes fewer steps
than the three-dimensional one, the Euclidean acoustic bound being √2 and not √3.

```
       no refinement                          refined box, subcycled
cfl   steps  unrelaxed         relaxed        | unrelaxed         relaxed       relaxed, one retry
0.4     34   1.909e-3  1.000   1.891e-3       | 1.508e-3  1.000   1.49460e-3    1.49470e-3
0.2     68   3.813e-3  1.997   1.891e-3       | 3.010e-3  1.996   1.49467e-3    1.49467e-3
0.1    135   7.551e-3  3.956   1.891e-3       | 5.956e-3  3.949   1.49463e-3    1.49468e-3
```

Relaxed, the composite loss is constant to five figures across the fourfold CFL change, and
so with a retry induced on top (the last column). The refined run loses about a fifth less,
the level filtering at a third of the spacing, 24 points per wavelength against 8.

### The relaxed formulation

`filter_cfl` is the CFL at which one pass is applied at full strength. Below it the state is
relaxed toward the filtered image, with the weight capped at w = 1, which holds the
dissipation per unit time fixed:

```
Q ← (1 − w) Q + w F(Q)
```

With `filter_cfl = 0` the unrelaxed path is taken exactly, as it is by any pass at or above
the reference CFL, so numbers taken at the reference are unchanged at `filter_cfl = 0.35`.

The weight is directional, as the filter is. Each pass along `d` reads

    w_d = filter_interval · dt · r_d · √n / filter_cfl,    r_d = max (|u_d| + c) / h_d

with the maximum over the domain and `n` the number of active dimensions. `max_rate`
evaluates the three `r_d` alongside the selecting rate and reduces them in the same
`Allreduce`, five scalars in place of two, and `run!` records them in `filter_rate_prev`.
The `√n` is the ratio of the Euclidean acoustic rate to the one-dimensional one on an
isotropic grid, keeping `filter_cfl` in the convention of `cfl`. The step, the artificial
coefficients and the physical diffusivities are outside the weight.

**No single scalar rate meets the gate.** A weight on the maximum rate that sizes the step
gives 68k weighted passes on an aspect-ratio-16 grid and completes the aligned Noh case with
a wrong solution ([directional bulk viscosity](shocks.md#directional-bulk-viscosity)). An
acoustic-limited run beside it made 3.8k. A weight on the Euclidean acoustic rate
`c √(Σ_d 1/h_d²)` would still make the coarse direction's passes grow as `√(1 + AR²)`, and a
molecular diffusive rate carries the same `1/h²` penalty on a stretched wall-resolved grid.
The gate was one profile at every aspect ratio:

```
AR   steps   limit      plateau   deficit   shock    transverse
 1     741   diffusive  3.9841    62%       0.2120   5.4e-12
 4    4968   diffusive  3.9837    63%       0.2121   9.5e-11
16   70453   diffusive  3.9837    63%       0.2121   6.8e-8

N = 400
 1                      3.9957    62%       0.2046
 4                      3.9956    62%       0.2046
```

At N = 400 the two milder aspect ratios agree to the fourth digit of the plateau and in
deficit and shock position. The step is sized as under the scalar weight, so the step counts
are the scalar weight's to within the change of trajectory.

The directional weight lowers the shear layer's relaxed loss by 1.5%, from 2.351e-3:
`u_x(y)` is filtered only by the `y` pass, whose weight reads `√3 c / h` rather than a total
carrying the advective `|u_x| / h`; the subcycled case has no advective term and did not
move. On Taylor–Green with the artificial properties off, the reference-CFL run is no longer
bit-identical to the unrelaxed one, since the weight dips below one wherever the velocity at
the largest-rate point is spread over two or three axes. With them on, both resolutions are
artificial-diffusion-limited for part of their length, so the filter's share falls, μ\*'s
rises and the misfits improve:

```
Taylor–Green at the reference CFL, scalar weight → directional weight
art off, 32³    peak moved 0.08%, misfits 0.06% and 0.03%
art on          filter share       KE misfit          peak time
32³             83.7% → 81.2%      0.157 → 0.147
64³             67.2% → 62.2%      0.0492 → 0.0435    8.35 → 9.05, reference 8.97
```

The 128³ fits, at `filter_cfl = 0` where the weight is one, are untouched. Every Noh row is
diffusion-limited at the front under C_β = 1 and moved toward a weaker filter; Shu–Osher's
L1 fell 0.7%, while Lax and Woodward are acoustic-limited and did not move.

### The timestep moves the attribution, not the total

Taylor–Green at 32³, Re = 1600, artificial properties on, Gaussian smoother, at the
dissipation peak:

```
filter_cfl   cfl     peak -dKE/dt        filter    mu*     molecular
0 (default)  0.35    1.4216e-2 @ 6.58    82.2%     5.1%    12.6%
0 (default)  0.17    1.4297e-2 @ 6.61    85.0%     3.6%    11.4%
0.35         0.35    1.4216e-2 @ 6.58    82.2%     5.1%    12.6%
0.35         0.17    1.4373e-2 @ 6.44    82.1%     5.2%    12.7%
```

At the reference CFL the relaxed run reproduces the unrelaxed one to every printed digit,
the `w = 1` path. Halving the CFL changes the peak dissipation by 0.6% but moves the μ\*
share by 29% relative, from the timestep alone at fixed `C_mu`; under the relaxation both
shares hold. The sinks compete for a fixed supply set at the large scales, so a filter
taking more at the grid scale leaves less for μ\* and molecular dissipation. Under
`filter_cfl = 0`, then, `C_mu` is conditional on the CFL, and a refit must state it. 32³ only.

### The fit instrument

Each point is scored by the relative L2 distance of its kinetic-energy and −dKE/dt histories
from the vendored reference over every step, and by the peak seen through the run's own
window, a curve fit rather than the single scalar `C_mu` was fitted on. Both misfits are
normalized by the reference's own RMS, the rate is compared only where the full window fits
inside the run, and the truncated final step is excluded. At 32³ and α = 0.45 a shakeout
reproduces the channel shares exactly. Its peak is 1.4217e-2. Nothing follows from that grid
about the default: the best misfits there are 0.12 and 0.67, and the weakest filter wins
where the filter carries most of the sink.

### Cadence and alpha are one axis

Six points at 128³, two α values crossed with three values of `filter_interval`:

```
interval  alphaf   (1 − 2α)/interval   steps   peak vs window   KE misfit   -dKE/dt misfit   filter
1         0.45     0.10                11737       −5.40%        7.019e-3     5.666e-2         36.4%
2         0.45     0.05                11938       −1.50%        5.447e-3     3.881e-2         33.2%
4         0.45     0.025               12191       +0.65%        4.913e-3     2.627e-2         29.0%
1         0.49     0.02                12267       +1.47%        4.887e-3     2.526e-2         27.4%
2         0.49     0.01                12521       +3.07%        4.911e-3     2.682e-2         24.4%
4         0.49     0.005               12821       +3.12%        4.825e-3     2.800e-2         20.2%
```

**Cadence and α are not independent settings.** The rows are ordered by
`(1 − 2α)/interval`, the per-pass strength divided by the cadence. On that axis the
dissipation misfit is single-valued with its minimum at 0.02, and the α sweep's points lie
on the same curve. The two closest settings on the combined axis, the third and fourth rows,
agree closely, and their differences are the instrument's resolving power:

```
third and fourth rows, 25% apart on the axis    agree to
KE misfit                                       0.5%
-dKE/dt misfit                                  4%
filter share                                    1.6 points
```

Fit α at `interval = 1`; the equivalence holds on smooth turbulence only, untested on
a shocked case.

### The relaxation leg

Four points at 128³ and α = 0.45, two CFLs crossed with two values of `filter_cfl`, without
snapshots, since a shortened landing step pays a full pass under `filter_cfl = 0`:

```
cfl    filter_cfl   steps   peak vs window   peak time   KE misfit   -dKE/dt misfit
0.17   0            23021       −8.04%          −0.68     8.867e-3      8.896e-2
0.35   0            11737       −5.41%          −0.05     7.019e-3      5.666e-2
0.17   0.35         23472       −5.20%          −0.08     7.013e-3      5.827e-2
0.35   0.35         11737       −5.41%          −0.05     7.019e-3      5.666e-2
```

Unrelaxed, halving the CFL doubles the passes and moves every estimator away from the
reference. Relaxed, the two CFLs agree, and the fourth row reproduces the second in every
digit:

```
halving the CFL      KE misfit     -dKE/dt misfit   peak time
filter_cfl = 0       26% worse     57% worse        0.63 earlier, 7% of the run
filter_cfl = 0.35    within 0.1%   within 2.8%      within 0.03
```

**A lower CFL is a stronger filter under `filter_cfl = 0`.** Halving the timestep doubles
the numerical dissipation per unit time and moves the answer further from the reference,
with nothing in the output to indicate it; retries, subcycled levels and shortened output
steps do the same locally. Relaxation removes the dependence and lets an α fitted at one CFL
be used at another. The reference CFL is one number for every dimensionality while the CFL
rate is not, so the one-dimensional battery had to clear the change at its production CFL
numbers ([the shock battery](shocks.md#the-shock-battery)).

## Filtering on non-uniform volumes

```text
julia --project=. -t 1 bench/archive/filter_conservation.jl
```

The alternative to the unweighted component filter on cylindrical, spherical and stretched
grids is the volume-weighted form of the [public Pyranda
implementation](https://github.com/LLNL/pyranda/tree/master/pyranda/parcop), which filters
J·q and divides by a cell volume passed through the same filter (`filter` in
`parcop/operators.f90`, `CellVolS` in `parcop/mesh.f90`), flipping the radial parity since
the cylindrical volume is odd across the axis. It sits behind `filter_weighting = :volume`.

**What conservation is, discretely.** A directional pass is a matrix M on each component
along a line, and the node quadrature weights node i by V_i = w_i J_i h, w trapezoidal
(`quad_weight`, without the edge factor `volume_integral` applies at a curved face). The
pass conserves Σ V_i q_i for every q exactly when Mᵀ V = V, so d = Mᵀ V − V is one pass's
defect and d_i / V_i the fraction of node i's content created or destroyed. Constant
preservation, the unit row sums of M, is a different property, held to 1e-15 by both forms
on every line below. The operators come from unit impulses on lines of 64 nodes at α = 0.45
via `filter_state!`.

```
                                        max |d_i|/V_i (row)   rows 9..56   Σ|d|/ΣV
cartesian periodic         none, volume     4e-16                4e-16       2e-16
cartesian walls, cascade   none, volume     4.1e-2 (1)           1.6e-3      3.6e-3
cartesian walls, onesided  none, volume     2.9e-2 (5)           2.0e-3      3.6e-3
stretched a = 0.5          none             4.1e-2 (wall)        1.7e-3      5.5e-3
                           volume           4.1e-2 (wall)        1.6e-3      5.4e-3
cylindrical axis           none             4.0e-2 (wall)        1.8e-3      3.6e-3   axis row 8.6e-3
                           volume           1.5e-1 (axis)        5.2e-3      4.1e-3
spherical origin           none             4.1e-2 (wall)        2.0e-3      5.4e-3   origin rows 6e-11
                           volume           4.1e-2 (wall)        1.6e-3      5.1e-3   origin rows 2e-14
spherical poles, θ line    none             8.6e-3 (pole)        7.3e-5      9.5e-5
                           volume           1.5e-1 (pole)        5.2e-3      2.9e-3
```

The defect of a closed line is the closure rows'. Past the first rows the cascade filter's
column sums alternate with a ratio per row equal to the root of α r² + r + α, so the defect
decays slowly into the interior, and more slowly under a weaker filter:

```
cascade filter column sums     α = 0.45                     α = 0.49
rows 1–3                       +2.0e-2, −4.0e-2, +2.2e-2
alternating ratio per row      −0.627                       −0.817
twelfth row                    3.8e-4                       1.1e-3
```

The one-sided rows move the peak to rows 3–7 without reducing the total. Stretched and
curvilinear lines carry the same wall defect as the uniform one, and the weighting changes
the interior figure by under 1e-4. At the folds the forms part: the even J = r² of the
spherical origin conserves to round-off under both, while the odd J of the cylindrical axis
and the poles costs the weighted form a first-row defect of 0.15, the folded operator of the
product having no unit column sums. The unweighted
filter's non-conservation on a non-uniform volume is real but is the wall closure's, present
on a uniform Cartesian grid too.

**The runs.** The battery's curvilinear cases, with the filter's own change of total mass
and energy over the run relative to the largest total seen. The runs filter through a
callback with the solver's own pass disabled, at the script's α = 0.45 rather than the
default 0.47.

```
                    plateau    wall deficit   shock     filter mass   filter energy   steps
Noh nu=1  none      3.9988      23.9%        0.2021     +1.64e-5      -1.65e-5       3557
          volume    3.9988      23.9%        0.2021     +1.64e-5      -1.65e-5       3557
Noh nu=2  none     15.0086      55.6%        0.2091     -4.09e-6      +3.24e-7       2205
          volume   15.0168      46.3%        0.2090     +1.04e-5      -6.60e-8       2197
Noh nu=3  none     62.5634      24.9%        0.2089     +2.00e-11     -3.85e-11      1092
          volume   62.6028      38.8%        0.2088     +1.93e-11     -3.76e-11      1086

Sedov     none     R_s 0.8085 (+1.06%)   peak rho 5.127   filter mass -9.8e-14   3143
          volume   R_s 0.8084 (+1.05%)   peak rho 5.144   filter mass -1.2e-14   3149

shock/interface, 121 points, t = 0.15        worst Y            width   steps
  uniform          none, volume              -0.0045 / 1.0045   3       122
  clustered a=0.5  none                      -0.0019 / 1.0019   4       196
                   volume                    -0.0021 / 1.0021   4       196
```

The planar case and the uniform interface are the same run to every digit, the weighting
being skipped on a uniform volume. Over a run the filter's change of total mass is small at
the planar wall and the axis and at round-off at the origin, where both forms conserve as
the operator rows predict; the energy change is of the same order or smaller. The weighting
lowers the wall deficit at the axis at 2.5 times the filter's mass change, and raises it at
the origin, where neither form changes the mass, so that change is in the filtered image's
shape at the singular cell. Sedov does not move, and on the clustered interface the
weighted form overshoots slightly more. Two changes since the weighting was first measured
moved the planar row: the one-sided rows made the default at `d611bae`, which create two
orders less filter mass on a closed line ([the filter's wall rows](walls.md#the-filters-wall-rows)),
and the slip wall's flux contract at `e2a6c4f`, which took this script's wall deficit from
49.8% to 23.9%.

**Decision.** `filter_weighting = :none` stays the default, `:volume` the measured
alternative. A filter conserving on a closed line would have to change the closure rows,
which is the wall-closure question of [the filter's wall rows](walls.md#the-filters-wall-rows), not
a metric one. At the symmetry cell the weighting is a per-case adjustment, not a correction.
