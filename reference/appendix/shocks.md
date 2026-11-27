# Calibration appendix: Shocks and the artificial properties

This file holds the sections of the measurement record on the shock battery, the CFL
ceiling at the symmetry cell, the energy ahead of a shock, the directional bulk
viscosity, the aligned Noh mode, false activation on smooth fields and the
ablation-driven capsule. [CALIBRATION_APPENDIX.md](../CALIBRATION_APPENDIX.md) indexes
the record and states its conventions.

## The shock battery

```text
julia --project=. -t 1 bench/artcal.jl <part>   # beta kappa D Y detector field filter bulk brill2025 response
julia --project=. -t 16 test/validation.jl
```

The one-dimensional cases are defined in `test/cases.jl` and shared with
`test/validation.jl`. Every grid here is uniform and Cartesian, or has its angular
directions collapsed, so the physical and computational spacings coincide.

**ν is the Noh geometry index**, one more than the exponent of r in the area weight. The
cases run at γ = 5/3. The exact post-shock compression is then 4^ν:

```
ν   geometry      symmetry point     exact plateau
1   planar        slip wall           4
2   cylindrical   axis fold          16
3   spherical     origin fold        64
```

**Noh plat/exact** is the post-shock density plateau over its exact value, so an exact
plateau reads 1.0000. The **deficit** is the density shortfall at the symmetry point, which
is wall heating. The **Lax L1 / contact** columns are the mean density error against the
exact Riemann solution and the 10–90% contact width, which measures the broadening from the
regularization. The **Shu train amp** is the peak-to-trough density in the post-shock
entropy wave train, which pulls against every damping constant. The **WC peak** is the peak
density in the Woodward–Colella collision, a survival check at a 10⁵ pressure ratio. A
**NaN** is a positivity loss or a stall; every case carries an `nmax`, and `StepControl`
raises `SolverFailure` about 150 steps before the collapse.

Every constant table is one-dimensional. `C_mu` multiplies the shear component of the
artificial stress, which these cases reach only through the stress trace, so its table
constrains stability and not shear accuracy and [Taylor-Green](filter.md#taylor-green) carries the
shear measurement; `C_beta`, `C_kappa` and `C_D` are fully exercised. The constants were
fitted under `compact_filter(0.45)` at full strength every step, and none of them moves
under the relaxed default, measured below.

### The current reading

`test/validation.jl` at `-t 1` under the current defaults (C6 `:neutral3`,
`compact_filter(0.47, closures = :onesided)`, `filter_cfl = 0.35`, `smoother = :gaussian`,
`detector = :species_d8`, the node-centred sensor rows and the slip-wall flux contract). The
second column places `SymmetryPlaneBC` where the `SlipWallBC` was ([the face-centred
symmetry plane](walls.md#the-face-centred-symmetry-plane)):

```
case                                   node-centred wall            symmetry plane
Woodward–Colella N=800, t=0.038  L1 3.153e-2, peak 6.6384 @0.7772   L1 2.982e-2, 6.6283 @0.7781
Noh ν=1 cold N=400, cfl 0.15     3.9988, 24%, shock 0.2021,         3.9990, 27%, 0.2024,
                                 pre-shock L1 2.19e-5, 10 cells     1.63e-5, 8 cells
                                 e < 0, e_min −0.0293               e_min −0.0094
Noh aligned N=100, AR=4          4.0030, 33%, 0.2084, 4925 steps,   3.9994, 30%, 0.2093,
                                 transverse 1.8e-9, 96 cells e < 0  4887 steps, 2.0e-10, 84 cells
Noh ν=2                          15.0099, 55%, 0.2091, 6 cells      –
Noh ν=3                          62.5811, 25%, 0.2089, 8 cells      –
Noh plane AR=2 (four inflows)    11.862, 0.2356, L1 0.895, 745 steps, 545 cells e < 0
Lax                              L1 ρ 4.930e-3, u 7.462e-3, p 7.551e-3
Shu–Osher                        L1 ρ 6.741e-3, train L1 2.090e-2, train peak 4.6799
Sedov                            R_s 0.8083 (+1.04%), peak ρ 5.140, 4 cells e < 0, e_min −0.00229
shock/SF6 interface              worst Y −0.0050 / 1.0050, width 5 cells, 642 steps
```

The cells with e < 0 sit ahead of the shock fronts
([negative internal energy ahead of a shock](#negative-internal-energy-ahead-of-a-shock)).

### C_beta, the shock constant

Under the default smoother at `NOH_CFL = 0.15`:

```
C_beta    | Noh1 plat  deficit | Noh3 plat | Lax L1  contact | Shu train | WC peak
0         |     NaN       NaN  |      NaN  | 5.7e-3   0.0032 |    1.6610 |    NaN
0.25      |  0.9995       +56% |   1.0098  | 4.8e-3   0.0041 |    1.6408 | 6.4151
0.5       |  0.9994       +61% |   0.9931  | 4.9e-3   0.0045 |    1.6297 | 6.4742
1.0    *  |  0.9993       +64% |   0.9751  | 5.0e-3   0.0053 |    1.6180 | 6.6049
2.0       |     NaN       NaN  |   0.9569  | 5.2e-3   0.0063 |    1.6055 | 6.6859
4.0       |     NaN       NaN  |   0.9408  | 5.4e-3   0.0074 |    1.5945 | 6.8366
```

At `C_beta = 0` there is no shock regularization and every strong-shock case loses
positivity; the Lax tube survives on the compact filter alone, with the narrowest contact in
the table since no artificial bulk viscosity broadens it. At the upper end planar Noh does
not complete, so the upper bound is a stability bound and moves with the CFL. Between the
limits every accuracy measure varies monotonically: a larger constant widens the Lax contact,
damps the Shu–Osher train and lowers the spherical plateau, which crosses exact between 0.25
and 0.5. Accuracy alone puts the optimum near 0.4. The default of 1.0 is a robustness
margin. Against half that value its contact is 18% wider and its wave train 0.7% weaker.

In the `:delta4` columns of the CFL ladder under the detector refit below, `C_beta` trades
the planar and cylindrical ceilings against the spherical one monotonically, and the default
maximizes the spherical ceiling.

**Recommendation:** retain 1.0. Use 0.5 for interface-dominated Richtmyer–Meshkov or
Rayleigh–Taylor work with moderate shocks, noting that it costs the spherical origin half
its timestep. Values below 0.25 or above 2 are not recommended.

### C_kappa, the conductivity

Under `smoother = :compact`, kept as the record of that setting:

```
C_kappa   | Noh1 plat  deficit | Noh3 plat | Lax L1  contact | WC peak
0         |  0.9998       +64% |      NaN  | 4.8e-3   0.0051 | 6.6847
0.0025    |  0.9997       +61% |   0.9728  | 4.9e-3   0.0051 | 6.6390
0.01   *  |  0.9992       +58% |   0.9732  | 5.0e-3   0.0051 | 6.5731
0.04      |  0.9980       +57% |   0.9748  | 5.3e-3   0.0052 | 6.4934
0.16      |  0.9986       +70% |   0.9796  | 5.8e-3   0.0052 | 6.3176
```

Artificial conduction transports spuriously deposited entropy out of the stagnation cell and
excessive conduction adds its own error, so wall heating falls to 0.04 and then rises.
Spherical Noh does not complete at `C_kappa = 0`, so regularizing the momentum equation
through `C_beta` does not by itself carry a converging strong shock. Lax L1 grows across the
range while the contact barely moves, since κ\* diffuses temperature and the Lax contact is
nearly isothermal. The lever inverts under the default `:gaussian` smoother (below).

**Recommendation:** retain 0.01. Zero is not recommended.

**The cold-state limit is not reached.** κ\* is built as `C_kappa · (ρc/T_ion) · sensor`,
and the 1/T factor does not collapse `dt` in a cold ambient, because the step is limited by
the artificial bulk viscosity:

```
planar Noh, cfl 0.15, N = 400, the limiting rate sampled every 100 steps
p0      steps   plateau   deficit  shock    inadmissible  e_min/e0    dt median  dt min
1e-2    3718    3.8900    +63%     0.2119    2            -0.66       1.64e-4    1.51e-4
1e-3    3655    3.9858    +63%     0.2052    4            -28         1.65e-4    1.48e-4
1e-4    3650    3.9959    +63%     0.2046    7            -255        1.65e-4    1.55e-4
1e-5    3652    3.9969    +63%     0.2045   10            -2.4e3      1.66e-4    1.12e-4
1e-6    3650    3.9970    +63%     0.2045   11            -2.3e4      1.65e-4    1.55e-4
1e-8    3652    3.9971    +63%     0.2045   17            -2.2e6      1.65e-4    1.55e-4
```

The step count and the median step are unchanged over six decades of ambient pressure. The
diffusive rate is the limiter on about 85% of the samples, and the bulk viscosity is its
largest term at both pressures sampled; only a tenfold `C_kappa` makes κ\* the limiter:

```
p0 = 1e-4 and 1e-8 alike, at step 2000
largest (μ* + β*)/(ρh²)    262
largest κ*/(ρ c_v h²)      14.6
acoustic rate              488

C_kappa    steps at both pressures
0          3580
0.1        4330, κ* the limiter
```

The singular factor is not reached. At a cell whose internal energy is negative,
`primitives!` floors T_ion at 1e-300 and the sound speed vanishes with it, so ρc/T_ion
evaluates to about 1e-140 and κ\* is zero on those cells rather than large. A cold ambient
changes only the count of inadmissible cells: the precursor's negative internal energy has a
fixed absolute amplitude of about −0.035, and the ambient internal energy falls beneath it.
`artificial_conductivity_scale` remains the EOS dispatch point where a tabular or
condensed-matter model sets its own scale.

### C_D, the species diffusivity

```text
julia --project=. -t 1 bench/artcal.jl D Y bulk alphaf=0.47
```

Every table of this section was measured under the defaults, the `:species_d8` detector and
the `:partial_density` channel except where a row names another, and under the solver's
filter strength, 0.47, which `alphaf` sets in place of the script's 0.45. A sharp binary
interface advected on the periodic slab with two such edges (`bench/artcal.jl D`):

```
u = 1, t = 0.5, N = 256, initial 10–90% width 2h = 0.0078
C_D       | interface width
0         |  0.01773
0.01      |  0.01773
0.03      |  0.01774
0.1       |  0.01777
0.3       |  0.01786
1      *  |  0.01813
3         |  0.01875
10        |  0.01997
```

With D\* disabled the interface still more than doubles in width, so the compact filter is
the dominant source of broadening for a passive interface. The default adds 2% to that
width, and the sweep up to `C_D = 10` adds 13%. Under `:fickian` at two species the two
per-species diffusivities agree to round-off, since the second mass fraction is one minus
the first and gives the same detector output and the same excursion.

The passive width does not measure what C_D is for. Behind a shocked interface the mass
fractions ring inside [0, 1], where the bound of the next section is zero, and C_D is the
only damping. The measure is the total variation of the final profile beyond the 1 a
monotone one carries (`TV − 1`), with the grid-scale content |δ⁴Y|/16 on the light side
more than four cells from the interface. On the shocked air/SF6 interface of the next
section (`bench/artcal.jl Y` prints every column but the light-side one; `cells` is the
final width):

```
C_D       | 2h: TV − 1   light-side   worst min Y   cells   steps | 4h: TV − 1   worst min Y
0.01      |     0.0574   3.64e-3      -0.0102       3       650   |     0.0704   -0.0026
0.03      |     0.0602   3.29e-3      -0.0099       3       650   |     0.0699   -0.0025
0.1       |     0.0648   2.05e-3      -0.0094       4       648   |     0.0639   -0.0024
0.3       |     0.0136   1.57e-3      -0.0075       4       645   |     0.0069   -0.0021
1      *  |     0.0045   1.99e-3      -0.0050       5       642   |     0.0051   -0.0012
3         |     0.0040   7.36e-4      -0.0036       8       642   |     0.0039   -0.0005
10        |     0.0010   1.10e-3      -0.0011       9       644   |     0.0010   -0.0002
```

At 0.1 the excess is a two-cell lump of the heavy gas, Y ≈ 0.035, separated from the
interface by one cell of nearly pure light gas, and it is the same excess at every smaller
constant and on the wider initial interface. The default removes it, and the excess is flat
from 1 to 3 while the shocked interface widens from five cells to eight. A smaller lump
forms in the He/CO2 shock tube and persists to a larger constant, under either
thermodynamics:

```
1-D reduction of the bench/archive/he_co2_shock_tube.jl tube: 10:1 helium driver, flat He/CO2 contact
at x = 3 m, slip walls, 768 nodes, to 2.5 ms (scratch script, no bench entry). TV − 1 of
Y_CO2 is the time mean from 0.77 ms, when the transmitted shock has crossed the contact;
light-side is |δ⁴Y_CO2|/16 at the end, more than four cells from the contact.
C_D     channel           thermodynamics        TV − 1    light-side   worst Y    steps
0.01    partial_density   NASA-9                0.1978    5.78e-3      -0.0039    2003
0.03    partial_density   NASA-9                0.0401    3.85e-3      -0.0038    1997
0.05    partial_density   NASA-9                0.0126    3.17e-3      -0.0037    1996
0.1     partial_density   NASA-9                0.0128    2.53e-3      -0.0033    1995
0.3     partial_density   NASA-9                0.0139    3.11e-3      -0.0019    1994
1       partial_density   NASA-9                0.0023    2.04e-3      -0.0025    1990
3       partial_density   NASA-9                0.0018    6.02e-4      -0.0012    1985
10      partial_density   NASA-9                0.0019    4.86e-4      -0.0008    1988
1       fickian           NASA-9                0.0035    2.21e-3      -0.0020    1995
1       bulk              NASA-9                0.0022    1.92e-3      -0.0023    1990
0.01    partial_density   calorically perfect   0.1842    6.39e-3      -0.0035    2000
1       partial_density   calorically perfect   0.0022    1.82e-3      -0.0023    1991
```

At 0.1 the tube's final profile carries a lump of CO2 under a percent in Y, separated from
the contact by a cell of nearly pure helium. The ringing stays at that level from 0.05 to
0.3. The default removes the lump and cuts the ringing to a fifth, and larger constants
leave it near that level. The calorically perfect gas rings as NASA-9 does, so the lump does
not come from the thermodynamics. What remains at the default on both cases is a grid-scale
trail on the light side, which `C_D = 3` cuts about threefold at three more cells of shocked
width on the air/SF6 case.

On the two-dimensional tube a larger constant reduces the worst undershoot and shortens the
run:

```
2-D deck of bench/archive/he_co2_shock_tube.jl, 384 × 24 to 2.5 ms (scratch script, same setup)
C_D     worst undershoot over the run   steps
0.1     −4.67e-3                        1758
0.3     −3.27e-3                        1691
1    *  −1.83e-3                        1651
3       −8.92e-4                        1649
```

Pyranda's own decks (`shockBubble.py`, `triplePoint.py`) write the species diffusivity as

```
D = 2e-4 ρ h² ring(Y)/Δt,   Δt = h/(|u| + c) at their CFL of 1,   ring = 240 × compact_d8
```

which is C_D ≈ 0.05 on `:d8` scaled by |u| + c in place of c, about 0.07 behind the shock
in the He/CO2 tube; `RM3D.py` and `RT3D.py` use half that. The default is twenty times
their constant. At their constant the tube's ringing is a sixteenth of its value at 0.01,
while on the air/SF6 interface the lump persists at every swept constant up to 0.1.

**Recommendation:** 1.0, the default, the lowest value of the sweep that removes the lump on
both cases. It costs 2% on the passive width, one cell of shocked width against 0.3, and no
steps. Larger values cut the remaining trail at the cost of a wider shocked interface. Under
`:fickian` the interface pressure error grows with the constant (see [the partial-density
species channel](#the-partial-density-species-channel)).

### C_Y, the mass-fraction bound

```text
julia --project=. -t 1 bench/artcal.jl Y alphaf=0.47
```

A shocked species interface rings, and the ringing grows as the initial interface narrows. A
shock in air runs into a tanh interface with SF6 under the default `ArtificialProperties`
and with the bound off, at the solver's filter strength (scratch script; `bench/artcal.jl Y`
prints the worst excursions, the final widths and the `C_Y` sweep below):

```
Mach 1.5, air γ = 1.4, R = 1; SF6 γ = 1.09, density 5.04; N = 400, Dirichlet ends, cfl = 0.4
initial   | defaults: worst Y   final Y range       | C_Y = 0: worst Y    final Y range       | width (cells)
2h        | -0.0050 / +1.0050   -0.0006 / +1.0006   | -0.0079 / +1.0079   -0.0038 / +1.0038   | 6 → 5
4h        | -0.0012 / +1.0012   -0.0007 / +1.0007   | -0.0043 / +1.0043   -0.0037 / +1.0037   | 12 → 6
8h        | clean               clean               | clean               clean               | 24 → 9
```

The shock compresses the interface by the density ratio across it, and the ringing is set by
the cells the interface spans after compression. At the default `C_D` the ringing sensor
alone holds the 2h interface to under a percent outside the unit interval. The bound takes a
third off the run's worst excursion and cuts the final one sixfold. With the bound off, the
numerical settings move the excursion little, and a smaller `C_D` loses the interface:

```
2h interface, C_Y = 0, each setting varied alone   worst min Y                        steps
defaults                                           −0.0079                            641
N = 800                                            −0.0076                            1297
N = 1600                                           −0.0068                            2610
compact_filter(0.47, closures = :cascade)          −0.0079                            641
compact_filter(0.3)                                −0.0411                            635
cfl = 0.2                                          −0.0077                            1279
beta_sensor = :dilatation                          −0.0080                            641
detector = :delta4                                 −0.0055                            640
detector = :d8                                     −0.0088                            620
C_D = 0.01                                         outside the species band, stopped
C_D = 0                                            outside the species band, stopped
artificial properties disabled                     outside the species band, stopped
```

Resolution, CFL, the filter's closure rows, the sensor field and the detector move the
excursion by a third at most, and a stronger filter raises it fivefold. At Cook's constant,
`C_D = 0.01`, and below it the ringing sensor does not hold an interface the shock has
thinned, and the run leaves the band of `StepControl.species_band` and stops under the
strict validity policy.

The species diffusivity therefore carries the mass-fraction bound of [Cook (2007, eq.
18)](https://doi.org/10.1063/1.2728937), zero wherever Y lies in the unit interval:

```
D*_k = c · G[ max( C_D h |D Y_k| , C_Y h max(0, −Y_k, Y_k − 1) ) ]
```

with G the sensor smoother and D the species detector. Cook used `C_Y = 100`. His later
form [(2009, appendix A, eq. A8)](https://doi.org/10.1063/1.3139305) writes the bound as
twice the max, with `C_Y = 50`. Both Cook forms use the Δ²/Δt scaling. [Shankar, Kawai & Lele
(2011, eq. A4)](https://doi.org/10.1063/1.3553282) carry `C_Y = 100` into the cΔ scaling
used here. This implementation uses a maximum rather than their additive combination, the
later operator documented by [Brill, Olson & Bokman (2025, eq.
24)](https://arxiv.org/abs/2503.12680). Cook's scaling differs from the cΔ form by 1/CFL, so
the published values map to values near 100 here. On the 2h case under the defaults:

```
C_Y      | worst Y            | final Y range      | steps
0        | -0.0079 / +1.0079  | -0.0038 / +1.0038  | 641
50       | -0.0062 / +1.0062  | -0.0010 / +1.0010  | 641
100   *  | -0.0050 / +1.0050  | -0.0006 / +1.0006  | 642
200      | -0.0039 / +1.0039  | -0.0004 / +1.0004  | 644
500      | -0.0025 / +1.0025  | -0.0003 / +1.0003  | 647
1000     | -0.0018 / +1.0018  | -0.0002 / +1.0002  | 652
```

The worst excursion falls steadily with the constant, more than threefold from 50 to 1000.
The step count rises over the same range by under 2%. The form of the term was chosen under `:delta4` at
`C_D = 0.01`, where the bound carries the whole excursion; the rows are kept as the record of
that choice:

```
:delta4, C_D = 0.01, 2h interface
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

Sum and max are indistinguishable, as are smoothing the two terms separately and smoothing
their max once (Cook's single filter of the bracket), which costs no line solve beyond the
ringing sensor's. The unsmoothed term is a grid-scale diffusivity that helps less and
triples the step count through `compute_dt`. The bound keeps its effect on a stronger shock,
where at Cook's constant it is the difference between a bounded interface and a mass
fraction below −1:

```
Mach 3, 2h interface, permissive validity (scratch script)
config                   worst min Y    steps
C_Y = 0                  −0.0257        1408
C_Y = 100  *             −0.0105        1411
C_D = 0.01, C_Y = 0      −1.2807        1415
C_D = 0.01, C_Y = 100    −0.0152        1422
```

The bound enters the diffusive rate like the rest of D\*. With the form-variant rows above,
excluding it from the rate was stable to `C_Y = 500` and saved 1% of the steps. At 1000 the
excluded form took a square root of a negative pressure, while the included form completes
the sweep above. Cook's Δ²/Δt scaling fixes the bound's diffusion number by construction, so
keeping it in the rate is the same choice.

**The dead band.** The term is not inert on a smooth profile that touches a bound: the
intermediate Runge–Kutta stages leave the unit interval although the completed steps never
do, and without a dead band the bound raises the error by a factor of 20 to 350:

```
two-species advection of Y₁ = (1 + cos 2πx)/2 at uniform ρ, p and u, one period, cfl = 0.4;
L2 error of Y₁ (scratch script)
config                       N = 32     N = 64     N = 128    N = 256    orders
defaults                     7.54e-08   1.73e-09   6.33e-11   3.33e-12   5.45 4.77 4.25
C_Y = 0                      7.54e-08   1.73e-09   6.33e-11   3.33e-12   5.45 4.77 4.25
C_D = 0                      7.51e-08   1.73e-09   6.33e-11   3.33e-12   5.44 4.77 4.25
C_D = 0, Y_tolerance = 0     1.62e-06   1.60e-07   1.41e-08   1.18e-09   3.34 3.50 3.58
Y_tolerance = 0              1.64e-06   1.61e-07   1.41e-08   1.18e-09   3.35 3.51 3.59
species channel off          7.51e-08   1.73e-09   6.33e-11   3.33e-12   5.44 4.77 4.25
```

The excursion appears only at the second to fifth stages, at 6e-6 on 64 points. It falls 4×
per doubling. The dead band of `Y_tolerance = 1e-4` restores the `C_Y = 0` errors to the
last digit and leaves the worst excursion of the shocked 2h interface unchanged in four
digits. Evaluating at stage 1 only does the same; the dead band stores nothing and is the
form adopted.

**Cost on a smooth profile.** The default species channel costs nothing on the resolved
cosine: its errors match those with the channel off to within half a percent at the coarsest
grid and to three digits above it. The profile touches both bounds and the dead band keeps
the bound inert there, so the species detector's response to the cosine is below anything
the error shows.

**The surrounding formulation**, measured under `:fickian`, `:delta4` and `C_D = 0.01`. A
resting two-gas interface stays in pressure and temperature equilibrium under the bound, and
its mass-fraction overshoot comes from the filter. On a stationary interface at uniform p
and T and zero velocity for t = 0.25, the compact filter
alone leaves p, u and T uniform to 1e-14. It is the whole source of the ±1e-3 mass-fraction
overshoot of a resting 2h interface, since the uniform-(p, T) state is a linear subspace of
the conserved variables for ideal gases of constant c_v, which a componentwise linear filter
preserves. The artificial species flux with its enthalpy energy flux holds T uniform to
1e-6. The interdiffusion velocity it generates is u ≈ −D\* ∇ln ρ, 1.4e-4 at D\* ≈ 1e-6. The
same flux carrying species internal energy instead, which Brill et al. argue for, leaves a
secular temperature drift 50× larger. Three species at equal density under the per-species
diffusivities hold bulk density to 4e-14, so the correction velocity already gives what
Brill et al. obtain from a shared diffusivity. Under the default `:partial_density` channel a
uniform (u, p, T) state stays uniform to round-off ([the partial-density species
channel](#the-partial-density-species-channel)).

**Recommendation:** `C_Y = 100` with `Y_tolerance = 1e-4`, the defaults. At the default
`C_D` the bound takes a third off the shocked interface's worst excursion and more than half
at Mach 3 for a fraction of a percent more steps, and the dead band keeps it inert on a
smooth profile. At a smaller
`C_D` it is what holds a shock-thinned interface. Larger values hold the excursion tighter at
a few more steps.

### The sensor smoother

Cook's sensor smoothing is `gbar` in Pyranda, which resolves to `cgfs4` in its [public
stencil source](https://github.com/LLNL/pyranda/tree/b4e0afc),
`pyranda/parcop/stencils.f90`: an explicit nine-point symmetric stencil with `nol = 0` and
`implicit_op = .false.`, with weights

```
centre       ±1           ±2           ±3          ±4
3565/10368   3091/12960   1997/25920   149/12960   107/103680
```

Over the common denominator 103680 these sum to exactly 1, so constants are preserved
without cancellation; at boundaries the overhanging weight folds onto the mirror point.
There is no linear solve and no interface reduction, and the halo is four.
`ArtificialProperties.smoother` takes that Gaussian or one pass of `compact_filter(0.45)`.
As a Cook test filter the compact pass is close to the identity over the whole resolved
band, at a distributed line solve per active dimension per sensor:

```
transfer function       k/π = 0.25   k/π = 0.75
compact_filter(0.45)    0.999        0.85
Gaussian                0.663        0.021
```

Under the Gaussian the curved geometries' CFL ceilings rise, the spherical one most. The
highest CFL reaching `t_final` with a correct plateau, the ladder extended until both settings fail:

```
Noh geometry              :compact   :gaussian
nu = 1  planar wall          0.2        0.2
nu = 2  cylindrical axis     0.15       0.2
nu = 3  spherical origin     0.15       0.4
```

Both fail at 0.5 everywhere, so 0.4 is a measured ceiling and not a table edge, and the
origin, the least forgiving fold under `:compact`, becomes the most forgiving one. The
mechanism is sensor intermittency at the damage site ([the CFL
restriction](#the-cfl-restriction-and-the-symmetry-cell)); the cost moves the same way
([operator and step cost](cost.md#operator-and-step-cost)).

Accuracy changes little either way, and wall heating is the one clear regression: the planar
wall deficit rises from +58% to +64% while plateaux and the Shu–Osher train move in the
fourth digit. **More κ\* does not recover the wall heating.** The planar wall deficit over a
`C_kappa` sweep under each smoother:

```
C_kappa      0      0.0025    0.01*     0.04     0.16
compact    +64%     +61%      +58%      +57%     +70%
gaussian   +65%     +64%      +64%      +68%     +91%
```

The lever inverts. The Gaussian widens the κ\* footprint and lowers its peak, so added
conductivity spreads over a region rather than concentrating in the wall cell where the
entropy error is deposited. At `C_kappa = 0` the spherical case fails under `:compact` and
completes under `:gaussian`, and its plateau is better under `:gaussian` across the sweep.
**Three dimensions are neutral.** Under either smoother the filter still dominates the
Taylor–Green sink at 64³. The μ\* share of that sink moves from 4.0% to 4.5%, so `C_mu` is
not refitted.

**Recommendation:** `:gaussian`, the default. The wall-heating regression is a property of
the smoother and is largely recovered by `detector = :d8` or by the one-sided filter wall
rows ([the filter's wall rows](walls.md#the-filters-wall-rows)).

### The ringing detector

`ring()` in `parcop/operators.f90` dispatches to `d8x/d8y/d8z`, a full compact operator
(`c10d8`, `parcop/stencils.f90`) with a pentadiagonal left-hand side, a nine-point
right-hand side, and its own symmetric and antisymmetric boundary closures. `artificial.jl`
uses the undivided fourth difference δ⁴, following [Cook
(2007)](https://doi.org/10.1063/1.2728937) literally. `ArtificialProperties.detector` selects between
them: `:delta4` is δ⁴ and `:d8` is [`compact_d8`](../../src/core/kernels_banded.jl), the reference
operator transcribed. The default, `:species_d8`, applies `:d8` to the mass and mole
fractions and δ⁴ to every other sensed field, so on a single-species case it is `:delta4`
exactly. The interior rows of the two are

```
:delta4   δ⁴f_i = f_{i-2} − 4 f_{i-1} + 6 f_i − 4 f_{i+1} + f_{i+2}
:d8       1.5 g_{i-2} + 14 g_{i-1} + 29 g_i + 14 g_{i+1} + 1.5 g_{i+2} = 60 δ⁸f_i,
```

δ⁸ the undivided eighth difference. It is planned as a symmetric operator rather than a
derivative, and its four closure rows fold the overhanging interior weights onto the
half-offset mirror (`reference/DESIGN.md`). Every closure row's weights sum to zero,
measured at 1.2e-16.

**Normalization.** The coefficients are divided by ζ = 29 to put a unit diagonal on the
left-hand side, and by a further 240. The second factor sets the response to a grid-to-grid
oscillation to 16, the value undivided δ⁴ gives there, so both detectors agree at the
wavelength both exist to catch and diverge only below it (the velocity ratios of the
response table below). Without it `:d8` would produce sensors 240× larger at the Nyquist
and the four constants would need refitting by two orders. This is not Pyranda's own
normalization.

**Results**, on top of `smoother = :gaussian`: the failure the Gaussian fixes is β\*
intermittency at a symmetry cell, and a sharper high-pass makes narrower sensor spikes, so
`:d8` against the compact smoother would be rejected for a defect of the smoother.

```
detector    Noh1 plat  deficit   Noh3 plat   Lax L1   contact   Shu train   WC peak
delta4       0.9993     +64%      0.9751     5.0e-3   0.0053     1.6180     6.6049
d8           0.9997     +53%      0.9951     4.7e-3   0.0044     1.6360     6.3953
```

Six of seven columns improve under `:d8`, several well beyond the fourth digit that
separates the β\* sensor variants: the spherical plateau error falls fivefold, wall heating
recovers eleven of the points the smoother change cost, and the Shu–Osher train gains 1.1%.
Woodward–Colella peak density is the one regression. The CFL ladder splits (the sensor-field
table below): `:d8` raises the planar and cylindrical ceilings past 1.0 and lowers the
spherical one. The `1.0+` is not a table edge, since the plateau under `:d8` is flat to four
digits from 0.15 to 1.0 in both geometries. The spherical failure is at the origin cell, the
symmetry-cell startup mechanism of [the CFL
restriction](#the-cfl-restriction-and-the-symmetry-cell), with `:d8` moving its threshold
and not its character.

**Selectivity depends on the sensor field**, and is available to the κ\* and D\* channels
and largely unavailable to μ\* and β\*. `bench/artcal.jl response` puts a velocity sine of
one wavelength on a periodic 64-point line and reads the peak coefficient back:

```
k/pi   ppw  |  mu* from |S|:   d4       d8    ratio  |  mu* from u:    d4       d8     ratio
0.125  16.0 |             7.24e-5  4.20e-5   1.72    |            4.11e-6  4.18e-10  9.8e+03
0.250   8.0 |             3.31e-4  2.58e-4   1.28    |            4.71e-5  8.29e-08     569
0.500   4.0 |             2.44e-3  2.44e-3   1.00    |            3.93e-4  1.51e-05      26
0.750   2.7 |             8.33e-4  6.49e-4   1.28    |            1.60e-3  5.07e-04    3.16
1.000   2.0 |             0.00e+0  0.00e+0    ---    |            3.14e-3  3.14e-03       1
```

Applied to the velocity the two detectors reproduce their designed ratios to four figures;
applied to |S| they differ by a factor of 1.8 or less from 32 points per wavelength down to
the Nyquist, because |S| has a cusp wherever the strain passes through zero, a cusp is
grid-scale structure at any resolution, and no detector is insensitive to one. On a pressure
wave resolved over 64 points the two differ by a factor of 2.6e6 on κ\*, whose input is the
internal energy. On β\* the same wave gives a factor of 1.8. The last row is a property of
every differentiated field: a centered scheme has zero modified wavenumber at the Nyquist,
so |S| and ∇·u vanish identically for a two-point velocity wave. The calculation is stable
in spite of that because grid-scale dissipation comes from the compact filter, not the Cook
properties.

**The `C_beta` refit under `:d8`**, back to back on one machine with a fresh `:delta4`
control:

```
                :delta4                                  :d8
C_beta   Noh1 deficit  Noh3 plat  contact  Shu     Noh1 deficit  Noh3 plat  contact  Shu
0.25        +56%        1.0098    0.0041  1.6408      +33%          NaN     0.0039  1.6489
0.5         +61%        0.9931    0.0045  1.6297      +44%          NaN     0.0041  1.6420
1.0   *     +64%        0.9751    0.0053  1.6180      +53%        0.9951    0.0044  1.6360
2.0         NaN           0.9569  0.0063  1.6055      +58%        0.9767    0.0046  1.6300
4.0         NaN           0.9408  0.0074  1.5945      +62%        0.9578    0.0051  1.6232
```

The viable window moves and the optimum inside it does not. Under `:delta4` the window is
bounded above by planar Noh, and under `:d8` it is bounded below by the two converging
geometries. **The two windows intersect in the single value 1.0**, the default. Under `:d8`
the response to the constant is flatter on every smooth measure: across the sweep the
contact broadens 31%, against 80% under `:delta4`. The CFL ladder per `C_beta`:

```
             nu = 1    nu = 2                nu = 3
C_beta    :d4  :d8   :d4     :d8          :d4   :d8
0.25      1.0+ 1.0+  1.0     0.4 only     0.2   none
0.5       0.4  1.0+  0.4     >= 0.4       0.25  none
1.0  *    0.25 1.0+  0.2     1.0+         0.4   0.25
2.0       none 1.0+  none    0.4          0.3   0.3
```

No `C_beta` in the sample recovers the spherical origin under `:d8`. The best available
ceiling, at `C_beta = 2`, stays below the one `:delta4` reaches at the default constant, and
it raises the spherical plateau error to 2.33%, against 0.49% at the default constant.
**`C_beta = 1.0` is retained under `:d8`**, and the detector decision falls to the origin
cell alone. One measurement is in the detector's favour: at the default constant the worst
geometry's ceiling rises under `:d8`, 0.2 to 0.25.

**A failure that gets worse as the timestep falls.** Two cells of the `:d8` ladder are not
ceilings, since the cylindrical case fails below a CFL at which it completes:

```
:d8, ν = 2       completes at cfl    fails at cfl
C_beta = 0.25    0.4                 1.0, and 0.3 and below
C_beta = 0.5     1.0 and 0.4         0.3 and below
```

Both are positivity losses and not the step cap. A CFL-type stability restriction
cannot produce a failure that appears only below a CFL; a per-step operation can, since a
fixed physical interval integrated at half the timestep applies it twice as many times, and
the per-step operation in the loop is the compact filter ([the filter's
dissipation](filter.md#the-filters-dissipation)). The sign of the dependence is the evidence; the
accumulation itself has not been measured step by step.

**Recommendation:** δ⁴ for the sensors of μ\*, β\* and κ\*, which the default `:species_d8`
keeps on every field but the mass and mole fractions. The battery favours `:d8`, as do the
planar and cylindrical ceilings, but the general guidance for converging shocks rests on the
spherical case, where `:d8` costs 40% of the timestep. The four constants are also the δ⁴
fit, and a detector that changes the sensor's spatial support by this much has no claim on
them. Nothing is shown wrong with `:d8`; `:delta4` survives the origin excursion at a larger
timestep. The species fields under `:d8` are measured under [false activation on smooth
fields](#false-activation-on-smooth-fields).

### The sensor fields and the compression switch

Cook's [2007 model](https://doi.org/10.1063/1.2728937) builds μ\* and β\* from the strain
magnitude |S| = sqrt(S_ij S_ij); his [2009 model](https://doi.org/10.1063/1.3139305) changes
β\* to the dilatation. Brill, Olson & Bokman (2025, eq. 25) build μ\* from the velocity
components, the form of Pyranda's `ringV(u, v, w)`, the ring of each velocity component
along each direction reduced by `MAX` over the nine pairs, and β\* from `ring(∇·u)`; neither
field carries an absolute value. Most of Pyranda's example decks keep μ\* on `ring(|S|)`
([Pyranda's implementation](#the-partial-density-species-channel)). `ArtificialProperties.mu_sensor`
(`:strain`, `:velocity`), `ArtificialProperties.beta_sensor` and `ArtificialProperties.reduction` (`:sum`, `:max`)
select between them. The weight is h_d for a field carrying one velocity derivative fewer
against h_d² for |S| and ∇·u, and the two coincide at the grid scale, so the four constants
transfer between sensor fields as they do between detectors. With S the strain-rate tensor,
δ⁴ the undivided fourth difference, Δ = ∇·u, ω the vorticity and ε a fixed regularizer at
the literature value 1e-32, `beta_sensor` has four settings:

```
beta_sensor           sensor                                         added cost
:strain (default)     Cook (2007), Σ_d h_d²|δ⁴_d S|                  –
:gated_strain         :strain × the compression switch               one pointwise pass,
                      H(−Δ)·Δ²/(Δ² + |ω|² + ε)                       no line solves
:dilatation           rebuilt from Δ and gated, the full form of     one further
                      Mani, Larsson and Moin (JCP 228, 2009)         smoothing pass
:ungated_dilatation   :dilatation without the switch, the reference form
```

The battery under each combination of detector and sensor fields:

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

**μ\* from the velocity components moves no column past the fourth digit**, since every case
here is one-dimensional at `C_mu = 0.002`. Under `:d8`, **β\* from the dilatation improves
four columns and degrades two**: the spherical plateau error falls from 0.49% to 0.24% and
the Lax contact sharpens. The Shu–Osher train loses 0.5% and the Woodward–Colella peak 1.5%.
Under `:delta4` the same change gives up more and gains less. The gated and switched
forms, under `:delta4` and the `:compact` smoother:

```
sensor       | Noh1 plat  deficit | Noh3 plat | Lax L1  contact | Shu train | WC peak
strain     * |  0.9992       +58% |   0.9732  | 5.0e-3   0.0051 |    1.6185 | 6.5731
gated_strain |  0.9992       +58% |   0.9722  | 5.0e-3   0.0052 |    1.6177 | 6.5850
dilatation   |  0.9990       +59% |      NaN  | 5.0e-3   0.0051 |    1.6226 | 6.5315
```

`:gated_strain` is a wash on accuracy, every column moving in the fourth digit and both
ways, and the Shu–Osher gain belongs to the sensor field and not to the switch. The CFL
ceilings:

```
                             nu = 1     nu = 2      nu = 3
detector  mu*       beta*     wall       axis       origin
delta4    strain    strain      0.2       0.2         0.4
delta4    velocity  strain      0.2       0.2         0.4
delta4    any       ungated_dil 0.2      none         0.2
d8        strain    strain      1.0+      1.0+        0.25
d8        velocity  strain      1.0+      1.0+        0.25
d8        any       ungated_dil 1.0+      0.2         0.2
```

The ungated dilatation keeps the spherical origin and loses the cylindrical axis;
`:dilatation`, the same sensor with the switch applied, loses both, so the spherical loss is
the switch's and the cylindrical loss the sensor field's. The gate on the strain sensor
raises the cylindrical ceiling instead, and leaves planar and spherical Noh unmoved:

```
:delta4, :compact smoother    ν = 2 ceiling
strain                        0.15
gated_strain                  between 0.2 and 0.25
```

Under the gate the plateau at 0.2 agrees with its own value at 0.15, so the larger step is a
completion and not a run that avoided the positivity check.

**Why the dilatation sensor loses the axis.** The cause is at the coordinate fold and is
visible at t = 0 before any shock forms. Noh starts from u_r = −1 everywhere, for which the
strain and dilatation sensors are analytically identical away from the axis: Δ = −1/r and
|S| = 1/r with zero vorticity, so the switch is exactly 1 and β\* agrees bit for bit. At the
first few cells of the axis the discrete radial derivative of u_r is no longer zero, and
there the two forms diverge, because Δ = S_rr + S_θθ adds two same-signed components where
|S| = √(S_rr² + S_θθ²) partially cancels them. Cylindrical Noh at N = 256, t = 0:

```
i   r        |S|      div       beta* strain   beta* dilatation
1   0.00196  612.0    -847.8    1.746e-02      3.790e-02
2   0.00587  204.1     -57.9    1.466e-02      4.106e-02
3   0.00978  110.9    -145.1    3.771e-03      2.087e-02
6   0.02153   46.5     -44.1    0              8.089e-04
120 0.46771    2.14     -2.14   3.856e-12      3.856e-12
256 1.00000    1.00     -1.00   1.199e-07      1.199e-07
```

The run loses positivity on the first step, and doubling `C_beta` repeatedly up to 8 does
not recover it. This is a property of the fold rather than of the shock capturing, so the
option is not recommended in converging geometry at any setting of the constants.

**Switch selectivity and decomposition sensitivity.** The gate removes nearly all of β\* on
a field with no compression but not its maximum, and neither compression-keyed setting
reproduces to round-off when the process grid changes. On a solenoidal Taylor–Green field at
32³, where β\* has nothing legitimate to do, `:gated_strain` removes 99.4% of the summed β\*.
It leaves 71 points above 1e-12. The maximum survives, at 0.42 of the ungated value. Those
surviving points are the cusps of |S|: the strain sensor is a fourth difference, so it peaks
where |S| passes through zero with a kink, and the switch degenerates there because the
vorticity vanishes too. A relative ε scaled to the local |S| was tried and dropped, since
the scale it would use vanishes with |S|. `:dilatation` has no such points and its β\* falls
to 1e-15 of the ungated maximum. Summed over the domain, three different split axes agree
to:

```
beta_sensor     relative spread of the summed β* over three split axes
strain          1e-14
gated_strain    2e-6
dilatation      2e-7
```

The cause is the switch and not a missing halo exchange, since the sensor fields themselves
reproduce to 1e-14: H(−Δ) is discontinuous at Δ = 0, and with ε at 1e-32 the ratio has not
decayed by the time Δ reaches round-off, so a point whose dilatation cancels to zero carries
either no β\* or the full C_β·ρ·sensor depending on the last bit. Anything relying on
bit-identical results across process grids should stay on `:strain`.

**Recommendations.** Retain `:strain`, `:strain` and `:sum`. Reach for `:gated_strain` in
cylindrical converging geometry, where it buys a 33% larger timestep for one pointwise pass
and costs nothing measurable elsewhere; the default would move only on a re-baseline across
the battery and a second geometry showing the same gain, which the symmetry-cell measurement
constrains: the gate relieves the axis cell, and the planar wall and spherical origin are
measured not to respond. `:dilatation` suits Cartesian shock-dominated work with no
coordinate fold, where the Shu–Osher amplitude gain is useful. `mu_sensor = :velocity`
cannot be judged until `C_mu` is refitted under it, since it moves the μ\* share of the
Taylor–Green sink by a third with the constant held fixed.

### The filter strength on the battery

`bench/artcal.jl filter` runs the battery at five filter strengths, each case at its
production settings: Noh at `cfl = 0.15`, Lax and Shu–Osher at 0.4, Woodward–Colella at 0.3.
Under the default `filter_cfl = 0.35`, where Noh receives 0.43 of a pass per step,
Woodward–Colella 0.86, and the two tubes run above the reference and receive a full pass:

```
alphaf   Noh1 plat  deficit | Noh2 plat | Noh3 plat | Lax L1  | Shu amp | WC peak
0.40      0.9992     +64%   |  0.9374   |  0.9762   | 5.1e-3  | 1.6146  | 6.5900
0.45      0.9990     +63%   |  0.9376   |  0.9766   | 5.0e-3  | 1.6180  | 6.6075
0.486     0.9989     +56%   |  0.9380   |  0.9776   | 4.9e-3  | 1.6192  | 6.6163
0.49      0.9989     +55%   |  0.9380   |  0.9775   | 5.0e-3  | 1.6192  | 6.6312
0.499     0.9984     +48%   |  0.9358   |    NaN    | 5.7e-3  | 1.6223  | 6.6917
```

Every case completes through α = 0.49, the Woodward–Colella collision included. The columns
whose reference value comes from outside the code agree with the Taylor–Green fit: both
curved Noh geometries are closest to exact at α = 0.49, the Lax error is flat across the
three middle strengths and worse at the weakest, and the Shu–Osher amplitude rises
monotonically as the filter weakens. At α = 0.499 the spherical Noh loses positivity, the
cylindrical plateau falls and the Lax error rises, so the other converging geometry and the
shock tube degrade before the origin fails. The ceilings do not move with α:

```
cfl   alphaf | Noh1 plat  deficit | Noh2 plat | Noh3 plat | WC peak
0.4   0.45   |    NaN      NaN    |   NaN     |   NaN     | 6.6105
0.4   0.49   |    NaN      NaN    |   NaN     |   NaN     | 6.6368
0.3   0.45   |  1.0017    −387%   |   NaN     |  0.9764   | 6.6075
0.3   0.49   |  1.0000    −139%   |   NaN     |  0.9775   | 6.6312
0.2   0.45   |  0.9993     +58%   |  0.9380   |  0.9766   | 6.6077
0.2   0.49   |  0.9992     +49%   |  0.9382   |  0.9775   | 6.6310
```

The single apparent exception is not a raised ceiling: the planar case at `cfl = 0.3` keeps
positivity at the weaker strengths but returns a plateau above one with a wall excess where
every healthy row carries a deficit, larger at the weaker of the two, a changed failure mode
rather than a working configuration.

**The stability edge.** The strength at which the converging cases stop being stabilized is
the upper bound on α that no Taylor–Green leg produced. The per-pass strength in force is

```
ε = (1 − 2α) · w,   w = min(1, cfl / filter_cfl)
```

and both curved Noh geometries were run at eight points, chosen so that the relaxed and
unrelaxed rows interleave on ε:

```
N = 256, cfl = 0.15, default constants
alphaf    filter_cfl   w      ε        | Noh2 plat | Noh3 plat  deficit
0.49      0            1      0.02000  |   0.9377  |   0.9769     +28%
0.4975    0            1      0.00500  |   0.9379  |   0.9771     +30%
0.4925    0.6          0.25   0.00375  |   0.9378  |   0.9770     +30%
0.49875   0            1      0.00250  |   0.9374  |   0.9765     +31%
0.495     0.6          0.25   0.00250  |   0.9375  |   0.9766     +33%
0.4995    0            1      0.00100  |   0.9361  |     NaN
0.4975    0.6          0.25   0.00125  |   0.9366  |     NaN
0.499     0.6          0.25   0.00050  |   0.9337  |     NaN
```

**The edge is a property of the per-pass strength in force, not of α or of the formulation,
and it sits between ε = 0.00125 and 0.0025** for the spherical Noh. At equal ε the two
formulations agree to 1e-4 in both plateaus and both fail below the edge, and the
cylindrical plateau degrades monotonically toward it without failing. Retries spend that
margin: `StepControl` lowers the CFL by `cfl_backoff = 0.5` each time, and under the relaxed
formulation each halving halves ε with it. Under `filter_cfl = 0.35` the margin of α = 0.45
at the Noh CFL is seventeen times the edge:

```
filter_cfl = 0.35    ε at the Noh CFL    after retries
α = 0.45             0.043               0.0027 on the fourth
α = 0.49             0.0086              at the edge on the second
```

**The constants do not move under a weaker filter.** `bench/artcal.jl beta kappa D Y` was
run at the two candidate defaults and at the adopted one, beside a baseline at the fitted
configuration in the same environment, which reproduces every table above to the fourth
digit.

```
background                  | Noh3 at C_beta=1 | Noh1 deficit at C_kappa=0.01 | mix width at C_D=0.01 | worst Y at C_Y=100
α = 0.45 unrelaxed (fitted) |     0.9751       |     +64%, the trough         |   0.01820             |   −0.0135
α = 0.49 unrelaxed          |     0.9769       |     +59%, the trough         |   0.01800             |   −0.0136
α = 0.45, filter_cfl = 0.6  |     0.9775       |     +60%, the trough         |   0.01812             |   −0.0114
α = 0.45, filter_cfl = 0.35 |     0.9766       |     +63%, the trough         |   0.01820             |   −0.0135
```

None of the four moves. `C_beta = 0` loses positivity on every strong-shock case under all
three backgrounds; the accuracy optimum stays between 0.25 and 0.5; the Shu–Osher amplitude
and the contact width vary with the same slopes; and the robustness bound at `C_beta = 2`
holds. The `C_kappa` wall-heating trough sits at 0.01 under all three, and the `C_D` and
`C_Y` sweeps keep their ordering and nearly their size:

```
range over the three backgrounds
C_D, width change over the 64-fold sweep     18 to 21%
C_Y = 100, worst excursion                   1.1 to 1.4%
C_Y = 200, worst excursion                   0.9 to 1.0%
```

The one visible change is the `C_Y = 1000` row, a filter interaction at a value no default
is near.

**The default `filter_cfl = 0.35`, measured at α = 0.45.** At `cfl = 0.35` the weight is one
and every result above that CFL, the Taylor–Green fits included, is unchanged bit for bit. Below
it the strength falls with the CFL, the α = 0.479 equivalent at the Noh CFL, with a margin
of seventeen times the edge and eight times after one retry. A run at a low CFL then takes
the same dissipation per unit time as one at 0.35 rather than several times more, and
lowering the CFL, retrying, or writing output more often no longer moves the answer away
from the reference. α = 0.49 is the fitted value for smooth turbulence at both Taylor–Green
resolutions and stays a per-run selection. At 256³ it improves the kinetic-energy misfit by a
third, at the cost of a margin of three times the edge at the Noh CFL, which the second retry
spends. The supporting measurements are in [the filter's dissipation](filter.md#the-filters-dissipation)
and [Taylor-Green](filter.md#taylor-green), where α = 0.45 at full strength is too strong on every
estimator at both resolutions and the interior optimum at 128³ is a subgrid tuning.

**The default α = 0.47.** The strength that suits smooth flow and the one that suits a shock
through a refinement boundary differ. A weaker filter lets a precursor of the shock survive
ahead of it, and a coarse–fine interface amplifies it. On the Sod case of
`test/level_tests.jl` (N = 201, one refined region, t = 0.1), the largest momentum ahead of
the shock reads:

```
alphaf   unrefined   refined
0.45     3.8e-10     3.4e-10
0.47     1.4e-9      1.35e-8
0.49     4.3e-8      9.0e-7
```

The default sits between the two regimes; 0.45 and 0.49 remain per-run selections for
shocks through refined levels and for resolved turbulence. At the Noh CFL its per-pass
strength is 0.026, ten times the edge and five after one retry. The battery under it is
[the current reading](#the-current-reading), and `test/validation.jl` carries the full set.

The pins in `test/cases.jl` moved with the default. A default move has to move them with
`Numerics`, or the battery measures a configuration the solver no longer runs, as happened
when the smoother default moved. Six tests whose assertions are recorded trajectories pin
`filter_cfl = 0` explicitly and say so.

### The Brill 2025 parameter set

The coefficient families and their original values come from [Cook (2007, eqs.
15–18)](https://doi.org/10.1063/1.2728937). [Brill, Olson & Bokman (2025, eqs.
22–29)](https://arxiv.org/abs/2503.12680) use a later set with an eighth-derivative
detector, the max over directions, μ\* from the velocity components and β\* from the ungated
dilatation. Their detector F (eq. 22) is the eighth derivative times Δ⁸, which is Pyranda's
`c10d8` with a two-point response of 3840, and `compact_d8` is the same operator divided by
240 ([Pyranda's implementation](#the-partial-density-species-channel)). μ\* (eq. 25), β\*
(eq. 26) and the species diffusivity of their new method (eqs. 34–37, c_s F Δ) contain no
Δt; only the traditional D\* of eq. 24 carries Δ²/Δt. Each constant multiplying a detector
output therefore converts by 240:

```
constant   Cook (2007, eqs. 15–18)   Brill et al. (2025)      converted, ×240
C_mu       0.002                     C_μ = 1e-4               0.024
C_beta     1                         C_β = 7e-2               16.8
C_kappa    0.01                      C_κ = 1e-3               0.24
C_D        0.003                     C_D,Y = C_D,V = 2e-4     0.048
C_Y        100                       C_O,Y = 100              200, on twice the excursion
```

The converted C_kappa holds up to the rate factor of Pyranda's
κ\* = C_κ ρ c_v G[h² ring(T)]/(T Δt), which eq. 27 does not settle. The rows keep the default
C_Y. Converting by 1/CFL instead gives constants 96 times too small.

In `bench/artcal.jl brill2025`, `sensors` is `detector = :d8, reduction = :max, mu_sensor =
:velocity, beta_sensor = :dilatation`; `Brill` is the converted set, `ungated` the same
under `beta_sensor = :ungated_dilatation`, theirs; the last row holds C_beta at the default.

```
config               | Noh1 plat   def | Noh2 plat   def | Noh3 plat   def | Lax L1  | Shu tr | WC peak | mix wid | SI minY  wid
default              |    0.9997   +24% |    0.9380   +55% |    0.9774   +29% | 5.0e-03 | 1.6183 |  6.6068 | 0.01817 | -0.0046    5
sensors only         |    0.9998   +27% |    0.9483   +60% |    0.9940   +40% | 4.7e-03 | 1.6287 |  6.4178 | 0.01817 | -0.0058    6
sensors, Brill       |    0.9997   +26% |    0.9093   +65% |    0.9216   +42% | 5.6e-03 | 1.5823 |  6.8131 | 0.01788 | -0.0072    3
ungated, Brill       |    0.9996   +27% |    0.9095   +64% |    0.9205   +38% | 5.7e-03 | 1.5768 |  6.6891 | 0.01788 | -0.0074    3
sensors, Brill, Cb=1 |    0.9997   +11% |    0.9561   +41% |    1.0058   +39% | 5.0e-03 | 1.5714 |  6.1650 | 0.01788 | -0.0141    4
```

The converted set over-damps, and the loss is the bulk constant, gated or not: its C_beta of
16.8 puts both curved plateaus several percent further below exact than the sensors alone
do, and C_beta = 1 with the other three constants converted restores them. Every row
completes all three Noh geometries at the default CFL; the CFL ladder was not run. The
Shu–Osher train loses 2.2 to 2.9% against the default under every converted row, while the
sensors alone strengthen it. C_mu and C_kappa in that set are 12 and 24 times the defaults.
The default row senses the species fields with `:d8` and the sensors-only row every field,
both at C_D = 1; they hold the shocked interface's excursion below every converted row,
whose C_D is 0.048. Within the converted rows the excursion doubles when C_beta returns to
1, so there it follows the bulk viscosity. None of the converted rows is the default.

### The bulk species channel

`ArtificialProperties.species_flux = :bulk` replaces the Fickian artificial species flux by one
diffusive flux F_q = −D_b ∇q on every conserved variable, with D_b built from the Fickian
channel's own bracket sensed on both the mass and the mole fraction of every species
(`reference/DESIGN.md`, "The species channel"). Everything below is on the one-dimensional
cases under `detector = :delta4` and `C_D = 0.01` unless a row says otherwise, the Gaussian
smoother, D_b in the timestep, at CFL 0.4. "Default" in this section names `:fickian`, the
reference channel of these rows.

Brill et al. find that the traditional Fickian channel leaves a pressure oscillation on an
advected bubble that their diffused-density formulation removes. Their section 3.1 reports
the end pressure oscillation:

```
a bubble of density R in gas of density 1 at p = 1, T = 1 and u = 10, ten periods through a
periodic unit square, N_p points across the interface, N_p Δ = 0.05
formulation           end pressure oscillation                                stable at N_p = 7
traditional Fickian   "at best around 1% of the pressure and at worst          for small R
                      greater than 25%"
diffused density      O(1e-6 to 1e-12)                                        for R ≥ 100
```

The slab analogue `brill_slab` in `test/cases.jl` is one-dimensional with N = 20 N_p. Under
the Fickian channel it reproduces the paper's picture, somewhat better than the paper
reports it. The error is the Fickian channel's alone: at uniform u, p, T every linear
operator of the scheme preserves the uniform state to round-off, and the Fickian enthalpy
flux Σ_k h_k J_k is the one that does not at unequal gas constants. Switching that channel
off with nothing in its place takes every completing row to 1e-12 to 1e-10 while losing two
rows to instability, and adding the bulk flux beside the Fickian one moves nothing, so no
added term removes the Fickian pressure error.

What the channel does depends on the field D_b is sensed on: X the mole fraction, Y the mass
fraction, XY the maximum over both.

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

At R = 1000 and N_p = 7 every row fails, with the density undershooting beside the jump over
seven cells before any diffusivity has acted; R = 10 completes on every row. The two
sensors do different jobs. The mole fraction is the volume fraction and sits on the density
jump, so a sensor on it carries the ratio-100 shocked interface, which X and XY complete and
Y and the Fickian channel do not. The mass fraction on the light side of a ratio-R interface
amplifies a volume-fraction excursion by up to R, so a sensor on it bounds Y where the
mole-fraction sensor is blind to that excursion. The maximum over both is the reference
implementation's own combination (their eqs. 33–37), and it is at least as good as either
and as the default in every column. The channel is sensed on XY; the single-field forms
exist only in the prototype.

At equal molecular weights X ≡ Y and the bulk flux of ρY_k at uniform ρ is the Fickian flux,
so `species_advection` reproduces the default's 10–90% width to eight digits and the smooth
bounded profile its L2 errors to seven. On a resting air/SF6 interface at uniform p and T
the bulk channel keeps velocity, pressure and temperature at round-off while the density
relaxes, where the enthalpy flux of the Fickian default drives a velocity and a pressure
error:

```
resting air/SF6 interface, 298 steps
channel     max |u|    pressure           max |δT/T|   ρ relaxes by
bulk        1.2e-14    |δp/p| 3.8e-14     4.0e-14      2.0e-2
fickian     2.2e-4     1.3e-4
```

The channel costs n_cons gradient line solves per direction in place of the Fickian
flux's n_species, four more, plus one further detector and smoother pass per species.

The two constants under the channel (`bench/artcal.jl bulkconst`), on the same rows, with
the Fickian default as the reference line. The rows at C_D = 0.005 and 0.02 lie between
their neighbours and are not printed.

```
channel  C_D     C_Y | mix width | SI 5.04: worst Y / cells / steps | SI 100                | slab 100/7: max|p-1| / worst Y / rho_min / steps
fickian  0.01    100 | 0.01820   | -0.0129 / 4 / 647                | FAIL                  | 1.86e-02 / -0.0810 / 0.9949 / 4207
bulk     0.0025   50 | 0.01795   | -0.0170 / 3 / 641                | -0.0318 / 7 / 674     | 6.30e-11 / -0.0950 / 0.9907 / 4207
bulk     0.0025  100 | 0.01795   | -0.0112 / 3 / 645                | -0.0197 / 7 / 684     | 3.43e-11 / -0.0645 / 0.9973 / 4242
bulk     0.0025  200 | FAIL      | -0.0060 / 3 / 648                | -0.0222 / 7 / 715     | 3.73e-11 / -0.0440 / 0.9985 / 4275
bulk     0.01     50 | 0.01819   | -0.0164 / 3 / 641                | -0.0311 / 7 / 674     | 4.27e-11 / -0.0947 / 0.9900 / 4161
bulk     0.01    100 | 0.01820   | -0.0108 / 3 / 644                | -0.0193 / 7 / 684     | 4.92e-11 / -0.0643 / 0.9938 / 4201
bulk     0.01    200 | 0.01820   | -0.0087 / 4 / 647                | -0.0183 / 7 / 712     | 6.16e-11 / -0.0440 / 0.9961 / 4254
bulk     0.04     50 | 0.01902   | -0.0149 / 4 / 640                | -0.0278 / 7 / 674     | 4.94e-11 / -0.0935 / 0.9924 / 4056
bulk     0.04    100 | 0.01902   | -0.0101 / 4 / 643                | -0.0201 / 7 / 682     | 4.29e-11 / -0.0636 / 0.9953 / 4085
bulk     0.04    200 | 0.01902   | -0.0067 / 4 / 647                | -0.0283 / 7 / 707     | 4.81e-11 / -0.0439 / 0.9971 / 4129
bulk     0.01      0 | 0.01819   | -0.1923 / 4 / 640                | FAIL                  | 6.50e-11 / -2.6201 / 0.5489 / 4116
```

C_D moves the shocked excursion by a tenth over a sixteenfold range and the advected width
by 6%; C_Y is the active constant and trades the excursion for steps. The two failed
advection rows are the mass-fraction bound and not the channel: the two channels are
bit-identical on that equal-weight case, and at C_Y = 200 both lose positivity on the first
step, since a bound that switches on inside a step is not in the step size set before it.
Under `:delta4` these rows kept C_D = 0.01 and C_Y = 100 for the channel; the channels at
the current detector and C_D are compared in [the partial-density species
channel](#the-partial-density-species-channel). The three-dimensional measurements are in
[their own section](transport.md#the-bulk-species-channel-in-three-dimensions). Open beside them: the
ratio-1000 failures, which are the transmitted shock's foot on the shock case and
the seven-cell density jump on the slab, neither a species-channel failure; the unequal-γ
drift is answered there.

### The partial-density species channel

The partial-density channel, `species_flux = :partial_density`, is the form of eqs. 38–40
of Brill, Olson & Bokman. The partial densities diffuse with the shared D_b of the bulk
channel, and their mass flux enters momentum as (ΣJ)u and energy as (ΣJ)|u|²/2 + Σ e_k J_k
(`reference/DESIGN.md`, "The species channel"). It is the default. The one-dimensional
battery rows of the previous section were measured under all three channels across C_D, at
the default detector and the solver's filter strength (scratch script, no bench entry;
`bench/artcal.jl bulk alphaf=0.47` gives the default-constant row of each). The advected
width is the same under every channel, since its two gases share one molecular weight, and
is in the table of [C_D](#c_d-the-species-diffusivity):

```
V 5–95 is the volume-fraction width of `volume_width`, in cells
channel          C_D  | SI 5.04: TV−1 / worst Y / width / steps | SI 100: TV−1 / worst Y / V 5–95 / steps | slab 100: max|p−1| / worst Y / steps
fickian          0.3  | 0.0270 / -0.0083 / 4 / 649             | FAIL                                    | 4.95e-02 / -0.0860 / 4096
fickian          1    | 0.0068 / -0.0065 / 5 / 646             | FAIL                                    | 6.30e-02 / -0.0743 / 4070
fickian          3    | 0.0065 / -0.0040 / 8 / 646             | FAIL                                    | 9.02e-02 / -0.0482 / 4056
bulk             0.3  | 0.0160 / -0.0069 / 4 / 643             | 0.0232 / -0.0121 / 4.35 / 685           | 5.32e-11 / -0.0692 / 4042
bulk             1    | 0.0043 / -0.0049 / 5 / 638             | 0.0027 / -0.0090 / 4.78 / 682           | 5.05e-11 / -0.0547 / 4016
bulk             3    | 0.0033 / -0.0032 / 7 / 636             | 0.0017 / -0.0058 / 5.49 / 681           | 4.08e-11 / -0.0443 / 4000
partial_density  0.3  | 0.0136 / -0.0075 / 4 / 645             | 0.0881 / -0.0165 / 4.69 / 700           | 5.88e-11 / -0.0692 / 4042
partial_density  1    | 0.0045 / -0.0050 / 5 / 642             | 0.0021 / -0.0093 / 5.12 / 701           | 5.18e-11 / -0.0547 / 4016
partial_density  3    | 0.0040 / -0.0036 / 8 / 642             | 0.0010 / -0.0056 / 5.83 / 713           | 4.60e-11 / -0.0443 / 4000
```

The Fickian pressure error on the slab grows with C_D, by under twofold over the tenfold
range, and the two consistent channels hold it at round-off at every C_D. On the ratio-5.04
shock the consistent channels agree, and at the default constant the Fickian one rings half
as much again. At ratio 100 the Fickian channel fails at every constant. The bulk channel
holds the ringing better only below the default: at 0.3 the partial-density channel
rings four times as much as the bulk one, and from the default up it rings less, at the same
worst mass fraction and a volume-fraction width about 7% wider. The bulk channel's ringing at
ratio 100 is sensitive to the filter: at the script's 0.45 it is seven times its value here,
while the partial-density channel's is unchanged ([the species-only
split](#the-species-only-split) holds those rows).

On an advected He/CO2 slab either consistent channel holds the pressure at round-off. The
Fickian error depends weakly on the constant: it less than doubles from 0.1 to 3. The
mass-fraction bound accounts for most of it at the smallest constant and almost none from
the default up, and from the default up the Brill slab's Fickian error is the same without
the bound:

```
He/CO2 slab: CO2 over half of a periodic 1 m line of helium, tanh edges over 3h, N = 256,
u = 100 m/s, uniform p = 1 atm and T = 300 K, two periods (scratch script, no bench entry);
the last column is the Brill slab of the table above under the same constants
channel           C_D   C_Y   thermodynamics        max |δp/p|   | Brill slab max |p − 1|
fickian           0.1   100   NASA-9                3.15e-5      | 3.33e-2
fickian           1     100   NASA-9                3.46e-5      | 6.30e-2
fickian           3     100   NASA-9                5.71e-5      | 9.02e-2
fickian           0.1   0     NASA-9                1.19e-5      | lost positivity
fickian           1     0     NASA-9                3.35e-5      | 5.88e-2
fickian           3     0     NASA-9                5.71e-5      | 9.00e-2
fickian           1     100   calorically perfect   3.76e-5      |
bulk              1     100   NASA-9                6.60e-13     |
partial_density   1     100   NASA-9                6.58e-13     |
partial_density   1     100   calorically perfect   6.53e-13     |
```

On the one-dimensional He/CO2 tube the consistent channels ring alike at the default
constant and the Fickian one half as much again (the channel rows of the table of
[C_D](#c_d-the-species-diffusivity)).

On the two-dimensional tube the consistent channels cut the grid-scale pressure behind the
contact six- to fifteenfold against the Fickian channel, and the bulk channel's roll-up is
the weakest of the three:

```
2-D deck of bench/archive/he_co2_shock_tube.jl, 768 × 48 to 2.5 ms at C_D = 1 (scratch script, same
setup). Grid-scale p is max |δ⁴_x p|/16p over pure helium (Y_CO2 < 1e-3) within 0.25 m
behind the contact on each row, six cells and more from it. The helium dip is 1 − Y_CO2 at
its minimum on the line y = L/2 beyond the first point where Y_CO2 exceeds 0.9, and max |v|
the largest transverse velocity, both at 2.5 ms.
channel          | steps | grid-scale p at 1.5 / 2.0 / 2.5 ms | worst Y_CO2 | helium dip | max |v|
fickian          | 3427  | 1.0e-3 / 9.8e-4 / 9.3e-4           | -1.71e-3    | 13.3%      | 152.6 m/s
bulk             | 3233  | 6.5e-5 / 9.4e-5 / 1.1e-4           | -1.11e-3    | 4.5%       | 99.5 m/s
partial_density  | 3439  | 1.0e-4 / 1.7e-4 / 1.5e-4           | -1.22e-3    | 14.6%      | 163.8 m/s
```

The bulk channel entrains a third as much helium into the CO2 head and carries two thirds of
the transverse velocity, as its added viscosity ρD_b∇u would; the partial-density channel
reproduces the Fickian roll-up on both measures to within a tenth. Which roll-up is right
is open, since no converged reference exists for it and no published comparison of RM
mixing between these forms was found.

The consistent channels add the conserved gradients and a second sensed field, and two
reductions keep the partial-density right-hand side at the Fickian one's. Under
`ConstantTransport(mu0 = 0)` the shared-D_b channels skip `grad_Y`, whose only other reader
is `NSCBCInflowBC`, which computes it itself. With two species only the first is sensed,
since the second's detector outputs and excursions equal the first's to round-off. On the
two-species tube of `bench/phases.jl` the Fickian channel's mass-fraction gradients and the
partial-density gradients that take their place cost the same, and the partial-density
right-hand side came out 4% below the Fickian one in both processes:

```text
julia --project=. -t 16 bench/phases.jl species_flux=<channel>   # twice per channel, alternating
```

```
two-species tube, 512 × 32, per right-hand side (minimum of 20), one figure per process;
scalar gradients are those of T_ion and, under :fickian, of the mass fractions
channel           right-hand side    artificial         scalar gradients   channel gradients   derivative line solves
fickian           1.425, 1.442 ms    0.514, 0.637 ms    0.146, 0.173 ms    –                   24
partial_density   1.373, 1.381 ms    0.658, 0.538 ms    0.057, 0.048 ms    0.112, 0.126 ms     24
bulk              1.618, 1.654 ms    0.664, 0.619 ms    0.058, 0.053 ms    0.443, 0.435 ms     32
```

The bulk channel costs about 14% more per right-hand side, in its n_cons conserved
gradients.

In the literature, Brill, Olson & Bokman and Aslani & Regele (Int. J. Numer. Meth. Fluids
88, 2018) both reject the Fickian flux with the enthalpy term on this pressure argument and
diffuse the partial densities with consistency terms; PadeOps (Lele group, `cgrid.F90`)
uses the Fickian form with the enthalpy flux, and Pyranda's example decks carry no species
term in the energy equation at all.

**Pyranda's implementation.** Read at [LLNL/pyranda](https://github.com/LLNL/pyranda)
`b4e0afc`, the upstream master of 24 June 2026; no upstream branch computes a sharpening
flux, a volume fraction or a partial-density flux, and the Miranda reader reads Miranda's
volume fractions and a single `diffusivity` field (`pyrandaMirandaReader.py:24–46`). Source
paths are under `pyranda/`, deck paths under `examples/`.

- *Detector.* `ring(f)` is `ringS(f, 2)` and `ringV(a, b, c)` is `ringV(·, 1)`
  (`parcop/parcop.f90:313–333`), giving MAX_d h_d^L |d8_d f|, maximized over the three
  components for `ringV` (`parcop/operators.f90:615–699`). `d8x` is undivided, with the
  division by h⁸ commented out (`parcop/compact_operators.f90:280–313`), and the weights
  are `c10d8` unscaled (`parcop/stencils.f90:515–621`; `parcop/compact_basetype.f90:65–209`
  applies no factor): the interior row of [the ringing detector](#the-ringing-detector),
  with closure rows folded onto the half-offset mirror. A grid-to-grid oscillation of
  amplitude A returns

  ```
  c10d8: 60·256A/4 = 3840A        compact_d8: 16A
  ```

  so the ratio of 240 holds at every wavenumber, since the two are the same operator. The
  low-wavenumber gain is (kh)⁸, that of Brill's F (eq. 22), so the conversion holds for any
  consistent eighth-derivative operator Miranda might use, below the grid scale.
- *Fields, per deck.* β\* from `ring(∇·u)` in every deck, ungated; μ\* from
  `abs(ring(|S|))` (weight h²) in most decks and from `ringV(u, v, w)` (weight h) only in
  `equation_library.py:66,106` and `naca_curv.py:106`; κ\* from
  `ring(T)`; the species term from `ring(Y)` of one species (`shockBubble.py:89–108`,
  `RT3D.py:133–150`).
- *Formulas.*

  ```
  β*  = C_β G[ρ ring(∇·u)]                                          C_β = 0.07
  μ*  = C_μ ρ G[|ring(|S|)|]   (or G[ρ ringV(u)])                    C_μ = 1e-3 or 1e-4
  κ*  = C_κ G[ρ c_v ring(T)/(T Δt)]                                  C_κ = 1e-3
  ρD* = G[ρ max(C_D ring(Y), C_Y (|Y| − 1 + |1 − Y|) h_min²)/Δt]     C_D = 1e-4 to 2e-4,
                                                                     C_Y = 10 or 100
  Δt  = 1/max(Σ_d |u_d|/h_d + c/h_min), lowered by 0.2 of the diffusive limits
  ```

  Δt is the variable `:dt:`, the stability limit at CFL 1 from `dt.courant`, with the
  diffusive limits applied (`pyrandaTimestep.py:42–75`), so h²/Δt is a global rate times h
  and not the local sound speed. μ\* and β\* contain no Δt. G is `gbar`, the nine-point
  Gaussian `cgfs4` (`parcop/stencils.f90:1387–1476`), applied to the coefficient after the
  product with ρ.
  One D\* serves both species, and h_min is `gridLen`, min_d h_d (`parcop/mesh.f90:211`).
- *Species flux.* Cook's Fickian form for two species with no enthalpy flux: J = ρD\*∇Y_A,
  subtracted from ρY_A and added to ρY_B, with no term in momentum or energy
  (`shockBubble.py:65–69,106–108`). Brill's partial-density flux (eqs. 32–40), the
  maximum over Y and V (eqs. 33–37) and the sharpening flux (eq. 68) are absent, as is any
  positivity treatment beyond the C_Y term.
- *Filter.* `fbar` is `sfilter`, spec `sfspec = 2` (`parcop/compact.f90:28`), `c8ff8`, a
  pentadiagonal eighth-order filter with α = 0.66624 and β = 0.16688
  (`parcop/stencils.f90:713–835`). Its transfer function integrates to nine tenths, which
  matches Brill's eighth-order filter removing the top 10% of wavenumbers. Its closure rows
  telescope: identity at the boundary node, then three-, five- and seven-point symmetric
  rows (`:763–786`). The decks put `fbar` in the equation block, so it runs after every
  Runge–Kutta stage (`pyranda.py:800–810`, `updateVars` at 396).

Pyranda therefore carries the traditional method of Brill's section 3 (eqs. 16–27, with
Pyranda's Fickian flux in place of eq. 16's enthalpy flux) and none of section 4. The source
settles the detector's normalization, the sensed fields, the Gaussian, the filter and its
cadence, and the meaning of Δt in Pyranda. Which Δt Miranda uses in eq. 24, how Miranda
discretizes eqs. 32–40 and 68, and how eq. 27 reads are not in it.

**Recommendation:** `:partial_density` as the default, at C_D = 1. It holds a uniform
(u, p, T) state to round-off, rings no more than `:bulk` on the one-dimensional rows at the
default constant, keeps the Fickian roll-up and costs no more per step than `:fickian`.
`:bulk` where an added viscosity and conduction at the interface are wanted, at about a
seventh more per right-hand side. `:fickian` only to reproduce Cook's form.

### The CFL rate convention

The [public Pyranda kernels](https://github.com/LLNL/pyranda/tree/master/pyranda/parcop)
implement the same Cook method, and reading them against `artificial.jl` identifies five
differences that bear on the constants. Four are measured in the subsections above (the
detector, the sensor fields, the smoother) and in [filtering on non-uniform
volumes](filter.md#filtering-on-non-uniform-volumes), which also records that Pyranda's conservative
filter normalizes by a filtered cell volume, available behind `filter_weighting = :volume`
and not the default. The fifth is the CFL rate convention.

Pyranda forms `Σ_d |u_d|/Δ_d` for advection and adds `|c|/min_d(Δ_d)` once for the acoustic
part, then takes the diffusive limits as separate minima. `max_rate` had summed `(|u_d| +
c)/h_d` over active dimensions and folded the diffusive rate into the same sum, counting the
sound speed once per active dimension. Neither convention is the linear bound: the acoustic
operator's eigenvalue at wavevector k is i c |k′| with k′ the modified-wavenumber vector, so
on an isotropic grid the three-dimensional limit on dt is 1/√3 of the one-dimensional limit
at the same cell. `max_rate` now takes the acoustic part as the Euclidean bound,
`Σ_d |u_d|/h_d + c · sqrt(Σ_d 1/h_d²)` plus the diffusive term, Pyranda's structure with
the Euclidean norm in place of the minimum spacing. In one dimension it is the previous rate
exactly, so the battery and every one-dimensional guard are bit-identical; on an isotropic
three-dimensional grid the acoustic part is √3 smaller, and a nominal CFL quoted for such a
run before the change reads as `cfl_old / √3` after it. The Taylor–Green fits therefore read
`cfl = 0.35` throughout this file.

The five-stage Carpenter–Kennedy scheme is stable on the imaginary axis to 3.34 and the C6
modified wavenumber peaks at 1.99, which predicts a one-dimensional acoustic ceiling of
nominal `cfl = 1.68`, times one plus the Mach number for the advective part. On
Taylor–Green the measured ceiling sits where that bound puts it:

```
Taylor–Green, 32³, Re = 1600, to t = 10
                 art off (acoustic + molecular)        art on
cfl    steps   peak -dKE/dt   KE misfit  rate misfit   steps   KE misfit  rate misfit
0.35   2700    1.4205e-2      1.323e-1   7.025e-1      3401    1.474e-1   7.499e-1
0.7    1353    1.3302e-2      1.200e-1   6.685e-1      1794    1.394e-1   7.514e-1
1.05    902    1.2982e-2      1.130e-1   6.374e-1      1243    1.353e-1   7.261e-1
1.4     677    1.2759e-2      1.082e-1   6.153e-1       958    1.324e-1   7.113e-1
1.6     593    1.2646e-2      1.061e-1   6.044e-1       848    1.310e-1   7.061e-1
1.75    542    1.2601e-2      1.046e-1   5.978e-1       781    1.300e-1   7.026e-1
1.9     510    unstable                                 724    1.291e-1   6.996e-1
2.1     506    unstable                                 669    1.269e-1   6.964e-1
```

The acoustic ceiling lies between nominal 1.75 and 1.9. The bound rises from its zero-Mach
value above to 1.85 at Ma 0.1. An unstable row is one whose step count stops falling with
the CFL and whose energy budget breaks; neither raises `SolverFailure` within t = 10. With the
artificial properties on the diffusive rate adds to the denominator, so nominal 2.1 is the
art-off step at about 1.7 and completes. Every estimator improves monotonically up to the
edge in both arms: under `filter_cfl = 0` each step is one filter pass, so a longer step is
less dissipation per unit time.

## The CFL restriction and the symmetry cell

```text
julia --project=. -t 1 bench/archive/nohprobe.jl 1 cfl=0.15 nmax=5000 every=2000 floor=1e-8 scope=representable
```

No setting of the four constants stabilizes a converging strong shock at the default `cfl =
0.5`. The restriction is a symmetry-plane startup problem: the wall, axis or origin cell,
not the shock front. The planar and cylindrical ceilings were the unprimed first step and
are gone; the spherical origin's stands at 0.5.

### The first step of a run

`max_rate` builds its diffusive rate from the artificial coefficient arrays as the last
right-hand-side evaluation left them, and a fresh solver has none, so an unprimed first step
is sized on the acoustic and advective rates alone. Planar and cylindrical Noh start with
u = −1 against the wall or the axis, where the strain sensor is largest. `run!` primes that
step by evaluating the right-hand side of the initial state once beforehand. Priming
removes the planar and cylindrical ceilings and leaves the spherical one. The highest CFL
reaching the final time with a correct plateau, under the defaults:

```
N = 400 / 256 / 256, the ν = 3 warm start, to t = 0.6
                    nu = 1 wall          nu = 2 axis          nu = 3 origin
unprimed            0.25 (0.3 wrong)     0.2  (0.25 fails)    0.5 (0.6 fails @71)
primed              none to 0.9          none to 0.9          0.5 (0.6 fails @71)
primed, first dt    1.04e-3 at 0.9       1.77e-3 at 0.9       unchanged
```

Primed, the plateaus hold across the ladder:

```
primed          cfl            plateau/exact     wall deficit
nu = 1 wall     0.25 to 0.9    0.9989–0.9990     63% → 57%
nu = 2 axis                    0.9375–0.9381
```

Unprimed, the wall run marked wrong completes with a cold density spike of 4.9 times the
plateau on the wall cell. The failing axis run is over-dense and cold from step 25 and fails
at step 110. Under `detector = :d8` the wall and
the axis complete through cfl 1.0 even unprimed.

The origin does not move, because its excursion lands well after the first step, at
t ≈ 0.39 from the warm start at t₀ = 0.3. Every CFL above the ceiling fails in the
excursion, primed or not:

```
ν = 3, primed or not    cfl 0.6   0.7   0.8   0.9
fails at step               71    60    52    47
0.7 to 0.9 each fail at t = 0.386 to 0.387
```

In the area form of the radial pressure term ([grid-scale
growth](walls.md#grid-scale-growth-at-the-r-z-axis)) the ceiling is 0.3, and the two explanations
below were measured and rejected in that form. **The density proportionality of β\***:
rebuilt as `C_beta · ρ̃ · sensor` with ρ̃ the Gaussian-smoothed density, β\* rises at the
origin during the excursion and no ceiling moves:

```
area form                                   β* on ρ              β* on ρ̃
origin β*/line max, cfl 0.4, step 100       0.43                 0.61
wall, cfl 0.3                               completes wrong      fails at step 659
axis, cfl 0.25                              fails at step 110    fails at step 386
origin, cfl 0.4                             fails at step 106    fails at step 109
```

**The per-step filter strength**: the ladders read the same under `filter_cfl = 0`, 0.35
and 0.7 in all three geometries.

### Where the restriction originates

Five explanations were measured wrong. Do not reopen them without new evidence.

**Not the timestep predictor.** Linear extrapolation with `StepControl(predict = n)` and a
cap on the step's growth only delay the failure:

```
Noh ν = 1, N = 400, cfl = 0.3     fails at step
no lookahead                      175
predict = 3                       179
predict = 30                      200
max_growth = 1.05                 186
```

In the per-step trace density falls for 150 steps while `dt` and the rate stay nearly
constant, and the diffusive rate climbs to 1.5e4 and `dt` collapses only after positivity is
lost:

```
step    dt        rate    ρ_min
25      2.1e-4    1418    0.951
125     1.8e-4    1718    0.398
```

The cause is spatial, not temporal.

**Not insufficient β\* reach, and not insufficient β\* magnitude.** Over a complete ν = 1
run at `cfl = 0.15`, the furthest cell ahead of the front carrying above a thousandth of the
domain maximum β\*, beside the worst-affected cell:

```
step    x_sh/h | rho_min   i | e/e0_min   i  n_e<0 | b*@e/b*max  reach/h
 500      9.81 | 0.98875  17 |   -469.4  12      7 |      0.166    14.19
1500     27.63 | 0.99008  35 |   -421.7  30      8 |      0.192    14.37
2500     45.47 | 0.98851  53 |   -465.4  48      7 |      0.161    14.53
3500     63.30 | 0.98720  71 |   -498.7  66      7 |      0.140    14.70

the failing cfl = 0.3, first 125 steps:     b*@e/b*max 0.84–1.00
```

Reach holds between 14.2 and 14.7 cells over the run. The worst cell sits only 3–5 cells
ahead of the front, well inside that reach, so a wider sensor stencil would address a
deficit that is not present, and the affected cell carries a sixth of the domain's peak
artificial bulk viscosity throughout. At the failing CFL it carries nearly the full maximum
from the start (the table's last line).

**Not the fold closure**, which is sixth to seventh order and the most accurate region of
the line ([fold order](walls.md#fold-order-and-geometry-limits)). **Not sensor blindness at the
fold**: during the excursion that fails, β\* at the origin reaches the line maximum under
both detectors.

The failure starts at the symmetry plane. In the planar case the wall cell degrades first,
and the density hole then opens between the wall and the front while the pre-shock field is
still undisturbed:

```
ν = 1, cfl = 0.3
step 5      the wall cell, i = 1, has negative internal energy
step 125    density hole at i = 3, front at cell 4.4, pre-shock field within 1% of unity
            from cell 11 outward
```

The spherical origin fails the same way, more abruptly. One step before the failure the
line's density is still above the ambient, while the origin cell flows outward against the
inflow with a large positive internal energy beside a negative one, under a β\* far above
the front's:

```
ν = 3, one step before the failure    origin cell    neighbour    front
density minimum over the line         1.92
u                                     +5.6, against an inflow of −1
e/e₀                                  +2.6e5         −1.6e4
β*                                    5.7                         0.06
```

`:gated_strain` at ν = 2, `cfl = 0.2`, the one setting that moves a ceiling, migrates the
worst-energy cell off the axis and reduces its magnitude about sixtyfold.

### Sensor intermittency at the damage site

Sampling the ν = 3 origin at 25-step intervals, where both smoothers complete at `cfl =
0.15`, β\* at the damaged cell over the line maximum:

```
step        150    175    200    225    250    275    300    325    350    375    400
compact    0.393  0.004  0.302  0.007  0.305  0.009  0.300  0.011  0.306  0.007  0.213
gaussian   0.103  0.336  0.118  0.429  0.415  0.120  0.143  0.131  0.108  0.375  0.130

under both: reach/h 6–11 cells, n_e<0 7–9
```

Under `:compact` the cell being damaged carries about 30% of the domain maximum on one
sample and under 1% on the next. Under `:gaussian` it never falls below 0.100. Reach and the
count of negative-energy cells are the same under both smoothers (the table's last line), so
reach is not the discriminating quantity. The same signature precedes the failure at
`cfl = 0.3`.

Sensor roughness, not β\* magnitude, discriminates, and the restriction at the fold is set
by the continuity of β\*. The undivided δ⁴ applied to |S| produces a spiky field, and the
damaged cell, drifting outward over the run, lands alternately on spikes and in troughs
under a smoother close to the identity across the resolved band, while a nine-point Gaussian
spreads each spike widely enough that no trough remains. A measurement of the detector must
therefore be taken on top of the Gaussian smoother, since a sharper high-pass produces
narrower spikes; Pyranda pairs `:d8` with the Gaussian and never with a near-identity
filter. The planar probe above is sampled too coarsely for an alternation of this period,
so its steady 0.14–0.19 is consistent with either a steady sensor or the average of an
alternation; re-running it at `every = 25` would settle that and has not been done.

### The origin cell is a startup transient

`bench/archive/nohprobe.jl` reports the symmetry cell on every line, since the argmin columns track
the front and hide it. Spherical Noh, N = 256, `rho1/rho2` is that cell over its neighbour:

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

The table was taken with the radial pressure term in the area form, where the ceiling was
0.3. Under ∂p/∂r the same excursion ends the run at cfl 0.6 under the default detector:

```
∂p/∂r, default detector, cfl 0.6, fails at step 71
step   rho1/rho2   b*1/max
 60     0.9788      0.230
 65     1.0509      0.901
 70     0.5764      0.662
```

The symmetry cell is quiescent for most of the run, so whatever sets the ceiling does not
act gradually from the start; during the excursion β\* at the origin reaches the domain
maximum in both surviving configurations. Every configuration has the excursion, and the
ceiling is whether the cell survives it. The excursion is physical, not grid-scale. Under
refinement the peak of `b*1/max` lands at t ≈ 0.394 on every grid, so the step number scales
with N while the time does not, a resolved feature of the warm start at t₀ = 0.3:

```
:delta4, cfl 0.3    N = 128   256   512
step of the peak         58   160   342
the peak weakens from 1.000 to 0.950 under refinement
```

At the moment of failure the regularization is suppressed by the evacuation itself. Since
β\* is proportional to density by construction (`src/physics/artificial.jl`), it falls with the
symmetry cell's density over the last rows of the `:d8` column above, and the profile at
the failing step shows the evacuated origin with its velocity reversed:

```
:d8, cfl 0.3, the failing step    origin   next cell   the one after
ρ                                 38.1     136.7       162.1
u                                 +1.19
```

The mechanism is consistent with the numbers but not demonstrated: a β\* that does not
vanish with the density was run and the ceiling did not move.

### Negative internal energy in completed runs

The `n_e<0` column is nonzero in every sampled configuration, completed runs included: up
to eleven interior cells carry negative internal energy, travelling with the front, for the
whole ν = 1 validation case, and ten remain at its end. No diagnostic reports it, since
`primitives!` floors T_ion at 1e-300 wherever e ≤ 0 and the positivity check in `max_rate`
reads ρ, which stays positive. The quantity is ill-conditioned here: at the Noh ambient the
internal energy is 1.5e-4 against a kinetic energy of 0.5. So e is a difference of terms
agreeing to within 0.03%, and a rounding-level error in either flips its sign. The κ\*
sensor is built on e, so this also bounds the artificial conductivity.

The affected cells are not a rounding step below zero. Over the complete planar validation
case many cells have e < 0, and none has a non-positive total energy or density. They are
not a wall layer either: at step 500 they lie ahead of the front as an odd-even oscillation
of the internal energy, with the wall cell hot:

```
ν = 1 validation case: 3756 steps at cfl 0.15, plateau 3.9971
cell-steps with e < 0            25179
with E ≤ 0 or ρ ≤ 0              none

step 500: front at cell 11.8, wall cell at 9200 e₀; the cells with e < 0 lie at cells 14 to 28
e/e₀ cell by cell    −297, +323, −176, +57, +9, −30, +33, −22, +15, −6, +3, +1, −0.3
```

The oscillation is the compact scheme's precursor, and the calculation still reaches the
plateau to within 0.07%. The same precursor on Woodward–Colella and Sedov, its scaling with
the jump and the repairs measured against it are under [negative internal energy ahead of a
shock](#negative-internal-energy-ahead-of-a-shock).

**Repairing those cells is a percent-level intervention and terminates the run.** Under the
failsafe that repairs negative internal energy, the run gains mass and energy and loses
momentum until it fails. The planar case at cfl 0.15, measured at commit `7a24584`, before
`run!` primed the first step:

```
StepControl(floor_ratio = 1e-8, floor_scope = :internal_energy)
cells repaired on step 1      five
cell-steps repaired           256
mass added                    1.4007
energy added                  9.3e29
momentum removed              1.7e10
end                           :dt_collapse at step 19
```

The repair damps the velocity where the total energy is still positive and raises the total
energy where no kinetic energy is left to convert, so the cost belongs to the case and not
to the choice of repair. The three geometries at cfl 0.15 under each policy, with the first
step primed (commit `c407e0b`):

```
nu  policy                  end                 steps   plateau   deficit  shock    cells e<0 (max, span)  low-e cell-steps  repairs   mass added  energy added  momentum removed
1   permissive              completed           3650    3.9959    +63%     0.2046   8, 15                  0                 0         0           0             0
1   representable           completed           3650    3.9959    +63%     0.2046   8, 15                  24250             0         0           0             0
1   internal_energy         dt_collapse @15       15    -         -        -        -                      289               289       1.0e1       2.8e35        6.8e11
1   validity = :repair      invalid_state @end  3650    -         -        -        8, 15                  24257             0         0           0             0
2   permissive              completed           2245    15.002    +55%     0.2092   7, 13                  0                 0         0           0             0
2   representable           completed           2245    15.002    +55%     0.2092   7, 13                  13032             0         0           0             0
2   internal_energy         dt_collapse @17       17    -         -        -        -                      293               296       2.7e-3      6.3e29        3.3e8
3   permissive              completed           1113    62.501    +28%     0.2090   9, 17                  0                 0         0           0             0
3   representable           completed           1113    62.501    +28%     0.2090   9, 17                  7756              0         0           0             0
3   internal_energy         dt_collapse @102     102    -         -        -        -                      360               360       1.6e-3      6.1e16        7.2e7
```

The default `floor_scope = :representable` repairs only what no frame can represent and
counts the rest, reproducing the permissive trajectory exactly in every geometry; `validity
= :repair` runs that trajectory and then rejects the state it ends on, since the
representable repair leaves the negative internal energy in place. The closing states carry
inadmissible cells in every geometry:

```
closing state         nu = 1    nu = 2    nu = 3
inadmissible cells    7         6         8
e_min                 −0.038    −0.018    −0.020
```

[State validation](../DESIGN.md#state-validity-and-its-policy) puts admissibility to the EOS,
and a calorically perfect gas answers that a cell with e < 0 is outside its domain. That
verdict is correct over 25179 cell-steps of a run that reaches the right answer, so a strict
check on the returned state rejects a completed and correct Noh run. The Noh cases therefore
run under `validity = :permissive`, guarded on the state they end with as well as on their
solution error. The mass-fraction side of the same check has its own
[band](transport.md#the-species-validity-band).

### Recovery strategy

Rollback retains a larger CFL after the excursion because the restriction is confined to it.
`StepControl(retries = 4)` from an initial `cfl = 0.9` under the current defaults, with the
first step primed and the coefficient arrays banked beside the savepoint:

```
nu   start cfl   recovered cfl   retries   steps   plateau/exact
1    0.9         0.9             0          582    0.9990
2    0.9         0.9             0          364    0.9381
3    0.9         0.45            1          333    0.9779
3    0.15 fixed  –               –         1080    0.9778
3    0.5         0.5             0          323    0.9777
```

The planar and cylindrical cases need no retry. The spherical case fails in the excursion,
rolls back and completes at the halved CFL with the plateau of the fixed low-CFL run in a
third of its steps, and `solver.cfl` records the accepted value. The coefficient arrays have
to be banked with the state: a restore leaving the failed trajectory's coefficients in place
throttled each retry's first step, and the ν = 3 case then ended in `:no_progress` sized by
coefficients of order 1e57.

**Recommendation:** the default `cfl = 0.5` with `StepControl(retries = 4)` for automatic
recovery; the warm-started origin completes at 0.5 without a retry. Under `detector = :d8`
the origin's ceiling is 0.4, and the default CFL fails at step 78.

### The singular start and the warm start

The spherical case is warm-started at t₀ = 0.3 from the exact solution with a tanh blend of
width 4h at the shock. Primed, the singular t = 0 start completes only at a low CFL, as it
does in the area form of the radial pressure term, and the retry ladder recovers it:

```
singular t = 0 start, primed    result
cfl 0.15                        completes
cfl 0.2                         fails at step 5
cfl 0.3 and 0.5                 fail at step 2
retries = 4 from 0.5            completes at 0.125 in 2586 steps
area form, cfl 0.15             fails at step 36
area form, cfl 0.05             completes
```

Either way the plateau is 18% low and the wall deficit 77%. The start time and the blend at
N = 256, cfl 0.15:

```
nu  t0    blend   N      steps   plateau/exact   deficit   shock
3   0     4h      256    2155    0.8220          77%       0.2151
3   0.1   4h      256    1802    0.7597          10%       0.2181
3   0.1   4h      512    3668    0.8813          13%       0.2085
3   0.1   4h     1024    7401    0.9408          22%       0.2042
3   0.2   4h      256    1453    0.8783          20%       0.2175
3   0.3   2h      256    1102    0.9489          18%       0.2093
3   0.3   4h      256    1080    0.9778          25%       0.2089
3   0.3   8h      256    1040    0.9811          32%       0.2132
3   0.3   4h      512    2202    0.9916          27%       0.2039
3   0.3   4h     1024    4445    0.9963          32%       0.2017
2   0.1   4h      256    1882    0.8704           2%       0.2132
2   0.2   4h      256    1514    0.9749           6%       0.2127
2   0.3   4h      256    1114    1.0412           2%       0.2001
```

The plateau error is made while the shock is within a few cells of the origin. From
t₀ = 0.1, where the shock starts at 8.5 cells, the plateau converges under refinement at
first order. From t₀ = 0.3, where it starts at 26 cells, the error is an order of magnitude
smaller at every resolution. Halving the blend width from 4h lowers the plateau by 3%, and
doubling it raises the plateau by a third of a percent at the cost of the shock position.
The width of 4h is therefore retained. The battery starts the cylindrical case singular at
t = 0, which the axis takes, and its plateau of 0.938 is a startup error of the same kind.
The warm start at t₀ = 0.3 returns the shock position exact (the last row) and completes
through cfl 0.5 even unprimed.

## Negative internal energy ahead of a shock

```text
julia --project=. -t 1 bench/archive/shockfoot.jl
julia --project=. -t 1 bench/archive/shockfoot.jl part=defaults cases=sedov
```

The transcript of the first command is `bench/archive/results/shockfoot.txt`.

Woodward–Colella, planar Noh and Sedov from `test/cases.jl` carry points with
ρe = E − ½|m|²/ρ ≤ 0 under the defaults from early in each run:

```
case                     N     first step with ρe ≤ 0   steps
Woodward–Colella         800   9                        4380
planar Noh, cfl 0.15     400   1                        3512
Sedov                    256   89                       3146
```

The script applies the filter pass from a step callback and reads the state on both sides of
it. The trajectories are the production cases': under `part=identity` the final densities agree
with those of `woodward()`, `noh_case(1)` and `sedov()` exactly, and the closing state reports
agree for Noh and Sedov (`woodward()` returns none).

### Where they sit

Point-steps with ρe ≤ 0 summed over each run, before and after the filter pass; the most
after one pass; the points made bad by the step, made bad by a pass and cured by a pass;
the lowest ρe; and the distance of the bad points ahead of the nearest front, in cells:

```
case     bad before / after   max per step   step / pass / cured   min ρe before → after   ahead
WC       36657 / 37114        15             582 / 3552 / 3095     −26.6 → −31.1           1–17
Noh      30596 / 30308        11             1604 / 825 / 1113     −0.138 → −0.132         2–22
Sedov    20869 / 20981        12             75 / 955 / 843        −0.162 → −0.213         1–26
```

Nearly all the bad point-steps lie ahead of the nearest front, in the undisturbed gas, in an
odd-even pattern, and none of them is at a wall. The exceptions lie behind a front or in a
step with no front:

```
case     bad point-steps not ahead of a front
WC       30, all but one after the collision of the blasts
Noh      397, among them the first step's, which precede the front
Sedov    0
```

On Noh the first bad points appear at the first step, at the odd cells 3 to 7, before any filter
pass. On Woodward–Colella and Sedov the first appears after a pass, and over each run the passes
lower the minimum there, while on Noh they raise it slightly. A pass turns about as many points
bad as it cures.

The points are not a rounding effect. At Noh's median bad point K exceeds E by 0.35% of K. On
Woodward–Colella and Sedov 98% of the bad points have a negative total energy, which no
cancellation produces. Noh and Sedov end on a few inadmissible points ahead of the front, and
Woodward–Colella on none:

```
closing state   points   cells ahead of the front   front at            E     e_min
Noh             10       2 to 21                    x = 0.206 to 0.253  > 0   −0.0293
Sedov           4        2, 4, 7 and 10             r = 0.815 to 0.852  < 0   −0.0023
WC              0
```

### Scale

The undershoot does not scale with the ambient state. Woodward–Colella to the collision of the
blasts at three ambient pressures, and Noh and Sedov at their own:

```
case, ambient p   bad after the pass   min ρe after   depth     ambient ρe
WC, 0.01          37085                −31.1          −5.1%     2.3e-4
WC, 0.1           20668                −30.9          −4.8%     2.3e-3
WC, 1             4628                 −27.9          −2.5%     2.3e-3
Noh                                                   −2.3%     7.8e-5
Sedov                                                 −3.6%     1.4e-4
```

The depth is the lowest ρe of a bad point ahead of a front over ρe three cells behind that
front, and the last column is the ambient ρe over that same reference. Across a hundredfold
change of the ambient pressure the lowest ρe stays between −28 and −31 and the count of bad
points falls. Over the same range the ambient ρe rises from 0.025 to 2.5. The undershoot is 11
to 300 times the ambient internal energy in every case. A floor scaled to the ambient state is
therefore one to two orders of magnitude smaller than the deficit it repairs.

### Ablations and coefficients

No ablation and no coefficient removes the points. Bad point-steps after the pass and the
lowest ρe after it, Woodward–Colella to the collision of the blasts and Noh and Sedov to their
end times; "cap" marks a run stopped by the 4000-step cap at the time given, short of that end:

```
variant                Woodward–Colella             Noh                            Sedov
default                37085, −31.1                 30308, −0.132                  20981, −0.213
filter off             17931, −9.72                 fails @242, ρ < 0              fails @109, ρ < 0
artificial props off   64019, −277 (cap, t 0.0216)  fails @68, ρ < 0               26771, −5.10
C_beta = 2             30765, −12.3                 25514, −0.113                  18314, −0.0539
C_beta = 4             23514, −6.17                 20996, −0.0881 (cap, t 0.549)  14550, −0.0223
C_kappa = 0.1          35435, −17.4                 31226, −0.104 (cap, t 0.579)   19510, −0.210
ungated dilatation     37232, −33.7                 31083, −0.137                  21192, −0.344
```

Without the filter points still turn bad in the step, 2332 of them on Woodward–Colella, and
Noh and Sedov lose positivity. Without the artificial properties the minimum is nine times deeper
on Woodward–Colella and 24 times deeper on Sedov, and Noh loses positivity. Raising `C_beta`
from its default of one makes the lowest ρe shallower faster than 1/C_beta on Woodward–Colella
and Sedov. On Noh it is 15% and 33% shallower at the two larger values, the second over a run
stopped short. A tenfold `C_kappa` nearly halves the depth on Woodward–Colella and leaves
Sedov's unchanged. The ungated dilatation sensor deepens the minimum on all three cases. Both
constants are fitted to the battery's accuracy ([C_beta](#c_beta-the-shock-constant)).

### Repairs

No repair holds all three guards of `test/validation.jl`. Each repair runs to the case's end
time; the bad counts are point-steps after the pass (after the failsafe for the two floors), and
the energy is what the repair added, as a fraction of the case's initial total:

```
guards: L1 ρ below 6e-2 (WC), R_s within 3% (Sedov), plateau within 1% of 4 (Noh)
                           Woodward–Colella           Sedov                      Noh
repair                     bad     L1 ρ     energy    bad     R_s      energy    bad     plateau  energy
none                       37114   3.15e-2  –         20981   +1.04%   –         30308   3.9988   –
pass limited to ρe/2       7279    1.14e-1  +2.5%     2727    +4.16%   +17.3%    17695   3.9329   +1.7%
pass limited to ρe/100     10011   1.09e-1  +2.3%     3566    +4.04%   +16.4%    18142   3.9347   +1.7%
failsafe :representable    1179    1.13e-1  +2.3%     547     +2.95%   +13.3%    30308   3.9988   0
failsafe :internal_energy  0       9.06e-2  +2.4%     0       +1.41%   +13.5%    0       3.9055   0
```

The limits stop a filter pass from lowering a point's internal energy below a half, or a
hundredth, of its value before the pass; the failsafe rows run at `floor_ratio = 0.01`
under each `floor_scope`. On Noh the representable floor repairs none of the 30371 point-steps
below the floor, all with E > 0, so its row is the unrepaired run. The internal-energy floor
completes Noh at this ratio and fails at step 15 at `floor_ratio = 1e-8`
([above](#negative-internal-energy-in-completed-runs)). On Noh it adds no energy and removes
0.025 of momentum, converting kinetic energy into internal energy.

Every repair moves Woodward–Colella's L1 past its guard. The two limits also fail Sedov's radius
and Noh's plateau. Both floors pass Sedov, and the internal-energy floor fails Noh's plateau.
Where a repair changes the total energy it adds 1.7% to 17% of the case's initial energy, since a
floor or a limit raises the negative half of the oscillation about the ambient state and leaves
the positive half. The limits also change the mass, since a limit scales the whole correction of
the conserved state at a point:

```
mass change under the two limits   Woodward–Colella   Noh     Sedov
                                   +0.5% to +0.6%     +0.2%   −0.1%
```

A blend toward a first-order positivity-preserving flux (Hu, Adams & Shu 2013), for which the
compact divergence is written in face-flux form, is measured below.

### The face-flux limiter

```text
julia --project=. -t 1 bench/archive/positivity.jl
julia --project=. -t 1 bench/archive/positivity.jl part=variants cases=noh variants=none,A+B
julia --project=. -t 1 bench/archive/positivity.jl part=cost cases=woodward,noh,sedov,noh2,noh3 steps=1000
julia --project=. -t 1 bench/archive/positivity.jl part=cost cases=rz,blast steps=300 N=64
julia --project=. -t 1 bench/archive/positivity.jl part=patches patches=2,3
julia --project=. -t 1 bench/archive/positivity.jl part=cost cases=blast steps=300 N=80
julia --project=. -t 1 bench/archive/positivity.jl part=cost cases=blast steps=300 N=80 patches=2
julia --project=. -t 1 bench/archive/positivity.jl part=levels levels=wc,wcsub,wcregrid,noh85,noh43,noh85sub,noh43sub,nohuniform,a19,tiles,tilessub
julia --project=. -t 1 bench/archive/positivity.jl part=levelcost levels=wc,wcsub,noh85,tiles,a19
julia --project=. -t 1 bench/archive/positivity.jl part=levels levels=noh81,noh81tiles,noh81tilessub,noh81regrid,noh81regridsub
julia --project=. -t 1 bench/archive/positivity.jl part=levels levels=a19,a19tiles,a19tilessub,a19regrid,a19regridsub a19n=33
julia --project=. -t 1 bench/archive/positivity.jl part=tilecost levels=a19,a19tiles a19n=33 reps=3
julia --project=. -t 1 bench/archive/positivity.jl part=levels levels=a19regrid a19n=32
julia --project=. -t 1 bench/archive/positivity.jl part=patches patches=2,3 flux=ghost
julia --project=. -t 1 bench/archive/positivity.jl part=levels flux=ghost levels=wc,wcsub,wcregrid,noh85,noh43,noh85sub,noh43sub,tiles,tilessub,noh81,noh81tiles,noh81tilessub,noh81regrid,noh81regridsub
julia --project=. -t 1 bench/archive/positivity.jl part=levels flux=ghost levels=a19,a19tiles,a19tilessub,a19regrid,a19regridsub a19n=33
julia --project=. -t 1 bench/archive/positivity.jl part=levelcost flux=ghost levels=wc,wcsub,noh85,tiles,a19
julia --project=. -t 1 bench/archive/positivity.jl part=tilecost flux=ghost levels=a19,a19tiles a19n=33 reps=3
julia --project=. -t 1 bench/archive/positivity.jl part=tiledrift
julia --project=. -t 1 bench/archive/positivity.jl part=tiledrift ledger=true conservative=false
julia --project=. -t 1 bench/archive/positivity.jl part=tiledrift ledger=true drift=regridbox pulse= planar=
julia --project=. -t 1 bench/archive/positivity.jl part=levels conservative=false levels=wcregrid,tiles,tilessub,noh81tiles,noh81tilessub,noh81regrid,noh81regridsub
julia --project=. -t 1 bench/archive/positivity.jl part=levels conservative=false levels=a19tiles,a19tilessub,a19regrid,a19regridsub a19n=33
julia --project=. -t 1 bench/archive/filter_conservation.jl parts=shared N=48
```

On a closed line the collocated divergence is a difference of face fluxes under node
weights W with Σ W_i (D f)_i = f_N − f_1, so the face flux is a running sum of the
derivative, computed without a line solve. The weights solved from the solver's own `:neutral3` plan
depart from h only within a few nodes of a wall:

```
node from either wall   1        2        3        4        5        6        7        beyond 12
W/h                     0.2149   1.5887   0.6547   0.9793   1.0646   0.9977   1.0001   1 to 1.7e-13
```

The running sum of a stage increment meets the far wall's flux to 1e-14 with the artificial
fluxes included. The limiter blends each face of a Runge–Kutta stage's increment toward the
Lax–Friedrichs flux times the stage's time advance (c_{k+1} − c_k) Δt (variant A), and each face
of a filter pass's correction toward zero (variant B), with θ per face from the half states of
its two cells. Both act only at the faces they limit: a smooth acoustic pulse between slip walls
with A + B on limits no face in 474 steps and is identical to the unlimited run. ε is the bound
kept on ρ and ρe, the limiter's 1e-13 or a fraction of the initial minimum. Under the defaults,
one dimension:

```
case  variant          bad points    min ρe       metric                        steps  faces limited
                       pre / post    post pass                                         stage, pass
WC    none             36657/37114   −31.1        L1 3.1527e-2                  4380
WC    A                33170/35085   −10.5        L1 3.1472e-2                  4374   1.9%
WC    B                27573/24645   −11.4        L1 3.1547e-2                  4374   –, 1.7%
WC    A+B, 1e-13       0/0           1.0e-13      L1 3.1600e-2                  5727   0.43%, 0.68%
WC    A+B, 0.01 min    0/0           2.5e-4       L1 3.1519e-2                  4366   1.7%, 2.0%
WC    A-rhs+B, 0.01    179/175       −0.049       L1 3.1553e-2                  4364   1.6%, 2.0%
Noh   none             30596/30308   −0.132       3.9988, 24.1%, 0.2021         3512
Noh   A                14027/17150   −0.017       3.9992, 24.7%, 0.2021         4034   2.1%
Noh   B                negative density at step 435
Noh   A+B, 1e-13       0/0           1.0e-13      3.9992, 24.5%, 0.2021         5144   0.87%, 1.0%
Noh   A+B, 0.01 min    0/0           1.5e-6       3.9989, 24.6%, 0.2021         4757   2.0%, 2.7%
Noh   A+B, 0.1 min     0/0           1.5e-5       3.9990, 24.6%, 0.2021         4195   1.9%, 2.7%
Noh   A-rhs+B, 0.01    947/615       −4.5e-4      3.9970, 32.5%, 0.1998         4458   1.8%, 2.7%
```

A + B leaves no inadmissible point at any stage or either side of any pass, holds both
guards, and changes mass and energy by at most 1.8e-16 of the totals. Either half alone
leaves the base states the other half makes bad. `A-rhs` limits only the stage's own
right-hand side and leaves the low-storage history unlimited, the form that needs no face
register in several dimensions; it carries no guarantee, leaves bad points, and moves Noh's
wall deficit and shock. The extra steps at ε = 1e-13 come from cells held near ε, whose
artificial conductivity κ\* = ρc/T sizes more than a third of the steps. ε at a fraction of
the initial minimum removes those steps on Woodward–Colella and halves them on Noh:

```
A+B, ε = 1e-13                     WC     Noh
steps sized by κ* at a held cell   1989   2091
```

The first-order bound 2τα/W is largest at the wall node, where the weight is smallest:

```
first-order bound 2τα/W   wall node   inside
WC, cfl 0.3               0.938       0.31
Noh, cfl 0.15             0.33
```

At the default cfl 0.5 the bound therefore fails at a `:neutral3` wall node.

`part=src` runs the cases through `run!` with `Numerics(positivity_limiter = true)`, the
package's A + B. A cell's bound is 1% of the smaller of the initial minimum and its own base
value, and on ρe at least 16 eps |E| (`LIMITER_ENERGY_ULPS`). The minimum is taken over the
interior points where ρ and ρe are positive, so that a state entering `run!` with a point at or
below zero internal energy still sets a positive bound. Both cases end with no inadmissible
point after any step and no unguaranteed cell side, and both guards hold:

```
part=src, limiter on   steps   metric                                      faces limited, stage / pass
WC                     5512    L1 ρ 3.1550e-2                               0.829% / 1.38%
Noh                    5160    plateau 3.9988, deficit 24.4%, shock 0.2021  1.36% / 1.70%
```

The bench's `A+B` with the bound at 1% of the initial minimum alone takes fewer steps on both
cases: the cell bound lets cells settle below that minimum, where κ\* sizes the step.

`part=src cases=sedov,noh2,noh3` runs the radial cases of the validation battery through the
fold. The limiter leaves no inadmissible point on any of them, at most about a fifth more steps,
and moves the plateaus and the peak density by under half a percent:

```
limiter on (off)   steps         metric                                                bad unlimited
Sedov              3362 (3146)   R_s 0.8083 (+1.04%), peak ρ 5.162 (5.140)             20981
Noh ν = 2          2633 (2188)   plateau 15.054 (15.010), deficit 55%, shock 0.2087    14249
Noh ν = 3          1148 (1080)   plateau 62.708 (62.581), deficit 25%, shock 0.2086    8542

Sedov, relative change of the total   Σ W J ρ    Σ W J E
limiter on                            1.2e-13    1.5e-13
limiter off                           7.7e-14    1.5e-13
```

Sedov conserves mass and energy to round-off with the limiter on, since a limited filter pass
weights the components by J even at the origin. The cylindrical and spherical Noh leave no
unguaranteed side. Sedov leaves 2904, nearly all at a cell whose node terms alone would take it
below its bound: the hoop stress of β\* at the shock accelerates the cold gas past its internal
energy. Those cells take first-order fluxes at both faces. The face at the axis was limited at
7 of 13165 stages on the cylindrical Noh, and never on the spherical Noh or Sedov.

The Supernova remnant tutorial completes with the limiter on under strict validity, and its
inferred explosion energy moves by at most one unit in the third digit:

```
Supernova remnant tutorial   steps   energy at 5, 10 and 20 kyr, 10⁵¹ erg
limiter on                   2126    0.963, 0.975, 0.982
limiter off                  2009    0.963, 0.976, 0.982
```

`part=rz N=96` (and `N=128`) runs the same Sedov on the r-z quarter plane, folded at the
axis and by a symmetry plane at z = 0, against the spherical line at the same spacing:

```
                      steps   bad point-steps   R_s along r / z / diagonal   peak ρ
r-z 96², off          1737    732282            0.8176 / 0.8167 / 0.8192     4.358
r-z 96², on           2047    0                 0.8192 / 0.8174 / 0.8167     4.379
sphere 96, on         1351    0                 0.8177                        4.138
r-z 128², off         2269    1246063           0.8138 / 0.8132 / 0.8150     4.652
r-z 128², on          2873    0                 0.8158 / 0.8135 / 0.8136     4.668
sphere 128, on        1719    0                 0.8137                        4.587
```

The limited r-z radii agree with the sphere's to 0.25% along every direction. The face at the
axis is limited at under 2% of the axis line-stages and kept within the guarantee: that face
carries the scheme's own O(h²) flux, up to a twelfth of node 1's rate, and the radius along the
axis is the sphere's.

```
r-z, limited   axis line-stages limited    unguaranteed sides
96²            18705 of 982560 (1.9%)      183977
128²           26245 of 1838720 (1.4%)     307011

spherical line, N = 64, deposit three nodes wide   fails at step
unlimited                                          28
limited                                            43
```

The unguaranteed sides are, as on the line, cells whose node terms alone leave their bound: a
diagnostic run at N = 64 found every one of them there, none at the plane, the axis or a
first-order half state. The spherical line at N = 64 fails limited as unlimited, since the node
terms at the origin need the whole cell from step 15.

The limited step costs about 1.3 times the unlimited one on the planar line cases, 1.6 on the
radial lines and about twice on the planes. `part=cost` times a step with the limiter off and on
over the first steps of each case from fresh solvers, three off/on pairs alternated in one
process, and takes the medians of the times and of the pairs' ratios. The blast is a periodic
square with the Sedov deposit at its centre:

```
case, steps timed        off, ms/step   on, ms/step   on/off
Woodward–Colella, 1000   0.429          0.547         1.29
Noh, 1000                0.219          0.279         1.28
Sedov, 1000              0.178          0.285         1.60
Noh ν = 2, 1000          0.162          0.273         1.67
Noh ν = 3, 1000          0.177          0.284         1.61
r-z Sedov 64², 300       2.299          4.838         2.11
blast 64², 300           2.026          3.875         1.91
```

A θ pass indexing the fields by a node's linear index runs in two thirds of the time of
one indexing them by a Cartesian index, whose stride arithmetic each array repeats. On the
r-z plane the radial pass that forms each face's speed and each cell's rate with its node
terms is about two fifths of the limiter's time in a sampling profile, its largest part.

`part=patches` runs the two line cases on two and three same-level patches along x under
`patch_interfaces = :closure`. A patch's line closes at an interface with the `:cascade3` rows,
end weight 0.369 h against 0.215 h at a `:neutral3` wall, and the shared node is the end node of
both patches; the averaging after each stage makes it the mean of the two patches' updates. The
end face at the shared node is limited as well, toward the point flux of the node's state, by
one θ on both patches. Left unlimited, as at a wall, it let inadmissible points form at the
shared node, where the two patches' sensors give the artificial fluxes different values. At the
battery's resolution:

```
case   patches   steps off / on   bad point-steps off / on   L1 ρ against one patch, off / on
WC     1         4380 / 5512
WC     2         4383 / 5540      36965 / 0                  8.49e-4 / 1.53e-3
WC     3         4385 / 5500      36727 / 0                  4.17e-3 / 4.97e-3
Noh    1         3512 / 5160
Noh    2         3511 / 5164      30295 / 0                  9.10e-6 / 2.62e-4
Noh    3         3511 / 5123      30315 / 0                  9.01e-6 / 2.47e-4

one patch, L1 ρ after a relative change of 1e-15 to the initial energy, off / on
WC     3.48e-8 / 3.66e-4
Noh    1.07e-5 / 2.95e-4
```

No limited run leaves an inadmissible point-step or an unguaranteed side. A limited run is far
more sensitive to round-off than an unlimited one, since θ is not continuous in round-off, and
the limited Noh runs on patches differ from one patch by less than the 1e-15 change moves the
one-patch run. The interface test of `test/positivity_tests.jl`, a periodic
blast on two patches with the artificial properties and the filter off, conserves the composite
Σ W Q to 2.6e-14 of the totals, with the end face carrying the face register at that end and
its corrections entered there. On the periodic blast of 80² the limited step costs as much
relative to the unlimited one on two patches as on one:

```
periodic blast 80², 300 steps   off, ms/step   on, ms/step   on/off
one patch                       3.251          6.100         1.88
two patches                     3.567          6.544         1.85
```

By default setup closes a limited run's interfaces with the closure rows. Under
`PatchInterfaces(limited_ghost = true)` (`flux=ghost` sets it) it keeps `patch_interfaces = :ghost`, the
gradient plans difference the whole flux of an interface dimension from ghost values, and the
limiter takes the line as open there: weight h up to the end node and h/2 at it, the end face
at the node, and the face values anchored by the derivative's interior face relation measured at
the third face from the low end. On a smooth periodic flux of two patches the open end's face
values meet the point flux at the shared nodes to round-off, and its interior faces come nearly
two orders of magnitude closer to the exact flux than the closure rows' faces beside an
interface:

```
ρu = 0.5(1 + 0.2 sin 2πx), 64 nodes in two patches, one stage   error of the face values
:ghost, at the two shared nodes, against the point flux          1.3e-11
:ghost, every interior face, against the exact flux              1.7e-5
:closure, faces beside an interface, against the exact flux      1.3e-3
```

The same cases under `:ghost`:

```
case   patches   steps off / on   bad point-steps off / on   L1 ρ against one patch, off / on
WC     2         4382 / 5480      36922 / 0                  8.16e-4 / 9.74e-4
WC     3         4383 / 5560      36661 / 0                  2.72e-3 / 2.72e-3
Noh    2         3512 / 5134      30312 / 0                  9.48e-6 / 2.69e-4
Noh    3         3511 / 5122      30312 / 0                  1.21e-5 / 2.75e-4
```

No limited run leaves an inadmissible point-step or an unguaranteed side. Woodward–Colella on
patches comes closer to the one-patch run under `:ghost` than under `:closure`, limited or not.
On device storage under the KernelAbstractions CPU backend the limited runs on two patches and
through a box, at the global step and subcycled, reproduce the host bitwise under `:ghost` as
under `:closure` (`test/device_tests.jl`).

The limiter leaves no inadmissible point-step on any refined layout, the parent's shell
included. `part=levels` runs refined levels under `patch_interfaces = :closure`. A refined patch
closes its lines at a coarse-fine face with the same interface rows, and its end node there,
which the parent's shell overwrites, is not held. The parent's nodes the restriction overwrites
are exempt from their cell bounds and kept out of their held neighbours' first-order fluxes where
not admissible. Held to their bounds under subcycling, where the parent steps them at up to the
CFL number 0.75, they fell to temperatures near zero, and κ\* = ρc/T there sized the root step.
Both nodes of a face beside such a node take one first-order flux, with the node replaced by its
held neighbour, so that the two corrections of the face agree and the face register carries no
difference to the far end of the line. A shell node the interpolation leaves inadmissible
between admissible parent nodes takes the multilinear, linear-in-time interpolant of the parent.
The inadmissible point-steps after every step, those on a refined patch's coarse-fine plane
apart:

```
case                                   steps off / on   bad off: shell, rest   bad on: shell, rest
WC 200, box, global step               1985 / 2349      145, 14950             0, 0
WC 200, box, subcycled                 1133 / 1561      130, 10341             0, 0
WC 200, regridded box                  3202 / 4259      0, 32845               0, 0
WC 200, regridded tiles of 8           1094 / 1416      0, 8433                0, 0
WC 200, regridded tiles, subcycled     1085 / 1413      0, 8442                0, 0
Noh ν = 2, box over 85 nodes           6640 / 9220      0, 56700               0, 0
Noh ν = 2, box over 43 nodes           6024 / 8155      176, 50648             0, 0
Noh ν = 2, 85 nodes, subcycled         2214 / 2989      0, 18925               0, 0
Noh ν = 2, 43 nodes, subcycled         2330 / 4167      73, 19809              0, 0
r-z converging shock, corner box       628 / 627        0, 911                 0, 0
blast 64², tiles of 8                  231 / 256        1539, 9877             0, 0
blast 64², tiles of 8, subcycled       137 / 216        1517, 6902             0, 0
```

A limited box that holds the shock's whole path reproduces the limited uniform run at the
level's spacing. A box shorter than the final front radius, which the shock leaves through the
coarse-fine face, loses a twentieth of the plateau with or without the limiter.

```
cylindrical Noh at t = 0.6        plateau   deficit  shock    mass
uniform at the level's spacing    15.6674   50.7%    0.2032   −4.4e-7
box over 85 nodes                 15.6672   50.9%    0.2032   −5.2e-7
uniform, unlimited                15.6523   50.9%    0.2033
box over 85 nodes, unlimited      15.6523   50.9%    0.2033
box over 43 nodes                 14.8280   50.6%    0.2086   −1.2e-4
box over 43 nodes, unlimited      14.9258   50.2%    0.2104   +3.7e-6
```

The 3410 unguaranteed sides of the shorter box are the parent's, from the shock's arrival in its
covered nodes. Unlimited, the shell nodes left inadmissible lie between parent nodes that are
inadmissible themselves, which no interpolant repairs.

On a refined level the limiter costs least on the one-dimensional boxes and most on the r-z
corner box, under either interface flux. `part=levelcost` times the first steps of the same cases
as `part=cost` does, and `part=levelcost flux=ghost` the same under `:ghost`:

```
                                       :closure                              :ghost
case                                   off, ms/step   on, ms/step   on/off   off, ms/step   on, ms/step   on/off
WC 200, box, global step               0.351          0.346         1.04     0.356          0.389         1.04
WC 200, box, subcycled                 0.582          0.714         1.23     0.693          0.774         1.12
Noh ν = 2, box over 85 nodes           0.487          0.641         1.30     0.520          0.671         1.30
blast 64², tiles of 8                  12.266         17.500        1.43     16.212         22.056        1.33
r-z converging shock 64², corner box   18.439         34.507        1.86     23.890         39.322        1.65
```

On the radial grids a face two tiles share along r is limited as a same-level interface is: the
end cell takes both its faces and its geometric part in one part, its end face blended toward
the node's area-weighted point flux and pressure by the θ of the face beside it, and the
exchange gives both tiles the smaller θ. The cases are the cylindrical Noh of the level rows,
the level over the first 81 root nodes in ten lattice cells of edge 8, and the A19 converging
shock with tiles over the three by three lattice cells within the corner box:

```
case                                   steps off / on   bad off: shell, rest   bad on: shell, rest
Noh ν = 2, box over 81 nodes           6641 / 9217      0, 56681               0, 0
Noh ν = 2, tiles of 8                  6842 / 10180     0, 77151               0, 0
Noh ν = 2, tiles of 8, subcycled       2279 / 3422      0, 25696               0, 0
Noh ν = 2, regridded tiles             6848 / 10292     0, 77094               0, 0
Noh ν = 2, regridded tiles, subcycled  2285 / 3447      10, 26016              0, 0
r-z converging shock 33², corner box   322 / 322        0, 105                 0, 0
r-z converging shock 33², tiles of 8   329 / 329        0, 389                 0, 0
the same, subcycled                    132 / 201        0, 146                 0, 0
the same, regridded tiles              329 / 334        0, 366                 0, 0
regridded tiles, subcycled             120 / 199        0, 156                 0, 0
regridded tiles at 32²                 321 / 325        0, 344                 0, 0
```

At 32² the last lattice cell at the outer faces spans only 7 root spacings. The line of a tile
over it, 22 nodes between its interface and the wall, takes no face form. Under the limiter
a regrid does not create that tile and the cell stays at the root's resolution, and setup
refuses a level given over it. The limited converging-shock tiles leave 25 to 41 unguaranteed
sides.

On the cylindrical Noh the limited tiles reproduce the limited box's plateau and shock, and the
limiter reduces the tiled level's own mass drift, which `part=tiledrift` below attributes, to
between a fifth and a third:

```
Noh ν = 2, level over 81 nodes         plateau   deficit   shock    mass       unguaranteed
box, limited                           15.6672   50.8%     0.2032   −5.2e-7
tiles of 8, limited                    15.6637   51.1%     0.2032   +5.92e-5   0
tiles of 8, subcycled, limited         15.6645   50.6%     0.2032   +6.32e-5   8
regridded tiles, limited               15.6645   51.3%     0.2032   +7.57e-5   18
regridded, subcycled, limited          15.6680   51.0%     0.2032   +9.07e-5   18
box, unlimited                         15.6523   50.9%     0.2033   −9.7e-7
tiles of 8, unlimited                  15.7202   50.2%     0.2029   +2.97e-4
tiles of 8, subcycled, unlimited       15.7216   50.0%     0.2029   +2.89e-4
regridded tiles, unlimited             15.7219   50.2%     0.2029   +2.81e-4
regridded, subcycled, unlimited        15.7315   49.9%     0.2028   +3.24e-4
```

On the converging shock the limited tiled step costs about a quarter more than the limited box's
under either interface flux. `part=tilecost` times the first 150 steps of the box and the tiles,
the four runs alternated three times in one process:

```
converging shock 33²        off, ms/step   on, ms/step
:closure, corner box        5.341          9.277
:closure, tiles of 8        6.898          11.830
:ghost, corner box          7.377          11.127
:ghost, tiles of 8          9.226          14.220

limited tiles against       unlimited tiles   limited box
:closure                    1.72              1.28
:ghost                      1.54              1.27
```

`part=levels flux=ghost` runs the same refined cases under `patch_interfaces = :ghost`, a
refined patch's lines open at its coarse-fine and shared faces:

```
case                                   steps off / on   bad off: shell, rest   bad on: shell, rest
WC 200, box, global step               1993 / 2355      148, 15096             0, 0
WC 200, box, subcycled                 1133 / 1578      130, 10354             0, 0
WC 200, regridded box                  3201 / 4219      0, 32821               0, 0
WC 200, regridded tiles of 8           1094 / 1410      0, 8433                0, 0
WC 200, regridded tiles, subcycled     1085 / 1437      0, 8443                0, 0
Noh ν = 2, box over 85 nodes           6641 / 9242      0, 56695               0, 0
Noh ν = 2, box over 43 nodes           6006 / 8191      108, 50224             0, 0
Noh ν = 2, 85 nodes, subcycled         2214 / 2985      0, 18929               0, 0
Noh ν = 2, 43 nodes, subcycled         2321 / 4190      24, 19540              0, 0
Noh ν = 2, tiles of 8 over 81 nodes    6636 / 8846      0, 64289               0, 0
the same, subcycled                    2212 / 2897      0, 21440               0, 0
the same, regridded                    6641 / 9279      5, 64730               0, 0
regridded and subcycled                2213 / 3075      7, 21663               0, 0
blast 64², tiles of 8                  224 / 249        1530, 9589             0, 0
blast 64², tiles of 8, subcycled       136 / 227        1539, 6752             0, 0
r-z converging shock 33², corner box   322 / 322        0, 109                 0, 0
r-z converging shock 33², tiles of 8   322 / 323        0, 365                 0, 0
the same, subcycled                    130 / 199        0, 136                 0, 0
the same, regridded tiles              323 / 328        0, 336                 0, 0
regridded tiles, subcycled             118 / 199        0, 137                 0, 0
```

Under `:ghost` the limited Noh box over 85 nodes reproduces the closure rows' metrics, and the
limited tiles, static or regridded, drift by about as much as under `:closure`:

```
:ghost, Noh ν = 2                    plateau   deficit   shock    mass       unguaranteed
box over 85 nodes, limited           15.6673   50.8%     0.2032   −5.2e-7
the same under :closure              15.6672   50.9%     0.2032   −5.2e-7
box over 43 nodes, limited           14.8038   50.6%     0.2086   −1.2e-4    3786
box over 43 nodes, unlimited         14.9507   50.7%     0.2109   +8.4e-6
tiles of 8 over 81 nodes, limited    15.6612   50.8%     0.2032   +6.78e-5   0
tiles of 8 over 81 nodes, unlimited  15.6526   50.7%     0.2033   −2.45e-6
regridded tiles, limited             15.6614   50.4%     0.2032   +7.59e-5   0
regridded tiles, unlimited           15.6532   50.7%     0.2033   −1.56e-5
```

The tiled rows leave 0 to 48 unguaranteed sides, the most on the converging shock. A regrid's
fill takes the shell's admissible fallback (`_fill_fallback!`) in every limited run, since a
fill can otherwise put a node of a new tile at negative ρe in the cold gas ahead of the shock,
whose cell then takes θ = 0 at both faces until it recovers. The costs under `:ghost` are in the
two cost tables above.

With the identity filter row at a node two tiles share, a tiled level's mass drift enters at
the shared faces as the shock crosses them, and the filter passes carry most of it.
`part=tiledrift ledger=true` runs the same cylindrical Noh, the level over the first 81 root
nodes, unlimited and under `patch_interfaces = :closure` except where a row says otherwise, on
one layout and with one change at a time. The table below was taken with the identity row
(`conservative=false` on the current tree) at package commit `2d5178f5` on Julia 1.13.3
(transcript `bench/archive/results/tiledrift.txt`). Beside the composite mass it prints two that use no
covered mask and no junction weight: each refined patch's single-patch quadrature plus the
root's from the level's last node out (split), and the root's alone over the whole line, which
holds the restricted level (root). Drift is against the exact mass at t = 0.6:

```
layout, change                 steps  plateau  deficit  shock   mass      split     root
uniform at the level's spacing 6640   15.6523  50.9%    0.2033  −4.5e-7
box                            6641   15.6523  50.9%    0.2033  −9.7e-7   −1.9e-6   +5.5e-4
one tile of edge 80            the box in every printed digit
box, subcycled                 2213   15.6524  50.8%    0.2033  −9.3e-7   −1.8e-6   +5.5e-4
box, interface_flux = :ghost   6640   15.6523  50.9%    0.2033  −9.7e-7   −1.9e-6   +5.5e-4
tiles of 8                     6837   15.8915  49.6%    0.2024  +1.73e-3  +1.73e-3  +2.55e-3
tiles of 16                    6737   15.7918  51.0%    0.2028  +9.2e-4   +9.2e-4   +1.96e-3
tiles of 8, subcycled          2277   15.8828  49.4%    0.2024  +1.56e-3  +1.56e-3  +2.33e-3
regridded tiles of 8           6836   15.9152  49.5%    0.2023  +1.73e-3
tiles of 8, uncorrected        6837   15.8915  49.6%    0.2024  +1.73e-3  +1.73e-3  +2.55e-3
tiles of 8, sensor clamped     6827   15.8429  49.9%    0.2026  +1.05e-3  +1.05e-3  +1.84e-3
tiles of 8, :ghost             6634   15.6554  50.8%    0.2033  +1.9e-5   +1.8e-5   +5.7e-4
box, limited                   9217   15.6672  50.8%    0.2032  −5.2e-7   −1.9e-6   +6.3e-4
tiles of 8, limited            10221  15.6815  50.5%    0.2033  +4.15e-4  +4.13e-4  +1.14e-3
box, limited, :ghost           9225   15.6674  50.7%    0.2032  −5.2e-7   −1.9e-6   +6.3e-4
tiles of 8, limited, :ghost    8881   15.6628  50.9%    0.2033  +2.09e-4  +2.08e-4  +8.50e-4
tiles of 16, limited, :ghost   9019   15.6674  50.7%    0.2032  +1.2e-4   +1.2e-4   +7.6e-4
```

The drift is in the solution. The split mass agrees with the composite to 1e-6 on every static
layout. The root's mass, which carries the root's own quadrature of the shock on the box, is
2.0e-3 higher on the tiles. The mass enters at the faces two tiles share, where each tile's
divergence closes with the closure rows of `:closure`, as the shock crosses one. A single tile
reproduces the box in every printed digit. The composite drift's history rises in steps of about
2e-4 at the times the shock reaches a tile face, and more slowly between them, so doubling the
edge halves the drift:

```
times of the drift history's steps   t
tiles of 8                           0.188, 0.28, 0.375, 0.47, 0.56
tiles of 16                          0.188, 0.375, 0.56
```

Up to t = 0.16 the drift stays within 3e-5. The coarse-fine face at r = 0.313 is not involved:
the shock does not reach it, and with the correction off every printed digit of the final state is
unchanged. Subcycling and regridding change the drift by a tenth or less, and clamping the
sensor at the faces (`SENSOR_INTERFACE_GHOSTS` off) lowers it by two fifths. Under
`interface_flux = :ghost`, where the flux through the face is differentiated from ghost values
with the interior stencil, the drift is 1% of its value under `:closure`. A smooth density pulse
at rest crossing the same faces to t = 0.3 drifts by −1.9e-9 on the tiles under `:closure`, so
the closure-row faces conserve mass on resolved data. Without the filter the tiles stop at
t = 0.021 and the box at 0.104. The budget ledger attributes the drift instead: on the tiles
under `:closure` the filter passes carry about three quarters of it (the off columns of the
toggle table below). The identity row at a node two tiles share does not conserve, as a closure
row at a wall does not.

Under the positivity limiter with the identity row the filter passes still carry most of the
tiles' drift, and the limited box drifts by under 1e-6, as the unlimited one does. The limited
`:ghost` tiles drift half as much as the limited `:closure` tiles and lie closer to the uniform
run, but ten times as much as the unlimited `:ghost` tiles. The filter's share there grows
sixfold under the limiter, and the stage's changes from −5e-6 to +5e-5.

The planar Noh (ν = 1) on the same layouts shows the same drift on the tiles with the identity
row, limited or not (the off column of the planar table below). Its root has 400 nodes and the
level covers the first 81. The drift there is read at t = 0.5, before the shock reaches the
level's outer face at x = 0.2005. At t = 0.6 the shock lies on that face, so the planar runs'
final metrics and unmasked masses measure the coarse-fine face and not the tiles.

With the identity row the tiles are closer than the box to the exact solution in the L1 error
and the shock position, and further from it in the plateau's flatness:

```
identity row, against the exact solution   L1 ∫|ρ − ρ_exact| r dr over the level   shock    mean |ρ − 16|, plateau core
tiles of 8                                 1.398e-2                                0.2024   0.377
box                                        1.456e-2                                0.2033   0.348
exact                                                                              0.2
```

The plateau core's mean rises by 0.24 and its spread grows, with the mass gained at the faces.
Against the uniform run at the level's spacing, which the box matches closely, the tiles differ
by 61% of their own error against the exact solution (the L1 column of the toggle table below).

Closing the state filter at a node two same-level patches or two tiles share with conservative
rows takes the filter passes' share of the tiled level's mass drift down to the box's; what
remains of the drift under `:closure` enters through the right-hand side at the closure-row
faces. The rows are the default (package commits `be418e3`, `5853b90` and `b4388f9`), the test
toggle `CONSERVATIVE_INTERFACE_FILTER` off restoring the identity row for the off columns below,
and `_conservative_filter_rows` in src/core/kernels.jl describes them: the two
faces beside the shared node take the explicit value g/σ of the filter's face flux, g the face
stencil of the residual B − A and σ the left-hand side's row sum, so that a patch's corrections
telescope to the same face value on both sides. A coarse-fine end keeps the identity row, its node
overwritten by the shell. A limited filter pass holds the shared node, its end face and the face
beside it in one part with one θ, which the exchange of a same-level interface makes the same on
both patches.

The rows leave a level with no same-level face bit for bit as it is: boxes at the global step and
subcycled, limited and unlimited, under either flux, regridded, on the host and under the
KernelAbstractions CPU backend.
The table gives the cylindrical Noh's mass drift against the exact mass, the filter
passes' share of it from the ledger, and the L1 density difference from the uniform run at the
level's spacing over the level's region:

```
cylindrical Noh, level over root nodes 1:81, t = 0.6, Julia 1.13.3
                               toggle off                       toggle on
layout, change                 mass      filter    L1           steps  plateau  deficit  shock   mass      filter    L1
box                            −9.7e-7   −4.6e-7   1.105e-5     6641   15.6523  50.9%    0.2033  −9.7e-7   −4.6e-7   1.105e-5
tiles of 8                     +1.73e-3  +1.32e-3  8.474e-3     6842   15.7202  50.2%    0.2029  +2.97e-4  −2.24e-6  4.894e-3
tiles of 16                    +9.16e-4  +6.82e-4  4.630e-3     6734   15.6917  50.9%    0.2031  +1.32e-4  −9.36e-7  2.881e-3
tiles of 8, subcycled          +1.56e-3                         2279   15.7216  50.0%    0.2029  +2.89e-4  −1.95e-6  4.819e-3
regridded tiles of 8           +1.73e-3  +1.35e-3  8.700e-3     6848   15.7219  50.2%    0.2029  +2.81e-4  +1.63e-5  4.932e-3
regridded box                  +1.88e-5  +2.93e-5               6647   15.6517  50.7%    0.2033  +1.88e-5  +2.93e-5  4.116e-4
tiles of 8, sensor clamped     +1.05e-3                         6842   15.6924  50.6%    0.2035  +1.94e-4  −4.16e-6  4.832e-3
tiles of 8, :ghost             +1.87e-5  +2.38e-5  6.396e-4     6636   15.6526  50.7%    0.2033  −2.45e-6  −3.33e-7  5.416e-4
box, limited                   −5.18e-7  −4.58e-7  6.436e-4     9217   15.6672  50.8%    0.2032  −5.18e-7  −4.58e-7  6.436e-4
tiles of 8, limited            +4.15e-4  +3.49e-4  1.729e-3     10180  15.6637  51.1%    0.2032  +5.92e-5  +5.43e-5  1.315e-3
tiles of 8, limited, :ghost    +2.09e-4  +1.56e-4  7.407e-4     8846   15.6612  50.8%    0.2032  +6.78e-5  +6.62e-6  7.063e-4
tiles of 16, limited, :ghost   +1.18e-4  +8.78e-5  7.214e-4     9012   15.6647  50.9%    0.2032  +4.12e-5  +5.43e-6  7.678e-4
```

The regridded box, which has no same-level face and moves with the shock, takes a larger filter
share than the regridded tiles do with the toggle on. The filter share left on regridded tiles is
therefore the moving coarse-fine face's, which a static level's face, never reached by the shock,
does not see. Limited, the filter share under `:closure` is eight times the one under `:ghost`.
Under `:closure` a limited face's correction is distributed by the divergence's closure-row
weights, which are not the composite quadrature's h and h/2; at an open `:ghost` end they are.

On the planar Noh, before the shock reaches the level's outer face, the rows take the tiles'
drift down by about an order of magnitude under `:closure`, limited or not. A smooth pulse at
rest crossing the tile faces drifts by under two parts in a billion either way:

```
                                 planar Noh, drift at t = 0.5    smooth pulse, drift to t = 0.3
layout, change                   off        on                   off        on
box                              −7.0e-6    −7.0e-6              +1.12e-10  +1.12e-10
tiles of 8                       +3.9e-3    +4.1e-4              −1.91e-9   +1.29e-9
tiles of 8, :ghost               +3.6e-5    −1.1e-5              +1.05e-10  +1.05e-10
tiles of 8, limited              +8.8e-4    +1.2e-4
tiles of 8, limited, :ghost      +3.1e-4    +8.5e-5
```

The conservative rows conserve the composite pass to round-off for every filter that takes rows,
and they converge to the single line's pass one order faster than the identity row.
`bench/archive/filter_conservation.jl parts=shared N=48` assembles the composite operator of a periodic
line held as two patches sharing two nodes, each patch's line closed by the interface rows and its
ghost layers holding the other patch's nodes, a shared node the mean of its two copies. The table
gives the defect d = Mᵀ V − V relative to V = h and the composite pass against the single line's
on a smooth wave:

```
wave sin(2π · 3i/n + 0.37)
filter                  rows           max |d|/V   Σ|d|/ΣV   composite − single, n = 48, 96, 192   order
compact_filter()        identity       2.20e-2     7.3e-3    2.4e-8    9.6e-11   3.8e-13           7.99
                        conservative   2.2e-16     7.2e-17   1.1e-8    2.1e-11   4.0e-14           9.02
compact_filter(0.45)    identity       3.42e-2     9.6e-3    4.1e-8    1.6e-10   6.4e-13           7.99
                        conservative   4.4e-16     9.0e-17   1.7e-8    3.3e-11   6.4e-14           9.01
pyranda_filter()        identity       1.35e-3     9.5e-4    1.9e-9    6.2e-12   5.9e-14           7.51
                        conservative   3.3e-16     9.7e-17   1.2e-9    2.4e-12   1.2e-13           6.61
gaussian_filter()       none           2.2e-16     9.3e-18   0         0         0
```

The identity row's defect sits at the shared node and its neighbours with alternating sign. Under
either row set the two copies of a shared node agree bit for bit. The pentadiagonal row's last
figure is at round-off, and the explicit Gaussian reads the ghosts through its interior stencil
and is the single line's operator. In the solver, one pass over a level of six tiles spanning a
periodic line, with a density step at two shared nodes, keeps the tiles' mass to round-off under
the rows (`test/conservation_tests.jl`).

The rows keep the interface order, and a filtered two-patch run comes closer to the one-patch run
with them than without. The filtered interface rows of `test/convergence.jl` that have no
same-level face reproduce their recorded orders with the toggle on to every printed digit. The
table gives the orders of the rows with a shared face, added for this measurement (the entropy
wave of `test/smooth_cases.jl` under `compact_filter(0.45)` at every step), and the two-patch
run's maximum density difference from the one-patch run:

```
row                                      order, off   order, on
two levels, tiles of 4                   6.87         6.87
two patches, k = 1                       6.66         6.70
two patches, k = 3                       6.77         6.78

two patches against one patch, k = 3     N = 48     N = 96     N = 192
toggle off                               1.660e-6   1.488e-8   1.109e-10
toggle on                                1.322e-6   1.036e-8   7.819e-11
the one-patch run's own error            5.240e-7   5.468e-9   9.848e-11
```

The validation level rows of `test/validation.jl`, a box at the axis, have no same-level face and
do not move. Tiles over the same nodes drift less with the toggle on:

```
cylindrical Noh, level over root nodes 1:85     mass, off    mass, on
tiles of 8                                      +1.83e-5     −2.56e-6
```

Under the positivity limiter the rows leave no inadmissible point-step on any tiled row of the
level tables, under either interface flux, and the Noh tiles' drift falls by a factor of six to
seven under `:closure` and of three under `:ghost`. `part=levels` runs the rows with the limiter
on, `conservative=false` giving the off columns; the table gives steps and unguaranteed sides,
and the Noh tiles' mass drift:

```
case                                   :closure, off   :closure, on   :ghost, off   :ghost, on
WC 200, regridded box                  4259, 0         4259, 0        4219, 0       4219, 0
WC 200, regridded tiles of 8           1420, 0         1416, 0        1429, 0       1410, 0
WC 200, regridded tiles, subcycled     1440, 0         1413, 0        1459, 0       1437, 0
blast 64², tiles of 8                  256, 1          256, 0         248, 1        249, 1
blast 64², tiles of 8, subcycled       219, 0          216, 0         226, 2        227, 0
Noh ν = 2, tiles of 8                  10221, 0        10180, 0       8881, 0       8846, 0
Noh ν = 2, tiles of 8, subcycled       3217, 12        3422, 8        2903, 0       2897, 0
Noh ν = 2, regridded tiles             10114, 28       10292, 18      9329, 38      9279, 0
Noh ν = 2, regridded, subcycled        3420, 8         3447, 18       3139, 264     3075, 0
r-z converging shock 33², tiles of 8   330, 31         329, 31        322, 32       323, 48
the same, subcycled                    201, 50         201, 41        199, 36       199, 36
the same, regridded tiles              334, 36         334, 35        326, 31       328, 35
regridded tiles, subcycled             199, 39         199, 31        198, 44       199, 40

Noh ν = 2, mass drift                  :closure, off   :closure, on   :ghost, off   :ghost, on
tiles of 8                             +4.15e-4        +5.92e-5       +2.09e-4      +6.78e-5
tiles of 8, subcycled                  +4.25e-4        +6.32e-5       +2.16e-4      +6.90e-5
regridded tiles                        +4.48e-4        +7.57e-5       +2.32e-4      +7.59e-5
regridded, subcycled                   +5.59e-4        +9.07e-5       +2.35e-4      +7.73e-5
```

With the rows the limited `:closure` tiles take within 6% of the steps they take without. The
regridded, subcycled tiles under `:ghost` leave many more unguaranteed sides with the identity row
than with the rows, a count whose cause was not traced; the regridded box, which has no
same-level face, is unchanged.

The rows depend neither on the rank count nor on the storage, and cost nothing measurable. With
the toggle on, the time reached and the composite mass agree to 5e-15 relative between one, two,
four and eight ranks on four tiles of a periodic square, on twelve subcycled tiles of a level on
a line and six limited ones, on two root patches whose lines split over the ranks, unlimited
and limited, and on a box level split over the ranks. Under the KernelAbstractions CPU backend the
patched, refined, tiled and limited runs of `test/device_tests.jl` reproduce the host bitwise with
the toggle on, as does a level whose tiles of one extent fall into separate stacks by face kind;
the testsets' assertions on the number of stacks hold with the toggle off only. The step's cost
with the toggle on, against off, medians of five alternations in one process:

```
blast 64²                     on / off
tiles of 8                    1.011
tiles of 8, limited           1.014
two patches, limited          1.000
```

## Directional bulk viscosity

```text
julia --project=. -t 1 bench/archive/anisotropic.jl
```

A directional artificial bulk viscosity carries one coefficient per grid direction with the
matching per-direction diffusive step limit, where the scalar form loses a factor of the
aspect ratio in the step (Olson & Lele 2012, 2013; Comput. Sci. Disc. 5, 014008 and J.
Comput. Phys. 246, 207). The battery is one-dimensional throughout and blind to the
difference, so two anisotropic Noh implosions were built against `noh_exact`, on Cartesian
grids whose spacing along the first dimension is 1/AR of the spacing along the second.

- `noh_aligned`: the planar (ν = 1) implosion along the coarse dimension, with `nx = 12`
  periodic points across the fine one. The data carry no variation along dimension 1, so the
  solution is the one-dimensional profile at every station, the sensor's fine-direction term
  is zero exactly, and the forms differ only in the step they take. The largest transverse
  variation of the density at the end (`uniformity`) is amplified round-off.
- `noh_cartesian`: the cylindrical (ν = 2) implosion on the plane [−L, L]² converging on its
  central node, with the exact time-dependent inflow on all four faces: a curved front
  oblique to the grid at every angle over an irrotational pre-shock flow. On a walled
  quadrant the two walls and their corner dominated every measure (a corner deficit of 79%
  at AR 1), so the case takes the full plane. The warm start of the spherical case rings
  here, so it starts singular, which the scalar form completes at `cfl = 0.3`. The plateau is
  under-resolved, under the scalar form still 12% low at N = 64, so the comparison is
  between forms at equal resolution, not against the exact plateau.

### The forms

Cook's scalar coefficient is β\* = C_β ρ G[Σ_d Δ_d² |δ⁴_d S|], with G the smoother, entering
every normal stress as β\* ∇·u and the step through 2β\*/ρ · Σ_d 1/Δ_d². Two directional
forms were measured as an `ArtificialProperties` option, which is not in the package. Each
stored β\*_d in three arrays, entered direction d's normal stress as β\*_d ∇·u (the
reference's eq. 5), and was charged in the step as 2 Σ_d β\*_d/(ρ Δ_d²):

- the **sensor split**, β\*_d = C_β ρ G[Δ_d² |δ⁴_d S|]: each direction's own ringing
  weighted by its own spacing and smoothed on its own. Under Σ_d the three sum to the scalar
  coefficient to round-off, and in one dimension the arrays are bitwise the scalar ones; on
  an isotropic grid the coefficients still differ by direction;
- the **spacing-scaled split**, β\*_d = β\* (Δ_d/Δ_max)² from the scalar coefficient as
  built: the coarsest direction keeps β\* exactly, a finer one takes the coefficient the
  same ringing would have produced on a grid of its own spacing, every β\*_d ≤ β\*, and on
  an isotropic grid every one is β\* bitwise, the two properties the reference states and
  the sensor split lacks. Its construction could not be read in full, so both were measured.

Both remove the aspect-ratio penalty from the step exactly: on the aligned case the fine
direction's coefficient is zero under the first and Δ_x²/Δ_y² of β\* under the second, so
Σ_d β\*_d/Δ_d² is the coarse direction's alone.

### The aligned case

N = 100, `cfl = 0.3`, `filter_cfl = 0.35`, under a filter weight reading the step's rate:

```
AR   form          steps   limit      plateau   deficit   shock    transverse
 1   scalar          761   diffusive  3.9855     64%      0.2126   3.1e-9
 1   scaled          761   diffusive  3.9855     64%      0.2126   2.3e-9
 1   directional     491   diffusive  3.9846     64%      0.2128   7.2e-11
 4   scalar         5149   diffusive  3.9161     50%      0.2116   1.2e-10
 4   scaled         1139   diffusive  3.9848     62%      0.2120   1.0e-7
 4   directional    1107   diffusive  3.9849     63%      0.2120   4.6e-8
16   scalar        67994   diffusive  1.0288   −501%      0.0465   1.0e-9
16   scaled           step collapse at t = 0.56
16   directional    4471   acoustic   3.9305     53%      0.2137   1.3e-6
```

Where they complete, the directional forms divide the step count by about the aspect ratio,
the predicted gain. Even the one-dimensional battery is diffusion-limited at the front under
C_β = 1, so the aspect ratio enters the scalar form's step as AR² through 1/Δ_x² and the
directional forms' as AR through the acoustic rate.

At the largest aspect ratio the scalar form completes with a wrong solution: the wall
density is 24 against the exact 4, and the front lies far inside the exact position. The
cause is not the bulk viscosity, whose flux along the fine direction is zero on this flow,
but the relaxed filter reading the rate that sized the step: the scalar run makes 68k
weighted passes against the directional run's 3.8k. The fix, a directional filter weight,
is in [the filter's dissipation](filter.md#the-filters-dissipation), where the same case reads one
profile at every aspect ratio.

### The curved case

On the curved front both directional forms fail at AR 4 and above, while the scalar form
completes to AR 8. Where they complete, they leave five to seven times the scalar form's
pre-shock vorticity at AR 2. |ω| and |∇·u| are the largest values over the pre-shock region
r > 1.2 R_s at the end:

```
N = 24 per half-side, cfl = 0.3, p₀ = 1e-4, to t = 0.6
AR   form          outcome                          steps   plateau  center   L1 rho   |omega|  |div u|
 1   scalar        completes                          339   11.443    64%     1.055     1.27    21.0
 1   scaled        completes (= scalar, bitwise)      338   11.443    64%     1.055     1.27    21.0
 1   directional   completes                          320   12.332    57%     0.834     5.77    22.8
 2   scalar        completes                          752   11.766    41%     0.898     2.48    36.3
 2   scaled        completes                          531   11.896    29%     0.848    13.6     30.8
 2   directional   completes                          556   12.218    28%     0.760    16.4     28.2
 3   directional   negative density at t = 0.274
 4   scalar        completes                         2992   12.554    40%     0.904    24.0     60.5
 4   scaled        negative density at t = 0.100
 4   directional   negative density at t = 0.101
 8   scalar        completes                        13561   12.942    35%     0.919    24.0     60.5
 8   scaled        step collapse at t = 0.020
 8   directional   negative density at t = 0.021
```

The failure does not move with the step, the reduction or the ambient pressure, and comes
earlier on the finer N = 48 grid. The sensor field moves it without removing it. At AR 4
unless stated:

```
variant                         form               outcome
N = 48                          both directional   fails earlier than at N = 24
cfl 0.3, 0.15, 0.075            both directional   fails at t = 0.095–0.10
p₀ = 1e-2, 1e-1                 both directional   fails at t = 0.09–0.12
                                scalar             completes
:ungated_dilatation             spacing-scaled     fails at t = 0.52
:gated_strain                   spacing-scaled     fails at t = 0.087
:dilatation, N = 24             spacing-scaled     completes
:dilatation, N = 48 or AR 8     spacing-scaled     fails
:dilatation, AR 8               scalar             completes
```

The mechanism is the stress form. A scalar bulk viscosity exerts the force ∇(β\* ∇·u), the
gradient of a scalar, which cannot create vorticity; with unequal coefficients the force is
(∂_x(β\*_x ∇·u), ∂_y(β\*_y ∇·u)), which is not a gradient, and in Noh's cold, converging,
irrotational pre-shock gas with no pressure to resist it, it does. Early in the AR 4 run the
directional forms carry about twenty times the scalar form's pre-shock vorticity, several
times the largest dilatation, and drive the internal energy far below zero:

```
AR 4, t = 0.06        largest pre-shock |omega|   worst internal energy
scalar                19.9                        −0.095
directional forms     428, 414                    −6.2
largest dilatation    59–86
```

The run then cavitates just outside the front, at 70–76° from the fine axis, where the
y-strain has turned to expansion. The density there falls to 0.47 in gas that should be at
3.2, and the step collapses. The same want of fine-direction damping shows in the aligned
case's transverse round-off, two to three orders larger under the directional forms at AR 4
and 16. The compression switch removes the rotational force where the flow has begun to
expand, postponing the failure without acting before the vorticity exists.

### Decision

Neither directional form is adopted, and the implementation was not retained. The benefit is
real on the aligned control, but the form fails on the curved case, the case a stretched
grid is used for, by a mechanism intrinsic to a bulk stress with unequal diagonal
coefficients rather than to a constant or a sensor. The reference's success on a blast wave
and a nozzle boundary layer is neither reproduced nor contradicted here: both carry pressure
everywhere, and neither converges cold gas onto a point. The two cases remain, guarded in
`test/validation.jl` under the scalar form, with `bench/archive/anisotropic.jl` as the instrument
for the scalar form's aspect-ratio penalty.

## The aligned Noh transverse mode

```text
julia --project=. -t 1 bench/archive/noh_transverse.jl trace sample=100
julia --project=. -t 1 bench/archive/noh_transverse.jl seeds seed_mode=2 sample=100
julia --project=. -t 1 bench/archive/noh_transverse.jl uniform seed_mode=2 seed=1e-10 sample=50
julia --project=. -t 1 bench/archive/noh_transverse.jl warm t0=0.1 seed_mode=2 seed=1e-10
julia --project=. -t 1 bench/archive/noh_transverse.jl widths sample=100
julia --project=. -t 1 bench/archive/noh_transverse.jl seed_channels seed_mode=2 seed=1e-10 sample=100
julia --project=. -t 1 bench/archive/noh_transverse.jl extended tfinal=2.0 sample=200 nmax=20000
```

The aligned Noh case grows a transverse disturbance from round-off. The baseline is the
validation case, under the slip-wall flux contract, node-centred sensor rows and C6
`:neutral3`. A post-step observer reads only the conserved state during one continuous
`run!` and subtracts transverse station 1 before projecting a line, so a transversely
constant state has exactly zero modal content. `uniformity` is the largest absolute density
difference from the first station; the largest transverse spread and the Fourier amplitude
at each wall-normal node are different norms of the same state. The baseline's dominant
density mode is m = 2, a six-cell transverse wavelength, near the moving shock.

| baseline | |
|---|---:|
| periodic transverse nodes | 12 |
| wall-normal nodes | 100 |
| spacing ratio | 4 |
| final time | 0.6 |
| `uniformity` | 2.052e-7 |
| largest transverse spread | 3.07e-7 |

**The burst is not a measured eigenvalue.** The natural m = 2 component first becomes
wall-local, then undergoes a short increase with its maximum ahead of the shock; later
samples put its maximum within the four-cell shock window:

| step | time | m = 2 density A | location |
|---:|---:|---:|---|
| 1,500 | 0.16802 | 2.372e-11 | shock window |
| 2,500 | 0.29273 | 3.648e-11 | wall |
| 3,000 | 0.35513 | 1.345e-7 | bulk, ahead of shock |
| 3,500 | 0.41748 | 1.985e-7 | shock window |
| 4,500 | 0.54194 | 2.939e-7 | shock window |
| 4,966 | 0.60000 | 1.328e-7 | shock window |

Across the burst interval the apparent rate is `log(1.345e-7 / 3.648e-11) / (0.35513 -
0.29273) = 131.6`, but at the end of that interval the maximum sits outside the four-cell
shock window, 7.3 wall-normal cells from the analytic shock, so the rate is not the growth of
an amplitude continuously localized at the shock. It is a finite-window rate of a
path-dependent transient: the amplitude falls again after the burst, and controlled
perturbations follow different paths. The final spectrum is dominated by a low transverse
mode rather than a single alternating grid mode:

| final density A | m1 | m2 | m3 | m4 |
|---|---:|---:|---:|---:|
| baseline | 3.45e-8 | 1.33e-7 | 2.21e-9 | 6.25e-11 |

**The same wall without a shock does not amplify it.** The two-dimensional `:neutral3` wall
holds round-off and slightly damps a controlled perturbation over the validation horizon,
agreeing with the uniform-state Jacobians of [constant
annihilation](walls.md#constant-annihilation); it certifies nothing about every perturbation of
either state. The control uses the identical strip, filter, artificial properties and slip
wall, filled with the exact planar-Noh post-shock state and holding it at the far Dirichlet
end. A relative density seed multiplies the post-shock density, so its initial amplitude is
four times the seed.

| post-shock state rho = 4, u = 0, p = 4/3 | station variation | initial m = 2 A | final A | gain | finite-horizon rate | transverse-velocity A |
|---|---:|---:|---:|---:|---:|---:|
| unseeded | 1.91e-13 | | 7.96e-14 | | | |
| relative seed 1e-10 | | 4e-10 | 3.446e-10 | 0.861 | −0.25 | 4.36e-15 |

**Controlled amplitude.** A controlled seed grows by a finite gain and not as the constant
exponential growth of an unstable wall eigenmode. The two small seeds take finite gains
differing by about 4% with no positive late-time fitted rate, and the largest seed takes a
much larger gain and generates the m = 4 harmonic. The seed multiplies the initial density
by `1 + a cos(2 pi m x/Lx)` at fixed pressure and velocity. For m = 2:

| initial A | final m = 2 A | final m = 4 A | gain | `log(gain)/0.6` | fitted rate over the second half |
|---:|---:|---:|---:|---:|---:|
| 1e-12 | 6.868e-11 | | 68.7 | 7.05 | -1.22 |
| 1e-10 | 6.594e-9 | | 65.9 | 6.98 | -0.10 |
| 1e-8 | 1.443e-5 | 7.96e-7 | 1,443 | 12.12 | -- |

Extrapolating the small-seed gains does not predict the unseeded baseline: inserting even
the smallest seed changes the round-off history and suppresses the baseline's large burst,
so endpoint changes under an unseeded ablation are not channel causality. The singular
startup sets the natural burst history, and a resolved shock can still amplify an imposed
disturbance. From the smoothed exact profile the unseeded station variation stays three
orders below the cold start's, while the seeded m = 2 run ends with its maximum at the
shock.

| start | unseeded station variation | seeded m = 2 final A | gain |
|---|---:|---:|---:|
| cold singular start (the baseline) | 2.052e-7 | as above | as above |
| smoothed exact profile at t = 0.1, evolved for 0.5 | 1.389e-10 | 3.129e-8 | about 313 |

Changing the transverse point count at fixed spacing changes both the available modes and
the round-off trajectory, with neither a monotone width law nor one selected wavelength.
The nx = 16 row exceeds the nx = 12 validation guard.

| nx | final `uniformity` | dominant mode | location |
|---:|---:|---|---|
| 10 | 3.132e-9 | m = 2 | shock |
| 12 | 2.052e-7 | m = 2 | |
| 16 | 6.932e-7 | m = 2 | |
| 24 | 3.457e-10 | m = 5 | bulk |

Conductivity damps and relocates the controlled small mode: adding it to beta reduces the
matched seeded gain and moves the maximum from the wall to the shock, without identifying
one nonlinear feedback behind the natural burst. Artificial-property comparisons use the
same m = 2, 1e-10 seed so that a different initial round-off realization is not the
comparison:

| active artificial channels | final m = 2 A | gain | `log(gain)/0.6` | location |
|---|---:|---:|---:|---|
| defaults | 6.594e-9 | 65.9 | 6.98 | bulk |
| beta only | 2.221e-8 | 222 | 9.00 | wall |
| beta + mu | 1.810e-8 | 181 | 8.66 | wall |
| beta + kappa | 2.555e-9 | 25.6 | 5.40 | shock |
| defaults with `C_D = 0` | 6.594e-9 | 65.9 | 6.98 | bulk |

`C_D = 0` is bit-identical to the default in this single-species case. The unseeded endpoint
variations under the same ablations are associations with different numerical trajectories.
With the filter off, or with all artificial properties or beta off, the cold run fails with
negative density, so those runs cannot decide whether the removed channel generates or
damps the completed run's mode.

| unseeded | |
|---|---|
| endpoint variation, beta only | 9.78e-11 |
| endpoint variation, beta + mu | 1.57e-10 |
| endpoint variation, beta + kappa | 4.47e-10 |
| filter off | negative density at step 1,335 |
| all artificial properties off, or beta off | negative density near t = 0.074 |

**Later evolution.** The validation endpoint is not saturation. Run on to t = 2.0, the
baseline undergoes a second burst and settles into a bounded oscillatory range, the dominant
mode again m = 2 with its maximum in the shock window. From t = 1.35 the station variation
stays of order 3e-4, about three orders above the validation endpoint. The run is one
continuous `run!` with no landing imposed at the validation time, and the shock stays
inside the domain. It completes in 16,219 steps.

| t | station variation |
|---:|---:|
| 0.604 | 2.13e-7 |
| 1.102 | 1.71e-6 |
| 1.351 | 3.20e-4 |
| 1.475 | 4.24e-4 |
| 2.0 | 2.87e-4 |

The evidence supports a path-sensitive transverse interaction with the captured Noh shock
and the history produced where that shock leaves the wall: the natural disturbance moves
from the wall into the bulk and later the shock window, disappears to round-off in the
identical no-shock strip, depends strongly on cold versus warm startup and on transverse
extent, has approximately linear finite gain for small imposed modes, and develops harmonics
and a higher bounded amplitude when driven farther. No unique nonlinear feedback among the
filter, shock capture and artificial properties is isolated by these measurements.

The validation guard, `uniformity < 5e-7`, sits a factor 2.44 above the current
measurement. It is a deterministic regression envelope for the baseline's preset and
horizon, not a stability bound for another width, an injected disturbance or later
evolution. At a face-centred symmetry plane the same case reads 5.135e-10 and its
guard is 5e-9 ([the face-centred symmetry plane](walls.md#the-face-centred-symmetry-plane)).

## False activation on smooth fields

```text
julia --project=. -t 1 bench/archive/falseactivation.jl                  # every part
julia --project=. -t 1 bench/archive/falseactivation.jl composition thermal acoustic probe   # no battery
julia --project=. -t 1 bench/archive/falseactivation.jl species            # the species-only split
julia --project=. -t 16 bench/archive/falseactivation.jl speciescost       # its cost
julia --project=. -t 16 bench/archive/he_co2_shock_tube.jl 384 24 2.5e-3 detector=species C_D=1 every=5
julia --project=. -t 1 bench/archive/interfacewidth.jl                 # width on several definitions
julia --project=. -t 1 bench/archive/interfacewidth.jl configs ratios=100 only=5,9 N=800
```

The rows labelled `default` in this section and its `δ4` rows are δ⁴ on every sensor at
`C_D = 0.1`, the default before the species-only split replaced it; bench/archive/falseactivation.jl
pins them. Three periodic one-dimensional cases at the production constants, C6 `:neutral3`,
`compact_filter(0.45)` every step, `filter_cfl = 0.35` and `cfl = 0.5`, inviscid. Every
slab edge has a fixed physical width W, so N·W is the number of cells per W. The cases:

```
edge profile   (1 + tanh(x/W))/2 near an edge, W = 1/16
composition    two-gas slab, γ = 1.4, gas constants 1 and 0.2 (density ratio 5), uniform p = T = 1,
               advected at u = 1 for one period; judged on Y_heavy against the exact translation
thermal        one-gas slab of temperature ratio 1.5 at uniform p, advected the same way; judged on T
acoustic       right-running Gaussian pulse p′ = 0.01 exp(−(x/0.05)²) crossing the resting
               composition slab to t = 0.45, when its reflected and transmitted parts are separate;
               judged on p′/0.01 against the art-off run at N = 1024
```

The coefficient columns are the maxima over space and time of D\*/(c h), β\*/(ρ c h) and
κ\*/(ρ c_p c h), each against the grid diffusivity c h. The errors are L1; `on − off` is the
L1 difference from the same run with `enabled = false` and the same filter, which is the
error the channels deposit, and `E off` is the error of that art-off run against the
reference.

```
case         N  N·W | D*/ch    β*/ρch   κ*/ρcpch |  E off     on − off | rates: coefficient  on − off
composition  64   4 | 6.73e-4  8.7e-14  7.94e-5  | 7.21e-5   3.24e-4  |
composition 128   8 | 5.01e-5  1.3e-13  6.57e-6  | 7.13e-7   1.21e-5  |   D* 3.75, κ* 3.60    4.74
composition 256  16 | 3.44e-6  1.5e-13  4.52e-7  | 9.97e-9   5.11e-7  |   D* 3.87, κ* 3.86    4.57
thermal      64   4 | 0        1.23e-6  1.80e-5  | 4.35e-5   5.23e-6  |
thermal     128   8 | 0        5.03e-8  1.52e-6  | 4.96e-7   2.82e-7  |   κ* 3.56, β* 4.61    4.22
thermal     256  16 | 0        7.54e-10 1.06e-7  | 6.18e-9   1.23e-8  |   κ* 3.84, β* 6.06    4.51
acoustic     64   4 | 6.83e-4  2.08e-2  8.37e-5  | 6.01e-3   2.03e-3  |
acoustic    128   8 | 5.41e-5  6.51e-3  7.34e-6  | 2.47e-4   1.89e-4  |   β* 1.68             3.42
acoustic    256  16 | 3.74e-6  1.42e-3  5.20e-7  | 4.08e-6   1.04e-5  |   β* 2.20             4.19
```

**On a resolved composition gradient D\* is the leading error at every resolution.** The
coefficient falls at the δ⁴ sensor's fourth order and its deposit at the rates of the table,
while the scheme without the channels converges at 6.2 to 6.7. The ratio of deposit to
art-off error therefore grows with resolution, from 4.5 on the coarsest grid to 51 on the
finest. `C_D = 0` alone returns the art-off run to round-off (below). κ\* is active on the
same slab at an eighth of D\*, since the internal energy c_v T jumps with the gas constant
at uniform T, and deposits nothing measurable, since its flux reads ∇T, which is zero.

**On a thermal gradient κ\* is the error**: it overtakes the scheme's own error between
N·W = 8 and 16.

**On the acoustic pulse β\* is the error** and falls slowest. It is read from |S|, whose
cusps at the zeros of the strain are grid-scale at any resolution, so the coefficient falls
at 1.7 to 2.2. At N·W = 8 it is still a hundredth of c h. The coefficient and its deposit
are linear in the amplitude, and `C_beta = 0` removes 98% of the deposit. D\* and κ\* sit
on the resting interface at the composition case's values throughout. In every case the
deposit exceeds the art-off error at N·W = 16.

```
acoustic pulse at p′ = 0.001:   β*/ρch 6.3e-4,   on − off 2.1e-5
```

Each factor varied alone on the middle grid, with the state filter's transfer function at
three wavenumbers, its cadence in steps and the mean relaxation weight of a pass:

```
N = 128 (N·W = 8)
row                  | T(π/4) T(π/2) T(3π/4) every  w   | comp: D*/ch  on−off  | therm: κ*/ρcpch on−off | acou: β*/ρch on−off
default              | 1.0000 0.9938 0.8540    1  1.00  | 5.01e-5  1.21e-5     | 1.52e-6  2.82e-7       | 6.51e-3  1.89e-4
C_D = 0              | 1.0000 0.9938 0.8540    1  1.00  | 0        1.8e-15     | 1.52e-6  2.82e-7       | 6.51e-3  1.88e-4
C_beta = 0           | 1.0000 0.9938 0.8540    1  1.00  | 5.01e-5  1.21e-5     | 1.52e-6  2.82e-7       | 0        4.11e-6
C_kappa = 0          | 1.0000 0.9938 0.8540    1  1.00  | 5.01e-5  1.21e-5     | 0        4.9e-15       | 6.51e-3  1.89e-4
C_mu = 0             | 1.0000 0.9938 0.8540    1  1.00  | 5.01e-5  1.21e-5     | 1.52e-6  2.82e-7       | 6.51e-3  1.89e-4
beta :gated_strain   | 1.0000 0.9938 0.8540    1  1.00  | 5.01e-5  1.21e-5     | 1.52e-6  2.82e-7       | 6.57e-3  1.22e-4
beta :dilatation     | 1.0000 0.9938 0.8540    1  1.00  | 5.01e-5  1.21e-5     | 1.52e-6  2.82e-7       | 1.04e-3  4.26e-5
beta :ungated_dil    | 1.0000 0.9938 0.8540    1  1.00  | 5.01e-5  1.21e-5     | 1.52e-6  2.82e-7       | 1.04e-3  5.45e-5
detector :d8         | 1.0000 0.9938 0.8540    1  1.00  | 1.76e-8  4.06e-9     | 4.70e-10 7.32e-11      | 3.51e-3  1.01e-4
smoother :compact    | 1.0000 0.9938 0.8540    1  1.00  | 5.67e-5  1.56e-5     | 1.70e-6  3.91e-7       | 1.04e-2  1.93e-4
species_flux :bulk   | 1.0000 0.9938 0.8540    1  1.00  | 5.01e-5  1.21e-5     | –                      | 6.50e-3  1.93e-4
species_flux :fickian| 1.0000 0.9938 0.8540    1  1.00  | 4.90e-5  1.37e-5     | –                      | 6.51e-3  2.72e-4
filter α = 0.49      | 1.0000 0.9988 0.9654    1  1.00  | 5.01e-5  1.22e-5     | 1.52e-6  2.90e-7       | 6.79e-3  2.03e-4
filter off           | 1      1      1         0  –     | 5.01e-5  1.24e-5     | 1.52e-6  2.94e-7       | 6.81e-3  2.52e-4
cfl 0.25             | 1.0000 0.9938 0.8540    1  0.71  | 5.01e-5  1.21e-5     | 1.52e-6  2.78e-7       | 6.44e-3  1.85e-4
cfl 0.25, filter_cfl 0 | 1.0000 0.9938 0.8540  1  1.00  | 5.01e-5  1.21e-5     | 1.52e-6  2.75e-7       | 6.33e-3  1.81e-4
```

The deposit is linear in the constant of the channel that carries it, and the μ\* channel
carries none of it in one dimension. Of the sensor fields only the β\* settings move
anything, and only on the acoustic case: the dilatation sensor lowers the coefficient
sixfold and the deposit 4.4-fold, since ∇·u carries no absolute value; the compression
switch alone takes a third off. The detector is the largest lever on D\* and κ\*: `:d8`
lowers both coefficients by three to four orders and the deposits by the same, to below
the art-off error, and halves β\* on the pulse. The `:compact` smoother raises every row by
10 to 60%. The species channel changes nothing on a slab at uniform pressure except under
`:fickian`, whose volume flux between unequal gas constants drives a velocity and gives the
composition case a β\* of 5.6e-6. That flux adds 13% to the composition case's deposit and
44% to the pulse's. The filter moves no coefficient and no deposit by more than 5% on the
advected slabs. On the pulse the deposit rises as the filter weakens, by 7% at α = 0.49.
With the filter off it rises by 33%, because the unfiltered state carries more grid-scale
strain for |S| to read. At `cfl = 0.5` the relaxed pass is at full strength. At half that
step it relaxes to the weight in the table, and the deposit is unchanged while the art-off
error grows by 17 to 26%. Setting `filter_cfl = 0` at the same step raises that error a
further 31% on both advected slabs, the per-step dissipation the relaxation removes.

**A contact localization separable from the compression sensing.** Two candidates, each
acting on the κ\* and D\* sensors and leaving β\* as it is. The first, evaluated on stored
states and not run, gates the fourth difference by its ratio R to the total variation over
the same five points. It suppresses the sensor on the smooth slabs but would also halve it
where the channels are wanted, so it cannot leave the battery rows unmoved at the present
constants:

```
R = |δ⁴f| / Σ|Δf|: (kh)³/4 on a resolved sine, 2 on a grid oscillation, 3 beside a one-cell step
gated sensor |δ⁴f| · min(1, R/R₀) at R₀ = 1, share of the peak kept
composition slab, N = 64 / 128 / 256   2.1% / 0.26% / 0.034%   (the same on the internal
                                                                  energy and the thermal slab)
shocked air/SF6 interface              54%
Lax internal energy                    52%
```

The second is the detector split: `:d8` on the internal energy and the mass and mole
fractions, δ⁴ on |S|. The solver has one detector for every field, so the script redefines
the detector dispatch for a `:d8` run (bench only). On the smooth cases it reproduces the
`:d8` row for D\* and κ\* and the δ⁴ row for β\*. On the battery (the columns of [the ringing
detector](#the-ringing-detector), with Noh at `cfl = 0.15` and the spherical origin again at
0.3):

```
config  C_D  | Noh1 plat def | Noh2 plat def | Noh3 plat def | Noh3 @0.3 | Lax L1  contact | Shu tr | WC peak | SI TV−1  min Y  | mix wid | comp. D*/ch  on−off
δ4 *    0.1  | 0.9997  +24%  | 0.9380  +55%  | 0.9774  +29%  |  0.9773   | 5.0e-3  0.0053  | 1.6183 | 6.6068  | 0.0066  −0.0098 | 0.02024 | 5.01e-5  1.21e-5
d8      0.1  | 0.9997  +22%  | 0.9500  +43%  | 0.9983  +37%  |  NaN      | 4.7e-3  0.0044  | 1.6369 | 6.4182  | 0.0707  −0.0129 | 0.01789 | 1.76e-8  4.06e-9
split   0.1  | 0.9997  +26%  | 0.9378  +59%  | 0.9772  +29%  |  0.9771   | 4.9e-3  0.0053  | 1.6204 | 6.5914  | 0.0719  −0.0104 | 0.01789 | 1.76e-8  4.06e-9
split   0.3  |                                                                                           | 0.0278  −0.0092 | 0.01796 | 5.27e-8  1.22e-8
split   1    |                                                                                           | 0.0046  −0.0047 | 0.01817 | 1.76e-7  4.07e-8
split   3    |                                                                                           | 0.0047  −0.0032 | 0.01868 | 5.32e-7  1.22e-7
```

The split keeps what the full `:d8` loses at the origin: the spherical plateau and the
`cfl = 0.3` completion are those of δ⁴ to the fourth digit, so the origin ceiling belongs to
β\*'s detector. The κ\* half of the split moves the single-species rows, two to four points
of wall heating on the planar wall and the axis, 0.2% on the Woodward–Colella peak and
0.1% on the Shu–Osher train; the
D\* half cannot, since it is not computed without species. At `C_D = 0.1` the split
detector passes the period-four trail behind the shocked interface that δ⁴ damps, and TV − 1
rises tenfold. At `C_D = 1` the split holds the shocked interface
tighter than the default in both TV − 1 and worst Y and narrows the passive width by 10%.
On the smooth composition slab it deposits 300 times less than the default. A D\*-only
split, δ⁴ on every field but the mass and mole fractions, would leave every single-species
row bit-identical by construction and carry the whole composition-slab result, since κ\*
deposits nothing there; the thermal slab's κ\* error would remain. It is measured below.

### The species-only split

`:d8` on the mass and mole fractions, δ⁴ on every other field (bench/archive/detector_split.jl,
selected by `SPECIES_ONLY[]`). Under the default `:partial_density` channel those two
fields are sensed only inside `bulk_diffusivity!`, so the redefinition brackets that call;
the Lax tube under the split is the δ⁴ run bit for bit (max |Δρ| = 0), as every
single-species row is by construction. C_D is swept on the species rows: the shocked
air/SF6 interface (the ringing measure, the worst Y over the run and the final width in
cells), the same interface and the Brill slab (worst Y and the pressure error) at density
ratio 100, the passive advected width, and the composition slab at two resolutions:

```
config   C_D | SF6: TV−1  min Y  wid | R100: TV−1  min Y  wid steps | slab100: min Y  |p−1| | mix wid | comp N·W=4 D*/ch on−off | N·W=8 D*/ch on−off
δ4 *     0.1 | 0.0066  −0.0098   4   | 0.3243  −0.0189   7   704    | −0.0620  5.2e-11      | 0.02024 | 6.73e-4  3.24e-4        | 5.01e-5  1.21e-5
species  0.1 | 0.0720  −0.0109   4   | 0.5094  −0.0186   7   709    | −0.0628  4.4e-11      | 0.01789 | 7.74e-6  2.44e-6        | 1.76e-8  4.06e-9
species  0.3 | 0.0300  −0.0097   4   | 0.1466  −0.0168   8   695    | −0.0599  4.9e-11      | 0.01796 | 2.31e-5  7.29e-6        | 5.27e-8  1.22e-8
species  1   | 0.0045  −0.0046   5   | 0.0020  −0.0092  10   696    | −0.0472  6.1e-11      | 0.01817 | 7.58e-5  2.40e-5        | 1.76e-7  4.07e-8
species  3   | 0.0046  −0.0032   8   | 0.0021  −0.0041  11   704    | −0.0325  5.5e-11      | 0.01868 | 2.16e-4  6.99e-5        | 5.32e-7  1.22e-7
```

The SF6 row and the passive width at `C_D = 1` repeat the energy-and-species split's to
the second digit, so κ\*'s detector plays no part in them. At density ratio 100 the δ⁴
default leaves a strong ringing behind the shocked interface. The split at `C_D = 1` removes
nearly all of it and halves the worst mass fraction, at the cost of a wider interface; the
Brill slab's worst Y improves as well, and its pressure stays at round-off. TV − 1 is flat
from `C_D = 1` to `C_D = 3` on both shocked rows while the width and the smooth deposit keep
growing, so `C_D = 1` is the refit. It deposits 13 times less than the default on the
composition slab at the coarser resolution and 300 times less at the finer.

The split at `C_D = 1` is the default, `detector = :species_d8`, and applies to all three
species channels. The other two channels on the `shock_interface` and `brill_slab` cases of
test/cases.jl, the default against `detector = :delta4, C_D = 0.1` (V 5–95 is the
volume-fraction width of `volume_width`, in cells):

```
channel     R    config   | TV−1    min Y   count V 5–95 steps | Brill slab p error
:bulk       5.04 δ4 0.1   | 0.0060  −0.0088   4   4.78   641   |
                 split 1  | 0.0053  −0.0042   5   4.85   637   |
:bulk       100  δ4 0.1   | 0.0861  −0.0171   8   4.22   677   |
                 split 1  | 0.0195  −0.0091   8   4.81   673   |
:fickian    5.04 δ4 0.1   | 0.0215  −0.0115   4   4.53   646   | 3.98e-2
                 split 1  | 0.0073  −0.0080   5   5.03   644   | 5.54e-2
                 δ4 1     | 0.0060  −0.0054   8   6.28   643   |
```

Under `:fickian` the split cuts the ringing threefold at a width between the two δ⁴ rows
and raises the enthalpy-flux pressure error on the slab by 39%, the same proportionality to
the diffusivity as a larger `C_D`. The serial suite guards the high-ratio interface under
both consistent channels on the volume-fraction width, at most 7 cells, and on the ringing
measure, below 0.04, since `width_cells` counts the heavy-gas tail there.

The He/CO2 tube of bench/archive/he_co2_shock_tube.jl, in two dimensions with a 10:1 driver,
compares δ⁴ at `C_D = 0.1` with the split at `C_D = 1` on a coarse and a fine grid. The
incident shock strikes the perturbed contact between 0.5 and 1.0 ms. Worst Y is the most
negative mass fraction over every fifth step; the integral width is ∫ Ȳ(1 − Ȳ) dx of the
plane-averaged CO2 fraction; TV − 1 and the width in cells are read on the line y = 0
through the contact's leading point:

```
grid     config        | worst Y   | ∫Ȳ(1−Ȳ)dx mm at 1.0 1.5 2.0 2.5 ms | TV−1 (y=0) at 1.0 1.5 2.0 2.5 ms | wid 2.5 ms | steps
384x24   δ4 C_D=0.1    | −3.29e-3  | 18.51 32.60 46.17 56.73            | 0.0178 0.0224 0.0208 0.0606      |  6         | 1684
384x24   species C_D=1 | −2.28e-3  | 18.66 33.00 46.90 57.66            | 0.0048 0.0137 0.0241 0.0604      |  6         | 1645
768x48   δ4 C_D=0.1    | −3.04e-3  | 17.80 31.35 43.33 53.66            | 0.0181 0.0184 0.0165 0.0175      |  7         | 3499
768x48   species C_D=1 | −2.34e-3  | 17.88 31.75 43.99 54.35            | 0.0175 0.0196 0.0168 0.0230      |  7         | 3434
```

The split lowers the worst undershoot by 23 to 31%. It widens the integral mixing width by
0.8 to 1.3%, a fifth of what halving the spacing moves it by. The ringing on the sampled
line is the same within the spread between the two grids. On the line through the trailing
point the ringing measure reaches 0.11 to 0.31 at the last sample under both detectors,
where the rolled-up contact crosses the line more than once, so it measures the roll-up and
is left out.

The cost, `compute_artificial!` and `compute_rhs!` as the minimum over 30 repetitions
interleaved in one process at `-t 16`: the split adds the two pentadiagonal solves per
direction of the Y₁ and X₁ detectors.

```
case                       | artificial ms  rhs ms | ratio to δ4: art  rhs
tube 512x32, 2 species δ4  |  0.586          2.344 |  1.000  1.000
               species     |  0.812          2.531 |  1.386  1.080
               :d8         |  1.047          2.684 |  1.788  1.145
box 64³, 2 species     δ4  |  7.710         25.639 |  1.000  1.000
               species     |  8.411         26.773 |  1.091  1.044
               :d8         |  9.165         27.260 |  1.189  1.063
```

The right-hand side costs 4 to 8% more, about half of what `:d8` on every field adds.

### Interface width against Brill, Olson and Bokman

[Brill, Olson & Bokman (J. Comput. Phys. 542,
2025)](https://doi.org/10.1016/j.jcp.2025.114366) report no interface thickness in cells.
Section, equation and figure numbers here are those of arXiv:2503.12680v2. None of their
tests is a planar shocked interface:

```
test                                        section, figures         density ratio
three-material bubble advected for ten      Secs. 3.1, 5.1;          R = 10 to 1e6
  periods: completion and final pressure      Figs. 2, 6
  error against the initial width N_p
  (99% of a tanh inside N_p cells)
shock–helium bubble                         Sec. 5.2; Figs. 7, 8     R = 7.25 with N_p = 14;
                                                                     R ≈ 7.25e5
Rayleigh–Taylor layer                       Sec. 5.3; Fig. 9         R = 5000
two drops in a reversing shear              Sec. 5.4; Figs. 10, 11
nitrogen bubble in water                    Sec. 5.5
```

The thin interfaces of the figures are held by the conservative diffuse-interface flux of
their eq. 68, whose one-dimensional equilibrium is a logistic profile of width ε, with
ε = Δ by default. Their species diffusivity (eqs. 33–37) is c_s Δ times the
eighth-derivative ringing of Y_i and V_i, maximized over the two, which is the
`:partial_density` form here. If Miranda's F (eq. 22) is Pyranda's `c10d8` operator, their
constants convert to C_D under `:d8` here by the ratio of the two operators' responses to a
grid-to-grid oscillation:

```
equilibrium profile       V = 1/(1 + exp(−x/ε)); ε = 1.2Δ in the R ≈ 7.25e5 shock–bubble,
                          1.5Δ in the water case
oscillation response      Pyranda's operator 3840, compact_d8 16
their C_D,V = 2e-4        C_D = 0.048 under :d8 here
shocked high-ratio runs   2.4 (R ≈ 7.25e5), 4.8 (Rayleigh–Taylor), 288 (water)
```

On a logistic V profile the mass fraction is a logistic of the same ε displaced by ε ln R
(their appendix A), so Y and V have the same 5–95% width. The `width_cells` of
`shock_interface` counts the points inside the same band of Y_air, which at high density
ratio covers only the light side of the interface:

```
5–95% width of Y and of V      2ε ln 19 = 5.9ε
10–90% width                   2ε ln 9 = 4.4ε
maximum-gradient thickness     1/max|∂V/∂x| = 4ε
width_cells                    0.05 < Y_air < 0.95, at R = 100 0.84 < V_air < 0.9995
```

`bench/archive/interfacewidth.jl` reads the final profile of `shock_interface` on each definition:
`count` is `width_cells`, `core` the same count on the contiguous run through Y_air = 1/2,
and the interpolated widths are in cells. Reference profiles, evaluated in closed form:

```
profile                              | Y 5–95  V 5–95  V 10–90  V grad
equilibrium of eq. 68, ε = Δ         |  5.9     5.9     4.4      4.0
                       ε = 1.2Δ      |  7.1     7.1     5.3      4.8
                       ε = 1.5Δ      |  8.8     8.8     6.6      6.0
their eq. 30 tanh, N_p = 7           |  3.9     3.9     2.9      2.6
                   N_p = 14          |  7.7     7.7     5.8      5.3
their eqs. 82, 85 erf(x/2Δ)          |  –       4.7     3.6      3.5
```

The initial interface of `shock_interface`, a tanh of Y over 2h, is the ε = Δ equilibrium
(measured row `initial` below, which carries the bias of linear interpolation). The final
profiles at t = 0.25 on N = 400 (`configs`) follow, `*` marking the default. In the
configurations `species` is the species-only split of bench/archive/detector_split.jl at the C_D
given, `d8` the detector on every field, and `c_air` and `c_hvy` a uniform sound speed in
D_b alone, the pre-shock value of either gas at R = 100:

```
c_air = √1.4, c_hvy = √(1.09/100)

config           R     | count core | Y 5–95 V 5–95 V10–90 V grad |  TV−1    min Y  steps
initial          100   |   6    6   |  6.09   6.06   4.51   4.14  | 0        0          0
δ4 0.1  *        5.04  |   4    4   |  4.79   4.78   3.88   3.80  | 0.0066  −0.0098   644
δ4 1             5.04  |   8    8   |  8.22   7.47   5.84   5.30  | 0.0043  −0.0028   638
species 0.05     5.04  |   4    4   |  3.98   4.27   3.33   3.42  | 0.0750  −0.0108   646
species 0.3      5.04  |   4    4   |  4.55   4.56   3.62   3.63  | 0.0300  −0.0097   643
species 1        5.04  |   5    5   |  5.82   5.25   4.19   3.81  | 0.0045  −0.0046   640
species 3        5.04  |   8    8   |  7.31   6.42   4.89   4.30  | 0.0046  −0.0032   640
d8 1             5.04  |   6    6   |  5.93   5.40   4.30   3.85  | 0.0044  −0.0054   621
bulk δ4 0.1      5.04  |   4    4   |  4.90   4.78   3.87   3.89  | 0.0060  −0.0088   641
δ4 0.1  *        100   |   7    7   |  7.78   5.24   2.69   2.08  | 0.3243  −0.0189   704
δ4 1             100   |  10   10   | 10.47   5.57   4.04   3.75  | 0.0038  −0.0066   694
species 0.05     100   |   6    4   |  4.65   3.49   2.43   1.82  | 0.5235  −0.0197   716
species 0.3      100   |   8    8   |  8.31   5.10   2.75   2.26  | 0.1466  −0.0168   695
species 1        100   |  10   10   | 10.32   5.27   3.63   3.05  | 0.0020  −0.0092   696
species 3        100   |  11   11   | 11.61   5.81   4.16   4.07  | 0.0021  −0.0041   704
d8 1             100   |  10   10   | 10.33   5.22   3.68   2.94  | 0.0014  −0.0090   670
bulk δ4 0.1      100   |   8    8   |  8.14   4.22   2.84   2.59  | 0.0861  −0.0171   677
species 1 c_air  100   |   9    9   |  9.14   6.48   4.53   4.42  | 0.0150  −0.0097   688
species 1 c_hvy  100   |   8    8   |  8.49   5.15   3.49   2.72  | 0.1347  −0.0594   680
species 10 c_hvy 100   |   9    9   |  9.04   6.22   4.45   4.34  | 0.0301  −0.0305   684
δ4 0.1  *        1000  | lost positivity
δ4 1             1000  |  10   10   | 10.84   4.77   4.45   1.95  | 0.0520  −0.0141   702
species 1        1000  |  11   11   | 10.22   4.71   4.42   1.51  | 0.2046  −0.0183   709
species 3        1000  |  13   13   | 12.27   7.70   7.05   2.05  | 0.0662  −0.0146   720
d8 1             1000  |  10   10   | 10.20   4.73   4.43   1.67  | 0.2299  −0.0181   683
bulk δ4 0.1      1000  | lost positivity
N = 800:
δ4 0.1  *        100   |   8    8   |  8.71   3.77   2.52   2.20  | 0.1755  −0.0195  1408
species 1        100   |  11   11   | 11.41   6.97   4.64   3.65  | 0.0010  −0.0083  1398
species 1 c_air  100   |  10   10   | 10.02   8.47   5.91   5.51  | 0.0250  −0.0086  1386
```

And against time at R = 100 (`history`; the shock reaches the interface near t = 0.113):

```
config     t    | count | Y 5–95 V 5–95 V10–90 V grad |  TV−1
δ4 0.1  *  0.10 |   7   |  5.31   5.67   4.64   3.89  | 0.2731
           0.15 |   5   |  3.99   4.54   3.72   1.30  | 0.4195
           0.20 |   6   |  4.59   2.75   1.85   1.49  | 0.4869
           0.25 |   7   |  7.78   5.24   2.69   2.08  | 0.3243
           0.35 |   9   |  8.50   5.86   2.05   1.74  | 0.1564
species 1  0.10 |   7   |  5.53   5.71   4.67   4.42  | 0.1736
           0.15 |   7   |  7.26   4.58   3.76   2.22  | 0.0161
           0.20 |   9   |  9.16   5.26   3.89   2.41  | 0.0053
           0.25 |  10   | 10.32   5.27   3.63   3.05  | 0.0020
           0.35 |  11   | 11.13   6.65   4.53   2.86  | 0.0034
```

**In volume fraction the interfaces at R = 100 are as thin as Brill's sharpened one.** The
rows that do not ring, C_D = 1 under either detector or both and the split at the largest
C_D, are slightly narrower in V than the ε = Δ equilibrium on both interpolated widths. The
δ⁴ default has the same 5–95% width around a steeper core. The ringing accompanies a core
compressed below about three cells, while the eq. 68 equilibrium holds the core at four
cells at ε = Δ:

```
rows of the table above      maximum-gradient thickness
TV − 1 above 0.08            below 2.8 cells
TV − 1 below 0.03            above 2.9 cells
```

The ten cells of `width_cells` under the split are a tail of heavy gas on the air side.
In the split's profile V_air changes over ten cells towards the air and over three towards
the heavy gas, and `width_cells` measures only the first. A logistic of the
same 5–95% volume-fraction width would count five. Part of the asymmetry is the sound speed
in D_b = c·G[·], √R larger in the light gas. With a uniform c the count falls by one or
two, but the ringing returns, most strongly at the heavy gas's value and still at ten times
the constant. Brill's D\* carries the local c_s as well, so the same tail is expected under
their method without eq. 68.

```
split, R = 100: V_air from 0.83 to 0.9995 towards the air, from 0.83 to 0.08 towards the heavy gas
```

The widths are set in cells: at N = 800 the counts grow by one cell and the physical width
halves. Before the shock both rows hold the initial width within half a cell. The shock
compresses the interface, and under the split the count then grows through the run, while
the volume-fraction width changes by under a cell until t = 0.25. At R = 1000 the δ⁴
default and the bulk channel lose positivity, and the other rows complete with the ringing
back and a core of under three cells. A mass-fraction count of six at R = 100 needs the
tail removed, which the equilibrium of an eq. 68 flux does and no constant of the present
channel does.

## The ablation rate of the Briareus-driven capsule

```text
julia --project=<env> -t 4 examples/briareus_capsule.jl nx=1400 tfinal=10.5 rate=<A>
```

`<env>` is an environment with both CompactLES and Briareus (the script's header). In the
one-dimensional stage-0 capsule of the example, ṁ = A T³ (A in g cm⁻² s⁻¹), the ablation
pressure is calibrated separately to the 145 Mbar peak, and the carbon is an ideal gas. The
HYBRID-E targets and the provenance of the rows, the variants and the calibration table on
the [ablation-drive page](../../docs/src/explanation/ablation-drive.md):

```
HYBRID-E targets (PRE 109 Table I and HYDRA)
peak velocity                    380–400 km/s
ablator remaining at the peak    5–6.6 %

rows and grid runs    Briareus research notes, "Capsule drive hand-over and CompactLES stage 0"
variants              "Where the stagnation lag comes from" (2026-10-27)
roadmap               item A22, commits 0911282 and df8b336
calibration table     the run of 2026-10-24 against Briareus 785684d
```

| A | origin of A | peak velocity | ablator remaining | stagnation after bang |
|---|---|---|---|---|
| 2.85e5 | fitted to the remaining mass | 340 km/s | in the 5–6.6 % band | 0.6–0.7 ns |
| 3.5e5 | | 386 km/s | 1.45 % | 0.16 ns |
| 4.15e5 | Olson 2011 planar HDC, low end | | none | |
| 5.0e5 | Olson 2011 planar HDC | run fails at 7.33 ns | 0.11 % | |

Both measured planar HDC rates in the table over-ablate this shell; Olson's rates span
A ≈ 4.15–5.2e5. At the fitted A only the example's grid completes, the coarser and the
finer grid failing on `dt_collapse`, so the velocity is read to ±10 %.

```
A = 2.85e5    nx = 700 and 2000 fail, nx = 1400 completes
```

No fitted A gives the velocity, the remaining mass and the timing together with ideal-gas
carbon. Of eight variants each changing one closure, a stronger foot or stiff carbon brings
the first shock to the keyhole timing, but slows the shell, leaves more of the ablator and
makes the shell stagnate later. Only more ablation closes the late lag (the 3.5e5 row).

```
stronger foot or stiff carbon (γ = 3)
first shock against the keyhole timing    3.4–3.7 ns against 3.86–3.91
peak velocity                             301 km/s
ablator left                              7–9 %
stagnation                                10.4–10.5 ns
```

The example has no keyword for the variants. Its run is stopped once the velocity has fallen
below four fifths of its peak, so no stagnation time is printed; the stagnation column and the
variants' timings come from the runs of the notes.
