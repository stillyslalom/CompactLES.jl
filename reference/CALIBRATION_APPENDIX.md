# Calibration appendix: the measurements

This file is the record behind [CALIBRATION.md](CALIBRATION.md): every sweep, ladder and
control run that produced the defaults there, including the explanations that were measured
and rejected and the results that moved nothing. Read CALIBRATION.md for the defaults and
for which setting to change; read this file before moving a default, before re-deriving a
rejected explanation, and when a number in a recommendation needs its provenance. The open
items are listed there too, one line each.

One section per instrument. Each section opens with the command that reproduces it, and each
table names the sweep and the background it ran under; a table whose background is not the
current default is labelled with the setting it was taken under and kept as the record of
that setting. A quantity is recorded once, at its current value: where a number moved, the
section that moved it says so in one sentence and the older figure is gone.

## Contents

1. [The shock battery](#the-shock-battery) (`bench/artcal.jl`, `test/cases.jl`)
2. [Taylor-Green](#taylor-green) (`bench/tgv_energy.jl`, `bench/tgv_spectrum.jl`)
3. [The filter's dissipation](#the-filters-dissipation) (`bench/filterrate.jl`)
4. [Filtering on non-uniform volumes](#filtering-on-non-uniform-volumes)
   (`bench/filter_conservation.jl`)
5. [The CFL restriction and the symmetry cell](#the-cfl-restriction-and-the-symmetry-cell)
   (`bench/nohprobe.jl`)
6. [Directional bulk viscosity](#directional-bulk-viscosity) (`bench/anisotropic.jl`)
7. [The smooth-evolution accuracy matrix](#the-smooth-evolution-accuracy-matrix)
   (`bench/boundaryorder.jl`, `bench/temporalorder.jl`, `test/convergence.jl`)
8. [The filter's wall rows](#the-filters-wall-rows) (`bench/wallfilter.jl`)
9. [Wall closures in production](#wall-closures-in-production) (`bench/wallclosure.jl`)
10. [The wall flux contracts](#the-wall-flux-contracts) (`test/wall_flux_tests.jl`)
11. [Constant annihilation](#constant-annihilation) (`bench/constantfloor.jl`)
12. [Closure certificates](#closure-certificates) (`bench/closurecertify.jl`,
    `bench/neutralsearch8.jl`, `bench/neutralsearch10.jl`)
13. [The fifth-order closure search](#the-fifth-order-closure-search)
    (`bench/closuresearch.jl`, `bench/closuredamping.jl`)
14. [The sensor operators at walls](#the-sensor-operators-at-walls) (`bench/sensorwall.jl`)
15. [The face-centred symmetry plane](#the-face-centred-symmetry-plane)
    (`bench/wallclosure.jl`, `test/mpi_tests.jl`)
16. [The aligned Noh transverse mode](#the-aligned-noh-transverse-mode)
    (`bench/noh_transverse.jl`)
17. [The inflow transverse terms](#the-inflow-transverse-terms) (`bench/nscbcinflow.jl`)
18. [Fold order and geometry limits](#fold-order-and-geometry-limits) (`bench/foldorder.jl`)
19. [Operator and step cost](#operator-and-step-cost) (`bench/derivcost.jl`,
    `bench/phases.jl`)
20. [AMR](#amr) (`bench/amr_transfer.jl`, `bench/leveltransfer.jl`,
    `test/level_tests.jl`)
21. [Temperature-dependent transport](#temperature-dependent-transport)
    (`test/transport_tests.jl`, `test/transport_integration_tests.jl`)
22. [The bulk species channel in three dimensions](#the-bulk-species-channel-in-three-dimensions)
    (`bench/bulkchannel.jl`, `bench/bulkentropy.jl`)
23. [The species validity band](#the-species-validity-band) (`bench/speciesband.jl`)

## The shock battery

```text
julia --project=. -t 1 bench/artcal.jl <part>   # beta kappa D Y detector field filter bulk brill2025 response
julia --project=. -t 16 test/validation.jl
```

The one-dimensional cases are defined in `test/cases.jl` and shared with
`test/validation.jl`. Every grid here is uniform and Cartesian, or has its angular
directions collapsed, so the physical and computational spacings coincide.

**ν is the Noh geometry index**: ν = 1 planar (a slip wall), ν = 2 the cylindrical axis
fold, ν = 3 the spherical origin fold, one more than the exponent in the r^(ν−1) area
weight; the exact post-shock compression is 4^ν at γ = 5/3. **Noh plat/exact** is the
post-shock density plateau over its exact value (4, 16, 64), exact ratio 1.0000. **deficit**
is the density shortfall at the symmetry point, which is wall heating. **Lax L1 / contact**
are the mean density error against the exact Riemann solution and the 10–90% contact width,
which measures the broadening from the regularization. **Shu train amp** is the
peak-to-trough density in the post-shock entropy wave train, which pulls against every
damping constant. **WC peak** is the peak density in the Woodward–Colella collision, a
survival check at a 10⁵ pressure ratio. **NaN** is a positivity loss or a stall; every case
carries an `nmax`, and `StepControl` raises `SolverFailure` about 150 steps before the
collapse.

Every constant table is one-dimensional. `C_mu` multiplies the shear component of the
artificial stress, which these cases reach only through the stress trace, so its table
constrains stability and not shear accuracy and [Taylor-Green](#taylor-green) carries the
shear measurement; `C_beta`, `C_kappa` and `C_D` are fully exercised. The constants were
fitted under `compact_filter(0.45)` at full strength every step, and none of them moves
under the relaxed default, measured below.

### The current reading

`test/validation.jl` at `-t 16` under the current defaults (C6 `:neutral3`,
`compact_filter(0.45, closures = :onesided)`, `filter_cfl = 0.35`, `smoother = :gaussian`,
`detector = :delta4`, the node-centred sensor rows and the slip-wall flux contract). The
second column places `SymmetryPlaneBC` where the `SlipWallBC` was ([the face-centred
symmetry plane](#the-face-centred-symmetry-plane)):

```
case                                   node-centred wall            symmetry plane
Woodward–Colella N=800, t=0.038  L1 3.2153e-2, peak 6.6166 @0.7785  L1 3.0330e-2, 6.6140 @0.7781
Noh ν=1 cold N=400, cfl 0.15     3.9988, 23.9%, shock 0.2021,       3.9990, 25.0%, 0.2024,
                                 pre-shock L1 3.603e-6, 6 cells     2.953e-6, 7 cells
                                 e < 0, e_min −0.0315               e_min −0.0047
Noh aligned N=100, AR=4          4.0035, 32.5%, 0.2084, 4966 steps, 3.9974, 27.8%, 0.2093,
                                 transverse 2.052e-7                4938 steps, 5.135e-10
Noh ν=2                          15.0086, 55%, 0.2091               –
Noh ν=3                          62.5547, 29%, 0.2089               –
Noh plane AR=2 (four inflows)    11.854, 0.236, L1 0.890, 750 steps –
Lax                              L1 ρ 4.987e-3, u 7.467e-3, p 7.556e-3, contact 0.0053
Shu–Osher                        L1 ρ 6.804e-3, train L1 2.087e-2, train peak 4.6800
Sedov                            R_s 0.8085 (+1.06%), peak ρ 5.127, e_min −0.00427
shock/SF6 interface              worst Y −0.0129 / 1.0129, width 4 cells, 647 steps
```

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
limits the accuracy measures vary monotonically: the wave train loses 2.8% of its amplitude
over 0.25 to 4 and the ν = 3 plateau crosses exact between 0.25 and 0.5. Accuracy alone
gives an optimum near 0.4; the default of 1.0 buys a robustness margin at the cost of an 18%
wider contact and 0.7% lower wave-train amplitude relative to 0.5.

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
and the 1/T factor does not collapse `dt` in a cold ambient. Planar Noh at cfl 0.15, N =
400, the limiting rate sampled every 100 steps:

```
p0      steps   plateau   deficit  shock    inadmissible  e_min/e0    dt median  dt min
1e-2    3718    3.8900    +63%     0.2119    2            -0.66       1.64e-4    1.51e-4
1e-3    3655    3.9858    +63%     0.2052    4            -28         1.65e-4    1.48e-4
1e-4    3650    3.9959    +63%     0.2046    7            -255        1.65e-4    1.55e-4
1e-5    3652    3.9969    +63%     0.2045   10            -2.4e3      1.66e-4    1.12e-4
1e-6    3650    3.9970    +63%     0.2045   11            -2.3e4      1.65e-4    1.55e-4
1e-8    3652    3.9971    +63%     0.2045   17            -2.2e6      1.65e-4    1.55e-4
```

The step count and the median step are unchanged over six decades of ambient pressure. The
limiter is the diffusive rate on about 85% of the samples, and on the line it is the bulk
viscosity: at step 2000 the largest `(μ* + β*)/(ρh²)` is 262 against a largest `κ*/(ρ c_v
h²)` of 14.6 and an acoustic rate of 488, at p₀ = 1e-4 and 1e-8 alike. `C_kappa = 0` leaves
the step count at 3580 at both pressures and `C_kappa = 0.1` makes κ\* the limiter and
lengthens the run to 4330 steps, again at both. The singular factor is not reached: at a
cell whose internal energy is negative, `primitives!` floors T_ion at 1e-300, the sound
speed vanishes with it, and ρc/T_ion evaluates to about 1e-140, so κ\* is zero on those
cells rather than large. A cold ambient changes the count of inadmissible cells, because the
precursor's negative internal energy is a fixed absolute amplitude of about −0.035 and the
ambient internal energy falls beneath it. `artificial_conductivity_scale` remains the EOS
dispatch point where a tabular or condensed-matter model sets its own scale.

### C_D, the species diffusivity

A sharp binary interface advected at u = 1 for t = 0.5 on 256 points, initial 10–90% width
2h = 0.0078, on the periodic slab with two such edges (`bench/artcal.jl D`):

```
C_D       | interface width
0         |  0.01786
0.0025    |  0.01795
0.01      |  0.01819
0.03      |  0.01877
0.04      |  0.01903
0.1    *  |  0.02024
0.16      |  0.02112
0.3       |  0.02254
```

With D\* disabled the interface still more than doubles in width, so the compact filter is
the dominant source of broadening for a passive interface. At `n_species == 2` the
per-species sensor machinery is a measurable no-op (`D*_1` and `D*_2` agree to 4.8e-16).

The passive width does not measure what C_D is for. Behind a shocked interface the mass
fractions ring inside [0, 1], where the bound of the next section is zero, and C_D is the
only damping. The measure is the total variation of the final profile beyond the 1 a
monotone one carries (`TV − 1`), with the grid-scale content |δ⁴Y|/16 on the light side
more than four cells from the interface. On the shocked air/SF6 interface of the next
section, under `species_flux = :fickian` and the other constants at their defaults:

```
C_D       | 2h: TV − 1   light-side   worst min Y   steps | 4h: TV − 1   worst min Y
0.01      |     0.0788   2.2e-3       -0.0129       647   |     0.0673   -0.0026
0.02      |     0.0781   2.2e-3       -0.0130       647   |     0.0651   -0.0025
0.03      |     0.0766   2.2e-3       -0.0129       647   |     0.0646   -0.0025
0.05      |     0.0691   2.1e-3       -0.0127       646   |     0.0609   -0.0023
0.07      |     0.0495   1.8e-3       -0.0115       646   |     0.0396   -0.0022
0.1    *  |     0.0215   1.5e-3       -0.0115       646   |     0.0194   -0.0021
0.3       |     0.0083   1.5e-3       -0.0084       644   |     0.0071   -0.0014
```

The excess at 0.01 is a two-cell lump of the heavy gas, Y ≈ 0.08, separated from the
interface by one cell of Y ≈ 0 on the light side; the 4h initial interface shows a similar
excess. The lump is also present on the one-dimensional reduction of `examples/shock_tube.jl`
(He/CO2, NASA-9 thermodynamics, 768 cells, flat interface, to 2.5 ms), where the time
mean of TV − 1 after the shock crosses is 0.113 at 0.01, 0.013 at 0.03 and 0.05, 0.011 at
0.1 and 0.003 at 0.3, in 2000, 1997, 1995, 1992 and 1981 steps. On that case the bulk
channel (0.114) and a calorically perfect EOS (0.094) leave it where it is, and the
commit that introduced the bound gives 0.094 on the perfect-gas form, so it predates
every later change (these rows and the next are all Fickian). What remains at 0.1 is a trail of period four cells on the light
side, growing toward the interface to a few 1e-3 in Y, ρ and p, which only 0.3 reduces
(fivefold); the compact filter passes that wavelength and δ⁴ responds to it at a quarter
of its grid-oscillation weight. On the two-dimensional example at half resolution
(384 × 24) the worst undershoot over the run falls from −1.0e-2 to −3e-3 at 0.1 and
−1.8e-3 at 0.3, and the steps to 2.25 ms from 1859 to 1591 and 1530.

Pyranda's own decks (`shockBubble.py`, `triplePoint.py`) write the species diffusivity as
2e-4 ρ h² ring(Y)/Δt, with Δt = h/(|u| + c) at their CFL of 1 and ring = 240 × this
code's `compact_d8`, which is C_D ≈ 0.05 on `:d8` scaled by |u| + c in place of c, about
0.07 behind the shock in the He/CO2 tube; `RM3D.py` and `RT3D.py` use half that. Under
`:d8` at 0.05 the He/CO2 tube reads 0.022, against 0.19 at 0.01. The Pyranda comparison deck
(`bench/he_co2_shock_tube.jl`) carried this code's former 0.01, so both codes rang alike
there.

**Recommendation:** 0.1, the lowest value that removes the lump on both cases. It costs
11% on the passive width and no steps, and under `:fickian` it raises the interface pressure
error tenfold, which is one reason the default channel changed (see [the partial-density
species channel](#the-partial-density-species-channel)).

### C_Y, the mass-fraction bound

A shocked species interface rings. Measured on a Mach 1.5 shock in air (γ = 1.4, R = 1)
running into a tanh interface with SF6 (γ = 1.09, density 5.04) on 400 points, Dirichlet
ends, `cfl = 0.4`, default `ArtParams`:

```
initial interface | worst Y          | final Y range     | width (cells)
2h                | -0.204 / +1.204  | -0.009 / +1.098   | 6 → 3
4h                | -0.023 / +1.023  | -0.0004 / +1.021  | 12 → 4
8h                | clean            | clean             | 14
```

The shock compresses the interface by the density ratio across it, and the ringing is a
two-cell odd-even train on the light side set by the cells the interface spans after
compression. Nothing else varied moves it: N = 800 and 1600 at a fixed 2h interface,
`closures = :onesided`, `beta_sensor = :dilatation`, `detector = :d8`, `cfl = 0.2` and
`compact_filter(0.3)` all leave the worst excursion between −0.18 and −0.29. `C_D = 1`, a
hundred times Cook, reaches −0.008, and disabling the artificial properties gives −0.43.
Cook's D\* = C_D c h |δ⁴Y| peaks near 2e-5 in the train, a diffusive time across a cell of
order one against a shock crossing of 0.005, so it cannot hold an interface the shock has
thinned.

The species diffusivity therefore carries the mass-fraction bound of [Cook (2007, eq.
18)](https://doi.org/10.1063/1.2728937), zero wherever 0 ≤ Y ≤ 1:

    D*_k = c · G[ max( C_D h |δ⁴Y_k| , C_Y h max(0, −Y_k, Y_k − 1) ) ]

with G the sensor smoother. Cook used `C_Y = 100`; his later form [(2009, appendix A, eq.
A8)](https://doi.org/10.1063/1.3139305) writes the bound as (|Y| − 1 + |1 − Y|), twice the
max, with `C_Y = 50`, and both Cook forms use the Δ²/Δt scaling. [Shankar, Kawai & Lele
(2011, eq. A4)](https://doi.org/10.1063/1.3553282) carry `C_Y = 100` into the cΔ scaling
used here. This implementation uses a maximum rather than their additive combination, the
later operator documented by [Brill, Olson & Bokman (2025, eq.
24)](https://arxiv.org/abs/2503.12680). Cook's scaling differs from the cΔ form by 1/CFL, so
the published values map to values near 100 here. Measured on the 2h case at `C_D = 0.01`:

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

Sum and max are indistinguishable, as are smoothing the two terms separately and smoothing
their max once (Cook's single filter of the bracket), which costs no line solve beyond the
ringing sensor's. The unsmoothed term is a grid-scale diffusivity that helps less and
triples the step count through `compute_dt`. The residual 1% at `C_Y = 100` is the compact
scheme's dispersion at a three-cell contact; at 4h it is 0.25%, and at Mach 3 the term takes
the worst excursion from −0.94 to −0.043. The bound enters the diffusive rate like the rest
of D\*: excluding it is stable to `C_Y = 500` and saves 1% of the steps, and at 1000 it
takes a square root of a negative pressure. Cook's Δ²/Δt scaling fixes the bound's diffusion
number by construction, so keeping it in the rate is the same choice.

**The dead band.** The term is not inert on a smooth profile that touches a bound. Uniform
advection of Y = (1 + cos 2πx)/2 for one period measures order 4.9 in L2 at `C_Y = 0` and
3.6 at 100, although the completed steps never leave [0, 1]: the excursion appears only at
Runge–Kutta stages 2–5, at 6e-6 on 64 points and falling 4× per doubling. `Y_tolerance =
1e-4` restores the `C_Y = 0` errors to the last digit and leaves the shock case unchanged,
worst −0.0135 against −0.0134. Evaluating at stage 1 only does the same; the dead band
stores nothing and is the form adopted.

**Cost on a smooth profile.** On two-species advection at uniform ρ, p and u with Y₁ = (1 +
cos 2πx)/2, the L2 error under the default D\* with its bound is 3.12e-6 at N = 32 and
1.15e-10 at N = 256 (order 4.84 → 4.97); with the species channel off it is 7.85e-8 and
3.23e-12 (order 5.50 → 4.29), 40× and 35× smaller. The profile touches 0 and 1 and the dead
band keeps the bound inert there, so the factor belongs to the ringing sensor responding to
a resolved cosine.

**The surrounding formulation**, on a stationary two-gas interface at uniform p, T and u = 0
for t = 0.25. The compact filter alone leaves p, u and T at 1e-14 and is the whole source of
the ±1e-3 mass-fraction overshoot of a resting 2h interface, since the uniform-(p, T) state
is a linear subspace of the conserved variables for ideal gases of constant c_v, which a
componentwise linear filter preserves. The artificial species flux with its enthalpy energy
flux generates the interdiffusion velocity u ≈ −D\* ∇ln ρ (1.4e-4 at D\* ≈ 1e-6) and holds T
uniform to 1e-6; the same flux carrying species internal energy instead, which Brill et al.
argue for, leaves a secular temperature drift 50× larger. Three species at equal density
under the per-species diffusivities hold bulk density to 4e-14, so the correction velocity
already gives what Brill et al. obtain from a shared diffusivity.

**Recommendation:** `C_Y = 100` with `Y_tolerance = 1e-4`, the defaults.

### The sensor smoother

Cook's sensor smoothing is `gbar` in Pyranda, which resolves to `cgfs4` in its [public
stencil source](https://github.com/LLNL/pyranda/tree/b4e0afc),
`pyranda/parcop/stencils.f90`: an explicit nine-point symmetric stencil with `nol = 0` and
`implicit_op = .false.`, weights 3565/10368, 3091/12960, 1997/25920, 149/12960 and
107/103680. Over the common denominator 103680 these sum to exactly 1, so constants are
preserved without cancellation; at boundaries the overhanging weight folds onto the mirror
point. There is no linear solve and no interface reduction, and the halo is four.
`ArtParams.smoother` selects between that Gaussian and one pass of `compact_filter(0.45)`,
whose transfer function is 0.999 at k/π = 0.25 and 0.85 at 0.75 where the Gaussian reads
0.663 and 0.021: as a Cook test filter the compact pass is close to the identity over the
whole resolved band, at a distributed line solve per active dimension per sensor. Highest
CFL reaching `t_final` with a correct plateau, the ladder extended until both settings fail:

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
([operator and step cost](#operator-and-step-cost)).

Accuracy is mixed and small, with wall heating the one clear regression: the ν = 1 deficit
moves +58% to +64% while plateaux and the Shu–Osher train move in the fourth digit. **κ\*
cannot buy the wall heating back**: the ν = 1 wall deficit over a `C_kappa` sweep under
each smoother,

```
C_kappa      0      0.0025    0.01*     0.04     0.16
compact    +64%     +61%      +58%      +57%     +70%
gaussian   +65%     +64%      +64%      +68%     +91%
```

The lever inverts. The Gaussian widens the κ\* footprint and lowers its peak, so added
conductivity spreads over a region rather than concentrating in the wall cell where the
entropy error is deposited. At `C_kappa = 0` the ν = 3 case fails under `:compact` and
completes under `:gaussian`, and the ν = 3 plateau is better under `:gaussian` across the
sweep. **Three dimensions are neutral**: the μ\* share of the Taylor–Green sink at 64³ moves
from 4.0% to 4.5% and the filter still dominates, so the budget does not demand a `C_mu`
refit.

**Recommendation:** `:gaussian`, the default. The wall-heating regression is a property of
the smoother and is largely recovered by `detector = :d8` or by the one-sided filter wall
rows ([the filter's wall rows](#the-filters-wall-rows)).

### The ringing detector

`ring()` in `parcop/operators.f90` dispatches to `d8x/d8y/d8z`, a full compact operator
(`c10d8`, `parcop/stencils.f90`) with a pentadiagonal left-hand side, a nine-point
right-hand side, and its own symmetric and antisymmetric boundary closures. `artificial.jl`
uses the undivided fourth difference δ⁴ = (1, −4, 6, −4, 1), following [Cook
(2007)](https://doi.org/10.1063/1.2728937) literally. `ArtParams.detector` selects between
them: `:delta4` is the default and `:d8` is [`compact_d8`](../src/kernels_banded.jl), the
reference operator transcribed, with interior rows

```
1.5 g_{i-2} + 14 g_{i-1} + 29 g_i + 14 g_{i+1} + 1.5 g_{i+2} = 60 δ⁸f_i,
```

δ⁸ the undivided eighth difference. It is planned as a symmetric operator rather than a
derivative, and its four closure rows fold the overhanging interior weights onto the
half-offset mirror (`reference/DESIGN.md`). Every closure row's weights sum to zero,
measured at 1.2e-16.

**Normalization.** The coefficients are divided by ζ = 29 to put a unit diagonal on the
left-hand side, and by a further 240. The second factor sets the response to a grid-to-grid
oscillation to 16, the value undivided δ⁴ gives there, so both detectors agree at the
wavelength both exist to catch and diverge only below it (response ratio 569× at k/π = 0.25,
26× at 0.5, 3.2× at 0.75, 1× at Nyquist). Without it `:d8` would produce sensors 240× larger
at the Nyquist and the four constants would need refitting by two orders. This is not
Pyranda's own normalization.

**Results**, on top of `smoother = :gaussian`: the failure the Gaussian fixes is β\*
intermittency at a symmetry cell, and a sharper high-pass makes narrower sensor spikes, so
`:d8` against the compact smoother would be rejected for a defect of the smoother.

```
detector    Noh1 plat  deficit   Noh3 plat   Lax L1   contact   Shu train   WC peak
delta4*      0.9993     +64%      0.9751     5.0e-3   0.0053     1.6180     6.6049
d8           0.9997     +53%      0.9951     4.7e-3   0.0044     1.6360     6.3953
```

Six of seven columns improve, several well beyond the fourth digit that separates the β\*
sensor variants: the ν = 3 plateau error falls from 2.45% to 0.42%, wall heating recovers
eleven of the points the smoother change cost, and the Shu–Osher train gains 1.1%.
Woodward–Colella peak density is the one regression. The CFL ladder splits (the sensor-field
table below): `:d8` takes the planar wall and the cylindrical axis from 0.2 to beyond 1.0
and the spherical origin from 0.4 to 0.25. The `1.0+` is not a table edge, since the plateau
under `:d8` is flat to four digits from 0.15 to 1.0 in both geometries. The spherical
failure is at the origin cell, the symmetry-cell startup mechanism of [the CFL
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
wave resolved over 64 points the two differ by 2.6e6 on κ\*, whose input is the internal
energy, against 1.8 on β\*. The last row is a property of every differentiated field: a
centered scheme has zero modified wavenumber at the Nyquist, so |S| and ∇·u vanish
identically for a two-point velocity wave. The calculation is stable in spite of that
because grid-scale dissipation comes from the compact filter, not the Cook properties.

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

The viable window moves and the optimum inside it does not. Under `:delta4` it is bounded
above by planar Noh and runs 0.25 to 1.0; under `:d8` it is bounded below by the two
converging geometries and runs 1.0 to 4.0. **The two windows intersect in the single value
1.0**, the default. `:d8` also flattens the response to the constant on every smooth
measure: over 0.25 to 4 the contact broadens 31% under `:d8` against 80% under `:delta4`.
The CFL ladder per `C_beta`:

```
             nu = 1    nu = 2                nu = 3
C_beta    :d4  :d8   :d4     :d8          :d4   :d8
0.25      1.0+ 1.0+  1.0     0.4 only     0.2   none
0.5       0.4  1.0+  0.4     >= 0.4       0.25  none
1.0  *    0.25 1.0+  0.2     1.0+         0.4   0.25
2.0       none 1.0+  none    0.4          0.3   0.3
```

No `C_beta` in the sample recovers the spherical origin under `:d8`. The best available is
0.3 at `C_beta = 2`, below the 0.4 that `:delta4` reaches at the default constant, and it
costs 2.33% ν = 3 plateau error against 0.49%. **`C_beta = 1.0` is retained under `:d8`**,
and the detector decision falls to the origin cell alone. One reading favours the detector:
at `C_beta = 1.0` the worst geometry improves under `:d8`, 0.2 to 0.25.

**A failure that gets worse as the timestep falls.** Two cells of the `:d8` ladder are not
ceilings. At `C_beta = 0.25`, ν = 2 completes at cfl = 0.4 and fails at 1.0 and at
everything from 0.3 down; at `C_beta = 0.5` it completes at 1.0 and 0.4 and fails from 0.3
down. Both are positivity losses and not the step cap. A CFL-type stability restriction
cannot produce a failure that appears only below a CFL; a per-step operation can, since a
fixed physical interval integrated at half the timestep applies it twice as many times, and
the per-step operation in the loop is the compact filter ([the filter's
dissipation](#the-filters-dissipation)). The sign of the dependence is the evidence; the
accumulation itself has not been measured step by step.

**Recommendation:** `:delta4` remains the default. The battery favours `:d8`, as do the
planar and cylindrical ceilings, but the general guidance for converging shocks rests on the
spherical case, where `:d8` costs 40% of the timestep. The four constants are also the δ⁴
fit, and a detector that changes the sensor's spatial support by this much has no claim on
them. Nothing is shown wrong with `:d8`; `:delta4` survives the origin excursion at a larger
timestep.

### The sensor fields and the compression switch

Cook's [2007 model](https://doi.org/10.1063/1.2728937) builds μ\* and β\* from the strain
magnitude |S| = sqrt(S_ij S_ij); his [2009 model](https://doi.org/10.1063/1.3139305) changes
β\* to the dilatation. Pyranda builds μ\* from `ringV(u, v, w)`, the ring of each velocity
component along each direction reduced by `MAX` over the nine pairs, and β\* from
`ring(∇·u)`. Neither reference field carries an absolute value. `ArtParams.mu_sensor`
(`:strain`, `:velocity`), `ArtParams.beta_sensor` and `ArtParams.reduction` (`:sum`, `:max`)
select between them. The weight is h_d for a field carrying one velocity derivative fewer
against h_d² for |S| and ∇·u, and the two coincide at the grid scale, so the four constants
transfer between sensor fields as they do between detectors. With S the strain-rate tensor,
δ⁴ the undivided fourth difference, Δ = ∇·u, ω the vorticity and ε a fixed regularizer at
the literature value 1e-32, `beta_sensor` has four settings: `:strain` (default) is the Cook
(2007) form Σ_d h_d²|δ⁴_d S|; `:gated_strain` multiplies it by the compression switch
H(−Δ)·Δ²/(Δ² + |ω|² + ε), one pointwise pass and no line solves; `:dilatation` additionally
rebuilds the sensor from Δ, the full form of Mani, Larsson and Moin (JCP 228, 2009), at one
further smoothing pass; and `:ungated_dilatation` is that sensor without the switch, the
reference form.

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
here is one-dimensional at `C_mu = 0.002`. **β\* from the dilatation improves four columns
and degrades two**: under `:d8` the ν = 3 plateau error falls from 0.49% to 0.24% and the
Lax contact sharpens, while the Shu–Osher train loses 0.5% and the Woodward–Colella peak
1.5%. Under `:delta4` the same change gives up more and gains less. The gated and switched
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
the switch's and the cylindrical loss the sensor field's. `:gated_strain` raises the
cylindrical ceiling instead, from 0.15 to between 0.2 and 0.25 under the `:compact`
smoother, with the plateau at 0.2 agreeing with its own value at 0.15, so the larger step is
a completion and not a run that avoided the positivity check. Planar and spherical Noh are
unmoved by the switch.

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

The run loses positivity on the first step, and raising `C_beta` to 2, 4 or 8 does not
recover it. This is a property of the fold rather than of the shock capturing, so the option
is not recommended in converging geometry at any setting of the constants.

**Switch selectivity and decomposition sensitivity.** On a solenoidal Taylor–Green field at
32³, where β\* has nothing legitimate to do, `:gated_strain` removes 99.4% of the summed β\*
and leaves 71 of 32768 points above 1e-12. The maximum survives, at 0.42 of the ungated
value. Those surviving points are the cusps of |S|: the strain sensor is a fourth
difference, so it peaks where |S| passes through zero with a kink, and the switch
degenerates there because the vorticity vanishes too. A relative ε scaled to the local |S|
was tried and reverted, since the scale it would use vanishes with
|S|. `:dilatation` has no such points and its β\* falls to 1e-15 of the ungated maximum.
Neither compression-keyed setting reproduces to round-off when the process grid changes:
summed over the domain, three different split axes agree to 2e-6 relative for
`:gated_strain` and 2e-7 for `:dilatation`, against 1e-14 for the strain sensor. The cause
is the switch and not a missing halo
exchange, since the sensor fields themselves reproduce to 1e-14: H(−Δ) is discontinuous at
Δ = 0, and with ε at 1e-32 the ratio has not decayed by the time Δ reaches round-off, so a
point whose dilatation cancels to zero carries either no β\* or the full C_β·ρ·sensor
depending on the last bit. Anything relying on bit-identical results across process grids
should stay on `:strain`.

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
curved Noh geometries are closest at α = 0.49, the Lax L1 error is flat from 0.45 to 0.49
and worse at 0.499, and the Shu–Osher amplitude rises monotonically as the filter weakens.
At α = 0.499 the spherical Noh loses positivity, the cylindrical plateau falls and the Lax
error rises, so the other converging geometry and the shock tube degrade before the origin
fails. The ceilings do not move with α:

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
the upper bound on α that no Taylor–Green leg produced. The per-pass strength in force is ε
= (1 − 2α) · w with w = min(1, cfl / filter_cfl), and the two Noh geometries at eight points
chosen so that the relaxed and unrelaxed rows interleave on ε:

```
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

At equal ε the two formulations agree to 1e-4 in both plateaus and both fail below it. **The
edge is a property of the per-pass strength in force, not of α or of the formulation, and it
sits between ε = 0.00125 and 0.0025** for the spherical Noh at N = 256, `cfl = 0.15` and the
default constants; the cylindrical plateau degrades monotonically toward the edge without
failing. Retries spend that margin: `StepControl` lowers the CFL by `cfl_backoff = 0.5` each
time, and under the relaxed formulation each halving halves ε with it. Under `filter_cfl =
0.35`, α = 0.45 has ε = 0.043 at the Noh CFL, seventeen times the edge, and reaches 0.0027
on the fourth retry; α = 0.49 has 0.0086 and reaches the edge on the second.

**The constants do not move under a weaker filter.** `bench/artcal.jl beta kappa D Y` at the
two candidate defaults and at the adopted one, beside a baseline at the fitted configuration
in the same environment, which reproduces every table above to the fourth digit.

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
holds. The `C_kappa` wall-heating trough sits at 0.01 under all three, the `C_D` ordering is
identical with the 64-fold sweep moving the width by 18 to 21%, and `C_Y = 100` and 200 hold
the excursion at 1.1 to 1.4% and 0.9 to 1.0%. The one visible change is the `C_Y = 1000`
row, a filter interaction at a value no default is near.

**The default: `filter_cfl = 0.35` at α = 0.45.** At `cfl = 0.35` the weight is one and
every result above that CFL, the Taylor–Green fits included, is unchanged bit for bit. Below
it the strength falls with the CFL, the α = 0.479 equivalent at the Noh CFL, with a margin
of seventeen times the edge and eight times after one retry. A run at a low CFL then takes
the same dissipation per unit time as one at 0.35 rather than several times more, and
lowering the CFL, retrying, or writing output more often no longer moves the answer away
from the reference. α = 0.49 is the fitted value for smooth turbulence at 128³ and 256³ and
stays a per-run selection: it improves the 256³ kinetic-energy misfit by a third at the cost
of a margin of three times the edge at the Noh CFL, which the second retry spends. The
supporting measurements are in [the filter's dissipation](#the-filters-dissipation) and
[Taylor-Green](#taylor-green), where α = 0.45 at full strength is too strong on every
estimator at 128³ and 256³ and the interior optimum at 128³ is a subgrid tuning.

The pins in `test/cases.jl` moved with the default. A default move has to move them with
`Numerics`, or the battery measures a configuration the solver no longer runs, as happened
when the smoother default moved. Six tests whose assertions are recorded trajectories pin
`filter_cfl = 0` explicitly and say so.

### The Brill 2025 parameter set

The coefficient families and their original values come from [Cook (2007, eqs.
15–18)](https://doi.org/10.1063/1.2728937): `C_mu = 0.002`, `C_beta = 1`, `C_kappa = 0.01`,
`C_D = 0.003`, `C_Y = 100`. [Brill, Olson & Bokman (2025, eqs.
22–29)](https://arxiv.org/abs/2503.12680) use a later set with an eighth-derivative
detector, the max over directions, μ\* from the velocity components and β\* from the
dilatation; their values are an order of magnitude or more below Cook's and use a Δ²/Δt
scaling that differs from the cΔ one here by about 1/CFL. In `bench/artcal.jl brill2025`,
`sensors` is `detector = :d8, reduction = :max, mu_sensor = :velocity, beta_sensor =
:dilatation`; `x1` is C_mu = 2.5e-4, C_beta = 0.175, C_kappa = 2.5e-3, C_D = 5e-4, their
values in this scaling; `x4` is four times that; C_Y = 100 throughout.

```
config             | Noh1 plat   def | Noh2 plat   def | Noh3 plat   def | Lax L1  | Shu tr | WC peak | mix wid | SI minY  wid
default            |    0.9993   +64% |    0.9367   +56% |    0.9751   +27% | 5.0e-03 | 1.6180 |  6.6050 | 0.01820 | -0.0135    4
sensors only       |    0.9997   +51% |    0.9477   +56% |    0.9929   +36% | 4.7e-03 | 1.6289 |  6.4269 | 0.01787 | -0.0176    4
sensors, x1        |    0.9996   +22% |    0.9650   +39% |       NaN  +NaN% | 5.1e-03 | 1.6490 |  6.3467 | 0.01787 | -0.0203    4
sensors, x4        |    0.9997   +48% |    0.9518   +52% |    1.0006   +37% | 4.7e-03 | 1.6321 |  6.3836 | 0.01787 | -0.0184    4
sensors, x1, Cb=1  |    0.9997   +52% |    0.9472   +57% |    0.9923   +36% | 4.7e-03 | 1.6312 |  6.4693 | 0.01787 | -0.0178    4
```

The combination survives both converging geometries at the default CFL, which `:dilatation`
alone does not; whether `:d8`'s fold closure or the max reduction is responsible is not
separated here, and the CFL ladder was not run. The one loss, spherical Noh at `x1`, is
C_beta = 0.175 and not the dilatation switch, since C_beta = 1.0 with the other three
constants at `x1` is indistinguishable from `sensors only` in every column. Against the
default, the sensors buy the Noh plateaus and Lax and cost 2.7% of the Woodward peak, 0.7%
of the Shu–Osher train and 30% on the shocked interface's excursion. The strongest refit
candidate is `x4`, the only row with ν = 3 within 0.06%. None of it is the default; the four
constants were calibrated under Cook's 2007 sensor construction.

### The bulk species channel

`ArtParams.species_flux = :bulk` replaces the Fickian artificial species flux by one
diffusive flux F_q = −D_b ∇q on every conserved variable, with D_b built from the Fickian
channel's own bracket sensed on both the mass and the mole fraction of every species
(`reference/DESIGN.md`, "The species channel"). Everything below is on the one-dimensional
cases at the package's constants, the Gaussian smoother, D_b in the timestep, at CFL 0.4.

Section 3.1 of Brill et al. advects a bubble of density R in gas of density 1 at p = 1, T =
1 and u = 10 for ten periods through a periodic unit square, with N_p points across the
interface and N_p Δ = 0.05, and reports the end pressure oscillation: for their traditional
Fickian formulation "at best around 1% of the pressure and at worst greater than 25%",
stable at N_p = 7 for small R; for their diffused-density formulation O(1e-6 to 1e-12) and
stable at N_p = 7 for R ≥ 100. `brill_slab` in `test/cases.jl` is the one-dimensional slab
analogue with N = 20 N_p; under the default channel it reproduces the paper's picture,
somewhat better than the paper reports it. The error is the Fickian channel's alone: at
uniform u, p, T every linear operator of the scheme preserves the uniform state to
round-off, and the Fickian enthalpy flux Σ_k h_k J_k is the one that does not at unequal gas
constants. Switching that channel off with nothing in its place takes every completing row
to 1e-12 to 1e-10 while losing two rows to instability, and adding the bulk flux beside the
Fickian one moves nothing, so no added term removes the Fickian pressure error.

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

R = 1000 at N_p = 7 fails on every row, with the density undershooting beside a jump of 1000
over seven cells before any diffusivity has acted; R = 10 completes on every row. The two
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
bounded profile its L2 errors to seven. The resting air/SF6 interface at uniform p and T
holds max |u| at 1.2e-14, |δp/p| at 3.8e-14 and |δT/T| at 4.0e-14 over 298 steps while ρ
relaxes by 2.0e-2, where the default channel's enthalpy flux drives u to 2.2e-4 and p to
1.3e-4. The channel costs n_cons gradient line solves per direction in place of the Fickian
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
C_D = 0.01 and C_Y = 100 stay. The three-dimensional measurements are in [their own
section](#the-bulk-species-channel-in-three-dimensions). Open beside them: the
ratio-1000 failures, which are the transmitted shock's foot on the shock case and
the seven-cell density jump on the slab, neither a species-channel failure; the unequal-γ
drift is answered there.

### The partial-density species channel

`species_flux = :partial_density` is the form of Brill, Olson & Bokman (2025, eqs. 38–40):
the partial densities diffuse with the shared D_b of the bulk channel, and their mass
flux enters momentum as (ΣJ)u and energy as (ΣJ)|u|²/2 + Σ e_k J_k (`reference/DESIGN.md`,
"The species channel"). It is the default. The one-dimensional battery rows of the
previous section, under all three channels across C_D (scratch script, no bench entry;
`bench/artcal.jl bulk` gives the default-constant row of each):

```
channel          C_D  | mix width | SI 5.04: TV−1 / worst Y / width / steps | SI 100: TV−1 / worst Y / steps | slab 100: max|p−1| / worst Y / steps
fickian          0.01 | 0.01819   | 0.0788 / -0.0129 / 4 / 647             | FAIL                           | 1.86e-02 / -0.0810 / 4207
fickian          0.1  | 0.02024   | 0.0215 / -0.0115 / 4 / 646             | FAIL                           | 3.98e-02 / -0.0792 / 4094
fickian          0.3  | 0.02254   | 0.0083 / -0.0084 / 5 / 644             | FAIL                           | 4.90e-02 / -0.0755 / 4052
bulk             0.01 | 0.01819   | 0.0733 / -0.0108 / 3 / 644             | 0.4090 / -0.0193 / 684         | 4.92e-11 / -0.0643 / 4201
bulk             0.1  | 0.02024   | 0.0060 / -0.0088 / 4 / 641             | 0.0861 / -0.0171 / 677         | 4.66e-11 / -0.0620 / 4049
bulk             0.3  | 0.02254   | 0.0037 / -0.0038 / 7 / 636             | 0.0026 / -0.0128 / 673         | 5.75e-11 / -0.0575 / 4008
partial_density  0.01 | 0.01819   | 0.0759 / -0.0111 / 4 / 647             | 0.5431 / -0.0220 / 719         | 6.08e-11 / -0.0643 / 4201
partial_density  0.1  | 0.02024   | 0.0066 / -0.0098 / 4 / 644             | 0.3243 / -0.0189 / 704         | 4.98e-11 / -0.0620 / 4049
partial_density  0.3  | 0.02254   | 0.0042 / -0.0058 / 7 / 640             | 0.0533 / -0.0169 / 693         | 4.73e-11 / -0.0575 / 4008
```

The Fickian pressure error on the slab grows with C_D; the two consistent channels hold it
at round-off at every C_D. On the ratio-5.04 shock the two agree. At ratio 100 both complete
and the partial-density channel rings four times as much as the bulk one at 0.1, since it
adds no viscosity; 0.3 brings it to 0.053. A He/CO2 slab advected at 100 m/s through 256
points for two periods at uniform p and T gives a maximum pressure error of 3.7e-5 under the
Fickian channel at C_D = 0.01, 3.7e-4 at 0.1, and 7.5e-13 under either consistent channel,
under NASA-9 thermodynamics and a perfect gas alike. On the one-dimensional reduction of
`examples/shock_tube.jl` (previous section's measure) the time mean of TV − 1 at C_D = 0.1
is 0.011 Fickian, 0.0096 bulk and 0.0097 partial density, and 0.0020 partial density at 0.3,
all within 1979 to 1992 steps.

The two-dimensional example itself, 768 × 48 to 2.5 ms at C_D = 0.1, `-t 16` on the
development workstation. The time per step is the range over alternating runs of each
channel to 1.5 ms, taken after the two reductions described below. Grid-scale pressure is max |δ⁴_x p|/16p over pure helium within
0.25 m behind the interface, excluding six cells beside it; the helium dip is 1 − Y_CO2 at
its minimum inside the mushroom head on the line y = L/2 at 2.5 ms:

```
channel          | steps | s/step          | grid-scale p at 1.5 / 2.0 / 2.5 ms | worst Y_CO2 | helium dip
fickian          | 3576  | 0.0221 .. 0.0242 | 3.2e-4 / 7.7e-4 / 6.2e-4           | -4.6e-3     | 15%
bulk             | 3327  | 0.0267 .. 0.0297 | 3.3e-5 / 1.5e-4 / 1.3e-4           | -3.9e-3     | 3.5%
partial_density  | 3572  | 0.0250 .. 0.0275 | 6.5e-5 / 1.9e-4 / 1.4e-4           | -3.7e-3     | 15%
```

Under the Fickian channel concentric pressure ripples four to five cells in wavelength
spread from the interface through the helium; the consistent channels remove them, and what
remains has a period of eight to ten cells. The bulk channel also weakens the vortex cores
visibly (their pressure minima and the v field) and entrains less helium into the head,
which is its added viscosity ρD_b∇u; the partial-density channel reproduces the Fickian
roll-up. No converged reference decides which roll-up is right, and no published comparison
of RM mixing between these forms was found.

As first implemented the partial-density channel cost 27% per step over the Fickian one and
the bulk channel 41%. `bench/phases.jl` on its two-species tube (512 × 32, per right-hand
side) put the difference in the conserved gradients (0.25 ms of 2.85, four line solves) and
the sensor on both Y and X (0.19 ms). Two reductions followed. Under `Transport(mu0 = 0)`
the shared-D_b channels skip `grad_Y`, whose only other reader is `NSCBCInflowBC`, which
now computes it itself; that returns the four solves. With two species only the first is
sensed, since the second's detector outputs and excursions equal the first's to round-off;
that takes the sensor phase from 0.80 to 0.64 ms, the Fickian channel's figure. The tube
then runs the partial-density right-hand side in 2.38 to 2.61 ms against the Fickian 2.27,
with 24 line solves each, and the bulk one in 3.16 ms with 32.

In the literature, Brill, Olson & Bokman and Aslani & Regele (Int. J. Numer. Meth. Fluids
88, 2018) both reject the Fickian flux with the enthalpy term on this pressure argument and
diffuse the partial densities with consistency terms; PadeOps (Lele group, `cgrid.F90`)
uses the Fickian form with the enthalpy flux, and Pyranda's example decks carry no species
term in the energy equation at all.

**Recommendation:** `:partial_density` as the default, with C_D = 0.1; `:bulk` at a density
ratio of 100 or more, where its viscosity holds the ringing; `:fickian` only to reproduce
Cook's form.

### The CFL rate convention

The [public Pyranda kernels](https://github.com/LLNL/pyranda/tree/master/pyranda/parcop)
implement the same Cook method, and reading them against `artificial.jl` identifies five
differences that bear on the constants. Four are measured in the subsections above (the
detector, the sensor fields, the smoother) and in [filtering on non-uniform
volumes](#filtering-on-non-uniform-volumes), which also records that Pyranda's conservative
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
nominal `cfl = 1.68`, times (1 + Ma) for the advective part. Measured on Taylor–Green at
32³, Re = 1600, to t = 10:

```
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

The acoustic ceiling lies between nominal 1.75 and 1.9, where the bound puts it: 1.68 at Ma
0 and 1.85 at Ma 0.1. An unstable row is one whose step count stops falling with the CFL and
whose energy budget breaks; neither raises `SolverFailure` within t = 10. With the
artificial properties on the diffusive rate adds to the denominator, so nominal 2.1 is the
art-off step at about 1.7 and completes. Every estimator improves monotonically up to the
edge in both arms: under `filter_cfl = 0` each step is one filter pass, so a longer step is
less dissipation per unit time.

## Taylor-Green

```text
julia --project=. bench/tgv_energy.jl 128 10 configs=on:1 smoother=gaussian cfl=0.35 alphaf=0.45
julia --project=. bench/tgv_spectrum.jl <checkpoint>
```

`bench/tgv_energy.jl` runs Taylor–Green at Re = 1600 and splits −dKE/dt by mechanism. The
reference peak is 1.28575e-2 at t = 8.97, from the tabulated 512³ pseudo-spectral solution
vendored at `data/spectral_Re1600_512.gdiag`
([provenance](../data/README.md#taylor-green-reference-solution)), and every comparison
passes that solution through the run's own boxcar.

**Read the rate over a window.** The instantaneous rate is `(filter loss)/dt + physical` and
carries the full `dt` jitter: `dt` swings ±12% step to step at 128³ with the artificial
properties on, which against a 37% filter share predicts the ∓4% one-step scatter, while
`C_mu` is ranked on differences well under 1%. The 501-step windowed rate holds within-run
scatter near 0.3%. A boxcar over a curved peak reads low, 0.10% at the 128³ half-width and
3.0% at a width of 0.5; at 32³ the same 501 steps span 0.8 time units and cost 5.2%. The
truncated final step, which pays a full filter pass at short `dt`, is excluded.

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

The residual is one point low here, an adequate proxy at 32³; if the absorbed error holds
near one point of −dKE/dt at every resolution, the correction is about 1% of the channel at
32³ and 8% at 256³, unmeasured. The negative `unattr` means the channels slightly
over-account for the loss; the trial step and the windowed estimate are both candidates and
neither has been isolated. With the filter off at 16³ the budget closes to 0.0%, molecular
dissipation at 100.8% and pressure work at −0.8%, fixing the sign convention: pressure work
exchanges energy between the kinetic and internal forms rather than removing it. A
diagnostic here must restore the artificial coefficient arrays it writes, since `max_rate`
sizes the next step from that storage; the budget pass brackets itself with `art_block` and
`set_art_block!`, and 32³ to t = 10 agrees on the peak to five digits either way.

### Channel shares under refinement

Shares at the dissipation peak, artificial properties on, `filter_interval = 1`, residual
definition. Every row but 128³ uses the Gaussian smoother; 128³ ran under `:compact`:

```
resolution   peak -dKE/dt        vs reference   molecular   mu*    beta*   filter
32³          1.4216e-2 @ 6.58      +16.4%        12.6%     5.1%    ~0%     82.2%
64³          1.2459e-2 @ 8.49        —           33.8%     4.5%    0.0%    61.6%
128³         1.2044e-2 @ 9.06       −6.0%        60.4%     2.3%    0.0%    37.3%
256³         1.3043e-2 @ 8.84       +1.6%        86.4%     0.8%    0.0%    12.8%
```

The windowed reference is 1.2210e-2 at 32³, 1.2809e-2 at 128³ and 1.2844e-2 at 256³, the
widths falling with the step count. The 64³ row's step count was not recorded, so its window
cannot be reconstructed; against the raw tabulated peak it is 3.1% low, and since the window
only lowers the reference, −3.1% is the most negative value that row can take. β\* is
negligible at every resolution, dilatation being negligible at Ma 0.1. The trend is not
monotone and the rows differ in smoother, backend and machine (128³ `:compact` on 224
rzhound ranks over two nodes, 20–25 minutes each; 256³ Gaussian on 4 MI300A APUs at `-t 1`,
24,490 steps in 3.69 h), so resolution and configuration cannot be separated. **Refinement
does not separate μ\* from the filter**: their ratio is 0.060 at 32³, 0.062 at 128³ and
0.0625 at 256³.

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

The same point under the Gaussian gives shares 61.1 / 2.4 / 36.4 against 60.4 / 2.3 / 37.3
and a peak 0.63% away, so the conclusions are unaffected.

**The 128³ peak does not determine `C_mu`.** The sweep has no crossing: every value
underpredicts the reference peak and raising the constant moves further away. The 16-fold
range spans 1.7% in the peak while the 128³ residual is 6% and of one sign, three and a half
times larger, and the 256³ row overshoots on the same estimator, so that residual belongs to
the resolution and configuration. **Retain 0.002**, consistent with the case rather than
determined by it. μ\* costs 43% of wall time at 128³, 21% per step from `compute_artificial!`
plus 18% more steps.

**Removing μ\* improves the fit at 64³.** Three controls at 64³ with the Gaussian smoother,
`cfl = 0.35` and `filter_interval = 1`:

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

μ\* carries 17.6% of the sink at α = 0.499 and 4.7% at α = 0.45 and the histories are worse
for it in both arms, so the best-fitting `C_mu` here is zero. The case is close to resolved
at 64³ and closer above, so what is measured is excess dissipation added, not subgrid
dissipation contributed; `C_mu` needs a case with an unresolved cascade, and the upper bound
still comes from the one-dimensional battery. The budget closes to between −3.0% and +1.1%
across the α = 0.45 arm and the deficit grows to −12.8% as the damping is removed, between
grid-scale energy from a central compact scheme without dealiasing and a windowed −dKE/dt
read against instantaneous channels on the curved flank of the peak, unseparated. A
proportional bias in the filter probe is excluded, the deficit being largest where the
filter's share is smallest.

### The filter is the stabilizer

At 32³, `filter_interval = 4` diverges and `filter_interval = 0` fails with
`SolverFailure(:negative_density)` at t = 5.32. At 128³:

```
config              peak -dKE/dt        mol    mu*   filter   steps   wall
art ON,  filter 1   1.2044e-2 @ t≈9.06  60.4%  2.3%   37.3%   12739   1520 s
art OFF, filter 1   1.2153e-2 @ t≈9.00  62.5%  0      37.5%   10830   1063 s
art ON,  filter 0   SolverFailure(:negative_density) at t = 4.66, step 8515
```

Removing the filter at 128³ kills the run earlier than at 32³ despite its 37% share there,
while the run with the artificial properties disabled and the filter on reaches t = 10. The
128³ failure is a clean energy blow-up, not the dispersive-undershoot signature of the shock
cases: KE tracks the filtered run to within 0.1% until t ≈ 4.4, turns upward and triples
before positivity goes, with `dt` collapsing to 2e-60, and Taylor–Green is unforced so
rising KE is numerical. The filter is necessary and sufficient for stability at 128³ and the
Cook properties are neither; only the art-on leg was run at 256³.

### The alpha sweep at 128 cubed

Five values at 128³, 224 ranks, `cfl = 0.35`, `filter_interval = 1`, `filter_cfl = 0`,
Gaussian smoother, `C_mu = 0.002`. The α = 0.486 row sits where the other four interpolate
the peak crossing:

```
alphaf   steps   peak -dKE/dt        vs window   KE misfit   -dKE/dt misfit   filter
0.40     11504   1.1814e-2 @ 8.38      −7.76%     8.986e-3     8.903e-2        38.7%
0.45     11737   1.2120e-2 @ 8.89      −5.40%     7.019e-3     5.666e-2        36.4%
0.486    12145   1.2865e-2 @ 8.85      +0.39%     4.919e-3     2.723e-2        29.7%
0.49     12267   1.3004e-2 @ 8.85      +1.46%     4.887e-3     2.526e-2        27.4%
0.499    13240   1.3258e-2 @ 8.88      +3.37%     4.750e-3     2.910e-2        15.7%
```

Both misfits fall from α = 0.40 to 0.49 by 46% and 72%, against 19% and 8% across the same
three values at 32³, where the dissipation misfit was not monotone. The 32³ misfit is
dominated by a resolution error common to every α, at 0.12 to 0.15; at 128³ that floor is
fifteen times smaller and α accounts for most of what remains.

**The dissipation misfit has a minimum at α = 0.49.** On the natural axis `1 − 2α`, the
per-pass strength, the kinetic-energy misfit falls by 6.5e-3 per decade over the first
interval, then 3.0e-3, then 0.14e-3, a floor α cannot lower; the dissipation misfit turns
instead, 15% worse at α = 0.499 than at 0.49 after improving 55% over the preceding
interval. **The peak crosses the reference between α = 0.45 and 0.49**: interpolating in
log(1 − 2α) places the total-dissipation match at α ≈ 0.486, just below the misfit minimum,
with misfits 0.7% and 7.8% worse than at α = 0.49. The peak is not a fit criterion on its
own; the peak time carries less still, on a 1.6% plateau from t = 8.15 to 8.79 at α = 0.40.

Molecular, μ\* and filter shares at the peak are 59.6 / 1.7 / 38.7 at α = 0.40,
61.1 / 2.4 / 36.4 at 0.45, 68.0 / 4.6 / 27.4 at 0.49 and 75.9 / 8.4 / 15.7 at 0.499. Across
the first tenfold reduction in `1 − 2α` the filter's dissipation falls only 22% while
molecular dissipation rises 25% and μ\* triples: a weaker filter acts on a larger grid-scale
amplitude, so its dissipation responds far less than its coefficient. Over the last decade
it falls 41% and the resolved and μ\* channels more than replace it, so **the minimum is a
joint one**, the best α at `C_mu = 0.002` rather than a property of the filter alone.

The α = 0.45 row differs from the `:compact` 128³ rows above at matching resolution, rank
count, CFL, cadence and `C_mu`. A 32³ A/B gives 3516 steps under `:compact` against 3078
under the Gaussian, the same sign and order as the 8.5% step-count gap at 128³, consistent
with `plan_direction` skipping the Gaussian's line solve and its collective interface stage.

### Spectra

`bench/tgv_spectrum.jl` takes the shell-averaged kinetic-energy spectrum from a checkpoint
offline; the solver computes no transform. The spectrum sums to the solver's own kinetic
energy to 4e-4 relative, the Parseval check, the residual being the density fluctuation.

At 32³, t = 9, two settings separate by a factor of ten in the grid-scale band and three and
a half decades at k = 16 (4.76e-10 at α = 0.40, 1.21e-6 at 0.49), where the histories
separated them by 13% and the peak by 3%. The compensated spectrum `k^(5/3) E(k)` is flat to
k ≈ 7 at α = 0.40 and to k ≈ 10 at 0.49. At 128³, against the reference at t = 9, 8.6404e-2:

```
alphaf   sum E(k)     vs reference   share above k = 32
0.40     8.4654e-2       −2.02%            0.110%
0.45     8.5096e-2       −1.51%            0.267%
0.49     8.5428e-2       −1.13%            0.681%
0.499    8.5361e-2       −1.21%            1.326%
```

Every setting has lost too much energy by t = 9, the deficit smallest at α = 0.49, a third
estimator agreeing with the misfit minimum and the peak crossing. The band above half the
Nyquist wavenumber grows by about 2.4 per step in α while `1 − 2α` falls by factors of two,
five and ten, so it shows no feature at the optimum. Below k ≈ 15 the spectra agree to a few
percent in no consistent order and separate monotonically only from k ≈ 18 upward: the
filter sets the top quarter of the wavenumbers, under 1% of the kinetic energy.

**The excess dissipation is concentrated in transition rather than at the peak.** Against
the reference at the same instants, −dKE/dt is 11.8%, 9.7% and 6.5% high at t ≈ 4.4 for α =
0.40, 0.45 and 0.49, and 11.1%, 6.9% and 2.6% high at t ≈ 8. The run's own peak then arrives
early and turns over below the reference's, a shape difference rather than a shortage.

**The tails carry no pile-up at any α, including the one the histories reject.** Every tail
steepens toward Nyquist; the four settings sit there at 3.3e-11, 3.3e-10, 5.9e-8 and 2.5e-6,
so α = 0.499 holds 43 times the grid-scale energy of the fitted value and is still three and
a half orders below the spectral peak. The high-wavenumber share is therefore a one-sided
bounding check, and the fit belongs to the history misfit wherever the history is meaningful.

At Re = 1600 and ε ≈ 1.28e-2 the Kolmogorov scale is η ≈ 1.18e-2, so `k_max η` is 0.75 at
128³ and 1.5 at 256³: the dissipation range is not resolved at the screening resolution.
Molecular dissipation at the peak is 55%, 58%, 69% and 78% of the reference total at the
four α, so even at the weakest filter a fifth of the physical dissipation is off the grid.

### The 256 cubed confirmation

Three values at 256³, 896 ranks over eight rzhound nodes, otherwise at the settings of the α
sweep, 1.7 hours each at 65 Mpoint-steps/s:

```
alphaf   steps   peak -dKE/dt        vs window   KE misfit   -dKE/dt misfit   filter
0.45     23178   1.3019e-2 @ 8.86      +1.36%     1.191e-3     1.226e-2         11.8%
0.49     23923   1.2916e-2 @ 8.86      +0.55%     8.122e-4     8.618e-3          6.7%
0.499    24783   1.2838e-2 @ 8.88      −0.07%     5.161e-4     7.457e-3          2.3%
```

**The minimum does not transfer.** Both misfits fall monotonically through α = 0.499 and the
128³ turn is absent; the peak reverses as well, every setting overshooting, with the
overshoot falling as the filter weakens, the boundary fit seen at 32³ and not at 128³. The
channel split gives the reason: refinement takes the filter's share of the sink from 36.4%
to 11.8% at α = 0.45 and from 15.7% to 2.3% at 0.499, molecular dissipation carries 87% to
96%, and `k_max η` is 1.5, so the dissipation range is resolved and nothing is left for a
numerical sink. The best-scoring strength at 128³ is the one that best replaces the missing
dissipation range there.

**α = 0.45 is too strong at both resolutions, and that conclusion transfers**: 0.49 beats it
on every estimator at 128³ and 256³, and the same ordering holds on the one-dimensional
battery. The interior optimum does not, so **α = 0.49 is a 128³ subgrid tuning and not a
fitted constant.** The 256³ rows place no upper bound inside the range run; that comes from
[the battery's stability edge](#the-shock-battery). The misfits fall by factors of six to
nine and three to five under refinement at fixed α, so the 256³ numbers are not at a floor.

### The mu star channel on Taylor-Green

No case in the battery exercises μ\*, so the sensor-field change is measured where that
channel carries a share of the sink. 64³, Re = 1600, to t = 10, default smoother, at the peak:

```
mu*        reduction   steps   t_peak   peak -dKE/dt   molecular   mu*    filter
strain*    sum*        5888     8.49      1.2459e-2      33.8%     4.5%   61.6%
velocity   sum         5884     8.49      1.2421e-2      33.6%     6.0%   60.4%
strain     max         5682     8.97      1.2496e-2      33.3%     2.7%   64.0%
```

β\* is 0.0% in all three. The velocity sensor raises the μ\* share by a third at the expense
of the filter's, and the directional maximum cuts it by 40%, the maximum over three
directions being smaller than their sum; both are moves of one to two points in a channel
the filter dominates, indistinguishable from a rescaling of `C_mu`. The peak time under
`:max` is not: it lands exactly on the reference peak time, the other two settings being
half a time unit early. All three 64³ peaks are below the tabulated reference.

## The filter's dissipation

```text
julia --project=. -t 1 bench/filterrate.jl filter_cfl=0.4
```

A parallel shear layer, `u_x = 0.1 sin(4y)` at uniform ρ and p, is an exact steady solution
of the Euler equations and stays one discretely, since every x-derivative vanishes, so the
filter is the only mechanism that can change its kinetic energy. N = 32 to t = 0.5, zero
viscosity, artificial properties off. Two rejected alternatives: a broadband field loses 64%
of its kinetic energy within tens of steps and cannot lose more, collapsing the spread
across a 4× CFL change to 1.2%; a velocity sine at uniform pressure is an acoustic
oscillation hundreds of times faster than the filter. Total energy shows nothing either, a
symmetric filter on a periodic grid conserving each discrete sum exactly.

### Dissipation per application

```
cfl    steps   unrelaxed          relaxed
0.4      42    2.357e-3  1.000    2.31554e-3
0.2      84    4.707e-3  1.997    2.31552e-3
0.1     168    9.386e-3  3.982    2.31552e-3
```

Unrelaxed, the loss tracks the step count (1 : 2.00 : 4.00), not the elapsed time: at half
the CFL a calculation applies twice the subgrid dissipation over the same interval. Relaxed,
it is constant to six figures across a fourfold change in timestep.

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

Unrelaxed, the run at `cfl = 0.4` that retried once loses 52% more than the one that did
not, and the run at 0.1 that retried twice 3.5 times as much. Relaxed, every entry is the
unretried value, the first pass after the rollback reading the halved `dt · rate` as taken.

**Subcycling.** A refined level filters at its own cadence and its pass reads the root
`dt · rate`, an upper bound on the level's own product. A level must nest four root nodes
inside the root on every side, so the refined region is a box, which costs `u_x(y)` its
exactness; the passive variant `u_z = 0.1 sin(4y)` with `u_x = u_y = 0` in a planar
`(32, 32, 1)` run stays exactly steady, so `planar=true component=3 refine=16` is the case,
a subcycled level-1 box of 16 root nodes about the centre. The planar run takes 34 steps at
`cfl = 0.4`, the Euclidean acoustic bound being √2 and not √3.

```
       no refinement                          refined box, subcycled
cfl   steps  unrelaxed         relaxed        | unrelaxed         relaxed
0.4     34   1.909e-3  1.000   1.891e-3       | 1.508e-3  1.000   1.49460e-3
0.2     68   3.813e-3  1.997   1.891e-3       | 3.010e-3  1.996   1.49467e-3
0.1    135   7.551e-3  3.956   1.891e-3       | 5.956e-3  3.949   1.49463e-3
```

Relaxed, the composite loss is constant to five figures across the fourfold CFL change, and
so with a retry induced on top (1.49470e-3, 1.49467e-3, 1.49468e-3). The refined run loses
about a fifth less, the level filtering at a third of the spacing, 24 points per wavelength
against 8.

### The relaxed formulation

`filter_cfl` is the CFL at which one pass is applied at full strength. Below it the state is
relaxed toward the filtered image, Q ← (1 − w) Q + w F(Q), capped at w = 1, holding the
dissipation per unit time fixed. `filter_cfl = 0` takes the unrelaxed path exactly, as does
any pass at or above the reference CFL, so numbers at the reference survived the move to 0.35.

The weight is directional, as the filter is. Each pass along `d` reads

    w_d = filter_interval · dt · r_d · √n / filter_cfl,    r_d = max (|u_d| + c) / h_d

with the maximum over the domain and `n` the number of active dimensions. `max_rate`
evaluates the three `r_d` alongside the selecting rate and reduces them in the same
`Allreduce`, five scalars in place of two, and `run!` records them in `filter_rate_prev`.
The `√n` is the ratio of the Euclidean acoustic rate to the one-dimensional one on an
isotropic grid, keeping `filter_cfl` in the convention of `cfl`. The step, the artificial
coefficients and the physical diffusivities are outside the weight.

**No single scalar rate meets the gate.** The maximum rate that sizes the step ran 68k
weighted passes on an aspect-ratio-16 grid where an acoustic-limited run beside it made
3.8k, and completed the aligned Noh case with a wrong solution ([directional bulk
viscosity](#directional-bulk-viscosity)). A weight on the Euclidean acoustic rate
`c √(Σ_d 1/h_d²)` would still make the coarse direction's passes grow as `√(1 + AR²)`, and a
molecular diffusive rate carries the same `1/h²` penalty on a stretched wall-resolved grid.
The gate was one profile at every aspect ratio:

```
AR   steps   limit      plateau   deficit   shock    transverse
 1     741   diffusive  3.9841    62%       0.2120   5.4e-12
 4    4968   diffusive  3.9837    63%       0.2121   9.5e-11
16   70453   diffusive  3.9837    63%       0.2121   6.8e-8
```

At N = 400, AR 1 and AR 4 read 3.9957 and 3.9956, 62%, 0.2046, the battery's ν = 1 row, and
the step counts are the same to within the trajectory's change.

The shear layer's relaxed loss is 2.3155e-3 in place of 2.351e-3, 1.5% less: `u_x(y)` is
filtered only by the `y` pass, whose weight reads `√3 c / h` rather than a total carrying
the advective `|u_x| / h`; the subcycled case has no advective term and did not move. On
Taylor–Green at 32³ and 64³ with the artificial properties off, the reference-CFL run is no
longer bit-identical to the unrelaxed one, the weight dipping below one wherever the
velocity at the largest-rate point is spread over two or three axes; the 32³ peak moved
0.08% and the misfits 0.06% and 0.03%. With them on, both resolutions are
artificial-diffusion-limited for part of their length: the filter's share falls 83.7% to
81.2% at 32³ and 67.2% to 62.2% at 64³, μ\*'s rises, the misfits improve 0.157 to 0.147 and
0.0492 to 0.0435, and the 64³ peak moves 8.35 to 9.05 against the reference 8.97. The 128³
fits, at `filter_cfl = 0` where the weight is one, are untouched. Every Noh row is
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
inside the run, and the truncated final step is excluded. A 32³ shakeout returns 1.4217e-2
at α = 0.45 with the channel shares reproduced exactly. Nothing follows from 32³ about the
default: the best misfits there are 0.12 and 0.67, and the weakest filter wins on an 82%
filter share.

### Cadence and alpha are one axis

Six points at 128³, two α values crossed with `filter_interval` 1, 2 and 4:

```
interval  alphaf   steps   peak vs window   KE misfit   -dKE/dt misfit   filter
1         0.45     11737       −5.40%        7.019e-3     5.666e-2         36.4%
2         0.45     11938       −1.50%        5.447e-3     3.881e-2         33.2%
4         0.45     12191       +0.65%        4.913e-3     2.627e-2         29.0%
1         0.49     12267       +1.47%        4.887e-3     2.526e-2         27.4%
2         0.49     12521       +3.07%        4.911e-3     2.682e-2         24.4%
4         0.49     12821       +3.12%        4.825e-3     2.800e-2         20.2%
```

The rows are ordered by `(1 − 2α)/interval`, the per-pass strength divided by the cadence,
taking the values 0.10, 0.05, 0.025, 0.02, 0.01 and 0.005 down the table. On that axis the
dissipation misfit is single-valued with its minimum at 0.02, and the α sweep's points lie
on the same curve. **Cadence and α are not independent settings.** The two closest settings
on the combined axis are 25% apart on it and agree to 0.5% in the kinetic-energy misfit, 4%
in the dissipation misfit and 1.6 points in the filter's share, the instrument's resolving
power. Fit α at `interval = 1`; the equivalence holds on smooth turbulence only, untested on
a shocked case.

### The relaxation leg

Four points at 128³ and α = 0.45, `cfl` 0.17 and 0.35 crossed with `filter_cfl` 0 and 0.35,
without snapshots, since a shortened landing step pays a full pass under `filter_cfl = 0`:

```
cfl    filter_cfl   steps   peak vs window   peak time   KE misfit   -dKE/dt misfit
0.17   0            23021       −8.04%          −0.68     8.867e-3      8.896e-2
0.35   0            11737       −5.41%          −0.05     7.019e-3      5.666e-2
0.17   0.35         23472       −5.20%          −0.08     7.013e-3      5.827e-2
0.35   0.35         11737       −5.41%          −0.05     7.019e-3      5.666e-2
```

Unrelaxed, halving the CFL doubles the passes and costs 26% in the kinetic-energy misfit,
57% in the dissipation misfit and 0.63 in the peak time, 7% of the run. Relaxed, the two
agree to 0.1% and 2.8% in the misfits and to 0.03 in peak time, and the fourth row
reproduces the second in every digit.

**A lower CFL is a stronger filter under `filter_cfl = 0`.** Halving the timestep doubles
the numerical dissipation per unit time and moves the answer further from the reference,
with nothing in the output to indicate it; retries, subcycled levels and shortened output
steps do the same locally. Relaxation removes the dependence and lets an α fitted at one CFL
be used at another. The reference CFL is one number for every dimensionality while the CFL
rate is not, so the one-dimensional battery had to clear the change at its production CFL
numbers ([the shock battery](#the-shock-battery)).

## Filtering on non-uniform volumes

```text
julia --project=. -t 1 bench/filter_conservation.jl
```

The alternative to the unweighted component filter on cylindrical, spherical and stretched
grids is the volume-weighted form of the [public Pyranda
implementation](https://github.com/LLNL/pyranda/tree/master/pyranda/parcop), which filters
J·q and divides by a cell volume passed through the same filter (`filter` in
`parcop/operators.f90`, `CellVolS` in `parcop/mesh.f90`), flipping the radial parity since
the cylindrical volume is odd across the axis. It sits behind `filter_weighting = :volume`.

**What conservation is, discretely.** A directional pass is a matrix M on each component
along a line, and `volume_integral` weights node i by V_i = w_i J_i h, w trapezoidal. The
pass conserves Σ V_i q_i for every q exactly when Mᵀ V = V, so d = Mᵀ V − V is one pass's
defect and d_i / V_i the fraction of node i's content created or destroyed. Constant
preservation, M 1 = 1, is a different property both forms hold to 1e-15 on every line below.
The operators come from unit impulses on lines of 64 nodes at α = 0.45 via `filter_state!`.

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

The defect of a closed line is the closure rows': the cascade filter's column sums are
+2.0e-2, −4.0e-2, +2.2e-2 at rows 1–3 and then alternate at −0.627 per row, the root of
α r² + r + α at α = 0.45, so the twelfth row still carries 3.8e-4 (at α = 0.49 the root is
−0.817 and the twelfth row 1.1e-3). The one-sided rows move the peak to rows 3–7 without
reducing the total. Stretched and curvilinear lines carry the same wall defect as the
uniform one, and the weighting changes the interior figure by under 1e-4. At the folds the
forms part: the even J = r² of the spherical origin conserves to round-off under both, while
the odd J of the cylindrical axis and the poles costs the weighted form a first-row defect
of 0.15, the folded operator of the product having no unit column sums. The unweighted
filter's non-conservation on a non-uniform volume is real but is the wall closure's, present
on a uniform Cartesian grid too.

**The runs.** The battery's curvilinear cases, with the filter's own change of total mass,
momentum and energy over the run relative to the largest total seen. The runs filter through
a callback with the solver's own pass disabled, reproducing `test/validation.jl` to four
digits under `:none`.

```
                    plateau    wall deficit   shock     filter mass   filter energy   steps
Noh nu=1  none      3.9959      62.9%        0.2046     -1.14e-3      +3.41e-4       3650
          volume    3.9959      62.9%        0.2046     -1.14e-3      +3.41e-4       3650
Noh nu=2  none     15.0020      55.2%        0.2092     -4.33e-6      +7.86e-7       2245
          volume   15.0109      44.7%        0.2091     +1.44e-5      +1.84e-6       2227
Noh nu=3  none     62.5012      27.8%        0.2090     +8.38e-7      +8.38e-7       1113
          volume   62.5671      45.9%        0.2089     -5.96e-7      -5.98e-7       1105

Sedov     none     R_s 0.8086 (+1.07%)   peak rho 5.124   filter mass -1.1e-13   3527
          volume   R_s 0.8084 (+1.05%)   peak rho 5.144   filter mass -1.4e-14   3184

shock/interface, 121 points, t = 0.15        worst Y            width   steps
  uniform          none, volume              -0.0137 / 1.0137   3       125
  clustered a=0.5  none                      -0.0076 / 1.0076   3       196
                   volume                    -0.0083 / 1.0083   3       196
```

The planar case and the uniform interface are the same run to every digit, the weighting
being skipped on a uniform volume. The filter's mass defect over a run is 1e-3 on the planar
wall and below 2e-5 on the curved metrics under either form, momentum and energy alike. The
weighting moves the wall deficit down at the axis, where it triples the filter's mass tally,
and up at the origin, where both forms conserve to round-off, so that change is in the
filtered image's shape at the singular cell. Sedov does not move; the interface rings more.

**Decision.** `filter_weighting = :none` stays the default, `:volume` the measured
alternative. A filter conserving on a closed line would have to change the closure rows,
which is the wall-closure question of [the filter's wall rows](#the-filters-wall-rows), not
a metric one. At the symmetry cell the weighting is a per-case adjustment, not a correction.

## The CFL restriction and the symmetry cell

```text
julia --project=. -t 1 bench/nohprobe.jl 1 cfl=0.15 nmax=5000 every=2000 floor=1e-8 scope=representable
```

No setting of the four constants stabilizes a converging strong shock at the default `cfl =
0.5`. The restriction is a symmetry-plane startup problem: the wall, axis or origin cell,
not the shock front. The planar and cylindrical ceilings were the unprimed first step and
are gone; the spherical origin's stands at 0.3.

### The first step of a run

`max_rate` builds its diffusive rate from the artificial coefficient arrays as the last
right-hand-side evaluation left them, and a fresh solver has none, so an unprimed first step
is sized on the acoustic and advective rates alone. Planar and cylindrical Noh start with
u = −1 against the wall or the axis, where the strain sensor is largest. `run!` primes that
step by evaluating the right-hand side of the initial state once beforehand. Ladders under
the defaults (N = 400/256/256, the ν = 3 warm start), highest CFL reaching t = 0.6 with a
correct plateau:

```
                    nu = 1 wall          nu = 2 axis          nu = 3 origin
unprimed            0.25 (0.3 wrong)     0.2  (0.25 fails)    0.3 (0.4 fails @106)
primed              none to 0.9          none to 0.9          0.3 (0.4 fails @106)
primed, first dt    1.04e-3 at 0.9       1.77e-3 at 0.9       unchanged
```

Primed, the wall's plateau is 0.9989–0.9990 of exact from 0.25 to 0.9 with the wall deficit
falling from 63% to 57%, and the axis's 0.9375–0.9381. The unprimed wall at 0.3 completed in
a wrong state, a cold density spike of 4.9 times the plateau on the wall cell; the unprimed
axis at 0.25 failed at step 110, over-dense and cold from step 25. Under `detector = :d8`
the wall and the axis complete through cfl 1.0 even unprimed.

The origin does not move: its excursion lands at t ≈ 0.39 from the warm start at t₀ = 0.3,
well after the first step, and 0.4 fails at step 106 primed or not. Two explanations were
measured and rejected. **The density proportionality of β\***: rebuilt as `C_beta · ρ̃ ·
sensor` with ρ̃ the Gaussian-smoothed density, it raised β\* at the origin during the
excursion (0.61 against 0.43 of the line maximum at step 100 of the cfl 0.4 run) and moved
no ceiling, the wall's 0.3 failure becoming explicit at step 659, the axis's 0.25 failure
moving from step 110 to 386 and the origin failing at step 109 against 106. **The per-step
filter strength**: the ladders read the same under `filter_cfl = 0`, 0.35 and 0.7 in all
three geometries.

### Where the restriction originates

Five explanations were measured wrong. Do not reopen them without new evidence.

**Not the timestep predictor.** Linear extrapolation with `StepControl(predict = n)` moves
Noh ν = 1, N = 400, cfl = 0.3 from failure at step 175 with no lookahead to 179 with three
steps and 200 with thirty; capping growth at `max_growth = 1.05` moves it to 186. In the
per-step trace density falls for 150 steps while `dt` and the rate stay nearly constant
(2.1e-4 and 1418 at step 25, 1.8e-4 and 1718 at step 125, with ρ_min 0.951 then 0.398), and
the diffusive rate climbs to 1.5e4 and `dt` collapses only after positivity is lost. The
cause is spatial, not temporal.

**Not insufficient β\* reach, and not insufficient β\* magnitude.** Over a complete ν = 1
run at `cfl = 0.15`, the furthest cell ahead of the front carrying above a thousandth of the
domain maximum β\*, beside the worst-affected cell:

```
step    x_sh/h | rho_min   i | e/e0_min   i  n_e<0 | b*@e/b*max  reach/h
 500      9.81 | 0.98875  17 |   -469.4  12      7 |      0.166    14.19
1500     27.63 | 0.99008  35 |   -421.7  30      8 |      0.192    14.37
2500     45.47 | 0.98851  53 |   -465.4  48      7 |      0.161    14.53
3500     63.30 | 0.98720  71 |   -498.7  66      7 |      0.140    14.70
```

Reach holds at 14.2–14.8 cells while the worst cell sits 3–5 cells ahead of the front, so a
wider sensor stencil would address a deficit that is not present, and the affected cell
carries a sixth of the domain's peak artificial bulk viscosity throughout. At the failing
`cfl = 0.3` the same column reads 0.84–1.00 over the first 125 steps.

**Not the fold closure**, which is sixth to seventh order and the most accurate region of
the line ([fold order](#fold-order-and-geometry-limits)). **Not sensor blindness at the
fold**: during the excursion that fails, β\* at the origin reaches the line maximum under
both detectors.

The failure starts at the symmetry plane. At ν = 1, `cfl = 0.3` the first cell to degrade is
the wall cell i = 1, whose internal energy is negative by step 5; the density hole at step
125 is at i = 3, between the wall and a front then at cell 4.4, the pre-shock field still
within 1% of unity from cell 11 outward. The ν = 3 origin fails the same way, more abruptly:
one step before the failure the density minimum over the line is still 1.92, and the origin
cell then carries an outward u = +5.6 against an inflow of −1, with e/e₀ = +2.6e5 against
−1.6e4 in its neighbour, under a β\* of 5.7 where the front carries 0.06. `:gated_strain` at
ν = 2, `cfl = 0.2`, the one setting that moves a ceiling, migrates the worst-energy cell off
the axis and reduces its magnitude about sixtyfold.

### Sensor intermittency at the damage site

Sampling the ν = 3 origin at 25-step intervals, where both smoothers complete at `cfl =
0.15`, β\* at the damaged cell over the line maximum:

```
step        150    175    200    225    250    275    300    325    350    375    400
compact    0.393  0.004  0.302  0.007  0.305  0.009  0.300  0.011  0.306  0.007  0.213
gaussian   0.103  0.336  0.118  0.429  0.415  0.120  0.143  0.131  0.108  0.375  0.130
```

Under `:compact` the cell being damaged carries about 30% of the domain maximum on one
sample and under 1% on the next; under `:gaussian` it never falls below 0.100. `reach/h` is
6–11 cells for both and `n_e<0` holds at 7–9 for both, so reach is not the discriminating
quantity. The same signature precedes the failure at `cfl = 0.3`.

Sensor roughness, not β\* magnitude, discriminates, and the restriction at the fold is set
by the continuity of β\*. The undivided δ⁴ applied to |S| produces a spiky field, and the
damaged cell, drifting outward over the run, lands alternately on spikes and in troughs
under a smoother close to the identity across the resolved band, while a nine-point Gaussian
spreads each spike widely enough that no trough remains. A measurement of the detector must
therefore be taken on top of the Gaussian smoother, since a sharper high-pass produces
narrower spikes; Pyranda pairs `:d8` with the Gaussian and never with a near-identity
filter. The ν = 1 probe above samples every 1000 steps, too coarse for an alternation of
this period, so its steady 0.14–0.21 is consistent with either a steady sensor or the
average of an alternation; re-running it at `every = 25` would settle that and has not been
done.

### The origin cell is a startup transient

`bench/nohprobe.jl` reports the symmetry cell on every line, since the argmin columns track
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

The symmetry cell is quiescent for most of the run, so whatever sets the ceiling does not
act gradually from the start; during the excursion β\* at the origin reaches the domain
maximum in both surviving configurations. Every configuration has the excursion, and the
ceiling is whether the cell survives it. The excursion is physical, not grid-scale: under
refinement at `:delta4`, cfl 0.3, the peak of `b*1/max` lands at t ≈ 0.394 at N = 128, 256
and 512 (steps 58, 160 and 342) and weakens from 1.000 to 0.950, so the step number scales
with N while the time does not, a resolved feature of the warm start at t₀ = 0.3.

At the moment of failure the regularization is suppressed by the evacuation itself. β\* is
proportional to density by construction (`src/artificial.jl`), and at the failing step the
symmetry cell has thinned to 0.23 of its neighbour while β\* there has fallen from 0.304 to
0.018 of the line maximum, the profile reading ρ = 38.1 at the origin against 136.7 and
162.1 at the next two cells with the velocity reversed to +1.19. The mechanism is consistent
with the numbers but not demonstrated: a β\* that does not vanish with the density was run
and the ceiling did not move.

### Negative internal energy in completed runs

The `n_e<0` column is nonzero in every sampled configuration, completed runs included: six
to eight interior cells carry negative internal energy, travelling with the front, for the
whole ν = 1 validation case. No diagnostic reports it, since `primitives!` floors T_ion at
1e-300 wherever e ≤ 0 and the positivity check in `max_rate` reads ρ, which stays positive.
The quantity is ill-conditioned here: at the Noh ambient p₀ = 1e-4 the internal energy is
1.5e-4 against a kinetic energy of 0.5, so e is a difference of terms agreeing to within
0.03% and a rounding-level error in either flips its sign. The κ\* sensor is built on e, so
this also bounds the artificial conductivity.

The affected cells are not a rounding step below zero. Over the complete ν = 1 validation
case, 3756 steps at cfl 0.15 returning a plateau of 3.9971, 25179 cell-steps have e < 0 and
none has E ≤ 0 or ρ ≤ 0. They are not a wall layer either: the step-500 profile, front at
cell 11.8, puts them at cells 14 to 28, ahead of the front, as an odd-even oscillation of
the internal energy (−297, +323, −176, +57, +9, −30, +33, −22, +15, −6, +3, +1, −0.3 e₀ cell
by cell), with the wall cell hot at 9200 e₀. The oscillation is the compact scheme's
precursor, and the calculation still reaches the plateau to within 0.07%.

**Repairing those cells is a percent-level intervention and terminates the run.** Under
`StepControl(floor_ratio = 1e-8, floor_scope = :internal_energy)` the failsafe repairs five
cells on step 1 and 256 cell-steps in all, adding 1.4007 of mass and 9.3e29 of energy and
removing 1.7e10 of momentum before the run fails at step 19 with `:dt_collapse`. The repair
damps the velocity where the total energy is still positive and raises the total energy
where no kinetic energy is left to convert, so the cost belongs to the case and not to the
choice of repair. The three geometries at cfl 0.15 under each policy:

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
7, 6 and 8 inadmissible cells with e_min at −0.038, −0.018 and −0.020.

[State validation](DESIGN.md#state-validity-and-its-policy) puts admissibility to the EOS,
and a calorically perfect gas answers that a cell with e < 0 is outside its domain. That
verdict is correct over 25179 cell-steps of a run that reaches the right answer, so a strict
check on the returned state rejects a completed and correct Noh run. The Noh cases therefore
run under `validity = :permissive`, guarded on the state they end with as well as on their
solution error. The mass-fraction side of the same check has its own
[band](#the-species-validity-band).

### Recovery strategy

Rollback retains a larger CFL after the excursion because the restriction is confined to it.
`StepControl(retries = 4)` from an initial `cfl = 0.9` under the current defaults, with the
first step primed and the coefficient arrays banked beside the savepoint:

```
nu   start cfl   recovered cfl   retries   steps   plateau/exact
1    0.9         0.9             0          582    0.9990
2    0.9         0.9             0          364    0.9381
3    0.9         0.225           2          635    0.9768
```

The planar and cylindrical cases need no retry. The spherical case fails at step 45 in the
excursion, rolls back to step 25 at 0.45, fails again at step 68 and completes from step 50
at 0.225; the fixed `cfl = 0.15` run takes 1113 steps for a plateau of 0.9766, so recovery
is about twice as fast, and `solver.cfl` records the accepted value. The coefficient arrays
have to be banked with the state: a restore leaving the failed trajectory's coefficients in
place throttled each retry's first step, and the ν = 3 case then ended in `:no_progress`
sized by coefficients of order 1e57.

**Recommendation:** the default `cfl = 0.5` with `StepControl(retries = 4)` for automatic
recovery, or `cfl = 0.3` for a converging shock at a spherical origin. Under `detector =
:d8` the origin's ceiling is 0.25.

### The singular start and the warm start

The spherical case is warm-started at t₀ = 0.3 from the exact solution with a tanh blend of
width 4h at the shock. Primed, the singular t = 0 start survives its first step but fails at
step 40 at cfl 0.15 and completes only at cfl 0.05, or through `retries = 4` down to 0.075,
in either case with a plateau 18% low and a wall deficit of 77%. The start time and the
blend at N = 256, cfl 0.15:

```
nu  t0    blend   N      steps   plateau/exact   deficit   shock
3   0     4h      256    4400*   0.8207          77%       0.2152
3   0.1   4h      256    1851    0.7573          11%       0.2183
3   0.1   4h      512    3786    0.8803          13%       0.2086
3   0.1   4h     1024    7675    0.9404          25%       0.2043
3   0.2   4h      256    1493    0.8759          21%       0.2177
3   0.3   2h      256    1134    0.9478          22%       0.2094
3   0.3   4h      256    1113    0.9766          28%       0.2090
3   0.3   8h      256    1066    0.9805          32%       0.2133
3   0.3   4h      512    2279    0.9911          30%       0.2039
3   0.3   4h     1024    4615    0.9961          36%       0.2018
2   0.1   4h      256    1882    0.8704           2%       0.2132
2   0.2   4h      256    1514    0.9749           6%       0.2127
2   0.3   4h      256    1114    1.0412           2%       0.2001
```

`*` through retries, at cfl 0.075. The plateau error is made while the shock is within a few
cells of the origin: from t₀ = 0.1, where the shock starts at 8.5 cells, the plateau
converges under refinement at first order (24%, 12%, 6% low), and from t₀ = 0.3, where it
starts at 26 cells, at 2.3%, 0.9% and 0.4%. The blend width moves the plateau by 3% either
way at 4h and the wider blend costs the shock position, so 4h is retained. The battery
starts the cylindrical case singular at t = 0, which the axis takes, and its plateau of
0.938 is a startup error of the same kind: the warm start returns 1.041 with the shock
position exact, and warm-started at t₀ = 0.3 the axis completes through cfl 0.5 even
unprimed.

## Directional bulk viscosity

```text
julia --project=. -t 1 bench/anisotropic.jl
```

A directional artificial bulk viscosity carries one coefficient per grid direction with the
matching per-direction diffusive step limit, where the scalar form loses a factor of the
aspect ratio in the step (Olson & Lele, Comput. Sci. Disc. 5, 014008, 2012; J. Comput. Phys.
246, 207, 2013). The battery is one-dimensional throughout and blind to the difference, so
two anisotropic Noh implosions were built against `noh_exact`, on Cartesian grids whose
dimension-1 spacing is 1/AR of the dimension-2 spacing.

- `noh_aligned`: the planar (ν = 1) implosion along the coarse dimension, with `nx = 12`
  periodic points across the fine one. The data carry no variation along dimension 1, so the
  solution is the one-dimensional profile at every station, the sensor's fine-direction term
  is zero exactly, and the forms differ only in the step they take. The largest transverse
  variation of the density at the end (`uniformity`) is amplified round-off.
- `noh_cartesian`: the cylindrical (ν = 2) implosion on the plane [−L, L]² converging on its
  central node, with the exact time-dependent inflow on all four faces: a curved front
  oblique to the grid at every angle over an irrotational pre-shock flow. On a walled
  quadrant the two walls and their corner dominated every measure (a plateau of 12.5 against
  16 and a corner deficit of 79% at AR 1), so the case takes the full plane. The warm start
  of the spherical case rings here, so it starts singular, which the scalar form completes
  at `cfl = 0.3`. At N = 24 per half-side the plateau is 11.4–12.9 at every AR under the
  scalar form, 12% low at N = 64, so the comparison is between forms at equal resolution,
  not against the exact plateau.

### The forms

Cook's scalar coefficient is β\* = C_β ρ G[Σ_d Δ_d² |δ⁴_d S|], with G the smoother, entering
every normal stress as β\* ∇·u and the step through 2β\*/ρ · Σ_d 1/Δ_d². Two directional
forms were implemented behind an `ArtParams` option and removed after the measurement. Each
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

The step gain is the predicted one, 4.5× at AR 4 and 15× at AR 16. Even the one-dimensional
battery is diffusion-limited at the front under C_β = 1, so the aspect ratio enters the
scalar form's step as AR² through 1/Δ_x² and the directional forms' as AR through the
acoustic rate.

The scalar form's run at AR 16 is wrong although it completes: the wall density is 24
against the exact 4 and the front has reached 0.047 in place of 0.2. The cause is not the
bulk viscosity, whose flux along the fine direction is zero on this flow, but the relaxed
filter reading the rate that sized the step, the AR 16 run making 68k weighted passes
against the directional run's 3.8k. The fix, a directional filter weight, is in [the
filter's dissipation](#the-filters-dissipation), where the same case reads one profile at
every aspect ratio.

### The curved case

N = 24 per half-side, `cfl = 0.3`, p₀ = 1e-4, to t = 0.6. |ω| and |∇·u| are the largest
values over the pre-shock region r > 1.2 R_s at the end:

```
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

At N = 48 both forms fail at AR 4 earlier than at N = 24. The failure moves neither with the
CFL (0.3, 0.15 and 0.075 fail at t = 0.095–0.10) nor with the reduction nor with the ambient
pressure (p₀ = 1e-2 and 1e-1 fail at 0.09–0.12 under both forms while the scalar form
completes both). The sensor field moves it and does not remove it: under the spacing-scaled
split, `:ungated_dilatation` fails at t = 0.52, `:gated_strain` at 0.087, and `:dilatation`
completes AR 4 at N = 24 and fails at N = 48 and at AR 8, where the scalar dilatation form
completes.

The mechanism is the stress form. A scalar bulk viscosity exerts the force ∇(β\* ∇·u), the
gradient of a scalar, which cannot create vorticity; with unequal coefficients the force is
(∂_x(β\*_x ∇·u), ∂_y(β\*_y ∇·u)), which is not a gradient, and in Noh's cold, converging,
irrotational pre-shock gas with no pressure to resist it, it does. At t = 0.06 on the AR 4
grid the largest pre-shock vorticity is 19.9 under the scalar form and 428 and 414 under the
two directional ones, against a largest dilatation of 59–86, and the worst internal energy
is −0.095 against −6.2. The run then cavitates just outside
the front, at 70–76° from the fine axis, where the density reaches 0.47 in gas that should
be at 3.2 and the y-strain has turned to expansion, and the step collapses. The same want of
fine-direction damping shows in the aligned case's transverse round-off, amplified to 1e-6
by a directional run where the scalar form holds it at 1e-10. The compression switch removes
the rotational force where the flow has begun to expand, postponing the failure without
acting before the vorticity exists.

### Decision

Neither directional form is adopted, and the implementation was not retained. The benefit is
real on the aligned control, but the form fails on the curved case, the case a stretched
grid is used for, by a mechanism intrinsic to a bulk stress with unequal diagonal
coefficients rather than to a constant or a sensor. The reference's success on a blast wave
and a nozzle boundary layer is neither reproduced nor contradicted here: both carry pressure
everywhere, and neither converges cold gas onto a point. The two cases remain, guarded in
`test/validation.jl` under the scalar form, with `bench/anisotropic.jl` as the instrument
for the scalar form's aspect-ratio penalty.

## The smooth-evolution accuracy matrix

```text
julia --project=. -t 1 bench/boundaryorder.jl
julia --project=. -t 16 test/convergence.jl
```

The matrix records the order of the solution at a wall or an interface after a smooth
evolution, separated from the interior, with the time error controlled, unfiltered and
filtered. The cases are `test/smooth_cases.jl`, shared with `test/convergence.jl`, which
guards fifteen of the rows. Serial Float64, every evolution at `cfl = 0.25` with the
artificial properties off; the same walls with the properties on are in [wall closures in
production](#wall-closures-in-production).

**The cases and their references.** Every case is one-dimensional along x. The standing wave
`rho = 1 + 0.05 cos(πx)`, `u = 0.05 sin(πx)`, `p = rho^γ` on [0, 1] between slip walls, or
between adiabatic no-slip walls with `mu0 = 0.005` (u vanishes and T is even at each wall,
so the data satisfy both conditions), integrated to t = 0.4, a third of an acoustic transit.
Its density and pressure are even and its velocity odd about both walls, so it is the
restriction of a periodic problem on [0, 2), and a periodic run on 2(N − 1) nodes at the
same spacing is the wall run without its closure rows. The difference between the two, at
the same step, is the closure defect alone, derivative and filter rows together; the
difference from a periodic run at four times the resolution is the total error, and both are
tabulated. The decaying shear mode `v = 0.1 sin(πx) exp(−μπ²t)` at uniform rho and p between
no-slip walls is exact once a source removes the viscous heating `μ v_x²`; its tangential
stress `μ v_x` is nonzero at the wall, so the wall rows differentiate a nontrivial flux. The
entropy wave `rho = 1 + 0.2 sin(k(x − u₀t) + φ)` on the periodic [0, 2π), the wave of the
patch and level tests, is exact; the viscous standing wave on the periodic [0, 2) has its
fine reference on 1728 nodes, which nests every level's nodes. Refinement regions are fixed
in physical space at every N.

Errors are maximum norms over regions: the wall window (the first and last four nodes of a
physical boundary), the interface window (the same at a patch or level end), the covered
parent nodes under a child level, and the interior; `l2` is the composite volume-weighted
root-mean-square through the package's masked quadrature, covered parents excluded. Orders
are between successive resolutions against the actual spacing. The `dt` column is the
relative change of the windowed error when the step is halved, below 0.01 on every row
cited as an order unless said otherwise.

### Closure truncation and the instantaneous right-hand side

One derivative of x^(q+1), q the closure rows' exactness degree, on 17 to 129 nodes, wall
window:

```
                     N=17       33         65         129        orders
C6 cascade3, x^4     1.092e-3   1.365e-4   1.706e-5   2.132e-6   3.00 / 3.00 / 3.00
C6 cascade4, x^5     2.397e-4   1.498e-5   9.363e-7   5.852e-8   4.00 / 4.00 / 4.00
C6 BL, x^6           1.540e-4   4.812e-6   1.504e-7   4.700e-9   5.00 / 5.00 / 5.00
C8 BL, x^8           2.872e-4   2.252e-6   1.759e-8   1.382e-10  6.99 / 7.00 / 6.99
```

The rows' formal orders are 3, 4, 5 and 7, and the interior converges at the same rate in
every case, the wall defect carried inward by the compact solve. The `exp(sin(3x))` slopes
of the convergence suite sit above these by the field's phase and are regression guards, not
the rows' order.

The instantaneous error of the assembled Navier–Stokes right-hand side on the exact initial
data, density component, wall window / interior:

```
                          N=49                    97                      193                     wall orders
inviscid wall, cascade3   9.848e-7 / 1.779e-8     6.166e-8 / 1.116e-9     3.856e-9 / 6.978e-11    4.00 / 4.00
inviscid wall, cascade4   6.724e-7 / 7.731e-9     4.234e-8 / 4.863e-10    2.651e-9 / 3.044e-11    3.99 / 4.00
inviscid wall, C6 BL      3.588e-8 / 3.878e-10    5.702e-10 / 6.134e-12   8.954e-12 / 9.606e-14   5.98 / 5.99
inviscid wall, C8 BL      5.441e-9 / 3.199e-10    2.211e-11 / 1.299e-12   1.356e-13 / 5.829e-15   7.94 / 7.35
shear mode (rho v), cascade3   6.217e-7 / 3.165e-8   7.775e-8 / 3.961e-9   9.719e-9 / 4.953e-10  3.00 / 3.00
shear mode (rho v), cascade4   6.383e-7 / 1.081e-8   7.996e-8 / 1.357e-9   1.000e-8 / 1.699e-10  3.00 / 3.00
shear mode (rho v), C6 BL      2.896e-9 / 5.562e-11  9.098e-11 / 1.743e-12 2.964e-12 / 5.344e-14 4.99 / 4.94
shear mode (rho v), C8 BL      8.903e-10 / 1.889e-11 7.597e-12 / 1.624e-13 2.286e-12 / 5.300e-14 6.87 / roundoff
```

The viscous wall reads the same density rows as the inviscid one, the mass flux having no
viscous term, and its energy rows two orders lower at the wall, 2.6 / 2.0 under `:cascade3`
against the inviscid 4.0 / 4.0: the conductive flux is a second derivative of an even
temperature, so the rows' third-order defect in `T_x` is second order once differentiated
again. The standing wave's mass flux is odd about the wall, so the leading error term of
every closure, proportional to a derivative of the flux that vanishes there, drops out and
the rows read one order above their formal one, while the shear mode's tangential stress is
even and reads the formal 3 and 5. The viscous term is two derivatives: `:cascade4`'s
fourth-order defect in `v_x` becomes third order once differentiated again, no better than
`:cascade3` on the shear mode, while the Brady–Livescu rows' leading term again vanishes on
the odd `v` and they read 5 after the second derivative.

### Walls

The standing wave's wall window at t = 0.4, density, against the fine reference; the row
against the mirror agrees to three digits everywhere the closure defect is above 1e-12:

```
                                     N=49        97          193         orders          l2 orders
inviscid, C6 cascade3, unfiltered    2.843e-7    1.949e-8    1.217e-9    3.87 / 4.00     4.25 / 4.62
inviscid, C6 cascade3, cascade filter 2.891e-5   8.736e-6    2.359e-6    1.73 / 1.89     2.16 / 2.11
inviscid, C6 cascade3, onesided      2.580e-7    1.353e-8    1.254e-9    4.25 / 3.43     4.70 / 4.12
inviscid, C6 cascade4, unfiltered    3.137e-8    6.000e-9    2.110e-9    2.39 / 1.51     3.16 / 0.64
inviscid, C6 cascade4, cascade filter 4.858e-5   1.285e-5    3.306e-6    1.92 / 1.96     2.27 / 2.15
inviscid, C6 cascade4, onesided      4.168e-8    6.826e-8    7.814e-7    −0.71 / −3.52   0.49 / −3.78
inviscid, C6 BL, unfiltered          2.537e-9    4.503e-11   8.802e-13   5.82 / 5.68     6.44 / 5.53
inviscid, C6 BL, cascade filter      4.039e-5    1.415e-5    4.028e-6    1.51 / 1.81     2.25 / 2.19
inviscid, C6 BL, onesided            2.517e-9    4.485e-11   8.737e-13   5.81 / 5.68     6.32 / 5.53
inviscid, C8 BL, unfiltered          1.818e-10   4.603e-12   4.134e-13   5.30 / 3.48     time-limited
inviscid, C8 BL, cascade filter      negative density at step 15 / 17 / 19
inviscid, C8 BL, onesided            1.519e-10   4.527e-12   2.855e-13   5.07 / 3.99     time-limited
viscous, C6 cascade3, unfiltered     2.336e-7    1.583e-8    1.019e-9    3.88 / 3.96     4.56 / 4.53
viscous, C6 cascade3, cascade filter 2.712e-5    7.371e-6    2.132e-6    1.88 / 1.79     2.11 / 2.09
viscous, C6 cascade3, onesided       1.538e-7    1.247e-8    8.703e-10   3.62 / 3.84     4.28 / 4.39
viscous, C6 cascade4, unfiltered     3.387e-8    1.065e-9    3.421e-11   4.99 / 4.96     5.37 / 5.22
viscous, C6 cascade4, onesided       3.707e-8    1.306e-9    5.270e-11   4.83 / 4.63     5.21 / 5.07
viscous, C6 BL, unfiltered           1.941e-9    4.784e-11   8.411e-13   5.34 / 5.83     6.26 / 6.41
viscous, C6 BL, onesided             1.976e-9    4.376e-11   7.807e-13   5.50 / 5.81     6.16 / 6.32
viscous, C8 BL, unfiltered           1.112e-10   9.266e-13   2.853e-14   6.91 / roundoff
viscous, C8 BL, cascade filter       negative density at step 63 / 190 / 1055
viscous, C8 BL, onesided             2.126e-10   1.019e-12   3.864e-14   7.71 / roundoff
```

The shear mode's tangential momentum, exact:

```
                              N=49        97          193         orders
C6 cascade3, unfiltered       7.924e-9    3.293e-10   1.164e-11   4.59 / 4.82
C6 cascade3, cascade filter   6.452e-6    5.969e-7    4.976e-8    3.43 / 3.58
C6 cascade3, onesided         1.586e-9    7.802e-11   3.574e-12   4.35 / 4.45
C6 cascade4, unfiltered       9.437e-10   5.930e-11   2.358e-12   3.99 / 4.65
C6 cascade4, onesided         2.114e-9    7.152e-11   2.172e-12   4.89 / 5.04
C6 BL, unfiltered             1.692e-10   1.359e-12   1.052e-14   6.96 / 7.01
C6 BL, cascade filter         7.507e-6    1.128e-6    1.427e-7    2.73 / 2.98
C6 BL, onesided               1.943e-10   1.692e-12   1.334e-14   6.84 / 6.99
C8 BL, unfiltered             8.431e-13   2.160e-15   roundoff    8.61
C8 BL, cascade filter         negative density at step 86 / 270, dt collapse at 5309
C8 BL, onesided               1.063e-12   4.356e-15   roundoff    7.93
```

1. **The cascade filter is the accuracy of every filtered wall.** Under it every derivative
   closure reads 1.5–2.3 in the wall window and a hundred to a thousand times the unfiltered
   error at N = 193. The one-sided rows return each closure to within a few percent of its
   unfiltered error, and the filtered rows' `dt` column is 0.01–0.05 where the unfiltered
   ones read 0.001: the filter's defect alone depends on the step. The F2 row's defect is
   `O(h²) f''` at the wall, and the shear mode shows it: its filtered field is odd about the
   wall, `f''` vanishes there, and the cap is 3.5 instead of 1.8. The default filter rows
   moved for this reason ([the filter's wall rows](#the-filters-wall-rows)).

2. **The unfiltered default wall is fourth order in evolution**, 3.9 on the standing wave
   with and without viscosity, 4.6–4.8 on the shear mode whose right-hand side is third
   order at the wall: the solution norm gains at least an order over the pointwise
   truncation, as Gustafsson's theorem allows, and the `l2` order is half an order above the
   windowed one because the defect occupies a window of fixed node count.

3. **C6 Brady–Livescu reads 5.7–5.8 at a wall**, unfiltered or under the one-sided rows,
   inviscid and viscous, and 7.0 on the shear mode; C8 Brady–Livescu reaches 7–8 before the
   time error or roundoff floors it near 1e-13 at N = 97, and fails on smooth data under the
   cascade filter in every configuration measured, so the recorded shock failures of that
   pair are not shock failures.

4. **`:cascade4` carries an undamped mode at an inviscid wall.** Without a filter its error
   stops converging by N = 193; under the one-sided filter it grows with N; under the
   cascade filter it is capped at 1.9 as every closure is. Viscosity damps the mode, and the
   viscous wall reads 5.0 unfiltered. `:cascade4` needs the F2 row.

5. **The mirror and the fine reference agree**, so the closure defect is the whole of the
   wall error at every C6 row, and the interior error (8.1e-11, 4.3e-12, 2.6e-13 at N = 49,
   97, 193) is two to three orders below it; that interior number converges at 4.0–4.2 at
   `cfl = 0.25`, the time integrator's order.

### Patch interfaces and refinement levels

The entropy wave through a same-level interface (two root patches, periodic, so both patch
ends are interfaces), interface window, t = 0.5:

```
                                   N=48        96          192         orders          l2 orders
k=3, C6, unfiltered                5.682e-4    1.653e-5    6.315e-7    5.10 / 4.71     5.21 / 4.61
k=3, C6, cascade filter            2.997e-4    1.330e-5    3.698e-7    4.49 / 5.17     5.02 / 4.80
k=3, C6 BL, unfiltered             1.381e-4    1.394e-6    1.119e-8    6.63 / 6.96     6.92 / 7.34
k=3, C6 BL, cascade filter         1.771e-4    2.534e-6    2.334e-8    6.13 / 6.76     6.52 / 7.24
k=3, C10, unfiltered               5.800e-4    1.639e-5    5.941e-7    5.14 / 4.79     5.13 / 4.65
k=1, C6, unfiltered                8.105e-7    9.544e-8    8.262e-9    3.09 / 3.53     4.00 / 4.32
k=1, C6, cascade filter            1.077e-6    4.235e-8    3.553e-9    4.67 / 3.58     4.73 / 4.58
k=1, C6 BL, unfiltered             7.105e-8    7.909e-10   8.611e-12   6.49 / 6.52     6.95 / 7.03
k=1, C6 BL, cascade filter         1.141e-7    1.744e-9    1.700e-11   6.03 / 6.68     6.59 / 7.11
k=1, C10, unfiltered               8.173e-7    9.800e-8    8.523e-9    3.06 / 3.52     4.04 / 4.35
```

The k = 3 wave is pre-asymptotic at the root spacing (kh = 0.39 at N = 48) and its 5.1 / 4.7
is the approach to the asymptotic 3.1 / 3.5 the k = 1 wave reads over the same three grids,
so `test/convergence.jl` gates this row on k = 1. The same-level interface's right-hand-side
error is 27 times the two-level one's at the same root spacing, 3³ for the fine level's
spacing; the C10 rows are the C6 rows to three digits at every interface, the divergence's
closure rows there being the C6 cascade whichever interior is chosen.

The entropy wave, k = 3, through the nests, interface window; the C10 rows are the C6 rows
to two digits and are omitted:

```
                                                 N=48        96          192         orders          l2 orders
2 levels, C6, unfiltered                         1.420e-5    1.335e-6    9.408e-8    3.41 / 3.83     3.62 / 3.53
2 levels, C6, cascade filter                     4.766e-6    2.433e-7    1.577e-8    4.29 / 3.95     4.59 / 4.08
2 levels, C6 BL, unfiltered                      2.667e-7    4.111e-9    6.401e-11   6.02 / 6.01     6.02 / 5.99
2 levels, C6 BL, cascade filter                  1.567e-6    4.131e-8    7.378e-10   5.25 / 5.81     6.82 / 6.57
2 levels subcycled, C6, unfiltered               1.420e-5    1.321e-6    9.198e-8    3.43 / 3.84     3.64 / 3.56
2 levels subcycled, C6 BL, unfiltered            2.670e-7    4.128e-9    6.480e-11   6.02 / 5.99     6.02 / 5.98
3 levels, C6, unfiltered                         1.244e-5    1.101e-6    7.407e-8    3.50 / 3.89     3.58 / 3.52
3 levels, C6, cascade filter                     4.998e-6    1.579e-7    1.097e-8    4.98 / 3.85     5.99 / 4.27
3 levels, C6 BL, unfiltered                      2.667e-7    4.111e-9    6.391e-11   6.02 / 6.01     6.02 / 5.99
3 levels subcycled, C6, unfiltered               1.244e-5    1.086e-6    7.192e-8    3.52 / 3.92     3.61 / 3.55
3 levels subcycled, C6 BL, unfiltered            2.670e-7    4.127e-9    6.491e-11   6.02 / 5.99     6.01 / 5.98
viscous standing wave, 2 levels, C6              1.318e-7    1.079e-8    8.597e-10   3.61 / 3.65     3.69 / 3.79
viscous standing wave, 2 levels, C6 BL           6.755e-10   1.418e-11   3.511e-13   5.57 / 5.34     6.09 / 5.63
viscous standing wave, 3 levels, C6              9.841e-8    9.216e-9    8.284e-10   3.42 / 3.48     3.38 / 3.57
```

A level interface reads the C6 closure cascade at the fine spacing, 3.4–3.9, and C6
Brady–Livescu 6.0, on every nest, subcycled or not, inviscid or viscous; the covered parent
nodes and the interior sit at or below the interface window throughout, and the composite
`l2` follows the window. The subcycled rows differ from the global-step ones in the third
digit and their `dt` column reads 0.01–0.02 at N = 192 against 0.001–0.006, the Hermite
shell's time error beginning to show under the sixth-order closure. The filter lowers the
interface error of the default closure and raises Brady–Livescu's sixfold; at a level
interface the filter's closed rows act at the fine spacing, and the 1.8 cap of the wall rows
does not appear within these grids.

**The other components.** The momentum and energy of the inviscid wall and the two-level
entropy wave, unfiltered:

```
                                     N=49/48     97/96       193/192     orders
inviscid wall, cascade3, rho u       4.801e-8    2.282e-9    9.429e-11   4.39 / 4.60
inviscid wall, cascade3, E           1.043e-6    7.114e-8    4.436e-9    3.87 / 4.00
inviscid wall, C6 BL, rho u          6.084e-10   3.863e-12   5.091e-14   7.30 / 6.25 (time-limited at 193)
inviscid wall, C6 BL, E              8.409e-9    1.427e-10   3.051e-12   5.88 / 5.55
2 levels, cascade3, rho u            7.099e-6    6.675e-7    4.704e-8    3.41 / 3.83
2 levels, cascade3, E                1.775e-6    1.669e-7    1.176e-8    3.41 / 3.83
2 levels, C6 BL, rho u               1.334e-7    2.056e-9    3.201e-11   6.02 / 6.01
2 levels, C6 BL, E                   3.334e-8    5.140e-10   8.022e-12   6.02 / 6.00
```

The energy reads the density's order at the wall and the momentum half an order more; at a
level interface the three components read one order to two digits, so the density rows the
guards are set from are representative.

### Level transfer order

`julia --project=. -t 1 bench/boundaryorder.jl study=transfer,transfertime`
(about thirteen minutes; `coupling="(key=value,)"` reruns every row under
another interface coupling). The level rows at each
`level_interpolation_order`, default interface coupling, interface window at
N = 192 with the orders over 48 / 96 / 192; "same" is identical to the
last digit across orders 4, 6 and 8.

```
                                            order 4             order 6             order 8
1-D, 2 levels, C6, unfiltered               9.408e-8 (same)     same                same         3.41 / 3.83
1-D, 2 levels, C6, filtered                 8.968e-8 4.31/3.99  1.577e-8 4.29/3.95  1.561e-8 4.10/3.91
1-D, 2 levels subcycled / 3 levels, C6      9.198e-8 / 7.192e-8, same at every order
1-D viscous, 2 levels, C6                   4.575e-10 3.80/4.17 8.597e-10 3.61/3.65 8.616e-10 3.61/3.65
1-D, 2 levels, C6 BL, unfiltered            6.401e-11 (same)    same                same         6.02 / 6.01
1-D, 2 levels, C6 BL, filtered              3.309e-7 3.67/3.96  7.378e-10 5.25/5.81 1.049e-10 6.98/6.78
1-D, 2 levels subcycled / 3 levels, C6 BL   6.480e-11 / 6.491e-11, same at every order
1-D viscous, 2 levels, C6 BL                1.239e-9 2.90/2.87  3.511e-13 5.57/5.34 1.080e-13 6.20/5.98
1-D, 2 levels, C10, filtered                9.047e-8 4.30/3.99  1.592e-8 4.31/3.95  1.575e-8 4.13/3.91
2-D, 2 levels, C6, unfiltered               1.019e-7 3.98/3.57  1.840e-8 3.35/3.82  1.840e-8 3.35/3.82
2-D, 2 levels, C6, filtered                 8.095e-8 3.90/4.00  3.720e-9 4.05/4.04  3.697e-9 3.92/4.01
2-D, 2 levels, C6 BL, unfiltered            1.024e-7 3.66/3.89  9.156e-11 5.72/5.81 7.739e-12 5.63/5.71
2-D, 2 levels, C6 BL, filtered              6.894e-7 2.61/2.61  6.043e-10 4.87/4.69 1.279e-11 6.69/6.58
```

The 2-D case is the oblique entropy wave (k = (2, 1), u = (0.5, 0.25)) through
a square level (`entropy2d_case`). In one dimension the imposed boundary
plane is a single node coincident with a parent node, so an inviscid,
unfiltered run never reads an interpolated value and its rows cannot depend
on the order; the transfer enters through the ghosts a gradient (viscous) or
a filter reads, and in two dimensions through the planes, interpolated along
the other dimension. Under the default coupling order 6 is saturated
everywhere but the viscous row, where order 4 happens to measure lower.
Under Brady–Livescu interface rows, which stand in here for a sixth-order
interface divergence, order 6 holds the filtered, viscous and 2-D rows to
4.7–5.8 and order 8 lifts them to 5.6–7.0 with errors 3 to 47 times lower at
N = 192; the 2-D filtered rows read `dt` 0.066 at N = 192, the time error
beginning to show. C8 Brady–Livescu at a level interface is unstable at every
order: the inviscid error grows between N = 96 and 192 and the filtered runs
lose positivity by t = 0.4. The C10 interior under the C8 filter reads the C6
rows.

The default-row C8 and C10 levels (`derivs=c8,c10 orders=6,8,10`) read the
same at orders 6, 8 and 10 on every row, to the second digit: C8 1.288e-7
(3.27 / 3.85) and C10 1.432e-7 (3.61 / 3.76) unfiltered, both 1.57–1.59e-8
filtered, the viscous rows 9.2e-10 and 9.8e-10 at 3.5–3.6. The closure rows
bind, so matching the order to the operator costs nothing there. Under the
ghost fluxes (`coupling="(interface_flux=:ghost,)" orders=8,10
derivs=c6,c8,c10`) the shell binds instead:

```
                                            order 8              order 10
2 levels, C6, unfiltered                    6.380e-11 6.02/6.01  the same
2 levels, C6, filtered                      1.049e-10 6.96/6.78  the same
2 or 3 levels subcycled, C6                 6.481e-11 6.02/5.99  the same
viscous, 2 levels, C6                       1.060e-13 6.29/6.06  1.059e-13 6.31/6.06
2-D, 2 levels, C6, unfiltered / filtered    4.071e-12 / 6.636e-12, 6.08/5.89 / 6.87/6.91; the same
2 levels, C8, unfiltered                    1.873e-12 7.27/7.46  8.715e-14 8.00/7.77
viscous, 2 levels, C8                       9.497e-13 at N = 96  1.354e-14 at N = 96
2 levels, C10, unfiltered                   1.392e-12 7.70/7.55  1.643e-14 9.80/7.23
viscous, 2 levels, C10                      3.646e-13 at N = 96  1.932e-14 at N = 96
2 levels, C8 or C10, filtered               8.13e-11 7.14/7.04   the same
```

C6 is saturated at order 8; C8 reaches its eighth order and C10 its tenth
only at order 10, below which the shell sets the constant, and the viscous
rows reach the 1e-14 floor at N = 96. The filtered rows read the C8 filter.
The subcycled C8 and C10 rows (7.2–7.8 over N = 48 / 96) turn to the
Hermite shell's time error by N = 192, `dt` 0.4–0.9.

**Time.** The temporal order of these rows is in [temporal order](#temporal-order).

### Repeated filtering

The inviscid wall under `:cascade3`, the total against the fine reference, wall window /
interior; `passes` is the number of filter applications:

```
N = 49                              passes   wall        interior     l2
unfiltered                             0     2.843e-7    5.135e-8     3.571e-8
cascade, every step, relaxed          97     2.891e-5    1.860e-5     8.905e-6
cascade, every 2nd step, relaxed      48     2.734e-5    1.488e-5     7.293e-6
cascade, every 4th step, relaxed      24     2.005e-5    1.071e-5     4.876e-6
cascade, every step, unrelaxed        97     3.565e-5    2.247e-5     1.120e-5
onesided, every step, relaxed         97     2.580e-7    4.400e-8     3.297e-8
onesided, every step, unrelaxed       97     2.865e-7    5.202e-8     3.651e-8
periodic mirror, cascade, every step  97     –           8.798e-11    4.030e-11  (unfiltered 8.131e-11)

N = 193
unfiltered                             0     1.217e-9    1.342e-10    7.650e-11
cascade, every step, relaxed         386     2.359e-6    1.137e-6     4.616e-7
cascade, every 2nd step, relaxed     193     2.189e-6    9.290e-7     3.611e-7
cascade, every 4th step, relaxed      96     1.592e-6    6.091e-7     2.245e-7
cascade, every step, unrelaxed       386     2.687e-6    1.378e-6     5.841e-7
onesided, every step, relaxed        386     1.254e-9    1.272e-10    7.277e-11
onesided, every step, unrelaxed      386     1.575e-9    2.924e-10    1.162e-10
periodic mirror, cascade, every step 386     –           2.827e-13    1.112e-13  (unfiltered 2.642e-13)
```

The cascade filter's wall error is not proportional to the number of passes: halving the
cadence at fixed spacing removes 4–7% of it and quartering it 30%, and the relaxation
removes 12–19%. The defect is the deposit of an `O(h²)` disturbance two cells from the wall
on every pass, propagated and damped by the wall rows and the interior operator rather than
set by the deposit rate alone; the error at 386 passes is 1900 times the unfiltered one at
N = 193, and every reduction of the cadence at a fixed spacing buys under a factor of two.
The one-sided rows under the same 386 passes are within
3% of the unfiltered error relaxed, and on the periodic mirror the eighth-order interior
pass moves the interior error by under 10% at every N, so the whole of the filter's wall
defect is the F2 row.

### The timestep floor

The finest grid of each family under the sixth-order closure, whose spatial error is small
enough for the time error to show, against the fine reference:

```
inviscid wall N=193, C6 BL     cfl 0.5      0.25        0.125       0.0625
  wall                          4.641e-12   8.802e-13   9.137e-13   8.751e-13
  interior                      3.976e-12   2.420e-13   4.396e-14   6.217e-14
periodic mirror N=193, C6
  interior                      4.018e-12   2.642e-13   3.564e-14   3.397e-14
entropy wave N=192, 3 levels, C6 BL, global step
  interface                     6.396e-11   6.391e-11   6.384e-11   6.394e-11
entropy wave N=192, 3 levels, C6 BL, subcycled
  interface                     8.007e-11   6.491e-11   6.397e-11   6.402e-11
```

The interior's time error is fourth order in the step (a factor of 15 from `cfl = 0.5` to
0.25, the fourth-order integrator's 16) and floors at 3e-14 against the reference's own
error; the wall window's closure defect is reached by `cfl = 0.25`, which is why the walls
run there and why the C8 rows, whose closure defect is below 1e-12 at N = 97, are measured
and not gated. At a level interface the global-step run is spatially limited at every step
tested, and the subcycled run's Hermite shell adds 25% at `cfl = 0.5` and 1.5% at 0.25.

### Temporal order

```text
julia --project=. -t 1 bench/temporalorder.jl [restrict=step|stage|none]
```

Each row integrates one case on one grid in equal steps through `run!` and reads the
maximum difference over the conserved components against the same case on the same grid
in at least 16 times as many steps, so the spatial error cancels. The inflow cases carry
their time dependence in the boundary data alone: a supersonic entropy wave through a
`DirichletBC` holding the exact state, and a uniform stream under an `NSCBCInflowBC` whose
target velocity and temperature oscillate, both with an `NSCBCOutflowBC`. `test/convergence.jl`
guards four rows on smaller grids. Steps, error and the orders between successive step
counts:

```
periodic standing wave, 96 nodes     20 1.248e-7   40 7.710e-9   80 4.787e-10  160 2.982e-11   4.02 / 4.01 / 4.00
slip walls, N = 49                   20 2.468e-7   40 1.700e-8   80 1.055e-9   160 6.794e-11   3.86 / 4.01 / 3.96
Dirichlet inflow g(t), N = 65        80 3.361e-7  160 2.280e-8  320 1.474e-9   640 9.367e-11   3.88 / 3.95 / 3.98
NSCBC inflow target(t), N = 65       40 5.517e-6   80 4.016e-7  160 2.422e-8   320 1.466e-9    3.78 / 4.05 / 4.05
filter, periodic, relaxed            20 1.248e-7   40 7.702e-9   80 4.790e-10  160 2.999e-11   4.02 / 4.01 / 4.00
filter, periodic, full strength      20 1.240e-7   40 6.949e-9   80 4.415e-10  160 6.947e-10   4.16 / 3.98 / -0.65
filter, walls, one-sided, relaxed    20 2.301e-7   40 3.953e-8   80 3.521e-8   160 1.683e-8    2.54 / 0.17 / 1.06
filter, walls, cascade, relaxed      20 4.774e-5   40 2.389e-5   80 9.042e-6   160 4.386e-6    1.00 / 1.40 / 1.04

dt = h/5 and h/10 by N (h/5 is cfl 0.64 on the inflow wave), order in h at h/5
                                     N = 33     65         129        257        orders in h         in dt at 257
periodic entropy wave, u0 = 2        2.539e-7   1.594e-8   9.960e-10  6.255e-11  3.99 / 4.00 / 3.99  3.84
Dirichlet data at the stage time     4.906e-7   5.470e-8   1.232e-8   2.929e-9   3.16 / 2.15 / 2.07  3.92
Dirichlet data through dq/dt         4.422e-7   3.119e-8   2.160e-9   1.702e-10  3.83 / 3.85 / 3.67  4.08
NSCBC target(t)                      1.347e-7   5.991e-8   2.575e-8   1.105e-8   1.17 / 1.22 / 1.22  4.04
NSCBC constant target, pulse         4.108e-7   2.296e-8   1.444e-9   9.074e-11  4.16 / 3.99 / 3.99  4.00

entropy wave, two levels, N = 96, t = 0.5, composite
global step, steps                   40         80         160        320
  default                            3.016e-8   1.535e-8   7.651e-9   3.726e-9    0.97 / 1.00 / 1.04
  ghost fluxes                       9.659e-11  2.236e-11  9.823e-12  4.836e-12   2.11 / 1.19 / 1.02
  C6 BL                              8.269e-11  2.805e-11  1.323e-11  6.320e-12   1.56 / 1.08 / 1.07
  default, restrict=stage            1.455e-10  9.680e-12  6.020e-13  9.326e-14   3.91 / 4.01 / 2.69
  default, restrict=none             1.455e-10  9.683e-12  6.133e-13  1.499e-13   3.91 / 3.98 / 2.03
subcycled, steps                     14         20         28         40         56         80
  default                            8.139e-8   5.789e-8   4.223e-8   3.014e-8   2.176e-8   1.532e-8    0.96 / 0.94 / 0.95 / 0.97 / 0.98
  ghost fluxes                       5.730e-9   1.405e-9   3.973e-10  1.202e-10  4.887e-11  2.455e-11   3.94 / 3.75 / 3.35 / 2.68 / 1.93
  C6 BL                              4.596e-9   1.144e-9   3.282e-10  1.016e-10  4.633e-11  2.692e-11   3.90 / 3.71 / 3.29 / 2.33 / 1.52
  default, restrict=none             4.115e-9   9.882e-10  2.573e-10  6.178e-11  1.608e-11  3.863e-12   4.00 / 4.00 / 4.00 / 4.00 / 4.00
  ghost fluxes, restrict=none        6.007e-9   1.427e-9   3.689e-10  8.807e-11  2.282e-11  5.436e-12   4.03 / 4.02 / 4.02 / 4.01 / 4.02
```

On a fixed grid the update is fourth order with the boundary data evaluated at the
stage time, Dirichlet and NSCBC alike. At a fixed ratio of step to spacing the temporal
error of time-dependent boundary data falls more slowly than the interior's h⁴: as h² for
the Dirichlet state, largest at the first nodes of the inflow, and as h^1.2 for the NSCBC
target, whose faces keep h⁴ with constant targets and a pulse in the interior. The same
Dirichlet data imposed through their time derivative on the face node's right-hand side
(a bench-local condition) keep 3.7–3.9. This is the boundary-data order reduction of
Runge–Kutta methods of stage order one (Carpenter, Gottlieb, Abarbanel and Don, SIAM J.
Sci. Comput. 16, 1995).

The relaxed filter is a splitting first order in the step: where a pass changes the
solution, as the cascade rows do at a wall, the error is first order, and on the smooth
periodic wave the change is below the integrator's error. At full strength the filter's
error grows with the number of passes.

The level interface is first order in the step under the package's schedule, global step
and subcycled alike, about 1/45 of the spatial interface error (1.342e-6) at 40 steps.
Injecting the fine solution into the covered parent nodes before every stage of the global
step (`restrict=stage`), or never (`restrict=none`), gives fourth order, so the term is the
once-per-step injection, after which the coarse operator advances the covered nodes for a
whole step. It scales with the interface's spatial defect: the ghost-flux and
Brady–Livescu rows are 770 and 590 times smaller than the default's at 320 steps. The
subcycled Hermite shell's fourth order shows above it at the largest steps.

### What the matrix settles

The wall order a run sees is the filter's, 1.8 under the cascade rows for any derivative
closure and the closure's own under the one-sided rows; every interface, same-level or
coarse-fine, reads the C6 cascade's 3.5 at its own spacing whatever the interior scheme. The
guards in `test/convergence.jl` are set from these runs and are bit-reproducible on this
workstation at one thread; they measure the walls against the mirror, so the matrix's fine
references are not needed by them. Two-dimensional walls with a tangential inviscid flow,
isothermal walls, moving refinement and Float32 are not in the matrix. A wall placed on the
half-offset mirror carries no closure row at all and reads round-off ([the face-centred
symmetry plane](#the-face-centred-symmetry-plane)).

### Grid convergence of the shock battery

```
N         | Noh1 plat  deficit | Lax L1  | mix width
128       |  0.9945       +60% | 1.3e-2  |  0.03581
256       |  0.9994       +59% | 7.1e-3  |  0.01820
512       |  0.9991       +58% | 3.9e-3  |  0.00931
1024      |  0.9992       +56% | 1.9e-3  |  0.00481
```

**Lax L1 halves per doubling**, giving first-order L1 convergence for the captured
discontinuity regardless of interior order; the sixth- and tenth-order convergence lives in
`test/convergence.jl`, on smooth fields. **Interface width halves per doubling**, so the
regularization follows the mesh and does not settle at a fixed physical scale, as required
for the Cook artificial properties to act as a subgrid model. **Wall heating does not
converge away**, 60% to 56% over an 8× refinement, the known character of the Noh problem:
the entropy error is deposited once, in the first cell at shock formation, and remains
there. Its spatial extent decreases with the cell size, so the integrated error vanishes
while the pointwise error does not.

## The filter's wall rows

```text
julia --project=. -t 1 bench/wallfilter.jl
```

`compact_filter` leaves row 1 unfiltered and applies centered F2/F4/F6 rows at rows 2–4. One
pass |F f − f| on a smooth closed line is second order along the whole line, since the compact
solve carries the row-2 error inward (1.88 in the max norm, 2.21 in L2), so the filter capped
the wall order of every filtered run with an O(h²) deposit two cells from the wall each step.

`compact_filter(closures = :onesided)` replaces rows 2–4 by the one-sided eighth-order rows of
Gaitonde and Visbal, derived from polynomial exactness through degree 7 plus a Nyquist zero as
the interior stencil is. They reproduce the centered stencil at the centered point to 1e-16
and the published row 2 to 1e-14, and one pass is eighth order everywhere (8.07 max norm, 8.75
L2). Rows 2 and 3 alone exceed unit gain at some wavenumbers (1.10 and 1.03 at αf = 0.45; 1.39
and 1.32 at αf = 0), yet ‖F¹⁰⁰‖₂ is 1.05 against the cascade's 1.14 at αf = 0.45 and N = 64,
and 1.42 against 1.35 only at αf = 0. Both keep every eigenvalue inside the unit disk apart
from the two exact ones at 1, the constant and the unfiltered end rows.

### The battery under both row sets

Every case of the battery has closed ends: the Dirichlet inflows of the tubes and the outer
boundaries of the folded Noh cases take the closure rows as a wall does. At the relaxed
default, `filter_cfl = 0.35`, C6 `:cascade3`:

```
case               rows       plateau   deficit   shock    inadmissible  e_min
Noh nu=1 N=400     cascade    0.9989    60%       0.2044   8             -0.0218
                   onesided   0.9975    50%       0.2043   7             -0.0176
Noh nu=1 N=800     cascade    0.9992    61%       0.2027   6             -0.0271
                   onesided   0.9977    43%       0.2026   6             -0.0356
Woodward N=800     cascade    L1 rho 3.217e-2  peak 6.6162 at 0.7785  rho_min 0.1479
                   onesided   L1 rho 3.222e-2  peak 6.6166 at 0.7785  rho_min 0.1478
Noh nu=2, nu=3, Lax, Shu-Osher, Sedov, shock/SF6: both row sets agree to every
                   digit printed
```

Unrelaxed, the gain at the shocked wall is larger, 64% to 29% at N = 400 and 66% to 30% at N
= 800, with the plateau 0.03% nearer the exact value and the front 0.002 nearer: relaxation
weakens every pass near the diffusion-limited front under `C_beta = 1`, shrinking the
cascade's F2 deposit there. The sets differ only where a wall carries a gradient, the nine
other rows agreeing under both at the folds, the tubes, the blast and the shocked interface.

### The closure-compatibility table

Every derivative closure under both sets on the wall-bounded shock cases, relaxed default:

```
case               derivative      cascade rows                            one-sided rows
Woodward N=800     C6 :cascade3    L1 3.217e-2, peak 6.6162                L1 3.222e-2, peak 6.6166
                   C6 :cascade4    L1 3.217e-2, peak 6.6161                negative density, step 2470, t = 0.019
                   C6 BL           negative density, step 851, t = 0.006   L1 3.218e-2, peak 6.6166
                   C8 BL           negative density, step 56               L1 3.229e-2, peak 6.6248
Noh cold N=400     C6 :cascade3    plateau 0.9989, deficit 60%             plateau 0.9975, deficit 50%
                   C6 :cascade4    plateau 0.9989, deficit 66%             plateau 0.9973, wall density 5.04 (overshoot 26%)
                   C6 BL           negative density, step 65               negative density, step 201
                   C8 BL           negative density, step 41               negative density, step 21
Noh warm t0=0.3    C6 :cascade3    rho[1:4] 3.990 3.994 3.998 4.001       4.025 3.988 3.993 4.007
                   C6 :cascade4    rho[1:4] 3.983 3.990 3.997 4.001       4.086 4.075 3.916 3.928
                   C6 BL           rho[1:4] 3.937 3.964 3.990 4.006       3.992 3.996 3.996 4.000
                   C8 BL           negative density, step 143, t = 0.024  3.895 3.995 4.001 3.997
```

`:cascade4` needs the F2 row, failing Woodward–Colella and overshooting the cold wall without
it; C6 Brady–Livescu needs the one-sided rows at a shock-bounded wall and takes no singular
start under either; C8 Brady–Livescu fails the cold start under both.

The C6 Brady–Livescu failure under the cascade rows is a wall mode driven by the artificial
bulk viscosity, not by the discontinuity. A planar Noh warm-started from the exact solution
holds a uniform ρ = 4, u = 0 plateau at the wall and nothing reaches the wall before the run
ends; with an unprimed first step and an unrelaxed weight the wall density under
`:brady_livescu` departed from 4 at t ≈ 0.06, doubled every ≈ 0.01 time units and ended the
run at ρ_wall = 20.3 against 3.99 under `:cascade3`. Zeroing `C_kappa` and `C_mu` together
left that growth unchanged, so β\* is the driver; zeroing `C_beta` loses the shock instead.
Where β\* is not active at a wall the rows are innocuous: a 1% Gaussian pressure pulse
between two slip walls holds `:cascade3`'s wall density to four digits over three acoustic
transits, and the mirrored Noh problem on (−1, 1) with inflow at both ends and no wall
completes with `:cascade3`'s profile. The scalar spectra do not predict the failure. The sets
are separated by the wall mode of D(β D), the diffusion operator the bulk term assembles from
the first-derivative rows, whose largest real eigenvalue at N = 64 and h = 1 is 1.3e-6 for
`:cascade3`, 1.7e-5 for `:cascade4`, 1.4e-4 for C6 `:brady_livescu` and 3.7e-3 for C8
`:brady_livescu` with the end rows free, and −2.5e-3 for all five with both end rows
injected; only the momentum is injected at a slip wall, so the density and energy rows see
the free-end spectrum.

### The reflected pulse

A left-moving simple wave of the ideal gas, `p = 1 + amp exp(-((x - 0.5)/0.05)^2)` with `rho
= p^(1/γ)` and `u = -2(c - c0)/(γ - 1)`, between slip walls on [0, 1] at `cfl = 0.4`,
against its periodic mirror on [0, 2) at the same spacing and step. The mirror carries the
pulse and its image about x = 1, so its restriction to [0, 1] is the wall problem at every
time and evaluates no closure row. The pulse reaches the wall near t = 0.42 and returns to
its origin near t = 0.85; the density is read at t = 0.7, the reflected pulse then in the
interior. At amp = 0.01 the wave steepens over about seven domain lengths and stays smooth;
at amp = 0.1 it shocks about 0.7 into its path.

```
amp 0.01, artificial properties on         N=49        97          193         385         orders (interior)
cascade    wall                            4.5e-6      1.2e-6      2.1e-7      2.9e-8
           interior                        2.70e-4     9.13e-5     2.00e-5     4.28e-6     1.56 / 2.19 / 2.22
           l2                              6.83e-5     1.90e-5     3.98e-6     8.77e-7
onesided   wall                            2.1e-4      7.8e-6      2.1e-7      8.1e-9
           interior                        3.19e-4     2.12e-5     8.93e-7     3.08e-8     3.91 / 4.57 / 4.86
           l2                              1.15e-4     5.25e-6     1.76e-7     6.09e-9

amp 0.1, t = 0.7 (shocked after the reflection)
                                           N=97        193         385         769
cascade    interior                        9.81e-4     5.86e-4     2.20e-4     8.07e-5     0.74 / 1.41 / 1.45
           l2                              1.63e-4     5.64e-5     1.68e-5     4.17e-6
onesided   interior                        1.99e-3     7.14e-4     7.71e-5     5.98e-7     1.48 / 3.21 / 7.01
           l2                              7.96e-4     2.28e-4     1.81e-5     1.16e-7
```

The artificial properties move nothing here, the amp = 0.01 rows with them off agreeing to
two digits. On the resolved pulse the one-sided rows are a hundred times more accurate by N =
385 and converge at 4.5–4.9 where the cascade rows cap at 2.2; at N = 49, where σ is 2.4
cells, they are worse, and on the steepening pulse they are worse up to N = 193, equal at 385
and a hundred times better at 769. The mid-resolution penalty is larger with the artificial
properties off, so it is not a sensor or bulk-viscosity coupling but the pre-asymptotic
behaviour of rows 2 and 3, whose gain exceeds unity at high wavenumber, on a reflection
resolved over fewer than about ten cells. The crossover moves to coarser grids as the
amplitude falls (N = 193 at amp = 0.03).

### Conservation and floor budgets

The runs filter through a callback with the solver's own pass off, so the change of the
totals across each pass is the filter's alone. `drift` is the change over the run relative to
the largest total seen and `filter` the passes' share; the remainder is the derivative
closure's, which is not summation-by-parts, and the floor's.

```
                                     mass drift (filter)        energy drift (filter)      floor
Woodward N=800, cascade              -3.81e-6 (-3.21e-6)        -1.93e-5 (-1.57e-5)        off
Woodward N=800, onesided             -1.77e-7 (+5.71e-8)        -1.32e-7 (+3.74e-8)        off
Woodward N=800, cascade, floor 1e-6  -3.80e-6 (-3.20e-6)        +2.35e-2 (-1.53e-5)        3257 steps, 15186 cells
Woodward N=800, onesided, floor 1e-6 +5.61e-7 (+2.11e-8)        +2.35e-2 (+2.61e-8)        3254 steps, 15169 cells
steepening pulse N=385, cascade      -2.85e-7 (-2.27e-7)        -4.68e-7 (-3.66e-7)        off
steepening pulse N=385, onesided     -2.91e-8 (+5.11e-10)       -4.26e-8 (+1.01e-9)        off
planar Noh N=400, cascade            filter -1.09e-3            filter +1.61e-4            0 cells under floor 1e-6
planar Noh N=400, onesided           filter -3.31e-5            filter -1.12e-5            0 cells under floor 1e-6
```

Between two walls the one-sided rows cut the run's mass and energy drift by twenty to a
hundred times, and the filter's own share of it by two orders. The floor, where on, fires on
the same steps and repairs the same cells to a tenth of a percent under both sets, and a
species layer against the wall reads the same mass-fraction range, interface width and
species masses to six digits.

### Float32

Planar Noh at N = 400 in Float32 reads the Float64 rows to every digit printed under both
sets. The smooth pulse against its mirror floors near 1e-5 in Float32 from N = 97 under both,
the one-sided wall window three times the cascade's at N = 193 and its interior and `l2`
below it; the wider row carries more round-off but no floor of the Brady–Livescu kind.
Woodward–Colella at N = 800 completes in Float32 under neither set, losing positivity in the
first blast, a limitation of the 10⁵ pressure ratio, so Noh alone carries the comparison.

### The decision

`compact_filter` takes `closures = :onesided` by default. The wall window of a filtered run
then converges at the derivative closure's own order instead of 1.8, a resolved wall-bounded
evolution carries a hundred to a thousand times less error, the filter creates two orders
less mass and energy on any closed line, the planar Noh wall deficit is 10–18 points smaller
under the relaxed default and 35 under the unrelaxed one, and the C6 and C8 Brady–Livescu
rows run at a shocked wall. The cost is a two- to threefold penalty on a reflection resolved
over fewer than about ten cells, a plateau 0.15% lower on the relaxed Noh wall, and pairing
`:cascade4`, which needs the F2 row, with `compact_filter(closures = :cascade)`. The
validation guards were re-baselined and the stored references regenerated; regression rows
asserting a cascade measurement pin it, as they pin `filter_cfl = 0`.

The cascade's F2 row was damping a linear slip-wall instability of the derivative closure
exactly; removing it exposed that mode, measured under [constant
annihilation](#constant-annihilation) and removed by the neutral closure rows.

## Wall closures in production

```text
julia --project=. -t 1 bench/wallclosure.jl
julia --project=. -t 1 bench/closurequalify.jl parts=dilatation schemes=neutral3,brady_livescu
julia --project=. -t 1 bench/closurequalify.jl parts=jacobian schemes=neutral3 jns=51,101 jwalls=slip
```

Unless a table says otherwise the case is the standing wave or the shear mode of
`test/smooth_cases.jl` at N = 49 / 97 / 193, `cfl = 0.25`, t = 0.4, artificial properties on,
under `compact_filter(0.45, closures = :onesided)` every step with `filter_cfl = 0.35`, and
the error is the four-node wall window against the periodic mirror at the same spacing, which
carries no closure rows. The closure sets are C6 `:neutral3` (the default), `:cascade3`, C6
and C8 `:brady_livescu`, and the archived fifth-order rows of [the fifth-order closure
search](#the-fifth-order-closure-search).

### Smooth walls with the artificial properties on

Wall errors under `beta_sensor = :dilatation` at CFL 0.25, then successive orders at CFL 0.25
and 0.125; all 450 derivative-only paired comparisons reach t = 0.4. These are field-specific
evolution orders; the derivative rows retain their formal boundary moments.

| rows | case | errors, N = 49 / 97 / 193 | orders, CFL 0.25 | orders, CFL 0.125 |
|---|---|---|---|---|
| neutral3 | inviscid slip | 5.802e-07 / 3.987e-08 / 2.588e-09 | 3.86 / 3.95 | 3.87 / 3.95 |
| neutral3 | viscous no-slip | 4.159e-07 / 2.999e-08 / 2.006e-09 | 3.79 / 3.90 | 3.80 / 3.90 |
| neutral3 | adiabatic shear | 1.439e-08 / 6.176e-10 / 2.442e-11 | 4.54 / 4.66 | 4.55 / 4.66 |
| neutral3 | isothermal shear | 1.437e-08 / 6.173e-10 / 2.441e-11 | 4.54 / 4.66 | 4.55 / 4.66 |
| Brady–Livescu | inviscid slip | 2.459e-09 / 4.851e-11 / 6.630e-13 | 5.66 / 6.19 | 5.67 / 6.11 |
| Brady–Livescu | viscous no-slip | 1.952e-09 / 4.409e-11 / 8.038e-13 | 5.47 / 5.78 | 5.47 / 5.77 |
| Brady–Livescu | adiabatic shear | 1.942e-10 / 1.692e-12 / 1.342e-14 | 6.84 / 6.98 | 6.84 / 6.97 |
| Brady–Livescu | isothermal shear | 1.942e-10 / 1.692e-12 / 1.354e-14 | 6.84 / 6.97 | 6.84 / 6.97 |

The isothermal rows read the adiabatic ones to three digits under every closure, so the
isothermal flux path adds no closure defect on a compatible case; the viscous slip wall with a
tangential shear reads the viscous no-slip row to three digits, tabulated under [the wall flux
contracts](#the-wall-flux-contracts). The fine shear errors lie near 1e-14, so the last pair's
near-seventh-order slopes touch the floating-point floor.

**The inviscid slip wall isolates the strain sensor's cusp.** Wall errors at N = 193 and CFL
0.25 under the three sensor settings, archived fifth-order rows beside the production sets:

| rows | strain | dilatation | properties off |
|---|---|---|---|
| neutral3 | 2.582e-09 | 2.588e-09 | 2.588e-09 |
| Brady–Livescu | 5.441e-11 | 6.630e-13 | 7.976e-13 |
| unfiltered-search | 4.762e-11 | 7.934e-13 | 9.137e-13 |
| filtered-objective | 7.462e-11 | 1.369e-12 | 1.403e-12 |
| DE | 1.135e-10 | 1.479e-12 | 1.499e-12 |

For Brady–Livescu the reduction is a factor of 82, and the same sensor change removes the
inviscid cap for all four fifth-order coefficient sets; the neutral closure's larger
truncation error hides it. Dilatation is not a general improvement: the viscous standing waves
and the two shear contracts already agree with their properties-off controls under both
sensors. Unfiltered, the inviscid and viscous waves under dilatation read 3.99 / 4.04 and 4.01
/ 4.01 for `:neutral3` and 5.86 / 5.77 and 5.32 / 5.91 for Brady–Livescu, at both CFL numbers.

### Which channel carries the residual

The inviscid slip wall under C6 Brady–Livescu with one constant at a time, the constants
zeroed with the machinery enabled, and the other smoother, detector and sensor fields,
node-centred sensor rows throughout:

```
variant                       N=49        97          193         orders
off                           2.430e-9    4.919e-11   7.976e-13   5.63 / 5.95
all on                        2.260e-8    4.866e-10   5.441e-11   5.54 / 3.16
C_mu only                     2.461e-9    4.854e-11   6.681e-13
C_beta only                   2.255e-8    4.853e-10   5.425e-11
C_kappa only                  2.430e-9    4.916e-11   8.524e-13
all zero, enabled             2.430e-9    4.919e-11   7.976e-13
smoother = :compact           3.088e-8    3.969e-10   5.033e-11   6.28 / 2.98
detector = :d8                3.698e-9    9.936e-11   1.305e-11   5.22 / 2.93
mu_sensor = :velocity         2.255e-8    4.853e-10   5.428e-11
beta_sensor = :dilatation     2.459e-9    4.851e-11   6.630e-13   5.66 / 6.19
```

The constants zeroed with the machinery enabled reproduce the properties-off row bitwise, so
the coefficient arithmetic adds nothing. **β\* carries the whole residual**: the `C_beta` row
equals the all-on row to three digits, and μ\* and κ\* sit where their constants put them.
The carrier is the strain sensor's cusp, not an operator's wall closure. `detect_sum!` under
`:delta4` on a field exactly even about both walls reproduces the periodic mirror bitwise at
the first six nodes; the dilatation crosses zero in the interior at node 98 of the N = 193
run, where β\* reads 5.505e-8 against a 3e-13 wall-region background, identically in the wall
run and the mirror run; the strain run's wall-region β\* is noisy node to node where the
dilatation run's is smooth and matches its own mirror to between 1e-5 and 1e-2; and the two
agree at step 1 and at 2.5e-13 wall error at step 143, separating late to 5.4e-11 (strain)
and 8.2e-13 (dilatation) at t = 0.4, with a wall-localized difference profile. Under
`:dilatation` the two detectors are indistinguishable at the wall, so `:d8` carries the same
cusp limit as `:delta4`.

Under the default closures every variant reads its own closure error, a thousand times the
residual, so the effect is invisible there.

### The reflected pulse under each closure

The pulse of [the filter's wall rows](#the-filters-wall-rows), density against its mirror at
t = 0.7, `cfl = 0.4`, artificial properties on:

```
amp 0.01                 N=49        97          193         385         interior orders
C6 cascade3  wall        2.087e-4    7.791e-6    2.070e-7    8.060e-9
             interior    3.186e-4    2.116e-5    8.929e-7    3.083e-8    3.91 / 4.57 / 4.86
             l2          1.149e-4    5.252e-6    1.762e-7    6.090e-9
C6 BL        wall        1.953e-4    4.389e-6    4.569e-8    1.532e-10
             interior    2.343e-4    1.402e-5    3.137e-7    1.586e-8    4.06 / 5.48 / 4.31
             l2          1.037e-4    3.542e-6    7.785e-8    3.207e-9
C8 BL        wall        1.418e-4    5.951e-7    1.493e-9    2.293e-11
             interior    3.347e-4    1.927e-5    4.248e-7    2.894e-8    4.12 / 5.50 / 3.88
             l2          1.345e-4    4.486e-6    8.424e-8    4.254e-9

amp 0.1 (shocked after the reflection)
                         N=97        193         385         769
C6 cascade3  wall        3.323e-3    7.459e-4    6.037e-5    1.424e-7
             interior    1.987e-3    7.143e-4    7.711e-5    5.980e-7    1.48 / 3.21 / 7.01
C6 BL        wall        4.538e-4    2.702e-5    2.957e-7    1.281e-8
             interior    1.247e-3    5.425e-4    6.126e-5    6.435e-7    1.20 / 3.15 / 6.57
C8 BL        wall        3.415e-3    4.148e-4    1.700e-6    2.100e-8
             interior    2.547e-3    7.812e-4    1.162e-4    2.470e-6    1.71 / 2.75 / 5.56
```

At the wall window the Brady–Livescu rows are fifty times more accurate than the cascade by N
= 385 on the smooth pulse and two hundred times on the steepening one; in the interior, where
the reflected pulse sits at t = 0.7, the three closures agree within a factor of two at every
N above 49, the error there being the filter's and the time integrator's. On the coarsest
grid, where the pulse is 2.4 cells wide, every closure reads the same 1e-4. Under `:neutral3`
the pulse reads 1.192e-9 at N = 385 and amplitude 0.01, and 1.906e-7 at N = 769 and 0.1.

### The stable CFL range

The inviscid wall at N = 97 with the artificial properties on, against the mirror at the
same CFL number, which carries the interior scheme and the filter but no closure row:

```
cfl     C6 cascade3   C6 BL         C8 BL                        mirror
0.25    1.379e-8      1.134e-9      1.354e-9                     completes
0.50    1.542e-8      1.135e-9      1.330e-9                     completes
0.75    1.672e-8      1.565e-9      2.608e-9                     completes
1.00    1.742e-8      1.435e-9      2.342e-9                     completes
1.25    1.782e-8      1.097e-9      negative density, step 30    completes
1.50    1.805e-8      8.453e-10     negative density, step 12    completes
1.75    1.809e-8      1.224e-9      negative density, step 8     completes
2.00    8.542e-5      5.458e-3      negative density, step 8     completes
```

C6 Brady–Livescu holds its wall error flat to `cfl = 1.75`, as the cascade does; at 2.0 both
complete but their difference from the mirror jumps four orders. C8 Brady–Livescu fails from
1.25 with its mirror completing, so the T8 rows carry a stability limit of their own at a
smooth wall. On the shocked cases C6 Brady–Livescu completes every case at every CFL number
the cascade does, reads Woodward–Colella's `L1` within 0.1% of the cascade's from 0.15 to 1.2,
and holds the warm Noh wall within 0.2% of 4 to `cfl = 1.2` where the cascade's first node
reads 4.01–4.03; C8 Brady–Livescu completes Woodward–Colella to 1.2 within 1% but departs from
the warm Noh wall from `cfl = 0.6` (4.15, then 4.34 at 0.9), fails it at 1.2, and fails the
steepening pulse from 1.2.

### How resolved a start must be

Planar Noh at N = 400 from the exact solution at t0, `cfl = 0.15`. The front is at t0/3, the
cell is 0.0025, and the warm start blends the plateau into the inflow over four cells about
the front, so below t0 ≈ 0.03 the initial state is not the exact one under any closure:

```
t0      front (cells)   C6 cascade3                C6 BL                       C8 BL
0.3     40              4.025 3.988 3.993 4.007    3.992 3.996 3.996 4.000     3.895 3.995 4.001 3.997
0.1     13              4.057 3.968 4.021 4.010    3.985 3.976 4.029 4.005     negative density, t = 0.075
0.03    4               4.365 4.662 4.788 4.360    12.446 6.769 4.785 4.210    negative density, step 3
0.01    1.3             11.403 4.930 3.655 3.756   4.931 4.989 4.167 4.030     negative density, step 20
0.003   0.4             90.740 4.308 2.892 3.764   negative density, t = 0.18  negative density, step 21
0.0     singular        1.990 2.560 3.288 3.743    negative density, t = 0.006 negative density, step 21
```

`StepControl(retries = 4)` changes no row, so this is not the startup restriction of [the CFL
section](#the-cfl-restriction-and-the-symmetry-cell). C6 Brady–Livescu holds the wall wherever
the cascade does; the two sets differ on the singular start, which the cascade completes with
its deficit and the rows do not complete at all. C8 Brady–Livescu takes only the 40-cell
start. On the current solver, with the sensor mirrors and the slip-wall flux contract in
place, both Brady–Livescu and the archived fifth-order unfiltered-search rows complete the
cold N = 200 preset that failed earlier, so those older failure steps are not current
evidence; the change appears in the strain control too and is not attributable to the sensor.

### The round-off floor

One derivative of `exp(sin(3x))` on the closed line, Float64, wall window:

```
N        C6 cascade3   C6 BL        C8 BL
193      9.075e-6      7.219e-9     2.872e-10
385      1.118e-6      1.689e-10    6.901e-12
769      1.387e-7      7.837e-12    1.063e-11
1537     1.728e-8      5.329e-11    2.491e-11
3073     2.155e-9      6.106e-11    7.566e-11
orders   3.0 throughout  5.4 / 4.4 / floor  5.4 / floor, rising
```

The cascade converges at 3.0 to the last grid with no floor in sight. The Brady–Livescu rows
floor near 1e-11 on a derivative of magnitude 8, C6 from N ≈ 800 and C8 from N ≈ 400, the C8
floor rising with N as the condition number does; that floor is four orders below the cascade's
error there, and the cascade would reach it near N = 10⁴, so in Float64 the rows are the more
accurate set at any resolution a run uses. The interior sits 20–100 times below the wall.

The conditioning does show in Float32 on the same derivative: the sets floor between 1e-3 and
5e-3 absolute and rise with N (C6 BL 1.02e-3, 1.21e-3, 2.80e-3, 4.52e-3 at N = 24, 48, 96, 192
against the cascade's 6.24e-3, 6.88e-4, 8.99e-5, 9.82e-5), so from N = 48 up the default
closure is the more accurate one in Float32, the precision the device path runs at, which
`test/float32_validation.jl` pins. That floor does not reach the solution: a Float32 reflected
pulse floors near 3e-5 from N = 97 under every closure where Float64 reads 8e-9 (cascade) and
1.5e-10 (C6 BL) at N = 385, and on the steepening pulse the C6 Brady–Livescu wall window is
half the cascade's. Float32 is neither a reason to avoid the rows nor a reason to prefer them.

### The minimum extent

`plan_direction` raises when a block is too short for its rows, so the minimum is read by
construction, on one rank (both ends of the dimension closed) and on two ranks split along it:

```
scheme            both ends closed   one end closed
C6 cascade3       5                  5
C6 cascade4       5                  5
C6 BL             9                  5
C8 cascade3       7                  7
C8 BL             13                 7
C10               7                  7
filter, either    9                  9
```

The filter's nine points bind every configuration but C8 Brady–Livescu with both ends closed,
which needs thirteen; C6 Brady–Livescu changes no minimum extent. The decomposed solve
reproduces the rows' polynomial exactness across an eight-way split, degree 5 under C6 and 7
under C8 to 1e-9, which `test/mpi_tests.jl` checks beside the cascade's degree 3.

### A two-dimensional wall

The Cartesian Noh plane at N = 24, AR = 2 (`noh_cartesian`, the exact inflow on four faces,
so the closure rows see an oblique inflow and four corners):

```
                   plateau   center deficit   front x / y / diag        L1 rho   steps
C6 cascade3 cold   11.862    44%              0.2357 0.2328 0.2343      0.907    759
C6 BL cold         11.883    44%              0.2357 0.2314 0.2344      0.940    817
C6 cascade3 warm   16.574    13%              0.2213 0.2244 0.2252      0.462    281
C6 BL warm         16.574    13%              0.2214 0.2244 0.2252      0.482    279
C8 BL cold         timestep collapse at t = 0.25
C8 BL warm         negative density at t = 0.20
```

C6 Brady–Livescu reads the cascade's plateau, deficit and fronts to three digits on both
starts, the cold one included, since this case's singular start is at the centre of the plane
and no closure row sees it; the rows fail only where singular data sit on the row itself.

### The production Jacobian and the uniform state

Centered Jacobians of the production step about a uniform state (ρ = 0.9, tangential 0.1, p
= 1.1), artificial properties off, cfl 0.5, the one-sided filter unrelaxed where on, N = 51
unless stated. The ladder at 3e-6 and 3e-5 moves the neutral readings by ±5e-9 and the
cascade's by 1e-10:

```
                                      C6 :neutral3   C6 :cascade3   C8 :neutral3   C8 :cascade3   C10 :neutral3  C10 :cascade3
slip, unfiltered                      1.0000000000   1.0175743678   1.0000000001   1.0104697113   1.0000000001   1.0198057979
slip, one-sided filter                1.0000000011   1.0094669437   1.0000000033   1.0085087244   1.0000000035   1.0081446666
slip, unfiltered, N = 101             1.0000000001   1.0090626412   1.0000000001   1.0082617272   1.0000000001   1.0119295843
slip, one-sided filter, N = 101       1.0000000010   1.0040111557   1.0000000032                  1.0000000034   1.0029634131
Dirichlet ends, unfiltered            1.0000000000   1.0000000000   1.0000000001                  1.0000000001   1.0000000001
Dirichlet ends, one-sided filter      1.0000000018   1.0000000006   1.0000000023                  1.0000000008   1.0000000027
no-slip μ = 0.005, one-sided filter   1.0000000186   1.0000000070   1.0000000216   1.0000000079   1.0000000256   1.0000000093
2-D slip box 13 × 13, unfiltered      1.0000000001   1.0369419374
```

The 2-D row is the full production Jacobian of a square between four slip walls, corners
included. An unfiltered viscous no-slip wall reads 1 + 1e-7 to 3e-7 for every closure set, the
cascade included, and is not a closure effect. The uniform state under the default relaxed
filter every step, cfl 0.5, Float64, max |u_n| at t = 10 / 20 / 30 / 40:

```
N = 51    C6 :neutral3    2.2e-14  3.9e-14  3.1e-14  3.4e-14
          C6 :cascade3    4.0e-10  7.0e-05  negative density at t = 29.75
          C8 :neutral3    2.7e-14  3.5e-14  4.1e-14  3.2e-14
          C8 :cascade3    7.1e-11  3.9e-06  2.0e-03  3.7e-03
          C10 :neutral3   1.9e-14  3.6e-14  3.7e-14  3.4e-14
          C10 :cascade3   3.6e-11  1.2e-06  2.9e-03  1.8e-03 (saturated)
N = 101   C6 :neutral3    2.5e-14  2.9e-14  4.9e-14  8.0e-14
          C6 :cascade3    9.2e-11  2.9e-06  9.1e-02  negative density at t = 36.09
          C8 :neutral3    2.2e-14  2.1e-14  5.1e-14  5.3e-14
          C8 :cascade3    1.7e-11  8.4e-08  3.5e-04  1.2e-03
          C10 :neutral3   2.8e-14  4.1e-14  6.7e-14  1.2e-13
          C10 :cascade3   3.1e-12  4.5e-09  7.1e-06  8.0e-04
```

The neutral rows hold the uniform state at round-off for forty time units at every interior
and both line lengths, where the cascade rows grow the slip-wall mode of [constant
annihilation](#constant-annihilation) and two of the runs lose positivity.

The neutral rows' accuracy cost against the cascade, N = 49 / 97 / 193:

```
one derivative of exp(sin 3x), wall window     8.57e-4  1.01e-4  1.22e-5   vs  6.29e-4  7.46e-5  9.07e-6
standing wave, slip walls, t = 0.4, wall       7.22e-7  4.57e-8  2.78e-9   vs  2.84e-7  1.95e-8  1.22e-9
  orders                                       3.98  4.04                  vs  3.87  4.00
standing wave, no-slip μ = 0.005, wall         5.76e-7  3.60e-8  2.24e-9   vs  2.34e-7  1.58e-8  1.02e-9
reflected pulse, art off, t = 0.7, wall        3.04e-5  7.58e-7  3.49e-8   vs  2.63e-4  4.26e-6  1.08e-8
  interior                                     2.97e-4  2.78e-5  1.47e-6   vs  3.79e-4  2.14e-5  9.17e-7
  l2                                           8.32e-5  5.27e-6  2.31e-7   vs  1.39e-4  4.59e-6  1.42e-7
```

The wall-window error after an evolution is 2.5 times the cascade's at the same order (1.35
times on one derivative, where only row 1 enters); the pulse's wall-window error is eight
times smaller and its interior error 1.5 times larger from N = 97. On the battery the neutral
rows read the cascade's ν = 2 and ν = 3 rows in every printed digit and the planar wall in the
fourth digit.

### The decisions

**C6 `:neutral3` is the default.** Every long inviscid run between slip walls or symmetry
planes grew the cascade's mode, and the alternatives were a knob (`compact_filter(closures =
:cascade)` with its second-order wall defect, viscosity, or Brady–Livescu with its cold-start
failure); the cost is a factor 2.5 in the wall error constant at unchanged orders and a
fourth-digit move of the planar Noh battery. `:cascade3` stays available for comparison with
earlier results, and the same two neutral rows are the C8 and C10 defaults ([closure
certificates](#closure-certificates)).

Two consequences follow from the rows' weights rather than their stability. A Float32
freestream at a wall is no longer exact: the cascade's dyadic weights annihilated a constant
in floating point, and the neutral rows' thirds leave the round-off of their products, 2.2e-6
on the walled Cartesian case of `test/float32_validation.jl`. And the flux divergence at a
patch or level interface end keeps one-sided rows, a flux array carrying no ghosts; under the
neutral rows the entropy-wave interface-window errors were two to five times larger, so
`interface_divergence_closures` keeps the cascade rows there, an interface imposing no
injected condition, and the interface baselines are unchanged to every printed digit.

**C6 `:brady_livescu` with the default `compact_filter` is a supported wall configuration**
within these limits: a wall whose initial state is resolved (the front thirteen cells out on
the warm Noh ladder; no singular start on a closure row), any CFL number the default closure
completes the case at, Float64 or Float32, block extents no smaller than the filter requires,
serial or decomposed. Its wall solution is sixth order with the artificial properties off and
fourth order with them on under the strain sensor, at an error fifteen times below the
cascade's, and under `beta_sensor = :dilatation` it recovers the closure's own order at an
error eighty times smaller again. `test/validation.jl` guards it on Woodward–Colella and the
warm Noh wall.

**C8 `:brady_livescu` is not supported at a wall**: the T8 rows fail a smooth wall from `cfl
= 1.25` with the mirror completing, the warm Noh wall from 0.9, the two-dimensional plane on
both starts, and every start of the planar case but the 40-cell one. They remain available
for a periodic or interior block.

## The wall flux contracts

```text
julia --project=. -t 1 bench/boundaryorder.jl wall_only=true
julia --project=. -t 16 test/wall_flux_tests.jl
```

A wall correction sets the assembled normal flux before halo exchange and compact divergence
rather than subtracting terms from it, so it also removes the normal `:bulk` component flux
and is independent of the EOS energy gauge. It changes no derivative or filter coefficient.

### The no-slip wall

Each species flux is zero at an impermeable noncatalytic wall. Adiabatic total-energy flux is
zero; isothermal energy flux is `-(mu0 * cp_mix / Pr + kappa_art) * grad_T_ion[d]`, removing
species enthalpy transport while retaining pressure and viscous traction. On the incompatible
linear-temperature probe the two energy fluxes change from `[-0.005, -0.005]` to exactly
`[0.0, 0.0]`; the regression also compares the second compact right-hand-side row with the
uncorrected slip-wall case, so an endpoint-only patch cannot satisfy it.
`test/wall_flux_tests.jl` runs in Float64 with the default C6 closure, unit length and
density, ideal gas R = 1 and gamma = 1.4, and no filtering or artificial transport:

| Check | Measurement | Regression guard |
|---|---|---|
| Insulated conduction, N=33 / 65, temperature max error | 1.3972e-6 / 3.0304e-7 | fine error <5e-7 and reduction >4 |
| Same, absolute trapezoidal domain-energy drift | 4.0894e-8 / 4.3029e-9 | fine drift <1e-8 |
| Species cosine diffusion, N=65, mass-fraction max error | 2.4444e-7 | <3e-7 |
| Isothermal, initial integrated RHS minus boundary heat rate | 2.7576e-6 | absolute defect <4e-6 |
| Isothermal, evolved energy rate minus time-averaged boundary heat rate | 1.8716e-6 | absolute defect <3e-6 |

The insulated temperature is `1 + 0.08 exp(-alpha*4pi^2*t) cos(2pi*x)` with `mu0=0.015`,
`Pr=0.8` and final time 0.002, an analytic momentum source balancing its pressure gradient so
that conduction evolves through the computed energy right-hand side; the species case uses
`Y1=0.5+0.1 cos(2pi*x)`, `mu0=0.012` and `Sc=0.75`, pressure and temperature uniform to
roundoff. At N=65 the instantaneous mixed temperature/composition probe measures interior
right-hand-side max errors 3.6932e-6 (species) and 7.7558e-6 (energy), both guarded at 1e-5.
The isothermal case starts at `T=1+0.05 sin(pi*x)` with `Twall=1`, `mu0=0.01` and final time
2e-5; the residual between its energy-change rate and the trapezoidal integral of the endpoint
heat rates includes spatial quadrature and hard temperature enforcement, and is neither a
residual normal species flux nor a claim of exact discrete conservation. An N=33 filter-only
probe with high-frequency cosine fields changes the trapezoidal species-1 mass by +4.5253e-4
and total energy by +1.1230e-3 while the adiabatic wall fluxes stay exactly zero, so no global
filter-conservation claim follows from this hook.

Coverage: the direct face tests seed artificial conductivity and species diffusivity
independently of the detector, `:fickian` and `:bulk` both, over both precisions, all six
physical faces and their corners, ideal/NASA-9 and stiffened-gas EOS, nonsingular
cylindrical/spherical metrics, `SwitchableBC` and KernelAbstractions CPU execution. The
24-case `bench/wallflux.jl` matrix also passes on a Radeon RX 6800 XT at zero maximum CPU/GPU
evolved-state difference for every precision, normal, thermal condition and species channel.
The abstract face dispatch costs a fixed 16 B per active face and nothing proportional to wall
area (48³ right-hand side 208 B before and 304 B after; 48³ five-stage step 1552 B and 2032 B;
the isothermal wall hook 0 B at every size; JET 1 / 2 before and 2 / 3 after, the expected
`correct_flux!` dispatch).

### The slip wall

Without a `correct_flux!` method the fluxes assembled at a slip-wall node reach the compact
divergence as the interior formulas produce them, which under a physical shear viscosity
leaves a conductive heat flux across an adiabatic symmetry plane and a shear traction on it,
and the near-wall solution error stops converging. The case is the standing wave between slip
walls at N = 49 / 97 / 193, `cfl = 0.25`, t = 0.4, C6 `:neutral3`, artificial properties off,
the density in the four-node wall window against the periodic run at the same spacing.
Filtered, at `filter_cfl = 0.35`:

| μ | path | N = 49 | N = 97 | N = 193 | orders |
|---|---|---|---|---|---|
| 5e-3 | no hook | 5.733e-6 | 4.336e-6 | 3.991e-6 | 0.40 / 0.12 |
| 5e-3 | contract | 4.140e-7 | 2.996e-8 | 2.005e-9 | 3.79 / 3.90 |
| 5e-4 | no hook | 6.567e-7 | 9.289e-8 | 5.167e-8 | 2.82 / 0.85 |
| 5e-4 | contract | 5.309e-7 | 3.549e-8 | 2.304e-9 | 3.90 / 3.95 |

Unfiltered, which is what `test/convergence.jl` runs, the uncorrected orders are 0.91 / 0.24
under `:neutral3` and 0.59 / 0.12 under `:cascade3` at μ = 5e-3, and 3.06 / 1.25 and 2.48 /
1.01 at μ = 5e-4.

**Which flux carries it.** A slip-wall node in a one-dimensional standing wave carries an
exactly zero species flux, ΣY = 1 making the correction velocity cancel the single species'
diffusive flux, and an exactly zero tangential momentum flux, the transverse dimensions being
collapsed. Imposing either alone reproduces the uncorrected row digit for digit (5.733e-6 /
4.336e-6 / 3.991e-6, 0.40 / 0.12), while imposing the energy flux alone reproduces the full
contract (4.140e-7 / 2.996e-8 / 2.005e-9, 3.79 / 3.90). The energy flux at the wall node is
the conductive term alone: at N = 97, μ = 5e-3 and t = 0.4 the four components read species 0,
normal momentum 1.070, tangential momentum 0, and energy −2.1378e-6, equal to −(μ c_p/Pr)
∂T/∂n to every digit where the mirror run's coincident node reads −7.2e-15. The convective,
viscous work, enthalpy and `:bulk` terms vanish because u_n is enforced to zero on the plane.

**Why the order collapses.** The closure rows themselves are not at fault: on the exactly
even initial data they return ∂T/∂n at their own order, third for `:neutral3`. That value
against time, filtered, μ = 5e-3:

| path | N | t = 0 | t = 0.01 | t = 0.05 | t = 0.1 | t = 0.2 | t = 0.4 |
|---|---|---|---|---|---|---|---|
| no hook | 49 | 3.901e-6 | 1.353e-5 | 4.258e-5 | 6.050e-5 | 8.973e-5 | 1.133e-4 |
| no hook | 97 | 4.885e-7 | 4.237e-6 | 1.624e-5 | 3.134e-5 | 5.704e-5 | 8.551e-5 |
| no hook | 193 | 6.109e-8 | 1.833e-6 | 1.038e-5 | 2.290e-5 | 4.608e-5 | 7.429e-5 |
| contract | 49 | 3.901e-6 | 1.050e-5 | 1.628e-5 | 1.060e-5 | 3.822e-6 | 1.404e-6 |
| contract | 97 | 4.885e-7 | 1.970e-6 | 1.663e-6 | 1.139e-6 | 4.298e-7 | 1.225e-7 |
| contract | 193 | 6.109e-8 | 2.608e-7 | 1.957e-7 | 1.353e-7 | 5.210e-8 | 1.313e-8 |

The t = 0 column converges at 3.00 / 3.00. Without the contract the t = 0.4 column converges
at 0.41 / 0.20; with it, at 3.52 / 3.22. The closure rows' truncation error is a heat flux
across a plane that conducts none; the temperature defect it leaves in the first few cells
regenerates the gradient, and the two settle at a level that no longer follows h.

**The tangential traction.** The one-dimensional case cannot separate the tangential momentum
flux, zero there. A two-dimensional case can: slip walls on both ends of x, periodic in y over
16 nodes, ρ = 1 + 0.05 cos(πx)(1 + 0.3 cos y), u = 0.05 sin(πx) cos y, v = 0.05 cos(πx) sin y,
p = ρ^1.4, μ = 5e-3, t = 0.2, against the periodic run on the doubled x domain, the error
being the maximum density difference over the plane:

| imposed at the wall node | N = 25 | N = 49 | N = 97 | orders |
|---|---|---|---|---|
| nothing | 1.066e-5 | 4.155e-6 | 3.109e-6 | 1.36 / 0.42 |
| energy flux | 1.017e-5 | 7.699e-7 | 1.080e-7 | 3.72 / 2.83 |
| the contract | 1.035e-5 | 6.701e-7 | 4.355e-8 | 3.95 / 3.94 |

The energy flux carries most of the defect here too and the tangential traction the rest. A
one-dimensional case with a tangential velocity reproduces the contract's 3.79 / 3.90 with the
artificial properties on or off, and its uncorrected row is within 2% of the same case without
a tangential component, so the traction contributes little without wall-parallel variation.

**The contract.** `correct_flux!(::SlipWallBC, ...)` writes zero on the owned wall plane for
every species flux, every tangential momentum flux and the total normal energy flux, leaving
the normal momentum flux untouched; that flux carries the pressure, the normal viscous stress
and the β\* dilatational term, all even about the plane. The wall stays adiabatic:
`SlipWallBC` carries no wall temperature and a symmetry plane admits no conductive exchange.
With the artificial properties on, the same three components lose their κ\* ∂T/∂n, D\* ∂Y/∂n
and tangential μ\* contributions at every slip wall; on a smooth single-species field at `mu0
= 0` only κ\* ∂T/∂n survives and κ\* is negligible there, so the inviscid smooth rows do not
move and the battery's shocked slip walls do.

**What moved.** The two battery cases that reflect a Noh implosion off a slip wall lose wall
heating, the κ\* flux the wall was conducting being one of its sources: the planar plateau
moves 3.9883 to 3.9988 and its deficit 50% to 24%, the aligned case 3.9792 to 4.0035 and 54%
to 33%. The aligned case's transverse round-off grows by a factor of 27 ([the aligned Noh
transverse mode](#the-aligned-noh-transverse-mode)); every other battery row holds to the
digits printed. No row of `test/convergence.jl` moved: with `mu0 = 0` and the artificial
properties off the wall-plane fluxes the contract writes are already exactly zero, so it is a
no-op there, and the file's new row is the viscous slip wall at 4.00 beside the viscous
no-slip wall's 4.00. The inviscid slip-wall rows of `bench/wallclosure.jl` hold to four digits
and the smooth reflected pulse is unchanged; the shocked pulse moves in the third digit, where
the reflection leaves a nonzero κ\* at the wall. The viscous slip-wall table matches the
viscous no-slip table to three digits at every closure:

| closure | properties | N = 49 | N = 97 | N = 193 | orders |
|---|---|---|---|---|---|
| C6 `:neutral3` | off | 4.145e-7 | 3.000e-8 | 2.009e-9 | 3.79 / 3.90 |
| | on | 4.165e-7 | 3.003e-8 | 2.010e-9 | 3.79 / 3.90 |
| C6 `:cascade3` | off | 1.544e-7 | 1.251e-8 | 8.727e-10 | 3.63 / 3.84 |
| | on | 1.554e-7 | 1.253e-8 | 8.730e-10 | 3.63 / 3.84 |
| C6 Brady–Livescu | off | 1.972e-9 | 4.406e-11 | 8.094e-13 | 5.48 / 5.77 |
| | on | 1.972e-9 | 4.407e-11 | 8.074e-13 | 5.48 / 5.77 |
| C8 Brady–Livescu | off | 1.996e-10 | 9.719e-13 | 1.843e-14 | 7.68 / 5.72 |
| | on | 2.052e-10 | 1.007e-12 | 1.554e-14 | 7.67 / 6.02 |

## Constant annihilation

```text
julia --project=. -t 1 bench/constantfloor.jl
```

A periodic or interior row differences its taps before it multiplies, so a constant is
annihilated exactly whatever its size. A closure row is a weighted sum over the first points
of the line, so a constant c leaves (Σ w_j) c from the weights' rounding plus the rounding of
the products and their accumulation, of order eps · c · Σ|w_j| / h before the line solve
amplifies it. At issue is whether that residual affects an evolution or a useful precision
range, and whether the anchored form Σ w_j (f_j − f_1), exact on a constant, buys anything.
Every derivative and filter preset is measured in both precisions with the plan's prescaled
coefficients, at N = 33, 129 and 513; the normalized residuals do not depend on N.

**The weight sums.** The cascade rows' weights are dyadic rationals and sum to zero exactly
in both precisions, stored and prescaled by 1/h when L = 1 makes 1/h an integer; at L = 2π
the prescaled row 1 sums to −1.8e-15 (Float64) and −9.5e-7 (Float32). The Brady–Livescu rows
sum to 1e-16 to 4e-15 stored in Float64 and 1e-7 to 2e-6 in Float32, and the prescaled sums
scale with 1/h. The filter rows' right-hand sides sum to their left-hand sides exactly in
Float64 (4e-16 at worst) and to 1e-7 to 4e-7 in Float32; the Gaussian and the Pyranda filter
likewise.

**A constant on the closed line.** The closure rows' fill residual before the solve and the
solved residual at the wall window and in the interior, normalized to eps-scale (× h / c for
a derivative, / c for a filter), N = 129, the generic constant 12345.678, L = 2π. The
anchored fill reads exactly zero on every derivative row and 2.5e-8 at most on a filter row:

```
                     Float64: fill      solved wall   interior  |  Float32: fill    solved wall   interior
C6 cascade3          1.7e-16   3.9e-16       6.0e-18  |  3.1e-8   7.0e-8        1.1e-9
C6 cascade4          4.3e-16   2.5e-15       3.9e-17  |  1.1e-7   6.4e-7        9.8e-9
C6 BL                5.2e-16   1.4e-14       4.3e-16  |  3.6e-7   2.3e-6        4.7e-8
C8 cascade3          3.2e-17   4.7e-17       9.6e-18  |  (as C10)
C8 BL                9.6e-15   1.6e-14       9.8e-16  |  2.3e-6   1.1e-5        5.7e-7
C10                  1.7e-16   3.8e-16       7.5e-18  |  3.1e-8   1.1e-7        1.6e-8
filter onesided      5.9e-16   1.2e-15       1.0e-15  |  5.0e-7   8.7e-7        4.8e-7
filter cascade       3.0e-16   4.4e-16       1.0e-15  |  1.8e-7   4.0e-7        2.4e-7
gaussian             1.5e-16   1.5e-16       0        |  1.0e-7   7.9e-8        7.9e-8
pyranda filter       0         9.0e-15       1.5e-14  |  9.4e-8   6.9e-6        2.9e-5
```

The three sources separate: the weights' own sums are at most a tenth of the fill residual
(row 2 of C8 Brady–Livescu excepted, at half); the products and their accumulation are the
rest of the fill, 2–40 eps; and the solve amplifies the cascade's fill by 2 and the
Brady–Livescu fill by 5–40, their closed-line condition numbers showing. The interior carries
a tenth to a hundredth of the wall residual. In absolute terms the derivative of a constant c
reads eps · c / h × (2–40) at the wall: with c = 1e5 and h = 0.01 that is 1e-9 in Float64 and
1–20 in Float32.

**A perturbation over a constant.** f = c + sin(3x) against 3 cos(3x), N = 129, errors
relative to 3, the wall window under the plain and the anchored rows against the interior,
which is the floor the stored field itself sets:

```
                  Float64: c=0      1e3       1e6       1e9     |  Float32: c=0      1         1e3       1e6
C6 cascade3 plain      3.9e-7    3.9e-7    4.2e-7    2.4e-5   |  2.3e-6    1.2e-5    8.8e-3    1.9e+1
            anchored   3.9e-7    3.9e-7    4.0e-7    1.2e-5   |  5.8e-7    1.2e-5    2.4e-3    4.6e+0
            interior   6.2e-9    6.2e-9    7.7e-9    4.0e-6   |  2.4e-6    6.8e-6    2.5e-3    2.4e+0
C6 BL       plain      3.4e-10   2.1e-9    5.9e-7    4.6e-4   |  4.7e-5    4.4e-4    3.8e-1    3.6e+2
            anchored   3.4e-10   3.4e-10   7.2e-8    3.3e-5   |  3.5e-5    1.0e-4    1.5e-2    1.4e+1
            interior   3.9e-12   4.8e-11   1.5e-8    1.4e-5   |  2.4e-6    1.3e-5    1.2e-2    1.1e+1
```

The anchored rows recover a factor of two to four for the cascade and ten to thirty for
Brady–Livescu at the wall, within a factor of two of the interior's floor. That floor is the
stored field's quantization, eps · c, differentiated, which no row can remove; the difference
shows only where the interior is already at 1e-3 relative in Float32 (c = 1e3 over a unit
perturbation) or 1e-6 in Float64 (c = 1e9). In Float64 an offset below 1e6 moves nothing.

### The consequence in a run

A uniform state on 101 nodes between slip walls, with a tangential velocity so the x-momentum
flux is the constant pressure through the wall rows and every other x flux is zero, the
artificial properties on, a periodic line as the control. The right-hand side of the initial
state, x-momentum, wall window / interior:

```
                       Float64 cascade3    Float64 BL         Float32 cascade3   Float32 BL
p = 1 (exact)          0 / 0               3.5e-12 / 9.4e-14  0 / 0              4.6e-4 / 1.1e-5
p = 1.1, rho = 0.9     3.2e-14 / 4.9e-16   5.3e-12 / 1.4e-13  2.6e-5 / 3.9e-7    2.9e-4 / 1.3e-5
p = 1e5, rho = 1.2     0 / 0               3.0e-7 / 8.2e-9    1.1 / 1.7e-2       17 / 0.25
periodic               0 / 0               0 / 0              0 / 0              0 / 0
```

The wall-normal velocity |u| and the pressure's relative drift after 500 / 1000 / 2000 steps
at cfl 0.5, wall window:

```
                                   Float64                              Float32
p = 1.1, cascade3, filter off      1.3e-14 / 9.2e-13 / 9.5e-9          1.0e-5 / 7.2e-4 / 6.2e-3
p = 1.1, cascade3, filter on       4.8e-14 / 2.9e-13 / 1.2e-11         2.5e-5 / 1.4e-4 / 1.2e-3
p = 1.1, BL, filter off            1.4e-14 / 3.8e-14 / 3.5e-14         1.0e-5 / 2.9e-5 / 1.1e-5
p = 1.1, BL, filter on             2.8e-14 / 4.5e-14 / 3.9e-13         1.5e-5 / 5.7e-5 / 1.5e-4
p = 1.1, periodic, filter on       7.4e-15 / 5.6e-15 / 3.1e-15         2.7e-6 / 4.3e-6 / 1.5e-5
p = 1e5, cascade3, filter off      0 / 0 / 0                           1.6e-3 / 1.3e-1 / 1.9
p = 1e5, cascade3, filter on       4.9e-12 / 2.3e-11 / 8.0e-10         3.0e-3 / 1.5e-2 / 2.6e-1
p = 1e5, BL, filter off            1.8e-12 / 7.4e-12 / 4.5e-12         1.1e-3 / 6.5e-3 / 1.1e-2
p = 1e5, periodic, filter on       8.2e-13 / 1.4e-12 / 1.3e-12         3.5e-3 / 6.4e-3 / 1.1e-2
```

The periodic control with the filter on drifts at 1e-5 relative in Float32 over 2000 passes,
interior rows included: the filter's own constant-passing round-off, the same order as the
closure rows' residual, which anchoring the derivative rows would not touch. Under the cascade
rows with a nonzero seed the velocity accumulates neither linearly, as a constant residual
would give, nor as √steps, as a random walk would: it multiplies by 70 per thousand steps in
Float64 and Float32 alike, while the Brady–Livescu rows hold their seed and the periodic
control holds its own. That growth is a mode of the closed line, and the round-off seed only
sets when it becomes visible.

### The slip-wall mode

The wall-normal velocity from the round-off seed at 1000 / 2000 / 4000 steps in Float64, the
generic uniform state, and the growth rate per unit time fitted between the last two
readings (c = 1.31 on a unit domain, so a unit of time is 0.76 acoustic transits):

```
                                              1000        2000        4000       rate
N=101 C6 cascade3, filter off                 1.1e-12     9.5e-9      6.4e-3     +1.79
N=101 C6 cascade3, filter on (relaxed)        3.1e-13     1.2e-11     1.8e-8     +0.96
N=101 C6 cascade4, filter off                 2.2e-2      negative density at step 1680
N=101 C6 cascade4, filter on                  2.1e-2      negative density at step 1362
N=101 C6 BL, filter off                       4.4e-14     5.1e-14     1.0e-13    +0.09
N=101 C6 BL, filter on                        7.0e-14     3.9e-13     6.3e-12    +0.36
N=101 C6 cascade3, filter off, art off        1.1e-12     9.5e-9      8.4e-2     +2.12
N=101 C6 cascade3, filter off, v = 0          3.5e-13     2.2e-9      6.6e-3     +1.98
N=101 C6 cascade3, filter off, v = 0.5        2.2e-12     2.2e-8      6.6e-3     +1.69
N=51  C6 cascade3, filter off                 1.7e-9      9.6e-3      1.1e-2     saturated
N=201 C6 cascade3, filter off                 4.0e-14     1.7e-12     3.3e-9     +1.99
N=101 C6 cascade3, filter off, cfl 0.25       2.2e-14     1.3e-12     1.3e-8     +2.41
```

One step linearized about the uniform state by centered differences at N = 51, cfl 0.5,
artificial properties off (their sensors are not differentiable there, and the rows above
show they do not set the rate, only the saturation near
|u| = 1e-2): the largest eigenvalue modulus of the amplification matrix as a rate
per unit time, the count of eigenvalues outside the unit circle, and the share of the leading
eigenvector's norm within four nodes of a wall. A filtered row applies the unrelaxed pass
after the step.

```
                                              |λ|max        rate      growing   wall share
C6 cascade3, unfiltered                       1.01757437    +2.28     54 / 255  0.48
C6 cascade3, onesided filter                  1.00946695    +1.23     56 / 255  0.43
C6 cascade3, cascade filter                   1.00000004    +0.00      6 / 255  0.03
C6 cascade3, onesided filter, αf = 0.40       1.01075467    +1.40     59 / 255  0.42
C6 cascade3, onesided filter, αf = 0.30       1.01397722    +1.82     56 / 255  0.54
C6 cascade4, unfiltered                       1.05524354    +7.03     55 / 255  0.80
C6 BL, unfiltered                             1.00000002    +0.00     48 / 255  0.98
C6 BL, onesided filter                        1.00443450    +0.58     14 / 255  0.13
C8 cascade3, unfiltered                       1.01046971    +1.36     60 / 255  0.33
C8 BL, unfiltered                             1.00100131    +0.13     53 / 255  0.14
C10 (cascade rows), unfiltered                1.01980580    +2.57     51 / 255  0.47
C6 cascade3, no-slip, μ = 0.005               1.00000031    +0.00      1 / 255  0.59
C6 cascade3, slip, μ = 0.005                  1.00028758    +0.04     10 / 255  0.16
C6 cascade3, Dirichlet ends, unfiltered       1.00000001    +0.00     49 / 255  0.01
C6 cascade3, Dirichlet ends, onesided filter  1.00000004    +0.00      8 / 255  0.01
C6 cascade3, unfiltered, N = 101              1.00906264    +2.36    110 / 505  0.27
```

The centered difference at `delta = 1e-5 · max(|Q|, 1)` resolves |λ| to about 1e-9, so a row
reading 1.0000000x is neutral and its growing count is round-off; the `ladder` keyword repeats
a row at 3e-6 and 3e-5 and the readings move by ±1e-9. The linearized rates reproduce the
time-domain ones, so the growth is a linear instability of the discrete step at an inviscid
slip wall under the cascade closures: the same rate at N = 51, 101 and 201 and at half the
step, so an O(c/L) mode and not a grid mode, with half of its eigenvector within four nodes of
the walls. The cascade filter's F2 row damps it exactly, which is why it went unseen; the
one-sided rows halve it, and strengthening them makes it worse, rows 2 and 3 of that set
exceeding unit gain at high wavenumber. Physical viscosity at a no-slip wall removes it and at
a slip wall leaves 0.04; Dirichlet ends are neutral, so the shock tubes and the Noh inflows
are outside it. Under the defaults a Float64 seed reaches |u| = 1e-2 in about 30 time units
and a Float32 seed in about 15, and the artificial properties then hold it there; the
battery's wall cases end at t = 0.6 or earlier and do not see it, while a long inviscid run
between slip walls or symmetry planes does. The neutral closure rows remove the mode ([closure
certificates](#closure-certificates)), and it is absent from a face-centred symmetry plane.

### The decision on anchored rows

No change to the closure rows for constant annihilation. The residual a closure row leaves on
a constant is round-off in every precision: 2–40 eps relative to c/h at the wall, a tenth to a
hundredth of that in the interior, from the products and their accumulation, the weights' own
sums below that and the solve's amplification above it for the Brady–Livescu sets. Anchoring
would make a uniform state's residual exactly zero and buy a factor of two to thirty on a
perturbation over an offset, but only where the stored field's quantization already floors the
interior at the same level, c/f' above 1e6 in Float64 and above 1e3 in Float32; in a filtered
Float32 run the filter's own constant-passing round-off drifts the interior at the same 1e-5
per 2000 passes. The slip-wall mode sets the practical consequence of the seed, and no
anchoring changes its growth. The Float32 SI-unit case, a 1 Pa/m spurious gradient at a wall
under p = 1e5 Pa and 1.9 m/s by t = 0.03 s through the mode, is a precision-policy matter and
not a closure one.

## Closure certificates

```text
julia --project=. -t 16 bench/closurecertify.jl
julia --project=. -t 1 bench/neutralsearch8.jl parts=scan grid=81
julia --project=. -t 1 bench/neutralsearch10.jl parts=validate,scan,wide,errors
julia --project=. -t 16 bench/closurecertify.jl wall=folded schemes=c6,c8,c10 parts=spectrum,pseudo,transient
```

The instability of [constant annihilation](#constant-annihilation) is removed by new closure
rows: `:neutral3` is the default of `lele_d1_6`, `lele_d1_8` and `lele_d1_10`, the same two
rows over each interior. The production checks are in [wall closures in
production](#wall-closures-in-production).

### The instruments

`closurecertify.jl` builds the closure rows from the exact rationals of the (a, b, c) family,
checks them against `lele_d1_6()` and `lele_d1_6(closures = :cascade3)`, and measures the
injected acoustic operator L on the 2N − 2 unknowns (p on every node, u on the interior
nodes) at c = L = 1, so rates are per unit time and ‖L‖₂ grows in proportion to N. Byers'
test gives the pseudospectral abscissa in fourteen bisections per α_ε; a dense minimum of
σ_min((x + iy)I − L) over y reproduces ε to 1e-6 relative. `rtol` floors α_ε at `rtol · ‖L‖₂`,
7e-8 at N = 201 under the default 1e-10, so readings at ε ≤ 1e-7 use `rtol = 1e-13`.
‖exp(tL)‖₂ comes from one warm-started subspace eigendecomposition, verified against a dense
operator norm at nine times to 2.3e-12 relative over 646 times, so the last window is a lower
bound and the eigenvector condition number the rigorous one.

Beside it, a linear model of the production step: the closed-line derivative matrix D (read
off the production operator, or assembled from the closure rows; the two agree to 6e-14), the
Euler flux Jacobian at the uniform state, the five-stage low-storage Runge–Kutta tableau, and
the wall's normal momentum zeroed at each stage as `enforce!` does, its energy correction
quadratic and dropping out of the linearization. The amplification matrix agrees with the
production Jacobian to 8e-11 in every entry at N = 51, and the instability is already in the
semi-discrete operator: the injected operator's largest real part is 2.31 per unit time
against the measured 2.28. The acoustic pair drives the entropy and tangential components and
not the reverse, so the reduced system (p, u) on 2N unknowns carries the whole spectrum and
evaluates in a few milliseconds at N = 51. Its unstable eigenvalue has imaginary part 89 at
N = 51, 4.6 points per wavelength: a near-grid-scale mode with an O(1) growth rate, removed
exactly by the second-order F2 row and not by the one-sided filter rows, whose gain exceeds
one at high wavenumber.

`neutralsearch8.jl` and `neutralsearch10.jl` define the three-row families below, verify them
against the production plans by differentiating unit vectors (agreement 4e-16 to 9e-16,
banded assembly included), scan at N = 51 and 101, sweep the finalists over every N from 12
to 600 and every tenth to 1200, and run the production Jacobian and the forty-time-unit
uniform state. One growth evaluation at N = 51 is 2 to 5 ms; a 649-length sweep is 220 to
300 s per member.

### The C6 family

The cascade's rows are the unique third-order three-point row 1 and the unique fourth-order
three-point Padé row 2; widening each by one point frees three coefficients at fixed order:

```
g_1 + a g_2 = Σ_{k=1}^{4} w_k f_k          third order, a free
b g_1 + g_2 + c g_3 = Σ_{k=1}^{5} w_k f_k   fourth order, b and c free
```

with (a, b, c) = (2, 1/4, 1/4) the cascade and a = 3 the `:cascade4` row 1. A scan of 16,900
grid points over a ∈ [0, 6], b, c ∈ [−1, 1.5] at N = 51 found 98 at which every eigenvalue of
the injected acoustic operator lies on the imaginary axis to 1e-14, none with a growth rate
between 1e-6 and 1e-2, and the rest unstable; the fourth-order five-point row-1 family with
the same row 2 has no neutral member. Only 4 of the 98 stayed neutral over twelve line
lengths between 51 and 296. Mapped in the (b, c) plane at fixed a over eight line lengths,
the neutral set is a curved band about 0.1 wide in c, running from (b, c) ≈ (0.56, 0.45) to
(0.8, −0.1) at a = 0, with a hole near b = 0.68. A sweep over every N from 301 to 600 and
every tenth to 1200 separated two rational members: (1/4, 3/5, 1/5) grows at up to 0.042 per
unit time at N = 371 and every 44th node count after it, while (0, 3/5, 3/10) reads below
3e-12 at every N from 12 to 1200. Within the a = 0 band the error constants fall with b, so
the adopted member is the one at the band's low-b edge that passes the sweep:

```
row 1  (0, 1, 0)        [-11/6, 3, -3/2, 1/3]                 explicit, third order
row 2  (3/5, 1, 3/10)   [-59/40, 41/30, -3/10, 1/2, -11/120]  compact, fourth order
```

The closed line's condition number is 5.0 against the cascade's 16. The other a = 0 band
members, (16/25, 9/50), (33/50, 7/50), (7/10, 1/25) and (19/25, −3/50), pass the same
production Jacobians and complete cold Noh at deficits 52.9–53.1%, with wall errors rising
with b; (1/4, 3/5, 1/5), (0, 4/5, 0) and (1/2, 1/2, 1/2) are neutral at N = 51 and 101 but
fail the N sweep, the Dirichlet ends or the filtered step, so two resolutions do not select a
member.

**Two alternatives to new rows are closed negative.** A filter mirroring only the normal
acoustic pair, replacing p and u_n on rows 2–4 of each wall by their node-centered
parity-mirror filtered values, is neutral (1.0000000021 at N = 51) and holds a uniform state
at 2e-14 to t = 40, but the standing wave's wall order falls from 3.84 to 3.40, its interior
order from 4.22 to 3.03, and the reflected pulse's wall order from 5.9 to 3.1; one pass of
either filter on the smooth solution agrees with the exact mirror pass to 1e-14, so the loss
comes from the two row sets' different response to the wall truncation content the cascade
rows inject every step. Brady–Livescu rows on the flux divergence only, with the gradient,
sensor and filter plans left on the cascade, have a Jacobian equal to full Brady–Livescu's
row for row and fail the singular cold planar Noh start at step 61 against full
Brady–Livescu's step 201, with cfl 0.05 moving the failure without removing it; the converse
swap completes the case, so the divergence rows are necessary and sufficient for the failure,
and the hybrid forfeits the rows' accuracy at every gradient-bearing wall.

### The pseudospectral measurement

The adopted rows (0, 3/5, 3/10), the neighbour (1/4, 3/5, 1/5), and the cascade as the
unstable control. ε is absolute, in the units of L's entries; `max κ` is the largest
Bauer–Fike eigenvalue condition number ‖v‖‖w‖; K(L) is the Kreiss constant sup α_ε/ε over ε
= 1e-2 to 1e-8:

```
                 N    ‖L‖₂     max Re λ    cond(V)   max κ    α_ε/ε             K(L)
adopted         25   8.55e1   +3.6e-15     3.08     1.2478   1.2477–1.2478     1.25
adopted         51   1.78e2   +2.1e-14     5.24     1.9728   1.9661–1.9662     1.97
adopted        201   7.13e2   +1.1e-13     5.05     1.9281   1.9281–1.9283     1.93
adopted        371   1.32e3   +3.4e-13     6.34     2.4085   2.4084–2.4122     2.41
adopted        801   2.85e3   +6.3e-13     8.75     3.3020
neighbour      201   6.86e2   +9.6e-14     9.40     3.8852   3.8850–3.8854     3.89
neighbour      371   1.27e3   +2.011e-2    7.33     3.0771   22 … 2.0e4        ∞
neighbour      415   1.42e3   +2.869e-2    5.28     2.2405   30 … 2.9e4        ∞
neighbour      801   2.74e3   +1.1e-12     8.96     3.7703
cascade         51   2.22e2   +1.767       34.2     9.1372   180 … 1.8e8       ∞
cascade        201   8.89e2   +1.775      150.5    37.133    181 … 1.8e8       ∞
cascade        801   3.56e3   +2.073      124.6    29.277
```

For a neutral member α_ε equals max κ · ε to four or five digits at every ε over six decades,
so the ε-pseudospectrum is the first-order eigenvalue perturbation and there is no non-normal
amplification. The Kreiss constant of the adopted rows is 1.25 to 2.41 over N = 25 to 415 and
the eigenvector condition number 3.1 to 8.8 to N = 801, without a trend in N. For the
cascade, and for the neighbour at its resonant node counts, α_ε tends to the positive
spectral abscissa as ε → 0 and the Kreiss constant is unbounded; the cascade's α_ε above its
abscissa is again κ · ε, so its instability is spectral and not pseudospectral. The bound
‖|V||V⁻¹|‖₂ grows linearly with N and is of no use.

The maximum of ‖exp(tL)‖₂ over t ∈ [0, 20], the maximum over the fast window [0, 50/‖L‖₂],
and the same maximum in the trapezoid quadrature norm of (p, u):

```
                 N    max ‖e^{tL}‖₂   at t     fast window   energy norm
adopted         25       2.677        1.03       2.6108        2.020
adopted         51       3.783       18.13       2.6102        3.756
adopted        201       3.677        8.00       2.6102        3.617
adopted        401       3.138       16.13       2.6102        3.110
neighbour      201       7.809       13.87       2.5578        7.788
neighbour      401       5.163        7.33       2.5578        5.157
cascade         51     8.1e15        20.00       8.2996       7.3e15
cascade        401     5.3e17        20.00       8.2666       5.2e17
```

The adopted rows amplify by at most 3.8 over twenty time units at every N from 25 to 401, the
Euclidean and quadrature-norm figures agree to three digits, and the initial fast transient
is independent of N. The spectral abscissa is below 1e-12 at every node count measured, so
cond(V) bounds ‖exp(tL)‖ for all time, at 8.75 out to N = 801.

### The resonance

Over N = 340 to 470 the adopted rows read at most 8.5e-13 and the neighbour leaves the axis
at N = 371, 415 and 459 only. The C6 modified wavenumber k′(θ) = (2 · (7/9) sin θ + 2 ·
(1/36) sin 2θ)/(1 + (2/3) cos θ) peaks at 1.9894 at θ = 2.2671, so below that frequency the
interior carries two wavenumbers for one frequency and each closed line holds two ladders of
modes, one per branch. The colliding pair at N = 371 is one mode from each branch:

```
   N     ω          Re λ        peaks θ (weight)              branch     detuning
  370   500.0375   −1.2e-14    1.3600 (1.00)                  1
  370   502.6665   −8.5e-14    2.7801 (1.00)                  2         −0.3746
  371   503.0798   ±2.011e-2   1.3651 (1.00), 2.7801 (0.40)   merged     0
  372   503.1503   +7.7e-14    1.3626 (1.00)                  1
  372   503.4540   +3.9e-14    2.7827 (1.00)                  2         +0.3037
```

The two eigenvalues merge into a quartet ±0.0201 ± 503.08i at N = 371 and separate again at
N = 372, so the bubble is narrower than one node count and the unstable N are isolated
points. The modes are propagating, not evanescent wall modes, with 0.27 to 0.38 of their norm
within eight nodes of a wall against 0.21 for a uniform profile. Every resonance sits at the
same frequency and the same wall phases (M = N − 1):

```
   N     rate        ωh        θ₁        θ₂        θ₁M/π      θ₂M/π
  371   +2.011e-2   1.35968   1.36495   2.78161   160.7569   327.6032
  415   +2.869e-2   1.35882   1.36407   2.78192   179.7581   366.6015
  459   +3.182e-2   1.35813   1.36336   2.78216   198.7587   405.6001
  503   +3.236e-2   1.35755                       217.7589   444.5988
  547   +3.112e-2   1.35707                       236.7588   483.5976
  591   +2.829e-2   1.35666                       255.7584   522.5964
```

ωh is fixed at 1.3585 ± 0.0015 (4.62 points per wavelength), the fractional parts of θ₁M/π
and θ₂M/π are fixed at 0.758 and 0.600, and the integer parts advance by 19 and 39 per 44
nodes. The period follows from the interior dispersion alone: along k′(θ₁) = k′(θ₂) with r =
dθ₂/dθ₁ = k″(θ₁)/k″(θ₂), a resonance recurs at a node-count step ΔM with Δm = round(θ₁ΔM/π)
when Δq = r Δm + ((θ₂ − r θ₁)/π) ΔM is an integer; at the measured frequency r = −0.346815
and (θ₂ − r θ₁)/π = 1.036097, and ΔM = 44 gives |Δq − round Δq| of 0.0012 against 0.025 or
more for every other step under seventy. It predicts 371, 415, 459, 503, 547 and 591, which
the scans found, and nothing else between N = 300 and 620. The period is a property of the C6
interior row and not of the closure, which sets the two wall phases and with them whether the
coincidence family lands on integer node counts. Frequency coincidence alone is not
sufficient: the branch-2 mode comes within 0.019 of a branch-1 partner at N = 379 for the
adopted rows and within 0.008 at N = 394 for the neighbour, and both stay neutral there.

### The structural attempt

Two conditions were solved exactly, at a four-node wall depth. First, a symmetric H with H D
+ Dᵀ H supported on the two wall corners, the admissible set the null space of the support
constraint taken by SVD, and the minimum eigenvalue maximized over it at tr H = N. A
well-conditioned positive-definite H within 15% of the identity exists for all three
closures, the cascade included (λ_min/λ_max +0.848 for the adopted rows at N = 51, +0.834 for
the neighbour, +0.517 for the cascade, at constraint residuals of 4e-16), so the support
condition does not separate a neutral closure from an unstable one; a reading that it did
came from a least-squares projection stalled at a relative residual of 1e-4.

Second, the exact Lyapunov certificate: for a real L with imaginary spectrum, P = Re(V⁻ᴴ
V⁻¹) is symmetric positive definite with P L + Lᵀ P = 0 to round-off, and ‖exp(tL)‖ ≤
√cond(P) for all time. It exists, at cond(P) of 10.5 (N = 21) and 27.5 (N = 51) for the
adopted rows and 1170 for the cascade, whose residual is 7.1e-4 rather than 1e-15 since its
spectrum is not imaginary. It has no structure: P is block diagonal in (p, u) with dense
blocks, its interior diagonal varies by a factor of seven without settling, and a corner of
the N = 51 solution at depth 6, 8 or 10 padded with the identity interior leaves an
off-corner residual of 2.5e-4 to 2.7e-4 at N = 51 to 401, falling like 1/N and never
vanishing, with under 4% change from depth 6 to 10. No fixed-depth corner block of the
Sharan, Brady and Livescu form emerges.

**Prior art.** The recipe is published; the result is not. Carpenter, Gottlieb and Abarbanel
(ICASE 91-71, J. Comput. Phys. 108, 1993, §7) widened compact closure rows into three- and
four-parameter families and searched for a left-half-plane spectrum on scalar advection with
a Dirichlet inflow; Zingg and Lederle widened the two boundary rows of explicit schemes by
one point each and fixed the two-parameter family by the spectrum; Brady and Livescu
(Computers & Fluids 183, 2019) fixed free boundary coefficients by the stability of the
injected nonlinear Euler problem, with conservation constraints, at one order below the
interior. Row 1 of the neutral set is Lele's (1992, eq. 4.1.3) third-order one-sided family
at α = 0, the explicit difference. Sharan, Brady and Livescu (SIAM J. Numer. Anal. 60, 2022)
treat strongly imposed boundary conditions by the energy method: their test system U_t + U_x
= 0, V_t − V_x = 0 with U(0) = τ₁ V(0), V(1) = τ₂ U(1) is the slip-wall acoustic pair in
characteristic variables, and at τ = 1 their Figure 2 shows the CGA closures, Strand's SBP
stencils and the Cook–Riley C6 compact scheme with eigenvalues in the right half-plane, the
mode measured here; their remedy is a nonsquare operator with full-norm corner blocks and a
skew Q = HD outside them, for explicit interiors up to 3-6-3, and their Theorem 1 is the
Lyapunov certificate above. Not found in any of these: a compact interior with exactly
neutral closures, the reflecting two-wall injected operator as the selection problem, a
two-parameter neutral set, or an instability at particular line lengths only (the literature
expects the boundary-dependent spectrum to be independent of N, Beam and Warming 1993; the
nearest frameworks are CGA's N-parity-dependent neutral example, eq. 78–80, and
Bonnet-Eymard, Coulombel and Faye, arXiv:2504.00667, on wave packets coupling two
boundaries). A neutral spectrum of a non-normal operator is necessary and not sufficient.

### The C8 family

The C8 interior reaches ±3, so a closed edge takes three rows. Widening each cascade row by
one point at fixed order:

```
g_1 + a g_2 = Σ_{k=1}^{4} w_k f_k                third order,  a free
b g_1 + g_2 + c g_3 = Σ_{k=1}^{5} w_k f_k         fourth order, b and c free
d g_2 + g_3 + e g_4 = Σ_{k=1}^{6} w_k f_k         sixth order on 2d + e = 1
```

Six weights and two left-hand-side coordinates against seven conditions leave the sixth-order
members of row 3 on a line, not a plane: the degree-6 residual is 12 − 24d − 12e, and the
line passes through the cascade's (1/3, 1/3). Off the line the row is fifth order. The family
reproduces the C8 cascade rows and the C6 `:neutral3` rows exactly, and the assembled line
agrees with the production plan to 4.4e-16. The neutral count on a 41 × 41 grid in (b, c) ∈
[−1, 1.5]² at N = 51 and 101, against a and against the position of row 3 on the sixth-order
line:

```
  a \ d      0     1/5    1/4    1/3    9/20
  0        132     73     76     60     71
  1/2       50     22     22     28     40
  1         12      6      7      7     17
  3/2        3      0      1      4      1
  2          1      0      0      0      1
  5/2        0      0      0      0      0
```

The neutral set again sits at a = 0, an explicit third-order row 1, and is empty from a = 5/2,
as for C6, but at a = 0 it is a two-dimensional region in (b, c) rather than a band 0.1 wide.
The wall-window error at a = 0 is set by row 1 alone, so every a = 0 member has the same wall
error and the discriminators are the interior error and the sweep. The unique seventh-order
row 3 on the line, (d, e) = (1/4, 1/2), is neutral at N = 51 and 101, cuts the interior error
by 20%, and fails the sweep from N = 18. The sweep over 649 line lengths, reject above 1e-10:

```
  (a, b, c) + (d, e)                max Re λ     at N    verdict
  (0, 3/5, 3/10) + (1/3, 1/3)       +1.9e-12     1180    passes every N
  (0, 7/10, 1/25) + (1/3, 1/3)      +2.5e-12      740    passes every N
  (0, 3/5, 1/4) + (1/3, 1/3)        +9.8e-2      1040    fails from N = 104
  (0, 3/4, 1/4) + (1/5, 3/5)        +2.2e-1       580    fails from N = 97
  (0, 4/5, 1/10) + (1/5, 3/5)       +2.3e-1       780    fails from N = 118
  (0, 3/4, 1/10) + (1/4, 1/2)       +1.3e-1       478    fails from N = 130
  (0, 5/9, 1/10) + (2/5, 1/5)       +1.8e-1       436    fails from N = 142
  (0, 16/25, 9/50) + (1/3, 1/3)     +2.9e-2        92    fails at N = 92 only (to 400)
  (0, 33/50, 7/50) + (1/3, 1/3)     +2.8e-2       345    fails at N = 37 and 345 (to 400)
  (0, 3/5, 3/10) + (1/4, 1/2)       +6.3e-1       384    fails from N = 18 (to 400)
```

The two survivors keep the C6 interior row on row 3 and carry C6-neutral rows 1 and 2. A
prefilter of 750 rational members over twenty line lengths keeps 79, all with the same wall
error; the two of lowest interior error among them grow at 0.22 per unit time from N = 97 on
the full sweep, so twenty line lengths are a prefilter and not a verdict. The adopted set is
(0, 3/5, 3/10; 1/3, 1/3), preferred over the runner-up for its denominators, its interior
error (5.65e-6 against 5.90e-6 at N = 97) and its viscous Jacobian reading.

### The C10 family

The pentadiagonal interior also reaches ±3 and takes three rows, whose left-hand sides carry
the full band:

```
g_1 + a g_2 + a₂ g_3 = Σ_{k=1}^{4} w_k f_k                    third order
b g_1 + g_2 + c g_3 + c₂ g_4 = Σ_{k=1}^{5} w_k f_k             fourth order
d₋₂ g_1 + d g_2 + g_3 + e g_4 + e₂ g_5 = Σ_{k=1}^{7} w_k f_k   sixth order
```

Row 3 takes seven points because a row with M right-hand-side points and a fixed left-hand
side is exact through degree M − 1: the cascade's five-point row is sixth order only because
(1/3, 1/3) makes it centred, and at (1/3, 1/3) the seven-point row collapses to the cascade's
with w₆ = w₇ = 0 exactly. The banded assembly agrees with the production `BandPlan` to 5.6e-16
to 8.9e-16. With row 3 held at the C6 interior row, the scan over (b, c) ∈ [−1, 1.5]² at
N = 51, pruned over N = 31, 79, 101 and 151:

```
  a       neutral at N = 51   surviving five lengths   b range          c range
  0            137 of 1681            24               [0.56, 0.88]     [−0.13, 0.63]
  1/2           56 of 1681            18               [0.50, 0.63]     [0.13, 0.75]
  1             16 of 1681             2               [0.50, 0.50]     [0.25, 0.31]
  3/2            4 of 1681             1               [0.44, 0.44]     [0.25, 0.25]
  2              2 of 1681             0
  5/2 to 6     0 to 1 of 1681          0
```

a = 0 is forced as before, and no growth rate between 1e-12 and 1e-8 occurs, so the
threshold is not a knob. The sweep over 649 line lengths:

```
  (a, b, c [, d, e])                 max Re λ     at N    N above 1e-10
  (0, 3/5, 3/10)                     +1.7e-12     1090        0
  (0, 16/25, 9/50)                   +5.7e-2       516        6
  (0, 7/10, 1/25)                    +3.5e-2       127        2
  (0, 11/20, 1/2)                    +1.1e-1      1130       29
  (0, 3/5, 0)                        +6.1e-1      1100      285
  (0, 3/4, 3/4, 1/8, 1/4)            +8.2e-1      1120      475
  (0, 9/10, 3/5, 1/8, 1/4)           +7.8e-2      1160        8
  (0, 19/20, 2/5, 1/8, 1/4)          +6.9e-2       990        3
```

(0, 3/5, 3/10) with the cascade's row 3 is the only member that never exceeds 1e-10. Two
rational members of the C6 band that are neutral for C6 at every N fail for C10 at N = 315
and N = 127, so the resonance is specific to the interior. Widening row 3 buys accuracy and
no neutrality: the lowest interior error constants sit at (d, e) = (1/8, 1/4), where the
interior error at N = 97 falls to 3.5e-8 against 7.8e-6 for the selected rows and 2.3e-6 for
the cascade, and every one of the ten lowest fails a staged sweep, the best at N = 226 and
seven further node counts. A seeded random search over all nine coordinates (40,000 draws,
1956 neutral at N = 51, 121 passing every stage) found nothing below 1.2e-7. The adopted set
is the C6 `:neutral3` rows padded with zeros at ±2 over the unchanged row 3, so the minimum
extent and the halo reach do not move.

### Error constants and the decision

One derivative of exp(sin 3x) on the closed line, wall window of four nodes then the
interior, N = 49 / 97 / 193, with the closed line's condition number:

```
                          wall                              interior                      cond
C8 :neutral3     8.573e-4  1.011e-4  1.224e-5     4.878e-5  5.648e-6  6.751e-7     6.95
C8 :cascade3     6.309e-4  7.492e-5  9.107e-6     1.422e-5  1.676e-6  2.029e-7    16.25
C8 runner-up     8.573e-4  1.011e-4  1.224e-5     5.097e-5  5.903e-6  7.058e-7     6.94
C8 :brady_livescu 2.150e-6 3.277e-8  2.872e-10    1.307e-7  1.951e-9  1.697e-11 4348.9
C10 :neutral3    8.574e-4  1.011e-4  1.224e-5     6.710e-5  7.769e-6  9.287e-7    20.69
C10 :cascade3    6.328e-4  7.514e-5  9.134e-6     1.961e-5  2.311e-6  2.797e-7    24.59
C6 :neutral3     8.573e-4  1.011e-4  1.224e-5     3.387e-5  3.922e-6  4.689e-7     4.98
C6 :cascade3     6.287e-4  7.465e-5  9.075e-6     9.841e-6  1.161e-6  1.405e-7    15.97
```

The wall-window error is set by row 1 and is the same for the neutral rows on every interior,
1.35 times the cascade's at unchanged third order; the interior-window error is 3.4 times the
cascade's on every interior, at unchanged order. The C8 condition number falls from 16 to 7;
the C10 one is set by the pentadiagonal interior and reads 20.7 for every band member against
24.6 for the cascade.

On C8 and C10 the two neutral rows are followed by the C6 interior row the cascade already
used there, so `neutral_closures` builds the three-row set from the C6 table and that row;
`:cascade3` is kept for comparison. The closed C8 and C10 studies of `test/convergence.jl`
read the C6 study's errors to the printed digits. A patch or level interface keeps the
cascade rows for every neutral set. The certificate is the measured one: a Kreiss constant
below 2.5, an eigenvector condition number below 9 to N = 801, and a transient amplification
below 4 over twenty time units, with no trend in N. No N-independent structural certificate
has been found, and the line-length resonance of the neighbouring members is a collision of
the two interior wavenumber branches of a compact scheme, whose period the interior sets and
whose occurrence the closure's wall phases decide, so the selection sweeps over every line
length and not over a sample.

### The same model on a face-centred symmetry plane

`bench/closuresearch.jl` holds `folded_derivative_matrix` (the interior stencil on every row,
the taps outside the line folded back with the field's sign, the ghost coupling of the result
folded onto the diagonal with the opposite sign) and `folded_acoustic_operator`, L = [0
−D_odd; −D_even 0] on the 2N unknowns (p even, u odd) with no endpoint elimination;
`bench/closurecertify.jl` takes `wall=folded`. The folded matrix reproduces the package's
fold plan on unit vectors to 0.0 (C6, C8) and 7.8e-16 (C10), equals the periodic operator on
2N nodes restricted by parity to 1.3e-15, and differentiates cos(πx) and sin(πx) at 6.00
(C6), 7.97 (C8) and the round-off floor from N = 20 (C10).

The folded L is antisymmetric to round-off: ‖L + Lᵀ‖/‖L‖ is 3e-16 to 8e-16 and the commutator
‖LᵀL − LLᵀ‖/‖L‖² 6e-16 to 1e-15 at N = 25 to 801 for all three schemes, where the node-centred
operator reads 0.84 and 0.97. Every unknown has the same cell measure and the mirror holds no
node, so the restriction argument gives skew-adjointness in the Euclidean inner product
itself, not similarity to it. The consequences follow without a search:

```
                    spectral abscissa    Kreiss ratio α_ε/ε      max_t ‖exp(tL)‖, t ≤ 20
folded, N = 25      +3.2e-15             1.0000                  1
folded, N = 51                           1.0000–1.0001           1
folded, N = 101                          1.0000–1.0003           1
folded, N = 201                          1.0000–1.0012           1
folded, N = 801     +7.1e-13
node :neutral3      +3.6e-15 … +6.3e-13  1.25 / 1.97 / 1.26 / 1.93   2.68 / 3.78 / 2.61 / 3.68
```

The Kreiss ratios hold over ε = 1e-2 to 1e-6 for C6, C8 and C10 alike; at ε = 1e-8 both
operators rise, the classification tolerance sitting on top of ε. The folded semigroup is an
isometry: the subspace iteration and the dense 2-norm agree to 3.8e-13 at every sampled time
to t = 20, in the Euclidean and in the cell-measure norm, and the eigenvector condition
numbers of 1.1 to 7.1 are the general eigensolver's response to the doubly degenerate ± pairs
of a normal operator. The node-count scan over every N from 12 to 600 and every tenth to 1200
finds no unstable length: the largest real part is +1.137e-12 (C6), +1.592e-12 (C8) and
+1.577e-12 (C10). Wall times at `-t 16`: spectrum 19 s, pseudo 118 s, transient 106 s, the
scan 999 s for the three schemes.

## The fifth-order closure search

```text
julia --project=. -t 1 bench/closuresearch.jl mode=report
julia --project=. -t 1 bench/closurequalify.jl schemes=de parts=polynomial,jacobian,smooth,stress
julia --project=. -t 1 bench/closureenergy.jl
julia --project=. -t 1 bench/closuredamping.jl parts=jacobian,uniform schemes=candidate strength=0.1 components=acoustic
```

A dead end kept for its instruments and its constraints. The search constructs fifth-order
boundary rows while retaining the C6 interior and its tridiagonal solve, and measures them
through the production derivative and timestep. Nothing established here is an energy bound
or an impossibility result. The archived vectors, seeds, populations, generation counts and
training grids are in `bench/closuresearch_results.md`.

**The family.** Each row reads the first six field values. With a unit diagonal on the
implicit side, row 1 has one free superdiagonal and rows 2–4 have two free off-diagonals
each; the seven parameters determine the explicit weights by exactness on monomials of
degrees zero through five, in rational arithmetic. Brady–Livescu T6 belongs to this family,
with additional conservation constraints the search does not impose. A seven-point extension
adds a multiple of `[1,-6,15,-20,15,-6,1]`, which annihilates degrees zero through five, for
eleven free parameters. The acoustic screening operator evolves pressure on all N points and
normal velocity on the N−2 interior points; finite-grid spectral neutrality is not an energy
estimate uniform in N, and eigenvector conditioning and sampled resolvents are diagnostics,
not proofs.

**Four candidate sets and what killed each.**

- `ClosureSearch.unfiltered_scheme()`, six points, selected on unfiltered acoustic growth and
  conditioning (implicit condition number 141 against Brady–Livescu's 1192, moment residual
  7.0e-17, production differentiation of x^6 at order 5.00000). Its filtered production
  Jacobian reads 1.006528 at N = 51 and 1.002726 at N = 101, growth rates 0.85 and 0.71 per
  unit time, persisting across the perturbation ladder, and although neutral at its training
  sizes it grows at 0.662 c/L at N = 171 in the sweep over every N from 12 through 200.
- `ClosureSearch.candidate_scheme()`, six points, selected against both the unfiltered
  growth and the filtered step. Its filtered radii improve to 1.0000148 and 1.0000059 but
  exceed the model gate of 1 + 1e-10, and the held-out unfiltered sweep finds growth 1.041
  c/L at N = 415.
- `ClosureSearch.de_scheme()`, seven points, differential evolution optimizing stability
  before conditioning. It retains fifth-order accuracy (moment defect 9.995e-17, x^6 orders
  4.99964 to 5.00000) and its slip and no-slip radii sit at the Jacobian's differencing
  floor, but its Dirichlet radius is 1.121 unfiltered and 1.0746 filtered at N = 79, 101 and
  171 alike, growth rates +14.67, +18.81 and +31.98 per unit time, which the one-sided filter
  does not remove. The `:neutral3` control at the same settings reads 1.000000001380
  filtered, so the growth belongs to these rows. The filter does remove its one unfiltered
  slip resonance, at N = 171.
- The joint derivative/filter treatment of `bench/closuredamping.jl`: the filtered-objective
  candidate with an extra rank-one wall pass `Fwall = I − σ v vᵀ/(vᵀv)`, `v =
  [1,-6,15,-20,15,-6,1]`, σ = 0.1, which preserves degree-five polynomials and is an O(h^5)
  boundary perturbation at fixed hyperbolic CFL. Applied to every conserved component it is
  rejected: the acoustic block alone appears neutral but the full production Jacobian reads
  1.00087 at N = 51 and 101, the missing modes the scalar entropy and tangential branches
  governed by the composite scalar filter. Applied only to pressure and normal velocity, with
  total energy adjusted for both the pressure and kinetic-energy changes, it passes the full
  Jacobian at 1 + 3.4e-9, holds a uniform state to t = 40 below 2.5e-13, and reads
  smooth-wall orders 5.94 / 6.06 with the artificial properties off. Over every N from 14
  through 200 plus five larger probes, at three CFL numbers, the only failures are N = 14 and
  N = 15. The prototype is a serial one-dimensional calorically perfect gas.

**An exact energy-norm feasibility probe.** `bench/closureenergy.jl` tests a restricted
construction: symmetric implicit A, `B + B' = (5/3) diag(-1,0,...,1)`, fifth-order boundary
moments, and the unchanged C6 interior. Positive A would give an SBP norm `H = (3/5)A`, and
decoupling its endpoint would make strong injection orthogonal in that norm. Exact rational
elimination finds inconsistent moment systems for tridiagonal boundary blocks with 4–12
closure rows, and full boundary blocks also fail in the tested widths; each rejection carries
a checked exact left-null certificate `y'M = 0, y'b = 1`. The standard explicit sixth-order
control admits the expected fifth-order restricted-full-norm moment families at 7 and 8 rows,
verifying the machinery. This rules out the tested simple norm ansatz, not other compact SBP
norms.

**Brady–Livescu also has a line-length resonance.** The published T6 control is not
unconditionally neutral without filtering: at N = 171 the production Jacobian reads 1.000751
unfiltered, growth 0.334 per unit time, and 1.000239 under the unrelaxed default filter,
growth 0.106, both persisting across the perturbation ladder. Its neutral readings at N = 51
and 101 do not generalize.

**The verdict.** A fifth-order closure is useful only where the artificial properties are
off or the dilatation sensor is on. Under the strain sensor the sensor's own cusp caps an
inviscid wall at fourth order under any closure ([wall closures in
production](#wall-closures-in-production)), and the production workloads of this solver are
shocked and cold-started. With the sensor mirrors and the slip-wall flux contract in place,
Brady–Livescu and the unfiltered-search rows complete cold planar Noh at N = 200 and CFL 0.3
with seven inadmissible cells, while the filtered-objective candidate fails at step 30, the
seven-point DE rows at step 28 and the joint treatment at step 31, under both sensors. All
five derivative choices complete Woodward–Colella and the t0 = 0.1 warm start. Completion
under a permissive policy is not an admissibility result, and one extent, CFL and end time do
not establish a cold-start envelope.

No fifth-order route is a production candidate, so the thread is closed. The independent
limitations remain: Brady–Livescu has a filtered slip-wall resonance, the unfiltered-search
set has the N = 171 resonance, the filtered-objective candidate has a held-out line-length
failure, and DE has a filtered Dirichlet radius of 1.0746. A resumption would measure
closures under `beta_sensor = :dilatation`, where the wall is no longer capped, rather than
begin at the derivative rows. The instruments are kept: `bench/closuresearch.jl` holds the
family, the reduced models and the searches, `bench/closurequalify.jl` the production
polynomial, Jacobian, evolution and shock checks, `bench/closureenergy.jl` the exact
energy-norm probe, and `bench/closuredamping.jl` the joint trial. Search assembly agrees
with production basis-vector applications to 1.066e-14 for the derivative and exactly for
the filter.

## The sensor operators at walls

```text
julia --project=. -t 1 bench/sensorwall.jl
```

Three operators around the artificial-property sensors carried closure rows that do not
reproduce a reflecting wall: `delta4_sum!`, which read its taps past a closed edge by
clamping the index; the `:gaussian` smoother, whose rows fold onto the half-offset mirror
and so sit half a cell out at a node-centred wall; and `ring_sum!`, the `:d8` detector. All
three now use node-centred rows at a wall and the half-offset mirror at a fold; the accuracy
gained is in [wall closures in production](#wall-closures-in-production).

### The extensions

`delta4_sum!` reads its taps at a reflecting wall from the node-centred mirror of the
interior, ghost `2-q` low and `2n-q` high, with the field's sign: +1 for every scalar sensed
and for the tangential velocities, −1 for the wall-normal component, through
`velocity_mu!`'s `wall_parity` keyword. `sensor_mirror(bc)` names the reflecting faces,
`true` for `SlipWallBC` and `NoSlipWallBC` and `false` by default, so Dirichlet,
extrapolation and NSCBC faces and interface ends keep the clamp; the `:delta4` path queries
it per call, while the setup-time `planned_sensor_mirror(bc)` gives a `SwitchableBC` face
the wall rows only when both of its conditions are mirrors.

`wall_closures(scheme, σ)`, in `kernels.jl` and `kernels_banded.jl`, folds a symmetric
scheme's interior stencil, taps and left-hand-side unknowns alike, onto the node-centred
mirror, q < 1 onto 2 − q, with the sign σ, inheriting a filter's unit row sum and the eighth
derivative's zero row sum from the interior weights. The `:gaussian` smoother takes the
σ = +1 rows at every face `sensor_mirror` names, its input a detector output past an
absolute value and so even; the `:d8` detector takes both signs, a pair per dimension
indexed by the wall sign that aliases to one plan where neither face is a wall, leaving the
default configuration's plans and memory unchanged. A fold's closed far end takes the same
rows, and `FoldSpec.ring_plans` became 2×2, ghost parity by wall sign. Patched runs apply
the smoother rows at physical wall faces only. The `:compact` smoother shares the state
filter's plans, carries no half-cell shift and is unchanged.

At a coordinate fold `delta4_sum!` takes the half-offset mirror with the sign `parity[d]`:
at a self-paired fold the mirror is the signed line itself; at a paired fold every field
goes through the even/odd butterfly of `folds.jl`, unchanged, since the per-half mirror
signs derive from e(−r, θ) = ½[σ f(Mx) + σ² f(x)] and are independent of σ. The paired path
keeps the wall mirror too, the pairing map acting on the angular coordinates alone and
commuting with the reflection about the wall node. The clamp remains at closed edges that
are neither wall nor fold; on a half-offset grid mirror and clamp differ on one tap, so the
change reaches one cell per folded end.

### What the clamp cost

At a fold, for an even field the clamp misplaces one δ⁴ tap by a term that the vanishing
edge derivative makes O(h²); for an odd field the edge derivative is the largest quantity
there and the same tap is wrong at O(h). On u_r = r at the cylindrical axis, N = 32, which
should produce no sensor at all, the clamp gives μ\* = 1.2e-6 on the axis cell against 0 for
the mirror, or C_mu·ρ·h² of spurious viscosity on the cell where every converging case
fails.

A scratch probe applies `detect_sum!` under `:delta4` at weight power 2 to ρ = 1 + r² and to
exp(−4r²), comparing the first interior cell against the analytic δ⁴ of the smooth even
extension through r < 0. The self-paired axisymmetric cylinder and the paired resolved-θ
cylinder and spherical origin give identical numbers:

```
N    field        cell 1 clamp   cell 1 mirror   exact
32   1 + r²       2.031364e-06   2.237789e-19    2.237789e-19
32   exp(−4r²)    8.278876e-06   1.942393e-07    1.942393e-07
64   1 + r²       1.230085e-07   5.506717e-20    5.506717e-20
64   exp(−4r²)    4.943444e-07   2.920133e-09    2.920133e-09
```

On exp(−4r²) the clamp reads 42.62 times the exact value at cell 1 for N = 32 and 169.29
times it for N = 64, growing as h⁻² as an O(h²) relative error predicts; the mirror reads
1.0000 to every printed digit, and cells 2, 3, 4 and 8 read 1.0000 under both extensions.

### The operator probe

`bench/sensorwall.jl` prints |wall − periodic| at a node over the input amplitude, on an
even field cos(πx) + 0.5cos(5πx) + 0.1cos(13πx) and an odd one sin(πx) + 0.1sin(13πx), both
of period 2 and reflecting about the nodes at x = 0 and x = 1. Nodes 1, 2 and 6 of the low
wall, N = 49 / 97 / 193, under the half-offset rows these replaced:

```
operator / field        node   N=49        97          193         orders
:d8, even                  1   4.637e-03   1.075e-03   2.629e-04   2.11 / 2.03
                           2   1.014e-02   2.334e-03   5.725e-04
                           6   1.290e-03   3.221e-04   7.967e-05
:d8, odd                   1   1.535e-02   7.152e-03   3.527e-03   1.10 / 1.02
                           2   3.260e-02   1.553e-02   7.680e-03
                           6   4.180e-03   2.147e-03   1.069e-03
:gaussian, even            1   1.871e-02   5.225e-03   1.345e-03   1.84 / 1.96
                           2   4.315e-03   1.158e-03   2.949e-04
                           3   5.554e-04   1.460e-04   3.696e-05
                           4   4.047e-05   1.040e-05   2.619e-06
:compact, even             1   3.310e-06   1.358e-08   5.375e-11   7.93 / 7.98
```

The high wall reads the low one in every row. `:gaussian` is exactly 0 at nodes 5 and 6, its
rows reaching four nodes, and `:compact` reads between 1.3e-05 and 1.4e-10 at nodes 2 to 6;
`:delta4` sits at the 1e-15 floor at nodes 1 and 2 and is exactly 0 at nodes 3 to 6, on both
parities and at both walls.

Under the node-centred rows, `:d8` reads at most 1.994e-15 on the even field and 2.545e-15
on the odd one at any of the six nodes of either wall at any resolution, `:gaussian` at most
2.776e-16 at nodes 1 to 4 and exactly 0 at nodes 5 and 6, and `:delta4` and `:compact` are
unchanged in every digit. A one-dimensional cross-check at N = 49 reads the same way, the
smoother reproducing the periodic run to between 1.5e-16 and 2.9e-16 relative and
`ring_along!` to between 1e-12 and 3e-11, the high-pass's own cancellation floor. At a
symmetry plane the same probe reads round-off for every operator on both parities with no
wall rows at all ([the face-centred symmetry plane](#the-face-centred-symmetry-plane)).

### What moved

Every smooth-wall row with the artificial properties off is bit-identical, as is
`test/convergence.jl` study by study (every study there runs with the properties off). With
them on, the wall errors of the Brady–Livescu closures fall by one to two orders at the
viscous and shear walls, to within about 1% of their properties-off values, and the
`detector = :d8` channel row falls by factors 2.7 to 5.3. The inviscid slip wall under the
strain sensor is then limited by that sensor's cusp rather than by any operator's edge, and
the two default closures move in the third digit.

On the battery the mirror moved the planar Noh wall deficit and the aligned Noh case and
nothing else past the printed digits. The aligned case's transverse round-off read 1.4e-8
under the clamp, 2.8e-6 under the `:delta4` wall mirror alone, and 7.8e-9 under the full set
([the aligned Noh transverse mode](#the-aligned-noh-transverse-mode)). The fold rows moved
in the fourth or fifth digit and no guard failed. Spherical Noh completed 544 steps at CFL
0.30 before the fold change and 543 after, and failed on negative density at 0.40 at step
105 before and 106 after.

A scratch MPI check on the cylindrical axis with θ split over two ranks gives np = 1 and np
= 2 agreeing to 1.2e-15 relative on Σμ\*, 6e-16 on Σβ\* and 5e-16 on Σκ\*, with no deadlock;
between clamp and mirror Σμ\* and Σβ\* move in the eleventh digit while Σκ\* falls by a
factor 5.9, so the internal-energy sensor at the axis carried most of the clamp's spurious
contribution. No MPI phase runs the detector across a paired fold, and the butterfly
exchange every scalar sensor now carries there is untimed.

`_device_plan` copies the closure rows generically. On an AMD RX 6800 XT (gfx1030) through
AMDGPU.jl, on a (48, 16, 16) slab with slip walls on dimension 1, `ring_plans[1]` holds two
distinct plans and `ring_plans[2]` one aliased plan as on the host, and `mu_art`,
`beta_art`, `kappa_art` and `dQ` after one `compute_rhs!` agree bitwise with the
`CPUBackend` solver from the same standing-wave data, over the interior and over the six
nodes nearest each wall, as does the state after five steps; the comparison repeats under
`mu_sensor = :velocity`, which exercises the odd wall sign, and under the `detector =
:delta4` control.

In the default configuration the extra rows cost nothing, the pair aliasing one plan. Under
`:d8` with a wall they add one plan per walled dimension: at (96, 64, 64) the solver holds
four distinct ring plans against three and 12.00 MiB of packed-line buffer against 9.00 MiB,
and `Base.summarysize` reads 289.01 MiB against 285.50 MiB. Construction does not move, and
per call the walled configuration is 4 to 8% cheaper in both runs, inside the run-to-run
spread.

`bench/jetcheck.jl` reports one dispatch site for the hook, `sensor_mirror` through
`_face_mirror`, so `compute_rhs!` reads 3 and `step!` 4 with every other entry point 0. Two
rejected spellings measured five sites (a bare call whose `Any` return destroyed the
pointwise body's specialization) and one extra `convert` site (a `::Bool` annotation); the
adopted `@noinline` `@nospecialize` form compared with `=== true` holds it to one.
`bench/audit.jl` reads +1536 B per call at 48³ in `compute_artificial!`, `compute_rhs!` and
`step!`, constant rather than per point (three extra scalars captured per threaded region;
0 B at `-t 1`), with every inference row unchanged.

Two test guards moved with the `:d8` wall rows. The `:d8 through a coordinate-singularity
fold` window narrowed to the inner half, because exp(−4r²) has slope −0.147 at the outer
slip wall and is not the reflection the wall rows continue it as, which reads 3.6e-3 at the
wall node. The `u_r = r` axis check under `:d8` moved from 1e-14 to 1e-12 because that field
is not odd about the outer wall, the slip condition leaves a kink there, and the
pentadiagonal inverse carries a decaying tail of that mismatch to the axis.

## The face-centred symmetry plane

```text
julia --project=. -t 1 bench/wallclosure.jl parts=smooth wall=folded
julia --project=. -t 1 bench/sensorwall.jl wall=folded
julia --project=. -t 1 bench/closurequalify.jl parts=jacobian schemes=neutral3 jns=51,101 jwalls=symmetry,slip
mpiexec -n 4 julia --project=. -t 1 test/mpi_tests.jl "phases=symmetry plane"
```

An inviscid slip wall is a symmetry plane: density, pressure, energy, species and the
tangential velocity are even about it and the normal velocity odd. `SymmetryPlaneBC` places
the plane half a cell outside the end node, on the half-offset grid the coordinate folds
use, every operator running its interior stencil over the mirrored halo with no closure row.
It is the self-paired fold of the axisymmetric axis with the slip wall's parities, on any
dimension whose scale factors do not depend on the folded coordinate; `DESIGN.md` has the
mechanism and the setup rules.

### Mirror equivalence

The folded operator is the periodic operator on the doubled line restricted by parity, so a
run between symmetry planes on [0, 1] at N nodes and the periodic run on [0, 2) at 2N nodes
with origin h/2 differ by round-off for every operator in the step. The `symmetry plane: the
mirror of the periodic run on the doubled line` testset measures that at N = 32 over thirty
to forty steps, compact filter every step at full strength and artificial properties on,
relative maximum norm over every conserved component on the coincident nodes:

```text
                  :delta4/:gaussian  :delta4/:compact  :d8/:gaussian  :d8/:compact
lele_d1_6              2.52e-15          3.36e-15        3.36e-15       3.53e-15
lele_d1_8              2.35e-15          3.02e-15        3.19e-15       3.53e-15
lele_d1_10             3.19e-15          3.36e-15        3.02e-15       3.53e-15
viscous, mu0 = 0.005   3.15e-15
2-D, plane on dim 2, tangential velocity   5.54e-15    viscous 4.33e-15
```

Two species with mass fractions even about both planes pass the same guard, set at 2e-14. A
uniform multispecies state between six planes gives a `compute_rhs!` output of exactly zero,
and five steps leave a spread and a drift of 1.9e-14.

### One derivative, one filter pass, and the sensors

`test/convergence.jl`, global maximum norm of one derivative between two planes on [0, 1],
the field even (exp(cos πx)) or odd (sin(πx) exp(cos πx)) about both planes:

```text
                                 Ns            errors                              order
C6 symmetry planes, even       24/48/96    1.903e-6  2.980e-8  4.656e-10          6.00
C6 symmetry planes, odd        24/48/96    7.732e-6  1.211e-7  1.893e-9           6.00
C8 symmetry planes, even       16/24/32    1.562e-6  5.974e-8  5.914e-9           8.05
C8 symmetry planes, odd        16/24/32    6.724e-6  2.686e-7  2.711e-8           7.95
C10 symmetry planes, even      12/16/24    7.977e-7  4.414e-8  6.667e-10         10.23
C10 symmetry planes, odd       12/16/24    3.919e-6  2.066e-7  3.400e-9          10.17
C8 filter pass, planes         16/24/32/48 4.051e-7  1.711e-8  1.762e-9  7.016e-11  7.88
```

The resolutions fall with the order because a field this smooth reaches round-off quickly.
`bench/foldorder.jl` with a plane at the low end and a node-centred `SlipWallBC` at the high
end splits the norm: the fold window converges at 7.01 (even) and 6.05 (odd) with the
maximum at the last node, where the wall window reads 2.99 and 3.76, and both rows are
bitwise identical to the cylindrical-axis rows of the same file.

`bench/sensorwall.jl wall=folded` runs the `:delta4` and `:d8` detectors on an even and an
odd field and the `:gaussian` and `:compact` smoothers on the even field, at a plane against
the periodic mirror, N = 49 / 97 / 193, both faces. The maximum relative departure over
every combination and node is 9.1e-15, the typical entry 1e-16 to 3e-15, many entries
exactly zero, against the 2.8e-16 (smoother) and 2.5e-15 (detector) of the node-centred wall
rows of [the sensor operators at walls](#the-sensor-operators-at-walls).

### The smooth wall matrix

`bench/wallclosure.jl parts=smooth wall=folded`, the wall-window maximum norm at t = 0.4
against the mirror at the same spacing, the default filter every step, cfl 0.25. The
node-centred rows carry a closure defect; the folded rows are round-off:

```text
                                        node-centred                          folded
inviscid slip, C6, art off     5.783e-7  3.984e-8  2.588e-9 (3.86/3.94)   4.4e-15  4.2e-15  7.1e-15
inviscid slip, C6, art on      5.666e-7  3.939e-8  2.582e-9 (3.85/3.93)   3.7e-15  3.8e-15  1.0e-14
inviscid slip, C8, art on      5.669e-7  3.936e-8  2.585e-9 (3.85/3.93)   4.7e-15  6.7e-15  7.3e-15
inviscid slip, C10, art on     5.661e-7  3.930e-8  2.584e-9 (3.85/3.93)   5.1e-15  8.9e-15  5.1e-15
viscous slip + shear, C6, on   4.165e-7  3.003e-8  2.010e-9 (3.79/3.90)   3.3e-15  3.2e-15  7.3e-15
viscous slip + shear, C8, on   4.152e-7  3.007e-8  2.016e-9 (3.79/3.90)   4.8e-15  4.4e-15  3.9e-15
viscous slip + shear, C10, on  4.120e-7  2.998e-8  2.014e-9 (3.78/3.90)   3.6e-15  1.3e-15  2.0e-15
```

The largest folded entry over the thirty-six rows is 1.5e-14. A mirror at the same spacing
therefore measures nothing about a plane, and the evolution rows of `test/convergence.jl`
take a five-times-finer folded mirror as their reference, an odd refinement so every study
node is a reference node, carrying 1/625 of the study's step error. The wall window then
reads the run's own total error and equals the interior column in every row:

```text
                                        Ns          errors                        order
inviscid planes, C6, unfiltered        25/49/97   1.719e-9   7.399e-11  4.084e-12  4.46
inviscid planes, C6, one-sided filter  25/49/97   2.353e-9   7.970e-11  4.092e-12  4.69
inviscid planes, C8, unfiltered        25/49/97   8.689e-10  5.900e-11  3.838e-12  4.00
inviscid planes, C10, unfiltered       25/49/97   8.693e-10  5.901e-11  3.844e-12  4.00
inviscid planes, C6, cfl/4             25/49/97   8.587e-10  1.526e-11  2.782e-13  5.93
viscous slip planes, C6, unfiltered    25/49/97   8.751e-10  1.528e-11  2.445e-13  6.04
```

The 4.00 of the C8 and C10 rows is the time integrator. Halving the step at N = 49 takes
the C6 error 7.40e-11 to 1.87e-11 and quartering it to 1.53e-11: a fourth-order time error
over a fixed spatial floor of 1.5e-11, and the cfl/4 row is that floor's sixth-order
sequence. The C6 rows read above 4 because that floor is still visible at N = 25, where
they sit twice as high as C8 and C10; the gap is 25% at N = 49 and 6% at N = 97. The
viscous row's step is diffusion-limited and shows the spatial order directly.

The rows first ran on 49/97/193. The finest grid's difference was then 2.5e-13, and at that
level the step's accumulated round-off differs between hosts: the same code gave the C10
row 3.844e-12 / 2.580e-13 and an order of 3.96 on the workstation and 3.838e-12 / 2.456e-13
and 4.00 on the GitHub runner, with the N = 49 column identical to four digits. Julia
1.11.4 and 1.13.0 on the workstation agree to every printed digit, so the spread is the
hardware's arithmetic and not the compiler's. At the coarser ladder's finest grid the two
hosts differ by 0.16% and the fitted order by 0.001.

### The production Jacobian

`bench/closurequalify.jl parts=jacobian jwalls=symmetry,slip`, the finite-differenced
`step!` about a uniform state, radius and rate per unit time at the three differencing steps
δ = 3e-6 / 1e-5 / 3e-5:

```text
 N   wall      filter   radius − 1                    rate
 51  symmetry  off      2.5e-10  7.6e-11  1.9e-11     +3.2e-8  +1.0e-8  +2.5e-9
 51  symmetry  on       1.3e-10  9.0e-11  5.0e-12     +1.6e-8  +1.2e-8  +6.5e-10
 51  slip      on       3.9e-9   1.1e-9   3.2e-10     +5.1e-7  +1.5e-7  +4.2e-8
101  symmetry  on       1.3e-10  1.7e-10  3.2e-11     +3.4e-8  +4.4e-8  +8.4e-9
101  slip      on       3.8e-9   9.7e-10  2.1e-10     +9.9e-7  +2.5e-7  +5.6e-8
```

The plane's radius is one to the finite-difference noise, and its filtered rate is six to
thirty times below the node-centred slip wall's, the one configuration in which the neutral
rows show a measurable positive rate. C8 and C10 measured through `production_jacobian`
directly at δ = 1e-5 read radius − 1 of 7.6e-11 / 1.0e-11 (C8, N = 51, unfiltered /
filtered), 1.2e-10 / 1.7e-10 (N = 101), 8.1e-11 / 3.9e-11 and 1.1e-10 / 1.6e-11 (C10), where
their filtered slip-wall counterparts read 2.7e-9 at both N. The exact linear model of the
folded operator, antisymmetric with an isometric semigroup, is in [closure
certificates](#closure-certificates).

### MPI and device parity

The `symmetry plane` phase builds every case twice, on the process grid and whole on
`MPI.COMM_SELF`, comparing the rank's interior block against the serial rebuild for the
`compute_rhs!` output and the state after filtered steps: planes at both ends of the split
dimension on each axis in Float64 and Float32 with two species, physical viscosity,
`detector = :d8` on one axis and `species_flux = :bulk` on another; a corner with planes at
both ends of two split dimensions; and the Noh layout, `DirichletBC` low and the plane high.
All 22 checks pass at np = 2 and np = 4, Float64 residuals 7.1e-15 to 2.3e-14 against a
tolerance of 1e-8 and Float32 ones 2.4e-7 to 6.7e-6 against 5e-4, with two ranks at np = 4
owning neither plane and still reaching the folded solve. Device plans inherit the fold from
the wrapped host plan's factorization, and the two device runs of `test/device_tests.jl` are
bitwise against the CPU solver.

### The battery at a plane

`test/cases.jl` gives `woodward`, `noh_case` (ν = 1) and `noh_aligned` a `folded` keyword
that places `SymmetryPlaneBC()` where the `SlipWallBC` was, the spacing following the grid
(h = L/N between two planes, L/(N − ½) from a plane to the Dirichlet inflow) and feeding the
collapsed extents and the blend width, so the two runs differ in the wall placement alone.
`test/validation.jl` runs each folded case beside its original under the same bounds. The
wall deficit is sampled at node 1 on both grids, the wall itself node-centred and half a
cell inside the plane folded; `noh_plane_density`, the even continuation (9ρ₁ − ρ₂)/8 onto
the plane, prints beside it. The readings are in [the shock battery](#the-shock-battery):
the plane improves Woodward–Colella by 6% in L1 with the contact where it was, reads the
planar Noh wall deficit one point higher at six times less negative a wall-layer internal
energy, cuts the aligned Noh deficit by five points, and holds the aligned case's transverse
round-off four hundred times smaller, so that row's uniformity guard is 5e-9 where the
node-centred row's is 5e-7. Every pre-existing row reads its recorded value to four digits.
The planar deficit repeats the earlier finding: the wall heating of Noh is a property of the
captured shock's start at the wall, not of the closure rows.

## The aligned Noh transverse mode

```text
julia --project=. -t 1 bench/noh_transverse.jl trace sample=100
julia --project=. -t 1 bench/noh_transverse.jl seeds seed_mode=2 sample=100
julia --project=. -t 1 bench/noh_transverse.jl uniform seed_mode=2 seed=1e-10 sample=50
julia --project=. -t 1 bench/noh_transverse.jl warm t0=0.1 seed_mode=2 seed=1e-10
julia --project=. -t 1 bench/noh_transverse.jl widths sample=100
julia --project=. -t 1 bench/noh_transverse.jl seed_channels seed_mode=2 seed=1e-10 sample=100
julia --project=. -t 1 bench/noh_transverse.jl extended tfinal=2.0 sample=200 nmax=20000
```

The aligned Noh case grows a transverse disturbance from round-off. The baseline is the
validation case: 12 periodic transverse nodes, 100 wall-normal nodes, spacing ratio 4, final
time 0.6, under the slip-wall flux contract, node-centred sensor rows and C6 `:neutral3`. A
post-step observer reads only the conserved state during one continuous `run!` and subtracts
transverse station 1 before projecting a line, so a transversely constant state has exactly
zero modal content. `uniformity` is the largest absolute density difference from the first
station; the largest transverse spread and the Fourier amplitude at each wall-normal node
are different norms of the same state. The baseline completes in 4,966 steps with
`uniformity = 2.052e-7`, spread 3.07e-7, and dominant density amplitude 1.328e-7 in mode
m = 2, a six-cell transverse wavelength, near the moving shock.

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
0.29273) = 131.6`, but at step 3,000 the maximum sits 7.3 wall-normal cells from the
analytic shock and outside the four-cell shock window, so the rate is not the growth of an
amplitude continuously localized at the shock. It is a finite-window rate of a
path-dependent transient: the amplitude falls again after the burst, and controlled
perturbations follow different paths. The final spectrum, m1 3.45e-8, m2 1.33e-7, m3 2.21e-9
and m4 6.25e-11, is dominated by a low transverse mode rather than a single alternating grid
mode.

**The same wall without a shock does not amplify it.** The control uses the identical strip,
filter, artificial properties and slip wall, filled with the exact planar-Noh post-shock
state (rho = 4, u = 0, p = 4/3) and holding it at the far Dirichlet end. Unseeded it
finishes at station 1.91e-13 and m = 2 amplitude 7.96e-14; with an m = 2 relative density
seed of 1e-10, which begins at 4e-10 since it multiplies rho = 4, it finishes at 3.446e-10,
a gain of 0.861 and a finite-horizon rate of −0.25, with a transverse-velocity amplitude of
4.36e-15. The two-dimensional `:neutral3` wall therefore holds round-off and slightly damps
the controlled perturbation over the validation horizon, agreeing with the uniform-state
Jacobians of [constant annihilation](#constant-annihilation), and certifies nothing about
every perturbation of either state.

**Controlled amplitude.** The seed multiplies the initial density by `1 + a cos(2 pi m
x/Lx)` at fixed pressure and velocity. For m = 2:

| initial A | final m = 2 A | gain | `log(gain)/0.6` | fitted rate over the second half |
|---:|---:|---:|---:|---:|
| 1e-12 | 6.868e-11 | 68.7 | 7.05 | -1.22 |
| 1e-10 | 6.594e-9 | 65.9 | 6.98 | -0.10 |
| 1e-8 | 1.443e-5 | 1,443 | 12.12 | -- |

The two small seeds take finite gains differing by about 4% with no positive late-time
fitted rate; the 1e-8 seed takes a much larger gain and generates an m = 4 harmonic of
7.96e-7. Neither is the constant exponential growth of an unstable wall eigenmode, and
extrapolating the small-seed gains does not predict the unseeded baseline: inserting even
the 1e-12 seed changes the round-off history and suppresses the baseline's large burst, so
endpoint changes under an unseeded ablation are not channel causality. From the smoothed
exact profile at t = 0.1, evolved for 0.5, the unseeded station variation is 1.389e-10
against 2.052e-7 from the cold singular start, while the seeded m = 2 run finishes at
3.129e-8, a gain of about 313, with its maximum at the shock: the singular startup sets the
natural burst history, and a resolved shock can still amplify an imposed disturbance.

Changing the transverse point count at fixed spacing changes both the available modes and
the round-off trajectory, with neither a monotone width law nor one selected wavelength: nx
= 10 ends at 3.132e-9 (m = 2 at the shock), nx = 12 at 2.052e-7 (m = 2), nx = 16 at 6.932e-7
(m = 2) and nx = 24 at 3.457e-10 (m = 5, in the bulk). The nx = 16 row exceeds the nx = 12
validation guard.

Artificial-property comparisons use the same m = 2, 1e-10 seed so that a different initial
round-off realization is not the comparison:

| active artificial channels | final m = 2 A | gain | `log(gain)/0.6` | location |
|---|---:|---:|---:|---|
| defaults | 6.594e-9 | 65.9 | 6.98 | bulk |
| beta only | 2.221e-8 | 222 | 9.00 | wall |
| beta + mu | 1.810e-8 | 181 | 8.66 | wall |
| beta + kappa | 2.555e-9 | 25.6 | 5.40 | shock |
| defaults with `C_D = 0` | 6.594e-9 | 65.9 | 6.98 | bulk |

`C_D = 0` is bit-identical to the default in this single-species case. Adding conductivity
to beta reduces the matched seeded gain and moves the maximum from the wall to the shock,
establishing damping and relocation for the controlled small mode without identifying one
nonlinear feedback behind the natural burst. The unseeded endpoint variations under the same
ablations (9.78e-11, 1.57e-10, 4.47e-10) are associations with different numerical
trajectories. With the filter off the cold run fails with negative density at step 1,335 and
with all artificial properties off or beta off near t = 0.074, so those runs cannot decide
whether the removed channel generates or damps the completed run's mode.

**Later evolution.** The baseline initial data evolved in one continuous run to t = 2.0,
with no landing imposed at t = 0.6, completes in 16,219 steps with the shock inside the
domain and undergoes a second burst before settling into a bounded oscillatory range: the
station variation reads 2.13e-7 at t = 0.604, 1.71e-6 at 1.102, 3.20e-4 at 1.351, 4.24e-4 at
1.475 and 2.87e-4 at 2.0, the dominant mode again m = 2 with its maximum in the shock
window. The order 3e-4 level over t = 1.35 to 2.0 is about three orders above the validation
endpoint, so the first t = 0.6 plateau is not saturation.

The evidence supports a path-sensitive transverse interaction with the captured Noh shock
and the history produced where that shock leaves the wall: the natural disturbance moves
from the wall into the bulk and later the shock window, disappears to round-off in the
identical no-shock strip, depends strongly on cold versus warm startup and on transverse
extent, has approximately linear finite gain for small imposed modes, and develops harmonics
and a higher bounded amplitude when driven farther. No unique nonlinear feedback among the
filter, shock capture and artificial properties is isolated by these measurements.

The validation guard is `uniformity < 5e-7` for exactly N = 100, AR = 4, nx = 12 and t =
0.6, a factor 2.44 above the current measurement: a deterministic regression envelope for
that preset and horizon, not a stability bound for another width, an injected disturbance or
later evolution. At a face-centred symmetry plane the same case reads 5.135e-10 and its
guard is 5e-9 ([the face-centred symmetry plane](#the-face-centred-symmetry-plane)).

## The inflow transverse terms

```text
julia --project=. -t 1 bench/nscbcinflow.jl pulse vortex outflow betas=0,0.15,0.3,0.5,0.7,0.85,1 etas=4,16,64
julia --project=. -t 1 bench/nscbcinflow.jl pulse vortex outflow M=0.1 betas=0,-1,0.9,1 etas=16
julia --project=. -t 1 bench/nscbcinflow.jl pulse vortex outflow M=0.5 betas=0,-1,0.5,1 etas=16
julia --project=. -t 1 bench/nscbcinflow.jl vortex betas=0,0.5,1 etas=256,1024
```

Every imposed amplitude of `NSCBCInflowBC` carries `−beta_t·𝒯`, the
weighted transverse contribution of its characteristic (the derivation is in
`src/nscbc.jl`); `NSCBCOutflowBC` has carried the same weight on its one
incoming wave since its introduction. The instrument measures both faces on a
two-dimensional stream at Mach 0.3 (γ = 1.4, p = ρ = 1, 64 points per unit
length, periodic width 1, artificial properties off, the default filter every
step) against the same run on a domain extended two units past the face under
test, where the disturbance meets no boundary in the run: the difference is
the face's reflection or the error of its imposition. A pressure pulse of
amplitude 1e-3 and radius 0.1 released half a unit inside the face meets it
at every incidence angle; the deviation is scaled by the largest excursion the
extended run carries across the plane of the face (2.1e-4 at the inflow,
2.6e-4 at the outflow). An isentropic vortex of peak velocity 0.3 of the
stream and radius 0.1 enters through the inflow from a time-dependent target,
the analytic vortex at the plane, or leaves through the outflow; the
deviation is scaled by its peak velocity and by its pressure depression
(1.5e-2). The vortex rows run to the time the centre reaches one unit inside
the face; the pulse rows to the time the reflection is half a unit back
inside.

### The pulse at the inflow

Maximum and root-mean-square deviation of p over the region the reflection
has crossed, over the incident amplitude, at the default relaxation rates:

| `beta_t` | max | rms |
|---|---|---|
| 0 (LODI) | 0.220 | 0.045 |
| 0.15 | 0.126 | 0.025 |
| 0.3 (= M) | 0.047 | 0.0093 |
| 0.5 | 0.048 | 0.0096 |
| 0.7 | 0.100 | 0.017 |
| 0.85 | 0.119 | 0.020 |
| 1 | 0.128 | 0.022 |

The reflection is least at a weight near the Mach number, a factor 4.7 under
the LODI form, and the full share halves the LODI reflection.

### The vortex at the inflow

Maximum and root-mean-square deviation of u over the peak velocity, the
maximum of v likewise, and the maximum of p over the pressure depression,
once the centre is one unit inside:

| `eta` | `beta_t` | max u | rms u | max v | max p |
|---|---|---|---|---|---|
| 4 | 0 | 0.93 | 0.168 | 1.21 | 0.97 |
| 4 | 0.5 | 0.84 | 0.157 | 1.26 | 0.95 |
| 4 | 1 | 0.63 | 0.107 | 1.10 | 0.83 |
| 16 | 0 | 0.52 | 0.107 | 1.22 | 0.67 |
| 16 | 0.5 | 0.39 | 0.079 | 0.99 | 0.54 |
| 16 | 1 | 0.26 | 0.043 | 0.58 | 0.36 |
| 64 | 0 | 0.22 | 0.036 | 0.44 | 0.40 |
| 64 | 0.3 | 0.17 | 0.028 | 0.35 | 0.31 |
| 64 | 0.5 | 0.14 | 0.023 | 0.30 | 0.25 |
| 64 | 0.7 | 0.10 | 0.018 | 0.24 | 0.18 |
| 64 | 1 | 0.063 | 0.011 | 0.16 | 0.091 |
| 256 | 0 | 0.061 | 0.0093 | 0.11 | 0.11 |
| 256 | 0.5 | 0.037 | 0.0058 | 0.074 | 0.065 |
| 256 | 1 | 0.016 | 0.0029 | 0.040 | 0.024 |
| 1024 | 0 | 0.016 | 0.0024 | 0.027 | 0.028 |
| 1024 | 0.5 | 0.011 | 0.0015 | 0.018 | 0.017 |
| 1024 | 1 | 0.0050 | 0.0008 | 0.010 | 0.0068 |

At the default rates (0.28) the vortex does not enter at all under any
weight: the error is the vortex itself. A relaxation admits a structure only
when its time L_ref/(η c) is short against the passage time r_v/u_0, 0.28
here against 6 at the default rate and 0.03 at η = 64, and the error then
falls as 1/η up to the largest rate run, with no stiffness at η = 1024. At
every rate the full share is the best weight, by a factor 3 to 4 over the
LODI form from η = 64 up and with the error falling monotonically in the
weight; the pressure error falls fastest, 0.40 to 0.091 at η = 64.

### Across the Mach number

The same two disturbances at Mach 0.1, 0.3 and 0.5 (the domain and the
disturbances unchanged, η = 16 for the vortex), the pulse's maximum
deviation over its incident amplitude and the vortex's maximum u over its
peak velocity and maximum p over its pressure depression:

| M | `beta_t` | pulse max | vortex max u | vortex max p |
|---|---|---|---|---|
| 0.1 | 0 | 0.158 | 0.55 | 0.94 |
| 0.1 | M | 0.093 | 0.52 | 0.86 |
| 0.1 | 1 | 0.114 | 0.082 | 0.115 |
| 0.3 | 0 | 0.220 | 0.52 | 0.67 |
| 0.3 | M | 0.047 | 0.45 | 0.60 |
| 0.3 | 1 | 0.128 | 0.26 | 0.36 |
| 0.5 | 0 | 0.376 | 0.66 | 0.92 |
| 0.5 | M | 0.025 | 0.57 | 0.78 |
| 0.5 | 1 | 0.112 | 0.44 | 0.61 |

The ordering is the same at every Mach number: the Mach-number weight
reflects the pulse least and the full share admits the vortex best, by a
factor 7 over the LODI form at Mach 0.1, where the Mach-number weight is
nearly the LODI form. The full share reflects the pulse less than the LODI
form at every Mach number.

### The same disturbances at the outflow

| `beta_t` | pulse max | pulse rms | vortex max u | vortex rms u | vortex max v | vortex max p |
|---|---|---|---|---|---|---|
| 0 (LODI) | 0.106 | 0.015 | 0.033 | 0.013 | 0.045 | 0.69 |
| 0.15 | 0.060 | 0.0079 | 0.021 | 0.0088 | 0.034 | 0.50 |
| 0.3 (= M, default) | 0.097 | 0.0067 | 0.015 | 0.0065 | 0.024 | 0.27 |
| 0.5 | 0.128 | 0.014 | 0.0093 | 0.0043 | 0.010 | 0.10 |
| 0.7 (= 1 − M) | 0.194 | 0.022 | 0.0061 | 0.0026 | 0.0012 | 0.050 |
| 0.85 | 0.242 | 0.028 | 0.057 | 0.0071 | 0.067 | 0.098 |
| 1 | 0.284 | 0.033 | 0.022 | 0.0034 | 0.019 | 0.10 |

The outflow's reflection of the pulse is least at the Mach number in the
root-mean-square and at half of it in the maximum, and the vortex leaves
most cleanly at 1 − M, where the pressure error is a fourteenth of the LODI
form's and a fifth of the default's. At Mach 0.1 the pulse reflection is
0.187, 0.117 and 0.273 under the LODI form, the Mach number and the full
share, and at Mach 0.5 it is 0.052, 0.131 and 0.279; the vortex leaves at
Mach 0.5 with a pressure error of 0.48, 0.19 and 0.37, and 0.064 at 1 − M.
The default is unchanged by this measurement: the weight that carries a
vortex out best, 1 − M, reflects the pulse worst at every Mach number but
0.1, and the outflow's task is the pulse's.

### The decision

`NSCBCInflowBC` defaults to `beta_t = 1`, the full share, which is the form
the literature restates for an inflow and the one under which the imposed
state follows its target through a transverse flow: the frozen-characteristic
property `test/runtests.jl` asserts holds only there. The cost is the pulse
reflection above, 0.13 of the incident amplitude against 0.047 at the Mach
number; a run whose inflow admits no structure and faces outgoing waves
selects `beta_t = -1`. `NSCBCOutflowBC` keeps the Mach number.

## Fold order and geometry limits

```text
julia --project=. -t 1 bench/foldorder.jl
```

`test/convergence.jl` reports a global max norm, and every one of its fold studies closes
the outer end with a `SlipWallBC` whose closure rows measure about 3 on their own. Splitting
that norm by region, on the same fields and resolutions, separates the two ends:

```
study                              fold(1:3)   mid    outer(3)   global argmax
C6, both ends walls (control)         3.23     3.19     3.17       i = n
cylindrical axis, odd  (u_r-like)     6.05     3.76     3.71       i = n
cylindrical axis, even (scalar)       7.01     2.99     3.00       i = n
spherical origin, even (scalar)       7.00     2.97     2.99       i = n
spherical origin, odd  (u_r-like)     6.07     3.86     3.81       i = n
```

**The global maximum sits at the outer wall in every fold study.** The fold's own error
converges at 6.05 to 7.01 and is three to five orders of magnitude below the interior: at N
= 96 the spherical origin carries 7.0e-12 against 1.9e-7 in the middle of the line. Every
global error `test/convergence.jl` prints equals the outer-window norm to every digit
printed, so the guarded numbers in that file are measurements of the outer wall taken
through a norm insensitive to the fold. With walls at both ends the same window reports
3.23, so the split does detect a third-order closure where one is present. The middle of the
line converges at 3 as well, the compact scheme's line-global coupling carrying the wall's
closure error inward rather than a property of the fold. Both parities were measured,
including the odd one that `test/convergence.jl` does not cover and that a converging
calculation differentiates at the origin. A face-centred symmetry plane reads the same
fold-window orders bitwise ([the face-centred symmetry
plane](#the-face-centred-symmetry-plane)).

**The spherical origin requires initial data resolved over ≳3 cells.** A blast initialized
as a top hat with a 1–2 cell transition loses positivity within tens of steps; at 3 cells
and wider it runs to completion. The cylindrical axis accepts a 1-cell transition and the
same top hat completes in Cartesian, so this is specific to the origin fold and its
antipodal pairing, and `test/cases.jl` initializes Sedov with a Gaussian deposit. Why the
origin fold is less forgiving than the cylindrical axis is open; the fold order above rules
out the closure.

**The spherical origin is incompatible with the singular t = 0 start of Noh.** Every CFL and
every constant setting fails, since the exact solution requires 64× compression to appear at
r = 0 instantaneously. A warm start from the exact solution at t = 0.3 integrates to 0.6,
testing maintenance of the solution through the origin without the initialization
singularity. The cylindrical axis accepts the cold start at 16× compression.

## Operator and step cost

```text
MPIEXEC=$(julia --project=. -e 'using MPI; MPI.mpiexec(c -> print(c))')
"$MPIEXEC" -n 8 julia --project=. -t 1 bench/derivcost.jl 128 30 dims=2,2,2
"$MPIEXEC" -n 8 julia --project=. -t 1 bench/derivcost.jl 128 10 dims=2,2,2 phases=true cases=periodic
julia --project=. -t 1 bench/phases.jl
```

Run-to-run spread on this workstation is 10–20%, so every ratio below is formed within a
process and every absolute figure is a median of three.

### The derivative operators

Wall-clock cost per grid point per step of `lele_d1_6`, `lele_d1_8` and `lele_d1_10` on a
single-species ideal-gas box, the Taylor–Green field as the initial state, the default
filter every step, `ArtParams()` defaults, cfl 0.5, Float64, on a 12th-gen Core i9-12900K (8
performance and 8 efficiency cores, 24 threads) under Microsoft MPI 10.1 from the JLL. Each
cell is warmed over three steps and timed over thirty through `solver.wall_total`, spanning
`max_rate`, `apply_bcs!`, the stages and the filter pass, reduced as the maximum over ranks
after a barrier; a fourth process with the operators in reverse order reproduces the forward
ratios inside the spread. The production configuration is eight ranks at one thread each.

```
                                            C6        C8        C10      C8/C6          C10/C6
128^3 on 8 ranks (2,2,2), periodic, -t 1   444.1     461.5     517.3    1.035–1.054    1.157–1.170
128^3 on 8 ranks, slip walls in x          447.3     459.2     518.1    1.010–1.041    1.090–1.167
64^3 on 8 ranks (32^3 per rank), periodic  503.3     520.7     604.8    0.979–1.087    1.202–1.231
64^3 on 8 ranks, slip walls in x           500.4     531.3     608.5    1.042–1.065    1.203–1.239
64^3 on 1 rank, -t 1, periodic            1353.5    1486.1    1748.8    1.072–1.106    1.266–1.308
64^3 on 1 rank, -t 16, periodic            402.6     416.9     498.5    1.030–1.042    1.220–1.251
64^3 on 1 rank, -t 16, slip walls in x     371.3     385.1     426.6    1.034–1.039    1.107–1.199
```

ns per point per step, global points over the slowest rank's wall; the ratio columns span
the processes. The phases of one right-hand-side evaluation at 128³ on eight ranks,
periodic, the maximum over ranks of each rank's minimum over repeated calls on a settled
state, ns per point (the first four inside `compute_rhs!`, `filter_state!` outside it):

```
                           8 ranks, -t 1                 1 rank, -t 16
                            C6      C8      C10          C6      C8      C10
velocity gradients         9.38   10.86   13.20         8.69    9.87   14.01
scalar gradients           5.23    6.28    8.01         6.03    6.60    9.28
artificial properties     18.77   20.14   17.63        16.06   16.27   15.55
assemble_fluxes!           7.32    7.32    6.86         8.42    9.19    9.28
compute_rhs! (whole)      73.64   82.75   90.26        71.32   73.27   92.20
filter_state!             24.35   21.98   21.63        19.19   18.25   19.31
```

In the production configuration C8 costs 4% of a step over C6 and C10 16%; at 32³ per rank
the C8 figure stays where it is and the C10 figure rises to about 20%, while every
operator's cost per point rises 13% from the communication share. The single-rank
single-thread ratios, 10% and 29%, are the undiluted arithmetic: the derivative solves are a
fifth to a quarter of the right-hand side, the artificial-property pass and the filter are
flat across the operators to within 3%, and a decomposed line solve adds a reduced-interface
stage and a halo exchange per direction that no operator changes, diluting C8's extra
multiply-adds more than C10's extra band since the pentadiagonal interface stage is itself
wider. The closed configuration costs the same as the periodic one on eight ranks and less
on one, where the closed line solve carries no cyclic correction. The pentadiagonal plans
add 1.1 MiB to a 234 MiB per-rank footprint at 64³ per rank, and the allocation per step
(7.7 KiB per rank) is the same across the operators.

Rank placement is not controllable with this launcher. Microsoft MPI's `-affinity` and
`-affinity_layout seq:P` options have no effect through the JLL `mpiexec`, which runs
without the `smpd` service and sets no affinity mask: sampled over two seconds of work,
every rank migrated over the whole machine including the efficiency cores, and two ranks
were seen on one logical CPU at the same instant. The process-to-process spread at 64³ per
rank is 2.7–4.2% on five of six cells (one 15% outlier), the size of the whole C8 effect,
and 19–30% at 32³ per rank in the periodic cells, so the C8 figure is quoted as a few
percent. Pinning would take a `SetProcessAffinityMask` call inside the run, not a launcher
flag.

### The sensor phase

`bench/phases.jl` on the two-species tube, back to back on one machine. The `artificial`
phase is `compute_artificial!`, and `line solves` counts the directional solves per
right-hand-side evaluation as (filter and gradient) + (`:d8` detector):

```
setting                          artificial   % of RHS   compute_rhs!   line solves
strain / strain / delta4*         1.050 ms     26.0%       4.178 ms      24 + 0
mu_sensor = velocity              1.535 ms     33.7%       4.690 ms      24 + 0
beta_sensor = ungated_dilatation  1.287 ms     29.8%       4.349 ms      24 + 0
detector = d8                     1.733 ms     35.9%       4.798 ms      24 + 8
mu_sensor = velocity, d8          2.678 ms     47.1%       5.775 ms      24 + 14
```

The velocity sensor detects three fields where the strain sensor detects one, which is +46%
on the sensor phase under δ⁴. Paired with `:d8` it adds six more pentadiagonal solves per
right-hand side, and the sensor phase then costs more than everything else in the evaluation
combined. `:d8` alone is eight pentadiagonal solves per right-hand side, one per active
dimension per sensor, for +80% on the sensor phase and +19% on the whole evaluation in a
separate back-to-back pair (0.946 ms and 23.5% against 1.707 ms and 34.6%); against a 10–20%
run-to-run spread the phase figure is resolved and the total is marginal.

The smoother moves the same phase the other way: the `artificial` phase falls from 1.360 ms
to 0.971 ms, 31.8% to 24.8% of the right-hand side, from `:compact` to the default
`:gaussian`, against a run-to-run spread of about 1.3% over three `:compact` readings. The
Gaussian is an explicit nine-point stencil with no line solve and no interface reduction,
where the compact pass costs a distributed line solve per active dimension per sensor,
`n_species` of them per evaluation for the species sensors.

`compute_artificial!` is therefore 24.8–26.0% of the multicomponent right-hand side under
the default smoother, most of it in the smoothing of the sensors, one sweep per species. At
`n_species == 2` that per-species machinery is a measurable no-op, and it earns its cost
only at three or more species; cutting it is a numerics decision, a shared against a
per-species sensor, and not a code tweak.

## AMR

Patch AMR, the level hierarchy and the device backend. The mechanism is in
`reference/AMR_GPU.md`; the numbers are here.

### bench/interfaceconservation.jl: composite conservation budgets

`mpiexec -n 8 julia --project=. -t 1 bench/interfaceconservation.jl N=96 ny=24
tfinal=50.26548245743669 moving_tfinal=50.26548245743669 samples=16 check=true`,
and the same at 1, 2 and 4 ranks.

The case is periodic passive-species shear on [0, 8π) × [0, 2π): two
thermodynamically identical ideal species at uniform density and pressure,
u = 1, v = 0.08 sin(x/4), and the composition a periodic pair of perturbed
tanh sheets, width parameter 0.22 for the fixed layouts and 0.18 for moving
refinement. 96 × 24, Float64, C6, the default C8 filter every step, CFL 0.45,
no physical transport, the artificial properties off except the
mass-fraction bound, permissive validity (below). The layouts are the uniform
grid, two same-level patches, a two-level and a three-level nest each with
global steps and subcycled, and the two-level nest regridded 16 times under
species-gradient tagging. Every run lasts two domain transits, t = 16π.

The budgets were declared before the measurement: 1e-3 of each initial
species mass, of total mass and of total energy; 1e-3 of M₀c₀ per momentum
component, which gives the initially zero transverse components a scale; the
same for each layout's initial sampling offset from the uniform run; 0.01 in
molecular mixing and 1% of the domain length in mix width for the mixing
comparisons. The conserved snapshot is the covered-cell node quadrature with
shared planes counted once, and each layout's drift is taken from its own
initial integral, sampled sixteen times so that a cancelling excursion is not
hidden by the endpoint. The regrid ledger records the jump across each
transfer separately from the evolution between transfers.

**Fixed layouts.** The sampled maximum of the normalized drift over two
transits, and the largest mass-fraction excursion outside [0, 1] seen at any
sample. The drift is an exchange between the two species: total mass,
momentum and energy hold to 1e-11 in every layout, and the two species masses
move by equal and opposite amounts.

| layout | root steps | max drift | species excursion |
|---|---|---|---|
| uniform | 1184 | 1e-13 (round-off) | 0 |
| two same-level patches | 1184 | 1.6e-9 | 4.2e-5 |
| two levels, global step | 3532 | 7.0e-5 | 0 |
| two levels, subcycled | 1184 | 6.9e-5 | 0 |
| three levels, global step | 10570 | 7.6e-5 | 5.8e-4 |
| three levels, subcycled | 1184 | 7.6e-5 | 0 |

The initial sampling offsets from the uniform run are 8.1e-6 (two levels)
and 8.7e-6 (three levels); the mixing comparisons stay within 2.1e-3 of the
domain length in width and 1.2e-3 in molecular mixing over the history, both
an order inside their budgets. The numbers are the same to the printed
precision at 1, 2, 4 and 8 ranks, apart from the round-off of the uniform
baseline.

**Moving refinement.** Sixteen regrids over two transits, global step and
subcycled.

| quantity | global step | subcycled |
|---|---|---|
| per-regrid jump, typical | 5e-5 | 5e-5 |
| sum of the absolute jumps | 8.4e-4 | 8.4e-4 |
| sampled max drift, transfers included | 1.0e-4 | 1.0e-4 |
| sampled max drift, transfers subtracted | 2.1e-4 | 2.2e-4 |
| final drift | 1.5e-5 | 2.2e-5 |
| species excursion | 7.4e-4 | 5.6e-4 |

The absolute sum grows with the regrid count while the signed sum stays near
2e-5, so the transfers do not accumulate a bias at this count; the budget
compares the absolute sum, and a run regridding more than about twenty times
at this spacing would exceed it on that measure alone. Regridding a
three-level hierarchy is unsupported and is not measured.

**Why the runs are permissive, and the bound is on.** With the artificial
properties off entirely, strict validity ended the three-level global-step
run at t = 37.7, the long moving-refinement run at t = 25.1 and an unfiltered
uniform run at t = 28.3, in every case on mass fractions past the 1e-4 dead
band. A per-step probe of the three-level run shows no growth anywhere else:
velocity, pressure and density stay at their initial values to four digits,
and the step size is steady. The undershoot sits on the root and level-1
nodes along the edges of the nested box, is advected with the flow, and grows
from 1e-4 at t = 34 to 3e-4 at t = 38; the mass-fraction bound at its default
slows it but does not hold it below the dead band, since it acts in
proportion to the excursion. The same layout subcycled shows no excursion at
all. The drift up to the point of failure was inside the budget in every one
of those runs, so the validity failure and the conservation budget are
independent measurements, and the instrument reports the excursion beside
the drift instead of stopping on it.

**Decision.** Every layout is inside its conservation budget, the fixed
layouts by an order of magnitude, so no surface-flux correction is enabled and the coupling stays as
it is; the requirements a correction would have to meet are in
[AMR_GPU.md](AMR_GPU.md#composite-conservation-budgets). The root-edge
mass-fraction undershoot under global stepping is an imposed-shell accuracy
question for N11, not a conservation one. No turbulent, variable-density,
shock or hardware-GPU claim follows from these passive-species cases.

### bench/interfacesensor.jl: the sensors and the filter at an interface

`julia --project=. -t 16 bench/interfacesensor.jl` (`probe`, `crossing`,
`filter` and `undershoot`; the toggle `SENSOR_INTERFACE_GHOSTS` selects
the detector's interface taps).

**The operator probe.** One application of each operator on
ρ = 1 + 0.3 sin(2π(x + 1/7)) + 0.05 sin(10π(x + 1/7)), p = 1, u = 0 on
[0, 1), against the uniform periodic grid of the same spacing: the level-1
patch of a two-level nest at h/3, its shell imposed from the root's own
initialization, and the two patches of a same-level pair. The entry is
|interface − uniform| over the input amplitude at the first node in from
the face; the orders are log₂ of successive ratios. The detector runs at
weight power zero on the internal energy.

| operator, face | N = 48 | 96 | 192 | orders |
|---|---|---|---|---|
| δ⁴, ghost taps, shell | 2.2e-5 | 3.3e-7 | 5.0e-9 | 6.06 / 6.07 |
| δ⁴, clamped taps, shell | 2.3e-2 | 1.2e-2 | 5.8e-3 | 1.00 / 1.00 |
| δ⁴, ghost taps, same level | 0 | 0 | 0 | bitwise |
| δ⁴, clamped taps, same level | 2.8e-2 | 1.6e-2 | 8.4e-3 | 0.78 / 0.96 |
| `:gaussian` smoother, shell | 5.9e-3 | 3.0e-3 | 1.5e-3 | 0.95 / 0.98 |
| `:compact` smoother, shell | 9.4e-11 | 3.7e-13 | 1.5e-15 | 7.98 / 7.94 |
| filter, extended rows, shell, node 2 | 1.4e-7 | 2.0e-9 | 3.0e-11 | 6.08 / 6.08 |
| filter, onesided rows, shell, node 2 | 6.5e-12 | 3.6e-13 | 1.6e-15 | 4.17 / 7.78 |
| filter, cascade rows, shell, node 2 | 3.2e-5 | 9.6e-6 | 2.6e-6 | 1.75 / 1.90 |
| filter, extended rows, same level | 2.4e-7 | 9.7e-10 | 3.8e-12 | 7.96 / 7.99 |
| filter, onesided rows, same level, node 2 | 7.0e-7 | 3.6e-9 | 2.4e-11 | 7.59 / 7.23 |
| filter, cascade rows, same level, node 2 | 3.5e-4 | 1.1e-4 | 2.8e-5 | 1.73 / 1.95 |

The clamp is a first-order error in the sensor at the two nodes it
touches and exactly zero beyond them; the ghost taps carry the shell's
order-6 interpolation error at a coarse-fine face and reproduce the
uniform grid bitwise at a same-level face. The `:gaussian` smoother's
closed-edge rows add a first-order error over four nodes where the
`:compact` smoother stays at its own order. The filter's identity row
leaves node 1 at round-off under every row set. At a coarse-fine face the
extended rows read the imposed ghosts and so carry the interpolation
error into nodes 2 to 6, three decades above the filter's own change on
this field; the one-sided rows read nothing and stay at round-off. At a
same-level face the ghosts are exact neighbour data and the ordering
reverses: the extended rows track the uniform filter at order 8 and the
closed rows depart from it. The cascade rows are the worst set at both
faces.

**The Sod crossing.** Root N = 201 on [0, 1] between slip walls, CFL 0.4,
the default artificial properties and filter, the level-1 box over
[0.6, 0.8], run to t = 0.2 so that the shock enters through the low face
near t = 0.057 and leaves through the high face near t = 0.17. The base
row is C6, subcycled, one tile, `filter_cfl = 0.35`, the default rows,
ghost taps; each other row changes one setting. The columns are the
momentum ahead of the shock at t = 0.1 (x > 0.85, the level suite's gate
quantity), the minimum density and pressure over every patch on any
step, the artificial diffusivity number ((μ\* + β\*)/ρ + κ\*/(ρc_p) +
max D\*)/(ch) over the four fine nodes nearest each face at t = 0.03,
before the shock arrives, and at t = 0.1, and the density error against
the uniform run at the fine spacing over those nodes at t = 0.2. No row
counted an inadmissible point on any step.

| row | noise | ρ_min | p_min | shell number, t = 0.03 | t = 0.1 | shell error, t = 0.2 |
|---|---|---|---|---|---|---|
| base | 7.4e-11 | 0.061 | 0.040 | 9.8e-5 | 2.5e-3 | 7.7e-3 |
| clamped taps | 7.4e-11 | 0.061 | 0.040 | 9.7e-5 | 2.5e-3 | 7.7e-3 |
| C10 | 6.4e-10 | 0.067 | 0.045 | 2.6e-4 | 6.3e-3 | 7.7e-3 |
| global step | 3.5e-10 | 0.062 | 0.038 | 1.3e-4 | 2.3e-3 | 8.9e-3 |
| tile 8 | 7.0e-11 | 0.061 | 0.040 | 9.8e-5 | 4.4e-1 | 8.3e-2 |
| three levels | 2.9e-10 | 0.061 | 0.040 | 9.4e-5 | 1.2e-2 | 6.3e-3 |
| `filter_cfl = 0` | 1.0e-10 | 0.054 | 0.033 | 1.6e-4 | 4.1e-3 | 8.6e-3 |
| `:cascade4` rows | 7.1e-11 | 0.061 | 0.040 | 1.2e-4 | 2.7e-3 | 7.9e-3 |
| `:brady_livescu` rows | 7.6e-11 | 0.061 | 0.040 | 4.1e-5 | 4.8e-2 | 8.2e-3 |
| artificial properties off | 3.9e-11 | 0.090 | 0.050 | 0 | 0 | 1.4e-3 |
| three same-level patches | 3.6e-10 | 0.062 | 0.041 | 1.8e-5 | 4.5e-1 | 7.4e-2 |
| two species, ghost taps | 7.2e-11 | 0.061 | 0.041 | 7.1e-5 | 2.7e-2 | 7.7e-3 |
| two species, clamped taps | 7.2e-11 | 0.061 | 0.041 | 7.0e-5 | 2.7e-2 | 7.7e-3 |

The minimum density is the step data's start-up transient at step 3, and
the uniform run reads the same 0.062 at both spacings. Under the tiled
level and the three-patch layout the shell window includes the tile and
patch faces the shock is crossing at t = 0.1, so those two rows read the
shock itself. The tap policy moves nothing past the third digit on a
shock, whose own δ⁴ signal is orders above the clamp's error; the
two-species contact carries the composition through both faces with a
mass-fraction excursion of 1.4e-8 over the shell against 4.0e-4 at the
contact itself, under either policy. The `:brady_livescu` rows put
twenty times the base row's sensor on the shell during the crossing and
complete it; no closure candidate fails this gate and none is promoted
by it.

A discontinuity initialized on a same-level plane is the one failure.
With two patches the shared plane is node 101, the Sod diaphragm, and
the run fails on negative density at step 2 with ρ = −50 at the plane;
the artificial properties off (step 10) and the filter off (step 2) fail
the same way, the one-sided interface rows complete the run, a diaphragm
one node to the right of the plane completes it, one node to the left or
adjacent on a 200-node grid fails, and a coarse-fine box face on the
diaphragm completes. The extended gradient rows read the jump through
the ghosts while the divergence keeps its one-sided rows, and the
mismatch on a step that is one node wide is what fails; a captured shock
arriving from the interior is several nodes wide and crosses either kind
of face without incident, as every other row shows.

**The filter's change at the shell.** The density change of each filter
pass, its maximum over the run at each distance from a coarse-fine face
relative to its maximum over the fine interior, global stepping. The
entropy wave is ρ = 1 + 0.1 sin(2πx), u = 1, p = 1 at root N = 96 to
t = 0.5; the Sod crossing is the base case above to t = 0.2; the
two-dimensional row is the wave on (96, 24, 1) with a two-dimensional
box, where the plane column is the transverse pass.

| case, rows | interior | plane | node 2 | node 3 | node 4 | node 6 |
|---|---|---|---|---|---|---|
| wave, extended | 2.8e-11 | 8e-6 | 2.8 | 3.9 | 3.9 | 2.0 |
| wave, onesided | 3.1e-10 | 0 | 1.4 | 2.2 | 2.5 | 1.7 |
| wave, cascade | 4.3e-7 | 0 | 3.3 | 2.1 | 2.0 | 1.6 |
| Sod, extended | 2.8e-4 | 2e-13 | 5.5 | 5.3 | 3.1 | 1.2 |
| Sod, onesided | 1.6e-3 | 4e-14 | 1.1 | 1.7 | 1.9 | 1.5 |
| Sod, cascade | 3.1e-4 | 2e-13 | 8.8 | 7.7 | 4.1 | 2.0 |
| 2-D wave, extended | 8.4e-10 | 1.4 | 14 | 18 | 8.0 | 1.7 |
| 2-D wave, onesided | 6.4e-9 | 0.26 | 14 | 14 | 11 | 4.5 |
| 2-D wave, cascade | 6.3e-7 | 8.0 | 24 | 20 | 13 | 1.7 |

Under the default extended rows the filter acts three to five times
harder on the two nodes beside a coarse-fine face than in the fine
interior, and fifteen times harder in two dimensions, where the
transverse pass also moves the plane node itself by its interior level
before the shell is re-imposed. The one-sided rows filter the whole fine
interior five to ten times harder, since their closed rows make content
the extended rows do not, and the cascade rows are four decades worse on
the smooth wave.

**The root-edge undershoot.** The two-species layer of
`bench/interfaceconservation.jl` on (96, 24, 1) with only the
mass-fraction bound on, permissive, over two transits, sampled eight
times: the minimum mass fraction by distance from the level-1 box edge,
in root nodes on the root and in fine nodes from the imposed plane on
each fine level. The toggle changes no digit in any row, since the bound
is the only species sensor and it is zero inside [0, 1].

| nest, stepping | t = 37.7, root face node | root outside, 2–3 nodes | level-1 plane | t = 50.3, root face node | root outside, 4+ nodes | level-1 interior | level-2 interior |
|---|---|---|---|---|---|---|---|
| two levels, global | +1.6e-4 | +1.3e-4 | +1.6e-4 | +1.3e-4 | +3.9e-5 | +2.5e-4 | — |
| two levels, subcycled | +2.9e-4 | +1.3e-4 | +2.9e-4 | +2.6e-4 | +1.0e-4 | +4.1e-4 | — |
| two levels, global, no filter | −2.4e-4 | −2.9e-4 | −2.4e-4 | −4.7e-4 | −4.9e-4 | −3.6e-4 | — |
| three levels, global | −1.2e-4 | −2.5e-4 | −1.2e-4 | −5.1e-4 | −5.8e-4 | −3.4e-4 | −9.8e-5 |
| three levels, subcycled | +2.9e-4 | +1.3e-4 | +2.9e-4 | +2.6e-4 | +1.0e-4 | +4.1e-4 | +6.4e-4 |
| three levels, global, unrelaxed filter | −1.1e-4 | −2.5e-4 | −1.1e-4 | −5.0e-4 | −5.8e-4 | −3.4e-4 | −9.8e-5 |

A positive entry is a minimum above zero, so the two-level nests and both
subcycled nests carry no undershoot at any sample. Without the filter the
sheet undershoots on its own, at the box face and four or more root nodes
outside it alike, which is the unfiltered ringing the conservation
instrument's header describes and not an interface effect. The
three-level nest under global stepping is the one refined case that
undershoots, and it does so at the level-1 boundary plane first, at
t = 37.7 on the coarse node coincident with that plane and the two or
three root nodes outside it while the root interior and level 2 are still
positive; by t = 50.3 the flow has carried it over the whole root outside
the box and through level 1. The unrelaxed filter reproduces every entry
to two digits. Direct weight measurements below show that global stepping
uses the finest directional rate on every patch; the root pass is not
relaxed in proportion to its own coarse spacing.

**Global-step follow-up (N12).** Reducing the two-level CFL from 0.45 to
0.15 matches the three-level root-step scale without adding another
interface. The two-level run then develops an undershoot too, so the
second transfer is not necessary for its onset. A diagnostic driver that
restricts fine data onto the parent every third step retains the same
RK stages, shell impositions and filter passes. It produces almost the
same undershoot, ruling out the frequency of fine-to-coarse restriction.

| two-level run, CFL 0.15 | t | completed steps | root face / level-1 plane | root outside, 4+ nodes | level-1 interior, 5+ nodes |
|---|---|---|---|---|---|
| production | 37.6991 | 7929 | +2.348e-4 | +8.906e-5 | +7.005e-4 |
| production | 43.9823 | 9251 | +3.027e-5 | +6.388e-5 | +7.047e-5 |
| production | 50.2655 | 10572 | -7.997e-5 | -2.395e-4 | +2.588e-4 |
| restriction every third step | 50.2655 | 10572 | -8.091e-5 | -2.397e-4 | +2.567e-4 |

Keeping the small timestep but filtering every third step restores the
ordinary two-level physical filter cadence and remains positive at every
sample: at t = 50.2655 the root face minimum is +1.347e-4, the root outside
four nodes is +3.852e-5, and the level-1 plane and interior minima are
+1.002e-4 and +2.511e-4. Thus the smaller RK timestep alone does not cause
the undershoot; the changed filter application schedule is required in
this reproducer.

The startup weights are identical on every patch of a hierarchy:

| levels | CFL | filter interval | dt | x weight | y weight |
|---|---|---|---|---|---|
| 2 | 0.45 | 1 | 0.0143184855 | 1 | 0.830373924 |
| 3 | 0.45 | 1 | 0.00478678860 | 1 | 0.827484401 |
| 2 | 0.15 | 1 | 0.00477282850 | 0.482471992 | 0.276791308 |
| 2 | 0.15 | 3 | 0.00477282850 | 1 | 0.830373924 |

Under global stepping the finest directional rate compensates for the
smaller timestep in the per-pass weight, while another level adds about
three times as many coarse-grid filter passes per unit physical time.
Reducing CFL on a fixed hierarchy also changes the number and weights of
the interleaved filter and shell operations; the x weight's cap makes even
the integrated relaxation strength different in these rows. The experiments
identify sensitivity to this filter schedule, not a CFL violation or a
restriction-frequency failure, and do not isolate an individual operator
commutator. Keep subcycling as the demonstrated remedy; the subsequent
[level-aware filter qualification](#benchlevelfilterjl-level-aware-filtering-under-global-stepping)
measures candidate global-step policies and their accuracy tradeoffs.

Reproduce with `julia --project=. -t 1 bench/interfacesensor.jl undershoot
layerN=96 layerny=24 layer_tfinal=50.26548245743669 samples=8 nmax=40000
undershoot_depths=2 "undershoot_variants=depth-3 dt" undershoot_ghosts=on`;
select `"undershoot_variants=sparse projection"` for the restriction
ablation or `"undershoot_variants==two levels, matched dt, filter every 3"`
for the filter cadence. The leading equals sign selects the exact label
instead of comma-separated substrings. These are controlled changes to
operation cadence, not a proposal to weaken synchronization in production.

**Decision.** The detector reads the ghost layers at an interface face
for every field recovered over the padded extent, the default; the strain
and dilatation sensors, computed on the interior, keep the clamp, and the
smoother keeps its closed-edge rows since its input has no ghosts. The
filter keeps the extended interface rows; the cascade rows are the worst
set at either kind of face. `detector = :d8` stays rejected in patched and
refined runs, since no interface rows exist for its pentadiagonal
operator. The closure candidates all pass the crossing gate, which
therefore promotes none of them. The step-on-a-plane failure and the
three-level undershoot under global stepping are recorded as limitations:
the first is an initial-data restriction with a stated workaround, the
second is removed by subcycling.

### bench/boundaryorder.jl idiv: the interface divergence rows

```text
julia --project=. -t 1 bench/boundaryorder.jl study=idiv
julia --project=. -t 1 bench/interfacesensor.jl crossing "crossing_variants=base,idiv,reversed,two patches,three patches"
julia --project=. -t 4 bench/interfaceconservation.jl N=96 ny=24 tfinal=8.0 moving_tfinal=8.0 idiv=brady_livescu
```

`interface_divergence` replaces the flux divergence's closure rows at patch and level
interface ends and nowhere else; the derivative operator stays `lele_d1_6()`, so a wall keeps
the `:neutral3` rows. Without it an interface takes the `:cascade3` rows, and those rows below
reproduce [the matrix](#patch-interfaces-and-refinement-levels) to every digit. Serial
Float64, `cfl = 0.25`, the interface window of the matrix, root N = 48, 96, 192 (49, 97, 193
for the wall rows); the `dt` column is below 0.1 on every row cited as an order unless
marked.

```
                                            default                cascade4               Brady–Livescu
                                            N=192     orders       N=192     orders       N=192     orders
RHS on exact data, two patches, k=3         2.395e-5  3.36 / 3.25  3.546e-6  3.98 / 4.03  3.861e-7  5.27 / 5.41
RHS on exact data, 2 levels, k=3            1.849e-6  3.03 / 3.02  1.815e-8  4.23 / 4.14  2.435e-9  5.02 / 5.03
two patches, entropy k=3                    6.315e-7  5.10 / 4.71  2.716e-7  5.23 / 5.02  1.119e-8  6.63 / 6.96
two patches, entropy k=1                    8.262e-9  3.09 / 3.53  1.078e-9  5.64 / 4.97  8.611e-12 6.49 / 6.52
two patches, standing wave                  1.949e-8  3.41 / 3.87  6.000e-9  5.45 / 2.39  4.504e-11 5.77 / 5.82
two patches, standing wave, filtered        1.025e-8  3.63 / 3.90  9.671e-9  4.32 / 2.00  4.196e-11 5.97 / 5.45
2 levels, entropy k=3                       9.408e-8  3.41 / 3.83  4.428e-10 5.71 / 3.92  6.401e-11 6.02 / 6.01
2 levels subcycled, entropy k=3             9.198e-8  3.43 / 3.84  4.475e-10 5.74 / 3.88  6.480e-11 6.02 / 5.99
3 levels, entropy k=3                       7.407e-8  3.50 / 3.89  4.481e-10 5.72 / 3.89  6.391e-11 6.02 / 6.01
3 levels subcycled, entropy k=3             7.192e-8  3.52 / 3.92  4.410e-10 5.75 / 3.87  6.491e-11 6.02 / 5.99
2 levels, tile 4, subcycled, entropy k=3    7.205e-8  3.99 / 3.72  1.572e-8  4.20 / 3.36  6.481e-11 6.02 / 5.99
2 levels, entropy k=3, filtered             1.577e-8  4.29 / 3.95  1.049e-10 7.01 / 6.80  7.378e-10 5.25 / 5.81
2 levels, entropy k=1                       7.380e-10 3.37 / 3.76  8.036e-12 4.31 / 4.93  5.174e-14 5.33 / time
2 levels, standing wave                     1.030e-9  3.37 / 3.47  9.571e-12 4.76 / 4.59  4.998e-13 5.13 / 5.11
walls and an interface, two patches         4.234e-10 5.04 / 5.66  1.322e-7  2.38 / 1.20  2.375e-9  4.56 / 5.19
walls and an interface, filtered            1.860e-10 4.26 / 4.86  9.169e-8  2.72 / −3.01 1.211e-10 5.20 / 4.03
viscous, two patches, extended gradients    2.469e-8  3.62 / 3.87  1.419e-9  4.87 / 4.99  6.850e-11 4.92 / 5.66
viscous, two patches, one-sided gradients   2.470e-6  1.91 / 0.91  4.788e-6  0.49 / 0.23  4.770e-6  0.84 / 0.16
viscous, 2 levels, extended gradients       8.597e-10 3.61 / 3.65  7.309e-12 4.85 / 4.75  3.511e-13 5.57 / 5.34
viscous, 2 levels, one-sided gradients      3.455e-9  3.44 / 3.66  3.652e-10 2.92 / 3.07  4.927e-10 3.25 / 3.30
```

The standing wave through two levels at `cfl = 0.125` and root N = 96, 192, 384 reads
2.022e-11, 3.459e-13, 6.084e-14 under the Brady–Livescu rows (5.08, then the round-off
floor, `dt` 0.004 at N = 192) and 1.109e-8, 7.778e-10, 7.504e-11 under the default. The
walls-and-interface rows' wall window is 3.1e-9 (default) and 2.8e-9 (Brady–Livescu) at
N = 193, the `:neutral3` wall's fourth order, and the interface window reads that error.

In Float32 the entropy wave k = 1 reads 3e-6 to 6e-5 in the interface window at every N and
under every source, rising with N; no row converges and no source lowers the floor.

The acoustic pulse of the reflection tests, the left-running characteristic upstream of
the first face over the pulse amplitude, against the run without the face:

```
                      N = 96                           N = 192                          N = 384
                      default   cascade4  BL           default   cascade4  BL           default   cascade4  BL
two patches           9.740e-6  6.060e-5  1.543e-5     2.049e-6  8.356e-6  1.139e-6     2.685e-7  1.897e-6  2.477e-8
2 levels              1.182e-6  1.241e-6  1.189e-6     1.188e-8  3.062e-9  1.240e-9     8.427e-10 1.046e-10 6.066e-11
2 levels subcycled    8.314e-7  8.928e-7  8.933e-7     1.051e-8  2.897e-9  1.099e-9     7.058e-10 8.713e-11 5.242e-11
```

**Shocks.** The Sod crossing of
[the interface sensor instrument](#benchinterfacesensorjl-the-sensors-and-the-filter-at-an-interface),
root N = 201, `cfl = 0.4`, the artificial properties and the filter live; shell and root
columns are the density error against the uniform run at t = 0.2:

| row | default | `:cascade4` | Brady–Livescu |
|---|---|---|---|
| two levels subcycled: steps, p_min, shell, root | 373, 0.0405, 7.7e-3, 2.4e-2 | 373, 0.0405, 7.9e-3, 2.4e-2 | 373, 0.0405, 8.2e-3, 2.6e-2 |
| global dt: p_min, shell | — | 0.0382, 9.0e-3 | 0.0382, 1.0e-2 |
| three levels: shell | — | 6.5e-3 | 7.0e-3 |
| tile 8: p_min, shell | — | 0.0405, 8.3e-2 | 0.0206, 7.8e-2 |
| three patches: p_min, shell, root | 0.0412, 7.4e-2, 2.7e-2 | 0.0412, 7.7e-2, 2.7e-2 | 0.0205, 6.8e-2, 5.5e-2 |
| diaphragm on the shared plane | negative density, step 2 | step 1 | step 1 |
| the same, one-sided gradients | completes, p_min 0.097 | negative density, step 113 | step 2 |

The mirrored tube, the shock running right to left through the same faces, reproduces every
printed digit of the forward rows. The Brady–Livescu rows halve the minimum pressure where
the shock crosses a same-level face, and neither candidate survives a discontinuity on a
shared plane; both remove the one-sided gradient rows as a workaround for it.

**Conservation.** The layer of
[the conservation instrument](#benchinterfaceconservationjl-composite-conservation-budgets)
on (96, 24, 1) to t = 8: the nests and the regrid histories agree across the three sources
to the second digit (6.9e-5 to 7.1e-5 fixed-nest drift, 2.2e-4 cumulative regrid jump, all
within the 1e-3 budget). The same-level layout's drift is 5.2e-10, 5.9e-10 and 9.8e-11
under the default, `:cascade4` and Brady–Livescu rows, and its largest mass-fraction
excursion 3.1e-5, 2.2e-4 and 4.0e-4.

**Decision.** The default stays. `:cascade4` is rejected: the undamped mode it carries at an
inviscid wall appears at an interface on the standing wave and grows between walls and an
interface. The Brady–Livescu rows reach 5.8–7.0 at a same-level interface and 6.0 at every
coarse-fine nest on the entropy waves, but 5.1 on the acoustic standing wave at a
coarse-fine face, their formal fifth order, below the 5.5 target; they halve the minimum
pressure of a shock crossing a same-level face and raise the same-level mass-fraction
excursion thirteenfold. The option stays experimental, a Float64 smooth-flow setting.

### bench/boundaryorder.jl gflux: the divergence through interface ends from ghost fluxes

```text
julia --project=. -t 1 bench/boundaryorder.jl study=gflux
julia --project=. -t 1 bench/interfacesensor.jl crossing "crossing_variants=base,gflux,two patches,three patches"
julia --project=. -t 4 bench/interfaceconservation.jl N=96 ny=24 tfinal=8.0 moving_tfinal=8.0 iflux=ghost [mu=1e-3]
```

`interface_flux = :ghost` evaluates the inviscid flux on the padded block, the interface ghosts
included, and differences it through the gradient plans' interface rows. The molecular flux
joins it: its ghost values at a same-level face are the neighbouring patch's own interior
flux, exchanged after every patch of the level has evaluated, and at a coarse-fine face they
are evaluated from the conserved gradients of the interpolated fine box, taken with explicit
seventh-order rows at the box ends. The artificial fluxes and the wall corrections take the
one-sided rows of `div_plans`, the default's or an `interface_divergence` source's. The
ghost column runs at the ghost path's default level interpolation order, 8. Serial Float64,
`cfl = 0.25`, the matrix's interface window, root N = 48, 96, 192 (49, 97, 193 for the wall
rows), beside the default and Brady–Livescu (BL) columns of the section above, which this
study reproduces to every digit. The polynomial row is `polynomial_case(96)`, every inviscid
flux component of degree at most five, at the interface window:

```
                                            default                BL                     ghost
                                            N=192     orders       N=192     orders       N=192     orders
polynomial RHS, two patches (ρu, E)         2.09e-8, 1.06e-8       4.58e-12, 9.25e-13     2.99e-14, 2.18e-14
polynomial RHS, 2 levels (ρu, E)            7.73e-10, 4.20e-10     5.68e-12, 1.01e-11     1.58e-13, 1.81e-13
RHS on exact data, two patches, k=3         2.395e-5  3.36 / 3.25  3.861e-7  5.27 / 5.41  1.785e-9  5.97 / 5.99
RHS on exact data, 2 levels, k=3            1.849e-6  3.03 / 3.02  2.435e-9  5.02 / 5.03  1.280e-10 6.39 / 6.00
two patches, entropy k=3                    6.315e-7  5.10 / 4.71  1.119e-8  6.63 / 6.96  1.416e-10 6.75 / 6.78
two patches, entropy k=1                    8.262e-9  3.09 / 3.53  8.611e-12 6.49 / 6.52  6.950e-14 6.88 / 6.70
two patches, standing wave                  1.949e-8  3.41 / 3.87  4.504e-11 5.77 / 5.82  4.308e-12 5.44 / time
two patches, standing wave, filtered        1.025e-8  3.63 / 3.90  4.196e-11 5.97 / 5.45  4.516e-12 5.35 / time
2 levels, entropy k=3                       9.408e-8  3.41 / 3.83  6.401e-11 6.02 / 6.01  6.380e-11 6.02 / 6.01
2 levels subcycled, entropy k=3             9.198e-8  3.43 / 3.84  6.480e-11 6.02 / 5.99  6.481e-11 6.02 / 5.99
3 levels, entropy k=3                       7.407e-8  3.50 / 3.89  6.391e-11 6.02 / 6.01  6.379e-11 6.02 / 6.01
3 levels subcycled, entropy k=3             7.192e-8  3.52 / 3.92  6.491e-11 6.02 / 5.99  6.481e-11 6.02 / 5.99
2 levels, tile 4, subcycled, entropy k=3    7.205e-8  3.99 / 3.72  6.481e-11 6.02 / 5.99  6.481e-11 6.02 / 5.99
2 levels, entropy k=3, filtered             1.577e-8  4.29 / 3.95  7.378e-10 5.25 / 5.81  1.049e-10 6.96 / 6.78
2 levels, entropy k=1                       7.380e-10 3.37 / 3.76  5.174e-14 5.33 / time  3.431e-14 6.01 / 5.77
2 levels, standing wave                     1.030e-9  3.37 / 3.47  4.998e-13 5.13 / 5.11  1.666e-13 6.12 / 5.94
2 levels subcycled, standing wave           9.640e-10 3.37 / 3.47  1.519e-11 4.70 / time  1.938e-11 4.84 / time
walls and an interface, two patches         4.234e-10 5.04 / 5.66  2.375e-9  4.56 / 5.19  1.525e-10 4.27 / 4.98
walls and an interface, filtered            1.860e-10 4.26 / 4.86  1.211e-10 5.20 / 4.03  1.316e-10 4.89 / 4.12
viscous, two patches                        2.469e-8  3.62 / 3.87  6.850e-11 4.92 / 5.66  3.151e-13 6.75 / 6.43
viscous, 2 levels                           8.597e-10 3.61 / 3.65  3.511e-13 5.57 / 5.34  1.060e-13 6.29 / 6.06
viscous, 2 levels subcycled                 8.529e-10 3.61 / 3.64  3.400e-13 5.53 / 5.39  1.054e-13 6.28 / 6.06
```

At a same-level face the interface window's error equals the interior's on every ghost row,
viscous ones included (the entropy wave k = 3 reads 1.42e-10 at the face and 1.55e-10 inside);
the inviscid standing-wave slopes there are set by the time integrator (`dt` column 0.83–0.86
at N = 192). The BL rows on the remainder change nothing without the artificial properties:
the ghost path leaves no remainder, and the `ghost+BL` column repeats the ghost column to
every digit. At a coarse-fine face the ghost state is the interpolation of the parent, whose
O(h^p) value error enters the differenced inviscid flux at h^(p−1) and the molecular flux at
h^(p−2). At order 6 the RHS on exact data converged at 5.01 against the BL rows' 5.02 with a
constant eight times larger and the evolution errors sat five to twelve times above the BL
rows; at order 8 every coarse-fine row matches or leads them. The viscous right-hand side
on exact data within four nodes of an interface end, against the periodic operator at the
patch's spacing (the check of `test/patch_tests.jl`, on one species), converges at 5.92 / 5.98
through two patches and 6.73 / 5.44 / 7.02 through a level (N = 48 to 384; 6.23 then the 1e-12
floor at order 10). Under the third-order closure rows of the operator's own set at the box ends, the level row
had instead stalled at 5e-11, a second-order floor: the box ends lie eight to eleven fine
nodes from the ghost layers, and a closure row's error decays by 2 − √3 per node.

In Float32 the entropy wave k = 1 keeps the floor of the other sources (4.4e-6 two patches,
8.4e-6 two levels at N = 192).

The acoustic pulse of the reflection tests, left-running characteristic over the amplitude:

```
                      N = 96                           N = 192                          N = 384
                      default   BL        ghost        default   BL        ghost        default   BL        ghost
two patches           9.740e-6  1.543e-5  1.540e-6     2.049e-6  1.139e-6  2.049e-8     2.685e-7  2.477e-8  5.775e-10
2 levels              1.182e-6  1.189e-6  9.773e-7     1.188e-8  1.240e-9  9.675e-10    8.427e-10 6.066e-11 2.031e-11
2 levels subcycled    8.314e-7  8.933e-7  4.824e-7     1.051e-8  1.099e-9  1.417e-9     7.058e-10 5.242e-11 5.227e-11
```

**Shocks.** The Sod crossing, root N = 201, `cfl = 0.4`, the artificial properties and the
filter live (their fluxes on the one-sided remainder), inviscid, at the ghost path's default
order 8; every ghost figure repeats the order-6 measurement to the printed digit but the
three-level global-step shell (7.0e-3) and the two-species excursions (3.1e-8, 4.1e-4):

| row | default | ghost |
|---|---|---|
| two levels subcycled: steps, p_min, shell, root | 373, 0.0405, 7.7e-3, 2.4e-2 | 373, 0.0405, 7.3e-3, 2.4e-2 |
| global dt: steps, p_min, shell | — | 708, 0.0382, 8.4e-3 |
| three levels: p_min, shell | — | 0.0405, 5.8e-3 |
| three levels global dt: p_min, shell | 0.0252, 5.9e-3 | 0.0251, 7.1e-3 |
| tile 8: p_min, shell | — | 0.0405, 8.3e-2 |
| three patches: p_min, shell, root | 0.0412, 7.4e-2, 2.7e-2 | 0.0412, 7.9e-2, 2.8e-2 |
| diaphragm on the shared plane | negative density, step 2 | completes: 303 steps, p_min 0.072, shell 1.0e-2, root 1.0e-1 |
| the same, art off | — | negative density, step 20 |
| two species: shell Y excursion, elsewhere | — | 2.8e-8, 3.5e-4 |

The reversed tube reproduces every printed digit of the forward ghost row. On the shared plane the
one-sided gradient workaround reads p_min 0.097, shell 4.6e-2 and root 2.4e-1 in the same run.

**Conservation.** The layer on (96, 24, 1) to t = 8, inviscid: the same-level layout's drift
is 6.2e-13 and its mass-fraction excursion zero (5.2e-10 and 3.1e-5 under the default rows),
and every nest and regrid history agrees with the default to the third digit. With
`mu=1e-3` on every layout, the uniform run included: the same-level drift is 5.3e-13 against
2.2e-10 under the closure rows, the nests drift 6.75e-5 to 6.92e-5 against 6.76e-5 to 6.93e-5,
the regrid histories sum to 2.24e-4 in both, and no layout leaves [0, 1].

**Cost.** Serial, one thread, median step over interleaved runs in one process, ghost over
default: (64, 48, 48) in two patches 1.18 inviscid and 1.38 with the artificial properties and
`mu = 1e-3`; a 2-D subcycled level of sixteen 37-node tiles 1.24 inviscid and 1.90 viscous. The
inviscid overhead is one padded pass per component and interface dimension. The molecular
part adds a pass, a halo exchange and a line solve per component and interface dimension,
and at a coarse-fine face the compact derivatives of every component along every dimension
on the interpolated box at each shell imposition, which a profile of the tiled level puts at
about a fifth of the step; a tile with no parent-fed face skips them. The `ghost_flux`
arrays add one state-sized array per interface dimension of a patch.

**Decision.** Same-level interfaces: the ghost fluxes remove the interface from the error
budget, inviscid and viscous (the interface window equals the interior at every
order-measuring row), cut the acoustic reflection forty-fold below the BL rows at N = 384, keep
the default's shock minimum pressure, conserve to round-off without a mass-fraction excursion,
and are the first treatment under which the diaphragm on a shared plane runs with the extended
gradients. Coarse–fine faces: at the ghost path's default order 8 it reads 5.8–7.0 on every
smooth row the time integrator leaves, viscous included, and matches or leads the BL rows
everywhere, so one treatment
serves both interface kinds. It stays experimental: the promotion to the default is ROADMAP
N15b, which names what remains.

### bench/levelfilter.jl: level-aware filtering under global stepping

N12a compares the production filter with two benchmark-only policies. The
driver explicitly loads `bench/level_filter_policy.jl` in a fresh process;
it does not install a solver option or change the package default. These
are Float64 CPU measurements on an Intel Core i9-12900K, Julia 1.11.4,
one Julia thread per process. The helper pays for an extra rate sweep and
collective; no performance conclusion is drawn from it.

For patch level `l`, the normalized policy forms the globally reduced
directional envelope `q_d = max(r_patch,d / 3^l)` and uses `3^l q_d` in
that patch's existing relaxed filter weight. The CFL rate that sizes the
step is unchanged. The reduction includes all ranks, with zeros from
nonowners, and is rebuilt at every rate estimate, including the first
estimate after regridding. This is the same conservative envelope used by
subcycling, expressed in each patch's physical time units.

The cadence comparator uses that envelope but filters level `l` only
every `3^(L-1-l)` global steps, multiplying its weight by the same stride;
`L` is the number of levels. It is restricted to `filter_interval=1`.
It multiplies the latest timestep rather than accumulating elapsed time,
so shortened steps and an unfinished final stride are not a general
physical-cadence prescription. Both trials delegate subcycling and
`filter_cfl=0` to production unchanged. The helper also provides an
actual-per-level envelope for diagnostic checks; that variant is outside
the qualification matrix below.

**Two-transit layer and composite budgets.** The conservation instrument
runs the bound-only two-species layer on `(96,24,1)` to
`t=50.26548245743669`, with sixteen equally spaced observations, CFL 0.45
and permissive validity. Each drift is relative to that layout's own
initialized composite quadrature. The evolution budget remains 1e-3 for
mass, each species mass, energy and momentum scaled by initial mass times
sound speed; it was not adjusted for these candidates.

| policy, global stepping | levels | steps | maximum sampled conserved drift | maximum species excursion |
|---|---|---|---|---|
| production | 2 | 3532 | 6.951e-5 | 0 |
| normalized | 2 | 3532 | 6.948e-5 | 0 |
| cadence | 2 | 3532 | 6.943e-5 | 0 |
| production | 3 | 10570 | 7.567e-5 | 5.82e-4 |
| normalized | 3 | 10568 | 7.578e-5 | 0 |
| cadence | 3 | 10568 | 7.573e-5 | 0 |

Both candidates remove the sampled late undershoot while remaining well
inside the conservation allowance. Subcycling remains positive at both depths
and its production maximum drifts are 6.943e-5 and 7.574e-5. The agreement
of the two candidate layer outcomes does not make their operators
equivalent: one applies a weak pass at every fine-sized step, the other
interleaves fewer stronger passes with shell imposition.

Reproduce with `julia --project=. -t 1 bench/levelfilter.jl
policy=normalized instrument=budgets N=96 ny=24
tfinal=50.26548245743669 moving_tfinal=50.26548245743669 samples=16
check=true`. Use `parts=mixing layouts=uniform,depth3 stepping=global`
to isolate the long three-level rank comparison while retaining its
uniform attribution baseline.

**Moving refinement.** The sharper moving layer uses the same grid and
end time, with sixteen explicitly sampled regrids of a two-level nest.
The table separates evolution from the immediate transfer jumps, using
the existing 1e-3 allowance for each reported conserved-budget measure.

| policy, global stepping | sum of absolute transfer jumps | maximum total drift | maximum drift with transfers subtracted | species excursion |
|---|---|---|---|---|
| production | 8.423e-4 | 1.002e-4 | 2.136e-4 | 7.42e-4 |
| normalized | 8.423e-4 | 1.006e-4 | 2.200e-4 | 6.76e-4 |
| cadence | 8.422e-4 | 1.009e-4 | 2.213e-4 | 5.53e-4 |

Every conservation measure passes, but moving refinement still has a
species excursion beyond the validity dead band under either trial.
The unchanged subcycled control has cumulative jumps 8.422e-4, maximum
total drift 1.009e-4, transfer-subtracted drift 2.213e-4 and species
excursion 5.56e-4. These policies therefore resolve the fixed-layer
reproducer, not general species positivity. Three-level moving
refinement is unsupported and is not measured.

**Smooth evolution and reflection.** The shared entropy-wave case runs to
`t=0.5` at root `N=48,96,192`, CFL 0.25 and 0.125, with the filter on.
The interface maximum counts actual patch faces, not MPI block ends;
the L2 norm uses the solver's masked composite quadrature. The following
orders fit all three spacings at CFL 0.25.

| policy, global stepping | levels | interface order | L2 order | L2 error, N=192 |
|---|---|---|---|---|
| production | 2 | 4.120 | 4.334 | 3.162e-9 |
| normalized | 2 | 4.100 | 4.116 | 3.523e-9 |
| cadence | 2 | 4.106 | 4.120 | 3.501e-9 |
| production | 3 | 4.416 | 5.126 | 2.177e-9 |
| normalized | 3 | 4.101 | 4.112 | 3.569e-9 |
| cadence | 3 | 4.111 | 4.121 | 3.489e-9 |

Both trials bring the two depths close to the subcycled order, about
4.09 at the interface and 4.11 in L2. They increase the finest-grid
global-step L2 error by about 11% at two levels and 60--64% at three
levels. Thus the long-layer improvement does not mean every smooth error
is smaller. Across both stepping modes the largest relative change on
halving CFL is 1.24% (production), 1.04% (normalized) and 1.62% (cadence),
below the accuracy instrument's 10% temporal-contamination criterion.
These measured orders do not identify an individual operator mode or
establish sixth-order AMR.

The filtered acoustic pulse crosses both faces of the nest at root
`N=192`, measured at `t=pi/sqrt(1.4)` against a filtered uniform run.
In serial, the maximum of the upstream pressure-wake and leftgoing-characteristic
errors, divided by pulse amplitude, is 3.26e-9 for production, 1.53e-9
for normalized rates and 1.46e-9 for cadence, across two/three levels
and both stepping modes. All pass the pre-existing 1% reflection bound.
Subcycled accuracy and reflection rows are unchanged by the trial hooks.
Reproduce with `bench/levelfilter.jl policy=normalized instrument=accuracy
parts=all ns=48,96,192` after `julia --project=. -t 1`.

**Shock crossings.** The sensor instrument's Sod shock enters and exits
the refinement box by `t=0.2`, root `N=201`, with a uniform fine-spacing
reference. All listed runs completed with zero inadmissible steps. The
minimum density and pressure include startup; the disturbance is the
momentum ahead of the shock at `t=0.1`, `x>0.85`.

| policy, global stepping | levels | minimum density | minimum pressure | disturbance ahead of shock |
|---|---|---|---|---|
| production | 2 | 0.06239 | 0.03822 | 3.543e-10 |
| normalized | 2 | 0.07956 | 0.05828 | 1.216e-10 |
| cadence | 2 | 0.07670 | 0.05181 | 2.210e-10 |
| production | 3 | 0.05121 | 0.02518 | 4.446e-9 |
| normalized | 3 | 0.08174 | 0.06098 | 5.048e-11 |
| cadence | 3 | 0.07740 | 0.05315 | 2.403e-10 |

The two-species global-step crossing also stays admissible under every
policy. Its final mass-fraction excursion at the contact changes from
4.951e-4 (production) to 3.990e-4 (normalized) and 3.909e-4 (cadence);
the corresponding shell excursions are 6.953e-8, 1.328e-8 and 6.283e-9.
The subcycled two- and three-level rows reproduce production under both
trials. This tests passage into and out of a refined region; it does not
remove the separate discontinuity-on-a-same-level-plane limitation.

Reproduce the crossing rows with
`julia --project=. -t 1 bench/levelfilter.jl policy=normalized
instrument=sensors parts=crossing N=201
"crossing_variants=base,global,three levels,two species"`, substituting
`policy=default` or `policy=cadence` for the controls.

**Rank coverage.** The normalized three-level global-step layer was run
over both transits at 1, 2, 4 and 8 ranks. Every run took 10568 steps,
with maximum sampled conserved drift 7.578e-5, zero species excursion,
and matching printed width and molecular-mixing histories. The
conservation, initial-sampling and mixing-comparison checks all pass.

All three policies' smooth and reflection matrices were repeated at four
and eight ranks, with the normalized matrix also repeated at two ranks.
Eight ranks use `ns=96,192`; the 48-node root cannot supply the filter's
nine nodes per rank. Over common resolutions, the largest absolute
difference from serial in any reported smooth error norm is below
3.6e-14. Every reflection row passes. The standalone
`bench/level_filter_policy_check.jl` checks default delegation, shortened
steps, skipped and due cadence passes, subcycle trajectory equality, and
propagation of a fastest rate present only on deep-level owners to root
ranks that own no deep tile.

The policy checks pass at 1, 2, 4 and 8 ranks. Normalized-policy moving
regrid smoke runs pass at four and eight ranks in both stepping modes;
the eight-rank global shock crossings reproduce the serial diagnostics
at the printed precision, with zero inadmissible steps. The full moving
two-transit comparison above was measured in serial.

**Decision.** Retain the production default and subcycling as the supported
workaround. The normalized envelope is a viable benchmark-only candidate
for the fixed Cartesian layer: it removes that undershoot and preserves
the declared conservation and reflection budgets, but raises the finest-grid
error in the measured smooth case and does not ensure species positivity after
moving refinement. The cadence comparator offers no demonstrated general
advantage over rate normalization and needs elapsed-time and phase handling
before it could become a production policy. This qualification is bounded
to the tested Float64 CPU cases; it does not identify an operator mode or
promote a general filter-interval multiplier.

### bench/substeprates.jl: refreshed refined-level rates

`bench/substeprates.jl` measures the rates after the refined RK right-hand
sides refresh the artificial coefficients, using the production subcycling
driver. The root estimate is taken after startup priming or regrid priming,
and the state filter uses the production cadence and relaxation.

Reproduction: `julia --project=. -t 1 bench/substeprates.jl steps=3
levels=3,4 regrid_steps=6`. Serial Float64 on an Intel Core i9-12900K,
Julia 1.11.4, one Julia thread: 201 root nodes, a Sod jump at x = 0.69
inside the deepest nested box, root CFL 0.2. Both startup maxima occur
in the first root step:

| levels | peak refreshed substep CFL | divided by root CFL target | refined level | substep | RK stage |
|---|---|---|---|---|---|
| 3 | 1.607612 | 8.03806 | 1 | 3 | 5 |
| 4 | 1.704789 | 8.52395 | 1 | 2 | 1 |

The location is the level with the largest observed rate, not necessarily
the finest level. These are rate-growth measurements, not stability or
solution-accuracy certificates.

The supported two-level moving-box case changes layout four times in six
root steps. Its largest changed-layout CFL is 0.188657 (0.943283 times the
root target), on root step 2, refined level 1, substep 6, RK stage 1,
after the box changes from offset/extent (50, 31) to (87, 30). The largest
unchanged-layout value is 0.198363. These readings include the new regrid
coefficient refresh and therefore measure its resulting trajectory.

The original proposed absolute ceiling of 1 was rejected: the existing
subcycled Sod precompile case at root CFL 0.2 reached a refreshed refined
stage CFL of 1.011060 on its first step. The rate used to choose a step and
the largest rate within its RK trajectory are different quantities; that
observation alone does not establish a stability ceiling. The refreshed-rate
check therefore remains opt-in (`StepControl.substep_cfl = 0` by default),
with a positive value interpreted as an absolute ceiling qualified for the
case. A violation returns through the level communicators to the root's
savepoint/retry path before any accepted-step clock or callback advances.

Every regrid check now refreshes artificial coefficients before the next
root estimate, including checks that keep the layout. This makes the refresh
cadence independent of timing-based ownership moves; a changed layout saves
the refreshed coefficients in its retry state. This applies to both direct
`regrid!` calls and the run-loop hook, for box and tiled refinement. Dynamic
regridding below the first refined level remains unsupported; the deeper
startup cases do not qualify that missing capability.

The `test/mpi_tests.jl` moving-Sod references were remeasured with this
refresh cadence, keeping the existing round-off tolerances. Serial and
two-rank runs agree to round-off; the changed times follow from using fresh
coefficients in the root CFL estimate, rather than the previous stale arrays.
All cases use 400 root nodes, root CFL 0.2 and the unrelaxed filter.

| regression fixture | completed steps | serial time | final offset/extent or tile offsets |
|---|---|---|---|
| distributed regrid | 61 | 0.005505030961213709 | 159 / 43 |
| level rank subsets | 41 | 0.0055135979946853665 | 170 / 25 |
| tiled regrid | 41 | 0.003194481761204692 | 168, 176, 184 |
| stored ownership / rebalance | 41 | 0.003194481019080914 | 168, 176, 184 |

**Cost and dense-output decision.** Two warm whole runs precede five pairs
of three-step smooth runs, with the guard disabled and enabled on identical
initial data. The table reports the minimum time in each group; differences
between the two depth-dependent percentages are not resolved against
workstation timing noise.

| levels | guard off, ms/root step | guard on, ms/root step | scan overhead | estimated endpoint share of RHS time |
|---|---|---|---|---|
| 3 | 2.472 | 2.738 | 10.7% | 5.48% |
| 4 | 9.152 | 9.923 | 8.4% | 4.63% |

The endpoint estimate weights measured per-level RHS costs by the recursive
evaluation counts. Three levels use 65 stage RHS evaluations and 4 endpoint
evaluations; four use 200 and 13. Minimum warm RHS costs from five samples
are (39.8, 18.6, 26.0) μs and (32.3, 19.2, 27.0, 38.5) μs respectively.
This is an estimate for the measured serial layouts, excluding endpoint box
gathers; it is neither an equal-cost count nor a distributed speedup bound.
Retain cubic Hermite output: removing this small RHS share does not justify
a new temporal reconstruction and its accuracy qualification. Revisit the
choice if a measured layout makes parent endpoint work dominant.

### bench/amr_transfer.jl: the 3:1 transfer pair

`julia --project=. bench/amr_transfer.jl` (`7dbf319`; banded schemes
`8271bd7`).

The Pyranda pair (invertible compact filter, Gaussian of width 3Δx) bound to
`plan_direction` as a `CompactScheme` (restriction) and a
`BandedCompactScheme` (prolongation), the 3:1 sampling convention pinned
numerically because the public kernels do not specify it.

| quantity | measured |
|---|---|
| pair round-trip, closed / periodic | 1.6e-15 / 2.7e-15 |
| restriction as left inverse (coarse → fine → coarse) | 8.9e-16 |
| fine → coarse → fine order, Lagrange 4 / 6 / 8 | 3.97 / 5.93 / 7.97 |
| constant under restriction / prolongation | last bit / 13 ULP |
| round-trip order within 6 points of a closed end | 3 (the closure order) |
| plans against parity-extended full lines, both schemes, both signs | ≤ 1e-13 |
| prolongation gain at the fine Nyquist | 20.24 |
| closure condition number | 33 |

Conditioning in situ, on a 2h captured shock: the smoothed δ⁴ sensor
round-trips at 1.03–1.13, the state undershoots ≤ 3% of ambient at the shock,
and pollution decays ≈ 3.4× per point away from it. Settled the default
coupling's tolerance of captured features and sized `tag_buffer = 4`.

**Closure localization.** The response of the compact solve to a unit error in
the first ghost layer decays into the patch at exactly the root of the LHS
symbol.

| scheme | rate per point | response at 8 fine points | at 12 points |
|---|---|---|---|
| C6 | 0.382 | — | five orders down |
| C8 | 0.451 | — | — |
| C10 | 0.556 (1.8× per point) | 5e-3 | 5e-4 |

Settled: the default 4-coarse-cell tagging buffer holds two orders at C10
where it holds five at C6, which is enough; the buffers did not move.

### bench/leveltransfer.jl: the live interpolation order

`julia --project=. -t 1 bench/leveltransfer.jl orders=2,4,6,8,10` (about a
minute).

The live coupling by itself, on analytic data with the root exact: the
`level_interpolation_order` Lagrange interpolation that fills a refined
patch's shell and a freshly created fine region, the restriction, and
repeated regrids. 2-D fields are periodic on [0, 2π)² with a square level
fixed at 5L/12..7L/12; the smooth field is 1 + 0.2 sin(3x + 0.37) cos(2y − 0.1);
N is the root count.

**Values.** A tensor polynomial of degree p − 1 per dimension is reproduced
in the shell and in a fresh fill to 1e-14 of its maximum (degree p: 7.5e-3,
3.4e-4, 4.7e-5, 1.6e-5, 1.3e-5 at p = 2, 4, 6, 8, 10). On the smooth field:

| order | ghosts, N = 48 / 96 / 192 | boundary planes, N = 192 | fresh fill, N = 192 | order |
|---|---|---|---|---|
| 2 | 4.86e-3 / 1.23e-3 / 3.01e-4 | 1.25e-4 | 3.09e-4 | 2.0 |
| 4 | 1.15e-4 / 7.28e-6 / 4.46e-7 | 2.23e-7 | 4.57e-7 | 4.0 |
| 6 | 3.28e-6 / 5.27e-8 / 8.09e-10 | 4.46e-10 | 8.30e-10 | 6.0 |
| 8 | 1.18e-7 / 4.66e-10 / 1.80e-12 | 9.37e-13 | 1.80e-12 | 8.0 |
| 10 | 6.70e-9 / 6.68e-12 / 6.66e-15 | 2.11e-15 | 6.66e-15 | 10.0 |

At order 8 the outermost ghost layer's stencil sits one node inward of
centred (`LEVEL_BUFFER = 4`), and at order 10 the outer two sit two and one
nodes inward; the ghost column reads the order regardless.

**Derivatives.** `interp` is the fine patch's operator on the imposed shell
minus the same operator on exact shell data, the transfer's own contribution;
`total` is against the exact derivative. The first derivative goes through
the gradient plans, whose extended interface rows read the ghosts; the
second applies the divergence plans (one-sided at the interface end, the
viscous flux path) to that first derivative, along x (xx) and along y (xy).
Windows are the four fine nodes nearest an end along the differentiated
dimension; N = 192, orders over 48 / 96 / 192:

| order | 1st interp | 1st interp on a plane | xx interp | xy interp | xx total | fill, 1st interp |
|---|---|---|---|---|---|---|
| 2 | 3.66e-3 (1.0) | 1.62e-2 (1.0) | 0.61 (0.0) | 0.50 (0.0) | 0.61 | 2.21e-2 (1.0) |
| 4 | 6.54e-6 (3.0) | 2.39e-5 (3.0) | 3.77e-4 (2.0) | 6.52e-4 (2.0) | 3.76e-4 | 3.94e-5 (3.0) |
| 6 | 1.31e-8 (5.0) | 4.59e-8 (5.0) | 7.53e-7 (4.0) | 1.22e-6 (4.0) | 1.78e-6 (3.4) | 7.86e-8 (5.0) |
| 8 | 2.75e-11 (7.0) | 9.65e-11 (7.0) | 1.58e-9 (6.0) | 2.56e-9 (6.0) | 1.29e-6 (3.1) | 1.65e-10 (7.0) |
| 10 | 1.06e-13 (9.0, 8.4) | 2.27e-13 (9.0) | 8.01e-12 (8.0, 7.0) | 5.94e-12 (7.9) | 1.29e-6 (3.1) | 3.93e-13 (8.9) |

An O(h^p) value error enters the first derivative at h^(p−1) and the second
at h^(p−2), each at the fine spacing, as the counting predicts, the order-10
rows reaching round-off by N = 192; a fresh fill
carries the first-derivative loss over the whole patch, not only at its
edge. The second-derivative total is the divergence's own closure error at
order 8, while at order 6 the transfer term is 64% of the total at N = 96
and 42% at N = 192, so under the default coupling the viscous path is where
the order-6 transfer is visible.

**Restriction.** Injection writes the fine coincident values onto the
covered root nodes exactly (20, 157 and 814 written nodes at N = 48, 96,
192; difference at round-off). The filtered restriction on point samples
errs by 1.68e-3, 4.63e-4, 1.16e-4, order 1.9 / 2.0: it is not a
point-sample transfer.

**Repeated regrids.** A 1-D level moved ±N/24 root nodes sixteen times with
no evolution, restricted after each: the fine error after the last regrid is
1.09 to 1.16 times the error after the first at every order and N, and the
order of both is the interpolation order (2.0, 4.0, 6.0, 8.0, 9.9; 6.98e-10 and
7.63e-10 at order 6, N = 192). Regrids do not accumulate transfer error.

**Positivity.** A fresh fill of the step 1 → 0.125 of width w root cells:
at w = 0.5 the fill undershoots the low state by 1.2%, 2.3%, 2.9% and 3.2%
of the jump at orders 4, 6, 8 and 10 (overshoot 0.8%, 0.4%, 0.03%, 0.05%);
order 2 is monotone, and at w = 1 and 2 no order leaves the range. The
moving-region Sod gate of `test/level_tests.jl` (N = 201, subcycled, regrid
every five steps) is order-independent: composite density error 8.7e-4,
8.4e-4, 9.5e-4, 8.9e-4, 9.2e-4 at orders 2, 4, 6, 8, 10, minimum density
0.1243–0.1248 and minimum pressure 0.0992–0.0995, all 628 steps.

**Budgets.** `bench/interfaceconservation.jl` at `interpolation_order=8`,
serial, `layouts=uniform,depth2`, otherwise the command of that section:
two levels drift 6.95e-5 (global) and 6.94e-5 (subcycled); the sixteen
regrids sum to 8.41e-4 in absolute jumps, sampled drift with transfers
1.0e-4, transfers subtracted 2.1e-4 / 2.2e-4, final 1.7e-5 / 2.5e-5,
species excursion 7.4e-4 / 5.6e-4. Every figure matches order 6 to the
digits that section prints. At `interpolation_order=10`, measured later on
the same command: two levels drift 6.88e-5 (global) and 6.89e-5
(subcycled); the eight regrids of that run sum to 2.20e-4 in absolute jumps,
transfers subtracted 1.20e-4, final 3.8e-6 / 4.5e-6, and no species leaves
[0, 1]; every figure is inside its budget.

Settled: the live order is a choice among 2, 4, 6, 8 and 10, by default the
derivative operator's interior order, two more under `interface_flux =
:ghost`. Orders 8 and 10 fit the default halo and buffer, the off-centre
outer stencils included, cost nothing measurable in conservation, positivity
on resolved data or regrid drift, and remove the transfer from the
second-derivative and filtered error budgets; order 2 is the only monotone
choice.

### test/patch_tests.jl: same-level patch interfaces

`julia --project=. test/runtests.jl` (the patch testsets; `8eda454`, C10 rows
`8271bd7` and `433f2ef`).

Two conforming patches, manufactured smooth solutions across the interface,
since the bit-exact oracle does not survive it.

| gate | C6 | C10 |
|---|---|---|
| entropy-wave order across the interface | 3.09 / 3.53 | 3.06 / 3.52 |
| entropy-wave error, 48 / 96 / 192 | — | 8.2e-7, 9.8e-8, 8.5e-9 |
| acoustic pulse reflected amplitude at 192 | 2.3e-3 | 4.1e-3 (7.5e-2 at 96, 7.7e-5 at 384) |
| pulse reflection order | ≈ 5 | — |
| viscous wave through the interface rows, `:extended` | 4.3 / 4.0 | 3.93 / 3.92 |
| viscous errors at 48 / 96 / 192 | 6.1e-5, 3.0e-6, 1.9e-7 | 6.0e-5, 4.0e-6, 2.6e-7 |
| the same with `:onesided` | 2.1 / 1.7 (8.4e-5, 2.0e-5, 5.9e-6) | 2.00 / 1.67 |
| degree-9 polynomial through both closed ends | — | 2e-12, against 1e-4 for the cascade rows |
| conservation drift, long periodic run | 1.2e-8 relative (single patch 4.5e-15) | — |

Rank partitioning reproduces the serial two-patch answer bitwise at one rank
per patch; once a patch itself decomposes, agreement is round-off, 3.1e-15 at
np = 4 against a 9.5e-8 signal, with identical step counts.

Settled: the divergence's one-sided rows bind the inviscid orders, so
`interface_rhs = :extended` is measurable only through the gradients, where it
is worth two orders; C10 interface rows are the default halo's, no wider.

### test/level_tests.jl: the level hierarchy

`julia --project=. test/runtests.jl` (the level testsets); MPI legs through
`test/mpi_tests.jl`. Commits `772e304` and `e215c5a`, tiles `eb0cb38`,
covered masks `8d32ab3`, checkpoint `4b42724`.

**Coupling choice.** The invertible pair as the live coupling measures order
1.3–1.7 on a manufactured solution, since prolongation's input must be samples
of the filtered field and the live coarse solution is not. Order-6 Lagrange up
with coincident-node injection down measures 3.46 / 3.64 at errors three
decades lower, and is the default.

**Buffers.** Restricting to the fine boundary closes an amplifying loop
through the imposed shell, gain ≈ 2 per step; `RESTRICT_MARGIN = 2` makes the
growth flat. The C10 sweep over `RESTRICT_MARGIN` 0–3 with `LEVEL_BUFFER` 4
and 6, on the two-level entropy wave and the subcycled Sod crossing:

| margin | 0 | 1 | 2 | 3 |
|---|---|---|---|---|
| wave error at 96, C10 | 5.1e-9 | 5.1e-9 | 5.7e-9 | 5.4e-9 |
| wave error at 96, C6 | 5.0e-9 | 5.0e-9 | 7.7e-9 | 5.8e-9 |
| error growth t = 0.5 → 2, C10 | 2.2 | 2.2 | 3.3 | 1.9 |
| error growth t = 0.5 → 2, C6 | 1.7 | 1.7 | 2.5 | 1.6 |

Sod noise stays between 3.5e-10 and 8.8e-10 throughout, and `LEVEL_BUFFER = 6`
reproduces the buffer-4 numbers to the last digit at both schemes. Settled:
the constants stay at 4 and 2.

**Subcycling.** Orders are unchanged at a third of the steps, and the
subcycled Sod gate improves on the global-dt one: ahead-of-shock noise
5.7e-11 against 6.4e-10, mass drift 9.8e-5 against 1.36e-4. At C10 the
two-level wave measures 3.84 / 3.54 (8.2e-8, 5.7e-9, 4.9e-10; subcycled
3.88 / 3.53) and the Sod crossing leaves 5.4e-10 of momentum noise ahead of
the shock (subcycled 4.0e-10; C6 6.4e-10 and 1.3e-10).

**Distributed coupling.** Replicating the interpolation chain per rank put the
3-D cost case at 85% of the uniform-fine wall; distributing the chains by
conserved component and sharing the shell ring through one Allgatherv brought
it to 49%, serial results bit-identical.

**Tiles.** A tiled level costs nothing visible against the one-patch level:
1-D entropy wave at tile 8, 6.0e-10 against 6.2e-10 at N = 192, orders
3.95 / 3.91; a 2×2 tile nest in 2-D, corner included, 4.29e-8 against 4.27e-8.
A flat pairwise pass over a node shared by four tiles with copies 1, 2, 3, 4
ends at 2.23, 2.68, 2.41, 2.68 instead of the mean, which is why
`_sync_level_records!` is dimension-phased.

**Moving regions.** Sod at N = 201 coarse against a 601-node uniform-fine
reference: composite density error 2.8e-3 where uniform-coarse gives 7.3e-2,
26× better, with fine resolution over a third of the domain. Shu–Osher: 10×
better in L∞ over the wave train, 6.7× in L1, at 2497 coarse steps against the
reference's 4662. The tiled regrid reproduces the Sod gate.

**Composite diagnostics.** Taylor–Green at 24³ with an 8³ region refined
off-centre, 61 steps: the masked composite energy history stays within 2.5e-4
of the single-level history, the fine sampling's own quadrature difference,
where the unmasked sum sits 1.4e-2 above it. Settled the per-orthant covered
mask against a per-dimension factor.

**Restart.** The tiled and box Sod regrid cases checkpointed at step 23 and
continued to step 130 agree with the uninterrupted run at every slot, tag
history included, serially and (the tiled case at 400 nodes, step 21 → 41) at
np = 2, 4 and 8. A twelve-tile wave written on half the ranks and restored on
all of them, rebuilt from a six-tile initial region, continues to 1e-12 in the
wave error.

**Reproducibility tier.** A tile owned by a proper subset reproduces the
every-rank answer to round-off, not bitwise: 0 to 6e-15 on the tiled wave
cases at np = 2, 4 and 8, and the tiled Sod regrid with rebalancing on reaches
the serial time to 5e-18. A decomposed patch sits at 1e-15 to 6e-15.

### bench/amr_tiles.jl: tiled cover and per-tile cost

`julia --project=. bench/amr_tiles.jl [N=192] [tile=6] [steps=3]` (`eb0cb38`;
workspace pooling `41d212f`, patch-type compile `556d69a`).

Annular tag set, N = 192, r0 = 0.75, w = 0.02, buffer 1:

| tile edge | cover of the bounding box | tiles | fine nodes each | memory each |
|---|---|---|---|---|
| 6 | 41% | 208 | 19² | 0.40 MB |
| 12 | 47% | 80 | 37² | 1.1 MB |

Settled: the lattice cover is worth having, and the ratio improves with a
thinner shell, which is the implosion argument for tiles over one box.

Per-tile cost: setup is 0.06–0.12 s per tile in plan construction (N = 96: 64
tiles in 7.6 s, 196 in 11.4 s), which argues for tile edges of 12 or more in
3-D. Warm construction of plans, scratch, transfers and communicators is
0.03 s for eight tiles. Native code compiles once per distinct `Patch` type at
1.8 s on the CPU backend and 3.9 s on the device backend, and `promote_typeof`
specialization cost 0.5–1.7 s per distinct patch count, which is why every
refined face carries `InterfaceBC` and `solver.patches` is a typed vector.

Shared RHS workspace at tile 6 (208 tiles of 19² plus the root): 103.0 MB over
the patch set before the pooling, 60.3 MB after, a factor of 1.71, with
0.207 MB of each tile's 0.398 MB shared. The warm step is 0.55 s before and
0.56 s after at one rank on 16 threads, inside the run-to-run spread.

Workstation pathology: any 2-D case at np = 8 runs at ~7 s/step, one patch or
four tiles alike, against ~0.5 s at np = 4, of the kind `CLUSTER.md` records
for hybrid cores; the MPI suite's tiled check is bounded to ten steps for it.

Open: a rough count puts the RHS work of 40 tiles of 37² on a 96² root near
2 s per subcycled step, against one cold measurement of 8–10 s; the warm
annular reading at a different configuration matches expectation. Remeasure
warm at the original configuration.

### bench/amr_cost.jl: the mixing cost case

`mpiexec -n 8 julia --project=. -t 1 bench/amr_cost.jl 48 1.0` (`6382d8e`;
masked quadrature and tag criteria `f8e1a23`).

A heavy-gas blob mixing case on a 48³ root grid with a subcycled, regridding
region covering a sixth of the volume, against uniform 48³ and 142³ references
at t = 1. The metric is ∫Y(1−Y)dV on the shared coarse lattice.

| configuration | mixedness error | wall | memory |
|---|---|---|---|
| coarse 48³ | 6.9e-3 | 18 s | 179 MiB |
| composite, δ⁴ρ tag | 1.5e-3 (4.6× closer) | 204 s (43% of fine) | 655 MiB (24% of fine) |
| composite, sensor tag (`tag=sensor sensor=0.02`) | 1.7e-3 | 35% of fine | — |
| fine 142³ | reference | 471 s | 2737 MiB |

Single runs at np = 8; read the ratios, not the third digit. Settled: the
composite buys most of the fine answer at a third to a half of its cost, and
the artificial diffusivity number alone tracks the blob to the same final
region (nothing above 0.02 farther than 0.6 from the interface on the coarse
grid; a captured Sod shock reads about 2 under the default C_β).

Dead end: pointwise in-region error is the wrong metric. Coarse and composite
both sit at max ≈ 0.19 against fine there, the sub-cell displacement of a
near-discontinuous interface.

### bench/amr_balance.jl: rebalance and migration mechanics

`mpiexec -n 4 julia --project=. -t 1 bench/amr_balance.jl [N=800] [interval=10]
[steps=200] [persist=1]` (`603af33`).

A 1-D Sod shock crossing a tiled refined level, np = 4, N = 800. With
rebalancing off the stored groups drift to two tiles on ranks 0–1 against four
on 2–3 and the per-check max/mean busy time climbs to 1.3–1.5. With it on at
threshold 1 and persist 1, every check repartitions, one tile per rank where
the count allows, and the partition then follows the timing noise from check
to check (1.0–1.7). Settled: mechanics only, and the reason `rebalance` and
`rebalance_persist` exist; the workstation cannot say what a rebalance is
worth, since per-rank costs on rzhound and rzadams differ from it by 27–66×
and move with rank placement. A moved tile here is one kilobyte.

`MIGRATION_AUDIT` holds the migrated state bitwise against the replicated
carry it replaced, at zero differing slots, in the MPI suite.

### bench/device_solver.jl: the device battery

`julia --project=<env-with-AMDGPU> -t 8 bench/device_solver.jl backend=amdgpu`.
RX 6800 XT, AMDGPU.jl on Windows/HIP; residency and stacked storage `602002c`
and `346f83e`. Workstation numbers are evidence that the structural pitfalls
are gone, not a performance claim about the target machine: RDNA2 runs vector
FP64 at 1/16 the FP32 rate where MI300A runs it at full rate.

| measurement | value |
|---|---|
| 64³ TGV full step, device F64 / F32 | 0.146 / 0.117 s per step |
| the same on the 8-thread CPU | ~0.12 / ~0.10 s per step |
| isolated flux assembly, 64³ two-species | 9.9× the CPU |
| staged halo and pair copies | 0.6–6.6% of device wall |
| reduced-interface copies | 2–5% of device wall |
| first-launch kernel compilation | ~9 s per body |
| KA-CPU against `@threaded` at 64³ | 2.8× (flux assembly) to 40–50× (RK update) slower |
| removing the per-launch synchronize | 28–32% off the device step |

Settled: the CPU keeps `@threaded`; the one unconditional fence is the
reduced-solve one; the device floor is launch submission, not arithmetic.

**Stacked tile storage**, warm steps, one run each, the 8-thread CPU as
reference:

| case | before stacking | after |
|---|---|---|
| 1-D tiled regridding Sod, 8 tiles of 25 fine nodes, subcycled | 0.11 s (CPU 1.1 ms) | 0.049 s |
| two-slab viscous wave | 9 ms (CPU 0.1 ms) | 9.8 ms (root level is not stacked) |
| 3-D level, twelve 16³ tiles in one stack, subcycled | — | 0.354 s (CPU 0.576 s) |
| unstacked 64³ TGV | 0.146 s | 0.164–0.176 s |

Settled: the twelve-tile 3-D level is the first tiled configuration on which
the device leads. The residual floor is the work that stays per tile: the
shell impositions, the interface records, and `max_rate`'s two reductions per
tile. Whether the fill and scatter kernels' fourth index dimension of extent
one costs anything on the unstacked TGV is at the edge of the run-to-run
spread and needs a repeated-process measurement.

**Precision.** Uniform Float32 on the CPU at 64³ TGV, t = 10: identical peak
dissipation to the printed precision, 2.00× smaller footprint, 1.10× wall, and
mean-density drift 1.4e-4 against 7.5e-13. The drift is why Float32 is not the
default. On device Float32 runs at 1.25× the Float64 rate, because the step is
bounded by launch submission and fences rather than arithmetic.

### bench/device_mpi.jl: distributed device runs

`mpiexec -n 2 julia --project=<env-with-AMDGPU> bench/device_mpi.jl
backend=amdgpu` (repeat at 4 and 8; `6f14a0c`). Every distributed device stage
was accepted on `max |device − cpu| = 0` over full runs at np = 2, 4 and 8, in
both launch-policy modes. The staged transfer volumes are the halo and
reduced-interface percentages recorded above.

### probes/device_floors.jl: rzadams floors and the wait stall

`julia --project=<env> probes/device_floors.jl`. rzadams MI300A, ROCm 6.4.3,
logs in `bench/logs/rzadams_20260819*.txt`.

| measurement | value |
|---|---|
| kernel submission | 10 µs |
| launch + synchronize round trip | 25 µs |
| line solves | 0.14–0.40 ms per apply |
| 64³ TGV over 4 APUs | 0.074–0.088 s per step |
| F64 / F32 whole-step ratio | ~1.2 |
| 256³ single-species TGV over 4 APUs | 0.35 s per step baseline; 24,490 steps to t = 10 in 3.69 h |

Open: an intermittent stall mode in which every device wait costs an integer
number of milliseconds (13.000 ms medians) for seconds to beyond 30 s. It sits
below the Julia layer and inflated the 256³ run's solver average to 0.52 s per
step even at `-t 1`. `reference/bugreports/rocm_wait_stall_report.md` has the
characterization. Until it is resolved, run device-resident rzadams jobs at
`-t 1` per rank; a wall number from a multithreaded process is untrustworthy
without a stall watch beside it.

### Dead ends and one-line lessons

- An explicit Gaussian pass ahead of restriction (Pyranda's `c4ff3` role):
  cuts the shock undershoot 2.8× but raises the total round-trip error. Kept
  as a tool, not a default.
- A `::Type` argument through the `pointwise!` launcher: 9× as silent
  per-point runtime dispatch.
- Holding the `FieldVector` tuple form on the host: 3× on `assemble_fluxes!`.
- A keyed, `Any`-typed staging-buffer cache: 33 dispatch sites in
  `compute_rhs!`'s jetcheck report without ever executing. Reverted.
- A kernel-argument tuple longer than 32 elements: `InvalidIRError` on device.
- KA-CPU equality cannot certify the device path: four defects passed every
  KA-CPU test and failed only on real device storage.
- Judge a cost case on the quantity the refinement predicts, not on pointwise
  in-region error.

## Temperature-dependent transport

```text
julia --project=. -t 1 test/transport_tests.jl
julia --project=. -t 1 test/transport_integration_tests.jl
julia --project=. -t 1 bench/transport.jl
```

N8 retains `Transport()` as the default and adds `CeaTransport(eos)` as an
opt-in dimensional gas model. The bundled table supplies 66 pure-species
viscosity/conductivity records and 41 binary viscosity-interaction records,
but no binary diffusion coefficients. Unity-Lewis diffusion is therefore the
CEA model's default; mixture-averaged mass diffusion requires explicitly
supplied `BinaryDiffusion` data. The mixture rules and units are documented in
[thermodynamics](../docs/src/explanation/thermodynamics.md).

The coefficient checks evaluate independent literal CEA rows in SI units,
test the pure/trace and species-permutation limits, and distinguish the
mass-gradient diffusion convention from the mole-gradient convention using
unequal ternary coefficients. The evolution cases use a periodic Fourier
profile with analytic time-dependent forcing, no filter or artificial
transport, and nonlinear temperature-dependent conductivity or
composition-dependent corrected species diffusivity. These are manufactured
solution checks of the implementation, not experimental calibration of the
gas model or its binary diffusion inputs.

On Julia 1.11.4, Windows x86-64, with `cfl=0.15` and final time `0.01`,
the maximum solution errors are:

| Manufactured solution | N = 32 | N = 64 |
|---|---:|---:|
| Temperature, linear-in-temperature conductivity | 5.42455e-12 | 8.48210e-14 |
| Mass fraction, composition-dependent corrected diffusion | 5.67204e-11 | 8.71969e-13 |

The integration checks exercise isothermal and adiabatic wall fluxes,
cylindrical/spherical viscous momentum sources, and single/composite-patch
dissipation. The high-diffusivity timestep check compares the absolute rate
against the analytic acoustic-plus-diffusive bound; host and
KernelAbstractions CPU paths agree. Float32 coefficients retain their scalar
type. Hardware GPU coverage is unavailable in this environment.

### bench/neutraldiffusion.jl: the Marrero--Mason fits

```text
julia --project=. -t 1 bench/neutraldiffusion.jl [degree]
julia --project=. -t 1 test/neutral_diffusion_tests.jl
```

N8a vendors Tables 12 and 13 of Marrero and Mason (1972) as
`MARRERO_MASON_1972`: 77 rows, 74 distinct pairs, H2-D2 the only hydrogen
isotopologue pair. Two Opus agents transcribed the tables independently
from the NIST reprint scan at 600--800 dpi and a script diffed the 77 rows
with no disagreement. The printed digits sit in the row constructors and
are converted to SI on construction. `neutral_binary_diffusion` fits the
log-polynomial of `BinaryDiffusionPolynomial` to each row over its stated
range, anchored at that range's geometric midpoint so overlapping rows are
selected unambiguously; the largest relative departure from the source equation over all
77 rows, by polynomial degree, is:

| degree | 6 | 8 | 10 (default) | 12 |
|---|---:|---:|---:|---:|
| worst row residual | 1.66e-2 | 2.84e-3 | 3.53e-4 | 3.32e-5 |

The worst row at every degree is the 3He-4He correlation from 1.74 K; at
the default degree H2-D2 over 14--10^4 K is 2.16e-5, He-Kr 4.76e-6 and
N2-CO 8.56e-6, and every eq (4.3-2) row without a Sutherland term is at
round-off. The paper's uncertainty limits are 1--3% at 300 K and 10--20%
at 10^4 K, so the default degree is an order below the tightest of them
on every row.

The H2-D2 equation is checked against the paper's own Table 20 curve-fit
nodes (viscosity-derived below 1000 K, molecular-beam above), which the
paper fitted with `s` fixed at 1.500. The equation departs from those
nodes by the scatter its deviation plots show, and the polynomial adds
under 5e-5 to that:

| T [K] | 14.12 | 20.32 | 90.0 | 26.09 | 70.32 | 293.0 | 986 | 3313 | 10000 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| equation vs node | -3.15% | +3.08% | -6.88% | +3.42% | -1.64% | -0.30% | +4.84% | +2.45% | -1.60% |

The 90.0 K node is printed out of temperature order in the paper and is
the largest departure.

`data/songwang_extract.jl` fits the 26 Song--Wang pairs from the
supporting information's text layer. An independent Xpdf/layout recovery
in `data/songwang_verify.jl` checks the pinned publisher PDF, all 78 mixture
tables and all 702 equimolar nodes. PyMuPDF cell-order, Xpdf layout-column and
pypdf layout-row recoveries agreed exactly on all 2106 `(T,D12)` tuples. A
visual audit of Table S18 independently confirmed its 27 D12 values, cm²/s
units, 101.3 kPa pressure and 2.0% expanded uncertainty at 0.95 confidence
with `k = 2`.
A degree-4 polynomial in
log(T / 300 K) reproduces every one of the 27 nodes per pair to between
2.2e-5 and 6.7e-5, the table's own five-digit rounding; the 0.25 and 0.75
mole-fraction tables sit within 1.7e-3 to 5.0e-3 of the equimolar one,
which is the composition dependence the model drops. The three sources
agree within their stated uncertainties where they overlap, in cm^2/s at
1 atm:

| pair | Müller--Klemm measured, 297.15 K | Song--Wang calculated, 298.15 K | Marrero--Mason, 297.15 K |
|---|---:|---:|---:|
| H2-HD | 1.349 ± 0.013 | 1.351 | — |
| H2-D2 | 1.268 ± 0.013 | — | 1.246 |
| HD-D2 | 1.126 ± 0.011 | 1.130 | — |
| H2-HT | 1.274 ± 0.025 | 1.284 | — |
| H2-DT | 1.212 ± 0.024 | 1.242 | — |
| H2-T2 | 1.207 ± 0.030 | 1.214 | — |
| D2-HT | 1.044 ± 0.021 | 1.045 | — |
| D2-DT | 0.989 ± 0.020 | 0.992 | — |
| D2-T2 | 0.956 ± 0.019 | 0.956 | — |
| He-H2 | — | 1.574 (4He) | 1.535 at 298.15 K |

The one-kelvin offset between the measured and calculated columns is
worth 0.6%. H2-DT is the widest gap at 2.5%, inside the 2% measurement
and 2% calculation limits combined; the H2-D2 correlation sits 1.7% below
the measurement it never used, inside its 2% group II limit at 300 K.


The constant-model allocation and JET dispatch audits match their pre-change
baselines. A warmed 16-by-12 NASA-9 H2/N2 case with artificial transport off
allocates 4048 bytes per RHS and 176 per CFL evaluation under both constant
transport and either CEA diffusion mode; 1000 pointwise coefficient evaluations
add no allocations beyond the 16-byte scalar measurement overhead. The CEA
coefficient return is fully inferred. Existing convergence and shock-validation
guards pass without baseline or tolerance changes; the MPI core gate passes at
two ranks and its selected phases at eight ranks.

## Neutral polynomial transport

`test/neutral_transport_integration_tests.jl` isolates ordinary binary diffusion
with a periodic cosine at uniform pressure and temperature and synthetic equal
species thermodynamics. It uses the sourced H2-D2 Marrero--Mason coefficient;
this is a numerical closure test, not an isotope experiment. A separate NASA-9
H2/D2 test checks physical species identities, the correction-velocity mass
constraint, nonzero diffusive enthalpy transport, and species permutation.
Thermal diffusion is absent in both cases.

The manufactured case uses a 0.01 m periodic interval, 300 K, 101325 Pa,
mean mass fraction 0.45 and cosine amplitude 0.08, evolved to 0.001 s with
C6, CFL 0.35, and filtering and artificial transport disabled. Both synthetic
species have gas constant 1.0e-4 and gamma 1.4 to isolate diffusion from sound
propagation; these are not the physical isotope thermodynamics.

| Nodes | Maximum mass-fraction error |
|---:|---:|
| 24 | 1.176427e-9 |
| 48 | 1.827178e-11 |

The error decreases by 64.4 under doubling, consistent with sixth order.
Species conservation and uniform pressure and temperature are gated alongside
this result. The source fits and the constant transport defaults are unchanged.

`test/neutral_transport_coefficients_tests.jl` gates exact species order,
scalar conversion, the direct binary limit, pure and trace compositions,
strict inclusive source bounds and finite positive coefficients.
`test/neutral_transport_domain_tests.jl` exercises collective rejection on
single patches, disjoint patch owners and subcycled fine-level subsets,
including the KernelAbstractions CPU launch path. These checks do not add
data or change the provenance and source distinctions recorded above.

## The bulk species channel in three dimensions

`bench/bulkchannel.jl` runs `species_flux = :bulk` against the default `:fickian` on two
three-dimensional cases at the package's constants, eight ranks at `-t 1`. The slab is
`brill_slab` as a sphere: the heavy gas at uniform p = 1 and T = 1 advected at u = 10 along
x through a periodic cube of 48³ points, seven cells across the interface, for two transits,
both gases at γ = 1.4. The bubble is `shock_interface` as a sphere: a Mach 1.5 air shock
into a sphere of radius 0.1 of the heavy gas (γ = 1.09) on a 2h interface, in
(0, 1) × (0, 0.5)² on 128 × 64 × 64 points, Dirichlet ends and periodic sides, to t = 0.25.
Each runs at density ratio 5.04 and 100. The bubble's mixedness is ∫ Y_air Y_heavy dV,
theta is `molecular_mixing` and W `mix_width` along x, and "outside" counts the interior
points beyond the 1e-4 dead band at the end.

```
slab: max|p-1| / max|u-u0|/u0 / worst Y / final rho_min / steps / ms per step
fickian, R = 5.04 | 7.9e-03 / 1.2e-03 / -0.011 / 0.9941 / 334 / 66.9
fickian, R = 100  | 1.8e-01 / 1.0e-02 / -0.103 / 0.9646 / 476 / 65.6
bulk, R = 5.04    | 1.1e-12 / 2.1e-14 / -0.011 / 0.9955 / 329 / 78.4
bulk, R = 100     | 2.2e-11 / 3.2e-13 / -0.064 / 0.9770 / 408 / 78.9

bubble: worst Y / outside / mixedness / theta / W / steps / ms per step
fickian, R = 5.04 | -0.0116 / 4414 / 7.562e-04 / 0.2733 / 0.0443 / 435 / 237
fickian, R = 100  | -0.0284 / 8997 / 1.311e-03 / 0.1984 / 0.1057 / 688 / 237
bulk, R = 5.04    | -0.0123 / 4377 / 7.556e-04 / 0.2733 / 0.0442 / 418 / 307
bulk, R = 100     | -0.0321 / 8857 / 1.317e-03 / 0.1987 / 0.1060 / 594 / 306
```

The slab carries the one-dimensional result into three dimensions: the Fickian pressure
error is the enthalpy flux's and grows with the ratio, the bulk channel's stays at round-off,
and at ratio 100 the bulk channel holds the mass fraction to two thirds of the Fickian
excursion. On the shocked sphere the two channels are indistinguishable at what the grid
resolves: the excursions agree within 15% with the bulk channel's the larger, the mixing
measures to three digits, the kinetic-energy histories to half a percent, and the Fickian
channel completes ratio 100, where the one-dimensional case fails on the transmitted
shock's foot at 400 points. The channel costs 17% per step on the slab and 30% on the
bubble: the n_cons − n_species extra gradient solves per direction and the second sensor
field per species.

The `budget` part evaluates the right-hand side three times on one state, on the run's own
solver, on one with C_D = C_Y = 0 and on one with every artificial constant zero, so the
species channel's rate of change of kinetic energy stands beside the three artificial
viscosities'. The isolation is exact, since neither constant enters μ\*, β\* or κ\*. The
Fickian flux has no momentum component and its kinetic-energy rate is zero by construction
(1e-19 measured). Under the bulk channel −D_b ∇(ρu) is a viscosity, and on the bubble its
sink runs at 0.4 to 1.6% of the artificial viscosities' at ratio 5.04 and 2 to 5% at ratio
100 once the shock has crossed the sphere, with max D_b at 1e-3 to 7e-3 against max μ\*/ρ at
1e-5: the composition varies where the velocity is smooth. The compact filter's sink is
outside this budget.

The `constants` part, the bubble at 96 × 48 × 48 under the bulk channel:

```
worst Y / points outside the band / theta
C_D    | R = 5.04: C_Y 50 / 100 / 200                          | R = 100: C_Y 50 / 100 / 200
0.005  | -0.016 / 2747 / 0.333, -0.011 / 2342 / 0.333, -0.009 / 2030 / 0.334 | -0.042 / 5626 / 0.241, -0.032 / 4816 / 0.241, -0.069 / 4037 / 0.242
0.01   | -0.016 / 2723 / 0.333, -0.011 / 2290 / 0.334, -0.009 / 1978 / 0.334 | -0.042 / 5563 / 0.242, -0.031 / 4707 / 0.242, -0.098 / 3941 / 0.243
0.02   | -0.015 / 2631 / 0.335, -0.011 / 2226 / 0.335, -0.009 / 1934 / 0.336 | -0.041 / 5561 / 0.244, -0.030 / 4574 / 0.245, -0.056 / 3935 / 0.245
```

C_D moves the excursion by 5% over a fourfold range and the mixing measures by a percent;
C_Y = 100 is the minimum of the excursion at ratio 100, where 200 doubles or triples it,
and at ratio 5.04 the gain from 100 to 200 is a fifth. The constants stay where the
Fickian channel put them. With the heavy gas at γ = 1.09 the slab's Fickian pressure error
is unchanged (7.9e-3 and 2.1e-1) and the bulk channel's stays at round-off, so the
unequal-γ contact drift with no shock is the Fickian enthalpy flux as well.

`bench/bulkentropy.jl` measures what the discrete operators make of the entropy inequalities
of `reference/DESIGN.md` ("The species channel", property 3 and its partial-density form), on a Taylor–Green velocity
field at Mach 0.1 carrying a sphere of the heavy gas (ratio 5.04, γ = 1.09) through a
periodic cube of 32³ points to t = 2, 46 steps, and on the one-dimensional `brill_slab`. It
derives the entropy variables w = ∂(ρs)/∂q of an ideal mixture and checks them against
central differences (7e-10 relative over 64 states); integrates the channel's semi-discrete
production ∫ w·R dV, with R the right-hand-side difference above, beside the continuous
production on the solver's own gradients, the quadratic form D_b ∇qᵀ(−η'')∇q for the bulk
channel and Σ_k R_k D_b |∇ρ_k|²/ρ_k for the partial-density one, their difference being the
defect of a derivative that is not summation-by-parts against the quadrature; and records
∫ρs after every Runge–Kutta step and after every filter pass, the pass run from the
callback and verified bitwise against an ordinary run. Points where a partial density is
nonpositive are excluded from the production integrals and counted, and floored in ∫ρs.

```
semi-discrete, 32^3, C_D = 0.1, t = 0.44 .. 1.77: P_channel / defect from the continuous form
partial density | +7.12e-2 .. +7.30e-2 / -9e-7 .. -1.1e-6
bulk            | +7.18e-2 .. +7.35e-2 / -9e-7 .. -1.1e-6
fickian         | +5.60e-2 .. +5.73e-2 / n/a

fully discrete: steps / RK decreases (worst) / filter decreases (worst) / total change
32^3 partial |   46 /    0            /   43 (-6e-6)    | +1.4e-1
32^3 bulk    |   46 /    0            /   43 (-6e-6)    | +1.4e-1
32^3 fickian |   46 /    0            /   43 (-6e-6)    | +1.1e-1
32^3 art off |   45 /   45 (-1e-7)    /   42 (-7e-6)    | -1.7e-4
slab partial | 4049 / 1996 (-9e-4)    / 4033 (-1e-4)    | +1.1e-2
slab bulk    | 4049 / 1996 (-9e-4)    / 4033 (-1e-4)    | +1.1e-2
slab fickian | 4094 / 2080 (-3e-3)    / 1987 (-1e-4)    | -2.2e-2
slab art off | fails at step 187
```

The channel's semi-discrete production is positive at every sample under all three channels
and is the whole right-hand side's, and the two consistent channels' agree with their
continuous forms to 1.5e-5 relative: the non-SBP defect is not where the inequality is
lost. The partial-density channel produces slightly less than the bulk one, whose added
viscosity and conduction produce the difference; on the slab, at uniform u and T, the two
are the same operator and their records agree to the digits printed. The complete update
does not inherit it. On the resolved three-dimensional case the Runge–Kutta step never
lowers ∫ρs with a channel on and the filter pass lowers it on nearly every step, by a
tenth of a percent of what the channel produces; on the slab, whose interface holds mass
fractions outside [0, 1] throughout, the Runge–Kutta step lowers it on half the steps under
every channel. The
inequality is a property of the continuous model and of the semi-discrete channel term; the
filter and the bounded-fraction excursions are outside it.

## The species validity band

`julia --project=. bench/speciesband.jl` (about three minutes, serial). The cases are
`species_advection`, `shock_interface` and `brill_slab` of `test/cases.jl` under the
defaults; `worst run` is the largest max(−Y_k, Y_k − 1) over every point and completed step,
`worst end` the same on the state the run returns, and the last column counts end-state
points beyond four candidate bands.

```
case                         N   worst run   worst end   end points beyond 1e-4 / 1e-3 / 1e-2 / 5e-2
advection, 2 cells          64   1.415e-04   1.394e-04   2 / 0 / 0 / 0
advection, 2 cells         256   3.995e-04   3.351e-04   6 / 0 / 0 / 0
advection, 2 cells        1024   3.995e-04   2.761e-04   6 / 0 / 0 / 0
advection, 1/32 physical   128   0           0           0 / 0 / 0 / 0
advection, 1/32 physical  1024   0           0           0 / 0 / 0 / 0
shock, C_Y = 100           100   1.144e-02   2.201e-03   4 / 1 / 0 / 0
shock, C_Y = 100           400   9.844e-03   9.215e-04   4 / 0 / 0 / 0
shock, C_Y = 100          1600   8.862e-03   4.022e-04   4 / 0 / 0 / 0
shock, C_Y = 0             200   6.620e-02   1.018e-02   7 / 3 / 1 / 0
shock, C_Y = 0             400   5.987e-02   8.050e-03   7 / 4 / 0 / 0
shock, C_Y = 0             800   5.445e-02   7.076e-03   11 / 4 / 0 / 0
slab, Np = 7               140   6.205e-02   1.897e-03   9 / 2 / 0 / 0
slab, Np = 14              280   4.250e-03   1.638e-03   8 / 1 / 0 / 0
slab, Np = 28              560   2.313e-04   1.982e-04   4 / 0 / 0 / 0
```

The excursion of an interface held at grid scale does not converge with the grid. The
advected slab with two-cell edges overshoots by 4.0e-4 at every N from 256 up, while edges
of fixed physical width stay inside [0, 1] exactly once they span four cells; the shocked
interface holds 0.9–1.1e-2 from N = 100 to 1600, since the shock compresses it to the few
cells the artificial diffusivity sets, which a shock-capturing run has somewhere. Only the slab, whose interface is resolved over
`Np` cells by construction, converges (from 6.2e-2 at the default Np = 7 to 2.3e-4 at 28);
below Np = 7 it loses positivity. The endpoints are smaller than the transients in every
case and stay below 3e-3. With the mass-fraction bound off, the shocked interface reaches
0.054–0.066 and ends at 0.007–0.010; before the defaults moved to the partial-density
channel at C_D = 0.1 it reached 0.2 and ended at 0.09–0.15, the ringing C_D now damps.

The earlier species test borrowed `ArtParams.Y_tolerance = 1e-4`, the bound's dead band,
which every run in the table crosses at its endpoint; under it the Mach 1.5 case, the slab,
the uniform advection and the He/CO2 shock tube of `examples/shock_tube.jl` (117 of 6144
points at nx = 384, ny = 16) all fail a strict check, and a run with retries repeats its
trajectory four times before failing. `StepControl.species_band = 0.05` sits above every
bound-on excursion measured here and the Mach 3 worst of −0.043 recorded under
[C_Y](#c_y-the-mass-fraction-bound), and below the bound-off transient, which still crosses
it.
The one bound-on transient above the band is the slab at Np = 7, which only a
`validity_interval` check or a guard would see. With the band, the species cases, both
examples and the small configurations of `src/precompile.jl` run under the default
`:strict`; the He/CO2 example completes at its default 768 × 48.

What remains permissive is the negative internal energy of a shock converging into a cold
or near-vacuum ambient ([below](#negative-internal-energy-in-completed-runs)): the three Noh
geometries, the two anisotropic Cartesian Noh cases and Sedov. The cylindrical converging
shock of `examples/converging_shock.jl`, whose ambient is at p = 1, completes strict at nx =
256, 512 and 1024. An ideal gas at e < 0 has no temperature, so no threshold can admit
these cells honestly, and `:strict` stays the default: each of the six cases bounds its
closing inadmissible count and e_min in `test/validation.jl`.
