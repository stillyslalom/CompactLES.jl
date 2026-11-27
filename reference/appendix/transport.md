# Calibration appendix: Transport, species and interfaces

This file holds the sections of the measurement record on molecular transport, the
species channel, the composition bounds and the material interfaces.
[CALIBRATION_APPENDIX.md](../CALIBRATION_APPENDIX.md) indexes the record and states its
conventions.

## Temperature-dependent transport

```text
julia --project=. -t 1 test/transport_tests.jl
julia --project=. -t 1 test/transport_integration_tests.jl
julia --project=. -t 1 bench/archive/transport.jl
```

N8 retains `ConstantTransport()` as the default and adds `CeaTransport(eos)` as an
opt-in dimensional gas model. The bundled table supplies 66 pure-species
viscosity/conductivity records and 41 binary viscosity-interaction records,
but no binary diffusion coefficients. Unity-Lewis diffusion is therefore the
CEA model's default; mixture-averaged mass diffusion requires explicitly
supplied `BinaryDiffusion` data. The mixture rules and units are documented in
[thermodynamics](../../docs/src/explanation/thermodynamics.md).

The coefficient checks evaluate independent literal CEA rows in SI units,
test the pure/trace and species-permutation limits, and distinguish the
mass-gradient diffusion convention from the mole-gradient convention using
unequal ternary coefficients. The evolution cases use a periodic Fourier
profile with analytic time-dependent forcing, no filter or artificial
transport, and nonlinear temperature-dependent conductivity or
composition-dependent corrected species diffusivity. These are manufactured
solution checks of the implementation, not experimental calibration of the
gas model or its binary diffusion inputs.

Both maximum solution errors fall at sixth order under doubling, the order of the default
derivative operator:

| Manufactured solution, Julia 1.11.4, Windows x86-64, `cfl=0.15`, final time `0.01` | N = 32 | N = 64 |
|---|---:|---:|
| Temperature, linear-in-temperature conductivity | 5.42455e-12 | 8.48210e-14 |
| Mass fraction, composition-dependent corrected diffusion | 5.67204e-11 | 8.71969e-13 |

The integration checks exercise isothermal and adiabatic wall fluxes,
cylindrical/spherical viscous momentum sources, and single/composite-patch
dissipation. The high-diffusivity timestep check compares the absolute rate
against the analytic acoustic-plus-diffusive bound; host and
KernelAbstractions CPU paths agree. Float32 coefficients retain their scalar
type. Hardware GPU coverage is unavailable in this environment.

### bench/archive/neutraldiffusion.jl: the Marrero--Mason fits

```text
julia --project=. -t 1 bench/archive/neutraldiffusion.jl [degree]
julia --project=. -t 1 test/neutral_diffusion_tests.jl
```

`MARRERO_MASON_1972` holds Tables 12 and 13 of Marrero and Mason. Its 77 rows
cover 74 distinct pairs, and H2-D2 is the only hydrogen isotopologue pair. Two
independent transcriptions from the NIST reprint scan at 600--800 dpi, diffed by
script, agree on every row. The printed digits sit in the row constructors and
are converted to SI on construction. `neutral_binary_diffusion` fits the
log-polynomial of `BinaryDiffusionPolynomial` to each row over its stated
range, anchored at that range's geometric midpoint so overlapping rows are
selected unambiguously.

At the default degree the fit reproduces every row at least an order of magnitude
inside the paper's tightest uncertainty limits, and the residual falls by about an
order of magnitude for each two degrees added. The largest relative departure from
the source equation over all rows, by polynomial degree:

| degree | 6 | 8 | 10 (default) | 12 |
|---|---:|---:|---:|---:|
| worst row residual | 1.66e-2 | 2.84e-3 | 3.53e-4 | 3.32e-5 |

| row, at the default degree (the worst row at every degree is 3He-4He from 1.74 K) | relative residual |
|---|---:|
| H2-D2 over 14--10^4 K | 2.16e-5 |
| N2-CO | 8.56e-6 |
| He-Kr | 4.76e-6 |
| eq (4.3-2) rows without a Sutherland term | round-off |

| paper's uncertainty limits | at 300 K | at 10^4 K |
|---|---:|---:|
| relative | 1--3% | 10--20% |

The H2-D2 equation departs from the paper's own Table 20 curve-fit nodes by the
scatter of the paper's deviation plots, and the polynomial adds under 5e-5 to that.
The nodes are viscosity-derived below 1000 K and from molecular beams above, and
the paper fitted them with `s` fixed at 1.500:

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
visual audit confirmed the 27 `D12` values of Table S18. They are given in cm²/s at
101.3 kPa. Their stated expanded uncertainty is 2.0% at 0.95 confidence (`k = 2`).

A degree-4 polynomial in log(T / 300 K) reproduces each pair's nodes to the
table's own five-digit rounding. The composition dependence the model drops,
the difference between the equimolar table and those at other mole fractions,
is nearly two orders of magnitude larger than that:

```
Song--Wang tables, relative difference              range
polynomial fit vs the nodes of each pair            2.2e-5 .. 6.7e-5
0.25 and 0.75 mole-fraction tables vs equimolar     1.7e-3 .. 5.0e-3
```

The three sources agree within their stated uncertainties where they overlap, in
cm^2/s at 1 atm:

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
worth 0.6%. H2-DT shows the widest gap, 2.5%, inside the two 2% limits of
measurement and calculation combined. The H2-D2 correlation, which did not use the
measurement, sits 1.7% below it, inside its 2% group II limit.


CEA transport allocates no more than constant transport and adds no dispatch site.
The allocation and JET dispatch audits of the constant model are unchanged, a
warmed NASA-9 H2/N2 case with artificial transport off allocates the same under
constant transport and under either CEA diffusion mode, and the CEA coefficient
return is fully inferred.

```
warmed 16-by-12 case, artificial transport off               bytes
per right-hand side                                           4048
per CFL evaluation                                            176
1000 pointwise coefficient evaluations, beyond the overhead   none
scalar measurement overhead                                   16
```

Existing convergence and shock-validation guards pass without baseline or tolerance
changes; the MPI core gate passes at two ranks and its selected phases at eight
ranks.

## Neutral polynomial transport

`test/neutral_transport_integration_tests.jl` isolates ordinary binary diffusion
with a periodic cosine at uniform pressure and temperature and synthetic equal
species thermodynamics. It uses the sourced H2-D2 Marrero--Mason coefficient;
this is a numerical closure test, not an isotope experiment. A separate NASA-9
H2/D2 test checks physical species identities, the correction-velocity mass
constraint, nonzero diffusive enthalpy transport, and species permutation.
Thermal diffusion is absent in both cases.

The manufactured case is run with filtering and artificial transport disabled:

| periodic interval | T | p | mean mass fraction | cosine amplitude | final time | derivative | CFL |
|---:|---:|---:|---:|---:|---:|---|---:|
| 0.01 m | 300 K | 101325 Pa | 0.45 | 0.08 | 0.001 s | C6 | 0.35 |

Both synthetic
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

`bench/archive/bulkchannel.jl` runs `species_flux = :bulk` against `:fickian` on two
three-dimensional cases at the package's constants, eight ranks at `-t 1`, each at density
ratios 5.04 and 100. The slab is `brill_slab` as a sphere of the heavy gas advected through
a periodic cube at uniform pressure and temperature. The bubble is `shock_interface` as a
sphere: an air shock into a sphere of the heavy gas, with Dirichlet ends and periodic
sides. The bubble's mixedness is ∫ Y_air Y_heavy dV, theta is `molecular_mixing` and W
`mix_width` along x, and "outside" counts the interior points beyond the 1e-4 dead band at
the end.

```
case     state                                   gases                  grid and domain                          end
slab     p = 1, T = 1, u = 10 along x;           both γ = 1.4           48³ periodic                             two transits
         seven cells across the interface
bubble   Mach 1.5 shock; sphere of radius 0.1     heavy gas γ = 1.09     128 × 64 × 64 on (0, 1) × (0, 0.5)²      t = 0.25
         on a 2h interface
```

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
excursion. On the shocked sphere the two channels are indistinguishable at the grid's
resolution. The excursions agree within 15%, the bulk channel's the larger, the mixing
measures agree to three digits and the kinetic-energy histories to half a percent. The
Fickian channel completes ratio 100 here, where the one-dimensional case fails on the
transmitted shock's foot at 400 points. The bulk channel costs 17% more per step on the
slab and 30% on the bubble: the n_cons − n_species extra gradient solves per direction and
the second sensor field per species.

The `budget` part evaluates the right-hand side three times on one state, on the run's own
solver, on one with C_D = C_Y = 0 and on one with every artificial constant zero, so the
species channel's rate of change of kinetic energy stands beside the three artificial
viscosities'. The isolation is exact, since neither constant enters μ\*, β\* or κ\*. The
Fickian flux has no momentum component and its kinetic-energy rate is zero by construction.
Under the bulk channel −D_b ∇(ρu) is a viscosity, and on the bubble its sink remains a few
percent of the artificial viscosities' at most, although max D_b exceeds max μ\*/ρ by two
orders of magnitude or more: the composition varies where the velocity is smooth. The
compact filter's sink is outside this budget.

```
bubble, once the shock has crossed the sphere          R = 5.04        R = 100
bulk channel's kinetic-energy sink / artificial        0.4 .. 1.6%     2 .. 5%
  viscosities'
max D_b                                                1e-3 .. 7e-3
max μ*/ρ                                               1e-5
Fickian kinetic-energy rate, measured                  1e-19
```

The `constants` part repeats the bubble on a coarser grid under the bulk channel:

```
worst Y / points outside the band / theta, bubble at 96 × 48 × 48, bulk channel
C_D    | R = 5.04: C_Y 50 / 100 / 200                          | R = 100: C_Y 50 / 100 / 200
0.005  | -0.016 / 2747 / 0.333, -0.011 / 2342 / 0.333, -0.009 / 2030 / 0.334 | -0.042 / 5626 / 0.241, -0.032 / 4816 / 0.241, -0.069 / 4037 / 0.242
0.01   | -0.016 / 2723 / 0.333, -0.011 / 2290 / 0.334, -0.009 / 1978 / 0.334 | -0.042 / 5563 / 0.242, -0.031 / 4707 / 0.242, -0.098 / 3941 / 0.243
0.02   | -0.015 / 2631 / 0.335, -0.011 / 2226 / 0.335, -0.009 / 1934 / 0.336 | -0.041 / 5561 / 0.244, -0.030 / 4574 / 0.245, -0.056 / 3935 / 0.245
```

The bulk channel keeps the Fickian channel's constants. C_D moves the excursion by 5% over a
fourfold range and the mixing measures by a percent. At the higher ratio the excursion is
smallest at C_Y = 100 and doubles or triples at 200, while at the lower ratio the same step
reduces it by only a fifth.

The unequal-γ contact drift with no shock is also the Fickian enthalpy flux's. With the
heavy gas of the slab at γ = 1.09, the Fickian pressure error keeps its size and the bulk
channel's stays at round-off:

```
slab, unequal γ                fickian pressure error
R = 5.04                       7.9e-3
R = 100                        2.1e-1
```

`bench/archive/bulkentropy.jl` measures what the discrete operators make of the entropy inequalities
of `reference/DESIGN.md` ("The species channel", property 3 and its partial-density form),
on a Taylor–Green velocity field carrying a sphere of the heavy gas through a periodic cube
and on the one-dimensional `brill_slab`. The entropy variables w = ∂(ρs)/∂q of an ideal
mixture are derived in the script and agree with central differences to 7e-10 relative over
64 states. The script integrates the channel's semi-discrete production ∫ w·R dV, with R
the right-hand-side difference above, beside the continuous production on the solver's own
gradients: the quadratic form D_b ∇qᵀ(−η'')∇q for the bulk channel and
Σ_k R_k D_b |∇ρ_k|²/ρ_k for the partial-density one. Their difference is the defect of a
derivative that is not summation-by-parts against the quadrature. The script also records
∫ρs after every Runge–Kutta step and after every filter pass, the pass run from the
callback and verified bitwise against an ordinary run. Points where a partial density is
nonpositive are excluded from the production integrals and counted, and floored in ∫ρs.

```
Taylor-Green case: Mach 0.1, heavy-gas sphere at ratio 5.04 and γ = 1.09, 32³ periodic, to t = 2

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

## The contact with a temperature jump

`bench/archive/thermalcontact.jl`, serial at `-t 1`, all parts at their defaults (about four
minutes). The slab is a layer of CO2 filling the middle half of a periodic domain of He at
uniform pressure and velocity, with physical transport off and the production filter and
constants. `ideal` is the `IdealMixture` of the same two species with heat capacities fixed
at a reference temperature; `nasa9` raises CO2's c_p by about 40% between the two layer
temperatures. The method and the continuous rates are in `reference/DESIGN.md`, "The species
channel".

```
domain        1 m periodic, He at 300 K, CO2 layer at 300 or 900 K
state         uniform 1e5 Pa and 100 m/s
interfaces    0.05 m thick: 6.4 cells at N = 128, 12.8 at 256
ideal         heat capacities at 298.15 K: γ = 5/3 (He) and 1.29 (CO2)
```

The initial rates: the pressure change per τ = h/(U + c_He) over p0, discrete (the
right-hand side increment through the EOS linearization) against the continuous model's
rate in the temperature-equation form with the same coefficient field; U_LM is the
low-Mach velocity amplitude the continuous source sets up. "convection" is the scheme with
the artificial properties off, whose continuous rate is zero; each other row is an
increment. max |∂_t u| τ is below 7e-13 m/s on every row, the uniform-u identity.

```
T_CO2 = 900 K     | max|dp| disc / cont / rel diff / U_LM m/s
                  | nasa9, N = 128                  | nasa9, N = 256                  | ideal, N = 128                  | ideal, N = 256
convection        | 2.3e-03 / 0 / - / -             | 4.4e-05 / 0 / - / -             | 1.9e-03 / 0 / - / -             | 3.1e-05 / 0 / - / -
kappa*            | 2.9e-04 / 2.9e-04 / 5e-14 / 0.31 | 9.2e-06 / 9.2e-06 / 2e-12 / 0.015 | 4.6e-04 / 4.6e-04 / 6e-14 / 0.57 | 2.2e-05 / 2.2e-05 / 1e-12 / 0.034
fickian           | 1.3e-02 / 1.5e-02 / 0.125 / 13.9 | 5.2e-04 / 5.2e-04 / 0.0070 / 0.88 | 1.4e-02 / 1.5e-02 / 0.138 / 13.7 | 5.2e-04 / 5.2e-04 / 0.0076 / 0.88
bulk              | 4.3e-03 / 4.3e-03 / 0.041 / 4.20 | 1.6e-04 / 1.6e-04 / 0.0046 / 0.24 | 3.1e-03 / 3.1e-03 / 0.069 / 2.88 | 1.2e-04 / 1.2e-04 / 0.0059 / 0.16
partial_density   | 5.4e-03 / 5.5e-03 / 0.028 / 4.93 | 1.7e-04 / 1.7e-04 / 0.0019 / 0.31 | 5.0e-03 / 5.1e-03 / 0.017 / 4.83 | 1.6e-04 / 1.6e-04 / 0.0035 / 0.30

T_CO2 = 300 K     | fickian 1.2e-02 (N = 128), 4.5e-04 (256), discrete = continuous to 1e-13, U_LM 12.6 and 0.77 m/s,
                  | under both EOS; bulk, partial density, kappa* and convection below 1e-16
```

At uniform T the Fickian pressure error is its own continuous model's volume source, to
round-off; the consistent channels have none. At the temperature jump every row has one.
Each channel's discrete rate converges to its continuous model's: under doubling the
relative difference between them falls 5 to 18 times. The model's own rate falls 26 to 32
times, since D_b is sensed at the grid scale and the interface is better resolved. The
partial-density and bulk models call for a third of the Fickian velocity and κ\* for a
twentieth to a sixtieth of it. The convective error on the coarser grid is of the same
order as the channel's source and falls 52 to 62 times under doubling. The NASA-9 heat
capacities raise it by a fifth over the constant-c_p control, so most of it comes from the
unequal γ.

The advected slab over one transit: the largest spatial deviation of p from its mean over
p0 and of u from 100 m/s over the run, and the mean-pressure change at the end. "off" has
the artificial properties disabled, "none" has them on with C_D = C_Y = 0. The last line
gives the mean pressure of the completely mixed state.

```
T_CO2 = 900 K     | max|p−p̄|/p0 / max|u−U| m/s / (p̄−p0)/p0
                  | nasa9, N = 128             | nasa9, N = 256             | ideal, N = 128             | ideal, N = 256
off               | FAIL step 2378             | 4.7e-04 / 0.26 / -7.6e-05  | 9.3e-03 / 3.77 / -1.7e-03  | 3.4e-04 / 0.17 / -1.1e-05
none              | 1.9e-02 / 6.16 / +5.8e-03  | 4.6e-04 / 0.25 / +1.4e-04  | 9.2e-03 / 3.43 / +8.1e-04  | 3.4e-04 / 0.15 / +1.9e-04
fickian           | 3.1e-02 / 15.8 / +8.3e-03  | 1.8e-03 / 1.03 / +1.3e-03  | 2.6e-02 / 14.6 / +4.5e-03  | 1.9e-03 / 0.99 / +1.0e-03
bulk              | 2.1e-02 / 6.86 / +1.5e-02  | 9.1e-04 / 0.45 / +4.6e-03  | 1.4e-02 / 4.74 / +1.1e-02  | 6.7e-04 / 0.31 / +3.4e-03
partial_density   | 2.0e-02 / 8.10 / +1.2e-02  | 8.3e-04 / 0.58 / +3.3e-03  | 1.4e-02 / 6.34 / +9.1e-03  | 7.5e-04 / 0.45 / +2.5e-03

T_CO2 = 300 K     | fickian 2.7e-02 / 15.7 (N = 128), 1.8e-03 / 1.05 (256) under both EOS;
                  | every other row below 5e-14 / 2e-11

mixed state       | (p̄−p0)/p0 = 0.359 under NASA-9 and 0.237 under the control at 900 K,
                  | zero to round-off at 300 K
```

The measured velocity drift of each channel is about the convective one ("none") plus its
continuous U_LM:

```
N = 256, NASA-9, m/s     "none" + U_LM     measured max|u−U|
partial_density          0.25 + 0.31       0.58
bulk                     0.25 + 0.24       0.45
fickian                  0.25 + 0.88       1.03
```

The pressure deviation is of order ρcU_LM/p0 over the He, 5e-4 for the default at
N = 256. The mean pressure moves toward the mixed value in every channel, fastest under
bulk, whose added conduction exchanges the most heat. The one failure is the NASA-9 slab
at N = 128 with the artificial properties off. The rows without a species channel at
N = 128 take `StepControl` retries near the end of the transit and complete.

The identities are checked on a shocked contact, a Mach 1.5 shock in the He into the CO2
at 900 K between Dirichlet ends. The rest of the setup and the ranges of Φ and Π, against
which the residuals are read, are in the first lines of the table.

```
shock at x = 0.15, CO2 from x = 0.35, N = 400, to 400 µs, sampled every 40 steps
Φ = 1.0e7 to 2.0e7 W/m², Π = −1.0e8 to −1.3e8 W/m² throughout

W/m², over the 21 samples      | nasa9                  | ideal
∫rK_ch, steps 40–280          | |.| ≤ 5.4e1            | |.| ≤ 4.9e1
∫rK_ch, steps 320–480         | −4.9e4 to +3.2e2       | −1.8e4 to +5.8e3
∫rK_ch, steps 520–840         | |.| ≤ 4.5              | |.| ≤ 0.7
∫rK_sp                        | |.| < 4e-10            | |.| < 3e-10
∫rK_cv                        | −3.1e4 to −1.8e5       | −4.5e4 to −1.8e5
∫rP, steps 40–440             | |.| ≤ 4.1              | |.| ≤ 4.7
∫rP, steps 480–840            | +7.3e2 to −9.0e3       | +7.4e2 to −9.4e3
rel |rT_ch|                   | 0.14 to 1.08           | 0.06 to 0.80
rel |rK_cv|                   | 0.017 to 0.15          | 0.019 to 0.15
rel |rP|                      | 0.019 to 0.16          | 0.018 to 0.14
```

The channel's kinetic-energy exchange takes either sign and is small except while the
transmitted and reflected waves cross the interface, when it reaches 3e-3 of Φ; before and
after it stays below 1e-5 of Φ. The convective exchange is dissipative, at 0.2% to 1% of Φ.
The split form of the channel's momentum term integrates to round-off, as skew-symmetry
requires.
∫rP departs from zero only through the boundary rows once the reflected wave reaches the
Dirichlet end, at 7e-5 of Π. The heating residual has
an order-one local relative error at the two-cell shocked interface under either EOS; at
the resolved contact of the `rates` table the same residual is the 0.2–3% `rel diff` of
the partial-density row.

**Decision:** the default is retained. At the temperature jump the partial-density
channel's pressure and velocity drift is its own continuous model's volume source,
reproduced to 0.2% at N = 256, of the same order as the bulk channel's and a third of the
Fickian one, and it adds to a convective pressure-equilibrium error of similar size that
no channel removes. The product form of the consistency fluxes is kept.

### The energy budget and the layer widths of the shocked tube

`bench/archive/tubebudget.jl`, serial at `-t 16`. The tube is the He/CO2 deck of
`bench/archive/he_co2_shock_tube.jl` under NASA-9 thermodynamics with Euler fluxes, measured under
`detector = :delta4` at `C_D = 0.1` and `compact_filter(0.45)`; [the partial-density
channel's](shocks.md#the-partial-density-species-channel) two-dimensional comparison runs the same
deck calorically perfect under the current defaults. A callback replays every accepted step with
each stage's right-hand side split by term; the replayed state agreed bitwise with the one
`run!` held after every step of every run, so the `repairs` row is zero throughout. Changes
are in J per metre of depth on the NASA-9 energy scale.

```
grids               768 × 48, nine minutes per channel; 384 × 24, under three
C_D                 0.1, to 2.5 ms
incident shock      reaches the contact before 1.0 ms
transmitted shock   reaches the end wall between 2.0 and 2.5 ms
initial energies    KE0 = 0, IE0 = −4.387e5 J/m
```

The default channel on the finer grid:

```
term             | to 2.0 ms: ΔKE / ΔIE  | to 2.5 ms: ΔKE / ΔIE / ΔE
pressure         | +92534 / −92534       | +107120 / −107176 / −56.0
convection       | −57.4 / +57.4         | −169.3 / +126.0 / −43.3
mu*              | −24.3 / +24.3         | −36.7 / +36.9 / +0.22
beta*            | −8882 / +8882         | −13247 / +13329 / +82.2
kappa*           | 0 / −9e-8             | 0 / +39.4 / +39.4
species channel  | −0.78 / +0.78         | −1.14 / +1.14 / −5e-9
molecular        | 5e-13 / −6e-13        | 7e-13 / −7e-13 / −2e-14
wall flux        | 0 / +6e-8             | −0.0023 / −48.3 / −48.3
wall state       | 0 / −1e-7             | 0 / −1.01 / −1.01
filter           | +2.13 / −2.13         | +20.8 / −14.0 / +6.87
repairs          | 0 / 0                 | 0 / 0 / 0
measured         | +83571.4 / −83571.4   | +93686.3 / −93706.1 / −19.9
closure          | −4e-8 / −4e-7         | −7e-8 / −4e-7
```

The pressure work supplies the kinetic energy, and β\* returns 12.4% of it to internal
energy at the shocks. The other sinks are small: μ\* returns 0.03% and the convective
term's discrete kinetic-energy loss is 0.16%. The filter's net, about half the μ\* row, is
an early removal followed by a larger addition. `molecular` is the remainder of the
assembled flux after every named part, zero under the Euler deck, so its row is the check
that the split is complete.

```
filter, ΔKE     removed up to 1.0 ms: 18.2 J/m     added after: 39 J/m
```

Until the transmitted shock reaches the end wall, the total-energy change and every wall
row stay below 1e-3 J/m. The shock's reflection brings in the slip wall's flux correction,
the wall-node enforcement and a total-energy change of 5e-5 of IE0, spread over the
pressure, convection, β\*, κ\* and filter rows by the closure rows. On the coarser grid the
same change is +687 J/m. The rows that differ between the channels, on the finer grid:

```
ΔKE, 2.5 ms, J/m | species channel | mu*   | beta*  | filter | convection | measured
fickian          | 0               | −37.0 | −13330 | +19.8  | −170.3     | +93636.8
bulk             | −134.1          | −35.1 | −13103 | +21.5  | −171.7     | +93742.8
partial_density  | −1.14           | −36.7 | −13247 | +20.8  | −169.3     | +93686.3
```

The Fickian channel carries no net mass flux, so it changes neither the kinetic energy nor
∫E. The bulk channel's kinetic-energy sink is 3.7 times the μ\* row, which is its added
viscosity ρD_b∇u; the partial-density channel's is its mass-flux exchange, 1e-5 of the
pressure work. Molecular transport, `CeaTransport` at unity Lewis number on the default
channel, removes 0.33 J/m of kinetic energy and moves every other row by less than 0.1%.

A conservative term without net mass flux shows in the integrals only at the walls, so
where each term puts heat is measured separately: its sensible heating
δ(ρe) − Σ_k e_k δρ_k = ρc_v δT split between the gases by their shares of the local heat
capacity. Equal and opposite entries are a transfer from the He to the CO2, entries of one
sign a source or a sink. To 2.5 ms, He / CO2 in J/m:

```
                 | 768 × 48, partial_density | 384 × 24: fickian / bulk / partial_density
kappa*           | −26 / +65                 | −58 / +104    −12 / +52     −32 / +76
species channel  | −394 / −427               | −2742 / +3868 −848 / +500   −481 / −479
molecular (CEA)  | −2.4 / +3.4               | –
filter           | +129 / +58                | +117 / +6     +126 / +21    +131 / +7
```

The Fickian channel moves heat from the He into the CO2 and on the coarser grid adds
1100 J/m, its enthalpy flux's volume source. The partial-density channel moves none: it
removes sensible heat from both gases in about equal parts. Its heating is
−Σ_k c_v,k J_k·∇T, and the CO2 flux into the hotter He outweighs the He flux into the
CO2. The bulk channel lies between the two. κ\* moves 26–58 J/m, an order of magnitude
more than the physical conduction. The finer-grid column is the `CeaTransport` run, whose
κ\* and channel entries match the Euler run's to 1%.

The widths are measured in cells on the band within 0.04 m of the layer. Y, u and T are
the thicknesses of the species, velocity and temperature layers as the script header
defines them, ΔU and ΔT the velocity jump and the temperature variation across the layer.
Doubling the band to 0.08 m moves every thickness by at most 11% from the first sample on,
the velocity layer's the most. Earlier, at 1.0 ms, the reflected shock is inside the wider
band.

```
band: 8 cells at 768 × 48, 4 at 384 × 24
768 × 48          | 1.5 ms: Y / u / T  ΔU  ΔT | 2.0 ms                    | 2.5 ms
fickian           | 4.46 / 3.64 / 4.80  210 203 | 4.69 / 3.62 / 4.86  225 216 | 4.98 / 3.65 / 4.96  237 219
bulk              | 4.60 / 4.21 / 4.85  165 145 | 4.90 / 3.93 / 4.90  139 143 | 4.95 / 3.76 / 4.70  130 133
partial_density   | 4.52 / 3.75 / 4.47  215 170 | 4.73 / 3.72 / 4.41  227 172 | 5.00 / 3.79 / 4.48  238 168
384 × 24          |                             |                             |
fickian           | 3.95 / 2.90 / 3.96  200 182 | 4.19 / 3.17 / 4.17  199 209 | 4.34 / 3.29 / 4.32  213 217
bulk              | 4.12 / 3.39 / 4.03  165 133 | 4.41 / 3.37 / 4.07  133 131 | 4.59 / 3.33 / 4.14  122 127
partial_density   | 4.01 / 2.92 / 3.84  202 157 | 4.23 / 3.31 / 3.91  201 170 | 4.43 / 3.52 / 4.02  211 170
```

All three layers are set by the grid, and none is a physical width. Halving the spacing
multiplies each thickness in cells by only 1.08 to 1.15. In metres each layer therefore
thins by a factor 1.7 to 1.85. From 2.0 ms on the velocity layer is the thinnest under
every channel. Its thickness is 0.7 to 0.8 of the species layer's. The temperature layer is
as wide as the species layer under the Fickian channel and slightly narrower under the
other two.

The channels differ in the jumps rather than the widths. At the last sample on both grids
the bulk channel carries 42–45% less velocity jump than the other two, the weakening of the
vortex sheet by its viscosity. The temperature variation is 22–23% smaller under the
default than under the Fickian channel. Under the bulk channel it is smaller again, by
21–25%. The two consistent channels' species layers are within 4% of each other on the
finer grid.

**Decision:** the default is retained. On the tube the partial-density channel converts
1e-5 of the pressure work to internal energy and leaves the species and velocity layers
as wide as the Fickian channel leaves them; the bulk channel
removes a hundred times more kinetic energy and nearly half of the vortex sheet's velocity
jump, and the Fickian channel moves heat across the contact through its enthalpy flux. No
measurement here decides which roll-up is right; that waits for a converged reference.

## The species validity band

`julia --project=. bench/archive/speciesband.jl` (about three minutes, serial). The cases are
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
advected slab with two-cell edges overshoots by the same amount at every N from 256 up,
while edges of fixed physical width produce no excursion once they span four cells. The
shocked interface's transient excursion is the same on the coarsest grid and the finest,
since the shock compresses the interface to the few cells the artificial diffusivity sets,
which a shock-capturing run has somewhere. Only the slab, whose interface is resolved over
`Np` cells by construction, converges with `Np`, and below the default Np = 7 it loses
positivity. The endpoints are smaller than the transients in every case and stay below
3e-3. With the mass-fraction bound off the shocked interface's excursion is about six
times larger. It was larger still before the partial-density channel at C_D = 0.1 became
the default, and C_D damps that ringing:

```
shock, C_Y = 0, before the partial-density channel   worst run 0.2   worst end 0.09–0.15
```

The validity band `StepControl.species_band = 0.05` is set between the bound-on and the
bound-off excursions. It sits above every bound-on excursion measured here but one, and
above the Mach 3 worst of −0.043 recorded under [C_Y](shocks.md#c_y-the-mass-fraction-bound). The
exception is the slab's transient at Np = 7, which only a `validity_interval` check or a
guard would see. The bound-off transient still crosses the band. With the band, the species
cases, both examples and the small configurations of `src/precompile.jl` run under the
default `:strict`; the He/CO2 example completes at its default 768 × 48.

A band at the bound's dead band, `ArtificialProperties.Y_tolerance = 1e-4`, is too tight:
the runs of the table that produce an excursion cross it at their endpoint. Under it the
Mach 1.5 case, the slab, the uniform advection and the He/CO2 shock tube of
`examples/shock_tube.jl` all fail a strict check, and a run with retries repeats its
trajectory four times before failing.

```
He/CO2 shock tube, nx = 384, ny = 16: 117 of 6144 points beyond the dead band
```

What remains permissive is the negative internal energy of a shock converging into a cold
or near-vacuum ambient ([below](shocks.md#negative-internal-energy-in-completed-runs)): the three Noh
geometries, the two anisotropic Cartesian Noh cases and Sedov. The cylindrical converging
shock of `examples/converging_shock.jl`, whose ambient is at p = 1, completes strict. It
does so at every nx tested, from 256 to 1024. An ideal gas at e < 0 has no temperature, so
no threshold can admit these cells honestly, and `:strict` stays the default: each of the
six cases bounds its closing inadmissible count and e_min in `test/validation.jl`.

## The gas-gas acoustic interface

```text
julia --project=. bench/archive/interfaceacoustics.jl art=on,off
julia --project=. bench/archive/interfaceacoustics.jl n=1600,3200 delta=4 cases=lh art=on,off
```

`interface_pulse` of `test/cases.jl`: a periodic line of air and a gas of SF6's density
ratio and γ at uniform p = T = 1, joined by tanh interfaces `delta` cells wide. A
right-running Gaussian pulse crosses one interface, from air into the heavy gas (`lh`) or
back (`hl`), under the default numerics, inviscid.

```
line        [0, 2]: air (R = 1, γ = 1.4) on [0, 1], heavy gas (R = 1/5.04, γ = 1.09) on [1, 2]
pulse       amplitude 1e-4, width σ = 0.04
impedance   sharp-interface solution R = ±0.32906, T = 1.32906 (lh) and 0.67094 (hl)
```

R and T are projections of the outgoing characteristics onto the exact pulse shapes at
their exact positions, energy is R² + (Z_i/Z_t)T², and err is max |p' − p'_exact| / amp.
`cells_t` is the transmitted width σ c_t/c_i in cells. Two-cell interfaces:

| case | art | N | cells_t | R err | T err | energy | err |
|---|---|---|---|---|---|---|---|
| lh | on | 400 | 3.1 | −8.52e-2 | −1.73e-2 | 0.9281 | 0.284 |
| lh | on | 800 | 6.3 | −3.39e-2 | −1.02e-3 | 0.9775 | 0.152 |
| lh | on | 1600 | 12.6 | −1.08e-2 | 5.89e-4 | 0.9938 | 0.079 |
| lh | on | 3200 | 25.2 | −3.18e-3 | 3.74e-4 | 0.9984 | 0.040 |
| lh | off | 1600 | 12.6 | −8.28e-3 | −5.37e-4 | 0.9939 | 0.081 |
| lh | off | 3200 | 25.2 | −2.16e-3 | −9.38e-5 | 0.9985 | 0.042 |
| hl | on | 400 | 20.4 | 4.07e-2 | 5.54e-5 | 0.9750 | 0.106 |
| hl | on | 800 | 40.7 | 1.25e-2 | 2.71e-4 | 0.9927 | 0.059 |
| hl | on | 1600 | 81.4 | 3.52e-3 | 1.44e-4 | 0.9981 | 0.031 |
| hl | on | 3200 | 162.8 | 9.78e-4 | 6.91e-5 | 0.9995 | 0.016 |
| hl | off | 1600 | 81.4 | 2.79e-3 | −2.42e-5 | 0.9981 | 0.030 |
| hl | off | 3200 | 162.8 | 7.05e-4 | 5.91e-6 | 0.9996 | 0.015 |

The reflection deficit falls by 3.1–3.6 per doubling at the finer steps, approaching the
second order in the interface width that a smooth transition gives. T is within 6e-4 of the
impedance value at N ≥ 1600 in both directions. The deficit is set by the width of the
interface, not by the discretization: at a fixed physical width the reflection error is the
same in four cells as in two on half the grid. The artificial properties move R less than
the width does, by 8e-3 on the coarsest grid and by under 3e-3 on the finer grids with
two-cell interfaces. The max-norm error falls at first order, consistent with a phase lag of
the pulses proportional to the interface width.

```
lh, fixed physical width      R err, art off   art on
four cells at N = 1600        −2.85e-2         −2.89e-2
two cells at N = 800          −2.86e-2
```

`test/validation.jl` guards the N = 1600 rows:

```
|R err| < 0.02,   |T err| < 2e-3,   |energy − 1| < 0.012,   err < 0.15
```

## The interface sharpening flux

```text
julia --project=. -t 1 bench/archive/sharpening.jl interface ratios=100,1000 gammas=0.3,1,3 widths=1 ungated=1
julia --project=. -t 1 bench/archive/sharpening.jl slab gammas=0.3,1
```

`ArtificialProperties.C_sharpen` (`reference/DESIGN.md`, "The species channel"): the flux of
Brill, Olson & Bokman (eqs. 68–71) on the partial-density channel, with Γ = C_sharpen · c,
ε = `sharpen_width` times the largest local spacing, the pair normals from smoothed
gradients and the gate on ε_g = ε + D_b/Γ. The paper fixes the form, the defaults ε = Δ and
Γ = max|u| with other values in a few of its cases, and the Gaussian-filtered normal of
eq. 69. It gives no step restriction, no treatment of a vanishing normal and no
localization. Everything else here is this code's choice.

```
the paper's other values   ε = 1.2Δ and 1.5Δ in two cases, Γ = 0.5 max|u| in one
```

**The shocked interface** (`shock_interface`, N = 400, t = 0.25, production defaults
otherwise; columns as in "Interface width against Brill, Olson and Bokman"; `ungated` holds
the gate open, which is the paper's unlocalized form):

```
config                  ratio | count | Y 5-95 V 5-95 V10-90 V grad |   TV−1    min Y steps
unsharpened               100 |    10 |  10.65   5.12   3.61   3.03 |  0.0021  -0.0093   701
C 0.3 w 1                 100 |     7 |   7.52   5.20   3.83   3.37 |  0.0603  -0.0312  1081
C 1 w 1                   100 |     7 |   6.47   5.15   4.09   3.60 |  0.0423  -0.0402  1927
C 3 w 1                   100 |    10 |   5.96   5.44   4.29   3.82 |  0.2732  -0.1137  4127
C 0.3 w 1 ungated         100 |     7 |   7.53   5.20   3.83   3.37 |  0.0628  -0.0293  1081
C 1 w 1 ungated           100 |     7 |   6.48   5.15   4.09   3.61 |  0.0195  -0.0404  1976
C 3 w 1 ungated           100 |     6 |   6.01   5.44   4.29   3.82 |  0.0539  -0.1156  4642
unsharpened              1000 |    12 |  11.08   7.01   6.63   1.30 |  0.0796  -0.0135   729
C 0.3 w 1                1000 |    25 |  24.94   6.84   6.43   1.73 |  2.3233  -0.2036  1706
C 1 w 1                  1000 | lost positivity
C 3 w 1                  1000 | lost positivity
C 0.3 w 1 ungated        1000 |    19 |  17.81   6.91   6.49   1.80 |  2.4579  -0.1784  1747
C 1 w 1 ungated          1000 | lost positivity
C 3 w 1 ungated          1000 | lost positivity
```

At R = 100 the flux brings the mass-fraction width down to near that of the ε = Δ
equilibrium, while the volume-fraction width barely changes. The sharpened runs ring and
undershoot where the channel alone does not, and take 1.5 to 6.6 times the steps. The gate
is open on the shocked interface, so the gated and ungated rows differ only through the
ringing it lets pass or cuts. At R = 1000 no setting improves on the unsharpened channel.
C_sharpen = 0.3 completes with the ringing measure at 2.3, and every larger value loses
positivity. The form without smoothed normals and clipped fractions (raw gradients, the
sensor's clamped mole fraction, the gate on ε alone) did not thin the interface at the
smaller constant and rang far more at the larger:

```
raw-gradient form   count 10 for C_sharpen ≤ 0.3, TV − 1 of 1.5 at C_sharpen = 1
```

**The smooth slab** (the composition slab of `bench/archive/falseactivation.jl`; its edges are
logistic profiles N·W/2 cells thick; `frac on` is the largest fraction of points at which
the flux was nonzero, `deposit` the L1 difference in Y_heavy from the unsharpened run):

```
molecular-weight ratio 5, one period at u = 1
   N  N·W/2   C |  |S|/rho c  frac on |  E sharp   E plain  deposit | steps sharp plain
  64   2.0   0.3 |   1.34e-02    1.000 | 6.04e-02 5.10e-05 6.04e-02 |   416   280
  64   2.0     1 |   4.43e-02    1.000 | 6.36e-02 5.10e-05 6.36e-02 |   734   280
 128   4.0   0.3 |   1.45e-04    0.234 | 1.88e-04 5.85e-07 1.88e-04 |   832   559
 128   4.0     1 |   4.82e-04    0.266 | 3.42e-04 5.85e-07 3.42e-04 |  1468   559
 256   8.0   0.3 |   3.30e-05    0.121 | 2.84e-05 9.55e-09 2.84e-05 |  1664  1118
 256   8.0     1 |   1.14e-04    0.137 | 5.41e-05 9.55e-09 5.41e-05 |  2936  1118
```

The gate does not keep the flux off a smooth slab. On the coarsest grid the edges are
interfaces by the gate's criterion and the flux sharpens them, with a deposit of 6e-2. On
the finer grids the edges themselves are gated off, but the flux still acts on 12 to 27% of
the points. Where the minority fraction approaches zero with zero slope, at the middle of
each plateau, the logistic thickness V(1 − V)/|∇V| is the distance to the extremum and
falls below 2ε_g within a few cells of it. The flux there is of the size of the minority
fraction, and the deposit falls about six times per doubling, but it is 300 to 6000 times
the unsharpened channel's error. The step count rises with the rate term on every row,
whether the gate opens or not, by 1.5 times at the smaller C_sharpen and 2.6 at the larger.
A composition profile that stays away from the pure fluids keeps the gate closed everywhere
and the right-hand side bitwise equal to the unsharpened one:

```
test/sharpening_tests.jl   V between 0.2 and 0.8 on 64 points
```

**Open.** The flux thins the R = 100 interface in mass fraction as the paper's
equilibrium predicts, but with the ringing and the undershoot above; it fails at R = 1000;
and the gate does not separate a smooth plateau edge from an interface tail. The
questions for the authors are the discretization of eq. 68 (whether the compressive term
is differenced as here, with the same compact divergence as every other flux), how Miranda
limits the step, and whether their runs localize the flux at all. The He/CO2 tube completes
on a coarse grid in under four times the steps, but on the example's default grid the
sharpened run had not finished after ten times the unsharpened wall time and was stopped.
The tube budget with the flux (`bench/archive/tubebudget.jl sharpen=`, which attributes it to its
own row) was not run.

```
He/CO2 tube              C_sharpen   steps or wall time
192 × 12, to 0.3 ms      0.3         157 steps
                         1           332 steps
                         off         93 steps
768 × 48, to 2.5 ms      0.3         not reached after 16 minutes
                         off         about 90 s
```
