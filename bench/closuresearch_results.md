# Fifth-order closure-search results (60da34b)

This is an archive of the search and its then-current solver measurements.
N6h subsequently changed the slip-wall flux contract. The current-solver
smooth-wall and shock remeasurement, including superseded cold-Noh failures,
is recorded under [N6j](../reference/CALIBRATION_APPENDIX.md#fifth-order-closures-under-the-dilatation-sensor).

Instrument: `bench/closuresearch.jl`. All rows are exact on monomials of
degree 0:5 by construction. Growth rates below are from the reduced injected
slip-wall acoustic operator on a unit domain, after eliminating the two fixed
wall-normal velocity degrees of freedom.

Usage:

```text
julia --project=. bench/closuresearch.jl mode=report sweep=false
julia --project=. bench/closuresearch.jl mode=report sweep=true
julia --project=. bench/closuresearch.jl mode=search objective=unfiltered passes=6 trials=160 seed=6033502 train_ns=17,31,51,79,101
julia --project=. bench/closuresearch.jl mode=search objective=filtered passes=3 trials=160 seed=6033502 train_ns=17,31,51,79,101 filtered_ns=51,101
julia --project=. bench/closuresearch.jl mode=search objective=filtered support=7 null_scale=0.01 passes=2 trials=80 seed=6033502 train_ns=17,31,51,171 filtered_ns=51,101 sweep=true
julia --project=. bench/closuresearch.jl mode=de start=combined population=32 generations=20 seed=6033502 train_ns=17,24,31,51,79 filtered_ns=17,24,31,51 lhs_radius=0.3 null_radius=0.002
julia --project=. bench/closuresearch.jl mode=validate
```

The search is a deterministic Gaussian local search, not a coordinate search.
It starts at BL, proposes Gaussian perturbations in 7 or 11 coordinates,
begins at scale 0.35, and multiplies the scale by 0.55 after each pass. For
seven-point support, `null_scale` separately scales the four moment-null
coordinates because their sixth-difference basis weights are large.

## Controls

The reconstructed Brady–Livescu T6 rows are neutral at the sampled sizes,
with `cond(A) = 1.192e3`. Their eigenvector condition number grows from 61.7
at N=17 to 1.73e3 at N=101. The reduced RK-plus-filter model reproduces the
production N=51 radius `1.0044345097`, validating the filter objective.
Direct basis-vector comparisons of the assembled derivative and filter
matrices against the production plan fill and line solve are printed by every
invocation. At N=51 their maximum-norm mismatches are `1.066e-14` for the
derivative and exactly zero for the filter.

## Rejected narrow-training search

Settings: unfiltered objective, seed 6033502, 4 passes, 160 trials per pass,
training N = 17, 31, 51. Parameters:

```
[6.4089241038953615,
 0.7617934113123362, 2.8621916354179087,
 0.3850784639705582, -2.41117011554896,
-1.9982359735375275, -0.8721341786371131]
```

This reduced `cond(A)` to 43.0 and was neutral on the training sizes, but the
held-out N=101 growth was `+0.1768113 c/L`. It established that even five
widely separated training sizes were needed before broader sweeps.

## Rejected unfiltered-only search

Settings: unfiltered objective, seed 6033502, 6 passes, 160 trials per pass,
training N = 17, 31, 51, 79, 101.

Parameters:

```
[6.151263016311877,
 0.5195089207724447, 2.750826276249132,
-0.052963961495487745, -1.8799327335582026,
-1.3133286079021156, -0.9951264983380909]
```

It is neutral (to about 1e-12) at the trained N = 17, 31, 51, 79, 101 and
reduces `cond(A)` to 141. It is not neutral over all line lengths: the sweep
over every N=12:200 finds growth `+0.6623355 c/L` at N=171. Production gives
filtered radii 1.0065282547 (N=51) and 1.0027256897 (N=101), worse than BL.
It retains fifth-order polynomial accuracy and smooth unfiltered evolution,
but cold planar Noh fails at step 316 (BL fails at step 36 in the same
qualification). It is preserved as `unfiltered_scheme()` only as a negative
control.

## Rejected filtered-objective search

Settings: combined unfiltered plus filtered objective, seed 6033502, 3 passes,
160 trials per pass, unfiltered training N = 17, 31, 51, 79, 101, filtered
training N = 51, 101.

Parameters:

```
[6.44549783494442,
 0.5895659645871965, 3.0445191662818054,
-0.29695415441823525, -1.3290740341334466,
-0.6671659659185682, -0.3155538974259181]
```

The filtered objective greatly reduces the filter instability but does not
meet the hard `rho <= 1 + 1e-10` gate: radii are 1.0000148042 at N=51 and
1.0000058667 at N=101. A held-out sweep also finds unfiltered growth
`+1.041406 c/L` at N=415. It is exposed by `candidate_scheme()` as an
experimental/rejected candidate for reproducibility, not as a production
recommendation.

## Seven-point RHS extension

`extended_fifth_order_family` adds one coefficient per row in the nullspace of
the degree-0:5 moments, using the sixth-forward-difference vector
`[1,-6,15,-20,15,-6,1]`. With the seven LHS coordinates this is an
11-parameter family; four zero null coordinates recover the BL seed.

The first bounded run used the combined objective, seed 6033502, 2 passes,
80 trials per pass, `null_scale=0.01`, unfiltered training N = 17, 31, 51,
171, and filtered training N = 51, 101. It produced:

```
[5.865108300073922,
 0.7148278140675014, 2.560882104279839,
-0.425957889629787, -0.7460618884054762,
-0.49589740860778764, 0.462617680701808,
 0.0002402835560477558, -0.0046302932937616675,
-0.0039561728150460515, -0.0005473929487322502]
```

It fails the gates: unfiltered growth is `+0.1798302 c/L` at N=79,
filtered radii are 1.0018929333 (N=51) and 1.0000000334 (N=101), and the
held-out sweep peaks at `+0.5604155 c/L` at N=174. An adaptive follow-up from
this vector (2 passes, 40 trials) promoted N = 79, 101, 171, 174 into the
unfiltered training set and found no improving proposal. This is a bounded
negative result, not an impossibility result.

## Outcome

No fifth-order candidate passed both the line-length sweep and the one-sided
filtered injected-acoustic gate. These finite searches do not establish
impossibility. They do show that neutrality at a handful of sizes is a poor
selection criterion, even when it accompanies a large conditioning gain.

## Feasibility-first differential evolution

The next search removed the conditioning term entirely. Its score is the
worst violation of the hard gates `max Re(lambda) <= 1e-8` and filtered
`rho <= 1 + 1e-10`. Deterministic differential evolution used seed 6033502.

Starting from the seven-point extension of the filtered-objective candidate,
population 32, 20 generations, LHS radius 0.3 and null-coordinate radius
0.002, trained on unfiltered N = 17, 24, 31, 51, 79 and filtered N = 17, 24,
31, 51. It reduced the normalized violation from `7.845207e6` to
`4.003618e6`; filtered N=24 and 31 remained the limiting grids.

An adaptive round started from that result with population 32, 20 generations,
LHS radius 0.15, null radius 0.001, and added unfiltered N=102. It found a
small-grid-feasible member:

```
[7.156459972509367,
 0.6323236119854221, 2.877900223287518,
-0.21138702841158744, -1.7063879918846345,
-0.6357547501751779, -0.32613617214582513,
-0.0003047194454015689, 0.004429430394889424,
 0.0026707913832941135, -0.0011088799189850822]
```

Every trained unfiltered growth was below `1.5e-12`; filtered excesses at
N=17,24,31,51 were at roundoff. Selected filtered held-outs N=79,101,171,257,
415 were also at roundoff. The broad unfiltered sweep then exposed growth
`+0.1647166 c/L` at N=47, while the minimum N=12 had filtered excess
`3.307893e-7`.

A final adaptive round promoted unfiltered N=47 and filtered N=12 (population
32, 30 generations, LHS radius 0.1, null radius 0.0005). It removed the N=47
growth on the training set but stalled at normalized violation `1817.034`,
set by filtered N=12 excess `1.818034e-7`. Thus the substantive feasibility
search produced no member satisfying both gates; conditioning was never part
of these differential-evolution objectives.

The exact final vector is exposed as `de_scheme()` under the deliberately
limited name “experimental filtered-only DE fifth-order closure”:

```
[6.956459972509367,
 0.6794866541559612, 2.677900223287518,
-0.22613805071814658, -1.9063879918846345,
-0.6489148076249647, -0.515310296896472,
 0.0005902020384870031, 0.005347848654867536,
 0.003081657613812831, -0.002108879918985082]
```

It is not a viable extent-restricted configuration. A filtered-only sweep over
every N=17:200 plus 257, 371, 415, 459 and 601 fails first at N=18. The model
includes both the acoustic block and the unchanged scalar filter branch for
entropy and tangential modes. With production relaxation weight
`min(cfl/0.35,1)`, the failures are:

- CFL 0.5: N = 18, 21; maximum 1.000816366521 at N=18.
- CFL 0.25: N = 18,19,21,22,23,24,26,27,29,30,31,32,33,34,35,37,38,40,41,
  42,43,44,45,46,48,49,51,52,53,54,56,57,59,60,62,64,65,67,70; maximum
  1.001029740929 at N=18.
- CFL 0.125: N = 18,21,24,26,27,29,30,32,35,38,40,43; maximum
  1.000334567273 at N=18.

No tested N above 70 fails, so N>=71 is a measured bounded filtered-only
configuration over these extents, not a proof or production qualification.
Sparse-grid filtered neutrality did not generalize to adjacent short lines.

## Corrected acoustic-only damping sweeps

Instrument: `bench/closuredamping.jl` with the six-point filtered-objective
rows of `candidate_scheme()`, the default one-sided filter, and the rank-one
wall pass at strength 0.1 applied to pressure and normal velocity only. The
reduced model now carries the unchanged scalar filter branch alongside the
acoustic block; the earlier sweeps predate that correction. Each run tests
every N from 14 through 200 plus N = 257, 371, 415, 459 and 601, 192 extents
in total, with the production relaxation weight `min(cfl/0.35,1)`.

```text
julia --project=. -t 1 bench/closuredamping.jl parts=sweep schemes=candidate strength=0.1 components=acoustic firstn=14 lastn=200 cfl=0.5 ns=257,371,415,459,601
julia --project=. -t 1 bench/closuredamping.jl parts=sweep schemes=candidate strength=0.1 components=acoustic firstn=14 lastn=200 cfl=0.25 ns=257,371,415,459,601
julia --project=. -t 1 bench/closuredamping.jl parts=sweep schemes=candidate strength=0.1 components=acoustic firstn=14 lastn=200 cfl=0.125 ns=257,371,415,459,601
```

Failures against the gate `rho <= 1 + 1e-10`:

- CFL 0.5: 1 of 192; N = 14 radius 1.000798320701; maximum at N = 14.
- CFL 0.25: 2 of 192; N = 14 radius 1.000305699970 and N = 15 radius
  1.000038823300; maximum 1.000305699970 at N = 14.
- CFL 0.125: 1 of 192; N = 14 radius 1.000139037562; maximum at N = 14.

The failing extents are the shortest lines the seven-node wall pass admits;
`damping_matrix` requires N >= 14 for one block. Including the scalar branch
does not add failures at the larger extents.

## Production qualification of `de_scheme()`

The final differential-evolution vector was run through the production
derivative and timestep. `bench/closurequalify.jl` and
`bench/closuredamping.jl` now accept the name `de` for it.

```text
julia --project=. -t 1 bench/closurequalify.jl schemes=de parts=polynomial
julia --project=. -t 1 bench/closurequalify.jl schemes=de parts=jacobian jns=79,101,171 jwalls=slip,dirichlet,noslip
julia --project=. -t 1 bench/closurequalify.jl schemes=de parts=smooth ns=97,193,385
julia --project=. -t 1 bench/closurequalify.jl schemes=de parts=stress
```

The normalized degree-0:5 moment defect is 9.995e-17. Production
differentiation of x^6 gives wall errors 2.762275e-04 (N = 17),
8.634284e-06 (N = 33), 2.698214e-07 (N = 65) and 8.432376e-09 (N = 129),
successive orders 4.99964, 5.00000 and 4.99992.

Timestep Jacobian at CFL 0.5, centered perturbations 3e-6, 1e-5 and 3e-5.
Radii at perturbation 1e-5:

| N | wall | filter=false | filter=true |
|---:|---|---:|---:|
| 79 | slip | 1.000000000168 | 1.000000001794 |
| 79 | dirichlet | 1.121213271953 | 1.074557089012 |
| 79 | noslip | 1.000000000127 | 1.000000000098 |
| 101 | slip | 1.000000000168 | 1.000000001977 |
| 101 | dirichlet | 1.121213271988 | 1.074557088965 |
| 101 | noslip | 1.000000000074 | 1.000000000208 |
| 171 | slip | 1.001061474880 | 1.000000002368 |
| 171 | dirichlet | 1.121213271979 | 1.074557089007 |
| 171 | noslip | 1.000000000108 | 0.999999999915 |

The slip and no-slip readings stay at the differencing floor across the three
perturbation sizes, apart from the unfiltered slip resonance at N = 171, whose
growth 4.718428e-01 per unit time is identical at all three; the filter
removes it. The Dirichlet growth is neither a floor artifact nor a filter
artifact: the rates are +2.334710e+01 (N = 79), +2.993218e+01 (N = 101) and
+5.088470e+01 (N = 171) unfiltered, +1.467386e+01, +1.881264e+01 and
+3.198149e+01 filtered, each identical across the perturbation ladder. The
`:neutral3` control at N = 79 and perturbation 1e-5 gives Dirichlet radii
1.000000000058 unfiltered and 1.000000001380 filtered.

Smooth wall evolution to t = 0.4 at N = 97, 193 and 385 against the periodic
mirror at the same spacing, successive wall-error orders:

| viscous | art | filter | CFL | orders |
|---|---|---|---:|---|
| false | false | false | 0.250 | 5.340 / 4.415 |
| false | false | false | 0.125 | 5.368 / 4.741 |
| false | false | true | 0.250 | 5.968 / 5.320 |
| false | false | true | 0.125 | 5.965 / 5.000 |
| false | true | false | 0.250 | 5.385 / 5.342 |
| false | true | false | 0.125 | 5.377 / 5.366 |
| false | true | true | 0.250 | 3.763 / 6.027 |
| false | true | true | 0.125 | 3.793 / 6.027 |
| true | false | false | 0.250 | 5.087 / 2.800 |
| true | false | false | 0.125 | 5.047 / 3.707 |
| true | false | true | 0.250 | 5.824 / 5.291 |
| true | false | true | 0.125 | 5.820 / 5.182 |
| true | true | false | 0.250 | 4.213 / 3.949 |
| true | true | false | 0.125 | 4.206 / 3.966 |
| true | true | true | 0.250 | 4.148 / 4.061 |
| true | true | true | 0.125 | 4.149 / 4.060 |

No combination failed, including the unfiltered ones. The finest pair reaches
error levels of 1e-13 to 1e-14 with the artificial properties off, where the
measured order is no longer resolved.

Shock stress at N = 200, CFL 0.3, 30,000-step ceiling:

- cold planar Noh: `SolverFailure(:negative_density)` at step 32,
  t = 0.008525, dt = 3.928e-5, minimum mixture density -0.028935.
- planar Noh started at t0 = 0.1: completes, with
  `StateReport(200 points, 8 inadmissible; rho_min = 0.9913129038239328,
  e_min = -0.023009457232873655)` under the permissive validity policy.
- Woodward–Colella: completes.

The cold failure is earlier than the unfiltered-only six-point rows' failure
(step 316) and close to Brady–Livescu's (step 36). The measured filtered-only
range of the reduced model therefore does not carry into a cold-started shock.
