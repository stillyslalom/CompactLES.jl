# Calibration: the defaults and how to tune them

This file holds the calibrated defaults of the artificial-property and
compact-filter settings, and says which setting to change when a run
misbehaves. It is capped at 300 lines; the measurements are in
[CALIBRATION_APPENDIX.md](CALIBRATION_APPENDIX.md), one section per instrument.
`bench/artcal.jl` reproduces the one-dimensional results on the cases of
`test/cases.jl` that `test/validation.jl` guards, and `bench/tgv_energy.jl` the
Taylor-Green results.

## The defaults

```
ArtParams(enabled = true,
          C_mu = 0.002, C_beta = 1.0, C_kappa = 0.01, C_D = 0.01, C_Y = 100,
          Y_tolerance = 1e-4, mu_sensor = :strain, beta_sensor = :strain,
          reduction = :sum, smoother = :gaussian, detector = :delta4,
          species_flux = :fickian)
Numerics(deriv = lele_d1_6(closures = :neutral3),
         filt = compact_filter(0.45, closures = :onesided), filter_interval = 1,
         filter_cfl = 0.35, filter_weighting = :none, cfl = 0.5,
         control = StepControl())
```

| Setting | Default | Status | Basis |
|---|---|---|---|
| `C_beta` | 1.0 | keep | Accuracy optimum near 0.4; 1.0 maximizes the spherical-origin CFL ceiling and is the one value viable under both detectors ([battery](CALIBRATION_APPENDIX.md#the-shock-battery)). |
| `C_kappa` | 0.01 | keep | The wall-heating trough under the default smoother sits at 0.01; zero loses spherical Noh under `:compact` ([battery](CALIBRATION_APPENDIX.md#the-shock-battery)). |
| `C_mu` | 0.002 | keep | Inert in one dimension; above 0.008 spherical Noh fails. Taylor-Green is consistent with the value and cannot select it ([Taylor-Green](CALIBRATION_APPENDIX.md#taylor-green)). |
| `C_D` | 0.01 | keep | The filter dominates interface broadening; a 64-fold sweep moves the width by 22% ([battery](CALIBRATION_APPENDIX.md#the-shock-battery)). |
| `C_Y` | 100 | keep | A shocked 2h interface rings to ±0.2 without the bound and ±0.013 with it ([battery](CALIBRATION_APPENDIX.md#the-shock-battery)). |
| `Y_tolerance` | 1e-4 | keep | A dead band that restores the unbounded order on a smooth profile touching 0 or 1 ([battery](CALIBRATION_APPENDIX.md#the-shock-battery)). |
| `mu_sensor`, `beta_sensor`, `reduction` | `:strain`, `:strain`, `:sum` | keep | The alternatives move no battery column past the fourth digit, or lose a converging geometry ([battery](CALIBRATION_APPENDIX.md#the-shock-battery)). |
| `smoother` | `:gaussian` | keep | Raises the spherical-origin ceiling from 0.15 to 0.4 in its sweep, runs 29% cheaper, and costs seven points of planar wall heating ([battery](CALIBRATION_APPENDIX.md#the-shock-battery)). |
| `detector` | `:delta4` | provisional | `:d8` improves six of seven battery columns and halves the wall deficit at high CFL, and lowers the origin ceiling from 0.3 to 0.25 ([battery](CALIBRATION_APPENDIX.md#the-shock-battery)). |
| `species_flux` | `:fickian` | provisional | `:bulk` removes the pressure error of an advected interface at a large density ratio and matches the default on a shocked sphere at a third more per step; the constants are fitted on the Fickian channel and insensitive under this one ([battery](CALIBRATION_APPENDIX.md#the-bulk-species-channel), [three dimensions](CALIBRATION_APPENDIX.md#the-bulk-species-channel-in-three-dimensions)). |
| `cfl` | 0.5 | keep | Use 0.3 or `StepControl(retries = 4)` for a converging shock at a spherical origin, whose ceiling is 0.3; walls and axes carry none ([CFL](CALIBRATION_APPENDIX.md#the-cfl-restriction-and-the-symmetry-cell)). |
| `StepControl.substep_cfl` | 0 (disabled) | opt-in | An absolute ceiling on refreshed refined-stage CFL, with collective rollback. Qualify a positive ceiling for the case; accepted startup transients can exceed the root target ([substep rates](CALIBRATION_APPENDIX.md#benchsubstepratesjl-refreshed-refined-level-rates)). |
| `deriv` closure rows | `:neutral3` | keep | Neutral at an inviscid slip wall, where the cascade rows grow a wall-normal velocity; the C6 rows carry a measured pseudospectral certificate ([certificates](CALIBRATION_APPENDIX.md#closure-certificates)). |
| `compact_filter` α | 0.45 | too strong | 0.49 fits at 128³ and at 256³ and clears the battery; the stability edge is α = 0.49875 at full strength ([Taylor-Green](CALIBRATION_APPENDIX.md#taylor-green)). |
| `compact_filter` closures | `:onesided` | keep | The cascade's F2 row caps every filtered wall at second order; the one-sided rows return the wall to the derivative closure's order and take 10 to 18 points off the planar Noh wall deficit, at two to three times the error on a reflection resolved over fewer than ten cells ([filter wall rows](CALIBRATION_APPENDIX.md#the-filters-wall-rows)). |
| `filter_interval` | 1 | keep | Redundant with α: the dissipation per unit time depends on the two only through (1 − 2α) / `filter_interval` ([filter dissipation](CALIBRATION_APPENDIX.md#the-filters-dissipation)). |
| `filter_cfl` | 0.35 | keep | Makes the filter's dissipation a rate, invariant to the CFL, to landing steps, to retries and to subcycling; clears the battery at its production CFL numbers ([filter dissipation](CALIBRATION_APPENDIX.md#the-filters-dissipation)). |
| `filter_weighting` | `:none` | keep | The volume-weighted form conserves no better on a closed line, is 17 times less conservative at an axis or a pole, and moves the Noh wall deficit in opposite directions at the axis and the origin ([non-uniform volumes](CALIBRATION_APPENDIX.md#filtering-on-non-uniform-volumes)). |
| `NSCBCInflowBC` `beta_t` | 1 | keep | The full transverse share admits an entering vortex with a third of the LODI error at every relaxation rate and is the only weight under which the imposed state follows its target through a transverse flow; it reflects an oblique pulse at 0.13 of the incident amplitude against 0.047 at the outflow's Mach-number weight ([inflow transverse terms](CALIBRATION_APPENDIX.md#the-inflow-transverse-terms)). |
| `NSCBCOutflowBC` `beta_t` | −1 (local Mach) | keep | Least pulse reflection near the Mach number; a vortex leaves most cleanly at 1 − M, measured at one Mach number only ([inflow transverse terms](CALIBRATION_APPENDIX.md#the-inflow-transverse-terms)). |

Every constant above was fitted under `compact_filter(0.45)` applied at full
strength every step, and the four that could depend on it have been re-swept at
α = 0.49 and under `filter_cfl = 0.35` without moving. For a strong shock the CFL
number sets stability more than any constant in the list.

## Which setting to change

Each entry gives the symptom, the setting to move and in which direction, and the
cost. The link is to the section holding the evidence.

- **A converging shock loses positivity at the spherical origin early in the
  run.** Lower `cfl` to 0.3, or keep 0.5 and set `StepControl(retries = 4)`,
  which recovers the case in about half the steps of a fixed `cfl = 0.15`. Do
  not lower `C_beta` below 0.5 for the origin
  ([CFL](CALIBRATION_APPENDIX.md#the-cfl-restriction-and-the-symmetry-cell)).
- **A spherical-origin run fails within tens of steps of a sharp start.** Warm
  the run from a profile resolved over three cells or more; the origin fold
  cannot take the singular start of Noh, which the cylindrical axis accepts
  ([fold order](CALIBRATION_APPENDIX.md#fold-order-and-geometry-limits)).
- **A resolved or smooth solution is over-dissipated.** Weaken the filter with
  `compact_filter(0.49)`, which fits at 128³ and at 256³. The filter is the
  sink, 37% of the Taylor-Green dissipation at 128³, and raising `C_mu` does not
  help ([Taylor-Green](CALIBRATION_APPENDIX.md#taylor-green)).
- **A smooth wave train or a contact is over-damped.** `C_beta = 0.5` keeps
  0.7% more Shu-Osher amplitude and an 18% narrower contact than 1.0, at the
  cost of half the spherical origin's timestep; `detector = :d8` keeps 1.1% more
  amplitude ([battery](CALIBRATION_APPENDIX.md#the-shock-battery)).
- **Lowering the CFL or writing output more often changes the solution.** Keep
  the default `filter_cfl = 0.35`, under which the filter's dissipation per unit
  time is invariant to the step. At `filter_cfl = 0` each pass dissipates a
  fixed amount, so more steps mean more dissipation
  ([filter dissipation](CALIBRATION_APPENDIX.md#the-filters-dissipation)).
- **A grid of high aspect ratio takes many small steps.** The scalar bulk
  viscosity sets the step from the finest direction. One coefficient per
  direction removes that penalty and makes vorticity in cold pre-shock gas, so
  it was measured and not adopted; the relaxed filter's weight is directional
  and carries no such penalty
  ([directional β\*](CALIBRATION_APPENDIX.md#directional-bulk-viscosity)).
- **Mass fractions leave [0, 1] at a shocked interface.** The default
  `C_Y = 100` holds a 2h interface to ±0.013 where it rings to ±0.2 without it.
  Resolve the interface over 4h for a quarter of a percent, or 8h for a clean
  profile; resolution, closures, detector, sensor field, CFL and filter strength
  each move the excursion by nothing
  ([battery](CALIBRATION_APPENDIX.md#the-shock-battery)).
- **A shocked species interface at a large density ratio fails, or an advected
  one drifts in pressure.** In one dimension at a density ratio of 100 the
  Fickian channel fails and `species_flux = :bulk` completes; the Fickian
  enthalpy flux moves the pressure of an advected interface where the bulk
  channel holds it to round-off, in one dimension and three; on a shocked
  sphere in three dimensions the two are indistinguishable
  ([battery](CALIBRATION_APPENDIX.md#the-bulk-species-channel),
  [three dimensions](CALIBRATION_APPENDIX.md#the-bulk-species-channel-in-three-dimensions)).
- **A passive interface broadens faster than the species diffusivity explains.**
  The filter is the broadening: a passive interface more than doubles in width
  with D\* off, and a 64-fold sweep in `C_D` moves the width by 22%. Weaken the
  filter rather than `C_D` ([battery](CALIBRATION_APPENDIX.md#the-shock-battery)).
- **Wall heating at a stagnation wall.** Keep the default filter closure rows,
  which took the planar Noh deficit from 60% to 50% at N = 400; `detector = :d8`
  lowers it further at high CFL. Raising `C_kappa` does not reduce it under the
  default smoother, and the deficit does not converge away with resolution
  ([filter wall rows](CALIBRATION_APPENDIX.md#the-filters-wall-rows)).
- **A long inviscid run between slip walls or symmetry planes grows a
  wall-normal velocity from nothing.** The run is on the `:cascade3` rows, which
  carry the slip-wall mode. Use the default `:neutral3` rows, which are neutral
  there, or `compact_filter(closures = :cascade)`, which damps the mode at a
  second-order wall defect
  ([the mode](CALIBRATION_APPENDIX.md#constant-annihilation)).
- **A slip wall limits the accuracy of a smooth run.** Replace `SlipWallBC()`
  with `SymmetryPlaneBC()` in a single unrefined, unstretched patch. The wall
  then carries no closure row and reads the interior order; the end node moves
  half a cell inside the plane
  ([symmetry plane](CALIBRATION_APPENDIX.md#the-face-centred-symmetry-plane)).
- **A run with same-level patches fails within its first steps.** The initial
  data has a discontinuity within one node of a shared patch plane, which the
  default ghost-reading interface rows do not survive; a shock arriving from
  the interior crosses the plane without incident. Move the plane, resolve the
  discontinuity over a few cells, or set `interface_rhs = :onesided`
  ([interface sensors](CALIBRATION_APPENDIX.md#benchinterfacesensorjl-the-sensors-and-the-filter-at-an-interface)).
- **A patched, refined or switching face needs a reflecting condition.**
  `SymmetryPlaneBC` is unavailable there; use `SlipWallBC()`, which carries the
  same flux contract at the cost of a closure row
  ([symmetry plane](CALIBRATION_APPENDIX.md#the-face-centred-symmetry-plane)).
- **A run on `:cascade4` derivative rows destabilizes at an inviscid wall.**
  Those rows need the F2 row the default filter closure removes, so pair them
  with `compact_filter(closures = :cascade)`. `:cascade3` and `:brady_livescu`
  take the default filter rows
  ([wall closures](CALIBRATION_APPENDIX.md#wall-closures-in-production)).
- **A wall that must carry more than third order, on a face that cannot be a
  symmetry plane.** Use C6 `:brady_livescu` under the default filter rows, which
  raises the smooth wall order from 3.17 to 5.88 on resolved initial data. C8
  `:brady_livescu` is not supported at a wall
  ([wall closures](CALIBRATION_APPENDIX.md#wall-closures-in-production)).
- **A wall solution loses its closure order once the artificial properties are
  on.** The detector reads a closed edge from the node-centred mirror of the
  interior, so a viscous or shear wall keeps its order. An inviscid slip wall is
  held at fourth order by the strain sensor's cusp, which
  `beta_sensor = :dilatation` removes
  ([sensor operators](CALIBRATION_APPENDIX.md#the-sensor-operators-at-walls)).
- **The wall order of a filtered smooth evolution is lower than the derivative
  closure's.** The filter's closure rows set it: the default one-sided rows read
  3.8 next to the wall where the cascade rows read 1.8, whatever the derivative
  closure
  ([evolution accuracy](CALIBRATION_APPENDIX.md#the-smooth-evolution-accuracy-matrix)).
- **A converging case fails below a CFL number rather than above one.** That is
  the filter, not a CFL restriction: under `filter_cfl = 0` a smaller step
  applies more passes per unit time. The default relaxation removes it
  ([filter dissipation](CALIBRATION_APPENDIX.md#the-filters-dissipation)).
- **The geometry ahead is unknown and a converging shock is possible.**
  `detector = :d8` improves most of the battery and both Cartesian ceilings;
  `:delta4` is the default only because the guidance for converging shocks
  rests on the spherical case
  ([battery](CALIBRATION_APPENDIX.md#the-shock-battery)).
- **The sensor phase is a large share of the step.** `compute_artificial!` is a
  quarter of the multicomponent right-hand side under the default `:gaussian`
  smoother, most of it the line solves that smooth one sensor per species.
  Cutting it means choosing a shared sensor over a per-species one, which is a
  numerics decision ([step cost](CALIBRATION_APPENDIX.md#operator-and-step-cost)).
- **The run ends on a state the EOS rejects.** Converging shocks carry six to
  eight cells of negative internal energy for the whole run and still reach the
  exact plateau; repairing those cells terminates the run. Use
  `validity = :permissive` with the default `floor_scope = :representable`, and
  bound the count in a guard
  ([CFL](CALIBRATION_APPENDIX.md#the-cfl-restriction-and-the-symmetry-cell)).
- **A smooth run is suspected of an artificial-property error.**
  `ArtParams(enabled = false)` skips the sensors and the coefficient
  calculation entirely. Taylor-Green at 128³ completes without them, which it
  does not without the filter
  ([Taylor-Green](CALIBRATION_APPENDIX.md#taylor-green)).
- **A published Cook-family parameter set is preferred to these constants.**
  `bench/artcal.jl brill2025` runs the smaller set of Brill, Olson and Bokman
  through the battery; it survives both converging geometries and belongs to
  its own scaling ([battery](CALIBRATION_APPENDIX.md#the-shock-battery)).
- **The ambient is cold.** κ\* is written as ρc/T_ion and is not singular in
  practice: the sound speed vanishes with the temperature at a floored cell.
  A cold ambient changes the count of cells the EOS calls inadmissible,
  since the precursor's negative internal energy is a fixed absolute amplitude
  ([battery](CALIBRATION_APPENDIX.md#the-shock-battery)).
- **Results differ across process grids.** Stay on `beta_sensor = :strain`. The
  switched forms carry a discontinuous compression switch and reproduce across
  decompositions eight orders less well than the strain sensor
  ([battery](CALIBRATION_APPENDIX.md#the-shock-battery)).
- **A run at a curved coordinate singularity does not conserve under the
  filter.** Keep `filter_weighting = :none`. Neither form conserves on a closed
  line; the volume-weighted one is 17 times less conservative at an axis or a
  pole
  ([non-uniform volumes](CALIBRATION_APPENDIX.md#filtering-on-non-uniform-volumes)).
- **Float32 or device runs.** Keep the default closure rows. In Float32 the
  Brady-Livescu rows floor one derivative at 1e-3 against the cascade's 1e-4,
  although a wall evolution floors near 3e-5 under either closure
  ([wall closures](CALIBRATION_APPENDIX.md#wall-closures-in-production)).
- **A three-dimensional CFL number looks small next to the literature.**
  `max_rate` takes the acoustic rate as the Euclidean bound `c · sqrt(Σ 1/h_d²)`,
  the linear limit on an isotropic grid, where
  [Pyranda](https://github.com/LLNL/pyranda) counts the sound speed once and
  reads √3 times this one
  ([CFL](CALIBRATION_APPENDIX.md#the-cfl-restriction-and-the-symmetry-cell)).

## Known limitations of the calibration

- The compact filter holds the solver together and has never been calibrated:
  every constant above is conditional on it
  ([Taylor-Green](CALIBRATION_APPENDIX.md#taylor-green)).
- `C_mu` is active but not fitted. A fit needs a case with an unresolved
  cascade; the Taylor-Green peak cannot select it
  ([Taylor-Green](CALIBRATION_APPENDIX.md#taylor-green)).
- At two species the per-species sensor machinery is a measurable no-op, and
  the species constants earn their cost only at three or more
  ([battery](CALIBRATION_APPENDIX.md#the-shock-battery)).
- A converging strong shock at the spherical origin is limited to `cfl` 0.3, at
  every resolution, by an excursion of the origin cell
  ([CFL](CALIBRATION_APPENDIX.md#the-cfl-restriction-and-the-symmetry-cell)).
- `filter_state!` is not conservative on non-Cartesian metrics under either
  weighting
  ([non-uniform volumes](CALIBRATION_APPENDIX.md#filtering-on-non-uniform-volumes)).
