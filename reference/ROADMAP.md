# CompactLES task roadmap

Open work for compressible, variable-density mixing and implosion, grouped by
track and ordered within each track. Closed items are listed at the end, one
line each with the commit that delivered them; the measurements behind them are
in [CALIBRATION_APPENDIX.md](CALIBRATION_APPENDIX.md) and the methods in
[DESIGN.md](DESIGN.md).

## Found by the examples

Defects found while building the Examples pages, filed here as found; each
example they block is held until the item closes.

- [ ] **A19 — Remove the grid-scale pattern tiled levels leave behind a
  converging shock.** On `bench/levelpattern.jl`'s spherical shock
  converging on an r-z quadrant, the mean undivided fourth difference of ρ
  in the shell behind the shock is, against the uniform run on the level's
  nodes, 1.00 times for one box and 1.19 times for fixed and for regridded
  tiles ([measurements](CALIBRATION_APPENDIX.md#benchlevelpatternjl-the-grid-scale-pattern-of-a-refined-level)).
  The regridded excess was A18's mechanism, removed by commit `04d3cff`;
  the sharpening flux's filtered gradients are smoothed over the level
  (commit `ef5a52d`). The tiled remainder is the explicit face rows of the
  filter and the derivatives, for which cheaper rows were measured and do
  not help. Holds the Spherical implosion example
  (`CompactLES_tutorial_protos/held/spherical_implosion/`).
  **Gate:** the example's refined run consistent with uniform 384².

## How to use this plan

- Unchecked boxes are open deliverables. IDs are stable and name
  dependencies, so an item keeps its ID when it moves between sections.
- [Order of work](#order-of-work) gives the sequence across tracks; within a
  track the items are listed in the order they should be taken.
- A measurement task is complete when its result and the resulting decision,
  including a decision to retain the current method, are recorded: one line
  under [Closed](#closed) naming the commit, its table in
  [CALIBRATION_APPENDIX.md](CALIBRATION_APPENDIX.md), and its default in
  [CALIBRATION.md](CALIBRATION.md) if a default moved.
- Close implementation tasks with the stated regression checks and the applicable
  repository validation gate. A numerical change requires an explained baseline
  update, not a relaxed test tolerance.
- Historical references to model debts 1/2/3 correspond to N1 / R3+N3 / V1;
  former Phase 2 items correspond to H1–H8, and G4b to S4.

## Order of work

V4 comes first: it sets the cost of every later commit. The next three are
independent and may proceed together, subject to one heavy run at a time.

1. **V4** trims the gate and the test suite, so that a commit pays only for
   the checks its change can affect.
2. **N4** settles `C_mu` before any external comparison: the filter fits
   behind the α = 0.47 default were made at `C_mu = 0.002`, and V1 compares
   the defaults.
3. **V1** runs the Pyranda comparisons and one Richtmyer–Meshkov experiment.
   It supplies N18's remaining reference and V2's use case.
4. **A7** draws the material boundary and the per-component field declaration
   before H3 adds the first new evolved field.
5. **H3**, then **H4**: two temperatures with an ionization closure and the
   electron–ion exchange, then electron and ion conduction through the implicit
   stage. H5's table readers are delivered; its adapters follow A7.
6. **N22** sharpens the species channel's interfaces. The flux is delivered
   off by default; its localization and density ratio 1000 remain, and
   become urgent when a case at density ratio 100 or more is in production.
7. Cluster campaigns run as allocation allows: S12's rzhound probe, S15's
   later stages, then S1 and S2 on rzadams.

The remaining items wait for a named target: H6, H7 and H9 for a radiation-,
laser- or burn-dominated case; H4a, H5a and H5b for cold DT and the ablator;
M1–M3 for an immiscible interface; S10 for a body that is not a coordinate
surface; H8 for a magnetized target.

## Numerics and calibration

- [ ] **N4 — Decide the artificial shear viscosity.** Scored on the history
  misfit, Taylor–Green at Re = 1600 fits best at `C_mu = 0` at 64³ and at 128³,
  under both the default and the near-off filter, and at 128³ a fifth of
  the physical dissipation lies off the grid
  ([Taylor-Green](CALIBRATION_APPENDIX.md#taylor-green); commits `c3f4f82`,
  `7e37a04`). One-dimensional shocks cannot determine the shear channel, so no
  case in the repository supports a positive value. Cook's 0.002 was never
  fitted: its evidence is decaying turbulence at Re_λ ≈ 720 on 192³ (Kang et
  al.), under a 2/3-rule dealiasing or the sharp `pyranda_filter()`, not the
  default filter
  ([where 0.002 comes from](CALIBRATION_APPENDIX.md#c_mu-the-shear-viscosity)).
  Either set the default to zero once the one-dimensional battery and a
  shocked three-dimensional case clear it, or rerun that decaying-turbulence
  case under the default `compact_filter` and `pyranda_filter()`, which tells
  whether a positive `C_mu` belongs to the method or to Cook's filter; a
  low-pressure vortex core is a separate, stability case. The filter fits
  behind the α = 0.47 default were made at `C_mu = 0.002`, so a change to
  `C_mu` reopens them. Retain `C_beta = 1` unless new evidence overturns its
  completed refit; record error and dissipation attribution.

- [ ] **N18 — Finish qualifying the species channel's transport consistency.**
  The three channels are compared on the one-dimensional battery and the
  shocked He/CO2 tube under one sensor, one operator set and one step
  ([the channel](CALIBRATION_APPENDIX.md#the-partial-density-species-channel)),
  and the default's uniform (u, p, T) invariance and entropy inequality are
  derived and measured ([DESIGN.md](DESIGN.md#the-species-channel); commit
  `64b8e14`). The invariance argument is for a uniform temperature. At a
  temperature jump each channel's drift is its own continuous model's volume
  source, the discrete flux identities hold as products, and no split form is
  needed ([the thermal contact](CALIBRATION_APPENDIX.md#the-contact-with-a-temperature-jump),
  commit `5281eac`). On the shocked tube the energy change is attributed per
  term with a closed budget, and the three layer widths are grid-set while the
  channels differ in the jumps across them (commit `5d529cb`). Remaining: a
  grid-converged reference, or the V1 Pyranda run, for the He/CO2 roll-up,
  which no measurement yet decides between the bulk and the partial-density
  form.
  **Depends on:** V1 for the reference, or a grid-converged run of this code.
  **Gate:** the appendix tables, the default retained or moved with its
  [CALIBRATION.md](CALIBRATION.md) line, and `test/validation.jl` baselines
  explained.

- [ ] **N22 — Add an interface sharpening flux to the species channel.**
  The artificial species diffusivity holds a shocked interface free of ringing
  only with a core of about three cells, and it leaves a tail of the heavy gas
  on the light side that the mass-fraction width reads as 10 cells at density
  ratio 100
  ([interface width](CALIBRATION_APPENDIX.md#interface-width-against-brill-olson-and-bokman)).
  Brill, Olson and Bokman (J. Comput. Phys. 542, 114366, 2025; eq. 68 of
  arXiv:2503.12680v2) pair the same partial-density channel with a
  sharpening flux whose equilibrium profile is V = 1/(1 + e^(−x/ε)), ε = Δ by
  default, which removes that tail. Add it to the species channel on the
  collocated compact operators, in divergence form, with its consistency
  terms on momentum and energy derived as the channel's are, and localized so
  that it acts only at a material interface and not on a smooth composition
  gradient. Define ε, the strength Γ and the step restriction; state which
  fraction it sharpens (volume or mass) and why; and keep single-species runs
  bit-identical. Pyranda has no implementation (upstream `b4e0afc`, every
  branch), so the form comes from eqs. 68–71: J_i = −ρ_i Γ(ε∇V_i −
  Σ_j V_i V_j n̂_ij), added to the channel flux before its consistency terms,
  with V the mole fraction for ideal gases; its discretization, the
  regularization of n̂ away from interfaces and the step limit are questions
  for the authors.
  M2 is the phase-field counterpart for immiscible materials on the staggered
  operators; share the analysis but not the requirement that M1 come first.
  Delivered, opt-in and experimental (commits `2da228c`, `64109d7`,
  `d1836dd`): `C_sharpen` and `sharpen_width` on the partial-density channel,
  sharpening the volume fraction with the ρ_k weights that keep a uniform
  (u, p, T) state uniform, with gated and regularized normals, its rate in
  `max_rate`, and the default path bit-identical. It thins the interface at
  density ratio 100 at a cost in ringing and steps, improves nothing at 1000,
  and acts on the plateaus of a smooth slab, so `C_sharpen` defaults to 0
  ([the sharpening flux](CALIBRATION_APPENDIX.md#the-interface-sharpening-flux)).
  Remaining: fix the localization on the smooth slab, then qualify at
  density ratio 1000; the discretization, normal
  regularization and step-limit questions to the authors; the He/CO2 tube
  and the N20 budgets with the flux on; MPI and GPU coverage.
  **Depends on:** nothing. **Gate:** the shocked interface at
  density ratios 100 and 1000 against the unsharpened channel (width in
  volume and mass fraction, TV−1, worst Y), the smooth-slab deposit of
  `bench/falseactivation.jl`, the He/CO2 tube, the N20 budgets, and
  `test/validation.jl` baselines explained.

- [ ] **N24 — Remove the grid-scale growth at the coordinate folds.**
  The area form of a fold's pressure term differs from the gradient at the
  grid scale, and the fold's one-step map grows at a rate per step
  independent of N
  ([grid-scale growth at the r-z axis](CALIBRATION_APPENDIX.md#grid-scale-growth-at-the-r-z-axis)).
  Delivered: the gradient form on θ-collapsed r-z (commit `17b0e44`) and on
  the spherical radial and polar momenta (commit `da0c733`); the axis runaway,
  the αf dependence of Noh's axis deficit and the origin's resolution
  requirement are gone, and Noh's origin ceiling rises from 0.3 to 0.5.
  Remaining: a strong blast through the origin still has a CFL ceiling of
  0.2 to 0.3 in either form, where the axis and the plane take 0.5, and the
  mechanism is open; the resolved-θ axis carries the defect unfiltered only:
  the isentropic vortex of the Axis-crossing vortex tutorial
  (`docs/literate/axis_crossing_vortex.jl`, 32 × 64) runs away at the axis
  with the filter off and loses positivity near t = 0.9.
  **Gate:** the origin blast at `cfl = 0.5` without retries.

- [ ] **N26 — Keep the internal energy positive ahead of a strong shock.**
  Measured (`bench/shockfoot.jl`, commit `105f787`;
  [negative internal energy ahead of a shock](CALIBRATION_APPENDIX.md#negative-internal-energy-ahead-of-a-shock)):
  on Woodward–Colella, Noh and Sedov the inadmissible cells are the compact
  scheme's odd-even undershoot ahead of a captured shock, whose depth is set
  by the jump and not by the ambient state, so an ambient internal energy
  below a few percent of the jump goes negative. Round-off does not cause
  it; the filter spreads it but damps it overall, and the artificial
  properties reduce it without removing it. Every repair measured, a
  per-point limit on the filter's correction and both scopes of the
  positivity floor, moves a validation error outside its guard, so the cases
  stay under `validity = :permissive`. The divergence is already a
  difference of face fluxes under node weights, so a blend toward a
  first-order flux (Hu, Adams and Shu 2013) applies as a correction at the
  faces it limits. In one dimension (`bench/positivity.jl`;
  [the face-flux limiter](CALIBRATION_APPENDIX.md#the-face-flux-limiter)) the
  limit on each Runge–Kutta stage's increment, which needs a face register
  per direction under the low-storage integrator, together with the limit on
  each filter pass leaves no inadmissible point on Woodward–Colella and planar
  Noh with both guards held; either alone, or the stage limit without the
  registers, does not. `Numerics(positivity_limiter = true)` carries both
  limits on the Cartesian metric, one patch, serial or decomposed, with ε at
  1% of the minimum ρ and ρe entering `run!` (commits `305de3f`,
  `0d05acd`); the other configurations are N32. Each cell's bound is 1% of
  the smaller of that minimum and its own value (commit `4d7fc5a`), which
  cuts the unguaranteed sides of a 2-D Sedov quadrant by two orders.
  The θ passes run as line sweeps over linearly indexed fields (commit
  `cb0b33a`), so the limited step costs 1.1 to 1.7 times the unlimited one.
  A `run!` entering with a marginally inadmissible point keeps the limiter
  on, with ε taken over the admissible points (commit `736d12b`).
  Remaining: wall nodes above cfl ≈ 0.32, where the first-order bound fails;
  a cell whose source terms alone take it below its bound, which is left
  unguaranteed and can end a limited run slightly negative; and the radial
  rate pass, the largest remaining cost on the r-z plane.
  **Gate:** the Sedov and Noh cases of `test/cases.jl` and the Supernova
  remnant tutorial under the default strict validity, their errors
  unchanged, and no inadmissible cell during the Woodward–Colella run.

- [ ] **N32 — Carry the positivity limiter to every configuration.** N26's
  limiter starts on the Cartesian metric, one patch, host storage. The goal is
  parity with the uniform Cartesian run on every supported configuration, and
  on the refined and device paths a run time below it. The one-dimensional
  radial lines folded at r = 0, the spherical origin and the θ-collapsed
  axis, carry it (commits `4d7fc5a`, `71f4fb1`): Sedov, Noh ν = 2, 3 and the
  Supernova remnant tutorial run under strict validity with no inadmissible
  point, at about 1.4 times the unlimited step. The r-z plane carries it with
  the axis, a symmetry plane at z = 0 and their corner (commits `d50697d`,
  `451285c`): Sedov leaves no inadmissible point and its shock radius is
  within 0.25% of the spherical line's in every direction, at about 1.7
  times the unlimited step. Same-level patches carry it under
  `interface_flux = :closure` (commit `7a191c3`): the face at the shared
  node is limited by one θ on both patches and carries its own register,
  and Woodward–Colella and planar Noh on two and three patches leave no
  inadmissible point. Refined levels carry it (commits `d3bd6af`,
  `af002dd`): boxes and tiles, static or regridded, nested, global step or
  subcycled, on the Cartesian grid, and one box per level on the radial
  grids. Coarse-fine end nodes and the parent's overwritten nodes are not
  held, and their inadmissible states are kept out of held cells'
  first-order fluxes; Woodward–Colella with a box or tiles, the Noh level
  rows and the A19 corner box leave no inadmissible point, at 1.15 to 1.6
  times the unlimited step. Under the limiter the interfaces close with the
  closure rows, with a warning: under `:ghost` the inviscid flux closes the
  interface with the gradient's rows (composite weight h at the shared
  node) and the rest with the divergence's one-sided rows (twice the end
  weight), so no one face form covers both. Two routes remain: write the
  difference as a node term in each cell's limit, as the radial lines'
  geometric part is, or difference the rest through the gradient rows too
  (`GHOST_FLUX_REMAINDER`, A17 item 2), which leaves one face form with
  weight h up to an interface end. The second keeps an interface's order
  and moved no shock measure by more than 2%; under it the limiter needs a register
  for the gradient-plan solves and open-end anchors. Either is measured
  before it replaces `:closure`, which lowers the interface's order. The parent's shell falls back to the
  multilinear, linear-in-time interpolant of the parent wherever the
  Lagrange chain or the Hermite blend leaves a node inadmissible and the
  fallback is admissible, on host and device storage, limiter on or off
  (commit `c85a315`); the correction pass gives both nodes of a face beside an
  unheld inadmissible node one first-order flux (commit `6debed0`). Every
  refined case of `bench/positivity.jl part=levels`, the blast with tiles
  included, then leaves no inadmissible point. Device storage carries it
  (commits `222894a`, `bacc29f`) on one patch, the radial lines, the r-z plane,
  same-level patches and refined boxes, bitwise against the host under the
  KernelAbstractions CPU backend and on an RX 6800 XT; a θ pass's tallies,
  a decomposed line's offsets around the Allgather, the interface θ
  exchange and the ε minima stage through the host. In order: stacked
  device tiles, whose batched right-hand side holds no per-tile registers
  (setup rejects a tiled device level), and tiles on the radial grids;
  then the remaining metrics and the
  tabulated and NASA-9 equations of state, whose admissibility test is not
  linear. Each extension's face weights come from its own rows: the metric's
  face areas, the fold's mirrored rows, the interface rows. The parent's
  derivative mask (A18) stays off under the limiter, whose stage face fluxes
  are a running sum of a divergence the mask does not conserve: registers
  holding the masked divergence, or restoring its dropped source at the
  masked nodes, broke the refined Noh and Woodward–Colella rows
  ([measurements](CALIBRATION_APPENDIX.md#benchmovingleveljl-disturbances-a-moving-level-carries)).
  **Depends on:** N26's stage in `src/`.
  **Gate:** per configuration, no inadmissible point on its strong-shock
  validation cases (Sedov and Noh ν = 2, 3 for the folds; the Noh level rows
  and the A19 reproducer for refinement) with their guards held, bit-identity
  with the limiter inactive, the device path bitwise against the host, and
  the step's cost against the unlimited run.

- [ ] **N27 — Decide the artificial species diffusivity's scale in slow flows.**
  The artificial species diffusivity scales with the sound speed times the
  spacing, so on a coarse grid a slow instability loses growth as the sound
  speed rises at fixed flow: in the Rayleigh–Taylor tutorial's setup
  (`docs/literate/rayleigh_taylor.jl`) on 73 × 12 nodes the growth rate
  falls by about 1.5% between ambient pressures 20 and 160, and turning the
  artificial diffusivity off removes the fall. A related effect at a shock:
  in the Richtmyer–Meshkov tutorial (`docs/literate/richtmyer_meshkov.jl`)
  the diffusivity widens an interface the shock has compressed below about
  three cells, the 5 mm layer compressed to 3.1 mm spreading to 4.5–4.9 mm
  on 24 rows and staying at 3.5 mm on 36, which lowers the growth rate by
  2%. Decide whether the species
  channel's scale should follow the flow speed where the flow is slow, or
  whether the dependence stays and is documented with its size.
  **Gate:** the tutorial's growth rate on the coarse grid independent of the
  ambient pressure to within its resolved-grid value, or the decision
  recorded with its measurement.

## Validation and verification

- [ ] **V4 — Cut the gate's wall time.**
  CI built the package image twice per job; with one build (commit
  `8ea9ebf`) each Julia 1.13 serial job fell by about 6 min and the
  numerics job from 49.1 to 44.5 min. The `allocate_state` barrier
  (commit `a8812af`) cut the serial testsets by 3–6%, and
  `runtests.jl timing=true` (commit `725bf7d`) prints each testset's
  compile and run time. Splitting the numerics job (commit `fa10874`) cut a
  CI run from 44.5 min to a median of 26.4 (20 runs, 22.3 to 32.2); the
  eight-rank job is the longest in 13 of 20 (median 24.1 min), the serial
  Julia 1 and pre jobs follow at 24.0. The eight-rank selection is
  `RANK_SHAPE_PHASES` in `test/mpi_tests.jl` (`rank_shape=true`), and the
  uniform half of the phase change at eight ranks, the C10 two-slab device
  layout and the Float32 NASA-9 device run are cut (commit `14c4c15`).
  Remaining:
  1. The two-rank job rebuilt its image on 4 of 6 runs with unchanged
     source, about 7 min each, while the eight-rank job reused its own.
     Read the "Report the restored package image" step and fix the cause;
     if it is the runner CPU, the fix is `JULIA_CPU_TARGET`. The image
     limit stays at Julia's default of 10, least-recently-used, one image
     per source tree.
  2. Trim the step-bound eight-rank phases and the serial Julia 1 and pre
     jobs.
  **Gate:** the measured wall of the core gate before and after, and the
  coverage of `bench/coverage.jl` held or any loss listed.

- [ ] **V1 — Complete independent solver and experiment comparisons.**
  Run CompactLES against Pyranda on Re = 1600 Taylor–Green and one
  Richtmyer–Meshkov shock tube, comparing dissipation histories, spectra and
  mix widths; `pyranda_filter()` and the He/CO2 comparison deck (commit
  `9e640ab`) and the record of Pyranda's numerics (commit `29364ad`) are in
  place. The experiment comparison is the Validation page
  `examples/jacobs_air_sf6.jl`: Collins and Jacobs (2002) and Jacobs and
  Krivets (2005), digitized, against a whole-tube calculation in the
  single-shock window, where the growth does not depend on `C_mu`.
  **Depends on:** N4 for the defaults compared.
  **Deliver:** reproducible inputs, reference provenance, uncertainty/error measures,
  and a docs validation section. Keep analytic validation, external data, and
  self-generated regression profiles explicitly distinguished.

- [ ] **V2 — Qualify the turbulent inflow on a V1 experiment.**
  `TurbulentInflow`, a seeded random-Fourier-mode target with prescribed
  Reynolds stresses and integral scale, reproducible across decomposition and
  restart, landed in commit `ebbe4d6`. Remaining: a V1 experiment use case,
  and the NSCBC relaxation rates (`eta_u`, `eta_T`) that admit the
  fluctuation without damping it.
  **Depends on:** V1.

- [ ] **V3 — Close gaps in automated verification.**
  In place: the temporal-order studies (commit `d8ea472`), with the
  once-per-step level restriction retained (commits `9370da5`, `e7dc4ca`;
  [temporal order](CALIBRATION_APPENDIX.md#temporal-order)); the R1–R4
  regressions and the skip accounting of the serial suite (commit `26ae691`);
  the Makie checks in CI's documentation job; and the scheduled validation
  battery (`.github/workflows/validation.yml`). Remaining: parallel-HDF5
  execution in CI where the required stack exists (verified locally under
  S6); each instrument's current transcript under `bench/results/<script>.txt`,
  so that an appendix table is a quoted output and never a retyped one, each
  landing with that script's next run rather than assembled from the tables it
  would replace. The runner taking medians over repeated processes, with an
  order-alternated paired mode for a before/after comparison, is
  `bench/repeat.jl` (commit `2290c83`), and its first paired transcript is
  `bench/results/derivcost.txt` (commit `a2d7311`).
  **Gate:** KA-on-CPU equality does not substitute for hardware-GPU tests
  under S1.

## Refinement for the production geometry

Node-centered patch and level coupling is interpolation and injection with
compact interface closures, not a conservative flux reconciliation; under the
N10 budgets no surface-flux correction is needed. The measured interface orders
are in the appendix's smooth-evolution
[accuracy matrix](CALIBRATION_APPENDIX.md#the-smooth-evolution-accuracy-matrix);
the designs and the fallback analysis are in [AMR_GPU.md](AMR_GPU.md). The
instruments for a new interface or boundary closure are the N6 matrix
(`bench/boundaryorder.jl`, gated in `test/convergence.jl`),
`bench/wallfilter.jl`, `bench/wallclosure.jl` and `bench/leveltransfer.jl`; a
candidate is qualified on the N10 budgets and the N11 shock crossings before
promotion.

- [ ] **S15 — Run a refined hierarchy across many nodes.**
  The target is a mixing layer or a plane shock resolved by a tiled level, or
  a nested pair of them, on thousands of ranks, with each rank's memory,
  traffic and setup cost following the tiles it holds rather than the level's
  tile count, which a weak-scaled run grows with the ranks. Stages, in order:
  1. Done with `LevelCoupling`: the level coupling moves point to point
     (`LevelCoupling`), a transfer's chains and boxes exist on its tile's ranks
     only, interface tags number the messages of one rank pair, and owner
     sizing starts at the count a tile admits; `bench/amr_scaling.jl` measures
     the weak scaling ([table](CALIBRATION_APPENDIX.md#benchamr_scalingjl-weak-scaling-of-the-level-coupling)).
  2. A nested hierarchy at scale, which needs S8's deep-regrid items
     (point-to-point migration of a moved survivor in place of the replicated
     gather over the run, rebalancing, the device backend).
  3. Decide the one-box path at scale. A tile's chain runs over its whole box
     on each of up to `n_cons` of its ranks, so a one-box level (`tile = 0`)
     over many ranks holds a box-sized chain per rank. Either partition the
     chain by fine block or have `AMR` choose a tile edge when the run spans
     many ranks; the second changes a default and needs a maintainer decision.
  4. Tagging at scale: `_tag_tiles` Allreduces a dense flag per lattice cell
     at every check; replace it with a sparse reduction if a cluster profile
     shows it.
  5. Qualify on rzhound with the system MPI: `bench/amr_scaling.jl dim=3`
     weak-scaled to at least a thousand ranks, then a production-shaped 3-D
     case (a shock through a perturbed heavy-gas interface, tiles tagged by
     the mass-fraction gradient and the sensor) against a uniform-fine
     reference at a size both fit, with restart on another rank count and a
     refined HDF5 dump (a spatial collection per patch with blanking, which
     the HDF5 `FieldWriter` rejects today).
  **Gate:** per-rank level memory and coupling wall flat in the weak-scaling
  table at the largest count available, the MPI suite at 2, 4 and 8 ranks,
  and bitwise agreement of every stage that only moves data.

- [ ] **A15 — Nest each refined patch inside one parent patch.**
  A level-ℓ ≥ 2 patch lies within one parent patch, and its box reads that
  parent's padded array after the parent's halo exchange (under subcycling
  the right-hand side's halos too, for the Hermite box), so ownership is a
  tree: a parent tile and its children migrate together, and co-located
  children couple on one rank. The tile lattice already nests children in
  one parent cell; the one-box path splits at parent tile boundaries. Level
  1 keeps the root coupling, since the root is one patch over many ranks.
  **Depends on:** S15 stage 2 and S8's migration, which it reshapes.
  **Gate:** the level rows of `test/convergence.jl` unchanged, the migration
  audit bitwise, the coupling's point-to-point traffic per rank falling in
  `bench/amr_scaling.jl` when children are placed with their parents.

- [ ] **A10 — Let a refined level reach every domain boundary.**
  A level reaches a wall, a symmetry plane, an NSCBC face and the r-z axis
  (N23), and crosses a periodic seam (commit `a642ad7`; the regrid and
  shapes only under `level_boundaries`). Remaining: one box spanning a
  whole period, and the target case, the 3-D vortex ring fired from a tube's
  top face into an air/SF6 interface and then shocked through it (the
  axisymmetric form is `examples/vortex_ring_shock.jl`), with the ring
  formed at the injector face rather than by a body force inside the
  domain. Under the body force one refined box follows the ring well, and
  lattice tiles of small edge cost many times more per step than the box
  for a compact feature.
  **Gate:** a shock and an interface followed into a wall and through a
  periodic seam, with the conservation and interface-reflection tests of a
  uniformly fine run; the vortex-ring case with the ring formed at the face.

- [ ] **S8 — Extend deep regridding and multiblock geometry when a target needs them.**
  Tiled regridding below level 1 landed with A12; add rebalancing,
  point-to-point migration and the device backend to it, which S15 stage 2
  and a nested capsule under N23 need. For geometric multiblock use, extend
  beyond the current slab layout with explicit adjacency and compatible
  geometry, and give a same-level patched run a checkpoint.
  **Depends on:** a concrete case.
  **Gate:** interface accuracy, conservation budgets, restart, and distributed
  consistency. Refinement across a fold stays forbidden pending a separate
  design; N23's axis tile (commit `c2e7a39`) is that design for the
  θ-collapsed axis only.

- [ ] **A14 — Add a conservative flux correction at coarse-fine faces.**
  The level coupling interpolates and injects and reconciles no fluxes. A
  strong shock leaving a static tile on Noh loses mass at the crossing, at
  the r-z axis and at a symmetry plane alike, and the plateau inside the tile
  falls to about the coarse level's
  ([the level tests](CALIBRATION_APPENDIX.md#testlevel_testsjl-the-level-hierarchy)).
  Regridding that follows the shock avoids most crossings, but the dendritic
  layouts of [AMR_GPU.md](AMR_GPU.md#long-term-target-dendritic-meshes) put
  shocks across coarse-fine faces in every converging run without AMR. Design
  a flux register for the node-centred coupling: the coarse face flux
  replaced by the time- and area-integrated fine flux, under global stepping
  and subcycling, over tiles and ranks, and on the metric's face areas.
  **Gate:** mass, momentum and energy to round-off on the Noh crossing rows
  of `test/validation.jl`; the smooth interface orders of
  `test/convergence.jl` unchanged or better.

- [ ] **A16 — Make the level coupling stable without the filter.**
  Under the default ghost fluxes and no filter, the one-step map of a
  refined run has a real eigenvalue above one at the edge of the restriction
  window, a density offset with a sawtooth component, whose rate rises with
  resolution; it sets the order of a second nested level. A filter pass at
  any tested cadence removes it; the closure rows are stable at lower order;
  interpolation order and a wider restriction margin do not help
  (`bench/couplingspectrum.jl`,
  [the level tests](CALIBRATION_APPENDIX.md#testlevel_testsjl-the-level-hierarchy)).
  Candidates: a dissipative band at coarse-fine faces active without the
  global filter, or a restriction window blended at its edge instead of
  injected up to a hard cutoff; each needs a stability argument checked on
  the spectrum.
  **Gate:** no eigenvalue above the uniform run's on the nest and one-level
  cases at N = 36 to 144 unfiltered; the unfiltered interface rows of
  `test/convergence.jl` unchanged or better.

- [ ] **A18 — Remove the disturbances a moving refined level leaves behind.**
  The parent's filter pass leaves out the residual of a restricted feature
  it cannot resolve (commit `04d3cff`), which brought the Advected bubbles
  tutorial's disturbance to 1 to 10 times the uniform fine grid's. A
  converging shock's level still carries 37 times the fine grid's ahead of
  the front. The cause is the parent's compact gradients and divergences
  carrying the restricted shock's tail to the nodes beside a coarse-fine
  face; the regrid fill only moves it inward with the level. Dropping their
  source at those nodes (`MASK_CHILD_DERIVATIVE`) brings the level to 1.16
  times the fine grid's, but a Sod shock crossing a level then drifts five
  to twenty times as far in composite mass, so the mask is off by default
  ([measurements](CALIBRATION_APPENDIX.md#benchmovingleveljl-disturbances-a-moving-level-carries)).
  Remaining: turn it on once A14's flux correction holds the composite
  budget, or decide the trade without it.
  **Depends on:** A14, or a decision.
  **Gate:** on the shock configuration, the level's |ρ − 1| more than 0.02
  inside the front within a small factor of the uniform fine grid's, with
  the two-level Sod drifts of `test/level_tests.jl` inside their guards.

- [ ] **A17 — Make refinement beat the uniform fine grid in time to solution.**
  `bench/amrwin.jl` times warm root-only, box, tiled and uniform-fine runs
  of a planar Sod tube, three blobs in a periodic box, the converging shock
  at the r-z axis and `bench/amr_cost.jl`'s 3-D blob, and splits each wall
  into point-steps and cost per point-step, with a phase profile and an
  account of what sized each root step
  ([measurements](CALIBRATION_APPENDIX.md#benchamrwinjl-time-to-solution-against-the-uniform-fine-grid)).
  The box is below the fine wall on the Sod tube and the 3-D blob, the tiles
  on the Sod tube only, and on the blob problems only the tiles reach the
  fine answer. The remaining work, by measured share:
  1. The level's step count. A parent's nodes that restriction overwrites,
     more than `LEVEL_BUFFER` inside the child, are held to a CFL ceiling of
     0.75 rather than the solver's (commit `f1d420c`; excluding them made
     the axis shock run away): the Sod level takes the fine run's step count,
     the tiled Sod tube runs at 0.73 of the fine wall, the shock falls from
     1115 to 763 root steps. On the box blobs and the 3-D box the artificial
     rate holds the level at 2.2 to 2.6 times the fine count, the 3-D box's
     root step bounded by its covered root nodes on most steps; the 3-D
     tiles take 1.56 times it under the level's hyperbolic rate.
  2. The tiles' cost per point, 1.3 to 1.5 times the fine grid's (2.4 on
     the shock): the shell fill runs only where a tile has a parent-fed
     face, each tile's right-hand side reads the level pass's velocity
     gradients, `Patch` is mutable (commit `3c2bc6a`) and the ghost-flux
     split runs one body per kind of component over the region the line
     fills read (commit `74343a7`). The level-wide artificial coefficients cost
     what a uniform patch's do per point. What remains is the ghost-flux
     divergence's second line solve per component on each interface
     dimension, whose one-sided rows carry their own left-hand side, so
     removing it is a change of numerics. Differencing the remainder through
     the gradient rows (`GHOST_FLUX_REMAINDER`, opt-in, host only) removes
     it, raises an interface's order on exact data from 3 to 6 and moves
     every shock measure by under 2%
     ([measurements](CALIBRATION_APPENDIX.md#benchinterfacesensorjl-the-sensors-and-the-filter-at-an-interface));
     making it the default needs its device kernels and the full gate.
  3. The cover: tiles of edge 8 and 16 cover the whole fine grid on both
     blob problems; revisit `tag_buffer`, the tag threshold and the tile
     edge against the error they buy.
  4. Robustness: the tiled shock ends with negative internal energy at the
     axis, and the box leaves the species band after regrid checks when
     the root under-resolves the edge (A18).
  **Gate:** on the Sod tube, the blob case and the 3-D case, the refined wall
  below the uniform fine wall at a composite error no larger than the
  tiles' recorded in the appendix; bit-identical numerics for any change
  that claims to touch cost only.

- [ ] **A13 — Refinement on the remaining metrics and layouts.**
  N23 takes the cylindrical metric and the symmetry plane. Refinement still
  rejects the spherical metric and stretched grids, and a same-level
  `patch_grid` run rejects folds, stretching along the patched dimension, a
  patch decomposition across a `SymmetryPlaneBC`, the `:d8` detector on every
  sensor, and the pentadiagonal filters, which refinement admits (commit
  `32b9de0`). Each extension needs its transfer closures and fold-aware
  gathers; order them by use.
  **Gate:** per extension, the refined-versus-uniform convergence rows of
  `test/convergence.jl`.

## HED physics

H1 is the principal infrastructure dependency for stiff diffusion. These are
separate capabilities with independent verification gates, not a claim that new
equation layouts alone make the current RHS a general multiphysics solver.
Ownership and coupling follow [the material interface design](DESIGN.md#material-and-physics-interfaces);
DT is the first cold-material path, with C/CH/CD coverage tracked separately in
H5b. Every integration preserves A7's ideal analytic execution contract.

LANL's [Riot](https://lanl.github.io/riot/main/index.html), a finite-volume
radiation-hydrodynamics code on Parthenon and Kokkos with singularity-eos and
singularity-opac, carries most of this stack, and its dependency order is the
one adopted here: an electron EOS and a mean ionization per material, then the
electron–ion exchange, then implicit electron and ion conduction, then
radiation coupled to the electrons, and the laser after the ionization it
absorbs on. Its regression problems supply several of the gates below. Three
of its choices are not taken: its stiff packages are operator-split in a fixed
order after the explicit stages, where H2 keeps them in one ARK residual; its
electron conduction has no flux limiter and its conduction solve no multigrid
preconditioner; and its P1 relaxation factor stands in for a radiation flux
limiter.

- [ ] **H1 — Complete the implicit diffusion infrastructure.**
  Follow [IMPLICIT.md](IMPLICIT.md): the conservative staggered compact
  operator, solved matrix-free by a Krylov method preconditioned with a
  multigrid cycle on the second-order operator. Delivered: the design (commit
  `99f0197`); stage 1, the staggered operator, periodic, walled, folded and on
  every supported metric (commits `52a58b8`, `32842fc`), accurate but not
  exactly symmetric at the cylindrical axis and the spherical poles, where no
  symmetric closure keeps fourth order (commit `b80d0fb`); and stage 2, the
  one-patch implicit stage solved by conjugate gradients, or GMRES at the axis
  and poles, under a line-relaxation V-cycle, whose iteration count is flat
  in grid and stiffness except on a spherical grid with an origin (commit
  `8819db1`). Remaining, each when a case needs it: a smoother for that grid
  (plane relaxation is the candidate), agglomeration of the coarsest level at
  large rank counts, and stage 4, the implicit stage on refined levels and on
  device storage, designed after H2 has measured costs. Keep operators and
  communication in core numerics with optional workspace allocated only when used.
  **Gate:** manufactured constant/variable-coefficient heat conduction in every
  supported metric, distributed residual/convergence studies, and freestream
  preservation. Variable coefficients require a policy for rebuilding the
  preconditioner's second-order operator.

- [ ] **H10 — Carry the implicit stage to the production configuration.**
  The implicit conduction of H1 and H2 runs on one host-storage patch
  without refinement, which the refined, decomposed capsule run cannot use.
  Design the stage for that configuration: the Krylov solve and its
  multigrid preconditioner over ranks and the compact interface solves, the
  implicit stage across same-level patches and refined levels (a composite
  operator, or level-by-level solves consistent with the level coupling and
  the subcycled step), and device storage. The single-patch stage remains
  the reference that verifies and benchmarks it. H3 onward inherits this
  constraint.
  **Depends on:** H2.
  **Gate:** the H2 pulse and temporal-order rows reproduced on a decomposed
  run, on a refined level against the uniform-fine run, and under the
  device backend; iteration counts flat in rank count.

- [ ] **A7 — Implement material interfaces with an explicit analytic fast path.**
  Follow [the interface design](DESIGN.md#material-and-physics-interfaces):
  separate local material queries, solver field adapters, and flux closures;
  define composition/energy metadata, typed results/status, and setup checks.
  The typed status of `state_admissibility` and `mixture_temperature_status`
  (commit `4be4ac0`) and the public EOS hooks (commit `3a80e1b`) are its first
  pieces. Preserve the existing ideal/stiffened analytic recovery and scalar
  transport methods, public constructors, and readable arithmetic. Introduce
  an internal `MaterialModels` boundary for standalone evaluators/readers,
  removing EOS field assumptions from adapters, as a closed set of concrete
  isbits models mirrored onto device storage, the form `Nasa9Mixture` already
  takes (commit `b78acd4`). Let a physics component declare its fields
  (evolved or derived, exchanged, checkpointed, written), so that H3's
  electron energy and H6's radiation energy reach the halo exchange, the
  checkpoint record and output from the declaration rather than through edits
  to the `n_cons` layout; the `StateDescriptor` metadata of Parthenon, on which
  Riot's packages register their fields, is the precedent. No package
  extraction or HED state allocation is required for this step; numerical
  formulas and default policies stay unchanged.
  **Sequence:** inventory current hooks and exercise independent queries with
  existing ideal, NASA-9 and transport models; qualify a tabulated model with
  H5 before stabilizing the rich interface. H1 and offline material/transport
  work may proceed independently; H3 uses the relevant A7 contracts as they
  mature.
  **Gate:** existing input decks and numerical baselines, independent query and
  adapter agreement, inference/allocations, and unchanged ideal workspace,
  gradient, launch and collective counts. Compare before/after full-step costs
  on single/multispecies ideal cases, including constant and analytic transport,
  plus NASA-9 and stiffened-gas controls; run `bench/jetcheck.jl` and
  `bench/audit.jl` and relevant CPU/MPI/device gates. Use paired measurements and
  repeated processes on the same hardware; no reproducible ideal-fluid slowdown
  or added per-point allocation is accepted to accommodate the rich path.
  Record measurements in the appendix and unavailable hardware explicitly.

- [ ] **H3 — Add two-temperature energy evolution with an ionization closure.**
  Extend the equation, flux and recovery interfaces for `T_ion` and `T_ele`:
  electron pressure, a mean ionization Z̄ per material (full ionization first,
  then the Thomas–Fermi fit of More that Riot uses), and an electron EOS per
  material (ideal electrons first), from which the electron density follows.
  Specify independent energy variables, binding-energy references, pressure
  work and the exchange mapping; prevent double counting in total-energy and
  subsystem equations. Declare where the artificial viscous heating of a shock
  is deposited, and whether the electrons carry an energy with its pdV work or
  an entropy; the choice sets the ion–electron partition at a shock. The
  electron–ion exchange (Landau–Spitzer with a declared Coulomb logarithm) is
  a pointwise implicit component of H2 and needs no global solve. Radiation
  enters with H6 as an energy field, not as a third material temperature;
  `T_rad` names the temperature derived from that energy. Declare gradient,
  boundary, wave-speed and regularization requirements at setup.
  **Depends on:** A7's contracts and H2.
  **Gate:** zero-dimensional electron–ion relaxation against its exact
  solution (Riot's `ei_relax`), total-energy conservation and the equilibrium
  limit, two-temperature linear acoustic modes, a two-temperature Sod tube
  under each electron-energy form, and the Shafranov two-temperature shock
  structure once H4 supplies electron conduction.

- [ ] **H4 — Add electron and ion conduction and ion viscosity.**
  Implement flux-limited Spitzer–Härm electron conduction, with its
  Z̄-dependent coefficient, and Braginskii ion conduction and viscosity through
  the implicit stage. Treat this as the hot-limit closure; wider material
  conduction consumes H5a/H5b state and validated coefficients without imposing
  their cost on scalar transport. The limiter makes the coefficient depend on
  the gradient, so the Picard iteration may stall, and JFNK with the limited
  flux is the fallback ([IMPLICIT.md](IMPLICIT.md#open-questions)).
  **Depends on:** H1–H3 and the dispatchable transport interface.
  **Gate:** the self-similar nonlinear heat wave of Zel'dovich and Raizer, a
  Coggeshall problem with hydrodynamics, limiter behavior at a steep front,
  H3's Shafranov shock, and coupled energy budgets.

- [ ] **H5 — Add tabulated EOS support.**
  Implement IONMIX reading and thermodynamically consistent interpolation/inversion.
  For SESAME, read the SP5 HDF5 files of the singularity-eos toolchain, which
  Riot reads, into isbits interpolators of this package: singularity-eos has
  Fortran and Python bindings but none for Julia, and a C shim over it would
  run on the host only. SESAME access remains subject to data licensing.
  Exercise A7's rich queries and shared interpolation/derivative results with
  a bounded table model; qualify forward/recovery adapters before stabilizing
  that interface. Declare frozen/equilibrium derivatives and supported
  temperature partitions; a 1T table alone does not supply a 2T EOS or the
  populations required by transport.
  Delivered, standalone and unexported: the IONMIX4/6 reader and writer
  with bilinear (ln T, ln ρ) interpolation, its own derivatives, column
  inversion and the `TABLE_` statuses (commit `5598c9c`), and the SESAME
  ASCII 2 reader and writer for records 201 and 301–306 on the same table
  machinery, with an opt-in free-energy-consistent bicubic Hermite route
  (commit `44ca2f8`). Remaining: the SP5 HDF5 form in the HDF5 extension, a
  check against a real table of each format, the A7 adapters and the
  artificial-conductivity scale. Thermodynamic consistency between nodes is
  a declared property of a table model, not a contract requirement: the
  adapters read the interpolant's own c² and c_v rather than recomputing
  them from p and e. A SESAME adapter carries an energy reference per
  material, since the zero of energy varies between tables, and falls back
  to the 306 cold curve below the first positive isotherm.
  **Depends on:** the relevant A7 contracts; reader/data work may precede them.
  **Gate:** table-node/interpolation checks, inverse consistency, derivatives,
  phase/domain handling, and an EOS-specific artificial-conductivity scale.

- [ ] **H6 — Add radiation diffusion, gray before multigroup.**
  Evolve a radiation energy per group, with `T_rad` derived from it, coupled to
  the electron energy through H2/H3's residual and energy contract, and define
  the opacity and group interfaces; opacity consumes the same material and
  population state as transport. Use flux-limited diffusion with a
  Levermore–Pomraning or Larsen limiter on the staggered operator, and solve
  the groups together in one Newton iteration, as Riot's multigroup solve
  does, rather than group by group. Riot's P1 form (the group energy at the
  cells, the flux at the faces, a relaxation factor in place of a limiter)
  maps onto the staggered operator and is the alternative to compare.
  **Depends on:** H1–H3 and suitable EOS/opacity data.
  **Gate:** Su–Olson nonequilibrium diffusion, a gray Marshak wave, the
  Lowrie–Edwards steady radiative shock coupled to the hydrodynamics, group
  convergence, positivity policy, and matter–radiation energy conservation.

- [ ] **H7 — Add laser deposition for a specified experiment.**
  Implement geometric-optics rays with refraction from the electron-density
  gradient, inverse-bremsstrahlung absorption into the electron energy,
  rank-to-rank ray transfer, and deposition through the source interface;
  Riot's laser package has this form.
  **Depends on:** H3's electron density and temperature and nothing beyond
  them, so it may precede H4–H6 when suitable material data are available.
  **Gate:** ray paths in a linear density ramp against the analytic
  trajectory, one-dimensional absorption against the analytic attenuation, and
  absorbed/deposited energy budgets.

- [ ] **H9 — Add thermonuclear burn with local deposition.**
  DT and DD reactivities from the Bosch–Hale fits as sources on the isotope
  partial densities, which the species channel already carries, with the
  charged products deposited locally in the ion and electron energies by a
  declared split and the neutrons escaping into a yield diagnostic. Local
  deposition is also Riot's model; charged-particle transport is a separate,
  later item.
  **Depends on:** H3 for the deposition split, and H4a's isotope labels.
  **Gate:** constant-temperature burn against the integrated reactivity,
  species and energy budgets closed with the deposited and escaped energy, and
  yield convergence in the step.

- [ ] **H4a — Add isotope-resolved H/D/T ion transport.**
  The coefficient foundation, a standalone Stanton–Murillo evaluator for fully
  ionized H/D/T interdiffusion with its regime diagnostics, is in
  `src/ion_transport.jl` (commit `7c33651`) and does not enter the runtime
  flux. Remaining ([the transport plan](TRANSPORT.md#staged-execution-and-gates)):
  the concentration, pressure, ion/electron temperature and electric-field
  driving terms with a defined ambipolar closure, the plasma cross-model
  verification, and the flux integration. Validate dense-plasma models
  separately from the weak-coupling limit.
  **Depends on:** A7/H3 for coupled flux and energy evolution, H1/H2 when stiff,
  and H5 or an explicit ionization closure outside the fully ionized regime.
  Coefficient and flux-reference work can proceed before these dependencies.
  **Gate:** independent H–D/H–T/D–T references, isotope permutation and trace
  limits, zero-net-mass flux, charge/current closure, driven isotope separation,
  total-energy budgets, and declared coupling/degeneracy validity ranges.

- [ ] **H5a — Evolve DT from its true cold state through heating.**
  Prioritize open isotope-resolved data and models covering the intended cold phase,
  molecular dissociation, partial ionization, and the ionized limit. Derive EOS,
  charge populations, electron density, and transport from a consistent material
  state, including latent, dissociation, and ionization energies where relevant.
  Declare transported isotope inventory separately from equilibrium molecular
  and charge populations; finite-rate populations require their own evolution.
  Couple neutral, charged, and electron collision channels without an arbitrary
  temperature switch or applying fully ionized coefficients to cold fuel.
  **Depends on:** H3–H5 and H4a; H6 for a radiation-driven front.
  **Gate:** a cold DT target heats under a conduction/radiation wave without a
  manually frozen region, with independently checked cold equilibrium, front
  propagation, energy accounting, phase/charge limits, and timestep convergence.
  State physical transport suppression separately from any numerical limiter;
  data coverage and transition assumptions follow [the transport plan](TRANSPORT.md).

- [ ] **H5b — Qualify additional HED materials and their mixtures.**
  Inventory open EOS, population and transport coverage for C, CH and CD with
  explicit composition, material form, phase, units and energy references.
  Reuse the A7/H5 interface and H5a methodology, qualifying one declared material
  and density/temperature path at a time. Compound tables and physical mixtures
  require explicit mixing/equilibrium closures; ideal species mixing and isotope
  substitution do not establish cold-material validity.
  **Depends on:** A7/H5 for runtime material queries; H3/H4 and H1/H2 for the
  selected coupled heating case, H6 if radiation-driven. Coverage and standalone
  reference work may begin before DT evolution is complete.
  **Gate:** source/provenance and missing-range inventory, independent cold and
  warm references, inversion/derivative and population consistency, then energy
  and front-convergence tests for each claimed material. Qualify fuel–ablator
  mixture/contact closures separately; preserve A7's ideal-performance gate.

- [ ] **A8 — Qualify optional material-package extraction after interface use.**
  Decide whether to retain the internal module or extract it after A7 and H5
  exercise the boundary. Require a concrete reuse, dependency/data, or release
  benefit; independent packaging is not a prerequisite for HED development.
  If extracting, keep material evaluation independent of CompactLES/MPI, supply
  CompactLES adapters through an optional extension, preserve supported imports,
  and move `thermodynamic_model` and the `_record_fields` overrides of the
  checkpoint record with the types. Avoid a separate interfaces package
  until multiple consumers need it.
  **Gate:** record the extraction/retention decision; for extraction, independent
  package tests, extension loading/compatibility and coupled numerical tests,
  A7's ideal-performance gate, and measured load/precompile cost. Package/module
  boundaries must still allow solver-owned fusion and optional bulk evaluation.

- [ ] **H8 — Add MHD only for a magnetized target.**
  Extend the equation and flux interfaces and evaluate GLM divergence cleaning
  for the finite-difference scheme.
  **Gate:** standard wave/shock problems, divergence-error control, and energy
  budgets. Plan magnetized transport separately if the target requires it.

## Multimaterial interfaces

Every species today is a miscible gas: the channel diffuses partial densities,
the sensor reads mass and mole fractions, and the mole fraction is the volume
fraction only for ideal gases at one pressure and temperature. `StiffenedGas`
is a single material. An interface between immiscible materials, a gas over a
liquid or fuel against an ablator, needs a phase that may hold several miscible
species, a volume-fraction closure recovering one pressure, and one temperature
if that closure is chosen, from the mixture energy, and a species flux that
sharpens the interface instead of diffusing it. No current target requires
this; the items wait for a case and follow
[the material interface design](DESIGN.md#material-and-physics-interfaces).
H5's tables are related infrastructure, not this closure.

- [ ] **M1 — Add a mixture thermodynamics that distinguishes phases from species.**
  Define a phase as a set of miscible species; allow molecular mixing within a
  phase and none between phases unless a physical model supplies it. Recover
  the common pressure and temperature from the volume closure Σ α_k = 1 and
  the mixture energy, and expose per material the density, internal energy,
  enthalpy, volume fraction, sound speed and the EOS derivatives the fluxes and
  the artificial conductivity scale read. Riot's multimaterial state is this
  closure: per-material partial densities and one bulk energy under
  pressure–temperature equilibrium, with a closed-form solution where every
  material at a point is an ideal gas; the PTE solvers of singularity-eos are
  the reference algorithms. Start with an ideal-gas and a
  stiffened-gas mixture; tables follow once the pure and trace limits and the
  inverse recovery are verified. The transport model then takes zero
  intermaterial diffusion with nonzero viscosity and conductivity, which the
  dispatchable transport of N8 admits as a rule and not a coefficient.
  Admissibility is the EOS's, through `state_admissibility`: no universal
  positive-pressure or positive-internal-energy rule, and an absent phase
  handled apart from a trace-material floor.
  **Depends on:** A7's queries and status types; H5 for tables.
  **Gate:** pure and trace limits, recovery residuals, hyperbolicity,
  insensitivity to an initialized trace fraction, and A7's ideal-fluid
  performance gate.

- [ ] **M2 — Add a conservative sharpening flux on the staggered operators.**
  The conservative diffuse-interface flux of Jain, Mani and Moin (2020) and
  its accurate variant (Jain 2022) balance a diffusion against the nonlinear
  sharpening term Γ(ε∇φ − φ(1 − φ)n̂); each is a divergence and needs its
  fields at the half nodes. Build it on the staggered compact operators of H1
  ([IMPLICIT.md](IMPLICIT.md)), not on a separate finite-volume path, which
  the design commitments exclude. Verify the discrete diffusion–sharpening
  balance at low order before raising it; derive the momentum and energy
  consistency terms from the complete species flux, sharpening included, for
  the equilibrium model M1 selects, rather than importing the incompressible
  phase-field equation; define the regularization parameters, the interface
  normal, the step restriction and the supported thickness in cells. Verify
  pairwise symmetry, Σ α_k = 1, permutation invariance of the phase labels,
  and that an absent phase creates no material and leaves the fewer-phase
  solution unchanged. Compare the two forms on shape error, grid alignment,
  robustness and cost.
  **Depends on:** M1; the staggered operators of H1 stage 1 are delivered.
  **Gate:** stationary and obliquely advected interfaces, a three-material
  junction and an equal-density interface, under the interface accuracy rows
  of `test/convergence.jl` and the N10 budgets.

- [ ] **M3 — Qualify the interfaces under-resolved and shocked, then add physics by target.**
  Deliberately under-resolved droplets and thinning ligaments, measuring mass,
  shape, breakup time, satellite sizes and mixing rather than snapshots, with
  the mean and maximum interface thickness reported and saddle points at
  breakup handled explicitly; a resolution and regularization-timescale sweep
  through a strong shock–interface interaction, confirming the sharpening
  still acts at the stable settings; acoustic reflection and transmission at a
  material interface against an independent reference; and, under refinement,
  how the grid-dependent thickness changes at a coarse–fine face and the mass
  and energy transient it leaves. Physics beyond the isobaric-isothermal
  closure is conditional on a target: assess whether instantaneous thermal
  equilibrium between materials serves the collapse, rebound or heat-transfer
  problem at hand, and compare against a five-equation reference where
  compression or thermal nonequilibrium matters, keeping that nonequilibrium
  distinct from H3's temperature split; capillarity, if required, as a
  pressure-balanced surface tension with its energy accounting, checked on
  Laplace pressure, spurious currents and capillary waves; phase change and
  interphase mass transfer as separately scoped models. Sharpening is not
  surface tension, and no discretization corrects an unsuitable closure.
  **Depends on:** M1, M2; H3 for a two-temperature comparison.
  **Gate:** the cases above with reproducible inputs, source revision,
  reference provenance and the supported parameter envelope, and negative
  results recorded with the simpler method retained where an extension does
  not improve the target metric.

## Scale, devices and performance

Detailed mechanisms and measurements remain in [AMR_GPU.md](AMR_GPU.md) and
[CLUSTER.md](CLUSTER.md). Patch AMR, device execution, stacked tile launches, and
opt-in Float32 already exist; the tasks below extend or validate them. S15 is
under [the refinement track](#refinement-for-the-production-geometry).

- [ ] **S12 — Remove the O(P²) replicated interface solve for many-rank scaling.**
  The reduced interface matrix is dense of order 2qP and every rank applies
  its LU to its own lines, about 8q²P² operations per line against 9qn for
  the local sweep. Per rank on a uniform 3-D grid that is a fixed 8q²N² per
  solve while the local work shrinks as N³/P³: an Amdahl term whose
  operation-count crossover is P ≈ 6.6 (q = 1) or 5.2 (q = 2) per direction
  at N = 256 and 10.4 / 8.3 at N = 1024, a few hundred to a thousand ranks.
  The 224-rank rzhound run with dims (8,7,4) was at that crossover along x.
  Per-step collective latency is not the limit: Allgathers span one direction's
  P ranks and the measured node scaling holds to four nodes
  ([CLUSTER.md](CLUSTER.md)).
  Stages, in order:
  1. Instrument first: `bench/reducedsolve.jl` (commit `3fb37a9`) times the
     local sweep, the Allgather and the reduced solve separately for both
     solvers, and its run on rzhound (`bench/slurm/s12_reducedsolve.sbatch`)
     remains.
  2. Done in commit `6bfeb55`: a pivoted band LU of the block-tridiagonal
     reduced matrix, periodic lines by an interleaved ordering, the dense LU
     kept only at P = 1; the reduced stage is faster than the dense solve at
     every P measured on the workstation and departs from serial within the
     dense solve's range.
  3. Only if the Allgather volume (2qP lines' worth per rank) then shows in
     the probe: a distributed reduced solve in which only neighbors exchange,
     the SPIKE recursion, which also removes the replicated factorization.
  Process-grid guidance follows from the same count: for a fixed rank count
  the per-rank reduced cost is proportional to the sum of the cubes of the
  per-direction rank counts, so near-uniform grids minimize it and slabs
  maximize it. **Depends on:** system-MPI cluster time for stage 1. **Gate:**
  the MPI suite at 2, 4 and 8 ranks, `bench/tgv_energy.jl` reproducing serial
  energy histories to round-off, and the node-scaling table re-measured at
  the largest rank count available with the probe's breakdown beside it.

- [ ] **S1 — Complete target-machine GPU measurements and resolve the wait stall.**
  Continue the rzadams/MI300A campaign using the existing measurements as a baseline,
  not as an unmeasured port. Characterize/resolve the intermittent ROCm wait stall
  in [rocm_wait_stall_report.md](bugreports/rocm_wait_stall_report.md) before interpreting
  performance changes. Record hardware, MPI stack, precision, synchronization
  policy, repeat variability, and correctness with every result. Profile a
  `Nasa9Mixture` step there too with `bench/device_nasa9.jl`, which has run only
  on the workstation GPU. The LLVM 22.1.8 `llc` in `AMDGPU_LLVM_Backend_jll`
  miscompiles a short-circuit choice between two kernel arguments on gfx942 as
  on gfx1030 ([amdgpu_shortcircuit_bug_report.md](bugreports/amdgpu_shortcircuit_bug_report.md));
  a target run uses an AMDGPU.jl release on the version-23 backend, or relies on
  the `ifelse` rule holding in every kernel.
  **Gate:** full single/refined/tiled runs on target hardware, including real device
  communication paths; report measured reproducibility rather than universal
  bitwise claims. Validate other advertised backends on their own hardware.

- [ ] **S2 — Measure and address compact-solve and transfer scaling limits.**
  Profile replicated dense interface solves, host/device fences and transfers,
  and the level coupling's point-to-point exchanges (S15) as line-rank count and
  region size grow. Compare gather-solve-scatter, on-device reduced solves and
  GPU-aware MPI only where profiles justify them.
  Measure stacked `max_rate` reductions before adding another launch optimization;
  revisit extra streams only if batching still leaves useful concurrency.
  The replicated interface solve has a cost model and a planned fix in S12.
  **Depends on:** reliable S1 timing. **Gate:** crossover data and preserved
  distributed accuracy, not a workstation-only speedup claim.

- [ ] **S3 — Complete production tile/ownership cost studies.**
  Build the 3-D implosion-shell benchmark at realistic tile sizes; measure
  per-imposition latency and repeat the previously cold tiled-overhead case warm.
  Quantify the benefits of rebalancing (commit `603af33`) and point-to-point
  migration (commit `3980639`), including root-work bias in measured tile
  weights; subtract that baseline only when its impact is established.
  **Gate:** repeated-process timings, memory, imbalance, migration cost, and accuracy
  against coarse and uniform-fine references on the target clusters.

- [ ] **S4 — Choose mixed-precision policy (former G4b).**
  Separate state, geometry, clock, solve, and accumulation precision explicitly;
  the savepoint clock is already held in Float64 in either precision (commit
  `bdf1ddf`). Use existing Float32 CPU/device histories and S1/S2 profiles to
  compare policies.
  **Gate:** conservation drift, thermodynamic recovery, closure conditioning,
  timestep progress, memory, and throughput; no CPU-default change from speed alone.

- [ ] **S7 — Validate thread pinning and cluster placement.**
  Wire pinning only after controlled target-cluster trials: fixed-rank comparisons
  on identical masks, repeated-process medians, and one rank per NUMA domain with
  pinned threads on the suitable Julia runtime.
  Preserve the SMT-sibling collapse reproducer and prepare evidence for site
  support. **Gate:** measured benefit over scheduler binding and repeatable launch
  guidance in [CLUSTER.md](CLUSTER.md).

- [ ] **S13 — Cut the remaining cost of the NASA-9 temperature inversion.**
  `recover_primitives!` under `Nasa9Mixture` runs a safeguarded Newton solve
  per point and per Runge–Kutta stage, which makes the recovery several times
  as costly as under `IdealMixture` and the CPU step a third dearer
  ([NASA-9 on the device](CALIBRATION_APPENDIX.md#benchdevice_nasa9jl-nasa-9-on-the-device)).
  The interval table is isbits and the
  inversion and every per-point species loop share the powers of T (commits
  `9e0f126`, `405037f`, measured with `bench/nasa9_inversion.jl`). The
  convergence criterion is eps^(2/3), floored at 1e-10 (commit `dc67f71`), which
  saves most of one iteration, and the iterates stay inside the fitted range
  (commit `3ede230`). What remains is a warm start from the stored
  `T_ion` field, which would cut the Newton iterations per point. It trades away
  the state-only seed that `mixture_temperature_status` documents for
  bit-for-bit agreement between serial and decomposed runs and for restart
  independence; only worth it if the model is still far from the ideal-gas
  step cost. Keep the polynomial powers literal (`T^4`): a repeated product is
  not bit-identical to the library power and moves every baseline for nothing.
  **Depends on:** a baseline decision.
  **Gate:** the core gate with explained baseline updates; time the inversion
  before and after at four species in the same session.

- [ ] **S11 — Fuse each compact line solve into one threaded region.** Low
  priority: on the 2-D case that motivated it, `mpiexec -n 8 -t 1` already
  beats what fused threads would reach, so this waits for a setting in which
  threads are the only parallelism, such as the host-side interface stage of
  a device plan.
  A line solve is three regions today (fill, solve, scatter) and each carries
  a spawn/join floor and a barrier, with the solved block crossing cores
  between them. On a 37k-point planar case at eight threads that granularity,
  not bandwidth, is the loss against ranks: many short regions per RHS, and
  ranks faster per step than pinned threads ([CLUSTER.md](CLUSTER.md),
  hybrid-desktop paragraph). A prototype in which each thread fills, solves
  and scatters its own chunk of lines in one region sped up the x-direction
  filter solve at `-t 8`, at parity serially and bitwise identical, provided
  the 16-wide column blocking of `solve_cols!` is kept inside each chunk. The transposed y/z path fuses the same way over
  chunks of x-columns, since a chunk's fill, row-sweep solve and scatter are
  all contiguous in it. Where the dimension is decomposed the reduced
  interface stage is a collective between the local solve and the spike
  correction, so the fusion there is two regions, not one, and the collective
  stays outside both. The `@threaded` docstring names this reorganization as
  the condition for re-measuring the threading backend; do that after, not
  before. **Gate:** the core gate, `bench/jetcheck.jl` and `bench/audit.jl`
  before and after, bitwise agreement of the serial suite, the MPI suite at
  2, 4 and 8 ranks, and the 768×48 thread table re-measured pinned.

## Geometry

- [ ] **S9 — Finish azimuthal mode truncation.**
  Follow [MODE_TRUNCATION.md](MODE_TRUNCATION.md). Stages 1 and 2, cylindrical
  (`polar_truncation`, off by default) with θ serial or decomposed, are
  delivered (commits `c5ac69f`, `851503e`, with the checkpoint record in
  `4fa1a55`); calibration/defaults and the spherical azimuth remain. The rate
  cap also caps the θ rate that `filter_weight` reads, so the θ filter pass
  weakens wherever the inner rings set that rate; the calibration measures it.
  On the axis-crossing vortex (`bench/axisvortex.jl`) the error grows with the
  number of truncated steps, and at κ = 2 the second ring's limit of 2 holds
  the density error near 45 times the untruncated run's where κ = 1 comes
  within a factor 1.33 ([the measurement](CALIBRATION_APPENDIX.md#azimuthal-mode-truncation)).
  **Gate:** conservation and mode errors, pole/axis behavior, and achieved CFL/cost
  benefit. Keep this acoustic/geometric restriction distinct from the diffusive
  restriction addressed by H1.

- [ ] **S10 — Add immersed boundaries on demand.**
  Follow [IMMERSED.md](IMMERSED.md): level-set geometry and graded post-stage
  Brinkman/reset blend; validation/calibration; normal extrapolation refinements;
  then moving bodies. Include force and heat-flux accounting.
  **Depends on:** a target needing non-coordinate-surface geometry; the first stage
  need not wait for HED or multiblock extensions.
  **Gate:** the design's stage-specific accuracy, stability, and force/energy
  budgets, followed by backend/AMR compatibility checks.

## Deferred scope and non-goals

- **Wall-resolved/wall-modeled LES:** deferred until a target requires it;
  current no-slip support serves verification and does not establish wall-model fidelity.
- **Radiation transport beyond diffusion** (discrete ordinates, Monte Carlo):
  deferred until a target shows H6's diffusion inadequate.
- **RANS mix models** such as Riot's BHR-3.1: a comparison for resolved mix
  widths, not a model to carry.
- **Symbolic frontend:** optional only after hand-written equation sets define a
  usable interface; a macro may emit typed code, but runtime PDE-string evaluation
  remains excluded.
- **Excluded:** cell-by-cell/oct-tree AMR, unstructured meshes, a parallel Godunov
  solver path based on Riemann solvers/flux limiters, and general-purpose CFD scope.
  Instrumented state repair remains allowed and must retain intervention budgets.
- **Design commitments:** compact structured operators, patch-based refinement,
  explicit geometry/collective contracts, and measurable accuracy for mixing and
  implosion. Revisit a restriction through a concrete case and a documented design.

## Closed

### Runtime and boundary correctness

- [x] **R1** — Diagnostics and field output no longer disturb the artificial
  coefficients the next timestep check reads (commit `4be4ac0`).
- [x] **R2** — Endpoint conversion and progress checks turn an unrepresentable
  advancing step into a diagnosed failure instead of a stalled clock
  (commit `4be4ac0`).
- [x] **R3** — An EOS-aware validity policy checks accepted and returned states
  under `:strict`, `:permissive` or `:repair` (commit `4be4ac0`).
- [x] **R4** — The NASA-9 inversion reports convergence and extrapolation status
  instead of returning an unchecked estimate (commit `4be4ac0`).
- [x] **R5** — `correct_flux!` imposes the adiabatic impermeable no-slip wall's
  species and energy flux contract on the assembled flux before divergence
  (commit `51d7b2a`).
- [x] **N28** — `NSCBCOutflowBC` carries the first-order Bayliss–Turkel
  curvature term at an outer radial face, and the oscillating-sphere dipole
  reflects 0.05 of the plane form's 1/(2kR) (commit `6c97f4f`).
- [x] **N29** — The inflow face's transverse terms carry the u_r/r of the
  collapsed θ, so the r-z face beside the axis no longer grows vorticity
  after a shock leaves through it (commit `f74b76f`).
- [x] **N31** — The vorticity burst at the inflow face beside the axis was a
  pressure pulse crossing the face while the face held u_r against its
  radial gradient; the transverse pressure gradient of the inflow's
  transverse terms now takes the weight min(β_t, M) (commit `7187b27`).
- [x] **N33** — The r-z axis face carries −h² ρ u_r′(0)/12, a twelfth of the
  first node's mass rate and the rate of change of the midpoint rule's own
  error at the axis; the scheme is unchanged, and with an (h²/24) q₁ end
  correction the r-z mass is conserved to fourth order (commit `3742897`).
- [x] **N30** — The outflow's transverse terms carry a collapsed transverse
  dimension's curvature row, as the inflow's do; the default dipole
  reflection is unchanged at four figures (commit `d5fd609`).

### Filtering, regularization, boundaries and species

- [x] **N1** — The compact filter default is `filter_cfl = 0.35`, with the
  relaxed formulation measured invariant to the step (commit `dbe2899`); α is
  0.47 since commit `ccb7e81`.
- [x] **N2** — Volume-weighted filtering on non-uniform metrics was measured
  against the unweighted default and the current method retained (commit `811d382`).
- [x] **N3** — `run!` primes the artificial coefficients, which removed the wall
  and axis CFL ceilings; the spherical origin keeps its own (commit `c407e0b`).
- [x] **N5** — Directional bulk viscosity was measured on anisotropic Noh cases
  and rejected; the scalar form is retained (commit `7dd161f`).
- [x] **N5a** — `filter_weight` reads each direction's own hyperbolic rate rather
  than the rate that sized the step (commit `d1de5cc`).
- [x] **N6** — The wall and interface accuracy probes became the smooth-evolution
  matrix of `bench/boundaryorder.jl`, with its orders gated in
  `test/convergence.jl` (commit `a1d7660`).
- [x] **N6a** — `compact_filter` defaults to `closures = :onesided`, which cuts
  the planar Noh wall deficit at the cost of a sharper reflection
  (commit `d611bae`).
- [x] **N6b** — C6 `:brady_livescu` is a supported wall configuration within
  measured limits, C8 is not, and the default is unchanged (commit `a619a6e`).
- [x] **N6c** — The constant-annihilation audit closed with no change and found
  the slip-wall mode of the cascade closures (commit `02c34eb`).
- [x] **N6d** — The `:neutral3` rows, spectrally neutral at an inviscid slip
  wall, became the C6 default, with the cascade rows kept at interfaces
  (commit `1e76f50`).
- [x] **N6e** — The `:delta4` detector reads past a reflecting wall from the
  node-centred mirror of the interior instead of clamping the index
  (commit `60da34b`).
- [x] **N6f** — The detector takes the half-offset mirror at a coordinate fold
  for an even field as well as an odd one (commit `abe3d8d`).
- [x] **N6g** — `wall_closures` gives the `:gaussian` smoother and the `:d8`
  detector wall rows folded from their own interior stencils (commit `ae574ca`).
- [x] **N6h** — `SlipWallBC` imposes the symmetry plane's flux contract on the
  assembled wall-plane flux (commit `e2a6c4f`).
- [x] **N6i** — The aligned Noh transverse mode is a shock interaction rather
  than an autonomous wall mode, and its guard is retained (commit `f81b1c6`).
- [x] **N6j** — The fifth-order closure candidates were re-measured under
  `beta_sensor = :dilatation` and the production choices retained
  (commit `de30ccc`).
- [x] **N6k** — The neutral rows carry a measured pseudospectral certificate and
  are the `:neutral3` defaults of C8 and C10 as well (commit `de30ccc`).
- [x] **N6l** — `SymmetryPlaneBC` folds a slip wall on a face-centred mirror and
  reads the interior order, for a single unrefined patch (commit `8a5bdfe`).
- [x] **N6m** — C6 `:brady_livescu` takes every start of the planar Noh
  ladder on the current solver, the singular one included, and its remaining
  failures are CFL ceilings that no initial-data measure orders, so the
  resolved-start requirement is lifted by measurement and no initial-data
  check is added; C8 `:brady_livescu` stays unsupported at a wall (commit `4faed86`).
- [x] **N7** — `NSCBCInflowBC` carries the Yoo–Im transverse terms on every
  incoming wave, at the full share by default, measured on an oblique pulse
  and an entering vortex at three Mach numbers (commit `d3d3f30`).
- [x] **N8** — Dispatchable CEA transport supplies temperature-dependent mixture
  viscosity and conductivity, with unity-Lewis diffusion or mixture-averaged
  diffusivities from supplied binary data (commit `6594afd`).
- [x] **N8a** — Validated, species-labelled neutral pair fits feed the
  mixture-averaged solver closure with strict collective domain rejection,
  including subcycled AMR, with the sources kept distinct (commit `711cecf`).
- [x] **N9** — The bulk species channel was measured in three dimensions
  against the Fickian one, its constants confirmed, and its entropy inequality
  restated as a continuous-model property; `:fickian` stayed the default
  (commit `7a86a2c`).
- [x] **N9a** — `:partial_density`, the partial-density channel of Brill, Olson
  and Bokman with its consistency fluxes on momentum and energy, is the
  default; `:bulk` remains for density ratios of 100 or more and `:fickian`
  reproduces Cook's form (commit `a2afda2`).
- [x] **N19** — False activation on smooth fields was measured, and the
  species fields are sensed with `:d8` at `C_D = 1` by default, every other
  sensor keeping δ⁴ (commits `ee1087b`, `0a64d6f`, `7a45c8c`, `072d184`).
  Reduced filtering waits until the discretization is shown stable without it.
- [x] **N20** — The boundedness mechanisms are written down, the sweep sees
  regridded and subcycled states, an acoustic pulse meets the interface
  impedances, and the composite budgets are attributed per mechanism with the
  boundary-flux closure verified; the coarse–fine drift converges and needs
  no conservative correction (commits `5dbb5ff`, `2a5ece9`, `a889b61`).
- [x] **N21** — The composition clip of the repair acts only outside
  `StepControl.species_band`, which now bounds both sides through a test shared
  with the validation sweep (commit `0293798`).
- [x] **N25** — The annulus slip wall's second order was the data's: the
  standing wave breaks the curved wall's second compatibility condition, and
  a compatible pulse converges at the Cartesian wall's order (commit
  `6fa4131`).

### AMR numerics

- [x] **N23** — Refined levels reach the r-z axis, the axis/z = 0 corner, a
  symmetry plane, walls and NSCBC faces, regrid and nest there under
  `level_boundaries` (on by default), restrict through either `:inject` or
  `:filter`, and run on stacked device tiles (commits `c2e7a39`, `56993d0`,
  `ec9361f`, `2306e87`).
- [x] **N10** — Composite conservation budgets were measured on long
  passive-species mixing and moving-refinement runs across rank counts,
  refinement depths and subcycling, and every layout is inside its
  declared budget, so no surface-flux correction is enabled (commit
  `74895a4`).
- [x] **N11** — The δ⁴ detector reads the ghost layers at patch and
  coarse-fine faces for every field recovered over the padded extent, and
  the sensors, the filter rows and the closure candidates were measured at
  imposed shells on shock crossings, with the step-on-a-plane failure and
  the three-level global-step undershoot recorded (commit `b4fe9d6`).
- [x] **N12** — Regrid checks refresh coefficients, opt-in refined-stage CFL
  violations use collective rollback, and three-/four-level rate and cost
  measurements retain Hermite output; the global-step undershoot depends on
  the filter application schedule (commit `e7bb381`).
- [x] **N12a** — Level-aware filter trials remove the fixed-layer
  undershoot within composite budgets; smooth-error and moving-refinement
  tradeoffs retain the production default and subcycling (commit `fbb3ad3`).
- [x] **N13** — Mass fractions are validated against a measured
  `StepControl.species_band` of 0.05, so every species case and both examples
  run under the retained `:strict` default; only cold-ambient converging shocks
  opt out, with bounded counts (commit `c9d41f4`).
- [x] **N14** — `interface_divergence` selects the flux divergence's closure rows at
  interface ends; the default stays, `:cascade4` is rejected and the Brady–Livescu
  rows stay an experimental Float64 option (commit `6928fe5`).
- [x] **N15** — `interface_flux = :ghost` differences the inviscid flux through
  interface ends from ghost fluxes; it leads at same-level faces, and at
  coarse–fine faces under `level_interpolation_order = 8`, and stays
  experimental (commit `b42e819`).
- [x] **N15a** — Under `interface_flux = :ghost` the molecular flux is
  differenced through interface ends from ghost fluxes, a same-level face's taken
  from the neighbour's flux records and a coarse-fine face's from the shell's
  gradient ring; the viscous interface rows reach the interior order, and
  promotion moves to N15b (commit `decf45a`).
- [x] **N15b** — `interface_flux = :ghost` is the default at patch and level
  interfaces and inert without one; `:closure` remains for shock-dominated and
  Float32 runs, and a configuration `:ghost` does not support raises an
  `ArgumentError` naming it (commit `e9a91ed`).
- [x] **N16** — `level_interpolation_order` (2, 4, 6 or 8) sets the live
  transfer order; 6 stays the default, which the default interface rows
  saturate, and 8 is the opt-in for viscous, filtered, multidimensional or
  `interface_divergence` runs (commit `c7d26e3`).
- [x] **N17** — `level_interpolation_order` defaults to the derivative
  operator's interior order, two more under `interface_flux = :ghost` up to 10;
  order 10 is exact to degree 9 behind the four-node buffer, and a C6 run under
  the closure rows is unchanged (commit `decf45a`).
- [x] **V5** — A tile takes only the rank counts its process grid admits,
  which is not monotone in the count, so the full MPI suite passes at eight
  ranks (commit `c4bbd8e`).

### API, state ownership, and reproducibility

- [x] **A1** — `precision` on `Numerics` and `Solver` converts every typed
  component, mixed types are rejected at setup, and `bench/audit.jl` checks
  Float32 point bodies for promotion (commit `5bd9b46`).
- [x] **A2** — Field ownership and freshness are documented in `patches.jl`,
  a scratch read names the patch that wrote it, and a `:repair` before the
  step renews the prepared state (commit `57b4ed2`).
- [x] **A3** — Construction lives in `construction.jl` and the `Solver` container
  in `solver.jl`, moved verbatim with byte-identical inference and allocation
  audits; `rhs.jl` holds only the right-hand side (commit `f123739`).
- [x] **A4** — Parameter ranges fail early with named errors, `run!`'s absolute
  `tfinal`/`nmax` are enforced, and a rollback restores switches, triggers and
  writer frames (commit `99be93c`).
- [x] **A5** — Checkpoints carry a versioned configuration record compared
  at load: thermodynamics and layout strictly, the other groups under `allow`
  (commit `77b3adb`).
- [x] **A6** — The supported-combinations page lists what setup accepts and
  the error of each rejection, held to the code by `test/capability_tests.jl`
  (commit `fb3b242`).
- [x] **A9** — `line_sample`, `field_slice` and the Makie recipes take the
  state vector of a refined run and sample root nodes from the finest level
  holding them (commit `e5d521d`).
- [x] **A11** — A tiled, regridded level may hold no tiles, so a run starts
  unrefined, refines when tags appear and empties when they vanish; the box
  keeps its region (commit `1a97ab4`).
- [x] **A12** — A tiled hierarchy regrids every level up to `max_levels`, each
  tagged on its parent; a three-level shock–contact run lands an order of
  magnitude closer to the uniform-fine reference than the root (commit
  `737550c`).
- [x] **A14** — `Hydrostatic` sets the pressure of an initial condition in
  discrete balance with the run's derivative operator, and an unfiltered
  two-fluid column stays at rest to round-off (commit `acfe5cb`).
- [x] **A15** — `CompositeBC` divides a face among member conditions by a
  coordinate mask; a walled-orifice jet and a two-slot stagnation plane run
  on it (commit `9574aea`).

### Scale, devices and I/O

- [x] **S5** — `Nasa9Mixture` runs on device storage through an isbits mirror
  carrying the flattened interval table (commit `b78acd4`).
- [x] **S6** — `test/hdf5_tests.jl` passes on a parallel libhdf5 at one, two
  and four ranks under both transfer modes, and the collective mode, measured
  faster on a local disk, is the default (commit `c8df974`).
- [x] **S14** — The decomposed `pyranda_filter` solve's departure from serial is
  the round-off floor of its ill-conditioned left-hand side, the same fraction
  of cond(A)·eps as every other operator's, and is recorded as such
  ([the measurement](CALIBRATION_APPENDIX.md#the-decomposed-line-solve-against-serial))
  (commit `1723ea3`).
- [x] **S16** — Mode truncation keeps one more mode in the radial and
  azimuthal momenta than in the scalars, with a floor of 2 and the θ cap at
  the momenta's mode, and is stable under the artificial properties (commit
  `f1171de`).

### HED physics
- [x] **H2** — `ImplicitConduction` integrates the molecular conduction in
  the implicit half of ARK4(3)6L[2]SA on one host-storage patch, decomposed
  or not, with collective rollback of a failed stage; a `cfl` chosen for the
  default integrator carries over (commits `9fd6365`, `9c2edd8`).
