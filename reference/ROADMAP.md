# CompactLES task roadmap

Prioritized open work for compressible, variable-density mixing and implosion.
The September 2026 source review adds runtime and API corrections to the existing
numerics, validation, AMR/GPU, and high-energy-density (HED) backlog.
The wall/interface follow-up adds R5, expands N6, and sequences N14–N17 and N15b.
Completed work is recorded by the commit that delivered it, and the measurements
behind it are in [CALIBRATION_APPENDIX.md](CALIBRATION_APPENDIX.md); method
details are in [DESIGN.md](DESIGN.md).

## How to use this plan

- Unchecked boxes are open deliverables. Stable IDs identify dependencies.
- Start with P0 correctness, then P1 numerical credibility and API contracts.
  P2 work can proceed independently when its stated prerequisites are met.
- A measurement task is complete when its result and the resulting decision,
  including a decision to retain the current method, are recorded: one line
  here naming the commit, its table in
  [CALIBRATION_APPENDIX.md](CALIBRATION_APPENDIX.md), and its default in
  [CALIBRATION.md](CALIBRATION.md) if a default moved.
- Close implementation tasks with the stated regression checks and the applicable
  repository validation gate. A numerical change requires an explained baseline
  update, not a relaxed test tolerance.
- Historical references to model debts 1/2/3 correspond to N1 / R3+N3 / V1 below;
  former Phase 2 items correspond to H1–H8.

## P0: runtime and boundary correctness

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

## P1: numerical credibility

### Filtering, regularization, and boundaries

- [x] **N1** — The compact filter default is `filter_cfl = 0.35` at α = 0.45,
  with the relaxed formulation measured invariant to the step (commit `dbe2899`).
- [x] **N2** — Volume-weighted filtering on non-uniform metrics was measured
  against the unweighted default and the current method retained (commit `811d382`).
- [x] **N3** — `run!` primes the artificial coefficients, which removed the wall
  and axis CFL ceilings; the spherical origin keeps its own (commit `c407e0b`).

- [ ] **N4 — Refit artificial shear viscosity after the filter policy is fixed.**
  Fit `C_mu` under the adopted smoother/detector on a 3-D case with an
  unresolved cascade, scored on the history misfit and not the peak, which is
  void as an estimator at 128³. One-dimensional shocks cannot determine the
  shear channel, and on Taylor–Green at 64³ the best-fitting `C_mu` is zero
  under both the production and the near-off filter
  ([Taylor-Green](CALIBRATION_APPENDIX.md#taylor-green)), so confirm that at
  128³ and choose the case accordingly; that confirmation is the one item N1
  left open, and it needs cluster time
  ([n1_recal128.sbatch](../bench/slurm/n1_recal128.sbatch) holds the leg).
  **Depends on:** N1 and the 3-D campaign. Retain `C_beta=1` unless new evidence
  overturns its completed refit; record error and dissipation attribution.

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
- [ ] **N6m — Admit higher-order wall closures behind an initial-data check.**
  A production closure must take a singular start on its own rows, so every
  derivative operator uses third-order rows at a wall and C8 and C10 gain
  nothing on a wall-bounded run. C6 `:brady_livescu` holds the wall wherever
  the cascade rows do and fails only when singular data sit on the closure
  rows themselves; C8 `:brady_livescu` needs the front tens of cells from the
  wall. Replace the singular-start requirement for opt-in higher-order rows
  with a setup-time check of the initial state near each closed edge that
  raises an error on data the rows cannot take, and qualify the rows on the
  starts the check admits
  ([wall closures in production](CALIBRATION_APPENDIX.md#wall-closures-in-production)).
  Known obstacles: the Brady–Livescu rows grow at some line lengths under the
  default filter, and no C8 or C10 set holds the warm starts
  ([the fifth-order closure search](CALIBRATION_APPENDIX.md#the-fifth-order-closure-search)).
  **Depends on:** N6b and N6j.

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
  restated as a continuous-model property; `:fickian` stays the default
  (commit `7a86a2c`).

### AMR numerics

Node-centered patch and level coupling is interpolation and injection with
compact interface closures, not a conservative flux reconciliation. The measured
interface orders are in the appendix's smooth-evolution
[accuracy matrix](CALIBRATION_APPENDIX.md#the-smooth-evolution-accuracy-matrix);
the designs and the fallback analysis are in [AMR_GPU.md](AMR_GPU.md).

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
  gradient ring; viscous interface rows read 6.1–6.8, and promotion moves to
  N15b (commit `decf45a`).

- [ ] **N15b — Promote `interface_flux = :ghost` to the default.** The ghost
  path leads the closure rows on every smooth row, inviscid and viscous, keeps
  the shock minima and conserves as they do (N15, N15a). Before it becomes the
  default: take the coarse-fine gradient ring on the device (device plans on the
  fine box and a device ring; a device patch now downloads the box at every
  imposition), give a custom EOS the temperature-gradient hook the coarse-fine
  molecular flux needs, re-record every guard and serial value a patched or
  refined default run moves (`test/convergence.jl` interface rows, the MPI
  suite's patch and level phases, the tutorials), and settle Float32, where the
  ghost path gains nothing. The artificial fluxes keep the one-sided rows.
  **Gate:** the full gate, the device suite and a GPU run of a refined viscous
  case; curvilinear metrics need ghost `area_d` and a matching GCL operator and
  are out of scope until a case needs them.

- [x] **N16** — `level_interpolation_order` (2, 4, 6 or 8) sets the live
  transfer order; 6 stays the default, which the default interface rows
  saturate, and 8 is the opt-in for viscous, filtered, multidimensional or
  `interface_divergence` runs (commit `c7d26e3`).

- [x] **N17** — `level_interpolation_order` defaults to the derivative
  operator's interior order, two more under `interface_flux = :ghost` up to 10;
  order 10 is exact to degree 9 behind the four-node buffer, and a C6 run under
  the closure rows is unchanged (commit `decf45a`).

Boundary/interface sequence: the N6 matrix (`bench/boundaryorder.jl`, gated in
`test/convergence.jl`), N6a's trial battery (`bench/wallfilter.jl`) and N6b's
qualification (`bench/wallclosure.jl`), with the `idiv`, `gflux` and `transfer`
studies and `bench/leveltransfer.jl`, are the instruments for N15b.
Qualify interface candidates under N10 and N11 before promotion. Extending N6l's
face-centred fold to patched and refined runs is open, as is switching the
tutorials whose walls are true symmetry planes; those runs keep the node-centred
`SlipWallBC` until then. Coordinate temporal certification with V3 and default
filter time-scaling with N1.

### Independent validation and regression coverage

- [ ] **V1 — Complete independent solver and experiment comparisons.**
  Run CompactLES versus Pyranda for Re=1600 TGV and one RM shock tube, comparing
  dissipation histories, spectra, and mix widths. Select one published RM experiment
  with documented initial/boundary conditions and obtain missing specifications.
  **Depends on:** R1–R4; coordinate compute with N1.
  **Deliver:** reproducible inputs, reference provenance, uncertainty/error measures,
  and a docs validation section. Keep analytic validation, external data, and
  self-generated regression profiles explicitly distinguished.

- [ ] **V2 — Add reproducible turbulent inflow generation.**
  Implement a digital-filter or synthetic-eddy utility producing a
  `(x, y, z, t) -> Prim` target for Dirichlet or NSCBC inflow.
  **Gate:** prescribed statistics/correlations, reproducible seeds across
  decomposition and restart, and a V1 experiment use case.

- [ ] **V3 — Close gaps in automated verification.**
  The temporal-order studies landed in commit `d8ea472`
  ([temporal order](CALIBRATION_APPENDIX.md#temporal-order)): the level
  interface is first order in `dt` from the once-per-step injection into the
  covered parent nodes, and injecting before every stage restores fourth order
  at one restriction per stage; deciding that is open. Turn R1–R4 probes into
  durable regressions. Add a scheduled full shock-validation battery and explicit Makie extension
  checks; retain HDF5 tests in the package test target and add parallel-HDF5
  execution where the required stack exists.
  Keep each instrument's current transcript under `bench/results/<script>.txt`
  so that an appendix table is a quoted output and never a retyped one; a
  transcript is the script's own output, so each lands with that script's
  next run rather than being assembled from the tables it would replace.
  **Gate:** CI distinguishes skipped/unavailable coverage from passing coverage.
  KA-on-CPU equality does not substitute for hardware-GPU tests under S1.

## P1: API, state ownership, and reproducibility

- [x] **A1** — `precision` on `Numerics` and `Solver` converts every typed
  component, mixed types are rejected at setup, and `bench/audit.jl` checks
  Float32 point bodies for promotion (commit `5bd9b46`).

- [ ] **A2 — Make cache ownership and freshness explicit.**
  Inventory persistent integrator state, per-patch primitives/geometry, shared RHS
  scratch, and diagnostic storage. Define invalidation after initialization,
  stepping, filtering, regridding, and restart; audit property forwarding and
  `prepared`/`primitives_current` assumptions. Include optional material caches,
  nonlinear trial states, and rollback invalidation under the
  [material interface design](DESIGN.md#material-and-physics-interfaces).
  **Gate:** multi-patch consumers cannot silently read another patch's scratch;
  R1 remains fixed without duplicating every patch workspace.

- [ ] **A3 — Separate solver construction from RHS execution.**
  Move configuration validation, plan/hierarchy construction, and device setup out
  of `rhs.jl` behind explicit construction boundaries. Preserve specialized
  runtime types and per-patch function barriers; setup-only abstract fields do not
  by themselves justify a type-system rewrite.
  **Gate:** unchanged numerical results plus inference/allocation comparisons for
  the existing hot paths; preserve compatibility of supported accessors.

- [x] **A4** — Parameter ranges fail early with named errors, `run!`'s absolute
  `tfinal`/`nmax` are enforced, and a rollback restores switches, triggers and
  writer frames (commit `99be93c`).

- [x] **A5** — Checkpoints carry a versioned configuration record compared
  at load: thermodynamics and layout strictly, the other groups under `allow`
  (commit `77b3adb`). Under A8, move `thermodynamic_model` and the
  `_record_fields` overrides with the types.

- [ ] **A6 — Publish a capability and extension matrix tied to setup checks.**
  Cover EOS × backend × precision × metric × patch/refinement mode, including
  static nested levels versus two-level regridding and checkpoint restrictions.
  Correct stale README/reference claims against current code and tested hardware.
  Clarify that `EquationSet` currently owns layout/parity while flux assembly
  remains Navier–Stokes-specific; define the additional hooks needed by H3/H6/H8.
  **Gate:** accepted/rejected combinations and extension examples are tested;
  avoid promising unsupported feature combinations.

- [ ] **A7 — Implement material interfaces with an explicit analytic fast path.**
  Follow [the interface design](DESIGN.md#material-and-physics-interfaces):
  separate local material queries, solver field adapters, and flux closures;
  define composition/energy metadata, typed results/status, and setup checks.
  Preserve the existing ideal/stiffened analytic recovery and scalar transport
  methods, public constructors, and readable arithmetic. Introduce an internal
  `MaterialModels` boundary for standalone evaluators/readers, removing EOS field
  assumptions from adapters. No package extraction or HED state allocation is
  required for this step; numerical formulas and default policies stay unchanged.
  **Sequence:** inventory current hooks and exercise independent queries with
  existing ideal, NASA-9 and transport models; coordinate cache/setup ownership
  with A2/A3 and precision with A1; qualify a tabulated model with H5 before
  stabilizing the rich interface. H1 and offline material/transport work may
  proceed independently; H3 uses the relevant A7 contracts as they mature.
  **Gate:** existing input decks and numerical baselines, independent query and
  adapter agreement, inference/allocations, and unchanged ideal workspace,
  gradient, launch and collective counts. Compare before/after full-step costs
  on single/multispecies ideal cases, including constant and analytic transport,
  plus NASA-9 and stiffened-gas controls; run `bench/jetcheck.jl` and
  `bench/audit.jl` and relevant CPU/MPI/device gates. Use paired measurements and
  repeated processes on the same hardware; no reproducible ideal-fluid slowdown
  or added per-point allocation is accepted to accommodate the rich path.
  Record measurements in the appendix and unavailable hardware explicitly.

- [ ] **A8 — Qualify optional material-package extraction after interface use.**
  Decide whether to retain the internal module or extract it after A7 and H5
  exercise the boundary. Require a concrete reuse, dependency/data, or release
  benefit; independent packaging is not a prerequisite for HED development.
  If extracting, keep material evaluation independent of CompactLES/MPI, supply
  CompactLES adapters through an optional extension, preserve supported imports,
  and coordinate restart identity with A5. Avoid a separate interfaces package
  until multiple consumers need it.
  **Gate:** record the extraction/retention decision; for extraction, independent
  package tests, extension loading/compatibility and coupled numerical tests,
  A7's ideal-performance gate, and measured load/precompile cost. Package/module
  boundaries must still allow solver-owned fusion and optional bulk evaluation.

- [x] **A9** — `line_sample`, `field_slice` and the Makie recipes take the
  state vector of a refined run and sample root nodes from the finest level
  holding them (commit `e5d521d`).

- [ ] **A10 — Let a refined level reach and cross the domain boundary.**
  Every level now stays `max(n_halo, 4)` parent nodes inside its parent, so a
  feature at a wall, an inflow face, or across a periodic seam stays coarse
  (a warning now says so for a box and for shapes; the tiled path drops the
  margin band silently). AMReX-style grids touch physical faces and wrap
  periodic ones. Needs boundary conditions evaluated on fine patches, a
  periodic image of the parent box gather, and wrapped tile regions.
  The target case is the 3-D vortex ring fired from a tube's top face into an
  air/SF6 interface and then shocked through it (the axisymmetric form is
  `examples/vortex_ring_shock.jl`): the ring forms at the injector face, so
  without this item it can only be formed inside the domain by a body force.
  In that workaround one refined box follows the ring well; lattice tiles of
  small edge cost many times more per step than the box for a compact feature.
  **Gate:** a shock and an interface followed into a wall and through a
  periodic seam, with the conservation and interface-reflection tests of a
  uniformly fine run; the vortex-ring case with the ring formed at the face.

- [x] **A11** — A tiled, regridded level may hold no tiles, so a run starts
  unrefined, refines when tags appear and empties when they vanish; the box
  keeps its region (commit `1a97ab4`).

- [ ] **A12 — Regrid more than one level, with `max_levels`.**
  Regridding moves one refined level; nested levels are static. Sensor-driven
  nesting to a requested depth is the common AMR interface (AMReX `max_level`,
  Trixi's `AMRController`, Basilisk `adapt_wavelet(..., maxlevel)`).
  **Depends on:** the level-ℓ tag sweep over tiled parents.
  **Gate:** a three-level shock–interface run whose finest level follows the
  feature, against a uniformly fine reference.

- [ ] **A13 — Refinement scope beyond uniform Cartesian grids.**
  Refinement rejects cylindrical and spherical metrics, stretched grids,
  symmetry planes, pentadiagonal filters and the `:d8` detector, which rules out
  the axisymmetric shock-tube and vortex-ring configurations. Each needs its
  transfer closures and fold-aware gathers; order them by use.
  **Gate:** per extension, the refined-versus-uniform convergence rows of
  `test/convergence.jl`.

- [x] **A14** — `Hydrostatic` sets the pressure of an initial condition in
  discrete balance with the run's derivative operator, and an unfiltered
  two-fluid column stays at rest to round-off (commit `acfe5cb`).

- [x] **A15** — `CompositeBC` divides a face among member conditions by a
  coordinate mask; a walled-orifice jet and a two-slot stagnation plane run
  on it (commit `9574aea`).

## P2: scale, devices, I/O, and geometry

Detailed mechanisms and measurements remain in [AMR_GPU.md](AMR_GPU.md) and
[CLUSTER.md](CLUSTER.md). Patch AMR, device execution, stacked tile launches, and
opt-in Float32 already exist; the tasks below extend or validate them.

- [ ] **S1 — Complete target-machine GPU measurements and resolve the wait stall.**
  Continue the rzadams/MI300A campaign using the existing measurements as a baseline,
  not as an unmeasured port. Characterize/resolve the intermittent ROCm wait stall
  in [rocm_wait_stall_report.md](bugreports/rocm_wait_stall_report.md) before interpreting
  performance changes. Record hardware, MPI stack, precision, synchronization
  policy, repeat variability, and correctness with every result.
  **Gate:** full single/refined/tiled runs on target hardware, including real device
  communication paths; report measured reproducibility rather than universal
  bitwise claims. Validate other advertised backends on their own hardware.

- [ ] **S2 — Measure and address compact-solve and transfer scaling limits.**
  Profile replicated dense interface solves, host/device fences and transfers,
  parent-box gathers, and replicated AMR memory as line-rank count and region size
  grow. Compare gather-solve-scatter, on-device reduced solves, GPU-aware MPI, and
  partitioned transfers only where profiles justify them.
  Measure stacked `max_rate` reductions before adding another launch optimization;
  revisit extra streams only if batching still leaves useful concurrency.
  The replicated interface solve has a cost model and a planned fix in S12.
  **Depends on:** reliable S1 timing. **Gate:** crossover data and preserved
  distributed accuracy, not a workstation-only speedup claim.

- [ ] **S3 — Complete production tile/ownership cost studies.**
  Build the 3-D implosion-shell benchmark at realistic tile sizes; measure
  per-imposition latency and repeat the previously cold tiled-overhead case warm.
  Quantify rebalance and migration benefits, including root-work bias in measured
  tile weights; subtract that baseline only when its impact is established.
  **Gate:** repeated-process timings, memory, imbalance, migration cost, and accuracy
  against coarse and uniform-fine references on the target clusters.

- [ ] **S4 — Choose mixed-precision policy (former G4b).**
  Separate state, geometry, clock, solve, and accumulation precision explicitly.
  Use existing Float32 CPU/device histories and S1/S2 profiles to compare policies.
  **Gate:** conservation drift, thermodynamic recovery, closure conditioning,
  timestep progress, memory, and throughput; no CPU-default change from speed alone.

- [ ] **S5 — Add the NASA-9 device coefficient mirror.**
  Flatten interval tables into a fixed-width adapted representation and port
  inversion/recovery with R4's status and domain policy.
  **Gate:** CPU/device recovery and full-run comparisons over interval joins and
  difficult states in both precisions, followed by actual-device profiling.

- [ ] **S6 — Verify collective HDF5 writes on a parallel libhdf5.**
  `FieldWriter(format = :hdf5)` and its XDMF temporal collection landed in
  commit `c04abf1`, and every block write issues an empty-selection H5Dwrite
  on a rank without a block, exercised only under the serialized backend.
  Run `test/hdf5_tests.jl` against a parallel libhdf5 built for the run's MPI,
  including a slice that leaves ranks with no selection, and measure whether
  the collective transfer mode on the block datasets pays there.
  **Depends on:** a cluster with parallel HDF5. A refined solver's HDF5 dump
  (a spatial collection per patch with blanking) waits for a case.

- [ ] **S7 — Validate thread pinning and cluster placement.**
  Wire pinning only after controlled target-cluster trials: fixed-rank comparisons
  on identical masks, repeated-process medians, and one rank per NUMA domain with
  pinned threads on the suitable Julia runtime.
  Preserve the SMT-sibling collapse reproducer and prepare evidence for site
  support. **Gate:** measured benefit over scheduler binding and repeatable launch
  guidance in [CLUSTER.md](CLUSTER.md).

- [ ] **S8 — Extend refinement and multiblock geometry when a target needs them.**
  For a nested implosion, implement regridding below level 1 with descendant
  re-nesting, ownership, transfer, and restart updates. For geometric multiblock
  use, extend beyond the current slab layout with explicit adjacency and compatible
  geometry; add same-level patch checkpoint support.
  **Depends on:** a concrete case, N10–N12, and relevant A5 contracts.
  **Gate:** interface accuracy, conservation budgets, restart, and distributed
  consistency. Fold-adjacent refinement stays forbidden pending a separate design.

- [ ] **S9 — Implement azimuthal mode truncation in staged form.**
  Follow [MODE_TRUNCATION.md](MODE_TRUNCATION.md): serial cylindrical,
  decomposed angle, calibration/defaults, then spherical azimuth.
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

- [ ] **S11 — Fuse each compact line solve into one threaded region.** Low
  priority: on the 2-D case that motivated it, `mpiexec -n 8 -t 1` at 12 ms/step
  already beats the fused-threads target of about 15, so this waits for a
  setting in which threads are the only parallelism, such as the host-side
  interface stage of a device plan.
  A line solve is three regions today (fill, solve, scatter) and each carries
  a spawn/join floor and a barrier, with the solved block crossing cores
  between them. On a 37k-point planar case at eight threads that granularity,
  not bandwidth, is the loss against ranks: 195 regions per RHS at about 54 µs
  serial work each, ranks 2x faster per step than pinned threads
  ([CLUSTER.md](CLUSTER.md), hybrid-desktop paragraph). A prototype in which
  each thread fills, solves and scatters its own chunk of lines in one region
  measured 1.86x on the x-direction filter solve at `-t 8`, parity serially
  and bitwise identical, provided the 16-wide column blocking of `solve_cols!`
  is kept inside each chunk. The transposed y/z path fuses the same way over
  chunks of x-columns, since a chunk's fill, row-sweep solve and scatter are
  all contiguous in it. Where the dimension is decomposed the reduced
  interface stage is a collective between the local solve and the spike
  correction, so the fusion there is two regions, not one, and the collective
  stays outside both. The `@threaded` docstring names this reorganization as
  the condition for re-measuring the threading backend; do that after, not
  before. **Gate:** the core gate, `bench/jetcheck.jl` and `bench/audit.jl`
  before and after, bitwise agreement of the serial suite, the MPI suite at
  2, 4 and 8 ranks, and the 768×48 thread table re-measured pinned.

- [ ] **S12 — Remove the O(P²) replicated interface solve for many-rank scaling.**
  The reduced interface matrix is dense of order 2qP and every rank applies
  its LU to its own lines, about 8q²P² operations per line against 9qn for
  the local sweep. Per rank on a uniform 3-D grid that is a fixed 8q²N² per
  solve while the local work shrinks as N³/P³: an Amdahl term whose
  operation-count crossover is P ≈ 6.6 (q = 1) or 5.2 (q = 2) per direction
  at N = 256 and 10.4 / 8.3 at N = 1024, a few hundred to a thousand ranks.
  The 224-rank rzhound run with dims (8,7,4) was at that crossover along x.
  Per-step collective latency is not the limit: Allgathers span one direction's
  P ranks and the measured node scaling was 93% per doubling at four nodes.
  Stages, in order:
  1. Instrument first: a probe that times the local sweep, the Allgather and
     `_reduced_solve!` separately at the target P and N, for both `LineSolver`
     and `BandLineSolver`, so the table becomes a measurement and the
     wall-clock crossover (later than the count, since the dense solve runs
     cache-resident while the sweep streams memory) is known.
  2. Banded factorization of the reduced matrix, which is block-tridiagonal in
     P because a rank's interface unknowns couple only to its neighbors:
     per-line cost O(q²P), per-rank cost back to N²/P. Covers the
     tridiagonal and pentadiagonal paths alike; the Allgather is unchanged.
     Round-off, not bitwise, agreement with the dense solve; the MPI suite's
     3e-15 expectation for the decomposed solve is the bar.
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

- [ ] **S13 — Cut the remaining cost of the NASA-9 temperature inversion.**
  `recover_primitives!` under `Nasa9Mixture` runs a safeguarded Newton solve
  per point and per Runge–Kutta stage, and that solve is the ~6× step cost of
  the NASA-9 model over `IdealMixture`. The interval table is isbits and the
  inversion and every per-point species loop share the powers of T (commits
  `9e0f126`, `405037f`, measured with `bench/nasa9_inversion.jl`). What
  remains, in order of payoff per risk:
  1. Relax the convergence criterion from 32 eps toward 1e-10 relative,
     which saves about one of the four or five iterations. Newton's last
     iteration exists to certify the previous one. A numerics decision: it
     moves recovered temperatures at the 1e-10 level and so the serial and
     MPI baselines.
  2. Warm-start from the stored `T_ion` field, one or two iterations instead
     of four or five. It trades away the state-only seed that
     `mixture_temperature_status` documents for bit-for-bit agreement between
     serial and decomposed runs and for restart independence; only worth it
     if 1 leaves the model still far from the ideal-gas step cost.
  Keep the polynomial powers literal (`T^4`): a repeated product is not
  bit-identical to the library power and moves every baseline for nothing.
  **Depends on:** a baseline decision.
  **Gate:** the core gate with explained baseline updates; time the inversion
  before and after at four species in the same session.

## P2/P3: high-energy-density physics

H1 is the principal infrastructure dependency for stiff diffusion. These are
separate capabilities with independent verification gates, not a claim that new
equation layouts alone make the current RHS a general multiphysics solver.
Ownership and coupling follow [the material interface design](DESIGN.md#material-and-physics-interfaces);
DT is the first cold-material path, with C/CH/CD coverage tracked separately in
H5b. Every integration preserves A7's ideal analytic execution contract.

- [ ] **H1 — Build implicit diffusion infrastructure.**
  Reuse the distributed banded kernels for ADI or line-relaxation smoothing;
  compare geometric multigrid and Krylov outer solves. Keep operators and
  communication in core numerics with optional workspace allocated only when used.
  **Gate:** manufactured constant/variable-coefficient heat conduction in every
  supported metric, distributed residual/convergence studies, and freestream
  preservation. Variable coefficients require a factorization-update policy.

- [ ] **H2 — Add compatible IMEX time integration.**
  Evaluate established IMEX-ARK tableaus before implementing new ones; define a
  compatible explicit/implicit pair and workspace contract. Accept component
  contributions to a joint residual and consistent linearization/Jacobian action;
  define coefficient refresh, nonlinear trial invalidation and collective retry.
  Independent physics components must not force sequential split updates.
  **Depends on:** H1. **Gate:** temporal order, stiff stability, source/diffusion
  splitting error, and recovery from a failed implicit solve.

- [ ] **H3 — Add separate ion/electron/radiation energy evolution.**
  Extend the equation/flux/recovery interfaces for `T_ion`, `T_ele`, and `T_rad`,
  electron pressure, and electron–ion equilibration. Specify independent energy
  variables, binding-energy references, pressure work and exchange mappings;
  prevent double counting in total-energy and subsystem equations. Declare
  gradient, boundary, wave-speed and regularization requirements at setup.
  **Depends on:** A6/A7 contracts, R3/R4, and H2 for stiff coupling.
  **Gate:** total-energy conservation, equilibrium limits, and independent
  relaxation problems before coupled implosion runs.

- [ ] **H4 — Add electron thermal conduction.**
  Implement flux-limited Spitzer–Härm transport through the implicit solver.
  Treat this as the hot-limit closure; wider material conduction consumes H5a/H5b
  state and validated coefficients without imposing their cost on scalar transport.
  **Depends on:** H1–H3 and the dispatchable transport interface.
  **Gate:** analytic/manufactured transport limits, limiter behavior, and coupled
  energy budgets.

- [ ] **H4a — Add isotope-resolved H/D/T ion transport.**
  Begin with independently verified unmagnetized, fully ionized coefficients;
  add concentration, pressure, ion/electron temperature, and electric-field
  driving terms with a defined ambipolar closure. Validate dense-plasma models
  separately from the weak-coupling limit; see [the transport plan](TRANSPORT.md).
  **Depends on:** A6/A7/H3 for coupled flux and energy evolution, H1/H2 when stiff,
  and H5 or an explicit ionization closure outside the fully ionized regime.
  Coefficient and flux-reference work can proceed before these dependencies.
  **Gate:** independent H–D/H–T/D–T references, isotope permutation and trace
  limits, zero-net-mass flux, charge/current closure, driven isotope separation,
  total-energy budgets, and declared coupling/degeneracy validity ranges.

- [ ] **H5 — Add tabulated EOS support.**
  Implement IONMIX reading and thermodynamically consistent interpolation/inversion;
  consider SESAME later subject to data access/licensing. Exercise A7's rich
  queries and shared interpolation/derivative results with a bounded table model;
  qualify forward/recovery adapters before stabilizing that interface. Declare
  frozen/equilibrium derivatives and supported temperature partitions; a 1T table
  alone does not supply a 2T EOS or the populations required by transport.
  **Depends on:** R3/R4, A5 and the relevant A7 contracts; reader/data work may
  precede their completion.
  **Gate:** table-node/interpolation checks, inverse consistency, derivatives,
  phase/domain handling, and an EOS-specific artificial-conductivity scale.

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

- [ ] **H6 — Add flux-limited radiation diffusion, gray before multigroup.**
  Define opacity and group interfaces plus radiation-energy components. Opacity
  consumes the same material/population state as transport; couple exchange
  through H2/H3's residual and energy contract, with explicit group conventions.
  **Depends on:** H1–H3 and suitable EOS/opacity data.
  **Gate:** independent diffusion/equilibration references, group convergence,
  positivity policy, and matter–radiation energy conservation.

- [ ] **H7 — Add laser deposition for a specified experiment.**
  Implement geometric-optics rays, inverse-bremsstrahlung absorption, rank-to-rank
  ray transfer, and deposition through the source interface.
  **Gate:** ray trajectories and absorbed/deposited energy budgets.
  This can precede the full HED stack when suitable material data are available.

- [ ] **H8 — Add MHD only for a magnetized target.**
  Extend the equation and flux interfaces and evaluate GLM divergence cleaning
  for the finite-difference scheme.
  **Gate:** standard wave/shock problems, divergence-error control, and energy
  budgets. Plan magnetized transport separately if the target requires it.

## Deferred scope and non-goals

- **Wall-resolved/wall-modeled LES:** deferred until a target requires it;
  current no-slip support serves verification and does not establish wall-model fidelity.
- **Symbolic frontend:** optional only after hand-written equation sets define a
  usable interface; a macro may emit typed code, but runtime PDE-string evaluation
  remains excluded.
- **Excluded:** cell-by-cell/oct-tree AMR, unstructured meshes, a parallel Godunov
  solver path based on Riemann solvers/flux limiters, and general-purpose CFD scope.
  Instrumented state repair remains allowed and must retain intervention budgets.
- **Design commitments:** compact structured operators, patch-based refinement,
  explicit geometry/collective contracts, and measurable accuracy for mixing and
  implosion. Revisit a restriction through a concrete case and a documented design.
