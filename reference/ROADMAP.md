# CompactLES task roadmap

Prioritized open work for compressible, variable-density mixing and implosion.
The September 2026 source review adds runtime and API corrections to the existing
numerics, validation, AMR/GPU, and high-energy-density (HED) backlog.
The wall/interface follow-up adds R5, expands N6, and sequences N14–N16 from
the experiments in [BOUNDARY_ACCURACY.md](BOUNDARY_ACCURACY.md).
Completed implementation and benchmark history belong in [HISTORY.md](HISTORY.md);
method details belong in [DESIGN.md](DESIGN.md).

## How to use this plan

- Unchecked boxes are open deliverables. Stable IDs identify dependencies.
- Start with P0 correctness, then P1 numerical credibility and API contracts.
  P2 work can proceed independently when its stated prerequisites are met.
- A measurement task is complete when its result and resulting decision are
  recorded, including a decision to retain the current method.
- Close implementation tasks with the stated regression checks and the applicable
  repository validation gate. Numerical changes require explained baseline updates
  in [CALIBRATION.md](CALIBRATION.md), not merely relaxed test tolerances.
- Historical references to model debts 1/2/3 correspond to N1 / R3+N3 / V1 below;
  former Phase 2 items correspond to H1–H8.

The review baseline is Julia 1.11.4 on Windows: 2,025 serial assertions and the
spatial convergence suite passed. HDF5 and Makie tests were skipped locally;
multi-rank and hardware-GPU tests were not run in that review. The R1–R3 probes
below exposed behavior outside those passing checks.

## P0: runtime and boundary correctness

- [x] **R1 — Make diagnostics independent of integrator state.**
  In a 64-point Sod case at CFL 0.15, after one step, requesting
  `field_array(..., :beta_art)` changed the next dt from 0.000587255 to
  0.000961137 without changing Q. Artificial-field output and `dissipation_rate`
  also overwrite coefficients consumed by the next CFL check.
  Separate diagnostic storage or preserve the integrator coefficients; audit
  primitive refreshes used by sensor-based regridding as well.
  **Gate:** identical dt histories, states, and regrid decisions with and without
  observational calls at the same step boundaries, on single and multiple patches,
  CPU/device paths, and MPI. Test output scheduling separately.
  **Code:** [viz.jl](../src/viz.jl), [io.jl](../src/io.jl),
  [diagnostics.jl](../src/diagnostics.jl), [timestep.jl](../src/timestep.jl).

- [x] **R2 — Guarantee clock progress and reachable endpoints.**
  A Float32 run with Float64 `tfinal=0.7` stalls at 0.699999988079071,
  repeatedly taking a 1.1920929e-8 remainder that cannot advance its clock.
  Define endpoint conversion/tolerance semantics and check progress after all
  clipping, including callback landing. Return a diagnosed failure for an
  unrepresentable advancing step instead of running to `nmax`.
  **Gate:** Float32/Float64 endpoints on both sides of representable values,
  large restart times, scheduled callbacks, and subcycling all terminate with
  documented time accuracy. Avoid an endless test by bounding steps.
  **Code:** [timestep.jl](../src/timestep.jl), [callbacks.jl](../src/callbacks.jl).

- [x] **R3 — Validate accepted and returned states under an explicit policy.**
  The current guard checks mixture density and dt before stepping. Probes completed
  with ideal-gas specific internal energy -1 or species densities (-0.1, 1.1).
  A uniform density sink took rho from 1 to -1 at `tfinal=0.02` and returned
  normally because no next iteration checked the result.
  Add EOS-aware checks of finite conserved values, partial densities, and
  thermodynamic admissibility at initialization and before accepting/returning a
  step, including `nmax` and callback exits. Distinguish strict rejection,
  explicitly permissive research runs, and optional repair; report substitutions.
  Do not impose e > 0 universally: formation-energy gauges and EOS domains differ.
  **Gate:** invalid final states trigger the selected policy, failures remain
  collective and retryable, and permissive/repair modes report their interventions.
  Preserve and quantify the known Noh repair tradeoff under N3.
  **Code:** [stepcontrol.jl](../src/stepcontrol.jl),
  [physics.jl](../src/physics.jl), [problem.jl](../src/problem.jl).

- [x] **R4 — Make NASA-9 recovery failure observable.**
  `mixture_temperature` currently returns its estimate after 30 iterations or
  nonpositive mixture cv without a final residual/status check. Outside-range
  polynomial evaluation also proceeds silently.
  Add residual-based success criteria, diagnosed failure, and an explicit
  extrapolation policy; use a safeguarded inversion where the EOS admits a bracket.
  **Gate:** interval joins, temperature extremes, invalid compositions, and
  nonconvergence behave consistently in both precisions; connect failure to R3.
  **Code:** [physics.jl](../src/physics.jl).

- [x] **R5 — Enforce the adiabatic, impermeable no-slip wall flux contract.**
  `NoSlipWallBC()` zeros velocity but leaves the normal conductive energy flux
  unconstrained. With rho=1, u=0, p=1+0.1x and mu0=0.01, the audit obtains
  energy flux -0.005 at both endpoints after enforcement and RHS evaluation.
  This incompatible-state probe demonstrates missing flux imposition; it is not
  a convergence study. Define the noncatalytic species and thermal wall contract,
  including molecular/artificial transport and the `:bulk` species channel.
  Implement it in the flux/derivative boundary treatment before divergence, or
  with an equivalent correction to every affected RHS row; changing only the
  endpoint energy update does not correct the compact divergence nearby.
  The first implementation candidate is a wall-flux hook after complete flux
  assembly (including `:bulk`) and before flux exchange/divergence, so corrected
  boundary fluxes enter every affected compact row.
  Retain prescribed temperature and account for heat exchange at isothermal walls.
  **Gate:** zero normal energy/species leakage at an adiabatic impermeable wall;
  compatible manufactured insulated conduction and species-diffusion evolution;
  isothermal heat-flux/energy balance; both faces in x/y/z, corners, supported
  metrics/EOS, Float32/Float64, and ranks that do not own a wall. Exercise filter
  and artificial-transport paths separately and preserve collective ordering.
  Track whole-domain budget defects separately from pointwise wall enforcement.
  **Code:** [boundary.jl](../src/boundary.jl), [rhs.jl](../src/rhs.jl),
  [runtests.jl](../test/runtests.jl),
  [boundaryorder.jl](../bench/boundaryorder.jl).
  **Delivered:** `correct_flux!` with serial/MPI and hardware regressions;
  [measured errors and budgets](CALIBRATION.md#no-slip-wall-flux-contract-r5-september-2026),
  [completion record](HISTORY.md#no-slip-wall-flux-contract-september-2026).

## P1: numerical credibility

### Filtering, regularization, and boundaries

- [ ] **N1 — Calibrate filtering and settle its time-scaling policy.**
  The rate-scaled `filter_cfl` mechanism is delivered but opt-in; the default still
  dissipates per application. Fit alpha, cadence, and reference CFL jointly
  against 128³ TGV histories and spectra. Measure shock-battery sensitivity and
  smooth-turbulence budgets under subcycling, variable dt, retries, and shortened
  output steps.
  **Depends on:** R1–R3; cluster time.
  **Deliver:** a reproducible fit and explicit default decision in
  [CALIBRATION.md](CALIBRATION.md). Do not fit under one formulation and then
  silently switch to the other.
  **Delivered so far:** the reference history, vendored as
  [`data/spectral_Re1600_512.gdiag`](../data/README.md#taylor-green-reference-solution),
  which supersedes the planned figure digitization; the fit instrument in
  [tgv_energy.jl](../bench/tgv_energy.jl), which crosses `alphaf`, `filter_cfl`
  and `cfl` against `configs` and scores each point by the relative-L2 misfit of
  its kinetic-energy and −dKE/dt histories, validated at 32³ against the recorded
  archive ([the fit instrument](CALIBRATION.md#the-fit-instrument)); and the
  spectra, as `snapshots=` HDF5 dumps postprocessed offline by
  [tgv_spectrum.jl](../bench/tgv_spectrum.jl), which at 32³ separate two α values
  by a factor of ten in the grid-scale band where the histories separate them by
  13% ([spectra](CALIBRATION.md#spectra)). No distributed FFT exists or is
  needed.
  **Consequence already recorded:** the rounded `1.2e-2 at t = 9` used throughout
  the earlier work is 6.7% below the tabulated peak, which withdraws the
  `C_mu ≈ 0.004` candidate and voids the peak as an estimator at 128³
  ([Taylor–Green](CALIBRATION.md#taylorgreen)). N4 inherits the history misfit in
  its place.
  **Measured at 128³:** the α leg, three values under the Gaussian smoother at
  `cfl = 0.6` and `filter_interval = 1`. Both history misfits are monotone in α
  and fall 46% and 72% from α = 0.40 to 0.49, against 19% and 8% at 32³, so the
  signal grew under refinement and the history fit is usable at this resolution.
  Monotone is a boundary fit: the histories rank the weakest filter first and
  cannot select α on their own. The peak crosses the reference between 0.45 and
  0.49, which brackets the total-dissipation match at α ≈ 0.486 but does not
  restore the peak as a criterion
  A fourth point at α = 0.499 then turns: the dissipation misfit is 15% worse
  there than at 0.49 and the kinetic-energy misfit has flattened onto a floor,
  so **α = 0.49 is a minimum and the first interior optimum in this
  calibration**. The peak crossing puts the total-dissipation match at α ≈ 0.486,
  so the two estimators agree on a band of roughly 0.485 to 0.49
  ([the α sweep](CALIBRATION.md#the-alpha-sweep-at-128)). The minimum is joint
  with `C_mu`, whose channel grows from 1.7% of the sink at α = 0.40 to 8.4% at
  0.499 as the filter gives way. The energy at t = 9 is closest to the reference at
  α = 0.49 as well, a third estimator on the same point. The spectra bound α from
  below only: the settings differ just above k ≈ 18, in a band carrying under
  1.5% of the energy, and every tail steepens smoothly through Nyquist with no
  grid-scale pile-up, including at α = 0.499, where the history has already
  turned ([the spectra](CALIBRATION.md#the-spectra-at-128)). The high-wavenumber
  share ranks filter strength without locating the optimum at this resolution,
  which reverses its role at 32³.
  **Remaining:** whether the band transfers to 256³, where `k_max η` reaches 1.5
  and the dissipation range is resolved; the cadence and relaxation legs; and
  the shock battery, which has to confirm that a weaker filter still stabilizes
  the shocked cases before any default moves.

- [ ] **N2 — Measure and implement conservative filtering on nonuniform metrics.**
  Compare current unweighted component filtering with the reference's
  volume-weighted field divided by a volume passed through the same filter.
  Establish the actual discrete conservation property rather than assuming that
  constant preservation proves it.
  **Gate:** constants, volume-integrated mass/momentum/energy defects, folds,
  stretching, and converging-shock behavior; unchanged uniform Cartesian results.

- [ ] **N3 — Resolve symmetry-cell startup robustness and the cold-state limit.**
  Instrument planar, cylindrical, and spherical Noh with the existing floor tally;
  compare permissive, representable-repair, and internal-energy-repair trajectories.
  Measure the pressureless wall layer and repair budgets as well as plateau and
  shock position. Test density-proportional beta feedback and filter-rate dependence.
  Separately evaluate a nonsingular gas-model artificial-conductivity scale near
  cold ambient states, retaining the EOS dispatch hook.
  **Gate:** configuration-specific CFL envelopes, conservation/error budgets, and
  a justified treatment of the spherical singular start and initial smoothing.
  Consult [CALIBRATION.md](CALIBRATION.md) before reopening rejected predictor,
  sensor-reach, or fold-order explanations; the old universal CFL 0.15 description
  is obsolete.

- [ ] **N4 — Refit artificial shear viscosity after the filter policy is fixed.**
  Fit `C_mu` under the adopted smoother/detector on 3-D TGV and mixing cases.
  One-dimensional shocks cannot determine the shear channel.
  **Depends on:** N1 and the 3-D campaign. Retain `C_beta=1` unless new evidence
  overturns its completed refit; record error and dissipation attribution.

- [ ] **N5 — Establish an anisotropic case before directional bulk viscosity.**
  Add a strongly stretched or anisotropic validation case, then compare scalar and
  directional beta with the matching directional diffusive timestep constraint.
  **Gate:** a measurable accuracy/stability benefit on that case; a null result on
  isotropic cases is not justification for implementation.

- [ ] **N6 — Establish spatial boundary/interface accuracy acceptance studies.**
  Promote the audit's polynomial and phase-varied evolution probes into durable
  studies of operator truncation, one filter pass, instantaneous RHS error, and
  final-time solution error. The default wall numbers are derivative order 3.17
  and one-pass filter order 1.88; neither is a measured evolution order.
  C6/C8 Brady–Livescu rows have pointwise orders 5/7, despite the existing
  field-specific 5.88/7.91 fits. Use actual h, fixed physical refinement endpoints,
  several fields/phases, at least three resolutions above roundoff, separate fold,
  outer-wall, interface and interior norms, and composite volume-weighted norms
  excluding covered parents. Retain fold studies as controls for wall pollution.
  **Deliver:** a reproducible accuracy matrix and regression gates for smooth
  inviscid and viscous walls, same-level interfaces, and two-/three-level AMR;
  compare unfiltered and filtered evolution. Sweep dt until temporal differences
  are below 10% of the spatial error used for a slope; V3 owns pure temporal-order
  certification. Report repeated-filter accumulation and cadence explicitly.
  **Depends on:** R5 for claims about adiabatic viscous walls; inviscid/interface
  studies can proceed immediately. Preserve or explain changes to historical
  regression guards in [CALIBRATION.md](CALIBRATION.md).
  **Code:** [convergence.jl](../test/convergence.jl),
  [patch_tests.jl](../test/patch_tests.jl), [level_tests.jl](../test/level_tests.jl),
  [boundaryorder.jl](../bench/boundaryorder.jl).

- [ ] **N6a — Qualify the one-sided wall filter and decide its default.**
  Compare `compact_filter(closures=:onesided)` against `:cascade`, initially with
  C6 `:cascade3` derivatives. The one-pass wall slope rises from 1.88 to 8.07;
  earlier planar Noh runs reduced wall heating from 64% to 27% at N=400.
  Reproduce those outcomes under the current solver, then compare smooth evolution,
  wall energy/species budgets, acoustic pulses, Woodward–Colella, and cold/warm Noh.
  **Depends on:** relevant N6 studies; R5 for thermal-wall cases. Coordinate
  time-scaling with N1, but the wall comparison need not wait for cluster TGV data.
  Hold alpha, cadence and rate parameters fixed within each comparison and
  stratify results by time-scaling formulation; N1 owns their fit and time-policy
  decision. Select only the wall-row default here. N11 owns imposed AMR shells.
  **Gate:** improved smooth errors with bounded positivity/repair and conservation
  budgets in both precisions; a documented closure-compatibility table and default
  decision, updated wall calibration, and the repository numerical gate. Do not
  combine cascade4 with the one-sided filter as an assumed safe upgrade: that pair
  has recorded instability even on a smooth pulse.

- [ ] **N6b — Qualify a high-order physical-wall configuration.**
  Evaluate C6 Brady–Livescu with the N6a filter first; evaluate C8 separately.
  Establish solution order, the stable CFL range, and conditioning/error floors
  for the complete derivative/filter/variable-diffusion update, including
  `D(beta D)`. Test smooth compatible walls before shock-loaded and cold-start
  walls. The recorded cold-Noh failures and Float32 wall errors near 1e-3 prevent
  treating these rows as a universal default.
  **Depends on:** N6/N6a and R5 for viscous thermal walls; coordinate startup
  measurements with N3 and any mixed-precision remedy with S4.
  **Gate:** an explicitly bounded supported configuration with measured solution
  order, precision and minimum-extent limits; retain the robust alternative when
  a target fails. A C10 wall closure requires a separate derivation and validation,
  not reuse of a favorable C6/C8 slope.

- [ ] **N6c — Decide whether constant-annihilation roundoff needs a change.**
  Measure derivative residuals on scaled constants and small perturbations over
  large offsets, separating coefficient cancellation, solve conditioning, and
  summation roundoff in both precisions. Compare anchored-difference rows only
  if the defect affects an evolution error or useful precision range.
  **Gate:** record a no-change decision for a roundoff-only result; otherwise
  demonstrate a practical reduction without degrading polynomial accuracy,
  decomposition agreement, inference, or allocations. This is independent of
  the truncation-order fixes in N6a/N14.
  **Code:** [kernels.jl](../src/kernels.jl), [kernels_banded.jl](../src/kernels_banded.jl).

- [ ] **N7 — Complete NSCBC inflow transverse coupling.**
  Add the Yoo–Im transverse terms that exist for outflow but not inflow.
  **Gate:** oblique/acoustic and vortical inflow tests, reflection measurements,
  geometry restrictions, time-dependent targets, and MPI collective consistency.

- [ ] **N8 — Introduce a dispatchable temperature-dependent transport model.**
  Connect the bundled NASA CEA transport data: reader, per-species viscosity and
  conductivity, mixture rule, and mixture-averaged diffusivities, retaining a
  unity-Lewis fallback. Keep a function barrier suitable for later plasma transport.
  **Gate:** independent coefficient checks, mixture limits, thermal/species
  diffusion solutions, and consistent timestep limits. Document the current
  constant-coefficient/single-Schmidt limitation until delivery.

- [ ] **N9 — Validate the bulk species channel beyond its existing cases.**
  Measure `species_flux=:bulk` in 3-D shock/mixing runs and calibrate its inherited
  constants. Compare pressure equilibrium, species bounds, energy budgets, and cost
  with the Fickian channel. Distinguish continuous-model entropy properties from
  any verified property of the complete discrete update.
  **Depends on:** N1 and V1; keep default selection evidence-based.

### AMR numerics

Current node-centered coupling is interpolation/injection with compact interface
closures, not a conservative flux reconciliation. In the review, two-level Sod
mass drift was 1.36e-4; smooth two-level C6 orders were 3.46–3.64.
The follow-up isolates the default divergence closures: fourfold CFL reduction
barely changes those errors, while C6 Brady–Livescu reaches 5.99/5.75 on a
phase-shifted smooth wave at CFL 0.125, with about 1,000 times less error at N=192.
These are serial Float64, unfiltered inviscid results, not production qualification.
The existing designs and fallback analysis remain in [AMR_GPU.md](AMR_GPU.md).

- [ ] **N10 — Bound and reduce interface conservation drift.**
  Measure mass, momentum, energy, and mixing diagnostics over long mixing-layer and
  moving-interface runs, separating same-level, coarse–fine, and regrid defects.
  Set application error budgets; implement the designed surface-flux correction
  when drift exceeds them. Retain SBP–SAT as the documented fallback.
  **Gate:** composite budgets across rank counts, refinement depths, and subcycling,
  with smooth accuracy and reflection checks. Transfer invertibility is not a
  conservation proof.
  Establish comparison budgets before promoting N14–N16 candidates; this task
  owns the conservation correction, while those tasks own spatial accuracy.

- [ ] **N11 — Validate sensors and filters at imposed fine shells.**
  Add targeted crossing-shock reflection gates for closed-edge-clamped sensors;
  measure filter changes to imposed shell nodes and compare one-sided filter rows.
  **Gate:** localized errors/reflections and positivity excursions across interface
  locations, C6/C10, and tiled layouts. Coordinate cadence studies with N1.
  Include N14 closure candidates; physical-wall filter selection remains N6a.

- [ ] **N12 — Check fine-level rates during startup and regrid transients.**
  Measure rate growth over the substeps covered by one root CFL estimate, especially
  at three or more levels. Add a refreshed-coefficient substep check where needed.
  **Gate:** route a violation to the collective rollback/acceptance path from R3;
  an exception inside recursive stepping must not bypass retry handling.

- [ ] **N13 — Settle the default state-validity policy and its species band.**
  Ten shipped cases select `validity = :permissive`, in three groups: the
  mass-fraction band at a filtered species interface, including one at uniform
  p, u and rho where nothing but the filter acts; negative internal energy at
  the Noh wall and behind the Sedov blast; and the near-vacuum a strong shock
  leaves. A default that most shock-capturing runs must opt out of is either
  the wrong default or the wrong threshold, and the two are separable. The
  species test currently borrows `ArtParams.Y_tolerance`, which was calibrated
  as the dead band of a regularization term and not as a validity bound.
  Measure what excursion a converged interface actually carries as a function
  of resolution, decide the threshold from that, and then decide whether
  `:strict` or `:permissive` is the better default.
  **Gate:** the shipped cases pass under the chosen default without per-case
  opt-outs beyond those the physics genuinely requires, and each remaining
  opt-out keeps a bound on affected-cell count and worst defect. Record the
  threshold and its basis in [CALIBRATION.md](CALIBRATION.md).
  **Code:** [problem.jl](../src/problem.jl), [stepcontrol.jl](../src/stepcontrol.jl),
  [cases.jl](../test/cases.jl).

- [ ] **N14 — Select interface divergence closures independently of wall closures.**
  Add an opt-in policy applied only to same-level and coarse–fine interface ends
  of `div_plans`, independently of gradient `interface_rhs` and physical-wall rows.
  Prototype an optional `interface_divergence` closure-source scheme, defaulting
  to `nothing`: require its interior coefficients to match `deriv` and use only
  its closure rows at interface ends. This admits custom schemes without silently
  applying C6 rows to a different interior; reject mismatches during setup.
  Compare C6 cascade3, cascade4, and Brady–Livescu first; C8 is a separately
  qualified extension. Preserve the current default and reject unsupported
  scheme/extent/halo combinations, including an unimplemented C10 closure choice.
  Carry the policy through fine/same-level plan construction, tiled device plans,
  `RegridSpec`, regrid/rebalance rebuilds, and supported restart continuation.
  A5 owns configuration provenance and incompatible-restart rejection.
  **Depends on:** N6's inviscid/interface studies. Obtain N10 conservation budgets
  and N11 reflection/positivity gates before promotion; physical-wall work and R5
  do not block this experiment.
  **Gate:** target at least 5.5 observed solution order for C6 Float64 on multiple
  resolved smooth inviscid fields at both same-level and coarse–fine interfaces,
  with temporal error controlled; document viscous order and the usable Float32
  error floor separately. Compare error magnitudes, acoustic reflection, drift,
  both directions of shock crossing, moving/tiled refinement, and subcycling.
  Test mixed physical/interface ends, both `interface_rhs` settings, both precisions,
  MPI np=2/4/8, and host/device plans; state hardware coverage explicitly.
  Initial opt-in implementation must retain default regression behavior. Deliver
  a recorded selection decision: retain the default, promote a qualified candidate
  for a bounded tier, or keep it experimental. A candidate failing stability or
  N10/N11 checks is not promoted; smooth-only results may justify a documented
  Float64 opt-in, not general sixth-order AMR.
  **Code:** [problem.jl](../src/problem.jl), [rhs.jl](../src/rhs.jl),
  [patches.jl](../src/patches.jl),
  [levels.jl](../src/levels.jl), [regrid.jl](../src/regrid.jl),
  [kernels.jl](../src/kernels.jl), [io_levels.jl](../src/io_levels.jl).

- [ ] **N15 — Design and trial divergence with valid current-stage ghost fluxes.**
  Start only if N14 misses a stated accuracy/stability target or a case requires
  higher-order viscous/C8/C10 interfaces. Compare local inviscid ghost-flux
  evaluation from exchanged state with a phased assemble/exchange/diverge RHS.
  For viscous/artificial fluxes, specify the required gradient/coefficient data,
  coarse–fine representation and interpolation, and subcycled stage-time source.
  A same-level exchange alone does not supply nonconforming or Hermite-time fluxes.
  **Deliver:** a dependency/storage/collective schedule and bounded prototype,
  first at same-level inviscid interfaces, then viscous and coarse–fine interfaces.
  Account for the shared RHS workspace: retain only justified interface data or
  quantify the memory cost of persistent per-patch fluxes. Keep GCL divergence
  and diagnostic freshness consistent; no cross-patch collective may be inserted
  into a sequential per-patch RHS without changing its schedule.
  **Depends on:** the N14 decision and N6 tests for design/prototyping; N10 budgets
  before promotion. Coordinate ownership with A2 and ghost-value accuracy with
  N16. N10 owns flux reconciliation.
  **Gate:** polynomial/RHS consistency at both interface ends, full inviscid and
  viscous evolution orders, reflection and conservation budgets, nested-subcycle
  timing, and no stale data/deadlocks under MPI or device execution. Compare
  memory, allocations, inference and step cost with N14. Reusing gradient plans
  without populating valid flux ghosts is not an implementation of this item.

- [ ] **N16 — Make live transfer order explicit and qualify spatial accuracy tiers.**
  Measure interpolation and restriction separately from divergence, first on
  point samples and then through first/second derivatives and regrids. Live
  prolongation currently hardcodes order 6; the standalone weights support even
  orders through 8. Expose a validated live interpolation-order choice and thread
  it through initial creation, regridding, Hermite shells, and CPU/device chains,
  with sufficient buffer/halo/stencil extents and early configuration checks.
  Retain point-sample semantics; the filtered/deconvolving pair is not an upgrade
  for unfiltered coarse data.
  **Depends on:** N6 for measurement; qualify evolution with the selected N14 or
  N15 coupling and N10's composite budgets. Begin with a C6 target; derive wider
  transfer/closure support for C8/C10 only as a separately measured extension.
  **Gate:** polynomial/value-transfer exactness, derivative consistency, repeated
  regrid error, positivity and composite mass/momentum/energy budgets, followed
  by smooth inviscid and viscous solution orders at fixed physical interfaces.
  An O(h^r) value error can enter first/second derivatives as O(h^(r-1))/O(h^(r-2));
  interpolation order alone is not the acceptance test. Publish separate spatial,
  filter, temporal and subcycled-boundary orders: LSRK and cubic Hermite remain
  fourth order in time, and the C8 filter must be included in any C10 claim.
  Escalate to a compatible conservative/SBP–SAT design only with an explicit
  decision under N10/N15; energy-compatible interpolation is not a drop-in table
  for the current compact operators.
  **Code:** [problem.jl](../src/problem.jl), [transfer.jl](../src/transfer.jl),
  [levels.jl](../src/levels.jl),
  [regrid.jl](../src/regrid.jl), [timestep.jl](../src/timestep.jl).

Boundary/interface sequence: begin R5 and N6 together. N6's inviscid/interface
subset unlocks N14 without waiting for R5; run N6a wall-filter trials alongside
it, adding thermal-wall cases once R5 passes. N6b follows the wall-filter
decision; N6c is an independent roundoff audit. Use N10/N11 to qualify interface
candidates before promotion; invoke N15 only when the smaller closure change
misses a target. N16 transfer measurements may begin with N6, while final accuracy
qualification follows the selected interface treatment. Coordinate temporal
certification with V3 and default filter time-scaling with N1.

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
  Add dedicated temporal-order studies for the full RK update and subcycled
  boundary forcing, with spatial and filter errors controlled. Turn R1–R4 probes
  into durable regressions.
  Separate pure temporal certification here from N6's dt-sensitivity checks;
  qualify the improved N14–N16 coupling, where Hermite error can become visible.
  Add a scheduled full shock-validation battery and explicit Makie extension
  checks; retain HDF5 tests in the package test target and add parallel-HDF5
  execution where the required stack exists.
  **Gate:** CI distinguishes skipped/unavailable coverage from passing coverage.
  KA-on-CPU equality does not substitute for hardware-GPU tests under S1.

## P1: API, state ownership, and reproducibility

- [ ] **A1 — Give precision one coherent public entry point.**
  Normalize EOS, transport, artificial controls, schemes, geometry, and runtime
  types from an explicit precision choice, or reject conflicts at setup with a
  useful message. Storage currently follows `Transport{T}`; Float32 transport
  plus default Float64 `ArtParams` produces a conversion MethodError.
  **Gate:** concise Float32/Float64 input decks, early mixed-type diagnostics,
  and no accidental promotion in audited CPU/device kernels. Coordinate with R2.

- [ ] **A2 — Make cache ownership and freshness explicit.**
  Inventory persistent integrator state, per-patch primitives/geometry, shared RHS
  scratch, and diagnostic storage. Define invalidation after initialization,
  stepping, filtering, regridding, and restart; audit property forwarding and
  `prepared`/`primitives_current` assumptions.
  **Gate:** multi-patch consumers cannot silently read another patch's scratch;
  R1 remains fixed without duplicating every patch workspace.

- [ ] **A3 — Separate solver construction from RHS execution.**
  Move configuration validation, plan/hierarchy construction, and device setup out
  of `rhs.jl` behind explicit construction boundaries. Preserve specialized
  runtime types and per-patch function barriers; setup-only abstract fields do not
  by themselves justify a type-system rewrite.
  **Gate:** unchanged numerical results plus inference/allocation comparisons for
  the existing hot paths; preserve compatibility of supported accessors.

- [ ] **A4 — Tighten and document public runtime contracts.**
  Validate physical/numerical parameter ranges and finite inputs, including
  transport coefficients, CFL/backoff, grid sizes, and primitive composition.
  Document or simplify `step!` clock ownership, absolute `nmax`, initialization
  without reset, cached fields, and scalar versus state-vector APIs.
  Audit rollback across callback effects: schedule rewind does not restore a
  switched boundary or arbitrary user state.
  **Gate:** early actionable errors and explicit continuation/rollback semantics.

- [ ] **A5 — Strengthen restart configuration compatibility.**
  Add versioned EOS parameter/data fingerprints and numerical/transport/boundary
  provenance. Current type names and component names admit identically named
  species with different thermodynamic constants.
  Define exact continuation versus an intentional configuration change; record
  supported overrides and caller-owned callback/writer state.
  **Gate:** incompatible thermodynamics are rejected, equivalent configurations
  round-trip, and existing same-rank/changed-rank continuation tiers remain tested.
  **Code:** [io.jl](../src/io.jl), [io_levels.jl](../src/io_levels.jl).

- [ ] **A6 — Publish a capability and extension matrix tied to setup checks.**
  Cover EOS × backend × precision × metric × patch/refinement mode, including
  static nested levels versus two-level regridding and checkpoint restrictions.
  Correct stale README/reference claims against current code and tested hardware.
  Clarify that `EquationSet` currently owns layout/parity while flux assembly
  remains Navier–Stokes-specific; define the additional hooks needed by H3/H6/H8.
  **Gate:** accepted/rejected combinations and extension examples are tested;
  avoid promising unsupported feature combinations.

## P2: scale, devices, I/O, and geometry

Detailed mechanisms and measurements remain in [AMR_GPU.md](AMR_GPU.md) and
[CLUSTER.md](CLUSTER.md). Patch AMR, device execution, stacked tile launches, and
opt-in Float32 already exist; the tasks below extend or validate them.

- [ ] **S1 — Complete target-machine GPU measurements and resolve the wait stall.**
  Continue the rzadams/MI300A campaign using the existing measurements as a baseline,
  not as an unmeasured port. Characterize/resolve the intermittent ROCm wait stall
  in [rocm_wait_stall_report.md](rocm_wait_stall_report.md) before interpreting
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

- [ ] **S6 — Deliver HDF5 time-series output and verify collective writes.**
  Add `FieldWriter(format=:vtk/:hdf5)` routing through `_write_dump!`; current
  routing is VTK-only. Implement the extension-backed XDMF temporal collection,
  reusing the field descriptors emitted by the frame writer, including species
  expansion and vector component counts.
  Rewrite a complete collection after each completed frame, inline each frame's
  Grid/Time/topology/geometry/attributes, use relative paths, and keep individual
  frames independently readable; do not depend on XInclude reader support.
  **Gate:** scheduled times, restart frame indices, fields/stride/slice, rank-0
  collection ownership, and continued readability after interruption.
  Separately test with parallel libhdf5 built against the run's MPI: ranks owning
  no selected slice must participate with empty selections, not skip H5Dwrite.

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

## P2/P3: high-energy-density physics

H1 is the principal infrastructure dependency for stiff diffusion. These are
separate capabilities with independent verification gates, not a claim that new
equation layouts alone make the current RHS a general multiphysics solver.

- [ ] **H1 — Build implicit diffusion infrastructure.**
  Reuse the distributed banded kernels for ADI or line-relaxation smoothing;
  compare geometric multigrid and Krylov outer solves.
  **Gate:** manufactured constant/variable-coefficient heat conduction in every
  supported metric, distributed residual/convergence studies, and freestream
  preservation. Variable coefficients require a factorization-update policy.

- [ ] **H2 — Add compatible IMEX time integration.**
  Evaluate established IMEX-ARK tableaus before implementing new ones; define a
  compatible explicit/implicit pair and workspace contract.
  **Depends on:** H1. **Gate:** temporal order, stiff stability, source/diffusion
  splitting error, and recovery from a failed implicit solve.

- [ ] **H3 — Add separate ion/electron/radiation energy evolution.**
  Extend the equation/flux/recovery interfaces for `T_ion`, `T_ele`, and `T_rad`,
  electron pressure, and electron–ion equilibration.
  **Depends on:** A6, R3/R4, and H2 for stiff coupling.
  **Gate:** total-energy conservation, equilibrium limits, and independent
  relaxation problems before coupled implosion runs.

- [ ] **H4 — Add electron thermal conduction.**
  Implement flux-limited Spitzer–Härm transport through the implicit solver.
  **Depends on:** H1–H3 and the dispatchable transport interface.
  **Gate:** analytic/manufactured transport limits, limiter behavior, and coupled
  energy budgets.

- [ ] **H5 — Add tabulated EOS support.**
  Implement IONMIX reading and thermodynamically consistent interpolation/inversion;
  consider SESAME later subject to data access/licensing.
  **Depends on:** R3/R4 and A5.
  **Gate:** table-node/interpolation checks, inverse consistency, derivatives,
  phase/domain handling, and an EOS-specific artificial-conductivity scale.

- [ ] **H6 — Add flux-limited radiation diffusion, gray before multigroup.**
  Define opacity and group interfaces plus radiation-energy components.
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
