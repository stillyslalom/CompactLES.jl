# CompactLES task roadmap

Prioritized open work for compressible, variable-density mixing and implosion.
The September 2026 source review adds runtime and API corrections to the existing
numerics, validation, AMR/GPU, and high-energy-density (HED) backlog.
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

## P0: runtime correctness

- [ ] **R1 — Make diagnostics independent of integrator state.**
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

- [ ] **R2 — Guarantee clock progress and reachable endpoints.**
  A Float32 run with Float64 `tfinal=0.7` stalls at 0.699999988079071,
  repeatedly taking a 1.1920929e-8 remainder that cannot advance its clock.
  Define endpoint conversion/tolerance semantics and check progress after all
  clipping, including callback landing. Return a diagnosed failure for an
  unrepresentable advancing step instead of running to `nmax`.
  **Gate:** Float32/Float64 endpoints on both sides of representable values,
  large restart times, scheduled callbacks, and subcycling all terminate with
  documented time accuracy. Avoid an endless test by bounding steps.
  **Code:** [timestep.jl](../src/timestep.jl), [callbacks.jl](../src/callbacks.jl).

- [ ] **R3 — Validate accepted and returned states under an explicit policy.**
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

- [ ] **R4 — Make NASA-9 recovery failure observable.**
  `mixture_temperature` currently returns its estimate after 30 iterations or
  nonpositive mixture cv without a final residual/status check. Outside-range
  polynomial evaluation also proceeds silently.
  Add residual-based success criteria, diagnosed failure, and an explicit
  extrapolation policy; use a safeguarded inversion where the EOS admits a bracket.
  **Gate:** interval joins, temperature extremes, invalid compositions, and
  nonconvergence behave consistently in both precisions; connect failure to R3.
  **Code:** [physics.jl](../src/physics.jl).

## P1: numerical credibility

### Filtering, regularization, and boundaries

- [ ] **N1 — Calibrate filtering and settle its time-scaling policy.**
  The rate-scaled `filter_cfl` mechanism is delivered but opt-in; the default still
  dissipates per application. Complete the van Rees dissipation-history
  digitization and fit alpha, cadence, and reference CFL jointly against 128³ TGV
  histories and spectra. Measure shock-battery sensitivity and smooth-turbulence
  budgets under subcycling, variable dt, retries, and shortened output steps.
  **Depends on:** R1–R3; cluster time and independent reference data.
  **Deliver:** a reproducible fit and explicit default decision in
  [CALIBRATION.md](CALIBRATION.md). Do not fit under one formulation and then
  silently switch to the other.

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

- [ ] **N6 — Quantify boundary-order and constant-annihilation errors.**
  The review measured C6/C8/C10 periodic orders 6.01/8.00/10.04, default C6 wall
  order 3.17, and default filter one-pass wall order 1.88.
  Add complete evolution studies separating derivative closures, filter closures,
  temporal error, and fold versus outer-wall error; document useful closure choices
  and Float32 conditioning limits.
  Measure residuals on scaled constants before adopting anchored-difference rows.
  **Gate:** demonstrated practical benefit for a closure change; record a
  roundoff-only result without unnecessary rewrites.
  **Code:** [kernels.jl](../src/kernels.jl), [convergence.jl](../test/convergence.jl).

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
The existing designs and fallback analysis remain in [AMR_GPU.md](AMR_GPU.md).

- [ ] **N10 — Bound and reduce interface conservation drift.**
  Measure mass, momentum, energy, and mixing diagnostics over long mixing-layer and
  moving-interface runs, separating same-level, coarse–fine, and regrid defects.
  Set application error budgets; implement the designed surface-flux correction
  when drift exceeds them. Retain SBP–SAT as the documented fallback.
  **Gate:** composite budgets across rank counts, refinement depths, and subcycling,
  with smooth accuracy and reflection checks. Transfer invertibility is not a
  conservation proof.

- [ ] **N11 — Validate sensors and filters at imposed fine shells.**
  Add targeted crossing-shock reflection gates for closed-edge-clamped sensors;
  measure filter changes to imposed shell nodes and compare one-sided filter rows.
  **Gate:** localized errors/reflections and positivity excursions across interface
  locations, C6/C10, and tiled layouts. Coordinate cadence studies with N1.

- [ ] **N12 — Check fine-level rates during startup and regrid transients.**
  Measure rate growth over the substeps covered by one root CFL estimate, especially
  at three or more levels. Add a refreshed-coefficient substep check where needed.
  **Gate:** route a violation to the collective rollback/acceptance path from R3;
  an exception inside recursive stepping must not bypass retry handling.

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
