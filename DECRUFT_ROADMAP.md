# CompactLES.jl Structural De-Crufting & Unification Roadmap

A working plan, kept in the branch so that a session resuming the work has
it. Part 1 records the maintainer's decisions and where execution stands;
Part 2 is the roadmap as the maintainer wrote it, with its line references
as of the commit the work started from (`bc7f6ee`). Where the two disagree,
Part 1 governs. Delete this file when Phase 9 closes.

## Part 1: Decisions and status

### Decisions

| | Decision |
|---|---|
| D1 | Historical bench scripts and their raw output move to `bench/archive/`, which no check parses. |
| D2 | A settled `Ref` toggle is deleted with the A/B tests of its rejected branch; assertions on the production default stay. |
| D3 | The flat `Solver(; ...)` constructor is removed outright in Phase 4, with no deprecation shim. Every caller in `src/`, `test/`, `bench/`, `examples/`, `probes/`, `docs/` and the precompile workload moves to `Problem`/`Numerics`/`AMR`. Scripts under `bench/archive/` are not repaired. |
| D4 | `allocate_state(solver)` returns one unified `SimulationState` on every layout (Phase 6, an API break). |
| D5 | `incompressible.jl`, `ionmix.jl` and `sesame.jl` become internal submodules, `CompactLES.Incompressible` and `CompactLES.EOSTables`. |

Further rulings made during execution:

- **Mask group deferred.** `MASK_CHILD_DERIVATIVE`, `MASK_CHILD_SPECIES_GRADIENTS`,
  `MASK_CHILD_SENSORS`, `MASK_CHILD_RESIDUAL`, `CHILD_MASK_THRESHOLD` and
  `CHILD_MASK_REACH` are kept as switches, in `src/instrumentation.jl`, and
  not hardcoded.
- **Switch-free tests first.** Before keeping a `Ref` for a test's sake, look
  for a test of the same property that needs no switch. `FORCE_PRESYNC_RESTRICT`
  and `REFLUX_GATED` went this way.
- **Promotions.** `SUBDIVIDE_LATTICE` is `AMR(subdivide_lattice = true)`;
  `REFLUX` is `PatchInterfaces(conservative = true)`;
  `LIMITED_GHOST_INTERFACES` is `PatchInterfaces(limited_ghost = false)`,
  documented with its rank-dependence caveat.
- **Phase 2C deferred to Phase 6.** `BUDGET_LEDGER` stays a global until the
  step drivers are merged; the observer parameter then goes onto the one
  merged driver. The ledger keeps its per-stage attribution and its
  boundary-flux closure: the hooks are not coarsened to phase boundaries.
- **Gate cadence.** The full gate runs once per phase, at its boundary.
  Between boundaries a commit takes the checks its change needs: the affected
  test files, a smoke test, and the reference, docrefs and scriptrefs checks
  when their inputs change.
- **Left in place, by ruling.** `THREAD_MIN_WORK` stays in
  `core/threading.jl` as a session tunable, `FIXED_LATTICE_COSTS` beside the
  lattice choice in `amr/regrid_lattice.jl`, `WALL_SIGNAL` in
  `solver/callbacks.jl` and `BLOCK_TRANSFER` in the HDF5 extension. The
  direct assignments to a knob that remain in the tests also stay.

### Status

| Phase | State |
|---|---|
| 0 | Done. |
| 1 | Done. |
| 2A | Done: settled toggles deleted, three promoted, the rest gathered in `src/instrumentation.jl` with `with_knob`. About thirty direct assignments to a knob remain in the tests where the surrounding code was irregular. |
| 2B | Done: `RegridSpec.whole_sensors` replaced by a trailing `mask_child` argument. |
| 2C | Deferred to Phase 6. |
| Checkpoint 1 | Reviewed; the Phase 2 end gate passed at `891a7ee`. |
| 3 | Done: the four large files split, `RefluxCapture` and the regrid records moved ahead of their users, `src/` grouped into `core`, `physics`, `amr`, `solver` and `io`. The phase gate passed at `31e4b8b`. |
| 4 | Done: `RegridSpec` reads the schemes from the solver, the root's slab records live in its `Level`, the flat `Solver(; ...)` constructor is removed and `_Solver` runs as named phases over a `SolverBuild` record. The flat keyword vocabulary survives only in `test/flat_solver.jl`, an adapter for the suites, and the test configurations precompile under the `precompile_tests` preference. The phase gate passed at `517267c`. |
| Checkpoint 2 | Awaiting review. |
| 5 | Done: the compute path takes a `PatchSolver` alone, built by `sole_patch` at the entry points, and `SolverLike` is removed. `Patch` holds `PatchPlans` and `PatchAMR`; `Solver` holds `GridGeometry`, `ClockState` and `WallAccounting`; the limiter holds four groups and leases the workspace scratch it borrows. The phase gate passed at `f2b754f`. Departures from the text: `PatchAMR` is always present, so a configuration keeps one patch type; `positivity` is `Union{Nothing,LimiterState}`, with the single-patch `PatchLimiters` left to Phase 6; the forwarding `getproperty` methods remain. The precompile cost did not fall, and no workload entry or `_cold` barrier was removed. |
| 6 to 9 | Not started. |

Known gaps in coverage from the working environment: no hardware GPU, no
parallel HDF5 against a system MPI, and the Makie suite has not been run.

## Part 2: The roadmap

## Ground Rules for Execution

1. **Zero numerical drift**: Every phase below is a structural refactor. [`test/convergence.jl`](test/convergence.jl) must remain **bit-identical** (orders and error magnitudes) and [`test/validation.jl`](test/validation.jl) must match to four significant digits unless a non-default ablation branch is being removed.
2. **Zero dispatch/allocation regressions**: Record [`bench/jetcheck.jl`](bench/jetcheck.jl) and [`bench/audit.jl`](bench/audit.jl) before Phase 1 and after every `src/` phase. Never add keyword or optional trailing arguments to probed entry points (`compute_rhs!`, `step!`) without updating `bench/audit.jl`'s probe tuple.
3. **Respect repository guards**: Run [`test/docrefs_tests.jl`](test/docrefs_tests.jl), [`test/scriptrefs_tests.jl`](test/scriptrefs_tests.jl), [`test/reference_tests.jl`](test/reference_tests.jl), and [`test/api_surface_tests.jl`](test/api_surface_tests.jl) whenever docs, scripts, `reference/`, or public/exported symbols change.
4. **Preserve rationale comments**: Move derivation and measurement comments with the code they explain (`CLAUDE.md` Conventions); never delete them during splits or deduplication.
5. **Commit granularity**: One logical sub-step per commit on `main` (per `CLAUDE.md`). Complete each phase's gate before advancing.

### Maintainer Decisions Required Up Front

Confirm these five choices before starting Phase 1:

- **D1 (`bench/` archive)**: Move historical one-off sweep scripts + `bench/{results,logs,slurm}` into `bench/archive/` (excluded from `scriptrefs_tests.jl`), or move them to a separate repo/tag and delete from `main`?
- **D2 (A/B ablation tests)**: When deleting settled `Ref` toggles in Phase 2, delete the old A/B comparison testsets (that asserted behavior under the rejected `false` branch), keeping only assertions on the production default?
- **D3 (`Solver(; 40 kwargs...)` compatibility in Phase 4)**: Keep `Solver(; ...)` as a thin compatibility adapter over `Problem`/`Numerics`/`AMR`, or migrate internal callers/tests and deprecate the flat 40-kwarg constructor?
- **D4 (State container API in Phase 6)**: Keep `allocate_state(solver)` returning a bare `ConservedState` on single-patch runs and normalize internally at public boundaries (non-breaking), or return one unified `SimulationState` container across all layouts (cleaner, eliminates `Base.inferencebarrier`, minor API break)?
- **D5 (Unintegrated subsystems in Phase 8)**: Wrap [`incompressible.jl`](src/incompressible.jl), [`ionmix.jl`](src/ionmix.jl), and [`sesame.jl`](src/sesame.jl) in internal submodules (`CompactLES.Incompressible`, `CompactLES.EOSTables`), or extract them into companion packages?

---

## Phase 0: Baseline & Dead-Code Quick Wins

**Goal**: Capture performance/dispatch baselines and remove unambiguous dead code before touching structure.

- Record baseline outputs of `bench/jetcheck.jl`, `bench/audit.jl`, `test/convergence.jl`, `test/validation.jl`, `test/runtests.jl`, and `mpiexec -n 2` / `-n 8 rank_shape=true` `test/mpi_tests.jl`.
- Delete [`_reflux_budget`](src/reflux.jl#L1214) (0 callers across the repo).
- Audit the three bench-only internal helpers — [`_fill_lines!`](src/operators.jl#L263-L272), [`_fill_t!`](src/operators.jl#L496), and `_reduced_matrix` in [tridiag.jl](src/tridiag.jl) (called only by `bench/reducedsolve.jl`): move them into `bench/reducedsolve.jl` or archive with it in Phase 1.

**Gate**: `test/runtests.jl` + `test/scriptrefs_tests.jl`.

---

## Phase 1: Break the `bench/` & Meta-Test Lock-In

**Goal**: Stop 97 historical scratch scripts from acting as hard API contracts that block internal refactoring.

1. **Partition `bench/` into active vs. archived (per D1)**:
   - **Active (`bench/`)**: Keep only scripts referenced by `CLAUDE.md`, `docs/`, `README.md`, or active workflows (~12–15 scripts: `artcal.jl`, `audit.jl`, `boundaryorder.jl`, `coverage.jl`, `derivcost.jl`, `device_solver.jl`, `jetcheck.jl`, `phases.jl`, `repeat.jl`, `tgv_energy.jl`, plus active AMR/MPI benchmarks such as `amrwin.jl`, `rmicost.jl`, `device_mpi.jl`, `interfaceconservation.jl`).
   - **Archive (`bench/archive/` or external tag)**: Move the ~80 one-off historical search/calibration scripts (`neutralsearch8.jl`, `neutralsearch10.jl`, `closuresearch.jl`, `closurequalify.jl`, `closurecertify.jl`, `axisvariant.jl`, `axisrunaway.jl`, and the 8 unreferenced scripts `bcbench.jl`, `composite_faces.jl`, `footprint.jl`, `jetwhere.jl`, `level_filter_accuracy.jl`, `profile.jl`, `stepcost.jl`, `threadscale.jl`), plus `bench/results/`, `bench/logs/`, `bench/slurm/`, and `bench/closuresearch_results.md`.
2. **Scope [`test/scriptrefs_tests.jl`](test/scriptrefs_tests.jl)**:
   - Update `SCRIPT_DIRS` to check `examples/`, `docs/literate/`, `probes/`, and active top-level `bench/*.jl` — explicitly skipping `bench/archive/`.
   - Remove the private `_name_point!` arity checker unless an active script invokes internal `_name_point!` kernels directly.
3. **Update relative links**: Adjust [`reference/CALIBRATION_APPENDIX.md`](reference/CALIBRATION_APPENDIX.md) or [`test/reference_tests.jl`](test/reference_tests.jl) so archived scripts do not block internal symbol cleanup.

**Gate**: `test/scriptrefs_tests.jl`, `test/reference_tests.jl`, `test/runtests.jl`.

---

## Phase 2: Prune Global Mutable `Ref` Switches & Decouple `BUDGET_LEDGER`

**Goal**: Eliminate process-global mutable numerical toggles from `src/` hot paths and remove the 79 scattered conservation-ledger instrumentation calls.

### 2A. Triage all 31 module-level `Ref`s

| Disposition | Targets | Action |
|---|---|---|
| **Delete & hardcode default** (settled ablations) | `GHOST_FLUX_REMAINDER`, `GHOST_REMAINDER_EXTRAPOLATE`, `GHOST_REMAINDER_DEGREE` ([rhs.jl](src/rhs.jl#L34-L42)); `MASK_CHILD_DERIVATIVE`, `MASK_CHILD_SPECIES_GRADIENTS`, `MASK_CHILD_SENSORS` ([rhs.jl](src/rhs.jl#L343-L385)); `MASK_CHILD_RESIDUAL`, `CHILD_MASK_THRESHOLD`, `CHILD_MASK_REACH`, `FORCE_PRESYNC_RESTRICT` ([timestep.jl](src/timestep.jl#L1915)); `SENSOR_INTERFACE_GHOSTS` ([artificial.jl](src/artificial.jl#L468)); `CONSERVATIVE_INTERFACE_FILTER` ([kernels.jl](src/kernels.jl#L594)); `REFLUX_GATED` ([reflux.jl](src/reflux.jl#L828)) | Inline the settled defaults (`true` / `false` / `0.01` / `2` / `5`), delete unreachable fallback branches (keep `_gradient_ring_built` only for reachable `art.enabled == false` + molecular transport configs), and prune A/B tests in `test/level_tests.jl`, `test/conservation_tests.jl`, `test/interface_reflection_tests.jl` per D2. |
| **Promote to explicit config** (legitimate runtime behavior) | `SUBDIVIDE_LATTICE` ([regrid.jl](src/regrid.jl#L1198) — currently mutated at top of `test/hdf5_tests.jl:28` and `test/mpi_tests.jl:98`), `LIMITED_GHOST_INTERFACES` ([positivity.jl](src/positivity.jl#L682)), `REFLUX` ([reflux.jl](src/reflux.jl#L97)) | Move into explicit `AMR` / `PatchInterfaces` / internal `Solver` config fields (or drop the `REFLUX` toggle if positivity/device tests only needed it during bring-up). Remove global test-file mutations of `SUBDIVIDE_LATTICE`. |
| **Consolidate into one `src/instrumentation.jl`** | `THREAD_MIN_WORK`; `FORCE_KA`, `FORCE_DEVICE_EXCHANGE`, `DEVICE_SYNC`; `NODE_SIZE`, `TILE_SLOTS`, `MIGRATION_AUDIT`, `MIGRATION_AUDIT_RESULT`, `RESTRICTION_COUNT`, `LATTICE_TRACE`, `TRACK_DEVICE_TRANSFERS`, `DEVICE_TRANSFER_{BYTES,TIME}`, `REDUCED_TRANSFER_{BYTES,TIME}` | Group remaining test/profiling knobs into one internal module/file with scoped `with_knob(...)` reset helpers so state cannot leak across testsets. Delete counters whose only consumer was archived in Phase 1. |

### 2B. Remove [`RegridSpec.whole_sensors`](src/levels.jl#L4375-L4376) action-at-a-distance flag

- Pass an explicit `mask_child::Bool=true` argument into `_tag_coefficients!` / `compute_artificial!` instead of temporarily mutating `spec.whole_sensors` and reading it inside [`_sensor_mask_patch`](src/rhs.jl#L393-L401).

### 2C. Decouple [`BUDGET_LEDGER`](src/ledger.jl#L59) from production drivers

- Replace the 79 `_ledger*` calls woven through [timestep.jl](src/timestep.jl) (60), [positivity.jl](src/positivity.jl) (16), and [level_sensors.jl](src/level_sensors.jl) (3) with an optional `observer=nothing` parameter passed into the step driver (zero-cost via `::Nothing` dispatch, no global singleton) and coarsen hooks to phase boundaries (step, filter, sync, restrict, reflux, regrid).

**Gate**: Full core gate + `mpiexec -n 2,4,8` `test/mpi_tests.jl` + `test/hdf5_tests.jl` + `bench/jetcheck.jl` + `bench/audit.jl`. Verify `test/convergence.jl` is bit-identical.

---

## Phase 3: Fix Include-Order Displacement & Split God Files

**Goal**: Make `src/` navigable without changing types, semantics, or public API.

1. **Split the four largest files along their existing section headers**:
   - [`src/levels.jl`](src/levels.jl) (4,439 LOC) → `levels_types.jl`, `levels_coupling.jl`, `levels_transfer.jl`.
   - [`src/regrid.jl`](src/regrid.jl) (3,299 LOC) → `regrid_tagging.jl`, `regrid_lattice.jl`, `regrid_swap.jl`.
   - [`src/positivity.jl`](src/positivity.jl) (3,304 LOC) → `positivity_setup.jl`, `positivity_step.jl`, `positivity_kernels.jl`.
   - [`src/timestep.jl`](src/timestep.jl) (2,913 LOC) → `timestep_rk.jl`, `runloop.jl`.
2. **Relocate displaced structs into their owning domains**:
   - Move [`RefluxCapture`](src/patches.jl#L388) out of `patches.jl` into `reflux_types.jl` (included ahead of `patches.jl`).
   - Move [`LatticeCosts`](src/levels.jl#L4252) and [`RegridSpec`](src/levels.jl#L4338) out of `levels.jl` into `regrid_types.jl` (included ahead of `solver.jl`).
3. **Organize `src/` into subdirectories** (still `include`d from `src/CompactLES.jl`):
   - `src/core/`, `src/physics/`, `src/amr/`, `src/solver/`, `src/io/`.

**Gate**: Core gate + `test/docrefs_tests.jl` + `test/reference_tests.jl`. `test/convergence.jl` bit-identical.

---

## Phase 4: Decompose `_Solver`, Unify Construction State, & Fold `patch_grid` into `levels[1]`

**Goal**: Break the 1,000-line, 40-kwarg [`_Solver`](src/construction.jl#L295-L1294) monolith, remove duplicated config records, and eliminate the parallel `patch_grid` vs. `Level` interface-record representation.

1. **Split `_Solver` ([`src/construction.jl:295-1294`](src/construction.jl#L295-L1294)) into focused phase builders**:
   - `_validate_geometry_and_folds`, `_build_root_plans_and_patch`, `_build_patch_grid`, `_build_refined_hierarchy`, `_build_regrid_spec`, `_attach_positivity_and_implicit`.
2. **Deduplicate construction config between [`SchemeSettings`](src/solver.jl#L16-L24) and [`RegridSpec`](src/levels.jl#L4338-L4377)**:
   - Remove the 6 duplicated scheme fields from `RegridSpec` and read them from `solver.schemes`.
   - Merge `RegridSpec.created` (level 1) and `RegridSpec.deep_created` (levels 2+) into a single `created::Vector{Dict{BlockRegion,Int}}`.
3. **Make `Problem` + `Numerics` + `AMR` the canonical construction input (per D3)**:
   - Replace [`_UNREFINED`](src/problem.jl#L1121-L1127) and the 40-kwarg flattening in [`_setup_with_amr_keywords`](src/problem.jl#L1220-L1256) by passing structured config objects directly to internal builders; keep `Solver(; ...)` as a thin compatibility adapter if retaining it.
4. **Collapse level-0 `patch_grid` records into [`Level`](src/levels.jl#L1589-L1614) (`levels[1]`)**:
   - Remove `ghost_sends`, `ghost_recvs`, and `plane_pairs` from [`Solver`](src/solver.jl#L64-L66); store 1-D slab-patch records in `levels[1].{ghost_sends,ghost_recvs,plane_pairs}` with `phases = (d==1, d==2, d==3)`.
   - Unify `exchange_patch_ghosts!` / `average_shared_planes!` ([patches.jl](src/patches.jl#L1473-L1547)) with `_sync_level_records!` ([levels.jl](src/levels.jl#L1953-L1980)), removing all `root ? solver.plane_pairs : lev.plane_pairs[d]` forks across [rhs.jl](src/rhs.jl#L2100-L2110) and [positivity.jl](src/positivity.jl#L1459).

**Gate**: Core gate + `test/capability_matrix.jl` + `test/amr_frontend_tests.jl` + `test/patch_tests.jl` + `test/io_tests.jl` + `test/hdf5_tests.jl` + `mpiexec -n 2,4,8` `test/mpi_tests.jl` + `bench/jetcheck.jl` + `bench/audit.jl`.

---

## Phase 5: Unify Internal Compute on `PatchSolver`, Decompose God Structs, & Shrink `precompile.jl`

**Goal**: Cut duplicate `Solver` vs. `PatchSolver` compilations, reduce `Patch`'s 13 type parameters and `Solver`'s 44 fields, eliminate hidden scratch aliasing, and trim compiler-fighting scaffolding (`_cold`, `precompile.jl`).

1. **Standardize all internal compute routines on [`PatchSolver`](src/patches.jl#L897-L913) only**:
   - At public entry points (`step!`, `compute_rhs!`, `apply_bcs!`, `compute_artificial!`, `max_rate`, `filter_state!`), wrap a single-patch `Solver` in `PatchSolver(solver, patches[1])` once (2-field stack immutable, zero allocation).
   - Change internal compute signatures from `SolverLike` to `PatchSolver`, eliminating duplicate compilation of the entire RHS/BC/filter/sensor tree for both `Solver` and `PatchSolver`.
   - Retire `SolverLike` and demote `Base.getproperty(::Solver, ...)` ([solver.jl](src/solver.jl#L136-L143)) to a cold compatibility convenience accessor for external callers/tests.
2. **Split [`Patch`](src/patches.jl#L482-L582) (37 fields, 13 type params)**:
   - Group operator plans into `PatchPlans` and extract optional AMR/reflux/tiled-sensor fields (`covered`, `overwritten`, `level_scratch`, `ghost_flux`, `sensed_fields`, `child_deep`, `child_masked`, `reflux_captures`, `reflux_deferred`) into `PatchAMR` (`nothing` on single-level runs).
   - Measure every type-parameter change against `bench/jetcheck.jl` and `bench/audit.jl` (preserving concrete plan tuple specialization where `src/rhs.jl:79-90` showed erasing it caused 6–18 kB/RHS allocations).
3. **Slim [`Solver`](src/solver.jl#L35-L129) (44 fields)**:
   - Group related runtime fields into `ClockState{T}`, `WallAccounting`, and `GridGeometry{T,M,St}`.
   - Replace `positivity::Any` with a typed `Union{Nothing,PatchLimiters}` (and tighten `PatchLimiters`'s `Vector{Any}` fields where it doesn't add `Solver` type parameters).
4. **Remove hidden scratch aliasing in [`PositivityLimiter`](src/positivity.jl#L265-L331) (45 fields)**:
   - Group fields into `LimiterWeights`, `LimiterLineBuffers`, and `LimiterCounters`.
   - Replace implicit aliasing into `RHSWorkspace.flux[1,:]`, `flux[2,:]`, `flux[2,1]`, `flux[2,2]`, `flux[3,1]`, and `tmp_b` with an explicit `ScratchPool` / named phase-borrow helpers (`borrow_stage_buffers`, `borrow_filter_buffers`) so buffer lifetimes are structural rather than convention-only.
5. **Re-audit [`src/precompile.jl`](src/precompile.jl) (644 LOC) and the 64 `_cold` barriers**:
   - Re-measure precompile time and `--trace-compile` on `test/mpi_tests.jl` now that `Solver`/`PatchSolver` duality and `Patch` type bloat are reduced; prune obsolete `_cold` barriers and redundant entries in `_mpi_suite_workload`.

**Gate**: Full gate (`test/runtests.jl`, `test/convergence.jl` bit-identical, `test/validation.jl`, full `mpi_tests.jl` at `np = 2, 4, 8`, `test/device_tests.jl`, `test/positivity_mpi.jl`, `test/capability_matrix.jl`, `bench/jetcheck.jl`, `bench/audit.jl`, and before/after precompile timing).

---

## Phase 6: Unify State Representation & Consolidate the 5 Parallel RK/Step Drivers

**Goal**: Eliminate the 67 dual `ConservedState` vs. `Vector{<:ConservedState}` overloads across 21 files and collapse the 5 parallel RK stepping implementations into one engine.

1. **Unify state handling (per D4)**:
   - *If non-breaking*: Add a tiny internal normalizer `_as_states(Q::ConservedState) = (Q,)` / `_as_states(states::Vector{<:ConservedState}) = states` at public boundaries (`run!`, `step!`, `save_checkpoint`, `save_vtk`, `save_hdf5`, `diagnostics.jl`, `viz.jl`, `phases.jl`, `modes.jl`), delete the duplicate single-state internal methods, and drive all sync/reflux checks from `_multipatch(solver)` / `nlevels(solver)` rather than container type (fixing the silent-sync-skip hazard documented in [solver.jl](src/solver.jl#L329-L333)).
   - *If unified `SimulationState` (0.2)*: Return one `SimulationState` type from `allocate_state(solver)` (forwarding array indexing to `states[1]` when single-patch), removing `Base.inferencebarrier` from [solver.jl](src/solver.jl#L335) and retiring dual `Vector{<:ConservedState}` dispatch altogether.
2. **Consolidate the 5 RK/step drivers**:
   - Unify single-patch [`step!`](src/timestep.jl#L145), multi-patch [`step!`](src/timestep.jl#L205), subcycled [`_advance_level!`](src/timestep.jl#L770), and positivity-limited [`_limited_run_step!`](src/positivity.jl#L1217-L1258) / [`_limited_subcycled_step!`](src/positivity.jl#L1303) into **one** level-stepping driver parameterized on a zero-cost concrete stage/filter policy (`StandardStepPolicy` vs. `LimitedStepPolicy`).
   - Verify via `bench/jetcheck.jl` and `bench/audit.jl` that single-patch runs incur zero dynamic dispatch or allocation overhead through the unified driver.

**Gate**: Full gate (`test/runtests.jl`, `test/convergence.jl` bit-identical, `test/validation.jl`, full `mpi_tests.jl` at `np = 2, 4, 8`, `test/positivity_tests.jl`, `test/positivity_mpi.jl`, `test/device_tests.jl`, `bench/jetcheck.jl`, `bench/audit.jl`).

---

## Phase 7: Deduplicate Cross-Cutting Pipelines (Sensors, I/O, Backend Traits)

**Goal**: Remove parallel re-implementations across artificial properties, I/O formats, and host/device dispatch.

1. **Unify [`src/artificial.jl`](src/artificial.jl) (1,513 LOC) and [`src/level_sensors.jl`](src/level_sensors.jl) (693 LOC)**:
   - Factor `artificial.jl` into composable stage primitives (`_sensed_fields!`, `_detect_sensor!`, `_smooth_dir!`, `_assemble_coefficients!`, `_sharpening_gradients!`) accepting an optional inter-tile exchange hook (`nothing` on single-patch, `_exchange_field_ghosts!` on tiled levels), eliminating the duplicated pipeline in `level_sensors.jl`.
2. **Extract a backend-agnostic I/O frame/checkpoint layer across [`src/io.jl`](src/io.jl), [`src/io_levels.jl`](src/io_levels.jl), and [`ext/CompactLESHDF5Ext.jl`](ext/CompactLESHDF5Ext.jl)**:
   - Centralize configuration records, hierarchy metadata, slicing/subsampling, derived-field evaluation, and host staging into one internal `CheckpointBundle` / `FieldFrame` extraction layer so binary checkpoint, VTK, and HDF5/XDMF become thin format encoders/decoders.
3. **Localize `_cpu_storage` / `FORCE_KA` / `DevicePlan` branching behind backend traits**:
   - Consolidate the 96 scattered host-vs-device checks across 20 files into a small trait layer in `pointwise.jl` / `halo.jl` / `lines_device.jl` (`host_copy`, `reduce_max`, `stage_plane!`, `apply_line!`) so physics, AMR, and I/O orchestration stop branching directly on `_cpu_storage`.

**Gate**: Core gate + `test/io_tests.jl` + `test/hdf5_tests.jl` (serial & MPI) + `test/makie_tests.jl` (from docs env) + `test/device_tests.jl` + `mpiexec -n 2,4,8` `test/mpi_tests.jl` + `bench/jetcheck.jl` + `bench/audit.jl`.

---

## Phase 8: Isolate Unintegrated Subsystems (per D5)

**Goal**: Keep the main `CompactLES` namespace and load footprint focused on the compressible solver until peripheral features integrate with `Solver`/`EOS`.

- **[`src/incompressible.jl`](src/incompressible.jl) (486 LOC)**: Move into `module Incompressible` (or extract to a sibling package). Shares none of `Solver`/`Problem`/`Numerics`/`ConservedState`/I/O/AMR/GPU and is neither exported, `public`, nor documented.
- **[`src/ionmix.jl`](src/ionmix.jl) (759 LOC) & [`src/sesame.jl`](src/sesame.jl) (896 LOC)**: Move into `module EOSTables` (or extract to a companion package) until `IonmixEOS <: EOS` and `SesameEOS <: EOS` are implemented.
- **Keep [`src/scriptargs.jl`](src/scriptargs.jl) (105 LOC)** in place: it is small, documented in `docs/src/reference/runtime.md`, pinned in `test/api_surface_tests.jl`, and used across all `examples/*.jl`.

**Gate**: Core gate + `test/api_surface_tests.jl` + `test/incompressible_tests.jl` + `test/ionmix_tests.jl` + `test/sesame_tests.jl` + `test/docrefs_tests.jl`.

---

## Phase 9: Modularize Monolithic Tests & Trim `CLAUDE.md` / `reference/` Churn

**Goal**: Lower day-to-day maintenance drag in `test/`, `CLAUDE.md`, and `reference/`.

1. **Split the three test monoliths (45% of `test/`'s 34,894 LOC)**:
   - [`test/serial_suite.jl`](test/serial_suite.jl) (5,906 LOC, 108 `@testset`s) → per-subsystem test files included from a thin driver.
   - [`test/level_tests.jl`](test/level_tests.jl) (4,535 LOC, 57 `@testset`s) → `level_coupling_tests.jl`, `level_regrid_tests.jl`, `level_reflux_tests.jl`, `level_subcycle_tests.jl`.
   - [`test/mpi_tests.jl`](test/mpi_tests.jl) (5,293 LOC) → phase files under `test/mpi/` behind a thin runner retaining `phases=` and `rank_shape=true`.
2. **Trim [`CLAUDE.md`](CLAUDE.md) (862 LOC → target ≤ 250 LOC)**:
   - Delete the 360-line `## Naming` glossary ([L291–651](CLAUDE.md#L291-L651)) that duplicates docstrings/comments (keep only the ~15-line naming conventions block).
   - Compress `## The gate` and `## Known limitations` into terse checklists linking into `reference/`.
3. **Contain `reference/` churn**:
   - Remove the 71 completed `## Closed` items from [`reference/ROADMAP.md`](reference/ROADMAP.md) (git log is the history).
   - Move `reference/bugreports/` (5 upstream Julia/AMDGPU/ROCm reports) out of `reference/` once filed upstream, leaving one-line issue links in `CLAUDE.md` Traps.
   - Split the 12,608-line [`reference/CALIBRATION_APPENDIX.md`](reference/CALIBRATION_APPENDIX.md) by topic (`reference/appendix/{shocks,walls,amr,transport,implicit}.md`) and update [`test/reference_tests.jl`](test/reference_tests.jl) accordingly.

**Gate**: Full serial suite + `mpiexec -n 2` and `-n 8 rank_shape=true` `test/mpi_tests.jl` + `test/docrefs_tests.jl` + `test/reference_tests.jl` + `test/scriptrefs_tests.jl`.

---

## Recommended Handoff Checkpoints

Pause for maintainer review at each checkpoint:

1. **Checkpoint 1 (Phases 0–2)**: Breaks `bench/` lock-in, removes global mutable ablation switches and `BUDGET_LEDGER` scattering. Highest ROI, lowest risk.
2. **Checkpoint 2 (Phases 3–4)**: Reorganizes `src/`, decomposes `_Solver`, deduplicates config records, and folds `patch_grid` into `levels[1]`.
3. **Checkpoint 3 (Phases 5–6)**: Core architectural simplification (`PatchSolver`-only internals, slimmed `Patch`/`Solver`/`PositivityLimiter`, unified state representation, single RK/level stepper, shrunk `precompile.jl`). Guard every commit with `jetcheck.jl` and `audit.jl`.
4. **Checkpoint 4 (Phases 7–9)**: Cross-cutting pipeline deduplication (sensors, I/O, backend traits), subsystem isolation, test modularity, and `CLAUDE.md`/`reference/` cleanup.
