# Working in CompactLES

Orientation for coding agents, and deliberately short: it carries only what you
need before touching anything. **It does not describe the solver**, and it is not
the place to record a result.

| file | what is in it |
|---|---|
| `README.md` | usage |
| `reference/DESIGN.md` | the numerics — source map, compact solve, folds, GCL, NSCBC |
| `reference/CLUSTER.md` | MPI configuration, launch rules, sizing, measured scaling |
| `reference/CALIBRATION.md` | the artificial-property constants, the CFL restriction, TGV |
| `reference/ROADMAP.md` | positioning, comparisons, open work items |
| `reference/HISTORY.md` | completed phases and their measured outcomes |
| `reference/AMR_GPU.md` | the patch-AMR + GPU implementation plan, staged with gates |
| `reference/IMMERSED.md` | the immersed-boundary design: level-set bodies, blend imposition |
| `reference/MODE_TRUNCATION.md` | azimuthal mode truncation plan for the pole CFL squeeze |

Read those for anything about *what* the code does, *where* it runs, or *where it
is going*. This file is about *how to work on it*. When something you measure
turns out to be durable, it belongs in one of the files above, with a one-line
pointer here only if an instance would go wrong without it.

## Environment

Compat bounds live in `Project.toml`; MPI.jl 0.20 is the one that matters,
since its API changed either side of that.

`mpiexec` is usually not on PATH — MPI.jl provides whatever launcher this
checkout is configured against, which is a JLL binary on a workstation and the
system MPI (often the scheduler's launcher) on a cluster. Ask it rather than
hardcoding a path:

```bash
julia --project=. -e 'using MPI; MPI.mpiexec(c -> println(c))'
```

Before reading any scaling or timing number, check what you are running on.
`Threads.nthreads()`, the physical core count, and whether the cores are
uniform all change the interpretation — a hybrid performance/efficiency-core
desktop will spread threads across both and understate scaling. `ThreadPinning`
supplies the topology queries `clusterprobe.jl` uses. Nothing pins threads yet —
the querying half is what has been validated.

Windows checkouts may have CRLF line endings. Helper scripts that match
multi-line text against `\n` will silently find nothing; match a line at a time.

**On a cluster, read `reference/CLUSTER.md` before doing anything else, and do
not skip it because the code runs.** MPI.jl defaults to a bundled JLL that
satisfies the scheduler perfectly well on one node and never reaches the
interconnect off it — **27x–66x** slower on rzhound, with no symptom but speed.
`LocalPreferences.toml` is per-project, so `--project` silently determines which MPI
you get. `reference/CLUSTER.md` also holds the launch-line rules
(`--cpu-bind=threads`, `-t 1`, and why a rank must never hold both SMT threads of
a core), the sizing tools, and the measured scaling.

`Manifest.toml` is gitignored, so a fresh checkout needs `Pkg.instantiate()`
before anything runs. `bench/tgv_energy.jl` is the intended first real workload
on a cluster: the one bench script whose reductions are all collective and which
reproduces serial numbers bit-for-bit under decomposition.

## The gate

Run all of it before calling a change safe.

```bash
MPIEXEC=$(julia --project=. -e 'using MPI; MPI.mpiexec(c -> print(c))')

julia --project=. test/runtests.jl        # 78 testsets, 0 failures
julia --project=. test/convergence.jl     # measured orders, see below
julia --project=. test/validation.jl      # shock-capturing battery, ~25 s
for np in 2 4 8; do
  "$MPIEXEC" -n $np julia --project=. test/mpi_tests.jl   # 95/95 each
done
julia --project=. bench/jetcheck.jl       # inference
julia --project=. bench/audit.jl          # allocation + non-concrete SSA
```

`test/hdf5_tests.jl` covers the HDF5 extension and is skipped by the gate above:
HDF5 is a `[weakdeps]` entry, so it is not loadable from the package
environment. `runtests.jl` prints when it skips. To run it, use an environment
carrying both CompactLES and HDF5, serially and under `mpiexec` — the
decomposition-independent restart writes on one process grid and reads on
another, which np = 1 cannot exercise. `hdf5_parallel()` reports which write
backend the libhdf5 build selects; it is `false` on a workstation, and the
collective path can only be exercised where a parallel libhdf5 built against
the run's MPI exists.

`test/convergence.jl` prints measured orders against regression guards baked
into the file: C6 6.01, C10 10.04, C6 wall closures 3.17, cylindrical axis odd
3.71 / even 3.00, resolved-θ 3.71, spherical origin 2.99. **For a change not
meant to affect numerics these should come out bit-identical, down to the error
magnitudes.** A moved digit means you hit something real — chase it before
moving on.

`test/validation.jl` prints measured errors against guards baked into its
header. Unlike the convergence orders these are **not** bit-reproducible: each
case integrates thousands of steps through a nonlinear sensor, so an arithmetic
reassociation anywhere in the artificial-property path moves the fourth
significant figure. Four digits is the level to compare at; a moved *third*
digit is real. The cases themselves live in `test/cases.jl`, shared with
`bench/artcal.jl` so the calibration study and the guards cannot drift apart —
add a case there, not in either consumer.

`bench/jetcheck.jl` and `bench/audit.jl` print counts, not pass/fail. Record
them before a change and compare after, reading the delta rather than the
absolute count. Most remaining dispatch sites are `Metric` / `EOS` /
`BoundaryCondition` field reads behind function barriers, one dispatch per
array pass rather than per point, so treat a *new* site as the signal. The counts overlap
between entry points (`step!` contains `compute_rhs!`), so compare like with
like and never sum them.

`audit.jl`'s inference probe reads `code_typed` at a spelled-out signature. A
**keyword argument on a probed entry point, or an optional trailing argument the
probe omits**, resolves that signature to the short forwarding method rather than
the body, and the reported count falls to 1 while measuring nothing.
`compute_rhs!` and `step!` therefore carry a trailing `Bool` positionally. If you
add another optional argument to either, extend the probe tuple in the same
commit.

`bench/coverage.jl`'s header has the coverage sequence. Serial alone reaches
94.8% of executable lines and the full set with MPI 97.2%; the difference is
distributed-solve and off-rank-fold code that only a decomposed run touches.
Julia's `.cov` output omits methods that were never compiled, so the percentage
flatters you — watch the executable-line count too.

**The test suite does not import `bench/` or `examples/`.** They stay green
while broken. Run them after any cross-cutting change.

## Naming

Names are spelled out rather than abbreviated. Current vocabulary:

- `solver`, `decomp`, `n_global`, `n_local`, `n_halo`, `n_halo_d`, `offset`,
  `neighbors`, `send_buf`/`recv_buf`, `sub_rank`/`sub_size`, `pad` (the
  per-dimension halo pad, as a local)
- `n_species`, `n_cons`, `i_mom`, `i_energy`, `Y`, `cp_mix`
- `mu_art`, `beta_art`, `kappa_art`, `D_art`, `C_mu`/`C_beta`/`C_kappa`/`C_D`,
  `mu_sensor` (`:strain` or `:velocity`), `beta_sensor` (`:strain`,
  `:gated_strain`, `:dilatation` or `:ungated_dilatation`), `reduction` (`:sum`
  or `:max`), `smoother` (`:gaussian` or `:compact`), `detector` (`:delta4` or
  `:d8`)
- `grad_u`, `grad_T_ion`, `grad_Y`, `strain_mag`, `sensor`, `sensor_sp`
- `inv_J`, `area_d`, `inv_h`, `inv_r`, `cot_over_r`, `coord_shift`, `flux`
- `filter_interval` (cadence in steps) vs `filter_cfl` (the reference CFL at
  which a filter pass is full strength; 0 disables the relaxation), `filter_weight`
- `deriv_plans`, `filter_plans`, `line_solver`, `plan` (a DirPlan) vs `plane`
  (a wall plane), `fold`, `pair`
- `plane_profile`, `profile_spacing`, `mix_width`, `molecular_mixing`,
  `quad_weight`, `cell_measure`
- `eos_phi`, `eos_dphi_dY`, `art_conductivity_scale`, `species_energy`,
  `mixture_temperature` (the EOS contract; the whole list is at the top of
  `physics.jl`)
- `control` (a `StepControl`), `max_rate`, `predicted_dt`, `check_step`,
  `dt_prev`, `rate_prev`, `savepoint`
- `trigger` (an `AtTime` / `EveryTime` / `EveryStep` / `WhenState`), `effect!`,
  `fired!`, `next_time`, `rewind!`, `landing_steps`,
  `switch!`/`switched` (a `SwitchableBC`)
- `writer` (a `FieldWriter`), `frame_prefix`, `collection`, `wall_io`
- `region` (a `BlockRegion`: global offset plus extent, the patch-layout and
  HDF5 hyperslab unit and deliberately not a `Decomp`), `owned_region`,
  `hdf5_parallel`
- `patch` (a `Patch`: per-patch state split out of `Solver`), `patch_grid`,
  `patches`, `patch_regions`, `backend` (an `AbstractBackend`; `CPUBackend`),
  `interface_rhs` (`:extended` or `:onesided`), `div_plans` (divergence plans,
  = `deriv_plans` except at interface ends), `ghost_sends`/`ghost_recvs`/
  `plane_pairs`, `sync_patches!`, `eachpatch`; a routine below the step
  drivers takes a `SolverLike` (single-patch `Solver` or `PatchSolver`), and
  a single-patch `Solver` forwards patch-owned property names to its sole
  patch, so `solver.rho` and `solver.decomp` still read as before
- `refine` (a `BlockRegion` in root node space), `level_transfer` (a
  `LevelTransfer`), `level_restriction` (`:inject` or `:filter`),
  `prolong_level_ghosts!`, `restrict_level!`, `sync_levels!`,
  `LEVEL_BUFFER`, `RESTRICT_MARGIN`; `Patch.h` is the patch's own spacing
  (h/3 on a level-1 patch), which the property forwarding serves as
  `solver.h`
- `subcycle` (the Berger–Oliger mode flag), `subcycled_step!`,
  `save_level_box!`/`hermite_level_shell!` (the Hermite box, `box_Q0` ..
  `box_dQ1`), `regrid` (a `RegridSpec`), `regrid_interval`,
  `tag_threshold`/`tag_buffer`, `tagged_region`, `regrid!`
- `pointwise!` (the shared launcher of every per-point loop: `Array` storage
  takes `@threaded`, device storage a KernelAbstractions kernel),
  `pointwise_ka!`, `FORCE_KA` (test/bench toggle), and the `_point!` suffix
  for a per-point body. A body takes plain arrays and scalars, never the
  solver, and never a `Type` argument — a `Type` inside the launcher's
  Vararg defeats specialization and turns the body call into a per-point
  runtime dispatch (measured 9× on `assemble_fluxes!`).
- `FieldVector`/`FieldMatrix` and `field_tuples` (the launchable forms of
  `Y`, `D_art`, `grad_u`, `grad_Y`, `flux`: zero-cost host wrappers that
  adapt to isbits `DeviceFieldVector`/`DeviceFieldMatrix` tuples at device
  launch). Never hand a bare `Vector`/`Matrix` of arrays to a device
  kernel — it hangs in kernel-argument adaptation instead of erroring —
  and never hold the tuple form on the host: runtime tuple indexing cost
  `assemble_fluxes!` 3× on the `@threaded` path when it was tried.
- `DeviceBackend` (an `AbstractBackend` wrapping a KernelAbstractions
  backend, behind `field(backend, decomp)` and `allocate_state`),
  `device_plan`/`DevicePlan` (device mirror of a `DirPlan`/`BandPlan`:
  fill, sweep, spike correction and scatter as KA kernels in a
  (lines × n) layout, one thread per line; the reduced interface stage
  stays host-side through the wrapped plan's `line_solver`, so the device
  method of `apply_along!` is collective exactly as the host one is), and
  `colwise` (dim 1 mirrors the `solve_col!` banded arithmetic so the
  KA-CPU comparison stays bitwise per dimension)
- `refresh_primitives!`, `mixture_density`, `boundary_plane` (the in-flight
  state-query API; primitives are stale inside a callback, see the
  `refresh_primitives!` docstring)
- `gidx` (interior indices → padded) and `interior_index` (its inverse); a
  padded index goes through the latter before reaching `xcoord`
- `validate_bc` (the setup-time boundary-condition hook), `unit_scalefactor`
- `outer_indices` (the flattened outer iteration space of a pointwise nest; see
  Threading), `prepared`/`primitives_current` (the trailing flag by which `run!`
  tells `step!` that the state is already exchanged and its primitives current)

**Temperature is `T_ion`.** There is one temperature today; the name keeps
`T_ele` / `T_rad` free for a 2T or 3T model without a second API break. Bare
`T` is the element-type parameter and nothing else.

Short names that are deliberate — don't "fix" them: `Q`, `dQ`, `du`, `d`
(dimension), `sp` (species), `I` (CartesianIndex), `σ`/`σf`/`σg` (parity
signs), `ξ` (computational coordinate), `h`, `c`, `p`, and `s` as a band offset
in `banded.jl` / `operators_banded.jl`, where it is the matrix-index convention
the surrounding comments define (`Ab[q+1+s, i]`).

## Traps

**MPI collectives cannot sit below an early return.** `deriv_along!` and
friends are distributed solves along a dimension, so *every* rank must call
them. A boundary routine that returns early on ranks not owning the wall plane
deadlocks. Both `correct_rhs!` methods in `nscbc.jl` hoist their collectives
above the `plane === nothing` return for this reason — follow that shape when
adding a boundary condition. The symptom is zero CPU on every rank, not a
crash, so it looks like a hang rather than a bug.

**A boundary condition that changes mid-run must change on every rank at the
same step.** This is the same collective trap from the other side: `SwitchableBC`
forwards to `after` only once switched. If `after` runs collectives that
`before` does not, as `NSCBCOutflowBC` does, then ranks disagreeing about the
switch is a deadlock, not a wrong answer. `WhenState` reduces its condition
across the communicator for exactly this reason, so drive a switch from a
`Callback` and never from a rank-local test. `AtTime` and `EveryStep` are safe
without a reduction because `t` and `step` advance identically everywhere. The
MPI suite pins this with a condition that is deliberately true on one rank only;
removing the reduction hangs that test rather than failing it.

**Minimum local extent per dimension.** `plan_direction` errors when a rank's
block is too small for the scheme: C6 needs 5 points, C10 needs 7, and the C8
filter needs 9. The filter is the binding constraint, so a grid that is fine
for derivatives can still fail once filtering is on — which is why transverse
extents of 12 or 16 are common in the test suite. Rank count multiplies this:
`n_global[d]` must stay ≥ 9 × `dims[d]`.

**Timing noise.** Run-to-run spread is easily 10–20% for an identical
configuration. Order-of-magnitude results are safe; few-percent ones are not
resolved by a single run. `bench/phases.jl` (per-phase budget) is more useful
than a flat sampling profile, which is dominated by the compact line solves and
says little.

**Julia soft scope.** A top-level `for` in a script that reassigns a variable
bound outside it throws `UndefVarError`. Wrap script bodies in a function.

**A run that fails does not stop.** Losing positivity drives the diffusive rate
in `compute_dt` up until `dt` collapses, and the run then grinds forever at no
progress — so a sweep that visits bad configurations must pass a low `nmax`, and
`run!` now applies `StepControl` floors so this raises `SolverFailure` instead.
`primitives!` substitutes benign placeholders wherever ρ ≤ 0, which is why the
positivity check reads ρ out of `Q` directly in `max_rate` rather than trusting
`solver.rho`.

## Threading

`@threaded <work> for ... end` (`src/threading.jl`) uses `Threads.@threads`
only when `work >= THREAD_MIN_WORK` (32768, override via `CL_THREAD_MIN_WORK`)
*and* the loop has more than one trip; it runs serially otherwise. Allocation is
per-region-per-thread rather than per-point, so without the threshold small cases
pay spawn cost for nothing.

Total work is not the whole test, because only the loop `@threaded` wraps is
divided. A nest threaded on its outermost index runs serially whenever the third
dimension is collapsed, which covers every pointwise loop of a planar
`(nx, ny, 1)` run. **A pointwise nest therefore iterates
`outer_indices(n2, n3)`**, a single flattened `CartesianIndices` over its two
outer indices unpacked with `j, k = Tuple(jk)`, which divides over the second
dimension instead. Write a new one that way. Iteration order is unchanged, so
this is not a numerics change.

The backend choice is tasks, not Polyester, and that was measured rather than
assumed. The numbers and the reasoning are in the `@threaded` docstring —
read it before reaching for `@batch`, including the note on what makes the
decision worth revisiting.

## Conventions

- Keep source lines under ~95 characters. Four existing lines exceed it; don't
  add more.
- Comments explain *why*, and several encode a measurement or a derivation
  (`threading.jl` on the threading backend, `metric.jl` on the discrete GCL,
  `folds.jl` on the antipodal butterfly, `decomposition.jl` on the
  `Dims_create` signature). Move them with the code; don't delete the
  reasoning.
- Prose: avoid overuse of "Claudisms" including agency for inanimate things,
  setup-and-payoff, "what matters is/the thing that/is what makes/worth keeping",
  emphatic fragments, editorializing, and em-dash asides. The reader expects
  clear academic prose explaining the package - do not disappoint them. You are
  not writing clickbait and do not need to micromanage the reader's emotional state.
  Most prose in this repository is model output: if you notice surrounding prose
  has poor style while making an edit, stick to these guidelines and flag the
  prose for correction.
- `bench/` is scratch tooling, not tests: `audit.jl` (allocation, inference),
  `jetcheck.jl` / `jetwhere.jl` (dispatch sites, and where they come from),
  `phases.jl` (RHS phase budget), `profile.jl`, `scaling.jl`, `threadscale.jl`,
  `bcbench.jl`, `coverage.jl`, `artcal.jl` (artificial-property sweeps),
  `nohprobe.jl` (per-step Noh state probe: where β\* is, where the internal
  energy goes negative, and what the symmetry cell is doing),
  `foldorder.jl` (convergence error split by region of the line — the fold end
  against the outer wall, which a global max norm conflates),
  `filterrate.jl` (whether the state filter dissipates per application or per
  unit time), `amr_transfer.jl` (AMR transfer-pair conditioning plus the
  Stage 1 measurement battery: sampling convention, round-trip orders,
  sensor-injection amplification, pollution decay),
  `pointwise_ka.jl` (the G1 acceptance table: each pointwise phase timed
  through `@threaded` and the KernelAbstractions CPU backend),
  `device_bringup.jl` (the G1 kernels on an actual GPU — needs an
  environment carrying a device package; see the header for the
  collection-argument hang it exists to warn about),
  `tgv_energy.jl` (Taylor–Green kinetic-energy budget split
  by dissipation channel; the one bench script that runs usefully under
  `mpiexec`, and the intended first workload on a cluster).
- Scripts take their settings from `ARGS`, not the environment: positional
  values first, then `key=value`, parsed by `script_args` (`src/scriptargs.jl`)
  against a defaults `NamedTuple` that doubles as the schema. Give a new script
  the same shape. An unknown key is an error there on purpose — the environment
  form it replaced answered a typo by running the default for the length of the
  job and saying nothing. The two remaining `CL_*` variables
  (`CL_THREAD_MIN_WORK`, `CL_ERROR_BACKTRACE`) are read from inside the package,
  where there is no `ARGS` to consult.
- Wrap an MPI driver in `mpi_main` and report progress with `ProgressLog`
  rather than hand-rolling either. An uncaught exception is 448 stacktraces at
  448 ranks; a hand-rolled progress loop got at least one of printing-from-one-
  rank, flushing, and not-timing-its-own-reduction wrong every time. Both
  docstrings carry the reasoning. Ctrl-C has no code fix — pass
  `--handle-signals=no` for sweeps.
- A sweep over configurations expected to include bad ones must pass a low
  `nmax`. A configuration that loses positivity does not crash: the diffusive
  rate in `compute_dt` climbs until `dt` collapses and the run grinds, so one
  bad point costs more wall time than the whole sweep. `test/cases.jl` threads
  an `nmax` through every case for this.
- The 1-D cases run single-threaded on purpose: each threaded region is well
  under `THREAD_MIN_WORK`, so `-t auto` buys nothing and ~4% CPU on a 24-thread
  box is the expected reading, not a bottleneck.
- Run artifacts (`*.dat`, `*.vtr`, `*.pvtr`, `*.ckpt`, `*.cov`) are gitignored.
  Prefer `git add <paths>` over `git add -A` after running examples.
- **Commit to `main`.** This is a single-maintainer repository and the default
  agent habit of opening a branch per change is noise here — do not create one
  unless asked. Commit only when asked, and run the gate above first.

## Known limitations

Each of these cost someone real time to establish. The measurements and the
rejected hypotheses are in the file named — read it before re-deriving any of
them.

- **Strong shocks need `cfl ≤ 0.15`.** The failure starts at the symmetry plane
  (wall, axis or origin cell) as the shock forms, *not* ahead of the front and
  *not* from the one-step lag in `compute_dt`. Both of those were the standing
  explanations and both are measured wrong: the lag was tested with a rate
  predictor to 30 steps of lookahead, and β\* is measured to reach three to five
  times further ahead of the front than the damage extends while sitting at or
  near its own domain maximum on the cell that fails.
  A dilatation-keyed β\* was tested and rejected: the ceiling does not move, and
  it loses both converging Noh geometries at the coordinate fold. Gating the
  *strain* sensor on compression (`beta_sensor = :gated_strain`) does move it,
  0.15 → 0.2, but only for the cylindrical geometry, and it does so by relieving
  the axis cell rather than the shock. `StepControl(retries = 4)` beats a
  globally lowered CFL, because the restriction is a startup one.
  The eighth-derivative detector (`detector = :d8`) lifts the restriction
  outright at the planar wall and the cylindrical axis, 0.2 → 1.0 or beyond,
  and costs the spherical origin 0.4 → 0.25. **Every discretization-order
  candidate is now ruled out**, including the fold closure, which is sixth to
  seventh order at the fold — the third-order figure long attributed to it is
  the outer wall's, read through a global max norm (`bench/foldorder.jl`). The
  origin cell instead evacuates during a startup transient at a fixed physical
  time that every configuration passes through, so treat this as a robustness
  problem at a symmetry cell, not a numerics problem at a fold.
  → `reference/CALIBRATION.md`
- **κ\* is singular as `T_ion` → 0**, so a cold ambient below p ≈ 1e-3 collapses
  the diffusive timestep. `art_conductivity_scale` is an EOS dispatch point, so a
  tabular model can supply its own; the gas models still divide by `T_ion`.
- **The spherical origin fold is much less forgiving than the cylindrical axis.**
  It needs initial data resolved over ≳3 cells and will not take Noh's singular
  t = 0 start, both of which the axis handles. Nobody has worked out why.
- **The compact filter, not the Cook artificial properties, is what holds this
  solver together — and it has never been calibrated.** At 128³ TGV the filter
  supplies 37% of the energy sink yet removing it kills the run, while removing
  the artificial properties entirely does not. So every `C_mu` number is
  conditional on `compact_filter(0.45)` applied every step, and `C_mu` itself is
  active but not yet fitted. → `reference/CALIBRATION.md`
- **`compute_artificial!` is ~31% of the multicomponent RHS**, in the filter
  line-solves that smooth the sensors, one sweep per species. At `n_species == 2`
  that machinery is measurably a no-op (`D*_1` and `D*_2` agree to 4.8e-16), and
  it only earns its cost at three or more species. Cutting it is a numerics
  decision (shared vs per-species sensor), not a code tweak.
- The NASA CEA thermo and limited transport databases ship verbatim in `data/`
  with their Apache license and notice. `read_nasa9` handles multi-interval
  thermo records; the transport table is not yet connected to `Transport`.
- Cluster-side open questions — whether `ThreadPinning`'s pinning API would buy
  anything, and the unexplained ~4300x SMT-sibling collapse — are in
  `reference/CLUSTER.md`.
- Wanted: a `bench/` runner taking medians over repeated *processes*, to get
  under the 10–20% run-to-run spread; and `FoldSpec` parameterization, which
  would close several remaining JET dispatch sites.
