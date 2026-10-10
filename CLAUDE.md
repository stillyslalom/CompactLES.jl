# Working in CompactLES

Orientation for coding agents, kept short: it carries only what you
need before touching anything. **It does not describe the solver**, and it is not
the place to record a result.

| file | what is in it |
|---|---|
| `README.md` | usage |
| `reference/DESIGN.md` | the numerics — source map, compact solve, folds, GCL, NSCBC |
| `reference/CLUSTER.md` | MPI configuration, launch rules, sizing, measured scaling |
| `reference/CALIBRATION.md` | the calibrated defaults and which setting to change for an instability or an over-dissipated solution |
| `reference/CALIBRATION_APPENDIX.md` | every measurement, one section per instrument: the sweeps, dead ends and null results behind the defaults |
| `reference/ROADMAP.md` | positioning, comparisons, open work items; a closed item is one line naming its commit |
| `reference/AMR_GPU.md` | the patch-AMR + GPU design as delivered, measured lessons, roadmap |
| `reference/IMMERSED.md` | the immersed-boundary design: level-set bodies, blend imposition |
| `reference/MODE_TRUNCATION.md` | azimuthal mode truncation plan for the pole CFL squeeze |
| `reference/IMPLICIT.md` | the implicit-diffusion and IMEX design for the HED items: staggered operator, preconditioned Krylov solve, ARK pair |
| `docs/` | the Documenter site: `make.jl` and the `src/` pages. `docs/src/tutorials/` and `docs/build/` are generated (from `docs/literate/`) and gitignored, so edit `literate/`, never `src/tutorials/` |

Read those for anything about *what* the code does, *where* it runs, or *where it
is going*. This file is about *how to work on it*. Each kind of information
has one place: a measured digit goes in the appendix, once, in the section of
the instrument that produced it, and a newer measurement replaces the older
one rather than annotating it; the git log is the history, so a closed roadmap
item is one line and its commit message is the account; every other file gets
at most three sentences and a link, or nothing. Results of a session are
reported in the reply, not recorded in the repository. `test/reference_tests.jl`
enforces the layout, and a pointer appears here only if an instance would go
wrong without it.

## Environment

Compat bounds live in `Project.toml`; MPI.jl 0.20 requires special attention
because its API changed on either side of that release.

`mpiexec` is usually not on PATH. MPI.jl provides whatever launcher this
checkout is configured against, which is a JLL binary on a workstation and the
system MPI (often the scheduler's launcher) on a cluster. Query it for the
launcher; do not hardcode a path:

```bash
julia --project=. -e 'using MPI; MPI.mpiexec(c -> println(c))'
```

Before interpreting any scaling or timing number, identify the hardware.
`Threads.nthreads()`, the physical core count, and whether the cores are
uniform all change the interpretation; a hybrid performance/efficiency-core
desktop will spread threads across both and understate scaling. `ThreadPinning`
supplies the topology queries `clusterprobe.jl` uses. Nothing pins threads yet;
only the querying half has been validated. The JLL `mpiexec` on Windows sets
no affinity mask either (`-affinity_layout` has no effect without the `smpd`
service), so ranks migrate across the performance and efficiency cores and a
timing at eight ranks carries a 3–4% process-to-process spread at 64³ per
rank; form ratios within a process, as `bench/derivcost.jl` does.

Windows checkouts may have CRLF line endings. Helper scripts that match
multi-line text against `\n` will silently find nothing; match a line at a time.

**On a cluster, read `reference/CLUSTER.md` before doing anything else, and do
not skip it because the code runs.** MPI.jl defaults to a bundled JLL that
satisfies the scheduler perfectly well on one node and never reaches the
interconnect off it: **27x–66x** slower on rzhound, with no symptom but speed.
`LocalPreferences.toml` is per-project, so `--project` silently selects the MPI
implementation. `reference/CLUSTER.md` also holds the launch-line rules
(`--cpu-bind=threads`, `-t 1`, and why a rank must never hold both SMT threads of
a core), the sizing tools, and the measured scaling.

`Manifest.toml` is gitignored, so a fresh checkout needs `Pkg.instantiate()`
before anything runs. Precompiling the package takes about a minute and a
half, most of it the workload in `src/precompile.jl`, which runs the test
suites' solver configurations at np = 1 under a singleton `MPI.Init` so that
a test rank compiles a quarter of what it otherwise would; it is skipped
under a system MPI. Two of its blocks sit behind preferences that are off
by default: the device block behind `precompile_device`, since only
`test/device_tests.jl` and the MPI suite's device phases execute that path,
and the Float32 block behind `precompile_float32`, since every process that
loads the image reads that block and only the suites' Float32 testsets and
phases use it. Turn both on before running the serial or MPI suite locally,
as CI does, or those suites compile the trees themselves
(`PRECOMPILE_DEVICE` and `PRECOMPILE_FLOAT32` in that file have the call).
Julia 1.11+ keys the cache on content, so `touch`
does not rebuild it; force one with
`Base.compilecache(Base.identify_package("CompactLES"))`, though a changed
preference rebuilds on its own. `bench/tgv_energy.jl` is the intended first production workload
on a cluster: the one bench script whose reductions are all collective. Those
reductions are order-dependent `Allreduce(+)`, so it reproduces serial numbers
to round-off (of order 1e-14 relative), not bit-for-bit; `test/mpi_tests.jl`
measures the decomposed compact solve it rests on at np = 4.

## The gate

Before committing, run the checks required by the affected behavior below.
For mixed changes, take the union of the requirements. If the impact is
unclear, run the full gate. Report what ran, any baseline changes, and any
required coverage that remains unavailable.

- **Prose, comments, and docstrings only:** run `test/docrefs_tests.jl`, and
  `test/reference_tests.jl` when `reference/`, `README.md` or this file
  changed. Build the documentation when changing Documenter configuration,
  page generation, or executable documentation examples.
- **Benchmarks, examples, and standalone tooling:** run the affected scripts
  with a small representative workload. Changes to shared package code also
  require the checks below.
- **Tests only:** run the affected suite, including its enclosing runner when
  the file is not standalone. Changes to numerical guards or stored references
  require the corresponding numerical suite and an explanation of the new
  baseline.
- **Solver behavior or shared package infrastructure:** run the core gate
  below. This includes changes to operators, boundaries, thermodynamics,
  transport, filtering, timestepping, state/storage, AMR, and precompilation.
  An isolated change to output, plotting, or a utility requires its affected
  tests and the serial suite; add numerical or MPI checks when its effects
  reach those paths.
- **Setup checks:** a change to what `Solver` or `setup` accepts or rejects
  also runs `test/capability_matrix.jl`, which builds every accepted row of
  the supported-combinations page (weekly in CI, about 80 s).
- **Distributed algorithms:** run the core gate and the full MPI suite at
  2, 4, and 8 ranks for changes to decomposition, communication, ownership,
  distributed solves, folds, or AMR synchronization.
- **Performance-sensitive code:** also run `bench/jetcheck.jl` and
  `bench/audit.jl`, comparing before and after in the same environment.
  This applies to hot-path arithmetic, dispatch, types, storage, threading,
  and changes to the audit probes themselves.
- **Extensions and device paths:** also run the affected feature checks.
  HDF5 changes require `test/hdf5_tests.jl` in an environment carrying HDF5;
  Makie changes require `test/makie_tests.jl` from the docs environment.
  Run these serially and under MPI when restart, collective output, or
  distributed extraction is affected. Changes to the parallel-HDF5 backend
  require that backend; device-path changes require relevant hardware-GPU
  checks. A skipped check is unavailable coverage, not a pass.

The core gate is:

```bash
MPIEXEC=$(julia --project=. -e 'using MPI; MPI.mpiexec(c -> print(c))')

julia --project=. -O1 test/runtests.jl   # compile-bound; see the note there
julia --project=. test/convergence.jl
julia --project=. test/validation.jl
"$MPIEXEC" -n 2 julia --project=. -t 1 test/mpi_tests.jl
"$MPIEXEC" -n 8 julia --project=. -t 1 test/mpi_tests.jl rank_shape=true
```

The 8-rank selection is `RANK_SHAPE_PHASES` in `test/mpi_tests.jl`, which
`.github/workflows/CI.yml` runs through the same `rank_shape=true`.
It exercises rank-dependent block sizes and process-grid shapes.

For the full gate, replace the two MPI commands with full-suite runs at
2, 4, and 8 ranks, and add both performance audits and
`test/capability_matrix.jl`. Use the full gate for
broad numerical refactors, release validation, or uncertain impact.
Affected extension and hardware checks remain additional requirements.

A passing `runtests.jl` already satisfies the documentation-reference check;
do not run it separately unless documentation inputs changed afterward.

**Run `test/docrefs_tests.jl` before every push unless it already passed,
standalone or through `runtests.jl`, for the current documentation inputs.
Rerun it after any subsequent docstring or `docs/` edit.**
The documentation job is the slowest leg of CI and the last
to report, and it has failed repeatedly on a `[`Name`](@ref)` whose target no
`@docs` block renders: Documenter cannot resolve the link and fails the
build. The file resolves every `@ref` in the rendered docstrings and pages
the way Documenter will, without building the docs, in about a second after
the package loads; it also checks that every `@docs` entry is a documented
binding and that every exported name with a docstring is rendered somewhere
(`checkdocs = :exports`). A private name may be linked with `@ref` only if
it is listed in a `@docs` block on some page; otherwise write it in plain
backticks. `runtests.jl` includes the same check as its last testset.

The `159 testsets` and the `242/242` are counted by different mechanisms and are
not comparable. `runtests.jl` runs everything inside one top-level
`@testset` and prints one summary tree at the end: the top row totals the
`@test`s, and each row beneath it is one `@testset` of the suite, the ones
its includes contribute too (`float32_validation.jl`, `device_tests.jl`,
`patch_tests.jl`, `level_tests.jl`, `io_tests.jl`, `docrefs_tests.jl`, and
the device/adapt testsets); the `159` counts those rows. A failure is
recorded and the suite runs on, so one run reports every failure, and the
outer testset raises at the end. `mpi_tests.jl` uses `Test` not
at all and counts individual `check(name, val, tol)` assertions, printing
`passed/total` and exiting nonzero on any failure. Adding a testset moves the
first number, adding one assertion moves the second, and neither is a fraction
of the other, so treat both as figures to record before a change and compare
after.

`test/hdf5_tests.jl` covers the HDF5 extension and is skipped by the gate above:
HDF5 is a `[weakdeps]` entry, so it is not loadable from the package
environment. `runtests.jl` records a skipped extension suite as a broken test
and prints why; the test argument `require=hdf5,makie` turns the skip of a
named suite into a failure, which is how CI's serial job asserts that the
HDF5 suite ran. To run it, use an environment
carrying both CompactLES and HDF5, serially and under `mpiexec`; the
decomposition-independent restart writes on one process grid and reads on
another, which np = 1 cannot exercise. `hdf5_parallel()` reports which write
backend the libhdf5 build selects. It is `true` on this workstation under the
Windows HDF5_jll, an MPI build over Microsoft MPI, and under the WSL conda
stack `reference/CLUSTER.md` describes, which is where a parallel libhdf5
against a system MPI is exercised without a cluster; the serialized relay
needs a serial libhdf5 (conda `hdf5=*=nompi*`) pointed at through the same
preference.

`test/makie_tests.jl` covers the Makie extension and is skipped the same way,
but for a different reason: HDF5 is in `Project.toml`'s test target and
CairoMakie is not, because resolving and precompiling it for two
testsets is out of proportion to what they cover. **`Pkg.test` therefore never
runs them.** The Makie extension is verified only from the docs environment,
which carries CairoMakie, and under `mpiexec` for the
decomposition-independent profile; CI's documentation job runs both.

`test/convergence.jl` prints measured orders against regression guards baked
into the file: C6 6.01, C8 8.00, C10 10.04, C6 wall closures 3.18
(`:cascade3` 3.17, `:cascade4` 4.02, `:brady_livescu` 5.88), C8 wall closures
3.18 (`:brady_livescu` 7.91), C10 wall closures 3.18, filter pass
`:cascade` 1.88 / `:onesided` 8.07,
symmetry planes 6.00 / 6.00 (C6), 8.05 / 7.95 (C8) and 10.23 / 10.17 (C10)
on an even and an odd field, with a filter pass between planes at 7.88,
cylindrical axis odd 3.76 / even 2.99, resolved-θ 3.76, spherical origin
2.97, closure rows on a polynomial 3.00 / 3.00 / 4.00 / 5.00 / 3.00 / 7.00 /
3.00 (C6 `:neutral3`, `:cascade3`, `:cascade4`, `:brady_livescu`, then C8
`:neutral3` and `:brady_livescu`, then C10 `:neutral3`), staggered
operators 6.04 / 6.07 / 5.97 / 6.06 periodic (D_s, G, interpolation, L) and
6.00 / 6.02 / 5.99 / 6.02 under the wall mirror and 6.01 / 5.98 / 6.00 /
5.98 between symmetry planes (L on an odd field 6.01), L on the metric 6.06
(stretched), 5.96 (cylindrical axis), 5.07 (resolved-θ axis), 4.03
(spherical origin), 5.33 (spherical poles) and 1.00 (curved wall), wall
evolution 4.01 (`:cascade3` 3.93, cascade filter 1.94, one-sided filter
3.90, `:brady_livescu` 5.73, viscous no-slip 4.00, viscous slip 4.00,
shear mode 4.67), a pulse on slip walls 4.88 (Cartesian line) / 4.62 (r-z
annulus), against the run at a third of the spacing,
symmetry-plane evolution 4.46 (one-sided filter 4.69, C8 4.00, C10 4.00,
viscous slip with a tangential shear 6.04),
interface evolution 6.79 (two patches), 6.01 (two levels), 3.62 / 6.01
(two levels under the closure rows, C6 / `:brady_livescu`), 6.00 (three
levels subcycled), 6.87 (two levels filtered), 5.93 (two levels,
`:brady_livescu` with the `:d8` detector, closure rows), 6.01 (two levels,
pentadiagonal filter), a level at a slip wall 4.68 and at a symmetry plane
6.33 (the fine wall window against the uniform run at the fine spacing;
1.55 under the `:filter` restriction), three levels at a symmetry plane
5.49 (the second level's plane window, filtered), a level at the r-z axis
6.43 / 6.47 (inviscid / viscous, filtered, the fine axis window) and at the
corner of the axis and a plane at z = 0 7.11 (the
fold window), a level on an r-z annulus 5.70 / 5.41 (inviscid / viscous, the
fine interface window against the uniform run at the fine spacing), a level
at an NSCBC outflow 6.45 (the fine face window against the uniform run at
the fine spacing), a level across a periodic seam 6.24 (the fine interface
window against the uniform run at the fine spacing), temporal order 3.99 / 4.09
(Dirichlet / NSCBC inflow data),
1.00 / 3.85 (two levels, global step / subcycled), 4.02 / 3.63 (the
additive pair, periodic / no-slip walls at Pr = 0.007). The default closure of all three derivative presets is
`:neutral3`; the coordinate-singularity studies close
their outer end with a wall and use the default rows, and the interface
studies take the default ghost fluxes but for the three closure-row studies,
where an interface divergence selects the cascade rows or the source's.
A symmetry plane plans no closure row; its evolution rows are measured
against a five-times-finer folded mirror, since the mirror at the same
spacing reproduces the run to round-off.
The evolution
rows share `test/smooth_cases.jl` with `bench/boundaryorder.jl`; add a smooth
case there, not in either consumer.
**For a change not
expected to affect numerics these should come out bit-identical, down to the error
magnitudes.** A moved digit indicates a real change; chase it before moving
on. Each study asserts both a wide guard, which fails when the order
has regressed, and a ±0.02 guard against the number above, which fails when it
has drifted; the two print different diagnostics. Update a `recorded` value only
together with the list here and the table in the file's header.

`test/validation.jl` prints measured errors against guards baked into its
header. Unlike the convergence orders these are **not** bit-reproducible: each
case integrates thousands of steps through a nonlinear sensor, so an arithmetic
reassociation anywhere in the artificial-property path moves the fourth
significant figure. Four digits is the level to compare at; a moved *third*
digit is real. The cases themselves live in `test/cases.jl`, shared with
`bench/artcal.jl` so the calibration study and the guards cannot drift apart;
add a case there, not in either consumer.

`bench/jetcheck.jl` and `bench/audit.jl` print counts, not pass/fail. Record
them before a change and compare after, and read the delta, not the absolute
count. `jetcheck.jl` reports zero dispatch sites at every probed entry point
but the intended ones, so *any* report elsewhere is a regression. The
baseline is one site in `apply_bcs!` (`enforce!`) and six in
`compute_rhs!`, and eight in `step!`: those seven and the `_cold` barrier in
front of `_build_reflux!`, which rebuilds the coarse-fine captures after a
layout change. Both `compute_dt` probes report one: the `_cold` barrier in
front of `_local_max_rate_launch`, which host storage takes only under the
`FORCE_KA` test toggle. Of the six in `compute_rhs!`, three are the boundary hooks
(`correct_flux!`, `correct_rhs!`, and `sensor_mirror` through the detector's
`_face_mirror`): face conditions are stored abstractly on the `Patch` so that a
combination of them does not recompile the right-hand-side tree, and the
docstring there has the measurement. Two are `_cold` barriers
(`_cold_bulk_gradients!`, `_cold_ghost_flux_divergence!`), which keep paths a
configuration never takes out of every solver type's compile; the note on
`_cold` in timestep.jl has the rule. One is `_launch_fluxes!` on `Val` of the
species count, taken once per whole-array flux assembly so that the per-point
species sums unroll without the species count entering the solver type. The
script drops reports from inside a dispatched callee, which JET infers at
abstract types that never run, and probes those callees at concrete types
instead. The plan-operator handle of
`compute_artificial!` is `@nospecialize`, which JET's inference does not see,
so its dispatches do not appear here. The counts overlap between entry points
(`step!` contains `compute_rhs!`), so compare like with like and never sum them.

`audit.jl`'s inference probe reads `code_typed` at a spelled-out signature. A
**keyword argument on a probed entry point, or an optional trailing argument the
probe omits**, resolves that signature to the short forwarding method, not the
body, and the reported count falls to 1 while measuring nothing.
`compute_rhs!` and `step!` therefore carry a trailing `Bool` positionally. If you
add another optional argument to either, extend the probe tuple in the same
commit.

`bench/coverage.jl`'s header has the coverage sequence. Serial alone reaches
94.8% of executable lines and the full set with MPI 97.2%; the difference is
distributed-solve and off-rank-fold code that only a decomposed run touches.
Julia's `.cov` output omits methods that were never compiled, so the percentage
overstates coverage; watch the executable-line count too.

**The test suite does not import `bench/` or `examples/`.** They stay green
while broken. After a cross-cutting API change, identify and run the affected
consumers with small representative workloads.

## Naming

Names are spelled out in full. Current vocabulary:

- `solver`, `decomp`, `n_global`, `n_local`, `n_halo`, `n_halo_d`, `offset`,
  `neighbors`, `send_buf`/`recv_buf`, `sub_rank`/`sub_size`, `pad` (the
  per-dimension halo pad, as a local), `free_communicators!` (called when a
  `Decomp` is permanently dropped; MPI frees nothing until GC otherwise),
  `owns_communicators` (false on one rank of `COMM_WORLD`/`COMM_SELF`, which
  a `Decomp` borrows instead of building a topology; `cart_rank` covers that
  case)
- `n_species`, `n_cons`, `i_mom`, `i_energy`, `Y`, `cp_mix`
- `mu_art`, `beta_art`, `kappa_art`, `D_art`, `C_mu`/`C_beta`/`C_kappa`/`C_D`,
  `C_Y`/`Y_tolerance` (the mass-fraction bound and its dead band),
  `mu_sensor` (`:strain` or `:velocity`), `beta_sensor` (`:strain`,
  `:gated_strain`, `:dilatation` or `:ungated_dilatation`), `reduction` (`:sum`
  or `:max`), `smoother` (`:gaussian` or `:compact`), `detector`
  (`:species_d8`, the default, `:d8` on the mass and mole fractions and δ⁴ on
  every other sensed field; `:delta4`; or `:d8`), `species_flux`
  (`:partial_density`, the default, one `D_b` on the partial densities with
  the mass flux carried into momentum and energy;
  `:bulk`, the same `D_b` on every conserved variable; `:fickian`, the
  per-species flux with the correction velocity), `D_b` (built by
  `bulk_diffusivity!` from `mole_fraction` and the mass fractions and stored in
  every `D_art[k]`, with the conserved gradients in the workspace's `grad_Q`),
  `_shared_species_diffusivity` (whether a solver's channel uses `D_b`: false
  for `:fickian` and for a single species), `_species_gradients_skipped`
  (whether `compute_rhs!` leaves `grad_Y` to the one boundary condition that
  reads it)
- `C_sharpen` (the interface sharpening flux's Γ/c, 0 off) and
  `sharpen_width` (its ε in local spacings), `_sharpening` (whether a
  solver's channel carries it), `_sharpening_fluxes!` / `_sharpen_flux_point!`
  (the flux, held in the `grad_Q` columns past the partial densities),
  `SHARPEN_GATE` / `SHARPEN_NORMAL` / `SHARPEN_MAX_SPECIES`,
  `_sharpening_rate` (its term in `max_rate`), `_material_density_ratio`
- `grad_u`, `grad_T_ion`, `grad_Y`, `strain_mag`, `sensor`, `sensor_sp`
- `inv_J`, `area_d`, `inv_h`, `inv_r`, `cot_over_r`, `coord_shift`, `flux`
- `filter_interval` (cadence in steps) vs `filter_cfl` (the reference CFL at
  which a filter pass is full strength; 0 disables the relaxation), `filter_weight`
- `StateFilter`, `PatchInterfaces`, `Execution` (the `Numerics` groups; the
  solver keeps the flat names, so `StateFilter.interval` is `filter_interval`
  and `PatchInterfaces.flux` is `interface_flux`), `_with` (the copy-with-changes
  constructor behind `Numerics(base; ...)` and its siblings)
- `deriv_plans`, `filter_plans`, `line_solver`, `plan` (a DirPlan) vs `plane`
  (a wall plane), `fold`, `pair`, `symplane` (the per-dimension pair of
  `SymmetryPlaneBC` flags the constructor folds on; a self-paired fold with
  `pair === nothing`), `paired_fold` (whether any fold owns a butterfly, the
  only case that allocates `pairbuf`/`pairout`)
- `plane_profile`, `profile_spacing`, `mix_width`, `molecular_mixing`,
  `quad_weight`, `cell_measure`, `line_profile` (a transverse-plane average
  along an axis) vs `line_sample` (the field on one grid line)
- `eos_phi`, `eos_dphi_dY`, `artificial_conductivity_scale`, `species_energy`,
  `mixture_temperature` (the EOS contract; the whole list is at the top of
  `physics.jl`)
- `control` (a `StepControl`), `max_rate`, `predicted_dt`, `check_step`,
  `dt_prev`, `rate_prev`, `savepoint`
- `validity` (the `StepControl` policy: `:strict`, `:permissive` or `:repair`),
  `state_report` (the collective sweep) and its `StateReport`, `state_valid`,
  `check_validity` (the pure verdict, as `check_step` is), `validate_state!`
  (sweep, apply the policy, report), `StateGuard`/`state_guard` (the same per
  accepted step, as a callback), `state_admissibility` (the EOS dispatch point
  deciding whether a point is in the model's domain; there is no universal
  e > 0 test) and the `STATE_` flags it returns,
  `mixture_temperature_status` and the `TEMPERATURE_` flags (whether the
  NASA-9 inversion converged and whether it was extrapolated),
  `extrapolate` (`Nasa9Mixture`: `:linear`, the default, `:polynomial` or
  `:missing`)
- `trigger` (an `AtTime` / `EveryTime` / `EveryStep` / `WhenState`), `effect!`,
  `fired!`, `next_time`, `rewind!`, `landing_steps`
- the phase change `setup(solver, Q; bcs, sources, transport, numerics)`
  (phases.jl), `inputs` (the `Solver` field holding the `Problem` and
  `Numerics` that `setup` built it from), `_check_phase`, `_carry_phase!`
  (the checkpoint image written to memory by `_write_checkpoint` and read
  back by `_read_checkpoint!`), `_carry_accounts!`, `_prime_phase!` (the
  artificial coefficients of a phase switching them on, through `_prime_art!`)
- `writer` (a `FieldWriter`), `frame_prefix`, `collection`, `wall_io`,
  `piece` (one patch's block of one rank in a VTK dump; the multiblock
  writer names it `_patch_piece_name` and lists it in the `.vtm` through
  `_write_multiblock`), `_patch_slice` (a root plane in a patch's node
  space), `_vtk_ghost_points` / `VTK_HIDDENPOINT` (the blanking array)
- `n_art_fields`, `art_block`, `set_art_block!` (the artificial coefficient
  record a checkpoint carries beside `Q`; `max_rate` sizes the next step
  from it), `HierarchyRecord` / `LevelRecord` (what a checkpoint carries of
  the hierarchy), `hierarchy_record` (rank 0 assembles and broadcasts it),
  `restore_hierarchy!` (brings a solver to the recorded hierarchy),
  `_replace_level!` (rebuilds level 1 from a region list and owner ranges,
  every tile fresh), `_held_tiles` (this rank's `(level, tile, patch
  index)` triples)
- `region` (a `BlockRegion`: global offset plus extent, the patch-layout and
  HDF5 hyperslab unit and not a `Decomp`), `owned_region`,
  `hdf5_parallel`
- `patch` (a `Patch`: per-patch state split out of `Solver`), `patch_grid`,
  `patches`, `patch_regions`, `backend` (an `AbstractBackend`; `CPUBackend`),
  `interface_rhs` (`:extended` or `:onesided`), `div_plans` (divergence plans,
  = `deriv_plans` except at interface ends), `ghost_sends`/`ghost_recvs`/
  `plane_pairs`, `sync_patches!`, `eachpatch`; a routine below the step
  drivers takes a `SolverLike` (single-patch `Solver` or `PatchSolver`), and
  a single-patch `Solver` forwards patch-owned property names to its sole
  patch, so `solver.rho` and `solver.decomp` still read as before
- `rhs_workspace` (an `RHSWorkspace`: the scratch of one right-hand-side
  evaluation, `grad_u`, `grad_T_ion`, `grad_Y`, `strain_mag`, `sensor`,
  `sensor_sp`, `tmp_a`, `tmp_b`, `ring_buf`, and `flux`, held once per
  distinct padded local extent on a rank rather than once per patch, since a rank
  advances its patches in sequence and none of it outlives the evaluation
  that filled it; a level's tiles share one set), `rhs_workspace_pool` and
  `rhs_workspace!` (the pool and its lookup on that extent; a regrid seeds
  the pool from the sets its patches already hold, departing tiles
  included). The names reach callers through the same property forwarding, so
  `solver.grad_u` and `solver.tmp_a` read as before, and
  `_is_workspace_prop` is the branch that routes them
- `refine` (a `BlockRegion` in the parent level's node space, or a vector
  of them for a nested chain), `levels` (a `Vector{Level}`, root first;
  each `Level` holds `index`, `patches` as indices into `solver.patches`,
  `transfers`, one `LevelTransfer` per patch with `coarse_indices` /
  `fine_index` / `imposed`, and the level's own `ghost_sends` /
  `ghost_recvs` / `plane_pairs`), `tile` (the lattice edge in parent
  nodes; 0 = one patch per level), `_tile_span`/`_lattice_tile`/
  `_level_tiles`/`_tile_faces`, `_in_shell`, `phases` (the dimension-phased
  level sync), `_combine_planes!`/`_seed_planes!`, `_erode`, `refined_region`,
  `level_regions`, `nlevels`, `level_comm` (a `LevelComm`: the rank subset
  owning one level, holding its communicator, whether this rank is in it,
  whether the communicator was split for it, and its size; the root's spans
  the whole run, a level that fits every rank holds its parent's
  communicator, and no split exists in that case), `root_level_comm` /
  `absent_level_comm` / `split_level_comm` / `free_level_comm!`,
  `_level_ranks` (the largest rank count one tile admits under
  `_amr_dims`), `owners` (per tile, its rank range in the level's
  communicator) and `_tile_owners` (the ranges of a level's tiles, laid along
  a Morton curve by tile volume, through `_sfc_order`/`_morton` and
  `_rank_counts`), `group` (a `TileGroup`: this rank's group communicator,
  its rank range and whether it was split; one per rank per level, shared
  by the tiles of the range), `split_tile_comm` / `free_tile_group!` /
  `absent_tile_group`, `tiles` (the tile of each patch a rank holds on a
  level; a rank holds the root and the tiles of its own range only, so
  `LevelTransfer.fine_index` and `coarse_local` are local indices, 0 where
  the rank holds no piece), `level_restriction` (`:inject` or
  `:filter`), `prolong_level_ghosts!`, `restrict_level!`, `sync_levels!`,
  `_sync_level!`, `LEVEL_BUFFER`, `RESTRICT_MARGIN`; `Patch.h` is the
  patch's own spacing (h/3 on a level-1 patch), which the property
  forwarding serves as `solver.h`
- `_place_tiles` (the stored-ownership rule at a regrid: survivors keep
  their range in `Level.owners`, fresh tiles take the free ranks or join
  the nearest survivor's group), `rebalance` (the `Solver`/`Numerics`
  keyword and `RegridSpec` field: max/mean busy-time threshold, 0 off),
  `rebalance_persist` (keyword; `RegridSpec.persist`), `streak`,
  `imbalance`, `wall_mark`/`wait_mark`/`wall_regrid`, `_rebalance_due!`
  (the collective check; returns the Allgathered per-rank busy time when
  due), `_measured_weights`, `busy` (a rank's step wall less its waiting),
  `wall_wait`/`wait_total` (`Solver` fields: time inside the run-wide
  collectives and the level record exchange, charged through `_wait!`),
  `_migrate_tile!` (a moved tile's solution, old owners' blocks to new
  owners' blocks, point-to-point from the two Allgathered block tables;
  `old_piece` is the rank's old state and decomposition held past the
  swap for the sends), `_gather_tile`/`_carry_over!` (the box regrid's
  replicated carry, kept as the migration's reference), `MIGRATION_AUDIT`
  / `MIGRATION_AUDIT_RESULT` (the test hook comparing the two bitwise)
- `_level_artificial!` (the artificial coefficients of a tiled level with a
  shared face, computed over the whole level in stages separated by exchanges
  over the level's records, `src/level_sensors.jl`), `sensed_fields` (a
  `Patch` field: a tile's strain magnitude and dilatation between those
  stages), `InterfaceSmoothPlans` (a refined patch's smoother plans, `ghost`
  reading the interface ghost layers the pass fills, `closed` the scheme's
  rows), `coefficients_current` (`compute_rhs!`'s second trailing flag)
- `interface_flux` (`:closure` or `:ghost`), `_ghost_viscous` (whether the
  ghost path carries the molecular flux), `ghost_flux` (a `Patch` field: the
  molecular flux of each interface dimension, interior values from the
  right-hand side, ghost values from the flux records or the gradient ring),
  `_level_ghost_fluxes!` (the second phase of a level's right-hand side),
  `ShellGradients`/`gradients` (a `LevelTransfer`'s gradient ring beside the
  shell ring), `default_interpolation_order` (the order a solver given none
  takes, from `deriv` and `interface_flux`)
- `subcycle` (the Berger–Oliger mode flag), `subcycled_step!` and the
  recursive `_advance_level!` beneath it, `save_level_boxes!`/
  `hermite_level_shell!` (the Hermite box, `box_Q0` .. `box_dQ1`),
  `regrid` (a `RegridSpec`), `regrid_interval`, `tag_threshold`/
  `tag_buffer`, `tagged_region`, `regrid!`
- `coupling` (a `Level`'s `LevelCoupling`: its rank's pieces of the
  point-to-point traffic with the parent, built by `build_level_coupling`
  from two Allgathered tables of `OwnedBlock`s, one `CouplingPiece` per
  block of one message), `_exchange_boxes!` (parent state to the tiles'
  owners: the shells, `save_level_boxes!`, `_fill_tiles_from_parent!`),
  `_exchange_restriction!` (coincident samples to the parent, through
  `_restrict_tiles!`), `_pair_tags` (a record's tag, its index among the
  messages between its two ranks)
- The tag criteria, a union evaluated by `_tag_sweep!` over the parent
  level's state, each a `pointwise!` body writing into `RegridSpec.tags`:
  `_tag_delta4_point!` (always on, `tag_threshold`), `_tag_sensor_point!`
  (`tag_sensor_threshold`, on the artificial diffusivity number
  `((μ* + β*)/ρ + κ*/(ρ c_p) + max D*)/(c h)` read from the persistent
  coefficient arrays, never from the pooled workspace's `sensor`),
  `_tag_gradient_point!` (`tag_gradient_threshold`, mass-fraction change
  per cell), `_tag_vorticity_point!` (`tag_vorticity_threshold`), and
  `_tag_predicate!` (`tag_predicate`, a user `(patch, I) -> Bool` closure).
  A body writes a level, `TAG_MARK` above the threshold or `TAG_HOLD` above
  the threshold over `untag_ratio` (keyword; `RegridSpec.untag_ratio`),
  and `_tag_level`/`_raise_tag!` combine them; the hold level keeps an
  existing tile (or the current box) and calls for no new one.
  `tile_lifetime` (keyword; `RegridSpec.lifetime`) is the minimum age of a
  tile in regrid checks, `checks` counts the checks and `created` (per
  current tile region, the check it was created at) is the tag history,
  derived from the reduced flags on every rank
- `covered` (a `Patch` field: per node, the orthants of its quadrature
  cell a child level covers, as bits; host `UInt8` over the padded
  extent, written by `_fill_covered!` at setup and at every regrid from
  the child regions every rank of the parent's level holds),
  `uncovered_fraction` / `uncovered_plane_fraction` (the volume and
  in-plane weights a mask byte gives), `_composite` (whether a solver's
  diagnostics take the multi-patch forms), `_composite_profile` /
  `_plane_accumulate!` (the composite plane average at the root's
  stations). A diagnostic's `Vector` form is the composite, masked one; the
  single-array form is the one-patch quadrature and applies no mask
- `junction` (one coarse-fine face of one tile as the conservative coupling
  treats it, its `box` the parent lines crossing it, `_reflux_junctions`),
  `RefluxCapture` / `reflux_captures` (a `Patch` field: the patch's side of
  each junction it takes part in, parent or child, with the per-line
  `stage`, `du` and `reg` registers), `REFLUX` (test/bench toggle),
  `_reflux_open!` / `_reflux_close!` (the hooks in the two divergence
  funnels), `_reflux_fold!`, `_reflux_filter!`, `_reflux_begin_step!`,
  `_reflux_apply!`, `_junction_omega` (the weights of Ω, the conserved
  quadrature's correction beside a junction), `_conserved_fraction` (that
  quadrature's node factor), `GATE_REACH` / `GATE_WIDTH` / `_reflux_density!`
  / `_reflux_gated` (the gate, tested on the window's densities summed over
  the level's ranks, so it reads no halo),
  `RefluxCarry` (what the positivity guard holds back), `REFLUX_DEFER` /
  `_reflux_component!` (the child's Ω rate taken once per component inside
  `compute_rhs!`'s loop), `_reflux_gather!` / `_reflux_guard!` /
  `_reflux_change!` (the apply's per-patch passes, function barriers),
  `_junction_integral` (Ω in `volume_integral`), `_ledger_junction_faces!`
  (the ledger's junction columns from the registers)
- `pointwise!` (the shared launcher of every per-point loop: `Array` storage
  takes `@threaded`, device storage a KernelAbstractions kernel),
  `pointwise_ka!`, `FORCE_KA` (test/bench toggle), and the `_point!` suffix
  for a per-point body. `ghosts` (the `delta4_sum!` keyword declaring that a
  sensed field carries valid interface ghosts: the primitives, the internal
  energy and the mass and mole fractions do, the strain magnitude and the
  dilatation do not) and `SENSOR_INTERFACE_GHOSTS` (the test/bench toggle
  that clamps those taps instead). A body takes plain arrays and scalars, never the
  solver, and never a `Type` argument: a `Type` inside the launcher's
  Vararg defeats specialization and turns the body call into a per-point
  runtime dispatch (measured 9× on `assemble_fluxes!`).
- `FieldVector`/`FieldMatrix` and `field_tuples` (the launchable forms of
  `Y`, `D_art`, `grad_u`, `grad_Y`, `flux`: zero-cost host wrappers that
  adapt to isbits `DeviceFieldVector`/`DeviceFieldMatrix` tuples at device
  launch). Never hand a bare `Vector`/`Matrix` of arrays to a device
  kernel (it hangs in kernel-argument adaptation instead of erroring),
  and never hold the tuple form on the host: runtime tuple indexing cost
  `assemble_fluxes!` 3× on the `@threaded` path when it was tried.
- `DeviceBackend` (an `AbstractBackend` wrapping a KernelAbstractions
  backend, behind `field(backend, decomp)` and `allocate_state`),
  `device_plan`/`DevicePlan` (device mirror of a `DirPlan`/`BandPlan`:
  fill, sweep, spike correction and scatter as KA kernels in a
  (lines × n) layout, one thread per line; the reduced interface stage
  stays host-side through the wrapped plan's `line_solver`, so the device
  method of `apply_along!` is collective, as the host one is), and
  `colwise` (dim 1 mirrors the `solve_col!` banded arithmetic so the
  KA-CPU comparison stays bitwise per dimension), `FORCE_DEVICE_EXCHANGE`
  (test toggle routing host arrays through every device-storage branch,
  which `_device_path` tests; `_cpu_storage` alone routes `pointwise!`)
- `StackedArray` (the stacked storage of a device level's tiles: one array,
  the tiles' padded blocks along the third dimension at a fixed `stride`,
  `ntiles` of them; tiles hold plain views, the level's spanning patch the
  wrapper, and a `pointwise!` routed on it launches every tile), `TileStack`
  / `Level.stacks` (the spanning `patch` and its `members`, one stack per
  padded extent), `_stack_state` (the members' shared state as one stacked
  `ConservedState`), `_stacked_level`, `_build_level_patches` /
  `_build_tile_stack` (the shared tile construction of the constructor, the
  tiled regrid and the restart), `_fine_decomp` / `_fine_plans` /
  `_patch_arrays` / `_assemble_patch`, `_view_workspace`, `_copy_tile!`,
  `_level_rhs!` / `_level_update!` / `_level_filter!` (the per-level phases
  the drivers run per patch or per stack), `padded_extent` (the box a
  full-array launch iterates; never `size` of an array that may be a stack),
  `lines_factor` (`plan_direction`'s line multiplier for a batched plan),
  `DevicePlan.ntiles` / `.stride`, `_tile_ranges`, `_upload!` / `_assign!`
- `LevelScratch` (a fine `Patch`'s device storage of its level transfer:
  the Hermite boxes and the chain `stages` over the rank's components;
  empty on the host backend), `BoxFill`/`HermiteFill` (the stage-0
  source of an imposition, `_fill_stage0!` host and `_fill_stage0_dev!`
  device), `_interpolate_dev!`/`_interp_point!`, `_ring_pack_point!`,
  `_upload_hermite!`, `_tag_rho_point!`
- `refresh_primitives!`, `mixture_density`, `boundary_plane` (the in-flight
  state-query API; primitives are stale inside a callback, see the
  `refresh_primitives!` docstring)
- `padded_index` (interior indices → padded) and `interior_index` (its inverse); a
  padded index goes through the latter before reaching `xcoord`
- `validate_bc` (the setup-time boundary-condition hook), `unit_scalefactor`
- `implicit` (the `Numerics`/`Solver` keyword; an `ImplicitConduction`, or
  `nothing` for the low-storage integrator), `step_rule` (`:error`,
  `:temperature` or `:none`), `WithoutConduction` (the explicit half's
  transport), `ImexIntegrator` (`solver.implicit`: the tableau, the
  `DiffusionStage`, the stage registers, `dt_limit` from the step rule and
  `dt_cap` from failed stages), `_imex_advance!`/`_imex_attempt!`,
  `diffusion_operator!`, `capacity` (the `m` of `(m − γΔt L)` in
  `solve_stage!`)
- `outer_indices` (the flattened outer iteration space of a pointwise nest; see
  Threading), `prepared`/`primitives_current` (the trailing flag by which `run!`
  tells `step!` that the state has been exchanged and its primitives are current)
- `IonmixTable` (an IONMIX4/6 table in SI, not an `EOS`), `read_ionmix` /
  `write_ionmix`, `table_value` / `table_opacity` / `table_state` (the
  bilinear interpolant in `(ln T, ln ρ)` and its own derivatives),
  `table_temperature_status` and the `TABLE_` flags, `monotone` (per density
  node, whether the energies increase with temperature)
- `boundary` (a `LevelTransfer` field: per face, on a domain face whose root
  condition the tile carries), `_level_boundary_condition` (which conditions
  qualify), `_boundary_faces` / `_region_boundaries` / `_level_extent`,
  `_box_buffer` / `_box_shift` / `_box_extent` (the box's per-face buffer,
  none at a boundary face), `_interface_dims`; `folded` (a `LevelTransfer`
  field: the boundary faces on a symmetry plane or the r-z axis),
  `_level_fold_condition` (which conditions fold), `_level_fold_faces`,
  `_fold_lead` (the fine node a tile takes beyond the coincident lattice
  there), `_fine_region` (a tile's region in its own level's node space),
  `_mirror_folded_box!` (the box across the plane), `FoldSpec.div_plans`
  and `FoldRingPlans` (a refined patch's fold, whose far end is an
  interface), `_level_span` (a level's node range, extended at a fold past
  the lattice coincident with the root's, so a region below the first level
  reaching the fold starts at a negative offset), `_region_folds` (the
  folded faces of the regions a regrid or a restart builds),
  `_tile_fine_extent` (a tile's node count with its folded faces)
- `level_boundaries` (keyword, on by default; `RegridSpec.boundaries`: tags
  and shapes may place a level on a domain face it can carry, and across a
  periodic seam), `_placement_faces` / `_lattice_reach` / `_feasible_nodes` / `_placement_extent` (the faces and
  node interval placement reaches), `_placed_reach` (a level's lattice
  reach under the solver's rules), `_unbuffered_faces` (the placed faces
  whose box takes no buffer, every one but a fold), `_warn_margin_band`,
  `_shared_boundary`
  (the domain-face planes a carry or a migration keeps)
- `_level_period` (level ℓ's node-space period along a periodic dimension,
  3^ℓ N, 0 elsewhere; `LevelTransfer.period` is its parent's), `_images` /
  `_shifted` / `_canonical` (a region's periodic images and its stored form,
  offset in [0, P)), `CouplingPiece.image`, `_wrapped_parts` /
  `_wrapped_cells` / `_wrap_feasible` (the lattice over [1, P + 1], its last
  cell ending on the seam), `_placement_period`, `_seam_arc`,
  `_nearest_image`, `_domain_coordinate` (a node past the domain's face read a
  period back, for a user function of position)
- `SesameTable` (one material of a SESAME ASCII 2 library in SI, not an
  `EOS`), its `SesameComponent`s `total` / `ion` / `electron` / `nuclear`
  (the 301 / 303 / 304 / 305 records, each on its own grid) and
  `SesameColdCurve` `cold` (306), `read_sesame` / `write_sesame`,
  `interpolation` (`:bilinear`, or `:free_energy`: e and p from one bicubic
  Hermite interpolant of A), `free_energy_xy` (the node cross derivative),
  `dropped` (nodes at zero removed on reading), `_LogGridTable` (the
  supertype the shared location, bilinear interpolant and inversion take)

**Temperature is `T_ion`.** There is one temperature today; the name keeps
`T_ele` / `T_rad` free for a 2T or 3T model without a second API break. Bare
`T` is the element-type parameter and nothing else.

The following short names are conventions; do not "fix" them: `Q`, `dQ`, `du`, `d`
(dimension), `sp` (species), `I` (CartesianIndex), `σ`/`σf`/`σg` (parity
signs), `ξ` (computational coordinate), `h`, `c`, `p`, and `s` as a band offset
in `banded.jl` / `operators_banded.jl`, where it is the matrix-index convention
the surrounding comments define (`Ab[q+1+s, i]`).

## Traps

**MPI collectives cannot sit below an early return.** `deriv_along!` and
friends are distributed solves along a dimension, so *every* rank must call
them. A boundary routine that returns early on ranks not owning the wall plane
deadlocks. Both `correct_rhs!` methods in `nscbc.jl` hoist their collectives
above the `plane === nothing` return for this reason; follow that shape when
adding a boundary condition. The symptom is zero CPU on every rank, not a
crash, so it presents as a hang.

**A boundary condition that changes mid-run must change on every rank at the
same step.** This is the same collective trap from the other side: a phase
change (`setup(solver, Q; bcs)`) takes the new conditions from the step the
first `run!` ended on. If the new condition runs collectives the old one does
not, as `NSCBCOutflowBC` does, then ranks disagreeing about the stop is a
deadlock, not a wrong answer. `WhenState` therefore reduces its condition
across the communicator; end a run from a `Callback` and never from a
rank-local test. `AtTime` and `EveryStep` are safe without a reduction because
`t` and `step` advance identically everywhere. The MPI suite pins this with a
condition that is true on one rank only: without the reduction the `callback
consistency` phase fails and the `phase change` phase hangs.

**Minimum local extent per dimension.** `plan_direction` errors when a rank's
block is too small for the scheme: C6 needs 5 points, C10 needs 7, and the C8
filter needs 9. The filter is the binding constraint, so a grid that is fine
for derivatives can still fail once filtering is on, and the test suite uses
transverse extents of 12 or 16 to clear it. Rank count multiplies this:
`n_global[d]` must stay ≥ 9 × `dims[d]`.

**Timing noise.** Run-to-run spread is easily 10–20% for an identical
configuration. Order-of-magnitude results are safe; few-percent ones are not
resolved by a single run. `bench/phases.jl` (per-phase budget) is more useful
than a flat sampling profile, which is dominated by the compact line solves and
shows little else.

**Julia soft scope.** A top-level `for` in a script that reassigns a variable
bound outside it throws `UndefVarError`. Wrap script bodies in a function.

**A nondeterministic crash or bitwise failure in the serial suite under Julia
1.12.7 on Windows may not be yours.**
`reference/bugreports/julia_codegen_bug_report.md` diagnoses a concurrent-JIT
race: three `EXCEPTION_ACCESS_VIOLATION` signatures in the compile path, a
crash line that moves between runs, and the `device line solves` testset
failing a bitwise comparison on a few of forty. Read it before bisecting a
change against a failure of that shape.

**No `@Const` in a KernelAbstractions kernel.** On the CPU backend a `@Const`
argument makes KernelAbstractions wrap the body in `@aliasscope`, which Julia
1.13.0 miscompiles for a loop that stores an element the next iteration loads
(the Thomas sweep) once bounds checks are elided; 1.11 and 1.12 have a related
form (KernelAbstractions.jl#652). `--check-bounds=yes` hides it, so the serial
CI job passes while the MPI leg fails. Filed as JuliaLang/julia#63129;
`reference/bugreports/julia_aliasscope_bug_report.md` has the reproducer.

**No short-circuit condition choosing between two uniform values in a device
body.** `x = (a || b) ? p : q`, with `p` and `q` kernel arguments and `b`
varying per thread, can give every thread `q`: the LLVM 22.1.8 `llc` that
GPUCompiler runs for every AMD target, gfx942 included, drops the phi while
structurizing the control flow, under any Julia version. The
KernelAbstractions CPU backend is correct, so no CI job sees it. Write
`ifelse(a | b, p, q)`, as `_delta4_signed_point!` does, and trust
`bench/device_solver.jl` on a GPU, not the KA-CPU suite, for a body of that
shape; `reference/bugreports/amdgpu_shortcircuit_bug_report.md` has the
reproducers.

**A run that fails does not stop.** Losing positivity drives the diffusive rate
in `compute_dt` up until `dt` collapses, and the run then grinds forever at no
progress. A sweep that visits bad configurations must pass a low `nmax`, and
`run!` applies `StepControl` floors so this raises `SolverFailure` instead.
`primitives!` substitutes benign placeholders wherever ρ ≤ 0, so the positivity
check in `max_rate` reads ρ out of `Q` directly and not from `solver.rho`.

## Threading

`@threaded <work> for ... end` (`src/threading.jl`) uses `Threads.@threads`
only when `work >= THREAD_MIN_WORK` (1024 points per thread × the session's
thread count; override the total via `CL_THREAD_MIN_WORK`) *and* the loop has
more than one trip; it runs serially otherwise. Allocation is
per region per thread, not per point, so without the threshold small cases pay
spawn cost for nothing.

Total work is not the whole test, because only the loop `@threaded` wraps is
divided. A nest threaded on its outermost index runs serially whenever the third
dimension is collapsed, which covers every pointwise loop of a planar
`(nx, ny, 1)` run. **A pointwise nest therefore iterates
`outer_indices(n2, n3)`**, a single flattened `CartesianIndices` over its two
outer indices unpacked with `j, k = Tuple(jk)`, which divides over the second
dimension instead. Write a new one that way. Iteration order is unchanged, so
this is not a numerics change.

The backend choice is tasks, not Polyester, and it rests on measurement. The
numbers and the reasoning are in the `@threaded` docstring; read it before reaching for
`@batch`, including the note on the conditions under which the decision should
be revisited.

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
  emphatic fragments, editorializing, and em-dash asides. Write clear academic
  prose that explains the package without managing the reader's emotional state.
  Most prose in this repository is model output. When an edit exposes poor style
  in the surrounding prose, follow these guidelines and flag it for correction.
- `bench/` is scratch tooling, not tests. Each script's header comment says
  what it measures and how to run it; read that before running one. `probes/`
  holds the environment diagnostics (MPI configuration, depot cost, launch
  sizing, device bring-up and floors, thread spawn floor, step profiling on a
  slow machine); they measure the machine, not the solver, and run from the
  checkout root with `--project=.`.
- Scripts take their settings from `ARGS`, not the environment: positional
  values first, then `key=value`, parsed by `script_args` (`src/scriptargs.jl`)
  against a defaults `NamedTuple` that doubles as the schema. Give a new script
  the same shape. An unknown key is an error: the environment
  form it replaced answered a typo by running the default for the length of the
  job and saying nothing. The two remaining `CL_*` variables
  (`CL_THREAD_MIN_WORK`, `CL_ERROR_BACKTRACE`) are read from inside the package,
  where there is no `ARGS` to consult.
- Wrap an MPI driver in `mpi_main` and report progress with `ProgressLog`;
  do not hand-roll either. An uncaught exception is 448 stacktraces at
  448 ranks; a hand-rolled progress loop got at least one of printing-from-one-
  rank, flushing, and not-timing-its-own-reduction wrong every time. Both
  docstrings carry the reasoning. Ctrl-C has no code fix; pass
  `--handle-signals=no` for sweeps.
- A sweep over configurations expected to include bad ones must pass a low
  `nmax`. A configuration that loses positivity does not crash: the diffusive
  rate in `compute_dt` climbs until `dt` collapses and the run grinds, so one
  bad point costs more wall time than the whole sweep. `test/cases.jl` threads
  an `nmax` through every case for this.
- The 1-D cases run single-threaded: each threaded region is well
  under `THREAD_MIN_WORK`, so `-t auto` buys nothing and ~4% CPU on a 24-thread
  box is the expected reading, not a bottleneck.
- Run artifacts (`*.dat`, `*.vtr`, `*.pvtr`, `*.ckpt`, `*.cov`) are gitignored.
  Prefer `git add <paths>` over `git add -A` after running examples.
- **Commit to `main`.** This is a single-maintainer repository and the default
  agent habit of opening a branch per change is noise here; do not create one
  unless asked. Commit only when asked, and first complete the applicable checks
  under "The gate."
- **A roadmap closure rides the next commit; it does not get one of its own.**
  After a commit lands, fill its hash into the `ROADMAP.md` line it closes, but
  leave that edit uncommitted rather than pushing a standalone "close N_" commit.
  Fold it into whichever substantial commit comes next, silently: that commit's
  message describes only its own work, not the bookkeeping it carries along.

## Known limitations

Each of these took time to establish. The measurements and the
rejected hypotheses are in the file named; read it before re-deriving any of
them.

- **A strong shock at the spherical origin has a lower CFL ceiling than at
  the planar wall or the cylindrical axis**, which complete Noh from 0.9
  since `run!` primes the artificial coefficients before the first step. The
  warm-started Noh is limited to 0.5 by an excursion of the origin cell near
  t = 0.39, the singular t = 0 start to 0.15 and a top-hat blast to 0.2–0.3;
  `StepControl(retries = 4)` recovers each. The discretization-order
  explanations, the density proportionality of β\* and the per-step filter
  strength have been ruled out; the area form of the radial pressure term set
  a lower ceiling and a resolution requirement on the initial data, and the
  blast fails in either form. → `reference/CALIBRATION_APPENDIX.md`
- **κ\* is written as `ρc/T_ion`** and is not singular in practice: the sound
  speed vanishes with the temperature at a floored cell, and on Noh the κ\*
  rate is an order below the β\* rate at every ambient pressure from 1e-2 to
  1e-8. `artificial_conductivity_scale` remains an EOS dispatch point.
- **The compact filter, not the Cook artificial properties, holds this solver
  together, and its fit is joint with `C_mu`.** At 128³ TGV the filter
  supplies 37% of the energy sink yet removing it kills the run, while removing
  the artificial properties entirely does not. The filter was fitted at
  αf = 0.45 and re-swept at 0.49, both at `C_mu = 0.002` (the default is
  0.47), and Taylor-Green selects no positive `C_mu`, so a change to either
  reopens the other (roadmap N4). → `reference/CALIBRATION_APPENDIX.md`
- **`compute_artificial!` is 24.8–26.0% of the multicomponent RHS** under the
  default `:gaussian` smoother (the 31.8% figure is the `:compact` one),
  in the filter line-solves that smooth the sensors, one sweep per species. At
  `n_species == 2` that machinery is measurably a no-op (`D*_1` and `D*_2` agree
  to round-off), and it only earns its cost at three or more species. Cutting
  it is a numerics decision (shared vs per-species sensor), not a code tweak.
- The NASA CEA thermo and limited transport databases are bundled verbatim in `data/`
  with their Apache license and notice. `read_nasa9` handles multi-interval
  thermo records; `CeaTransport` connects the transport fits with a unity-Lewis
  fallback, or mixture-averaged diffusion from supplied binary data.
- Cluster-side open questions (whether `ThreadPinning`'s pinning API would buy
  anything, and the unexplained ~4300x SMT-sibling collapse) are in
  `reference/CLUSTER.md`.
- **The relaxed filter reads a directional hyperbolic rate, not the rate
  that sized the step.** `filter_weight` scales a pass along `d` by
  `dt · (|u_d| + c)/h_d · √n / filter_cfl`, so its dissipation per unit time
  is invariant to the step, to the diffusive rates and to the spacing of the
  other directions; reading the step's own maximum completed an
  aspect-ratio-16 Noh run with a wrong solution. A directional β\* was
  measured and rejected for making vorticity in cold gas on a curved front.
  → `reference/CALIBRATION_APPENDIX.md`
- A timing difference under the 10–20% run-to-run spread is resolved with
  `bench/repeat.jl`, which takes medians over repeated *processes* and, given
  two commands, alternates them and reports the paired ratio.
