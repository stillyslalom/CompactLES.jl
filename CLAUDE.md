# Working in CompactLES

Orientation for coding agents, kept short: it carries only what you
need before touching anything. **It does not describe the solver**, and it is not
the place to record a result.

| file | what is in it |
|---|---|
| `README.md` | usage |
| `reference/DESIGN.md` | the numerics — source map, compact solve, folds, freestream preservation, NSCBC |
| `reference/CLUSTER.md` | MPI configuration, launch rules, sizing, measured scaling |
| `reference/CALIBRATION.md` | the calibrated defaults and which setting to change for an instability or an over-dissipated solution |
| `reference/CALIBRATION_APPENDIX.md` | the index of every measurement, one section per instrument in the topic files of `reference/appendix/` |
| `reference/ROADMAP.md` | positioning, comparisons, open work items; a closed item is deleted |
| `reference/AMR_GPU.md` | the patch-AMR + GPU design as delivered, measured lessons, roadmap |
| `reference/TRANSPORT.md` | molecular transport and the species channel |
| `reference/IMMERSED.md` | the immersed-boundary design: level-set bodies, blend imposition |
| `reference/MODE_TRUNCATION.md` | azimuthal mode truncation plan for the pole CFL squeeze |
| `reference/IMPLICIT.md` | the implicit-diffusion and IMEX design: staggered operator, Krylov solve, ARK pair |
| `reference/INCOMPRESSIBLE.md` | the incompressible solver (`CompactLES.Incompressible`) |
| `docs/` | the Documenter site. `docs/src/tutorials/` and `docs/build/` are generated from `docs/literate/` and gitignored: edit `literate/` |

This file is about *how to work on the code*; the files above say what it does,
where it runs and where it is going. Each kind of information has one place. A
measured digit goes in the appendix, once, in the section of the instrument that
produced it, and a newer measurement replaces the older one. The git log is the
history: a closed roadmap item is deleted by the commit that closes it. Every
other file gets at most three sentences and a link. Reference prose leaves the
figures to tables: a sentence outside a table or fenced block carries at most two
numerals, citations aside. Results of a session are reported in the reply, not
recorded in the repository. `test/reference_tests.jl` enforces the layout.

## Environment

- Compat bounds are in `Project.toml`. MPI.jl's API changed on either side of
  0.20; check which side a call is written for.
- `mpiexec` is usually not on PATH. Query the launcher this checkout is
  configured against rather than hardcoding one:
  `julia --project=. -e 'using MPI; MPI.mpiexec(c -> println(c))'`.
- Identify the hardware before reading a timing: `Threads.nthreads()`, physical
  cores, and whether they are uniform. A hybrid desktop spreads threads over
  performance and efficiency cores, and nothing pins threads or ranks
  (`ThreadPinning` is not a dependency; the Windows JLL `mpiexec` sets no
  affinity). Form timing ratios within a process, as `bench/derivcost.jl` does.
- Windows checkouts may have CRLF line endings; a helper script matching
  multi-line text against `\n` finds nothing. Match a line at a time.
- **On a cluster, read `reference/CLUSTER.md` first, even when the code runs.**
  The bundled JLL MPI never reaches the interconnect off one node and runs
  27x–66x slower with no other symptom. `LocalPreferences.toml` is per project,
  so `--project` selects the MPI implementation.
- `Manifest.toml` is gitignored: a fresh checkout needs `Pkg.instantiate()`.
  Most precompile time is the workload in `src/precompile.jl` (skipped under a
  system MPI): a primary part for common configurations and a secondary part of
  similar cost for rarer ones. Keep that balance; add no block for a test.
- The test suites' configurations precompile in the helper package
  `test/TestPrecompile`. Run `julia --project=. test/test_image.jl` once, in one
  process, before a serial or MPI suite (`mpi_suite=true` before the MPI one, as
  CI does); its header has the options. Julia 1.11+ keys the cache on content,
  so `touch` does not rebuild it; force one with
  `Base.compilecache(Base.identify_package("CompactLES"))`.
- `bench/tgv_energy.jl` is the first production workload on a cluster: its
  reductions are all collective and reproduce serial numbers to round-off.

## The gate

Before committing, run the checks the change requires; take the union for a
mixed change and the full gate when the impact is unclear. Report what ran,
any baseline change, and any required coverage that was unavailable.

| change | checks |
|---|---|
| prose, comments, docstrings | `test/docrefs_tests.jl`; `test/reference_tests.jl` if `reference/`, `README.md` or this file changed; a docs build for Documenter configuration or executable examples |
| benchmarks, examples, tooling | `test/scriptrefs_tests.jl`, then the affected scripts on a small workload |
| tests only | the affected suite and its runner; a changed guard or stored reference also runs its numerical suite, with the new baseline explained |
| solver behavior or shared infrastructure | the core gate (operators, boundaries, thermodynamics, transport, filtering, timestepping, storage, AMR, precompilation) |
| what `Solver` or `setup` accepts | also `test/capability_matrix.jl` (about 80 s) |
| distributed algorithms | the core gate and the full MPI suite at 2, 4 and 8 ranks |
| hot-path arithmetic, dispatch, types, storage, threading | also `bench/jetcheck.jl` and `bench/audit.jl`, before and after in one environment |
| extensions and devices | `test/hdf5_tests.jl` in an environment with HDF5, `test/makie_tests.jl` from the docs environment, serially and under MPI where output is collective; device paths on a GPU. A skipped check is unavailable coverage, not a pass |

```bash
MPIEXEC=$(julia --project=. -e 'using MPI; MPI.mpiexec(c -> print(c))')
julia --project=. -O1 test/runtests.jl   # compile-bound; see the note there
julia --project=. test/convergence.jl
julia --project=. test/validation.jl
"$MPIEXEC" -n 2 julia --project=. -t 1 test/mpi_tests.jl
"$MPIEXEC" -n 8 julia --project=. -t 1 test/mpi_tests.jl rank_shape=true
```

The full gate replaces the two MPI lines with the full suite at 2, 4 and 8
ranks and adds both performance audits and `test/capability_matrix.jl`.

- **Run `test/docrefs_tests.jl` before every push** unless it passed for the
  current docstrings and `docs/`. The documentation job is the slowest CI leg
  and fails on a `[`Name`](@ref)` no `@docs` block renders. A private name may
  be linked only if a `@docs` block lists it; otherwise use plain backticks.
  A passing `runtests.jl` includes this check and the script-reference check.
- The serial suite (`test/serial_suite.jl`, its parts under `test/serial/` and
  the subsystem files it includes) runs inside one top-level `@testset` and
  records every failure. `mpi_tests.jl` (phases under `test/mpi/`) counts
  `check` assertions and exits nonzero on a failure. The two totals are not
  comparable; compare each against a run of the base commit.
- `test/hdf5_tests.jl` and `test/makie_tests.jl` are skipped by the gate:
  HDF5 and CairoMakie are weak dependencies. `require=hdf5,makie` makes the
  skip a failure. `Pkg.test` never runs the Makie suite; CI's documentation
  job does. `hdf5_parallel()` reports which write backend libhdf5 selects.
- `test/convergence.jl` prints orders against the guards in its header. **A
  change not meant to touch numerics leaves it bit-identical, error magnitudes
  included;** chase a moved digit. Update a `recorded` value only together with
  that table. Smooth cases live in `test/smooth_cases.jl`.
- `test/validation.jl` is not bit-reproducible: compare at four significant
  digits, where a moved third digit is real. Cases live in `test/cases.jl`,
  shared with `bench/artcal.jl`.
- `bench/jetcheck.jl` and `bench/audit.jl` print counts: compare the delta,
  like with like, and never sum entry points. The jetcheck baseline is one site
  in `apply_bcs!` (`enforce!`), six in `compute_rhs!` (three boundary hooks,
  two `_cold` barriers, `_launch_fluxes!` on `Val` of the species count), the
  same seven in `step!`, and one in each `compute_dt` probe. Any other report is
  a regression; the script's header explains what it drops.
- `audit.jl` reads `code_typed` at a spelled-out signature. A keyword or an
  omitted optional trailing argument on a probed entry point resolves it to the
  forwarding method and the count falls to 1, measuring nothing. Extend the
  probe tuple in the same commit that adds one.
- `bench/coverage.jl`'s header has the coverage sequence; `.cov` output omits
  uncompiled methods, so watch the executable-line count too.
- The suites never import `bench/` or `examples/`. `test/scriptrefs_tests.jl`
  parses every script outside `bench/archive/` for names that no longer exist;
  a script that resolves can still compute the wrong thing, so run the affected
  consumers too.

## Naming

Names are spelled out in full: `solver`, `decomp`, `n_global`, `n_halo`,
`n_species`, `mu_art`, `filter_interval`. Before naming something, find the
neighboring names in the source and follow them; a name's meaning is in its
docstring or the comment where it is defined. **Temperature is `T_ion`**, which
keeps `T_ele` and `T_rad` free for a two- or three-temperature model; bare `T`
is the element-type parameter. A per-point body carries the `_point!` suffix,
and a padded index goes through `interior_index` before reaching `xcoord`.

These short names are conventions; do not "fix" them: `Q`, `dQ`, `du`, `d`
(dimension), `sp` (species), `I` (CartesianIndex), `σ`/`σf`/`σg` (parity signs),
`ξ` (computational coordinate), `h`, `c`, `p`, and `s` as a band offset in
`banded.jl` and `operators_banded.jl` (`Ab[q+1+s, i]`).

## Traps

- **No MPI collective below an early return.** `deriv_along!` and its kin are
  distributed solves that every rank must call. Both `correct_rhs!` methods in
  `nscbc.jl` hoist their collectives above the `plane === nothing` return;
  follow that shape. A deadlock shows as zero CPU on every rank.
- **A boundary condition that changes mid-run changes on every rank at one
  step.** A phase change takes the new conditions from the step the first
  `run!` ended on, and ranks that disagree deadlock. `WhenState` reduces its
  condition; end a run from a `Callback`, never a rank-local test.
- **Minimum local extent.** `plan_direction` needs 5 points per rank for C6, 7
  for C10 and 9 for the C8 filter, which binds: `n_global[d] ≥ 9 × dims[d]`.
- **A failed run grinds.** Lost positivity drives `dt` toward zero rather than
  crashing; sweeps over bad configurations pass a low `nmax`, and `run!`'s
  `StepControl` floors raise `SolverFailure`. `primitives!` substitutes benign
  values where ρ ≤ 0, so `max_rate` reads ρ from `Q`.
- **Timing noise** is 10–20% run to run. Resolve small differences with
  `bench/repeat.jl`; `bench/phases.jl` is more useful than a flat profile.
- **Julia soft scope.** A top-level `for` reassigning an outer variable throws
  `UndefVarError`; wrap script bodies in a function.
- **A per-point body takes plain arrays and scalars**, never the solver and
  never a `Type` argument: a `Type` in the launcher's Vararg becomes a per-point
  dispatch (9x on `assemble_fluxes!`). Hand a device kernel the `FieldVector`
  and `FieldMatrix` forms, never a bare `Vector` of arrays (it hangs in argument
  adaptation), and never hold their tuple form on the host.
- **A nondeterministic crash under Julia 1.12.7 on Windows may not be yours**:
  `reference/bugreports/julia_codegen_bug_report.md` diagnoses a JIT race.
- **No `@Const` in a KernelAbstractions kernel.** Julia 1.13.0 miscompiles the
  `@aliasscope` it implies for a recurrence once bounds checks are elided, and
  `--check-bounds=yes` hides it
  ([JuliaLang/julia#63129](https://github.com/JuliaLang/julia/issues/63129);
  KernelAbstractions.jl#652).
- **No short-circuit choice between two uniform values in a device body.**
  AMDGPU's `llc` drops the phi; write `ifelse(a | b, p, q)`. The KA-CPU suite
  cannot see it (`reference/bugreports/amdgpu_shortcircuit_bug_report.md`).
- **No exception path in a device body.** AMDGPU under Julia 1.12 and 1.13
  rejects a kernel that keeps a `throw` (`InvalidIRError ... kernel_state()`):
  use `_unchecked_div`/`_unchecked_rem`, keep accesses in `@inbounds`, narrow as
  `x % UInt8` (`reference/bugreports/amdgpu_exception_path_bug_report.md`).

## Threading

`@threaded <work> for ... end` (`src/core/threading.jl`) threads only above
`THREAD_MIN_WORK` (1024 points per thread, `CL_THREAD_MIN_WORK` overrides) and
divides only the loop it wraps. **A pointwise nest therefore iterates
`outer_indices(n2, n3)`**, unpacked as `j, k = Tuple(jk)`, so a planar run
still divides over its second dimension. The backend is tasks, not Polyester,
on measurement; read the `@threaded` docstring before reaching for `@batch`.
The 1-D cases run single-threaded by design.

## Conventions

- Keep source lines under ~95 characters.
- Comments explain *why*, and several encode a measurement or a derivation.
  Move them with the code; do not delete the reasoning.
- Prose: write clear academic prose. Avoid agency for inanimate things,
  setup-and-payoff, "what matters is" and its relatives, emphatic fragments,
  editorializing and em-dash asides. When an edit exposes poor prose nearby,
  follow these rules and flag it.
- `bench/` is scratch tooling; each script's header says what it measures and
  how to run it. `bench/archive/` keeps the scripts behind recorded
  measurements, unchecked and possibly stale. `probes/` measures the machine.
- Scripts take settings from `ARGS` through `script_args`, positional values
  then `key=value`, against a defaults `NamedTuple`; an unknown key is an error.
- Wrap an MPI driver in `mpi_main` and report progress with `ProgressLog`;
  their docstrings say why. Pass `--handle-signals=no` for sweeps.
- Run artifacts (`*.dat`, `*.vtr`, `*.pvtr`, `*.ckpt`, `*.cov`) are gitignored;
  prefer `git add <paths>` after running examples.
- **Commit to `main`.** This is a single-maintainer repository: create no branch
  unless asked, commit only when asked, and first complete the gate.
- A roadmap item is deleted by the commit that closes it; that commit's message
  is the record.

## Known limitations

Each took time to establish; read the cited file before re-deriving one.

- A strong shock at the spherical origin has a lower CFL ceiling than at a
  planar wall or the cylindrical axis; `StepControl(retries = 4)` recovers each
  case measured (`reference/appendix/shocks.md`).
- The compact filter, not the artificial properties, holds the solver
  together, and its fit is joint with `C_mu`: a change to either reopens the
  other (roadmap N4; `reference/appendix/filter.md`).
- The relaxed filter reads a directional hyperbolic rate, not the rate that
  sized the step, on measurement (`reference/appendix/filter.md`).
- κ\* is written as `ρc/T_ion` and is not singular in practice;
  `artificial_conductivity_scale` remains an EOS dispatch point.
- `compute_artificial!` is about a quarter of the multicomponent right-hand
  side, in the per-species sensor smoothing, a no-op at two species. Cutting it
  is a numerics decision, not a code tweak (`reference/appendix/cost.md`).
- The NASA CEA thermo and transport databases are bundled verbatim in `data/`
  with their license.
- Open cluster questions, thread pinning among them, are in
  `reference/CLUSTER.md`.
