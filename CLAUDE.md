# Working in CompactLES

Orientation for coding agents. **This file does not describe the solver** —
`README.md` covers usage and `DESIGN.md` covers the numerics and mechanics
(source map, distributed compact solve, folds, GCL, NSCBC). Read those for
anything about *what* the code does. This file is about *how to work on it*.

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
is a dependency for the cluster case; it is not wired up yet and can only be
validated on the target.

Windows checkouts may have CRLF line endings. Helper scripts that match
multi-line text against `\n` will silently find nothing; match a line at a time.

## The gate

Run all of it before calling a change safe.

```bash
MPIEXEC=$(julia --project=. -e 'using MPI; MPI.mpiexec(c -> print(c))')

julia --project=. test/runtests.jl        # 31 testsets, 0 failures
julia --project=. test/convergence.jl     # measured orders, see below
julia --project=. test/validation.jl      # shock-capturing battery, ~25 s
for np in 2 4 8; do
  "$MPIEXEC" -n $np julia --project=. test/mpi_tests.jl   # 26/26 each
done
julia --project=. bench/jetcheck.jl       # inference
julia --project=. bench/audit.jl          # allocation + non-concrete SSA
```

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
them before a change and compare after; what matters is the delta. Most
remaining dispatch sites are `Metric` / `EOS` / `BoundaryCondition` field reads
behind function barriers — one dispatch per array pass, not per point — so
treat a *new* site as the signal, not the absolute number. The counts overlap
between entry points (`step!` contains `compute_rhs!`), so compare like with
like and never sum them.

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
- `mu_art`, `beta_art`, `kappa_art`, `D_art`, `C_mu`/`C_beta`/`C_kappa`/`C_D`
- `grad_u`, `grad_T_ion`, `grad_Y`, `strain_mag`, `sensor`, `sensor_sp`
- `inv_J`, `area_d`, `inv_h`, `inv_r`, `cot_over_r`, `coord_shift`, `flux`
- `deriv_plans`, `filter_plans`, `line_solver`, `plan` (a DirPlan) vs `plane`
  (a wall plane), `fold`, `pair`
- `plane_profile`, `profile_spacing`, `mix_width`, `molecular_mixing`,
  `quad_weight`, `cell_measure`
- `eos_phi`, `eos_dphi_dY`, `art_conductivity_scale`, `species_energy`,
  `mixture_temperature` (the EOS contract; the whole list is at the top of
  `physics.jl`)
- `control` (a `StepControl`), `max_rate`, `predicted_dt`, `check_step`,
  `dt_prev`, `rate_prev`, `savepoint`

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
and runs serially otherwise. Allocation is per-region-per-thread rather than
per-point, so without the threshold small cases pay spawn cost for nothing.

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
- `bench/` is scratch tooling, not tests: `audit.jl` (allocation, inference),
  `jetcheck.jl` / `jetwhere.jl` (dispatch sites, and where they come from),
  `phases.jl` (RHS phase budget), `profile.jl`, `scaling.jl`, `threadscale.jl`,
  `bcbench.jl`, `coverage.jl`, `artcal.jl` (artificial-property sweeps),
  `amr_transfer.jl`.
- A sweep over configurations expected to include bad ones must pass a low
  `nmax`. A configuration that loses positivity does not crash — the diffusive
  rate in `compute_dt` climbs until `dt` collapses and the run grinds — so one
  bad point costs more wall time than the whole sweep. `test/cases.jl` threads
  an `nmax` through every case for this.
- The 1-D cases run single-threaded on purpose: each threaded region is well
  under `THREAD_MIN_WORK`, so `-t auto` buys nothing and ~4% CPU on a 24-thread
  box is the expected reading, not a bottleneck.
- Run artifacts (`*.dat`, `*.vtr`, `*.pvtr`, `*.ckpt`, `*.cov`) are gitignored.
  Prefer `git add <paths>` over `git add -A` after running examples.

## Open items

- **Strong shocks need `cfl ≤ 0.15`**, and the cause is a dispersive undershoot
  at the shock that the artificial viscosity does not damp — the state loses
  ~60% of its pre-shock density over 150 steps while `dt` and the CFL rate sit
  flat, then positivity goes. It is NOT the one-step lag in `compute_dt`; that
  hypothesis was tested with a rate predictor at up to 30 steps of lookahead and
  rejected, and the predictor is shipped off by default with the measurement in
  `src/stepcontrol.jl`. Fixing it properly is a spatial-regularization question.
  `StepControl(retries = 4)` is the practical mitigation and is strictly better
  than a lowered CFL where it applies, because the restriction is a startup one.
- **κ\* is singular as `T_ion` → 0.** `art_conductivity_scale` is now an EOS
  dispatch point rather than inline ideal-gas algebra, so a condensed-matter or
  tabular model can supply its own; the ideal-gas method still divides by
  `T_ion` and a cold ambient below p ≈ 1e-3 collapses the diffusive timestep.
  Fixing it for gases is a numerics decision, not a refactor.
- **The spherical origin fold is much less forgiving than the cylindrical axis**
  — it needs initial data resolved over ≳3 cells and will not take Noh's
  singular t = 0 start, both of which the cylindrical axis handles. Nobody has
  worked out why.
- `C_mu` is uncalibrated for its actual purpose. Every case in
  `test/validation.jl` is 1-D, where the shear artificial viscosity is inert; it
  needs a 3-D run with resolved shear (the Taylor–Green harness behind
  `CL_RUN_TG=1` is the obvious vehicle).
- `Nasa9Mixture` has no coefficient database, deliberately — see the note in
  `physics.jl`. Shipping one is a data question and would let
  `examples/shock_tube.jl` use a real CO₂ cp(T).
- `compute_artificial!` is ~31% of the multicomponent RHS, spent in the filter
  line-solves that smooth the sensors, one sweep per species. Cutting it is a
  numerics decision (shared vs per-species sensor), not a code tweak. Note that
  at `n_species == 2` the per-species machinery is measurably a no-op: on a
  sharp binary interface `D*_1` and `D*_2` agree to 4.8e-16 relative and the
  correction velocity is 2.5e-18 against a species gradient of 84. It only
  earns its cost at three or more species.
- `ThreadPinning` is a dependency but unused; only validatable on a cluster.
- A `bench/` runner taking medians over repeated *processes*, to get under the
  timing noise above.
- `FoldSpec` parameterization would close several remaining JET dispatch sites.
