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
supplies the topology queries `clusterprobe.jl` uses; see **Cluster launches**
below. Nothing pins threads yet — the querying half is what has been validated.

Windows checkouts may have CRLF line endings. Helper scripts that match
multi-line text against `\n` will silently find nothing; match a line at a time.

**On a cluster, MPI configuration is the first hurdle and it is not a code
change.** MPI.jl defaults to its bundled JLL. That JLL can satisfy the
scheduler's launcher perfectly well *on a single node*, over shared memory —
which is what makes this so easy to miss. It forms a correct communicator,
reproduces the physics bit-for-bit, and only falls apart once a run spans two
nodes, because it never reaches the interconnect. Measured on rzhound, 256³ TGV
at 224 ranks over two nodes: **15.19 s/step on the JLL, 0.571 s/step on the
system MVAPICH2 — 27x, same launch line, same decomposition.** On-node the two
are indistinguishable (1.434 vs 1.372 s/step at 56 ranks, inside run-to-run
spread), so nothing you measure on one node will warn you.

Point it at the system MPI once per checkout. The module has to be loaded first
so `libmpi` is on `LD_LIBRARY_PATH`, and naming the launcher matters — without
`mpiexec=`, `MPI.mpiexec` keeps handing back the JLL's binary and the helper at
the top of this file quietly lies to you:

```bash
module load <site mpi>          # module avail mvapich2 / mpich / openmpi
mpicc -show                     # <-- the authoritative -L path for THIS module
julia --project=. -e 'using Pkg; Pkg.add("MPIPreferences")'
julia --project=. -e 'using MPIPreferences; MPIPreferences.use_system_binary( \
    library_names=["<that path>/libmpi.so"], mpiexec="srun")'
```

The bare call fails and an absolute path is what works. A site module sets `PATH`
but often *not* `LD_LIBRARY_PATH`, because compiled binaries find their libraries
by RPATH; `dlopen("libmpi")` has no RPATH to lean on and reports "MPI library
could not be found" for every candidate name even though the module loaded
cleanly. **Do not reach for `$MPI_ROOT`** — it is unset on LC machines. `mpicc
-show` names the exact `-L` directory of the build the module resolved to, which
is the one whose Slurm integration the site tests; a bare
`find /usr/tce/packages -name 'libmpi.so*'` turns up a dozen per-compiler builds
with no indication which is live. `libmpi` is a C library, so the compiler in
that path does not affect the ABI Julia needs — but if `dlopen` then complains
about a missing Intel runtime (`libimf`, `libsvml`), switch to a gcc build of the
same version. Prefer an MPICH-ABI implementation (MVAPICH2, MPICH) over OpenMPI
where both are offered.

On a csh login shell: `module` exists only in csh, so the module load and the
Julia call have to run there, while `2>/dev/null` does not parse in csh ("Ambiguous
output redirect") — use `>&` or run diagnostics in a bash subshell, remembering
that bash inherits `PATH` but not the `module` function.

Then, if `MPI.Init` hangs or every rank reports itself as rank 0 of a 1-rank
world, the launcher needs a bootstrap it isn't defaulting to: `srun --mpi=list`,
then `--mpi=pmi2` for MVAPICH2 or `--mpi=pmix` for OpenMPI. The rank-0-of-one
symptom is the dangerous one — it produces plausible output at 1/N the speed
instead of an error.

Then **restart Julia** — the preference is read at precompile — and verify with
`MPI.versioninfo()`, or run `clusterprobe.jl`, whose `MPI binary` line reports
`system` versus a `_jll` outright and flags the JLL on any multi-node run. Do
this inside an allocation if the login and compute nodes carry different stacks.

The preference lands in `LocalPreferences.toml`, which is gitignored: it names
one machine's library by path and must not travel between checkouts.

**That file is per-project, so `--project` silently decides which MPI you get.**
A second environment that `dev`s this package — say a driver project alongside
`~/.julia/dev/CompactLES` — does not inherit the preference, and running the same
script against the wrong one falls back to the JLL with no symptom other than
speed. Measured both ways at 256³: 224 ranks over 2 nodes, 15.19 s/step on the
JLL against 0.571; 448 ranks over 4 nodes, 19.6 against 0.2966 — **27x and 66x**,
the penalty growing with off-node share (25% then 50% of dimension 1's links).
Configure every environment you launch from, and run `clusterprobe.jl` with the
same `--project` as the solver, or its `MPI binary` line describes a different
environment than the one you are timing.

With a driver environment separate from the checkout, run from the driver and
locate the scripts through `pkgdir`. `--project=.` then always picks up the
configured environment, and a pull into the checkout needs no copying:

```bash
srun -n 448 --cpu-bind=threads julia --project=. -t 1 \
    -e 'using CompactLES; include(joinpath(pkgdir(CompactLES), "examples", "taylor_green.jl"))'
```

Resist promoting any of this to a login file. An invisible default that redirects
every later invocation is the `JULIA_NUM_THREADS` trap wearing a different hat.

`Manifest.toml` is gitignored, so a fresh checkout needs `Pkg.instantiate()`
before anything runs. `bench/tgv_energy.jl` takes its grid, end time, and
configuration list from `ARGS`/`ENV` precisely so a batch script can drive it
without edits; it is the intended first real workload, since it is the one bench
script whose reductions are all collective and which reproduces serial numbers
bit-for-bit under decomposition.

## Cluster launches

**Measure the launch before you believe a timing from it.** `clusterprobe.jl`
takes seconds, runs no solver steps, and reports what the ranks actually got:
node distribution, the MPI library in use, each rank's CPU affinity mask as
logical *and* physical CPUs, whether any CPU is double-booked, the cgroup memory
limit against what `Sys.total_memory()` claims, `MPI.Init` and package-load time,
the resulting `n_local`, whether `@threaded` engages at all, and the share of
Cartesian neighbour links that stay intra-NUMA / cross-socket / off-node.

```bash
srun -n 56 --cpu-bind=threads julia --project=. clusterprobe.jl
```

Run it once per new machine, and again whenever a launch flag changes. Every
rule below came out of it, and each one was a *failed* hypothesis first.

**`clusterlaunch.jl` sizes the run before you queue it.** It launches nothing and
needs no allocation — given a grid and a core budget it enumerates the legal rank
counts, taking the minimum-extent floor from the scheme objects themselves rather
than a remembered 9, and reports points/rank, whether `@threaded` engages, and
the padded-to-interior ratio (the redundant halo arithmetic, which is what
actually limits how far this solver decomposes). It names two picks, because they
answer different questions: the widest count whose halo cost is still in hand,
and the widest legal one for when wall time is the constraint.

```bash
CL_NODES=36 CL_CORES_PER_NODE=112 julia --project=. clusterlaunch.jl 256
```

The gap between those two is the real sizing signal. At 64³ on one 112-core node
it is 8 ranks against 112 (1.95x vs 4.25x halo) — that grid is simply too small
for the node, which is the arithmetic reason `reference/CALIBRATION.md` wants a
larger one, not just a physics preference. At 256³ on 36 nodes it is 448 against
4032, and the whole allocation becomes defensible.

**Scheduler flags do not mean what they say.** Do not assume `--cpus-per-task=N`
yields N CPUs. Where the scheduler allocates at core granularity with SMT on,
`-c 1` yields both threads of one core (2 logical CPUs) and `-c 2` yields one
thread (1 logical CPU) — measured, inverted, reproducible across nodes. The mask
is the ground truth; the flag is a request.

**Never hand a rank both SMT threads of one physical core.** On rzhound (2
sockets, 8 NUMA domains, 112 cores, SMT2) that configuration ran the 64³ TGV at
154 s/step against 0.036 s/step for the same 56 physical cores bound one thread
per rank — a factor of ~4300, reproduced four times. Memory, NUMA placement,
package-load time, `MPI.Init` time, thread count and rank count were all measured
identical across the fast and slow cases; the mask is the only difference and no
mechanism has been established. `--cpu-bind=threads` avoids it. Treat
`SMT sibling in a rank's own mask: false` in the probe as a pre-flight check.

**Threads are usually inert under decomposition — spend the cores on ranks.**
`@threaded` runs serially unless a region's work reaches `THREAD_MIN_WORK`
(32768 points), so once `prod(n_local)` falls below that, every thread past the
first is idle no matter what `-t` says. 64³ over 56 ranks is 4608 points/rank,
seven times under; asking for `-t 56` there measured 1.7x *slower* than `-t 1`
from runtime and GC-thread overhead alone. Compute points/rank before requesting
threads, and remember `-t` is per rank: `-n R -t T` asks for R×T compute threads
plus R×(T/2) GC threads, which is how a launch ends up with ~19,000 threads on a
112-core node and never reaches step 1.

The threshold is per-rank, so this is a statement about the grid and not about
threading in general. 256³ stays above it out to 448 ranks (36864 points/rank),
where threads *would* engage — that regime is untested, and the 64³ result above
does not transfer to it, because there `@threaded` never engaged at all and the
1.7x was pure overhead. Anything measured there needs a probe first to confirm
the mask actually gives a rank more than one core to put a thread on.

**Rank count has a hard ceiling per grid.** `plan_direction` needs 9 points per
rank per dimension for the C8 filter, so `n_global[d] >= 9 * dims[d]`. At 64³
that caps you near 112 ranks; past it the run errors rather than degrading. Going
wider is a reason to raise the grid, not to fight the decomposition.

**Judge placement by cross-socket bytes, not percentage.** `Cart_create` is
row-major, so the *last* dimension is the fastest-varying rank index and its
neighbours are consecutive ranks. A dimension whose extent equals the socket
count must cross sockets under every possible mapping, and confining the crossing
to one axis is the good outcome — on rzhound `socket == coords[3]` exactly, which
reads as "100% cross-socket on dim 3" while leaving dims 1 and 2 fully
intra-socket. Weigh links × `n_halo` × the transverse plane before reaching for
an explicit `dims=`; reshuffling to a different axis there was ~17%, not a win.

**Check the shell rc before blaming the code.** `JULIA_NUM_THREADS` set in a
login file silently overrides the default in every launch that does not pass `-t`
explicitly (a command-line `-t` wins, so it corrupts probes rather than runs).
The probe prints `nthreads` for exactly this reason.

**Multi-node adds two failure modes single-node never shows.** MPI.jl's bundled
JLL can satisfy the scheduler's launcher on one node over shared memory and still
fail to reach the interconnect off it, so `MPIPreferences.use_system_binary()`
before the first multi-node run rather than after it fails — the symptom is a
hang at startup, not an error. And the probe's `off-node` column only becomes
non-zero there, which makes it the first place to look when per-step time jumps
on crossing a node boundary.

**You will not get the same node twice on a busy machine, so size the effect to
the instrument.** Order-of-magnitude differences read straight through node-to-
node variance; the few-percent kind does not, and neither does a scaling curve.
Hold placement fixed and repeat across whatever nodes the queue gives you, or
accept that the comparison is measuring the queue.

**Pack the nodes, then scale by adding nodes.** Measured, 256³ TGV under system
MVAPICH2 with `--cpu-bind=threads -t 1`, seconds per step:

| nodes | 56 ranks/node | 112 ranks/node |
|-------|---------------|----------------|
| 1     | 1.434         | 1.107          |
| 2     | 0.649         | 0.571          |
| 4     | 0.338         | 0.2966         |

Two effects, and they pull opposite ways without contradicting each other. At a
fixed *rank* count, halving ranks-per-node is worth **1.7x** (1.107 → 0.649 at
112 ranks; 0.571 → 0.338 at 224 — the same ratio twice, at different widths).
That is memory bandwidth per rank, not compute: the rank count and decomposition
are identical within each pair, and the compact line solves are bandwidth-bound.
It is the same mechanism behind the superlinear speedups you will see comparing
against a single-core baseline.

At a fixed *node* count, packing all 112 cores still wins, by 1.30x on one node
and **1.14x** on two and four. So the second 56 cores per node deliver only ~14%
more throughput for twice the core-hours. Pack if you are charged node-hours;
half-pack if core-hours. Do not extrapolate the 1.30 → 1.14 trend to an
inversion — it plateaus, which was checked rather than assumed.

Node scaling packed is 1.107 → 0.571 → 0.2966: 1.94x and 1.93x per doubling,
93% efficiency at four nodes. Off-node cost is close to free at this size once
the system MPI is in place, so scale by nodes and do not agonize over placement.

**Keep failure output survivable at scale.** An uncaught exception is reported
by every rank, so a `SolverFailure` at 448 ranks is 448 stacktraces. Wrap the
driver in `mpi_main`: full backtrace from rank 0, one line from anywhere else
that failed, then `MPI.Abort` so ranks still blocked in a collective are killed
rather than each printing a dump of its own. Non-zero ranks yield briefly before
aborting, because whoever calls `MPI_Abort` first silences everyone — including
rank 0 mid-backtrace, which is the output worth keeping. Use
`CL_ERROR_BACKTRACE=all` when the failure is rank-local, since rank 0 is then
blocked in a collective and nothing prints a location.

Ctrl-C is a separate problem with no code fix: it arrives while a rank sits
inside an MPI call, and Julia'''s signal handler prints a backtrace per rank
before anything can intercept it. Pass `--handle-signals=no` to `julia` for
sweeps, accepting that SIGSEGV backtraces go with it.

**Report progress with `ProgressLog`, not a hand-rolled callback.** `run!` fills
`solver.wall_step` and `solver.wall_total` on every step (rank-local by design —
a reduction there would be a collective every step for every run, read or not),
and `ProgressLog` turns those into a flushed progress line. Three details went
wrong in every hand-rolled copy this replaced: printing from all ranks rather
than one, not flushing (a batch launcher block-buffers stdout, so a healthy run
looks hung for minutes), and timing the diagnostic's own reduction as solver
cost. The `quantity` hook runs on every rank and may reduce; only the printing is
rank-guarded, which is the collective ordering from `nscbc.jl` again.

## The gate

Run all of it before calling a change safe.

```bash
MPIEXEC=$(julia --project=. -e 'using MPI; MPI.mpiexec(c -> print(c))')

julia --project=. test/runtests.jl        # 40 testsets, 0 failures
julia --project=. test/convergence.jl     # measured orders, see below
julia --project=. test/validation.jl      # shock-capturing battery, ~25 s
for np in 2 4 8; do
  "$MPIEXEC" -n $np julia --project=. test/mpi_tests.jl   # 30/30 each
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
- `trigger` (an `AtTime` / `EveryStep` / `WhenState`), `effect!`, `fired!`,
  `next_time`, `switch!`/`switched` (a `SwitchableBC`)

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
forwards to `after` only once switched, and if `after` runs collectives that
`before` does not — `NSCBCOutflowBC` does — then ranks disagreeing about the
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
  `amr_transfer.jl`, `tgv_energy.jl` (Taylor–Green kinetic-energy budget split
  by dissipation channel; the one bench script that runs usefully under
  `mpiexec`, and the intended first workload on a cluster).
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
- `C_mu` is uncalibrated for its actual purpose, and `bench/tgv_energy.jl` found
  out why it is harder than "run a 3-D case". TGV does make μ\* active (28% of
  the viscous budget at 32³, against inert in every 1-D case), but the **compact
  filter supplies ~83% of the total energy sink**, and it cannot be turned off to
  isolate μ\*: `filter_interval = 4` diverges and `0` fails with
  `SolverFailure(:negative_density)`. The filter is the primary stabilizer at
  coarse resolution, so it and `C_mu` have to be calibrated as a pair. Measured
  at 32³ only — the filter share did *not* fall from 16³ to 32³, and 64³ is ~13
  min per configuration, so confirming it at a quotable resolution wants a
  cluster. Numbers and caveats in `reference/CALIBRATION.md`.
- The NASA CEA thermo and limited transport databases ship verbatim in `data/`,
  with their Apache license and notice. `read_nasa9` handles multi-interval
  thermo records; the transport table is not yet connected to `Transport`.
- `compute_artificial!` is ~31% of the multicomponent RHS, spent in the filter
  line-solves that smooth the sensors, one sweep per species. Cutting it is a
  numerics decision (shared vs per-species sensor), not a code tweak. Note that
  at `n_species == 2` the per-species machinery is measurably a no-op: on a
  sharp binary interface `D*_1` and `D*_2` agree to 4.8e-16 relative and the
  correction velocity is 2.5e-18 against a species gradient of 84. It only
  earns its cost at three or more species.
- `ThreadPinning`'s querying API drives `clusterprobe.jl`; its *pinning* API is
  still unused. Whether pinning helps is open — on the one machine measured, the
  scheduler's own binding was already near-optimal (see **Cluster launches**),
  and the launch-line rules there matter more than anything pinning would add.
- **A rank holding both SMT threads of one core is catastrophic and unexplained.**
  ~4300x on rzhound, with memory, NUMA placement, load times, thread count and
  rank count all measured identical between the fast and slow cases. Four
  hypotheses were tested and rejected (cgroup memory starvation, collectives
  paying an OS timeslice, a sick node, NUMA placement). `--cpu-bind=threads`
  avoids it entirely, so this is a curiosity rather than a blocker — but the
  signature is sharp enough to be worth reporting to site support.
- A `bench/` runner taking medians over repeated *processes*, to get under the
  timing noise above.
- `FoldSpec` parameterization would close several remaining JET dispatch sites.
