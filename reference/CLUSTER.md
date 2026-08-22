# Running CompactLES on a cluster

This document covers MPI configuration, launch verification, run sizing, and
scheduler launch rules. `CLAUDE.md` contains a summary and refers to this
document for cluster-specific procedures.

The rules are based on `clusterprobe.jl` measurements and tests of alternative
hypotheses. Numerical results are from rzhound and rzwhippet (LLNL), which
share one topology: 2 sockets, 8 NUMA domains, 112 cores, SMT2. Each number is
labeled with its machine. The mechanisms may generalize, but machine-to-machine
differences exceed run-to-run spread, so values from different machines are not
comparable — even the meaning of a scheduler flag differs between them (see
[Launch rules](#launch-rules)).

## Contents

1. [Configuring MPI](#configuring-mpi)
2. [Measuring the launch](#measuring-the-launch)
3. [Launch rules](#launch-rules)
4. [Scaling, measured](#scaling-measured)
5. [Open cluster questions](#open-cluster-questions)

## Configuring MPI

MPI.jl defaults to its bundled JLL. On a single node, that implementation can
satisfy the scheduler launcher over shared memory, form a correct communicator,
and reproduce the physics bit-for-bit. A multi-node run may nevertheless fail to
use the interconnect. Measured on rzhound, 256³ TGV
at 224 ranks over two nodes: **15.19 s/step on the JLL, 0.571 s/step on the
system MVAPICH2 — 27x, same launch line, same decomposition.** On-node the two
are indistinguishable (1.434 vs 1.372 s/step at 56 ranks, inside run-to-run
spread), so single-node timing does not identify the configuration error.

Configure the system MPI once per checkout. Load its module first so that
`libmpi` is on `LD_LIBRARY_PATH`, and specify the launcher. Without `mpiexec=`,
`MPI.mpiexec` can continue to return the JLL launcher:

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
cleanly. `$MPI_ROOT` is unset on LC machines. `mpicc -show` names the exact `-L`
directory of the build resolved by the module, which
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
world, the launcher requires an explicit bootstrap: `srun --mpi=list`,
then `--mpi=pmi2` for MVAPICH2 or `--mpi=pmix` for OpenMPI. The rank-0-of-one
rank-0-of-one case produces plausible output at approximately 1/N of the
expected speed rather than raising an error.

Then **restart Julia**, since the preference is read at precompile, and verify with
`MPI.versioninfo()`, or run `clusterprobe.jl`, whose `MPI binary` line reports
`system` versus a `_jll` outright and flags the JLL on any multi-node run. Do
this inside an allocation if the login and compute nodes carry different stacks.

The preference lands in `LocalPreferences.toml`, which is gitignored: it names
one machine's library by path and must not travel between checkouts.

`LocalPreferences.toml` is project-specific, so `--project` determines the MPI
implementation.
A second environment that `dev`s this package, say a driver project alongside
`~/.julia/dev/CompactLES`, does not inherit the preference, and running the same
script against the wrong one falls back to the JLL with no symptom other than
speed. Measured both ways at 256³: 224 ranks over 2 nodes, 15.19 s/step on the
JLL against 0.571; 448 ranks over 4 nodes, 19.6 against 0.2966 — **27x and 66x**,
the penalty growing with off-node share (25% then 50% of dimension 1's links).
Each launch environment requires its own configuration. Run `clusterprobe.jl`
with the same `--project` as the solver so its `MPI binary` line describes the
environment being timed.

With a driver environment separate from the checkout, run from the driver and
locate the scripts through `pkgdir`. `--project=.` then always picks up the
configured environment, and a pull into the checkout needs no copying:

```bash
srun -n 448 --cpu-bind=threads julia --project=. -t 1 \
    -e 'using CompactLES; include(joinpath(pkgdir(CompactLES), "examples", "taylor_green.jl"))'
```

Do not place project-specific MPI preferences in a login file because they would
redirect unrelated Julia environments.

### One home directory, several clusters

LC home directories are shared across the RZ clusters, so one checkout and one
`~/.julia` depot serve every machine — and one `LocalPreferences.toml` cannot
be correct for more than one of them. A checkout configured against rzadams'
`libmpi_cray` fails loudly on rzhound, since the Cray name resolves nowhere
there; the reverse misconfiguration, or an unconfigured environment, falls back
to the JLL silently. Keep one driver environment per architecture instead,
using Julia's [shared
environments](https://pkgdocs.julialang.org/v1/environments/#Shared-environments)
named by `$SYS_TYPE` (`toss_4_x86_64_ib` on rzhound, `toss_4_x86_64_ib_cray`
on rzadams) and selected from `.cshrc`:

```csh
if ($?SYS_TYPE) then
    setenv JULIA_PROJECT "@$SYS_TYPE"
    setenv JULIA_DEPOT_PATH "$HOME/.julia/$SYS_TYPE"":$HOME/.julia:"
    switch ($SYS_TYPE)
        case toss_4_x86_64_ib_cray:     # rzadams
            module load cray-mpich rocm
            breaksw
        case toss_4_x86_64_ib:          # rzhound
            module load mvapich2
            breaksw
    endsw
endif
```

The depot line is written in two adjacent quoted strings because csh parses a
`:` immediately after a variable name as a history-style modifier; adjacent
strings concatenate. Its three entries are the per-architecture depot, which
receives every write including precompile caches; the shared `~/.julia`, which
must be listed explicitly and serves package sources, artifacts, and
registries read-through; and a trailing empty entry, which expands to Julia's
bundled system depots only — it does *not* include the user depot (verified on
1.11.4), so omitting the middle entry orphans every installed package.

A shared environment lives at `~/.julia/environments/<name>/` and is
addressable from anywhere as `--project=@<name>` or
`Pkg.activate("<name>"; shared = true)`; both `JULIA_PROJECT` and `--project`
accept the `@` form, and a name that does not exist yet resolves to the path
it will occupy, so the first `Pkg.add` creates it (verified on Julia 1.11.4).
`JULIA_PROJECT` then selects the right environment for every `julia`
invocation that does not pass `--project`, which makes launch lines portable
between the clusters. This does not conflict with the rule against preferences
in login files: the login file selects a project, and the preferences stay in
that project's `LocalPreferences.toml`. An explicit `--project` still
overrides the variable, so `--project=.` in the checkout runs against whatever
the checkout's own preference file holds. Set each environment up once on its
own cluster: `Pkg.develop` the checkout, add `MPIPreferences` (and the device
package where there is a device — preferences apply only to packages present
in the environment, so an `[AMDGPU]` block is inert without
`Pkg.add("AMDGPU")`), then configure the MPI preference as above. The Cray PE sets
`LD_LIBRARY_PATH`, so the bare name `libmpi_cray` resolves on rzadams; the
MVAPICH2 module does not, which is why rzhound needs the absolute path from
`mpicc -show`.

Sharing the depot's `compiled/` directory between the clusters does not work
in practice, which is why the depot line above is part of the standing
configuration rather than a contingency. In principle the cache slots are
keyed by CPU target and preferences and the machines should coexist; measured
on rzhound (Julia 1.12.7), precompilation instead failed repeatedly with
"Image file failed consistency check: maybe opened the wrong version?" while
recompiling two dozen stdlibs, and deleting the caches of the packages named
in the error did not clear it — the corrupt image belonged to a dependency
the error never named. After introducing the per-architecture depot, wipe the
poisoned shared cache once (`rm -rf ~/.julia/compiled/v1.12`, the directory
tracking the Julia minor version in use): the shared depot remains in the
search stack, so a bad image there is still findable until removed. The first
`Pkg.precompile()` in the new stack rebuilds everything into the
per-architecture depot; it completed cleanly on rzhound, and rzadams builds
its own set on its next login with no further action.

## Measuring the launch

Run `clusterprobe.jl` before interpreting performance measurements. It takes
seconds, runs no solver steps, and reports the allocated resources:
node distribution, the MPI library in use, each rank's CPU affinity mask as
logical *and* physical CPUs, whether any CPU is double-booked, the cgroup memory
limit against what `Sys.total_memory()` claims, `MPI.Init` and package-load time,
the resulting `n_local`, whether `@threaded` engages at all, and the share of
Cartesian neighbour links that stay intra-NUMA / cross-socket / off-node.

```bash
srun -n 56 --cpu-bind=threads julia --project=. clusterprobe.jl 128     # from the checkout
srun -n 56 --cpu-bind=threads julia --project=. \
    -e 'using CompactLES; include(joinpath(pkgdir(CompactLES), "clusterprobe.jl"))' 128
```

The second form applies to a driver environment. `clusterprobe.jl` is located at
the repository root rather than in `bench/`, so a bare
`julia --project=. clusterprobe.jl` from the driver does not find the file.
Running it from the checkout instead reports the checkout's MPI configuration,
which may differ from that of the driver. Use the solver's `--project` and locate
the script through `pkgdir`.

One measurement degrades under the `-e` form: `using CompactLES` there happens
before the script's own `t_start`, so `pkg load (s)` reports ~0. `MPI.Init (s)`
is unaffected because loading MPI does not initialize it. To measure package
load time, resolve the path first and pass it as a script argument.

Run it once per machine and whenever a launch flag changes. Supply the intended
grid because it determines the decomposition; a probe at the default 64³ can
report `n_local` and scheme-floor values that do not apply to the target job.

`clusterlaunch.jl` sizes a run without launching it or requiring an allocation.
Given a grid and a core budget, it enumerates the legal rank
counts, taking the minimum-extent floor from the scheme objects themselves rather
than a remembered 9, and reports points/rank, whether `@threaded` engages, and
the padded-to-interior ratio (the redundant halo arithmetic, which is what
limits useful decomposition). It reports two configurations: the largest rank
count within the selected halo-overhead limit and the largest legal rank count.

```bash
julia --project=. clusterlaunch.jl 256 nodes=36 cores_per_node=112
```

The gap between these configurations measures the cost of using the complete
allocation. At 64³ on one 112-core node, the two choices are 8 and 112 ranks
(1.95x and 4.25x halo overhead), indicating that the grid is too small for the
node. At 256³ on 36 nodes, the corresponding choices are 448 and 4032 ranks.


## Launch rules

Scheduler flags do not necessarily correspond to the resulting CPU mask, and
the mapping differs between machines. On rzhound, where the scheduler
allocates at core granularity with SMT enabled, `-c 1` yields both threads of
one core (2 logical CPUs) and `-c 2` yields one thread (1 logical CPU) —
measured, inverted, reproducible across nodes. On rzwhippet, `-c` counts
physical cores: `--ntasks-per-node=56 -c 2` fills a 112-core node exactly and
gives each rank 2 physical cores with no SMT siblings, while `-c 4` at 56
tasks/node is refused outright ("CPU count per node can not be satisfied").
Read the flag as a request and the probe's affinity mask as the record of
what the scheduler actually allocated.

`-N` alone does not spread ranks across nodes. Without `--ntasks-per-node`,
Slurm may block-pack: `-N 2 -n 112 --cpu-bind=threads` on rzwhippet placed
111 ranks on one node and 1 on the other, one logical CPU per rank, leaving
the second node nearly idle. The failure mode is not an error but a plausible
timing — that configuration ran the 256³ TGV at 1.655 s/step, self-consistent
with 111 single-CPU ranks sharing one node — and it invalidated a full matrix
of threaded measurements before the probe's node-distribution line revealed
the packing. Pass `--ntasks-per-node` whenever ranks-per-node is below full
packing, and read the node distribution before trusting any number.

Avoid assigning both SMT threads of one physical core to a rank. On rzhound (2
sockets, 8 NUMA domains, 112 cores, SMT2) that configuration ran the 64³ TGV at
154 s/step against 0.036 s/step for the same 56 physical cores bound one thread
per rank — a factor of ~4300, reproduced four times. Memory, NUMA placement,
package-load time, `MPI.Init` time, thread count and rank count were all measured
identical across the fast and slow cases; the mask is the only difference and no
mechanism has been established. `--cpu-bind=threads` avoids it. Treat
`SMT sibling in a rank's own mask: false` in the probe as a pre-flight check.

Shared-memory threads are generally inactive under fine decomposition.
`@threaded` runs serially unless a region's work reaches `THREAD_MIN_WORK`
(1024 points per thread times the rank's `-t`, so the bar rises with the
threads requested), and once `prod(n_local)` falls below it every thread past
the first is idle no matter what `-t` says. 64³ over 56 ranks is 4608
points/rank, twelve times under the `-t 56` bar; asking for `-t 56` there
measured 1.7x *slower* than `-t 1` from runtime and GC-thread overhead alone.
Because `-t` applies per rank,
`-n R -t T` requests R×T compute threads plus R×(T/2) GC threads. One tested
configuration consequently created approximately 19,000 threads on a 112-core
node and did not reach the first step.

Threads also reduced performance in cases where `@threaded` was active. The
following measurements, on rzhound, use 256³ on two nodes and all 112 cores
per node:

| ranks × threads | s/step | cores/rank from the probe |
|-----------------|--------|---------------------------|
| 224 × 1         | 0.568  | 1–1                       |
| 112 × 2         | 0.865  | 2–2                       |
| 56 × 4          | 2.17   | 3–5 (ragged, see below)   |
| 28 × 8          | 6.93   | 7–14 (ragged)             |

The 112 × 2 row is the clean comparison: against 112 × 1 at 0.649, a second
thread per rank made it 1.33 times slower while using twice the cores. Rank
count, `(7,4,4)` decomposition, and halo overhead were unchanged. The solver is
memory-bandwidth-bound, so a rank saturates the available bandwidth with one
thread while additional threads add task and barrier overhead across
approximately 150 regions per RHS call. The measured configuration therefore
supports `-t 1` under MPI.

`--cpu-bind=threads` also distributes CPUs *non-uniformly* below 56 ranks/node —
3 to 5 cores per rank at 28/node, 7 to 14 at 14/node, measured the same on
both machines — and an explicit `-c` does not repair it: on rzwhippet,
`--ntasks-per-node=28 -c 4` still produced 3–5 cores per rank. With
collectives every step the slowest rank determines performance. The nonuniform
masks confound the bottom two rows of both threads tables, so they do not
isolate thread scaling. Inspect the mask before interpreting any `-t > 1`
result.

Rank count has a fixed ceiling for a given grid. `plan_direction` needs 9 points per
rank per dimension for the C8 filter, so `n_global[d] >= 9 * dims[d]`. At 64³
this limits the calculation to approximately 112 ranks; larger counts produce an
error rather than degraded performance. A larger rank count requires a larger
grid.

Evaluate placement by cross-socket data volume rather than link percentage.
`Cart_create` is
row-major, so the *last* dimension is the fastest-varying rank index and its
neighbours are consecutive ranks. A dimension whose extent equals the socket
count must cross sockets under every possible mapping. On rzhound, the crossing
is confined to one axis and `socket == coords[3]`, which
reads as "100% cross-socket on dim 3" while leaving dims 1 and 2 fully
intra-socket. Weight links by `n_halo` and the transverse-plane size before
selecting an explicit `dims=`; moving the crossing to another axis was
approximately 17% slower in this measurement.

A `JULIA_NUM_THREADS` setting in a login file overrides the default in launches
that do not pass `-t` explicitly. A command-line `-t` takes precedence, so the
environment setting can affect probes even when production runs are unaffected.
The probe therefore reports `nthreads`.

Multi-node runs add two failure modes. MPI.jl's bundled JLL can satisfy the
scheduler launcher on one node over shared memory but fail to reach the
interconnect. Configure `MPIPreferences.use_system_binary()` before the first
multi-node run; the failure may appear as a startup hang. In addition, the
probe's `off-node` column becomes nonzero only in a multi-node run and should be
examined when step time increases across a node boundary.

Node-to-node variation on a busy machine limits the precision of performance
comparisons. Order-of-magnitude differences exceed this variance, whereas
few-percent effects and scaling curves require fixed placement or repeated
measurements across the nodes assigned by the scheduler.

### Threads versus ranks, re-measured (rzwhippet, 2026-08)

The question was reopened after the fused scatters, the blocked x-sweep
solve, and the per-thread `THREAD_MIN_WORK` default changed the threading
economics: roughly 30 fewer threaded regions per RHS call (~120 remain), a
faster serial baseline, and no spawns wasted on undersized regions. Measured
on rzwhippet under Julia 1.12.7 — 256³ TGV with artificial properties on, 60
steps, single runs, launch flags per the rules above and every row's mask
probe-verified:

| ranks × threads | s/step | cores/rank from the probe |
|-----------------|--------|---------------------------|
| 224 × 1         | 0.640  | 1–1                       |
| 112 × 2         | 1.206  | 2–2                       |
| 56 × 4          | 2.94   | 3–5 (ragged)              |
| 28 × 8          | 7.97   | 7–14 (ragged)             |

The verdict at fixed core count is unchanged: two threads per rank on
verified two-core masks lost to pure per-core ranks by 1.88x, with
`@threaded` fully engaged (~150k points/rank against a 2048-point floor).
The condensation benefit that motivated the hybrid is real on the
communication side — at 28 ranks the decomposition drops to (7, 2, 2) with
dimension 3 entirely intra-NUMA and only dimension 1 off-node — but the
thread-side losses consume it entirely. Not yet measured here: the
fixed-rank control (112 × 1 on the identical `-c 2` masks; the analogous
rzhound comparison is what separated bandwidth-per-rank from active thread
overhead) and medians over repeats.

The per-region cost was measured directly with `bench/spawnfloor.jl` (one
rank on one node, one process per `-t`). The spawn/join floor of a single
`Threads.@threads` region grows linearly at ~0.6–0.8 µs per thread through
`-t 56` (0.86 µs at 1 thread, 41 µs at 56). At ~120 regions per RHS call
that is 0.2–0.5 ms per RHS at `-t 2`–`-t 8`, three orders of magnitude under
the step time, so the table's losses are not spawn cost; they are memory
bandwidth and per-region barriers. The linear growth itself is inherent to
`@threads`, which spawns one task per thread and joins a barrier over all of
them. At `-t 112` the floor was pathological rather than linear (minimum
838 µs, median 20 ms): that part is a Julia runtime defect, the scheduler
wake-storm of [JuliaLang/julia#50425](https://github.com/JuliaLang/julia/issues/50425)
— each region entry woke every thread through a serial lock/signal/unlock
loop — fixed by [#61826](https://github.com/JuliaLang/julia/pull/61826)
(June 2026, backported to 1.13). Under 1.13-rc3 the floor remains linear at
a similar slope while the `-t 112` median collapses 30x (21.7 ms → 0.72 ms).
A Julia upgrade therefore repairs only the extreme tail and does not change
the economics at thread counts a production hybrid would use.

Community experience agrees: task-spawn floors of 3.5–70 µs across machines
are documented in the Julia Discourse thread ["Overhead of
`Threads.@threads`"](https://discourse.julialang.org/t/overhead-of-threads-threads/53964),
whose consensus is that `@threads` does not suit many small regions, and
Trixi.jl — the nearest comparable solver, which ships a hybrid MPI-threads
mode — published its 2026 large-scale Taylor–Green scaling as pure MPI
without multithreading.

## Scaling, measured

The measured scaling strategy packs each node before adding nodes. The following
table gives seconds per step for 256³ TGV on rzhound under system MVAPICH2 with
`--cpu-bind=threads -t 1`:

| nodes | 56 ranks/node | 112 ranks/node |
|-------|---------------|----------------|
| 1     | 1.434         | 1.107          |
| 2     | 0.649         | 0.571          |
| 4     | 0.338         | 0.2966         |

Two effects act in opposite directions. At a fixed rank count, halving
ranks-per-node improves performance by a factor of 1.7 (1.107 → 0.649 at
112 ranks; 0.571 → 0.338 at 224 — the same ratio twice, at different widths).
That is memory bandwidth per rank, not compute: the rank count and decomposition
are identical within each pair, and the compact line solves are bandwidth-bound.
The same memory-bandwidth effect produces superlinear speedup relative to a
single-core baseline.

At a fixed *node* count, packing all 112 cores still wins, by 1.30x on one node
and 1.14 times on two and four. The second 56 cores per node therefore deliver
approximately 14% more throughput for twice the core-hours. Full packing favors
node-hour efficiency, whereas half packing favors core-hour efficiency. The
measured trend plateaus rather than inverting.

Node scaling packed is 1.107 → 0.571 → 0.2966: 1.94x and 1.93x per doubling,
93% efficiency at four nodes. With the system MPI, off-node communication has a
small measured cost at this size.

## Open cluster questions

- Whether shared-memory threads can ever beat per-core ranks was re-measured
  on rzwhippet after the threading-economics changes and the verdict is
  unchanged: 1.88x slower at fixed core count with verified masks (see
  [Threads versus ranks, re-measured](#threads-versus-ranks-re-measured-rzwhippet-2026-08)).
  Outstanding: the fixed-rank 112 × 1 control on identical masks, medians
  over repeats, and the one configuration with a theoretical opening — one
  rank per NUMA domain with pinned threads under Julia ≥ 1.13 — which depends
  on the pinning item below and on masks the scheduler will not produce
  unaided.
- `ThreadPinning`'s querying API drives `clusterprobe.jl`; its *pinning* API is
  still unused. Whether pinning helps is open — on the one machine measured, the
  scheduler's own binding was already near-optimal (see [Launch
  rules](#launch-rules)),
  and the launch-line rules there matter more than anything pinning would add.
- Assigning both SMT threads of one core to a rank causes an unexplained
  performance reduction of approximately 4300-fold on rzhound. Memory, NUMA
  placement, load times, thread count, and
  rank count all measured identical between the fast and slow cases. Four
  hypotheses were tested and rejected (cgroup memory starvation, collectives
  paying an OS timeslice, a sick node, NUMA placement). `--cpu-bind=threads`
  avoids it. The reproducible signature should be reported to site support.
