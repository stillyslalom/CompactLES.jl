# Running CompactLES on a cluster

Everything about getting this code onto a machine with a scheduler: pointing
MPI.jl at the system library, measuring what a launch actually got, sizing a run
before queueing it, and the launch-line rules. `CLAUDE.md` carries only the
one-paragraph summary and sends you here.

**Every rule below came out of `clusterprobe.jl`, and each was a *failed*
hypothesis first.** Numbers are measured on rzhound (LLNL): 2 sockets, 8 NUMA
domains, 112 cores, SMT2. Treat the mechanisms as portable and the numbers as
one machine's.

## Contents

1. [Configuring MPI](#configuring-mpi)
2. [Measuring the launch](#measuring-the-launch)
3. [Launch rules](#launch-rules)
4. [Scaling, measured](#scaling-measured)
5. [Open cluster questions](#open-cluster-questions)

## Configuring MPI

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
the top of `CLAUDE.md` quietly lies to you:

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

## Measuring the launch

**Measure the launch before you believe a timing from it.** `clusterprobe.jl`
takes seconds, runs no solver steps, and reports what the ranks actually got:
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

The second form is the one to use from a driver environment, and it is easy to
get wrong in the direction that fails loudly *and* the direction that fails
quietly. `clusterprobe.jl` lives at the repo root, not in `bench/`, so a bare
`julia --project=. clusterprobe.jl` from the driver finds no such file — while
running it from the checkout instead finds the file and then reports on the
*checkout's* MPI configuration rather than the one the solver will use, which is
the whole thing the probe exists to catch. Same `--project` as the solver,
script reached through `pkgdir`.

One measurement degrades under the `-e` form: `using CompactLES` there happens
before the script's own `t_start`, so `pkg load (s)` reports ~0. `MPI.Init (s)`
is unaffected (loading MPI is not initializing it), as is everything else. If the
load time is what you are chasing, resolve the path first and pass it as a script
argument instead.

Run it once per new machine, and again whenever a launch flag changes. Every
rule below came out of it, and each one was a *failed* hypothesis first. Pass the
grid you actually intend to run — it decides the decomposition, so at the default
64³ the `n_local` and scheme-floor lines describe a different job than yours.

**`clusterlaunch.jl` sizes the run before you queue it.** It launches nothing and
needs no allocation — given a grid and a core budget it enumerates the legal rank
counts, taking the minimum-extent floor from the scheme objects themselves rather
than a remembered 9, and reports points/rank, whether `@threaded` engages, and
the padded-to-interior ratio (the redundant halo arithmetic, which is what
actually limits how far this solver decomposes). It names two picks, because they
answer different questions: the widest count whose halo cost is still in hand,
and the widest legal one for when wall time is the constraint.

```bash
julia --project=. clusterlaunch.jl 256 nodes=36 cores_per_node=112
```

The gap between those two is the real sizing signal. At 64³ on one 112-core node
it is 8 ranks against 112 (1.95x vs 4.25x halo) — that grid is simply too small
for the node, which is the arithmetic reason `reference/CALIBRATION.md` wants a
larger one, not just a physics preference. At 256³ on 36 nodes it is 448 against
4032, and the whole allocation becomes defensible.


## Launch rules

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

And threads do not help even where `@threaded` genuinely engages. Measured at
256³ on two nodes, every configuration using all 112 cores per node:

| ranks × threads | s/step | cores/rank from the probe |
|-----------------|--------|---------------------------|
| 224 × 1         | 0.568  | 1–1                       |
| 112 × 2         | 0.865  | 2–2                       |
| 56 × 4          | 2.17   | 3–5 (ragged, see below)   |
| 28 × 8          | 6.93   | 7–14 (ragged)             |

The 112 × 2 row is the clean comparison: against 112 × 1 at 0.649, a second
thread per rank made it **1.33x slower while using twice the cores** — same rank
count, same `(7,4,4)` decomposition, same halo, nothing changed but the thread.
This solver is memory-bandwidth-bound (see the ranks-per-node note below), so a
rank is already saturated with one thread and the second buys no throughput while
still paying task spawn and a barrier across ~150 regions per RHS call, five
stages a step. **Use `-t 1` under MPI and spend the cores on ranks.**

`--cpu-bind=threads` also distributes CPUs *non-uniformly* below 56 ranks/node —
3 to 5 cores per rank at 28/node, 7 to 14 at 14/node. With collectives every
step the slowest rank paces all of them, so a ragged mask is worse than a
uniformly smaller one, and the bottom two rows above are contaminated by it
rather than being clean threading measurements. Probe the mask before believing
any `-t > 1` timing.

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


## Scaling, measured

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

## Open cluster questions

- `ThreadPinning`'s querying API drives `clusterprobe.jl`; its *pinning* API is
  still unused. Whether pinning helps is open — on the one machine measured, the
  scheduler's own binding was already near-optimal (see [Launch
  rules](#launch-rules)),
  and the launch-line rules there matter more than anything pinning would add.
- **A rank holding both SMT threads of one core is catastrophic and unexplained.**
  ~4300x on rzhound, with memory, NUMA placement, load times, thread count and
  rank count all measured identical between the fast and slow cases. Four
  hypotheses were tested and rejected (cgroup memory starvation, collectives
  paying an OS timeslice, a sick node, NUMA placement). `--cpu-bind=threads`
  avoids it entirely, so this is a curiosity rather than a blocker — but the
  signature is sharp enough to be worth reporting to site support.
