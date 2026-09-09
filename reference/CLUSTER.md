# Running CompactLES on a cluster

MPI configuration, launch verification, run sizing, and scheduler launch rules.
`CLAUDE.md` carries a summary and refers here.

## Contents

1. [Configuring MPI](#configuring-mpi)
2. [Measuring the launch](#measuring-the-launch)
3. [Launch rules](#launch-rules)
4. [Scaling, measured](#scaling-measured)
5. [Open cluster questions](#open-cluster-questions)

## Configuring MPI

MPI.jl defaults to a bundled JLL. On one node it satisfies the launcher over
shared memory and reproduces the physics bit-for-bit, so single-node timing
cannot detect a misconfiguration. Off-node it may fail to reach the interconnect,
running an order of magnitude or more slower with no symptom but speed. At 256³
TGV the penalty grew with the off-node share: **27x at 224 ranks over 2 nodes,
66x at 448 ranks over 4 nodes**, same launch line each time.

Configure the system MPI once per checkout. Load its module first (so `libmpi` is
on `LD_LIBRARY_PATH`) and name the launcher explicitly; without `mpiexec=`,
`MPI.mpiexec` can keep returning the JLL launcher:

```bash
module load <site mpi>          # module avail mvapich2 / mpich / openmpi
mpicc -show                     # <-- the authoritative -L path for THIS module
julia --project=. -e 'using Pkg; Pkg.add("MPIPreferences")'
julia --project=. -e 'using MPIPreferences; MPIPreferences.use_system_binary( \
    library_names=["<that path>/libmpi.so"], mpiexec="srun")'
```

Use an absolute path. A module sets `PATH` but often not
`LD_LIBRARY_PATH`, because binaries find their libraries by RPATH while
`dlopen("libmpi")` has none and reports "MPI library could not be found" for every
candidate. `mpicc -show` names the `-L` directory of the build the module
resolved, the one whose scheduler integration the site tests; a bare `find` for
`libmpi.so*` turns up a dozen per-compiler builds with no indication which is
live. `libmpi` is C, so the path's compiler does not affect the ABI, but if
`dlopen` then wants a missing Intel runtime (`libimf`, `libsvml`), switch to a gcc
build of the same version. Prefer an MPICH-ABI implementation (MVAPICH2, MPICH)
over OpenMPI.

Three csh traps apply, since LC logins default to tcsh and every command in
this file is a `julia -e` one-liner. `module` exists only in csh, and
`2>/dev/null` does not parse there ("Ambiguous output redirect"); use `>&`
instead. A single-quoted string may not span a newline, so a multi-line
`-e '...'` ends the quote at the first line break and reports "Unmatched '".
And `!` triggers history expansion even inside single quotes, so
`set_preferences!` and `set_libraries!` must be written `set_preferences\!`.
Running the setup commands from a `bash` subshell avoids all three; `setenv`
variables and loaded modules are inherited.

If `MPI.Init` then hangs, or every rank reports itself as rank 0 of a 1-rank
world (plausible output at ~1/N speed, not an error), the launcher needs a
bootstrap flag: `srun --mpi=list`, then `--mpi=pmi2` for MVAPICH2 or `--mpi=pmix`
for OpenMPI. Then **restart Julia** (the preference is read at precompile) and
verify with `clusterprobe.jl`, whose `MPI binary` line reports `system` vs a
`_jll` and flags the JLL on any multi-node run.

Selecting the system binary also switches off the package's precompile
workload (`src/precompile.jl`), which initializes MPI as a singleton inside
the precompile process to build its solvers. Under the bundled binaries that
cuts the per-rank compile of a test run by about three quarters; under a
scheduler's process manager a singleton init on a login node may fail or
hang, and installing the package must not depend on it. Whether the workload
can run on a given cluster is untested; the check is a plain `julia
--project=. -e 'using MPI; MPI.Init()'` on the login node with the preference
set, and the gate to lift is the `MPIPreferences.binary` test at the top of
that file.

The preference lands in `LocalPreferences.toml`: gitignored, naming one machine's
library by path, and per-project, so `--project` determines the MPI. A driver
project that `dev`s this package does not inherit the preference and falls back to
the JLL silently, so probe with the same `--project` as the solver, and do not
put these preferences in a login file (they would redirect unrelated
environments). Run from the driver and locate scripts through `pkgdir` so
`--project=.` always picks up the configured environment:

```bash
srun -n 448 --cpu-bind=threads julia --project=. -t 1 \
    -e 'using CompactLES; include(joinpath(pkgdir(CompactLES), "examples", "taylor_green.jl"))'
```

### HDF5 against a system MPI

Setting `binary = "system"` makes HDF5_jll select its MPI-enabled variant,
which pulls in `MPICH_jll`. That package's `__init__` `dlopen`s an artifact
`libmpifort.so` that needs the symbol `MPIR_fortran_false`, which MPICH's
`libmpi` exports and MVAPICH2's does not, so `using HDF5` fails to precompile.
The MPICH ABI is compatible; the JLL's Fortran layer is not.

The fix is a system parallel HDF5 built against the same MPI, which removes
HDF5_jll and MPICH_jll from the load path. `HDF5.API.set_libraries!` cannot
install it, since it requires `using HDF5` to succeed first, so write the
preference directly:

```bash
julia --project=. -e 'using Preferences; set_preferences!(
    Base.UUID("f67ccb44-e63f-5c2f-98bd-6dc0ccc4ba2f"),
    "libhdf5" => "<prefix>/lib/libhdf5.so",
    "libhdf5_hl" => "<prefix>/lib/libhdf5_hl.so"; force = true)'
```

Before relying on it, check that `ldd <prefix>/lib/libhdf5.so | grep -i mpi`
resolves to the same `libmpi` given to MPIPreferences. Load the HDF5 module in
the launch environment as well as at setup, since `dlopen` has no RPATH, and
verify with `HDF5.API.libhdf5`, `HDF5.API.h5_get_libversion()` and
`hdf5_parallel()`. A leftover failure to precompile HDF5_jll is benign once the
preference is set: the JLL remains a dependency but is never loaded.

On rzhound this is `hdf5-parallel/1.14.0` under the
`MPI/intel-classic/2021.6.0-magic/mvapich2/2.3.7` hierarchy. It gives
`hdf5_parallel() == true` and exercises the collective-open path in
`with_shared_file`, which no workstation reaches: a 64 cubed snapshot written
that way at 16 ranks reproduces a serial single-process write to five or six
digits per spectral shell, diverging only where the two grids differ.

Write cost is negligible. Two 151 MB checkpoints from a 128 cubed run over 224
ranks cost under 1.5% of an 1176 s run, and that figure also includes the
per-step reductions of the same callback list. The transfer-mode note at the
top of `ext/CompactLESHDF5Ext.jl` records the measurement and the resulting
decision.

### One home directory, several clusters

When a home directory is shared across clusters, one `~/.julia` depot serves
every machine and no single `LocalPreferences.toml` can be correct for all of
them: a
checkout configured against one machine's MPI fails loudly on another when the
library name resolves nowhere, or silently falls back to the JLL when it
half-resolves. Keep one driver environment per architecture instead, using Julia
[shared
environments](https://pkgdocs.julialang.org/v1/environments/#Shared-environments)
named by an architecture variable (`$SYS_TYPE` on LC) and selected from the login
file:

```csh
if ($?SYS_TYPE) then
    setenv JULIA_PROJECT "@$SYS_TYPE"
    setenv JULIA_DEPOT_PATH "$HOME/.julia/$SYS_TYPE"":$HOME/.julia:"
    setenv OPENBLAS_NUM_THREADS 1
    switch ($SYS_TYPE)
        case toss_4_x86_64_ib_cray:     module load cray-mpich rocm ; breaksw
        case toss_4_x86_64_ib:          module load mvapich2 ; breaksw
    endsw
endif
```

The depot line uses two adjacent quoted strings because csh reads a `:` after a
variable name as a history modifier. Its three entries are the per-architecture
depot (takes every write, including precompile caches); the shared `~/.julia`,
listed explicitly for package sources, artifacts, and registries; and a trailing
empty entry, which expands to Julia's bundled system depots *only*; omit the
middle entry and every installed package is orphaned. A shared environment
resolves anywhere as `--project=@<name>`, and the first `Pkg.add` creates it. Set
each up once on its own cluster (`Pkg.develop` the checkout, add `MPIPreferences`
and any device package, then configure MPI as above); a preference block is inert
without its package present.

`OPENBLAS_NUM_THREADS` sits in the same block because it is a property of the
machine rather than of a run, and a rank that inherits the default thread count
pays for it on every compact solve; see the BLAS rule under Launch rules.

Sharing the depot's `compiled/` cache between clusters fails in practice, so the
per-architecture depot is standing configuration. The slots should coexist
(keyed by CPU target and preferences), but precompilation instead failed
repeatedly with "Image file failed consistency check", and the corrupt image
belonged to a dependency the error never named. After splitting the depot, wipe
the poisoned shared cache once (`rm -rf ~/.julia/compiled/v<minor>`): the shared
depot stays in the search stack, so a bad image there is findable until removed.

## Measuring the launch

Run `clusterprobe.jl` before interpreting any performance number. It runs no
solver steps and reports the allocated resources: node distribution, the MPI
library in use, each rank's affinity mask as logical *and* physical CPUs, whether
any CPU is double-booked, the cgroup memory limit against `Sys.total_memory()`,
`MPI.Init` and package-load time, `n_local`, whether `@threaded` engages at all,
and the share of neighbour links that stay intra-NUMA / cross-socket / off-node.

```bash
srun -n 56 --cpu-bind=threads julia --project=. clusterprobe.jl 128     # from the checkout
srun -n 56 --cpu-bind=threads julia --project=. \
    -e 'using CompactLES; include(joinpath(pkgdir(CompactLES), "clusterprobe.jl"))' 128
```

The second form is for a driver environment, whose MPI configuration differs from
the checkout's. The script lives at the repository root, where a bare path from
the driver does not resolve, so locate it through `pkgdir`. (The `-e` form loads
CompactLES before the script's `t_start`, so its `pkg load (s)` reads ~0; pass
the resolved path as an argument to measure it.) Run once per machine and
whenever a launch flag changes, and supply the intended grid: it sets the
decomposition, so a default-64³ probe can report floor values that do not apply
to the target job.

`clusterlaunch.jl` sizes a run without an allocation. Given a grid and core
budget it enumerates the legal rank counts (taking the minimum-extent floor from
the scheme objects) and reports points/rank, whether `@threaded` engages, and the
padded-to-interior halo ratio, which limits useful decomposition. It
reports the largest rank count within the halo-overhead limit and the largest
legal one; the gap measures the cost of using the whole allocation.

```bash
julia --project=. clusterlaunch.jl 256 nodes=36 cores_per_node=112
```

## Launch rules

**Threaded BLAS costs a factor of ninety and buys nothing.** The only BLAS this
solver calls is the reduced interface stage of the compact solve
(`_reduced_solve!` in `src/tridiag.jl`): a 2P x 2P system, 2x2 on a single rank,
with one right-hand side per line, solved once per dimension per field per
Runge-Kutta stage. OpenBLAS forks its thread pool for each of those and waits on
the join, so the cost is paid per call and grows with the node's core count while
the arithmetic it parallelizes stays trivial. On rzhound (112 logical CPUs,
Sapphire Rapids) a 32³ step measured 7.0 s at the default thread count and
0.079 s under `OPENBLAS_NUM_THREADS=1`, with nothing else changed; the profile
put 123 of the working thread's 191 samples in OpenBLAS's `exec_blas_async_wait`
before and one in a BLAS kernel after. The same setting is worth 12-17% on a
24-thread desktop, so this is not a cluster-only rule, it is a large-node rule
that a cluster makes unmissable. Set `OPENBLAS_NUM_THREADS=1` in the launch
environment; `clusterprobe.jl` and `depotprobe.jl` both report the count and
flag anything above 1.

**Scheduler flags do not necessarily correspond to the CPU mask, and the mapping
differs between machines.** On one, `-c` counted logical CPUs with SMT on, so
`-c 1` gave both threads of a core and `-c 2` gave one thread, inverted and
reproducible. On another, `-c` counted physical cores. Read the flag as a request
and the probe's mask as the record of what was allocated.

**`-N` alone does not spread ranks across nodes.** Without `--ntasks-per-node`,
Slurm may block-pack. One case put 111 ranks on one node and 1 on the other,
with a plausible self-consistent timing and no error, invalidating a matrix of
measurements before the imbalance appeared in the probe's node-distribution
line. Pass
`--ntasks-per-node` below full packing and read the node distribution first.

**Never assign both SMT threads of one core to a rank.** One machine ran 64³ TGV
~4300x slower with both siblings in the mask than with the same core count bound
one thread per rank, reproduced four times, with memory, NUMA, load times, and
thread/rank count all identical; the mask was the only difference and no
mechanism was found. `--cpu-bind=threads` avoids it; treat `SMT sibling in a
rank's own mask: false` as a pre-flight check.

**Threads are usually inactive under fine decomposition.** `@threaded` runs
serially unless a region reaches `THREAD_MIN_WORK` (1024 points per thread × the
rank's `-t`), and once `prod(n_local)` falls below it every thread past the first
idles. 64³ over 56 ranks is 4608 points/rank, well under the `-t 56` bar; asking
for `-t 56` there measured slower than `-t 1` from runtime and GC overhead alone.
Because `-t` is per rank, `-n R -t T` requests R×T compute plus R×(T/2) GC
threads; one configuration created ~19,000 threads on a node and never reached
the first step.

**Even where `@threaded` is active, threads lose to ranks at fixed core count.**
The solver is memory-bandwidth-bound, so a rank saturates bandwidth with one
thread while more threads add task and barrier overhead across ~120–150 regions
per RHS call. Representative 256³ TGV on two full nodes:

| ranks × threads | s/step | cores/rank |
|-----------------|--------|------------|
| 224 × 1         | 0.64   | 1–1        |
| 112 × 2         | 1.2    | 2–2 (verified) |
| 56 × 4          | 2.9    | 3–5 (ragged) |
| 28 × 8          | 8.0    | 7–14 (ragged) |

In the 112 × 2 row, every rank had a verified two-core mask and `@threaded`
was fully engaged. It was ~1.9x slower than 224 single-threaded ranks at the
same core count. Condensation does benefit the communication side, but the
thread-side losses consume it. These measurements support `-t 1` under MPI.
Spawn cost was ruled out directly (`bench/spawnfloor.jl`: ~0.6–0.8 µs per thread,
orders of magnitude under the step time), so the losses are bandwidth and
per-region barriers. Trixi.jl, the nearest comparable solver and one that offers a
hybrid mode, likewise published its large-scale TGV scaling as pure MPI.

**`--cpu-bind=threads` distributes CPUs non-uniformly below full packing** (3–5
cores per rank at half, 7–14 at a quarter), and an explicit `-c` does not repair
it. With collectives every step the slowest rank sets the pace, so these ragged
masks confound the bottom rows above; inspect the mask before trusting any
`-t > 1` result.

**Rank count has a fixed ceiling for a given grid.** The C8 filter needs 9 points
per rank per dimension, so `n_global[d] >= 9 * dims[d]`; at 64³ this caps the run
near 112 ranks, and larger counts fail at setup with an error, not a slowdown.

**Evaluate placement by cross-socket data volume, not link percentage.**
`Cart_create` is row-major, so the last dimension's neighbours are consecutive
ranks; a dimension whose extent equals the socket count must cross sockets under
every mapping. Weight links by `n_halo` and transverse-plane size before setting
an explicit `dims=`; moving the crossing to another axis was ~17% slower in one
measurement.

A `JULIA_NUM_THREADS` in a login file overrides the default when a launch omits
`-t`; a command-line `-t` wins, so the setting can still affect probes. The probe
reports `nthreads` for this reason.

### Batch scripts

`bench/slurm/` holds the scripts submitted for the N1 filter campaign,
`n1_legs23.sbatch` and `n1_closure128.sbatch` at 224 ranks and
`n1_confirm256.sbatch` at 896, kept as worked examples rather than as tooling.
Four practices in them are not implied by the launch rules above and belong in
any new script:

- **Precompile serially, in the same environment, inside the job.** One `julia
  --project=. -e 'using CompactLES, HDF5; println(hdf5_parallel())'` before the
  first `srun`. Every checkout shares one depot pidfile, so several hundred
  ranks racing for it is the documented way to hang a job with no output, and
  selecting a system MPI turns off the workload in `src/precompile.jl`, so the
  ranks would otherwise compile it themselves.
- **Guard the environment variables the script reads, and exit if one is
  unset.** A login shell that did not export `$CLES` or `$LFILES` otherwise
  spends the allocation running from the wrong directory or writing nowhere.
- **Run `lfs setstripe` on the output directory before the run**, with a count
  matching the writer's parallelism. HDF5 snapshots at 256³ are 1.2 GB each.
- **Run `clusterprobe.jl` at any new node or rank count**, in the same
  allocation, before the long `srun`. It records the CPU mask, the node
  distribution and the decomposition against which the timings are then read.

The login shell on rzhound is tcsh, and the csh traps under
[Configuring MPI](#configuring-mpi) apply inside a batch script as well, so
keep inline `julia -e` expressions to one line. Pass `--handle-signals=no` to
a sweep; Ctrl-C has no code fix.

## Scaling, measured

The strategy packs each node before adding nodes. Representative seconds per step
for 256³ TGV under system MPI with `--cpu-bind=threads -t 1`:

| nodes | 56 ranks/node | 112 ranks/node |
|-------|---------------|----------------|
| 1     | 1.434         | 1.107          |
| 2     | 0.649         | 0.571          |
| 4     | 0.338         | 0.297          |

Two effects act oppositely. At fixed *rank* count, halving ranks-per-node
improves performance ~1.7x, attributable to memory bandwidth per rank, since the
compact line solves are bandwidth-bound and the decomposition is identical
within each pair.
At fixed *node* count, packing all cores still wins but by a shrinking margin
(1.30x on one node, ~1.14x on two and four): the second half of the cores gives
~14% more throughput for twice the core-hours, so full packing favors node-hour
efficiency and half packing favors core-hour. Node scaling packed was ~1.93x per
doubling (93% at four nodes), so off-node communication costs little here.

## Open cluster questions

- Whether shared-memory threads can beat per-core ranks: no at fixed core count
  on every machine measured (~1.9x slower with verified masks). Outstanding are
  the fixed-rank control on identical masks, medians over repeats, and the one
  configuration with a theoretical opening (one rank per NUMA domain with pinned
  threads under Julia ≥ 1.13), which depends on the pinning item below and on
  masks the scheduler will not produce unaided.
- `ThreadPinning`'s querying API drives `clusterprobe.jl`; its *pinning* API is
  unused. Whether it helps is open; on the machines measured the scheduler's own
  binding was near-optimal, and the launch-line rules had a greater effect.
- The ~4300x SMT-sibling collapse is unexplained after four hypotheses were
  tested and rejected (cgroup memory starvation, collectives paying an OS
  timeslice, a sick node, NUMA placement). The reproducible signature should be
  reported to site support.
