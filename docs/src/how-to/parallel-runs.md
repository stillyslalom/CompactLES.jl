# Run in parallel

CompactLES combines MPI domain decomposition with shared-memory threading.
MPI calls occur outside threaded regions and require the `:funneled` thread
level.

## Obtain the configured launcher

Do not assume that `mpiexec` on `PATH` belongs to the MPI library used by
MPI.jl:

```julia
using MPI

MPI.mpiexec() do mpiexec
    julia = Base.julia_cmd()
    run(`$mpiexec -n 4 $julia --project=. -t 2 case.jl`)
end
```

Initialize MPI in the case script before setup:

```julia
using MPI
MPI.Init(threadlevel = :funneled)
using CompactLES
```

Wrap a driver in [`mpi_main`](@ref) so an exception is reported once and all
ranks return a nonzero status without printing one full stacktrace per rank.

## Choose a process grid

With `Numerics(dims = nothing)`, MPI distributes ranks across resolved
dimensions. Supply `(p1, p2, p3)` to control it:

```julia
Numerics(n_global = (512, 128, 1), dims = (4, 2, 1))
```

The following constraints are enforced:

- `prod(dims)` equals communicator size;
- a collapsed dimension has process-grid extent one; and
- every local extent is large enough for the selected operators.

The default compact filter requires at least nine points per local resolved
extent, giving the useful preflight condition
`n_global[d] >= 9dims[d]`. C6 alone requires five and C10 alone seven, but the
filter is normally the binding constraint.

## Combine ranks and threads

Point and line loops are threaded only when their work estimate exceeds the
internal `CompactLES.THREAD_MIN_WORK` threshold (by default 1024 points per
thread times the session's thread count) and only when the loop has more than
one iteration to divide. Consequently, one-dimensional and small
documentation cases often run intentionally on one thread. More threads do not
compensate for a small local block.

A grid with one collapsed dimension is not a small block, and its pointwise loops
are threaded. They iterate their two outer indices as a single flattened space,
so a planar `(nx, ny, 1)` or axisymmetric `(nr, 1, nz)` run divides over
whichever of the two is resolved.

Threading a block does not make it as fast as splitting it into ranks. At a
fixed core count, one-thread ranks have beaten multithreaded ranks by about
2x on every machine measured, from a two-species 768×48 tube on an eight-core
workstation to 256³ Taylor--Green on two 112-core nodes. On a workstation,
launch anything above 1-D as `mpiexec -n 8 julia -t 1` rather than
`julia -t 8`; on a hybrid performance/efficiency-core desktop, also confine
the process to the performance cores when timing. The mechanism, the
measurements and the two configurations in which threads do have an opening
are in [Threads and ranks](@ref).

## Launch on a cluster

MPI.jl defaults to a bundled MPI binary. On one node it satisfies the
scheduler's launcher over shared memory and reproduces the physics
bit-for-bit, so a single-node timing cannot detect a misconfiguration. Off one
node it may never reach the interconnect, and the only symptom is speed: a
256³ Taylor--Green run measured 27 times slower at 224 ranks over two nodes
and 66 times slower at 448 ranks over four, on the same launch line.
Configure the system MPI once per checkout, with the site module loaded so
its `libmpi` can be found, and name the launcher explicitly:

```julia
using MPIPreferences
MPIPreferences.use_system_binary(
    library_names = ["<the module's library directory>/libmpi.so"],
    mpiexec = "srun")
```

The preference is stored per project in `LocalPreferences.toml`, so the
`--project` a launch names selects the MPI implementation.

The measured launch rules are:

- Set `OPENBLAS_NUM_THREADS=1` in the launch environment. The only BLAS call
  is the small reduced interface stage of each compact solve, and a threaded
  OpenBLAS forks and joins its pool on every one: a 32³ step measured 7.0 s at
  the default thread count of a 112-core node and 0.079 s at one thread. The
  package sets one thread itself when the variable is unset.
- Run one thread per rank (`-t 1`). The solver is memory-bandwidth-bound, and
  at a fixed core count ranks beat threads: 256³ Taylor--Green on two full
  nodes took 0.64 s per step at 224 single-threaded ranks and 1.2 s at 112
  ranks of two threads each.
- Bind one thread per core (`--cpu-bind=threads`) and never give a rank both
  simultaneous-multithreading siblings of one core. One machine ran about
  4300 times slower with both siblings in a rank's mask at the same core
  count, and the cause was not found.
- Pass `--ntasks-per-node` below full packing. `-N` alone lets the scheduler
  block-pack one node, and the result is a plausible timing with no error.
- Read the allocated CPU mask rather than the scheduler flags, since `-c`
  counts logical CPUs on some machines and physical cores on others.
  `probes/clusterprobe.jl` prints the mask, the node distribution and the MPI
  binary in use, and `probes/clusterlaunch.jl` sizes a launch for a grid.

## Respect collective ordering

Directional compact derivatives are collective along their MPI
subcommunicator. Every rank must enter them in the same order. In particular:

- do not return from boundary code before collective derivatives have run;
- switch a [`SwitchableBC`](@ref) from a globally consistent callback; and
- call collective diagnostics on every rank, even if only rank zero prints.

A collective-ordering error usually appears as a zero-CPU hang, not an
exception.

## Record enough information to interpret timing

Report Julia version, MPI implementation, rank and thread counts, process grid,
physical core topology, binding policy, global grid, and local extents.
Single-run differences of a few percent are not resolved reliably on typical
workstations; repeat complete processes and compare distributions.
