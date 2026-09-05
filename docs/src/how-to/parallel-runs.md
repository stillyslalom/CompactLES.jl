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

On a cluster, use physical cores and verify binding. A rank must not receive
both simultaneous-multithreading siblings of one core while another physical
core is idle. The repository's
[`reference/CLUSTER.md`](https://github.com/stillyslalom/CompactLES.jl/blob/main/reference/CLUSTER.md)
records machine-specific configuration and measured launch rules.

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
