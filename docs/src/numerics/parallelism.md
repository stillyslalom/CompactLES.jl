# Parallel decomposition

CompactLES uses a three-dimensional Cartesian MPI process grid. Each rank owns
a rectangular interior block plus halos. Shared-memory threads operate within
that block, while MPI exchanges halos and couples the compact line solves.

## Process-grid rules

`Numerics(dims=nothing)` asks MPI to distribute ranks over active dimensions.
Set `dims=(p₁,p₂,p₃)` to control the layout explicitly. The following
invariants are enforced:

- `prod(dims)` equals the MPI communicator size;
- collapsed dimensions (`n_global[d] == 1`) have `dims[d] == 1`; and
- each local extent satisfies the selected scheme's minimum size.

With the default filter, a useful preflight bound is
`n_global[d] ≥ 9dims[d]` for each decomposed direction.

## Why the compact solve is collective

A compact derivative is globally coupled along each grid line. Splitting that
line across ranks does not turn it into independent local derivatives.
CompactLES factors the local banded blocks, forms a reduced interface system,
solves that system on the one-dimensional subcommunicator, and back-substitutes
locally.

Consequently, every rank in a directional subcommunicator must enter an
operator application. Boundary code must not return early on ranks that do not
own the physical face until all collective derivative calls are complete.

## Threads

Line and point loops use shared-memory threads only above
[`THREAD_MIN_WORK`](@ref), which defaults to 32768 units of work and can be
overridden with `CL_THREAD_MIN_WORK`. This avoids task-launch overhead on small
local blocks. MPI is initialized with the `:funneled` thread level; MPI calls
remain outside threaded regions.

## Launcher portability

Do not hard-code `mpiexec`. MPI.jl selects either its artifact-provided MPI or
the configured system implementation:

```julia
using MPI
MPI.mpiexec() do mpiexec
    run(`$mpiexec -n 8 $(Base.julia_cmd()) --project=. -t 2 your_case.jl`)
end
```

Scaling results depend on Julia thread placement, physical core topology, and
the MPI implementation. Record those alongside timings.
