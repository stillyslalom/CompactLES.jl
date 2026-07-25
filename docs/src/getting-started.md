# Getting started

The frontend separates the physical problem from numerical and parallel
choices. A [`Problem`](@ref) contains the domain, boundary conditions,
pointwise initial condition, and current model selections. A [`Numerics`](@ref)
contains the grid, compact schemes, CFL, filtering cadence, and process grid.

## A minimal periodic problem

Start Julia with the project environment and initialize MPI. `setup` also
initializes MPI when necessary, but explicit initialization makes the requested
thread level clear in applications.

```@example quickstart
using MPI
MPI.Init(threadlevel=:funneled)

using CompactLES

problem = Problem(
    name = "one-dimensional wave",
    domain = ((0.0, 2π), (0.0, 1.0), (0.0, 1.0)),
    bcs = ntuple(_ -> (PeriodicBC(), PeriodicBC()), 3),
    ic = (x, y, z) -> Prim(
        u = (0.05sin(x), 0.0, 0.0),
        p = 1.0,
        rho = 1.0,
    ),
)

numerics = Numerics(
    n_global = (64, 1, 1),
    deriv = lele_d1_6(),
    filt = compact_filter(0.45),
    cfl = 0.5,
)

solver, Q = setup(problem, numerics)
run!(solver, Q; tfinal=0.01)
nothing # hide
```

A dimension with one global point is collapsed: it has no derivative, halo
exchange, or decomposition. This makes the same three-dimensional frontend
usable for one- and two-dimensional cases.

## Reuse and observe the run

Retain a [`Workspace`](@ref) when advancing in several calls:

```@example quickstart
workspace = Workspace(Q)
run!(solver, Q; workspace, tfinal=0.02)
run!(solver, Q; workspace, tfinal=0.03)
nothing # hide
```

The callback runs after every complete step:

```@example quickstart
run!(solver, Q; workspace, tfinal=0.04,
     callback=(solver, Q) -> begin
         solver.step % 100 == 0 || return
         @show solver.t dt_report(solver, Q)
     end)
nothing # hide
```

[`compute_dt`](@ref) applies the configured CFL to acoustic, diffusive, and
collapsed-coordinate curvature rates. [`dt_report`](@ref) identifies the
global limiting location and rate class when a timestep is unexpectedly small.

## Run with MPI

Ask MPI.jl for the launcher configured for the current environment:

```julia
using MPI
MPI.mpiexec() do mpiexec
    run(`$mpiexec -n 4 $(Base.julia_cmd()) --project=. examples/taylor_green.jl`)
end
```

See [Parallel decomposition](@ref) for process-grid constraints.
