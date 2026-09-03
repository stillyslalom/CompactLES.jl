# Write output and restart

## Write a visualization frame

```julia
save_vtk(
    solver,
    Q,
    "output/flow";
    fields = (:rho, :velocity, :pressure, :schlieren),
    stride = 2,
)
```

Open the resulting `.pvtr` or `.pvts` container in ParaView or VisIt. Each MPI
rank writes one piece. Resolved cylindrical and spherical angles produce a
curvilinear grid with positions and velocity vectors rotated into Cartesian
components.

Useful derived fields include `:vorticity`, `:qcriterion`, `:divergence`,
`:mach`, `:schlieren`, `:sensor`, and the artificial-property fields. See
[`DEFAULT_VTK_FIELDS`](@ref) and [`save_vtk`](@ref) for the full list.

## Reduce output volume

`stride` selects points on a common global lattice:

```julia
save_vtk(solver, Q, "coarse"; stride = (4, 2, 1))
```

Write one global-index plane with `slice`:

```julia
save_vtk(solver, Q, "midplane"; slice = (3, 128), stride = 2)
```

Slicing usually saves more space than striding for a three-dimensional time
series. Field construction still occurs on the local block before sampling, so
these options reduce I/O volume and leave RHS cost unchanged.

## Write a time series

[`FieldWriter`](@ref) numbers frames and maintains a `.pvd` collection with
their physical times:

```julia
writer = FieldWriter(
    "output/frame";
    fields = (:rho, :pressure, :velocity),
    slice = (3, 128),
)

output = Callback(EveryTime(0.01), writer)
run!(solver, Q; tfinal = 1.0, callback = output)
```

Because `EveryTime` lands steps exactly, the collection has a uniform physical
time axis.

## Write a refined run

A refined (or patch-partitioned) solver holds its state as the vector
[`allocate_state`](@ref) returns. `save_vtk` and [`FieldWriter`](@ref) take
that vector and write one `.vtr` piece per patch under a `.vtm` multiblock
index; open the `.vtm`. Coarse nodes a finer level covers are blanked, so
the composite shows the finest data everywhere. `slice` names a plane of the
root lattice, which each patch maps to its own coincident node:

```julia
save_vtk(solver, states, "output/amr"; fields = (:rho, :schlieren), slice = (3, 64))
```

## Restart with the same decomposition

The dependency-free checkpoint path writes one file per rank:

```julia
save_checkpoint(solver, Q, "restart/state")
load_checkpoint!(solver, Q, "restart/state")
```

The restarting solver must have the same global grid and process grid. The
checkpoint carries the artificial coefficient arrays beside the state, so
the restarted run takes the same first step as the uninterrupted one and
continues it bit for bit.

## Restart a refined run

Both checkpoint forms take the state vector and record the hierarchy: the
tile layout and stored ownership of every level, the tag history a
regridding run needs, and every tile's state.

```julia
save_checkpoint_hdf5(solver, states, "restart/amr")

# Later, in a fresh process at any rank count: build the solver as before
# (any initial `refine` region, the same `tile`), then load.
solver = Solver(; refine = initial, tile = 8, regrid_interval = 5, kwargs...)
states = allocate_state(solver)
load_checkpoint_hdf5!(solver, states, "restart/amr")
run!(solver, states; tfinal = 2.0)
```

The load rebuilds the recorded tile layout in place of the initial one and
resizes `states` with it, so build any [`Workspace`](@ref) after the load.
On the rank count that wrote the file the stored ownership is restored and
the run continues bit for bit, later regrids included; on another rank
count the level is partitioned afresh and the continuation agrees to
round-off. A hierarchy deeper than two levels is static and must be built
with the recorded `refine` regions. The per-rank `save_checkpoint` form
restores onto the rank count that wrote it only.

## Restart on another rank count

Load HDF5 before CompactLES uses the extension:

```julia
using HDF5
using CompactLES

save_checkpoint_hdf5(solver, Q, "restart/global")
load_checkpoint_hdf5!(solver, Q, "restart/global")
```

The shared checkpoint stores one global state and can be restored onto a
different decomposition. [`hdf5_parallel`](@ref) reports whether the loaded
HDF5 library can write collectively through the run's MPI implementation;
otherwise rank zero gathers and writes.

For shared visualization output, [`save_hdf5`](@ref) writes one `.h5` frame and
an XDMF sidecar. This avoids one file per rank per frame at large process
counts.
