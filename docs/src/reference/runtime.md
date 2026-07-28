# Runtime and output

```@meta
CurrentModule = CompactLES
```

## Right-hand side and integration

```@docs
StepControl
SolverFailure
PLANCK_TIME
Workspace
compute_rhs!
apply_bcs!
compute_dt
dt_report
step!
run!
filter_state!
```

## Step callbacks and switchable boundaries

Triggers fire only between completed steps and produce the same verdict on every
rank. These requirements preserve collective ordering; the implementation note
at the top of `src/callbacks.jl` gives the details. [`SwitchableBC`](@ref)
appears here because it must be driven by a [`Callback`](@ref).

For [`AtTime`](@ref) and [`EveryTime`](@ref), `run!` shortens the preceding
`StepControl(landing_steps = ...)` steps so that a step ends at the scheduled
instant. `EveryTime` produces a uniform time axis for periodic output.

```@docs
Trigger
AtTime
EveryTime
EveryStep
WhenState
Callback
rewind!
SwitchableBC
switch!
```

## Reading the state between steps

Between steps, the conserved array `Q` is current, whereas the primitive fields
on the solver correspond to the input state of the last RK stage. The following
functions read `Q` without depending on its component layout.
[`refresh_primitives!`](@ref) updates the primitive fields before a callback or
custom diagnostic reads them.

```@docs
refresh_primitives!
mixture_density
velocity
total_energy
mass_fraction
boundary_plane
```

## Checkpoint and visualization output

[`FieldWriter`](@ref) pairs with a trigger, numbers its frames, and writes a
`.pvd` collection recording the physical time of each frame. `save_vtk` provides
an individual field dump.

Both select what to write through `fields` and subsample through `stride`. The
metric determines the grid type: a resolved angular dimension produces a
curvilinear `.pvts` with explicit Cartesian positions and rotated vectors, and
every other grid produces a rectilinear `.pvtr`. Strided points are selected on
the global index, so the per-rank pieces sample one common lattice and still
tile the coarse grid.

`save_hdf5` writes a field dump as one shared file with an XDMF3 sidecar, which
is the form to use at large rank counts: the VTK path writes one file per rank
per frame, this one file per frame. `save_checkpoint_hdf5` and
`load_checkpoint_hdf5!` store the state as one global array, so a checkpoint
restores onto any rank count rather than onto the decomposition that wrote it.
All three live in a package extension and require `using HDF5`;
`hdf5_available` and `hdf5_parallel` report whether the extension is loaded and
whether its libhdf5 supports MPI-parallel writes.

```@docs
save_checkpoint
load_checkpoint!
save_hdf5
save_checkpoint_hdf5
load_checkpoint_hdf5!
hdf5_available
hdf5_parallel
BlockRegion
save_vtk
DEFAULT_VTK_FIELDS
container_extension
FieldWriter
```
