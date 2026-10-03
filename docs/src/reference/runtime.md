# Runtime and output

```@meta
CurrentModule = CompactLES
```

## Right-hand side and integration

```@docs
Solver
StepControl
SolverFailure
FloorTally
Workspace
ConservedState
compute_rhs!
CompactLES.compute_primitives_and_gradients!
apply_bcs!
compute_dt
dt_report
max_rate
step!
run!
CompactLES.positivity_floors
CompactLES.apply_positivity_floor!
CompactLES.positivity_counts
filter_state!
CompactLES.filter_weight
mpi_main
```

## Implicit conduction

`Numerics(implicit = ImplicitConduction())` advances the molecular heat
conduction implicitly and every other term explicitly, by an additive
Runge–Kutta pair, so that the conductive rate no longer limits the step.
[`step!`](@ref) and [`run!`](@ref) take the pair in place of the default
integrator, and a stage that does not converge is a [`SolverFailure`](@ref)
that `StepControl(retries = ...)` recovers.

```@docs
ImplicitConduction
```

## State validity

`StepControl(validity = ...)` decides what happens to a state the validation
rejects. [`setup`](@ref) validates the initial state; a [`StateGuard`](@ref)
validates each accepted state during a run, including the one the run returns.
Whether a given internal energy is admissible is asked of the equation of state
through [`state_admissibility`](@ref), because the gauge and the domain belong
to the model.

```@docs
StateReport
state_report
state_valid
validate_state!
StateGuard
state_guard
CompactLES.check_validity
CompactLES.check_step
```

## Step callbacks

Triggers fire only between completed steps and produce the same verdict on every
rank. These requirements preserve collective ordering; the implementation note
at the top of `src/callbacks.jl` gives the details. A phase change,
[`setup`](@ref)`(solver, Q; bcs)`, rests on the same agreement: a
[`Callback`](@ref) whose effect returns `true` ends the run on the same step
on every rank, and the next phase's conditions start there everywhere.

For [`AtTime`](@ref) and [`EveryTime`](@ref), `run!` shortens the preceding
`StepControl(landing_steps = ...)` steps to end a step at the scheduled
instant. `EveryTime` produces a uniform time axis for periodic output.

```@docs
Trigger
AtTime
EveryTime
EveryStep
WhenState
Callback
ProgressLog
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
restores onto any rank count, not only onto the decomposition that wrote it.
All three live in a package extension and require `using HDF5`;
`hdf5_available` and `hdf5_parallel` report whether the extension is loaded and
whether its libhdf5 supports MPI-parallel writes.
`FieldWriter(prefix; format = :hdf5)` writes a time series of `save_hdf5`
dumps with an XDMF temporal collection `prefix.xmf` in place of the `.pvd`, and
also requires `using HDF5`.

A refined solver's state is a vector of per-patch arrays, and every writer
and checkpoint above except the HDF5 field dumps takes it: `save_vtk` and
`FieldWriter` then write one piece per patch under a `.vtm` multiblock index
with the covered coarse nodes blanked, and the checkpoints record the
hierarchy (the tile layout, the stored ownership and the tag history) beside
every tile's state, so a restart rebuilds the level on whatever rank count it
is given.

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
CompactLES.container_extension
FieldWriter
```

## Developer internals: script argument helpers

Repository examples and benchmark drivers use `script_args` and `script_grid`
to build a defaults `NamedTuple` command-line schema. They are not part of the
input-deck or runtime API and are documented only for source cross-references.

```@docs
CompactLES.script_args
CompactLES.script_grid
```
