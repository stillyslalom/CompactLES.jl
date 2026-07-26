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

Triggers fire only between completed steps, and every one is globally consistent
across ranks — see the note at the top of `src/callbacks.jl` for why both of
those are structural rather than incidental. [`SwitchableBC`](@ref) is documented
here rather than with the other boundary conditions because it exists to be
driven by a [`Callback`](@ref), and is unsafe driven any other way.

```@docs
Trigger
AtTime
EveryStep
WhenState
Callback
SwitchableBC
switch!
```

## Checkpoint and visualization output

```@docs
save_checkpoint
load_checkpoint!
save_vtk
```
