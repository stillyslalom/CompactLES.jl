# Control and diagnose a run

## Retain stage storage

[`Workspace`](@ref) owns the two arrays used by the low-storage Runge--Kutta
scheme. Retain one when advancing in several calls:

```julia
workspace = Workspace(Q)
run!(solver, Q; workspace, tfinal = 0.1)
run!(solver, Q; workspace, tfinal = 0.2)
```

`tfinal` is an absolute simulation time, not an interval measured from the
current state.

## Report progress

[`ProgressLog`](@ref) is MPI-aware and flushes every line for batch logs:

```julia
progress = ProgressLog(
    every = 20,
    tfinal = 1.0,
    label = "TKE",
    quantity = turbulent_kinetic_energy,
)

run!(solver, Q; tfinal = 1.0, callback = progress)
```

The diagnostic runs on every rank and may perform collective reductions; only
printing is restricted to rank zero.

## Schedule an action in physical time

Use `AtTime` for selected instants and `EveryTime` for a uniform time series:

```julia
checkpoint = Callback(AtTime([0.25, 0.5]),
                      (solver, Q) -> save_checkpoint(solver, Q, "state"))

samples = Callback(EveryTime(0.01), sample!)
run!(solver, Q; tfinal = 1.0, callback = (checkpoint, samples))
```

`run!` shortens preceding steps so they land exactly on time triggers. Use
`EveryStep(n)` for a step-based cadence and `WhenState(predicate)` for an event
defined by the solution. `WhenState` reduces the Boolean decision across the
communicator.

## Diagnose a small timestep

```julia
report = dt_report(solver, Q)
```

The result reports `dt`, the owning rank, local index, physical coordinates,
direction, and limiting mechanism. The mechanism is one of:

- `:acoustic`: advection and sound propagation;
- `:diffusive`: molecular or artificial transport; or
- `:curvature`: a collapsed-coordinate geometric source.

A diffusive limit localized at a shock can be physical for the configured
artificial properties. A limit near a resolved polar axis often reflects the
small physical spacing ``r\Delta\theta``.

## Bound failure and recovery

Every production and parameter-sweep call should set `nmax`. A failed state can
otherwise reduce `dt` repeatedly while making negligible progress.

```julia
control = StepControl(
    retries = 4,
    cfl_backoff = 0.5,
    floor_ratio = 1e-8,
)

numerics = Numerics(n_global = (256, 1, 1), control = control)
run!(solver, Q; tfinal = 0.5, nmax = 100_000)
```

On a recoverable failure, `run!` restores its savepoint, reduces the CFL, and
retries. An exhausted retry budget or timestep floor raises
[`SolverFailure`](@ref) with the step, time, timestep, CFL, and reason.

## Choose what an invalid state does

`run!` sizes and checks each step from the state that enters it, so the state it
*returns* is inspected by nothing: an `nmax`, `tfinal`, or callback exit leaves
the last result unchecked. [`state_guard`](@ref) closes that by validating every
accepted state, including the final one.

```julia
control = StepControl(validity = :permissive)
run!(solver, Q; tfinal = 0.5, nmax = 100_000,
     callback = state_guard(solver, Q; control = control))
```

`StepControl(validity = ...)` decides what a rejected state means.

| Mode | Effect |
|---|---|
| `:strict` (default) | raise [`SolverFailure`](@ref)`(:invalid_state)` |
| `:permissive` | accept the state and report what it contains |
| `:repair` | repair through the positivity failsafe, report the substitutions, reject the rest |

`:repair` also needs a positive `floor_ratio`, which is where its floors come
from. A guard raises from inside a callback, so a strict rejection is not
rolled back; the retryable checks remain the ones `run!` applies before a step.

Whether a given internal energy is admissible is asked of the equation of state,
not fixed by the integrator, because the gauge that places its zero belongs to
the model. Converging-shock runs integrate through cells an ideal gas calls
inadmissible and still reach the correct answer, so a guard on such a run wants
`:permissive`; `reference/CALIBRATION.md` records the budget. [`setup`](@ref)
applies the same policy to the initial state.

## Read the completed state

Between steps, `Q` is current. Cached primitive arrays on `solver` still
correspond to the input of the final RK stage. Use state-query helpers such as
[`mixture_density`](@ref), [`velocity`](@ref), and [`mass_fraction`](@ref), or
call [`refresh_primitives!`](@ref) before reading `solver.rho`, `solver.p`, and
related arrays.
