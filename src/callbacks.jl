# Step-boundary callbacks: a trigger deciding *when*, an effect deciding *what*.
#
# The split is DiffEqCallbacks' `condition`/`affect!` design, deliberately
# without its continuous (root-found) callbacks. Two reasons, both structural
# rather than a matter of effort:
#
#   - The low-storage RK45 keeps no dense output between step boundaries, so
#     there is nothing to root-find a crossing against without re-integrating.
#   - A trigger that fired mid-step would break two invariants the solver rests
#     on. The five RK stages of one step must agree on the boundary conditions,
#     or the integration is inconsistent; and every rank must agree on the
#     decision, or a boundary routine that only some ranks think is active
#     deadlocks on its collectives (see the collective note in `nscbc.jl`).
#
# Firing only between completed steps satisfies both, and costs nothing that
# matters: the physical events these exist to catch — a wave nearing a boundary,
# an interaction finishing — are not resolved to better than a timestep anyway.
#
# Rank agreement is the trap worth designing out rather than documenting. `t`
# and `step` advance identically on every rank (`dt` comes out of an Allreduce),
# so `AtTime`, `EveryTime` and `EveryStep` are globally consistent for free.
# `WhenState` is the only one that reads the field, and it reduces its own
# condition across the communicator before answering — so a user predicate that
# happens to be true on one rank alone still fires everywhere, and cannot wedge
# the run.

"""
    Trigger

Decides *when* a [`Callback`](@ref) fires. Implementations are [`AtTime`](@ref),
[`EveryTime`](@ref), [`EveryStep`](@ref), and [`WhenState`](@ref).

A trigger implements `fired!(trigger, solver, Q) -> Bool`, which is called after
each completed step. It may also implement `next_time(trigger, solver)` so that
`run!` can end a step at a scheduled instant, and `rewind!(trigger, t, step)` to
restore its schedule after a `StepControl` rollback. The verdict must be
identical on every rank; see the implementation note at the top of this file.
"""
abstract type Trigger end

# Landing exactly on a requested time is the one thing a caller cannot arrange
# from outside: it needs `dt` clipped before the step, which only `run!` can do.
# Everything else reports Inf and lets the timestep alone.
next_time(::Trigger, solver) = Inf

"""
    rewind!(trigger, t, step)

Restore a trigger's schedule to `(t, step)`. [`run!`](@ref) calls this method
when `StepControl` restores a savepoint. Scheduled instants between the
savepoint and the failed state are then visited on the replacement trajectory.

Only the schedule is restored; callback effects are not reversed. Repeatable
effects, such as an output file, can be evaluated again at the same instant.
Non-repeatable effects cannot be reversed by the trigger. Consequently,
[`WhenState`](@ref) remains in its existing armed state: a `SwitchableBC` that
has already switched cannot be restored to its earlier condition.

The default is a no-op, which is correct for any trigger reading only `step`.
"""
rewind!(::Trigger, t, step) = nothing

"""
    AtTime(t)
    AtTime([t1, t2, ...])

Fire once at each listed time. `run!` shortens `dt` so that a step ends at the
next scheduled instant. Times are visited in order and each fires once; a time
already behind `solver.t` fires on the next completed step.
"""
mutable struct AtTime <: Trigger
    times::Vector{Float64}
    next::Int
end

AtTime(times::AbstractVector) = AtTime(sort(collect(Float64, times)), 1)
AtTime(t::Real) = AtTime([Float64(t)], 1)

next_time(trigger::AtTime, solver) =
    trigger.next > length(trigger.times) ? Inf : trigger.times[trigger.next]

function fired!(trigger::AtTime, solver, Q)
    target = next_time(trigger, solver)
    isfinite(target) || return false
    # dt was clipped to land on `target`, so equality is the expected case; the
    # tolerance only absorbs the rounding in `solver.t += dt`. It is orders of
    # magnitude below any dt, so this cannot fire a step early.
    solver.t >= target - _land_tol(target) || return false
    trigger.next += 1
    return true
end

function rewind!(trigger::AtTime, t, step)
    idx = findfirst(target -> target > t + _land_tol(target), trigger.times)
    trigger.next = idx === nothing ? length(trigger.times) + 1 : idx
    return nothing
end

# The landing tolerance shared by both time triggers, defined once so that the
# schedule restored by `rewind!` cannot disagree with the one `fired!` reads.
_land_tol(t) = 8eps(max(abs(t), one(t)))

"""
    EveryTime(interval; start = 0.0)

Fire at `start + n*interval` for every integer `n`, with `run!` ending a step at
each instant. This is the unbounded counterpart to [`AtTime`](@ref) and produces
an evenly spaced output schedule. Unlike a materialized
`AtTime(start:interval:tfinal)`, it requires no advance `tfinal` and can resume
from a restart. The next instant is computed from `solver.t` when the trigger is
first consulted; a run resumed at t = 0.55 with `interval = 0.1` therefore fires
next at 0.6.

Globally consistent without a reduction, since `t` is the same on every rank.
"""
mutable struct EveryTime <: Trigger
    interval::Float64
    start::Float64
    next::Float64      # NaN until the first query anchors it to solver.t
end

function EveryTime(interval::Real; start::Real=0.0)
    interval > 0 || throw(ArgumentError("EveryTime interval must be positive"))
    return EveryTime(Float64(interval), Float64(start), NaN)
end

# Instants are computed as start + n*interval rather than by accumulating the
# interval, so that a long run's schedule does not drift with the rounding in t.
function _instant_after(trigger::EveryTime, t)
    n = max(floor((t - trigger.start) / trigger.interval) + 1, 0.0)
    inst = trigger.start + n * trigger.interval
    # floor is exact, the division is not. An instant landing on or behind t
    # would fire immediately and then again on the next step.
    while inst <= t + _land_tol(t)
        n += 1
        inst = trigger.start + n * trigger.interval
    end
    return inst
end

function next_time(trigger::EveryTime, solver)
    isnan(trigger.next) && (trigger.next = _instant_after(trigger, solver.t))
    return trigger.next
end

function fired!(trigger::EveryTime, solver, Q)
    target = next_time(trigger, solver)
    solver.t >= target - _land_tol(target) || return false
    trigger.next = _instant_after(trigger, solver.t)
    return true
end

# Re-anchoring on the next query is sufficient, since the schedule is a function
# of t alone.
rewind!(trigger::EveryTime, t, step) = (trigger.next = NaN; nothing)

"""
    EveryStep(interval = 1)

Fire every `interval` completed steps. Globally consistent without a reduction,
since `solver.step` is the same on every rank.
"""
struct EveryStep <: Trigger
    interval::Int
    function EveryStep(interval::Int=1)
        interval >= 1 || throw(ArgumentError("EveryStep interval must be >= 1"))
        new(interval)
    end
end

fired!(trigger::EveryStep, solver, Q) = solver.step % trigger.interval == 0

"""
    WhenState(condition; once = true)

Fire when `condition(solver, Q)` first returns `true`, or on every step it does
when `once = false`.

The condition is evaluated on each rank and reduced across the communicator, so
it may be true only on the rank that owns a boundary plane. The condition must
not perform this reduction itself because the framework already supplies the
collective. A condition may read a local field plane and return a local verdict.
"""
mutable struct WhenState{F} <: Trigger
    condition::F
    once::Bool
    done::Bool
end

WhenState(condition; once::Bool=true) = WhenState(condition, once, false)

function fired!(trigger::WhenState, solver, Q)
    trigger.done && return false
    # Reduced as an Int with `max` rather than a Bool with `|`, matching the
    # reductions elsewhere in the solver; semantically this is a logical OR.
    local_hit = trigger.condition(solver, Q)::Bool
    hit = MPI.Allreduce(Int(local_hit), max, solver.decomp.comm) > 0
    hit && trigger.once && (trigger.done = true)
    return hit
end

"""
    Callback(trigger, effect!)

Pair a [`Trigger`](@ref) with `effect!(solver, Q)`, run after a completed step
(and after any filtering) when the trigger fires. Returning `true` from
`effect!` asks `run!` to stop; any other value continues.

Pass one to `run!` as `callback=`, or pass several as a tuple. A bare function
is also accepted and runs after every step.
"""
struct Callback{Tr<:Trigger,F}
    trigger::Tr
    effect!::F
end

# --- Dispatch over what `run!` was handed: nothing, a bare callable, one
# Callback, or a tuple of them. Tuples recurse rather than iterate so the whole
# thing stays inferable with mixed trigger types, the same reason `sources` is a
# tuple (see sources.jl).

callback_next_time(::Nothing, solver) = Inf
callback_next_time(cb::Callback, solver) = next_time(cb.trigger, solver)
callback_next_time(::Tuple{}, solver) = Inf
callback_next_time(cbs::Tuple, solver) =
    min(callback_next_time(first(cbs), solver),
        callback_next_time(Base.tail(cbs), solver))
# Bare callables carry no schedule. Less specific than every method above.
callback_next_time(_, solver) = Inf

rewind_callbacks!(::Nothing, t, step) = nothing
rewind_callbacks!(::Tuple{}, t, step) = nothing
rewind_callbacks!(cb::Callback, t, step) = rewind!(cb.trigger, t, step)
rewind_callbacks!(cbs::Tuple, t, step) =
    (rewind_callbacks!(first(cbs), t, step);
     rewind_callbacks!(Base.tail(cbs), t, step))
rewind_callbacks!(_, t, step) = nothing

run_callbacks!(::Nothing, solver, Q) = false
run_callbacks!(::Tuple{}, solver, Q) = false

function run_callbacks!(cb::Callback, solver, Q)
    fired!(cb.trigger, solver, Q) || return false
    return cb.effect!(solver, Q) === true
end

function run_callbacks!(cbs::Tuple, solver, Q)
    # Both sides run: an early return would let the first stop request skip the
    # rest, and a diagnostic callback silently missing its last step is exactly
    # the kind of bug that gets blamed on the physics.
    stop_first = run_callbacks!(first(cbs), solver, Q)
    stop_rest = run_callbacks!(Base.tail(cbs), solver, Q)
    return stop_first || stop_rest
end

# A bare callable keeps its old contract exactly: called every step, return value
# ignored. Honouring `=== true` here would silently turn any existing callback
# whose last expression happens to be a comparison into an early stop.
run_callbacks!(cb, solver, Q) = (cb(solver, Q); false)

# --- Progress reporting.
#
# Every example grew its own copy of this loop, and every copy got at least one
# of the three cluster details wrong: printing from all ranks instead of one,
# forgetting to flush (srun buffers stdout, so a run appears hung for minutes),
# and timing the diagnostic's own reduction as if it were solver cost. The
# collective-ordering rule from the top of this file applies too — `quantity`
# runs on every rank, and only the printing is guarded.

struct ProgressEffect{Q,I}
    quantity::Q
    label::String
    io::I
    tfinal::Float64
    imbalance::Bool
end

"""
    ProgressLog(; every = 10, quantity = nothing, label = "", tfinal = Inf,
                  imbalance = false, io = stdout)

Construct a [`Callback`](@ref) that reports the step, `t`, `dt`, and wall time
per step from `solver.wall_step`.

`quantity(solver, Q) -> Real` adds one scalar column, named by `label`. It is
called on every rank and may reduce, as `volume_integral` and related diagnostics
do. Only output is restricted to rank 0.

`tfinal` adds percent-complete and a projected remaining time, from the mean
step so far rather than the last one. `imbalance = true` adds the min/max spread
of step time across ranks at the cost of two additional reductions per report.

Output is flushed on every line because launchers commonly block-buffer cluster
stdout; flushing preserves progress visibility during a run.

```julia
run!(solver, Q; tfinal = 1.0,
     callback = ProgressLog(every = 10, tfinal = 1.0, label = "TKE",
                            quantity = turbulent_kinetic_energy))
```
"""
function ProgressLog(; every::Int=10, quantity=nothing, label::AbstractString="",
                     tfinal::Real=Inf, imbalance::Bool=false, io=stdout)
    return Callback(EveryStep(every),
                    ProgressEffect(quantity, String(label), io,
                                   Float64(tfinal), imbalance))
end

function (effect::ProgressEffect)(solver, Q)
    comm = solver.decomp.comm
    # Both collectives are unconditional: every rank reaches them, before the
    # rank-0 guard below. Reversing that order is the deadlock in `nscbc.jl`.
    value = effect.quantity === nothing ? nothing : effect.quantity(solver, Q)
    lo, hi = if effect.imbalance
        (MPI.Allreduce(solver.wall_step, min, comm),
         MPI.Allreduce(solver.wall_step, max, comm))
    else
        (0.0, 0.0)
    end
    MPI.Comm_rank(comm) == 0 || return false

    Printf.@printf(effect.io, "step %7d  t = %11.5e  dt = %10.4e  %8.4g s/step",
                   solver.step, solver.t, solver.dt_prev, solver.wall_step)
    if isfinite(effect.tfinal) && effect.tfinal > 0
        frac = clamp(solver.t / effect.tfinal, 0.0, 1.0)
        # Mean step so far, not the last one: dt varies with the CFL rate and a
        # single sample projects wildly at a startup transient or after a retry.
        mean_step = solver.step > 0 ? solver.wall_total / solver.step : 0.0
        remaining = frac > 0 ? mean_step * solver.step * (1 - frac) / frac : NaN
        Printf.@printf(effect.io, "  %5.1f%%  eta %s", 100 * frac,
                       _duration(remaining))
    end
    effect.imbalance &&
        Printf.@printf(effect.io, "  spread %.3g-%.3g s", lo, hi)
    if value !== nothing
        isempty(effect.label) ? Printf.@printf(effect.io, "  %.8e", value) :
            Printf.@printf(effect.io, "  %s = %.8e", effect.label, value)
    end
    println(effect.io)
    # Without this a run under srun can sit silent for minutes and look hung.
    flush(effect.io)
    return false
end

function _duration(seconds::Float64)
    isfinite(seconds) || return "  --  "
    seconds < 90 && return Printf.@sprintf("%4.0fs", seconds)
    seconds < 5400 && return Printf.@sprintf("%4.1fm", seconds / 60)
    return Printf.@sprintf("%4.1fh", seconds / 3600)
end
