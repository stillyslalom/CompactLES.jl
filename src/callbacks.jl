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
# so `AtTime` and `EveryStep` are globally consistent for free. `WhenState` is
# the only one that reads the field, and it reduces its own condition across the
# communicator before answering — so a user predicate that happens to be true on
# one rank alone still fires everywhere, and cannot wedge the run.

"""
    Trigger

Decides *when* a [`Callback`](@ref) fires. Implementations are [`AtTime`](@ref),
[`EveryStep`](@ref), and [`WhenState`](@ref).

A trigger implements `fired!(trigger, solver, Q) -> Bool`, called after each
completed step, and may optionally implement `next_time(trigger, solver)` to ask
`run!` to clip `dt` so a step lands exactly on a scheduled instant. Whatever it
does, it must answer identically on every rank — see the note at the top of this
file.
"""
abstract type Trigger end

# Landing exactly on a requested time is the one thing a caller cannot arrange
# from outside: it needs `dt` clipped before the step, which only `run!` can do.
# Everything else reports Inf and lets the timestep alone.
next_time(::Trigger, solver) = Inf

"""
    AtTime(t)
    AtTime([t1, t2, ...])

Fire once at each listed time. `run!` clips `dt` so a step lands exactly on the
next one rather than overshooting it, which is what makes this usable for
switching behaviour at a prescribed instant. Times are visited in order and each
fires once; one already behind `solver.t` fires on the next completed step.
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
    solver.t >= target - 8eps(max(target, one(target))) || return false
    trigger.next += 1
    return true
end

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

The condition is evaluated per rank and then reduced across the communicator, so
it may legitimately be true on only the rank owning a boundary plane. **Do not
reduce inside the condition**; that would be a second collective and the
framework already did the first one. Reading a plane of a field and returning a
local verdict is the intended shape.
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

Pass one to `run!` as `callback=`, or several as a tuple. A bare function is
still accepted there and runs every step, so existing callers are unaffected.
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

A ready-made progress line, returned as a [`Callback`](@ref) to hand straight to
`run!`. Reports step, `t`, `dt`, and wall time per step from `solver.wall_step`.

`quantity(solver, Q) -> Real` adds one scalar column, `label` names it. It is
called on **every** rank and may reduce (`volume_integral` and friends do), so
write it as a normal collective diagnostic and let this handle the rank guard.

`tfinal` adds percent-complete and a projected remaining time, from the mean
step so far rather than the last one. `imbalance = true` adds the min/max spread
of step time across ranks, at the cost of two extra reductions per report — the
cheap way to see a decomposition load-imbalance without a profiler.

Output is flushed on every line. On a cluster stdout is block-buffered through
the launcher, and an unflushed run is indistinguishable from a hung one.

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
