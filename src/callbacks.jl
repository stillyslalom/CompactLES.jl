# Step-boundary callbacks: a trigger deciding *when*, an effect deciding *what*.
#
# The split is DiffEqCallbacks' `condition`/`affect!` design without its
# continuous (root-found) callbacks. The omission is structural, for two
# reasons:
#
#   - The low-storage RK45 keeps no dense output between step boundaries, so
#     there is nothing to root-find a crossing against without re-integrating.
#   - A trigger that fired mid-step would break two invariants the solver rests
#     on. The five RK stages of one step must agree on the boundary conditions,
#     or the integration is inconsistent; and every rank must agree on the
#     decision, or a boundary routine that only some ranks think is active
#     deadlocks on its collectives (see the collective note in `nscbc.jl`).
#
# Firing only between completed steps preserves both invariants. The trigger
# time is resolved to one timestep, consistent with the available RK state.
#
# `t` and `step` advance identically on every rank because `dt` comes from an
# `Allreduce`. `AtTime`, `EveryTime`, and `EveryStep` are therefore globally
# consistent without another reduction. `WhenState` is the only trigger that
# reads the field. It reduces its condition across the communicator, so a
# predicate that is true on one rank fires on every rank and preserves
# collective ordering.

"""
    Trigger

Controls *when* a [`Callback`](@ref) fires. Implementations are [`AtTime`](@ref),
[`EveryTime`](@ref), [`EveryStep`](@ref), and [`WhenState`](@ref).

A trigger implements `fired!(trigger, solver, Q) -> Bool`, which is called after
each completed step. It may also implement `next_time(trigger, solver)`, allowing
`run!` to end a step at a scheduled instant, and `rewind!(trigger, t, step)` to
restore its schedule after a `StepControl` rollback. The verdict must be
identical on every rank; see the implementation note at the top of this file.
"""
abstract type Trigger end

"""
    next_time(trigger, solver) -> Real

Next physical time at which `trigger` wants [`run!`](@ref) to end a completed
step exactly. The default is `Inf`, so a custom [`Trigger`](@ref) that is not
scheduled in physical time need not implement it. The result must be identical
on every rank.
"""
next_time(::Trigger, solver) = Inf

"""
    fired!(trigger, solver, Q) -> Bool

Return whether a [`Callback`](@ref) should run after the current completed
step. A custom [`Trigger`](@ref) must implement this method. The verdict must be
identical on every rank; a state-dependent trigger must perform any necessary
collective reduction itself.
"""
fired!(trigger::Trigger, solver, Q) =
    throw(ArgumentError("$(typeof(trigger)) must implement fired!(trigger, solver, Q)"))

"""
    rewind!(trigger, t, step)
    rewind!(effect, t, step)

Restore a trigger's schedule to `(t, step)`. [`run!`](@ref) calls this method
when `StepControl` restores a savepoint, for the trigger and then for the
effect of every [`Callback`](@ref). Scheduled instants between the savepoint
and the failed state are then visited on the replacement trajectory.

The built-in triggers restore their schedules: [`AtTime`](@ref) and
[`EveryTime`](@ref) from `t`, and a `once = true` [`WhenState`](@ref) that
fired after `step` is re-armed. A [`FieldWriter`](@ref) effect drops the
frames it recorded after `t` from its collection, and the replacement
trajectory's frames take their file names. Any other effect is not reversed:
state a user effect keeps outside the solver stays as the abandoned
trajectory left it unless that effect implements this method.

The default is a no-op, which is correct for any trigger reading only `step`
and for any effect that keeps no state.
"""
rewind!(::Trigger, t, step) = nothing
rewind!(effect, t, step) = nothing

"""
    fires_at_start(trigger, solver) -> Bool

Whether `trigger` is due at the initial state, before any step. [`run!`](@ref)
asks this once, when it starts from step 0, and runs the effect of each callback
whose trigger answers `true`, so that a scheduled output includes the initial
condition. [`AtTime`](@ref) and [`EveryTime`](@ref) answer `true` when one of
their instants is the initial time, and consume it; the default is `false`.
The verdict must be identical on every rank.
"""
fires_at_start(::Trigger, solver) = false

"""
    AtTime(t)
    AtTime([t1, t2, ...])

Fire once at each listed time. The list is sorted on construction, and `run!`
shortens `dt` to end a step at the next scheduled instant. A listed time equal to
the initial time fires before the first step. Times are
visited in order and each fires once; a time behind `solver.t` fires on
the next completed step. At most one time is consumed per step, so several times
behind `solver.t` are visited one per step.
"""
mutable struct AtTime <: Trigger
    times::Vector{Float64}
    next::Int
end

function AtTime(times::AbstractVector)
    all(isfinite, times) ||
        throw(ArgumentError("AtTime: every time must be finite, got $(collect(times))"))
    return AtTime(sort(collect(Float64, times)), 1)
end
AtTime(t::Real) = AtTime([t])

next_time(trigger::AtTime, solver) =
    trigger.next > length(trigger.times) ? Inf : trigger.times[trigger.next]

function fired!(trigger::AtTime, solver, Q)
    target = next_time(trigger, solver)
    isfinite(target) || return false
    # dt was clipped to land on `target`, so equality is the expected case; the
    # tolerance only absorbs the rounding in `solver.t += dt`. It is orders of
    # magnitude below any dt, so this cannot fire a step early.
    solver.t >= target - _land_tol(target, solver.t) || return false
    trigger.next += 1
    return true
end

function fires_at_start(trigger::AtTime, solver)
    target = next_time(trigger, solver)
    isfinite(target) && abs(solver.t - target) <= _land_tol(target, solver.t) ||
        return false
    trigger.next += 1
    return true
end

function rewind!(trigger::AtTime, t, step)
    idx = findfirst(target -> target > t + _land_tol(target), trigger.times)
    trigger.next = idx === nothing ? length(trigger.times) + 1 : idx
    return nothing
end

# The landing tolerance is shared by both time triggers, preventing disagreement
# between the schedule restored by `rewind!` and the one `fired!` reads.
_land_tol(t) = 8eps(max(abs(t), one(t)))

# The tolerance a trigger compares the clock against. The schedule is held in
# Float64 and the clock is the solver's element type, which may be narrower: a
# step landed on `target` reaches the nearest value that clock can represent,
# up to half its own spacing below the instant. The tolerance is therefore
# measured in the clock's precision. With a Float64 clock this is the
# single-argument form above, unchanged.
_land_tol(target, t) = _land_tol(oftype(t, max(abs(target), one(target))))

"""
    EveryTime(interval; start = 0.0)

Fire at `start + n*interval` for every integer `n >= 0`, with `run!` ending a step
at each instant. An instant at the initial time fires before the first step, so
`Callback(EveryTime(dt), FieldWriter(...))` writes the initial condition as its
first frame. This is the unbounded counterpart to [`AtTime`](@ref) and produces
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
    isfinite(interval) && interval > 0 ||
        throw(ArgumentError("EveryTime: interval must be finite and positive, " *
                            "got $interval"))
    isfinite(start) ||
        throw(ArgumentError("EveryTime: start must be finite, got $start"))
    return EveryTime(Float64(interval), Float64(start), NaN)
end

# Instants are computed as start + n*interval, not by accumulating the interval,
# preventing schedule drift from rounding in t during a long run.
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
    solver.t >= target - _land_tol(target, solver.t) || return false
    trigger.next = _instant_after(trigger, solver.t)
    return true
end

function fires_at_start(trigger::EveryTime, solver)
    t = solver.t
    n = round((t - trigger.start) / trigger.interval)
    n >= 0 || return false
    inst = trigger.start + n * trigger.interval
    abs(t - inst) <= _land_tol(inst, t) || return false
    # The next instant is anchored after t, which `_instant_after` already does.
    trigger.next = _instant_after(trigger, t)
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
when `once = false`. The condition must return a `Bool`.

The condition is evaluated on each rank and reduced across the communicator, so
it may be true only on the rank that owns a boundary plane. The callback
performs this reduction, so the condition may read a local field plane and
return a rank-local result.

A `once = true` trigger records the step at which it fired. A
[`StepControl`](@ref) rollback to a savepoint before that step re-arms it, so
the condition is evaluated again on the replacement trajectory.
"""
mutable struct WhenState{F} <: Trigger
    condition::F
    once::Bool
    done::Bool
    fired_step::Int     # completed step at which a once-trigger fired; -1 before
end

WhenState(condition; once::Bool=true) = WhenState(condition, once, false, -1)

function fired!(trigger::WhenState, solver, Q)
    trigger.done && return false
    # Reduced as an Int with `max`, not a Bool with `|`, matching the
    # reductions elsewhere in the solver; semantically this is a logical OR.
    local_hit = trigger.condition(solver, Q)::Bool
    hit = MPI.Allreduce(Int(local_hit), max, solver.comm) > 0
    if hit && trigger.once
        trigger.done = true
        trigger.fired_step = solver.step
    end
    return hit
end

# A firing after the savepoint's step happened on the abandoned trajectory.
function rewind!(trigger::WhenState, t, step)
    if trigger.done && trigger.fired_step > step
        trigger.done = false
        trigger.fired_step = -1
    end
    return nothing
end

# --- The end of the job.
#
# A checkpoint schedule in simulation time and a job limit in wall time drift
# apart as `dt` shrinks, so a run killed at its limit loses the wall time
# since its last checkpoint. `BeforeWallLimit` fires once when the job is
# about to end, read from the job's deadline, a signal, or both.
#
# A process-directed signal is delivered on whichever thread the kernel picks,
# an OpenBLAS or MPI thread included, and Julia code cannot run on a thread
# the runtime does not know: a `@cfunction` handler storing one atomic
# segfaulted when signals arrived during a multithreaded matrix product. The
# signal is therefore caught by libuv's watcher, whose native handler is safe
# on any thread and only writes to a pipe; the event loop, polled at each
# check, then runs the Julia callback on a Julia thread. The flag it sets
# enters the same reduction as the deadline test. There is one flag per
# process, since a signal means the end of the job for every trigger alike.
const WALL_SIGNAL = Threads.Atomic{Int}(0)

function _wall_signal_callback(handle::Ptr{Cvoid}, sig::Cint)
    WALL_SIGNAL[] = Int(sig)
    return nothing
end

# The watcher of each signal, installed once per process.
const WALL_SIGNAL_WATCHERS = Dict{Int,Ptr{Cvoid}}()

# Signal numbers by name. The values differ between Linux and the BSDs,
# macOS included.
function _signal_number(name::Symbol)
    linux = Sys.islinux()
    table = (USR1=linux ? 10 : 30, USR2=linux ? 12 : 31, TERM=15, INT=2, HUP=1)
    haskey(table, name) ||
        throw(ArgumentError("BeforeWallLimit: unknown signal :$name; name one of " *
                            "$(keys(table)) or pass its number"))
    return table[name]
end
_signal_number(sig::Integer) = Int(sig)

# Julia's own handling, on unless `--handle-signals=no`, blocks these signals
# in its threads and waits for them on a thread of its own, so a handler
# installed beside it would never run.
_signals_installable() = !Sys.iswindows() && Base.JLOptions().handle_signals == 0

function _install_wall_signals(signals)
    numbers = unique(Int[_signal_number(s) for s in signals])
    (isempty(numbers) || !_signals_installable()) && return Int[]
    callback = @cfunction(_wall_signal_callback, Cvoid, (Ptr{Cvoid}, Cint))
    for sig in numbers
        haskey(WALL_SIGNAL_WATCHERS, sig) && continue
        # Zeroed: at exit Julia closes every handle of its loop and reads a
        # non-null `data` field as a Julia object.
        handle = Libc.calloc(1, ccall(:uv_handle_size, Csize_t, (Cint,), Base.UV_SIGNAL))
        Base.iolock_begin()
        try
            status = ccall(:uv_signal_init, Cint, (Ptr{Cvoid}, Ptr{Cvoid}),
                           Base.eventloop(), handle)
            status == 0 || (Libc.free(handle); throw(Base._UVError("uv_signal_init", status)))
            status = ccall(:uv_signal_start, Cint, (Ptr{Cvoid}, Ptr{Cvoid}, Cint),
                           handle, callback, sig)
            if status != 0
                ccall(:uv_close, Cvoid, (Ptr{Cvoid}, Ptr{Cvoid}), handle, C_NULL)
                throw(Base._UVError("uv_signal_start($sig)", status))
            end
            # A watcher does not keep the loop, and so the process, alive.
            ccall(:uv_unref, Cvoid, (Ptr{Cvoid},), handle)
        finally
            Base.iolock_end()
        end
        WALL_SIGNAL_WATCHERS[sig] = handle
    end
    return numbers
end

"""
    BeforeWallLimit(; margin = 600.0, deadline = nothing, walltime = nothing,
                      signals = (:USR1, :TERM), check_every = 1)

Fire once when the job running the solver is about to reach its time limit,
so that the [`Callback`](@ref) effect can write a checkpoint before the
scheduler kills the job. An effect that writes the checkpoint and returns
`true` ends the run:

```julia
stop = Callback(BeforeWallLimit(margin = 600.0),
                (solver, Q) -> (save_checkpoint(solver, Q, "restart/state"); true))
run!(solver, Q; tfinal, callback = (ProgressLog(), stop))
```

The trigger fires on the first check at which either of two conditions holds.

The first is the wall-clock deadline. The trigger fires when the job would
have less than `margin` seconds left at the following check, which is
`check_every` steps of the last step's wall time later. `margin` should cover
the checkpoint write and anything else the job does after `run!` returns. The
deadline is taken from the first of these that gives one:

- `deadline`, an absolute time in seconds since the Unix epoch, as `time()`
  returns it;
- `walltime`, in seconds from the trigger's construction;
- the `SLURM_JOB_END_TIME` environment variable, where Slurm sets it;
- under Slurm, `squeue -h -j \$SLURM_JOB_ID -o %L`, the job's remaining time;
- under Flux, `flux job timeleft`.

The scheduler is queried once, at the first check, by rank 0 alone. When none
of these gives a deadline, the trigger fires on a signal only.

The second condition is the receipt of one of `signals`, by name (`:USR1`,
`:USR2`, `:TERM`, `:INT`, `:HUP`) or by number, on any rank. The signal only
sets a flag, which the next check reads. Slurm sends `SIGTERM` at the time
limit and `SIGKILL` after its `KillWait`, often 30 s, which is seldom long
enough for a step and a checkpoint, so an earlier signal is the useful one:
`#SBATCH --signal=USR1@600` sends `SIGUSR1` to every task 600 s before the
limit. The form `--signal=B:USR1@600` signals the batch shell alone, and
reaches the ranks only through a `trap` that forwards it.

The signals are watched from construction, and only in a Julia started with
`--handle-signals=no`, as an MPI launch at `-t 1` should be. Julia's own
handling otherwise takes them (`SIGTERM` ends the process and `SIGUSR1`
prints a profile), and the trigger reads the deadline alone. With more than
one thread `--handle-signals=no` is not usable at all, since Julia's garbage
collector then needs its own signal handlers. No signal is watched on
Windows. Once watched, the listed signals set the flag for the rest of the
process instead of ending it, so a `SIGTERM` that arrives outside `run!` is
ignored until `SIGKILL` follows it.

The check is one `Allreduce` of an `Int` per `check_every` steps, and every
rank reaches the same verdict. The trigger fires once. A
[`StepControl`](@ref) rollback to a savepoint before the step it fired on
re-arms it, so that a run stopped by it and then rolled back by the final
state check stops again on the replacement trajectory.

`trigger.deadline` and `trigger.source` report the deadline the first check
settled on (`Inf` and `:none` without one) and where it came from.
"""
mutable struct BeforeWallLimit <: Trigger
    margin::Float64
    check_every::Int
    deadline::Float64     # seconds since the epoch, Inf for none
    source::Symbol        # where `deadline` came from
    resolved::Bool        # whether the first check has settled `deadline`
    signals::Vector{Int}  # the signals watched
    done::Bool
    fired_step::Int       # completed step at which it fired; -1 before
end

# The sources of a deadline, in the order `_job_deadline` tries the
# scheduler's; the first check broadcasts rank 0's as its index here.
const DEADLINE_SOURCES = (:none, :deadline, :walltime, :slurm_env, :squeue, :flux)

function BeforeWallLimit(; margin::Real=600.0, deadline::Union{Real,Nothing}=nothing,
                         walltime::Union{Real,Nothing}=nothing,
                         signals=(:USR1, :TERM), check_every::Int=1)
    isfinite(margin) && margin >= 0 ||
        throw(ArgumentError("BeforeWallLimit: margin must be finite and " *
                            "non-negative, got $margin"))
    check_every >= 1 ||
        throw(ArgumentError("BeforeWallLimit: check_every must be >= 1, " *
                            "got $check_every"))
    deadline !== nothing && walltime !== nothing &&
        throw(ArgumentError("BeforeWallLimit: give deadline or walltime, not both"))
    given, source = if deadline !== nothing
        isnan(deadline) &&
            throw(ArgumentError("BeforeWallLimit: deadline must be a time, got NaN"))
        Float64(deadline), :deadline
    elseif walltime !== nothing
        walltime >= 0 ||
            throw(ArgumentError("BeforeWallLimit: walltime must be non-negative, " *
                                "got $walltime"))
        time() + Float64(walltime), :walltime
    else
        Inf, :none
    end
    return BeforeWallLimit(Float64(margin), check_every, given, source, false,
                           _install_wall_signals(signals), false, -1)
end

function fired!(trigger::BeforeWallLimit, solver, Q)
    trigger.done && return false
    solver.step % trigger.check_every == 0 || return false
    comm = solver.comm
    rank = MPI.Comm_rank(comm)
    # Settled from rank 0's reading, so every rank compares against the same
    # instant: a `walltime` is counted from each rank's own construction, and
    # the scheduler is asked once, by one rank.
    if !trigger.resolved
        deadline, source = trigger.deadline, trigger.source
        rank == 0 && source === :none && ((deadline, source) = _job_deadline())
        trigger.deadline = MPI.Allreduce(rank == 0 ? deadline : -Inf, max, comm)
        index = rank == 0 ? findfirst(==(source), DEADLINE_SOURCES)::Int : 0
        trigger.source = DEADLINE_SOURCES[MPI.Allreduce(index, max, comm)]
        trigger.resolved = true
        rank == 0 && _report_deadline(trigger)
    end
    # A signal received since the last check is still in libuv's pipe until
    # the loop runs, which it does here and otherwise only when a task waits.
    isempty(trigger.signals) || Base.process_events()
    # The clock is read on rank 0 alone; the reduction carries its verdict
    # and any rank's signal to every rank.
    late = rank == 0 &&
           time() + trigger.check_every * solver.wall_step >=
           trigger.deadline - trigger.margin
    hit = MPI.Allreduce(Int(late) | Int(WALL_SIGNAL[] != 0), max, comm) > 0
    if hit
        trigger.done = true
        trigger.fired_step = solver.step
    end
    return hit
end

function rewind!(trigger::BeforeWallLimit, t, step)
    if trigger.done && trigger.fired_step > step
        trigger.done = false
        trigger.fired_step = -1
    end
    return nothing
end

# The job's end, in seconds since the epoch, and where it was read; `Inf` and
# `:none` outside a scheduler or when every query fails. Rank 0 only.
function _job_deadline(env=ENV)
    ending = tryparse(Float64, get(env, "SLURM_JOB_END_TIME", ""))
    ending !== nothing && ending > 0 && return ending, :slurm_env
    job = get(env, "SLURM_JOB_ID", "")
    if !isempty(job)
        left = _slurm_seconds(_command_output(`squeue -h -j $job -o %L`))
        left !== nothing && return time() + left, :squeue
    end
    if haskey(env, "FLUX_JOB_ID")
        out = _command_output(`flux job timeleft`)
        left = out === nothing ? nothing : tryparse(Float64, strip(out))
        left !== nothing && return time() + left, :flux
    end
    return Inf, :none
end

# The standard output of `cmd`, or `nothing` when it is missing, fails or
# takes longer than `timeout` seconds, so that an unresponsive scheduler
# controller delays the first step by at most that.
function _command_output(cmd::Cmd; timeout::Real=30)
    out = IOBuffer()
    process = try
        run(pipeline(cmd; stdout=out, stderr=devnull); wait=false)
    catch
        return nothing
    end
    timer = Timer(_ -> kill(process), timeout)
    try
        wait(process)
    catch
    finally
        close(timer)
    end
    success(process) || return nothing
    return String(take!(out))
end

# Slurm's remaining-time field in seconds: `[days-]hours:minutes:seconds`,
# `minutes:seconds`, or a bare count of minutes; `nothing` for `UNLIMITED`,
# `NOT_SET`, `INVALID` or anything else that is not a time.
function _slurm_seconds(text)
    text === nothing && return nothing
    s = strip(text)
    has_days = occursin('-', s)
    days, rest = has_days ? split(s, '-'; limit=2) : ("0", s)
    d = tryparse(Int, days)
    parts = map(x -> tryparse(Int, x), split(rest, ':'))
    (d === nothing || any(isnothing, parts) || length(parts) > 3) && return nothing
    # After a day count the fields run from hours; without one, two fields
    # are minutes and seconds and one is minutes.
    h, m, sec = has_days ? vcat(parts, zeros(Int, 3 - length(parts))) :
                length(parts) == 3 ? parts :
                length(parts) == 2 ? [0; parts] : [0, parts[1], 0]
    return Float64(((d * 24 + h) * 60 + m) * 60 + sec)
end

function _report_deadline(trigger::BeforeWallLimit)
    signals = isempty(trigger.signals) ? "" :
              " or on signal " * join(trigger.signals, " or ")
    if isfinite(trigger.deadline)
        at = Libc.strftime("%Y-%m-%d %H:%M:%S", trigger.deadline)
        left = strip(_duration(max(trigger.deadline - time(), 0.0)))
        @info "BeforeWallLimit: the job ends at $at (from $(trigger.source)), " *
              "in $left; the trigger fires $(trigger.margin) s before it$signals."
    elseif !isempty(get(ENV, "SLURM_JOB_ID", "")) || haskey(ENV, "FLUX_JOB_ID")
        @warn "BeforeWallLimit: the scheduler gave no time limit for this job; " *
              (isempty(signals) ? "with no signal installed, the trigger will not fire." :
                                  "the trigger fires$signals only.")
    end
    return nothing
end

"""
    Callback(trigger, effect!)

Pair a [`Trigger`](@ref) with `effect!(solver, Q)`, run after a completed step
(and after any filtering) when the trigger fires. Returning `true` from
`effect!` requests that `run!` stop; any other value continues.

Pass one to `run!` as `callback=`, or pass several as a tuple. Every element of a
tuple runs, including after one has requested a stop. A bare function is also
accepted; it runs after every step and its return value is ignored.

A [`StepControl`](@ref) rollback abandons the steps after its savepoint, whose
callbacks have already run. [`rewind!`](@ref) is then called on the trigger and
on `effect!`; an effect that keeps state across steps implements it to undo
what it did on the abandoned steps.
"""
struct Callback{Tr<:Trigger,F}
    trigger::Tr
    effect!::F
end

# --- Dispatch over what `run!` was handed: nothing, a bare callable, one
# Callback, or a tuple of them. Tuple dispatch recurses without iteration, so the
# whole call stays inferable with mixed trigger types, the same reason `sources` is a
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
rewind_callbacks!(cb::Callback, t, step) =
    (rewind!(cb.trigger, t, step); rewind!(cb.effect!, t, step))
rewind_callbacks!(cbs::Tuple, t, step) =
    (rewind_callbacks!(first(cbs), t, step);
     rewind_callbacks!(Base.tail(cbs), t, step))
rewind_callbacks!(_, t, step) = nothing

# Frames written after the savepoint's time belong to the abandoned trajectory.
# Dropping them from the frame list gives the replacement frames their indices
# and file names; the collection file is rewritten with the next frame.
function rewind!(writer::FieldWriter, t, step)
    keep = count(τ -> τ <= t + _land_tol(t), writer.times)
    resize!(writer.times, keep)
    length(writer.grids) > keep && resize!(writer.grids, keep)
    writer.index = writer.start_index + keep
    return nothing
end

# The initial-state pass: only callbacks whose trigger is due at the start run.
run_start_callbacks!(::Nothing, solver, Q) = false
run_start_callbacks!(::Tuple{}, solver, Q) = false
run_start_callbacks!(cb::Callback, solver, Q) =
    fires_at_start(cb.trigger, solver) && cb.effect!(solver, Q) === true
function run_start_callbacks!(cbs::Tuple, solver, Q)
    stop_first = run_start_callbacks!(first(cbs), solver, Q)
    stop_rest = run_start_callbacks!(Base.tail(cbs), solver, Q)
    return stop_first || stop_rest
end
run_start_callbacks!(_, solver, Q) = false

# The per-step pass returns `(stop, ran)`: whether an effect asked to end the
# run, and whether any effect ran at all. An effect may write the state, so
# `run!` repeats the level restriction before the next step whenever one ran.
# A trigger decides identically on every rank (see the top of this file), so
# `ran` is the same everywhere.
run_callbacks!(::Nothing, solver, Q) = (false, false)
run_callbacks!(::Tuple{}, solver, Q) = (false, false)

function run_callbacks!(cb::Callback, solver, Q)
    fired!(cb.trigger, solver, Q) || return (false, false)
    return (cb.effect!(solver, Q) === true, true)
end

function run_callbacks!(cbs::Tuple, solver, Q)
    # Both sides run: an early return would let the first stop request skip the
    # rest, causing a diagnostic callback to miss its final step.
    stop_first, ran_first = run_callbacks!(first(cbs), solver, Q)
    stop_rest, ran_rest = run_callbacks!(Base.tail(cbs), solver, Q)
    return (stop_first || stop_rest, ran_first || ran_rest)
end

# A bare callable keeps its old contract exactly: called every step, return value
# ignored. Honouring `=== true` here would silently turn any existing callback
# whose last expression happens to be a comparison into an early stop.
run_callbacks!(cb, solver, Q) = (cb(solver, Q); (false, true))

# The calls `run!`'s loop makes into the callbacks. The loop holds the callback
# unspecialized, so each is a dynamic call, compiled per callback type, and none
# allocates: a returned float or tuple would be boxed, so the next instant is
# written into a cell of the clock's type and the two flags of the per-step
# pass return as a small integer (`stop + 2 ran`), which Julia keeps
# preallocated; and a `ConservedState`, an immutable wrapper that would be
# boxed at each call, crosses as its array and is rewrapped on arrival.
@inline _crossing(Q::ConservedState) = (parent(Q), Val(true))
@inline _crossing(Q) = (Q, Val(false))
@inline _arrived(A, ::Val{true}) = ConservedState(A)
@inline _arrived(Q, ::Val{false}) = Q

function _next_instant!(cell::Base.RefValue, callback, solver)
    cell[] = oftype(cell[], callback_next_time(callback, solver))
    return nothing
end

_start_callbacks(callback, solver, qx, wrapped::Val) =
    run_start_callbacks!(callback, solver, _arrived(qx, wrapped))

function _step_callbacks(callback, solver, qx, wrapped::Val)
    stop, ran = run_callbacks!(callback, solver, _arrived(qx, wrapped))
    return Int(stop) + 2 * Int(ran)
end

# --- Progress reporting.
#
# Examples once contained separate copies of this loop, with errors in at least
# one of three cluster details: printing from all ranks, not one,
# forgetting to flush (srun buffers stdout, so a run appears hung for minutes),
# and timing the diagnostic's own reduction as if it were solver cost. The
# collective-ordering rule from the top of this file applies too: `quantity`
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

Construct a [`Callback`](@ref) that reports the step, `t`, the `dt` of the
completed step (`solver.dt_prev`), and wall time per step from
`solver.wall_step`. It reports once every `every` completed steps, writes to
`io`, and never stops a run.

`quantity(solver, Q) -> Real` adds one scalar column, named by `label`. It is
called on every rank and may reduce, as `volume_integral` and related diagnostics
do. Only output is restricted to rank 0.

`tfinal` adds percent-complete and a projected remaining time, from the mean
step so far, not the last one. `imbalance = true` adds the min/max spread
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
    comm = solver.comm
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
