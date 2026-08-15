# Timestep control policy: floors, prediction, and recovery.
#
# Separated from timestep.jl only because `Solver` carries a `StepControl` field
# and therefore has to see the type first. The logic that needs a Solver —
# `predicted_dt` and `run!` — stays with the integrator.

# The failure mode this exists for is not a crash. When the artificial
# properties lose control of a strong shock the state goes unphysical, the
# diffusive term in the CFL rate climbs, `dt` collapses, and the run grinds
# forever making no progress — burning hours to produce nothing, with no signal
# beyond a step counter that keeps incrementing.
#
# WHAT THE COLLAPSE ACTUALLY IS. It is tempting to read that as a timestep
# problem: `compute_dt` builds its diffusive rate from the PREVIOUS step's
# artificial coefficients, so at a forming shock the step is chosen from
# coefficients that are already stale, and the obvious fix is to extrapolate the
# rate forward. That fix was built (`predict` below) and measured, and it does
# not work. Noh at nu=1, N=400, cfl=0.3 fails at step 175 with no lookahead and
# at step 179 with three steps of it; thirty steps of lookahead buys 25 steps.
# The trace says why:
#
#     step   25  dt=2.1e-4  rate=1418   rho_min=0.951
#     step   75  dt=1.8e-4  rate=1633   rho_min=0.798
#     step  125  dt=1.8e-4  rate=1718   rho_min=0.398
#     step  175  dt=4.8e-5  rate=1.5e4  rho_min=0.252   <- and then negative
#
# The pre-shock density is exactly 1 in the exact solution. It is already 5%
# low by step 25 and 60% low by step 125, while `dt` and the rate sit flat. The
# state is being eaten by a dispersive undershoot at the shock that the
# artificial viscosity is not damping; positivity goes at ~175, and only THEN
# does the rate explode. The dt collapse is the symptom, arriving about 150
# steps after the disease.
#
# So the three mechanisms here are not equals:
#
#   1. FLOORS and the POSITIVITY CHECK turn the silent grind into an exception
#      with a reason attached, and catch it ~150 steps earlier than a dt-based
#      test alone would. On by default.
#   2. The PREDICTOR is retained, off by default, with the measurement above
#      recorded so the next person does not rebuild it. It perturbs the step
#      sequence of healthy runs for no demonstrated benefit.
#   3. RETRY is the mechanism that actually helps, and it helps more than the
#      guidance it replaces. Rolling back and halving the CFL recovers Noh nu=1
#      from cfl 0.9 to 0.45 in 1433 steps with the correct plateau (0.9989),
#      against 4485 steps at a globally fixed cfl 0.15 for the same answer. The
#      CFL restriction is a STARTUP restriction — once the shock has formed
#      cleanly the large step is fine — which is exactly what a global CFL
#      cannot express and a rollback can.
#
# What none of this catches: a run that completes and is simply wrong. Noh
# nu=1 at cfl=0.5 reaches t_final without tripping anything and returns a
# plateau 2% of the exact value. Floors detect non-computation, not error.

"""
Planck time in seconds, the unconditional failsafe floor on `dt`. `check_step`
rejects any step not strictly greater than it whatever the rest of a
[`StepControl`](@ref) says, so for a calculation expressed in physical units a
step this small ends the run as numerical collapse.

The comparison is made against `dt` in whatever units the calculation carries.
In a nondimensional calculation 5.4e-44 is a numerical failsafe rather than a
physical scale, and a calculation whose meaningful timestep is of comparable
magnitude cannot run at all: `dt_min` raises the floor and nothing lowers it,
so such a problem has to be rescaled.
"""
const PLANCK_TIME = 5.391247e-44

"""
    StepControl(; kwargs...)

Policy for how [`run!`](@ref) chooses, floors, and recovers a timestep.

## Prediction

Both prediction controls are disabled by default because they change the step
sequence and did not prevent failure in the measurements recorded at the top of
this file.

- `predict = 0.0` — steps of linear extrapolation of the CFL rate, compensating
  for `compute_dt` seeing the artificial coefficients one step late. The
  extrapolation is one-sided and only ever raises the rate, so it can shorten a
  step but never lengthen one. Measured: no lookahead between 0 and 30 steps
  prevents the Noh failure at cfl = 0.3.
- `max_growth = 0.0` — largest multiple of the previous step that `dt` may take;
  0 disables the cap. A value of 1.05 delayed the same failure from step 175 to
  186 but did not prevent it.

## Landing on a scheduled instant

- `landing_steps = 2` — how many steps ahead of a scheduled callback time
  ([`AtTime`](@ref), [`EveryTime`](@ref)) `run!` may start shortening the step
  so that a step ends at that instant. The remaining interval is divided into
  equal steps, with no step below `dt/2`. A value of 1 applies a hard clip and
  can produce an arbitrarily small step immediately before an output time.
  Larger values distribute the adjustment over more steps while operating
  farther below the CFL limit. A value below 1 is rejected at construction.
  `tfinal` always uses a hard clip.

## Floors

- `dt_min = 0.0` — absolute floor on `dt`; 0 disables it. The unconditional
  [`PLANCK_TIME`](@ref) floor and the relative floor below apply either way.
- `dt_min_ratio = 1e-8` — floor relative to the largest `dt` taken so far in
  the current `run!` call; 0 disables it. This detects collapse without prior
  knowledge of the problem's units or scales. Physical variation in `dt` is
  generally a few orders of magnitude, whereas a collapse spans ten or more in
  the measured cases. A rollback resets the reference, so after a retry the
  floor measures the replacement trajectory alone.

## Recovery

- `retries = 0` — number of times to roll back to the last savepoint and retry
  with a reduced CFL; 0 disables recovery. In the validation cases,
  `retries = 4` recovers every Noh geometry from `cfl = 0.9` in fewer steps than
  running the complete calculation at the lower stable CFL.
- `cfl_backoff = 0.5` — multiplier applied to the solver's current CFL on each
  retry, so successive retries compound.
- `savepoint_interval = 25` — steps between savepoints, counted on the solver's
  step number. The savepoint costs one extra state array and is allocated only
  when `retries > 0`; a value of 0 or less leaves the state on entry to `run!`
  as the only rollback target.

Rollback can recover an abrupt failure caused by an excessive initial CFL. It
does not recover gradual degradation when the most recent savepoint already
contains a loss of positivity. Such cases require a lower initial CFL or a
larger `savepoint_interval` that places the savepoint earlier in the trajectory.
"""
Base.@kwdef struct StepControl
    predict::Float64 = 0.0
    max_growth::Float64 = 0.0
    landing_steps::Int = 2
    dt_min::Float64 = 0.0
    dt_min_ratio::Float64 = 1e-8
    retries::Int = 0
    cfl_backoff::Float64 = 0.5
    savepoint_interval::Int = 25
    # `landing_steps = 0` does not simply disable the shortening. The gap clip
    # is the same expression, so a scheduled instant would be overshot and its
    # trigger would fire late.
    function StepControl(predict, max_growth, landing_steps, dt_min, dt_min_ratio,
                         retries, cfl_backoff, savepoint_interval)
        landing_steps >= 1 ||
            throw(ArgumentError("StepControl: landing_steps must be >= 1 " *
                                "(1 is a hard clip onto the scheduled time)"))
        new(predict, max_growth, landing_steps, dt_min, dt_min_ratio,
            retries, cfl_backoff, savepoint_interval)
    end
end

"""
    SolverFailure

Thrown by [`run!`](@ref) when the timestep or the state fails a
[`StepControl`](@ref) check and no retries remain. `reason` is one of
`:nonfinite`, `:planck`, `:dt_min`, `:dt_collapse`, or `:negative_density`. The
remaining fields record the state the check rejected: `step`, `t`, `dt`, `cfl`,
and a `detail` string describing the failure, which `showerror` prints below the
summary line.
"""
struct SolverFailure <: Exception
    reason::Symbol
    step::Int
    t::Float64
    dt::Float64
    cfl::Float64
    detail::String
end

function Base.showerror(io::IO, e::SolverFailure)
    print(io, "SolverFailure(:", e.reason, ") at step ", e.step,
          ", t = ", e.t, ", dt = ", e.dt, ", cfl = ", e.cfl, "\n  ", e.detail)
    if e.reason in (:dt_collapse, :dt_min, :planck, :negative_density)
        print(io, "\n  Most likely the artificial properties have lost a strong ",
                  "shock. Run with\n  `StepControl(retries = 4)` to have run! roll ",
                  "back and lower the CFL by itself —\n  usually better than ",
                  "guessing a lower cfl up front, since the restriction is\n  ",
                  "normally a startup transient. See reference/CALIBRATION.md.")
    end
end

"""
In-memory rollback point: the conserved state plus the clock that goes with it.
`Q` is a private copy of the conserved array, which [`run!`](@ref) allocates
when `StepControl.retries > 0` and refreshes in place at each savepoint; `t`
and `step` are the solver clock at the moment of that copy.

The CFL is excluded because reductions to it must persist across rollbacks and
compound across retries; it remains on the solver when this state is restored.
"""
mutable struct Savepoint{A}
    Q::A
    t::Float64
    step::Int
end

"""
    check_step(control, dt, rho_min, dt_seen, step, t, cfl) -> Union{Nothing,SolverFailure}

Apply the floors and the positivity check to a proposed step. `dt` is the step
about to be taken, `rho_min` the global minimum mixture density and `dt_seen`
the largest `dt` this run has taken so far; `step`, `t` and `cfl` are recorded
on the returned [`SolverFailure`](@ref) and are not tested. Returns `nothing`
when the step is acceptable.

Pure: the caller decides whether to throw or to retry. Not collective either,
but every argument is a reduced quantity or advances identically on all ranks
(`rho_min` and the rate behind `dt` come from the collective in
[`max_rate`](@ref)), so the verdict agrees across the communicator and
[`run!`](@ref) can roll back on it without deadlocking.
"""
function check_step(control::StepControl, dt, rho_min, dt_seen, step, t, cfl)
    fail(reason, detail) = SolverFailure(reason, step, t, dt, cfl, detail)
    isfinite(dt) ||
        return fail(:nonfinite, "timestep is $dt; the state has gone non-finite")
    rho_min > 0 ||
        return fail(:negative_density,
                    "minimum mixture density is $rho_min; the state is unphysical " *
                    "and primitives! is substituting placeholders for it")
    dt > PLANCK_TIME ||
        return fail(:planck, "timestep is below the Planck time ($PLANCK_TIME)")
    dt >= control.dt_min ||
        return fail(:dt_min, "timestep is below the configured dt_min " *
                             "($(control.dt_min))")
    if control.dt_min_ratio > 0 && dt_seen > 0
        dt >= control.dt_min_ratio * dt_seen ||
            return fail(:dt_collapse,
                        "timestep has collapsed to $(dt / dt_seen) of the largest " *
                        "seen this run ($dt_seen)")
    end
    return nothing
end

