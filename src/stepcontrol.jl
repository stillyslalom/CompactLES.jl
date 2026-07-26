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
Planck time in seconds. The unconditional failsafe floor on `dt`: no physical
problem has a timescale below it, so a step this small means the run is not
computing anything and never will.

This floor is unit-blind and deliberately so. A non-dimensional run has no
seconds in it, and 5.4e-44 is then simply an absurdity threshold rather than a
physical statement — which is the intent, since it is a tripwire and not a
model. If you genuinely work in units where the interesting timescale is
1e-40, set `dt_min` yourself and this floor is the least of your problems.
"""
const PLANCK_TIME = 5.391247e-44

"""
    StepControl(; kwargs...)

Policy for how [`run!`](@ref) chooses, floors, and recovers a timestep.

## Prediction — both off by default

Neither is on by default: both perturb the step sequence of a healthy run and
neither was measured to prevent a failure. See the note at the top of this file
for what was measured. They are kept because they are the obvious things to
reach for, and the measurement is worth more than the code.

- `predict = 0.0` — steps of linear extrapolation of the CFL rate, compensating
  for `compute_dt` seeing the artificial coefficients one step late. Measured:
  no lookahead between 0 and 30 steps prevents the Noh failure at cfl = 0.3.
- `max_growth = 0.0` — cap on how fast `dt` may grow between steps; 0 disables.
  Measured: 1.05 delays the same failure from step 175 to 186. A delay, not a
  fix.

## Floors

- `dt_min = 0.0` — absolute floor. 0 means only [`PLANCK_TIME`](@ref) applies.
- `dt_min_ratio = 1e-8` — floor relative to the largest `dt` this run has seen.
  This is the one that actually catches a collapse, because it needs no
  knowledge of the problem's units or scales: legitimate physical variation in
  `dt` over a run is a few orders of magnitude, and a collapse is ten or more.

## Recovery

- `retries = 0` — number of times to roll back to the last savepoint and retry
  with a reduced CFL. 0 fails fast. This is the mechanism worth turning on:
  `retries = 4` recovers every Noh geometry from a deliberately bad `cfl = 0.9`,
  in fewer steps than running the whole thing at the CFL that would have been
  safe from the start.
- `cfl_backoff = 0.5` — CFL multiplier applied on each retry.
- `savepoint_interval = 25` — steps between savepoints. Only allocated when
  `retries > 0`; costs one extra state array.

Rollback recovers an ABRUPT failure — a CFL guessed too high, which blows up
in the first tens of steps and is fine once past it. It does not recover
GRADUAL degradation: if the state has been quietly losing positivity for a
hundred steps before the check fires, the last savepoint is already damaged and
no amount of backing off from it helps. In that regime the answer is a lower
CFL from the start, or a larger `savepoint_interval` so the rollback reaches
further back.
"""
Base.@kwdef struct StepControl
    predict::Float64 = 0.0
    max_growth::Float64 = 0.0
    dt_min::Float64 = 0.0
    dt_min_ratio::Float64 = 1e-8
    retries::Int = 0
    cfl_backoff::Float64 = 0.5
    savepoint_interval::Int = 25
end

"""
    SolverFailure

Thrown by [`run!`](@ref) when the timestep or the state fails a
[`StepControl`](@ref) check and no retries remain. `reason` is one of
`:nonfinite`, `:planck`, `:dt_min`, `:dt_collapse`, or `:negative_density`.
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

Deliberately no CFL. The reduced CFL has to survive a rollback and compound
across retries, so it lives on the solver and is never restored from here.
"""
mutable struct Savepoint{A}
    Q::A
    t::Float64
    step::Int
end

"""
    check_step(control, dt, rate, rho_min, dt_seen, solver) -> Union{Nothing,SolverFailure}

Apply the floors and the positivity check. Returns `nothing` when the step is
acceptable. Pure: the caller decides whether to throw or to retry.
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

