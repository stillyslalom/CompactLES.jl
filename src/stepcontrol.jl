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
# --- What the collapse actually is -------------------------------------------
#
# It is tempting to read that as a timestep problem: `compute_dt` builds its
# diffusive rate from the *previous* step's artificial coefficients, so at a
# forming shock the step is chosen from coefficients that are already stale, and
# the obvious fix is to extrapolate the rate forward. That fix was built
# (`predict` below) and measured, and it does not work. Noh at nu=1, N=400,
# cfl=0.3 fails at step 175 with no lookahead and at step 179 with three steps
# of it; thirty steps of lookahead buys 25 steps. The trace says why:
#
#     step   25  dt=2.1e-4  rate=1418   rho_min=0.951
#     step   75  dt=1.8e-4  rate=1633   rho_min=0.798
#     step  125  dt=1.8e-4  rate=1718   rho_min=0.398
#     step  175  dt=4.8e-5  rate=1.5e4  rho_min=0.252   <- and then negative
#
# The pre-shock density is exactly 1 in the exact solution. It is already 5%
# low by step 25 and 60% low by step 125, while `dt` and the rate sit flat. The
# state is being eaten by a dispersive undershoot at the shock that the
# artificial viscosity is not damping; positivity goes at ~175, and only then
# does the rate explode. The dt collapse is the symptom, arriving about 150
# steps after the disease.
#
# So the four mechanisms here are not equals:
#
#   1. The floors and the positivity check turn the silent grind into an
#      exception with a reason attached, and catch it ~150 steps earlier than
#      a dt-based test alone would. On by default.
#   2. The predictor is retained, off by default, with the measurement above
#      recorded so the next person does not rebuild it. It perturbs the step
#      sequence of healthy runs for no demonstrated benefit.
#   3. Retry is the mechanism that actually helps, and it helps more than the
#      guidance it replaces. Rolling back and halving the CFL recovers Noh nu=1
#      from cfl 0.9 to 0.45 in 1433 steps with the correct plateau (0.9989),
#      against 4485 steps at a globally fixed cfl 0.15 for the same answer. The
#      CFL restriction is a startup restriction — once the shock has formed
#      cleanly the large step is fine — which is exactly what a global CFL
#      cannot express and a rollback can.
#   4. The state floor repairs rather than detects, and is off by default. The
#      first three all read a reduced scalar, and `bench/nohprobe.jl` measured
#      what that misses: six to eight interior cells carry negative internal
#      energy, travelling with the front, for the whole duration of Noh runs
#      that complete and pass their guards. Nothing reports it, because
#      `primitives!` floors T_ion at 1e-300 wherever e <= 0 and the positivity
#      check above reads rho, which stays positive throughout. `floor_ratio`
#      makes those cells visible and repairable. See `apply_positivity_floor!`.
#
# --- What the floor may not repair -------------------------------------------
#
# Those negative-e cells are not a rounding artifact, and repairing them is not
# cheap. Over the complete shipped Noh nu=1 case at cfl 0.15, which takes 3724
# steps and returns a plateau of 3.997 against the exact 4, the state carries
# 24991 cell-steps of e < 0, peaking at 8 cells at once and reaching -718 e0.
# Total energy density and mixture density stay positive at every cell of every
# step. The wall region runs as a pressureless layer, and the scheme remains
# stable there for the whole run.
#
# Forcing e back above zero costs a 5% velocity damping on the worst cell after
# a single step (-337 e0; the next four need 0.7%, 0.08%, 0.05%, 0.02%). Applied
# every step it removes 1.5e-4 of the total momentum on step 1, produces cells
# of negative *total* energy by step 4 that the unrepaired trajectory never
# produces, and the run then fails at step 18 with :dt_collapse. Raising E
# instead of damping the velocity is the same order of intervention, about 11%
# of that cell's energy, so the cost belongs to the case rather than to the
# choice of repair.
#
# `floor_scope` follows from that measurement. The default repairs only what is
# unrepresentable, which on that case is nothing at all, and counts the negative
# internal energy so that it is reported rather than silent. `:internal_energy`
# opts into the repair above, for sweeps and development runs where the goal is
# a bounded run rather than a faithful trajectory.
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
contains a loss of positivity. Such cases require a lower initial CFL, a larger
`savepoint_interval` that places the savepoint earlier in the trajectory, or the
state floor below.

## The positivity failsafe

- `floor_ratio = 0.0` — inspect the conserved state after every completed step
  and repair it where it has left the physical state space; 0 disables the
  failsafe entirely. The two floors are derived from the state entering
  [`run!`](@ref), as `floor_ratio` times its global minimum mixture density and
  its global minimum internal energy, so the setting carries no units and needs
  no advance knowledge of the problem's scales, on the same reasoning as
  `dt_min_ratio` above.
- `floor_scope = :representable` — how much of that state space to insist on.
  `:representable` repairs a mixture density below the floor and a total energy
  density below it, and *counts* internal energy below the floor without
  touching it. `:internal_energy` additionally repairs the internal energy, by
  the velocity damping `apply_positivity_floor!` describes.

The distinction is measured rather than stylistic, and the comment at the top of
`stepcontrol.jl` carries the numbers. Negative internal energy is a persistent
feature of converging-shock runs that complete and pass their guards, where
density and total energy stay positive throughout; repairing it every step is a
percent-level intervention that terminates such a run. `:representable`
therefore makes an existing calculation observable without changing it, while
`:internal_energy` suits sweeps and development runs where the goal is a bounded
run rather than a faithful trajectory.

Under either scope, every firing is counted in `solver.floor_tally`
([`FloorTally`](@ref)) and reported. Enabling the failsafe removes
`:negative_density` as a route to [`SolverFailure`](@ref), since the density
that check reads can no longer reach zero; `:dt_collapse` and `:nonfinite` still
terminate a diverging run, so a sweep does not grind. The repair is a recovery
from states the scheme has already left rather than a change to the scheme.
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
    floor_ratio::Float64 = 0.0
    floor_scope::Symbol = :representable
    # `landing_steps = 0` does not simply disable the shortening. The gap clip
    # is the same expression, so a scheduled instant would be overshot and its
    # trigger would fire late.
    function StepControl(predict, max_growth, landing_steps, dt_min, dt_min_ratio,
                         retries, cfl_backoff, savepoint_interval, floor_ratio,
                         floor_scope)
        landing_steps >= 1 ||
            throw(ArgumentError("StepControl: landing_steps must be >= 1 " *
                                "(1 is a hard clip onto the scheduled time)"))
        # An upper bound as well as a lower one. The floors are fractions of the
        # initial minima, so a ratio at or above 1 would floor the state at its
        # own starting minimum and clamp the physics rather than the failure.
        0 <= floor_ratio < 1 ||
            throw(ArgumentError("StepControl: floor_ratio must be in [0, 1), " *
                                "got $floor_ratio"))
        floor_scope in (:representable, :internal_energy) ||
            throw(ArgumentError("StepControl: floor_scope must be :representable " *
                                "or :internal_energy, got :$floor_scope"))
        new(predict, max_growth, landing_steps, dt_min, dt_min_ratio,
            retries, cfl_backoff, savepoint_interval, floor_ratio, floor_scope)
    end
end

"""
Running count of what the positivity failsafe has seen and repaired, carried on
the solver as `floor_tally` and accumulated over every [`run!`](@ref) call.

- `steps`: steps on which the failsafe saw or repaired at least one cell.
- `cells`: repaired cells, summed over steps and ranks. A cell repaired on
  twenty steps counts twenty times.
- `low_energy`: cells *observed* carrying internal energy below the floor,
  counted the same way. Under `floor_scope = :representable` these are not
  repaired, so on a run whose density and total energy stay positive it and
  `steps` are the only fields that move.
- `mass`: mixture mass added by the density floor.
- `energy`: total energy added, contributed only by the branch that cannot damp
  the velocity because there is no kinetic energy left to convert.
- `momentum`: momentum magnitude removed by the velocity damping.

The three physical quantities are volume-weighted on the same convention as
[`volume_integral`](@ref), so each is directly comparable with the integral of
the field it perturbs. All six are global: `run!` reduces the per-step tally
across the communicator before accumulating it here. Every field stays zero
under the default `StepControl(floor_ratio = 0)`.
"""
mutable struct FloorTally
    steps::Int
    cells::Int
    low_energy::Int
    mass::Float64
    energy::Float64
    momentum::Float64
end

FloorTally() = FloorTally(0, 0, 0, 0.0, 0.0, 0.0)

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

