# Timestep control policy: floors, prediction, and recovery.
#
# Separated from timestep.jl only because `Solver` carries a `StepControl` field
# and therefore has to see the type first. The logic that needs a Solver,
# `predicted_dt` and `run!`, stays with the integrator.

# When the artificial properties do not stabilize a strong shock, the state
# loses positivity, the diffusive term in the CFL rate rises, and `dt`
# collapses. Without these checks, the step counter can continue increasing
# while simulated time advances negligibly.
#
# --- Observed collapse -------------------------------------------------------
#
# `compute_dt` uses the previous step's artificial coefficients, so a forming
# shock is initially measured with coefficients one step out of date. Linear
# rate extrapolation (`predict` below) was implemented to test whether this lag
# causes the collapse. It does not. For Noh at nu=1, N=400, and cfl=0.3, failure
# occurs at step 175 without lookahead and step 179 with three-step lookahead;
# thirty-step lookahead postpones failure by 25 steps. The measured trace is:
#
#     step   25  dt=2.1e-4  rate=1418   rho_min=0.951
#     step   75  dt=1.8e-4  rate=1633   rho_min=0.798
#     step  125  dt=1.8e-4  rate=1718   rho_min=0.398
#     step  175  dt=4.8e-5  rate=1.5e4  rho_min=0.252   <- and then negative
#
# The exact pre-shock density is 1. The computed density is 5% low by step 25
# and 60% low by step 125 while `dt` and the rate remain nearly constant. A
# dispersive undershoot at the shock is not sufficiently damped by the
# artificial viscosity. Positivity fails near step 175, after which the rate
# rises sharply. The timestep collapse therefore follows the first measurable
# state error by about 150 steps.
#
# The four mechanisms address different parts of this sequence:
#
#   1. The floors and positivity check raise a diagnosed exception about 150
#      steps earlier than a `dt`-only test. They are enabled by default.
#   2. The predictor is retained, off by default, with the measurement above
#      recorded to prevent duplicate work. It changes the step sequence of
#      stable runs without a measured benefit.
#   3. Retry restores a savepoint and lowers the CFL. In Noh at nu=1, rollback
#      from cfl 0.9 to 0.45 completes in 1433 steps with plateau 0.9989. A run
#      fixed globally at cfl 0.15 requires 4485 steps for the same result. The
#      lower CFL is required during shock formation but not after the shock has
#      formed, which a rollback can represent without reducing every step.
#   4. The state floor repairs the state, where the first three detect failure,
#      and is disabled by default. The first three mechanisms read reduced
#      scalars. `bench/nohprobe.jl` measured six to eight interior cells with
#      negative internal energy moving with the front throughout Noh runs that
#      otherwise complete and pass their guards. `primitives!` floors `T_ion`
#      at 1e-300 where `e <= 0`, while the positivity check reads density, which
#      remains positive. `floor_ratio` counts and optionally repairs those
#      cells. See `apply_positivity_floor!`.
#
# --- Repair scope ------------------------------------------------------------
#
# Those negative-e cells are not a rounding artifact, and repairing them is not
# cheap. Over the complete Noh nu=1 validation case at cfl 0.15, which takes 3724
# steps and returns a plateau of 3.997 against the exact 4, the state carries
# 24991 cell-steps of e < 0, peaking at 8 cells at once and reaching -718 e0.
# Total energy density and mixture density stay positive at every cell of every
# step. The wall region runs as a pressureless layer, and the scheme remains
# stable there for the whole run.
#
# Raising e above zero requires 5% velocity damping on the worst cell after
# a single step (-337 e0; the next four need 0.7%, 0.08%, 0.05%, 0.02%). Applied
# every step it removes 1.5e-4 of the total momentum on step 1, produces cells
# of negative *total* energy by step 4 that the unrepaired trajectory never
# produces, and the run then fails at step 18 with :dt_collapse. Raising E
# and damping the velocity are the same order of intervention, about 11%
# of that cell's energy, so the cost belongs to the case, not to the choice of
# repair.
#
# These measurements determine `floor_scope`. The default repairs only states
# whose total energy is below `ρ * e_floor`; no such state occurs in this
# case. It still counts negative internal energy for reporting.
# `:internal_energy` additionally repairs those cells when a bounded trajectory
# is required despite the measured change to the solution.
#
# These checks do not detect a run that completes with a wrong answer. Noh
# nu=1 at cfl=0.5 reaches t_final without tripping anything and returns a
# plateau 2% of the exact value. Floors detect the specified invalid states,
# not general solution error.

"""
Planck time in seconds, the unconditional failsafe floor on `dt`. `check_step`
rejects any step not strictly greater than it whatever the rest of a
[`StepControl`](@ref) specifies, so for a calculation expressed in physical units a
step this small ends the run as numerical collapse.

The comparison is made against `dt` in whatever units the calculation carries.
In a nondimensional calculation 5.4e-44 is a numerical failsafe, not a
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

- `predict = 0.0`: steps of linear extrapolation of the CFL rate, compensating
  for `compute_dt` seeing the artificial coefficients one step late. The
  extrapolation is one-sided and only ever raises the rate, so it can shorten a
  step but never lengthen one. Measured: no lookahead between 0 and 30 steps
  prevents the Noh failure at cfl = 0.3.
- `max_growth = 0.0`: largest multiple of the previous step that `dt` may take;
  0 disables the cap. A value of 1.05 delayed the same failure from step 175 to
  186 but did not prevent it.

## Landing on a scheduled instant

- `landing_steps = 2`: how many steps ahead of a scheduled callback time
  ([`AtTime`](@ref), [`EveryTime`](@ref)) `run!` may start shortening the step
  to end a step at that instant. The remaining interval is divided into
  equal steps, with no step below `dt/2`. A value of 1 applies a hard clip and
  can produce an arbitrarily small step immediately before an output time.
  Larger values distribute the adjustment over more steps while operating
  farther below the CFL limit. A value below 1 is rejected at construction.
  `tfinal` always uses a hard clip.

## Floors

- `dt_min = 0.0`: absolute floor on `dt`; 0 disables it. The unconditional
  internal `PLANCK_TIME` floor and the relative floor below apply either way.
- `dt_min_ratio = 1e-8`: floor relative to the largest `dt` taken so far in
  the current `run!` call; 0 disables it. This detects collapse without prior
  knowledge of the problem's units or scales. Physical variation in `dt` is
  generally a few orders of magnitude, whereas a collapse spans ten or more in
  the measured cases. A rollback resets the reference, so after a retry the
  floor measures the replacement trajectory alone.

## Recovery

- `retries = 0`: number of times to roll back to the last savepoint and retry
  with a reduced CFL; 0 disables recovery. In the validation cases,
  `retries = 4` recovers every Noh geometry from `cfl = 0.9` in fewer steps than
  running the complete calculation at the lower stable CFL.
- `cfl_backoff = 0.5`: multiplier applied to the solver's current CFL on each
  retry, so successive retries compound.
- `savepoint_interval = 25`: steps between savepoints, counted on the solver's
  step number. The savepoint costs one extra state array and is allocated only
  when `retries > 0`; a value of 0 or less leaves the state on entry to `run!`
  as the only rollback target.

Rollback can recover an abrupt failure caused by an excessive initial CFL. It
does not recover gradual degradation when the most recent savepoint itself
contains a loss of positivity. Such cases require a lower initial CFL, a larger
`savepoint_interval` that places the savepoint earlier in the trajectory, or the
state floor below.

## The positivity failsafe

- `floor_ratio = 0.0`: inspect the conserved state after every completed step
  and repair it where it has left the physical state space; 0 disables the
  failsafe entirely. The two floors are derived from the state entering
  [`run!`](@ref), as `floor_ratio` times its global minimum mixture density and
  its global minimum internal energy, so the setting carries no units and needs
  no advance knowledge of the problem's scales, on the same reasoning as
  `dt_min_ratio` above.
- `floor_scope = :representable`: how much of that state space to insist on.
  `:representable` repairs a mixture density below the floor and a total energy
  density below it, and *counts* internal energy below the floor without
  touching it. `:internal_energy` additionally repairs the internal energy, by
  the velocity damping `apply_positivity_floor!` describes.

The distinction rests on measurement; the comment at the top of
`stepcontrol.jl` carries the numbers. Negative internal energy is a persistent
feature of converging-shock runs that complete and pass their guards, where
density and total energy stay positive throughout; repairing it every step is a
percent-level intervention that terminates such a run. `:representable`
therefore makes an existing calculation observable without changing it, while
`:internal_energy` suits sweeps and development runs where the goal is a bounded
run, not a faithful trajectory.

Under either scope, every firing is counted in `solver.floor_tally`
([`FloorTally`](@ref)) and reported. Enabling the failsafe removes
`:negative_density` as a route to [`SolverFailure`](@ref), since the density
that check reads can no longer reach zero; `:dt_collapse` and `:nonfinite` still
terminate a diverging run after the corresponding threshold is crossed. The
repair changes states produced by the scheme but does not change the scheme.
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
    # `landing_steps = 0` does not disable the shortening. The gap clip
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
        # own starting minimum, clamping the physics along with the failure.
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

- `steps`: steps on which the failsafe detected or repaired at least one cell.
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
        print(io, "\n  One possible cause is loss of positivity near a strong shock. ",
                  "`StepControl(retries = 4)`\n  restores a savepoint and lowers ",
                  "the CFL after this failure, while retaining the initial CFL ",
                  "after\n  a successful startup. See reference/CALIBRATION.md.")
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
    guard::Int   # step at or below which re-banking is suppressed after a
                 # rollback; -1 when none (see `run!`)
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

