# The outer run loop: the priming of the first step, the rollback of a
# failed one, `run!`, and the error reporting of an MPI driver. The step it
# takes is in timestep_rk.jl.

# --- The first step of a run -------------------------------------------------
#
# `max_rate` builds its diffusive rate from the artificial coefficient arrays
# as the last right-hand-side evaluation left them, and a freshly built
# solver has had none: sized on the acoustic and advective rates alone, its
# first step ignores whatever the initial data does to the artificial
# properties, and one step is enough to decide a converging shock. Planar
# and cylindrical Noh start with u = -1 against the wall or the axis; sized
# that way the run loses positivity above cfl 0.25 at the wall and 0.2 at the
# axis, and sized from the coefficients the first evaluation produces it
# completes at cfl 0.9 in both geometries with the plateau of the cfl 0.15
# run. The spherical origin's ceiling of 0.3 is set by a later excursion and
# does not move. The rollback recovery that appeared to lift the wall ceiling
# rested on the same effect from the other side: the failed trajectory's
# coefficients, then left in place by the restore, throttled the retry's
# first step. So the first evaluation is made once more here, into the
# workspace scratch the RK accumulator forgets (RKA[1] = 0), before the
# savepoint is taken. A solver past step 0 carries its coefficients already,
# from its last step or from a checkpoint's art block, and is left alone; a
# phase change that switches the properties on primes it through `_prime_art!`
# (phases.jl).
function _prime_coefficients!(solver::Solver, Q, workspace)
    (solver.art.enabled && solver.step == 0) || return solver
    return _prime_art!(solver, Q, workspace)
end
function _prime_art!(solver::Solver, Q, workspace)
    _presync!(solver, Q)
    _prime_rhs!(solver, Q, workspace)
    return solver
end
# `Q` and the workspace are the run's state and stage arrays, or the per-patch
# vectors a composite driver holds.
function _prime_rhs!(solver::Solver, Q, workspace)
    if !_multipatch(solver)
        apply_bcs!(solver, Q)
        compute_rhs!(solver, Q, workspace.dQ, false)
        return Q
    end
    states, dQs = _patch_states(Q), _patch_states(workspace.dQ)
    for lev in getfield(solver, :levels)
        status = _level_rhs!(solver, lev, states, dQs, false)
        _check_transport_status(solver, status)
    end
    return Q
end

# The coefficients `_prime_rhs!` leaves, without the rest of the right-hand
# sides: per level, the boundary conditions and the opening of `compute_rhs!`
# up to its artificial-property step, or the level-wide pass with the
# compression switch after it, as `_level_rhs!` takes them. The fluxes, the
# divergences, the ghost fluxes and the junction captures they would write are
# overwritten by the next step's first evaluation before anything reads them,
# and `max_rate` reads the coefficients, the state and the primitives alone.
function _prime_level_coefficients!(solver::Solver, states::AbstractVector,
                                    mask_child::Bool=true)
    for lev in getfield(solver, :levels)
        status = _prepare_level_transport!(solver, lev, states, false, true,
                                           solver.comm)
        _check_transport_status(solver, status)
        domain = transport_has_domain(solver.transport)
        prepared, enforce = domain, !domain
        if _level_sensors(solver, lev)
            _level_coefficient_pass!(_cold(solver), lev, states, prepared, enforce,
                                     mask_child)
        elseif isempty(lev.stacks)
            _foreach_tile(_enforced_tile_coefficients!, solver, lev, states, prepared,
                          enforce, mask_child)
        else
            patches = getfield(solver, :patches)
            enforce && for pi in lev.patches
                apply_bcs!(PatchSolver(solver, patches[pi]), states[pi])
            end
            for st in lev.stacks
                _coefficient_step!(PatchSolver(solver, st.patch),
                                   _stack_state(st, states), prepared, false,
                                   mask_child)
            end
        end
    end
    return states
end

# The opening of `compute_rhs!` through its artificial-property step, with the
# derivative mask released as the right-hand side's end releases it.
function _coefficient_step!(ps, Q, prepared::Bool, current::Bool,
                            mask_child::Bool=true)
    _gradient_step!(ps, Q, prepared, current)
    _validate_transport_state!(ps, Q; current=true)
    _artificial_step!(ps, Q, current, mask_child)
    _release_child_mask!(ps)
    return nothing
end

# One tile's part of `_prime_level_coefficients!`, with the ledger hooks of
# `_enforced_tile_rhs!`.
function _enforced_tile_coefficients!(solver, pi::Int, states, prepared::Bool,
                                      enforce::Bool, mask_child::Bool=true)
    patches = getfield(solver, :patches)
    if enforce
        _ledger_open!(solver, states, pi)
        _tile_bcs!(solver, _cold(patches[pi]), states, pi)
        _ledger!(solver, states, :wall_enforce, pi)
    end
    _tile_coefficients!(solver, _cold(patches[pi]), states, pi, prepared, mask_child)
    return nothing
end
_tile_coefficients!(solver, p, states, pi::Int, prepared::Bool, mask_child::Bool) =
    _coefficient_step!(PatchSolver(solver, p), states[pi], prepared, false, mask_child)

# The level-wide pass of `_sensor_level_rhs!` and the compression switch each
# tile's right-hand side applies after it.
function _level_coefficient_pass!(solver::Solver, lev::Level, states, prepared::Bool,
                                  enforce::Bool, mask_child::Bool=true)
    patches = getfield(solver, :patches)
    if enforce
        if isempty(lev.stacks)
            _foreach_tile(_enforced_tile!, solver, lev, states)
        else
            for pi in lev.patches
                _tile_bcs!(solver, _cold(patches[pi]), states, pi)
            end
        end
    end
    _level_artificial!(solver, lev, states, prepared, nothing, mask_child)
    _each_unit(_unit_switch!, solver, lev, states)
    return nothing
end
_unit_switch!(ps, Q, ::Nothing) = _coefficient_step!(ps, Q, true, true)

# Abandon the current trajectory, restore the savepoint and lower the CFL,
# returning the new attempt count. Raises `failure` instead when there is no
# savepoint or the retries are spent. Every failure `run!` can recover from
# arrives here, so the state checks, the step checks and the endpoint check
# share one recovery rather than one bypassing it.
function _rollback!(solver, Q, workspace, callback, control, save, failure,
                    attempts, rank)
    (save === nothing || attempts >= control.retries) && throw(failure)
    _restore_state!(Q, save.Q)
    _ledger_rebase!(solver, Q, :rollback)
    _restore_art!(solver, save.art)
    solver.t = save.t
    solver.step = save.step
    # Compounding: the CFL is reduced from its current value and never
    # restored from the savepoint, so three retries give backoff^3.
    solver.cfl *= control.cfl_backoff
    save.guard = failure.step
    # The rate history belongs to the abandoned trajectory; keeping it
    # would have the predictor extrapolate from states that no longer
    # exist. The caller drops dt_seen for the same reason, or the relative
    # floor would immediately fire again against a dt from before the rollback.
    solver.dt_prev = zero(solver.dt_prev)
    solver.rate_prev = zero(solver.rate_prev)
    solver.filter_rate_prev = map(zero, solver.filter_rate_prev)
    # A trajectory that failed NON-FINITE also leaves NaN in the
    # low-storage accumulator, which the restore does not touch and whose
    # usual amnesia via RKA[1] = 0 cannot forget NaN (0.0 · NaN is NaN);
    # it re-failed every retry on a subcycled Sod whose plain restart at
    # the same lowered CFL succeeded. The artificial coefficient arrays
    # were the other such place until the savepoint began carrying them.
    failure.reason in (:nonfinite, :substep_cfl) && _reset_workspace!(workspace)
    # Instants between the savepoint and the failure were visited on a
    # trajectory that no longer exists. Re-arm them so the replacement
    # trajectory visits them too; see `rewind!` for what is and is not
    # rolled back.
    rewind_callbacks!(callback, save.t, save.step)
    attempts += 1
    rank == 0 && @warn "run!: $(failure.reason) at step $(failure.step); " *
                       "rolled back to step $(save.step) and lowered cfl to " *
                       "$(solver.cfl) (retry $attempts of $(control.retries))"
    return attempts
end

# Interface and level consistency before the pre-step reads. A single-patch
# layout has neither and skips this entirely. `restrict = false` omits the
# restriction and keeps the shell imposition; `run!` passes it when the
# levels are as its previous post-step synchronization left them. The
# injection writes parent nodes at least `RESTRICT_MARGIN` from a parent-fed
# face and reads the fine nodes coincident with them. The shell imposition
# writes none of those, and the shared-plane averaging leaves them unchanged,
# since the planes were averaged after the last stage or filter pass, so a
# repeated restriction would write every node's own value.
function _presync!(solver, Q, restrict::Bool=true)
    _multipatch(solver) || return Q
    states = _patch_states(Q)
    _ledger_open!(solver, states)
    sync_patches!(solver, states)
    _ledger!(solver, states, :same_level)
    # `sync_levels!` is the restriction followed by the prolongation.
    if restrict
        restrict_level!(solver, states)
        _ledger!(solver, states, :restrict)
    end
    prolong_level_ghosts!(solver, states)
    _ledger!(solver, states, :shell)
    return Q
end

# Per-step level maintenance, after the state filter: restrict the fine state
# onto the covered coarse region, then re-impose the fine shell from the
# restricted coarse state. No-ops without refinement.
function _post_step!(solver, Q)
    _multipatch(solver) || return Q
    states = _patch_states(Q)
    _ledger_open!(solver, states)
    restrict_level!(solver, states)
    _ledger!(solver, states, :restrict)
    # Under subcycling each deeper parent took its correction after its own
    # step (`_advance_level!`); here the root's, or every level's.
    nlev = length(getfield(solver, :levels))
    for ℓp in (getfield(solver, :subcycle) ? (1:min(1, nlev - 1)) : (1:nlev-1))
        _reflux_apply!(solver, states, ℓp)
    end
    _ledger!(solver, states, :reflux)
    prolong_level_ghosts!(solver, states)
    _ledger!(solver, states, :shell)
    return Q
end

"""
    run!(solver, Q; tfinal, nmax=typemax(Int), callback=nothing, control=solver.control)
    run!(solver, Q, workspace; tfinal, ...)

Advance until `solver.t` reaches `tfinal` or `solver.step` reaches `nmax`,
filtering the conserved variables every `solver.filter_interval` steps and
invoking `callback` after each step. Returns `Q`, which is advanced in place.

`tfinal` and `nmax` are absolute: `tfinal` is a value of the solver clock
`solver.t` and `nmax` a value of the step counter `solver.step`, both counted
from the solver's construction (or from the checkpoint it was loaded from),
not from this call. A second `run!` on the same solver therefore continues
from where the first stopped, and takes `nmax = solver.step + n` for `n`
more steps. An `ArgumentError` is raised for a `tfinal` that is NaN or behind
`solver.t` and for a negative `nmax`. An `nmax` at or below a nonzero
`solver.step` while `solver.t` is short of `tfinal` takes no step and logs a
warning on rank 0. `tfinal == solver.t` returns at once and `nmax = 0` on a
new solver takes no step, which checks a deck without advancing it; both still
validate the state, and at step 0 they run the initial-state callbacks.

A continuing call starts from the solver as the last one left it: the lowered
`solver.cfl` of any retries, the step and rate history `dt_prev` and
`rate_prev` read by `control.predict` and `control.max_growth`, and the
artificial coefficients of the last right-hand side, which size the first
step (only a solver at step 0 evaluates the right-hand side once before its
first step to form them). The positivity floors, the `dt_min_ratio`
reference and the rollback savepoint are formed afresh from the state
entering each call. The same holds after [`initialize!`](@ref) has written a
new state into a solver that has already stepped: the clock, the counters,
the rate history and the coefficients are those of the old state.

The first form allocates a [`Workspace`](@ref) per call; pass `workspace` (as
the third positional argument or the keyword of the same name) to reuse one.

`callback` may be a bare `callback(solver, Q)` invoked every step (the original
contract, return value ignored), a [`Callback`](@ref) pairing a trigger with an
effect, or a tuple of those. A scheduled [`AtTime`](@ref) or [`EveryTime`](@ref)
trigger shortens `dt` over the preceding `control.landing_steps` steps to end a
step at the scheduled instant without an arbitrarily small final step. Starting
from step 0, a callback whose trigger is due at the initial time (see
[`fires_at_start`](@ref)) runs once before the first step. An
effect returning `true` ends the run after that step. `tfinal` uses a direct
clip, so scheduled callbacks do not alter the step sequence of a run that has
none.

`control` is a [`StepControl`](@ref) governing timestep prediction, the floors
below which the run is declared failed, and whether a failure is recoverable by
rolling back and lowering the CFL. It defaults to the one the solver was built
with. On an unrecoverable failure this throws [`SolverFailure`](@ref) and does
not continue with a collapsed timestep; the note at the top of
`stepcontrol.jl` records what that failure mode looks like and which of the
three mechanisms was measured to help.

## State validity

The state entering this call and the state it returns are both validated under
`control.validity`, and `control.validity_interval` adds the state entering
every nth step. The returned state is checked at each of the three ways a run
can end: reaching `tfinal`, reaching `nmax`, and a callback effect returning
`true`. A rejection is a `SolverFailure(:invalid_state)` handed to the same
rollback the step checks use, so `control.retries` recovers from it by
restoring the savepoint and lowering the CFL rather than raising past that
recovery. A run whose physics legitimately visits inadmissible states selects
`validity = :permissive`, which accepts and reports them, as the cold-ambient
Noh and Sedov cases in `test/cases.jl` do.

Recovery from a rejected endpoint costs the trajectory. The rollback restores
the last savepoint, so the run repeats every step from there, and it does so
once per retry. A state that is inadmissible because the run's physics ends
that way is inadmissible at any CFL, so such a run spends its whole retry
budget re-integrating before failing. That is a reason to select
`:permissive` for a case known to end that way rather than to rely on the
retries absorbing it.

When `control.retries > 0`, the CFL is lowered in place on each retry.
Consequently, `solver.cfl` after a completed run records the value used to
complete the calculation and can be supplied to the next run's `Numerics`.

When `control.floor_ratio > 0`, `apply_positivity_floor!` inspects the conserved
state after each completed step and repairs it as far as `control.floor_scope`
allows, against floors `positivity_floors` derives once from the state this call
starts with. The first firing warns, the totals warn again when the run ends,
and `solver.floor_tally` carries them either way.

## The endpoint and the clock

The clock `solver.t` has the solver's element type, and `tfinal` is converted
to it before anything is compared against it. The run therefore ends at the
value of that type nearest `tfinal`, within half the floating-point spacing
there: exactly `tfinal` in a Float64 solver given a Float64 endpoint, and
within 6e-8 relative in a Float32 one. Comparing against an unconverted
endpoint instead leaves a remainder no step can close, since adding it to the
clock returns the clock. The same conversion applies to a scheduled callback
instant, and [`AtTime`](@ref) and [`EveryTime`](@ref) measure their landing
tolerance in the clock's precision, so a scheduled instant is landed on in
either precision.

The step is tested for progress once every clip has been applied, on the stored
result of `solver.t + dt`. A step that leaves the clock where it was, having
been shortened by neither the endpoint nor a scheduled instant, raises
[`SolverFailure`](@ref) with reason `:no_progress`: the calculation has reached
a timestep its own clock cannot resolve, which no retry can improve, and the
run stops there rather than exhausting `nmax` at no advance. A landing step
that fails the same test is discarded in favour of the full step, the instant
being inside the trigger's tolerance already.

Each step updates `solver.t`, `solver.step`, `solver.dt_prev`,
`solver.rate_prev`, `solver.filter_rate_prev`, `solver.tstage`,
`solver.wall_step` and
`solver.wall_total`; a rolled-back iteration records no wall time. The loop is
collective through [`max_rate`](@ref) and the line solves beneath
[`step!`](@ref), so every rank must call it with the same `tfinal`, `nmax` and
callbacks.
"""
function run!(solver::Solver, Q, workspace::Workspace;
              tfinal, nmax::Int=typemax(Int), callback=nothing,
              control::StepControl=solver.control)
    # The loop is reached through a barrier. A caller compiled around `setup`,
    # such as a script's `simulate` function or an `mpi_main` block, holds the
    # partially typed solver `setup` infers to, and inference of that caller
    # otherwise walks the whole loop at abstract types before it runs a line:
    # 2.1 s of a 4 s first call of a one-dimensional run, all of it discarded
    # once the call dispatches on the concrete solver. Behind the barrier the
    # loop is inferred only for the concrete types it is called with, at the
    # cost of one dynamic dispatch per call.
    _check_not_released(solver)
    _cold(_run!)(solver, Q, workspace, tfinal, nmax, callback, control)
    return Q
end

# The loop takes the callback unspecialized, so that it is compiled once per
# solver type and not again for each script's callbacks, whose types are its
# own closures. It reaches them through two dynamic calls per step, which
# allocate nothing (see `_step_callbacks`).
function _run!(solver::Solver, Q::SimulationState, workspace::Workspace, tfinal,
               nmax::Int,
               @nospecialize(callback), control::StepControl)
    rank = MPI.Comm_rank(solver.comm)
    instant = Ref(solver.t)
    # The endpoint is carried in the solver's own time type from here on. A
    # `tfinal` the clock cannot represent is not reachable by any sequence of
    # steps, and comparing against the unconverted value leaves a remainder
    # that no step can close: a Float32 run given `tfinal = 0.7` stopped
    # advancing at 0.699999988079071 and took a 1.1920929e-8 remainder for
    # every remaining step of its `nmax`.
    tfin = oftype(solver.t, tfinal)
    _check_run_limits(solver, tfinal, tfin, nmax)
    _prime_coefficients!(solver, Q, workspace)
    # Off by default (`retries = 0`), so behind `_cold`: `save` is then
    # untyped, and read only on the rollback and savepoint paths, both cold.
    save = control.retries > 0 ? _savepoint(_cold(solver), Q) : nothing
    attempts = 0
    dt_seen = 0.0
    rho_floor, e_floor = positivity_floors(solver, Q, control)
    # The positivity limiter's bounds, from the same state; false without the
    # limiter. Reduced, so every rank takes the limited path or none does.
    limiting = _positivity_setup!(solver, Q)
    # The floors come from the state entering the run, so they are derived
    # before it is validated: a repair mode needs scales from a state that was
    # still valid, and this is the last point at which that is known.
    # `validate_state!` with its default floors and warning, called positionally.
    _, failure_0 = _apply_validity!(solver, Q, control, "the state entering run!",
                                    (0.0, 0.0), true)
    failure_0 === nothing || throw(failure_0)
    _validate_transport_state!(solver, Q)
    if control.floor_ratio > 0 && rho_floor <= 0
        rank == 0 && @warn "run!: the positivity failsafe is inactive. The global " *
                           "minimum density or internal energy of the state " *
                           "entering this run is not positive, so floor_ratio " *
                           "has nothing to scale."
    end
    # Snapshot of the cumulative tally, so the summary below reports this
    # call's own repairs, not everything the solver has accumulated.
    ft0 = solver.floor_tally
    floor_0 = (steps=ft0.steps, cells=ft0.cells, low_energy=ft0.low_energy,
               mass=ft0.mass, energy=ft0.energy, momentum=ft0.momentum)
    # `save.guard` suppresses a new savepoint at or below the failing step after
    # rollback. Without the guard, the retry replaces the valid savepoint with
    # the state that failed. Each subsequent retry then restores that state and
    # changes only the CFL. The observed sequence rolled back to step 180 and
    # failed again at step 180 four times. `regrid!` observes the same guard.
    # `stopped` carries a callback's request to end the run to the endpoint
    # check below rather than out of the loop, so that a callback exit is
    # validated on the same terms as reaching `tfinal` or `nmax`.
    stopped = false
    # The initial state, after it has been validated and its artificial
    # coefficients primed, so an output of either at t = 0 reads what the first
    # step starts from. A restarted solver (step > 0) has written its initial
    # frames already.
    solver.step == 0 && _start_callbacks(callback, solver, Q)::Bool &&
        (stopped = true)
    # Whether the levels are as the previous iteration's post-step
    # synchronization left them, so that the pre-step restriction would repeat
    # it (see `_presync!`). False entering the call, since the state may have
    # been written since the last one, and cleared by everything below that
    # can write the state between the two: a regrid check, a rollback, a
    # validity or positivity repair, and a callback effect. Every input is
    # replicated or reduced, so the collective restriction is taken or skipped
    # on every rank together. The filtered restriction reads the imposed
    # shell through its whole-patch line solve and is never skipped.
    levels_synced = false
    restrict_repeats = getfield(solver, :schemes).level_restriction === :inject
    while true
        if stopped || !(solver.t < tfin && solver.step < nmax)
            _validate_transport_state!(solver, Q)
            # The state this run is about to return. Checking it here rather
            # than after the loop keeps it on the retry path: a rejection rolls
            # back to the savepoint and lowers the CFL like any other failure,
            # so the endpoint cannot deliver a state that bypassed recovery.
            _ledger_open!(solver, Q)
            _, failure = _apply_validity!(solver, Q, control,
                                          "the state run! returns",
                                          (rho_floor, e_floor), true)
            _ledger!(solver, Q, :repair)
            failure === nothing && break
            attempts = _rollback!(_cold(solver), Q, workspace, callback, control,
                                  save, failure, attempts, rank)::Int
            dt_seen = 0.0
            stopped = false
            levels_synced = false
            continue
        end
        # Timed from here, not around step! alone: max_rate carries the
        # per-step Allreduce and the filter is a full set of line solves, so both
        # are step cost a user is trying to see. Callbacks are outside it, since
        # a progress callback that reduces a diagnostic would otherwise time
        # itself and report that as solver cost.
        wall_0 = time_ns()
        solver.wall_wait = 0.0
        # The regrid is attributed by the first hook after it, which sees
        # the regrid check counter move and rebases every patch.
        _ledger_open!(solver, Q)
        _maybe_regrid!(solver, Q, workspace, save, control,
                       (rho_floor, e_floor)) && (levels_synced = false)
        _presync!(solver, Q, !levels_synced)
        levels_synced = false
        # Boundary conditions before the rate measurement, for two reasons. The
        # step should be sized from the state it is about to advance, and the
        # previous iteration's filter_state! has smeared whatever the conditions
        # impose on the edge planes. With Q settled here, max_rate's halo exchange
        # and primitives pass also serve stage 1, as `prepared` below
        # asserts; that removes one sixth of both from the per-step cost.
        solver.tstage = solver.t
        _ledger_open!(solver, Q)
        apply_bcs!(solver, Q)
        _ledger!(solver, Q, :wall_enforce)
        rate, rho_min, filter_rate = max_rate(solver, Q)
        dt = predicted_dt(solver, control, rate)
        failure = check_step(control, dt, rho_min, dt_seen, solver.step,
                             solver.t, solver.cfl)
        # The state entering this step, on the requested cadence. The sweep
        # calls the EOS at every interior point, so it is off unless asked for;
        # `check_step`'s reduced scalars are the cheap check that always runs.
        if failure === nothing && control.validity_interval > 0 &&
           solver.step % control.validity_interval == 0
            repaired_0 = solver.floor_tally.cells
            _ledger_open!(solver, Q)
            _, failure = _apply_validity!(solver, Q, control,
                                          "the state entering the step",
                                          (rho_floor, e_floor), true)
            _ledger!(solver, Q, :repair)
            # A `:repair` policy rewrites interior points after `max_rate` has
            # read them, which leaves the halos, the boundary values and the
            # primitives that `prepared` asserts behind the state. The tally is
            # reduced, so every rank takes this branch together.
            if failure === nothing && solver.floor_tally.cells != repaired_0
                _presync!(solver, Q)
                _ledger_open!(solver, Q)
                apply_bcs!(solver, Q)
                _ledger!(solver, Q, :wall_enforce)
                rate, rho_min, filter_rate = max_rate(solver, Q)
                dt = predicted_dt(solver, control, rate)
                failure = check_step(control, dt, rho_min, dt_seen, solver.step,
                                     solver.t, solver.cfl)
            end
        end
        if failure !== nothing
            attempts = _rollback!(_cold(solver), Q, workspace, callback, control,
                                  save, failure, attempts, rank)::Int
            dt_seen = 0.0
            continue
        end
        # Q has passed its health check, the only point at which it is
        # known good: the checks run on the state entering a step, so saving
        # after stepping would bank a state nothing has yet vetted.
        if save !== nothing && control.savepoint_interval > 0 &&
           solver.step > save.guard && solver.step % control.savepoint_interval == 0
            _bank_savepoint!(_cold(solver), save, Q)
        end
        dt_seen = max(dt_seen, dt)
        # Clip to the endpoint AFTER the checks, and to the next scheduled
        # callback time as well: an AtTime trigger has to be landed on exactly,
        # not overshot, and nothing outside run! can reach dt to arrange that.
        # The gap is only applied when it is positive: a requested time behind
        # solver.t would otherwise drive dt to zero or negative, which stalls
        # the run and fires nothing.
        dt = min(dt, tfin - solver.t)
        # An instant beyond `tfinal` is not reachable in this run, and aiming at
        # one makes the soft landing below halve the step against a target it
        # never reaches. For example, `EveryTime(0.003)` run to
        # `tfinal = 0.009` schedules its third instant at `0.006 + 0.003`, which
        # is one ULP ABOVE the 0.009 literal, so every step from there on had
        # `gap` a shade over `tfinal - solver.t` and the division by
        # `ceil(gap/dt) == 2` repeatedly halved dt: 1.9e-13, 9.7e-14, 4.9e-14,
        # and so on for forty steps until `solver.t + dt` rounded to `tfinal`.
        # The run reached the endpoint and fired the trigger, but the existing
        # checks did not detect the additional steps.
        #
        # The instant is discarded, not clamped to `tfinal`, which keeps the
        # endpoint out of the soft landing, per the note below. `tfinal` is
        # still clipped to, one line up; it is never subdivided toward.
        #
        # The schedule is kept in Float64 and the clock may be narrower, so the
        # instant is converted before the arithmetic below, exactly as `tfinal`
        # is. The trigger's landing tolerance is measured in the clock's
        # precision for the same reason (`_land_tol` in callbacks.jl).
        _next_instant!(instant, callback, solver)
        next_instant = instant[]
        gap = next_instant - solver.t
        # Soft landing. Clipping directly to the gap lands exactly but leaves an
        # arbitrarily small step before a scheduled instant: a dump every 1e-4
        # against a CFL step of 3.7e-5 gives steps of 3.7, 3.7, 3.7, 0.15 e-5.
        # That wastes a step and, with `max_growth` enabled, throttles the
        # several that follow. Dividing the remaining gap into `ceil(gap/dt)`
        # equal steps lands equally exactly with no step below dt/2.
        #
        # This applies to scheduled callback times only, not to `tfinal`. At the
        # endpoint no following step remains to be affected, and restricting the
        # change here leaves a run without scheduled callbacks stepping as it did
        # before, which is the condition the validation guards were measured
        # under.
        if next_instant <= tfin && gap > 0 && gap < dt * control.landing_steps
            landing = gap / ceil(gap / dt)
            # A landing step below the resolution of the clock would stall the
            # run against an instant it cannot separate from `solver.t`. The
            # instant is inside the trigger's landing tolerance there, so the
            # trigger fires on the full step instead.
            _advances(solver.t, landing) && (dt = landing)
        end
        # Progress, after every clip: the floors, the endpoint and the landing
        # are all applied above, so `dt` here is the step the clock is asked to
        # take.
        if !_advances(solver.t, dt)
            # The endpoint clip is the binding one, so `solver.t` is the closest
            # the clock comes to `tfin` and the run has arrived.
            dt >= tfin - solver.t && break
            throw(SolverFailure(:no_progress, solver.step, solver.t, dt,
                                solver.cfl,
                                "the step does not advance the clock: t + dt " *
                                "is t in $(typeof(solver.t)) arithmetic, " *
                                "whose spacing at this t is $(eps(solver.t))"))
        end
        prepared = true         # see the apply_bcs!/max_rate note above
        failure = limiting ?
                  _limited_run_step!(_cold(solver), Q, workspace, dt, prepared,
                                     control)::Union{Nothing,SolverFailure} :
                  _run_step!(solver, Q, workspace, dt, prepared, control)
        if failure !== nothing
            attempts = _rollback!(_cold(solver), Q, workspace, callback, control,
                                  save, failure, attempts, rank)::Int
            dt_seen = 0.0
            continue
        end
        solver.t += dt
        solver.step += 1
        solver.dt_prev = dt
        solver.rate_prev = rate
        solver.filter_rate_prev = filter_rate
        if solver.filter_interval > 0 && solver.step % solver.filter_interval == 0
            _ledger_open!(solver, Q)
            limiting ? _limited_filter_state!(_cold(solver), Q) : filter_state!(solver, Q)
            _ledger!(solver, Q, :filter)
        end
        # Once per step, after the filter and before the failsafe; see
        # `truncate_modes!` for why the halos may stay stale here.
        if _truncating(solver.truncation)
            _ledger_open!(solver, Q)
            truncate_modes!(solver, Q)
            _ledger!(solver, Q, :truncation)
        end
        _post_step!(solver, Q)
        levels_synced = restrict_repeats
        # After the filter, not immediately after step!, ensuring the state
        # entering the next iteration's checks is the repaired one whichever of
        # the two damaged it. The compact filter is not monotone, so it can
        # produce sub-floor values itself. The repaired count is reduced.
        _ledger_open!(solver, Q)
        rho_floor > 0 && _positivity_failsafe!(_cold(solver), Q, rho_floor,
                                                e_floor, control, floor_0,
                                                rank)::Bool && (levels_synced = false)
        _ledger!(solver, Q, :repair)
        # A rollback `continue`s above this, so an abandoned iteration never
        # records a step time; wall_total counts work that stood.
        _validate_transport_state!(solver, Q)
        solver.wall_step = (time_ns() - wall_0) / 1e9
        solver.wall_total += solver.wall_step
        solver.wait_total += solver.wall_wait
        _ledger_open!(solver, Q)
        flags = _step_callbacks(callback, solver, Q)::Int
        _ledger!(solver, Q, :callback)
        flags & 1 != 0 && (stopped = true)
        flags & 2 != 0 && (levels_synced = false)
    end
    ft = solver.floor_tally
    if ft.steps > floor_0.steps && rank == 0
        repaired = ft.cells - floor_0.cells
        @warn "run!: the positivity failsafe was active on " *
              "$(ft.steps - floor_0.steps) of this run's steps, over which it " *
              "saw $(ft.low_energy - floor_0.low_energy) cell(s) below the " *
              "internal-energy floor and repaired $repaired. Mass added " *
              "$(ft.mass - floor_0.mass), energy added " *
              "$(ft.energy - floor_0.energy), momentum removed " *
              "$(ft.momentum - floor_0.momentum)." *
              (repaired > 0 ? " A repaired cell makes this a repaired " *
                              "trajectory rather than a converged one." : "")
    end
    return Q
end

# The endpoint and the step cap against the solver's clock, which every rank
# holds identically, so a rejection is raised everywhere.
function _check_run_limits(solver, tfinal, tfin, nmax)
    isnan(tfinal) && throw(ArgumentError("run!: tfinal must be a number, got NaN"))
    tfin < solver.t &&
        throw(ArgumentError("run!: tfinal = $tfinal is behind the solver clock " *
                            "t = $(solver.t); the clock is not reset between runs"))
    nmax >= 0 || throw(ArgumentError("run!: nmax must be >= 0, got $nmax"))
    # A second `run!` given the first one's `nmax` returns at once. A call that
    # takes no step can be intended (a deck check), so this warns and runs.
    if solver.step > 0 && nmax <= solver.step && solver.t < tfin &&
       MPI.Comm_rank(solver.comm) == 0
        @warn "run!: nmax = $nmax does not exceed solver.step = $(solver.step), " *
              "so no step is taken. nmax counts the solver's steps since " *
              "construction, not this call's; pass nmax = solver.step + n for " *
              "n more steps"
    end
    return nothing
end

# The clock is held in Float64 whatever the element type: a Float32 time
# widens exactly and narrows back to itself on a rollback.
_savepoint(solver, Q) =
    Savepoint(_snapshot(Q), _art_snapshot(solver), Float64(solver.t), solver.step, -1)

# The savepoint bank and the positivity failsafe of `run!`, behind `_cold`:
# each runs only under a `StepControl` setting that is off by default.
function _bank_savepoint!(solver, save, Q)
    _restore_state!(save.Q, Q)
    _bank_art!(save.art, solver)
    save.t = solver.t
    save.step = solver.step
    return nothing
end

function _positivity_failsafe!(solver, Q, rho_floor, e_floor, control, floor_0,
                               rank)
    tally = apply_positivity_floor!(solver, Q, rho_floor, e_floor,
                                    control.floor_scope;
                                    species_band=control.species_band)
    if tally.cells > 0 || tally.low_energy > 0
        ft = record_floor!(solver, tally)
        ft.steps == floor_0.steps + 1 && rank == 0 &&
            @warn "run!: the positivity failsafe saw $(tally.low_energy) " *
                  "cell(s) below the internal-energy floor and repaired " *
                  "$(tally.cells) at step $(solver.step), t = $(solver.t). " *
                  "Later steps are counted in solver.floor_tally and " *
                  "summarized when this run ends."
    end
    return tally.cells > 0
end

function run!(solver::Solver, Q; workspace=nothing, kwargs...)
    _check_not_released(solver)
    work = workspace === nothing ? Workspace(Q) : workspace
    return run!(solver, Q, work; kwargs...)
end

# --- Top-level error reporting under many ranks.
#
# An uncaught exception is reported by every rank. A `SolverFailure` at 448
# ranks therefore produces 448 stacktraces and several thousand lines of
# repeated output. Because the failures handled here arise from collective
# quantities (`max_rate` reduces, `check_step` reads the reduced result), they
# are usually identical on every rank, 447 of those copies are redundant.
#
# One backtrace, from rank 0, plus a one-line identification from anywhere else
# that failed; the latter identifies rank-local errors such as `plan_direction`
# rejecting one rank's block. `MPI_Abort` then terminates ranks still blocked in
# a collective before each prints a separate signal-handler dump.

"""
    mpi_main(body; comm = MPI.COMM_WORLD, exitcode = 1, grace = 0.5)

Run `body()` under a top-level error guard for large rank counts. Rank 0 prints a
full backtrace, other failing ranks print one line, and the function then calls
`MPI.Abort(comm, exitcode)`, which does not return. The return value on the
successful path is that of `body`. A failing rank other than rank 0 sleeps
`grace` seconds before aborting to keep the abort from truncating rank 0's
backtrace; see the comment in the body.

MPI drivers should use this guard to prevent an uncaught exception from
producing a full backtrace on every rank. For interrupts, pass
`--handle-signals=no` to `julia`; this function cannot intercept Julia's signal
handler while a rank is inside an MPI call.

Set `CL_ERROR_BACKTRACE=all` for a rank-local failure when rank 0 may remain
blocked in a collective and cannot print the primary backtrace.

```julia
mpi_main() do
    run!(solver, Q; tfinal = 1.0, callback = ProgressLog())
end
```
"""
function mpi_main(body; comm::MPI.Comm=MPI.COMM_WORLD, exitcode::Int=1,
                 grace::Float64=0.5)
    try
        return body()
    catch err
        rank = MPI.Comm_rank(comm)
        # One write, not several. Concurrent ranks writing a line in pieces
        # interleave character-by-character, and the result is unreadable well
        # long before 448 ranks are involved.
        # A failure on one rank alone leaves rank 0 blocked in a collective, so
        # no backtrace is printed and the one-line message has no location. Set
        # CL_ERROR_BACKTRACE=all to get one from every failing rank: the right
        # setting for chasing a rank-local error, and the wrong one at 448 ranks
        # when they all fail together.
        all_bt = get(ENV, "CL_ERROR_BACKTRACE", "") == "all"
        if rank == 0 || all_bt
            print(stderr, "\nrank " * string(rank) * ": " *
                          sprint(showerror, err, catch_backtrace()) * "\n")
        else
            print(stderr, "rank " * string(rank) * ": " *
                          sprint(showerror, err) * "\n")
        end
        flush(stderr)
        # The first `MPI_Abort` terminates every rank and may truncate rank 0's
        # backtrace. Nonzero ranks therefore wait briefly before aborting. If
        # rank 0 did not fail, it remains blocked in a collective; the delay then
        # adds only `grace` seconds before termination.
        rank == 0 || sleep(grace)
        MPI.Abort(comm, exitcode)
    end
end
