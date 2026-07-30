# Time integration: five-stage fourth-order low-storage Runge–Kutta
# (Carpenter & Kennedy 1994), CFL-limited timestep, and the outer run loop
# with per-step conservative-variable filtering.

const RKA = (0.0,
             -567301805773.0 / 1357537059087.0,
             -2404267990393.0 / 2016746695238.0,
             -3550918686646.0 / 2091501179385.0,
             -1275806237668.0 / 842570457699.0)

const RKB = (1432997174477.0 / 9575080441755.0,
             5161836677717.0 / 13612068292357.0,
             1720146321549.0 / 2090206949498.0,
             3134564353537.0 / 4481467310338.0,
             2277821191437.0 / 14882151754819.0)

const RKC = (0.0,
             1432997174477.0 / 9575080441755.0,
             2526269341429.0 / 6820363962896.0,
             2006345519317.0 / 3224310063776.0,
             2802321613138.0 / 2924317926251.0)

"""
    predicted_dt(solver, control, rate) -> dt

Turn a measured CFL rate into the step to take: extrapolate the rate forward by
`control.predict` steps, then cap the growth against the previous step.
"""
function predicted_dt(solver::Solver, control::StepControl, rate)
    r = rate
    if control.predict > 0 && solver.rate_prev > 0
        # Linear extrapolation, one-sided: only ever raise the rate. A falling
        # rate means the flow is relaxing, and stepping out on that prediction
        # is how you overshoot the next shock.
        r = max(r, r + control.predict * (r - solver.rate_prev))
    end
    dt = solver.cfl / r
    if control.max_growth > 0 && solver.dt_prev > 0
        dt = min(dt, control.max_growth * solver.dt_prev)
    end
    return dt
end

"""
    Workspace(Q)
    Workspace(solver)

Reusable low-storage RK stage arrays. Pass a retained workspace to [`run!`](@ref)
or [`step!`](@ref) to avoid reallocating stage storage across calls.
"""
struct Workspace{A<:AbstractArray}
    dQ::A
    du::A
end

Workspace(Q::AbstractArray) = Workspace(zero(Q), zero(Q))
Workspace(solver::Solver) = Workspace(allocate_state(solver), allocate_state(solver))
"""
    step!(solver, Q, dQ, du, dt, prepared=false)

Advance the conserved state by one five-stage, fourth-order low-storage
Runge--Kutta step of size `dt`. `dQ` and `du` are caller-provided work arrays of
the same shape as `Q`.

`prepared = true` asserts that boundary conditions are already enforced on `Q` at
`solver.t` and that the primitive fields are current for it, which lets the first
stage skip a halo exchange and a primitives pass. [`run!`](@ref) can assert this
because [`max_rate`](@ref) has just performed both. A direct caller should leave
the default in place unless it has performed both itself. The argument is
positional for the reason given under [`compute_rhs!`](@ref).
"""
function step!(solver::Solver, Q, dQ, du, dt, prepared::Bool=false)
    decomp = solver.decomp
    o1, o2, o3 = decomp.n_halo_d
    nx, ny, nz = decomp.n_local
    for stage in 1:5
        solver.tstage = solver.t + RKC[stage] * dt
        # RKC[1] = 0, so a prepared caller's boundary values are the ones this
        # stage would compute; nothing between there and here has touched Q.
        first_prepared = prepared && stage == 1
        first_prepared || apply_bcs!(solver, Q)
        compute_rhs!(solver, Q, dQ, first_prepared)
        A = RKA[stage]
        B = RKB[stage]
        for c in 1:solver.equations.n_cons
            @threaded nx*ny*nz for jk in outer_indices(ny, nz)
                j, k = Tuple(jk)
                @inbounds for i in 1:nx
                    v = A * du[i+o1, j+o2, k+o3, c] + dt * dQ[i+o1, j+o2, k+o3, c]
                    du[i+o1, j+o2, k+o3, c] = v
                    Q[i+o1, j+o2, k+o3, c] += B * v
                end
            end
        end
    end
    solver.tstage = solver.t + dt
    apply_bcs!(solver, Q)
    return Q
end

step!(solver::Solver, Q, workspace::Workspace, dt, prepared::Bool=false) =
    step!(solver, Q, workspace.dQ, workspace.du, dt, prepared)

"""
    compute_dt(solver, Q)

CFL-limited timestep from the acoustic (|u_d| + c) / (h_d · scalefactor_d)
rate and a diffusive rate built from the current (possibly artificial)
transport coefficients, reduced over all ranks. Calls `primitives!` so the
estimate is EOS-aware; artificial coefficients lag by one step, which
[`StepControl`](@ref) exists to compensate for.
"""
compute_dt(solver::Solver, Q) = solver.cfl / max_rate(solver, Q)[1]

"""
    max_rate(solver, Q) -> (rate, rho_min)

Global maximum of the CFL rate, and the global minimum mixture density taken
directly from `Q`. Both quantities are evaluated in the same loop and collective.
The direct density check is required because `primitives!` substitutes finite
placeholders where ρ ≤ 0; the resulting CFL rate can remain finite after the
state has lost positivity. Without this check, failure is not detected until
the diffusive term subsequently drives `dt` toward zero.
"""
function max_rate(solver::Solver, Q)
    decomp = solver.decomp
    exchange_state!(Q, decomp)   # keep halos consistent for primitives!
    primitives!(solver, Q)
    o1, o2, o3 = decomp.n_halo_d
    nx, ny, nz = decomp.n_local
    n_species = solver.equations.n_species
    tr = solver.transport
    rate = 0.0
    ρ_min = Inf
    @inbounds for k in 1:nz, j in 1:ny, i in 1:nx
        I = CartesianIndex(i + o1, j + o2, k + o3)
        ρQ = 0.0
        for sp in 1:n_species
            ρQ += Q[I, sp]
        end
        ρ_min = min(ρ_min, ρQ)
        ρ = solver.rho[I]
        ri = 1 / ρ
        c = solver.c[I]
        cp = solver.cp_mix[I]
        uv = (solver.u[I], solver.v[I], solver.w[I])
        acc = 0.0
        dsum = 0.0
        for d in 1:3
            decomp.active[d] || continue      # no resolved variation
            idx = solver.inv_h[d][I] / solver.h[d]      # inverse physical spacing
            acc += (abs(uv[d]) + c) * idx
            dsum += idx * idx
        end
        # Curvature-source stiffness. When an angular dimension is RESOLVED,
        # its source rate (|u_ang|/r) is smaller than its advective rate
        # (|u_ang|/(r Δang)) and is already covered above. When it is
        # COLLAPSED — axisymmetric-with-swirl being the important case — the
        # loop skips it entirely, yet ρu_θ²/r still drives u_r as a stiff
        # source at small r. That term is added here.
        acc += curvature_rate(solver, solver.metric, I, uv)
        Dmax = tr.mu0 / (tr.Sc * ρ)
        for sp in 1:solver.equations.n_species
            Dmax = max(Dmax, tr.mu0 / (tr.Sc * ρ) + solver.D_art[sp][I])
        end
        ν = (tr.mu0 + solver.mu_art[I] + solver.beta_art[I]) * ri +
            (tr.mu0 * cp / tr.Pr + solver.kappa_art[I]) * ri / cp + Dmax
        acc += 2 * ν * dsum
        rate = max(rate, acc)
    end
    # One collective, not two: both quantities are reduced with `max` by
    # negating the density, and this runs every step of every run.
    red = MPI.Allreduce([rate, -ρ_min], max, decomp.comm)
    return (red[1], -red[2])
end

"""
    curvature_rate(solver, metric, I, uv)

Rate contribution from geometric momentum sources on angular dimensions that
are collapsed (and therefore contribute no advective CFL term). Zero in
Cartesian coordinates and whenever the corresponding dimension is resolved.
"""
curvature_rate(solver, ::CartesianMetric, I, uv) = 0.0

function curvature_rate(solver, ::CylindricalMetric, I, uv)
    solver.decomp.active[2] && return 0.0          # covered by the θ advective term
    return abs(uv[2]) * solver.inv_r[I]          # ρu_θ²/r driving u_r
end

function curvature_rate(solver, ::SphericalMetric, I, uv)
    a = 0.0
    solver.decomp.active[2] || (a += abs(uv[2]) * solver.inv_r[I])
    solver.decomp.active[3] ||
        (a += abs(uv[3]) * (solver.inv_r[I] + abs(solver.cot_over_r[I])))
    return a
end

"""
    dt_report(solver, Q)

Diagnostic companion to `compute_dt`: returns a NamedTuple naming the global
timestep limiter — `(dt, rank, index, coords, dim, kind)` where `kind` is
`:acoustic`, `:diffusive`, or `:curvature`. Periodic evaluation distinguishes a
physical timestep restriction from one imposed by azimuthal spacing near a
coordinate singularity; see the CFL discussion in the README.
"""
function dt_report(solver::Solver, Q)
    decomp = solver.decomp
    exchange_state!(Q, decomp)
    primitives!(solver, Q)
    o1, o2, o3 = decomp.n_halo_d
    tr = solver.transport
    best = (rate=-Inf, i=0, j=0, k=0, dim=0, kind=:none)
    @inbounds for k in 1:decomp.n_local[3], j in 1:decomp.n_local[2], i in 1:decomp.n_local[1]
        I = CartesianIndex(i + o1, j + o2, k + o3)
        ρ = solver.rho[I]; ri = 1 / ρ; c = solver.c[I]; cp = solver.cp_mix[I]
        uv = (solver.u[I], solver.v[I], solver.w[I])
        acc = 0.0; dsum = 0.0; wdim = 0; wrate = -Inf
        for d in 1:3
            decomp.active[d] || continue
            idx = solver.inv_h[d][I] / solver.h[d]
            rd = (abs(uv[d]) + c) * idx
            acc += rd; dsum += idx * idx
            rd > wrate && (wrate = rd; wdim = d)
        end
        crate = curvature_rate(solver, solver.metric, I, uv)
        Dmax = tr.mu0 / (tr.Sc * ρ)
        for sp in 1:solver.equations.n_species
            Dmax = max(Dmax, tr.mu0 / (tr.Sc * ρ) + solver.D_art[sp][I])
        end
        ν = (tr.mu0 + solver.mu_art[I] + solver.beta_art[I]) * ri +
            (tr.mu0 * cp / tr.Pr + solver.kappa_art[I]) * ri / cp + Dmax
        drate = 2 * ν * dsum
        total = acc + crate + drate
        if total > best.rate
            kind = drate > max(wrate, crate) ? :diffusive :
                   crate > wrate ? :curvature : :acoustic
            best = (rate=total, i=i, j=j, k=k, dim=wdim, kind=kind)
        end
    end
    grate = MPI.Allreduce(best.rate, max, decomp.comm)
    mine = best.rate >= grate ? MPI.Comm_rank(decomp.comm) : typemax(Int)
    owner = MPI.Allreduce(mine, min, decomp.comm)
    return (dt=solver.cfl / grate, rank=owner, index=(best.i, best.j, best.k),
            coords=(xcoord(solver, 1, best.i), xcoord(solver, 2, best.j),
                    xcoord(solver, 3, best.k)),
            dim=best.dim, kind=best.kind)
end

"""
    run!(solver, Q; tfinal, nmax=typemax(Int), callback=nothing, control=solver.control)

Advance to `tfinal` (or `nmax` steps), filtering the conserved variables every
`s.filter_interval` steps and invoking `callback` after each step.

`callback` may be a bare `callback(solver, Q)` invoked every step (the original
contract, return value ignored), a [`Callback`](@ref) pairing a trigger with an
effect, or a tuple of those. A scheduled [`AtTime`](@ref) or [`EveryTime`](@ref)
trigger shortens `dt` over the preceding `control.landing_steps` steps so that a
step ends at the scheduled instant without an arbitrarily small final step. An
effect returning `true` ends the run after that step. `tfinal` uses a direct
clip, so scheduled callbacks do not alter the step sequence of a run that has
none.

`control` is a [`StepControl`](@ref) governing timestep prediction, the floors
below which the run is declared failed, and whether a failure is recoverable by
rolling back and lowering the CFL. It defaults to the one the solver was built
with. On an unrecoverable failure this throws [`SolverFailure`](@ref) rather
than continuing with a collapsed timestep; see the implementation note at the
top of this file.

When `control.retries > 0`, the CFL is lowered in place on each retry.
Consequently, `solver.cfl` after a completed run records the value used to
complete the calculation and can be supplied to the next run's `Numerics`.
"""
function run!(solver::Solver, Q, workspace::Workspace;
              tfinal, nmax::Int=typemax(Int), callback=nothing,
              control::StepControl=solver.control)
    rank = MPI.Comm_rank(solver.decomp.comm)
    save = control.retries > 0 ? Savepoint(copy(Q), solver.t, solver.step) : nothing
    attempts = 0
    dt_seen = 0.0
    # Savepoints are suppressed at or below this step after a rollback. Without
    # it a retry re-saves its way forward to the state that failed and then
    # "rolls back" onto it, so every further retry starts from the corrupt state
    # and only the CFL moves. Observed as: rolled back to step 180, failed at
    # step 180, four times.
    guard_step = -1
    while solver.t < tfinal && solver.step < nmax
        # Timed from here rather than around step! alone: max_rate carries the
        # per-step Allreduce and the filter is a full set of line solves, so both
        # are step cost a user is trying to see. Callbacks are outside it — a
        # progress callback that reduces a diagnostic would otherwise be timing
        # itself, and reporting that as solver cost.
        wall_0 = time_ns()
        # Boundary conditions before the rate measurement, for two reasons. The
        # step should be sized from the state it is about to advance, and the
        # previous iteration's filter_state! has smeared whatever the conditions
        # impose on the edge planes. With Q settled here, max_rate's halo exchange
        # and primitives pass also serve stage 1, which is what `prepared` below
        # asserts; that removes one sixth of both from the per-step cost.
        solver.tstage = solver.t
        apply_bcs!(solver, Q)
        rate, rho_min = max_rate(solver, Q)
        dt = predicted_dt(solver, control, rate)
        failure = check_step(control, dt, rho_min, dt_seen, solver.step,
                             solver.t, solver.cfl)
        if failure !== nothing
            (save === nothing || attempts >= control.retries) && throw(failure)
            attempts += 1
            copyto!(Q, save.Q)
            solver.t = save.t
            solver.step = save.step
            # Compounding: the CFL is reduced from its current value and never
            # restored from the savepoint, so three retries give backoff^3.
            solver.cfl *= control.cfl_backoff
            guard_step = failure.step
            # The rate history belongs to the abandoned trajectory; keeping it
            # would have the predictor extrapolate from states that no longer
            # exist. dt_seen goes too, or the relative floor would immediately
            # fire again against a dt from before the rollback.
            solver.dt_prev = zero(eltype(Q))
            solver.rate_prev = zero(eltype(Q))
            dt_seen = 0.0
            # Instants between the savepoint and the failure were visited on a
            # trajectory that no longer exists. Re-arm them so the replacement
            # trajectory visits them too; see `rewind!` for what is and is not
            # rolled back.
            rewind_callbacks!(callback, save.t, save.step)
            rank == 0 && @warn "run!: $(failure.reason) at step $(failure.step); " *
                               "rolled back to step $(save.step) and lowered cfl to " *
                               "$(solver.cfl) (retry $attempts of $(control.retries))"
            continue
        end
        # Q has just passed its health check, which is the only moment it is
        # known good — the checks run on the state entering a step, so saving
        # after stepping would bank a state nothing has yet vetted.
        if save !== nothing && control.savepoint_interval > 0 &&
           solver.step > guard_step && solver.step % control.savepoint_interval == 0
            copyto!(save.Q, Q)
            save.t = solver.t
            save.step = solver.step
        end
        dt_seen = max(dt_seen, dt)
        # Clip to the endpoint AFTER the checks, and to the next scheduled
        # callback time as well: an AtTime trigger has to be landed on exactly,
        # not overshot, and nothing outside run! can reach dt to arrange that.
        # The gap is only applied when it is positive — a requested time already
        # behind solver.t would otherwise drive dt to zero or negative, which
        # stalls the run rather than firing anything.
        dt = min(dt, tfinal - solver.t)
        gap = callback_next_time(callback, solver) - solver.t
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
        if gap > 0 && gap < dt * control.landing_steps
            dt = gap / ceil(gap / dt)
        end
        prepared = true         # see the apply_bcs!/max_rate note above
        step!(solver, Q, workspace, dt, prepared)
        solver.t += dt
        solver.step += 1
        solver.dt_prev = dt
        solver.rate_prev = rate
        if solver.filter_interval > 0 && solver.step % solver.filter_interval == 0
            filter_state!(solver, Q)
        end
        # A rollback `continue`s above this, so an abandoned iteration never
        # records a step time — wall_total counts work that stood.
        solver.wall_step = (time_ns() - wall_0) / 1e9
        solver.wall_total += solver.wall_step
        run_callbacks!(callback, solver, Q) && break
    end
    return Q
end

function run!(solver::Solver, Q; workspace=nothing, kwargs...)
    work = workspace === nothing ? Workspace(Q) : workspace
    return run!(solver, Q, work; kwargs...)
end

"""
    filter_state!(solver, Q)

Apply the compact filter to every conserved component along every active
dimension, with batched per-dimension halo exchange and axis parity routing
(ρu_r and ρu_θ are odd across the axis; everything else is even).
"""
function filter_state!(solver::Solver, Q)
    decomp = solver.decomp
    comps = [view(Q, :, :, :, c) for c in 1:solver.equations.n_cons]
    for d in 1:3
        decomp.active[d] || continue
        exchange_dim_batch!(comps, decomp, d)
        for c in 1:solver.equations.n_cons
            filt_along!(solver.tmp_a, comps[c], solver, d, cons_parity(solver, d, c))
            copy_interior!(comps[c], solver.tmp_a, decomp)
        end
    end
    return Q
end

# --- Top-level error reporting under many ranks.
#
# An uncaught exception is reported by *every* rank, so a SolverFailure at 448
# ranks produces 448 stacktraces — several thousand lines burying the one fact
# you need. And because the failures that matter here are raised off collective
# quantities (`max_rate` reduces, `check_step` reads the reduced result), they
# are usually identical on every rank, so 447 of those copies carry nothing.
#
# One backtrace, from rank 0, plus a one-line identification from anywhere else
# that failed — which is what catches the genuinely rank-local errors, like
# `plan_direction` rejecting one rank's block. Then MPI_Abort, so the ranks still
# blocked in a collective are killed outright rather than each unwinding through
# Julia's signal handler and printing a dump of its own.

"""
    mpi_main(body; comm = MPI.COMM_WORLD, exitcode = 1)

Run `body()` under a top-level error guard for large rank counts. Rank 0 prints a
full backtrace, other failing ranks print one line, and the function then calls
`MPI.Abort`. The return value is that of `body`.

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
        # before you get to 448 of them.
        # A failure on one rank alone leaves rank 0 blocked in a collective, so
        # nobody prints a backtrace and the one-liner carries no location. Set
        # CL_ERROR_BACKTRACE=all to get one from every failing rank — the right
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
        # MPI_Abort kills the whole job, so whoever calls it first silences
        # everyone else — including rank 0 mid-backtrace, which is exactly the
        # output worth keeping. Non-zero ranks therefore yield briefly. If rank 0
        # did not fail it is blocked in a collective and gets killed regardless,
        # so the only cost is `grace` seconds on the way down.
        rank == 0 || sleep(grace)
        MPI.Abort(comm, exitcode)
    end
end
