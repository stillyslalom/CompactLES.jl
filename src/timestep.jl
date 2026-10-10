# Time integration: five-stage fourth-order low-storage Runge–Kutta
# (Carpenter & Kennedy 1994), CFL-limited timestep, and the outer run loop
# with conservative-variable filtering every `filter_interval` steps.

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
`control.predict` steps, then cap the growth against the previous step. `rate`
is the reduced rate from [`max_rate`](@ref); the result is `solver.cfl` divided
by the extrapolated rate, capped at `control.max_growth * solver.dt_prev`.

Each is skipped when its control is zero, which is the default for both, and
also while `solver.rate_prev` and `solver.dt_prev` are still zero, which holds
on a freshly built solver and after a rollback but not on a second `run!` of the
same solver. The extrapolation is one-sided and only ever raises the rate; see
the comment in the body. A solver built with [`ImplicitConduction`](@ref)
also caps the result at the accuracy limit its step rule set after the
previous step. Nothing on the solver is modified here.
"""
function predicted_dt(solver::Solver, control::StepControl, rate)
    r = rate
    if control.predict > 0 && solver.rate_prev > 0
        # Linear extrapolation, one-sided: only ever raise the rate. A falling
        # rate indicates a relaxing flow, and stepping out on that prediction
        # overshoots the next shock.
        r = max(r, r + control.predict * (r - solver.rate_prev))
    end
    dt = solver.cfl / r
    if control.max_growth > 0 && solver.dt_prev > 0
        dt = min(dt, control.max_growth * solver.dt_prev)
    end
    return _implicit_step_limit(solver, dt)
end

# The accuracy limit of the implicit half (imex_step.jl); none by default.
@inline _implicit_step_limit(solver::Solver, dt) = dt

# The clock advance `solver.t += dt` performs. The sum is formed in the wider
# of the two types and stored in the clock's own, so a step below the spacing
# of the floating-point grid at `t` leaves the clock where it was. Every
# endpoint, clip and landing decision in `run!` is tested through this function
# rather than on `dt` alone, since only the stored result decides whether the
# run advances.
_advances(t, dt) = oftype(t, t + dt) > t

"""
    Workspace(Q)
    Workspace(solver)

Reusable low-storage RK stage arrays. Pass a retained workspace to [`run!`](@ref)
or [`step!`](@ref) to avoid reallocating stage storage across calls.
"""
struct Workspace{A}
    dQ::A
    du::A
end

Workspace(Q::AbstractArray) = Workspace(zero(Q), zero(Q))
Workspace(states::Vector{<:ConservedState}) =
    Workspace(_zero_states(states), _zero_states(states))
Workspace(solver::Solver) = Workspace(allocate_state(solver), allocate_state(solver))

# Zero states shaped like `states`, the stacking included: the tiles of a
# stacked level (levels.jl) are views of one array, and their stage arrays
# must be views of one array too, so a stack's members take views of one
# fresh zero over the same parent.
function _zero_states(states::Vector{<:ConservedState})
    out = similar(states)
    zeros_of = IdDict{Any,Any}()
    for (i, Q) in enumerate(states)
        data = parent(Q)
        if data isa SubArray
            z = get!(() -> zero(parent(data)), zeros_of, parent(data))
            out[i] = ConservedState(view(z, parentindices(data)...))
        else
            out[i] = zero(Q)
        end
    end
    return out
end
"""
    step!(solver, Q, dQ, du, dt, prepared=false)
    step!(solver, Q, workspace, dt, prepared=false)

Advance the conserved state by one five-stage, fourth-order low-storage
Runge--Kutta step of size `dt`, returning `Q`. `dQ` and `du` are caller-provided
work arrays of the same shape as `Q`, holding the stage right-hand side and the
low-storage accumulator; both are overwritten during the step, as is `Q`. A
[`Workspace`](@ref) supplies the pair. `solver.tstage` is left at
`solver.t + dt`, and boundary conditions are enforced on `Q` at that time before
returning.

The clock is advanced by [`run!`](@ref). This function does not advance
`solver.t` or `solver.step`, and it does not apply the state filter, the
callbacks, the positivity failsafe or the state validation, all of which
`run!` performs between steps. Nor does it apply the positivity limiter of
`Numerics(positivity_limiter = true)`, which `run!` applies inside its own
step and filter pass. A driver that
calls it directly advances the clock itself, `solver.t += dt` and
`solver.step += 1`, before the next call. Otherwise every stage of the next
step is evaluated at the old time, and anything scheduled on `solver.t` or
`solver.step` (a time-dependent boundary condition, the filter cadence of a
later `run!`) reads the stale value.

Every rank must call this function because each stage evaluates
[`compute_rhs!`](@ref).

A solver built with [`ImplicitConduction`](@ref) advances instead by one step
of the additive Runge–Kutta pair, `dQ` and `du` unused, and throws
[`SolverFailure`](@ref) when an implicit stage does not converge. Its step
rule may subdivide the step; the state returned is the one at `solver.t + dt`
either way.

`prepared = true` asserts that boundary conditions are enforced on `Q` at
`solver.t` and that the primitive fields are current for it, which lets the first
stage skip a halo exchange and a primitives pass. [`run!`](@ref) can assert this
because it applies the boundary conditions itself immediately before calling
[`max_rate`](@ref), which performs the exchange and the primitives pass.
Between that call and the step, `run!` writes to `Q` only when the `:repair`
validity policy repairs a point, and it then repeats the boundary conditions
and the rate measurement. The regrid, the filter, the positivity failsafe and the callbacks
run before the next iteration's measurement. A direct caller should leave the
default in place unless it has done the same. The argument is positional for
the reason given under [`compute_rhs!`](@ref).
"""
function step!(solver::Solver, Q, dQ, du, dt, prepared::Bool=false)
    decomp = solver.decomp
    for stage in 1:5
        solver.tstage = solver.t + oftype(solver.t, RKC[stage]) * dt
        # RKC[1] = 0, so a prepared caller's boundary values are the ones this
        # stage would compute; nothing between there and here has touched Q.
        first_prepared = prepared && stage == 1
        if !first_prepared
            _ledger_open!(solver, Q)
            apply_bcs!(solver, Q)
            _ledger!(solver, Q, :wall_enforce)
        end
        compute_rhs!(solver, Q, dQ, first_prepared)
        _ledger_faces!(solver, 1)
        _ledger_open!(solver, Q)
        _rk_update!(decomp, solver.equations.n_cons, Q, dQ, du,
                    RKA[stage], RKB[stage], dt)
        _ledger_update!(solver, Q, dQ, RKA[stage], RKB[stage], dt)
    end
    solver.tstage = solver.t + dt
    _ledger_open!(solver, Q)
    apply_bcs!(solver, Q)
    _ledger!(solver, Q, :wall_enforce)
    _validate_transport_state!(solver, Q)
    return Q
end

# The low-storage stage update over one patch's interior, shared between the
# single-patch and multi-patch step drivers.
@inline function _rk_point!(Q, dQ, du, A, B, dt, c, o1, o2, o3, i, j, k)
    @inbounds begin
        v = A * du[i+o1, j+o2, k+o3, c] + dt * dQ[i+o1, j+o2, k+o3, c]
        du[i+o1, j+o2, k+o3, c] = v
        Q[i+o1, j+o2, k+o3, c] += B * v
    end
    return nothing
end

function _rk_update!(decomp::Decomp, n_cons::Int, Q, dQ, du, A, B, dt)
    o1, o2, o3 = decomp.n_halo_d
    nx, ny, nz = decomp.n_local
    T = eltype(Q)
    At, Bt, dtt = T(A), T(B), T(dt)
    for c in 1:n_cons
        pointwise!(_rk_point!, Q, nx, ny, nz, Q, dQ, du,
                   At, Bt, dtt, c, o1, o2, o3)
    end
    return Q
end

"""
    step!(solver, states, dQs, dus, dt, prepared=false)

Multi-patch form of [`step!`](@ref): `states`, `dQs` and `dus` are vectors
aligned with `solver.patches`. Each stage evaluates every local patch's RHS and
update in the global patch order, then makes the interfaces consistent with
[`sync_patches!`](@ref), the shared-plane averaging and ghost refill after
every stage. Every rank in `solver.comm` must call this function. A solver built
with `subcycle = true` delegates to `subcycled_step!` instead.
"""
function step!(solver::Solver, states::Vector{<:ConservedState},
               dQs::Vector{<:ConservedState}, dus::Vector{<:ConservedState},
               dt, prepared::Bool=false)
    getfield(solver, :subcycle) &&
        return subcycled_step!(solver, states, dQs, dus, dt, prepared)
    levels = getfield(solver, :levels)
    patches = getfield(solver, :patches)
    _reflux_begin_step!(solver, states)
    for stage in 1:5
        solver.tstage = solver.t + oftype(solver.t, RKC[stage]) * dt
        first_prepared = prepared && stage == 1
        # Level by level, which is the global patch order (the root's
        # patches, then each level's tiles), so a stacked level's tiles
        # evaluate together.
        for lev in levels
            status = _level_rhs!(solver, lev, states, dQs, first_prepared)
            _check_transport_status(solver, status)
        end
        for lev in levels
            _ledger_open!(solver, states, lev)
            _level_update!(solver, lev, states, dQs, dus, RKA[stage], RKB[stage], dt)
            _ledger_update!(solver, states, dQs, RKA[stage], RKB[stage], dt, lev)
        end
        _ledger_open!(solver, states)
        sync_patches!(solver, states)
        _ledger!(solver, states, :same_level)
        prolong_level_ghosts!(solver, states)
        _ledger!(solver, states, :shell)
    end
    solver.tstage = solver.t + dt
    _ledger_open!(solver, states)
    for (i, p) in enumerate(patches)
        apply_bcs!(PatchSolver(solver, p), states[i])
    end
    _ledger!(solver, states, :wall_enforce)
    _validate_transport_state!(solver, states)
    return states
end

# --- Concurrent tiles ---------------------------------------------------------
#
# A tile of a refined level is small: a lattice cell of 12 parent nodes holds
# 37 fine nodes a side, 1369 points in two dimensions, so every threaded region
# inside its evaluation is below THREAD_MIN_WORK and runs serially, and with
# the tiles advanced one after another a tiled level ran no faster at eight
# threads than at one. Between the level's exchanges the tiles are independent:
# each phase of a tile reads its own state, ghost layers and coefficients and
# writes its own. Such a phase therefore runs the tiles concurrently, one task
# per scratch set: the tiles sharing an `RHSWorkspace` run in level order in
# one task, so a set serves one tile at a time, and `rhs_workspace!` spreads
# a level's tiles over `_workspace_slots` sets where this applies. Each tile
# takes the arithmetic it takes alone, so the state is the same bit for bit at
# every thread count.
#
# The groups are formed only where every tile is below THREAD_MIN_WORK and
# nothing inside a tile's evaluation is shared with another tile or
# communicates: host storage, every tile held whole by this rank (its line
# solves and halo fills are then local, and no MPI call is made off the main
# thread), no paired fold, whose butterfly exchanges with a partner, no
# `CompositeBC` face, whose scratch is filled on first use, no transport
# domain check, which reduces inside every right-hand side, and the budget
# ledger off, whose accumulators are the rank's. A tile's coarse-fine junction
# hooks write its own captures and read its own deferral flag
# (`Patch.reflux_deferred`), so a tile with captures runs with the others.

# The groups of `lev`'s tiles that may run concurrently, each a list of patch
# indices in level order, or `nothing` where the level runs tile by tile.
function _tile_groups(@nospecialize(solver::Solver), lev::Level)
    (Threads.nthreads() > 1 && length(lev.patches) > 1 && isempty(lev.stacks)) ||
        return nothing
    (BUDGET_LEDGER.on || transport_has_domain(solver.transport)) && return nothing
    patches = getfield(solver, :patches)
    groups = Vector{Int}[]
    sets = Any[]
    for pi in lev.patches
        p = patches[pi]
        _tile_concurrent(p) || return nothing
        a = getfield(p, :rhs_workspace).tmp_a
        g = findfirst(s -> s === a, sets)
        if g === nothing
            push!(sets, a)
            push!(groups, [pi])
        else
            push!(groups[g], pi)
        end
    end
    return length(groups) > 1 ? groups : nothing
end

function _tile_concurrent(@nospecialize(p::Patch))
    _cpu_storage(p.rho) && prod(p.decomp.dims) == 1 || return false
    # A larger tile threads inside its own evaluation instead.
    prod(p.decomp.n_local) < THREAD_MIN_WORK[] || return false
    any(f -> f !== nothing && f.pair !== nothing, p.folds) && return false
    return !any(bc -> bc[1] isa CompositeBC || bc[2] isa CompositeBC, p.bcs)
end

# `f(solver, pi, args...)` for every tile `pi` of `lev`: concurrently in the
# groups of `_tile_groups`, or in level order where there are none.
# Collectives must stay outside `f`; every rank calls this with the same `lev`.
function _foreach_tile(f::F, solver::Solver, lev::Level,
                       args::Vararg{Any,N}) where {F,N}
    groups = _tile_groups(solver, lev)
    if groups === nothing
        for pi in lev.patches
            f(solver, pi, args...)
        end
        return nothing
    end
    tasks = Vector{Task}(undef, length(groups))
    for (k, g) in enumerate(groups)
        tasks[k] = Threads.@spawn _tile_group!(f, solver, g, args...)
    end
    _wait_tiles(tasks)
    return nothing
end

function _tile_group!(f::F, solver::Solver, group::Vector{Int},
                      args::Vararg{Any,N}) where {F,N}
    for pi in group
        f(solver, pi, args...)
    end
    return nothing
end

# Wait for every task before raising the first failure, so that no tile is
# still running when the caller unwinds, and raise the exception the tile
# threw, not the task's wrapper, which a caller testing for a
# `SolverFailure` would not recognize.
function _wait_tiles(tasks::Vector{Task})
    failure = nothing
    for t in tasks
        try
            wait(t)
        catch e
            failure === nothing &&
                (failure = e isa TaskFailedException ? e.task.result : e)
        end
    end
    failure === nothing || throw(failure)
    return nothing
end

# --- Per-level phases -------------------------------------------------------
#
# The three phases a level's tiles take between synchronizations: boundary
# enforcement and the right-hand side, the stage update, and the state
# filter. On a level without stacked storage each runs patch by patch. On a
# device level with stacks (levels.jl) each runs once per stack, on the
# spanning patch and the stacked state the members' views share, so the
# launches and the compact solves' interface fences are paid per level, not
# per tile; the per-point arithmetic is the per-tile one, so both forms give
# the same state bit for bit. The boundary conditions are still enforced per
# tile: they read each tile's own faces, and every refined face enforces
# nothing, so the loop is cheap and stays general.

# `prepared` is `compute_rhs!`'s trailing flag; `enforce` applies the boundary
# conditions first, which a prepared state never needs and the Hermite
# endpoint's extra evaluation skips too, its state being enforced already.
function _level_rhs!(solver::Solver, lev::Level, states, dQs, prepared::Bool,
                     enforce::Bool=!prepared, comm=solver.comm)
    status = _prepare_level_transport!(solver, lev, states, prepared, enforce, comm)
    status == 0 || return status
    if transport_has_domain(solver.transport)
        prepared = true
        enforce = false
    end
    patches = getfield(solver, :patches)
    if _level_sensors(solver, lev)
        # A tiled level with shared faces computes its artificial
        # coefficients over the whole level first (level_sensors.jl); a
        # configuration either takes this on every step or never.
        _sensor_level_rhs!(_cold(solver), lev, states, dQs, prepared, enforce)
    elseif isempty(lev.stacks)
        _foreach_tile(_enforced_tile_rhs!, solver, lev, states, dQs,
                      prepared, enforce)
    else
        # A stacked level is device storage, which the ledger does not
        # sweep, so it takes no hooks.
        enforce && for pi in lev.patches
            apply_bcs!(PatchSolver(solver, patches[pi]), states[pi])
        end
        for st in lev.stacks
            compute_rhs!(PatchSolver(solver, st.patch), _stack_state(st, states),
                         _stack_state(st, dQs), prepared)
        end
    end
    # The molecular flux through interface ends reads every patch's interior
    # flux, so it follows the whole level (rhs.jl, phase two). A setup
    # constant, the same on every rank.
    (_ghost_viscous(solver) || _ghost_remainder(solver)) &&
        _level_ghost_fluxes!(solver, lev, states, dQs,
                             lev.index == 0 ? solver.comm : lev.level_comm.comm)
    return UInt8(0)
end

# The boundary conditions, when `enforce`, and the right-hand side of tile
# `pi`, with their ledger hooks: one tile's part of `_level_rhs!`.
function _enforced_tile_rhs!(solver, pi::Int, states, dQs, prepared::Bool,
                             enforce::Bool)
    patches = getfield(solver, :patches)
    if enforce
        _ledger_open!(solver, states, pi)
        _tile_bcs!(solver, _cold(patches[pi]), states, pi)
        _ledger!(solver, states, :wall_enforce, pi)
    end
    _tile_rhs!(solver, _cold(patches[pi]), states, dQs, pi, prepared, false)
    _ledger_faces!(solver, pi)
    return nothing
end

function _level_update!(solver::Solver, lev::Level, states, dQs, dus, A, B, dt)
    patches = getfield(solver, :patches)
    n_cons = solver.equations.n_cons
    if isempty(lev.stacks)
        _foreach_tile(_tile_update!, solver, lev, n_cons, states, dQs, dus,
                      A, B, dt)
        # The junction registers follow the stage update one tile at a time.
        for pi in lev.patches
            _reflux_fold!(solver, patches[pi], dQs[pi], A, B, dt)
        end
        return states
    end
    for st in lev.stacks
        _rk_update!(st.patch.decomp, n_cons, _stack_state(st, states),
                    _stack_state(st, dQs), _stack_state(st, dus), A, B, dt)
    end
    return states
end

# The update and the filter of tile `p`, `states[pi]`, behind a barrier on
# its concrete type, as `_tile_rhs!` is (`_unit_call`): the patch and the state
# vectors are heap objects, while its decomposition and a `PatchSolver` would be
# boxed as arguments of the dynamic call.
_tile_rk_update!(p, n_cons::Int, states, dQs, dus, pi::Int, A, B, dt) =
    _rk_update!(p.decomp, n_cons, states[pi], dQs[pi], dus[pi], A, B, dt)
_tile_filter!(solver, p, states, pi::Int) =
    filter_state!(PatchSolver(solver, p), states[pi])

# The forms `_foreach_tile` calls.
_tile_update!(solver, pi::Int, n_cons::Int, states, dQs, dus, A, B, dt) =
    _tile_rk_update!(_cold(getfield(solver, :patches)[pi]), n_cons, states, dQs, dus,
                     pi, A, B, dt)
_tile_filter!(solver, pi::Int, states) =
    _tile_filter!(solver, _cold(getfield(solver, :patches)[pi]), states, pi)

function _level_filter!(solver::Solver, lev::Level, states)
    if isempty(lev.stacks)
        _foreach_tile(_tile_filter!, solver, lev, states)
        return states
    end
    for st in lev.stacks
        filter_state!(PatchSolver(solver, st.patch), _stack_state(st, states))
    end
    return states
end

step!(solver::Solver, Q, workspace::Workspace, dt, prepared::Bool=false) =
    step!(solver, Q, workspace.dQ, workspace.du, dt, prepared)

"""
    subcycled_step!(solver, states, dQs, dus, dt, prepared=false)

Berger–Oliger subcycled step over the whole level hierarchy: advance the
root level by one step of `dt` with every finer level frozen, then each
finer level by three steps of a third of its parent's step, recursively,
with each refined patch's shell imposed at every stage time from the cubic
Hermite reconstruction of its parent's trajectory
([`hermite_level_shell!`](@ref)). The `t^{n+1}` Hermite endpoint costs one
extra RHS evaluation per step of every level that has children, taken
before the children's substeps so it samples the parent trajectory, not the
restricted composite.

Each level below the root filters its own state at its own step cadence:
the substep of level ℓ with global index `3·(parent index) + m` filters when
`filter_interval` divides it, the root's index being `solver.step`, giving
every level the same one-pass-per-step cadence as an unrefined run. Under a
positive `filter_cfl` a refined pass's relaxation weight reads the root's
`dt` and the per-direction rates of [`max_rate`](@ref), which that function
scales by the level's substep ratio, an upper bound on the level's own
directional CFL, so the filter is never weaker than the convention intends.
`run!`'s own per-step filter pass covers the root level only in this mode,
and the post-step restriction
then rebuilds every covered region from the filtered finer state. Levels two
or more below the root are restricted onto their parent inside the driver,
after each parent substep, so that the parent's next substep starts from
the composite.

Selected by `step!` when the solver was built with `subcycle = true`. Every
rank must take the same substep sequence because the Hermite box saves gather
over the parent communicator and each shell imposition performs the
component-distributed chain's ring `Allgatherv`.
"""
function subcycled_step!(solver::Solver, states::Vector{<:ConservedState},
                         dQs::Vector{<:ConservedState},
                         dus::Vector{<:ConservedState}, dt,
                         prepared::Bool=false)
    status, guard = _subcycled_step_status!(solver, states, dQs, dus, dt,
                                             prepared, solver.control)
    status == SUBSTEP_CFL && throw(_substep_cfl_failure(solver, guard, dt,
                                                        solver.control))
    status == SUBSTEP_INVALID && throw(_substep_invalid_failure(solver, guard, dt))
    _check_transport_status(solver, status)
    return states
end

# A refreshed coefficient field is available only after `_level_rhs!`.  A
# subcycled level can cross a newly formed front before the root recomputes its
# CFL rate, so inspect every refreshed RK stage rather than relying on the next
# root-step estimate. The status deliberately climbs out of the recursive
# schedule: throwing on a child owner would strand its parent peers in the box
# gathers and bypass `run!`'s rollback path.
const SUBSTEP_CFL = UInt8(0x80)

mutable struct SubstepCFLGuard{T}
    rate::T
    dt::T
    level::Int
    stage::Int
    count::Int
    # The substep state sweep below: the level and step index of the rejected
    # substep and the report, on the owners of that level; zeros elsewhere.
    invalid_level::Int
    invalid_count::Int
    invalid_report::StateReport
end

SubstepCFLGuard(dt) = SubstepCFLGuard(zero(dt), zero(dt), 0, 0, 0, 0, 0, StateReport())

# A refined level's substep ends at a state that no root-step sweep reads:
# the first two substeps of each parent step are overwritten by the third
# before the composite is swept, and the positivity failsafe acts on the
# composite only. On the validity cadence under `:strict`, each substep of
# a level below the root is swept over that level's patches and reduced over
# its communicator, after its filter and shell imposition; a rejection climbs
# out of the recursion as a status, as `SUBSTEP_CFL` does, and reaches
# `run!`'s rollback. `:repair` is left to the composite, since its floors and
# its reduction span the whole run, and `:permissive` would only report.
const SUBSTEP_INVALID = UInt8(0x40)

_substep_validity(solver, control) =
    control.validity === :strict && control.validity_interval > 0 &&
    solver.step % control.validity_interval == 0

function _substep_validity_status!(solver, lev, states, control, guard, count)
    patches = getfield(solver, :patches)
    # The sweep is a host loop and the storage choice is solver-wide, so
    # every owner of the level skips it together (see `state_report`).
    _cpu_storage(states[1]) || return UInt8(0)
    acc = _empty_local_report()
    for pi in lev.patches
        acc = _merge_local_report(acc,
                  _local_state_report(PatchSolver(solver, patches[pi]), states[pi],
                                      control.species_band))
    end
    report = _reduce_state_report(solver, acc, lev.level_comm.comm)
    state_valid(report) && return UInt8(0)
    guard.invalid_level = lev.index
    guard.invalid_count = count
    guard.invalid_report = report
    return SUBSTEP_INVALID
end

# Collective over the run: the status reaches every root rank together, and
# the one extra reduction gives each of them the owners' report.
function _substep_invalid_failure(solver, guard, root_dt)
    r = guard.invalid_report
    t0 = time_ns()
    counts = MPI.Allreduce([guard.invalid_level, guard.invalid_count, r.points,
                            r.nonfinite, r.negative_density, r.negative_species,
                            r.inadmissible, r.unrecoverable, r.extrapolated],
                           max, solver.comm)
    mins = MPI.Allreduce([r.rho_min, r.e_min], min, solver.comm)
    _wait!(solver, t0)
    report = StateReport(counts[3:9]..., mins[1], mins[2])
    return SolverFailure(:invalid_state, solver.step, solver.t, root_dt, solver.cfl,
        "the state after substep $(counts[2]) of refined level $(counts[1]) " *
        "rejected under validity = :strict: $report")
end

function _refreshed_substep_status!(solver, lev, states, dt, stage, count,
                                    control, guard)
    (lev.index > 0 && control.substep_cfl > 0) || return UInt8(0)
    rate = zero(dt)
    patches = getfield(solver, :patches)
    for pi in lev.patches
        r = _local_max_rate(PatchSolver(solver, patches[pi]), states[pi])[1]
        rate = max(rate, r)
    end
    # A NaN cannot be meaningfully ordered by the MPI max reduction.  It is a
    # CFL violation in its own right, represented by Inf so every rank agrees.
    rate = isfinite(rate) ? rate : oftype(rate, Inf)
    t0 = time_ns()
    rate = MPI.Allreduce(rate, max, lev.level_comm.comm)
    _wait!(solver, t0)
    cfl = dt * rate
    if cfl > guard.dt * guard.rate
        guard.rate = rate
        guard.dt = dt
        guard.level = lev.index
        guard.stage = stage
        guard.count = count
    end
    return cfl <= control.substep_cfl ? UInt8(0) : SUBSTEP_CFL
end

function _substep_cfl_failure(solver, guard, root_dt, control)
    local_cfl = guard.dt * guard.rate
    t0 = time_ns()
    cfl = MPI.Allreduce(local_cfl, max, solver.comm)
    _wait!(solver, t0)
    return SolverFailure(:substep_cfl, solver.step, solver.t, root_dt, solver.cfl,
        "refreshed refined-level substep CFL is $cfl, above the configured " *
        "StepControl.substep_cfl = $(control.substep_cfl)")
end

# `limiter` is the positivity limiters of the patches (`_limited_run_step!`),
# or `nothing`.
function _subcycled_step_status!(solver::Solver, states, dQs, dus, dt,
                                 prepared::Bool, control, limiter=nothing)
    t0 = solver.t
    guard = SubstepCFLGuard(dt)
    _reflux_begin_step!(solver, states)
    # `solver.step` counts completed steps; the level counts below are
    # one-based indices of the step in progress.
    status = _advance_level!(solver, 1, states, dQs, dus, t0, dt, prepared,
                             solver.step + 1, dt, 1, control, guard, limiter)
    status == 0 || return status, guard
    solver.tstage = t0 + dt
    _validate_transport_state!(solver, states)
    return UInt8(0), guard
end

# `step!` keeps its direct contract: a substep violation is thrown there.  The
# outer driver needs the same completed-or-rejected distinction without an
# exception escaping recursive stepping, so it uses this private form and lets
# `_rollback!` decide whether the rejection is recoverable.
function _run_step!(solver, Q, workspace, dt, prepared, control)
    getfield(solver, :subcycle) || return (step!(solver, Q, workspace, dt, prepared);
                                           nothing)
    return _subcycled_run_step!(_cold(solver), Q, workspace, dt, prepared,
                                control)::Union{Nothing,SolverFailure}
end

function _subcycled_run_step!(solver, Q, workspace, dt, prepared, control)
    status, guard = _subcycled_step_status!(solver, Q, workspace.dQ, workspace.du,
                                             dt, prepared, control)
    status == 0 && return nothing
    status == SUBSTEP_CFL && return _substep_cfl_failure(solver, guard, dt, control)
    status == SUBSTEP_INVALID && return _substep_invalid_failure(solver, guard, dt)
    _check_transport_status(solver, status)
    return nothing
end

# The one-based global index of substep `mc` of a child level under its
# parent's step `count`: the parent's previous steps contributed three each.
# Level 1 therefore counts 3s + mc under root step s + 1, as it always did,
# and level 2 counts 1 .. 9 under the first root step; the earlier
# `3 count + mc` on a 0-based root count gave level 2 the counts 4 .. 12,
# the wrong number and phase of filter passes at depth two and below.
_child_count(count::Int, mc::Int) = 3 * (count - 1) + mc

# A path a run either takes or never takes, gated by a setting (subcycling,
# ghost interface fluxes, the shared-diffusivity species channels, the
# positivity failsafe, savepoints) or by a failure (rollback), is called
# through this barrier. The
# callee is then compiled on first use, for the solver type that takes it,
# rather than with every solver type whose driver can reach the branch; a
# run that never takes it pays nothing, and one that does pays one dynamic
# dispatch per call. Per-type codegen of the drivers otherwise carries every
# such path whether or not the configuration can reach it.
@inline _cold(x) = Base.inferencebarrier(x)

# One step of size `dt` from `t0` on level `ℓ` (1-based index into
# `solver.levels`), followed by three substeps of each child and their
# restriction back. `count` is this level's global step index (the filter
# cadence); `parent_dt` and `m` place the step as substep `m` of its parent's
# step, which the Hermite shell reads as θ = (m − 1 + RKC) / 3; both are
# unused at the root. The operation order at two levels is the one the
# two-level driver established, so a two-level run is unchanged by the
# recursion. Under the positivity limiter (`limiter`, the patches' limiters)
# each stage's increment is limited after the Hermite endpoint's save, which
# keeps the right-hand side, and a refined level's filter passes are limited.
function _advance_level!(solver::Solver, ℓ::Int, states, dQs, dus, t0, dt,
                         prepared::Bool, count::Int, parent_dt, m::Int,
                         control, guard, limiter=nothing)
    levels = getfield(solver, :levels)
    patches = getfield(solver, :patches)
    lev = levels[ℓ]
    # A child level with no tiles (a regridded level before its first tag or
    # after its last) takes no Hermite endpoint and no substeps, so the step
    # is the one a level without children takes. Its transfers are held by
    # every owner of this level, so the test is uniform over them.
    child = ℓ < length(levels) && !isempty(levels[ℓ+1].transfers) ?
            levels[ℓ+1] : nothing
    T = typeof(dt)
    # Communication ownership follows the data at each level:
    #
    # - Patch line solves, halo exchanges, and shell impositions use the
    #   owning tile's communicator.
    # - Same-level records use point-to-point operations on level ℓ's
    #   communicator.
    # - Child box gathers and restriction use level ℓ's communicator because
    #   they read and update parent data.
    # - Recursive child stepping uses level ℓ + 1's communicator and is
    #   entered only by its owners.
    #
    # Every owner of level ℓ enters this function. A tile shell imposition is
    # entered only by the ranks that own that tile.
    function shell!(θ)
        _ledger_open!(solver, states, lev)
        _foreach_tile(_tile_hermite_shell!, solver, lev, lev, states, θ, parent_dt)
        _ledger!(solver, states, :shell, lev)
    end
    # Hermite endpoints for the children: the RHS at t^n falls out of stage 1
    # (RKC[1] = 0, so stage 1's dQ is the RHS on the unmodified Q). Every
    # patch of this level has its RHS before the gathers, since a child's box
    # may span several of them.
    # The box exchanges are entered by every owner of this level, so a rank
    # waits here for the parent ranks holding its tiles' boxes; the rebalance
    # measure counts that as waiting, not work.
    function save_boxes!(at_end)
        child === nothing && return nothing
        wall_gather = time_ns()
        save_level_boxes!(solver, child, states, dQs, at_end)
        _wait!(solver, wall_gather)
        return nothing
    end
    for stage in 1:5
        solver.tstage = t0 + oftype(t0, RKC[stage]) * dt
        first_prepared = prepared && stage == 1
        ℓ > 1 && shell!((T(m - 1) + T(RKC[stage])) / T(3))
        status = _level_rhs!(solver, lev, states,
                             _limiter_stage_rhs(limiter, dQs, stage, dt), first_prepared,
                             !first_prepared, lev.level_comm.comm)
        status == 0 || return status
        status = _refreshed_substep_status!(solver, lev, states, dt, stage,
                                            count, control, guard)
        status == 0 || return status
        stage == 1 && save_boxes!(false)
        _limit_level_stage!(limiter, solver, lev, states, dQs, dus, stage, dt)
        _ledger_open!(solver, states, lev)
        _level_update!(solver, lev, states, dQs, dus, RKA[stage], RKB[stage], dt)
        _ledger_update!(solver, states, dQs, RKA[stage], RKB[stage], dt, lev)
        # Same-level consistency before the next stage's shell imposition,
        # which leaves the shared faces to these records.
        _sync_level!(solver, states, lev)
        _ledger!(solver, states, :same_level, lev)
    end
    solver.tstage = t0 + dt
    θ_end = T(m) / T(3)
    ℓ > 1 && shell!(θ_end)
    _ledger_open!(solver, states, lev)
    for pi in lev.patches
        apply_bcs!(PatchSolver(solver, patches[pi]), states[pi])
    end
    _ledger!(solver, states, :wall_enforce, lev)
    if ℓ > 1 && solver.filter_interval > 0 && count % solver.filter_interval == 0
        _limited_level_filter!(limiter, solver, lev, states)
        _ledger!(solver, states, :filter, lev)
        _sync_level!(solver, states, lev)
        _ledger!(solver, states, :same_level, lev)
        # The filter is not shell-preserving; re-impose the forcing so the
        # next substep (or the restriction) reads a consistent boundary.
        shell!(θ_end)
    end
    if ℓ > 1 && _substep_validity(solver, control)
        status = _substep_validity_status!(solver, lev, states, control, guard, count)
        status == 0 || return status
    end
    child === nothing && return _prepare_level_transport!(
        solver, lev, states, false, false, lev.level_comm.comm)
    # The t^{n+1} Hermite endpoint: the conditions are already enforced on
    # this state, so the right-hand side alone, with its primitives refreshed.
    status = _level_rhs!(solver, lev, states, dQs, false, false, lev.level_comm.comm)
    status == 0 || return status
    save_boxes!(true)
    dtf = dt / T(3)
    # A rank owning level ℓ but not level ℓ+1 skips the substeps: they carry
    # only level-(ℓ+1) collectives, which its owners alone enter. The substep
    # count is fixed, so the two rank sets do not diverge.
    if child.level_comm.owned
        for mc in 1:3
            status = _advance_level!(solver, ℓ + 1, states, dQs, dus,
                            t0 + (mc - 1) * dtf, dtf, false,
                            _child_count(count, mc), dt, mc, control, guard, limiter)
            status == 0 || break
        end
    end
    if transport_has_domain(solver.transport) || control.substep_cfl > 0 ||
       _substep_validity(solver, control)
        status = MPI.Allreduce(status, max, lev.level_comm.comm)
        status == 0 || return status
    end
    # The root's restriction is `run!`'s, after its filter pass; every deeper
    # level restricts here so its parent's next substep sees the composite.
    if ℓ > 1
        _ledger_open!(solver, states, lev)
        _restrict_tiles!(solver, states, child)
        _ledger!(solver, states, :restrict, lev)
        _reflux_apply!(solver, states, ℓ)
        _ledger!(solver, states, :reflux, lev)
        # The restriction can change nodes beside a tile interface of this
        # level; its neighbors' ghosts must see them before the next substep.
        _sync_level!(solver, states, lev)
        _ledger!(solver, states, :same_level, lev)
    end
    return UInt8(0)
end

"""
    compute_dt(solver, Q)

CFL-limited timestep from the advective, acoustic and diffusive rates of
[`max_rate`](@ref), reduced over all ranks. This is `solver.cfl` divided by
the rate that function returns, discarding the density and the per-direction
rates it returns alongside, so it carries that function's collective and its
side effects on `Q`'s halos and on the primitive fields.

The artificial coefficients are those left by the previous step, so the
diffusive rate lags by one. `StepControl.predict` extrapolates against that lag
and is off by default; the note at the top of `stepcontrol.jl` records the
measurement behind that default. A freshly built solver has no previous step,
so [`run!`](@ref) evaluates the right-hand side of the initial state once
before its first step to fill the arrays; the comment on that priming in
`timestep.jl` records what sizing the first step without them cost.
"""
compute_dt(solver::Solver, Q) = solver.cfl / max_rate(solver, Q)[1]
compute_dt(solver::Solver, states::Vector{<:ConservedState}) =
    solver.cfl / max_rate(solver, states)[1]

"""
    max_rate(solver, Q) -> (rate, rho_min, direction_rates)

Global maximum of the CFL rate, the global minimum mixture density taken
directly from `Q`, and the global maximum of each direction's one-dimensional
hyperbolic rate `(|u_d| + c) / h_d`, as a 3-tuple with zeros on collapsed
dimensions. All are evaluated in the same loop and reduced by one `Allreduce`,
so every rank must call this and all receive the same values.

The rate at a point is

    Σ_d |u_d| / h_d  +  c · sqrt(Σ_d 1 / h_d²)  +  curvature  +  2ν Σ_d 1 / h_d²

over the active dimensions, with `h_d` the local physical spacing, `ν` the
largest kinematic diffusivity of `_diffusive_rate` and the curvature
term of `curvature_rate` covering collapsed angular dimensions. The
advective and diffusive parts sum over dimensions because their symbols do,
`u · k` and `ν |k|²`. The acoustic part does not: its symbol is `c |k′|` with
`k′` the modified-wavenumber vector, so its bound on a tensor-product grid is
the Euclidean combination, `√3` times the one-dimensional rate on an isotropic
three-dimensional grid. That rate reproduces the acoustic ceiling measured on
Taylor–Green at 32³, the RK imaginary-axis limit 3.34 over the C6
modified-wavenumber peak 1.99, times `√3`.

The per-direction rates size nothing; [`filter_weight`](@ref) reads them,
recorded by [`run!`](@ref) beside the step taken, so that a directional filter
pass is relaxed against the rate of its own direction and not against the
maximum that sized the step.

Before the loop it exchanges `Q`'s halos and refreshes the primitive fields from
`Q`, which [`run!`](@ref) relies on when it passes `prepared = true` to
[`step!`](@ref).

The direct density check is required because `primitives!` substitutes finite
placeholders where ρ ≤ 0; the resulting CFL rate can remain finite after the
state has lost positivity. Without this check, failure is not detected until
the diffusive term subsequently drives `dt` toward zero.
"""
function max_rate(solver::Solver, Q)
    exchange_state!(Q, solver.decomp)   # keep halos consistent for primitives!
    primitives!(solver, Q)
    _validate_transport_state!(solver, Q; current=true)
    rate, ρ_min, dir = _local_max_rate(solver, Q)
    # One collective, not five: every quantity is reduced with `max` by
    # negating the density, and this runs every step of every run.
    t0 = time_ns()
    red = MPI.Allreduce([rate, -ρ_min, dir[1], dir[2], dir[3]], max, solver.comm)
    _wait!(solver, t0)
    return (red[1], -red[2], (red[3], red[4], red[5]))
end

"""
    max_rate(solver, states::Vector) -> (rate, rho_min, direction_rates)

Multi-patch form: the per-patch exchange, primitives pass and interior sweep
run patch by patch, and the quantities reduce over `solver.comm`, the whole
rank set, exactly once, hoisted outside the patch loop as the collective
discipline requires.

On a refined run a parent's nodes deep inside a child level, which the
child's restriction overwrites after every step, are held to a CFL number of
`OVERWRITTEN_CFL` = 0.75 rather than the solver's, when the solver's is the
smaller: their rate enters scaled by `cfl / OVERWRITTEN_CFL`. There the
parent's artificial coefficients, sensed at the parent spacing on the
restricted fine solution, would otherwise size the parent's step. A node is
held so when every node within `LEVEL_BUFFER` of it is fully covered, which
leaves every node that outlives the step or feeds the child's ghost layers
at the solver's CFL number. The density minimum and the direction rates
take every node unscaled.
"""
function max_rate(solver::Solver, states::Vector{<:ConservedState})
    _validate_transport_state!(solver, states)
    T = typeof(solver.cfl)
    rate = zero(T)
    ρ_min = T(Inf)
    dir = ntuple(_ -> zero(T), 3)
    subcycle = getfield(solver, :subcycle)
    patches = getfield(solver, :patches)
    # Each patch's sweep into `swept`, a level's tiles concurrently where they
    # may be (`_foreach_tile`), then the reduction in patch order.
    S = eltype(eltype(states))
    swept = Vector{Tuple{S,S,NTuple{3,S}}}(undef, length(patches))
    for lev in getfield(solver, :levels)
        _foreach_tile(_tile_rate!, solver, lev, states, swept)
    end
    for i in eachindex(patches)
        r, m, rd = swept[i]
        # A subcycled level ℓ advances at dt / 3^ℓ, so its rate constrains the
        # coarse step three times more weakly per level. The per-direction
        # rates are scaled the same way, so that a level's filter pass, which
        # reads the root's `dt`, sees the level's own directional CFL.
        if subcycle
            scale = oftype(r, 3)^getfield(patches[i], :level)
            r /= scale
            rd = rd ./ scale
        end
        rate = max(rate, r)
        ρ_min = min(ρ_min, m)
        dir = max.(dir, rd)
    end
    t0 = time_ns()
    red = MPI.Allreduce([rate, -ρ_min, dir[1], dir[2], dir[3]], max, solver.comm)
    _wait!(solver, t0)
    return (red[1], -red[2], (red[3], red[4], red[5]))
end

# The exchange, primitives pass and interior sweep of tile `pi` of `max_rate`.
function _tile_rate!(solver, pi::Int, states, swept)
    swept[pi] = _patch_rate!(solver, _cold(getfield(solver, :patches)[pi]), states[pi])
    return nothing
end
function _patch_rate!(solver, p, Q)
    ps = PatchSolver(solver, p)
    exchange_state!(Q, ps.decomp)
    primitives!(ps, Q)
    return _local_max_rate(ps, Q)
end

# Charge the time since `t0` (a `time_ns` reading) to the rank's waiting
# account for this step; see the `wall_wait` field of `Solver`.
_wait!(solver::Solver, t0::UInt64) =
    (solver.wall_wait += (time_ns() - t0) / 1e9; nothing)

# The interior sweep of max_rate over one patch, rank-local and free of
# collectives. `Array` storage keeps the fused serial loop; device storage
# (and FORCE_KA, which is how the test suite pins the launch path on one
# machine) evaluates the rate through a pointwise body into `tmp_a`/`tmp_b`
# and the per-direction rates into `grad_T_ion`, and reduces each with the
# storage's own `maximum`/`minimum`. All of that scratch is free here:
# max_rate runs before the step's first RHS evaluation, which refills it.
# Maximum and minimum are exact and order-independent, so the two paths agree
# bitwise. The azimuthal rate cap (modes.jl) is read by the loop alone, which
# setup's host-backend restriction on `polar_truncation` makes sufficient.
function _local_max_rate(solver::SolverLike, Q)
    _cpu_storage(Q) || return _local_max_rate_launch(solver, Q)
    if !FORCE_KA[] || _truncating(solver.truncation)
        return _local_max_rate_loop(solver, Q)
    end
    # Host storage takes the launch only under the test toggle, so the launch
    # is behind `_cold` there rather than compiled into every host solver type.
    T = eltype(Q)
    return _local_max_rate_launch(_cold(solver), Q)::Tuple{T,T,NTuple{3,T}}
end

# The diffusive rate shared by the three sweeps below. The thermal term is
# κ/(ρ cv): the internal-energy equation reduces to ρ cv DT/Dt = ∇·(κ∇T), so
# cv, not cp, sets its explicit limit, and the earlier κ/(ρ cp) admitted a
# step γ times too long wherever conduction (molecular or κ*) governed. The
# placeholder state `primitives!` writes where ρ ≤ 0 has cv = 0, and a point
# like that has failed the positivity check `max_rate` reports, so the
# fallback there only needs to stay finite.
@inline function _diffusive_rate(eos, ρ, p, T_ion, cp, molecular,
                                 mu_art, beta_art, kappa_art, D_art, I,
                                 n_species)
    ri = one(ρ) / ρ
    Dmax = molecular.D[1]
    for sp in 1:n_species
        Dmax = max(Dmax, molecular.D[sp] + D_art[sp][I])
    end
    κ = molecular.kappa + kappa_art[I]
    cv = mixture_cv(eos, ρ, p, T_ion, cp)
    thermal = κ * ri / (cv > 0 ? cv : cp)
    return (molecular.mu + mu_art[I] + beta_art[I]) * ri + thermal + Dmax
end

@inline function _rate_point!(rate_out, rhoq_out, dir_out, Q, rho, u, v, w,
                              parr, Tarr, carr, cparr, eos, mu_art, beta_art,
                              kappa_art, D_art, inv_h1, inv_h2, inv_h3, inv_r,
                              cot_over_r, metric, act, hh, transport, Y,
                              n_species, sharp, held, o, i, j, k)
    @inbounds begin
        I = CartesianIndex(i + o[1], j + o[2], k + o[3])
        T = eltype(rho)
        ρQ = zero(T)
        for sp in 1:n_species
            ρQ += Q[I, sp]
        end
        rhoq_out[I] = ρQ
        ρ = rho[I]
        c = carr[I]
        cp = cparr[I]
        uv = (u[I], v[I], w[I])
        ih = (inv_h1, inv_h2, inv_h3)
        acc = zero(T)
        dsum = zero(T)
        for d in 1:3
            act[d] || continue
            idx = ih[d][I] / hh[d]
            acc += abs(uv[d]) * idx
            dsum += idx * idx
            dir_out[d][I] = (abs(uv[d]) + c) * idx
        end
        acc += c * sqrt(dsum)                 # the acoustic symbol is c |k'|
        acc += _curvature_rate_point(metric, act[2], act[3], inv_r, cot_over_r,
                                     I, uv)
        molecular = transport_at(transport, eos, Tarr, rho, cparr, Y, I)
        ν = _diffusive_rate(eos, ρ, parr[I], Tarr[I], cp, molecular,
                            mu_art, beta_art, kappa_art, D_art, I, n_species)
        acc += _sharpening_rate(sharp, c, ih, hh, act, I)
        rate_out[I] = (acc + 2 * ν * dsum) * _rate_weight(held, I)
    end
    return nothing
end

# The factor on the rate at node `I`: `held = (overwritten, weight)` with the
# patch's `overwritten` mask and `_overwritten_weight`, or `nothing` on a
# patch with no overwritten node, whose launch then reads no mask.
@inline _rate_weight(::Nothing, I) = true
@inline _rate_weight(held, I) =
    @inbounds ifelse(held[1][I] != 0, held[2], one(held[2]))

# The factor `max_rate` scales an overwritten node's rate by, so that the
# node is held to `OVERWRITTEN_CFL` where the solver's CFL number is lower.
_overwritten_weight(cfl) = min(one(cfl), cfl / oftype(cfl, OVERWRITTEN_CFL))

function _local_max_rate_launch(solver::SolverLike, Q)
    decomp = solver.decomp
    o1, o2, o3 = decomp.n_halo_d
    nx, ny, nz = decomp.n_local
    tr = solver.transport
    ft = solver.field_tuples
    dir_out = solver.grad_T_ion
    overwritten = solver.overwritten
    pointwise!(_rate_point!, solver.tmp_a, nx, ny, nz,
               solver.tmp_a, solver.tmp_b, dir_out, Q, solver.rho, solver.u,
               solver.v, solver.w, solver.p, solver.T_ion, solver.c,
               solver.cp_mix, solver.eos, solver.mu_art, solver.beta_art,
               solver.kappa_art, ft.D_art, solver.inv_h[1], solver.inv_h[2],
               solver.inv_h[3], solver.inv_r, solver.cot_over_r, solver.metric,
               decomp.active, solver.h,
               tr, ft.Y, solver.equations.n_species,
               _sharpening_constants(solver),
               isempty(overwritten) ? nothing :
                   (overwritten, _overwritten_weight(eltype(Q)(solver.cfl))),
               (o1, o2, o3))
    interior = (o1+1:o1+nx, o2+1:o2+ny, o3+1:o3+nz)
    rates = view(solver.tmp_a, interior...)
    rhos = view(solver.tmp_b, interior...)
    dir = ntuple(3) do d
        decomp.active[d] ? maximum(view(dir_out[d], interior...)) : zero(eltype(Q))
    end
    return (maximum(rates), minimum(rhos), dir)
end

function _local_max_rate_loop(solver::SolverLike, Q)
    decomp = solver.decomp
    nx, ny, nz = decomp.n_local
    T = eltype(Q)
    # The fields are read into a tuple before the sweep, so the per-point body
    # takes plain arrays and no property lookup through the solver.
    fields = (Q, solver.rho, solver.c, solver.cp_mix, solver.u, solver.v, solver.w,
              solver.p, solver.T_ion, solver.inv_h, solver.h, decomp.active,
              decomp.n_halo_d, solver.equations.n_species, solver.transport,
              solver.truncation, _sharpening_constants(solver), solver.eos,
              solver.field_tuples.Y, solver.mu_art, solver.beta_art,
              solver.kappa_art, solver.D_art, solver.metric, solver.inv_r,
              solver.cot_over_r, solver.overwritten,
              _overwritten_weight(T(solver.cfl)))
    return _max_rate_sweep(fields, nx, ny, nz)
end

# The threaded sweep, keyed on the field tuple, whose type depends on the
# element and array types, the EOS, the transport and the metric but not on
# the plans, folds or boundary conditions, so solver types differing only in
# those share it.
function _max_rate_sweep(fields, nx::Int, ny::Int, nz::Int)
    T = eltype(fields[1])
    # Chunks of the flattened (j, k) range, each reduced on its own and then
    # combined. Maximum and minimum are exact, so the result does not depend
    # on how the points are grouped: the threaded sweep returns the serial
    # one's values bit for bit.
    outer = outer_indices(ny, nz)
    chunks = _line_chunks(length(outer))
    parts = Vector{NTuple{5,T}}(undef, length(chunks))
    @threaded nx * ny * nz for t in 1:length(chunks)
        @inbounds parts[t] = _rate_sweep(fields, outer, chunks[t], nx)
    end
    rate, ρ_min, r1, r2, r3 = zero(T), T(Inf), zero(T), zero(T), zero(T)
    for part in parts
        rate = max(rate, part[1])
        ρ_min = min(ρ_min, part[2])
        r1 = max(r1, part[3]); r2 = max(r2, part[4]); r3 = max(r3, part[5])
    end
    return (rate, ρ_min, (r1, r2, r3))
end

function _rate_sweep(fields, outer, range, nx)
    Q, rho, carr, cparr, u, v, w, parr, Tarr, inv_h, h, active, halo, n_species,
        tr, modes, sharp, eos, Y, mu_art, beta_art, kappa_art, D_art, metric,
        inv_r, cot_over_r, overwritten, weight = fields
    o1, o2, o3 = halo
    capped = _truncating(modes)
    T = eltype(Q)
    rate = zero(T)
    ρ_min = T(Inf)
    r1 = zero(T); r2 = zero(T); r3 = zero(T)
    masked = !isempty(overwritten)
    @inbounds for jk in range
        j, k = Tuple(outer[jk])
        for i in 1:nx
            I = CartesianIndex(i + o1, j + o2, k + o3)
            ρQ = zero(T)
            for sp in 1:n_species
                ρQ += Q[I, sp]
            end
            ρ_min = min(ρ_min, ρQ)
            ρ = rho[I]
            c = carr[I]
            cp = cparr[I]
            uv = (u[I], v[I], w[I])
            acc = zero(T)
            dsum = zero(T)
            for d in 1:3
                active[d] || continue      # no resolved variation
                idx = inv_h[d][I] / h[d]      # inverse physical spacing
                # At a truncated ring the θ spacing is that of the highest mode kept.
                d == 2 && capped && (idx = _theta_spacing(modes, I[1], idx))
                acc += abs(uv[d]) * idx
                dsum += idx * idx
                # The direction's own hyperbolic rate, the same expression as the
                # launch path's so the two agree bitwise.
                rd = (abs(uv[d]) + c) * idx
                if d == 1
                    r1 = max(r1, rd)
                elseif d == 2
                    r2 = max(r2, rd)
                else
                    r3 = max(r3, rd)
                end
            end
            acc += c * sqrt(dsum)                 # the acoustic symbol is c |k'|
            # Curvature-source stiffness. When an angular dimension is RESOLVED,
            # its source rate (|u_ang|/r) is smaller than its advective rate
            # (|u_ang|/(r Δang)) and is covered above. When it is COLLAPSED
            # (axisymmetric flow with swirl is the important case), the
            # loop skips it entirely, yet ρu_θ²/r still drives u_r as a stiff
            # source at small r. That term is added here.
            acc += _curvature_rate_point(metric, active[2], active[3], inv_r,
                                         cot_over_r, I, uv)
            molecular = transport_at(tr, eos, Tarr, rho, cparr, Y, I)
            ν = _diffusive_rate(eos, ρ, parr[I], Tarr[I], cp, molecular, mu_art,
                                beta_art, kappa_art, D_art, I, n_species)
            acc += _sharpening_rate(sharp, c, inv_h, h, active, I)
            acc += 2 * ν * dsum
            # An overwritten node's rate is scaled (`Patch.overwritten`); its
            # density and its direction rates above, which the positivity check
            # and the filter's relaxation read, are not.
            masked && overwritten[I] != 0 && (acc *= weight)
            rate = max(rate, acc)
        end
    end
    return (rate, ρ_min, r1, r2, r3)
end

"""
    curvature_rate(solver, metric, I, uv)

Rate contribution from geometric momentum sources on angular dimensions that
are collapsed (and therefore contribute no advective CFL term). Zero in
Cartesian coordinates and whenever the corresponding dimension is resolved.
"""
curvature_rate(solver, metric::Metric, I, uv) =
    _curvature_rate_point(metric, solver.decomp.active[2],
                          solver.decomp.active[3], solver.inv_r,
                          solver.cot_over_r, I, uv)

# The launchable form: plain arrays and activity flags in place of the solver,
# shared with the `_rate_point!` kernel body of `max_rate`.
@inline _curvature_rate_point(::CartesianMetric, a2, a3, inv_r, cot_over_r,
                              I, uv) = zero(first(uv))

@inline function _curvature_rate_point(::CylindricalMetric, a2, a3, inv_r,
                                       cot_over_r, I, uv)
    a2 && return zero(first(uv))            # covered by θ advection
    return @inbounds abs(uv[2]) * inv_r[I]  # ρu_θ²/r driving u_r
end

@inline function _curvature_rate_point(::SphericalMetric, a2, a3, inv_r,
                                       cot_over_r, I, uv)
    a = zero(first(uv))
    @inbounds begin
        a2 || (a += abs(uv[2]) * inv_r[I])
        a3 || (a += abs(uv[3]) * (inv_r[I] + abs(cot_over_r[I])))
    end
    return a
end

"""
    dt_report(solver, Q)

Diagnostic companion to [`compute_dt`](@ref), returning the NamedTuple
`(dt, rank, index, coords, dim, kind)`. `dt` is the same limited timestep
`compute_dt` returns and `rank` is the rank owning the point that set it; the
lowest such rank is named if several tie. `index` is that point's rank-local,
one-based interior index, `coords` its physical coordinates, `dim` the direction
carrying the largest acoustic rate there, and `kind` is `:acoustic`,
`:diffusive` or `:curvature`, whichever of the diffusive rate, the curvature rate
and the acoustic rate in direction `dim` is largest, ties going to `:acoustic`.
The acoustic entry in that comparison is the one direction, not the sum over
directions that the selecting rate and `dt` are built from.

Only `dt` and `rank` are global. The remaining four describe the calling rank's
own local maximum, so read them from the rank named by `rank`.

Every rank in `solver.comm` must enter the two `Allreduce`s, halo exchange, and
primitive refresh that [`max_rate`](@ref) also performs. Periodic evaluation
distinguishes a physical
timestep restriction from one imposed by azimuthal spacing near a coordinate
singularity; see the CFL discussion in the README.
"""
function dt_report(solver::Solver, Q)
    _cpu_storage(Q) ||
        error("dt_report is a host sweep; copy the state to a CPU solver " *
              "for this diagnostic on a DeviceBackend")
    decomp = solver.decomp
    exchange_state!(Q, decomp)
    primitives!(solver, Q)
    _validate_transport_state!(solver, Q; current=true)
    o1, o2, o3 = decomp.n_halo_d
    tr = solver.transport
    modes = solver.truncation
    capped = _truncating(modes)
    best = (rate=-Inf, i=0, j=0, k=0, dim=0, kind=:none)
    @inbounds for k in 1:decomp.n_local[3], j in 1:decomp.n_local[2], i in 1:decomp.n_local[1]
        I = CartesianIndex(i + o1, j + o2, k + o3)
        ρ = solver.rho[I]; ri = 1 / ρ; c = solver.c[I]; cp = solver.cp_mix[I]
        uv = (solver.u[I], solver.v[I], solver.w[I])
        acc = 0.0; dsum = 0.0; wdim = 0; wrate = -Inf
        for d in 1:3
            decomp.active[d] || continue
            idx = solver.inv_h[d][I] / solver.h[d]
            d == 2 && capped && (idx = _theta_spacing(modes, I[1], idx))
            # The per-direction figure is the diagnostic; the selecting total
            # below combines the acoustic part as `max_rate` does.
            rd = (abs(uv[d]) + c) * idx
            acc += abs(uv[d]) * idx; dsum += idx * idx
            rd > wrate && (wrate = rd; wdim = d)
        end
        acc += c * sqrt(dsum)
        acc += _sharpening_rate(_sharpening_constants(solver), c, solver.inv_h,
                                solver.h, decomp.active, I)
        crate = curvature_rate(solver, solver.metric, I, uv)
        molecular = transport_at(tr, solver.eos, solver.T_ion, solver.rho,
                                  solver.cp_mix, solver.field_tuples.Y, I)
        ν = _diffusive_rate(solver.eos, ρ, solver.p[I], solver.T_ion[I], cp,
                            molecular, solver.mu_art, solver.beta_art,
                            solver.kappa_art, solver.D_art, I,
                            solver.equations.n_species)
        drate = 2 * ν * dsum
        total = acc + crate + drate
        if total > best.rate
            kind = drate > max(wrate, crate) ? :diffusive :
                   crate > wrate ? :curvature : :acoustic
            best = (rate=total, i=i, j=j, k=k, dim=wdim, kind=kind)
        end
    end
    grate = MPI.Allreduce(best.rate, max, solver.comm)
    mine = best.rate >= grate ? MPI.Comm_rank(solver.comm) : typemax(Int)
    owner = MPI.Allreduce(mine, min, solver.comm)
    return (dt=solver.cfl / grate, rank=owner, index=(best.i, best.j, best.k),
            coords=(xcoord(solver, 1, best.i), xcoord(solver, 2, best.j),
                    xcoord(solver, 3, best.k)),
            dim=best.dim, kind=best.kind)
end

"""
    positivity_floors(solver, Q, control) -> (rho_floor, e_floor)

Absolute floors for `apply_positivity_floor!`, derived from `Q` as
`control.floor_ratio` times the global minimum mixture density and the global
minimum internal energy over the interior.

Returns `(0, 0)`, which leaves the failsafe inactive, when `floor_ratio` is zero
and also when either reference is not itself positive: a state that has left
the physical space supplies no scale to floor against, so [`run!`](@ref) warns
and invents none.

Every rank in `solver.comm` must enter the single `Allreduce`. `run!` calls this
function once before its loop, not once per step, so a second `run!` on the same
solver re-derives the floors from that call's initial state.
"""
function positivity_floors(solver::Solver, Q, control::StepControl)
    control.floor_ratio > 0 || return (0.0, 0.0)
    _cpu_storage(Q) ||
        error("StepControl.floor_ratio: the positivity failsafe is a host " *
              "sweep and is not yet supported on a DeviceBackend")
    ρ_min, e_min = _local_positivity_mins(solver, Q)
    t0 = time_ns()
    red = MPI.Allreduce([ρ_min, e_min], min, solver.comm)
    _wait!(solver, t0)
    (red[1] > 0 && red[2] > 0) || return (0.0, 0.0)
    return (control.floor_ratio * red[1], control.floor_ratio * red[2])
end

function positivity_floors(solver::Solver, states::Vector{<:ConservedState},
                           control::StepControl)
    control.floor_ratio > 0 || return (0.0, 0.0)
    ρ_min = Inf
    e_min = Inf
    for (ps, Q) in eachpatch(solver, states)
        r, e = _local_positivity_mins(ps, Q)
        ρ_min = min(ρ_min, r)
        e_min = min(e_min, e)
    end
    t0 = time_ns()
    red = MPI.Allreduce([ρ_min, e_min], min, solver.comm)
    _wait!(solver, t0)
    (red[1] > 0 && red[2] > 0) || return (0.0, 0.0)
    return (control.floor_ratio * red[1], control.floor_ratio * red[2])
end

function _local_positivity_mins(solver::SolverLike, Q)
    decomp = solver.decomp
    o1, o2, o3 = decomp.n_halo_d
    nx, ny, nz = decomp.n_local
    n_species = solver.equations.n_species
    m1, m2, m3 = solver.equations.i_mom
    i_energy = solver.equations.i_energy
    # Arithmetic in the state's own type, so the failsafe reads the same
    # numbers the solver does under Float32; the reduced minima are Float64.
    T = eltype(Q)
    ρ_min = Inf
    e_min = Inf
    @inbounds for k in 1:nz, j in 1:ny, i in 1:nx
        I = CartesianIndex(i + o1, j + o2, k + o3)
        ρ = zero(T)
        for sp in 1:n_species
            ρ += Q[I, sp]
        end
        ρ_min = min(ρ_min, ρ)
        # The internal energy is not recoverable where the density is not
        # positive. Such a point drives ρ_min below zero and disables the
        # failsafe on the line below, so skipping it here cannot hide anything.
        ρ > 0 || continue
        ke = (Q[I, m1]^2 + Q[I, m2]^2 + Q[I, m3]^2) / (2ρ)
        e_min = min(e_min, (Q[I, i_energy] - ke) / ρ)
    end
    return (ρ_min, e_min)
end

"""
    apply_positivity_floor!(solver, Q, rho_floor, e_floor, scope;
                            species_band = solver.control.species_band) -> tally

Inspect the interior of `Q` for points outside the physical state space, repair
them in place as far as `scope` allows, and report the global
`(cells, low_energy, mass, energy, momentum, species)` tally of what that
cost, `species` holding each species' mass change. The
floors come from `positivity_floors`, `scope` is `StepControl.floor_scope`, and
[`run!`](@ref) applies this after each completed step when
`StepControl.floor_ratio` is set.

Three repairs, in the order they have to run, since each depends on the state
the previous one leaves:

1. **A composition outside the species band**, a mass fraction below
   `-species_band` or above `1 + species_band`, has its negative partial
   densities clipped to zero and the positive ones rescaled onto the mixture
   density the point carried, which leaves that density and therefore the
   mixture mass exactly unchanged. The species masses are not conserved, and
   their change is tallied. The band is the one the validation sweep reads
   ([`state_report`](@ref)), so a point inside it, which a verdict accepts, is
   left as it is; `species_band = 0` clips every negative partial density.
2. **A mixture density below `rho_floor`** is raised to it, distributed over the
   positive partial densities, or onto the first species where there are none,
   which is the composition `primitives!` substitutes at a point it
   cannot invert. Nothing conserves mass here, so the addition is tallied.
3. **Energy**, at a point whose internal energy `E/ρ − ½|u|²` is below
   `e_floor`. The point is counted as `low_energy` whatever the scope. It is
   repaired when `scope === :internal_energy`, and under `:representable` only
   when the *total* energy density satisfies `E < ρ * e_floor`. The repair
   damps the velocity, scaling the
   momentum by `sqrt((E/ρ − e_floor) / (½|u|²))` with `E` untouched, so it moves
   kinetic energy into internal energy and creates none: total energy is
   conserved exactly and the momentum removed is tallied. Where there is no
   kinetic energy to convert, `E` being at or below the floor, the
   fallback raises `E` instead and conserves the momentum exactly. Each branch
   conserves one of the two exactly and tallies what it did to the other.

`:representable` repairs only when `E < ρ * e_floor`, which also selects the
fallback. That scope therefore always raises the energy and never damps a
velocity.

Every rank in `solver.comm` enters one `Allreduce` of the tally.
Each rank receives the same totals, and rank 0 reports the whole domain rather
than its own block.

The repair writes the interior only. Halos are left as the step left them, and
nothing downstream depends on that: [`max_rate`](@ref) exchanges before reading
anything at the top of the next iteration.
"""
function apply_positivity_floor!(solver::Solver, Q, rho_floor, e_floor,
                                 scope::Symbol;
                                 species_band::Real=solver.control.species_band)
    species = zeros(solver.equations.n_species)
    tally = _local_positivity_repair!(solver, Q, rho_floor, e_floor, scope,
                                      species_band, species)
    return _reduce_floor_tally(solver, tally, species)
end

function apply_positivity_floor!(solver::Solver, states::Vector{<:ConservedState},
                                 rho_floor, e_floor, scope::Symbol;
                                 species_band::Real=solver.control.species_band)
    acc = (0.0, 0.0, 0.0, 0.0, 0.0)
    species = zeros(solver.equations.n_species)
    for (ps, Q) in eachpatch(solver, states)
        acc = acc .+ _local_positivity_repair!(ps, Q, rho_floor, e_floor, scope,
                                               species_band, species)
    end
    return _reduce_floor_tally(solver, acc, species)
end

# One `Allreduce` of the five scalars and the per-species changes.
function _reduce_floor_tally(solver, tally, species)
    t0 = time_ns()
    red = MPI.Allreduce(vcat(collect(tally), species), +, solver.comm)
    _wait!(solver, t0)
    return (cells=round(Int, red[1]), low_energy=round(Int, red[2]), mass=red[3],
            energy=red[4], momentum=red[5], species=red[6:end])
end

function _local_positivity_repair!(solver::SolverLike, Q, rho_floor, e_floor,
                                   scope::Symbol, species_band::Real,
                                   species::Vector{Float64}=zeros(
                                       solver.equations.n_species))
    decomp = solver.decomp
    o1, o2, o3 = decomp.n_halo_d
    nx, ny, nz = decomp.n_local
    n_species = solver.equations.n_species
    m1, m2, m3 = solver.equations.i_mom
    i_energy = solver.equations.i_energy
    dV = cell_measure(solver)
    repair_e = scope === :internal_energy
    # The rescale below divides by the positive part of the species mass,
    # which is bounded away from zero only because the floor itself is;
    # `run!` guarantees that by construction and this makes it local.
    rho_floor > 0 || error("apply_positivity_floor!: rho_floor must be positive")
    T = eltype(Q)
    band = T(species_band)
    cells = 0.0; low_energy = 0.0; mass = 0.0; energy = 0.0; momentum = 0.0
    # Serial, as `max_rate` is: one pass per step over the same interior, and
    # only when a run enables the failsafe.
    @inbounds for k in 1:nz
        wk = quad_weight(solver, 3, k)
        for j in 1:ny
            wj = wk * quad_weight(solver, 2, j)
            for i in 1:nx
                I = CartesianIndex(i + o1, j + o2, k + o3)
                ρ = zero(T)
                q_min = T(Inf)
                q_max = T(-Inf)
                for sp in 1:n_species
                    q = Q[I, sp]
                    ρ += q
                    q_min = min(q_min, q)
                    q_max = max(q_max, q)
                end
                # The same test and arithmetic as the validation sweep, so the
                # clip acts on exactly the points a verdict would reject.
                clip = _outside_species_band(q_min, q_max, ρ, band)
                if !clip && ρ >= rho_floor
                    ke = (Q[I, m1]^2 + Q[I, m2]^2 + Q[I, m3]^2) / (2ρ)
                    Q[I, i_energy] - ke >= ρ * e_floor && continue
                end
                # dV / inv_J is the physical cell volume, since inv_J carries any
                # stretching. This follows `volume_integral`'s convention, so a
                # tally is comparable with an integral of the field it perturbs.
                vol = wj * quad_weight(solver, 1, i) * _edge_factors(solver, i, j, k, I) *
                      dV / solver.inv_J[I]
                repaired = false
                if clip && ρ >= rho_floor
                    repaired = true
                    pos = zero(T)
                    for sp in 1:n_species
                        pos += max(Q[I, sp], zero(T))
                    end
                    # pos >= ρ >= rho_floor > 0, so the rescale only ever
                    # shrinks and never divides by zero.
                    s = ρ / pos
                    for sp in 1:n_species
                        old = Q[I, sp]
                        Q[I, sp] = max(old, zero(T)) * s
                        species[sp] += (Q[I, sp] - old) * vol
                    end
                end
                if ρ < rho_floor
                    pos = zero(T)
                    for sp in 1:n_species
                        pos += max(Q[I, sp], zero(T))
                    end
                    if pos > 0
                        s = T(rho_floor) / pos
                        for sp in 1:n_species
                            old = Q[I, sp]
                            Q[I, sp] = max(old, zero(T)) * s
                            species[sp] += (Q[I, sp] - old) * vol
                        end
                    else
                        for sp in 1:n_species
                            old = Q[I, sp]
                            Q[I, sp] = sp == 1 ? T(rho_floor) : zero(T)
                            species[sp] += (Q[I, sp] - old) * vol
                        end
                    end
                    mass += (rho_floor - ρ) * vol
                    ρ = T(rho_floor)
                    repaired = true
                end
                ke = (Q[I, m1]^2 + Q[I, m2]^2 + Q[I, m3]^2) / (2ρ)
                target = ρ * T(e_floor)
                if Q[I, i_energy] - ke < target
                    low_energy += 1.0
                    if repair_e || Q[I, i_energy] < target
                        if Q[I, i_energy] > target && ke > 0
                            s = sqrt((Q[I, i_energy] - target) / ke)
                            pmag = sqrt(Q[I, m1]^2 + Q[I, m2]^2 + Q[I, m3]^2)
                            Q[I, m1] *= s; Q[I, m2] *= s; Q[I, m3] *= s
                            momentum += (1 - s) * pmag * vol
                        else
                            energy += (target + ke - Q[I, i_energy]) * vol
                            Q[I, i_energy] = target + ke
                        end
                        repaired = true
                    end
                end
                repaired && (cells += 1.0)
            end
        end
    end
    return (cells, low_energy, mass, energy, momentum)
end

"""
    record_floor!(solver, tally) -> FloorTally

Accumulate one step's positivity-floor `tally` onto `solver.floor_tally` and
return it. Bookkeeping only. [`run!`](@ref) does the reporting, warning on the
first firing of a run and again in summary when that run ends, because a front
carrying a handful of repaired cells for the length of a run would otherwise
produce thousands of identical warnings.

The tally is cumulative across `run!` calls on the same solver, as the
wall-clock fields beside it are.
"""
function record_floor!(solver::Solver, tally)
    ft = solver.floor_tally
    ft.steps += 1
    ft.cells += tally.cells
    ft.low_energy += tally.low_energy
    ft.mass += tally.mass
    ft.energy += tally.energy
    ft.momentum += tally.momentum
    isempty(ft.species) && append!(ft.species, zeros(length(tally.species)))
    ft.species .+= tally.species
    return ft
end

# Multi-patch counterparts of the per-step state operations `run!` composes.
# Each loops this rank's patches in the global order; none reduces.
apply_bcs!(solver::Solver, states::Vector{<:ConservedState}) =
    (foreach(((ps, Q),) -> apply_bcs!(ps, Q), eachpatch(solver, states)); states)

# Under subcycling the fine level filters itself inside `subcycled_step!` at
# its own step cadence, so the per-coarse-step pass here covers level 0 only.
function filter_state!(solver::Solver, states::Vector{<:ConservedState})
    subcycle = getfield(solver, :subcycle)
    for lev in getfield(solver, :levels)
        # A subcycled refined level filters at its own cadence inside the
        # driver (`_advance_level!`).
        subcycle && lev.index > 0 && continue
        _level_filter!(solver, lev, states)
    end
    _ledger!(solver, states, :filter)
    sync_patches!(solver, states)
    _ledger!(solver, states, :same_level)
    return states
end

_zero_state!(Q) = fill!(Q, 0)
_zero_state!(states::Vector{<:ConservedState}) =
    (foreach(Q -> fill!(Q, 0), states); states)
_reset_workspace!(w::Workspace) = (_zero_state!(w.dQ); _zero_state!(w.du); w)

# Savepoint copies for either state representation. A stacked tile's state
# is a view; its copy is a dense array of the same storage, and the restore
# assigns back into the view by broadcast, the form every backend runs as a
# kernel (a `copyto!` between a device view and a dense device array is not
# a contract every backend keeps). Dense host states keep `copyto!`.
_snapshot(Q) = copy(Q)
_snapshot(states::Vector{<:ConservedState}) = [_dense_copy(Q) for Q in states]
_dense_copy(Q::ConservedState) = ConservedState(_dense_copy(parent(Q)))
_dense_copy(a::AbstractArray) = copy(a)
_dense_copy(v::SubArray) = (d = similar(parent(v), size(v)); d .= v; d)
_restore_state!(dst, src) = copyto!(dst, src)
function _restore_state!(dst::Vector{<:ConservedState}, src::Vector{<:ConservedState})
    for i in eachindex(dst)
        _assign!(parent(dst[i]), parent(src[i]))
    end
    return dst
end

# The artificial coefficient arrays banked beside the state. `max_rate` sizes
# the first step after a rollback from them, so restoring the state alone
# leaves that step sized by the coefficients of the trajectory that failed. A
# failure that ends finite but enormous, as the spherical Noh does from
# cfl = 0.9, then drives the retry's first dt to zero and `:no_progress`
# follows before a single retried step is taken; a NaN failure re-fails the
# same way. With the arrays restored the retry starts as a checkpoint restart
# of the savepoint's instant would (`art_block`). Nothing is banked when the
# artificial properties are off; the arrays are then zero and never read.
_art_arrays(p) = (p.mu_art, p.beta_art, p.kappa_art, p.D_art...)
_art_snapshot(solver::Solver) =
    solver.art.enabled ?
    Vector{Any}[Any[_dense_copy(a) for a in _art_arrays(p)]
                for p in getfield(solver, :patches)] : Vector{Any}[]
function _bank_art!(saved, solver::Solver)
    for (arrays, p) in zip(saved, getfield(solver, :patches))
        for (s, a) in zip(arrays, _art_arrays(p))
            s .= a
        end
    end
    return saved
end
function _restore_art!(solver::Solver, saved)
    for (arrays, p) in zip(saved, getfield(solver, :patches))
        for (s, a) in zip(arrays, _art_arrays(p))
            a .= s
        end
    end
    return solver
end

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
_prime_rhs!(solver::Solver, Q, workspace) =
    (apply_bcs!(solver, Q); compute_rhs!(solver, Q, workspace.dQ, false); Q)
function _prime_rhs!(solver::Solver, states::Vector{<:ConservedState}, workspace)
    for lev in getfield(solver, :levels)
        status = _level_rhs!(solver, lev, states, workspace.dQ, false)
        _check_transport_status(solver, status)
    end
    return states
end

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

# Interface and level consistency before the pre-step reads. The single-patch
# path has neither and skips this entirely. `restrict = false` omits the
# restriction and keeps the shell imposition; `run!` passes it when the
# levels are as its previous post-step synchronization left them. The
# injection writes parent nodes at least `RESTRICT_MARGIN` from a parent-fed
# face and reads the fine nodes coincident with them. The shell imposition
# writes none of those, and the shared-plane averaging leaves them unchanged,
# since the planes were averaged after the last stage or filter pass, so a
# repeated restriction would write every node's own value.
_presync!(solver, Q, restrict::Bool=true) = Q
function _presync!(solver, states::Vector{<:ConservedState}, restrict::Bool=true)
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
    return states
end

# Test toggle: `run!` restricts before every step, as it did before the
# pre-step restriction became conditional, so a test can compare the two
# schedules bit for bit in one process.
const FORCE_PRESYNC_RESTRICT = Ref(false)

# Per-step level maintenance, after the state filter: restrict the fine state
# onto the covered coarse region, then re-impose the fine shell from the
# restricted coarse state. No-ops without refinement.
_post_step!(solver, Q) = Q
function _post_step!(solver, states::Vector{<:ConservedState})
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
    return states
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
    _cold(_run!)(solver, Q, workspace, tfinal, nmax, callback, control)
    return Q
end

# The loop takes the callback unspecialized, so that it is compiled once per
# solver type and not again for each script's callbacks, whose types are its
# own closures. It reaches them through two dynamic calls per step, which
# allocate nothing (see `_step_callbacks`).
function _run!(solver::Solver, Q, workspace::Workspace, tfinal, nmax::Int,
               @nospecialize(callback), control::StepControl)
    rank = MPI.Comm_rank(solver.comm)
    qx, wrapped = _crossing(Q)
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
    solver.step == 0 && _start_callbacks(callback, solver, qx, wrapped)::Bool &&
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
        _presync!(solver, Q, !levels_synced || FORCE_PRESYNC_RESTRICT[])
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
        flags = _step_callbacks(callback, solver, qx, wrapped)::Int
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
    work = workspace === nothing ? Workspace(Q) : workspace
    return run!(solver, Q, work; kwargs...)
end

"""
    filter_weight(solver, d) -> w

Relaxation weight for one [`filter_state!`](@ref) pass along dimension `d`, in
`(0, 1]`.

`solver.filter_cfl == 0` returns `1`, which is the unrelaxed formulation: the
filter is applied at full strength on every pass, so it removes energy per
*application*, not per unit time, and its effective dissipation depends on the
timestep. Halving the CFL then doubles the number of applications covering the
same interval and doubles the dissipation, so the subgrid dissipation does not
converge as `dt → 0` at fixed resolution.

A positive `filter_cfl` restores that convergence by scaling the weight with the
step taken and the rate of the direction swept,

    w_d = filter_interval · dt · r_d · √n / filter_cfl

capped at 1, with `r_d` the global maximum of the one-dimensional hyperbolic
rate `(|u_d| + c) / h_d` that [`max_rate`](@ref) returns beside the rate that
sized the step, and `n` the number of active dimensions. `dt · r_d` is the
directional CFL of the step taken, including `StepControl` backoff and the
shortening applied to land on a callback instant. `√n` is the ratio of the
Euclidean acoustic rate `c √(Σ 1/h_d²)` to the one-dimensional one on an
isotropic grid, so `filter_cfl` keeps the convention of `cfl`: in one
dimension exactly, and on an isotropic grid up to the advective share of the
rate, a pass is at full strength at or above `cfl = filter_cfl` whenever the
step is acoustic-limited, and the dissipation per unit time is invariant
below it.

The rate read is the hyperbolic one and not the maximum that sized the step,
for two reasons. Where the diffusive rate governs, the passes per unit time
would otherwise follow it: under a scalar β* on a grid of aspect ratio 16
the planar Noh run made fifteen times the passes of the same run on a
square grid and completed with a wrong solution, wall density 24 against 4.
And the rate is directional because the filter is: the coarse direction's
pass is relaxed against the coarse direction's rate, so a fine transverse
spacing, which raises the Euclidean rate by the aspect ratio, changes
nothing along the coarse direction. The step, the artificial coefficients
and the physical diffusivities are all outside the weight; the filter's
dissipation per unit time is set by the resolved advection and acoustics
alone.

Because `dt · r_d` is read and not `solver.cfl`, a shortened step filters
proportionally less, which removes the truncated-final-step artifact in
`bench/tgv_energy.jl`.

The weight is also 1 whenever `dt_prev == 0`, since there is no step to scale
against. That holds on a freshly built solver and after a rollback; inside
`run!` the step is recorded before the filter runs, so the relaxed weight is in
force from the first pass.
"""
function filter_weight(solver::SolverLike{T}, d::Int) where {T}
    solver.filter_cfl > 0 || return one(T)
    solver.dt_prev > 0 || return one(T)
    n_active = count(solver.decomp.active)
    w = solver.filter_interval * solver.dt_prev * solver.filter_rate_prev[d] *
        sqrt(T(n_active)) / solver.filter_cfl
    return min(one(T), w)
end

"""
    filter_state!(solver, Q)

Apply the compact filter to every conserved component of `Q` in place along every
active dimension, with batched per-dimension halo exchange and fold parity
routing from `cons_parity`: under the fold on the swept dimension each momentum
component takes the antipodal sign of its own velocity component, and partial
densities and energy are even. At the cylindrical axis this makes ρu_r and ρu_θ
odd and everything else even.

Every rank must call this function because each directional pass is a
distributed line solve. `solver.tmp_a` is scratch here. The filtered result
lands in the interior of `Q` only; each pass writes the halos from the exchange,
so they are left
inconsistent with the filtered interior until the next step exchanges again.

Under a positive `filter_cfl` the result is relaxed toward the filtered state
and not replaced by it, with the weight [`filter_weight`](@ref) supplies for
the dimension swept. The weight is a reduced quantity by construction, since
`filter_rate_prev` comes from the collective in `max_rate`, so every rank
blends by the same amount without a further reduction here.

Under `filter_weighting = :volume` on a non-uniform cell volume, which is any
cylindrical, spherical or stretched grid, each directional pass filters the
volume-weighted component and divides by the volume passed through the same
pass,

    q̄ = F_d(J q) / F_d(J),

with J = 1 / `inv_J`, the form of the public Pyranda implementation. Under
the fold on the swept dimension the product J·q folds with the component's
sign times `volume_parity`, since the cylindrical J = r is odd across
the axis. F_d(J) is rebuilt at every pass into `solver.tmp_b`, one line solve
per dimension beside the `n_cons` of the components; the weighted component
goes through `solver.tmp_a`, and its filtered image lands in the interior of
`Q` before the division and the relaxation. A uniform Cartesian grid skips
the weighting, so there the pass is the unweighted operator bit for bit, as
it is under the default `:none` everywhere.

On a patch with a child level, the unweighted pass leaves out the
high-pass residual of the covered nodes where the parent spacing does not
resolve the child's solution: the filtered state is `f + A⁻¹ M (B − A) f`
for the filter `A f̄ = B f`, with `M` zero at a fully covered node of an
interior row within two nodes of which the density has a relative undivided
fourth difference along the line above 0.01, and one elsewhere. Features of
the restricted child solution thinner than the parent spacing then do not
spread along the parent's lines. On such a patch the pass costs one more
line solve per component and direction.

Neither form conserves the volume integrals of the state on a closed line:
the defect of a pass is the closure rows' and is the same on a uniform and
on a clustered grid, and the unweighted operator preserves a uniform state
on any metric because it never reads the volume. The weighted form is the
less conservative at a cylindrical axis or a spherical pole, where the odd-parity
fold of the product does not have unit column sums, and it moves the Noh wall
deficit in opposite directions at the axis and at the origin; the
measurements are `bench/filter_conservation.jl`'s and the default rests on
them.
"""
function filter_state!(solver::SolverLike, Q)
    decomp = solver.decomp
    comps = [view(Q, :, :, :, c) for c in 1:solver.equations.n_cons]
    weighted = _weighted_filter(solver)
    masked = _masked_filter(solver, Q, weighted)
    for d in 1:3
        decomp.active[d] || continue
        w = filter_weight(solver, d)
        exchange_dim_batch!(comps, decomp, d)
        weighted && _filter_volume!(solver, d)
        mask = _child_mask!(solver, Q, comps, d, masked)
        for c in 1:solver.equations.n_cons
            σ = cons_parity(solver, d, c)
            if weighted
                _filter_weighted!(comps[c], solver, d, σ, w)
                continue
            end
            _filter_line!(solver.tmp_a, comps[c], solver, d, σ, mask)
            _reflux_filter!(solver, comps[c], solver.tmp_a, c, d, w)
            # w == 1 takes the original path exactly, so a pass at or above the reference
            # CFL stays bit-identical to the unrelaxed solver.
            if w == 1
                copy_interior!(comps[c], solver.tmp_a, decomp)
            else
                blend_interior!(comps[c], solver.tmp_a, w, decomp)
            end
        end
    end
    return Q
end

# --- The filter on a parent level ---------------------------------------------
#
# A pass of the compact filter is f + A⁻¹(B − A)f: the residual (B − A)f is an
# explicit high-pass stencil, local to a few nodes, and A⁻¹ spreads it along
# the line, decaying by about 0.7 per node at αf = 0.47. On a parent level the
# nodes a child covers carry the child's restricted solution. Where that
# solution holds a feature thinner than the parent spacing, the residual is
# large, and A⁻¹ carried it to the uncovered parent nodes and to the covered
# ones the child's ghost layers are interpolated from. From there it entered
# the child through those layers and through the fill of newly covered nodes
# at a regrid, and the density tag marked it. The pass on a patch with a child
# level therefore drops the residual at the nodes a child covers fully (mask
# byte 0xFF) where the parent spacing does not resolve the density along the
# line, at the node or within two nodes of it: f + A⁻¹ M (B − A) f, with M
# zero there and one elsewhere, formed as the plain pass less A⁻¹ applied to
# the dropped residual. The restriction that follows overwrites most of the
# covered nodes' own values. Where the restricted solution is smooth its
# residual is kept: dropping it there makes the pass weaker on one side of the
# coarse-fine face than on the other, a step that raised the error of smooth
# waves through a level by up to a factor of three. Closure rows keep their
# residual, so the mask applies on the interior rows only, and a line with no
# masked node takes the plain pass unchanged. The measurements are under
# bench/movinglevel.jl in reference/CALIBRATION_APPENDIX.md.

# Test and bench toggle: `false` gives every patch the plain pass, the
# comparison in bench/movinglevel.jl.
const MASK_CHILD_RESIDUAL = Ref(true)

# The resolution test of the mask: the undivided fourth difference of the
# density along the line, relative to the density, above the threshold at the
# node or at one of `CHILD_MASK_REACH` nodes on either side of it. The
# threshold is the default density tag's halved, the tag's hold level, on one
# dimension; the reach takes in the tails of a feature, whose residual the
# threshold alone left to spread. Both are measured, with the smooth level
# rows of test/convergence.jl unchanged under them; test and bench toggles.
const CHILD_MASK_THRESHOLD = Ref(0.01)
const CHILD_MASK_REACH = Ref(2)

# Whether the pass on this patch takes the mask: the patch has a child level
# holding a patch. The test reads the level hierarchy, which every rank of the
# patch holds, so every rank of a line takes the extra solve or none does. A
# stacked level takes the plain pass.
_has_child(solver::SolverLike, Q) = false
function _has_child(ps::PatchSolver, Q)
    MASK_CHILD_RESIDUAL[] || return false
    levels = getfield(ps.solver, :levels)
    child = ps.patch.level + 2
    child <= length(levels) && !isempty(levels[child].transfers) || return false
    return !(parent(Q) isa StackedArray)
end

@inline _fully_covered(m::UInt8) = m == 0xff
@inline _fully_covered(m) = !iszero(m)

# M along `d` as ones where the residual is dropped, in the free scratch
# `sensor_sp`, or `nothing` on a dimension a pair fold carries (its pass runs
# through the butterfly and stays plain). The halos of `Q` along `d` are
# current; at a self-paired fold the partial densities are mirror-filled here,
# as their own pass would. On device storage the covered bytes are uploaded
# into the scratch first and the body reads them back from it.
_masked_filter(solver::SolverLike, Q, weighted::Bool) =
    !weighted & _has_child(solver, Q)

# The pass's mask along `d`, `nothing` when the pass takes none, and the pass
# of one component under it. Kept out of `filter_state!` so that its body
# carries no branches on the mask, which bench/audit.jl's inference count
# reads as non-concrete control-flow values.
_child_mask!(solver::SolverLike, Q, comps, d::Int, masked::Bool) =
    masked ? _child_mask!(solver, Q, comps, d) : nothing

function _filter_line!(out, f, solver::SolverLike, d::Int, σ::Int, mask)
    filt_along!(out, f, solver, d, σ)
    mask === nothing || _drop_child_residual!(out, f, solver, d, σ, mask)
    return nothing
end

_child_mask!(solver::SolverLike, Q, comps, d::Int) = nothing
function _child_mask!(ps::PatchSolver, Q, comps, d::Int)
    fold = _fold_at(ps, d)
    fold === nothing || fold.pair === nothing || return nothing
    decomp = ps.decomp
    ns = ps.equations.n_species
    if fold !== nothing
        for c in 1:ns
            fold_fill!(comps[c], decomp, d, fold.lo, fold.hi, cons_parity(ps, d, c))
        end
    end
    m = ps.sensor_sp
    covered = ps.patch.covered
    cov = covered
    if _device_path(parent(Q))
        T = eltype(m)
        _upload!(m, T[b == 0xff ? one(T) : zero(T) for b in covered])
        cov = m
    end
    o1, o2, o3 = decomp.n_halo_d
    nx, ny, nz = decomp.n_local
    reach = clamp(CHILD_MASK_REACH[], 0, decomp.n_halo_d[d] - 2)
    pointwise!(_child_mask_point!, m, nx, ny, nz, m, cov, parent(Q), ns,
               eltype(m)(CHILD_MASK_THRESHOLD[]), reach, d, o1, o2, o3)
    return m
end

@inline function _density_at(Q, J, ns)
    acc = zero(eltype(Q))
    for s in 1:ns
        acc += @inbounds Q[J, s]
    end
    return acc
end

@inline function _child_mask_point!(m, cov, Q, ns, thr, reach, d, o1, o2, o3,
                                    i, j, k)
    @inbounds begin
        I = CartesianIndex(i + o1, j + o2, k + o3)
        e = CartesianIndex(Int(d == 1), Int(d == 2), Int(d == 3))
        under = false
        for t in -reach:reach
            J = I + t * e
            ρ0 = _density_at(Q, J, ns)
            δ4 = _density_at(Q, J - 2e, ns) - 4 * _density_at(Q, J - e, ns) + 6 * ρ0 -
                 4 * _density_at(Q, J + e, ns) + _density_at(Q, J + 2e, ns)
            under |= abs(δ4) > thr * abs(ρ0)
        end
        drop = _fully_covered(cov[I]) & under
        m[I] = ifelse(drop, one(eltype(m)), zero(eltype(m)))
    end
    return nothing
end

# The interior row of the residual (B − A)f: `r[1]` at the node and `r[m + 1]`
# on its two neighbors at distance m, for the plan's own prescaled
# right-hand side and left-hand side.
_residual_stencil(plan::DevicePlan) = _residual_stencil(plan.host)
_residual_stencil(plan::DirPlan) =
    _residual_stencil(plan.a0, plan.ci, (plan.scheme.alpha,))
_residual_stencil(plan::BandPlan) = _residual_stencil(plan.a0, plan.ci, plan.scheme.lhs)
function _residual_stencil(a0::T, ci, lhs) where {T}
    length(ci) <= 4 && length(lhs) <= 4 ||
        error("the masked filter pass takes a half-width of at most 4")
    at(v, m) = m <= length(v) ? T(v[m]) : zero(T)
    return ntuple(m -> m == 1 ? a0 - one(T) : at(ci, m - 1) - at(lhs, m - 1), 5)
end

_rows_closed(plan::DevicePlan) = _rows_closed(plan.host)
_rows_closed(plan) = (plan.lo_closed ? length(plan.clo) : 0,
                      plan.hi_closed ? length(plan.chi) : 0)

# The residual at the nodes of the interior rows along `d` the mask selects,
# zero elsewhere. `f` carries current halos along `d`, exchanged or
# mirror-filled.
@inline function _child_residual_point!(s, f, mask, r, M, d, row_lo, row_hi,
                                        o1, o2, o3, i, j, k)
    @inbounds begin
        I = CartesianIndex(i + o1, j + o2, k + o3)
        e = CartesianIndex(Int(d == 1), Int(d == 2), Int(d == 3))
        acc = r[1] * f[I]
        for m in 1:M
            acc += r[m + 1] * (f[I + m * e] + f[I - m * e])
        end
        row = d == 1 ? i : d == 2 ? j : k
        s[I] = ifelse((row_lo <= row) & (row <= row_hi) & _fully_covered(mask[I]),
                      acc, zero(acc))
    end
    return nothing
end

@inline function _subtract_interior_point!(dst, src, o1, o2, o3, i, j, k)
    @inbounds begin
        I = CartesianIndex(i + o1, j + o2, k + o3)
        dst[I] -= src[I]
    end
    return nothing
end

# `out` holds the plain pass of `f` along `d`; subtract A⁻¹ of the residual
# the mask drops, through `solver.tmp_b`. A pair fold's pass runs through its
# butterfly and is left plain.
function _drop_child_residual!(out, f, solver::SolverLike, d::Int, σ::Int, mask)
    fold = _fold_at(solver, d)
    fold === nothing || fold.pair === nothing || return out
    plan = fold === nothing ? _operator_plan(solver.filter_plans, d) :
                              _fold_plan(fold, σ, Val(:filter), 1, false)
    decomp = solver.decomp
    o1, o2, o3 = decomp.n_halo_d
    nx, ny, nz = decomp.n_local
    nlo, nhi = _rows_closed(plan)
    r = _residual_stencil(plan)
    M = something(findlast(m -> !iszero(r[m + 1]), 1:4), 0)
    s = solver.tmp_b
    pointwise!(_child_residual_point!, s, nx, ny, nz, s, f, mask, r, M, d,
               nlo + 1, decomp.n_local[d] - nhi, o1, o2, o3)
    solve_along!(s, plan, s, decomp)
    pointwise!(_subtract_interior_point!, out, nx, ny, nz, out, s, o1, o2, o3)
    return out
end

# Whether this solver's state filter weights by the cell volume: the option is
# on and the volume is not uniform. A refined level is unstretched, so it takes
# the weighted path on the axisymmetric cylindrical metric only, with its own
# J = r, which is analytic on the ghost layers its interface rows read.
_weighted_filter(solver::SolverLike) =
    solver.filter_weighting === :volume &&
    !(solver.metric isa CartesianMetric && all(isnothing, solver.stretch))

# 1 / F_d(J) over the interior of `solver.tmp_b`. J is filled over the padded
# extent of `solver.tmp_a` from `inv_J`, which is analytic there, so the pass
# reads current halos along `d` without an exchange; a fold on `d` folds J
# with its own parity.
function _filter_volume!(solver::SolverLike, d::Int)
    decomp = solver.decomp
    n1, n2, n3 = padded_extent(decomp)
    pointwise!(_reciprocal_point!, solver.tmp_a, n1, n2, n3, solver.tmp_a,
               solver.inv_J)
    filt_along!(solver.tmp_b, solver.tmp_a, solver, d,
                volume_parity(solver.metric, d))
    o1, o2, o3 = decomp.n_halo_d
    nx, ny, nz = decomp.n_local
    pointwise!(_reciprocal_interior_point!, solver.tmp_b, nx, ny, nz,
               solver.tmp_b, o1, o2, o3)
    return solver.tmp_b
end

# One component's volume-weighted pass along `d`: J·q over the padded extent
# into `solver.tmp_a` (its halos along `d` were exchanged with `q`; no other
# halo is read), the filter of that into the interior of `q` itself, then the
# division by F_d(J) from `_filter_volume!`, blended with the unfiltered
# component read back as (J q) / J where the pass is relaxed.
function _filter_weighted!(q, solver::SolverLike, d::Int, σ::Int, w)
    decomp = solver.decomp
    n1, n2, n3 = padded_extent(decomp)
    pointwise!(_volume_weight_point!, solver.tmp_a, n1, n2, n3, solver.tmp_a, q,
               solver.inv_J)
    filt_along!(q, solver.tmp_a, solver, d, σ * volume_parity(solver.metric, d))
    o1, o2, o3 = decomp.n_halo_d
    nx, ny, nz = decomp.n_local
    if w == 1
        pointwise!(_scale_interior_point!, q, nx, ny, nz, q, solver.tmp_b,
                   o1, o2, o3)
    else
        pointwise!(_weighted_blend_point!, q, nx, ny, nz, q, solver.tmp_a,
                   solver.inv_J, solver.tmp_b, one(w) - w, w, o1, o2, o3)
    end
    return q
end

@inline function _reciprocal_point!(out, a, i, j, k)
    @inbounds out[i, j, k] = 1 / a[i, j, k]
    return nothing
end

@inline function _reciprocal_interior_point!(a, o1, o2, o3, i, j, k)
    @inbounds begin
        I = CartesianIndex(i + o1, j + o2, k + o3)
        a[I] = 1 / a[I]
    end
    return nothing
end

@inline function _volume_weight_point!(out, q, inv_J, i, j, k)
    @inbounds out[i, j, k] = q[i, j, k] / inv_J[i, j, k]
    return nothing
end

@inline function _scale_interior_point!(q, s, o1, o2, o3, i, j, k)
    @inbounds begin
        I = CartesianIndex(i + o1, j + o2, k + o3)
        q[I] *= s[I]
    end
    return nothing
end

@inline function _weighted_blend_point!(q, jq, inv_J, inv_FJ, w1, w, o1, o2, o3,
                                        i, j, k)
    @inbounds begin
        I = CartesianIndex(i + o1, j + o2, k + o3)
        q[I] = w1 * (jq[I] * inv_J[I]) + w * (q[I] * inv_FJ[I])
    end
    return nothing
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
