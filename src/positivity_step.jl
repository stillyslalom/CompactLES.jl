# The positivity limiter in the step: the bounds of a run, the limited stage
# on Cartesian and radial lines, and the limited filter pass. The header of
# positivity_setup.jl describes the method.

# --- The bounds of a run ---------------------------------------------------

# ε from the state entering `run!`, the same rule at a restart and after a
# phase change: `LIMITER_FRACTION` of the global minimum of ρ and of ρe over
# the interior points where each is positive. Collective over the solver's
# communicator. Returns whether the limiter acts in this run. A limited run
# can end with a point marginally below zero, a cell whose node terms alone
# took it there, and the next run starts from that state; the point's bound is
# the floor relative to its E (`_limiter_cell_bound`), above its own ρe, so its
# faces take the first-order flux where its half states need it and it is
# counted unguaranteed, and the rest of the grid keeps the scale of its own
# state. Only a state without a positive density or internal energy anywhere
# gives no scale, and the run then proceeds unlimited with a warning, as the
# failsafe does.
_positivity_setup!(solver, Q) = false
function _positivity_setup!(solver, Q::Union{ConservedState,Vector{<:ConservedState}})
    lim = getfield(solver, :positivity)
    lim === nothing && return false
    # The limiter is off by default, so its bounds and the limiter
    # construction a refined run reaches through them sit behind `_cold`; a
    # declared limiter type in the call would otherwise have them compiled
    # with every solver type's run loop.
    return _positivity_bounds!(lim, _cold(solver), Q)::Bool
end

_positivity_bounds!(lim::PositivityLimiter, solver, Q) =
    _set_limiter_bounds!((lim,), solver, _limiter_minima(solver, Q))

# The patches' minima, over every patch this rank holds, reduced once.
function _positivity_bounds!(pl::PatchLimiters, solver, states)
    lims = _follow_patches!(solver, pl)
    ρmin = Inf
    emin = Inf
    for (ps, Q) in eachpatch(solver, states)
        r, e = _limiter_minima(ps, Q)
        ρmin = min(ρmin, r)
        emin = min(emin, e)
    end
    return _set_limiter_bounds!(lims, solver, (ρmin, emin))
end

# The limiters of the patches this rank holds now, built for a patch that has
# none (every patch of a refined solver as its first `run!` starts, and the
# patches a regrid replaced) with the run's bounds, and dropped with a patch a
# regrid dropped, its counts kept. Rank-local: a patch's limiter reads only
# the patch.
function _follow_patches!(solver, pl::PatchLimiters)
    patches = getfield(solver, :patches)
    if !(length(pl.patches) == length(patches) &&
         all(i -> pl.patches[i] === patches[i], eachindex(patches)))
        _rebuild_limiters!(solver, pl)
    end
    # The exempt nodes are written over the interior by a regrid; a θ pass
    # reads them on the rank halos as well, as the rank holding them does.
    for (lim, p) in zip(pl.limiters, patches)
        isempty(lim.free[4]) || exchange_halos!(lim.free[4], p.decomp)
    end
    return pl.limiters
end

function _rebuild_limiters!(solver, pl::PatchLimiters)
    patches = getfield(solver, :patches)
    held = IdDict{Any,Any}(zip(pl.patches, pl.limiters))
    template = isempty(pl.limiters) ? nothing : first(pl.limiters)
    limiters = Any[]
    for p in patches
        lim = pop!(held, p, nothing)
        if lim === nothing
            lim = PositivityLimiter(PatchSolver(solver, p); separate=pl.separate)
            if template !== nothing
                lim.eps_rho = template.eps_rho
                lim.eps_e = template.eps_e
                lim.active = template.active
            end
        end
        push!(limiters, lim)
    end
    for lim in values(held)
        pl.retired .+= lim.counts
    end
    pl.limiters = limiters
    pl.patches = Any[p for p in patches]
    return pl
end

# This rank's minimum ρ over the interior points where it is positive, and of
# ρe where both are. A state in device storage is copied to the host for the
# sweep, once per `run!`.
_limiter_minima(solver, Q) =
    _device_path(Q) ?
    _limiter_minima_host(solver, ConservedState(Array(_dense_copy(parent(Q))))) :
    _limiter_minima_host(solver, Q)

function _limiter_minima_host(solver, Q)
    decomp = solver.decomp
    o1, o2, o3 = decomp.n_halo_d
    nx, ny, nz = decomp.n_local
    ns = solver.equations.n_species
    m1, m2, m3 = solver.equations.i_mom
    ie = solver.equations.i_energy
    ρmin = Inf
    emin = Inf
    @inbounds for k in 1:nz, j in 1:ny, i in 1:nx
        I = CartesianIndex(i + o1, j + o2, k + o3)
        ρ = zero(eltype(Q))
        for sp in 1:ns
            ρ += Q[I, sp]
        end
        ρ > 0 || continue
        ρmin = min(ρmin, ρ)
        e = Q[I, ie] - (Q[I, m1]^2 + Q[I, m2]^2 + Q[I, m3]^2) / (2ρ)
        e > 0 && (emin = min(emin, e))
    end
    return ρmin, emin
end

# The run's ε on every limiter of `lims` from this rank's minima, reduced over
# the solver's communicator.
function _set_limiter_bounds!(lims, solver, minima)
    red = MPI.Allreduce([minima[1], minima[2]], min, solver.comm)
    active = isfinite(red[1]) && isfinite(red[2])
    for lim in lims
        T = eltype(lim.eps_rho)
        lim.active = active
        if active
            lim.eps_rho = T(LIMITER_FRACTION * red[1])
            lim.eps_e = T(LIMITER_FRACTION * red[2])
        end
    end
    if !active && MPI.Comm_rank(solver.comm) == 0
        @warn "run!: the positivity limiter is inactive in this run. No point " *
              "of the state entering it has a positive density and internal " *
              "energy, so there is no bound to scale."
    end
    return active
end

"""
    positivity_counts(solver) -> NamedTuple

What the positivity limiter of `solver` has done since it was built, summed
over the ranks and the patches: the interior faces tested at the Runge–Kutta
stages (a shared node's end face once) and at the filter passes, the faces
limited in each, the unguaranteed cell sides, where the first-order bound
failed or the first-order half state was not admissible, on a radial grid
also those of a cell whose source and boundary terms alone would leave the
bound, and on a radial grid the stage faces limited at r = 0 (`axis_limited`, counted in `stage_limited` too). Collective
over the solver's communicator.
"""
function positivity_counts(solver)
    lim = getfield(solver, :positivity)
    lim === nothing && throw(ArgumentError(
        "positivity_counts: the solver was built without positivity_limiter"))
    local_counts = lim isa PositivityLimiter ? lim.counts :
                   foldl((s, l) -> s .+ l.counts, lim.limiters; init=copy(lim.retired))
    c = MPI.Allreduce(local_counts, +, solver.comm)
    return (stage_faces=c[1], stage_limited=c[2], filter_faces=c[3],
            filter_limited=c[4], unguaranteed=c[5], axis_limited=c[6])
end

# --- The limited step ------------------------------------------------------

# `step!` with the stage limiter between each stage's right-hand side and its
# low-storage update; with no face limited it makes the calls of `step!` in
# the same order and the same arithmetic.
function _limited_run_step!(solver, Q, workspace, dt, prepared::Bool, control)
    lim = getfield(solver, :positivity)
    dQ, du = workspace.dQ, workspace.du
    decomp = solver.decomp
    for stage in 1:5
        solver.tstage = solver.t + oftype(solver.t, RKC[stage]) * dt
        first_prepared = prepared && stage == 1
        if !first_prepared
            _ledger_open!(solver, Q)
            apply_bcs!(solver, Q)
            _ledger!(solver, Q, :wall_enforce)
        end
        T = eltype(lim.eps_rho)
        compute_rhs!(solver, Q, _LimiterRHS(dQ, lim, T(RKA[stage]), T(dt)),
                     first_prepared)
        _ledger_faces!(solver, 1)
        _limit_stage!(lim, solver, Q, dQ, du, stage, dt)
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
    return nothing
end

# The multi-patch `step!`, with the stage limiter of every patch
# (`_limit_level_stage!`) between the right-hand sides and the stage updates,
# level by level as `step!` takes them; a subcycled solver takes
# `subcycled_step!`'s schedule with the limiter (`_advance_level!`). A patch's
# lines end at a same-level interface as at a wall, with the weights of the
# interface rows or of an open end, and `_exchange_interface_theta!` has what
# the two patches share there. A refined patch's lines end at a coarse-fine
# face in the same way, and the end node there, which the parent's shell
# overwrites after the stage, is left alone as a Dirichlet node is. Returns a
# `SolverFailure` where the subcycled schedule rejects a substep, and
# `nothing` otherwise.
function _limited_run_step!(solver, states::Vector{<:ConservedState}, workspace, dt,
                            prepared::Bool, control)
    lims = _follow_patches!(solver, getfield(solver, :positivity))
    getfield(solver, :subcycle) &&
        return _limited_subcycled_step!(solver, states, workspace, dt, prepared, control,
                                        lims)
    patches = getfield(solver, :patches)
    levels = getfield(solver, :levels)
    dQs, dus = workspace.dQ, workspace.du
    for stage in 1:5
        solver.tstage = solver.t + oftype(solver.t, RKC[stage]) * dt
        first_prepared = prepared && stage == 1
        limited = _limiter_stage_rhs(lims, dQs, stage, dt)
        for lev in levels
            status = _level_rhs!(solver, lev, states, limited, first_prepared)
            _check_transport_status(solver, status)
        end
        for lev in levels
            _limit_level_stage!(lims, solver, lev, states, dQs, dus, stage, dt)
        end
        for lev in levels
            _ledger_open!(solver, states, lev)
            _level_update!(solver, lev, states, dQs, dus, RKA[stage], RKB[stage], dt)
            _ledger_update!(solver, states, dQs, RKA[stage], RKB[stage], dt, lev)
        end
        _ledger_open!(solver, states)
        sync_patches!(solver, states)
        _ledger!(solver, states, :same_level)
        if length(levels) > 1
            prolong_level_ghosts!(solver, states)
            _ledger!(solver, states, :shell)
        end
    end
    solver.tstage = solver.t + dt
    _ledger_open!(solver, states)
    for (i, p) in enumerate(patches)
        apply_bcs!(PatchSolver(solver, p), states[i])
    end
    _ledger!(solver, states, :wall_enforce)
    _validate_transport_state!(solver, states)
    return nothing
end

# `_subcycled_run_step!` with the stage limiter on every level's stages and the
# filter limiter on every refined level's passes (`_advance_level!`).
function _limited_subcycled_step!(solver, states, workspace, dt, prepared::Bool,
                                  control, lims)
    status, guard = _subcycled_step_status!(solver, states, workspace.dQ, workspace.du,
                                             dt, prepared, control, lims)
    status == 0 && return nothing
    status == SUBSTEP_CFL && return _substep_cfl_failure(solver, guard, dt, control)
    status == SUBSTEP_INVALID && return _substep_invalid_failure(solver, guard, dt)
    _check_transport_status(solver, status)
    return nothing
end

# The right-hand side arrays a limited stage hands `_level_rhs!`, one per
# patch (`_LimiterRHS`); the plain arrays where the step is not limited.
_limiter_stage_rhs(::Nothing, dQs, stage::Int, dt) = dQs
function _limiter_stage_rhs(lims::Vector{Any}, dQs, stage::Int, dt)
    T = eltype(first(lims).eps_rho)
    return [_LimiterRHS(dQs[i], lims[i], T(RKA[stage]), T(dt)) for i in eachindex(dQs)]
end

# The time advance of stage k, c_{k+1} − c_k with c_6 = 1.
_stage_advance(stage::Int) = (stage < 5 ? RKC[stage+1] : 1.0) - RKC[stage]

# The right-hand side array a limited stage hands `compute_rhs!`: the
# workspace's dQ, indexed through, with the divergence of each flux taken
# once more out of the fused subtraction. `div_subtract_along!` on it runs the
# derivative into scratch and subtracts it, the two-pass form the folds take,
# bit for bit the fused one, and updates the direction's register
# r_d ← A r_d + dt D_d F_d and the face register at the global low face from
# the same derivative, so the registers cost no line solve of their own. Every
# other phase of the right-hand side reads and writes it as dQ.
struct _LimiterRHS{T,D<:AbstractArray{T,4},L} <: AbstractArray{T,4}
    dQ::D
    lim::L
    A::T
    dt::T
end
Base.parent(x::_LimiterRHS) = parent(x.dQ)
Base.size(x::_LimiterRHS) = size(x.dQ)
Base.axes(x::_LimiterRHS) = axes(x.dQ)
Base.IndexStyle(::Type{<:_LimiterRHS{T,D}}) where {T,D} = IndexStyle(D)
Base.@propagate_inbounds Base.getindex(x::_LimiterRHS, I...) = getindex(x.dQ, I...)
Base.@propagate_inbounds Base.setindex!(x::_LimiterRHS, v, I...) =
    setindex!(x.dQ, v, I...)
Base.view(x::_LimiterRHS, I...) = view(x.dQ, I...)
@inline _cpu_storage(x::_LimiterRHS) = _cpu_storage(x.dQ)
@inline _kernel_arg(x::_LimiterRHS) = _kernel_arg(x.dQ)
# It crosses into the ghost-differenced divergence of an interface dimension
# whole (rhs.jl), where the gradient plans' subtractions take the registers.
@inline _ghost_rhs_arg(x::_LimiterRHS) = x
_rhs_state(x::_LimiterRHS) = x

div_subtract_along!(dQ::_LimiterRHS, c::Int, f, solver::SolverLike, d::Int, σf::Int,
                    inv_J) = _limited_subtract!(dQ, c, c, f, solver, d, σf, inv_J)

# The radial pressure gradient, kept in register n_cons + 1 of its direction.
pressure_subtract_along!(dQ::_LimiterRHS, c::Int, solver::SolverLike, d::Int) =
    _limited_subtract!(dQ, c, solver.equations.n_cons + 1, solver.p, solver, d, 1,
                       solver.inv_h[d])

# The subtraction from component `c` and the update of register `slot`.
function _limited_subtract!(dQ::_LimiterRHS, c::Int, slot::Int, f, solver, d::Int,
                            σf::Int, inv_J)
    div_along!(solver.tmp_a, f, solver, d, σf)
    return _limited_registers!(dQ, c, slot, f, solver, d, σf, inv_J)
end

# Under `interface_flux = :ghost` the flux of an interface dimension, and the
# radial pressure there, go through the gradient plans
# (`_ghost_flux_divergence!`, `_ghost_flux_solves!`), whose subtraction takes
# the registers in the same way. The derivative is the one
# `_ext_subtract_along!` forms, through the fused route's plan where that
# route is taken, so the subtraction is bit for bit the unlimited one.
_ext_subtract_along!(dQ::_LimiterRHS, c::Int, f, solver::SolverLike, d::Int, inv_J,
                     σf::Int) = _limited_ext_subtract!(dQ, c, c, f, solver, d, σf, inv_J)

_ext_pressure_subtract!(dQ::_LimiterRHS, c::Int, solver::SolverLike, d::Int) =
    _limited_ext_subtract!(dQ, c, solver.equations.n_cons + 1, solver.p, solver, d, 1,
                           solver.inv_h[d])

function _limited_ext_subtract!(dQ::_LimiterRHS, c::Int, slot::Int, f, solver, d::Int,
                                σf::Int, inv_J)
    fold = _fold_at(solver, d)
    if fold !== nothing && !_fold_active(fold) && !(fold_dplan(fold, 1) isa DevicePlan)
        apply_along!(solver.tmp_a, fold_dplan(fold, 1), f, solver.decomp)
    else
        deriv_along!(solver.tmp_a, f, solver, d, σf)
    end
    return _limited_registers!(dQ, c, slot, f, solver, d, σf, inv_J)
end

# The derivative of `f` in `tmp_a` subtracted from component `c` and taken into
# register `slot`, and the face registers at the line's ends updated from it:
# at the global low face from node 1's point flux, the relation of a periodic
# line or an open end, or the fold (`_limiter_anchor_point!`), and at a high
# end closed by the interface rows from its last node's point flux.
function _limited_registers!(dQ::_LimiterRHS, c::Int, slot::Int, f, solver, d::Int,
                             σf::Int, inv_J)
    lim = dQ.lim
    decomp = solver.decomp
    nx, ny, nz = decomp.n_local
    o1, o2, o3 = decomp.n_halo_d
    pointwise!(_limiter_subtract_point!, dQ.dQ, nx, ny, nz,
               dQ.dQ, lim.registers[d, slot], solver.tmp_a, inv_J, c, dQ.A, dQ.dt,
               o1, o2, o3)
    if decomp.coords[d] == 0
        nA, nB = _transverse(decomp.n_local, d)
        pointwise!(_limiter_anchor_point!, solver.tmp_a, nA, nB, 1,
                   lim.anchor[d], f, solver.tmp_a, lim.weights[d], dQ.A, dQ.dt, slot, d,
                   lim.periodic[d] || lim.open[d][1], lim.fold_lo[d], σf,
                   lim.anchor_face[d], lim.deriv_lhs, lim.deriv_rhs, o1, o2, o3)
    end
    if lim.interface[d][2] && !lim.open[d][2] && decomp.coords[d] == decomp.dims[d] - 1
        nA, nB = _transverse(decomp.n_local, d)
        pointwise!(_limiter_end_anchor_point!, solver.tmp_a, nA, nB, 1,
                   lim.anchor_hi[d], f, dQ.A, dQ.dt, slot, d, decomp.n_local[d],
                   o1, o2, o3)
    end
    return dQ
end

function _limit_stage!(lim::PositivityLimiter, solver, Q, dQ, du, stage::Int, dt)
    lim.active || return nothing
    for d in 1:3
        solver.decomp.active[d] || continue
        _stage_theta!(lim, solver, Q, dQ, du, stage, dt, d) &&
            _stage_correct!(lim, solver, Q, dQ, stage, dt, d)
    end
    return nothing
end

# The stage limiter of every patch this rank holds on a level, a direction at
# a time: each patch's θ, the exchange that gives the faces about a shared node
# the same θ on both patches, then each patch's corrections. A patch without a
# same-level interface shares its face values and θ with the other patches of
# its extent (the workspace) and takes its corrections at once. The level's
# shared planes are the root's records at the root and the level's records of
# the direction, in the level's communicator, below it.
_limit_level_stage!(::Nothing, solver, lev, states, dQs, dus, stage::Int, dt) = nothing
function _limit_level_stage!(lims::Vector{Any}, solver, lev, states, dQs, dus,
                             stage::Int, dt)
    first(lims).active || return nothing
    isempty(lev.patches) && return nothing
    patches = getfield(solver, :patches)
    root = lev.index == 0
    for d in 1:3
        PatchSolver(solver, patches[first(lev.patches)]).decomp.active[d] || continue
        due = falses(length(patches))
        for i in lev.patches
            ps = PatchSolver(solver, patches[i])
            lim = lims[i]
            due[i] = _stage_theta!(lim, ps, states[i], dQs[i], dus[i], stage, dt, d)
            if due[i] && !any(any, lim.interface)
                _stage_correct!(lim, ps, states[i], dQs[i], stage, dt, d)
                due[i] = false
            end
        end
        records = root ? solver.plane_pairs : lev.plane_pairs[d]
        _exchange_interface_theta!(solver, lims, d, due, records,
                                   root ? solver.comm : lev.level_comm.comm)
        for i in lev.patches
            due[i] && _stage_correct!(lims[i], PatchSolver(solver, patches[i]), states[i],
                                      dQs[i], stage, dt, d)
        end
    end
    return nothing
end

# The face values and θ of direction d at a stage, and whether a face is
# limited where this rank's nodes take its correction; a radial line of a
# patch without a same-level interface takes its whole pass here.
function _stage_theta!(lim::PositivityLimiter, solver, Q, dQ, du, stage::Int, dt, d::Int)
    T = eltype(lim.eps_rho)
    B = T(RKB[stage])
    τ = T(_stage_advance(stage) * dt)
    _line_faces!(lim, solver, Q, d, 1, B, zero(T))
    decomp = solver.decomp
    # An open end's face value is the running sum's end itself.
    if _limiter_ends(lim, decomp, d)[5] && !lim.open[d][2]
        nA, nB = _transverse(decomp.n_local, d)
        pointwise!(_limiter_end_face_point!, solver.tmp_a, nA, nB, 1,
                   lim.faces, lim.anchor_hi[d], B, lim.n_comp[d], decomp.n_local[d], d,
                   decomp.n_halo_d...)
    end
    if lim.radial[d]
        # A tile's correction waits for the exchange of θ at its shared faces.
        any(any, lim.interface) &&
            return _radial_theta!(lim, solver, Q, dQ, du, d, T(RKA[stage]), B, T(dt), τ,
                                  lim.theta)
        _limit_radial!(lim, solver, Q, dQ, du, d, T(RKA[stage]), B, T(dt), τ)
        return false
    end
    limited, shared = _limit_faces!(lim, solver, Q, d, τ, 1)
    lim.counts[2] += limited
    return limited + shared > 0
end

function _stage_correct!(lim::PositivityLimiter, solver, Q, dQ, stage::Int, dt, d::Int)
    decomp = solver.decomp
    T = eltype(lim.eps_rho)
    B = T(RKB[stage])
    τ = T(_stage_advance(stage) * dt)
    lim.radial[d] && return _radial_correct!(lim, solver, Q, dQ, d, B, T(dt), τ, lim.theta)
    o1, o2, o3 = decomp.n_halo_d
    inv_B = one(T) / B
    inv_Bdt = one(T) / (B * T(dt))
    n = decomp.n_local[d]
    nA, nB = _transverse(decomp.n_local, d)
    pointwise!(_limiter_correct_point!, solver.tmp_a, n, nA, nB,
               _LinearState(dQ), lim.register_fields[d], lim.faces, lim.theta,
               _LinearState(Q), (solver.u, solver.v, solver.w), solver.c, solver.p,
               lim.weights[d], d, one(T), τ, 1, (inv_B, inv_Bdt),
               _limiter_layout(solver), _limiter_ends(lim, decomp, d), n,
               false, lim.volume, solver.inv_J, lim.areal[d],
               (lim.anchor[d], lim.anchor_hi[d]), lim.free, o1, o2, o3)
    return nothing
end

# A shared interface node is the last node of one patch and the first of the
# other, and its two copies are averaged after every stage, so the node's
# update is the mean of the two patches' updates, each of the node's end
# weight. Each patch keeps its own update of the node within the node's bound,
# and their mean, the admissible set being convex, is within it as well. The
# update takes the face beside the node and the face at the node itself, the
# patch's end face, which carries the point flux there under the closure rows
# and the open end's face value under `:ghost`. The first-order flux of
# that face is the point flux of the node's state, the same on both sides, so
# where the two patches' fluxes there agree (the inviscid flux of the averaged
# state) limiting it on both sides by one θ conserves, and where they differ
# (the artificial fluxes, from sensors each patch smooths with its own closure
# rows) it moves the scheme's own mismatch toward zero. The node's limit is
# taken along the segment that blends both its faces by one θ, so it holds for
# any smaller θ on both, and the θ pass gives the end face its neighbour's θ.
# The exchange below gives both patches the smaller of their two, at the end
# face and at each patch's face beside the node, from the shared-plane records
# the averaging uses. Collective among the ranks holding the shared planes. A
# filter pass whose rows conserve at the node takes the same exchange
# (`_limited_level_filter!`), with the end face blended toward zero; `tally` is
# the slot of `counts` that its newly limited faces enter.
function _exchange_interface_theta!(solver, lims, d::Int, due, records, comm,
                                     tally::Int=2)
    isempty(records) && return nothing
    _device_path(lims[first(records).patch].theta) &&
        return _exchange_interface_theta_staged!(solver, lims, d, due, records, comm, tally)
    patches = getfield(solver, :patches)
    me = MPI.Comm_rank(comm)
    normal(pl) = any(lims[pl.patch].interface[d])
    reqs = MPI.Request[]
    for pl in records
        normal(pl) && pl.partner != me || continue
        push!(reqs, MPI.Irecv!(pl.buf, comm; source=pl.partner, tag=pl.tag))
    end
    for pl in records
        normal(pl) && pl.partner != me || continue
        lim = lims[pl.patch]
        s = _end_shift(patches[pl.patch].decomp, pl.mine, d)
        idx = 1
        for I in CartesianIndices(pl.mine)
            pl.sbuf[idx] = lim.theta[_shift_index(I, d, s)]
            idx += 1
        end
        push!(reqs, MPI.Isend(pl.sbuf, comm; dest=pl.partner, tag=pl.sendtag))
    end
    MPI.Waitall(reqs)
    for pl in records
        normal(pl) || continue
        lim = lims[pl.patch]
        s = _end_shift(patches[pl.patch].decomp, pl.mine, d)
        local_pair = pl.partner == me
        other = local_pair ? lims[pl.partner_patch].theta : nothing
        so = local_pair ? _end_shift(patches[pl.partner_patch].decomp, pl.theirs, d) : 0
        theirs = local_pair ? CartesianIndices(pl.theirs) : nothing
        decomp = patches[pl.patch].decomp
        idx = 1
        for I in CartesianIndices(pl.mine)
            Ie = _shift_index(I, d, s)
            mine = lim.theta[Ie]
            partner = local_pair ? other[_shift_index(theirs[idx], d, so)] : pl.buf[idx]
            idx += 1
            # A level's records reach into the transverse halos, where no θ
            # pass runs.
            _interior_transverse(I, decomp, d) || continue
            m = min(mine, partner)
            m < mine || continue
            # The end face and the face beside the node: the end face is the
            # node's high face at a patch's high end (shift 0) and its low face
            # at a low end (shift −1).
            Ib = _shift_index(Ie, d, s == 0 ? -1 : 1)
            lim.theta[Ie] = m
            lim.theta[Ib] = m
            due[pl.patch] = true
            # Faces this patch counts that the exchange limits: at a high end
            # both, at a low end the face beside the node.
            mine == 1 && (lim.counts[tally] += s == 0 ? 2 : 1)
        end
    end
    return nothing
end

# The exchange above on device storage. Each record's planes of θ cross to the
# host as `_pack_fields!` stages a block: the end faces for the message and
# the comparison, the faces beside the node, and on a local pairing the
# partner's end faces, read after the records before it have written them, as
# the host loop reads them. The minima are taken on the host in the same order
# and both planes are written back where a face changed.
function _exchange_interface_theta_staged!(solver, lims, d::Int, due, records, comm,
                                            tally::Int=2)
    patches = getfield(solver, :patches)
    me = MPI.Comm_rank(comm)
    normal(pl) = any(lims[pl.patch].interface[d])
    end_planes(pl) = _shift_ranges(pl.mine, d,
                                   _end_shift(patches[pl.patch].decomp, pl.mine, d))
    reqs = MPI.Request[]
    for pl in records
        normal(pl) && pl.partner != me || continue
        push!(reqs, MPI.Irecv!(pl.buf, comm; source=pl.partner, tag=pl.tag))
    end
    for pl in records
        normal(pl) && pl.partner != me || continue
        _pack_fields!(pl.sbuf, (lims[pl.patch].theta,), end_planes(pl))
        push!(reqs, MPI.Isend(pl.sbuf, comm; dest=pl.partner, tag=pl.sendtag))
    end
    MPI.Waitall(reqs)
    for pl in records
        normal(pl) || continue
        lim = lims[pl.patch]
        T = eltype(lim.theta)
        s = _end_shift(patches[pl.patch].decomp, pl.mine, d)
        re = _shift_ranges(pl.mine, d, s)
        rb = _shift_ranges(re, d, s == 0 ? -1 : 1)
        m = prod(length.(re))
        own = Vector{T}(undef, m)
        beside = Vector{T}(undef, m)
        _pack_fields!(own, (lim.theta,), re)
        _pack_fields!(beside, (lim.theta,), rb)
        partner = if pl.partner == me
            so = _end_shift(patches[pl.partner_patch].decomp, pl.theirs, d)
            v = Vector{T}(undef, m)
            _pack_fields!(v, (lims[pl.partner_patch].theta,), _shift_ranges(pl.theirs, d, so))
            v
        else
            pl.buf
        end
        decomp = patches[pl.patch].decomp
        changed = false
        for (idx, I) in enumerate(CartesianIndices(pl.mine))
            _interior_transverse(I, decomp, d) || continue
            mine = own[idx]
            v = min(mine, partner[idx])
            v < mine || continue
            own[idx] = v
            beside[idx] = v
            changed = true
            due[pl.patch] = true
            mine == 1 && (lim.counts[tally] += s == 0 ? 2 : 1)
        end
        if changed
            _unpack_fields!((lim.theta,), own, re)
            _unpack_fields!((lim.theta,), beside, rb)
        end
    end
    return nothing
end

# Padded ranges `r` moved by `s` along d.
_shift_ranges(r::NTuple{3,UnitRange{Int}}, d::Int, s::Int) =
    ntuple(e -> e == d ? (first(r[e])+s:last(r[e])+s) : r[e], 3)

_interior_transverse(I, decomp, d) =
    all(e -> e == d || decomp.n_halo_d[e] < I[e] <= decomp.n_halo_d[e] + decomp.n_local[e],
        1:3)

# The shift from a shared plane node to the slot of the patch's end face: 0 at
# the patch's high end, where face f sits at node f, and −1 at its low end.
_end_shift(decomp, region, d) = first(region[d]) == decomp.n_halo_d[d] + 1 ? -1 : 0
_shift_index(I::CartesianIndex{3}, d, s) =
    CartesianIndex(ntuple(e -> e == d ? I[e] + s : I[e], 3))

_limiter_layout(solver) =
    (solver.equations.n_species, solver.equations.i_mom...,
     solver.equations.i_energy, solver.equations.n_cons)

# The face values of direction d into `lim.faces`: the local running sums of
# each line, the line totals gathered over the direction's sub-communicator,
# and the offsets that make them one running sum along the global line,
# anchored at the global low face. `mode` 1 is the stage register, scaled by
# `scale` = B_k and anchored at the face register; `mode` 2 a filter pass of
# weight `wf`, its constant removed through the filter's face relation. A
# patch interface anchors a line here as a wall does.
function _line_faces!(lim::PositivityLimiter, solver, Q, d::Int, mode::Int, scale,
                      wf, cmask=nothing, drop=nothing)
    decomp = solver.decomp
    o1, o2, o3 = decomp.n_halo_d
    n = decomp.n_local[d]
    nA, nB = _transverse(decomp.n_local, d)
    # A stage takes every register of d, the radial pressure's included; a
    # filter pass the conserved components.
    n_comp = mode == 1 ? lim.n_comp[d] : solver.equations.n_cons
    volume = mode == 2 && lim.radial[d]
    # A line this rank holds whole takes its offsets from its own anchor or
    # measurement, and the scan finishes its face values.
    whole = decomp.dims[d] == 1
    whole && mode == 2 && lim.measure[d] == 0 && _no_measure_error(d)
    pointwise!(_limiter_scan_point!, solver.tmp_a, 1, nA, nB,
               lim.faces, lim.register_fields[d], _LinearState(Q), lim.base, lim.weights[d],
               solver.h[d], lim.totals[d], lim.aux[d], lim.anchor[d], n_comp, n, d, mode,
               mode == 2 ? lim.measure[d] : 0, lim.filter_lhs, lim.filter_psi, wf,
               volume, lim.volume, solver.inv_J, cmask, drop, lim.dropped[d],
               lim.dropped_mid[d], (whole, lim.periodic[d], scale), o1, o2, o3)
    whole && return nothing
    _line_offsets!(lim, decomp, d, mode, cmask === nothing ? zero(wf) : wf)
    last_rank = decomp.coords[d] == decomp.dims[d] - 1
    pointwise!(_limiter_offset_point!, solver.tmp_a, n + 1, nA, nB,
               lim.faces, lim.offset[d], lim.wrap[d],
               mode == 1 ? lim.zero_plane[d] : lim.cstar[d], scale,
               lim.periodic[d] && last_rank, n, d, n_comp, volume, lim.volume,
               o1, o2, o3)
    return nothing
end

# The offsets of this rank's running sums along a decomposed line. One
# `Allgather` of the line totals (and of the low face's anchor or the filter
# relation's measurement) over the direction's sub-communicator, entered by
# every rank of it; the prefix sums are formed in rank order on every rank, so
# the face two ranks share carries the same value on both.
function _line_offsets!(lim::PositivityLimiter, decomp, d::Int, mode::Int, wd=0)
    tot = lim.totals[d]
    # The stage's anchor is the face register at the global low face.
    aux = mode == 1 ? lim.anchor[d] : lim.aux[d]
    L = length(tot)
    send = lim.send[d]
    # Device storage crosses to the host message here, and the offsets formed
    # on the host are uploaded after: one copy of each plane per pass.
    copyto!(send, 1, vec(tot), 1, L)
    copyto!(send, L + 1, vec(aux), 1, L)
    if !iszero(wd)
        copyto!(send, 2L + 1, vec(lim.dropped[d]), 1, L)
        copyto!(send, 3L + 1, vec(lim.dropped_mid[d]), 1, L)
    end
    _device_path(tot) ? _staged_offsets!(lim, decomp, d, mode, wd) :
                        _form_offsets!(lim.offset[d], lim.wrap[d], lim.cstar[d], lim, decomp,
                                       d, mode, wd)
    return nothing
end

# The offsets formed in the host copies of the planes and uploaded.
function _staged_offsets!(lim::PositivityLimiter, decomp, d::Int, mode::Int, wd)
    planes = _staged_planes!(lim, d)
    _form_offsets!(planes..., lim, decomp, d, mode, wd)
    copyto!(lim.offset[d], planes[1])
    copyto!(lim.wrap[d], planes[2])
    copyto!(lim.cstar[d], planes[3])
    return nothing
end

# The host copies of the offset planes of d, allocated at the first device pass.
function _staged_planes!(lim::PositivityLimiter{T}, d::Int) where {T}
    if isempty(lim.staged)
        for e in 1:3
            dims = size(lim.offset[e])
            push!(lim.staged, (zeros(T, dims), zeros(T, dims), zeros(T, dims)))
        end
    end
    return lim.staged[d]
end

# The Allgather of the message `send` holds, and the offsets of this rank's
# running sums, the global low face's value and the filter's constant formed
# from it into `off`, `wrap` and `cstar`, host arrays. A pass that dropped a
# child's residual (`wd`, its weight, nonzero) also gathers each rank's sums of
# the dropped residual, and the constant takes the parts between the middle
# face and the measuring rank's face that the other ranks hold.
function _form_offsets!(off, wrap, cstar, lim::PositivityLimiter, decomp, d::Int,
                        mode::Int, wd=0)
    L = length(off)
    P = decomp.dims[d]
    send, recv = lim.send[d], lim.recv[d]
    send[end] = mode == 2 && lim.measure[d] > 0 ? 1 : 0
    MPI.Allgather!(send, MPI.UBuffer(recv, length(send)), decomp.sub[d])
    stride = length(send)
    me = decomp.coords[d]
    # The lowest rank holding a face where the filter relation is interior.
    v = -1
    if mode == 2
        for r in 0:P-1
            recv[r*stride+stride] > 0 && (v = r; break)
        end
        v < 0 && _no_measure_error(d)
    end
    for e in 1:L
        O = mode == 1 ? recv[L+e] : zero(eltype(recv))
        wrap[e] = O
        Ov = O
        for r in 0:P-1
            r == me && (off[e] = O)
            r == v && (Ov = O)
            O += recv[r*stride+e]
        end
        cstar[e] = mode == 2 ? Ov + recv[v*stride+L+e] : zero(eltype(recv))
    end
    if mode == 2 && !iszero(wd)
        lhs = Array(lim.filter_lhs)
        sum_l = one(eltype(recv))
        for s in eachindex(lhs)
            sum_l += 2 * lhs[s]
        end
        for e in 1:L
            χ = zero(eltype(recv))
            for r in 0:P-1
                r < v && (χ += recv[r*stride+2L+e])
                r != v && (χ -= recv[r*stride+3L+e])
            end
            cstar[e] -= wd * χ / sum_l
        end
    end
    return nothing
end

@noinline _no_measure_error(d) =
    error("positivity limiter: no rank holds an interior face of dimension $d " *
          "for the filter's face relation")

# Whether this rank holds the global low and high ends of a closed line of d;
# a fold is not a closed end, and its face is limited.
_limiter_closed(lim, decomp, d) =
    (!lim.periodic[d] && !lim.fold_lo[d] && decomp.offset[d] == 0,
     !lim.periodic[d] && decomp.offset[d] + decomp.n_local[d] == decomp.n_global[d])

# Whether this rank holds the fold at the global low end of d.
_limiter_fold(lim, decomp, d) = lim.fold_lo[d] && decomp.offset[d] == 0

# The ends of d a θ pass and a correction take: the closed ends and the fold,
# as above, and which closed end is a patch interface.
function _limiter_ends(lim, decomp, d)
    closed_lo, closed_hi = _limiter_closed(lim, decomp, d)
    return (closed_lo, closed_hi, _limiter_fold(lim, decomp, d),
            closed_lo && lim.interface[d][1], closed_hi && lim.interface[d][2])
end

# The ends of d a filter pass's θ pass and correction take: as above, the last
# two flags marking a closed end at a same-level interface whose filter row
# conserves. Its end node is then held as an interior node is, in one part
# with both its faces: the end face, at the node, takes the θ of the face
# beside it, and the exchange gives both patches the smaller of their two.
function _filter_ends(lim, decomp, d)
    closed_lo, closed_hi = _limiter_closed(lim, decomp, d)
    return (closed_lo, closed_hi, _limiter_fold(lim, decomp, d),
            closed_lo && lim.conservative[d][1], closed_hi && lim.conservative[d][2])
end

# The bound of a cell whose base state aggregates to q: the smaller of the run's
# ε and the fraction ε[3] of the cell's own ρ and ρe, at least zero, and for ρe
# at least ε[4] |E|, ε[4] = `LIMITER_ENERGY_ULPS` eps(T) (see The bound, above).
# Returns the bounds on ρ and ρe and, third, the least ρe a state may hold and
# count as holding the bound: the bound on ρe with the floor less the
# tolerance, ε[5] |E|, in place of the floor, ε[5] = (`LIMITER_ENERGY_ULPS` −
# `LIMITER_ENERGY_TOLERANCE`) eps(T). It is the bound itself wherever the floor
# does not set the bound.
@inline function _limiter_cell_bound(q, ε)
    ρ = q[1]
    e = ρ > 0 ? _limiter_internal(q) : zero(ρ)
    fraction = min(ε[2], ε[3] * max(e, zero(e)))
    E = abs(q[5])
    return (min(ε[1], ε[3] * max(ρ, zero(ρ))), max(fraction, ε[4] * E),
            max(fraction, ε[5] * E))
end

# The run's bounds, the cell fraction, the relative floor of the bound on ρe and
# the floor less the tolerance, as the per-point bodies take them.
function _limiter_bounds(lim)
    T = typeof(lim.eps_rho)
    return (lim.eps_rho, lim.eps_e, T(LIMITER_CELL_FRACTION), LIMITER_ENERGY_ULPS * eps(T),
            (LIMITER_ENERGY_ULPS - LIMITER_ENERGY_TOLERANCE) * eps(T))
end

# θ per face of direction d into `lim.theta`, a line per body, then the faces and
# the unguaranteed sides of this rank counted. Returns the number of faces
# limited that this rank owns, and whether its low face, which the rank below
# owns and counts, is limited anywhere: its node 1 takes that face's
# correction too.
function _limit_faces!(lim::PositivityLimiter, solver, Q, d::Int, τ, mode::Int)
    decomp = solver.decomp
    o1, o2, o3 = decomp.n_halo_d
    n = decomp.n_local[d]
    nA, nB = _transverse(decomp.n_local, d)
    ends = mode == 1 ? _limiter_ends(lim, decomp, d) : _filter_ends(lim, decomp, d)
    closed_hi, fold = ends[2], ends[3]
    pointwise!(_limiter_theta_point!, solver.tmp_a, 1, nA, nB,
               lim.theta, lim.line_counts[d], _linear(Q), lim.faces,
               (solver.u, solver.v, solver.w), solver.c, solver.p, lim.inv_weights,
               lim.free, decomp.active, d, n, ends, solver.h[d],
               τ, mode, _limiter_bounds(lim), _limiter_layout(solver),
               mode == 2 ? lim.radial[d] : lim.areal[d],
               mode == 2 ? lim.volume_state : (true, true, true, true, true), solver.inv_J,
               mode == 1 && lim.areal[d], lim.rates, o1, o2, o3)
    return _limiter_tally!(lim, decomp, d, closed_hi, mode, fold, ends[5])
end

# The tallies a θ pass left per line in `line_counts[d]`: the faces limited,
# the low face limited where the rank below owns it, the unguaranteed sides,
# and the faces limited at r = 0. A fold's face is this rank's own and counted
# with its faces.
function _limiter_tally!(lim, decomp, d, closed_hi, mode, fold, end_face=false)
    n = decomp.n_local[d]
    tallies = lim.line_counts[d]
    nA, nB = size(tallies, 2), size(tallies, 3)
    limited, shared, sides, axis = _device_path(tallies) ? _tally_reduced(tallies) :
                                   _tally_sum(tallies)
    faces = ((closed_hi ? n - 1 : n) + (fold ? 1 : 0) + (end_face ? 1 : 0)) * nA * nB
    lim.counts[mode == 1 ? 1 : 3] += faces
    lim.counts[5] += sides
    lim.counts[6] += axis
    return limited, shared
end

# The four tallies summed over the lines: a host loop, and on device storage a
# reduction over the lines there with one copy of its four sums to the host.
function _tally_sum(tallies)
    limited = 0
    shared = 0
    sides = 0
    axis = 0
    @inbounds for b in 1:size(tallies, 3), a in 1:size(tallies, 2)
        limited += tallies[1, a, b]
        shared += tallies[2, a, b]
        sides += tallies[3, a, b]
        axis += tallies[4, a, b]
    end
    return limited, shared, sides, axis
end

function _tally_reduced(tallies)
    s = Array(sum(tallies; dims=(2, 3)))
    return s[1], s[2], s[3], s[4]
end

# --- The stage on a radial line ------------------------------------------

# The stage limiter along a radial line d: each node's rate, θ per face, and
# the corrections, which a limited fold face also takes into the face
# register at the fold. `A` and `B` are the stage's low-storage coefficients.
function _limit_radial!(lim::PositivityLimiter, solver, Q, dQ, du, d::Int, A, B, dt, τ)
    _radial_theta!(lim, solver, Q, dQ, du, d, A, B, dt, τ, solver.tmp_b) || return nothing
    _radial_correct!(lim, solver, Q, dQ, d, B, dt, τ, solver.tmp_b)
    return nothing
end

# The rates and θ of a radial line d into `theta`, the tallies counted, and
# whether a face is limited where this rank's nodes take its correction. A
# line's end at a face shared with another tile (`_limiter_ends`) takes its
# end face's speed from its end node alone, as a closed far end does.
function _radial_theta!(lim::PositivityLimiter, solver, Q, dQ, du, d::Int, A, B, dt, τ,
                        theta)
    decomp = solver.decomp
    o1, o2, o3 = decomp.n_halo_d
    n = decomp.n_local[d]
    nA, nB = _transverse(decomp.n_local, d)
    closed_lo, closed_hi, fold, end_lo, end_hi = _limiter_ends(lim, decomp, d)
    uvw = (solver.u, solver.v, solver.w)
    lay = _limiter_layout(solver)
    bounds = _limiter_bounds(lim)
    regs = (lim.register_fields[1], lim.register_fields[2], lim.register_fields[3])
    speeds = lim.speeds
    Ql = _LinearState(Q)
    dQl = _LinearState(dQ)
    pointwise!(_limiter_radial_rate_point!, lim.rates, 1, nA, nB,
               lim.rates, speeds, Ql, dQl, _LinearState(du), regs, uvw, solver.c,
               solver.p, solver.rho, solver.inv_J, solver.inv_h[d], lim.weights,
               lim.areas[d], decomp.active, d, (fold, closed_hi, n, end_lo), (A, B, dt),
               bounds, lay, lim.free, o1, o2, o3)
    # The face two ranks share takes the same θ on both, from both cells'
    # rates, along r here and along z in the pass that follows.
    for e in 1:3
        decomp.active[e] && decomp.dims[e] > 1 && exchange_dim!(lim.rates, decomp, e)
    end
    pointwise!(_limiter_radial_theta_point!, solver.tmp_a, 1, nA, nB,
               theta, lim.line_counts[d], Ql, lim.faces, lim.rates, uvw, solver.c,
               solver.p, solver.rho, solver.inv_J, solver.inv_h[d], lim.weights[d],
               lim.areas[d], speeds, lim.free, d,
               (fold, closed_lo, closed_hi, n, end_lo, end_hi), τ, bounds, lay, o1, o2, o3)
    limited, shared = _limiter_tally!(lim, decomp, d, closed_hi, 1, fold, end_hi)
    lim.counts[2] += limited
    return limited + shared > 0
end

# The corrections of a radial line d from the θ in `theta`.
function _radial_correct!(lim::PositivityLimiter, solver, Q, dQ, d::Int, B, dt, τ, theta)
    decomp = solver.decomp
    o1, o2, o3 = decomp.n_halo_d
    n = decomp.n_local[d]
    nA, nB = _transverse(decomp.n_local, d)
    _, _, fold, end_lo, end_hi = _limiter_ends(lim, decomp, d)
    pointwise!(_limiter_radial_correct_point!, solver.tmp_a, n, nA, nB,
               _LinearState(dQ), lim.register_fields[d], lim.faces, theta, _LinearState(Q),
               (solver.u, solver.v, solver.w), solver.p, solver.inv_J, solver.inv_h[d],
               lim.weights[d], lim.areas[d], lim.speeds, (lim.anchor[d], lim.anchor_hi[d]),
               d, τ, (one(B) / B, one(B) / (B * dt)), _limiter_layout(solver),
               (fold, end_lo, end_hi, n), lim.free, o1, o2, o3)
    return nothing
end

@inline _limiter_gamma(ρ, c, p) = p > 0 ? max(ρ * c * c / p, one(p)) : one(p)

# The Lax–Friedrichs speed at which Q ∓ F̃(Q)/a is admissible for the flux
# F̃ = G + ζ p e_r of a cell whose pressure fraction at the face is ζ:
# (1 − λv)(1 − λv(1 + (γ − 1)(1 − ζ))) > λ² ζ² c² (γ − 1)/(2γ), λ = 1/a,
# v = ±u, holds for a ≥ max(1, |1 + (γ − 1)(1 − ζ)|)|u| + ζ c √((γ − 1)/(2γ)),
# and the second term is kept at least c so that ζ = 1 gives |u| + c. The
# node's `_radial_node` holds s = √((γ − 1)/(2γ)).
@inline function _radial_speed(x, ζ)
    k = abs(1 + (x.γ - 1) * (1 - ζ))
    return max(one(k), k) * abs(x.u) + max(one(k), ζ * x.s) * x.c
end

# What a node contributes to the speeds of its two faces.
Base.@propagate_inbounds function _radial_node(ud, c, p, ρ, inv_J, inv_h, L)
    γ = _limiter_gamma(ρ[L], c[L], p[L])
    return (u=ud[L], c=c[L], γ=γ, s=sqrt((γ - 1) / (2γ)), ih=inv_h[L], iJ=inv_J[L])
end

# The speed at a face of area Ā between nodes l and r (`_radial_node`): the
# largest over both states and both cells' ζ = (inv_h / inv_J) / Ā, since each
# cell's half state holds both states. With r = l, the speed of one state at a
# closed end.
@inline function _radial_face_speed(l, r, Ā)
    ζL = l.ih / (l.iJ * Ā)
    ζR = r.ih / (r.iJ * Ā)
    return max(_radial_speed(l, ζL), _radial_speed(l, ζR), _radial_speed(r, ζL),
               _radial_speed(r, ζR))
end

# The area-weighted flux of a state per unit area, G = (ρu, m u, (E + p)u)
# along d, without the pressure, aggregated.
@inline _radial_flux(q, u, p) = (q[1] * u, q[2] * u, q[3] * u, q[4] * u, (q[5] + p) * u)

# The node change per unit time of an area face value G and a pressure face
# value Π, as a cell with inv_J/W = sJ and inv_h/W = sh takes them.
@inline _radial_vector(G, Π, sJ, sh, d) =
    (sJ * G[1], sJ * G[2] + (d == 1) * sh * Π, sJ * G[3] + (d == 2) * sh * Π,
     sJ * G[4] + (d == 3) * sh * Π, sJ * G[5])

# The node terms' part of the stage increment at I, aggregated: B (A du + dt dQ)
# less the directions' face parts −B inv_J r_e (and −B inv_h r_p for the
# radial pressure of d), which leaves B times the node terms' history.
Base.@propagate_inbounds function _limiter_node_increment(Q, dQ, du, regs, inv_J, inv_h,
                                                          act, d, A, B, dt, lay, I)
    ns, m1, m2, m3, ie, nc = lay
    md = d == 1 ? m1 : d == 2 ? m2 : m3
    T = typeof(B)
    out = (zero(T), zero(T), zero(T), zero(T), zero(T))
    # The directions' register sets differ in length (the radial pressure's
    # is one longer), so the tuple is read at literal positions: a runtime
    # index into it does not compile for a device.
    for cc in 1:nc
        x = iszero(A) ? dt * dQ[I, cc] : A * du[I, cc] + dt * dQ[I, cc]
        act[1] && (x += inv_J[I] * regs[1][cc][I])
        act[2] && (x += inv_J[I] * regs[2][cc][I])
        act[3] && (x += inv_J[I] * regs[3][cc][I])
        if cc == md
            r = d == 1 ? regs[1][nc+1][I] : d == 2 ? regs[2][nc+1][I] : regs[3][nc+1][I]
            x += inv_h[I] * r
        end
        x *= B
        slot = cc <= ns ? 1 : cc == m1 ? 2 : cc == m2 ? 3 : cc == m3 ? 4 : cc == ie ? 5 : 0
        slot > 0 && (out = Base.setindex(out, out[slot] + x, slot))
    end
    return out
end

# The largest s ≥ 0 for which q + s δ keeps ρ ≥ ε[1] and ρe ≥ ε[3], the
# threshold of `_limiter_admissible`: the smaller of the density's root and the
# first positive root of 2ρ(s)(E(s) − ε[3]) − |m(s)|², a quadratic in s;
# Inf when neither bound is reached. The admissible set is convex, so [0, s]
# is admissible.
@inline function _limiter_reach(q, δ, ε)
    T = typeof(q[1])
    s = T(Inf)
    δ[1] < 0 && (s = min(s, (q[1] - ε[1]) / -δ[1]))
    a0 = 2 * q[1] * (q[5] - ε[3]) - (q[2] * q[2] + q[3] * q[3] + q[4] * q[4])
    a0 > 0 || return zero(T)
    a1 = 2 * (q[1] * δ[5] + δ[1] * (q[5] - ε[3])) -
         2 * (q[2] * δ[2] + q[3] * δ[3] + q[4] * δ[4])
    a2 = 2 * δ[1] * δ[5] - (δ[2] * δ[2] + δ[3] * δ[3] + δ[4] * δ[4])
    disc = a1 * a1 - 4 * a2 * a0
    if iszero(a2)
        a1 < 0 && (s = min(s, a0 / -a1))
    elseif disc >= 0
        # The two roots, formed without cancellation.
        w = -(a1 + copysign(sqrt(disc), a1)) / 2
        r1 = w / a2
        r2 = iszero(w) ? T(Inf) : a0 / w
        r1 > 0 && (s = min(s, r1))
        r2 > 0 && (s = min(s, r2))
    end
    return max(s, zero(T))
end

# The speed of each face of a radial line d into `speeds`, the face f at node
# f, and the rate of each cell over the share its face and geometric parts
# keep into `rates`: Σ R / (1 − β_n), with R the per-volume rates
# Ā a inv_J/W of its two faces (`AXIS_SHARE` of the outer face's for the face
# at the fold), γ u⁺ ΔĀ inv_J/W of its geometric part and 2(|u_e| + c)/W_e of
# every other active direction, and β_n the node part's share; Inf where the
# node part alone needs the whole cell. The face at a closed far end takes its
# one state, and so does the end face at a low end shared with another tile;
# the fold's face takes none. The θ and correction passes read the speeds, and
# the sweep carries a node's part of them to its next face.
@inline function _limiter_radial_rate_point!(rates, speeds, Q, dQ, du, regs, uvw, c, p, ρ,
                                             inv_J, inv_h, W, areas, act, d, ends, coeffs,
                                             bounds, lay, free, o1, o2, o3, _, a, b)
    @inbounds begin
        T = eltype(rates)
        fold, closed_hi, n, end_lo = ends
        A, B, dt = coeffs
        o = d == 1 ? o1 : d == 2 ? o2 : o3
        n1, n2 = size(rates, 1), size(rates, 2)
        sd = _line_stride(d, n1, n2)
        L0 = _line_linear(d, 0, a, b, o1, o2, o3, n1, n2)
        ud = uvw[d]
        lo = _radial_node(ud, c, p, ρ, inv_J, inv_h, L0)
        lo_bad = _limiter_unheld_bad(Q, free, L0, _line_node(d, 0, a, b, o1, o2, o3), lay)
        a_lo = zero(T)
        for pp in 0:n
            L = L0 + pp * sd
            hi = _radial_node(ud, c, p, ρ, inv_J, inv_h, L + sd)
            hi_bad = _limiter_unheld_bad(Q, free, L + sd,
                                         _line_node(d, pp + 1, a, b, o1, o2, o3), lay)
            Āp = areas[pp+o]
            # A node the limiter does not hold whose state is not admissible
            # leaves the face the speed of the other.
            a_hi = fold && pp == 0 ? zero(T) :
                   closed_hi && pp == n ? _radial_face_speed(lo, lo, Āp) :
                   end_lo && pp == 0 ? _radial_face_speed(hi, hi, Āp) :
                   lo_bad ? _radial_face_speed(hi, hi, Āp) :
                   hi_bad ? _radial_face_speed(lo, lo, Āp) :
                   _radial_face_speed(lo, hi, Āp)
            speeds[L] = a_hi
            if pp >= 1
                # The cell at node pp, between faces pp − ½ and pp + ½.
                I = _line_node(d, pp, a, b, o1, o2, o3)
                Ām = areas[pp-1+o]
                Rp = Āp * a_hi
                Rm = fold && pp == 1 ? T(AXIS_SHARE) * Rp : Ām * a_lo
                R = inv_J[L] / W[d][pp+o] *
                    (Rp + Rm + lo.γ * max(ud[L], zero(T)) * (Āp - Ām))
                for e in 1:3
                    (e == d || !act[e]) && continue
                    R += 2 * (abs(uvw[e][L]) + c[L]) / W[e][I[e]]
                end
                q = _limiter_state(Q, L, lay)
                δ = _limiter_node_increment(Q, dQ, du, regs, inv_J, inv_h, act, d, A, B,
                                            dt, lay, L)
                βn = one(T) / _limiter_reach(q, δ, _limiter_cell_bound(q, bounds))
                rates[L] = βn < 1 ? R / (1 - βn) : T(Inf)
            end
            lo = hi
            lo_bad = hi_bad
            a_lo = a_hi
        end
    end
    return nothing
end

# Whether the node at linear index L (Cartesian I) is one the limiter does not
# hold and its state is not admissible: a shell node interpolated across a
# shock, or an exempt parent node.
Base.@propagate_inbounds _limiter_unheld_bad(Q, free, L, I, lay) =
    !_limiter_constrained(free, I) && !_limiter_positive(_limiter_state(Q, L, lay))

# θ at each face of a radial line d, the face f in place of node f, and the
# line's tallies into `counts` (see `_limiter_tally!`), as the Cartesian θ
# pass takes them at a stage. A cell's part for the face is
# q ∓ (1/β)[H − τ (Ā G(q), p) as a node change], 1/β its rate over the face's,
# and its first-order counterpart the Lax–Friedrichs half state; the cell at
# the fold takes the fold face with the axis share, whose first-order values
# are 0 and τ p_1, and the cell on a closed far end takes its boundary face,
# never limited, in one part with its interior face, as on a Cartesian line.
# The end cell at a face shared with another tile takes its end face in one
# part with its interior face too, the end face blended toward τ times the
# node's point flux (`_radial_end_flux`), and the end face takes the θ of the
# face beside it. A side is guaranteed when τ times its cell's rate is below
# one; a cell whose node part needs the whole cell takes the first-order flux.
@inline function _limiter_radial_theta_point!(theta, counts, Q, faces, rates, uvw, c, p,
                                              ρ, inv_J, inv_h, W, areas, speeds, free, d,
                                              ends, τ, bounds, lay, o1, o2, o3, _, a, b)
    @inbounds begin
        T = eltype(theta)
        fold, closed_lo, closed_hi, n, end_lo, end_hi = ends
        o = d == 1 ? o1 : d == 2 ? o2 : o3
        n1, n2 = size(theta, 1), size(theta, 2)
        sd = _line_stride(d, n1, n2)
        L0 = _line_linear(d, 0, a, b, o1, o2, o3, n1, n2)
        ud = uvw[d]
        np = lay[6] + 1
        limited = 0
        shared = 0
        sides = 0
        axis = 0
        lo = _radial_cell(Q, free, inv_J, inv_h, W, bounds, lay, L0,
                          _line_node(d, 0, a, b, o1, o2, o3), o)
        for f in 0:n
            Ll = L0 + f * sd
            Lr = Ll + sd
            hi = _radial_cell(Q, free, inv_J, inv_h, W, bounds, lay, Lr,
                              _line_node(d, f + 1, a, b, o1, o2, o3), f + 1 + o)
            flag = 0
            θ = one(T)
            if !((closed_lo && f == 0) || (closed_hi && f == n))
                fold_face = fold && f == 0
                wall_r = closed_hi && f == n - 1
                wall_l = end_lo && f == 1
                Ā = areas[f+o]
                G = _limiter_face_state(faces, Ll, lay)
                Π = faces[np][Ll]
                cl = !fold_face && lo.constrained
                cr = hi.constrained
                a_face = fold_face ? zero(T) : speeds[Ll]
                ql, εl, sJl, shl = lo.q, lo.ε, lo.sJ, lo.sh
                qr, εr, sJr, shr = hi.q, hi.ε, hi.sJ, hi.sh
                # The low cell's part for its outer face, and for the end cell at
                # a shared low end its part for both faces, q + k (H_0 − H_1).
                if wall_l
                    Ā0 = areas[f-1+o]
                    γ = _limiter_gamma(ρ[Ll], c[Ll], p[Ll])
                    kl = rates[Ll] / (sJl * (Ā * a_face + Ā0 * speeds[Ll-sd] +
                                             γ * max(ud[Ll], zero(T)) * (Ā - Ā0)))
                    bl = _limiter_axpy(ql, kl, _radial_vector(
                                       _limiter_face_state(faces, Ll - sd, lay),
                                       faces[np][Ll-sd], sJl, shl, d))
                else
                    kl = rates[Ll] / (sJl * Ā * a_face)
                    bl = _limiter_axpy(ql, kl * τ,
                                       _radial_vector(map(x -> Ā * x,
                                                          _radial_flux(ql, ud[Ll], p[Ll])),
                                                      p[Ll], sJl, shl, d))
                end
                Hl = _radial_vector(G, Π, sJl, shl, d)
                # The high cell's part for its inner face.
                if fold_face
                    Ā1 = areas[1+o]
                    kr = rates[Lr] / (sJr * T(AXIS_SHARE) * Ā1 * speeds[Lr])
                    br = _limiter_axpy(qr, -kr * τ, _radial_vector(
                                       (zero(T), zero(T), zero(T), zero(T), zero(T)),
                                       p[Lr], sJr, shr, d))
                elseif wall_r
                    Āw = areas[f+1+o]
                    aw = speeds[Lr]
                    γ = _limiter_gamma(ρ[Lr], c[Lr], p[Lr])
                    kr = rates[Lr] / (sJr * (Ā * a_face + Āw * aw +
                                             γ * max(ud[Lr], zero(T)) * (Āw - Ā)))
                    br = _limiter_axpy(qr, -kr, _radial_vector(
                                       _limiter_face_state(faces, Lr, lay), faces[np][Lr],
                                       sJr, shr, d))
                else
                    kr = rates[Lr] / (sJr * Ā * a_face)
                    br = _limiter_axpy(qr, -kr * τ,
                                       _radial_vector(map(x -> Ā * x,
                                                          _radial_flux(qr, ud[Lr], p[Lr])),
                                                      p[Lr], sJr, shr, d))
                end
                Hr = _radial_vector(G, Π, sJr, shr, d)
                okl = isfinite(kl)
                okr = isfinite(kr)
                need_l = cl && !(okl && _limiter_admissible(_limiter_axpy(bl, -kl, Hl), εl))
                need = need_l ||
                       (cr && !(okr && _limiter_admissible(_limiter_axpy(br, kr, Hr), εr)))
                if need
                    # First order: τ Ā times the Rusanov flux, and τ {p}; at the
                    # fold 0 and τ p_1.
                    if fold_face
                        GL = (zero(T), zero(T), zero(T), zero(T), zero(T))
                        ΠL = τ * p[Lr]
                    else
                        # A node the limiter does not hold whose state is not
                        # admissible takes the held cell's state.
                        own_l = !lo.constrained && !_limiter_positive(ql)
                        own_r = !hi.constrained && !_limiter_positive(qr)
                        qa, ua, pa = own_l ? (qr, ud[Lr], p[Lr]) : (ql, ud[Ll], p[Ll])
                        qb, ub, pb = own_r ? (ql, ud[Ll], p[Ll]) : (qr, ud[Lr], p[Lr])
                        Fl = _radial_flux(qa, ua, pa)
                        Fr = _radial_flux(qb, ub, pb)
                        GL = map((x, y, u, v) ->
                                     τ * Ā * ((x + y) / 2 - a_face * (v - u) / 2),
                                 Fl, Fr, qa, qb)
                        ΠL = τ * (pa + pb) / 2
                    end
                    if cl
                        GLl, ΠLl = wall_l ?
                                   _radial_end_flux(GL, ΠL,
                                                    _limiter_face_state(faces, Ll - sd, lay),
                                                    faces[np][Ll-sd], ql, ud[Ll], p[Ll],
                                                    areas[f-1+o], τ) : (GL, ΠL)
                        t, ok = okl ?
                                _limiter_side(bl, Hl, _radial_vector(GLl, ΠLl, sJl, shl, d),
                                              -kl, εl) : (zero(T), false)
                        θ = min(θ, t)
                        (ok && τ * rates[Ll] < 1) || (flag |= 1)
                    end
                    if cr
                        GLr, ΠLr = wall_r && end_hi ?
                                   _radial_end_flux(GL, ΠL,
                                                    _limiter_face_state(faces, Lr, lay),
                                                    faces[np][Lr], qr, ud[Lr], p[Lr],
                                                    areas[f+1+o], τ) : (GL, ΠL)
                        t, ok = okr ?
                                _limiter_side(br, Hr, _radial_vector(GLr, ΠLr, sJr, shr, d),
                                              kr, εr) : (zero(T), false)
                        θ = min(θ, t)
                        (ok && τ * rates[Lr] < 1) || (flag |= 2)
                    end
                end
            end
            theta[Ll] = θ
            if θ < 1
                (f >= 1 || fold) ? (limited += 1) : (shared += 1)
                fold && f == 0 && (axis += 1)
            end
            f >= 1 && (flag & 1) != 0 && (sides += 1)
            f <= n - 1 && (flag & 2) != 0 && (sides += 1)
            lo = hi
        end
        # An end face at a shared node takes the θ of the face beside it; the
        # tile whose high end it is counts it.
        if end_hi
            θe = theta[L0+(n-1)*sd]
            theta[L0+n*sd] = θe
            θe < 1 && (limited += 1)
        end
        if end_lo
            θe = theta[L0+sd]
            theta[L0] = θe
            θe < 1 && (shared += 1)
        end
        counts[1, a, b] = limited
        counts[2, a, b] = shared
        counts[3, a, b] = sides
        counts[4, a, b] = axis
    end
    return nothing
end

# The first-order face values toward which the end cell at a shared node
# blends, as the face beside it sees them: that face's Rusanov flux GL and
# pressure ΠL, plus the end face's own values Ge and Πe, which the cell's part
# has already taken out, less τ times the node's point flux on the end face's
# area Āe. At θ = 0 the end face then carries the point flux.
@inline function _radial_end_flux(GL, ΠL, Ge, Πe, q, u, p, Āe, τ)
    F = _radial_flux(q, u, p)
    return (map((x, g, f) -> x + g - τ * Āe * f, GL, Ge, F), ΠL + Πe - τ * p)
end

# A radial cell's state and bound, its inverse Jacobian and inverse spacing
# over its weight W[pw], and whether no face condition overwrites it: what a
# θ pass reads of it at both its faces.
Base.@propagate_inbounds function _radial_cell(Q, free, inv_J, inv_h, W, bounds, lay, L,
                                               I, pw)
    q = _limiter_state(Q, L, lay)
    return (q=q, ε=_limiter_cell_bound(q, bounds), sJ=inv_J[L] / W[pw],
            sh=inv_h[L] / W[pw], constrained=_limiter_constrained(free, I))
end

# The corrections at node pp of a radial line from its two faces, where either
# is limited: δ = (θ − 1)(Φ − Φ_L) per face for each conserved component and
# for the pressure, the node changing by inv_J (δ_{p−½} − δ_{p+½})/W_p and,
# on the radial momentum, by inv_h times the pressure's. Each enters dQ / (B dt)
# and leaves its register, and a limited fold face enters the face register at
# the fold, whose running sum the next stage starts from. An end face at a
# node shared with another tile takes the node's point flux as its first-order
# value and enters the face register at its end, `anchors[1]` at the low end
# and `anchors[2]` at the high end, as on a Cartesian line.
@inline function _limiter_radial_correct_point!(dQ, regs, faces, theta, Q, uvw, p,
                                                inv_J, inv_h, W, areas, speeds, anchors,
                                                d, τ, scales, lay, ends, free, o1, o2, o3,
                                                pp, a, b)
    @inbounds begin
        T = eltype(theta)
        n1, n2 = size(theta, 1), size(theta, 2)
        I = _line_linear(d, pp, a, b, o1, o2, o3, n1, n2)
        sd = _line_stride(d, n1, n2)
        Im = I - sd
        Ip = I + sd
        θm = theta[Im]
        θp = theta[I]
        (θm < 1 || θp < 1) || return nothing
        o = d == 1 ? o1 : d == 2 ? o2 : o3
        ns, m1, m2, m3, ie, nc = lay
        md = d == 1 ? m1 : d == 2 ? m2 : m3
        inv_B, inv_Bdt = scales
        fold, end_lo, end_hi, n = ends
        anchor, anchor_hi = anchors
        fold_face = fold && pp == 1
        end_l = end_lo && pp == 1
        end_r = end_hi && pp == n
        ud = uvw[d]
        Ām = areas[pp-1+o]
        Āp = areas[pp+o]
        am = θm < 1 && !fold_face ? speeds[Im] : zero(T)
        ap = θp < 1 ? speeds[I] : zero(T)
        # The nodes replaced in the first-order flux, as the face pass replaced
        # them: a neighbour the limiter does not hold whose state is not
        # admissible by this node, and this node, if it is one, by the
        # neighbour, so both nodes of a face take one flux (see
        # `_limiter_correct_point!`). An end face's two nodes are this one.
        bad_m = !fold_face && _limiter_unheld_bad(Q, free, Im,
                                                  _line_node(d, pp - 1, a, b, o1, o2, o3),
                                                  lay)
        bad_p = _limiter_unheld_bad(Q, free, Ip, _line_node(d, pp + 1, a, b, o1, o2, o3),
                                    lay)
        Jm = bad_m | end_l ? I : Im
        Jp = bad_p | end_r ? I : Ip
        own = _limiter_unheld_bad(Q, free, I, _line_node(d, pp, a, b, o1, o2, o3), lay)
        Km = own & !end_l ? Im : I
        Kp = own & !end_r ? Ip : I
        inv_W = one(T) / W[pp+o]
        for cc in 0:nc
            # cc = 0 is the pressure, kept in register nc + 1.
            slot = cc == 0 ? nc + 1 : cc
            change = zero(T)
            if θm < 1
                gl = cc == 0 ? τ * (fold_face ? p[I] : (p[Jm] + p[Km]) / 2) :
                     fold_face ? zero(T) :
                     τ * Ām * _radial_lf_component(Q, ud, p, cc, Jm, Km, am, ie)
                δ = (θm - 1) * (faces[slot][Im] - gl)
                change += δ
                (fold_face | end_l) && (anchor[slot, a, b] += δ * inv_B)
            end
            if θp < 1
                gl = cc == 0 ? τ * (p[Kp] + p[Jp]) / 2 :
                     τ * Āp * _radial_lf_component(Q, ud, p, cc, Kp, Jp, ap, ie)
                δp = (θp - 1) * (faces[slot][I] - gl)
                change -= δp
                end_r && (anchor_hi[slot, a, b] += δp * inv_B)
            end
            change *= inv_W
            if cc == 0
                dQ[I, md] += inv_h[I] * change * inv_Bdt
            else
                dQ[I, cc] += inv_J[I] * change * inv_Bdt
            end
            regs[slot][I] -= change * inv_B
        end
    end
    return nothing
end

# Component `cc` of the Rusanov flux per unit area between L and R at speed a,
# the area-weighted flux without the pressure.
Base.@propagate_inbounds function _radial_lf_component(Q, ud, p, cc, L, R, a, ie)
    uL, uR = ud[L], ud[R]
    qL, qR = Q[L, cc], Q[R, cc]
    fL = qL * uL
    fR = qR * uR
    if cc == ie
        fL += p[L] * uL
        fR += p[R] * uR
    end
    return (fL + fR) / 2 - a * (qR - qL) / 2
end

# --- The limited filter pass ---------------------------------------------

# `filter_state!` under the weighting `:none` with the filter limiter after
# each directional pass; with no face limited the same passes and the same
# arithmetic. A fold's face takes its limit from node 1 alone: the filter's
# face flux there vanishes for an even field and not for an odd one.
_limited_filter_state!(solver, Q::ConservedState) =
    _limited_filter_pass!(getfield(solver, :positivity), solver, Q)

# The multi-patch `filter_state!`, each patch's passes limited. The filter's
# rows at a node two same-level patches share conserve, and the averaging that
# follows takes the mean of the node's two limited updates
# (`_limited_level_filter!`); the shell imposition overwrites a refined
# patch's node at a coarse-fine face, where the filter's row is the identity.
# A subcycled refined level filters inside `_advance_level!`
# (`_limited_level_filter!`).
function _limited_filter_state!(solver, states::Vector{<:ConservedState})
    lims = _follow_patches!(solver, getfield(solver, :positivity))
    subcycle = getfield(solver, :subcycle)
    for lev in getfield(solver, :levels)
        subcycle && lev.index > 0 && continue
        _limited_level_filter!(lims, solver, lev, states)
    end
    _ledger!(solver, states, :filter)
    sync_patches!(solver, states)
    _ledger!(solver, states, :same_level)
    return states
end

# `_level_filter!` with each patch's passes limited, and `_level_filter!`
# itself where the step is not limited. Where the conservative rows close a
# same-level interface the passes run a direction at a time over the level,
# as the stage limiter's do (`_limit_level_stage!`): each patch's θ, the
# exchange that gives the faces about a shared node one θ on both patches,
# then the corrections.
_limited_level_filter!(::Nothing, solver, lev, states) =
    _level_filter!(solver, lev, states)
function _limited_level_filter!(lims::Vector{Any}, solver, lev, states)
    patches = getfield(solver, :patches)
    if !any(i -> any(any, lims[i].conservative), lev.patches)
        for i in lev.patches
            _limited_filter_pass!(lims[i], PatchSolver(solver, patches[i]), states[i])
        end
        return states
    end
    root = lev.index == 0
    for d in 1:3
        PatchSolver(solver, patches[first(lev.patches)]).decomp.active[d] || continue
        due = falses(length(patches))
        for i in lev.patches
            ps = PatchSolver(solver, patches[i])
            lim = lims[i]
            w = filter_weight(ps, d)
            due[i] = _limited_filter_theta!(lim, ps, states[i], d, w,
                                            _masked_filter(ps, states[i], false))
            # A patch without such an end shares its face values and θ with
            # the other patches of its extent and takes its corrections at once.
            if due[i] && !any(any, lim.conservative)
                _limited_filter_correct!(lim, ps, states[i], d, w)
                due[i] = false
            end
        end
        records = root ? solver.plane_pairs : lev.plane_pairs[d]
        _exchange_interface_theta!(solver, lims, d, due, records,
                                   root ? solver.comm : lev.level_comm.comm, 4)
        for i in lev.patches
            due[i] || continue
            ps = PatchSolver(solver, patches[i])
            _limited_filter_correct!(lims[i], ps, states[i], d, filter_weight(ps, d))
        end
    end
    return states
end

# A pass that drops a child's residual (`_drop_child_residual!`) keeps the
# filter's face relation only up to the dropped residual summed along the
# line, so the constant the relation measures depends on the face it is
# measured at. A line one rank holds measures it at its middle face; split
# across ranks it is measured at another face, and measured there as the
# unmasked relation the constant differed, and a limited face with it, on
# another rank count. The scan measures it relative to the middle face,
# through the dropped residual summed between the two faces
# (`_limiter_scan_point!`, `_form_offsets!`), which leaves a line one rank
# holds as it was. Returns the mask and the rows, residual stencil and local
# middle face the drop takes, or `nothing` twice where the pass drops
# nothing, a pair fold's included; a self-paired fold active on the line is
# left out as well, its base state's mirrored halos being filled only inside
# the pass.
_dropped_rows(solver, d::Int, ::Nothing) = (nothing, nothing)
function _dropped_rows(solver, d::Int, mask)
    fold = _fold_at(solver, d)
    fold === nothing || !_fold_active(fold) && fold.pair === nothing ||
        return (nothing, nothing)
    plan = fold === nothing ? _operator_plan(solver.filter_plans, d) :
                              _fold_plan(fold, 1, Val(:filter), 1, false)
    r = _residual_stencil(plan)
    M = something(findlast(m -> !iszero(r[m + 1]), 1:4), 0)
    nlo, nhi = _rows_closed(plan)
    decomp = solver.decomp
    return mask, (r, M, nlo + 1, decomp.n_local[d] - nhi,
                  decomp.n_global[d] ÷ 2 - decomp.offset[d])
end

# The passes of `filter_state!` under the weighting `:none`, the residual a
# child level covers dropped as there (`_child_mask!`), each followed by the
# filter limiter.
function _limited_filter_pass!(lim::PositivityLimiter, solver, Q)
    masked = _masked_filter(solver, Q, false)
    for d in 1:3
        solver.decomp.active[d] || continue
        w = filter_weight(solver, d)
        _limited_filter_theta!(lim, solver, Q, d, w, masked) &&
            _limited_filter_correct!(lim, solver, Q, d, w)
    end
    return Q
end

# The pass along d of weight `w`, its face values and θ, and whether a face is
# limited where this rank's nodes take its correction.
function _limited_filter_theta!(lim::PositivityLimiter, solver, Q, d::Int, w,
                                masked::Bool)
    decomp = solver.decomp
    n_cons = solver.equations.n_cons
    comps = [view(Q, :, :, :, c) for c in 1:n_cons]
    n1, n2, n3 = padded_extent(decomp)
    exchange_dim_batch!(comps, decomp, d)
    mask = _child_mask!(solver, Q, comps, d, masked)
    for c in 1:n_cons
        pointwise!(_copy_component_point!, solver.tmp_a, n1, n2, n3,
                   lim.base[c], Q, c)
    end
    for c in 1:n_cons
        _filter_line!(solver.tmp_a, comps[c], solver, d, cons_parity(solver, d, c),
                      mask)
        if w == 1
            copy_interior!(comps[c], solver.tmp_a, decomp)
        else
            blend_interior!(comps[c], solver.tmp_a, w, decomp)
        end
    end
    cmask, drop = _dropped_rows(solver, d, mask)
    _line_faces!(lim, solver, Q, d, 2, one(w), w, cmask, drop)
    # The half states are read from the state before the pass.
    limited, shared = _limit_faces!(lim, solver, _BaseState(lim.base), d,
                                    zero(w), 2)
    lim.counts[4] += limited
    return limited + shared > 0
end

# The corrections of the pass along d from its face values and θ.
function _limited_filter_correct!(lim::PositivityLimiter, solver, Q, d::Int, w)
    decomp = solver.decomp
    o1, o2, o3 = decomp.n_halo_d
    n = decomp.n_local[d]
    nA, nB = _transverse(decomp.n_local, d)
    Ql = _LinearState(Q)
    pointwise!(_limiter_correct_point!, solver.tmp_a, n, nA, nB,
               Ql, lim.register_fields[d], lim.faces, lim.theta, Ql,
               (solver.u, solver.v, solver.w), solver.c, solver.p,
               lim.weights[d], d, solver.h[d], zero(w), 2, (one(w), one(w)),
               _limiter_layout(solver),
               _filter_ends(lim, decomp, d), n,
               lim.radial[d], lim.volume, solver.inv_J, false,
               (lim.anchor[d], lim.anchor_hi[d]), lim.free,
               o1, o2, o3)
    return nothing
end

# The pre-pass state of a filter pass, held per component in the workspace,
# indexed as a `_LinearState` is.
struct _BaseState{F}
    fields::F
end
Base.@propagate_inbounds Base.getindex(s::_BaseState, l::Int, c::Int) = s.fields[c][l]
Adapt.adapt_structure(to, s::_BaseState) = _BaseState(adapt(to, s.fields))

# The state a θ pass reads, in the linear form.
_linear(Q::ConservedState) = _LinearState(Q)
_linear(s::_BaseState) = s
