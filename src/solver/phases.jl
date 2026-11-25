# Phase changes: a run advanced under one set of boundary conditions, sources,
# transport or numerics continues under another. `run!` ends the first phase
# on a decision every rank shares (`tfinal`, `nmax`, or a callback effect
# returning `true`), and `setup(solver, Q; ...)` builds the next phase's
# solver and carries the run into it.
#
# The carry is this rank's checkpoint image (io.jl) written to memory and read
# back into the new solver, so a phase change continues exactly as a
# checkpoint written at the stop and loaded into the next phase's solver
# would: the header checks, the hierarchy restore and the run state are one
# code path, so the equivalence holds by construction and a test pins it. The
# image costs one copy of this rank's state for the duration of the call.
#
# The conditions change between two steps, never inside one, so every
# Runge–Kutta stage of a step sees one condition, and on every rank at the
# same step, so a condition whose correction is collective (the NSCBC faces)
# is entered by all ranks or by none. Both hold because the stop decision is
# collective; `WhenState` reduces its condition across the communicator for
# that reason.

const PHASE_SOURCE = "the previous phase"

"""
    setup(solver, Q; bcs, sources, transport, numerics) -> (solver, Q)

Build the solver of the next phase of a run and carry the run into it: the
same problem with the boundary conditions `bcs`, the source tuple `sources`,
the transport model `transport` or the [`Numerics`](@ref) `numerics` replaced,
continuing from `solver` and its state `Q` where [`run!`](@ref) left them.
An omitted keyword keeps the previous phase's value. `solver` must have been
built by [`setup`](@ref)`(problem, numerics)` (or by an earlier phase
change), which records the problem and numerics this method starts from.

```julia
solver, Q = setup(problem, numerics)
# End the first phase on a step that lands on t_open exactly.
run!(solver, Q; tfinal, callback = Callback(AtTime(t_open), Returns(true)))
solver, Q = setup(solver, Q; bcs = (x_bcs, y_bcs, (SlipWallBC(), NSCBCInflowBC(post))))
run!(solver, Q; tfinal)
```

The first phase may end at `tfinal` or `nmax`, or on any [`Callback`](@ref)
whose effect returns `true`; a [`WhenState`](@ref) trigger ends it when a
condition on the state is first met. A scheduled [`AtTime`](@ref) instant is
approached by the soft landing `run!` applies to every scheduled instant,
whereas `tfinal` clips the last step directly and may leave a short one. Every
rank decides either end identically, so the next phase may take a condition
whose correction is collective, such as [`NSCBCOutflowBC`](@ref), on every
rank at the same step.

The new solver is built by `setup(problem, numerics)` from the changed
inputs, and then takes the run state a checkpoint carries: the conserved
state of every patch of every level, the refinement hierarchy with its tile
ownership and regrid history, `t`, `step`, the step history `dt_prev`,
`rate_prev` and `filter_rate_prev`, and the artificial coefficients, with the
positivity-floor tally and the wall-clock totals besides. The result is the
same, bit for bit, as writing a checkpoint at the stop with
[`save_checkpoint`](@ref) and loading it into the next phase's solver with
[`load_checkpoint!`](@ref). `solver.cfl`, which a [`StepControl`](@ref)
retry lowers, is carried as well unless `numerics` gives a different `cfl`,
which then takes effect. The step control, the filter, the artificial-property
parameters and the rest of `numerics` are the next phase's.

A phase may also switch the artificial properties on or off through
`numerics.art.enabled`, which a checkpoint restart refuses. Switched off,
the previous phase's coefficients are dropped. Switched on, the coefficients
are computed from the carried state by one right-hand-side evaluation before
this method returns, as [`run!`](@ref) computes them before the first step
of a run, so the next phase's first step is sized and dissipated as the
first step of a run started from the carried state.

The grid, the process grid and the conserved layout are kept, and a change
that would alter them raises an `ArgumentError`: a different `n_global`,
`n_halo` or `execution` (communicator, `dims`, backend, `patch_grid`), and a
boundary condition that changes a dimension's periodicity or its fold
([`AxisBC`](@ref), [`OriginBC`](@ref), [`PoleBC`](@ref),
[`SymmetryPlaneBC`](@ref)), either of which moves the grid points. The
equation of state, the metric and the domain are not keywords. A refined
level's tile edge must also be kept, and a same-level `patch_grid` layout,
which has no checkpoint, has no phase change either.

Rebind both results, as above: the returned state is new storage. On host
storage the previous solver and state are left as they were. On device storage
their device arrays are freed before this method returns, since the garbage
collector may leave them allocated for the rest of the run, and
[`run!`](@ref) refuses the previous solver afterwards. Callbacks are not part of
the solver, so a [`FieldWriter`](@ref) passed to the next phase's `run!`
continues its frame sequence. Collective over the solver's communicator.
"""
function setup(prev::Solver, Q; bcs=nothing, sources=nothing, transport=nothing,
               numerics::Union{Nothing,Numerics}=nothing)
    inputs = getfield(prev, :inputs)
    inputs === nothing &&
        throw(ArgumentError("setup(solver, Q): this solver was built by " *
                            "Solver(; ...), which records no Problem and Numerics " *
                            "for the next phase to start from; build the first " *
                            "phase with setup(problem, numerics)"))
    prob0, num0 = inputs.problem, inputs.numerics
    prob = Problem(name=prob0.name, eos=prob0.eos,
                   transport=transport === nothing ? prob0.transport : transport,
                   metric=prob0.metric,
                   sources=sources === nothing ? prob0.sources : sources,
                   domain=prob0.domain, bcs=bcs === nothing ? prob0.bcs : bcs,
                   ic=prob0.ic)
    num = numerics === nothing ? num0 : numerics
    _check_phase(prob0, num0, prob, num)
    next, Q_next = setup(prob, num)
    _carry_phase!(next, Q_next, prev, Q)
    # The retry-lowered CFL continues unless the next phase asks for another.
    num.cfl == num0.cfl || (next.cfl = oftype(next.cfl, num.cfl))
    _release_previous_phase!(prev, Q, next, Q_next)
    return next, Q_next
end

# The previous phase's device arrays, freed at once. Left to the garbage
# collector they outlive the phase: a device array is freed by its finalizer,
# which only a full collection runs once the previous solver has aged into the
# old generation, and the host collector, which keys on host allocation, has
# no reason to make one. A full collection does not free them either while the
# function that made the phase change is running, since its frame still roots
# the previous solver whatever the variable is rebound to. Measured on a
# 128 × 32 × 32 two-species phase change made inside a function, twenty
# incremental collections and one full one left both phases resident; on a
# 16 GB card running a 7 M node case that spilled into host memory and made
# each step 60% slower for the rest of the run. Any array the next phase
# reaches as well is kept.
function _release_previous_phase!(prev::Solver, Q, next::Solver, Q_next)
    _cpu_storage(Q_next isa ConservedState ? Q_next : first(Q_next)) && return prev
    keep = _device_arrays(next, Q_next)
    for a in _device_arrays(prev, Q)
        a in keep || KernelAbstractions.unsafe_free!(a)
    end
    setfield!(prev, :released, true)
    return prev
end

const _HostDense = isdefined(Core, :GenericMemory) ? Union{Array,Core.GenericMemory} :
                                                     Array

# The device arrays reachable from `roots`, by identity: the leaves of a walk
# through fields, tuples and arrays of references, a device array being a
# dense array that is not host memory. Types and modules are values here, not
# containers, so the walk stays within the solver's own data.
function _device_arrays(roots...)
    found = IdSet{Any}()
    seen = IdSet{Any}()
    stack = Any[roots...]
    while !isempty(stack)
        x = pop!(stack)
        (isbits(x) || x isa Union{Type,Module,Symbol,String,Task}) && continue
        x in seen && continue
        push!(seen, x)
        if x isa _HostDense
            isbitstype(eltype(x)) && continue
            for i in eachindex(x)
                isassigned(x, i) && push!(stack, x[i])
            end
        elseif x isa DenseArray
            push!(found, x)
        else
            for f in 1:nfields(x)
                isdefined(x, f) && push!(stack, getfield(x, f))
            end
        end
    end
    return found
end

function _check_not_released(solver::Solver)
    getfield(solver, :released) &&
        throw(ArgumentError("run!: this solver's device storage was freed by the " *
                            "phase change setup(solver, Q); continue with the " *
                            "solver and state that call returned"))
    return solver
end

# The changes that would move the grid points, the process grid or the
# conserved layout, refused before anything is built. The checkpoint header
# checks the same things again during the carry; these name the keyword.
function _check_phase(prob0::Problem, num0::Numerics, prob::Problem, num::Numerics)
    phase_error(msg) = throw(ArgumentError("setup(solver, Q): " * msg))
    for d in 1:3, side in 1:2
        a, b = prob0.bcs[d][side], prob.bcs[d][side]
        face = "dimension $d, " * (side == 1 ? "low" : "high") * " face"
        isperiodic(a) == isperiodic(b) ||
            phase_error("the $face changes from $(type_name(a)) to " *
                        "$(type_name(b)), which changes the dimension's " *
                        "periodicity and with it the grid spacing")
        (_is_fold_bc(a) || _is_fold_bc(b)) && typeof(a) != typeof(b) &&
            phase_error("the $face changes from $(type_name(a)) to " *
                        "$(type_name(b)); a fold condition places the grid " *
                        "points and is kept between phases")
    end
    for (name, a, b) in (("n_global", num0.n_global, num.n_global),
                         ("n_halo", num0.n_halo, num.n_halo))
        a == b ||
            phase_error("numerics.$name is $b and the previous phase's is $a; " *
                        "a phase change keeps the grid and the fields it carries")
    end
    refined(n) = n.amr !== nothing
    refined(num0) == refined(num) ||
        phase_error("the " * (refined(num0) ? "previous" : "next") * " phase has " *
                    "a refinement hierarchy and the other has none; a phase " *
                    "change carries the hierarchy and cannot add or remove it")
    e0, e = num0.execution, num.execution
    for (name, a, b) in (("comm", e0.comm, e.comm), ("dims", e0.dims, e.dims),
                         ("patch_grid", e0.patch_grid, e.patch_grid),
                         ("backend", typeof(e0.backend), typeof(e.backend)))
        a == b ||
            phase_error("numerics.execution.$name differs from the previous " *
                        "phase's; a phase change keeps the process grid and " *
                        "the storage")
    end
    return nothing
end

function _carry_phase!(next::Solver, Q_next, prev::Solver, Q)
    _multipatch(prev) &&
        throw(ArgumentError("setup(solver, Q): this solver holds a patch layout; " *
                            "pass the state vector allocate_state returned"))
    buf = _write_checkpoint(IOBuffer(), prev, Q)
    seekstart(buf)
    _read_checkpoint!(buf, next, Q_next, PHASE_SOURCE, CONFIG_ALLOWABLE_GROUPS;
                      phase=true)
    refresh_primitives!(next, Q_next)
    _prime_phase!(next, Q_next, prev)
    _carry_accounts!(next, prev)
    return Q_next
end

function _carry_phase!(next::Solver, states_next::Vector{<:ConservedState},
                       prev::Solver, states::Vector{<:ConservedState})
    _check_hierarchy_layout(prev, "setup(solver, Q)")
    buf = _write_checkpoint(IOBuffer(), prev, states, hierarchy_record(prev))
    seekstart(buf)
    _read_checkpoint!(buf, next, states_next, PHASE_SOURCE, CONFIG_ALLOWABLE_GROUPS;
                      phase=true)
    # As `load_checkpoint!` leaves a loaded hierarchy; see the note there.
    _presync!(next, states_next, false)
    refresh_primitives!(next, states_next)
    _prime_phase!(next, states_next, prev)
    _carry_accounts!(next, prev)
    return states_next
end

# The artificial coefficients of a phase that switches the properties on,
# which the previous phase did not compute and the image does not carry.
# `run!` computes them only at step 0, so they are computed here from the
# carried state, by the evaluation `run!` makes there, into a scratch that is
# dropped on return. Without it the first step is sized on the acoustic and
# advective rates alone, the case the note on `_prime_coefficients!` describes.
function _prime_phase!(next::Solver, Q, prev::Solver)
    (next.art.enabled && !prev.art.enabled) || return next
    return _prime_art!(next, Q, Workspace(Q))
end

# What a checkpoint leaves behind and a run in memory keeps: the failsafe's
# tally, whose totals `run!` reports per call, and the wall-clock totals a
# `ProgressLog` projects the remaining time from. None of it enters a step.
function _carry_accounts!(next::Solver, prev::Solver)
    setfield!(next, :floor_tally, deepcopy(getfield(prev, :floor_tally)))
    for name in (:wall_step, :wall_total, :wall_wait, :wait_total)
        setfield!(next, name, getfield(prev, name))
    end
    return next
end
