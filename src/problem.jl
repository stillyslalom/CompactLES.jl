# Frontend: problem specification decoupled from the numerical backend.
#
# The user-facing vocabulary is primitive and pointwise. Initial conditions
# are functions (x₁, x₂, x₃) → Prim; Dirichlet boundary forcing is a function
# (x₁, x₂, x₃, t) → Prim evaluated at the RK stage time. A `Problem` bundles
# physics, domain, boundary conditions, and the IC, with no reference to
# grids, ranks, halos, or conserved layouts, while `Numerics` bundles
# resolution and scheme choices. `setup(problem, numerics)` combines them and
# returns `(solver, Q)`. The same `Problem` can therefore be run at different
# resolutions, orders, or eventually on a different backend without changing
# the physics description. Primitive-to-conserved conversion follows the EOS
# contract that future cubic or tabular models can also implement.

"""
    Prim(; u=(0,0,0), p=NaN, T_ion=NaN, rho=NaN, Y=(1.0,))

Pointwise primitive state used by initial and prescribed-boundary functions.
All values are converted to `Float64`.

# Keywords

- `p`, `rho`, `T_ion`: pressure, mixture density, and temperature. Supply
  **exactly two**; [`conserved_from_prim`](@ref) derives the third through the
  EOS. `(p, rho)` and `(p, T_ion)` suit a specified pressure field, `(rho, T_ion)`
  a stratified or isothermal state whose pressure follows from the profile.
- `u`: three physical velocity components in the coordinate-aligned orthonormal
  basis. The default is a stationary state.
- `Y`: species mass fractions, in the order used to construct the EOS. They
  must sum to one; the default is a single species with unit mass fraction.

The omitted quantity is stored as `NaN` and is not filled in: a `Prim` records
what was specified, and the conversion never reads the value it derives. Code
reading a field back must therefore expect `NaN` in whichever one the caller
left out.

The number of entries in `Y` is checked against [`nspecies`](@ref) when the
state is converted with [`conserved_from_prim`](@ref). CompactLES does not
clip negative or out-of-range primitive values.
"""
struct Prim{N}
    Y::NTuple{N,Float64}
    u::NTuple{3,Float64}
    p::Float64
    T_ion::Float64
    rho::Float64
end

function Prim(; u=(0.0, 0.0, 0.0), p::Real=NaN, T_ion::Real=NaN, rho::Real=NaN,
              Y=(1.0,))
    count(isnan, (Float64(p), Float64(T_ion), Float64(rho))) == 1 ||
        error("Prim: specify exactly two of p, rho, and T_ion; the EOS derives " *
              "the third")
    Yt = Tuple(Float64.(Y))
    # The sum is taken in Float64, but the entries may arrive in a narrower type
    # whose own rounding is all the accuracy there is: (0.6f0, 0.4f0) widens to a
    # sum 3.0e-8 off unity, which a fixed 1e-10 rejects. Scale the tolerance with
    # the input's eps, as `setup`'s angle_tol does.
    Ytol = max(1e-10, 8 * length(Yt) *
                      maximum(x -> Float64(eps(float(typeof(x)))), Y;
                              init=eps(Float64)))
    abs(sum(Yt) - 1) < Ytol || error("Prim: mass fractions must sum to 1")
    Prim{length(Yt)}(Yt, Tuple(Float64.(u)), Float64(p), Float64(T_ion), Float64(rho))
end

"""
    conserved_from_prim(eos, pr::Prim) -> NTuple
    conserved_from_prim(equations, eos, pr::Prim) -> NTuple

Convert a pointwise primitive state to the conserved layout owned by
`equations`. The two-argument form uses [`NavierStokes1T`](@ref).

For `NavierStokes1T`, the result is
`(rho*Y[1], ..., rho*Y[Ns], rho*u[1], rho*u[2], rho*u[3], rho*E)`,
where the final entry is total energy per volume. The EOS supplies whichever of
density and temperature the [`Prim`](@ref) omitted, from `(p, T_ion, Y)` or
`(p, rho, Y)`. A `Prim` giving both needs no pressure: the conserved state is a
function of `(rho, T_ion, u, Y)` alone, so `p` may be left out.

A method throws if the number of mass fractions does not match the EOS.
Custom equation sets or EOS models extend the three-argument form.
"""
function conserved_from_prim(::NavierStokes1T, eos::IdealMixture, pr::Prim{N}) where {N}
    N == nspecies(eos) ||
        error("Prim carries $N mass fractions; EOS has $(nspecies(eos)) species")
    Rm = 0.0; cvm = 0.0
    for k in 1:N
        Rm  += pr.Y[k] * eos.Rk[k]
        cvm += pr.Y[k] * eos.cvk[k]
    end
    ρ = isnan(pr.rho) ? pr.p / (Rm * pr.T_ion) : pr.rho
    T_ion = isnan(pr.T_ion) ? pr.p / (ρ * Rm) : pr.T_ion
    ke = 0.5 * (pr.u[1]^2 + pr.u[2]^2 + pr.u[3]^2)
    (ntuple(k -> ρ * pr.Y[k], N)...,
     ρ * pr.u[1], ρ * pr.u[2], ρ * pr.u[3],
     ρ * (cvm * T_ion + ke))
end

function conserved_from_prim(::NavierStokes1T, eos::StiffenedGas, pr::Prim{N}) where {N}
    N == 1 || error("StiffenedGas is single-component; Prim carries $N mass fractions")
    R = gas_constant(eos)
    # p + p∞ = ρ R T_ion is the whole of the thermal EOS.
    ρ = isnan(pr.rho) ? (pr.p + eos.p_inf) / (R * pr.T_ion) : pr.rho
    T_ion = isnan(pr.T_ion) ? (pr.p + eos.p_inf) / (ρ * R) : pr.T_ion
    ke = 0.5 * (pr.u[1]^2 + pr.u[2]^2 + pr.u[3]^2)
    # ρe = ρ c_v T_ion + p∞, and E = ρe + ρ·ke.
    (ρ, ρ * pr.u[1], ρ * pr.u[2], ρ * pr.u[3],
     ρ * eos.cv * T_ion + eos.p_inf + ρ * ke)
end

function conserved_from_prim(::NavierStokes1T, eos::Nasa9Mixture, pr::Prim{N}) where {N}
    N == nspecies(eos) ||
        error("Prim carries $N mass fractions; EOS has $(nspecies(eos)) species")
    Rm = 0.0
    for k in 1:N
        Rm += pr.Y[k] * eos.Rk[k]
    end
    ρ = isnan(pr.rho) ? pr.p / (Rm * pr.T_ion) : pr.rho
    T_ion = isnan(pr.T_ion) ? pr.p / (ρ * Rm) : pr.T_ion
    e = 0.0
    for k in 1:N
        e += pr.Y[k] * species_energy(eos, k, T_ion)
    end
    ke = 0.5 * (pr.u[1]^2 + pr.u[2]^2 + pr.u[3]^2)
    (ntuple(k -> ρ * pr.Y[k], N)...,
     ρ * pr.u[1], ρ * pr.u[2], ρ * pr.u[3], ρ * (e + ke))
end

conserved_from_prim(eos::EOS, pr::Prim) =
    conserved_from_prim(NavierStokes1T(eos), eos, pr)
conserved_from_prim(species::IdealSpecies, pr::Prim) =
    conserved_from_prim(IdealMixture(species), pr)

@inline function write_conserved!(Q, I, solver, pr::Prim)
    q = conserved_from_prim(solver.equations, solver.eos, pr)
    @inbounds for c in 1:solver.equations.n_cons
        Q[I, c] = q[c]
    end
    return Q
end

"""
    initialize!(solver, Q, ic)

Overwrite the rank-local interior of conserved state `Q` from
`ic(x1, x2, x3) -> Prim`, evaluated at physical coordinates. The callback receives
neither ranks and halos nor the conserved-component layout, and should be pure
because it can be called concurrently from multiple threads.

This function leaves halo cells unchanged and does not reset solver time,
timestep history, or diagnostics. Use it to reuse an existing solver and
allocation for a different initial state with the same EOS, geometry, and
numerical configuration. It returns `Q`.

Rank-local and non-collective. Each rank writes only its own block, so the
halos hold whatever they held before and a caller needing them current must
exchange afterwards; [`run!`](@ref) and [`step!`](@ref) do so themselves.
"""
initialize!(solver::SolverLike, Q, ic) = _initialize!(solver, solver.eos, Q, ic)

initialize!(solver::Solver, states::Vector{<:ConservedState}, ic) =
    (foreach(((ps, Q),) -> _initialize!(ps, ps.eos, Q, ic),
             eachpatch(solver, states)); states)

function _initialize!(solver::SolverLike, eos, Q, ic)
    if _cpu_storage(Q)
        _initialize_interior!(solver, Q, ic)
    else
        # `ic` is an arbitrary host closure evaluated at physical coordinates,
        # so a device-resident patch initializes through a host staging copy:
        # download (preserving whatever the halos held, per the contract
        # above), fill the interior on the host, upload. Setup-only cost.
        Qh = Array(parent(Q))
        _initialize_interior!(solver, Qh, ic)
        copyto!(parent(Q), Qh)
    end
    return Q
end

function _initialize_interior!(solver::SolverLike, Q, ic)
    decomp = solver.decomp
    o1, o2, o3 = decomp.n_halo_d
    nx, ny, nz = decomp.n_local
    @threaded nx*ny*nz for jk in outer_indices(ny, nz)
        j, k = Tuple(jk)
        x2 = xcoord(solver, 2, j)
        x3 = xcoord(solver, 3, k)
        for i in 1:nx
            x1 = xcoord(solver, 1, i)
            write_conserved!(Q, CartesianIndex(i + o1, j + o2, k + o3), solver,
                             ic(x1, x2, x3))
        end
    end
    return Q
end

# ---------------------------------------------------------------------------
# State validation.
#
# Three questions are asked of every interior point: are its conserved values
# finite, are its partial densities positive, and does the EOS accept the point
# as one of its own. The first two are properties of the numbers; the third is
# `state_admissibility`, because the internal-energy gauge and the domain of
# validity belong to the model and not to the integrator (physics.jl carries the
# argument). What is done about a rejected point is `StepControl.validity`, and
# nothing here decides it.
#
# The sweep is serial and reduces once, on the pattern of `positivity_floors`.
# It is not a per-step cost unless a run installs a `StateGuard`; `setup` runs
# it once on the initial state.

"""
    state_report(solver, Q) -> StateReport

Inspect the interior of the conserved state and return the reduced
[`StateReport`](@ref) of what it contains. `Q` may be a single conserved array or
the vector of patch states of a multi-patch solver, in which case every patch
this rank holds is inspected.

Every rank in `solver.comm` must call this, since it ends in two `Allreduce`s;
each receives the totals over the whole domain rather than its own block. The
state is read and never written, and the halos are not inspected: a physical-edge
halo is never assigned by a boundary condition and holds whatever the last
exchange left there.

A state on device storage is not inspected and comes back as an empty report.
The sweep is a host loop, as the positivity failsafe is, and the storage choice
is solver-wide, so this returns early on every rank at once and cannot deadlock.
"""
function state_report(solver::Solver, Q)
    _cpu_storage(Q) || return StateReport()
    return _reduce_state_report(solver, _local_state_report(solver, Q))
end

function state_report(solver::Solver, states::Vector{<:ConservedState})
    isempty(states) || _cpu_storage(states[1]) || return StateReport()
    acc = _empty_local_report()
    for (ps, Q) in eachpatch(solver, states)
        acc = _merge_local_report(acc, _local_state_report(ps, Q))
    end
    return _reduce_state_report(solver, acc)
end

_empty_local_report() = (0, 0, 0, 0, 0, 0, 0, Inf, Inf)

_merge_local_report(a, b) =
    (a[1] + b[1], a[2] + b[2], a[3] + b[3], a[4] + b[4], a[5] + b[5],
     a[6] + b[6], a[7] + b[7], min(a[8], b[8]), min(a[9], b[9]))

# Two reductions rather than one: the counts add and the extrema do not. Both
# are small and this runs once per validation, not once per step of a run that
# has not asked for one.
function _reduce_state_report(solver::Solver, local_report)
    t0 = time_ns()
    counts = MPI.Allreduce(collect(Float64.(local_report[1:7])), +, solver.comm)
    extrema_reduced = MPI.Allreduce([local_report[8], local_report[9]], min,
                                    solver.comm)
    _wait!(solver, t0)
    return StateReport(round(Int, counts[1]), round(Int, counts[2]),
                       round(Int, counts[3]), round(Int, counts[4]),
                       round(Int, counts[5]), round(Int, counts[6]),
                       round(Int, counts[7]), extrema_reduced[1],
                       extrema_reduced[2])
end

function _local_state_report(solver::SolverLike, Q)
    decomp = solver.decomp
    o1, o2, o3 = decomp.n_halo_d
    nx, ny, nz = decomp.n_local
    eos = solver.eos
    n_species = solver.equations.n_species
    n_cons = solver.equations.n_cons
    m1, m2, m3 = solver.equations.i_mom
    i_energy = solver.equations.i_energy
    # The dead band of the artificial mass-fraction bound, reused here so the
    # validation and the regularization agree on what counts as an excursion.
    # Without it a mass fraction of −1e-17, which a filtered interface produces
    # over most of the domain, would be reported as an invalid state.
    Y_tolerance = solver.art.Y_tolerance
    # Arithmetic in the state's own type, so the validation reads the same
    # numbers the solver does under Float32; the reduced extrema are Float64.
    T = eltype(Q)
    points = 0; nonfinite = 0; negative_density = 0; negative_species = 0
    inadmissible = 0; unrecoverable = 0; extrapolated = 0
    ρ_min = Inf; e_min = Inf
    @inbounds for k in 1:nz, j in 1:ny, i in 1:nx
        I = CartesianIndex(i + o1, j + o2, k + o3)
        points += 1
        finite = true
        for c in 1:n_cons
            finite &= isfinite(Q[I, c])
        end
        # Nothing below this is meaningful at a point carrying NaN or Inf, and
        # the density and energy extrema would be poisoned by one.
        if !finite
            nonfinite += 1
            continue
        end
        ρ = zero(T)
        q_min = T(Inf)
        for sp in 1:n_species
            q = Q[I, sp]
            ρ += q
            q_min = min(q_min, q)
        end
        ρ_min = min(ρ_min, ρ)
        # The internal energy is not recoverable where the density is not
        # positive, and neither is the composition the EOS would be asked about.
        if !(ρ > 0)
            negative_density += 1
            continue
        end
        q_min < -T(Y_tolerance) * ρ && (negative_species += 1)
        ri = one(T) / ρ
        ke = (Q[I, m1]^2 + Q[I, m2]^2 + Q[I, m3]^2) / (2ρ)
        e = (Q[I, i_energy] - ke) / ρ
        e_min = min(e_min, e)
        flags = state_admissibility(eos, ρ, e, sp -> Q[I, sp] * ri, n_species)
        (flags & STATE_INADMISSIBLE) != 0 && (inadmissible += 1)
        (flags & STATE_UNRECOVERABLE) != 0 && (unrecoverable += 1)
        (flags & STATE_EXTRAPOLATED) != 0 && (extrapolated += 1)
    end
    return (points, nonfinite, negative_density, negative_species, inadmissible,
            unrecoverable, extrapolated, ρ_min, e_min)
end

"""
    validate_state!(solver, Q; control, stage, floors, warn) -> StateReport

Sweep the state with [`state_report`](@ref), apply `control.validity` to what it
finds, and return the report. Under `:strict` a rejected state raises
[`SolverFailure`](@ref)`(:invalid_state)`. Under `:permissive` the state is
accepted and, when `warn` is true, a rank-0 warning names what it contains.
Under `:repair` the positivity failsafe repairs the state first, against
`floors = (rho_floor, e_floor)` from `positivity_floors`, reports the
substitutions it made, and then rejects whatever remains.

`floors` defaults to `(0, 0)`, which leaves nothing to repair with, so a caller
in `:repair` mode must supply floors derived from a state that was still valid.
Deriving them from the state under validation is not that: `positivity_floors`
returns zeros for a state whose minima are not positive, which is exactly the
case a repair is wanted for. [`setup`](@ref) therefore validates the initial
state without repairing it.

`stage` names what is being validated in the failure message and the warning.
Collective: every rank must call this with the same arguments, and the verdict
comes from reduced counts, so a rejection is raised on every rank at once.
"""
function validate_state!(solver::Solver, Q; control::StepControl=solver.control,
                         stage::AbstractString="state",
                         floors::Tuple{Float64,Float64}=(0.0, 0.0),
                         warn::Bool=true)
    report, failure = _apply_validity!(solver, Q; control=control, stage=stage,
                                       floors=floors, warn=warn)
    failure === nothing || throw(failure)
    return report
end

# The same sweep, repair and verdict, returning the failure instead of raising
# it. `run!` needs the verdict as a value so that a rejection can take the
# rollback path its other failures take; every caller that has no trajectory to
# roll back to goes through `validate_state!` and raises.
function _apply_validity!(solver::Solver, Q; control::StepControl=solver.control,
                          stage::AbstractString="state",
                          floors::Tuple{Float64,Float64}=(0.0, 0.0),
                          warn::Bool=true)
    report = state_report(solver, Q)
    rank = MPI.Comm_rank(solver.comm)
    if control.validity === :repair && !state_valid(report) && floors[1] > 0
        tally = apply_positivity_floor!(solver, Q, floors[1], floors[2],
                                        control.floor_scope)
        if tally.cells > 0 || tally.low_energy > 0
            record_floor!(solver, tally)
            warn && rank == 0 &&
                @warn "validate_state!: repaired $(tally.cells) cell(s) of " *
                      "$stage and saw $(tally.low_energy) below the " *
                      "internal-energy floor. Mass added $(tally.mass), energy " *
                      "added $(tally.energy), momentum removed $(tally.momentum)."
        end
        report = state_report(solver, Q)
    end
    failure = check_validity(control, report, stage, solver.step, solver.t,
                             solver.dt_prev, solver.cfl)
    if failure === nothing && warn && rank == 0 && !state_valid(report)
        @warn "validate_state!: $stage accepted under validity = " *
              ":$(control.validity). $report"
    end
    return report, failure
end

"""
    StateGuard(solver, Q; control = solver.control)

A per-step state validation, to be paired with a [`Trigger`](@ref) and passed to
[`run!`](@ref) as a callback. [`state_guard`](@ref) is the one-line form.

`run!` checks the state entering each step through [`check_step`](@ref) and the
reduced quantities [`max_rate`](@ref) produces, which covers the mixture density
and the timestep but not the composition, the finiteness of every component, or
the thermodynamic domain of the EOS. Nor is any check applied to the state a run
returns: the last step's result is inspected only by the iteration that follows
it, and there is none after an `nmax`, `tfinal`, or callback exit. A guard runs
after every completed step, including that last one, and so covers both.

The floors the `:repair` mode needs are derived from `Q` at construction, which
is collective, on the same reasoning as [`run!`](@ref): they scale with the state
the run starts from, which is the last state known to be valid.

A guard counts its own work in `checks` and `rejected` and warns once per run
rather than once per step, since a front carrying a handful of rejected points
would otherwise produce thousands of identical warnings. Under
`StepControl(validity = :strict)` it raises [`SolverFailure`](@ref) from inside
the callback. That failure is collective, since the report behind it is reduced,
but it is not the retryable path: `run!` rolls back on what `check_step`
rejects, and an exception from a callback is not caught. A run that wants
rollback should keep its state checks in `check_step` and use the guard to
diagnose what the state contains.
"""
mutable struct StateGuard
    control::StepControl
    rho_floor::Float64
    e_floor::Float64
    checks::Int
    rejected::Int
    reported::Bool
end

function StateGuard(solver::Solver, Q; control::StepControl=solver.control)
    rho_floor, e_floor = control.validity === :repair ?
                         positivity_floors(solver, Q, control) : (0.0, 0.0)
    return StateGuard(control, rho_floor, e_floor, 0, 0, false)
end

function (guard::StateGuard)(solver::Solver, Q)
    guard.checks += 1
    report = state_report(solver, Q)
    if guard.control.validity === :repair && !state_valid(report) &&
       guard.rho_floor > 0
        tally = apply_positivity_floor!(solver, Q, guard.rho_floor,
                                        guard.e_floor, guard.control.floor_scope)
        (tally.cells > 0 || tally.low_energy > 0) && record_floor!(solver, tally)
        report = state_report(solver, Q)
    end
    state_valid(report) && return false
    guard.rejected += 1
    failure = check_validity(guard.control, report, "the state after step " *
                             string(solver.step), solver.step, solver.t,
                             solver.dt_prev, solver.cfl)
    failure === nothing || throw(failure)
    if !guard.reported && MPI.Comm_rank(solver.comm) == 0
        guard.reported = true
        @warn "StateGuard: the state after step $(solver.step), t = " *
              "$(solver.t), was accepted under validity = " *
              ":$(guard.control.validity). $report. Later steps are counted in " *
              "the guard's `rejected` field and not warned about again."
    end
    return false
end

"""
    state_guard(solver, Q; control = solver.control, interval = 1) -> Callback

A [`StateGuard`](@ref) paired with `EveryStep(interval)`, ready to pass to
[`run!`](@ref) as `callback`. Construct the guard directly when the counts it
accumulates are wanted after the run.
"""
state_guard(solver::Solver, Q; control::StepControl=solver.control,
            interval::Int=1) =
    Callback(EveryStep(interval), StateGuard(solver, Q; control=control))

"""
    tanh_blend(x, x0, delta)

Return `0.5 * (1 + tanh((x - x0) / delta))`, a smooth transition centered at
`x0`. For positive `delta`, the value rises from zero to one as `x` increases;
the magnitude of `delta` sets the transition width in the same units as `x`.
"""
tanh_blend(x, x0, δ) = 0.5 * (1 + tanh((x - x0) / δ))

# ---------------------------------------------------------------------------
# Time-dependent Dirichlet boundary forcing.

"""
    DirichletBC(fun)

Hard prescription of the full state on a boundary plane from
`fun(x₁, x₂, x₃, t) -> Prim`, evaluated at the RK stage time. This is the right
tool for supersonic or forced inflow, pistons, and oscillating
drivers. It over-constrains a subsonic boundary, where it will reflect the
outgoing acoustic wave; use [`NSCBCInflowBC`](@ref) there, which relaxes the
incoming amplitudes toward the same targets and accepts a stage-time `target`
function of this shape.

The boundary condition is parameterized by the callback type. Storing
`fun::Function` would require runtime dispatch in `enforce!` and infer its
`Prim` result as `Any`; specializing on `typeof(fun)` keeps both values concrete.
"""
struct DirichletBC{F} <: BoundaryCondition
    fun::F
end

function enforce!(bc::DirichletBC, Q, solver, d, side)
    plane = wallplane(solver.decomp, d, side)
    plane === nothing && return nothing
    t = solver.tstage
    if _cpu_storage(Q)
        @inbounds for I in plane
            i, j, k = interior_index(solver, I)
            write_conserved!(Q, I, solver,
                             bc.fun(xcoord(solver, 1, i), xcoord(solver, 2, j),
                                    xcoord(solver, 3, k), t))
        end
        return nothing
    end
    # `fun` is a host closure, so a device-resident patch evaluates the plane
    # block on the host and uploads it, one small strided assignment per face
    # per RK stage, the Dirichlet staging cost recorded in reference/AMR_GPU.md.
    r1, r2, r3 = plane.indices
    T = eltype(Q)
    n_cons = solver.equations.n_cons
    block = Array{T,4}(undef, length(r1), length(r2), length(r3), n_cons)
    @inbounds for (kk, K) in enumerate(r3), (jj, J) in enumerate(r2),
                  (ii, I1) in enumerate(r1)
        I = CartesianIndex(I1, J, K)
        i, j, k = interior_index(solver, I)
        q = conserved_from_prim(solver.equations, solver.eos,
                                bc.fun(xcoord(solver, 1, i), xcoord(solver, 2, j),
                                       xcoord(solver, 3, k), t))
        for c in 1:n_cons
            block[ii, jj, kk, c] = q[c]
        end
    end
    dev = similar(parent(Q), size(block))
    copyto!(dev, block)
    view(parent(Q), r1, r2, r3, :) .= dev
    nothing
end

# ---------------------------------------------------------------------------
# Backend-decoupled problem and numerics bundles.

"""
    Problem(; domain, bcs, ic, name="problem", eos=IdealSpecies("gas"; R=1, gamma=1.4),
            transport=Transport(), metric=CartesianMetric(), sources=())

Physical specification independent of grid resolution and process count. A
`Problem` can therefore be reused with different [`Numerics`](@ref) objects.

# Required keywords

- `domain`: three `(lo, hi)` coordinate intervals. [`setup`](@ref) requires
  every interval to have positive extent.
- `bcs`: three `(low, high)` pairs of [`BoundaryCondition`](@ref) objects, one
  pair per coordinate direction. A periodic direction must use `PeriodicBC` at
  both ends; a collapsed direction must also use a periodic pair.
- `ic`: pointwise function `(x1, x2, x3) -> Prim`. It should be pure because
  setup can evaluate it from multiple threads.

# Optional keywords

- `name`: human-readable problem label, used in displays. Default: `"problem"`.
- `eos`: equation of state and species definition. An [`IdealSpecies`](@ref)
  is promoted to a one-species [`IdealMixture`](@ref). Default: a
  nondimensional gas with `R = 1` and `gamma = 1.4`.
- `transport`: constant molecular transport properties. Default:
  [`Transport()`](@ref), which has zero molecular viscosity.
- `metric`: coordinate metric. Default: [`CartesianMetric()`](@ref).
- `sources`: tuple of explicit source objects applied to the RHS. Default: `()`.
  See [`add_source!`](@ref) when defining a custom source.

Coordinates passed to `ic` and to boundary callbacks follow `metric`. Species
mass fractions in every returned `Prim` must follow the order defined by `eos`.
"""
struct Problem
    name::String
    eos::EOS
    transport::Transport
    metric::Metric
    sources::Tuple
    domain::NTuple{3,Tuple{Float64,Float64}}
    bcs::NTuple{3,Tuple{BoundaryCondition,BoundaryCondition}}
    ic::Function
end

function Problem(; name="problem", eos=_default_ideal_mixture(),
                 transport=Transport(), metric=CartesianMetric(), sources=(),
                 domain, bcs, ic)
    return Problem(String(name), _as_eos(eos), transport, metric, sources,
                   domain, bcs, ic)
end

"""
    Numerics(; n_global, deriv=lele_d1_6(), filt=compact_filter(0.45),
             art=ArtParams(), cfl=0.5, control=StepControl(),
             filter_interval=1, filter_cfl=0.0, dims=nothing, n_halo=4,
             stretch=(nothing, nothing, nothing))

Grid, scheme, timestep, and decomposition choices used to realize a
[`Problem`](@ref).

# Keywords

- `n_global`: required three-tuple giving the global point count in each
  coordinate direction. A count of one collapses that direction: it has no
  derivative, halo, or decomposition.
- `deriv`: compact first-derivative scheme. Default: [`lele_d1_6()`](@ref);
  [`lele_d1_8()`](@ref) and [`lele_d1_10()`](@ref) are the higher-order presets.
- `filt`: compact filter applied to the conserved state. Default:
  [`compact_filter(0.45)`](@ref), where values nearer `0.5` filter more weakly.
  It doubles as the artificial-property sensor smoother only under
  `ArtParams(smoother = :compact)`; the default `:gaussian` smoother is an
  explicit stencil that ignores this keyword.
- `art`: artificial-property coefficients. Default: [`ArtParams()`](@ref).
- `cfl`: multiplier used by [`compute_dt`](@ref). Default: `0.5`. Strong shocks
  can require a lower startup value.
- `control`: timestep prediction, failure floors, and retry policy. Default:
  [`StepControl()`](@ref).
- `filter_interval`: state-filter cadence in completed steps. A positive value
  `k` applies `filt` every `k` steps, counted from the completed step number.
  The default `1` filters every step; `0` disables state filtering, which leaves
  `filt` unused altogether unless `ArtParams(smoother = :compact)` also selects
  it as the sensor smoother.
- `filter_cfl`: reference CFL for a full-strength filter pass, making the
  filter's dissipation a rate, not a per-application amount. The default
  `0.0` disables the relaxation and reproduces the unrelaxed solver bit for bit:
  each pass replaces the state with its filtered image, so halving the CFL
  doubles the number of passes over an interval and doubles the dissipation. A
  positive value instead relaxes toward the filtered state by
  `filter_interval · dt · rate / filter_cfl`, capped at one, which holds the
  dissipation per unit time fixed below that CFL. See
  [`filter_weight`](@ref).
- `dims`: MPI process-grid dimensions. `nothing` lets MPI distribute ranks over
  resolved directions. An explicit tuple must have product equal to the
  communicator size and must contain `1` in every collapsed direction.
- `n_halo`: halo layers on each side of a resolved local block. Default: `4`;
  it must be at least the largest explicit stencil half-width used by the
  selected schemes.
- `stretch`: one entry per direction, each either `nothing` for a uniform grid
  or a [`Stretch`](@ref). A mapping must span the corresponding `Problem.domain`
  interval and can be used only in a nonperiodic, non-folded direction.

Compact plans impose a scheme-dependent minimum rank-local extent. With the
defaults, each resolved local extent needs at least nine points because the
filter is the binding scheme. Reduce decomposition in that direction or
increase `n_global` if setup reports a smaller local block.

# Refinement keywords (reference/AMR_GPU.md)

- `refine`: a `BlockRegion` in root node space selecting static refinement
  at ratio 3 over that region, or a vector of them selecting a nested
  hierarchy, region ℓ given in level ℓ−1's node space (level-(ℓ−1) node
  `g` is level-ℓ node `3(g − 1) + 1`). Default `nothing`. The
  [`Solver`](@ref) constructor enforces the scope (Cartesian, unstretched,
  unfolded, a tridiagonal filter, nesting).
- `level_restriction`: `:inject` (default) writes the fine coincident-node
  values onto the covered region of the parent; `:filter` applies the
  invertible transfer pair's anti-alias filter first.
- `subcycle`: `false` (default) advances every level at the global dt;
  `true` selects the Berger–Oliger step, three steps of a third of the
  parent's step on each refined level, recursively, with Hermite boundary
  forcing. Requires `refine`.
- `regrid_interval`: `0` (default) keeps the region static; a positive `K`
  retags the coarse level every `K` steps and moves the region to the
  buffered bounding box of the tagged cells. Requires `refine` with a single
  region.
- `tag_threshold` (default `0.02`) and `tag_buffer` (default `4`): the
  tagging threshold on the relative undivided fourth difference of the
  mixture density, and the coarse-cell buffer added around tagged cells.
  The tag is the union of this criterion with the three below and the
  predicate, each evaluated per point over the parent level's state; every
  one of those is off by default.
- `tag_sensor_threshold` (default `0`, off): a threshold on the artificial
  diffusivity number ((μ\\* + β\\*)/ρ + κ\\*/(ρ c_p) + max_k D\\*_k) / (c h),
  the artificial diffusivity of the last right-hand-side evaluation in units
  of the acoustic cell diffusivity, read from the coefficient arrays the
  scheme itself wrote. It is the scheme's own statement that a feature is
  under-resolved; a captured Sod shock reads about 2 under the default
  `C_beta`. Zero wherever `art.enabled` is false.
- `tag_gradient_threshold` (default `0`, off): a threshold on the
  mass-fraction change per cell, max_k |δY_k| over the centered difference
  of one cell, for mixing layers. Dimensionless; 0.05 tags an interface
  resolved over about ten cells.
- `tag_vorticity_threshold` (default `0`, off): a threshold on the vorticity
  magnitude |∇ × u| from centered differences, in the run's units of
  inverse time.
- `tag_predicate` (default `nothing`): a function `(patch, I) -> Bool` over
  the parent patch (a [`PatchSolver`](@ref)) and the padded
  `CartesianIndex` of one interior node; `true` tags the node. It runs on
  the host, serially, at the regrid cadence, on every rank that holds a
  piece of the parent, and its coordinates come from
  [`xcoord`](@ref) through `interior_index`.
- `untag_ratio` (default `2`) and `tile_lifetime` (default `1`): the
  derefinement hysteresis. A node above a criterion's threshold divided by
  `untag_ratio` holds an existing tile (the current box, with `tile = 0`)
  without calling for a new one, so a tile at the edge of a feature does
  not flicker as the feature crosses the threshold; `1` disables the hold
  band. A tile is not dropped before `tile_lifetime` regrid checks have
  passed since its creation. The solver stores the tag history required by
  both forms of hysteresis, with identical values on every rank.
- `tile`: `0` (default) covers each refined region with one patch; a
  positive edge (in parent nodes, at least 3) covers it with the tiles of a
  global lattice of that edge instead, abutting tiles sharing their
  interface plane and coupled as root slabs are. Regridding then moves
  tiles in and out of the set, a surviving tile never changing its region.
- `rebalance` (default `0`, off) and `rebalance_persist` (default `2`): a
  threshold on the ratio of the largest to the mean per-rank busy time over
  a regrid interval, measured by the run, above which a tiled level is
  repartitioned on those measurements once the ratio has exceeded it at
  that many consecutive regrid checks. Requires `tile` and
  `regrid_interval`. Off, a surviving tile keeps its owner ranks across
  every regrid.
"""
Base.@kwdef struct Numerics
    n_global::NTuple{3,Int}
    deriv::AbstractCompactScheme = lele_d1_6()
    filt::AbstractCompactScheme = compact_filter(0.45)
    art::ArtParams = ArtParams()
    cfl::Float64 = 0.5
    control::StepControl = StepControl()
    filter_interval::Int = 1
    filter_cfl::Float64 = 0.0
    dims::Union{Nothing,NTuple{3,Int}} = nothing
    n_halo::Int = 4
    comm::MPI.Comm = MPI.COMM_WORLD
    stretch::NTuple{3,Union{Nothing,Stretch}} = (nothing, nothing, nothing)
    patch_grid::NTuple{3,Int} = (1, 1, 1)
    backend::AbstractBackend = CPUBackend()
    interface_rhs::Symbol = :extended
    refine::Union{Nothing,BlockRegion,Vector{BlockRegion}} = nothing
    level_restriction::Symbol = :inject
    subcycle::Bool = false
    regrid_interval::Int = 0
    tag_threshold::Float64 = 0.02
    tag_buffer::Int = 4
    tag_sensor_threshold::Float64 = 0.0
    tag_gradient_threshold::Float64 = 0.0
    tag_vorticity_threshold::Float64 = 0.0
    tag_predicate::Union{Nothing,Function} = nothing
    untag_ratio::Float64 = 2.0
    tile_lifetime::Int = 1
    tile::Int = 0
    rebalance::Float64 = 0.0
    rebalance_persist::Int = 2
end

"""
    setup(prob, num) -> (solver, Q)

Construct the distributed solver for `prob` using `num`, allocate a
halo-padded [`ConservedState`](@ref), and initialize its rank-local interior
from `prob.ic`.

Setup validates domain extents, boundary pairing, metric singularities,
stretch mappings, equation/EOS species counts, process-grid dimensions, halo
width, and scheme-specific local grid minima. MPI is initialized if necessary.
The returned `solver` owns the operator plans and runtime state; `Q` contains
the conserved variables and is ready to pass to [`run!`](@ref).

The initialized state is then validated with [`validate_state!`](@ref) under
`num.control`, so an initial condition that is not finite, has a nonpositive
density, or lies outside the thermodynamic domain of the EOS is rejected here
rather than integrated. `StepControl(validity = :permissive)` accepts it with a
report instead. No repair is applied at this point, whatever the mode, because
the floors a repair needs would have to come from the state being validated.
[`initialize!`](@ref) applies no validation of its own: it is rank-local and
non-collective, and this check is neither.

Collective over `num.comm` (`MPI.COMM_WORLD` by default): the decomposition
is built with `MPI.Cart_create` and its sub-communicators (on more than one
rank; a single rank borrows `num.comm` itself), so every rank of
that communicator must call `setup` with the same `prob` and `num`. A split
communicator lets two independent solvers share one job.
"""
function setup(prob::Problem, num::Numerics)
    origin = ntuple(d -> prob.domain[d][1], 3)
    L_domain = ntuple(d -> prob.domain[d][2] - prob.domain[d][1], 3)
    all(>(0), L_domain) || error("domain extents must be positive")
    for d in 1:3
        st = num.stretch[d]
        st === nothing && continue
        isapprox(st.x(0.0), prob.domain[d][1]; atol=1e-10 * L_domain[d]) &&
        isapprox(st.x(1.0), prob.domain[d][2]; atol=1e-10 * L_domain[d]) ||
            error("stretch mapping for dim $d does not span the domain: " *
                  "x(0) = $(st.x(0.0)), x(1) = $(st.x(1.0)), " *
                  "domain = $(prob.domain[d])")
    end
    solver = Solver(n_global=num.n_global, L_domain=L_domain, bcs=prob.bcs,
               eos=prob.eos, transport=prob.transport, art=num.art,
               metric=prob.metric, stretch=num.stretch, sources=prob.sources,
               origin=origin,
               deriv=num.deriv, filt=num.filt,
               cfl=num.cfl, control=num.control,
               filter_interval=num.filter_interval,
               filter_cfl=num.filter_cfl,
               dims=num.dims, n_halo=num.n_halo, comm=num.comm,
               patch_grid=num.patch_grid, backend=num.backend,
               interface_rhs=num.interface_rhs, refine=num.refine,
               level_restriction=num.level_restriction, subcycle=num.subcycle,
               regrid_interval=num.regrid_interval,
               tag_threshold=num.tag_threshold, tag_buffer=num.tag_buffer,
               tag_sensor_threshold=num.tag_sensor_threshold,
               tag_gradient_threshold=num.tag_gradient_threshold,
               tag_vorticity_threshold=num.tag_vorticity_threshold,
               tag_predicate=num.tag_predicate,
               untag_ratio=num.untag_ratio, tile_lifetime=num.tile_lifetime,
               tile=num.tile, rebalance=num.rebalance,
               rebalance_persist=num.rebalance_persist)
    Q = allocate_state(solver)
    initialize!(solver, Q, prob.ic)
    validate_state!(solver, Q; control=num.control, stage="the initial state")
    return solver, Q
end
