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
  must be nonnegative and sum to one, each within a rounding tolerance; the
  default is a single species with unit mass fraction.

The omitted quantity is stored as `NaN` and is not filled in: a `Prim` records
what was specified, and the conversion never reads the value it derives. Code
reading a field back must therefore expect `NaN` in whichever one the caller
left out.

The given values must be finite, `rho` positive and `T_ion` nonnegative, and
the constructor raises an `ArgumentError` otherwise. The pressure is not
checked here, since its admissible range depends on the EOS; the state a
pressure produces is checked by the state validation of [`setup`](@ref). The
number of entries in `Y` is checked against [`nspecies`](@ref) when the state
is converted with [`conserved_from_prim`](@ref). CompactLES does not clip
primitive values.
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
        throw(ArgumentError("Prim: specify exactly two of p, rho, and T_ion; the " *
                            "EOS derives the third"))
    # NaN marks the omitted quantity, so only an infinite value is caught here.
    all(x -> !isinf(x), (p, T_ion, rho)) ||
        throw(ArgumentError("Prim: p, rho and T_ion must be finite, got " *
                            "p = $p, rho = $rho, T_ion = $T_ion"))
    isnan(rho) || rho > 0 ||
        throw(ArgumentError("Prim: rho must be positive, got $rho"))
    # A cold state (T_ion = 0, or p = 0 beside rho) is a legitimate initial
    # condition under `StepControl(validity = :permissive)`.
    isnan(T_ion) || T_ion >= 0 ||
        throw(ArgumentError("Prim: T_ion must be nonnegative, got $T_ion"))
    length(u) == 3 && all(isfinite, u) ||
        throw(ArgumentError("Prim: u must hold three finite components, got $u"))
    all(isfinite, Y) ||
        throw(ArgumentError("Prim: mass fractions must be finite, got $Y"))
    Yt = Tuple(Float64.(Y))
    # The sum is taken in Float64, but the entries may arrive in a narrower type
    # whose own rounding is all the accuracy there is: (0.6f0, 0.4f0) widens to a
    # sum 3.0e-8 off unity, which a fixed 1e-10 rejects. Scale the tolerance with
    # the input's eps, as `setup`'s angle_tol does.
    Ytol = max(1e-10, 8 * length(Yt) *
                      maximum(x -> Float64(eps(float(typeof(x)))), Y;
                              init=eps(Float64)))
    abs(sum(Yt) - 1) < Ytol ||
        throw(ArgumentError("Prim: mass fractions must sum to 1, got $Y " *
                            "(sum $(sum(Yt)))"))
    all(>=(-Ytol), Yt) ||
        throw(ArgumentError("Prim: mass fractions must be nonnegative, got $Y"))
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
`ic(x1, x2, x3) -> Prim`, or its spacing-aware form
`ic(x1, x2, x3, h) -> Prim`, evaluated at physical coordinates. `h` is the
smallest physical mesh spacing at that point, excluding collapsed directions;
it includes local stretching, metric scale factors, and refined-patch spacing.
The callback receives neither ranks and halos nor the conserved-component
layout, and should be pure because it can be called concurrently from multiple
threads.

This function leaves halo cells unchanged and does not reset solver time,
timestep history, or diagnostics. Use it to reuse an existing solver and
allocation for a different initial state with the same EOS, geometry, and
numerical configuration. It returns `Q`.

On a solver that has already stepped, a following [`run!`](@ref) continues
from the old clock and step counter, so its `tfinal` and `nmax` are counted
from there, and it sizes its first step from the artificial coefficients and
the rate history of the old state. To run the new state as a new calculation
from `t = 0`, also set `solver.t`, `solver.step`, `solver.dt_prev` and
`solver.rate_prev` to zero; `run!` then forms the coefficients from the new
state before its first step and runs the initial-state callbacks.

Rank-local and non-collective. Each rank writes only its own block, so the
halos hold whatever they held before and a caller needing them current must
exchange afterwards; [`run!`](@ref) and [`step!`](@ref) do so themselves.
"""
initialize!(solver::SolverLike, Q, ic) =
    _initialize!(solver, solver.eos, Q, _bind_initial(ic, solver))

initialize!(solver::Solver, states::Vector{<:ConservedState}, ic) =
    (foreach(((ps, Q),) -> _initialize!(ps, ps.eos, Q, _bind_initial(ic, ps)),
             eachpatch(solver, states)); states)

# An initial condition that needs the run's EOS or its resolved directions
# (a `Layers`, regions.jl) is completed here; a plain function is used as is.
_bind_initial(ic, solver) = ic

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

# The coordinate a user's function of position sees at interior index `i` (an
# initial condition, an AMR predicate): `xcoord`, except on a refined tile
# across a periodic seam, whose nodes past the domain's face (its coordinates
# run on there, `_level_period`) are read a period back, where the function
# is defined. A node inside the domain keeps its coordinate bit for bit.
@inline _domain_coordinate(solver::Solver, d::Int, i::Int) = xcoord(solver, d, i)
function _domain_coordinate(ps::PatchSolver, d::Int, i::Int)
    patch = ps.patch
    patch.level == 0 && return xcoord(ps, d, i)
    P = _level_period(ps.solver, patch.level)[d]
    g = patch.region.offset[d] + ps.decomp.offset[d] + i
    return P > 0 && g > P ? global_xcoord(ps, d, g - P) :
           P > 0 && g < 1 ? global_xcoord(ps, d, g + P) : xcoord(ps, d, i)
end

function _initialize_interior!(solver::SolverLike, Q, ic)
    decomp = solver.decomp
    o1, o2, o3 = decomp.n_halo_d
    nx, ny, nz = decomp.n_local
    x1_0 = _domain_coordinate(solver, 1, 1)
    x2_0 = _domain_coordinate(solver, 2, 1)
    x3_0 = _domain_coordinate(solver, 3, 1)
    cb = initial_callback(ic, x1_0, x2_0, x3_0,
                          point_spacing(solver, CartesianIndex(o1 + 1, o2 + 1,
                                                                 o3 + 1)))
    @threaded nx*ny*nz for jk in outer_indices(ny, nz)
        j, k = Tuple(jk)
        x2 = _domain_coordinate(solver, 2, j)
        x3 = _domain_coordinate(solver, 3, k)
        for i in 1:nx
            x1 = _domain_coordinate(solver, 1, i)
            I = CartesianIndex(i + o1, j + o2, k + o3)
            write_conserved!(Q, I, solver,
                             pointwise_initial(cb, x1, x2, x3,
                                               point_spacing(solver, I)))
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
    state_report(solver, Q; species_band = solver.control.species_band)
        -> StateReport

Inspect the interior of the conserved state and return the reduced
[`StateReport`](@ref) of what it contains. `Q` may be a single conserved array or
the vector of patch states of a multi-patch solver, in which case every patch
this rank holds is inspected. A point with a mass fraction below
`-species_band` or above `1 + species_band` is counted as a negative species
(the second implies a negative partner); the policy functions pass the band of the
[`StepControl`](@ref) whose verdict they take.

Every rank in `solver.comm` must call this, since it ends in two `Allreduce`s;
each receives the totals over the whole domain rather than its own block. The
state is read and never written, and the halos are not inspected: a physical-edge
halo is never assigned by a boundary condition and holds whatever the last
exchange left there.

A state on device storage is not inspected and comes back as an empty report.
The sweep is a host loop, as the positivity failsafe is, and the storage choice
is solver-wide, so this returns early on every rank at once and cannot deadlock.
"""
function state_report(solver::Solver, Q;
                      species_band::Real=solver.control.species_band)
    _cpu_storage(Q) || return StateReport()
    return _reduce_state_report(solver,
                                _local_state_report(solver, Q, species_band))
end

function state_report(solver::Solver, states::Vector{<:ConservedState};
                      species_band::Real=solver.control.species_band)
    isempty(states) || _cpu_storage(states[1]) || return StateReport()
    acc = _empty_local_report()
    for (ps, Q) in eachpatch(solver, states)
        acc = _merge_local_report(acc, _local_state_report(ps, Q, species_band))
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
# `comm` is a level's communicator for the substep sweep of a refined level.
function _reduce_state_report(solver::Solver, local_report, comm=solver.comm)
    t0 = time_ns()
    counts = MPI.Allreduce(collect(Float64.(local_report[1:7])), +, comm)
    extrema_reduced = MPI.Allreduce([local_report[8], local_report[9]], min, comm)
    _wait!(solver, t0)
    return StateReport(round(Int, counts[1]), round(Int, counts[2]),
                       round(Int, counts[3]), round(Int, counts[4]),
                       round(Int, counts[5]), round(Int, counts[6]),
                       round(Int, counts[7]), extrema_reduced[1],
                       extrema_reduced[2])
end

# Whether a composition lies outside the species band: a partial density below
# `-band * ρ` or above `(1 + band) * ρ`, given the smallest and largest partial
# densities and their sum `ρ > 0`. With two species the two sides are the same
# point up to rounding. With more, an excess of one species can be shared among
# partners that each stay inside the band, and only the upper side sees it.
# The failsafe clips on this test with the same arithmetic, so every point the
# report counts is a point `:repair` acts on, and no other.
@inline function _outside_species_band(q_min, q_max, ρ, band)
    ρ_band = band * ρ
    return q_min < -ρ_band || q_max - ρ > ρ_band
end

function _local_state_report(solver::SolverLike, Q, species_band)
    decomp = solver.decomp
    o1, o2, o3 = decomp.n_halo_d
    nx, ny, nz = decomp.n_local
    eos = solver.eos
    n_species = solver.equations.n_species
    n_cons = solver.equations.n_cons
    m1, m2, m3 = solver.equations.i_mom
    i_energy = solver.equations.i_energy
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
        q_max = T(-Inf)
        for sp in 1:n_species
            q = Q[I, sp]
            ρ += q
            q_min = min(q_min, q)
            q_max = max(q_max, q)
        end
        ρ_min = min(ρ_min, ρ)
        # The internal energy is not recoverable where the density is not
        # positive, and neither is the composition the EOS would be asked about.
        if !(ρ > 0)
            negative_density += 1
            continue
        end
        _outside_species_band(q_min, q_max, ρ, T(species_band)) &&
            (negative_species += 1)
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
function _validity_repair!(solver, Q, floors, control, stage, warn, rank)
    tally = apply_positivity_floor!(solver, Q, floors[1], floors[2],
                                    control.floor_scope;
                                    species_band=control.species_band)
    if tally.cells > 0 || tally.low_energy > 0
        record_floor!(solver, tally)
        warn && rank == 0 &&
            @warn "validate_state!: repaired $(tally.cells) cell(s) of " *
                  "$stage and saw $(tally.low_energy) below the " *
                  "internal-energy floor. Mass added $(tally.mass), energy " *
                  "added $(tally.energy), momentum removed $(tally.momentum)."
    end
    return nothing
end

function _apply_validity!(solver::Solver, Q; control::StepControl=solver.control,
                          stage::AbstractString="state",
                          floors::Tuple{Float64,Float64}=(0.0, 0.0),
                          warn::Bool=true)
    band = control.species_band
    report = state_report(solver, Q; species_band=band)
    rank = MPI.Comm_rank(solver.comm)
    if control.validity === :repair && !state_valid(report) && floors[1] > 0
        # Off by default (`validity = :strict`), so behind `_cold` (timestep.jl).
        _validity_repair!(_cold(solver), Q, floors, control, stage, warn, rank)
        report = state_report(solver, Q; species_band=band)
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

`run!` validates the state on entry and return. Its per-step [`check_step`](@ref)
and [`max_rate`](@ref) checks cover mixture density and timestep limits; a
positive `StepControl.validity_interval` additionally requests periodic full
state validation. A guard inspects every completed state, including its
composition, component finiteness, and EOS domain, and records rejected states
in its own counters.

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
    band = guard.control.species_band
    report = state_report(solver, Q; species_band=band)
    if guard.control.validity === :repair && !state_valid(report) &&
       guard.rho_floor > 0
        tally = apply_positivity_floor!(solver, Q, guard.rho_floor,
                                        guard.e_floor, guard.control.floor_scope;
                                        species_band=band)
        (tally.cells > 0 || tally.low_energy > 0) && record_floor!(solver, tally)
        report = state_report(solver, Q; species_band=band)
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
`fun(x₁, x₂, x₃, t) -> Prim`, or the spacing-aware
`fun(x₁, x₂, x₃, t, h) -> Prim`, evaluated at the RK stage time. `h` is the
smallest physical mesh spacing at the boundary point. This is the right tool
for supersonic or forced inflow, pistons, and oscillating
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

"""
    DirichletBC(state::Prim)

A constant [`DirichletBC`](@ref) holding `state` on the whole face, for
example the post-shock state from [`shock_jump`](@ref) at a supersonic inflow.
"""
DirichletBC(state::Prim) = DirichletBC(_ConstantState(state))

struct _ConstantState{N}
    state::Prim{N}
end
(c::_ConstantState)(x1, x2, x3, t) = c.state

function enforce!(bc::DirichletBC, Q, solver, d, side)
    plane = wallplane(solver.decomp, d, side)
    plane === nothing && return nothing
    t = solver.tstage
    I0 = first(plane)
    i0, j0, k0 = interior_index(solver, I0)
    cb = boundary_callback(bc.fun, xcoord(solver, 1, i0), xcoord(solver, 2, j0),
                           xcoord(solver, 3, k0), t, point_spacing(solver, I0))
    if _cpu_storage(Q)
        @inbounds for I in plane
            i, j, k = interior_index(solver, I)
            x1, x2, x3 = xcoord(solver, 1, i), xcoord(solver, 2, j),
                         xcoord(solver, 3, k)
            write_conserved!(Q, I, solver,
                             pointwise_boundary(cb, x1, x2, x3, t,
                                                point_spacing(solver, I)))
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
        x1, x2, x3 = xcoord(solver, 1, i), xcoord(solver, 2, j),
                     xcoord(solver, 3, k)
        q = conserved_from_prim(solver.equations, solver.eos,
                                pointwise_boundary(cb, x1, x2, x3, t,
                                                   point_spacing(solver, I)))
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
            transport=ConstantTransport(), metric=CartesianMetric(), sources=())

Physical specification independent of grid resolution and process count. A
`Problem` can therefore be reused with different [`Numerics`](@ref) objects.

# Required keywords

- `domain`: three `(lo, hi)` coordinate intervals. [`setup`](@ref) requires
  every interval to have positive extent.
- `bcs`: three entries, one per direction. Each is a [`BoundaryCondition`](@ref)
  for both faces, or a `(low, high)` pair. For example,
  `(SlipWallBC(), PeriodicBC(), PeriodicBC())`. A singleton `(condition,)` is
  also accepted and expanded to a pair. A periodic or collapsed direction
  must be periodic at both ends.
- `ic`: pointwise function `(x1, x2, x3) -> Prim` or
  `(x1, x2, x3, h) -> Prim`, with `h` the minimum local physical spacing over
  resolved directions. It should be pure because setup can call it on threads.

# Optional keywords

- `name`: human-readable problem label, used in displays. Default: `"problem"`.
- `eos`: equation of state and species definition. An [`IdealSpecies`](@ref)
  is promoted to a one-species [`IdealMixture`](@ref). Default: a
  nondimensional gas with `R = 1` and `gamma = 1.4`.
- `transport`: molecular transport model. Default:
  [`ConstantTransport()`](@ref), which has zero molecular viscosity.
- `metric`: coordinate metric. Default: [`CartesianMetric()`](@ref).
- `sources`: tuple of explicit source objects applied to the RHS. Default: `()`.
  See [`add_source!`](@ref) when defining a custom source.

Coordinates passed to `ic` and to boundary callbacks follow `metric`. Species
mass fractions in every returned `Prim` must follow the order defined by `eos`.
"""
struct Problem{F}
    name::String
    eos::EOS
    transport::AbstractTransport
    metric::Metric
    sources::Tuple
    domain::NTuple{3,Tuple{Float64,Float64}}
    bcs::NTuple{3,Tuple{BoundaryCondition,BoundaryCondition}}
    ic::F
end

function Problem(; name="problem", eos=_default_ideal_mixture(),
                 transport=ConstantTransport(), metric=CartesianMetric(), sources=(),
                 domain, bcs, ic)
    return Problem{typeof(ic)}(String(name), _as_eos(eos), transport, metric, sources,
                   domain, _face_conditions(bcs), ic)
end

# A copy of `x` with the fields named in `kw` replaced, built through the type's
# positional constructor, so a parametric type converts the new values to its
# own parameter and a validating inner constructor still validates.
function _with(x::S, kw) where {S}
    names = fieldnames(S)
    for k in keys(kw)
        k in names || throw(ArgumentError("$(nameof(S)) has no field `$k`; its fields " *
                                          "are $(join(names, ", "))"))
    end
    return S(map(f -> haskey(kw, f) ? kw[f] : getfield(x, f), names)...)
end

"""
    StateFilter(scheme = compact_filter(0.47); interval = 1, cfl = 0.35,
                weighting = :none)
    StateFilter(base::StateFilter; keywords...)

The state filter of [`Numerics`](@ref): the compact filter applied to the
conserved state between steps, and its cadence and strength. The second form
copies `base` with the given fields replaced. `Numerics(filter = scheme)` is
shorthand for `StateFilter(scheme)`, and `Numerics(filter = nothing)` for
`StateFilter(interval = 0)`.

- `scheme`: the compact filter. Default: [`compact_filter(0.47)`](@ref),
  where values nearer `0.5` filter more weakly. It doubles as the
  artificial-property sensor smoother only under
  `ArtificialProperties(smoother = :compact)`; the default `:gaussian`
  smoother is an explicit stencil that ignores it.
- `interval`: cadence in completed steps. A positive value `k` applies the
  filter every `k` steps, counted from the completed step number. The default
  `1` filters every step; `0` disables state filtering, which leaves `scheme`
  unused unless `ArtificialProperties(smoother = :compact)` selects it as the
  sensor smoother.
- `cfl`: reference CFL of a full-strength filter pass, making the filter's
  dissipation a rate, not a per-application amount. A pass along a direction
  relaxes toward the filtered state by `interval · dt · r_d · √n / cfl`,
  capped at one, with `r_d` that direction's one-dimensional hyperbolic rate
  `(|u_d| + c) / h_d` and `n` the number of active dimensions, which holds the
  dissipation per unit time fixed below that CFL and independent of the
  diffusive rates, physical or artificial, and of the spacing of the other
  directions. The default `0.35` is the reference CFL of the Taylor–Green
  fits: at or above it, on an isotropic grid under an acoustic-limited step,
  every pass is at full strength, and below it the run receives the
  dissipation per unit time of one at the reference. `0.0` disables the
  relaxation: each pass then replaces the state with its filtered image, so
  halving the CFL doubles the number of passes over an interval and doubles
  the dissipation. See [`filter_weight`](@ref).
- `weighting`: how the filter treats a non-uniform cell volume. `:none`, the
  default, filters each conserved component unweighted. `:volume` filters the
  component weighted by the cell volume and divides by the volume passed
  through the same filter, the form of the public Pyranda implementation. Both
  preserve a uniform state exactly on every metric. The conservation defect of
  a pass sits in the wall closure rows under either form and is the same on
  uniform and clustered grids; at a cylindrical axis or a spherical pole the
  weighted form is the less conservative of the two, and on the Noh
  implosions it moves the wall deficit in opposite directions at the axis and
  at the origin, which is why it is not the default. On a uniform Cartesian
  grid the two are one operator bit for bit. See [`filter_state!`](@ref).

`interval` must be nonnegative and `cfl` finite and nonnegative; both are
checked when the solver is built.
"""
struct StateFilter
    scheme::AbstractCompactScheme
    interval::Int
    cfl::Float64
    weighting::Symbol
end

StateFilter(scheme::AbstractCompactScheme=compact_filter(); interval::Integer=1,
            cfl::Real=0.35, weighting::Symbol=:none) =
    StateFilter(scheme, interval, cfl, weighting)
StateFilter(base::StateFilter; kw...) = _with(base, kw)

_state_filter(filter::StateFilter) = filter
_state_filter(scheme::AbstractCompactScheme) = StateFilter(scheme)
_state_filter(::Nothing) = StateFilter(interval=0)
_state_filter(x) =
    throw(ArgumentError("Numerics: filter must be a StateFilter, a compact filter " *
                        "scheme or nothing, got $(typeof(x))"))

"""
    PatchInterfaces(; flux = :ghost, rhs = :extended, divergence = nothing)
    PatchInterfaces(base::PatchInterfaces; keywords...)

How the flux divergence closes at a patch or level interface: a plane shared
by two same-level patches (`Execution(patch_grid = ...)`, tiles) or the face of
a refined level. It has nothing to do with a material interface between two
fluids. Without a patch or level interface these settings have no effect. The
second form copies `base` with the given fields replaced;
`Numerics(patch_interfaces = :closure)` is shorthand for
`PatchInterfaces(flux = :closure)`. The solver keywords, checkpoint records
and error messages name the three fields `interface_flux`, `interface_rhs`
and `interface_divergence`.

- `flux`: `:ghost` (default) differentiates the inviscid and molecular fluxes
  through the interface from ghost values with the interior stencil;
  `:closure` takes one-sided closure rows there. Choose `:closure` for
  shock-dominated runs and Float32 runs, where the ghost fluxes add 11 to 50%
  to the step without lowering the error, and wherever `:ghost` is not
  supported: a curvilinear or stretched grid, `rhs = :onesided`, or a user EOS
  at a refined level with molecular transport, each of which setup rejects
  under `:ghost` with an `ArgumentError`. See
  [Choose numerics](@ref).
- `rhs`: `:extended` (default) evaluates the gradient and divergence rows at
  an interface end from exchanged ghost data; `:onesided` closes them with
  one-sided rows, as at a boundary. `:onesided` requires `flux = :closure`, and
  is a workaround for initial data with a discontinuity within one node of a
  shared patch plane.
- `divergence`: `nothing` (default), or a scheme with the interior
  coefficients and element type of `deriv` whose closure rows replace the
  flux divergence's rows at interface ends only, such as
  `lele_d1_6(closures = :brady_livescu)`. Under `flux = :ghost` it affects
  only the artificial fluxes and the wall corrections. Experimental and
  Float64 only.
"""
struct PatchInterfaces
    flux::Symbol
    rhs::Symbol
    divergence::Union{Nothing,AbstractCompactScheme}
end

PatchInterfaces(; flux::Symbol=:ghost, rhs::Symbol=:extended,
                divergence::Union{Nothing,AbstractCompactScheme}=nothing) =
    PatchInterfaces(flux, rhs, divergence)
PatchInterfaces(base::PatchInterfaces; kw...) = _with(base, kw)

_patch_interfaces(interfaces::PatchInterfaces) = interfaces
_patch_interfaces(flux::Symbol) = PatchInterfaces(flux=flux)
_patch_interfaces(x) =
    throw(ArgumentError("Numerics: patch_interfaces must be a PatchInterfaces or a " *
                        "Symbol, got $(typeof(x))"))

"""
    Execution(; dims = nothing, comm = MPI.COMM_WORLD, backend = CPUBackend(),
              precision = nothing, patch_grid = (1, 1, 1))
    Execution(base::Execution; keywords...)

Where and in what arithmetic a [`Numerics`](@ref) discretization runs: the
process grid, the communicator, the storage backend, the floating-point type
and the same-level patch layout. The second form copies `base` with the given
fields replaced.

- `dims`: MPI process-grid dimensions. `nothing` lets MPI distribute ranks
  over resolved directions. An explicit tuple must have product equal to the
  communicator size and must contain `1` in every collapsed direction.
- `comm`: the communicator the solver spans. Every rank of it must call
  [`setup`](@ref) with the same `Problem` and `Numerics`; a split
  communicator lets two independent solvers share one job.
- `backend`: storage and execution backend. `CompactLES.CPUBackend()` is the
  default; wrap a `CUDABackend()` or `ROCBackend()` in `DeviceBackend` after
  loading the matching GPU package.
- `precision`: `Float32` or `Float64`, or `nothing` (the default). When set,
  the EOS, the transport model, `art`, `deriv`, the filter scheme and the
  interface divergence scheme are converted to this type, and the solver
  stores and computes in it. Left at `nothing`, those components must all
  carry one type, which the solver adopts; components of different types
  raise an `ArgumentError` at setup that names the type of each.
- `patch_grid`: same-level slab patches per direction, at most one
  direction above one. It excludes an explicit `dims` and refinement; see
  [Supported combinations](@ref) for the rest of its scope.
"""
struct Execution
    dims::Union{Nothing,NTuple{3,Int}}
    comm::MPI.Comm
    backend::AbstractBackend
    precision::Union{Nothing,Type{<:AbstractFloat}}
    patch_grid::NTuple{3,Int}
end

Execution(; dims=nothing, comm::MPI.Comm=MPI.COMM_WORLD,
          backend::AbstractBackend=CPUBackend(), precision=nothing,
          patch_grid=(1, 1, 1)) =
    Execution(dims, comm, backend, precision, patch_grid)
Execution(base::Execution; kw...) = _with(base, kw)

"""
    Numerics(; n_global, deriv=lele_d1_6(), filter=StateFilter(),
             art=ArtificialProperties(), cfl=0.5, control=nothing,
             patch_interfaces=PatchInterfaces(), execution=Execution(), amr=nothing,
             polar_truncation=0.0, stretch=(nothing, nothing, nothing),
             implicit=nothing)
    Numerics(base::Numerics; keywords...)
    Numerics(preset::NamedTuple; keywords...)

Grid, scheme, timestep, and decomposition choices used to realize a
[`Problem`](@ref). The second form copies `base` with the given keywords
replaced. The third starts from a preset of [`Presets`](@ref), whose
keywords the ones given after it override.

# Keywords

- `n_global`: required three-tuple giving the global point count in each
  coordinate direction. A count of one collapses that direction: it has no
  derivative, halo, or decomposition.
- `deriv`: compact first-derivative scheme. Default: [`lele_d1_6()`](@ref);
  [`lele_d1_8()`](@ref) and [`lele_d1_10()`](@ref) are the higher-order
  operators.
- `filter`: the state filter, a [`StateFilter`](@ref). A compact scheme `s`
  is shorthand for `StateFilter(s)`, and `nothing` for
  `StateFilter(interval = 0)`, which filters nothing. Default: `StateFilter()`,
  [`compact_filter(0.47)`](@ref) every step, relaxed below a
  CFL of 0.35.
- `art`: artificial-property coefficients. Default: [`ArtificialProperties()`](@ref).
- `cfl`: multiplier used by [`compute_dt`](@ref). Default: `0.5`. Strong shocks
  can require a lower startup value.
- `control`: timestep prediction, failure floors, and retry policy, a
  [`StepControl`](@ref). Left at `nothing`, the default, [`setup`](@ref)
  takes `StepControl()`, or `StepControl(retries = 4)` when the problem has
  an [`OriginBC`](@ref): a strong shock at the spherical origin can lose
  positivity at the default CFL, a converging one from a singular start or
  above a CFL of 0.5 and a blast above 0.2 to 0.3, and the retries recover
  it at a lowered CFL. A `StepControl` given here is used as it is.
- `patch_interfaces`: how the flux divergence closes at a patch or level
  interface, a [`PatchInterfaces`](@ref); a `Symbol` is shorthand for
  `PatchInterfaces(flux = symbol)`. Without such an interface it has no
  effect.
- `execution`: the process grid, communicator, backend, precision and
  same-level patch layout, an [`Execution`](@ref).
- `amr`: `nothing`, or an [`AMR`](@ref) selecting adaptive refinement.
- `polar_truncation`: azimuthal mode truncation near a cylindrical axis with
  resolved θ. `0.0`, the default, disables it. A value κ ≥ 1 projects each
  ring of fixed r and z, once per step, onto its azimuthal Fourier modes
  m ≤ M = max(2, ⌊πr/(κΔr)⌋), and the radial and azimuthal momenta onto
  m ≤ M + 1; the timestep charges the θ direction at the spacing of the
  highest mode kept, about κΔr, instead of rΔθ. A scalar that is smooth
  through the axis carries mode m only as r^m, and the radial and azimuthal
  velocity components carry it as r^(m−1), so the modes removed are ones the
  inner rings cannot resolve, and without the truncation the step is sized by
  their spacing. Use it when [`dt_report`](@ref) names θ at the innermost
  rings as the limiting direction; a larger κ removes more modes and allows a
  longer step. Modes 0 to 2 are always kept, so the ring sums of the
  conserved variables and a uniform freestream are unchanged. The cost is one
  projection per step over the rings below the threshold radius, small against
  a filter pass; when θ is split across ranks, the ranks sharing a ring also
  gather it once per step. Requires `CylindricalMetric`, θ periodic over 2π,
  an unstretched radial dimension, a single patch without refinement, and the
  host backend.
- `stretch`: one entry per direction, each either `nothing` for a uniform grid
  or a [`Stretch`](@ref). A mapping must span the corresponding `Problem.domain`
  interval and can be used only in a nonperiodic, non-folded direction.
- `implicit`: `nothing`, the default, for the low-storage explicit
  integrator, or an [`ImplicitConduction`](@ref), which integrates the
  molecular heat conduction implicitly and every other term explicitly, so
  that the conductive rate no longer limits the step.

`n_halo`, the halo layers on each side of a resolved local block, is also
accepted. It is 4, which covers every stencil the package builds, and is not a
tuning parameter.

Compact plans impose a scheme-dependent minimum rank-local extent. With the
defaults, each resolved local extent needs at least nine points because the
filter is the binding scheme. Setup checks every block of the process grid
against this minimum before building a plan and raises an `ArgumentError`
naming the dimension, the scheme and the extent required; reduce the
decomposition in that direction or increase `n_global`.

`cfl` must be finite and positive, the filter's `interval` nonnegative, and
its `cfl` finite and nonnegative. These and the parameter ranges of `art`, the
transport model and the EOS are checked when the solver is built.

# Deprecated keywords

The flat keywords of earlier versions are accepted with a deprecation warning
and folded into their groups: `filt`, `filter_interval`, `filter_cfl` and
`filter_weighting` into `filter` (as `scheme`, `interval`, `cfl` and
`weighting`); `interface_flux`, `interface_rhs` and `interface_divergence`
into `patch_interfaces` (as `flux`, `rhs` and `divergence`); and `dims`,
`comm`, `backend`, `precision` and `patch_grid` into `execution`. The flat
refinement keywords (`refine`, `subcycle`, `tile`, `regrid_interval`, the tag
thresholds and the rest) still select refinement, cannot be combined with
`amr`, and are written in `AMR(...)` instead; the
[AMR reference](@ref "Adaptive mesh refinement") lists them.
"""
struct Numerics
    n_global::NTuple{3,Int}
    deriv::AbstractCompactScheme
    filter::StateFilter
    art::ArtificialProperties
    cfl::Float64
    control::Union{Nothing,StepControl}
    patch_interfaces::PatchInterfaces
    execution::Execution
    amr::Union{Nothing,AMR}
    polar_truncation::Float64
    stretch::NTuple{3,Union{Nothing,Stretch}}
    implicit::Union{Nothing,ImplicitConduction}
    n_halo::Int
    legacy_amr::NamedTuple
end

# The flat refinement keywords `Numerics` accepted before `AMR`, with their
# defaults. They still reach the solver unchanged when `amr` is not given.
const _AMR_LEGACY_DEFAULTS = (
    refine=nothing, level_restriction=:inject, level_interpolation_order=nothing,
    subcycle=false, regrid_interval=0, tag_threshold=0.02, tag_buffer=4,
    tag_sensor_threshold=0.0, tag_gradient_threshold=0.0,
    tag_vorticity_threshold=0.0, tag_predicate=nothing, untag_ratio=2.0,
    tile_lifetime=1, tile=0, rebalance=0.0, rebalance_persist=2, max_levels=nothing,
)

# The flat keywords folded into a group: keyword => (group, field).
const _NUMERICS_FLAT = (
    filt=(:filter, :scheme), filter_interval=(:filter, :interval),
    filter_cfl=(:filter, :cfl), filter_weighting=(:filter, :weighting),
    interface_flux=(:patch_interfaces, :flux),
    interface_rhs=(:patch_interfaces, :rhs),
    interface_divergence=(:patch_interfaces, :divergence),
    dims=(:execution, :dims), comm=(:execution, :comm),
    backend=(:execution, :backend), precision=(:execution, :precision),
    patch_grid=(:execution, :patch_grid),
)

const _NUMERICS_GROUP_TYPES = (filter=:StateFilter, patch_interfaces=:PatchInterfaces,
                               execution=:Execution)

Numerics(; kw...) = _numerics(_AMR_LEGACY_DEFAULTS; kw...)

# A preset from `Presets` is a NamedTuple of keywords, overridden by those
# given after it.
Numerics(preset::NamedTuple; kw...) = Numerics(; merge(preset, values(kw))...)

function Numerics(base::Numerics; kw...)
    fields = (:n_global, :deriv, :filter, :art, :cfl, :control, :patch_interfaces,
              :execution, :amr, :polar_truncation, :stretch, :implicit, :n_halo)
    inherited = NamedTuple{fields}(map(f -> getfield(base, f), fields))
    return _numerics(base.legacy_amr; merge(inherited, values(kw))...)
end

function _numerics(legacy_amr::NamedTuple; n_global, deriv=lele_d1_6(),
                   filter=StateFilter(), art=ArtificialProperties(), cfl=0.5,
                   control=nothing, patch_interfaces=PatchInterfaces(),
                   execution=Execution(), amr=nothing, polar_truncation=0.0,
                   stretch=(nothing, nothing, nothing), implicit=nothing, n_halo=4,
                   flat...)
    execution isa Execution ||
        throw(ArgumentError("Numerics: execution must be an Execution, got " *
                            "$(typeof(execution))"))
    groups = Dict{Symbol,Any}(:filter => _state_filter(filter),
                              :patch_interfaces => _patch_interfaces(patch_interfaces),
                              :execution => execution)
    folded = String[]
    refinement = Symbol[]
    for (k, v) in pairs(flat)
        if haskey(_NUMERICS_FLAT, k)
            group, field = _NUMERICS_FLAT[k]
            groups[group] = _with(groups[group], NamedTuple{(field,)}((v,)))
            type = _NUMERICS_GROUP_TYPES[group]
            push!(folded, "`$k` is the `$field` of `$group = $type(...)`")
        elseif haskey(_AMR_LEGACY_DEFAULTS, k)
            legacy_amr = merge(legacy_amr, NamedTuple{(k,)}((v,)))
            push!(refinement, k)
        else
            throw(ArgumentError("Numerics has no keyword `$k`"))
        end
    end
    isempty(folded) ||
        Base.depwarn("Numerics: flat keywords are deprecated; " * join(folded, ", ") *
                     ".", :Numerics; force=true)
    isempty(refinement) ||
        Base.depwarn("Numerics: the flat refinement keywords " *
                     join(("`$k`" for k in refinement), ", ") * " are deprecated; " *
                     "write them in `amr = AMR(...)`.", :Numerics; force=true)
    return Numerics(n_global, deriv, groups[:filter], art, cfl, control,
                    groups[:patch_interfaces], groups[:execution], amr,
                    polar_truncation, stretch, implicit, n_halo, legacy_amr)
end

_legacy_amr_keywords(num::Numerics) = num.legacy_amr

# The step policy of a problem whose deck gives none. A strong shock at a
# spherical origin is limited to a lower CFL than at a wall or an axis, a
# converging one by an excursion of the origin cell at every resolution, and
# rollback with a lowered CFL recovers it; no other geometry needs the
# retries or the savepoint they keep.
_default_control(prob::Problem) =
    any(pair -> any(bc -> bc isa OriginBC, pair), prob.bcs) ? StepControl(retries=4) :
                                                               StepControl()

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

Collective over `num.execution.comm` (`MPI.COMM_WORLD` by default): the decomposition
is built with `MPI.Cart_create` and its sub-communicators (on more than one
rank; a single rank borrows that communicator itself), so every rank of
that communicator must call `setup` with the same `prob` and `num`. A split
communicator lets two independent solvers share one job.
"""
function setup(prob::Problem, num::Numerics)
    solver, Q = _setup(prob, num)
    # The deck as given, before the default control is filled in, so a phase
    # change to other boundary conditions derives that default again.
    setfield!(solver, :inputs, (problem=prob, numerics=num))
    return solver, Q
end

function _setup(prob::Problem, num::Numerics)
    num.control === nothing && (num = Numerics(num; control=_default_control(prob)))
    legacy = _legacy_amr_keywords(num)
    if num.amr !== nothing
        num.implicit === nothing ||
            throw(ArgumentError("implicit conduction runs on a single patch without " *
                                "refinement; remove amr or implicit"))
        legacy == _AMR_LEGACY_DEFAULTS ||
            throw(ArgumentError("use amr=AMR(...) or the legacy refinement keywords, " *
                                "not both"))
        return _setup_amr(prob, num, num.amr)
    end
    return _setup_with_amr_keywords(prob, num, legacy)
end

function _setup_with_amr_keywords(prob::Problem, num::Numerics, kw::NamedTuple;
                                  seed_only::Bool=false)
    origin = ntuple(d -> prob.domain[d][1], 3)
    L_domain = ntuple(d -> prob.domain[d][2] - prob.domain[d][1], 3)
    all(d -> all(isfinite, prob.domain[d]), 1:3) ||
        throw(ArgumentError("Problem domain endpoints must be finite, got " *
                            "$(prob.domain)"))
    all(>(0), L_domain) ||
        throw(ArgumentError("Problem domain extents must be positive, got " *
                            "$(prob.domain)"))
    for d in 1:3
        st = num.stretch[d]
        st === nothing && continue
        isapprox(st.x(0.0), prob.domain[d][1]; atol=1e-10 * L_domain[d]) &&
        isapprox(st.x(1.0), prob.domain[d][2]; atol=1e-10 * L_domain[d]) ||
            error("stretch mapping for dim $d does not span the domain: " *
                  "x(0) = $(st.x(0.0)), x(1) = $(st.x(1.0)), " *
                  "domain = $(prob.domain[d])")
    end
    solver = Solver(; n_global=num.n_global, L_domain=L_domain, bcs=prob.bcs,
               eos=prob.eos, transport=prob.transport, art=num.art,
               metric=prob.metric, stretch=num.stretch, sources=prob.sources,
               origin=origin,
               deriv=num.deriv, filt=num.filter.scheme,
               cfl=num.cfl, control=num.control,
               filter_interval=num.filter.interval,
               filter_cfl=num.filter.cfl,
               filter_weighting=num.filter.weighting,
               polar_truncation=num.polar_truncation, implicit=num.implicit,
               dims=num.execution.dims, n_halo=num.n_halo, comm=num.execution.comm,
               patch_grid=num.execution.patch_grid, backend=num.execution.backend,
               interface_rhs=num.patch_interfaces.rhs,
               interface_divergence=num.patch_interfaces.divergence,
               interface_flux=num.patch_interfaces.flux,
               precision=num.execution.precision, kw...)
    Q = allocate_state(solver)
    if seed_only
        # The temporary fine cover exists only to plan the initial tagging.
        # Do not evaluate the user's spacing-aware IC on nodes that may never
        # belong to the selected hierarchy. Validate the coarse state before
        # any sensor divides by its density or recovers thermodynamics.
        root = PatchSolver(solver, first(getfield(solver, :patches)))
        initialize!(root, Q[1], prob.ic)
        report = _cpu_storage(Q[1]) ?
                 _reduce_state_report(solver,
                                      _local_state_report(root, Q[1],
                                                          num.control.species_band)) :
                 StateReport()
        failure = check_validity(num.control, report, "the initial coarse state",
                                 solver.step, solver.t, solver.dt_prev, solver.cfl)
        failure === nothing || throw(failure)
        return solver, Q
    end
    initialize!(solver, Q, prob.ic)
    validate_state!(solver, Q; control=num.control, stage="the initial state")
    return solver, Q
end
