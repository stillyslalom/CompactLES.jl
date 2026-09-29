# The step of a solver built with `ImplicitConduction`: the six stages of the
# additive pair of imex.jl, the outer iteration of each implicit stage, and
# the step rule that limits the next step by the conduction's accuracy. A
# single patch on host storage (`_validate_implicit`).
#
# A step of `dt` from `run!` is one attempt at the whole interval. An attempt
# the step rule rejects (the embedded estimate above its tolerance, or the
# temperature change above twice its target) is discarded and the interval
# is covered by equal substeps no longer than the limit the rule gives, each
# tested the same way; the state returned is
# the one at t + dt whatever the subdivision, so the endpoint clip and the
# landing on scheduled instants in `run!` hold unchanged. A stage that does
# not converge ends the step with `SolverFailure(:implicit_solve)`, which
# `run!` hands to the rollback of `StepControl`. The failed step times the
# rollback's `cfl_backoff` then caps every later step of the solver, as the
# lowered CFL persists, since the lowered CFL alone does not shorten a step
# the accuracy limit or the endpoint sets.

# The solver types built with `ImplicitConduction`. The parameters carry
# `Solver`'s own bounds: written without them, the alias is no subtype of
# `Solver` as a signature and the methods below rank below the generic ones.
const ImexSolver =
    Solver{T,Eq,E,<:WithoutConduction{T}} where {T,Eq<:EquationSet,E<:EOS}

# A pair of attempts rejected in a row halves the limit at least, so this
# bounds the subdivision of one interval at far below the dt floors.
const IMEX_MAX_REJECTIONS = 40

_zero_molecular_diffusion(tr::WithoutConduction) = _zero_molecular_diffusion(tr.transport)

function _implicit_step_limit(solver::ImexSolver, dt)
    integ = getfield(solver, :implicit)
    return min(dt, oftype(dt, integ.dt_limit), oftype(dt, integ.dt_cap))
end

function step!(solver::ImexSolver, Q::ConservedState, dQ, du, dt, prepared::Bool)
    failure = _imex_advance!(solver, Q, dt, prepared, solver.control)
    failure === nothing || throw(failure)
    return Q
end

_run_step!(solver::ImexSolver, Q, workspace, dt, prepared, control) =
    _imex_advance!(solver, Q, dt, prepared, control)

# Whether `flag` holds on every rank of `comm`.
_on_every_rank(flag::Bool, comm) = MPI.Allreduce(Int(flag), min, comm) == 1

_imex_failure(solver, dt, detail) =
    SolverFailure(:implicit_solve, solver.step, Float64(solver.t), Float64(dt),
                  Float64(solver.cfl), detail)

# Advance `Q` from `solver.t` to `solver.t + dt`, in one attempt or in the
# substeps the step rule calls for. Returns `nothing` or the failure.
function _imex_advance!(solver, Q, dt, prepared::Bool, control)
    integ = getfield(solver, :implicit)
    T = typeof(solver.t)
    t_now = solver.t
    remaining = T(dt)
    h = remaining
    left = 1
    rejections = 0
    while left > 0
        status = _imex_attempt!(solver, integ, Q, t_now, h, prepared)
        prepared = false
        if status isa SolverFailure
            integ.dt_cap = T(control.cfl_backoff) * min(integ.dt_cap, h)
            return status
        end
        if status
            integ.accepted += 1
            t_now += h
            remaining -= h
            left -= 1
            # Resubdivide what is left when the limit has fallen below the
            # pieces already planned.
            left > 0 && integ.dt_limit < h || continue
        else
            integ.rejected += 1
            rejections += 1
            rejections > IMEX_MAX_REJECTIONS &&
                return _imex_failure(solver, h, "the step rule rejected " *
                                     "$rejections attempts on one interval")
        end
        piece = min(remaining, integ.dt_limit, integ.dt_cap)
        left = max(1, ceil(Int, remaining / piece))
        h = remaining / left
    end
    return nothing
end

# One attempt at [t0, t0 + h]: the six stages, the combination and the step
# rule. Returns `true` when accepted, `false` when rejected (with `Q`
# restored), or a `SolverFailure`.
function _imex_attempt!(solver, integ::ImexIntegrator{T}, Q, t0, h,
                        prepared::Bool) where {T}
    copyto!(integ.base, Q)
    γh = integ.gamma * h
    for i in 1:ARK436_STAGES
        solver.tstage = t0 + integ.nodes[i] * h
        if i > 1
            _stage_explicit_part!(Q, integ, solver, i, h)
            failure = _solve_implicit_stage!(solver, integ, Q, γh)
            failure === nothing || return failure
            _stage_tendency!(integ, solver, Q, i, γh)
        end
        first = prepared && i == 1
        first || apply_bcs!(solver, Q)
        compute_rhs!(solver, Q, integ.explicit[i], first)
        if i == 1
            # The primitives are those of Qⁿ: the tendency of the explicit
            # first stage, and the thermal energy the step rule measures by.
            valid = _refresh_conduction!(integ, solver)
            _on_every_rank(valid, solver.comm) ||
                return _imex_failure(solver, h, "the temperature or the heat " *
                                     "capacity entering the step is not finite and positive")
            diffusion_operator!(integ.implicit[1], integ.stage, solver.T_ion, integ.kappa)
            _thermal_scale!(integ, solver)
        end
    end
    measure = _combine_stages!(Q, integ, solver, h)
    measure = MPI.Allreduce(measure, max, solver.comm)
    integ.measure = measure
    isfinite(measure) ||
        return _imex_failure(solver, h, "the step rule's measure is not finite")
    accept = _step_rule!(integ, measure, h)
    if !accept
        copyto!(Q, integ.base)
        return false
    end
    solver.tstage = t0 + h
    apply_bcs!(solver, Q)
    _validate_transport_state!(solver, Q)
    return true
end

# The next step's accuracy limit from this step's measure, and whether this
# step stands. Both rules read a fraction of the temperature at the start of
# the step, divided by its target, so a measure of one is the target met.
function _step_rule!(integ::ImexIntegrator{T}, measure, h) where {T}
    rule = integ.settings.step_rule
    if rule === :none
        integ.dt_limit = T(Inf)
        return true
    end
    grow, shrink = T(5), T(1) / 5
    if rule === :error
        # The embedded solution is third order, so the estimate falls as h⁴.
        factor = measure == 0 ? grow :
                 clamp(T(0.9) * T(measure)^(-T(1) / 4), shrink, grow)
        integ.dt_limit = h * factor
        return measure <= 1
    end
    # :temperature. The change is first order in h. A step that changed the
    # temperature by more than twice the target, as the first step of a run
    # can, having no previous step to size it, is taken again shorter.
    factor = measure == 0 ? grow : clamp(1 / T(measure), shrink, grow)
    integ.dt_limit = h * factor
    return measure <= 2
end

# Q = Qⁿ + h Σ_{j<i} (a^E_ij F_E,j + a^I_ij F_I,j) over the interior, with
# its energy kept as R, the explicit part of stage i, and replaced by the
# first trial of the outer iteration: the kinetic energy of stage i with the
# specific internal energy of stage i − 1, which `Q` holds on entry. R itself
# carries the explicit evaluation of the conduction at the earlier stages,
# which at a large step is far from the stage solution and can be negative.
function _stage_explicit_part!(Q, integ::ImexIntegrator{T}, solver, i, h) where {T}
    decomp = solver.decomp
    eq = solver.equations
    ie = eq.i_energy
    Q0, FE, FI = integ.base, integ.explicit, integ.implicit
    ae, ai = integ.explicit_matrix, integ.implicit_matrix
    o1, o2, o3 = decomp.n_halo_d
    nx, ny, nz = decomp.n_local
    ht = T(h)
    @inbounds for I in interior(decomp)
        ρ, kinetic = _density_and_kinetic(Q, I, eq)
        integ.guess[I] = (Q[I, ie] - kinetic) / ρ
    end
    for c in 1:eq.n_cons
        @threaded nx*ny*nz for jk in outer_indices(ny, nz)
            j, k = Tuple(jk)
            @inbounds for ii in 1:nx
                I = CartesianIndex(ii + o1, j + o2, k + o3)
                acc = zero(T)
                for m in 1:i-1
                    acc += ae[i, m] * FE[m][I, c]
                end
                if c == ie
                    for m in 1:i-1
                        acc += ai[i, m] * FI[m][I]
                    end
                end
                Q[I, c] = Q0[I, c] + ht * acc
            end
        end
    end
    @inbounds for I in interior(decomp)
        integ.known[I] = Q[I, ie]
        ρ, kinetic = _density_and_kinetic(Q, I, eq)
        Q[I, ie] = kinetic + ρ * integ.guess[I]
    end
    return Q
end

@inline function _density_and_kinetic(Q, I, eq)
    ρ = zero(eltype(Q))
    for sp in 1:eq.n_species
        ρ += Q[I, sp]
    end
    m1, m2, m3 = eq.i_mom
    return ρ, (Q[I, m1]^2 + Q[I, m2]^2 + Q[I, m3]^2) / (2ρ)
end

# The coefficient refresh of the conduction component: κ(T) from the
# solver's transport model and the capacity m = ρ c_v, at the primitives the
# solver holds. Returns whether this rank's trial is valid: temperature and
# capacity finite and positive, κ finite and nonnegative.
function _refresh_conduction!(integ::ImexIntegrator{T}, solver) where {T}
    tr = solver.transport.transport
    eos = solver.eos
    T_ion, rho, p, cp = solver.T_ion, solver.rho, solver.p, solver.cp_mix
    Y = solver.field_tuples.Y
    floor = temperature_floor(T)
    valid = true
    @inbounds for I in interior(solver.decomp)
        κ = transport_at(tr, eos, T_ion, rho, cp, Y, I).kappa
        m = rho[I] * mixture_cv(eos, rho[I], p[I], T_ion[I], cp[I])
        integ.kappa[I] = κ
        integ.capacity[I] = m
        valid &= isfinite(κ) & (κ >= 0) & isfinite(m) & (m > 0) &
                 isfinite(T_ion[I]) & (T_ion[I] > floor)
    end
    return valid
end

# ρ c_v T at the start of the step, the scale of both step rules.
function _thermal_scale!(integ::ImexIntegrator, solver)
    T_ion = solver.T_ion
    @inbounds for I in interior(solver.decomp)
        integ.thermal[I] = integ.capacity[I] * T_ion[I]
    end
    return integ
end

# The outer iteration of an implicit stage: E, the stage energy held in Q, is
# driven to G(E) = E − R − γh L_κ(T) T = 0 by Newton steps in T with κ lagged,
# (m − γh L_κ) δT = −G, E ← E + m δT. The iteration stops when the residual
# falls below `rtol` times the thermal energy; each linear solve is asked for
# half of that, so a κ independent of T and an energy linear in T stop after
# one. Every decision reads reduced values, so every rank takes it.
function _solve_implicit_stage!(solver, integ::ImexIntegrator{T}, Q, γh) where {T}
    settings = integ.settings
    decomp = solver.decomp
    ie = solver.equations.i_energy
    stage = integ.stage
    rtol = T(settings.rtol)
    sums = zeros(T, 2)
    for k in 0:settings.max_outer
        primitives!(solver, Q)
        valid = _refresh_conduction!(integ, solver)
        _on_every_rank(valid, solver.comm) ||
            return _imex_failure(solver, γh / integ.gamma, "an implicit trial state " *
                                 "has a temperature or heat capacity that is not " *
                                 "finite and positive")
        diffusion_operator!(integ.tendency, stage, solver.T_ion, integ.kappa)
        integ.outer += 1
        fill!(sums, zero(T))
        @inbounds for I in interior(decomp)
            G = Q[I, ie] - integ.known[I] - γh * integ.tendency[I]
            integ.defect[I] = -G
            w = stage.volume[I]
            e = integ.capacity[I] * solver.T_ion[I]
            sums[1] += w * G * G
            sums[2] += w * e * e
        end
        MPI.Allreduce!(sums, +, solver.comm)
        gnorm, enorm = sqrt(sums[1]), sqrt(sums[2])
        gnorm <= rtol * enorm && return nothing
        k == settings.max_outer &&
            return _imex_failure(solver, γh / integ.gamma, "an implicit stage did not " *
                                 "converge in $(settings.max_outer) outer iterations " *
                                 "(residual $(gnorm / enorm) of the thermal energy)")
        fill!(integ.increment, zero(T))
        lin = clamp(rtol * enorm / (2 * gnorm), eps(T), T(1) / 10)
        res = solve_stage!(integ.increment, stage, integ.defect, integ.kappa, γh;
                           rtol=lin, maxiter=settings.max_iterations,
                           capacity=integ.capacity)
        integ.solves += 1
        integ.krylov += res.iterations
        res.converged ||
            return _imex_failure(solver, γh / integ.gamma, "an implicit linear solve " *
                                 "did not converge in $(settings.max_iterations) " *
                                 "iterations (residual $(res.residual))")
        @inbounds for I in interior(decomp)
            Q[I, ie] += integ.capacity[I] * integ.increment[I]
        end
    end
    return nothing
end

# F_I of stage i from the converged stage equation, (E_i − R_i)/(γh), rather
# than by applying the operator again: the stage residual then enters the
# step multiplied by h, where the operator would multiply it by the stiff
# rate.
function _stage_tendency!(integ::ImexIntegrator{T}, solver, Q, i, γh) where {T}
    ie = solver.equations.i_energy
    FI = integ.implicit[i]
    inv = one(T) / T(γh)
    @inbounds for I in interior(solver.decomp)
        FI[I] = (Q[I, ie] - integ.known[I]) * inv
    end
    return FI
end

# Qⁿ⁺¹ = Qⁿ + h Σ_j b_j (F_E,j + F_I,j) over the interior, and this rank's
# largest step-rule measure: under `:error` the embedded estimate of the
# implicit tendency, h Σ_j (b_j − b̂_j) F_I,j, and under `:temperature` its
# whole contribution h Σ_j b_j F_I,j, each over ρ c_v T at the start of the
# step and the rule's target.
function _combine_stages!(Q, integ::ImexIntegrator{T}, solver, h) where {T}
    decomp = solver.decomp
    eq = solver.equations
    ie = eq.i_energy
    Q0, FE, FI = integ.base, integ.explicit, integ.implicit
    b, bh = integ.weights, integ.embedded
    o1, o2, o3 = decomp.n_halo_d
    nx, ny, nz = decomp.n_local
    ht = T(h)
    s = ARK436_STAGES
    for c in 1:eq.n_cons
        @threaded nx*ny*nz for jk in outer_indices(ny, nz)
            j, k = Tuple(jk)
            @inbounds for ii in 1:nx
                I = CartesianIndex(ii + o1, j + o2, k + o3)
                acc = zero(T)
                for m in 1:s
                    acc += b[m] * FE[m][I, c]
                end
                if c == ie
                    for m in 1:s
                        acc += b[m] * FI[m][I]
                    end
                end
                Q[I, c] = Q0[I, c] + ht * acc
            end
        end
    end
    rule = integ.settings.step_rule
    rule === :none && return zero(T)
    target = T(rule === :error ? integ.settings.tolerance : integ.settings.target_change)
    measure = zero(T)
    @inbounds for I in interior(decomp)
        acc = zero(T)
        for m in 1:s
            w = rule === :error ? b[m] - bh[m] : b[m]
            acc += w * FI[m][I]
        end
        measure = max(measure, abs(ht * acc) / integ.thermal[I])
    end
    return measure / target
end
