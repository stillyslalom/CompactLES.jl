# Additive Runge–Kutta integration of the molecular conduction: the pair
# ARK4(3)6L[2]SA of Kennedy and Carpenter (Appl. Numer. Math. 44, 2003),
# whose explicit half advances the right-hand side without the conduction
# and whose ESDIRK half advances the conduction through the staggered stage
# of implicit.jl. The step drivers are in imex_step.jl; this file holds the
# tableau, the settings, the transport of the explicit half and the storage.
#
# Stage equations. With F_E the explicit right-hand side and F_I the implicit
# tendency, stage i of a step of size Δt from Qⁿ is
#
#     Q_i = Qⁿ + Δt Σ_{j<i} (a^E_ij F_E(Q_j) + a^I_ij F_I(Q_j)) + γΔt F_I(Q_i),
#
# γ = a^I_ii = 1/4 for i ≥ 2, and the first stage is Qⁿ itself. The step is
# Qⁿ⁺¹ = Qⁿ + Δt Σ_j b_j (F_E(Q_j) + F_I(Q_j)), with one weight vector for
# both halves; b̂ gives the third-order embedded solution.
#
# The implicit components. A component is a term of the energy equation that
# the implicit half carries. It contributes, at a trial state,
#
#   1. its tendency to the one stage residual
#      G(E) = E − R − γΔt Σ_c F_c(T(E)), with R the explicit part of the stage;
#   2. its linearization to the one linear system of the outer iteration,
#      (m − γΔt Σ_c ∂F_c/∂T) δT = −G, m = ρ c_v, the coefficients lagged at
#      the trial state;
#   3. a coefficient refresh from the trial state's primitives, made for
#      every component before any of them is applied, once per outer
#      iteration;
#   4. a validity test of the trial state.
#
# The first trial of a stage is the previous stage's specific internal
# energy at the stage's own density and momentum.
#
# All components enter one residual and one linear solve, so no component is
# updated before another reads the state. A trial whose temperature or
# capacity is not finite and positive, or whose linear solve does not reach
# its tolerance, invalidates the stage; the stage then fails on every rank,
# since both tests reduce over the run, and the step reports a
# `SolverFailure(:implicit_solve)` to `run!`'s rollback. The molecular
# conduction is the one component: its tendency is the staggered operator
# `J⁻¹ ∂(C κ ∂T)` of `DiffusionStage`, its linearization that operator on
# the lagged κ, which the stage solves with the capacity m, and its refresh
# the evaluation of κ(T) by the solver's transport model. A pointwise
# exchange term joins the linear system as a diagonal block of the
# preconditioner (reference/IMPLICIT.md, "The linear solve").
#
# The explicit half carries every other term, the artificial conductivity
# κ* included: the solver's transport is wrapped in `WithoutConduction`,
# which returns the molecular coefficients with κ = 0, so the explicit fluxes
# and the explicit diffusive rate of `max_rate` omit the molecular
# conduction without a branch in the right-hand side.

# --- The tableau -------------------------------------------------------------
#
# The coefficients as published; a^I is exact and a^E is given to thirteen
# digits, so the explicit order conditions hold to about 1e-26 in exact
# arithmetic (the serial suite checks both).

# The strictly lower (explicit) or lower (implicit) triangle of a 6 × 6
# tableau from its rows, each a tuple of the entries left of the diagonal
# and, for the implicit half, the diagonal.
function _tableau(rows...)
    A = zeros(Rational{Int64}, 6, 6)
    for (i, row) in enumerate(rows), (j, a) in enumerate(row)
        A[i, j] = a
    end
    return A
end

const ARK436_EXPLICIT = _tableau(
    (),
    (1//2,),
    (13861//62500, 6889//62500),
    (-116923316275//2393684061468, -2731218467317//15368042101831,
     9408046702089//11113171139209),
    (-451086348788//2902428689909, -2682348792572//7519795681897,
     12662868775082//11960479115383, 3355817975965//11060851509271),
    (647845179188//3216320057751, 73281519250//8382639484533,
     552539513391//3454668386233, 3354512671639//8306763924573, 4040//17871))

const ARK436_IMPLICIT = _tableau(
    (),
    (1//4, 1//4),
    (8611//62500, -1743//31250, 1//4),
    (5012029//34652500, -654441//2922500, 174375//388108, 1//4),
    (15267082809//155376265600, -71443401//120774400, 730878875//902184768,
     2285395//8070912, 1//4),
    (82889//524892, 0, 15625//83664, 69875//102672, -2260//8211, 1//4))

const ARK436_WEIGHTS = Rational{Int64}[82889//524892, 0, 15625//83664, 69875//102672,
                                       -2260//8211, 1//4]

const ARK436_EMBEDDED = Rational{Int64}[4586570599//29645900160, 0,
                                        178811875//945068544, 814220225//1159782912,
                                        -3700637//11593932, 61727//225920]

const ARK436_NODES = Rational{Int64}[0, 1//2, 83//250, 31//50, 17//20, 1]

const ARK436_STAGES = 6

# --- Settings ------------------------------------------------------------------

"""
    ImplicitConduction(; step_rule=:error, tolerance=1e-3, target_change=0.05,
                       rtol=1e-8, max_iterations=50, max_outer=10)

Moves the molecular heat conduction into the implicit half of the additive
Runge–Kutta pair ARK4(3)6L[2]SA (Kennedy and Carpenter 2003), selected by
passing it as the `implicit` keyword of [`Numerics`](@ref) or
[`Solver`](@ref). Every other term, the artificial conductivity included,
stays in the explicit half. The conduction is discretized by the staggered
conservative operator and solved at each implicit stage by conjugate
gradients preconditioned with a multigrid cycle (GMRES at a cylindrical axis
or spherical poles), so the conductive rate no longer limits the step; the
step is the acoustic and advective limit of `cfl`, further limited by the
accuracy rule below.

Use it when the conductive rate `κ/(ρ c_v h²)` exceeds the acoustic rate
`c/h` by an order of magnitude or more. The integrator takes six right-hand
side evaluations per step against the default integrator's five, holds seven
copies of the state beside it, and adds one or more linear solves per stage;
the explicit half's stability region differs from the default integrator's,
so its `cfl` limit is not the same. Requires a single patch on host storage
without refinement, and adiabatic walls (slip or no-slip), symmetry planes,
coordinate folds or periodic ends.

- `step_rule`: how the conduction's accuracy limits the step. `:error` sizes
  it from the pair's embedded error estimate of the implicit tendency, as a
  relative temperature error, rejecting and subdividing a step whose error
  exceeds `tolerance`. `:temperature` sizes it so that the conduction
  changes the temperature by the fraction `target_change` per step,
  rejecting a step that changed it by more than twice that. `:none` takes the
  explicit limit alone.
- `rtol`: the stage residual at which an outer iteration stops, relative to
  the thermal energy `ρ c_v T`. A linear κ and an energy linear in T converge
  in one linear solve.
- `max_iterations`, `max_outer`: the Krylov iterations of one linear solve
  and the outer iterations of one stage. A stage that does not converge
  within them fails with `SolverFailure(:implicit_solve)`, which
  `StepControl(retries = ...)` recovers by rolling back and lowering the step.
"""
struct ImplicitConduction
    step_rule::Symbol
    tolerance::Float64
    target_change::Float64
    rtol::Float64
    max_iterations::Int
    max_outer::Int
    function ImplicitConduction(step_rule, tolerance, target_change, rtol,
                                max_iterations, max_outer)
        step_rule in (:error, :temperature, :none) ||
            throw(ArgumentError("ImplicitConduction: step_rule must be :error, " *
                                ":temperature or :none, got :$step_rule"))
        for (name, v) in (("tolerance", tolerance), ("target_change", target_change),
                          ("rtol", rtol))
            isfinite(v) && v > 0 ||
                throw(ArgumentError("ImplicitConduction: $name must be finite " *
                                    "and positive, got $v"))
        end
        max_iterations >= 1 && max_outer >= 1 ||
            throw(ArgumentError("ImplicitConduction: max_iterations and max_outer " *
                                "must be at least 1"))
        new(step_rule, tolerance, target_change, rtol, max_iterations, max_outer)
    end
end

ImplicitConduction(; step_rule::Symbol=:error, tolerance::Real=1e-3,
                   target_change::Real=0.05, rtol::Real=1e-8, max_iterations::Int=50,
                   max_outer::Int=10) =
    ImplicitConduction(step_rule, Float64(tolerance), Float64(target_change),
                       Float64(rtol), max_iterations, max_outer)

# --- The transport of the explicit half -------------------------------------

"""
    WithoutConduction(transport)

The transport model of the explicit half under [`ImplicitConduction`](@ref):
the viscosity and the species diffusivities of `transport`, with the thermal
conductivity returned as zero. The implicit half evaluates the conductivity
from `transport` itself.
"""
struct WithoutConduction{T,Tr<:AbstractTransport{T}} <: AbstractTransport{T}
    transport::Tr
end

@inline function transport_at(tr::WithoutConduction, eos, temperature, rho, cp, Y, I)
    m = transport_at(tr.transport, eos, temperature, rho, cp, Y, I)
    return (mu=m.mu, kappa=zero(m.kappa), D=m.D)
end

@inline function transport_coefficients(tr::WithoutConduction, eos, temperature, rho,
                                        cp, Y)
    m = transport_coefficients(tr.transport, eos, temperature, rho, cp, Y)
    return (mu=m.mu, kappa=zero(m.kappa), D=m.D)
end

validate_transport(tr::WithoutConduction, eos) = validate_transport(tr.transport, eos)
transport_has_domain(tr::WithoutConduction) = transport_has_domain(tr.transport)
transport_domain_status(tr::WithoutConduction, eos, temperature, rho, Y) =
    transport_domain_status(tr.transport, eos, temperature, rho, Y)

# --- Storage ------------------------------------------------------------------

"""
    ImexIntegrator

The implicit half of a solver built with [`ImplicitConduction`](@ref): the
settings, the tableau in the solver's element type, the conduction stage
(`DiffusionStage`), the stage registers (the explicit right-hand side of
every stage, the implicit energy tendency of every stage and the state at the
start of the step), the conduction's lagged coefficients, the accuracy limit
on the next step, the cap a failed stage leaves on every later step, and
counters of the work done. Held by the solver as
`solver.implicit`; `nothing` there selects the default integrator.
"""
mutable struct ImexIntegrator{T,S}
    settings::ImplicitConduction
    explicit_matrix::Matrix{T}
    implicit_matrix::Matrix{T}
    weights::Vector{T}
    embedded::Vector{T}
    nodes::Vector{T}
    gamma::T
    stage::DiffusionStage{T}
    base::S                               # Qⁿ
    explicit::Vector{S}                   # F_E of each stage
    implicit::Vector{Array{T,3}}          # F_I of each stage (energy)
    known::Array{T,3}                     # R, the explicit part of the stage
    guess::Array{T,3}                     # e of the previous stage, per unit mass
    kappa::Array{T,3}                     # κ(T) at the trial state
    capacity::Array{T,3}                  # m = ρ c_v at the trial state
    tendency::Array{T,3}                  # Σ_c F_c at the trial state
    defect::Array{T,3}                    # the right-hand side −G
    increment::Array{T,3}                 # δT
    thermal::Array{T,3}                   # ρ c_v T at the start of the step
    dt_limit::T                           # the accuracy limit on the next step
    dt_cap::T                             # the limit left by failed stages
    measure::T                            # the step rule's last reading
    accepted::Int
    rejected::Int
    solves::Int                           # linear solves
    krylov::Int                           # Krylov iterations over them
    outer::Int                            # outer iterations
end

function ImexIntegrator(solver, settings::ImplicitConduction)
    T = eltype(solver.inv_J)
    stage = DiffusionStage(solver; parity=1)
    decomp = solver.decomp
    Q = allocate_state(solver)
    f() = field(decomp)
    ImexIntegrator{T,typeof(Q)}(settings, T.(ARK436_EXPLICIT), T.(ARK436_IMPLICIT),
                                T.(ARK436_WEIGHTS), T.(ARK436_EMBEDDED),
                                T.(ARK436_NODES), T(ARK436_IMPLICIT[2, 2]), stage, Q,
                                [allocate_state(solver) for _ in 1:ARK436_STAGES],
                                [f() for _ in 1:ARK436_STAGES],
                                f(), f(), f(), f(), f(), f(), f(), f(),
                                T(Inf), T(Inf), zero(T), 0, 0, 0, 0, 0)
end
