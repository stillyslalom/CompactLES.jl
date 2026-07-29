# Frontend: problem specification decoupled from the numerical backend.
#
# The user-facing vocabulary is primitive and pointwise. Initial conditions
# are functions (x₁, x₂, x₃) → Prim; Dirichlet boundary forcing is a function
# (x₁, x₂, x₃, t) → Prim evaluated at the RK stage time. A `Problem` bundles
# physics, domain, boundary conditions, and the IC — with no reference to
# grids, ranks, halos, or conserved layouts — while `Numerics` bundles
# resolution and scheme choices. `setup(problem, numerics)` marries the two
# and returns (solver, Q). The same Problem can therefore be re-run at
# different resolutions, orders, or (eventually) against a different backend
# without touching the physics description; primitive→conserved conversion is
# EOS business and flows through the same contract future cubic/tabular
# models implement.

"""
    Prim(; u=(0,0,0), p, T_ion=NaN, rho=NaN, Y=(1.0,))

Pointwise primitive state used by initial and prescribed-boundary functions.
All values are converted to `Float64`.

# Keywords

- `p`: pressure. This keyword is required.
- `u`: three physical velocity components in the coordinate-aligned orthonormal
  basis. The default is a stationary state.
- `T_ion`: temperature. Specify this or `rho`, but not both.
- `rho`: mixture density. Specify this or `T_ion`, but not both.
- `Y`: species mass fractions, in the order used to construct the EOS. They
  must sum to one; the default is a single species with unit mass fraction.

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

function Prim(; u=(0.0, 0.0, 0.0), p::Real, T_ion::Real=NaN, rho::Real=NaN, Y=(1.0,))
    (isnan(T_ion) ⊻ isnan(rho)) ||
        error("Prim: specify exactly one of T_ion or rho")
    Yt = Tuple(Float64.(Y))
    abs(sum(Yt) - 1) < 1e-10 || error("Prim: mass fractions must sum to 1")
    Prim{length(Yt)}(Yt, Tuple(Float64.(u)), Float64(p), Float64(T_ion), Float64(rho))
end

"""
    conserved_from_prim(eos, pr::Prim) -> NTuple
    conserved_from_prim(equations, eos, pr::Prim) -> NTuple

Convert a pointwise primitive state to the conserved layout owned by
`equations`. The two-argument form uses [`NavierStokes1T`](@ref).

For `NavierStokes1T`, the result is
`(rho*Y[1], ..., rho*Y[Ns], rho*u[1], rho*u[2], rho*u[3], rho*E)`,
where the final entry is total energy per volume. The EOS obtains density from
`(p, T_ion, Y)` or temperature from `(p, rho, Y)`, according to which value was
supplied to `pr`.

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
`ic(x1, x2, x3) -> Prim`, evaluated at physical coordinates. The callback sees
neither ranks and halos nor the conserved-component layout, and should be pure
because it can be called concurrently from multiple threads.

This function leaves halo cells unchanged and does not reset solver time,
timestep history, or diagnostics. Use it to reuse an existing solver and
allocation for a different initial state with the same EOS, geometry, and
numerical configuration. It returns `Q`.
"""
initialize!(solver::Solver, Q, ic) = _initialize!(solver, solver.eos, Q, ic)

function _initialize!(solver::Solver, eos, Q, ic)
    decomp = solver.decomp
    o1, o2, o3 = decomp.n_halo_d
    nx, ny, nz = decomp.n_local
    @threaded nx*ny*nz for k in 1:nz
        x3 = xcoord(solver, 3, k)
        for j in 1:ny
            x2 = xcoord(solver, 2, j)
            for i in 1:nx
                x1 = xcoord(solver, 1, i)
                write_conserved!(Q, CartesianIndex(i + o1, j + o2, k + o3), solver,
                                 ic(x1, x2, x3))
            end
        end
    end
    return Q
end

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
`fun(x₁, x₂, x₃, t) -> Prim`, evaluated at the RK stage time — suitable for
supersonic/forced inflow, pistons, and oscillating drivers. For subsonic
inflow a characteristic (NSCBC) treatment is physically preferable and
remains a TODO; a full-state Dirichlet over-constrains subsonic boundaries
and will reflect.

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
    n_halo_d = solver.decomp.n_halo_d
    t = solver.tstage
    @inbounds for I in plane
        x1 = xcoord(solver, 1, I[1] - n_halo_d[1])
        x2 = xcoord(solver, 2, I[2] - n_halo_d[2])
        x3 = xcoord(solver, 3, I[3] - n_halo_d[3])
        write_conserved!(Q, I, solver, bc.fun(x1, x2, x3, t))
    end
    nothing
end

# ---------------------------------------------------------------------------
# Backend-decoupled problem and numerics bundles.

"""
    Problem(; domain, bcs, ic, name="problem", eos=single_species(),
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
- `eos`: equation of state and species definition. Default:
  [`single_species()`](@ref).
- `transport`: constant molecular transport properties. Default:
  [`Transport()`](@ref), which has zero molecular viscosity.
- `metric`: coordinate metric. Default: [`CartesianMetric()`](@ref).
- `sources`: tuple of explicit source objects applied to the RHS. Default: `()`.
  See [`add_source!`](@ref) when defining a custom source.

Coordinates passed to `ic` and to boundary callbacks follow `metric`. Species
mass fractions in every returned `Prim` must follow the order defined by `eos`.
"""
Base.@kwdef struct Problem
    name::String = "problem"
    eos::EOS = single_species()
    transport::Transport{Float64} = Transport()
    metric::Metric = CartesianMetric()
    sources::Tuple = ()
    domain::NTuple{3,Tuple{Float64,Float64}}
    bcs::NTuple{3,Tuple{BoundaryCondition,BoundaryCondition}}
    ic::Function
end

"""
    Numerics(; n_global, deriv=lele_d1_6(), filt=compact_filter(0.45),
             art=ArtParams(), cfl=0.5, control=StepControl(),
             filter_interval=1, dims=nothing, n_halo=4,
             stretch=(nothing, nothing, nothing))

Grid, scheme, timestep, and decomposition choices used to realize a
[`Problem`](@ref).

# Keywords

- `n_global`: required three-tuple giving the global point count in each
  coordinate direction. A count of one collapses that direction: it has no
  derivative, halo, or decomposition.
- `deriv`: compact first-derivative scheme. Default: [`lele_d1_6()`](@ref).
- `filt`: compact state and sensor filter. Default:
  [`compact_filter(0.45)`](@ref), where values nearer `0.5` are weaker.
- `art`: artificial-property coefficients. Default: [`ArtParams()`](@ref).
- `cfl`: multiplier used by [`compute_dt`](@ref). Default: `0.5`. Strong shocks
  can require a lower startup value.
- `control`: timestep prediction, failure floors, and retry policy. Default:
  [`StepControl()`](@ref).
- `filter_interval`: state-filter cadence in completed steps. A positive value
  `k` applies `filt` every `k` steps. The default `1` filters every step; `0`
  disables state filtering. The filter is still used to smooth
  artificial-property sensors.
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
"""
Base.@kwdef struct Numerics
    n_global::NTuple{3,Int}
    deriv::AbstractCompactScheme = lele_d1_6()
    filt::AbstractCompactScheme = compact_filter(0.45)
    art::ArtParams{Float64} = ArtParams()
    cfl::Float64 = 0.5
    control::StepControl = StepControl()
    filter_interval::Int = 1
    dims::Union{Nothing,NTuple{3,Int}} = nothing
    n_halo::Int = 4
    stretch::NTuple{3,Union{Nothing,Stretch}} = (nothing, nothing, nothing)
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
               dims=num.dims, n_halo=num.n_halo)
    Q = allocate_state(solver)
    initialize!(solver, Q, prob.ic)
    return solver, Q
end
