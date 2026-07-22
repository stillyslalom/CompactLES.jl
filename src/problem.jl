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
    Prim(; u=(0,0,0), p, T=NaN, rho=NaN, Y=(1.0,))

Pointwise primitive state: velocity components (physical, coordinate-aligned),
pressure, mass fractions, and exactly one of temperature or density (the EOS
supplies the other).
"""
struct Prim{N}
    Y::NTuple{N,Float64}
    u::NTuple{3,Float64}
    p::Float64
    T::Float64
    rho::Float64
end

function Prim(; u=(0.0, 0.0, 0.0), p::Real, T::Real=NaN, rho::Real=NaN, Y=(1.0,))
    (isnan(T) ⊻ isnan(rho)) ||
        error("Prim: specify exactly one of T or rho")
    Yt = Tuple(Float64.(Y))
    abs(sum(Yt) - 1) < 1e-10 || error("Prim: mass fractions must sum to 1")
    Prim{length(Yt)}(Yt, Tuple(Float64.(u)), Float64(p), Float64(T), Float64(rho))
end

"""
    conserved_from_prim(eos, pr) -> NTuple

EOS contract: primitive → conserved (ρY₁..ρY_Ns, ρu, ρv, ρw, E). Implemented
here for ideal mixtures; future EOS models supply their own method.
"""
function conserved_from_prim(eos::IdealMixture, pr::Prim{N}) where {N}
    N == nspecies(eos) ||
        error("Prim carries $N mass fractions; EOS has $(nspecies(eos)) species")
    Rm = 0.0; cvm = 0.0
    for k in 1:N
        Rm  += pr.Y[k] * eos.Rk[k]
        cvm += pr.Y[k] * eos.cvk[k]
    end
    ρ = isnan(pr.rho) ? pr.p / (Rm * pr.T) : pr.rho
    T = isnan(pr.T) ? pr.p / (ρ * Rm) : pr.T
    ke = 0.5 * (pr.u[1]^2 + pr.u[2]^2 + pr.u[3]^2)
    (ntuple(k -> ρ * pr.Y[k], N)...,
     ρ * pr.u[1], ρ * pr.u[2], ρ * pr.u[3],
     ρ * (cvm * T + ke))
end

@inline function write_conserved!(Q, I, s, pr::Prim)
    q = conserved_from_prim(s.eos, pr)
    @inbounds for c in 1:s.ncons
        Q[I, c] = q[c]
    end
    return Q
end

"""
    initialize!(s, Q, ic)

Fill the interior of `Q` from `ic(x₁, x₂, x₃) -> Prim`, evaluated at physical
coordinates. The IC function never sees ranks, halos, offsets, or component
layouts. IC functions should be pure (they are called from multiple threads).
"""
initialize!(s::Solver, Q, ic) = _initialize!(s, s.eos, Q, ic)

function _initialize!(s::Solver, eos, Q, ic)
    dec = s.dec
    o1, o2, o3 = dec.Hd
    nx, ny, nz = dec.nloc
    Threads.@threads for k in 1:nz
        x3 = xcoord(s, 3, k)
        for j in 1:ny
            x2 = xcoord(s, 2, j)
            for i in 1:nx
                x1 = xcoord(s, 1, i)
                write_conserved!(Q, CartesianIndex(i + o1, j + o2, k + o3), s,
                                 ic(x1, x2, x3))
            end
        end
    end
    return Q
end

"Smoothed step: 0 for x ≪ x0, 1 for x ≫ x0, tanh transition of width δ."
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
"""
struct DirichletBC <: BoundaryCondition
    fun::Function
end

function enforce!(bc::DirichletBC, Q, s, d, side)
    pl = wallplane(s.dec, d, side)
    pl === nothing && return nothing
    Hd = s.dec.Hd
    t = s.tstage
    @inbounds for I in pl
        x1 = xcoord(s, 1, I[1] - Hd[1])
        x2 = xcoord(s, 2, I[2] - Hd[2])
        x3 = xcoord(s, 3, I[3] - Hd[3])
        write_conserved!(Q, I, s, bc.fun(x1, x2, x3, t))
    end
    nothing
end

# ---------------------------------------------------------------------------
# Backend-decoupled problem and numerics bundles.

"""
    Problem(; eos, transport, metric, domain, bcs, ic, name)

Physics and geometry only: EOS, transport, metric, coordinate `domain` as
three (lo, hi) tuples, boundary conditions, and the IC function. Contains no
resolution, scheme, or parallel information.
"""
Base.@kwdef struct Problem
    name::String = "problem"
    eos::EOS = single_species()
    transport::Transport{Float64} = Transport()
    metric::Metric = CartesianMetric()
    domain::NTuple{3,Tuple{Float64,Float64}}
    bcs::NTuple{3,Tuple{BoundaryCondition,BoundaryCondition}}
    ic::Function
end

"""
    Numerics(; nglob, kwargs...)

Discretization and runtime choices: resolution, derivative/filter schemes,
artificial-property constants, CFL, filter interval, process grid, halo width.
"""
Base.@kwdef struct Numerics
    nglob::NTuple{3,Int}
    deriv::AbstractCompactScheme = lele_d1_6()
    filt::AbstractCompactScheme = compact_filter(0.45)
    art::ArtParams{Float64} = ArtParams()
    cfl::Float64 = 0.5
    filter_interval::Int = 1
    dims::Union{Nothing,NTuple{3,Int}} = nothing
    H::Int = 4
    stretch::NTuple{3,Union{Nothing,Stretch}} = (nothing, nothing, nothing)
end

"""
    setup(prob, num) -> (solver, Q)

Construct the backend for `prob` at the resolution and schemes of `num` and
return the solver plus the initialized conserved state.
"""
function setup(prob::Problem, num::Numerics)
    origin = ntuple(d -> prob.domain[d][1], 3)
    Ldom = ntuple(d -> prob.domain[d][2] - prob.domain[d][1], 3)
    all(>(0), Ldom) || error("domain extents must be positive")
    for d in 1:3
        st = num.stretch[d]
        st === nothing && continue
        isapprox(st.x(0.0), prob.domain[d][1]; atol=1e-10 * Ldom[d]) &&
        isapprox(st.x(1.0), prob.domain[d][2]; atol=1e-10 * Ldom[d]) ||
            error("stretch mapping for dim $d does not span the domain: " *
                  "x(0) = $(st.x(0.0)), x(1) = $(st.x(1.0)), " *
                  "domain = $(prob.domain[d])")
    end
    s = Solver(nglob=num.nglob, Ldom=Ldom, bcs=prob.bcs,
               eos=prob.eos, transport=prob.transport, art=num.art,
               metric=prob.metric, stretch=num.stretch, origin=origin,
               deriv=num.deriv, filt=num.filt,
               cfl=num.cfl, filter_interval=num.filter_interval,
               dims=num.dims, H=num.H)
    Q = allocate_state(s)
    initialize!(s, Q, prob.ic)
    return s, Q
end
