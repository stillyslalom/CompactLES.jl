# Constant-density incompressible Navier–Stokes on the staggered compact
# operators of staggered.jl, projected through the pressure solve of
# implicit.jl. It shares the compressible solver's decomposition, halo
# exchange, distributed line solves and Krylov–multigrid stage, and none of
# its conserved-variable machinery.
#
# Arrangement. The pressure (divided by the density) lives at the nodes and
# velocity component i at the midpoints along dimension i, the nodes along
# the others: the marker-and-cell arrangement, with the sixth-order compact
# staggered derivatives and explicit interpolations (Lele 1992; Nagarajan,
# Lele and Ferziger, JCP 191, 2003). The divergence of a velocity is
# Σ_i G_i u_i at the nodes, the gradient of a node field D_s,i p at the
# midpoints along i, and their composition Σ_i G_i D_s,i is the operator L
# of the implicit stage at unit coefficient. The projection
#
#     u ← u − D_s φ,   L φ = Σ_i G_i u_i,
#
# therefore leaves Σ_i G_i u_i zero to the tolerance of the solve: the
# projection is exact, not approximate, and needs no filter on the pressure.
# A collocated arrangement would compose two wide compact derivatives, whose
# product vanishes at the Nyquist mode and leaves the odd–even pressure mode
# undamped; its exact projection would need a face velocity carried beside
# the node one.
#
# Momentum. The divergence form of the convective term, ∂_j(u_j u_i), is
# formed for j = i from u_i² at the nodes, the component interpolated there,
# differentiated back to the midpoints; for j ≠ i the product is formed where
# the two components meet, at the midpoints along both i and j, from u_j
# interpolated along i and u_i interpolated along j, and differentiated along
# j back onto u_i's points. The advective form u_j ∂_j u_i takes the same
# transporting velocity at the same points, multiplies it by the staggered
# derivative of u_i there, and interpolates the product back. The default is
# their mean, the skew-symmetric form: the midpoint-to-node interpolation is
# the transpose of the node-to-midpoint one (the taps are symmetric), and
# G_j is minus the adjoint of D_s,j in the node and midpoint weights, so the
# two forms' contributions to d/dt ½|u|² cancel term by term and the
# convective term does no work on the discrete kinetic energy, divergence
# free or not (Morinishi et al., JCP 143, 1998, for the second-order case).
# The divergence form alone conserves momentum instead, and at an
# under-resolved scale it feeds energy back into the resolved motion. The
# viscous term is ν Σ_j ∂_j∂_j u_i, each second derivative the composition of
# the two staggered first derivatives, so it is the operator L along each
# direction.
#
# Time. The right-hand side is projected rather than the state: P is linear
# and constant, so du/dt = P N(u) is an ordinary differential equation on
# the divergence-free fields, and the five-stage low-storage Runge–Kutta
# scheme of timestep.jl keeps its fourth order. Each stage costs one pressure
# solve, warm-started from the previous stage's φ, which is the kinematic
# pressure at the stage time. The solve also removes the state's divergence
# over one step (`_project!`), so its residuals do not accumulate.
#
# Walls. A closed dimension is bounded by walls through its end nodes, which
# the staggered plans treat as mirror planes: the normal component is odd,
# the pressure even, and a tangential component even at a slip wall and odd
# at a no-slip wall. The slip wall is an exact symmetry of the equations, so
# the mirror keeps the interior order there. A no-slip tangential velocity is
# not odd about the wall (its second derivative carries the pressure
# gradient), and the mirror is first order at the wall node.
#
# Ranks. Every operator is a directional plan on the distributed line
# solves, and the pressure solve is the collective conjugate-gradient
# iteration of implicit.jl, so the solver runs decomposed over any process
# grid the staggered plans admit (four points per rank and direction), and
# every rank reaches every collective. Storage is on the host, as the
# implicit stage's is.

"""
    IncompressibleSolver(n_global; lengths, periodic, nu=0, walls=:slip,
                         convection=:skew, forcing=nothing, rtol=1e-12,
                         maxiter=200, T=Float64, dims=nothing, comm=MPI.COMM_WORLD)

A constant-density incompressible Navier–Stokes solver on a uniform
Cartesian grid of `n_global` nodes, with pressure at the nodes and
each velocity component at the midpoints along its own direction, the
sixth-order staggered compact operators throughout, and an exact projection
through the pressure solve of `DiffusionStage` at every stage of the
low-storage Runge–Kutta step.

A periodic dimension of length `L` holds `N` nodes at spacing `L/N` from
zero; a closed one holds them at spacing `L/(N − 1)` with a wall through each
end node, slip or no-slip by `walls` (`:slip` keeps the interior order, and
`:noslip` is first order at the wall node). `nu` is the kinematic
viscosity. `convection` is `:skew`, the skew-symmetric form, which does no
work on the discrete kinetic energy, or `:divergence`, which conserves
momentum and can feed energy into an under-resolved scale. `forcing`, when
given, is a function `(i, x, y, z, t)` returning the acceleration of
component `i` at a point. Two or three active
dimensions are supported, decomposed over the ranks of `comm` like any
`Decomp`; storage is on the host.

Set the initial velocity with `set_velocity!`, advance with `step!` or
`run!`, and read `kinetic_energy`, `max_divergence` and the kinematic
pressure `solver.phi` of the last projection.
"""
mutable struct IncompressibleSolver{T,F}
    decomp::Decomp{T}
    n_global::NTuple{3,Int}
    lengths::NTuple{3,T}
    h::NTuple{3,T}
    dims::Vector{Int}
    nu::T
    noslip::Bool
    skew::Bool                    # skew-symmetric convection, else divergence form
    forcing::F
    U::NTuple{3,Array{T,3}}       # component i at the midpoints along i
    dU::NTuple{3,Array{T,3}}      # low-storage register
    R::NTuple{3,Array{T,3}}       # projected right-hand side of a stage
    phi::Array{T,3}               # kinematic pressure of the last projection
    divergence::Array{T,3}
    rhs::Array{T,3}
    work::NTuple{3,Array{T,3}}
    viscous::Array{T,3}           # Σ_j ∂_j∂_j u_i of one component
    unit::Array{T,3}              # the unit coefficient of the pressure operator
    capacity::Array{T,3}          # zero: the stage solves −L φ = r
    plans::Dict{Tuple{Symbol,Int,Int},StaggeredPlan{T}}
    stage::DiffusionStage{T}
    rtol::T
    maxiter::Int
    t::T
    step::Int
    iterations::Int               # Krylov iterations of the last projection
    total_iterations::Int
end

function IncompressibleSolver(n_global::NTuple{3,Int};
                              lengths=(2π, 2π, 2π),
                              periodic::NTuple{3,Bool}=(true, true, true),
                              nu::Real=0, walls::Symbol=:slip,
                              convection::Symbol=:skew, forcing=nothing,
                              rtol::Real=1e-12, maxiter::Int=200, T::Type=Float64,
                              dims=nothing, comm=MPI.COMM_WORLD)
    walls in (:slip, :noslip) || throw(ArgumentError(
        "walls must be :slip or :noslip, not :$walls"))
    convection in (:skew, :divergence) || throw(ArgumentError(
        "convection must be :skew or :divergence, not :$convection"))
    count(>(1), n_global) >= 2 || throw(ArgumentError(
        "the incompressible solver needs two or three active dimensions"))
    decomp = Decomp{T}(n_global, periodic; dims=dims, comm=comm)
    active = [d for d in 1:3 if decomp.active[d]]
    h = ntuple(3) do d
        decomp.active[d] || return one(T)
        T(lengths[d]) / (decomp.periodic[d] ? n_global[d] : n_global[d] - 1)
    end
    f() = field(decomp)
    stage = DiffusionStage(decomp, h; parity=1, singular=true)
    unit = f(); unit .= one(T)
    IncompressibleSolver(decomp, n_global, ntuple(d -> T(lengths[d]), 3), h, active,
                         T(nu), walls === :noslip, convection === :skew, forcing,
                         (f(), f(), f()),
                         (f(), f(), f()), (f(), f(), f()), f(), f(), f(),
                         (f(), f(), f()), f(), unit, f(),
                         Dict{Tuple{Symbol,Int,Int},StaggeredPlan{T}}(), stage,
                         T(rtol), maxiter, zero(T), 0, 0, 0)
end

# The plan of `op` along `d` for an input of parity `σ` at the walls; one
# per key, built on first use. Every rank reaches the same keys in the same
# order, so the collective factorizations match.
function _plan(s::IncompressibleSolver, op::Symbol, d::Int, σ::Int)
    σ = s.decomp.periodic[d] ? 1 : σ
    get!(s.plans, (op, d, σ)) do
        plan_staggered(s.decomp, op, d, s.h[d]; parity=σ)
    end
end

function _along!(out, s::IncompressibleSolver, op::Symbol, d::Int, σ::Int, f)
    exchange_dim!(f, s.decomp, d)
    apply_along!(out, _plan(s, op, d, σ), f, s.decomp)
    return out
end

# The parity of velocity component `i` across the walls of dimension `d`.
_velocity_parity(s::IncompressibleSolver, i::Int, d::Int) =
    i == d ? -1 : (s.noslip ? -1 : 1)

# The slot of the midpoint above the last node of a closed dimension holds no
# data and stays zero.
function _clear_dummy!(f, s::IncompressibleSolver, d::Int)
    decomp = s.decomp
    at_hi_edge(decomp, d) || return f
    selectdim(f, d, decomp.n_halo_d[d] + decomp.n_local[d]) .= 0
    return f
end

# Position of local padded index `i` along `d`, on a node or a midpoint.
_coordinate(s::IncompressibleSolver{T}, d::Int, i::Int, mid::Bool) where {T} =
    (s.decomp.offset[d] + i - s.decomp.n_halo_d[d] - 1 + (mid ? T(1) / 2 : 0)) * s.h[d]

function _position(s::IncompressibleSolver, I::CartesianIndex{3}, component::Int)
    ntuple(d -> _coordinate(s, d, I[d], d == component), 3)
end

function _forcing!(R, s::IncompressibleSolver{T}, i::Int, t) where {T}
    fill!(R, zero(T))
    s.forcing === nothing && return R
    for I in interior(s.decomp)
        x, y, z = _position(s, I, i)
        R[I] = s.forcing(i, x, y, z, t)
    end
    return R
end

# Σ_j ∂_j∂_j u_i into `s.viscous`, each second derivative the composition of
# the two staggered first derivatives. Collective.
function _laplacian!(s::IncompressibleSolver{T}, i::Int) where {T}
    a, c = s.work[1], s.work[3]
    out = s.viscous
    fill!(out, zero(T))
    for j in s.dims
        if j == i
            _along!(a, s, :to_node, i, -1, s.U[i])
            _along!(c, s, :to_mid, i, 1, a)
        else
            σ = _velocity_parity(s, i, j)
            _along!(a, s, :to_mid, j, σ, s.U[i])
            _along!(c, s, :to_node, j, -σ, a)
        end
        out .+= c
    end
    return _clear_dummy!(out, s, i)
end

# N(u) = −C_i(u) + ν ∂_j∂_j u_i + f_i into `s.R`, unprojected, with C_i the
# convective term in the solver's form. Collective.
function _momentum!(s::IncompressibleSolver{T}, t) where {T}
    a, b, c = s.work
    U = s.U
    w = s.skew ? T(1) / 2 : one(T)
    for i in s.dims
        R = _forcing!(s.R[i], s, i, t)
        for j in s.dims
            if j == i
                # a = u_i at the nodes, odd at the walls of i.
                _along!(a, s, :interpolate_to_node, i, -1, U[i])
                b .= a .* a
                _along!(c, s, :to_mid, i, 1, b)
                R .-= w .* c
                if s.skew
                    _along!(b, s, :to_node, i, -1, U[i])
                    b .*= a
                    _along!(c, s, :interpolate, i, -1, b)
                    R .-= w .* c
                end
            else
                # a = u_j at the midpoints along i and j, odd at the walls of j.
                σ = _velocity_parity(s, i, j)
                _along!(a, s, :interpolate, i, _velocity_parity(s, j, i), U[j])
                _along!(b, s, :interpolate, j, σ, U[i])
                b .*= a
                _along!(c, s, :to_node, j, -σ, b)
                R .-= w .* c
                if s.skew
                    _along!(b, s, :to_mid, j, σ, U[i])
                    b .*= a
                    _along!(c, s, :interpolate_to_node, j, σ, b)
                    R .-= w .* c
                end
            end
        end
        if !iszero(s.nu)
            R .+= s.nu .* _laplacian!(s, i)
        end
        _clear_dummy!(R, s, i)
    end
    return s.R
end

# Σ_i G_i F_i into `s.divergence`.
function _divergence!(s::IncompressibleSolver{T}, F) where {T}
    div = s.divergence
    fill!(div, zero(T))
    for i in s.dims
        _along!(s.work[1], s, :to_node, i, -1, F[i])
        div .+= s.work[1]
    end
    return div
end

function _weighted_sum(s::IncompressibleSolver{T}, f) where {T}
    acc = zero(T)
    V = s.stage.volume
    @inbounds for I in interior(s.decomp)
        acc += V[I] * f[I]
    end
    return MPI.Allreduce(acc, +, s.decomp.comm)
end

function _remove_mean!(f, s::IncompressibleSolver)
    mean = _weighted_sum(s, f) / _weighted_sum(s, s.unit)
    for I in interior(s.decomp)
        f[I] -= mean
    end
    return f
end

_l2(s::IncompressibleSolver, f) = sqrt(_weighted_sum(s, f .^ 2))

# F ← F − D_s φ with L φ = Σ_i G_i F_i + rate Σ_i G_i u_i, in place on the
# midpoint fields `F`. With `rate` zero this is the projection. Given the
# reciprocal of the step, F becomes the rate that also removes the state's
# divergence over one step: the solve's residual and round-off then decay
# instead of accumulating step by step, and the added term vanishes on a
# divergence-free state, so the scheme is unchanged there. The solve stops at
# `rtol` relative to the larger of the divergence and the divergence scale of
# F itself, ‖F‖/h, so that a field already divergence free to round-off is
# not iterated against its own round-off. Collective.
function _project!(s::IncompressibleSolver{T}, F, rate=zero(T)) where {T}
    if iszero(rate)
        div = _divergence!(s, F)
        s.rhs .= .-div
    else
        div = _divergence!(s, s.U)
        s.rhs .= .-T(rate) .* div
        div = _divergence!(s, F)
        s.rhs .-= div
    end
    _remove_mean!(s.rhs, s)
    dnorm = _l2(s, s.rhs)
    scale = sqrt(sum(_l2(s, F[i])^2 for i in s.dims)) / minimum(s.h[d] for d in s.dims)
    if iszero(dnorm)
        s.iterations = 0
    else
        rtol = s.rtol * max(one(T), scale / dnorm)
        result = solve_stage!(s.phi, s.stage, s.rhs, s.unit, one(T); rtol=rtol,
                              maxiter=s.maxiter, capacity=s.capacity)
        result.converged || error("pressure solve did not converge in " *
                                  "$(result.iterations) iterations (residual " *
                                  "$(result.residual))")
        s.iterations = result.iterations
        s.total_iterations += result.iterations
    end
    _remove_mean!(s.phi, s)
    for i in s.dims
        _along!(s.work[1], s, :to_mid, i, 1, s.phi)
        F[i] .-= s.work[1]
        _clear_dummy!(F[i], s, i)
    end
    return F
end

"""
    set_velocity!(solver::IncompressibleSolver, u)

Set the velocity from `u(x, y, z)`, a function returning the three
components at a point, each sampled on its own staggered points, and project
it onto the discretely divergence-free fields. Resets the clock, the step
count and the pressure. Collective.
"""
function set_velocity!(s::IncompressibleSolver{T}, u) where {T}
    for i in s.dims
        Ui = s.U[i]
        fill!(Ui, zero(T))
        for I in interior(s.decomp)
            Ui[I] = u(_position(s, I, i)...)[i]
        end
        _clear_dummy!(Ui, s, i)
        fill!(s.dU[i], zero(T))
    end
    fill!(s.phi, zero(T))
    _project!(s, s.U)
    fill!(s.phi, zero(T))
    s.t = zero(T); s.step = 0; s.total_iterations = 0
    return s
end

# The projected right-hand side at time `t` into `s.R`; `s.phi` becomes the
# kinematic pressure at `t`, to the state's divergence over `dt`, which is
# the solve tolerance. Collective.
function _rhs!(s::IncompressibleSolver{T}, t, dt) where {T}
    _momentum!(s, t)
    _project!(s, s.R, inv(T(dt)))
    return s.R
end

# One step of the five-stage low-storage Runge–Kutta scheme of timestep.jl,
# with the right-hand side of every stage projected. Collective.
function step!(s::IncompressibleSolver{T}, dt::Real) where {T}
    dt = T(dt)
    for k in 1:5
        _rhs!(s, s.t + T(RKC[k]) * dt, dt)
        A, B = T(RKA[k]), T(RKB[k])
        for i in s.dims
            s.dU[i] .= A .* s.dU[i] .+ dt .* s.R[i]
            s.U[i] .+= B .* s.dU[i]
        end
    end
    s.t += dt
    s.step += 1
    return s
end

# The step at Courant number `cfl` on the advective rate Σ_d max|u_d|/h_d,
# with the viscous rate 2.5 ν Σ_d 1/h_d² beside it. The staggered derivative's
# largest modified wavenumber is 2.61/h and the staggered second derivative's
# 6.79/h², so `cfl` = 1 sits inside the scheme's imaginary-axis limit and the
# viscous weight keeps the real-axis one. Collective.
function compute_dt(s::IncompressibleSolver{T}; cfl::Real=0.8) where {T}
    rate = zero(T)
    for d in s.dims
        umax = MPI.Allreduce(maximum(abs, view(s.U[d], interior(s.decomp))), max,
                             s.decomp.comm)
        rate += umax / s.h[d]
    end
    rate += T(5) / 2 * s.nu * sum(1 / s.h[d]^2 for d in s.dims)
    rate > 0 || error("no advective or viscous rate to size the step")
    return T(cfl) / rate
end

# Advance to `t_end` at steps of `compute_dt` (or a fixed `dt`), at most
# `nmax` of them, calling `callback(solver)` after each. Collective.
function run!(s::IncompressibleSolver{T}; t_end::Real, cfl::Real=0.8, dt=nothing,
              nmax::Int=typemax(Int), callback=nothing) where {T}
    while s.t < t_end && s.step < nmax
        Δt = dt === nothing ? compute_dt(s; cfl=cfl) : T(dt)
        Δt = min(Δt, T(t_end) - s.t)
        step!(s, Δt)
        callback === nothing || callback(s)
    end
    return s
end

# The local part of ∫ f g dV over component `i`'s points, by the
# trapezoidal node weights along each dimension and the midpoint weights
# along the component's own direction.
function _component_dot(s::IncompressibleSolver{T}, i::Int, f, g) where {T}
    acc = zero(T)
    pad = CartesianIndex(s.decomp.n_halo_d)
    nw = s.stage.node_weight
    @inbounds for I in CartesianIndices(s.decomp.n_local)
        w = one(T)
        for d in 1:3
            w *= d == i ? s.h[d] : nw[d][I[d]]
        end
        acc += w * f[I+pad] * g[I+pad]
    end
    return acc
end

"""
    kinetic_energy(solver::IncompressibleSolver)

The kinetic energy per unit density, ½ ∫ |u|² dV, by the trapezoidal node
weights along each dimension and the midpoint weights along a component's
own direction. Collective.
"""
kinetic_energy(s::IncompressibleSolver) =
    MPI.Allreduce(sum(_component_dot(s, i, s.U[i], s.U[i]) for i in s.dims), +,
                  s.decomp.comm) / 2

"""
    viscous_dissipation(solver::IncompressibleSolver)

The rate at which the discrete viscous term removes kinetic energy,
−ν Σ_i ∫ u_i ∂_j∂_j u_i dV in the weights of `kinetic_energy`, which on a
periodic or slip-walled domain is ν ∫ |∇u|² dV. The difference between it
and −dK/dt is the energy the convective term and the time integration
remove. Collective.
"""
function viscous_dissipation(s::IncompressibleSolver{T}) where {T}
    acc = zero(T)
    for i in s.dims
        acc += _component_dot(s, i, s.U[i], _laplacian!(s, i))
    end
    return -s.nu * MPI.Allreduce(acc, +, s.decomp.comm)
end

"""
    max_divergence(solver::IncompressibleSolver)

The largest magnitude of the discrete divergence Σ_i G_i u_i over the nodes.
Collective.
"""
function max_divergence(s::IncompressibleSolver)
    div = _divergence!(s, s.U)
    return MPI.Allreduce(maximum(abs, view(div, interior(s.decomp))), max, s.decomp.comm)
end

"""
    node_velocity!(out, solver::IncompressibleSolver, i)

Write velocity component `i`, interpolated at sixth order from its midpoints
to the nodes, into the interior of `out`. Collective along dimension `i`.
"""
node_velocity!(out, s::IncompressibleSolver, i::Int) =
    _along!(out, s, :interpolate_to_node, i, -1, s.U[i])
