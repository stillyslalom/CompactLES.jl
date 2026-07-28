# Orthogonal curvilinear metrics: Cartesian (x, y, z), cylindrical (r, θ, z),
# and spherical (r, θ, φ) with θ the polar angle. Coordinates are uniformly
# spaced in the computational directions; geometry enters through scale
# factors h_d(x), which for these metrics gives three ingredients:
#
#   1. Divergence:  ∇·F = (1/J) Σ_d ∂(A_d F_d)/∂x_d with J = h₁h₂h₃ and
#      A_d = J / h_d, applied per flux component with the compact derivative.
#   2. Gradients:   (∇f)_d = (1/h_d) ∂f/∂x_d, and physical velocity-gradient
#      components pick up curvature corrections from the rotating unit vectors.
#   3. Momentum sources: the divergence of the momentum flux tensor Π
#      (Π_ab = ρ u_a u_b + p δ_ab − τ_ab) acquires per-metric algebraic terms.
#
# Geometry arrays (J⁻¹, A_d, 1/h_d, 1/r, cotθ/r) are filled analytically over
# the full padded arrays, so they need no halo exchange; scale factors are
# clamped away from zero so stale physical-edge halos stay finite (they are
# never read). Coordinate singularities are NOT treated: cylindrical and
# spherical domains must exclude the axis/origin (r > 0) and the poles
# (0 < θ < π). Set the `origin` keyword of `Solver` accordingly.

"""Abstract supertype for orthogonal coordinate metrics."""
abstract type Metric end

"""
    CartesianMetric()

Cartesian `(x, y, z)` coordinates with unit scale factors.
"""
struct CartesianMetric   <: Metric end

"""
    CylindricalMetric()

Cylindrical `(r, theta, z)` coordinates. Velocity components are physical
components `(u_r, u_theta, u_z)` in the local orthonormal basis.
"""
struct CylindricalMetric <: Metric end   # (r, θ, z)

"""
    SphericalMetric()

Spherical `(r, theta, phi)` coordinates with `theta` the polar angle. Velocity
components are physical components in the local orthonormal basis.
"""
struct SphericalMetric   <: Metric end   # (r, θ_polar, φ)

"""
    Stretch(x, dxdξ)

Per-dimension monotone grid mapping for stretched meshes: the computational
coordinate ξ is uniform on [0, 1] and `x(ξ)` gives the physical coordinate,
with `dxdξ(ξ)` its derivative. Composes with any base metric (the mapping
Jacobian multiplies that dimension's scale factor), so clustered radial grids
in cylindrical coordinates work the same way as clustered Cartesian ones.
Stretched dimensions must be non-periodic.
"""
struct Stretch{F,G}
    x::F
    dxdξ::G
end

"""
    sine_cluster(lo, hi, ξc, a)

Closed-form interior clustering: x(ξ) = lo + L(ξ − (a/2π)[sin 2π(ξ−ξc) +
sin 2πξc]), dx/dξ = L(1 − a cos 2π(ξ−ξc)). Spacing is smallest at ξ = ξc
(fractional position of the cluster point), largest opposite it, with ratio
(1+a)/(1−a); any 0 ≤ a < 1 keeps the map monotone. One-sided clustering requires
a custom monotone map.
"""
function sine_cluster(lo::Real, hi::Real, ξc::Real, a::Real)
    0 <= a < 1 || error("sine_cluster: need 0 ≤ a < 1")
    L = hi - lo
    Stretch(ξ -> lo + L * (ξ - (a / (2π)) * (sin(2π * (ξ - ξc)) + sin(2π * ξc))),
            ξ -> L * (1 - a * cos(2π * (ξ - ξc))))
end

scalefactors(::CartesianMetric,   x1, x2, x3) = (1.0, 1.0, 1.0)
scalefactors(::CylindricalMetric, r,  θ,  z ) = (1.0, r, 1.0)
scalefactors(::SphericalMetric,   r,  θ,  φ ) = (1.0, r, r * sin(θ))

# Computational coordinate of full-array index `if_` along d, and the
# corresponding physical coordinate plus mapping Jacobian (clamped into the
# map's domain for halo layers beyond closed physical edges — those geometry
# values are never read).
@inline function _phys_and_jac(solver, d::Int, if_::Int)
    ξ = solver.origin[d] + solver.coord_shift[d] +
        (solver.decomp.offset[d] + (if_ - solver.decomp.n_halo_d[d]) - 1) * solver.h[d]
    st = solver.stretch[d]
    st === nothing && return ξ, 1.0
    ξc = clamp(ξ, 0.0, 1.0)
    return st.x(ξc), st.dxdξ(ξc)
end

"Fill the geometric arrays of the solver over the full padded extent."
function init_geometry!(solver)
    decomp = solver.decomp
    n_halo = decomp.n_halo
    tiny = 1e-300
    nxf, nyf, nzf = size(solver.inv_J)
    for k in 1:nzf, j in 1:nyf, i in 1:nxf
        x1, m1 = _phys_and_jac(solver, 1, i)
        x2, m2 = _phys_and_jac(solver, 2, j)
        x3, m3 = _phys_and_jac(solver, 3, k)
        h1, h2, h3 = scalefactors(solver.metric, x1, x2, x3)
        h1 *= m1; h2 *= m2; h3 *= m3
        h1 = max(h1, tiny); h2 = max(h2, tiny); h3 = max(h3, tiny)
        J = h1 * h2 * h3
        solver.inv_J[i, j, k] = 1 / J
        solver.area_d[1][i, j, k] = J / h1
        solver.area_d[2][i, j, k] = J / h2
        solver.area_d[3][i, j, k] = J / h3
        solver.inv_h[1][i, j, k] = 1 / h1
        solver.inv_h[2][i, j, k] = 1 / h2
        solver.inv_h[3][i, j, k] = 1 / h3
        if solver.metric isa CylindricalMetric
            solver.inv_r[i, j, k] = 1 / max(x1, tiny)
        elseif solver.metric isa SphericalMetric
            solver.inv_r[i, j, k] = 1 / max(x1, tiny)
            solver.cot_over_r[i, j, k] = cos(x2) / max(x1 * sin(x2), tiny)   # cotθ / r
        end
    end
    solver.metric isa SphericalMetric && solver.decomp.active[2] && gcl_cotr!(solver)
    return solver
end

# Discrete Geometric Conservation Law correction for the spherical θ-momentum
# source. The θ-momentum balance for a uniform state pits the flux divergence
# −inv_J·D_ξ2(A₂·p) against the source +(cotθ/r)·Π_φφ (metric.jl below), where
# A₂ = r sinθ is the θ area factor and D_ξ2 is the compact θ-derivative used in
# the divergence loop (rhs.jl). Analytically inv_J·∂_ξ2(A₂) = cotθ/r and the two
# cancel; discretely D_ξ2(sinθ) ≠ cosθ, so the analytically sampled cotθ/r
# leaves an O(h⁴) freestream residual (a GCL truncation error, not a bug).
#
# We restore exact discrete freestream preservation by REDEFINING cotθ/r as
# inv_J·D_ξ2(A₂): the identical operator (same compact scheme, same pole/antipodal
# fold, same antipodal sign σ as the m2 pressure flux) applied to the identical
# area factor. By linearity the source then cancels the divergence node-by-node
# for any uniform state, by construction. The correction is O(h⁴) from the
# analytic value, so it does not degrade the scheme's formal order elsewhere.
function gcl_cotr!(solver)
    decomp = solver.decomp
    # Same antipodal sign the flux-divergence loop uses for the θ-momentum
    # (m2) pressure flux across the θ pole fold (ignored when there is no fold).
    σ = solver.folds[2] === nothing ? 1 : solver.folds[2].sigflux[solver.equations.i_mom[2]]
    # D_ξ2(A₂) with the m2 fold sign
    deriv_along!(solver.tmp_a, solver.area_d[2], solver, 2, σ)
    o1, o2, o3 = decomp.n_halo_d
    nx, ny, nz = decomp.n_local
    @inbounds for k in 1:nz, j in 1:ny, i in 1:nx
        I = CartesianIndex(i + o1, j + o2, k + o3)
        solver.cot_over_r[I] = solver.inv_J[I] * solver.tmp_a[I]
    end
    return solver
end

# ---------------------------------------------------------------------------
# Physical-component velocity gradients. compute_rhs! first scales the raw
# coordinate derivatives by 1/h_d (which now also carries any stretching
# Jacobian); these functions then ADD the curvature corrections only, so the
# same corrections serve stretched and unstretched grids. Convention matches rhs.jl:
# grad_u[d, j] is the d-direction derivative of the j-th velocity component.

function metric_correct_gradients!(solver, ::CartesianMetric)
    return solver   # scale factors are unity; nothing to do
end

function metric_correct_gradients!(solver, ::CylindricalMetric)
    decomp = solver.decomp; o1, o2, o3 = decomp.n_halo_d
    nx, ny, nz = decomp.n_local
    grad_u = solver.grad_u
    @threaded nx*ny*nz for k in 1:nz
        @inbounds for j in 1:ny, i in 1:nx
            I = CartesianIndex(i + o1, j + o2, k + o3)
            ir = solver.inv_r[I]
            grad_u[2, 1][I] -= solver.v[I] * ir
            grad_u[2, 2][I] += solver.u[I] * ir
        end
    end
    return solver
end

function metric_correct_gradients!(solver, ::SphericalMetric)
    decomp = solver.decomp; o1, o2, o3 = decomp.n_halo_d
    nx, ny, nz = decomp.n_local
    grad_u = solver.grad_u
    @threaded nx*ny*nz for k in 1:nz
        @inbounds for j in 1:ny, i in 1:nx
            I = CartesianIndex(i + o1, j + o2, k + o3)
            ir  = solver.inv_r[I]               # 1/r
            ctr = solver.cot_over_r[I]          # cotθ / r
            ur, uθ, uφ = solver.u[I], solver.v[I], solver.w[I]
            grad_u[2, 1][I] -= uθ * ir
            grad_u[2, 2][I] += ur * ir
            grad_u[3, 1][I] -= uφ * ir
            grad_u[3, 2][I] -= uφ * ctr
            grad_u[3, 3][I] += ur * ir + uθ * ctr
        end
    end
    return solver
end

# ---------------------------------------------------------------------------
# Momentum sources from ∇·Π in curvilinear coordinates, added to dQ after the
# metric divergence of the fluxes. Π_ab = ρ u_a u_b + p δ_ab − τ_ab is
# reconstructed pointwise from the stored primitives and gradients.

add_metric_sources!(solver, dQ, Q, ::CartesianMetric) = solver

@inline function _Pi(solver, Q, I, a, b)
    grad_u = solver.grad_u
    μ = solver.transport.mu0 + solver.mu_art[I]
    β = solver.beta_art[I]
    divu = grad_u[1, 1][I] + grad_u[2, 2][I] + grad_u[3, 3][I]
    τ = μ * (grad_u[a, b][I] + grad_u[b, a][I]) + (a == b ? (β - 2μ/3) * divu : 0.0)
    uv = (solver.u[I], solver.v[I], solver.w[I])
    solver.rho[I] * uv[a] * uv[b] + (a == b ? solver.p[I] : 0.0) - τ
end

function add_metric_sources!(solver, dQ, Q, ::CylindricalMetric)
    decomp = solver.decomp; o1, o2, o3 = decomp.n_halo_d
    nx, ny, nz = decomp.n_local
    m = solver.equations.i_mom
    @threaded nx*ny*nz for k in 1:nz
        @inbounds for j in 1:ny, i in 1:nx
            I = CartesianIndex(i + o1, j + o2, k + o3)
            ir = solver.inv_r[I]
            dQ[I, m[1]] += ir * _Pi(solver, Q, I, 2, 2)      # +Π_θθ / r
            dQ[I, m[2]] -= ir * _Pi(solver, Q, I, 2, 1)      # −Π_θr / r
        end
    end
    return solver
end

function add_metric_sources!(solver, dQ, Q, ::SphericalMetric)
    decomp = solver.decomp; o1, o2, o3 = decomp.n_halo_d
    nx, ny, nz = decomp.n_local
    m = solver.equations.i_mom
    @threaded nx*ny*nz for k in 1:nz
        @inbounds for j in 1:ny, i in 1:nx
            I = CartesianIndex(i + o1, j + o2, k + o3)
            ir  = solver.inv_r[I]
            ctr = solver.cot_over_r[I]
            Pθθ = _Pi(solver, Q, I, 2, 2)
            Pφφ = _Pi(solver, Q, I, 3, 3)
            dQ[I, m[1]] += ir * (Pθθ + Pφφ)
            dQ[I, m[2]] += -ir * _Pi(solver, Q, I, 2, 1) + ctr * Pφφ
            dQ[I, m[3]] += -ir * _Pi(solver, Q, I, 3, 1) - ctr * _Pi(solver, Q, I, 3, 2)
        end
    end
    return solver
end
