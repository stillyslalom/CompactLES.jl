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

abstract type Metric end
struct CartesianMetric   <: Metric end
struct CylindricalMetric <: Metric end   # (r, θ, z)
struct SphericalMetric   <: Metric end   # (r, θ_polar, φ)

"""
    Stretch(x, dxds)

Per-dimension monotone grid mapping for stretched meshes: the computational
coordinate s is uniform on [0, 1] and `x(s)` gives the physical coordinate,
with `dxds(s)` its derivative. Composes with any base metric (the mapping
Jacobian multiplies that dimension's scale factor), so clustered radial grids
in cylindrical coordinates work the same way as clustered Cartesian ones.
Stretched dimensions must be non-periodic.
"""
struct Stretch
    x::Function
    dxds::Function
end

"""
    sine_cluster(lo, hi, sc, a)

Closed-form interior clustering: x(s) = lo + L(s − (a/2π)[sin 2π(s−sc) +
sin 2πsc]), dx/ds = L(1 − a cos 2π(s−sc)). Spacing is smallest at s = sc
(fractional position of the cluster point), largest opposite it, with ratio
(1+a)/(1−a); any 0 ≤ a < 1 keeps the map monotone. For boundary-layer-style
one-sided clustering supply your own monotone map.
"""
function sine_cluster(lo::Real, hi::Real, sc::Real, a::Real)
    0 <= a < 1 || error("sine_cluster: need 0 ≤ a < 1")
    L = hi - lo
    Stretch(s -> lo + L * (s - (a / (2π)) * (sin(2π * (s - sc)) + sin(2π * sc))),
            s -> L * (1 - a * cos(2π * (s - sc))))
end

scalefactors(::CartesianMetric,   x1, x2, x3) = (1.0, 1.0, 1.0)
scalefactors(::CylindricalMetric, r,  θ,  z ) = (1.0, r, 1.0)
scalefactors(::SphericalMetric,   r,  θ,  φ ) = (1.0, r, r * sin(θ))

# Computational coordinate of full-array index `if_` along d, and the
# corresponding physical coordinate plus mapping Jacobian (clamped into the
# map's domain for halo layers beyond closed physical edges — those geometry
# values are never read).
@inline function _phys_and_jac(s, d::Int, if_::Int)
    ξ = s.origin[d] + s.cshift[d] +
        (s.dec.off[d] + (if_ - s.dec.Hd[d]) - 1) * s.h[d]
    st = s.stretch[d]
    st === nothing && return ξ, 1.0
    sc = clamp(ξ, 0.0, 1.0)
    return st.x(sc), st.dxds(sc)
end

"Fill the geometric arrays of the solver over the full padded extent."
function init_geometry!(s)
    dec = s.dec
    H = dec.H
    tiny = 1e-300
    nxf, nyf, nzf = size(s.Jinv)
    for k in 1:nzf, j in 1:nyf, i in 1:nxf
        x1, m1 = _phys_and_jac(s, 1, i)
        x2, m2 = _phys_and_jac(s, 2, j)
        x3, m3 = _phys_and_jac(s, 3, k)
        h1, h2, h3 = scalefactors(s.metric, x1, x2, x3)
        h1 *= m1; h2 *= m2; h3 *= m3
        h1 = max(h1, tiny); h2 = max(h2, tiny); h3 = max(h3, tiny)
        J = h1 * h2 * h3
        s.Jinv[i, j, k] = 1 / J
        s.Adim[1][i, j, k] = J / h1
        s.Adim[2][i, j, k] = J / h2
        s.Adim[3][i, j, k] = J / h3
        s.invh[1][i, j, k] = 1 / h1
        s.invh[2][i, j, k] = 1 / h2
        s.invh[3][i, j, k] = 1 / h3
        if s.metric isa CylindricalMetric
            s.rinv[i, j, k] = 1 / max(x1, tiny)
        elseif s.metric isa SphericalMetric
            s.rinv[i, j, k] = 1 / max(x1, tiny)
            s.cotr[i, j, k] = cos(x2) / max(x1 * sin(x2), tiny)   # cotθ / r
        end
    end
    return s
end

# ---------------------------------------------------------------------------
# Physical-component velocity gradients. compute_rhs! first scales the raw
# coordinate derivatives by 1/h_d (which now also carries any stretching
# Jacobian); these functions then ADD the curvature corrections only, so the
# same corrections serve stretched and unstretched grids. Convention matches rhs.jl:
# G[d, j] is the d-direction derivative of the j-th velocity component.

function metric_correct_gradients!(s, ::CartesianMetric)
    return s   # scale factors are unity; nothing to do
end

function metric_correct_gradients!(s, ::CylindricalMetric)
    dec = s.dec; o1, o2, o3 = dec.Hd
    nx, ny, nz = dec.nloc
    G = s.G
    Threads.@threads for k in 1:nz
        @inbounds for j in 1:ny, i in 1:nx
            I = CartesianIndex(i + o1, j + o2, k + o3)
            ir = s.rinv[I]
            G[2, 1][I] -= s.v[I] * ir
            G[2, 2][I] += s.u[I] * ir
        end
    end
    return s
end

function metric_correct_gradients!(s, ::SphericalMetric)
    dec = s.dec; o1, o2, o3 = dec.Hd
    nx, ny, nz = dec.nloc
    G = s.G
    Threads.@threads for k in 1:nz
        @inbounds for j in 1:ny, i in 1:nx
            I = CartesianIndex(i + o1, j + o2, k + o3)
            ir  = s.rinv[I]          # 1/r
            ctr = s.cotr[I]          # cotθ / r
            ur, uθ, uφ = s.u[I], s.v[I], s.w[I]
            G[2, 1][I] -= uθ * ir
            G[2, 2][I] += ur * ir
            G[3, 1][I] -= uφ * ir
            G[3, 2][I] -= uφ * ctr
            G[3, 3][I] += ur * ir + uθ * ctr
        end
    end
    return s
end

# ---------------------------------------------------------------------------
# Momentum sources from ∇·Π in curvilinear coordinates, added to dQ after the
# metric divergence of the fluxes. Π_ab = ρ u_a u_b + p δ_ab − τ_ab is
# reconstructed pointwise from the stored primitives and gradients.

add_metric_sources!(s, dQ, Q, ::CartesianMetric) = s

@inline function _Pi(s, Q, I, a, b)
    G = s.G
    μ = s.transport.mu0 + s.mua[I]
    β = s.betaa[I]
    divu = G[1, 1][I] + G[2, 2][I] + G[3, 3][I]
    τ = μ * (G[a, b][I] + G[b, a][I]) + (a == b ? (β - 2μ/3) * divu : 0.0)
    uv = (s.u[I], s.v[I], s.w[I])
    s.rho[I] * uv[a] * uv[b] + (a == b ? s.p[I] : 0.0) - τ
end

function add_metric_sources!(s, dQ, Q, ::CylindricalMetric)
    dec = s.dec; o1, o2, o3 = dec.Hd
    nx, ny, nz = dec.nloc
    m = s.mom
    Threads.@threads for k in 1:nz
        @inbounds for j in 1:ny, i in 1:nx
            I = CartesianIndex(i + o1, j + o2, k + o3)
            ir = s.rinv[I]
            dQ[I, m[1]] += ir * _Pi(s, Q, I, 2, 2)      # +Π_θθ / r
            dQ[I, m[2]] -= ir * _Pi(s, Q, I, 2, 1)      # −Π_θr / r
        end
    end
    return s
end

function add_metric_sources!(s, dQ, Q, ::SphericalMetric)
    dec = s.dec; o1, o2, o3 = dec.Hd
    nx, ny, nz = dec.nloc
    m = s.mom
    Threads.@threads for k in 1:nz
        @inbounds for j in 1:ny, i in 1:nx
            I = CartesianIndex(i + o1, j + o2, k + o3)
            ir  = s.rinv[I]
            ctr = s.cotr[I]
            Pθθ = _Pi(s, Q, I, 2, 2)
            Pφφ = _Pi(s, Q, I, 3, 3)
            dQ[I, m[1]] += ir * (Pθθ + Pφφ)
            dQ[I, m[2]] += -ir * _Pi(s, Q, I, 2, 1) + ctr * Pφφ
            dQ[I, m[3]] += -ir * _Pi(s, Q, I, 3, 1) - ctr * _Pi(s, Q, I, 3, 2)
        end
    end
    return s
end
