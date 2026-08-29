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
# the full padded arrays, so they need no halo exchange. Scale factors are
# clamped away from zero, which keeps the halo layers beyond a physical edge
# finite: A_d there is multiplied into the flux over the whole array (rhs.jl),
# and the product is discarded, either because the closure rows at a closed
# edge read interior points only or because `fold_fill!` overwrites it at a
# fold.
#
# Nothing in this file treats a coordinate singularity. A cylindrical or
# spherical domain either excludes the axis/origin (r > 0) and the poles
# (0 < θ < π), with the `origin` keyword of `Solver` set accordingly, or
# declares them through `AxisBC`, `OriginBC` and `PoleBC`, which place the grid
# half-offset from the singular set and fold the operators across it (folds.jl).

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

`x` and `dxdξ` are called at setup only, over the full padded index range. The
argument is clamped to [0, 1] first, so both need only be defined on the unit
interval, including in the halo layers. A stretched dimension must be
non-periodic and must carry no coordinate fold, and its mapping must reproduce
the endpoints of the corresponding `Problem.domain` interval. [`setup`](@ref)
checks all three.
"""
struct Stretch{F,G}
    x::F
    dxdξ::G
end

"""
    sine_cluster(lo, hi, ξc, a)

Closed-form interior clustering, returned as a [`Stretch`](@ref) over the
interval `[lo, hi]` of length L = hi − lo: x(ξ) = lo + L(ξ − (a/2π)[sin 2π(ξ−ξc)
+ sin 2πξc]), dx/dξ = L(1 − a cos 2π(ξ−ξc)). The map reproduces `lo` at ξ = 0
and `hi` at ξ = 1 for any `ξc`, so it satisfies the domain check in
[`setup`](@ref).

Spacing is smallest at ξ = ξc (fractional position of the cluster point),
largest opposite it, with ratio (1+a)/(1−a); any 0 ≤ a < 1 keeps the map
monotone and a larger value is an error. One-sided clustering requires a custom
monotone map.
"""
function sine_cluster(lo::Real, hi::Real, ξc::Real, a::Real)
    0 <= a < 1 || error("sine_cluster: need 0 ≤ a < 1")
    L = hi - lo
    Stretch(ξ -> lo + L * (ξ - (a / (2π)) * (sin(2π * (ξ - ξc)) + sin(2π * ξc))),
            ξ -> L * (1 - a * cos(2π * (ξ - ξc))))
end

scalefactors(::CartesianMetric,   x1, x2, x3) = (one(x1), one(x2), one(x3))
scalefactors(::CylindricalMetric, r,  θ,  z ) = (one(r), r, one(z))
scalefactors(::SphericalMetric,   r,  θ,  φ ) = (one(r), r, r * sin(θ))

"""
    unit_scalefactor(metric, d) -> Bool

Whether direction `d` has scale factor one everywhere under `metric`, i.e. its
computational coordinate is a length and not an angle. A `Stretch` on `d` is not
considered: the mapping Jacobian scales the coordinate derivative, which every
consumer already divides out through `inv_h`, whereas an angular direction also
contributes curvature terms to the physical velocity gradients. Characteristic
boundary conditions use this to reject faces their wave analysis does not cover.
"""
unit_scalefactor(::CartesianMetric,   d::Int) = true
unit_scalefactor(::CylindricalMetric, d::Int) = d != 2          # θ carries r
unit_scalefactor(::SphericalMetric,   d::Int) = d == 1          # θ, φ carry r

# Computational coordinate of full-array index `if_` along d, and the
# corresponding physical coordinate plus mapping Jacobian. ξ is clamped into
# the map's domain for halo layers beyond a closed physical edge, whose geometry
# values do not reach the answer (see the header).
@inline function _phys_and_jac(solver, d::Int, if_::Int)
    ξ = solver.origin[d] + solver.coord_shift[d] +
        (solver.region.offset[d] + solver.decomp.offset[d] +
         (if_ - solver.decomp.n_halo_d[d]) - 1) * solver.h[d]
    st = solver.stretch[d]
    st === nothing && return ξ, one(ξ)
    ξc = clamp(ξ, zero(ξ), one(ξ))
    return st.x(ξc), st.dxdξ(ξc)
end

"""
Fill the geometric arrays of `solver` over the full padded extent, in place, and
return `solver`. No halo exchange is needed, since every value is evaluated from
the global index.

Under `SphericalMetric` with a resolved θ the last step is `gcl_cotr!`, which
runs a distributed compact solve along θ. Every rank must therefore reach this
function in that configuration; the two conditions guarding the call are global
properties, so no rank can take the other branch on its own.
"""
function init_geometry!(solver)
    if _cpu_storage(solver.inv_J)
        _fill_geometry!(solver, solver.inv_J, solver.area_d, solver.inv_h,
                        solver.inv_r, solver.cot_over_r)
    else
        # The analytic evaluation runs per global index through the host-side
        # stretch closures, so a device-resident patch fills host mirrors once
        # at setup and uploads them; nothing here repeats during a run.
        T = eltype(solver.inv_J)
        sz = size(solver.inv_J)
        inv_J = zeros(T, sz)
        area_d = (zeros(T, sz), zeros(T, sz), zeros(T, sz))
        inv_h = (zeros(T, sz), zeros(T, sz), zeros(T, sz))
        inv_r = zeros(T, sz)
        cot_over_r = zeros(T, sz)
        _fill_geometry!(solver, inv_J, area_d, inv_h, inv_r, cot_over_r)
        copyto!(solver.inv_J, inv_J)
        for d in 1:3
            copyto!(solver.area_d[d], area_d[d])
            copyto!(solver.inv_h[d], inv_h[d])
        end
        copyto!(solver.inv_r, inv_r)
        copyto!(solver.cot_over_r, cot_over_r)
    end
    # The momentum sources read the discrete-GCL cotθ/r; everything else
    # (the velocity-gradient correction, the rate estimate) keeps the
    # analytic value. Without a resolved θ the two coincide.
    copyto!(solver.cot_over_r_gcl, solver.cot_over_r)
    solver.metric isa SphericalMetric && solver.decomp.active[2] && gcl_cotr!(solver)
    return solver
end

# The analytic fill of `init_geometry!`, into the supplied host arrays: the
# solver's own fields on a CPU patch, staging mirrors on a device one.
function _fill_geometry!(solver, inv_J, area_d, inv_h, inv_r, cot_over_r)
    tiny = positive_floor(eltype(inv_J))
    nxf, nyf, nzf = size(inv_J)
    for k in 1:nzf, j in 1:nyf, i in 1:nxf
        x1, m1 = _phys_and_jac(solver, 1, i)
        x2, m2 = _phys_and_jac(solver, 2, j)
        x3, m3 = _phys_and_jac(solver, 3, k)
        h1, h2, h3 = scalefactors(solver.metric, x1, x2, x3)
        h1 *= m1; h2 *= m2; h3 *= m3
        h1 = max(h1, tiny); h2 = max(h2, tiny); h3 = max(h3, tiny)
        J = h1 * h2 * h3
        inv_J[i, j, k] = 1 / J
        area_d[1][i, j, k] = J / h1
        area_d[2][i, j, k] = J / h2
        area_d[3][i, j, k] = J / h3
        inv_h[1][i, j, k] = 1 / h1
        inv_h[2][i, j, k] = 1 / h2
        inv_h[3][i, j, k] = 1 / h3
        if solver.metric isa CylindricalMetric
            inv_r[i, j, k] = 1 / max(x1, tiny)
        elseif solver.metric isa SphericalMetric
            inv_r[i, j, k] = 1 / max(x1, tiny)
            cot_over_r[i, j, k] = cos(x2) / max(x1 * sin(x2), tiny)   # cotθ / r
        end
    end
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
# for any uniform state, by construction.
#
# The result lives in `cot_over_r_gcl`, read by `add_metric_sources!` alone,
# and not in `cot_over_r`, which `metric_correct_gradients!` reads for the
# curvature terms of ∇u. The discrete value deviates from the analytic one at
# the order of the θ closure, measured third order on a spherical shell both
# at the walls and in the interior (the compact solve is global), so sharing
# one array put a third-order error into `grad_u[3,2]`, `grad_u[3,3]` and
# everything built from them at every scheme order.
function gcl_cotr!(solver)
    decomp = solver.decomp
    # Same antipodal sign the flux-divergence loop uses for the θ-momentum
    # (m2) pressure flux across the θ pole fold (ignored when there is no fold).
    σ = solver.folds[2] === nothing ? 1 : solver.folds[2].sigflux[solver.equations.i_mom[2]]
    # D_ξ2(A₂) with the m2 fold sign, through the divergence plans so the
    # cancellation holds against the operator the divergence loop applies.
    div_along!(solver.tmp_a, solver.area_d[2], solver, 2, σ)
    o1, o2, o3 = decomp.n_halo_d
    nx, ny, nz = decomp.n_local
    pointwise!(_gcl_cotr_point!, solver.cot_over_r_gcl, nx, ny, nz,
               solver.cot_over_r_gcl, solver.inv_J, solver.tmp_a, o1, o2, o3)
    return solver
end

# Interior only: the halo layers keep the analytic value copied in by
# `init_geometry!`, and no consumer of `cot_over_r_gcl` reads a halo.
@inline function _gcl_cotr_point!(cot_over_r_gcl, inv_J, tmp_a, o1, o2, o3,
                                  i, j, k)
    @inbounds begin
        I = CartesianIndex(i + o1, j + o2, k + o3)
        cot_over_r_gcl[I] = inv_J[I] * tmp_a[I]
    end
    return nothing
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

@inline function _grad_corr_cyl_point!(grad_u, u, v, inv_r, o1, o2, o3, i, j, k)
    @inbounds begin
        I = CartesianIndex(i + o1, j + o2, k + o3)
        ir = inv_r[I]
        grad_u[2, 1][I] -= v[I] * ir
        grad_u[2, 2][I] += u[I] * ir
    end
    return nothing
end

function metric_correct_gradients!(solver, ::CylindricalMetric)
    decomp = solver.decomp; o1, o2, o3 = decomp.n_halo_d
    nx, ny, nz = decomp.n_local
    pointwise!(_grad_corr_cyl_point!, solver.inv_r, nx, ny, nz,
               solver.field_tuples.grad_u, solver.u, solver.v, solver.inv_r,
               o1, o2, o3)
    return solver
end

@inline function _grad_corr_sph_point!(grad_u, u, v, w, inv_r, cot_over_r,
                                       o1, o2, o3, i, j, k)
    @inbounds begin
        I = CartesianIndex(i + o1, j + o2, k + o3)
        ir  = inv_r[I]                      # 1/r
        ctr = cot_over_r[I]                 # cotθ / r
        ur, uθ, uφ = u[I], v[I], w[I]
        grad_u[2, 1][I] -= uθ * ir
        grad_u[2, 2][I] += ur * ir
        grad_u[3, 1][I] -= uφ * ir
        grad_u[3, 2][I] -= uφ * ctr
        grad_u[3, 3][I] += ur * ir + uθ * ctr
    end
    return nothing
end

function metric_correct_gradients!(solver, ::SphericalMetric)
    decomp = solver.decomp; o1, o2, o3 = decomp.n_halo_d
    nx, ny, nz = decomp.n_local
    pointwise!(_grad_corr_sph_point!, solver.inv_r, nx, ny, nz,
               solver.field_tuples.grad_u, solver.u, solver.v, solver.w,
               solver.inv_r, solver.cot_over_r, o1, o2, o3)
    return solver
end

# ---------------------------------------------------------------------------
# Momentum sources from ∇·Π in curvilinear coordinates, added to dQ after the
# metric divergence of the fluxes. Π_ab = ρ u_a u_b + p δ_ab − τ_ab is
# reconstructed pointwise from the stored primitives and gradients.

add_metric_sources!(solver, dQ, Q, ::CartesianMetric) = solver

# The stress-tensor sample takes the field arrays rather than the solver so
# the momentum-source bodies below stay launchable as device kernels.
@inline function _Pi(grad_u, mu0, mu_art, beta_art, rho, u, v, w, p, I, a, b)
    μ = mu0 + mu_art[I]
    β = beta_art[I]
    divu = grad_u[1, 1][I] + grad_u[2, 2][I] + grad_u[3, 3][I]
    τ = μ * (grad_u[a, b][I] + grad_u[b, a][I]) +
        (a == b ? (β - 2μ/3) * divu : zero(divu))
    uv = (u[I], v[I], w[I])
    rho[I] * uv[a] * uv[b] + (a == b ? p[I] : zero(p[I])) - τ
end

@inline function _metric_src_cyl_point!(dQ, grad_u, mu0, mu_art, beta_art,
                                        rho, u, v, w, p, inv_r, m1, m2,
                                        o1, o2, o3, i, j, k)
    @inbounds begin
        I = CartesianIndex(i + o1, j + o2, k + o3)
        ir = inv_r[I]
        # +Π_θθ / r on radial momentum, −Π_θr / r on the azimuthal one.
        dQ[I, m1] += ir * _Pi(grad_u, mu0, mu_art, beta_art, rho, u, v, w, p, I, 2, 2)
        dQ[I, m2] -= ir * _Pi(grad_u, mu0, mu_art, beta_art, rho, u, v, w, p, I, 2, 1)
    end
    return nothing
end

function add_metric_sources!(solver, dQ, Q, ::CylindricalMetric)
    decomp = solver.decomp; o1, o2, o3 = decomp.n_halo_d
    nx, ny, nz = decomp.n_local
    m = solver.equations.i_mom
    pointwise!(_metric_src_cyl_point!, solver.inv_r, nx, ny, nz,
               dQ, solver.field_tuples.grad_u, solver.transport.mu0, solver.mu_art,
               solver.beta_art, solver.rho, solver.u, solver.v, solver.w,
               solver.p, solver.inv_r, m[1], m[2], o1, o2, o3)
    return solver
end

@inline function _metric_src_sph_point!(dQ, grad_u, mu0, mu_art, beta_art,
                                        rho, u, v, w, p, inv_r, cot_over_r,
                                        m1, m2, m3, o1, o2, o3, i, j, k)
    @inbounds begin
        I = CartesianIndex(i + o1, j + o2, k + o3)
        ir  = inv_r[I]
        ctr = cot_over_r[I]
        Pθθ = _Pi(grad_u, mu0, mu_art, beta_art, rho, u, v, w, p, I, 2, 2)
        Pφφ = _Pi(grad_u, mu0, mu_art, beta_art, rho, u, v, w, p, I, 3, 3)
        dQ[I, m1] += ir * (Pθθ + Pφφ)
        dQ[I, m2] += -ir * _Pi(grad_u, mu0, mu_art, beta_art, rho, u, v, w, p, I, 2, 1) +
                     ctr * Pφφ
        dQ[I, m3] += -ir * _Pi(grad_u, mu0, mu_art, beta_art, rho, u, v, w, p, I, 3, 1) -
                     ctr * _Pi(grad_u, mu0, mu_art, beta_art, rho, u, v, w, p, I, 3, 2)
    end
    return nothing
end

function add_metric_sources!(solver, dQ, Q, ::SphericalMetric)
    decomp = solver.decomp; o1, o2, o3 = decomp.n_halo_d
    nx, ny, nz = decomp.n_local
    m = solver.equations.i_mom
    pointwise!(_metric_src_sph_point!, solver.inv_r, nx, ny, nz,
               dQ, solver.field_tuples.grad_u, solver.transport.mu0, solver.mu_art,
               solver.beta_art, solver.rho, solver.u, solver.v, solver.w,
               solver.p, solver.inv_r, solver.cot_over_r_gcl, m[1], m[2], m[3],
               o1, o2, o3)
    return solver
end
