# Seams between mechanisms that are each correct alone: the antipodal pairing
# against the velocity-component parity, the GCL correction against the
# velocity-gradient correction, the banded x sweep against the y/z sweeps, the
# thermal diffusion limit against the EOS. Each case here is one that the
# rest of the suite passed while the seam was wrong. Included from runtests.jl,
# which supplies `CL`, `per3`, `fillf!` and `ferr`.

@testset "resolved axis: velocity components of a smooth Cartesian field" begin
    # A uniform Cartesian flow has u_r = cosθ and u_θ = −sinθ: both vary in θ
    # and are constant along every line through the axis, so their radial
    # derivatives vanish. The even/odd butterfly must mirror the even
    # combination as even for ANY σ; mirroring it with σ = −1 gave an O(1/h)
    # error on the two cells beside the axis.
    solver = Solver(n_global=(24, 32, 1), L_domain=(1.0, 2π, 1.0),
                    metric=CylindricalMetric(),
                    bcs=((AxisBC(), SlipWallBC()), per3[2], per3[3]),
                    art=ArtParams(enabled=false))
    f = CL.field(solver.decomp); df = CL.field(solver.decomp)
    for (fun, dfun, σ) in (((r, θ, z) -> cos(θ), (r, θ, z) -> 0.0, -1),
                           ((r, θ, z) -> -sin(θ), (r, θ, z) -> 0.0, -1),
                           # u_r of V = (x, 0): r cos²θ, radial derivative cos²θ
                           ((r, θ, z) -> r * cos(θ)^2, (r, θ, z) -> cos(θ)^2, -1))
        fillf!(solver, f, fun)
        CL.exchange_halos!(f, solver.decomp)
        CL.deriv_along!(df, f, solver, 1, σ); CL._scale_grad!(df, solver, 1)
        @test ferr(solver, df, dfun) < 1e-12
    end
    # The δ⁴ detector on the same field, through the same butterfly: the
    # sensor beside the axis must sit at the interior floor, not 10⁴ above it.
    solver = Solver(n_global=(24, 64, 1), L_domain=(1.0, 2π, 1.0),
                    metric=CylindricalMetric(),
                    bcs=((AxisBC(), SlipWallBC()), per3[2], per3[3]),
                    art=ArtParams(mu_sensor=:velocity))
    Q = allocate_state(solver)
    initialize!(solver, Q, (r, θ, z) -> Prim(u=(cos(θ), -sin(θ), 0.0), p=1.0, rho=1.0))
    apply_bcs!(solver, Q)
    CL.exchange_state!(Q, solver.decomp); CL.primitives!(solver, Q)
    out = zero(solver.u)
    CL.detect_sum!(out, solver.u, solver, 1; parity=(-1, 1, 1))
    o1, o2, o3 = solver.decomp.n_halo_d
    n_th = solver.decomp.n_local[2]
    axis = maximum(abs, out[o1+1:o1+2, o2+1:o2+n_th, o3+1])
    inner = maximum(abs, out[o1+3:o1+8, o2+1:o2+n_th, o3+1])
    @test axis <= 2 * inner
end

@testset "spherical: analytic cotθ/r for ∇u, discrete-GCL cotθ/r for sources" begin
    # The two live in separate arrays. The GCL value deviates from the analytic
    # one at the order of the θ closure; sharing one array put that deviation
    # into grad_u at every scheme order.
    solver = Solver(n_global=(16, 32, 12), L_domain=(1.0, π/2, 2π),
                    origin=(1.0, π/4, 0.0), metric=SphericalMetric(),
                    bcs=((SlipWallBC(), SlipWallBC()), (SlipWallBC(), SlipWallBC()),
                         per3[3]), art=ArtParams(enabled=false))
    o1, o2, o3 = solver.decomp.n_halo_d
    nx, ny = solver.decomp.n_local[1], solver.decomp.n_local[2]
    ana(i, j) = cos(xcoord(solver, 2, j)) / (xcoord(solver, 1, i) * sin(xcoord(solver, 2, j)))
    e_an = maximum(abs(solver.cot_over_r[i+o1, j+o2, 1+o3] / ana(i, j) - 1)
                   for i in 1:nx, j in 1:ny)
    e_gcl = maximum(abs(solver.cot_over_r_gcl[i+o1, j+o2, 1+o3] / ana(i, j) - 1)
                    for i in 1:nx, j in 1:ny)
    @test e_an == 0
    @test 1e-6 < e_gcl < 1e-4
    # Freestream preservation still holds through the GCL array.
    Q = allocate_state(solver)
    initialize!(solver, Q, (r, θ, φ) -> Prim(u=(0, 0, 0), p=1.0, rho=1.0))
    apply_bcs!(solver, Q)
    dQ = zero(Q); compute_rhs!(solver, Q, dQ)
    @test maximum(abs, dQ) < 1e-12
end

@testset "banded solve: one rounding path for every sweep direction" begin
    # A C10 derivative of the same profile along x and along y must agree
    # bitwise; the x sweep divided by the pivot where the y sweep multiplied by
    # its reciprocal, and the two differed in the last bit.
    solver = Solver(n_global=(24, 24, 12), L_domain=(1.0, 1.0, 0.5),
                    bcs=((SlipWallBC(), SlipWallBC()), (SlipWallBC(), SlipWallBC()),
                         per3[3]), deriv=lele_d1_10(), art=ArtParams(enabled=false))
    f = CL.field(solver.decomp); g = CL.field(solver.decomp)
    d1 = CL.field(solver.decomp); d2 = CL.field(solver.decomp)
    fn(x, y, z) = exp(sin(3x)) * (1 + 0.1z)
    fillf!(solver, f, fn)
    fillf!(solver, g, (x, y, z) -> fn(y, x, z))
    CL.exchange_halos!(f, solver.decomp); CL.exchange_halos!(g, solver.decomp)
    CL.deriv_along!(d1, f, solver, 1, 1)
    CL.deriv_along!(d2, g, solver, 2, 1)
    o1, o2, o3 = solver.decomp.n_halo_d
    n = solver.decomp.n_local[1]
    @test all(d1[o1+i, o2+j, o3+k] == d2[o2+j, o1+i, o3+k]
              for i in 1:n, j in 1:n, k in 1:solver.decomp.n_local[3])
end

@testset "diffusive rate: thermal term is κ/(ρ cv)" begin
    # With conduction dominant, the rate must scale with cv, not cp: ideal gas
    # at γ = 1.4, a 40% larger thermal rate than the cp form gives.
    n = (16, 16, 12)
    solver = Solver(n_global=n, L_domain=(1.0, 1.0, 0.75), bcs=per3,
                    transport=Transport(mu0=1e-2, Pr=1e-3, Sc=1.0),
                    art=ArtParams(enabled=false))
    Q = allocate_state(solver)
    initialize!(solver, Q, (x, y, z) -> Prim(u=(0, 0, 0), p=1.0, rho=1.0))
    rate, _ = CL.max_rate(solver, Q)
    o1, o2, o3 = solver.decomp.n_halo_d
    I = CartesianIndex(o1 + 1, o2 + 1, o3 + 1)
    ρ = solver.rho[I]; cp = solver.cp_mix[I]; c = solver.c[I]
    cv = CL.mixture_cv(solver.eos, ρ, solver.p[I], solver.T_ion[I], cp)
    @test cv ≈ cp / 1.4
    tr = solver.transport
    dsum = sum(d -> (1 / solver.h[d])^2, 1:3)
    ν = tr.mu0 / ρ + (tr.mu0 * cp / tr.Pr) / (ρ * cv) + tr.mu0 / (tr.Sc * ρ)
    expected = c * sum(d -> 1 / solver.h[d], 1:3) + 2ν * dsum
    @test rate ≈ expected rtol = 1e-12
end
