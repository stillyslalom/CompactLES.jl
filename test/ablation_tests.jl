# The stage-0 x-ray ablation drive (src/ablation.jl): its closure, its angular
# factor, and the exact radial integrals of its force and mass sink.
#
# Standalone: julia --project=. test/ablation_tests.jl

using MPI
MPI.Initialized() || MPI.Init(threadlevel=:funneled)
using CompactLES
using Test

@testset "ablation drive" begin
    ts = [0.0, 1.0, 2.0, 8.0, 9.0]
    Tr = [50.0, 150.0, 150.0, 300.0, 200.0]
    d = AblationDrive(ts, Tr; pressure=4.0)
    # closure: P = P0 T^3.3 (t/1 ns)^-0.09 Mbar, ṁ = 3e5 T³ g/cm²/s, T in heV
    @test ablation_pressure(d, 1.0) ≈ 4.0 * 1.5^3.3
    @test ablation_pressure(d, 8.0) ≈ 4.0 * 3.0^3.3 * 8.0^-0.09
    @test ablation_pressure(d, 0.0) ≈ 4.0 * 0.5^3.3 * 0.1^-0.09      # t_min
    @test CompactLES.ablation_rate(d, 8.0) ≈ 3e5 * 27
    @test CompactLES.drive_temperature(d, 5.0) ≈ 225.0
    @test CompactLES.drive_temperature(d, 20.0) == 200.0               # held
    # HYBRID-E calibration: peak 145 Mbar, foot above 12 Mbar at 150 eV
    p0 = ablation_pressure_scale(ts, Tr; peak=145.0)
    dc = AblationDrive(ts, Tr; pressure=p0)
    @test maximum(ablation_pressure(dc, τ) for τ in ts) ≈ 145.0
    @test ablation_pressure(dc, 1.0) > 12.0
    # angular factor: 1 + Σ (aₗ/a₀) Pₗ(μ); P2 > 0 raises the poles
    M = [0.0 0.0 0.0 0.0 0.0; 0.1 0.1 0.1 0.1 0.1; 0.0 0.0 0.0 0.0 0.0; 0.02 0.02 0.02 0.02 0.02]
    da = AblationDrive(ts, Tr; modes=M)
    P2(μ) = (3μ^2 - 1) / 2
    P4(μ) = (35μ^4 - 30μ^2 + 3) / 8
    for μ in (-1.0, -0.3, 0.0, 0.5, 1.0)
        @test CompactLES.drive_factor(da, μ, 3.0) ≈ 1 + 0.1P2(μ) + 0.02P4(μ)
    end
    @test CompactLES.drive_modes(da, 3.0) == (0.0, 0.1, 0.0, 0.02)
    @test_throws ArgumentError AblationDrive(ts, Tr[1:3])
    @test_throws ArgumentError AblationDrive(ts, Tr; modes=zeros(2, 3))
    @test occursin("AblationDrive(5 times", sprint(show, da))

    # The source on a shell: the force and mass sink integrate exactly to P and ṁ along
    # each radial line, act on the ablator only, and carry its energy.
    unit = (pressure=1.0, mass_flux=1.0, time=1.0)
    eos = IdealMixture([IdealSpecies{Float64}("abl", 1.0, 5 / 3),
                        IdealSpecies{Float64}("gas", 2.0, 5 / 3)])
    ic = (r, θ, φ) -> begin
        # local: the testset assigns ρa below, which the closure would otherwise share
        # across the threads that evaluate it
        local b, ρa, ρg
        b = tanh_blend(r, 0.6, 0.02) - tanh_blend(r, 0.8, 0.02)
        ρa, ρg = 1.0 * b + 1e-6, 0.01
        Prim(rho=ρa + ρg, p=0.1, Y=(ρa / (ρa + ρg), ρg / (ρa + ρg)))
    end
    function source_rhs(drive, nθ)
        θdom = nθ == 1 ? (π / 2, π / 2 + 1) : (0.0, π)
        θbc = nθ == 1 ? PeriodicBC() : (PoleBC(), PoleBC())
        prob = Problem(eos=eos, metric=SphericalMetric(), domain=((0.0, 1.0), θdom, (0.0, 1.0)),
                       bcs=((OriginBC(), SlipWallBC()), θbc, PeriodicBC()), ic=ic)
        solver, Q = setup(prob, Numerics(n_global=(200, nθ, 1)))
        dQ = similar(Q)
        fill!(parent(dQ), 0.0)
        CompactLES.add_source!(drive, solver, dQ, Q, 3.0)
        return solver, Q, dQ
    end
    d1 = AblationDrive(ts, Tr; modes=M, units=unit, layer=0.02)
    solver, Q, dQ = source_rhs(d1, 1)
    o = solver.decomp.n_halo_d[1]
    h = solver.h[1]
    eq = solver.equations
    m1, ie = eq.i_mom[1], eq.i_energy
    rows = o .+ (1:solver.decomp.n_local[1])
    P, mdot = ablation_pressure(d1, 3.0), CompactLES.ablation_rate(d1, 3.0)
    @test sum(dQ[rows, 1, 1, m1]) * h ≈ -P rtol = 1e-12          # 1-D: the angle average
    @test sum(dQ[rows, 1, 1, 1]) * h ≈ -mdot rtol = 1e-12
    @test all(iszero, dQ[rows, 1, 1, 2])                           # the gas keeps its mass
    @test all(<=(0), dQ[rows, 1, 1, 1])
    # at rest, the energy removed is the ablated mass's own: Σ ṁ w E/ρ
    E = [Q[i, 1, 1, ie] / (Q[i, 1, 1, 1] + Q[i, 1, 1, 2]) for i in rows]
    @test sum(dQ[rows, 1, 1, ie]) ≈ sum(dQ[rows, 1, 1, 1] .* E) rtol = 1e-12
    # the layer sits at the outer edge of the shell (0.8), inside it
    rc = [CompactLES.xcoord(solver, 1, i) for i in 1:solver.decomp.n_local[1]]
    f = -dQ[rows, 1, 1, m1]
    centroid = sum(rc .* f) / sum(f)
    @test 0.75 < centroid < 0.8
    @test all(>=(0), f) && sum(f[rc .< 0.7]) < 1e-6 * sum(f)
    # the body-force form keeps the two-sided layer, beyond the front as well (a force
    # on tenuous ablator outside it would accelerate that without bound)
    @test sum(f[rc .> 0.95]) < 1e-12 * sum(f)
    @test maximum(abs, dQ[rows[rc .> 0.95], 1, 1, 1]) < 1e-12 * mdot
    # r–θ: each line carries P (1 + Σ aₗ/a₀ Pₗ)^(3.3/4)
    solver2, Q2, dQ2 = source_rhs(d1, 16)
    o1, o2 = solver2.decomp.n_halo_d[1], solver2.decomp.n_halo_d[2]
    rows2 = o1 .+ (1:solver2.decomp.n_local[1])
    for j in 1:solver2.decomp.n_local[2]
        μ = cos(CompactLES.xcoord(solver2, 2, j))
        f = CompactLES.drive_factor(d1, μ, 3.0)
        @test sum(dQ2[rows2, o2 + j, 1, m1]) * h ≈ -P * f^(3.3 / 4) rtol = 1e-12
        @test sum(dQ2[rows2, o2 + j, 1, 1]) * h ≈ -mdot * f^(3 / 4) rtol = 1e-12
    end
    # driver-gas form: the ablated mass becomes driver gas (mass, momentum and energy
    # unchanged), and the driver gas's pressure relaxes to P_abl
    dd = AblationDrive(ts, Tr; units=unit, layer=0.02, driver=2, relaxation=0.1)
    sd, Qd, dQd = source_rhs(dd, 1)
    @test sum(dQd[rows, 1, 1, 1]) * h ≈ -mdot rtol = 1e-12
    @test dQd[rows, 1, 1, 2] ≈ -dQd[rows, 1, 1, 1]
    @test all(iszero, dQd[rows, 1, 1, m1])
    ρd = Qd[rows, 1, 1, 1] .+ Qd[rows, 1, 1, 2]
    Yd = Qd[rows, 1, 1, 2] ./ ρd
    # the front, as the source finds it: the ρₐ(−∂ρₐ/∂r)⁺-weighted mean radius
    ρa = [Qd[i, 1, 1, 1] for i in (rows[1] - 1):(rows[end] + 1)]
    gf = [max(ρa[k], 0) * max(-(ρa[k+1] - ρa[k-1]), 0) for k in 2:length(ρa)-1]
    gf[1] = gf[end] = 0.0
    Rf = sum(gf .* rc) / sum(gf)
    @test 0.78 < Rf < 0.82
    # the piston acts at and beyond the front, with weight exp(−x²) inside it
    x = (rc .- Rf) ./ 0.02
    S = ifelse.(x .> 0, 1.0, exp.(-x .^ 2))
    @test dQd[rows, 1, 1, ie] ≈ S .* Yd .^ 2 .* (P .- sd.p[rows, 1, 1]) ./ ((5 / 3 - 1) * 0.1)
    # the conversion takes all the ablator beyond the front (weight ρₐ there, so none
    # is stranded behind an accelerating shell), the Gaussian layer inside it
    k = rows[1] - 1
    w = [Qd[k + i, 1, 1, 1] * (x[i] > 0 ? 1.0 : exp(-x[i]^2)) for i in eachindex(rc)]
    @test -dQd[rows, 1, 1, 1] ≈ mdot .* w ./ (sum(w) * h)
    far = rc .> Rf + 0.1
    @test all(<(0), dQd[rows[far], 1, 1, 1])
    @test_throws ArgumentError AblationDrive(ts, Tr; driver=1)
    # unsupported layouts are refused
    bad = Problem(eos=eos, metric=CylindricalMetric(), domain=((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)),
                  bcs=((AxisBC(), SlipWallBC()), PeriodicBC(), PeriodicBC()), ic=(r, θ, z) -> ic(r, θ, z))
    sb, Qb = setup(bad, Numerics(n_global=(64, 1, 1)))
    @test_throws ArgumentError CompactLES.add_source!(d1, sb, similar(Qb), Qb, 0.0)
end

# The binned form (src/ablation_bins.jl): the same drive on r–z and 3-D Cartesian
# grids, the front and the layer normalization summed over angular bins.
@testset "ablation drive, binned geometry" begin
    ts = [0.0, 1.0, 2.0, 8.0, 9.0]
    Tr = [50.0, 150.0, 150.0, 300.0, 200.0]
    unit = (pressure=1.0, mass_flux=1.0, time=1.0)
    eos = IdealMixture([IdealSpecies{Float64}("abl", 1.0, 5 / 3),
                        IdealSpecies{Float64}("gas", 2.0, 5 / 3)])
    function shell(R)
        local b = tanh_blend(R, 0.6, 0.02) - tanh_blend(R, 0.8, 0.02)
        local ρa = b + 1e-6
        return Prim(rho=ρa + 0.01, p=0.1, Y=(ρa / (ρa + 0.01), 0.01 / (ρa + 0.01)))
    end
    function rhs(prob, n, d; t=3.0)
        solver, Q = setup(prob, Numerics(n_global=n))
        dQ = similar(Q)
        fill!(parent(dQ), 0.0)
        CompactLES.add_source!(d, solver, dQ, Q, t)
        return solver, Q, dQ
    end
    # Schmidt harmonics: Ŷ20 = P2, Ŷ22 = (√3/2) sin²θ cos 2φ, Ŷ3,-1 = √(1/6)(3/2)(5μ²−1) sinθ sinφ
    H = zeros(16, length(ts))
    H[1, :] .= 1
    H[7, :] .= 0.05                      # (2, 0)
    H[9, :] .= 0.1                       # (2, 2)
    H[12, :] .= 0.03                     # (3, -1)
    dh = AblationDrive(ts, Tr; harmonics=H)
    @test dh isa AblationDrive{3} && CompactLES.drive_modes(dh, 1.0) == (0.0, 0.05, 0.0)
    for (μ, φ) in ((0.3, 0.7), (-0.8, 2.5), (1.0, 0.0), (0.0, 4.0))
        s = sqrt(1 - μ^2)
        ref = 1 + 0.05 * (3μ^2 - 1) / 2 + 0.1 * sqrt(3) / 2 * s^2 * cos(2φ) +
              0.03 * sqrt(1 / 6) * 1.5 * (5μ^2 - 1) * s * sin(φ)
        @test CompactLES.drive_factor(dh, μ, φ, 1.0) ≈ ref
    end
    @test CompactLES.drive_factor(dh, 0.3, 1.0) ≈ 1 + 0.05 * (3 * 0.09 - 1) / 2   # m = 0 only
    @test_throws ArgumentError AblationDrive(ts, Tr; harmonics=H, modes=zeros(3, 5))
    @test_throws ArgumentError AblationDrive(ts, Tr; harmonics=zeros(5, 5))
    @test_throws ArgumentError AblationDrive(ts, Tr; geometry=:rays)
    # On a 1-D spherical grid the binned form is the per-line form, to rounding
    sph = Problem(eos=eos, metric=SphericalMetric(), domain=((0.0, 1.0), (π / 2, π / 2 + 1), (0.0, 1.0)),
                  bcs=((OriginBC(), SlipWallBC()), PeriodicBC(), PeriodicBC()), ic=(r, θ, φ) -> shell(r))
    for driver in (0, 2)
        # (relaxation: the binned conversion is capped at ρₐ/relaxation per point, far
        # above the unit system's ṁ here only for a tiny relaxation time)
        kw = (; units=unit, layer=0.02, driver, relaxation=1e-12)
        _, _, a = rhs(sph, (200, 1, 1), AblationDrive(ts, Tr; kw...))
        _, _, b = rhs(sph, (200, 1, 1), AblationDrive(ts, Tr; kw..., geometry=:binned))
        @test maximum(abs, parent(a) .- parent(b)) <= 1e-12 * maximum(abs, parent(a))
    end
    # r–z (axis): the ablated mass is ṁ per unit front area, R_f ≈ 0.79 at every angle;
    # θ spans 1 rad, so the grid holds the solid angle 2
    cyl = Problem(eos=eos, metric=CylindricalMetric(), domain=((0.0, 1.0), (0.0, 1.0), (-1.0, 1.0)),
                  bcs=((AxisBC(), SlipWallBC()), PeriodicBC(), (SlipWallBC(), SlipWallBC())),
                  ic=(r, θ, z) -> shell(sqrt(r^2 + z^2)))
    db = AblationDrive(ts, Tr; units=unit, layer=0.02, driver=2, bins=(32, 1), relaxation=1e-12)
    s3, Q3, dQ3 = rhs(cyl, (100, 1, 200), db)
    o1, o3 = s3.decomp.n_halo_d[1], s3.decomp.n_halo_d[3]
    M = sum(dQ3[o1+i, 1, o3+k, 1] * CompactLES.xcoord(s3, 1, i) * s3.h[1] * s3.h[3]
            for i in 1:100, k in 1:200)
    G = CompactLES._bin_geometry(db, s3)
    bins = CompactLES._ablation_bins(db, ((s3, Q3, G),), 32, Float64, nothing, 3.0)
    @test all(R -> 0.78 < R < 0.80, bins.Rf)
    Rf2 = sum(bins.Rf .^ 2) / 32
    @test M ≈ -CompactLES.ablation_rate(db, 3.0) * 2 * Rf2 rtol = 2e-2
    @test all(iszero, dQ3[:, :, :, 3:5])                 # the driver form moves no momentum
    # a P2 drive: the source's pressure follows (1 + a₂P₂)^(3.3/4) at each cell's angle
    M2 = [0.0 0 0 0 0; 0.2 0.2 0.2 0.2 0.2]
    dp = AblationDrive(ts, Tr; units=unit, layer=0.02, driver=0, modes=M2, bins=(32, 1),
                       relaxation=1e-12)
    s4, Q4, dQ4 = rhs(cyl, (100, 1, 200), dp)
    m3 = s4.equations.i_mom[3]
    pole = sum(dQ4[o1+i, 1, o3+k, m3] for i in 1:10, k in 1:200 if CompactLES.xcoord(s4, 3, k) > 0)
    @test pole < 0                                       # the z force points inward at the pole
    # 3-D Cartesian: the body force is radial and inward, with no net force for a
    # centred capsule; a Y22 drive pushes harder along ±x than along ±y
    car = Problem(eos=eos, metric=CartesianMetric(), domain=((-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0)),
                  bcs=((SlipWallBC(), SlipWallBC()), (SlipWallBC(), SlipWallBC()), (SlipWallBC(), SlipWallBC())),
                  ic=(x, y, z) -> shell(sqrt(x^2 + y^2 + z^2)))
    H22 = zeros(9, length(ts))
    H22[1, :] .= 1
    H22[9, :] .= 0.2
    d3 = AblationDrive(ts, Tr; units=unit, layer=0.05, driver=0, harmonics=H22, bins=(8, 16),
                       relaxation=1e-12)
    s5, Q5, dQ5 = rhs(car, (40, 40, 40), d3)
    o = s5.decomp.n_halo_d
    m1, m2, m3 = s5.equations.i_mom
    x(i) = CompactLES.xcoord(s5, 1, i)
    Fx = [dQ5[o[1]+i, o[2]+j, o[3]+k, m1] for i in 1:40, j in 1:40, k in 1:40]
    Fy = [dQ5[o[1]+i, o[2]+j, o[3]+k, m2] for i in 1:40, j in 1:40, k in 1:40]
    # (measured 0.8e-5 and 3e-5: nodes on the diagonals sit on bin edges, where rounding
    # bins a node and its mirror image differently)
    @test abs(sum(Fx)) < 1e-4 * sum(abs, Fx) && abs(sum(Fy)) < 1e-4 * sum(abs, Fy)
    @test all(i -> Fx[i, 20, 20] * x(i) <= 0, 1:40)       # inward along x
    ax, ay = sum(abs, Fx[:, 20:21, 20:21]), sum(abs, Fy[20:21, :, 20:21])
    @test ax / ay ≈ ((1 + 0.2 * sqrt(3) / 2) / (1 - 0.2 * sqrt(3) / 2))^(3.3 / 4) rtol = 0.05
    # unsupported layouts
    @test_throws ArgumentError rhs(Problem(eos=eos, metric=CartesianMetric(),
                                           domain=((-1.0, 1.0), (-1.0, 1.0), (0.0, 1.0)),
                                           bcs=((SlipWallBC(), SlipWallBC()), (SlipWallBC(), SlipWallBC()), PeriodicBC()),
                                           ic=(x, y, z) -> shell(hypot(x, y))), (32, 32, 1), d3)
    @test_throws ArgumentError rhs(cyl, (100, 1, 200),
                                   AblationDrive(ts, Tr; units=unit, center=(0.1, 0.0, 0.0)))
end
