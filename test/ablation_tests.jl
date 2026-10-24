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
    @test dQd[rows, 1, 1, ie] ≈ Yd .^ 2 .* (P .- sd.p[rows, 1, 1]) ./ ((5 / 3 - 1) * 0.1)
    @test_throws ArgumentError AblationDrive(ts, Tr; driver=1)
    # unsupported layouts are refused
    bad = Problem(eos=eos, metric=CylindricalMetric(), domain=((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)),
                  bcs=((AxisBC(), SlipWallBC()), PeriodicBC(), PeriodicBC()), ic=(r, θ, z) -> ic(r, θ, z))
    sb, Qb = setup(bad, Numerics(n_global=(64, 1, 1)))
    @test_throws ArgumentError CompactLES.add_source!(d1, sb, similar(Qb), Qb, 0.0)
end
