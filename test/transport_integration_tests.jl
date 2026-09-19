module TransportIntegrationTests

using MPI
MPI.Initialized() || MPI.Init(threadlevel=:funneled)
using CompactLES, Test
const CL = CompactLES
const PER = (PeriodicBC(), PeriodicBC())

# Independent literal Ar fits from trans.inp, 200--1000 K, in SI units.
argon_mu(T) = 1e-7 * exp(0.61205763log(T) - 67.714354/T +
                         190.40660/T^2 + 2.1588272)
argon_kappa(T) = 1e-4 * exp(0.60968928log(T) - 70.892249/T +
                            584.20624/T^2 + 1.9337152)

@testset "CEA wall heat flux uses the wall temperature" begin
    eos = Nasa9Mixture(["Ar"])
    tr = CeaTransport(eos)
    for isothermal in (false, true), ka in (false, true)
        walls = isothermal ? (NoSlipWallBC(Twall=300), NoSlipWallBC(Twall=600)) :
                             (NoSlipWallBC(), NoSlipWallBC())
        s = Solver(n_global=(32, 1, 1), L_domain=(0.01, 1.0, 1.0),
                   bcs=(walls, PER, PER), eos=eos, transport=tr,
                   art=ArtParams(enabled=false), filter_interval=0)
        Q = allocate_state(s)
        initialize!(s, Q, (x, y, z) -> Prim(rho=1.0, T_ion=300 + 30000x))
        CL.FORCE_KA[] = ka
        try
            apply_bcs!(s, Q)
            compute_rhs!(s, Q, zero(Q))
        finally
            CL.FORCE_KA[] = false
        end
        for (i, temp) in ((1, 300), (32, 600))
            I = gidx(s, i, 1, 1)
            @test s.flux[1, 1][I] == 0
            expected = isothermal ? -argon_kappa(temp) * 30000 : 0.0
            @test s.flux[1, s.equations.i_energy][I] ≈ expected atol=2e-9
        end
    end
end

@testset "CEA viscosity reaches curvilinear momentum sources" begin
    eos = Nasa9Mixture(["Ar"])
    tr = CeaTransport(eos)
    a = 0.3
    # On a shell, u_theta=a*r^2 (cylindrical) or u_phi=a*r^2 at the
    # equator (spherical) has tau_r,tangent = mu(T)*a*r. The geometric
    # tangent-momentum source alone is therefore mu(T)*a.
    for metric in (CylindricalMetric(), SphericalMetric()), ka in (false, true)
        spherical = metric isa SphericalMetric
        tangent = spherical ? 3 : 2
        s = Solver(n_global=(32, 1, 1), L_domain=(1.0, 1.0, 1.0),
                   origin=(1.0, pi/2, 0.0),
                   bcs=((ExtrapolationBC(), ExtrapolationBC()), PER, PER),
                   eos=eos, transport=tr, metric=metric,
                   art=ArtParams(enabled=false), filter_interval=0)
        Q = allocate_state(s)
        initialize!(s, Q, (r, theta, phi) ->
            Prim(rho=1.0, T_ion=300 + 100r,
                 u=spherical ? (0.0, 0.0, a*r^2) : (0.0, a*r^2, 0.0)))
        dQ = zero(Q)
        CL.FORCE_KA[] = ka
        try
            CL.compute_primitives_and_gradients!(s, Q)
            CL.add_metric_sources!(s, dQ, Q, metric)
        finally
            CL.FORCE_KA[] = false
        end
        for i in (8, 16, 24)
            I = gidx(s, i, 1, 1)
            r = xcoord(s, 1, i)
            @test dQ[I, s.equations.i_mom[tangent]] ≈ a*argon_mu(300 + 100r) rtol=1e-8
        end
    end
end

@testset "CEA dissipation diagnostic samples local viscosity" begin
    eos = Nasa9Mixture(["Ar"])
    tr = CeaTransport(eos)
    # A periodic transverse shear has tau:grad(u) = mu(T)*(du_y/dx)^2.
    for patch_grid in ((1, 1, 1), (2, 1, 1))
        s = Solver(n_global=(64, 1, 1), L_domain=(1.0, 1.0, 1.0),
                   bcs=(PER, PER, PER), eos=eos, transport=tr,
                   patch_grid=patch_grid, art=ArtParams(enabled=false),
                   filter_interval=0)
        Q = allocate_state(s)
        initialize!(s, Q, (x, y, z) ->
            Prim(rho=1.0, T_ion=500 + 100cospi(2x), u=(0.0, sinpi(2x), 0.0)))
        if Q isa Vector
            sync_patches!(s, Q)
        end
        measured = dissipation_rate(s, Q)
        # Independent fine periodic quadrature of the analytic integrand.
        reference = sum(argon_mu(500 + 100cospi(2i/4096)) *
                        (2pi*cospi(2i/4096))^2 for i in 0:4095) / 4096
        @test measured ≈ reference rtol=2e-5
    end
end

end
