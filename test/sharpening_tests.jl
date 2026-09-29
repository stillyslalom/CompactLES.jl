# The interface sharpening flux of `ArtificialProperties.C_sharpen`: the
# uniform (u, p, T) state it must leave uniform, the shocked interface it
# exists to hold thin, the smooth composition it must leave alone, the
# single-species run it must not touch, and its kernel path.
#
# Serial. Included by runtests.jl, and runnable on its own:
#
#   julia --project=. test/sharpening_tests.jl

if !@isdefined(CL)
    using MPI
    MPI.Init(threadlevel=:funneled)
    using CompactLES
    using Test
    using Printf
    const CL = CompactLES
    include(joinpath(@__DIR__, "references.jl"))
    include(joinpath(@__DIR__, "cases.jl"))
end
using CompactLES: padded_index

sh_eos(R) = IdealMixture([IdealSpecies{Float64}("light", 1.0, 1.4),
                          IdealSpecies{Float64}("heavy", 1 / R, 1.09)])

# A periodic line at uniform u, p and T holding the volume fraction `V(x)` of
# a heavy gas of density ratio `R`.
function sh_line(V; N=200, R=20.0, u=0.0, art)
    eos = sh_eos(R)
    h = 1.0 / N
    prob = Problem(eos=eos, transport=ConstantTransport(mu0=0.0),
                   domain=((0.0, 1.0), (0.0, h), (0.0, h)), bcs=per3,
                   ic=(x, y, z) -> begin
                       v = V(x)
                       ρ = (1 - v) + v * R
                       Prim(Y=((1 - v) / ρ, v * R / ρ), rho=ρ, u=(u, 0.0, 0.0),
                            p=1.0)
                   end)
    return setup(prob, Numerics(n_global=(N, 1, 1), art=art, cfl=0.4,
                                control=StepControl(validity=:permissive)))
end

@testset "sharpening flux: invariance, shocked interface, localization" begin
    sharp = ArtificialProperties(C_sharpen=0.5)

    # (a) A resting slab of density ratio 20 with two-cell edges at uniform
    # p, T and u = 0. Each species flux is ρ_k times a volume-fraction flux,
    # ρ_k R_k = p/T is common to every species, and the fractions and their
    # gradients sum to one and to zero pointwise, so Σ_k R_k S_k = 0 and the
    # flux moves the partial densities and leaves u, p and T at round-off.
    let slab(x) = CL.tanh_blend(x, 0.3, 2 / 200) - CL.tanh_blend(x, 0.7, 2 / 200)
        solver, Q = sh_line(slab; art=sharp)
        eqs = solver.equations
        nx = solver.decomp.n_local[1]
        Rk, cvk = solver.eos.Rk, solver.eos.cvk
        umax = Ref(0.0); dpmax = Ref(0.0); dTmax = Ref(0.0)
        function drift(s, Q)
            for i in 1:nx
                I = padded_index(s, i, 1, 1)
                ρ = Q[I, 1] + Q[I, 2]
                y = Q[I, 1] / ρ
                u = Q[I, eqs.i_mom[1]] / ρ
                T = (Q[I, eqs.i_energy] / ρ - u^2 / 2) / (y * cvk[1] + (1 - y) * cvk[2])
                umax[] = max(umax[], abs(u))
                dpmax[] = max(dpmax[], abs(ρ * (y * Rk[1] + (1 - y) * Rk[2]) * T - 1))
                dTmax[] = max(dTmax[], abs(T - 1))
            end
        end
        run!(solver, Q; tfinal=0.1, nmax=600, callback=drift)
        @test completed(solver, 0.1)
        @test umax[] < 1e-13
        @test dpmax[] < 1e-13
        @test dTmax[] < 1e-13
    end

    # (b) The Mach 1.5 shocked interface at density ratio 100. Without the
    # flux the mass fraction of the light gas spans ten cells between 0.05
    # and 0.95, most of it a tail of the heavy gas on the light side; at
    # `C_sharpen = 1` the flux removes the tail (measured 7 cells) at the
    # price of more ringing and a deeper undershoot (TV − 1 0.042 against
    # 0.0021, worst Y −0.040 against −0.0093, in 1927 steps against 701).
    let plain = shock_interface(rho_heavy=100.0, nmax=1500),
        sharpened = shock_interface(art=ArtificialProperties(C_sharpen=1.0),
                                    rho_heavy=100.0, nmax=3000)
        @test plain.completed && sharpened.completed
        @test all(isfinite, sharpened.rho) && minimum(sharpened.rho) > 0
        @test sharpened.width_cells <= plain.width_cells - 3
        @test sharpened.worst_min_Y > -0.06
        @test sharpened.worst_max_Y < 1.06
        @test sum(abs, diff(sharpened.Y_air)) - 1 < 0.08
    end

    # (c) A composition gradient resolved over many cells, V between 0.2 and
    # 0.8 on 64 points: its local logistic thickness V(1 − V)/|∇V| is at
    # least 8.5 cells, above the 3ε at which the gate closes, so the gate is
    # zero at every point and the right-hand side is the unsharpened one bit
    # for bit.
    let V(x) = 0.5 + 0.3 * sin(2π * x)
        s1, Q1 = sh_line(V; N=64, u=0.5, art=ArtificialProperties())
        s2, Q2 = sh_line(V; N=64, u=0.5, art=sharp)
        dQ1 = zero(Q1); dQ2 = zero(Q2)
        CL.compute_rhs!(s1, Q1, dQ1)
        CL.compute_rhs!(s2, Q2, dQ2)
        @test parent(dQ1) == parent(dQ2)
    end

    # (d) A single species has no interface: the run is the unsharpened one
    # bit for bit, the step rate included.
    let a = lax(), b = lax(; art=sharp)
        @test a[2] == b[2] && a[3] == b[3] && a[4] == b[4]
    end

    # (e) The KernelAbstractions path reproduces the threaded one bitwise.
    let slab(x) = CL.tanh_blend(x, 0.3, 2 / 64) - CL.tanh_blend(x, 0.7, 2 / 64)
        s1, Q1 = sh_line(slab; N=64, u=0.5, art=sharp)
        run!(s1, Q1; nmax=20, tfinal=1.0)
        local s2, Q2
        CL.FORCE_KA[] = true
        try
            s2, Q2 = sh_line(slab; N=64, u=0.5, art=sharp)
            run!(s2, Q2; nmax=20, tfinal=1.0)
        finally
            CL.FORCE_KA[] = false
        end
        @test s1.step == s2.step
        @test parent(Q1) == parent(Q2)
    end

    # (g) Two dimensions: an interface normal to x with a small cosine
    # perturbation in y. The transverse flux follows the transverse gradient,
    # a fraction of the normal one, and the normal flux matches the flux of
    # the unperturbed interface on the same grid.
    let
        function tilted(pert)
            eos = sh_eos(20.0)
            per = (PeriodicBC(), PeriodicBC())
            prob = Problem(eos=eos, transport=ConstantTransport(mu0=0.0),
                           domain=((0.0, 1.0), (0.0, 0.25), (0.0, 0.25)),
                           bcs=(per, per, per),
                           ic=(x, y, z) -> begin
                               xi = 0.5 + pert * cos(2π * y / 0.25)
                               θ = CL.tanh_blend(x, xi, 2 / 64) -
                                   CL.tanh_blend(x, 0.9, 2 / 64)
                               ρ = (1 - θ) + 20θ
                               Prim(Y=((1 - θ) / ρ, 20θ / ρ), rho=ρ, p=1.0)
                           end)
            s, Q = setup(prob, Numerics(n_global=(64, 16, 1), art=sharp))
            dQ = zero(Q)
            CL.compute_rhs!(s, Q, dQ)
            o, n = s.decomp.n_halo_d, s.decomp.n_local
            r = (o[1]+1:o[1]+n[1], o[2]+1:o[2]+n[2], 1:1)
            return maximum(abs, view(s.grad_Q[1, 3], r...)),
                   maximum(abs, view(s.grad_Q[2, 3], r...))
        end
        Sx0, Sy0 = tilted(0.0)
        Sx, Sy = tilted(0.25 / 64)
        @test Sy0 < 1e-12 * Sx0
        @test 0 < Sy < 0.5 * Sx
        @test isapprox(Sx, Sx0; rtol=0.2)
    end

    # (f) Setup rejections.
    per = ntuple(_ -> (PeriodicBC(), PeriodicBC()), 3)
    build(art; eos=sh_eos(5.0)) = Solver(n_global=(32, 1, 1), L_domain=(1.0, 1.0, 1.0),
                                         bcs=per, eos=eos, art=art)
    @test_throws ArgumentError build(ArtificialProperties(C_sharpen=-1.0))
    @test_throws ArgumentError build(ArtificialProperties(C_sharpen=0.5, sharpen_width=0.0))
    @test_throws ArgumentError build(ArtificialProperties(C_sharpen=0.5, species_flux=:bulk))
    @test_throws ArgumentError build(ArtificialProperties(C_sharpen=0.5, enabled=false))
    four = IdealMixture([IdealSpecies{Float64}("s$k", 1.0 / k, 1.4) for k in 1:4])
    @test_throws ArgumentError build(ArtificialProperties(C_sharpen=0.5); eos=four)
    @test build(ArtificialProperties(C_sharpen=0.5)) isa Solver
end
