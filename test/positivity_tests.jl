# The positivity limiter of `Numerics(positivity_limiter = true)`: the face
# form of the divergence it rests on, a run it never acts in (bitwise the
# unlimited run), a strong shock in one dimension that it keeps admissible,
# and the bound it takes from the state entering `run!`. The rejected
# configurations are in capability_tests.jl, the decomposed run in the MPI
# suite's "positivity limiter" phase.
#
# Serial. Included by runtests.jl, and runnable on its own:
#
#   julia --project=. test/positivity_tests.jl

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

const POS_GAS = IdealSpecies("gas"; gamma=1.4, R=1.0)

# A closed line's points with ρ ≤ 0 or ρe ≤ 0, counted after every step.
function pos_bad_points(solver, Q; tfinal, nmax=typemax(Int))
    bad = Ref(0)
    count_bad = (s, q) -> (r = state_report(s, q);
                           bad[] += r.negative_density + r.inadmissible; nothing)
    run!(solver, Q; tfinal, nmax, callback=count_bad)
    return bad[]
end

@testset "positivity limiter" begin
    @testset "face weights of the divergence" begin
        # Σ_i W_i (D f)_i = f_N − f_1 on every closed line and 0 on every
        # periodic one, for each direction's plan and a random f.
        bcs = ((SlipWallBC(), SlipWallBC()), (PeriodicBC(), PeriodicBC()),
               (NoSlipWallBC(), NoSlipWallBC()))
        for deriv in (lele_d1_6(), lele_d1_8(), lele_d1_6(closures=:brady_livescu))
            s = Solver(n_global=(56, 16, 52), L_domain=(1.0, 0.3, 1.1), bcs=bcs,
                       deriv=deriv, eos=POS_GAS, positivity_limiter=true)
            lim = getfield(s, :positivity)
            dec = s.decomp
            o = dec.n_halo_d
            f = similar(s.tmp_a)
            f .= sin.(LinearIndices(f)) .* 0.5
            out = zero(f)
            worst = 0.0
            for d in 1:3
                CL.exchange_dim!(f, dec, d)
                CL.div_along!(out, f, s, d, 1)
                n = dec.n_local[d]
                nA, nB = CL._transverse(dec.n_local, d)
                for b in 1:nB, a in 1:nA
                    node(p) = CL._line_node(d, p, a, b, o...)
                    S = sum(lim.weights[d][p+o[d]] * out[node(p)] for p in 1:n)
                    ref = dec.periodic[d] ? 0.0 : f[node(n)] - f[node(1)]
                    worst = max(worst, abs(S - ref))
                end
            end
            @test worst < 1e-12
            @test all(>(0), lim.weights[1]) && all(>(0), lim.weights[3])
        end
        # The wall node's weight under the default closure rows, which sets
        # the first-order bound there.
        W1 = getfield(Solver(n_global=(48, 1, 1), L_domain=(1.0, 1.0, 1.0),
                             bcs=((SlipWallBC(), SlipWallBC()), per3[2], per3[3]),
                             eos=POS_GAS, positivity_limiter=true),
                      :positivity).weights[1]
        @test W1[5] * 47 ≈ 0.2149 atol = 1e-4
    end

    @testset "a run it never acts in is the unlimited run" begin
        prob = Problem(eos=POS_GAS, transport=ConstantTransport(mu0=1e-4),
                       domain=((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)),
                       bcs=((SlipWallBC(), SlipWallBC()), per3[2], per3[3]),
                       ic=(x, y, z) -> Prim(rho=1.0, u=(0.0, 0.1, 0.0),
                                            p=1 + 1e-2 * exp(-((x - 0.3)^2 +
                                                                (y - 0.5)^2) / 0.01)))
        runs = map((false, true)) do on
            s, Q = setup(prob, Numerics(n_global=(48, 40, 1), positivity_limiter=on,
                                        art=ArtificialProperties(enabled=true)))
            run!(s, Q; tfinal=1.0, nmax=40)
            (s, Q)
        end
        (s0, Q0), (s1, Q1) = runs
        counts = CL.positivity_counts(s1)
        @test counts.stage_faces > 0 && counts.filter_faces > 0
        @test counts.stage_limited == 0 && counts.filter_limited == 0
        @test s0.step == s1.step && s0.t == s1.t
        @test parent(Q0) == parent(Q1)
    end

    @testset "a strong shock in one dimension stays admissible" begin
        # Woodward–Colella at a quarter of the battery's resolution: the
        # unlimited run leaves points with ρe ≤ 0 ahead of the fronts, the
        # limited one none, at the same closing density profile.
        N = 200
        h = 1.0 / (N - 1)
        prob = Problem(eos=POS_GAS, transport=ConstantTransport(mu0=0.0),
                       domain=((0.0, 1.0), (0.0, h), (0.0, h)),
                       bcs=((SlipWallBC(), SlipWallBC()), per3[2], per3[3]),
                       ic=(x, y, z) -> Prim(rho=1.0, u=(0.0, 0.0, 0.0),
                            p=1000 * (1 - tanh_blend(x, 0.1, 2h)) +
                              0.01 * (tanh_blend(x, 0.1, 2h) - tanh_blend(x, 0.9, 2h)) +
                              100 * tanh_blend(x, 0.9, 2h)))
        results = map((false, true)) do on
            s, Q = setup(prob, Numerics(n_global=(N, 1, 1), cfl=0.3,
                                        art=ArtificialProperties(enabled=true),
                                        filter=StateFilter(compact_filter(); cfl=0.35),
                                        control=StepControl(validity=:permissive),
                                        positivity_limiter=on))
            bad = pos_bad_points(s, Q; tfinal=WC_T)
            (s, Q, bad, case_line_profile(s, Q)[2])
        end
        (s0, Q0, bad0, ρ0), (s1, Q1, bad1, ρ1) = results
        @test bad0 > 0
        @test bad1 == 0
        @test s1.t == s0.t
        counts = CL.positivity_counts(s1)
        @test counts.stage_limited > 0 && counts.filter_limited > 0
        # The limiter acts ahead of the fronts; the profile moves little.
        @test sum(abs, ρ1 .- ρ0) / sum(abs, ρ0) < 0.02
        lim = getfield(s1, :positivity)
        @test lim.eps_rho ≈ 0.01 && lim.eps_e ≈ 0.01 * 0.01 / 0.4
    end

    @testset "the bound follows the state entering run!" begin
        prob = Problem(eos=POS_GAS, domain=((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)),
                       bcs=((SlipWallBC(), SlipWallBC()), per3[2], per3[3]),
                       ic=(x, y, z) -> Prim(rho=2 + sin(2π * x), u=(0.0, 0.0, 0.0),
                                            p=1.0))
        s, Q = setup(prob, Numerics(n_global=(64, 1, 1), positivity_limiter=true))
        run!(s, Q; tfinal=1.0, nmax=2)
        lim = getfield(s, :positivity)
        ρmin = minimum(Q[CL.padded_index(s, i, 1, 1), 1] for i in 1:64)
        @test lim.eps_rho == 0.01 * ρmin || lim.eps_rho ≈ 0.01 * ρmin
        # A continued run takes the bound again from the state it starts from.
        Q .*= 2
        run!(s, Q; tfinal=1.0, nmax=3)
        @test lim.eps_rho ≈ 0.02 * ρmin rtol = 0.05
    end
end
