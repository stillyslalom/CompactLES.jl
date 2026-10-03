# The positivity limiter of `Numerics(positivity_limiter = true)`: the face
# form of the divergence it rests on, a run it never acts in (bitwise the
# unlimited run), a strong shock in one dimension that it keeps admissible,
# and the bound it takes from the state entering `run!`; on the radial lines
# folded at the cylindrical axis and the spherical origin, the face form with
# the fold's face value, a run it never acts in, a blast it keeps admissible
# while conserving, and the Noh implosions under strict validity. The rejected
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

# A radial line of the unit interval folded at r = 0: the spherical origin
# (θ at π/2) or the cylindrical axis, a slip wall outside.
function pos_radial(metric; N, ic=(r, a, b) -> Prim(rho=1.0, u=(0.0, 0.0, 0.0),
                                                    p=1 + 1e-2 * exp(-20r^2)),
                    kw...)
    sphere = metric isa SphericalMetric
    prob = Problem(eos=POS_GAS, transport=ConstantTransport(mu0=0.0), metric=metric,
                   domain=((0.0, 1.0), sphere ? (π / 2, π / 2 + 1) : (0.0, 1.0),
                           (0.0, 1.0)),
                   bcs=((sphere ? OriginBC() : AxisBC(), SlipWallBC()), per3[2], per3[3]),
                   ic=ic)
    return setup(prob, Numerics(; n_global=(N, 1, 1), kw...))
end

# The r-z plane over the unit square, folded at the axis and by a symmetry
# plane at z = 0, slip walls outside.
function pos_rz(; N, ic, kw...)
    prob = Problem(eos=POS_GAS, transport=ConstantTransport(mu0=0.0),
                   metric=CylindricalMetric(), domain=((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)),
                   bcs=((AxisBC(), SlipWallBC()), per3[2], (SymmetryPlaneBC(), SlipWallBC())),
                   ic=ic)
    return setup(prob, Numerics(; n_global=(N, 1, N), kw...))
end

# The fold's face value of the running sum of `div`, the divergence of `f`
# with parity σ, as a stage takes it.
function pos_fold_face(lim, f, div, σ, halo)
    anchor = zeros(1, 1, 1)
    CL._limiter_anchor_point!(anchor, f, div, lim.weights[1], 0.0, 1.0, 1, 1, false,
                              true, σ, 0, lim.deriv_lhs, lim.deriv_rhs, halo..., 1, 1, 1)
    return anchor[1]
end

# A deterministic field in [−½, ½), from a linear congruential generator.
function pos_noise(N)
    s = UInt64(0x9e3779b97f4a7c15)
    return [(s = s * 0x5851f42d4c957f2d + 0x14057b7ef767814f; (s >> 11) / 2.0^53 - 0.5)
            for _ in 1:N]
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

    @testset "a limited blast conserves on a periodic grid" begin
        # Periodic in both directions, so W = h and the corrections conserve
        # the plain sums; the filter conserves there too.
        prob = Problem(eos=POS_GAS, transport=ConstantTransport(mu0=0.0),
                       domain=((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)), bcs=per3,
                       ic=(x, y, z) -> Prim(rho=1.0, u=(0.0, 0.0, 0.0),
                                            p=1e-3 + exp(-((x - 0.5)^2 + (y - 0.5)^2) /
                                                         0.004)))
        s, Q = setup(prob, Numerics(n_global=(48, 48, 1), cfl=0.4,
                                    art=ArtificialProperties(enabled=true),
                                    control=StepControl(validity=:permissive),
                                    positivity_limiter=true))
        total(c) = sum(Q[CL.padded_index(s, i, j, 1), c] for i in 1:48, j in 1:48)
        mass0, energy0 = total(1), total(5)
        bad = pos_bad_points(s, Q; tfinal=1.0, nmax=40)
        counts = CL.positivity_counts(s)
        @test counts.stage_limited > 0 && counts.filter_limited > 0
        @test bad == 0
        @test abs(total(1) - mass0) / mass0 < 1e-13
        @test abs(total(5) - energy0) / energy0 < 1e-13
    end

    @testset "face form of a radial line folded at r = 0" begin
        # The running sum from the fold's face value returns the far node's
        # flux, and the interior face relation holds at every face from the
        # fold up, the mirror taking F̂_{−s} = σ F̂_s and f_{1−l} = σ f_l.
        N = 96
        worst_far = 0.0
        worst_relation = 0.0
        for metric in (SphericalMetric(), CylindricalMetric()),
            deriv in (lele_d1_6(), lele_d1_8(), lele_d1_10())
            s, _ = pos_radial(metric; N, deriv, positivity_limiter=true)
            lim = getfield(s, :positivity)
            o = s.decomp.n_halo_d[1]
            idx(i) = CL.padded_index(s, i, 1, 1)
            r = [CL.xcoord(s, 1, i) for i in 1:N]
            lhs, rhs = lim.deriv_lhs, lim.deriv_rhs
            for σ in (1, -1), g in (cos.(3 .* r) .* (σ > 0 ? 1 : r) .+ r .^ 2,
                                    pos_noise(N))
                f = zero(s.tmp_a)
                for i in 1:N
                    f[idx(i)] = g[i]
                end
                out = zero(f)
                CL.div_along!(out, f, s, 1, σ)
                F0 = pos_fold_face(lim, f, out, σ, s.decomp.n_halo_d)
                F = F0 .+ [0.0; cumsum([lim.weights[1][o+i] * out[idx(i)] for i in 1:N])]
                worst_far = max(worst_far, abs(F[end] - g[N]))
                face(k) = k >= 0 ? F[k+1] : σ * F[1-k]
                node(l) = l >= 1 ? g[l] : σ * g[1-l]
                for k in 0:N÷2
                    left = face(k) + sum(lhs[q] * (face(k - q) + face(k + q))
                                         for q in eachindex(lhs))
                    right = sum(rhs[m] * sum(node(k + l) for l in 1-m:m)
                                for m in eachindex(rhs))
                    worst_relation = max(worst_relation, abs(left - right))
                end
            end
            # The face value of an even field is the Shu–Osher flux function at
            # r = 0: 1 for a constant, −h²/12 for r².
            h = s.h[1]
            for (g, F0) in ((ones(N), 1.0), (r .^ 2, -h^2 / 12))
                f = zero(s.tmp_a)
                for i in 1:N
                    f[idx(i)] = g[i]
                end
                out = zero(f)
                CL.div_along!(out, f, s, 1, 1)
                @test pos_fold_face(lim, f, out, 1, s.decomp.n_halo_d) ≈ F0 rtol = 1e-9
            end
        end
        @test worst_far < 1e-12
        @test worst_relation < 1e-12
    end

    @testset "a radial run it never acts in is the unlimited run" begin
        for metric in (SphericalMetric(), CylindricalMetric())
            runs = map((false, true)) do on
                s, Q = pos_radial(metric; N=64, positivity_limiter=on,
                                  art=ArtificialProperties(enabled=true))
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
    end

    @testset "a limited blast through the origin conserves and stays admissible" begin
        # Σ W J ρ and Σ W J E change only through the outer face, where the
        # slip wall carries none: the spherical mass and energy fluxes are odd
        # at the origin, so the fold's face carries none either, and a limited
        # filter pass weights both by J.
        blast(r, θ, φ) = Prim(rho=1.0, u=(0.0, 0.0, 0.0), p=1e-4 + exp(-(r / 0.08)^2))
        s, Q = pos_radial(SphericalMetric(); N=96, ic=blast, cfl=0.3,
                          art=ArtificialProperties(enabled=true),
                          control=StepControl(validity=:permissive),
                          positivity_limiter=true)
        lim = getfield(s, :positivity)
        o = s.decomp.n_halo_d[1]
        total(c) = sum(lim.weights[1][o+i] / s.inv_J[CL.padded_index(s, i, 1, 1)] *
                       Q[CL.padded_index(s, i, 1, 1), c] for i in 1:96)
        mass0, energy0 = total(1), total(5)
        bad = pos_bad_points(s, Q; tfinal=1.0, nmax=60)
        counts = CL.positivity_counts(s)
        @test counts.stage_limited > 0 && counts.filter_limited > 0
        @test bad == 0
        @test abs(total(1) - mass0) / mass0 < 1e-13
        @test abs(total(5) - energy0) / energy0 < 1e-13
    end

    @testset "Noh through the axis and the origin under strict validity" begin
        # The validation battery's configurations at a third of its resolution,
        # under the default strict validity, which rejects an inadmissible state.
        for ν in (2, 3)
            prob = noh_problem(ν; N=96)
            s, Q = setup(prob, Numerics(n_global=(96, 1, 1), cfl=NOH_CFL,
                                        art=ArtificialProperties(enabled=true),
                                        filter=StateFilter(compact_filter(); cfl=0.35),
                                        positivity_limiter=true))
            run!(s, Q; tfinal=NOH_T - Dict(NOH_T0)[ν])
            @test s.t ≈ NOH_T - Dict(NOH_T0)[ν]
            counts = CL.positivity_counts(s)
            @test counts.stage_limited > 0
            report = state_report(s, Q)
            @test report.inadmissible == 0 && report.negative_density == 0
        end
    end

    @testset "an r-z run it never acts in is the unlimited run" begin
        pulse(r, θ, z) = Prim(rho=1.0, u=(0.0, 0.0, 0.0),
                              p=1 + 1e-2 * exp(-20 * (r^2 + (z - 0.2)^2)))
        runs = map((false, true)) do on
            s, Q = pos_rz(; N=48, ic=pulse, positivity_limiter=on,
                          art=ArtificialProperties(enabled=true))
            run!(s, Q; tfinal=1.0, nmax=30)
            (s, Q)
        end
        (s0, Q0), (s1, Q1) = runs
        counts = CL.positivity_counts(s1)
        @test counts.stage_faces > 0 && counts.filter_faces > 0
        @test counts.stage_limited == 0 && counts.filter_limited == 0
        @test s0.step == s1.step && s0.t == s1.t
        @test parent(Q0) == parent(Q1)
    end

    @testset "an r-z blast at the corner of the axis and the plane stays admissible" begin
        # The blast sits on the corner cell, so its first steps limit the axis
        # face, the plane face and the corner together.
        blast(r, θ, z) = Prim(rho=1.0, u=(0.0, 0.0, 0.0),
                              p=1e-4 + exp(-(r^2 + z^2) / 0.08^2))
        s, Q = pos_rz(; N=48, ic=blast, cfl=0.3, art=ArtificialProperties(enabled=true),
                      control=StepControl(validity=:permissive), positivity_limiter=true)
        bad = pos_bad_points(s, Q; tfinal=1.0, nmax=40)
        counts = CL.positivity_counts(s)
        @test counts.stage_limited > 0 && counts.filter_limited > 0
        @test bad == 0
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
