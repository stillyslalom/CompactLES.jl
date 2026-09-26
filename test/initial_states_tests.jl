# Initial states: composition by name, the shock relations, and `Layers`.
#
# Serial. Included by runtests.jl, and runnable on its own:
#
#   julia --project=. test/initial_states_tests.jl

using MPI
MPI.Initialized() || MPI.Init(threadlevel=:funneled)
using CompactLES
using Test

@testset "shock relations reproduce the constant-γ closed forms" begin
    γ = 1.4
    eos = IdealMixture([IdealSpecies("a"; R=287.0, gamma=γ),
                        IdealSpecies("b"; R=57.4, gamma=5 / 3)])
    pre = Prim(Y=(1.0, 0.0), p=1e5, T_ion=300.0)
    c1 = thermodynamic_state(eos, pre).c
    for M in (1.2, 2.0, 5.0)
        jump = shock_jump(eos, pre, M)
        post = thermodynamic_state(eos, jump.post)
        @test post.p / 1e5 ≈ 1 + 2γ / (γ + 1) * (M^2 - 1) rtol = 1e-12
        ratio = (γ + 1) * M^2 / ((γ - 1) * M^2 + 2)
        @test post.rho / thermodynamic_state(eos, pre).rho ≈ ratio rtol = 1e-12
        @test jump.velocity ≈ M * c1 * (1 - 1 / ratio) rtol = 1e-12
        @test jump.shock_speed ≈ M * c1 rtol = 1e-14
        # The mirror image travels the other way at the same strength.
        back = shock_jump(eos, pre, M; direction=-1)
        @test back.velocity ≈ -jump.velocity rtol = 1e-12
    end
    M = 2.0
    p41 = (1 + 2γ / (γ + 1) * (M^2 - 1)) *
          (1 - (γ - 1) / (γ + 1) * (M - 1 / M))^(-2γ / (γ - 1))
    driver = driver_pressure(eos, Prim(Y=(1.0, 0.0), p=1e5, T_ion=300.0), pre, M)
    @test driver.p / 1e5 ≈ p41 rtol = 1e-9
    # The reflection of a Mach 2 shock in a γ = 1.4 gas is Mach √3.
    reflection = reflected_shock(eos, shock_jump(eos, pre, M).post)
    @test reflection.Mach ≈ sqrt(3) rtol = 1e-10
    @test abs(reflection.post.u[1]) < 1e-9
    @test_throws ArgumentError shock_jump(eos, pre, 0.9)
end

@testset "a calorically imperfect jump conserves mass, momentum and energy" begin
    eos = Nasa9Mixture(["N2", "He"])
    pre = Prim(Y=(1.0, 0.0), p=101_325.0, T_ion=300.0, u=(20.0, 0.0, 0.0))
    s1 = thermodynamic_state(eos, pre)
    jump = shock_jump(eos, pre, 3.0)
    s2 = thermodynamic_state(eos, jump.post)
    # Shock-frame velocities of the gas on either side.
    v1 = s1.u[1] - jump.shock_speed
    v2 = s2.u[1] - jump.shock_speed
    @test s1.rho * v1 ≈ s2.rho * v2 rtol = 1e-10
    @test s1.p + s1.rho * v1^2 ≈ s2.p + s2.rho * v2^2 rtol = 1e-10
    @test s1.h + v1^2 / 2 ≈ s2.h + v2^2 / 2 rtol = 1e-10
    tube = shock_tube(eos, Prim(Y=(0.0, 1.0), p=1e5, T_ion=300.0),
                      Prim(Y=(1.0, 0.0), p=101_325.0, T_ion=300.0), 2.0)
    @test tube.driver.p > tube.shocked.p > tube.driven.p
    @test tube.reflected.T_ion > tube.shocked.T_ion
end

@testset "mass_fractions by name" begin
    eos = Nasa9Mixture(["He", "CO2"])
    W = (4.002602, 44.0095)
    Y = mass_fractions(eos, "He" => 0.5, "CO2" => 0.5; basis=:mole)
    @test Y[1] ≈ W[1] / (W[1] + W[2]) rtol = 1e-4
    @test sum(Y) ≈ 1 atol = 1e-14
    @test mole_fractions(eos, Y)[1] ≈ 0.5 rtol = 1e-12
    @test mass_fractions(eos, "CO2" => 1.0; basis=:mass) == (0.0, 1.0)
    @test_throws ArgumentError mass_fractions(eos, "Ar" => 1.0; basis=:mole)
    @test_throws ArgumentError mass_fractions(eos, "He" => 0.5; basis=:mole)
    @test_throws UndefKeywordError mass_fractions(eos, "He" => 1.0)
end

@testset "Layers: pressure equilibrium, painter's order, :X output" begin
    eos = Nasa9Mixture(["N2", "SF6"])
    air = Prim(Y=(1.0, 0.0), p=101_325.0, T_ion=300.0)
    heavy = Prim(Y=(0.0, 1.0), p=101_325.0, T_ion=300.0)
    jump = shock_jump(eos, air, 1.5)
    bcs = ((SlipWallBC(), SlipWallBC()), PeriodicBC(), PeriodicBC())
    prob = Problem(eos=eos, domain=((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)), bcs=bcs,
                   ic=Layers(air, Slab(1, hi=0.2) => jump.post,
                             Slab(1, lo=0.6) => heavy;
                             width=Cells(1)))
    solver, Q = setup(prob, Numerics(n_global=(128, 1, 1)))
    x, p = line_profile(solver, Q, :p)
    _, T = line_profile(solver, Q, :T_ion)
    _, rho = line_profile(solver, Q, :rho)
    _, X2 = line_profile(solver, Q, :X; species=2)
    _, Y2 = line_profile(solver, Q, :Y; species=2)
    # Across the N2/SF6 interface both gases are at 101325 Pa and 300 K, and
    # the volume mixing keeps both through the transition.
    interface = findall(xi -> 0.4 < xi < 0.8, x)
    @test maximum(abs.(p[interface] .- 101_325.0)) < 1e-8 * 101_325.0
    @test maximum(abs.(T[interface] .- 300.0)) < 1e-8 * 300.0
    # Far from every transition each region carries its own state.
    s_post = thermodynamic_state(eos, jump.post)
    @test rho[1] ≈ s_post.rho rtol = 1e-12
    @test p[1] ≈ s_post.p rtol = 1e-12
    @test rho[end] ≈ thermodynamic_state(eos, heavy).rho rtol = 1e-12
    # Half the volume at the interface is SF6.
    mid = argmin(abs.(x .- 0.6))
    @test X2[mid] ≈ 0.5 atol = 0.1
    @test all(i -> X2[i] ≈ mole_fractions(eos, (1 - Y2[i], Y2[i]))[2], eachindex(x))

    # Two regions and no keywords is a call a positional constructor could claim.
    @test Layers(air, Slab(1, hi=0.2) => heavy, Slab(1, lo=0.6) => heavy).width ==
          Cells(3)

    # A later region covers an earlier one where they overlap.
    covered = Layers(air, Slab(1, hi=0.5) => heavy, Slab(1, hi=0.25) => jump.post;
                     width=Cells(1))
    s2, Q2 = setup(Problem(eos=eos, domain=((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)),
                           bcs=bcs, ic=covered), Numerics(n_global=(64, 1, 1)))
    _, Y2b = line_profile(s2, Q2, :Y; species=2)
    @test Y2b[1] < 1e-10
    @test Y2b[24] ≈ 1 atol = 1e-5
    @test Y2b[end] == 0.0
end

@testset "shapes ignore collapsed directions" begin
    eos = Nasa9Mixture(["N2", "He"])
    air = Prim(Y=(1.0, 0.0), p=1e5, T_ion=300.0)
    helium = Prim(Y=(0.0, 1.0), p=1e5, T_ion=300.0)
    per = PeriodicBC()
    # The collapsed z coordinate is far from the sphere's center, which a
    # shape reading it would take to mean the bubble is not in this plane.
    prob = Problem(eos=eos, domain=((0.0, 1.0), (0.0, 1.0), (5.0, 6.0)),
                   bcs=(per, per, per),
                   ic=Layers(air, Sphere((0.5, 0.5, 0.0), 0.2) => helium,
                             Box((0.0, 0.0, 0.0), (0.1, 0.1, 0.0)) => helium;
                             width=0.005))
    solver, Q = setup(prob, Numerics(n_global=(32, 32, 1)))
    Yhe = field_array(solver, Q, :Y; species=2)
    @test Yhe[gidx(solver, 17, 17, 1)] ≈ 1 atol = 1e-12
    # On the box's boundary half the volume is helium, a mole fraction of 1/2
    # and a mass fraction of W_He / (W_He + W_N2).
    Xhe = field_array(solver, Q, :X; species=2)
    @test Xhe[gidx(solver, 1, 1, 1)] ≈ 0.5 rtol = 1e-12
    @test Yhe[gidx(solver, 17, 30, 1)] < 1e-12
    @test signed_distance(Sphere((0.0, 0.0, 0.0), 1.0), (2.0, 0.0, 9.0),
                          (true, false, false)) ≈ 1.0
    @test signed_distance(!Sphere((0.0, 0.0, 0.0), 1.0) ∩ Box((-2, -2, -2), (2, 2, 2)),
                          (1.5, 0.0, 0.0), (true, true, true)) ≈ -0.5
end

@testset "boundary conditions from a Prim" begin
    eos = Nasa9Mixture(["N2", "He"])
    air = Prim(Y=(1.0, 0.0), p=1e5, T_ion=300.0)
    post = shock_jump(eos, air, 1.3).post
    inflow = NSCBCInflowBC(post; eta_u=1.0)
    @test inflow.T_ion == post.T_ion && inflow.u[1] == post.u[1]
    @test inflow.eta_u == 1.0
    @test DirichletBC(post).fun(0.0, 0.0, 0.0, 1.0) === post
    @test_throws ArgumentError NSCBCInflowBC(Prim(Y=(1.0, 0.0), p=1e5, rho=1.0))
end

@testset "AMR accepts a static predicate and composite profiles" begin
    prob = Problem(domain=((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)),
                   bcs=(PeriodicBC(), PeriodicBC(), PeriodicBC()),
                   ic=(x, y, z) -> Prim(p=1.0, rho=1.0 + 0.1sin(2π * x)))
    num = Numerics(n_global=(96, 1, 1), filter_interval=0,
                   art=ArtParams(enabled=false),
                   amr=AMR(initial=(x, y, z) -> abs(x - 0.5) < 0.05))
    solver, states = setup(prob, num)
    @test nlevels(solver) == 2
    region = refined_region(solver)
    @test region.offset[1] < 48 < region.offset[1] + region.extent[1]
    x, rho = line_profile(solver, states, :rho)
    @test length(x) == 96
    @test maximum(abs.(rho .- (1.0 .+ 0.1sin.(2π .* x)))) < 1e-12
    @test_throws ArgumentError setup(prob, Numerics(n_global=(96, 1, 1),
        art=ArtParams(enabled=false),
        amr=AMR(initial=:sensor, tag_sensor_threshold=1.0)))
end

@testset "erf transitions and multimode interfaces" begin
    @test CompactLES._erf(0.5) ≈ 0.5204998778130465 rtol = 1e-15
    @test CompactLES._erf(2.0) ≈ 0.9953222650189527 rtol = 1e-15
    @test all(x -> abs(CompactLES._erf(x)) <= 1, range(-7, 7, length=1401))
    m = Multimode(lengths=(0.05,), modes=1:8, rms=5e-4, spectrum=n -> n^-2.0)
    ys = range(0, 0.05, length=4001)[1:end-1]
    @test sqrt(sum(abs2, m.(ys)) / length(ys)) ≈ 5e-4 rtol = 1e-10
    @test Multimode(lengths=(0.05,), modes=1:8, rms=5e-4)(0.01) ==
          Multimode(lengths=(0.05,), modes=1:8, rms=5e-4)(0.01)
    @test m(0.01) != Multimode(lengths=(0.05,), modes=1:8, rms=5e-4, seed=2,
                               spectrum=n -> n^-2.0)(0.01)
    # A surface band counts each ± pair of wave vectors once.
    @test length(Multimode(lengths=(1.0, 1.0), modes=1:1, rms=1.0).k) == 2
    @test_throws ArgumentError Layers(Prim(p=1.0, rho=1.0); profile=:cosine)
end

@testset "exact Riemann problem" begin
    eos = IdealSpecies("gas"; R=1.0, gamma=1.4)
    sod = riemann_interface(eos, Prim(p=1.0, rho=1.0), Prim(p=0.1, rho=0.125))
    # Toro, Riemann Solvers, Table 4.2, test 1.
    @test sod.p_star ≈ 0.30313 atol = 1e-5
    @test sod.u_star ≈ 0.92745 atol = 1e-5
    @test sod.left_wave === :rarefaction && sod.right_wave === :shock
    @test sod.right_speed ≈ 1.75216 atol = 1e-5
    # A shock refracting into a heavier gas reflects a shock.
    n9 = Nasa9Mixture(["N2", "SF6"])
    air = Prim(Y=(1.0, 0.0), p=101_325.0, T_ion=295.0)
    sf6 = Prim(Y=(0.0, 1.0), p=101_325.0, T_ion=295.0)
    r = riemann_interface(n9, shock_jump(n9, air, 1.5).post, sf6)
    @test r.left_wave === :shock && r.right_wave === :shock
    @test thermodynamic_state(n9, r.left).p ≈ thermodynamic_state(n9, r.right).p rtol = 1e-10
end

@testset "scheduled boundary switch and a ramped boundary state" begin
    eos = IdealMixture(IdealSpecies("air"; R=287.0, gamma=1.4))
    air = Prim(p=1e5, T_ion=300.0)
    jump = shock_jump(eos, air, 1.5)
    t_on = 2e-5
    fire = Ramp(eos, air, jump.post; start=t_on, duration=Cells(3),
                speed=jump.shock_speed)
    # The ramp is exact at both ends and moves monotonically between them.
    @test fire(0.0, 0.0, 0.0, t_on, 1e-3) === air
    @test fire(0.0, 0.0, 0.0, t_on + 3e-3 / jump.shock_speed, 1e-3) === jump.post
    mid = fire(0.0, 0.0, 0.0, t_on + 1.5e-3 / jump.shock_speed, 1e-3)
    @test 1e5 < mid.p < jump.post.p && 0 < mid.u[1] < jump.post.u[1]
    @test_throws ArgumentError Ramp(eos, air, jump.post; start=0.0, duration=Cells(3))

    face = SwitchableBC(SlipWallBC(), DirichletBC(fire); at=t_on)
    prob = Problem(eos=eos, domain=((0.0, 0.2), (0.0, 1.0), (0.0, 1.0)),
                   bcs=((face, SlipWallBC()), PeriodicBC(), PeriodicBC()),
                   ic=(x, y, z) -> air)
    solver, Q = setup(prob, Numerics(n_global=(64, 1, 1)))
    times = Float64[]
    run!(solver, Q; tfinal=4e-5, nmax=400,
         callback=(s, q) -> push!(times, s.t))
    # A step ends on the switch time, the face switched there, and the shock
    # entered: the pressure at the far wall's side has not yet risen but the
    # near end has.
    @test any(t -> t == t_on, times)
    @test switched(face)
    _, p = line_profile(solver, Q, :p)
    @test p[2] > 1.5e5
    @test p[end] ≈ 1e5 rtol = 1e-6
    # A rollback to before the switch time restores the earlier condition.
    CompactLES.rewind_scheduled_switches!(solver.bcs, 1e-5, 1e-20)
    @test !switched(face)
    # A switch already due when a run starts applies before its first step.
    CompactLES.apply_scheduled_switches!(solver.bcs, 3e-5, 0.0)
    @test switched(face)
end

@testset "AMR takes nested shapes, resolves defaults, and names its scope" begin
    per = PeriodicBC()
    prob = Problem(domain=((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)), bcs=(per, per, per),
                   ic=(x, y, z) -> Prim(p=1.0, rho=1.0 + 0.1sin(2π * x)))
    base = (n_global=(96, 1, 1), filter_interval=0, art=ArtParams(enabled=false))
    nested = AMR(initial=[Box((0.3, 0, 0), (0.7, 1, 1)), Box((0.45, 0, 0), (0.55, 1, 1))])
    solver, states = setup(prob, Numerics(; base..., amr=nested))
    @test nlevels(solver) == 3
    r1, r2 = level_regions(solver, 1)[1], level_regions(solver, 2)[1]
    h = 1 / 96
    @test r1.offset[1] * h <= 0.3 && (r1.offset[1] + r1.extent[1] - 1) * h >= 0.7
    @test r2.offset[1] * h / 3 <= 0.45 && (r2.offset[1] + r2.extent[1] - 1) * h / 3 >= 0.55
    @test getfield(solver, :regrid) === nothing
    # A shape reaching the boundary is refined up to the margin, with a warning.
    wide = AMR(initial=Box((0.0, 0, 0), (0.3, 1, 1)))
    s2, _ = @test_logs (:warn, r"domain boundary") match_mode = :any setup(prob,
        Numerics(; base..., amr=wide))
    @test level_regions(s2, 1)[1].offset[1] == 4
    # The sensor follows its feature by default, at an interval set by the CFL.
    step_prob = Problem(domain=((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)), bcs=(per, per, per),
                        ic=(x, y, z) -> Prim(p=1.0, rho=1 + 0.3exp(-((x - 0.5) / 0.03)^2)))
    s3, _ = setup(step_prob, Numerics(n_global=(96, 1, 1), cfl=0.5, amr=AMR()))
    @test getfield(s3, :regrid).interval == 4
    # Scope violations are argument errors naming AMR.
    @test_throws ArgumentError setup(prob, Numerics(; base..., amr=nested,
                                                    filt=pyranda_filter()))
    err = try
        setup(prob, Numerics(; base..., amr=nested, filt=pyranda_filter()))
    catch e
        e
    end
    @test occursin("AMR", err.msg) && occursin("compact_filter", err.msg)
    @test volume_integral(solver, states, :rho) ≈ 1.0 rtol = 1e-10
end

@testset "Hydrostatic: a stratified column stays at rest" begin
    # Heavy gas over light under gravity, slip walls along it. Equal γ keeps the
    # pressure linear in the conserved variables across the transition.
    eos = IdealMixture([IdealSpecies("heavy"; R=1.0, gamma=1.4),
                        IdealSpecies("light"; R=3.0, gamma=1.4)])
    heavy = Prim(Y=(1.0, 0.0), p=10.0, rho=3.0)
    light = Prim(Y=(0.0, 1.0), p=10.0, rho=1.0)
    layers = Layers(light, Slab(1, lo=0.5) => heavy)
    n = 64
    w = 3 / (n - 1)
    # The continuous hydrostatic pressure of the same density profile.
    ρ(x) = 1 + 2 * (1 + tanh((x - 0.5) / w)) / 2
    Ρ(x) = x + (x + w * log(cosh((x - 0.5) / w)))
    continuous = (x, y, z, h) -> begin
        s = CompactLES.BoundLayers(layers, eos, (true, false, false), 2)(x, y, z, h)
        Prim(Y=s.Y, p=10.0 - (Ρ(x) - Ρ(1.0)), rho=s.rho)
    end
    walls = (SlipWallBC(), SlipWallBC())
    force = ConstantBodyForce((-1.0, 0.0, 0.0))
    domain = ((0.0, 1.0), (0.0, 1.0), (0.0, 1.0))
    function column(ic; filter_interval=0, nmax=2000)
        prob = Problem(eos=eos, domain=domain, bcs=(walls, PeriodicBC(), PeriodicBC()),
                       ic=ic, sources=(force,))
        s, q = setup(prob, Numerics(n_global=(n, 1, 1), filter_interval=filter_interval,
                                    art=ArtParams(enabled=false)))
        run!(s, q; tfinal=1e9, nmax=nmax)
        _, u = line_profile(s, q, :u)
        return s, q, maximum(abs, u)
    end
    balanced = Hydrostatic(layers; p_ref=10.0, at=1.0)
    # About 16 acoustic crossings of the column.
    solver, Q, u_balanced = column(balanced)
    _, _, u_continuous = column(continuous)
    @test u_balanced < 1e-13
    @test u_continuous > 1e-9
    x, p = line_profile(solver, Q, :p)
    @test p[end] ≈ 10.0 rtol = 1e-12
    _, rho = line_profile(solver, Q, :rho)
    @test rho[1] ≈ 1.0 rtol = 1e-8
    @test rho[end] ≈ 3.0 rtol = 1e-8
    # The filter does not commute with the derivative at the closure rows, so a
    # filtered run holds the balance to the end residual, not to round-off.
    _, _, u_filtered = column(balanced; filter_interval=1, nmax=200)
    @test u_filtered < 1e-8

    # Gravity along y, on a plane, against no-slip walls.
    prob = Problem(eos=eos, domain=domain,
                   bcs=(PeriodicBC(), (NoSlipWallBC(), NoSlipWallBC()), PeriodicBC()),
                   ic=Hydrostatic(Layers(light, Slab(2, lo=0.5) => heavy);
                                  p_ref=10.0, at=0.0),
                   sources=(ConstantBodyForce((0.0, -1.0, 0.0)),))
    plane, Qp = setup(prob, Numerics(n_global=(12, 48, 1), filter_interval=0,
                                     art=ArtParams(enabled=false)))
    run!(plane, Qp; tfinal=1e9, nmax=200)
    m = maximum(I -> max(abs(Qp[I, 3]), abs(Qp[I, 4])),
                CompactLES.interior(plane.decomp))
    @test m < 1e-13
    @test field_array(plane, Qp, :p)[gidx(plane, 5, 1, 1)] ≈ 10.0 rtol = 1e-12

    bad(bcs, sources; kw...) =
        setup(Problem(eos=eos, domain=domain, bcs=bcs, sources=sources,
                      ic=Hydrostatic(layers; p_ref=10.0, at=get(kw, :at, 1.0))),
              Numerics(n_global=(n, 1, 1)))
    per = PeriodicBC()
    @test_throws ArgumentError bad((per, per, per), (force,))
    @test_throws ArgumentError bad((walls, per, per), ())
    @test_throws ArgumentError bad((walls, per, per), (force,); at=2.0)
    @test_throws ArgumentError bad((walls, per, per),
                                   (ConstantBodyForce((-1.0, 0.5, 0.0)),))
end
