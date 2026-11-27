# The serial suite's tests of the artificial properties, their sensors and
# detectors, and the species channels. Included by test/serial_suite.jl, which loads
# the package and defines the imports and helpers used here; not runnable on its
# own.

@testset "multi-species artificial diffusivity responds to Y gradients" begin
    # artificial.jl computes per-species D* only when n_species > 1; that branch was
    # never executed, since the only multi-species test disables art.
    eos = IdealMixture([IdealSpecies{Float64}("a", 1.0, 1.4),
                        IdealSpecies{Float64}("b", 0.2, 1.09)])
    solver = build_solver(
        Problem(domain=((0.0, 1.0), (0.0, 0.2), (0.0, 0.2)), bcs=per3, eos=eos, ic=nothing),
        Numerics(n_global=(64, 12, 12), art=ArtificialProperties(enabled=true)))
    Q = allocate_state(solver)
    dQ = zero(Q)
    # smooth composition: D* must stay tiny
    initialize!(solver, Q, (x, y, z) -> Prim(Y=(0.5 + 0.1sin(2π * x), 0.5 - 0.1sin(2π * x)),
                                        p=1.0, rho=1.0))
    compute_rhs!(solver, Q, dQ)
    smooth_max = max(maximum(solver.D_art[1]), maximum(solver.D_art[2]))
    # sharp interface: D* must switch on
    initialize!(solver, Q, (x, y, z) -> begin
        θ = tanh_blend(x, 0.5, 0.01)
        Prim(Y=(1 - θ, θ), p=1.0, rho=1.0)
    end)
    compute_rhs!(solver, Q, dQ)
    sharp_max = max(maximum(solver.D_art[1]), maximum(solver.D_art[2]))
    @test sharp_max > 100 * max(smooth_max, 1e-14)
    @test all(isfinite, solver.D_art[1]) && all(isfinite, solver.D_art[2])
end

@testset "artificial-property weights are the local physical spacing" begin
    # Every sensor weight is h[d] / inv_h[d][I], the arclength of one
    # computational cell, and the mass-fraction bound is the geometric mean of
    # the active ones. Two grids carrying the same physical spacing must
    # therefore produce the same coefficients whatever computational spacing
    # they are written in. N − 1 is a power of two so that the coordinates,
    # the spacings and the Jacobian are all exact and nothing below turns on
    # a last-bit difference in the grid itself.
    Nx = 33
    eos2 = IdealMixture([IdealSpecies{Float64}("a", 1.0, 1.4),
                         IdealSpecies{Float64}("b", 0.2, 1.09)])
    # One closure type for both mappings below, so the two stretched runs share
    # a single `Solver` specialization instead of compiling one each.
    linmap(a) = Stretch(ξ -> a * ξ, ξ -> a)
    function art_coefficients(L, st)
        s = build_solver(
            Problem(domain=((0.0, L), (0.0, 0.2), (0.0, 0.2)), eos=eos2,
                    bcs=((SlipWallBC(), SlipWallBC()), per3[2], per3[3]), ic=nothing),
            Numerics(n_global=(Nx, 12, 12), art=ArtificialProperties(enabled=true),
                     stretch=(st, nothing, nothing)))
        Q = allocate_state(s)
        initialize!(s, Q, (x, y, z) -> begin
            θ = tanh_blend(x, 0.5L, 0.03L)
            Prim(Y=(1 - θ, θ), rho=1 + θ, p=1.0,
                 u=(0.4sin(2π * x / L), 0.0, 0.0))
        end)
        # Drive one plane's mass fractions out of [0, 1] so the bound term,
        # the only consumer of the geometric mean, is exercised as well.
        for k in 1:s.decomp.n_local[3], j in 1:s.decomp.n_local[2]
            I = padded_index(s, Nx ÷ 4, j, k)
            ρ = Q[I, 1] + Q[I, 2]
            Q[I, 1] -= 0.05ρ
            Q[I, 2] += 0.05ρ
        end
        apply_bcs!(s, Q)
        compute_rhs!(s, Q, zero(Q))
        (copy(s.mu_art), copy(s.beta_art), copy(s.kappa_art),
         copy(s.D_art[1]), copy(s.D_art[2]))
    end
    names = ("mu_art", "beta_art", "kappa_art", "D_art[1]", "D_art[2]")

    # An identity mapping leaves inv_h at exactly one, so the per-point weight
    # is the scalar it replaced and every coefficient reproduces bitwise.
    plain = art_coefficients(1.0, nothing)
    ident = art_coefficients(1.0, linmap(1.0))
    for (nm, a, b) in zip(names, ident, plain)
        a == b || println("identity stretch moved $nm")
        @test a == b
    end
    @test maximum(plain[1]) > 0 && maximum(plain[4]) > 0

    # A mapping of constant Jacobian 2 over (0, 2) is the uniform grid of the
    # same physical spacing written in half the computational spacing. The
    # coefficients agree to round-off; weighting by the computational spacing
    # instead makes them differ by 2^wpow, a factor of four on mu*/beta* and
    # two on kappa*/D*.
    uniform = art_coefficients(2.0, nothing)
    linear = art_coefficients(2.0, linmap(2.0))
    for (nm, a, b) in zip(names, linear, uniform)
        scale = maximum(abs, b)
        rel = maximum(abs, a - b) / scale
        rel < 1e-12 || println("linear stretch moved $nm by $rel relative")
        @test scale > 0
        @test rel < 1e-12
    end
end

@testset "stretched shocked interface: the species bound holds" begin
    # The only case combining Numerics.stretch with the multi-species sensors.
    # The cluster point sits just short of the interface, so the shock crosses
    # it where the physical spacing is about half the computational one and the
    # two weightings differ most. Cheap by construction: 121 points, and the
    # run stops once the transmitted shock has cleared the interface.
    st = sine_cluster(0.0, 1.0, SI_X_IFACE, 0.5)
    r = shock_interface(N=121, tfin=0.15, delta=2.0, nmax=4000, stretch1=st)
    @test r.completed
    @test all(isfinite, r.Y_air)
    # Measured -0.0076 / 1.0076 over 196 steps; the guard is roughly 4x that,
    # the same run on a uniform grid of the same point count reaching 0.0137.
    @test -0.03 < r.worst_min_Y && r.worst_max_Y < 1.03
    @test 0 < r.width_cells < 30
end

@testset "consistent species channels: invariance, shocked interface, slab, width" begin
    # `species_flux = :bulk` diffuses every conserved variable with one D_b, and
    # the default `:partial_density` diffuses the partial densities with it and
    # carries their mass flux into momentum and energy (reference/DESIGN.md,
    # "The species channel"). Four measured properties are guarded for both.
    fick = ArtificialProperties(enabled=true, species_flux=:fickian)
    for bulk in (ArtificialProperties(enabled=true, species_flux=:bulk),
                 ArtificialProperties(enabled=true, species_flux=:partial_density))
        # (a) A resting air/SF6 interface at uniform p, T and u = 0, with the
        # artificial properties and the filter on: every operator of either
        # channel preserves the uniform state to round-off while the partial
        # densities interdiffuse, so u, p and T stay at round-off and ρ moves.
        # Under the Fickian channel the enthalpy flux drives u to 1e-4 here.
        let N = 400, h = 1.0 / N, δ = 2h
            eos = IdealMixture([IdealSpecies{Float64}("air", 1.0, 1.4),
                                IdealSpecies{Float64}("sf6", 1 / 5.04, 1.09)])
            Rk, cvk = eos.Rk, eos.cvk
            two(x) = (θ = tanh_blend(x, 0.3, δ) - tanh_blend(x, 0.7, δ); (1 - θ, θ))
            prob = Problem(eos=eos, transport=ConstantTransport(mu0=0.0),
                           domain=((0.0, 1.0), (0.0, h), (0.0, h)), bcs=per3,
                           ic=(x, y, z) -> begin
                               Y = two(x)
                               Prim(Y=Y, rho=1 / (Y[1] * Rk[1] + Y[2] * Rk[2]),
                                    u=(0.0, 0.0, 0.0), p=1.0)
                           end)
            solver, Q = setup(prob, Numerics(n_global=(N, 1, 1), art=bulk, cfl=0.4,
                                             control=StepControl(validity=:permissive)))
            nx = solver.decomp.n_local[1]
            eqs = solver.equations
            rho0 = [Q[padded_index(solver, i, 1, 1), 1] + Q[padded_index(solver, i, 1, 1), 2] for i in 1:nx]
            umax = Ref(0.0); dpmax = Ref(0.0); dTmax = Ref(0.0); drmax = Ref(0.0)
            function drift(s, Q)
                for i in 1:nx
                    I = padded_index(s, i, 1, 1)
                    ρ = Q[I, 1] + Q[I, 2]
                    y = Q[I, 1] / ρ
                    u = Q[I, eqs.i_mom[1]] / ρ
                    e = Q[I, eqs.i_energy] / ρ - 0.5u^2
                    T = e / (y * cvk[1] + (1 - y) * cvk[2])
                    p = ρ * (y * Rk[1] + (1 - y) * Rk[2]) * T
                    umax[] = max(umax[], abs(u))
                    dpmax[] = max(dpmax[], abs(p - 1))
                    dTmax[] = max(dTmax[], abs(T - 1))
                    drmax[] = max(drmax[], abs(ρ - rho0[i]) / rho0[i])
                end
            end
            run!(solver, Q; tfinal=0.25, nmax=1000, callback=drift)
            # Measured 1.2e-14, 3.8e-14, 4.0e-14 and 2.0e-2 over 298 steps; the
            # Fickian channel measured 2.2e-4, 1.3e-4, 2.5e-5 and 1.8e-2 here
            # under `detector = :delta4, C_D = 0.1`.
            @test completed(solver, 0.25)
            @test umax[] < 1e-13
            @test dpmax[] < 1e-13
            @test dTmax[] < 1e-13
            @test drmax[] > 1e-3
        end

        # (b) The Mach 1.5 shocked interface at density ratio 100 and 2h: the
        # consistent channels carry it (measured worst Y −0.0091 in 673 steps,
        # volume-fraction width 4.8 cells and TV − 1 0.020 under `:bulk`;
        # −0.0092 in 696, 5.3 cells and 0.0020 under `:partial_density`);
        # the Fickian channel loses positivity at step 427 with a DomainError out
        # of the sound speed, so a bounded attempt must either raise or stop
        # short of completion.
        let r = shock_interface(art=bulk, delta=2.0, rho_heavy=100.0, nmax=1500)
            @test r.completed
            @test all(isfinite, r.rho) && minimum(r.rho) > 0
            @test r.worst_min_Y > -0.03
            @test r.worst_max_Y < 1.03
            # The width is read in volume fraction: at this density ratio the
            # mass-fraction count of `width_cells` reads the heavy-gas tail
            # on the air side, 0.84 < V < 0.9995, not the interface. The
            # total variation is the ringing the species detector removes.
            tv_excess = sum(abs, diff(r.Y_air)) - 1
            @test volume_width(r.x, r.Y_air, 100.0) <= 7
            @test tv_excess < 0.04
        end

        # (c) The Brill slab at density ratio 100 and 7 cells per interface, ten
        # periods: the pressure error at the end is round-off under both channels
        # (measured about 5e-11 in 4049 steps) and 5.5e-2 under the
        # Fickian one (4.0e-2 under `detector = :delta4, C_D = 0.1`), whose
        # enthalpy flux is the one operator that moves ρE
        # across a uniform-pressure interface of unequal gas constants.
        let r = brill_slab(art=bulk, nmax=6000)
            @test r.completed
            @test all(isfinite, r.rho) && minimum(r.rho) > 0
            @test r.p_error < 1e-9
            @test r.u_error < 1e-9
            @test r.worst_min_Y > -0.1
        end

        # (d) At equal molecular weights the mole fraction is the mass fraction,
        # the bulk flux of ρY_k at uniform ρ is the Fickian flux, and the two
        # channels' energy fluxes both vanish, so the advected interface's width
        # agrees to round-off accumulation (measured identical to eight digits,
        # the mass fractions within 5e-14).
        let (xs_f, Yf, _, _, okf) = species_advection(art=fick, nmax=2000),
            (xs_b, Yb, _, _, okb) = species_advection(art=bulk, nmax=2000)
            @test okf && okb
            wf = contact_width(xs_f, Yf, 0.0, 1.0)
            wb = contact_width(xs_b, Yb, 0.0, 1.0)
            @test abs(wb - wf) < 1e-6 * wf
        end
    end

    # The Fickian channel loses positivity on case (b).
    let ok = try
            shock_interface(art=fick, delta=2.0, rho_heavy=100.0, nmax=600).completed
        catch err
            err isa DomainError || err isa SolverFailure || rethrow()
            false
        end
        @test !ok
    end

    # The option is validated at setup.
    @test_throws ErrorException build_solver(
        Problem(domain=((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)), bcs=per3, ic=nothing),
        Numerics(n_global=(32, 12, 12), art=ArtificialProperties(species_flux=:brill)))
end

@testset "consistent species channels on patched, refined and tiled layouts" begin
    # A slab of density ratio 20 advected at uniform (u, p, T) through a
    # periodic domain, under both `species_flux = :bulk` and the default
    # `:partial_density`. Every operator either channel meets on these layouts
    # is linear on the conserved variables, the interface rows and the level
    # transfers included, so u, p and T must stay at round-off on every
    # layout while the partial densities interdiffuse (reference/DESIGN.md,
    # "The species channel"). The Fickian channel's enthalpy flux moves p to
    # 1e-4 on the same case. Measured drifts are 1e-14 on each layout below;
    # D_b is small on the smooth slab but must be nonzero on every patch,
    # which is what shows the channel is built and running there.
    eos = IdealMixture([IdealSpecies{Float64}("light", 1.0, 1.4),
                        IdealSpecies{Float64}("heavy", 1 / 20, 1.4)])
    Rk, cvk = eos.Rk, eos.cvk
    function slab(channel; patch_grid=(1, 1, 1), interface_rhs=:extended,
                           interface_flux=:ghost, refine=nothing, tile=0)
        N = 96
        h = 1.0 / N
        s = build_solver(
            Problem(domain=((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)), bcs=per3, eos=eos,
                    transport=ConstantTransport(mu0=0.0), ic=nothing),
            Numerics(n_global=(N, 1, 1),
                     art=ArtificialProperties(enabled=true, species_flux=channel),
                     filter=StateFilter(interval=1),
                     patch_interfaces=PatchInterfaces(flux=interface_flux,
                                                      rhs=interface_rhs),
                     execution=Execution(patch_grid=patch_grid),
                     amr=AMR(initial=refine, regrid_interval=0, tile=tile)))
        states = allocate_state(s)
        initialize!(s, states, (x, y, z) -> begin
            V = (1 - tanh((abs(x - 0.5) - 0.25) / 3h)) / 2
            ρ = V * 20 + (1 - V)
            Prim(Y=(1 - V * 20 / ρ, V * 20 / ρ), rho=ρ, u=(1.0, 0.0, 0.0), p=1.0)
        end)
        drift = Ref(0.0)
        function watch(sol, Qs)
            for (ps, Q) in CL.eachpatch(sol, Qs.patches)
                eq = ps.equations
                for i in 1:ps.decomp.n_local[1]
                    I = padded_index(ps, i, 1, 1)
                    ρ = Q[I, 1] + Q[I, 2]
                    y = Q[I, 1] / ρ
                    u = Q[I, eq.i_mom[1]] / ρ
                    e = Q[I, eq.i_energy] / ρ - 0.5u^2
                    T = e / (y * cvk[1] + (1 - y) * cvk[2])
                    p = ρ * (y * Rk[1] + (1 - y) * Rk[2]) * T
                    drift[] = max(drift[], abs(u - 1), abs(p - 1), abs(T - 1))
                end
            end
        end
        run!(s, states; tfinal=1.0, nmax=40, callback=watch)
        sv = states.patches
        Dmax = [maximum(ps.D_art[1]) for (ps, _) in CL.eachpatch(s, sv)]
        rho = Dict{Int,Float64}()          # root nodes, by global index
        for (ps, Q) in CL.eachpatch(s, sv)
            ps.patch.level == 0 || continue
            for i in 1:ps.decomp.n_local[1]
                I = padded_index(ps, i, 1, 1)
                rho[ps.patch.region.offset[1] + i] = Q[I, 1] + Q[I, 2]
            end
        end
        return (s=s, states=sv, drift=drift[], Dmax=Dmax, rho=rho)
    end
    for channel in (:bulk, :partial_density)
        single = slab(channel)
        @test single.s.step == 40
        @test single.drift < 1e-12
        @test single.Dmax[1] > 0
        # Two patches under both interface forms: the conserved gradients take
        # the interface plans `grad_Y` takes. The patched root differs from the
        # single patch by the interface rows' own error (measured 2e-5 with the
        # extended rows and 5e-4 with the one-sided ones on a wider slab), not
        # by the channel. The one-sided form takes the closure rows.
        for irhs in (:extended, :onesided)
            iflux = irhs === :onesided ? :closure : :ghost
            r = slab(channel; patch_grid=(2, 1, 1), interface_rhs=irhs,
                     interface_flux=iflux)
            @test npatches(r.s) == 2
            @test r.s.step == 40
            @test r.drift < 1e-12
            @test all(>(0), r.Dmax)
            d = maximum(abs(r.rho[k] - single.rho[k]) for k in keys(single.rho))
            @test d < 1e-2
        end
        # One refined level over the slab's left edge, as a box and as a
        # lattice of tiles; the fine patches carry their own D_b.
        for tile in (0, 8)
            r = slab(channel; refine=BlockRegion((16, 0, 0), (24, 1, 1)),
                     tile=tile)
            @test npatches(r.s) == (tile == 0 ? 2 : 4)
            @test r.s.step == 40
            @test all(Q -> all(isfinite, parent(Q)), r.states)
            @test r.drift < 1e-12
            @test all(>(0), r.Dmax)
        end
    end
end

@testset "compression-keyed beta sensors: gated_strain and dilatation" begin
    # The point of both non-default sensors is that bulk viscosity stops firing
    # on vortical structures and on expansions. Both halves are checked here,
    # along with the requirement that mu* — which keeps the strain sensor in
    # every case — is bit-identical across the three settings.
    tgv(sensor) = begin
        s = build_solver(
            Problem(domain=((0.0, 2π), (0.0, 2π), (0.0, 2π)), bcs=per3, ic=nothing),
            Numerics(n_global=(32, 32, 32),
                     art=ArtificialProperties(enabled=true, beta_sensor=sensor)))
        Q = allocate_state(s)
        initialize!(s, Q, (x, y, z) -> Prim(rho=1.0, p=100.0,
            u=(sin(x)cos(y)cos(z), -cos(x)sin(y)cos(z), 0.0)))
        compute_rhs!(s, Q, zero(Q))
        s
    end
    ss, sg, sd = tgv(:strain), tgv(:gated_strain), tgv(:dilatation)
    # Taylor–Green is solenoidal, so neither compression-keyed sensor has
    # anything to fire on; the strain sensor responds to the vortical structure and
    # fires. :dilatation collapses to round-off because its sensor is built
    # from ∇·u. :gated_strain keeps the firing sensor and leans on the switch
    # alone, which removes 99.4% of the total but not the maximum: the strain
    # sensor peaks at the cusps of |S|, and the switch degenerates at exactly
    # those points, where the vorticity vanishes along with |S| itself. The
    # test is therefore on the total, with the surviving peak recorded here
    # rather than asserted away.
    bsum(s) = sum(s.beta_art[padded_index(s, i, j, k)] for i in 1:32, j in 1:32, k in 1:32)
    @test maximum(sd.beta_art) < 1e-12 * maximum(ss.beta_art)
    @test bsum(sg) < 1e-2 * bsum(ss)
    # Over the interior: the ghost layers of a coefficient array that held a
    # sensor keep the sensor's exchanged values, which no reader takes.
    I = CL.interior(ss.decomp)
    @test sd.mu_art[I] == ss.mu_art[I]
    @test sg.mu_art[I] == ss.mu_art[I]
    @test all(isfinite, sd.beta_art) && all(isfinite, sg.beta_art)

    # A velocity ramp: beta* must be exactly zero wherever the flow expands,
    # and must still switch on where it compresses.
    ramp(sensor) = begin
        s = build_solver(
            Problem(domain=((0.0, 1.0), (0.0, 0.2), (0.0, 0.2)), bcs=per3, ic=nothing),
            Numerics(n_global=(64, 12, 12),
                     art=ArtificialProperties(enabled=true, beta_sensor=sensor)))
        Q = allocate_state(s)
        initialize!(s, Q, (x, y, z) -> Prim(rho=1.0, p=1.0,
                                            u=(0.5tanh((x - 0.5) / 0.02), 0.0, 0.0)))
        compute_rhs!(s, Q, zero(Q))
        s
    end
    div1(s, I) = s.grad_u[1, 1][I] + s.grad_u[2, 2][I] + s.grad_u[3, 3][I]
    rstrain = ramp(:strain)
    for sensor in (:gated_strain, :dilatation)
        s = ramp(sensor)
        idx = [padded_index(s, i, 1, 1) for i in 1:64]
        expanding = [I for I in idx if div1(s, I) > 0]
        @test !isempty(expanding)
        @test all(I -> s.beta_art[I] == 0.0, expanding)
        @test maximum(s.beta_art) > 0.1 * maximum(rstrain.beta_art)
    end
    # gated_strain multiplies the strain sensor by a factor in [0, 1], so it can
    # only ever reduce beta*, never move it somewhere new.
    sg2 = ramp(:gated_strain)
    @test all(i -> sg2.beta_art[padded_index(sg2, i, 1, 1)] <=
                   rstrain.beta_art[padded_index(rstrain, i, 1, 1)] + 1e-300, 1:64)

    @test_throws ErrorException build_solver(
        Problem(domain=((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)), bcs=per3, ic=nothing),
        Numerics(n_global=(16, 16, 16), art=ArtificialProperties(beta_sensor=:bogus)))
end

@testset "compact_d8 ring detector: symbol, closures, and selectivity" begin
    # The d8 detector differs below the Nyquist,
    # so the operator is pinned against its analytic symbol rather than
    # against a convergence order. It is also the only symmetric banded
    # scheme in the package, hence the only exercise of BandPlan's filter-side
    # sign conventions (RHS added rather than subtracted, high-edge closure
    # rows mirrored rather than negated).
    N = 64
    art8 = ArtificialProperties(detector=:d8)
    s = build_solver(
        Problem(domain=((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)), bcs=per3, ic=nothing),
        Numerics(n_global=(N, 1, 1), art=art8))
    f = CL.field(s.decomp); out = CL.field(s.decomp)
    pad = s.decomp.n_halo_d[1]
    A(k) = 1 + 2 * (14 / 29) * cos(k) + 2 * (3 / 58) * cos(2k)
    # RHS is 60/(240·29) times the undivided eighth difference (2 − 2cos k)⁴.
    symbol(k) = (60 / (240 * 29)) * (2 - 2cos(k))^4 / A(k)
    for kf in (0.25, 0.5, 0.75, 1.0)
        k = kf * π
        for i in 1:N
            f[i+pad, 1, 1] = cos(k * i)
        end
        CL.exchange_halos!(f, s.decomp)
        # Each dimension's entry is a (wall sign +1, wall sign −1) pair; on a
        # periodic dimension the two slots hold one plan.
        CL.apply_along!(out, s.ring_plans[1].closed[1], f, s.decomp)
        i0 = N ÷ 2
        # rtol is 1e-9, not machine epsilon: the detector is a high-pass, so a
        # well-resolved wave is the difference of coefficients of order 1
        # producing an answer of order 1e-4, and three digits go to
        # cancellation. That is a property of the operator, not slack here.
        @test out[i0+pad, 1, 1] / cos(k * i0) ≈ symbol(k) rtol = 1e-9
    end
    # Normalization: the response at two points per wavelength is 16, matching
    # undivided δ⁴ there. This separation lets the four
    # constants transfer between detectors.
    @test symbol(π) ≈ 16.0 rtol = 1e-12
    # Selectivity against undivided δ⁴ on the same normalization: measured
    # 569× at eight points per wavelength and 26× at four.
    d4(k) = (2 - 2cos(k))^2
    @test symbol(0.25π) < d4(0.25π) / 500
    @test symbol(0.5π) < d4(0.5π) / 25

    # Constants must be annihilated exactly through the four closure rows, not
    # by cancellation: every row's weights sum to zero by construction. A slip
    # wall takes the node-centred rows, an extrapolation face the scheme's own.
    for lo in (SlipWallBC(), ExtrapolationBC())
        sw = build_solver(
            Problem(domain=((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)),
                    bcs=((lo, lo), per3[2], per3[3]), ic=nothing),
            Numerics(n_global=(N, 1, 1), art=art8))
        fw = CL.field(sw.decomp); ow = CL.field(sw.decomp)
        fill!(fw, 1.0)
        CL.exchange_halos!(fw, sw.decomp)
        CL.apply_along!(ow, sw.ring_plans[1].closed[1], fw, sw.decomp)
        @test maximum(abs, ow[(pad+1):(pad+N), 1, 1]) < 1e-14
    end

    @test_throws ErrorException build_solver(
        Problem(domain=((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)), bcs=per3, ic=nothing),
        Numerics(n_global=(16, 16, 16), art=ArtificialProperties(detector=:bogus)))
end

@testset "sensor operators at a reflecting wall" begin
    # The `:gaussian` smoother and the `:d8` detector carry closure rows that
    # fold onto the half-offset mirror, half a cell out at a node-centred wall.
    # A reflecting face takes `wall_closures` instead, so a field exactly even
    # about both wall nodes, or exactly odd for a wall-normal velocity, must
    # reproduce the periodic run on the same spacing: the extended domain of
    # 2(N − 1) cells whose restriction is the wall run.
    N = 49
    wallbc = ((SlipWallBC(), SlipWallBC()), per3[2], per3[3])
    even_field(x) = cospi(x) + 0.5cospi(5x) + 0.1cospi(13x)
    odd_field(x) = sinpi(x) + 0.1sinpi(13x)
    both(art) = (build_solver(
        Problem(domain=((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)), bcs=wallbc, ic=nothing),
        Numerics(n_global=(N, 1, 1), art=art)),
                 build_solver(
                     Problem(domain=((0.0, 2.0), (0.0, 1.0), (0.0, 1.0)), bcs=per3,
                             ic=nothing), Numerics(n_global=(2(N - 1), 1, 1), art=art)))
    function load(s, fn)
        f = CL.field(s.decomp)
        pad = s.decomp.n_halo_d[1]
        for i in 1:s.decomp.n_local[1]
            f[i+pad, 1, 1] = fn(CL.xcoord(s, 1, i))
        end
        CL.exchange_halos!(f, s.decomp)
        f
    end
    # The first and last six nodes of the wall run, and the same nodes of the
    # periodic one.
    win(s, a) = (pad = s.decomp.n_halo_d[1];
                 [a[i+pad, 1, 1] for i in [1:6; (N-5):N]])

    sw, sp = both(ArtificialProperties())
    fw, fp = load(sw, even_field), load(sp, even_field)
    CL.smooth!(fw, sw)
    CL.smooth!(fp, sp)
    @test maximum(abs, win(sw, fw) .- win(sp, fp)) <
          1e-14 * maximum(abs, win(sp, fp))

    dw, dp = both(ArtificialProperties(detector=:d8))
    for (fn, σw) in ((even_field, 1), (odd_field, -1))
        gw, gp = load(dw, fn), load(dp, fn)
        ow, op = CL.field(dw.decomp), CL.field(dp.decomp)
        CL.detect_sum!(ow, gw, dw, 1; wall_parity=(σw, 1, 1))
        CL.detect_sum!(op, gp, dp, 1)
        # The detector is a high-pass: on a resolved field it is the difference
        # of order-one coefficients producing an answer four orders smaller, so
        # 1e-8 relative is its round-off and not slack here.
        @test maximum(abs, win(dw, ow) .- win(dp, op)) <
              1e-8 * maximum(win(dp, op))
    end
    # An odd field vanishes on the wall node, and the odd row 1 returns that
    # exactly rather than to round-off.
    gw = load(dw, odd_field)
    ow = CL.field(dw.decomp)
    CL.detect_sum!(ow, gw, dw, 1; wall_parity=(-1, 1, 1))
    @test ow[1+dw.decomp.n_halo_d[1], 1, 1] == 0.0

    # The rows themselves: built from the interior weights, so the filter's
    # unit row sum and the eighth derivative's zero row sum are inherited.
    a, b = 3565 / 10368, 3091 / 12960
    c, d, e = 1997 / 25920, 149 / 12960, 107 / 103680
    grows = CL.wall_closures(gaussian_filter(), 1)
    @test length(grows) == 4
    @test all(row -> sum(row.rhs) ≈ 1.0, grows)
    @test grows[1].rhs ≈ [a, 2b, 2c, 2d, 2e]
    @test grows[2].rhs ≈ [b, a + c, b + d, c + e, d, e]
    @test grows[3].rhs ≈ [c, b + d, a + e, b, c, d, e]
    @test grows[4].rhs ≈ [d, c + e, b, a, b, c, d, e]
    drows = CL.wall_closures(compact_d8(), 1)
    @test all(row -> abs(sum(row.rhs)) < 1e-15, drows)
    # Row 1 folds each left-hand-side band onto the interior side, doubling it
    # for an even field; at σ = −1 the whole row collapses to g₁ = a₀ f₁.
    @test drows[1].lhs ≈ [0, 0, 1, 2 * 14 / 29, 2 * 3 / 58]
    @test drows[2].lhs ≈ [0, 14 / 29, 1 + 3 / 58, 14 / 29, 3 / 58]
    orows = CL.wall_closures(compact_d8(), -1)
    @test orows[1].lhs == [0, 0, 1, 0, 0]
    @test orows[1].rhs ≈ [35 / 58, 0, 0, 0, 0]
    # An antisymmetric scheme has no such fold.
    @test_throws ErrorException CL.wall_closures(lele_d1_6(), 1)

    # A fold's far end may be a wall. The outer end of a radial line takes the
    # same rows, so the fold holds one detector plan per ghost parity and per
    # wall sign.
    sa = build_solver(
        Problem(domain=((0.0, 1.0), (0.0, 1.0), (0.0, 0.5)), metric=CylindricalMetric(),
                bcs=((AxisBC(), SlipWallBC()), per3[2], per3[3]), ic=nothing),
        Numerics(n_global=(32, 1, 12), art=ArtificialProperties(detector=:d8)))
    @test sa.folds[1].ring_plans.ghost === sa.folds[1].ring_plans.closed
    rp = sa.folds[1].ring_plans.closed
    @test length(rp) == 2 && all(p -> length(p) == 2, rp)
    @test rp[1][1] !== rp[1][2] && rp[2][1] !== rp[2][2]
    # An outer end that reflects nothing keeps one plan in both slots.
    sx = build_solver(
        Problem(domain=((0.0, 1.0), (0.0, 1.0), (0.0, 0.5)), metric=CylindricalMetric(),
                bcs=((AxisBC(), ExtrapolationBC()), per3[2], per3[3]), ic=nothing),
        Numerics(n_global=(32, 1, 12), art=ArtificialProperties(detector=:d8)))
    @test sx.folds[1].ring_plans.closed[1][1] === sx.folds[1].ring_plans.closed[1][2]
end

@testset "pyranda_filter: symbol, Nyquist zero, 9/10 integral, closures" begin
    # Pyranda's c8ff8 transcribed as a symmetric banded scheme, pinned against
    # its analytic symbol the way the d8 detector is; the same filter-side
    # sign conventions of BandPlan carry it.
    N = 64
    s = build_solver(
        Problem(domain=((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)), bcs=per3, ic=nothing),
        Numerics(n_global=(N, 1, 1), filter=pyranda_filter()))
    f = CL.field(s.decomp)
    pad = s.decomp.n_halo_d[1]
    β, α = 1.6688e-1, 6.6624e-1
    a, b, c, d, e = 9.9965e-1, 6.6652e-1, 1.6674e-1, 4.0e-5, -5.0e-6
    symbol(k) = (a + 2b * cos(k) + 2c * cos(2k) + 2d * cos(3k) + 2e * cos(4k)) /
                (1 + 2α * cos(k) + 2β * cos(2k))
    for kf in (0.25, 0.5, 0.75)
        k = kf * π
        for i in 1:N
            f[i+pad, 1, 1] = cos(k * i)
        end
        filter_field!(f, s)
        i0 = N ÷ 2
        @test f[i0+pad, 1, 1] / cos(k * i0) ≈ symbol(k) rtol = 1e-10
    end
    @test symbol(0.0) ≈ 1.0 rtol = 1e-14
    @test abs(symbol(π)) < 1e-12
    @test symbol(0.5π) > 0.99
    ks = range(0, π; length=20001)
    @test sum(symbol.(ks)) * step(ks) / π ≈ 0.9 atol = 2e-3
    # Grid-to-grid oscillation on a constant is removed to round-off.
    for i in 1:N
        f[i+pad, 1, 1] = 1.0 + 0.5 * (-1)^i
    end
    filter_field!(f, s)
    @test maximum(abs(f[i+pad, 1, 1] - 1.0) for i in 1:N) < 1e-12
    # Closed edges: the telescoping rows pass a constant exactly, and each
    # row's two sides sum equally by construction.
    for row in pyranda_filter().closures
        @test sum(row.lhs) ≈ sum(row.rhs) rtol = 1e-14
    end
    sw = build_solver(
        Problem(domain=((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)),
                bcs=((SlipWallBC(), SlipWallBC()), per3[2], per3[3]), ic=nothing),
        Numerics(n_global=(N, 1, 1), filter=pyranda_filter()))
    fw = CL.field(sw.decomp)
    fill!(fw, 1.0)
    filter_field!(fw, sw)
    @test maximum(abs(fw[i+pad, 1, 1] - 1.0) for i in 1:N) < 1e-13
end

@testset "detector = :d8 through compute_artificial!" begin
    # End to end on all three dimensions and both sensor weights, against the
    # δ⁴ default on identical data. A near-grid-scale ramp must still fire; a
    # resolved wave must fire far less than δ⁴ does, which is the reason the
    # detector exists.
    #
    # The selectivity assertion is on κ\*, not β\*, and that is the measurement
    # rather than a convenience. κ\*'s input is the internal energy, which is
    # smooth when the flow is; β\*'s is |S|, which has a cusp wherever the
    # strain passes through zero, and a cusp is a grid-scale feature no
    # detector can decline to see. On a resolved sine the two detectors
    # therefore differ by 2.6e6 on κ\* and by a factor of 1.8 on β\*. See
    # `reference/CALIBRATION_APPENDIX.md`.
    sensors(detector, fn) = begin
        s = build_solver(
            Problem(domain=((0.0, 1.0), (0.0, 0.2), (0.0, 0.2)), bcs=per3, ic=nothing),
            Numerics(n_global=(64, 12, 12),
                     art=ArtificialProperties(enabled=true, detector=detector)))
        Q = allocate_state(s)
        initialize!(s, Q, fn)
        compute_rhs!(s, Q, zero(Q))
        s
    end
    ramp = (x, y, z) -> Prim(rho=1.0, p=1.0, u=(0.5tanh((x - 0.5) / 0.02), 0.0, 0.0))
    wave = (x, y, z) -> Prim(rho=1.0, p=1.0 + 0.1sinpi(2x), u=(0.0, 0.0, 0.0))
    r4, r8 = sensors(:delta4, ramp), sensors(:d8, ramp)
    w4, w8 = sensors(:delta4, wave), sensors(:d8, wave)
    for s in (r8, w8)
        @test all(isfinite, s.beta_art) && all(isfinite, s.mu_art)
        @test all(isfinite, s.kappa_art)
        @test all(>=(0.0), s.beta_art)
    end
    # A ramp two to three cells wide is close enough to the grid scale that
    # the shared normalization holds; an order of magnitude either way would
    # mean the scaling is wrong. Measured 0.23.
    @test 0.1 < maximum(r8.beta_art) / maximum(r4.beta_art) < 10
    # A pressure wave resolved over 64 points is where they must not agree.
    # Measured 3.9e-7.
    @test maximum(w8.kappa_art) < 1e-4 * maximum(w4.kappa_art)
    @test maximum(w4.kappa_art) > 0
end

@testset "detector = :d8 through a coordinate-singularity fold" begin
    # BandPlan's fold assembly with a SYMMETRIC scheme, plus the :ring role in
    # fold_apply!. The C10 fold test covers the antisymmetric case only.
    s = build_solver(
        Problem(domain=((0.0, 1.0), (0.0, 1.0), (0.0, 0.5)), metric=CylindricalMetric(),
                bcs=((AxisBC(), SlipWallBC()), per3[2], per3[3]), ic=nothing),
        Numerics(n_global=(64, 1, 12), art=ArtificialProperties(enabled=true, detector=:d8)))
    f = CL.field(s.decomp); out = CL.field(s.decomp)
    # An even function of r: d8 across the axis fold must stay smooth and
    # small, and in particular must not blow up in the first cells.
    fillf!(s, f, (r, θ, z) -> exp(-4r^2))
    CL.exchange_halos!(f, s.decomp)
    CL.ring_along!(out, f, s, 1, 1)
    @test all(isfinite, out)
    # The window stops at the domain's midpoint. This field has a slope at the
    # outer slip wall, so it is not the reflection the wall closure rows
    # continue it as, and the detector reads that mismatch over the last cells;
    # the fold at the other end is what this case measures.
    @test maximum(abs, out[padded_index(s, i, 1, k)] for i in 1:32, k in 1:12) < 1e-3
    # And the full sensor path runs through the fold.
    Q = allocate_state(s)
    initialize!(s, Q, (r, θ, z) -> Prim(rho=1.0, p=1.0, u=(-1.0, 0.0, 0.0)))
    compute_rhs!(s, Q, zero(Q))
    @test all(isfinite, s.beta_art) && maximum(s.beta_art) > 0
end

@testset "sensor fields: mu* from the velocity, beta* from the dilatation" begin
    # Cook (2007) builds both sensors from |S|; Cook (2009) changes beta* to
    # ∇·u. Pyranda builds them from the velocity components and from ∇·u. The
    # difference between the two is the absolute value, and these tests pin its
    # two consequences.
    oned(art, fn; N=64) = begin
        s = build_solver(
            Problem(domain=((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)), bcs=per3, ic=nothing),
            Numerics(n_global=(N, 1, 1), art=art))
        Q = allocate_state(s)
        initialize!(s, Q, fn)
        compute_rhs!(s, Q, zero(Q))
        s
    end
    # 1. The two-point wave. A centered derivative annihilates it exactly, so
    # |S| and ∇·u are identically zero on the shortest wave the grid carries
    # and every sensor built from them returns zero there. The velocity sensor
    # returns the full undivided response, C_mu·ρ·16h.
    nyq = (x, y, z) -> Prim(rho=1.0, p=1.0, u=(cospi(64x), 0.0, 0.0))
    @test maximum(oned(ArtificialProperties(mu_sensor=:strain), nyq).mu_art) == 0.0
    sv = oned(ArtificialProperties(mu_sensor=:velocity), nyq)
    @test maximum(sv.mu_art[CL.interior(sv.decomp)]) ≈ 0.002 * 16 / 64 rtol = 1e-12
    @test maximum(oned(ArtificialProperties(beta_sensor=:ungated_dilatation),
                       nyq).beta_art) == 0.0

    # 2. The detector's selectivity survives the field change, which is the
    # reason for making it. On a wave resolved over eight points the two
    # detectors are built to differ by 569×, and they do so when applied to a
    # smooth field. Applied to |S| the same pair differs by 1.28×, the cusps
    # where the strain passes through zero being grid-scale structure at every
    # wavelength. Measured 569 and 1.28.
    wave = (x, y, z) -> Prim(rho=1.0, p=1.0, u=(cospi(16x), 0.0, 0.0))
    peak(ms, det) =
        maximum(oned(ArtificialProperties(mu_sensor=ms, detector=det), wave).mu_art)
    @test peak(:velocity, :d8) < peak(:velocity, :delta4) / 100
    @test peak(:strain, :d8) > peak(:strain, :delta4) / 10

    # 3. Σ_d against MAX. They are the same operation in one dimension, and the
    # reduction is a per-direction one, so MAX can never exceed Σ_d anywhere.
    ramp1 = (x, y, z) -> Prim(rho=1.0, p=1.0, u=(0.5tanh((x - 0.5) / 0.02), 0.0, 0.0))
    @test oned(ArtificialProperties(reduction=:sum), ramp1).mu_art ==
          oned(ArtificialProperties(reduction=:max), ramp1).mu_art
    tgv3(red) = begin
        s = build_solver(
            Problem(domain=((0.0, 2π), (0.0, 2π), (0.0, 2π)), bcs=per3, ic=nothing),
            Numerics(n_global=(32, 32, 32), art=ArtificialProperties(reduction=red)))
        Q = allocate_state(s)
        initialize!(s, Q, (x, y, z) -> Prim(rho=1.0, p=100.0,
            u=(sin(x)cos(y)cos(z), -cos(x)sin(y)cos(z), 0.0)))
        compute_rhs!(s, Q, zero(Q))
        s
    end
    tsum, tmax = tgv3(:sum), tgv3(:max)
    @test all(tmax.mu_art .<= tsum.mu_art)
    @test maximum(tmax.mu_art) < maximum(tsum.mu_art)

    # 4. Parity across a fold. u_r is odd there, so the detector requires the
    # sign. With it, u_r = r produces no sensor at all, that being the regular
    # behaviour of a radial velocity at the axis, while a uniform u_r produces
    # one on the first cell, the folded field having a genuine kink there.
    axial(art, fn) = begin
        s = build_solver(
            Problem(domain=((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)), metric=CylindricalMetric(),
                    bcs=((AxisBC(), SlipWallBC()), per3[2], per3[3]), ic=nothing),
            Numerics(n_global=(32, 1, 1), art=art))
        Q = allocate_state(s)
        initialize!(s, Q, fn)
        compute_rhs!(s, Q, zero(Q))
        s
    end
    lin = (r, θ, z) -> Prim(rho=1.0, p=1.0, u=(r, 0.0, 0.0))
    uni = (r, θ, z) -> Prim(rho=1.0, p=1.0, u=(-1.0, 0.0, 0.0))
    # δ⁴ across the fold reads the half-offset mirror for an odd field, so this
    # is exact; the d8 closure reaches it through a line solve, hence round-off.
    # Its floor is the wider one, and not for the fold's sake: u_r = r is not
    # odd about the outer wall either, the slip condition leaving a kink on the
    # wall node, and the pentadiagonal inverse carries a decaying tail of that
    # mismatch back across the 31 cells to the axis. Measured 8.7e-14 against a
    # wall-node 5.5e-5 and the uniform case's 1.6e-4 at the axis below.
    sl = axial(ArtificialProperties(mu_sensor=:velocity), lin)
    @test sl.mu_art[padded_index(sl, 1, 1, 1)] == 0.0
    s8 = axial(ArtificialProperties(mu_sensor=:velocity, detector=:d8), lin)
    @test s8.mu_art[padded_index(s8, 1, 1, 1)] < 1e-12
    su = axial(ArtificialProperties(mu_sensor=:velocity), uni)
    @test su.mu_art[padded_index(su, 1, 1, 1)] > 0

    # 5. The ungated dilatation sensor is the form the reference uses. The
    # gated one is that sensor multiplied by the Ducros switch, so it is never
    # larger, and is exactly zero wherever the flow expands.
    ug = oned(ArtificialProperties(beta_sensor=:ungated_dilatation), ramp1)
    g = oned(ArtificialProperties(beta_sensor=:dilatation), ramp1)
    @test all(g.beta_art .<= ug.beta_art .+ 1e-300)
    div1(s, I) = s.grad_u[1, 1][I] + s.grad_u[2, 2][I] + s.grad_u[3, 3][I]
    expanding = [padded_index(ug, i, 1, 1) for i in 1:64 if div1(ug, padded_index(ug, i, 1, 1)) > 0]
    @test !isempty(expanding)
    @test all(I -> g.beta_art[I] == 0.0, expanding)
    @test any(I -> ug.beta_art[I] > 0.0, expanding)

    # 6. The channels are independent: rebuilding one leaves the other's
    # numbers bit-identical to the default configuration's.
    base = oned(ArtificialProperties(), ramp1)
    @test oned(ArtificialProperties(mu_sensor=:velocity), ramp1).beta_art == base.beta_art
    sdil = oned(ArtificialProperties(beta_sensor=:ungated_dilatation), ramp1)
    @test sdil.mu_art[CL.interior(sdil.decomp)] == base.mu_art[CL.interior(base.decomp)]
    @test oned(ArtificialProperties(mu_sensor=:velocity), ramp1).mu_art != base.mu_art

    @test_throws ErrorException build_solver(
        Problem(domain=((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)), bcs=per3, ic=nothing),
        Numerics(n_global=(16, 16, 16), art=ArtificialProperties(mu_sensor=:bogus)))
    @test_throws ErrorException build_solver(
        Problem(domain=((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)), bcs=per3, ic=nothing),
        Numerics(n_global=(16, 16, 16), art=ArtificialProperties(reduction=:bogus)))
end
