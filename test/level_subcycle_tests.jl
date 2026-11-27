# The level tests of subcycling: its guards, the interpolation order of the live
# transfers, and the subcycled manufactured solution and Sod tube. Included by
# test/level_tests.jl, which defines the helpers used here.

@testset "subcycle and regrid configuration guards" begin
    per3l = ntuple(_ -> (PeriodicBC(), PeriodicBC()), 3)
    mk(; subcycle=false, regrid_interval=0, refine=nothing) = build_solver(
        Problem(domain=((0.0, 2π), (0.0, 1.0), (0.0, 1.0)), bcs=per3l, ic=nothing),
        Numerics(n_global=(96, 1, 1),
                 amr=AMR(initial=refine, subcycle=subcycle, regrid_interval=regrid_interval)))
    # Both features require an initial refined region.
    @test_throws ErrorException mk(subcycle=true)
    @test_throws ErrorException mk(regrid_interval=10)
    @test_throws ErrorException mk(refine=BlockRegion((40, 0, 0), (16, 1, 1)),
                                   regrid_interval=-1)
end

@testset "level interpolation order: configuration, exactness and rebuilds" begin
    per3l = ntuple(_ -> (PeriodicBC(), PeriodicBC()), 3)
    mk(p; regrid_interval=0, tag_threshold=CompactLES.DEFAULT_TAG_THRESHOLD, tag_buffer=4,
          tag_predicate=nothing) = build_solver(
        Problem(domain=((0.0, 2π), (0.0, 1.0), (0.0, 1.0)), bcs=per3l, ic=nothing),
        Numerics(n_global=(96, 1, 1), art=ArtificialProperties(enabled=false),
                 filter=StateFilter(interval=0),
                 amr=AMR(initial=BlockRegion((40, 0, 0), (17, 1, 1)),
                         level_interpolation_order=p, regrid_interval=regrid_interval,
                         tag_threshold=tag_threshold, tag_buffer=tag_buffer));
        tag_predicate=tag_predicate)
    for bad in (0, 3, 5, 12)
        @test_throws ErrorException mk(bad)
    end
    chain_order(s) = only(unique(pl.interp_order
                                 for pl in getfield(s, :levels)[2].transfers[1].pplans))
    # By default the derivative operator's interior order, two more under
    # the ghost fluxes, and at most 10.
    for (deriv, closure, ghost) in ((lele_d1_6(), 6, 8), (lele_d1_8(), 8, 10),
                                    (lele_d1_10(), 10, 10),
                                    (lele_d1_6(closures=:brady_livescu), 6, 8))
        for (flux, p) in ((:closure, closure), (:ghost, ghost))
            @test chain_order(build_solver(
                Problem(domain=((0.0, 2π), (0.0, 1.0), (0.0, 1.0)), bcs=per3l, ic=nothing),
                Numerics(n_global=(96, 1, 1), deriv=deriv,
                         patch_interfaces=PatchInterfaces(flux=flux),
                         amr=AMR(initial=BlockRegion((40, 0, 0), (17, 1, 1)),
                                 regrid_interval=0)))) == p
        end
    end
    # The default interface flux is the ghost path's, and so is its order.
    @test chain_order(build_solver(
        Problem(domain=((0.0, 2π), (0.0, 1.0), (0.0, 1.0)), bcs=per3l, ic=nothing),
        Numerics(n_global=(96, 1, 1),
                 amr=AMR(initial=BlockRegion((40, 0, 0), (17, 1, 1)),
                         regrid_interval=0)))) == 8
    @test CL.default_interpolation_order(compact_filter(0.45)) == 6
    # The shell reproduces a polynomial of degree p − 1 and not one of
    # degree p: the chain runs at the order requested.
    for p in (2, 4, 6, 8, 10)
        s = mk(p)
        @test chain_order(s) == p
        errs = map((p - 1, p)) do deg
            states = allocate_state(s)
            f(x) = (3 * (x - π))^deg
            for (k, Q) in enumerate(states.patches)
                ps = PatchSolver(s, s.patches[k])
                pad = ps.decomp.n_halo_d[1]
                for I in CartesianIndices(size(Q)[1:3])
                    Q[I, 1] = f(xcoord(ps, 1, I[1] - pad))
                    Q[I, 5] = 2.5
                end
            end
            prolong_level_ghosts!(s, states)
            ps = PatchSolver(s, s.patches[2])
            pad = ps.decomp.n_halo_d[1]
            slots = CartesianIndices(size(states.patches[2])[1:3])
            maximum(abs(states.patches[2][I, 1] - f(xcoord(ps, 1, I[1] - pad))) for I in slots) /
                maximum(abs(states.patches[2][I, 1]) for I in slots)
        end
        @test errs[1] < 1e-13
        @test errs[2] > 1e-10
    end
    # A regrid rebuilds its transfer at the configured order.
    center = Ref(Float64(π))
    predicate = (patch, I) -> abs(xcoord(patch, 1, interior_index(patch, I)[1]) -
                                  center[]) < π / 6
    s = mk(8; regrid_interval=1, tag_threshold=1e6, tag_buffer=2,
           tag_predicate=predicate)
    @test getfield(s, :schemes).level_interpolation_order == 8
    states = allocate_state(s)
    initialize!(s, states, (x, y, z) -> Prim(rho=1 + 0.2sin(x), u=(0.5, 0, 0), p=1.0))
    center[] = π + 0.5
    getfield(s, :regrid).checks += 1     # the run! cadence hook's count
    @test CL.regrid!(s, states, Workspace(states), nothing)
    @test chain_order(s) == 8
    # `AMR` carries the keyword for an explicit region, and the
    # predicate-selected initial layout (built by the regrid machinery) keeps it.
    prob = Problem(domain=((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)), bcs=per3l,
                   ic=(x, y, z) -> Prim(p=1.0, rho=1.0, u=(0.0, 0.0, 0.0)))
    s4, _ = setup(prob, Numerics(n_global=(96, 1, 1), filter=nothing,
                                 amr=AMR(initial=BlockRegion((40, 0, 0), (17, 1, 1)),
                                         level_interpolation_order=4)))
    @test chain_order(s4) == 4
    s8, _ = setup(prob, Numerics(n_global=(96, 1, 1), filter=nothing,
                                 art=ArtificialProperties(enabled=false),
                                 amr=AMR(initial=(x, y, z, t) -> abs(x - 0.7) < 0.04,
                                         level_interpolation_order=8)))
    @test chain_order(s8) == 8
end

@testset "subcycled two levels: manufactured solution across the boundary" begin
    errs = [_level_wave_error(N; subcycle=true, cfl=SUBCYCLE_CFL) for N in (48, 96, 192)]
    orders = [log2(errs[i] / errs[i+1]) for i in 1:2]
    @info "subcycled two-level entropy wave" errs orders
    # Measured 1.5e-10 / 2.0e-12 / 3.4e-14, orders 6.25 / 5.83, against the
    # global step's 1.5e-10 / 2.0e-12 / 3.3e-14 at the default cfl. Under
    # the closure rows at the default cfl 8.4e-8 / 7.5e-9 / 5.9e-10, orders
    # 3.49 / 3.66. dt is coarse-limited (the fine rate enters the reduction
    # divided by 3), so N = 192 takes 223 steps here against the global
    # step's 159.
    @test all(>(5.0), orders)
    @test errs[2] < 1e-11
end

@testset "subcycled Sod through the refinement boundary" begin
    wall2 = (SlipWallBC(), SlipWallBC())
    per = (PeriodicBC(), PeriodicBC())
    ic(x, y, z) = x < 0.5 ? Prim(u=(0, 0, 0), p=1.0, rho=1.0) :
                            Prim(u=(0, 0, 0), p=0.1, rho=0.125)
    N = 201
    # The static gate's configuration with subcycling on: the same region,
    # CFL, and crossing schedule, so the guards compare directly. C10 runs
    # the same crossing with the sensors and gradients through its own
    # interface rows.
    for (label, deriv, n_halo) in (("C6", lele_d1_6(), 4), ("C10", lele_d1_10(), 4))
        solver = build_solver(
            Problem(domain=((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)), bcs=(wall2, per, per),
                    ic=nothing),
            Numerics(n_global=(N, 1, 1), cfl=0.4, deriv=deriv, n_halo=n_halo,
                     amr=AMR(initial=BlockRegion((120, 0, 0), (41, 1, 1)),
                             regrid_interval=0, subcycle=true)))
        states = allocate_state(solver)
        initialize!(solver, states, ic)
        m0 = _two_level_mass(solver, states)
        run!(solver, states; tfinal=0.1, nmax=20000)
        ps = PatchSolver(solver, solver.patches[1])
        pad = ps.decomp.n_halo_d[1]
        m1 = solver.equations.i_mom[1]
        noise = maximum(abs(states.patches[1][i + pad, 1, 1, m1]) for i in 172:N)
        @info "subcycled Sod through refinement boundary, $label" noise
        # Measured 1.5e-8 against the global-dt gate's 1.4e-8; C10 9.7e-9.
        @test noise < 1e-7
        for (psq, Q) in CL.eachpatch(solver, states)
            n = psq.decomp.n_local[1]
            padq = psq.decomp.n_halo_d[1]
            @test minimum(Q[i + padq, 1, 1, 1] for i in 1:n) > 0.05
        end
        run!(solver, states; tfinal=0.2, nmax=40000)
        drift = abs(_two_level_mass(solver, states) - m0) / m0
        @info "subcycled two-level Sod mass drift, $label" drift
        # Measured 1.13e-6 (C6) and 7.8e-7 (C10) against the global step's
        # 7.0e-6.
        @test drift < 1e-5
    end
end
