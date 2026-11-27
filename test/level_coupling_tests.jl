# The level tests of the level coupling: two and three levels and tiled levels,
# their manufactured solutions and Sod tubes, the configuration guards, and the
# exchanges across shared tile faces. Included by test/level_tests.jl, which defines
# the helpers used here.

@testset "refined solver rejects unsupported configurations" begin
    per3l = ntuple(_ -> (PeriodicBC(), PeriodicBC()), 3)
    mk(; deriv=lele_d1_6(), n_halo=4, art=ArtificialProperties(), filt=compact_filter(),
         metric=CartesianMetric(), origin=(0.0, 0.0, 0.0), patch_grid=(1, 1, 1)) =
        build_solver(Problem(domain=((origin[1], origin[1] + 2π),
                                     (origin[2], origin[2] + 1.0),
                                     (origin[3], origin[3] + 1.0)),
                             bcs=per3l, metric=metric, ic=nothing),
                     Numerics(n_global=(96, 1, 1), deriv=deriv, n_halo=n_halo, art=art,
                              filter=filt, execution=Execution(patch_grid=patch_grid),
                              amr=AMR(initial=BlockRegion((40, 0, 0), (16, 1, 1)),
                                      regrid_interval=0)))
    # C10 closes the coarse-fine boundary with two compact rows reading
    # four ghost layers, the default halo; a narrower one is refused (the
    # root's C8 filter trips first at three).
    @test_throws ErrorException mk(deriv=lele_d1_10(), n_halo=3)
    @test npatches(mk(deriv=lele_d1_10())) == 2
    # The d8 detector and a pentadiagonal filter close the coarse-fine
    # boundary with banded interface rows of their own.
    @test npatches(mk(art=ArtificialProperties(detector=:d8))) == 2
    @test npatches(mk(filt=pyranda_filter())) == 2
    @test_throws "refinement requires CartesianMetric" mk(metric=SphericalMetric(),
                                                          origin=(0.5, π / 2, 0.0))
    @test_throws ErrorException mk(patch_grid=(2, 1, 1))
    # A region reaching the seam of a periodic dimension runs across it, but
    # one refined box does not close on itself around the period.
    @test npatches(build_solver(
        Problem(domain=((0.0, 2π), (0.0, 1), (0.0, 1)), bcs=per3l, ic=nothing),
        Numerics(n_global=(96, 1, 1),
                 amr=AMR(initial=BlockRegion((0, 0, 0), (16, 1, 1)), regrid_interval=0)))) == 2
    @test_throws "periodic dimension 1" build_solver(
        Problem(domain=((0.0, 2π), (0.0, 1), (0.0, 1)), bcs=per3l, ic=nothing),
        Numerics(n_global=(96, 1, 1),
                 amr=AMR(initial=BlockRegion((0, 0, 0), (90, 1, 1)), regrid_interval=0)))
    # A refined solver reports two patches, level 1 second, at h/3.
    solver = mk()
    @test npatches(solver) == 2
    ps = PatchSolver(solver, solver.patches[2])
    @test ps.patch.level == 1
    @test ps.h[1] ≈ (2π / 96) / 3
    @test ps.decomp.n_local[1] == 3 * 16 - 2
    # Coincident nodes agree in physical coordinates.
    pc = PatchSolver(solver, solver.patches[1])
    @test xcoord(ps, 1, 1) ≈ xcoord(pc, 1, 41) atol = 1e-14
    @test xcoord(ps, 1, 46) ≈ xcoord(pc, 1, 56) atol = 1e-13
end

@testset "two levels: manufactured solution across the coarse-fine boundary" begin
    errs = [_level_wave_error(N) for N in (48, 96, 192)]
    orders = [log2(errs[i] / errs[i+1]) for i in 1:2]
    @info "two-level entropy wave" errs orders
    # Measured 1.5e-10 / 2.0e-12 / 3.3e-14, orders 6.25 / 5.90, under the
    # default ghost fluxes. Under the closure rows 8.5e-8 / 7.7e-9 / 6.2e-10,
    # orders 3.46 / 3.64: the one-sided divergence closures bind, as at a
    # same-level patch interface.
    @test all(>(5.0), orders)
    @test errs[2] < 1e-11
    errsc = [_level_wave_error(N; interface_flux=:closure) for N in (48, 96, 192)]
    ordersc = [log2(errsc[i] / errsc[i+1]) for i in 1:2]
    @info "two-level entropy wave, closure rows" errsc ordersc
    @test all(>(3.0), ordersc)
    @test errsc[2] < 3e-8
    # The deconvolution/filter coupling measures order 1.3-1.7 (see the
    # src/amr/levels_types.jl header); pin that it stays selectable and stable.
    ef = _level_wave_error(96; mode=:filter)
    @test 1e-5 < ef < 1e-4
    # C10 at the ghost path's order-10 interpolation: measured 2.7e-13 /
    # 1.9e-14 / 8.2e-15, at the round-off floor from N = 96. Under its
    # two-row interface closures 8.2e-8 / 5.7e-9 / 4.9e-10, orders 3.84 /
    # 3.54, the divergence's one-sided rows binding here as well.
    errs10 = [_level_wave_error(N; deriv=lele_d1_10())
              for N in (48, 96, 192)]
    orders10 = [log2(errs10[i] / errs10[i+1]) for i in 1:2]
    @info "two-level entropy wave, C10" errs10 orders10
    @test errs10[1] < 1e-12
    @test errs10[3] < 5e-14
end

@testset "two levels: Sod shock through the refinement boundary" begin
    wall2 = (SlipWallBC(), SlipWallBC())
    per = (PeriodicBC(), PeriodicBC())
    ic(x, y, z) = x < 0.5 ? Prim(u=(0, 0, 0), p=1.0, rho=1.0) :
                            Prim(u=(0, 0, 0), p=0.1, rho=0.125)
    N = 201
    # Region x ∈ [0.6, 0.8]: the shock (speed ≈ 1.75) enters through the low
    # face at t ≈ 0.057 and leaves through the high face at t ≈ 0.17, so one
    # run crosses the boundary in both directions, with the Cook sensors and
    # the state filter live throughout — the in-situ form of the
    # sensor-injection measurement.
    solver = build_solver(
        Problem(domain=((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)), bcs=(wall2, per, per),
                ic=nothing),
        Numerics(n_global=(N, 1, 1), cfl=0.4,
                 amr=AMR(initial=BlockRegion((120, 0, 0), (41, 1, 1)), regrid_interval=0)))
    states = allocate_state(solver)
    initialize!(solver, states, ic)
    m0 = _two_level_mass(solver, states)
    run!(solver, states; tfinal=0.1, nmax=20000)
    # Ahead of the shock (x > 0.85) the exact solution is still quiescent, so
    # any momentum there beyond round-off is interface-generated noise.
    # Measured 1.4e-8 under the default filter, with the unrefined run at
    # 1.4e-9. The state filter's strength sets both: at αf = 0.45 they read
    # 3.4e-10 and 3.8e-10, and at 0.49 9.0e-7 and 4.3e-8.
    ps = PatchSolver(solver, solver.patches[1])
    pad = ps.decomp.n_halo_d[1]
    m1 = solver.equations.i_mom[1]
    noise = maximum(abs(states.patches[1][i + pad, 1, 1, m1]) for i in 172:N)
    @info "Sod through refinement boundary" noise
    @test noise < 1e-7
    # Positivity holds through both crossings (interior only; physical-edge
    # halos are never written and stay zero).
    for (psq, Q) in CL.eachpatch(solver, states)
        n = psq.decomp.n_local[1]
        padq = psq.decomp.n_halo_d[1]
        @test minimum(Q[i + padq, 1, 1, 1] for i in 1:n) > 0.05
    end
    # Run on to t = 0.2 (shock exits the region) and measure conservation.
    run!(solver, states; tfinal=0.2, nmax=40000)
    drift = abs(_two_level_mass(solver, states) - m0) / m0
    @info "two-level Sod mass drift" drift
    # Measured 7.0e-6 under the defaults, the coarse-fine correction and the
    # parent's derivative mask. The unrefined run drifts 5e-11.
    @test drift < 1e-5

    # The :d8 detector and the pentadiagonal filter at the same faces. The
    # momentum ahead of the shock measured 7.3e-6 (2.0e-6 on the unrefined
    # run with the same numerics) and the mass drift 5.4e-9.
    s8 = build_solver(
        Problem(domain=((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)), bcs=(wall2, per, per),
                ic=nothing),
        Numerics(n_global=(N, 1, 1), cfl=0.4, art=ArtificialProperties(detector=:d8),
                 filter=pyranda_filter(),
                 amr=AMR(initial=BlockRegion((120, 0, 0), (41, 1, 1)), regrid_interval=0)))
    q8 = allocate_state(s8)
    initialize!(s8, q8, ic)
    m8 = _two_level_mass(s8, q8)
    run!(s8, q8; tfinal=0.1, nmax=20000)
    p8 = PatchSolver(s8, s8.patches[1])
    @test maximum(abs(q8.patches[1][padded_index(p8, i, 1, 1), m1]) for i in 172:N) < 1e-4
    run!(s8, q8; tfinal=0.2, nmax=40000)
    @test s8.t ≈ 0.2
    for (psq, Q) in CL.eachpatch(s8, q8)
        @test minimum(Q[padded_index(psq, i, 1, 1), 1] for i in 1:psq.decomp.n_local[1]) > 0.05
    end
    @test abs(_two_level_mass(s8, q8) - m8) / m8 < 1e-5
end

@testset "two levels: 2-D refined region advects a smooth vortex field" begin
    # A genuinely multi-dimensional check that the tensor-product transfer
    # chain and the 2-D shell imposition hold together: entropy wave advected
    # diagonally through a 2-D refined region.
    per3l = ntuple(_ -> (PeriodicBC(), PeriodicBC()), 3)
    u0, v0 = 0.4, 0.3
    solver = build_solver(
        Problem(domain=((0.0, 2π), (0.0, 2π), (0.0, 1.0)), bcs=per3l, ic=nothing),
        Numerics(n_global=(48, 48, 1), art=ArtificialProperties(enabled=false),
                 filter=StateFilter(interval=0),
                 amr=AMR(initial=BlockRegion((18, 18, 0), (12, 12, 1)), regrid_interval=0)))
    states = allocate_state(solver)
    initialize!(solver, states, (x, y, z) ->
        Prim(u=(u0, v0, 0), p=1.0, rho=1.0 + 0.1 * sin(x) * sin(y)))
    run!(solver, states; tfinal=0.5)
    e = 0.0
    for (ps, Q) in CL.eachpatch(solver, states)
        n = ps.decomp.n_local
        for j in 1:n[2], i in 1:n[1]
            I = padded_index(ps, i, j, 1)
            exact = 1.0 + 0.1 * sin(xcoord(ps, 1, i) - u0 * solver.t) *
                          sin(xcoord(ps, 2, j) - v0 * solver.t)
            e = max(e, abs(Q[I, 1] - exact))
        end
    end
    @info "2-D refined advection" e
    # Measured 5.5e-11 (4.3e-8 under the closure rows).
    @test e < 1e-9
end

@testset "three levels: configuration and geometry" begin
    per3l = ntuple(_ -> (PeriodicBC(), PeriodicBC()), 3)
    r1 = BlockRegion((40, 0, 0), (16, 1, 1))   # level-1 nodes 121..166
    mk(r2; regrid_interval=0) = build_solver(
        Problem(domain=((0.0, 2π), (0.0, 1.0), (0.0, 1.0)), bcs=per3l, ic=nothing),
        Numerics(n_global=(96, 1, 1),
                 amr=AMR(initial=[r1, r2], regrid_interval=regrid_interval)))
    # Nesting is checked against the parent patch (level-1 node space),
    # not the root grid.
    @test_throws ErrorException mk(BlockRegion((122, 0, 0), (10, 1, 1)))
    @test_throws ErrorException mk(BlockRegion((150, 0, 0), (14, 1, 1)))
    @test_throws ErrorException mk(BlockRegion((130, 0, 0), (3, 1, 1)))
    # Regridding stays two-level.
    @test_throws ErrorException mk(BlockRegion((130, 0, 0), (20, 1, 1)),
                                   regrid_interval=5)
    solver = mk(BlockRegion((130, 0, 0), (20, 1, 1)))
    @test nlevels(solver) == 3
    @test npatches(solver) == 3
    @test [p.level for p in solver.patches] == [0, 1, 2]
    @test refined_region(solver, 2).offset == (130, 0, 0)
    p0 = PatchSolver(solver, solver.patches[1])
    p1 = PatchSolver(solver, solver.patches[2])
    p2 = PatchSolver(solver, solver.patches[3])
    @test p2.h[1] ≈ p0.h[1] / 9
    @test p2.decomp.n_local[1] == 3 * 20 - 2
    # Level-2 node 1 is level-1 node 131, patch-local 11; level-2 node 7 is
    # level-1 node 133, patch-local 13, which is root node 45.
    @test xcoord(p2, 1, 1) ≈ xcoord(p1, 1, 11) atol = 1e-14
    @test xcoord(p2, 1, 7) ≈ xcoord(p1, 1, 13) atol = 1e-14
    @test xcoord(p1, 1, 13) ≈ xcoord(p0, 1, 45) atol = 1e-14
end

@testset "three levels: manufactured solution across nested boundaries" begin
    for subcycle in (false, true)
        errs = [_level_wave_error(N; levels=3, subcycle=subcycle,
                                  cfl=subcycle ? SUBCYCLE_CFL : 0.5)
                for N in (48, 96, 192)]
        orders = [log2(errs[i] / errs[i+1]) for i in 1:2]
        @info "three-level entropy wave" subcycle errs orders
        # Measured 1.45e-10 / 1.90e-12 / 5.2e-14, orders 6.25 / 5.20, at the
        # global step and 1.44e-10 / 1.91e-12 / 5.9e-14, orders 6.24 / 5.01,
        # subcycled at SUBCYCLE_CFL: the two-level figures with a second
        # coarse-fine boundary pair inside the first. The N = 192 errors lie
        # near the round-off floor, which lowers the second order, so the
        # subcycled second order has the wider guard. Under the closure
        # rows at the default cfl, orders 3.31 / 3.74 and 3.35 / 3.79.
        @test orders[1] > 5.5
        @test orders[2] > (subcycle ? 4.5 : 5.0)
        @test errs[2] < 1e-11
    end
end

@testset "three levels: Sod through nested refinement boundaries" begin
    wall2 = (SlipWallBC(), SlipWallBC())
    per = (PeriodicBC(), PeriodicBC())
    ic(x, y, z) = x < 0.5 ? Prim(u=(0, 0, 0), p=1.0, rho=1.0) :
                            Prim(u=(0, 0, 0), p=0.1, rho=0.125)
    N = 201
    # The two-level Sod gate's region with a level-2 patch over the middle
    # of the level-1 patch (extent 121): the shock crosses four coarse-fine
    # boundaries with the sensors and filter live.
    solver = build_solver(
        Problem(domain=((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)), bcs=(wall2, per, per),
                ic=nothing),
        Numerics(n_global=(N, 1, 1), cfl=0.4,
                 amr=AMR(initial=[BlockRegion((120, 0, 0), (41, 1, 1)),
                             BlockRegion((390, 0, 0), (60, 1, 1))],
                         regrid_interval=0, subcycle=true)))
    states = allocate_state(solver)
    initialize!(solver, states, ic)
    run!(solver, states; tfinal=0.1, nmax=20000)
    ps = PatchSolver(solver, solver.patches[1])
    pad = ps.decomp.n_halo_d[1]
    m1 = solver.equations.i_mom[1]
    # Momentum ahead of the shock on the two-level gate's schedule (t = 0.1,
    # x > 0.85) is refinement-boundary noise; measured 3.1e-8 against the
    # two-level figure's 1.4e-8, so the inner boundary pair about doubles it.
    noise = maximum(abs(states.patches[1][i + pad, 1, 1, m1]) for i in 172:N)
    @info "three-level Sod noise ahead of the shock" noise
    @test noise < 1e-7
    run!(solver, states; tfinal=0.2, nmax=40000)
    @test all(all(isfinite, parent(Q)) for Q in states.patches)
    for (psq, Q) in CL.eachpatch(solver, states)
        n = psq.decomp.n_local[1]
        @test minimum(Q[padded_index(psq, i, 1, 1), 1] for i in 1:n) > 0.05
    end
end

@testset "tiled level: lattice cover, faces, and configuration guards" begin
    per3l = ntuple(_ -> (PeriodicBC(), PeriodicBC()), 3)
    mk(; tile=0) = build_solver(
        Problem(domain=((0.0, 2π), (0.0, 1.0), (0.0, 1.0)), bcs=per3l, ic=nothing),
        Numerics(n_global=(192, 1, 1),
                 amr=AMR(initial=BlockRegion((80, 0, 0), (32, 1, 1)), regrid_interval=0,
                         tile=tile)))
    @test_throws ErrorException mk(tile=2)
    @test_throws ErrorException mk(tile=-1)
    # Lattice cells of edge 8 meeting nodes 81..112: cells 10..13, each of
    # nine nodes sharing its planes with its neighbors.
    solver = mk(tile=8)
    regs = level_regions(solver, 1)
    @test [(r.offset[1], r.extent[1]) for r in regs] ==
          [(80, 9), (88, 9), (96, 9), (104, 9)]
    @test npatches(solver) == 5
    @test_throws ErrorException refined_region(solver)
    lev = solver.levels[2]
    @test length(lev.plane_pairs[1]) == 6 && length(lev.ghost_sends[1]) == 6
    @test lev.phases == (true, false, false)
    # Faces: the outer faces are parent-fed, the inner ones shared.
    @test [lt.imposed[1] for lt in lev.transfers] ==
          [(true, false), (false, false), (false, false), (false, true)]
    @test solver.patches[2].faces[1] == (0, 2)
    @test solver.patches[3].faces[1] == (1, 3)
    @test solver.patches[2].bcs[1][2] isa InterfaceBC
    @test !CL.parent_fed(solver.patches[2].bcs[1][2])
    @test solver.patches[2].bcs[1][1] == CoarseFineBC()
    @test CL.parent_fed(solver.patches[2].bcs[1][1])
    # One face-condition type on every tile, so every tile of a level has the
    # one Patch type and the drivers compile once per level, not per face
    # pattern (see `InterfaceBC`).
    @test allequal(typeof(solver.patches[i]) for i in 2:length(solver.patches))
    # A box node on a lattice plane pulls in the cell on its inner side only.
    @test CL._tile_span(81, 112, 8) == 10:13
    @test CL._tile_span(82, 113, 8) == 10:13
    @test CL._tile_span(81, 81, 8) == 9:10
    # A tile that would leave the nesting margin is clipped, and dropped when
    # fewer than four nodes remain: region 5..24 at margin 4 keeps cell 0
    # clipped to nodes 5..9, then cells 1 and 2.
    s2 = build_solver(
        Problem(domain=((0.0, 2π), (0.0, 1.0), (0.0, 1.0)), bcs=per3l, ic=nothing),
        Numerics(n_global=(192, 1, 1),
                 amr=AMR(initial=BlockRegion((4, 0, 0), (20, 1, 1)), regrid_interval=0,
                         tile=8)))
    @test [(r.offset[1], r.extent[1]) for r in level_regions(s2, 1)] ==
          [(4, 5), (8, 9), (16, 9)]
end

@testset "tiled level: manufactured solution across tile and level faces" begin
    errs = [_level_wave_error(N; tile=8) for N in (48, 96, 192)]
    orders = [log2(errs[i] / errs[i+1]) for i in 1:2]
    @info "tiled entropy wave" errs orders
    # Measured 1.2e-10 / 1.9e-12 / 3.5e-14, orders 5.99 / 5.77, beside the
    # one-patch level's 1.5e-10 / 2.0e-12 / 3.3e-14: the tile interfaces
    # inside the level cost nothing visible at N = 192.
    @test all(>(5.0), orders)
    @test errs[3] < 1e-12
    # Subcycled, one tiled level at SUBCYCLE_CFL: measured 3.6e-14 against
    # the one-patch level's 3.4e-14.
    es = _level_wave_error(192; tile=8, subcycle=true, cfl=SUBCYCLE_CFL)
    @info "subcycled tiled entropy wave" es
    @test es < 1e-13
end

@testset "tiled level: 2-D tile nest with corners" begin
    per3l = ntuple(_ -> (PeriodicBC(), PeriodicBC()), 3)
    u0, v0 = 0.4, 0.3
    function vortex(; tile, subcycle)
        solver = build_solver(
            Problem(domain=((0.0, 2π), (0.0, 2π), (0.0, 1.0)), bcs=per3l, ic=nothing),
            Numerics(n_global=(48, 48, 1), art=ArtificialProperties(enabled=false),
                     filter=StateFilter(interval=0),
                     amr=AMR(initial=BlockRegion((18, 18, 0), (12, 12, 1)),
                             regrid_interval=0, subcycle=subcycle, tile=tile)))
        states = allocate_state(solver)
        initialize!(solver, states, (x, y, z) ->
            Prim(u=(u0, v0, 0), p=1.0, rho=1.0 + 0.1 * sin(x) * sin(y)))
        run!(solver, states; tfinal=0.5)
        e = 0.0
        for (ps, Q) in CL.eachpatch(solver, states)
            n = ps.decomp.n_local
            for j in 1:n[2], i in 1:n[1]
                I = padded_index(ps, i, j, 1)
                exact = 1.0 + 0.1 * sin(xcoord(ps, 1, i) - u0 * solver.t) *
                              sin(xcoord(ps, 2, j) - v0 * solver.t)
                e = max(e, abs(Q[I, 1] - exact))
            end
        end
        return e, npatches(solver)
    end
    # Four 7×7 tiles meeting at a corner, both stepping modes: measured
    # 5.5e-11 at the global step and 6.2e-11 subcycled against the one-patch
    # level's 5.5e-11, so the corner ghosts and the two-way shared faces are
    # consistent.
    for subcycle in (false, true)
        e, np = vortex(tile=6, subcycle=subcycle)
        @info "2-D tile nest" subcycle e
        @test np == 5
        @test e < 1e-9
    end
end

@testset "tiled level: multi-tile corner consensus and corner ghosts" begin
    # Four tiles meet at one fine node. Pairwise averaging in one flat pass
    # leaves the four copies unequal (a later pair reads a value an earlier
    # pair changed); the dimension-phased sync gives every copy the mean,
    # and the phase-2 strips, padded along x, fill the diagonal ghosts.
    per3l = ntuple(_ -> (PeriodicBC(), PeriodicBC()), 3)
    solver = build_solver(
        Problem(domain=((0.0, 2π), (0.0, 2π), (0.0, 1.0)), bcs=per3l, ic=nothing),
        Numerics(n_global=(48, 48, 1),
                 amr=AMR(initial=BlockRegion((18, 18, 0), (12, 12, 1)), regrid_interval=0,
                         tile=6)))
    states = allocate_state(solver)
    for (i, Q) in enumerate(states.patches)
        fill!(parent(Q), i - 1.0)              # tiles 2..5 carry 1..4
    end
    lev = solver.levels[2]
    @test lev.phases == (true, true, false)
    CL._sync_level_records!(solver, states.patches, lev)
    # Tile order is lattice order: A (18,18), B (24,18), C (18,24), D (24,24),
    # each 19 fine nodes; the common corner is A(19,19), B(1,19), C(19,1),
    # D(1,1).
    at(p, i, j) = (ps = PatchSolver(solver, solver.patches[p]);
                   pad = ps.decomp.n_halo_d; states.patches[p][i + pad[1], j + pad[2], 1 + pad[3], 1])
    @test at(2, 19, 19) == at(3, 1, 19) == at(4, 19, 1) == at(5, 1, 1) == 2.5
    # Two-tile planes away from the corner take the pair mean.
    @test at(2, 19, 10) == at(3, 1, 10) == 1.5
    @test at(2, 10, 19) == at(4, 10, 1) == 2.0
    # Diagonal corner ghosts hold the diagonal neighbor's interior.
    @test at(5, 0, 0) == 1.0 && at(2, 20, 20) == 4.0
    @test at(3, 0, 20) == 3.0 && at(4, 20, 0) == 2.0
    # Face ghosts hold the face neighbor's interior, and a parent-fed
    # ghost is untouched.
    @test at(5, 0, 10) == 3.0 && at(5, 10, 0) == 2.0
    @test at(2, 0, 10) == 1.0
    # Idempotent: a second pass changes nothing.
    snapshot = [copy(parent(Q)) for Q in states.patches]
    CL._sync_level_records!(solver, states.patches, lev)
    @test all(parent(states.patches[i]) == snapshot[i] for i in eachindex(states.patches))
end

@testset "tiled level: one RHS workspace per padded extent" begin
    # A rank advances its patches in sequence and nothing in the RHS scratch
    # outlives the evaluation that filled it, so that scratch is pooled on the
    # padded local extent (patches.jl): a lattice level's tiles, whose extents
    # are equal by construction, hold one set between them. Nothing else in the
    # gate moves if the pooling silently stops. The velocity gradients are the
    # exception on a level that computes its artificial coefficients
    # level-wide, as this one does: each tile holds its own. One set per
    # extent is the layout of tiles evaluated one after another, which a
    # session of one thread takes; `TILE_SLOTS` holds it at any thread count.
    per3l = ntuple(_ -> (PeriodicBC(), PeriodicBC()), 3)
    solver = CL.with_knob(CL.TILE_SLOTS => 1) do
        build_solver(
            Problem(domain=((0.0, 2π), (0.0, 2π), (0.0, 1.0)), bcs=per3l, ic=nothing),
            Numerics(n_global=(48, 48, 1),
                     amr=AMR(initial=BlockRegion((18, 18, 0), (12, 12, 1)),
                             regrid_interval=0, tile=6)))
    end
    padded(p) = ntuple(d -> p.decomp.n_local[d] + 2 * p.decomp.n_halo_d[d], 3)
    extents = unique(padded(p) for p in solver.patches)
    sets = unique(objectid(p.rhs_workspace.tmp_a) for p in solver.patches)
    tiles = solver.patches[2:end]
    shared(f) = all(f(p.rhs_workspace) === f(tiles[1].rhs_workspace) for p in tiles)
    @test length(solver.patches) == 5          # the root slab plus four tiles
    @test length(extents) == 2                 # the root's extent and the tiles'
    @test length(sets) == length(extents)
    @test shared(w -> w.tmp_a) && shared(w -> w.flux) && shared(w -> w.sensor)
    @test shared(w -> w.strain_mag) && shared(w -> w.sensors_filled_by)
    @test length(unique(objectid(p.rhs_workspace.grad_u[1, 1]) for p in tiles)) == 4
    @test all(p.field_tuples.grad_u[2, 1] === p.rhs_workspace.grad_u[2, 1] for p in tiles)
    @test solver.patches[1].rhs_workspace.tmp_a !== solver.patches[2].rhs_workspace.tmp_a
    # Whatever a patch is handed carries that patch's own extent.
    @test all(size(p.rhs_workspace.tmp_a) == padded(p) for p in solver.patches)
    @test all(size(p.rhs_workspace.flux[1, 1]) == padded(p) for p in solver.patches)
    # The single-patch property forwarding serves the workspace's own arrays,
    # and the launchable wrappers hold those same objects.
    single = build_solver(
        Problem(domain=((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)), bcs=per3l, ic=nothing),
        Numerics(n_global=(16, 16, 1)))
    ws = single.patches[1].rhs_workspace
    @test single.tmp_a === ws.tmp_a && single.tmp_b === ws.tmp_b
    @test single.sensor === ws.sensor && single.strain_mag === ws.strain_mag
    @test single.grad_u === ws.grad_u && single.flux === ws.flux
    @test single.field_tuples.grad_u[1, 1] === ws.grad_u[1, 1]
    @test single.field_tuples.flux[1, 1] === ws.flux[1, 1]
    @test single.field_tuples.Y[1] === single.patches[1].Y[1]
end

@testset "tiled level: a scratch read names the patch that wrote it" begin
    # The tiles below share one RHS workspace, so after a step its sensors
    # are the last tile's. scalar_field refuses them for any other tile rather
    # than return that tile's values. Their velocity gradients are each
    # tile's own, the level computing its artificial coefficients level-wide,
    # so the fields derived from them read for every tile; with the artificial
    # properties off the tiles share those too, and the read is refused.
    per3l = ntuple(_ -> (PeriodicBC(), PeriodicBC()), 3)
    initial(x, y, z) =
        Prim(rho=1 + 0.2sin(x) * cos(y), u=(0.3sin(y), 0.2cos(x), 0.0), p=1.0)
    gradient_names = (:divergence, :qcriterion, :vorticity_magnitude)
    # The tiles share one workspace as a session of one thread lays them out
    # (`TILE_SLOTS`, see the testset above).
    solver, plain = CL.with_knob(CL.TILE_SLOTS => 1) do
        build_solver(
            Problem(domain=((0.0, 2π), (0.0, 2π), (0.0, 1.0)), bcs=per3l, ic=nothing),
            Numerics(n_global=(48, 48, 1),
                     amr=AMR(initial=BlockRegion((18, 18, 0), (12, 12, 1)),
                             regrid_interval=0, tile=6))),
        build_solver(
            Problem(domain=((0.0, 2π), (0.0, 2π), (0.0, 1.0)), bcs=per3l, ic=nothing),
            Numerics(n_global=(48, 48, 1), art=ArtificialProperties(enabled=false),
                     amr=AMR(initial=BlockRegion((18, 18, 0), (12, 12, 1)),
                             regrid_interval=0, tile=6)))
    end
    states = allocate_state(solver)
    initialize!(solver, states, initial)
    run!(solver, states; tfinal=1.0, nmax=1)
    patches = solver.patches
    first_tile, last_tile = patches[2], patches[end]
    ws = first_tile.rhs_workspace
    @test ws.sensor === last_tile.rhs_workspace.sensor
    @test ws.gradients_filled_by[] === first_tile.amr.covered
    @test last_tile.rhs_workspace.gradients_filled_by[] === last_tile.amr.covered
    @test ws.sensors_filled_by[] === last_tile.amr.covered
    @test patches[1].rhs_workspace.sensors_filled_by[] === patches[1].amr.covered
    ps_first = CL.PatchSolver(solver, first_tile)
    ps_last = CL.PatchSolver(solver, last_tile)
    for name in (:sensor, :strain_mag)
        @test_throws ArgumentError CL.scalar_field(ps_first, name)
        @test CL.scalar_field(ps_last, name) isa AbstractArray
    end
    for name in gradient_names
        @test CL.scalar_field(ps_first, name) isa AbstractArray
        @test CL.scalar_field(ps_last, name) isa AbstractArray
    end
    plain_states = allocate_state(plain)
    initialize!(plain, plain_states, initial)
    run!(plain, plain_states; tfinal=1.0, nmax=1)
    plain_first, plain_last = plain.patches[2], plain.patches[end]
    @test plain_first.rhs_workspace === plain_last.rhs_workspace
    @test plain_first.rhs_workspace.gradients_filled_by[] === plain_last.amr.covered
    for name in gradient_names
        @test_throws ArgumentError CL.scalar_field(CL.PatchSolver(plain, plain_first), name)
        @test CL.scalar_field(CL.PatchSolver(plain, plain_last), name) isa AbstractArray
    end
    # What a silent read would have returned differs from the tile's own field.
    # The strain magnitude is a pointwise function of the tile's gradients, so
    # the tile's own pass reproduces the level's in the interior.
    nodes = ntuple(d -> first_tile.decomp.n_halo_d[d] .+
                        (1:first_tile.decomp.n_local[d]), 3)
    stale = Array(CL.scalar_field(ps_last, :strain_mag))
    own = field_array(solver, states, :strain_mag)
    @test own[2][nodes...] != stale[nodes...]
    # Recomputed for the first tile, the read is that tile's and passes.
    fresh = CL.preserving_artificial(solver) do
        CL.compute_primitives_and_gradients!(ps_first, states.patches[2])
        CL.compute_artificial!(ps_first, states.patches[2])
        Array(CL.scalar_field(ps_first, :strain_mag))
    end
    @test fresh[nodes...] == own[2][nodes...]
    @test_throws ArgumentError CL.scalar_field(ps_last, :sensor)
    # A surviving tile keeps its identity through a regrid's repatching.
    @test CL._repatch(first_tile, 99, first_tile.faces, first_tile.bcs,
                      first_tile.comm).amr.covered === first_tile.amr.covered
    # The multi-patch forward names the scratch as shared.
    err = try
        solver.sensor
    catch e
        e
    end
    @test occursin("right-hand-side scratch", sprint(showerror, err))
end

@testset "tiled level: concurrent tiles take the arithmetic of sequential ones" begin
    # Tiles spread over several workspace sets and evaluated as tasks
    # (`_foreach_tile`) give the state of tiles evaluated in level order on one
    # set, bit for bit: with and without subcycling, through the level-wide
    # artificial-property pass, the coarse-fine junction captures, each tile
    # holding its own deferral, and the shell impositions.
    # `TILE_SLOTS` forces the concurrent path at any thread count; on one
    # thread the tasks run one after another in another order than the level's.
    per3l = ntuple(_ -> (PeriodicBC(), PeriodicBC()), 3)
    initial(x, y, z) =
        Prim(rho=1 + 0.2sin(x) * cos(y), u=(0.3sin(y), 0.2cos(x), 0.0),
             p=1.0 + 0.1exp(-4((x - π)^2 + (y - π)^2)))
    function evolve(slots, subcycle)
        CL.with_knob(CL.TILE_SLOTS => slots) do
            s = build_solver(
                Problem(domain=((0.0, 2π), (0.0, 2π), (0.0, 1.0)), bcs=per3l, ic=nothing),
                Numerics(n_global=(48, 48, 1),
                         amr=AMR(initial=BlockRegion((18, 18, 0), (12, 12, 1)),
                                 regrid_interval=0, subcycle=subcycle, tile=6)))
            q = allocate_state(s)
            initialize!(s, q, initial)
            run!(s, q; tfinal=1.0, nmax=3)
            sets = length(unique(objectid(p.rhs_workspace.tmp_a)
                                 for p in s.patches[2:end]))
            # The interiors: a halo may hold values no pass reads.
            inner(p, x) = parent(x)[ntuple(d -> p.decomp.n_halo_d[d] .+
                                              (1:p.decomp.n_local[d]), 3)..., :]
            [inner(p, x) for (p, x) in zip(s.patches, q.patches)], s, sets
        end
    end
    for subcycle in (false, true)
        serial, s1, sets1 = evolve(1, subcycle)
        tasks, s3, sets3 = evolve(3, subcycle)
        @test sets1 == 1 && sets3 == 3
        @test any(p -> !isempty(p.amr.reflux_captures), s3.patches[2:end])
        @test length(serial) == length(tasks)
        @test all(a == b for (a, b) in zip(serial, tasks))
        @test s1.t == s3.t
    end
end

@testset "tiled level: artificial coefficients across shared faces" begin
    # Inside one tile the strain magnitude has no ghosts at a face shared with
    # another tile, so a sensor built tile by tile clamps its detector and
    # closes its smoother there. A level with shared faces computes its
    # coefficients over the whole level instead (`_level_artificial!`), and
    # four tiles then carry those of one patch over the same region but for
    # the tiles' own gradient rows at the shared faces.
    per3l = ntuple(_ -> (PeriodicBC(), PeriodicBC()), 3)
    ic(x, y, z) = Prim(rho=1 + 0.2sin(x) * cos(y), u=(0.3sin(2y), 0.2cos(3x), 0.0),
                       p=1.0)
    mk(tile, region) = build_solver(
        Problem(domain=((0.0, 2π), (0.0, 2π), (0.0, 1.0)), bcs=per3l, ic=nothing),
        Numerics(n_global=(48, 48, 1),
                 amr=AMR(initial=region, regrid_interval=0, tile=tile)))
    st = mk(6, BlockRegion((18, 18, 0), (12, 12, 1)))
    # The box over the union of the tiles, which the lattice widens past the
    # requested region.
    regs = level_regions(st, 1)
    lo = ntuple(d -> minimum(r.offset[d] for r in regs), 3)
    hi = ntuple(d -> maximum(r.offset[d] + r.extent[d] for r in regs), 3)
    sb = mk(0, BlockRegion(lo, hi .- lo))
    qt, qb = allocate_state(st), allocate_state(sb)
    initialize!(st, qt, ic)
    initialize!(sb, qb, ic)
    @test CL._level_sensors(st, st.levels[2])
    @test !CL._level_sensors(sb, sb.levels[2])
    @test all(length(p.amr.sensed_fields) == 1 for p in st.patches[2:end])
    @test isempty(sb.patches[2].amr.sensed_fields)
    CL._prime_coefficients!(st, qt, Workspace(qt))
    CL._prime_coefficients!(sb, qb, Workspace(qb))
    box = sb.patches[2]
    inner(p) = ntuple(d -> p.decomp.n_halo_d[d] .+ (1:p.decomp.n_local[d]), 3)
    # The largest difference from the box over the four tiles, relative to
    # the box's peak.
    function deviation(name, tiles=[getfield(p, name) for p in st.patches],
                       B=getfield(box, name))
        ib = inner(box)
        peak = maximum(abs, view(B, ib...))
        worst = 0.0
        for (li, p) in enumerate(st.patches)
            p.level == 1 || continue
            it = inner(p)
            o = p.region.offset .- box.region.offset
            A = tiles[li]
            for j in eachindex(it[2]), i in eachindex(it[1])
                b = B[ib[1][o[1] + i], ib[2][o[2] + j], ib[3][1]]
                worst = max(worst, abs(A[it[1][i], it[2][j], it[3][1]] - b) / peak)
            end
        end
        return worst
    end
    level = Dict(name => deviation(name) for name in (:mu_art, :kappa_art))
    primed = Dict(name => [copy(getfield(p, name)) for p in st.patches]
                  for name in (:mu_art, :beta_art, :kappa_art))
    # The same tiles each computing their own, as a tile without a shared face
    # does.
    for (ps, Q) in CL.eachpatch(st, qt)
        ps.patch.level == 1 || continue
        CL.compute_primitives_and_gradients!(ps, Q)
        CL.compute_artificial!(ps, Q)
    end
    own = Dict(name => deviation(name) for name in (:mu_art, :kappa_art))
    @info "tiled coefficients against one patch" level own
    # Measured 9.4e-7 (μ*, the tiles' gradient rows at the shared faces) and
    # 2.8e-16 (κ*, whose field has no derivative), against 0.53 and 0.11
    # tile by tile.
    @test level[:mu_art] < 1e-5
    @test level[:kappa_art] < 1e-13
    @test own[:mu_art] > 0.1 && own[:kappa_art] > 0.01
    # Output recomputes the level-wide coefficients, those of the level's
    # right-hand side, and restores the tiles' own that the arrays now hold.
    tile_values = [copy(p.mu_art) for p in st.patches]
    for name in (:mu_art, :beta_art, :kappa_art)
        @test field_array(st, qt, name) == primed[name]
    end
    @test all(p.mu_art == a for (p, a) in zip(st.patches, tile_values))
    snaps = field_snapshot(st, qt; fields=[:mu_art])
    @test count(snaps) do snap
        li = findfirst(p -> p.level == snap.level && p.region.offset == snap.offset,
                       st.patches)
        snap[:mu_art] == primed[:mu_art][li][inner(st.patches[li])...]
    end == length(st.patches)
    # The level's smoothed internal-energy sensor, held per tile through the
    # pass, is the box's up to the order of a sum.
    @test deviation(:sensor, field_array(st, qt, :sensor),
                    field_array(sb, qb, :sensor)[2]) < 1e-13
end

@testset "tiled level: sharpening gradients across shared faces" begin
    # The sharpening flux builds its pair normals and its gate from smoothed
    # volume-fraction gradients. A level with shared faces smooths them over
    # the whole level (`_level_sharpening_gradients!`), as one patch over the
    # same region does, but for the tiles' own derivative rows at the shared
    # faces; smoothed tile by tile they take the closure rows of a field
    # without ghosts there.
    per3l = ntuple(_ -> (PeriodicBC(), PeriodicBC()), 3)
    eos = IdealMixture([IdealSpecies{Float64}("light", 1.0, 1.4),
                        IdealSpecies{Float64}("heavy", 0.2, 1.09)])
    art = ArtificialProperties(C_sharpen=0.5)
    # A heavy-gas band at uniform p and T, its volume fraction `v`.
    function ic(x, y, z)
        v = 0.5 + 0.5tanh((sin(x) + 0.5cos(y)) / 0.3)
        rho = (1 - v) + 5v
        return Prim(Y=((1 - v) / rho, 5v / rho), rho=rho, u=(0.1, 0.0, 0.0), p=1.0)
    end
    mk(tile, region) = build_solver(
        Problem(domain=((0.0, 2π), (0.0, 2π), (0.0, 1.0)), bcs=per3l, eos=eos, ic=nothing),
        Numerics(n_global=(48, 48, 1), art=art,
                 amr=AMR(initial=region, regrid_interval=0, tile=tile)))
    st = mk(6, BlockRegion((18, 18, 0), (12, 12, 1)))
    regs = level_regions(st, 1)
    lo = ntuple(d -> minimum(r.offset[d] for r in regs), 3)
    hi = ntuple(d -> maximum(r.offset[d] + r.extent[d] for r in regs), 3)
    sb = mk(0, BlockRegion(lo, hi .- lo))
    qt, qb = allocate_state(st), allocate_state(sb)
    initialize!(st, qt, ic)
    initialize!(sb, qb, ic)
    @test CL._level_sensors(st, st.levels[2])
    # The strain magnitude, then one gradient slot per dimension.
    @test all(length(p.amr.sensed_fields) == 4 for p in st.patches[2:end])
    @test isempty(sb.patches[2].amr.sensed_fields)
    CL._prime_coefficients!(st, qt, Workspace(qt))
    CL._prime_coefficients!(sb, qb, Workspace(qb))
    # The smoothed gradients a patch forms alone, from its current primitives.
    function own(ps)
        g = ntuple(_ -> zero(ps.rho), 3)
        CL._volume_fraction_gradients!(g, ps, 1)
        foreach(d -> CL.smooth!(g[d], ps), 1:2)
        return g
    end
    box = sb.patches[2]
    reference = own(CL.PatchSolver(sb, box))
    inner(p) = ntuple(d -> p.decomp.n_halo_d[d] .+ (1:p.decomp.n_local[d]), 3)
    # The largest difference from the box over the tiles along `d`, relative
    # to the box's peak.
    function deviation(gradient, d)
        ib = inner(box)
        B = reference[d]
        peak = maximum(abs, view(B, ib...))
        worst = 0.0
        for p in st.patches
            p.level == 1 || continue
            it = inner(p)
            o = p.region.offset .- box.region.offset
            A = gradient(p)
            for j in eachindex(it[2]), i in eachindex(it[1])
                b = B[ib[1][o[1] + i], ib[2][o[2] + j], ib[3][1]]
                worst = max(worst, abs(A[it[1][i], it[2][j], it[3][1]] - b) / peak)
            end
        end
        return worst
    end
    level = [deviation(p -> CL._sharpen_gradient(p, art, 1, d), d) for d in 1:2]
    tiles = [deviation(p -> own(CL.PatchSolver(st, p))[d], d) for d in 1:2]
    @info "tiled sharpening gradients against one patch" level tiles
    # Measured 4.7e-6 along x over the level, from the tiles' derivative rows,
    # and 4.0e-16 along y, against 0.047 and 0.044 tile by tile.
    @test maximum(level) < 3e-5
    @test minimum(tiles) > 0.02
end
