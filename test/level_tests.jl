# Two-level refinement tests: manufactured smooth solution spanning the
# coarse-fine boundary, a Sod shock through the refinement boundary in both
# directions with Cook sensors live, conservation drift, and the
# configuration guards; then the subcycled, tagged, and regridded variants.
# Serial; the decomposed forms live in mpi_tests.jl.
#
# Guards are measured values with headroom, per the convergence-suite
# convention. The interpolation/injection coupling and the reasons the
# invertible filter pair is not the default are documented at the top of
# src/levels.jl.

@testset "refined solver rejects unsupported configurations" begin
    per3l = ntuple(_ -> (PeriodicBC(), PeriodicBC()), 3)
    mk(; kw...) = Solver(; n_global=(96, 1, 1), L_domain=(2π, 1.0, 1.0),
                         bcs=per3l, refine=BlockRegion((40, 0, 0), (16, 1, 1)),
                         kw...)
    @test_throws ErrorException mk(deriv=lele_d1_10())
    @test_throws ErrorException mk(art=ArtParams(detector=:d8))
    @test_throws ErrorException mk(metric=CylindricalMetric())
    @test_throws ErrorException mk(patch_grid=(2, 1, 1))
    # Nesting margin: a region reaching the boundary is refused.
    @test_throws ErrorException Solver(n_global=(96, 1, 1), L_domain=(2π, 1, 1),
                                       bcs=per3l,
                                       refine=BlockRegion((0, 0, 0), (16, 1, 1)))
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

function _level_wave_error(N; mode=:inject, tfinal=0.5, subcycle=false,
                           levels=2, tile=0)
    per3l = ntuple(_ -> (PeriodicBC(), PeriodicBC()), 3)
    u0 = 0.5
    r1 = BlockRegion((N ÷ 2 - N ÷ 12, 0, 0), (N ÷ 6, 1, 1))
    # A third level over the middle half of the level-1 patch, in level-1
    # node space, so the nest scales with N.
    e1 = 3 * (N ÷ 6) - 2
    r2 = BlockRegion((3 * r1.offset[1] + e1 ÷ 4, 0, 0), (e1 ÷ 2, 1, 1))
    solver = Solver(n_global=(N, 1, 1), L_domain=(2π, 1.0, 1.0), bcs=per3l,
                    art=ArtParams(enabled=false), filter_interval=0,
                    level_restriction=mode, subcycle=subcycle, tile=tile,
                    refine=levels == 3 ? [r1, r2] : r1)
    states = allocate_state(solver)
    initialize!(solver, states, (x, y, z) ->
        Prim(u=(u0, 0, 0), p=1.0, rho=1.0 + 0.2 * sin(x)))
    run!(solver, states; tfinal=tfinal)
    e = 0.0
    for (ps, Q) in CL.eachpatch(solver, states)
        for i in 1:ps.decomp.n_local[1]
            I = gidx(ps, i, 1, 1)
            e = max(e, abs(Q[I, 1] - (1.0 + 0.2 * sin(xcoord(ps, 1, i) -
                                                      u0 * solver.t))))
        end
    end
    return e
end

@testset "two levels: manufactured solution across the coarse-fine boundary" begin
    errs = [_level_wave_error(N) for N in (48, 96, 192)]
    orders = [log2(errs[i] / errs[i+1]) for i in 1:2]
    @info "two-level entropy wave" errs orders
    # Measured 8.5e-8 / 7.7e-9 / 6.2e-10, orders 3.46 / 3.64: the one-sided
    # divergence closures bind, as at a same-level patch interface.
    @test all(>(3.0), orders)
    @test errs[2] < 3e-8
    # The deconvolution/filter coupling measures order 1.3-1.7 (see the
    # src/levels.jl header); pin that it stays selectable and stable.
    ef = _level_wave_error(96; mode=:filter)
    @test 1e-5 < ef < 1e-4
end

# Trapezoid mass over the two-level composite: the coarse level outside the
# covered region plus the fine level inside it, the shared boundary plane
# taking half weight from each.
function _two_level_mass(solver, states, N)
    region = CL.refined_region(solver)
    ps = PatchSolver(solver, solver.patches[1])
    pad = ps.decomp.n_halo_d[1]
    h = ps.h[1]
    lo = region.offset[1] + 1
    hi = region.offset[1] + region.extent[1]
    m = 0.0
    for i in 1:N
        w = (i == 1 || i == N) ? 0.5 : 1.0
        (lo < i < hi) && continue
        (i == lo || i == hi) && (w = 0.5)
        m += w * states[1][i + pad, 1, 1, 1] * h
    end
    pf = PatchSolver(solver, solver.patches[2])
    padf = pf.decomp.n_halo_d[1]
    hf = pf.h[1]
    nf = pf.decomp.n_local[1]
    for i in 1:nf
        w = (i == 1 || i == nf) ? 0.5 : 1.0
        m += w * states[2][i + padf, 1, 1, 1] * hf
    end
    return m
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
    solver = Solver(n_global=(N, 1, 1), L_domain=(1.0, 1.0, 1.0),
                    bcs=(wall2, per, per), cfl=0.4,
                    refine=BlockRegion((120, 0, 0), (41, 1, 1)))
    states = allocate_state(solver)
    initialize!(solver, states, ic)
    m0 = _two_level_mass(solver, states, N)
    run!(solver, states; tfinal=0.1, nmax=20000)
    # Ahead of the shock (x > 0.85) the exact solution is still quiescent, so
    # any momentum there beyond round-off is interface-generated noise.
    # Measured 6.4e-10 with the unrefined run at 1.9e-10.
    ps = PatchSolver(solver, solver.patches[1])
    pad = ps.decomp.n_halo_d[1]
    m1 = solver.equations.i_mom[1]
    noise = maximum(abs(states[1][i + pad, 1, 1, m1]) for i in 172:N)
    @info "Sod through refinement boundary" noise
    @test noise < 1e-8
    # Positivity holds through both crossings (interior only; physical-edge
    # halos are never written and stay zero).
    for (psq, Q) in CL.eachpatch(solver, states)
        n = psq.decomp.n_local[1]
        padq = psq.decomp.n_halo_d[1]
        @test minimum(Q[i + padq, 1, 1, 1] for i in 1:n) > 0.05
    end
    # Run on to t = 0.2 (shock exits the region) and measure conservation.
    run!(solver, states; tfinal=0.2, nmax=40000)
    drift = abs(_two_level_mass(solver, states, N) - m0) / m0
    @info "two-level Sod mass drift" drift
    # Measured 1.36e-4 (:inject; :filter halves it at three decades of smooth
    # accuracy — src/levels.jl header). The unrefined run drifts 5e-11.
    @test drift < 5e-4
end


@testset "two levels: 2-D refined region advects a smooth vortex field" begin
    # A genuinely multi-dimensional check that the tensor-product transfer
    # chain and the 2-D shell imposition hold together: entropy wave advected
    # diagonally through a 2-D refined region.
    per3l = ntuple(_ -> (PeriodicBC(), PeriodicBC()), 3)
    u0, v0 = 0.4, 0.3
    solver = Solver(n_global=(48, 48, 1), L_domain=(2π, 2π, 1.0), bcs=per3l,
                    art=ArtParams(enabled=false), filter_interval=0,
                    refine=BlockRegion((18, 18, 0), (12, 12, 1)))
    states = allocate_state(solver)
    initialize!(solver, states, (x, y, z) ->
        Prim(u=(u0, v0, 0), p=1.0, rho=1.0 + 0.1 * sin(x) * sin(y)))
    run!(solver, states; tfinal=0.5)
    e = 0.0
    for (ps, Q) in CL.eachpatch(solver, states)
        n = ps.decomp.n_local
        for j in 1:n[2], i in 1:n[1]
            I = gidx(ps, i, j, 1)
            exact = 1.0 + 0.1 * sin(xcoord(ps, 1, i) - u0 * solver.t) *
                          sin(xcoord(ps, 2, j) - v0 * solver.t)
            e = max(e, abs(Q[I, 1] - exact))
        end
    end
    @info "2-D refined advection" e
    # Measured 4.3e-8.
    @test e < 1e-6
end

# --- Subcycling, tagging, regridding -----------------------------------------

@testset "subcycle and regrid configuration guards" begin
    per3l = ntuple(_ -> (PeriodicBC(), PeriodicBC()), 3)
    mk(; kw...) = Solver(; n_global=(96, 1, 1), L_domain=(2π, 1.0, 1.0),
                         bcs=per3l, kw...)
    # Both features require an initial refined region.
    @test_throws ErrorException mk(subcycle=true)
    @test_throws ErrorException mk(regrid_interval=10)
    @test_throws ErrorException mk(refine=BlockRegion((40, 0, 0), (16, 1, 1)),
                                   regrid_interval=-1)
end

@testset "level rank subsets: sizing and the serial ownership" begin
    # `_level_ranks` sizes a level's rank subset from the largest count for
    # which every tile clears the C8 filter's nine-point minimum. The numbers
    # below are the ones the MPI suite's subset checks are built on, pinned
    # here because a serial run exercises the sizing rule without ever
    # reaching a split communicator.
    act = (true, false, false)
    r(ext) = BlockRegion((96, 0, 0), (ext, 1, 1))
    # 4 coarse nodes are 10 fine ones: one rank, since even 10 ÷ 2 = 5 < 9.
    @test CL._level_ranks([r(4)], act, 1) == 1
    @test CL._level_ranks([r(4)], act, 8) == 1
    # 8 coarse nodes are 22 fine: two ranks (22 ÷ 3 = 7 falls short).
    @test CL._level_ranks([r(8)], act, 2) == 2
    @test CL._level_ranks([r(8)], act, 8) == 2
    # 24 coarse nodes are 70 fine: seven, not eight (70 ÷ 8 = 8).
    @test CL._level_ranks([r(24)], act, 8) == 7
    # A level takes the whole set whenever it fits, and the smallest tile of
    # the set binds.
    @test CL._level_ranks([r(64)], act, 4) == 4
    @test CL._level_ranks([r(64), r(4)], act, 4) == 1
    # Per-tile owners: four 9-node tiles (25 fine, two ranks each at most).
    t(k) = BlockRegion((80 + 8k, 0, 0), (9, 1, 1))
    four = [t(0), t(1), t(2), t(3)]
    # One tile reduces to `_level_ranks`.
    @test CL._tile_owners([t(0)], act, 8) == ([0:1], 2)
    # Ranks at least as many as tiles: each tile its own range, by weight,
    # capped by what it admits; the level takes the union.
    @test CL._tile_owners(four, act, 4) == ([0:0, 1:1, 2:2, 3:3], 4)
    @test CL._tile_owners(four, act, 8) == ([0:1, 2:3, 4:5, 6:7], 8)
    @test CL._tile_owners(four, act, 16) == ([0:1, 2:3, 4:5, 6:7], 8)
    # Unequal weights: 25 and 49 fine nodes over six ranks split 2 : 4.
    @test CL._tile_owners([t(0), BlockRegion((88, 0, 0), (17, 1, 1))], act, 6) ==
          ([0:1, 2:5], 6)
    # More tiles than ranks: one rank per tile, the curve cut into runs.
    @test CL._tile_owners(four, act, 2) == ([0:0, 0:0, 1:1, 1:1], 2)
    # The curve is a Morton order, not the lattice raster: a 3 × 2 block of
    # tiles over three ranks pairs the (0,0)/(12,0) and (0,12)/(12,12)
    # tiles, then the x = 24 column, where the raster would pair (24,0) with
    # (0,12).
    act2 = (true, true, false)
    six = [BlockRegion((x, y, 0), (13, 13, 1)) for x in (0, 12, 24), y in (0, 12)]
    @test CL._sfc_order(vec(six)) == [1, 2, 4, 5, 3, 6]
    @test CL._tile_owners(vec(six), act2, 3) ==
          ([0:0, 0:0, 2:2, 1:1, 1:1, 2:2], 3)
    # A serial refined solver holds every level unsplit, so no communicator
    # is created and none is freed.
    per3l = ntuple(_ -> (PeriodicBC(), PeriodicBC()), 3)
    solver = Solver(n_global=(192, 1, 1), L_domain=(2π, 1.0, 1.0), bcs=per3l,
                    refine=r(8))
    lc = solver.levels[2].level_comm
    @test lc.owned
    @test !lc.scoped
    @test lc.size == 1
    @test solver.levels[1].level_comm.size == 1
    @test npatches(solver) == 2
    # Likewise for the tile group: one rank spans every tile, so the group
    # is the level's communicator itself, and the tiles are held in order.
    tiled = Solver(n_global=(192, 1, 1), L_domain=(2π, 1.0, 1.0), bcs=per3l,
                   refine=BlockRegion((80, 0, 0), (32, 1, 1)), tile=8)
    lev = tiled.levels[2]
    @test lev.owners == [0:0, 0:0, 0:0, 0:0]
    @test !lev.group.scoped && lev.group.ranks == 0:0
    @test lev.tiles == [1, 2, 3, 4] && lev.patches == [2, 3, 4, 5]
    @test [lt.fine_index for lt in lev.transfers] == [2, 3, 4, 5]
end

@testset "stored ownership across regrids and measured rebalance weights" begin
    # `_place_tiles` keeps a survivor's range and places fresh tiles among
    # the ranks left free; `_tile_owners` would shift every range behind a
    # tile entering ahead on the curve. A synthetic tile-set sequence, no
    # run: 9-node 1-D tiles (25 fine nodes, two ranks each at most) over
    # eight ranks.
    act = (true, false, false)
    t(k) = BlockRegion((80 + 8k, 0, 0), (9, 1, 1))
    four = [t(0), t(1), t(2), t(3)]
    o0, n0 = CL._tile_owners(four, act, 8)
    @test (o0, n0) == ([0:1, 2:3, 4:5, 6:7], 8)
    # The shock moves right: t0 leaves, t4 enters. The survivors keep their
    # ranges and t4 takes the ranks t0 freed, where the fresh partition
    # would have moved all three survivors down.
    oA, nA = CL._place_tiles([t(1), t(2), t(3), t(4)], act, 8, four, o0)
    @test (oA, nA) == ([2:3, 4:5, 6:7, 0:1], 8)
    @test CL._tile_owners([t(1), t(2), t(3), t(4)], act, 8)[1] == o0
    # No rank free: a fresh tile joins the group of the survivor nearest it
    # on the curve, here t4's.
    oB, nB = CL._place_tiles([t(1), t(2), t(3), t(4), t(5)], act, 8,
                             [t(1), t(2), t(3), t(4)], oA)
    @test (oB, nB) == ([2:3, 4:5, 6:7, 0:1, 0:1], 8)
    # Two departures free a run of four ranks; the one fresh tile takes the
    # two it admits, and the other two hold nothing on the level.
    oC, nC = CL._place_tiles([t(3), t(4), t(5), t(6)], act, 8,
                             [t(1), t(2), t(3), t(4), t(5)], oB)
    @test (oC, nC) == ([6:7, 0:1, 0:1, 2:3], 8)
    # Free ranks with a gap: the ranks dealt to a fresh tile are cut at the
    # gap, so a group stays contiguous (ranks 0, 2, 3 free; the tile takes
    # 0 alone). The level's rank count follows the highest rank in use.
    oD, nD = CL._place_tiles([t(1), t(2), t(3)], act, 6, [t(1), t(2)], [1:1, 4:5])
    @test (oD, nD) == ([1:1, 4:5, 0:0], 6)
    # A departure from the top of the level shrinks it.
    @test CL._place_tiles([t(1)], act, 6, [t(1), t(2)], [1:1, 4:5]) == ([1:1], 2)
    # With no survivor the level is partitioned afresh.
    @test CL._place_tiles([t(5), t(6)], act, 8, [t(1)], [3:4]) ==
          CL._tile_owners([t(5), t(6)], act, 8)
    # Measured weights: a tile costs the busy time of its owner ranks, a
    # group's shared by volume, and a fresh tile the mean cost per node.
    w = CL._measured_weights([t(1), t(2), t(3)], act, [t(1), t(2)],
                             [0:1, 2:2], [1.0, 1.0, 3.0])
    @test w == [2.0, 3.0, 2.5]
    wg = CL._measured_weights([t(1), t(2)], act, [t(1), t(2)], [0:0, 0:0], [4.0])
    @test wg == [2.0, 2.0]
    @test CL._measured_weights([t(1), t(3)], act, [t(1)], [0:0], [0.0]) ==
          [25.0, 25.0]
    # The weighted partition: shares 1.6 : 2.4 : 2.0 of six ranks, at the
    # two-rank cap each.
    @test CL._tile_owners([t(1), t(2), t(3)], act, 6; weights=w) ==
          ([0:1, 2:3, 4:5], 6)
    # Unequal weights past the cap: the heavy tile stops at two ranks.
    @test CL._tile_owners([t(1), t(2)], act, 4; weights=[1.0, 9.0]) == ([0:1, 2:3], 4)
    @test CL._rank_counts([1.0, 9.0], [2, 2], 4) == [2, 2]
    @test CL._rank_counts([1.0, 9.0], [4, 4], 4) == [1, 3]
    # Configuration guards.
    per3l = ntuple(_ -> (PeriodicBC(), PeriodicBC()), 3)
    mk(; kw...) = Solver(n_global=(192, 1, 1), L_domain=(2π, 1.0, 1.0), bcs=per3l,
                         refine=BlockRegion((80, 0, 0), (32, 1, 1)); kw...)
    @test_throws ErrorException mk(tile=8, regrid_interval=5, rebalance=0.5)
    @test_throws ErrorException mk(tile=8, rebalance=1.5)
    @test_throws ErrorException mk(regrid_interval=5, rebalance=1.5)
    @test_throws ErrorException mk(tile=8, regrid_interval=5, rebalance=1.5,
                                   rebalance_persist=0)
    spec = getfield(mk(tile=8, regrid_interval=5, rebalance=1.5), :regrid)
    @test spec.rebalance == 1.5 && spec.persist == 2 && spec.streak == 0
    # A serial rebalance never fires: one rank's max/mean is exactly one.
    sa = mk(tile=8, regrid_interval=5, rebalance=1.0, rebalance_persist=1)
    Q = allocate_state(sa)
    initialize!(sa, Q, (x, y, z) -> Prim(u=(0, 0, 0), p=1.0, rho=1.0 + 0.1sin(x)))
    run!(sa, Q; tfinal=1.0, nmax=12)
    spec = getfield(sa, :regrid)
    @test spec.imbalance == 1.0 && spec.streak == 0
    @test sa.wait_total >= 0 && sa.wait_total <= sa.wall_total
end

@testset "tile migration: interior rule and the rank's own copy" begin
    # `_migrate_tile!` moves a tile's solution between two decompositions
    # of it from their block tables alone. On one rank both blocks are the
    # whole tile, so the one message is the rank's copy between its own
    # blocks: the interior, one node off every boundary plane, arrives and
    # the shell keeps the destination's values, as `_carry_over!` of the
    # replicated gather leaves it.
    per3l = ntuple(_ -> (PeriodicBC(), PeriodicBC()), 3)
    solver = Solver(n_global=(192, 1, 1), L_domain=(2π, 1.0, 1.0), bcs=per3l,
                    refine=BlockRegion((80, 0, 0), (32, 1, 1)), tile=8,
                    regrid_interval=5)
    states = allocate_state(solver)
    initialize!(solver, states,
                (x, y, z) -> Prim(u=(0, 0, 0), p=1.0, rho=1.0 + 0.1sin(x)))
    lev = getfield(solver, :levels)[2]
    li = lev.patches[1]
    lt = lev.transfers[lev.tiles[1]]
    dp = getfield(solver, :patches)[li].decomp
    n_cons = solver.equations.n_cons
    Q_old = states[li]
    Q_new = ConservedState(fill(-1.0, size(parent(Q_old))))
    Nf = CL.fine_extent(lt.region, lt.active)
    CL._migrate_tile!(Float64, Q_new, dp, Q_old, dp, Nf, lt.active,
                      lt.fine_blocks, lt.fine_blocks, dp.comm)
    pad = dp.n_halo_d
    inner = ntuple(d -> lt.active[d] ? ((2 + pad[d]):(Nf[d] - 1 + pad[d])) : (1:1), 3)
    @test view(parent(Q_new), inner..., :) == view(parent(Q_old), inner..., :)
    untouched = trues(size(parent(Q_new)))
    untouched[inner..., :] .= false
    @test all(parent(Q_new)[untouched] .== -1.0)
    Qref = ConservedState(fill(-1.0, size(parent(Q_old))))
    gathered = CL._gather_tile(Float64, lt.region, lt.active, n_cons, lt, solver,
                               states)
    CL._carry_over!(Qref, dp, lt.region, gathered, Nf, lt.region, lt.active, n_cons)
    @test parent(Qref) == parent(Q_new)
end

@testset "subcycled two levels: manufactured solution across the boundary" begin
    errs = [_level_wave_error(N; subcycle=true) for N in (48, 96, 192)]
    orders = [log2(errs[i] / errs[i+1]) for i in 1:2]
    @info "subcycled two-level entropy wave" errs orders
    # Measured 8.4e-8 / 7.5e-9 / 5.9e-10, orders 3.49 / 3.66 — within a few
    # percent of the global-dt coupling's figures, so the cubic Hermite
    # boundary data does not bind. The step count drops threefold: dt is now
    # coarse-limited (the fine rate enters the reduction divided by 3).
    @test all(>(3.0), orders)
    @test errs[2] < 3e-8
end

@testset "subcycled Sod through the refinement boundary" begin
    wall2 = (SlipWallBC(), SlipWallBC())
    per = (PeriodicBC(), PeriodicBC())
    ic(x, y, z) = x < 0.5 ? Prim(u=(0, 0, 0), p=1.0, rho=1.0) :
                            Prim(u=(0, 0, 0), p=0.1, rho=0.125)
    N = 201
    # The static gate's configuration with subcycling on: the same region,
    # CFL, and crossing schedule, so the guards compare directly.
    solver = Solver(n_global=(N, 1, 1), L_domain=(1.0, 1.0, 1.0),
                    bcs=(wall2, per, per), cfl=0.4, subcycle=true,
                    refine=BlockRegion((120, 0, 0), (41, 1, 1)))
    states = allocate_state(solver)
    initialize!(solver, states, ic)
    m0 = _two_level_mass(solver, states, N)
    run!(solver, states; tfinal=0.1, nmax=20000)
    ps = PatchSolver(solver, solver.patches[1])
    pad = ps.decomp.n_halo_d[1]
    m1 = solver.equations.i_mom[1]
    noise = maximum(abs(states[1][i + pad, 1, 1, m1]) for i in 172:N)
    @info "subcycled Sod through refinement boundary" noise
    # Measured 1.3e-10 against the global-dt gate's 6.4e-10 (5.7e-11 under the
    # former κ/(ρ cp) diffusive limit; the cv form takes different steps).
    @test noise < 1e-8
    for (psq, Q) in CL.eachpatch(solver, states)
        n = psq.decomp.n_local[1]
        padq = psq.decomp.n_halo_d[1]
        @test minimum(Q[i + padq, 1, 1, 1] for i in 1:n) > 0.05
    end
    run!(solver, states; tfinal=0.2, nmax=40000)
    drift = abs(_two_level_mass(solver, states, N) - m0) / m0
    @info "subcycled two-level Sod mass drift" drift
    # Measured 9.8e-5 against the global-dt gate's 1.36e-4.
    @test drift < 5e-4
end

@testset "tagging and regridding track a Sod shock" begin
    wall2 = (SlipWallBC(), SlipWallBC())
    per = (PeriodicBC(), PeriodicBC())
    ic(x, y, z) = x < 0.5 ? Prim(u=(0, 0, 0), p=1.0, rho=1.0) :
                            Prim(u=(0, 0, 0), p=0.1, rho=0.125)
    N = 201
    Nf = 3N - 2
    tf = 0.15
    # Uniform-fine reference and uniform-coarse baseline. CFL 0.2
    # throughout: the initial discontinuity sits inside the refined region, and a subcycled fine
    # level runs three substeps on one rate measurement, which tightens the
    # documented startup restriction (reference/AMR_GPU.md, two-level
    # refinement).
    sf = Solver(n_global=(Nf, 1, 1), L_domain=(1.0, 1.0, 1.0),
                bcs=(wall2, per, per), cfl=0.2)
    Qf = allocate_state(sf)
    initialize!(sf, Qf, ic)
    run!(sf, Qf; tfinal=tf, nmax=40000)
    padF = sf.decomp.n_halo_d[1]
    rho_ref = [Qf[i + padF, 1, 1, 1] for i in 1:Nf]
    sc = Solver(n_global=(N, 1, 1), L_domain=(1.0, 1.0, 1.0),
                bcs=(wall2, per, per), cfl=0.2)
    Qc = allocate_state(sc)
    initialize!(sc, Qc, ic)
    run!(sc, Qc; tfinal=tf, nmax=40000)
    padC = sc.decomp.n_halo_d[1]
    e_base = maximum(abs(Qc[i + padC, 1, 1, 1] - rho_ref[3i - 2]) for i in 1:N)
    # AMR: subcycled with the region retagged every 5 coarse steps.
    sa = Solver(n_global=(N, 1, 1), L_domain=(1.0, 1.0, 1.0),
                bcs=(wall2, per, per), cfl=0.2, subcycle=true,
                regrid_interval=5, refine=BlockRegion((85, 0, 0), (31, 1, 1)))
    states = allocate_state(sa)
    initialize!(sa, states, ic)
    lt0 = getfield(sa, :levels)[2].transfers[1]
    decomp0 = sa.patches[2].decomp
    run!(sa, states; tfinal=tf, nmax=40000)
    region = CL.refined_region(sa)
    lo = region.offset[1] + 1
    hi = region.offset[1] + region.extent[1]
    # The region moved off its initial site and holds the shock (x ≈ 0.76).
    shock_node = round(Int, (0.5 + 1.75 * sa.t) * (N - 1)) + 1
    @info "regrid tracking" region=(lo, hi) shock_node
    @test region.offset[1] != 85
    @test lo < shock_node < hi
    # The regrids dropped the initial transfer and fine patch. Their
    # communicators must be freed at the drop, not left to the garbage
    # collector: at the regrid cadence the finalizer backlog exhausts
    # MPICH's 2048-context-id budget whenever collection lags.
    @test lt0.pdecomps[1].comm == CL.MPI.COMM_NULL
    @test decomp0.comm == CL.MPI.COMM_NULL
    padc = sa.patches[1].decomp.n_halo_d[1]
    padf = sa.patches[2].decomp.n_halo_d[1]
    e_amr = 0.0
    for i in 1:N
        v = lo <= i <= hi ? states[2][3 * (i - lo) + 1 + padf, 1, 1, 1] :
                            states[1][i + padc, 1, 1, 1]
        e_amr = max(e_amr, abs(v - rho_ref[3i - 2]))
    end
    @test nlevels(sa) == 2
    @info "moving-region Sod vs uniform fine" e_amr e_base
    # Measured: composite 2.7e-3 against the uniform-coarse baseline's
    # 7.3e-2 (3.9e-3 under the former κ/(ρ cp) diffusive limit, whose steps
    # differ); the refinement recovers most of the uniform-fine answer at a
    # third of the fine points.
    @test e_amr < 1.5e-2
    @test e_base > 5e-2
    @test e_amr < e_base / 3
    for (psq, Q) in CL.eachpatch(sa, states)
        n = psq.decomp.n_local
        padq = psq.decomp.n_halo_d[1]
        @test minimum(Q[i + padq, 1, 1, 1] for i in 1:n[1]) > 0.05
    end
    # The global-dt coupling regrids through the same machinery.
    sg = Solver(n_global=(N, 1, 1), L_domain=(1.0, 1.0, 1.0),
                bcs=(wall2, per, per), cfl=0.2, regrid_interval=5,
                refine=BlockRegion((85, 0, 0), (31, 1, 1)))
    states_g = allocate_state(sg)
    initialize!(sg, states_g, ic)
    run!(sg, states_g; tfinal=0.05, nmax=20000)
    @test CL.refined_region(sg).offset[1] != 85
end

# --- Three levels ------------------------------------------------------------

@testset "three levels: configuration and geometry" begin
    per3l = ntuple(_ -> (PeriodicBC(), PeriodicBC()), 3)
    r1 = BlockRegion((40, 0, 0), (16, 1, 1))   # level-1 nodes 121..166
    mk(r2; kw...) = Solver(; n_global=(96, 1, 1), L_domain=(2π, 1.0, 1.0),
                           bcs=per3l, refine=[r1, r2], kw...)
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
        errs = [_level_wave_error(N; levels=3, subcycle=subcycle)
                for N in (48, 96, 192)]
        orders = [log2(errs[i] / errs[i+1]) for i in 1:2]
        @info "three-level entropy wave" subcycle errs orders
        # Measured 9.0e-8 / 9.1e-9 / 6.8e-10, orders 3.31 / 3.74 at the
        # global dt and 9.0e-8 / 8.8e-9 / 6.3e-10, orders 3.35 / 3.79
        # subcycled: the two-level figures (3.46 / 3.64) with a second
        # coarse-fine boundary pair inside the first.
        @test all(>(3.0), orders)
        @test errs[2] < 3e-8
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
    solver = Solver(n_global=(N, 1, 1), L_domain=(1.0, 1.0, 1.0),
                    bcs=(wall2, per, per), cfl=0.4, subcycle=true,
                    refine=[BlockRegion((120, 0, 0), (41, 1, 1)),
                            BlockRegion((390, 0, 0), (60, 1, 1))])
    states = allocate_state(solver)
    initialize!(solver, states, ic)
    run!(solver, states; tfinal=0.1, nmax=20000)
    ps = PatchSolver(solver, solver.patches[1])
    pad = ps.decomp.n_halo_d[1]
    m1 = solver.equations.i_mom[1]
    # Momentum ahead of the shock on the two-level gate's schedule (t = 0.1,
    # x > 0.85) is refinement-boundary noise; measured 6.4e-10, the
    # two-level figure, so the inner boundary pair adds nothing visible.
    noise = maximum(abs(states[1][i + pad, 1, 1, m1]) for i in 172:N)
    @info "three-level Sod noise ahead of the shock" noise
    @test noise < 1e-8
    run!(solver, states; tfinal=0.2, nmax=40000)
    @test all(all(isfinite, parent(Q)) for Q in states)
    for (psq, Q) in CL.eachpatch(solver, states)
        n = psq.decomp.n_local[1]
        @test minimum(Q[gidx(psq, i, 1, 1), 1] for i in 1:n) > 0.05
    end
end


# --- Tiled levels ------------------------------------------------------------

@testset "tiled level: lattice cover, faces, and configuration guards" begin
    per3l = ntuple(_ -> (PeriodicBC(), PeriodicBC()), 3)
    mk(; kw...) = Solver(; n_global=(192, 1, 1), L_domain=(2π, 1.0, 1.0),
                         bcs=per3l, refine=BlockRegion((80, 0, 0), (32, 1, 1)),
                         kw...)
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
    @test solver.patches[2].bcs[1][1] isa CoarseFineBC
    # A box node on a lattice plane pulls in the cell on its inner side only.
    @test CL._tile_span(81, 112, 8) == 10:13
    @test CL._tile_span(82, 113, 8) == 10:13
    @test CL._tile_span(81, 81, 8) == 9:10
    # A tile that would leave the nesting margin is clipped, and dropped when
    # fewer than four nodes remain: region 5..24 at margin 4 keeps cell 0
    # clipped to nodes 5..9, then cells 1 and 2.
    s2 = Solver(n_global=(192, 1, 1), L_domain=(2π, 1.0, 1.0), bcs=per3l,
                refine=BlockRegion((4, 0, 0), (20, 1, 1)), tile=8)
    @test [(r.offset[1], r.extent[1]) for r in level_regions(s2, 1)] ==
          [(4, 5), (8, 9), (16, 9)]
end

@testset "tiled level: manufactured solution across tile and level faces" begin
    errs = [_level_wave_error(N; tile=8) for N in (48, 96, 192)]
    orders = [log2(errs[i] / errs[i+1]) for i in 1:2]
    @info "tiled entropy wave" errs orders
    # Measured 1.39e-7 / 9.0e-9 / 6.0e-10, orders 3.95 / 3.91, beside the
    # one-patch level's 8.5e-8 / 7.7e-9 / 6.2e-10: the tile interfaces
    # inside the level cost nothing visible at N = 192.
    @test all(>(3.0), orders)
    @test errs[3] < 2e-9
    # Subcycled, one tiled level: measured 5.8e-10 against 5.9e-10.
    @test _level_wave_error(192; tile=8, subcycle=true) < 2e-9
end

@testset "tiled level: 2-D tile nest with corners" begin
    per3l = ntuple(_ -> (PeriodicBC(), PeriodicBC()), 3)
    u0, v0 = 0.4, 0.3
    function vortex(; tile, subcycle)
        solver = Solver(n_global=(48, 48, 1), L_domain=(2π, 2π, 1.0), bcs=per3l,
                        art=ArtParams(enabled=false), filter_interval=0,
                        subcycle=subcycle, tile=tile,
                        refine=BlockRegion((18, 18, 0), (12, 12, 1)))
        states = allocate_state(solver)
        initialize!(solver, states, (x, y, z) ->
            Prim(u=(u0, v0, 0), p=1.0, rho=1.0 + 0.1 * sin(x) * sin(y)))
        run!(solver, states; tfinal=0.5)
        e = 0.0
        for (ps, Q) in CL.eachpatch(solver, states)
            n = ps.decomp.n_local
            for j in 1:n[2], i in 1:n[1]
                I = gidx(ps, i, j, 1)
                exact = 1.0 + 0.1 * sin(xcoord(ps, 1, i) - u0 * solver.t) *
                              sin(xcoord(ps, 2, j) - v0 * solver.t)
                e = max(e, abs(Q[I, 1] - exact))
            end
        end
        return e, npatches(solver)
    end
    # Four 7×7 tiles meeting at a corner, both stepping modes: measured
    # 4.29e-8 against the one-patch level's 4.27e-8, so the corner ghosts
    # and the two-way shared faces are consistent.
    for subcycle in (false, true)
        e, np = vortex(tile=6, subcycle=subcycle)
        @info "2-D tile nest" subcycle e
        @test np == 5
        @test e < 1e-7
    end
end

@testset "tiled regridding tracks a Sod shock" begin
    wall2 = (SlipWallBC(), SlipWallBC())
    per = (PeriodicBC(), PeriodicBC())
    ic(x, y, z) = x < 0.5 ? Prim(u=(0, 0, 0), p=1.0, rho=1.0) :
                            Prim(u=(0, 0, 0), p=0.1, rho=0.125)
    N = 201
    sa = Solver(n_global=(N, 1, 1), L_domain=(1.0, 1.0, 1.0),
                bcs=(wall2, per, per), cfl=0.2, subcycle=true,
                regrid_interval=5, refine=BlockRegion((85, 0, 0), (31, 1, 1)),
                tile=8)
    states = allocate_state(sa)
    initialize!(sa, states, ic)
    initial = level_regions(sa, 1)
    transfers0 = getfield(sa, :levels)[2].transfers
    run!(sa, states; tfinal=0.15, nmax=40000)
    regs = level_regions(sa, 1)
    shock_node = round(Int, (0.5 + 1.75 * sa.t) * (N - 1)) + 1
    @info "tiled regrid tracking" tiles=[(r.offset[1], r.extent[1]) for r in regs] shock_node
    @test regs != initial
    # The tile set changed, so every setup-time transfer was replaced and
    # its chain communicators were freed at the drop.
    @test all(lt.pdecomps[1].comm == CL.MPI.COMM_NULL for lt in transfers0)
    # Every tile is a lattice cell, the set is contiguous, and it holds the
    # shock.
    @test all(r -> r.offset[1] % 8 == 0 && r.extent[1] == 9, regs)
    @test all(i -> regs[i+1].offset[1] == regs[i].offset[1] + 8, 1:length(regs)-1)
    @test regs[1].offset[1] < shock_node <= regs[end].offset[1] + 9
    @test all(all(isfinite, parent(Q)) for Q in states)
    for (psq, Q) in CL.eachpatch(sa, states)
        n = psq.decomp.n_local[1]
        @test minimum(Q[gidx(psq, i, 1, 1), 1] for i in 1:n) > 0.05
    end
    @test length(sa.patches) == length(states) == length(regs) + 1
end


@testset "tiled level: multi-tile corner consensus and corner ghosts" begin
    # Four tiles meet at one fine node. Pairwise averaging in one flat pass
    # leaves the four copies unequal (a later pair reads a value an earlier
    # pair changed); the dimension-phased sync gives every copy the mean,
    # and the phase-2 strips, padded along x, fill the diagonal ghosts.
    per3l = ntuple(_ -> (PeriodicBC(), PeriodicBC()), 3)
    solver = Solver(n_global=(48, 48, 1), L_domain=(2π, 2π, 1.0), bcs=per3l,
                    refine=BlockRegion((18, 18, 0), (12, 12, 1)), tile=6)
    states = allocate_state(solver)
    for (i, Q) in enumerate(states)
        fill!(parent(Q), i - 1.0)              # tiles 2..5 carry 1..4
    end
    lev = solver.levels[2]
    @test lev.phases == (true, true, false)
    CL._sync_level_records!(solver, states, lev)
    # Tile order is lattice order: A (18,18), B (24,18), C (18,24), D (24,24),
    # each 19 fine nodes; the common corner is A(19,19), B(1,19), C(19,1),
    # D(1,1).
    at(p, i, j) = (ps = PatchSolver(solver, solver.patches[p]);
                   pad = ps.decomp.n_halo_d; states[p][i + pad[1], j + pad[2], 1 + pad[3], 1])
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
    snapshot = [copy(parent(Q)) for Q in states]
    CL._sync_level_records!(solver, states, lev)
    @test all(parent(states[i]) == snapshot[i] for i in eachindex(states))
end

@testset "tiled level: one RHS workspace per padded extent" begin
    # A rank advances its patches in sequence and nothing in the RHS scratch
    # outlives the evaluation that filled it, so that scratch is pooled on the
    # padded local extent (patches.jl): a lattice level's tiles, whose extents
    # are equal by construction, hold one set between them. Nothing else in the
    # gate moves if the pooling silently stops.
    per3l = ntuple(_ -> (PeriodicBC(), PeriodicBC()), 3)
    solver = Solver(n_global=(48, 48, 1), L_domain=(2π, 2π, 1.0), bcs=per3l,
                    refine=BlockRegion((18, 18, 0), (12, 12, 1)), tile=6)
    padded(p) = ntuple(d -> p.decomp.n_local[d] + 2 * p.decomp.n_halo_d[d], 3)
    extents = unique(padded(p) for p in solver.patches)
    sets = unique(objectid(p.rhs_workspace) for p in solver.patches)
    @test length(solver.patches) == 5          # the root slab plus four tiles
    @test length(extents) == 2                 # the root's extent and the tiles'
    @test length(sets) == length(extents)
    @test all(p.rhs_workspace === solver.patches[2].rhs_workspace
              for p in solver.patches[2:end])
    @test solver.patches[1].rhs_workspace !== solver.patches[2].rhs_workspace
    # Whatever a patch is handed carries that patch's own extent.
    @test all(size(p.rhs_workspace.tmp_a) == padded(p) for p in solver.patches)
    @test all(size(p.rhs_workspace.flux[1, 1]) == padded(p) for p in solver.patches)
    # The single-patch property forwarding serves the workspace's own arrays,
    # and the launchable wrappers hold those same objects.
    single = Solver(n_global=(16, 16, 1), L_domain=(1.0, 1.0, 1.0), bcs=per3l)
    ws = single.patches[1].rhs_workspace
    @test single.tmp_a === ws.tmp_a && single.tmp_b === ws.tmp_b
    @test single.sensor === ws.sensor && single.strain_mag === ws.strain_mag
    @test single.grad_u === ws.grad_u && single.flux === ws.flux
    @test single.field_tuples.grad_u[1, 1] === ws.grad_u[1, 1]
    @test single.field_tuples.flux[1, 1] === ws.flux[1, 1]
    @test single.field_tuples.Y[1] === single.patches[1].Y[1]
end

@testset "tiled regrid seeds fresh tiles from surviving neighbors" begin
    wall2 = (SlipWallBC(), SlipWallBC())
    per = (PeriodicBC(), PeriodicBC())
    ic(x, y, z) = x < 0.5 ? Prim(u=(0, 0, 0), p=1.0, rho=1.0) :
                            Prim(u=(0, 0, 0), p=0.1, rho=0.125)
    N = 201
    sa = Solver(n_global=(N, 1, 1), L_domain=(1.0, 1.0, 1.0),
                bcs=(wall2, per, per), cfl=0.2, subcycle=true,
                regrid_interval=5, refine=BlockRegion((85, 0, 0), (31, 1, 1)),
                tile=8)
    states = allocate_state(sa)
    initialize!(sa, states, ic)
    run!(sa, states; tfinal=1.0, nmax=20)
    before = level_regions(sa, 1)
    # Perturb a survivor's interior so the seeding is visible: the plane a
    # fresh tile shares with it must carry the survivor's value exactly.
    workspace = CL.Workspace(states)
    changed = CL.regrid!(sa, states, workspace, nothing)
    after = level_regions(sa, 1)
    @test changed && after != before
    for (i, r) in enumerate(after), (j, s) in enumerate(after)
        (r in before) && !(s in before) || continue
        r.offset[1] + r.extent[1] - 1 == s.offset[1] || continue
        ps = PatchSolver(sa, sa.patches[i + 1])
        qs = PatchSolver(sa, sa.patches[j + 1])
        nr = ps.decomp.n_local[1]
        @test states[i + 1][gidx(ps, nr, 1, 1), 1] == states[j + 1][gidx(qs, 1, 1, 1), 1]
    end
end

@testset "tag criteria: gradient, vorticity, sensor and predicate" begin
    wall2 = (SlipWallBC(), SlipWallBC())
    per = (PeriodicBC(), PeriodicBC())
    per3l = ntuple(_ -> per, 3)
    N = 201
    two = IdealMixture([IdealSpecies{Float64}("a", 1.0, 1.4),
                        IdealSpecies{Float64}("b", 1.0, 1.4)])
    r0 = BlockRegion((20, 0, 0), (31, 1, 1))
    mk(; kw...) = Solver(n_global=(N, 1, 1), L_domain=(1.0, 1.0, 1.0),
                         bcs=(wall2, per, per), eos=two, cfl=0.2,
                         refine=r0; kw...)
    # Configuration guards: a criterion is a regrid setting, and a
    # threshold is non-negative.
    @test_throws ErrorException mk(regrid_interval=5, tag_gradient_threshold=-1)
    @test_throws ErrorException mk(tag_gradient_threshold=0.05)
    @test_throws ErrorException mk(tag_sensor_threshold=0.1)
    @test_throws ErrorException mk(tag_vorticity_threshold=1.0)
    @test_throws ErrorException mk(tag_predicate=(p, I) -> true)
    @test_throws ErrorException mk(regrid_interval=5, tag_predicate=1)
    # Numerics carries the four settings to the solver.
    pred0 = (p, I) -> false
    num = Numerics(n_global=(N, 1, 1), refine=r0, regrid_interval=5,
                   tag_gradient_threshold=0.05, tag_sensor_threshold=0.1,
                   tag_vorticity_threshold=2.0, tag_predicate=pred0)
    prob = Problem(domain=((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)),
                   bcs=(wall2, per, per), eos=two,
                   ic=(x, y, z) -> Prim(Y=(1.0, 0.0), rho=1.0, p=1.0, u=(0, 0, 0)))
    sn, _ = setup(prob, num)
    specn = getfield(sn, :regrid)
    @test specn.gradient_threshold == 0.05 && specn.sensor_threshold == 0.1
    @test specn.vorticity_threshold == 2.0 && specn.predicate === pred0
    @test size(specn.tags) == size(sn.patches[1].rho)

    # Gradient: a mass-fraction interface at x = 0.75 in a uniform-density,
    # uniform-pressure field, which the δ⁴ρ criterion cannot see. The
    # per-cell change peaks at h/(2w) = 0.125, so 0.05 tags the interface
    # core and 0.5 tags nothing.
    node(x) = round(Int, x * (N - 1)) + 1
    icY(x, y, z) = (θ = 0.5 * (1 + tanh((x - 0.75) / 0.02));
                    Prim(Y=(1 - θ, θ), rho=1.0, p=1.0, u=(0, 0, 0)))
    for (thr, hit) in ((0.0, false), (0.05, true), (0.5, false))
        sg = mk(regrid_interval=5, tag_gradient_threshold=thr)
        states = allocate_state(sg)
        initialize!(sg, states, icY)
        r = CL.tagged_region(sg, states[1])
        if hit
            @test r !== nothing
            @test r.offset[1] < node(0.75) <= r.offset[1] + r.extent[1]
            @test r.extent[1] < 40
        else
            @test r === nothing
        end
    end

    # Vorticity: a Gaussian vortex (|ω| = 2A at its center, A = 1) in a
    # uniform-density 2-D field; a threshold of 1 tags its core, 10 nothing.
    xc = 4.5
    icω(x, y, z) = (g = exp(-((x - xc)^2 + (y - xc)^2) / 0.25);
                    Prim(u=(-(y - xc) * g, (x - xc) * g, 0.0), rho=1.0, p=1.0))
    N2 = 48
    gc = round(Int, xc * N2 / (2π)) + 1
    for (thr, hit) in ((0.0, false), (1.0, true), (10.0, false))
        sv = Solver(n_global=(N2, N2, 1), L_domain=(2π, 2π, 1.0), bcs=per3l,
                    refine=BlockRegion((8, 8, 0), (12, 12, 1)),
                    regrid_interval=5, tag_vorticity_threshold=thr)
        states = allocate_state(sv)
        initialize!(sv, states, icω)
        r = CL.tagged_region(sv, states[1])
        if hit
            @test r !== nothing
            for d in 1:2
                @test r.offset[d] < gc - 1 && gc + 1 <= r.offset[d] + r.extent[d]
                @test r.extent[d] < N2 ÷ 2
            end
        else
            @test r === nothing
        end
    end

    # Sensor: the artificial diffusivity number of the last RHS. The Sod
    # shock runs ten steps with the δ⁴ criterion parked at a threshold
    # nothing reaches; the number is then measured over the root, and half
    # of its maximum tags the shock where twice its maximum tags nothing.
    icS(x, y, z) = x < 0.5 ? Prim(Y=(1.0, 0.0), u=(0, 0, 0), p=1.0, rho=1.0) :
                             Prim(Y=(1.0, 0.0), u=(0, 0, 0), p=0.1, rho=0.125)
    ss = mk(regrid_interval=1000, tag_threshold=1e6,
            refine=BlockRegion((85, 0, 0), (31, 1, 1)))
    states = allocate_state(ss)
    initialize!(ss, states, icS)
    run!(ss, states; tfinal=1.0, nmax=10)
    root = PatchSolver(ss, ss.patches[1])
    qmax = 0.0
    for i in 1:root.decomp.n_local[1]
        I = gidx(root, i, 1, 1)
        ν = (root.mu_art[I] + root.beta_art[I]) / root.rho[I] +
            root.kappa_art[I] / (root.rho[I] * root.cp_mix[I]) +
            maximum(D[I] for D in root.D_art)
        qmax = max(qmax, ν / (root.c[I] * root.h[1]))
    end
    @info "artificial diffusivity number at the Sod shock" qmax
    @test qmax > 0
    spec = getfield(ss, :regrid)
    @test CL.tagged_region(ss, states[1]) === nothing
    spec.sensor_threshold = 2 * qmax
    @test CL.tagged_region(ss, states[1]) === nothing
    spec.sensor_threshold = qmax / 2
    r = CL.tagged_region(ss, states[1])
    @test r !== nothing
    shock = node(0.5 + 1.75 * ss.t)
    @test r.offset[1] < shock <= r.offset[1] + r.extent[1]
    @test r.extent[1] < 60

    # Predicate: a closure over the parent patch and a padded index, here
    # x > 0.8, whose tagged set is that interval buffered and clamped to
    # the margin; in union with the δ⁴ tag at the initial discontinuity
    # the box spans both.
    predx = (p, I) -> xcoord(p, 1, interior_index(p, I)[1]) > 0.8
    sp = mk(regrid_interval=5, tag_predicate=predx)
    states = allocate_state(sp)
    initialize!(sp, states, (x, y, z) -> Prim(Y=(1.0, 0.0), rho=1.0, p=1.0,
                                             u=(0, 0, 0)))
    r = CL.tagged_region(sp, states[1])
    margin = getfield(sp, :regrid).margin
    # The first node past x = 0.8, in the solver's own arithmetic.
    g1 = findfirst(g -> (g - 1) * (1.0 / (N - 1)) > 0.8, 1:N)
    @test r == BlockRegion((g1 - 1 - 4, 0, 0), (N - margin - (g1 - 1 - 4), 1, 1))
    initialize!(sp, states, icS)
    r = CL.tagged_region(sp, states[1])
    @test r.offset[1] < node(0.5) && r.offset[1] + r.extent[1] == N - margin
end
