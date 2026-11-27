# The level tests of regridding: rank ownership and rebalance, the tile lattice,
# tile migration, the tag criteria and their hysteresis, and regridded hierarchies
# of several levels. Included by test/level_tests.jl, which defines the helpers used
# here.

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
    # The scan starts at the count the regions admit rather than at the rank
    # count, and agrees with a scan from the rank count down, in 1-D, 2-D and
    # 3-D, with a region clipped narrower than nine fine nodes along one
    # dimension among them.
    function scanned(regions, a, n)
        for p in n:-1:1
            all(regions) do reg
                ext = CL.fine_extent(reg, a)
                CL._amr_dims_or_nothing(ext, ntuple(d -> ext[d] > 1, 3), p) !== nothing
            end && return p
        end
        return 1
    end
    act3 = (true, true, true)
    cases = [([r(24)], act), ([r(64), r(8)], act),
             ([BlockRegion((8, 8, 0), (12, 20, 1))], (true, true, false)),
             ([BlockRegion((8, 8, 8), (12, 9, 16))], act3),
             ([BlockRegion((8, 8, 8), (12, 3, 16))], act3)]
    @test all(CL._level_ranks(regs, a, n) == scanned(regs, a, n)
              for (regs, a) in cases, n in 1:48)
    # At a cluster's rank count the sizing costs what it does at a
    # workstation's: a 3-D tile of 25³ fine nodes admits eight ranks.
    cube = BlockRegion((8, 8, 8), (9, 9, 9))
    @test CL._level_ranks([cube], act3, 1_000_000) == 8
    # Per-tile owners: four 9-node tiles (25 fine, two ranks each at most).
    t(k) = BlockRegion((80 + 8k, 0, 0), (9, 1, 1))
    four = [t(0), t(1), t(2), t(3)]
    # The one patch of a box level (`whole = false`) reduces to `_level_ranks`.
    @test CL._tile_owners([t(0)], act, 8; whole=false) == ([0:1], 2)
    # Under `whole = false`, ranks at least as many as tiles: each tile its
    # own range, by weight, capped by what it admits; the level takes the
    # union.
    @test CL._tile_owners(four, act, 4; whole=false) == ([0:0, 1:1, 2:2, 3:3], 4)
    @test CL._tile_owners(four, act, 8; whole=false) == ([0:1, 2:3, 4:5, 6:7], 8)
    @test CL._tile_owners(four, act, 16; whole=false) == ([0:1, 2:3, 4:5, 6:7], 8)
    # Unequal weights: 25 and 49 fine nodes over six ranks split 2 : 4.
    @test CL._tile_owners([t(0), BlockRegion((88, 0, 0), (17, 1, 1))], act, 6;
                          whole=false) == ([0:1, 2:5], 6)
    # A tiled level holds each tile whole on one rank, the k-th tile on the
    # curve on rank k − 1, and the ranks past the last tile hold nothing on
    # the level.
    @test CL._tile_owners([t(0)], act, 8) == ([0:0], 1)
    @test CL._tile_owners(four, act, 8) == ([0:0, 1:1, 2:2, 3:3], 4)
    @test CL._tile_owners(four, act, 16) == ([0:0, 1:1, 2:2, 3:3], 4)
    # More tiles than ranks: one rank per tile, the curve cut into runs,
    # either way.
    @test CL._tile_owners(four, act, 2) == ([0:0, 0:0, 1:1, 1:1], 2)
    @test CL._tile_owners(four, act, 2; whole=false) == ([0:0, 0:0, 1:1, 1:1], 2)
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
    solver = build_solver(
        Problem(domain=((0.0, 2π), (0.0, 1.0), (0.0, 1.0)), bcs=per3l, ic=nothing),
        Numerics(n_global=(192, 1, 1), amr=AMR(initial=r(8), regrid_interval=0)))
    lc = solver.levels[2].level_comm
    @test lc.owned
    @test !lc.scoped
    @test lc.size == 1
    @test solver.levels[1].level_comm.size == 1
    @test npatches(solver) == 2
    # Likewise for the tile group: one rank spans every tile, so the group
    # is the level's communicator itself, and the tiles are held in order.
    tiled = build_solver(
        Problem(domain=((0.0, 2π), (0.0, 1.0), (0.0, 1.0)), bcs=per3l, ic=nothing),
        Numerics(n_global=(192, 1, 1),
                 amr=AMR(initial=BlockRegion((80, 0, 0), (32, 1, 1)), regrid_interval=0,
                         tile=8)))
    lev = tiled.levels[2]
    @test lev.owners == [0:0, 0:0, 0:0, 0:0]
    @test !lev.group.scoped && lev.group.ranks == 0:0
    @test lev.tiles == [1, 2, 3, 4] && lev.patches == [2, 3, 4, 5]
    @test [lt.fine_index for lt in lev.transfers] == [2, 3, 4, 5]
end

@testset "stored ownership across regrids and measured rebalance weights" begin
    # `_place_tiles` keeps a survivor's rank and places fresh tiles on the
    # ranks left free; `_tile_owners` would shift every rank behind a tile
    # entering ahead on the curve. A synthetic tile-set sequence, no run:
    # 9-node 1-D tiles (25 fine nodes) over eight ranks, each tile whole on
    # one rank.
    act = (true, false, false)
    t(k) = BlockRegion((80 + 8k, 0, 0), (9, 1, 1))
    four = [t(0), t(1), t(2), t(3)]
    o0, n0 = CL._tile_owners(four, act, 8)
    @test (o0, n0) == ([0:0, 1:1, 2:2, 3:3], 4)
    # The shock moves right: t0 leaves, t4 enters. The survivors keep their
    # ranks and t4 takes the rank t0 freed, where the fresh partition would
    # have moved all three survivors down.
    oA, nA = CL._place_tiles([t(1), t(2), t(3), t(4)], act, 8, four, o0)
    @test (oA, nA) == ([1:1, 2:2, 3:3, 0:0], 4)
    @test CL._tile_owners([t(1), t(2), t(3), t(4)], act, 8)[1] == o0
    # A fifth tile takes the lowest rank no tile holds, and the level grows.
    oB, nB = CL._place_tiles([t(1), t(2), t(3), t(4), t(5)], act, 8,
                             [t(1), t(2), t(3), t(4)], oA)
    @test (oB, nB) == ([1:1, 2:2, 3:3, 0:0, 4:4], 5)
    # Two departures free ranks 1 and 2; the one fresh tile takes rank 1, and
    # rank 2 holds nothing on the level.
    oC, nC = CL._place_tiles([t(3), t(4), t(5), t(6)], act, 8,
                             [t(1), t(2), t(3), t(4), t(5)], oB)
    @test (oC, nC) == ([3:3, 0:0, 4:4, 1:1], 5)
    # A stored range of several ranks, which a box level's checkpoint or one
    # written by an earlier version records, survives as it is. The level's
    # rank count follows the highest rank in use.
    oD, nD = CL._place_tiles([t(1), t(2), t(3)], act, 6, [t(1), t(2)], [1:1, 4:5])
    @test (oD, nD) == ([1:1, 4:5, 0:0], 6)
    # A departure from the top of the level shrinks it.
    @test CL._place_tiles([t(1)], act, 6, [t(1), t(2)], [1:1, 4:4]) == ([1:1], 2)
    # With no survivor the level is partitioned afresh.
    @test CL._place_tiles([t(5), t(6)], act, 8, [t(1)], [3:3]) ==
          CL._tile_owners([t(5), t(6)], act, 8)
    # Fresh tiles outnumbering the free ranks go one at a time to the rank
    # left least loaded, the nearest on the curve among equals. One rank free
    # for three fresh tiles: the free rank takes the first and, by proximity
    # among equal loads, the second; the third goes to t3's.
    @test CL._place_tiles([t(1), t(2), t(3), t(4), t(5), t(6)], act, 4,
                          [t(1), t(2), t(3)], [0:0, 1:1, 2:2]) ==
          ([0:0, 1:1, 2:2, 3:3, 3:3, 2:2], 4)
    # None free: each fresh tile joins the rank left least loaded.
    @test CL._place_tiles([t(1), t(2), t(3), t(4), t(5), t(6)], act, 3,
                          [t(1), t(2), t(3)], [0:0, 1:1, 2:2]) ==
          ([0:0, 1:1, 2:2, 2:2, 1:1, 0:0], 3)
    # The volume check against a fresh partition. Every tile on a rank of its
    # own is the best partition there is, whichever ranks hold them, so C's
    # placement, with rank 2 idle inside the level, is kept, and so is A's.
    @test CL._balanced_owners([t(3), t(4), t(5), t(6)], act, 8, oC, nC) == (oC, nC)
    @test CL._balanced_owners([t(1), t(2), t(3), t(4)], act, 8, oA, nA) == (oA, nA)
    # Two tiles to a rank with ranks to spare is twice the peak of a fresh
    # partition, which is taken.
    @test CL._balanced_owners([t(3), t(4), t(5), t(6)], act, 8,
                              [0:0, 0:0, 1:1, 1:1], 2) ==
          CL._tile_owners([t(3), t(4), t(5), t(6)], act, 8)
    # Five single-rank tiles on two ranks: three on one rank is the best any
    # partition does and is kept, four is a third worse and is repartitioned,
    # whole tiles largest first onto the least loaded rank, the nearest on
    # the curve among equals.
    five = [t(1), t(2), t(3), t(4), t(5)]
    @test CL._knapsack_owners(five, act, 2) == ([0:0, 1:1, 1:1, 0:0, 0:0], 2)
    @test CL._balanced_owners(five, act, 2, [0:0, 0:0, 0:0, 1:1, 1:1], 2) ==
          ([0:0, 0:0, 0:0, 1:1, 1:1], 2)
    @test CL._balanced_owners(five, act, 2, [0:0, 1:1, 1:1, 1:1, 1:1], 2) ==
          CL._knapsack_owners(five, act, 2)
    # The same repartition with every tile a survivor: the knapsack's rank
    # holding t1, t4 and t5 becomes rank 1, which holds the most of them.
    @test CL._balanced_owners(five, act, 2, [1:1, 1:1, 1:1, 1:1, 0:0], 2, trues(5)) ==
          ([1:1, 0:0, 0:0, 1:1, 1:1], 2)
    # Unequal tiles: a 49-node tile and two 25-node ones split 49 : 50.
    big = BlockRegion((0, 0, 0), (17, 1, 1))
    @test CL._knapsack_owners([t(1), big, t(3)], act, 2) == ([1:1, 0:0, 1:1], 2)
    # Two 37-node tiles and three 25-node ones: largest first leaves 87 : 62,
    # and swapping a 37 for a 25 between the two ranks evens it to 75 : 74.
    a, b = BlockRegion((0, 0, 0), (13, 1, 1)), BlockRegion((20, 0, 0), (13, 1, 1))
    @test CL._knapsack_owners([a, b, t(1), t(2), t(3)], act, 2) ==
          ([1:1, 1:1, 0:0, 0:0, 0:0], 2)
    # Measured weights: a tile costs the busy time of its owner ranks, a
    # group's shared by volume, and a fresh tile the mean cost per node.
    w = CL._measured_weights([t(1), t(2), t(3)], act, [t(1), t(2)],
                             [0:1, 2:2], [1.0, 1.0, 3.0])
    @test w == [2.0, 3.0, 2.5]
    wg = CL._measured_weights([t(1), t(2)], act, [t(1), t(2)], [0:0, 0:0], [4.0])
    @test wg == [2.0, 2.0]
    @test CL._measured_weights([t(1), t(3)], act, [t(1)], [0:0], [0.0]) ==
          [25.0, 25.0]
    # The weighted partition of whole tiles: one rank each whatever the
    # weights, with the ranks at least as many as the tiles.
    @test CL._tile_owners([t(1), t(2), t(3)], act, 6; weights=w) ==
          ([0:0, 1:1, 2:2], 3)
    # Under `whole = false`: shares 1.6 : 2.4 : 2.0 of six ranks, at the
    # two-rank cap each.
    @test CL._tile_owners([t(1), t(2), t(3)], act, 6; weights=w, whole=false) ==
          ([0:1, 2:3, 4:5], 6)
    # Unequal weights past the cap: the heavy tile stops at two ranks.
    @test CL._tile_owners([t(1), t(2)], act, 4; weights=[1.0, 9.0], whole=false) ==
          ([0:1, 2:3], 4)
    @test CL._rank_counts([1.0, 9.0], [2, 2], 4) == [2, 2]
    @test CL._rank_counts([1.0, 9.0], [4, 4], 4) == [1, 3]
    # Admission is not monotone in the rank count: three 25 × 25 fine tiles
    # over eight ranks admit two ranks each or four, never three, so the
    # shares of 8/3 round to 4 : 2 : 2 rather than 3 : 3 : 2.
    lattice = [BlockRegion((8k, 0, 0), (9, 1, 9)) for k in 0:2]
    @test CL._tile_owners(lattice, (true, false, true), 8; whole=false) ==
          ([0:3, 4:5, 6:7], 8)
    # Configuration guards.
    per3l = ntuple(_ -> (PeriodicBC(), PeriodicBC()), 3)
    mk(; tile=0, regrid_interval=0, rebalance=0, rebalance_persist=2) = build_solver(
        Problem(domain=((0.0, 2π), (0.0, 1.0), (0.0, 1.0)), bcs=per3l, ic=nothing),
        Numerics(n_global=(192, 1, 1),
                 amr=AMR(initial=BlockRegion((80, 0, 0), (32, 1, 1)),
                         regrid_interval=regrid_interval, tile=tile, rebalance=rebalance,
                         rebalance_persist=rebalance_persist)))
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

@testset "per-dimension tile edges and lattice subdivision" begin
    # `tile` takes one edge or one per dimension, x first, an omitted
    # dimension taking the last entry given.
    @test CL._tile_edges(12) == (12, 12, 12)
    @test CL._tile_edges((4, 12)) == (4, 12, 12)
    @test CL._tile_edges((4, 12, 8)) == (4, 12, 8)
    @test !CL._tiled(0) && !CL._tiled((0, 0, 0)) && CL._tiled((4, 12))
    @test_throws ArgumentError CL._tile_edges((4, 12, 8, 2))
    @test_throws ArgumentError CL._tile_edges(4.0)
    per3l = ntuple(_ -> (PeriodicBC(), PeriodicBC()), 3)
    mk(; tile=0, regrid_interval=0) = build_solver(
        Problem(domain=((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)), bcs=per3l, ic=nothing),
        Numerics(n_global=(48, 48, 1),
                 amr=AMR(initial=BlockRegion((16, 16, 0), (17, 17, 1)),
                         regrid_interval=regrid_interval, tile=tile)))
    @test_throws ErrorException mk(tile=(4, 2))
    @test_throws ErrorException mk(tile=(4, 0))
    # A thin-normal lattice: tiles four parent nodes along x spanning the
    # region along y.
    s = mk(tile=(4, 16))
    @test [lt.region for lt in s.levels[2].transfers] ==
          [BlockRegion((16 + 4k, 16, 0), (5, 17, 1)) for k in 0:3]
    @test getfield(mk(tile=(4, 16), regrid_interval=2), :regrid).tile == (4, 16, 16)
    @test AMR(tile=(4, 12)).tile == (4, 12)
    # The chain of lattices a level may take: the largest edge halved in turn,
    # the lowest dimension first among equals, down to four parent nodes,
    # and only along a dimension where the halved lattice reaches the domain
    # faces the configured one does. A dimension of 12 nodes ending on a
    # face its lattice reaches keeps an edge of 4 out (the last cell would
    # span 3 parent cells, under the margin), and one rank takes the
    # configured lattice alone.
    none = CL._NO_BOUNDARY
    @test CL._edge_chain((16, 16, 16), (true, true, false), (48, 48, 1), none, none,
                         4, 8, 4, 8) ==
          [(16, 16, 16), (8, 16, 16), (8, 8, 16), (4, 8, 16), (4, 4, 16)]
    faces = ((true, true), (true, true), (true, true))
    bare = ((true, true), (false, false), (false, false))
    @test CL._edge_chain((12, 12, 12), (true, true, true), (73, 12, 12), faces, bare,
                         4, 8, 4, 8) == [(12, 12, 12), (6, 12, 12), (6, 6, 12), (6, 6, 6)]
    @test CL._edge_chain((12, 12, 12), (true, true, true), (73, 12, 12), faces, bare,
                         4, 8, 4, 1) == [(12, 12, 12)]
    @test CL._edge_chain((4, 12, 12), (true, true, false), (101, 12, 1), faces, bare,
                         4, 8, 4, 8) == [(4, 12, 12), (4, 6, 12)]
    # The choice among them, from the predicted seconds of the level on
    # each and of the rest of the step: the shortest step, the coarsest among
    # equals, unless the current lattice's step is within a tenth of it.
    @test CL._cheapest_lattice([10.0, 6.0, 7.0], 2.0, 1, false) == 2
    @test CL._cheapest_lattice([10.0, 9.5, 12.0], 2.0, 1, false) == 1
    @test CL._cheapest_lattice([10.0, 9.5, 12.0], 2.0, 1, true) == 2
    @test CL._cheapest_lattice([10.0, 9.5, 12.0], 2.0, nothing, false) == 2
    @test CL._cheapest_lattice([5.0, 5.0], 0.0, nothing, true) == 1
    # A level taking a small share of the step keeps its lattice although
    # another would halve its own time.
    @test CL._cheapest_lattice([1.0, 0.5], 20.0, 1, false) == 1
    # Under measured coefficients a level leaves its lattice once the model
    # has preferred the same other one at three consecutive checks; a check
    # preferring the level's own, or a third lattice, starts the count again.
    # Fixed coefficients, a level holding no tile and a lattice outside the
    # chain take the cheapest at once.
    let chain = [(16, 16, 16), (8, 16, 16), (8, 8, 16)], measured = CL.LatticeCosts()
        persist(c, cheapest, current=1, fresh=false) =
            CL._persistent_lattice!(c, 1, chain, cheapest, current, fresh)
        @test persist(measured, 2) == 2
        measured.timing = true
        @test [persist(measured, i) for i in (2, 2, 2)] == [1, 1, 2]
        @test [persist(measured, i) for i in (2, 2, 1, 2, 2, 2)] == [1, 1, 1, 1, 1, 2]
        @test [persist(measured, i) for i in (2, 2, 3, 3, 3)] == [1, 1, 1, 1, 3]
        @test isempty(measured.favored)
        @test persist(measured, 2, 1, true) == 2
        @test persist(measured, 2, nothing) == 2
        # Each level counts on its own.
        @test CL._persistent_lattice!(measured, 1, chain, 2, 1, false) == 1
        @test CL._persistent_lattice!(measured, 2, chain, 2, 1, false) == 1
        @test measured.favored == Dict(1 => ((8, 16, 16), 1), 2 => ((8, 16, 16), 1))
    end
    # The level's price: per rank, s (a V + b V g + k) over its tiles and
    # m s per message (one coupling per tile, one per face shared with
    # another rank's tile), at the most loaded rank. Two abutting 13 × 13
    # tiles with interface ends along both dimensions.
    pair = [BlockRegion((0, 0, 0), (5, 5, 1)), BlockRegion((4, 0, 0), (5, 5, 1))]
    act = (true, true, false)
    bnd = [none, none]
    one_rank = [0:0, 0:0]
    two_ranks = [0:0, 1:1]
    costs = CL.LatticeCosts()
    price(owners, np; s=1.0) = CL._layout_cost(costs, pair, bnd, bnd, owners, act,
                                              (0, 0, 0), s, np)
    @test price(one_rank, 1) == 338
    @test price(two_ranks, 2) == 169
    @test price(two_ranks, 2; s=3.0) == 507
    costs.b, costs.k, costs.m = 1.0, 5.0, 10.0
    @test CL._rank_messages(pair, two_ranks, CL._tile_faces(pair), 2) == [2, 2]
    @test CL._rank_messages(pair, one_rank, CL._tile_faces(pair), 1) == [2]
    @test price(one_rank, 1) == 2 * (169 + 2 * 169 + 5) + 10 * 2
    @test price(two_ranks, 2) == 169 + 2 * 169 + 5 + 10 * 2
    # The fit behind the coefficients: y = α x + β, both non-negative, from
    # the sums; through the origin where x does not spread.
    fit(x, y) = CL._nonnegative_fit(sum(x .^ 2), sum(x), Float64(length(x)),
                                    sum(x .* y), sum(y), CL.LATTICE_SPREAD)
    @test all(fit([1.0, 2.0, 3.0], [5.0, 7.0, 9.0]) .≈ (2.0, 3.0))
    @test all(fit([2.0, 2.0, 2.0], [7.0, 7.0, 7.0]) .≈ (3.5, 0.0))
    @test all(fit([1.0, 2.0, 3.0], [1.0, 3.0, 5.0]) .≈ (22 / 14, 0.0))
    @test all(fit([1.0, 2.0, 3.0], [6.0, 5.0, 4.0]) .≈ (0.0, 5.0))
end

@testset "lattice decision over nodes and at large tile counts" begin
    act = (true, true, true)
    # Tiles of a lattice of edges (4, 6, 8) over a 41 × 31 × 33 node space,
    # every other cell, clipped at the ends.
    lattice = [BlockRegion((4k1, 6k2, 8k3),
                           (min(5, 41 - 4k1), min(7, 31 - 6k2), min(9, 33 - 8k3)))
               for k3 in 0:3, k2 in 0:4, k1 in 0:9 if isodd(k1 + k2 + k3) || k1 < 3]
    lattice = [r for r in lattice if all(>=(2), r.extent)]
    # The face table by table lookup against the pairwise comparison it
    # replaces, the seam of a period included.
    function faces_pairwise(regions, period)
        same_node(a, b, d) = period[d] == 0 ? a == b : mod(a - b, period[d]) == 0
        map(eachindex(regions)) do p
            rp = regions[p]
            ntuple(3) do d
                lo = hi = 0
                for q in eachindex(regions)
                    q == p && continue
                    rq = regions[q]
                    all(e -> e == d || (same_node(rq.offset[e], rp.offset[e], e) &&
                                        rq.extent[e] == rp.extent[e]), 1:3) || continue
                    same_node(rq.offset[d] + rq.extent[d] - 1, rp.offset[d], d) && (lo = q)
                    same_node(rp.offset[d] + rp.extent[d] - 1, rq.offset[d], d) && (hi = q)
                end
                (lo, hi)
            end
        end
    end
    @test CL._tile_faces(lattice) == faces_pairwise(lattice, (0, 0, 0))
    @test CL._tile_faces(lattice, (40, 0, 0)) == faces_pairwise(lattice, (40, 0, 0))
    @test any(f -> f[1][1] != 0, CL._tile_faces(lattice, (40, 0, 0)))
    # The wanted tiles through the old tiles' index against the test of every
    # cell against every old tile it replaces, on the current lattice and on
    # a finer one, with cells at the hold level and young old tiles.
    per3l = ntuple(_ -> (PeriodicBC(), PeriodicBC()), 3)
    spec = getfield(build_solver(
        Problem(domain=((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)), bcs=per3l, ic=nothing),
        Numerics(n_global=(48, 1, 1),
                 amr=AMR(initial=BlockRegion((16, 0, 0), (17, 1, 1)), regrid_interval=5,
                         tile=8))), :regrid)
    N, lo, hi = (49, 37, 25), (5, 5, 5), (45, 33, 21)
    a0, a1 = (8, 8, 8), (4, 8, 8)
    tiles(a) = [t for k3 in 0:3, k2 in 0:4, k1 in 0:6
                for t in (CL._lattice_tile((k1, k2, k3), act, a, lo, hi),)
                if t !== nothing]
    old = [t for (i, t) in enumerate(tiles(a0)) if i % 3 != 0]
    spec.checks, spec.lifetime = 5, 3
    empty!(spec.created[1])
    foreach(((i, r),) -> spec.created[1][r] = isodd(i) ? 4 : 0, enumerate(old))
    function wanted_pairwise(flags, K, a, current)
        out = BlockRegion[]
        young(r) = spec.checks - get(spec.created[1], r, 0) < spec.lifetime
        for k3 in 0:K[3]-1, k2 in 0:K[2]-1, k1 in 0:K[1]-1
            t = CL._lattice_tile((k1, k2, k3), act, a, lo, hi)
            t === nothing && continue
            f = flags[k1 + 1, k2 + 1, k3 + 1]
            ok = f == CL.TAG_MARK || (current ? t in old && (f == CL.TAG_HOLD || young(t)) :
                 (over = [o for o in old if CL._overlaps(t, o, act, (0, 0, 0))];
                  !isempty(over) && (f == CL.TAG_HOLD || any(young, over))))
            ok && push!(out, t)
        end
        return out
    end
    for (a, current) in ((a0, true), (a1, false))
        K = ntuple(d -> (N[d] - 1) ÷ a[d] + 1, 3)
        flags = Int8[(i * 7 + j * 3 + k) % 4 == 0 ? CL.TAG_MARK :
                     (i + j * k) % 3 == 0 ? CL.TAG_HOLD : 0
                     for i in 1:K[1], j in 1:K[2], k in 1:K[3]]
        got = CL._lattice_wanted(flags, K, a, act, lo, hi, spec, old, current, (0, 0, 0))
        @test got == wanted_pairwise(flags, K, a, current)
        @test !isempty(got)
    end
    # The region index finds every region a query overlaps.
    bins = CL._RegionIndex(old, (0, 0, 0))
    for t in tiles(a1)
        @test [i for (i, o) in enumerate(old) if CL._overlaps(t, o, act, (0, 0, 0))] ⊆
              CL._candidates(bins, t)
    end
    # The chain stops at the cell budget, the configured lattice always kept.
    none = CL._NO_BOUNDARY
    full = CL._edge_chain((16, 16, 16), act, (97, 97, 97), none, none, 4, 8, 4, 8)
    @test length(full) == 7
    @test CL._edge_chain((16, 16, 16), act, (97, 97, 97), none, none, 4, 8, 4, 8,
                         7 * 7 * 13) == full[1:2]
    @test CL._edge_chain((16, 16, 16), act, (97, 97, 97), none, none, 4, 8, 4, 8, 1) ==
          full[1:1]
    @test CL._lattice_budget(8) == 1 << 20 && CL._lattice_budget(11200) == 256 * 11200
    # Placement across nodes: one node is the single-node placement; two
    # nodes of four ranks each take a run of the curve, balanced over their
    # ranks, and every tile of a run on its own node's ranks.
    act2 = (true, true, false)
    sq = [BlockRegion((4k1, 4k2, 0), (5, 5, 1)) for k2 in 0:3 for k1 in 0:5]
    @test CL._knapsack_owners(sq, act2, 8; nodes=zeros(Int, 8)) ==
          CL._knapsack_owners(sq, act2, 8)
    nodes = [r ÷ 4 for r in 0:7]
    owners, np = CL._knapsack_owners(sq, act2, 8; nodes)
    @test np == 8 && all(o -> length(o) == 1, owners)
    curve = CL._sfc_order(sq)
    on = [nodes[first(owners[t]) + 1] for t in curve]
    @test on == sort(on) && count(==(0), on) == 12
    @test maximum(count(o -> o == r:r, owners) for r in 0:7) == 3
    # A renumbering to keep survivors stays within each node.
    kept = CL._keep_survivors(owners, sq, act2, 8, [5:5 for _ in sq], trues(length(sq));
                              nodes)
    @test all(nodes[first(kept[t]) + 1] == nodes[first(owners[t]) + 1] for t in eachindex(sq))
    @test CL._keep_survivors(owners, sq, act2, 8, [5:5 for _ in sq], trues(length(sq))) ==
          CL._keep_survivors(owners, sq, act2, 8, [5:5 for _ in sq], trues(length(sq));
                             nodes=zeros(Int, 8))
    # Messages split by node: a pair of abutting tiles on two ranks of one
    # node exchange within it, on two nodes across, and each coupling with
    # the parent counts as leaving the node once the run spans two.
    pair = [BlockRegion((0, 0, 0), (5, 5, 1)), BlockRegion((4, 0, 0), (5, 5, 1))]
    pf = CL._tile_faces(pair)
    @test CL._rank_messages(pair, [0:0, 1:1], pf, 2, [0, 0]) == ([2, 2], [0, 0])
    @test CL._rank_messages(pair, [0:0, 1:1], pf, 2, [0, 1]) == ([0, 0], [2, 2])
    @test CL._rank_messages(pair, [0:0, 1:1], pf, 4, [0, 0, 1, 1]) == ([1, 1, 0, 0],
                                                                       [1, 1, 0, 0])
    costs = CL.LatticeCosts()
    costs.a, costs.m, costs.m_x = 0.0, 1.0, 10.0
    bnd = [none, none]
    @test CL._layout_cost(costs, pair, bnd, bnd, [0:0, 1:1], act2, (0, 0, 0), 1.0, 2,
                          [0, 1]) == 20.0
    @test CL._layout_cost(costs, pair, bnd, bnd, [0:0, 1:1], act2, (0, 0, 0), 1.0, 2,
                          [0, 0]) == 2.0
    # The message coefficients: on one node the seconds over the messages;
    # both apart from exact observations that vary apart; one for every
    # message where the two counts keep one proportion.
    sums(obs) = foldl((s, (x1, x2, y)) -> CL._message_sums!(s, x1, x2, y), obs;
                      init=zeros(13))
    fitm(obs) = (s = sums(obs); CL._message_fit(s[9], s[10], s[11], s[12], s[13],
                                                CL.LATTICE_SPREAD))
    @test all(fitm([(2.0, 0.0, 3.0), (4.0, 0.0, 5.0)]) .≈ (8 / 6, 8 / 6))
    @test all(fitm([(2.0, 1.0, 7.0), (1.0, 3.0, 16.0), (3.0, 0.0, 3.0)]) .≈ (1.0, 5.0))
    @test all(fitm([(2.0, 1.0, 4.0), (4.0, 2.0, 8.0)]) .≈ (4 / 3, 4 / 3))
    @test fitm(Tuple{Float64,Float64,Float64}[]) == (0.0, 0.0)
    # A fit on its boundary at a = 0 takes the tiles' seconds per node alone:
    # y = 6, 5, 4 at x = 1, 2, 3 falls with x, and a = Σxy / Σx² = 2.
    fitted = CL.LatticeCosts()
    fitted.stats = [14.0, 6.0, 3.0, 28.0, 15.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    @test CL._fit_lattice_costs(fitted)[1:4] == [1.0, 2.0, 0.0, 0.0]
    @test CL._fit_lattice_costs(CL.LatticeCosts())[1] == 0.0
    # A check rebuilds on the busiest rank the tiles it holds.
    @test CL._most_held([0:0, 1:1, 0:0, 2:3]) == 2 && CL._most_held(UnitRange{Int}[]) == 0
    # One process is one node; the toggle groups consecutive ranks.
    @test CL._node_map(MPI.COMM_WORLD) == [0]
    CL.with_knob(CL.NODE_SIZE => 2) do
        @test CL._node_map(MPI.COMM_WORLD) == [0]
    end
end

@testset "tile migration: interior rule and the rank's own copy" begin
    # `_migrate_tile!` moves a tile's solution between two decompositions
    # of it from their block tables alone. On one rank both blocks are the
    # whole tile, so the one message is the rank's copy between its own
    # blocks: the interior, one node off every boundary plane, arrives and
    # the shell keeps the destination's values, as `_carry_over!` of the
    # replicated gather leaves it.
    per3l = ntuple(_ -> (PeriodicBC(), PeriodicBC()), 3)
    solver = build_solver(
        Problem(domain=((0.0, 2π), (0.0, 1.0), (0.0, 1.0)), bcs=per3l, ic=nothing),
        Numerics(n_global=(192, 1, 1),
                 amr=AMR(initial=BlockRegion((80, 0, 0), (32, 1, 1)), regrid_interval=5,
                         tile=8)))
    states = allocate_state(solver)
    initialize!(solver, states,
                (x, y, z) -> Prim(u=(0, 0, 0), p=1.0, rho=1.0 + 0.1sin(x)))
    lev = getfield(solver, :levels)[2]
    li = lev.patches[1]
    lt = lev.transfers[lev.tiles[1]]
    dp = getfield(solver, :patches)[li].decomp
    n_cons = solver.equations.n_cons
    Q_old = states.patches[li]
    Q_new = ConservedState(fill(-1.0, size(parent(Q_old))))
    Nf = CL.fine_extent(lt.region, lt.active)
    blocks = lev.coupling.fine_blocks[lev.tiles[1]]
    CL._migrate_tile!(Float64, Q_new, dp, Q_old, dp, Nf, lt.active, blocks, blocks,
                      dp.comm)
    pad = dp.n_halo_d
    inner = ntuple(d -> lt.active[d] ? ((2 + pad[d]):(Nf[d] - 1 + pad[d])) : (1:1), 3)
    @test view(parent(Q_new), inner..., :) == view(parent(Q_old), inner..., :)
    untouched = trues(size(parent(Q_new)))
    untouched[inner..., :] .= false
    @test all(parent(Q_new)[untouched] .== -1.0)
    Qref = ConservedState(fill(-1.0, size(parent(Q_old))))
    gathered = CL._gather_tile(Float64, lt.region, lt.active, n_cons, lt, solver,
                               states.patches)
    CL._carry_over!(Qref, dp, lt.region, gathered, Nf, lt.region, lt.active, n_cons)
    @test parent(Qref) == parent(Q_new)
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
    sf = build_solver(
        Problem(domain=((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)), bcs=(wall2, per, per),
                ic=nothing), Numerics(n_global=(Nf, 1, 1), cfl=0.2))
    Qf = allocate_state(sf)
    initialize!(sf, Qf, ic)
    run!(sf, Qf; tfinal=tf, nmax=40000)
    padF = sf.decomp.n_halo_d[1]
    rho_ref = [Qf[i + padF, 1, 1, 1] for i in 1:Nf]
    sc = build_solver(
        Problem(domain=((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)), bcs=(wall2, per, per),
                ic=nothing), Numerics(n_global=(N, 1, 1), cfl=0.2))
    Qc = allocate_state(sc)
    initialize!(sc, Qc, ic)
    run!(sc, Qc; tfinal=tf, nmax=40000)
    padC = sc.decomp.n_halo_d[1]
    e_base = maximum(abs(Qc[i + padC, 1, 1, 1] - rho_ref[3i - 2]) for i in 1:N)
    # AMR: subcycled with the region retagged every 5 coarse steps.
    sa = build_solver(
        Problem(domain=((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)), bcs=(wall2, per, per),
                ic=nothing),
        Numerics(n_global=(N, 1, 1), cfl=0.2,
                 amr=AMR(initial=BlockRegion((85, 0, 0), (31, 1, 1)), subcycle=true,
                         regrid_interval=5)))
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
    @test released_communicators(lt0.pdecomps[1])
    @test released_communicators(decomp0)
    padc = sa.patches[1].decomp.n_halo_d[1]
    padf = sa.patches[2].decomp.n_halo_d[1]
    e_amr = 0.0
    for i in 1:N
        v = lo <= i <= hi ? states.patches[2][3 * (i - lo) + 1 + padf, 1, 1, 1] :
                            states.patches[1][i + padc, 1, 1, 1]
        e_amr = max(e_amr, abs(v - rho_ref[3i - 2]))
    end
    @test nlevels(sa) == 2
    @info "moving-region Sod vs uniform fine" e_amr e_base
    # Measured: composite 8.6e-4 (9.5e-4 under the closure rows) against the
    # uniform-coarse baseline's 7.8e-2; the refinement recovers most of the
    # uniform-fine answer at a third of the fine points.
    @test e_amr < 1.5e-2
    @test e_base > 5e-2
    @test e_amr < e_base / 3
    for (psq, Q) in CL.eachpatch(sa, states)
        n = psq.decomp.n_local
        padq = psq.decomp.n_halo_d[1]
        @test minimum(Q[i + padq, 1, 1, 1] for i in 1:n[1]) > 0.05
    end
    # The global-dt coupling regrids through the same machinery.
    sg = build_solver(
        Problem(domain=((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)), bcs=(wall2, per, per),
                ic=nothing),
        Numerics(n_global=(N, 1, 1), cfl=0.2,
                 amr=AMR(initial=BlockRegion((85, 0, 0), (31, 1, 1)), regrid_interval=5)))
    states_g = allocate_state(sg)
    initialize!(sg, states_g, ic)
    run!(sg, states_g; tfinal=0.05, nmax=20000)
    @test CL.refined_region(sg).offset[1] != 85
end

@testset "tiled regridding tracks a Sod shock" begin
    wall2 = (SlipWallBC(), SlipWallBC())
    per = (PeriodicBC(), PeriodicBC())
    ic(x, y, z) = x < 0.5 ? Prim(u=(0, 0, 0), p=1.0, rho=1.0) :
                            Prim(u=(0, 0, 0), p=0.1, rho=0.125)
    N = 201
    sa = build_solver(
        Problem(domain=((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)), bcs=(wall2, per, per),
                ic=nothing),
        Numerics(n_global=(N, 1, 1), cfl=0.2,
                 amr=AMR(initial=BlockRegion((85, 0, 0), (31, 1, 1)), subcycle=true,
                         regrid_interval=5, tile=8)))
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
    @test all(released_communicators(lt.pdecomps[1]) for lt in transfers0)
    # Every tile is a lattice cell, the set is in lattice order and holds
    # the shock. It need not be contiguous: under the default hold band the
    # rarefaction head, a marginal feature, keeps its tiles at the far end.
    @test all(r -> r.offset[1] % 8 == 0 && r.extent[1] == 9, regs)
    @test all(i -> regs[i+1].offset[1] > regs[i].offset[1], 1:length(regs)-1)
    @test regs[1].offset[1] < shock_node <= regs[end].offset[1] + 9
    @test Set(keys(getfield(sa, :regrid).created[1])) == Set(regs)
    @test all(all(isfinite, parent(Q)) for Q in states.patches)
    for (psq, Q) in CL.eachpatch(sa, states)
        n = psq.decomp.n_local[1]
        @test minimum(Q[padded_index(psq, i, 1, 1), 1] for i in 1:n) > 0.05
    end
    @test length(sa.patches) == length(states.patches) == length(regs) + 1
end

@testset "regridding rebuilds the d8 detector and a pentadiagonal filter" begin
    # Rebuilt patches take the detector's interface plans, its buffer and the
    # banded filter. Composite mass drift measured 6.3e-6 (tiled) and 3.5e-7
    # (box), the regrids' transfers included.
    wall2 = (SlipWallBC(), SlipWallBC())
    per = (PeriodicBC(), PeriodicBC())
    ic(x, y, z) = x < 0.5 ? Prim(u=(0, 0, 0), p=1.0, rho=1.0) :
                            Prim(u=(0, 0, 0), p=0.1, rho=0.125)
    for tile in (8, 0)
        s = build_solver(
            Problem(domain=((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)), bcs=(wall2, per, per),
                    ic=nothing),
            Numerics(n_global=(201, 1, 1), cfl=0.2, art=ArtificialProperties(detector=:d8),
                     filter=pyranda_filter(),
                     amr=AMR(initial=BlockRegion((85, 0, 0), (31, 1, 1)), subcycle=true,
                             regrid_interval=5, tag_buffer=16, tile=tile)))
        states = allocate_state(s)
        initialize!(s, states, ic)
        b0 = CL._conserved_budget(s, states)
        initial = level_regions(s, 1)
        run!(s, states; tfinal=0.2, nmax=40000)
        @test s.t ≈ 0.2
        @test level_regions(s, 1) != initial
        @test all(p -> p.level == 0 ||
                       p.plans.ring_plans[1] isa CL.InterfaceRingPlans, s.patches)
        for (psq, Q) in CL.eachpatch(s, states)
            n = psq.decomp.n_local[1]
            @test minimum(Q[padded_index(psq, i, 1, 1), 1] for i in 1:n) > 0.05
        end
        b1 = CL._conserved_budget(s, states)
        @test abs(b1.total_mass - b0.total_mass) / b0.total_mass < 5e-4
        @test abs(b1.total_energy - b0.total_energy) / b0.total_energy < 5e-4
    end
end

@testset "tiled regrid seeds fresh tiles from surviving neighbors" begin
    wall2 = (SlipWallBC(), SlipWallBC())
    per = (PeriodicBC(), PeriodicBC())
    ic(x, y, z) = x < 0.5 ? Prim(u=(0, 0, 0), p=1.0, rho=1.0) :
                            Prim(u=(0, 0, 0), p=0.1, rho=0.125)
    N = 201
    # The unrelaxed filter (filter_cfl = 0): the regrid below has to create a
    # fresh tile beside a survivor at exactly this step, which the tag state
    # at step 65 does under it (the run's own checks before it leave three
    # tiles, and this one adds a fourth). The step depends on the default
    # filter strength, on the root step size and on the parent's pass over
    # the covered nodes (8 under `MASK_CHILD_RESIDUAL[] = false`).
    sa = build_solver(
        Problem(domain=((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)), bcs=(wall2, per, per),
                ic=nothing),
        Numerics(n_global=(N, 1, 1), cfl=0.2, filter=StateFilter(cfl=0.0),
                 amr=AMR(initial=BlockRegion((85, 0, 0), (31, 1, 1)), subcycle=true,
                         regrid_interval=5, tile=8)))
    states = allocate_state(sa)
    initialize!(sa, states, ic)
    run!(sa, states; tfinal=1.0, nmax=65)
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
        @test states.patches[i + 1][padded_index(ps, nr, 1, 1), 1] == states.patches[j + 1][padded_index(qs, 1, 1, 1), 1]
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
    mk(; regrid_interval=0, tag_gradient_threshold=0, tag_sensor_threshold=0,
         tag_vorticity_threshold=0, tag_predicate=nothing,
         tag_threshold=CompactLES.DEFAULT_TAG_THRESHOLD, refine=r0) = build_solver(
        Problem(domain=((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)), bcs=(wall2, per, per), eos=two,
                ic=nothing),
        Numerics(n_global=(N, 1, 1), cfl=0.2,
                 amr=AMR(initial=refine, regrid_interval=regrid_interval,
                         tag_threshold=tag_threshold,
                         tag_sensor_threshold=tag_sensor_threshold,
                         tag_gradient_threshold=tag_gradient_threshold,
                         tag_vorticity_threshold=tag_vorticity_threshold));
        tag_predicate=tag_predicate)
    # Configuration guards: a criterion is a regrid setting, and a
    # threshold is non-negative.
    @test_throws ErrorException mk(regrid_interval=5, tag_gradient_threshold=-1)
    @test_throws ErrorException mk(tag_gradient_threshold=0.05)
    @test_throws ErrorException mk(tag_sensor_threshold=0.1)
    @test_throws ErrorException mk(tag_vorticity_threshold=1.0)
    @test_throws ErrorException mk(tag_predicate=(p, I) -> true)
    @test_throws ErrorException mk(regrid_interval=5, tag_predicate=1)
    # `AMR` carries the three thresholds to the solver.
    num = Numerics(n_global=(N, 1, 1),
                   amr=AMR(initial=r0, regrid_interval=5, tag_gradient_threshold=0.05,
                           tag_sensor_threshold=0.1, tag_vorticity_threshold=2.0))
    prob = Problem(domain=((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)),
                   bcs=(wall2, per, per), eos=two,
                   ic=(x, y, z) -> Prim(Y=(1.0, 0.0), rho=1.0, p=1.0, u=(0, 0, 0)))
    sn, _ = setup(prob, num)
    specn = getfield(sn, :regrid)
    @test specn.gradient_threshold == 0.05 && specn.sensor_threshold == 0.1
    @test specn.vorticity_threshold == 2.0
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
        r = CL.tagged_region(sg, states.patches[1])
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
        sv = build_solver(
            Problem(domain=((0.0, 2π), (0.0, 2π), (0.0, 1.0)), bcs=per3l, ic=nothing),
            Numerics(n_global=(N2, N2, 1),
                     amr=AMR(initial=BlockRegion((8, 8, 0), (12, 12, 1)), regrid_interval=5,
                             tag_vorticity_threshold=thr)))
        states = allocate_state(sv)
        initialize!(sv, states, icω)
        r = CL.tagged_region(sv, states.patches[1])
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

    # Sensor: the artificial diffusivity number of the coefficients a regrid
    # check computes from the state, which keep the sensors a step drops where
    # the level's restriction overwrites the root, the shock's here. The Sod
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
    spec = getfield(ss, :regrid)
    spec.sensor_threshold = 1.0         # on, so that the check's pass runs
    CL._tag_coefficients!(ss, states.patches)
    spec.sensor_threshold = 0.0
    root = PatchSolver(ss, ss.patches[1])
    qmax = 0.0
    for i in 1:root.decomp.n_local[1]
        I = padded_index(root, i, 1, 1)
        ν = (root.mu_art[I] + root.beta_art[I]) / root.rho[I] +
            root.kappa_art[I] / (root.rho[I] * root.cp_mix[I]) +
            maximum(D[I] for D in root.D_art)
        qmax = max(qmax, ν / (root.c[I] * root.h[1]))
    end
    @info "artificial diffusivity number at the Sod shock" qmax
    @test qmax > 0
    @test CL.tagged_region(ss, states.patches[1]) === nothing
    spec.sensor_threshold = 2 * qmax
    @test CL.tagged_region(ss, states.patches[1]) === nothing
    spec.sensor_threshold = qmax / 2
    r = CL.tagged_region(ss, states.patches[1])
    @test r !== nothing
    shock = node(0.5 + 1.75 * ss.t)
    @test r.offset[1] < shock <= r.offset[1] + r.extent[1]
    @test r.extent[1] < 60

    # Predicate: a closure over the parent patch and a padded index, here
    # x > 0.8, whose tagged set is that interval buffered and reaching the
    # slip wall at x = 1; in union with the δ⁴ tag at the initial
    # discontinuity the box spans both.
    predx = (p, I) -> xcoord(p, 1, interior_index(p, I)[1]) > 0.8
    sp = mk(regrid_interval=5, tag_predicate=predx)
    states = allocate_state(sp)
    initialize!(sp, states, (x, y, z) -> Prim(Y=(1.0, 0.0), rho=1.0, p=1.0,
                                             u=(0, 0, 0)))
    r = CL.tagged_region(sp, states.patches[1])
    # The first node past x = 0.8, in the solver's own arithmetic.
    g1 = findfirst(g -> (g - 1) * (1.0 / (N - 1)) > 0.8, 1:N)
    @test r == BlockRegion((g1 - 1 - 4, 0, 0), (N - (g1 - 1 - 4), 1, 1))
    initialize!(sp, states, icS)
    r = CL.tagged_region(sp, states.patches[1])
    @test r.offset[1] < node(0.5) && r.offset[1] + r.extent[1] == N
end

@testset "tag hysteresis: hold band, lifetime and the creation record" begin
    # A synthetic tag sequence, no time stepping: a quiescent field with two
    # Gaussian density bumps of width 2h, an anchor at x = 0.3 that always
    # tags and a feature at x = 0.7 whose amplitude each check sets, so the
    # feature's tag quantity max Σ|δ⁴ρ|/ρ lands above the threshold, in
    # the hold band, or below it. The quantity is evaluated here on the
    # same lattice with the same stencil, so the bands are asserted, not
    # assumed.
    wall2 = (SlipWallBC(), SlipWallBC())
    per = (PeriodicBC(), PeriodicBC())
    N = 201
    h = 1 / (N - 1)
    thr = 0.02
    bump(x, x0, a) = a * exp(-((x - x0) / (2h))^2)
    function quantity(a)
        ρ = [1 + bump((g - 1) * h, 0.7, a) for g in 1:N]
        return maximum(abs(sum(CL.D4[m + 3] * ρ[g + m] for m in -2:2)) / ρ[g]
                       for g in 3:N-2)
    end
    a_of(target) = target / quantity(1e-3) * 1e-3   # linear at small a
    a_hi, a_mid, a_lo = a_of(1.5thr), a_of(0.75thr), a_of(0.25thr)
    @test quantity(a_hi) > thr
    @test thr / 2 < quantity(a_mid) < thr
    @test quantity(a_lo) < thr / 2
    amp = Ref(a_hi)
    ic(x, y, z) = Prim(u=(0, 0, 0), p=1.0,
                       rho=1.0 + bump(x, 0.3, a_hi) + bump(x, 0.7, amp[]))
    anchor, feature = round(Int, 0.3 * (N - 1)) + 1, round(Int, 0.7 * (N - 1)) + 1
    has(regs, g) = any(r -> r.offset[1] < g <= r.offset[1] + r.extent[1], regs)
    mk(; untag_ratio=2, tile_lifetime=1, tile=0) = build_solver(
        Problem(domain=((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)), bcs=(wall2, per, per),
                ic=nothing),
        Numerics(n_global=(N, 1, 1),
                 amr=AMR(initial=BlockRegion((50, 0, 0), (20, 1, 1)), regrid_interval=1,
                         tag_threshold=thr, tag_buffer=1, untag_ratio=untag_ratio,
                         tile_lifetime=tile_lifetime, tile=tile)))
    # One regrid check with the feature at amplitude `a`: the state is
    # re-initialized and the cadence hook runs, which is what counts checks.
    function check!(solver, states, workspace, a)
        amp[] = a
        initialize!(solver, states, ic)
        solver.step += 1
        CL._maybe_regrid!(solver, states, workspace, nothing)
        return level_regions(solver, 1)
    end
    @test_throws ErrorException mk(untag_ratio=0.5)
    @test_throws ErrorException mk(tile_lifetime=0)

    # Tiled, default hysteresis (ratio 2, lifetime 1).
    let
        sa = mk(tile=8)
        states = allocate_state(sa)
        ws = CL.Workspace(states)
        spec = getfield(sa, :regrid)
        @test spec.checks == 0 && all(==(0), values(spec.created[1]))
        @test Set(keys(spec.created[1])) == Set(level_regions(sa, 1))
        r1 = check!(sa, states, ws, a_hi)
        @test has(r1, anchor) && has(r1, feature)
        @test spec.checks == 1
        @test all(r -> spec.created[1][r] == (has([r], feature) ? 1 : 0), r1)
        @test Set(keys(spec.created[1])) == Set(r1)
        # The hold band keeps the feature's tile; below it the tile goes.
        r2 = check!(sa, states, ws, a_mid)
        @test r2 == r1
        r3 = check!(sa, states, ws, a_lo)
        @test has(r3, anchor) && !has(r3, feature)
        @test Set(keys(spec.created[1])) == Set(r3)
        # Fading back into the hold band creates nothing.
        r4 = check!(sa, states, ws, a_mid)
        @test r4 == r3
    end
    # Tiled, no hold band: the feature's tile is dropped as soon as it
    # falls below the threshold.
    let
        sa = mk(tile=8, untag_ratio=1)
        states = allocate_state(sa)
        ws = CL.Workspace(states)
        r1 = check!(sa, states, ws, a_hi)
        @test has(r1, feature)
        r2 = check!(sa, states, ws, a_mid)
        @test has(r2, anchor) && !has(r2, feature)
    end
    # Tiled, lifetime 3: the setup-time tiles beyond the anchor's tags and
    # the feature's tile each survive until their third check.
    let
        sa = mk(tile=8, tile_lifetime=3)
        states = allocate_state(sa)
        ws = CL.Workspace(states)
        spec = getfield(sa, :regrid)
        initial = level_regions(sa, 1)
        r1 = check!(sa, states, ws, a_hi)
        @test all(r -> r in r1, initial) && has(r1, feature)
        r2 = check!(sa, states, ws, a_lo)
        @test r2 == r1
        r3 = check!(sa, states, ws, a_lo)
        @test has(r3, feature) && !all(r -> r in r3, initial) && has(r3, anchor)
        r4 = check!(sa, states, ws, a_lo)
        @test !has(r4, feature) && has(r4, anchor)
        @test spec.checks == 4 && Set(keys(spec.created[1])) == Set(r4)
    end
    # The box (tile = 0): spans both features, holds through the band,
    # then shrinks to the anchor; with a lifetime of 2 it cannot move at
    # its first check.
    let
        sa = mk()
        states = allocate_state(sa)
        ws = CL.Workspace(states)
        spec = getfield(sa, :regrid)
        r1 = check!(sa, states, ws, a_hi)
        @test has(r1, anchor) && has(r1, feature) && length(r1) == 1
        @test spec.created[1] == Dict(r1[1] => 1)
        r2 = check!(sa, states, ws, a_mid)
        @test r2 == r1
        r3 = check!(sa, states, ws, a_lo)
        @test has(r3, anchor) && !has(r3, feature)
        @test spec.created[1] == Dict(r3[1] => 3)
        sb = mk(tile_lifetime=2)
        states = allocate_state(sb)
        ws = CL.Workspace(states)
        initial = level_regions(sb, 1)
        @test check!(sb, states, ws, a_hi) == initial
        r2 = check!(sb, states, ws, a_hi)
        @test r2 != initial && has(r2, feature)
    end
    # A hold-only signal keeps the box whole. With no anchor, a feature at
    # the hold level inside the initial box leaves it as it is, where the
    # bounds of the held nodes alone would have shrunk it onto them; the
    # same feature at the tag level moves the box onto itself.
    let
        icf(x, y, z) = Prim(u=(0, 0, 0), p=1.0, rho=1.0 + bump(x, 0.3, amp[]))
        sc = mk()
        states = allocate_state(sc)
        ws = CL.Workspace(states)
        initial = level_regions(sc, 1)
        @test has(initial, anchor)
        amp[] = a_mid
        initialize!(sc, states, icf)
        @test CL.tagged_region(sc, states.patches[1]) === nothing
        sc.step += 1
        CL._maybe_regrid!(sc, states, ws, nothing)
        @test level_regions(sc, 1) == initial
        amp[] = a_hi
        initialize!(sc, states, icf)
        sc.step += 1
        CL._maybe_regrid!(sc, states, ws, nothing)
        r = level_regions(sc, 1)
        @test r != initial && has(r, anchor) && r[1].extent[1] < 12
    end
end

@testset "box regrid clamps into the nesting margin" begin
    # A tagged set inside the upper margin band: both ends clamp onto the
    # feasible interval and the four-node minimum widens it inward, so the
    # box never reaches nodes the parent gather does not fill. The old
    # clamp kept the tagged end and widened outward, and interpolating a
    # constant state into the box then read zeros.
    wall2 = (SlipWallBC(), SlipWallBC())
    per = (PeriodicBC(), PeriodicBC())
    N = 40
    edge = (p, I) -> interior_index(p, I)[1] == N
    # The margin holds at every face with the placement on faces off.
    s = build_solver(
        Problem(domain=((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)), bcs=(wall2, per, per),
                ic=nothing),
        Numerics(n_global=(N, 1, 1),
                 amr=AMR(initial=BlockRegion((10, 0, 0), (8, 1, 1)), regrid_interval=1,
                         tag_buffer=0, level_boundaries=false)); tag_predicate=edge)
    states = allocate_state(s)
    initialize!(s, states, (x, y, z) -> Prim(u=(0, 0, 0), p=1.0, rho=1.0))
    margin = getfield(s, :regrid).margin
    r = CL.tagged_region(s, states.patches[1])
    @test r == BlockRegion((N - margin - 4, 0, 0), (4, 1, 1))
    ws = CL.Workspace(states)
    s.step += 1
    CL._maybe_regrid!(s, states, ws, nothing)
    @test level_regions(s, 1) == [r]
    @test all(Q -> all(isfinite, parent(Q)), states.patches)
    ps = PatchSolver(s, s.patches[2])
    @test all(abs(states.patches[2][padded_index(ps, i, 1, 1), 1] - 1) < 1e-12
              for i in 1:ps.decomp.n_local[1])
    # The substep counts of the recursive driver: one-based at every
    # level, level 2 taking 1 .. 9 under the first root step.
    counts2 = [CL._child_count(CL._child_count(1, m1), m2) for m1 in 1:3, m2 in 1:3]
    @test sort(vec(counts2)) == 1:9
    @test [CL._child_count(s0 + 1, m) for s0 in 0:1 for m in 1:3] == 1:6
end

@testset "tiled regrid starts unrefined and refines on a boundary-driven shock" begin
    # A tube at rest with a Dirichlet inflow that ramps to the post-shock
    # state of a Mach 1.5 shock over 0.04 < t < 0.05. The initial state tags
    # nothing, so the frontend starts with a tiled level holding no tiles;
    # the shock fired from the inflow creates them and they follow it. Runs
    # checkpointed before and after the creation continue bit for bit.
    per = (PeriodicBC(), PeriodicBC())
    ramp(t) = clamp((t - 0.04) / 0.01, 0.0, 1.0)
    inflow = DirichletBC((x, y, z, t) -> (w = ramp(t);
        Prim(rho=1 + 0.8621w, u=(0.8216w, 0.0, 0.0), p=1 + 1.4583w)))
    rest(x, y, z) = Prim(rho=1.0, u=(0.0, 0.0, 0.0), p=1.0)
    prob = Problem(eos=IdealSpecies("gas"; gamma=1.4, R=1.0),
                   domain=((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)),
                   bcs=((inflow, SlipWallBC()), per, per), ic=rest)
    num = Numerics(n_global=(121, 1, 1), cfl=0.3,
                   amr=AMR(initial=:sensor, tile=8, regrid_interval=5,
                           tag_buffer=2, subcycle=true))
    same(a, b, decomp) = (inner = CL.interior(decomp);
                          parent(a)[inner, :] == parent(b)[inner, :])
    empty_level(s, states) = isempty(level_regions(s, 1)) && nlevels(s) == 2 &&
        length(states.patches) == length(s.patches) == 1 &&
        !getfield(s, :levels)[2].level_comm.owned &&
        all(==(0), getfield(s, :patches)[1].amr.covered)
    s, states = setup(prob, num)
    @test empty_level(s, states)
    dir = mktempdir()
    # Measured: the first tile appears at the check after step 20
    # (t = 0.043), a second after step 40 and a third after step 95, the
    # last holding the shock.
    run!(s, states; tfinal=1.0, nmax=15)
    @test empty_level(s, states)
    save_checkpoint(s, states, joinpath(dir, "before"))
    run!(s, states; tfinal=1.0, nmax=50)
    regs50 = level_regions(s, 1)
    @test !isempty(regs50) && length(states.patches) == length(s.patches) == 1 + length(regs50)
    save_checkpoint(s, states, joinpath(dir, "after"))
    run!(s, states; tfinal=1.0, nmax=110)
    regs = level_regions(s, 1)
    # The shock is the first root node below the mean of the two densities.
    root = CL.PatchSolver(s, s.patches[1])
    shock = findfirst(i -> states.patches[1][padded_index(root, i, 1, 1), 1] < 1.43, 1:121)
    @info("unrefined start", tiles50=Tuple(r.offset[1] for r in regs50),
          tiles80=Tuple(r.offset[1] for r in regs), shock)
    @test length(regs) > length(regs50)
    @test regs[end].offset[1] > regs50[end].offset[1]
    @test any(r -> r.offset[1] < shock <= r.offset[1] + r.extent[1], regs)
    @test all(all(isfinite, parent(Q)) for Q in states.patches)
    for stem in ("before", "after")
        r, rs = setup(prob, num)
        load_checkpoint!(r, rs, joinpath(dir, stem))
        @test level_regions(r, 1) == (stem == "before" ? BlockRegion[] : regs50)
        run!(r, rs; tfinal=1.0, nmax=110)
        @test r.t == s.t && level_regions(r, 1) == regs
        @test getfield(r, :regrid).created == getfield(s, :regrid).created
        @test length(rs.patches) == length(states.patches) &&
              all(same(rs.patches[i], states.patches[i], s.patches[i].decomp) for i in eachindex(states.patches))
    end
    rm(dir; recursive=true)
    # A check at which nothing tags removes every tile; the next tag, from
    # the inflow's shock re-forming on the quiescent state, creates them again.
    initialize!(s, states, rest)
    workspace = CL.Workspace(states)
    getfield(s, :regrid).checks += 1
    @test CL.regrid!(s, states, workspace, nothing)
    @test empty_level(s, states) && length(workspace.dQ.patches) == 1
    @test isempty(getfield(s, :regrid).created[1])
    run!(s, states; tfinal=1.0, nmax=s.step + 10)
    @test !isempty(level_regions(s, 1))
    @test all(all(isfinite, parent(Q)) for Q in states.patches)
end

@testset "tag sweep over a tiled parent matches a uniform sweep" begin
    # A level-1 tile set swept with its edges open reads the same density
    # differences as a uniform grid at the level-1 spacing: exactly across
    # the tiles' shared faces, whose ghosts the level records fill from the
    # neighbor, and away from the parent-fed faces, whose shell is the
    # parent's interpolation rather than the initial condition.
    wall2 = (SlipWallBC(), SlipWallBC())
    per = (PeriodicBC(), PeriodicBC())
    ic(x, y, z) = Prim(u=(0, 0, 0), p=1.0, rho=1.0 + 0.5 * tanh((x - 0.5) / 0.005))
    N = 81
    mk(n; refine=nothing, max_levels=nothing) = build_solver(
        Problem(domain=((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)), bcs=(wall2, per, per),
                ic=nothing),
        Numerics(n_global=(n, 1, 1),
                 amr=AMR(initial=refine, regrid_interval=1, tile=8, max_levels=max_levels)))
    s = mk(N; refine=BlockRegion((24, 0, 0), (33, 1, 1)), max_levels=3)
    @test [r.offset[1] for r in level_regions(s, 1)] == [24, 32, 40, 48]
    states = allocate_state(s)
    initialize!(s, states, ic)
    CL._presync!(s, states)
    u = mk(3 * (N - 1) + 1; max_levels=2)
    Qu = allocate_state(u)
    initialize!(u, Qu, ic)
    marks(dict) = (g1, g2, g3, level) -> (dict[g1] = max(get(dict, g1, 0), level); nothing)
    tiled = Dict{Int,Int8}()
    for li in getfield(s, :levels)[2].patches
        p = s.patches[li]
        CL._tag_sweep!(marks(tiled), s, p, states.patches[li], zeros(Int8, size(p.amr.covered)),
                       false)
    end
    uniform = Dict{Int,Int8}()
    CL._tag_sweep!(marks(uniform), u, Qu.patches[1])
    # Level-1 nodes 73 .. 169 are the tiles'; two nodes in from each end the
    # δ⁴ taps stay off the imposed shell. The shared faces are 97, 121, 145.
    inner = 75:167
    @test count(g -> get(uniform, g, 0) == CL.TAG_MARK, inner) > 4
    @test all(g -> get(tiled, g, 0) == get(uniform, g, 0), inner)
end

@testset "max_levels: configuration and levels created on demand" begin
    wall2 = (SlipWallBC(), SlipWallBC())
    per = (PeriodicBC(), PeriodicBC())
    mk(; max_levels=nothing, regrid_interval=0, tile=0, refine=nothing, rebalance=0,
         subcycle=false) = build_solver(
        Problem(domain=((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)), bcs=(wall2, per, per),
                ic=nothing),
        Numerics(n_global=(81, 1, 1),
                 amr=AMR(initial=refine, subcycle=subcycle, regrid_interval=regrid_interval,
                         tile=tile, rebalance=rebalance, max_levels=max_levels)))
    r1 = BlockRegion((20, 0, 0), (41, 1, 1))
    r2 = BlockRegion((80, 0, 0), (30, 1, 1))
    # A level refine does not give starts empty, which needs tiles and a
    # regrid; more than one regridded level needs tiles and no rebalancing;
    # max_levels cannot drop a level refine gives.
    @test_throws ErrorException mk(max_levels=3, regrid_interval=4)
    @test_throws ErrorException mk(max_levels=3, tile=8)
    @test_throws ErrorException mk(refine=[r1, r2], regrid_interval=4)
    @test_throws ErrorException mk(max_levels=3, tile=8, regrid_interval=4,
                                   rebalance=1.5)
    @test_throws ErrorException mk(refine=[r1, r2], max_levels=2)
    s = mk(max_levels=3, tile=8, regrid_interval=4, subcycle=true)
    @test nlevels(s) == 3 && npatches(s) == 1
    @test all(isempty(level_regions(s, ℓ)) for ℓ in 1:2)
    @test length(getfield(s, :regrid).created) == 2
    s = mk(refine=r1, max_levels=3, tile=8, regrid_interval=4)
    @test length(level_regions(s, 1)) == 6 && isempty(level_regions(s, 2))
    prob = Problem(eos=IdealSpecies("gas"; gamma=1.4, R=1.0),
                   domain=((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)),
                   bcs=(wall2, per, per),
                   ic=(x, y, z) -> Prim(rho=1.0, u=(0.0, 0.0, 0.0), p=1.0))
    @test_throws ArgumentError setup(prob, Numerics(n_global=(81, 1, 1),
                                                    amr=AMR(max_levels=3)))
end

@testset "three regridded levels follow a shock through an interface" begin
    # A shock driven from the left into light gas crosses a contact into
    # heavy gas at x = 0.5. The AMR frontend builds three levels from the
    # tags of the initial state, every level regrids, and the finest follows
    # the leading shock. The composite is compared with a uniform run at the
    # finest spacing and with the root alone.
    wall2 = (SlipWallBC(), SlipWallBC())
    per = (PeriodicBC(), PeriodicBC())
    ic(x, y, z) = x < 0.25 ? Prim(u=(0, 0, 0), p=1.0, rho=1.0) :
                  x < 0.5  ? Prim(u=(0, 0, 0), p=0.1, rho=0.125) :
                             Prim(u=(0, 0, 0), p=0.1, rho=0.5)
    prob = Problem(eos=IdealSpecies("gas"; gamma=1.4, R=1.0),
                   domain=((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)),
                   bcs=(wall2, per, per), ic=ic)
    N = 81
    # A discontinuity inside the refined levels at t = 0 needs a CFL of 0.2
    # under subcycling; the references take the same.
    num(n; amr=nothing) = Numerics(n_global=(n, 1, 1), cfl=0.2, amr=amr)
    amr = AMR(initial=:sensor, tile=8, regrid_interval=4, subcycle=true,
              max_levels=3)
    s, states = setup(prob, num(N; amr))
    @test nlevels(s) == 3
    initial2 = level_regions(s, 2)
    @test !isempty(level_regions(s, 1)) && !isempty(initial2)
    # Level 2 lies in level 1's tiles with the nesting margin.
    parents = [CL._erode(CL._fine_region(lt.region, (true, false, false)), lt.imposed,
                         (true, false, false))
               for lt in getfield(s, :levels)[2].transfers]
    @test all(r -> CL._covered_by(CL._buffered(r, (true, false, false), 4), parents),
              initial2)
    # The leading shock is the last root node above the ahead pressure by
    # half the jump; it lies in a level-2 tile at every sampled time.
    function shock_in_finest(s, states)
        _, p = line_sample(s, states, :p)
        i = findlast(>(0.2), p)
        g = 3 * (i - 1) + 1
        return any(r -> r.offset[1] < g <= r.offset[1] + r.extent[1],
                   level_regions(s, 2))
    end
    dir = mktempdir()
    for t in (0.05, 0.1, 0.15)
        run!(s, states; tfinal=t, nmax=10000)
        @test shock_in_finest(s, states)
    end
    save_checkpoint(s, states, joinpath(dir, "deep"))
    # The load rebuilds the refined tiles' ghost rings as the post-step
    # synchronization left them, so the whole padded array matches: the
    # first regrid check after a restart tags on those rings.
    loaded_solver, loaded = setup(prob, num(N; amr))
    load_checkpoint!(loaded_solver, loaded, joinpath(dir, "deep"))
    @test length(loaded.patches) == length(states.patches) &&
          all(parent(loaded.patches[i]) == parent(states.patches[i]) for i in eachindex(states.patches))
    run!(s, states; tfinal=0.25, nmax=10000)
    @test shock_in_finest(s, states)
    regs2 = level_regions(s, 2)
    @test maximum(r -> r.offset[1], regs2) > maximum(r -> r.offset[1], initial2)
    @test length(states.patches) == length(s.patches)
    @test all(all(isfinite, parent(Q)) for Q in states.patches)
    _, rho = line_sample(s, states, :rho)
    r, Qr = setup(prob, num(N))
    run!(r, Qr; tfinal=0.25, nmax=10000)
    _, rho_root = line_sample(r, Qr, :rho)
    f, Qf = setup(prob, num(9 * (N - 1) + 1))
    run!(f, Qf; tfinal=0.25, nmax=100000)
    _, rho_fine = line_sample(f, Qf, :rho)
    rho_fine = rho_fine[1:9:end]
    e_composite = sum(abs, rho .- rho_fine) / N
    e_root = sum(abs, rho_root .- rho_fine) / N
    # Measured: 2.2e-4 against 2.36e-2 for the root alone.
    @info "three-level shock and interface" e_composite e_root
    @test e_composite < 3e-3
    @test e_root > 8 * e_composite
    # The checkpoint taken while level 2 exists continues bit for bit.
    rs_solver, rs = setup(prob, num(N; amr))
    load_checkpoint!(rs_solver, rs, joinpath(dir, "deep"))
    run!(rs_solver, rs; tfinal=0.25, nmax=10000)
    @test rs_solver.t == s.t && rs_solver.step == s.step
    @test all(ℓ -> level_regions(rs_solver, ℓ) == level_regions(s, ℓ), 1:2)
    spec, rspec = getfield(s, :regrid), getfield(rs_solver, :regrid)
    @test rspec.created == spec.created
    same(i) = (inner = CL.interior(s.patches[i].decomp);
               parent(rs.patches[i])[inner, :] == parent(states.patches[i])[inner, :])
    @test length(rs.patches) == length(states.patches) && all(same, eachindex(states.patches))
    rm(dir; recursive=true)
end

@testset "the pre-step restriction runs only after the state was written" begin
    per = (PeriodicBC(), PeriodicBC())
    wall2 = (SlipWallBC(), SlipWallBC())
    # Four tiles meeting at a corner, walls along y, and a filter pass on
    # every other step, so both of the step's endings reach the restriction.
    function nest(; subcycle=false)
        s = build_solver(
            Problem(domain=((0.0, 2π), (0.0, 2π), (0.0, 1.0)), bcs=(per, wall2, per),
                    ic=nothing),
            Numerics(n_global=(48, 48, 1), filter=StateFilter(interval=2),
                     amr=AMR(initial=BlockRegion((18, 18, 0), (12, 12, 1)),
                             regrid_interval=0, subcycle=subcycle, tile=6)))
        q = allocate_state(s)
        initialize!(s, q, (x, y, z) -> Prim(u=(0.4, 0.3 * sin(x) * sin(y), 0),
                                            p=1.0, rho=1.0 + 0.1 * sin(x) * cos(y)))
        return s, q
    end
    sod_ic(x, y, z) = x < 0.5 ? Prim(u=(0, 0, 0), p=1.0, rho=1.0) :
                                Prim(u=(0, 0, 0), p=0.1, rho=0.125)
    function sod(; N=81,subcycle=false, regrid_interval=0, refine=nothing,
                        max_levels=nothing)
        s = build_solver(
            Problem(domain=((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)), bcs=(wall2, per, per),
                    ic=nothing),
            Numerics(n_global=(N, 1, 1), cfl=0.2,
                     amr=AMR(initial=refine, subcycle=subcycle,
                             regrid_interval=regrid_interval, tile=8, max_levels=max_levels)))
        q = allocate_state(s)
        initialize!(s, q, sod_ic)
        return s, q
    end
    function counted(s, q; kw...)
        n0 = CL.RESTRICTION_COUNT[]
        run!(s, q; tfinal=1.0, kw...)
        return CL.RESTRICTION_COUNT[] - n0
    end
    # A callback that writes nothing. `run!` cannot know that, and takes the
    # state as written.
    written = (_, _) -> nothing
    let
        # Five steps of a continuing call: one restriction per step after the
        # step, and one before the first step, where the state may have been
        # written since the previous call. Before, every step restricted twice.
        s, q = nest()
        run!(s, q; tfinal=1.0, nmax=2)
        @test counted(s, q; nmax=s.step + 5) == 6
        # A bare callback runs every step and may write the state, so the
        # run restricts before every step, as it did before the pre-step
        # restriction became conditional.
        @test counted(s, q; nmax=s.step + 5, callback=written) == 10
        @test counted(s, q; nmax=s.step + 3, callback=written) == 6
    end
    # Each case run under both schedules, compared bit for bit, halos
    # included: as it is, and with `written` beside its own callback, which
    # restores the restriction before every step.
    function same_both_ways(build, runner)
        out = map((written, nothing)) do extra
            s, q = build()
            runner(s, q, extra)
            (s.t, s.step, s.cfl, s.floor_tally.cells, [copy(parent(Q)) for Q in q.patches])
        end
        return isequal(out[1], out[2]), out[2]
    end
    beside(callback, extra) = extra === nothing ? callback :
                              callback === nothing ? extra : (callback, extra)
    steps(n; callback=nothing, kw...) = (s, q, extra) ->
        run!(s, q; tfinal=1.0, nmax=n, callback=beside(callback, extra), kw...)
    # A callback writing the refined state every third step, and one spoiling
    # a root density once at step 5, which rolls back to step 4.
    bump = Callback(EveryStep(3), (s, q) -> (foreach(Q -> parent(Q) .*= 1 + 1e-6,
                                                     q.patches[2:end]); false))
    spoil(done) = Callback(EveryStep(), (s, q) -> (s.step == 5 && !done[] &&
        (done[] = true; q.patches[1][padded_index(first(CL.eachpatch(s, q))[1], 3, 3, 1), 1] = -1.0);
        false))
    cases = [
        "static" => (nest, steps(8)),
        "subcycled" => (() -> nest(subcycle=true), steps(6)),
        "callback" => (nest, steps(8; callback=bump)),
        "rollback" => (nest, (s, q, extra) ->
            run!(s, q; tfinal=1.0, nmax=7, callback=beside(spoil(Ref(false)), extra),
                 control=StepControl(retries=1, savepoint_interval=2))),
        "tiled regrid" => (() -> sod(N=201, subcycle=true, regrid_interval=5,
                                     refine=BlockRegion((85, 0, 0), (31, 1, 1))),
                           steps(20)),
        "three regridded levels" => (() -> sod(refine=BlockRegion((20, 0, 0), (41, 1, 1)),
                                               max_levels=3, regrid_interval=4),
                                     steps(16)),
        "positivity failsafe" => (() -> sod(refine=BlockRegion((30, 0, 0), (21, 1, 1))),
                                  steps(20; control=StepControl(floor_ratio=0.999))),
    ]
    for (name, (build, runner)) in cases
        same, (_, _, cfl, repaired, _) = same_both_ways(build, runner)
        @test same
        name == "rollback" && @test cfl < 0.5
        name == "positivity failsafe" && @test repaired > 0
        same || @info "pre-step restriction schedules differ" name
    end
end
