# Per-rank load under a tiled regrid, and what a rebalance does to it
# (reference/AMR_GPU.md, ownership and load balance). A 1-D Sod shock
# crosses a tiled refined level whose tile set moves under stored
# ownership. At every regrid check the script Allgathers each rank's busy
# time over the interval, its step wall less its time inside the run-wide
# collectives, the quantity `_rebalance_due!` balances on, and prints the
# ratio of the largest to the mean with the tile owner ranges. Two runs:
# rebalancing off, where a fresh tile joins the nearest survivor's group
# once no rank is free and the groups drift apart in tile count; and on at
# a threshold of one, which any measured spread exceeds, so the level is
# repartitioned on its measured weights at every check that persists.
#
# Run: mpiexec -n 4 julia --project=. -t 1 bench/amr_balance.jl \
#          [N=800] [interval=10] [steps=200] [persist=1]
#
# Mechanics only. The workstation figures say whether the measurement and
# the repartition behave, not what a rebalance is worth: that is a cluster
# measurement (reference/CLUSTER.md), where per-rank costs differ from the
# desktop's by 27–66x and move with rank placement, and the run-to-run
# spread of any timing here is 10–20%. Observed at np = 4, N = 800: off,
# the stored groups drift to two tiles on ranks 0–1 against four on 2–3
# and the per-check max/mean climbs to 1.3–1.5; on at threshold 1 with
# persist 1, every check repartitions, one tile per rank where the count
# allows, and the partition then follows the timing noise from check to
# check (1.0–1.7), which is what the threshold and `rebalance_persist`
# exist to damp. The last column is the transient memory of the tiles a
# check moved: the replica the gather would have held on every rank
# against the largest block a rank received through `_migrate_tile!`.
# Here a tile is 25 fine nodes of five components, so a moved tile is
# 1.0 KiB replicated against 0.9 KiB migrated (its 23-node interior on
# one rank), a figure that says only that the migration moves the interior
# and nothing else; the case where the replica is the regrid's cost is the
# implosion shell at a cluster rank count (reference/AMR_GPU.md).

using MPI
MPI.Init(threadlevel=:funneled)
using CompactLES
using Printf
const CL = CompactLES

function track(args, rebalance; quiet=false)
    comm = MPI.COMM_WORLD
    np = MPI.Comm_size(comm)
    rank = MPI.Comm_rank(comm)
    N = args.N
    wall2 = (SlipWallBC(), SlipWallBC())
    per = (PeriodicBC(), PeriodicBC())
    x0 = 0.45
    node0 = round(Int, x0 * (N - 1))
    solver = Solver(n_global=(N, 1, 1), L_domain=(1.0, 1.0, 1.0),
                    bcs=(wall2, per, per), cfl=0.2, subcycle=true,
                    regrid_interval=args.interval, tag_buffer=2, tile=8,
                    refine=BlockRegion((8 * (node0 ÷ 8) - 8, 0, 0), (16, 1, 1)),
                    rebalance=rebalance, rebalance_persist=args.persist)
    states = allocate_state(solver)
    initialize!(solver, states, (x, y, z) -> x < x0 ?
        Prim(u=(0, 0, 0), p=1.0, rho=1.0) : Prim(u=(0, 0, 0), p=0.1, rho=0.125))
    mark_wall = Ref(0.0)
    mark_wait = Ref(0.0)
    total = Ref(0.0)
    n_cons = solver.equations.n_cons
    prev = Dict{BlockRegion,UnitRange{Int}}()
    rank == 0 && !quiet && @printf("%6s %11s %9s %6s  %-24s %s\n",
                                   "step", "t", "max/mean", "tiles", "owner ranges",
                                   "moved: replica / migrated KiB per rank")
    function report(s, _)
        busy = (s.wall_total - mark_wall[]) - (s.wait_total - mark_wait[])
        mark_wall[] = s.wall_total
        mark_wait[] = s.wait_total
        total[] += busy
        walls = MPI.Allgather(busy, comm)
        ratio = maximum(walls) / (sum(walls) / np)
        lev = getfield(s, :levels)[2]
        ranges = join(("$(first(o))-$(last(o))" for o in lev.owners), " ")
        # Transient memory of the tiles a regrid moved since the last
        # report: the replica the gather would have held on every rank
        # against the block this rank received, at most over ranks.
        replica = 0
        migrated = 0
        for (lt, o) in zip(lev.transfers, lev.owners)
            (haskey(prev, lt.region) && prev[lt.region] != o) || continue
            Nf = CL.fine_extent(lt.region, lt.active)
            replica += prod(Nf) * n_cons * sizeof(Float64)
            b = lt.fine_blocks[rank + 1]
            got = ntuple(d -> lt.active[d] ?
                         length(intersect(b.offset[d]+1:b.offset[d]+b.extent[d],
                                          2:Nf[d]-1)) : 1, 3)
            migrated += prod(got) * n_cons * sizeof(Float64)
        end
        empty!(prev)
        for (lt, o) in zip(lev.transfers, lev.owners)
            prev[lt.region] = o
        end
        migrated = MPI.Allreduce(migrated, max, comm)
        if rank == 0 && !quiet
            moved = replica == 0 ? "" :
                    @sprintf("%.1f / %.1f", replica / 1024, migrated / 1024)
            @printf("%6d %11.4e %9.3f %6d  %-24s %s\n", s.step, s.t, ratio,
                    length(lev.owners), ranges, moved)
        end
        return false
    end
    run!(solver, states; tfinal=1.0, nmax=args.steps,
         callback=Callback(EveryStep(args.interval), report))
    totals = MPI.Allgather(total[], comm)
    ratio = maximum(totals) / (sum(totals) / np)
    return ratio, MPI.Allreduce(solver.wall_total, max, comm)
end

function main()
    args = script_args(ARGS, (N=800, interval=10, steps=200, persist=1);
                       positional=(:N, :interval, :steps, :persist))
    rank = MPI.Comm_rank(MPI.COMM_WORLD)
    rank == 0 && println("np = $(MPI.Comm_size(MPI.COMM_WORLD)), N = $(args.N), ",
                         "regrid every $(args.interval) steps, $(args.steps) steps")
    # A warm-up run pays the compilation, so the two timed runs compare.
    track(merge(args, (steps=2 * args.interval,)), 1.0; quiet=true)
    rank == 0 && println("\n--- rebalance off (stored ownership only)")
    r_off, w_off = track(args, 0.0)
    rank == 0 && println("\n--- rebalance on (threshold 1, persist $(args.persist))")
    r_on, w_on = track(args, 1.0)
    if rank == 0
        @printf("\nwhole run, max/mean busy time: off %.3f, on %.3f\n", r_off, r_on)
        @printf("wall of the slowest rank: off %.2f s, on %.2f s\n", w_off, w_on)
    end
end

mpi_main(main)
