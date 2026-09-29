# Weak scaling of a tiled refined level's coupling (reference/AMR_GPU.md,
# ownership and load balance): a planar refined slab, the shape a mixing
# layer or a plane shock takes, spans the transverse extent of a root grid
# that grows with the rank count, so the tiles per rank and the root block
# per rank stay fixed. A coupling whose per-rank cost follows the tiles a rank
# holds prints flat columns; one that replicates every tile's data on every
# rank grows with np.
#
# Columns, each the largest over ranks: the level's memory on a rank
# (`Base.summarysize` of the `Level`, which holds the transfers, the
# same-level records and the coupling's exchange plan, not the tiles'
# patches), the wall of one `sync_levels!` (the restriction and the shell
# imposition of every tile, median of `reps` after a warm-up), and the wall
# of one global step. Setup is timed once and includes compilation on the
# first configuration only if run at np = 1 first.
#
# Run: mpiexec -n 4 julia --project=. -t 1 bench/amr_scaling.jl \
#          [N=96] [tile=8] [dim=2] [steps=3] [reps=20]
#
# `N` is the root nodes per rank along the slab (the transverse extent of a
# 2-D run is N·np; a 3-D run is N·np by 32), `dim` 2 or 3. Workstation
# timings carry the 10–20% spread and the 2-D np = 8 pathology of
# reference/AMR_GPU.md; read the memory column and the growth of the
# coupling wall, not the third digit.

using MPI
MPI.Init(threadlevel=:funneled)
using CompactLES
using Printf
using Statistics
const CL = CompactLES

function main()
    args = CompactLES.script_args(ARGS, (N=96, tile=8, dim=2, steps=3, reps=20);
                                  positional=(:N, :tile, :dim, :steps, :reps))
    comm = MPI.COMM_WORLD
    np = MPI.Comm_size(comm)
    rank = MPI.Comm_rank(comm)
    per = (PeriodicBC(), PeriodicBC())
    Nx = 64
    Ny = args.N * np
    Nz = args.dim == 3 ? 32 : 1
    margin = 8
    thick = 2 * args.tile
    x0 = Nx ÷ 2 - thick ÷ 2
    region = BlockRegion((x0, margin, args.dim == 3 ? margin : 0),
                         (thick, Ny - 2margin, args.dim == 3 ? Nz - 2margin : 1))
    L = (1.0, Ny / Nx, args.dim == 3 ? Nz / Nx : 1.0)
    ic(x, y, z) = Prim(u=(0.1, 0, 0), p=1.0,
                       rho=1.0 + 0.5 * tanh((x - 0.5) / 0.05) +
                           0.01 * sin(2π * y / L[2]))
    # The unrefined root alone first: its step wall is the reference the
    # refined step's growth with np is read against.
    root = Solver(n_global=(Nx, Ny, Nz), L_domain=L, bcs=(per, per, per), cfl=0.4)
    Q = allocate_state(root)
    initialize!(root, Q, ic)
    run!(root, Q; tfinal=1.0, nmax=1)
    MPI.Barrier(comm)
    t_root = @elapsed run!(root, Q; tfinal=1.0, nmax=1 + args.steps)
    t_setup = @elapsed begin
        solver = Solver(n_global=(Nx, Ny, Nz), L_domain=L, bcs=(per, per, per),
                        cfl=0.4, refine=region, tile=args.tile)
        states = allocate_state(solver)
        initialize!(solver, states, ic)
    end
    lev = getfield(solver, :levels)[2]
    ntiles = length(lev.owners)
    held = length(lev.patches)
    mem = Base.summarysize(lev)
    # One call first, so the timed ones exclude compilation.
    CL.sync_levels!(solver, states)
    walls = Float64[]
    for _ in 1:args.reps
        MPI.Barrier(comm)
        push!(walls, @elapsed CL.sync_levels!(solver, states))
    end
    wsync = median(walls)
    workspace = CL.Workspace(states)
    run!(solver, states, workspace; tfinal=1.0, nmax=1)
    MPI.Barrier(comm)
    t_run = @elapsed run!(solver, states, workspace; tfinal=1.0, nmax=1 + args.steps)
    finite = all(all(isfinite, parent(Q)) for Q in states)
    red(x, op) = MPI.Allreduce(x, op, comm)
    mem_max = red(mem, max)
    mem_min = red(mem, min)
    held_max = red(held, max)
    wsync_max = red(wsync, max)
    wstep_max = red(t_run / args.steps, max)
    wroot_max = red(t_root / args.steps, max)
    setup_max = red(t_setup, max)
    finite = red(Int(finite), min) == 1
    if rank == 0
        @printf("np %3d  %d-D root %dx%dx%d  tile %d: %d tiles, ≤%d per rank\n",
                np, args.dim, Nx, Ny, Nz, args.tile, ntiles, held_max)
        @printf("  level memory per rank  %.2f MB (min %.2f)\n",
                mem_max / 1e6, mem_min / 1e6)
        @printf("  sync_levels! wall      %.2f ms\n", 1e3 * wsync_max)
        @printf("  step wall              %.1f ms (root alone %.1f ms)\n",
                1e3 * wstep_max, 1e3 * wroot_max)
        @printf("  setup %.1f s   finite %s\n", setup_max, finite)
    end
end

mpi_main(main)
