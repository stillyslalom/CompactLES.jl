# Tiled-level cover of an annular tag set (reference/AMR_GPU.md, tiles and
# adjacency): the lattice cover of a converging-shell-shaped feature against
# its bounding box, the number the tiled level exists to reduce. A 2-D
# Cartesian root grid carries a thin annular density bump; one regrid
# replaces the initial cover of the whole annulus box by the tiles that
# meet the tagged (buffered) ring, and the script reports the cover ratio,
# the per-tile memory, and the wall time of a few subcycled steps.
#
# Run: julia --project=. bench/amr_tiles.jl [N=192] [tile=6] [steps=3]
#
# Measured on the workstation (N = 192, r0 = 0.75, w = 0.02, buffer 1):
# tile 6 covers 41% of the bounding box with 208 tiles of 19² fine nodes
# (0.40 MB each); tile 12 covers 47% with 80 tiles of 37² (1.1 MB each).
# Setup costs 0.06–0.12 s per tile in plan construction (N = 96: 64 tiles
# in 7.6 s, 196 in 11.4 s), which is why tile edges below about 12 are
# impractical in 3-D. At tile 6 the patch set totals 60.3 MB against
# 103.0 MB before the shared RHS workspace (0.207 MB of each tile's
# 0.398 MB is now pooled), and the warm step is 0.55–0.56 s.

using CompactLES
const CL = CompactLES
using MPI
using Printf

function main()
    args = CompactLES.script_args(ARGS, (N=192, tile=6, steps=3); positional=(:N, :tile, :steps))
    N = args.N
    per3 = ntuple(_ -> (PeriodicBC(), PeriodicBC()), 3)
    r0, w = 0.75, 0.02
    ic(x, y, z) = (r = sqrt(x^2 + y^2);
                   Prim(u=(0, 0, 0), p=1.0, rho=1.0 + 0.5 * exp(-((r - r0) / w)^2)))
    margin = 8
    t_setup = @elapsed begin
        solver = Solver(n_global=(N, N, 1), L_domain=(2.0, 2.0, 1.0), bcs=per3,
                        origin=(-1.0, -1.0, 0.0), cfl=0.4, subcycle=true,
                        regrid_interval=1, tag_buffer=1, tag_threshold=0.02,
                        refine=BlockRegion((margin, margin, 0),
                                           (N - 2margin, N - 2margin, 1)),
                        tile=args.tile)
        states = allocate_state(solver)
        initialize!(solver, states, ic)
    end
    n0 = length(level_regions(solver, 1))
    workspace = CL.Workspace(states)
    t_regrid = @elapsed CL.regrid!(solver, states, workspace, nothing)
    regs = level_regions(solver, 1)
    lo = ntuple(d -> minimum(r.offset[d] for r in regs), 2)
    hi = ntuple(d -> maximum(r.offset[d] + r.extent[d] for r in regs), 2)
    box = prod(hi .- lo)
    cover = sum(prod(r.extent[1:2]) for r in regs)
    @printf("tile %d: %d tiles at setup (%.1f s), %d after the regrid (%.1f s)\n",
            args.tile, n0, t_setup, length(regs), t_regrid)
    @printf("cover / bounding box = %d / %d = %.3f\n", cover, box, cover / box)
    # summarysize over the whole vector counts a shared RHS workspace once,
    # so the total is the figure the pooling moves; the per-tile number counts
    # the shared set again for each tile and is an upper bound.
    @printf("patch memory: %.1f MB over %d patches, %.2f MB per tile\n",
            Base.summarysize(solver.patches) / 1e6, length(solver.patches),
            Base.summarysize(solver.patches[2]) / 1e6)
    # One step first, so the timed steps exclude compilation of the
    # multi-tile paths.
    run!(solver, states, workspace; tfinal=1.0, nmax=1)
    t_run = @elapsed run!(solver, states, workspace; tfinal=1.0,
                          nmax=1 + args.steps)
    @printf("%d subcycled steps after warm-up: %.2f s/step, finite %s\n",
            args.steps, t_run / args.steps,
            all(all(isfinite, parent(Q)) for Q in states))
end

main()
