# Launch planner: what rank count should this grid run at, and what does the
# launch line look like?
#
# The sizing rules are not obvious and they interact — the compact filter sets a
# floor on how far a dimension can be split, `@threaded` sets a floor on when a
# thread is worth asking for, and the halo pad means over-decomposing costs real
# arithmetic rather than just communication. This enumerates the candidates and
# says which are legal, rather than leaving each of those to be rediscovered.
#
# Runs serially and launches nothing; it is a planning tool.
#
#   julia --project=. clusterlaunch.jl 128
#   julia --project=. clusterlaunch.jl 256,256,512
#   julia --project=. clusterlaunch.jl 256 nodes=36 cores_per_node=112
#
# Positional grid, written as N or NX,NY,NZ, then `key=value` options — parsed by
# `script_args` (src/scriptargs.jl), which explains why these are arguments and
# not environment variables:
#
#   nodes            nodes available          (default 1)
#   cores_per_node   physical cores per node  (default: this machine's count)
#   ranks            explicit rank counts to consider, comma-separated
#   scheme           c6 (default) or c10
#   filter           true (default) to include the filter's floor, false to drop it

using MPI
using Printf
using CompactLES
const CL = CompactLES

MPI.Initialized() || MPI.Init()

const opt = script_args(ARGS, (grid = "64", nodes = 1,
                               cores_per_node = Sys.CPU_THREADS ÷ 2,
                               ranks = "", scheme = "c6", filter = true);
                        positional = (:grid,))

const grid = script_grid(opt.grid)
const nodes = opt.nodes
const cores_per_node = opt.cores_per_node
const total_cores = nodes * cores_per_node
const use_filter = opt.filter
# Rejected rather than silently treated as c6: the old ENV form fell through to
# the default on any typo, and a mis-sized C10 run is the kind of thing you find
# out about from `plan_direction` an allocation later.
const deriv = opt.scheme == "c10" ? lele_d1_10() :
              opt.scheme == "c6" ? lele_d1_6() :
              error("scheme must be c6 or c10, got '$(opt.scheme)'")
const n_halo = 4

# Minimum local extent per dimension, from the schemes themselves rather than a
# remembered number. This mirrors the guard in `plan_direction`
# (src/operators.jl, src/operators_banded.jl) — if that expression changes, this
# is the other place that has to move.
scheme_min(s) = max(2 * CL.nclosure(s) + 1, 2 * CL.halfwidth(s) + 1,
                    2 * (hasproperty(s, :q) ? s.q : 0) + 1)
const nmin = use_filter ? max(scheme_min(deriv), scheme_min(compact_filter())) :
             scheme_min(deriv)

active(d) = grid[d] > 1

"""Process grid MPI would choose for `np` ranks over the active dimensions."""
function process_grid(np)
    nact = count(active, 1:3)
    nact == 0 && return (1, 1, 1)
    adims = Int.(MPI.Dims_create(np, zeros(Cint, nact)))
    return ntuple(d -> active(d) ? popfirst!(adims) : 1, 3)
end

"""Smallest block any rank owns, which is the one the scheme floors apply to."""
min_local(np) = (dims = process_grid(np); ntuple(d -> grid[d] ÷ dims[d], 3))

function evaluate(np)
    dims = process_grid(np)
    nl = min_local(np)
    legal = all(d -> !active(d) || nl[d] >= nmin, 1:3)
    interior = prod(nl)
    # Padded volume divided by interior: the arithmetic a rank does per useful
    # point. Communication is the usual worry when over-decomposing, but with a
    # 4-deep halo on every active face the redundant *compute* grows faster.
    padded = prod(ntuple(d -> active(d) ? nl[d] + 2 * n_halo : nl[d], 3))
    return (np = np, dims = dims, n_local = nl, legal = legal,
            pts = interior, overhead = padded / interior,
            threads = interior >= CL.THREAD_MIN_WORK[])
end

candidates = if !isempty(opt.ranks)
    parse.(Int, split(opt.ranks, ','))
else
    # Every divisor of the core budget, so a suggestion is always something the
    # allocation can actually be filled with.
    sort([r for r in 1:total_cores if total_cores % r == 0])
end

println("grid            : ", grid)
println("budget          : ", nodes, " node(s) x ", cores_per_node,
        " cores = ", total_cores)
println("scheme          : ", deriv.name,
        use_filter ? " + $(compact_filter().name)" : " (filter floor ignored)")
println("min local extent: ", nmin, " points per rank per active dimension")
println("THREAD_MIN_WORK : ", CL.THREAD_MIN_WORK[])
println()
println("  ranks   process grid    min n_local      pts/rank  halo mult  threads?")

viable = eltype([evaluate(1)])[]
for np in candidates
    e = evaluate(np)
    e.legal || continue
    push!(viable, e)
    Printf.@printf("  %5d   %-14s  %-14s  %8d   %6.2fx   %s\n",
                   e.np, string(e.dims), string(e.n_local), e.pts, e.overhead,
                   e.threads ? "yes" : "no")
end

if isempty(viable)
    println("  (none: even 1 rank cannot hold ", nmin,
            " points per active dimension)")
    exit(1)
end

# Two picks, because they answer different questions and only the caller knows
# which one is being asked. Efficiency is the last rank count whose redundant
# halo arithmetic is still in hand; the cutoff is a judgement, not a measured
# optimum — past 2x padded-to-interior a rank does more halo than interior, and
# strong scaling has stopped being real. Throughput is simply the widest legal
# count, which is what you want when wall time is the constraint and you have
# already accepted paying for it.
efficient = last(filter(e -> e.overhead <= 2.0, viable))
widest = last(viable)

println()
if efficient.np == widest.np
    println("recommended     : ", widest.np, " ranks, dims ", widest.dims,
            ", ", widest.pts, " pts/rank, halo ",
            round(widest.overhead; digits = 2), "x")
else
    println("best efficiency : ", efficient.np, " ranks, dims ", efficient.dims,
            ", ", efficient.pts, " pts/rank, halo ",
            round(efficient.overhead; digits = 2), "x")
    println("widest legal    : ", widest.np, " ranks, dims ", widest.dims,
            ", ", widest.pts, " pts/rank, halo ",
            round(widest.overhead; digits = 2), "x")
    println("                  ", widest.np, " ranks buys wall time at ",
            round(widest.overhead / efficient.overhead; digits = 2),
            "x the arithmetic per useful point.")
    widest.overhead > 3 &&
        println("                  Above 3x this grid is too small for the ",
                "allocation; raise the grid.")
end
best = widest
println("threads         : -t 1", best.threads ? "" :
        "   (@threaded is inert below THREAD_MIN_WORK -- extra threads idle)")
if best.np > cores_per_node
    println("spans           : ", cld(best.np, cores_per_node),
            " nodes -- confirm off-node cost BEFORE trusting this count.")
    println("                  Run clusterprobe.jl at this width and check that")
    println("                  `MPI binary` is the system MPI, not a _jll. A")
    println("                  bundled JLL can work perfectly on one node and")
    println("                  fall off a cliff on two.")
end
println()
println("srun -n ", best.np, " --cpu-bind=threads julia -t 1 --project=. <script>")
println()
println("Verify the launch before trusting a timing from it:")
# Carrying the grid through matters: the probe's own default is 64³, and its
# scheme-floor check is only about the run you are planning if it is given the
# same grid this just sized.
println("  srun -n ", best.np,
        " --cpu-bind=threads julia --project=. clusterprobe.jl ", opt.grid)
println("Check `SMT sibling in a rank's own mask: false` -- see the cluster")
println("section of CLAUDE.md for why that one matters more than it looks.")

MPI.Finalize()
