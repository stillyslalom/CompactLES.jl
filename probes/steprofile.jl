# Where does one step's time actually go?
#
# `bench/phases.jl` replays compute_rhs! phase by phase, but it is fixed at 64^3
# with twenty reps per phase and is unusable on a machine where a single step
# already costs seconds. This takes a sampling profile of a few steps at a grid
# you choose and prints the flat table, which is enough to separate the two
# cases that matter when a step is unexplainably slow:
#
#   flat, spread over many CompactLES frames -- bandwidth-bound as usual, and
#     the slowdown is in code generation or memory rather than in one call;
#   piled up in one place -- an MPI entry point, a solve, a single kernel --
#     which names the cause outright.
#
# The file paths in the flat output are the useful column: frames in libmpi or
# in a wait loop read very differently from frames in src/.
#
#   julia --project=. probes/steprofile.jl
#   julia --project=. probes/steprofile.jl grid=48 steps=5 mincount=100
#
# Options are parsed by `script_args` (src/scriptargs.jl): grid (cube edge),
# steps (profiled steps, after one warm-up step that is not profiled), mincount
# (frames with fewer samples are not printed), art (artificial properties on),
# delay (sampling interval in seconds).

using MPI
MPI.Init(threadlevel=:funneled)
using CompactLES
using Profile, Printf

const CL = CompactLES
const per3 = ntuple(_ -> (PeriodicBC(), PeriodicBC()), 3)

opt = CL.script_args(ARGS, (grid = 32, steps = 3, mincount = 50, art = false,
                            delay = 0.002); positional = (:grid,))

solver = Solver(n_global = (opt.grid, opt.grid, opt.grid),
                L_domain = (2π, 2π, 2π), bcs = per3,
                art = ArtParams(enabled = opt.art))
Q = allocate_state(solver)
initialize!(solver, Q, (x, y, z) -> Prim(u = (0.1sin(x), 0, 0), p = 1.0,
                                         rho = 1.0))

# One step first, so the profile carries no compilation.
warm = @elapsed run!(solver, Q; tfinal = 1e9, nmax = 1)
@printf("warm-up step: %.3f s at %d^3 (%d points)\n", warm, opt.grid,
        opt.grid^3)

# A step here can be seconds, so the buffer has to hold many samples; the
# default is sized for a profile of milliseconds.
Profile.init(n = 10^7, delay = opt.delay)
Profile.clear()
target = solver.step + opt.steps
profiled = @elapsed (Profile.@profile run!(solver, Q; tfinal = 1e9,
                                           nmax = target))
@printf("profiled %d steps in %.3f s (%.3f s/step)\n\n", opt.steps, profiled,
        profiled / opt.steps)

# The flat table counts a frame in every sample it appears in, so `run!` and
# `step!` sit at the top of it by construction. Self time is the leaf of each
# backtrace, which is the column that says where the work is.
function leaf_counts()
    data = try
        Profile.fetch(include_meta = false)
    catch
        Profile.fetch()
    end
    lidict = Profile.getdict(data)
    counts = Dict{String,Int}()
    total = 0
    leaf = true
    for ip in data
        if ip == 0
            leaf = true
            continue
        end
        if leaf
            frames = get(lidict, ip, nothing)
            label = "unknown"
            if frames !== nothing && !isempty(frames)
                f = first(frames)
                label = string(basename(string(f.file)), ":", f.line, "  ",
                               f.func)
            end
            counts[label] = get(counts, label, 0) + 1
            total += 1
            leaf = false
        end
    end
    return counts, total
end

counts, total = leaf_counts()
if total > 0
    println("self time by leaf frame (", total, " samples):")
    for (label, n) in first(sort(collect(counts); by = last, rev = true), 25)
        @printf("  %5.1f%%  %6d  %s\n", 100n / total, n, label)
    end
    println()
end

println("inclusive frames (every sample the frame appears in):")
Profile.print(format = :flat, sortedby = :count, mincount = opt.mincount)

println()
println("Read the file column, not just the function names. Samples resting in")
println("libmpi or a polling loop mean the cost is not the arithmetic; samples")
println("spread thinly across src/ frames mean it is.")
