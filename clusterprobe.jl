# Diagnostic: what is this launch actually getting, and does the rank placement
# suit the decomposition?
#
# On a busy production cluster you cannot hold the node fixed, so comparing two
# timings from two different nodes measures the queue as much as the change.
# Placement, by contrast, is deterministic: the rank -> CPU -> NUMA mapping and
# the Cartesian neighbour graph are both properties of the launch, not of which
# node you happened to draw. This script measures those, so a placement question
# can be settled without a single solver step.
#
#   srun -n 56 julia --project=. clusterprobe.jl
#   srun -n 56 --cpu-bind=threads julia --project=. clusterprobe.jl 128
#   srun -n 56 --cpu-bind=threads julia --project=. clusterprobe.jl 256,256,512
#
# The one argument is the grid you actually intend to run, as N or NX,NY,NZ
# (default 64, i.e. 64^3). It decides the decomposition, so the scheme-floor and
# points-per-rank checks below describe a different run if it is left at the
# default. Parsed by `script_args` (src/scriptargs.jl).

const t_start = time()

using MPI
MPI.Init(threadlevel=:funneled)
const t_mpi = time()

using CompactLES
using ThreadPinning
const t_load = time()

# Linux exposes the affinity mask the scheduler handed this process. This is the
# number that matters, not Sys.CPU_THREADS: the whole node is visible to every
# rank regardless of what it was actually given.
#
# The mask SIZE alone is not enough, twice over. Ranks all bound to CPU 0 and
# ranks spread over 56 distinct cores both report a size of 1, so the union over
# ranks is what distinguishes them. And logical CPUs are not cores: with SMT2 a
# mask of {5, 117} is ONE core with both its threads, while {5, 6} is two cores
# with one thread each. Same size, very different machine.
function cpus_allowed()
    try
        for line in eachline("/proc/self/status")
            startswith(line, "Cpus_allowed_list:") || continue
            spec = strip(split(line, ':')[2])
            out = Int[]
            for part in split(spec, ',')
                lo_hi = split(part, '-')
                if length(lo_hi) == 1
                    push!(out, parse(Int, lo_hi[1]))
                else
                    append!(out, parse(Int, lo_hi[1]):parse(Int, lo_hi[2]))
                end
            end
            return out
        end
    catch
    end
    return Int[]
end

# ThreadPinning's getcpuid()/getnumanode() need sched_getcpu, which restricts
# them to Linux and to the calling thread. The mask above already names the
# rank's CPUs portably, so use ThreadPinning only for the static topology —
# socket(i) / numa(i) return cpuid lists and work anywhere.
const TOPO = try
    (sockets = [Set(socket(i)) for i in 1:nsockets()],
     numas   = [Set(numa(i))   for i in 1:nnuma()])
catch
    (sockets = Set{Int}[], numas = Set{Int}[])
end

domain_of(groups, cpu) = something(findfirst(g -> cpu in g, groups), 0)

function physical_cores(mask)
    cores = Set{Int}()
    for c in mask
        rep = c
        try
            s = strip(read("/sys/devices/system/cpu/cpu$c/topology/thread_siblings_list",
                           String))
            siblings = Int[]
            for part in split(s, ',')
                lo_hi = split(part, '-')
                length(lo_hi) == 1 ? push!(siblings, parse(Int, lo_hi[1])) :
                    append!(siblings, parse(Int, lo_hi[1]):parse(Int, lo_hi[2]))
            end
            isempty(siblings) || (rep = minimum(siblings))
        catch
        end
        push!(cores, rep)
    end
    return cores
end

# Slurm caps a step's memory per CPU, and Julia's GC sizes its heap from
# Sys.total_memory(), which reports the whole node and ignores the cgroup. The
# mount layout is not portable, so resolve the path from /proc/self/cgroup and
# walk upward — the limit is often set on an ancestor.
function cgroup_mem_limit()
    v2, v1 = String[], String[]
    try
        for line in eachline("/proc/self/cgroup")
            f = split(line, ':'; limit = 3)
            length(f) == 3 || continue
            if f[2] == ""                                 # cgroup v2: "0::/path"
                push!(v2, "/sys/fs/cgroup" * f[3] * "/memory.max")
            elseif occursin("memory", f[2])               # v1: "N:memory:/path"
                push!(v1, "/sys/fs/cgroup/memory" * f[3] * "/memory.limit_in_bytes")
            end
        end
    catch
    end
    for leaf in vcat(v2, v1)
        dir, base = dirname(leaf), basename(leaf)
        while startswith(dir, "/sys/fs/cgroup")
            try
                s = strip(read(joinpath(dir, base), String))
                s == "max" || return parse(Int, s)
                return typemax(Int)
            catch
            end
            dir = dirname(dir)
        end
    end
    return -1
end

const grid = script_grid(script_args(ARGS, (grid = "64",);
                                     positional = (:grid,)).grid)
decomp = Decomp(grid, (true, true, true))

# Gather over the Cartesian communicator, not COMM_WORLD: Cart_create is free to
# reorder, and decomp.neighbors names ranks in the reordered space. Indexing the
# gathered vector by those ranks only works if the gather agrees with them.
const comm = decomp.comm
const rank = MPI.Comm_rank(comm)
const np   = MPI.Comm_size(comm)

const mask = cpus_allowed()
const cpu  = isempty(mask) ? -1 : minimum(mask)

# Printed and flushed before the gather below, so a hang is localizable rather
# than silent. Reaching this line means MPI.Init and Cart_create/Comm_split have
# all completed on every rank; hanging before it points at the launcher's PMI
# bootstrap, and hanging after it points at the first real data exchange.
if rank == 0
    println("MPI.Init + Cart_create OK: ", np, " ranks, ",
            round(t_mpi - t_start; digits = 2), " s to init")
    flush(stdout)
end

info = (host = gethostname(),
        mask = mask,
        cpus = length(mask),
        ncores = length(physical_cores(mask)),
        socket = domain_of(TOPO.sockets, cpu),
        numa = domain_of(TOPO.numas, cpu),
        onht = try cpu >= 0 && ishyperthread(cpu) catch; false end,
        nthreads = Threads.nthreads(),
        ngcthreads = Threads.ngcthreads(),
        memlimit = cgroup_mem_limit(),
        rss = Sys.maxrss(),
        t_mpi = t_mpi - t_start,
        t_load = t_load - t_mpi,
        coords = decomp.coords,
        neighbors = decomp.neighbors,
        n_local = decomp.n_local,
        pts = prod(decomp.n_local))

all_info = MPI.gather(info, comm; root = 0)

if rank == 0
    hosts = unique(getfield.(all_info, :host))
    ext(f) = (minimum(getfield.(all_info, f)), maximum(getfield.(all_info, f)))
    gb(x) = x < 0 ? "?" : x == typemax(Int) ? "unlimited" :
            string(round(x / 2^30; digits = 2), " GiB")

    println("ranks           : ", np)
    println("nodes           : ", length(hosts), "  ",
            [(h, count(x -> x.host == h, all_info)) for h in hosts])
    println("MPI library     : ", MPI.MPI_LIBRARY, " ", MPI.MPI_LIBRARY_VERSION)
    # Provenance, not just identity. A JLL can satisfy the scheduler's launcher
    # on one node over shared memory and still never reach the interconnect off
    # it, so the name and version alone cannot tell you whether a multi-node run
    # will be fast. `binary` says "system" or names the _jll outright.
    binary = try string(MPI.MPIPreferences.binary) catch; "unknown" end
    println("MPI binary      : ", binary,
            endswith(binary, "_jll") && length(hosts) > 1 ?
            "   <-- BUNDLED JLL ON A MULTI-NODE RUN; see reference/CLUSTER.md" : "")
    println("threadlevel     : ", MPI.Query_thread())
    println("node topology   : ", nsockets(), " sockets, ", nnuma(), " NUMA domains, ",
            ncores(), " cores, ", nsmt(), " threads/core")
    println()

    println("per rank (min..max over ranks):")
    println("  CPUs allowed  : ", ext(:cpus), " logical, ", ext(:ncores), " physical")
    for h in hosts
        on_h = filter(x -> x.host == h, all_info)
        union_h = union(getfield.(on_h, :mask)...)
        overlap = sum(length, getfield.(on_h, :mask)) - length(union_h)
        println("  on ", h, " : ", length(union_h), " logical CPUs for ",
                length(on_h), " ranks",
                overlap > 0 ? "   <-- $overlap CPU(s) DOUBLE-BOOKED" : "")
    end
    println("  SMT sibling in a rank's own mask : ",
            any(x -> x.cpus > x.ncores, all_info))
    println("  ranks sitting on a hyperthread   : ", count(x -> x.onht, all_info),
            " of ", np)
    println("  -t nthreads   : ", ext(:nthreads), ", GC threads ", ext(:ngcthreads))
    println("  cgroup memory : ", gb(minimum(getfield.(all_info, :memlimit))),
            " per rank   (Julia sees Sys.total_memory() = ",
            gb(Int(Sys.total_memory())), ")")
    println("  RSS at probe  : ", gb(maximum(getfield.(all_info, :rss))),
            "  -- the solver adds to this")
    println("  MPI.Init  (s) : ", round.(ext(:t_mpi);  digits = 2))
    println("  pkg load  (s) : ", round.(ext(:t_load); digits = 2))
    println()

    println("decomposition of ", grid, " over ", decomp.dims)
    println("  n_local       : ", ext(:n_local))
    println("  points/rank   : ", ext(:pts))
    # Decomp itself does not enforce the scheme floors -- plan_direction does,
    # later, inside Solver. So the probe can happily report a decomposition that
    # no solver will accept, which is exactly what happened at 224 ranks against
    # the default 64^3 (n_local[1] = 8 against a filter minimum of 9). Check it
    # here, with the floor taken from the schemes rather than a remembered 9.
    scheme_min(s) = max(2 * CompactLES.nclosure(s) + 1,
                        2 * CompactLES.halfwidth(s) + 1,
                        2 * (hasproperty(s, :q) ? s.q : 0) + 1)
    nmin = max(scheme_min(lele_d1_6()), scheme_min(compact_filter()))
    tight = [d for d in 1:3 if decomp.active[d] && minimum(x -> x.n_local[d],
                                                           all_info) < nmin]
    println("  min local extent needed: ", nmin, " (C6 + C8 filter)")
    if isempty(tight)
        println("  all active dimensions clear the scheme floor.")
    else
        println("  ILLEGAL on dim(s) ", tight, " -- plan_direction will error in")
        println("  Solver. Use fewer ranks in those dimensions or a larger grid;")
        println("  pass the grid you actually intend to run as the argument.")
    end
    println("  THREAD_MIN_WORK = ", CompactLES.THREAD_MIN_WORK[],
            " (", CompactLES.THREAD_MIN_WORK_PER_THREAD, " per thread x -t ",
            Threads.nthreads(), ")")
    engages = maximum(getfield.(all_info, :pts)) >= CompactLES.THREAD_MIN_WORK[]
    println("  @threaded engages on the largest per-rank loop: ", engages)
    engages || println("  -> every `-t` thread beyond the first is idle in this run.")
    println()

    # The payload. Every halo exchange and every distributed line solve runs
    # along the Cartesian neighbour graph, so what matters is not how many NUMA
    # domains are in play but whether NEIGHBOURING ranks share one. A placement
    # that round-robins consecutive ranks across sockets puts one whole
    # dimension's traffic on the inter-socket link.
    println("neighbour locality (share of links, per dimension):")
    println("  dim   intra-NUMA  cross-NUMA  cross-socket   off-node")
    for d in 1:3
        same_numa = same_sock = cross_numa = cross_sock = off_node = 0
        for me in all_info
            for nb_rank in me.neighbors[d]
                nb_rank < 0 && continue          # PROC_NULL at an open edge
                nb = all_info[nb_rank + 1]
                if nb.host != me.host
                    off_node += 1
                elseif nb.socket != me.socket
                    cross_sock += 1
                elseif nb.numa != me.numa
                    cross_numa += 1
                else
                    same_numa += 1
                end
                same_sock += 1
            end
        end
        total = max(same_sock, 1)
        pct(x) = lpad(string(round(100 * x / total; digits = 1), "%"), 10)
        println("   ", d, "  ", pct(same_numa), "  ", pct(cross_numa), "  ",
                pct(cross_sock), "  ", pct(off_node))
    end
    println()
    println("  dims=", decomp.dims, " -> Cart_create is row-major, so dimension 3 is")
    println("  the fastest-varying rank index and its neighbours are consecutive ranks.")
    println("  A high cross-socket share there is not automatically a defect: when a")
    println("  dimension's extent equals the socket count, every mapping must cross it,")
    println("  and confining the crossing to ONE axis is the good outcome -- measured on")
    println("  rzhound, socket == coords[3] exactly, leaving dims 1 and 2 fully intra-")
    println("  socket. Judge by cross-socket BYTES (links x n_halo x the transverse")
    println("  plane), not by the percentage, before reaching for an explicit dims=.")
end

MPI.Barrier(comm)
MPI.Finalize()
