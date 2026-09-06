# Diagnostic: what does the depot cost this launch, and can a staged one be used
# without any rank recompiling?
#
# A production run at large rank count pays package loading and compilation on
# every rank at once. Two things decide that cost and neither shows up in a
# timing: which filesystem the depot sits on, and whether the precompile caches
# in it are valid for the flags the launch line actually uses. An invalid cache
# is not an error today, it is every rank rebuilding it. This script measures
# both, and the staging path that avoids them.
#
#   julia --project=. depotprobe.jl                      # login node, all phases
#   julia --project=. depotprobe.jl /l/ssd/$USER/stage   # also stage and reload
#   srun -n 448 --cpu-bind=threads julia --project=. -t 1 depotprobe.jl
#
# Serial runs do the filesystem survey, the MPI.Init probe, the flag matrix and
# the staging round trip. Those spawn subprocesses, so they are skipped when the
# script is itself launched at more than one rank. Multi-rank runs report the
# per-rank load and first-step cost, which is the number staging exists to move:
# run it at several rank counts, and again with JULIA_DEPOT_PATH pointed at a
# staged depot, and compare.
#
# Options, parsed by `script_args` (src/scriptargs.jl):
#   stage=<dir>   directory to stage a depot copy into; empty skips the phase
#   grid=<N>      minimum cube edge for the per-rank step timing; raised to
#                 9 x the largest decomposition dimension so it survives the
#                 C8 filter's per-rank floor at any rank count (default 16)
#   steps=<N>     steps to time per rank; the first carries its compilation
#                 and the rest give the steady-state cost (default 4)
#   sweep=<N>     time steps at N grid sizes (g, 2g, ...) to separate a cost
#                 paid per point from one paid per call; 1 disables (default 2)
#   flags=<bool>  run the launch-flag matrix (default true)
#   io=<bool>     time a 64 MiB write and read on each node-local candidate
#   sizes=<bool>  measure depot directory sizes with du (default true)
#   timeout=<s>   ceiling on every subprocess and on du (default 120)
#   maxstage=<GiB> refuse to stage a depot larger than this (default 25)

const t_start = time()

using MPI
MPI.Init(threadlevel=:funneled)
const t_mpi = time()

using CompactLES
const t_load = time()

const CL = CompactLES
const OPT = CL.script_args(ARGS, (stage = "", grid = 16, flags = true, io = true,
                                  sizes = true, steps = 4, sweep = 2,
                                  timeout = 120.0,
                                  maxstage = 25.0);
                           positional = (:stage,))

const comm = MPI.COMM_WORLD
const rank = MPI.Comm_rank(comm)
const np = MPI.Comm_size(comm)
const per3 = ntuple(_ -> (PeriodicBC(), PeriodicBC()), 3)

# A bare interpreter path rather than `Base.julia_cmd()`, which propagates this
# process's own -O and --check-bounds into every probe and would make the flag
# matrix below compare a setting against itself.
const JULIA = joinpath(Sys.BINDIR, Base.julia_exename())
const PROJECT = something(Base.active_project(), "@.")
const HEAD = "-"^72

"""
    probe(cmd; timeout) -> (status, seconds, output)

Run `cmd` with a wall-clock ceiling. `status` is `:ok`, `:fail` or `:timeout`.

The ceiling is the point of this helper. A singleton `MPI.Init` under a
scheduler's process manager is reported to hang rather than to fail, and a probe
that inherits the hang reports nothing at all.
"""
function probe(cmd::Cmd; timeout::Float64 = OPT.timeout)
    log = tempname()
    killed = Ref(false)
    t0 = time()
    p = try
        run(pipeline(cmd; stdout = log, stderr = log); wait = false)
    catch err
        return (status = :fail, seconds = 0.0, output = sprint(showerror, err))
    end
    timer = Timer(timeout) do _
        if process_running(p)
            killed[] = true
            try kill(p) catch end
        end
    end
    try wait(p) catch end
    close(timer)
    seconds = time() - t0
    text = try read(log, String) catch; "" end
    rm(log; force = true)
    status = killed[] ? :timeout : (p.exitcode == 0 ? :ok : :fail)
    return (status = status, seconds = seconds, output = strip(text))
end

firstline(s) = isempty(s) ? "" : first(split(s, '\n'; limit = 2))

# Compile time is measured, not inferred from a first-minus-second difference:
# on a slow node a step costs more than the compilation inside it and the
# difference is noise, which is what the counter avoids. Same accounting as
# test/timing.jl, including its caveat that the counter sums compiler work over
# threads and can exceed wall time when there is more than one.
const COMPILE_TIMING = isdefined(Base, :cumulative_compile_timing) &&
                       isdefined(Base, :cumulative_compile_time_ns)
if COMPILE_TIMING
    Base.cumulative_compile_timing(true)
end

function compile_ns()
    isdefined(Base, :cumulative_compile_time_ns) || return UInt64(0)
    t = Base.cumulative_compile_time_ns()
    return t isa Tuple ? t[1] : t
end

"""
Evaluate `f()`, returning `(value, wall, compile, bytes, gc)`.

Allocation is measured beside the clock because the failure mode that looks
like this one — a step far slower than the machine's own arithmetic, with no
compilation in it — is usually inference giving up and putting a runtime
dispatch, and an allocation, on a path that should have neither. A solver step
on a fixed grid allocates a bounded amount; bytes proportional to the point
count are the diagnosis.
"""
function timed(f)
    c0 = compile_ns()
    b0 = Base.gc_bytes()
    g0 = Base.gc_time_ns()
    t0 = time()
    v = f()
    return (v, time() - t0, (compile_ns() - c0) / 1e9, Base.gc_bytes() - b0,
            (Base.gc_time_ns() - g0) / 1e9)
end

# stat -f names the filesystem behind a path, which is the whole question for a
# depot: "nfs" or "lustre" answers differently from "tmpfs" or "xfs". Linux only;
# a machine that does not answer reports "?" rather than failing the run.
function fsinfo(path)
    try
        out = readchomp(pipeline(`stat -f -c "%T %a %S" $path`; stderr = devnull))
        t, avail, bs = split(out)
        return (type = t, free = parse(Int, avail) * parse(Int, bs))
    catch
        return (type = "?", free = -1)
    end
end

function dirbytes(path)
    isdir(path) || return -1
    OPT.sizes || return -2
    r = probe(`du -sb $path`)
    r.status == :ok || return -1
    return something(tryparse(Int, String(first(split(r.output)))), -1)
end

gib(x) = x == -2 ? "skipped" : x < 0 ? "?" :
         string(round(x / 2^30; digits = 2), " GiB")

function writable(path)
    isdir(path) || return false
    f = joinpath(path, ".depotprobe_" * string(getpid()))
    try
        write(f, "x")
        rm(f; force = true)
        return true
    catch
        try rm(f; force = true) catch end
        return false
    end
end

# 64 MiB out and back. Node-local storage that is really a network mount shows up
# here and nowhere else in this report.
function io_rate(path)
    OPT.io || return (w = -1.0, r = -1.0)
    buf = rand(UInt8, 64 * 2^20)
    f = joinpath(path, ".depotprobe_io_" * string(getpid()))
    try
        tw = @elapsed open(io -> write(io, buf), f, "w")
        tr = @elapsed read(f)
        rm(f; force = true)
        return (w = length(buf) / 2^20 / tw, r = length(buf) / 2^20 / tr)
    catch
        try rm(f; force = true) catch end
        return (w = -1.0, r = -1.0)
    end
end

# How many CPUs this process was actually given, and whether a cgroup caps the
# share it may use of them. A login node hands out a fraction of a core to a
# heavy process, which slows arithmetic and MPI polling alike and is the first
# thing to rule out before believing any per-step number. The mount layout is
# not portable, so resolve from /proc/self/cgroup and walk upward, as
# clusterprobe.jl does for the memory limit.
function cgroup_cpu_quota()
    leaves = String[]
    try
        for line in eachline("/proc/self/cgroup")
            f = split(line, ':'; limit = 3)
            length(f) == 3 || continue
            if f[2] == ""
                push!(leaves, "/sys/fs/cgroup" * f[3] * "/cpu.max")
            elseif occursin("cpu,", f[2]) || f[2] == "cpu"
                push!(leaves, "/sys/fs/cgroup/cpu" * f[3] * "/cpu.cfs_quota_us")
            end
        end
    catch
    end
    for leaf in leaves
        dir, base = dirname(leaf), basename(leaf)
        while startswith(dir, "/sys/fs/cgroup")
            try
                s = strip(read(joinpath(dir, base), String))
                if base == "cpu.max"
                    parts = split(s)
                    parts[1] == "max" && return Inf
                    return parse(Float64, parts[1]) / parse(Float64, parts[2])
                else
                    q = parse(Float64, s)
                    q < 0 && return Inf
                    p = parse(Float64,
                              strip(read(joinpath(dir, "cpu.cfs_period_us"),
                                         String)))
                    return q / p
                end
            catch
            end
            dir = dirname(dir)
        end
    end
    return -1.0
end

function cpus_allowed()
    try
        for line in eachline("/proc/self/status")
            startswith(line, "Cpus_allowed_list:") || continue
            spec = strip(split(line, ':')[2])
            n = 0
            for part in split(spec, ',')
                lo_hi = split(part, '-')
                n += length(lo_hi) == 1 ? 1 :
                     parse(Int, lo_hi[2]) - parse(Int, lo_hi[1]) + 1
            end
            return n
        end
    catch
    end
    return -1
end

triad!(c, a, b, s) = (@inbounds @simd for i in eachindex(c)
    c[i] = a[i] + s * b[i]
end; c)

function dotp(a, b)
    s = zero(eltype(a))
    @inbounds @simd for i in eachindex(a)
        s += a[i] * b[i]
    end
    return s
end

# A floating-point and a memory-bandwidth baseline with no MPI, no allocation
# and no solver in them. If these are as slow as the solver, the machine is
# throttled and nothing about the package is implicated; if they are normal
# while a step is not, the problem is in the code or the Julia version.
function report_cpu()
    n = 1 << 20
    a, b, c = rand(n), rand(n), zeros(n)
    triad!(c, a, b, 1.0001)
    dotp(a, b)
    reps = 20
    t1 = @elapsed for _ in 1:reps
        triad!(c, a, b, 1.0001)
    end
    acc = 0.0
    t2 = @elapsed for _ in 1:reps
        acc += dotp(a, b)
    end
    quota = cgroup_cpu_quota()
    println(HEAD)
    println("this process's CPU:")
    println("  CPUs allowed  : ", cpus_allowed(), " of ", Sys.CPU_THREADS,
            " on the node")
    println("  cgroup quota  : ",
            quota < 0 ? "unreadable" : isinf(quota) ? "unlimited" :
            string(round(quota; digits = 3), " CPUs"))
    println("  triad         : ", round(24e-9 * n * reps / t1; digits = 2),
            " GB/s")
    isfinite(acc) || println("  (dot produced a non-finite sum)")
    println("  dot           : ", round(2e-9 * n * reps / t2; digits = 2),
            " GFLOP/s")
    println("  these carry no MPI and no solver. A step that is slow while",
            " these are normal")
    println("  is the code; slow together is the machine, and a login node",
            " throttles both.")
end

function report_environment()
    println(HEAD)
    println("julia           : ", VERSION, "  (", JULIA, ")")
    println("project         : ", PROJECT)
    fl = Base.JLOptions()
    println("launch flags    : check_bounds=", Int(fl.check_bounds),
            "  opt_level=", Int(fl.opt_level),
            "  nthreads=", Threads.nthreads())
    println("                  a cache is keyed on the first two; see the flag",
            " matrix below")
    binary = try string(MPI.MPIPreferences.binary) catch; "unknown" end
    println("MPI binary      : ", binary,
            endswith(binary, "_jll") ?
            "   <-- BUNDLED JLL; see reference/CLUSTER.md" : "")
    println("MPI library     : ", MPI.MPI_LIBRARY, " ", MPI.MPI_LIBRARY_VERSION)
    # The preference is per-project, so `--project=.` in the package directory
    # reports the package's own environment and not the one a production launch
    # uses. Probing the wrong project reports the wrong MPI and, because the
    # workload below is gated on it, the wrong precompile answer too.
    if endswith(binary, "_jll")
        println("                  this project is not configured against the",
                " system MPI, so it is")
        println("                  not what production runs; re-probe with the",
                " --project the launch")
        println("                  line uses. Selecting the system binary also",
                " closes the gate below.")
    end
    # The workload in src/precompile.jl is gated on this. Under a system binary it
    # does not run, so every rank compiles the solver tree itself at startup,
    # which is the cost the rest of this report is about.
    println("precompile workload : ",
            binary == "system" ?
                "SKIPPED (system MPI); every rank compiles at startup" :
                "built into the package image")
end

function report_depot()
    println(HEAD)
    println("depot entries (JULIA_DEPOT_PATH); the first takes every write:")
    for (i, d) in enumerate(DEPOT_PATH)
        fs = fsinfo(d)
        println("  [", i, "] ", d)
        println("      exists=", isdir(d), "  writable=", writable(d),
                "  fs=", fs.type, "  free=", gib(fs.free))
        for sub in ("compiled", "packages", "artifacts")
            p = joinpath(d, sub)
            isdir(p) && println("      ", rpad(sub, 10), gib(dirbytes(p)))
        end
    end
end

function report_nodelocal()
    println(HEAD)
    println("node-local candidates (staging targets):")
    seen = String[]
    for p in filter(!isempty, [get(ENV, "TMPDIR", ""), "/dev/shm", "/tmp",
                               "/var/tmp", get(ENV, "SCRATCH", "")])
        p in seen && continue
        push!(seen, p)
        if !isdir(p)
            println("  ", rpad(p, 22), "absent")
            continue
        end
        fs = fsinfo(p)
        r = io_rate(p)
        rate = r.w < 0 ? "" :
               string("  write ", round(Int, r.w), " MiB/s  read ",
                      round(Int, r.r), " MiB/s")
        println("  ", rpad(p, 22), "fs=", rpad(fs.type, 10),
                "free=", rpad(gib(fs.free), 12), "writable=", writable(p), rate)
    end
end

# Can the precompile workload's singleton MPI.Init run here? Under a scheduler's
# process manager it may fail, or hang, and the timeout separates those.
function report_mpi_init()
    println(HEAD)
    code = "using MPI; MPI.Init(); print(\"WORLD=\", MPI.Comm_size(MPI.COMM_WORLD)); " *
           "MPI.Finalize()"
    # `existing` on both, so a probe run on a login node can never write a cache
    # into a shared depot as a side effect of being asked a question.
    r = probe(`$JULIA --project=$PROJECT --startup-file=no
               --compiled-modules=existing --pkgimages=existing -e $code`)
    world = match(r"WORLD=(\d+)", r.output)
    println("singleton MPI.Init (subprocess, ", round(Int, OPT.timeout),
            " s ceiling): ", r.status, "  ", round(r.seconds; digits = 1), " s")
    if r.status == :ok && world !== nothing
        println("  world size ", world[1],
                "; the src/precompile.jl gate can be lifted here")
    elseif r.status == :ok
        println("  exited cleanly but printed no world size: ",
                firstline(r.output))
    elseif r.status == :timeout
        println("  hung; the gate must stay closed, or the workload has to be",
                " built inside a one-rank job step")
    else
        println("  ", firstline(r.output))
    end
end

# Which launch flags throw the staged caches away. Each probe loads the package
# under --compiled-modules=strict, which errors when a cache is missing instead
# of rebuilding it silently, so a failure here names a flag that would put every
# rank of a production launch into a rebuild.
function report_flags()
    OPT.flags || return
    println(HEAD)
    println("launch-flag matrix (strict load of CompactLES):")
    for extra in (String[], ["-O3"], ["-O0"], ["--check-bounds=yes"],
                  ["--check-bounds=no"], ["-t", "4"], ["--min-optlevel=1"])
        cmd = `$JULIA --project=$PROJECT --startup-file=no
               --compiled-modules=strict --pkgimages=existing $extra
               -e "using CompactLES"`
        r = probe(cmd)
        label = isempty(extra) ? "(as this process)" : join(extra, " ")
        println("  ", rpad(label, 22),
                r.status == :ok ? "reuses the cache" :
                "REBUILDS  <-- not in a launch line unless the depot was " *
                "built with it")
    end
end

# Pack, unpack, load. tar rather than a recursive copy because it is what a
# scheduler broadcast moves: one object instead of the depot's inode count.
function report_stage()
    isempty(OPT.stage) && return
    println(HEAD)
    src = first(DEPOT_PATH)
    if !isdir(src)
        println("stage: depot ", src, " absent")
        return
    end
    # A depot accumulates one image set per flag and preference combination and
    # is not collected aggressively, so the first entry can be tens of GiB. Size
    # it before copying: this phase is often run on a login node.
    bytes = begin
        r = probe(`du -sb $src`; timeout = max(OPT.timeout, 300.0))
        r.status == :ok ? something(tryparse(Int, String(first(split(r.output)))),
                                    -1) : -1
    end
    println("stage source ", src, "  ", gib(bytes))
    if bytes > OPT.maxstage * 2^30
        println("  above maxstage=", OPT.maxstage, " GiB; skipped. Raise it, or",
                " stage a depot built for this campaign rather than a shared one.")
        return
    end
    mkpath(OPT.stage)
    tarball = joinpath(OPT.stage, "depot.tar")
    dest = joinpath(OPT.stage, "depot")
    long = max(OPT.timeout, 900.0)
    println("staging ", src, " -> ", OPT.stage)
    # Both tar calls run inside the stage directory with a relative archive
    # name. tar reads an `-f` argument containing a colon as host:path, which a
    # Windows drive letter trips; a relative name never reaches that rule.
    rp = probe(Cmd(`tar -cf depot.tar -C $(dirname(src)) $(basename(src))`;
                   dir = OPT.stage); timeout = long)
    if rp.status != :ok
        println("  pack ", rp.status, ": ", firstline(rp.output))
        return
    end
    println("  pack   ", round(rp.seconds; digits = 1), " s  ",
            gib(filesize(tarball)))
    rm(dest; force = true, recursive = true)
    mkpath(dest)
    ru = probe(Cmd(`tar -xf depot.tar -C $dest --strip-components=1`;
                   dir = OPT.stage); timeout = long)
    if ru.status != :ok
        println("  unpack ", ru.status, ": ", firstline(ru.output))
        return
    end
    println("  unpack ", round(ru.seconds; digits = 1), " s")
    code = "t = @elapsed (using CompactLES); print(round(t; digits = 2))"
    sep = Sys.iswindows() ? ";" : ":"
    r = probe(setenv(`$JULIA --project=$PROJECT --startup-file=no
                      --compiled-modules=strict --pkgimages=existing -e $code`,
                     "JULIA_DEPOT_PATH" => dest * sep))
    println("  strict load from the staged depot: ", r.status,
            r.status == :ok ? "  " * r.output * " s to load CompactLES" : "")
    r.status == :ok || println("  ", firstline(r.output))
end

# What every rank pays at startup. The gap between the first and second step is
# the compilation the package image did not cover; the load time is what the
# depot's filesystem costs when every rank reads it at once.
# The C8 filter needs 9 points per rank per dimension, so a fixed cube edge
# stops decomposing somewhere above a handful of ranks. Size it from the
# decomposition MPI would pick. The second argument to Dims_create is the dims
# array, not the dimension count; see the note in src/decomposition.jl.
function probe_grid()
    dims = try
        Int.(MPI.Dims_create(np, zeros(Cint, 3)))
    catch
        [np, 1, 1]
    end
    return max(OPT.grid, 9 * maximum(dims))
end

"""
    step_costs(g, nsteps) -> NamedTuple

Build a solver at cube edge `g` and time `nsteps` steps, one per `run!` call, so
the first carries its compilation and the rest give the steady-state cost.
`waits` is this rank's time inside the run-wide collectives, as `Solver` charges
it; it does not include the compact solves' own exchanges.
"""
function step_costs(g, nsteps)
    s, t_build, c_build, _, _ = timed() do
        Solver(n_global = (g, g, g), L_domain = (2π, 2π, 2π), bcs = per3,
               art = ArtParams(enabled = false))
    end
    Q = allocate_state(s)
    _, t_init, c_init, _, _ = timed() do
        initialize!(s, Q, (x, y, z) -> Prim(u = (0.1sin(x), 0, 0), p = 1.0,
                                            rho = 1.0))
    end
    walls, comps, waits = Float64[], Float64[], Float64[]
    bytes, gcs = Float64[], Float64[]
    for k in 1:nsteps
        _, w, c, b, gc = timed() do
            run!(s, Q; tfinal = 1e9, nmax = k)
        end
        push!(walls, w)
        push!(comps, c)
        push!(waits, s.wall_wait)
        push!(bytes, Float64(b))
        push!(gcs, gc)
    end
    return (walls = walls, comps = comps, waits = waits, bytes = bytes,
            gcs = gcs, points = prod(s.decomp.n_local), t_build = t_build,
            c_build = c_build, t_init = t_init, c_init = c_init)
end

median_of(v) = isempty(v) ? 0.0 : sort(v)[max(1, cld(length(v), 2))]

# Whether a step costs per point or per call. Doubling the edge multiplies the
# points by eight; a step time that does not follow is dominated by something
# paid once per call rather than per point, which on a shared node is usually
# the MPI library's wait behaviour and not the arithmetic.
function report_sweep(base, nsteps)
    OPT.sweep <= 1 && return
    rows = NTuple{5,Float64}[]
    for m in 1:OPT.sweep
        g = base * m
        k = m == 1 ? nsteps : 2
        r = step_costs(g, k)
        steady = median_of(r.walls[2:end])
        push!(rows, (Float64(g), Float64(r.points), steady,
                     median_of(r.bytes[2:end]), median_of(r.gcs[2:end])))
    end
    hi = MPI.Allreduce([r[3] for r in rows], max, comm)
    hb = MPI.Allreduce([r[4] for r in rows], max, comm)
    hg = MPI.Allreduce([r[5] for r in rows], max, comm)
    rank == 0 || return
    println()
    println("  step cost against grid size, max over ranks:")
    println("    ", rpad("grid", 7), rpad("points", 10), rpad("s/step", 10),
            rpad("ns/point", 11), rpad("MiB/step", 11), rpad("B/point", 10),
            "gc s")
    for (i, r) in enumerate(rows)
        println("    ", rpad(Int(r[1]), 7), rpad(Int(r[2]), 10),
                rpad(round(hi[i]; digits = 3), 10),
                rpad(round(Int, 1e9 * hi[i] / r[2]), 11),
                rpad(round(hb[i] / 2^20; digits = 1), 11),
                rpad(round(Int, hb[i] / r[2]), 10),
                round(hg[i]; digits = 3))
    end
    if length(rows) >= 2
        grew = rows[end][2] / rows[1][2]
        slowed = hi[1] > 0 ? hi[end] / hi[1] : 0.0
        println("    points x", round(grew; digits = 1), ", step time x",
                round(slowed; digits = 1),
                slowed < 0.25 * grew ?
                "  -- paid per call, not per point" :
                "  -- paid per point; the node is doing the arithmetic")
    end
end

function report_ranks()
    g = probe_grid()
    nsteps = max(2, OPT.steps)
    r = step_costs(g, nsteps)
    t_build, c_build, t_init, c_init = r.t_build, r.c_build, r.t_init, r.c_init
    walls, comps = r.walls, r.comps
    rest = median_of(walls[2:end])
    names = ("MPI.Init", "package load", "Solver", "initialize!", "step 1",
             "steps 2..$nsteps")
    mine = [t_mpi - t_start, t_load - t_mpi, t_build, t_init, walls[1], rest]
    comp = [0.0, 0.0, c_build, c_init, comps[1], sum(comps[2:end])]
    lo = MPI.Allreduce(mine, min, comm)
    hi = MPI.Allreduce(mine, max, comm)
    tot = MPI.Allreduce(mine, +, comm)
    chi = MPI.Allreduce(comp, max, comm)
    ctot = MPI.Allreduce(comp, +, comm)
    if rank != 0
        return g, nsteps
    end
    println(HEAD)
    println("per-rank startup at ", np, " rank(s), grid ", g, "^3, seconds:")
    println("  ", rpad("phase", 16), rpad("min", 9), rpad("mean", 9),
            rpad("max", 9), "compiling (max)")
    for (i, name) in enumerate(names)
        println("  ", rpad(name, 16), rpad(round(lo[i]; digits = 2), 9),
                rpad(round(tot[i] / np; digits = 2), 9),
                rpad(round(hi[i]; digits = 2), 9),
                i <= 2 ? "n/a" : string(round(chi[i]; digits = 2)))
    end
    if COMPILE_TIMING
        println("  the last column is compiler work, from the same counter",
                " test/timing.jl uses;")
        println("  it sums over threads, so it can exceed the wall time",
                " beside it.")
    else
        println("  this Julia has no compile-time counter, so the last column",
                " reads zero and")
        println("  means nothing. Compare step 1 against the later steps",
                " instead: they carry")
        println("  no compilation, so a step as slow as the first is slow",
                " for another reason.")
    end
    println("  compiling after the package loaded, summed over ranks: ",
            round(sum(ctot[3:end]); digits = 1), " s of allocation")
    println("  package load, summed over ranks:                       ",
            round(tot[2]; digits = 1), " s of allocation")
    return g, nsteps
end

CL.mpi_main() do
    if rank == 0
        println("\n=== depotprobe: depot cost and staging, ", np, " rank(s) ===")
        report_environment()
        report_cpu()
        report_depot()
        if np == 1
            report_nodelocal()
            report_mpi_init()
            report_flags()
            report_stage()
        else
            println(HEAD)
            println("subprocess phases (node-local survey, MPI.Init, flag matrix,",
                    " staging) are")
            println("skipped at np > 1; run this script serially for those.")
        end
        flush(stdout)
    end
    MPI.Barrier(comm)
    g, nsteps = report_ranks()
    report_sweep(g, nsteps)
    rank == 0 && println(HEAD)
end
