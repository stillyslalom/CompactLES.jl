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
                                  sizes = true, steps = 4, timeout = 120.0,
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
if isdefined(Base, :cumulative_compile_timing)
    Base.cumulative_compile_timing(true)
end

function compile_ns()
    isdefined(Base, :cumulative_compile_time_ns) || return UInt64(0)
    t = Base.cumulative_compile_time_ns()
    return t isa Tuple ? t[1] : t
end

"Evaluate `f()`, returning `(value, wall_seconds, compile_seconds)`."
function timed(f)
    c0 = compile_ns()
    t0 = time()
    v = f()
    return (v, time() - t0, (compile_ns() - c0) / 1e9)
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

function report_ranks()
    g = probe_grid()
    nsteps = max(2, OPT.steps)
    s, t_build, c_build = timed() do
        Solver(n_global = (g, g, g), L_domain = (2π, 2π, 2π), bcs = per3,
               art = ArtParams(enabled = false))
    end
    Q = allocate_state(s)
    _, t_init, c_init = timed() do
        initialize!(s, Q, (x, y, z) -> Prim(u = (0.1sin(x), 0, 0), p = 1.0,
                                            rho = 1.0))
    end
    # One step per call, so the first carries its compilation and the rest give
    # the steady-state cost of a step on this node.
    walls, comps = Float64[], Float64[]
    for k in 1:nsteps
        _, w, c = timed() do
            run!(s, Q; tfinal = 1e9, nmax = k)
        end
        push!(walls, w)
        push!(comps, c)
    end
    rest = sort(walls[2:end])[max(1, cld(length(walls) - 1, 2))]
    names = ("MPI.Init", "package load", "Solver", "initialize!", "step 1",
             "steps 2..$nsteps")
    mine = [t_mpi - t_start, t_load - t_mpi, t_build, t_init, walls[1], rest]
    comp = [0.0, 0.0, c_build, c_init, comps[1], sum(comps[2:end])]
    lo = MPI.Allreduce(mine, min, comm)
    hi = MPI.Allreduce(mine, max, comm)
    tot = MPI.Allreduce(mine, +, comm)
    chi = MPI.Allreduce(comp, max, comm)
    ctot = MPI.Allreduce(comp, +, comm)
    rank == 0 || return
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
    println("  the last column is compiler work, from the same counter",
            " test/timing.jl uses;")
    println("  it sums over threads, so it can exceed the wall time beside it.")
    println("  compiling after the package loaded, summed over ranks: ",
            round(sum(ctot[3:end]); digits = 1), " s of allocation")
    println("  package load, summed over ranks:                       ",
            round(tot[2]; digits = 1), " s of allocation")
end

CL.mpi_main() do
    if rank == 0
        println("\n=== depotprobe: depot cost and staging, ", np, " rank(s) ===")
        report_environment()
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
    report_ranks()
    rank == 0 && println(HEAD)
end
