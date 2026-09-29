# Medians over repeated processes: runs a command several times, one process
# after another and never two at once, extracts labelled numbers from what each
# process prints, and reports per label the median, minimum, maximum, relative
# spread (max - min)/median and the number of runs, followed by the raw values.
#
# Run-to-run spread on a desktop is 10-20% for an identical configuration
# (CLAUDE.md, Timing noise), and much of it is per process: one heap, one set of
# compiled code, one placement of threads on performance or efficiency cores for
# the life of the process. Repetitions inside one process do not sample that
# variation; separate processes do.
#
# One command gives the single-command summary. Two commands, each introduced by
# a bare `--`, give the paired comparison: the runner alternates them in the
# order A B, B A, A B, ... and forms the paired ratio B/A of each adjacent pair.
# Slow drift across the session (thermal throttling, a migration to efficiency
# cores, another process starting) then affects both members of a pair nearly
# equally and cancels in the ratio. Reversing the order in every second pair
# cancels a drift linear in time, which a fixed A-then-B order would leave in
# every ratio as the drift over one run.
#
# Usage: the runner's own `key=value` options, then `--` and the command. A
# command whose first word ends in `.jl` is a Julia script, launched with this
# Julia binary and its `-O`, `-g` and `--startup-file` settings, `-t` from
# `threads`, `--project` from `project`, and `julia_flags` appended last so that
# they override the others. Any other command runs verbatim.
#
#   julia -t 16 bench/repeat.jl runs=5 -- bench/phases.jl
#
# `bench/derivcost.jl` prints its results as `row,` lines; this pattern labels
# each by case and operator and takes the ns/pt/step column:
#
#   ROW='pattern=^row,([^,]+),([^,]+),(?:[^,]*,){5}([^,]+)'
#   julia -t 16 bench/repeat.jl runs=3 "$ROW" -- bench/derivcost.jl 32 10 cases=periodic
#   julia -t 16 bench/repeat.jl runs=6 "$ROW" \
#       -- ../before/bench/derivcost.jl 64 30 -- bench/derivcost.jl 64 30
#
# and under MPI, where the command runs verbatim and a timeout guards a hang:
#
#   MPIEXEC=$(julia --project=. -e 'using MPI; MPI.mpiexec(c -> print(c))')
#   julia bench/repeat.jl runs=5 timeout=600 "$ROW" \
#       -- "$MPIEXEC" -n 8 julia --project=. -t 1 bench/derivcost.jl 128 30
#
# The runner itself loads only Base and Printf (it includes `src/scriptargs.jl`
# rather than loading the package), so it adds no compile work and takes no part
# in the precompile lock. Precompile each environment serially before a sweep
# anyway, or let `warmup` discard the run that pays for it: a precompile inside
# a timed run is a large outlier.
#
# Comparing a before and an after:
#
# - Paired (recommended): one invocation with both commands. For the two trees of
#   a before/after, give each tree's script by path; with `project` empty each
#   script runs against the checkout that contains it. The difference is resolved
#   when every paired ratio lies on the same side of 1 (the `above 1` column
#   reads 0/n or n/n). With no true difference that happens by chance with
#   probability 2^(1-n): 1/8 at n = 4, 1/16 at n = 5, 1/32 at n = 6, so take six
#   pairs for a few-percent claim. The median paired ratio is the estimate.
# - Unpaired: two invocations with the same runner flags and the same `threads`,
#   one per command. The difference of medians is resolved when the two
#   [min, max] ranges do not overlap, which in practice requires a difference
#   larger than the spread column of either run. Drift between the invocations
#   lands entirely in the difference, so prefer the paired form.
#
# Options (before the first `--`):
#   runs        timed runs per command (default 5); in the paired form, pairs.
#   warmup      runs per command before the timed ones, run and discarded
#               (default 1). The first run of a session pays for any stale
#               package image and for the file cache.
#   min_runs    fewest successful timed runs (pairs in the paired form) for exit
#               status 0 (default 0: half of `runs`, rounded up). A run that
#               exits nonzero or times out is reported with its exit code and
#               excluded; in the paired form it excludes its pair.
#   pattern     regex applied to every line of a run's stdout and stderr, every
#               match counted. The last capture group is the value; the other
#               groups, joined by a space, are the label. With one group the
#               label is `value`. The default matches `name = number` and
#               `name: number` at the start of a line or after `,`, `;`, a tab
#               or two spaces, where a name is words separated by single spaces.
#               A label seen a second time in one run is recorded as `label#2`.
#   keys        comma-separated labels to report, without the `#n` suffix
#               (default: every label found). A listed label that no run
#               printed is an error.
#   threads     `-t` for a Julia script (default 0: this runner's own thread
#               count, so `julia -t 16 bench/repeat.jl ...` runs its script at
#               16 threads).
#   project     `--project` for a Julia script (default: the nearest directory
#               above the script holding a `Project.toml`).
#   julia_flags extra flags for a Julia script, space-separated, e.g.
#               `julia_flags=-O1` or `julia_flags=--check-bounds=no`.
#   timeout     seconds after which a run is killed and counted as failed
#               (default 0: none). A run that loses positivity grinds rather
#               than crashing. Killing an `mpiexec` launcher on Windows may leave
#               its ranks running; check for orphans by command line.
#   out         transcript path (default none): the header, every run's full
#               output and the summary. The convention for an instrument's
#               record is `bench/results/<script>.txt`.
#   echo        print every run's output as it arrives (default false; the
#               terminal otherwise shows one status line per run).
#
# The process wall time of every run, compilation included, is always reported
# as `process wall [s]`.
#
# Prints tables and asserts nothing about the numbers; the exit status is
# nonzero when too few runs succeeded or no value was extracted.

include(joinpath(@__DIR__, "..", "src", "scriptargs.jl"))
using Printf

const DEFAULTS = (runs = 5, warmup = 1, min_runs = 0, pattern = "", keys = "",
                  threads = 0, project = "", julia_flags = "", timeout = 0.0,
                  out = "", echo = false)

const NUMBER = raw"([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)"
const NAME = raw"[A-Za-z_][\w!.\[\]()/^*-]*(?: [\w!.\[\]()/^*-]+)*"
const DEFAULT_PATTERN = raw"(?:^|[,;]\s*|\t|\s{2,})(" * NAME * raw")\s*[:=]\s*" * NUMBER
const WALL_KEY = "process wall [s]"

struct RunResult
    command::Int                  # 1 = A, 2 = B
    exitcode::Int
    timed_out::Bool
    wall::Float64
    values::Vector{Pair{String,Float64}}
end

succeeded(r::RunResult) = r.exitcode == 0 && !r.timed_out

function median_of(v)
    s = sort(v)
    n = length(s)
    return isodd(n) ? s[(n + 1) ÷ 2] : (s[n ÷ 2] + s[n ÷ 2 + 1]) / 2
end

# The runner's options are the arguments before the first bare `--`; each `--`
# after that begins a command. A bare `--` cannot occur inside a command.
function split_arguments(args)
    at = findall(==("--"), args)
    isempty(at) && throw(ArgumentError("no command: give it after a bare `--`"))
    length(at) > 2 &&
        throw(ArgumentError("at most two commands (A and B), each after `--`"))
    bounds = vcat(at, length(args) + 1)
    commands = [String.(args[bounds[i] + 1:bounds[i + 1] - 1]) for i in 1:length(at)]
    any(isempty, commands) && throw(ArgumentError("an empty command after `--`"))
    return args[1:at[1] - 1], commands
end

function enclosing_project(path)
    dir = dirname(path)
    while true
        isfile(joinpath(dir, "Project.toml")) && return dir
        parent = dirname(dir)
        parent == dir && return dirname(@__DIR__)
        dir = parent
    end
end

function build_command(words, opt)
    endswith(words[1], ".jl") || return Cmd(words)
    script = abspath(words[1])
    isfile(script) || throw(ArgumentError("no script at $(words[1])"))
    project = isempty(opt.project) ? enclosing_project(script) : abspath(opt.project)
    threads = opt.threads > 0 ? opt.threads : Threads.nthreads()
    flags = split(opt.julia_flags)
    return `$(Base.julia_cmd()) --project=$project -t $threads $flags $script $(words[2:end])`
end

function extract(lines, rx, wanted)
    values = Pair{String,Float64}[]
    seen = Dict{String,Int}()
    for line in lines, m in eachmatch(rx, line)
        caps = m.captures
        value = caps[end] === nothing ? nothing : tryparse(Float64, caps[end])
        value === nothing && continue
        label = length(caps) == 1 ? "value" :
                join((strip(c) for c in caps[1:end - 1] if c !== nothing), " ")
        isempty(wanted) || label in wanted || continue
        n = seen[label] = get(seen, label, 0) + 1
        push!(values, (n == 1 ? label : "$label#$n") => value)
    end
    return values
end

# One process. Stdout is read line by line as it arrives, so `echo` shows
# progress and the transcript holds whatever a killed run printed; stderr is
# collected beside it and searched after stdout.
function run_once(cmd, command, opt, rx, wanted, say, transcript)
    err = IOBuffer()
    lines = String[]
    t0 = time()
    proc = open(pipeline(ignorestatus(cmd); stderr=err), "r")
    timed_out = Ref(false)
    timer = opt.timeout > 0 ? Timer(opt.timeout) do _
        process_running(proc) && (timed_out[] = true; kill(proc))
    end : nothing
    for line in eachline(proc)
        push!(lines, line)
        opt.echo && println(line)
        transcript === nothing || println(transcript, line)
    end
    wait(proc)
    timer === nothing || close(timer)
    wall = time() - t0
    errlines = split(String(take!(err)), r"\r?\n"; keepempty=false)
    if !isempty(errlines)
        opt.echo && foreach(println, errlines)
        if transcript !== nothing
            println(transcript, "--- stderr")
            foreach(l -> println(transcript, l), errlines)
        end
    end
    values = extract(vcat(lines, errlines), rx, wanted)
    push!(values, WALL_KEY => wall)
    result = RunResult(command, proc.exitcode, timed_out[], wall, values)
    if !succeeded(result)
        tail = errlines[max(1, end - 9):end]
        isempty(tail) || say("    last lines of stderr:\n" *
                             join(("      " * l for l in tail), "\n") * "\n")
    end
    return result
end

status(r) = r.timed_out ? "timed out" : "exit $(r.exitcode)"

# Labels in the order of first appearance, over the runs given.
function labels_of(results)
    order = String[]
    for r in results, (k, _) in r.values
        k in order || push!(order, k)
    end
    return order
end

value_of(r, key) = (i = findfirst(p -> first(p) == key, r.values);
                    i === nothing ? nothing : last(r.values[i]))

fmt(x) = @sprintf("%11.4g", x)

function single_summary(results, say)
    ok = filter(succeeded, results)
    isempty(ok) && return 0, String[]
    labels = labels_of(ok)
    width = max(12, maximum(length, labels))
    say(@sprintf("\n  %-*s %4s %11s %11s %11s %8s\n", width, "label", "n",
                 "median", "min", "max", "spread"))
    for key in labels
        v = [x for x in (value_of(r, key) for r in ok) if x !== nothing]
        m = median_of(v)
        say(@sprintf("  %-*s %4d %s %s %s %7.1f%%\n", width, key, length(v), fmt(m),
                     fmt(minimum(v)), fmt(maximum(v)),
                     100 * (maximum(v) - minimum(v)) / abs(m)))
    end
    say("\n  raw values in run order (- where a run printed no such label)\n")
    for key in labels
        say(@sprintf("  %-*s %s\n", width, key, join(
            (x === nothing ? "          -" : fmt(x) for x in
             (value_of(r, key) for r in ok)), " ")))
    end
    return length(ok), labels
end

function paired_summary(pairs, say)
    ok = [(a, b) for (a, b) in pairs if succeeded(a) && succeeded(b)]
    isempty(ok) && return 0, String[]
    labels = labels_of(vcat(first.(ok), last.(ok)))
    labels = filter(k -> any(value_of(a, k) !== nothing && value_of(b, k) !== nothing
                             for (a, b) in ok), labels)
    width = max(12, maximum(length, labels; init=0))
    # Left of the bar the two medians and their ratio; right of it the paired
    # ratios B/A, one per pair.
    say(@sprintf("\n  %-*s %4s %11s %11s %9s | %9s %7s %7s %8s\n", width, "label",
                 "n", "A median", "B median", "B/A", "paired", "min", "max",
                 "above 1"))
    for key in labels
        both = [(value_of(a, key), value_of(b, key)) for (a, b) in ok]
        both = [(x, y) for (x, y) in both if x !== nothing && y !== nothing]
        a, b = first.(both), last.(both)
        ratio = b ./ a
        ma, mb = median_of(a), median_of(b)
        say(@sprintf("  %-*s %4d %s %s %9.4f | %9.4f %7.4f %7.4f %8s\n",
                     width, key, length(both), fmt(ma), fmt(mb), mb / ma,
                     median_of(ratio), minimum(ratio), maximum(ratio),
                     "$(count(>(1), ratio))/$(length(ratio))"))
    end
    say("\n  raw values in pair order (A, B, B/A)\n")
    for key in labels
        say("  $key\n")
        for (tag, pick) in (("A", p -> value_of(p[1], key)),
                           ("B", p -> value_of(p[2], key)))
            say(@sprintf("    %-3s %s\n", tag, join(
                (x === nothing ? "          -" : fmt(x) for x in map(pick, ok)), " ")))
        end
        say(@sprintf("    %-3s %s\n", "B/A", join((
            (x = value_of(a, key); y = value_of(b, key);
             x === nothing || y === nothing ? "          -" : fmt(y / x))
            for (a, b) in ok), " ")))
    end
    return length(ok), labels
end

function main(args)
    own, words = split_arguments(args)
    opt = script_args(own, DEFAULTS)
    opt.runs >= 1 || throw(ArgumentError("runs must be at least 1"))
    rx = Regex(isempty(opt.pattern) ? DEFAULT_PATTERN : opt.pattern)
    ncaps = Int(Base.PCRE.info(rx.regex, Base.PCRE.INFO_CAPTURECOUNT, UInt32))
    ncaps >= 1 || throw(ArgumentError("pattern has no capture group for the value"))
    wanted = Set(String.(strip.(split(opt.keys, ','; keepempty=false))))
    min_runs = opt.min_runs > 0 ? opt.min_runs : cld(opt.runs, 2)
    cmds = [build_command(w, opt) for w in words]
    paired = length(cmds) == 2

    transcript = nothing
    if !isempty(opt.out)
        mkpath(dirname(abspath(opt.out)))
        transcript = open(opt.out, "w")
    end
    say(s) = (print(s); flush(stdout);
              transcript === nothing || (print(transcript, s); flush(transcript)))

    say("=== repeated processes: $(opt.runs) timed $(paired ? "pairs" : "runs"), " *
        "$(opt.warmup) warm-up run(s) per command\n")
    say("    $(gethostname()), $(Sys.CPU_NAME), $(Sys.CPU_THREADS) logical CPUs, " *
        "$(Sys.KERNEL), Julia $(VERSION)\n")
    for (i, c) in enumerate(cmds)
        say("    $(paired ? ("A", "B")[i] : "command"): $(join(c.exec, " "))\n")
    end
    say("    pattern: $(rx.pattern)\n")
    isempty(wanted) || say("    keys: $(join(sort!(collect(wanted)), ", "))\n")
    paired && say("    order: A B, B A, A B, ... (reversed in every second pair)\n")

    tag(i) = paired ? ("A", "B")[i] : ""
    function launch(i, what)
        transcript === nothing ||
            println(transcript, "\n--- $what $(tag(i)): $(join(cmds[i].exec, " "))")
        r = run_once(cmds[i], i, opt, rx, wanted, say, transcript)
        nvalues = length(r.values) - 1
        say(@sprintf("  %-14s %-2s %-10s %8.1f s  %d value(s)%s\n", what, tag(i),
                     status(r), r.wall, nvalues, succeeded(r) ? "" : ", excluded"))
        return r
    end

    for w in 1:opt.warmup, i in eachindex(cmds)
        launch(i, "warm-up $w")
    end
    if paired
        pairs = Tuple{RunResult,RunResult}[]
        for p in 1:opt.runs
            order = isodd(p) ? (1, 2) : (2, 1)
            r = Dict(i => launch(i, "pair $p/$(opt.runs)") for i in order)
            push!(pairs, (r[1], r[2]))
        end
        nok, labels = paired_summary(pairs, say)
    else
        results = [launch(1, "run $n/$(opt.runs)") for n in 1:opt.runs]
        nok, labels = single_summary(results, say)
    end

    missing_keys = setdiff(wanted, (replace(k, r"#\d+$" => "") for k in labels))
    failed = false
    if nok < min_runs
        say("\n  $nok successful $(paired ? "pair(s)" : "run(s)"), fewer than " *
            "min_runs = $min_runs\n")
        failed = true
    end
    if nok > 0 && !isempty(missing_keys)
        say("\n  no run printed: $(join(sort!(collect(missing_keys)), ", "))\n")
        failed = true
    end
    if nok > 0 && all(==(WALL_KEY), labels)
        say("\n  the pattern matched nothing; only the process wall time was recorded\n")
        failed = true
    end
    transcript === nothing || close(transcript)
    return failed ? 1 : 0
end

exit(main(ARGS))
