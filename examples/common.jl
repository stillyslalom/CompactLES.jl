# Figure output, run caching and provenance for the documented examples: the
# Literate scripts in this directory that docs/make.jl lists in `EXAMPLES` and
# shows, without executing them, on the Examples pages. An example includes this
# file by path. It is not part of the package because it stamps documentation
# figures and has no use in a calculation, and it needs nothing beyond Base,
# three standard libraries and the `MPI` binding that `using CompactLES` brings
# into scope.

using Dates, Serialization, TOML

const CHECKOUT = normpath(joinpath(@__DIR__, ".."))
const CACHE_ROOT = joinpath(@__DIR__, "cache")
const RUN_STAMPS = NamedTuple[]

git(args...) = try
    readchomp(`git -C $CHECKOUT $args`)
catch
    ""
end

"""
    figure_dir(name; smoke) -> String

The directory an example writes its figures to: `docs/src/assets/examples/<name>`
for a full run, whose figures are committed, and a fresh temporary directory
for a smoke run, which must never replace them.
"""
function figure_dir(name; smoke::Bool)
    smoke && return mktempdir()
    dir = normpath(joinpath(@__DIR__, "..", "docs", "src", "assets", "examples", name))
    mkpath(dir)
    return dir
end

"""
    cached(f, name, key; smoke) -> value

The value of `f()`, a run of the example `name`, kept with the run's provenance
in `examples/cache/<name>/<key>.jls` so that the page's figures can be redrawn
without repeating it. `key` names the run's parameters; a run whose file exists
is read back, so the cache must be deleted after a change to the package that
should reach the figures. Every rank calls `f` when the file is missing, since a
run is collective, and rank 0 writes it. A smoke run neither reads nor writes
the cache. The provenance of every run, read or computed, is collected for
`write_provenance`.
"""
function cached(f, name, key; smoke::Bool)
    smoke && return f()
    file = joinpath(CACHE_ROOT, name, key * ".jls")
    if isfile(file)
        entry = deserialize(file)
        push!(RUN_STAMPS, merge(entry.stamp, (fresh = false,)))
        return entry.value
    end
    t0 = time()
    value = f()
    stamp = (commit = git("rev-parse", "HEAD"),
             dirty = !isempty(git("status", "--porcelain", "--", "src", "ext",
                                  "Project.toml")),
             date = Dates.format(Dates.now(), "yyyy-mm-dd"), julia = string(VERSION),
             ranks = MPI.Comm_size(MPI.COMM_WORLD), threads = Threads.nthreads(),
             hardware = strip(Sys.cpu_info()[1].model), wall = time() - t0)
    push!(RUN_STAMPS, merge(stamp, (fresh = true,)))
    if MPI.Comm_rank(MPI.COMM_WORLD) == 0
        mkpath(dirname(file))
        serialize(file, (; value, stamp))
    end
    return value
end

"""
    write_provenance(dir; command, settings, grid, wall, inputs = ())

Write `provenance.toml` into `dir`, from which docs/make.jl builds the note at
the head of the example's page. It records the commit and whether the package
source (`src`, `ext`, `Project.toml`) or any further path in `inputs`, relative
to the checkout, differed from it; the Julia version, rank and thread counts,
the processor and the date; and the arguments: `command`, the line that
reproduces the run; `settings`, the parsed `script_args` tuple; `grid`, a short
description of the grids; and `wall`, the script's wall time in seconds. Only
rank 0 writes.

When the runs went through `cached`, the commit, the date and the
machine are those of the runs, not of the script that drew the figures, and the
wall time is that of the runs plus the drawing: a run read from the cache adds
its recorded time, and one computed now is already in `wall`. Runs made at
different commits are recorded as the latest of them, marked dirty.

The example's own script is left out of the dirty test, since a new or
revised page is committed after the run that produced its figures.
"""
function write_provenance(dir; command::AbstractString, settings::NamedTuple,
                          grid::AbstractString, wall::Real, inputs = ())
    MPI.Comm_rank(MPI.COMM_WORLD) == 0 || return nothing
    commit = git("rev-parse", "HEAD")
    dirty = !isempty(git("status", "--porcelain", "--", "src", "ext", "Project.toml",
                         inputs...))
    machine = (julia = string(VERSION), ranks = MPI.Comm_size(MPI.COMM_WORLD),
               threads = Threads.nthreads(), hardware = strip(Sys.cpu_info()[1].model))
    date = Dates.format(Dates.now(), "yyyy-mm-dd")
    if !isempty(RUN_STAMPS)
        latest = RUN_STAMPS[argmax([s.date for s in RUN_STAMPS])]
        commits = unique(s.commit for s in RUN_STAMPS)
        commit, date = latest.commit, latest.date
        dirty = dirty || length(commits) > 1 || any(s.dirty for s in RUN_STAMPS)
        machine = (julia = latest.julia, ranks = latest.ranks, threads = latest.threads,
                   hardware = latest.hardware)
        wall += sum(s.wall for s in RUN_STAMPS if !s.fresh; init = 0.0)
    end
    record = Dict{String,Any}(
        "command" => command,
        "date" => date,
        "commit" => isempty(commit) ? "unknown" : commit,
        "dirty" => dirty,
        "julia" => machine.julia,
        "ranks" => machine.ranks,
        "threads" => machine.threads,
        "hardware" => machine.hardware,
        "grid" => grid,
        "wall_seconds" => round(Float64(wall); digits = 1),
        "settings" => Dict{String,Any}(String(k) => (v isa Symbol ? String(v) : v)
                                       for (k, v) in pairs(settings)),
    )
    open(joinpath(dir, "provenance.toml"), "w") do io
        TOML.print(io, record; sorted = true)
    end
    return nothing
end
