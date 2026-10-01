# Figure output and provenance for the documented examples: the Literate
# scripts in this directory that docs/make.jl lists in `EXAMPLES` and shows,
# without executing them, on the Examples pages. An example includes this file
# by path. It is not part of the package because it stamps documentation
# figures and has no use in a calculation, and it needs nothing beyond Base,
# two standard libraries and the `MPI` binding that `using CompactLES` brings
# into scope.

using Dates, TOML

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
    write_provenance(dir; command, settings, grid, wall, inputs = ())

Write `provenance.toml` into `dir`, from which docs/make.jl builds the note at
the head of the example's page. It records the commit and whether the package
source (`src`, `ext`, `Project.toml`) or any further path in `inputs`, relative
to the checkout, differed from it; the Julia version, rank and thread counts,
the processor and the date; and the arguments: `command`, the line that
reproduces the run; `settings`, the parsed `script_args` tuple; `grid`, a short
description of the grids; and `wall`, the run's wall time in seconds. Only
rank 0 writes.

The example's own script is left out of the dirty test, since a new or
revised page is committed after the run that produced its figures.
"""
function write_provenance(dir; command::AbstractString, settings::NamedTuple,
                          grid::AbstractString, wall::Real, inputs = ())
    MPI.Comm_rank(MPI.COMM_WORLD) == 0 || return nothing
    root = normpath(joinpath(@__DIR__, ".."))
    git(args...) = try
        readchomp(`git -C $root $args`)
    catch
        ""
    end
    commit = git("rev-parse", "HEAD")
    changed = git("status", "--porcelain", "--", "src", "ext", "Project.toml", inputs...)
    record = Dict{String,Any}(
        "command" => command,
        "date" => Dates.format(Dates.now(), "yyyy-mm-dd"),
        "commit" => isempty(commit) ? "unknown" : commit,
        "dirty" => !isempty(changed),
        "julia" => string(VERSION),
        "ranks" => MPI.Comm_size(MPI.COMM_WORLD),
        "threads" => Threads.nthreads(),
        "hardware" => strip(Sys.cpu_info()[1].model),
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
