# Parse the .cov files julia --code-coverage=user leaves next to src/*.jl and
# report per-file line coverage plus the uncovered lines themselves.
#
# This script only reads what a coverage-enabled run produced. Every run drops
# its own .cov file per source file, named with the pid; they are merged here
# by summing hit counts, so a line counts as covered if ANY run hit it. Clear
# the old ones first or a stale run inflates the result:
#
#   rm -f src/*.cov test/*.cov
#   julia --project=. --code-coverage=user test/runtests.jl
#   julia --project=. --code-coverage=user test/convergence.jl
#   mpiexec -n 2 julia --project=. --code-coverage=user test/mpi_tests.jl
#   mpiexec -n 4 julia --project=. --code-coverage=user test/mpi_tests.jl
#   mpiexec -n 8 julia --project=. --code-coverage=user test/mpi_tests.jl
#   julia --project=. bench/coverage.jl
#
# The MPI runs matter: serial alone reaches 94.8% of executable lines, the full
# set 97.2%, and everything in between is distributed-solve and off-rank-fold
# code that only a decomposed run reaches.
#
#   .cov format: one line per source line, prefixed
#     "-"  not executable (comment, blank, `end`, ...)
#     "0"  executable, never run
#     "N"  executable, run N times
using Printf

const SRC = joinpath(@__DIR__, "..", "src")

"Merge every <file>.jl.*.cov into a hit-count vector (nothing = not executable)."
function merged(file::String)
    covs = filter(f -> startswith(f, basename(file) * ".") && endswith(f, ".cov"),
                  readdir(SRC))
    isempty(covs) && return nothing
    acc = Union{Nothing,Int}[]
    for c in covs
        for (i, ln) in enumerate(eachline(joinpath(SRC, c)))
            tok = strip(first(ln, 9))
            v = tok == "-" ? nothing : parse(Int, tok)
            if i > length(acc)
                push!(acc, v)
            elseif acc[i] !== nothing && v !== nothing
                acc[i] += v
            elseif v !== nothing
                acc[i] = v
            end
        end
    end
    acc
end

function report()
srcs = sort(filter(f -> endswith(f, ".jl"), readdir(SRC)))
total_cov = 0; total_exe = 0
gaps = Dict{String,Vector{Int}}()

@printf("%-24s %8s %8s %7s\n", "file", "covered", "total", "pct")
println("-"^52)
for f in srcs
    acc = merged(f)
    acc === nothing && continue
    exe = count(!isnothing, acc)
    cov = count(x -> x !== nothing && x > 0, acc)
    exe == 0 && continue
    total_cov += cov; total_exe += exe
    pct = 100cov / exe
    @printf("%-24s %8d %8d %6.1f%%\n", f, cov, exe, pct)
    miss = [i for (i, v) in enumerate(acc) if v !== nothing && v == 0]
    isempty(miss) || (gaps[f] = miss)
end
println("-"^52)
@printf("%-24s %8d %8d %6.1f%%\n", "TOTAL", total_cov, total_exe,
        100total_cov / max(total_exe, 1))

println("\n\n=== uncovered lines ===")
for f in sort(collect(keys(gaps)))
    lines = readlines(joinpath(SRC, f))
    println("\n--- $f (", length(gaps[f]), " uncovered) ---")
    # collapse runs of consecutive uncovered lines into blocks
    miss = gaps[f]
    i = 1
    while i <= length(miss)
        j = i
        while j < length(miss) && miss[j+1] == miss[j] + 1
            j += 1
        end
        a, b = miss[i], miss[j]
        for l in a:min(b, a + 5)
            @printf("  %4d  %s\n", l, rstrip(get(lines, l, "")))
        end
        b > a + 5 && @printf("  %4d..%d  (%d more)\n", a + 6, b, b - a - 5)
        i = j + 1
    end
end
end

report()
