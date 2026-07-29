# Phase timing for the long-running scripts.
#
# `test/convergence.jl` and `test/mpi_tests.jl` run for minutes on a CI runner,
# and a job that reports only its total gives no way to separate process
# startup and compilation from the numerics the script exists to check. Both
# scripts wrap their studies in `@phase` and print `timing_report()` at the
# end, so every run carries its own cost breakdown and a regression appears as
# a line item rather than as a slower job.
#
# The compile column comes from `Base.cumulative_compile_time_ns`, which counts
# the time the compiler itself was running. That is the quantity to watch: a
# phase whose cost is nearly all compilation is paying for code generation, not
# for the calculation, and is made cheaper by running fewer distinct
# specializations rather than by shrinking the grid.
#
# The counter sums compiler work over threads, so with more than one thread a
# phase can report more compile time than wall time. Read the column as
# compiler work performed, not as a fraction of the phase.
#
# Include this first, before `using`, so the package load lands inside a phase
# instead of ahead of the first measurement.

using Printf

const T_SCRIPT_START = time()
const PHASE_LOG = Tuple{String,Float64,Float64}[]

# Compile-time accounting is off by default and costs a little to keep on, which
# is why `@time` switches it on only around its own body. A whole script is a
# long enough window that the overhead does not register.
if isdefined(Base, :cumulative_compile_timing)
    Base.cumulative_compile_timing(true)
end

"Nanoseconds spent compiling so far in this process, 0 if unavailable."
function compile_ns()
    isdefined(Base, :cumulative_compile_time_ns) || return UInt64(0)
    t = Base.cumulative_compile_time_ns()
    t isa Tuple ? t[1] : t
end

"""
    @phase "name" expr

Evaluate `expr`, recording its wall time and the share of it spent compiling.
Expands to a plain block so that `using` and other top-level-only forms stay
legal inside it.
"""
macro phase(name, ex)
    quote
        local t0 = time()
        local c0 = compile_ns()
        $(esc(ex))
        push!(PHASE_LOG, ($(esc(name)), time() - t0,
                          (compile_ns() - c0) / 1e9))
    end
end

"""
    timing_report(io = stdout; title = "phase timing", extra = ())

Print the recorded phases longest-first, with the compile share of each. `extra`
is a list of `name => seconds` pairs appended below the total, which the MPI
suite uses to report the spread across ranks.

Unaccounted time is the difference between the script's own elapsed time and
the sum of its phases. A large value means work is happening outside any
`@phase`, which is a gap in the instrumentation rather than a finding.
"""
function timing_report(io::IO = stdout; title = "phase timing", extra = ())
    total = time() - T_SCRIPT_START
    compile = sum(p -> p[3], PHASE_LOG; init = 0.0)
    phased = sum(p -> p[2], PHASE_LOG; init = 0.0)
    println(io, "\n=== $title ===")
    println(io, "  phase                                     wall     compiling")
    for (name, wall, comp) in sort(PHASE_LOG; by = p -> -p[2])
        pct = wall > 0 ? round(Int, 100 * comp / wall) : 0
        @printf(io, "  %-38s %7.2f s  %7.2f s (%3d%%)\n", name, wall, comp, pct)
    end
    @printf(io, "  %-38s %7.2f s\n", "unaccounted", total - phased)
    @printf(io, "  %-38s %7.2f s  %7.2f s (%3d%%)\n", "TOTAL (script)", total,
            compile, total > 0 ? round(Int, 100 * compile / total) : 0)
    for (name, seconds) in extra
        @printf(io, "  %-38s %7.2f s\n", name, seconds)
    end
    println(io)
    return total
end
