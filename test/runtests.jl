# Serial test suite (run with: julia --project=. -O1 test/runtests.jl, or
# under one MPI rank). The suite itself is test/serial_suite.jl.
#
# One top-level testset holds the whole suite, the included files' testsets
# too (nesting is dynamic, so a testset created inside an `include` call made
# from here reports under this one). A failure is then recorded and the suite
# runs on to report every failure in one pass, with the outer testset raising
# at the end; a bare top-level testset would raise at its own end and stop
# the file there.
#
# The body is included rather than written inside the testset: `include`
# lowers and runs a file one top-level statement at a time, where a body
# written inline is one expression of some 65,000 statements, and lowering
# that alone measured 33.5 s against 1.15 s for the statements separately.
#
# The suite is compile-bound, so the documented command and CI's serial job
# run it at -O1, which reuses the package's -O2 image without a rebuild and
# roughly halves the suite's compile time. The numerical suites and the MPI
# suite stay at the default -O2.
#
# The summary tree gives each testset's wall time but not how much of it was
# compilation, which is most of it in this suite. The test argument
# `timing=true` wraps every top-level testset of the included files in
# `@phase` (test/timing.jl) and prints the wall and compile table of
# convergence.jl and mpi_tests.jl at the end, longest first. It is off by
# default because the table runs to some 300 lines.
using Test
using CompactLES: script_args

const RUN_OPTS = script_args(ARGS, (require = "", timing = false))
RUN_OPTS.timing && include("timing.jl")

# Wraps each top-level `@testset` in `@phase` under its own name and passes the
# wrapper on to every file a file includes, through an `include` written
# inside a top-level `if` or `||` as well.
function timed_testsets(ex)
    ex isa Expr || return ex
    if ex.head === :macrocall && ex.args[1] === Symbol("@testset")
        # An interpolated name is reported as written: a loop testset's
        # variable is not defined where the wrapper evaluates the name.
        names = [a isa String ? a : strip(string(a), '"') for a in ex.args[3:end]
                 if a isa String || (a isa Expr && a.head === :string)]
        name = isempty(names) ? "testset at $(ex.args[2])" : first(names)
        return Expr(:macrocall, Symbol("@phase"), ex.args[2], name, ex)
    elseif ex.head === :call && ex.args[1] === :include && length(ex.args) == 2
        return Expr(:call, :include, timed_testsets, ex.args[2])
    elseif ex.head in (:block, :if, :elseif, :||, :&&, :toplevel)
        return Expr(ex.head, map(timed_testsets, ex.args)...)
    end
    return ex
end

@testset "CompactLES serial suite" verbose=true begin
    if RUN_OPTS.timing
        include(timed_testsets, "serial_suite.jl")
    else
        include("serial_suite.jl")
    end
end

RUN_OPTS.timing && timing_report(; title="serial suite timing")

println("serial tests complete")
