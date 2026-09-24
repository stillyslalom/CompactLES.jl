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
using Test

@testset "CompactLES serial suite" verbose=true begin
    include("serial_suite.jl")
end

println("serial tests complete")
