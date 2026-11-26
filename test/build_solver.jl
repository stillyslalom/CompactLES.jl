# A solver of a `Problem` under `Numerics`, without a state: the form the test
# suites, the test-only precompile workload (test/precompile_workload.jl) and
# the benchmark scripts build their configurations in, then fill a state with
# `allocate_state` and `initialize!`. `setup` is the entry point of a run.
#
# Included by whichever file needs it, the test image's module among them, so
# every name is qualified and nothing is imported.

"""
    build_solver(prob, num; tag_predicate = nothing, equations = nothing) -> Solver

Build the solver `setup(prob, num)` would build, without allocating or
validating a state, so `prob.ic` may be `nothing`. It differs from `setup` in
two further ways the suites rely on. A `num.control` of `nothing` becomes
`StepControl()`, not the problem's default, which adds retries at a spherical
origin. The refinement is `num.amr` as given (`LevelSetup(num.amr)`): a region
list builds its levels at once, with no initial tagging pass. `tag_predicate`
is a `(patch, I) -> Bool` closure over the parent patch, and `equations` the
`EquationSet` owning the conserved layout.
"""
function build_solver(prob::CompactLES.Problem, num::CompactLES.Numerics;
                      tag_predicate=nothing, equations=nothing)
    C = CompactLES
    num.control === nothing && (num = C.Numerics(num; control=C.StepControl()))
    lv = num.amr === nothing ? C.LevelSetup(; tag_predicate) :
                               C.LevelSetup(num.amr; tag_predicate)
    return C._build_solver(prob, num, lv; equations)
end
