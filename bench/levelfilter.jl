# N12a qualification driver. Run each policy in a fresh Julia process:
#   julia --project=. -t 1 bench/levelfilter.jl policy=normalized instrument=budgets \
#       N=96 ny=24 tfinal=50.26548245743669 moving_tfinal=50.26548245743669 samples=16
#   julia --project=. -t 1 bench/levelfilter.jl policy=cadence instrument=accuracy
#   julia --project=. -t 1 bench/levelfilter.jl policy=default instrument=sensors \
#       parts=undershoot undershoot_ghosts=on
# Remaining arguments belong to the original instrument, including its smoke
# switches and completion/budget checks. No package default is changed.

using CompactLES
using MPI

let
    own = filter(a -> startswith(a, "policy=") || startswith(a, "instrument="), ARGS)
    opts = CompactLES.script_args(own, (policy="default", instrument="budgets"))
    instruments = Dict("budgets" => "interfaceconservation.jl",
                       "sensors" => "interfacesensor.jl",
                       "accuracy" => "level_filter_accuracy.jl")
    haskey(instruments, opts.instrument) ||
        error("instrument must be budgets, sensors, or accuracy")
    filter!(a -> !(startswith(a, "policy=") || startswith(a, "instrument=")), ARGS)
    include(joinpath(@__DIR__, "level_filter_policy.jl"))
    LevelFilterPolicy.set_policy!(Symbol(opts.policy))
    MPI.Initialized() || MPI.Init(threadlevel=:funneled)
    MPI.Comm_rank(MPI.COMM_WORLD) == 0 &&
        println("N12a benchmark-only filter policy: ", opts.policy)
    include(joinpath(@__DIR__, instruments[opts.instrument]))
end
