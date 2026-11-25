"""
    TestPrecompile

The test suites' precompile workload as a package of its own. The suites build
many more `Solver` types than a user's run does, and a multi-rank test run
compiles each of them independently on every rank. This package runs those
configurations (`test/precompile_workload.jl`) while it precompiles, so that
their compiled trees are in its image; a suite that loads it finds them there.
CompactLES's own image holds none of them, and CompactLES does not know this
package exists.

`test/test_image.jl` enables the package for a checkout, builds the image and
is what a suite includes to load it. The preferences are read at precompile
time from `test/TestPrecompile/LocalPreferences.toml`, which that script
writes:

- `device`: also the KernelAbstractions device path, reached on this machine
  by a `DeviceBackend(KernelAbstractions.CPU())` solver under `FORCE_KA` and
  `FORCE_DEVICE_EXCHANGE`. `test/device_tests.jl` and the MPI suite's device
  phases run it and nothing else does.
- `float32`: also the single-precision configurations.
- `mpi_suite`: also the configurations of the compile-bound phases of the
  multi-rank suite's eight-rank selection (`rank_shape=true` in
  `test/mpi_tests.jl`): the positivity limiter on each of its layouts, the
  symmetry planes, a refined level at a symmetry plane and on the r-z axis,
  the staggered diffusion operator and the implicit stage on their metrics,
  and the implicit-conduction pair. The block reproduces each phase's types,
  not its data; a phase whose configuration changes without a matching change
  there still runs, and compiles its tree on every rank.

A `Decomp` needs an initialized communicator, so the workload initializes MPI
as a singleton inside the precompile process, and is skipped under a system
MPI for the reason given in CompactLES's `src/precompile.jl`.
"""
module TestPrecompile

using CompactLES
using CompactLES: CPUBackend, Decomp, DiffusionStage, FORCE_DEVICE_EXCHANGE, FORCE_KA,
                  StaggeredDiffusion, _scale_grad!, _sync_level_records!, apply_along!,
                  apply_bcs!, compute_rhs!, deriv_along!, device_plan, eachpatch,
                  exchange_halos!, field, interior, plan_direction, positivity_counts,
                  positivity_floors, solve_stage!, stage_operator!, staggered_diffusion!,
                  with_knob, xcoord
using KernelAbstractions: KernelAbstractions
using MPI: MPI
using MPIPreferences: MPIPreferences
using PrecompileTools: @setup_workload, @compile_workload
using Preferences: @load_preference

const PRECOMPILE_DEVICE = @load_preference("device", false)
const PRECOMPILE_FLOAT32 = @load_preference("float32", false)
const PRECOMPILE_MPI_SUITE = @load_preference("mpi_suite", false)

include(joinpath(@__DIR__, "..", "..", "precompile_workload.jl"))

if MPIPreferences.binary != "system"
@setup_workload begin
    MPI.Initialized() || MPI.Init(threadlevel=:funneled)
    @compile_workload begin
        cpu = KernelAbstractions.CPU()
        _test_suite_workload(cpu)
        PRECOMPILE_MPI_SUITE && _mpi_suite_workload(cpu)
    end
end
end

end
