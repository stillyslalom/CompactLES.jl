# The serial test suite's body, run by test/runtests.jl inside the suite's one
# top-level testset (see the note there). This file loads the package and
# defines the imports and helpers the suite shares; the tests are in the files
# of test/serial/, one per subsystem, and in the subsystem files included after
# them. Ordered so failures localize: solvers → operators → closures/folds →
# metric → full RHS. Multi-rank checks live in mpi_tests.jl.

using MPI
MPI.Init(threadlevel=:funneled)
@isdefined(build_solver) || include(joinpath(@__DIR__, "build_solver.jl"))
@isdefined(use_test_image) || include(joinpath(@__DIR__, "test_image.jl"))
using CompactLES
using CompactLES: validate_bc, sensor_mirror, isperiodic, state_admissibility, EquationSet,
                  NavierStokes1T, rewind!, compute_rhs!, apply_bcs!, filter_state!,
                  max_rate, step!, npatches, ConservedState, interior_index, padded_index,
                  xcoord, global_xcoord, StateReport, CompactScheme, BandedCompactScheme,
                  pade_d1_4, compact_d8
import CompactLES: Decomp, exchange_halos!, interior, field, DirPlan, BandPlan,
                   DevicePlan, device_plan, apply_along!, filter_field!,
                   amr_transfer_schemes, amr_restriction_scheme,
                   amr_prolongation_scheme, amr_interpolation_weights,
                   TransferPlan, plan_transfer, restrict!, prolong!,
                   plan_direction, Patch, PatchSolver, InterfaceBC, CoarseFineBC,
                   LevelTransfer, Level, LevelComm, exchange_patch_ghosts!,
                   average_shared_planes!, prolong_level_ghosts!, restrict_level!,
                   scalar_field, container_extension, PLANCK_TIME,
                   THREAD_MIN_WORK, script_args, script_grid
using Test, LinearAlgebra, Random

const CL = CompactLES
Random.seed!(7)

mkslv(; n_global, deriv=lele_d1_6(), eos=IdealSpecies("gas"; R=1.0, gamma=1.4),
      sources=(), transport=ConstantTransport(), control=StepControl()) =
    build_solver(Problem(domain=((0.0, 2π), (0.0, 2π), (0.0, 2π)), bcs=per3, eos=eos,
                         transport=transport, sources=sources, ic=nothing),
                 Numerics(n_global=n_global, deriv=deriv, control=control,
                          art=ArtificialProperties(enabled=false)))

"Max interior error of a scalar field against an analytic function."
function ferr(solver, f, fn)
    e = 0.0
    for k in 1:solver.decomp.n_local[3], j in 1:solver.decomp.n_local[2], i in 1:solver.decomp.n_local[1]
        e = max(e, abs(f[padded_index(solver, i, j, k)] -
                       fn(xcoord(solver, 1, i), xcoord(solver, 2, j), xcoord(solver, 3, k))))
    end
    e
end

fillf!(solver, f, fn) = (for k in 1:solver.decomp.n_local[3], j in 1:solver.decomp.n_local[2],
                        i in 1:solver.decomp.n_local[1]
    f[padded_index(solver, i, j, k)] = fn(xcoord(solver, 1, i), xcoord(solver, 2, j),
                                          xcoord(solver, 3, k))
end; f)

# Analytic references (exact Riemann solver, Noh, Sedov) live in one place so
# the serial suite and test/validation.jl measure against the same solution.
include("references.jl")

# The shock-capturing case setups, shared with test/validation.jl and
# bench/artcal.jl. This suite runs only the cheap ones; the battery in
# validation.jl runs them at their reference resolutions. `per3`, used
# throughout below, is defined there.
include("cases.jl")

include("serial/frontend.jl")
include("serial/float32.jl")
include("serial/operators.jl")
include("serial/implicit.jl")
include("serial/closures_filters.jl")
include("serial/transfer.jl")
include("serial/geometry.jl")
include("serial/rhs_boundaries.jl")
include("serial/artificial.jl")
include("serial/step_control.jl")
include("serial/thermodynamics.jl")
include("serial/output.jl")
include("serial/runtime.jl")
include("serial/kernels.jl")

include("wall_flux_tests.jl")
include("composite_face_tests.jl")
include("transport_tests.jl")
include("binary_diffusion_tests.jl")
include("neutral_diffusion_tests.jl")
include("transport_integration_tests.jl")
include("neutral_transport_integration_tests.jl")
include("neutral_transport_coefficients_tests.jl")
include("neutral_transport_domain_tests.jl")
include("ionmix_tests.jl")
include("sesame_tests.jl")
include("device_tests.jl")
include("patch_tests.jl")
include("level_tests.jl")
include("substep_rate_tests.jl")
include("conservation_tests.jl")
include("interface_reflection_tests.jl")
include("sharpening_tests.jl")
include("positivity_tests.jl")
include("seam_tests.jl")
include("io_tests.jl")
include("runloop_tests.jl")
include("phase_tests.jl")
include("incompressible_tests.jl")
include("docrefs_tests.jl")
# The scripts under bench/, examples/, probes/ and docs/literate/, which no
# testset runs, are parsed for CompactLES names and keywords that do not exist.
include("scriptrefs_tests.jl")
include("reference_tests.jl")
include("api_surface_tests.jl")
include("pointwise_callbacks_tests.jl")
include("amr_frontend_tests.jl")
include("initial_states_tests.jl")
include("turbulent_inflow_tests.jl")
include("boundary_shorthand_tests.jl")
include("capability_tests.jl")
include("ablation_tests.jl")

# The extension suites run only where their weak dependency loads. A skip is
# recorded as a broken test, so the summary tree shows it in its own column
# rather than as a pass, and `require=hdf5,makie` (a test argument:
# `Pkg.test(test_args=["require=hdf5"])`, or on the command line) turns the
# skip of a named suite into a failure, for a job that expects the suite to
# run. `timing` is read by runtests.jl and listed here so that it parses.
const SUITE_OPTS = script_args(ARGS, (require = "", timing = false))
const REQUIRED_SUITES = Symbol.(filter(!isempty, split(SUITE_OPTS.require, ',')))

# HDF5 is a weak dependency and is not loadable from the package environment
# alone; the test target carries it, so `Pkg.test` runs the suite.
hdf5_loaded = try
    @eval using HDF5
    true
catch
    false
end
if hdf5_loaded
    include("hdf5_tests.jl")
else
    println("HDF5 not loadable in this environment — extension tests SKIPPED. " *
            "Run test/hdf5_tests.jl from an environment carrying both, and " *
            "under mpiexec for the decomposition-independent restart.")
    @testset "HDF5 extension: skipped" begin
        @test hdf5_loaded skip=!(:hdf5 in REQUIRED_SUITES)
    end
end

# A Makie backend is likewise a weak dependency, absent from the package
# environment. The extraction API it exercises is exported by the core, so this
# skip only foregoes the extension's plotting methods and the collective-profile
# check under decomposition.
#
# Unlike HDF5, CairoMakie is not in Project.toml's test target: it
# is a heavy dependency to resolve and precompile for two testsets, so
# `Pkg.test` never reaches this branch. The Makie extension is verified from
# the docs environment, which carries CairoMakie already; CI's documentation
# job runs test/makie_tests.jl there.
makie_loaded = try
    @eval using CairoMakie
    true
catch
    false
end
if makie_loaded
    include("makie_tests.jl")
else
    println("Makie backend not loadable in this environment — extension tests " *
            "SKIPPED. CairoMakie is not in the test target, so Pkg.test always " *
            "skips these: run test/makie_tests.jl from the docs environment, " *
            "and under mpiexec for the decomposition-independent profile.")
    @testset "Makie extension: skipped" begin
        @test makie_loaded skip=!(:makie in REQUIRED_SUITES)
    end
end


# Briareus (the view-factor hohlraum code that drives `AblationDrive`) is not
# registered, so it is in no test target: run test/briareus_ext_tests.jl from an
# environment that develops both packages.
briareus_loaded = try
    @eval using Briareus
    true
catch
    false
end
if briareus_loaded
    include("briareus_ext_tests.jl")
else
    println("Briareus not loadable in this environment — extension tests SKIPPED. " *
            "Run test/briareus_ext_tests.jl from an environment carrying both.")
    @testset "Briareus extension: skipped" begin
        @test briareus_loaded skip=!(:briareus in REQUIRED_SUITES)
    end
end
