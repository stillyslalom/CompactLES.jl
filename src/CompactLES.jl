"""
    CompactLES

Compressible large-eddy/direct-simulation solver using high-order compact
finite differences, compact filtering, five-stage fourth-order low-storage
Runge–Kutta time integration, shared-memory threading, and MPI domain
decomposition. The compact line solves remain globally coupled across
decomposed dimensions.

The public frontend separates a pointwise [`Problem`](@ref) from its
[`Numerics`](@ref); [`setup`](@ref) returns the distributed solver and the
initialized conserved state.
"""
module CompactLES

using MPI
using LinearAlgebra
using Printf

# `clusterprobe.jl` accesses `ThreadPinning` through CompactLES because its
# driver environment lists only this package. The solver itself does not use
# thread-pinning queries. `import` keeps exported names such as `ncores` and
# `nsockets` out of this namespace.
import ThreadPinning

include("threading.jl")
include("decomposition.jl")
include("pointwise.jl")
include("halo.jl")
include("tridiag.jl")
include("banded.jl")
include("lines_transposed.jl")
include("kernels.jl")
include("kernels_banded.jl")
include("physics.jl")
include("nasa9_data.jl")
include("equations.jl")
include("boundary.jl")
include("operators.jl")
include("operators_banded.jl")
include("lines_device.jl")
include("transfer.jl")
include("metric.jl")
include("folds.jl")
include("artificial.jl")
include("stepcontrol.jl")
include("sources.jl")
include("patches.jl")
include("levels.jl")
include("rhs.jl")
include("nscbc.jl")
include("io.jl")
include("hdf5.jl")
include("callbacks.jl")
include("timestep.jl")
include("regrid.jl")
include("io_levels.jl")
include("diagnostics.jl")
include("viz.jl")
include("problem.jl")
include("scriptargs.jl")
include("display.jl")

# Common input-deck and runtime surface. Lower-level decomposition, directional
# plan, transfer, and hierarchy records remain supported through qualified
# `CompactLES.name` access and are documented as the advanced API.
export ConservedState, allocate_state
export CompactScheme, ClosureRow, lele_d1_6, lele_d1_8, pade_d1_4, compact_filter
export gaussian_filter
export BandedCompactScheme, BandedClosureRow, lele_d1_10, compact_d8
export BoundaryCondition, PeriodicBC, SlipWallBC, NoSlipWallBC
export ExtrapolationBC, AxisBC, OriginBC, PoleBC
export enforce!, correct_rhs!, validate_bc, isperiodic
export NSCBCOutflowBC, NSCBCInflowBC, DirichletBC, save_checkpoint, load_checkpoint!, save_vtk
export FieldWriter, DEFAULT_VTK_FIELDS
export BlockRegion, hdf5_available, hdf5_parallel
export save_checkpoint_hdf5, load_checkpoint_hdf5!, save_hdf5
export SwitchableBC, switch!, switched
export Prim, Problem, Numerics, setup, initialize!, conserved_from_prim, tanh_blend
export EOS, IdealSpecies, IdealMixture, nspecies, Transport
export StiffenedGas, Nasa9Interval, Nasa9Species, Nasa9Mixture
export nasa9_constant_cp, read_nasa9
export recover_primitives!, species_names, species_enthalpy
export eos_phi, eos_dphi_dY, artificial_conductivity_scale, wall_internal_energy
export EquationSet, NavierStokes1T
export conserved_parity
export Metric, CartesianMetric, CylindricalMetric, SphericalMetric
export Stretch, sine_cluster
export ArtParams, Solver
export CPUBackend, DeviceBackend
export npatches, eachpatch, sync_patches!
export nlevels, refined_region, level_regions, sync_levels!
export ConstantBodyForce, add_source!
export Workspace, compute_rhs!, apply_bcs!, compute_dt, dt_report, step!, run!, mpi_main
export StepControl, SolverFailure, max_rate, FloorTally
export Trigger, AtTime, EveryStep, EveryTime, WhenState, Callback, ProgressLog
export fired!, next_time, rewind!
export refresh_primitives!, mixture_density, velocity, total_energy, mass_fraction
export boundary_plane
export volume_integral, volume_average, domain_volume, plane_profile
export profile_coordinate, profile_spacing
export field_array, line_profile, line_sample, field_slice, cartesian_slice
export revolve_profile
export profileplot, profileplot!, fieldheatmap, fieldheatmap!, makie_available
export mix_width, molecular_mixing, species_pdf
export tke_profile, turbulent_kinetic_energy, dissipation_rate
export xcoord, global_xcoord, gidx, interior_index, filter_state!

__init__() = __init_threading__()

# Precompilation. Two mechanisms: the signature-directed statements below, and
# the executed workload in precompile.jl.
#
# The statements cover the shared floor of every `Solver` specialization.
# `apply_along!` and the halo exchanges take an array and a plan, never a
# Metric, EOS, or BoundaryCondition, so there are exactly two plan types and
# one element type and this list cannot grow with the number of physics
# configurations. They are signature-directed, so they cost only their own
# compilation and need no communicator. Do not extend this list to entry
# points that take a `Solver`: those are combinatorial in Metric x EOS x
# BoundaryCondition x scheme, and a fixed subset of them as bare statements
# bloats the image for configurations a run never builds.
#
# The workload compiles call trees keyed on the `Solver` type. It runs the
# configurations built by the test suites, producing a curated subset instead
# of a combinatorial one. Its cost, measured effect, and system-MPI exclusion
# are documented in precompile.jl.
let A3 = Array{Float64,3}, D = Decomp{Float64}
    for P in (DirPlan{Float64}, BandPlan{Float64})
        precompile(apply_along!, (A3, P, A3, D))
    end
    precompile(exchange_halos!, (A3, D))
    precompile(exchange_dim_batch!, (Vector{A3}, D, Int))
    precompile(field, (D,))
end

include("precompile.jl")

end # module
