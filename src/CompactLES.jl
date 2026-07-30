"""
    CompactLES

Compressible large-eddy/direct-simulation solver using high-order compact
finite differences, compact filtering, five-stage fourth-order low-storage
Runge--Kutta time integration,
shared-memory threading, and MPI domain decomposition. The compact line solves
remain globally coupled across decomposed dimensions.

The public frontend separates a pointwise [`Problem`](@ref) from its
[`Numerics`](@ref); [`setup`](@ref) constructs the distributed solver and
initialized conserved state.
"""
module CompactLES

using MPI
using LinearAlgebra
using Printf

include("threading.jl")
include("decomposition.jl")
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
include("metric.jl")
include("folds.jl")
include("artificial.jl")
include("stepcontrol.jl")
include("sources.jl")
include("rhs.jl")
include("nscbc.jl")
include("io.jl")
include("hdf5.jl")
include("callbacks.jl")
include("timestep.jl")
include("diagnostics.jl")
include("viz.jl")
include("problem.jl")
include("scriptargs.jl")
include("display.jl")

export Decomp, ConservedState, exchange_halos!, interior, field, allocate_state
export CompactScheme, ClosureRow, lele_d1_6, pade_d1_4, compact_filter
export BandedCompactScheme, BandedClosureRow, lele_d1_10
export BoundaryCondition, PeriodicBC, SlipWallBC, NoSlipWallBC, ExtrapolationBC, AxisBC, OriginBC, PoleBC
export NSCBCOutflowBC, NSCBCInflowBC, DirichletBC, save_checkpoint, load_checkpoint!, save_vtk
export FieldWriter, DEFAULT_VTK_FIELDS, container_extension
export BlockRegion, hdf5_available, hdf5_parallel
export save_checkpoint_hdf5, load_checkpoint_hdf5!, save_hdf5
export SwitchableBC, switch!, switched
export Prim, Problem, Numerics, setup, initialize!, conserved_from_prim, tanh_blend
export DirPlan, BandPlan, apply_along!, filter_field!
export EOS, IdealSpecies, IdealMixture, single_species, nspecies, Transport
export StiffenedGas, Nasa9Interval, Nasa9Species, Nasa9Mixture
export nasa9_constant_cp, read_nasa9
export EquationSet, NavierStokes1T
export Metric, CartesianMetric, CylindricalMetric, SphericalMetric
export Stretch, sine_cluster
export ArtParams, Solver
export ConstantBodyForce, add_source!, add_sources!
export Workspace, compute_rhs!, apply_bcs!, compute_dt, dt_report, step!, run!, mpi_main
export StepControl, SolverFailure, PLANCK_TIME, max_rate
export Trigger, AtTime, EveryStep, EveryTime, WhenState, Callback, ProgressLog, rewind!
export refresh_primitives!, mixture_density, velocity, total_energy, mass_fraction
export boundary_plane
export volume_integral, volume_average, domain_volume, plane_profile
export profile_coordinate, profile_spacing
export field_array, scalar_field, line_profile, field_slice, cartesian_slice
export revolve_profile
export profileplot, profileplot!, fieldheatmap, fieldheatmap!, makie_available
export mix_width, molecular_mixing, species_pdf
export tke_profile, turbulent_kinetic_energy, dissipation_rate
export xcoord, global_xcoord, gidx, interior_index, filter_state!
export THREAD_MIN_WORK
export script_args, script_grid

__init__() = __init_threading__()

# Precompilation, kept deliberately to the signatures that cannot multiply.
#
# The cost this addresses is that every distinct `Solver{...}` type forces a
# fresh compilation of `deriv_along!` and the whole call tree below it, and a
# test process builds a dozen of them. `--trace-compile` names only the entry
# point, so the tree is invisible there, but it is where the time goes: a
# multi-rank test run is 44-74% compilation, paid independently by every rank.
#
# Only the shared floor of that tree is listed. `apply_along!` and the halo
# exchanges take an array and a plan, never a Metric, EOS, or BoundaryCondition,
# so there are exactly two plan types and one element type and the list cannot
# grow with the number of physics configurations. Compiling them into the
# package image leaves each `Solver` specialization with the thin wrapper above
# them rather than the line solves themselves.
#
# Nothing here executes: `@compile_workload` would need a communicator, and MPI
# calls during precompilation are not allowed. These are signature-directed, so
# they cost only their own compilation.
#
# Do not extend this list to entry points that take a `Solver`. Those are
# combinatorial in Metric x EOS x BoundaryCondition x scheme, and precompiling
# any fixed subset of them mostly bloats the image for configurations a given
# run never builds.
let A3 = Array{Float64,3}
    for P in (DirPlan{Float64}, BandPlan{Float64})
        precompile(apply_along!, (A3, P, A3, Decomp))
    end
    precompile(exchange_halos!, (A3, Decomp))
    precompile(exchange_dim_batch!, (Vector{A3}, Decomp, Int))
    precompile(field, (Decomp,))
end

end # module
