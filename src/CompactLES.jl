"""
    CompactLES

Compressible large-eddy/direct-simulation solver using high-order compact
finite differences, compact filtering, low-storage RK45 time integration,
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
include("callbacks.jl")
include("timestep.jl")
include("diagnostics.jl")
include("problem.jl")
include("scriptargs.jl")

export Decomp, exchange_halos!, interior, field, allocate_state
export CompactScheme, ClosureRow, lele_d1_6, pade_d1_4, compact_filter
export BandedCompactScheme, BandedClosureRow, lele_d1_10
export BoundaryCondition, PeriodicBC, SlipWallBC, NoSlipWallBC, ExtrapolationBC, AxisBC, OriginBC, PoleBC
export NSCBCOutflowBC, NSCBCInflowBC, DirichletBC, save_checkpoint, load_checkpoint!, save_vtk
export FieldWriter, DEFAULT_VTK_FIELDS, container_extension
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
export mix_width, molecular_mixing, species_pdf
export tke_profile, turbulent_kinetic_energy, dissipation_rate
export xcoord, gidx, filter_state!
export THREAD_MIN_WORK
export script_args, script_grid

__init__() = __init_threading__()

end # module
