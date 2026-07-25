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
include("equations.jl")
include("boundary.jl")
include("operators.jl")
include("operators_banded.jl")
include("metric.jl")
include("folds.jl")
include("artificial.jl")
include("sources.jl")
include("rhs.jl")
include("nscbc.jl")
include("io.jl")
include("timestep.jl")
include("problem.jl")

export Decomp, exchange_halos!, interior, field, allocate_state
export CompactScheme, ClosureRow, lele_d1_6, pade_d1_4, compact_filter
export BandedCompactScheme, BandedClosureRow, lele_d1_10
export BoundaryCondition, PeriodicBC, SlipWallBC, NoSlipWallBC, ExtrapolationBC, AxisBC, OriginBC, PoleBC
export NSCBCOutflowBC, NSCBCInflowBC, DirichletBC, save_checkpoint, load_checkpoint!, save_vtk
export Prim, Problem, Numerics, setup, initialize!, conserved_from_prim, tanh_blend
export DirPlan, BandPlan, apply_along!, filter_field!
export EOS, IdealSpecies, IdealMixture, single_species, nspecies, Transport
export EquationSet, NavierStokes1T
export Metric, CartesianMetric, CylindricalMetric, SphericalMetric
export Stretch, sine_cluster
export ArtParams, Solver
export ConstantBodyForce, add_source!, add_sources!
export Workspace, compute_rhs!, apply_bcs!, compute_dt, dt_report, step!, run!
export xcoord, gidx, filter_state!
export THREAD_MIN_WORK

__init__() = __init_threading__()

end # module
