"""
    CompactLES

Draft compressible LES solver on structured Cartesian grids using Lele-style
compact (Padé) finite differences, Cook-style artificial fluid properties for
shock and subgrid regularization, a Gaitonde-Visbal compact filter, low-storage
RK45 time integration, shared-memory threading over grid lines, and MPI domain
decomposition with halo exchange plus a distributed (spike / reduced-interface)
tridiagonal solve for the globally coupled compact schemes.

Untested draft: written for MPI.jl v0.20; expect to shake out small issues on
first run (see README.md).
"""
module CompactLES

using MPI
using LinearAlgebra
using Printf

include("decomposition.jl")
include("halo.jl")
include("tridiag.jl")
include("banded.jl")
include("lines_transposed.jl")
include("kernels.jl")
include("kernels_banded.jl")
include("physics.jl")
include("boundary.jl")
include("operators.jl")
include("operators_banded.jl")
include("metric.jl")
include("folds.jl")
include("artificial.jl")
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
export Metric, CartesianMetric, CylindricalMetric, SphericalMetric
export Stretch, sine_cluster
export ArtParams, Solver
export compute_rhs!, apply_bcs!, compute_dt, dt_report, step!, run!, xcoord, gidx, filter_state!

end # module
