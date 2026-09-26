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

The `MPI` module is re-exported, so an input deck needs only CompactLES in its
environment: `using CompactLES` is enough to call `MPI.Init` and to pass
`MPI.COMM_WORLD` to [`Numerics`](@ref).
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
include("transport.jl")
include("neutral_diffusion_data.jl")
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
include("precision.jl")
include("patches.jl")
include("levels.jl")
include("solver.jl")
include("construction.jl")
include("rhs.jl")
include("pointwise_callbacks.jl")
include("transport_domain.jl")
include("nscbc.jl")
include("io.jl")
include("hdf5.jl")
include("callbacks.jl")
include("timestep.jl")
include("regrid.jl")
include("amr_frontend.jl")
include("io_levels.jl")
include("diagnostics.jl")
include("viz.jl")
include("problem.jl")
include("thermo_states.jl")
include("regions.jl")
include("scriptargs.jl")
include("display.jl")

# MPI is re-exported so that an input deck needs one package in its environment
# and one `using` line: `using CompactLES` binds `MPI` for `MPI.Init`,
# `MPI.COMM_WORLD`, and the rest of the qualified MPI API. Only the module name
# is re-exported; MPI.jl's own exports (`mpiexec`, `UBuffer`, `VBuffer`) still
# need `using MPI` or an `MPI.` prefix.
export MPI

# Common input-deck and runtime surface. Lower-level decomposition, directional
# plan, transfer, and hierarchy records remain supported through qualified
# `CompactLES.name` access and are documented as the advanced API. The region
# shapes and the vendored diffusion tables are in the submodules `Regions` and
# `DiffusionData`, loaded with their own `using`, since several of their names
# are common in plotting and geometry packages.
export allocate_state
export lele_d1_6, lele_d1_8, compact_filter
export gaussian_filter
export lele_d1_10, pyranda_filter
export BoundaryCondition, PeriodicBC, SlipWallBC, NoSlipWallBC
export ExtrapolationBC, AxisBC, OriginBC, PoleBC, SymmetryPlaneBC
export NSCBCOutflowBC, NSCBCInflowBC, DirichletBC, save_checkpoint, load_checkpoint!, save_vtk
export FieldWriter
export BlockRegion
export save_checkpoint_hdf5, load_checkpoint_hdf5!, save_hdf5
export SwitchableBC, switch!, switched, CompositeBC
export Prim, Problem, Numerics, AMR, setup, initialize!, conserved_from_prim, tanh_blend
export EOS, IdealSpecies, IdealMixture, nspecies, ConstantTransport
export AbstractTransport, BinaryDiffusion, BinaryDiffusionPolynomial, binary_diffusivity
export CeaTransport, read_cea_transport, transport_coefficients
export neutral_binary_diffusion, temperature_domain
export StiffenedGas, Nasa9Interval, Nasa9Species, Nasa9Mixture
export nasa9_constant_cp, read_nasa9
export species_names
export CartesianMetric, CylindricalMetric, SphericalMetric
export Stretch, sine_cluster
export ArtificialProperties, Solver
export DeviceBackend
export nlevels, refined_region, level_regions
export ConstantBodyForce
export Workspace, compute_dt, dt_report, run!, mpi_main
export StepControl, SolverFailure
export state_report, state_valid, validate_state!
export StateGuard, state_guard
export Trigger, AtTime, EveryStep, EveryTime, WhenState, Callback, ProgressLog
export refresh_primitives!, mixture_density, velocity, total_energy, mass_fraction
export boundary_plane
export volume_integral, volume_average, domain_volume, plane_profile
export profile_coordinate, profile_spacing
export field_array, line_profile, line_sample, field_slice, cartesian_slice
export field_snapshot, cartesian_coordinates
export revolve_profile
export profileplot, profileplot!, fieldheatmap, fieldheatmap!
export mix_width, molecular_mixing, species_pdf
export tke_profile, turbulent_kinetic_energy, dissipation_rate
export thermodynamic_state, mass_fractions, mole_fractions
export shock_jump, driver_pressure, reflected_shock, shock_tube
export Cells, Hydrostatic, Ramp, Multimode, riemann_interface
export TurbulentInflow

# Supported names reached as `CompactLES.name`: the submodules, the hooks a new
# boundary condition, equation of state, trigger or source implements, the
# runtime and storage internals a hand-written driver calls, and the scheme
# constructors. `public` is Julia 1.11 syntax, so it is parsed only there.
@static if VERSION >= v"1.11.0-DEV.469"
    eval(Meta.parse("""public Regions, DiffusionData,
        enforce!, correct_flux!, correct_rhs!, validate_bc, sensor_mirror, isperiodic,
        eos_phi, eos_dphi_dY, artificial_conductivity_scale, wall_internal_energy,
        state_admissibility, conserved_parity, species_enthalpy,
        Metric, EquationSet, NavierStokes1T,
        fired!, next_time, rewind!, fires_at_start, add_source!,
        compute_rhs!, apply_bcs!, recover_primitives!, filter_state!, max_rate,
        FloorTally, step!, sync_patches!, sync_levels!, eachpatch, npatches,
        ConservedState, CPUBackend, interior_index, padded_index, xcoord, global_xcoord,
        DEFAULT_VTK_FIELDS, StateReport, FieldSnapshot,
        makie_available, hdf5_available, hdf5_parallel,
        ClosureRow, BandedClosureRow, CompactScheme, BandedCompactScheme,
        pade_d1_4, compact_d8"""))
end

__init__() = (__init_threading__(); __init_blas__())

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
