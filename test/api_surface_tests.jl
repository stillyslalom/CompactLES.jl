# Focused checks for the user-facing thermodynamics and API surface.
#
# This file is included by test/runtests.jl after MPI has been initialized, but
# can also be run directly with `julia --project=. test/api_surface_tests.jl`.

using MPI
MPI.Initialized() || MPI.Init(threadlevel=:funneled)

using CompactLES
using Test

const API_BCS = ntuple(_ -> (PeriodicBC(), PeriodicBC()), 3)
const API_DOMAIN = ((0.0, 1.0), (0.0, 1.0), (0.0, 1.0))
api_ic(x, y, z) = Prim(rho=1.0, p=1.0)

# Deliberate compatibility manifest: changing an export is an API decision, not
# an incidental consequence of adding a binding to the module.
const EXPECTED_EXPORTS = Set(Symbol.(split("""
ArtParams AtTime AxisBC BandedClosureRow BandedCompactScheme BlockRegion
BoundaryCondition CPUBackend Callback CartesianMetric ClosureRow CompactScheme
ConservedState ConstantBodyForce CylindricalMetric DEFAULT_VTK_FIELDS
DeviceBackend DirichletBC EOS EquationSet EveryStep EveryTime ExtrapolationBC
FieldWriter FloorTally IdealMixture IdealSpecies Metric NSCBCInflowBC
NSCBCOutflowBC Nasa9Interval Nasa9Mixture Nasa9Species NavierStokes1T
NoSlipWallBC Numerics OriginBC PeriodicBC PoleBC Prim Problem ProgressLog
SlipWallBC Solver SolverFailure SphericalMetric StepControl StiffenedGas Stretch
SwitchableBC Transport Trigger WhenState Workspace add_source! allocate_state
apply_bcs! artificial_conductivity_scale boundary_plane cartesian_slice
compact_d8 compact_filter compute_dt compute_rhs! conserved_from_prim
conserved_parity correct_rhs! dissipation_rate domain_volume dt_report eachpatch
enforce! eos_dphi_dY eos_phi field_array field_slice fieldheatmap fieldheatmap!
filter_state! fired! gaussian_filter gidx global_xcoord hdf5_available
hdf5_parallel initialize! interior_index isperiodic lele_d1_10 lele_d1_6
lele_d1_8 level_regions line_profile line_sample load_checkpoint!
load_checkpoint_hdf5!
makie_available mass_fraction max_rate mix_width mixture_density molecular_mixing
mpi_main nasa9_constant_cp next_time nlevels npatches nspecies pade_d1_4
plane_profile profile_coordinate profile_spacing profileplot profileplot!
read_nasa9 recover_primitives! refined_region refresh_primitives! revolve_profile
rewind! run! save_checkpoint save_checkpoint_hdf5 save_hdf5 save_vtk setup
sine_cluster species_enthalpy species_names species_pdf step! switch! switched
sync_levels! sync_patches! tanh_blend tke_profile total_energy
turbulent_kinetic_energy validate_bc velocity volume_average volume_integral
wall_internal_energy xcoord
""")))

const ADVANCED_QUALIFIED_API = (
    :Decomp, :exchange_halos!, :interior, :field,
    :DirPlan, :BandPlan, :DevicePlan, :device_plan, :apply_along!, :filter_field!,
    :TransferPlan, :plan_transfer, :restrict!, :prolong!,
    :Patch, :PatchSolver, :Level, :LevelComm,
    :script_args, :script_grid,
)

const EXTENSION_API = (
    :recover_primitives!, :species_names, :species_enthalpy, :eos_phi,
    :eos_dphi_dY, :artificial_conductivity_scale, :wall_internal_energy,
    :conserved_parity, :enforce!, :correct_rhs!, :validate_bc, :isperiodic,
    :fired!, :next_time, :rewind!,
)

@testset "public API manifest" begin
    actual = Set(filter(!=(:CompactLES), names(CompactLES)))
    @test actual == EXPECTED_EXPORTS
    @test !isdefined(CompactLES, :single_species)

    for name in ADVANCED_QUALIFIED_API
        @test isdefined(CompactLES, name)
        @test name ∉ actual
    end

    for name in EXTENSION_API
        @test name ∈ actual
        @test Base.Docs.doc(getproperty(CompactLES, name)) !== nothing
    end
end

@testset "thermodynamics API" begin
    @testset "explicit IdealSpecies validation" begin
        gas = IdealSpecies("gas"; R=1.0, gamma=1.4)
        @test gas.name == "gas"
        @test gas.R == 1.0
        @test gas.gamma == 1.4
        @test_throws ArgumentError IdealSpecies("bad"; R=0.0, gamma=1.4)
        @test_throws ArgumentError IdealSpecies("bad"; R=1.0, gamma=1.0)
        @test_throws ArgumentError IdealSpecies("bad"; R=1.0)
        @test_throws ArgumentError IdealSpecies("bad"; gamma=1.4)
    end

    @testset "NASA-backed ideal species" begin
        co2 = IdealSpecies("CO2")
        nasa_co2 = read_nasa9("CO2"; reference=:formation)
        nasa_eos = Nasa9Mixture([nasa_co2])
        cp_ref = CompactLES.species_cp(nasa_eos, 1, 298.15)
        @test co2.name == "CO2"
        @test co2.R ≈ nasa_co2.R
        @test co2.gamma ≈ cp_ref / (cp_ref - co2.R)
        @test co2.gamma > 1
        @test_throws ArgumentError IdealSpecies("not-a-CEA-species")
        @test_throws ArgumentError IdealSpecies("H2O(L)")
    end

    @testset "database mixture convenience" begin
        mix = IdealMixture(["He", "CO2"])
        explicit = IdealMixture([IdealSpecies("He"), IdealSpecies("CO2")])
        @test [species.name for species in mix.sp] == ["He", "CO2"]
        @test mix.Rk ≈ explicit.Rk
        @test mix.cpk ≈ explicit.cpk
        @test IdealMixture(Float32, ("He", "CO2")).Rk isa Vector{Float32}

        nasa = Nasa9Mixture(["He", "CO2"])
        nasa_explicit = Nasa9Mixture(read_nasa9(["He", "CO2"]))
        @test [species.name for species in nasa.sp] == ["He", "CO2"]
        @test nasa.Rk ≈ nasa_explicit.Rk
        @test Nasa9Mixture(Float32, ("He", "CO2")).Rk isa Vector{Float32}
    end
end

@testset "one-species EOS promotion" begin
    species = IdealSpecies("gas"; R=1.0, gamma=1.4)
    problem = Problem(domain=API_DOMAIN, bcs=API_BCS, ic=api_ic, eos=species)
    @test problem.eos isa IdealMixture
    @test nspecies(problem.eos) == 1
    @test problem.eos.sp[1].name == "gas"
    @test problem.eos.sp[1].R == species.R
    @test problem.eos.sp[1].gamma == species.gamma

    solver = Solver(n_global=(12, 12, 12), L_domain=(1.0, 1.0, 1.0),
                    bcs=API_BCS, eos=species, comm=MPI.COMM_SELF,
                    dims=(1, 1, 1), art=ArtParams(enabled=false))
    @test solver.eos isa IdealMixture
    @test nspecies(solver.eos) == 1
end

@testset "default EOS compatibility" begin
    problem = Problem(domain=API_DOMAIN, bcs=API_BCS, ic=api_ic)
    @test problem.eos isa IdealMixture
    @test nspecies(problem.eos) == 1
    @test problem.eos.sp[1].name == "gas"
    @test problem.eos.sp[1].R == 1.0
    @test problem.eos.sp[1].gamma == 1.4
end

@testset "mutating plotting bindings carry documentation" begin
    @test Base.Docs.doc(CompactLES.profileplot!) !== nothing
    @test Base.Docs.doc(CompactLES.fieldheatmap!) !== nothing
end
