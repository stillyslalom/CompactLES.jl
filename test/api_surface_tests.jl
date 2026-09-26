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

# Deliberate compatibility manifest: changing an export or a `public`
# declaration is an API decision, not an incidental consequence of adding a
# binding to the module.
const EXPECTED_EXPORTS = Set(Symbol.(split("""
AMR AbstractTransport ArtificialProperties AtTime AxisBC BinaryDiffusion
BinaryDiffusionPolynomial BlockRegion BoundaryCondition Callback CartesianMetric
CeaTransport Cells CompositeBC ConstantBodyForce ConstantTransport CylindricalMetric
DeviceBackend DirichletBC EOS EveryStep EveryTime ExtrapolationBC FieldWriter
Hydrostatic IdealMixture IdealSpecies MPI Multimode NSCBCInflowBC NSCBCOutflowBC
Nasa9Interval Nasa9Mixture Nasa9Species NoSlipWallBC Numerics OriginBC PeriodicBC
PoleBC Prim Problem ProgressLog Ramp SlipWallBC Solver SolverFailure SphericalMetric
StateGuard StepControl StiffenedGas Stretch SwitchableBC SymmetryPlaneBC Trigger
TurbulentInflow WhenState Workspace
allocate_state binary_diffusivity boundary_plane cartesian_coordinates cartesian_slice
compact_filter compute_dt conserved_from_prim dissipation_rate domain_volume
driver_pressure dt_report field_array field_slice field_snapshot fieldheatmap
fieldheatmap! gaussian_filter initialize! lele_d1_10 lele_d1_6 lele_d1_8
level_regions line_profile line_sample load_checkpoint! load_checkpoint_hdf5!
mass_fraction mass_fractions mix_width mixture_density mole_fractions
molecular_mixing mpi_main nasa9_constant_cp neutral_binary_diffusion nlevels
nspecies plane_profile profile_coordinate profile_spacing profileplot profileplot!
pyranda_filter read_cea_transport read_nasa9 reflected_shock refined_region
refresh_primitives! revolve_profile riemann_interface run! save_checkpoint
save_checkpoint_hdf5 save_hdf5 save_vtk setup shock_jump shock_tube sine_cluster
species_names species_pdf state_guard state_report state_valid switch! switched
tanh_blend temperature_domain thermodynamic_state tke_profile total_energy
transport_coefficients turbulent_kinetic_energy validate_state! velocity
volume_average volume_integral
""")))

# Supported but not exported: reached as `CompactLES.name`, declared `public`
# on Julia 1.11 and later.
const EXPECTED_PUBLIC = Set(Symbol.(split("""
Regions DiffusionData
enforce! correct_flux! correct_rhs! validate_bc sensor_mirror isperiodic
eos_phi eos_dphi_dY artificial_conductivity_scale wall_internal_energy
state_admissibility conserved_parity species_enthalpy Metric EquationSet
NavierStokes1T fired! next_time rewind! fires_at_start add_source!
compute_rhs! apply_bcs! recover_primitives! filter_state! max_rate FloorTally
step! sync_patches! sync_levels! eachpatch npatches
ConservedState CPUBackend interior_index padded_index xcoord global_xcoord
DEFAULT_VTK_FIELDS StateReport FieldSnapshot
makie_available hdf5_available hdf5_parallel
ClosureRow BandedClosureRow CompactScheme BandedCompactScheme pade_d1_4 compact_d8
""")))

# The submodules' own exports, loaded with `using CompactLES.Regions` and
# `using CompactLES.DiffusionData`. `Cells` is exported by both CompactLES and
# `Regions`, as one binding.
const EXPECTED_REGIONS = Set(Symbol.(split("""
Shape Slab Box Ellipsoid Sphere Cylinder LevelSet signed_distance Cells Layer Layers
""")))
const EXPECTED_DIFFUSION_DATA = Set(Symbol.(split("""
H_ION_MASS D_ION_MASS T_ION_MASS StantonMurilloDiagnostics
stanton_murillo_interdiffusivity
MarreroMasonPair MARRERO_MASON_1972 MARRERO_MASON_UNCERTAINTY marrero_mason_pair
marrero_mason_pairs marrero_mason_diffusivity
SongWangPair SONG_WANG_2016 SONG_WANG_UNCERTAINTY song_wang_pair song_wang_diffusivity
MuellerKlemmPair MUELLER_KLEMM_1970 MUELLER_KLEMM_TEMPERATURE MUELLER_KLEMM_PRESSURE
mueller_klemm_pair neutral_binary_sources neutral_binary_diffusion_residual
""")))

const ADVANCED_QUALIFIED_API = (
    :Decomp, :exchange_halos!, :interior, :field,
    :DirPlan, :BandPlan, :DevicePlan, :device_plan, :apply_along!, :filter_field!,
    :TransferPlan, :plan_transfer, :restrict!, :prolong!,
    :Patch, :PatchSolver, :Level, :LevelComm,
    :script_args, :script_grid,
)

const EXTENSION_API = (
    :recover_primitives!, :species_enthalpy, :eos_phi,
    :eos_dphi_dY, :artificial_conductivity_scale, :wall_internal_energy,
    :state_admissibility,
    :conserved_parity, :enforce!, :correct_flux!, :correct_rhs!, :validate_bc, :sensor_mirror,
    :isperiodic,
    :fired!, :next_time, :rewind!,
)

# `Base.Docs.doc` of a plain object is a REPL method from Julia 1.11 on, so a
# test process that never loads REPL raises a MethodError. The module's own doc
# table is the portable lookup, and it answers the narrower question this file
# asks: is a docstring attached to this binding at all.
function has_docstring(mod::Module, name::Symbol)
    binding = Base.Docs.Binding(mod, name)
    value = getfield(mod, name)
    # A module's docstring is kept in that module's own table.
    tables = value isa Module ? (Base.Docs.meta(value), Base.Docs.meta(binding.mod)) :
             (Base.Docs.meta(binding.mod),)
    return any(table -> haskey(table, binding), tables)
end
has_docstring(name::Symbol) = has_docstring(CompactLES, name)

exported_names(mod::Module) =
    Set(n for n in names(mod) if n !== nameof(mod) && Base.isexported(mod, n))

@testset "public API manifest" begin
    actual = exported_names(CompactLES)
    @test actual == EXPECTED_EXPORTS
    @test !isdefined(CompactLES, :single_species)
    @test isempty(EXPECTED_PUBLIC ∩ actual)

    for name in ADVANCED_QUALIFIED_API
        @test isdefined(CompactLES, name)
        @test name ∉ actual
    end

    for name in EXPECTED_PUBLIC
        @test isdefined(CompactLES, name)
        @test has_docstring(name)
        @static if VERSION >= v"1.11.0-DEV.469"
            @test Base.ispublic(CompactLES, name)
        end
    end

    for name in EXTENSION_API
        @test name ∈ EXPECTED_PUBLIC
    end

    @test exported_names(CompactLES.Regions) == EXPECTED_REGIONS
    @test exported_names(CompactLES.DiffusionData) == EXPECTED_DIFFUSION_DATA
    @test CompactLES.Regions.Cells === CompactLES.Cells
    for (mod, expected) in ((CompactLES.Regions, EXPECTED_REGIONS),
                            (CompactLES.DiffusionData, EXPECTED_DIFFUSION_DATA))
        for name in expected
            @test has_docstring(mod, name)
            @test name ∉ actual || name === :Cells
        end
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
        tuple_names = ("He", SubString("CO2", 1, 3))
        @test [species.name for species in IdealMixture(tuple_names).sp] ==
              ["He", "CO2"]
        @test IdealMixture(Float32, tuple_names).Rk isa Vector{Float32}

        nasa = Nasa9Mixture(["He", "CO2"])
        nasa_explicit = Nasa9Mixture(read_nasa9(["He", "CO2"]))
        @test [species.name for species in nasa.sp] == ["He", "CO2"]
        @test nasa.Rk ≈ nasa_explicit.Rk
        @test [species.name for species in Nasa9Mixture(tuple_names).sp] ==
              ["He", "CO2"]
        @test Nasa9Mixture(Float32, tuple_names).Rk isa Vector{Float32}
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
                    dims=(1, 1, 1), art=ArtificialProperties(enabled=false))
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
    @test has_docstring(:profileplot!)
    @test has_docstring(:fieldheatmap!)
end
