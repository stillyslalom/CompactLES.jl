# Precompile workload.
#
# The first run of a script compiles the call tree beneath `setup`, `run!`
# and `compute_rhs!` for the `Solver` type it builds, and that type carries
# the precision, the metric, the equation of state and transport types, the
# fold of each face and which dimensions are active. The workload below runs
# a few such configurations on small grids at np = 1 so that their trees are
# in the package image.
#
# It is in two parts of about equal precompile time. The primary part covers
# what most scripts do: a periodic box, a line and a plane between walls or
# open ends, one or two ideal gases or a NASA-9 mixture, the artificial
# properties and the state filter at their defaults, scheduled callbacks. The
# secondary part covers less common configurations whose trees are expensive
# to compile at run time: a regridded refined level, the r-z axis, the
# spherical origin, symmetry planes. Every configuration is built as a script
# builds it, through `setup(Problem, Numerics)`.
#
# The test suites build many more `Solver` types than a user's run does, and
# a multi-rank test run compiles each of them independently on every rank.
# Their configurations are a separate workload, test/precompile_workload.jl,
# which this file includes and runs only under the preferences below.
#
# A `Decomp` needs an initialized communicator, so the workload initializes
# MPI as a singleton inside the precompile process. That is fine with the
# bundled binaries, and it is skipped when `MPIPreferences.binary` is
# `"system"`: a singleton init on a cluster login node may fail or hang under
# the scheduler's process manager, and installing the package must not depend
# on it. Whether it can be enabled there is an open item in
# reference/CLUSTER.md.
#
# One limit. Closures are typed by their defining module, so a script still
# compiles `pointwise_initial` for its own initial-condition function, and
# its own functions in full; a solver call inside them reaches the image's tree
# when the image holds that `Solver` type.

using PrecompileTools
using MPIPreferences: MPIPreferences
using Preferences: @load_preference

"""
Whether the workload covers the serial and multi-rank test suites' solver
configurations (`_test_suite_workload` in `test/precompile_workload.jl`):
every metric and fold through the right-hand side, the sensors and detectors,
same-level patches, static, tiled, subcycled and regridded levels. Off by
default, since a user's run builds few of these types and every package
precompile takes longer with the block on.

Turn it on where the suites run, which is what CI does:

```julia
using Preferences, CompactLES
Preferences.set_preferences!(CompactLES, "precompile_tests" => true)
```

The preference is read at precompile time, so setting it rebuilds the package
image.
"""
const PRECOMPILE_TESTS = @load_preference("precompile_tests", false)

"""
Whether the workload covers the KernelAbstractions device path, which it
reaches on this machine by running a `DeviceBackend(KernelAbstractions.CPU())`
solver under `FORCE_KA` and `FORCE_DEVICE_EXCHANGE`. Off by default:
a run on ordinary host storage never enters that path, and a run on a real
device needs its own backend's kernels rather than the CPU backend's, so what
the block bakes in serves the suites that pin the two against each other and
nobody else. Measured on the workstation, leaving it out takes the package
precompile from 96 s to 78 s and the image from 141 MB to 121 MB.

Turn it on where `test/device_tests.jl` or the MPI suite's device phases run,
which is what CI does, since they otherwise compile that tree themselves and
every rank of an MPI leg pays for it separately:

```julia
using Preferences, CompactLES
Preferences.set_preferences!(CompactLES, "precompile_device" => true)
```

The preference is read at precompile time, so setting it rebuilds the package
image; nothing needs to be forced by hand.
"""
const PRECOMPILE_DEVICE = @load_preference("precompile_device", false)

"""
Whether the workload covers single precision: a `Float32` solver on a periodic
box and on a refined line, each advanced through `run!`. A second element type
compiles a second copy of the tree beneath `step!`, and every process that
loads the package reads that copy from the image whether or not it runs in
`Float32`. Off by default. On the development workstation the block adds
about 20 s to the package precompile, 50 MiB to the image and 0.1 s to every
`using CompactLES`; without it, the first `Float32` run of a process compiles
for about 13 s more, on every rank of an MPI run.

Turn it on where `Float32` runs are the rule rather than the exception, and
where the serial suite or the MPI suite runs, which is what CI does:

```julia
using Preferences, CompactLES
Preferences.set_preferences!(CompactLES, "precompile_float32" => true)
```

Like `precompile_device`, the preference is read at precompile time, so
setting it rebuilds the package image.
"""
const PRECOMPILE_FLOAT32 = @load_preference("precompile_float32", false)

"""
Whether the workload builds the solver configurations of the compile-bound
phases of the multi-rank suite's eight-rank selection (`rank_shape=true` in
`test/mpi_tests.jl`): the positivity limiter on each of its layouts, the
symmetry planes, a refined level at a symmetry plane and on the r-z axis, the
staggered diffusion operator and the implicit stage on their metrics, and the
implicit-conduction pair. Every rank of that suite otherwise compiles each of
them for itself, and on a runner with fewer cores than ranks the ranks share
the cores while they compile. Off by default, since a user's run builds few of
these types and every package precompile takes longer with the block on.

Turn it on where the suite runs at many ranks, as CI's eight-rank job does:

```julia
using Preferences, CompactLES
Preferences.set_preferences!(CompactLES, "precompile_mpi_suite" => true)
```

Like the other preferences, it is read at precompile time, so setting it
rebuilds the package image. The block reproduces each phase's types, not its
data: the precision, metric, equation of state, transport, folded faces,
active dimensions, refinement and the keyword types of the `Solver` call. A
phase whose configuration changes without a matching change here still runs,
and compiles its tree on every rank.
"""
const PRECOMPILE_MPI_SUITE = @load_preference("precompile_mpi_suite", false)

# ---- Primary: what most scripts build -------------------------------------

# A periodic box and a box between walls at the default numerics, and a phase
# change from the wall to an outflow.
function _workload_box()
    per = (PeriodicBC(), PeriodicBC())
    walls = (SlipWallBC(), SlipWallBC())
    for xbc in (per, walls)
        s, Q = setup(Problem(domain=((0.0, 1.0), (0.0, 0.25), (0.0, 0.25)),
                             bcs=(xbc, per, per),
                             ic=(x, y, z) -> Prim(u=(0, 0, 0),
                                                  p=1 + 4exp(-200(x - 0.3)^2),
                                                  rho=1 + exp(-200(x - 0.3)^2))),
                     Numerics(n_global=(16, 12, 12)))
        run!(s, Q; tfinal=1e9, nmax=2)
        compute_dt(s, Q)
        xbc === walls || continue
        s, Q = setup(s, Q; bcs=((SlipWallBC(), NSCBCOutflowBC(pinf=1.0)), per, per))
        run!(s, Q; tfinal=1e9, nmax=4)
    end
    return nothing
end

# Output and diagnostics on the periodic box: a VTK dump, a `FieldWriter`
# on a schedule, a checkpoint written and read back, and a volume integral.
function _workload_output()
    per = (PeriodicBC(), PeriodicBC())
    s, Q = setup(Problem(domain=((0.0, 1.0), (0.0, 0.25), (0.0, 0.25)),
                         bcs=(per, per, per),
                         ic=(x, y, z) -> Prim(u=(0.1sinpi(2x), 0, 0), p=1.0,
                                              rho=1 + 0.1cospi(2x))),
                 Numerics(n_global=(16, 12, 12)))
    mktempdir() do dir
        writer = FieldWriter(joinpath(dir, "frame"))
        run!(s, Q; tfinal=1e9, nmax=2, callback=Callback(EveryStep(1), writer))
        save_vtk(s, Q, joinpath(dir, "state"))
        save_checkpoint(s, Q, joinpath(dir, "restart"))
        load_checkpoint!(s, Q, joinpath(dir, "restart"))
    end
    volume_integral(s, Q, :rho)
    return nothing
end

# What a script calls before and around `setup`, at types every script
# shares whatever its solver: species by name from the bundled CEA data,
# compositions, states, the shock and interface solutions, and the
# scheduled triggers, whose sort a script would otherwise compile. Then a
# two-gas line between walls with the artificial properties on, the most
# common start of a script, run under the scheduled triggers.
function _workload_line()
    per = (PeriodicBC(), PeriodicBC())
    walls = (SlipWallBC(), SlipWallBC())
    mix = IdealMixture(["Air", "SF6"])
    light = Prim(Y=mass_fractions(mix, "Air" => 1.0; basis=:mole), p=1e5,
                 T_ion=300.0)
    heavy = Prim(Y=mass_fractions(mix, "Air" => 0.25, "SF6" => 0.75; basis=:mole),
                 p=1e5, T_ion=300.0)
    thermodynamic_state(mix, light)
    incident = shock_jump(mix, light, 1.2)
    riemann_interface(mix, incident.post, heavy)
    AtTime([2e-6, 1e-6])
    EveryTime(1e-6)
    s, Q = setup(Problem(eos=mix, domain=((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)),
                         bcs=(walls, per, per),
                         ic=(x, y, z) -> x < 0.5 ? incident.post : heavy),
                 Numerics(n_global=(32, 1, 1)))
    run!(s, Q; tfinal=1e9, nmax=2,
         callback=(Callback(EveryTime(1e-6), (s, q) -> nothing),
                   Callback(AtTime([1e-6, 2e-6]), (s, q) -> nothing),
                   Callback(EveryStep(1), (s, q) -> nothing)))
    return nothing
end

# The same line with temperature-dependent heat capacities: a NASA-9 mixture
# of two gases at rest across a smoothed pressure jump.
function _workload_nasa9()
    per = (PeriodicBC(), PeriodicBC())
    walls = (SlipWallBC(), SlipWallBC())
    mix = Nasa9Mixture(["He", "Air"])
    s, Q = setup(Problem(eos=mix, domain=((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)),
                         bcs=(walls, per, per),
                         ic=(x, y, z) -> begin
                             θ = 0.5 * (1 + tanh((x - 0.5) / 0.05))
                             Prim(Y=(1 - θ, θ), p=2e5 - 1e5 * θ, T_ion=300.0)
                         end),
                 Numerics(n_global=(32, 1, 1)))
    run!(s, Q; tfinal=1e9, nmax=2)
    return nothing
end

# Two ideal gases on a plane, walls along x and periodic along y.
function _workload_plane()
    per = (PeriodicBC(), PeriodicBC())
    walls = (SlipWallBC(), SlipWallBC())
    mix = IdealMixture([IdealSpecies("light"; R=1.0, gamma=1.4),
                        IdealSpecies("heavy"; R=0.2, gamma=1.09)])
    s, Q = setup(Problem(eos=mix, domain=((0.0, 1.0), (0.0, 0.5), (0.0, 1.0)),
                         bcs=(walls, per, per),
                         ic=(x, y, z) -> begin
                             θ = 0.5 * (1 + tanh((x - 0.5 - 0.02cos(4π * y)) / 0.05))
                             Prim(Y=(1 - θ, θ), p=1.0, T_ion=1.0)
                         end),
                 Numerics(n_global=(32, 16, 1)))
    run!(s, Q; tfinal=1e9, nmax=2)
    return nothing
end

# ---- Secondary: less common, expensive to compile at run time --------------

# A folded line between two face-centred symmetry planes, with the
# artificial properties and the state filter on.
function _workload_symmetry()
    per = (PeriodicBC(), PeriodicBC())
    sym = (SymmetryPlaneBC(), SymmetryPlaneBC())
    s, Q = setup(Problem(domain=((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)),
                         bcs=(sym, per, per),
                         ic=(x, y, z) -> begin
                             rho = 1 + 0.05cos(pi * x)
                             Prim(u=(0.05sin(pi * x), 0, 0), p=rho^1.4,
                                  rho=rho)
                         end),
                 Numerics(n_global=(16, 1, 1)))
    run!(s, Q; tfinal=1e9, nmax=2)
    return nothing
end

# A refined level of tiles on a periodic plane, selected by the sensor and
# regridded as the run goes.
function _workload_amr()
    per = (PeriodicBC(), PeriodicBC())
    s, Q = setup(Problem(domain=((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)),
                         bcs=(per, per, per),
                         ic=(x, y, z) -> begin
                             bump = exp(-((x - 0.4)^2 + (y - 0.5)^2) / 0.004)
                             Prim(u=(0.5, 0.3, 0), p=1.0, rho=1 + 2bump)
                         end),
                 Numerics(n_global=(48, 48, 1), amr=AMR(tile=12)))
    run!(s, Q; tfinal=1e9, nmax=6)
    level_regions(s, 1)
    return nothing
end

# The r-z plane of a cylinder, the axis at r = 0.
function _workload_axis()
    per = (PeriodicBC(), PeriodicBC())
    walls = (SlipWallBC(), SlipWallBC())
    s, Q = setup(Problem(metric=CylindricalMetric(),
                         domain=((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)),
                         bcs=((AxisBC(), SlipWallBC()), per, walls),
                         ic=(r, θ, z) -> Prim(rho=1.0, u=(0, 0, 0),
                                              p=0.1 + exp(-(r^2 + (z - 0.5)^2) / 0.02))),
                 Numerics(n_global=(24, 1, 24)))
    run!(s, Q; tfinal=1e9, nmax=2)
    return nothing
end

# The radial line of a sphere, the origin at r = 0.
function _workload_origin()
    per = (PeriodicBC(), PeriodicBC())
    s, Q = setup(Problem(metric=SphericalMetric(),
                         domain=((0.0, 1.0), (pi / 2, pi / 2 + 1), (0.0, 1.0)),
                         bcs=((OriginBC(), SlipWallBC()), per, per),
                         ic=(r, θ, φ) -> Prim(rho=1.0, u=(0, 0, 0),
                                              p=0.1 + exp(-(r / 0.2)^2))),
                 Numerics(n_global=(32, 1, 1)))
    run!(s, Q; tfinal=1e9, nmax=2)
    return nothing
end

# Single precision, under `PRECOMPILE_FLOAT32`: a periodic box and a line
# with a refined region.
function _workload_float32()
    per = (PeriodicBC(), PeriodicBC())
    T = Float32
    single = Execution(storage=Array{T})
    s, Q = setup(Problem(domain=((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)),
                         bcs=(per, per, per),
                         ic=(x, y, z) -> Prim(rho=1 + T(0.01) * sin(2π * x), T_ion=one(T),
                                              u=(T(0.1), zero(T), zero(T)))),
                 Numerics(n_global=(16, 12, 12), execution=single,
                          art=ArtificialProperties(enabled=false), cfl=0.2))
    run!(s, Q; tfinal=T(0.01), nmax=2)
    compute_dt(s, Q)
    s, Q = setup(Problem(domain=((0.0, 2π), (0.0, 1.0), (0.0, 1.0)),
                         bcs=(per, per, per),
                         ic=(x, y, z) -> Prim(u=(T(0.5), zero(T), zero(T)), p=one(T),
                                              rho=1 + T(0.1) * sin(x))),
                 Numerics(n_global=(48, 1, 1), execution=single, filter=nothing,
                          art=ArtificialProperties(enabled=false), cfl=0.2,
                          amr=AMR(initial=BlockRegion((20, 0, 0), (8, 1, 1)),
                                  regrid_interval=0)))
    run!(s, Q; tfinal=T(0.01), nmax=2)
    return nothing
end

const _TEST_WORKLOAD = PRECOMPILE_TESTS || PRECOMPILE_DEVICE || PRECOMPILE_MPI_SUITE
_TEST_WORKLOAD && include(joinpath(@__DIR__, "..", "test", "precompile_workload.jl"))

if MPIPreferences.binary != "system"
@setup_workload begin
    MPI.Initialized() || MPI.Init(threadlevel=:funneled)
    @compile_workload begin
        _workload_box()
        _workload_output()
        _workload_line()
        _workload_nasa9()
        _workload_plane()
        _workload_symmetry()
        _workload_amr()
        _workload_axis()
        _workload_origin()
        PRECOMPILE_FLOAT32 && _workload_float32()
        if _TEST_WORKLOAD
            cpu = KernelAbstractions.CPU()
            (PRECOMPILE_TESTS || PRECOMPILE_DEVICE) && _test_suite_workload(cpu)
            PRECOMPILE_MPI_SUITE && _mpi_suite_workload(cpu)
        end
    end
end
end
