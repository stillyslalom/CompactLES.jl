# Cheat sheet

A compact reference for constructing a run. CompactLES attaches no units to
values: any consistent unit system works, and a deck built from an explicit
`IdealSpecies` is usually nondimensional. For details, follow the links in the
tables or start with [Define a problem](@ref).

## Smallest complete deck

```julia
using CompactLES   # re-exports MPI
MPI.Init(threadlevel=:funneled)

gas = IdealSpecies("gas"; R=1.0, gamma=1.4)
problem = Problem(
    eos=gas,
    domain=((0.0, 2π), (0.0, 2π), (0.0, 2π)),
    bcs=(PeriodicBC(), PeriodicBC(), PeriodicBC()),
    ic=(x, y, z) -> Prim(u=(0.0, 0.0, 0.0), p=1.0, rho=1.0),
)
numerics = Numerics(n_global=(64, 64, 64))
solver, Q = setup(problem, numerics)
run!(solver, Q; tfinal=1.0, nmax=100_000)
```

A deck proceeds from `Prim` (the state at a point) to `Problem` (the physics)
to `Numerics` (the discretization), then `setup` and `run!`. One `Problem` can
be reused with several resolutions or schemes.

## Pointwise state: `Prim`

`Prim(; u=(0,0,0), p=NaN, T_ion=NaN, rho=NaN, Y=(1.0,))`

| Keyword | Meaning | Default and rule |
|---|---|---|
| `u` | The three physical velocity components | `(0,0,0)` |
| `p` | Pressure | Give exactly two of `p`, `rho`, and `T_ion`; the EOS determines the third |
| `rho` | Mixture density | As for `p` |
| `T_ion` | Temperature (the model has one) | As for `p` |
| `Y` | Species mass fractions | `(1.0,)`; order follows the EOS, and the fractions must sum to one |

Initial conditions and `DirichletBC` functions return a `Prim`. The omitted
quantity stays `NaN` in the object and is computed only when the state is
converted to conserved variables (`conserved_from_prim`).

| Need | Use |
|---|---|
| Composition by species name | `mass_fractions(eos, "He" => 0.95, "Air" => 0.05; basis=:mole)` |
| Derived quantities (`rho`, `c`, `gamma`, `h`, ...) | `thermodynamic_state(eos, prim)` |
| Post-shock state | `shock_jump(eos, pre, Mach).post` |
| Shock-tube states and driver pressure | `shock_tube(eos, driver, driven, Mach)` |
| Several regions with smooth transitions | `Layers(background, shape => prim, ...; width=Cells(3))` |
| Pressure in hydrostatic balance with the body force | `Hydrostatic(ic; p_ref, at)` |
| Shapes | `Slab`, `Box`, `Sphere`, `Ellipsoid`, `Cylinder`, `LevelSet`; combined with `∪`, `∩`, `setdiff`, `!` |
| Load the shapes and `Layers` | `using CompactLES.Regions` |
| Constant boundary state | `DirichletBC(prim)`, `NSCBCInflowBC(prim)` |

## Physical model: `Problem`

```julia
Problem(; domain, bcs, ic, name="problem",
        eos=IdealSpecies("gas"; R=1, gamma=1.4), transport=ConstantTransport(),
        metric=CartesianMetric(), sources=())
```

| Keyword | Meaning | Default or shape |
|---|---|---|
| `domain` | Coordinate interval `(lo, hi)` per dimension | Required: `((lo,hi),(lo,hi),(lo,hi))` |
| `bcs` | Per dimension, one condition for both faces or a `(low, high)` pair | Required: `(x,y,z)` |
| `ic` | `(x1,x2,x3) -> Prim` or `(x1,x2,x3,h) -> Prim` | Required; a pure function of its arguments, since every rank evaluates it |
| `name` | Label for display and output | `"problem"` |
| `eos` | Species and equation of state | `IdealSpecies("gas"; R=1, gamma=1.4)` |
| `transport` | Molecular transport model | `ConstantTransport()` (`mu0=0`, so inviscid; `Pr=0.7`, `Sc=0.7`) |
| `metric` | Coordinate system | `CartesianMetric()` |
| `sources` | Tuple of source terms | `()` |

### Equation of state and species

```julia
IdealSpecies("CO2")                         # NASA-9 properties at 298.15 K
IdealSpecies("gas"; R=1.0, gamma=1.4)       # explicit calorically perfect gas
IdealMixture(["He", "CO2"])                 # constant cp for each named species
Nasa9Mixture(["He", "CO2"])                 # temperature-dependent cp (NASA-9)
Nasa9Mixture([Nasa9Species("He"), Nasa9Species("CO2")]; T_guess=300.0)
StiffenedGas(gamma=4.4, p_inf=6.0e8, cv=1816.0, name="liquid")  # the defaults
```

`IdealSpecies("CO2")` evaluates the bundled NASA-9 thermodynamics at the
reference temperature, 298.15 K, and holds them fixed; it is not a
temperature-dependent EOS. Every constructor that reads the database returns
SI properties (`R` in J/(kg K), temperatures in K), so the rest of the deck
must be in SI as well. The order of the names in an `IdealMixture` or
`Nasa9Mixture` is the order of every `Prim.Y`. A single `IdealSpecies` is
stored internally as a one-species mixture. See
[Thermodynamics and transport](@ref).

`ConstantTransport(mu0=..., Pr=..., Sc=...)` holds the molecular properties
constant: the thermal conductivity is `mu0 * cp / Pr`, and the species
diffusivity, the same for every species, is `mu0 / (rho * Sc)`.
`CeaTransport(eos)` gives temperature-dependent viscosity and conductivity
with unity-Lewis diffusion; mixture-averaged diffusion requires binary
diffusion data, a `BinaryDiffusion` or `BinaryDiffusionPolynomial`. See
[Thermodynamics and transport](@ref).

### Geometry

| Metric | Coordinates and velocity | Usual boundary treatment |
|---|---|---|
| `CartesianMetric()` | `(x,y,z)`, `(u,v,w)` | Walls or periodic faces |
| `CylindricalMetric()` | `(r,θ,z)`, `(u_r,u_θ,u_z)` | `AxisBC()` at `r=0`; `θ` may be collapsed |
| `SphericalMetric()` | `(r,θ,φ)`, orthonormal components | `OriginBC()` at `r=0`, `PoleBC()` at both polar ends |

Grid stretching is set by the `Numerics` keyword `stretch=(sx,sy,sz)`, whose
entries are `nothing` or `sine_cluster(lo,hi,ξc,a)`. A stretched direction
can be neither periodic nor folded, and its mapping must span the domain
interval of that direction.

### Boundary conditions

Each entry of `bcs = (xfaces, yfaces, zfaces)` is either one condition for both
faces or a `(low, high)` pair. A triply periodic domain, for example, uses
`bcs = (PeriodicBC(), PeriodicBC(), PeriodicBC())`.

A *fold* is a boundary across which the solution is continued by reflection
rather than closed with one-sided stencils: the cylindrical axis, the
spherical origin and poles, and a symmetry plane.

| Condition | Constructor | Use |
|---|---|---|
| Periodic | `PeriodicBC()` | Both ends of a direction; required for a collapsed dimension |
| Slip wall | `SlipWallBC()` | Impermeable, adiabatic wall without shear stress, through the end node |
| Symmetry plane | `SymmetryPlaneBC()` | A fold half a cell beyond the end node, at the interior order of accuracy; Cartesian directions or cylindrical `z` only, not with `patch_grid`; a refined level may reach it on the CPU backend |
| No-slip wall | `NoSlipWallBC()` / `NoSlipWallBC(Twall=...)` | Adiabatic / isothermal viscous wall |
| Extrapolation | `ExtrapolationBC()` | Zeroth-order extrapolation from the interior |
| Full prescribed state | `DirichletBC((x,y,z,t) -> Prim(u=(1.0,0,0), p=1.0, rho=1.0))` | Forced or supersonic inflow |
| Characteristic outflow | `NSCBCOutflowBC(pinf=...)` | Subsonic outlet with low reflection |
| Characteristic inflow | `NSCBCInflowBC(u=(1.0,0,0), T_ion=1.0, Y=[1.0])` | Subsonic inlet with low reflection |
| Coordinate fold | `AxisBC()`, `OriginBC()`, `PoleBC()` | The singular points of cylindrical and spherical coordinates |

To change a boundary condition partway through a calculation, end the run
with a `Callback` whose effect returns `true`, then continue with
`solver, Q = setup(solver, Q; bcs)`. The new conditions must keep each
dimension's periodicity and folds.

An initial condition may take `(x,y,z,h)`, and a `DirichletBC` or NSCBC target
may take `(x,y,z,t,h)`, where `h` is the smallest local grid spacing over the
directions that are not collapsed. The shorter forms also work; when a
function accepts both, the longer form is used. Use `tanh_blend(x,x0,2h)` for
a transition two cells wide that scales with the grid, or a fixed physical
width for a resolved material layer.

## Discretization: `Numerics`

```julia
Numerics(; n_global, deriv=lele_d1_6(), filter=StateFilter(),
    art=ArtificialProperties(), cfl=0.5, control=nothing,
    patch_interfaces=PatchInterfaces(), execution=Execution(), amr=nothing,
    polar_truncation=0.0, stretch=(nothing,nothing,nothing))
Numerics(base; cfl=0.3)                     # a copy of `base` with cfl replaced
```

| Keyword | Meaning | Default |
|---|---|---|
| `n_global` | Global grid points in `(x,y,z)` | Required; `1` collapses a direction |
| `deriv` | Compact first-derivative scheme | `lele_d1_6()` |
| `filter` | Filter applied to the conserved state: a `StateFilter`, a scheme, or `nothing` | `StateFilter()` |
| `art` | Artificial properties | `ArtificialProperties()` |
| `cfl` | CFL number | `0.5` |
| `control` | Timestep control: landing on output times, rollback and retry, floors | `nothing`: `StepControl()`, or `StepControl(retries=4)` with an `OriginBC` |
| `patch_interfaces` | Treatment of the flux divergence where patches or levels meet: a `PatchInterfaces`, or a `Symbol` for its `flux` | `PatchInterfaces()` |
| `execution` | Process grid, communicator, storage, and same-level patches | `Execution()` |
| `amr` | Adaptive refinement: regions, tagging, subcycling, load balancing | `nothing` |
| `polar_truncation` | Margin κ of the azimuthal mode truncation near a cylindrical axis with resolved θ | `0.0` (off) |
| `stretch` | Grid stretching per direction | all `nothing` |

`ArtificialProperties`, `StepControl`, `AMR`, `StateFilter`, `PatchInterfaces`,
and `Execution` have the same copy form: `StepControl(control; retries=4)`.

### Settings for particular regimes: `Presets`

The defaults are the general shock-capturing configuration. Each function in
`Presets` returns the `Numerics` keywords for a regime the defaults do not
suit. Keywords given after it override it, and `merge` combines two:

```julia
Numerics(Presets.resolved(); n_global=(128,128,128), cfl=0.4)
Numerics(merge(Presets.converging(), Presets.refined_shock()); n_global=(256,1,1))
```

| Function | Regime | Sets |
|---|---|---|
| `Presets.resolved()` | Resolved or smooth flow, turbulence, acoustics | `filter=compact_filter(0.49)` |
| `Presets.refined_shock()` | A shock crossing patch or refinement interfaces | `filter=compact_filter(0.45)`, `patch_interfaces=:closure` |
| `Presets.converging(; cold_ambient=false)` | A shock converging on an axis or origin | `StepControl(retries=4)`; `validity=:permissive` for a cold ambient |
| `Presets.smooth_walls()` | Smooth flow whose accuracy is limited at walls | `deriv=lele_d1_6(closures=:brady_livescu)` |

### State filter: `StateFilter`

`StateFilter(scheme=compact_filter(0.47); interval=1, cfl=0.35, weighting=:none)`

| Field | Meaning | Default |
|---|---|---|
| `scheme` | Compact filter applied to the conserved state | `compact_filter(0.47)` |
| `interval` | Filter every `k` completed steps | `1`; `0` disables |
| `cfl` | CFL number at which a filter pass has full strength; a smaller step filters less. `0` gives full strength at every step | `0.35` |
| `weighting` | `:none`, or `:volume` to filter the state weighted by cell volume | `:none` |

`filter = compact_filter(0.49)` means `StateFilter(compact_filter(0.49))`, and
`filter = nothing` means `StateFilter(interval=0)`.

### Patch and level interfaces: `PatchInterfaces`

`PatchInterfaces(; flux=:ghost, rhs=:extended, divergence=nothing)` applies
only where two patches or two refinement levels meet, not at an interface
between materials.

| Field | Meaning | Default |
|---|---|---|
| `flux` | `:ghost` differentiates the inviscid and molecular fluxes across an interface using ghost values from the neighbouring patch. `:closure` uses one-sided boundary rows at each patch end; it is required on a curvilinear or stretched grid, with `rhs = :onesided`, and for a user-defined EOS on a viscous refined level | `:ghost` |
| `rhs` | How the gradient and divergence rows are closed at an interface: `:extended` or `:onesided` | `:extended` |
| `divergence` | Scheme whose boundary rows give the flux divergence at interface ends; experimental, Float64 only | `nothing` |

`patch_interfaces = :closure` means `PatchInterfaces(flux=:closure)`. Error
messages and checkpoint records name the fields `interface_flux`,
`interface_rhs`, and `interface_divergence`.

### Where it runs: `Execution`

`Execution(; dims=nothing, comm=MPI.COMM_WORLD, storage=Array, patch_grid=(1,1,1))`

| Field | Meaning | Default |
|---|---|---|
| `dims` | MPI process grid: ranks per dimension | `nothing` (chosen automatically) |
| `comm` | MPI communicator | `MPI.COMM_WORLD` |
| `storage` | Array type and floating-point type: `Float32`, `CuArray`, `ROCArray{Float32}`, ... | `Array` (host; each component keeps its own type) |
| `patch_grid` | Split the domain into slab patches along one dimension; not with explicit `dims` or AMR | `(1,1,1)` |

Along each decomposed dimension, a rank's block needs enough points for the
stencils in use: nine with the defaults. Every rank in `comm` must call
`setup` with the same `Problem` and `Numerics`.

### Schemes, filters, and sensors

| Choice | Constructors and values |
|---|---|
| Derivative | `pade_d1_4()`, `lele_d1_6()`, `lele_d1_8()`, `lele_d1_10()` |
| State filter | `compact_filter(0.47)` (`closures=:onesided` or `:cascade`), `pyranda_filter()`, `gaussian_filter()`, `compact_d8()` |
| Artificial-property sensors | `mu_sensor`: `:strain` or `:velocity`; `beta_sensor`: `:strain`, `:gated_strain`, `:ungated_dilatation`, or `:dilatation` |
| How sensors combine | `reduction=:sum` or `:max`; `smoother=:gaussian` or `:compact` |
| Shock and interface detector | `detector=:species_d8` (the default: `:d8` on the mass and mole fractions and `:delta4` on the other fields), `:delta4`, or `:d8` |

## Artificial properties: `ArtificialProperties`

```julia
ArtificialProperties(; enabled=true, C_mu=0.002, C_beta=1.0, C_kappa=0.01, C_D=1.0,
          C_Y=100.0, Y_tolerance=1e-4,
          mu_sensor=:strain, beta_sensor=:strain, reduction=:sum,
          smoother=:gaussian, detector=:species_d8, species_flux=:partial_density)
```

`enabled=false` runs without artificial properties. `C_mu`, `C_beta`,
`C_kappa`, and `C_D` scale the artificial shear viscosity, bulk viscosity,
conductivity, and species diffusivity. `C_Y` scales an additional artificial
diffusivity that holds the mass fractions within `[0, 1]`; it acts once a
fraction leaves that interval by more than `Y_tolerance`. The sensor keywords
choose the fields the properties are computed from, and `reduction`,
`smoother`, and `detector` choose how they are combined. `species_flux` sets
how the species diffusivity enters the equations: the default diffuses the
partial densities and keeps the pressure uniform across an interface between
gases of unequal molecular weight, `:bulk` diffuses every conserved variable,
and `:fickian` is Cook's per-species flux (see
[Filtering and artificial properties](@ref)).

## Timestep control: `StepControl`

```julia
StepControl(; predict=0.0, max_growth=0.0, landing_steps=2,
    dt_min=0.0, dt_min_ratio=1e-8, retries=0, cfl_backoff=0.5,
    savepoint_interval=25, floor_ratio=0.0, floor_scope=:representable,
    validity=:strict, species_band=0.05, validity_interval=0, substep_cfl=0.0)
```

| Keyword | Meaning |
|---|---|
| `predict` | Steps of linear extrapolation of the CFL rate, which can only shorten a step; `0` disables |
| `max_growth` | Largest multiple of the previous `dt` a step may take; `0` disables |
| `landing_steps` | Steps over which `dt` may be shortened to land exactly on an `AtTime`/`EveryTime` instant; minimum `1` |
| `dt_min` / `dt_min_ratio` | Absolute floor / floor relative to the largest `dt` so far; `0` disables each |
| `retries` / `cfl_backoff` | Number of rollbacks to the last savepoint, and the factor applied to the CFL number at each |
| `savepoint_interval` | Steps between rollback savepoints |
| `floor_ratio` | Strength of the positivity failsafe, relative to the minimum density and internal energy at the start of `run!`; `0` disables |
| `floor_scope` | What the failsafe repairs: `:representable` (density and total energy) or `:internal_energy` (also internal energy) |
| `validity` | What a failed state check does: `:strict` raises, `:permissive` reports, `:repair` applies the failsafe first |
| `species_band` | A mass fraction outside `[-species_band, 1 + species_band]` is rejected, and clipped by the failsafe; default `0.05` |
| `validity_interval` | Check the state entering every `k`-th step; `0` checks only the start and end of a run |
| `substep_cfl` | Largest CFL allowed after a refined level updates its artificial properties within a step; `0` disables the check |

In production runs, set `nmax` in `run!`. A run that loses positivity does not
stop by itself: `dt` shrinks toward zero and the run makes no progress.
`nmax` bounds the number of steps, and the `dt` floors in `StepControl` raise
an error when the step collapses. Use `retries=2` to `4`
for difficult start-up transients, and a lower `cfl` for converging shocks.

`setup` checks the initial state under `validity`. To check each accepted state
during a run, including the one it returns, pass a guard as a callback:

```julia
run!(solver, Q; tfinal=1.0, callback=state_guard(solver, Q))
```

A shock converging into a cold or near-vacuum ambient passes through cells
outside the domain of the equation of state, such as cells of negative
internal energy, and needs `validity=:permissive`, which reports those cells
instead of rejecting the state.

## Run control, callbacks, and output

```julia
cb = Callback(EveryTime(0.01),
              (solver, Q) -> save_vtk(solver, Q, "out/frame"))
run!(solver, Q; tfinal=1.0, nmax=100_000, callback=cb)
```

| Need | Use |
|---|---|
| One or more physical times | `AtTime([0.25, 0.5])` |
| Uniform time schedule | `EveryTime(Δt; start=0.0)` |
| Every `n` steps | `EveryStep(n)` |
| When a condition on the state holds | `WhenState((solver,Q)->Bool)` |
| Progress | `ProgressLog(every=10, tfinal=1.0, quantity=turbulent_kinetic_energy, label="TKE")` |
| VTK time series | `FieldWriter("out/field")` with `Callback` |
| Single VTK frame | `save_vtk(solver, Q, path; fields=(:rho, :velocity, :p), stride=2)` |
| HDF5/XDMF frame | `save_hdf5(solver, Q, path; fields=(:rho, :velocity, :p), stride=2)` after `using HDF5` |
| Restart | `save_checkpoint`/`load_checkpoint!`; the HDF5 versions restart on a different rank count |
| Read fields | `field_array`, `line_sample` (one grid line), `line_profile` (average over planes normal to an axis), `field_slice`, `cartesian_slice`; `field_snapshot` (whole grid in memory) |
| Plot | `profileplot`/`fieldheatmap`/`meshplot` after loading a Makie backend |

`quantity` is any `(solver, Q) -> Real`, such as `volume_integral` or the
functions on the [Diagnostics](@ref) page. `fields` is a tuple drawn from
`:rho`, `:p`, `:T_ion`, `:c`, `:velocity`, `:Y`, `:X` (mole fraction), `:mach`,
`:divergence`, `:vorticity`, `:vorticity_magnitude`, `:qcriterion`,
`:schlieren`, `:strain_mag`, `:sensor`, `:mu_art`, `:beta_art`, `:kappa_art`,
and `:D_art`; `DEFAULT_VTK_FIELDS` is `(:rho, :velocity, :p, :T_ion, :Y)`.
`stride` is one `Int` or a 3-tuple.

An `AtTime` or `EveryTime` instant at the initial time fires before the first
step, so a `FieldWriter` sequence begins with the initial condition. Callbacks
run between completed steps. Whether a callback fires, and any quantity it
reduces over ranks, must be the same on every rank. Call `refresh_primitives!`
before a callback reads the stored primitive arrays (`solver.p`,
`solver.T_ion`, ...); `Q` itself is current between steps.

## MPI, threads, and GPU

```julia
Numerics(n_global=(256,64,64), execution=Execution(dims=(4,2,1)))
Numerics(n_global=(256,64,64), execution=Execution(storage=CuArray))
```

Initialize MPI once in the script (`MPI.Init(threadlevel=:funneled)`), and
launch with `mpiexec -n N julia -t T --project=. deck.jl`. Prefer ranks to
threads: at a fixed core count, single-threaded ranks have been about twice as
fast as multithreaded ranks on every machine measured, including a 2-D case on
a workstation. Use `mpiexec -n 8 julia -t 1` rather than `julia -t 8` for
anything beyond 1-D. [Threads and ranks](@ref) gives the reasons and the
exceptions.

`storage` sets where the arrays live and their element type: `Float32` or
`Float64` on the host, `CuArray` or `ROCArray` on a GPU after `using CUDA` or
`using AMDGPU`, and `CuArray{Float32}` or `ROCArray{Float32}` for both. A device
solver may be decomposed, split into patches, refined, or tiled; it does not
support `level_restriction=:filter`, `polar_truncation`, or regridding more
than one refined level. [Supported combinations](@ref) lists the rest.

## Adaptive refinement

```julia
Numerics(n_global=(96,48,48),
    amr=AMR(initial=(x,y,z,t) -> abs(x - 0.5) < 0.1,
            regrid_interval=20, subcycle=true))
```

`initial` chooses what is refined at setup:

- `:sensor`: the cells the tag criteria select on the initial state, followed
  as they move.
- A predicate `(x,y,z,t) -> Bool`: the nodes where it holds, re-evaluated at
  every regrid check, so the region can move on a prescribed path. The
  predicate alone selects the region unless a finite `tag_threshold` is given,
  in which case the density criterion adds to it.
- A shape or a predicate `(x,y,z) -> Bool`: a fixed region.
- A vector of nested shapes, finest last, for several fixed levels; or a
  `BlockRegion(offset, extent)`, or a vector of them, given in node indices
  of each level's parent for an exact layout.

Each level is three times finer than its parent. If the sensor or predicate
selects no nodes at setup, `setup` raises an error instead of creating an
arbitrary refined patch, except in a tiled run that regrids, which then starts
with no refined tiles.

`regrid_interval=0` keeps the initial layout; a positive `K` retags the parent
level every `K` steps, and the default for a moving region is set from the CFL
number and the tag buffer. `tile` covers a refined level with tiles of a global
lattice instead of one box, `max_levels` sets the number of levels, and
`subcycle=true` advances a child level three times per parent step. Inspect
the layout with `level_regions(solver, level)`; `refined_region(solver, level)`
requires exactly one patch on that level. The
[AMR reference](@ref "Adaptive mesh refinement") gives the full keyword table,
the support limits, and guidance on conservation and restart.

## Common traps

- A collapsed dimension (`n_global[d] == 1`) must be periodic at both ends and
  is not decomposed.
- `bcs` has one entry per dimension, either one condition for both faces or a
  `(low, high)` pair. It is not a flat six-tuple.
- The order and length of `Y` must match the EOS exactly, and the fractions
  must sum to one.
- `Prim` requires exactly two of `p`, `rho`, and `T_ion`.
- `IdealSpecies("CO2")`, `IdealMixture(["He", "CO2"])`, and `Nasa9Mixture`
  carry SI gas constants. Combined with a nondimensional state `p = 1`,
  `rho = 1`, they give a temperature near `1/R`, a small fraction of a kelvin.
- Every MPI rank must call `setup` and every callback that reduces over ranks.
- After `Q` changes, call `refresh_primitives!` before reading the stored
  primitive arrays `solver.p`, `solver.T_ion`, and the others.
- Folds have restrictions on geometry and on the parity of the fields; read
  the boundary reference before placing `AxisBC`, `OriginBC`, `PoleBC`, or
  `SymmetryPlaneBC`. A fold shifts the grid: the first node of a folded end
  lies half a cell from the plane.
- A compact direction needs a large enough block on each rank; a setup error
  about stencil width is usually fixed by using fewer ranks along that
  dimension in `dims`.
- The HDF5 output and the Makie plots are available only after `using HDF5` or
  loading a Makie backend.
- Sources and callback effects must give the same result on every rank;
  `WhenState` performs the reduction that makes its condition agree.

For extension authors, see [Extending CompactLES](@ref). For the supported
operator and decomposition interfaces, see [Operators and decomposition](@ref).
