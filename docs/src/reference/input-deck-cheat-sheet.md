# Input deck cheat sheet

This is the terse reference for constructing a run. CompactLES attaches no
units: any consistent unit system works, and a deck built from an explicit
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

The lifecycle is `Prim` pointwise state → `Problem` physics → `Numerics`
discretization → `setup` → `run!`. `Problem` can be reused with several
resolutions or schemes.

## Pointwise state: `Prim`

`Prim(; u=(0,0,0), p=NaN, T_ion=NaN, rho=NaN, Y=(1.0,))`

| Keyword | Meaning | Rule/default |
|---|---|---|
| `u` | Three physical velocity components | `(0,0,0)` |
| `p` | Pressure | Supply exactly two of `p`, `rho`, `T_ion` |
| `rho` | Mixture density | EOS derives the omitted thermodynamic value |
| `T_ion` | Single temperature | Use consistent temperature units |
| `Y` | Species mass fractions | Tuple order follows the EOS; defaults to `(1.0,)` and must sum to one |

`Prim` is returned by initial conditions and `DirichletBC` functions. The
omitted quantity remains `NaN` in the object; it is derived only during
`conserved_from_prim`.

## Physical model: `Problem`

```julia
Problem(; domain, bcs, ic, name="problem",
        eos=IdealSpecies("gas"; R=1, gamma=1.4), transport=Transport(),
        metric=CartesianMetric(), sources=())
```

| Keyword | Meaning | Default/shape |
|---|---|---|
| `domain` | Coordinate intervals `(lo, hi)` | Required `((lo,hi),(lo,hi),(lo,hi))` |
| `bcs` | A symmetric condition or low/high pair per coordinate | Required `(x,y,z)` |
| `ic` | `(x1,x2,x3) -> Prim` or `(x1,x2,x3,h) -> Prim` | Required; keep it pure |
| `name` | Display/output label | `"problem"` |
| `eos` | Species and thermodynamic closure | `IdealSpecies("gas"; R=1, gamma=1.4)` |
| `transport` | Molecular transport model | `Transport()` (`mu0=0`, `Pr=0.7`, `Sc=0.7`) |
| `metric` | Coordinate geometry | `CartesianMetric()` |
| `sources` | Tuple of explicit source objects | `()` |

### EOS and species choices

```julia
IdealSpecies("CO2")                         # NASA-9 reference-state lookup
IdealSpecies("gas"; R=1.0, gamma=1.4)       # explicit calorically perfect gas
IdealMixture(["He", "CO2"])                 # one constant-cp species per name
Nasa9Mixture(["He", "CO2"])                 # temperature-dependent NASA-9 model
Nasa9Mixture([Nasa9Species("He"), Nasa9Species("CO2")]; T_guess=300.0)
StiffenedGas(gamma=4.4, p_inf=6.0e8, cv=1816.0, name="liquid")  # the defaults
```

`IdealSpecies("CO2")` samples the bundled NASA-9 thermodynamics at the
reference temperature (298.15 K); it is a reference-temperature approximation,
not a temperature-dependent NASA-9 EOS. Every database-backed constructor
returns SI properties (`R` in J/(kg K), temperatures in K), so the rest of the
deck must be in SI too. Names in an `IdealMixture` or `Nasa9Mixture` define the
order required by every `Prim.Y`. A single `IdealSpecies` is promoted
internally to the one-species mixture representation. See
[Thermodynamics and species transport](@ref).

`Transport(mu0=..., Pr=..., Sc=...)` uses constant molecular properties: the
thermal conductivity is `mu0 * cp / Pr` and the species diffusivity, common to
all species, is `mu0 / (rho * Sc)`.
Use `CeaTransport(eos)` for temperature-dependent viscosity and conductivity
with unity-Lewis diffusion; mixture-averaged diffusion requires a supplied
`BinaryDiffusion` model. See [Thermodynamics and species transport](@ref).

### Geometry

| Metric | Coordinates/velocity | Typical boundary treatment |
|---|---|---|
| `CartesianMetric()` | `(x,y,z)`, `(u,v,w)` | Ordinary walls or periodic faces |
| `CylindricalMetric()` | `(r,θ,z)`, `(u_r,u_θ,u_z)` | `AxisBC()` at `r=0`; `θ` may be collapsed |
| `SphericalMetric()` | `(r,θ,φ)`, orthonormal components | `OriginBC()` at `r=0`, `PoleBC()` at both polar ends |

Optional `stretch=(sx,sy,sz)` entries are `nothing` or `sine_cluster(lo,hi,ξc,a)`.
Stretching is non-periodic, cannot cross a fold, and must span the corresponding
domain interval.

### Boundary tuple layout

Each entry of `bcs = (xfaces, yfaces, zfaces)` is either one condition for both
faces or a `(low, high)` pair. For example, a triply periodic domain uses
`bcs = (PeriodicBC(), PeriodicBC(), PeriodicBC())`.

| Condition | Constructor | Use |
|---|---|---|
| Periodic | `PeriodicBC()` | Both ends of a direction; required for collapsed dimensions |
| Slip wall | `SlipWallBC()` | Impermeable adiabatic symmetry plane on the end node |
| Symmetry plane | `SymmetryPlaneBC()` | The same plane half a cell outside the end node, folded by parity at the interior order; Cartesian or cylindrical `z`, single unrefined patch |
| No-slip wall | `NoSlipWallBC()` / `NoSlipWallBC(Twall=...)` | Adiabatic / isothermal viscous wall |
| Extrapolation | `ExtrapolationBC()` | Zeroth-order boundary extrapolation |
| Full prescribed state | `DirichletBC((x,y,z,t) -> Prim(u=(1.0,0,0), p=1.0, rho=1.0))` | Forced or supersonic inflow |
| Characteristic outflow | `NSCBCOutflowBC(pinf=...)` | Subsonic outlet |
| Characteristic inflow | `NSCBCInflowBC(u=(1.0,0,0), T_ion=1.0, Y=[1.0])` | Subsonic inlet |
| Coordinate fold | `AxisBC()`, `OriginBC()`, `PoleBC()` | Matching cylindrical/spherical singular geometry |

`SwitchableBC(before, after)` allows the solver to switch from one boundary condition 
(`before`) to another (`after`) mid-run; call `switch!` from
a globally consistent callback. Fold conditions cannot be switched.

ICs can accept `(x,y,z,h)` and `DirichletBC` or NSCBC targets can accept
`(x,y,z,t,h)`, where `h` is the minimum local physical spacing over resolved
directions. The shorter forms still work; the longer form wins if both are
applicable. Use `tanh_blend(x,x0,2h)` for a numerical two-cell transition,
or a fixed physical width for a resolved material layer.

## Discretization: `Numerics`

```julia
Numerics(; n_global, deriv=lele_d1_6(), filt=compact_filter(0.45),
    art=ArtParams(), cfl=0.5, control=StepControl(), filter_interval=1,
    filter_cfl=0.35, filter_weighting=:none,
    dims=nothing, n_halo=4, comm=MPI.COMM_WORLD,
    stretch=(nothing,nothing,nothing), patch_grid=(1,1,1),
    backend=CPUBackend(), interface_rhs=:extended,
    interface_divergence=nothing, interface_flux=:closure,
    amr=nothing)                             # AMR(...) groups refinement options
```

| Keyword | Meaning | Default |
|---|---|---|
| `n_global` | Global points in `(x,y,z)` | Required; `1` collapses a direction |
| `deriv` | First-derivative compact scheme | `lele_d1_6()` |
| `filt` | Conserved-state compact filter | `compact_filter(0.45)` |
| `art` | Artificial properties | `ArtParams()` |
| `cfl` | CFL multiplier | `0.5` |
| `control` | Timestep landing, recovery, and floors | `StepControl()` |
| `filter_interval` | Apply filter every `k` completed steps | `1`; `0` disables |
| `filter_cfl` | Reference CFL of a full-strength filter pass; `0` unrelaxed | `0.35` |
| `filter_weighting` | `:none` or volume-weighted state filtering | `:none` |
| `dims` | MPI process grid | `nothing` (automatic) |
| `n_halo` | Halo layers per side | `4` |
| `comm` | MPI communicator | `MPI.COMM_WORLD` |
| `stretch` | Per-direction `Stretch` mappings | all `nothing` |
| `patch_grid` | Slab patches along one dimension; excludes explicit `dims` and AMR | `(1,1,1)` |
| `backend` | Storage/execution backend | `CPUBackend()` |
| `interface_rhs` | Patch-interface closure policy | `:extended` |
| `interface_divergence` | Scheme supplying the flux divergence's closure rows at interface ends; experimental, Float64 only | `nothing` |
| `interface_flux` | `:ghost` differentiates the inviscid flux through interfaces from ghost values; experimental | `:closure` |
| `amr` | Refinement, tagging, subcycling, and balancing configuration | `nothing` |

Each resolved rank-local dimension needs enough points for the selected
stencils (nine with the defaults). Every rank in `comm` must call `setup` with
the same `Problem` and `Numerics`.

### Schemes, filters, and sensors

| Choice | Constructors/values |
|---|---|
| Derivative | `pade_d1_4()`, `lele_d1_6()`, `lele_d1_8()`, `lele_d1_10()` |
| State filter | `compact_filter(0.45)` (`closures=:onesided` or `:cascade`), `pyranda_filter()`, `gaussian_filter()`, `compact_d8()` |
| Artificial sensors | `mu_sensor`: `:strain` or `:velocity`; `beta_sensor`: `:strain`, `:gated_strain`, `:ungated_dilatation`, or `:dilatation` |
| Sensor combination | `reduction=:sum` or `:max`; `smoother=:gaussian` or `:compact` |
| Detector | `detector=:delta4` (default) or `:d8` |

## Artificial properties: `ArtParams`

```julia
ArtParams(; enabled=true, C_mu=0.002, C_beta=1.0, C_kappa=0.01, C_D=0.1,
          C_Y=100.0, Y_tolerance=1e-4,
          mu_sensor=:strain, beta_sensor=:strain, reduction=:sum,
          smoother=:gaussian, detector=:delta4, species_flux=:partial_density)
```

Set `enabled=false` for an inviscid/unregularized experiment. `C_mu`, `C_beta`,
`C_kappa`, and `C_D` control artificial shear viscosity, bulk viscosity,
conductivity, and species diffusion; `C_Y` bounds the mass fractions to
`[0, 1]` beyond a dead band of `Y_tolerance`. The sensor symbols select the
fields; `reduction`, `smoother`, and `detector` select how they are combined.
`species_flux` selects how the species diffusivity enters the equations: the
default diffuses the partial densities and holds a uniform pressure across an
interface of unequal molecular weights, `:bulk` diffuses every conserved
variable, and `:fickian` is Cook's per-species flux (see
[Filtering and artificial properties](@ref)).

## Timestep control: `StepControl`

```julia
StepControl(; predict=0.0, max_growth=0.0, landing_steps=2,
    dt_min=0.0, dt_min_ratio=1e-8, retries=0, cfl_backoff=0.5,
    savepoint_interval=25, floor_ratio=0.0, floor_scope=:representable,
    validity=:strict, validity_interval=0, substep_cfl=0.0)
```

| Keyword | Meaning |
|---|---|
| `predict` | One-sided CFL-rate lookahead steps |
| `max_growth` | Maximum multiple of previous `dt`; `0` disables |
| `landing_steps` | Steps allowed to land on `AtTime`/`EveryTime` instants; minimum `1` |
| `dt_min` / `dt_min_ratio` | Absolute / relative timestep floors; `0` disables each |
| `retries` / `cfl_backoff` | Rollback attempts and multiplicative CFL reduction |
| `savepoint_interval` | Steps between rollback savepoints |
| `floor_ratio` | Positivity failsafe strength; `0` disables |
| `floor_scope` | `:representable` or `:internal_energy` repair policy |
| `validity` | `:strict`, `:permissive`, or `:repair` state-validation policy |
| `species_band` | Mass fraction below `-species_band` is rejected; default `0.05` |
| `validity_interval` | Validate every `k` entering states; `0` checks only run endpoints |
| `substep_cfl` | Absolute refined-substep tripwire after refreshed coefficients; `0` disables it |

For production runs set `nmax` in `run!`; use `retries=2–4` for difficult startup
transients and lower `cfl` for converging shocks.

`setup` validates the initial state under `validity`. To validate each accepted
state during a run, including the one it returns, pass a guard as a callback:

```julia
run!(solver, Q; tfinal=1.0, callback=state_guard(solver, Q))
```

A shock converging into a cold or near-vacuum ambient integrates through cells
the equation of state calls inadmissible and needs `validity=:permissive`, which
reports them instead of rejecting them.

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
| Step cadence | `EveryStep(n)` |
| State event | `WhenState((solver,Q)->Bool)` |
| Progress | `ProgressLog(every=10, tfinal=1.0, quantity=turbulent_kinetic_energy, label="TKE")` |
| VTK time series | `FieldWriter("out/field")` with `Callback` |
| Single VTK frame | `save_vtk(solver, Q, path; fields=(:rho, :velocity, :p), stride=2)` |
| HDF5/XDMF frame | `save_hdf5(solver, Q, path; fields=(:rho, :velocity, :p), stride=2)` after `using HDF5` |
| Restart | `save_checkpoint`/`load_checkpoint!`; HDF5 variants for rank-count-independent restart |
| Read fields | `field_array`, `line_sample` (one grid line), `line_profile` (transverse-plane mean), `field_slice`, `cartesian_slice`; `field_snapshot` (whole grid in memory) |
| Plot | `profileplot`/`fieldheatmap` after loading a Makie backend |

`quantity` is any `(solver, Q) -> Real`, such as `volume_integral` or the
diagnostics on the [Diagnostics](@ref) page. `fields` is a tuple drawn from
`:rho`, `:p`, `:T_ion`, `:c`, `:velocity`, `:Y`, `:mach`, `:divergence`,
`:vorticity`, `:vorticity_magnitude`, `:qcriterion`, `:schlieren`,
`:strain_mag`, `:sensor`, `:mu_art`, `:beta_art`, `:kappa_art`, and `:D_art`;
`DEFAULT_VTK_FIELDS` is `(:rho, :velocity, :p, :T_ion, :Y)`. `stride` is one
`Int` or a 3-tuple.

Callbacks fire between completed steps. Their trigger verdict and any collective
diagnostic must be consistent across ranks. Call `refresh_primitives!` before a
callback reads cached primitive arrays; `Q` itself is current between steps.

## MPI, threads, and GPU

```julia
Numerics(n_global=(256,64,64), dims=(4,2,1), comm=MPI.COMM_WORLD)
Numerics(n_global=(256,64,64), backend=DeviceBackend(CUDABackend()))
```

Initialize MPI once (`MPI.Init(threadlevel=:funneled)`), and launch with
`mpiexec -n N julia -t T --project=. deck.jl`. Prefer ranks to threads: at a
fixed core count, single-threaded ranks have beaten multithreaded ranks by
about 2x on every machine measured, a 2-D workstation case included, so use
`-t 1` under `mpiexec` and `mpiexec -n 8 -t 1` rather than `-t 8` for anything
above 1-D. The reasons and the exceptions are in [Threads and ranks](@ref).
`CPUBackend()` is the default; wrap a
`CUDABackend()` or `ROCBackend()` in `DeviceBackend` after loading the matching
GPU package. A device solver may be decomposed, patched, refined, or tiled; it
excludes `level_restriction=:filter` and `Nasa9Mixture`.

## Adaptive refinement

```julia
Numerics(n_global=(96,48,48),
    amr=AMR(initial=(x,y,z,t) -> abs(x - 0.5) < 0.1,
            tag_threshold=Inf, regrid_interval=20, subcycle=true))
```

`AMR(initial=:sensor)` selects the first fine region from enabled tags on the
initialized coarse state; a physical `(x,y,z,t)->Bool` predicate selects by
coordinates and is reused at each regrid check. The predicate is unioned with
sensor tags, so `tag_threshold=Inf` disables the default density criterion for
an exclusively geometric selection. An explicit `BlockRegion(offset, extent)`
selects a known location in parent node space; a vector supplies static nested
levels. Refinement ratio is three. If a sensor or predicate selects no nodes
at setup, `setup` raises an error instead of creating an arbitrary fine patch.

`regrid_interval=0` keeps the initial layout; a positive interval retags one
refined level. `tile` optionally partitions that level into a global lattice,
and `subcycle=true` advances a child three times per parent step. Inspect the
layout with `level_regions(solver, level)`; `refined_region(solver, level)`
requires exactly one patch on that level. Legacy flat refinement keywords on
`Numerics` remain available for existing decks, but cannot be combined with
`amr=AMR(...)`. The [AMR reference](@ref "Adaptive mesh refinement") gives the
full keyword table, support limits, and conservation and restart guidance.

## Common traps

- A collapsed dimension (`n_global[d] == 1`) must be periodic at both ends and
  is not decomposed.
- `bcs` has one entry per dimension: a single condition for both faces or a
  `(low, high)` pair. It is not a flat six-tuple.
- `Y` ordering and length must match the EOS exactly, and fractions must sum to one.
- `Prim` requires exactly two of `p`, `rho`, and `T_ion`.
- `IdealSpecies("CO2")`, `IdealMixture(["He", "CO2"])`, and `Nasa9Mixture`
  carry SI gas constants. Combined with a nondimensional `p = 1`, `rho = 1`
  state they give a temperature near `1/R`, a small fraction of a kelvin.
- Setup and all collective callbacks must be entered by every MPI rank.
- Use `refresh_primitives!` after changing/advancing `Q` before reading cached
  `solver.p`, `solver.T_ion`, or related primitive arrays.
- Fold conditions have geometry and parity restrictions; read the boundary
  reference before placing `AxisBC`, `OriginBC`, `PoleBC` or
  `SymmetryPlaneBC`. A fold moves the grid: the first node of a folded end
  sits half a cell inside the plane.
- A resolved compact direction needs a sufficiently large rank-local block;
  reducing `dims` can fix a setup error about stencil width.
- `using HDF5` and a Makie backend are required before their optional I/O/plot
  extensions become available.
- Keep source and callback effects deterministic across ranks; `WhenState`
  performs the required Boolean reduction.

For extension authors, see [Extending CompactLES](@ref). For supported advanced
operator/decomposition details, see [Operators and decomposition](@ref).
