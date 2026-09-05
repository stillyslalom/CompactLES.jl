# Input deck cheat sheet

This is the terse reference for constructing a run. CompactLES attaches no
units: any consistent unit system works, and a deck built from an explicit
`IdealSpecies` is usually nondimensional. For details, follow the links in the
tables or start with [Define a problem](@ref).

## Smallest complete deck

```julia
using MPI
MPI.Init(threadlevel=:funneled)
using CompactLES

gas = IdealSpecies("gas"; R=1.0, gamma=1.4)
problem = Problem(
    eos=gas,
    domain=((0.0, 2π), (0.0, 2π), (0.0, 2π)),
    bcs=ntuple(_ -> (PeriodicBC(), PeriodicBC()), 3),
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
| `bcs` | Low/high condition for each coordinate | Required `((lo,hi),(lo,hi),(lo,hi))` |
| `ic` | `(x1,x2,x3) -> Prim` | Required; keep it pure |
| `name` | Display/output label | `"problem"` |
| `eos` | Species and thermodynamic closure | `IdealSpecies("gas"; R=1, gamma=1.4)` |
| `transport` | Molecular viscosity and Prandtl/Schmidt numbers | `Transport()` (`mu0=0`, `Pr=0.7`, `Sc=0.7`) |
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

`bcs = ((xlow, xhigh), (ylow, yhigh), (zlow, zhigh))`.

| Condition | Constructor | Use |
|---|---|---|
| Periodic | `PeriodicBC()` | Both ends of a direction; required for collapsed dimensions |
| Slip wall | `SlipWallBC()` | Impermeable inviscid wall |
| No-slip wall | `NoSlipWallBC()` / `NoSlipWallBC(Twall=...)` | Adiabatic / isothermal viscous wall |
| Extrapolation | `ExtrapolationBC()` | Zeroth-order boundary extrapolation |
| Full prescribed state | `DirichletBC((x,y,z,t) -> Prim(u=(1.0,0,0), p=1.0, rho=1.0))` | Forced or supersonic inflow |
| Characteristic outflow | `NSCBCOutflowBC(pinf=...)` | Subsonic outlet |
| Characteristic inflow | `NSCBCInflowBC(u=(1.0,0,0), T_ion=1.0, Y=[1.0])` | Subsonic inlet |
| Coordinate fold | `AxisBC()`, `OriginBC()`, `PoleBC()` | Matching cylindrical/spherical singular geometry |

`SwitchableBC(before, after)` allows the solver to switch from one boundary condition 
(`before`) to another (`after`) mid-run; call `switch!` from
a globally consistent callback. Fold conditions cannot be switched.

## Discretization: `Numerics`

```julia
Numerics(; n_global, deriv=lele_d1_6(), filt=compact_filter(0.45),
    art=ArtParams(), cfl=0.5, control=StepControl(), filter_interval=1,
    filter_cfl=0.0, dims=nothing, n_halo=4, comm=MPI.COMM_WORLD,
    stretch=(nothing,nothing,nothing), patch_grid=(1,1,1),
    backend=CPUBackend(), interface_rhs=:extended,
    refine=nothing)                          # plus the AMR keywords below
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
| `filter_cfl` | Rate-normalized filter reference CFL | `0.0` (unrelaxed) |
| `dims` | MPI process grid | `nothing` (automatic) |
| `n_halo` | Halo layers per side | `4` |
| `comm` | MPI communicator | `MPI.COMM_WORLD` |
| `stretch` | Per-direction `Stretch` mappings | all `nothing` |
| `patch_grid` | Slab patches along one dimension; excludes explicit `dims` and `refine` | `(1,1,1)` |
| `backend` | Storage/execution backend | `CPUBackend()` |
| `interface_rhs` | Patch-interface closure policy | `:extended` |

Each resolved rank-local dimension needs enough points for the selected
stencils (nine with the defaults). Every rank in `comm` must call `setup` with
the same `Problem` and `Numerics`.

### Schemes, filters, and sensors

| Choice | Constructors/values |
|---|---|
| Derivative | `pade_d1_4()`, `lele_d1_6()`, `lele_d1_8()`, `lele_d1_10()` |
| State filter | `compact_filter(0.45)`, `gaussian_filter()`, `compact_d8()` |
| Artificial sensors | `mu_sensor`: `:strain` or `:velocity`; `beta_sensor`: `:strain`, `:gated_strain`, `:ungated_dilatation`, or `:dilatation` |
| Sensor combination | `reduction=:sum` or `:max`; `smoother=:gaussian` or `:compact` |
| Detector | `detector=:delta4` (default) or `:d8` |

## Artificial properties: `ArtParams`

```julia
ArtParams(; enabled=true, C_mu=0.002, C_beta=1.0, C_kappa=0.01, C_D=0.01,
          mu_sensor=:strain, beta_sensor=:strain, reduction=:sum,
          smoother=:gaussian, detector=:delta4)
```

Set `enabled=false` for an inviscid/unregularized experiment. `C_mu`, `C_beta`,
`C_kappa`, and `C_D` control artificial shear viscosity, bulk viscosity,
conductivity, and species diffusion. The sensor symbols select the fields;
`reduction`, `smoother`, and `detector` select how they are combined.

## Timestep control: `StepControl`

```julia
StepControl(; predict=0.0, max_growth=0.0, landing_steps=2,
    dt_min=0.0, dt_min_ratio=1e-8, retries=0, cfl_backoff=0.5,
    savepoint_interval=25, floor_ratio=0.0, floor_scope=:representable)
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

For production runs set `nmax` in `run!`; use `retries=2–4` for difficult startup
transients and lower `cfl` for converging shocks.

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
| Read fields | `field_array`, `line_sample` (one grid line), `line_profile` (transverse-plane mean), `field_slice`, `cartesian_slice` |
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
`mpiexec -n N julia -t T --project=. deck.jl`. `CPUBackend()` is the default; wrap a
`CUDABackend()` or `ROCBackend()` in `DeviceBackend` after loading the matching
GPU package. A device solver may be decomposed, patched, refined, or tiled; it
excludes `level_restriction=:filter` and `Nasa9Mixture`.

## AMR

```julia
Numerics(n_global=(48,48,48),
    refine=BlockRegion((16,16,16), (16,16,16)),
    subcycle=true, regrid_interval=20)
```

`refine` accepts one `BlockRegion(offset, extent)` in root node space, or a
vector of them for a nested hierarchy, each region given in its parent level's
node space. The refinement ratio is three. Useful controls are `level_restriction=:inject` or `:filter`,
`subcycle=false/true`, `regrid_interval`, `tag_threshold`, `tag_buffer`,
`tag_sensor_threshold`, `tag_gradient_threshold`, `tag_vorticity_threshold`,
`tag_predicate`, `untag_ratio`, `tile_lifetime`, `tile`, `rebalance`, and
`rebalance_persist`. Current AMR is Cartesian-only.

## Common traps

- A collapsed dimension (`n_global[d] == 1`) must be periodic at both ends and
  is not decomposed.
- `bcs` is nested by dimension, then `(low, high)`; it is not a flat six-tuple.
- `Y` ordering and length must match the EOS exactly, and fractions must sum to one.
- `Prim` requires exactly two of `p`, `rho`, and `T_ion`.
- `IdealSpecies("CO2")`, `IdealMixture(["He", "CO2"])`, and `Nasa9Mixture`
  carry SI gas constants. Combined with a nondimensional `p = 1`, `rho = 1`
  state they give a temperature near `1/R`, a small fraction of a kelvin.
- Setup and all collective callbacks must be entered by every MPI rank.
- Use `refresh_primitives!` after changing/advancing `Q` before reading cached
  `solver.p`, `solver.T_ion`, or related primitive arrays.
- Fold conditions have geometry and parity restrictions; read the boundary
  reference before placing `AxisBC`, `OriginBC`, or `PoleBC`.
- A resolved compact direction needs a sufficiently large rank-local block;
  reducing `dims` can fix a setup error about stencil width.
- `using HDF5` and a Makie backend are required before their optional I/O/plot
  extensions become available.
- Keep source and callback effects deterministic across ranks; `WhenState`
  performs the required Boolean reduction.

For extension authors, see [Extending CompactLES](@ref). For supported advanced
operator/decomposition details, see [Operators and decomposition](@ref).
