# CompactLES

A compressible large-eddy-simulation / direct-simulation solver for the
multicomponent Navier–Stokes equations, written in pure Julia. It combines
high-order compact (Padé) finite differences with Cook-style artificial fluid
properties for shock and subgrid regularization, and runs on shared-memory
threads and distributed MPI ranks with no dependencies beyond `MPI.jl`,
`LinearAlgebra`, and `Printf`.

The solver is designed around a clean split between *what you simulate* (a
`Problem`: gas model, geometry, boundary conditions, initial state) and *how it
is discretized* (a `Numerics`: resolution, scheme order, CFL, process grid). The
same problem re-runs at any resolution or scheme order without change.

```julia
using MPI; MPI.Init(threadlevel=:funneled)
using CompactLES

prob = Problem(
    domain = ((0.0, 2π), (0.0, 2π), (0.0, 2π)),
    bcs    = ntuple(_ -> (PeriodicBC(), PeriodicBC()), 3),
    ic     = (x, y, z) -> Prim(u=(sin(x)*cos(y), -cos(x)*sin(y), 0.0),
                               p=71.43, rho=1.0))

s, Q = setup(prob, Numerics(nglob=(64, 64, 64)))
run!(s, Q; tfinal=1.0)
```

## Features

- **High-order compact spatial operators.** Sixth-order tridiagonal Lele
  derivatives by default; a tenth-order pentadiagonal derivative is available,
  and you can supply your own compact scheme. An eighth-order Gaitonde–Visbal
  compact filter provides dealiasing and near-wall dissipation control.
- **Shock and subgrid capturing.** Cook (2007) artificial shear viscosity,
  bulk viscosity, conductivity, and species diffusivity, driven by
  high-derivative sensors — no Riemann solvers or flux limiters.
- **Multicomponent thermodynamics.** Any number of species, each with its own
  transport equation, behind a pluggable equation-of-state interface (ideal
  mixture provided).
- **Curvilinear geometry.** Cartesian, cylindrical (r, θ, z), and spherical
  (r, θ, φ) coordinates, with regularized treatment of the cylindrical axis and
  the spherical origin and poles. Collapsed dimensions give cheap 1-D and 2-D
  (including axisymmetric-with-swirl) runs.
- **Stretched meshes.** Per-dimension monotone grid clustering that composes
  with any coordinate system.
- **Boundary conditions.** Periodic, slip / no-slip (adiabatic or isothermal)
  walls, characteristic NSCBC subsonic inflow and outflow, and time-dependent
  Dirichlet forcing (pistons, oscillating drivers, supersonic inflow).
- **Parallelism.** MPI 3-D domain decomposition with a genuinely distributed
  tridiagonal / pentadiagonal solve for the globally coupled compact schemes,
  plus shared-memory threading over grid lines.
- **I/O.** Dependency-free per-rank checkpoint/restart and parallel VTK
  (`.pvtr`) output for ParaView / VisIt.

## Installation

CompactLES is a Julia package targeting Julia ≥ 1.9 and `MPI.jl` v0.20. From the
repository root:

```
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

`MPI.jl` ships its own MPI binary by default, so no system MPI installation is
required to get started.

## Running

Serial or shared-memory threaded (set `-t` to the thread count):

```
julia --project=. -t auto examples/taylor_green.jl
```

Distributed with MPI (`-n` ranks, each with `-t` threads):

```
mpiexec -n 8 julia --project=. -t 2 examples/taylor_green.jl
mpiexec -n 4 julia --project=. -t 2 examples/shock_tube.jl
mpiexec -n 2 julia --project=. -t 2 examples/piston_driver.jl
```

If `mpiexec` is not on your `PATH`, use the launcher bundled with `MPI.jl`:

```
julia --project=. -e 'using MPI; print(MPI.mpiexec(f -> f))'
```

Each rank's local extent in every decomposed dimension must be at least 9 (the
filter closure plus the stencil width). Choose the process grid (`dims=` in
`Numerics`) so that thin dimensions are not split too finely.

## Specifying a problem

Problems are described in **primitive, pointwise** terms and never reference
ranks, halos, or the conserved-variable layout.

A **`Prim`** is a pointwise state — velocity, pressure, composition, and exactly
one of temperature or density (the EOS supplies the other):

```julia
Prim(; u=(0,0,0), p, T=NaN, rho=NaN, Y=(1.0,))
```

A **`Problem`** bundles the physics and geometry:

```julia
prob = Problem(
    eos       = single_species(gamma=1.4),          # or IdealMixture([...])
    transport = Transport(mu0=1/1600, Pr=0.7, Sc=0.7),
    metric    = CartesianMetric(),                  # or Cylindrical/Spherical
    domain    = ((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)),  # (lo, hi) per dimension
    bcs       = ((SlipWallBC(), NSCBCOutflowBC(pinf=0.1)),
                 (PeriodicBC(), PeriodicBC()),
                 (PeriodicBC(), PeriodicBC())),
    ic        = (x, y, z) -> Prim(u=(0,0,0), p=1.0, rho=1.0))
```

A **`Numerics`** bundles the discretization and runtime choices:

```julia
num = Numerics(
    nglob           = (256, 64, 64),      # global grid
    deriv           = lele_d1_6(),        # or lele_d1_10(), or your own
    filt            = compact_filter(0.45),
    art             = ArtParams(enabled=true),   # Cook artificial properties
    cfl             = 0.5,
    filter_interval = 1,                  # filter every N steps (0 disables)
    dims            = nothing,            # process grid; nothing → auto
    stretch         = (nothing, nothing, nothing))  # optional grid clustering
```

Then `setup` marries the two and returns the solver plus the initialized
conserved state, and `run!` advances it:

```julia
s, Q = setup(prob, num)
run!(s, Q; tfinal=0.25, nmax=100_000, callback=(s, Q) -> ...)
```

Initial-condition and Dirichlet-forcing functions are plain, pure functions of
physical coordinates (and time, for forcing). A convergence study is just a loop
over `Numerics` with the same `Problem`.

## Capabilities at a glance

| Category        | Provided |
|-----------------|----------|
| Derivatives     | `lele_d1_6` (C6, tridiagonal), `lele_d1_10` (C10, pentadiagonal), `pade_d1_4`, custom `CompactScheme` / `BandedCompactScheme` |
| Filter          | `compact_filter(αf)` — eighth-order Gaitonde–Visbal, boundary cascade |
| Time integration| Five-stage fourth-order low-storage Carpenter–Kennedy RK45 |
| Geometry        | `CartesianMetric`, `CylindricalMetric`, `SphericalMetric`; collapsed 1-D/2-D; `Stretch` / `sine_cluster` meshes |
| Walls           | `SlipWallBC`, `NoSlipWallBC(Twall=...)` (adiabatic or isothermal) |
| Open boundaries | `NSCBCOutflowBC(pinf=...)`, `NSCBCInflowBC(u=..., T=..., Y=...)`, `ExtrapolationBC` |
| Forcing         | `DirichletBC((x,y,z,t) -> Prim)` |
| Singular axes   | `AxisBC` (cylindrical axis), `OriginBC` (spherical origin), `PoleBC` (spherical poles) |
| Thermodynamics  | `IdealMixture` of `IdealSpecies`; EOS interface for custom models |
| Regularization  | Cook artificial μ\*, β\*, κ\*, D\* (`ArtParams`) |
| I/O             | `save_checkpoint` / `load_checkpoint!`, `save_vtk` |

## Examples

| File | Demonstrates |
|------|--------------|
| `examples/taylor_green.jl`     | Taylor–Green vortex at Re = 1600; periodic box, kinetic-energy diagnostic |
| `examples/shock_tube.jl`       | Two-gas shock tube; multicomponent, artificial properties, slip walls, optional NSCBC outflow and grid clustering |
| `examples/piston_driver.jl`    | Oscillating full-state Dirichlet driver with non-reflecting NSCBC outflow |
| `examples/converging_shock.jl` | Cylindrically converging shock; 1-D radial run on the regularized axis |

## Output and restart

- **Checkpoints:** `save_checkpoint(s, Q, "prefix")` writes one dependency-free
  binary file per rank; `load_checkpoint!(s, Q, "prefix")` restores it. Restarts
  require the same global grid and decomposition.
- **Visualization:** `save_vtk(s, Q, "prefix")` writes per-rank `.vtr` plus a
  `.pvtr` container with density, velocity, pressure, temperature, and mass
  fractions on the physical grid (stretch mappings included). Open the `.pvtr`
  in ParaView or VisIt.

## Testing

A serial suite and a multi-rank MPI suite are included:

```
julia --project=. test/runtests.jl                 # serial
mpiexec -n 4 julia --project=. -t 1 test/mpi_tests.jl   # distributed
```

The serial suite covers operator accuracy (spectral convergence of the compact
derivatives, C10-vs-C6, the transposed y/z path), closed-domain closure
exactness, filter behavior, the coordinate-singularity folds, freestream
preservation in every metric (Cartesian, cylindrical, spherical, stretched,
axis, origin+poles), EOS round-trips, discrete conservation, NSCBC, checkpoint
round-trips, and a full Sod shock tube validated against the exact Riemann
solution. The MPI suite exercises the code paths that only run when a dimension
is split across ranks: the distributed spike solve (tridiagonal and
pentadiagonal), cross-rank halo exchange, off-rank folds, the discrete GCL
across rank boundaries, and telescoping flux conservation. The multi-rank suite
exits nonzero on any failure, so it is CI-gateable.

## Status and limitations

This is research code under active development. The core solver — compact
operators, RK45 integration, artificial properties, multicomponent transport,
the curvilinear metrics, and the distributed solve — is covered by the test
suites above and validated against analytic references (Sod, freestream, rigid
rotation, manufactured fold solutions).

The coordinate-singularity machinery for the *fully resolved* spherical origin
and poles (as opposed to the widely used cylindrical axis) is the newest and
least-exercised part of the code; the full-ball origin+poles combination in
particular warrants scrutiny before production use. Other current limitations:

- Float64 only.
- NSCBC inflow transverse-term accounting is not yet implemented (outflow
  has it).
- Output is checkpoint and VTK only — no HDF5/XDMF, no GPU path.
- Wall boundary conditions assume coordinate-surface walls.
- Soret/Dufour effects and reacting-source hooks are not implemented.

## Learn more

See [DESIGN.md](DESIGN.md) for the package architecture and a walkthrough of the
solver mechanics — the frontend/backend split, the distributed compact solve,
the RHS assembly, the artificial-property model, the curvilinear metrics and
discrete GCL, and the coordinate-singularity folds — plus guidance on extending
the code with new schemes, boundary conditions, physics, and equations of state.
