# CompactLES

[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://stillyslalom.github.io/CompactLES.jl/dev/)
[![Build Status](https://github.com/stillyslalom/CompactLES.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/stillyslalom/CompactLES.jl/actions/workflows/CI.yml?query=branch%3Amain)

![Taylor–Green vorticity, a multicomponent shock-interface interaction, and a cylindrical converging shock](docs/src/assets/readme_header.png)

*CompactLES output: Taylor–Green vorticity magnitude, a He/CO₂
shock–interface interaction (white: `Y_CO₂ = 0.5`), and an axisymmetric
cylindrical converging shock. At `64³`, the measured Taylor–Green dissipation
peak is `ε = 0.01265` at `t = 9.13`, compared with the pseudo-spectral result
`ε = 0.01289` at `t = 8.86` from
[van Rees et al. (2011)](https://doi.org/10.1016/j.jcp.2010.11.031).
Reproduce the figure with
[`docs/figures/readme_header.jl`](docs/figures/readme_header.jl).*

A compressible large-eddy-simulation / direct-simulation solver for the
multicomponent Navier–Stokes equations, inspired by 
[Pyranda](https://github.com/LLNL/pyranda), written in pure Julia. It combines
high-order compact (Padé) finite differences with Cook-style artificial fluid
properties for shock and subgrid regularization, and runs on shared-memory
threads and distributed MPI ranks with no dependencies beyond `MPI.jl`,
`LinearAlgebra`, and `Printf`.

The solver is designed around a clean split between what you simulate (a
`Problem`: gas model, geometry, boundary conditions, initial state) and how it
is discretized (a `Numerics`: resolution, scheme order, CFL, process grid). The
same problem re-runs at any resolution or scheme order without change.

```julia
using MPI; MPI.Init(threadlevel=:funneled)
using CompactLES

prob = Problem(
    domain = ((0.0, 2π), (0.0, 2π), (0.0, 2π)),
    bcs    = ntuple(_ -> (PeriodicBC(), PeriodicBC()), 3),
    ic     = (x, y, z) -> Prim(u=(sin(x)*cos(y), -cos(x)*sin(y), 0.0),
                               p=71.43, rho=1.0))

solver, Q = setup(prob, Numerics(n_global=(64, 64, 64)))
run!(solver, Q; tfinal=1.0)
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
Prim(; u=(0,0,0), p, T_ion=NaN, rho=NaN, Y=(1.0,))
```

A **`Problem`** bundles the physics and geometry:

```julia
prob = Problem(
    eos       = single_species(gamma=1.4),          # or IdealMixture([...])
    transport = Transport(mu0=1/1600, Pr=0.7, Sc=0.7),
    metric    = CartesianMetric(),                  # or Cylindrical/Spherical
    sources   = (ConstantBodyForce((0.0, -9.81, 0.0)),),
    domain    = ((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)),  # (lo, hi) per dimension
    bcs       = ((SlipWallBC(), NSCBCOutflowBC(pinf=0.1)),
                 (PeriodicBC(), PeriodicBC()),
                 (PeriodicBC(), PeriodicBC())),
    ic        = (x, y, z) -> Prim(u=(0,0,0), p=1.0, rho=1.0))
```

A **`Numerics`** bundles the discretization and runtime choices:

```julia
num = Numerics(
    n_global        = (256, 64, 64),      # global grid
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
solver, Q = setup(prob, num)
workspace = Workspace(Q)  # retain stage arrays for reuse/splitting/IMEX
run!(solver, Q; workspace, tfinal=0.25, nmax=100_000,
     callback=(solver, Q) -> ...)
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
| Open boundaries | `NSCBCOutflowBC(pinf=...)`, `NSCBCInflowBC(u=..., T_ion=..., Y=...)`, `ExtrapolationBC` |
| Forcing         | Typed source tuples (`ConstantBodyForce`) and time-dependent `DirichletBC` |
| Singular axes   | `AxisBC` (cylindrical axis), `OriginBC` (spherical origin), `PoleBC` (spherical poles) |
| Thermodynamics  | `IdealMixture` of `IdealSpecies`; EOS interface for custom models |
| Regularization  | Cook artificial μ\*, β\*, κ\*, D\* (`ArtParams`) |
| I/O             | `save_checkpoint` / `load_checkpoint!`, `save_vtk` |

## Timestep and CFL near coordinate singularities

Worth understanding before running resolved-angle polar grids. `compute_dt`
uses true physical spacings (`inv_h` carries the metric scale factor and any
stretching Jacobian), so the estimate is *correct* at a singularity — it will
not silently under-restrict. But correct is not the same as cheap, and the
three regimes behave very differently:

- **Collapsed angular dimension** (1-D radial, axisymmetric, axisymmetric with
  swirl): no pathology at all. The skipped dimension contributes no advective
  term, so `dt` is set by the radial spacing exactly as in a Cartesian run.
  This is why `examples/converging_shock.jl` runs at a sane timestep right onto
  the axis.
- **Resolved θ in cylindrical**: the azimuthal spacing is r·Δθ, so at the first
  half-offset node (r ≈ R/2N_r) the acoustic limit is tighter than the radial
  one by roughly N_θ/π. With N_θ = 64 that is a ~20× penalty, and it is a
  property of the polar grid, not of this implementation — every explicit polar
  solver pays it.
- **Spherical**: worse, because the origin and the poles compound. The φ
  spacing is r·sinθ·Δφ, which collapses both as r → 0 and as sinθ → 0; at the
  first node off a pole, sinθ ≈ Δθ/2.

The diffusive limit degrades faster still (it scales with the *square* of the
inverse spacing), and artificial bulk viscosity peaks exactly where a
converging shock reaches the axis — the worst-case combination. If you run
resolved-angle problems seriously, plan on one of the standard remedies:
azimuthal mode truncation or radius-dependent filter strength near the axis, an
implicit/IMEX treatment of the azimuthal direction, or local time stepping.
None is implemented here.

Collapsed angular dimensions have one subtlety the loop above would otherwise
miss: the direction is skipped entirely, but the geometric source ρu_θ²/r still
drives u_r stiffly at small r. For axisymmetric-with-swirl the neglected rate is
|u_θ|/r, which at the first node is the same order as the radial acoustic rate —
enough to eat the CFL safety margin without ever appearing in the estimate.
`curvature_rate` adds it (cylindrical and spherical, only for collapsed angular
dimensions, since resolved ones already cover it through the advective term).

`dt_report(solver, Q)` names the global limiter — value, owning rank, index,
physical coordinates, direction, and whether it is acoustic, diffusive, or
curvature-driven. Call it every few hundred steps to confirm a run is limited by
the physics you care about rather than by the azimuthal spacing at a
singularity.

## Examples

| File | Demonstrates |
|------|--------------|
| `examples/taylor_green.jl`     | Taylor–Green vortex at Re = 1600; periodic box, kinetic-energy diagnostic |
| `examples/shock_tube.jl`       | He-driven Richtmyer–Meshkov shock tube in SI units; multicomponent EOS, artificial properties, closed reflecting tube, optional NSCBC outflow |
| `examples/piston_driver.jl`    | Oscillating full-state Dirichlet driver with non-reflecting NSCBC outflow |
| `examples/converging_shock.jl` | Cylindrically converging shock; 1-D radial run on the regularized axis |

## Output and restart

- **Checkpoints:** `save_checkpoint(solver, Q, "prefix")` writes one dependency-free
  binary file per rank; `load_checkpoint!(solver, Q, "prefix")` restores it. Restarts
  require the same global grid and decomposition.
- **Visualization:** `save_vtk(solver, Q, "prefix")` writes per-rank `.vtr` plus a
  `.pvtr` container with density, velocity, pressure, temperature, and mass
  fractions on the physical grid (stretch mappings included). Open the `.pvtr`
  in ParaView or VisIt.

## Testing

Three suites, ordered so a failure points at one layer:

```
julia --project=. test/runtests.jl                      # serial unit tests
mpiexec -n 4 julia --project=. -t 1 test/mpi_tests.jl   # distributed
julia --project=. -t auto test/convergence.jl           # order studies
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

Coverage is measured with `julia --code-coverage=user` over all three suites
(and the MPI one at more than one rank count), then summarised by
`bench/coverage.jl`. **Read the denominator, not the percentage.** Julia marks
lines belonging to never-compiled methods as non-executable, so an entirely
untested function drops out of the denominator instead of counting as a miss —
`io.jl` and `nscbc.jl` both reported 100% while `save_vtk` and the whole
`NSCBCInflowBC` path had never been compiled. A run that adds tests should be
expected to *increase* the executable-line count, and that increase is the real
measure of what got covered.

`test/convergence.jl` is the slower second line of defence: it prints *observed*
orders and guards them against regression. Measured on the current code, in the
max norm: ≈6 for `lele_d1_6` and ≈10 for `lele_d1_10` in the periodic interior,
but only ≈3 wherever a boundary closure or a coordinate-singularity fold is
active — closed domains, the cylindrical axis, the spherical origin. The error
there is dominated by the first node or two off the wall or axis, which is also
why the fold tolerances in the serial suite are looser than the interior ones.
That is a property of the closure cascade rather than a defect, and a
*collapsed* slope (≈0 rather than ≈3) is the signature to look for: a fold sign
error produces O(1) error at the first node, which makes this the fastest way to
localize the antipodal sign tables. Set `CL_RUN_TG=1` to add the Taylor–Green
Re = 1600 dissipation history.

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
- Soret/Dufour effects and reacting-chemistry models are not implemented;
  reactions can use the typed source interface.

## Learn more

The [development documentation](https://stillyslalom.github.io/CompactLES.jl/dev/)
covers the established frontend, compact operators, parallel decomposition,
runtime diagnostics, and validation workflow. See [DESIGN.md](reference/DESIGN.md) for a
deeper walkthrough of the solver mechanics and the parts of the physics
architecture that are still evolving.
