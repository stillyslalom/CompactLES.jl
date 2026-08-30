# CompactLES

[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://stillyslalom.github.io/CompactLES.jl/dev/)
[![Build Status](https://github.com/stillyslalom/CompactLES.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/stillyslalom/CompactLES.jl/actions/workflows/CI.yml?query=branch%3Amain)

![Taylor–Green vorticity, a multicomponent shock-interface interaction, and a cylindrical converging shock](docs/src/assets/readme_header.png)

*Taylor–Green vorticity, a He/CO₂ shock–interface interaction, and a cylindrical
converging shock. Reproduce the figure with [`docs/figures/readme_header.jl`](docs/figures/readme_header.jl).*

CompactLES is a compressible large-eddy / direct-simulation solver for the
multicomponent Navier–Stokes equations, written in pure Julia and inspired by
[Pyranda](https://github.com/LLNL/pyranda). It pairs high-order compact (Padé)
finite differences with Cook-style artificial fluid properties for shock and
subgrid regularization, and runs on threads, distributed MPI ranks, and GPUs.

A run is specified as a **`Problem`** (the physics: gas model, geometry, boundary
conditions, initial state) evaluated by a **`Numerics`** (the discretization:
resolution, scheme, CFL, process grid), so the same problem can be run at any
resolution or scheme order unchanged.

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

- **High-order compact operators.** Sixth-order tridiagonal Lele derivatives by
  default, tenth-order pentadiagonal available, and an eighth-order
  Gaitonde–Visbal compact filter for dealiasing and near-wall dissipation.
- **Shock and subgrid capturing** without Riemann solvers or limiters: Cook
  (2007) artificial shear/bulk viscosity, conductivity, and species diffusivity,
  driven by high-derivative sensors.
- **Multicomponent thermodynamics.** Any number of transported species behind a
  pluggable EOS; `IdealMixture`, `Nasa9Mixture` (NASA CEA piecewise cp), and
  `StiffenedGas`.
- **Curvilinear geometry.** Cartesian, cylindrical, and spherical coordinates
  with regularized axis, origin, and poles; collapsed dimensions give efficient
  1-D, 2-D, and axisymmetric-with-swirl runs; optional grid stretching.
- **Boundary conditions.** Periodic, slip / no-slip (adiabatic or isothermal)
  walls, Navier-Stokes characteristic subsonic inflow/outflow, and time-dependent Dirichlet forcing.
- **Parallelism.** MPI 3-D decomposition with a distributed tridiagonal /
  pentadiagonal solve for the globally coupled compact schemes, over threads.
- **Adaptive refinement.** Block-structured AMR with sensor-driven tagging and
  regridding, optionally Berger–Oliger subcycled; currently Cartesian-only and
  one refined region.
- **GPU execution.** A `KernelAbstractions.jl` device backend runs the whole
  solver on any supported GPU, bit-for-bit against the CPU, in Float64 or Float32.
- **Diagnostics and I/O.** Coordinate-system-aware, MPI-reduced mixing diagnostics;
  checkpoint/restart and VTK or HDF5/XDMF output for ParaView / VisIt.

## Installation

Targets Julia ≥ 1.10 and `MPI.jl` v0.20. Its dependencies are `MPI.jl`,
`ThreadPinning.jl`, and `KernelAbstractions.jl` / `Adapt.jl`; HDF5 and Makie are
weak dependencies, loaded only when imported. A GPU run additionally needs a
device package (`AMDGPU.jl` or `CUDA.jl`) in the active environment.

```
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

`MPI.jl` includes an MPI binary, so desktop use requires no user-managed
binaries. On a cluster, configure `MPI.jl` against the
[system's MPI binary](https://juliaparallel.org/MPI.jl/stable/configuration/#using_system_mpi)
before running: the bundled binary satisfies a single-node run but may fail to
use the interconnect off it, and the launch rules and measured penalties are in
[`reference/CLUSTER.md`](https://github.com/stillyslalom/CompactLES.jl/blob/main/reference/CLUSTER.md).

## Running

```
julia --project=. -t auto examples/taylor_green.jl              # threaded
mpiexec -n 8 julia --project=. -t 2 examples/taylor_green.jl    # MPI × threads
srun -N 2 -n 64 julia --project=. -t 2 examples/taylor_green.jl # Slurm, Flux, etc.
```

If `mpiexec` is not on `PATH`, get the launcher `MPI.jl` is configured against
with `julia --project=. -e 'using MPI; print(MPI.mpiexec(f -> f))'`.

Each rank's local extent in a decomposed dimension must be at least 9 (the filter
closure plus stencil width), so choose `dims=` in `Numerics` to keep thin
dimensions from splitting too finely.

## Specifying a problem

Problems are described in **primitive, pointwise** terms and never reference
ranks, halos, or the conserved-variable layout. A `Prim` gives velocity,
composition, and exactly two of pressure, density and temperature; the EOS
supplies the third:

```julia
Prim(; u=(0,0,0), p, T_ion=NaN, rho=NaN, Y=(1.0,))
```

```julia
prob = Problem(
    eos       = single_species(gamma=1.4),              # or IdealMixture([...])
    transport = Transport(mu0=1/1600, Pr=0.7, Sc=0.7),  # viscosity, Prandtl, Schmidt
    metric    = CartesianMetric(),                      # or Cylindrical / Spherical
    sources   = (ConstantBodyForce((0.0, -9.81, 0.0)),),
    domain    = ((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)),   # (lo, hi) per dimension
    bcs       = ((SlipWallBC(), NSCBCOutflowBC(pinf=0.1)),
                 (PeriodicBC(), PeriodicBC()),
                 (PeriodicBC(), PeriodicBC())),
    ic        = (x, y, z) -> Prim(u=(0,0,0), p=1.0, rho=1.0))

num = Numerics(
    n_global = (256, 64, 64),                # global grid
    deriv    = lele_d1_6(),                  # or lele_d1_8(), lele_d1_10(), or a custom scheme
    cfl      = 0.5,
    control  = StepControl(retries=4),       # roll back and lower cfl on failure
    dims     = nothing)                      # process grid; nothing → auto
```

`setup` returns the solver and its initialized conserved state; `run!` advances
it, taking an optional `callback`. `ProgressLog` is a ready-made callback that
prints step, time, `dt`, wall time, and an optional reduced diagnostic:

```julia
solver, Q = setup(prob, num)
run!(solver, Q; tfinal=0.25, nmax=100_000,
     callback=ProgressLog(every=10, tfinal=0.25, label="TKE",
                          quantity=turbulent_kinetic_energy))
```

Initial-condition and forcing functions are plain functions of physical
coordinates (and time, for forcing); a convergence study is the same `Problem`
across a sequence of `Numerics`.

## Adaptive refinement and GPU execution

AMR & GPU are selected within `Numerics`.
`refine` places a refined level at ratio 3 over a `BlockRegion` of the
coarse grid (a zero-based node offset and extent), or a nested chain of them
(a vector, each region in the parent level's node space), optionally tiled on a
lattice (`tile`), subcycled, and regridding:

```julia
num = Numerics(n_global = (48, 48, 48),
               refine   = BlockRegion((16, 16, 16), (16, 16, 16)),
               subcycle = true, regrid_interval = 20)
```

`backend = DeviceBackend(ka)` moves the whole solver onto a GPU,
where `ka` is `CUDABackend()` for Nvidia or `ROCBackend()` for AMD.

```julia
using AMDGPU  # or CUDA
num = Numerics(n_global = (64, 64, 64), backend = DeviceBackend(ROCBackend()))
```

Refinement is currently Cartesian-only and single-region; a device run takes a single
patch per solver.

## Timestep near coordinate singularities

`compute_dt` uses physical spacings, so the estimate stays conservative near a
singularity. A collapsed angular dimension (1-D radial, axisymmetric) keeps a
Cartesian-like step, but a resolved polar angle pays roughly N_θ/π, a ~20×
penalty at N_θ = 64, like any resolved-singularity polar grid. `dt_report(solver,
Q)` locates the CFL-limiting cell and whether the limit is acoustic, diffusive, or
curvature-driven. The standard remedies (azimuthal mode truncation, IMEX, local
time stepping) are not yet implemented.

## Examples

| File | Demonstrates |
|------|--------------|
| `examples/taylor_green.jl`     | Taylor–Green vortex at Re = 1600; periodic box, kinetic-energy diagnostic |
| `examples/shock_tube.jl`       | He-driven Richtmyer–Meshkov shock tube; multicomponent EOS, artificial properties, optional NSCBC outflow |
| `examples/piston_driver.jl`    | Oscillating full-state Dirichlet driver with non-reflecting NSCBC outflow |
| `examples/converging_shock.jl` | Cylindrically converging shock; 1-D radial run on the regularized axis |

## Output and restart

- **Checkpoints.** `save_checkpoint` / `load_checkpoint!` write one binary file
  per rank and restart onto the same grid and decomposition; with `using HDF5`,
  `save_checkpoint_hdf5` / `load_checkpoint_hdf5!` use one shared `.h5` and
  restore onto **any** rank count, for resuming at a different scale.
- **Visualization.** `save_vtk` writes a parallel VTK container (curvilinear for
  a resolved polar grid, rectilinear otherwise); `save_hdf5` writes one shared
  `.h5` per frame with an XDMF sidecar. Both take `fields` (primitives plus
  derived fields such as `:vorticity`, `:qcriterion`, `:schlieren`, and the
  artificial-property internals), `stride` subsampling, and `slice`.
- **Time series.** `Callback(EveryTime(Δt), FieldWriter("out/field"))` dumps on
  a time schedule and writes a `.pvd` collection to animate against time.

## Testing

```
julia --project=. test/runtests.jl                      # serial unit tests
mpiexec -n 4 julia --project=. -t 1 test/mpi_tests.jl   # distributed (phases=a,b selects)
julia --project=. -t auto test/convergence.jl           # order studies
julia --project=. -t auto test/validation.jl            # shock-capturing battery
```

The serial suite covers operator accuracy, closure exactness, the
coordinate-singularity folds, freestream preservation in every metric, EOS
round-trips, discrete conservation, NSCBC, and a Sod tube against the exact
Riemann solution. The MPI suite exercises the distributed
spike solve, halo exchange, off-rank folds, and the GCL across rank boundaries. 
`convergence.jl` tests spatial order of accuracy (≈6 / ≈10 in
the interior, ≈3 at closures and folds) against regression, and `validation.jl`
runs Lax, Sedov–Taylor, Noh, Shu–Osher, and Woodward–Colella.

## Status and limitations

Research code under active development. The core solver (compact operators,
LSRK(5,4) integration, artificial properties, multicomponent transport, the
curvilinear metrics, and the distributed solve) is covered by the suites above
and validated against analytic references.

- Float64 by default; a uniform Float32 mode (CPU and GPU) halves the memory
  footprint but carries a mean-density drift of order 1e-4.
- A converging strong shock is CFL-limited at the symmetry cell, where the state
  loses positivity as the shock forms: 0.4 at the spherical origin, 0.2 at the
  cylindrical axis and the planar wall. `StepControl(retries=4)` handles it
  automatically, and `ArtParams(detector=:d8)` lifts the restriction at the
  planar wall and cylindrical axis at the cost of the origin (0.4 → 0.25).
- Converging-shock runs carry cells of negative internal energy at the front,
  while density, total energy, and the Noh plateau stay sound (within 0.07%); an
  optional floor repairs negative-energy cells.
- `Transport` uses constant properties (the bundled NASA transport table is not
  yet wired in); NSCBC inflow transverse terms are not implemented.
- The GPU backend runs a single patch per solver with host-staged MPI, and
  refinement is Cartesian-only, single-region, with a small interface
  conservation drift.
- Wall boundary conditions assume coordinate-surface walls; Soret/Dufour and
  reacting chemistry are not built in (reactions can use the source interface).

## Learn more

The [documentation](https://stillyslalom.github.io/CompactLES.jl/dev/)
covers the frontend, compact operators, parallel decomposition, diagnostics, and
validation.
