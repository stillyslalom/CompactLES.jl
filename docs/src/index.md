# CompactLES.jl

CompactLES is a compressible large-eddy/direct-simulation solver for the
multicomponent Navier--Stokes equations. It combines high-order compact finite
differences, compact filtering, localized artificial fluid properties,
low-storage Runge--Kutta time integration, and distributed MPI line solves.

![Taylor–Green vorticity, a multicomponent shock-interface interaction, and a
cylindrical converging shock](assets/readme_header.png)

The frontend separates a physical [`Problem`](@ref)—EOS, transport, geometry,
boundary conditions, sources, and initial state—from [`Numerics`](@ref), which
contains the grid, algorithms, CFL, and process decomposition. One problem can
therefore be studied at several resolutions without rewriting its physics.

## Choose a path

- **Learn by running a calculation.** Begin with
  [Coalescing shock](@ref), a one-dimensional sound wave that steepens into
  a shock, checked against its exact solution.
- **See longer calculations against references.** The examples, starting
  with [Shock-capturing tests](@ref), show figures computed once and stamped
  with the commit and settings that produced them.
- **Build an input deck quickly.** Use the [Cheat sheet](@ref) for
  the complete constructor vocabulary, defaults, and common recipes.
- **Complete a specific task.** Use the how-to guides to
  [Define a problem](@ref), [Choose boundary conditions](@ref),
  [Control a run](@ref), or [Write output and restart](@ref).
- **Understand the model.** Start with [Governing equations](@ref), then read
  the explanation of discretization, regularization, thermodynamics, geometry,
  open boundaries, and parallel algorithms.
- **Look up exact behavior.** The reference section separates the exported
  input/runtime API, supported advanced numerical and extension APIs, and
  developer internals rendered only for cross-references.

## How the tutorials build

The tutorials are ordered so that each adds one layer to the preceding
calculations. [Coalescing shock](@ref) introduces the state, grid,
CFL-controlled time advancement, and output path. [Shock tube](@ref) adds
multicomponent gases, initial conditions built from regions, and shock
regularization. [Acoustic interface](@ref) adds regions whose state varies
in space, and follows a sound pulse through the interface between two gases.
[Sound absorption](@ref) adds molecular viscosity and heat conduction.
[Loschmidt cell](@ref) adds binary diffusion from tabulated coefficients and
measures the coefficient back from the decay of a composition mode.
[Richtmyer–Meshkov instability](@ref) adds a second grid dimension, symmetry
planes and characteristic inflow and outflow faces, and compares the growth of
a shocked interface with Richtmyer's model.
[Rayleigh–Taylor instability](@ref) adds a body force and an initial state in
hydrostatic balance with it, and compares the growth of a diffuse interface
with linear theory.
[Supernova remnant](@ref) adds a spherical grid resolved in radius alone,
continued through its origin, and astrophysical units, and compares the growth
of a blast wave with the Sedov–Taylor solution.
[Imploding shock](@ref) adds a refined level that follows a shock converging
on the axis of a cylinder, and compares the shock trajectory with
Guderley's self-similar law. [Advected bubbles](@ref) adds a refined level
made of tiles, which follows three bubbles of different gases across the
periodic faces of a box, and compares each with its exact, translated shape.
[Axis-crossing vortex](@ref) resolves the angle of a cylinder, holds an exact
solution on its outer boundary, and carries a vortex across the axis.
[Oscillating sphere](@ref) resolves the polar angle of a spherical grid,
continues the flow through the poles, and compares the sound radiated by an
oscillating sphere with the exact dipole field.

The tutorials give enough explanation to run and interpret each calculation.
For the mathematical development, read [Governing equations](@ref) followed by
[Discretization](@ref); the later explanation pages deepen
the regularization, thermodynamics, and geometry introduced along the way.

## Prerequisites and conventions

The manual assumes graduate coursework in fluid mechanics, thermodynamics,
vector calculus, and numerical partial differential equations. It does not
assume prior knowledge of Lele compact differences, Cook artificial
properties, NSCBC, coordinate folds, or distributed banded solves; those are
introduced before their solver-specific details.

Inputs may use SI or consistently nondimensional variables. The package does
not attach units, so mixing unit systems is not detected automatically.

## Scope

The current equation set has one temperature and no built-in reactions.
Molecular transport can use constant properties or temperature-dependent CEA
fits, with unity-Lewis diffusion or mixture-averaged diffusion from supplied
binary data. The bundled neutral-gas correlations feed the latter through
validated polynomial fits with collective runtime domain checks; see
[Thermodynamics and transport](@ref).

Cartesian, cylindrical, and spherical coordinates are available, including
regularized axes, origins, and poles. Unstretched Cartesian runs also support
nested refinement, lattice tiles, dynamic regridding, and optional
Berger–Oliger subcycling. [`AMR`](@ref) groups region selection, tagging, and
time stepping. CPU and device backends support MPI decomposition and refined
layouts. [Adaptive mesh refinement](@ref) describes their setup and restrictions.

These paths do not all have equal maturity. Refinement balances the fluxes
across a coarse–fine interface only where a feature the coarse grid does not
resolve is crossing it, and only on the host backend; elsewhere filtering and
interface coupling can change composite conserved quantities by amounts that
fall with the spacing.
Each explanation page states the relevant evidence and limitations.

!!! warning "Research software"
    CompactLES is research code under active development. Validate a
    configuration against an analytic, experimental, or independently
    implemented reference before drawing scientific conclusions from it.
