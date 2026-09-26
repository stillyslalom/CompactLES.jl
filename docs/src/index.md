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
  [Your first CompactLES simulation](@ref), a one-dimensional acoustic pulse
  that builds an `x`--`t` diagram.
- **Build an input deck quickly.** Use the [Input deck cheat sheet](@ref) for
  the complete constructor vocabulary, defaults, and common recipes.
- **Complete a specific task.** Use the how-to guides to
  [Define a problem](@ref), [Choose boundary conditions](@ref),
  [Control and diagnose a run](@ref), or [Write output and restart](@ref).
- **Understand the model.** Start with [Governing equations](@ref), then read
  the explanation of discretization, regularization, thermodynamics, geometry,
  open boundaries, and parallel algorithms.
- **Look up exact behavior.** The reference section separates the exported
  input/runtime API, supported advanced numerical and extension APIs, and
  developer internals rendered only for cross-references.

## How the tutorials build

The tutorials are ordered so that each adds one layer to the preceding
calculations. [Your first CompactLES simulation](@ref) introduces the state,
grid, CFL-controlled time advancement, and output path. The shock tube adds
filtering and artificial transport, and the multicomponent example adds
thermodynamic closure and species storage. [Evolve a molecular mixing layer](@ref)
then introduces binary diffusion. [Follow a moving feature with refinement](@ref)
shows how a physical-coordinate selector places fine cells. The geometry
sequence starts with a collapsed radial calculation before resolving a full
cylinder and, finally, a sphere with an origin and poles.

The tutorials give enough explanation to run and interpret each calculation.
For the mathematical development, read [Governing equations](@ref) followed by
[Spatial and temporal discretization](@ref); the later explanation pages deepen
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
[Thermodynamics and species transport](@ref).

Cartesian, cylindrical, and spherical coordinates are available, including
regularized axes, origins, and poles. Unstretched Cartesian runs also support
nested refinement, lattice tiles, dynamic regridding, and optional
Berger–Oliger subcycling. [`AMR`](@ref) groups region selection, tagging, and
time stepping. CPU and device backends support MPI decomposition and refined
layouts. [Adaptive mesh refinement](@ref) describes their setup and restrictions.

These paths do not all have equal maturity. Refinement has no conservative
refluxing (a correction that balances fluxes across coarse–fine interfaces),
and filtering and interface coupling can change composite conserved quantities.
Each explanation page states the relevant evidence and limitations.

!!! warning "Research software"
    CompactLES is research code under active development. Validate a
    configuration against an analytic, experimental, or independently
    implemented reference before drawing scientific conclusions from it.
