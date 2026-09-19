# Extending CompactLES

```@meta
CurrentModule = CompactLES
```

This page defines the supported extension surface. The names below are public
bindings, but they are primarily for authors of custom physical models and
conditions rather than ordinary input decks. Implement methods on the concrete
type you own; do not modify solver arrays from a pointwise method unless its
contract explicitly permits it.

## EOS implementations

Subtype [`EOS`](@ref) and implement the following methods:

| Method | Responsibility |
|---|---|
| `nspecies(eos)` | Number of transported species |
| `species_names(eos)` | Optional labels in conserved/output order |
| `recover_primitives!(solver, eos, Q)` | Recover padded density, velocity, pressure, temperature, sound speed, heat capacity, and mass fractions |
| `conserved_from_prim(equations, eos, pr)` | Convert a [`Prim`](@ref) to the selected conserved layout |
| `species_enthalpy(eos, k, T_ion)` | Species specific enthalpy |
| `eos_phi(eos, rho, p, T_ion, cp_mix)` | Pressure-to-energy derivative used by NSCBC |
| `eos_dphi_dY(eos, k, rho, p, T_ion, cp_mix)` | Composition derivative of `eos_phi` |
| `artificial_conductivity_scale(eos, rho, c, T_ion, cp_mix)` | Conductivity scale per artificial sensor |
| `wall_internal_energy(eos, Q, I, n_species, Twall)` | Internal energy at an isothermal wall |

`species_names` may use the fallback labels when names are not meaningful.
Array-level recovery must leave finite placeholders at invalid padded points so
the following stencil passes remain safe. Keep the hot-loop methods concrete and
type-stable. See [Thermodynamics and species transport](@ref) for the physical
meaning of the derivatives.

```@docs
recover_primitives!
species_names
species_enthalpy
eos_phi
eos_dphi_dY
artificial_conductivity_scale
wall_internal_energy
```

## Molecular transport

Subtype [`AbstractTransport`](@ref) with the solver's floating-point type and
implement `transport_coefficients(model, eos, temperature, rho, cp, Y)`.
Return `(mu=..., kappa=..., D=...)`, with one molecular diffusivity per species
in `Y` order. The flux, timestep, wall, geometry, and dissipation paths all use
this coefficient interface; artificial coefficients are added separately.

Keep the model and returned coefficients concrete and allocation-free for
pointwise evaluation. The built-in CEA model stores fixed-size coefficient
tuples so its transport data can cross a device-kernel boundary. This hook
replaces scalar coefficient laws within the existing flux closure. Plasma
transport with pressure, temperature and electric driving terms requires
additional state and a constrained flux closure; supplying different `D` values
does not implement those capabilities.
The generic host adapter assembles `Y` from the species fields; models with a
fixed species count can specialize `CompactLES.transport_at` as the CEA model
does to keep tuple construction statically sized, which is needed for device
kernels and for an allocation-free custom-model path.

## Equation sets

Subtype [`EquationSet`](@ref) and provide its conserved layout, primitive
conversion, and `conserved_parity`. The parity tuple records how each conserved
component reflects across a coordinate fold; it must agree with the velocity and
species ordering used by the equation set. `NavierStokes1T` is the built-in
one-temperature reference implementation.

The current RHS and primitive storage remain specific to the supported
Navier–Stokes system. A new layout and parity method alone do not implement
multi-temperature, radiation or MHD evolution.

```@docs
conserved_parity
```

## Boundary conditions

Subtype [`BoundaryCondition`](@ref) and implement the operation(s) the
condition needs:

- `isperiodic(bc)` declares periodicity during setup;
- `enforce!(bc, Q, solver, dim, side)` writes a physical boundary state;
- `correct_flux!(bc, solver, Q, dim, side)` imposes the normal assembled flux
  before exchange and compact divergence;
- `correct_rhs!(bc, solver, Q, dQ, dim, side)` changes the boundary RHS;
- `validate_bc(bc, metric, eos, dim, side)` rejects incompatible geometry or EOS.
- `sensor_mirror(bc)` declares a reflecting wall, so that the artificial-property
  detector reads past the face from the node-centred mirror instead of clamping,
  and the sensor smoother and the `:d8` detector close the face with the
  node-centred rows of `wall_closures` in place of their half-offset ones.

Periodic and fold behavior is collective setup state. If a boundary method
enters a collective derivative or reduction, every rank must reach it in the
same order. Conditions that do not need an operation may rely on the default
method. See [Choose boundary conditions](@ref) and [Characteristic open
boundaries](@ref).

```@docs
enforce!
correct_flux!
correct_rhs!
validate_bc
sensor_mirror
isperiodic
```

## Triggers and callbacks

Subtype [`Trigger`](@ref) when a callback needs a new firing policy. Implement
`fired!(trigger, solver, Q) -> Bool`; the verdict must be identical on every
rank. Implement `next_time(trigger, solver) -> Real` when `run!` should shorten
a step to land on a scheduled time, and `rewind!(trigger, t, step)` when a
rollback must restore trigger state. Pair the trigger with
`Callback(trigger, effect!)`.

```@docs
fired!
next_time
rewind!
```

## Sources

Source objects extend [`add_source!`](@ref), which receives the solver state and
adds its contribution to the RHS. Store source collections as tuples on
`Problem` so setup and the RHS remain specialized. A source should be pure with
respect to the state except for its intended RHS contribution.

## API stability

The decomposition, directional-plan, patch, transfer, and synchronization
records used to execute these interfaces are developer internals. They may be
rendered in the reference for cross-links, but are not exported or part of the
input-deck API. See [Operators and decomposition](@ref) for the supported
advanced numerical choices.
