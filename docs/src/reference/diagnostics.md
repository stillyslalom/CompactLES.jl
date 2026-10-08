# Diagnostics

```@meta
CurrentModule = CompactLES
```

All diagnostics account for metric quadrature. Scalar results and profiles are
reduced over `solver.decomp.comm` and must be called on every rank.

For a patched or refined solver, pass the state vector returned by
[`allocate_state`](@ref), rather than its root entry. The vector forms assemble
composite quadrature: a coarse node a child covers entirely contributes
nothing, a coarse node on a child's face contributes its whole cell, and the
child's nodes count from the third in from that face, so the two grids' cells
meet half a coarse cell inside the child and nothing is counted twice.
Profiles are reported at root-grid stations. Where the coarse--fine coupling
reconciles the face fluxes, while a feature the coarse grid does not resolve
crosses the face, it conserves this integral up to a term on the child's
nodes beside the face that vanishes where the flow there is uniform; elsewhere
a changing integral reveals the coupling's budget error.

## Integral and profile operations

```@docs
volume_integral
volume_average
domain_volume
plane_profile
profile_coordinate
profile_spacing
```

## Mixing measures

```@docs
mix_width
molecular_mixing
species_pdf
```

## Turbulence and dissipation

```@docs
tke_profile
turbulent_kinetic_energy
dissipation_rate
```
