# Diagnostics

```@meta
CurrentModule = CompactLES
```

All diagnostics account for metric quadrature. Scalar results and profiles are
reduced over `solver.decomp.comm` and must be called on every rank.

For a patched or refined solver, pass the state vector returned by
[`allocate_state`](@ref), rather than its root entry. The vector forms assemble
composite quadrature: a coarse cell covered by a child contributes only its
uncovered fraction, so it is not counted again beside the fine cells. Profiles
are reported at root-grid stations. This makes the result a useful physical
diagnostic of the composite mesh, but AMR transfer does not supply flux
refluxing, so a changing integral can still reveal coarse--fine budget error.

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
