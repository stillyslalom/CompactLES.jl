# Diagnostics

```@meta
CurrentModule = CompactLES
```

All diagnostics account for metric quadrature. Scalar results and profiles are
reduced over `solver.decomp.comm` and must be called on every rank.

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
