# Problem setup

```@meta
CurrentModule = CompactLES
```

## Pointwise state and frontend

```@docs
Prim
Problem
Numerics
setup
initialize!
conserved_from_prim
tanh_blend
```

## Regions

```@docs
Regions
Regions.Layers
Regions.Layer
Regions.Cells
Hydrostatic
Regions.Shape
Regions.Slab
Regions.Box
Regions.Ellipsoid
Regions.Sphere
Regions.Cylinder
Regions.LevelSet
Regions.signed_distance
Multimode
Ramp
TurbulentInflow
```

## Thermodynamic states and shock relations

```@docs
thermodynamic_state
mass_fractions
mole_fractions
shock_jump
driver_pressure
reflected_shock
shock_tube
riemann_interface
```

## Coordinate indexing

```@docs
xcoord
global_xcoord
padded_index
interior_index
```
