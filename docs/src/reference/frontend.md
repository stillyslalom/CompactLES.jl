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
Layers
Layer
Cells
Hydrostatic
Shape
Slab
Box
Ellipsoid
Sphere
Cylinder
LevelSet
signed_distance
Multimode
Ramp
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
gidx
interior_index
```
