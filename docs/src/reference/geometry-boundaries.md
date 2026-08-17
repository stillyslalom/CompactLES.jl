# Geometry and boundaries

```@meta
CurrentModule = CompactLES
```

## Metrics and stretching

```@docs
Metric
CartesianMetric
CylindricalMetric
SphericalMetric
Stretch
sine_cluster
```

## Boundary interface and ordinary conditions

```@docs
BoundaryCondition
PeriodicBC
SlipWallBC
NoSlipWallBC
ExtrapolationBC
DirichletBC
```

## NSCBC types

```@docs
NSCBCInflowBC
NSCBCOutflowBC
```

## Coordinate folds

```@docs
AxisBC
OriginBC
PoleBC
```

## Time-dependent selection

```@docs
SwitchableBC
switch!
switched
```

## Patch-interface markers

`Solver` places these on internal patch faces itself; they are not
user-supplied conditions.

```@docs
InterfaceBC
CoarseFineBC
```
