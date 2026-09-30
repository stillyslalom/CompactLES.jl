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
CompactLES.FaceConditions
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

## Folds

```@docs
AxisBC
OriginBC
PoleBC
SymmetryPlaneBC
```

## Changing a condition during a run

A run that changes a boundary condition ends at the change and continues in a
second phase built by [`setup`](@ref)`(solver, Q; bcs)`; see
[Change a boundary during a run](@ref).

## Composite faces

```@docs
CompositeBC
```

## Developer internals: patch-interface markers

`Solver` places these on internal patch faces itself; they are not
user-supplied conditions and are shown only to explain cross-references.

```@docs
CompactLES.InterfaceBC
CompactLES.CoarseFineBC
```
