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

## Reading a solver's domain

[`boundary_conditions`](@ref) and [`domain_bounds`](@ref) return the face
conditions and the bounds of the domain a solver was built on, and
[`profile_coordinate`](@ref) the node coordinates along one dimension, stretch
mapping and fold offset included. The output files record the same domain; see
[Read the record of an output file](@ref).

```@docs
boundary_conditions
domain_bounds
```

## Developer internals: patch-interface markers

`Solver` places these on internal patch faces itself; they are not
user-supplied conditions and are shown only to explain cross-references.

```@docs
CompactLES.InterfaceBC
CompactLES.CoarseFineBC
```
