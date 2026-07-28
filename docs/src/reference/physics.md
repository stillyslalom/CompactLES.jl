# Physics models

```@meta
CurrentModule = CompactLES
```

## Equation sets

```@docs
EquationSet
NavierStokes1T
```

## Equation-of-state interface and ideal gases

```@docs
EOS
nspecies
IdealSpecies
IdealMixture
single_species
Transport
```

## Temperature-dependent thermodynamics

```@docs
Nasa9Interval
Nasa9Species
Nasa9Mixture
nasa9_constant_cp
read_nasa9
```

## Condensed-material approximation

```@docs
StiffenedGas
```

## Artificial properties

```@docs
ArtParams
```

## Explicit sources

Source collections are concrete tuples stored on `Problem`. Custom source
types extend `add_source!`; `add_sources!` applies the complete tuple during an
RHS evaluation.

```@docs
ConstantBodyForce
add_source!
add_sources!
```
