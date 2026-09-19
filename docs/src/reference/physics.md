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
Transport
state_admissibility
```

## Molecular transport

```@docs
AbstractTransport
CeaTransport
BinaryDiffusion
BinaryDiffusionPolynomial
binary_diffusivity
read_cea_transport
transport_coefficients
```

## Ion transport reference

The standalone ion evaluator supplies a checked hot-plasma reference coefficient.
It does not provide a solver flux closure or a cold-to-warm material model.

```@docs
StantonMurilloDiagnostics
stanton_murillo_interdiffusivity
H_ION_MASS
D_ION_MASS
T_ION_MASS
```

## Temperature-dependent thermodynamics

A caloric equation of state gives internal energy as a function of temperature,
so recovering temperature from a conserved state is an inversion rather than a
division. `mixture_temperature_status` reports whether that inversion succeeded
and whether it was evaluated inside the range the fits cover;
`mixture_temperature` is its value-only form, used by the per-point primitives
recovery.

```@docs
Nasa9Interval
Nasa9Species
Nasa9Mixture
nasa9_constant_cp
read_nasa9
CompactLES.mixture_temperature
CompactLES.mixture_temperature_status
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
types extend `add_source!`; see [Extending CompactLES](@ref). The internal
`CompactLES.add_sources!` applies the complete tuple during an RHS evaluation
and is rendered only for cross-references.

```@docs
ConstantBodyForce
add_source!
CompactLES.add_sources!
```
