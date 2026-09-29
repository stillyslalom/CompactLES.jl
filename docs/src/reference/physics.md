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
ConstantTransport
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

## Neutral binary diffusion data

Three vendored dilute neutral-gas sources and the polynomial fits built
from them: the evaluated correlations of Marrero and Mason (1972) for
H2-D2 and the combustion, air, noble-gas and atomic pairs of that paper;
the calculated hydrogen-isotopologue and helium pairs of Song et al.
(2016); and the room-temperature measurements of Müller and Klemm (1970).
Measured, evaluated and calculated values stay in separate tables.
The polynomial model is usable by `CeaTransport` with
`diffusion=:mixture_averaged` when the EOS species order and all required
pure-species CEA records match. Its pair temperature ranges are checked during
the solver's flux and timestep preflight; see
[Thermodynamics and species transport](@ref).

```@docs
DiffusionData
DiffusionData.MarreroMasonPair
DiffusionData.MARRERO_MASON_1972
DiffusionData.MARRERO_MASON_UNCERTAINTY
DiffusionData.marrero_mason_pair
DiffusionData.marrero_mason_pairs
DiffusionData.marrero_mason_diffusivity
neutral_binary_diffusion
DiffusionData.neutral_binary_diffusion_residual
temperature_domain
DiffusionData.neutral_binary_sources
DiffusionData.SongWangPair
DiffusionData.SONG_WANG_2016
DiffusionData.SONG_WANG_UNCERTAINTY
DiffusionData.song_wang_pair
DiffusionData.song_wang_diffusivity
DiffusionData.MuellerKlemmPair
DiffusionData.MUELLER_KLEMM_1970
DiffusionData.MUELLER_KLEMM_TEMPERATURE
DiffusionData.MUELLER_KLEMM_PRESSURE
DiffusionData.mueller_klemm_pair
```

## Ion transport reference

The standalone ion evaluator supplies a checked hot-plasma reference coefficient.
It does not provide a solver flux closure or a cold-to-warm material model.

```@docs
DiffusionData.StantonMurilloDiagnostics
DiffusionData.stanton_murillo_interdiffusivity
DiffusionData.H_ION_MASS
DiffusionData.D_ION_MASS
DiffusionData.T_ION_MASS
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

## Tabulated plasma equation of state and opacities

An IONMIX4 or IONMIX6 file, the format FLASH reads, tabulates a
one-temperature plasma equation of state and group opacities on a grid of
temperature and ion number density. The reader converts it to SI, and the
evaluators interpolate it bilinearly in the logarithms of temperature and mass
density and invert it for temperature along a density column. The table is
not an [`EOS`](@ref), so a solver does not evaluate it.

A SESAME library in the LANL ASCII 2 format tabulates, per material, the
pressure, specific energy and Helmholtz free energy on a grid of density and
temperature, in total and split into a cold curve, an ion part and an electron
part. The reader converts a material to SI and evaluates it with the same
interpolant, domain policy and inversion, or, where the free energy is
tabulated, derives energy and pressure from one interpolant of the free energy
so that they satisfy the thermodynamic consistency relation between nodes. No
SESAME table is distributed with the package.

```@docs
CompactLES.IonmixTable
CompactLES.read_ionmix
CompactLES.write_ionmix
CompactLES.SesameTable
CompactLES.SesameComponent
CompactLES.SesameColdCurve
CompactLES.read_sesame
CompactLES.write_sesame
CompactLES.table_value
CompactLES.table_opacity
CompactLES.table_state
CompactLES.table_temperature_status
CompactLES.table_temperature
CompactLES.TABLE_OK
```

## Condensed-material approximation

```@docs
StiffenedGas
```

## Artificial properties

```@docs
ArtificialProperties
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
