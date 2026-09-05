# Thermodynamics and species transport

## Why an EOS is required

The conserved state provides density, momentum, and total energy. Fluxes and
timesteps additionally require pressure, temperature, sound speed, heat
capacity, and species enthalpies. An [`EOS`](@ref) supplies that closure.

## Calorically perfect ideal mixtures

An [`IdealSpecies`](@ref) has a constant specific gas constant `R` and heat
capacity ratio `gamma`. The mixture obeys

```math
p=\rho R_m T, \qquad
R_m=\sum_k Y_kR_k,
```

with constant species heat capacities

```math
c_{v,k}=\frac{R_k}{\gamma_k-1},\qquad
c_{p,k}=c_{v,k}+R_k.
```

Use `IdealSpecies("gas"; R=1.0, gamma=1.4)` for an explicit one-species gas;
`Problem` promotes it internally to the one-species [`IdealMixture`](@ref)
required by the solver. `IdealSpecies("CO2")` and other named species can be
sampled from the bundled NASA-9 database at the reference temperature
(298.15 K), producing a calorically perfect reference-state approximation.
`IdealMixture(["He", "CO2"])` is the concise named-species constructor for a
constant-cp mixture. This model is appropriate when the relevant temperature
range is narrow enough that heat-capacity variation is negligible.

## NASA-9 mixtures

[`Nasa9Mixture`](@ref) remains thermally ideal but evaluates piecewise
polynomials for each species heat capacity and enthalpy. Recovering temperature
from internal energy requires a bounded Newton iteration at each point.
Use `Nasa9Mixture(["He", "CO2"])` for a database-backed mixture, or construct
it from explicit [`Nasa9Species`](@ref) records when supplying a custom table.

[`read_nasa9`](@ref) reads the bundled NASA CEA database and derives specific
gas constants from molar mass. Its default `reference=:sensible` shifts the
enthalpy gauge so each species has zero enthalpy at 298.15 K. Use
`reference=:formation` when absolute formation enthalpy is required by a model
that interprets it.

The database fit has finite temperature intervals. Extrapolating far outside
them is not a validated thermodynamic model even if polynomial evaluation
returns a finite number.

## Stiffened gas

[`StiffenedGas`](@ref) is a single-component condensed-material approximation:

```math
p=(\gamma-1)\rho e-\gamma p_\infty.
```

The cohesive pressure `p_inf` raises sound speed at a given ordinary pressure.
Setting it to zero recovers the perfect-gas algebra. A stiffened gas is not a
general liquid or solid EOS; parameters must be fitted over the intended state
range.

## Primitive-to-conserved conversion

`Prim` accepts pressure, velocity, composition, and one of temperature or
density. [`conserved_from_prim`](@ref) applies the selected equation set and EOS
to calculate partial densities and total energy. The inverse bulk conversion
occurs during each RHS evaluation.

Between completed steps, prefer state-query functions reading `Q`. Call
`refresh_primitives!` before using cached pressure or temperature fields in a
callback.

## Species diffusion

Molecular diffusivity is presently common across species and is set by
`mu0/Sc`, the molecular viscosity over the Schmidt number. Artificial
diffusivity may differ by species because each mass fraction has its own sensor.
The correction-velocity flux

```math
\boldsymbol{J}_k=-\rho D_k\nabla Y_k
+\rho Y_k\sum_jD_j\nabla Y_j
```

enforces zero total diffusive mass flux. The energy flux includes species
enthalpy transport `sum(h_k J_k)`.

## Extension contract

A new EOS must define the complete closure used by the solver: the species
count and names, the bulk conserved-to-primitive recovery, the
primitive-to-conserved conversion, the species enthalpy, the pressure-to-energy
derivative and its composition derivatives used by NSCBC, the
artificial-conductivity scale, and the internal energy at an isothermal wall.
[Extending CompactLES](@ref) lists the hook for each of these with its
signature; the comment at the top of `src/physics.jl` records the mathematical
contracts.

These calls occur behind array-level function barriers. Dynamic dispatch is
therefore paid once per pass, not at every grid point.
