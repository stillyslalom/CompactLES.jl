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

## Temperature-dependent molecular transport

[`Transport`](@ref) retains constant viscosity and Prandtl/Schmidt numbers.
For dimensional gas calculations, use the bundled CEA pure-species fits:

```julia
eos = Nasa9Mixture(["N2", "O2"])
transport = CeaTransport(eos)
```

Pass both objects to `Problem`. Temperatures are kelvin, density is kg/m³,
viscosity is Pa s, conductivity is W/(m K), and diffusivities are m²/s.
The model follows the EOS species order and works with both calorically
perfect and NASA-9 ideal mixtures. Missing pure-species fits are setup errors.
Outside a tabulated temperature range, the nearest interval is extrapolated;
this does not extend the fit's physical validation range.

[`read_cea_transport`](@ref) reads the fixed-column `data/trans.inp` table.
Each viscosity or conductivity fit has the form

```math
\ln f_k=A_k\ln T+B_k/T+C_k/T^2+D_k.
```

The reader preserves the temperature intervals and binary viscosity-interaction
records. The mixture model uses the pure-species fits: Wilke's viscosity rule
and the Wassiljewa conductivity approximation with the same viscosity-based
interaction weights. It does not reproduce CEA's equilibrium/reacting
conductivity or use its binary viscosity-interaction fits as diffusion data.
The fit format and units follow
[NASA's transport-data specification](https://www.grc.nasa.gov/www/winddocs/user/files.html).

Writing mole fractions as ``X_k=Y_kR_k/\sum_jY_jR_j``, the mixture rules are

```math
\mu=\sum_i\frac{X_i\mu_i}{\sum_jX_j\phi_{ij}},\qquad
\kappa=\sum_i\frac{X_i\kappa_i}{\sum_jX_j\phi_{ij}},\qquad
\phi_{ij}=\frac{[1+\sqrt{\mu_i/\mu_j}(M_j/M_i)^{1/4}]^2}
{\sqrt{8(1+M_i/M_j)}}.
```

The default `diffusion=:unity_lewis` uses ``D_k=\kappa/(\rho c_p)``.
The bundled table has no binary diffusivities. To select
`diffusion=:mixture_averaged`, supply `binary_diffusion=BinaryDiffusion(...)`
with a symmetric matrix of measured or independently modelled reference
diffusivities in EOS species order. The binary model scales them by
``(T/T_{ref})^n p_{ref}/p``; its exponent is configurable and the reference
data and scaling must be appropriate to the gas and temperature range.

[`BinaryDiffusionPolynomial`](@ref) supplies checked, species-labelled
pair-specific dilute-neutral-gas fits. It records exact species names, a
temperature range for each pair, and a polynomial in ``\log(T/T_{ref})``.
[`neutral_binary_diffusion`](@ref) builds one from the correlations of
Marrero and Mason (1972), vendored as [`MARRERO_MASON_1972`](@ref) in SI
units with each pair's stated temperature range and reliability group.
[`neutral_binary_diffusion_residual`](@ref) reports how far a fit departs
from its source, and [`temperature_domain`](@ref) the interval every pair
of a model shares. H2-D2 is the only hydrogen isotopologue pair that
paper correlates; the HD and tritiated pairs come from the calculated
fits of [`SONG_WANG_2016`](@ref), and the room-temperature measurements
of [`MUELLER_KLEMM_1970`](@ref) are the independent check on both. The
builder takes a `source` preference list and never estimates a pair no
source carries; [`neutral_binary_sources`](@ref) reports the choice per
pair. Pass the result as `CeaTransport(eos; diffusion=:mixture_averaged,
binary_diffusion=model)`, with the exact EOS species order; every species
also needs its own CEA viscosity and conductivity record. Thus the standalone
HD and tritiated molecular coefficients do not by themselves provide complete
solver transport. The polynomial does not supply atomic H/D/T or ionized-plasma transport, which needs coupled
driving forces and field closure rather than scalar binary diffusivities.

Polynomial diffusion uses strict source ranges, including pairs whose species
are locally absent. The solver checks temperature, pressure and coefficient
representability before flux and timestep evaluation, and rejects an invalid
state collectively with `SolverFailure(:transport_domain)`, including on
partially owned AMR levels. This error does not retry or follow
`StepControl.validity`; there is no extrapolation or clamping policy.

For the mass-fraction gradients used by the solver, the mixture coefficient is

```math
\frac{1}{D_i}=\sum_{j\ne i}\frac{X_j}{D_{ij}}
+\frac{X_i}{\sum_{j\ne i}X_jM_j}
\sum_{j\ne i}\frac{X_jM_j}{D_{ij}}.
```

This is the mass-gradient convention of
[Cantera's mixture diffusion coefficients](https://cantera.org/3.1/cxx/d5/de4/GasTransport_8cpp_source.html).
It gives the binary coefficient for both species in a binary mixture. At an
exactly pure composition the present species has zero diffusion coefficient;
absent species retain their trace diffusivity. Soret, Dufour, pressure
diffusion, and plasma transport are outside this gas model.

The same local coefficients enter the fluxes, wall heat transfer, geometric
stress terms, and diagnostics. The timestep uses the largest species
diffusivity and the thermal rate ``\kappa/(\rho c_v)``, including artificial
contributions; the thermal stability rate uses ``c_v``, while unity Lewis uses
``c_p``. Verification is recorded in the
[transport checks](https://github.com/stillyslalom/CompactLES.jl/blob/main/reference/CALIBRATION_APPENDIX.md#temperature-dependent-transport).

## Species diffusion

The constant [`Transport`](@ref) uses the same molecular diffusivity
`mu0 / (rho * Sc)` for every species. [`CeaTransport`](@ref) offers a unity-Lewis
fallback or species-specific mixture-averaged diffusivities from supplied
binary data. Artificial diffusivity may differ by species because each mass
fraction has its own sensor.
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
