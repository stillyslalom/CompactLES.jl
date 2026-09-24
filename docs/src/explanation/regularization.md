# Filtering and artificial properties

Compact central schemes have little intrinsic dissipation. That is valuable for
resolved turbulence and harmful at shocks or near the grid cutoff, where
uncontrolled dispersive oscillations can destroy positivity. CompactLES uses
two distinct regularization mechanisms: localized artificial fluid properties
and a compact low-pass filter.

## High-derivative sensors

The artificial-property sensors use undivided fourth differences,

```math
\delta^4 f_i = f_{i-2}-4f_{i-1}+6f_i-4f_{i+1}+f_{i+2},
```

followed by one smoothing pass, an explicit nine-point Gaussian by default. A
fourth difference is small on a well-resolved smooth field and large where
variation approaches the grid scale. Because it is undivided, its response
contracts under refinement for a fixed smooth physical field.

Each direction's contribution carries a length weight: the local physical
spacing along that direction, which is the computational spacing multiplied
by the metric scale factor and by any [`Stretch`](@ref) mapping's Jacobian,
raised to the power the sensor's field requires. On a uniform Cartesian grid
that weight is the constant grid spacing. Under a stretch, or along a
resolved angular direction, it varies from point to point, and the sensor
follows the mesh rather than the computational index.

## Sensor construction

Every sensor is assembled in the same three stages. A field is selected, a
high-pass operator is applied to it along each active direction, and the
directional results are combined and smoothed. [`ArtParams`](@ref) carries one
setting per stage. The default sensor construction follows
[Cook (2007, eqs. 15--21)](https://doi.org/10.1063/1.2728937). The alternatives
are exposed in
[Pyranda's public kernels](https://github.com/LLNL/pyranda/tree/master/pyranda/parcop);
the dilatation sensor also appears in
[Cook (2009, appendix A)](https://doi.org/10.1063/1.3139305).

| Setting | Default | Alternative |
|---|---|---|
| `mu_sensor` | `:strain`, the strain-rate magnitude | `:velocity`, the three velocity components |
| `beta_sensor` | `:strain` | `:ungated_dilatation`, the dilatation ``\nabla\cdot u`` |
| `detector` | `:delta4`, the undivided fourth difference above | `:d8`, a compact eighth derivative |
| `reduction` | `:sum`, summation over directions | `:max`, the directional maximum |
| `smoother` | `:gaussian`, an explicit nine-point stencil | `:compact`, one pass of the state filter |

Only the two viscosities have a choice of field. Conductivity is always built
from the specific internal energy and species diffusivity from the mass
fractions. Two further `beta_sensor` settings, `:gated_strain` and
`:dilatation`, multiply the strain and dilatation sensors by a Ducros
compression switch, which is zero in expansion and small where vorticity
dominates dilatation.

The field and the detector are not independent choices. The strain-rate
magnitude is a Euclidean norm, so it has a cusp wherever the strain passes
through zero, and a cusp is grid-scale structure at any resolution. A sensor
built from that field therefore responds to smooth flow as though it were
unresolved. The additional selectivity of `:d8` below the Nyquist limit is then
unavailable: the two detectors differ by at most a factor of 1.8 at any
wavelength, against a designed factor of 569 at eight points per wavelength.
The velocity components and the dilatation carry no cusp, and through those
fields the two detectors separate as designed.

One property is common to every field obtained by differentiation. A centered
scheme has zero modified wavenumber at the two-point wave, so both the
strain-rate magnitude and the dilatation vanish identically for a grid-to-grid
velocity oscillation, and the sensors built from them return zero there. Only
`mu_sensor = :velocity` responds to that mode. Grid-scale content of the
conserved state is removed by the compact filter, not by the artificial
properties, consistent with the measured Taylor--Green dissipation budget: at
128³ the filter supplies 37% of the energy sink and the artificial shear
viscosity 2%, and removing the filter ends the run while removing the
artificial properties does not.

The defaults are unchanged because the alternatives were measured and did not
improve the validation battery. The velocity field for artificial shear
viscosity moves no case beyond its fourth digit, and the dilatation field for
artificial bulk viscosity improves several cases while losing the converging
cylindrical case entirely. Changing a field also changes the magnitude of the
sensor that multiplies the coefficient, so a value fitted under one field is
only a starting point under another.

## Four artificial properties

`ArtParams` controls:

- `C_mu`: artificial shear viscosity from the sensor named by `mu_sensor`;
- `C_beta`: artificial bulk viscosity from the sensor named by `beta_sensor`,
  which under the default setting is the same sensor;
- `C_kappa`: artificial conductivity from internal-energy variation; and
- `C_D`: per-species artificial diffusivity from mass-fraction variation,
  which damps the ringing inside `[0, 1]` behind a shocked interface,
  with `C_Y` the bound that raises it wherever a mass fraction leaves
  `[0, 1]` by more than `Y_tolerance`.

The resulting fields `mu_art`, `beta_art`, `kappa_art`, and `D_art` enter the
same stress, heat, and species fluxes as molecular transport. Setting
`enabled=false` skips their construction.

### Three forms of the species channel

`species_flux` selects how the artificial species diffusivity enters the
equations. All three forms use the same `C_D` and `C_Y`.

The default `:partial_density` diffuses each partial density with one
diffusivity shared by every species, built from the sensor on both the mass
and the mole fraction of every species. The mass it moves carries its own
momentum and kinetic energy, and each species carries its internal energy,
following [Brill, Olson & Bokman (2025)](https://arxiv.org/abs/2503.12680).
A state of uniform velocity, pressure and temperature therefore stays
uniform to round-off whatever the composition, and the channel adds no
viscosity or conduction of its own.

`:fickian` is Cook's per-species flux of the form the molecular one takes,
with the correction velocity that keeps the fluxes summing to zero and the
enthalpy flux in the energy equation. It moves no bulk mass. Between gases
of unequal molecular weight, a mass flux that sums to zero still carries a
net volume, so at a composition interface the enthalpy flux changes the
pressure of a state whose velocity, pressure and temperature are uniform.
The error is proportional to `C_D`. On an advected interface at a density
ratio of a hundred it reaches a percent of the pressure, and behind a
shocked helium/carbon dioxide interface it radiates sound four to five
cells in wavelength. It is the cheapest of the three forms.

`:bulk` diffuses every conserved variable with the shared diffusivity, the
parabolic regularization of
[Guermond & Popov (2014)](https://arxiv.org/abs/1212.5566). It holds a
uniform state as the default does. It also diffuses momentum and energy,
which acts as an added viscosity and conduction at the interface and damps
the roll-up of a shocked interface. As a continuous model it satisfies the
entropy inequality of the system; the discrete update does not inherit that
property.

Keep the default. Choose `:bulk` for a shocked interface at a density ratio
of a hundred or more, where its added dissipation holds the mass fractions
closer to [0, 1]. Choose `:fickian` to reproduce a code that uses Cook's
form. On a two-dimensional shocked helium/carbon dioxide interface the
default costs about a quarter more per step than `:fickian` and `:bulk`
about two fifths more, for the added gradient solves and the second sensor
field per species.

Bulk viscosity is the primary shock-spreading mechanism. Conductivity controls
thermal ringing and wall heating. Species diffusivity prevents an unresolved
composition interface from oscillating independently of density. Artificial
shear viscosity supplies subgrid dissipation in vortical flow, but its present
coefficient is not universally calibrated.

## Compact filtering

The eighth-order Gaitonde--Visbal filter solves a symmetric tridiagonal compact
relation. Its parameter `alphaf` lies between `-0.5` and `0.5`; values closer to
`0.5` are weaker. The default is

```julia
compact_filter(0.45)
```

Near a closed edge, the first point is unchanged and rows two to four use
the one-sided eighth-order rows of the same paper (`closures=:onesided`,
the default), which keep one pass eighth order along the whole line;
`closures=:cascade` substitutes the reduced-order cascade of centered
second-, fourth- and sixth-order filters, which caps a filtered wall at
second order and is the row set the `:cascade4` derivative closure needs.
`filter_interval=1` filters every conserved component after every
completed step. Zero disables state filtering.

`filter_cfl=0.35`, the default, makes the filter's dissipation a rate rather
than a per-application amount. Each directional pass relaxes the state
toward its filtered image with weight `filter_interval · dt · r_d · √n /
filter_cfl`, where `r_d` is the largest one-dimensional hyperbolic rate
`(|u_d| + c) / h_d` of the direction swept and `n` the number of active
dimensions, so a run at a lower CFL, a shortened step, a retry, a subcycled
level or a diffusion-limited step receives the same dissipation per unit
time, and a fine spacing in one direction leaves the passes along the others
unchanged. On an isotropic grid at or above the reference CFL a pass is
applied at full strength. `filter_cfl=0` restores the unrelaxed pass, whose
dissipation grows with the number of steps taken over an interval.

Filtering and artificial transport are not interchangeable. The filter acts on
the grid-scale content of the conserved state whether or not a shock sensor is
active. Current Taylor--Green measurements show that it supplies a substantial
part of the energy sink, and removing it can destabilize a calculation even
when artificial properties remain enabled.

## Stability and timestep coupling

Larger artificial coefficients increase diffusive stability rates and can make
the explicit timestep much smaller. `run!` evaluates the initial RHS before
choosing its first step, so startup estimates already include the artificial
coefficients. In the current converging-shock calibration, planar and
cylindrical cases complete from `cfl = 0.9` without retry; the spherical
origin requires a lower accepted CFL during the shock excursion. For a
converging shock at that origin, use `cfl = 0.3` under the default detector,
or allow retry control to reduce a larger initial CFL only when needed:

```julia
Numerics(
    n_global = (512, 1, 1),
    art = ArtParams(enabled = true),
    cfl = 0.5,
    control = StepControl(retries = 4),
)
```

The spherical-origin limit depends on the detector and the initial profile;
under `detector = :d8` the measured ceiling is 0.25. An initially unresolved
shock at the origin remains a separate limitation even when the first RHS is
primed. See the [calibration appendix](https://github.com/stillyslalom/CompactLES.jl/blob/main/reference/CALIBRATION_APPENDIX.md#recovery-strategy)
for the measured trajectories and accepted CFL values.

## Selecting coefficients

The defaults are a starting point, not a material model. For a new regime:

1. choose independent reference problems representing its shocks, contacts,
   and vortical flow;
2. inspect both solution error and artificial-property fields;
3. measure resolved and artificial dissipation where relevant;
4. repeat across resolution and CFL; and
5. record the filter strength and cadence with every coefficient result.

The parameter sweeps behind the defaults ran on the cases of `test/cases.jl`
and on Taylor--Green at 32³ to 128³. Those measurements should not be
generalized beyond those configurations without new evidence.
