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

followed by one compact smoothing pass. A fourth difference is small on a
well-resolved smooth field and large where variation approaches the grid scale.
Because it is undivided, its response contracts under refinement for a fixed
smooth physical field.

## Four artificial properties

`ArtParams` controls:

- `C_mu`: artificial shear viscosity from high derivatives of strain;
- `C_beta`: artificial bulk viscosity from the same sensor;
- `C_kappa`: artificial conductivity from internal-energy variation; and
- `C_D`: per-species artificial diffusivity from mass-fraction variation.

The resulting fields `mu_art`, `beta_art`, `kappa_art`, and `D_art` enter the
same stress, heat, and species fluxes as molecular transport. Setting
`enabled=false` skips their construction.

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

Near a closed edge, the first point is unchanged and the next rows use a
reduced-order cascade. `filter_interval=1` filters every conserved component
after every completed step. Zero disables state filtering.

Filtering and artificial transport are not interchangeable. The filter acts on
the grid-scale content of the conserved state whether or not a shock sensor is
active. Current Taylor--Green measurements show that it supplies a substantial
part of the energy sink, and removing it can destabilize a calculation even
when artificial properties remain enabled.

## Stability and timestep coupling

Larger artificial coefficients increase diffusive stability rates and can make
the explicit timestep much smaller. Converging strong shocks separately require
CFL at or below roughly 0.15 while the shock forms at a symmetry plane, a
restriction measured to originate at the wall, axis or origin cell rather than
in the artificial properties. Retry control is often cheaper than imposing that
small CFL throughout a calculation:

```julia
Numerics(
    n_global = (512, 1, 1),
    art = ArtParams(enabled = true),
    cfl = 0.5,
    control = StepControl(retries = 4),
)
```

The conductivity scale used by the gas models behaves as `rho*c/T_ion` and is
singular as temperature approaches zero. Extremely cold nondimensional ambient
states can therefore collapse the diffusive timestep.

## Selecting coefficients

The defaults are a starting point, not a material model. For a new regime:

1. choose independent reference problems representing its shocks, contacts,
   and vortical flow;
2. inspect both solution error and artificial-property fields;
3. measure resolved and artificial dissipation where relevant;
4. repeat across resolution and CFL; and
5. record the filter strength and cadence with every coefficient result.

`reference/CALIBRATION.md` records the current parameter sweeps, rejected
hypotheses, and identified operating limits. Those measurements should not be
generalized beyond the documented configurations without new evidence.
