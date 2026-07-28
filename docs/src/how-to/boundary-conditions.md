# Choose boundary conditions

Choose a boundary condition from the physical information available at the
face, not from the desired visual appearance of the solution.

| Physical boundary | Condition | Required information |
|:--|:--|:--|
| Periodic continuation | [`PeriodicBC`](@ref) | matching opposite face |
| Inviscid or symmetry wall | [`SlipWallBC`](@ref) | wall normal |
| Viscous solid wall | [`NoSlipWallBC`](@ref) | optional wall temperature |
| Supersonic or deliberately forced state | [`DirichletBC`](@ref) | full state as a function of position and time |
| Subsonic inflow | [`NSCBCInflowBC`](@ref) | velocity, temperature, composition |
| Subsonic outflow | [`NSCBCOutflowBC`](@ref) | far-field pressure and relaxation scale |
| Simple zero-normal-gradient approximation | [`ExtrapolationBC`](@ref) | no target state |
| Cylindrical axis | [`AxisBC`](@ref) | cylindrical metric and valid folded layout |
| Spherical origin or poles | [`OriginBC`](@ref), [`PoleBC`](@ref) | spherical metric and valid antipodal layout |

## Walls

`SlipWallBC()` removes normal velocity while retaining tangential velocity.
`NoSlipWallBC()` sets all velocity components to zero. The no-slip wall is
adiabatic by default; pass a finite temperature for an isothermal wall:

```julia
NoSlipWallBC(Twall = 300.0)
```

The thermodynamic wall state is computed through the selected EOS.

## Imposed full-state forcing

Use a [`DirichletBC`](@ref) when the complete state is physically prescribed,
as for a piston or a supersonic inlet:

```julia
driver(x, y, z, t) = Prim(
    u = (0.1sin(2pi * t), 0.0, 0.0),
    p = 1.0,
    rho = 1.0,
)

DirichletBC(driver)
```

The function is evaluated at Runge--Kutta stage time. A full-state condition
over-constrains subsonic flow and ordinarily reflects acoustic waves; use a
characteristic inflow there.

## Characteristic boundaries

For a constant subsonic inflow target:

```julia
NSCBCInflowBC(u = (0.2, 0.0, 0.0), T_ion = 1.0, Y = [1.0])
```

For a subsonic outflow:

```julia
NSCBCOutflowBC(pinf = 1.0, sigma = 0.25)
```

`pinf` is a relaxation target rather than a hard boundary pressure. Reducing
`sigma` transmits an outgoing transient with weaker pressure anchoring; making
it too small permits slow pressure drift. See
[Characteristic open boundaries](@ref) for the wave interpretation and model
limitations.

## Change a boundary during a run

Wrap two compatible conditions in [`SwitchableBC`](@ref), then switch from a
globally consistent callback:

```julia
face = SwitchableBC(
    SlipWallBC(),
    NSCBCOutflowBC(pinf = 1.0),
)

change = Callback(AtTime(0.2), (solver, Q) -> switch!(face))
run!(solver, Q; tfinal = 1.0, callback = change)
```

Every rank must switch at the same completed step because one condition may
enter MPI collectives that the other does not. `AtTime`, `EveryTime`, and
`WhenState` supply consistent trigger decisions. Do not call `switch!` from an
unreduced rank-local test.

Both wrapped conditions must agree on periodicity. Coordinate-fold conditions
cannot be wrapped because setup must identify them before constructing the
operator plans.
