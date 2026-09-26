# Choose boundary conditions

Choose a boundary condition from the physical information available at the
face, not from the desired visual appearance of the solution.

## Lay out the six faces

`bcs` has one entry per coordinate direction. A bare condition applies at both
ends, while a two-tuple makes the low and high faces different:

```julia
bcs = (
    (NSCBCInflowBC(u=(0.2, 0, 0), T_ion=1.0), NSCBCOutflowBC(pinf=1.0)),
    PeriodicBC(),
    SlipWallBC(),
)
```

The first entry above is an asymmetric streamwise pair; the second and third
use the same object at both faces. A periodic or collapsed direction must be
periodic at both ends. If a mutable [`SwitchableBC`](@ref) is used bare or on
both sides of a pair, both faces share its state: one `switch!` changes both.
Construct two wrappers when the two faces must switch independently.

| Physical boundary | Condition | Required information |
|:--|:--|:--|
| Periodic continuation | [`PeriodicBC`](@ref) | matching opposite face |
| Inviscid or symmetry wall on a node | [`SlipWallBC`](@ref) | wall normal |
| Symmetry plane half a cell outside the end node | [`SymmetryPlaneBC`](@ref) | wall normal; Cartesian or cylindrical z; a single unrefined patch |
| Viscous solid wall | [`NoSlipWallBC`](@ref) | optional wall temperature |
| Supersonic or fully prescribed state | [`DirichletBC`](@ref) | full state as a function of position and time |
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

All three wall conditions are impermeable and noncatalytic: every species has
zero normal flux, including molecular diffusion, artificial diffusion, and the
artificial `:bulk` species channel. The default adiabatic no-slip wall also has
exactly zero normal total-energy flux. An isothermal wall permits conductive
heat exchange with the molecular thermal conductivity selected by `transport`
plus `kappa_art`; its normal
energy flux contains no species enthalpy diffusion or bulk component diffusion.
Pressure and viscous wall stresses remain in the momentum flux.

The slip wall is a symmetry plane, so it is adiabatic and its normal
total-energy flux is exactly zero as well. Its tangential momentum fluxes are
zero too: the tangential velocity's normal derivative vanishes at a symmetry
plane, so there is no shear traction. The normal momentum flux remains and
carries the pressure, the normal viscous stress and the dilatational term.
Without the correction, a conductive heat flux and a shear traction remain at
the closure's truncation level, and under physical viscosity the near-wall
solution error stops converging.

These conditions are imposed on the assembled flux before compact divergence,
so they affect the nearby rows as well as the wall node. They do not establish a
global discrete conservation identity: compact differentiation, domain
quadrature, state enforcement at an isothermal wall, and filtering each have
their own budget contribution. Measure those separately when auditing heat or
species conservation.

### A symmetry plane without a node

`SymmetryPlaneBC()` places the reflecting plane half a cell outside the first
or last node instead of on it and continues the solution across it by parity,
so every operator applies its interior stencil and no closure row exists. The
slip wall's flux contract follows from the parities, with physical viscosity
as without it. Use it in place of `SlipWallBC()` wherever the face is a true
symmetry plane and the run is a single unrefined, unstretched patch. The grid
moves with it: the end node sits at `h/2` from the plane, with `h = L/(N − ½)`
for one plane and `L/N` for two, and a wall-normal profile station shifts by
half a cell. The condition cannot be switched during a run and is available
on every Cartesian dimension and on cylindrical z.

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

For forcing whose width should track the local mesh, use the five-argument
form. The final argument is the smallest physical spacing at that point after
stretching, coordinate-metric scaling, and AMR refinement; collapsed directions
do not participate.

```julia
driver(x, y, z, t, h) = Prim(
    u = (0.1sin(2pi * t) * tanh((x - 0.2) / (3h)), 0.0, 0.0),
    p = 1.0,
    rho = 1.0,
)
```

The original four-argument form remains valid. If a callable supports both
forms, CompactLES selects the longer one.

## Characteristic boundaries

For a constant subsonic inflow target:

```julia
NSCBCInflowBC(u = (0.2, 0.0, 0.0), T_ion = 1.0, Y = [1.0])
```

An inflow target may similarly vary at every stage and use local spacing:

```julia
target(x, y, z, t, h) = Prim(
    u = (0.2 * tanh((y - 0.5) / (2h)), 0.0, 0.0),
    rho = 1.0,
    T_ion = 1.0,
)
inlet = NSCBCInflowBC(u=(0.2, 0.0, 0.0), T_ion=1.0, Y=[1.0], target=target)
```

The target must specify `T_ion` and every species mass fraction. Its compatible
four-argument form `(x, y, z, t)` remains valid, and the five-argument form is
chosen when both apply. Pointwise NSCBC targets remain host-only.

A [`TurbulentInflow`](@ref) is such a target: the mean state plus a synthetic
velocity fluctuation with a prescribed Reynolds-stress tensor and integral
length scale, convected through the face at the mean velocity. The field is a
fixed function of position and time for a given `seed`, so a decomposed run
and a restarted one see the same inflow. The default relaxation rates damp the
fluctuation at the face; raise `eta_u` until the relaxation time
`Lref / (eta_u c)` is short against the passage time `length_scale / |u|`.

```julia
mean = Prim(u = (0.3, 0.0, 0.0), p = 1.0, T_ion = 1.0)
turbulence = TurbulentInflow(mean; length_scale = 0.05, intensity = 0.05,
                             seed = 1)
inlet = NSCBCInflowBC(mean; target = turbulence, eta_u = 5.0, eta_T = 5.0)
```

Pass `reynolds_stress` (a symmetric 3×3 matrix) in place of `intensity` for an
anisotropic tensor. The same object serves as a [`DirichletBC`](@ref) target at a
supersonic inflow.

For a subsonic outflow:

```julia
NSCBCOutflowBC(pinf = 1.0, sigma = 0.25)
```

`pinf` is a relaxation target, not a hard boundary pressure. Reducing
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

## Divide one face among conditions

[`CompositeBC`](@ref) assigns each point of a face to one of several member
conditions through a function of the point's coordinates. A jet entering
through an orifice in a wall is one example:

```julia
orifice = NSCBCInflowBC(u = (0.0, 0.3, 0.0), T_ion = 1.0,
                        target = (x, y, z, t) ->
                            Prim(u = (0.0, abs(x - 0.5) < 0.1 ?
                                          0.3 * cospi((x - 0.5) / 0.2)^2 : 0.0, 0.0),
                                 T_ion = 1.0, rho = 1.0))
face = CompositeBC((NoSlipWallBC(), orifice),
                   (x, y, z) -> abs(x - 0.5) < 0.1 ? 2 : 1)
bcs = (NoSlipWallBC(), (face, NoSlipWallBC()), PeriodicBC())
```

A characteristic inflow over the whole face with zero target velocity outside
the orifice does not work: the inflow formulation assumes that gas enters
everywhere on the face, and the flow the jet entrains leaves through the part
of the face at rest. With the composite, that part is a wall.

Every member runs on every rank, including the collectives of the
characteristic conditions, and the composite keeps each member's result where
the selector chose it. The change between members is sharp, so an inflow
member's target velocity should fall to zero at the edge of its region.
