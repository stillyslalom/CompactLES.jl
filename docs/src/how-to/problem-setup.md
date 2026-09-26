# Define a problem

A [`Problem`](@ref) describes physics independently of resolution and process
count. A [`Numerics`](@ref) supplies those numerical choices. Keeping the two
separate lets a convergence study reuse one physical specification unchanged.

## Define a pointwise state

An initial-condition function receives three physical coordinates and returns
a [`Prim`](@ref):

```julia
initial(x, y, z) = Prim(
    Y = (0.7, 0.3),
    u = (0.0, 0.0, 0.0),
    p = 101_325.0,
    T_ion = 300.0,
)
```

Specify exactly two of `p`, `T_ion`, and `rho`; the EOS determines the third.
Mass fractions must be in the same order as the EOS species and sum to one.
Initial-condition functions should be pure because setup may evaluate them
from multiple threads.

Use the optional fourth argument when the initial profile needs the local
mesh spacing. For example, regularize a sharp interface over two cells:

```julia
initial(x, y, z, h) = begin
    blend = tanh_blend(x, 0.5, 2h)
    Prim(Y=(1 - blend, blend), p=101_325.0, T_ion=300.0)
end
```

`h` is the smallest physical mesh spacing over resolved directions at that
point, including stretching and coordinate metric factors. A refined level
supplies its own spacing. Collapsed dimensions do not enter the minimum; an
entirely collapsed grid supplies zero. The original `(x,y,z)` form still works,
and a callable supporting both forms receives the longer one.

This keeps a numerical transition tied to resolution without capturing
`Numerics` inside `Problem`. A finite physical interface thickness should
instead remain fixed in physical units during a refinement study. A
two-cell transition describes regularization of a sharp interface, not a
material mixing length.

## Build a state from regions

A state made of several gases or several regions, such as a shocked gas, the
gas ahead of the shock and a second gas beyond an interface, is written as a
[`Layers`](@ref) initial condition: a background state overlaid by regions,
each a [`Shape`](@ref) paired with a `Prim`.

```julia
eos = Nasa9Mixture(["Air", "SF6"])
air = Prim(Y = mass_fractions(eos, "Air" => 1.0; basis = :mole),
           p = 101_325.0, T_ion = 295.0)
sf6 = Prim(Y = mass_fractions(eos, "SF6" => 1.0; basis = :mole),
           p = 101_325.0, T_ion = 295.0)
shocked = shock_jump(eos, air, 1.5).post

initial = Layers(air,
                 Slab(1, hi = 0.20) => shocked,
                 Slab(1, lo = (y, z) -> 0.25 + 0.002cos(2π * y / 0.05)) => sf6)
```

A later region covers an earlier one where they overlap. The shapes are
[`Slab`](@ref) (between two bounds along one axis, either of which may be a
function of the other two coordinates), [`Box`](@ref), [`Sphere`](@ref),
[`Ellipsoid`](@ref), [`Cylinder`](@ref) and [`LevelSet`](@ref) for any other
region, combined with `∪`, `∩`, `setdiff` and `!`. A shape ignores collapsed
directions, so a `Sphere` is a disc in a planar run.

Each region boundary is a tanh transition. `width = Cells(n)` gives it `n`
local mesh spacings, the default being three; a plain number gives a physical
thickness. Within a transition the states are mixed as volumes of gas: the
partial densities, momentum and pressure are volume-weighted, and the
temperature follows from the EOS. Two regions at the same pressure therefore
stay at that pressure through the transition, whatever their gases and
temperatures, and the transition adds no acoustic disturbance. Weighting the
temperature or the energy instead does not have this property when the two
gases differ in heat capacity ratio.

Under a [`ConstantBodyForce`](@ref), a column stays at rest only if its
pressure is in hydrostatic balance with the solver's own derivative operator.
A continuous hydrostatic profile is not: at rest the momentum right-hand side
equals the truncation error of the derivative, and acoustic waves appear from
the first step. [`Hydrostatic`](@ref) wraps an initial condition, keeps its
density, composition and velocity, and replaces its pressure with the discrete
balance, fixed by a reference pressure at one coordinate along the
acceleration:

```julia
initial = Hydrostatic(Layers(light, Slab(2, lo = 0.5) => heavy);
                      p_ref = 1e5, at = 1.0)
```

[`mass_fractions`](@ref) orders a composition given by species name, in mole
or mass fractions, and the `:X` output field reports mole fractions.

## Shocked states

[`shock_jump`](@ref) returns the state behind a normal shock of given Mach
number from the Rankine–Hugoniot relations, evaluated through the EOS, so a
[`Nasa9Mixture`](@ref) gets the jump of its temperature-dependent heat
capacities. [`shock_tube`](@ref) adds the driver pressure the shock requires
and the reflection from a closed end, and [`thermodynamic_state`](@ref)
reports density, temperature, sound speed and the other derived quantities of
any `Prim`. The post-shock state carries pressure and temperature, so it serves
directly as a region of a `Layers` initial condition and as the target of an
inflow condition:

```julia
incident = shock_jump(eos, air, 1.5)
bcs = ((NSCBCInflowBC(incident.post), SlipWallBC()), PeriodicBC(), PeriodicBC())
```

Placing the shock inside the initial domain avoids starting it at a boundary.
A boundary target that switches discontinuously in time takes effect in the
middle of a Runge–Kutta step, so its stages disagree about the boundary, and
the jump at the face has not yet been spread over any cells.

## Assemble the physical specification

```julia
problem = Problem(
    name = "example",
    eos = IdealMixture(["He", "CO2"]),
    transport = Transport(mu0 = 1e-5, Pr = 0.7, Sc = 0.7),
    metric = CartesianMetric(),
    sources = (),
    domain = ((0.0, 1.0), (0.0, 0.25), (0.0, 0.25)),
    bcs = ((SlipWallBC(), NSCBCOutflowBC(pinf = 101_325.0)),
           PeriodicBC(), PeriodicBC()),
    ic = initial,
)
```

Each `bcs` entry describes one coordinate direction. A bare condition such as
`PeriodicBC()` or `SlipWallBC()` applies to both faces; use `(low, high)` when
the faces differ. The same object is shared between symmetric faces, including
the switch state if it is a mutable `SwitchableBC`.

Coordinates and physical velocity components follow the selected metric. For
cylindrical coordinates they are ``(r,\theta,z)`` and ``(u_r, u_\theta, u_z)``;
for spherical coordinates they are ``(r,\theta,\phi)`` and the corresponding
orthonormal components.

## Select numerical choices

```julia
numerics = Numerics(
    n_global = (256, 1, 1),
    deriv = lele_d1_6(),
    filt = compact_filter(0.45),
    art = ArtParams(enabled = true),
    cfl = 0.5,
    control = StepControl(retries = 4),
    filter_interval = 1,
)

solver, Q = setup(problem, numerics)
```

A dimension with one point is collapsed and carries no derivative, halos, or
decomposition. With the default filter, every rank-local resolved extent must
contain at least nine points. See [Run in parallel](@ref) before setting
`dims` explicitly.

## Reinitialize an existing allocation

Call [`initialize!`](@ref) to replace the interior state without reconstructing
the solver and its operator plans:

```julia
initialize!(solver, Q, another_initial_condition)
```

This is useful for parameter studies that retain the same geometry, EOS,
resolution, and boundary types.
