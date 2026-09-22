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
