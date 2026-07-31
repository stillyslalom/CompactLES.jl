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

Specify exactly one of `T_ion` and `rho`; the EOS determines the other. Mass
fractions must be in the same order as the EOS species and sum to one.
Initial-condition functions should be pure because setup may evaluate them
from multiple threads.

Use [`tanh_blend`](@ref) to resolve an interface over several cells:

```julia
blend = tanh_blend(x, interface_position, 2grid_spacing)
```

An exact jump is mathematically well-defined but is not resolved by a finite
grid. A transition over two or three cells avoids an unnecessarily severe
first-stage transient.

## Assemble the physical specification

```julia
problem = Problem(
    name = "example",
    eos = single_species(gamma = 1.4),
    transport = Transport(mu0 = 1 / 1600, Pr = 0.7, Sc = 0.7),
    metric = CartesianMetric(),
    sources = (),
    domain = ((0.0, 1.0), (0.0, 0.25), (0.0, 0.25)),
    bcs = ((SlipWallBC(), NSCBCOutflowBC(pinf = 1.0)),
           (PeriodicBC(), PeriodicBC()),
           (PeriodicBC(), PeriodicBC())),
    ic = initial,
)
```

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
