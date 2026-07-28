# Curvilinear coordinates

CompactLES supports orthogonal Cartesian, cylindrical, and spherical metrics.
The numerical grid is uniform in computational coordinates; analytic metric
factors and optional one-dimensional stretch mappings convert derivatives and
fluxes to physical space.

## Coordinate conventions

| Metric | Coordinates | Physical velocity components |
|:--|:--|:--|
| `CartesianMetric` | `(x, y, z)` | `(u_x, u_y, u_z)` |
| `CylindricalMetric` | `(r, theta, z)` | `(u_r, u_theta, u_z)` |
| `SphericalMetric` | `(r, theta, phi)` | `(u_r, u_theta, u_phi)` |

Spherical `theta` is polar angle. Velocity components lie in the local
orthonormal basis and therefore have ordinary velocity units.

## Orthogonal metric factors

For scale factors `h_d`, the volume Jacobian and face-area factors are

```math
J=h_1h_2h_3,\qquad A_d=J/h_d.
```

Flux divergence is evaluated as `inv(J) * D_d(A_d F_d)`. Scalar gradients are
scaled by `1/h_d`; velocity gradients and momentum equations receive additional
rotating-basis curvature terms.

## Stretched meshes

A [`Stretch`](@ref) maps a uniform computational coordinate in `[0,1]` to a
physical coordinate. Its derivative multiplies the corresponding metric scale
factor. [`sine_cluster`](@ref) supplies an interior-clustering map:

```julia
stretch = (
    sine_cluster(0.0, 1.0, 0.4, 0.7),
    nothing,
    nothing,
)

Numerics(n_global = (256, 1, 1), stretch = stretch)
```

The amplitude must remain below one for monotonicity. A stretched dimension
must be nonperiodic. Excessive clustering reduces the smallest physical spacing
and hence the explicit timestep.

## Discrete geometric conservation

Analytic metric identities do not automatically cancel when each term is
discretized. In spherical coordinates, for example, differentiating a sampled
`sin(theta)` is not exactly the same as inserting an analytic `cos(theta)` in a
separate source term.

CompactLES constructs the spherical cotangent source with the same discrete
operator and area factor used by flux divergence. This discrete geometric
conservation law preserves a uniform state to roundoff. The test suite checks
freestream preservation in every metric and across MPI boundaries.

## Coordinate singularities

Nodes are half-offset from an axis, origin, or pole. Values beyond the
singularity are related to physical points on the other side by parity and, in
resolved angular cases, an antipodal mapping:

```math
(-r,\theta)\equiv(r,\theta+\pi)
```

for a cylindrical axis, with analogous spherical origin and pole relations.
Even/odd combinations reduce these mappings to folded compact solves.

- `AxisBC` handles a cylindrical axis.
- `OriginBC` handles a spherical origin.
- `PoleBC` is applied at both spherical polar boundaries.

Resolved antipodal dimensions impose parity and process-layout constraints;
setup validates them. The full spherical origin-plus-poles path is less mature
than Cartesian and collapsed-axis calculations and should receive dedicated
validation for each use.

## Timestep near a singularity

With resolved cylindrical angle, azimuthal spacing is `r * Delta theta`; near
the first radial node it can be much smaller than radial spacing. Spherical
azimuthal spacing is `r * sin(theta) * Delta phi`, becoming small near both the
origin and poles. Explicit acoustic rates scale with inverse spacing and
diffusive rates with inverse spacing squared.

When an angular dimension is collapsed, its derivative and small-cell
restriction disappear. Geometric source terms remain, and `compute_dt` adds a
curvature rate for collapsed swirl. Use `dt_report` to distinguish these cases.

## Scope of evidence

Axis and fold convergence is presently approximately third order because the
closure region controls the maximum error. The spherical origin also requires
smooth initial data resolved over several cells and does not support a singular
Noh initial condition at `t=0`. These are documented operating limits rather
than consequences of the continuum equations.
