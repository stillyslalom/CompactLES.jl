# Curvilinear coordinates

CompactLES supports orthogonal Cartesian, cylindrical, and spherical metrics.
The numerical grid is uniform in computational coordinates; analytic metric
factors and optional one-dimensional stretch mappings convert derivatives and
fluxes to physical space.

## Coordinate conventions

| Metric | Coordinates | Physical velocity components |
|:--|:--|:--|
| `CartesianMetric` | ``(x, y, z)`` | ``(u_x, u_y, u_z)`` |
| `CylindricalMetric` | ``(r,\theta,z)`` | ``(u_r, u_\theta, u_z)`` |
| `SphericalMetric` | ``(r,\theta,\phi)`` | ``(u_r, u_\theta, u_\phi)`` |

Spherical ``\theta`` is polar angle. Velocity components lie in the local
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

## Freestream preservation

A uniform gas at rest must stay at rest, with no force from the metric factors
alone. Analytic metric identities do not automatically cancel when each term
is discretized: in spherical coordinates the discrete derivative of a sampled
`sin(theta)` is not exactly a sampled `cos(theta)`, so an area-weighted
pressure divergence and its curvature source would not cancel exactly.

CompactLES therefore writes the pressure term of the spherical radial and
polar momenta, and of the radial momentum on an axisymmetric ``(r, z)`` grid
or a stretched cylindrical radial grid, as a gradient,
``\partial p/\partial r`` or ``(1/r)\,\partial p/\partial\theta``, whose
discrete value on a constant is zero to roundoff at every node. In the other
momentum equations the face-area factor does not vary along the derivative,
except in the radial momentum of an unstretched cylindrical grid with a
resolved angle, where it is ``r``. A compact derivative of a linear function
is exact, so there the divergence and the ``p/r`` source cancel to roundoff.
On a stretched radial grid ``r`` is not linear in the computational
coordinate, the cancellation would leave a residual of the order of the
scheme's truncation error, and the gradient form is taken instead. The
curvature terms that remain in the momentum sources use the analytic
``\cot\theta/r``.

A collapsed dimension takes no derivative. For a collapsed ``r``, and for a
collapsed spherical ``\theta`` away from ``\pi/2``, the divergence of a flux
through it would still contribute a curvature term, ``s/r`` (``s = 1``
cylindrical, ``2`` spherical) or ``\cot\theta/r`` times the flux. The
pressure parts of that term and of the momentum source cancel exactly, and
both are left out, so a gas at rest stays at rest. The rest of the term is
not carried, so such a grid is consistent only while the flow through the
collapsed direction vanishes: a spherical radial line belongs at
``\theta = \pi/2``, where the term is zero, and a collapsed ``r`` is
consistent only at or near rest. The cylindrical ``(r, z)`` and
``(r, \theta)`` grids and the spherical ``(r, \theta)`` grid omit nothing.
The test suite checks freestream preservation in every metric, on these
collapsed and stretched grids, and across MPI boundaries.

## Coordinate singularities

[Supernova remnant](@ref) introduces the physical picture at a spherical origin:
an axis or an origin is a symmetry location, not a material wall. A smooth
scalar such as density has the same value on the two sides (even parity), while
a velocity component whose basis direction reverses changes sign (odd parity).

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

With resolved cylindrical angle, azimuthal spacing is ``r\Delta\theta``; near
the first radial node it can be much smaller than radial spacing. Spherical
azimuthal spacing is ``r\sin\theta\,\Delta\phi``, becoming small near both the
origin and poles. Explicit acoustic rates scale with inverse spacing and
diffusive rates with inverse spacing squared.

For a resolved cylindrical angle, `Numerics(polar_truncation = κ)` lifts this
restriction at the inner rings. Each ring of fixed ``r`` and ``z`` is projected
once per step onto its azimuthal Fourier modes
``m \le \max(2, \lfloor \pi r / (\kappa \Delta r) \rfloor)``, which are the
modes a field smooth through the axis can carry there. The radial and
azimuthal momenta keep one mode more on every ring, since a velocity
component of a smooth field carries mode ``m`` at order ``r^{m-1}`` rather
than ``r^m``. The timestep charges the azimuthal direction at the spacing of
the highest mode kept, that of the momenta, instead of ``r\Delta\theta``.
Modes 0 to 2 are always kept, so ring sums and a uniform freestream are
unchanged. The truncation is off by default, and the spherical form is not
implemented.

When an angular dimension is collapsed, its derivative and small-cell
restriction disappear. Geometric source terms remain, and `compute_dt` adds a
curvature rate for collapsed swirl. Use `dt_report` to distinguish these cases.

## Scope of evidence

Axis and fold convergence is presently approximately third order because the
closure region controls the maximum error. The spherical origin also requires
smooth initial data resolved over several cells and does not support a singular
Noh initial condition at `t=0`. These are documented operating limits of the
discretization, not consequences of the continuum equations.
