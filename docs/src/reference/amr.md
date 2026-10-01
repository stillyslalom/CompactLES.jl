# Adaptive mesh refinement

```@meta
CurrentModule = CompactLES
```

[`AMR`](@ref) groups the refinement choices in `Numerics(amr=...)`. Refinement
uses a fixed ratio of three between levels on an unstretched Cartesian grid
or an axisymmetric (θ-collapsed) cylindrical one. A refined level may reach a
symmetry plane or the axis; its patch then starts half its own spacing from
it and folds there as the root does.
The coarse grid spans the full domain; fine patches replace its resolution
inside nested regions. Initial refinement and later movement are separate
decisions: `initial` chooses the first region, while `regrid_interval` controls
whether the solver retags after completed steps.

```julia
using CompactLES.Regions                                  # Sphere, Box

AMR(initial = :sensor)                                    # follow the features
AMR(initial = (x, y, z, t) -> abs(x - 0.5 - 0.2t) < 0.1)  # a prescribed path
AMR(initial = Sphere((0.5, 0.5, 0.5), 0.1))               # a fixed region
AMR(initial = [Box((0.2, 0, 0), (0.8, 1, 1)),             # fixed nested levels,
               Box((0.4, 0, 0), (0.6, 1, 1))])            # finest last
```

Every region is given in physical coordinates. A [`Shape`](@ref), or a vector
of nested shapes with the finest last, is covered by the nodes of each level's
parent, with no counting of nodes. A predicate `(x, y, z, t) -> Bool` sees the
node's coordinates and the solver time and is evaluated again at each regrid
check, so it can move the region on a prescribed path; `(x, y, z) -> Bool`
describes a fixed one. A predicate alone selects the region: the density
criterion defaults to off under a predicate, and giving `tag_threshold`
explicitly unites the two. `initial=:sensor` uses the enabled criteria on the
initialized coarse state. The selected nodes are buffered and covered by a
coarse-grid box or lattice tiles. If no node tags at setup, `setup` throws an
`ArgumentError`: give a shape for a uniform state.

A refined level stays at least `max(n_halo, 4)` of its parent's nodes inside
its parent. At the root's boundaries shapes and tags place a level on a face
carrying `SlipWallBC`, `NoSlipWallBC`, `NSCBCOutflowBC` or `NSCBCInflowBC`,
which the level then carries at its own spacing, so a feature at a wall or an
open face is refined up to it, and place a level across a periodic seam, so a
feature there is refined on both sides of it. They also place a level, at any
depth, on a `SymmetryPlaneBC` or the `AxisBC` of an r-z run. Any other face
keeps the band. A tiled level reaches a face only when the tile next to the
face's tile stays the margin inside the domain: the edge is at least
`max(n_halo, 4)`, and a partial last lattice cell at the high face spans at
least that many parent cells. Otherwise that face keeps the band. A
shape or a tagged feature reaching into the band of a face that keeps it is
refined only up to the band, with one warning. `level_boundaries = false`
keeps the band at every face.

`regrid_interval` defaults to zero, a fixed layout, for a shape, a region or a
predicate of position alone. For `:sensor` and a time-dependent predicate it
defaults to `tag_buffer / (2 cfl)` steps, the time a feature moving at the
CFL limit takes to cross half the buffer, so the feature cannot leave its
region between checks. `tile = 0`, the default, covers the tags with one box,
the cheaper cover of a single compact feature. A positive edge covers them with
lattice tiles so that separated features refine separately rather than as one
bounding box; each tile carries its own halo and transfer, so a small edge in
three dimensions costs more than the cells it saves. A run over many ranks
should give an edge: each tile's coupling to its parent runs on the few
ranks that hold the tile, while the interpolation over a single box does not
divide beyond one rank per conserved variable.
[`BlockRegion`](@ref)s remain available for an exact layout; a single region
may move when regridding is enabled, and a multi-level vector moves only with
a positive `tile`. A vector of shapes stays fixed.

| Keyword | Default | Meaning |
|:--|:--|:--|
| `initial` | `:sensor` | Sensor selection, a predicate `(x,y,z,t)->Bool` or `(x,y,z)->Bool`, a `Shape` or nested vector of shapes, or `BlockRegion`s |
| `regrid_interval` | `nothing` | Completed root steps between retagging; zero keeps the layout fixed; by default fixed for a shape or static predicate and `tag_buffer / (2 cfl)` for a followed feature |
| `tag_threshold` | `nothing` | Relative undivided fourth difference of mixture density; `0.02` by default, `Inf` (off) under a predicate |
| `tag_sensor_threshold` | `0` | Artificial diffusivity divided by acoustic cell diffusivity; zero disables |
| `tag_gradient_threshold` | `0` | Mass-fraction change per coarse cell; zero disables |
| `tag_vorticity_threshold` | `0` | Vorticity magnitude, in inverse-time units; zero disables |
| `tag_buffer` | `4` | Coarse nodes grown around marked nodes before boxing or tiling |
| `untag_ratio` | `2` | Hold threshold denominator for an existing tile; one removes hysteresis |
| `tile_lifetime` | `1` | Minimum number of regrid checks before a tile may be removed |
| `tile` | `0` | Zero makes one refined box; positive values at least three give a lattice tile edge in parent nodes |
| `level_boundaries` | `true` | Shapes and tags place a level on a wall or NSCBC face and across a periodic seam, and on a symmetry plane or the r-z axis; at `false` every level keeps the margin at every face |
| `rebalance` | `0` | Off at zero; otherwise minimum measured maximum/mean rank-busy-time ratio for tile repartitioning |
| `rebalance_persist` | `2` | Consecutive imbalanced checks required before repartitioning |
| `level_restriction` | `:inject` | Coincident fine-node injection; `:filter` anti-aliases before restriction and is serial-only |
| `level_interpolation_order` | `nothing` | Lagrange order, 2, 4, 6, 8 or 10, of the interpolation that fills fine ghost data and newly refined regions from the parent; `nothing` takes the derivative operator's interior order plus two, at most 10, under the default `PatchInterfaces(flux = :ghost)`, and the interior order under `:closure` |
| `subcycle` | `false` | At `true`, each fine level takes three steps per parent step with time-interpolated boundary data |

The density criterion is enabled by default except under a predicate. The
artificial-diffusivity criterion requires artificial transport to be enabled,
and `setup` rejects it otherwise. Sensor and gradient thresholds are dimensionless; the
vorticity threshold uses the run's units of inverse time. The fourth
difference detects unresolved density changes, while the gradient criterion
can target a mixing layer with little density contrast. Thresholds determine
*where* to spend grid points, not a new physical transport model.
The predicate must return `Bool` at every sampled node. The flat refinement
keywords that `Numerics` accepted before `AMR` are deprecated and cannot be
combined with `amr`.

After selecting the layout, setup evaluates the initial-condition function
directly at each fine node. During a run, a newly refined region is filled
from the current coarse solution and preserves fine data where regions
overlap.

The same interpolation fills each fine level's ghost data at every stage,
and `level_interpolation_order` sets its order. An interpolated value of
order p gives a first derivative of order p − 1 and a second derivative of
order p − 2. Under the default `PatchInterfaces(flux = :ghost)` the interface
divergence reads the interpolated values themselves, so the default order is
two above the derivative operator's, at most 10: 8 for [`lele_d1_6`](@ref)
and any scheme other than the built-in three, and 10 for [`lele_d1_8`](@ref)
and [`lele_d1_10`](@ref). Under `flux = :closure` the interface
divergence limits accuracy first, and the default matches the operator: 6, 8
and 10, and 6 for any other scheme. With C6 and the closure rows, an explicit
order 8 reduces the error in viscous, filtered or multidimensional runs and
with an interface `divergence` scheme, at no measurable cost in conservation or regrid
drift. For when to choose the closure rows, see
[Choose numerics](@ref). Orders above 2 are not
monotone: refilling a step narrower than one parent cell undershoots by 2.3%,
2.9% and 3.2% of the jump at orders 6, 8 and 10. Order 4 has no measured advantage.
`regrid_interval=0` keeps the initial layout fixed. A positive interval
regrids one refined level, or with `tile` every level up to `max_levels`,
each tagged on the level above it; without `tile` a vector of multiple
nested regions is static.
The solver retains all levels and restriction updates covered parent nodes.
Composite integrals and profiles avoid counting both parent and fine values
over the same physical region.

`BlockRegion(offset, extent)` uses zero-based offsets and node counts in its
parent level's lattice over the whole domain, not relative to the parent
patch; a nesting error prints the admissible offsets. A level's fine index corresponding to parent index `g` is
`3(g-1)+1`. The solver enforces a coarse-node nesting margin around each
fine region and a minimum fine patch extent. An explicit region may instead
reach a domain face carrying `SlipWallBC`, `NoSlipWallBC`, `NSCBCOutflowBC`,
`NSCBCInflowBC`, `SymmetryPlaneBC` or the `AxisBC` of an r-z run; its face
there then carries that condition at the fine spacing, and the margin applies
to its other faces. A level's node nearest a symmetry plane or the axis lies
half its spacing from it, outside the lattice of nodes coincident with the
root's, so a region below the first level reaching the low face starts at a
negative offset: its parent's first node, offset `-1` for the second level
and `-4` for the third (in general `(1 - 3^(ℓ-1))/2` for level ℓ), and a
region reaching the high face ends that many nodes past the parent's last
coincident node. An explicit region may also cross a periodic seam: its
offset may lie anywhere, its nodes past the last node of the period are the
first nodes again, and one box spans at most the period less the margin at
either end. Explicit regions provide
exact placement; sensor and predicate placement reaches those faces and
crosses a seam as well, and is clamped to the margin at every face under
`level_boundaries = false`.
Tiling uses a global lattice, so surviving tiles keep their locations as tags
move. A positive `regrid_interval` allows tiled regions to enter and leave;
`rebalance` may then move ownership among ranks after persistent measured
imbalance in a two-level hierarchy. A multi-level nested vector regrids only
with a positive `tile`.

Refinement uses interpolation to fill new fine nodes and restriction to
update covered parent nodes. Injection does not make arbitrary composite
integrals exactly conservative: monitor the hierarchy-aware mass and energy
integrals for the quantity of interest, especially as a front crosses a
coarse-fine interface or the layout changes. Checkpoint files record the
hierarchy and its refinement controls; resume with a compatible solver
configuration so the recorded layout can be restored. Subcycling can reduce
work on coarse levels relative to advancing every level at the fine-limited
step, but the refreshed fine-step stability rate still matters;
[`StepControl`](@ref) controls its failure policy.

```@docs
AMR
```

For transfer, subcycling, tiling, and the available tag thresholds, see
[Choose numerics](@ref) and
[Operators and decomposition](@ref).
