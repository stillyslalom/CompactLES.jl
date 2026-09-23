# Adaptive mesh refinement

```@meta
CurrentModule = CompactLES
```

[`AMR`](@ref) groups the refinement choices in `Numerics(amr=...)`. Refinement
uses a fixed ratio of three between levels on a Cartesian, unstretched grid.
The coarse grid spans the full domain; fine patches replace its resolution
inside nested regions. Initial refinement and later movement are separate
decisions: `initial` chooses the first region, while `regrid_interval` controls
whether the solver retags after completed steps.

```julia
amr = AMR(initial=(x, y, z, t) -> abs(x - 0.5) < 0.1,
          regrid_interval=5, tag_buffer=4)
num = Numerics(n_global=(129, 1, 1), amr=amr)
```

The physical predicate sees the node's actual coordinates and current solver
time. It selects nodes independently of root grid indices, and the same
predicate runs at each regrid check. The predicate's tags are **united** with
the density and other enabled sensor tags. Set `tag_threshold=Inf` to select
only the physical predicate when the optional thresholds are all zero.
`initial=:sensor` uses the enabled criteria on the initialized coarse state.
The selected nodes are buffered and covered by a coarse-grid box or lattice
tiles. If no node tags at setup, `setup` throws an `ArgumentError`: choose a
region explicitly for a uniform state. A `BlockRegion` or vector of nested
regions can also be supplied as `initial`; a single region may move when
regridding is enabled, while a multi-level vector remains static.

| Keyword | Default | Meaning |
|:--|:--|:--|
| `initial` | `:sensor` | Initial sensor selection, physical `(x,y,z,t)->Bool`, `BlockRegion`, or nested vector of regions |
| `regrid_interval` | `0` | Completed root steps between retagging; zero keeps the selected layout fixed |
| `tag_threshold` | `0.02` | Relative undivided fourth difference of mixture density; `Inf` disables this criterion |
| `tag_sensor_threshold` | `0` | Artificial diffusivity divided by acoustic cell diffusivity; zero disables |
| `tag_gradient_threshold` | `0` | Mass-fraction change per coarse cell; zero disables |
| `tag_vorticity_threshold` | `0` | Vorticity magnitude, in inverse-time units; zero disables |
| `tag_buffer` | `4` | Coarse nodes grown around marked nodes before boxing or tiling |
| `untag_ratio` | `2` | Hold threshold denominator for an existing tile; one removes hysteresis |
| `tile_lifetime` | `1` | Minimum number of regrid checks before a tile may be removed |
| `tile` | `0` | Zero makes one refined box; positive values at least three give a lattice tile edge in parent nodes |
| `rebalance` | `0` | Off at zero; otherwise minimum measured maximum/mean rank-busy-time ratio for tile repartitioning |
| `rebalance_persist` | `2` | Consecutive imbalanced checks required before repartitioning |
| `level_restriction` | `:inject` | Coincident fine-node injection; `:filter` anti-aliases before restriction and is serial-only |
| `level_interpolation_order` | `6` | Lagrange order, 2, 4, 6 or 8, of the interpolation that fills fine ghost data and newly refined regions from the parent |
| `subcycle` | `false` | At `true`, each fine level takes three steps per parent step with time-interpolated boundary data |

The density criterion is enabled by default, including when `initial` is a
predicate. The artificial-diffusivity criterion requires artificial transport
to be enabled. Sensor and gradient thresholds are dimensionless; the
vorticity threshold uses the run's units of inverse time. The fourth
difference detects unresolved density changes, while the gradient criterion
can target a mixing layer with little density contrast. Thresholds determine
*where* to spend grid points, not a new physical transport model.
The predicate must return `Bool` at every sampled node. Keep all refinement
controls inside `AMR(...)` when using `Numerics(amr=...)`; the legacy flat
`Numerics` refinement keywords remain available for existing decks but cannot
be combined with it.

After selecting the layout, setup evaluates the initial-condition function
directly at each fine node. During a run, a newly refined region is filled
from the current coarse solution and preserves fine data where regions
overlap.

The same interpolation fills each fine level's ghost data at every stage,
and `level_interpolation_order` sets its order. An interpolated value of
order p gives a first derivative of order p − 1 and a second derivative of
order p − 2. The default 6 does not limit accuracy under the default
interface rows. Order 8 reduces the error in viscous, filtered or
multidimensional runs and with `interface_divergence`, at no measurable cost
in conservation or regrid drift. Orders above 2 are not monotone: refilling
a step narrower than one parent cell overshoots by up to about 3% of the
jump. Order 4 has no measured advantage. `regrid_interval=0` keeps the initial layout fixed. Positive intervals
support one refined level; a vector of multiple nested regions is static.
The solver retains all levels and restriction updates covered parent nodes.
Composite integrals and profiles avoid counting both parent and fine values
over the same physical region.

`BlockRegion(offset, extent)` uses zero-based offsets and node counts in its
parent level. A level's fine index corresponding to parent index `g` is
`3(g-1)+1`. The solver enforces a coarse-node nesting margin around each
fine region and a minimum fine patch extent. Explicit regions provide exact
placement; sensor and predicate placement is clamped to that legal interior.
Tiling uses a global lattice, so surviving tiles keep their locations as tags
move. A positive `regrid_interval` allows tiled regions to enter and leave;
`rebalance` may then move ownership among ranks after persistent measured
imbalance. Multi-level nested vectors cannot currently regrid.

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
[Choose numerics for accuracy per cost](@ref) and
[Operators and decomposition](@ref).
