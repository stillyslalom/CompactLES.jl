# CompactLES — Patch AMR and the GPU backend

Part I describes the patch-based adaptive mesh refinement (AMR) and the
device backend as delivered: the constraints that bound the design, the
mechanisms, and the measurements behind them. Part II is the plan for the
production AMR the delivered capability is not yet: the structural
assumptions still in place, the design that replaces each, and the order of
the work with its gates. Prerequisite reading: `DESIGN.md` for the compact
operators, the distributed line solve, and the coordinate folds.

Every number here was produced by a bench script or testset and is usually
still guarded by one. Delivery history is in `HISTORY.md` and the git log.
Other documents cite the original plan's stage names; they map onto Part I
as Stage 1 → [Transfer operators](#transfer-operators), Stage 2 →
[Patches and same-level interfaces](#patches-and-same-level-interfaces),
Stages 3–4 → [Refinement](#refinement), G1–G4a → the subsections of
[The device backend](#the-device-backend), and sequencing items 1–3 →
[Refinement](#refinement) and
[Ownership and load balance](#ownership-and-load-balance).

## Contents

Part I: the delivered system

1. [Status](#status)
2. [Target problems](#target-problems)
3. [Constraints](#constraints)
4. [Transfer operators](#transfer-operators)
5. [Patches and same-level interfaces](#patches-and-same-level-interfaces)
6. [Refinement](#refinement)
7. [Ownership and load balance](#ownership-and-load-balance)
8. [I/O and restart](#io-and-restart)
9. [The device backend](#the-device-backend)
10. [Verification](#verification)
11. [Performance summary](#performance-summary)
12. [Lessons](#lessons)
13. [Scope boundaries today](#scope-boundaries-today)

Part II: the production AMR

14. [What remains](#what-remains)
15. [Design of the remaining work](#design-of-the-remaining-work)
16. [Open measurements](#open-measurements)
17. [Sequencing](#sequencing)

---

# Part I: the delivered system

## Status

One solver runs on four axes of configuration, combinable except where
[Scope boundaries today](#scope-boundaries-today) records an exclusion:

- **Same-level patches**: the domain tiles into conforming slab patches
  along one dimension, the rank set partitioned over them, coupled by ghost
  exchange, interface closure rows, and shared-plane averaging
  (`src/patches.jl`).
- **Nested refinement**: a chain of levels, each covering a region of its
  parent at ratio 3 with one patch or with the tiles of a global lattice,
  at a global timestep or Berger–Oliger subcycled recursively; a two-level
  hierarchy can move under sensor-driven tagging and regridding, tiles
  entering and leaving the set (`src/levels.jl`, `src/regrid.jl`).
- **Distribution**: each level decomposes over its own rank subset, a prefix
  of its parent's, and within it each tile over its own rank range, the
  ranges dealt out along a Morton curve by tile volume; a rank holds the
  root and the tiles of its range only. Ownership is stored across regrids,
  a survivor keeping its range; a tiled level repartitions on the per-rank
  busy time the run measures when that stays imbalanced past a threshold
  for a set number of checks, and a tile whose range moves migrates block
  to block. The level coupling gathers replicated region data and
  distributes its interpolation chains by conserved component.
- **Checkpoint and output**: a checkpoint carries the hierarchy (the tile
  layout and stored ownership of every level, the tag history) and every
  tile's state, in the shared HDF5 file or in per-rank files, and a restart
  rebuilds the level on whatever rank count it is given, continuing bit
  for bit on the writing one; field output is one piece per patch under a
  multiblock index with the covered coarse nodes blanked
  (`src/io_levels.jl`, `src/io.jl`).
- **Storage backend**: `CPUBackend` (`Array`s, `@threaded` loops) or
  `DeviceBackend(ka)` wrapping any KernelAbstractions backend. A
  device-resident solver (decomposed, refined, regridding, Float64 or
  Float32) reproduces the CPU solver bitwise over full runs.

The pointwise physics is written once as per-point bodies launched through
`pointwise!` (`src/pointwise.jl`); the compact line solves run through host
plans or their device mirrors (`src/lines_device.jl`) behind one
`apply_along!` entry point. There is no second code path for the device
beyond the launchers.

This is not yet a production AMR. [What remains](#what-remains) lists the
structural assumptions that separate it from one.

## Target problems

Two problem classes set the requirements. They pull in different directions
and the design has to serve both.

**An ICF implosion on a Cartesian grid.** The feature to resolve is a
spherical shell of radius r and thickness δ that shrinks over the run and
ends as a small hot spot. Fine volume for the three ways of covering it:

- one bounding box: (2r)³
- slabs along one axis, spanning the domain in the other two: never smaller
  than the box
- tiles of edge a intersecting the shell: ≈ 4πr²(δ + 2a)

At δ/r = 0.1 and a ≪ r the tiles refine about a third of what the box does,
and the ratio improves as the shell thins. Slab tiling buys nothing over the
one box that exists, so this problem requires full 3-D tile adjacency. It also
requires the layout to change topology, not just position: a shell of tiles
at level 1 during convergence, then one small box (with a level-2 nest
inside it) once the shell is gone. Cartesian avoids the folds, so the
fold-adjacent refinement prohibition (constraint 4) and the symmetry-cell
CFL squeeze of `CALIBRATION.md` do not apply; the price is resolving the
shell isotropically, which is what the tiles pay for.

**A shock tube with a mixing layer.** Only the mixing region, a slab or a
thin distorted sheet, needs resolution; the rest of the domain carries
smooth waves and a coarse level suffices. One or a few boxes serve, the
refined fraction is small, and the run is long, so what binds is the
conservation drift across level boundaries, the artificial-property sensor
behavior where a shock crosses a coarse–fine boundary, and the cost of the
level coupling per step rather than the tile geometry. The delivered
two-level solver is closest to this class; the measured mixing case under
[Measured costs](#measured-costs) is of this shape.

Both classes share the general demands: several levels, restart, diagnostics
that exclude covered coarse nodes, and a rank assignment that does not put
every rank on every level.

## Constraints

Each of these is structural or measured. They bound the delivered design and
still bind its extensions. Other files cite them by number; keep the
numbering.

1. **No line cut mid-solve.** A compact operator couples an entire grid line
   through a banded LHS. A patch is therefore a logically rectangular block
   of uniform resolution; the compact solve runs over the patch with ghost
   layers filled by same-level copy or coarse-level interpolation. Oct-tree
   cell-by-cell refinement is excluded (Non-goals in `ROADMAP.md`).
2. **Minimum extent 9 per resolved dimension.** The C8 filter closure is the
   binding scheme (`plan_direction` enforces it). This bounds the minimum
   patch size, the tagging granularity, and the rank count a small region
   can absorb (a 1-D fine patch needs ≥ 9·np nodes along its only active
   dimension; `_amr_dims` errors with the numbers before a regrid can die
   mid-run). Compact-scheme AMR is therefore coarse-grained by nature: a
   tile is at least 3 coarse nodes per side and realistically 6, so that
   the closure rows do not dominate it. This is permanent and should be
   expected, not worked around.
3. **Refinement ratio 3.** The transfer filter has width 3Δx, and the odd
   ratio keeps a coarse node coincident with the middle node of each fine
   triple, so both transfer operators stay centered, symmetric, and
   invertible on a node-centered grid. Do not attempt ratio 2.
4. **Folds require uniform structure in their paired dimension** (partner at
   +P/2 for P ranks along that dimension, reflected partner for the reversed
   one). Refinement across a fold's singular region or its antipodal partner
   is forbidden and rejected at setup. The singular region keeps uniform
   resolution.
5. **The bit-exact oracle survives only within a patch.** The
   spike/reduced-interface solve reproduces the single-domain answer at any
   rank count *within* one patch; patch interfaces are approximate by
   construction. Every interface therefore carries a replacement oracle: a
   manufactured smooth solution across it with a measured order, in the
   style of `test/convergence.jl`.
6. **The grid stays node-centered**, half-offset only where a fold requires
   it. The SBP–SAT multiblock literature is node-coincident too, so a
   cell-centered migration was never a prerequisite.
7. **Error localization is favorable.** The inverse of a compact LHS decays
   geometrically off the diagonal (≈ α^|i−j|, α = 1/3 for C6), so pollution
   injected at an inexact interface decays about 3× per point into the
   patch. Measured in situ at ≈ 3.4× per point (shock round-trip pollution,
   `bench/amr_transfer.jl`); the default 4-coarse-cell tagging buffer drops
   interface pollution two orders of magnitude and is sized from this
   number. The rate is the root of the LHS symbol, 0.382 per point for C6,
   0.451 for C8 and 0.556 for C10 (1.8× per point), and the derivative's
   response to a unit error in the first ghost layer, through the interface
   closure rows and the interior solve, decays at exactly those rates
   (`closure_localization`, same bench). At C10 that response is 5e-3 of
   the ghost error eight fine points in and 5e-4 at twelve, the default
   buffer's width, so the buffer holds two orders at C10 where it holds
   five at C6; the level buffers were remeasured at C10 with the same
   outcome ([Banded schemes at interfaces](#banded-schemes-at-interfaces)).

## Transfer operators

`src/transfer.jl`; measurements in `bench/amr_transfer.jl`, guarded by
serial testsets and a multi-rank section.

**Source.** Miranda's level transfer (`pyranda/parcop/stencils.f90`,
`cfamrcf`/`cfamrfc`, inside an `#if 0` block) is an invertible compact
filter pair whose transfer function is a Gaussian of width 3Δx:

```
alpha*fbar(i-1) + fbar(i) + alpha*fbar(i+1) = c*f(i-2) + b*f(i-1) + a*f(i) + b*f(i+1) + c*f(i+2)
alpha = -0.0321826755129339
    a =  0.4451523642186118
    b =  0.2207614172195584
    c =  0.0244797251582018
```

Restriction applies the filter; prolongation deconvolves it, using the same
coefficients with LHS and RHS roles swapped. Conservation comes from unit DC
gain (a + 2b + 2c = 1 + 2α to the last bit), not refluxing, and the
tabulated boundary closures preserve that gain. The extended-data closure
variant is tabulated identically to the one-sided one, "to maintain
invertibility" per the source comments, which set the precedent for every
interface closure in this code: the LHS couples no ghost unknown even where
ghost data is available.

**Mapping onto this code.** Restriction is a `CompactScheme` (tridiagonal
LHS, Gaussian RHS); prolongation a `BandedCompactScheme` with q = 2,
interior rows normalized to a unit diagonal. Both bind to a dimension with
the existing `plan_direction`, which supplies the distributed spike solve
and the fold parity variants without further code (measured ≤ 1e-13 against
parity-extended full lines, both schemes, both signs).

**The 3:1 sampling convention**, pinned numerically because the public
kernels do not fully specify it: restriction filters the fine line and takes
the coincident nodes (fine node 3m − 2 ↔ coarse node m); prolongation
injects coarse values onto coincident fine nodes, fills the intermediate
nodes by Lagrange interpolation (order 4/6/8, default 6), then deconvolves.
Measured: the pair round-trips at 1.6e-15 closed / 2.7e-15 periodic;
restriction is a left inverse of prolongation (coarse → fine → coarse exact
at 8.9e-16 for arbitrary data), while fine → coarse → fine converges at the
interpolation order (3.97 / 5.93 / 7.97); a constant survives restriction to
the last bit and prolongation to 13 ULPs; within 6 points of a closed end
the round-trip converges at ≈ 3 (the closure order).

**Conditioning.** Prolongation is deconvolution of a Gaussian (gain 20.24
at the fine Nyquist, closure condition number 33), benign while the coarse
field carries no content above its Nyquist. Measured in situ: the smoothed
δ⁴ sensor of a 2h shock round-trips at 1.03–1.13, the state undershoots
≤ 3% of ambient at the shock, and pollution decays ≈ 3.4× per point outside
it. An explicit Gaussian pass ahead of restriction (Miranda's `c4ff3` role)
cuts the undershoot 2.8× but raises total round-trip error; it is the tool
if regridding onto captured shocks proves positivity-limited, not a default.

**SBP–SAT is the fallback of last resort** and the theory guiding interface
placement (Carpenter, Gottlieb & Abarbanel 1993; Mattsson & Rydin 2022 on
implicit SBP; Nissen et al. 2015 on block-adaptive grids; Almquist & Dunham
2018 on non-conforming interfaces). Its costs (interface-order loss, penalty
terms above 4th order, a rewrite of the closure cascade and every
convergence guard) have not been justified by any measurement.

## Patches and same-level interfaces

`src/patches.jl`. A `Patch` carries a block's place in the global grid
(`region`), its communicator and `Decomp`, its operator plans and folds, and
every persistent field array, typed `A <: AbstractArray{T,3}` by the storage
backend; the scratch of one right-hand-side evaluation sits on an
`RHSWorkspace` the rank's patches share (see
[Tiles and adjacency](#tiles-and-adjacency)).
`Solver` holds the physics configuration plus this rank's patches; a
single-patch solver forwards patch-owned property names to its sole patch,
so the single-patch code path is the old code path. Routines between the
step drivers and the arrays take a `SolverLike`, so one body serves both.

Rank assignment partitions the rank set over patches proportionally to
volume (one `MPI.Comm_split`); a serial run advances every patch in
sequence. Coupling is three mechanisms between RK stages
(`sync_patches!`): shared-plane averaging, ghost refill, then each patch's
own halo exchange. Because a rank advances several patches sequentially, no
cross-patch communication may sit inside `compute_rhs!`.

The interface closure rows follow the Miranda precedent: gradient and filter
plans take `interface_closures` rows whose LHS couples no ghost unknown
while the RHS reads exchanged ghost layers (`interface_rhs = :extended`, the
default) or stays one-sided (`:onesided`). The flux divergence always keeps
the scheme's own one-sided closures (`Patch.div_plans`): ghost *fluxes*
would require a second communication phase inside the RHS, which the
sequential-patch order forbids. `gcl_cotr!` routes through the same
divergence plans so the discrete GCL cancellation is preserved. A banded
scheme takes `q` such rows per end
([Banded schemes at interfaces](#banded-schemes-at-interfaces)).

Measured at the two-conforming-patch gates: entropy-wave order 3.1–3.5
across the interface (the divergence's one-sided rows binding); acoustic
pulse reflected amplitude 2.3e-3 of incident at 192 points, converging at
≈ 5th order; conservation drift 1.2e-8 relative over a long periodic run
against 4.5e-15 single-patch. Rank partitioning reproduces the serial
two-patch answer bitwise at one rank per patch; once a patch itself
decomposes, agreement is round-off (3.1e-15 at np = 4 against a 9.5e-8
signal) with identical step counts. The inviscid gates run the divergence
alone, whose plans keep the one-sided rows, so they cannot tell
`:extended` from `:onesided`; the viscous-wave gate reaches the interface
rows through the gradients, and there the extended-data rows converge at
4.3 / 4.0 against the single-patch answer at the same N (6.1e-5, 3.0e-6,
1.9e-7 at 48 / 96 / 192 points) where the one-sided rows give 2.1 / 1.7
(8.4e-5, 2.0e-5, 5.9e-6).

### Banded schemes at interfaces

The pentadiagonal C10 (`lele_d1_10`) closes a patch interface with two
rows per end, one per left-hand-side band that would couple a ghost
unknown (`interface_closures(::BandedCompactScheme)`,
`src/kernels_banded.jl`). Each is an identity row whose right-hand side is
the explicit central difference of order 2(M + q) = 10, the interior formal
order, centered on its own node, so row 1 reads M + q = 5 ghost layers and
a patched or refined C10 run takes `n_halo ≥ 5`; the constructor refuses
the default four with the width to use. Rows 3 onward keep the interior
stencil, reading the exchanged ghosts. `plan_direction` for a `BandPlan`
takes the `lo_closures` / `hi_closures` hook of the tridiagonal plan with
the same reach checks, and the device mirror carries each row's first-point
offset as it does for a `DirPlan`, so the device line solve stays bitwise
against the host under decomposition. The flux divergence keeps the
scheme's own closure cascade (the C6 rows) as at every interface, and the
filter stays C8, so the minimum extent does not move. The `:d8` ring
detector, the other banded scheme, is the sensor's operator and keeps its
closed-edge rows at an interface as the sensor smoother does; patched runs
still reject it pending the sensor gate of item 8.

Measured at the gates: the two-patch entropy wave at C10 converges at
3.06 / 3.52 (8.2e-7, 9.8e-8, 8.5e-9 at 48 / 96 / 192; C6 3.09 / 3.53),
the divergence's one-sided rows binding as before; the pulse reflection is
4.1e-3 of the incident amplitude at 192 (7.5e-2 at 96, 7.7e-5 at 384),
above C6's 2.3e-3 by the larger mismatch between the interior rows and the
closure cascade; the viscous wave converges through the interface rows at
3.95 / 3.85 against the single-patch answer (5.2e-5, 3.3e-6, 2.3e-7)
where the one-sided rows give 2.00 / 1.67; and a degree-9 polynomial
differentiates to 3e-12 through both ends of a closed line whose ghosts
carry it, where the scheme's own cascade leaves 1e-4. Across the
coarse-fine boundary the two-level wave measures 3.84 / 3.54 (8.2e-8,
5.7e-9, 4.9e-10; subcycled 3.88 / 3.53), the Sod crossing leaves 3.5e-10
of momentum noise ahead of the shock (subcycled 4.1e-10; C6 6.4e-10 and
1.3e-10), and the tiled regrid tracks the shock with positivity intact.
Rank-partitioned, the viscous two-patch C10 run reproduces the serial
answer bitwise at one rank per patch and to round-off once a patch
decomposes, as C6 does.

The level buffers were remeasured at C10 with the two-level entropy wave
and the subcycled Sod crossing over `RESTRICT_MARGIN` 0–3 and
`LEVEL_BUFFER` 4 and 6, in a scratch copy of the tree with the constants
edited, the two testsets being the instrument. C10 tracks C6 at every
setting: the wave error at 96 nodes is 5.1e-9 / 5.1e-9 / 5.7e-9 / 5.4e-9
over margins 0–3 against C6's 5.0e-9 / 5.0e-9 / 7.7e-9 / 5.8e-9, the
growth of that error from t = 0.5 to t = 2 is 2.2 / 2.2 / 3.3 / 1.9
against 1.7 / 1.7 / 2.5 / 1.6, the Sod noise stays between 3.5e-10 and
8.8e-10, and `LEVEL_BUFFER = 6` reproduces the buffer-4 numbers to the
last digit at both schemes. Nothing at C10 calls for wider buffers, and the
constants stay where they are. The amplifying loop `RESTRICT_MARGIN` was
introduced against did not appear at margin 0 in this configuration for
either scheme, which is an observation about the current coupling, not a
reason to move a C6 constant.

## Refinement

`src/levels.jl`, `src/regrid.jl`. The hierarchy is a chain of levels, root
first in `solver.levels`, each level a set of patches covering regions of
its parent at ratio 3, node-centered (m parent nodes ↔ 3m − 2 fine nodes).
A `Level` holds its depth, the indices into `solver.patches` of the patches
this rank holds, one `LevelTransfer` per patch coupling it to its parents
(the buffered box, the shell ring, the Hermite storage `box_Q0` .. `box_dQ1`,
and the parent patches under the box), and the level's same-level interface
records. The flat `solver.patches` stays the drivers' iteration space, so
the property forwarding that serves a single-patch `Solver` is unchanged,
and the shared `max_rate` reduction supplies the timestep. A refined region
is specified in its parent's node space (`refine` as one `BlockRegion` or a
vector of them for a nested chain), so a level-ℓ patch's region offset is
`3 · (parent offset + region offset)`, and nesting requires
`max(n_halo, LEVEL_BUFFER)` parent nodes of margin at every depth. A level
is one patch over its region or, with `tile > 0`, the tiles of a global
lattice ([Tiles and adjacency](#tiles-and-adjacency)); every mechanism
below applies between a patch and its parents, and the measurements quoted
are two-level, one-patch ones unless stated.

### The coupling

**Interpolation and injection, not the invertible pair.** The pair's
contract is that prolongation input is samples of the *filtered* field,
which restricted data is and the live coarse solution is not. Deconvolving
point samples sharpens data that was never smoothed; filtering on the way
down writes an attenuated representation into a field of point samples;
both are O(h²) against the solution, and the manufactured-solution gate
measured order 1.3–1.7 through the pair. The default coupling is therefore
order-6 Lagrange interpolation up (`interpolate!`) and coincident-node
injection down (a sampled `gather_region!`), which measures order 3.46/3.64
with errors three decades lower. `level_restriction = :filter` keeps the
anti-aliasing pair selectable, and regridding initializes *new* fine cells
by interpolation for the same reason.

**Schedule.** After every RK stage, the fine patch's ghost ring *and*
boundary-plane nodes are overwritten from the coarse state interpolated
over a box extending `LEVEL_BUFFER = 4` coarse nodes beyond the region.
This is Dirichlet forcing by the coarse solution, and it removes the
inter-level drift mode without an averaging step. After every completed
step the fine state writes back onto the covered coarse region, holding
`RESTRICT_MARGIN = 2` coarse nodes off the boundary. The margin is not
optional: restricting all the way to the boundary closes an amplifying loop
through the fine solution's least-accurate nodes (its imposed shell) into
the closure rows that feed the next shell (measured gain ≈ 2 per step, flat
with the margin in place).

**Distribution of the coupling.** The coupling runs on a replicated-data,
distributed-work split. Data replicates: `gather_region!` assembles a node
region of a distributed field on every rank with one Allgatherv (the
buffered coarse box per shell imposition, the coincident nodes for `:inject`
restriction, the surviving fine state at a box regrid). Work distributes: a
subcycled step imposes the shell ~20 times, each a K-stage tensor-product
interpolation per component, and replicating that per rank put the 3-D
cost case at 85% of the uniform-fine wall; under `_impose_shell!` rank r
runs the chain only for components c ≡ r (mod np) and shares the thin shell
ring through one Allgatherv, which moves the same values instead of
recomputing them (serial results bit-identical) and brought the composite
to 49%. Every rank writes only the nodes it owns; tagging reduces its
bounds globally; a fine patch picks its process grid through `_amr_dims`
over the rank range it is assigned
([Ownership and load balance](#ownership-and-load-balance)). A patch's
buffered box may cross the boundaries of several parent patches once a
level holds more than one; the gathers and the restriction write-back run
per parent, which the replicated gather over the parent level handles by
construction.

### Subcycling

`subcycle = true` is Berger–Oliger: one coarse step with the fine level
frozen, then three fine steps of dt/3, the fine shell imposed at every fine
stage time from a cubic Hermite reconstruction of the coarse trajectory on
the buffered box (values and RHS at both step ends; O(dt⁴), matching
LSRK54, which has no free dense output). The t^{n+1} endpoint RHS costs one
extra coarse RHS per step, since the next step's stage-1 RHS arrives too
late and restriction invalidates a cached one. Each patch's rate is divided
by 3^level, so the step is coarse-limited. Each level filters its own state
at its own step cadence, which is dt-consistent whenever `filter_cfl > 0`.
Measured: the entropy-wave orders are unchanged by subcycling at a third of
the steps, and the subcycled Sod gate improves on the global-dt one
(ahead-of-shock noise 5.7e-11 vs 6.4e-10; mass drift 9.8e-5 vs 1.36e-4).

The step driver is recursive (`_advance_level!`): one step of level ℓ, the
extra RHS that saves the Hermite endpoint for its children, three substeps
of level ℓ+1 at dt/3, then restriction of ℓ+1 onto ℓ for every ℓ above the
root (the root's restriction stays in `run!`, after its filter pass). Each
level below the root filters itself at its own cadence, substep index
`3 · parent index + m`. The extra parent RHS recurs on every level that has
children, about 1/15 of the RHS work at similarly weighted levels, not a
fraction that compounds with depth. A three-level static nest on the
entropy wave converges at the two-level orders in both stepping modes, and
the three-level Sod gate runs at cfl 0.4 without a rate check per substep
(see [Ownership refinements and the rate check](#ownership-refinements-and-the-rate-check)
for what is not built).

### Tiles and adjacency

The fine level of an implosion is a shell of tiles, so a level can be a
set of fixed-size tiles on a lattice, not one box and not slabs. `tile` is
the lattice edge in parent nodes (0, the default, keeps one patch per
level; the floor is 3 from constraint 2); lattice cell k spans parent nodes
k·tile + 1 .. (k+1)·tile + 1, so abutting tiles share their interface plane
as root slabs do, and every tile face is either fully shared with one
neighbor or fully parent-fed. A static `refine` box expands to the lattice
cells that reach past its planes; cells at the domain edge are clipped to
the nesting margin and dropped below four nodes. The lattice is global so
that a regrid never changes a surviving tile's region.

Fixed tiles were chosen over Berger–Rigoutsos clustering for three reasons:
equal extents let the device batch line solves across a level's tiles
([Device](#device)); the lattice makes the face-closure decision binary;
and the measured waste on the target problems is the only thing that would
justify the clustering algorithm.

Same-level coupling reuses the root machinery: `build_interface_records`
over the level's tiles (each rank contributes a block of each tile it holds
to one Allgatherv, and the tag bound is checked against `MPI.tag_ub`),
records held on the `Level`, addressed in the level's own communicator,
point-to-point, and run by `sync_patches!` and, inside the subcycled
driver, after every stage of a level. Each face's treatment follows the
tile's neighbors: the shell imposition writes only parent-fed faces
(`LevelTransfer.imposed`, applied by `_in_shell`), the restriction margin
stands off parent-fed faces only, and the closure rows are the same
interface rows on every face, since both kinds read ghosts. Ordering on
every stage: same-level records after the update, then the parent
imposition at the head of the next stage, so a same-level neighbor's data
wins wherever both exist; and after a level receives its children's
restriction, its records run again before it is read (the restricted nodes
can sit beside a tile interface).

The records are dimension-phased (`_sync_level_records!`): the records of
dimension d span the transverse dimensions before d over their padded
ranges, with the per-patch halo exchange between phases. Two things depend
on that. A node shared by four tiles (eight in 3-D) reaches the mean of all
its copies only that way, since one flat pairwise pass reads values an
earlier pair has changed (a probe with copies 1, 2, 3, 4 ended at 2.23,
2.68, 2.41, 2.68); and the edge and corner ghosts of an interior tile,
which no shell writes and no face strip covers, are reached by the later
phases' strips through the earlier phases' ghosts, the argument `halo.jl`
makes for rank halos. A tile's buffered box may span several parent
patches (`coarse_indices`); the gathers and the restriction write-back run
per parent, and a child's buffered box must lie in the parents' own nodes,
their parent-fed planes eroded (imposed data is the class `RESTRICT_MARGIN`
exists to keep out of the closure rows). After a regrid a fresh tile takes
the planes it shares with surviving tiles from the survivors one-way
(`_seed_planes!`), and a tile about to leave restricts once more first.
Corner-coupled adjacency at level 0 does not arise; the root level stays a
slab layout or a single patch.

**Shared RHS scratch.** A rank advances its tiles in sequence and nothing
in the right-hand side's scratch outlives the evaluation that filled it, so
the gradients, the sensor fields and their scratch, and the assembled fluxes
live on an `RHSWorkspace` (`src/patches.jl`) shared by every patch of equal
padded extent; the conserved state, the artificial coefficients, the
primitives and the geometry remain per tile, each for a reason recorded on
that type (`mu_art`/`beta_art`/`kappa_art`/`D_art` survive into the next
`max_rate`, the primitives must be current on every tile for the prepared
first stage, `du` persists across stages, and geometry is persistent).
Sharing is keyed on the padded extent rather than served as views of one
largest allocation, so every array a patch holds is the one it would have
allocated alone, with its own storage type and strides, and the host and
device routing is untouched; a lattice level's tiles carry equal extents by
construction, so the key costs nothing. Two sensor fields are pooled
despite a reader outside the RHS: `scalar_field(solver, :strain_mag)` and
`:sensor` take a `Solver` and read through the single-patch property
forwarding, so they are reachable only where the pool is one patch's own
set, and a multi-patch output path will have to take storage of its own.
The sharing is guarded directly in `test/level_tests.jl`, since no other
figure in the gate moves if the pooling silently stops.

Measured: a tiled level costs nothing visible against the one-patch level
(1-D entropy wave, tile 8: 6.0e-10 against 6.2e-10 at N = 192, orders
3.95/3.91; a 2×2 tile nest in 2-D, corner included, 4.29e-8 against
4.27e-8), decomposed runs reproduce serial to round-off, and the tiled
regrid tracks the Sod shock through lattice cells that stay contiguous. On
an annular tag set in 2-D (`bench/amr_tiles.jl`, N = 192) the cover is 41%
of the bounding box at tile 6 and 47% at tile 12.

### Tagging and regridding

`regrid_interval = K` retags the parent level every K steps at the head of
the step loop. The tag is a union of criteria, each a per-point body over
the parent level's state run through `pointwise!` on the host
(`_tag_sweep!`, `src/regrid.jl`), writing into a host scratch on the
`RegridSpec`:

- the relative undivided fourth difference of the mixture density,
  Σ_d |δ⁴_d ρ|/ρ > `tag_threshold`, read from Q directly so that tagging
  needs no primitives pass; always on;
- the artificial diffusivity number
  ((μ\* + β\*)/ρ + κ\*/(ρ c_p) + max_k D\*_k)/(c h) > `tag_sensor_threshold`,
  the scheme's own statement that a feature is under-resolved. It is read
  from the per-patch coefficient arrays, which are C·ρ·max(sensor, 0) of
  the evaluation that wrote them and outlive it, never from the pooled
  workspace's `sensor`, which at the head of the step holds whichever
  patch of that padded extent evaluated last. A captured Sod shock reads
  about 2 under the default C_β;
- the mass-fraction change per cell, max_k |δY_k| over the centered
  difference of one cell, > `tag_gradient_threshold`, for mixing layers;
- the vorticity magnitude from centered differences > `tag_vorticity_threshold`;
- a user closure `tag_predicate(patch, I) -> Bool` over the parent
  `PatchSolver` and a padded index, run serially on the host.

Every criterion but the first is off by default. The tagged set is buffered
by `tag_buffer` (default 4, the pollution-decay figure of constraint 7) and
clamped to the nesting margin, and its bounds (or its lattice flags) are
reduced globally so every rank derives the same region; a rank-local
decision here would be a deadlock, since the ranks would then split
different communicators.

Derefinement carries hysteresis. A body writes one of two levels: the node
tags above its criterion's threshold and holds above the threshold over
`untag_ratio` (default 2). A held node keeps an existing tile and calls
for no new one, so a tile at the edge of a feature does not flicker as the
feature crosses the threshold; `untag_ratio = 1` reproduces the rule
without the band bit for bit. For the box, held nodes inside the current
box extend a box that some tagged node is rebuilding to reach them, and a
signal with no tagged node anywhere keeps the box whole, so the band
never shrinks the box onto its held nodes. A tile is not dropped
before `tile_lifetime` regrid checks (default 1) since its creation. The
history the two rules need is on the `RegridSpec`: `checks` counts the
regrid checks and `created` records, per current tile region, the check it
was created at. Both are derived from the reduced flags, so every rank
holds the same record, and both are state the checkpoint carries
([I/O and restart](#io-and-restart)). On
the Sod cases the band holds the marginal tile ahead of the shock one check
longer and stops its re-creation, and it keeps the rarefaction head's tiles,
a marginal feature that the threshold alone admits and drops.

The retry savepoint refreshes at a regrid, since restoring across a layout
change is not meaningful. Regridding is two-level: the root and one refined
level.

**The box regrid** (`tile = 0`) rebuilds the level's one patch over the
bounding box of the tagged set when that box moved. The new patch and
`LevelTransfer` are built from the schemes retained on the `RegridSpec`,
the new state is initialized by interpolation of the parent, and surviving
fine data is copied across the region overlap on the coincident lattice,
one node off both patches' boundary planes (the old plane was imposed data;
the new one is re-imposed by the next shell fill). The surviving state is
gathered replicated over the old region on every rank while the old
decomposition is live (`_gather_tile`), because the new region differs in
offset as well as in rank set and the overlap window is shifted
(`_carry_over!`). When no cell tags, or the box did not move, the current
region is kept: a feature fading below threshold leaves refinement where it
last was instead of collapsing it.

**The tiled regrid** is a set difference over the lattice (`_tag_tiles`,
one flag per lattice cell reduced over the communicator, then
`_regrid_tiles!`): a cell is wanted when any buffered node at the tag level
meets it, and an existing cell also when a held node meets it or its tile
is younger than the lifetime;
a wanted tile that exists keeps its region, and when its owner range
survives too it keeps its arrays and its state, taking only a new id,
faces and transfer; a newly wanted tile is built and initialized by
interpolation of the parent; an unwanted one is dropped, its last
restriction already on the parent. Between distinct lattice cells no
carry-over arises: they overlap in a shared plane at most, and that plane
takes the neighbor's values at the first averaging after the regrid, a
one-node perturbation of interpolation order. A refined level is never
empty: as for the box, a check at which no cell reaches the hold level
keeps the tile set as it is, whatever the lifetimes say, so a feature that
fades entirely leaves its last tiles in place rather than emptying the
level, which the hierarchy does not represent. A surviving tile whose owner
range moved is rebuilt on its new owners and takes its evolved interior back
by point-to-point migration
([Ownership and load balance](#ownership-and-load-balance)).

Measured on moving-region gates: Sod at N = 201 coarse against a 601-node
uniform-fine reference, composite density error 2.8e-3 where uniform-coarse
gives 7.3e-2 (26×), with fine resolution over a third of the domain;
Shu–Osher 10× better in L∞ over the wave train, 6.7× in L1, at 2497 coarse
steps against the reference's 4662. The tiled regrid reproduces the Sod
moving-region gate.

### Diagnostics on the composite grid

Every reduction that reports a physical quantity over a refined run must
exclude the coarse nodes a child level covers, and exclude them exactly
once; a double-counted covered node is the single most likely source of a
silently wrong number in a multi-level run. The data model is a per-patch
mask, `Patch.covered`: one host byte per node of the padded extent, each
bit the orthant of the node's quadrature cell (signs s_d along the three
dimensions, bit (s1 > 0) + 2(s2 > 0) + 4(s3 > 0)) that a child tile covers.
With node-centered quadrature a tile spanning parent nodes lo .. hi covers
the + side of a node for lo ≤ g ≤ hi − 1 and the − side for lo + 1 ≤ g ≤ hi,
so a node on a child's face keeps half its cell, an edge node three
quarters, a corner seven eighths, and a node shared by two abutting tiles
is covered by both halves; a union of lattice tiles is the union of their
bits, which is why the mask is per orthant and not a per-dimension factor.
`_fill_covered!` writes it at setup and at every regrid from the child
regions, which every rank of the parent's level holds (`Level.transfers`),
so it follows the tiles a rank holds and is readable on the parent's ranks
alone, where the covered nodes lie; it is rank-local and runs no
collective.

The diagnostics route through it in their `Vector` forms, one full padded
array (or one state) per held patch aligned with `solver.patches`:
`volume_integral`, `domain_volume`, `volume_average`, `plane_profile`,
`species_pdf`, `mix_width`, `molecular_mixing`, `tke_profile`,
`turbulent_kinetic_energy` and `dissipation_rate`. A coarse node's weight
is its quadrature weight times the fraction of its cell no child covers
(`uncovered_fraction`); a plane average uses the in-plane fraction
(`uncovered_plane_fraction`, the orthant pairs across the profile
direction), samples the root's stations only, and takes each finer patch's
coincident planes with that patch's own transverse cell measure, a plane
two abutting tiles share counting half from each. The fine patch's own
quadrature gives its boundary planes half weights, so the two sides of a
coarse-fine face sum to one, the same rule the same-level interface plane
follows. Every composite form accumulates the rank's patches and reduces
once over the root communicator, so a rank holding no piece of a refined
level enters the reduction with its root block and returns the same
number. The single-array forms are the one-patch quadrature as it always
was and apply no mask.

The composite quadrature is exact for a linear field on a wall-bounded
box, which pins it to round-off on one refined patch, a 3×3×3 tile nest
and a three-level nest, where the same sums without the masks carry the
covered volume twice (serial and MPI suites). On Taylor–Green at 24³ with an
8³ region refined off-center, the composite energy history over 61 steps
stays within 2.5e-4 of the single-level history, the fine sampling's own
quadrature difference, where the unmasked sum sits 1.4e-2 above it; the
same check runs in `bench/tgv_energy.jl` with `refine=`. The mixing-layer
cost case's ∫Y(1−Y)dV (`bench/amr_cost.jl`) is this quadrature.

### Startup and rollback traps

Subcycling widens `compute_dt`'s one-step lag to three fine substeps on one
rate measurement, so a discontinuity inside the refined region at t = 0
needs cfl ≤ 0.2 or `StepControl(retries)`. Depth widens the number of fine
substeps one root rate measurement covers, so a startup or regrid transient
at three or more levels is an unguarded case; the three-level Sod gate runs
at cfl 0.4 without a per-substep rate check, so it is not yet a compounding
ceiling. A non-finite trajectory leaves NaN in two places a state restore
does not touch: the artificial coefficient arrays (read by `max_rate`
before the retry's first RHS) and the low-storage `du` accumulator
(0.0·NaN = NaN, so `RKA[1] = 0` cannot forget it). Both resets are gated on
`:nonfinite` failures only; after a *finite* failure the stale coefficients
are a measured part of the recovery.

### Measured costs

**The cost case** (`bench/amr_cost.jl`, np = 8): a heavy-gas blob mixing
case on a 48³ root grid with a subcycled, regridding region covering a
sixth of the volume, against uniform 48³ and 142³ references at t = 1. On
∫Y(1−Y)dV the composite lands 5× closer to the fine answer than the coarse
run at 49% of the fine wall and 24% of its memory (342 s / 655 MiB vs
696 s / 2737 MiB; coarse 45 s / 179 MiB). Pointwise in-region error is the
wrong metric: coarse and composite both sit at max ≈ 0.19 against fine
there, the sub-cell displacement of a near-discontinuous interface.
Rerun with the masked quadrature and the tag criteria: under the δ⁴ρ tag
the composite's mixedness error is 1.5e-3 against the coarse run's
6.9e-3 (4.6× closer) at 43% of the fine wall (204 s against 471 s; the
coarse run 18 s), and under the sensor tag alone (`tag=sensor
sensor=0.02`, the δ⁴ criterion parked; the artificial diffusivity number
is confined to the interface shell, nothing above 0.02 farther than 0.6
from it on the coarse grid) 1.7e-3 at 35% of the fine wall, tracking the
blob to the same final region. The two wall figures are single runs at
np = 8 with the run-to-run spread `CLAUDE.md` records.

**Per-tile costs.** Setup costs 0.06–0.12 s per tile in plan construction,
which argues for tile edges of 12 or more in 3-D. The shared RHS scratch,
measured on the annular case (N = 192, tile 6, 208 tiles of 19² plus the
root): 103.0 MB over the patch set before the pooling, 60.3 MB after, a
factor of 1.71, with 0.207 MB of each tile's 0.398 MB shared. The warm
per-step wall of that case is 0.55 s before the pooling and 0.56 s after,
at one rank on 16 threads, inside the run-to-run spread;
`bench/amr_tiles.jl` times steps after a warm-up and is the instrument.
Under MPI each tile is decomposed over its own rank range, so a tile pays
its owners' collective latency per imposition, and on a many-tile run that
is one rank; the box gathers and restriction that feed it remain collective
over the parent level.

**Workstation pathology.** Any 2-D case at np = 8 on the workstation runs
at ~7 s/step, one patch or four tiles alike, against ~0.5 s at np = 4: a
machine pathology of the kind `CLUSTER.md` records for hybrid cores, not a
tile cost, and the reason the MPI suite's tiled check is bounded to ten
steps.

## Ownership and load balance

Every rank holding every level does not survive a shell of a few hundred
tiles or a small hot spot on many ranks (constraint 2 via `_amr_dims`), so
each level is owned by a rank subset and each tile by a range within it.

### Subsets, ranges and groups

A level is owned by a rank subset of its parent's, a contiguous prefix of
the parent level's communicator, so that the subsets nest and a rank
outside level ℓ is outside every level below it (`_level_ranks` sizes the
subset, `split_level_comm` and `free_level_comm!` create and release its
communicator; a level that fits every rank holds its parent's communicator
and no split exists). Within the subset each tile is owned by a contiguous
rank range, and the subset's size is the union of the ranges
(`_tile_owners`):

- With at least as many ranks as tiles, each tile takes its own range,
  sized by weight through largest-remainder rounding, at least one rank
  and at most the count the tile admits under the 9-point scheme minimum
  (`_rank_counts`); ranks beyond the cap total hold nothing on the level.
  A one-tile level therefore reduces to `_level_ranks`.
- With more tiles than ranks, each tile takes one rank: the curve is cut into
  `np` runs of about equal weight, by the position of each tile's weight
  midpoint.

The ranges are laid along a Morton curve over the tile offsets
(`_sfc_order`), so a contiguous run of ranks holds a compact set of tiles and
most same-level exchange stays local; the tiles keep their raster order for
their ids, records and transfers. The weight at setup is the fine volume
alone; a rebalance replaces it with measured cost (below).

A rank belongs to one group, the ranks sharing its range, and holds exactly
the tiles of that group (`TileGroup`, one communicator per group split off
the level's by `split_tile_comm`, and none when every tile spans the whole
level, which is every one-tile level and every serial run).
`Solver.patches` is the root followed by the rank's tiles of each level;
`Level.tiles` records which tile each entry is, and a `LevelTransfer`
carries the rank's local index of its fine patch and of each parent patch
under its box, zero where the rank holds no piece. The alternative
considered was placeholder `Patch` entries on non-owning ranks, keeping the
vector globally complete at the price of a guard at every driver, every
interface record and every transfer table, and of a `Patch` with no
communicator to decompose over; the local-index tables make the
placeholders unnecessary, and the same-level slab layout has always had a
rank-dependent patch vector.

### Collective scoping

Each collective is scoped to the rank set that owns the data it moves:

| collective | communicator | ranks entering |
|---|---|---|
| line solves and halo exchange | the tile's `Decomp` | the tile's owners |
| `_impose_shell!` ring Allgatherv | the tile's `Decomp` | the tile's owners |
| `_sync_level_records!`, `_seed_planes!` | the level's own, point-to-point | the level's owners |
| `_gather_box!`, `save_level_box!` | the parent level's own | the parent's owners |
| `_restrict_patch!` gather | the parent level's own | the parent's owners |
| `_tag_tiles`, the box regrid's carry-over | the root communicator | every rank |
| `_rebalance_due!` busy-time Allgather | the root communicator | every rank, when `rebalance > 0` |
| `_migrate_tile!` (point-to-point, no collective) | the root communicator | the moved tile's old and new owners |
| `max_rate`, `positivity_floors`, `WhenState` | the root communicator | every rank |
| the composite diagnostics (the `Vector` forms of `volume_integral` and the rest) | the root communicator | every rank, its own patches accumulated first |

The first two rows follow from the construction rather than from a separate
scoping decision. The records are built on the level's communicator, over a
block table each rank fills with the tiles it holds, and a record's partner
is a rank number in it; the exchanges themselves are `Isend`/`Irecv`, so a
rank holding one tile or none enters them with whatever records it has. The
two cross-level rows run on the parent level's communicator with every
block table in its rank order, and a rank contributes nothing for a parent
or fine patch it holds no piece of: a child's owner need not hold the
parent tiles under its box, and the covered nodes of the restriction lie on
parent ranks outside the child's subset. Scoping those to the child's owners
instead would require the child's ranks to hold the parent's covered blocks,
which is a rank-partitioned transfer and a separate piece of work. The rank
set of every collective follows from the level index and the tile geometry
alone, never from rank-local data, so no participation decision can turn
into a deadlock.

The rate reduction stays one `Allreduce` over the whole run at the step
boundary rather than one per level plus a combine: a rank reduces the maximum
over the patches it holds, and the maximum is exact and order-independent, so
the per-level and per-rank groupings give the same number. `dt`, `t`, `step`
and every trigger therefore agree on ranks that hold no part of a level, and
the retry decision, which reads only the reduced rate and density, remains
global.

`_advance_level!` is entered by exactly the ranks owning the level it
advances. Its stages, shell impositions and halo exchanges run on the tile
communicators, one tile after another in level order, and no group waits on
another; its records, and its children's box gathers and restriction, run on
the level's communicator. The recursion into a child is guarded on child
ownership, and the substep count is fixed, so the rank sets do not diverge.

### Stored ownership

`Level.owners` is the authority across regrids. At a regrid a surviving
tile keeps its range, and a fresh tile is placed by the rule above
restricted to the ranks the survivors leave free (`_place_tiles`): the free
ranks are dealt to the fresh tiles as if contiguous, a range that would
straddle a gap between them is cut at the gap so that every group stays a
contiguous rank range, and when no rank is free a fresh tile joins the
group of the survivor nearest it on the curve. The level's rank count is
one past the highest rank in use, so a departure can leave a rank inside
the level holding no tile; such a rank enters the level's point-to-point
records with none of its own and the cross-level gathers with nothing to
contribute, which the block tables allow. A recomputed partition at every
regrid would move a survivor whenever a tile entered ahead of it on the
curve; stored ownership moves state only when a rebalance decides to.

Each regrid frees the tile groups and splits them afresh; a surviving
tile's Cartesian communicator is independent of the group it was split from
and of the level communicator a resize replaces, so it survives both. The
communicators a regrid discards are freed at the regrid, for the reason
`free_communicators!` records (left to garbage collection they exhaust
MPI's context-id budget at the regrid cadence). A transfer's refinement
chains, eight decompositions per tile held on every rank of the parent's
subset, are built on `COMM_SELF` and own no communicators, so the budget
is spent on the tiles' own decompositions alone.

### Rebalancing on measured load

The partition is recomputed only by a rebalance. Every rank measures its own
busy time per step, the step wall less the time it spent inside the
collectives every rank enters (the rate and floor reductions, the level box
and restriction gathers) and in the refined level's record exchange
(`Solver.wall_wait`); the step wall alone is nearly uniform under any load,
since a lightly loaded rank simply waits longer at the same reductions. At
each regrid check the busy time over the interval, the check's own regrid
work excluded, is Allgathered over the root communicator, and the level is
repartitioned when the ratio of the largest to the mean exceeds
`RegridSpec.rebalance` at `RegridSpec.persist` consecutive checks
(`_rebalance_due!`); the streak resets at a balanced check, so tile flicker
at the tag boundary does not move state every interval. The weights are
then measured rather than assumed (`_measured_weights`): a tile costs the
busy time of its owner ranks, a group's shared among its tiles by fine
volume, and a fresh tile takes its volume at the mean measured cost per
node. The interface factor, which the per-apply reduced-solve fence makes
larger for small tiles than their volume, is therefore taken by the run
that it balances, which is the only place it can be taken, since the
per-rank costs on rzhound and rzadams differ from the workstation's by
27–66x and move with rank placement. The default is off (`rebalance = 0`),
and a threshold of one repartitions whenever the streak reaches `persist`,
since any measured spread exceeds it. `bench/amr_balance.jl` prints the
per-check max/mean, the owner ranges, and the transient memory of each
moved tile of a tiled Sod run with rebalancing off and on, as a check of
the mechanics only; on that case the stored groups drift apart in tile
count with rebalancing off, and with it on at threshold one every check
repartitions and the partition then follows the timing noise from check to
check, which is what the threshold and `persist` exist to damp.

### Migration of a moved tile

A surviving tile whose range a rebalance moved is rebuilt on its new owners
and takes its solution from its old owners' blocks directly
(`_migrate_tile!`). Both block tables are already Allgathered in the root
communicator's rank order, the old one on the live transfer and the new one
on the transfer the rebuilt tile receives, so every rank derives the
identical message list, one message per rank pair whose old and new blocks
meet inside the tile's interior, and posts `Isend`/`Irecv` on the root
communicator for its own entries only; a rank with nothing to send or
receive posts nothing, no collective runs, and a rank keeping a node
between its own two blocks copies it locally. The interior is one node off
both boundary planes along every active dimension, the same rule as the box
regrid's carry-over, since the old planes were imposed data and the new
ones are re-imposed at the next shell fill. The tiles migrate one at a
time, each after the collective interpolation that initializes its shell,
so one tag suffices and the `Waitall` of each precedes the next tile's
posts; the old decompositions are freed after the last, in the same order
on every rank. Transient memory per rank is the rank's received blocks
rather than a replica of the tile's whole state per moved tile.

The replicated gather of the box regrid doubles as the migration's
reference: under the `MIGRATION_AUDIT` test hook a tiled regrid gathers
each moved tile as well, applies the carry-over to a copy of the rebuilt
state, and counts the slots at which the migrated state differs. The MPI
suite runs the rebalancing Sod track with the hook on and holds the count
at zero, so the two paths agree bitwise where both apply.

### Reproducibility tier

A tile owned by a proper subset is a different decomposition of it, so it
reproduces the every-rank answer to round-off rather than bitwise, the tier
the MPI suite applies to decomposed patches; the suite measures 0 to 6e-15
on the tiled wave cases at np = 2, 4 and 8, and the tiled Sod regrid with
rebalancing on reaches the serial time to 5e-18. A rank outside a subset
follows the same step sequence and trigger firing as one inside it, which
the suite pins; the smallest such case refines a region of four coarse
nodes, which `_amr_dims` cannot split over two ranks and which runs with
the level owned by one.

## I/O and restart

A checkpoint of a refined run carries, beside the root's state, a record of
the hierarchy and every tile's state, and a restart rebuilds the level from
the record on whatever rank count it is given (`src/io_levels.jl`).

### What the checkpoint carries

Per refined level, the tile layout as the regrid derives it, the tile
regions in the parent level's node space, and the stored owner ranges
(`Level.owners`, the authority across regrids); then the regrid state: the
step of the last check, the check count and the per-tile creation record
the lifetime and the hold band read, and the rebalance streak. The tag
thresholds, the buffer and the interval are constructor configuration,
recorded for provenance and not restored; the tile edge is recorded and
checked, since a layout on one lattice cannot be placed on another. Rank 0
assembles the record, the level subsets being prefixes so that it holds
every level's layout, and broadcasts it.

Every patch's state is its interior block of Q and, when the artificial
properties are enabled, its μ*, β*, κ* and D* arrays. The coefficients are
state because `max_rate` sizes the next step from the ones the last
right-hand side left behind: a restart from Q alone measures its first rate
with zero coefficients and takes a first step the uninterrupted run would
not have, after which the nonlinear sensors amplify the difference. With
them a restart continues bit for bit. The primitives of the evaluation that
wrote the coefficients, which the sensor tag criterion reads beside them,
are not carried; a restart refreshes the primitives from the restored state,
so with `tag_sensor_threshold > 0` a marginal node may tag differently at
the first check after a restart. Every other criterion reads Q.

Both formats carry the same record. The shared HDF5 file
(`save_checkpoint_hdf5` on a state vector) holds it as readable tables
(`hierarchy/level<ℓ>/regions`, `.../owners`, `hierarchy/regrid/...`) and
each tile's state as a dataset over the tile's node space written in
per-rank hyperslabs (`levels/<ℓ>/tiles/<t>/Q`, `.../art`); the per-rank file
(`save_checkpoint`) holds the record as a flat image and this rank's blocks
of the tiles it holds, and restores onto the rank count that wrote it only.
The single-patch checkpoint carries the coefficient arrays and the level
count as well (format 4), so an unrefined restart continues bit for bit
too.

### Restart on any rank count

`restore_hierarchy!` compares the recorded level-1 layout with the
solver's, which the constructor built from whatever `refine` region it was
given. When the regions differ, or the rank count is the writing one and
the ownership differs, `_replace_level!` rebuilds the level: the shape of
`_regrid_tiles!` without the tag decision, the carry and the migration,
every tile fresh (its state comes from the file afterwards), the level and
tile communicators split afresh, the covered mask and the geometry
following. On the writing rank count the stored ownership is restored, and
the same tiles on the same rank ranges are the same decompositions, so the
run continues bit for bit, later regrids included. On another rank count
`_tile_owners` partitions the level afresh, a different decomposition of
the same tiles, and the continuation agrees to round-off, the tier a
subset-owned tile already holds ([Reproducibility tier](#reproducibility-tier)).
A level below the first is static, since regridding is two-level, so its
layout must be the solver's own and the constructor's `refine` supplies it;
rebuilding needs a `RegridSpec`, where the schemes a fresh tile is planned
with live. Every decision derives from the record, identical on every rank,
so the communicator splits are reached together.

Measured: the tiled and box Sod regrid cases checkpointed at step 23 and
continued to step 130 through the regrids in between agree with the
uninterrupted run at every slot, tag history included, serially and (the
tiled case at 400 nodes, checkpointed at step 21 and continued to 41) at
np = 2, 4 and 8; a twelve-tile wave written on half the ranks and restored
on all of them, rebuilt from a six-tile initial region, continues to 1e-12
in the wave error.

### Field output

`save_vtk` on a state vector writes one `.vtr` piece per patch per rank, in
the patch's own node space with physical coordinates, and rank 0 lists them
under a `.vtm` multiblock index, one block per level, from a gather of the
piece lists. A coarse node a child level covers entirely (`Patch.covered`
at every orthant bit) carries the `vtkGhostType` hidden-point flag, so the
composite renders the finest data at every point and the coarse-fine face,
whose coarse nodes are half covered, once. `stride` acts in each patch's
node space; `slice` names a root plane, mapped to each patch's coincident
node, and a patch the plane misses writes no piece. `FieldWriter` on a
state vector numbers `.vtm` frames into its `.pvd`. Not covered: a
shared-file (`save_hdf5`) dump of a hierarchy, which needs an XDMF tree of
grids, and the checkpoint of a same-level slab layout.

## The device backend

Architecture: KernelAbstractions.jl, one codebase, CPU and GPU backends. KA
is the one hard dependency; device packages (AMDGPU.jl, CUDA.jl) arrive
through the user's environment. `DeviceBackend(ka)` selects device storage
through the same `field(backend, decomp)` / `allocate_state` interface
`CPUBackend` uses.

### Pointwise kernels

Every pointwise phase is one shared `@inline` per-point `_point!` body
launched through `pointwise!`: `Array` storage takes the `@threaded` loop,
any other storage a KernelAbstractions kernel. The CPU keeps `@threaded`
because KA-CPU measured 2.8× (flux assembly) to 40–50× (RK update) slower
at 64³, a per-launch task-spawn cost with no work threshold.

Kernel-argument adaptation is the central design concern. A `Vector{A}` or
`Matrix{A}` kernel argument **hangs** in adaptation rather than erroring,
so field collections reach bodies as `FieldVector`/`FieldMatrix`: host
wrappers that adapt to isbits `NTuple` mirrors only at launch (holding the
tuple form on the host measured 3× on `assemble_fluxes!`). The gas-model
EOS objects adapt to coefficient mirrors the same way. Bodies take no
`::Type` argument (9× per-point dispatch on both paths), and a splatted
kernel-argument tuple longer than 32 elements is an `InvalidIRError` on
device; `test/device_tests.jl` asserts the budget for every body.

### Line solves

`device_plan` mirrors a host `DirPlan`/`BandPlan` onto a KA backend as a
`DevicePlan`: fill, elimination sweeps (one thread per line), spike
correction, and scatter as kernels in a (lines × n) layout for every
dimension. The reduced 2qP × 2qP interface stage stays host-side: a pack
kernel writes the 2q interface values per line, one device-to-host copy
feeds the same `_reduced_solve!` the host layouts share, and one copy
returns the corrections, so collective ordering is identical to the host
path. The kernels carry a `colwise` switch because the two host layouts
disagree in the banded solve (the x sweep divides by the diagonal where the
transposed sweeps multiply by its inverse); mirroring each dimension's own
convention keeps the device output bitwise per dimension.

### Residency

`Solver(backend = DeviceBackend(ka))` builds a fully device-resident solver:
every plan converts through `backend_plan`, every field and state allocates
on the backend, and every phase between the step drivers and the arrays is
a launchable body, a `DevicePlan` solve, or an exact storage-level
reduction. Three operations are host-staged: geometry (setup-time upload),
`initialize!` and `DirichletBC` host closures (staging blocks), and tagging
(a coarse-block download at the regrid cadence).

### Communication

A distributed device patch stages every message through the backend: a pack
kernel, one contiguous device↔host copy per message, then the unchanged MPI
path over the host halo buffers; the fold-pair Sendrecv and the regrid
gathers and migration stage the same way. `device_mpi_direct(backend)` is
the hook for direct device-pointer MPI and defaults to host staging. Staging
buffers allocate per exchange through `similar` (device allocators pool); a
keyed cache was reverted after its `Any`-typed lookup put 33 dispatch sites
into `compute_rhs!`'s jetcheck report. `TRACK_DEVICE_TRANSFERS` tallies the
staged copies.

### Launch policy

Kernels queue on the backend's one in-order stream without a host
synchronize per launch. The only unconditional fence is the one the
algorithm requires, before the host reads the packed interface values of
the reduced solve, once per `apply_along!`. `DEVICE_SYNC[] = true` restores
synchronize-per-launch and is the correctness fallback. Removing the
per-launch synchronize took 28–32% off the device step. Per-patch streams
are not implemented: supported device configurations hold one patch per
rank or a sequentially coupled coarse/fine pair, and the remaining gap to
CPU is bound by the per-apply reduced-solve fence, which a second stream
cannot remove. Batched cross-tile launches ([Device](#device) in Part II)
are the planned replacement for streams.

### Precision

Uniform Float32 is a supported opt-in end to end, with literals typed
against the state's eltype and `positive_floor` replacing raw 1e-300
guards. CPU measurement (64³ TGV, t = 10): identical peak dissipation to the
printed precision, 2.00× smaller footprint, 1.10× wall, and mean-density
drift 1.4e-4 against 7.5e-13; the drift is why Float32 is not the default.
On device Float32 runs 1.25× the Float64 rate, because the step is bounded
by launch submission and fences, not arithmetic. A mixed-precision policy
waits for a memory-capacity-bound case on the target machine, where FP64
runs at full rate and the question is traffic, not FLOPs.

## Verification

The oracle hierarchy, strongest first:

1. **Bitwise equality against the CPU solver over full runs.** Every phase
   is per-point independent, a line sweep mirrored operation-for-operation,
   an element copy, or an exact max/min reduction, so the whole composition
   is bitwise. Every device stage was accepted on `max |device − cpu| = 0`
   over full runs: smooth periodic and closed cases, NSCBC, folds,
   freestream/GCL, a 578-step Sod, refined static and regridding runs, 64³
   TGV histories in both precisions, and distributed runs at np = 2/4/8, in
   both launch-policy modes.
2. **Bitwise equality within a patch across configurations.** The
   single-patch and level paths are held bit-identical where the
   configuration admits it (`test/convergence.jl`, `test/level_tests.jl`),
   and the migration is held bitwise against the replicated carry it
   replaced; a restart on the writing rank count continues the run bitwise
   through its later regrids. A decomposed patch reproduces the serial
   answer to round-off (1e-15 to 6e-15), which is the tier for anything a
   subset owns, a restart on another rank count included.
3. **CPU-side pins without a GPU.** `FORCE_KA`, `FORCE_DEVICE_EXCHANGE`, and
   `DEVICE_SYNC` route ordinary arrays through the device paths, held
   bitwise by the serial and MPI suites. They cannot catch what only actual
   device storage exercises (lesson 3), so the real-GPU bench battery is
   part of the gate for every change to the device path.
4. **Manufactured-solution orders and physical-metric gates** where
   interfaces make bit-exactness unavailable by construction: the
   two-conforming-patch and level-boundary orders, the moving-region Sod
   and Shu–Osher gates, the cost case's ∫Y(1−Y)dV.

Bench scripts: `bench/amr_transfer.jl` (operator conditioning),
`bench/amr_tiles.jl` (tiled cover and per-step wall), `bench/amr_cost.jl`
(the cost case), `bench/amr_balance.jl` (rebalance and migration
mechanics), `bench/pointwise_ka.jl` (launcher acceptance),
`bench/device_bringup.jl` (kernel-level bring-up), `bench/device_solver.jl`
(whole-solver device battery), `bench/device_mpi.jl` (distributed device
runs with transfer accounting), `bench/device_floors.jl` (the cross-machine
instrument: launch floors, stall watch, line-solve matrix, TGV step table).
The device scripts need an environment carrying the device package; the
workstation keeps one at `~/.julia/dev/CompactLES_gpu_env`.

## Performance summary

The performance target is an LLNL rzadams / El Capitan-class machine
(MI300A APUs, GPU-aware Cray MPICH). Workstation numbers (RX 6800 XT,
AMDGPU.jl on Windows/HIP) are evidence that the structural pitfalls are
gone, never a performance claim about the target: RDNA2 runs vector FP64 at
1/16 the FP32 rate where MI300A runs it at full rate, the APU's unified
memory removes the staging economics every device↔host number is priced
in, and GPU-aware MPI flips `device_mpi_direct`. Run-to-run spread on the
workstation is 10–20%; read ratios, not third digits.

Workstation: 64³ TGV full step, device 0.146 s/step (Float64) / 0.117
(Float32) against ~0.12/0.10 for the 8-thread CPU, the floor being launch
submission plus one reduced-solve fence per apply; isolated kernels clear
CPU where launches amortize (flux assembly 9.9× at 64³ two-species);
staged halo/pair copies 0.6–6.6% of device wall, reduced-interface copies
2–5%; first-launch kernel compilation ~9 s per body.

rzadams (2026-08-19/20, `bench/logs/rzadams_20260819*.txt`): kernel
submission 10 µs, launch+sync 25 µs, line solves 0.14–0.40 ms/apply, the
64³ TGV over 4 APUs at 0.074–0.088 s/step with an F64/F32 ratio near 1.2,
and a 256³ single-species TGV over 4 APUs at 0.35 s/step baseline
(24,490 steps to t = 10 in 3.69 h; `CALIBRATION.md`). The open issue is an
intermittent stall mode in which every device wait costs an integer number
of milliseconds (13.000 ms medians) for seconds to beyond 30 s; it sits
below the Julia layer, and inflated the 256³ run's solver average to
0.52 s/step even at `-t 1`. `reference/rocm_wait_stall_report.md` has the
characterization. Until it is resolved, run device-resident rzadams jobs at
`-t 1` per rank, and treat any wall number from a multithreaded process as
untrustworthy without a stall watch beside it.

## Lessons

Ordered roughly by how expensive they were to learn. Each is enforced or
guarded somewhere; none should be re-derived.

1. **The live inter-level coupling must match the data's provenance.**
   Deconvolution belongs only where the matching convolution has been
   applied; on live point samples the invertible pair is O(h²) (order
   1.3–1.7) against 3.5–3.7 for interpolation/injection.
2. **Replicated coupling work scales as rank count × imposition count, and
   subcycling multiplies impositions.** Redistribute work, never re-derive
   values: distributing the chains by component recovered the cost
   advantage (85% → 49% of uniform-fine wall) while keeping serial results
   bit-identical.
3. **KA-CPU equality cannot certify the device path.** Four defects passed
   every KA-CPU test and failed only on real device storage (an un-adapted
   kernel argument, the >32-element splat, host-typed placeholders in
   `_build_fine_patch`, and a parameter named `parent` shadowing
   `Base.parent` in the covered-region write, whose host branch never calls
   it). On the KA CPU backend device arrays *are* host arrays, and the last
   of those shows the branch need not be a kernel to be device-only.
4. **A `Vector` of arrays as a kernel argument hangs; adaptation is a
   design surface.** Every new collection that reaches a kernel needs its
   adapt story decided first.
5. **Specialization heuristics are part of the interface.** A `::Type`
   through the launcher cost 9× as silent per-point dispatch; an
   `Any`-typed cache lookup behind a runtime branch put 33 dispatch sites
   into the RHS report without ever executing. Jetcheck deltas are the
   tripwire.
6. **Bitwise device equality is achievable and pays for itself as a test
   oracle.** A tolerance-based gate loose enough to pass accumulated
   round-off also passes a defect of that size; a last-bit difference
   localizes to one operation.
7. **The launch floor was host round trips, not kernels.** Deleting
   synchronization is the first tool against launch overhead; a fence the
   algorithm requires bounds what any stream topology can recover.
8. **Pointwise metrics misread interface-dominated fields.** Judge a cost
   case on the quantity the refinement exists to predict.
9. **Restriction must stand off the imposed boundary** (`RESTRICT_MARGIN`;
   gain ≈ 2 per step without it).
10. **NaN survives rollback** in the artificial coefficient arrays and the
    low-storage accumulator; the resets are gated on `:nonfinite` failures.
11. **A device package loaded inside `main` is a world-age trap.** Bench
    scripts load the backend at top level before `mpi_main`.
12. **`MPI.Dims_create` takes no account of scheme minima.** `_amr_dims`
    searches factorizations under the 9-point constraint and errors at
    setup or at the regrid that shrank the region.
13. **Every ownership decision derives from reduced data.** The rank set
    of a collective follows from the level index and the tile geometry, a
    rebalance from an Allgathered measurement, a migration from two
    Allgathered block tables; a rank-local decision at any of those points
    is a deadlock with no symptom but zero CPU on every rank.
14. **A partition recomputed at every regrid moves state that did not
    need to move.** Ownership is stored and moves only on a measured
    imbalance held over several checks; the workstation's timing noise
    alone repartitions a threshold-one run at every check.

## Scope boundaries today

Configurations rejected at setup, and the reason:

- **Patched runs** (same-level `patch_grid`) reject folds (constraint 4),
  the `:d8` detector, stretching along the patched dimension, and an
  explicit `dims`; C10 takes `n_halo ≥ 5`. The layout tiles slabs along
  one dimension, so corner-coupled adjacency does not arise. Field output
  takes the multiblock form; a slab layout has no checkpoint.
- **Refined runs** require Cartesian metric, no stretching, no folds, a
  tridiagonal filter, `:delta4`, one region per level, and no same-level
  `patch_grid` alongside; C10 takes `n_halo ≥ 5`.
  `level_restriction = :filter` is serial-only.
  Each region must nest by `max(n_halo, LEVEL_BUFFER)` parent nodes inside
  the patches of the level above and span ≥ 4 parent nodes per active
  dimension; a tiled level's tiles are clipped to that margin at the
  domain edge and must still lie inside the parent tiles. Regridding is
  two-level. Tiled levels take the host backend. Rebalancing requires a
  tiled, regridding level.
- **Device runs** take a single patch per solver (the interface-record
  copies are host loops), reject `Nasa9Mixture` (no fixed-width device
  mirror), a pointwise NSCBC inflow `target` (host closure),
  `StepControl.floor_ratio > 0` and `dt_report` (host sweeps), and `:filter`
  restriction.
- The artificial-property sensors are built per patch with closed-edge
  clamping at interfaces; no gate has measured the effect.

---

# Part II: the production AMR

## What remains

The delivered system rests on structural assumptions that a production AMR
replaces. Each names the design subsection that replaces it and the
sequencing item that delivers it.

| assumption | replaced by | item |
|---|---|---|
| tiled levels are host-only; the transfer chain and tagging run on the host | [Device](#device) | 7 |
| no rate check per substep; measured weights carry the root's work | [Ownership refinements and the rate check](#ownership-refinements-and-the-rate-check) | as needed |
| regridding is two-level | not sequenced; see [Ownership refinements and the rate check](#ownership-refinements-and-the-rate-check) | — |

The box regrid's replicated carry (`tile = 0`) is not on the list: the
tiled level is the production path, and the box path serves the one-patch
configurations that need no rank-partitioned carry.

## Design of the remaining work

Each subsection is the design as it stands; the order is the dependency
order, which [Sequencing](#sequencing) turns into deliverables with gates.

### Device

Four pieces, in order of dependence. The same-level interface-record
pack/copy loops stage through the backend as the halo path was staged
(mechanical). The level-transfer chain runs on the device: the gathered box
is uploaded once, the tensor-product Lagrange interpolation and the Hermite
blend are pointwise bodies, and the shell ring is packed on the device
before its Allgatherv, so a subcycled step's ~20 impositions cost no
host-side arithmetic and, on a unified-memory APU, no copies. Tagging
evaluates on the device as above.

The last piece replaces per-patch streams. With many equal-extent tiles per
rank, line solves batch across a level's tiles into one (lines × n) launch
per dimension, which `DevicePlan` already supports by layout, and the
pointwise bodies launch over a level's tiles as one index space. This
amortizes both the launch submission and the per-apply reduced-solve fence
over the tile count, which is the measured floor of the device step, and
it is why the tiles are fixed-size. Streams remain unbuilt unless a
measurement shows the batched launches still leave the device idle; if the
fence itself still binds after batching, the routes are an on-device
reduced solve (redundant per-rank solves on gathered ends) or GPU-aware MPI
through `device_mpi_direct`, both rzadams measurements.

`Nasa9Mixture` is the last EOS off the device: a flattened, fixed-width
interval table with `Adapt.adapt_structure` plus the Newton inversion in
the body; mechanical.

### Ownership refinements and the rate check

**The measured weights carry the root's work.** Each rank's busy time
includes its root-level work along with its tiles', which overstates the
cost of a tile on a rank holding few; a rank holding no tile measures that
baseline. Subtracting it is a refinement to make once a cluster case shows
the bias; the correction is the per-rank root work, which is nearly uniform
across ranks under a slab root decomposition and so cancels in the ratio
until tile counts per rank differ widely.

**A rate check per substep.** Depth widens the number of fine substeps one
root rate measurement covers, so a startup or regrid transient at three or
more levels is a real unguarded case, though not yet a measured ceiling.
When built it sits after the stage-1 RHS has refreshed the artificial
coefficients and before the update, limited at first to startup and regrid
steps, one `max_rate` reduction per substep, and it must return a failure
to `run!` rather than throw: the retry handling runs before `step!`, and an
exception inside `_advance_level!` would escape it. A violated rate then
becomes a `SolverFailure` retry rather than a silent overstep. Whether a
cheaper dense output than the recurring Hermite endpoint RHS is worth
building at three or more levels is a measurement to make once a case
demands it.

**Multi-level regridding** is not sequenced. `_advance_level!` and the
ownership tables are written per level, so the recursion admits a regrid at
every depth, but `regrid!` and `_regrid_tiles!` assume the root and one
refined level, and a regrid at depth ℓ must re-nest every level below it.
The implosion target needs it only once the shell collapses to a hot spot
with a level-2 nest inside; until a case demands it, it stays a two-level
mechanism.

### Open numerics

- **Conservation drift.** Compact closure rows at an interface do not
  telescope, so global conservation carries a drift term (1.2e-8 relative
  per long periodic run at a same-level interface; 1e-4 over a full
  two-level shock crossing) where a finite-volume code would reflux. On a
  long mixing-layer run that will show in ∫Y(1−Y). The surface-correction
  fallback on interface fluxes is designed but unbuilt; build it only when
  a case shows the drift competing with the answer, and measure the drift
  on the mixing-layer case first.
- **Sensors at level boundaries.** A shock crossing a coarse–fine boundary
  is where the per-patch, closed-edge-clamped sensor construction will
  show as a reflected wave. Needs a gate before either target problem is
  trusted.
- **The filter at the shell.** The fine patch filters its imposed shell
  nodes before they are overwritten. The `:onesided` filter rows that fixed
  the Brady–Livescu wall case (`CALIBRATION.md`) may matter here too;
  untested.
- **Filter cadence under subcycling** is measured on Sod gates only; the
  smooth-turbulence dissipation budget under subcycling ties into the
  filter calibration item in `ROADMAP.md`.
- **Fold-adjacent refinement** stays forbidden. Converging-shock problems on
  the folded grids use a globally fine level 0 in r near the fold; on a
  Cartesian grid the question does not arise.

## Open measurements

Measurements the design depends on that the workstation cannot make, or
has not made. Each is a bench run, not a code change, and the design above
is written so that its result changes a parameter or a decision rather than
a structure.

- **The 3-D tile shell.** The implosion's fine level as a shell of tiles at
  a memory-sized tile edge (12 or more, from the per-tile setup cost) is a
  bench measurement, not a testset; `bench/amr_tiles.jl` is the 2-D
  instrument and its 3-D counterpart is unwritten.
- **Per-imposition latency after tile ownership.** A tile pays its owners'
  collective latency per shell imposition, one rank on a many-tile run, and
  the parent-level gathers that feed it remain collective. Whether that
  latency still binds is a cluster measurement.
- **The tiled-level overhead question.** A rough count says the RHS work of
  the tiles, at 3 substeps per root step over padded extents, should land
  near 2 s per subcycled step for 40 tiles of 37² on a 96² root, where the
  one cold measurement was 8–10 s; the warm annular reading at a different
  configuration matches expectation. Remeasure warm at the original
  configuration.
- **What a rebalance is worth, and what the migration saves.** The
  per-rank costs on rzhound and rzadams differ from the workstation's by
  27–66x and move with rank placement, and on the workstation's tiled Sod a
  moved tile is one kilobyte. The case that decides both is the implosion
  at a rank count for which the replicated carry is measurably the
  regrid's cost.
- **The rzadams wait stall** ([Performance summary](#performance-summary))
  sits below the Julia layer and gates every device wall number on that
  machine.

## Sequencing

Each item names its gate. Nothing is built ahead of the item before it.
Items 1–6 are delivered and described in Part I: the level hierarchy with
its recursive driver ([Refinement](#refinement)), the tiled fine level with
full adjacency and the set-difference regrid
([Tiles and adjacency](#tiles-and-adjacency)), ownership
([Ownership and load balance](#ownership-and-load-balance)), which was
pulled ahead of tagging and I/O because a flat globally ordered
`solver.patches`, an equal patch count per rank, and full-communicator
gather tables on every `LevelTransfer` are structural, and diagnostics or
I/O built on them would have been rebuilt once ranks held different tile
subsets, the tag criteria with their hysteresis and the covered masks
([Tagging and regridding](#tagging-and-regridding),
[Diagnostics on the composite grid](#diagnostics-on-the-composite-grid)),
the checkpoint of the hierarchy with its restart on any rank count and
the multiblock field output ([I/O and restart](#io-and-restart)), whose
gate was a restart continuing the regridding cases bit for bit on the
writing rank count and to round-off on another, and the banded interface
closures with the buffers remeasured at C10
([Banded schemes at interfaces](#banded-schemes-at-interfaces)), whose
gate was the two-conforming-patch orders and reflection amplitude
reproduced at C10 and the localization study at C10 setting the buffers.
Their gates hold in the serial and MPI suites; the per-substep rate check
deferred from item 1 is under
[Ownership refinements and the rate check](#ownership-refinements-and-the-rate-check),
and the cluster measurements the items leave open are under
[Open measurements](#open-measurements).

7. **Device: staged interface records, device transfer chain and tagging,
   batched cross-tile launches.** Gate: bitwise equality against the CPU
   hierarchy over full refined runs; the implosion case's device step
   measured against its CPU step on the workstation and on rzadams.
8. **Numerics debts on the target problems**: conservation drift on the
   long mixing-layer run, the sensor at a level boundary under a crossing
   shock, filter rows at the shell. Each is a measurement first and a
   change only if the measurement demands one.

Items 1–6 make a run possible, restartable, its diagnostics counting each
node once, and C10 available on it; 7 makes it fast.
The conservation drift and the boundary numerics of item 8 remain open
debts on the target problems, so a production acceptance still needs a
conservation budget tied to the reported mixing and energy quantities. The
rate check and the root-work correction land inside whichever item first
needs them.
