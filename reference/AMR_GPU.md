# CompactLES — Patch AMR and the GPU backend

This document covers three things: the patch-based adaptive mesh refinement
(AMR) and device backend as delivered, the constraints and measurements that
bind any extension of them, and the design and sequencing for the production
AMR that the delivered capability is not yet. Prerequisite reading:
`DESIGN.md` for the compact operators, the distributed line solve, and the
coordinate folds.

Every number here was produced by a bench script or testset and is usually
still guarded by one. `HISTORY.md` and the git history name the delivery
stages of the original plan; they map onto this document as Stage 1 →
[Transfer operators](#transfer-operators), Stage 2 →
[Patches and same-level interfaces](#patches-and-same-level-interfaces),
Stages 3–4 → [Two-level refinement](#two-level-refinement), G1–G4a → the
subsections of [The device backend](#the-device-backend).

## Contents

1. [Status](#status)
2. [Target problems](#target-problems)
3. [Constraints](#constraints)
4. [Transfer operators](#transfer-operators)
5. [Patches and same-level interfaces](#patches-and-same-level-interfaces)
6. [Two-level refinement](#two-level-refinement)
7. [The device backend](#the-device-backend)
8. [Verification](#verification)
9. [Performance summary](#performance-summary)
10. [Lessons](#lessons)
11. [Scope boundaries today](#scope-boundaries-today)
12. [Production AMR: design](#production-amr-design)
13. [Sequencing](#sequencing)

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
  root and the tiles of its range only. The
  level coupling gathers replicated region data and distributes its
  interpolation chains by conserved component.
- **Storage backend**: `CPUBackend` (`Array`s, `@threaded` loops) or
  `DeviceBackend(ka)` wrapping any KernelAbstractions backend. A
  device-resident solver (decomposed, refined, regridding, Float64 or
  Float32) reproduces the CPU solver bitwise over full runs.

The pointwise physics is written once as per-point bodies launched through
`pointwise!` (`src/pointwise.jl`); the compact line solves run through host
plans or their device mirrors (`src/lines_device.jl`) behind one
`apply_along!` entry point. There is no second code path for the device
beyond the launchers.

What this is not: a production AMR. The tag is one criterion without
hysteresis, regridding is two-level only, a tile whose owner range moves at
a regrid is carried through a replicated gather rather than migrated
point-to-point, nothing rebalances on measured load, I/O is single-patch,
diagnostics do not mask covered nodes, tiled levels are host-only, and the
banded (C10) schemes have no interface closures.
Each of these is a structural assumption, and
[Production AMR: design](#production-amr-design) replaces them.

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
two-level solver is closest to this class; the measured mixing case below is
of this shape.

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
   number. The number is a C6 number; C10's inverse decays more slowly and
   the buffer must be remeasured for it.

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
divergence plans so the discrete GCL cancellation is preserved.
`interface_closures` is defined for `CompactScheme` only; the banded
schemes have none.

Measured at the two-conforming-patch gates: entropy-wave order 3.1–3.5
across the interface (the divergence's one-sided rows binding); acoustic
pulse reflected amplitude 2.3e-3 of incident at 192 points, converging at
≈ 5th order; conservation drift 1.2e-8 relative over a long periodic run
against 4.5e-15 single-patch. Rank partitioning reproduces the serial
two-patch answer bitwise at one rank per patch; once a patch itself
decomposes, agreement is round-off (3.1e-15 at np = 4 against a 9.5e-8
signal) with identical step counts.

## Two-level refinement

`src/levels.jl`, `src/regrid.jl`. One level-1 patch covers a
`refine::BlockRegion` of the root grid at ratio 3, node-centered (m coarse
nodes ↔ 3m − 2 fine nodes), joining `solver.patches` so the same drivers
advance it and the shared `max_rate` reduction supplies the timestep. The
hierarchy generalizes to a chain (`refine` as a vector of regions, each in
the parent level's node space; `solver.levels`, one `Level` per depth with
one `LevelTransfer` per patch) and each level to a set of lattice tiles
(`tile`), and every mechanism below applies between a patch and its
parents; the measurements are two-level, one-patch ones.

**The live coupling is interpolation and injection, not the invertible
pair.** The pair's contract is that prolongation input is samples of the
*filtered* field, which restricted data is and the live coarse solution is
not. Deconvolving point samples sharpens data that was never smoothed;
filtering on the way down writes an attenuated representation into a field
of point samples; both are O(h²) against the solution, and the
manufactured-solution gate measured order 1.3–1.7 through the pair. The
default coupling is therefore order-6 Lagrange interpolation up
(`interpolate!`) and coincident-node injection down (a sampled
`gather_region!`), which measures order 3.46/3.64 with errors three decades
lower. `level_restriction = :filter` keeps the anti-aliasing pair
selectable, and regridding initializes *new* fine cells by interpolation for
the same reason.

**Coupling schedule.** After every RK stage, the fine patch's ghost ring
*and* boundary-plane nodes are overwritten from the coarse state
interpolated over a box extending `LEVEL_BUFFER = 4` coarse nodes beyond the
region. This is Dirichlet forcing by the coarse solution, and it removes the
inter-level drift mode without an averaging step. After every completed step
the fine state writes back onto the covered coarse region, holding
`RESTRICT_MARGIN = 2` coarse nodes off the boundary. The margin is not
optional: restricting all the way to the boundary closes an amplifying loop
through the fine solution's least-accurate nodes (its imposed shell) into
the closure rows that feed the next shell (measured gain ≈ 2 per step, flat
with the margin in place).

**Subcycling** (`subcycle = true`) is Berger–Oliger: one coarse step with
the fine level frozen, then three fine steps of dt/3, the fine shell imposed
at every fine stage time from a cubic Hermite reconstruction of the coarse
trajectory on the buffered box (values and RHS at both step ends; O(dt⁴),
matching LSRK54, which has no free dense output). The t^{n+1} endpoint RHS
costs one extra coarse RHS per step, since the next step's stage-1 RHS
arrives too late and restriction invalidates a cached one. Each patch's rate
is divided by 3^level, so the step is coarse-limited. Each level filters its
own state at its own step cadence, which is dt-consistent whenever
`filter_cfl > 0`. Measured: the entropy-wave orders are unchanged by
subcycling at a third of the steps, and the subcycled Sod gate improves on
the global-dt one (ahead-of-shock noise 5.7e-11 vs 6.4e-10; mass drift
9.8e-5 vs 1.36e-4).

**Tagging and regridding** (`regrid_interval = K`). The tag is the relative
undivided fourth difference of the mixture density, Σ_d |δ⁴_d ρ|/ρ >
`tag_threshold`, read from Q directly; buffered by `tag_buffer` (default 4,
the pollution-decay figure) and clamped to the nesting margin. Clustering is
the bounding box of the tagged set. A regrid rebuilds the fine patch and
`LevelTransfer` from schemes retained on a `RegridSpec`, initializes the new
state by interpolation, and copies surviving fine data across the region
overlap on the coincident lattice, one node off both boundary planes. The
retry savepoint refreshes at a regrid.

Measured on moving-region gates: Sod at N = 201 coarse against a 601-node
uniform-fine reference, composite density error 2.8e-3 where uniform-coarse
gives 7.3e-2 (26×), with fine resolution over a third of the domain;
Shu–Osher 10× better in L∞ over the wave train, 6.7× in L1, at 2497 coarse
steps against the reference's 4662.

**Startup and rollback traps.** Subcycling widens `compute_dt`'s one-step
lag to three fine substeps on one rate measurement, so a discontinuity
inside the refined region at t = 0 needs cfl ≤ 0.2 or `StepControl(retries)`.
A non-finite trajectory leaves NaN in two places a state restore does not
touch: the artificial coefficient arrays (read by `max_rate` before the
retry's first RHS) and the low-storage `du` accumulator (0.0·NaN = NaN, so
`RKA[1] = 0` cannot forget it). Both resets are gated on `:nonfinite`
failures only; after a *finite* failure the stale coefficients are a
measured part of the recovery.

**Distribution.** Each level decomposes over its own rank subset (see
[Ownership and load balance](#ownership-and-load-balance)), and the
coupling runs on a replicated-data, distributed-work split. Data
replicates: `gather_region!` assembles a node region of a distributed field
on every rank with one Allgatherv (the buffered coarse box per shell
imposition, the coincident nodes for `:inject` restriction, the surviving
fine state at a regrid). Work distributes: a subcycled step imposes the
shell ~20 times, each a K-stage tensor-product interpolation per component,
and replicating that per rank put the 3-D cost case at 85% of the
uniform-fine wall; under `_impose_shell!` rank r runs the chain only for
components c ≡ r (mod np) and shares the thin shell ring through one
Allgatherv, which moves the same values instead of recomputing them (serial
results bit-identical) and brought the composite to 49%. Every rank writes
only the nodes it owns; tagging reduces its bounds globally; the fine patch
picks its process grid through `_amr_dims` over the rank range
[`_tile_owners`](#ownership-and-load-balance) assigns it.

**The cost case** (`bench/amr_cost.jl`, np = 8): a heavy-gas blob mixing
case on a 48³ root grid with a subcycled, regridding region covering a
sixth of the volume, against uniform 48³ and 142³ references at t = 1. On
∫Y(1−Y)dV the composite lands 5× closer to the fine answer than the coarse
run at 49% of the fine wall and 24% of its memory (342 s / 655 MiB vs
696 s / 2737 MiB; coarse 45 s / 179 MiB). Pointwise in-region error is the
wrong metric: coarse and composite both sit at max ≈ 0.19 against fine
there, the sub-cell displacement of a near-discontinuous interface.

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
path over the host halo buffers; the fold-pair Sendrecv stages the same way.
`device_mpi_direct(backend)` is the hook for direct device-pointer MPI and
defaults to host staging. Staging buffers allocate per exchange through
`similar` (device allocators pool); a keyed cache was reverted after its
`Any`-typed lookup put 33 dispatch sites into `compute_rhs!`'s jetcheck
report. `TRACK_DEVICE_TRANSFERS` tallies the staged copies.

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
cannot remove. The production design below is the event at which streams,
or the cross-patch batching that replaces them, become worth building.

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
2. **CPU-side pins without a GPU.** `FORCE_KA`, `FORCE_DEVICE_EXCHANGE`, and
   `DEVICE_SYNC` route ordinary arrays through the device paths, held
   bitwise by the serial and MPI suites. They cannot catch what only actual
   device storage exercises (lesson 3), so the real-GPU bench battery is
   part of the gate for every change to the device path.
3. **Manufactured-solution orders and physical-metric gates** where
   interfaces make bit-exactness unavailable by construction.

Bench scripts: `bench/amr_transfer.jl` (operator conditioning),
`bench/pointwise_ka.jl` (launcher acceptance), `bench/device_bringup.jl`
(kernel-level bring-up), `bench/device_solver.jl` (whole-solver device
battery), `bench/device_mpi.jl` (distributed device runs with transfer
accounting), `bench/amr_cost.jl` (the cost case), `bench/device_floors.jl`
(the cross-machine instrument: launch floors, stall watch, line-solve
matrix, TGV step table). The device scripts need an environment carrying
the device package; the workstation keeps one at
`~/.julia/dev/CompactLES_gpu_env`.

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

## Scope boundaries today

Configurations rejected at setup, and the reason:

- **Patched runs** (same-level `patch_grid`) reject folds (constraint 4),
  banded schemes (no interface closures), the `:d8` detector, stretching
  along the patched dimension, and an explicit `dims`. The layout tiles
  slabs along one dimension, so corner-coupled adjacency does not arise.
  Checkpoint/VTK output is single-patch.
- **Refined runs** require Cartesian metric, no stretching, no folds,
  tridiagonal schemes, `:delta4`, one region per level, and no same-level
  `patch_grid` alongside. `level_restriction = :filter` is serial-only.
  Each region must nest by `max(n_halo, LEVEL_BUFFER)` parent nodes inside
  the patches of the level above and span ≥ 4 parent nodes per active
  dimension; a tiled level's tiles are clipped to that margin at the
  domain edge and must still lie inside the parent tiles. Regridding is
  two-level. Tiled levels take the host backend.
- **Device runs** take a single patch per solver (the interface-record
  copies are host loops), reject `Nasa9Mixture` (no fixed-width device
  mirror), a pointwise NSCBC inflow `target` (host closure),
  `StepControl.floor_ratio > 0` and `dt_report` (host sweeps), and `:filter`
  restriction.
- The artificial-property sensors are built per patch with closed-edge
  clamping at interfaces; no gate has measured the effect.

## Production AMR: design

Each subsection replaces one of the structural assumptions listed under
[Status](#status). The order is the dependency order; the sequencing
section turns it into deliverables with gates.

### Level hierarchy

Delivered (sequencing item 1). `Level` holds a depth, the indices into
`solver.patches` of its patches, and one `LevelTransfer` per patch coupling
it to its parent (its own buffered box, shell ring, Hermite storage
`box_Q0` .. `box_dQ1`, and `coarse_index`); `solver.levels` is the chain,
root first. The flat `solver.patches` stays the drivers' iteration space,
so the property forwarding that serves a single-patch `Solver` is
unchanged, and the two-level results are bit-identical to before the
refactor (wave errors and step counts to the last digit). A refined
region is specified in its parent's node space, so a level-ℓ patch's region
offset is `3 · (parent offset + region offset)`, and nesting requires
`max(n_halo, LEVEL_BUFFER)` parent nodes of margin at every depth.

The step driver is recursive Berger–Oliger (`_advance_level!`): one step
of level ℓ, the extra RHS that saves the Hermite endpoint for its children,
three substeps of level ℓ+1 at dt/3, then restriction of ℓ+1 onto ℓ for
every ℓ above the root (the root's restriction stays in `run!`, after its
filter pass). Each level below the root filters itself at its own cadence,
substep index `3 · parent index + m`. The extra parent RHS recurs on every
level that has children; whether a cheaper dense output is worth it at
three or more levels is a measurement to make once a case demands it.

Still open here: the buffered box of a fine patch may cross the boundaries
of several parent patches once a level holds more than one, which the
replicated gather over the parent level handles by construction; and the
startup restriction compounds with depth, since the `compute_dt` lag that
gives cfl ≤ 0.2 at two levels widens at each level unless the rate is
re-evaluated after each substep (one `max_rate` reduction per substep,
which would turn a violated rate into a `SolverFailure` retry rather than
a silent overstep). That re-evaluation is not built.

### Tiles and adjacency

Delivered (sequencing item 2). The fine level of an implosion is a shell
of tiles, so a level is a set of fixed-size tiles on a lattice, not one box
and not slabs. `tile` is the lattice edge in parent nodes (0, the default,
keeps one patch per level; the floor is 3 from constraint 2); lattice cell
k spans parent nodes k·tile + 1 .. (k+1)·tile + 1, so abutting tiles share
their interface plane as root slabs do, and every tile face is either
fully shared with one neighbor or fully parent-fed. A static `refine` box
expands to the lattice cells that reach past its planes; cells at the
domain edge are clipped to the nesting margin and dropped below four
nodes. The lattice is global so that a regrid never changes a surviving
tile's region.

Same-level coupling reuses the root machinery: `build_interface_records`
over the level's tiles (each rank contributes a block of each tile it holds
to one Allgatherv, and the tag bound is checked against `MPI.tag_ub`),
records held on the `Level`, addressed in the level's own communicator,
point-to-point, and run by `sync_patches!` and,
inside the subcycled driver, after every stage of a level. Each face's
treatment follows the tile's neighbors: the shell imposition writes only
parent-fed faces (`LevelTransfer.imposed`, applied by `_in_shell`), the
restriction margin stands off parent-fed faces only, and the closure rows
are the same interface rows on every face, since both kinds read ghosts.
Ordering on every stage: same-level records after the update, then the
parent imposition at the head of the next stage, so a same-level
neighbor's data wins wherever both exist; and after a level receives its
children's restriction, its records run again before it is read (the
restricted nodes can sit beside a tile interface). The records are
dimension-phased (`_sync_level_records!`): the records of dimension d span
the transverse dimensions before d over their padded ranges, with the
per-patch halo exchange between phases. Two things depend on that. A node
shared by four tiles (eight in 3-D) reaches the mean of all its copies
only that way, since one flat pairwise pass reads values an earlier pair
has changed (a probe with copies 1, 2, 3, 4 ended at 2.23, 2.68, 2.41,
2.68); and the edge and corner ghosts of an interior tile, which no shell
writes and no face strip covers, are reached by the later phases' strips
through the earlier phases' ghosts, the argument `halo.jl` makes for rank
halos. A tile's buffered box may span several parent patches
(`coarse_indices`); the gathers and the restriction write-back run per
parent, and a child's buffered box must lie in the parents' own nodes,
their parent-fed planes eroded (imposed data is the class `RESTRICT_MARGIN`
exists to keep out of the closure rows). After a regrid a fresh tile takes
the planes it shares with surviving tiles from the survivors one-way
(`_seed_planes!`), and a tile about to leave restricts once more first.
Corner-coupled adjacency at level 0 does not arise; the root level stays
a slab layout or a single patch.

Not built, on review: a rate check per substep. Depth widens the number of
fine substeps one root rate measurement covers, so a startup or regrid
transient is a real unguarded case, but the three-level Sod gate runs at
cfl 0.4 without one, so it is not yet a compounding ceiling. When it is
built it should sit after the stage-1 RHS has refreshed the artificial
coefficients and before the update, limited at first to startup and
regrid steps, and it must return a failure to `run!` rather than throw:
the retry handling runs before `step!`, and an exception inside
`_advance_level!` would escape it. The recurring Hermite endpoint RHS is
about 1/15 of the RHS work at similarly weighted levels, not a fraction
that compounds with depth.

Measured: a tiled level costs nothing visible against the one-patch level
(1-D entropy wave, tile 8: 6.0e-10 against 6.2e-10 at N = 192, orders
3.95/3.91; a 2×2 tile nest in 2-D, corner included, 4.29e-8 against
4.27e-8), decomposed runs reproduce serial to round-off, and the tiled
regrid tracks the Sod shock through lattice cells that stay contiguous. On
an annular tag set in 2-D (`bench/amr_tiles.jl`, N = 192) the cover is 41%
of the bounding box at tile 6 and 47% at tile 12.

Costs, measured and not. Setup costs 0.06–0.12 s per tile in plan
construction, which argues for tile edges of 12 or more in 3-D. A rank
advances its tiles in sequence and nothing in the right-hand side's scratch
outlives the evaluation that filled it, so the gradients, the sensor fields
and their scratch, and the assembled fluxes live on an `RHSWorkspace`
(`src/patches.jl`) shared by every patch of equal padded extent; the
conserved state, the artificial coefficients, the primitives and the geometry
remain per tile, each for a reason recorded on that type. Sharing is keyed on
the padded extent rather than served as views of one largest allocation, so
every array a patch holds is the one it would have allocated alone, with its
own storage type and strides, and the host and device routing is untouched; a
lattice level's tiles carry equal extents by construction, so the key costs
nothing. Measured on the annular case (N = 192, tile 6, 208 tiles of 19²
plus the root): 103.0 MB over the patch set before, 60.3 MB after, a factor
of 1.71, with 0.207 MB of each tile's 0.398 MB now shared.

The warm per-step wall of that same case is 0.55 s before the pooling and
0.56 s after, at one rank on 16 threads, which is inside the run-to-run
spread; `bench/amr_tiles.jl` times steps after a warm-up and is the
instrument. That is the first clean reading for a tiled level, and it does
not settle the earlier open question, which was posed at a different
configuration: a rough count says the RHS work of the tiles, at 3 substeps
per root step over padded extents, should dominate and land near 2 s per
subcycled step for 40 tiles of 37² on a 96² root, where the cold measurement
was 8–10 s, so a real overhead may remain there. Separately,
any 2-D case at np = 8 on the workstation runs at ~7 s/step, one patch or
four tiles alike, against ~0.5 s at np = 4: a machine pathology of the
kind `CLUSTER.md` records for hybrid cores, not a tile cost, and the
reason the MPI suite's tiled check is bounded to ten steps. Under MPI each
tile is decomposed over its own rank range (sequencing item 3), so a tile
pays its owners' collective latency per imposition, and on a many-tile run
that is one rank; the box gathers and restriction that feed it remain
collective over the parent level. Whether the per-imposition latency still
binds after that reduction is a cluster measurement, not a workstation one,
and it has not been made.

Fixed tiles were chosen over Berger–Rigoutsos for three reasons: equal
extents let the device batch line solves across a level's tiles (below);
the lattice makes the face-closure decision binary; and the measured waste
on the target problems is the only thing that would justify the clustering
algorithm.

### Tagging and clustering

The tag is a union of criteria, each a per-point body over the parent
level's state:

- the relative δ⁴ρ tag that exists;
- the artificial-property `sensor` field, which is already computed,
  already smoothed, and is the scheme's own statement that a feature is
  under-resolved (a nonzero β* means the scheme has admitted it);
- `|∇Y|` for mixing layers, and vorticity magnitude for turbulence;
- a user closure `(patch, I) -> Bool` for problem-specific regions.

Derefinement uses hysteresis: a tag threshold and a lower untag threshold
(default ratio 2), and a minimum tile lifetime in regrid intervals, so a
tile at the edge of a feature does not flicker. Clustering is the set of
lattice tiles meeting any buffered tagged node, clamped to nesting
(delivered: `_tag_tiles`, one flag per lattice cell reduced over the
communicator); `tag_buffer` stays the pollution-decay figure and is
remeasured for C10.

A regrid at a tiled level is a set difference (delivered:
`_regrid_tiles!`): tiles that survive keep their arrays and state, taking
only a new id, faces and transfer; new tiles initialize by interpolation
from the parent; dropped tiles leave, their last restriction already on
the parent. No carry-over arises, since distinct lattice cells overlap in
a shared plane at most; that plane takes the neighbor's values at the
first averaging after the regrid, a one-node perturbation of interpolation
order. The one exception is a regrid that resizes the level's rank subset,
where the new tiles are decomposed over a different communicator and no array
survives: there a surviving tile's solution is gathered replicated over the
root communicator while its old decomposition is still live, then written
back onto the rebuilt tile, as the single-box regrid carries its overlap.
Lattice cells have equal extents, so this arises only when an edge-clipped
cell enters or leaves the set. The tag is still evaluated on the host; on the
device it becomes a
`pointwise!` body and an exact bounds reduction, so the regrid cadence
stops being a download.

### Ownership and load balance

Every rank holding every level does not survive a shell of a few hundred
tiles or a small hot spot on many ranks (constraint 2 via `_amr_dims`).

**Delivered: per-level subsets and per-tile owner ranges** (sequencing
item 3, second and third slices). A level is owned by a rank subset of its
parent's, a contiguous prefix of the parent level's communicator, so that the
subsets nest and a rank outside level ℓ is outside every level below it.
Within the subset each tile is owned by a contiguous rank range, and the
subset's size is the union of the ranges (`_tile_owners`):

- With at least as many ranks as tiles, each tile takes its own range,
  sized by tile volume through largest-remainder rounding, at least one rank
  and at most the count the tile admits under the 9-point scheme minimum
  (`_rank_counts`); ranks beyond the cap total hold nothing on the level.
  A one-tile level therefore reduces to `_level_ranks`, the sizing rule the
  second slice delivered.
- With more tiles than ranks, each tile takes one rank: the curve is cut into
  `np` runs of about equal weight, by the position of each tile's weight
  midpoint.

The ranges are laid along a Morton curve over the tile offsets
(`_sfc_order`), so a contiguous run of ranks holds a compact set of tiles and
most same-level exchange stays local; the tiles keep their raster order for
their ids, records and transfers, so nothing else changed. The weight is the
fine volume alone. The interface factor the design anticipates, since the
per-apply reduced-solve fence makes small tiles cost more than their volume,
is unmeasured and not applied.

A rank belongs to one group, the ranks sharing its range, and holds exactly
the tiles of that group (`TileGroup`, one communicator per group split off
the level's by `split_tile_comm`, and none when every tile spans the whole
level, which is every one-tile level, every serial run and every
configuration that existed before this slice). `Solver.patches` is the root
followed by the rank's tiles of each level; `Level.tiles` records which tile
each entry is, and a `LevelTransfer` carries the rank's local index of its
fine patch and of each parent patch under its box, zero where the rank holds
no piece. The alternative considered was placeholder `Patch` entries on
non-owning ranks, keeping the vector globally complete at the price of a
guard at every driver, every interface record and every transfer table, and
of a `Patch` with no communicator to decompose over; the local-index tables
make the placeholders unnecessary, and the same-level slab layout has always
had a rank-dependent patch vector.

Each collective is scoped to the rank set that owns the data it moves:

| collective | communicator | ranks entering |
|---|---|---|
| line solves and halo exchange | the tile's `Decomp` | the tile's owners |
| `_impose_shell!` ring Allgatherv | the tile's `Decomp` | the tile's owners |
| `_sync_level_records!`, `_seed_planes!` | the level's own, point-to-point | the level's owners |
| `_gather_box!`, `save_level_box!` | the parent level's own | the parent's owners |
| `_restrict_patch!` gather | the parent level's own | the parent's owners |
| `_tag_tiles`, `regrid!`'s carry-over | the root communicator | every rank |
| `max_rate`, `positivity_floors`, `WhenState` | the root communicator | every rank |

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

At a regrid the owner ranges are recomputed from the wanted tile set. A tile
whose region and range both survive keeps its arrays and its state; a tile
entering the set ahead of a survivor on the curve shifts the survivor's
range, and the survivor is then rebuilt on its new owners and takes its
evolved solution back through the replicated gather the box regrid has
always used for a moved region, one tile at a time. Each regrid frees the
tile groups and splits them afresh; a surviving tile's Cartesian
communicator is independent of the group it was split from. The
communicators a regrid discards are freed at the regrid, for the reason
`free_communicators!` records.

A tile owned by a proper subset is a different decomposition of it, so it
reproduces the every-rank answer to round-off rather than bitwise, the tier
the MPI suite already applies to decomposed patches; the suite measures 0 to
6e-15 on the tiled wave cases at np = 2, 4 and 8.

**Deferred.** Migration of a moved tile point-to-point, from its old owners'
blocks to its new owners', in place of the replicated gather, whose transient
memory is one tile's state per rank per moved tile. Rebalancing on a measured
max/mean of per-rank step wall, and the hysteresis that keeps rebalancing off
tile flicker, which needs the assignment to be stored rather than recomputed
so that a survivor can keep its owners against the deterministic partition.
The interface factor in the weight, once measured.

### Banded schemes

Two blockers, one tabulation and one remeasurement. `interface_closures`
needs a `BandedCompactScheme` method: for q = 2, two closure rows per end
whose LHS is restricted to the patch and whose RHS reads up to
`halfwidth + 2` ghosts, derived by Taylor matching as the tabulated wall
closures were, and gated by the same manufactured-solution interface study.
The pentadiagonal reduced solve then carries 2q interface unknowns per line,
which the device `DevicePlan` already mirrors. Separately, C10's LHS inverse
decays more slowly than C6's, so `LEVEL_BUFFER`, `RESTRICT_MARGIN`, and
`tag_buffer` must be remeasured with the `bench/amr_transfer.jl`
localization study before C10 runs refined. The filter stays C8, so the
minimum-extent constraint does not move.

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

### I/O, restart, diagnostics

A tile writes its `region` as a hyperslab the way a rank block does, so the
HDF5 checkpoint gains one group per level, each holding the level's fields
and a layout table of tile regions; restart reads the layout, rebuilds the
hierarchy on whatever rank count it is given, and reads hyperslabs. The
layout, the regrid spec, and the tag history that hysteresis needs are
state and go in the checkpoint. VTK output is one `.vtr` per tile under a
`.vtm` multiblock index, with covered coarse nodes blanked.

Every reduction that reports a physical quantity (`plane_profile`,
`mix_width`, the TGV energy history, the mixedness metric) must exclude
covered coarse nodes or it double-counts, and that is the single most
likely source of a silently wrong number in a multi-level run. Each patch
carries a `covered` mask, and every `quad_weight`/`cell_measure` reduction
routes through it.

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
- **`Nasa9Mixture` device mirror**: a flattened, fixed-width interval table
  with `Adapt.adapt_structure` plus the Newton inversion in the body. The
  last EOS off the device; mechanical.

## Sequencing

Each item names its gate. Nothing is built ahead of the item before it.

1. **Level hierarchy** (CPU). Delivered: `Level`, per-patch
   `LevelTransfer`, recursive `_advance_level!`. Gate held: single-patch
   and two-level paths bit-identical (`test/convergence.jl`,
   `test/level_tests.jl`, the MPI suite), and a three-level static nest on
   the entropy wave converges at the two-level orders in both stepping
   modes (`test/level_tests.jl`). Per-substep rate re-evaluation is
   deferred to the item that first needs it.
2. **Tiled fine level with full adjacency** (CPU). Delivered: lattice
   tiles, per-face treatment, corner ghosts through the records plus the
   halo exchange, set-difference regrid. Gate held: the 2-D tile nest with
   a corner at the one-patch level's error, the Sod moving-region gate
   reproduced with tiles, decomposed tiled runs at serial values, and an
   annular tag set in 2-D covered at 41–47% of its bounding box
   (`bench/amr_tiles.jl`); the 3-D shell awaits a memory-sized tile (the
   per-tile cost above) and is a bench measurement, not a testset.
3. **Ownership: per-level sub-communicators, per-tile owner ranges along a
   weighted space-filling curve, and the patch/workspace split.** Pulled
   ahead of tagging and I/O after
   review: the flat globally ordered `solver.patches`, the equal patch
   count per rank the block table assumes, and the full-communicator
   gather tables on every `LevelTransfer` are structural, and diagnostics
   or I/O built on them would be rebuilt once ranks hold different tile
   subsets. The patch/workspace split is delivered: `mu_art`/`beta_art`/
   `kappa_art`/`D_art` survive into the next `max_rate`, the primitives must
   be current on every tile for the prepared first stage, `du` persists
   across stages, and geometry is persistent, so `Patch` keeps those and the
   gradients, sensors, sensor scratch and assembled fluxes move to an
   `RHSWorkspace` shared by the rank's patches of equal padded extent
   (`src/patches.jl`; measurement under
   [Tiles and adjacency](#tiles-and-adjacency)). Gate held: the whole serial
   and MPI gate reproduces, `test/convergence.jl` bit-identically and
   `test/validation.jl` to every printed digit, with no change to the
   `bench/jetcheck.jl` reports or to the per-step allocation
   `bench/audit.jl` measures; and the sharing itself is guarded in
   `test/level_tests.jl`, since no other figure in the gate moves if the
   pooling silently stops. Two sensor fields are pooled despite a reader
   outside the RHS: `scalar_field(solver, :strain_mag)` and `:sensor` take a
   `Solver` and read through the single-patch property forwarding, so they
   are reachable only where the pool is one patch's own set, and a
   multi-patch output path will have to take storage of its own.

   Per-level sub-communicators are delivered as the second slice:
   `_level_ranks` sizes each level's subset, `split_level_comm` and
   `free_level_comm!` create and release it, and the level's own collectives
   move onto it while the cross-level gathers stay on the parent's; the
   communicator table and the structural decision behind the prefix subsets
   are under [Ownership and load balance](#ownership-and-load-balance). Gate
   held: `test/convergence.jl`, `test/validation.jl`, `bench/jetcheck.jl` and
   `bench/audit.jl` reproducing to every printed digit and `test/runtests.jl`
   to every logged value, since every level of every existing configuration
   owns the whole communicator and takes no split; the serial count moved
   119 → 120 for a testset pinning the sizing rule, and `test/mpi_tests.jl`
   122 → 142 at np = 2/4/8 for twenty checks holding a subset-owned level to
   the serial wave error, pinning the subset size, holding a rank outside a
   subset to the same step sequence and trigger firing, and reproducing the
   region track of a regrid that resizes the subset and frees the
   communicator it replaces. The smallest of those checks refines a region
   of four coarse nodes, which `_amr_dims` cannot split over two ranks and
   which runs with the level owned by one.

   Per-tile owner ranges along a Morton curve, weighted by tile volume, are
   the third slice: `_tile_owners` lays out the ranges, `split_tile_comm`
   gives each rank its group's communicator, the block tables and the
   transfer tables carry local indices with zero for a patch the rank holds
   no piece of, and a regrid keeps a tile in place when its region and its
   range both survive and otherwise rebuilds it, carrying a survivor's
   solution through the replicated gather; the design is under
   [Ownership and load balance](#ownership-and-load-balance). Gate held:
   `test/convergence.jl`, `bench/jetcheck.jl` and `bench/audit.jl`
   reproducing to every printed digit, `test/validation.jl` to every printed
   digit and `bench/amr_tiles.jl` to its header figures, since every
   existing configuration has every tile spanning its level and takes no
   new communicator; the serial count moved 120 → 121 for the docs
   cross-reference guard, with the partition rule pinned in the existing
   subset testset, and `test/mpi_tests.jl` 142 → 154 at np = 2/4/8 for the
   tiled level's ownership shape at each rank count (two tiles per rank, one
   rank per tile, two ranks per tile), a twelve-tile level with more tiles
   than ranks at the serial wave error, and a tiled Sod regrid whose owner
   ranges move, tracking the serial tile set and time. The tiled wave cases
   reproduce serial to 0–6e-15.

   Point-to-point migration, rebalancing on measured per-rank step wall with
   its hysteresis, and the interface factor in the weight remain unbuilt;
   the covered-mask data model is still to be defined here even if the
   diagnostic conversion follows. Gate for those: the implosion case at a
   rank count for which the replicated carry is measurably the regrid's
   cost; per-rank step wall spread measured before and after a rebalance.
4. **Tag criteria and hysteresis; covered masks in every diagnostic.** Gate:
   the mixing-layer cost case reproduced through the sensor-based tag; the
   TGV energy history on a refined run equal to the single-level history
   where the refined region is inactive.
5. **Multi-level HDF5 and VTK, restart of the hierarchy.** Gate: restart on
   a different rank count continues bit-identically for the delivered
   serial-restart cases and to round-off under MPI.
6. **Banded interface closures and remeasured buffers.** Gate: the
   two-conforming-patch orders and reflection amplitude reproduced at C10;
   the localization study at C10 setting the buffers.
7. **Device: staged interface records, device transfer chain and tagging,
   batched cross-tile launches.** Gate: bitwise equality against the CPU
   hierarchy over full refined runs; the implosion case's device step
   measured against its CPU step on the workstation and on rzadams.
8. **Numerics debts on the target problems**: conservation drift on the
   long mixing-layer run, the sensor at a level boundary under a crossing
   shock, filter rows at the shell. Each is a measurement first and a
   change only if the measurement demands one.

Items 1–3 make a run possible, 4 makes it usable, 5 and 7 make it fast, 6
is independent and lands whenever a case wants C10.
