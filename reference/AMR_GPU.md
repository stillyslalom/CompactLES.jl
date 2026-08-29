# CompactLES — Patch AMR and the GPU backend: design, lessons, roadmap

This document describes the patch-based adaptive mesh refinement (AMR) and
the GPU backend as delivered, records the measurements and lessons that
shaped them, and lays out the remaining work. It replaces the staged
implementation plan that produced the capability; the stage-by-stage gate
record lives in the git history (one commit per delivered stage or gate) and
in `HISTORY.md`. Prerequisite reading: `DESIGN.md` for the compact
operators, the distributed line solve, and the coordinate folds.

Everything here was gated on measurement. Where a number appears, a bench
script or testset produced it and usually still guards it; where a design
choice contradicts the original plan, the measurement that forced the change
is stated next to it.

`HISTORY.md` and the git history cite the old plan's stage and gate names.
They map onto this document as follows: Stage 1 (transfer operators)
→ [The transfer operators](#the-transfer-operators); Stage 2 (patch
abstraction, with GPU G0 storage generalization) → [Patches and same-level
interfaces](#patches-and-same-level-interfaces); Stages 3–4 (two-level
refinement, subcycling, tagging, regridding) →
[Two-level refinement](#two-level-refinement); G1 (pointwise kernels), G2
(device line solves), G3a (residency), G3b (device communication), G3c
(device AMR, plus the distributed level transfer), G3d (launch policy), and
G4a (Float32) → the corresponding subsections of
[The device backend](#the-device-backend) and
[Distributing the level transfer](#distributing-the-level-transfer); G4b
(mixed precision) → [Roadmap](#roadmap) item 2.

## Contents

1. [What exists](#what-exists)
2. [Why one architecture served both](#why-one-architecture-served-both)
3. [Constraints](#constraints)
4. [The transfer operators](#the-transfer-operators)
5. [Patches and same-level interfaces](#patches-and-same-level-interfaces)
6. [Two-level refinement](#two-level-refinement)
7. [Distributing the level transfer](#distributing-the-level-transfer)
8. [The device backend](#the-device-backend)
9. [Verification](#verification)
10. [Performance summary](#performance-summary)
11. [Lessons that were not obvious](#lessons-that-were-not-obvious)
12. [Scope boundaries](#scope-boundaries)
13. [Roadmap](#roadmap)

## What exists

One solver runs on four axes of configuration, combinable except where
[Scope boundaries](#scope-boundaries) records an exclusion:

- **Same-level patches**: the domain tiles into conforming slab patches, the
  rank set partitioned over them, coupled by ghost exchange, interface
  closure rows, and shared-plane averaging (`src/patches.jl`).
- **Two-level refinement**: one refined region at ratio 3, static or moving
  under sensor-driven tagging and regridding, at a global timestep or
  Berger–Oliger subcycled (`src/levels.jl`, `src/regrid.jl`).
- **Distribution**: both levels decompose over the whole rank set; the level
  coupling gathers replicated region data and distributes its interpolation
  chains by conserved component.
- **Storage backend**: `CPUBackend` (ordinary `Array`s, `@threaded` loops)
  or `DeviceBackend(ka)` wrapping any KernelAbstractions backend. A
  device-resident solver (decomposed, refined, regridding, Float64 or
  Float32) reproduces the CPU solver bitwise over full runs. The
  workstation's RX 6800 XT through AMDGPU.jl is the measured bring-up
  target; the deployment target is rzadams / El Capitan-class hardware
  (see [Performance summary](#performance-summary)).

The pointwise physics is written once as per-point bodies launched through
`pointwise!` (`src/pointwise.jl`); the compact line solves run through host
plans or their device mirrors (`src/lines_device.jl`) behind one
`apply_along!` entry point. There is no second code path for the device
beyond the launchers themselves.

## Why one architecture served both

Patch AMR replaces the one-global-array-per-field assumption with many
independently sized blocks, each with its own line plans and ghost regions.
A GPU port fits that shape: a patch is a natural launch unit with a
bounded working set. The two capabilities therefore shared one refactor:
the patch abstraction carried the storage-type generalization (fields typed
`A <: AbstractArray{T,3}`, allocation through a backend object) in the same
change, and every later device stage slotted into interfaces that refactor had
cut. Sequencing the AMR-shaped refactor first and the kernels second
avoided doing the decomposition work twice. Miranda's GPU implementation of
patch AMR was the evidence the shared architecture works; this repository is
now its own evidence.

## Constraints

Each of these is structural or measured; none is a preference. They bound
the design and still bind its extensions.

1. **No line cut mid-solve.** A compact operator couples an entire grid line
   through a banded LHS. A patch is therefore a logically rectangular block
   of uniform resolution; the compact solve runs over the patch with ghost
   layers filled by same-level copy or coarse-level interpolation. Oct-tree
   cell-by-cell refinement is excluded (Non-goals in `ROADMAP.md`).
2. **Minimum extent 9 per resolved dimension.** The C8 filter closure is the
   binding scheme (`plan_direction` enforces it). This bounds the minimum
   patch size, the tagging granularity, and, once a patch decomposes, the
   rank count a small region can absorb (a 1-D fine patch needs ≥ 9·np
   nodes along its only active dimension; `_amr_dims` errors with the
   numbers before a regrid can die mid-run).
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
   style of `test/convergence.jl`. The device backend supports a stronger
   oracle than the plan expected; see [Verification](#verification).
6. **The grid stays node-centered**, half-offset only where a fold requires
   it. The SBP–SAT multiblock literature is node-coincident too, so a
   cell-centered migration was never a prerequisite.
7. **Error localization is favorable.** The inverse of a compact LHS decays
   geometrically off the diagonal (≈ α^|i−j|, α = 1/3 for C6), so pollution
   injected at an inexact interface decays about 3× per point into the
   patch. Measured in situ at ≈ 3.4× per point (shock round-trip pollution,
   `bench/amr_transfer.jl`); the default 4-coarse-cell tagging buffer drops
   interface pollution two orders of magnitude and is sized from this
   number.

## The transfer operators

`src/transfer.jl`; measurements in `bench/amr_transfer.jl`, guarded by
serial testsets and a multi-rank section.

**Source.** Pyranda carries Miranda's Fortran kernels; the level-transfer
operators sit in `pyranda/parcop/stencils.f90` (`cfamrcf`/`cfamrfc`, inside
an `#if 0` block: the design, not a live implementation). Level transfer is
an invertible compact filter pair whose transfer function is a Gaussian of
width 3Δx:

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
tabulated boundary closures preserve that gain. Four closure variants exist
per end (odd-symmetric, one-sided, even-symmetric, extended-data); the
extended-data variant is tabulated identically to the one-sided one, "to
maintain invertibility" per the source comments, which set the precedent for
every interface closure in this code: the LHS couples no ghost unknown even
where ghost data is available.

**Mapping onto this code.** Restriction is a `CompactScheme` (tridiagonal
LHS, Gaussian RHS); prolongation a `BandedCompactScheme` with q = 2,
interior rows normalized to a unit diagonal. Both bind to a dimension with
the existing `plan_direction`, which supplies the distributed spike solve
and the fold parity variants without further code: the ± symmetric closure
variants are the `lo_fold`/`hi_fold` ghost-folding algebra (measured ≤ 1e-13
against parity-extended full lines, both schemes, both signs).

**The 3:1 sampling convention**, pinned numerically because the public
kernels do not fully specify it: restriction filters the fine line and takes
the coincident nodes (fine node 3m − 2 ↔ coarse node m); prolongation
injects coarse values onto coincident fine nodes, fills the intermediate
nodes by Lagrange interpolation (order 4/6/8, default 6), then deconvolves.
Measured properties:

- The pair round-trips at 1.6e-15 closed / 2.7e-15 periodic.
- Restriction is a **left inverse** of prolongation: coarse → fine → coarse
  is exact (8.9e-16) for arbitrary data, which keeps levels consistent
  without refluxing. Fine → coarse → fine converges at the interpolation
  order (measured 3.97 / 5.93 / 7.97); the subsampled content is
  unrecoverable by construction.
- A constant survives restriction to the last bit, prolongation to 13 ULPs.
- Within 6 points of a closed end the round-trip converges at ≈ 3 (the
  closure order) against 6 in the interior.

**Conditioning.** Prolongation is deconvolution of a Gaussian: gain 1.51 at
the coarse Nyquist, 20.24 at the fine Nyquist; the finite closure operator
has condition number 33. That is benign while the coarse field carries no
content above the coarse Nyquist and dangerous when something injects
grid-scale content near a transfer boundary. Measured in situ, the risk is
small: the smoothed δ⁴ sensor of a 2h shock profile round-trips at 1.03–1.13
(after the smoother there is no fine-Nyquist content left to amplify), the
state field undershoots ≤ 3% of ambient at the shock itself, and pollution
decays ≈ 3.4× per point outside the shock footprint. An explicit Gaussian
pass ahead of restriction (Miranda's `c4ff3` role) cuts the shock undershoot
2.8× but raises total round-trip error; it is not a default, and remains the
tool to reach for if regridding onto captured shocks proves
positivity-limited.

**SBP–SAT remains the fallback of last resort** and the theory guiding
interface placement: Carpenter, Gottlieb & Abarbanel (1993); Abarbanel &
Chertock (2000); Mattsson & Rydin (2022) on implicit SBP in a banded-norm
framework, the Padé-class reference; Nissen et al. (2015) on block-adaptive
grids; Almquist & Dunham (2018) on order-preserving interpolation at
non-conforming interfaces. The documented costs (interface-order loss,
penalty terms above 4th order, a rewrite of the closure cascade and every
convergence guard) were never justified: no measurement forced it. SAT
penalties act on the RHS in the same way as the NSCBC corrections, so an eventual
SBP–SAT implementation could subsume that machinery.

## Patches and same-level interfaces

`src/patches.jl`. A `Patch` carries a block's place in the global grid
(`region`), its communicator and `Decomp`, its operator plans and folds, and
every field array, typed `A <: AbstractArray{T,3}` by the storage backend.
`Solver` holds the physics configuration plus this rank's patches; a
single-patch solver forwards patch-owned property names to its sole patch
through `Base.getproperty`, which kept the refactor behavior-preserving: the
single-patch code path is the old code path, and the hard gate held it to
bit-identical convergence output. Routines between
the step drivers and the arrays take a `SolverLike` (single-patch `Solver`
or `PatchSolver`), so one body serves both.

Rank assignment partitions the rank set over patches proportionally to
volume (one `MPI.Comm_split`); a serial run advances every patch in
sequence. Interfaces are node-coincident, and coupling is three mechanisms
between RK stages, never inside a per-patch RHS (`sync_patches!`):
shared-plane averaging, ghost refill, then each patch's own halo exchange.
Because a rank advances several patches sequentially, no cross-patch
communication may sit inside `compute_rhs!`.

The interface closure rows follow the Miranda precedent: gradient and filter
plans take `interface_closures` rows whose LHS couples no ghost unknown
while the RHS reads exchanged ghost layers (`interface_rhs = :extended`, the
default) or stays one-sided (`:onesided`). The flux divergence always keeps
the scheme's own one-sided closures (`Patch.div_plans`): ghost *fluxes*
would require a second communication phase inside the RHS, which the
sequential-patch order forbids. `gcl_cotr!` routes through the same
divergence plans so the discrete GCL cancellation is preserved.

Measured at the two-conforming-patch gates: entropy-wave order 3.1–3.5
across the interface (the divergence's one-sided rows binding); acoustic
pulse reflected amplitude 2.3e-3 of incident at 192 points, converging at
≈ 5th order; conservation drift 1.2e-8 relative over a long periodic run
against 4.5e-15 single-patch; the drift term exists as predicted, and no
refluxing machinery is warranted. Rank partitioning reproduces the serial
two-patch answer bitwise at one rank per patch; once a patch itself
decomposes, closed lines take the spike path where the serial patch ran a
plain sweep, and agreement is round-off (3.1e-15 at np = 4 against a 9.5e-8
signal) with identical step counts.

## Two-level refinement

`src/levels.jl`, `src/regrid.jl`. One level-1 patch covers a
`refine::BlockRegion` of the root grid at ratio 3, node-centered (m coarse
nodes ↔ 3m − 2 fine nodes), joining `solver.patches` so the same drivers
advance it and the shared `max_rate` reduction supplies the timestep.

**The live coupling is interpolation and injection, not the invertible
pair.** This is the largest numerics finding of the AMR track. The pair's
contract is that prolongation input is samples of the *filtered* field —
which restricted data is, and the live coarse solution is not. Deconvolving
point samples sharpens data that was never smoothed; filtering on the way
down writes an attenuated representation into a field of point samples; both
are O(h²) against the solution, and the manufactured-solution gate measured
order 1.3–1.7 through the pair. The default coupling is therefore the
point-sample halves of the same machinery, order-6 Lagrange interpolation
up (`interpolate!`) and coincident-node injection down (a sampled
`gather_region!`), which measures order 3.46/3.64 with errors three decades
lower.
`level_restriction = :filter` keeps the anti-aliasing pair selectable, and
regridding still initializes *new* fine cells by interpolation for the same
reason: the coarse data are point samples wherever they are all there is.

**Coupling schedule.** After every RK stage, the fine patch's ghost ring
*and* boundary-plane nodes are overwritten from the coarse state
interpolated over a box extending `LEVEL_BUFFER = 4` coarse nodes beyond the
region. This is Dirichlet forcing by the coarse solution, and it removes
the inter-level drift mode without an averaging step. After every completed
step
the fine state writes back onto the covered coarse region, holding
`RESTRICT_MARGIN = 2` coarse nodes off the boundary. The margin is not
optional: restricting all the way to the coarse–fine boundary closes an
amplifying loop through the fine solution's least-accurate nodes (its
imposed shell) into the closure rows that feed the next shell (measured
gain ≈ 2 per step, flat with the margin in place).

**Subcycling** (`subcycle = true`) is Berger–Oliger: one coarse step with
the fine level frozen, then three fine steps of dt/3, the fine shell imposed
at every fine stage time from a cubic Hermite reconstruction of the coarse
trajectory on the buffered box (values and RHS at both step ends; O(dt⁴),
matching the integrator, since LSRK54 has no free dense output). The t^{n+1}
endpoint RHS is *not* free, since the next step's stage-1 RHS arrives too
late and restriction invalidates a cached one, so it costs one extra coarse RHS
per step. The dt reduction divides each patch's rate by 3^level, so the step
is coarse-limited. Each level filters its own state at its own step cadence,
as an unrefined run of either level would, which is dt-consistent whenever
`filter_cfl > 0`. Measured: the entropy-wave orders are unchanged
by subcycling (the Hermite boundary data does not bind) at a third of the
steps, and the subcycled Sod gate *improves* on the global-dt one
(ahead-of-shock noise 5.7e-11 vs 6.4e-10; mass drift 9.8e-5 vs 1.36e-4).

**Tagging and regridding** (`regrid_interval = K`). The tag is the relative
undivided fourth difference of the mixture density, Σ_d |δ⁴_d ρ|/ρ >
`tag_threshold`, read from Q directly so it needs no primitives pass;
buffered by `tag_buffer` (default 4, the pollution-decay figure) and clamped
to the nesting margin. Clustering is the bounding box of the tagged set —
one moving region, matching the one-fine-patch base. A regrid rebuilds the
fine patch and `LevelTransfer` from schemes retained on a `RegridSpec`,
initializes the new state by interpolation, and copies surviving fine data
across the region overlap on the coincident lattice, one node off both
boundary planes. The retry savepoint refreshes at a regrid, since a snapshot
with the old extents cannot restore onto the new layout.

Measured on moving-region gates. Sod at N = 201 coarse against a 601-node
uniform-fine reference: composite density error 2.8e-3 where uniform-coarse
gives 7.3e-2 (26×), with fine resolution over a third of the domain.
Shu–Osher: 10× better in L∞ over the wave train, 6.7× in L1, at 2497 coarse
steps against the reference's 4662. The 3-D cost case is under
[Distributing the level transfer](#distributing-the-level-transfer).

**Startup and rollback traps**, both paid for in debugging time. Subcycling
widens `compute_dt`'s one-step lag to three fine substeps on one rate
measurement, so a discontinuity inside the refined region at t = 0 needs
cfl ≤ 0.2 or `StepControl(retries)`, the same startup restriction
`reference/CALIBRATION.md` documents for strong shocks. And a non-finite
trajectory leaves NaN in two places a state restore does not touch: the
artificial coefficient arrays (read by `max_rate` before the retry's first
RHS) and the low-storage `du` accumulator (`RKA[1] = 0` cannot forget NaN,
since 0.0·NaN = NaN). Both resets are gated on `:nonfinite` failures only —
after a *finite* failure the stale coefficients are a measured part of the
recovery.

## Distributing the level transfer

Both levels decompose over the whole rank set; every rank holds a block of
each. The coupling runs on a replicated-data, distributed-work split:

- **Data replicates.** `gather_region!` assembles a node region of a
  distributed field on every rank with one Allgatherv; the buffered coarse
  box gathers per shell imposition, the `:inject` restriction is a sampled
  gather of the coincident nodes (identical values to the per-dimension
  subsampling chain it replaced), and a regrid gathers the surviving fine
  state
  once at the regrid cadence. Regions are small by construction, so
  replication is the right first distribution: no halo machinery
  enters the transfer, no consistency questions arise, and the serial path
  is the same code on one-rank communicators.
- **The interpolation chains distribute by conserved component**, and that
  split was forced by measurement. The first cut ran the chains
  replicated per rank; on the 3-D cost case at np = 8 that put the composite
  at 85% of the uniform-fine wall: a subcycled step imposes the shell ~20
  times (five coarse stages plus every fine stage's Hermite shell), each a
  K-stage tensor-product interpolation over the box per component, and
  replication multiplied all of it by the rank count. Under
  `_impose_shell!`, rank r now runs the chain only for components
  c ≡ r (mod np) and shares the thin shell ring (two slabs of thickness
  pad+1 per active dimension) through one Allgatherv per imposition. The
  same values move and are not recomputed, so serial results are
  bit-identical, and the composite fell to 49% of the fine wall.
- Every rank writes only the shell or covered nodes it owns, by patch-global
  index; tagging reduces its bounds globally so every rank derives the
  identical region; the fine patch picks its process grid through
  `_amr_dims` (largest smallest-block factorization subject to the 9-point
  minimum).

**The cost demonstration** (`bench/amr_cost.jl`, np = 8 on the workstation):
a heavy-gas blob mixing case on a 48³ root grid with a subcycled, regridding
refined region, against uniform 48³ and uniform 142³ references, t = 1. On
the physical mixing metric ∫Y(1−Y)dV the composite lands 5× closer to the
fine answer than the coarse run (+1.4e-3 vs +6.9e-3, 0.4% vs 1.9% relative)
at 49% of the fine wall and 24% of its memory (342 s / 655 MiB vs 696 s /
2737 MiB; coarse 45 s / 179 MiB). Two reading rules the case established:
pointwise in-region error is the wrong metric (both coarse and composite sit
at max ≈ 0.19 against fine there, the sub-cell displacement of a
near-discontinuous interface, not a mixing error; outside the region the
composite is 2.4× closer to fine), and the advantage widens as the refined
fraction shrinks; the case's region covers a sixth of the volume, a
conservative configuration for the claim.

## The device backend

Architecture: KernelAbstractions.jl, one codebase, CPU and GPU backends,
the XCALibre/Oceananigans pattern. KA is the one hard dependency a portable
GPU path cannot avoid; device packages (AMDGPU.jl, CUDA.jl) arrive only
through the user's environment. `DeviceBackend(ka)` selects device storage
through the same `field(backend, decomp)` / `allocate_state` interface
`CPUBackend` uses.

### Pointwise kernels

Every pointwise phase is one shared `@inline` per-point `_point!` body
beside its driver, launched through `pointwise!`: `Array` storage takes the
`@threaded` loop (bit-identical to the pre-kernel solver), any other storage
a KernelAbstractions kernel on `get_backend` of a representative array. The
routing resolves statically; `FORCE_KA` reroutes ordinary arrays through the
kernel path for the bitwise comparisons. The CPU keeps `@threaded` because
the acceptance measurement went against KA-CPU: 2.8× (flux assembly) to
40–50× (RK update) slower at 64³, a per-launch task-spawn cost with no work
threshold, the same profile `threading.jl` documents for unconditional
`@threads`; on small 2-D cases KA-CPU is *faster*, so the deficit is launch
policy, not codegen.

Kernel-argument adaptation is the central design concern. A `Vector{A}` or
`Matrix{A}` kernel argument does not error on device; it **hangs** in
adaptation, so the field collections reach bodies as
`FieldVector`/`FieldMatrix`: zero-cost host wrappers, built once per patch,
that adapt to isbits `NTuple` mirrors only at launch. Holding the tuple form
on the host was measured at 3× on `assemble_fluxes!` (runtime tuple indexing
in the species loops); the wrapper exists to avoid that. The gas-model
EOS objects adapt to coefficient mirrors the same way
(`IdealMixtureCoeffs`, `StiffenedGasCoeffs`); `ConservedState` adapts to
itself around the device-side array. Bodies take no `::Type` argument, since
a `Type` inside `pointwise!`'s Vararg defeats specialization and turns the
body call into per-point runtime dispatch, measured at 9× on the flux
assembly on *both* paths. A splatted kernel-argument tuple longer than 32
elements lowers through the dynamic apply and is an `InvalidIRError` on
device; the budget is 32 launcher arguments, or 35 declared once `i, j, k`
are counted, and the NSCBC bodies carry their scalars in small tuples to
stay within it. `test/device_tests.jl` asserts it for every body.

### Line solves

`device_plan` mirrors a host `DirPlan`/`BandPlan` onto a KA backend as a
`DevicePlan`: fill, elimination sweeps (one thread per line), spike
correction, and scatter as kernels in a (lines × n) layout for every
dimension, so consecutive threads touch consecutive memory at each sweep
position. The reduced 2qP × 2qP interface stage stays host-side: a pack
kernel writes the 2q interface values per line, one device-to-host copy
feeds the same `_reduced_solve!` (Allgather + prefactorized `ldiv!`) the
host layouts share, and one copy returns the corrections, so collective
ordering is identical to the host path. The kernels carry a `colwise` switch
because the two *host* layouts themselves disagree in the banded solve (the
x sweep divides by the diagonal where the transposed y/z sweeps multiply by
its inverse, and they accumulate the spike correction differently);
mirroring each dimension's own convention keeps the device output bitwise
per dimension.

### Residency

`Solver(backend = DeviceBackend(ka))` builds a fully device-resident solver:
construction converts every plan (fold parity plans included) through
`backend_plan`, allocates all fields, pair/ring scratch, and state on the
backend, and every phase between the step drivers and the arrays is a
launchable body, a `DevicePlan` solve, or an exact storage-level reduction.
`max_rate` evaluates the CFL rate through a pointwise body into scratch and
reduces with the storage's own `maximum`/`minimum`; these are exact, so the
result matches the fused CPU loop bitwise. Three operations are
host-staged, following the
I/O rule that gathers are explicit: geometry fills host mirrors once at setup
and uploads;
`initialize!` and `DirichletBC` evaluate their host closures into staging
blocks (setup-time, and one plane upload per stage respectively); tagging
downloads the coarse block at the regrid cadence.

### Communication

A distributed device patch stages every message through the backend: a
broadcast pack of the strided slab into a contiguous device buffer, one
contiguous device↔host copy per message, and the unchanged MPI path over the
host halo buffers; the fold-pair whole-block Sendrecv stages the same way
(`sendrecv_block!`), with tags and phase order identical to the direct host
path. `device_mpi_direct(backend)` is the capability hook for direct
device-pointer MPI and defaults to host staging; it is a runtime property of
the backend and MPI library together, and no measured stack (MS-MPI on this
workstation included) reports the capability. The staging buffers allocate
per exchange through `similar`, whose result type is inferable from the
field; a keyed cache was tried and reverted after its `Any`-typed lookup
put 33 runtime-dispatch sites into `compute_rhs!`'s jetcheck report (the
staged branch sits behind a runtime `Ref` read and is always inferred into
the RHS call graph). Device allocators pool, so the per-exchange cost is a
pool hit. `TRACK_DEVICE_TRANSFERS` tallies the staged halo/pair copies and
the reduced-interface copies separately.

### Launch policy

Kernels queue on the backend's one in-order stream without a host
synchronize per launch. The only unconditional fence is the one the
algorithm requires, before the host reads the packed interface values of
the reduced solve, once per `apply_along!`; every other host interaction is
a synchronous device↔host copy or a reduction, which drains the queue
itself. `DEVICE_SYNC[] = true` restores synchronize-per-launch and is the
correctness fallback, pinned by a testset and exposed as `sync=` in
`bench/device_solver.jl` so the two modes measure one flag apart.

Per-patch streams were planned and are not implemented, on the
measurement: supported device configurations hold one patch per rank or a
sequentially coupled coarse/fine pair, and the remaining gap to CPU after
the deferred policy is bound by the per-apply reduced-solve fence, a host
serialization a second stream cannot remove. Stream machinery would add
scheduling complexity to overlap kernels that an in-order queue issues back
to back. Revisit when same-level multi-patch reaches device (its patches are
independent within a stage) or when the reduced solve moves on-device.

### Precision

Uniform Float32 is a supported opt-in end to end: typed schemes, EOS
coefficients, transport and artificial controls through `Problem`/
`Numerics`, with literals typed against the state's eltype and
`positive_floor` replacing raw 1e-300 guards. The CPU measurement (64³ TGV,
t = 10): identical peak dissipation and timing to the printed precision,
2.00× smaller solver/state/RK footprint, 1.10× wall, and mean-density
drift 1.4e-4 against 7.5e-13; the drift is the reason Float32 is not the CPU
default. On device both precision histories reproduce the CPU numbers to
every printed digit, and Float32 runs 1.25× the Float64 device rate under
the deferred launch policy, above the CPU's 1.10× but well short of 2×,
because the step is bounded by launch submission and the reduced-solve
fences, not by arithmetic (the G2 FP64/FP32 kernel measurement:
1.03–1.10× on launch-bound phases, 1.73× on the compute-heaviest body, on
hardware whose vector FP64 is 1/16 the FP32 rate). Two open literal-audit
items: the Ducros epsilon is the Float64 value converted at use (safe,
representable, but not separately measured for Float32), and the
`sine_cluster` stretch closures evaluate in Float64 internally
(setup-only, converts at store).

## Verification

The oracle hierarchy, strongest first:

1. **Bitwise equality against the CPU solver, over full runs.** The original
   plan expected device bit-exactness to be lost to reassociation. It was
   not: every phase is either per-point independent, a line sweep mirrored
   operation-for-operation, an element copy, or an exact max/min reduction,
   so the whole composition is bitwise, and that equality is the cheapest
   and sharpest gate the device track has. Every device stage was accepted on
   `max |device − cpu| = 0` over full runs: smooth periodic and closed
   cases, NSCBC, folds, freestream/GCL, a 578-step Sod, refined static and
   regridding runs, 64³ TGV histories in both precisions, and distributed
   runs at np = 2/4/8, in both launch-policy modes.
2. **CPU-side pins without a GPU.** `FORCE_KA` routes ordinary arrays
   through the kernel path, `FORCE_DEVICE_EXCHANGE` through the staged
   exchange, and `DEVICE_SYNC` toggles the launch policy; the serial and MPI
   suites hold all of them bitwise against the default paths. They cannot
   catch anything that only actual device storage exercises (see the lessons
   below), so the real-GPU bench battery is mandatory for every change to
   the device path.
3. **Manufactured-solution orders and physical-metric gates** where
   interfaces make bit-exactness unavailable by construction: interface and
   two-level orders, reflection amplitudes, conservation drifts, the
   uniform-fine comparisons, the mixedness metric of the cost case.

Bench scripts: `bench/amr_transfer.jl` (operator conditioning and
round-trips), `bench/pointwise_ka.jl` (launcher acceptance),
`bench/device_bringup.jl` (kernel-level device bring-up; its header
documents the collection-argument hang), `bench/device_solver.jl` (the
whole-solver device battery and launch-policy timing),
`bench/device_mpi.jl` (distributed device runs with transfer accounting),
`bench/amr_cost.jl` (the refined-vs-uniform cost case). The device scripts
need an environment carrying CompactLES and the device package; the
workstation keeps one at `~/.julia/dev/CompactLES_gpu_env`.

## Performance summary

The performance target is an LLNL rzadams / El Capitan-class machine
(MI300A APUs, GPU-aware Cray MPICH), not the development workstation.
Everything below was measured on the workstation's RX 6800 XT (gfx1030,
AMDGPU.jl on Windows/HIP) against its own CPU, and is to be read as
evidence that the structural pitfalls (launch floors, transfer stalls,
adaptation hangs) are gone, never as a performance claim about the target.
The hardware differs in performance-relevant ways: consumer RDNA2 runs vector FP64 at
1/16 the FP32 rate where MI300A runs it at full rate; the APU's unified
memory removes the discrete-VRAM staging economics every device↔host
number below is priced in; and a GPU-aware MPI flips `device_mpi_direct`
from its host-staging default. Wall targets, the G4b precision policy, and
the reduced-solve question are rzadams measurements to make, not
extrapolations from this table. `bench/device_floors.jl` is the instrument
for those sessions: the same host-sanity probe, launch-floor timings,
stall watch, line-solve matrix, and TGV step table run unchanged on every
machine, allowing machine-specific behavior to be distinguished from an
implementation defect.

The first rzadams sessions (2026-08-19, `bench/logs/rzadams_20260819*.txt`)
established the baseline and one open issue. Baseline, per MI300A under
`flux run -N1 -n4 --exclusive`: kernel submission 10 µs, launch+sync 25 µs
(against 57–85 µs on the workstation), line solves 0.14–0.40 ms/apply with
C10 above C6 and F64 above F32 as arithmetic predicts, and the 64³ TGV
decomposed over 4 APUs at 0.074–0.088 s/step with an F64/F32 ratio near
1.2, consistent with full-rate FP64 on a fence-bound step. The open issue
is an intermittent stall mode: a process enters a state, sustained for
seconds to beyond 30 s, in which every device wait costs an integer number
of milliseconds (measured medians of exactly 13.000 ms, doubles at 26 ms),
compared with ~0.15 ms, then recovers. It strikes arbitrary scheme, dimension,
precision, and size cells and once held a whole TGV leg at 27×; the first
session's log looked the same before the larger sample resolved it, when
the pattern read as a C6/Float64 defect. It is not
garbage collection (one fully stalled 30 s window recorded zero GC time),
and it is not AMDGPU.jl's task-based synchronize fallback, which was the
first attribution and is measured wrong: setting the AMDGPU preference
`nonblocking_synchronization = false` (verified active by the recompile
and by the launch+sync floor dropping 25 → 19 µs) routes every wait
through a plain `hipStreamSynchronize` with no Julia tasks, and the stalls
persisted with the identical quantization; one rank spent 97% of its
watch at a p99 of exactly 26.004 ms. The quantized wait therefore sits
below the Julia layer, in HIP/ROCR's own stream wait or the driver, but
it requires a multithreaded Julia process: watch-only incidence runs
measured ~10 sustained episodes in 240 rank-seconds at `-t 8` (ranks up
to 89% stalled) against zero in 480 rank-seconds at `-t 1`, a
discrimination at the e⁻²⁰ level. The full characterization, the ruled-out
mechanisms, and the open experiments (thread-threshold scan, ROCR wait
mode, rocprof trace, LC ticket) are in
`reference/rocm_wait_stall_report.md`. Until it is resolved, run
device-resident rzadams jobs at `-t 1` per rank (the device step does not
need host threads and its measured cost is nil) and treat any wall
number from a multithreaded process as untrustworthy without a stall
watch beside it. Run-to-run spread on the workstation is
10–20%; read ratios, not third digits.

The first production-scale run (2026-08-20) adds a data point at `-t 1`: a 256³
single-species TGV, artificial properties on, decomposed over 4 APUs, 24,490
steps to t = 10 in 3.69 h (`bench/tgv_energy.jl`, recorded in
`reference/CALIBRATION.md`). The step baseline held at 0.35 s/step in Float64,
but the solver-time average was 0.52 s/step, about 1.5× the baseline, from
sporadic 100-step progress windows near 1.2 s/step (~3.5×) that make up roughly
20% of the run. This is at `-t 1`, where the watch-only runs above measured zero
episodes in 480 rank-seconds; over the ~52,000 rank-seconds here the incidence
is low but not zero, which the e⁻²⁰ discrimination at short duration does not
exclude. The measurement cannot attribute the elevated windows. They are
100-step averages, not per-wait timings; no stall watch ran beside the job;
and per-step-collective and GC jitter are not ruled out, although the
per-step kinetic-energy Allreduce is a separately accounted 437 s of diagnostics
and is not the cause, since the inflation sits in the solver time. A stall watch
at `-t 1` over a comparable duration, or per-step instrumentation of a
production run, would settle whether the quantized wait-stall persists at `-t 1`
at a low rate or these windows are something else. See
`reference/rocm_wait_stall_report.md`.

- 64³ TGV, single species, full step: device 0.146 s/step (Float64) and
  0.117 (Float32) under the deferred launch policy, against 0.203/0.171
  synchronized (the policy removed 28–32% of the step) and against
  ~0.12/0.10 for the 8-thread CPU, so the device stands at 0.83×/0.91× of
  CPU at this size. The remaining floor is launch submission plus one
  reduced-solve fence per line-solve apply.
- Isolated kernels clear CPU where launches amortize: the full flux-assembly
  body 9.9× over 8-thread `@threaded` at 64³ two-species; the δ⁴ stencil 4×
  in a single launch.
- Distributed device runs share the one workstation GPU, so their wall
  times measure contention, not scaling; their content is the bitwise gate
  and the transfer accounting (staged halo/pair copies 0.6–6.6% of device
  wall, reduced-interface copies 2–5%, fold-pair whole-block staging
  dominating staged volume).
- The AMR cost case: composite at 49% of uniform-fine wall, 24% of its
  memory, mixedness error 5× closer to fine than the coarse run (np = 8,
  region covering a sixth of the volume).
- First-launch kernel compilation is ~9 s per body on this stack; a fresh
  process pays minutes before the first timed step.

## Lessons that were not obvious

Ordered roughly by how expensive they were to learn. Each is enforced or
guarded somewhere; none should be re-derived.

1. **The live inter-level coupling must match the data's provenance.** The
   invertible filter pair is exact on data its own restriction produced;
   applied to live point samples it is O(h²) and measured order 1.3–1.7.
   Interpolation/injection measured 3.5–3.7. The general rule: deconvolution
   belongs only where the matching convolution has been applied.
2. **Replication of coupling work scales as rank count times imposition
   count, and subcycling multiplies impositions.** The chains were 20
   impositions per step; replicated, they erased the AMR cost advantage
   (85% of uniform-fine wall). Distributing by component with a thin-ring
   Allgatherv recovered it (49%) while moving the identical values, so the
   fix preserved bitwise serial equality. That is the pattern to prefer:
   redistribute work, never re-derive values.
3. **KA-CPU equality cannot certify the device path.** Three defects passed
   every KA-CPU test and failed only on real device storage: the un-adapted
   `ConservedState` kernel argument (does not lower), the >32-element splat
   (dynamic apply, `InvalidIRError`), and `_build_fine_patch`'s host-typed
   empty placeholders and unwrapped plans (type mismatch and scalar
   indexing). On the KA CPU backend, device arrays *are* host arrays, so
   storage-type and lowering defects are invisible by construction. The
   real-GPU battery is part of the gate.
4. **A `Vector` of arrays as a kernel argument hangs; adaptation is a
   design surface, not plumbing.** The `FieldVector`/`FieldMatrix` wrappers
   exist because the failure mode is a hang, not an error; the EOS mirrors
   exist because name strings and coefficient tables cannot cross; and the
   host must never hold the tuple form (3× on the threaded path). Every new
   collection that reaches a kernel needs its adapt story decided first.
5. **Specialization heuristics are part of the interface.** A `::Type`
   through the launcher's Vararg cost 9× as silent per-point dispatch; an
   `Any`-typed cache lookup behind a runtime `Ref` branch put 33 dispatch
   sites into the RHS report without ever executing. Jetcheck deltas are the
   tripwire for both, and the gate compares them probe for probe.
6. **Bitwise device equality is achievable, and it pays for itself as a test
   oracle.** Mirroring each host layout's arithmetic (the `colwise` switch),
   keeping reductions exact, and moving, not recomputing, values in the
   transfer paths gives a whole-run equality check over every phase at once,
   where a tolerance-based gate would have to be loose enough to pass the
   accumulated round-off and would then also pass a defect of that size.
   A difference in the last bit localizes to one operation.
7. **The launch floor was host round trips, not kernels.** Synchronizing
   per launch cost 28–32% of the device step; one in-order stream with a single
   algorithmic fence removed it without any event graph. Deleting
   synchronization, not streams and events, is the first tool against launch
   overhead, and a fence the algorithm requires (host reduced solve) bounds
   what any stream topology can recover.
8. **Pointwise metrics misread interface-dominated fields.** Coarse and
   composite both sit at max ≈ 0.19 against uniform-fine inside the refined
   region (sub-cell displacement of a near-discontinuous interface), while
   the physical mixing metric separates them by 5×. Cost cases must be
   judged on the quantity the refinement exists to predict.
9. **Restriction must stand off the imposed boundary.** Writing restricted
   fine data all the way to the coarse–fine boundary feeds the fine
   solution's least-accurate nodes back through the closure rows into the
   next shell: measured gain ≈ 2 per step. `RESTRICT_MARGIN = 2` breaks the
   loop.
10. **NaN survives rollback in two places**, the artificial coefficient
    arrays and the low-storage accumulator (0.0·NaN = NaN defeats
    `RKA[1] = 0`), and the resets are gated on `:nonfinite` failures
    because stale *finite* coefficients are a measured part of retry
    recovery.
11. **A device package loaded inside `main` is a world-age trap.** Its
    methods are newer than the running function's world; every launch then
    errors. Bench scripts load the backend at top level before `mpi_main`
    (`bench/tgv_energy.jl` carries the comment).
12. **`MPI.Dims_create`'s factorization takes no account of scheme minima.**
    If it picks a fine-patch grid, a regrid can die mid-run on the 9-point
    closure constraint; `_amr_dims` searches factorizations under it
    and errors at setup or at the regrid that shrank the region, with the
    numbers in the message.

## Scope boundaries

Configurations rejected at setup, and the reason for each:

- **Patched runs** (same-level `patch_grid`) reject folds (constraint 4),
  banded schemes (C10 interface closures are not tabulated), the `:d8`
  detector, stretching along the patched dimension, and an explicit `dims`.
  The layout generator tiles slabs along one dimension, so corner-coupled
  adjacency does not arise. Checkpoint/VTK output remains single-patch.
- **Refined runs** require Cartesian metric, no stretching, no folds,
  tridiagonal schemes, `:delta4`, one refined region, and no same-level
  `patch_grid` alongside. `level_restriction = :filter` is serial-only (a
  whole-patch line solve). The refined region must nest by
  `max(n_halo, LEVEL_BUFFER)` coarse nodes and span ≥ 4 coarse nodes per
  active dimension.
- **Device runs** take a single patch per solver (the same-level
  interface-record copies are host loops), reject `Nasa9Mixture` (no
  fixed-width device mirror yet), a pointwise NSCBC inflow `target` (host
  closure), `StepControl.floor_ratio > 0` and `dt_report` (host sweeps),
  and `:filter` restriction.
- The artificial-property sensors are built per patch with closed-edge
  clamping at interfaces; the gates run so far have not measured the effect.

## Roadmap

In rough priority order. Each item names the measurement or event that
should trigger it; none should be built speculatively.

1. **`Nasa9Mixture` device mirror.** A flattened, fixed-width interval table
   (intervals padded to a maximum count) with `Adapt.adapt_structure`, plus
   the Newton inversion in the kernel body. The last EOS off the device;
   mechanical now that the mirror pattern is proven twice.
2. **G4b — mixed precision policy.** Open, to be settled on the target
   machine, not the workstation: uniform Float32 buys 1.25× on the launch-
   and fence-bound RX 6800 XT step (1.10× on CPU) against a 1e-4
   conservation drift, but that ratio is a consumer-RDNA2
   number. MI300A runs FP64 at full rate, so on the target the question
   is memory traffic and capacity, not FLOPs.
   Reopen with rzadams measurements once a memory-capacity-bound case
   appears or the fence floor drops (items 3–4); the workstation matrix
   stays useful only for the drift and footprint halves.
3. **Same-level multi-patch on device.** Stage the interface-record
   pack/copy loops through the backend as the halo path was staged. This is
   also the event at which per-patch streams should be revisited: those
   patches are independent within an RK stage.
4. **On-device reduced solve, or device-aware MPI.** The per-apply fence is
   the measured remaining serialization. Two routes: move the small
   reduced solve onto the device (device collectives, or redundant
   per-rank solves on gathered ends), or a GPU-aware MPI stack via the
   `device_mpi_direct` hook plus pinned staging buffers. The workstation's
   stack supports neither; rzadams (GPU-aware Cray MPICH on MI300A, where
   unified APU memory also collapses the staging question) is where both
   routes get measured, and this item is the natural first rzadams
   campaign together with re-basing the performance summary there. Its
   first order of business is the intermittent wait-stall mode recorded in
   the performance summary: until a process can be shown stall-free, or
   stalls can be detected and excluded, no fence-floor measurement on that
   machine is trustworthy.
5. **Fine-level sub-communicators.** A small refined region at high rank
   count fails `_amr_dims` (9·np along a 1-D region). Letting a subset of
   ranks own the fine level, with the rest idling through its collectives,
   lifts the constraint at the cost of the patch-loop architecture assuming
   every rank holds every level. Trigger: a production case that needs
   small regions on many ranks.
6. **Rank-partitioned level transfer.** The replicated gathers cost region
   volume × rank count in message total and one region-sized replica per
   rank. Fine while regions are small; the follow-up if a measured case
   outgrows it is per-rank sub-box chains with footprint-propagated
   overlap margins (order-6 stencils through K stages), which is substantial
   bookkeeping and should wait for the case.
7. **Tile clustering and multiple fine regions.** The single bounding box
   over-refines L-shaped tagged sets. Fixed-size tiles first
   (edge ≥ 3 coarse cells ⇒ ≥ 9 fine points), Berger–Rigoutsos only if
   measured tile waste on real problems justifies it. Requires a
   multi-patch fine level (same-level adjacency machinery at level 1,
   reusing the Stage 2 interface records), which is CPU work before any
   device concern.
8. **Filter-cadence measurement on TGV** (the surviving piece of old
   risk 4): the per-level cadence convention is measured on Sod gates only;
   the smooth-turbulence dissipation budget under subcycling ties into the
   filter-calibration item in `ROADMAP.md` and `CALIBRATION.md`.
9. **Fold-adjacent refinement** stays forbidden, not solved.
   Converging-shock problems need resolution where refinement is
   excluded; the workaround is a globally fine level-0 in r near the fold.
   Lifting it requires extending the antipodal pairing across levels,
   design work justified only by a driving problem.
10. **Distributed `:filter` restriction.** The anti-alias pair is the tool
    if injection restriction proves positivity-limited on captured shocks
    (it has not yet); distributing it requires running the fine-side filter
    through the fine patch's own decomposed plans before the sampled
    gather. Do it when the shock case demands `:filter` under MPI, not
    before.
11. **Multi-patch I/O.** A patch writes its `region` in the same form as a rank
    block, so the HDF5 hyperslab extension is mechanical; VTK
    multiblock likewise. Trigger: the first multi-patch production run
    that needs restart or visualization, not before.
12. **Load-balance weighting.** Volume-proportional rank assignment ignores
    that subcycled fine patches cost 3× the steps; weight by
    volume × 3^level when partitioning a multi-patch fine level. Revisit
    with measurements once item 7 exists.

Conservation remains the one standing numerical debt:
compact closure rows at an interface do not telescope, so global
conservation carries a drift term (measured 1.2e-8 relative per long
periodic run at a Stage 2 interface; 1e-4 over a full two-level shock
crossing) where a finite-volume code would reflux. The shared-plane
averaging removes the worst mode, the surface-correction fallback on
interface fluxes is designed but unbuilt, and building it is warranted only
when a case shows the drift competing with the answer.
