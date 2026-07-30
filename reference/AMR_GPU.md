# CompactLES — Patch AMR and GPU port: implementation plan

This document is the implementation plan for patch-based adaptivity and the
GPU port, written to be executed stage by stage without re-deriving the
constraints. Prerequisite reading: `DESIGN.md` (compact operators, the
distributed solve, folds) and the [Evidence base](#evidence-base) below.
`ROADMAP.md` holds positioning and priorities; `HISTORY.md` records the
groundwork already done.

## Contents

1. [Why one plan for both](#why-one-plan-for-both)
2. [Constraints](#constraints)
3. [Evidence base](#evidence-base)
4. [Stage 1 — transfer operators as compact schemes](#stage-1--transfer-operators-as-compact-schemes)
5. [Stage 2 — patch abstraction, single level](#stage-2--patch-abstraction-single-level)
6. [Stage 3 — static two-level refinement, global timestep](#stage-3--static-two-level-refinement-global-timestep)
7. [Stage 4 — subcycling, tagging, and regridding](#stage-4--subcycling-tagging-and-regridding)
8. [GPU track](#gpu-track)
9. [Ordering and dependencies](#ordering-and-dependencies)
10. [Risks and open questions](#risks-and-open-questions)

## Why one plan for both

Patch AMR replaces the "one global array per field" assumption with many
independently sized blocks, each with its own line plans and ghost regions. A
GPU port wants exactly that shape: a patch is a natural kernel-launch unit with
a bounded working set, and a port against the current monolithic arrays would
have to be redone once patches exist. The two capabilities therefore share the
Stage 2 refactor: the patch abstraction and the storage-type generalization
(GPU Stage G0) touch the same structs and should be done in the same change.
Miranda's GPU implementation of patch AMR is evidence the shared architecture
works.

Sequencing: AMR-shaped refactor first, GPU kernels second. The reverse order
duplicates the decomposition work.

## Constraints

Each of these is structural or measured; none is a preference.

1. **No line cut mid-solve.** A compact operator couples an entire grid line
   through a banded LHS. A patch is therefore a logically rectangular block of
   uniform resolution; the compact solve runs over the patch with ghost layers
   filled by same-level copy or coarse-level interpolation. Oct-tree
   cell-by-cell refinement is excluded (see Non-goals in `ROADMAP.md`).
2. **Minimum extent 9 per resolved dimension.** The C8 filter closure is the
   binding scheme (`plan_direction` enforces this today). Practical transverse
   extents in the test suite are 12–16. This bounds the minimum patch size and
   therefore the tagging granularity.
3. **Refinement ratio 3.** The Miranda transfer filter has width 3Δx, and the
   odd ratio keeps a coarse node coincident with the middle node of each fine
   triple, so both transfer operators stay centered, symmetric, and
   invertible on a node-centered grid. Do not attempt ratio 2.
4. **Folds require uniform structure in their paired dimension** (partner at
   +P/2, reflected partner). Refinement across a fold's singular region or its
   antipodal partner region is forbidden; `setup` must reject it. The singular
   region keeps uniform resolution.
5. **The bit-exact oracle survives only within a patch.** The
   spike/reduced-interface solve reproduces the single-domain answer at any
   rank count *within* one patch; patch interfaces are approximate by
   construction. Every stage below that introduces an interface must bring its
   replacement oracle: a manufactured smooth solution across the interface
   with a measured order, in the style `test/convergence.jl` uses.
6. **The grid arrangement stays node-centered**, half-offset only where a fold
   requires it. The SBP–SAT multiblock literature is node-coincident and
   handles non-conforming interfaces with interpolation operators; a
   cell-centered migration is not a prerequisite (see `HISTORY.md`).
7. **Error localization is favorable.** The inverse of a compact LHS decays
   geometrically off the diagonal (≈ α^|i−j|, α = 1/3 for C6), so pollution
   injected at an inexact interface decays about 3× per point into the patch.
   Stage 2 should measure this decay once, directly, as the basis for choosing
   interface-adjacent buffer widths.

## Evidence base

### Miranda's transfer kernels (primary source)

Pyranda does not expose AMR, but it carries Miranda's Fortran kernels. The
level-transfer operators are in `pyranda/parcop/stencils.f90` — `cfamrcf` and
`cfamrfc`, inside an `#if 0` block, so they are the design rather than a live
implementation. The header defines the scheme:

```
INVERTIBLE AMR FILTER FOR COARSE-TO-FINE AND FINE-TO-COARSE OPERATIONS (fbar <---> f):
alpha*fbar(i-1) + fbar(i) + alpha*fbar(i+1) = c*f(i-2) + b*f(i-1) + a*f(i) + b*f(i+1) + c*f(i+2)
alpha = -0.0321826755129339
    a =  0.4451523642186118
    b =  0.2207614172195584
    c =  0.0244797251582018
The above coefficients match the transfer function of a gaussian filter of width 3*dx.
```

Level transfer is an invertible compact filter pair: restriction applies a
compact filter whose transfer function is a Gaussian of width 3Δx;
prolongation recovers the fine field by deconvolving the same filter — the two
routines are the identical coefficients with the LHS and RHS roles swapped
(`cfamrcf` solves the `[c,b,a,b,c]` pentadiagonal against an `[α,1,α]`
right-hand side; `cfamrfc` does the reverse). Five properties matter:

- **Refinement ratio 3, not 2**, matched to the 3Δx filter width, keeping both
  operators centered and invertible without staggering.
- **Conservation comes from unit DC gain**, not refluxing: a + 2b + 2c
  = 1 + 2α to the last bit (both 0.9356346489741322 in Float64), so the
  transfer preserves the mean exactly.
- **The boundary closures preserve that gain.** The first closure row is
  explicit and sums to exactly 1.0; the second has an LHS summing to the
  interior RHS within one ulp.
- **Boundary invertibility requires specialized closures.** Source comments:
  "no filter (current) or bad filter at boundary causes ringing in the
  solution" and "with extended boundary data : same as one-sided to maintain
  invertibility." Four closure variants are tabulated per end — odd-symmetric,
  one-sided, even-symmetric, and extended-data — on the `-1:2` index.
- **The parity machinery matches the existing fold formulation.**
  `lower_symm_weights`/`upper_symm_weights` fold a ghost coefficient onto the
  interior with a sign, structurally identical to the axis fold
  (`b[1] += σg·α` in `operators.jl`), and `nci` ("the parallel overlap of lhs
  stencil") indicates these run through the same distributed compact solve as
  everything else.

A companion family of coarsening filters exists: `c4ff3` ("compact filter for
AMR coarsening," whose transfer function integrates to 1/3 over [0, π] —
exactly the 3:1 spectral budget) and `cgff2` (a 5×5 Gaussian ≈ exp(−2k²/3)).
These are the anti-aliasing step ahead of 3:1 subsampling.

Public Pyranda also carries patch *infrastructure* without the logic:
`objects.f90` reserves (patch, level) tables (101 × 11), each with patch
metadata, its own Cartesian MPI communicator, compact plans, and mesh;
`comm.f90` creates the patch communicator with `MPI_Comm_split`; `ghost.f90`
exchanges halos only within a uniform patch communicator and applies
physical-boundary symmetry or extrapolation; `mesh.f90` allocates a SAMRAI
refinement-tag array the public tree never populates. The scheduler, patch
adjacency, regridding, subcycling, and tagging criteria are absent and are
designed here against this code's own constraints.

### Conditioning measurements (`bench/amr_transfer.jl`)

The periodic symbols and finite one-sided matrices were reconstructed directly
from the coefficients. For a 96-point line: prolongation has gain 1.50883 at
the representable coarse Nyquist (k = π/3) and 20.2393 at the fine Nyquist;
the finite closure operator has condition number 33.03 and spectral norm
20.19; six-point alternating noise is amplified 13.80 at an edge versus 16.11
in the interior (the closure is not the largest amplifier); the swapped pair
round-trips to 3.4e-15 in infinity norm.

Interpretation: prolongation is deconvolution of a Gaussian, so it amplifies
high wavenumbers by construction. That is benign while the coarse field
carries no content above the coarse Nyquist, and dangerous when a boundary
closure or an artificial-property sensor injects grid-scale content near a
transfer boundary — very likely why the Fortran comments fuss about ringing
and closure invertibility. Any operation near an interface that can inject
fine-Nyquist content must be filtered or excluded; the ~20× bound above sizes
the risk.

### SBP–SAT, the rigorous alternative

Retained as the fallback if deconvolution conditioning proves unacceptable in
Stage 3 measurements, and as the theory guiding interface placement:

- Carpenter, Gottlieb & Abarbanel (1993); Abarbanel & Chertock (2000) —
  stable boundary treatments for compact schemes.
- Mattsson & Rydin (2022) — implicit SBP operators on a banded-norm framework
  with weak (penalty) enforcement, 8th-order global convergence; the
  Padé-class case and principal reference.
- Nissen et al. (2015) — block-adaptive SBP–SAT grids, including junction
  points.
- Almquist & Dunham (2018) — order-preserving interpolation for
  non-conforming (refined) interfaces.

Three documented considerations: accuracy drops at SAT interfaces below the
interior order, so interfaces should be minimized; non-conforming interfaces
cost a global order unless order-preserving interpolation operators are used;
and stability proofs above 4th order require additional penalty terms.
CompactLES's `ClosureRow` cascade is classical Lele, not SBP; moving to SBP
would touch `kernels.jl`, `operators.jl`, the folds, and every guard in
`test/convergence.jl`, and is not undertaken unless Stage 3 measurements force
it. Note the family resemblance: SAT penalties act on the RHS exactly as the
NSCBC corrections in `nscbc.jl` do, so an eventual SBP–SAT implementation
could subsume that machinery rather than fight it.

HAMeRS (Wong & Lele: patch AMR on SAMRAI for RM/RT mixing) is the closest
published relative and uses explicit schemes; Miranda's implicit-compact
approach is the less documented one. Julia has no SAMRAI-maturity AMR
framework; Trixi's precedent is binding an established C library (P4est.jl).
The patch bookkeeping below is deliberately simple enough not to need either.

## Stage 1 — transfer operators as compact schemes

Small, independent of everything else, and can be done at any time.

**Deliverable.** A new `src/transfer.jl` defining the level-transfer operator
pair, plus tests. No solver changes.

**Implementation.**

- `amr_transfer_schemes()` returns the restriction and prolongation operators
  built from the tabulated coefficients. Restriction is a standard compact
  filter application: solve the `[α, 1, α]` tridiagonal LHS against the
  `[c, b, a, b, c]` RHS — expressible as a `CompactScheme` (symmetric flag,
  center weight a) exactly like `compact_filter`. Prolongation solves the
  `[c, b, a, b, c]` pentadiagonal LHS against the `[α, 1, α]` RHS — a
  `BandedCompactScheme` with q = 2. Both bind to a dimension with the existing
  `plan_direction`, which also supplies the distributed spike solve (the `nci`
  parallel overlap in Miranda's terms) and the fold parity variants for free.
- Implement the four boundary-closure variants per end (odd-symmetric,
  one-sided, even-symmetric, extended-data) as `ClosureRow` sets selected per
  patch face type. Start with one-sided and extended-data; the symmetric pair
  maps onto the existing fold machinery.
- The 3:1 sampling convention between the filter pair and the coarse grid is
  not fully specified by the public kernels (see
  [Risks](#risks-and-open-questions)); pin it down numerically before writing
  the patch code that consumes it. The working hypothesis: restriction =
  filter on the fine line, then take every third node (the ones coincident
  with coarse nodes); prolongation = inject coarse values onto every third
  fine node, fill the intermediate nodes by the deconvolution solve.
  `bench/amr_transfer.jl` already reconstructs the operators; extend it until
  the round-trip on smooth data reproduces its 3.4e-15 figure through the new
  `src/transfer.jl` types.

**Tests (gate for the stage).**

- Round-trip fine → coarse → fine on smooth periodic data ≤ 1e-14.
- DC gain exactly 1 (constant in, constant out, to the last bit) in the
  interior and through each closure variant.
- Measured order of the restriction/prolongation pair on a smooth field.
- The sensor-injection test named in the groundwork: apply the Cook δ⁴ sensor
  chain (`delta4_sum!` + `smooth!`) to data containing a captured-shock-like
  profile near a transfer boundary, prolong, and measure the amplification
  against the ~20× bound. This number decides how much buffering Stage 3
  interfaces need and whether the coarsening filter (`c4ff3` analog) must run
  before every restriction.

## Stage 2 — patch abstraction, single level

The long pole. Everything currently assuming one block per rank is
generalized to N same-level patches tiling the domain. This stage also
carries GPU Stage G0 (storage-type generalization), because both rewrite the
same struct definitions and doing them separately means touching every
signature twice.

### Data structures

Split `Solver` (currently physics config + one block's arrays and plans) into
configuration and per-patch state. Sketch, not final signatures:

```julia
struct Patch{T,A<:AbstractArray{T,3}}
    id::Int
    level::Int
    region::BlockRegion            # global offset + extent in level index space
    comm::MPI.Comm                 # ranks owning pieces of this patch
    decomp::Decomp                 # decomposition of the patch over comm
    h::NTuple{3,T}
    coord_shift::NTuple{3,T}
    faces::NTuple{3,NTuple{2,FaceKind}}   # Physical | Interface(nbr) | CoarseFine(...)
    deriv_plans; filter_plans; folds; pairbuf; pairout
    # geometry, primitives, gradients, artificial arrays, flux, scratch —
    # everything from `rho` through `flux` in the current Solver, typed A
end

mutable struct Solver{...}
    equations; eos; transport; art; metric; stretch; sources; bcs
    cfl; filter_interval; control
    patches::Vector{Patch{...}}    # this rank's patches, globally ordered
    comm::MPI.Comm                 # world
    t; tstage; step; dt_prev; rate_prev; wall_step; wall_total
end
```

The conserved state becomes one array per patch (`Vector{ConservedState}`
aligned with `solver.patches`), and `Workspace` follows. `BlockRegion`
(`io.jl`) is already the right region type and deliberately not a `Decomp`.

### Rank assignment and communicators

First cut: patches partition the rank set — each rank belongs to exactly one
patch's communicator, built once with `MPI.Comm_split`, with ranks allocated
to patches proportionally to patch volume. The degenerate case (one patch
spanning all ranks) must reproduce the current code path exactly. The
alternative (every rank participating in every patch, patches processed
sequentially) serializes the machine and is rejected.

Collective discipline, which is where this refactor can go wrong silently
(`CLAUDE.md` Traps):

- Line solves and line-wise reductions are collective over the *patch*
  communicator; only ranks in that communicator enter them. With disjoint
  rank sets this is deadlock-free by construction.
- `compute_dt`, `WhenState`, and diagnostics reduce over *world*. Hoist these
  outside the patch loop; every rank enters them exactly once per step
  regardless of how many patches it owns.
- Patches must be iterated in the same global order on every rank wherever a
  world collective appears inside the loop (preferably: never).

### Interface treatment (same level)

A patch face is `Physical` (existing BC machinery), or `Interface`: ghost
layers filled by copy from the abutting patch at the same level, then the line
solve closed with the extended-data closure rows from Stage 1's tabulated set
(interior-like stencils reading ghost data, one-sided beyond it). This is
Miranda's `ghost.f90` shape. Interface exchange is a new
`exchange_patch_ghosts!` alongside `halo.jl`: for each face pair, a
`Sendrecv!` between the owning rank subsets; same-rank faces are copies.
Since the grids are node-coincident, abutting patches share the interface
plane nodes; after each RK stage the shared plane is made consistent by
averaging the two patches' values (cheap, symmetric, removes the drift mode).

Accuracy at an interface drops toward the closure order, as at any closed
boundary; the α^|i−j| decay confines it. Patch layouts should therefore
minimize interface count — patches as large as memory allows, which is also
what the GPU wants.

### Mechanical refactor list

Every function that touches arrays gains a per-patch inner method with the
existing body, and an outer loop over `solver.patches`:

- `primitives!`, `compute_artificial!`, `assemble_fluxes!`,
  `compute_primitives_and_gradients!`, `compute_rhs!` (rhs.jl, artificial.jl)
- `step!`, `filter_state!`, `max_rate` (timestep.jl) — `max_rate` accumulates
  over patches, then one world `Allreduce`
- `apply_bcs!`, `wallplane` consumers (boundary.jl, nscbc.jl, problem.jl)
- diagnostics (volume integrals gain a patch loop before the world reduction)
- `io.jl` / `hdf5.jl` (a patch writes its `region` exactly as a rank block
  does today; the HDF5 hyperslab path needs no design change)

`xcoord`/`gidx` move to `(patch, d, i)` forms; keep thin `Solver` wrappers
that assert a single patch, so existing scripts keep working during the
transition.

### Verification gates

1. **One patch = current code**: the convergence guards in
   `test/convergence.jl` are bit-identical, and the full serial + MPI gate
   passes unchanged. This is a hard gate before any multi-patch work.
2. **Two conforming patches**: manufactured smooth solution across the
   interface with a measured order (expect ≈ 3, the closure-cascade order);
   an advected acoustic pulse crossing the interface with measured reflected
   amplitude; global conservation drift measured over a long periodic run and
   compared against the single-patch baseline (see
   [Risks](#risks-and-open-questions) on conservation).
3. **Bit-exactness within a patch across rank counts**: the existing MPI-suite
   oracle rerun with a 2-patch layout at 2/4/8 ranks.

Stage 2 alone already delivers conforming multiblock geometry, which is
independently useful (e.g., an L-shaped or annular-sector domain) even if
refinement never follows.

## Stage 3 — static two-level refinement, global timestep

Two levels, refinement regions fixed at setup, every level advancing with the
same `dt` (the global minimum). Removing subcycling from this stage isolates
the spatial coupling, which is where the numerical risk is.

- **Level layout.** `solver.levels::Vector{Vector{Patch}}`; level-1 patches
  cover user-specified regions at ratio 3, properly nested with a margin of
  at least `n_halo` coarse cells inside level 0, never overlapping a fold's
  singular or partner region (setup error otherwise, per constraint 4).
- **Fine ghost fill.** Same-level neighbor copy where available; otherwise
  spatial prolongation from the underlying coarse patch (Stage 1 operator)
  into the fine ghost layers, every RK stage. With a global dt there is no
  temporal interpolation: coarse and fine are at the same time at every
  stage boundary, and stage times coincide because dt is shared.
- **Restriction.** After each step, restrict fine Q onto the covered coarse
  region (coarsening filter, then 3:1 sample). The covered coarse region is
  otherwise still advanced — cheap, simple, and its values are overwritten;
  masking it out is an optimization for later.
- **Sensor convention across levels.** The Cook sensors are grid-based
  (undivided δ⁴ with h-power weights) and are computed per level with the
  level's own h. The same physical feature therefore receives less artificial
  property on the fine level — the desired behavior for a regularization that
  follows the mesh. Document this as the convention; do not blend sensors
  across levels.
- **Filtering.** Each level filters its own state at the shared cadence.
  Filtering and restriction both smooth; measure the combined dissipation on
  TGV with a static refined region before accepting the defaults (this
  interacts with the filter-calibration item in `ROADMAP.md`).

**Gates.** Manufactured smooth solution spanning the coarse–fine boundary
with measured order; a shock (Sod) passing through a refinement boundary in
both directions without spurious reflection above a set tolerance; the Stage 1
sensor-injection measurement repeated in situ with Cook sensors active near
the interface; conservation drift versus a uniform-fine reference.

## Stage 4 — subcycling, tagging, and regridding

- **Subcycling.** Ratio 3 in space gives ratio 3 in time: one coarse step,
  then three fine steps (Berger–Oliger order: coarse first, fine after, then
  restrict). Fine ghost data at fine stage times needs coarse values *between*
  coarse steps: store the coarse boundary rings (not full arrays) at t^n and
  t^{n+1} together with their RHS values — both already computed — and use
  cubic Hermite interpolation in time, O(dt⁴), matching the integrator. LSRK
  has no free dense output; the Hermite construction is the standard
  substitute.
- **Tagging.** The Cook δ⁴ sensors are exactly the "where is this
  under-resolved" signal: tag coarse cells where the smoothed sensor exceeds a
  threshold, buffer by B cells, and cluster. Clustering first cut: fixed-size
  tiles in coarse index space (tile edge ≥ the minimum-extent constraint after
  refinement, i.e. ≥ 3 coarse cells ⇒ ≥ 9 fine points, in practice 8–16
  coarse cells per tile edge); Berger–Rigoutsos later only if tile waste
  measured on real problems justifies it.
- **Regridding.** Every K coarse steps: retag, rebuild level-1 patch set,
  initialize new fine regions by prolongation, carry over surviving fine data
  by copy. Communicators for a changed patch set are rebuilt with
  `Comm_split`; this is setup-cost machinery, not per-step.
- **Gates.** Subcycled Sod and Shu–Osher through a moving refined region
  tracking the shock, compared against uniform-fine; wall-time and memory
  measurements demonstrating the point of the whole exercise (equal accuracy
  at reduced cost) on at least one 3-D mixing case.

## GPU track

Architecture: KernelAbstractions.jl, one codebase, CPU and GPU backends —
XCALibre's and Oceananigans' pattern. Two codebases are forbidden. KA becomes
a hard dependency (the one dependency a GPU port cannot avoid); CUDA.jl and
friends arrive only through the user's environment, optionally with package
extensions for backend-specific glue.

### G0 — storage generalization (executed inside Stage 2)

- Field arrays: `Array{T,3}` → type parameter `A<:AbstractArray{T,3}` on
  `Patch`; allocation goes through a backend object
  (`field(backend, decomp)`), defaulting to CPU.
- Halo and pair buffers: concretely `Float64` today (`decomposition.jl`),
  the known blocker for `Solver{T}` generality. Type them `T` and allocate
  them with the backend. This also unlocks the mixed-precision item in
  `ROADMAP.md` and verifies the existing `{T}` parameterization.
- `Vector{Array{T,3}}` fields (`Y`, `D_art`) and `Matrix{Array{T,3}}`
  (`grad_u`, `grad_Y`, `flux`) become collections of `A`.

### G1 — pointwise kernels

Convert the phases that are pointwise loops — per `bench/phases.jl` the
majority of the budget outside the line solves:

- `primitives!` (per EOS), `assemble_fluxes!`, the RK update in `step!`,
  `delta4_sum!`, the sensor-to-coefficient loops in `compute_artificial!`,
  `metric_correct_gradients!`, `add_metric_sources!`, `add_source!`,
  `_scale_grad!`, and the divergence accumulate loops in `compute_rhs!`.
- Kernel form: `@kernel` bodies indexed by `@index(Global, NTuple)` over the
  interior (or full padded extent, matching each loop today), launched with
  an ndrange covering all three dimensions. Those loops now iterate a flattened
  `outer_indices(n2, n3)` (see Threading in `CLAUDE.md`), which was the interim
  fix for the planar-2-D dead-threading defect; an ndrange over all three
  dimensions supersedes it, so drop the helper as each loop converts rather than
  translating it.
- CPU path: benchmark KA's CPU backend against the current `@threaded` loops
  with `bench/phases.jl`. Acceptance: within 10% per phase at 64³ on the
  reference workstation. If KA-CPU regresses beyond that, keep `@threaded`
  bodies for `A === Array` behind the same function names and route only
  device arrays to KA; a dispatch split on array type is acceptable, a
  hand-maintained second kernel body is not.
- Reductions: `max_rate` becomes a `mapreduce` over the interior with a tuple
  monoid `(rate, -ρ_min)` (GPUArrays supports this), then the existing world
  `Allreduce`.
- EOS on device: `IdealMixture` holds `Vector` coefficient tables and
  `Nasa9Mixture` holds nested jagged vectors. G1 supports `IdealMixture` and
  `StiffenedGas` first, converting `IdealMixture`'s vectors to tuples or
  device arrays via Adapt.jl. `Nasa9Mixture` needs a flattened, fixed-width
  device mirror (intervals padded to a maximum count); do it after the
  mechanism is proven on the simple models.

### G2 — line solves on device

The distributed compact solve is batched Thomas/banded sweeps: sequential
along a line, embarrassingly parallel across lines. The transposed
`(lines × n)` layout (`lines_transposed.jl`) was built so that fills, sweeps,
and corrections iterate contiguously across lines — which on a GPU is exactly
one thread per line with coalesced access at every sweep position.

- Kernels: gather/fill, forward elimination + back substitution (one thread
  per line, loop over n inside the thread), spike correction, scatter. The
  x-sweep's `(n × lines)` layout is the transposed case; either permute to
  the lines-inner layout on device or write the second kernel — measure,
  don't guess.
- The reduced 2qP × 2qP interface system stays host-side initially: copy the
  2q interface values per line to host, `Allgather`, `ldiv!` on the
  pre-factorized LU, copy corrections back. The transfer is 2q values per
  line, small next to the field itself; move it on-device only if profiling
  shows the transfer dominating.
- Halo/ghost exchange: pack/unpack kernels into contiguous device buffers;
  pass device pointers to MPI when the library is CUDA-aware
  (`MPI.has_cuda()`), else stage through pinned host buffers. Runtime
  detection, no build-time switch.

### G3 — residency and patches as launch units

- Audit with the vendor profiler that a full step performs no host transfers
  except the reduced solves, `Allreduce`, and halo staging.
- One patch = one stream of kernel launches; independent patches on a rank
  can overlap. This is where the Stage 2 architecture pays off.
- I/O gathers to host explicitly; no change to the HDF5/VTK design.

### GPU verification

Bit-exactness against CPU is not expected (different reassociation). Gates:

- `test/convergence.jl` measured orders identical to the guard table.
- Freestream preservation stays machine-zero in every metric: the discrete
  GCL cancels the same stored operator output node-by-node, which is
  arithmetic-order independent, so this gate is exact even on device.
- `test/validation.jl` guards matched at the documented comparison level
  (four significant figures, a moved third digit is real).
- The TGV −dKE/dt(t) history overlaid on the CPU curve.
- Per-phase speedups recorded with `bench/phases.jl` before/after, per the
  usual delta discipline.

## Ordering and dependencies

```
Stage 1 (transfer ops)          — independent, any time, small
Stage 2 (+G0, patch refactor)   — the long pole; hard-gated on bit-identity
G1 (pointwise kernels)  ─┐
Stage 3 (two-level)     ─┴─ either order or interleaved, both need Stage 2
G2 (device line solves)         — after G1
Stage 4 (subcycle/tag/regrid)   — after Stage 3
G3 (residency, streams)         — after G2, benefits from Stage 4 patches
```

Do Stage 2 only after the regularization debts in `ROADMAP.md` settle: the
refactor touches every file, and rebasing physics changes across it is the
expensive order.

## Risks and open questions

1. **The 3:1 sampling convention** between the filter pair and the coarse
   node set is inferred, not documented; Stage 1 resolves it numerically
   before any patch code consumes it. Symptom of getting it wrong: the
   round-trip fails at O(1), not subtly.
2. **Conservation at interfaces.** The transfer operators preserve the mean
   (unit DC gain), and restriction-over-covered-region keeps levels
   consistent, but the compact divergence with closure rows at an interface
   does not telescope exactly, so global conservation acquires a drift term
   where a finite-volume code would reflux. Measure it in Stage 2 gate 2 and
   Stage 3; the shared-plane averaging removes the worst mode, and the
   fallback is a surface correction on the interface fluxes. Do not build
   refluxing machinery speculatively.
3. **Deconvolution amplification** (~20× at fine Nyquist) interacting with
   Cook sensors near interfaces — the measured risk from the groundwork. The
   Stage 1 sensor-injection test sizes it; mitigations, in order: run the
   coarsening filter before every restriction, keep refinement boundaries
   the buffer distance away from tagged features, and only then consider
   SBP–SAT interfaces.
4. **Filter cadence across subcycled levels.** A fine level filtering every
   fine step filters 3× as often per unit time as its parent, and the
   filter's dissipation is per-application. This is the same dt-consistency
   defect already on the roadmap for the single-grid code, sharpened by
   subcycling. If the roadmap item lands first (filter strength tied to dt),
   the AMR consequence resolves itself; otherwise Stage 4 must pick a
   convention and measure its effect on TGV.
5. **Load balance.** Volume-proportional rank assignment ignores that fine
   patches cost 3× the steps under subcycling. Weight patch cost by
   level (volume × 3^level) when partitioning; revisit only with
   measurements.
6. **KernelAbstractions dependency** dilutes the dependency-light stance.
   Accepted: it is the single unavoidable dependency of a portable GPU path,
   with two established precedents in the Julia CFD ecosystem, and it arrives
   in the same change that deletes the three unused dependencies noted in
   `ROADMAP.md`.
7. **Fold-adjacent refinement** is forbidden, not solved. Converging-shock
   problems want resolution *at* the axis/origin, which is exactly where
   refinement is excluded. The mitigation is that level 0 can be globally
   fine in r near the fold (stretching is also excluded on folded dimensions,
   so this costs points); lifting the restriction is future work that
   requires extending the antipodal pairing across levels.
