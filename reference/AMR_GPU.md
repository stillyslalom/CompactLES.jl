# CompactLES — Patch AMR and GPU port: implementation plan

This document is the implementation plan for patch-based adaptive mesh
refinement (AMR) and the GPU port, written to be executed stage by stage
without re-deriving the constraints. Prerequisite reading: `DESIGN.md`
(compact operators, the distributed solve, folds) and the
[Evidence base](#evidence-base) below.
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
GPU port needs exactly that shape: a patch is a natural kernel-launch unit with
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
   +P/2 for P ranks along that dimension, reflected partner for the reversed
   one). Refinement across a fold's singular region or its antipodal partner
   region is forbidden; `setup` must reject it. The singular region keeps
   uniform resolution.
5. **The bit-exact oracle survives only within a patch.** The
   spike/reduced-interface solve reproduces the single-domain answer at any
   rank count *within* one patch; patch interfaces are approximate by
   construction. Every stage below that introduces an interface must bring its
   replacement oracle: a manufactured smooth solution across the interface
   with a measured order, in the style `test/convergence.jl` uses.
6. **The grid arrangement stays node-centered**, half-offset only where a fold
   requires it. The summation-by-parts / simultaneous-approximation-term
   (SBP–SAT) multiblock literature is node-coincident and handles
   non-conforming interfaces with interpolation operators; a cell-centered
   migration is not a prerequisite (see `HISTORY.md`).
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
- **Conservation comes from unit DC gain**, not refluxing: the zero-wavenumber
  (DC) gain of the transfer pair is a + 2b + 2c = 1 + 2α to the last bit (both
  0.9356346489741322 in Float64), so the transfer preserves the mean exactly.
- **The boundary closures preserve that gain.** The first closure row is
  explicit and sums to exactly 1.0; the second has an LHS summing to the
  interior RHS within one unit in the last place.
- **Boundary invertibility requires specialized closures.** Source comments:
  "no filter (current) or bad filter at boundary causes ringing in the
  solution" and "with extended boundary data : same as one-sided to maintain
  invertibility." Four closure variants are tabulated per end on the `-1:2`
  index: odd-symmetric, one-sided, even-symmetric, and extended-data.
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

HAMeRS (Wong & Lele: patch AMR on SAMRAI for Richtmyer–Meshkov and
Rayleigh–Taylor mixing) is the closest
published relative and uses explicit schemes; Miranda's implicit-compact
approach is the less documented one. Julia has no SAMRAI-maturity AMR
framework; Trixi's precedent is binding an established C library (P4est.jl).
The patch bookkeeping below is deliberately simple enough not to need either.

## Stage 1 — transfer operators as compact schemes

Small, independent of everything else, and can be done at any time.

**Status: delivered** (August 2026). `src/transfer.jl` carries the pair and
the sampling plumbing; the measurements below were taken with
`bench/amr_transfer.jl` and are guarded by four serial testsets plus a
multi-rank section.

- The pair maps onto the existing types with no new solver machinery:
  restriction is a `CompactScheme` (tridiagonal LHS, Gaussian RHS) and
  prolongation a `BandedCompactScheme` with q = 2, its interior rows divided
  by the Gaussian center weight so the LHS diagonal is one (the `compact_d8`
  normalization; closure rows enter verbatim, since row scaling leaves the
  solution unchanged). The source's third tabulated closure row per end is
  the interior stencil, so each scheme carries two genuine closure rows and
  the minimum line extent is 5.
- The extended-data closure variant is tabulated identically to the
  one-sided one — the source says so ("same as one-sided to maintain
  invertibility") — so one closure set serves physical edges and patch
  interfaces alike. The ±1 symmetric variants are exactly the
  `lo_fold`/`hi_fold` ghost-folding algebra: a half line under a parity fold
  with mirror halos reproduces the closed full line on parity-extended data
  row for row (measured ≤ 1e-13, both schemes, both signs).
- **The 3:1 sampling convention (risk 1) is resolved.** Restriction filters
  the fine line and takes the coincident nodes (fine node 3m − 2 ↔ coarse
  node m; a closed line has n_f = 3n_c − 2, a periodic one 3n_c).
  Prolongation interpolates the coarse field — which is samples of the
  *filtered* field — onto the two intermediate fine nodes of each interval
  by Lagrange interpolation (exact rational weights, order 4/6/8, default
  6), then deconvolves. Measured: the pair round-trips through the live
  plans at 1.6e-15 closed / 2.7e-15 periodic (the 3.4e-15 dense figure);
  restriction is a **left inverse** of prolongation, so coarse → fine →
  coarse is exact (8.9e-16) for arbitrary data, which is what keeps levels
  consistent without refluxing; fine → coarse → fine converges at the
  interpolation order (measured 3.97 / 5.93 / 7.97), the subsampled content
  being unrecoverable by construction. A constant survives restriction to
  the last bit and prolongation to 13 ULPs.
- At a closed end the fine → coarse → fine error within 6 points of the
  boundary converges at ≈ 3 (the closure order, consistent with the Stage 2
  interface expectation) against 6 in the interior.
- **The sensor-injection risk (risk 3) measures small.** The smoothed δ⁴
  sensor of a 2h shock profile round-trips with amplification 1.03–1.13
  regardless of distance from the end — after the smoother there is no
  fine-Nyquist content left for the deconvolution to amplify, so the ~20×
  symbol bound is not approached. The state field itself undershoots by
  ≤ 3% of ambient (−2.9e-2 on a 0.5 jump), independent of boundary
  proximity, and the round-trip error decays ≈ 3.4× per point once outside
  the shock's own footprint: 6.1e-2 at the shock, 1.6e-4 by 12 fine cells,
  7.4e-7 by 21. A tagging buffer of ~4 coarse cells therefore drops
  interface pollution by two orders of magnitude.
- **The c4ff3 question is answered for now**: an explicit Gaussian pass
  ahead of restriction cuts the shock undershoot 2.8× (−2.9e-2 → −1.1e-2)
  but raises the total round-trip error (it smears the profile further), so
  it is not a default; it is the tool to reach for if Stage 4 regridding
  onto captured shocks proves positivity-limited.

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
  not fully specified by the public kernels; it was pinned down numerically
  (see the status block and [Risks](#risks-and-open-questions)). The working
  hypothesis held once "fill the intermediate nodes" was made concrete:
  restriction = filter on the fine line, then take every third node;
  prolongation = inject coarse values onto the coincident fine nodes, fill
  the intermediate nodes by Lagrange interpolation of the (filtered) coarse
  field, then run the deconvolution solve. `bench/amr_transfer.jl`
  reconstructs the operators and reproduces its 3.4e-15 round-trip figure
  through the `src/transfer.jl` types.

**Tests (gate for the stage — delivered; the measured values are in the
status block and asserted with headroom in `test/runtests.jl`).**

- Round-trip through the operator pair ≤ 1e-14 on any data (measured
  ~2e-15), and coarse → fine → coarse ≤ 1e-14 through the full sampled
  transfer. The fine → coarse → fine direction is finite-order, not exact:
  3:1 subsampling discards content above the coarse Nyquist, so this
  round trip is the "measured order" item below rather than a machine-zero
  identity.
- DC gain exactly 1 (constant in, constant out, to the last bit) in the
  interior and through each closure variant. Measured: bit-exact through
  restriction, 13 ULPs through prolongation.
- Measured order of the restriction/prolongation pair on a smooth field:
  the interpolation order, 3.97 / 5.93 / 7.97 at orders 4 / 6 / 8.
- The sensor-injection test named in the groundwork: apply the Cook δ⁴ sensor
  chain (`delta4_sum!` + `smooth!`) to data containing a captured-shock-like
  profile near a transfer boundary, prolong, and measure the amplification
  against the ~20× bound. Measured 1.03–1.13; the buffering and
  coarsening-filter conclusions are in the status block.

## Stage 2 — patch abstraction, single level

The long pole. Everything currently assuming one block per rank is
generalized to N same-level patches tiling the domain. This stage also
carries GPU Stage G0 (storage-type generalization), because both rewrite the
same struct definitions and doing them separately means touching every
signature twice.

**Status: delivered** (August 2026), with the scope notes below. `Solver`
holds configuration plus a vector of `Patch` objects (`src/patches.jl`), each
with its own communicator, `Decomp`, plans, folds and field arrays typed
`A <: AbstractArray{T,3}`; allocation goes through a backend object
(`CPUBackend` today), `Decomp{T}` types the halo and pair buffers, and
`viz.jl` extraction preserves the element type — the G0 items. A single-patch
solver forwards the patch-owned property names to its sole patch through
`Base.getproperty`, which is what kept the refactor behavior-preserving: the
hard gate passed with `test/convergence.jl` bit-identical down to the error
magnitudes, 64/64 serial testsets, 93/93 MPI checks at np 2/4/8, validation
values unchanged at four significant figures, and a net *reduction* of two
runtime-dispatch reports in `bench/jetcheck.jl` (a `Nothing`-plan method made
the dead fold branch statically resolvable).

Multi-patch findings, from the two-conforming-patch gates
(`test/patch_tests.jl` serial, `test/mpi_tests.jl` under rank partitioning):

- The extended-data interface treatment measured well. Gradient and filter
  plans take `interface_closures` rows — a derivative end closes with one
  explicit central row of the interior's formal order reading `M + 1`
  exchanged ghost layers (the LHS couples no ghost unknown, per Miranda's
  invertibility convention); a compact filter end leaves the shared plane
  node to the averaging and lets rows ≥ 2 read ghosts. The flux divergence
  keeps the scheme's own one-sided closures (`Patch.div_plans`), since ghost
  *fluxes* would need a second communication phase inside the RHS, which the
  sequential-patch execution order forbids; `gcl_cotr!` routes through the
  same plans so the discrete GCL cancellation is preserved.
- Measured: the entropy-wave manufactured solution converges across the
  interface at order 3.1–3.5 (the gate expected ≈ 3, the divergence's
  one-sided rows binding); an acoustic pulse crosses with reflected
  amplitude 2.3e-3 of incident at 192 points, converging at ≈ 5th order
  (6.5e-2 at 96, 4.9e-5 at 384); conservation drift over a long periodic run
  is 1.2e-8 relative against 4.5e-15 single-patch, so the drift term exists
  as predicted and no refluxing machinery is warranted yet.
- Rank partitioning (one `MPI.Comm_split`, ranks proportional to patch
  volume) reproduces the serial two-patch answer bit for bit at one rank per
  patch, where every interface transfer moves values unchanged and the
  arithmetic order is identical. Once a patch is itself decomposed the
  reproduction is round-off-level rather than bitwise (3.1e-15 at np = 4,
  6.7e-16 at np = 8, against a 9.5e-8 error signal): the patch's closed lines
  then take the spike/reduced path where the undecomposed patch ran a plain
  Thomas sweep. The step counts and dt sequences agree exactly at every np.
- Interface consistency is three mechanisms between stages, never inside a
  per-patch RHS (`sync_patches!`): shared-plane averaging, ghost refill,
  then each patch's own halo exchange, in that order, after every RK stage.
  A rank advancing several patches sequentially is why no cross-patch
  communication may sit inside `compute_rhs!`.

Scope notes, i.e. what Stage 2 deliberately did not generalize. The layout
generator tiles slabs along one dimension (`patch_grid` with a single entry
> 1), so corner-coupled patch adjacency does not yet arise; patched runs
reject folds (constraint 4), banded schemes (C10 interface closures are not
tabulated), the `:d8` detector, stretching along the patched dimension, and
an explicit `dims`; the artificial-property sensors are built per patch with
closed-edge clamping at interfaces (documented, not measured to matter at
this stage); and checkpoint/VTK output remains single-patch (a patch writes
its `region` exactly as a rank block does, so the HDF5 hyperslab extension
is mechanical when Stage 3 needs it).

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
solve closed at the face with rows whose left-hand side couples no ghost
unknown (the ghost values belong to the neighboring patch's solve), while
their right-hand side may read the copied ghost data. Stage 1's reading of
the source sets the precedent for the left-hand side: Miranda tabulates the
extended-data transfer closures identically to the one-sided ones, "to
maintain invertibility", even where ghost data is available. Whether each
solver scheme's interface rows keep one-sided right-hand sides as well or
read across the face is a Stage 2 accuracy measurement, made against the
manufactured-solution gate below. This is Miranda's `ghost.f90` shape. Interface exchange is a new
`exchange_patch_ghosts!` alongside `halo.jl`: for each face pair, a
`Sendrecv!` between the owning rank subsets; same-rank faces are copies.
Since the grids are node-coincident, abutting patches share the interface
plane nodes; after each RK stage the shared plane is made consistent by
averaging the two patches' values (cheap, symmetric, removes the drift mode).

Accuracy at an interface drops toward the closure order, as at any closed
boundary; the α^|i−j| decay confines it. Patch layouts should therefore
minimize interface count, taking patches as large as memory allows, which also
suits the GPU.

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

**Status: delivered** (August 2026), first cut, with one numerics finding
that revises the plan's transfer prescription. `src/levels.jl` implements one
level-1 patch over a user-given `refine::BlockRegion` at ratio 3, joining
`solver.patches` so the Stage 2 drivers advance it and the shared `max_rate`
reduction supplies the global dt. Ghost coupling is the Dirichlet-forcing
form: after every RK stage the fine patch's ghost ring *and* boundary-plane
nodes are overwritten from the coarse state over a 4-coarse-node buffered
box; after every step the fine state is written back onto the covered coarse
region, holding 2 coarse nodes off the boundary (`RESTRICT_MARGIN` — writing
to the boundary closes a feedback loop through the fine solution's
least-accurate nodes, measured gain ≈ 2 per step; the margin flattens it).

**The finding: the invertible transfer pair is not the right live coupling.**
Its contract is that prolongation input is samples of the *filtered* field —
which restricted data is, and the live coarse solution is not. Deconvolving
point samples sharpens data that was never smoothed, and filtered restriction
writes an attenuated representation into a field of point samples; both are
O(h²) against the solution, and the manufactured-solution gate measured
order 1.3–1.7 through the pair. The default coupling is therefore the
point-sample halves of the same machinery — order-6 Lagrange interpolation
up (`interpolate!`), coincident-node injection down (`inject!`) — which
measures **order 3.46/3.64** with errors three decades lower.
`level_restriction = :filter` keeps the anti-aliasing pair selectable, and
the pair remains what Stage 4 regridding initializes *new* fine regions
from, where the coarse data is all there is.

Measured gates (`test/level_tests.jl`): entropy-wave MMS across the boundary
8.4e-8 / 7.7e-9 / 6.2e-10 at N = 48/96/192 (orders 3.46/3.64, one-sided
divergence closures binding as at a Stage 2 interface); Sod crossing the
refinement boundary in both directions with Cook sensors and the state
filter live — ahead-of-shock interface noise 6.4e-10 in momentum against
1.9e-10 unrefined, positivity held throughout (the in-situ form of the
Stage 1 sensor-injection measurement); two-level composite mass drift
1.36e-4 relative over the full crossing (`:filter` halves it; unrefined
5e-11); and a 2-D refined region under diagonal advection at 4.3e-8,
exercising the tensor-product transfer chain.

Scope of the first cut, enforced at setup: **serial only** (the coarse-box
copy and restriction write-back are direct array copies; distributing the
level transfer is the follow-up that also lifts `plan_transfer`'s alignment
restriction), Cartesian metric, unstretched, no folds (constraint 4),
tridiagonal schemes, `:delta4`, one refined region, and no same-level
`patch_grid` alongside refinement. Each level filters its own state at the
shared cadence and computes sensors with its own h, as prescribed; the
combined-dissipation TGV measurement prescribed below remains open, tied to
the filter-calibration item in `ROADMAP.md`.

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

**Status: delivered** (August 2026), first cut, on the Stage 3 single-region
base and inheriting its scope (serial, Cartesian, unstretched, tridiagonal
schemes, `:delta4`, one refined region). `subcycle = true` selects the
Berger–Oliger step in `timestep.jl`; `regrid_interval = K` retags every K
coarse steps through `src/regrid.jl`.

- **Subcycling** is as prescribed: coarse step first with the fine level
  frozen, then three fine steps of dt/3 whose shell is imposed at every fine
  stage time from a cubic Hermite reconstruction of the coarse trajectory on
  the buffered box. One deviation from the plan text: the `t^{n+1}` endpoint
  RHS is *not* already computed — the next step's stage-1 RHS arrives too
  late and restriction invalidates a cached one — so it costs one extra
  coarse RHS evaluation per step, a fifth of a coarse step on the coarse
  level only. The global dt reduction divides each patch's rate by
  3^level, so dt is coarse-limited. Each level filters its own state at its
  own step cadence (the fine level inside the subcycle loop); under
  `filter_cfl > 0` the fine pass reads the coarse `dt · rate` product, an
  upper bound on its own CFL.
- **Measured (test/level_tests.jl):** the entropy-wave MMS across the
  boundary is unchanged from the global-dt coupling — 8.4e-8 / 7.5e-9 /
  5.9e-10 at N = 48/96/192, orders 3.49/3.66, so the Hermite boundary data
  does not bind — at a third of the steps. The Stage 3 Sod gate rerun
  subcycled *improves*: ahead-of-shock noise 5.7e-11 against 6.4e-10, mass
  drift 9.8e-5 against 1.36e-4.
- **Tagging** is the relative undivided fourth difference of the mixture
  density, Σ_d |δ⁴_d ρ|/ρ > `tag_threshold` (default 0.02), read from Q
  directly so it needs no primitives pass; buffered by `tag_buffer` (default
  4, the Stage 1 pollution-decay figure) and clamped to the nesting margin.
  Clustering is the bounding box of the tagged set — a single moving region,
  matching the one-fine-patch base; the plan's fixed-size tiles wait on a
  multi-patch fine level. An empty tag set keeps the current region.
- **Regridding** rebuilds the fine patch and `LevelTransfer` from schemes
  retained on a `RegridSpec`, initializes the new fine state by the order-6
  point-sample interpolation (the Stage 3 finding applies to fresh regions
  too: the coarse data are point samples, so the deconvolving pair stays on
  the shelf), and copies surviving fine data across the region overlap on
  the coincident lattice, one node off both boundary planes. The retry
  savepoint is refreshed at a regrid, since a snapshot with the old extents
  cannot be restored onto the new layout.
- **Measured, moving-region gates:** Sod at N = 201 coarse (601-node
  uniform-fine reference, t = 0.15, regrid every 5): the region tracks the
  shock and the composite density error against uniform-fine is 2.8e-3
  where the uniform-coarse baseline is 7.3e-2 — 26× — with fine resolution
  over a third of the domain. Shu–Osher at N = 267 coarse (799-node
  reference, t = 1.8):
  the region grows to hold shock plus wave train, and over the train the
  composite is 10× better in L∞ (2.8e-2 vs 2.8e-1) and 6.7× in L1 at 2497
  coarse steps against the reference's 4662.
- **Startup finding.** Subcycling widens `compute_dt`'s one-step lag to
  three fine substeps on one rate measurement, so a discontinuity *inside*
  the refined region at t = 0 fails at cfl = 0.4 where the global-dt mode
  survived — the global-dt mode was implicitly running the whole composite
  at the fine level's dt. cfl ≤ 0.2 or `StepControl(retries)` clears it,
  the same startup restriction `reference/CALIBRATION.md` documents.
  Chasing it exposed two latent rollback defects, both now fixed: a
  non-finite trajectory left NaN in the artificial coefficient arrays
  (which `max_rate` reads before the retry's first RHS) and in the
  low-storage `du` accumulator (`RKA[1] = 0` cannot forget NaN, since
  `0.0 · NaN = NaN`), so every retry re-failed at the savepoint.
- **Still open from this stage's gate list:** the wall-time and memory
  demonstration on a 3-D mixing case (1-D wave-train accuracy is measured
  above; the cost case needs the distributed level transfer first), tile
  clustering, and the TGV filter-cadence measurement of risk 4 under the
  per-level cadence chosen here.

The plan text for the stage, for reference:

- **Subcycling.** Ratio 3 in space gives ratio 3 in time: one coarse step,
  then three fine steps (Berger–Oliger order: coarse first, fine after, then
  restrict). Fine ghost data at fine stage times needs coarse values *between*
  coarse steps: store the coarse boundary rings (not full arrays) at t^n and
  t^{n+1} together with their RHS values, both already computed, and use
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
- Field allocation and the halo and pair buffers: concretely `Float64` today
  (`decomposition.jl`, where `field` is a bare `zeros(...)`), the known blocker
  for `Solver{T}` generality. Type them `T` and allocate them with the backend.
  `viz.jl` converts to `Array{Float64,3}` on extraction and has to follow. This
  also unlocks the mixed-precision item in `ROADMAP.md` and verifies the
  existing `{T}` parameterization.
- `Vector{Array{T,3}}` fields (`Y`, `D_art`) and `Matrix{Array{T,3}}`
  (`grad_u`, `grad_Y`, `flux`) become collections of `A`.

### G1 — pointwise kernels

**Status: delivered** (August 2026), first cut: the kernel architecture, the
CPU acceptance measurement, and the routing decision. Every pointwise phase
on the list below (Nasa9 primitives excepted, per the plan) is one shared
`@inline` per-point body launched through `pointwise!` (`src/pointwise.jl`):
`Array` storage takes the `@threaded` loop, any other storage a
KernelAbstractions kernel on `get_backend` of a representative field array —
one generic `@kernel` splatting the body, so there is no hand-maintained
second kernel body anywhere. KernelAbstractions is the hard dependency the
plan accepted (the three unused dependencies it was to arrive alongside had
already been removed by the time G1 landed).

- **The acceptance measurement went against KA-CPU**
  (`bench/pointwise_ka.jl`, 8 threads): at 64³ the KA CPU backend is 2.8×
  (assemble_fluxes!), 5.7× (primitives!) and 40–50× (RK update, scale_grad)
  slower than `@threaded` — per-launch task spawn with no work threshold, the
  same cost profile `threading.jl` documents for unconditional `@threads`.
  On the small 512×32 two-species case KA-CPU is instead *faster* (0.42× on
  assemble_fluxes!, 0.55× on primitives!), so the deficit is the launch
  policy, not the generated code. Per the plan's contingency, `Array`
  storage keeps `@threaded` and only device arrays route to KA; the branch
  resolves statically, and jetcheck/audit are unchanged probe for probe.
- **The KA path reproduces the threaded path bitwise** over full runs — the
  bodies are per-point independent, so this is an equality gate, not a
  round-off one. `FORCE_KA` routes ordinary arrays through the KA CPU
  backend for exactly this testset and the bench.
- **One trap found and fixed:** a `::Type{T}` argument inside `pointwise!`'s
  Vararg defeats Julia's specialization heuristics, turning the body call
  into a per-point runtime dispatch — measured as assemble_fluxes! at 9× its
  cost on *both* paths. Bodies take no `Type` arguments; the element type
  comes off an array.
- `outer_indices` was not dropped as the plan anticipated: with `@threaded`
  retained as the CPU path it survives — but in exactly one place, the
  launcher inside `pointwise!`, rather than at every loop site.
- **First device execution is measured** (RX 6800 XT, gfx1030, AMDGPU.jl
  v2.7.2 on Windows against the HIP SDK — the workstation is a real target,
  no cluster required). The plain-argument bodies run on device through
  `pointwise_ka!` and through `pointwise!`'s automatic routing on a
  `ROCArray`, and reproduce the CPU results **bitwise**: the RK update, the
  δ⁴ detector body, and the gradient scale all at max |gpu − cpu| = 0.
  Timing at 64³: the δ⁴ stencil runs 4× faster than 8-thread `@threaded`
  in a single launch (0.075 ms vs 0.302 ms), while the RK update's five
  per-component launches run 0.6× — `pointwise_ka!` synchronizes per call,
  so many small launches pay the round trip that G3's streams-and-residency
  work exists to remove. First-launch kernel compilation is ~9 s per body.
- **The argument adaptation is delivered and measured.** The field
  collections reach the bodies as `FieldVector`/`FieldMatrix`
  (`src/pointwise.jl`, built once per patch as `Patch.field_tuples`):
  zero-cost host wrappers whose only job is to carry the Adapt rule, so
  the `@threaded` path indexes exactly what it always indexed while a
  device launch adapts them to isbits `NTuple` mirrors. The gas-model EOS
  objects adapt the same way (`IdealMixtureCoeffs`/`StiffenedGasCoeffs`,
  physics.jl), since both carry name strings and `IdealMixture` carries
  `Vector` tables — a bare collection kernel argument *hangs* in
  kernel-argument adaptation rather than raising anything, which is why
  the wrappers are mandatory. Two designs were measured: holding the
  tuple on the host cost `assemble_fluxes!` 3× on the `@threaded` path
  (runtime tuple indexing in the species loops), so the tuple materializes
  only at launch. With it, the *full flux-assembly body* — every
  collection plus the mirrored EOS — runs on the RX 6800 XT bitwise
  against the CPU and 9.9× faster than 8-thread `@threaded` at 64³, two
  species (0.236 ms vs 2.33 ms, `bench/device_bringup.jl`).
- **Remaining on the G-track before a whole solver lives on device:** G3a
  delivered the plan conversion and residency work (`max_rate`,
  initialization, boundaries, folds; see its status block), so what remains
  is G3b's halo/MPI path, the launch synchronization policy (G3d), and the
  `Nasa9Mixture` fixed-width mirror. Device allocation and operator-level
  line solves are delivered with G2 below; this paragraph previously still
  listed both as future work and is corrected here.

The plan text for the stage, for reference — convert the phases that are
pointwise loops, per `bench/phases.jl` the majority of the budget outside
the line solves:

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

**Status: delivered** (August 2026) at the operator level: `device_plan`
mirrors a host `DirPlan`/`BandPlan` onto a KernelAbstractions backend as a
`DevicePlan` (`src/lines_device.jl`), applied through the same `apply_along!`
entry point on fields living on that backend.

- **Layout**: every dimension uses the (lines × n) layout on device — one
  thread per line for the elimination sweeps, one thread per point for the
  fill, scatter, and spike-correction kernels, so consecutive threads touch
  consecutive memory at every sweep position. The plan's permute-vs-second-
  kernel question for the x sweep resolved to neither: the (lines × n) fill
  reads the field strided for dim 1, but the sweep is the O(n)-pass stage,
  so the sweep-coalesced layout is the only one implemented — and the
  measured dim-1 applies (0.71–0.74 ms at 64³) sit inside the dim-2/3
  spread, so no second layout is warranted at this size.
- **The reduced 2qP × 2qP interface stage stays host-side** as planned: a
  pack kernel writes the 2q interface values per line in the layout of
  `line_solver.ends`, one device-to-host copy feeds `_reduced_solve!` — the
  stage factored out of the two host layouts (`tridiag.jl`, `banded.jl`), so
  host and device run literally the same Allgather + prefactorized `ldiv!`
  code — and one host-to-device copy returns the per-line correction values.
  Collective ordering is identical to the host path, and the factoring
  merged the two JET-visible `ldiv!` dispatch sites on the `Any`-typed
  reduced LU into one (jetcheck deriv_along! 2 → 1, same site).
- **The comparison gate is bitwise, host convention per dimension.** The
  device kernels mirror the host arithmetic per line operation for
  operation. The two host layouts themselves disagree in the banded solve —
  the x sweep divides by the diagonal where the transposed y/z sweeps
  multiply by its inverse, and they accumulate the spike correction
  differently — so the kernels carry a `colwise` switch selecting the
  convention of the plan's dimension. With it, C6, C10, the C8 filter, the
  explicit Gaussian filter, and the d8 detector reproduce the host plans
  **bitwise** over closed edges, folds, and periodic wrap in all three
  dimensions: on the KA CPU backend (serial testset + a decomposed MPI
  testset that pins the cross-rank Allgather through the device staging),
  and on the RX 6800 XT itself (`bench/device_bringup.jl`), where every
  case is also max |gpu − cpu| = 0.
- **Timing at 64³ is a correctness milestone, not a win yet**: 0.68–0.80 ms
  per apply on device vs 0.31–0.96 ms for the 8-thread host path. Each apply
  is five synchronized launches plus the host round trip, and 64³ offers
  only 4096 lines of parallelism; this is the launch-latency profile G3's
  residency and stream work exists to remove, with the one already-won case
  (C10 dim 1: 0.73 vs 0.96 ms) showing where the sweeps themselves stand.
- `DeviceBackend` wraps a KA backend behind `field(backend, decomp)` and
  `allocate_state` — the allocation half of the seam. When G2 landed, a full
  `Solver` on a `DeviceBackend` was not yet supported; G3a has since
  delivered that for one patch on one rank (see its status block), leaving
  the distributed path to G3b and the `Nasa9Mixture` fixed-width mirror
  open.
- **FP64 vs FP32, measured on the RX 6800 XT at 64³** (RDNA2 vector FP64 is
  1/16 the FP32 rate, so an ALU-bound kernel would show a ratio near 16):
  the launch-bound pointwise phases run at 1.03–1.10×, the line solves at
  1.15–1.29×, and the compute-heaviest body (flux assembly, two species) at
  1.73× — everything is launch- and bandwidth-bound, and the FP64 ALU
  penalty is invisible at this size. FP32's payoff approaches a clean 2×
  only once G3 removes the launch floor. The attempt also showed the code
  is not yet FP32-clean: a Float64 `2/3` literal in the viscous stresses
  made the `τ` tuple heterogeneous under `Float32` and its runtime indexing
  an `InvalidIRError` on device (fixed, `rhs.jl`); bodies not yet run in
  FP32 (primitives, the artificial-property loops) may hide the same
  pattern, which the mixed-precision item in `ROADMAP.md` should sweep.

The plan text for the stage, for reference:

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
- Halo/ghost exchange: pack/unpack kernels into contiguous device buffers.
  A backend-specific capability query decides whether MPI may receive that
  device's pointers directly; otherwise stage through pinned host buffers.
  CUDA's capability query is one implementation, not the portable interface:
  the measured workstation target is AMD, and an MPI stack may support one
  device runtime without supporting another. Runtime detection, no build-time
  switch.

### G3 — residency and patches as launch units

G3 is four gates, not one porting pass. The earlier three-bullet version named
the desired end state but omitted the work between the operator-level G2 proof
and a supported `Solver` on `DeviceBackend`.

#### G3a — one device-resident patch

**Status: delivered** (August 2026). `Solver(backend = DeviceBackend(ka))` is
a supported configuration for one patch on one rank: construction converts
every compact plan — fold parity plans included — to a `DevicePlan` through
`backend_plan`, allocates every patch field, the pair and ring scratch, and
the conserved state on the backend, and each phase between the step drivers
and the arrays runs as a launchable body, a `DevicePlan` solve, or an exact
storage-level reduction.

- Converted to launchable form in this gate: wall-plane enforcement (slip,
  no-slip, extrapolation, through `plane_pointwise!`), both NSCBC plane
  corrections (the inflow's constant composition target materializes as a
  tuple at launch, exactly as the field collections do), the fold mirror
  fills, the local pair butterflies and the parity-select pass
  (`_pair_slot`), the d8 detector accumulation, `gcl_cotr!`,
  `copy_interior!` / `blend_interior!`, and the periodic halo wrap (a
  broadcast slab assignment on device storage). `max_rate` evaluates the CFL
  rate through a pointwise body into `tmp_a`/`tmp_b` and reduces with the
  storage's own `maximum`/`minimum`; max and min are exact, so the launch
  path agrees with the fused serial loop bitwise, and `FORCE_KA` pins that
  on ordinary arrays.
- **Two traps found on the first device runs**, both invisible on the KA CPU
  backend. A kernel argument of the un-adapted `ConservedState` wrapper does
  not lower, so the wrapper now carries an Adapt rule and reaches kernels
  wrapping the device-side array. And a splatted kernel-argument tuple longer
  than 32 elements lowers through the dynamic apply — an `InvalidIRError` on
  device — which the NSCBC bodies hit at 36 arguments; their scalars now ride
  in small tuples. Treat 32 as the body-argument budget.
- Host-staged on purpose, per the plan's I/O rule: geometry fills host
  mirrors once at setup and uploads; `initialize!` and `DirichletBC` evaluate
  their host closures into staging blocks (setup-time, and one plane upload
  per stage respectively).
- The conservative synchronized launch mode is retained throughout:
  `pointwise_ka!` and the `DevicePlan` stages synchronize as in G1/G2.
  Removing synchronization is G3d work and a correctness change. The
  vendor-profiler transfer audit is deferred to G3d for the same reason: at
  the measured launch-bound floor it would time synchronization, not overlap.
- Guarded at setup rather than failing mid-run: more than one rank or patch
  (G3b), `refine` (G3c), `Nasa9Mixture` (needs the fixed-width mirror), an
  NSCBC inflow with a pointwise `target`, `StepControl.floor_ratio > 0`, and
  `dt_report` (both host sweeps).
- **The comparison gate came out bitwise, on both backends.** On the KA CPU
  backend (`test/device_tests.jl`), full runs on
  `DeviceBackend(KernelAbstractions.CPU())` under `FORCE_KA` reproduce the
  `CPUBackend` solver exactly: multispecies closed/periodic tube, NSCBC duct
  with isothermal no-slip wall and relaxed filter, resolved-θ axis fold,
  Dirichlet-forced face, spherical origin+poles freestream RHS, `max_rate`,
  and a Float32 run. On the RX 6800 XT (`bench/device_solver.jl`) every
  gate case is max |device − cpu| = 0 over full runs: periodic smooth 32³,
  the closed multispecies tube, the NSCBC duct, the resolved-θ axis fold,
  spherical freestream/GCL (|dQ|max 5.7e-14 on both), a 578-step Sod
  validation at N = 400, and 20-step 64³ TGV histories in Float64 and
  Float32. The GPU-verification section's expectation that bit-exactness
  would be lost to reassociation did not materialize: every phase is either
  per-point independent, a host-mirrored sweep, or an exact max/min.
- **Timing at 64³ TGV, warm** (RX 6800 XT vs 8-thread CPU): device
  0.188 s/step vs 0.093 (Float64), 0.168 vs 0.072 (Float32) — 0.4–0.5× CPU.
  This is the launch-latency profile G2 predicted, now measured at whole-step
  granularity under the synchronized mode: a step is hundreds of small
  synchronized launches plus the reduced-solve round trips. G3d owns removing
  that floor; G3a is the correctness milestone.

The plan text for the gate, for reference:

- Convert every host `DirPlan`/`BandPlan` to a `DevicePlan` while a patch is
  constructed on `DeviceBackend`; G2 exposes the conversion but `Solver`
  construction does not yet select it.
- Move initialization, physical boundary conditions, NSCBC, fold fills and
  pair operations, and `max_rate` onto launchable bodies/reductions. Start with
  Cartesian periodic and closed-edge cases plus `IdealMixture` and
  `StiffenedGas`; the `Nasa9Mixture` fixed-width mirror is a later gate.
- Replace the unconditional synchronization in `pointwise_ka!` and between
  the line-solve subkernels with returned events and explicit dependencies.
  Keep a conservative synchronized mode until the full-step comparison is
  green; removing synchronization is a correctness change, not merely tuning.
- Audit with the vendor profiler that a full step performs no host transfers
  except the compact reduced solves and the global timestep reduction. I/O
  gathers to host explicitly; no change to the HDF5/VTK design.

Gate: an end-to-end single-patch run on device, not isolated bodies: periodic
and closed-boundary smooth convergence, one shock validation, freestream/GCL,
and TGV history against CPU, first in Float64 and then under G4's accepted
Float32 mode.

#### G3b — distributed device communication

- Pack/unpack halos and fold-pair slabs on device. Use direct device-pointer
  MPI only when the active array backend and MPI library report that exact
  capability; otherwise use reusable pinned host staging.
- Make `max_rate` a componentwise reduction of `(rate, -rho_min)` on device,
  followed by the existing world `Allreduce`. Do not rely on tuple ordering:
  the monoid is componentwise maximum.
- Preserve the compact reduced solve on the host initially, but batch and
  profile its device/host transfers before considering backend-specific
  device collectives.

Gate: the existing 2/4/8-rank operator and full-run comparisons on device,
with transfer volume and synchronization time recorded separately.

#### G3c — device-resident AMR

- Generalize `LevelTransfer`'s host-only `Array` stages and Hermite trajectory
  storage through the backend seam. Convert coarse-box copies, fine-shell
  writes, restriction, tagging, overlap copies, and regridding initialization
  from scalar host loops to kernels or backend copies.
- Distribute the level transfer, which is already the prerequisite for the
  Stage 4 three-dimensional cost demonstration. A `DeviceBackend + refine`
  configuration is unsupported until this gate lands; single-patch G3a does
  not imply it.

Gate: static and moving-region AMR comparisons against the CPU composite and
uniform-fine references, followed by the deferred 3-D wall-time/memory case.

#### G3d — patches as asynchronous launch units

- Give each concurrently executable patch a queue/stream and express halo,
  shared-plane, coarse/fine, and reduced-solve dependencies with events.
  Coarse and fine patches are not automatically independent under subcycling;
  overlap only branches whose dependency graph permits it.
- Remove host synchronizations only where a profile shows useful overlap.
  Record full-step improvement rather than isolated launch savings.

Gate: profiler evidence of overlap and no transfers beyond the documented MPI,
reduced-solve, and I/O staging. One stream remains the correctness fallback.

### G4 — Float32 and mixed precision

Two milestones are deliberately separate:

**Status: G4a in progress** (August 2026). The public `Problem`/`Numerics` path now
accepts `Transport{Float32}` and `ArtParams{Float32}`, `single_species(Float32)`
constructs typed EOS coefficients, and a serial test exercises
`setup -> compute_dt -> compute_rhs! -> run!` with Float32 state, schemes,
operators, artificial properties, RK arithmetic, and timestep reduction. The
first literal audit made the positive division floor representable, typed the
ideal/stiffened primitive bodies and Cook sensor bodies, and removed the RK and
`max_rate` Float64 promotions. The next matrix slice is also delivered: C6
accuracy is guarded down to its measured Float32 roundoff floor, the NASA-9
Newton inversion and constant-cp constructor are typed, a closed SlipWall
stiffened-gas step is finite, and static two-level AMR advances with Float32
state and timestep throughout. A third slice adds varying-cp multicomponent
NASA-9 recovery, typed NSCBC and isothermal no-slip state, a resolved-angle
axis fold, typed Hermite subcycle coefficients and regrid sensor
accumulation, and a moving subcycled/regridded Sod smoke test. The measured
serial gates are finite and positive throughout; the full serial suite passes.
A fourth slice delivers the test-scale CPU validation matrix in
`test/float32_validation.jl`: Cartesian, stretched, cylindrical, axis-fold
and spherical freestream/GCL; closed-wall and spherical smooth convergence;
an exact-Riemann Sod profile with mass drift below 1e-5; and a short TGV
kinetic-energy history whose Float32/Float64 difference stays below 1e-7.
It also makes spherical angular-domain validation respect the
solver type's representation of π instead of rejecting `Float32(π)` against
a fixed 1e-10 tolerance.

The fifth slice completes the CPU measurement with
`bench/tgv_energy.jl precision=both`. On the 24-thread workstation, the
published-reference 64³, t = 10, art-off TGV histories peak at 1.2471e-2
(`Float64`) and 1.2472e-2 (`Float32`), both at t = 8.93, versus the
reference 1.2e-2 near t = 9. Solver wall time is 621.41 s versus
563.52 s (1.10× speedup), while the measured solver/state/RK footprint is
219.5 versus 109.8 MiB (2.00× smaller). Mean-density drift is
7.49e-13 versus 1.39e-4. The same comparison at 32³ gives identical peaks
at the printed precision, 1.06× speedup, and
the same 2× memory reduction.

That accepts the CPU numerical and memory gates but does not justify making
Float32 the CPU default: its speedup is modest and its conservation drift is
material. Uniform Float32 remains opt-in, and the final G4a acceptance gate is
the same end-to-end history on a resident GPU patch after G3a. G4b must use
that device measurement when deciding whether any explicit Float64
accumulation role pays for itself; the diagnostics here already accumulate in
Float64, so diagnostic precision alone does not remove state drift.

Two literal-audit items remain open from item 1 below. The Ducros epsilon is
still the Float64 `DUCROS_EPS` converted to the state type at its point of
use (`compression_switch`, artificial.jl), not the separately measured value
the item calls for; the conversion is safe (1e-32 is representable in
Float32, and the compression branch guarantees a nonzero denominator) but is
an interim choice. The stretch-map closures built by `sine_cluster` evaluate
in Float64 internally, so `_phys_and_jac` returns Float64 on a stretched
Float32 grid; this runs once at setup and converts at store, and the
stretched Float32 freestream case passes, but it belongs to the same
literal sweep.

1. **Uniform Float32 execution.** Expose Float32 through `Problem`/`Numerics`,
   use Float32 schemes and EOS coefficients, and remove accidental Float64
   promotion from pointwise bodies, RK updates, sensors, geometry floors, and
   timestep reductions. Replace each small constant according to its meaning:
   typed zero/one for algebra, a representable positive floor for division,
   and a separately measured Ducros epsilon. A blanket cast is not a policy.
2. **Mixed precision.** After the uniform run is measured, choose explicit
   roles such as `Tstate` and `Taccum` (and only add a distinct plan/geometry
   type if measurements justify it). Candidate Float64 operations are global
   reductions, conservation diagnostics, simulation time, and possibly the
   reduced compact interface solve. State/RHS fields and the bulk kernels are
   the Float32 candidates. The choice must be representable in types rather
   than arising from Float64 literals in otherwise Float32 code.

Gates: CPU Float32 setup and a full run first; then the same GPU convergence,
freestream, validation and TGV gates as Float64. Record conservation drift and
positivity behavior on the AMR shock cases, phase timings, memory, and the
accuracy/performance delta for every candidate mixed policy. Select a shipped
policy only from that matrix; retaining Float64 accumulation is acceptable,
but it must be intentional and tested.

### GPU verification

Bit-exactness against CPU is not expected (different reassociation). Gates:

- `test/convergence.jl` measured orders identical to the guard table.
- Freestream preservation stays machine-zero in every metric: the discrete
  GCL cancels the same stored operator output node-by-node, which is
  arithmetic-order independent, so this gate is exact even on device.
- `test/validation.jl` guards matched at the documented comparison level
  (four significant figures, a moved third digit is real).
- The Taylor–Green vortex (TGV) −dKE/dt(t) history overlaid on the CPU
  curve.
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
G3a (one-patch residency)       — after G2
G4a (uniform Float32)           — begins on CPU; device gate follows G3a
G3b (device MPI)                — after G3a
G3c (device AMR)                — after G3b + distributed level transfer
G3d (streams/overlap)           — last, after residency is measured
G4b (mixed policy)              — selected from G3/G4a measurements
```

Do Stage 2 only after the regularization debts in `ROADMAP.md` settle: the
refactor touches every file, and rebasing physics changes across it is the
expensive order.

## Risks and open questions

1. **The 3:1 sampling convention** — resolved by Stage 1 (see its status
   block): filter-then-subsample / interpolate-then-deconvolve, with
   restriction a measured left inverse of prolongation. The remaining
   freedom is the interpolation order (default 6), which sets the
   fine → coarse → fine order and nothing else.
2. **Conservation at interfaces.** The transfer operators preserve the mean
   (unit DC gain), and restriction-over-covered-region keeps levels
   consistent, but the compact divergence with closure rows at an interface
   does not telescope exactly, so global conservation acquires a drift term
   where a finite-volume code would reflux. Measure it in Stage 2 gate 2 and
   Stage 3; the shared-plane averaging removes the worst mode, and the
   fallback is a surface correction on the interface fluxes. Do not build
   refluxing machinery speculatively.
3. **Deconvolution amplification** (~20× at fine Nyquist) interacting with
   Cook sensors near interfaces — sized by the Stage 1 sensor-injection
   measurement and found small: the smoothed sensor round-trips at ≤ 1.13×,
   and shock round-trip pollution decays ≈ 3.4× per point outside the shock
   footprint. The mitigation that remains live is keeping refinement
   boundaries a ~4-coarse-cell buffer away from tagged features; the
   coarsening filter ahead of restriction is measured (it trades accuracy
   for undershoot margin) and held in reserve, and SBP–SAT stays the
   fallback of last resort.
4. **Filter cadence across subcycled levels.** A fine level filtering every
   fine step filters 3× as often per unit time as its parent, and the
   filter's dissipation is per-application. This is the same dt-consistency
   defect already on the roadmap for the single-grid code, sharpened by
   subcycling. Stage 4 picked the convention — each level filters at its own
   step cadence, which is what an unrefined run of either level would do,
   and is dt-consistent whenever `filter_cfl > 0` — and measured it on the
   Sod gates only; the TGV measurement remains open, tied to the roadmap's
   filter-calibration item.
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
   problems need resolution *at* the axis/origin, which is exactly where
   refinement is excluded. The mitigation is that level 0 can be globally
   fine in r near the fold (stretching is also excluded on folded dimensions,
   so this costs points); lifting the restriction is future work that
   requires extending the antipodal pairing across levels.
