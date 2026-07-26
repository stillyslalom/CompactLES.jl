# CompactLES — Roadmap

Where this code sits relative to the codes it will be compared against, and what
it would take to get from shock tubes to NIF-relevant multiphysics. `README.md`
covers usage, `DESIGN.md` the numerics, `CLAUDE.md` how to work on it.

## Contents

1. [Positioning](#positioning)
2. [Comparison](#comparison)
3. [What actually blocks the target use cases](#what-actually-blocks-the-target-use-cases)
4. [Adaptivity — the compact-scheme constraint](#adaptivity--the-compact-scheme-constraint)
5. [Phase 0 — the extensibility seams](#phase-0--the-extensibility-seams-complete-july-2026)
6. [Phase 1 — make shock tubes unimpeachable](#phase-1--make-shock-tubes-unimpeachable-largely-complete-july-2026)
7. [Phase 2 — HED physics](#phase-2--hed-physics)
8. [Phase 3 — scale, portability, and adaptivity](#phase-3--scale-portability-and-adaptivity)
9. [Non-goals](#non-goals)
10. [Suggested ordering](#suggested-ordering)

## Positioning

It is worth being blunt about what CompactLES already is, because it determines
which comparisons are informative and which are category errors.

CompactLES offers tenth/sixth-order compact Padé
derivatives, a Gaitonde–Visbal compact filter, Cook artificial fluid properties
in place of Riemann solvers, five-stage low-storage RK45, structured curvilinear
grids, MPI. That is, item for item, the Miranda/Pyranda recipe — minus
patch-based adaptivity and a GPU backend. This is not a
criticism — it is the correct recipe for variable-density turbulent mixing and
shock–interface interaction, which is exactly the shock-tube and RM/RT physics
in the primary use case. It does mean that the interesting comparison with
Pyranda is about *scope and interface*, not numerics, and that the comparison
with FLASH is about *physics catalogue*, not accuracy.

Three things here are genuinely differentiated today and worth protecting in
every subsequent decision:

- **The regularized coordinate-singularity treatment.** The half-offset grid
  plus parity/antipodal folds for the cylindrical axis, spherical origin, and
  spherical poles — with a discrete GCL that preserves freestream to machine
  zero — is more than Pyranda or Miranda expose, and nothing in the Julia
  ecosystem has it. Converging-shock and spherical-implosion geometry is
  directly NIF-relevant, and this is the piece that is hard to rebuild.
- **The genuinely distributed compact solve.** The spike/reduced-interface
  banded solve reproduces the single-domain answer bit-for-bit at any rank
  count, with the reduced system factorized once at plan time. It is also, as
  argued below, the seed of the implicit infrastructure the HED physics needs.
- **The `Problem`/`Numerics` split.** Physics specification that never mentions
  ranks, halos, or the conserved layout. Every multiphysics code eventually
  wants this and almost none of them get it retroactively.

## Comparison

### Pyranda (LLNL)

The closest relative — the Miranda mini-app, Python-driven with a Fortran
kernel, 10th-order compact plus RK45, MPI, aimed at arbitrary hyperbolic
systems.

| | Pyranda | CompactLES |
|---|---|---|
| Spatial scheme | C10 compact | C6 default, C10, custom `CompactScheme` |
| Regularization | artificial bulk viscosity | Cook μ\*, β\*, κ\*, per-species D\* |
| Problem specification | Python DSL over symbolic PDE strings | typed `Problem` / `Numerics` |
| Geometry | Cartesian, curvilinear, immersed boundaries | Cartesian, cylindrical, spherical with regularized singularities; stretch maps |
| Lineage | Miranda validation heritage | validated against analytic references only |

Pyranda is also the most useful *reference implementation* available here, for a
reason unrelated to its own feature set: it carries Miranda's Fortran kernels in
`pyranda/parcop/`, including operators Pyranda itself does not expose. The
adaptivity section below is read directly from those, and the same source is
worth consulting before implementing anything else in this scheme family.

**What Pyranda has that CompactLES does not:** the DSL, which lets a user write
a new PDE system in ten lines without touching the solver; immersed boundaries,
which is how you get non-coordinate-surface geometry without unstructured
meshes; and the credibility of being the mini-app for a production code.

**The DSL is the interesting one, and it is a fork in the road.** Pyranda's
string-based equation interpreter buys generality at the cost of type safety and
of any physics that is not a flux divergence plus a source. Julia's answer to
the same problem is multiple dispatch on an equation-set type — which is what
Trixi does, and what Phase 0 below proposes. That path keeps inference and
allocation discipline (`bench/audit.jl`, `bench/jetcheck.jl`) while getting most
of the extensibility. If a symbolic frontend is later wanted for ergonomics, the
route is a macro that *emits* an equation set at parse time — Pyranda's brevity
with none of the runtime interpretation. What is excluded is evaluating equation
strings at runtime, not metaprogramming.

Immersed boundaries are worth considering later, but note the tension: a sharp
IB cut cell is fundamentally at odds with a line-global compact operator, which
is why IB in this scheme family is usually done as a smeared Brinkman penalty
term — i.e. as a *source term*, which Phase 0 provides for free.

### FLASH

Category error to compare on numerics; FLASH's hydro is second-order unsplit
PPM. The comparison that matters is the physics catalogue and AMR.

FLASH's HEDP capability is a set of units: 3T hydrodynamics with electron–ion
equilibration, multigroup radiation diffusion, tabulated multi-species EOS and
opacities (IONMIX), electron thermal conduction through an implicit diffusion
solver, laser energy deposition by geometric-optics ray tracing, MHD, and
anisotropic magnetized transport coefficients — all on a block-structured AMR
mesh, validated against HYDRA. That is the thing to aim at for NIF work, and it
is roughly a decade of physics implementation.

Two structural lessons, both to copy, but only one of them the way FLASH does it:

- **Copy the unit decomposition.** FLASH's physics units are separately
  configurable, each owning its own state and contributing to the update through
  a defined interface. This is why FLASH could grow 3T and MGD without
  rewriting hydro. CompactLES's equivalent is Phase 0.
- **Copy adaptivity, but patch-based, not octree.** FLASH's PARAMESH/AMReX
  refinement is oct-tree block AMR built around a second-order finite-volume
  update. The version that belongs here is *patch-based* (logically rectangular
  blocks of uniform resolution, SAMRAI-style), which is what Miranda uses and
  what makes adaptivity compatible with a compact scheme. See
  [Adaptivity](#adaptivity-the-compact-scheme-constraint) for why the
  distinction is the whole argument, and Phase 3 for the plan.

### Trixi.jl

The Julia reference for high-order conservation laws: nodal DG-SEM on a
quad/octree with AMR, `p4est` for unstructured curved meshes, entropy-stable and
kinetic-energy-preserving split forms, shock capturing with positivity limiting,
a large equation-set catalogue (Euler, MHD, shallow water, hyperbolic diffusion
for self-gravitating gas dynamics), and OrdinaryDiffEq.jl integration.

**What to learn from it:** the `equations` type parameter. Trixi's solvers are
generic over an equation set that owns the variable count, names, flux
functions, and conversions. Everything — the DG operator, AMR, the time
integrator, the output — is written against that interface. It is why adding
MHD to Trixi did not mean rewriting Euler. This is the single most transferable
idea in the comparison set and it is Phase 0 item 2.

**Where CompactLES stays ahead for the target problems:** compact FD is far
cheaper per degree of freedom than DG at comparable resolving power for smooth,
volume-filling turbulence, and the memory traffic profile is much friendlier.
Trixi has nothing resembling Cook artificial fluid properties, which is the
right regularization for material interfaces at high Atwood number — DG's
entropy-stable limiting is built for robustness at discontinuities, not for
controlled subgrid dissipation in a mixing layer. And Trixi has no regularized
polar/spherical singularity treatment.

**Where to consider borrowing rather than competing:** Trixi's use of
OrdinaryDiffEq.jl for time integration. If CompactLES grows an IMEX scheme
(Phase 2), the SciML ecosystem already has well-tested IMEX-ARK tableaus.

### XCALibre.jl

Second-order unstructured finite volume, incompressible and compressible, RANS
and LES, OpenFOAM/unv mesh import, CPU threads or GPU through
KernelAbstractions.jl. Different accuracy class and a different problem domain
(engineering CFD on complex geometry), so it is not a competitor.

It is, however, **the best available template for the GPU question**. A single
KernelAbstractions.jl codebase targeting NVIDIA, AMD, and Intel, with the CPU
path retained, is exactly the architecture Phase 3 wants — and XCALibre
demonstrates it is achievable by a small team in Julia. Also worth watching
Oceananigans.jl, which is the strongest example of a Julia structured-grid
solver that went GPU-first without losing readability.

## What actually blocks the target use cases

Everything below follows from the remaining architectural gaps. Phase 0 closed
the source/layout/storage seams in July 2026; the implicit solver and EOS
generalization remain blockers.

**The source and layout seams now exist.** `Solver.sources` is an inferable tuple
applied at RK stage time, and `NavierStokes1T <: EquationSet` owns the component
indices, names, conserved conversion, and fold parity. The former hard-coded
`n_species + 4` sites are gone. A 3T model still requires its physics, but no
longer requires rediscovering the conserved layout throughout the solver.

**There is no implicit or elliptic solver.** Radiation diffusion and electron
conduction are parabolic and stiff; explicit treatment is not an option at HED
conditions, where the conduction timestep can be orders of magnitude below the
acoustic one. This is the largest genuine gap.

*But the raw material is already here.* `tridiag.jl` and `banded.jl` implement a
distributed banded line solve with cross-rank coupling and a pre-factorized
reduced interface system. That is precisely the kernel an ADI or line-relaxation
diffusion solver needs, and precisely the smoother a geometric multigrid wants.
The path from "no implicit solver" to "implicit diffusion" is much shorter here
than it looks from outside, and it should be built on this machinery rather than
by bolting on a black-box linear algebra dependency.

**The EOS contract is now agnostic; one physical assumption survives it.** The
sites that quietly assumed ideal gas outside the function barrier — NSCBC's LODI
algebra and the artificial-conductivity scale — are EOS dispatch points as of
July 2026, and `StiffenedGas` and `Nasa9Mixture` exercise them. A SESAME reader
needs data plumbing and a table interpolator, not solver surgery.

The assumption that survives is physical rather than structural: the artificial
conductivity is still built as (ρc/T_ion)·sensor for every gas model here, which
is singular at a cold ambient. A tabular EOS is free to supply something else,
and a condensed-matter one will have to.

## Adaptivity — the compact-scheme constraint

Adaptivity is in long-term scope, and the constraint it imposes is specific
enough to be worth stating before the phases, because it shapes Phase 3 and
touches the correctness criterion the test suite is built on.

**The constraint is not "no AMR," it is "no line cut mid-solve."** A compact
operator couples an entire grid line through a banded LHS, so what it cannot
tolerate is a resolution change *inside* a solve. Patch-based AMR does not do
that: each patch is a logically rectangular block of uniform resolution, and the
compact solve runs over the patch with its ghost layers filled by copy from
same-level neighbors or by interpolation from the underlying coarse level.
Oct-tree cell-by-cell refinement is the incompatible one. This is the
distinction that matters, and the earlier framing here collapsed the two.

**What is actually given up is exactness, not formulation.** Today the
spike/reduced-interface solve reproduces the single-domain answer bit-for-bit at
any rank count — which is not just an aesthetic property, it is the *correctness
oracle* the MPI suite depends on. An interface bug produces O(1) error rather
than a small one, which is precisely how `test/mpi_tests.jl` catches it. Under
patch AMR, coarse-fine boundaries make the solve approximate by construction,
and that oracle stops working there. Any adaptivity work must bring its own
replacement — most plausibly a manufactured solution across a refinement
boundary with a measured order, in the style `test/convergence.jl` already uses.

**These two mechanisms compose rather than compete.** The natural design is to
keep the exact distributed solve *within* a refinement level, where neighbors
are at uniform resolution and the existing sub-communicator machinery already
applies, and to localize the approximation to coarse-fine interfaces only. That
preserves the bit-exact guarantee across the majority of patch boundaries and
confines the new error to where it is unavoidable.

**Why the error stays local.** The reassuring quantitative fact is that the
inverse of a compact LHS decays geometrically away from the diagonal — roughly
α^|i−j|, so α = 1/3 for C6. Pollution injected at an inexact patch boundary
decays by a factor of three per point along the line rather than contaminating
it globally. This is worth measuring early regardless of which route below is
taken, because it quantifies how much an approximate interface actually costs,
and it is an afternoon's experiment.

### Interface treatment: what the literature offers

The interface treatment, not the grid arrangement, is what actually decides this
design. Two answers exist: the one with proofs, below, and the one Miranda
actually uses, in the section after it. Read both — the second is the
recommendation, but the first is what to fall back on and what to measure
against.

The mature answer to "how do you couple high-order finite-difference blocks at
an interface with provable stability and conservation" is
**summation-by-parts operators with simultaneous-approximation-term (SAT)
coupling**, and it is a thirty-year literature rather than a recent idea. The
relevant chain, specifically for the *implicit/compact* case:

- Carpenter, Gottlieb & Abarbanel (1993) established stable, accurate boundary
  treatments for compact high-order schemes, with Abarbanel & Chertock (2000)
  following on strict stability and the role of boundary conditions.
- Mattsson & Rydin (2022) derived **implicit SBP operators** for first and
  second derivatives with boundary closures on a *banded-norm* SBP framework and
  weak (penalty) boundary enforcement, reaching 8th-order global convergence.
  This is the Padé-class case and it is the directly load-bearing reference.
- Nissen et al. (2015) built block-oriented adaptive grids on SBP–SAT, including
  a stable treatment of junction points where interfaces of different type meet.
- Almquist & Dunham (2018) supplied order-preserving interpolation operators for
  **non-conforming** (refined) interfaces, which is the coarse-fine case.

Worth knowing regardless of route: **CompactLES's current closures are not SBP.**
The `ClosureRow` cascade in `kernels.jl` implements the classical Lele one-sided
closures — stable in practice, without an energy estimate. Moving onto an SBP
footing would be a deeper change than anything else in this document, touching
`kernels.jl`, the closure rows, `operators.jl`, the folds, and every order in the
`test/convergence.jl` guard table. That cost is why the evidence below matters
so much.

One point in SBP's favor that is easy to miss: **SAT is weak enforcement of
boundary conditions through characteristic penalty terms**, which is
conceptually the same family as the NSCBC corrections already in `nscbc.jl` —
those also act on the RHS as wave-amplitude corrections rather than by hard
state enforcement. An SBP–SAT move is less alien to this codebase than it
sounds, and it would likely subsume rather than replace that machinery.

Three documented sharp edges, none fatal but all worth budgeting for:

- **Accuracy drops at SAT interfaces**, below the interior stencil order. Nissen
  et al.'s response is to minimize the number of SAT interfaces and run interior
  stencils across block boundaries wherever possible; they report the local
  reduction not severely degrading the propagated solution. Plan the block
  layout around this rather than discovering it.
- **Non-conforming interfaces cost a global order** unless order-preserving
  interpolation operators are used — that is precisely what Almquist & Dunham
  exists to fix, and it is a known trap in earlier work.
- **Stability proofs get harder above 4th order.** The usual argument needs
  interpolation operators to be norm-contracting, which holds at 2nd and 4th
  order but *not* at 6th; recovering provable stability there required new
  penalty terms. At the orders this code targets, that is the regime of interest,
  not a corner case.

### What Miranda actually does

The question above is settled by evidence rather than by choosing. Pyranda does
not expose AMR, but it carries Miranda's Fortran kernels, and the level-transfer
operators are in `pyranda/parcop/stencils.f90` — `cfamrcf` and `cfamrfc`, inside
an `#if 0` block, so they are the design rather than a live implementation. The
header states the whole scheme:

```
INVERTIBLE AMR FILTER FOR COARSE-TO-FINE AND FINE-TO-COARSE OPERATIONS (fbar <---> f):
alpha*fbar(i-1) + fbar(i) + alpha*fbar(i+1) = c*f(i-2) + b*f(i-1) + a*f(i) + b*f(i+1) + c*f(i+2)
alpha = -0.0321826755129339
    a =  0.4451523642186118
    b =  0.2207614172195584
    c =  0.0244797251582018
The above coefficients match the transfer function of a gaussian filter of width 3*dx.
```

It is neither SBP–SAT nor ghost-fill interpolation. **Level transfer is an
invertible compact filter pair.** Restriction applies a compact filter whose
transfer function is a Gaussian of width 3·Δx; prolongation recovers the fine
field by *deconvolving* the same filter — the two routines are the identical
coefficients with the LHS and RHS roles swapped (`cfamrcf` solves the
`[c,b,a,b,c]` pentadiagonal against an `[α,1,α]` right-hand side, `cfamrfc` does
the reverse). Five properties follow, and each one matters:

- **Refinement ratio 3, not 2**, matched to the 3·Δx filter width. The odd ratio
  is what lets a coarse node coincide with the middle of each fine triple, which
  keeps both operators centered and symmetric — and therefore invertible. This
  is also the clean answer to the question the earlier draft got wrong: Miranda
  avoids the staggering problem by choosing an odd refinement ratio, not by
  moving to a cell-centered grid.
- **Conservation comes from unit DC gain**, not from refluxing. The interior
  coefficients satisfy a + 2b + 2c = 1 + 2α to the last bit (both
  0.9356346489741322 in Float64), so the transfer preserves the mean exactly.
- **The boundary closures preserve that gain deliberately.** The first closure
  row is explicit and sums to exactly 1.0; the second has an LHS summing to the
  interior RHS within one ulp. Conservation at the transfer boundary is enforced
  by construction of the closure rows.
- **Invertibility at the boundary is the fragile part, and they say so.** The
  comments read "no filter (current) or bad filter at boundary causes ringing in
  the solution" and "with extended boundary data : same as one-sided to maintain
  invertibility." Four closure variants are tabulated per end — odd-symmetric,
  one-sided, even-symmetric, and extended-data — on the `-1:2` index.
- **The parity machinery is the same machinery.** `lower_symm_weights` and
  `upper_symm_weights` fold a ghost coefficient onto the interior with a sign
  (`alb(i2,j) += syml*alb(i1,j); alb(i1,j) = 0`). That is structurally identical
  to CompactLES's axis fold, `b[1] += σg·α` in `operators.jl`. And `nci`, "the
  parallel overlap of lhs stencil," says these run through the same distributed
  compact solve as everything else.

There is a companion family of coarsening filters — `c4ff3` ("compact filter for
AMR coarsening," built so the transfer function integrates to 1/3 over [0, π],
exactly the 3:1 spectral budget) and `cgff2` (a 5×5 Gaussian ≈ exp(−2k²/3)).

### Why this changes the cost estimate

This is dramatically cheaper for CompactLES than either route the previous draft
weighed, because **the transfer operators are just more compact schemes**, and
every piece they need already exists: `BandedCompactScheme` is the type,
`plan_direction` binds it to a dimension, the spike solve provides the `nci`
parallel overlap, the parity folds provide the symmetric closure variants, and
`compact_filter` is already this shape with a boundary cascade. Adding
Miranda-style level transfer is *writing down two schemes*, not rewriting the
operator layer.

What CompactLES would still lack is the patch and level data structure, the time
sub-cycling bookkeeping, and the refinement criteria — **infrastructure, not
numerics.** That is a much better problem to have, and it re-ranks the work:
the numerics risk drops sharply, and Phase 3's cost becomes dominated by the
patch abstraction, which is also what the GPU port wants anyway.

**What the public Pyranda patch kernel actually contains.** At commit `b4e0afc`,
`objects.f90` reserves `(patch, level)` tables (101 patches by 11 levels), each
with its own patch metadata, Cartesian MPI communicator, compact plans, and
mesh. `comm.f90` creates the patch communicator with `MPI_Comm_split` from the
patch color/key. `ghost.f90` exchanges halos only between ranks inside that
uniform patch communicator and applies physical-boundary symmetry or
extrapolation; it has no coarse–fine exchange. `mesh.f90` allocates a SAMRAI
refinement-tag array, but the public tree never populates or consumes it.
Consequently the scheduler, patch adjacency, regridding, time subcycling, and
tagging criteria are not present in Pyranda and cannot be inferred from it.
Also confirmed: `compact_r3.f90` and `compact_r4.f90` are rank-3 and rank-4
array application kernels, not refinement-ratio layers.

**Conditioning measurement.** `bench/amr_transfer.jl` reconstructs the periodic
symbols and finite one-sided matrices directly from `cfamrcf`/`cfamrfc`. For a
96-point line, prolongation has gain 1.50883 at the representable coarse
Nyquist (`k = π/3`) and 20.2393 at the fine Nyquist. The finite closure operator
has condition number 33.03 and spectral norm 20.19; six-point alternating noise
is amplified 13.80 at an edge versus 16.11 in the interior. The closure is
therefore not the worst amplifier. The swapped pair round-trips to
`3.4e-15` in infinity norm. The conclusion is narrower than “safe”: the coarse
band is well-conditioned, and the published closure preserves invertibility,
but any sensor or interface operation that injects fine-grid Nyquist content
will be amplified by roughly 20× and must be filtered or excluded.

Prolongation is deconvolution, and inverting
a Gaussian amplifies high wavenumbers by construction — the `[c,b,a,b,c]`
symbol is small near k = π, so its inverse is large there. This is benign in
principle, because a coarse field carries no content above the coarse Nyquist
(k = π/3 in fine units) and the deconvolution is only asked to amplify inside
the well-conditioned band. It stops being benign if a boundary closure or an
artificial-property sensor injects grid-scale content near a transfer boundary,
which is very likely why the source comments fuss about ringing and about
closures that "maintain invertibility." The measurement above bounds this risk;
the next test belongs in a real coarse–fine interface with Cook sensors active.

### SBP–SAT as the alternative

The SBP route remains the rigorous option and the one with proofs, but it is no
longer the default recommendation: Miranda demonstrates the cheaper design works
in this exact scheme family and physics. Keep it in view for two reasons. First,
if the deconvolution conditioning above turns out badly, SBP–SAT is the fallback
with guarantees rather than a research project. Second, `SummationByPartsOperators.jl`
(Ranocha, also a Trixi core developer) is the existing Julia package and
implementing the Mattsson–Rydin implicit operators is an open issue there, so
that path has a collaborator rather than requiring solo work.

A data point for calibration: **HAMeRS**, the closest *published* relative —
patch AMR on SAMRAI, high-order FD, built by Wong & Lele for exactly this RM/RT
mixing physics — uses explicit schemes (WCNS for shock capturing, explicit
6th-order FD for viscous terms), not implicit compact. Miranda's approach is
therefore not the only one in this space, and it is the less documented one.

### Remaining issues

- **Temporal prolongation.** Sub-cycled levels need ghost data at each of the
  five RK stage times, not just at step boundaries, which means Hermite
  interpolation in time as well as high-order interpolation in space.
- **Sensor consistency across levels.** The Cook artificial properties are built
  from *undivided* δ⁴ differences with per-dimension h weights — a deliberately
  grid-based regularization. Across a refinement boundary the same physical
  feature therefore gets a different μ\*, β\*, κ\*. This needs a defined
  convention, and it is a numerics decision rather than an implementation
  detail.
- **Folds and refinement.** The antipodal fold machinery assumes uniform global
  structure in the paired dimension (partner at +P/2, reflected partner). A
  refinement patch straddling the axis or a pole breaks that assumption. The
  pragmatic answer is to forbid refinement across a fold initially — the
  singular region is usually where you want uniform resolution anyway.

**Framework or hand-rolled** is the open decision. Julia has nothing at the
maturity of SAMRAI or AMReX; Trixi's precedent is to bind an established C
library (P4est.jl) rather than build, which preserves correctness at the cost of
the pure-Julia dependency story that `README.md` currently advertises. Worth
deciding deliberately rather than by default.

**A note on sourcing.** Miranda's adaptivity is not described in the public
literature — searches surface its numerics consistently and its AMR not at all,
and the Ares/Miranda Rayleigh–Taylor validation study attributes AMR to Ares. The
account above is read directly from the kernels Pyranda carries, which is
primary evidence for the *operator design* but says nothing about the patch
management, sub-cycling, or refinement criteria that live in Miranda proper.
Treat the numerics here as well-founded and the surrounding infrastructure as
still unknown.

## Phase 0 — the extensibility seams (complete, July 2026)

Completed as a behavior-preserving refactor. All convergence errors remained
bit-identical; serial and 2/4/8-rank gates pass. JET reports for `compute_rhs!`
dropped from 16 to 2, and axis-fold RHS allocation dropped from 8,336 B to
784 B per call.

**1. A source-term interface.** Add a `sources` tuple to `Solver` and a final
`add_sources!(solver, dQ, Q, t)` in `compute_rhs!` that dispatches per source
type. Keep it a tuple, not a vector of an abstract type, so it stays inferable —
check with `bench/jetcheck.jl`. Ship it with one trivial implementation
(constant body force or a Boussinesq gravity term) so the interface has a test.

**2. An `EquationSet` type owning the conserved layout.** Move `n_cons`,
`i_mom`, `i_energy`, component names, and the fold parity tables behind a type.
Close the three `n_species + 4` sites. `NavierStokes1T` is the current behavior;
`NavierStokes3T` becomes a new instance rather than a rewrite. This is the
Trixi lesson and it is the difference between 3T taking a week and taking a
month.

**3. Concretely parameterize `Solver`.** `eos::EOS`, `metric::Metric`, and
`folds::NTuple{3,Union{Nothing,FoldSpec}}` are abstract fields. This is already
on the open-items list in `CLAUDE.md` as "FoldSpec parameterization"; it is the
same job, and it gets strictly harder as the struct grows. Do it while it is
still 50 fields.

**4. Split the state allocation from the solver.** `run!` allocates `dQ` and
`du` internally. Operator splitting, sub-cycling, and IMEX all need to own the
stage storage. Hoist it into a `Workspace` the caller can hold.

## Phase 1 — make shock tubes unimpeachable (largely complete, July 2026)

The primary use case, and the foundation of credibility for everything after it.
The validation before this phase was honest but thin: Sod against the exact
Riemann solution, Taylor–Green, freestream, and manufactured fold solutions.

**A real validation battery — done.** `test/validation.jl` runs Lax, Shu–Osher,
Woodward–Colella, Sedov–Taylor and Noh, the last in all three geometries, in
about 25 seconds. It is deliberately split by what each case can be measured
against: Lax, Sedov and Noh have closed-form solutions and can therefore say the
code is *wrong*; Shu–Osher and Woodward–Colella are guarded against stored
4×-resolution profiles from this code and can only say it *changed*. Cases live
in `test/cases.jl`, shared with the calibration sweep so the two cannot drift.

Two operating limits fell out of building it, both real and both now documented
in `reference/CALIBRATION.md`: strong shocks need `cfl ≤ 0.15` because
`compute_dt` lags the artificial coefficients by a step, and the spherical origin
fold will not take initial data resolved over fewer than ~3 cells or the singular
t = 0 start of spherical Noh — while the cylindrical axis takes both. The second
of those is an unexplained asymmetry between two folds that are supposed to be
the same machinery, and it is the most interesting loose thread in this phase.

**Calibrate the artificial-property constants — done.** `bench/artcal.jl` sweeps
each constant over the battery; `reference/CALIBRATION.md` is the write-up. The
defaults survive, with one substantive correction (the CFL guidance above) and
three findings worth carrying forward: `C_beta` fails at *both* ends and its
upper bound is a timestep-stability bound rather than an accuracy one; `C_kappa`
= 0 will not run a converging strong shock at all, so the artificial conduction
is load-bearing and not just an accuracy term; and `C_D` is a weak knob, because
on a passive interface the compact filter does more smearing than D\* does.
`C_mu` remains uncalibrated for its actual purpose — every case in the battery
is 1-D, where the shear viscosity is inert. That needs a 3-D case.

**Mixing diagnostics — done.** `src/diagnostics.jl` provides metric-aware,
MPI-reduced volume integrals and plane-averaged profiles, and on top of them mix
width, Youngs' molecular mixing fraction θ, composition PDFs, Favre turbulent
kinetic energy, and resolved dissipation including the artificial contribution.
The quadrature is the load-bearing part — every mixing number is a ratio of two
integrals, so a wrong edge weight biases θ silently — and it is exact for a
constant on a Cartesian grid and second order at a node-centered curvilinear
edge.

**Non-ideal EOS — contract and NASA-9 data path done.** The blockers named
above are closed: φ = ∂(ρe)/∂p, ∂φ/∂Y_k, and the artificial-conductivity scale
are now EOS dispatch points rather than ideal-gas algebra inlined at their call
sites in `nscbc.jl` and `artificial.jl`. The full contract is written down at
the top of `physics.jl`.

Two models joined `IdealMixture` to prove the contract carries weight rather
than just existing: `StiffenedGas` (`p = (γ−1)ρe − γp∞`, exact perfect-gas limit
at p∞ = 0, the natural precursor to Mie–Grüneisen) and `Nasa9Mixture` with
temperature-dependent specific heats and a Newton inversion of e(T). The NASA
CEA thermodynamic and limited transport databases ship verbatim in `data/`,
beside the upstream Apache license and notice. `read_nasa9` parses the thermo
table's fixed-column, multi-interval records, derives R from molar mass, and
makes the energy reference explicit (`:sensible` by default, `:formation` for
the tabulated heat-of-formation gauge). `examples/shock_tube.jl` now uses real
He and CO₂ cp(T). The transport coefficients remain raw data until the constant
`Transport` model grows temperature-dependent properties and mixture rules.

What is *not* done: the κ\* singularity as T_ion → 0 is now visible and
dispatchable but not cured for gases, and that is a numerics decision (see
the open items in `CLAUDE.md`). Mie–Grüneisen proper is still ahead.

## Phase 2 — HED physics

This is the NIF path, and it is the expensive one. Ordered so each step is
usable on its own.

**1. Implicit diffusion infrastructure.** Everything else here depends on it.
Build it on the existing distributed banded solve: ADI or line-Jacobi sweeps as
the smoother, wrapped in either a geometric multigrid or a Krylov method
(Krylov.jl is the mature Julia option and is dependency-light). Validate against
a manufactured heat-conduction solution in every metric, and against the
existing freestream tests. **This single piece also solves the azimuthal-CFL
problem documented in the README** — implicit treatment of the θ direction is
one of the standard remedies listed there — so it earns its cost twice.

**2. IMEX time integration.** With the implicit solver in hand, an IMEX-ARK
scheme pairing the existing explicit RK45 on hydro with an implicit treatment of
diffusion. Phase 0 item 4 is the prerequisite. Consider borrowing tableaus from
the SciML ecosystem rather than hand-rolling.

**3. Three-temperature hydrodynamics.** T_ele and T_rad alongside T_ion — the
naming convention has reserved this from the start, which is why it costs an
equation set rather than an API break. Separate electron and ion energy
equations, an electron–ion equilibration source (Phase 0 item 1), and the
electron pressure contribution to the momentum flux.

**4. Electron thermal conduction.** Spitzer–Härm with a flux limiter, through
the Phase 2.1 solver. Straightforward once 1 and 3 exist.

**5. Tabulated EOS.** An IONMIX reader first (it is the simpler format and is
what FLASH's HEDP demos use), SESAME later if licensing permits. The Phase 1
EOS generalization is the prerequisite — do not attempt this before NSCBC and
the artificial-property model are EOS-agnostic.

**6. Multigroup radiation diffusion.** Flux-limited, gray first, then multigroup
with a group structure and an opacity interface mirroring the EOS contract.
Radiation groups become additional conserved components, which is what Phase 0
item 2 exists to make cheap.

**7. Laser energy deposition.** Geometric-optics ray tracing with inverse
bremsstrahlung absorption. Architecturally independent of everything above — it
is a source term (Phase 0 item 1) plus a ray tracer that hands rays between
ranks. Could be done earlier if a specific experiment demands it, and it is the
most visually compelling capability on this list.

**8. MHD, if magnetized targets matter.** Note for planning: the ∇·B constraint
with finite differences wants hyperbolic (GLM) divergence cleaning, which adds a
conserved component and a source term — again, Phase 0. Constrained transport is
not natural in this discretization.

## Phase 3 — scale, portability, and adaptivity

**Patch-based AMR.** The long-term target, with the design constraints and the
open framework decision set out in [Adaptivity](#adaptivity--the-compact-scheme-constraint)
above. Sequence it as: implement the Miranda-style invertible transfer filter
pair as two compact schemes and measure its conditioning under grid-scale noise
(cheap, and it is the one real numerics risk); introduce a patch abstraction at
a *single* level with conforming interfaces
(pure refactoring, independently verifiable against the existing bit-exact
tests, and immediately useful as multiblock geometry even if refinement never
follows); then non-conforming coarse-fine interfaces, spatial prolongation,
temporal prolongation for RK sub-cycling, and refinement criteria last.
Conforming multiblock before non-conforming refinement is the important ordering
— it is where the SBP–SAT literature's guarantees are cleanest and where a
mistake is cheapest to find. The refinement criterion is nearly free when it
arrives — the
Cook δ⁴ sensors in `artificial.jl` are already exactly the "where is this
under-resolved" signal an AMR tagger needs.

**GPU through KernelAbstractions.jl.** The pointwise kernels — `primitives!`,
`assemble_fluxes!`, the RK update, `compute_artificial!` — port
straightforwardly, and they are the majority of the phase budget outside the
line solves. The hard part is the distributed banded solve, where the batched
Thomas sweeps are sequential along the line but embarrassingly parallel across
lines, which is actually a reasonable GPU shape given the transposed layout
`lines_transposed.jl` already provides. Follow XCALibre's architecture; do not
maintain two codebases.

**Sequence GPU and AMR together, AMR first.** These are usually treated as
independent workstreams and here they should not be. Patch AMR breaks the "one
global array per field" assumption that currently pervades the code and replaces
it with many independently-sized blocks — which is precisely the shape a GPU
port wants, since a patch is a natural kernel-launch unit with a bounded working
set. Porting to GPU against monolithic arrays first means doing the same
decomposition work twice, in the harder order. That Miranda has a GPU
implementation of its patch AMR is evidence the combination is the right target
rather than two competing ones.

**Mixed precision.** `Solver{T}` is already parameterized; the stated blocker is
that halo buffers are concretely `Float64`. Worth closing as a small, contained
task — it also serves as a proof that the `{T}` parameterization is real.

**Parallel HDF5/XDMF output.** The checkpoint and VTK paths are per-rank plus a
rank-0 container, which does not survive large rank counts. `DESIGN.md` already
sketches the shape this would take.

**Multiblock geometry** comes nearly free once the patch abstraction exists —
same-level patches with their own line solves, coupled through interface
exchange, are structurally what multiblock needs. Worth noting as a payoff of
the AMR work rather than a separate effort, since it is also the route to
non-coordinate-surface geometry without going unstructured.

**ThreadPinning wiring**, listed as an open item in `CLAUDE.md` and only
validatable on the target cluster.

## Non-goals

Stating these explicitly, because each is something a reviewer will ask about
and the answer is "deliberately not."

- **Oct-tree / cell-by-cell AMR.** Patch-based refinement is in long-term scope
  (Phase 3); tree refinement down to individual cells is not, because it is the
  variant that genuinely conflicts with a line-coupled solve.
- **Unstructured meshes.** Same reason, more so. Multiblock patches are the
  answer to geometric complexity in this scheme family.
- **Riemann solvers and flux limiters.** The artificial-fluid-property approach
  is the design commitment. Adding a Godunov path would double the maintenance
  surface for a capability that other codes do better.
- **A string-based PDE interpreter**, in the Pyranda style. Note this is a
  narrower exclusion than "no symbolic frontend": a macro-based frontend that
  *lowers* to the equation-set dispatch of Phase 0 is legitimate and arguably the
  right long-term ergonomic answer, since it gets Pyranda's ten-line problem
  specification while the generated code stays concretely typed and stays
  visible to `bench/audit.jl` and `bench/jetcheck.jl`. The exclusion is on
  runtime string evaluation, not on metaprogramming. Sequencing matters: the
  dispatch layer has to exist and be exercised by hand-written equation sets
  before a macro is written to emit it, or the macro ends up defining the
  interface by accident.
- **Being a general-purpose CFD library.** XCALibre and Trixi occupy that
  ground. This code is for compressible, variable-density, shock-driven mixing
  and implosion, and its value is being excellent at that.

## Suggested ordering

If only three things happen, they should be:

1. **Phase 0 in full.** Done, July 2026. Behavior-preserving, and it is the
   difference between the rest of this document being tractable and being a
   rewrite. The bit-identical convergence guards made it verifiable.
2. **Phase 1's validation battery and EOS generalization.** Done, July 2026.
   This is what makes the primary use case defensible and simultaneously
   unblocks Phase 2.5.
3. **Phase 2.1, the implicit diffusion solver.** The one true architectural gap,
   with more of its foundation already built than is obvious, and it pays off
   twice by also addressing the polar CFL restriction.

Laser ray tracing (2.7) is the exception to the ordering: it is independent of
everything else and could be pulled forward if a specific NIF experiment needs
it before the rest of the HED stack exists.

Two items from the adaptivity discussion belong earlier than Phase 3, though
both are cheap and neither is structural:

- **Build the Miranda transfer filter pair and measure its conditioning.** The
  coefficients are known and the machinery to plan them already exists, so this
  is a small piece of work that de-risks the whole adaptivity plan. What to look
  for is the deconvolution's response to grid-scale content near a transfer
  boundary — the failure mode the Fortran comments hint at.
- **Ask about Miranda's patch infrastructure.** The operator design is settled
  from the kernels; the patch management, time sub-cycling, and refinement
  criteria are not, and they are what Phase 3's cost now turns on.

Deliberately *not* on this list: any change to the grid arrangement. An earlier
version of this document argued that moving to a uniformly cell-centered grid
was a now-or-never prerequisite for adaptivity. That was extrapolated from a
single recent preprint working in the explicit WENO-family finite-difference
lineage, and it does not survive contact with the SBP–SAT literature, where
multiblock interfaces are node-coincident and non-conforming interfaces are
handled by interpolation operators rather than by a staggered arrangement. If
anything the established route points the other way. The current mixed
convention — half-offset only where a fold requires it — should stay until
there is a specific reason to change it.
