# CompactLES — Roadmap

This roadmap compares CompactLES with related solvers and identifies the work
required to progress from shock-tube calculations to NIF-relevant multiphysics.
`README.md` covers usage, `DESIGN.md` describes the numerics, and `CLAUDE.md`
defines development procedures.

## Contents

1. [Positioning](#positioning)
2. [Comparison](#comparison)
3. [Remaining blockers](#remaining-blockers)
4. [Adaptivity — the compact-scheme constraint](#adaptivity--the-compact-scheme-constraint)
5. [Phase 0 — the extensibility seams](#phase-0--the-extensibility-seams-complete-july-2026)
6. [Phase 1 — shock-tube validation](#phase-1--shock-tube-validation-largely-complete-july-2026)
7. [Phase 2 — HED physics](#phase-2--hed-physics)
8. [Phase 3 — scale, portability, and adaptivity](#phase-3--scale-portability-and-adaptivity)
9. [Non-goals](#non-goals)
10. [Suggested ordering](#suggested-ordering)

## Positioning

The existing numerical method determines which comparisons are informative.

CompactLES offers tenth/sixth-order compact Padé
derivatives, a Gaitonde–Visbal compact filter, Cook artificial fluid properties
in place of Riemann solvers, five-stage low-storage RK45, structured curvilinear
grids, and MPI. These elements match the Miranda/Pyranda method except for
patch-based adaptivity and a GPU backend. They are appropriate for
variable-density turbulent mixing and shock–interface interaction, the primary
shock-tube and RM/RT use cases. Comparisons with Pyranda therefore concern scope
and interface, whereas comparisons with FLASH concern the physics catalogue.

Three current capabilities distinguish CompactLES and constrain subsequent
design decisions:

- **The regularized coordinate-singularity treatment.** The half-offset grid
  plus parity/antipodal folds for the cylindrical axis, spherical origin, and
  spherical poles — with a discrete GCL that preserves freestream to machine
  zero — is more than Pyranda or Miranda expose, and nothing in the Julia
  ecosystem has it. Converging-shock and spherical-implosion geometry is
  directly NIF-relevant, and this is the piece that is hard to rebuild.
- **The distributed compact solve.** The spike/reduced-interface
  banded solve reproduces the single-domain answer bit-for-bit at any rank
  count, with the reduced system factorized once at plan time. It is also, as
  argued below, the seed of the implicit infrastructure the HED physics needs.
- **The `Problem`/`Numerics` split.** The physics specification does not refer to
  ranks, halos, or the conserved layout. This separation is difficult to add
  after a multiphysics implementation has matured.

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

Pyranda also provides a reference implementation because it carries Miranda's
Fortran kernels in
`pyranda/parcop/`, including operators Pyranda itself does not expose. The
adaptivity section below is derived directly from those sources.

**Capabilities absent from CompactLES:** Pyranda provides a DSL for defining
a new PDE system in ten lines without touching the solver; immersed boundaries,
which provide non-coordinate-surface geometry without unstructured
meshes; and the credibility of being the mini-app for a production code.

**Equation specification.** Pyranda's string-based equation interpreter provides
generality at the cost of type safety and
of any physics that is not a flux divergence plus a source. Julia multiple
dispatch on an equation-set type, as used by Trixi and proposed in Phase 0,
provides an alternative. This design retains inference and
allocation discipline (`bench/audit.jl`, `bench/jetcheck.jl`) while getting most
of the extensibility. If a symbolic frontend is later wanted for ergonomics, the
route is a macro that *emits* an equation set at parse time. Runtime evaluation
of equation strings is excluded, whereas metaprogramming remains in scope.

Immersed boundaries remain a possible extension, although a sharp
IB cut cell is fundamentally at odds with a line-global compact operator, which
is why IB in this scheme family is usually done as a smeared Brinkman penalty
term, which can use the Phase 0 source-term interface.

### FLASH

FLASH uses second-order unsplit PPM, so the relevant comparison concerns its
physics catalogue and AMR rather than numerical order.

FLASH's HEDP capability is a set of units: 3T hydrodynamics with electron–ion
equilibration, multigroup radiation diffusion, tabulated multi-species EOS and
opacities (IONMIX), electron thermal conduction through an implicit diffusion
solver, laser energy deposition by geometric-optics ray tracing, MHD, and
anisotropic magnetized transport coefficients — all on a block-structured AMR
mesh, validated against HYDRA. This capability set defines the long-term target
for NIF applications and represents approximately a decade of physics
implementation.

Two structural patterns are relevant:

- **Physics-unit decomposition.** FLASH's physics units are separately
  configurable, each owning its own state and contributing to the update through
  a defined interface. This is why FLASH could grow 3T and MGD without
  rewriting hydro. CompactLES's equivalent is Phase 0.
- **Patch-based adaptivity.** FLASH's PARAMESH/AMReX
  refinement is oct-tree block AMR built around a second-order finite-volume
  update. The version that belongs here is *patch-based* (logically rectangular
  blocks of uniform resolution, SAMRAI-style), which is what Miranda uses and
  which makes adaptivity compatible with a compact scheme. See
  [Adaptivity](#adaptivity-the-compact-scheme-constraint) for why the
  distinction and Phase 3 for the proposed implementation.

### Trixi.jl

The Julia reference for high-order conservation laws: nodal DG-SEM on a
quad/octree with AMR, `p4est` for unstructured curved meshes, entropy-stable and
kinetic-energy-preserving split forms, shock capturing with positivity limiting,
a large equation-set catalogue (Euler, MHD, shallow water, hyperbolic diffusion
for self-gravitating gas dynamics), and OrdinaryDiffEq.jl integration.

**Relevant design pattern:** the `equations` type parameter. Trixi's solvers are
generic over an equation set that owns the variable count, names, flux
functions, and conversions. Everything — the DG operator, AMR, the time
integrator, the output — is written against that interface. It is why adding
MHD to Trixi did not require rewriting Euler. Phase 0 item 2 adopts this
equation-set interface.

**CompactLES advantages for the target problems:** compact finite differences
require less work per degree of freedom than DG at comparable resolving power
for smooth, volume-filling turbulence and have lower memory traffic.
Trixi has nothing resembling Cook artificial fluid properties, which is the
right regularization for material interfaces at high Atwood number — DG's
entropy-stable limiting is intended for robustness at discontinuities rather
than controlled subgrid dissipation in a mixing layer. Trixi also has no regularized
polar/spherical singularity treatment.

**Applicable external component:** Trixi uses OrdinaryDiffEq.jl for time
integration. A CompactLES IMEX scheme in Phase 2 could use the tested IMEX-ARK
tableaus in the SciML ecosystem.

### XCALibre.jl

Second-order unstructured finite volume, incompressible and compressible, RANS
and LES, OpenFOAM/unv mesh import, CPU threads or GPU through
KernelAbstractions.jl. Different accuracy class and a different problem domain
(engineering CFD on complex geometry), so it is not a competitor.

XCALibre provides a relevant GPU architecture: a single
KernelAbstractions.jl codebase targets NVIDIA, AMD, and Intel while retaining a
CPU path. This is the proposed architecture for Phase 3. Oceananigans.jl
provides an additional example of a Julia structured-grid solver designed for
GPU execution.

## Remaining blockers

Everything below follows from the remaining architectural gaps. Phase 0 closed
the source/layout/storage seams in July 2026; the implicit solver and EOS
generalization remain blockers.

**The source and layout seams now exist.** `Solver.sources` is an inferable tuple
applied at RK stage time, and `NavierStokes1T <: EquationSet` owns the component
indices, names, conserved conversion, and fold parity. The former hard-coded
`n_species + 4` sites are gone. A 3T model still requires its physics, but no
longer requires rediscovering the conserved layout throughout the solver.

**Implicit and elliptic solvers are absent.** Radiation diffusion and electron
conduction are parabolic and stiff; explicit treatment is not an option at HED
conditions, where the conduction timestep can be orders of magnitude below the
acoustic one. This is the largest remaining architectural gap.

`tridiag.jl` and `banded.jl` already implement a
distributed banded line solve with cross-rank coupling and a pre-factorized
reduced interface system. That is precisely the kernel an ADI or line-relaxation
diffusion solver needs, and precisely the smoother a geometric multigrid wants.
This existing machinery provides a substantial part of an implicit-diffusion
implementation and should form its basis instead of a separate linear-algebra
dependency.

**The EOS contract is now agnostic; one physical assumption survives it.** The
sites that previously assumed an ideal gas outside the function barrier — NSCBC's LODI
algebra and the artificial-conductivity scale — are EOS dispatch points as of
July 2026, and `StiffenedGas` and `Nasa9Mixture` exercise them. A SESAME reader
needs data plumbing and a table interpolator, not solver surgery.

The assumption that survives is physical rather than structural: the artificial
conductivity is still built as (ρc/T_ion)·sensor for every gas model here, which
is singular at a cold ambient. A tabular or condensed-matter EOS can provide an
alternative scale.

## Adaptivity — the compact-scheme constraint

Adaptivity is in long-term scope. Its interaction with the compact scheme shapes
Phase 3 and affects the correctness criterion used by the test suite.

**The constraint is not "no AMR," it is "no line cut mid-solve."** A compact
operator couples an entire grid line through a banded LHS, so what it cannot
tolerate is a resolution change *inside* a solve. Patch-based AMR does not do
that: each patch is a logically rectangular block of uniform resolution, and the
compact solve runs over the patch with its ghost layers filled by copy from
same-level neighbors or by interpolation from the underlying coarse level.
Oct-tree cell-by-cell refinement is the incompatible one. This is the
distinction that matters, and the earlier framing here collapsed the two.

**Patch interfaces replace exactness with a controlled approximation.** Today the
spike/reduced-interface solve reproduces the single-domain answer bit-for-bit at
any rank count, which provides the correctness criterion used by the MPI suite.
An interface defect produces O(1) error rather
than a small one, which is precisely how `test/mpi_tests.jl` catches it. Under
patch AMR, coarse-fine boundaries make the solve approximate by construction,
and that oracle stops working there. Any adaptivity work must bring its own
replacement, most plausibly a manufactured solution across a refinement
boundary with a measured order, in the style `test/convergence.jl` already uses.

**Distributed solves remain exact within each refinement level.** The design can
keep the exact distributed solve *within* a refinement level, where neighbors
are at uniform resolution and the existing sub-communicator machinery already
applies, and to localize the approximation to coarse-fine interfaces only. That
preserves the bit-exact guarantee across the majority of patch boundaries and
confines the new error to where it is unavoidable.

**Error localization.** The inverse of a compact LHS decays geometrically away
from the diagonal, approximately
α^|i−j|, so α = 1/3 for C6. Pollution injected at an inexact patch boundary
decays by a factor of three per point along the line rather than contaminating
it globally. An early measurement of this decay will quantify the error from an
approximate interface for either treatment considered below.

### Published interface treatments

Two interface treatments are relevant: the SBP–SAT formulation with stability
proofs and the compact-filter transfer used by Miranda. The latter is the
recommended initial implementation; the former provides a rigorous alternative
and a basis for comparison.

Summation-by-parts operators with simultaneous-approximation-term (SAT)
coupling provide a stable and conservative treatment of high-order
finite-difference block interfaces. The literature for the *implicit/compact*
case includes:

- Carpenter, Gottlieb & Abarbanel (1993) established stable, accurate boundary
  treatments for compact high-order schemes, with Abarbanel & Chertock (2000)
  following on strict stability and the role of boundary conditions.
- Mattsson & Rydin (2022) derived **implicit SBP operators** for first and
  second derivatives with boundary closures on a *banded-norm* SBP framework and
  weak (penalty) boundary enforcement, reaching 8th-order global convergence.
  This is the Padé-class case and the principal reference for the proposed work.
- Nissen et al. (2015) built block-oriented adaptive grids on SBP–SAT, including
  a stable treatment of junction points where interfaces of different type meet.
- Almquist & Dunham (2018) supplied order-preserving interpolation operators for
  **non-conforming** (refined) interfaces, which is the coarse-fine case.

CompactLES's current closures are not SBP.
The `ClosureRow` cascade in `kernels.jl` implements the classical Lele one-sided
closures — stable in practice, without an energy estimate. Moving onto an SBP
footing would be a deeper change than anything else in this document, touching
`kernels.jl`, the closure rows, `operators.jl`, the folds, and every order in the
`test/convergence.jl` guard table. The evidence below determines whether that
larger change is necessary.

SAT weakly enforces boundary conditions through characteristic penalty terms,
which is
conceptually the same family as the NSCBC corrections already in `nscbc.jl` —
those also act on the RHS as wave-amplitude corrections rather than by hard
state enforcement. An SBP–SAT implementation could therefore subsume the
existing NSCBC correction machinery rather than replace its underlying
formulation.

Three documented considerations apply:

- **Accuracy drops at SAT interfaces**, below the interior stencil order. Nissen
  et al.'s response is to minimize the number of SAT interfaces and run interior
  stencils across block boundaries wherever possible; they report the local
  reduction not severely degrading the propagated solution. Block layout should
  therefore minimize SAT interfaces.
- **Non-conforming interfaces cost a global order** unless order-preserving
  interpolation operators are used — that is precisely what Almquist & Dunham
  addresses; earlier work documents this loss of order.
- **Stability proofs above 4th order require additional terms.** The usual argument needs
  interpolation operators to be norm-contracting, which holds at 2nd and 4th
  order but *not* at 6th; recovering provable stability there required new
  penalty terms. This limitation applies directly to the target orders.

### Miranda's interface treatment

Pyranda does not expose AMR, but it carries Miranda's Fortran kernels. The
level-transfer
operators are in `pyranda/parcop/stencils.f90` — `cfamrcf` and `cfamrfc`, inside
an `#if 0` block, so they are the design rather than a live implementation. The
header defines the scheme:

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
the reverse). The pair has five relevant properties:

- **Refinement ratio 3, not 2**, matched to the 3·Δx filter width. The odd ratio
  allows a coarse node to coincide with the middle of each fine triple, keeping
  both operators centered, symmetric, and invertible. Miranda therefore avoids
  staggering through an odd refinement ratio rather than a cell-centered grid.
- **Conservation comes from unit DC gain**, not from refluxing. The interior
  coefficients satisfy a + 2b + 2c = 1 + 2α to the last bit (both
  0.9356346489741322 in Float64), so the transfer preserves the mean exactly.
- **The boundary closures preserve that gain.** The first closure
  row is explicit and sums to exactly 1.0; the second has an LHS summing to the
  interior RHS within one ulp. Conservation at the transfer boundary is enforced
  by construction of the closure rows.
- **Boundary invertibility requires specialized closures.** The
  comments read "no filter (current) or bad filter at boundary causes ringing in
  the solution" and "with extended boundary data : same as one-sided to maintain
  invertibility." Four closure variants are tabulated per end — odd-symmetric,
  one-sided, even-symmetric, and extended-data — on the `-1:2` index.
- **The parity machinery matches the existing fold formulation.** `lower_symm_weights` and
  `upper_symm_weights` fold a ghost coefficient onto the interior with a sign
  (`alb(i2,j) += syml*alb(i1,j); alb(i1,j) = 0`). That is structurally identical
  to CompactLES's axis fold, `b[1] += σg·α` in `operators.jl`. And `nci`, "the
  parallel overlap of lhs stencil," says these run through the same distributed
  compact solve as everything else.

There is a companion family of coarsening filters — `c4ff3` ("compact filter for
AMR coarsening," built so the transfer function integrates to 1/3 over [0, π],
exactly the 3:1 spectral budget) and `cgff2` (a 5×5 Gaussian ≈ exp(−2k²/3)).

### Implementation cost

The transfer operators fit the existing compact-scheme implementation.
`BandedCompactScheme` provides the type,
`plan_direction` binds it to a dimension, the spike solve provides the `nci`
parallel overlap, the parity folds provide the symmetric closure variants, and
`compact_filter` is already this shape with a boundary cascade. Adding
Miranda-style level transfer therefore requires two scheme definitions rather
than a new operator layer.

CompactLES would still require patch and level data structures, time-subcycling
bookkeeping, and refinement criteria. The numerical risk is consequently lower
than the infrastructure cost, which is dominated by the patch abstraction also
needed by the proposed GPU port.

**Public Pyranda patch infrastructure.** At commit `b4e0afc`,
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
therefore not the largest amplifier. The swapped pair round-trips to
`3.4e-15` in infinity norm. The coarse band is well-conditioned, and the
published closure preserves invertibility,
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
the next test should use a representative coarse–fine interface with Cook
sensors active.

### SBP–SAT as the alternative

The SBP route remains the rigorously supported alternative, but Miranda provides
a lower-cost design in the same scheme family and physics. SBP–SAT remains
relevant if deconvolution conditioning proves unacceptable.
`SummationByPartsOperators.jl`
(Ranocha, also a Trixi core developer) is the existing Julia package and
implementing the Mattsson–Rydin implicit operators is an open issue there,
providing a possible external collaboration path.

A data point for calibration: **HAMeRS**, the closest *published* relative —
patch AMR on SAMRAI, high-order FD, built by Wong & Lele for exactly this RM/RT
mixing physics — uses explicit schemes (WCNS for shock capturing, explicit
6th-order FD for viscous terms), not implicit compact. Miranda's approach is
therefore not the only one in this space, and it is the less documented one.

### Remaining issues

- **Temporal prolongation.** Sub-cycled levels need ghost data at each of the
  five RK stage times rather than only at step boundaries, which means Hermite
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
  initial implementation can forbid refinement across a fold and retain uniform
  resolution in the singular region.

**Framework selection** remains open. Julia has no package at the
maturity of SAMRAI or AMReX; Trixi's precedent is to bind an established C
library (P4est.jl) rather than build, which preserves correctness at the cost of
the pure-Julia dependency model described in `README.md`.

**Source basis.** Miranda's adaptivity is not described in the public
literature — searches surface its numerics consistently and its AMR not at all,
and the Ares/Miranda Rayleigh–Taylor validation study attributes AMR to Ares. The
account above is read directly from the kernels Pyranda carries, which is
primary evidence for the *operator design* but says nothing about the patch
management, sub-cycling, or refinement criteria that live in Miranda proper.
The numerical design is supported by primary-source kernels, whereas the
surrounding infrastructure is unspecified. Patch bookkeeping is a design
problem with published precedent (SAMRAI, AMReX, and the SBP–SAT block literature
above), and this code's own constraints — the fold restrictions, the
sub-communicator machinery, and the bit-exact-within-a-level guarantee constrain
most choices. Phase 3 does not depend on access to Miranda.

## Phase 0 — the extensibility seams (complete, July 2026)

Completed as a behavior-preserving refactor. All convergence errors remained
bit-identical; serial and 2/4/8-rank gates pass. JET reports for `compute_rhs!`
dropped from 16 to 2, and axis-fold RHS allocation dropped from 8,336 B to
784 B per call.

**1. A source-term interface.** Add a `sources` tuple to `Solver` and a final
`add_sources!(solver, dQ, Q, t)` in `compute_rhs!` that dispatches per source
type. A tuple rather than a vector of an abstract type preserves inference, as
verified with `bench/jetcheck.jl`. A minimal implementation, such as a constant
body force or Boussinesq gravity term, should test the interface.

**2. An `EquationSet` type owning the conserved layout.** Move `n_cons`,
`i_mom`, `i_energy`, component names, and the fold parity tables behind a type.
Close the three `n_species + 4` sites. `NavierStokes1T` is the current behavior;
`NavierStokes3T` then becomes a new instance rather than a rewrite, following
the equation-set pattern used by Trixi.

**3. Concretely parameterize `Solver`.** `eos::EOS`, `metric::Metric`, and
`folds::NTuple{3,Union{Nothing,FoldSpec}}` are abstract fields. This is already
under **Known limitations** in `CLAUDE.md` as "FoldSpec parameterization"; it is the
same parameterization task. Completing it before the structure grows limits the
scope of the change.

**4. Split the state allocation from the solver.** `run!` allocates `dQ` and
`du` internally. Operator splitting, sub-cycling, and IMEX all need to own the
stage storage. Move it into a caller-owned `Workspace`.

## Phase 1 — shock-tube validation (largely complete, July 2026)

This phase establishes the validation basis for the primary use case. Before the
phase, validation consisted of Sod against the exact
Riemann solution, Taylor–Green, freestream, and manufactured fold solutions.

**Validation battery — complete.** `test/validation.jl` runs Lax, Shu–Osher,
Woodward–Colella, Sedov–Taylor and Noh, the last in all three geometries, in
about 25 seconds. It is deliberately split by what each case can be measured
against. Lax, Sedov, and Noh have closed-form solutions and test absolute
accuracy; Shu–Osher and Woodward–Colella use stored 4×-resolution profiles from
this code and detect regressions relative to that baseline. Cases live
in `test/cases.jl`, shared with the calibration sweep so the two cannot drift.

Construction of the battery identified two operating limits, documented in
`reference/CALIBRATION.md`. Converging strong shocks require `cfl ≤ 0.15`
because insufficient spatial regularization permits a dispersive density
undershoot; tests exclude the one-step lag in `compute_dt` as the cause. The
spherical-origin fold also does not accept initial data resolved over fewer than
approximately three cells or the singular t = 0 start of spherical Noh, whereas
the cylindrical axis accepts both. This difference between folds remains
unexplained.

**Artificial-property calibration — complete.** `bench/artcal.jl` sweeps each
constant over the battery, with results in `reference/CALIBRATION.md`. The
defaults remain, with one substantive correction to the CFL guidance and three
findings: `C_beta` fails at both ends and its
upper bound is a timestep-stability bound rather than an accuracy one;
`C_kappa = 0` does not support a converging strong shock, so artificial conduction is
required for stability rather than only accuracy; `C_D` has weak influence because
on a passive interface the compact filter produces more broadening than D\*.
A separate 128³ Taylor–Green study constrains `C_mu` for resolved shear and is
consistent with the shipped default.

**Mixing diagnostics — complete.** `src/diagnostics.jl` provides metric-aware,
MPI-reduced volume integrals and plane-averaged profiles, and on top of them mix
width, Youngs' molecular mixing fraction θ, composition PDFs, Favre turbulent
kinetic energy, and resolved dissipation including the artificial contribution.
The quadrature determines these integral ratios. An incorrect edge weight biases
θ without changing the API. The implementation is exact for a
constant on a Cartesian grid and second order at a node-centered curvilinear
edge.

**Non-ideal EOS — contract and NASA-9 data path done.** The blockers named
above are closed: φ = ∂(ρe)/∂p, ∂φ/∂Y_k, and the artificial-conductivity scale
are now EOS dispatch points rather than ideal-gas algebra inlined at their call
sites in `nscbc.jl` and `artificial.jl`. The full contract is written down at
the top of `physics.jl`.

Two models exercise the contract in addition to `IdealMixture`:
`StiffenedGas` (`p = (γ−1)ρe − γp∞`, exact perfect-gas limit
at p∞ = 0, the natural precursor to Mie–Grüneisen) and `Nasa9Mixture` with
temperature-dependent specific heats and a Newton inversion of e(T). The NASA
CEA thermodynamic and limited transport databases ship verbatim in `data/`,
beside the upstream Apache license and notice. `read_nasa9` parses the thermo
table's fixed-column, multi-interval records, derives R from molar mass, and
makes the energy reference explicit (`:sensible` by default, `:formation` for
the tabulated heat-of-formation gauge). `examples/shock_tube.jl` now uses real
He and CO₂ cp(T). The transport coefficients remain raw data until the constant
`Transport` model grows temperature-dependent properties and mixture rules.

The κ\* singularity as T_ion → 0 is exposed through dispatch but remains for gas
models; resolving it is a numerical-model decision (see
**Known limitations** in `CLAUDE.md`). Mie–Grüneisen proper is still ahead.

## Phase 2 — HED physics

This phase supplies the NIF-oriented physics. Its items are ordered so that each
produces an independently usable capability.

**1. Implicit diffusion infrastructure.** Everything else here depends on it.
Build it on the existing distributed banded solve: ADI or line-Jacobi sweeps as
the smoother, wrapped in either a geometric multigrid or a Krylov method
(Krylov.jl is the mature Julia option and is dependency-light). Validate against
a manufactured heat-conduction solution in every metric, and against the
existing freestream tests. Implicit treatment of the θ direction also addresses
the azimuthal CFL restriction documented in the README.

**2. IMEX time integration.** With the implicit solver in hand, an IMEX-ARK
scheme pairing the existing explicit RK45 on hydro with an implicit treatment of
diffusion. Phase 0 item 4 is the prerequisite. Existing tableaus from the SciML
ecosystem should be evaluated before implementing new ones.

**3. Three-temperature hydrodynamics.** T_ele and T_rad alongside T_ion — the
naming convention has reserved this from the start, which is why it costs an
equation set rather than an API break. Separate electron and ion energy
equations, an electron–ion equilibration source (Phase 0 item 1), and the
electron pressure contribution to the momentum flux.

**4. Electron thermal conduction.** Spitzer–Härm with a flux limiter, through
the Phase 2.1 solver. This depends on items 1 and 3.

**5. Tabulated EOS.** An IONMIX reader first (it is the simpler format and is
what FLASH's HEDP demos use), SESAME later if licensing permits. The Phase 1
EOS generalization is the prerequisite — do not attempt this before NSCBC and
the artificial-property model are EOS-agnostic.

**6. Multigroup radiation diffusion.** Flux-limited, gray first, then multigroup
with a group structure and an opacity interface mirroring the EOS contract.
Radiation groups become additional conserved components, supported by the
Phase 0 item 2 equation-set interface.

**7. Laser energy deposition.** Geometric-optics ray tracing with inverse
bremsstrahlung absorption. Architecturally independent of everything above — it
is a source term (Phase 0 item 1) plus a ray tracer that hands rays between
ranks. This item can precede the others if required by a specific experiment.

**8. MHD, if magnetized targets matter.** The ∇·B constraint with finite
differences requires hyperbolic (GLM) divergence cleaning, which adds a
conserved component and a source term — again, Phase 0. Constrained transport is
not natural in this discretization.

## Phase 3 — scale, portability, and adaptivity

**Patch-based AMR.** The long-term target, with the design constraints and the
open framework decision set out in [Adaptivity](#adaptivity--the-compact-scheme-constraint)
above. The proposed sequence is to implement the Miranda-style invertible
transfer-filter pair as two compact schemes and measure its conditioning under
grid-scale noise; introduce a patch abstraction at
a *single* level with conforming interfaces
(pure refactoring, independently verifiable against the existing bit-exact
tests, and immediately useful as multiblock geometry even if refinement never
follows); then non-conforming coarse-fine interfaces, spatial prolongation,
temporal prolongation for RK sub-cycling, and refinement criteria last.
Conforming multiblock precedes non-conforming refinement because the SBP–SAT
guarantees are clearest for conforming interfaces and defects are easier to
isolate. The existing
Cook δ⁴ sensors in `artificial.jl` are already exactly the "where is this
under-resolved" signal an AMR tagger needs.

**GPU through KernelAbstractions.jl.** The pointwise kernels — `primitives!`,
`assemble_fluxes!`, the RK update, and `compute_artificial!` — account for most
of the phase budget outside the line solves and consist of pointwise operations.
The distributed banded solve requires separate treatment: the batched
Thomas sweeps are sequential along the line but embarrassingly parallel across
lines, a structure compatible with GPU execution given the transposed layout
`lines_transposed.jl` already provides. Follow XCALibre's architecture; do not
maintain two codebases.

**Sequence GPU and AMR together, with AMR first.** Patch AMR breaks the "one
global array per field" assumption that currently pervades the code and replaces
it with many independently-sized blocks — which is precisely the shape a GPU
port requires, since a patch is a natural kernel-launch unit with a bounded working
set. A GPU port against monolithic arrays would duplicate the subsequent
decomposition work. Miranda's GPU implementation of patch AMR provides evidence
that the two capabilities can share an architecture.

**Mixed precision.** `Solver{T}` is already parameterized; the remaining blocker
is the concrete `Float64` type of halo buffers. Generalizing those buffers would
also verify the existing `{T}` parameterization.

**Parallel HDF5/XDMF output.** Partially delivered. The shared-file checkpoint
(`save_checkpoint_hdf5` / `load_checkpoint_hdf5!`) writes the state as one global
array and restores it onto any rank count, which also removes the
same-decomposition restriction on restart. The visualization dump and its XDMF3
sidecar are not yet written, so field output still uses the per-rank VTK path.

Two constraints govern the remaining work. A parallel libhdf5 must be built
against the run's own MPI, which no binary artifact supplies; where it is
absent the writer falls back to a serialized token relay that produces the same
file at O(P) cost. And VTKHDF, the obvious modern container, has no
RectilinearGrid or StructuredGrid support as of format version 2.5, so the
sidecar should be XDMF3.

**Multiblock geometry** follows from the patch abstraction: same-level patches
with separate line solves and interface exchange provide the required structure.
This also supports non-coordinate-surface geometry without an unstructured mesh.

**ThreadPinning wiring**, listed as open in `reference/CLUSTER.md` and only
validatable on the target cluster.

## Non-goals

The following capabilities are explicitly outside the project scope.

- **Oct-tree / cell-by-cell AMR.** Patch-based refinement is in long-term scope
  (Phase 3); tree refinement down to individual cells is not, because it is the
  variant that is incompatible with a line-coupled solve.
- **Unstructured meshes.** A line-coupled solve instead uses multiblock patches
  for geometric complexity.
- **Riemann solvers and flux limiters.** The artificial-fluid-property approach
  is the design commitment. Adding a Godunov path would double the maintenance
  surface for a capability that other codes do better.
- **A string-based PDE interpreter**, in the Pyranda style. Note this is a
  narrower exclusion than "no symbolic frontend": a macro-based frontend that
  *lowers* to the equation-set dispatch of Phase 0 remains possible and can
  provide Pyranda's concise problem
  specification while the generated code stays concretely typed and stays
  visible to `bench/audit.jl` and `bench/jetcheck.jl`. The exclusion is on
  runtime string evaluation, not on metaprogramming. Sequencing matters: the
  dispatch layer has to exist and be exercised by hand-written equation sets
  before a macro emits it, so the handwritten equation sets define the
  interface.
- **Being a general-purpose CFD library.** XCALibre and Trixi occupy that
  ground. CompactLES is scoped to compressible, variable-density, shock-driven
  mixing and implosion.

## Suggested ordering

The three highest-priority items are:

1. **Phase 0 in full.** Completed in July 2026 as a behavior-preserving refactor.
   It provides the extension interfaces required by later phases, with
   bit-identical convergence guards verifying the change.
2. **Phase 1's validation battery and EOS generalization.** Completed in July
   2026. It establishes the validation basis for the primary use case and
   enables Phase 2.5.
3. **Phase 2.1, the implicit diffusion solver.** This closes the principal
   architectural gap and also addresses the polar CFL restriction.

Laser ray tracing (2.7) is the exception to the ordering: it is independent of
everything else and could be pulled forward if a specific NIF experiment needs
it before the rest of the HED stack exists.

Two preparatory adaptivity items can precede Phase 3:

- **Build the Miranda transfer filter pair and measure its conditioning.** The
  coefficients and planning machinery already exist. The measurement should
  quantify the deconvolution response to grid-scale content near a transfer
  boundary, corresponding to the failure mode identified in the Fortran comments.
- **Design the patch infrastructure independently.** The operator design is
  settled from the kernels, and that was the part with numerics risk. Patch
  management, time sub-cycling, and refinement criteria dominate the remaining
  Phase 3 cost. They are absent from the public Pyranda tree and must be designed
  against this code's constraints: the fold
  restrictions, the sub-communicator machinery, the bit-exact-within-a-level
  guarantee.

No change to the grid arrangement is currently proposed. An earlier
version of this document argued that moving to a uniformly cell-centered grid
was a now-or-never prerequisite for adaptivity. That was extrapolated from a
single recent preprint working in the explicit WENO-family finite-difference
lineage, and it does not survive contact with the SBP–SAT literature, where
multiblock interfaces are node-coincident and non-conforming interfaces are
handled by interpolation operators rather than by a staggered arrangement. If
The established literature does not support that prerequisite. The current mixed
convention — half-offset only where a fold requires it — should stay until
there is a specific reason to change it.
