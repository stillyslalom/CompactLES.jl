# CompactLES — Roadmap

This roadmap compares CompactLES with related solvers and identifies the work
required to progress from shock-tube calculations to multiphysics relevant to
the National Ignition Facility (NIF). `README.md` covers usage, `DESIGN.md`
describes the numerics, and `CLAUDE.md` defines development procedures.
Completed phases are recorded in `reference/HISTORY.md`; the patch-AMR and GPU
implementation plan is `reference/AMR_GPU.md`.

## Contents

1. [Positioning](#positioning)
2. [Comparison](#comparison)
3. [Remaining blockers](#remaining-blockers)
4. [Model debts — regularization and validation](#model-debts--regularization-and-validation)
5. [Phase 2 — high-energy-density physics](#phase-2--hed-physics)
6. [Phase 3 — scale, portability, and adaptivity](#phase-3--scale-portability-and-adaptivity)
7. [Non-goals](#non-goals)
8. [Suggested ordering](#suggested-ordering)

## Positioning

The existing numerical method determines which comparisons are informative.

CompactLES offers tenth/sixth-order compact Padé derivatives, a
Gaitonde–Visbal compact filter, Cook artificial fluid properties in place of
Riemann solvers, five-stage low-storage RK45, structured curvilinear grids,
and MPI. These elements match the Miranda/Pyranda method except for
patch-based adaptivity and a GPU backend. They are appropriate for
variable-density turbulent mixing and shock–interface interaction, the primary
shock-tube and RM/RT use cases (Richtmyer–Meshkov and Rayleigh–Taylor, the two
instabilities this solver is aimed at throughout). Comparisons with Pyranda
therefore concern scope and interface, whereas comparisons with FLASH concern
the physics catalogue.

Three current capabilities distinguish CompactLES and constrain subsequent
design decisions:

- **The regularized coordinate-singularity treatment.** The half-offset grid
  plus parity/antipodal folds for the cylindrical axis, spherical origin, and
  spherical poles — with a discrete geometric conservation law (GCL) that
  preserves freestream to machine zero — is more than Pyranda or Miranda
  expose, and nothing in the Julia ecosystem has it. Converging-shock and
  spherical-implosion geometry is directly NIF-relevant, and this is the piece
  that is hard to rebuild.
- **The distributed compact solve.** The spike/reduced-interface banded solve
  reproduces the single-domain answer bit-for-bit at any rank count, with the
  reduced system factorized once at plan time. It is also, as argued below,
  the seed of the implicit infrastructure the HED physics needs.
- **The `Problem`/`Numerics` split.** The physics specification does not refer
  to ranks, halos, or the conserved layout. This separation is difficult to
  add after a multiphysics implementation has matured.

## Comparison

### Pyranda (LLNL)

The closest relative — the Miranda mini-app, Python-driven with a Fortran
kernel, 10th-order compact plus RK45, MPI, aimed at arbitrary hyperbolic
systems.

| | Pyranda | CompactLES |
|---|---|---|
| Spatial scheme | C10 compact | C6 default, C10, custom `CompactScheme` |
| Regularization | artificial bulk viscosity | Cook μ\*, β\*, κ\*, per-species D\* |
| Problem specification | Python domain-specific language (DSL) over symbolic PDE strings | typed `Problem` / `Numerics` |
| Geometry | Cartesian, curvilinear, immersed boundaries | Cartesian, cylindrical, spherical with regularized singularities; stretch maps |
| Lineage | Miranda validation heritage | validated against analytic references only (see model debt 3) |

Pyranda also provides a reference implementation, because it carries Miranda's
Fortran kernels in `pyranda/parcop/`, including operators Pyranda itself does
not expose. The readings of those sources are recorded elsewhere: level
transfer in `reference/AMR_GPU.md`, the artificial-property and filter path in
`reference/CALIBRATION.md`, the immersed-boundary method in
`reference/IMMERSED.md`, and what the August 2026 pass changed in
`reference/HISTORY.md`.

Three structural comparisons came out of that reading and need no action. The
distributed compact solve is the same algorithm as Miranda's, down to the
factorized interface system and the single `Allgather` per application;
Miranda's additional gather-solve-scatter mode (`directcom = 2`) is the shape
to reach for if the interface solve ever appears in a scaling curve. Symmetry
is handled by equivalent means, Miranda folding parity into the coefficient
tables at setup where the fold plans plus `sigflux` carry the same algebra. And
the freestream claim above holds against the source: Miranda's spherical metric
sources use analytic `1/tan θ` and `1/sin θ`, so the discrete redefinition of
`cot_over_r` is genuinely beyond the reference, whose r = 0 is an ordinary
reflective symmetry plane rather than a half-offset antipodal fold.

#### Open work from the source comparison

In rough order of expected value. The sensor fields stood at the head of this
list and are now measured and closed (`reference/HISTORY.md`): μ\* from the
velocity components, β\* from the dilatation, and MAX against Σ_d over
directions. All three are `ArtParams` settings and none of them is a default.

1. **Conservative filtering on non-Cartesian metrics.** The reference filters
   the volume-weighted field and divides by a cell volume passed through the
   same filter once at setup, which reproduces constants exactly on a
   non-uniform metric; `filter_state!` filters the conserved components
   unweighted. Uniform Cartesian results are unaffected by construction, so the
   measurement is the converging cases plus a conservation-defect probe.
2. **A `C_mu` refit under the adopted smoother and detector.** Taylor–Green at
   64³ shows the μ\* channel moving 4.0% → 4.5% of the sink under the Gaussian,
   which does not demand a refit but does not rule one out. **Blocked** on the
   same 3-D campaign as the filter calibration, since no case in the
   one-dimensional battery gives the shear channel anything to do. The `C_beta`
   half is done and retained 1.0 under both detectors
   (`reference/CALIBRATION.md`); it also established that no `C_beta` in
   0.25–4 recovers the spherical-origin ceiling under `:d8`. The origin cell
   has since been accounted for as well, retiring the fold closure, so the
   detector default now rests on which configuration survives the startup
   excursion rather than on any unmeasured numerics.
3. **Directional artificial bulk viscosity**, with the per-direction diffusive
   timestep limit that goes with it. **Blocked**: no anisotropic or strongly
   stretched case exists in the validation battery, so measuring it against the
   present cases would produce a null result that means nothing. Build the case
   first.
4. **Anchored-difference closure rows.** The reference writes boundary rows as
   differences from the anchor point so a constant is annihilated exactly in
   floating point rather than by cancellation. **Gated** on measuring the
   present residual first; if it is 1e-16 times the field scale the change is
   cosmetic and should be recorded as such rather than made.

**Capabilities absent from CompactLES:** a DSL for defining a new PDE system
in ten lines without touching the solver; immersed boundaries, which provide
non-coordinate-surface geometry without unstructured meshes; and the
credibility of being the mini-app for a production code.

**Equation specification.** Pyranda's string-based equation interpreter
provides generality at the cost of type safety and of any physics that is not
a flux divergence plus a source. Julia multiple dispatch on an equation-set
type, as used by Trixi and adopted in Phase 0, provides the alternative while
retaining inference and allocation discipline (`bench/audit.jl`,
`bench/jetcheck.jl`). If a symbolic frontend is later wanted for ergonomics,
the route is a macro that *emits* an equation set at parse time; runtime
evaluation of equation strings is excluded (see Non-goals).

Immersed boundaries are designed in `reference/IMMERSED.md`: a sharp IB cut
cell is fundamentally at odds with a line-global compact operator, so the
design commits to the diffuse family — a graded post-stage state blend that
unifies smeared Brinkman penalization with Pyranda's level-set reset — staged
with validation gates.

### FLASH

FLASH uses second-order unsplit PPM (the piecewise parabolic method), so the
relevant comparison concerns its physics catalogue and adaptive mesh refinement
(AMR) rather than numerical order.

FLASH's high-energy-density physics (HEDP) capability is a set of units: 3T
hydrodynamics (separate ion, electron and radiation temperatures) with
electron–ion equilibration, multigroup radiation diffusion, tabulated
multi-species EOS and opacities (IONMIX), electron thermal conduction through
an implicit diffusion solver, laser energy deposition by geometric-optics ray
tracing, MHD, and anisotropic magnetized transport coefficients — all on a
block-structured AMR mesh, validated against HYDRA. This capability set defines
the long-term target for NIF applications and represents approximately a decade
of physics implementation.

Two structural patterns are relevant:

- **Physics-unit decomposition.** FLASH's physics units are separately
  configurable, each owning its own state and contributing to the update
  through a defined interface. This is why FLASH could grow 3T and multigroup
  diffusion (MGD) without rewriting hydro. CompactLES's equivalent is the
  Phase 0 seams (`reference/HISTORY.md`).
- **Patch-based adaptivity.** FLASH's PARAMESH/AMReX refinement is oct-tree
  block AMR built around a second-order finite-volume update. The version
  that belongs here is *patch-based* (logically rectangular blocks of uniform
  resolution, SAMRAI-style), which is what Miranda uses and which makes
  adaptivity compatible with a compact scheme. `reference/AMR_GPU.md` carries
  the constraint analysis and the implementation plan.

### Trixi.jl

The Julia reference for high-order conservation laws: nodal discontinuous
Galerkin spectral element (DG-SEM) on a quad/octree with AMR, `p4est` for
unstructured curved meshes, entropy-stable and kinetic-energy-preserving split
forms, shock capturing with positivity limiting, a large equation-set
catalogue, and OrdinaryDiffEq.jl integration.

**Relevant design pattern:** the `equations` type parameter. Trixi's solvers
are generic over an equation set that owns the variable count, names, flux
functions, and conversions; adding MHD to Trixi did not require rewriting
Euler. Phase 0 adopted this interface.

**CompactLES advantages for the target problems:** compact finite differences
require less work per degree of freedom than DG at comparable resolving power
for smooth, volume-filling turbulence and have lower memory traffic. Trixi
has nothing resembling Cook artificial fluid properties, which is the right
regularization for material interfaces at high Atwood number, and no
regularized polar/spherical singularity treatment.

**Applicable external component:** Trixi uses OrdinaryDiffEq.jl for time
integration. A CompactLES IMEX scheme in Phase 2 could use the tested
IMEX-ARK tableaus in the SciML ecosystem.

### XCALibre.jl

Second-order unstructured finite volume, incompressible and compressible,
RANS and LES, OpenFOAM/unv mesh import, CPU threads or GPU through
KernelAbstractions.jl. Different accuracy class and problem domain
(engineering CFD on complex geometry), so it is not a competitor. It provides
the GPU architecture adopted in `reference/AMR_GPU.md`: a single
KernelAbstractions.jl codebase targeting NVIDIA, AMD, and Intel while
retaining a CPU path. Oceananigans.jl is a second example of a Julia
structured-grid solver designed this way.

## Remaining blockers

**Implicit and elliptic solvers are absent.** Radiation diffusion and electron
conduction are parabolic and stiff; explicit treatment is not an option at HED
conditions, where the conduction timestep can be orders of magnitude below the
acoustic one. This is the largest remaining architectural gap. `tridiag.jl`
and `banded.jl` already implement a distributed banded line solve with
cross-rank coupling and a pre-factorized reduced interface system — precisely
the kernel an alternating-direction implicit (ADI) or line-relaxation
diffusion solver needs and precisely the smoother a geometric multigrid
requires. The implicit implementation should be built on that machinery rather
than on a separate linear-algebra dependency.

**The regularization model carries one structural debt.** The July 2026
calibration study established that the compact filter, not the Cook
properties, is the primary stabilizer at every measured resolution, while the
filter itself has never been calibrated and its dissipation is applied per
filter pass rather than per unit time (`reference/CALIBRATION.md`). Every
mixing result this solver produces is currently conditional on it, which is why
it is a work item below rather than a known limitation. The second debt in this
class, β\* keyed on the strain magnitude rather than the dilatation, has been
measured and closed (`reference/HISTORY.md`).

What that work left open is the converging-shock `cfl ≤ 0.15` ceiling itself,
and **every discretization-order candidate for it has now been measured and
ruled out**. The timestep predictor, the dilatation sensor and the sensor reach
went first. The fold closure, which stood last and longest, is not third order:
it is sixth to seventh order at the fold, and the third-order figure attributed
to it belongs to the outer wall through a global max norm. The companion
reading, that a selective detector is blind at the fold, fails on the same
measurement, since β\* at the origin reaches the line maximum during the
excursion that fails.

The origin cell instead evacuates during a startup transient that lands at a
fixed physical time regardless of resolution and that every configuration
passes through. That makes the ceiling a robustness problem at a symmetry cell
rather than a numerics problem. Model debt 2 below has since supplied an
instrument for that question: `StepControl.floor_ratio` counts sub-floor cells
and can repair them. The converging geometries have not been measured through
it, only the planar case. Two
mechanisms are live and neither is yet demonstrated: β\* is proportional to the
density, so it collapses on exactly the cell that is thinning, and the compact
filter is applied per step rather than per unit time, which is model debt 1 and
which the `C_beta` ladder implicates independently — under `:d8` at reduced β\*
the cylindrical axis fails *below* a CFL rather than above one
(`reference/CALIBRATION.md`).

**One EOS assumption survives the Phase 1 generalization.** The artificial
conductivity is still built as (ρc/T_ion)·sensor for every gas model, which is
singular at a cold ambient. The scale is an EOS dispatch point, so a tabular
or condensed-matter model can supply its own; making the gas-model form
non-singular is a numerics decision, tracked in `reference/CALIBRATION.md`.

## Model debts — regularization and validation

These items carry the physics credibility of the solver for its primary
mission. They are ordered by information gained per unit effort. Items 1 and 3
change guarded numbers, so each concludes by re-baselining `test/validation.jl`
and updating `reference/CALIBRATION.md`. Item 2 was expected to change them and
does not, its default scope being measured bit-identical to the unfloored
solver.

**1. Filter calibration and dt-consistency.** Two coupled problems, and the
second is now done. `compact_filter(0.45)` applied every step has never been
fitted to anything, while supplying 37–87% of the measured energy sink and
being necessary and sufficient for stability (`reference/CALIBRATION.md`).

The dt-consistency half is delivered. The filter removed energy per
*application* rather than per unit time, measured at a factor of 3.93 across a
4× CFL change on a case where the filter is the only sink, so the subgrid
dissipation did not converge as dt → 0 at fixed resolution. `Numerics.filter_cfl`
supplies the per-unit-time formulation, `Q ← (1−w)Q + w·F(Q)` with `w ∝ dt`,
which holds the loss constant to six significant figures over the same range
and also removes the truncated-final-step artifact, since a shortened step now
filters proportionally less. It is off by default. The measurement that makes
this matter for the rest of the debt: with `C_mu` held fixed, halving the CFL
moves the μ\* share of the Taylor–Green sink by 29% relative, so a constant
fitted under the unrelaxed formulation is only reproducible at the CFL it was
fitted at. It should also resolve the cross-level cadence question raised in
`reference/AMR_GPU.md`, which has not been checked.

What remains is the calibration proper: fit α and cadence against the digitized
van Rees −dKE/dt(t) history and spectra at 128³ (the digitization is already
listed in the calibration remainders), and measure the validation battery's
sensitivity to cadence. Deliverable: the
filter section of `reference/CALIBRATION.md` brought to the same standing as
the four-constant tables; it exists now and carries the dt-consistency half.
Cluster time is required; budget runs per the usual discipline.

Whether `filter_cfl` becomes the default is part of that fit rather than a
separate decision. It changes what α means, since under the relaxation α and
the reference CFL set the dissipation jointly, so fitting α first under the
old formulation and switching afterwards would waste the fit.

**2. A positivity failsafe — delivered** (`reference/HISTORY.md`).
`StepControl.floor_ratio` enables it and `floor_scope` sets how much of the
state space it insists on; both are off by default, and the repair narrows
rather than violates the Riemann-solver non-goal below, since the scheme is
unchanged and the failsafe recovers from states it has already left.

The delivery changed the scope of the repair, because building the floor made
the condition measurable and the measurement did not support repairing all of
it. The negative internal energy that motivated the debt is neither a rounding
artifact nor a state the scheme cannot handle: the shipped Noh ν = 1 case
carries 24991 cell-steps of it, reaching −718 ambient units, while density and
total energy stay positive throughout and the run still reaches its plateau to
within 0.07%. Forcing it positive costs a 5% velocity damping on the worst cell,
and the run fails at step 18. The default scope therefore repairs only what no
frame can represent and *counts* the rest, which closes the part of the debt the
evidence supports: the condition was invisible, floored silently by
`primitives!`. → `reference/CALIBRATION.md`

The measurement leaves one question open rather than answering it. A calculation
whose wall region runs as a pressureless layer for its whole duration reaches
the right plateau for reasons nobody has checked, and the `:internal_energy`
scope is the instrument for asking what that costs. The question belongs with
the filter calibration below rather than with the failsafe.

**3. External validation.** Everything to date compares against analytic
references or this code's own high-resolution profiles — the comparison table
above says so. Two campaigns close the gap. First, a direct
CompactLES-vs-Pyranda comparison (Pyranda is runnable): TGV at Re = 1600 and
one RM shock-tube case, comparing dissipation histories and mix widths, which
converts "matches the Miranda method" from a design claim into a measured
one. Second, one published RM experiment as a data target — community
benchmark cases with published initial-condition specifications are
accessible, and asking the originating groups for specifications beats
reverse-engineering them from figures. Deliverable: a validation section in
the docs comparing against something this code did not produce.

**4. Turbulent inflow generation.** RM/RT comparisons with experiments need a
perturbed or turbulent inflow. The digital-filter method (Klein et al., 2003)
or a synthetic-eddy variant fits the existing frontend with no solver surgery:
a generator utility producing an `(x, y, z, t) -> Prim` closure consumed by
`DirichletBC` or the `NSCBCInflowBC` target. Modest scope; pairs naturally
with item 3's experiment campaign.

**5. NSCBC completion.** Add the Yoo–Im transverse terms to the inflow
correction, mirroring the outflow implementation (the README documents the
asymmetry). This is the only piece outstanding; the setup-time validation of
the face and of the target composition landed with the near-term corrections
(`reference/HISTORY.md`).

**6. Temperature-dependent transport.** The NASA CEA transport table ships in
`data/` but is not connected; `Transport` is constant-coefficient with one
Schmidt number for all species. Implement the coefficient reader, per-species
μ_k(T) and λ_k(T) evaluations, a Wilke-type mixture rule, and
mixture-averaged diffusivities (unity-Lewis fallback retained). Structure it
the way the EOS contract is structured, as a dispatchable transport model
consumed behind the existing function barrier, so a plasma transport model
(Phase 2) is a new instance rather than a rewrite. Until this lands,
"multicomponent" accurately describes the thermodynamics and the species
transport equations, not the transport coefficients; the README should say
so.

**7. Wall treatment — deliberately deferred.** The target problems (RM/RT
mixing, converging shocks, implosions) are wall-free or slip-walled;
`NoSlipWallBC` exists for verification cases. Wall-resolved or wall-modeled
LES is out of scope until a target problem requires it. Recorded here so the
gap is a decision rather than an oversight.

## Phase 2 — HED physics

HED is high-energy-density: the pressure and temperature regime of inertial
confinement fusion, where radiation transport and electron conduction are
comparable to hydrodynamics in importance.

This phase supplies the NIF-oriented physics. Its items are ordered so that
each produces an independently usable capability.

**1. Implicit diffusion infrastructure.** Everything else here depends on it.
Build it on the existing distributed banded solve: ADI or line-Jacobi sweeps
as the smoother, wrapped in either a geometric multigrid or a Krylov method
(Krylov.jl is the mature, dependency-light Julia option). Validate against a
manufactured heat-conduction solution in every metric and against the
existing freestream tests. Implicit treatment of the θ direction also
addresses the azimuthal CFL restriction documented in the README.

**2. IMEX time integration.** With the implicit solver in hand, an IMEX-ARK
scheme pairing the existing explicit RK45 on hydro with implicit diffusion.
The caller-owned `Workspace` (Phase 0) is the prerequisite. Existing tableaus
from the SciML ecosystem should be evaluated before implementing new ones.

**3. Three-temperature hydrodynamics.** T_ele and T_rad alongside T_ion — the
naming convention has reserved this from the start, which is why it costs an
equation set rather than an API break. Separate electron and ion energy
equations, an electron–ion equilibration source (through the source
interface), and the electron pressure contribution to the momentum flux.

**4. Electron thermal conduction.** Spitzer–Härm with a flux limiter, through
the Phase 2.1 solver. Depends on items 1 and 3.

**5. Tabulated EOS.** An IONMIX reader first (the simpler format, and what
FLASH's HEDP demos use), SESAME later if licensing permits. The EOS contract
is ready (`reference/HISTORY.md`); this is data plumbing and a table
interpolator, not solver surgery. The κ\* scale for a tabular model comes
through the existing `art_conductivity_scale` dispatch point.

**6. Multigroup radiation diffusion.** Flux-limited, gray first, then
multigroup with a group structure and an opacity interface mirroring the EOS
contract. Radiation groups become additional conserved components, supported
by the equation-set interface.

**7. Laser energy deposition.** Geometric-optics ray tracing with inverse
bremsstrahlung absorption. Architecturally independent of everything above —
a source term plus a ray tracer that hands rays between ranks. This item can
precede the others if required by a specific experiment.

**8. MHD, if magnetized targets matter.** The ∇·B constraint with finite
differences requires hyperbolic (GLM) divergence cleaning, which adds a
conserved component and a source term. Constrained transport is not natural
in this discretization.

## Phase 3 — scale, portability, and adaptivity

**Patch-based AMR and the GPU port** are specified in `reference/AMR_GPU.md`,
including the compact-scheme constraints, the Miranda transfer-operator
evidence, the SBP–SAT fallback, staged implementation with verification gates,
and the argument for sequencing them together with AMR's patch refactor
first. Summary of the sequence: transfer operators as compact schemes
(independent, small); the patch abstraction at a single level with conforming
interfaces, carrying the storage-type generalization for GPU; static
two-level refinement at a global timestep; KernelAbstractions pointwise
kernels and device line solves; subcycling, sensor-driven tagging, and
regridding last. Conforming multiblock is independently useful geometry even
if refinement never follows.

**Mixed precision.** The storage blocker is cleared: AMR Stage 2 / GPU Stage
G0 (`reference/AMR_GPU.md`) delivered `Decomp{T}` with `T`-typed halo and
pair buffers, backend-routed allocation (`field(backend, decomp)`), and
eltype-preserving `viz.jl` extraction, so `T` now reaches every field array.
What remains for an actual `Float32` run is exercising it: the constants
sprinkled through the physics (`1e-300` clamps, `DUCROS_EPS`) assume Float64
range, and nobody has measured which phases tolerate reduced precision.

**Parallel HDF5/XDMF time series.** Single dumps and shared-file checkpoints
are delivered (`reference/HISTORY.md`). Two pieces remain, the first
specified below; the second is validating the collective write on a machine
with a parallel libhdf5 built against the run's MPI — only the serialized
token-relay fallback has been exercised.

### `FieldWriter` over HDF5

`FieldWriter` gains `format = :vtk` or `:hdf5`. `_write_dump!` already exists
as the seam and dispatches on it, calling `save_hdf5` for the latter; the
core stub in `src/hdf5.jl` supplies the error when the extension is absent,
so no availability check is needed at the call site. `fields`, `stride` and
`slice` flow through unchanged. The extension seam is the same one `save_hdf5`
uses: a core stub `_write_xdmf_collection!` routed through `_hdf5_required`,
with the real method in `ext/CompactLESHDF5Ext.jl`.

The collection file is where the work is. `.pvd` has no XDMF equivalent that
merely lists frames: a temporal collection is `<Grid GridType="Collection"
CollectionType="Temporal">` with one full `<Grid>` inlined per frame, each
carrying its own `<Time>`, topology, geometry and one `<Attribute>` per
field. Write it in full on every dump, for the reason `_write_pvd` already
does — a run killed by the scheduler then still leaves a collection naming
every completed frame, where an appended file would lack its closing tags and
not open at all. XInclude of the per-frame `.xmf` files was considered and
rejected: reader support for it is inconsistent, and the per-frame files are
worth keeping openable on their own.

#### The fork

Inlining an `<Attribute>` per frame requires each field's name and component
count, and `:Y` and `:D_art` expand to one entry per species while the vector
fields carry three components. Recomputing that in the collection
writer would duplicate the naming in `vtk_field_entries` and drift from it.
Resolve it by having `save_hdf5` hand back the `(name, ncomponents)`
descriptors it wrote and caching them on the writer at the first dump. The
descriptors are identical across frames, since `fields`, `stride` and `slice`
are fixed for the life of a writer.

Testing should assert that the frames land on the [`EveryTime`](@ref)
instants, that the collection holds one `<Grid>` per frame with the matching
`<Time>` value, and that each references `prefix_NNNN.h5` by a path relative
to the collection's own directory. Run it decomposed as well: the per-frame
writes are collective and only rank 0 writes the collection.

Slicing interacts with the second of those. A rank holding no part of the
requested plane currently skips its write, which is correct under the
serialized backend. A collective write instead requires every rank to call
`H5Dwrite` even with nothing selected, so enabling the parallel backend with
slicing needs an empty selection (`H5S_SELECT_NONE`) rather than a skipped
call. The VTK path has no such constraint, since each rank writes its own
file.

**Multiblock geometry** follows from the patch abstraction (AMR Stage 2):
same-level patches with separate line solves and interface exchange provide
the required structure without an unstructured mesh.

**Immersed boundaries** are specified in `reference/IMMERSED.md`: level-set
bodies imposed by a graded post-stage state blend (exact Brinkman relaxation,
degenerating to Pyranda-style hard reset at η = 0), with force and heat-flux
diagnostics from the imposition bookkeeping. The design is
ordering-independent of the patch refactor, since the imposition is pointwise
and mask-driven, so it can be implemented on today's arrays and ported
mechanically. Multiblock and IB are complements, not alternatives: multiblock
handles geometry that must be accurate (coordinate-aligned blocks at full
closure order), IB handles geometry that must merely exist (first-order at
the interface, any shape, no meshing).

**ThreadPinning wiring**, listed as open in `reference/CLUSTER.md` and only
validatable on the target cluster.

## Non-goals

The following capabilities are explicitly outside the project scope.

- **Oct-tree / cell-by-cell AMR.** Patch-based refinement is in long-term
  scope; tree refinement down to individual cells is not, because it is the
  variant that is incompatible with a line-coupled solve.
- **Unstructured meshes.** A line-coupled solve instead uses multiblock
  patches for geometric complexity.
- **Riemann solvers and flux limiters as the scheme.** The
  artificial-fluid-property approach is the design commitment; adding a
  Godunov path would double the maintenance surface for a capability other
  codes do better. This exclusion does not cover the instrumented positivity
  failsafe of model debt 2, which recovers from states the scheme has
  already left rather than altering the scheme.
- **A string-based PDE interpreter**, in the Pyranda style. This is narrower
  than "no symbolic frontend": a macro that *lowers* to the equation-set
  dispatch remains possible and can provide Pyranda's concise problem
  specification while the generated code stays concretely typed and visible
  to `bench/audit.jl` and `bench/jetcheck.jl`. The exclusion is on runtime
  string evaluation. Sequencing matters: hand-written equation sets define
  the interface before any macro emits it.
- **Being a general-purpose CFD library.** XCALibre and Trixi occupy that
  ground. CompactLES is scoped to compressible, variable-density,
  shock-driven mixing and implosion.

Wall-modeled LES is deferred rather than excluded; see model debt 7.

## Suggested ordering

The near-term corrections, the dilatation-gated β\* experiment, and the
reference-implementation pass that headed this list are done
(`reference/HISTORY.md`). That leaves:

1. **Filter calibration** (model debt 1), which every other physics number is
   conditional on. Requires cluster time; run it alongside the Pyranda
   comparison, which needs the same machines and cases. The dt-consistency half
   is done and `filter_cfl` exists; what is left is the α and cadence fit, and
   the decision on whether to ship the relaxation belongs to that fit.
2. **External validation** (model debt 3), promoted early because it changes
   how much the remaining calibration work can be trusted.
3. **Phase 2.1, the implicit diffusion solver** — the principal architectural
   gap, which also addresses the polar CFL restriction.
4. **AMR/GPU Stages 1–4 are delivered** (`reference/HISTORY.md` and the
   status blocks in `reference/AMR_GPU.md`): the transfer operators, the
   patch abstraction at a single level carrying the G0 storage
   generalization, static two-level refinement at a global timestep, and
   Berger–Oliger subcycling with tagging-driven regridding of a single
   moving region (all serial first cuts). Stages 2–4 were originally
   sequenced after the model debts above; they were executed early by
   explicit decision, kept behavior-preserving (the unrefined single-patch
   path is bit-identical), so physics changes rebase onto them cleanly.
   Next per the plan's ordering: G1 (KernelAbstractions pointwise kernels),
   plus distributing the level transfer, which also gates the Stage 4 cost
   demonstration on a 3-D mixing case.

The open items from the source comparison
([above](#open-work-from-the-source-comparison)) sit alongside these rather
than inside the ordering. Everything remaining there is now either blocked on a
missing case or gated on a measurement that has not been taken. The one
exception, the `C_beta` refit under `:d8`, is closed: the constant holds at 1.0
and the detector default turns out to rest on the spherical origin rather than
on the fit.

Two items sit outside the ordering because they are independent of everything
above and pullable on demand: laser ray tracing (Phase 2.7), if a specific
experiment needs it before the rest of the HED stack exists, and immersed
boundaries (`reference/IMMERSED.md`) Stage 1, when a target problem first
needs non-coordinate-surface geometry — its main sequencing constraint is
that the Stage 2 calibration sweep requires the same measurement discipline
as `bench/artcal.jl`, not that it depends on other work.
