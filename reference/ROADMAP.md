# CompactLES — Roadmap

This roadmap compares CompactLES with related solvers and identifies the work
required to progress from shock-tube calculations to NIF-relevant multiphysics.
`README.md` covers usage, `DESIGN.md` describes the numerics, and `CLAUDE.md`
defines development procedures. Completed phases are recorded in
`reference/HISTORY.md`; the patch-AMR and GPU implementation plan is
`reference/AMR_GPU.md`.

## Contents

1. [Positioning](#positioning)
2. [Comparison](#comparison)
3. [Remaining blockers](#remaining-blockers)
4. [Model debts — regularization and validation](#model-debts--regularization-and-validation)
5. [Phase 2 — HED physics](#phase-2--hed-physics)
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
shock-tube and RM/RT use cases. Comparisons with Pyranda therefore concern
scope and interface, whereas comparisons with FLASH concern the physics
catalogue.

Three current capabilities distinguish CompactLES and constrain subsequent
design decisions:

- **The regularized coordinate-singularity treatment.** The half-offset grid
  plus parity/antipodal folds for the cylindrical axis, spherical origin, and
  spherical poles — with a discrete GCL that preserves freestream to machine
  zero — is more than Pyranda or Miranda expose, and nothing in the Julia
  ecosystem has it. Converging-shock and spherical-implosion geometry is
  directly NIF-relevant, and this is the piece that is hard to rebuild.
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
| Problem specification | Python DSL over symbolic PDE strings | typed `Problem` / `Numerics` |
| Geometry | Cartesian, curvilinear, immersed boundaries | Cartesian, cylindrical, spherical with regularized singularities; stretch maps |
| Lineage | Miranda validation heritage | validated against analytic references only (see item 4 of the model debts) |

Pyranda also provides a reference implementation because it carries Miranda's
Fortran kernels in `pyranda/parcop/`, including operators Pyranda itself does
not expose. The AMR level-transfer analysis derived from those sources now
lives in `reference/AMR_GPU.md`.

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

Immersed boundaries are now designed in `reference/IMMERSED.md`: a sharp IB
cut cell is fundamentally at odds with a line-global compact operator, so the
design commits to the diffuse family — a graded post-stage state blend that
unifies smeared Brinkman penalization with Pyranda's level-set reset — staged
with validation gates.

### FLASH

FLASH uses second-order unsplit PPM, so the relevant comparison concerns its
physics catalogue and AMR rather than numerical order.

FLASH's HEDP capability is a set of units: 3T hydrodynamics with electron–ion
equilibration, multigroup radiation diffusion, tabulated multi-species EOS and
opacities (IONMIX), electron thermal conduction through an implicit diffusion
solver, laser energy deposition by geometric-optics ray tracing, MHD, and
anisotropic magnetized transport coefficients — all on a block-structured AMR
mesh, validated against HYDRA. This capability set defines the long-term
target for NIF applications and represents approximately a decade of physics
implementation.

Two structural patterns are relevant:

- **Physics-unit decomposition.** FLASH's physics units are separately
  configurable, each owning its own state and contributing to the update
  through a defined interface. This is why FLASH could grow 3T and MGD without
  rewriting hydro. CompactLES's equivalent is the Phase 0 seams
  (`reference/HISTORY.md`).
- **Patch-based adaptivity.** FLASH's PARAMESH/AMReX refinement is oct-tree
  block AMR built around a second-order finite-volume update. The version
  that belongs here is *patch-based* (logically rectangular blocks of uniform
  resolution, SAMRAI-style), which is what Miranda uses and which makes
  adaptivity compatible with a compact scheme. `reference/AMR_GPU.md` carries
  the constraint analysis and the implementation plan.

### Trixi.jl

The Julia reference for high-order conservation laws: nodal DG-SEM on a
quad/octree with AMR, `p4est` for unstructured curved meshes, entropy-stable
and kinetic-energy-preserving split forms, shock capturing with positivity
limiting, a large equation-set catalogue, and OrdinaryDiffEq.jl integration.

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
the kernel an ADI or line-relaxation diffusion solver needs and precisely the
smoother a geometric multigrid wants. The implicit implementation should be
built on that machinery rather than on a separate linear-algebra dependency.

**The regularization model carries two structural debts.** The July 2026
calibration study (`reference/CALIBRATION.md`) established that the compact
filter, not the Cook properties, is the primary stabilizer at every measured
resolution, while the filter itself has never been calibrated and its
dissipation is applied per filter pass rather than per unit time. Separately,
the β\* sensor is built from the strain magnitude rather than the dilatation,
which the shock-capturing literature superseded for documented reasons. Both
are elevated from "known limitations" to work items in the
[model debts](#model-debts--regularization-and-validation) section, because
every mixing result this solver produces is currently conditional on them.

**One EOS assumption survives the Phase 1 generalization.** The artificial
conductivity is still built as (ρc/T_ion)·sensor for every gas model, which is
singular at a cold ambient. The scale is an EOS dispatch point, so a tabular
or condensed-matter model can supply its own; making the gas-model form
non-singular is a numerics decision, tracked in `reference/CALIBRATION.md`.

## Model debts — regularization and validation

These items carry the physics credibility of the solver for its primary
mission. They are ordered by information gained per unit effort, and items 1–3
change guarded numbers, so each concludes by re-baselining
`test/validation.jl` and updating `reference/CALIBRATION.md`.

**1. A dilatation-based β\* with a shock switch.** The current sensor feeds
β\* from |∇⁴S|, so artificial bulk viscosity fires on vortical structures and
on expansions, not only in compression. The literature refinement
(Mani, Larsson & Moin, JCP 228, 2009; adopted in the Miranda lineage) builds
the bulk sensor from the dilatation, gated to negative dilatation with a
Ducros-style vorticity discriminator. Two existing measurements point at this
change: the Shu–Osher wave-train amplitude loss that pulls against every
constant, and the calibration study's own diagnosis of the `cfl ≤ 0.15`
restriction — "the sensor is not switching on early enough, or β\* is not
reaching far enough ahead of the front." A compression-keyed sensor is the
literature's answer to exactly that failure. Implementation is cheap:
`grad_u` is already available where the sensor is built, so the dilatation is
free; add a sensor selector to `ArtParams` (`:strain`, the current default,
and `:dilatation`), sweep with `bench/artcal.jl`, and test specifically
whether the Noh CFL ceiling moves. If it does, this single change retires the
worst documented limitation.

**2. Filter calibration and dt-consistency.** Two coupled problems. First,
`compact_filter(0.45)` applied every step has never been fitted to anything,
while supplying 37–87% of the measured energy sink and being necessary and
sufficient for stability (`reference/CALIBRATION.md`). Second — and stated
nowhere until now — the filter removes energy per *application*, not per unit
time, so the effective subgrid dissipation depends on the CFL number and
`filter_interval`, and does not converge as dt → 0 at fixed resolution. The
truncated-final-step artifact in `bench/tgv_energy.jl` (an 18% peak
overestimate from one short step) is a symptom. Work items: fit α and cadence
against the digitized van Rees −dKE/dt(t) history and spectra at 128³ (the
digitization is already listed in the calibration remainders); measure the
validation battery's sensitivity to cadence; and evaluate a per-unit-time
formulation — filter application as a relaxation `Q ← (1−w)Q + w·F(Q)` with
`w ∝ dt` — which would also resolve the cross-level cadence question raised
in `reference/AMR_GPU.md`. Deliverable: a filter section of
`reference/CALIBRATION.md` with the same standing as the four-constant
tables. Cluster time is required; budget runs per the usual discipline.

**3. A positivity failsafe.** `StepControl` detects positivity loss and can
roll back, but rollback recovers only abrupt failures; gradual degradation
lands the savepoint on an already-corrupt state, and sweeps still burn wall
time. Add a last-resort, conservation-aware local floor (clip negative
partial densities, renormalize, adjust energy consistently), applied only on
detection, counted, and reported loudly — never silent. This narrows rather
than violates the Riemann-solver non-goal below: the scheme is unchanged; the
failsafe is an instrumented recovery from states the scheme has already left.
The immediate beneficiaries are calibration sweeps and the AMR development
runs ahead.

**4. External validation.** Everything to date compares against analytic
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

**5. Turbulent inflow generation.** RM/RT comparisons with experiments need a
perturbed or turbulent inflow. The digital-filter method (Klein et al., 2003)
or a synthetic-eddy variant fits the existing frontend with no solver surgery:
a generator utility producing an `(x, y, z, t) -> Prim` closure consumed by
`DirichletBC` or the `NSCBCInflowBC` target. Modest scope; pairs naturally
with item 4's experiment campaign.

**6. NSCBC completion.** Add the Yoo–Im transverse terms to the inflow
correction, mirroring the outflow implementation (the README documents the
asymmetry). This is now the only piece outstanding: the setup-time validation of
the face and of the target composition landed with the near-term corrections
(`reference/HISTORY.md`).

**7. Temperature-dependent transport.** The NASA CEA transport table ships in
`data/` but is not connected; `Transport` is constant-coefficient with one
Schmidt number for all species. Implement the coefficient reader, per-species
μ_k(T) and λ_k(T) evaluations, a Wilke-type mixture rule, and
mixture-averaged diffusivities (unity-Lewis fallback retained). Structure it
the way the EOS contract is structured — a dispatchable transport model
consumed behind the existing function barrier — so a plasma transport model
(Phase 2) is a new instance rather than a rewrite. Until this lands,
"multicomponent" accurately describes the thermodynamics and the species
transport equations, not the transport coefficients; the README should say
so.

**8. Wall treatment — deliberately deferred.** The target problems (RM/RT
mixing, converging shocks, implosions) are wall-free or slip-walled;
`NoSlipWallBC` exists for verification cases. Wall-resolved or wall-modeled
LES is out of scope until a target problem requires it. Recorded here so the
gap is a decision rather than an oversight.

## Phase 2 — HED physics

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

**Mixed precision.** `Solver{T}` is already parameterized; the concrete
`Float64` halo buffers are the blocker. This generalization is folded into
AMR Stage 2 / GPU Stage G0 (`reference/AMR_GPU.md`) so the structs are
rewritten once.

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
`slice` flow through unchanged.

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

THE FORK. Inlining an `<Attribute>` per frame requires each field's name and
component count, and `:Y` and `:D_art` expand to one entry per species while
the vector fields carry three components. Recomputing that in the collection
writer would duplicate the naming in `vtk_field_entries` and drift from it.
Resolve it by having `save_hdf5` hand back the `(name, ncomponents)`
descriptors it wrote and caching them on the writer at the first dump. The
descriptors are identical across frames, since `fields`, `stride` and `slice`
are fixed for the life of a writer.

The extension seam is the same one `save_hdf5` uses: a core stub
`_write_xdmf_collection!` that routes through `_hdf5_required`, with the real
method defined in `ext/CompactLESHDF5Ext.jl`.

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
ordering-independent of the patch refactor — the imposition is pointwise and
mask-driven — so it can be implemented on today's arrays and ported
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
  failsafe of model-debt item 3, which recovers from states the scheme has
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

Wall-modeled LES is deferred rather than excluded; see model-debt item 8.

## Suggested ordering

The near-term corrections that used to head this list are done
(`reference/HISTORY.md`), which leaves:

1. **The dilatation-gated β\* experiment** (model debt 1). Days of work
   against the largest documented limitation; the outcome decides whether the
   `cfl ≤ 0.15` guidance and the `C_beta` tables get rewritten.
2. **Filter calibration** (model debt 2), which every other physics number is
   conditional on. Requires cluster time; run it alongside item 3's Pyranda
   comparison, which needs the same machines and cases.
3. **External validation** (model debt 4), promoted early because it changes
   how much the remaining calibration work can be trusted.
4. **Phase 2.1, the implicit diffusion solver** — the principal architectural
   gap, which also addresses the polar CFL restriction.
5. **AMR/GPU Stage 1** (transfer operators) any time — it is small and
   independent. **Stage 2** (the patch refactor) waits until the model debts
   above settle, because it touches every file and rebasing physics changes
   across it is the expensive order. Subsequent stages follow
   `reference/AMR_GPU.md`.

Two items sit outside the ordering because they are independent of everything
above and pullable on demand: laser ray tracing (Phase 2.7), if a specific
experiment needs it before the rest of the HED stack exists, and immersed
boundaries (`reference/IMMERSED.md`) Stage 1, when a target problem first
needs non-coordinate-surface geometry — its main sequencing constraint is
that the Stage 2 calibration sweep deserves the same measurement discipline
as `bench/artcal.jl`, not that it depends on other work.
