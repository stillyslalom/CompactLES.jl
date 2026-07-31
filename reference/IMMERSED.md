# CompactLES — Immersed boundaries: design

This document is the design for immersed-boundary (IB) geometry: solid bodies
that do not lie on coordinate surfaces, represented on the existing structured
grid rather than by a boundary-fitted or unstructured mesh. It is written to
be implemented stage by stage. Prerequisite reading: `DESIGN.md` (operators,
folds, the RHS walkthrough); `reference/AMR_GPU.md` shares the constraint
analysis style and one open interaction. `ROADMAP.md` holds sequencing.

Motivating use cases, in mission order: obstacles and machined features in
shock tubes (wedges, cylinders, splitter plates, directly relevant to
Richtmyer–Meshkov work, abbreviated RM throughout), mounting hardware and fill
tubes in converging geometry, and general
non-coordinate-surface targets that would otherwise force a different code.

## Contents

1. [Method selection under the compact-scheme constraint](#method-selection-under-the-compact-scheme-constraint)
2. [In-family precedent](#in-family-precedent)
3. [Design](#design)
4. [Stage 1 — geometry and blend imposition](#stage-1--geometry-and-blend-imposition)
5. [Stage 2 — validation battery and calibration](#stage-2--validation-battery-and-calibration)
6. [Stage 3 — normal extrapolation refinements](#stage-3--normal-extrapolation-refinements)
7. [Stage 4 — moving bodies](#stage-4--moving-bodies)
8. [Interplay with AMR, GPU, folds, and metrics](#interplay-with-amr-gpu-folds-and-metrics)
9. [Risks and open questions](#risks-and-open-questions)
10. [References](#references)

## Method selection under the compact-scheme constraint

The IB literature (Mittal & Iaccarino 2005 is the survey) divides into sharp
methods — cut cells and ghost cells, which modify the discretization at the
interface — and diffuse methods, which keep the discretization uniform and
represent the body through forcing or state imposition over a smeared
interface region.

**Sharp methods are structurally excluded here**, for the same reason oct-tree
adaptive mesh refinement (AMR) is (`reference/AMR_GPU.md`, constraint 1). A
compact operator couples an entire grid line through a banded LHS factorized
once at plan time; a cut cell removes unknowns mid-line and a ghost-cell
constraint imposes values
mid-line, either of which breaks the uniform banded structure and the
distributed spike solve, per geometry, per line. The field must therefore
exist and remain smooth *through* the solid, with the boundary condition
entering as forcing or as post-stage state imposition — the diffuse family.

Within that family, three formulations are relevant:

- **Brinkman volume penalization** (Angot, Bruneau & Fabrie 1999; compressible
  form Liu & Vasilyev 2007): add −(χ/η)(Q − Q_target) to the RHS, with Q the
  conserved state, Q_target the state the body imposes, χ the solid indicator,
  and η a small relaxation time. Penetration error scales as
  √η; the term is stiff for small η under explicit integration.
- **Characteristic/state-reset methods** (the ghost-fluid lineage, Fedkiw et
  al. 1999; extrapolation operators from Aslam 2004): after each step, rebuild
  the solid-region state from the fluid side by extrapolation along interface
  normals and impose the wall velocity. No stiffness and no η, at the cost of
  an extrapolation operator.
- **Penalization with shocks** (Boiron, Chiavassa & Donat 2009; spectral-family
  precedent Kevlahan & Ghidaglia 2001) demonstrates that both work at high
  Mach number with smeared interfaces of a few cells.

The design below unifies the first two: a graded state *blend* applied after
each RK stage,

    Q ← (1 − w) Q + w Q_target,     w = χ_s · (1 − e^{−Δt/η}),

where χ_s is the smeared solid fraction of the cell (`chi_solid` below, in
[0, 1]), Δt the step just taken, and w the resulting blend weight. This is the
exact integral of the Brinkman relaxation (unconditionally
stable at any η, no timestep coupling) and degenerates to the hard reset at
η → 0 (w = χ_s). One mechanism spans soft penalization to Pyranda-style
imposition, with η a measurable knob rather than a method fork. What
distinguishes the stages below is not the mechanism but how `Q_target` is
built: algebraic targets first, normal-extrapolated targets later.

This is a geometric-flexibility feature, not an accuracy one. Interface
accuracy is first order in the smearing half-width δ (the `delta` field, in
cells; distinct from the δ⁴ sensor difference used by the artificial
properties) regardless of the tenth-order interior, the same honesty that
applies to the closure-order drop at fitted walls. A calculation whose answer
depends on boundary-layer resolution at the body does not belong on an immersed
boundary in this code.

## In-family precedent

Pyranda ships a level-set immersed-boundary method (`pyranda/pyrandaIBM.py`):
the body is a signed distance function, the wall condition is imposed by
resetting the velocity in the solid region (with a smeared transition), and
scalars are extrapolated into the solid along level-set normals; a wall-model
variant exists. This is
the state-reset end of the design above, in the same numerics family, which
is the strongest available evidence the approach coexists with compact
operators, the filter, and artificial properties.

As with the AMR archaeology, the account above is from memory of the source
and must be verified against `pyrandaIBM.py` before Stage 3 fixes the
extrapolation details — primary-source reading paid for itself once already
(`reference/HISTORY.md`, adaptivity groundwork). Record findings here.

## Design

### Geometry specification

A body is a user-supplied pointwise function of physical coordinates,
matching the `ic` pattern — pure, evaluated through `xcoord`, never seeing
ranks or halos:

```julia
body = ImmersedBody(
    phi   = (x1, x2, x3) -> ...,   # signed distance-like: > 0 in fluid
    bc    = NoSlipIB(),            # SlipIB() | NoSlipIB(; Twall=NaN) — NaN adiabatic
    delta = 2.5,                   # interface smearing half-width, in cells
    eta   = 0.0,                   # relaxation time; 0 → hard imposition
)
Problem(..., bodies = (body,))     # tuple, like sources; empty default
```

At `setup`, per body, filled once over the padded arrays (static geometry):

- `phi_body` — φ sampled pointwise; no halo exchange needed (analytic, like
  the geometry arrays in `metric.jl`).
- `grad_phi` — ∇φ by the compact derivative at setup, normalized to `n_hat`.
  Its magnitude `|∇φ|` also normalizes the smearing so that φ need only be
  *distance-like* near the interface rather than an exact signed distance
  function:
  `chi_solid = ½(1 − tanh(φ / (delta · h_phys · |∇φ|)))`, with `h_phys` the
  local mean physical spacing (metric- and stretch-aware via `inv_h`).
- `chi_solid` — the smeared solid fraction above. Multiple bodies combine by
  `max`.
- A deep-solid anchor state: the conserved IC snapshot where
  `chi_solid > 0.99`, stored sparsely (index list + values), toward which the
  deep interior relaxes. This prevents slow drift of the fictitious solid
  state without costing a full extra state array.

Bodies are first-class on `Problem`/`Solver` rather than entries in
`sources`: the mask arrays require setup-time construction (grid-dependent,
which a user-built source tuple predates), and the imposition runs outside
the RHS (below), which a source cannot do.

### Targets per boundary-condition type

Built pointwise from the current state; velocities are physical components in
the local orthonormal basis, consistent with everything else:

- `SlipIB` — remove the normal velocity: `u_t = u − (u·n̂)n̂`, kinetic energy
  reduced accordingly (the `SlipWallBC` pattern); thermal state and
  composition untouched.
- `NoSlipIB` adiabatic (`Twall = NaN`) — `u_t = u_body` (zero for static
  bodies), kinetic energy removed; thermal state untouched. The residual
  spurious heat flux into the body is O(δ) and is what Stage 3 improves.
- `NoSlipIB` isothermal — additionally relax ρe toward
  `wall_internal_energy(eos, Q, I, n_species, Twall)`, reusing the existing
  per-EOS hook from `boundary.jl` unchanged.
- Species: never forced; Σ J_k = 0 machinery is untouched and the interface
  is impermeable through the velocity condition.

### Where the imposition runs

In `step!`, immediately after `apply_bcs!`, every RK stage — it *is* a
boundary condition, and the RHS then differentiates a state that already
honors the body, exactly as it does for wall planes. It is pointwise and
χ-masked, so the cost is negligible and there are no collectives (no
early-return trap). It is not an RHS source: putting −(χ/η)(q − q_target) in
`dQ` would couple η to the RK45 stability limit and poison `compute_dt`,
which the exact-blend form avoids entirely. `compute_dt` needs no
modification: the imposed state is quiescent, its sound speed is physical,
and the blend is unconditionally stable.

### Sensors, filter, and artificial properties

The compact filter and the Cook sensors run unmodified through the body. The
smeared interface is smooth over ~2δ cells, so the δ⁴ sensors see a resolved
feature and respond moderately — likely beneficial, since they damp exactly
the interface-generated noise the penalization literature worries about.
Whether artificial properties should be scaled down by `(1 − chi_solid)` deep
inside the body is a measurement question for Stage 2, not a design
commitment; the diffusive-rate contribution of interface β\* to `compute_dt`
is part of that same measurement.

### Diagnostics

The imposition is bookkept, which makes force and heat-flux extraction exact
with respect to what the scheme actually did: per step,

    F_body   = −Σ w (Q_target − Q)[momentum] · cell_measure / Δt
    q̇_body   = −Σ w (Q_target − Q)[energy]   · cell_measure / Δt

accumulated in the imposition loop and reduced on demand (`body_force`,
`body_heat_flux` — MPI-reduced like the existing diagnostics). The same sums
are the conservation-defect report: diffuse IB does not conserve inside the
body, and the defect should be printed by the validation cases rather than
hidden. `save_vtk`/`save_hdf5` gain `:chi_solid` and `:phi_body` derived
fields for inspection.

## Stage 1 — geometry and blend imposition

**Deliverable.** `src/immersed.jl`: `ImmersedBody`, `SlipIB`, `NoSlipIB`, the
setup-time mask/normal/anchor construction, the per-stage imposition, the
force/heat diagnostics, and the two output fields. Static geometry only.
Frontend: `Problem(..., bodies=...)` threaded through `setup` and `Solver`
(one new type-parameterized field, following the `sources` tuple pattern so
inference survives — verify with `bench/jetcheck.jl` deltas as usual).

**Gates.**

1. `bodies = ()` and a body with φ > 0 everywhere are both bit-identical to
   current `main` on the convergence guards (the no-op gate).
2. 1-D Sod against an immersed planar wall, compared against the same
   problem solved with a fitted `SlipWallBC` at the same location: reflected
   shock position and post-reflection state, error measured versus δ and
   versus h (expect first order in the interface region).
3. The serial suite and the MPI suite (counts in `CLAUDE.md`) pass with a body
   straddling rank boundaries (the mask is pointwise, so this should be trivially true;
   the test exists to keep it true).

## Stage 2 — validation battery and calibration

Measure before refining; this stage produces the numbers Stage 3 decisions
need. Cases small enough to respect the run-cost discipline, added to
`test/cases.jl` shape-compatibly so a future guard can consume them:

- **Rotated immersed plate** versus the fitted-grid solution — the
  angle-independence check that is the entire point of IB.
- **Supersonic cylinder**: shock standoff distance versus the Billig
  correlation; drag from `body_force`.
- **Low-Mach cylinder shedding** at Re = 100–200: Strouhal number against
  the established range (≈ 0.16–0.20) — exercises force diagnostics and
  long-time interface stability.
- **Shock–cylinder diffraction** (Bryson & Gross): shock trajectory
  comparison; the RM-relevant case and the sternest interface-noise test.
- **Parameter sweep** over δ (1.5–4 cells) and η (0, dt, 10 dt) in the
  `bench/artcal.jl` style, including the artificial-property masking
  question and the conservation-defect magnitude. Results land in a new
  section of this file, with recommended defaults, in the CALIBRATION.md
  format.

## Stage 3 — normal extrapolation refinements

Driven by Stage 2 measurements; skip whatever they do not justify.

- **Adiabatic condition done properly**: build the thermal part of
  `Q_target` by extrapolating T (or ρe) from the fluid side along −n̂,
  Aslam-style constant extrapolation — iterate
  `∂s/∂τ = −H(−φ) (n̂·∇s)`, with s the scalar being carried into the solid,
  τ a pseudo-time, H the Heaviside step and n̂ the interface normal, for ~5–10
  pseudo-steps with local first-order upwind differences (not the compact
  operators: locality and monotonicity are wanted here, spectral resolution is
  not). One halo exchange per sweep;
  cost is per-step but pointwise-cheap and confined to a band around the
  body.
- **Tangential-velocity fidelity for `SlipIB`** through the same
  extrapolation of u_t, if Stage 2 shows the algebraic target dragging the
  outer flow.
- Verify the Pyranda `pyrandaIBM.py` account (see
  [In-family precedent](#in-family-precedent)) before fixing conventions,
  and record the findings here.
- Wall models (Pyranda's `ibmWM` shows the shape) remain out of scope, per
  the wall-treatment deferral in `ROADMAP.md`; the extrapolation machinery
  built here is where one would attach.

## Stage 4 — moving bodies

- `phi = (x1, x2, x3, t)` and `u_body(x, t)` variants; masks, normals, and
  anchors rebuilt per step from the analytic function (pointwise, cheap; no
  re-planning — the operators never see the body, which is the payoff of the
  diffuse formulation).
- Fresh cells (solid → fluid) are handled by construction: the uncovered
  state was being relaxed toward a physical target throughout, so no special
  reinitialization is required — this claim is exactly what the gate tests.
- **Gates**: an impulsively started and an oscillating immersed piston versus
  the existing fitted `DirichletBC` piston (`examples/piston_driver.jl`)
  for radiated wave amplitude and phase; a moving cylinder versus the fixed
  cylinder in the moving frame.
- Body dynamics (fluid–structure coupling) are out of scope; `u_body` is
  prescribed.

## Interplay with AMR, GPU, folds, and metrics

- **Ordering-independent of the patch refactor.** The imposition is
  pointwise and mask-driven, so it works identically on today's monolithic
  arrays and on patches; implementing before AMR Stage 2 costs only the
  mechanical port of one loop and two setup arrays. If AMR lands first, χ
  and n̂ are built per patch and per level from the same analytic φ.
- **AMR synergy**: the interface is precisely where the sensor tagger will
  refine, and δ in cells means the physical smearing shrinks under
  refinement — the body sharpens where it is refined, for free. A body
  crossing a coarse–fine boundary needs the Stage 2 sweep repeated at the
  interface; note it in the AMR plan's Stage 3 gates when both exist.
- **GPU**: the imposition and mask construction are pointwise — trivial
  KernelAbstractions kernels under G1; the Stage 3 extrapolation sweeps are
  the only new stencil kernels and they are local.
- **Folds and metrics**: φ is a function of physical coordinates, so bodies
  compose with cylindrical/spherical metrics and stretch maps with no new
  machinery. Antipodal pairs both see the body automatically for the same
  reason. Bodies overlapping a fold's singular region are untested; forbid
  at setup initially, matching the AMR restriction, and lift on evidence.

## Risks and open questions

1. **Spurious acoustics from the interface under strong shocks** — the known
   failure mode of diffuse IB. Mitigations, in order: δ ≥ 2 cells (echoing
   the ≥ 3-cell resolution lesson from the spherical origin), the sensors
   already damping interface noise, and η > 0 softening the imposition. The
   shock–cylinder case is the designated probe.
2. **Thin geometry.** A body feature under ~3 cells thick cannot hold a χ
   plateau and leaks. No general setup-time detection is possible from a
   black-box φ; document the requirement beside the origin-fold resolution
   limit, and let the rotated-plate case quantify the minimum.
3. **First-order interface accuracy** is inherent to the diffuse family; the
   sharp alternatives are structurally excluded. If a future problem demands
   better, the summation-by-parts / simultaneous-approximation-term (SBP–SAT)
   multiblock route (`reference/AMR_GPU.md`) with a body-fitted block is the
   escape hatch, not a sharper IB.
4. **Filter-across-interface dissipation**: the compact filter smooths fluid
   state into the slaved solid state each application, an energy sink the
   force bookkeeping does not see. Stage 2's conservation-defect measurement
   covers it; if significant, the fix is including the filter's χ-region
   delta in the same bookkeeping, not exempting the region from filtering.
5. **The Pyranda account is from memory** until the Stage 3 verification
   task reads the source.
6. **`compute_dt` blind spot**: velocity inside the body is imposed ≈ 0
   *after* stages, but mid-stage the RHS can transiently accelerate solid
   cells; the CFL sweep sees the post-imposition state only. Expected
   harmless (the blend re-zeroes each stage); the long-shedding case is the
   detector if it is not.

## References

- Mittal & Iaccarino (2005), Annu. Rev. Fluid Mech. — IB taxonomy and survey.
- Angot, Bruneau & Fabrie (1999), Numer. Math. — Brinkman penalization and
  its error analysis.
- Liu & Vasilyev (2007), J. Comput. Phys. — Brinkman penalization for
  compressible flow.
- Boiron, Chiavassa & Donat (2009) — penalization with shocks at high Mach.
- Kevlahan & Ghidaglia (2001), Eur. J. Mech. B — penalization in a
  spectral-family discretization.
- Fedkiw, Aslam, Merriman & Osher (1999), J. Comput. Phys. — ghost-fluid
  state construction.
- Aslam (2004), J. Comput. Phys. — PDE-based constant/linear extrapolation.
- `pyranda/pyrandaIBM.py` — in-family level-set immersed-boundary method
  (verify per Stage 3).
