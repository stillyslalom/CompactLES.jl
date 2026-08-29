# CompactLES — Azimuthal mode truncation: implementation plan

This document is the implementation plan for azimuthal mode truncation near
coordinate singularities, the mitigation for the resolved-angle CFL squeeze,
written to be executed stage by stage without re-deriving the constraints.
Prerequisite reading: `DESIGN.md` (compact operators, folds, the distributed
solve), the "Timestep and CFL near coordinate singularities" section of
`README.md`, and `CALIBRATION.md` for the validation-guard conventions the
gates below inherit.

## Contents

1. [The problem, quantified](#the-problem-quantified)
2. [Why truncation and not the alternatives](#why-truncation-and-not-the-alternatives)
3. [Design decisions](#design-decisions)
4. [Stage 1 — serial cylindrical](#stage-1--serial-cylindrical)
5. [Stage 2 — decomposed θ](#stage-2--decomposed-θ)
6. [Stage 3 — measurement, calibration, defaults](#stage-3--measurement-calibration-defaults)
7. [Stage 4 — spherical φ](#stage-4--spherical-φ)
8. [Risks and open questions](#risks-and-open-questions)

## The problem, quantified

`max_rate` (timestep.jl) charges the acoustic rate in an angular dimension at
`(|u| + c) / (r·Δθ)`. On a half-offset polar grid the first node sits at
r₁ = Δr/2, where the azimuthal spacing is tighter than the radial one by a
factor of roughly N_θ/π — about 20× at N_θ = 64. The diffusive rate scales
with the square of the inverse spacing, so its penalty is that factor squared,
and the artificial bulk viscosity β\* peaks at the axis exactly when a
converging shock arrives, so the squared penalty is realized in precisely the
target problems. In spherical coordinates the φ spacing r·sinθ·Δφ goes to
zero at both the origin and the poles simultaneously.

The physics does not require this. Analytic regularity at the axis forces
azimuthal mode m to vanish like r^m, so at r₁ every mode above m ≈ 1–2 has
no resolvable content: the grid near the axis carries N_θ/2 modes of which
only about πr/Δr can exist in a smooth solution. The timestep is being paid
almost entirely for modes the continuous problem excludes.

Two consequences fix the shape of the fix:

1. **The unresolvable modes must be removed every step, not merely ignored.**
   Stability is a property of the scheme, not of the state. Discounting the
   modes in `compute_dt` without enforcing their absence lets them grow, and
   the failure mode is the `dt`-collapse grind described under Traps in
   `CLAUDE.md`, not a crash.
2. **The rate cap in `max_rate` and the enforcement must derive from the same
   table**, so that what is charged and what is removed cannot drift apart.

Precedent: this is the standard remedy in solvers of exactly this class.
HiPSTAR (Sandberg's compact-FD/spectral turbomachinery LES) runs production
with radius-dependent azimuthal mode reduction; latitude–longitude
atmospheric dynamical cores have used polar Fourier filters for decades;
Mohseni & Colonius (2000) is the reference pole treatment for the fold-style
grids this code already uses.

## Why truncation and not the alternatives

Evaluated and set aside, with the disqualifying constraint for each, so they
are not re-derived:

- **Radius-dependent strength of the existing C8 filter.** Nearly free
  (`blend_interior!` already blends with a scalar weight), but the C8
  passband is deliberately narrow: full strength removes only near-Nyquist
  content and cannot reach the mid-band modes (m ~ N_θ/4) that set `dt`.
  Buys perhaps 1.5–2×, not 20×.
- **Implicit/IMEX θ.** Needs block-coupled periodic implicit solves (the
  `line_solver` is scalar banded) plus an IMEX integrator replacing the
  low-storage RK45, and adds dispersion error where waves cross the implicit
  region.
- **Standalone local time stepping.** Conflicts with every collective in the
  step loop (`max_rate`, the filter, the folds). AMR subcycling
  (`AMR_GPU.md` Stage 4) is the sanctioned form of the idea.
- **Reduced-θ inner patch via patch AMR.** Legitimate long-term home: the
  fold region must stay uniform (`AMR_GPU.md` constraint 4), so the innermost
  patch is the coarse-θ one with 3× θ-refinement rings outward — the classic
  reduced polar grid. Requires θ-only anisotropic refinement ratios to be
  admitted into AMR Stages 2–3, which should be decided when those stages are
  designed. Not a near-term fix, and truncation remains useful on the coarse
  patch afterwards.
- **Cartesian core / overset patch.** Breaks the no-line-cut constraint and
  conservation at the seam; effectively a second solver.
- **FARGO-style orbital advection.** Relieves only |u_θ|-dominated CFL
  (rotation-dominated disks). In shock-driven convergence c dominates.
- **Radial stretching away from the axis.** `Stretch` cannot carry a fold
  (`setup` rejects it), so this is available only for annulus domains, which
  do not have the squeeze.

## Design decisions

Each is structural or follows from a repo convention; none is a preference.

1. **Direct low-mode projection, not an FFT.** Per (r, z) ring the truncated
   field is reconstructed from the Fourier coefficients a_m, m ≤ m_max,
   computed against explicit cos/sin tables: O(N_θ·m_max) per ring per
   component. m_max is small exactly where the truncation is active, so the
   total across all active rings of a z-plane is on the order of N_θ³/8π
   operations, negligible against one compact line-solve pass over the same
   plane. This avoids adding an FFT dependency, is an exact projection, and
   is idempotent. Basis tables are built at setup and live on the `Solver`
   as concrete-typed fields (the audit gate tolerates no new dispatch sites
   in a per-step loop).
2. **The mode limit.** Per radial index,

       mode_limit(r) = max(1, floor(π·r / (κ·Δr)))

   with Δr the physical radial spacing (`solver.h[1]`; uniform, since a
   stretched dimension cannot carry a fold) and κ ≥ 1 the safety margin
   exposed as the enabling keyword. This targets an effective azimuthal
   spacing of κ·Δr. The floor of 1 is load-bearing: the m = 0 coefficient is
   untouched, so ring sums of every conserved component — and hence mass and
   energy — are conserved exactly (uniform θ quadrature weights, metric
   factors independent of θ), and the Cartesian momentum components live in
   the m = ±1 modes of the physical-component momenta, so mode_limit ≥ 1
   preserves them exactly. A discretely sampled m = 1 field is projected
   without aliasing error for any N_θ ≥ 4, so uniform freestream (u_r =
   U·cosθ, u_θ = −U·sinθ) is preserved to roundoff.
3. **Active set.** Rings with mode_limit < N_θ/2, i.e. r ≲ κ·N_θ·Δr/(2π):
   about ten radial indices at N_θ = 64. Precomputed at setup as an index
   range plus the per-index limits.
4. **Placement: once per step in `run!`**, after `filter_state!` (when it
   fires) and before `apply_positivity_floor!`, writing the interior only.
   Halos are left stale by the same convention the floor uses; `max_rate`
   exchanges before reading anything at the top of the next iteration. Once
   per step rather than per RK stage is standard practice in the precedent
   codes; a mode above the cap grows for at most one step before removal,
   and the margin κ absorbs that (measured in Stage 3, see Risks).
5. **The rate cap.** In `max_rate` (and mirrored in `dt_report`, which
   duplicates the loop), the angular direction's inverse physical spacing is
   capped at `mode_limit(r) / (π·r)` at active rings, in both the acoustic
   term and the squared diffusive sum. The cap and the projection read the
   same table. The cap applies only when the feature is enabled; disabled,
   both loops are bit-identical to today's.
6. **Reproducibility over cheapness in the MPI path.** When θ is decomposed,
   the ring is gathered in global θ order (`Allgatherv` over
   `decomp.sub[2]`) and the projection evaluated redundantly on every rank
   of that sub-communicator, so the summation order is fixed and the
   projection itself is decomposition-independent bit-for-bit. The run as a
   whole is not: the collective reductions elsewhere are order-dependent
   `Allreduce(+)` and reproduce serially to round-off, of order 1e-14
   relative. The cheaper alternative (Allreduce of partial coefficient sums,
   2·mode_limit+1 values per ring) was rejected because it would put the
   projection in that second category as well, for a saving of a few values
   per ring.
7. **No fold interaction.** Rings sit at r > 0 and never touch the fold; the
   fold pairs radial lines. Fourier modes have definite parity under
   θ → θ + π (sign (−1)^m), so the projection commutes with the antipodal
   even/odd decomposition in `folds.jl`. Collapsed θ (`active[2] == false`)
   makes the feature a structural no-op, guarded explicitly.
8. **The keyword.** `Numerics(polar_truncation = 0.0)`: zero disables
   (default, keeping the whole gate bit-identical), a positive value is κ.
   Setup validates it is only meaningful for a cylindrical or spherical
   metric with the relevant angular dimension resolved, and errors otherwise,
   following `validate_bc`'s fail-at-setup convention. Proposed names, per
   the vocabulary in `CLAUDE.md`: `polar_truncation` (the keyword),
   `mode_limit` (the per-ring integer table), `truncate_modes!` (the per-step
   entry point).

## Stage 1 — serial cylindrical

Deliverables:

- Setup: the `mode_limit` table, the active index range, cos/sin basis
  tables, and validation of the keyword against metric and `active[2]`.
- `truncate_modes!(solver, Q)`: projection over active rings, all conserved
  components, interior only. Serial path (θ on one rank) only; `setup`
  rejects `polar_truncation > 0` with `dims[2] > 1` until Stage 2.
- The rate cap in `max_rate` and `dt_report`.
- The `run!` call site per design decision 4.

New serial testset (test/runtests.jl):

- Uniform freestream on a resolved-θ cylindrical disk is preserved to
  roundoff through truncation (the m = 1 case).
- Rigid rotation (m = 0 in physical components) is exactly untouched.
- A seeded high-m perturbation at an active inner ring is removed; the same
  perturbation at an inactive outer ring is untouched; applying
  `truncate_modes!` twice equals applying it once.
- Ring sums of mass, energy, and Cartesian momentum are conserved to
  roundoff across a truncation.

Gate: the full serial gate is bit-identical with the feature off, including
the convergence-guard error magnitudes to the digit; the new testset passes;
`bench/jetcheck.jl` and `bench/audit.jl` deltas show no new dispatch sites
and no new allocation in the step path.

## Stage 2 — decomposed θ

Deliverables:

- The `Allgatherv` ring path over `decomp.sub[2]` per design decision 6,
  with buffers sized at setup. Every rank of the θ sub-communicator calls
  the gather for every active ring; the ranks of one θ sub-communicator
  share the same radial block, so they agree on the active set and there is
  no cross-rank early-return hazard of the kind `nscbc.jl` guards against.
  Ranks whose radial block contains no active ring never enter the code at
  all, which is safe because the θ sub-communicator never spans radial
  blocks.
- Lift the Stage 1 `dims[2]` restriction.

New MPI test (test/mpi_tests.jl): the same seeded field truncated under
serial and under θ-splitting process grids matches bit-for-bit.

Gate: 90/90 plus the new tests at np = 2, 4, 8.

## Stage 3 — measurement, calibration, defaults

Deliverables:

- `bench/polarcfl.jl`, taking its settings from `ARGS` via `script_args`:
  a resolved-θ cylindrical converging shock (the perturbed Noh geometry from
  `test/cases.jl` extended to resolved θ, added there so guards and
  calibration cannot drift apart). Measured: achieved `dt` against the
  collapsed-θ baseline, the θ-limited fraction of the run via periodic
  `dt_report`, wall-clock cost of `truncate_modes!` per step, and stability
  as a function of κ ∈ {1, 1.5, 2}.
- The validation battery with truncation on: `test/validation.jl` guards at
  the four-digit comparison level, and the resolved-θ 3.71 convergence guard
  with the order preserved (error magnitudes at the innermost rings may
  move; record them).
- A chosen default κ, documented with the measurements in `CALIBRATION.md`.
- README: replace "None is implemented here" in the CFL section with the
  implemented status and the measured factor.

Sweep hygiene: κ values that fail will grind, not crash — pass a low `nmax`
and rely on the `StepControl` floors, per Traps. Stop at sufficient evidence;
each point is a full shock run.

Gate: a measured wall-clock speedup at N_θ = 64 recorded in
`CALIBRATION.md`; validation and convergence guards within their comparison
levels; README and `CALIBRATION.md` updated in the same change.

## Stage 4 — spherical φ

The φ dimension is periodic and its rings (fixed r, θ) never touch a fold,
so the machinery transfers directly with a two-dimensional limit table:

    mode_limit(r, θ) = max(1, floor(π·r·sinθ / (κ·Δ_ref))),
    Δ_ref = min(Δr, r·Δθ)

which relieves the pole squeeze (matching the local θ spacing) and the
origin squeeze in φ (matching the radial spacing) simultaneously. The rate
cap enters the d = 3 terms of `max_rate` keyed on the same table.

Tests mirror Stage 1 on a spherical shell with resolved poles and on the
full ball; the origin+poles combination is the least-exercised part of the
code (README), so the freestream and conservation checks carry the weight.

Out of scope, recorded for a later generation: the residual θ squeeze near
the spherical origin (spacing r·Δθ as r → 0). θ lines are not periodic
per-line; they continue through the poles into their antipodal partners, so
a great-circle truncation would reuse the `PairSpec` pairing to assemble the
full closed line. Nothing in this plan forecloses it.

Gate: spherical freestream preservation in every configuration of
`AxisBC`/`OriginBC`/`PoleBC` that resolves φ; the spherical convergence
guards; the serial and MPI gates.

## Risks and open questions

- **One step of growth between truncations.** A mode above the cap amplifies
  through five RK stages before removal. The amplification per step is
  finite and the mode is then zeroed exactly, so the loop is bounded if
  nonlinear production feeds from resolved modes, which is the standard
  argument in the atmospheric literature — but the margin has not been
  measured on this scheme. Stage 3's κ sweep is the measurement; if κ = 1 is
  unstable and κ = 2 is not, the default moves and the README states the
  achieved factor honestly.
- **Aliasing repopulation.** The projection is applied after the nonlinear
  step, so truncated modes are repopulated by aliasing each step and removed
  each step. Expected benign at these amplitudes; watch the inner-ring
  energy in Stage 3 rather than assuming.
- **Shock structure crossing the active region.** Near the axis a converging
  front is nearly axisymmetric by geometry, and regularity bounds its
  angular structure by the same r^m argument, so the cap discards nothing a
  smooth front carries. The perturbed-Noh case in Stage 3 is the check that
  perturbation growth through convergence is not artificially clipped:
  compare perturbation amplitudes at focus against a collapsed-θ
  linear-theory baseline.
- **What this does not fix.** The strong-shock `cfl ≤ 0.15` startup
  restriction (`CALIBRATION.md`) is a robustness problem at the symmetry
  cell and is unaffected; truncation removes the geometric penalty
  multiplying it, not the restriction itself. The two compose: a resolved-θ
  strong-shock run pays 0.15 times the geometric factor today, and 0.15
  alone afterwards.
- **Interaction with `filter_cfl`.** The truncation is an exact projection
  and idempotent, so it has no per-application-versus-per-unit-time ambiguity
  and needs no relaxation weight. No coupling is expected; `bench/filterrate.jl`
  is the instrument if one appears.
