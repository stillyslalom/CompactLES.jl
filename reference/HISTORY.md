# CompactLES — Completed work

This file records completed roadmap phases: what was done, when, and what was
measured, so `ROADMAP.md` carries only open work. It is scoped to roadmap-level
phases; the git history records individual changes. Durable numerical findings
remain in `reference/CALIBRATION.md` and `reference/CLUSTER.md`; this file
points at them and does not restate them.

## Contents

1. [Phase 0 — extensibility hooks (July 2026)](#phase-0--extensibility-hooks-july-2026)
2. [Phase 1 — shock-tube validation and EOS generalization (July 2026)](#phase-1--shock-tube-validation-and-eos-generalization-july-2026)
3. [Parallel HDF5/XDMF output (July 2026)](#parallel-hdf5xdmf-output-july-2026)
4. [Adaptivity groundwork (July 2026)](#adaptivity-groundwork-july-2026)
5. [Near-term corrections (July 2026)](#near-term-corrections-july-2026)
6. [The compression-keyed β\* sensors (July 2026)](#compression-keyed-beta-sensors-july-2026)
7. [The reference-implementation pass (August 2026)](#the-reference-implementation-pass-august-2026)
8. [The sensor fields (August 2026)](#the-sensor-fields-august-2026)
9. [The C_beta refit (August 2026)](#the-c_beta-refit-august-2026)
10. [The origin cell (August 2026)](#the-origin-cell-august-2026)
11. [Filter dt-consistency (August 2026)](#filter-dt-consistency-august-2026)
12. [The positivity failsafe (August 2026)](#the-positivity-failsafe-august-2026)
13. [AMR/GPU Stage 1 — level-transfer operators (August 2026)](#amrgpu-stage-1--level-transfer-operators-august-2026)
14. [AMR/GPU Stage 2 — patch abstraction and storage generalization (August 2026)](#amrgpu-stage-2--patch-abstraction-and-storage-generalization-august-2026)
15. [AMR/GPU Stage 3 — static two-level refinement (August 2026)](#amrgpu-stage-3--static-two-level-refinement-august-2026)
16. [AMR/GPU Stage 4 — subcycling, tagging, and regridding (August 2026)](#amrgpu-stage-4--subcycling-tagging-and-regridding-august-2026)
17. [GPU Stage G1 — pointwise kernels via KernelAbstractions (August 2026)](#gpu-stage-g1--pointwise-kernels-via-kernelabstractions-august-2026)

## Phase 0 — extensibility hooks (July 2026)

Completed as a behavior-preserving refactor. All convergence errors remained
bit-identical; serial and 2/4/8-rank gates passed throughout. JET reports for
`compute_rhs!` dropped from 16 to 2, and axis-fold RHS allocation dropped from
8,336 B to 784 B per call.

1. **A source-term interface.** `Solver.sources` is an inferable tuple applied
   at RK stage time through `add_sources!`; tuple recursion compiles away, so
   inference and the allocation discipline survive. `ConstantBodyForce` is the
   minimal implementation exercising the interface.
2. **An `EquationSet` type owning the conserved layout.** `NavierStokes1T`
   carries `n_cons`, `i_mom`, `i_energy`, component names, and the fold parity
   tables. The former hard-coded `n_species + 4` sites are gone; a 3T model now
   costs one equation set, with no rediscovery of the layout throughout the
   solver.
3. **Concrete parameterization of `Solver`** for the fields that were abstract.
   `FoldSpec` parameterization was left open, motivated by the JET dispatch
   sites it would close. Those sites turned out to be one reduced-stage
   `ldiv!` on the `Union{RedLU, Nothing}` field, closed in August 2026 by
   narrowing in both `_reduced_solve!` methods (`jetcheck.jl` now reports
   zero at every probed entry point), so the parameterization is no longer
   tracked as a work item.
4. **Caller-owned stage storage.** `Workspace` holds the two low-storage RK
   arrays, so splitting, subcycling, and future IMEX schemes can own their
   stage storage; `run!` no longer allocates it internally.

## Phase 1 — shock-tube validation and EOS generalization (July 2026)

**Validation battery.** `test/validation.jl` runs Lax, Shu–Osher,
Woodward–Colella, Sedov–Taylor, and Noh, the last in all three geometries, in
about 25 seconds. Lax, Sedov, and Noh compare against closed-form solutions and
test absolute accuracy; Shu–Osher and Woodward–Colella compare against stored
4×-resolution profiles from this code and detect regressions. Cases live in
`test/cases.jl`, shared with the calibration sweep so the two cannot drift.

Construction of the battery identified two operating limits, documented with
their measurements in `reference/CALIBRATION.md`: converging strong shocks
require `cfl ≤ 0.15`, and the spherical-origin fold does not accept initial data
resolved over fewer than about three cells, nor the singular t = 0 start of
spherical Noh. The cylindrical axis accepts both; the difference between folds
remains unexplained.

The CFL limit was attributed at the time to a dispersive density undershoot
outrunning the artificial viscosity, with timestep lag tested and excluded. The
undershoot attribution was itself withdrawn later on the `bench/nohprobe.jl`
measurements recorded under [the compression-keyed
sensors](#compression-keyed-beta-sensors-july-2026):
the failure begins at the symmetry plane, not ahead of the front.

**Artificial-property calibration.** `bench/artcal.jl` swept each constant over
the battery; results and recommendations are in `reference/CALIBRATION.md`.
The existing defaults were retained, with one substantive correction to the CFL
guidance and three findings: the `C_beta` upper bound is a stability bound,
not an accuracy bound; `C_kappa = 0` does not support a converging strong
shock; and `C_D` has weak influence because the compact filter broadens a
passive interface more than D\* does. A separate 128³ Taylor–Green study
constrained `C_mu` for resolved shear (optimum 0.004 ± 0.003, containing the
default) and established that the compact filter, not the Cook properties, is
the primary stabilizer at every resolution measured. That finding motivates the
filter-calibration item now in `ROADMAP.md`.

**Mixing diagnostics.** `src/diagnostics.jl` provides metric-aware, MPI-reduced
volume integrals and plane-averaged profiles, and on top of them mix width,
Youngs' molecular mixing fraction θ, composition PDFs, Favre turbulent kinetic
energy, and resolved dissipation including the artificial contribution. The
quadrature is exact for a constant on a Cartesian grid and second order at a
node-centered curvilinear edge.

**EOS generalization.** Two sites previously assumed an ideal gas outside the
function barrier, NSCBC's LODI algebra (φ = ∂(ρe)/∂p, ∂φ/∂Y_k) and the
artificial-conductivity scale. Both became EOS dispatch points. The full contract
is written down at the top of `physics.jl`. Two models exercise it beyond
`IdealMixture`: `StiffenedGas` (exact perfect-gas limit at p∞ = 0, the natural
precursor to Mie–Grüneisen) and `Nasa9Mixture` (piecewise temperature-dependent
cp with a bounded Newton inversion of e(T)). The NASA CEA thermodynamic and
transport databases are bundled verbatim in `data/` beside their Apache license;
`read_nasa9` parses the fixed-column multi-interval records and derives R from
molar mass. `examples/shock_tube.jl` uses real He and CO₂ cp(T).

The κ\* singularity as T_ion → 0 is exposed through dispatch but remains for
the gas models; it is a numerics decision listed in `ROADMAP.md`. The transport
table remains raw data; connecting it is likewise a roadmap item.

## Parallel HDF5/XDMF output (July 2026)

Delivered for single dumps and checkpoints, as a weak dependency
(`ext/CompactLESHDF5Ext.jl`), keeping the package core dependency-light.

- `save_hdf5` writes a field dump as one shared `.h5` per frame regardless of
  rank count, with an XDMF3 sidecar for ParaView/VisIt. It takes the same
  `fields`, `stride`, and `slice` selections as `save_vtk` and follows the same
  rectilinear/curvilinear rule.
- `save_checkpoint_hdf5` / `load_checkpoint_hdf5!` write the state as one
  global array and restore it onto any rank count and process grid, removing
  the same-decomposition restriction of the per-rank binary checkpoint.
- XDMF3 was chosen over VTKHDF because that container supported neither
  RectilinearGrid nor StructuredGrid as of format version 2.5, so a stretched
  or curvilinear grid could only be expressed there as an unstructured mesh.

Two pieces remain open and are specified in `ROADMAP.md`: routing `FieldWriter`
time series through the HDF5 path, and validating the collective write on a
machine with a parallel libhdf5 built against the run's MPI (only the
serialized token-relay fallback has been exercised).

## Adaptivity groundwork (July 2026)

Research inputs gathered ahead of any AMR implementation; the resulting plan is
`reference/AMR_GPU.md`.

- Miranda's level-transfer operators were read directly from the Fortran
  kernels Pyranda carries (`pyranda/parcop/stencils.f90`): an invertible
  compact filter pair matching a Gaussian of width 3Δx, refinement ratio 3,
  conservation by unit DC gain, with four boundary-closure variants per end.
  The surrounding patch management, subcycling, and tagging are not in the
  public tree and were designed independently in the plan.
- `bench/amr_transfer.jl` reconstructed the operators and measured their
  conditioning: the swapped pair round-trips to 3.4e-15, the coarse band is
  well conditioned, and grid-scale content is amplified by roughly 20× under
  prolongation, the quantitative bound that shapes the interface treatment in
  the plan.
- The SBP–SAT literature for implicit operators was surveyed as the rigorous
  alternative; the references and the three documented considerations are
  retained in `reference/AMR_GPU.md`. An earlier claim that adaptivity required
  moving to a cell-centered grid was withdrawn after that survey: multiblock
  SBP–SAT interfaces are node-coincident, and non-conforming interfaces are
  handled by interpolation operators, not by staggering.

## Near-term corrections (July 2026)

The fourteen items of the July 2026 code review, closed in one pass. Most were
documentation, naming, or dependency hygiene and are recorded only in the git
history. Four changed behavior, and the gate was compared line by line before and
after: convergence orders and error magnitudes bit-identical, the validation
battery reproducing every printed digit, 60/60 at 2, 4, and 8 ranks with identical
measured values, and no new dispatch site.

The additions that outlast the individual fixes:

- **`validate_bc(bc, metric, eos, d, side)`**, a setup-time hook called once per
  face and forwarded to both arms of a `SwitchableBC`. A boundary condition whose
  derivation restricts the geometry declares that restriction here, so it is
  neither rechecked per RHS call nor left in prose. Both NSCBC conditions now
  reject a face whose normal metric scale factor is not one, previously a
  documented but unenforced restriction and therefore a silent-wrong-answer path.
  `unit_scalefactor(metric, d)` in `metric.jl` is the predicate.
- **`interior_index`**, the public inverse of `gidx`, for the conversion between
  the padded indices `boundary_plane` yields and the interior ones `xcoord` takes.
- **`Prim` accepts any two of pressure, density, and temperature.** The conserved
  state is a function of `(rho, T_ion, u, Y)`, so `p` was never needed when the
  other two were given; this costs a constructor check and suits a stratified or
  isothermal initial state.
- **`step!` takes a trailing `prepared` flag.** `run!` now enforces boundary
  conditions at the top of the loop, ahead of the rate measurement, so the step is
  sized from the state it will advance and `max_rate`'s halo exchange and
  primitives pass serve the first RK stage as well. Verified to reproduce the
  unprepared path bit-for-bit. `step!` allocation at 48³ fell from 17,396,240 B to
  17,104,416 B.

**Threading a planar-2-D run.** Every pointwise loop threaded its outermost index,
so an `(nx, ny, 1)` grid divided a loop of one trip and ran serially at any thread
count while still paying the region-entry cost. Those loops now iterate a
flattened `outer_indices(n2, n3)`, which preserves iteration order and divides
over the second dimension when the third is collapsed; `_use_threads` also takes
the trip count and does not spawn for one. Measured at 256×256×1 on the
24-thread development workstation, `compute_rhs!` and `step!` per call:

    threads      before                  after
    1            12.62 ms   65.25 ms     12.76 ms   65.41 ms
    12           15.47 ms   86.73 ms      5.34 ms   24.66 ms
    24           21.23 ms  112.63 ms      5.53 ms   34.81 ms

Requesting threads had previously made such a run monotonically slower, by 1.68×
at 24 threads, where it now gives a 2.4× speedup at 12. Both figures lie well
outside the 10–20% run-to-run spread, and the serial column agrees to about 1%.
The 24-thread column falls behind the 12-thread one through the hybrid
performance/efficiency-core effect documented in `CLAUDE.md`. The
KernelAbstractions conversion in `reference/AMR_GPU.md` supersedes the flattening
with an ndrange over all three dimensions.

<a id="compression-keyed-beta-sensors-july-2026"></a>

## The compression-keyed β\* sensors (July 2026)

The first of the model debts, closed. The two halves of the literature
refinement are separable and behave differently, and that separation is the
result. Full measurements are in `reference/CALIBRATION.md`.

`ArtParams.beta_sensor` now takes three values. `:strain` is the Cook original
and remains the default, bit-identical to before the change. `:gated_strain`
multiplies that sensor by the Ducros-style compression switch
H(−Δ)·Δ²/(Δ² + |ω|² + ε), with Δ = ∇·u the dilatation, ω the vorticity, H the
Heaviside step and ε = 1e-32, for one pointwise pass and no line solves.
`:dilatation` is the full form of Mani, Larsson and Moin (JCP 228, 2009), which
also rebuilds the sensor from ∇·u and costs an additional smoothing pass per RHS
evaluation. `bench/artcal.jl sensor` sweeps all three, including a per-sensor
CFL ladder over the three Noh geometries.

Measurements:

- **The switch moves one CFL ceiling; the sensor change moves none.**
  `:gated_strain` completes cylindrical Noh at `cfl = 0.2`, where both other
  settings fail, reaching a plateau (0.9353) consistent with its own value at
  0.15 (0.9344), a 33% larger timestep for that geometry. Planar and spherical
  Noh are unmoved by all three settings, so the `cfl ≤ 0.15` guidance for
  converging shocks stands and is now governed by the spherical case alone.
- **`:dilatation` loses both converging geometries**, at the coordinate fold,
  not at the shock. Noh starts from uniform u_r = −1, for which the two
  sensors are analytically identical away from the axis; in the first cells
  Δ = S_rr + S_θθ adds two same-signed components where |S| takes their
  root-sum-square, making the dilatation sensor 2 to 70 times larger where the
  cell measure is smallest. Positivity is lost on step one, and raising
  `C_beta` does not recover it. `:gated_strain` applies the identical switch and
  keeps those geometries, which attributes the failure to the sensor field.
- **Accuracy is a wash for `:gated_strain`**: every column of the battery moves
  in the fourth digit and the movements go both ways. The 0.25% Shu–Osher
  wave-train gain belongs to `:dilatation` alone; gating without changing the
  sensor gives 0.05% *less* than the default.
- **The switch removes 99.4% of β\* on a solenoidal field but not its maximum.**
  On Taylor–Green at 32³, `:gated_strain` leaves 71 of 32768 points above 1e-12
  and a peak at 0.42 of the ungated value. Those are the cusps of |S|, where a
  fourth-difference sensor peaks because |S| passes through zero and
  where the switch degenerates because the vorticity vanishes with it. A
  relative ε scaled to the local |S| was tried and reverted: the scale it would
  use vanishes at the same points.
- **Neither compression-keyed setting is decomposition-independent to
  round-off**: 2e-6 relative for `:gated_strain` and 2e-7 for `:dilatation` over
  three split axes, against 1e-14 for the strain sensor. The sensor fields
  reproduce; H(−Δ) is discontinuous at Δ = 0 and the literature ε is too small
  to have decayed the ratio by the time Δ reaches round-off.

The reach hypothesis was tested and rejected.
The suspicion on record was that β\* does not reach far enough *ahead* of the
front, which pointed at a wider sensor stencil. `bench/nohprobe.jl` was written
to test it and refutes it three ways. β\* reach holds at 14.2–14.8 cells ahead of
the front through a complete run, while the damage sits 3–5 cells ahead. β\*
stands at 14–21% of its own domain maximum on the worst cell at the default CFL,
and at 84–100% of it at the failing one. And above the ceiling the failure does
not begin ahead of the front at all: at ν = 1 the wall cell degrades first,
within five steps, and at ν = 3 the origin cell carries an outward u = +5.6
against an inflow of −1 one step after a density minimum of 1.92 over the whole
line. The restriction is a symmetry-plane startup problem, consistent with
`:gated_strain` moving the one ceiling it moves by relieving the axis cell, not
the shock. Measurements are in
[CALIBRATION.md](CALIBRATION.md#where-the-restriction-originates).

The probe also surfaced a defect outside the item's scope. Runs that
**complete**, including the ν = 1 validation case, carry six to eight
cells of negative internal energy travelling with the front for their whole
duration. `primitives!` floors T_ion at 1e-300 and continues, while the
positivity check reads ρ, which stays positive. The internal energy is
ill-conditioned at the Noh ambient, not an ordinary accuracy error, so this
became a requirement on the positivity failsafe in `ROADMAP.md`: it has to cover
internal energy, not density alone.

Whether `:gated_strain` should become the default is left open: it changes
guarded numbers in the fourth digit across the battery, so the case needs a
re-baseline and a second geometry showing the same gain, and by the mechanism
above that geometry would have to be one whose ceiling is also set by a fold the
gate relieves.

Two things landed alongside. `bench/artcal.jl` now catches `SolverFailure` per
configuration and continues, closing the known limitation that a sweep died at
its first bad point; the failure is raised off a reduced quantity, so the catch
is safe under `mpiexec`. And `test/mpi_tests.jl` gained the first multi-rank
coverage of the artificial-property path at all: every other test in that file
disables it.

## The reference-implementation pass (August 2026)

Reading Miranda's kernels, carried by Pyranda in `pyranda/parcop/`, against
`artificial.jl` identified four differences in the Cook artificial-property
path. Two of the four are now implemented and measured. The measurements are in
`reference/CALIBRATION.md` under "Measured against the reference
implementation"; the decisions are recorded here.

**The sensor smoother** is an explicit nine-point Gaussian in the reference, not
a compact-filter pass. `ArtParams.smoother` offers both and `:gaussian` became
the default: it raises the spherical-origin CFL ceiling 0.15 → 0.4 and the
cylindrical 0.15 → 0.2, and makes the sensor phase 29% cheaper, at the cost of
about seven points of planar wall heating.

**The ringing detector** is a compact eighth derivative in the reference, not
Cook's undivided fourth difference. `ArtParams.detector` offers both;
`compact_d8` is the transcribed operator and `:delta4` remains the default.
The sequence was necessary: a sharper high-pass makes
narrower sensor spikes, and the defect the Gaussian fixes is β\* intermittency
at a symmetry cell, so measuring the detector against the old smoother would
have rejected it for the smoother's fault.

Measurements:

- **The planar and cylindrical CFL restrictions are lifted, not raised.** Under
  `:d8` both geometries reach `t_final` at `cfl = 1.0` with the plateau flat to
  four digits from 0.15 upward, against a ceiling of 0.2 under `:delta4`. This
  is the first setting measured against the `cfl ≤ 0.15` guidance that removes
  the artificial-property restriction outright.
- **The spherical origin goes the other way**, 0.4 → 0.25, and
  `bench/nohprobe.jl` puts the failure at the origin cell with β\* at 1.8% of
  its own maximum. δ⁴'s poor selectivity was also supplying a background
  regularization at the coordinate fold, where the field is smooth and even by
  construction; a selective detector correctly returns nothing there. With the
  timestep predictor, `C_beta`, the dilatation sensor and the sensor reach all
  ruled out previously, the third-order fold closure is the one candidate left
  and the origin cell is where to look.
- **Accuracy improves on six of seven battery columns**, several well beyond
  the fourth digit that separated the β\* sensor variants: the ν = 3 Noh plateau
  error falls 2.45% → 0.42%, ν = 1 wall heating recovers +64% → +53%, and the
  Shu–Osher wave train gains 1.1%. Woodward–Colella peak density is the
  regression, −3.2%.
- **The selectivity is unavailable to μ\* and β\*.** On a resolved wave the two
  detectors separate by 2.6e6 on κ\*, whose input is the internal energy, and by
  1.8 on β\*, whose input is |S|, which carries a cusp wherever the strain
  passes through zero. This is the same geometry that defeats the Ducros switch
  on a solenoidal field, seen from the other side, and it accounts for the
  reference building μ\* and β\* from velocity components and the dilatation. The
  detector result is therefore a lower bound on what the reference method gains,
  which promoted the sensor-field change to the head of the open list.
- **Cost** is eight pentadiagonal line solves per right-hand side, one per
  active dimension per sensor: +80% on the sensor phase and +19% on the whole
  evaluation for the two-species tube.

The default was left at `:delta4`. The battery favours `:d8`, as do the planar
and cylindrical ceilings, but the converging-shock guidance rests on the
spherical case alone. On that case, `:d8` costs 40% of its timestep, while the
four constants are still the δ⁴ fit. A `C_beta` refit and an account of the
origin cell would settle it.

Two findings from the same reading needed no code change. The reference counts
the sound speed once against the minimum spacing where `max_rate` counts it per
active dimension; the difference is recorded in `reference/CALIBRATION.md` and
not adopted, because it would silently rescale `cfl` for every
existing script. And `reference/IMMERSED.md`, written from Pyranda's
documentation, was corrected against `pyrandaIBM.py`. Of the differences left
outstanding, the sensor fields were taken up next and are recorded below;
conservative filtering on a non-Cartesian metric, directional bulk viscosity
and the anchored-difference closure rows are listed in `ROADMAP.md`.

`compact_d8` is the first symmetric banded scheme in the package, so it is also
the first exercise of `BandPlan` with filter-side conventions: right-hand side
added where a derivative's is subtracted, high-edge closure rows mirrored
where a derivative's are negated, and the filter parity used at a coordinate
fold. `test/mpi_tests.jl`
gained the corresponding coverage, the banded reduced-interface `Allgather`
having never before run inside `compute_artificial!`; both sensors reproduce to
round-off across three split axes. `bench/artcal.jl`, `bench/phases.jl` and
`bench/nohprobe.jl` now take their sensor-shape defaults from `ArtParams()`;
the spelled-out copies had gone stale against the default smoother. The
`phases.jl` line-solve counter includes the sensor path and reports the
derivative, smoother and detector solves separately.

## The sensor fields (August 2026)

The third difference identified by the same reading, taken up next because the
detector measurement placed it at the head of the list. Cook builds μ\* and β\*
from the strain magnitude |S|; Miranda builds μ\* from the velocity components
and β\* from the dilatation, neither of which carries an absolute value.
`ArtParams` gained `mu_sensor` (`:strain`, `:velocity`), `reduction` (`:sum`,
`:max`, the directional combination) and a fourth `beta_sensor` setting,
`:ungated_dilatation`, which is the reference's own β\*; the variant tested
previously was that sensor together with a Ducros switch. Every default is
unchanged. Measurements are in `reference/CALIBRATION.md` under "The sensors
read |S|".

**The sensor field determines how much of the detector's selectivity is
usable.** On a velocity sine with no time integration (`bench/artcal.jl
response`), the two detectors reproduce their designed separation of 569× at
eight points per wavelength and 26× at four, to four figures, when applied to
the velocity or the dilatation. Applied to |S| they lie within 1.8× of each
other at every wavelength, the cusps of |S| being grid-scale structure whatever
the flow. The previous pass inferred this from two cases; this measures it
across the spectrum.

**A two-point wave produces no response in any sensor built through a
derivative.** A centered scheme annihilates the Nyquist mode, so |S| and ∇·u
are identically zero there and both Cook's μ\* sensor and the reference's β\*
sensor return zero on the shortest wave the grid carries. Only the
velocity-component sensor responds to it. The property was not recorded here
before and is a second view of the earlier finding that grid-scale dissipation
comes from the compact filter, not from the Cook properties.

Case results:

- **μ\* from the velocity components is a null result on the battery.** Every
  column moves in the fourth or fifth digit under both detectors, and no CFL
  ceiling moves in any of the three Noh geometries. Every case there is
  one-dimensional and `C_mu` is 0.002, so the shear channel does very little in
  them, and this outcome was the expected one.
- **On Taylor–Green, where that channel is active, it behaves as the response
  measurement predicts.** At 64³ the μ\* share of the sink rises 4.5% → 6.0% at
  the expense of the filter's share, and the peak falls 0.3% toward the van
  Rees reference. The directional maximum instead cuts the share to 2.7% and
  moves the peak from t = 8.49 to 8.97, against a reference peak at t = 9.
  Neither effect is distinguishable from a rescaling of `C_mu` at this
  resolution. The velocity field costs +46% on the sensor phase, detecting
  three fields where the strain sensor detects one, and paired with `:d8` it
  makes the sensor phase larger than the rest of the right-hand side.
- **β\* from the ungated dilatation improves accuracy and loses the cylindrical
  axis.** Under `:d8` the ν = 3 plateau error falls 0.49% → 0.24%, wall heating
  +53% → +51% and the Lax contact 0.0044 → 0.0041, while the Shu–Osher train
  loses 0.5% and the Woodward–Colella peak 1.5%. The axis ceiling falls from
  1.0+ to 0.2 under `:d8`, and the geometry is lost outright under `:delta4`.
  The gated variant loses both converging geometries and this one loses only
  the axis, which separates the attribution: the spherical loss belongs to the
  switch and the cylindrical loss to the sensor field.
- **The directional reduction is invisible to the battery.** Every case there
  is one-dimensional, and Σ_d and MAX are the same operation in one dimension.

One defect was found and fixed in the process. `delta4_sum!` extends a field
past a closed edge by clamping the index, which is wrong at O(h), not O(h²),
for a field that is odd across a fold, and the velocity sensor is the
first such field. On u_r = r at the cylindrical axis, the regular behaviour of
a radial velocity, the clamp produced C_mu·ρ·h² of viscosity on the axis cell
where every converging case fails. The odd path now uses the half-offset
mirror, which annihilates that field exactly and agrees with the result `:d8`
obtains through its fold plans. The even path is left on the clamp, since
changing it would move guarded numbers for a second-order effect that has not
been measured; it is item 8 of the calibration remainders.

No default changed, and the item is closed on the measurement.
`mu_sensor = :velocity` has a mechanism behind it, no case in the battery that
resolves it, and a Taylor–Green effect indistinguishable from a rescaling of a
constant that has never been fitted; settling it needs the `C_mu` refit and the
filter calibration, which wait on the same 3-D campaign.
`beta_sensor = :ungated_dilatation` is a measured negative for converging
geometry. `reduction = :max` is a third setting acting on an unfitted constant.

## The C_beta refit (August 2026)

The last open item of the reference-implementation comparison that was neither
blocked on a missing case nor gated on an unmeasured residual. `:d8` changes the
sensor's spatial support enough that the default constants, fitted under δ⁴,
had no claim on it, and the detector's default was held pending the refit.
`bench/artcal.jl beta` and `bench/artcal.jl beta detector=d8`, with a fresh
`:delta4` control that reproduces the detector comparison in every column.
Measurements are in `reference/CALIBRATION.md` under "The C_beta refit under
`:d8`". No default changed.

**`C_beta = 1.0` survives the refit, and does so for a different reason than it
was chosen.** Under `:delta4` the viable window runs 0.25 to 1.0, bounded above
by planar Noh, and 1.0 is a robustness margin over an accuracy optimum near 0.4.
Under `:d8` the window runs 1.0 to 4.0, bounded below by both converging
geometries, which fail outright at 0.5. The two windows intersect in the single
value 1.0. `:d8` also flattens the response to the constant on every smooth
measure: over 0.25 → 4 the Lax contact broadens 31% against 80%, and the
Shu–Osher train loses 1.6% against 2.8%.

The refit does not rescue the spherical origin, resolving the detector decision.
The best ν = 3 ceiling available under `:d8` is
0.3 at `C_beta = 2`, below the 0.4 that `:delta4` reaches at the default
constant, and it costs the ν = 2 ceiling, the ν = 3 plateau and part of the
Shu–Osher train. The remaining decision therefore depends on the origin cell
alone, as the previous pass found.

Two findings came out of the CFL ladder the sweep gained, which reads the
constant at six timesteps where the earlier sweep read it at `NOH_CFL` alone.

- **`C_beta` trades the ceilings against each other and raises none.**
  At 0.25 the planar ceiling passes 1.0 and the cylindrical reaches 1.0 while
  the spherical falls to 0.2; at 1.0 the spherical is highest and the other two
  are lowest. This corrects a one-sided earlier result recorded in the
  calibration remainders, which tested a *larger* `C_beta` against the ν = 1
  restriction, found no movement, and concluded the constant was not involved.
- **Under `:d8` at reduced `C_beta` the cylindrical axis fails below a CFL,
  not above one**, verified as positivity loss and not as the step cap.
  No stability restriction produces that sign; a per-step operation does, since
  a fixed physical interval at half the timestep applies it twice as often. This
  was written up at the time as support for the third-order fold closure, which
  the next entry retires; the observation stands and the attribution does not.
  The per-step operation now carrying it is the compact filter, and the
  measurement is under [filter dt-consistency](#filter-dt-consistency-august-2026).

The sweep now distinguishes its two failure modes, printing positivity loss as
`NaN` and the step cap as `Inf`. Collapsing both into `NaN` made the second
finding look like an artifact and cost a separate probe to resolve.

## The origin cell (August 2026)

The remaining evidence for the detector decision concerned the origin cell.
Two instruments supplied it:
`bench/foldorder.jl`, new, which splits the convergence studies' error norm by
region of the line; and three columns added to `bench/nohprobe.jl` reporting the
symmetry cell on every line, not only when it is the worst cell.
Measurements are in `reference/CALIBRATION.md` under "The fold closure is not
third order" and "The origin cell is a startup transient". No default changed
and no solver source was touched.

**The fold closure is not third order, and the number behind that claim
belongs to the outer wall.** `test/convergence.jl` reports a global max norm, and every one
of its fold studies closes the outer end with a `SlipWallBC` measuring 3.17 on
its own. Split by region, the global maximum sits at the last interior cell in
all four studies, and the fold's own error converges at 6.05 to 7.01 while
sitting three to five orders of magnitude below the interior. A both-ends-walls
control reports 3.23 in the same window, so the instrument does detect a
third-order closure where one exists. Both parities were measured, including the
odd one `test/convergence.jl` does not cover and that a converging calculation
differentiates at the origin.

**The companion reading, that a selective detector is blind at the fold, is
wrong for the same case.** β\* at the origin is indeed negligible while the
field is smooth, but the cell does not fail while the field is smooth. During
the excursion that fails, β\* there reaches the *line maximum* under both
detectors.

**The origin cell evacuates during a startup transient.**
It holds `rho1/rho2` within 0.2% of unity for the first eighty-odd steps, then
undergoes one large excursion. `:delta4` at cfl 0.3 and `:d8` at cfl 0.25 pass
through it; `:d8` at cfl 0.3 enters it about forty steps earlier and the cell
collapses from 1.31 to 0.23 of its neighbour within one sampling interval, with
the internal energy reaching −6335 e₀. The excursion lands at t ≈ 0.394 at
N = 128, 256 and 512 while its step number scales with N, and its relative
amplitude weakens under refinement, so it is a resolved feature of the warm
start, not a grid-scale artifact.

This retires every discretization-order candidate for the converging-shock CFL
ceiling. Two mechanisms are live and neither is demonstrated. First, β\* is
proportional to the density (`src/artificial.jl:502`), so the regularization
collapses as the cell thins. Second, the compact filter is applied per step,
not per unit time, which is the standing model debt 1. The `C_beta` ladder
provides independent evidence for the second mechanism. The ceiling is
therefore a robustness problem at a symmetry cell, which is model debt 2, not a numerics
problem at a fold.

## Filter dt-consistency (August 2026)

The second half of model debt 1, taken up because the `C_beta` ladder pointed at
it: a failure that got worse as the timestep fell is the signature of a
per-step operation, and the filter is applied once per step.
`Numerics` gains `filter_cfl`, off by default. Measurements are in
`reference/CALIBRATION.md` under "The filter dissipates per application, not per
unit time"; the instrument is `bench/filterrate.jl`, new, and `bench/tgv_energy.jl`
gains `cfl` and `filter_cfl` options.

**The filter dissipated per application, by a factor of 3.93 across a 4× CFL
change.** On a parallel shear layer, which is an exact steady Euler solution and
so leaves the filter as the only sink of kinetic energy, the loss tracks the
step count, not the elapsed time (73 : 145 : 289 steps giving
1 : 1.98 : 3.93). Under `filter_cfl` the loss is constant to six significant
figures over the same range.

Two obvious cases do not measure this and both were tried first. A broadband
field saturates, losing 64% of its kinetic energy within tens of steps and then
no more, which collapses the spread to 1.2%. A velocity sine at uniform pressure
is an acoustic oscillation trading kinetic for internal energy far faster than
the filter acts. Total energy shows nothing either, because a symmetric filter
on a periodic grid conserves the discrete sum of every conserved variable
exactly.

**The cost to the calibration is in the attribution, not the total.** At 32³
Taylor–Green, halving the CFL changes peak dissipation by 0.6% but moves the
filter's share of it from 82.2% to 85.0% and the μ\* share from 5.1% to 3.6%,
with `C_mu` held fixed. The sinks compete and do not add: the cascade rate is
set at the large scales, so a filter taking more at the grid scale leaves less
for μ\* and molecular dissipation. A `C_mu` fitted under the unrelaxed
formulation is therefore only reproducible at the CFL it was fitted at. Under
the relaxation the split holds to 0.1 points.

The formulation is `Q ← (1 − w)Q + w·F(Q)` with
`w = filter_interval · dt · rate / filter_cfl`, capped at one. Reading `dt · rate`,
not `solver.cfl`, recovers the CFL of the step as taken, so `StepControl`
backoff and callback-landing shortening are both accounted for, which also
removes the truncated-final-step artifact. At `w = 1` the code takes the
original copy path and forms no blend, so the default is bit-identical: the
full gate reproduces every convergence order and error magnitude exactly, and
77/77 MPI checks pass at 2, 4 and 8 ranks.

Whether the relaxation should become the default belongs to the α and cadence
fit, not to this work, since under the relaxation α and the reference CFL
set the dissipation jointly.

## The positivity failsafe (August 2026)

Model debt 2, delivered as `StepControl.floor_ratio` and `floor_scope` with the
repair in `apply_positivity_floor!`. Both are off by default. Measurements are
in `reference/CALIBRATION.md` under "The negative internal energy is not a
rounding artifact".

The debt asked for a conservation-aware local floor covering internal energy and
not only density, applied on detection, counted, and reported loudly. The work
changed the *scope* of that repair, because building the floor made the
condition measurable for the first time and the measurement did not support
repairing all of it.

**The negative internal energy `bench/nohprobe.jl` found is not a rounding
artifact, and it is not a state the scheme cannot handle.** Over the complete
ν = 1 Noh validation case the state carries 24991 cell-steps of negative internal
energy reaching −718 ambient units, while the total energy density and the
mixture density stay positive at every cell of every one of its 3724 steps. The
run still reaches the plateau to within 0.07%. The wall region runs as a
pressureless layer, and the scheme remains stable there.

**Forcing that internal energy positive terminates the run.** The worst cell
needs a 5% velocity damping after one step. Applied every step it removes 1.5e-4
of the total momentum immediately and produces cells of negative total energy by
step 4 that the unrepaired trajectory never produces; the run then fails at step
18 with `:dt_collapse`, where the unfloored run completes. Raising the total
energy instead is the same order of intervention, so this is a property of the
case, not of the repair policy.

The failsafe therefore carries a scope. `:representable`, the default when
enabled, repairs a mixture density or a total energy density below the floor and
counts internal energy below it without touching it; on the Noh case it repairs
nothing and reproduces the unfloored run bit for bit while reporting all 24991
cell-steps. `:internal_energy` opts into the repair above, for sweeps and
development runs where a bounded run is wanted and a faithful trajectory is
not.

The floors themselves are unit-free, derived once per `run!` as `floor_ratio`
times the global minimum density and internal energy of the state that call
starts with, on the same reasoning as `dt_min_ratio`. The repair clips negative
partial densities and rescales the survivors onto the point's mixture density,
so mass is exactly preserved wherever the mixture density is healthy;
the internal-energy repair damps the velocity at fixed total energy, and the
fallback where there is no kinetic energy left to convert raises the total
energy at fixed momentum. Each branch conserves one of the two exactly and
tallies what it did to the other. `FloorTally` on the solver accumulates the
reduced counts; `run!` warns on the first firing and summarizes on the way out.

Enabling the failsafe removes `:negative_density` as a route to
`SolverFailure`, since the density that check reads can no longer reach zero;
`:dt_collapse` and `:nonfinite` remain, so a sweep still terminates and does
not grind. The default configuration is untouched: the full gate reproduces every
convergence order and error magnitude exactly, the validation battery matches
its guards to every printed digit, and MPI checks pass 90/90 at 2, 4 and 8
ranks, twelve of them new and covering decomposition-independence of the repair
and its tally.

## AMR/GPU Stage 1 — level-transfer operators (August 2026)

Stage 1 of `reference/AMR_GPU.md` is delivered: `src/transfer.jl` implements
the Miranda 3:1 invertible transfer pair as ordinary compact schemes:
restriction is a `CompactScheme` (tridiagonal left-hand side against the
pentadiagonal Gaussian) and prolongation a `BandedCompactScheme` with q = 2 and
its interior rows normalized to a unit diagonal, so `plan_direction` supplies
the distributed spike solve and the fold parity variants without new solver
machinery. `plan_transfer`, `restrict!` and `prolong!` bind the pair to a
dimension of matched fine/coarse decompositions; the transfer dimension itself
must not be decomposed until the patch stages own the 3:1 rank alignment.

The stage's open questions were resolved by measurement, recorded in the
Stage 1 status block of `reference/AMR_GPU.md`: the sampling convention is
filter-then-subsample against interpolate-then-deconvolve, under which
restriction is an exact left inverse of prolongation (coarse → fine → coarse at
8.9e-16 for arbitrary data) while fine → coarse → fine converges at the
selectable interpolation order (measured 3.97 / 5.93 / 7.97 at orders 4/6/8);
the smoothed Cook sensor of a captured-shock profile round-trips at ≤ 1.13×
against the feared ~20× deconvolution bound; shock round-trip pollution decays
≈ 3.4× per point outside the shock footprint, giving the ~4-coarse-cell buffer
figure for Stage 3/4 tagging; and the anti-aliasing prefilter question is
answered (trades accuracy for undershoot margin, held in reserve).

Verification: four new serial testsets (64 total) covering DC gain to the
last bit, closure-row sums, pair invertibility at 2e-15 through the live
plans, the left-inverse identity, measured round-trip order, the fold-variant
equivalence row for row, and the sensor-injection bounds; the MPI suite gains
a distributed pair round trip on split dimensions and a transverse-decomposed
transfer (93/93 at 2, 4 and 8 ranks). Convergence and validation guards are
unchanged to every printed digit, as they must be for a change that adds
operators without touching the solver.

## AMR/GPU Stage 2 — patch abstraction and storage generalization (August 2026)

Stage 2 of `reference/AMR_GPU.md` is delivered, carrying GPU Stage G0 in the
same change: `Solver` is split into physics configuration plus per-patch state
(`src/patches.jl`), each `Patch` holding its own communicator, `Decomp`,
operator plans, folds, and every field array typed
`A <: AbstractArray{T,3}` behind a backend object (`CPUBackend` default).
`Decomp{T}` types the halo and pair buffers, closing the known `Float64`
hardcodes in `decomposition.jl` and `viz.jl` and clearing the mixed-precision
storage blocker in `reference/ROADMAP.md`. A single-patch solver forwards the
patch-owned property names to its sole patch, so the entire existing surface
reads unchanged.

The hard gate held: `test/convergence.jl` bit-identical down to the error
magnitudes, validation guards matched to every printed digit, 64/64 serial
testsets and 93/93 MPI checks unchanged at 2, 4 and 8 ranks, and
`bench/jetcheck.jl` two reports *lower* (a `Nothing`-plan `apply_along!`
method makes the dead fold branch statically resolvable).

Multi-patch delivery, first cut: slab layouts along one dimension
(`patch_grid`), rank partitioning by one `MPI.Comm_split` proportional to
patch volume, interface ghost exchange and shared-plane averaging between RK
stages (`sync_patches!`), and extended-data interface closures whose
left-hand sides couple no ghost unknown. The measured gates are a manufactured
smooth solution at order 3.1–3.5 across the interface, pulse reflection
2.3e-3 of incident at 192 points converging at ≈ 5th order, conservation
drift 1.2e-8 relative against 4.5e-15 single-patch, and reproduction of the
serial two-patch answer under rank partitioning (bitwise at one rank per
patch, round-off-level once a patch is itself decomposed); they are recorded
with the scope restrictions (no folds, no banded schemes, no `:d8`,
single-patch I/O) in the Stage 2 status block of `reference/AMR_GPU.md`.
Serial coverage adds five testsets (69 total); the MPI suite adds the
rank-partitioned oracle (95/95 at 2, 4 and 8 ranks).

## AMR/GPU Stage 3 — static two-level refinement (August 2026)

Stage 3 of `reference/AMR_GPU.md` is delivered in its first cut: one level-1
patch at refinement ratio 3 over a setup-time region, advancing with the
global timestep inside `solver.patches`, coupled to the root level by a
per-stage ghost-and-boundary-plane imposition from the coarse state and a
per-step write-back of the fine state onto the covered region
(`src/levels.jl`). Two measured corrections to the plan's prescription came
out of the gates. First, the invertible Stage 1 transfer pair is not the
right live coupling: deconvolution assumes filtered-sample input and the
live coarse solution is point samples, so the pair measured order 1.3–1.7
where the point-sample halves (order-6 interpolation up, coincident-node
injection down) measure order 3.46/3.64 with errors three decades lower.
The pair stays selectable (`level_restriction = :filter`) and remains the
tool for initializing new fine regions in Stage 4. Second, restricting all
the way to the coarse-fine boundary closes an amplifying feedback loop
(measured gain ≈ 2 per step); holding the write-back two coarse nodes off
the boundary flattens it.

Gates, recorded in the Stage 3 status block and guarded in
`test/level_tests.jl`: manufactured solution across the boundary at order
3.46/3.64; Sod crossing the refinement boundary in both directions with the
sensors live at 6.4e-10 ahead-of-shock interface noise; two-level mass
drift 1.36e-4 over the crossing against 5e-11 unrefined; a 2-D refined
region under diagonal advection at 4.3e-8. Scope: serial, Cartesian,
unstretched, tridiagonal schemes, `:delta4`, one region. The unrefined
solver is untouched: convergence and validation guards reproduce to every
printed digit and the MPI suite is unchanged.

## AMR/GPU Stage 4 — subcycling, tagging, and regridding (August 2026)

Stage 4 of `reference/AMR_GPU.md` is delivered in its first cut on the
Stage 3 single-region base. `subcycle = true` selects the Berger–Oliger
step: the coarse level advances first with the fine level frozen, then the
fine level takes three steps of dt/3, its shell imposed at every fine stage
time from a cubic Hermite reconstruction of the coarse trajectory on the
buffered box (one extra coarse RHS evaluation per step supplies the second
endpoint). The global dt reduction weights each patch's rate by 3^-level,
so the step is coarse-limited, and each level filters its own state at its
own step cadence. `regrid_interval = K` adds tagging and regridding: every
K coarse steps the region moves to the buffered bounding box of the cells
where the relative undivided fourth difference of the mixture density
exceeds `tag_threshold`, the new fine patch is initialized by the order-6
point-sample interpolation, and surviving fine data is copied across the
overlap on the coincident lattice.

Measured gates (`test/level_tests.jl`, values in the Stage 4 status block):
the subcycled entropy-wave MMS reproduces the global-dt orders (3.49/3.66)
at a third of the steps, so the Hermite boundary data does not bind; the
Stage 3 Sod gate rerun subcycled improves both guards (noise 5.7e-11
against 6.4e-10, drift 9.8e-5 against 1.36e-4); a moving region tracks the
Sod shock with composite error 2.8e-3 against the uniform-fine reference
where the uniform-coarse baseline sits at 7.3e-2; and on Shu–Osher the
region grows to hold the wave train, 10× better in L-infinity over the
train than uniform-coarse at half the reference's steps. One robustness
finding: subcycling widens the `compute_dt` lag to three substeps, so a
discontinuity inside the region at t = 0 needs cfl ≤ 0.2 or
`StepControl(retries)`; making retries work exposed two latent
rollback defects (NaN surviving in the artificial coefficient arrays and in
the low-storage accumulator after a non-finite failure), both now fixed and
gated on that failure kind. Still open from the stage's gate list: the
3-D wall-time and memory demonstration, tile clustering, and the TGV
filter-cadence measurement.

## GPU Stage G1 — pointwise kernels via KernelAbstractions (August 2026)

G1 of the `reference/AMR_GPU.md` GPU track is delivered in its first cut:
every pointwise phase (both gas-model `primitives!` methods, the flux
assembly, the RK update, the δ⁴ detector and every sensor-to-coefficient
loop, the metric gradient corrections and momentum sources, the body-force
source, the gradient scaling, and the flux-divergence accumulation) is now
one shared per-point body launched through `pointwise!`
(`src/pointwise.jl`): `Array` storage takes the existing `@threaded` loop
and any other storage a KernelAbstractions kernel on its own backend, with
one generic kernel splatting the body so no second kernel body exists
anywhere. KernelAbstractions becomes the package's fifth dependency.

The plan's CPU acceptance measurement went against replacing `@threaded`:
at 64³ the KA CPU backend runs 2.8× (fluxes) to 40–50× (RK update) slower
(per-launch task spawn with no work threshold), while on a small 2-D case it
is up to 2.4× faster, so the deficit is launch policy, not generated code.
Per the plan's contingency the routing is static: `Array` on `@threaded`,
device arrays on KA. The KA CPU path reproduces the threaded path bitwise
over full runs (the new testset, 78 serial testsets total), the default
path is bit-identical to the pre-G1 solver on every convergence and
validation guard, and jetcheck/audit hold probe for probe. One trap is
recorded in `CLAUDE.md`: a `Type` argument inside the launcher's Vararg
defeats specialization and cost `assemble_fluxes!` 9× until removed.
Device bring-up started the same week on the workstation's RX 6800 XT
(AMDGPU.jl on Windows): the plain-argument bodies reproduce the CPU bitwise
on device through `pointwise_ka!` and the automatic routing, the δ⁴ stencil
runs 4× faster than 8-thread CPU in a single launch, and a collection-typed
kernel argument was found to hang in adaptation without raising an error, so
the isbits argument adaptation (with the `max_rate` mapreduce and the Nasa9
mirror) is the open G-track work, and
`bench/device_bringup.jl` is the script that reproduces the measurements.

The device-argument adaptation followed in the same push: the field
collections reach the per-point bodies as `FieldVector`/`FieldMatrix`
(zero-cost host wrappers, built once per patch as `Patch.field_tuples`,
whose only job is to carry the Adapt rule), and the gas-model EOS objects
adapt to isbits coefficient mirrors at kernel launch. A first design held
the tuples on the host and cost `assemble_fluxes!` 3× on the `@threaded`
path through runtime tuple indexing, so the tuples materialize only at
launch. With the adaptation in place the full flux-assembly body, every
collection plus the mirrored `IdealMixture`, runs on the RX 6800 XT
bitwise against the CPU and 9.9× faster than 8-thread `@threaded` at 64³
with two species; `bench/device_bringup.jl` carries the measurement. The
default path stayed bit-identical on every guard, and jetcheck/audit held
probe for probe.

## Wall closures and the tridiagonal C8 (August 2026)

The closed-domain order of the tridiagonal derivatives had sat at 3.17 since
the first convergence study, set by the third-order one-sided first row of
the Carpenter–Gottlieb–Abarbanel cascade. Two alternatives now sit behind a
`closures` keyword on `lele_d1_6` and the new `lele_d1_8`. `:cascade4`
substitutes Lele's fourth-order one-sided row (α = 3) and measures 4.02.
`:brady_livescu` applies the rows of Brady and Livescu (Computers & Fluids
2019), set 1 of the T6 and T8 databases in their Data in Brief companion:
every row one order below the interior, discretely conservative under
tabulated quadrature weights, and chosen by optimization on the 1-D Euler
equations for stability without a filter. All sixteen T6 sets were
evaluated from the published constraint files; every one is exact through
degree five on every row, and set 1 is both the published table and among
the best conditioned. Measured wall orders are 5.88 for C6 and 7.91 for C8.
The cost is conditioning: those rows are far from diagonally dominant, and
the closed line's condition number rises from 16 to about 1e3 (T6) and 4e3
(T8), and `:cascade3` therefore stays the default.

`lele_d1_8` is the seven-point tridiagonal member of Lele's family (α = 3/8,
a = 25/16, b = 1/5, c = −1/80), measuring 8.00 periodic. It exists because
the pentadiagonal C10 is confined to a single patch while the tridiagonal
solve carries the decomposed, multi-patch and device paths, so C8 is the
highest order available on the full stack; the KA-CPU device comparison is
bitwise for it and for both Brady–Livescu sets. Every default-path guard
came out bit-identical.

## Wall closures under shocks, and the filter's wall rows (August 2026)

The Brady–Livescu rows were then run against the wall-bounded shock cases,
which `test/cases.jl` now exposes through `deriv` and `filt` keywords on
`woodward` and `noh_case`. Both sets fail every case at every CFL under
the default configuration, and the cause was traced to a two-cell wall
mode of the bulk-viscosity operator D(β D) seeded each step by the state
filter's second-order F2 wall row: the warm-started planar Noh, with a
smooth plateau at the wall and nothing arriving, grows ρ_wall from 4 to 20
with κ\* and μ\* switched off, while the same rows are clean on a smooth
pulse with the filter on or off and on a wall-free mirrored Noh. One filter
pass measured on a closed line is second order along the whole line under
the cascade, so the filter, not the derivative closure, sets the wall order
of every filtered run.

`compact_filter(closures = :onesided)` adds the paper's one-sided
eighth-order rows, derived at construction from polynomial exactness plus a
Nyquist zero, not tabulated (the derivation reproduces the interior
stencil and the published row 2). One pass is then eighth order (8.07,
recorded in `test/convergence.jl` beside the cascade's 1.88), the closed
operator amplifies less under repeated application than the cascade, and on
the planar Noh case the wall density deficit falls from 64% to 27% at
N = 400 with the plateau and shock unchanged. Under it C6 `:brady_livescu`
survives Woodward–Colella and the warm-started Noh, the C8 set survives the
smooth pulse it failed under the cascade, and `:cascade4` fails even the
smooth pulse, since the F2 row damped its negative-real-part eigenvalues. The
default stays the cascade: the constants and the validation guards were
calibrated under it.

Two smaller items closed alongside. In Float32 the Brady–Livescu closed
lines floor near 1e-3 and the default cascade is the more accurate closure
from N = 48 up, pinned in `test/float32_validation.jl`; and `plan_direction`
now counts closure rows only at a rank's closed ends, so the T8 set needs 13
points only along a dimension closed at both ends, not on every dimension.
`test/convergence.jl` also prints each study's L2 order beside the guarded
max-norm one: 3.69 / 4.44 / 6.51 / 8.45 for the four C6 and C8 wall studies
against 3.17 / 4.02 / 5.88 / 7.91, the half-order the solution norm gains
over the pointwise wall error. Every previously recorded order and error
magnitude came out bit-identical.

## The level hierarchy (August 2026)

Item 1 of the production-AMR sequencing in `reference/AMR_GPU.md`. The
two-level coupling generalized to a chain of nested levels: `Level` holds
a depth, its patch indices, and one `LevelTransfer` per patch (now
carrying `coarse_index`), `solver.levels` replaces the single
`solver.level_transfer`, the coupling routines take a transfer explicitly
and loop over the hierarchy (root down for shell imposition, finest up for
restriction), and `subcycled_step!` became the recursive
`_advance_level!`, with the operation order at two levels preserved. A
refined region is given in its parent patch's node space; `refine` accepts
a vector of them.

Measured: the two-level entropy-wave errors and step counts are
bit-identical to the recorded serial references (6.178253464383943e-10 at
159 steps; subcycled 5.946196868222842e-10 at 56), every convergence
order and error magnitude came out bit-identical, and the jetcheck and
audit counts are unchanged. A three-level nest on the entropy wave
converges at 3.31 / 3.74 (global dt) and 3.35 / 3.79 (subcycled) against
the two-level 3.46 / 3.64, and a subcycled Sod through two nested region
pairs stays positive with ahead-of-shock noise 6.4e-10, the two-level
figure. The serial suite grew to 112 testsets and the MPI suite to 114
checks. Regridding remains two-level; per-substep rate re-evaluation is
deferred.

## Tiled levels (August 2026)

Item 2 of the production-AMR sequencing in `reference/AMR_GPU.md`. A
refined level is now a set of tiles on a global lattice of edge `tile`
parent nodes (0 keeps one patch per level), abutting tiles sharing their
interface plane and coupled through the root's interface records, held on
the `Level` and run by `sync_patches!` and after every stage of the
subcycled driver. The shell imposition writes only parent-fed faces
(`LevelTransfer.imposed`), the restriction margin stands off those faces
only, a transfer reads and writes several parent patches
(`coarse_indices`), and regions are given in the parent level's node space.
A two-level regrid on a tiled level is a set difference over lattice
cells: surviving tiles keep their arrays and state.

Measured: `tile = 0` bit-identical to before; a tiled 1-D level at 6.0e-10
against the one-patch 6.2e-10 (orders 3.95 / 3.91); a 2×2 tile nest in
2-D at 4.29e-8 against 4.27e-8; decomposed tiled runs at serial values to
round-off; the tiled regrid tracks the Sod shock through contiguous lattice
cells. An annular tag set in 2-D is covered at 41% (tile 6) and 47% (tile
12) of its bounding box (`bench/amr_tiles.jl`); per-tile scratch memory
(0.4–2.5 MB per 2-D tile) and setup (0.06–0.12 s per tile) argue for tile
edges of 12 or more in 3-D. The warm step cost of a tiled level is not yet
measured; the bench script is the instrument. The serial suite grew to
116 testsets and the MPI suite to 119 checks; the MPI tiled check is
bounded to ten steps because any 2-D case at np = 8 runs at ~7 s/step on
the workstation, one patch or four tiles alike.

## Review fixes on the tiled levels (August 2026)

A design review of the two items above found four defects, all fixed. A
node shared by four tiles (eight in 3-D) did not reach a common value
under one flat pass of pairwise plane averaging (copies 1, 2, 3, 4 ended
at 2.23, 2.68, 2.41, 2.68), and the diagonal corner ghosts of an interior
tile were written by nothing; the level records are now built and run per
dimension, each phase's records spanning the earlier dimensions' padded
ranges with the halo exchange between phases (`_sync_level_records!`),
which gives every copy the mean and fills the corners, pinned exactly in
both suites. A parent level's tile ghosts went stale after its children's
restriction; the level now re-syncs after restriction in the driver and
in `restrict_level!`. A child's buffered box could reach a parent's
imposed boundary plane; nesting is now checked against the parents' own
nodes (`_erode`). After a regrid a fresh tile now takes the planes it
shares with survivors one-way (`_seed_planes!`, through weighted plane
combination) and a departing tile restricts once more before removal.
The review also reordered the sequencing (ownership and the
patch/workspace split ahead of tagging and I/O) and recorded why a
per-substep rate check is not yet built; both are in `AMR_GPU.md`. The
tile=0 paths stay bit-identical; the 2×2 tile nest moved at the 1e-15
level. Serial suite 118 testsets; MPI 121 checks.

## Tag criteria, hysteresis and covered masks (September 2026)

The regrid tag became a union of per-point criteria over the parent
level's state, each a `pointwise!` body: the relative δ⁴ρ that existed,
the artificial diffusivity number of the last right-hand side read from
the per-patch coefficient arrays, the mass-fraction change per cell, the
vorticity magnitude, and a user closure; every new criterion is off by
default and the δ⁴ path reproduces the Sod regrid cases bit for bit.
Derefinement gained a hold band (`untag_ratio`, default 2) and a tile
lifetime, with the tag history (`checks`, `created`) derived from the
reduced flags on every rank; the band holds the marginal tile ahead of
the Sod shock one check longer and stops its re-creation, which moved the
MPI suite's pinned tile sets, and `untag_ratio = 1` reproduces the old
trajectory exactly. Every patch carries a per-orthant covered mask
written at setup and every regrid from the child regions, and the
diagnostics gained composite forms routed through it: exact for a linear
field on one refined patch, a tile nest and a three-level nest, with the
unmasked sums off by the covered volume. Measured: the refined
Taylor–Green energy history at 24³ with an 8³ region stays within 2.5e-4
of the single-level one over 61 steps where the unmasked sum sits 1.4e-2
above; the mixing-layer cost case at np = 8, its mixedness now the masked
quadrature, lands 4.6× closer to the fine answer than the coarse run under
the δ⁴ρ tag and 4.1× under the sensor tag alone, at 43% and 35% of the
fine wall. Serial suite 127 testsets; MPI 185 checks.

A review of the item found three defects, all fixed. A box regrid whose
tagged set lay inside the upper margin band kept the tagged end and
widened outward, so the box reached parent samples the gather never
fills and interpolated zeros into the fine state (a constant density came
out between −0.085 and 1.085); both ends now clamp into the feasible
interval, the widening runs inward, and the regrid re-asserts the nesting
rule. A hold-only tag signal rebuilt the box around the held nodes, a
41-node box shrinking to four, where the band's contract is to keep it;
a signal with no tagged node now keeps the box whole. The recursive
driver passed the root's 0-based step count into a 1-based recursion, so
level 2 saw substep counts 4–12 in place of 1–9 and `filter_interval > 1`
applied the wrong number and phase of passes at depth two; two-level runs
were unaffected. Serial suite 128 testsets.
