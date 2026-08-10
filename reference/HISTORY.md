# CompactLES — Completed work

This file records completed roadmap phases: what was done, when, and what was
measured, so `ROADMAP.md` carries only open work. It is scoped to roadmap-level
phases; the git history records individual changes. Durable numerical findings
remain in `reference/CALIBRATION.md` and `reference/CLUSTER.md`; this file
points at them rather than restating them.

## Contents

1. [Phase 0 — extensibility seams (July 2026)](#phase-0--extensibility-seams-july-2026)
2. [Phase 1 — shock-tube validation and EOS generalization (July 2026)](#phase-1--shock-tube-validation-and-eos-generalization-july-2026)
3. [Parallel HDF5/XDMF output (July 2026)](#parallel-hdf5xdmf-output-july-2026)
4. [Adaptivity groundwork (July 2026)](#adaptivity-groundwork-july-2026)
5. [Near-term corrections (July 2026)](#near-term-corrections-july-2026)
6. [Model debt 1 — the dilatation-gated β\* (July 2026)](#model-debt-1--the-dilatation-gated-beta-july-2026)
7. [The reference-implementation pass (August 2026)](#the-reference-implementation-pass-august-2026)

## Phase 0 — extensibility seams (July 2026)

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
   costs an equation set rather than a rediscovery of the layout throughout the
   solver.
3. **Concrete parameterization of `Solver`** for the fields that were abstract.
   `FoldSpec` parameterization remains open and is listed under Known
   limitations in `CLAUDE.md`.
4. **Caller-owned stage storage.** `Workspace` holds the two low-storage RK
   arrays, so splitting, subcycling, and future IMEX schemes can own their
   stage storage rather than having `run!` allocate it internally.

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
measurements recorded under [model debt 1](#model-debt-1--the-dilatation-gated-beta-july-2026):
the failure begins at the symmetry plane, not ahead of the front.

**Artificial-property calibration.** `bench/artcal.jl` swept each constant over
the battery; results and recommendations are in `reference/CALIBRATION.md`.
The shipped defaults were retained, with one substantive correction to the CFL
guidance and three findings: the `C_beta` upper bound is a stability bound
rather than an accuracy one; `C_kappa = 0` does not support a converging strong
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

**EOS generalization.** The sites that previously assumed an ideal gas outside
the function barrier — NSCBC's LODI algebra (φ = ∂(ρe)/∂p, ∂φ/∂Y_k) and the
artificial-conductivity scale — became EOS dispatch points. The full contract
is written down at the top of `physics.jl`. Two models exercise it beyond
`IdealMixture`: `StiffenedGas` (exact perfect-gas limit at p∞ = 0, the natural
precursor to Mie–Grüneisen) and `Nasa9Mixture` (piecewise temperature-dependent
cp with a bounded Newton inversion of e(T)). The NASA CEA thermodynamic and
transport databases ship verbatim in `data/` beside their Apache license;
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
  prolongation — the quantitative bound that shapes the interface treatment in
  the plan.
- The SBP–SAT literature for implicit operators was surveyed as the rigorous
  alternative; the references and the three documented considerations are
  retained in `reference/AMR_GPU.md`. An earlier claim that adaptivity required
  moving to a cell-centered grid was withdrawn after that survey: multiblock
  SBP–SAT interfaces are node-coincident, and non-conforming interfaces are
  handled by interpolation operators rather than by staggering.

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
  derivation restricts the geometry declares that restriction here rather than
  rechecking it per RHS call or leaving it in prose. Both NSCBC conditions now
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
the trip count and refuses to spawn for one. Measured at 256×256×1 on the
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

<a id="model-debt-1--the-dilatation-gated-beta-july-2026"></a>

## Model debt 1 — the dilatation-gated β\* (July 2026)

The first model debt, closed. The two halves of the literature refinement turn
out to be separable and to behave completely differently, which is the substance
of the result. Full measurements are in `reference/CALIBRATION.md`.

`ArtParams.beta_sensor` now takes three values. `:strain` is the Cook original
and remains the default, bit-identical to before the change. `:gated_strain`
multiplies that sensor by the Ducros-style compression switch
H(−Δ)·Δ²/(Δ² + |ω|² + ε), with Δ = ∇·u the dilatation, ω the vorticity, H the
Heaviside step and ε = 1e-32, for one pointwise pass and no line solves.
`:dilatation` is the full form of Mani, Larsson and Moin (JCP 228, 2009), which
also rebuilds the sensor from ∇·u and costs an additional smoothing pass per RHS
evaluation. `bench/artcal.jl sensor` sweeps all three, including a per-sensor
CFL ladder over the three Noh geometries.

What was measured:

- **The switch moves one CFL ceiling; the sensor change moves none.**
  `:gated_strain` completes cylindrical Noh at `cfl = 0.2`, where both other
  settings fail, reaching a plateau (0.9353) consistent with its own value at
  0.15 (0.9344) — a 33% larger timestep for that geometry. Planar and spherical
  Noh are unmoved by all three settings, so the `cfl ≤ 0.15` guidance for
  converging shocks stands and is now governed by the spherical case alone.
- **`:dilatation` loses both converging geometries**, at the coordinate fold
  rather than at the shock. Noh starts from uniform u_r = −1, for which the two
  sensors are analytically identical away from the axis; in the first cells
  Δ = S_rr + S_θθ adds two same-signed components where |S| takes their
  root-sum-square, making the dilatation sensor 2 to 70 times larger where the
  cell measure is smallest. Positivity is lost on step one, and raising
  `C_beta` does not recover it. `:gated_strain` applies the identical switch and
  keeps those geometries, which attributes the failure to the sensor field.
- **Accuracy is a wash for `:gated_strain`** — every column of the battery moves
  in the fourth digit and the movements go both ways. The 0.25% Shu–Osher
  wave-train gain belongs to `:dilatation` alone; gating without changing the
  sensor gives 0.05% *less* than the default.
- **The switch removes 99.4% of β\* on a solenoidal field but not its maximum.**
  On Taylor–Green at 32³, `:gated_strain` leaves 71 of 32768 points above 1e-12
  and a peak at 0.42 of the ungated value. Those are the cusps of |S|, where a
  fourth-difference sensor peaks precisely because |S| passes through zero and
  where the switch degenerates because the vorticity vanishes with it. A
  relative ε scaled to the local |S| was tried and reverted: the scale it would
  use vanishes at exactly those points.
- **Neither compression-keyed setting is decomposition-independent to
  round-off**: 2e-6 relative for `:gated_strain` and 2e-7 for `:dilatation` over
  three split axes, against 1e-14 for the strain sensor. The sensor fields
  reproduce; H(−Δ) is discontinuous at Δ = 0 and the literature ε is too small
  to have decayed the ratio by the time Δ reaches round-off.

Whether `:gated_strain` should become the default is left open: it changes
guarded numbers in the fourth digit across the battery, so the case needs a
re-baseline and a second geometry showing the same gain.

Two things landed alongside. `bench/artcal.jl` now catches `SolverFailure` per
configuration and continues, closing the known limitation that a sweep died at
its first bad point — the failure is raised off a reduced quantity, so the catch
is safe under `mpiexec`. And `test/mpi_tests.jl` gained the first multi-rank
coverage of the artificial-property path at all: every other test in that file
disables it.

## The reference-implementation pass (August 2026)

Reading Miranda's kernels, carried by Pyranda in `pyranda/parcop/`, against
`artificial.jl` identified four differences in the Cook artificial-property
path. Two of the four are now implemented and measured. The measurements are in
`reference/CALIBRATION.md` under "Measured against the reference
implementation"; what follows is what was decided.

**The sensor smoother** is an explicit nine-point Gaussian in the reference, not
a compact-filter pass. `ArtParams.smoother` offers both and `:gaussian` became
the default: it raises the spherical-origin CFL ceiling 0.15 → 0.4 and the
cylindrical 0.15 → 0.2, and makes the sensor phase 29% cheaper, at the cost of
about seven points of planar wall heating.

**The ringing detector** is a compact eighth derivative in the reference, not
Cook's undivided fourth difference. `ArtParams.detector` offers both;
`compact_d8` is the transcribed operator and `:delta4` remains the default.
The sequencing mattered and is worth keeping in view: a sharper high-pass makes
narrower sensor spikes, and the defect the Gaussian fixes is β\* intermittency
at a symmetry cell, so measuring the detector against the old smoother would
have rejected it for the smoother's fault.

What was measured:

- **The planar and cylindrical CFL restrictions are lifted, not raised.** Under
  `:d8` both geometries reach `t_final` at `cfl = 1.0` with the plateau flat to
  four digits from 0.15 upward, against a ceiling of 0.2 under `:delta4`. This
  is the first setting measured against the `cfl ≤ 0.15` guidance that removes
  the artificial-property restriction rather than moving it.
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
  1.8 on β\*, whose input is |S| — which carries a cusp wherever the strain
  passes through zero. This is the same geometry that defeats the Ducros switch
  on a solenoidal field, seen from the other side, and it is why the reference
  builds μ\* and β\* from velocity components and the dilatation instead. The
  detector result is therefore a lower bound on what the reference method gains,
  which promoted the sensor-field change to the head of the open list.
- **Cost** is eight pentadiagonal line solves per right-hand side, one per
  active dimension per sensor: +80% on the sensor phase and +19% on the whole
  evaluation for the two-species tube.

The default was left at `:delta4`. The battery favours `:d8` and the planar and
cylindrical ceilings favour it emphatically, but the converging-shock guidance
rests on the spherical case alone and that is the case it costs, while the four
constants are still the δ⁴ fit. A `C_beta` refit and an account of the origin
cell would settle it.

`compact_d8` is the first symmetric banded scheme in the package, so it is also
the first exercise of `BandPlan` with filter-side conventions: right-hand side
added rather than subtracted, high-edge closure rows mirrored rather than
negated, and the filter parity used at a coordinate fold. `test/mpi_tests.jl`
gained the corresponding coverage, the banded reduced-interface `Allgather`
having never before run inside `compute_artificial!`; both sensors reproduce to
round-off across three split axes. `bench/artcal.jl`, `bench/phases.jl` and
`bench/nohprobe.jl` now take their sensor-shape defaults from `ArtParams()`
rather than spelling them out, which is what let them go stale against the
shipped smoother, and the `phases.jl` line-solve counter reports the derivative,
smoother and detector solves separately instead of omitting the sensor path.
