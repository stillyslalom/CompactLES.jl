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
require `cfl ≤ 0.15` because a dispersive density undershoot outruns the
artificial viscosity (timestep lag was tested and excluded as the cause), and
the spherical-origin fold does not accept initial data resolved over fewer than
about three cells, nor the singular t = 0 start of spherical Noh. The
cylindrical axis accepts both; the difference between folds remains
unexplained.

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
