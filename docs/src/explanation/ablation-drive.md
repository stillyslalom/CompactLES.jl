# Ablation drive from a hohlraum code

```@meta
CurrentModule = CompactLES
```

CompactLES has no radiation transport, laser deposition, plasma equation of state or
ablation physics. To run the capsule-only step of LLNL's two-step method (an integrated
hohlraum calculation gives the radiation drive on the capsule, which then drives
capsule-only implosions), it takes the drive as a source term. [`AblationDrive`](@ref)
is the first, deliberately reduced, stage of that coupling: a rocket closure. The
hohlraum step is Briareus.jl, a
view-factor code, whose `CapsuleDrive` hands over the capsule's radiation temperature
`Tr(t)`, the capsule-averaged incident flux, its Legendre ratios `aₗ/a₀` (`l ≤ 8`,
`P2 > 0` pole-hot) on a uniform 50 ps grid, the radius the flux was evaluated on, and
provenance.

## Coupling in memory

Both codes are Julia, so the drive passes in memory, through a package extension,
rather than through a file. `CompactLESBriareusExt` (in this package, loaded when
`Briareus` is) adds one method, [`radiation_drive`](@ref)`(::Briareus.CapsuleDrive)`,
which reads the plain-data `CapsuleDrive` into an `AblationDrive`:

```julia
using Briareus, CompactLES
history = drive_history(model, op, beams_at, times; lmax = 8)   # Briareus
cd = capsule_drive(history, model)                              # Briareus → plain data
drive = radiation_drive(cd; peak = 145.0, driver = 3)           # CompactLES source
problem = Problem(...; metric = SphericalMetric(), sources = (drive,))
```

The extension lives here rather than in Briareus because the code it holds is
CompactLES physics (a source term, its calibration, and later the radiation boundary
condition), and it evolves with this solver; the Briareus side is a stable data type.
Neither package depends on the other: `AblationDrive` takes plain vectors, so any
hohlraum code (or a table) can drive it, and Briareus never loads MPI or
KernelAbstractions. A file format (HDF5) for non-Julia consumers can follow as a second
extension.

## The stage-0 model

The closure is the rocket model of indirect drive. With `T` the drive temperature in
heV, the ablation pressure and mass ablation rate are

```math
P_\text{abl} = P_0\,T^{3.3} \left(\frac{t}{1\,\text{ns}}\right)^{-0.09}\ \text{Mbar},
\qquad \dot m = 3\times10^5\,T^{3}\ \text{g cm}^{-2}\,\text{s}^{-1},
```

with carbon's subsonic exponents from constant-temperature slab simulations (Trickey,
Walsh and Pasley, HEDP 44, 100995 (2022)) and Lindl's mass ablation rate (Lindl et al.,
PoP 11, 339 (2004), Eq. 6-3), whose prefactor is the default `rate`. The NIF example
below calibrates `rate` to 2.85×10⁵, keeping the T³ scaling. The pressure prefactor is
not published; [`ablation_pressure_scale`](@ref)
sets it from the peak pressure, 145 Mbar for HYBRID-E, and the foot is then checked
against HYBRID-E's first-shock requirement, more than 12 Mbar at about 150 eV.

The angular structure enters through the local temperature,
``T_\text{eff}^4 = T_r^4\,(1 + \sum_l (a_l/a_0) P_l(\cos\theta))``, on a spherical
`(r, θ)` grid; a one-dimensional run takes the angle average.

**Where it acts.** An Eulerian code has no ablation surface. Along each radial line
the source finds the front as the mean radius of the ablator's outer density gradient,
weighted by `ρₐ (−∂ρₐ/∂r)⁺`, and acts on the ablator mass in a Gaussian layer of width
`layer` about it. Both weights are normalized by per-line sums, so the pressure and the
ablation rate the shell feels are exact on the grid, whatever the resolution. In the
driver-gas form both weights are one-sided: everything beyond the front counts with
weight one, the Gaussian applies only inside it (see below for why).

**How the pressure is applied.** Two forms are implemented:

- *Driver gas* (`driver = k`). The ablated mass is converted, in place, into species
  `k` (the outer gas: the hohlraum fill, then the blowoff), and species `k`'s pressure
  is relaxed to `P_abl` by an energy source. The driver gas is then a piston at the
  ablation pressure. This is the form that works: the shell's pressure follows the
  drive to within a few percent from the foot to the peak. Two details matter. (1) Any
  ablator beyond the front is converted first (weight `ρₐ` there). With a two-sided
  layer the partly converted ablator just outside the front gets no push, because the
  driver gas around it is held at uniform pressure. It stays behind as the shell
  accelerates: 22 % of the ablator at 6 ns in the NIF run. That layer then pulls the
  front marker off the shell, and from about 6 ns the shell stopped ablating. (2) The
  pressure relaxation, too, acts fully only beyond the front (`e^{−x²}` inside it), so
  driver gas converted inside the shell does not heat the shell toward `P_abl` behind
  the front.
- *Body force* (`driver = 0`). The force density `−P_abl w` on the layer, with the
  ablated mass removed from the problem. On cold ideal-gas ablators the accelerated
  layer lags the drive (the shell saw 18 Mbar while 117 Mbar was applied) and spikes
  into a one-cell sheet; the run fails at the rise of the NIF pulse. It is kept for
  experiments with other equations of state.

## Where the source hooks in

`compute_rhs!` assembles the flux divergence, the metric source terms and the boundary
corrections, and then calls `add_sources!` at the Runge–Kutta stage time, so
`add_source!(::AblationDrive, solver, dQ, Q, t)` sees the stage's conserved state, its
primitives (the driver-gas form reads the pressure) and the stage time, and adds to
`dQ`. Each evaluation makes three pointwise passes (front marker, layer weight, the
source itself) and three per-line sums.

- **Threads and GPUs.** Every pass is a `pointwise!` body over plain arrays and scalars,
  so it runs through the same KernelAbstractions path as the other sources. The
  per-line sums are `sum!` over the first dimension of device arrays. The drive table
  is interpolated on the host once per stage and passed as scalars. Not yet run on a
  device.
- **MPI.** The per-line sums are reduced over the radial sub-communicator, so the
  radius may be split across ranks (the reduction stages through a host copy).
- **Per-line form.** `SphericalMetric`, an unstretched `(r, θ)` grid and one patch.
  The binned form below takes the rest.

## Geometry: r–z and 3-D

NIF capsule work runs in three metrics: 1-D spherical, 2-D axisymmetric r–z
(cylindrical, the hohlraum axis as the grid axis) and 3-D Cartesian. On the last two
no grid line is radial, so the per-line search has nothing to sum along. The drive
therefore measures everything from the capsule centre (`center`, on the axis for r–z)
and sums over **angular bins** instead of lines: `bins = (nμ, nφ)` equal intervals of
`μ = cos θ` and `φ`, so the bins have equal solid angle. Each point's distance `R`,
its direction `(μ, φ)` and the components of `R̂` in the grid's local basis are
geometry, computed once per patch.

1. *Front.* The marker `g = ρₐ(−∂ρₐ/∂R)⁺`, with `∂/∂R = R̂·∇` from central
   differences along each grid direction (mirrored at an axis, origin, pole or
   symmetry plane, one-sided at any other block end), gives per bin
   `R_f = Σ g R dV/R² / Σ g dV/R²`.
2. *Normalization.* The layer weight `k` (the same one-sided or Gaussian weight as on
   lines, about `R_f` interpolated at the point) and the kernel
   `a = e^{−x²}/(√π δ)` give per bin `N = Σ a dV/R² / Σ k dV/R²`. Weighting by
   `dV/R²` makes each bin's sums `∫ … dr` per steradian, so `w = k N` is the per-line
   `k / ∫ k dr`: on a 1-D spherical grid the binned form (`geometry = :binned`)
   reproduces the per-line source to 1e-14, the calibration's convention included.
3. *Source.* `R_f` and `N` are interpolated bilinearly between bin centres at each
   point's direction, and the drive factor `T_eff⁴/T_r⁴` is evaluated there: the
   Legendre sum on r–z (the azimuthal average), every `m` of the Schmidt harmonics
   `bₗₘ` on a 3-D grid (`harmonics`, Briareus's `CapsuleDrive.ylm`, through the
   extension). The body force is `−P w R̂` in the grid's components.

**Refinement.** A patch sees only itself inside `add_source!`, and a tile of a
refined shell band holds a few cells of the front, so the bins cannot be a patch's
own. Before a level's right-hand side, at every stage, the solver calls
`prepare_source!(source, solver, level, states, comm)` (a hook in `_level_rhs!`,
entered by every rank of the level, where collectives are allowed), and the drive
sums its two passes over all the level's patches and reduces them over the level's
communicator; `release_source!` drops them after the level's right-hand side.
Every level carries its own consistent source. A coarse level under the band does
too, and its covered nodes are overwritten by the band's. `min_layer` widens the
layer to that many cells on a level whose spacing does not resolve it (the coarse
level under a 1 µm band). Shared tile planes count half in each tile's sums.
`ablation_band(drive; inside, outside, initial, metric)` is an `AMR(initial = …)`
predicate for a band about the root level's last `R_f(θ, φ)`; with
`regrid_interval` and `tile` it follows the shell as it converges.

**Device and MPI.** The passes are `pointwise!` bodies; the bin sums (a few thousand
bins) are taken on the host, so a device run copies two per-point arrays per pass.
Not yet run on a device. The single-patch path reduces over the patch's
communicator, the level path over the level's.

## What stage 0 can and cannot check

The rocket's outputs are its validation: the implosion velocity (HYBRID-E
380–400 km/s), the ablator mass remaining (5–6.6 %) and the coupled kinetic energy
(19–33 kJ). With ideal-gas, fully ionized materials and no conduction, shock timing,
adiabat, stagnation pressure and bang time are not predictions, and the energy the
driver gas receives is not the capsule's absorbed energy.

**The NIF drive (2026-10-24, `examples/briareus_capsule.jl`).** A static Briareus model
of an N221204-class hohlraum (11.24 × 6.4 mm gold, 3.1 mm entrance holes, the
delivered pulse by eye, the mock HYBRID-E pointing on 48 quads, no CBET; Briareus
785684d) gives a capsule `Tr` peaking at 297 eV. The calibration (`peak = 145`) sets
the pressure prefactor to 4.81 Mbar, so the foot gives 17 Mbar where `Tr` first
reaches 150 eV (more than 12, as required). The capsule is ideal-gas carbon (3.5 g/cc,
1.050–1.135 mm), DT ice and 0.3 mg/cc DT gas, with a 10 mg/cc He driver gas beyond it,
on a 1-D grid of 1400 points (1 µm), with a 5 µm layer and the driver-gas form.

*Robustness.* The first runs, with a 1 mg/cc driver, failed in the foot at about
0.5 ns unless `run!` was called in 50 ps pieces with retries. Held at `P_abl`, that
driver is a 2–7 keV ideal gas sitting against a 0.05 eV, 3.5 g/cc ablator. From 0.1 ns
on, the two or three cells of that contact have negative internal energy. The deficit
grows as the foot rises (1 to 5 Mbar between 0.3 and 0.5 ns) until a density goes
negative and the timestep collapses. A lower CFL did not help (0.2 failed too). A
10 mg/cc driver is ten times cooler and has a third of the sound speed. With it the
foot runs at CFL 0.4 with no retry (so does `ArtificialProperties(C_kappa = 0.1)`, which
smears the contact), and the negative-energy cells are then the ordinary ones at the
shock in the cold ablator. Its mass is about 1 % of the ablator's. The run is now one
`run!` call to past peak velocity. `StepControl(retries = 4)` recovers one or two
timestep collapses near 7.3 ns, when the first shock converges on the origin through
the 0.3 mg/cc gas, as in spherical Noh. The run takes 5–8 minutes on one thread, down
from 30 on four. The 2 µm grid now runs too.

*Calibration.* The mass ablation rate's prefactor `A` (`ṁ = A T³`) is set by the
ablator remaining at peak DT velocity. That is the ablator inside the front,
`r < R_f + 3·layer`. The rest of the unconverted ablator is a decompressed,
inward-moving tail behind the front (the third column). All rows are single `run!`
calls:

| `A` (g cm⁻² s⁻¹) | peak DT velocity | ablator remaining | unconverted ablator | KE, ablator + DT inside the front |
|---|---|---|---|---|
| 2.6×10⁵ | 320 km/s at 8.86 ns | 7.4 % | 27 % | 26.5 kJ |
| **2.85×10⁵** | **336 km/s at 8.76 ns** | **5.7 %** | 21 % | **25.8 kJ** |
| 3.0×10⁵ (Lindl) | 346 km/s at 8.70 ns | 4.8 % | 18 % | 25.1 kJ |
| 3.4×10⁵ | 375 km/s at 8.54 ns | 2.7 % | 9.7 % | 22.8 kJ |
| 3.8×10⁵ | 406 km/s at 8.36 ns | 0.9 % | 2.9 % | 19.2 kJ |
| HYBRID-E | 380–400 km/s | 5–6.6 % | | 19–33 kJ (imploding mass) |

The band 5–6.6 % gives `A` = 2.7–3.0×10⁵, so Lindl's Nova-era 3×10⁵ already sits at
its low-mass end. The T³ scaling is kept. Lindl's form ṁ ∝ I^{3/4} is T³, and the
rocket's exhaust velocity P/ṁ ∝ T^{0.5} (his Eq. 6-4) with carbon's P ∝ T^{3.3}
would give T^{2.8}. Most of the ablator goes in the 290–297 eV peak, so the exponent
hardly matters next to the prefactor. With both prefactors fixed, the implosion
velocity is the check. It is 328–344 km/s across the band, 10–14 % short of
380–400. The coupled energy, 25–26 kJ, is inside HYBRID-E's 19–33 kJ. Reaching
380–400 km/s takes `A` ≈ 3.5–3.75×10⁵, which leaves 1.5–2.5 % of the ablator. The
rocket cannot meet both targets with this pressure history, so the shortfall is in
P/ṁ at fixed mass, not in ṁ. Lindl's ideal rocket, 10⁷ T^{1/2} ln(m₀/m) cm/s, gives
395 km/s at this mass ratio (295 eV; 4.69 mg to 0.47 mg of ablator and DT). The hydrodynamic rocket loses 15 % to a drive that
falls from 8 ns, to convergence and to the fuel's pressure. Candidates, all beyond
stage 0: the ideal-gas carbon's adiabat (on the 2 µm grid the shell's peak density
falls from 18 g/cc at 5 ns to 5.6 g/cc at 7 ns), the 145 Mbar peak, and the gray
drive.

The calibration belongs to the drive. Briareus 3f25f29, with the late-time wall-loss
history, peaks at 285 eV. Recalibrated to 145 Mbar (prefactor 5.53 Mbar, 19.6 Mbar at
150 eV), the example with the same `A` gives 307 km/s at 9.12 ns, 8.2 % and 25.6 kJ.

The grid is not converged at this level. At 2 µm the same `A` gives 298 km/s and
7.3 %, against 336 km/s and 5.7 % at 1 µm, and the 0.5 µm run fails at the rise,
5.5 ns. Read the velocity as ±10 %. Before the one-sided weights, the
stranded layer counted as ablator remaining (19.6 % with Lindl's rate, 315 km/s, the
first result reported here). That number measured the model's defect, not the
ablation rate.

The body-force form failed at 5.1 ns (1 µm grid, before these changes).

**r–θ: the drive's P2 history on the shell.** The same drive was run with its Legendre
ratios through `l = 8` (P_abl ∝ (1 + Σ aₗ/a₀ Pₗ)^{3.3/4}) on an annulus 0.25–1.4 mm,
575 × 64 cells (2 µm radial, the full polar range, a slip wall at 0.25 mm, which only
matters near stagnation), and `A` = 2.85×10⁵. The run took about an hour on two
threads to 7.6 ns, where it was stopped. Legendre ratios of the front radius `R_f(θ)`
and of the DT-weighted shell radius, less their initial values (+0.05 %, the 64-cell
quadrature of a sphere):

| t (ns) | drive a₂/a₀ | front P2/P0 | DT shell P2/P0 | v (km/s) |
|---|---|---|---|---|
| 1.0 | +0.15 | −0.01 % | 0.00 % | 0 |
| 3.0 | +0.07 | −0.19 % | 0.00 % | 0 |
| 4.5 | +0.03 | −0.31 % | −0.11 % | 8 |
| 5.5 | −0.09 | −0.45 % | −0.40 % | 54 |
| 6.0 | −0.09 | −0.44 % | −0.44 % | 91 |
| 6.5 | −0.09 | −0.35 % | −0.33 % | 125 |
| 7.0 | −0.09 | −0.14 % | −0.10 % | 163 |
| 7.6 | −0.09 | +0.43 % | +0.50 % | 211 |

The pole-hot foot (P2 > 0 to 4.7 ns) flattens the shell at the poles (oblate,
P2 < 0). The pole-cold peak (−0.09 from 5 ns) stops the flattening at 5.5–6 ns and
reverses it, and the shell turns prolate near 7.1 ns. That is the swing: a shell shape
that changes sign during the implosion, with a lag of about 1 ns after the drive. It
is not yet Masse's sign inversion, which is a property of the response G(t, τ) to an
impulse at τ (pole-hot before ~4.5 ns gives an oblate hot spot, later a prolate one).
Here the drive's own P2 changes sign at 4.7 ns. Separating the two needs impulse runs
(the same drive with P2 only before, or only after, a time τ), carried to stagnation
on a grid that reaches the origin.

Stage 1 replaces the closure
with gray flux-limited diffusion and a Marshak boundary carrying `4F_inc(θ,t)/c`, with
table equations of state and flux-limited conduction.
