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
PoP 11, 339 (2004), Eq. 6-3). The prefactor is not published; [`ablation_pressure_scale`](@ref)
sets it from the peak pressure, 145 Mbar for HYBRID-E, and the foot is then checked
against HYBRID-E's first-shock requirement, more than 12 Mbar at about 150 eV.

The angular structure enters through the local temperature,
``T_\text{eff}^4 = T_r^4\,(1 + \sum_l (a_l/a_0) P_l(\cos\theta))``, on a spherical
`(r, θ)` grid; a one-dimensional run takes the angle average.

**Where it acts.** An Eulerian code has no ablation surface. Along each radial line
the source finds the front as the mean radius of the ablator's outer density gradient,
weighted by `ρₐ (−∂ρₐ/∂r)⁺`, and acts on the ablator mass in a Gaussian layer of width
`layer` about it. Both weights are normalized by per-line sums, so the pressure and the
ablation rate the shell feels are exact on the grid, whatever the resolution.

**How the pressure is applied.** Two forms are implemented:

- *Driver gas* (`driver = k`). The ablated mass is converted, in place, into species
  `k` (the outer gas: the hohlraum fill, then the blowoff), and species `k`'s pressure
  is relaxed to `P_abl` by an energy source. The driver gas is then a piston at the
  ablation pressure. This is the form that works: the shell's pressure follows the
  drive to within a few percent from the foot to the peak.
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
- **Limits.** `SphericalMetric`, an unstretched `(r, θ)` grid and one patch. The
  axisymmetric `r–z` grid with refinement, where the drive must be mapped from the
  sphere, is the next geometric step.

## What stage 0 can and cannot check

The rocket's outputs are its validation: the implosion velocity (HYBRID-E
380–400 km/s), the ablator mass remaining (5–6.6 %) and the coupled kinetic energy
(19–33 kJ). With ideal-gas, fully ionized materials and no conduction, shock timing,
adiabat, stagnation pressure and bang time are not predictions, and the energy the
driver gas receives is not the capsule's absorbed energy. Stage 1 replaces the closure
with gray flux-limited diffusion and a Marshak boundary carrying `4F_inc(θ,t)/c`, with
table equations of state and flux-limited conduction.
