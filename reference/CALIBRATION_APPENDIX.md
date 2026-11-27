# Calibration appendix: the measurements

This file indexes the record behind [CALIBRATION.md](CALIBRATION.md): every sweep, ladder
and control run that produced the defaults there, including the explanations that were measured
and rejected and the results that moved nothing. Read CALIBRATION.md for the defaults and
for which setting to change; read the record before moving a default, before re-deriving a
rejected explanation, and when a number in a recommendation needs its provenance. The open
items are listed there too, one line each.

One section per instrument. Each section opens with the command that reproduces it, and each
table names the sweep and the background it ran under; a table whose background is not the
current default is labelled with the setting it was taken under and kept as the record of
that setting. A quantity is recorded once, at its current value: where a number moved, the
section that moved it says so in one sentence and the older figure is gone.

The sections are kept in topic files under `appendix/`, listed below with the instrument
that reproduces each.

## Contents

### [Shocks and the artificial properties](appendix/shocks.md)

- [The shock battery](appendix/shocks.md#the-shock-battery)
  (`bench/artcal.jl`, `test/cases.jl`)
- [The CFL restriction and the symmetry cell](appendix/shocks.md#the-cfl-restriction-and-the-symmetry-cell)
  (`bench/archive/nohprobe.jl`)
- [Negative internal energy ahead of a shock](appendix/shocks.md#negative-internal-energy-ahead-of-a-shock)
  (`bench/archive/shockfoot.jl`)
- [Directional bulk viscosity](appendix/shocks.md#directional-bulk-viscosity)
  (`bench/archive/anisotropic.jl`)
- [The aligned Noh transverse mode](appendix/shocks.md#the-aligned-noh-transverse-mode)
  (`bench/archive/noh_transverse.jl`)
- [False activation on smooth fields](appendix/shocks.md#false-activation-on-smooth-fields)
  (`bench/archive/falseactivation.jl`)
- [The ablation rate of the Briareus-driven capsule](appendix/shocks.md#the-ablation-rate-of-the-briareus-driven-capsule)
  (`examples/briareus_capsule.jl`)

### [Taylor-Green and the filter](appendix/filter.md)

- [Taylor-Green](appendix/filter.md#taylor-green)
  (`bench/tgv_energy.jl`, `bench/archive/tgv_spectrum.jl`)
- [The filter's dissipation](appendix/filter.md#the-filters-dissipation)
  (`bench/archive/filterrate.jl`)
- [Filtering on non-uniform volumes](appendix/filter.md#filtering-on-non-uniform-volumes)
  (`bench/archive/filter_conservation.jl`)

### [Boundaries, closures and folds](appendix/walls.md)

- [The smooth-evolution accuracy matrix](appendix/walls.md#the-smooth-evolution-accuracy-matrix)
  (`bench/boundaryorder.jl`, `bench/archive/temporalorder.jl`, `test/convergence.jl`)
- [The filter's wall rows](appendix/walls.md#the-filters-wall-rows)
  (`bench/archive/wallfilter.jl`)
- [Wall closures in production](appendix/walls.md#wall-closures-in-production)
  (`bench/archive/wallclosure.jl`)
- [The wall flux contracts](appendix/walls.md#the-wall-flux-contracts)
  (`test/wall_flux_tests.jl`)
- [Constant annihilation](appendix/walls.md#constant-annihilation)
  (`bench/archive/constantfloor.jl`)
- [Closure certificates](appendix/walls.md#closure-certificates)
  (`bench/archive/closurecertify.jl`, `bench/archive/neutralsearch8.jl`,
  `bench/archive/neutralsearch10.jl`)
- [The fifth-order closure search](appendix/walls.md#the-fifth-order-closure-search)
  (`bench/archive/closuresearch.jl`, `bench/archive/closuredamping.jl`)
- [The sensor operators at walls](appendix/walls.md#the-sensor-operators-at-walls)
  (`bench/archive/sensorwall.jl`)
- [The face-centred symmetry plane](appendix/walls.md#the-face-centred-symmetry-plane)
  (`bench/archive/wallclosure.jl`, `test/mpi_tests.jl`)
- [The inflow transverse terms](appendix/walls.md#the-inflow-transverse-terms)
  (`bench/archive/nscbcinflow.jl`)
- [Fold order and geometry limits](appendix/walls.md#fold-order-and-geometry-limits)
  (`bench/archive/foldorder.jl`, `bench/archive/axismass.jl`)
- [Grid-scale growth at the r-z axis](appendix/walls.md#grid-scale-growth-at-the-r-z-axis)
  (`bench/archive/axisspectrum.jl`, `bench/archive/axisrunaway.jl`)
- [The outflow face on a radial grid](appendix/walls.md#the-outflow-face-on-a-radial-grid)
  (`bench/archive/sphereoutflow.jl`, `test/sphere_dipole.jl`)
- [The slip wall of the r-z annulus](appendix/walls.md#the-slip-wall-of-the-r-z-annulus)
  (`bench/archive/annuluswall.jl`, `test/convergence.jl`)

### [Refinement](appendix/amr.md)

- [AMR](appendix/amr.md#amr)
  (`bench/archive/amr_transfer.jl`, `bench/archive/leveltransfer.jl`,
  `test/level_tests.jl`, `bench/amrwin.jl`, `bench/archive/movinglevel.jl`,
  `bench/archive/levelpattern.jl`)

### [Transport, species and interfaces](appendix/transport.md)

- [Temperature-dependent transport](appendix/transport.md#temperature-dependent-transport)
  (`test/transport_tests.jl`, `test/transport_integration_tests.jl`)
- [Neutral polynomial transport](appendix/transport.md#neutral-polynomial-transport)
- [The bulk species channel in three dimensions](appendix/transport.md#the-bulk-species-channel-in-three-dimensions)
  (`bench/archive/bulkchannel.jl`, `bench/archive/bulkentropy.jl`)
- [The contact with a temperature jump](appendix/transport.md#the-contact-with-a-temperature-jump)
  (`bench/archive/thermalcontact.jl`, `bench/archive/tubebudget.jl`)
- [The species validity band](appendix/transport.md#the-species-validity-band)
  (`bench/archive/speciesband.jl`)
- [The gas-gas acoustic interface](appendix/transport.md#the-gas-gas-acoustic-interface)
  (`bench/archive/interfaceacoustics.jl`, `test/validation.jl`)
- [The interface sharpening flux](appendix/transport.md#the-interface-sharpening-flux)
  (`bench/archive/sharpening.jl`, `test/sharpening_tests.jl`)

### [Implicit and incompressible solves](appendix/implicit.md)

- [Stiff diffusion](appendix/implicit.md#stiff-diffusion)
  (`bench/archive/stiffdiffusion.jl`, `bench/archive/staggeredconduction.jl`,
  `bench/archive/staggeredfolds.jl`, `bench/archive/staggeredclosure.jl`,
  `bench/archive/implicitstage.jl`)
- [The incompressible prototype](appendix/implicit.md#the-incompressible-prototype)
  (`bench/archive/incompressible_checks.jl`, `bench/archive/incompressible_tgv.jl`)

### [Cost and output](appendix/cost.md)

- [Operator and step cost](appendix/cost.md#operator-and-step-cost)
  (`bench/derivcost.jl`, `bench/phases.jl`, `bench/archive/reducedsolve.jl`,
  `bench/archive/nasa9_inversion.jl`)
- [The shared-file write](appendix/cost.md#the-shared-file-write)
  (`bench/archive/hdf5xfer.jl`, `test/hdf5_tests.jl`)
- [Azimuthal mode truncation](appendix/cost.md#azimuthal-mode-truncation)
  (`polar_truncation`)
