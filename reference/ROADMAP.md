# CompactLES task roadmap

Open work for compressible, variable-density mixing and implosion, grouped by
track and ordered within each track. Closed items are listed at the end, one
line each with the commit that delivered them; the measurements behind them are
in [CALIBRATION_APPENDIX.md](CALIBRATION_APPENDIX.md) and the methods in
[DESIGN.md](DESIGN.md).

## How to use this plan

- Unchecked boxes are open deliverables. IDs are stable and name
  dependencies, so an item keeps its ID when it moves between sections.
- [Order of work](#order-of-work) gives the sequence across tracks; within a
  track the items are listed in the order they should be taken.
- A measurement task is complete when its result and the resulting decision,
  including a decision to retain the current method, are recorded: one line
  under [Closed](#closed) naming the commit, its table in
  [CALIBRATION_APPENDIX.md](CALIBRATION_APPENDIX.md), and its default in
  [CALIBRATION.md](CALIBRATION.md) if a default moved.
- Close implementation tasks with the stated regression checks and the applicable
  repository validation gate. A numerical change requires an explained baseline
  update, not a relaxed test tolerance.
- Historical references to model debts 1/2/3 correspond to N1 / R3+N3 / V1;
  former Phase 2 items correspond to H1–H8, and G4b to S4.

## Order of work

The first four are independent and may proceed together, subject to one heavy
run at a time.

1. **N23** refines the r-z capsule, the production configuration, which setup
   rejects today. Its first stage also delivers the physical-face half of A10.
2. **N4** settles `C_mu` before any external comparison: the filter fits
   behind the α = 0.47 default were made at `C_mu = 0.002`, and V1 compares
   the defaults.
3. **V1** runs the Pyranda comparisons and one Richtmyer–Meshkov experiment.
   It supplies N18's remaining reference and V2's use case.
4. **H2** integrates the existing molecular conduction implicitly with the ARK
   pair. H1's operator and one-patch solve are delivered, so the IMEX contract
   is verified on a known equation before any new physics depends on it.
5. **A7** draws the material boundary and the per-component field declaration
   before H3 adds the first new evolved field.
6. **H3**, then **H4**: two temperatures with an ionization closure and the
   electron–ion exchange, then electron and ion conduction through the implicit
   stage. H5's table reader may start at any point.
7. **N22** sharpens the species channel's interfaces, and becomes urgent when a
   case at density ratio 100 or more is in production.
8. Cluster campaigns run as allocation allows: S12's rzhound probe, S15's
   later stages, then S1 and S2 on rzadams.

The remaining items wait for a named target: H6, H7 and H9 for a radiation-,
laser- or burn-dominated case; H4a, H5a and H5b for cold DT and the ablator;
M1–M3 for an immiscible interface; S10 for a body that is not a coordinate
surface; H8 for a magnetized target.

## Numerics and calibration

- [ ] **N4 — Decide the artificial shear viscosity.** Scored on the history
  misfit, Taylor–Green at Re = 1600 fits best at `C_mu = 0` at 64³ and at 128³,
  under both the default and the near-off filter, and at 128³ a fifth of
  the physical dissipation lies off the grid
  ([Taylor-Green](CALIBRATION_APPENDIX.md#taylor-green); commits `c3f4f82`,
  `7e37a04`). One-dimensional shocks cannot determine the shear channel, so no
  case in the repository supports a positive value. Cook's 0.002 was never
  fitted: its evidence is decaying turbulence at Re_λ ≈ 720 on 192³ (Kang et
  al.), under a 2/3-rule dealiasing or the sharp `pyranda_filter()`, not the
  default filter
  ([where 0.002 comes from](CALIBRATION_APPENDIX.md#c_mu-the-shear-viscosity)).
  Either set the default to zero once the one-dimensional battery and a
  shocked three-dimensional case clear it, or rerun that decaying-turbulence
  case under the default `compact_filter` and `pyranda_filter()`, which tells
  whether a positive `C_mu` belongs to the method or to Cook's filter; a
  low-pressure vortex core is a separate, stability case. The filter fits
  behind the α = 0.47 default were made at `C_mu = 0.002`, so a change to
  `C_mu` reopens them. Retain `C_beta = 1` unless new evidence overturns its
  completed refit; record error and dissipation attribution.

- [ ] **N18 — Finish qualifying the species channel's transport consistency.**
  The three channels are compared on the one-dimensional battery and the
  shocked He/CO2 tube under one sensor, one operator set and one step
  ([the channel](CALIBRATION_APPENDIX.md#the-partial-density-species-channel)),
  and the default's uniform (u, p, T) invariance and entropy inequality are
  derived and measured ([DESIGN.md](DESIGN.md#the-species-channel); commit
  `64b8e14`). The invariance argument is for a uniform temperature. At a
  temperature jump each channel's drift is its own continuous model's volume
  source, the discrete flux identities hold as products, and no split form is
  needed ([the thermal contact](CALIBRATION_APPENDIX.md#the-contact-with-a-temperature-jump),
  commit `5281eac`). On the shocked tube the energy change is attributed per
  term with a closed budget, and the three layer widths are grid-set while the
  channels differ in the jumps across them (commit `5d529cb`). Remaining: a
  grid-converged reference, or the V1 Pyranda run, for the He/CO2 roll-up,
  which no measurement yet decides between the bulk and the partial-density
  form.
  **Depends on:** V1 for the reference, or a grid-converged run of this code.
  **Gate:** the appendix tables, the default retained or moved with its
  [CALIBRATION.md](CALIBRATION.md) line, and `test/validation.jl` baselines
  explained.

- [ ] **N22 — Add an interface sharpening flux to the species channel.**
  The artificial species diffusivity holds a shocked interface free of ringing
  only with a core of about three cells, and it leaves a tail of the heavy gas
  on the light side that the mass-fraction width reads as 10 cells at density
  ratio 100
  ([interface width](CALIBRATION_APPENDIX.md#interface-width-against-brill-olson-and-bokman)).
  Brill, Olson and Bokman (J. Comput. Phys. 542, 114366, 2025; eq. 68 of
  arXiv:2503.12680v2) pair the same partial-density channel with a
  sharpening flux whose equilibrium profile is V = 1/(1 + e^(−x/ε)), ε = Δ by
  default, which removes that tail. Add it to the species channel on the
  collocated compact operators, in divergence form, with its consistency
  terms on momentum and energy derived as the channel's are, and localized so
  that it acts only at a material interface and not on a smooth composition
  gradient. Define ε, the strength Γ and the step restriction; state which
  fraction it sharpens (volume or mass) and why; and keep single-species runs
  bit-identical. Pyranda has no implementation (upstream `b4e0afc`, every
  branch), so the form comes from eqs. 68–71: J_i = −ρ_i Γ(ε∇V_i −
  Σ_j V_i V_j n̂_ij), added to the channel flux before its consistency terms,
  with V the mole fraction for ideal gases; its discretization, the
  regularization of n̂ away from interfaces and the step limit are questions
  for the authors.
  M2 is the phase-field counterpart for immiscible materials on the staggered
  operators; share the analysis but not the requirement that M1 come first.
  **Depends on:** nothing. **Gate:** the shocked interface at
  density ratios 100 and 1000 against the unsharpened channel (width in
  volume and mass fraction, TV−1, worst Y), the smooth-slab deposit of
  `bench/falseactivation.jl`, the He/CO2 tube, the N20 budgets, and
  `test/validation.jl` baselines explained.

## Validation and verification

- [ ] **V1 — Complete independent solver and experiment comparisons.**
  Run CompactLES against Pyranda on Re = 1600 Taylor–Green and one
  Richtmyer–Meshkov shock tube, comparing dissipation histories, spectra and
  mix widths; `pyranda_filter()` and the He/CO2 comparison deck (commit
  `9e640ab`) and the record of Pyranda's numerics (commit `29364ad`) are in
  place. Select one published RM experiment with documented initial and
  boundary conditions and obtain the missing specifications from its authors.
  **Depends on:** N4 for the defaults compared.
  **Deliver:** reproducible inputs, reference provenance, uncertainty/error measures,
  and a docs validation section. Keep analytic validation, external data, and
  self-generated regression profiles explicitly distinguished.

- [ ] **V2 — Qualify the turbulent inflow on a V1 experiment.**
  `TurbulentInflow`, a seeded random-Fourier-mode target with prescribed
  Reynolds stresses and integral scale, reproducible across decomposition and
  restart, landed in commit `ebbe4d6`. Remaining: a V1 experiment use case,
  and the NSCBC relaxation rates (`eta_u`, `eta_T`) that admit the
  fluctuation without damping it.
  **Depends on:** V1.

- [ ] **V3 — Close gaps in automated verification.**
  In place: the temporal-order studies (commit `d8ea472`), with the
  once-per-step level restriction retained (commits `9370da5`, `e7dc4ca`;
  [temporal order](CALIBRATION_APPENDIX.md#temporal-order)); the R1–R4
  regressions and the skip accounting of the serial suite (commit `26ae691`);
  the Makie checks in CI's documentation job; and the scheduled validation
  battery (`.github/workflows/validation.yml`). Remaining: parallel-HDF5
  execution in CI where the required stack exists (verified locally under
  S6); each instrument's current transcript under `bench/results/<script>.txt`,
  so that an appendix table is a quoted output and never a retyped one, each
  landing with that script's next run rather than assembled from the tables it
  would replace; and a `bench/` runner taking medians over repeated processes,
  which the performance gates of A7 and S3 need to resolve differences under
  the run-to-run spread.
  **Gate:** KA-on-CPU equality does not substitute for hardware-GPU tests
  under S1.

## Refinement for the production geometry

Node-centered patch and level coupling is interpolation and injection with
compact interface closures, not a conservative flux reconciliation; under the
N10 budgets no surface-flux correction is needed. The measured interface orders
are in the appendix's smooth-evolution
[accuracy matrix](CALIBRATION_APPENDIX.md#the-smooth-evolution-accuracy-matrix);
the designs and the fallback analysis are in [AMR_GPU.md](AMR_GPU.md). The
instruments for a new interface or boundary closure are the N6 matrix
(`bench/boundaryorder.jl`, gated in `test/convergence.jl`),
`bench/wallfilter.jl`, `bench/wallclosure.jl` and `bench/leveltransfer.jl`; a
candidate is qualified on the N10 budgets and the N11 shock crossings before
promotion.

- [ ] **N23 — Refine axisymmetric r-z runs (high priority).** An ICF capsule
  in r-z with a symmetry plane at z = 0 is the production configuration, and
  setup rejects both halves of it: `AMR: requires CartesianMetric`, and
  `AMR: cannot refine a run with a SymmetryPlaneBC`. A Cartesian mock-up of
  the capsule shows the third gap: every level stays `max(n_halo, 4)` parent
  nodes off every root boundary, so the shell where it meets the symmetry
  plane (and, in r-z, the axis, where the hot spot forms) keeps root
  resolution. Stages, in order:
  1. A tile face on a root boundary: a refined tile carries the boundary's own
     condition on that face (wall, `SymmetryPlaneBC`, NSCBC) instead of a
     parent-fed shell, with the nesting margin applied to parent-fed faces
     only; the restriction margin and the covered mask follow. This extends
     N6l's face-centred fold to refined runs.
  2. `CylindricalMetric` with θ collapsed on refined levels: the per-tile
     metric (`inv_r`, face areas, the discrete GCL) at the fine spacing, and
     the composite quadrature and conservation budgets weighted by r.
  3. A tile on the axis: `AxisBC` on θ-collapsed r-z is a parity mirror, not
     the antipodal butterfly constraint 4 of `AMR_GPU.md` forbids refining
     across, so a tile face at r = 0 takes the mirror as its boundary
     condition. The half-cell offset carries over at ratio 3: the coarse node
     at h/2 puts the fine nodes nearest the axis at ±h/6. This is the one
     fold at which refinement is designed; the prohibition of S8 stands at
     the others.
  **Gate:** smooth-solution orders across a coarse-fine face in r-z and
  across a tile face on the axis and on the symmetry plane
  (`test/convergence.jl`), composite conservation with r weights, a
  converging shock refined onto the axis against a uniform-fine run, and the
  tutorials whose walls are true symmetry planes switched from `SlipWallBC`
  to `SymmetryPlaneBC`.

- [ ] **S15 — Run a refined hierarchy across many nodes.**
  The target is a mixing layer or a plane shock resolved by a tiled level, or
  a nested pair of them, on thousands of ranks, with each rank's memory,
  traffic and setup cost following the tiles it holds rather than the level's
  tile count, which a weak-scaled run grows with the ranks. Stages, in order:
  1. Done with `LevelCoupling`: the level coupling moves point to point
     (`LevelCoupling`), a transfer's chains and boxes exist on its tile's ranks
     only, interface tags number the messages of one rank pair, and owner
     sizing starts at the count a tile admits; `bench/amr_scaling.jl` measures
     the weak scaling ([table](CALIBRATION_APPENDIX.md#benchamr_scalingjl-weak-scaling-of-the-level-coupling)).
  2. A nested hierarchy at scale, which needs S8's deep-regrid items
     (point-to-point migration of a moved survivor in place of the replicated
     gather over the run, rebalancing, the device backend).
  3. Decide the one-box path at scale. A tile's chain runs over its whole box
     on each of up to `n_cons` of its ranks, so a one-box level (`tile = 0`)
     over many ranks holds a box-sized chain per rank. Either partition the
     chain by fine block or have `AMR` choose a tile edge when the run spans
     many ranks; the second changes a default and needs a maintainer decision.
  4. Tagging at scale: `_tag_tiles` Allreduces a dense flag per lattice cell
     at every check; replace it with a sparse reduction if a cluster profile
     shows it.
  5. Qualify on rzhound with the system MPI: `bench/amr_scaling.jl dim=3`
     weak-scaled to at least a thousand ranks, then a production-shaped 3-D
     case (a shock through a perturbed heavy-gas interface, tiles tagged by
     the mass-fraction gradient and the sensor) against a uniform-fine
     reference at a size both fit, with restart on another rank count and a
     refined HDF5 dump (a spatial collection per patch with blanking, which
     the HDF5 `FieldWriter` rejects today).
  **Gate:** per-rank level memory and coupling wall flat in the weak-scaling
  table at the largest count available, the MPI suite at 2, 4 and 8 ranks,
  and bitwise agreement of every stage that only moves data.

- [ ] **A10 — Let a refined level reach every domain boundary.**
  Every level now stays `max(n_halo, 4)` parent nodes inside its parent, so a
  feature at a wall, an inflow face, or across a periodic seam stays coarse
  (a warning says so for a box and for shapes; the tiled path drops the
  margin band silently). N23 stage 1 delivers the physical faces. Remaining:
  a periodic image of the parent data in the level coupling and wrapped tile
  regions, so a level crosses a periodic seam; the warning on the tiled path;
  and the target case, the 3-D vortex ring fired from a tube's top face into
  an air/SF6 interface and then shocked through it (the axisymmetric form is
  `examples/vortex_ring_shock.jl`). The ring forms at the injector face, so
  before N23 stage 1 it can only be formed inside the domain by a body force;
  in that workaround one refined box follows the ring well, and lattice tiles
  of small edge cost many times more per step than the box for a compact
  feature.
  **Depends on:** N23 stage 1.
  **Gate:** a shock and an interface followed into a wall and through a
  periodic seam, with the conservation and interface-reflection tests of a
  uniformly fine run; the vortex-ring case with the ring formed at the face.

- [ ] **S8 — Extend deep regridding and multiblock geometry when a target needs them.**
  Tiled regridding below level 1 landed with A12; add rebalancing,
  point-to-point migration and the device backend to it, which S15 stage 2
  and a nested capsule under N23 need. For geometric multiblock use, extend
  beyond the current slab layout with explicit adjacency and compatible
  geometry, and give a same-level patched run a checkpoint.
  **Depends on:** a concrete case.
  **Gate:** interface accuracy, conservation budgets, restart, and distributed
  consistency. Refinement across a fold stays forbidden pending a separate
  design; N23 stage 3 is that design for the θ-collapsed axis only.

- [ ] **A13 — Refinement on the remaining metrics and layouts.**
  N23 takes the cylindrical metric and the symmetry plane. Refinement still
  rejects the spherical metric and stretched grids, and a same-level
  `patch_grid` run rejects folds, stretching along the patched dimension, a
  patch decomposition across a `SymmetryPlaneBC`, the `:d8` detector on every
  sensor, and the pentadiagonal filters, which refinement admits (commit
  `32b9de0`). Each extension needs its transfer closures and fold-aware
  gathers; order them by use.
  **Gate:** per extension, the refined-versus-uniform convergence rows of
  `test/convergence.jl`.

## HED physics

H1 is the principal infrastructure dependency for stiff diffusion. These are
separate capabilities with independent verification gates, not a claim that new
equation layouts alone make the current RHS a general multiphysics solver.
Ownership and coupling follow [the material interface design](DESIGN.md#material-and-physics-interfaces);
DT is the first cold-material path, with C/CH/CD coverage tracked separately in
H5b. Every integration preserves A7's ideal analytic execution contract.

LANL's [Riot](https://lanl.github.io/riot/main/index.html), a finite-volume
radiation-hydrodynamics code on Parthenon and Kokkos with singularity-eos and
singularity-opac, carries most of this stack, and its dependency order is the
one adopted here: an electron EOS and a mean ionization per material, then the
electron–ion exchange, then implicit electron and ion conduction, then
radiation coupled to the electrons, and the laser after the ionization it
absorbs on. Its regression problems supply several of the gates below. Three
of its choices are not taken: its stiff packages are operator-split in a fixed
order after the explicit stages, where H2 keeps them in one ARK residual; its
electron conduction has no flux limiter and its conduction solve no multigrid
preconditioner; and its P1 relaxation factor stands in for a radiation flux
limiter.

- [ ] **H1 — Complete the implicit diffusion infrastructure.**
  Follow [IMPLICIT.md](IMPLICIT.md): the conservative staggered compact
  operator, solved matrix-free by a Krylov method preconditioned with a
  multigrid cycle on the second-order operator. Delivered: the design (commit
  `99f0197`); stage 1, the staggered operator, periodic, walled, folded and on
  every supported metric (commits `52a58b8`, `32842fc`), accurate but not
  exactly symmetric at the cylindrical axis and the spherical poles, where no
  symmetric closure keeps fourth order (commit `b80d0fb`); and stage 2, the
  one-patch implicit stage solved by conjugate gradients, or GMRES at the axis
  and poles, under a line-relaxation V-cycle, which holds 13–23 iterations
  flat in grid and stiffness except on a spherical grid with an origin (commit
  `8819db1`). Remaining, each when a case needs it: a smoother for that grid
  (plane relaxation is the candidate), agglomeration of the coarsest level at
  large rank counts, and stage 4, the implicit stage on refined levels and on
  device storage, designed after H2 has measured costs. Keep operators and
  communication in core numerics with optional workspace allocated only when used.
  **Gate:** manufactured constant/variable-coefficient heat conduction in every
  supported metric, distributed residual/convergence studies, and freestream
  preservation. Variable coefficients require a policy for rebuilding the
  preconditioner's second-order operator.

- [ ] **H2 — Integrate the existing conduction with the IMEX pair.**
  ARK4(3)6L[2]SA is the chosen pair and RKL2 super-time-stepping is rejected
  ([IMPLICIT.md](IMPLICIT.md#the-integrator)). The first implicit component is
  the present 1T molecular conduction (IMPLICIT.md stage 3), so the integrator
  is verified before H3 exists. Define the workspace contract; accept
  component contributions to a joint residual and a consistent linearization;
  define coefficient refresh, nonlinear trial invalidation and collective retry.
  Independent physics components must not force sequential split updates.
  Size the implicit half's step by accuracy, beside the acoustic limit of the
  explicit half: the pair's embedded error estimate or Riot's target
  fractional change of temperature per step, chosen by measurement
  ([IMPLICIT.md](IMPLICIT.md#open-questions)).
  **Depends on:** H1 stages 1 and 2, delivered.
  **Gate:** temporal order on the smooth-evolution cases, a Gaussian
  conduction pulse at steps far beyond the explicit diffusive limit (Riot's
  `conduction_analytic`), stiff stability at large R, the acoustic CFL limit
  of the explicit half, and a failed implicit solve recovered by collective
  rollback.

- [ ] **A7 — Implement material interfaces with an explicit analytic fast path.**
  Follow [the interface design](DESIGN.md#material-and-physics-interfaces):
  separate local material queries, solver field adapters, and flux closures;
  define composition/energy metadata, typed results/status, and setup checks.
  The typed status of `state_admissibility` and `mixture_temperature_status`
  (commit `4be4ac0`) and the public EOS hooks (commit `3a80e1b`) are its first
  pieces. Preserve the existing ideal/stiffened analytic recovery and scalar
  transport methods, public constructors, and readable arithmetic. Introduce
  an internal `MaterialModels` boundary for standalone evaluators/readers,
  removing EOS field assumptions from adapters, as a closed set of concrete
  isbits models mirrored onto device storage, the form `Nasa9Mixture` already
  takes (commit `b78acd4`). Let a physics component declare its fields
  (evolved or derived, exchanged, checkpointed, written), so that H3's
  electron energy and H6's radiation energy reach the halo exchange, the
  checkpoint record and output from the declaration rather than through edits
  to the `n_cons` layout; the `StateDescriptor` metadata of Parthenon, on which
  Riot's packages register their fields, is the precedent. No package
  extraction or HED state allocation is required for this step; numerical
  formulas and default policies stay unchanged.
  **Sequence:** inventory current hooks and exercise independent queries with
  existing ideal, NASA-9 and transport models; qualify a tabulated model with
  H5 before stabilizing the rich interface. H1 and offline material/transport
  work may proceed independently; H3 uses the relevant A7 contracts as they
  mature.
  **Gate:** existing input decks and numerical baselines, independent query and
  adapter agreement, inference/allocations, and unchanged ideal workspace,
  gradient, launch and collective counts. Compare before/after full-step costs
  on single/multispecies ideal cases, including constant and analytic transport,
  plus NASA-9 and stiffened-gas controls; run `bench/jetcheck.jl` and
  `bench/audit.jl` and relevant CPU/MPI/device gates. Use paired measurements and
  repeated processes on the same hardware; no reproducible ideal-fluid slowdown
  or added per-point allocation is accepted to accommodate the rich path.
  Record measurements in the appendix and unavailable hardware explicitly.

- [ ] **H3 — Add two-temperature energy evolution with an ionization closure.**
  Extend the equation, flux and recovery interfaces for `T_ion` and `T_ele`:
  electron pressure, a mean ionization Z̄ per material (full ionization first,
  then the Thomas–Fermi fit of More that Riot uses), and an electron EOS per
  material (ideal electrons first), from which the electron density follows.
  Specify independent energy variables, binding-energy references, pressure
  work and the exchange mapping; prevent double counting in total-energy and
  subsystem equations. Declare where the artificial viscous heating of a shock
  is deposited, and whether the electrons carry an energy with its pdV work or
  an entropy; the choice sets the ion–electron partition at a shock. The
  electron–ion exchange (Landau–Spitzer with a declared Coulomb logarithm) is
  a pointwise implicit component of H2 and needs no global solve. Radiation
  enters with H6 as an energy field, not as a third material temperature;
  `T_rad` names the temperature derived from that energy. Declare gradient,
  boundary, wave-speed and regularization requirements at setup.
  **Depends on:** A7's contracts and H2.
  **Gate:** zero-dimensional electron–ion relaxation against its exact
  solution (Riot's `ei_relax`), total-energy conservation and the equilibrium
  limit, two-temperature linear acoustic modes, a two-temperature Sod tube
  under each electron-energy form, and the Shafranov two-temperature shock
  structure once H4 supplies electron conduction.

- [ ] **H4 — Add electron and ion conduction and ion viscosity.**
  Implement flux-limited Spitzer–Härm electron conduction, with its
  Z̄-dependent coefficient, and Braginskii ion conduction and viscosity through
  the implicit stage. Treat this as the hot-limit closure; wider material
  conduction consumes H5a/H5b state and validated coefficients without imposing
  their cost on scalar transport. The limiter makes the coefficient depend on
  the gradient, so the Picard iteration may stall, and JFNK with the limited
  flux is the fallback ([IMPLICIT.md](IMPLICIT.md#open-questions)).
  **Depends on:** H1–H3 and the dispatchable transport interface.
  **Gate:** the self-similar nonlinear heat wave of Zel'dovich and Raizer, a
  Coggeshall problem with hydrodynamics, limiter behavior at a steep front,
  H3's Shafranov shock, and coupled energy budgets.

- [ ] **H5 — Add tabulated EOS support.**
  Implement IONMIX reading and thermodynamically consistent interpolation/inversion.
  For SESAME, read the SP5 HDF5 files of the singularity-eos toolchain, which
  Riot reads, into isbits interpolators of this package: singularity-eos has
  Fortran and Python bindings but none for Julia, and a C shim over it would
  run on the host only. SESAME access remains subject to data licensing.
  Exercise A7's rich queries and shared interpolation/derivative results with
  a bounded table model; qualify forward/recovery adapters before stabilizing
  that interface. Declare frozen/equilibrium derivatives and supported
  temperature partitions; a 1T table alone does not supply a 2T EOS or the
  populations required by transport.
  **Depends on:** the relevant A7 contracts; reader/data work may precede them.
  **Gate:** table-node/interpolation checks, inverse consistency, derivatives,
  phase/domain handling, and an EOS-specific artificial-conductivity scale.

- [ ] **H6 — Add radiation diffusion, gray before multigroup.**
  Evolve a radiation energy per group, with `T_rad` derived from it, coupled to
  the electron energy through H2/H3's residual and energy contract, and define
  the opacity and group interfaces; opacity consumes the same material and
  population state as transport. Use flux-limited diffusion with a
  Levermore–Pomraning or Larsen limiter on the staggered operator, and solve
  the groups together in one Newton iteration, as Riot's multigroup solve
  does, rather than group by group. Riot's P1 form (the group energy at the
  cells, the flux at the faces, a relaxation factor in place of a limiter)
  maps onto the staggered operator and is the alternative to compare.
  **Depends on:** H1–H3 and suitable EOS/opacity data.
  **Gate:** Su–Olson nonequilibrium diffusion, a gray Marshak wave, the
  Lowrie–Edwards steady radiative shock coupled to the hydrodynamics, group
  convergence, positivity policy, and matter–radiation energy conservation.

- [ ] **H7 — Add laser deposition for a specified experiment.**
  Implement geometric-optics rays with refraction from the electron-density
  gradient, inverse-bremsstrahlung absorption into the electron energy,
  rank-to-rank ray transfer, and deposition through the source interface;
  Riot's laser package has this form.
  **Depends on:** H3's electron density and temperature and nothing beyond
  them, so it may precede H4–H6 when suitable material data are available.
  **Gate:** ray paths in a linear density ramp against the analytic
  trajectory, one-dimensional absorption against the analytic attenuation, and
  absorbed/deposited energy budgets.

- [ ] **H9 — Add thermonuclear burn with local deposition.**
  DT and DD reactivities from the Bosch–Hale fits as sources on the isotope
  partial densities, which the species channel already carries, with the
  charged products deposited locally in the ion and electron energies by a
  declared split and the neutrons escaping into a yield diagnostic. Local
  deposition is also Riot's model; charged-particle transport is a separate,
  later item.
  **Depends on:** H3 for the deposition split, and H4a's isotope labels.
  **Gate:** constant-temperature burn against the integrated reactivity,
  species and energy budgets closed with the deposited and escaped energy, and
  yield convergence in the step.

- [ ] **H4a — Add isotope-resolved H/D/T ion transport.**
  The coefficient foundation, a standalone Stanton–Murillo evaluator for fully
  ionized H/D/T interdiffusion with its regime diagnostics, is in
  `src/ion_transport.jl` (commit `7c33651`) and does not enter the runtime
  flux. Remaining ([the transport plan](TRANSPORT.md#staged-execution-and-gates)):
  the concentration, pressure, ion/electron temperature and electric-field
  driving terms with a defined ambipolar closure, the plasma cross-model
  verification, and the flux integration. Validate dense-plasma models
  separately from the weak-coupling limit.
  **Depends on:** A7/H3 for coupled flux and energy evolution, H1/H2 when stiff,
  and H5 or an explicit ionization closure outside the fully ionized regime.
  Coefficient and flux-reference work can proceed before these dependencies.
  **Gate:** independent H–D/H–T/D–T references, isotope permutation and trace
  limits, zero-net-mass flux, charge/current closure, driven isotope separation,
  total-energy budgets, and declared coupling/degeneracy validity ranges.

- [ ] **H5a — Evolve DT from its true cold state through heating.**
  Prioritize open isotope-resolved data and models covering the intended cold phase,
  molecular dissociation, partial ionization, and the ionized limit. Derive EOS,
  charge populations, electron density, and transport from a consistent material
  state, including latent, dissociation, and ionization energies where relevant.
  Declare transported isotope inventory separately from equilibrium molecular
  and charge populations; finite-rate populations require their own evolution.
  Couple neutral, charged, and electron collision channels without an arbitrary
  temperature switch or applying fully ionized coefficients to cold fuel.
  **Depends on:** H3–H5 and H4a; H6 for a radiation-driven front.
  **Gate:** a cold DT target heats under a conduction/radiation wave without a
  manually frozen region, with independently checked cold equilibrium, front
  propagation, energy accounting, phase/charge limits, and timestep convergence.
  State physical transport suppression separately from any numerical limiter;
  data coverage and transition assumptions follow [the transport plan](TRANSPORT.md).

- [ ] **H5b — Qualify additional HED materials and their mixtures.**
  Inventory open EOS, population and transport coverage for C, CH and CD with
  explicit composition, material form, phase, units and energy references.
  Reuse the A7/H5 interface and H5a methodology, qualifying one declared material
  and density/temperature path at a time. Compound tables and physical mixtures
  require explicit mixing/equilibrium closures; ideal species mixing and isotope
  substitution do not establish cold-material validity.
  **Depends on:** A7/H5 for runtime material queries; H3/H4 and H1/H2 for the
  selected coupled heating case, H6 if radiation-driven. Coverage and standalone
  reference work may begin before DT evolution is complete.
  **Gate:** source/provenance and missing-range inventory, independent cold and
  warm references, inversion/derivative and population consistency, then energy
  and front-convergence tests for each claimed material. Qualify fuel–ablator
  mixture/contact closures separately; preserve A7's ideal-performance gate.

- [ ] **A8 — Qualify optional material-package extraction after interface use.**
  Decide whether to retain the internal module or extract it after A7 and H5
  exercise the boundary. Require a concrete reuse, dependency/data, or release
  benefit; independent packaging is not a prerequisite for HED development.
  If extracting, keep material evaluation independent of CompactLES/MPI, supply
  CompactLES adapters through an optional extension, preserve supported imports,
  and move `thermodynamic_model` and the `_record_fields` overrides of the
  checkpoint record with the types. Avoid a separate interfaces package
  until multiple consumers need it.
  **Gate:** record the extraction/retention decision; for extraction, independent
  package tests, extension loading/compatibility and coupled numerical tests,
  A7's ideal-performance gate, and measured load/precompile cost. Package/module
  boundaries must still allow solver-owned fusion and optional bulk evaluation.

- [ ] **H8 — Add MHD only for a magnetized target.**
  Extend the equation and flux interfaces and evaluate GLM divergence cleaning
  for the finite-difference scheme.
  **Gate:** standard wave/shock problems, divergence-error control, and energy
  budgets. Plan magnetized transport separately if the target requires it.

## Multimaterial interfaces

Every species today is a miscible gas: the channel diffuses partial densities,
the sensor reads mass and mole fractions, and the mole fraction is the volume
fraction only for ideal gases at one pressure and temperature. `StiffenedGas`
is a single material. An interface between immiscible materials, a gas over a
liquid or fuel against an ablator, needs a phase that may hold several miscible
species, a volume-fraction closure recovering one pressure, and one temperature
if that closure is chosen, from the mixture energy, and a species flux that
sharpens the interface instead of diffusing it. No current target requires
this; the items wait for a case and follow
[the material interface design](DESIGN.md#material-and-physics-interfaces).
H5's tables are related infrastructure, not this closure.

- [ ] **M1 — Add a mixture thermodynamics that distinguishes phases from species.**
  Define a phase as a set of miscible species; allow molecular mixing within a
  phase and none between phases unless a physical model supplies it. Recover
  the common pressure and temperature from the volume closure Σ α_k = 1 and
  the mixture energy, and expose per material the density, internal energy,
  enthalpy, volume fraction, sound speed and the EOS derivatives the fluxes and
  the artificial conductivity scale read. Riot's multimaterial state is this
  closure: per-material partial densities and one bulk energy under
  pressure–temperature equilibrium, with a closed-form solution where every
  material at a point is an ideal gas; the PTE solvers of singularity-eos are
  the reference algorithms. Start with an ideal-gas and a
  stiffened-gas mixture; tables follow once the pure and trace limits and the
  inverse recovery are verified. The transport model then takes zero
  intermaterial diffusion with nonzero viscosity and conductivity, which the
  dispatchable transport of N8 admits as a rule and not a coefficient.
  Admissibility is the EOS's, through `state_admissibility`: no universal
  positive-pressure or positive-internal-energy rule, and an absent phase
  handled apart from a trace-material floor.
  **Depends on:** A7's queries and status types; H5 for tables.
  **Gate:** pure and trace limits, recovery residuals, hyperbolicity,
  insensitivity to an initialized trace fraction, and A7's ideal-fluid
  performance gate.

- [ ] **M2 — Add a conservative sharpening flux on the staggered operators.**
  The conservative diffuse-interface flux of Jain, Mani and Moin (2020) and
  its accurate variant (Jain 2022) balance a diffusion against the nonlinear
  sharpening term Γ(ε∇φ − φ(1 − φ)n̂); each is a divergence and needs its
  fields at the half nodes. Build it on the staggered compact operators of H1
  ([IMPLICIT.md](IMPLICIT.md)), not on a separate finite-volume path, which
  the design commitments exclude. Verify the discrete diffusion–sharpening
  balance at low order before raising it; derive the momentum and energy
  consistency terms from the complete species flux, sharpening included, for
  the equilibrium model M1 selects, rather than importing the incompressible
  phase-field equation; define the regularization parameters, the interface
  normal, the step restriction and the supported thickness in cells. Verify
  pairwise symmetry, Σ α_k = 1, permutation invariance of the phase labels,
  and that an absent phase creates no material and leaves the fewer-phase
  solution unchanged. Compare the two forms on shape error, grid alignment,
  robustness and cost.
  **Depends on:** M1; the staggered operators of H1 stage 1 are delivered.
  **Gate:** stationary and obliquely advected interfaces, a three-material
  junction and an equal-density interface, under the interface accuracy rows
  of `test/convergence.jl` and the N10 budgets.

- [ ] **M3 — Qualify the interfaces under-resolved and shocked, then add physics by target.**
  Deliberately under-resolved droplets and thinning ligaments, measuring mass,
  shape, breakup time, satellite sizes and mixing rather than snapshots, with
  the mean and maximum interface thickness reported and saddle points at
  breakup handled explicitly; a resolution and regularization-timescale sweep
  through a strong shock–interface interaction, confirming the sharpening
  still acts at the stable settings; acoustic reflection and transmission at a
  material interface against an independent reference; and, under refinement,
  how the grid-dependent thickness changes at a coarse–fine face and the mass
  and energy transient it leaves. Physics beyond the isobaric-isothermal
  closure is conditional on a target: assess whether instantaneous thermal
  equilibrium between materials serves the collapse, rebound or heat-transfer
  problem at hand, and compare against a five-equation reference where
  compression or thermal nonequilibrium matters, keeping that nonequilibrium
  distinct from H3's temperature split; capillarity, if required, as a
  pressure-balanced surface tension with its energy accounting, checked on
  Laplace pressure, spurious currents and capillary waves; phase change and
  interphase mass transfer as separately scoped models. Sharpening is not
  surface tension, and no discretization corrects an unsuitable closure.
  **Depends on:** M1, M2; H3 for a two-temperature comparison.
  **Gate:** the cases above with reproducible inputs, source revision,
  reference provenance and the supported parameter envelope, and negative
  results recorded with the simpler method retained where an extension does
  not improve the target metric.

## Scale, devices and performance

Detailed mechanisms and measurements remain in [AMR_GPU.md](AMR_GPU.md) and
[CLUSTER.md](CLUSTER.md). Patch AMR, device execution, stacked tile launches, and
opt-in Float32 already exist; the tasks below extend or validate them. S15 is
under [the refinement track](#refinement-for-the-production-geometry).

- [ ] **S12 — Remove the O(P²) replicated interface solve for many-rank scaling.**
  The reduced interface matrix is dense of order 2qP and every rank applies
  its LU to its own lines, about 8q²P² operations per line against 9qn for
  the local sweep. Per rank on a uniform 3-D grid that is a fixed 8q²N² per
  solve while the local work shrinks as N³/P³: an Amdahl term whose
  operation-count crossover is P ≈ 6.6 (q = 1) or 5.2 (q = 2) per direction
  at N = 256 and 10.4 / 8.3 at N = 1024, a few hundred to a thousand ranks.
  The 224-rank rzhound run with dims (8,7,4) was at that crossover along x.
  Per-step collective latency is not the limit: Allgathers span one direction's
  P ranks and the measured node scaling was 93% per doubling at four nodes.
  Stages, in order:
  1. Instrument first: `bench/reducedsolve.jl` (commit `3fb37a9`) times the
     local sweep, the Allgather and the reduced solve separately for both
     solvers, and its run on rzhound (`bench/slurm/s12_reducedsolve.sbatch`)
     remains.
  2. Done in commit `6bfeb55`: a pivoted band LU of the block-tridiagonal
     reduced matrix, periodic lines by an interleaved ordering, the dense LU
     kept only at P = 1; the reduced stage is 2.3–5.6× faster at P = 2–16 on
     the workstation and departs from serial within the dense solve's range.
  3. Only if the Allgather volume (2qP lines' worth per rank) then shows in
     the probe: a distributed reduced solve in which only neighbors exchange,
     the SPIKE recursion, which also removes the replicated factorization.
  Process-grid guidance follows from the same count: for a fixed rank count
  the per-rank reduced cost is proportional to the sum of the cubes of the
  per-direction rank counts, so near-uniform grids minimize it and slabs
  maximize it. **Depends on:** system-MPI cluster time for stage 1. **Gate:**
  the MPI suite at 2, 4 and 8 ranks, `bench/tgv_energy.jl` reproducing serial
  energy histories to round-off, and the node-scaling table re-measured at
  the largest rank count available with the probe's breakdown beside it.

- [ ] **S1 — Complete target-machine GPU measurements and resolve the wait stall.**
  Continue the rzadams/MI300A campaign using the existing measurements as a baseline,
  not as an unmeasured port. Characterize/resolve the intermittent ROCm wait stall
  in [rocm_wait_stall_report.md](bugreports/rocm_wait_stall_report.md) before interpreting
  performance changes. Record hardware, MPI stack, precision, synchronization
  policy, repeat variability, and correctness with every result. Profile a
  `Nasa9Mixture` step there too with `bench/device_nasa9.jl`, which has run only
  on the workstation GPU. The LLVM 22.1.8 `llc` in `AMDGPU_LLVM_Backend_jll`
  miscompiles a short-circuit choice between two kernel arguments on gfx942 as
  on gfx1030 ([amdgpu_shortcircuit_bug_report.md](bugreports/amdgpu_shortcircuit_bug_report.md));
  a target run uses an AMDGPU.jl release on the version-23 backend, or relies on
  the `ifelse` rule holding in every kernel.
  **Gate:** full single/refined/tiled runs on target hardware, including real device
  communication paths; report measured reproducibility rather than universal
  bitwise claims. Validate other advertised backends on their own hardware.

- [ ] **S2 — Measure and address compact-solve and transfer scaling limits.**
  Profile replicated dense interface solves, host/device fences and transfers,
  and the level coupling's point-to-point exchanges (S15) as line-rank count and
  region size grow. Compare gather-solve-scatter, on-device reduced solves and
  GPU-aware MPI only where profiles justify them.
  Measure stacked `max_rate` reductions before adding another launch optimization;
  revisit extra streams only if batching still leaves useful concurrency.
  The replicated interface solve has a cost model and a planned fix in S12.
  **Depends on:** reliable S1 timing. **Gate:** crossover data and preserved
  distributed accuracy, not a workstation-only speedup claim.

- [ ] **S3 — Complete production tile/ownership cost studies.**
  Build the 3-D implosion-shell benchmark at realistic tile sizes; measure
  per-imposition latency and repeat the previously cold tiled-overhead case warm.
  Quantify the benefits of rebalancing (commit `603af33`) and point-to-point
  migration (commit `3980639`), including root-work bias in measured tile
  weights; subtract that baseline only when its impact is established.
  **Gate:** repeated-process timings, memory, imbalance, migration cost, and accuracy
  against coarse and uniform-fine references on the target clusters.

- [ ] **S4 — Choose mixed-precision policy (former G4b).**
  Separate state, geometry, clock, solve, and accumulation precision explicitly;
  the savepoint clock is already held in Float64 in either precision (commit
  `bdf1ddf`). Use existing Float32 CPU/device histories and S1/S2 profiles to
  compare policies.
  **Gate:** conservation drift, thermodynamic recovery, closure conditioning,
  timestep progress, memory, and throughput; no CPU-default change from speed alone.

- [ ] **S7 — Validate thread pinning and cluster placement.**
  Wire pinning only after controlled target-cluster trials: fixed-rank comparisons
  on identical masks, repeated-process medians, and one rank per NUMA domain with
  pinned threads on the suitable Julia runtime.
  Preserve the SMT-sibling collapse reproducer and prepare evidence for site
  support. **Gate:** measured benefit over scheduler binding and repeatable launch
  guidance in [CLUSTER.md](CLUSTER.md).

- [ ] **S13 — Cut the remaining cost of the NASA-9 temperature inversion.**
  `recover_primitives!` under `Nasa9Mixture` runs a safeguarded Newton solve
  per point and per Runge–Kutta stage, and that solve is the ~6× step cost of
  the NASA-9 model over `IdealMixture`. The interval table is isbits and the
  inversion and every per-point species loop share the powers of T (commits
  `9e0f126`, `405037f`, measured with `bench/nasa9_inversion.jl`). The
  convergence criterion is eps^(2/3), floored at 1e-10 (commit `dc67f71`), which
  saves most of one iteration, and the iterates stay inside the fitted range
  (commit `3ede230`). What remains is a warm start from the stored
  `T_ion` field, one or two iterations instead of about four. It trades away
  the state-only seed that `mixture_temperature_status` documents for
  bit-for-bit agreement between serial and decomposed runs and for restart
  independence; only worth it if the model is still far from the ideal-gas
  step cost. Keep the polynomial powers literal (`T^4`): a repeated product is
  not bit-identical to the library power and moves every baseline for nothing.
  **Depends on:** a baseline decision.
  **Gate:** the core gate with explained baseline updates; time the inversion
  before and after at four species in the same session.

- [ ] **S11 — Fuse each compact line solve into one threaded region.** Low
  priority: on the 2-D case that motivated it, `mpiexec -n 8 -t 1` at 12 ms/step
  already beats the fused-threads target of about 15, so this waits for a
  setting in which threads are the only parallelism, such as the host-side
  interface stage of a device plan.
  A line solve is three regions today (fill, solve, scatter) and each carries
  a spawn/join floor and a barrier, with the solved block crossing cores
  between them. On a 37k-point planar case at eight threads that granularity,
  not bandwidth, is the loss against ranks: 195 regions per RHS at about 54 µs
  serial work each, ranks 2x faster per step than pinned threads
  ([CLUSTER.md](CLUSTER.md), hybrid-desktop paragraph). A prototype in which
  each thread fills, solves and scatters its own chunk of lines in one region
  measured 1.86x on the x-direction filter solve at `-t 8`, parity serially
  and bitwise identical, provided the 16-wide column blocking of `solve_cols!`
  is kept inside each chunk. The transposed y/z path fuses the same way over
  chunks of x-columns, since a chunk's fill, row-sweep solve and scatter are
  all contiguous in it. Where the dimension is decomposed the reduced
  interface stage is a collective between the local solve and the spike
  correction, so the fusion there is two regions, not one, and the collective
  stays outside both. The `@threaded` docstring names this reorganization as
  the condition for re-measuring the threading backend; do that after, not
  before. **Gate:** the core gate, `bench/jetcheck.jl` and `bench/audit.jl`
  before and after, bitwise agreement of the serial suite, the MPI suite at
  2, 4 and 8 ranks, and the 768×48 thread table re-measured pinned.

## Geometry

- [ ] **S9 — Finish azimuthal mode truncation.**
  Follow [MODE_TRUNCATION.md](MODE_TRUNCATION.md). Stages 1 and 2, cylindrical
  (`polar_truncation`, off by default) with θ serial or decomposed, are
  delivered (commits `c5ac69f`, `851503e`, with the checkpoint record in
  `4fa1a55`); calibration/defaults and the spherical azimuth remain. The rate
  cap also caps the θ rate that `filter_weight` reads, so the θ filter pass
  weakens wherever the inner rings set that rate; the calibration measures it.
  **Gate:** conservation and mode errors, pole/axis behavior, and achieved CFL/cost
  benefit. Keep this acoustic/geometric restriction distinct from the diffusive
  restriction addressed by H1.

- [ ] **S10 — Add immersed boundaries on demand.**
  Follow [IMMERSED.md](IMMERSED.md): level-set geometry and graded post-stage
  Brinkman/reset blend; validation/calibration; normal extrapolation refinements;
  then moving bodies. Include force and heat-flux accounting.
  **Depends on:** a target needing non-coordinate-surface geometry; the first stage
  need not wait for HED or multiblock extensions.
  **Gate:** the design's stage-specific accuracy, stability, and force/energy
  budgets, followed by backend/AMR compatibility checks.

## Deferred scope and non-goals

- **Wall-resolved/wall-modeled LES:** deferred until a target requires it;
  current no-slip support serves verification and does not establish wall-model fidelity.
- **Radiation transport beyond diffusion** (discrete ordinates, Monte Carlo):
  deferred until a target shows H6's diffusion inadequate.
- **RANS mix models** such as Riot's BHR-3.1: a comparison for resolved mix
  widths, not a model to carry.
- **Symbolic frontend:** optional only after hand-written equation sets define a
  usable interface; a macro may emit typed code, but runtime PDE-string evaluation
  remains excluded.
- **Excluded:** cell-by-cell/oct-tree AMR, unstructured meshes, a parallel Godunov
  solver path based on Riemann solvers/flux limiters, and general-purpose CFD scope.
  Instrumented state repair remains allowed and must retain intervention budgets.
- **Design commitments:** compact structured operators, patch-based refinement,
  explicit geometry/collective contracts, and measurable accuracy for mixing and
  implosion. Revisit a restriction through a concrete case and a documented design.

## Closed

### Runtime and boundary correctness

- [x] **R1** — Diagnostics and field output no longer disturb the artificial
  coefficients the next timestep check reads (commit `4be4ac0`).
- [x] **R2** — Endpoint conversion and progress checks turn an unrepresentable
  advancing step into a diagnosed failure instead of a stalled clock
  (commit `4be4ac0`).
- [x] **R3** — An EOS-aware validity policy checks accepted and returned states
  under `:strict`, `:permissive` or `:repair` (commit `4be4ac0`).
- [x] **R4** — The NASA-9 inversion reports convergence and extrapolation status
  instead of returning an unchecked estimate (commit `4be4ac0`).
- [x] **R5** — `correct_flux!` imposes the adiabatic impermeable no-slip wall's
  species and energy flux contract on the assembled flux before divergence
  (commit `51d7b2a`).

### Filtering, regularization, boundaries and species

- [x] **N1** — The compact filter default is `filter_cfl = 0.35`, with the
  relaxed formulation measured invariant to the step (commit `dbe2899`); α is
  0.47 since commit `ccb7e81`.
- [x] **N2** — Volume-weighted filtering on non-uniform metrics was measured
  against the unweighted default and the current method retained (commit `811d382`).
- [x] **N3** — `run!` primes the artificial coefficients, which removed the wall
  and axis CFL ceilings; the spherical origin keeps its own (commit `c407e0b`).
- [x] **N5** — Directional bulk viscosity was measured on anisotropic Noh cases
  and rejected; the scalar form is retained (commit `7dd161f`).
- [x] **N5a** — `filter_weight` reads each direction's own hyperbolic rate rather
  than the rate that sized the step (commit `d1de5cc`).
- [x] **N6** — The wall and interface accuracy probes became the smooth-evolution
  matrix of `bench/boundaryorder.jl`, with its orders gated in
  `test/convergence.jl` (commit `a1d7660`).
- [x] **N6a** — `compact_filter` defaults to `closures = :onesided`, which cuts
  the planar Noh wall deficit at the cost of a sharper reflection
  (commit `d611bae`).
- [x] **N6b** — C6 `:brady_livescu` is a supported wall configuration within
  measured limits, C8 is not, and the default is unchanged (commit `a619a6e`).
- [x] **N6c** — The constant-annihilation audit closed with no change and found
  the slip-wall mode of the cascade closures (commit `02c34eb`).
- [x] **N6d** — The `:neutral3` rows, spectrally neutral at an inviscid slip
  wall, became the C6 default, with the cascade rows kept at interfaces
  (commit `1e76f50`).
- [x] **N6e** — The `:delta4` detector reads past a reflecting wall from the
  node-centred mirror of the interior instead of clamping the index
  (commit `60da34b`).
- [x] **N6f** — The detector takes the half-offset mirror at a coordinate fold
  for an even field as well as an odd one (commit `abe3d8d`).
- [x] **N6g** — `wall_closures` gives the `:gaussian` smoother and the `:d8`
  detector wall rows folded from their own interior stencils (commit `ae574ca`).
- [x] **N6h** — `SlipWallBC` imposes the symmetry plane's flux contract on the
  assembled wall-plane flux (commit `e2a6c4f`).
- [x] **N6i** — The aligned Noh transverse mode is a shock interaction rather
  than an autonomous wall mode, and its guard is retained (commit `f81b1c6`).
- [x] **N6j** — The fifth-order closure candidates were re-measured under
  `beta_sensor = :dilatation` and the production choices retained
  (commit `de30ccc`).
- [x] **N6k** — The neutral rows carry a measured pseudospectral certificate and
  are the `:neutral3` defaults of C8 and C10 as well (commit `de30ccc`).
- [x] **N6l** — `SymmetryPlaneBC` folds a slip wall on a face-centred mirror and
  reads the interior order, for a single unrefined patch (commit `8a5bdfe`).
- [x] **N6m** — C6 `:brady_livescu` takes every start of the planar Noh
  ladder on the current solver, the singular one included, and its remaining
  failures are CFL ceilings that no initial-data measure orders, so the
  resolved-start requirement is lifted by measurement and no initial-data
  check is added; C8 `:brady_livescu` stays unsupported at a wall (commit `4faed86`).
- [x] **N7** — `NSCBCInflowBC` carries the Yoo–Im transverse terms on every
  incoming wave, at the full share by default, measured on an oblique pulse
  and an entering vortex at three Mach numbers (commit `d3d3f30`).
- [x] **N8** — Dispatchable CEA transport supplies temperature-dependent mixture
  viscosity and conductivity, with unity-Lewis diffusion or mixture-averaged
  diffusivities from supplied binary data (commit `6594afd`).
- [x] **N8a** — Validated, species-labelled neutral pair fits feed the
  mixture-averaged solver closure with strict collective domain rejection,
  including subcycled AMR, with the sources kept distinct (commit `711cecf`).
- [x] **N9** — The bulk species channel was measured in three dimensions
  against the Fickian one, its constants confirmed, and its entropy inequality
  restated as a continuous-model property; `:fickian` stayed the default
  (commit `7a86a2c`).
- [x] **N9a** — `:partial_density`, the partial-density channel of Brill, Olson
  and Bokman with its consistency fluxes on momentum and energy, is the
  default; `:bulk` remains for density ratios of 100 or more and `:fickian`
  reproduces Cook's form (commit `a2afda2`).
- [x] **N19** — False activation on smooth fields was measured, and the
  species fields are sensed with `:d8` at `C_D = 1` by default, every other
  sensor keeping δ⁴ (commits `ee1087b`, `0a64d6f`, `7a45c8c`, `072d184`).
  Reduced filtering waits until the discretization is shown stable without it.
- [x] **N20** — The boundedness mechanisms are written down, the sweep sees
  regridded and subcycled states, an acoustic pulse meets the interface
  impedances, and the composite budgets are attributed per mechanism with the
  boundary-flux closure verified; the coarse–fine drift converges and needs
  no conservative correction (commits `5dbb5ff`, `2a5ece9`, `a889b61`).
- [x] **N21** — The composition clip of the repair acts only outside
  `StepControl.species_band`, which now bounds both sides through a test shared
  with the validation sweep (commit `0293798`).

### AMR numerics

- [x] **N10** — Composite conservation budgets were measured on long
  passive-species mixing and moving-refinement runs across rank counts,
  refinement depths and subcycling, and every layout is inside its
  declared budget, so no surface-flux correction is enabled (commit
  `74895a4`).
- [x] **N11** — The δ⁴ detector reads the ghost layers at patch and
  coarse-fine faces for every field recovered over the padded extent, and
  the sensors, the filter rows and the closure candidates were measured at
  imposed shells on shock crossings, with the step-on-a-plane failure and
  the three-level global-step undershoot recorded (commit `b4fe9d6`).
- [x] **N12** — Regrid checks refresh coefficients, opt-in refined-stage CFL
  violations use collective rollback, and three-/four-level rate and cost
  measurements retain Hermite output; the global-step undershoot depends on
  the filter application schedule (commit `e7bb381`).
- [x] **N12a** — Level-aware filter trials remove the fixed-layer
  undershoot within composite budgets; smooth-error and moving-refinement
  tradeoffs retain the production default and subcycling (commit `fbb3ad3`).
- [x] **N13** — Mass fractions are validated against a measured
  `StepControl.species_band` of 0.05, so every species case and both examples
  run under the retained `:strict` default; only cold-ambient converging shocks
  opt out, with bounded counts (commit `c9d41f4`).
- [x] **N14** — `interface_divergence` selects the flux divergence's closure rows at
  interface ends; the default stays, `:cascade4` is rejected and the Brady–Livescu
  rows stay an experimental Float64 option (commit `6928fe5`).
- [x] **N15** — `interface_flux = :ghost` differences the inviscid flux through
  interface ends from ghost fluxes; it leads at same-level faces, and at
  coarse–fine faces under `level_interpolation_order = 8`, and stays
  experimental (commit `b42e819`).
- [x] **N15a** — Under `interface_flux = :ghost` the molecular flux is
  differenced through interface ends from ghost fluxes, a same-level face's taken
  from the neighbour's flux records and a coarse-fine face's from the shell's
  gradient ring; viscous interface rows read 6.1–6.8, and promotion moves to
  N15b (commit `decf45a`).
- [x] **N15b** — `interface_flux = :ghost` is the default at patch and level
  interfaces and inert without one; `:closure` remains for shock-dominated and
  Float32 runs, and a configuration `:ghost` does not support raises an
  `ArgumentError` naming it (commit `e9a91ed`).
- [x] **N16** — `level_interpolation_order` (2, 4, 6 or 8) sets the live
  transfer order; 6 stays the default, which the default interface rows
  saturate, and 8 is the opt-in for viscous, filtered, multidimensional or
  `interface_divergence` runs (commit `c7d26e3`).
- [x] **N17** — `level_interpolation_order` defaults to the derivative
  operator's interior order, two more under `interface_flux = :ghost` up to 10;
  order 10 is exact to degree 9 behind the four-node buffer, and a C6 run under
  the closure rows is unchanged (commit `decf45a`).

### API, state ownership, and reproducibility

- [x] **A1** — `precision` on `Numerics` and `Solver` converts every typed
  component, mixed types are rejected at setup, and `bench/audit.jl` checks
  Float32 point bodies for promotion (commit `5bd9b46`).
- [x] **A2** — Field ownership and freshness are documented in `patches.jl`,
  a scratch read names the patch that wrote it, and a `:repair` before the
  step renews the prepared state (commit `57b4ed2`).
- [x] **A3** — Construction lives in `construction.jl` and the `Solver` container
  in `solver.jl`, moved verbatim with byte-identical inference and allocation
  audits; `rhs.jl` holds only the right-hand side (commit `f123739`).
- [x] **A4** — Parameter ranges fail early with named errors, `run!`'s absolute
  `tfinal`/`nmax` are enforced, and a rollback restores switches, triggers and
  writer frames (commit `99be93c`).
- [x] **A5** — Checkpoints carry a versioned configuration record compared
  at load: thermodynamics and layout strictly, the other groups under `allow`
  (commit `77b3adb`).
- [x] **A6** — The supported-combinations page lists what setup accepts and
  the error of each rejection, held to the code by `test/capability_tests.jl`
  (commit `fb3b242`).
- [x] **A9** — `line_sample`, `field_slice` and the Makie recipes take the
  state vector of a refined run and sample root nodes from the finest level
  holding them (commit `e5d521d`).
- [x] **A11** — A tiled, regridded level may hold no tiles, so a run starts
  unrefined, refines when tags appear and empties when they vanish; the box
  keeps its region (commit `1a97ab4`).
- [x] **A12** — A tiled hierarchy regrids every level up to `max_levels`, each
  tagged on its parent; a three-level shock–contact run is 16× closer to the
  uniform-fine reference than the root (commit `737550c`).
- [x] **A14** — `Hydrostatic` sets the pressure of an initial condition in
  discrete balance with the run's derivative operator, and an unfiltered
  two-fluid column stays at rest to round-off (commit `acfe5cb`).
- [x] **A15** — `CompositeBC` divides a face among member conditions by a
  coordinate mask; a walled-orifice jet and a two-slot stagnation plane run
  on it (commit `9574aea`).

### Scale, devices and I/O

- [x] **S5** — `Nasa9Mixture` runs on device storage through an isbits mirror
  carrying the flattened interval table (commit `b78acd4`).
- [x] **S6** — `test/hdf5_tests.jl` passes on a parallel libhdf5 at one, two
  and four ranks under both transfer modes, and the collective mode, measured
  faster on a local disk, is the default (commit `c8df974`).
- [x] **S14** — The decomposed `pyranda_filter` solve's departure from serial is
  the round-off floor of its ill-conditioned left-hand side, the same fraction
  of cond(A)·eps as every other operator's, and is recorded as such
  ([the measurement](CALIBRATION_APPENDIX.md#the-decomposed-line-solve-against-serial))
  (commit `1723ea3`).
