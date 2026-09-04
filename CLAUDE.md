# Working in CompactLES

Orientation for coding agents, kept short: it carries only what you
need before touching anything. **It does not describe the solver**, and it is not
the place to record a result.

| file | what is in it |
|---|---|
| `README.md` | usage |
| `reference/DESIGN.md` | the numerics — source map, compact solve, folds, GCL, NSCBC |
| `reference/CLUSTER.md` | MPI configuration, launch rules, sizing, measured scaling |
| `reference/CALIBRATION.md` | the artificial-property constants, the CFL restriction, TGV |
| `reference/ROADMAP.md` | positioning, comparisons, open work items |
| `reference/HISTORY.md` | completed phases and their measured outcomes |
| `reference/AMR_GPU.md` | the patch-AMR + GPU design as delivered, measured lessons, roadmap |
| `reference/IMMERSED.md` | the immersed-boundary design: level-set bodies, blend imposition |
| `reference/MODE_TRUNCATION.md` | azimuthal mode truncation plan for the pole CFL squeeze |
| `docs/` | the Documenter site: `make.jl` and the `src/` pages. `docs/src/tutorials/` and `docs/build/` are generated (from `docs/literate/`) and gitignored, so edit `literate/`, never `src/tutorials/` |

Read those for anything about *what* the code does, *where* it runs, or *where it
is going*. This file is about *how to work on it*. When a measurement proves
durable, it belongs in one of the files above, with a one-line pointer here
only if an instance would go wrong without it.

## Environment

Compat bounds live in `Project.toml`; MPI.jl 0.20 requires special attention
because its API changed on either side of that release.

`mpiexec` is usually not on PATH. MPI.jl provides whatever launcher this
checkout is configured against, which is a JLL binary on a workstation and the
system MPI (often the scheduler's launcher) on a cluster. Query it for the
launcher; do not hardcode a path:

```bash
julia --project=. -e 'using MPI; MPI.mpiexec(c -> println(c))'
```

Before interpreting any scaling or timing number, identify the hardware.
`Threads.nthreads()`, the physical core count, and whether the cores are
uniform all change the interpretation; a hybrid performance/efficiency-core
desktop will spread threads across both and understate scaling. `ThreadPinning`
supplies the topology queries `clusterprobe.jl` uses. Nothing pins threads yet;
only the querying half has been validated.

Windows checkouts may have CRLF line endings. Helper scripts that match
multi-line text against `\n` will silently find nothing; match a line at a time.

**On a cluster, read `reference/CLUSTER.md` before doing anything else, and do
not skip it because the code runs.** MPI.jl defaults to a bundled JLL that
satisfies the scheduler perfectly well on one node and never reaches the
interconnect off it: **27x–66x** slower on rzhound, with no symptom but speed.
`LocalPreferences.toml` is per-project, so `--project` silently selects the MPI
implementation. `reference/CLUSTER.md` also holds the launch-line rules
(`--cpu-bind=threads`, `-t 1`, and why a rank must never hold both SMT threads of
a core), the sizing tools, and the measured scaling.

`Manifest.toml` is gitignored, so a fresh checkout needs `Pkg.instantiate()`
before anything runs. Precompiling the package takes about a minute, most
of it the workload in `src/precompile.jl`, which runs the test suites'
solver configurations at np = 1 under a singleton `MPI.Init` so that a
test rank compiles a quarter of what it otherwise would; it is skipped
under a system MPI. Julia 1.11+ keys the cache on content, so `touch`
does not rebuild it; force one with
`Base.compilecache(Base.identify_package("CompactLES"))`. `bench/tgv_energy.jl` is the intended first production workload
on a cluster: the one bench script whose reductions are all collective. Those
reductions are order-dependent `Allreduce(+)`, so it reproduces serial numbers
to round-off (of order 1e-14 relative), not bit-for-bit; `test/mpi_tests.jl`
measures 3.1e-15 at np = 4 for the decomposed compact solve it rests on.

## The gate

Run all of it before calling a change safe.

```bash
MPIEXEC=$(julia --project=. -e 'using MPI; MPI.mpiexec(c -> print(c))')

julia --project=. test/runtests.jl        # 128 testsets, 0 failures
julia --project=. test/convergence.jl     # measured orders, see below
julia --project=. test/validation.jl      # shock-capturing battery, ~25 s
for np in 2 4 8; do
  "$MPIEXEC" -n $np julia --project=. -t 1 test/mpi_tests.jl   # 186/186 each
done
julia --project=. bench/jetcheck.jl       # inference
julia --project=. bench/audit.jl          # allocation + non-concrete SSA
julia --project=. test/docrefs_tests.jl   # every docs @ref resolves; also in runtests
```

**Run `test/docrefs_tests.jl` before every push, and after any docstring or
`docs/` edit.** The documentation job is the slowest leg of CI and the last
to report, and it has failed repeatedly on a `[`Name`](@ref)` whose target no
`@docs` block renders: Documenter cannot resolve the link and fails the
build. The file resolves every `@ref` in the rendered docstrings and pages
the way Documenter will, without building the docs, in about a second after
the package loads; it also checks that every `@docs` entry is a documented
binding and that every exported name with a docstring is rendered somewhere
(`checkdocs = :exports`). A private name may be linked with `@ref` only if
it is listed in a `@docs` block on some page; otherwise write it in plain
backticks. `runtests.jl` includes the same check as its last testset.

The `128 testsets` and the `185/185` are counted by different mechanisms and are
not comparable. `runtests.jl` runs everything inside one top-level
`@testset` and prints one summary tree at the end: the top row totals the
`@test`s, and each row beneath it is one `@testset` of the suite, the ones
its includes contribute too (`float32_validation.jl`, `device_tests.jl`,
`patch_tests.jl`, `level_tests.jl`, `io_tests.jl`, `docrefs_tests.jl`, and
the device/adapt testsets); the `128` counts those rows. A failure is
recorded and the suite runs on, so one run reports every failure, and the
outer testset raises at the end. `mpi_tests.jl` uses `Test` not
at all and counts individual `check(name, val, tol)` assertions, printing
`passed/total` and exiting nonzero on any failure. Adding a testset moves the
first number, adding one assertion moves the second, and neither is a fraction
of the other, so treat both as figures to record before a change and compare
after.

`test/hdf5_tests.jl` covers the HDF5 extension and is skipped by the gate above:
HDF5 is a `[weakdeps]` entry, so it is not loadable from the package
environment. `runtests.jl` prints when it skips. To run it, use an environment
carrying both CompactLES and HDF5, serially and under `mpiexec`; the
decomposition-independent restart writes on one process grid and reads on
another, which np = 1 cannot exercise. `hdf5_parallel()` reports which write
backend the libhdf5 build selects; it is `false` on a workstation, and the
collective path can only be exercised where a parallel libhdf5 built against
the run's MPI exists.

`test/makie_tests.jl` covers the Makie extension and is skipped the same way,
but for a different reason: HDF5 is in `Project.toml`'s test target and
CairoMakie is not, because resolving and precompiling it for two
testsets is out of proportion to what they cover. **`Pkg.test` therefore never
runs them.** The Makie extension is verified only from the docs environment,
which carries CairoMakie, and under `mpiexec` for the
decomposition-independent profile.

`test/convergence.jl` prints measured orders against regression guards baked
into the file: C6 6.01, C8 8.00, C10 10.04, C6 wall closures 3.17 (`:cascade4`
4.02, `:brady_livescu` 5.88), C8 wall closures `:brady_livescu` 7.91, filter
pass `:cascade` 1.88 / `:onesided` 8.07, cylindrical axis odd 3.71 / even
3.00, resolved-θ 3.71, spherical origin 2.99. **For a change not
expected to affect numerics these should come out bit-identical, down to the error
magnitudes.** A moved digit indicates a real change; chase it before moving
on. Each study now asserts both a wide guard, which fails when the order
has regressed, and a ±0.02 guard against the number above, which fails when it
has drifted; the two print different diagnostics. Update a `recorded` value only
together with the list here and the table in the file's header.

`test/validation.jl` prints measured errors against guards baked into its
header. Unlike the convergence orders these are **not** bit-reproducible: each
case integrates thousands of steps through a nonlinear sensor, so an arithmetic
reassociation anywhere in the artificial-property path moves the fourth
significant figure. Four digits is the level to compare at; a moved *third*
digit is real. The cases themselves live in `test/cases.jl`, shared with
`bench/artcal.jl` so the calibration study and the guards cannot drift apart;
add a case there, not in either consumer.

`bench/jetcheck.jl` and `bench/audit.jl` print counts, not pass/fail. Record
them before a change and compare after, and read the delta, not the absolute
count. `jetcheck.jl` reports zero dispatch sites at every probed entry point,
so *any* report is a regression. The counts overlap between entry points
(`step!` contains `compute_rhs!`), so compare like with like and never sum them.

`audit.jl`'s inference probe reads `code_typed` at a spelled-out signature. A
**keyword argument on a probed entry point, or an optional trailing argument the
probe omits**, resolves that signature to the short forwarding method, not the
body, and the reported count falls to 1 while measuring nothing.
`compute_rhs!` and `step!` therefore carry a trailing `Bool` positionally. If you
add another optional argument to either, extend the probe tuple in the same
commit.

`bench/coverage.jl`'s header has the coverage sequence. Serial alone reaches
94.8% of executable lines and the full set with MPI 97.2%; the difference is
distributed-solve and off-rank-fold code that only a decomposed run touches.
Julia's `.cov` output omits methods that were never compiled, so the percentage
overstates coverage; watch the executable-line count too.

**The test suite does not import `bench/` or `examples/`.** They stay green
while broken. Run them after any cross-cutting change.

## Naming

Names are spelled out in full. Current vocabulary:

- `solver`, `decomp`, `n_global`, `n_local`, `n_halo`, `n_halo_d`, `offset`,
  `neighbors`, `send_buf`/`recv_buf`, `sub_rank`/`sub_size`, `pad` (the
  per-dimension halo pad, as a local), `free_communicators!` (called when a
  `Decomp` is permanently dropped; MPI frees nothing until GC otherwise),
  `owns_communicators` (false on one rank of `COMM_WORLD`/`COMM_SELF`, which
  a `Decomp` borrows instead of building a topology; `cart_rank` covers that
  case)
- `n_species`, `n_cons`, `i_mom`, `i_energy`, `Y`, `cp_mix`
- `mu_art`, `beta_art`, `kappa_art`, `D_art`, `C_mu`/`C_beta`/`C_kappa`/`C_D`,
  `mu_sensor` (`:strain` or `:velocity`), `beta_sensor` (`:strain`,
  `:gated_strain`, `:dilatation` or `:ungated_dilatation`), `reduction` (`:sum`
  or `:max`), `smoother` (`:gaussian` or `:compact`), `detector` (`:delta4` or
  `:d8`)
- `grad_u`, `grad_T_ion`, `grad_Y`, `strain_mag`, `sensor`, `sensor_sp`
- `inv_J`, `area_d`, `inv_h`, `inv_r`, `cot_over_r`, `coord_shift`, `flux`
- `filter_interval` (cadence in steps) vs `filter_cfl` (the reference CFL at
  which a filter pass is full strength; 0 disables the relaxation), `filter_weight`
- `deriv_plans`, `filter_plans`, `line_solver`, `plan` (a DirPlan) vs `plane`
  (a wall plane), `fold`, `pair`
- `plane_profile`, `profile_spacing`, `mix_width`, `molecular_mixing`,
  `quad_weight`, `cell_measure`
- `eos_phi`, `eos_dphi_dY`, `art_conductivity_scale`, `species_energy`,
  `mixture_temperature` (the EOS contract; the whole list is at the top of
  `physics.jl`)
- `control` (a `StepControl`), `max_rate`, `predicted_dt`, `check_step`,
  `dt_prev`, `rate_prev`, `savepoint`
- `trigger` (an `AtTime` / `EveryTime` / `EveryStep` / `WhenState`), `effect!`,
  `fired!`, `next_time`, `rewind!`, `landing_steps`,
  `switch!`/`switched` (a `SwitchableBC`)
- `writer` (a `FieldWriter`), `frame_prefix`, `collection`, `wall_io`,
  `piece` (one patch's block of one rank in a VTK dump; the multiblock
  writer names it `_patch_piece_name` and lists it in the `.vtm` through
  `_write_multiblock`), `_patch_slice` (a root plane in a patch's node
  space), `_vtk_ghost_points` / `VTK_HIDDENPOINT` (the blanking array)
- `n_art_fields`, `art_block`, `set_art_block!` (the artificial coefficient
  record a checkpoint carries beside `Q`; `max_rate` sizes the next step
  from it), `HierarchyRecord` / `LevelRecord` (what a checkpoint carries of
  the hierarchy), `hierarchy_record` (rank 0 assembles and broadcasts it),
  `restore_hierarchy!` (brings a solver to the recorded hierarchy),
  `_replace_level!` (rebuilds level 1 from a region list and owner ranges,
  every tile fresh), `_held_tiles` (this rank's `(level, tile, patch
  index)` triples)
- `region` (a `BlockRegion`: global offset plus extent, the patch-layout and
  HDF5 hyperslab unit and not a `Decomp`), `owned_region`,
  `hdf5_parallel`
- `patch` (a `Patch`: per-patch state split out of `Solver`), `patch_grid`,
  `patches`, `patch_regions`, `backend` (an `AbstractBackend`; `CPUBackend`),
  `interface_rhs` (`:extended` or `:onesided`), `div_plans` (divergence plans,
  = `deriv_plans` except at interface ends), `ghost_sends`/`ghost_recvs`/
  `plane_pairs`, `sync_patches!`, `eachpatch`; a routine below the step
  drivers takes a `SolverLike` (single-patch `Solver` or `PatchSolver`), and
  a single-patch `Solver` forwards patch-owned property names to its sole
  patch, so `solver.rho` and `solver.decomp` still read as before
- `rhs_workspace` (an `RHSWorkspace`: the scratch of one right-hand-side
  evaluation, `grad_u`, `grad_T_ion`, `grad_Y`, `strain_mag`, `sensor`,
  `sensor_sp`, `tmp_a`, `tmp_b`, `ring_buf`, and `flux`, held once per
  distinct padded local extent on a rank rather than once per patch, since a rank
  advances its patches in sequence and none of it outlives the evaluation
  that filled it; a level's tiles share one set), `rhs_workspace_pool` and
  `rhs_workspace!` (the pool and its lookup on that extent; a regrid seeds
  the pool from the sets its patches already hold, departing tiles
  included). The names reach callers through the same property forwarding, so
  `solver.grad_u` and `solver.tmp_a` read as before, and
  `_is_workspace_prop` is the branch that routes them
- `refine` (a `BlockRegion` in the parent level's node space, or a vector
  of them for a nested chain), `levels` (a `Vector{Level}`, root first;
  each `Level` holds `index`, `patches` as indices into `solver.patches`,
  `transfers`, one `LevelTransfer` per patch with `coarse_indices` /
  `fine_index` / `imposed`, and the level's own `ghost_sends` /
  `ghost_recvs` / `plane_pairs`), `tile` (the lattice edge in parent
  nodes; 0 = one patch per level), `_tile_span`/`_lattice_tile`/
  `_level_tiles`/`_tile_faces`, `_in_shell`, `phases` (the dimension-phased
  level sync), `_combine_planes!`/`_seed_planes!`, `_erode`, `refined_region`,
  `level_regions`, `nlevels`, `level_comm` (a `LevelComm`: the rank subset
  owning one level, holding its communicator, whether this rank is in it,
  whether the communicator was split for it, and its size; the root's spans
  the whole run, a level that fits every rank holds its parent's
  communicator, and no split exists in that case), `root_level_comm` /
  `absent_level_comm` / `split_level_comm` / `free_level_comm!`,
  `_level_ranks` (the largest rank count one tile admits under
  `_amr_dims`), `owners` (per tile, its rank range in the level's
  communicator) and `_tile_owners` (the ranges of a level's tiles, laid along
  a Morton curve by tile volume, through `_sfc_order`/`_morton` and
  `_rank_counts`), `group` (a `TileGroup`: this rank's group communicator,
  its rank range and whether it was split; one per rank per level, shared
  by the tiles of the range), `split_tile_comm` / `free_tile_group!` /
  `absent_tile_group`, `tiles` (the tile of each patch a rank holds on a
  level; a rank holds the root and the tiles of its own range only, so
  `LevelTransfer.fine_index` and `coarse_local` are local indices, 0 where
  the rank holds no piece), `level_restriction` (`:inject` or
  `:filter`), `prolong_level_ghosts!`, `restrict_level!`, `sync_levels!`,
  `_sync_level!`, `LEVEL_BUFFER`, `RESTRICT_MARGIN`; `Patch.h` is the
  patch's own spacing (h/3 on a level-1 patch), which the property
  forwarding serves as `solver.h`
- `_place_tiles` (the stored-ownership rule at a regrid: survivors keep
  their range in `Level.owners`, fresh tiles take the free ranks or join
  the nearest survivor's group), `rebalance` (the `Solver`/`Numerics`
  keyword and `RegridSpec` field: max/mean busy-time threshold, 0 off),
  `rebalance_persist` (keyword; `RegridSpec.persist`), `streak`,
  `imbalance`, `wall_mark`/`wait_mark`/`wall_regrid`, `_rebalance_due!`
  (the collective check; returns the Allgathered per-rank busy time when
  due), `_measured_weights`, `busy` (a rank's step wall less its waiting),
  `wall_wait`/`wait_total` (`Solver` fields: time inside the run-wide
  collectives and the level record exchange, charged through `_wait!`),
  `_migrate_tile!` (a moved tile's solution, old owners' blocks to new
  owners' blocks, point-to-point from the two Allgathered block tables;
  `old_piece` is the rank's old state and decomposition held past the
  swap for the sends), `_gather_tile`/`_carry_over!` (the box regrid's
  replicated carry, kept as the migration's reference), `MIGRATION_AUDIT`
  / `MIGRATION_AUDIT_RESULT` (the test hook comparing the two bitwise)
- `subcycle` (the Berger–Oliger mode flag), `subcycled_step!` and the
  recursive `_advance_level!` beneath it, `save_level_box!`/
  `hermite_level_shell!` (the Hermite box, `box_Q0` .. `box_dQ1`),
  `regrid` (a `RegridSpec`), `regrid_interval`, `tag_threshold`/
  `tag_buffer`, `tagged_region`, `regrid!`
- The tag criteria, a union evaluated by `_tag_sweep!` over the parent
  level's state, each a `pointwise!` body writing into `RegridSpec.tags`:
  `_tag_delta4_point!` (always on, `tag_threshold`), `_tag_sensor_point!`
  (`tag_sensor_threshold`, on the artificial diffusivity number
  `((μ* + β*)/ρ + κ*/(ρ c_p) + max D*)/(c h)` read from the persistent
  coefficient arrays, never from the pooled workspace's `sensor`),
  `_tag_gradient_point!` (`tag_gradient_threshold`, mass-fraction change
  per cell), `_tag_vorticity_point!` (`tag_vorticity_threshold`), and
  `_tag_predicate!` (`tag_predicate`, a user `(patch, I) -> Bool` closure).
  A body writes a level, `TAG_MARK` above the threshold or `TAG_HOLD` above
  the threshold over `untag_ratio` (keyword; `RegridSpec.untag_ratio`),
  and `_tag_level`/`_raise_tag!` combine them; the hold level keeps an
  existing tile (or the current box) and calls for no new one.
  `tile_lifetime` (keyword; `RegridSpec.lifetime`) is the minimum age of a
  tile in regrid checks, `checks` counts the checks and `created` (per
  current tile region, the check it was created at) is the tag history,
  derived from the reduced flags on every rank
- `covered` (a `Patch` field: per node, the orthants of its quadrature
  cell a child level covers, as bits; host `UInt8` over the padded
  extent, written by `_fill_covered!` at setup and at every regrid from
  the child regions every rank of the parent's level holds),
  `uncovered_fraction` / `uncovered_plane_fraction` (the volume and
  in-plane weights a mask byte gives), `_composite` (whether a solver's
  diagnostics take the multi-patch forms), `_composite_profile` /
  `_plane_accumulate!` (the composite plane average at the root's
  stations). A diagnostic's `Vector` form is the composite, masked one; the
  single-array form is the one-patch quadrature and applies no mask
- `pointwise!` (the shared launcher of every per-point loop: `Array` storage
  takes `@threaded`, device storage a KernelAbstractions kernel),
  `pointwise_ka!`, `FORCE_KA` (test/bench toggle), and the `_point!` suffix
  for a per-point body. A body takes plain arrays and scalars, never the
  solver, and never a `Type` argument: a `Type` inside the launcher's
  Vararg defeats specialization and turns the body call into a per-point
  runtime dispatch (measured 9× on `assemble_fluxes!`).
- `FieldVector`/`FieldMatrix` and `field_tuples` (the launchable forms of
  `Y`, `D_art`, `grad_u`, `grad_Y`, `flux`: zero-cost host wrappers that
  adapt to isbits `DeviceFieldVector`/`DeviceFieldMatrix` tuples at device
  launch). Never hand a bare `Vector`/`Matrix` of arrays to a device
  kernel (it hangs in kernel-argument adaptation instead of erroring),
  and never hold the tuple form on the host: runtime tuple indexing cost
  `assemble_fluxes!` 3× on the `@threaded` path when it was tried.
- `DeviceBackend` (an `AbstractBackend` wrapping a KernelAbstractions
  backend, behind `field(backend, decomp)` and `allocate_state`),
  `device_plan`/`DevicePlan` (device mirror of a `DirPlan`/`BandPlan`:
  fill, sweep, spike correction and scatter as KA kernels in a
  (lines × n) layout, one thread per line; the reduced interface stage
  stays host-side through the wrapped plan's `line_solver`, so the device
  method of `apply_along!` is collective, as the host one is), and
  `colwise` (dim 1 mirrors the `solve_col!` banded arithmetic so the
  KA-CPU comparison stays bitwise per dimension), `FORCE_DEVICE_EXCHANGE`
  (test toggle routing host arrays through every device-storage branch,
  which `_device_path` tests; `_cpu_storage` alone routes `pointwise!`)
- `StackedArray` (the stacked storage of a device level's tiles: one array,
  the tiles' padded blocks along the third dimension at a fixed `stride`,
  `ntiles` of them; tiles hold plain views, the level's spanning patch the
  wrapper, and a `pointwise!` routed on it launches every tile), `TileStack`
  / `Level.stacks` (the spanning `patch` and its `members`, one stack per
  padded extent), `_stack_state` (the members' shared state as one stacked
  `ConservedState`), `_stacked_level`, `_build_level_patches` /
  `_build_tile_stack` (the shared tile construction of the constructor, the
  tiled regrid and the restart), `_fine_decomp` / `_fine_plans` /
  `_patch_arrays` / `_assemble_patch`, `_view_workspace`, `_copy_tile!`,
  `_level_rhs!` / `_level_update!` / `_level_filter!` (the per-level phases
  the drivers run per patch or per stack), `padded_extent` (the box a
  full-array launch iterates; never `size` of an array that may be a stack),
  `lines_factor` (`plan_direction`'s line multiplier for a batched plan),
  `DevicePlan.ntiles` / `.stride`, `_tile_ranges`, `_upload!` / `_assign!`
- `LevelScratch` (a fine `Patch`'s device storage of its level transfer:
  the Hermite boxes and the chain `stages` over the rank's components;
  empty on the host backend), `BoxFill`/`HermiteFill` (the stage-0
  source of an imposition, `_fill_stage0!` host and `_fill_stage0_dev!`
  device), `_interpolate_dev!`/`_interp_point!`, `_ring_pack_point!`,
  `_upload_hermite!`, `_tag_rho_point!`
- `refresh_primitives!`, `mixture_density`, `boundary_plane` (the in-flight
  state-query API; primitives are stale inside a callback, see the
  `refresh_primitives!` docstring)
- `gidx` (interior indices → padded) and `interior_index` (its inverse); a
  padded index goes through the latter before reaching `xcoord`
- `validate_bc` (the setup-time boundary-condition hook), `unit_scalefactor`
- `outer_indices` (the flattened outer iteration space of a pointwise nest; see
  Threading), `prepared`/`primitives_current` (the trailing flag by which `run!`
  tells `step!` that the state has been exchanged and its primitives are current)

**Temperature is `T_ion`.** There is one temperature today; the name keeps
`T_ele` / `T_rad` free for a 2T or 3T model without a second API break. Bare
`T` is the element-type parameter and nothing else.

The following short names are conventions; do not "fix" them: `Q`, `dQ`, `du`, `d`
(dimension), `sp` (species), `I` (CartesianIndex), `σ`/`σf`/`σg` (parity
signs), `ξ` (computational coordinate), `h`, `c`, `p`, and `s` as a band offset
in `banded.jl` / `operators_banded.jl`, where it is the matrix-index convention
the surrounding comments define (`Ab[q+1+s, i]`).

## Traps

**MPI collectives cannot sit below an early return.** `deriv_along!` and
friends are distributed solves along a dimension, so *every* rank must call
them. A boundary routine that returns early on ranks not owning the wall plane
deadlocks. Both `correct_rhs!` methods in `nscbc.jl` hoist their collectives
above the `plane === nothing` return for this reason; follow that shape when
adding a boundary condition. The symptom is zero CPU on every rank, not a
crash, so it presents as a hang.

**A boundary condition that changes mid-run must change on every rank at the
same step.** This is the same collective trap from the other side: `SwitchableBC`
forwards to `after` only once switched. If `after` runs collectives that
`before` does not, as `NSCBCOutflowBC` does, then ranks disagreeing about the
switch is a deadlock, not a wrong answer. `WhenState` therefore reduces its
condition across the communicator; drive a switch from a `Callback` and never
from a rank-local test. `AtTime` and `EveryStep` are safe without a reduction
because `t` and `step` advance identically everywhere. The MPI suite pins this
with a condition that is true on one rank only; removing the reduction turns
that test into a hang, not a failure.

**Minimum local extent per dimension.** `plan_direction` errors when a rank's
block is too small for the scheme: C6 needs 5 points, C10 needs 7, and the C8
filter needs 9. The filter is the binding constraint, so a grid that is fine
for derivatives can still fail once filtering is on, and the test suite uses
transverse extents of 12 or 16 to clear it. Rank count multiplies this:
`n_global[d]` must stay ≥ 9 × `dims[d]`.

**Timing noise.** Run-to-run spread is easily 10–20% for an identical
configuration. Order-of-magnitude results are safe; few-percent ones are not
resolved by a single run. `bench/phases.jl` (per-phase budget) is more useful
than a flat sampling profile, which is dominated by the compact line solves and
shows little else.

**Julia soft scope.** A top-level `for` in a script that reassigns a variable
bound outside it throws `UndefVarError`. Wrap script bodies in a function.

**A nondeterministic crash or bitwise failure in the serial suite under Julia
1.12.7 on Windows may not be yours.** `reference/julia_codegen_bug_report.md`
diagnoses a concurrent-JIT race: three `EXCEPTION_ACCESS_VIOLATION` signatures
in the compile path, a crash line that moves between runs, and the `device line
solves` testset failing a bitwise comparison on a few of forty. Read it before
bisecting a change against a failure of that shape.

**A run that fails does not stop.** Losing positivity drives the diffusive rate
in `compute_dt` up until `dt` collapses, and the run then grinds forever at no
progress. A sweep that visits bad configurations must pass a low `nmax`, and
`run!` now applies `StepControl` floors so this raises `SolverFailure` instead.
`primitives!` substitutes benign placeholders wherever ρ ≤ 0, so the positivity
check in `max_rate` reads ρ out of `Q` directly and not from `solver.rho`.

## Threading

`@threaded <work> for ... end` (`src/threading.jl`) uses `Threads.@threads`
only when `work >= THREAD_MIN_WORK` (1024 points per thread × the session's
thread count; override the total via `CL_THREAD_MIN_WORK`) *and* the loop has
more than one trip; it runs serially otherwise. Allocation is
per region per thread, not per point, so without the threshold small cases pay
spawn cost for nothing.

Total work is not the whole test, because only the loop `@threaded` wraps is
divided. A nest threaded on its outermost index runs serially whenever the third
dimension is collapsed, which covers every pointwise loop of a planar
`(nx, ny, 1)` run. **A pointwise nest therefore iterates
`outer_indices(n2, n3)`**, a single flattened `CartesianIndices` over its two
outer indices unpacked with `j, k = Tuple(jk)`, which divides over the second
dimension instead. Write a new one that way. Iteration order is unchanged, so
this is not a numerics change.

The backend choice is tasks, not Polyester, and it rests on measurement. The
numbers and the reasoning are in the `@threaded` docstring; read it before reaching for
`@batch`, including the note on the conditions under which the decision should
be revisited.

## Conventions

- Keep source lines under ~95 characters. Four existing lines exceed it; don't
  add more.
- Comments explain *why*, and several encode a measurement or a derivation
  (`threading.jl` on the threading backend, `metric.jl` on the discrete GCL,
  `folds.jl` on the antipodal butterfly, `decomposition.jl` on the
  `Dims_create` signature). Move them with the code; don't delete the
  reasoning.
- Prose: avoid overuse of "Claudisms" including agency for inanimate things,
  setup-and-payoff, "what matters is/the thing that/is what makes/worth keeping",
  emphatic fragments, editorializing, and em-dash asides. Write clear academic
  prose that explains the package without managing the reader's emotional state.
  Most prose in this repository is model output. When an edit exposes poor style
  in the surrounding prose, follow these guidelines and flag it for correction.
- `bench/` is scratch tooling, not tests. Each script's header comment says
  what it measures and how to run it; read that before running one.
- Scripts take their settings from `ARGS`, not the environment: positional
  values first, then `key=value`, parsed by `script_args` (`src/scriptargs.jl`)
  against a defaults `NamedTuple` that doubles as the schema. Give a new script
  the same shape. An unknown key is an error: the environment
  form it replaced answered a typo by running the default for the length of the
  job and saying nothing. The two remaining `CL_*` variables
  (`CL_THREAD_MIN_WORK`, `CL_ERROR_BACKTRACE`) are read from inside the package,
  where there is no `ARGS` to consult.
- Wrap an MPI driver in `mpi_main` and report progress with `ProgressLog`;
  do not hand-roll either. An uncaught exception is 448 stacktraces at
  448 ranks; a hand-rolled progress loop got at least one of printing-from-one-
  rank, flushing, and not-timing-its-own-reduction wrong every time. Both
  docstrings carry the reasoning. Ctrl-C has no code fix; pass
  `--handle-signals=no` for sweeps.
- A sweep over configurations expected to include bad ones must pass a low
  `nmax`. A configuration that loses positivity does not crash: the diffusive
  rate in `compute_dt` climbs until `dt` collapses and the run grinds, so one
  bad point costs more wall time than the whole sweep. `test/cases.jl` threads
  an `nmax` through every case for this.
- The 1-D cases run single-threaded: each threaded region is well
  under `THREAD_MIN_WORK`, so `-t auto` buys nothing and ~4% CPU on a 24-thread
  box is the expected reading, not a bottleneck.
- Run artifacts (`*.dat`, `*.vtr`, `*.pvtr`, `*.ckpt`, `*.cov`) are gitignored.
  Prefer `git add <paths>` over `git add -A` after running examples.
- **Commit to `main`.** This is a single-maintainer repository and the default
  agent habit of opening a branch per change is noise here; do not create one
  unless asked. Commit only when asked, and run the gate above first.

## Known limitations

Each of these took time to establish. The measurements and the
rejected hypotheses are in the file named; read it before re-deriving any of
them.

- **A converging strong shock is CFL-limited at the symmetry cell**, not ahead
  of the front. Under the default `smoother = :gaussian` the ceilings are 0.4 at
  the spherical origin and 0.2 at both the cylindrical axis and the planar wall;
  `detector = :d8` lifts the latter two past 1.0 and costs the origin 0.4 →
  0.25. The restriction is a startup one, so `StepControl(retries = 4)` beats a
  globally lowered CFL. Every discretization-order explanation has been measured
  and ruled out, the fold closure included. The `cfl ≤ 0.15` figure recorded
  here previously is the retired `:compact` smoother's.
  → `reference/CALIBRATION.md`
- **κ\* is singular as `T_ion` → 0**, so a cold ambient below p ≈ 1e-3 collapses
  the diffusive timestep. `art_conductivity_scale` is an EOS dispatch point, so a
  tabular model can supply its own; the gas models still divide by `T_ion`.
- **The spherical origin fold is much less forgiving than the cylindrical axis.**
  It needs initial data resolved over ≳3 cells and will not take Noh's singular
  t = 0 start, both of which the axis handles. The cause remains unknown.
- **The compact filter, not the Cook artificial properties, holds this solver
  together, and it has never been calibrated.** At 128³ TGV the filter
  supplies 37% of the energy sink yet removing it kills the run, while removing
  the artificial properties entirely does not. So every `C_mu` number is
  conditional on `compact_filter(0.45)` applied every step, and `C_mu` itself is
  active but not yet fitted. → `reference/CALIBRATION.md`
- **`compute_artificial!` is 24.8–26.0% of the multicomponent RHS** under the
  default `:gaussian` smoother (the 31.8% figure is the retired `:compact` one),
  in the filter line-solves that smooth the sensors, one sweep per species. At
  `n_species == 2` that machinery is measurably a no-op (`D*_1` and `D*_2` agree
  to 4.8e-16), and
  it only earns its cost at three or more species. Cutting it is a numerics
  decision (shared vs per-species sensor), not a code tweak.
- The NASA CEA thermo and limited transport databases are bundled verbatim in `data/`
  with their Apache license and notice. `read_nasa9` handles multi-interval
  thermo records; the transport table is not yet connected to `Transport`.
- Cluster-side open questions (whether `ThreadPinning`'s pinning API would buy
  anything, and the unexplained ~4300x SMT-sibling collapse) are in
  `reference/CLUSTER.md`.
- Wanted: a `bench/` runner taking medians over repeated *processes*, to get
  under the 10–20% run-to-run spread.
