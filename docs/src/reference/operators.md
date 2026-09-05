# Operators and decomposition

```@meta
CurrentModule = CompactLES
```

The compact-scheme definitions and the built-in scheme factories in the first
section are supported configuration API. The directional plans, decomposition
records, patch records, and AMR transfer machinery in the sections after it
are implementation-level building blocks: they remain documented for advanced
users and cross-references, but are not exported as input-deck vocabulary.

## Scheme definitions

```@docs
ClosureRow
CompactScheme
BandedClosureRow
BandedCompactScheme
pade_d1_4
lele_d1_6
lele_d1_8
lele_d1_10
compact_filter
gaussian_filter
compact_d8
```

## Decomposition and storage

Field allocation is routed through a storage backend, so the same setup code
can place arrays in device memory. [`CPUBackend`](@ref) is the default;
[`DeviceBackend`](@ref) allocates on a KernelAbstractions backend, and a
whole solver built on one runs resident there (`reference/AMR_GPU.md`).

```@docs
CompactLES.Decomp
CompactLES.interior
CompactLES.field
allocate_state
CompactLES.exchange_halos!
CompactLES.exchange_state!
CompactLES.exchange_dim!
CompactLES.exchange_dim_batch!
CompactLES.selfwrap
CompactLES.AbstractBackend
CPUBackend
DeviceBackend
```

## Directional plans

```@docs
CompactLES.DirPlan
CompactLES.BandPlan
CompactLES.plan_direction
CompactLES.interface_closures
CompactLES.apply_along!
CompactLES.filter_field!
CompactLES.THREAD_MIN_WORK
CompactLES.THREAD_MIN_WORK_PER_THREAD
CompactLES.DevicePlan
CompactLES.device_plan
CompactLES.backend_plan
```

## Patches

A multi-patch solver holds one [`Patch`](@ref) per tile of the domain and
synchronizes them between Runge–Kutta stages. The compute routines are written
against the property surface that [`PatchSolver`](@ref) and a single-patch
`Solver` present identically, so they run unchanged in either configuration.

```@docs
CompactLES.Patch
CompactLES.PatchSolver
CompactLES.RHSWorkspace
CompactLES.rhs_workspace!
npatches
eachpatch
sync_patches!
CompactLES.exchange_patch_ghosts!
CompactLES.average_shared_planes!
```

## AMR level transfer

```@docs
CompactLES.amr_transfer_schemes
CompactLES.amr_restriction_scheme
CompactLES.amr_prolongation_scheme
CompactLES.amr_interpolation_weights
CompactLES.TransferPlan
CompactLES.plan_transfer
CompactLES.restrict!
CompactLES.prolong!
CompactLES.LevelTransfer
CompactLES.Level
CompactLES.LevelComm
CompactLES.TileGroup
CompactLES._tile_owners
nlevels
refined_region
level_regions
sync_levels!
CompactLES.prolong_level_ghosts!
CompactLES.restrict_level!
CompactLES.gather_region!
CompactLES.GatherBuffers
CompactLES.HierarchyRecord
CompactLES.LevelRecord
CompactLES.hierarchy_record
CompactLES.restore_hierarchy!
```
