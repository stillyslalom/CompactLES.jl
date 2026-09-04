# Operators and decomposition

```@meta
CurrentModule = CompactLES
```

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
Decomp
interior
field
allocate_state
exchange_halos!
CompactLES.exchange_state!
CompactLES.exchange_dim!
CompactLES.exchange_dim_batch!
CompactLES.selfwrap
AbstractBackend
CPUBackend
DeviceBackend
```

## Directional plans

```@docs
DirPlan
BandPlan
plan_direction
CompactLES.interface_closures
apply_along!
filter_field!
THREAD_MIN_WORK
CompactLES.THREAD_MIN_WORK_PER_THREAD
DevicePlan
device_plan
CompactLES.backend_plan
```

## Patches

A multi-patch solver holds one [`Patch`](@ref) per tile of the domain and
synchronizes them between Runge–Kutta stages. The compute routines are written
against the property surface that [`PatchSolver`](@ref) and a single-patch
`Solver` present identically, so they run unchanged in either configuration.

```@docs
Patch
PatchSolver
RHSWorkspace
rhs_workspace!
npatches
eachpatch
sync_patches!
exchange_patch_ghosts!
average_shared_planes!
```

## AMR level transfer

```@docs
amr_transfer_schemes
amr_restriction_scheme
amr_prolongation_scheme
amr_interpolation_weights
TransferPlan
plan_transfer
restrict!
prolong!
LevelTransfer
Level
LevelComm
TileGroup
CompactLES._tile_owners
nlevels
refined_region
level_regions
sync_levels!
prolong_level_ghosts!
restrict_level!
CompactLES.gather_region!
CompactLES.GatherBuffers
CompactLES.HierarchyRecord
CompactLES.LevelRecord
CompactLES.hierarchy_record
CompactLES.restore_hierarchy!
```
