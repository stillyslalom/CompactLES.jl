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
lele_d1_10
compact_filter
gaussian_filter
compact_d8
```

## Decomposition and storage

```@docs
Decomp
interior
field
allocate_state
exchange_halos!
```

## Directional plans

```@docs
DirPlan
BandPlan
apply_along!
filter_field!
THREAD_MIN_WORK
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
```
