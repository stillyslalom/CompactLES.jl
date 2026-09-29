# Supported combinations

```@meta
CurrentModule = CompactLES
```

This page lists the configurations that [`setup`](@ref) and the low-level
[`Solver`](@ref) constructor accept, and the error each rejected combination
raises. The tables follow the checks in the code. The serial test suite
asserts every listed rejection, and a weekly test builds every accepted row
at a small size and advances it a few steps, so a change to a setup check
that this page does not follow fails one of the two.

"Device" means a [`DeviceBackend`](@ref) wrapping a KernelAbstractions
backend. The test suite runs the device path on the KernelAbstractions CPU
backend: the device kernels and plans run there, but GPU memory and
throughput are not measured. See [Run in parallel](@ref) for the hardware
caveats.

## Equations of state, backend and precision

| EOS | CPU | device | Float32 | patches and levels |
|:--|:--|:--|:--|:--|
| [`IdealMixture`](@ref), any species count | yes | yes | yes | yes |
| [`StiffenedGas`](@ref), one species | yes | yes | yes | yes |
| [`Nasa9Mixture`](@ref), any species count | yes | yes | yes | yes |
| a user subtype of [`EOS`](@ref) | yes | see below | constructed at the run's type | yes; a refined level with molecular transport takes `patch_interfaces = :closure` |

`Execution(precision = Float32)` in [`Numerics`](@ref) converts the built-in equations of
state, [`ConstantTransport`](@ref), [`CeaTransport`](@ref), the artificial
properties and the compact schemes. Any other component that carries a
floating-point type must be constructed at the run's type; one that is not
raises an `ArgumentError` naming it. Components of different types given
without `precision` raise an `ArgumentError` listing them.

A user EOS implements the methods listed in [Extending CompactLES](@ref). It
is passed to kernels unchanged, so on a GPU it must be an isbits type or
define an `Adapt.adapt_structure` method returning one, as the built-in
mixtures do. Setup does not check this; a non-isbits EOS on a GPU fails at the
first kernel launch.

## Geometry

| Geometry | device, Float32 | `patch_grid` | refinement | `polar_truncation` |
|:--|:--|:--|:--|:--|
| Cartesian, uniform | yes | yes | yes | no |
| Cartesian, stretched ([`Stretch`](@ref)) | yes | if the patched dimension is uniform, with `patch_interfaces = :closure` | no | no |
| Cylindrical, [`AxisBC`](@ref) at r = 0 | yes | no | θ collapsed; the first level may reach the axis | θ resolved over 2π, uniform r, host only |
| Cylindrical, annulus (no axis) | yes | θ collapsed, or with `patch_interfaces = :closure` | θ collapsed | θ resolved over 2π, uniform r, host only |
| Spherical, [`OriginBC`](@ref) and/or [`PoleBC`](@ref) | yes | no | no | no |
| [`SymmetryPlaneBC`](@ref) on a Cartesian face or a cylindrical z face | yes | no | no | as for the metric |

A stretched dimension must be non-periodic and cannot carry a fold (an axis,
origin, pole or symmetry plane). A resolved θ at the cylindrical axis, and a
resolved φ at the spherical origin or poles, needs an even point count over 2π.
The spherical origin requires a θ range symmetric about π/2 and the poles the
range (0, π). NSCBC faces require a unit scale factor in their normal
direction: any Cartesian face, a cylindrical r or z face, or a spherical r
face.

No grid node may lie on a coordinate singularity, where the volume Jacobian
vanishes. A radial domain that starts at r = 0 must close that end with
[`AxisBC`](@ref) or [`OriginBC`](@ref), and every θ node of a spherical grid
must lie strictly between 0 and π unless both θ ends carry [`PoleBC`](@ref);
these folds offset the nodes half a cell from the singularity. With θ
collapsed, the single θ node sits at the low end of the θ domain, so a domain
such as `(0, π)` places it on the pole.

## Patch and refinement layouts

| Layout | selected by | device | restrictions |
|:--|:--|:--|:--|
| one patch | default | yes | none |
| same-level slabs | `patch_grid` | yes | uniform Cartesian or a θ-collapsed cylindrical annulus, or a resolved-θ annulus or stretched grid under `patch_interfaces = :closure`, without folds or symmetry planes; tridiagonal filter; `:delta4` or `:species_d8` detector; no explicit `dims`; no refinement; no checkpoint |
| static nested levels | `AMR(initial = [shape, ...])` or a `BlockRegion` vector, `regrid_interval = 0` | yes | uniform Cartesian, or cylindrical with θ collapsed; a `BlockRegion` of the first level may reach a symmetry plane or the axis on the host backend, deeper levels staying inside it |
| one regridded box | `AMR(initial = ...)` with `regrid_interval > 0`, `tile = 0` | yes | as for static levels, with every level off the symmetry planes and the axis |
| regridded tiles, two levels | as above with `tile ≥ 3` | yes | as for static levels |
| regridded tiles, more than two levels | `tile ≥ 3` with `max_levels > 2` or a nested `BlockRegion` vector | no | host backend only; no `rebalance` |
| subcycled levels | `AMR(subcycle = true)` | yes | any refined layout |

`level_restriction = :filter` is accepted on the host backend of a serial
run only, and not with a level reaching a symmetry plane or the axis; the default
`:inject` has no restriction. At a patch or level
interface the default `PatchInterfaces(flux = :ghost)` requires
`rhs = :extended` and a uniform Cartesian grid or a uniform cylindrical grid
with θ collapsed, and with molecular
transport at a refined level one of the three built-in equations of state;
every other interface configuration requires `patch_interfaces = :closure`,
passed explicitly, and setup raises an `ArgumentError` rather than switching
to it. Without an interface the setting has no effect. `rebalance` requires a
tiled level with regridding and at most two levels.

## Checkpoints

The per-rank [`save_checkpoint`](@ref) restarts on the rank count and process
grid that wrote it; with HDF5 loaded, [`save_checkpoint_hdf5`](@ref) restarts
on any rank count. Both take the state of a refined solver as a vector and
record the hierarchy. A regridded hierarchy is rebuilt from the record, while
a static hierarchy of more than two levels must be constructed with the
recorded regions. A same-level `patch_grid` layout has no checkpoint.

A restart must match the element type, the equation of state, the species
and conserved layout, the metric, the global grid, the level count and the
number of artificial-coefficient fields. Differences in the numerics, the
transport model, the boundary conditions or the sources are refused unless
the load names the group in `allow`; see [Write output and restart](@ref).

## Setup errors

Each rejected combination raises at setup, before a state is allocated, except
the last three, which raise at the checkpoint call. The message contains the
text below.

| Combination | Error |
|:--|:--|
| `AxisBC` without `CylindricalMetric` | `AxisBC requires CylindricalMetric` |
| `OriginBC` without `SphericalMetric` | `OriginBC requires SphericalMetric` |
| `PoleBC` at one end of θ only | `PoleBC must be applied at both ends of θ` |
| `SymmetryPlaneBC` on a cylindrical r or θ face, or any spherical face | `SymmetryPlaneBC on dimension d requires CartesianMetric or the z dimension of CylindricalMetric` |
| NSCBC on an angular face | `the LODI formulation requires a face whose normal metric scale factor is one` |
| a stretched fold or symmetry plane | `folded dimensions cannot be stretched` |
| a stretched periodic dimension | `stretched dimensions must be non-periodic` |
| `patch_grid` with a fold | `patch decomposition across a coordinate fold is not supported` |
| `patch_grid` with a symmetry plane | `patch decomposition across a SymmetryPlaneBC is not supported` |
| `patch_grid` with a pentadiagonal filter | `patch interfaces carry closure variants for a tridiagonal filter only` |
| `patch_grid` with the `:d8` detector | `patch interfaces support the :delta4 and :species_d8 detectors only` |
| `patch_grid` with explicit `dims` | `an explicit process grid cannot combine with patch_grid` |
| `patch_grid` along a stretched dimension | `the patched dimension cannot be stretched` |
| `AMR` with `patch_grid` | `AMR: cannot be combined with a patch_grid` |
| `AMR` on a spherical metric or a resolved-θ cylindrical one | `AMR: requires CartesianMetric or CylindricalMetric with θ collapsed` |
| a refined region reaching a face other than a slip, no-slip, NSCBC, symmetry-plane or axis one, a `SwitchableBC` included | `a refined level cannot carry` |
| `AMR` on a stretched grid | `AMR: requires a uniform grid` |
| a level reaching a symmetry plane or the axis on a device | `a refined level reaching a SymmetryPlaneBC or an AxisBC runs on the host backend only` |
| a level reaching a symmetry plane or the axis with `level_restriction = :filter` | `a refined level reaching a SymmetryPlaneBC or an AxisBC takes level_restriction = :inject` |
| a level reaching a symmetry plane or the axis with regridding | `a refined level reaching a SymmetryPlaneBC or an AxisBC is placed at setup and is not regridded` |
| a nested level within the margin of a symmetry plane or the axis | `only the first refined level reaches a symmetry plane or the axis` |
| `AMR` regridding a vector of shapes | `AMR: regridding moves one refined level` |
| `max_levels > 2` without tiles and regridding | `requires tile > 0 and regridding` |
| more than one regridded level on a device | `regridding more than one refined level runs on the host backend only` |
| `level_restriction = :filter` on a device | `level_restriction = :filter is host-only` |
| `rebalance` without a tiled, regridded level | `rebalance repartitions a tiled level at the regrid cadence` |
| `interface_flux = :ghost`, the default, at an interface of a spherical, resolved-θ cylindrical or stretched grid | `interface_flux = :ghost (the default) requires an unstretched CartesianMetric, or CylindricalMetric with θ collapsed, at a patch or level interface` |
| `interface_flux = :ghost`, the default, with `interface_rhs = :onesided` | `interface_flux = :ghost (the default) reads the gradient plans' interface rows, which exist under interface_rhs = :extended only` |
| `interface_flux = :ghost`, the default, with molecular transport at a refined level and a user EOS | `interface_flux = :ghost (the default) with molecular transport at a refined level supports IdealMixture, Nasa9Mixture and StiffenedGas` |
| `polar_truncation` on a spherical or Cartesian metric | `polar_truncation applies to CylindricalMetric` |
| `polar_truncation` with θ collapsed or not over 2π | `polar_truncation requires θ resolved and periodic over 2π` |
| `polar_truncation` with a stretched radius | `polar_truncation requires an unstretched radial dimension` |
| `polar_truncation` with `patch_grid` or refinement | `polar_truncation takes a single patch without refinement` |
| `polar_truncation` on a device | `polar_truncation runs on the host backend only` |
| components of different floating-point types | `the solver components carry different floating-point types` |
| a `patch_grid` checkpoint | `a same-level patch layout (patch_grid) has no checkpoint` |
| a checkpoint loaded at another element type | `element type mismatch` |
| a checkpoint loaded under another EOS | `configuration mismatch` |
