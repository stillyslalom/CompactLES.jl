# Field extraction and plotting

```@meta
CurrentModule = CompactLES
```

[`field_array`](@ref) resolves a named report variable to a padded array through the
internal [`scalar_field`](@ref) catalog, the same catalog `save_vtk` uses, and refreshes it from
`Q`. [`line_profile`](@ref), [`line_sample`](@ref), and [`field_slice`](@ref)
build on it: a profile is a collective, area-weighted average over the planes
transverse to one axis, a sample is the field on one grid line of that axis
(the general replacement for a hand-written sampling loop), and a slice is a
rank-0 gather of a transverse plane. On a patched or refined solver each
takes the state vector in place of `Q`, and a sample or a slice is taken at
root-grid nodes from the finest level holding each node.
[`cartesian_slice`](@ref) resamples a curvilinear slice onto a Cartesian raster,
and [`revolve_profile`](@ref) revolves a collapsed radial profile into a disk.
See the tutorials for worked cylindrical and spherical initializations.

[`field_snapshot`](@ref) gathers every interior node of the requested fields,
without halo padding, to rank 0 as a [`FieldSnapshot`](@ref): the coordinate
vectors of the grid and one array per field, for postprocessing or plotting a
desktop-scale run in memory. A refined or patch-partitioned solver gives one
snapshot per patch, each with its level, offset and the nodes a finer level
covers; `normal` and `index` restrict the gather to one plane, returned for
each patch at its own spacing. [`cartesian_coordinates`](@ref) maps a
curvilinear snapshot's nodes to Cartesian positions.

The plotting functions live in a package extension and require a Makie backend
(`using CairoMakie` or `using GLMakie`); [`makie_available`](@ref) reports
whether it is loaded. [`profileplot`](@ref) draws a `line_profile`, and
[`fieldheatmap`](@ref) draws a `field_slice`, resampling and using an equal
aspect for a curvilinear plane; on a refined run it draws each level at its own
spacing. [`meshplot`](@ref) draws the node-centered cells and patch outlines of
every level in the same plane and coordinates, so `meshplot!` overlays the mesh
on a heatmap.

```@docs
field_array
CompactLES.scalar_field
line_profile
line_sample
field_slice
field_snapshot
FieldSnapshot
cartesian_coordinates
cartesian_slice
revolve_profile
makie_available
profileplot
profileplot!
fieldheatmap
fieldheatmap!
meshplot
meshplot!
```
