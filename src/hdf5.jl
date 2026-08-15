# Shared-file HDF5 output, and the block description it is phrased over.
#
# The implementation lives in a package extension (`ext/CompactLESHDF5Ext.jl`)
# so that the package keeps its dependency-free core. HDF5.jl requires a
# libhdf5 binary, and a *parallel* libhdf5 must additionally be built against
# the same MPI implementation the run uses. That is the same coupling
# `reference/CLUSTER.md` documents for MPI.jl itself, with the same failure
# mode: a mismatched build works until it does not. A cluster that cannot
# supply one must still be able to run the solver, so nothing here is loadable
# unless the caller writes `using HDF5`.
#
# WHY A SHARED FILE. The VTK path writes one file per rank per frame, which at
# 448 ranks and 100 frames is 44,800 files. That is a filesystem metadata
# problem before it is a bandwidth problem. A single file per dump also makes
# restart independent of the rank count, since the state is stored as one
# global array rather than as a set of per-rank blocks.
#
# TWO WRITE BACKENDS. With a parallel libhdf5 every rank writes its own
# hyperslab collectively, which is the configuration this exists for. Without
# one, HDF5 cannot be opened by more than one process at a time, so the ranks
# take turns: rank 0 creates the file and writes its block, then passes a token
# along the communicator, and each rank in turn opens the file, writes its
# block, and closes it. The result is byte-identical and the cost is O(P)
# serialized opens. It is intended for a workstation and for tests, not for a
# production run; `hdf5_parallel()` reports which backend is in use.

"""
    BlockRegion(offset, extent)

One rectangular block of a global array: its 0-based `offset` and its per-
dimension `extent`. This is the unit an HDF5 hyperslab is written from. It is
deliberately independent of [`Decomp`](@ref), so that the same write path serves
both a rank's owned block and, later, a refinement patch.
"""
struct BlockRegion
    offset::NTuple{3,Int}
    extent::NTuple{3,Int}
end

"The block of the global array owned by this rank."
owned_region(decomp::Decomp) =
    BlockRegion(ntuple(d -> decomp.offset[d], 3), ntuple(d -> decomp.n_local[d], 3))

"1-based index ranges of `region` within the global array, as an HDF5 hyperslab."
region_ranges(region::BlockRegion) =
    ntuple(d -> (region.offset[d]+1):(region.offset[d]+region.extent[d]), 3)

# --- Extension plumbing -----------------------------------------------------

_hdf5_extension() = Base.get_extension(@__MODULE__, :CompactLESHDF5Ext)

"""
    hdf5_available() -> Bool

Whether the HDF5 extension is loaded. It loads when the caller has HDF5.jl
available and writes `using HDF5`.
"""
hdf5_available() = _hdf5_extension() !== nothing

"""
    hdf5_parallel() -> Bool

Whether the loaded libhdf5 supports MPI-parallel I/O. `false` selects the
serialized write described at the top of `src/hdf5.jl`, which produces the same
file at O(P) cost. Errors when the extension is not loaded.
"""
function hdf5_parallel()
    ext = _hdf5_extension()
    ext === nothing && _hdf5_required("hdf5_parallel")
    return ext.has_parallel()
end

function _hdf5_required(name)
    error("$name requires the HDF5 extension. Add HDF5.jl to your environment " *
          "and write `using HDF5` alongside `using CompactLES`; the extension " *
          "loads itself. For MPI-parallel writes the libhdf5 binary must also " *
          "be built against the MPI this run uses — see the note at the top of " *
          "src/hdf5.jl.")
end

"""
    save_checkpoint_hdf5(solver, Q, prefix)

Write the interior of `Q` plus (t, step) to `prefix.h5` as one global array, and
return `prefix`. An existing file of that name is truncated. Collective.

Unlike [`save_checkpoint`](@ref), which writes one raw file per rank and can
only be restored onto the identical decomposition, this stores the state in
global index space. [`load_checkpoint_hdf5!`](@ref) therefore restores it onto
**any** rank count and process grid, so a run may be moved between machines or
resumed at a different scale.

The header describes the state well enough for the reader to reject a solver it
does not belong to: the format version, the global extent, the conserved count,
the species count, the conserved component names, the metric and EOS type names,
and the global coordinate vector along each dimension. The coordinates carry the
domain extent, the origin and any [`Stretch`](@ref) mapping, none of which the
extent alone constrains.

Requires `using HDF5`.
"""
save_checkpoint_hdf5(args...; kwargs...) = _hdf5_required("save_checkpoint_hdf5")

"""
    save_hdf5(solver, Q, prefix; fields = DEFAULT_VTK_FIELDS, stride = 1,
              slice = nothing)

Write a field dump to `prefix.h5` as one shared file, plus an XDMF3 sidecar
`prefix.xmf` written by rank 0 describing it, and return `prefix`. An existing
file of either name is truncated. Open the **`.xmf`** in ParaView or VisIt; the
`.h5` carries the data and no structure a reader can interpret on its own.
Collective.

`fields`, `stride` and `slice` mean what they do in [`save_vtk`](@ref), which
documents the available names, how points are selected for subsampling, and what
a slice writes. The grid follows the metric on the same rule: a resolved angular
dimension is written as a curvilinear mesh with explicit Cartesian positions and
rotated vectors, and every other grid as a rectilinear mesh with three
coordinate vectors.

This is the form to use at large rank counts. The VTK path writes one file per
rank per frame; this writes one file per frame regardless.

XDMF3 rather than VTKHDF: as of VTKHDF format version 2.5 that container
supports neither RectilinearGrid nor StructuredGrid, so a stretched or
curvilinear grid could only be expressed there as an unstructured mesh.

Requires `using HDF5`.
"""
save_hdf5(args...; kwargs...) = _hdf5_required("save_hdf5")

"""
    load_checkpoint_hdf5!(solver, Q, prefix)

Restore the interior of `Q` and (t, step) from the file written by
[`save_checkpoint_hdf5`](@ref), onto whatever decomposition `solver` has, and
return `Q`. Halos are left untouched. Every rank must call it, since each reads
its own block, but the read is independent and involves no communication.

The rank count and process grid need not match the ones that wrote the file, and
nothing else may differ. Every field of the header
[`save_checkpoint_hdf5`](@ref) records is compared against `solver` and any
mismatch throws: format version, global extent, conserved count, species count,
conserved component names, metric and EOS type names, and the coordinates along
each dimension. Coordinates are compared to a relative tolerance of 1e-10, which
separates a rebuilt identical grid from any different one.

Two limits apply. Species are identified by name, so two species sets that share
names while differing in their thermodynamic constants are not distinguished.
A checkpoint written in an earlier format is rejected rather than partially
validated.

Requires `using HDF5`.
"""
load_checkpoint_hdf5!(args...; kwargs...) = _hdf5_required("load_checkpoint_hdf5!")
