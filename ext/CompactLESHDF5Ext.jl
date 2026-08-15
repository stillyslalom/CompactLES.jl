module CompactLESHDF5Ext

# HDF5 implementation of the shared-file writes declared in src/hdf5.jl. Read
# the note at the top of that file first: it explains why this is an extension,
# and why there are two write backends rather than one.

using CompactLES
using CompactLES: BlockRegion, Decomp, Solver, owned_region, region_ranges
using MPI
using HDF5

has_parallel() = HDF5.has_parallel()

# Format 2 added the species set, the metric and the grid coordinates to the
# checkpoint header, which format 1 has no record of. The reader requires an
# exact match rather than accepting the older file, since the point of those
# fields is that a restart they do not cover cannot be validated at all.
const CKPT_FORMAT = 2

# --- Opening a shared file --------------------------------------------------
#
# `with_shared_file(body, path, mode, comm)` calls `body(file)` on every rank
# with a writable handle, under whichever backend the libhdf5 build supports.
#
# The serialized backend is a token relay. Rank 0 runs `body` first, because it
# is the rank that must create the file and its datasets; every other rank then
# waits for its predecessor, opens the existing file, runs `body`, and closes.
# Closing before passing the token is what makes this safe: two processes with
# the file open at once is precisely what a serial libhdf5 cannot do, and it
# corrupts rather than fails.

function with_shared_file(body, path::AbstractString, mode::AbstractString,
                          comm::MPI.Comm)
    if has_parallel()
        # One collective open. `dxpl_mpio=:collective` on the writes themselves
        # is set at the call sites, where the dataset is known.
        h5open(path, mode, comm) do file
            body(file)
        end
        MPI.Barrier(comm)
        return path
    end
    rank = MPI.Comm_rank(comm)
    size = MPI.Comm_size(comm)
    token = Ref(0)
    rank == 0 || MPI.Recv!(token, comm; source=rank - 1, tag=0)
    # Rank 0 creates ("w" / "cw"), everyone after appends to what exists.
    local_mode = rank == 0 ? mode : (mode == "r" ? "r" : "r+")
    h5open(path, local_mode) do file
        body(file)
    end
    rank == size - 1 || MPI.Send(token, comm; dest=rank + 1, tag=0)
    MPI.Barrier(comm)
    return path
end

# A dataset covering the whole global array, created once and written in
# per-rank pieces. Under the serialized backend only rank 0 creates it.
function shared_dataset(file, name::AbstractString, ::Type{T}, dims,
                        comm::MPI.Comm) where {T}
    if has_parallel() || MPI.Comm_rank(comm) == 0
        return create_dataset(file, name, datatype(T), dataspace(dims))
    end
    return file[name]
end

# --- Rank-independent metadata ----------------------------------------------
#
# Metadata carries the same value on every rank, but it cannot simply be written
# from rank 0 under the parallel backend: that file is open on the MPI-IO
# driver, where creating a group or a dataset is collective and a rank skipping
# one leaves the ranks with divergent file structure. So every rank reaches
# every creation call, and the write that follows is independent and done by
# rank 0 alone. Under the serialized backend rank 0 holds the file by itself and
# both halves are its own.

function write_meta!(g, name::AbstractString, value, rank::Int)
    dtype = datatype(value)
    dset = create_dataset(g, name, dtype, dataspace(value))
    rank == 0 && write_dataset(dset, dtype, value)
    close(dset)
    return nothing
end

# A Julia String maps to a variable-length HDF5 datatype, and parallel HDF5
# refuses to write one ("Parallel IO does not support writing VL or region
# reference datatypes yet"), so string metadata goes out as fixed-length
# null-padded records. HDF5.jl strips the padding on read and returns String.
function write_strings!(g, name::AbstractString, strs, rank::Int)
    width = maximum(ncodeunits, strs; init=1)
    dtype = HDF5.Datatype(HDF5.API.h5t_copy(HDF5.API.H5T_C_S1))
    HDF5.API.h5t_set_size(dtype, width)
    HDF5.API.h5t_set_strpad(dtype, HDF5.API.H5T_STR_NULLPAD)
    HDF5.API.h5t_set_cset(dtype, HDF5.API.H5T_CSET_UTF8)
    dset = create_dataset(g, name, dtype, dataspace((length(strs),)))
    if rank == 0
        buf = zeros(UInt8, width, length(strs))
        for (i, s) in enumerate(strs)
            copyto!(view(buf, 1:ncodeunits(s), i), codeunits(s))
        end
        write_dataset(dset, dtype, buf)
    end
    close(dset)
    return nothing
end

write_region3!(dset, region::BlockRegion, data) =
    (dset[region_ranges(region)...] = data)
write_region4!(dset, region::BlockRegion, data, ncomp::Int) =
    (dset[region_ranges(region)..., 1:ncomp] = data)
read_region3(dset, region::BlockRegion) = dset[region_ranges(region)...]
read_region4(dset, region::BlockRegion, ncomp::Int) =
    dset[region_ranges(region)..., 1:ncomp]

# --- Checkpoint / restart ---------------------------------------------------
#
# WHAT THE HEADER HAS TO PIN DOWN. The state is one flat global array of
# conserved components, and nothing in `state/Q` records what those components
# mean or where the points are. A restart onto a solver that disagrees is
# therefore not detectable from the array: it reads cleanly and means something
# else. The header records everything the interpretation depends on, and
# `load_checkpoint_hdf5!` checks all of it.
#
# The conserved layout needs the species set, not just `n_cons`. Two species
# sets of the same size give the same `n_cons`, so only `component_names`
# separates them, and it covers the momentum and energy slots at the same time.
#
# The grid needs the coordinates, not the extent. `n_global` alone admits a
# different domain length, a different origin, and any `Stretch` mapping; the
# global coordinate vector along each dimension pins all three at once. It is
# also the only form a `Stretch` can be stored in, being a pair of closures.
# Three vectors of `n_global[d]` doubles are negligible beside the state.
#
# The metric and the EOS are stored by type name. For the metric that is a
# complete description, the types being singletons, and the scale factors it
# implies are already reflected in the coordinates. For the EOS it is a coarse
# check that catches a different model at the same `n_cons`; the species
# themselves are identified by the names in `component_names`, so two species
# sets sharing names but differing in their thermodynamic constants are not
# distinguished. Recording those would need a serialization contract every EOS
# implements, which is a larger change than this.

type_name(x) = string(nameof(typeof(x)))

# Global coordinate vector along `d`, identical on every rank, so writing it
# needs no communication.
global_axis(solver::Solver, d::Int) =
    Float64[CompactLES.global_xcoord(solver, d, i)
            for i in 1:solver.decomp.n_global[d]]

# Compared with a tolerance rather than exactly: the stored vector and the one
# rebuilt here come from the same expression, but a `Stretch` mapping evaluated
# on a different machine or Julia version may land a few ulp away, and a grid
# that genuinely differs does so by orders of magnitude more than this.
function axis_matches(stored, mine)
    length(stored) == length(mine) || return false
    scale = max(1.0, maximum(abs, mine; init=0.0))
    return maximum(abs.(stored .- mine); init=0.0) <= 1e-10 * scale
end

function CompactLES.save_checkpoint_hdf5(solver::Solver, Q, prefix::AbstractString)
    decomp = solver.decomp
    comm = decomp.comm
    rank = MPI.Comm_rank(comm)
    n_cons = solver.equations.n_cons
    region = owned_region(decomp)
    o1, o2, o3 = decomp.n_halo_d
    nx, ny, nz = decomp.n_local
    block = Q[o1+1:o1+nx, o2+1:o2+ny, o3+1:o3+nz, :]
    path = string(prefix, ".h5")

    with_shared_file(path, "w", comm) do file
        if has_parallel() || rank == 0
            g = create_group(file, "meta")
            write_meta!(g, "format", CKPT_FORMAT, rank)
            write_meta!(g, "t", Float64(solver.t), rank)
            write_meta!(g, "step", Int64(solver.step), rank)
            write_meta!(g, "n_global", Int64[decomp.n_global...], rank)
            write_meta!(g, "n_cons", Int64(n_cons), rank)
            write_meta!(g, "n_species", Int64(solver.equations.n_species), rank)
            write_strings!(g, "component_names",
                           solver.equations.component_names, rank)
            write_strings!(g, "metric", [type_name(solver.metric)], rank)
            write_strings!(g, "eos", [type_name(solver.eos)], rank)
            cg = create_group(file, "grid")
            for d in 1:3
                write_meta!(cg, "xyz"[d:d], global_axis(solver, d), rank)
            end
        end
        dims = (decomp.n_global..., n_cons)
        dset = shared_dataset(file, "state/Q", Float64, dims, comm)
        write_region4!(dset, region, block, n_cons)
    end
    return prefix
end

function CompactLES.load_checkpoint_hdf5!(solver::Solver, Q, prefix::AbstractString)
    decomp = solver.decomp
    comm = decomp.comm
    n_cons = solver.equations.n_cons
    region = owned_region(decomp)
    o1, o2, o3 = decomp.n_halo_d
    nx, ny, nz = decomp.n_local
    path = string(prefix, ".h5")

    # Read is not a write, so every rank may open the file at once even without
    # a parallel build.
    h5open(path, "r") do file
        fmt = Int(read(file["meta/format"]))
        fmt == CKPT_FORMAT ||
            error("checkpoint format mismatch in $path: file is format $fmt, " *
                  "this version writes and reads format $CKPT_FORMAT. A file " *
                  "written before format $CKPT_FORMAT carries no grid or metric " *
                  "record and cannot be validated; rerun to regenerate it.")
        ng = read(file["meta/n_global"])
        Tuple(Int.(ng)) == Tuple(decomp.n_global) ||
            error("global grid mismatch: file has $(Tuple(Int.(ng))), " *
                  "solver has $(Tuple(decomp.n_global))")
        Int(read(file["meta/n_cons"])) == n_cons ||
            error("conserved layout mismatch: file has " *
                  "$(Int(read(file["meta/n_cons"]))), solver has $n_cons")
        # `n_cons` alone does not identify the conserved layout: two species
        # sets of the same size agree on it and mean different things.
        nsp = Int(read(file["meta/n_species"]))
        nsp == solver.equations.n_species ||
            error("species count mismatch: file has $nsp, solver has " *
                  "$(solver.equations.n_species)")
        names = String.(read(file["meta/component_names"]))
        names == solver.equations.component_names ||
            error("conserved component mismatch: file has $names, solver has " *
                  "$(solver.equations.component_names)")
        stored_metric = String(first(read(file["meta/metric"])))
        stored_metric == type_name(solver.metric) ||
            error("metric mismatch: file has $stored_metric, solver has " *
                  "$(type_name(solver.metric))")
        stored_eos = String(first(read(file["meta/eos"])))
        stored_eos == type_name(solver.eos) ||
            error("equation of state mismatch: file has $stored_eos, solver " *
                  "has $(type_name(solver.eos))")
        # The coordinates carry the domain extent, the origin and any `Stretch`,
        # none of which `n_global` constrains.
        for d in 1:3
            axis_matches(read(file["grid/" * "xyz"[d:d]]), global_axis(solver, d)) ||
                error("grid coordinate mismatch on dimension $d: the stored " *
                      "coordinates differ from this solver's, so the domain " *
                      "extent, the origin or a Stretch mapping is not the one " *
                      "the checkpoint was written on")
        end
        solver.t = read(file["meta/t"])
        solver.step = Int(read(file["meta/step"]))
        block = read_region4(file["state/Q"], region, n_cons)
        Q[o1+1:o1+nx, o2+1:o2+ny, o3+1:o3+nz, :] .= block
    end
    return Q
end

# --- Field dump, with an XDMF3 sidecar --------------------------------------
#
# Array layout is the whole of the care needed here. Julia is column-major and
# HDF5 is row-major, and HDF5.jl bridges them by reversing the dimension list on
# disk: a Julia (nx, ny, nz) dataset is reported by h5dump as nz x ny x nx.
# XDMF reads in the row-major convention, so a scalar's Dimensions are written
# "NZ NY NX".
#
# Vectors follow from the same rule. XDMF wants the component to vary fastest,
# so the Julia array must be (3, nx, ny, nz) — component first — which lands on
# disk as nz x ny x nx x 3 and is declared "NZ NY NX 3". That is exactly the
# interleaved layout `_interior_vector` already produces, so the payload is
# reshaped rather than permuted.

const XDMF_SCALAR = "Scalar"
const XDMF_VECTOR = "Vector"

_xdmf_dims(n) = string(n[3], " ", n[2], " ", n[1])

# Global coordinate vectors, identical on every rank, so no communication is
# needed to write them. A sliced dimension contributes its single coordinate.
function _coarse_axis(solver::Solver, d, stride, slice)
    n = solver.decomp.n_global[d]
    if slice !== nothing && Int(slice[1]) == d
        return Float64[CompactLES.global_xcoord(solver, d, Int(slice[2]))]
    end
    return Float64[CompactLES.global_xcoord(solver, d, g) for g in 1:stride[d]:n]
end

function CompactLES.save_hdf5(solver::Solver, Q, prefix::AbstractString;
                              fields=CompactLES.DEFAULT_VTK_FIELDS, stride=1,
                              slice=nothing)
    decomp = solver.decomp
    comm = decomp.comm
    rank = MPI.Comm_rank(comm)
    st = CompactLES._normalize_stride(stride)
    ranges = CompactLES._output_ranges(solver, st, slice)
    CompactLES._check_output(solver, st, slice, ranges)
    CompactLES._prepare_fields!(solver, Q, fields)
    curvilinear = CompactLES._curvilinear(solver)

    entries = Tuple{String,Int,Vector{Float32}}[]
    for name in fields
        append!(entries, CompactLES.vtk_field_entries(solver, Q, name,
                                                      curvilinear, ranges))
    end

    nglobal = CompactLES._output_global(solver, st, slice)
    nlocal = ntuple(d -> length(ranges[d]), 3)
    # Coarse-index offset of this rank's block, the same arithmetic the VTK
    # piece extents use. Empty on a rank holding no part of a slice.
    off, _ = CompactLES._output_piece_extent(solver, st, slice, ranges)
    mine = CompactLES._has_output(ranges)
    region = BlockRegion(off, nlocal)
    path = string(prefix, ".h5")

    with_shared_file(path, "w", comm) do file
        if has_parallel() || rank == 0
            g = create_group(file, "meta")
            write_meta!(g, "t", Float64(solver.t), rank)
            write_meta!(g, "step", Int64(solver.step), rank)
            write_meta!(g, "n_global", Int64[nglobal...], rank)
            write_meta!(g, "stride", Int64[st...], rank)
            write_meta!(g, "curvilinear", Int64(curvilinear), rank)
            if !curvilinear
                # The coordinate vectors are global, so every rank already holds
                # the same values and none of this needs communication.
                cg = create_group(file, "grid")
                for d in 1:3
                    write_meta!(cg, "xyz"[d:d],
                                _coarse_axis(solver, d, st, slice), rank)
                end
            end
        end
        if curvilinear
            # The dataset is created whether or not this rank holds part of the
            # plane, for the reason the field loop below gives: rank 0 may be one
            # of the ranks with nothing to write, and every rank must reach the
            # creation call under the parallel backend.
            dset = shared_dataset(file, "grid/points", Float64, (3, nglobal...), comm)
            if mine
                # One position per point, component first, so the sidecar can
                # point XDMF's XYZ geometry straight at it.
                pts = Array{Float64}(undef, 3, nlocal...)
                for (kk, k) in enumerate(ranges[3]), (jj, j) in enumerate(ranges[2]),
                    (ii, i) in enumerate(ranges[1])
                    x, y, z = CompactLES._cartesian_position(solver.metric,
                        xcoord(solver, 1, i), xcoord(solver, 2, j),
                        xcoord(solver, 3, k))
                    pts[1, ii, jj, kk] = x
                    pts[2, ii, jj, kk] = y
                    pts[3, ii, jj, kk] = z
                end
                r = region_ranges(region)
                dset[1:3, r...] = pts
            end
        end
        for (name, ncomp, data) in entries
            if !mine
                # No part of the plane on this rank. Under the serialized
                # backend that is simply nothing to write. A collective write
                # would instead need every rank to call H5Dwrite with an empty
                # selection; see the slicing note in reference/ROADMAP.md.
                shared_dataset(file, "fields/" * name, Float32,
                               ncomp == 1 ? nglobal : (ncomp, nglobal...), comm)
            elseif ncomp == 1
                dset = shared_dataset(file, "fields/" * name, Float32, nglobal, comm)
                write_region3!(dset, region, reshape(data, nlocal))
            else
                dset = shared_dataset(file, "fields/" * name, Float32,
                                      (ncomp, nglobal...), comm)
                r = region_ranges(region)
                dset[1:ncomp, r...] = reshape(data, ncomp, nlocal...)
            end
        end
    end

    rank == 0 && _write_xdmf(string(prefix, ".xmf"), basename(path), solver,
                             nglobal, entries, curvilinear)
    MPI.Barrier(comm)
    return prefix
end

function _write_xdmf(path, h5name, solver::Solver, nglobal, entries,
                     curvilinear::Bool)
    dims = _xdmf_dims(nglobal)
    open(path, "w") do io
        write(io, "<?xml version=\"1.0\" ?>\n")
        write(io, "<!DOCTYPE Xdmf SYSTEM \"Xdmf.dtd\" []>\n")
        write(io, "<Xdmf Version=\"3.0\">\n <Domain>\n")
        write(io, "  <Grid Name=\"mesh\" GridType=\"Uniform\">\n")
        write(io, "   <Time Value=\"", string(Float64(solver.t)), "\"/>\n")
        if curvilinear
            write(io, "   <Topology TopologyType=\"3DSMesh\" Dimensions=\"",
                      dims, "\"/>\n")
            write(io, "   <Geometry GeometryType=\"XYZ\">\n")
            write(io, "    <DataItem Dimensions=\"", dims, " 3\" NumberType=\"Float\" ",
                      "Precision=\"8\" Format=\"HDF\">", h5name, ":/grid/points",
                      "</DataItem>\n   </Geometry>\n")
        else
            write(io, "   <Topology TopologyType=\"3DRectMesh\" Dimensions=\"",
                      dims, "\"/>\n")
            write(io, "   <Geometry GeometryType=\"VXVYVZ\">\n")
            for d in 1:3
                write(io, "    <DataItem Dimensions=\"", string(nglobal[d]),
                          "\" NumberType=\"Float\" Precision=\"8\" Format=\"HDF\">",
                          h5name, ":/grid/", "xyz"[d:d], "</DataItem>\n")
            end
            write(io, "   </Geometry>\n")
        end
        for (name, ncomp, _) in entries
            kind = ncomp == 1 ? XDMF_SCALAR : XDMF_VECTOR
            shape = ncomp == 1 ? dims : string(dims, " ", ncomp)
            write(io, "   <Attribute Name=\"", name, "\" AttributeType=\"", kind,
                      "\" Center=\"Node\">\n")
            write(io, "    <DataItem Dimensions=\"", shape, "\" NumberType=\"Float\" ",
                      "Precision=\"4\" Format=\"HDF\">", h5name, ":/fields/", name,
                      "</DataItem>\n   </Attribute>\n")
        end
        write(io, "  </Grid>\n </Domain>\n</Xdmf>\n")
    end
    return path
end

end # module
