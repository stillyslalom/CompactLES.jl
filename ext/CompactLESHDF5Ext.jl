module CompactLESHDF5Ext

# HDF5 implementation of the shared-file writes declared in src/hdf5.jl. Read
# the note at the top of that file first: it explains why this is an extension,
# and why there are two write backends rather than one.

using CompactLES
using CompactLES: BlockRegion, Decomp, Solver, owned_region, region_ranges
using MPI
using HDF5

has_parallel() = HDF5.has_parallel()

const CKPT_FORMAT = 1

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

write_region3!(dset, region::BlockRegion, data) =
    (dset[region_ranges(region)...] = data)
write_region4!(dset, region::BlockRegion, data, ncomp::Int) =
    (dset[region_ranges(region)..., 1:ncomp] = data)
read_region3(dset, region::BlockRegion) = dset[region_ranges(region)...]
read_region4(dset, region::BlockRegion, ncomp::Int) =
    dset[region_ranges(region)..., 1:ncomp]

# --- Checkpoint / restart ---------------------------------------------------

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
            # Metadata is rank-independent, so under the serialized backend only
            # rank 0 writes it; under the parallel backend every rank writes the
            # same values collectively, which HDF5 requires for attributes.
            if rank == 0
                g = create_group(file, "meta")
                g["format"] = CKPT_FORMAT
                g["t"] = Float64(solver.t)
                g["step"] = Int64(solver.step)
                g["n_global"] = Int64[decomp.n_global...]
                g["n_cons"] = Int64(n_cons)
                g["n_species"] = Int64(solver.equations.n_species)
                g["component_names"] = solver.equations.component_names
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
        read(file["meta/format"]) == CKPT_FORMAT ||
            error("checkpoint format mismatch in $path")
        ng = read(file["meta/n_global"])
        Tuple(Int.(ng)) == Tuple(decomp.n_global) ||
            error("global grid mismatch: file has $(Tuple(Int.(ng))), " *
                  "solver has $(Tuple(decomp.n_global))")
        Int(read(file["meta/n_cons"])) == n_cons ||
            error("conserved layout mismatch: file has " *
                  "$(Int(read(file["meta/n_cons"]))), solver has $n_cons")
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

function _coarse_global(solver::Solver, stride)
    n = solver.decomp.n_global
    return ntuple(d -> length(1:stride[d]:n[d]), 3)
end

# Global coordinate vectors, identical on every rank, so no communication is
# needed to write them.
_coarse_axis(solver::Solver, d, stride) =
    Float64[CompactLES.global_xcoord(solver, d, g)
            for g in 1:stride[d]:solver.decomp.n_global[d]]

function CompactLES.save_hdf5(solver::Solver, Q, prefix::AbstractString;
                              fields=CompactLES.DEFAULT_VTK_FIELDS, stride=1)
    decomp = solver.decomp
    comm = decomp.comm
    rank = MPI.Comm_rank(comm)
    st = CompactLES._normalize_stride(stride)
    ranges = CompactLES._stride_ranges(solver, st)
    CompactLES._check_stride(solver, st, ranges)
    CompactLES._prepare_fields!(solver, Q, fields)
    curvilinear = CompactLES._curvilinear(solver)

    entries = Tuple{String,Int,Vector{Float32}}[]
    for name in fields
        append!(entries, CompactLES.vtk_field_entries(solver, Q, name,
                                                      curvilinear, ranges))
    end

    nglobal = _coarse_global(solver, st)
    nlocal = ntuple(d -> length(ranges[d]), 3)
    # Coarse-index offset of this rank's block, the same arithmetic the VTK
    # piece extents use.
    off = ntuple(d -> (decomp.offset[d] + first(ranges[d]) - 1) ÷ st[d], 3)
    region = BlockRegion(off, nlocal)
    path = string(prefix, ".h5")

    with_shared_file(path, "w", comm) do file
        if rank == 0
            g = create_group(file, "meta")
            g["t"] = Float64(solver.t)
            g["step"] = Int64(solver.step)
            g["n_global"] = Int64[nglobal...]
            g["stride"] = Int64[st...]
            g["curvilinear"] = Int64(curvilinear)
            if !curvilinear
                cg = create_group(file, "grid")
                for d in 1:3
                    cg["xyz"[d:d]] = _coarse_axis(solver, d, st)
                end
            end
        end
        if curvilinear
            # One position per point, component first, so the sidecar can point
            # XDMF's XYZ geometry straight at it.
            pts = Array{Float64}(undef, 3, nlocal...)
            for (kk, k) in enumerate(ranges[3]), (jj, j) in enumerate(ranges[2]),
                (ii, i) in enumerate(ranges[1])
                x, y, z = CompactLES._cartesian_position(solver.metric,
                    xcoord(solver, 1, i), xcoord(solver, 2, j), xcoord(solver, 3, k))
                pts[1, ii, jj, kk] = x
                pts[2, ii, jj, kk] = y
                pts[3, ii, jj, kk] = z
            end
            dset = shared_dataset(file, "grid/points", Float64, (3, nglobal...), comm)
            r = region_ranges(region)
            dset[1:3, r...] = pts
        end
        for (name, ncomp, data) in entries
            if ncomp == 1
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
