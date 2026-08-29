module CompactLESHDF5Ext

# HDF5 implementation of the shared-file writes declared in src/hdf5.jl. Read
# the note at the top of that file first: it explains why this is an extension,
# and why there are two write backends.

using CompactLES
using CompactLES: BlockRegion, Decomp, Solver, owned_region, region_ranges
using CompactLES: axis_matches, global_axis, type_name
using CompactLES: ensure_output_dir, restore_switches!, switch_codes
using MPI
using HDF5

has_parallel() = HDF5.has_parallel()

# Format 2 added the species set, the metric and the grid coordinates to the
# checkpoint header, which format 1 has no record of. Format 3 added the element
# type of the state and the mutable run state (`cfl`, `dt_prev`, `rate_prev`,
# and the `switched` flag of each boundary face); the reasoning is at the top of
# `src/io.jl`. The reader requires an exact match and does not accept an older
# file, since a restart those fields do not cover cannot be validated at all.
const CKPT_FORMAT = 3

# --- Opening a shared file --------------------------------------------------
#
# `with_shared_file(body, path, mode, comm)` calls `body(file)` on every rank
# with a writable handle, under whichever backend the libhdf5 build supports.
#
# The serialized backend is a token relay. Rank 0 runs `body` first, because it
# is the rank that must create the file and its datasets; every other rank then
# waits for its predecessor, opens the existing file, runs `body`, and closes.
# Closing before passing the token makes this safe: a serial libhdf5 cannot
# have the file open in two processes at once, and the result is corruption,
# not a failure.
#
# An exception inside `body` is a collective problem, not a local one. A
# rank that threw before passing its token leaves its successor blocked in
# `Recv!` and its predecessors blocked in the closing collective, so the token
# moves from a `finally` and the failure is then reduced across the
# communicator. The ranks that succeeded also raise, avoiding a return into a
# communicator whose next collective they no longer agree on.
#
# The parallel backend cannot be made safe to the same degree. `h5open` on the
# MPI-IO driver and every dataset creation under it are themselves collective, so
# a rank throwing partway through `body` has diverged from the others and
# blocks them where they stand. The reduction below still covers a failure raised
# on every rank, or one raised after the last collective call in `body`.

# --- Transfer mode under the parallel backend --------------------------------
#
# Nothing here sets `dxpl_mpio = :collective`, so every hyperslab write below
# goes out under HDF5's default independent transfer mode. That is correct but
# not fast: a collective transfer lets the MPI-IO layer aggregate the per-rank
# hyperslabs into a few large contiguous writes, accounting for most of a
# shared write scale at high rank counts.
#
# Adopting it is a behavioural change, not a keyword. A collective
# transfer requires every rank of the file's communicator to call H5Dwrite on
# the same dataset in the same order, and the slice path does not: a rank
# holding no part of the requested plane writes nothing today and would instead
# have to issue a write with an empty selection. Making that change belongs on a
# machine with a parallel libhdf5 built against the run's MPI, which is the only
# place it can be exercised: `hdf5_parallel()` is false on a workstation, where
# the serialized relay above runs instead and no transfer property applies.

# The relay token's tag. Nothing else uses 0 on these communicators; the halo
# families start at 10 (see the tag note in src/halo.jl).
const TOKEN_TAG = 0

function with_shared_file(body, path::AbstractString, mode::AbstractString,
                          comm::MPI.Comm)
    if has_parallel()
        failure = nothing
        try
            # One collective open.
            h5open(path, mode, comm) do file
                body(file)
            end
        catch e
            failure = e
        end
        _raise_shared_failure(failure, path, comm)
        return path
    end
    rank = MPI.Comm_rank(comm)
    nranks = MPI.Comm_size(comm)
    token = Ref(0)
    rank == 0 || MPI.Recv!(token, comm; source=rank - 1, tag=TOKEN_TAG)
    # Rank 0 creates ("w" / "cw"), everyone after appends to what exists.
    local_mode = rank == 0 ? mode : (mode == "r" ? "r" : "r+")
    failure = nothing
    try
        h5open(path, local_mode) do file
            body(file)
        end
    catch e
        failure = e
    finally
        rank == nranks - 1 || MPI.Send(token, comm; dest=rank + 1, tag=TOKEN_TAG)
    end
    _raise_shared_failure(failure, path, comm)
    return path
end

# The reduction doubles as the barrier that used to close `with_shared_file`:
# every rank has finished with the file by the time it returns.
function _raise_shared_failure(failure, path::AbstractString, comm::MPI.Comm)
    anyfail = MPI.Allreduce(failure === nothing ? 0 : 1, max, comm)
    failure === nothing || throw(failure)
    anyfail == 0 ||
        error("with_shared_file: another rank failed on $path and raised the " *
              "cause; this rank's own write completed")
    return nothing
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
# Metadata carries the same value on every rank, but it cannot be written only
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
# therefore not detectable from the array: it reads cleanly and decodes to a
# different state. The header records everything the interpretation depends on, and
# `load_checkpoint_hdf5!` checks all of it.
#
# The reasoning for each field is at the top of `src/io.jl`, whose per-rank
# checkpoint records the same set for the same reasons, and `type_name`,
# `global_axis` and `axis_matches` are taken from there, not restated here, so
# that the two paths cannot drift apart.

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
    ensure_output_dir(prefix, comm)

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
            write_strings!(g, "eltype", [string(eltype(Q))], rank)
            # The mutable run state, for the reasons src/io.jl gives: a retry
            # lowers `cfl`, the growth cap and `filter_weight` read `dt_prev`
            # and `rate_prev`, and a boundary face that has switched must not
            # come back unswitched on any rank.
            write_meta!(g, "cfl", Float64(solver.cfl), rank)
            write_meta!(g, "dt_prev", Float64(solver.dt_prev), rank)
            write_meta!(g, "rate_prev", Float64(solver.rate_prev), rank)
            write_meta!(g, "switched", switch_codes(solver), rank)
            cg = create_group(file, "grid")
            for d in 1:3
                write_meta!(cg, "xyz"[d:d], global_axis(solver, d), rank)
            end
        end
        dims = (decomp.n_global..., n_cons)
        # `eltype(Q)`, not Float64: a Float32 solver would otherwise write a
        # widened copy that no longer round-trips bit for bit.
        dset = shared_dataset(file, "state/Q", eltype(Q), dims, comm)
        try
            write_region4!(dset, region, block, n_cons)
        finally
            close(dset)
        end
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
        stored_eltype = String(first(read(file["meta/eltype"])))
        stored_eltype == string(eltype(Q)) ||
            error("element type mismatch: file holds $stored_eltype, this " *
                  "state array holds $(eltype(Q))")
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
        solver.cfl = read(file["meta/cfl"])
        solver.dt_prev = read(file["meta/dt_prev"])
        solver.rate_prev = read(file["meta/rate_prev"])
        restore_switches!(solver, read(file["meta/switched"]), path)
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
# Vectors follow from the same rule. XDMF requires the component to vary fastest,
# so the Julia array must be (3, nx, ny, nz), component first, which lands on
# disk as nz x ny x nx x 3 and is declared "NZ NY NX 3". That is the
# interleaved layout `_interior_vector` produces, so the payload is reshaped,
# not permuted.

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
    ensure_output_dir(prefix, comm)

    with_shared_file(path, "w", comm) do file
        if has_parallel() || rank == 0
            g = create_group(file, "meta")
            write_meta!(g, "t", Float64(solver.t), rank)
            write_meta!(g, "step", Int64(solver.step), rank)
            write_meta!(g, "n_global", Int64[nglobal...], rank)
            write_meta!(g, "stride", Int64[st...], rank)
            write_meta!(g, "curvilinear", Int64(curvilinear), rank)
            if !curvilinear
                # The coordinate vectors are global, so every rank holds
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
            try
                if mine
                    # One position per point, component first, so the sidecar
                    # can point XDMF's XYZ geometry straight at it.
                    pts = Array{Float64}(undef, 3, nlocal...)
                    for (kk, k) in enumerate(ranges[3]),
                        (jj, j) in enumerate(ranges[2]),
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
            finally
                close(dset)
            end
        end
        for (name, ncomp, data) in entries
            dims = ncomp == 1 ? nglobal : (ncomp, nglobal...)
            dset = shared_dataset(file, "fields/" * name, Float32, dims, comm)
            try
                # A rank holding no part of the plane creates the dataset and
                # writes nothing into it, which is all the serialized backend
                # needs. Under the parallel backend the write is independent
                # and not collective; see the note on `dxpl_mpio` at the
                # head of this file.
                if mine
                    if ncomp == 1
                        write_region3!(dset, region, reshape(data, nlocal))
                    else
                        r = region_ranges(region)
                        dset[1:ncomp, r...] = reshape(data, ncomp, nlocal...)
                    end
                end
            finally
                close(dset)
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
