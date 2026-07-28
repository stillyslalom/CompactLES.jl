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

end # module
