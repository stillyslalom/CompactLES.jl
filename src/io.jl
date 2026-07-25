# Checkpoint / restart: per-rank raw binary files with a small validating
# header. Deliberately dependency-free (no HDF5); restart requires the same
# global grid, decomposition, and conserved layout. Parallel-aware
# postprocessing formats are a separate concern.

const CKPT_MAGIC = 0x434c4553_434b5054   # "CLES CKPT"

_ckpt_name(prefix::AbstractString, rank::Int) =
    string(prefix, ".r", lpad(rank, 4, '0'), ".ckpt")

"""
    save_checkpoint(s, Q, prefix)

Write the interior of `Q` plus (t, step) to `prefix.rNNNN.ckpt`, one file per
rank. Collective only in the trivial sense that every rank writes; no
communication is involved.
"""
function save_checkpoint(s::Solver, Q, prefix::AbstractString)
    decomp = s.decomp
    rank = MPI.Comm_rank(decomp.comm)
    o1, o2, o3 = decomp.n_halo_d
    nx, ny, nz = decomp.n_local
    open(_ckpt_name(prefix, rank), "w") do io
        write(io, UInt64(CKPT_MAGIC))
        write(io, Int64(s.n_cons))
        for d in 1:3
            write(io, Int64(decomp.n_global[d]))
        end
        for d in 1:3
            write(io, Int64(decomp.n_local[d]))
        end
        for d in 1:3
            write(io, Int64(decomp.coords[d]))
        end
        write(io, Float64(s.t))
        write(io, Int64(s.step))
        write(io, Q[o1+1:o1+nx, o2+1:o2+ny, o3+1:o3+nz, :])
    end
    return prefix
end

"""
    load_checkpoint!(s, Q, prefix)

Restore the interior of `Q` and (t, step) from `prefix.rNNNN.ckpt`, verifying
grid, decomposition, and conserved-layout compatibility.
"""
function load_checkpoint!(s::Solver, Q, prefix::AbstractString)
    decomp = s.decomp
    rank = MPI.Comm_rank(decomp.comm)
    o1, o2, o3 = decomp.n_halo_d
    nx, ny, nz = decomp.n_local
    open(_ckpt_name(prefix, rank), "r") do io
        read(io, UInt64) == CKPT_MAGIC || error("not a CompactLES checkpoint")
        Int(read(io, Int64)) == s.n_cons || error("conserved layout mismatch")
        for d in 1:3
            Int(read(io, Int64)) == decomp.n_global[d] || error("global grid mismatch")
        end
        for d in 1:3
            Int(read(io, Int64)) == decomp.n_local[d] || error("decomposition mismatch")
        end
        for d in 1:3
            Int(read(io, Int64)) == decomp.coords[d] || error("rank layout mismatch")
        end
        s.t = read(io, Float64)
        s.step = Int(read(io, Int64))
        buf = Array{Float64}(undef, nx, ny, nz, s.n_cons)
        read!(io, buf)
        Q[o1+1:o1+nx, o2+1:o2+ny, o3+1:o3+nz, :] .= buf
    end
    return Q
end

# ---------------------------------------------------------------------------
# VTK output: dependency-free parallel rectilinear (.vtr per rank + .pvtr
# container) with appended raw binary, Float32 payloads, physical coordinates
# (stretch mappings included). Adjacent pieces abut without ghost overlap —
# fine for point rendering and slicing; cell-based filters may show seams.

function _vtk_extent(decomp::Decomp)
    lo = ntuple(d -> decomp.offset[d], 3)
    hi = ntuple(d -> decomp.offset[d] + decomp.n_local[d] - 1, 3)
    lo, hi
end

_extent_str(lo, hi) = join(("$(lo[d]) $(hi[d])" for d in 1:3), " ")

"""
    save_vtk(s, Q, prefix)

Write `prefix.rNNNN.vtr` per rank plus `prefix.pvtr` (rank 0) containing ρ,
velocity, p, T_ion, and the mass fractions as Float32 point data on the physical
rectilinear grid. Open the .pvtr in ParaView/VisIt.
"""
function save_vtk(s::Solver, Q, prefix::AbstractString)
    decomp = s.decomp
    rank = MPI.Comm_rank(decomp.comm)
    P = MPI.Comm_size(decomp.comm)
    nx, ny, nz = decomp.n_local
    o1, o2, o3 = decomp.n_halo_d
    lo, hi = _vtk_extent(decomp)

    primitives!(s, Q)   # halos may be stale but the interior is what we write

    fields = Vector{Tuple{String,Int,Vector{Float32}}}()   # (name, ncomp, data)
    scal(name, a) = push!(fields, (name, 1,
        Float32[a[i+o1, j+o2, k+o3] for i in 1:nx, j in 1:ny, k in 1:nz][:]))
    scal("rho", s.rho); scal("p", s.p); scal("T_ion", s.T_ion)
    for sp in 1:s.n_species
        scal("Y$(sp)", s.Y[sp])
    end
    vel = Vector{Float32}(undef, 3 * nx * ny * nz)
    idx = 1
    for k in 1:nz, j in 1:ny, i in 1:nx
        vel[idx]   = s.u[i+o1, j+o2, k+o3]
        vel[idx+1] = s.v[i+o1, j+o2, k+o3]
        vel[idx+2] = s.w[i+o1, j+o2, k+o3]
        idx += 3
    end
    push!(fields, ("velocity", 3, vel))

    coords = ntuple(d -> Float64[xcoord(s, d, i) for i in 1:decomp.n_local[d]], 3)

    fname = string(prefix, ".r", lpad(rank, 4, '0'), ".vtr")
    open(fname, "w") do io
        offs = Int[]
        offset = 0
        blk(nbytes) = (push!(offs, offset); offset += 8 + nbytes)
        for d in 1:3
            blk(8 * length(coords[d]))
        end
        for (_, nc, data) in fields
            blk(4 * length(data))
        end
        write(io, "<?xml version=\"1.0\"?>\n")
        write(io, "<VTKFile type=\"RectilinearGrid\" version=\"1.0\" ",
                  "byte_order=\"LittleEndian\" header_type=\"UInt64\">\n")
        write(io, "<RectilinearGrid WholeExtent=\"", _extent_str(lo, hi), "\">\n")
        write(io, "<Piece Extent=\"", _extent_str(lo, hi), "\">\n<Coordinates>\n")
        for d in 1:3
            write(io, "<DataArray type=\"Float64\" Name=\"", "xyz"[d],
                  "\" format=\"appended\" offset=\"", string(offs[d]), "\"/>\n")
        end
        write(io, "</Coordinates>\n<PointData>\n")
        for (fi, (name, nc, _)) in enumerate(fields)
            write(io, "<DataArray type=\"Float32\" Name=\"", name,
                  "\" NumberOfComponents=\"", string(nc),
                  "\" format=\"appended\" offset=\"", string(offs[3+fi]), "\"/>\n")
        end
        write(io, "</PointData>\n</Piece>\n</RectilinearGrid>\n")
        write(io, "<AppendedData encoding=\"raw\">\n_")
        for d in 1:3
            write(io, UInt64(8 * length(coords[d])))
            write(io, coords[d])
        end
        for (_, _, data) in fields
            write(io, UInt64(4 * length(data)))
            write(io, data)
        end
        write(io, "\n</AppendedData>\n</VTKFile>\n")
    end

    # Gather every rank's piece extent so rank 0 can write the container.
    mine = Int64[lo..., hi...]
    allext = MPI.Allgather(mine, decomp.comm)
    if rank == 0
        open(string(prefix, ".pvtr"), "w") do io
            glo = ntuple(d -> 0, 3)
            ghi = ntuple(d -> decomp.n_global[d] - 1, 3)
            write(io, "<?xml version=\"1.0\"?>\n")
            write(io, "<VTKFile type=\"PRectilinearGrid\" version=\"1.0\" ",
                      "byte_order=\"LittleEndian\" header_type=\"UInt64\">\n")
            write(io, "<PRectilinearGrid WholeExtent=\"", _extent_str(glo, ghi),
                      "\" GhostLevel=\"0\">\n<PCoordinates>\n")
            for d in 1:3
                write(io, "<PDataArray type=\"Float64\" Name=\"", "xyz"[d], "\"/>\n")
            end
            write(io, "</PCoordinates>\n<PPointData>\n")
            for (name, nc, _) in fields
                write(io, "<PDataArray type=\"Float32\" Name=\"", name,
                      "\" NumberOfComponents=\"", string(nc), "\"/>\n")
            end
            write(io, "</PPointData>\n")
            for r in 0:(P-1)
                e = allext[6r+1:6r+6]
                plo = (e[1], e[2], e[3]); phi = (e[4], e[5], e[6])
                write(io, "<Piece Extent=\"", _extent_str(plo, phi),
                      "\" Source=\"", basename(string(prefix, ".r",
                      lpad(r, 4, '0'), ".vtr")), "\"/>\n")
            end
            write(io, "</PRectilinearGrid>\n</VTKFile>\n")
        end
    end
    return prefix
end
