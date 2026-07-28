# Checkpoint / restart: per-rank raw binary files with a small validating
# header. Deliberately dependency-free (no HDF5); restart requires the same
# global grid, decomposition, and conserved layout. Parallel-aware
# postprocessing formats are a separate concern.

const CKPT_MAGIC = 0x434c4553_434b5054   # "CLES CKPT"

_ckpt_name(prefix::AbstractString, rank::Int) =
    string(prefix, ".r", lpad(rank, 4, '0'), ".ckpt")

"""
    save_checkpoint(solver, Q, prefix)

Write the interior of `Q` plus (t, step) to `prefix.rNNNN.ckpt`, one file per
rank. Collective only in the trivial sense that every rank writes; no
communication is involved.
"""
function save_checkpoint(solver::Solver, Q, prefix::AbstractString)
    decomp = solver.decomp
    rank = MPI.Comm_rank(decomp.comm)
    o1, o2, o3 = decomp.n_halo_d
    nx, ny, nz = decomp.n_local
    open(_ckpt_name(prefix, rank), "w") do io
        write(io, UInt64(CKPT_MAGIC))
        write(io, Int64(solver.equations.n_cons))
        for d in 1:3
            write(io, Int64(decomp.n_global[d]))
        end
        for d in 1:3
            write(io, Int64(decomp.n_local[d]))
        end
        for d in 1:3
            write(io, Int64(decomp.coords[d]))
        end
        write(io, Float64(solver.t))
        write(io, Int64(solver.step))
        write(io, Q[o1+1:o1+nx, o2+1:o2+ny, o3+1:o3+nz, :])
    end
    return prefix
end

"""
    load_checkpoint!(solver, Q, prefix)

Restore the interior of `Q` and (t, step) from `prefix.rNNNN.ckpt`, verifying
grid, decomposition, and conserved-layout compatibility.
"""
function load_checkpoint!(solver::Solver, Q, prefix::AbstractString)
    decomp = solver.decomp
    rank = MPI.Comm_rank(decomp.comm)
    o1, o2, o3 = decomp.n_halo_d
    nx, ny, nz = decomp.n_local
    open(_ckpt_name(prefix, rank), "r") do io
        read(io, UInt64) == CKPT_MAGIC || error("not a CompactLES checkpoint")
        Int(read(io, Int64)) == solver.equations.n_cons || error("conserved layout mismatch")
        for d in 1:3
            Int(read(io, Int64)) == decomp.n_global[d] || error("global grid mismatch")
        end
        for d in 1:3
            Int(read(io, Int64)) == decomp.n_local[d] || error("decomposition mismatch")
        end
        for d in 1:3
            Int(read(io, Int64)) == decomp.coords[d] || error("rank layout mismatch")
        end
        solver.t = read(io, Float64)
        solver.step = Int(read(io, Int64))
        buf = Array{Float64}(undef, nx, ny, nz, solver.equations.n_cons)
        read!(io, buf)
        Q[o1+1:o1+nx, o2+1:o2+ny, o3+1:o3+nz, :] .= buf
    end
    return Q
end

# ---------------------------------------------------------------------------
# VTK output: dependency-free parallel structured output with appended raw
# binary and Float32 payloads. The metric selects one of two grid types.
#
#   RectilinearGrid (.vtr / .pvtr) is used when the physical grid is a tensor
#     product of three coordinate vectors: Cartesian and stretched grids, and
#     polar grids whose angular dimensions are collapsed. An axisymmetric (r, z)
#     run is written this way because the meridional half-plane is the computed
#     domain and a rectangle represents it correctly.
#
#   StructuredGrid (.vts / .pvts) is used when an angular dimension is resolved.
#     Writing (r, θ, z) as a rectilinear grid would draw the domain unwrapped, as
#     a box in the angle rather than as an annulus, so these grids carry an
#     explicit Cartesian position per point. Vectors are rotated to the same
#     frame: the solver's velocity components are coordinate-aligned
#     (u_r, u_θ, u_z), and a glyph or streamline drawn from those components on a
#     wrapped grid would point in the wrong direction.
#
# Adjacent pieces abut without ghost overlap, which is adequate for point
# rendering and slicing but may produce seams under cell-based filters. This
# applies to both grid types and is unchanged from the original writer.

_extent_str(lo, hi) = join(("$(lo[d]) $(hi[d])" for d in 1:3), " ")

# A resolved angular dimension makes the grid curvilinear; a collapsed angular
# dimension leaves a tensor-product grid.
_curvilinear(solver::Solver) = _curvilinear(solver, solver.metric)
_curvilinear(solver::Solver, ::CartesianMetric) = false
_curvilinear(solver::Solver, ::CylindricalMetric) = solver.decomp.active[2]
_curvilinear(solver::Solver, ::SphericalMetric) =
    solver.decomp.active[2] || solver.decomp.active[3]

"Extension of the parallel container `save_vtk` will write for this solver."
container_extension(solver::Solver) = _curvilinear(solver) ? ".pvts" : ".pvtr"

# Coordinate surface → Cartesian position. Conventions are metric.jl's:
# cylindrical (r, θ, z), spherical (r, θ_polar, φ).
_cartesian_position(::CartesianMetric, x1, x2, x3) = (x1, x2, x3)
_cartesian_position(::CylindricalMetric, r, θ, z) = (r * cos(θ), r * sin(θ), z)
_cartesian_position(::SphericalMetric, r, θ, φ) =
    (r * sin(θ) * cos(φ), r * sin(θ) * sin(φ), r * cos(θ))

# ...and the matching rotation of a coordinate-aligned vector into that frame.
_cartesian_vector(::CartesianMetric, v, x1, x2, x3) = v
function _cartesian_vector(::CylindricalMetric, v, r, θ, z)
    vr, vθ, vz = v
    (vr * cos(θ) - vθ * sin(θ), vr * sin(θ) + vθ * cos(θ), vz)
end
function _cartesian_vector(::SphericalMetric, v, r, θ, φ)
    vr, vθ, vφ = v
    st, ct, sp, cp = sin(θ), cos(θ), sin(φ), cos(φ)
    (vr * st * cp + vθ * ct * cp - vφ * sp,
     vr * st * sp + vθ * ct * sp + vφ * cp,
     vr * ct - vθ * st)
end

"""
Field names [`save_vtk`](@ref) writes when none are requested. These are the
fields it wrote before field selection was added, so existing calls are
unaffected.
"""
const DEFAULT_VTK_FIELDS = (:rho, :velocity, :p, :T_ion, :Y)

# Preparation cost per name. A field derived from a velocity gradient requires a
# full gradient pass; an artificial coefficient requires that pass and
# `compute_artificial!`. Requesting neither retains the original cost of a single
# `primitives!` call.
_wants_gradients(name::Symbol) =
    name in (:vorticity, :vorticity_magnitude, :qcriterion, :divergence,
             :schlieren, :strain_mag)
_wants_artificial(name::Symbol) =
    name in (:mu_art, :beta_art, :kappa_art, :D_art, :sensor, :strain_mag)

function _prepare_fields!(solver::Solver, Q, fields)
    if any(_wants_gradients, fields) || any(_wants_artificial, fields)
        # This exchanges halos, so ρ and the velocities are valid into the halo
        # and the derivative passes remain accurate at a rank boundary. It also
        # overwrites `grad_u` and the artificial coefficients. That is harmless
        # between steps, since `compute_rhs!` rebuilds both at every stage.
        compute_primitives_and_gradients!(solver, Q)
        any(_wants_artificial, fields) && compute_artificial!(solver, Q)
    else
        primitives!(solver, Q)   # halos may be stale but the interior is what we write
    end
    return solver
end

# --- Subsampling -------------------------------------------------------------
#
# A stride writes every s-th point. Every rank must select the same s-th point,
# so the retained set is defined on the GLOBAL index as {1, 1+s, 1+2s, ...} and
# each rank writes its own intersection with that set. Striding a rank's block
# from its own first local point would give the pieces offset lattices that
# neither tile nor align. The result is a doubled or missing plane at a rank
# boundary rather than an error.
#
# Extents are then expressed in the COARSE index space, where global index g
# sits at (g − 1) ÷ s. The resulting per-rank ranges are contiguous, disjoint,
# and cover the coarse grid, as the parallel container requires.
#
# Subsampling reduces file size and write time by s³. It does not reduce the
# cost of producing the fields: a gradient or schlieren pass still runs over the
# whole local block before any point is sampled.

# Positivity is checked here rather than with the emptiness test below because
# `_stride_ranges` divides by the stride, and a zero would fault before any
# check could run. Every rank receives the same stride, so a local throw is
# already consistent across the communicator.
function _normalize_stride(stride)::NTuple{3,Int}
    st = stride isa Integer ? ntuple(_ -> Int(stride), 3) :
                              ntuple(d -> Int(stride[d]), 3)
    all(>=(1), st) ||
        throw(ArgumentError("save_vtk: stride must be >= 1 in every dimension, " *
                            "got $st"))
    return st
end

"Local interior indices this rank keeps along each dimension under `stride`."
function _stride_ranges(solver::Solver, stride::NTuple{3,Int})
    decomp = solver.decomp
    return ntuple(3) do d
        # Smallest local i ≥ 1 whose global index offset+i satisfies
        # (offset + i − 1) % s == 0.
        i0 = mod(-decomp.offset[d], stride[d]) + 1
        i0:stride[d]:decomp.n_local[d]
    end
end

# Coarse-index extent of this rank's piece, and of the whole grid.
_strided_piece_extent(solver::Solver, stride, ranges) = begin
    off = solver.decomp.offset
    lo = ntuple(d -> (off[d] + first(ranges[d]) - 1) ÷ stride[d], 3)
    hi = ntuple(d -> lo[d] + length(ranges[d]) - 1, 3)
    lo, hi
end

_strided_whole_extent(solver::Solver, stride) = begin
    n = solver.decomp.n_global
    ntuple(d -> 0, 3), ntuple(d -> length(1:stride[d]:n[d]) - 1, 3)
end

# A stride large enough to leave a rank with no points along a dimension gives
# that rank no valid piece to write. The test is collective so that every rank
# throws; a single rank throwing would leave the others blocked in the Allgather
# below.
function _check_stride(solver::Solver, stride::NTuple{3,Int}, ranges)
    bad = MPI.Allreduce(Int[isempty(ranges[d]) for d in 1:3], max,
                        solver.decomp.comm)
    any(>(0), bad) || return nothing
    dims = join(("xyz"[d] for d in 1:3 if bad[d] > 0), ", ")
    throw(ArgumentError("save_vtk: stride $stride leaves a rank with no points " *
                        "along $dims; reduce the stride or use fewer ranks in " *
                        "that dimension"))
end

_interior(solver::Solver, a, ranges) = begin
    o1, o2, o3 = solver.decomp.n_halo_d
    Float32[a[i+o1, j+o2, k+o3]
            for i in ranges[1], j in ranges[2], k in ranges[3]][:]
end

# Interleaved 3-component payload, rotated into the Cartesian frame when the
# grid being written is curvilinear.
function _interior_vector(solver::Solver, a1, a2, a3, rotate::Bool, ranges)
    npts = prod(length.(ranges))
    out = Vector{Float32}(undef, 3 * npts)
    o1, o2, o3 = solver.decomp.n_halo_d
    idx = 1
    for k in ranges[3], j in ranges[2], i in ranges[1]
        I = CartesianIndex(i + o1, j + o2, k + o3)
        v = (a1[I], a2[I], a3[I])
        if rotate
            v = _cartesian_vector(solver.metric, v, xcoord(solver, 1, i),
                                  xcoord(solver, 2, j), xcoord(solver, 3, k))
        end
        out[idx] = v[1]; out[idx+1] = v[2]; out[idx+2] = v[3]
        idx += 3
    end
    return out
end

# |∇ρ|, the numerical schlieren used to visualize shock structure. It requires
# its own derivative pass because `grad_u` carries velocity gradients only.
# Collective: `deriv_along!` is a distributed solve, and is safe here only
# because every rank traverses the same `fields` tuple in the same order.
function _schlieren(solver::Solver)
    g = similar(solver.rho)
    acc = zeros(eltype(solver.rho), size(solver.rho))
    for d in 1:3
        solver.decomp.active[d] || continue
        deriv_along!(g, solver.rho, solver, d, 1)   # a scalar is even across a fold
        _scale_grad!(g, solver, d)
        @inbounds for idx in eachindex(acc)
            acc[idx] += g[idx] * g[idx]
        end
    end
    @inbounds for idx in eachindex(acc)
        acc[idx] = sqrt(acc[idx])
    end
    return acc
end

# grad_u[d, j] is ∂u_j/∂x_d, so the curl reads the off-diagonal pairs.
_vorticity_arrays(solver::Solver) = begin
    g = solver.grad_u
    w1 = similar(solver.rho); w2 = similar(solver.rho); w3 = similar(solver.rho)
    @inbounds for idx in eachindex(w1)
        w1[idx] = g[2, 3][idx] - g[3, 2][idx]
        w2[idx] = g[3, 1][idx] - g[1, 3][idx]
        w3[idx] = g[1, 2][idx] - g[2, 1][idx]
    end
    (w1, w2, w3)
end

function _scalar_from(solver::Solver, f)
    out = similar(solver.rho)
    @inbounds for idx in eachindex(out)
        out[idx] = f(idx)
    end
    return out
end

"""
    vtk_field_entries(solver, Q, name, rotate) -> Vector{(label, ncomponents, data)}

Expand one requested field name into the Float32 payloads written by `save_vtk`.
Most names produce a single entry. `:Y` and `:D_art` produce one entry per
species, and the vector fields produce one three-component entry.
"""
function vtk_field_entries(solver::Solver, Q, name::Symbol, rotate::Bool, ranges)
    E = Tuple{String,Int,Vector{Float32}}
    one_scalar(label, a) = E[(label, 1, _interior(solver, a, ranges))]
    g = solver.grad_u
    if name === :rho
        return one_scalar("rho", solver.rho)
    elseif name === :p
        return one_scalar("p", solver.p)
    elseif name === :T_ion
        return one_scalar("T_ion", solver.T_ion)
    elseif name === :c
        return one_scalar("c", solver.c)
    elseif name === :velocity
        return E[("velocity", 3, _interior_vector(solver, solver.u, solver.v,
                                                  solver.w, rotate, ranges))]
    elseif name === :Y
        return E[("Y$(sp)", 1, _interior(solver, solver.Y[sp], ranges))
                 for sp in 1:solver.equations.n_species]
    elseif name === :mach
        return one_scalar("mach", _scalar_from(solver, idx ->
            sqrt(solver.u[idx]^2 + solver.v[idx]^2 + solver.w[idx]^2) / solver.c[idx]))
    elseif name === :divergence
        return one_scalar("divergence",
            _scalar_from(solver, idx -> g[1, 1][idx] + g[2, 2][idx] + g[3, 3][idx]))
    elseif name === :vorticity
        w1, w2, w3 = _vorticity_arrays(solver)
        return E[("vorticity", 3, _interior_vector(solver, w1, w2, w3, rotate, ranges))]
    elseif name === :vorticity_magnitude
        w1, w2, w3 = _vorticity_arrays(solver)
        return one_scalar("vorticity_magnitude",
            _scalar_from(solver, idx -> sqrt(w1[idx]^2 + w2[idx]^2 + w3[idx]^2)))
    elseif name === :qcriterion
        # Q = ½(Ω_ij Ω_ij − S_ij S_ij): positive where rotation beats strain,
        # which is the usual vortex-core criterion.
        return one_scalar("qcriterion", _scalar_from(solver, idx -> begin
            ss = 0.0; oo = 0.0
            for b in 1:3, a in 1:3
                s = 0.5 * (g[a, b][idx] + g[b, a][idx])
                o = 0.5 * (g[a, b][idx] - g[b, a][idx])
                ss += s * s; oo += o * o
            end
            0.5 * (oo - ss)
        end))
    elseif name === :schlieren
        return one_scalar("schlieren", _schlieren(solver))
    elseif name === :strain_mag
        return one_scalar("strain_mag", solver.strain_mag)
    elseif name === :sensor
        return one_scalar("sensor", solver.sensor)
    elseif name === :mu_art
        return one_scalar("mu_art", solver.mu_art)
    elseif name === :beta_art
        return one_scalar("beta_art", solver.beta_art)
    elseif name === :kappa_art
        return one_scalar("kappa_art", solver.kappa_art)
    elseif name === :D_art
        return E[("D_art$(sp)", 1, _interior(solver, solver.D_art[sp], ranges))
                 for sp in 1:solver.equations.n_species]
    end
    throw(ArgumentError("save_vtk: unknown field $name; known names are " *
                        "rho, p, T_ion, c, velocity, Y, mach, divergence, " *
                        "vorticity, vorticity_magnitude, qcriterion, schlieren, " *
                        "strain_mag, sensor, mu_art, beta_art, kappa_art, D_art"))
end

# Appended-data bookkeeping. Every block is an 8-byte UInt64 length followed by
# the payload (header_type="UInt64"), so an offset is the running sum of both.
function _appended_offsets(sizes)
    offs = Int[]
    total = 0
    for nbytes in sizes
        push!(offs, total)
        total += 8 + nbytes
    end
    return offs
end

function _write_appended(io, blocks)
    write(io, "<AppendedData encoding=\"raw\">\n_")
    for data in blocks
        write(io, UInt64(sizeof(data)))
        write(io, data)
    end
    write(io, "\n</AppendedData>\n</VTKFile>\n")
end

_piece_name(prefix, rank, ext) = string(prefix, ".r", lpad(rank, 4, '0'), ext)

"""
    save_vtk(solver, Q, prefix; fields = DEFAULT_VTK_FIELDS, stride = 1)

Write one piece per rank plus a parallel container on rank 0, as Float32 point
data on the physical grid, including stretch mappings. Open the container in
ParaView or VisIt: `prefix.pvtr`, or `prefix.pvts` when an angular dimension is
resolved. This writes a single dump; use [`FieldWriter`](@ref) for a sequence.

`fields` selects what is written, as a tuple of names. The default is
[`DEFAULT_VTK_FIELDS`](@ref). Available:

| name | meaning |
|---|---|
| `:rho`, `:p`, `:T_ion`, `:c` | density, pressure, temperature, sound speed |
| `:velocity` | three-component velocity |
| `:Y` | one scalar per species |
| `:mach` | \\|u\\|/c |
| `:divergence` | ∇·u |
| `:vorticity`, `:vorticity_magnitude` | ∇×u and its magnitude |
| `:qcriterion` | ½(Ω_ij Ω_ij − S_ij S_ij), positive in a vortex core |
| `:schlieren` | \\|∇ρ\\|, used to visualize shock structure |
| `:strain_mag`, `:sensor` | the artificial-property inputs |
| `:mu_art`, `:beta_art`, `:kappa_art`, `:D_art` | the artificial coefficients |

Preparation cost depends on the request. The default set requires one
`primitives!` pass, as before. A field derived from a velocity gradient adds a
full gradient pass, an artificial coefficient additionally requires
`compute_artificial!`, and `:schlieren` adds a further derivative pass. The
artificial fields report the local action of the regularization directly, which
no integrated diagnostic provides.

`stride` writes every `stride`-th point, given either as one `Int` for all three
dimensions or as a 3-tuple. Points are selected on the **global** index, so every
rank samples the same lattice and the pieces still tile; extents are written in
the resulting coarse index space. At 512³ a stride of 4 reduces a 2.7 GB default
dump to 43 MB.

Subsampling reduces file size and write time only. The fields are still produced
over the whole local block before any point is sampled, so a strided dump costs
the same to compute as a full one. A stride large enough to leave a rank with no
points throws, and does so collectively, so no rank is left blocked while
another raises.

Collective: every rank must call it with the same `fields` and `stride`, since
the derived fields run distributed solves in tuple order.
"""
function save_vtk(solver::Solver, Q, prefix::AbstractString;
                  fields=DEFAULT_VTK_FIELDS, stride=1)
    st = _normalize_stride(stride)
    ranges = _stride_ranges(solver, st)
    _check_stride(solver, st, ranges)
    _prepare_fields!(solver, Q, fields)
    curvilinear = _curvilinear(solver)
    entries = Tuple{String,Int,Vector{Float32}}[]
    for name in fields
        append!(entries, vtk_field_entries(solver, Q, name, curvilinear, ranges))
    end
    return curvilinear ? _save_vts(solver, entries, prefix, st, ranges) :
                         _save_vtr(solver, entries, prefix, st, ranges)
end

# --- Rectilinear: three coordinate vectors, no per-point geometry.

function _save_vtr(solver::Solver, entries, prefix::AbstractString, stride, ranges)
    decomp = solver.decomp
    rank = MPI.Comm_rank(decomp.comm)
    lo, hi = _strided_piece_extent(solver, stride, ranges)
    coords = ntuple(d -> Float64[xcoord(solver, d, i) for i in ranges[d]], 3)

    blocks = Any[coords[1], coords[2], coords[3]]
    append!(blocks, (data for (_, _, data) in entries))
    offs = _appended_offsets(sizeof.(blocks))

    open(_piece_name(prefix, rank, ".vtr"), "w") do io
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
        for (fi, (name, nc, _)) in enumerate(entries)
            write(io, "<DataArray type=\"Float32\" Name=\"", name,
                  "\" NumberOfComponents=\"", string(nc),
                  "\" format=\"appended\" offset=\"", string(offs[3+fi]), "\"/>\n")
        end
        write(io, "</PointData>\n</Piece>\n</RectilinearGrid>\n")
        _write_appended(io, blocks)
    end
    _write_parallel_container(solver, entries, prefix, lo, hi, false, stride)
    return prefix
end

# --- Structured: an explicit Cartesian position per point.

function _save_vts(solver::Solver, entries, prefix::AbstractString, stride, ranges)
    decomp = solver.decomp
    rank = MPI.Comm_rank(decomp.comm)
    lo, hi = _strided_piece_extent(solver, stride, ranges)

    points = Vector{Float64}(undef, 3 * prod(length.(ranges)))
    idx = 1
    for k in ranges[3], j in ranges[2], i in ranges[1]
        x, y, z = _cartesian_position(solver.metric, xcoord(solver, 1, i),
                                      xcoord(solver, 2, j), xcoord(solver, 3, k))
        points[idx] = x; points[idx+1] = y; points[idx+2] = z
        idx += 3
    end

    blocks = Any[points]
    append!(blocks, (data for (_, _, data) in entries))
    offs = _appended_offsets(sizeof.(blocks))

    open(_piece_name(prefix, rank, ".vts"), "w") do io
        write(io, "<?xml version=\"1.0\"?>\n")
        write(io, "<VTKFile type=\"StructuredGrid\" version=\"1.0\" ",
                  "byte_order=\"LittleEndian\" header_type=\"UInt64\">\n")
        write(io, "<StructuredGrid WholeExtent=\"", _extent_str(lo, hi), "\">\n")
        write(io, "<Piece Extent=\"", _extent_str(lo, hi), "\">\n<Points>\n")
        write(io, "<DataArray type=\"Float64\" NumberOfComponents=\"3\" ",
                  "format=\"appended\" offset=\"", string(offs[1]), "\"/>\n")
        write(io, "</Points>\n<PointData>\n")
        for (fi, (name, nc, _)) in enumerate(entries)
            write(io, "<DataArray type=\"Float32\" Name=\"", name,
                  "\" NumberOfComponents=\"", string(nc),
                  "\" format=\"appended\" offset=\"", string(offs[1+fi]), "\"/>\n")
        end
        write(io, "</PointData>\n</Piece>\n</StructuredGrid>\n")
        _write_appended(io, blocks)
    end
    _write_parallel_container(solver, entries, prefix, lo, hi, true, stride)
    return prefix
end

# --- The container, identical in shape for both grid types.

function _write_parallel_container(solver::Solver, entries, prefix, lo, hi,
                                   curvilinear::Bool, stride)
    decomp = solver.decomp
    rank = MPI.Comm_rank(decomp.comm)
    P = MPI.Comm_size(decomp.comm)
    # Gather every rank's piece extent so rank 0 can write the container. This
    # is the only communication in the writer, and the reason it is collective.
    allext = MPI.Allgather(Int64[lo..., hi...], decomp.comm)
    rank == 0 || return prefix

    kind = curvilinear ? "StructuredGrid" : "RectilinearGrid"
    piece_ext = curvilinear ? ".vts" : ".vtr"
    open(string(prefix, curvilinear ? ".pvts" : ".pvtr"), "w") do io
        glo, ghi = _strided_whole_extent(solver, stride)
        write(io, "<?xml version=\"1.0\"?>\n")
        write(io, "<VTKFile type=\"P", kind, "\" version=\"1.0\" ",
                  "byte_order=\"LittleEndian\" header_type=\"UInt64\">\n")
        write(io, "<P", kind, " WholeExtent=\"", _extent_str(glo, ghi),
                  "\" GhostLevel=\"0\">\n")
        if curvilinear
            write(io, "<PPoints>\n<PDataArray type=\"Float64\" ",
                      "NumberOfComponents=\"3\"/>\n</PPoints>\n")
        else
            write(io, "<PCoordinates>\n")
            for d in 1:3
                write(io, "<PDataArray type=\"Float64\" Name=\"", "xyz"[d], "\"/>\n")
            end
            write(io, "</PCoordinates>\n")
        end
        write(io, "<PPointData>\n")
        for (name, nc, _) in entries
            write(io, "<PDataArray type=\"Float32\" Name=\"", name,
                  "\" NumberOfComponents=\"", string(nc), "\"/>\n")
        end
        write(io, "</PPointData>\n")
        for r in 0:(P-1)
            e = allext[6r+1:6r+6]
            plo = (e[1], e[2], e[3]); phi = (e[4], e[5], e[6])
            write(io, "<Piece Extent=\"", _extent_str(plo, phi),
                  "\" Source=\"", basename(_piece_name(prefix, r, piece_ext)),
                  "\"/>\n")
        end
        write(io, "</P", kind, ">\n</VTKFile>\n")
    end
    return prefix
end
# ---------------------------------------------------------------------------
# Output as a callback effect.
#
# A dump sequence requires two things that callers previously supplied by hand:
# uniquely named files, and a container recording the physical time of each one.
# `save_vtk` given a bare prefix provides neither, so every example built its own
# `lpad` and then animated against the frame index because no file recorded the
# time. This supplies both, and takes its schedule from a trigger rather than
# implementing one.
#
# `_write_dump!` is the extension point. Naming, sequencing, and the collection
# file above it are independent of the output format. A format writing one
# shared file rather than one file per rank replaces that single call, as does
# one traversing a list of patches rather than the single block held by a rank.

"""
    FieldWriter(prefix; fields = DEFAULT_VTK_FIELDS, stride = 1, pad = 4,
                collection = true)

Write a numbered field dump each time its trigger fires, together with a `.pvd`
collection recording the physical time of every frame. Use it as the effect of a
callback:

```julia
run!(solver, Q; tfinal = 2.5e-3,
     callback = Callback(EveryTime(5e-5),
                         FieldWriter("out/field", fields = (:rho, :velocity,
                                                            :schlieren, :sensor))))
```

The first frame is written as `out/field_0000.pvtr` with its per-rank pieces, and
each frame is recorded in `out/field.pvd`. Opening the `.pvd` in ParaView or
VisIt animates against physical time; opening individual pieces animates against
the frame index. [`EveryTime`](@ref) supplies an evenly spaced time schedule.

`fields` and `stride` are passed through to [`save_vtk`](@ref), which documents
the available names, the cost of each, and how points are selected for
subsampling. On a grid with a resolved angular dimension the container is
`.pvts` rather than `.pvtr`, and the collection references it accordingly.

Like the wrapped `save_vtk` call, this operation is collective and must run on
every rank. A [`Callback`](@ref) provides that guarantee. The first dump creates
`dirname(prefix)`. The `wall_io` field records cumulative output time; callback
execution lies outside the interval recorded by `solver.wall_step`.
"""
mutable struct FieldWriter{F}
    prefix::String
    fields::F
    stride::NTuple{3,Int}
    pad::Int
    collection::Bool
    index::Int
    times::Vector{Float64}
    wall_io::Float64
end

FieldWriter(prefix::AbstractString; fields=DEFAULT_VTK_FIELDS, stride=1,
            pad::Int=4, collection::Bool=true) =
    FieldWriter(String(prefix), fields, _normalize_stride(stride), pad,
                collection, 0, Float64[], 0.0)

"Path stem of frame `m` (0-based) of a [`FieldWriter`](@ref), without extension."
frame_prefix(writer::FieldWriter, m::Integer) =
    string(writer.prefix, "_", lpad(m, writer.pad, '0'))

_write_dump!(writer::FieldWriter, solver, Q, stem) =
    save_vtk(solver, Q, stem; fields=writer.fields, stride=writer.stride)

function (writer::FieldWriter)(solver, Q)
    wall_0 = time_ns()
    comm = solver.decomp.comm
    if writer.index == 0
        # One rank creates the directory and the others wait for it. Concurrent
        # `mkpath` of the same path is not reliably idempotent across
        # filesystems, and a rank writing into a directory that does not yet
        # exist fails at `open` without indicating the cause.
        MPI.Comm_rank(comm) == 0 && !isempty(dirname(writer.prefix)) &&
            mkpath(dirname(writer.prefix))
        MPI.Barrier(comm)
    end
    _write_dump!(writer, solver, Q, frame_prefix(writer, writer.index))
    push!(writer.times, Float64(solver.t))
    writer.index += 1
    writer.collection && MPI.Comm_rank(comm) == 0 &&
        _write_pvd(writer, container_extension(solver))
    writer.wall_io += (time_ns() - wall_0) / 1e9
    return false
end

# Rewritten in full on every dump rather than appended to. The cost is O(frames)
# per frame, negligible at the frame counts actually written, and in exchange a
# run terminated by the scheduler still leaves a valid collection naming every
# completed frame. An appended file would lack its closing tags and would not
# open at all.
function _write_pvd(writer::FieldWriter, ext::AbstractString)
    open(string(writer.prefix, ".pvd"), "w") do io
        write(io, "<?xml version=\"1.0\"?>\n")
        write(io, "<VTKFile type=\"Collection\" version=\"1.0\" ",
                  "byte_order=\"LittleEndian\">\n<Collection>\n")
        for (m, t) in enumerate(writer.times)
            # Relative to the .pvd's own directory, which is where the pieces are.
            src = string(basename(frame_prefix(writer, m - 1)), ext)
            write(io, "<DataSet timestep=\"", string(t),
                  "\" group=\"\" part=\"0\" file=\"", src, "\"/>\n")
        end
        write(io, "</Collection>\n</VTKFile>\n")
    end
    return writer
end
