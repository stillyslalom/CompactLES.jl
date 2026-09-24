# Geometry- and variable-aware extraction of report fields for visualization,
# and the Makie plotting interface layered on top of it.
#
# The functions here are a collective, geometry-aware API that resolves any
# named scalar through the same catalog `save_vtk` uses (`scalar_field` in
# io.jl), so a profile or a slice is one call with the same meaning under MPI
# decomposition as in serial. A rank-local sampler wired to one variable and
# one geometry is correct only where the sampled line lives on one rank.
#
# Nothing here depends on a plotting package. `profileplot`, `fieldheatmap`, and
# their mutating forms are declared as stubs that error until a Makie backend is
# loaded; `ext/CompactLESMakieExt.jl` supplies the methods. This mirrors the
# HDF5 split in src/hdf5.jl exactly.

# --- Named scalar → padded array -------------------------------------------

"""
    field_array(solver, Q, name::Symbol; species = 1) -> Array{T,3}

The full padded array of the named scalar report variable, refreshed from `Q`,
in the solver's own element type.
This is the extraction primitive the profile and slice helpers build on, and it
composes with the reductions in diagnostics.jl (`plane_profile`,
`volume_integral`) and with `boundary_plane`.

`name` is resolved through [`scalar_field`](@ref), the catalog
[`save_vtk`](@ref) also uses, so the available names are identical: stored
primitives (`:rho`, `:p`, `:T_ion`, `:c`, `:u`, `:v`, `:w`), the per-species
`:Y` and `:D_art` (selected with `species`), and derived scalars (`:mach`,
`:divergence`, `:vorticity_magnitude`, `:qcriterion`, `:schlieren`,
`:strain_mag`, `:sensor`, `:mu_art`, `:beta_art`, `:kappa_art`).

`species` selects the array for `:Y` and `:D_art` and is ignored by every other
name.

Every rank must call this function because it calls
[`refresh_primitives!`](@ref), and a derived name runs the gradient pass and,
for an artificial coefficient, `compute_artificial!`. The gradient pass
overwrites `solver.grad_u` and the sensor and `tmp_a`/`tmp_b` scratch, which
`compute_rhs!` rebuilds at every stage. The artificial coefficient arrays are
restored to the values the integrator left, so calling this between steps does
not change the next timestep or a regrid decision. The returned array is a
fresh copy that remains valid past the next step.
"""
function field_array(solver::Solver, Q, name::Symbol; species::Int=1)
    return preserving_artificial(solver, _wants_artificial(name)) do
        if _wants_gradients(name) || _wants_artificial(name)
            # This refreshes primitives into the halos as part of the gradient
            # pass, so it subsumes `refresh_primitives!`.
            compute_primitives_and_gradients!(solver, Q)
            _wants_artificial(name) && compute_artificial!(solver, Q)
        else
            refresh_primitives!(solver, Q)
        end
        # An eltype-preserving copy: extraction must not hardcode Float64, since
        # the storage type is a backend decision.
        return Array(scalar_field(solver, name; species=species))
    end
end

# --- Line profiles ----------------------------------------------------------

"""
    line_profile(solver, Q, name; dim = 1, species = 1) -> (coord, value)

The transverse-plane average of the named scalar as a function of position
along `dim`, returned as the pair of vectors `(coord, value)`, both of length
`n_global[dim]` and identical on every rank. `value[i]` is the area-weighted
mean of the field over the plane through station `i`, so
`line_profile(solver, Q, :rho)` is the mean density profile along the first
axis: the radial or axial mean of a multidimensional field. It is not the
field on one grid line; [`line_sample`](@ref) is.

The coordinate comes from [`profile_coordinate`](@ref) and the value from
[`plane_profile`](@ref). When both transverse dimensions are collapsed, as in
the common tutorial case of `n_global = (N, 1, 1)`, the average is over a single
point and the profile coincides with `line_sample` through `(i, 1, 1)`. For a
line integral in place of an average, weight by [`profile_spacing`](@ref).

Every rank must call this function; see [`field_array`](@ref).
"""
function line_profile(solver::Solver, Q, name::Symbol; dim::Int=1, species::Int=1)
    1 <= dim <= 3 || throw(ArgumentError("line_profile: dim must be 1, 2, or 3"))
    f = field_array(solver, Q, name; species=species)
    coord = profile_coordinate(solver, dim)
    value = plane_profile(solver, f, dim)
    return coord, value
end

"""
    line_sample(solver, Q, name; dim = 1, index = (1, 1), at = nothing,
                species = 1) -> (coord, value)

The named scalar on one grid line in direction `dim`: the nodes whose global
index along `dim` runs over `1:n_global[dim]` while the two transverse global
indices are fixed. Returned as the pair of vectors `(coord, value)`, both of
length `n_global[dim]` and identical on every rank. This is a point sample,
not a reduction; [`line_profile`](@ref) is the transverse-plane average, and
the two agree only when both transverse dimensions are collapsed.

`index` fixes the transverse global indices, given for the two dimensions
other than `dim` in increasing order: `(j, k)` for `dim = 1`, `(i, k)` for
`dim = 2`, and `(i, j)` for `dim = 3`. An index outside `1:n_global` in its
dimension throws `ArgumentError`. `at` gives the transverse position in the
metric's own coordinates instead, in the same order, and each entry is
snapped to the nearest node of that dimension; passing both throws.

The value is the field at the node, in the solver's element type, and the
coordinate comes from [`global_xcoord`](@ref). Every rank must call this
function, including one whose block holds no part of the line; see
[`field_array`](@ref).
"""
function line_sample(solver::Solver, Q, name::Symbol; dim::Int=1,
                     index::Union{Nothing,NTuple{2,Int}}=nothing,
                     at::Union{Nothing,NTuple{2,Real}}=nothing, species::Int=1)
    1 <= dim <= 3 || throw(ArgumentError("line_sample: dim must be 1, 2, or 3"))
    index === nothing || at === nothing ||
        throw(ArgumentError("line_sample: give index or at, not both"))
    decomp = solver.decomp
    a, b = _plane_dims(dim)
    if at !== nothing
        ga, gb = _nearest_global_index(solver, a, at[1]),
                 _nearest_global_index(solver, b, at[2])
    else
        ga, gb = index === nothing ? (1, 1) : index
    end
    for (d, g) in ((a, ga), (b, gb))
        1 <= g <= decomp.n_global[d] ||
            throw(ArgumentError("line_sample: index $g out of range " *
                                "1:$(decomp.n_global[d]) along dimension $d"))
    end
    f = field_array(solver, Q, name; species=species)

    # Each node of the line has exactly one owner, so the sum of the zeroed
    # global vectors reproduces every owner's value bitwise, and Allreduce
    # replicates the line without per-dimension gather logic.
    n = decomp.n_global[dim]
    line = zeros(eltype(f), n)
    la, lb = ga - decomp.offset[a], gb - decomp.offset[b]
    if 1 <= la <= decomp.n_local[a] && 1 <= lb <= decomp.n_local[b]
        oa, ob, od = decomp.n_halo_d[a], decomp.n_halo_d[b], decomp.n_halo_d[dim]
        for li in 1:decomp.n_local[dim]
            I = _plane_index(dim, la + oa, lb + ob, li + od)
            line[decomp.offset[dim] + li] = f[I]
        end
    end
    line = MPI.Allreduce(line, +, decomp.comm)
    coord = [global_xcoord(solver, dim, g) for g in 1:n]
    return coord, line
end

# The global index along `d` whose coordinate is nearest `x`; the first of a
# tie. Coordinates are monotone in the index, stretched or not, so a linear
# scan of n_global[d] values is exact and cheap.
function _nearest_global_index(solver::Solver, d::Int, x::Real)
    best, dist = 1, Inf
    for g in 1:solver.decomp.n_global[d]
        e = abs(global_xcoord(solver, d, g) - x)
        e < dist && ((best, dist) = (g, e))
    end
    return best
end

# --- Two-dimensional slices -------------------------------------------------

"""
    field_slice(solver, Q, name; normal = 3, index = 1, species = 1)
        -> (x1, x2, values) or nothing

The plane of the named scalar transverse to dimension `normal`, taken at the
global index `index` along `normal`. Returns, on rank 0, the two in-plane
coordinate vectors and the `length(x1) × length(x2)` matrix of values; returns
`nothing` on every other rank, since a slice is a rank-0 gather, not a
replicated reduction. The in-plane axes are the two dimensions other than
`normal`, in increasing order (for `normal = 3` they are dimensions 1 and 2).

The coordinates are the metric's own, `(r, θ)`, `(r, φ)`, or a Cartesian pair,
so a curvilinear slice is on a coordinate surface. Pass the result through
[`cartesian_slice`](@ref), along with the in-plane dimension pair, to resample it
onto a Cartesian grid for a heatmap.

`index` is a global index and must lie in `1:n_global[normal]`; an index outside
that range throws `ArgumentError`, as does a `normal` outside `1:3`.

Every rank must call this function, including a rank whose block holds no part
of the requested plane; see [`field_array`](@ref).
"""
function field_slice(solver::Solver, Q, name::Symbol; normal::Int=3, index::Int=1,
                     species::Int=1)
    1 <= normal <= 3 || throw(ArgumentError("field_slice: normal must be 1, 2, or 3"))
    decomp = solver.decomp
    ng = decomp.n_global[normal]
    1 <= index <= ng ||
        throw(ArgumentError("field_slice: index $index out of range 1:$ng along " *
                            "dimension $normal"))
    f = field_array(solver, Q, name; species=species)

    a, b = _plane_dims(normal)
    na, nb = decomp.n_global[a], decomp.n_global[b]
    # Assemble the global plane by having each rank that owns the slice scatter
    # its local block into a zeroed global array, then sum onto rank 0. A slice
    # is two-dimensional and cheap, so the O(na·nb) reduction is not a concern,
    # and it needs no per-dimension gather logic.
    plane = zeros(Float64, na, nb)
    local_i = index - decomp.offset[normal]
    if 1 <= local_i <= decomp.n_local[normal]
        oa, ob = decomp.n_halo_d[a], decomp.n_halo_d[b]
        on = decomp.n_halo_d[normal]
        kn = local_i + on
        for lb in 1:decomp.n_local[b], la in 1:decomp.n_local[a]
            I = _plane_index(normal, la + oa, lb + ob, kn)
            plane[decomp.offset[a] + la, decomp.offset[b] + lb] = f[I]
        end
    end
    plane = MPI.Reduce(plane, +, decomp.comm; root=0)

    x1 = [global_xcoord(solver, a, g) for g in 1:na]
    x2 = [global_xcoord(solver, b, g) for g in 1:nb]
    MPI.Comm_rank(decomp.comm) == 0 || return nothing
    return x1, x2, plane
end

# The two in-plane dimensions transverse to `normal`, in increasing order.
_plane_dims(normal::Int) = normal == 1 ? (2, 3) : normal == 2 ? (1, 3) : (1, 2)

# CartesianIndex for in-plane locals (a, b) and normal local kn, given `normal`.
function _plane_index(normal::Int, ia::Int, ib::Int, kn::Int)
    normal == 1 && return CartesianIndex(kn, ia, ib)
    normal == 2 && return CartesianIndex(ia, kn, ib)
    return CartesianIndex(ia, ib, kn)
end

# --- Whole-grid gather -------------------------------------------------------
#
# A snapshot is every node of a block, gathered to rank 0 for in-memory
# postprocessing. Each rank serializes its interior blocks with their global
# placement, and rank 0 writes them into the assembled arrays; no rank but the
# root allocates the whole grid. `MPI.gather` counts bytes in a `Cint`, which
# caps one rank's share at 2 GiB, far beyond the desktop runs this serves;
# larger runs write VTK or HDF5.

"""
    FieldSnapshot

The unpadded fields of one block of nodes and the grid they sit on, as
[`field_snapshot`](@ref) returns them on rank 0. The block is the whole grid
for a single-patch solver and one patch for the patch-layout form.

- `coords`: three coordinate vectors; `coords[d][i]` is the metric coordinate
  of node `i` along dimension `d`, stretch mapping and fold offset included.
  The grid is their tensor product.
- `fields`: a `Dict{Symbol,Array}` keyed by the requested names. A scalar is an
  `n1 × n2 × n3` array. `:Y` and `:D_art` carry a fourth dimension over
  species, and `:velocity` and `:vorticity` a fourth of length 3 over
  components along the metric's own directions (`(u_r, u_θ, u_z)` on a
  cylindrical grid), not rotated into the Cartesian frame.
- `t`, `step`: the solver clock and step count when the snapshot was taken.
- `metric`: the solver's metric, which [`cartesian_coordinates`](@ref) reads.
- `level`, `offset`: the refinement level (0 is the root) and the block's node
  offset in that level's index space; `0` and `(0, 0, 0)` for a single patch.
- `covered`: `true` at a node whose quadrature cell a finer level covers
  entirely, where a composite view shows the finer block instead. All `false`
  on the finest level and for a single-patch solver.

`snap[name]` is `snap.fields[name]`, `keys(snap)` lists the names, and
`size(snap)` gives the node counts.
"""
struct FieldSnapshot{T,M<:Metric}
    coords::NTuple{3,Vector{T}}
    fields::Dict{Symbol,Array{T}}
    t::T
    step::Int
    metric::M
    level::Int
    offset::NTuple{3,Int}
    covered::BitArray{3}
end

Base.getindex(snap::FieldSnapshot, name::Symbol) = snap.fields[name]
Base.keys(snap::FieldSnapshot) = keys(snap.fields)
Base.haskey(snap::FieldSnapshot, name::Symbol) = haskey(snap.fields, name)
Base.size(snap::FieldSnapshot) = map(length, snap.coords)

function Base.show(io::IO, snap::FieldSnapshot{T}) where {T}
    print(io, "FieldSnapshot{")
    show(io, T)
    print(io, "}(")
    _show_dimensions(io, size(snap))
    snap.level == 0 || print(io, ", level ", snap.level)
    print(io, ", t=", snap.t, ", fields: ", join(sort!(collect(keys(snap))), ", "),
          ')')
end

"""
    cartesian_coordinates(snap::FieldSnapshot) -> (X, Y, Z)

The Cartesian position of every node of `snap`, as three arrays of its node
counts. On a Cartesian grid these repeat `snap.coords` over the grid; on a
cylindrical or spherical grid they map `(r, θ, z)` or `(r, θ, φ)` to `(x, y, z)`,
for a scatter or surface plot of a curvilinear block. Vector fields in the
snapshot remain in the metric's own components.
"""
function cartesian_coordinates(snap::FieldSnapshot{T}) where {T}
    n = size(snap)
    X, Y, Z = (Array{T}(undef, n) for _ in 1:3)
    x1, x2, x3 = snap.coords
    for k in 1:n[3], j in 1:n[2], i in 1:n[1]
        X[i, j, k], Y[i, j, k], Z[i, j, k] =
            _cartesian_position(snap.metric, x1[i], x2[j], x3[k])
    end
    return X, Y, Z
end

"""
    field_snapshot(solver, Q; fields = DEFAULT_VTK_FIELDS)
        -> FieldSnapshot or nothing
    field_snapshot(solver, states::Vector; fields = DEFAULT_VTK_FIELDS)
        -> Vector{FieldSnapshot} or nothing

Every interior node of the named fields, without halo padding, gathered with the
grid coordinates into a [`FieldSnapshot`](@ref) on rank 0, in the solver's
element type. Every other rank returns `nothing`. This is the in-memory
counterpart of [`save_vtk`](@ref) for postprocessing and plotting a
desktop-scale run: rank 0 holds the whole grid, so a large run is better
written with `save_vtk` or `save_hdf5`.

`fields` is a tuple or vector of the names `save_vtk` accepts: the scalars of
[`scalar_field`](@ref), the per-species `:Y` and `:D_art`, and the vectors
`:velocity` and `:vorticity`. Preparation costs what it does for `save_vtk`: a
field derived from a velocity gradient adds a gradient pass, an artificial
coefficient adds `compute_artificial!`, and the artificial coefficient arrays
are restored afterwards, so a snapshot between steps does not change the next
timestep.

The first form takes a single-patch solver and its state and returns the whole
grid. The second takes the state vector of a refined or patch-partitioned
solver and returns one snapshot per patch, ordered by level and then by node
offset; each carries its level, its offset in that level's index space, and
the nodes a finer level covers. Abutting root patches share their interface
plane, which appears in both.

Every rank of `solver.comm` must call this function with the same `fields`,
since the derived fields run distributed solves in the order given.
"""
function field_snapshot(solver::Solver, Q; fields=DEFAULT_VTK_FIELDS)
    patches = getfield(solver, :patches)
    length(patches) == 1 && nlevels(solver) == 1 &&
        only(patches).region.extent == solver.n_global ||
        throw(ArgumentError("field_snapshot: this solver holds several patches; " *
                            "pass the state vector allocate_state returns"))
    snaps = _snapshot(solver, [Q], fields)
    return snaps === nothing ? nothing : only(snaps)
end

function field_snapshot(solver::Solver, states::Vector{<:ConservedState};
                        fields=DEFAULT_VTK_FIELDS)
    return _snapshot(solver, states, fields)
end

const _SNAPSHOT_STACKED = (:velocity, :vorticity, :Y, :D_art)

function _snapshot(solver::Solver, states, fields)
    names = Tuple(fields)
    for name in names
        name in SCALAR_FIELD_NAMES || name in _SNAPSHOT_STACKED ||
            throw(ArgumentError("field_snapshot: unknown field $name; known " *
                                "names are $(join(SCALAR_FIELD_NAMES, ", ")), " *
                                join(_SNAPSHOT_STACKED, ", ")))
    end
    allunique(names) ||
        throw(ArgumentError("field_snapshot: fields $names repeat a name"))
    patches = getfield(solver, :patches)
    blocks = preserving_artificial(solver, any(_wants_artificial, names)) do
        # Patch order is the collective order of the derived fields, as in
        # the patch-layout `save_vtk`.
        map(eachindex(patches)) do li
            ps = PatchSolver(solver, patches[li])
            _prepare_fields!(ps, states[li], names)
            _snapshot_block(ps, names)
        end
    end
    gathered = MPI.gather(blocks, solver.comm; root=0)
    MPI.Comm_rank(solver.comm) == 0 || return nothing
    return _assemble_snapshots(solver, reduce(vcat, gathered), names)
end

# This rank's interior block of one patch, with its placement in the patch.
function _snapshot_block(ps::PatchSolver, names)
    decomp = ps.decomp
    n = decomp.n_local
    interior = ntuple(d -> decomp.n_halo_d[d] .+ (1:n[d]), 3)
    # Device storage is copied to the host whole before the interior is cut.
    grab(a) = (a isa Array ? a : Array(a))[interior...]
    stacked(arrays) = cat(map(grab, arrays)...; dims=4)
    data = map(names) do name
        name === :velocity && return stacked((ps.u, ps.v, ps.w))
        name === :vorticity && return stacked(_vorticity_arrays(ps))
        name === :Y && return stacked(ps.Y)
        name === :D_art && return stacked(ps.D_art)
        return grab(scalar_field(ps, name))
    end
    T = eltype(ps.rho)
    coords = ntuple(d -> T[xcoord(ps, d, i) for i in 1:n[d]], 3)
    region = ps.patch.region
    return (level=ps.patch.level, offset=region.offset, extent=region.extent,
            lo=decomp.offset, coords=coords, data=collect(data),
            covered=BitArray(ps.covered[interior...] .== 0xff))
end

# Rank 0: the blocks of every rank written into one snapshot per patch. A
# patch is named by its level and offset, which no two patches share.
function _assemble_snapshots(solver::Solver, blocks, names)
    T = eltype(first(blocks).coords[1])
    groups = Dict{Tuple{Int,NTuple{3,Int}},Vector{Any}}()
    for b in blocks
        push!(get!(() -> Any[], groups, (b.level, b.offset)), b)
    end
    snaps = FieldSnapshot{T,typeof(solver.metric)}[]
    for key in sort!(collect(keys(groups)))
        parts = groups[key]
        n = first(parts).extent
        coords = ntuple(d -> Vector{T}(undef, n[d]), 3)
        fields = Dict{Symbol,Array{T}}()
        for (m, name) in enumerate(names)
            a = first(parts).data[m]
            fields[name] = Array{T}(undef, n..., size(a)[4:end]...)
        end
        covered = falses(n)
        for b in parts
            span = ntuple(d -> b.lo[d] .+ (1:length(b.coords[d])), 3)
            for d in 1:3
                coords[d][span[d]] = b.coords[d]
            end
            for (m, name) in enumerate(names)
                f = fields[name]
                view(f, span..., ntuple(_ -> Colon(), ndims(f) - 3)...) .= b.data[m]
            end
            covered[span...] = b.covered
        end
        push!(snaps, FieldSnapshot(coords, fields, T(solver.t), Int(solver.step),
                                   solver.metric, key[1], key[2], covered))
    end
    return snaps
end

# --- Curvilinear slice → Cartesian raster -----------------------------------

"""
    cartesian_slice(metric, dims, x1, x2, values; n = 400, fill = NaN,
                    period = nothing) -> (X, Y, grid)
    cartesian_slice(solver, dims, x1, x2, values; period = :auto, kwargs...)

Resample a coordinate-surface slice, the `(x1, x2, values)` triple from
[`field_slice`](@ref), onto a uniform Cartesian raster suitable for a heatmap.
`dims` is the `(a, b)` pair of in-plane dimensions the slice spans: the two
dimensions other than `field_slice`'s `normal`, in increasing order, so
`normal = 3` gives `(1, 2)`. It fixes how `metric` maps a coordinate pair to a
Cartesian position. Returns the two Cartesian axis vectors `X`, `Y` (length `n`)
and the `n × n` grid, with `grid[i, j]` the value at `(X[i], Y[j])` and `fill`
(default `NaN`, which renders transparent) outside the sampled region.

The polar mapping is applied only when `dims == (1, 2)`: the cylindrical
`(r, θ)` plane, whose raster covers the physical disk or annulus, and the
spherical `(r, θ_polar)` meridian, mapped to `X = r sin θ`, `Y = r cos θ` and so
covering a half-disk or half-annulus. Every other pair, including a
`CartesianMetric` plane and a spherical `(r, φ)` one, is treated as
Cartesian and rastered on its own axes. Raster cells are filled by bilinear
interpolation in the coordinate values either way. For a collapsed radial
calculation with no resolved angle to slice, use [`revolve_profile`](@ref)
instead.

The second in-plane coordinate `x2` is treated as periodic when `period` is
given (its full period, e.g. `2π` for an azimuth): a wrapped node is appended so
the raster has no unfilled wedge across the `x2[end] → x2[1]` seam. The
`solver`-taking method fills `period` from the grid when the second in-plane
dimension is active and periodic, and passes `nothing` otherwise.
"""
function cartesian_slice(metric::Metric, dims::Tuple{Int,Int},
                         x1::AbstractVector, x2::AbstractVector,
                         values::AbstractMatrix; n::Int=400, fill=NaN,
                         period::Union{Nothing,Real}=nothing)
    a, b = dims
    # Close a periodic second coordinate by appending the first node shifted by
    # one period, so the seam between the last stored angle and the first is
    # interpolated across.
    if period !== nothing
        x2 = vcat(collect(float.(x2)), float(first(x2)) + float(period))
        values = hcat(values, values[:, 1:1])
    end
    # Corner Cartesian positions over the coordinate rectangle set the raster
    # bounds. Sampling the four edges is enough for the monotone maps here.
    xs = Float64[]; ys = Float64[]
    for (c1, c2) in ((first(x1), first(x2)), (first(x1), last(x2)),
                     (last(x1), first(x2)), (last(x1), last(x2)))
        for t in range(0, 1; length=17)
            X, Y = _slice_cartesian(metric, a, b, c1 + t * (last(x1) - c1), c2)
            push!(xs, X); push!(ys, Y)
            X2, Y2 = _slice_cartesian(metric, a, b, c1, c2 + t * (last(x2) - c2))
            push!(xs, X2); push!(ys, Y2)
        end
    end
    xlo, xhi = extrema(xs); ylo, yhi = extrema(ys)
    X = collect(range(xlo, xhi; length=n))
    Y = collect(range(ylo, yhi; length=n))
    grid = Base.fill(Float64(fill), n, n)

    c1lo, c1hi = first(x1), last(x1)
    c2lo, c2hi = first(x2), last(x2)
    for jj in 1:n, ii in 1:n
        c1, c2 = _slice_coordinate(metric, a, b, X[ii], Y[jj])
        (c1lo <= c1 <= c1hi && c2lo <= c2 <= c2hi) || continue
        grid[ii, jj] = _bilinear(x1, x2, values, c1, c2)
    end
    return X, Y, grid
end

function cartesian_slice(solver::Solver, dims, x1, x2, values; period=:auto, kwargs...)
    a, b = dims
    if period === :auto
        # The azimuth of a polar (r, θ) or (θ, φ) plane closes the raster when
        # that dimension is decomposed as periodic over its full extent.
        period = (solver.decomp.periodic[b] && solver.decomp.active[b]) ?
                 solver.decomp.n_global[b] * solver.h[b] : nothing
    end
    return cartesian_slice(solver.metric, dims, x1, x2, values; period=period, kwargs...)
end

"""
    revolve_profile(radius, values; n = 400, fill = NaN) -> (axis, disk)

Revolve a one-dimensional radial profile into a two-dimensional axisymmetric
raster: the `(axis, disk)` pair a heatmap draws as a disk of radius
`radius[end]`, with `values` interpolated radially and `fill` (default `NaN`,
transparent) outside. This is the view for a collapsed radial calculation, such
as the azimuthally symmetric run of the [radial coordinate tutorial](@ref
"Setting up a radial acoustic pulse") or the converging shock, where there is no
resolved angle to slice with [`field_slice`](@ref). The field is a function of
``r`` alone and the disk is its surface of revolution.

`axis` is the length-`n` Cartesian axis spanning `[-radius[end], radius[end]]`
and `disk` the `n × n` raster over it. `radius` must be sorted ascending and of
the same length as `values`; a length mismatch throws `ArgumentError`. Radii
inside `radius[1]` take `values[1]`. Use [`line_profile`](@ref) to obtain
the `(radius, values)` pair from a running solver. This is a pure function and
performs no communication.
"""
function revolve_profile(radius::AbstractVector, values::AbstractVector;
                         n::Int=400, fill=NaN)
    length(radius) == length(values) ||
        throw(ArgumentError("revolve_profile: radius and values differ in length"))
    outer = float(last(radius))
    axis = collect(range(-outer, outer; length=n))
    disk = Base.fill(Float64(fill), n, n)
    for j in eachindex(axis), i in eachindex(axis)
        r = hypot(axis[i], axis[j])
        r <= outer || continue
        if r <= radius[1]
            disk[i, j] = values[1]
        elseif r >= radius[end]
            disk[i, j] = values[end]
        else
            lo = searchsortedlast(radius, r)
            w = (r - radius[lo]) / (radius[lo + 1] - radius[lo])
            disk[i, j] = (1 - w) * values[lo] + w * values[lo + 1]
        end
    end
    return axis, disk
end

# Coordinate pair (along dims a, b) → in-plane Cartesian (X, Y). Each method is
# the inverse of the matching `_slice_coordinate` below, and the pair must stay
# consistent: a meridian embeds in a plane spanned by two Cartesian axes that
# are not `(a, b)` themselves (the spherical (r, θ) meridian lives in x–z with
# the pole vertical, not x–y), so slicing the full 3-D position by `(a, b)` is
# wrong and would collapse the plane onto a line.
_slice_cartesian(::CartesianMetric, a::Int, b::Int, ca, cb) = (ca, cb)
function _slice_cartesian(::CylindricalMetric, a::Int, b::Int, ca, cb)
    if (a, b) == (1, 2)      # (r, θ) disk in the x–y plane
        return ca * cos(cb), ca * sin(cb)
    end
    return ca, cb            # (r, z) or (θ, z): Cartesian in one axis
end
function _slice_cartesian(::SphericalMetric, a::Int, b::Int, ca, cb)
    if (a, b) == (1, 2)      # (r, θ_polar) meridian: X = r·sinθ, Y = r·cosθ
        return ca * sin(cb), ca * cos(cb)
    end
    return ca, cb
end

# Inverse of `_slice_cartesian` for the supported metrics: Cartesian identity,
# and the polar/meridional (r, angle) inversion via hypot/atan.
function _slice_coordinate(::CartesianMetric, a::Int, b::Int, X, Y)
    return X, Y
end
function _slice_coordinate(::CylindricalMetric, a::Int, b::Int, X, Y)
    # In-plane dims are (1, 2) = (r, θ); z-normal slice. Other orientations of a
    # cylindrical slice are Cartesian in one axis.
    if (a, b) == (1, 2)
        return hypot(X, Y), _wrap_angle(atan(Y, X))
    end
    return X, Y
end
function _slice_coordinate(::SphericalMetric, a::Int, b::Int, X, Y)
    # (1, 2) = (r, θ_polar) meridional plane: X is r·sinθ, Y is r·cosθ.
    if (a, b) == (1, 2)
        return hypot(X, Y), atan(X, Y)
    end
    return X, Y
end

_wrap_angle(θ) = θ < 0 ? θ + 2π : θ

# Bilinear interpolation of `values` (indexed by x1, x2 nodes) at (c1, c2).
function _bilinear(x1::AbstractVector, x2::AbstractVector, values::AbstractMatrix,
                   c1, c2)
    i = clamp(searchsortedlast(x1, c1), 1, length(x1) - 1)
    j = clamp(searchsortedlast(x2, c2), 1, length(x2) - 1)
    t = (c1 - x1[i]) / (x1[i + 1] - x1[i])
    u = (c2 - x2[j]) / (x2[j + 1] - x2[j])
    v00 = values[i, j];     v10 = values[i + 1, j]
    v01 = values[i, j + 1]; v11 = values[i + 1, j + 1]
    return (1 - t) * (1 - u) * v00 + t * (1 - u) * v10 +
           (1 - t) * u * v01 + t * u * v11
end

# --- Makie plotting interface (implemented in ext/CompactLESMakieExt.jl) ---------

_makie_extension() = Base.get_extension(@__MODULE__, :CompactLESMakieExt)

"""
    makie_available() -> Bool

Whether the Makie extension is loaded. Importing a supported Makie backend,
such as CairoMakie or GLMakie, alongside CompactLES activates the package
extension.
"""
makie_available() = _makie_extension() !== nothing

function _makie_required(name)
    error("$name requires the Makie extension. Add a Makie backend to your " *
          "environment and write `using CairoMakie` (or `using GLMakie`) " *
          "alongside `using CompactLES`; the extension loads itself.")
end

"""
    profileplot(solver, Q, name; dim = 1, species = 1, figure = (;), axis = (;),
                kwargs...) -> (figure, axis, plot)
    profileplot!(axis, solver, Q, name; dim = 1, species = 1, kwargs...) -> plot

Plot the [`line_profile`](@ref) of the named scalar along `dim`. `profileplot`
builds a figure with axis labels drawn from the metric (`r`/`θ`/`z` for
cylindrical, `r`/`θ`/`φ` for spherical, `x`/`y`/`z` for Cartesian);
`profileplot!` draws into an existing axis. On `profileplot`, `figure` and `axis`
are keyword collections forwarded to Makie's `Figure` and `Axis`. Extra keyword
arguments pass through to Makie's `lines!`.

Requires a Makie backend (see [`makie_available`](@ref)). Extraction through
[`field_array`](@ref) requires every rank to call this function. The profile is
replicated, so both forms return a plot on every rank; callers normally draw it
only on rank 0.
"""
profileplot(args...; kwargs...) = _makie_required("profileplot")
profileplot!(args...; kwargs...) = _makie_required("profileplot!")
@doc (@doc profileplot) profileplot!

"""
    fieldheatmap(solver, Q, name; normal = 3, index = 1, species = 1,
                 figure = (;), axis = (;), colorbar = true, kwargs...)
        -> (figure, axis, plot)
    fieldheatmap!(axis, solver, Q, name; normal = 3, index = 1, species = 1,
                  kwargs...) -> plot

Plot the [`field_slice`](@ref) of the named scalar as a heatmap. On a grid with
a resolved angular dimension the plane is resampled onto a Cartesian raster with
[`cartesian_slice`](@ref) and drawn with an equal data aspect, so a cylindrical
`(r, θ)` plane renders as its physical disk; on every other grid the coordinate
axes are used directly. On `fieldheatmap`, `figure` and `axis` are keyword
collections forwarded to Makie's `Figure` and `Axis`, and `colorbar = false`
omits the colorbar. Extra keyword arguments pass through to Makie's `heatmap!`.

Requires a Makie backend (see [`makie_available`](@ref)). Every rank must call
this function. The plot is produced on rank 0, and both forms return `nothing`
on other ranks.
"""
fieldheatmap(args...; kwargs...) = _makie_required("fieldheatmap")
fieldheatmap!(args...; kwargs...) = _makie_required("fieldheatmap!")
@doc (@doc fieldheatmap) fieldheatmap!

# Axis labels for a coordinate direction, by metric. Used by the extension.
coordinate_label(::CartesianMetric, d::Int) = ("x", "y", "z")[d]
coordinate_label(::CylindricalMetric, d::Int) = ("r", "θ", "z")[d]
coordinate_label(::SphericalMetric, d::Int) = ("r", "θ", "φ")[d]
