# Geometry- and variable-aware extraction of report fields for visualization,
# and the Makie plotting interface layered on top of it.
#
# The functions here are a collective, geometry-aware API that resolves any
# named scalar through the same catalog `save_vtk` uses (`scalar_field` in
# io.jl), so a profile or a slice is one call with the same meaning under MPI
# decomposition as in serial. A rank-local sampler wired to one variable and
# one geometry is correct only where the sampled line lives on one rank.
#
# Nothing here depends on a plotting package. `profileplot`, `fieldheatmap`,
# `meshplot`, and their mutating forms are declared as stubs that error until a
# Makie backend is loaded; `ext/CompactLESMakieExt.jl` supplies the methods.
# This mirrors the HDF5 split in src/hdf5.jl exactly.

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
`:Y`, `:X` (mole fraction) and `:D_art` (selected with `species`), and derived
scalars (`:mach`, `:divergence`, `:vorticity_magnitude`, `:qcriterion`,
`:schlieren`, `:strain_mag`, `:sensor`, `:mu_art`, `:beta_art`, `:kappa_art`).

`species` selects the array for `:Y`, `:X` and `:D_art` and is ignored by every
other name.

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

"""
    field_array(solver, states::Vector, name::Symbol; species = 1) -> Vector{Array}

The multi-patch form, for the state vector of a refined or patched run: one
refreshed copy of the named field per patch this rank holds, aligned with
`solver.patches`. This is the form the composite diagnostics take
([`plane_profile`](@ref), [`volume_integral`](@ref)), and the state-vector forms
of [`line_profile`](@ref), [`line_sample`](@ref) and [`field_slice`](@ref) call
it. Every rank must call it. The primitives are refreshed from `states`, and the
artificial coefficient arrays are restored as in the single-state form.
"""
function field_array(solver::Solver, states::Vector{<:ConservedState}, name::Symbol;
                     species::Int=1)
    patches = getfield(solver, :patches)
    return preserving_artificial(solver, _wants_artificial(name)) do
        map(eachindex(patches)) do li
            ps = PatchSolver(solver, patches[li])
            Q = states[li]
            if _wants_gradients(name) || _wants_artificial(name)
                compute_primitives_and_gradients!(ps, Q)
                _wants_artificial(name) && compute_artificial!(ps, Q)
            else
                refresh_primitives!(ps, Q)
            end
            Array(scalar_field(ps, name; species=species))
        end
    end
end

"""
    volume_integral(solver, Q, name::Symbol; species = 1) -> Float64

∫ f dV of the named field, for a single state or the state vector of a
refined run, through [`field_array`](@ref). Every rank must call it.
"""
volume_integral(solver::Solver, Q::Union{ConservedState,Vector{<:ConservedState}},
                name::Symbol; species::Int=1) =
    volume_integral(solver, field_array(solver, Q, name; species=species))

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
    line_profile(solver, states::Vector, name; dim = 1, species = 1) -> (coord, value)

The composite profile of a refined or patched run, at the root's stations:
each plane average combines the uncovered root nodes with the coinciding
nodes of every finer patch, as the composite [`plane_profile`](@ref) does.
"""
function line_profile(solver::Solver, states::Vector{<:ConservedState}, name::Symbol;
                      dim::Int=1, species::Int=1)
    1 <= dim <= 3 || throw(ArgumentError("line_profile: dim must be 1, 2, or 3"))
    fs = field_array(solver, states, name; species=species)
    return profile_coordinate(solver, dim), plane_profile(solver, fs, dim)
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
[`field_array`](@ref). A refined or patched solver takes the state vector
instead of `Q`, and this form throws `ArgumentError` on one.
"""
function line_sample(solver::Solver, Q, name::Symbol; dim::Int=1,
                     index::Union{Nothing,NTuple{2,Int}}=nothing,
                     at::Union{Nothing,NTuple{2,Real}}=nothing, species::Int=1)
    1 <= dim <= 3 || throw(ArgumentError("line_sample: dim must be 1, 2, or 3"))
    index === nothing || at === nothing ||
        throw(ArgumentError("line_sample: give index or at, not both"))
    _composite(solver) && throw(_state_vector_error("line_sample"))
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

"""
    line_sample(solver, states::Vector, name; dim = 1, index = (1, 1), at = nothing,
                species = 1) -> (coord, value)

The composite form, for the state vector of a refined or patched run. The line
runs through the root grid's nodes: `index` and `at` are root global indices
and coordinates, and `coord` is the root coordinate. Each value is the field
on the finest level that holds the root node, where a patch holds the nodes of
its own boundary planes. Where several patches of that level hold the node, on
a shared plane or a periodic seam, the value is their mean. The fine nodes
between root nodes are not sampled. Every rank of `solver.comm` must
call this function, including a rank that holds no patch of a refined level.
"""
function line_sample(solver::Solver, states::Vector{<:ConservedState}, name::Symbol;
                     dim::Int=1, index::Union{Nothing,NTuple{2,Int}}=nothing,
                     at::Union{Nothing,NTuple{2,Real}}=nothing, species::Int=1)
    1 <= dim <= 3 || throw(ArgumentError("line_sample: dim must be 1, 2, or 3"))
    index === nothing || at === nothing ||
        throw(ArgumentError("line_sample: give index or at, not both"))
    n_global = solver.n_global
    a, b = _plane_dims(dim)
    if at !== nothing
        ga = _nearest_index(_root_coordinates(solver, a), at[1])
        gb = _nearest_index(_root_coordinates(solver, b), at[2])
    else
        ga, gb = index === nothing ? (1, 1) : index
    end
    for (d, g) in ((a, ga), (b, gb))
        1 <= g <= n_global[d] ||
            throw(ArgumentError("line_sample: index $g out of range " *
                                "1:$(n_global[d]) along dimension $d"))
    end
    fs = field_array(solver, states, name; species=species)
    fixed = ntuple(d -> d == a ? ga : d == b ? gb : 0, 3)
    acc = MPI.Allreduce(_composite_samples(solver, fs, fixed), +, solver.comm)
    T = eltype(getfield(solver, :h))
    return _root_coordinates(solver, dim), T.(vec(_finest_values(acc)))
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

# The same rule over a coordinate vector.
function _nearest_index(xs::AbstractVector, x::Real)
    best, dist = 1, Inf
    for (g, xg) in enumerate(xs)
        e = abs(xg - x)
        e < dist && ((best, dist) = (g, e))
    end
    return best
end

_state_vector_error(name) =
    ArgumentError("$name: this solver holds several patches or levels; pass the " *
                  "state vector allocate_state returns")

# --- Composite point samples -------------------------------------------------
#
# A point sample of a refined or patched run is taken at root nodes, the
# stations the composite profiles use. A level-ℓ node m (that level's node
# space) lies on root node (m − 1) / 3^ℓ + 1 when 3^ℓ divides m − 1, wrapped
# onto the root's range at a periodic seam. Per root node and level the held
# values are summed and counted on this rank; after the reduction the value
# is the mean over the finest level with a nonzero count. Choosing by level
# rather than by the covered masks keeps a child's face nodes, which the
# parent's mask marks as partly covered, on the child's value, and needs no
# case for the corner and edge nodes of a tile nest.

# Sums and counts, as a (2, nlevels, m1, m2, m3) array: `fixed[d]` is the root
# index the sample is taken at along `d`, or 0 for every root node along `d`
# (then m_d = n_global[d], otherwise 1).
function _composite_samples(solver::Solver, fs::Vector, fixed::NTuple{3,Int})
    n_global = solver.n_global
    m = ntuple(d -> fixed[d] == 0 ? n_global[d] : 1, 3)
    acc = zeros(Float64, 2, nlevels(solver), m...)
    for (li, p) in enumerate(getfield(solver, :patches))
        decomp = p.decomp
        stride = 3^p.level
        # Per dimension, the (local interior index, output index) pairs.
        picks = ntuple(3) do d
            out = Tuple{Int,Int}[]
            for il in 1:decomp.n_local[d]
                node = p.region.offset[d] + decomp.offset[d] + il
                (node - 1) % stride == 0 || continue
                g = mod1((node - 1) ÷ stride + 1, n_global[d])
                if fixed[d] == 0
                    push!(out, (il, g))
                elseif g == fixed[d]
                    push!(out, (il, 1))
                end
            end
            out
        end
        _add_samples!(acc, fs[li], p.level + 1, decomp.n_halo_d, picks...)
    end
    return acc
end

function _add_samples!(acc, f, level, o, p1, p2, p3)
    @inbounds for (k, g3) in p3, (j, g2) in p2, (i, g1) in p1
        acc[1, level, g1, g2, g3] += f[i + o[1], j + o[2], k + o[3]]
        acc[2, level, g1, g2, g3] += 1
    end
    return acc
end

# The reduced samples: per node, the mean over the finest level holding it.
function _finest_values(acc::Array{Float64})
    out = Array{Float64}(undef, size(acc)[3:end])
    for I in CartesianIndices(out)
        level = findlast(l -> acc[2, l, I] > 0, 1:size(acc, 2))
        level === nothing &&
            error("composite sample: no patch holds root node $(Tuple(I))")
        out[I] = acc[1, level, I] / acc[2, level, I]
    end
    return out
end

# Root-grid coordinate of every node along `d`, local to each rank.
_root_coordinates(solver::Solver, d::Int) =
    Float64[_root_xcoord(solver, d, g) for g in 1:solver.n_global[d]]

function _root_xcoord(solver::Solver, d::Int, g::Int)
    ξ = getfield(solver, :origin)[d] + getfield(solver, :coord_shift)[d] +
        (g - 1) * getfield(solver, :h)[d]
    stretch = getfield(solver, :stretch)[d]
    return stretch === nothing ? ξ : stretch.x(ξ)
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
of the requested plane; see [`field_array`](@ref). A refined or patched solver
takes the state vector instead of `Q`, and this form throws `ArgumentError` on
one.
"""
function field_slice(solver::Solver, Q, name::Symbol; normal::Int=3, index::Int=1,
                     species::Int=1)
    1 <= normal <= 3 || throw(ArgumentError("field_slice: normal must be 1, 2, or 3"))
    _composite(solver) && throw(_state_vector_error("field_slice"))
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

"""
    field_slice(solver, states::Vector, name; normal = 3, index = 1, species = 1)
        -> (x1, x2, values) or nothing

The composite form, for the state vector of a refined or patched run. The
plane is the root grid's plane at root global index `index`, and each value is
taken as in the composite [`line_sample`](@ref): the field on the finest level
that holds the root node, averaged where several patches of that level hold
it. The result is on rank 0 and `nothing` on every other rank. Every rank of
`solver.comm` must call this function.
"""
function field_slice(solver::Solver, states::Vector{<:ConservedState}, name::Symbol;
                     normal::Int=3, index::Int=1, species::Int=1)
    1 <= normal <= 3 || throw(ArgumentError("field_slice: normal must be 1, 2, or 3"))
    n_global = solver.n_global
    ng = n_global[normal]
    1 <= index <= ng ||
        throw(ArgumentError("field_slice: index $index out of range 1:$ng along " *
                            "dimension $normal"))
    fs = field_array(solver, states, name; species=species)
    fixed = ntuple(d -> d == normal ? index : 0, 3)
    acc = MPI.Reduce(_composite_samples(solver, fs, fixed), +, solver.comm; root=0)
    a, b = _plane_dims(normal)
    MPI.Comm_rank(solver.comm) == 0 || return nothing
    plane = reshape(_finest_values(acc), n_global[a], n_global[b])
    return _root_coordinates(solver, a), _root_coordinates(solver, b), plane
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
  `n1 × n2 × n3` array. `:Y`, `:X` and `:D_art` carry a fourth dimension over
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
    field_snapshot(solver, Q; fields = DEFAULT_VTK_FIELDS, normal = nothing,
                   index = 1) -> FieldSnapshot or nothing
    field_snapshot(solver, states::Vector; fields = DEFAULT_VTK_FIELDS,
                   normal = nothing, index = 1) -> Vector{FieldSnapshot} or nothing

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

Where the partial densities of the state do not sum to a positive density,
`:rho` holds that sum as it is and every other field is `NaN`. The recovered
primitives hold placeholders there (ρ = 1, unit pressure, zero velocity), which
would show a failed state as a quiet one.

`normal` restricts the gather to one plane: the nodes at root global index
`index` along dimension `normal`, as in [`field_slice`](@ref). Each snapshot
then has one node along `normal`, and none is returned for a patch that does
not reach the plane. On a refined level the plane is the level's node
coinciding with the root node, so the snapshots of a refined run hold each
patch's plane at its own spacing, where the composite `field_slice` keeps only
the root's nodes.

Every rank of `solver.comm` must call this function with the same `fields`,
`normal` and `index`, since the derived fields run distributed solves in the
order given.
"""
function field_snapshot(solver::Solver, Q; fields=DEFAULT_VTK_FIELDS,
                        normal::Union{Nothing,Int}=nothing, index::Int=1)
    patches = getfield(solver, :patches)
    length(patches) == 1 && nlevels(solver) == 1 &&
        only(patches).region.extent == solver.n_global ||
        throw(ArgumentError("field_snapshot: this solver holds several patches; " *
                            "pass the state vector allocate_state returns"))
    snaps = _snapshot(solver, [Q], fields, _snapshot_plane(solver, normal, index))
    return snaps === nothing ? nothing : only(snaps)
end

function field_snapshot(solver::Solver, states::Vector{<:ConservedState};
                        fields=DEFAULT_VTK_FIELDS,
                        normal::Union{Nothing,Int}=nothing, index::Int=1)
    return _snapshot(solver, states, fields, _snapshot_plane(solver, normal, index))
end

const _SNAPSHOT_STACKED = (:velocity, :vorticity, :Y, :X, :D_art)

# The validated `(normal, index)` of a plane-restricted snapshot, or nothing.
function _snapshot_plane(solver::Solver, normal, index::Int)
    normal === nothing && return nothing
    1 <= normal <= 3 ||
        throw(ArgumentError("field_snapshot: normal must be 1, 2, or 3"))
    ng = solver.n_global[normal]
    1 <= index <= ng ||
        throw(ArgumentError("field_snapshot: index $index out of range 1:$ng " *
                            "along dimension $normal"))
    return (normal, index)
end

# The node of a level's own node space coinciding with root node `g` along
# `d`: a resolved dimension is refined threefold per level, a collapsed one
# keeps its single node.
_level_node(n_global::NTuple{3,Int}, level::Int, d::Int, g::Int) =
    n_global[d] > 1 ? (g - 1) * 3^level + 1 : g

function _snapshot(solver::Solver, states, fields, plane=nothing)
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
            # The plane as the node of this patch's level coinciding with it.
            at = plane === nothing ? nothing :
                 (plane[1], _level_node(solver.n_global, patches[li].level, plane...))
            _snapshot_block(ps, states[li], names, at)
        end
    end
    # A rank whose blocks miss the plane contributes none.
    gathered = MPI.gather(filter(!isnothing, blocks), solver.comm; root=0)
    MPI.Comm_rank(solver.comm) == 0 || return nothing
    return _assemble_snapshots(solver, reduce(vcat, gathered), names)
end

# This rank's interior block of one patch, with its placement in the patch;
# with a `plane`, a `(normal, node)` pair in the patch level's node space, the
# block's nodes on that plane, or nothing where the block misses it.
function _snapshot_block(ps::PatchSolver, Q, names, plane=nothing)
    decomp = ps.decomp
    region = ps.patch.region
    n = decomp.n_local
    ranges = ntuple(d -> 1:n[d], 3)
    offset, extent, lo = region.offset, region.extent, decomp.offset
    if plane !== nothing
        normal, node = plane
        il = node - region.offset[normal] - decomp.offset[normal]
        1 <= il <= n[normal] || return nothing
        ranges = Base.setindex(ranges, il:il, normal)
        offset = Base.setindex(offset, node - 1, normal)
        extent = Base.setindex(extent, 1, normal)
        lo = Base.setindex(lo, 0, normal)
    end
    interior = ntuple(d -> decomp.n_halo_d[d] .+ ranges[d], 3)
    # Device storage is copied to the host whole before the interior is cut.
    grab(a) = (a isa Array ? a : Array(a))[interior...]
    stacked(arrays) = cat(map(grab, arrays)...; dims=4)
    data = map(names) do name
        name === :velocity && return stacked((ps.u, ps.v, ps.w))
        name === :vorticity && return stacked(_vorticity_arrays(ps))
        name === :Y && return stacked(ps.Y)
        name === :X && return stacked(ntuple(sp -> _mole_fraction_array(ps, sp),
                                             ps.equations.n_species))
        name === :D_art && return stacked(ps.D_art)
        return grab(scalar_field(ps, name))
    end
    # The density the state holds, summed from the partial densities in the
    # order `mixture_density` sums them; where it is not positive the
    # primitives are placeholders, so `:rho` takes the sum and the rest NaN.
    Qa = parent(Q) isa Array ? parent(Q) : Array(parent(Q))
    Qh = Qa[interior..., 1:ps.equations.n_species]
    ρ = Qh[:, :, :, 1]
    for sp in 2:ps.equations.n_species
        ρ .+= view(Qh, :, :, :, sp)
    end
    bad = .!(ρ .> 0)
    if any(bad)
        for (name, a) in zip(names, data)
            for c in axes(a, 4)
                v = view(a, :, :, :, c)
                v[bad] .= name === :rho ? ρ[bad] : eltype(a)(NaN)
            end
        end
    end
    T = eltype(ps.rho)
    coords = ntuple(d -> T[xcoord(ps, d, i) for i in ranges[d]], 3)
    return (level=ps.patch.level, offset=offset, extent=extent,
            lo=lo, coords=coords, data=collect(data),
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

`axis = true` declares a cylindrical `(r, θ)` plane whose radial grid passes
through the origin, as an [`AxisBC`](@ref) grid does with its first node half a
spacing off the axis. A raster point inside `x1[1]` then lies on the diameter
joining the first node at its angle to the first node at the opposite angle,
and takes the linear interpolant between the two, so the disk is filled
through its center. It requires `period`. The `solver`-taking method sets it
for a cylindrical plane on an axis grid.
"""
function cartesian_slice(metric::Metric, dims::Tuple{Int,Int},
                         x1::AbstractVector, x2::AbstractVector,
                         values::AbstractMatrix; n::Int=400, fill=NaN,
                         period::Union{Nothing,Real}=nothing, axis::Bool=false)
    axis && period === nothing &&
        throw(ArgumentError("cartesian_slice: axis = true requires period"))
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
        (c1 <= c1hi && c2lo <= c2 <= c2hi) || continue
        if c1 >= c1lo
            grid[ii, jj] = _bilinear(x1, x2, values, c1, c2)
        elseif axis
            # The opposite angle, wrapped into the stored range.
            c2o = c2lo + mod(c2 + π - c2lo, period)
            vp = _bilinear(x1, x2, values, c1lo, c2)
            vo = _bilinear(x1, x2, values, c1lo, c2o)
            grid[ii, jj] = ((c1lo + c1) * vp + (c1lo - c1) * vo) / (2 * c1lo)
        end
    end
    return X, Y, grid
end

function cartesian_slice(solver::Solver, dims, x1, x2, values; period=:auto,
                         axis=:auto, kwargs...)
    a, b = dims
    if period === :auto
        # The azimuth of a polar (r, θ) or (θ, φ) plane closes the raster when
        # that dimension is decomposed as periodic over its full extent.
        period = (_root_periodic(solver, b) && solver.n_global[b] > 1) ?
                 solver.n_global[b] * getfield(solver, :h)[b] : nothing
    end
    if axis === :auto
        axis = Tuple(dims) == (1, 2) && period !== nothing && _through_axis(solver)
    end
    return cartesian_slice(solver.metric, dims, x1, x2, values; period=period,
                           axis=axis, kwargs...)
end

# Whether the radial grid of a cylindrical solver passes through the axis. Only
# an axis fold shifts a cylindrical radius half a spacing off its origin, since
# a symmetry plane is rejected on r.
_through_axis(solver::Solver) =
    solver.metric isa CylindricalMetric && getfield(solver, :coord_shift)[1] > 0

# Whether the root grid is periodic along `d`. A slab layout (`patch_grid`)
# splits one periodic dimension into patches that are not periodic on their
# own; its last slab then reaches the wrapped node n_global[d] + 1.
function _root_periodic(solver::Solver, d::Int)
    getfield(solver, :patches)[1].decomp.periodic[d] && return true
    nlevels(solver) == 1 || return false
    return any(r -> r.offset[d] + r.extent[d] > solver.n_global[d],
               getfield(solver, :patch_regions))
end

"""
    revolve_profile(radius, values; n = 400, fill = NaN) -> (axis, disk)

Revolve a one-dimensional radial profile into a two-dimensional axisymmetric
raster: the `(axis, disk)` pair a heatmap draws as a disk of radius
`radius[end]`, with `values` interpolated radially and `fill` (default `NaN`,
transparent) outside. This is the view for a collapsed radial calculation, such
as the spherical blast wave of the [supernova remnant tutorial](@ref
"Supernova remnant") or the converging shock, where there is no
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

# --- Mesh geometry -------------------------------------------------------------
#
# The node lines of every patch in one plane, from the patch layout alone: the
# root patches' regions and the refined levels' regions, which rank 0 holds in
# full, so no field is read and nothing is communicated.

# Per patch meeting the root plane at `index` transverse to `normal`, in level
# then patch order: its level and the metric coordinates of its nodes along
# the two in-plane dimensions. `closed` marks a plane whose second in-plane
# dimension wraps around a resolved angle, where the last node joins the
# first across the seam, and `axis` a closed (r, θ) plane whose radial lines
# pass through the origin to the opposite angle.
function _mesh_blocks(solver::Solver, normal::Int, index::Int)
    1 <= normal <= 3 || throw(ArgumentError("meshplot: normal must be 1, 2, or 3"))
    n_global = solver.n_global
    1 <= index <= n_global[normal] ||
        throw(ArgumentError("meshplot: index $index out of range " *
                            "1:$(n_global[normal]) along dimension $normal"))
    a, b = _plane_dims(normal)
    n_global[a] > 1 && n_global[b] > 1 ||
        throw(ArgumentError("meshplot: the plane transverse to dimension $normal " *
                            "has a collapsed dimension; the mesh is drawn in a " *
                            "plane of two resolved dimensions"))
    blocks = NamedTuple{(:level, :x1, :x2, :closed, :axis),
                        Tuple{Int,Vector{Float64},Vector{Float64},Bool,Bool}}[]
    for level in 0:(nlevels(solver) - 1)
        h = _level_spacing(solver, level)
        for r in _level_plane_regions(solver, level, normal, index)
            x1, x2 = (Float64[_level_xcoord(solver, h, d, r.offset[d] + i)
                              for i in 1:r.extent[d]] for d in (a, b))
            closed = level == 0 && _curvilinear(solver) && _root_periodic(solver, b) &&
                     r.extent[b] == n_global[b]
            closed && push!(x2, x2[1] + n_global[b] * h[b])
            axis = closed && (a, b) == (1, 2) && r.offset[a] == 0 && _through_axis(solver)
            push!(blocks, (; level, x1, x2, closed, axis))
        end
    end
    return blocks
end

# Per level meeting the plane, in level order, the level's lattice over the
# box enclosing its patches in the plane: node coordinates `x1`, `x2` and cell
# edges `e1`, `e2` along the two in-plane dimensions, and `shown`, the nodes
# of that box a composite view draws at this level: inside one of its patches
# and with a cell no finer level covers entirely, the rule `field_snapshot`
# applies to its `covered` mask. `closed` and `axis` are as in `_mesh_blocks`;
# a closed lattice repeats its first column at the wrapped node.
function _mesh_levels(solver::Solver, normal::Int, index::Int)
    blocks = _mesh_blocks(solver, normal, index)
    n_global = solver.n_global
    active = ntuple(d -> n_global[d] > 1, 3)
    a, b = _plane_dims(normal)
    out = []
    for level in unique(blk.level for blk in blocks)
        regions = _level_plane_regions(solver, level, normal, index)
        lo = (minimum(r.offset[a] for r in regions), minimum(r.offset[b] for r in regions))
        hi = (maximum(r.offset[a] + r.extent[a] for r in regions),
              maximum(r.offset[b] + r.extent[b] for r in regions))
        m = (hi[1] - lo[1], hi[2] - lo[2])
        shown = falses(m)
        for r in regions
            shown[r.offset[a] - lo[1] .+ (1:r.extent[a]),
                  r.offset[b] - lo[2] .+ (1:r.extent[b])] .= true
        end
        # A node is covered when every orthant of its cell lies in a child
        # region, per orthant the union over the children, as `_fill_covered!`
        # builds the mask; a collapsed normal is spanned by every region.
        node_n = _level_node(n_global, level, normal, index)
        children = level + 1 < nlevels(solver) ? level_regions(solver, level + 1) :
                   BlockRegion[]
        orthants = zeros(UInt8, m)
        for r in children
            rlo = ntuple(d -> r.offset[d] + 1, 3)
            rhi = ntuple(d -> r.offset[d] + r.extent[d], 3)
            plus_n = !active[normal] || rlo[normal] <= node_n < rhi[normal]
            minus_n = !active[normal] || rlo[normal] < node_n <= rhi[normal]
            plus_n || minus_n || continue
            for j in max(rlo[b], lo[2] + 1):min(rhi[b], hi[2]),
                i in max(rlo[a], lo[1] + 1):min(rhi[a], hi[1])
                bits = UInt8(0)
                for o in 0:7
                    sa, sb, sn = isodd(o), isodd(o >> 1), isodd(o >> 2)
                    (sa ? i < rhi[a] : i > rlo[a]) && (sb ? j < rhi[b] : j > rlo[b]) &&
                        (sn ? plus_n : minus_n) && (bits |= UInt8(1) << o)
                end
                orthants[i - lo[1], j - lo[2]] |= bits
            end
        end
        shown .&= orthants .!= 0xff
        h = _level_spacing(solver, level)
        x1 = [_level_xcoord(solver, h, a, g) for g in (lo[1] + 1):hi[1]]
        x2 = [_level_xcoord(solver, h, b, g) for g in (lo[2] + 1):hi[2]]
        blk = first(blk for blk in blocks if blk.level == level)
        if blk.closed
            push!(x2, x2[1] + n_global[b] * h[b])
            shown = hcat(shown, shown[:, 1])
        end
        push!(out, (; level, x1, x2, e1=_cell_edges(solver, a, x1),
                    e2=_cell_edges(solver, b, x2; closed=blk.closed), shown,
                    blk.closed, blk.axis))
    end
    return out
end

# The regions of a level's patches meeting the plane, in the level's own node
# space, as `_mesh_blocks` selects them.
function _level_plane_regions(solver::Solver, level::Int, normal::Int, index::Int)
    n_global = solver.n_global
    regions = level == 0 ? getfield(solver, :patch_regions) :
        [_fine_region(lt) for lt in getfield(solver, :levels)[level + 1].transfers]
    node = _level_node(n_global, level, normal, index)
    return [r for r in regions
            if r.offset[normal] < node <= r.offset[normal] + r.extent[normal]]
end

# The edges of the cells centered on the nodes `x` along dimension `d`: the
# midpoints between nodes, and half a spacing past each end node as a heatmap
# places its cells, but never past the domain's boundary on a non-periodic
# dimension, where a node on a wall owns the half cell inside it. A folded end
# (an axis, a pole, a symmetry plane) lies half a spacing past its first node,
# so there the half-spacing edge is the fold itself. With `closed`, `x` ends
# with the wrapped first node, and the first edge is the last less a period.
function _cell_edges(solver::Solver, d::Int, x::AbstractVector; closed::Bool=false)
    mids = [(x[i] + x[i + 1]) / 2 for i in 1:(length(x) - 1)]
    closed && return vcat(x[1] - (x[end] - x[end - 1]) / 2, mids)
    lo = x[1] - (x[2] - x[1]) / 2
    hi = x[end] + (x[end] - x[end - 1]) / 2
    if !_root_periodic(solver, d)
        # Within round-off of the boundary, or beyond it, the edge is the
        # boundary: the fold case lands on it up to round-off.
        blo, bhi = _domain_bounds(solver, d)
        tol = 1e-9 * abs(x[end] - x[1])
        lo = lo < blo + tol ? oftype(lo, blo) : lo
        hi = hi > bhi - tol ? oftype(hi, bhi) : hi
    end
    return vcat(lo, mids, hi)
end

# The physical extent of the domain along `d`. A stretched dimension stores its
# computational coordinate on [0, 1].
function _domain_bounds(solver::Solver, d::Int)
    stretch = getfield(solver, :stretch)[d]
    stretch === nothing || return stretch.x(0.0), stretch.x(1.0)
    lo = getfield(solver, :origin)[d]
    return lo, lo + getfield(solver, :L_domain)[d]
end

# The node spacing of a level, divided by three per level as the constructor
# divides it, so coordinates computed from it repeat the patches' own.
function _level_spacing(solver::Solver, level::Int)
    h = getfield(solver, :h)
    for _ in 1:level
        h = ntuple(d -> solver.n_global[d] > 1 ? h[d] / 3 : h[d], 3)
    end
    return h
end

# The coordinate of node `m` of a level's own node space along `d`, given that
# level's spacing; `global_xcoord` with the spacing passed in.
function _level_xcoord(solver::Solver, h, d::Int, m::Int)
    ξ = getfield(solver, :origin)[d] + getfield(solver, :coord_shift)[d] + (m - 1) * h[d]
    stretch = getfield(solver, :stretch)[d]
    return stretch === nothing ? ξ : stretch.x(ξ)
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
arguments pass through to Makie's `lines!`. For a refined or patched run, pass
the state vector as `Q`; the profile is then the composite `line_profile`.

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
`(r, θ)` plane renders as its physical disk, filled through the origin on an
[`AxisBC`](@ref) grid. On every other grid the coordinate axes are used
directly, and each node is drawn as the cell [`meshplot`](@ref) draws around
it, which ends on a wall rather than half a spacing past it. On
`fieldheatmap`, `figure` and `axis` are keyword
collections forwarded to Makie's `Figure` and `Axis`, and `colorbar = false`
omits the colorbar. Extra keyword arguments pass through to Makie's `heatmap!`.

For a patched run, pass the state vector as `Q`; the plane is then the
composite `field_slice` on the root grid's nodes. For a refined run, each
level is drawn at its own spacing, from the plane-restricted
[`field_snapshot`](@ref), as one heatmap over the box enclosing the level's
patches: coarse levels first, with the nodes a finer level covers and the nodes
outside every patch left transparent. The plot is then a vector of heatmaps,
one per level, sharing one `colorrange` (the extrema of the drawn values unless
given) and colormap, so its first element serves a `Colorbar`. Overlay
[`meshplot!`](@ref) to show the patches and their cells.

Requires a Makie backend (see [`makie_available`](@ref)). Every rank must call
this function. The plot is produced on rank 0, and both forms return `nothing`
on other ranks.
"""
fieldheatmap(args...; kwargs...) = _makie_required("fieldheatmap")
fieldheatmap!(args...; kwargs...) = _makie_required("fieldheatmap!")
@doc (@doc fieldheatmap) fieldheatmap!

"""
    meshplot(solver; normal = 3, index = 1, cells = true, grid = true,
             outlines = true, color = nothing, linewidth = 0.5,
             outline_linewidth = 1.5, figure = (;), axis = (;), kwargs...)
        -> (figure, axis, plots)
    meshplot!(axis, solver; normal = 3, index = 1, cells = true, grid = true,
              outlines = true, color = nothing, linewidth = 0.5,
              outline_linewidth = 1.5, kwargs...) -> plots

Draw the grid in the plane at root global index `index` transverse to
dimension `normal`: the cells centered on the nodes of every patch that
reaches the plane, and the outline of each patch through its outermost nodes.
A cell is bounded by the midpoints between nodes and extends half a spacing
past a patch's outermost nodes, except at a wall, where it ends on the wall,
and at a coordinate fold such as [`AxisBC`](@ref), where it ends on the fold,
so the radial lines of an axis grid pass through the origin. A resolved angle
is mapped to Cartesian axes along curved lines.

On a refined run each level is drawn at its own spacing, and only where no
finer level is drawn: the composite mesh of the nodes the solution is read
from, with the tile outlines marking where the resolution changes. The cells,
the coordinates and the composite are those [`fieldheatmap`](@ref) draws on
the same plane, so `meshplot!` composes with a heatmap in the same axis:

```julia
fig, ax, hm = fieldheatmap(solver, states, :rho; colormap = :inferno)
meshplot!(ax, solver; grid = false)        # tile outlines over the density
```

`cells = false` draws the lines through the nodes instead of the cell
boundaries. `grid` and `outlines` are `true`, `false`, or a collection of
levels to draw (0 is the root), so `outlines = 1:2` outlines the refined tiles
without the domain boundary. `color` is one color for every level or a vector
indexed by level + 1; the default is gray for the root and Makie's Wong palette
for the refined levels. `linewidth` applies to the grid and
`outline_linewidth` to the outlines. Extra keyword arguments pass through to
Makie's `lines!`. `meshplot` builds a figure with an equal data aspect,
forwarding `figure` and `axis` to Makie's `Figure` and `Axis`. The plot is a
vector of `Lines`, the grid of every drawn level first and then the outlines,
root first.

The mesh comes from the patch layout, which rank 0 holds in full, and reads no
field, so this function communicates nothing and need not be called on every
rank. Both forms return `nothing` on ranks other than rank 0 of `solver.comm`.
A plane with a collapsed in-plane dimension throws `ArgumentError`.
Requires a Makie backend (see [`makie_available`](@ref)).
"""
meshplot(args...; kwargs...) = _makie_required("meshplot")
meshplot!(args...; kwargs...) = _makie_required("meshplot!")
@doc (@doc meshplot) meshplot!

# Axis labels for a coordinate direction, by metric. Used by the extension.
coordinate_label(::CartesianMetric, d::Int) = ("x", "y", "z")[d]
coordinate_label(::CylindricalMetric, d::Int) = ("r", "θ", "z")[d]
coordinate_label(::SphericalMetric, d::Int) = ("r", "θ", "φ")[d]
