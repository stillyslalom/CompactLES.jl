# Geometry- and variable-aware extraction of report fields for visualization,
# and the Makie plotting seam layered on top of it.
#
# The tutorials used to each redefine a private `density_line(solver, Q)` that
# sampled `mixture_density` along dimension 1 at `(i, 1, 1)`. That helper was
# rank-local — correct only when the sampled line lived on one rank — and hard
# wired to one variable and one geometry. The functions here replace it with a
# collective, geometry-aware API that resolves any named scalar through the same
# catalog `save_vtk` uses (`scalar_field` in io.jl), so a profile or a slice is
# one call and means the same thing under MPI decomposition as in serial.
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

`name` is resolved through [`scalar_field`](@ref) — the same catalog
[`save_vtk`](@ref) uses — so the available names are identical: stored
primitives (`:rho`, `:p`, `:T_ion`, `:c`, `:u`, `:v`, `:w`), the per-species
`:Y` and `:D_art` (selected with `species`), and derived scalars (`:mach`,
`:divergence`, `:vorticity_magnitude`, `:qcriterion`, `:schlieren`,
`:strain_mag`, `:sensor`, `:mu_art`, `:beta_art`, `:kappa_art`).

`species` selects the array for `:Y` and `:D_art` and is ignored by every other
name.

Collective: it calls [`refresh_primitives!`](@ref), and a derived name runs the
gradient pass (and `compute_artificial!` for an artificial coefficient) that the
name needs. The gradient pass overwrites `solver.grad_u`, and the artificial
pass additionally overwrites the artificial coefficient and sensor fields and the
`tmp_a`/`tmp_b` scratch; `compute_rhs!` rebuilds all of them at every stage. The
returned array is a fresh copy, safe to keep past the next step.
"""
function field_array(solver::Solver, Q, name::Symbol; species::Int=1)
    if _wants_gradients(name) || _wants_artificial(name)
        # This refreshes primitives into the halos as part of the gradient pass,
        # so it subsumes `refresh_primitives!`.
        compute_primitives_and_gradients!(solver, Q)
        _wants_artificial(name) && compute_artificial!(solver, Q)
    else
        refresh_primitives!(solver, Q)
    end
    # An eltype-preserving copy: extraction must not hardcode Float64, since
    # the storage type is a backend decision.
    return Array(scalar_field(solver, name; species=species))
end

# --- Line profiles ----------------------------------------------------------

"""
    line_profile(solver, Q, name; dim = 1, species = 1) -> (coord, value)

A one-dimensional profile of the named scalar along `dim`, returned as the pair
of vectors `(coord, value)`, both of length `n_global[dim]` and identical on
every rank. This is the replacement for the tutorials' old `density_line`:
`line_profile(solver, Q, :rho)` is the mixture-density profile along the first
axis.

The coordinate comes from [`profile_coordinate`](@ref) and the value from
[`plane_profile`](@ref), so the semantics are that file's: `value` is the
area-weighted average over the two dimensions transverse to `dim`. When those
dimensions are collapsed — the common tutorial case of `n_global = (N, 1, 1)` —
the average is over a single point and the profile is exactly the line through
`(i, 1, 1)`. With a transverse dimension resolved it is the correct area mean,
which is usually what a radial or axial profile of a multidimensional field
should show. For a line integral rather than an average, weight by
[`profile_spacing`](@ref).

Collective (see [`field_array`](@ref)).
"""
function line_profile(solver::Solver, Q, name::Symbol; dim::Int=1, species::Int=1)
    1 <= dim <= 3 || throw(ArgumentError("line_profile: dim must be 1, 2, or 3"))
    f = field_array(solver, Q, name; species=species)
    coord = profile_coordinate(solver, dim)
    value = plane_profile(solver, f, dim)
    return coord, value
end

# --- Two-dimensional slices -------------------------------------------------

"""
    field_slice(solver, Q, name; normal = 3, index = 1, species = 1)
        -> (x1, x2, values) or nothing

The plane of the named scalar transverse to dimension `normal`, taken at the
global index `index` along `normal`. Returns, on rank 0, the two in-plane
coordinate vectors and the `length(x1) × length(x2)` matrix of values; returns
`nothing` on every other rank, since a slice is a rank-0 gather rather than a
replicated reduction. The in-plane axes are the two dimensions other than
`normal`, in increasing order (for `normal = 3` they are dimensions 1 and 2).

The coordinates are the metric's own, `(r, θ)`, `(r, φ)`, or a Cartesian pair,
so a curvilinear slice is on a coordinate surface. Pass the result through
[`cartesian_slice`](@ref), along with the in-plane dimension pair, to resample it
onto a Cartesian grid for a heatmap.

`index` is a global index and must lie in `1:n_global[normal]`; an index outside
that range throws `ArgumentError`, as does a `normal` outside `1:3`.

Collective (see [`field_array`](@ref)); every rank must call it, including a rank
whose block holds no part of the requested plane.
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
`CartesianMetric` plane and a spherical `(r, φ)` one, is treated as already
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
    # interpolated across rather than left blank.
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
# are not `(a, b)` themselves — the spherical (r, θ) meridian lives in x–z with
# the pole vertical, not x–y — so slicing the full 3-D position by `(a, b)` is
# wrong and would collapse the plane onto a line.
_slice_cartesian(::CartesianMetric, a::Int, b::Int, ca, cb) = (ca, cb)
function _slice_cartesian(::CylindricalMetric, a::Int, b::Int, ca, cb)
    if (a, b) == (1, 2)      # (r, θ) disk in the x–y plane
        return ca * cos(cb), ca * sin(cb)
    end
    return ca, cb            # (r, z) or (θ, z): already Cartesian in one axis
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
    # cylindrical slice are already Cartesian in one axis.
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

# --- Makie plotting seam (implemented in ext/CompactLESMakieExt.jl) ---------

_makie_extension() = Base.get_extension(@__MODULE__, :CompactLESMakieExt)

"""
    makie_available() -> Bool

Whether the Makie extension is loaded. It loads once the caller has a Makie
backend available and writes `using CairoMakie` (or `using GLMakie`) alongside
`using CompactLES`.
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

Requires a Makie backend (see [`makie_available`](@ref)). Collective, because it
extracts through [`field_array`](@ref), so every rank must call it. A profile is
replicated across ranks, so both forms return a plot on every rank; drawing on
rank 0 is the convention.
"""
profileplot(args...; kwargs...) = _makie_required("profileplot")
profileplot!(args...; kwargs...) = _makie_required("profileplot!")

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

Requires a Makie backend (see [`makie_available`](@ref)). Collective, so every
rank must call it; the plot is produced on rank 0 and both forms return
`nothing` on other ranks.
"""
fieldheatmap(args...; kwargs...) = _makie_required("fieldheatmap")
fieldheatmap!(args...; kwargs...) = _makie_required("fieldheatmap!")

# Axis labels for a coordinate direction, by metric. Used by the extension.
coordinate_label(::CartesianMetric, d::Int) = ("x", "y", "z")[d]
coordinate_label(::CylindricalMetric, d::Int) = ("r", "θ", "z")[d]
coordinate_label(::SphericalMetric, d::Int) = ("r", "θ", "φ")[d]
