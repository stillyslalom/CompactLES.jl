module CompactLESMakieExt

# Makie plotting for the extraction API declared in src/viz.jl. As with the HDF5
# extension, the core package declares `profileplot`, `fieldheatmap`,
# `meshplot`, and their mutating forms as stubs that error until a backend is
# loaded; this module adds the methods once the caller writes `using CairoMakie`
# (or `using GLMakie`).
#
# The extension holds no numerics. Every value it draws comes from the core
# `line_profile` / `field_slice` / `field_snapshot` / `cartesian_slice`, and
# every mesh line from `_mesh_blocks` and `_mesh_levels`, so a plot shows
# exactly the extracted data and the two cannot disagree. The extraction calls
# are collective; the figure is assembled on rank 0, where the gathered slice
# lives.

using CompactLES
using CompactLES: Solver, ConservedState, line_profile, field_slice, field_snapshot,
                  cartesian_slice, coordinate_label, nlevels, _plane_dims,
                  _curvilinear, _mesh_blocks, _mesh_levels, _slice_cartesian,
                  _level_spacing, _level_xcoord, _cell_edges
using Makie
using MPI

# --- Line profiles ----------------------------------------------------------

function CompactLES.profileplot!(ax, solver::Solver, Q, name::Symbol;
                                 dim::Int=1, species::Int=1, kwargs...)
    coord, value = line_profile(solver, Q, name; dim=dim, species=species)
    return lines!(ax, coord, value; kwargs...)
end

function CompactLES.profileplot(solver::Solver, Q, name::Symbol;
                                dim::Int=1, species::Int=1,
                                figure=(;), axis=(;), kwargs...)
    # Extraction is collective; every rank must reach it. Only rank 0 owns a
    # figure, but the profile itself is replicated, so a caller may render on
    # any rank; rank 0 is the convention.
    coord, value = line_profile(solver, Q, name; dim=dim, species=species)
    fig = Figure(; figure...)
    ax = Axis(fig[1, 1];
              xlabel=coordinate_label(solver.metric, dim),
              ylabel=_field_label(name, species),
              axis...)
    plt = lines!(ax, coord, value; kwargs...)
    return fig, ax, plt
end

# --- Field slices as heatmaps ----------------------------------------------

# Collective: the blocks a heatmap of the plane draws, each with its node
# coordinates `x1`, `x2`, the axes `h1`, `h2` it is drawn on, and its `values`,
# on rank 0, and `nothing` elsewhere. A block on the grid is drawn on its cell
# edges, so a wall node shows the half cell inside the domain and a fold's
# first cell reaches the fold; `curvilinear` marks a plane resampled onto
# Cartesian axes, whose raster is drawn around its sample points. A refined
# run gives one block per level at the level's own spacing, since the
# root-node composite `field_slice` would discard the fine nodes; refinement
# is Cartesian and unstretched, so those blocks need no resampling. A patched
# run without refinement keeps the composite, which loses nothing when every
# patch shares the root's spacing.
function _heatmap_blocks(solver::Solver, Q, name::Symbol, normal::Int, index::Int,
                         species::Int)
    if Q isa Vector{<:ConservedState} && nlevels(solver) > 1
        stacked = name in (:Y, :X, :D_art)
        stacked || name in CompactLES.SCALAR_FIELD_NAMES ||
            throw(ArgumentError("fieldheatmap: unknown field $name"))
        snaps = field_snapshot(solver, Q; fields=(name,), normal=normal, index=index)
        snaps === nothing && return nothing
        blocks = [_level_block(solver, filter(s -> s.level == level, snaps), name,
                               stacked, species, normal)
                  for level in sort!(unique(s.level for s in snaps))]
        return (; blocks, curvilinear=false, levels=true)
    end
    slice = field_slice(solver, Q, name; normal=normal, index=index, species=species)
    slice === nothing && return nothing
    x1, x2, values = slice
    a, b = _plane_dims(normal)
    if _curvilinear(solver)
        X, Y, grid = cartesian_slice(solver, (a, b), x1, x2, values)
        return (; blocks=[(; x1=X, x2=Y, h1=X, h2=Y, values=grid)], curvilinear=true,
                levels=false)
    end
    return (; blocks=[_edged_block(solver, a, b, x1, x2, values)], curvilinear=false,
            levels=false)
end

# A block drawn on the edges of its nodes' cells. A plane with a single node
# along one axis, as a collapsed dimension gives, has no cell edges there and
# is drawn around its nodes on both axes, since Makie takes both axes as
# centers or both as edges.
function _edged_block(solver::Solver, a::Int, b::Int, x1, x2, values)
    length(x1) > 1 && length(x2) > 1 ||
        return (; x1, x2, h1=x1, h2=x2, values)
    return (; x1, x2, h1=_cell_edges(solver, a, x1), h2=_cell_edges(solver, b, x2),
            values)
end

# The plane snapshots of one level's patches written into one array over the
# box enclosing them, on the level's uniform lattice. Nodes outside every
# patch, and nodes whose cell a finer level covers, are NaN, which a heatmap
# leaves transparent. Patches sharing an interface plane write equal values
# there.
function _level_block(solver::Solver, snaps, name::Symbol, stacked::Bool,
                      species::Int, normal::Int)
    a, b = _plane_dims(normal)
    lo = (minimum(s.offset[a] for s in snaps), minimum(s.offset[b] for s in snaps))
    hi = (maximum(s.offset[a] + size(s)[a] for s in snaps),
          maximum(s.offset[b] + size(s)[b] for s in snaps))
    grid = fill(NaN, hi[1] - lo[1], hi[2] - lo[2])
    for s in snaps
        f = s[name]
        values = Float64.(stacked ? f[:, :, :, species] : f)
        values[s.covered] .= NaN
        n = size(s)
        view(grid, s.offset[a] - lo[1] .+ (1:n[a]), s.offset[b] - lo[2] .+ (1:n[b])) .=
            reshape(values, n[a], n[b])
    end
    h = _level_spacing(solver, first(snaps).level)
    x1 = [_level_xcoord(solver, h, a, m) for m in (lo[1] + 1):hi[1]]
    x2 = [_level_xcoord(solver, h, b, m) for m in (lo[2] + 1):hi[2]]
    return _edged_block(solver, a, b, x1, x2, grid)
end

# One heatmap for a slice; for a refined run, one per level, root first,
# sharing a color range the caller may override.
function _draw_heatmaps!(ax, data; kwargs...)
    blocks = data.blocks
    data.levels || return heatmap!(ax, only(blocks).h1, only(blocks).h2,
                                   only(blocks).values; kwargs...)
    colorrange = get(kwargs, :colorrange) do
        finite = (v for blk in blocks for v in blk.values if isfinite(v))
        isempty(finite) ? (0.0, 1.0) : extrema(finite)
    end
    rest = Base.structdiff(values(kwargs), NamedTuple{(:colorrange,)})
    return [heatmap!(ax, blk.h1, blk.h2, blk.values; colorrange, rest...)
            for blk in blocks]
end

function CompactLES.fieldheatmap!(ax, solver::Solver, Q, name::Symbol;
                                  normal::Int=3, index::Int=1, species::Int=1,
                                  kwargs...)
    data = _heatmap_blocks(solver, Q, name, normal, index, species)
    data === nothing && return nothing         # off-rank-0: nothing to draw
    return _draw_heatmaps!(ax, data; kwargs...)
end

function CompactLES.fieldheatmap(solver::Solver, Q, name::Symbol;
                                 normal::Int=3, index::Int=1, species::Int=1,
                                 figure=(;), axis=(;), colorbar=true, kwargs...)
    data = _heatmap_blocks(solver, Q, name, normal, index, species)
    data === nothing && return nothing
    fig = Figure(; figure...)
    if data.curvilinear
        # A curvilinear plane is a physical disk or meridian; equal aspect keeps
        # it undistorted, and the axes are Cartesian, not coordinate.
        ax = Axis(fig[1, 1]; aspect=DataAspect(),
                  xlabel="x", ylabel="y", axis...)
    else
        a, b = _plane_dims(normal)
        ax = Axis(fig[1, 1];
                  xlabel=coordinate_label(solver.metric, a),
                  ylabel=coordinate_label(solver.metric, b), axis...)
    end
    plt = _draw_heatmaps!(ax, data; kwargs...)
    colorbar && Colorbar(fig[1, 2], plt isa Vector ? first(plt) : plt;
                         label=_field_label(name, species))
    return fig, ax, plt
end

# --- Mesh lines ---------------------------------------------------------------

# The Wong palette is the categorical order Makie itself cycles, and it keeps
# neighboring levels apart under the common color-vision deficiencies. The
# root is gray, so the refined levels stand out against it.
function _level_color(color, level::Int)
    color === nothing &&
        return level == 0 ? :gray50 : Makie.wong_colors()[mod1(level, 7)]
    color isa AbstractVector && return color[level + 1]
    return color
end

_selects(choice::Bool, level::Int) = choice
_selects(choice, level::Int) = level in choice

# Append the polyline through the coordinate pairs `(c1[i], c2[i])`, each
# segment divided into `sub` pieces so a line of constant radius follows its
# arc once mapped, and end it with the NaN break `lines!` separates on.
function _coordinate_line!(points, place, c1, c2, sub::Int)
    for i in 1:(length(c1) - 1), s in 0:(sub - 1)
        t = s / sub
        push!(points, place(c1[i] + t * (c1[i + 1] - c1[i]),
                            c2[i] + t * (c2[i + 1] - c2[i])))
    end
    push!(points, place(c1[end], c2[end]))
    push!(points, Point2{Float64}(NaN, NaN))
    return points
end

# Append the polylines along one coordinate at the fixed other coordinate `c`:
# one per run of consecutive segments that `present` marks, segment k spanning
# `p[k]` to `p[k + 1]`, less the parts inside a hole. A hole is the rectangle
# `((lo1, hi1), (lo2, hi2))` a finer level's cells fill, where its heatmap
# hides this level's; a line on a hole's boundary is kept. `along_first` says
# which of the two coordinates varies.
function _runs!(points, place, p, c, present, along_first::Bool, sub::Int, holes)
    k, K = 1, length(present)
    while k <= K
        if !present[k]
            k += 1
            continue
        end
        stop = k
        while stop < K && present[stop + 1]
            stop += 1
        end
        pieces = [(p[k], p[stop + 1])]
        for (r1, r2) in holes
            (span, fixed) = along_first ? (r1, r2) : (r2, r1)
            fixed[1] < c < fixed[2] || continue
            pieces = _subtract(pieces, span...)
        end
        for (s, e) in pieces
            run = vcat(s, [p[q] for q in (k + 1):stop if s < p[q] < e], e)
            line = fill(c, length(run))
            along_first ? _coordinate_line!(points, place, run, line, sub) :
                          _coordinate_line!(points, place, line, run, sub)
        end
        k = stop + 1
    end
    return points
end

# The rectangles the cells of the patches on `level` fill, one per patch; a
# refined level is Cartesian, so a rectangle in coordinates is one in the plot.
_holes(solver::Solver, blocks, level::Int, a::Int, b::Int) =
    [(extrema(_cell_edges(solver, a, blk.x1)), extrema(_cell_edges(solver, b, blk.x2)))
     for blk in blocks if blk.level == level]

# The intervals less the open interval (lo, hi).
function _subtract(intervals, lo, hi)
    out = eltype(intervals)[]
    for (s, e) in intervals
        if hi <= s || e <= lo
            push!(out, (s, e))
        else
            s < lo && push!(out, (s, lo))
            hi < e && push!(out, (hi, e))
        end
    end
    return out
end

# The mesh one level shows in a composite view, from its `shown` mask: with
# `cells`, every edge of a shown node's cell, and otherwise the node lines
# between shown neighbors. Where a finer level covers the level, it draws
# nothing, as its heatmap leaves those nodes transparent, and the lines stop
# at the `holes` the next level's cells fill, which the next level's heatmap
# paints over the half-covered cells along its patches. The radial lines of
# a grid through the axis run in to the origin, where they meet the lines of
# the opposite angle.
function _level_lines!(points, place, lev, sub::Int, cells::Bool, holes)
    shown = lev.shown
    if cells
        e1, e2 = lev.e1, lev.e2
        n1, n2 = length(e1) - 1, length(e2) - 1
        cell(i, j) = 1 <= i <= n1 && 1 <= j <= n2 && shown[i, j]
        # Around a closed angle the cell below the first edge is the last one,
        # and the edge at the wrapped end is the first edge again.
        below(j) = lev.closed ? mod1(j - 1, n2) : j - 1
        for j in 1:(lev.closed ? n2 : n2 + 1)
            present = [cell(i, below(j)) || cell(i, j) for i in 1:n1]
            _runs!(points, place, e1, e2[j], present, true, sub, holes)
        end
        for i in 1:(n1 + 1)
            lev.axis && i == 1 && continue              # the axis is a point
            present = [cell(i - 1, j) || cell(i, j) for j in 1:n2]
            _runs!(points, place, e2, e1[i], present, false, sub, holes)
        end
    else
        x1, x2 = lev.x1, lev.x2
        n1, n2 = size(shown)
        radial = lev.axis ? vcat(zero(x1[1]), x1) : x1
        for j in 1:(lev.closed ? n2 - 1 : n2)
            present = [shown[i, j] && shown[i + 1, j] for i in 1:(n1 - 1)]
            lev.axis && pushfirst!(present, shown[1, j])
            _runs!(points, place, radial, x2[j], present, true, sub, holes)
        end
        for i in 1:n1
            present = [shown[i, j] && shown[i, j + 1] for j in 1:(n2 - 1)]
            _runs!(points, place, x2, x1[i], present, false, sub, holes)
        end
    end
    return points
end

# The boundary through a patch's outermost nodes. A patch closed around the
# angle has no edge at the seam, only its inner and outer circles, and a patch
# through the axis only its outer circle.
function _outline!(points, place, blk, sub::Int)
    x1, x2 = blk.x1, blk.x2
    if blk.closed
        for c in (blk.axis ? (x1[end],) : (x1[1], x1[end]))
            _coordinate_line!(points, place, fill(c, length(x2)), x2, sub)
        end
        return points
    end
    c1 = vcat(x1, fill(x1[end], length(x2) - 1), reverse(x1)[2:end],
              fill(x1[1], length(x2) - 1))
    c2 = vcat(fill(x2[1], length(x1) - 1), x2, fill(x2[end], length(x1) - 1),
              reverse(x2))
    return _coordinate_line!(points, place, c1, c2, sub)
end

function CompactLES.meshplot!(ax, solver::Solver; normal::Int=3, index::Int=1,
                              cells::Bool=true, grid=true, outlines=true,
                              color=nothing, linewidth=0.5, outline_linewidth=1.5,
                              kwargs...)
    MPI.Comm_rank(solver.comm) == 0 || return nothing
    blocks = _mesh_blocks(solver, normal, index)
    levels = _mesh_levels(solver, normal, index)
    a, b = _plane_dims(normal)
    metric = solver.metric
    # The mapping `fieldheatmap` applies to the same plane: a resolved angle
    # goes to Cartesian axes, and its lines are subdivided to follow the arcs.
    curvilinear = _curvilinear(solver)
    place(c1, c2) = Point2{Float64}((curvilinear ?
                                     _slice_cartesian(metric, a, b, c1, c2) : (c1, c2))...)
    sub = curvilinear ? 8 : 1
    plots = Any[]
    for lev in levels
        _selects(grid, lev.level) || continue
        points = _level_lines!(Point2{Float64}[], place, lev, sub, cells,
                               _holes(solver, blocks, lev.level + 1, a, b))
        push!(plots, lines!(ax, points; color=_level_color(color, lev.level),
                            linewidth=linewidth, kwargs...))
    end
    for level in unique(blk.level for blk in blocks)
        _selects(outlines, level) || continue
        points = Point2{Float64}[]
        for blk in blocks
            blk.level == level && _outline!(points, place, blk, sub)
        end
        push!(plots, lines!(ax, points; color=_level_color(color, level),
                            linewidth=outline_linewidth, kwargs...))
    end
    return [p for p in plots]
end

function CompactLES.meshplot(solver::Solver; normal::Int=3, index::Int=1,
                             figure=(;), axis=(;), kwargs...)
    MPI.Comm_rank(solver.comm) == 0 || return nothing
    fig = Figure(; figure...)
    a, b = _plane_dims(normal)
    labels = _curvilinear(solver) ? ("x", "y") :
             (coordinate_label(solver.metric, a), coordinate_label(solver.metric, b))
    ax = Axis(fig[1, 1]; aspect=DataAspect(), xlabel=labels[1], ylabel=labels[2],
              axis...)
    plots = CompactLES.meshplot!(ax, solver; normal=normal, index=index, kwargs...)
    return fig, ax, plots
end

# A readable axis/colorbar label for a report variable.
function _field_label(name::Symbol, species::Int)
    name === :rho && return "ρ"
    name === :p && return "p"
    name === :T_ion && return "T"
    name === :Y && return "Y$(species)"
    name === :X && return "X$(species)"
    name === :D_art && return "D_art$(species)"
    return String(name)
end

end # module
