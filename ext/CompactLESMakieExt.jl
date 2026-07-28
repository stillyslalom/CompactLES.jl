module CompactLESMakieExt

# Makie plotting for the extraction API declared in src/viz.jl. As with the HDF5
# extension, the core package declares `profileplot`, `fieldheatmap`, and their
# mutating forms as stubs that error until a backend is loaded; this module adds
# the methods once the caller writes `using CairoMakie` (or `using GLMakie`).
#
# The extension holds no numerics. Every value it draws comes from the core
# `line_profile` / `field_slice` / `cartesian_slice`, so a plot means exactly the
# extracted data and the two cannot disagree. Those calls are collective; the
# figure is assembled on rank 0, where the gathered slice lives.

using CompactLES
using CompactLES: Solver, line_profile, field_slice, cartesian_slice,
                  coordinate_label, _plane_dims, _curvilinear
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
    # any rank — rank 0 is the convention.
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

function CompactLES.fieldheatmap!(ax, solver::Solver, Q, name::Symbol;
                                  normal::Int=3, index::Int=1, species::Int=1,
                                  kwargs...)
    slice = field_slice(solver, Q, name; normal=normal, index=index, species=species)
    slice === nothing && return nothing        # off-rank-0: nothing to draw
    x1, x2, values = slice
    dims = _plane_dims(normal)
    if _curvilinear(solver)
        X, Y, grid = cartesian_slice(solver, dims, x1, x2, values)
        return heatmap!(ax, X, Y, grid; kwargs...)
    end
    return heatmap!(ax, x1, x2, values; kwargs...)
end

function CompactLES.fieldheatmap(solver::Solver, Q, name::Symbol;
                                 normal::Int=3, index::Int=1, species::Int=1,
                                 figure=(;), axis=(;), colorbar=true, kwargs...)
    slice = field_slice(solver, Q, name; normal=normal, index=index, species=species)
    slice === nothing && return nothing
    x1, x2, values = slice
    dims = _plane_dims(normal)
    curvi = _curvilinear(solver)
    fig = Figure(; figure...)
    if curvi
        # A curvilinear plane is a physical disk or meridian; equal aspect keeps
        # it undistorted, and the axes are Cartesian rather than coordinate.
        ax = Axis(fig[1, 1]; aspect=DataAspect(),
                  xlabel="x", ylabel="y", axis...)
        X, Y, grid = cartesian_slice(solver, dims, x1, x2, values)
        plt = heatmap!(ax, X, Y, grid; kwargs...)
    else
        a, b = dims
        ax = Axis(fig[1, 1];
                  xlabel=coordinate_label(solver.metric, a),
                  ylabel=coordinate_label(solver.metric, b), axis...)
        plt = heatmap!(ax, x1, x2, values; kwargs...)
    end
    colorbar && Colorbar(fig[1, 2], plt; label=_field_label(name, species))
    return fig, ax, plt
end

# A readable axis/colorbar label for a report variable.
function _field_label(name::Symbol, species::Int)
    name === :rho && return "ρ"
    name === :p && return "p"
    name === :T_ion && return "T"
    name === :Y && return "Y$(species)"
    name === :D_art && return "D_art$(species)"
    return String(name)
end

end # module
