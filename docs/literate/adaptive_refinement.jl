# # Follow a moving feature with refinement
#
# Adaptive mesh refinement (AMR) concentrates grid points where a calculation
# needs them. This example advects a smooth density pulse at uniform velocity
# and pressure. Its exact solution is a translation, so the physical feature
# and the refinement policy can be checked independently. We compare a coarse
# grid, moving refinement on that same grid, and a uniformly fine grid.

using CompactLES
MPI.Initialized() || MPI.Init(threadlevel=:funneled)
using CairoMakie
using Printf
CairoMakie.activate!(type = "png")

speed = 0.5
center(t) = 0.30 + speed * t
density(x, t) = 1 + 0.2 * exp(-((x - center(t)) / 0.025)^2)

problem = Problem(
    name = "advected density pulse",
    domain = ((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)),
    bcs = (PeriodicBC(), PeriodicBC(), PeriodicBC()),
    ic = (x, y, z) -> Prim(rho = density(x, 0.0), p = 1.0,
                          u = (speed, 0.0, 0.0)),
)

# ## Select refinement in physical coordinates
#
# [`AMR`](@ref) belongs to `Numerics`: it describes how the problem is resolved.
# The predicate below selects nodes within 0.09 of the moving center. It does
# not depend on the root-grid node count. The selected nodes are buffered and
# enclosed in a rectangular patch; the patch need not have exactly the shape
# of the predicate's region.
#
# The default density sensor is disabled here with `tag_threshold = Inf` to
# isolate the prescribed motion. `initial = :sensor` would instead select the
# initial cover using the configured sensors. Predicate and sensor criteria
# are combined by union when both are enabled.

amr = AMR(
    initial = (x, y, z, t) -> abs(x - center(t)) < 0.09,
    tag_threshold = Inf,
    tag_buffer = 2,
    regrid_interval = 4,
    subcycle = true,
)

# A fixed region can be given as a shape instead of a predicate. The shapes
# are in the `CompactLES.Regions` submodule, which `using CompactLES` does not
# load. Makie also exports `Box` and `Sphere`, so after `using CairoMakie`
# write those two as `Regions.Box` and `Regions.Sphere`.

using CompactLES.Regions
fixed = AMR(initial = Slab(1, lo = 0.2, hi = 0.4), subcycle = true)

root_nodes = 64
common = (
    art = ArtificialProperties(enabled = false),
    filter_interval = 0,
    cfl = 0.35,
)
numerics = Numerics(; n_global = (root_nodes, 1, 1), amr, common...)
solver, states = setup(problem, numerics)
initial_cover = level_regions(solver, 1)
initial_mass = volume_integral(solver, states, :rho)

# There are two levels: the root covers the whole domain, and the refined
# level has three times its spatial resolution. `subcycle = true` advances
# that level in three substeps per root step. Removing it uses one common
# timestep constrained by the whole hierarchy. With `regrid_interval = 0`,
# the initial cover stays fixed instead of following the pulse.
#
# At setup the IC is evaluated on each level's own nodes. During regridding,
# new fine nodes receive interpolated evolved data; the IC is not reapplied.

# Record the actual patch after every step without prescribing extra output
# times. The width changes by whole coarse cells as the feature moves.

cover_times = [solver.t]
cover_history = [initial_cover]
record_cover = (s, Q) -> begin
    push!(cover_times, s.t)
    push!(cover_history, level_regions(s, 1))
    nothing
end

tfinal = 0.60
run!(solver, states; tfinal, nmax = 4000, callback = record_cover)
@assert solver.t == tfinal
final_cover = level_regions(solver, 1)

# ## Compare resolution and error
#
# The Gaussian width is 1.6 coarse spacings but 4.8 fine spacings. This makes
# the coarse-grid dispersion visible while the fine grid resolves the pulse.
# The center moves from 0.30 to 0.60; the refinement follows it. Its path is
# prescribed here, so this demonstrates regridding, not automatic feature
# detection by a sensor.
#
# Use identical physics, operators, CFL, and final time for both uniform
# runs. A periodic grid has `h = L/N`, so `3root_nodes` matches the AMR fine
# spacing. Each run chooses its own stable timestep; the uniform fine run
# takes three times as many steps as the coarse run in this case.

coarse, Qcoarse = setup(problem, Numerics(; n_global = (root_nodes, 1, 1), common...))
fine, Qfine = setup(problem, Numerics(; n_global = (3root_nodes, 1, 1), common...))
run!(coarse, Qcoarse; tfinal, nmax = 4000)
run!(fine, Qfine; tfinal, nmax = 4000)
@assert coarse.t == fine.t == tfinal

# ## Read the composite solution
#
# Pass the entire state vector to diagnostics. Selecting `states[1]` would
# omit the fine solution. [`line_profile`](@ref) and [`volume_integral`](@ref)
# take the state vector and a field name, refresh the primitive fields on
# every patch, and combine the levels. The profile is sampled at the root-grid
# stations, so this plot compares profiles on a common sampling grid. Sample
# the uniform fine solution at every third node for the same comparison. The
# separate mesh plot shows the fine nodes that this sampling leaves out. All
# ranks must enter the profile and integral reductions.

x, rho = line_profile(solver, states, :rho)
final_mass = volume_integral(solver, states, :rho)
relative_mass_change = (final_mass - initial_mass) / initial_mass
exact = density.(x, tfinal)
_, rho_coarse = line_profile(coarse, Qcoarse, :rho)
x_fine, rho_fine = line_profile(fine, Qfine, :rho)
errors = (abs.(rho_coarse .- exact), abs.(rho .- exact),
          abs.(rho_fine[1:3:end] .- exact))
sample_rms = map(e -> sqrt(sum(abs2, e) / length(e)), errors)
error_reduction = sample_rms[1] / sample_rms[2]

# Count stored interior nodes, including the coarse nodes under the fine
# patch: they still require storage. An extent of m parent nodes contains
# 3(m-1)+1 fine nodes. Halos and scratch arrays are excluded. This is a
# resolution/storage comparison, not a wall-time benchmark; subcycling,
# interpolation, regridding, and communication also contribute to cost.

stored_nodes = [root_nodes + sum(3(r.extent[1] - 1) + 1 for r in cover)
                for cover in cover_history]
node_range = extrema(stored_nodes)

# ## Watch the refined region move
#
# The blue band shows the fine patch tracking the exact pulse center.
# Regridding precedes each step, so the cover recorded at its end describes
# the preceding time interval in this plot.

tracking_fig = Figure(size = (760, 420), fontsize = 18) #hide
blue, orange, purple = :dodgerblue3, :darkorange2, :purple3 #hide
track = Axis(tracking_fig[1, 1], xlabel = "x", ylabel = "time", #hide
             title = "Refinement follows the pulse", yticks = 0:0.2:tfinal) #hide
for i in 1:length(cover_times)-1 #hide
    for r in cover_history[i+1] #hide
        lo, hi = r.offset[1] / root_nodes, (r.offset[1] + r.extent[1] - 1) / root_nodes #hide
        poly!(track, Point2f[(lo, cover_times[i]), (hi, cover_times[i]), #hide
                            (hi, cover_times[i+1]), (lo, cover_times[i+1])]; #hide
              color = (blue, 0.3), strokewidth = 0) #hide
    end #hide
end #hide
lines!(track, center.(cover_times), cover_times, color = :black, #hide
       linestyle = :dash, label = "exact pulse center") #hide
text!(track, 0.06, 0.50, text = "blue band: fine patch\nelsewhere: coarse grid", fontsize = 16) #hide
xlims!(track, 0, 1); ylims!(track, 0, tfinal) #hide
axislegend(track, position = :rb, labelsize = 16) #hide
tracking_fig

# ## See the local spacing change
#
# The composite mesh strips expose the threefold spacing change at the patch
# edges. They omit covered coarse nodes, which remain stored underneath and
# are included in the node count. Black triangles mark the pulse center.

mesh_fig = Figure(size = (760, 360), fontsize = 18) #hide
mesh = Axis(mesh_fig[1, 1], xlabel = "x", title = "Composite mesh, before and after", #hide
            yticks = ([0, 1], ["final", "initial"]), ygridvisible = false) #hide
for (row, cover) in ((1, initial_cover), (0, final_cover)) #hide
    r = only(cover) #hide
    lo, hi = r.offset[1] / root_nodes, (r.offset[1] + r.extent[1] - 1) / root_nodes #hide
    active_root = filter(xi -> xi < lo || xi > hi, x) #hide
    refined_x = range(lo, hi; length = 3(r.extent[1] - 1) + 1) #hide
    poly!(mesh, Point2f[(lo, row-0.2), (hi, row-0.2), (hi, row+0.2), (lo, row+0.2)]; #hide
          color = (blue, 0.12), strokewidth = 0) #hide
    root_ticks = [Point2f(xi, row + dy) for xi in active_root for dy in (-0.06, 0.06)] #hide
    fine_ticks = [Point2f(xi, row + dy) for xi in refined_x for dy in (-0.09, 0.09)] #hide
    linesegments!(mesh, root_ticks, linewidth = 1.2, color = :gray45) #hide
    linesegments!(mesh, fine_ticks, linewidth = 0.8, color = blue) #hide
    scatter!(mesh, [center(row == 1 ? 0.0 : tfinal)], [row + 0.33], marker = :dtriangle, color = :black) #hide
end #hide
xlims!(mesh, 0, 1); ylims!(mesh, -0.5, 1.7) #hide
text!(mesh, 0.03, 1.5, text = "gray: coarse nodes   blue: 3× finer spacing", fontsize = 16) #hide
node_text = @sprintf("Stored interior nodes: AMR %d–%d; uniform fine %d\nIncludes covered coarse nodes; excludes halos and scratch arrays", #hide
                     node_range[1], node_range[2], 3root_nodes) #hide
Label(mesh_fig[2, 1], node_text, fontsize = 16, tellwidth = false, padding = (0, 0, 8, 8)) #hide
mesh_fig

# ## Compare the final pulse
#
# The coarse run disperses the pulse and leaves an oscillatory wake. The
# AMR samples follow the exact translation and the uniform fine solution
# much more closely. The next plot separates their smaller errors.

solution_fig = Figure(size = (760, 420), fontsize = 18) #hide
solution = Axis(solution_fig[1, 1], xlabel = "x", ylabel = "density", #hide
                title = "Final pulse at t = $tfinal") #hide
xx = range(center(tfinal) - 0.13, center(tfinal) + 0.13; length = 600) #hide
lines!(solution, xx, density.(xx, tfinal), color = :black, label = "exact") #hide
lines!(solution, x, rho_coarse, color = orange, label = "coarse (64)") #hide
lines!(solution, x_fine, rho_fine, color = purple, linestyle = :dash, label = "uniform fine (192)") #hide
scatter!(solution, x, rho, color = blue, markersize = 6, label = "AMR (64 + patch)") #hide
xlims!(solution, first(xx), last(xx)) #hide
axislegend(solution, position = :rt, labelsize = 15) #hide
solution_fig

# ## Quantify the accuracy gained
#
# Errors and the RMS summary use only the common 64 stations, not a norm over
# every fine node. Errors below 1e-8 are drawn at that display floor; the RMS
# calculation uses the unclipped errors.

error_fig = Figure(size = (760, 450), fontsize = 18) #hide
errplot = Axis(error_fig[1, 1], xlabel = "x", ylabel = "absolute density error", #hide
               yscale = log10, title = "Error at the same 64 stations") #hide
for (e, color, label) in zip(errors, (orange, blue, purple), ("coarse", "AMR", "uniform fine")) #hide
    lines!(errplot, x, max.(e, 1e-8); color, label, linewidth = 2) #hide
end #hide
xlims!(errplot, 0, 1); ylims!(errplot, 0.5e-8, 0.03) #hide
text!(errplot, 0.03, 1.4e-8, text = "display floor", fontsize = 15, color = :gray40) #hide
axislegend(errplot, position = :rt, labelsize = 16) #hide
error_text = @sprintf("AMR: %.0f× smaller sampled RMS error than coarse alone", error_reduction) #hide
Label(error_fig[2, 1], error_text, fontsize = 18, tellwidth = false, padding = (0, 0, 8, 10)) #hide
error_fig

# Here AMR achieves an error comparable to uniform refinement with fewer
# stored interior nodes. This is a localized smooth-feature example, not a
# general efficiency guarantee. A broader refined region, multiple features,
# or strong coarse-fine interface errors can change that tradeoff.
#
# The relative mass change is a separate diagnostic of interface and transfer
# errors, complementary to the sampled density error:

relative_mass_change

# AMR adds interpolation and interface errors as well as resolution. The
# implementation has no conservative refluxing, which would correct the
# coarse and fine fluxes to share one conservation budget. Check composite
# mass and energy in addition to profile error for a production calculation.
# [Adaptive mesh refinement](@ref) describes sensor initialization, nested
# levels, tiling, checkpoint/restart, and the current support limits.
