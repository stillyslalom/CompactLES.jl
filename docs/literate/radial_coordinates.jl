# # Take one step in a radial coordinate
#
# Cylindrical coordinates are useful even when the solution has no angular
# dependence. Collapsing the angular and axial dimensions produces an
# axisymmetric radial calculation at one-dimensional cost. This tutorial sets
# up a smooth inward-propagating disturbance and advances exactly one step.

using MPI
MPI.Initialized() || MPI.Init(threadlevel=:funneled)

using CompactLES
using CairoMakie
CairoMakie.activate!(type = "png")

# ## Geometry and axis condition
#
# `CylindricalMetric` interprets coordinates as ``(r, theta, z)``. `AxisBC` is
# not a wall: it represents regular continuation through ``r=0`` using parity
# and a folded compact operator. Radial nodes are half-offset near the axis, so
# no stored point lies at the coordinate singularity.

gamma = 1.4
nr = 192

problem = Problem(
    name = "radial acoustic ring",
    eos = single_species(gamma = gamma),
    metric = CylindricalMetric(),
    domain = ((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)),
    bcs = ((AxisBC(), SlipWallBC()),
           (PeriodicBC(), PeriodicBC()),
           (PeriodicBC(), PeriodicBC())),
    ic = (r, theta, z) -> begin
        ring = exp(-((r - 0.65) / 0.07)^2)
        p = 1 + 0.02 * ring
        Prim(p = p, rho = p^(1 / gamma), u = (-0.01ring, 0.0, 0.0))
    end,
)

numerics = Numerics(
    n_global = (nr, 1, 1),
    art = ArtParams(enabled = false),
    cfl = 0.5,
    filter_interval = 0,
)

solver, Q = setup(problem, numerics)

function density_line(solver, Q)
    n = solver.decomp.n_local[1]
    [mixture_density(solver, Q, gidx(solver, i, 1, 1)) for i in 1:n]
end

r = [xcoord(solver, 1, i) for i in 1:nr]
density_initial = density_line(solver, Q)

# ## Inspect the timestep
#
# [`dt_report`](@ref) identifies the point and mechanism that limit the global
# explicit timestep. With both angular dimensions collapsed and no swirl, this
# case is limited by radial acoustic propagation rather than a vanishing
# azimuthal cell width.

limit = dt_report(solver, Q)
(; limit.dt, limit.coords, limit.dim, limit.kind)

# Advance to the first CFL time. `nmax=1` ensures the tutorial cannot turn into
# a long integration if a future change alters timestep selection.

run!(solver, Q; tfinal = limit.dt, nmax = 1)
density_after_one_step = density_line(solver, Q)

fig = Figure(size = (760, 600))
ax1 = Axis(fig[1, 1], ylabel = "density - 1",
           title = "Initial radial disturbance")
lines!(ax1, r, density_initial .- 1, color = :black)

ax2 = Axis(fig[2, 1], xlabel = "radius", ylabel = "density change",
           title = "Increment after one RK step")
lines!(ax2, r, density_after_one_step .- density_initial, color = :darkorange)
linkxaxes!(ax1, ax2)
fig

# A resolved angular dimension behaves differently. Its physical spacing is
# ``r Delta theta`` and becomes small near the axis, which can impose a much
# tighter explicit timestep. The collapsed calculation avoids that cost because
# there is no angular derivative, while the radial geometric source terms
# remain present. See [Curvilinear coordinates](@ref) for the resolved-angle
# restrictions and the spherical origin and pole treatments.
