# # Setting up a radial acoustic pulse
#
# Cylindrical coordinates are useful even when a solution has no angular or
# axial dependence. With one point in each of those dimensions, CompactLES
# omits their derivatives but retains the radial geometric terms. The result is
# an axisymmetric radial calculation at one-dimensional cost. This tutorial
# sets up a smooth inward-propagating disturbance and advances it by one
# timestep.

using MPI
MPI.Initialized() || MPI.Init(threadlevel=:funneled)

using CompactLES
using CairoMakie
CairoMakie.activate!(type = "png")

# ## Geometry and axis condition
#
# [`CylindricalMetric`](@ref) interprets coordinates and physical velocity
# components as ``(r, theta, z)`` and ``(u_r, u_theta, u_z)``. Concentric
# cylindrical surfaces have different areas, so even an angle-independent
# radial flux has a divergence proportional to
# ``r^{-1}\partial(r F_r)/\partial r``. This geometry remains active when the
# angular dimension is collapsed.
#
# The location ``r=0`` is a coordinate singularity, but it is not a material
# wall: a smooth physical field continues through the axis. Imagine extending
# the radial line to negative ``r``. Density and pressure take the same value
# at ``-r`` and ``r`` (even parity), while the radial component of a smooth
# velocity reverses direction (odd parity). [`AxisBC`](@ref) supplies that
# symmetry continuation to the compact derivative.
#
# The radial nodes are at ``r_i=(i-\tfrac12)\Delta r``. The first node is half
# a cell from the axis, so the solver never evaluates a stored state at
# ``r=0``. Internally, the parity continuation and the implicit compact stencil
# are combined into a folded operator; [Curvilinear coordinates](@ref) develops
# that construction after the physical picture established here.

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

# [`line_profile`](@ref) returns a `(coordinate, value)` pair. In a cylindrical
# geometry the first-axis coordinate is the radius, so this reads the density as
# a function of ``r`` directly.

r, density_initial = line_profile(solver, Q, :rho)

# ## Inspect the timestep
#
# [`dt_report`](@ref) identifies the point and mechanism that limit the global
# explicit timestep. With the angular and axial dimensions collapsed and no
# swirl, this
# case is limited by radial acoustic propagation rather than a vanishing
# azimuthal cell width.

limit = dt_report(solver, Q)
(; limit.dt, limit.coords, limit.dim, limit.kind)

# Advance by the CFL-limited interval reported above. One call to `run!` still
# performs all five Runge--Kutta stages; `nmax=1` limits the run to one complete
# timestep and keeps the tutorial cheap.

run!(solver, Q; tfinal = limit.dt, nmax = 1)
_, density_after_one_step = line_profile(solver, Q, :rho)

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
# ``r\Delta\theta`` and becomes small near the axis, which can impose a much
# tighter explicit timestep. The present calculation avoids that angular wave
# propagation because there is no angular derivative, while radial flux-area
# and curvature terms remain. The next tutorial resolves the angle and
# introduces the additional antipodal mapping and grid restrictions.
