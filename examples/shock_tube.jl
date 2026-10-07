# # Reshocked mixing layer
#
# A shock that crosses a perturbed interface between two gases leaves the
# perturbation growing, and [Richtmyer–Meshkov instability](@ref) follows a
# single mode of it from air into SF6 through its linear stage. In a shock tube
# closed at the far end, the shock transmitted into the heavy gas reflects from
# the end wall and crosses the interface a second time, now from the heavy gas
# into the light one. This reshock gives the layer a second and larger
# velocity jump at a time when its perturbations have grown to a finite
# amplitude. Vetter and Sturtevant (1995) studied an air/SF6 interface in a
# shock tube at Mach 1.5 with reshock from the end wall.
#
# This page calculates a layer between the same two gases, struck by a shock of
# the same strength, in two dimensions and with a band of modes on the
# interface. It follows the layer through the first shock, through reshock, and
# through the waves that pass between the layer and the end wall afterwards,
# and asks how the layer grows in each of these stages. The waves are compared
# with the exact solution in one dimension, and the width of the layer and its
# molecular mixing with a second calculation on a grid twice as coarse, to show
# what the calculation settles and what it does not.
#
# The script takes five settings: `ny`, the nodes across the tube, which sets
# the spacing ``\Delta x = L_y/n_y`` of the main run, with ``L_y`` the width of
# the tube, and of a second run at twice that spacing; `Mach`, the Mach number
# of the incident shock; `tfinal`, the time at which the runs end; `seed`,
# which draws the phases of the interface's modes; and `smoke`. With
# `smoke=true` the runs have 16 and 10 rows, which checks that the page still
# runs; the figures come from the full run.

const T_START = time() #src
using CompactLES   # re-exports MPI
using CompactLES.Regions
MPI.Initialized() || MPI.Init(threadlevel=:funneled)

using CairoMakie
using Printf
CairoMakie.activate!(type = "png")

include(joinpath(@__DIR__, "common.jl")) #src

opt = CompactLES.script_args(ARGS, (ny = 128, Mach = 1.5, tfinal = 3e-3, seed = 1,
                                    smoke = false))
grids = opt.smoke ? (16, 10) : (opt.ny, opt.ny ÷ 2)
outdir = figure_dir("shock_tube"; smoke = opt.smoke) #src
nothing #hide

# ## Gases and waves
#
# The air and the SF6 start at 295 K and atmospheric pressure. The incident
# shock heats the air to 390 K, and the shock reflected from the end wall heats
# the SF6 to 374 K. [`IdealMixture`](@ref) holds each species at its
# room-temperature heat capacities. With the temperature-dependent heat
# capacities of [`Nasa9Mixture`](@ref), the one-dimensional wave speeds below
# change by at most 2.4 m/s, and by at most 1.7%, for the slow shock reflected
# from the end wall; the velocity jumps and Atwood numbers change by less than
# 0.3%.
#
# For a plane, sharp interface the waves follow from three Riemann problems,
# each solved by [`riemann_interface`](@ref). When the incident shock arrives,
# the shocked air meets SF6 at rest: a shock is transmitted into the SF6,
# another is reflected into the air, and the interface moves off at the
# contact velocity, the velocity jump ``\Delta u`` of the first shock. When the
# transmitted shock reaches the end wall, the shocked SF6 meets its own mirror
# image, whose velocity is reversed, and a shock is reflected that brings the
# gas to rest. When that shock reaches the interface, the gas at rest behind it
# meets the shocked air: a shock is transmitted into the air, a rarefaction is
# reflected into the SF6, and the interface takes a new velocity. The
# difference between the two interface velocities is the velocity jump of
# reshock. The Atwood number ``A = (\rho_2 - \rho_1)/(\rho_2 + \rho_1)``, with
# ``\rho_2`` the density of the SF6 and ``\rho_1`` that of the air, is printed
# for the gases at rest and for the gases either side of the interface after
# each shock.

eos = IdealMixture(["Air", "SF6"])
p0, T0 = 101_325.0, 295.0
air = Prim(Y = mass_fractions(eos, "Air" => 1.0; basis = :mole), p = p0, T_ion = T0)
sf6 = Prim(Y = mass_fractions(eos, "SF6" => 1.0; basis = :mole), p = p0, T_ion = T0)
density(state) = thermodynamic_state(eos, state).rho
atwood(light, heavy) = (density(heavy) - density(light)) / (density(heavy) + density(light))

Lx, Ly = 0.5, 0.1
x_shock, x_interface = 0.20, 0.25
incident = shock_jump(eos, air, opt.Mach)
impact = riemann_interface(eos, incident.post, sf6)
t_impact = (x_interface - x_shock) / incident.shock_speed
t_wall = t_impact + (Lx - x_interface) / impact.right_speed
gas = impact.right
wall = riemann_interface(eos, gas, Prim(u = (-gas.u[1], 0.0, 0.0), p = gas.p,
                                        T_ion = gas.T_ion, Y = gas.Y))
x_wall_meet = x_interface + impact.u_star * (t_wall - t_impact)
t_reshock = t_wall + (Lx - x_wall_meet) / (impact.u_star - wall.left_speed)
x_reshock = x_wall_meet + impact.u_star * (t_reshock - t_wall)
reshock = riemann_interface(eos, impact.left, wall.left)

@printf("incident shock %.1f m/s; behind it %.1f m/s, %.1f kPa, %.1f K\n",
        incident.shock_speed, incident.velocity, incident.post.p / 1e3,
        incident.post.T_ion)
@printf("first shock at %.3f ms: transmitted shock %.1f m/s, reflected %s %.1f m/s, \
        interface %.1f m/s, A = %.3f before, %.3f after\n", 1e3t_impact,
        impact.right_speed, impact.left_wave, impact.left_speed, impact.u_star,
        atwood(air, sf6), atwood(impact.left, impact.right))
@printf("end wall at %.3f ms: reflected shock %.1f m/s\n", 1e3t_wall, wall.left_speed)
@printf("reshock at %.3f ms, %.1f mm from the wall: transmitted %s %.1f m/s, \
        reflected %s %.1f m/s, interface %.1f m/s, Δu = %.1f m/s, A = %.3f\n",
        1e3t_reshock, 1e3(Lx - x_reshock), reshock.left_wave, reshock.left_speed,
        reshock.right_wave, reshock.right_speed, reshock.u_star,
        impact.u_star - reshock.u_star, atwood(reshock.left, reshock.right))

# The interface and the reflected shock meet 32 mm from the end wall, and the
# interface then moves back toward the inflow. The velocity jump of reshock,
# 216 m/s, is larger than that of the first shock, 157.5 m/s. The Atwood number
# is 0.67 for the gases at rest, 0.73 for the gases either side of the
# interface after the first shock and 0.76 after reshock.
#
# ## Tube
#
# The tube is 0.5 m long and 0.1 m wide, and the third dimension is collapsed.
# The shock starts 0.20 m from the inflow end and travels toward increasing
# ``x``; the mean position of the interface is at 0.25 m and the end wall, a
# [`SlipWallBC`](@ref), at 0.5 m. Across the tube the faces are periodic, so
# the calculation has no side walls and no boundary layers on them.
#
# The shocked air enters at ``x = 0`` through an [`NSCBCInflowBC`](@ref), which
# relaxes the velocity, temperature and composition of the entering gas toward
# a target state and lets waves traveling upstream leave. The gas behind each
# wave that leaves through the face differs from the target before it, so the
# target, a function of position and time, switches as each one leaves: to the
# air behind the reflected shock at 1.30 ms, and to the air behind the shock
# transmitted at reshock at 2.64 ms. The times come from the one-dimensional
# solution. In the [Richtmyer–Meshkov instability](@ref) tutorial, the
# interface speeds up when the target does not switch.
#
# [`Multimode`](@ref) displaces the interface by a sum of the modes with 4 to
# 12 waves across the tube, wavelengths from 8.3 to 25 mm, of equal amplitude
# and with phases drawn from `seed`; the root-mean-square displacement is
# 0.5 mm. The `1` in [`Slab`](@ref CompactLES.Regions.Slab) is the first
# coordinate, along the tube. Across the displaced interface the two gases
# blend by volume along an error function, through a
# [`Layer`](@ref CompactLES.Regions.Layer) whose `width` makes the thickness of
# the blend, the difference in composition divided by its largest gradient,
# 5 mm. That is the thickness of the layer in
# the experiment of Collins and Jacobs that the tutorial repeats. The
# thickness is set in meters rather than in cells, so both grids start from
# the same layer. The incident shock is spread over the default three cells.

t_reflected = t_impact + x_interface / abs(impact.left_speed)
t_retransmitted = t_reshock + x_reshock / abs(reshock.left_speed)
inflow_state(x, y, z, t) = t < t_reflected ? incident.post :
                           t < t_retransmitted ? impact.left : reshock.left

modes, rms, delta = 4:12, 5e-4, 5e-3
eta = Multimode(lengths = (Ly,), modes = modes, rms = rms, seed = opt.seed,
                mean = x_interface)
problem = Problem(
    name = "air/SF6 Richtmyer–Meshkov with reshock",
    eos = eos,
    domain = ((0.0, Lx), (0.0, Ly), (0.0, 1.0)),
    bcs = ((NSCBCInflowBC(incident.post; target = inflow_state), SlipWallBC()),
           PeriodicBC(), PeriodicBC()),
    ic = Layers(air,
                Slab(1, hi = x_shock) => incident.post,
                Layer(Slab(1, lo = eta), sf6; width = delta / sqrt(pi));
                profile = :erf),
)
nothing #hide

# ## Run
#
# The main run has square cells at a spacing ``\Delta x = 0.78`` mm: 10.7 cells
# per shortest wavelength and 6.4 across the initial blend. The second run has
# ``\Delta x = 1.56`` mm. The numerics are the defaults. Every 5 µs each run
# records three averages over the planes of constant ``x``: the pressure, the
# mole fraction ``X`` of the SF6, and ``X(1 - X)``. At seven times it also
# keeps the mole fraction field.

frame_times = [0.0, 0.7, 1.45, 1.6, 2.0, 2.5, 3.0] .* 1e-3
frame_times = filter(<=(opt.tfinal), frame_times)

run_key(ny) = @sprintf("ny%d_Mach%g_t%g_seed%d", ny, opt.Mach, opt.tfinal, opt.seed) #src
function simulate(ny)
    cached("shock_tube", run_key(ny); smoke = opt.smoke) do #src
    nx = round(Int, Lx / Ly * ny)
    solver, Q = setup(problem, Numerics(n_global = (nx, ny, 1)))
    t = Float64[]
    p_mean, X_mean, XX_mean = Vector{Float64}[], Vector{Float64}[], Vector{Float64}[]
    frames = []
    x = Ref(Float64[])
    y = Ref(Float64[])
    sample = Callback(EveryTime(5e-6), function (solver, Q)
        snap = field_snapshot(solver, Q; fields = (:X, :p))
        X = snap[:X][:, :, 1, 2]                 # SF6 mole fraction
        x[], y[] = snap.coords[1], snap.coords[2]
        push!(t, solver.t)
        push!(p_mean, vec(sum(snap[:p][:, :, 1], dims = 2)) ./ ny)
        push!(X_mean, vec(sum(X, dims = 2)) ./ ny)
        push!(XX_mean, vec(sum(X .* (1 .- X), dims = 2)) ./ ny)
        if any(tf -> abs(solver.t - tf) < 2.5e-6, frame_times) &&
           all(f -> abs(f.t - solver.t) > 2.5e-6, frames)
            push!(frames, (t = solver.t, X = X))
        end
        nothing
    end)
    wall_time = @elapsed run!(solver, Q; tfinal = opt.tfinal, callback = sample)
    @printf("%d × %d nodes, h = %.2f mm: %d steps in %.0f s\n", nx, ny, 1e3Ly / ny,
            solver.step, wall_time)
    return (; nx, ny, h = Ly / ny, x = x[], y = y[], t, p_mean, X_mean, XX_mean, frames)
    end #src
end
runs = [simulate(ny) for ny in grids]
nothing #hide

# ## Waves
#
# The waves are found in the plane-averaged pressure. A shock is where the
# pressure crosses the middle of its one-dimensional jump, searching from the
# side the shock moves toward; the interface is where the plane-averaged mole
# fraction ``\langle X \rangle`` of SF6 first rises through one half, a
# position the figures call ``x_{50}``. Each velocity is the slope of a straight
# line fitted to the positions over a window of time, chosen so that no other
# wave crosses the feature in it.

function crossing(x, v, level; from = :left, above = true)
    test = above ? (>(level)) : (<(level))
    i = from === :left ? findfirst(test, v) : findlast(test, v)
    i === nothing && return NaN
    j = from === :left ? i - 1 : i + 1
    (j < 1 || j > length(v)) && return x[i]
    return x[j] + (level - v[j]) / (v[i] - v[j]) * (x[i] - x[j])
end

function slope(t, x, window)
    s = findall(k -> window[1] <= t[k] <= window[2] && isfinite(x[k]), eachindex(t))
    length(s) < 3 && return NaN
    a, b = t[s], x[s]
    am, bm = sum(a) / length(a), sum(b) / length(b)
    return sum((a .- am) .* (b .- bm)) / sum((a .- am) .^ 2)
end

p_star, p_wall, p_reshock = impact.p_star, wall.p_star, reshock.p_star
midway(a, b) = (a + b) / 2
function tracks(run)
    x = run.x
    return (
        incident = [crossing(x, p, midway(p0, incident.post.p); from = :right)
                    for p in run.p_mean],
        transmitted = [crossing(x, p, midway(p0, p_star); from = :right)
                       for p in run.p_mean],
        wall_reflected = [crossing(x, p, midway(p_star, p_wall); from = :right,
                                   above = false) for p in run.p_mean],
        retransmitted = [crossing(x, p, midway(p_star, p_reshock)) for p in run.p_mean],
        interface = [crossing(x, X, 0.5) for X in run.X_mean],
    )
end

windows = [
    ("incident shock", :incident, (10e-6, t_impact - 10e-6), incident.shock_speed),
    ("transmitted shock", :transmitted, (t_impact + 100e-6, t_wall - 100e-6),
     impact.right_speed),
    ("interface", :interface, (t_impact + 100e-6, t_reshock - 100e-6), impact.u_star),
    ("reflected shock", :wall_reflected, (t_wall + 50e-6, t_reshock - 100e-6),
     wall.left_speed),
    ("shock into the air", :retransmitted, (t_reshock + 100e-6, t_reshock + 400e-6),
     reshock.left_speed),
    ("interface after reshock", :interface, (t_reshock + 50e-6, t_reshock + 300e-6),
     reshock.u_star),
]
trs = [tracks(run) for run in runs]
@printf("%-26s %8s %8s %8s\n", "velocity (m/s), rows:", grids..., "1-D")
for (label, f, w, exact) in windows
    @printf("%-26s %8.1f %8.1f %8.1f\n", label,
            (slope(run.t, getfield(tr, f), w) for (run, tr) in zip(runs, trs))..., exact)
end

fig = Figure(size = (760, 520))
ax = Axis(fig[1, 1], xlabel = "x (m)", ylabel = "t (ms)")
fine = runs[1]
hm = heatmap!(ax, fine.x, 1e3 .* fine.t, log10.(reduce(hcat, fine.p_mean) ./ p0),
              colormap = :grays)
Colorbar(fig[1, 2], hm, label = "log₁₀(p / p₀), plane average")
seg(x1, t1, speed, t2) = ([x1, x1 + speed * (t2 - t1)], 1e3 .* [t1, t2])
oned = [
    seg(x_shock, 0.0, incident.shock_speed, t_impact),
    seg(x_interface, t_impact, impact.left_speed, t_reflected),
    seg(x_interface, t_impact, impact.right_speed, t_wall),
    seg(Lx, t_wall, wall.left_speed, t_reshock),
    seg(x_reshock, t_reshock, reshock.left_speed, min(t_retransmitted, opt.tfinal)),
    seg(x_reshock, t_reshock, reshock.right_speed,
        t_reshock + (Lx - x_reshock) / reshock.right_speed),
]
for (xs, ts) in oned
    lines!(ax, xs, ts, color = :orange, linewidth = 1.5)
end
interface_1d = [seg(x_interface, t_impact, impact.u_star, t_reshock),
                seg(x_reshock, t_reshock, reshock.u_star, opt.tfinal)]
for (xs, ts) in interface_1d
    lines!(ax, xs, ts, color = :orange, linestyle = :dash, linewidth = 1.5)
end
lines!(ax, trs[1].interface, 1e3 .* fine.t, color = :dodgerblue, linewidth = 1)
limits!(ax, 0, Lx, 0, 1e3opt.tfinal)
Legend(fig[2, 1:2], [LineElement(color = :orange), LineElement(color = :orange,
                     linestyle = :dash), LineElement(color = :dodgerblue)],
       ["1-D waves, sharp interface", "1-D interface", "mean mole fraction 0.5"],
       orientation = :horizontal, framevisible = false)
save(joinpath(outdir, "xt.png"), fig) #src
nothing #hide

# ![Plane-averaged pressure against position and time](../assets/examples/shock_tube/xt.png)
#
# The figure shows the run at ``\Delta x = 0.78`` mm, with the waves of the
# one-dimensional solution drawn over it. The fitted velocities are
#
# | velocity (m/s) | ``\Delta x = 0.78`` mm | ``\Delta x = 1.56`` mm | 1-D |
# |:--|--:|--:|--:|
# | incident shock | 515.4 | 511.3 | 516.5 |
# | shock transmitted into the SF6 | 241.0 | 240.9 | 241.1 |
# | interface after the first shock | 158.7 | 157.6 | 157.5 |
# | shock reflected from the end wall | −90.8 | −90.8 | −90.9 |
# | shock transmitted into the air at reshock | −403.8 | −405.3 | −404.2 |
# | interface after reshock | −64.2 | −60.9 | −58.7 |
#
# The shocks travel at their one-dimensional speeds to within 1% on both grids,
# and the reflected shock meets the layer where and when it meets the sharp
# interface. After the first shock, ``x_{50}`` moves at the contact velocity to
# within 0.8%. After reshock it moves faster than the one-dimensional
# interface, by 4% at ``\Delta x = 1.56`` mm and 9% at 0.78 mm.
#
# The plane ``x_{50}`` is not a material surface. After reshock the layer grows
# faster toward the air than toward the SF6: at ``\Delta x = 1.56`` mm, the
# plane where ``\langle X \rangle = 0.05`` moves at −78 m/s and the plane where
# it is 0.95 at −45 m/s. A position that follows the SF6 as a whole is that of
# a sharp interface holding the same volume of it, the end wall less
# ``\int \langle X \rangle\, dx``; at ``\Delta x = 1.56`` mm it moves at
# −58.4 m/s after reshock, within 0.5% of the one-dimensional interface.
#
# From 1.9 ms the layer leaves the dashed line, which continues the interface
# velocity of reshock and leaves out the waves that follow it. The rarefaction
# reflected into the SF6 at reshock reaches the end wall, 32 mm away, after
# 0.2 ms, is reflected there, and returns to the layer; the layer decelerates,
# comes to rest near 2.3 ms and then moves slowly toward the end wall.

# ## Width and mixing
#
# The width ``h`` of the layer is the distance between the planes where
# ``\langle X \rangle`` is 0.05 and 0.95. The molecular mixing fraction
#
# ```math
# \Theta = \frac{\int \langle X (1 - X) \rangle\, dx}
#               {\int \langle X \rangle \left(1 - \langle X \rangle\right) dx}
# ```
#
# compares the mixed gas in the layer with the amount its mean profile would
# hold if every plane had a uniform composition: it is 1 for a layer mixed
# across each plane and 0 for gases interleaved without mixing. Both measures
# use the mole fraction, which for gases at a common pressure and temperature
# is the fraction of the volume each occupies. [`mix_width`](@ref) and
# [`molecular_mixing`](@ref) form the same kind of measure from the mass
# fractions, which differ for these gases, whose molar masses differ by a
# factor of five: a mass fraction of one half is a mole fraction of 0.17.
# Every node of a plane has the same weight in the average, since the grid is
# uniform and periodic across the tube.

function mixing(run)
    h = [crossing(run.x, X, 0.95; from = :right, above = false) -
         crossing(run.x, X, 0.05) for X in run.X_mean]
    theta = [sum(XX) / sum(X .* (1 .- X)) for (X, XX) in zip(run.X_mean, run.XX_mean)]
    return (; h, theta)
end
layers = [mixing(run) for run in runs]
at(run, series, t) = series[argmin(abs.(run.t .- t))]

fig = Figure(size = (760, 520))
panels = [Axis(fig[i, 1], ylabel = l) for (i, l) in enumerate(("h (mm)", "Θ"))]
for (run, layer, color) in zip(runs, layers, Makie.wong_colors())
    lines!(panels[1], 1e3 .* run.t, 1e3 .* layer.h; color,
           label = @sprintf("Δx = %.2f mm", 1e3run.h))
    lines!(panels[2], 1e3 .* run.t, layer.theta; color)
end
for panel in panels
    vlines!(panel, 1e3 .* [t_impact, t_reshock], color = :gray, linestyle = :dot)
end
panels[2].xlabel = "t (ms)"
linkxaxes!(panels...)
hidexdecorations!(panels[1], grid = false)
Legend(fig[0, 1], panels[1], orientation = :horizontal, framevisible = false)
save(joinpath(outdir, "mixing.png"), fig) #src
nothing #hide

# ![Width and molecular mixing fraction of the layer](../assets/examples/shock_tube/mixing.png)
#
# The dotted lines mark the arrival of the incident shock and the reshock time
# of the one-dimensional solution. Each shock first compresses the layer, and
# its perturbations then grow. The calculation carries no molecular viscosity
# or diffusion: the gases mix through the artificial diffusivity and the
# filter, which act at the scale of the grid. The run prints the width at the
# start and just before the incident shock arrives,

for (run, layer) in zip(runs, layers)
    @printf("%3d rows: h = %.1f mm at the start, %.1f mm just before the shock arrives\n",
            run.ny, 1e3at(run, layer.h, 0.0), 1e3at(run, layer.h, t_impact - 5e-6))
end

# 6.8 and 6.7 mm at ``\Delta x = 0.78`` mm and 7.1 and 7.3 mm at 1.56 mm, so
# neither grid spreads the initial blend before the shock arrives. It also
# prints the growth rate of the width and the velocity of ``x_{50}`` over four
# windows, and the width and ``\Theta`` at four times:

phases = [("before reshock", (t_reshock - 500e-6, t_reshock - 100e-6)),
          ("after reshock", (t_reshock + 100e-6, t_reshock + 400e-6)),
          ("deceleration", (t_reshock + 500e-6, t_reshock + 800e-6)),
          ("late", (t_reshock + 900e-6, opt.tfinal))]
println("                                  dh/dt (m/s)        interface (m/s)")
@printf("%-16s %-14s %8d %8d %8d %8d\n", "", "window (ms)", grids..., grids...)
for (label, w) in phases
    @printf("%-16s %5.2f–%5.2f   %8.1f %8.1f %8.1f %8.1f\n", label, 1e3w[1], 1e3w[2],
            (slope(run.t, layer.h, w) for (run, layer) in zip(runs, layers))...,
            (slope(run.t, tr.interface, w) for (run, tr) in zip(runs, trs))...)
end
for (run, layer) in zip(runs, layers)
    for t in (t_reshock - 50e-6, t_reshock + 100e-6, t_reshock + 500e-6, opt.tfinal)
        @printf("%3d rows, %.2f ms: h = %5.1f mm, Θ = %.2f\n", run.ny, 1e3t,
                1e3at(run, layer.h, t), at(run, layer.theta, t))
    end
end

# Before reshock the two grids give the same width: 22.3 mm on both at 1.43 ms,
# growing at 9.6 and 8.8 m/s over the 0.4 ms before reshock. The sawtooth on
# the curve at ``\Delta x = 1.56`` mm, about 0.1 mm root mean square, has the
# period, about 10 µs, in which the layer moves one cell along the grid.
#
# Over the 0.3 ms after reshock the layer grows at 43.7 m/s at
# ``\Delta x = 0.78`` mm and 36.2 m/s at 1.56 mm, four to five times its rate
# before reshock. With the velocity jump and the Atwood number of reshock,
# ``(dh/dt)/(A \Delta u)`` is 0.27 and 0.22. The grids then differ more in when
# the layer grows than in how far: the finer grid leads by 2 mm (7%) at 2.0 ms,
# and at 3 ms the widths are 60.3 and 61.6 mm, within 2%. A growth rate fitted
# over a fixed window after reshock differs between the grids by up to 17% of
# the fine grid's rate, and these grids do not settle it.
#
# Of the four windows, the layer grows fastest between 2.0 and 2.3 ms, at 51
# and 56 m/s, while it decelerates. The run prints the plane-averaged pressure
# on either side of the layer during the deceleration and after it:

for t in (t_reshock + 650e-6, t_reshock + 1.2e-3)
    run, layer = runs[1], layers[1]
    k = argmin(abs.(run.t .- t))
    X = run.X_mean[k]
    lo, hi = crossing(run.x, X, 0.05), crossing(run.x, X, 0.95; from = :right,
                                                    above = false)
    p_at(x) = run.p_mean[k][argmin(abs.(run.x .- x))]
    @printf("%.2f ms: plane-mean pressure %.0f kPa 5 mm on the air side, \
            %.0f kPa 5 mm on the SF6 side\n", 1e3t, p_at(lo - 5e-3) / 1e3,
            p_at(hi + 5e-3) / 1e3)
end

# At 2.13 ms the pressure is 590 kPa 5 mm on the air side of the layer and
# 479 kPa 5 mm on the SF6 side. The pressure falls from the light gas to the
# heavy one, so the deceleration is in the sense that is Rayleigh–Taylor
# unstable. At 2.68 ms the difference has reversed, 505 against 559 kPa, and
# from 2.4 ms the layer grows at 17 to 20 m/s.
#
# The molecular mixing fraction is not settled. The first shock lowers
# ``\Theta`` as the perturbations grow, to 0.65 by 0.4 ms at
# ``\Delta x = 0.78`` mm and more slowly to about 0.85 at 1.56 mm; at 0.78 mm
# it then recovers to 0.78 by reshock. Reshock raises it on both grids for a
# short time before it falls again. It is lower at ``\Delta x = 0.78`` mm than
# at 1.56 mm at every time after the first shock, by as much as 0.2 at 0.4 ms,
# and at 3 ms it is 0.57 against 0.70.

# ## Images
#
# The SF6 mole fraction on both grids at seven times, in a window 100 mm long
# centered on ``x_{50}``; gray is beyond the end wall.

fig = Figure()
for (row, run) in enumerate(runs), (col, frame) in enumerate(run.frames)
    center = at(run, trs[row].interface, frame.t)
    center = isfinite(center) ? center : x_interface
    panel = Axis(fig[row, col], width = 92, height = 92, xticks = [-40, 0, 40],
                 backgroundcolor = :gray60,
                 title = row == 1 ? @sprintf("%.2f ms", 1e3frame.t) : "",
                 xlabel = row == length(runs) && col == 1 ? "x − x₅₀ (mm)" : "",
                 ylabel = col == 1 ? @sprintf("Δx = %.2f mm\ny (mm)", 1e3run.h) : "")
    heatmap!(panel, 1e3 .* (run.x .- center), 1e3 .* run.y, frame.X, colormap = :viridis,
             colorrange = (0, 1))
    limits!(panel, -50, 50, 0, 1e3Ly)
    col > 1 && hideydecorations!(panel)
    row < length(runs) && hidexdecorations!(panel)
end
Colorbar(fig[1:length(runs), end + 1], colormap = :viridis, limits = (0, 1),
         label = "SF6 mole fraction")
colgap!(fig.layout, 6)
resize_to_layout!(fig)
save(joinpath(outdir, "mole_fraction.png"), fig) #src
nothing #hide

# ![SF6 mole fraction](../assets/examples/shock_tube/mole_fraction.png)
#
# By 0.70 ms the modes have grown into spikes of SF6 in the air, rolled up at
# their tips. At 1.60 ms, just after reshock, the layer is compressed. From
# 2.0 ms it grows toward both gases, with rolled-up structures on the SF6 side
# as well as on the air side, and the structures merge into fewer and larger
# ones. The two grids show the same arrangement of structures through 1.60 ms
# and different ones from 2.0 ms. At 3 ms the run at ``\Delta x = 0.78`` mm
# holds filaments of each gas a few cells thick that the run at 1.56 mm does
# not have.
#
# ## What this checks
#
# - The shocks of the first impact, of the reflection from the end wall and of
#   reshock travel at the speeds of the one-dimensional Riemann solutions to
#   within 1% at ``\Delta x = 1.56`` and 0.78 mm.
# - The layer moves at the contact velocity after the first shock. After
#   reshock the plane where the mean mole fraction is one half drifts as the
#   layer grows unevenly, and at ``\Delta x = 1.56`` mm the volume of SF6 moves
#   at the contact velocity to within 0.5%.
# - Before reshock the width of the layer is the same on both grids. After
#   reshock the width differs by up to 7% between them, and the growth rate
#   over a fixed window by up to 17%; the widths at the end of the run agree to
#   2%.
# - The molecular mixing fraction is lower on the finer grid at every time
#   after the first shock, by as much as 0.2, and is not settled on these
#   grids.
# - The layer grows fastest while the waves reflected from the end wall
#   decelerate it with the pressure higher on the air side.

grid = "Δx = " * join((@sprintf("%.2f mm", 1e3r.h) for r in runs), " and ") #src
command = "julia --project=docs -t 8 examples/shock_tube.jl" #src
write_provenance(outdir; command, settings = opt, wall = time() - T_START, grid) #src
