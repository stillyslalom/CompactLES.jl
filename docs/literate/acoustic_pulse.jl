# # Your first CompactLES simulation
#
# This tutorial follows a small pressure disturbance around a periodic
# one-dimensional domain. It introduces the complete path from a physical
# problem specification to an evolved solution and an ``x``--``t`` diagram.
# Only the ``x`` direction is resolved; the other two dimensions contain one
# point and therefore incur no derivatives, halo storage, or filtering work.

using MPI
MPI.Initialized() || MPI.Init(threadlevel=:funneled)

using CompactLES
using CairoMakie
CairoMakie.activate!(type = "png")

# ## Specify the physical problem
#
# We use nondimensional variables and a calorically perfect gas. The background
# state is ``p_0 = rho_0 = 1``. A Gaussian pressure perturbation of amplitude
# ``10^{-3}`` is small enough for its propagation speed to be close to the
# acoustic speed ``c_0 = sqrt(gamma p_0/rho_0)``.

gamma = 1.4
amplitude = 1.0e-3
center = 0.30
width = 0.035

pressure(x) = 1 + amplitude * exp(-((x - center) / width)^2)

problem = Problem(
    name = "periodic acoustic pulse",
    eos = single_species(gamma = gamma),
    domain = ((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)),
    bcs = ntuple(_ -> (PeriodicBC(), PeriodicBC()), 3),
    ic = (x, y, z) -> begin
        p = pressure(x)
        Prim(p = p, rho = p^(1 / gamma))
    end,
)

# [`Numerics`](@ref) contains resolution and algorithmic choices rather than
# physics. Filtering is disabled for this smooth, well-resolved disturbance.

numerics = Numerics(
    n_global = (128, 1, 1),
    deriv = lele_d1_6(),
    art = ArtParams(enabled = false),
    cfl = 0.6,
    filter_interval = 0,
)

solver, Q = setup(problem, numerics)

# ## Read the state without depending on its layout
#
# ``Q`` stores conserved quantities, including halo points. [`gidx`](@ref)
# maps a local interior index to the padded array, while
# [`mixture_density`](@ref) avoids assuming where species densities reside.

function density_line(solver, Q)
    nx = solver.decomp.n_local[1]
    [mixture_density(solver, Q, gidx(solver, i, 1, 1)) for i in 1:nx]
end

x = [xcoord(solver, 1, i) for i in 1:solver.decomp.n_local[1]]
tfinal = 0.30
times = collect(range(0.0, tfinal; length = 61))
snapshots = [density_line(solver, Q)]

# [`AtTime`](@ref) asks `run!` to end steps exactly at the requested times.
# The columns of `rho_xt` therefore lie on a uniform physical-time axis even
# though the CFL-controlled timestep varies slightly.

record = Callback(AtTime(times[2:end]), function (solver, Q)
    push!(snapshots, density_line(solver, Q))
    nothing
end)

run!(solver, Q; tfinal, nmax = 10_000, callback = record)
rho_xt = reduce(hcat, snapshots)

# ## Interpret the result
#
# With zero initial velocity, the perturbation separates into equal left- and
# right-travelling waves. Periodicity makes a wave leaving one side re-enter
# from the other. The plotted quantity is the density perturbation rather than
# density itself so the small acoustic signal remains visible.

fig = Figure(size = (760, 420))
ax = Axis(fig[1, 1], xlabel = "x", ylabel = "time",
          title = "Density perturbation")
hm = heatmap!(ax, x, times, rho_xt .- 1; colormap = :balance)
Colorbar(fig[1, 2], hm, label = "rho - rho0")
fig
