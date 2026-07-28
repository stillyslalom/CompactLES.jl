# # Construct a multicomponent state
#
# A useful simulation begins with a thermodynamically consistent state. This
# tutorial constructs a helium--carbon-dioxide interface at uniform pressure
# and temperature using the bundled NASA-9 data. No hydrodynamic evolution is
# needed: the figure is produced directly from the initialized conserved state.

using MPI
MPI.Initialized() || MPI.Init(threadlevel=:funneled)

using CompactLES
using CairoMakie
CairoMakie.activate!(type = "png")

# ## Select a caloric model
#
# [`read_nasa9`](@ref) reads piecewise heat-capacity fits and derives each
# species gas constant from its molar mass. `Nasa9Mixture` is thermally ideal,
# ``p=rho R_m T``, but calorically imperfect: heat capacities depend on
# temperature. Species order fixes the order of every mass-fraction tuple.

species = read_nasa9(["He", "CO2"])
eos = Nasa9Mixture(species)

# ## Define pressure, temperature, and composition
#
# [`Prim`](@ref) requires pressure, mass fractions, and exactly one of density
# or temperature. Supplying temperature asks the EOS to calculate density.
# Here the pressure and temperature are uniform while composition varies, so
# the density change follows solely from the mixture gas constant.

nx = 192
transition_width = 0.025

problem = Problem(
    name = "helium--carbon-dioxide interface",
    eos = eos,
    transport = Transport(mu0 = 0.0),
    domain = ((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)),
    bcs = ((SlipWallBC(), SlipWallBC()),
           (PeriodicBC(), PeriodicBC()),
           (PeriodicBC(), PeriodicBC())),
    ic = (x, y, z) -> begin
        Yco2 = tanh_blend(x, 0.5, transition_width)
        Prim(Y = (1 - Yco2, Yco2), p = 101_325.0, T_ion = 300.0)
    end,
)

numerics = Numerics(
    n_global = (nx, 1, 1),
    art = ArtParams(enabled = false),
    filter_interval = 0,
)

solver, Q = setup(problem, numerics)

# ## Inspect the initialized conserved state
#
# CompactLES stores partial densities rather than mass fractions.
# [`line_profile`](@ref) recovers mixture quantities without exposing that
# component layout: `:rho` gives the mixture density, and `:Y` with a `species`
# index gives a mass fraction, in the order fixed above. No evolution is run, so
# these profiles describe the initialized state directly.

x, density = line_profile(solver, Q, :rho)
_, Yhe = line_profile(solver, Q, :Y; species = 1)
_, Yco2 = line_profile(solver, Q, :Y; species = 2)

fig = Figure(size = (760, 600))
ax1 = Axis(fig[1, 1], xlabel = "x", ylabel = "mass fraction",
           title = "Composition")
lines!(ax1, x, Yhe, label = "He")
lines!(ax1, x, Yco2, label = "CO2")
axislegend(ax1, position = :rc)

ax2 = Axis(fig[2, 1], xlabel = "x", ylabel = "density (kg m^-3)",
           title = "Density at 101325 Pa and 300 K")
lines!(ax2, x, density, color = :black)
fig

# The heavier molecular species has the smaller specific gas constant and is
# consequently denser at the same pressure and temperature. If density had
# been supplied instead, the EOS would have inferred temperature. Supplying
# both, or neither, is rejected so inconsistent initial thermodynamic data do
# not pass silently into the calculation.
