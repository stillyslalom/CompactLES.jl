# # Two-dimensional shock tube with a switchable boundary
#
# This tutorial computes the interaction of a planar shock with a perturbed
# interface separating a light gas from a heavy one — the Richtmyer–Meshkov
# problem — in a domain whose upstream boundary condition must change type
# during the calculation.
#
# The requirement arises from the wave structure rather than from the
# interaction itself. Sustaining the incident shock requires the shocked driven
# gas to be held at the upstream face, which is a subsonic inflow condition.
# The interaction, however, emits a reflected wave that propagates back toward
# that face. An inflow condition continues to impose the pre-interaction state
# and therefore does not transmit the reflected wave out of the domain; the
# wave is returned to the interface, and the second interaction that follows is
# an artifact of the truncated domain rather than a feature of the physical
# problem. The boundary must cease to impose a state and begin to absorb one at
# the time the reflected wave arrives.
#
# [`SwitchableBC`](@ref) provides that change of condition and
# [`WhenState`](@ref) supplies the criterion. The sections below construct the
# case, record the instant at which the switch occurs, and compare the result
# against an otherwise identical calculation in which the inflow condition is
# retained throughout.

using MPI
MPI.Initialized() || MPI.Init(threadlevel=:funneled)

using CompactLES
using CairoMakie
CairoMakie.activate!(type = "png")

# ## Gas properties and the incident shock
#
# Two ideal gases share a ratio of specific heats and differ only in gas
# constant, so that at equal pressure and temperature their density ratio is
# `RL / RH`. The value 4 adopted here corresponds to an Atwood number of 0.6.

γ, RL, RH = 1.4, 1.0, 0.25
eos = IdealMixture([IdealSpecies{Float64}("light", RL, γ),
                    IdealSpecies{Float64}("heavy", RH, γ)])

# The calculation begins with the incident shock already formed rather than
# with a diaphragm rupture. State 1 denotes the undisturbed driven gas and
# state 2 the region behind a shock of Mach number `MS` advancing into it,
# obtained from the normal-shock relations. Initializing state 2 directly
# places the shock–interface interaction at the beginning of the calculation.

MS = 1.5
p1 = ρ1 = T1 = 1.0
c1 = sqrt(γ * p1 / ρ1)
p2 = p1 * (2γ * MS^2 - (γ - 1)) / (γ + 1)
ρ2 = ρ1 * ((γ + 1) * MS^2) / ((γ - 1) * MS^2 + 2)
u2 = c1 * 2 / (γ + 1) * (MS^2 - 1) / MS
T2 = p2 / (ρ2 * RL)

(; p2, ρ2, u2, T2)

# ## Frame of reference
#
# A uniform velocity `U` is added to every region. It is selected so that the
# interface is nearly stationary after the shock has traversed it, which is the
# conventional frame for shock–interface calculations: the interface remains
# within the domain rather than being convected out of it.
#
# The translation also displaces both faces from `u = 0`. The NSCBC relaxation
# coefficient contains a factor `1 - M²`, and the transverse blending is
# weighted by the local Mach number, so a boundary evaluated at rest is
# evaluated in a degenerate limit.

U = -0.5
Ma_upstream = (u2 + U) / sqrt(γ * p2 / ρ2)

# ## Domain and initial condition
#
# The `x` axis spans the tube, `y` is periodic and contains a single wavelength
# of the interface perturbation, and `z` is collapsed to one point. A dimension
# with a single global point carries no derivative, halo exchange, or
# decomposition, which allows the three-dimensional frontend to compute a
# two-dimensional problem at two-dimensional cost.

nx, ny = 160, 80
Lx, Ly = 1.0, 0.5
hx = Lx / (nx - 1)
x_shock, x_iface, amp = 0.20, 0.40, 0.05

ic = function (x, y, z)
    xi = x_iface + amp * cos(2π * y / Ly)      # perturbed interface
    s = tanh_blend(x, x_shock, 2hx)            # 0 shocked, 1 undisturbed
    θ = tanh_blend(x, xi, 2hx)                 # 0 light,   1 heavy
    Prim(Y = (1 - θ, θ),
         u = ((1 - s) * u2 + U, 0.0, 0.0),
         p = (1 - s) * p2 + s * p1,
         T_ion = (1 - s) * T2 + s * T1)
end;

# ## Boundary conditions
#
# `NSCBCInflowBC` replaces each incoming characteristic with a relaxation
# toward a prescribed velocity, temperature, and composition. This is
# appropriate while the gas at the face remains uniform state 2, and
# inappropriate once the reflected shock has arrived and the state there has
# changed.
#
# `NSCBCOutflowBC` serves as the `after` condition. Its relaxation parameter
# `sigma` is deliberately small. The pressure at this face following passage of
# the reflected shock forms part of the solution of the interaction and is not
# known in advance; relaxing strongly toward a prescribed `pinf` would impose a
# value known to be incorrect and would displace the driven section with it. A
# weak relaxation transmits the outgoing wave while leaving the pressure
# largely unconstrained.

upstream = SwitchableBC(NSCBCInflowBC(u = (u2 + U, 0.0, 0.0), T_ion = T2,
                                      Y = [1.0, 0.0]),
                        NSCBCOutflowBC(pinf = p2, sigma = 0.05))

# No wave reaches the downstream face during the calculation — the transmitted
# shock advances only to approximately `x = 0.75` — so prescribing the
# undisturbed heavy state there with a `DirichletBC` is exact, and confines the
# discussion to a single boundary.

downstream = DirichletBC((x, y, z, t) -> Prim(Y = (0.0, 1.0), u = (U, 0.0, 0.0),
                                              p = p1, T_ion = T1))

problem(xlo_bc) = Problem(
    name = "2-D shock/interface interaction",
    eos = eos,
    transport = Transport(mu0 = 0.0),          # Euler + artificial regularization
    domain = ((0.0, Lx), (0.0, Ly), (0.0, hx)),
    bcs = ((xlo_bc, downstream),
           (PeriodicBC(), PeriodicBC()),
           (PeriodicBC(), PeriodicBC())),
    ic = ic,
)

numerics = Numerics(n_global = (nx, ny, 1), art = ArtParams(enabled = true),
                    cfl = 0.4, filter_interval = 1)

solver, Q = setup(problem(upstream), numerics)

# ## Field extraction
#
# The mixture density is the sum of the species densities and is available
# directly from the conserved array, by the same expression in serial and
# decomposed calculations.

function density(solver, Q)
    nxl, nyl, _ = solver.decomp.n_local
    [Q[gidx(solver, i, j, 1), 1] + Q[gidx(solver, i, j, 1), 2]
     for i in 1:nxl, j in 1:nyl]
end

xs = [xcoord(solver, 1, i) for i in 1:nx]
ys = [xcoord(solver, 2, j) for j in 1:ny];

# ## Switching criterion
#
# The reflected shock is taken to have arrived when the density at any point on
# the upstream plane exceeds that of state 2. `CompactLES.wallplane` returns
# that plane, or `nothing` on a rank that does not own it, so the predicate
# below yields a rank-local verdict; [`WhenState`](@ref) reduces it across the
# communicator before acting on it.
#
# The reduction is required for correctness rather than for convenience.
# `NSCBCOutflowBC` performs collective operations that `NSCBCInflowBC` does
# not. Were the ranks to disagree as to whether the switch had occurred, some
# would enter a collective that the others never reach, and the calculation
# would stall at zero CPU utilization rather than raise an error. A switch must
# therefore be driven by a [`Callback`](@ref), and never from an unreduced
# rank-local predicate.
#
# The maximum over the plane is used in preference to a centreline value
# because the perturbed interface renders the reflected shock non-planar: it
# reaches the boundary beneath the spike before it does beneath the bubble, and
# arrival is defined here as first contact at any transverse station.

function upstream_density(solver, Q)
    plane = CompactLES.wallplane(solver.decomp, 1, 1)
    plane === nothing && return -Inf         # this rank does not own the face
    return maximum(I -> Q[I, 1] + Q[I, 2], plane)
end

# The threshold must also exceed an unrelated transient. The initial shock is
# represented by a hyperbolic-tangent profile rather than by an exact discrete
# shock, and relaxes onto the profile the scheme supports, emitting a small
# acoustic pulse upstream. That pulse reaches the face at `t ≈ 0.19` and raises
# the density approximately 3% above state 2, whereas the reflected shock
# raises it by 10% and subsequently by 30%. A threshold of 6% separates them.

reflected_wave_arrived(solver, Q) = upstream_density(solver, Q) > 1.06ρ2

# ## Time integration
#
# Two callbacks are supplied: the first records the upstream density at regular
# intervals, so that the switch may be located afterwards, and the second
# performs the switch and records the time at which it occurs.

tfinal = 0.85
history = (t = Float64[], ρ = Float64[])
switch_time = Ref(NaN)

record! = Callback(EveryStep(5), function (solver, Q)
    push!(history.t, solver.t)
    push!(history.ρ, upstream_density(solver, Q))
    nothing
end)

absorb! = Callback(WhenState(reflected_wave_arrived), function (solver, Q)
    switch!(upstream)
    switch_time[] = solver.t
    nothing
end)

# Fields are stored at prescribed instants using [`AtTime`](@ref). `run!` clips
# `dt` so that a step terminates exactly on a scheduled instant rather than
# overshooting it.

snaps = Pair{Float64,Matrix{Float64}}[]
snapshot! = Callback(AtTime([0.10, 0.30, 0.85]), function (solver, Q)
    push!(snaps, solver.t => density(solver, Q))
    nothing
end)

run!(solver, Q; tfinal, nmax = 100_000,
     callback = (record!, absorb!, snapshot!))

switched(upstream), switch_time[]

# The switch occurred during the calculation, on the step at which the
# reflected shock reached the upstream plane.

# ## Evolution of the interaction
#
# Density is presented on a logarithmic scale. The calculation spans densities
# from 1 in the undisturbed light gas to approximately 9.5 in the shocked heavy
# gas, and on a linear scale the states below 4 are not distinguishable.

ρlims = log10.(extrema(reduce(vcat, [vec(ρ) for (_, ρ) in snaps])))

fig = Figure(size = (960, 330))
for (n, (t, ρ)) in enumerate(snaps)
    ax = Axis(fig[1, n], title = "t = $(round(t, digits=2))",
              xlabel = "x", ylabel = n == 1 ? "y" : "", aspect = DataAspect())
    heatmap!(ax, xs, ys, log10.(ρ), colormap = :turbo, colorrange = ρlims)
    n > 1 && hideydecorations!(ax, grid = false)
end
Colorbar(fig[1, 4], colormap = :turbo, limits = ρlims, label = "log₁₀ ρ")
fig

# At `t = 0.10` three states are present: the shocked driven gas held at the
# upstream face (cyan), the undisturbed driven gas ahead of the incident shock
# (dark), and the heavy gas beyond the perturbed interface (yellow). By
# `t = 0.30` the shock has traversed the interface, depositing baroclinic
# vorticity that inverts and amplifies the perturbation, driving a transmitted
# shock into the heavy gas (dark red) and returning a reflected shock upstream,
# visible as the broad feature propagating leftward through the cyan region. At
# `t = 0.85` the interface exhibits the characteristic Richtmyer–Meshkov spike
# and bubble, and the transmitted shock has advanced to approximately
# `x = 0.75`, short of the downstream face.

# ## Control case: the inflow condition retained
#
# The calculation is repeated with the upstream condition held fixed.

solver_h, Q_h = setup(problem(NSCBCInflowBC(u = (u2 + U, 0.0, 0.0), T_ion = T2,
                                            Y = [1.0, 0.0])), numerics)
history_h = (t = Float64[], ρ = Float64[])

record_h! = Callback(EveryStep(5), function (solver, Q)
    push!(history_h.t, solver.t)
    push!(history_h.ρ, upstream_density(solver, Q))
    nothing
end)

run!(solver_h, Q_h; tfinal, nmax = 100_000, callback = record_h!)
ρ_held = density(solver_h, Q_h);

# ## Effect of the switch
#
# The density on the upstream plane is sufficient to characterize the
# difference. The two calculations employ the same condition until the switch
# and are therefore identical up to that step, including through the startup
# transient at `t ≈ 0.19`, which both treat in the same way. Thereafter the
# retained inflow condition continues to relax toward state 2, which no longer
# describes the gas at the face, and the resulting mismatch propagates back
# into the domain; the switched boundary transmits the wave and settles at a
# lower level.

fig2 = Figure(size = (760, 340))
ax = Axis(fig2[1, 1], xlabel = "t", ylabel = "max ρ on the upstream plane")
vlines!(ax, switch_time[], color = :gray, linestyle = :dash)
text!(ax, switch_time[], 1.88, text = " switch", align = (:left, :bottom),
      color = :gray)
lines!(ax, history_h.t, history_h.ρ, label = "inflow held", linewidth = 2)
lines!(ax, history.t, history.ρ, label = "switched to outflow", linewidth = 2)
axislegend(ax, position = :lt)
fig2

# The difference between the two final density fields isolates the spurious
# reflection: the wave returned by the retained inflow condition, and the
# position it has reached by the end of the calculation.

fig3 = Figure(size = (620, 300))
ax = Axis(fig3[1, 1], xlabel = "x", ylabel = "y", aspect = DataAspect(),
          title = "ρ (held) − ρ (switched)")
Δ = ρ_held .- last(snaps).second
m = maximum(abs, Δ)
heatmap!(ax, xs, ys, Δ, colormap = :balance, colorrange = (-m, m))
Colorbar(fig3[1, 2], colormap = :balance, limits = (-m, m))
fig3

# The discrepancy is not confined to the neighbourhood of the upstream
# boundary. It has traversed the domain and is concentrated at the interface,
# so an inappropriate boundary condition here contaminates the quantity of
# interest rather than a boundary layer of cells.

# ## Remarks
#
# * Both conditions in a [`SwitchableBC`](@ref) must agree on periodicity, and
#   neither may be a fold condition (`AxisBC`, `OriginBC`, `PoleBC`); `setup`
#   identifies those by type, and a wrapped fold would not be constructed.
# * Prior to switching, the wrapper is bit-identical to the condition it wraps,
#   so it may be left in a problem specification that never switches.
# * `density` and `upstream_density` above are rank-local. This suffices for
#   `WhenState`, which reduces the verdict itself, but a diagnostic gathered in
#   this manner is correct only in serial. [`save_vtk`](@ref) is appropriate for
#   post-processing, as are the collective diagnostics described in
#   [Runtime and output](@ref), where `plane_profile` and related functions
#   reduce across the communicator.
# * The behaviour illustrated here is verified against a calculation on a domain
#   long enough that the upstream boundary cannot influence the result; see the
#   `SwitchableBC` testsets in `test/runtests.jl`.
