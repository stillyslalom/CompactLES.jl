# The artificial species channel in three dimensions: `species_flux = :bulk`
# against the default `:fickian`. The one-dimensional battery
# (`test/cases.jl`, `brill_slab` and `shock_interface`) is what the channel was
# built on; nothing has measured it on a three-dimensional case, where the
# artificial viscosities are active and the interface is curved. Four parts:
#
#   slab       a heavy blob advected through a periodic cube at uniform p and T,
#              the three-dimensional `brill_slab`: the pressure-equilibrium
#              measurement. Every linear operator of the scheme holds a uniform
#              (u, p, T) state to round-off; the Fickian enthalpy flux does not
#              at unequal gas constants, and this is where that shows.
#   bubble     a Mach 1.5 air shock into a sphere of the heavy gas, the
#              three-dimensional `shock_interface`: mass-fraction excursions,
#              mixing measures and the energy drift.
#   budget     the species channel's own energy budget on the bubble case, by
#              differencing two right-hand sides (below).
#   constants  the bubble case under `:bulk` over C_D x C_Y, since both
#              constants are inherited from the Fickian channel and neither has
#              been fitted for this one.
#
#   mpiexec -n 8 julia --project=. -t 1 bench/bulkchannel.jl
#   mpiexec -n 8 julia --project=. -t 1 bench/bulkchannel.jl slab N=64 periods=4
#   mpiexec -n 2 julia --project=. -t 1 bench/bulkchannel.jl bubble nx=32 ny=16 \
#       nmax=20                                                    # smoke run
#
# Positional: parts. Keys: channels (fickian, bulk, or both); ratio
# (comma-separated heavy/light density ratios at the pre-shock p and T; 5.04 is
# SF6 against air and 100 is the ratio the Fickian channel fails at in one
# dimension); N, periods, Np, gamma_heavy, diagonal (the slab's grid, number of
# domain transits, interface width in cells, the heavy gas's ratio of specific
# heats, and whether the advection is along x or along the cube diagonal); nx,
# ny, delta, tfin, Ly (the bubble's grid, interface width in cells, end time
# and transverse extent; nz = ny); nxc, nyc (the grid of the `constants` part
# alone, smaller because it runs nine points); C_Ds, C_Ys (that part's two
# axes); cfl; every (diagnostic and budget-sample cadence in steps); nmax;
# progress (`ProgressLog` cadence in steps, 0 off).
#
# Cost: the bubble grid is 128 x 64 x 64 and the run about 650 steps, so one
# channel-ratio point is a few minutes at eight ranks and the default
# `parts=slab,bubble,budget,constants` is a campaign, not a check. Run one part
# per invocation. The `budget` part evaluates three right-hand sides per sample,
# about 6% on top of the run at the default cadence, and prints one row per
# sample; the callbacks are outside `solver.wall_step`, so the steady ms/step
# column stays comparable with the `bubble` part's. It also holds three solvers
# at once, so run it under `mpiexec` at the production grid.
#
# The `budget` part evaluates the right-hand side three times on the same
# state: on the run's own solver (F), on a second solver with `C_D = C_Y = 0`
# (D), and on a third with every artificial constant zero (A). The species
# channel is then F - D and the three artificial viscosities together are
# D - A, so the species channel's kinetic-energy rate can be read beside the
# one mu*, beta* and kappa* produce, which is what the `chan/(sum)` column is.
# Both isolations are exact rather than approximate: `compute_artificial!`
# writes mu*, beta* and kappa* before the species sweep and neither C_D nor C_Y
# enters them, so F and D carry identical viscous coefficients, and with both
# constants zero the species bracket is zero at every point, hence
# D*_k = D_b = 0 in D and A alike. Each difference is therefore the divergence
# of its own fluxes alone, plus whatever the boundary corrections make of it at
# the two Dirichlet ends. The momentum and total-mass components of the species
# difference are identically zero under `:fickian` (the correction velocity
# keeps the species fluxes summing to zero and the flux has no momentum
# component), and the table prints their maxima so that this reads as a
# property of the instrument rather than an assumption. `max D` and
# `max mu*/rho` are the two diffusivities the rates come from, both read from
# the live solver's own coefficient arrays.
#
# Two channels are outside this budget. The compact filter's own sink is not
# measured here at all, and it is the dominant one on the cases measured so far
# (`bench/tgv_energy.jl` measures it directly on the Taylor-Green vortex);
# these columns therefore rank the artificial channels against each other and
# not against the total dissipation. Molecular transport is off on both cases,
# so `D - A` is purely artificial, and the three viscosities enter it together.
#
# The bubble's x ends are open: they hold their initial state through a
# `DirichletBC`, so the reported drift of the integrated total energy and mass
# is not a conservation error. No boundary flux integral is subtracted. What it
# is useful for is the comparison between the two channels on the same
# geometry, since both see the same ends. The same openness is why the budget
# part ranks the artificial channels against each other rather than against the
# net kinetic-energy change: on this case that change is dominated by the work
# the inflow does, and it is negative while the shock is crossing the domain.
#
# The mass-fraction extremes are the worst over every step, read from `Q` in a
# callback: the excursions of a shocked interface are transient and the final
# profile need not show them. The bubble case runs under
# `validity = :permissive` for the reason `shock_interface` records; its
# interface violates the mass-fraction dead band by design.
#
# Scratch tooling, like everything else in bench/: it prints tables and asserts
# nothing.

using MPI
MPI.Init(threadlevel=:funneled)
using CompactLES
using Printf

const CL = CompactLES
const PER = ntuple(_ -> (PeriodicBC(), PeriodicBC()), 3)

const DEFAULTS = (parts = "slab,bubble,budget,constants",
                  channels = "fickian,bulk", ratio = "5.04,100",
                  N = 48, periods = 2.0, Np = 7, gamma_heavy = 1.4,
                  diagonal = false,
                  nx = 128, ny = 64, delta = 2.0, tfin = 0.25, Ly = 0.5,
                  nxc = 96, nyc = 48,
                  C_Ds = "0.005,0.01,0.02", C_Ys = "50,100,200",
                  cfl = 0.4, every = 10, nmax = typemax(Int), progress = 0)

const opt = CompactLES.script_args(ARGS, DEFAULTS; positional = (:parts,))

const SLAB_U = 10.0                    # advection speed per active direction
const SLAB_RADIUS = 0.25               # blob radius in the unit cube
const BUBBLE_MACH = 1.5
const BUBBLE_X_SHOCK = 0.15
const BUBBLE_X_CENTRE = 0.4
const BUBBLE_RADIUS = 0.1
const BUBBLE_GAMMA_HEAVY = 1.09        # SF6, as in `shock_interface`
const BAND = 1e-4                      # mass-fraction band for the end count

# --- settings ---------------------------------------------------------------

"Comma-separated floats, as the sweep axes of the other bench scripts take."
function parse_floats(spec, key)
    out = Float64[]
    for item in split(String(spec), ',')
        s = strip(item)
        isempty(s) && continue
        v = tryparse(Float64, s)
        v === nothing && error("bad $key entry '$item', want a float")
        push!(out, v)
    end
    isempty(out) && error("$key list must not be empty")
    return out
end

function parse_names(spec, key, allowed)
    out = String[]
    for item in split(String(spec), ',')
        s = String(strip(item))
        isempty(s) && continue
        s in allowed || error("$key entry '$s' is not one of $(join(allowed, ", "))")
        push!(out, s)
    end
    isempty(out) && error("$key list must not be empty")
    return out
end

"A light gas and a heavy one whose density is `ratio` at the reference p and T."
mixture(ratio, gamma_heavy, light) =
    IdealMixture([IdealSpecies{Float64}(light, 1.0, 1.4),
                  IdealSpecies{Float64}("heavy", 1 / ratio, gamma_heavy)])

art_for(channel, C_D, C_Y) =
    ArtParams(enabled = true, C_D = C_D, C_Y = C_Y,
              species_flux = Symbol(channel))

# --- diagnostics ------------------------------------------------------------
#
# Each of these is collective: every rank must call it. The integrals go
# through `volume_integral`, which reads a field rather than a closure, so the
# pointwise quantity is built into a reusable padded buffer first.

"∫ f dV with f given pointwise by `body(I)` on padded indices."
function integrate(solver, buf, body)
    n1, n2, n3 = solver.decomp.n_local
    for k in 1:n3, j in 1:n2, i in 1:n1
        I = gidx(solver, i, j, k)
        buf[I] = body(I)
    end
    return volume_integral(solver, buf)
end

function kinetic_energy(solver, Q, buf)
    m1, m2, m3 = solver.equations.i_mom
    return integrate(solver, buf, I -> begin
        ρ = mixture_density(solver, Q, I)
        0.5 * (Q[I, m1]^2 + Q[I, m2]^2 + Q[I, m3]^2) / ρ
    end)
end

total_energy_integral(solver, Q, buf) =
    integrate(solver, buf, I -> Q[I, solver.equations.i_energy])

mass_integral(solver, Q, buf) =
    integrate(solver, buf, I -> mixture_density(solver, Q, I))

"∫ Y_air Y_heavy dV, the unnormalized mixedness. Primitives must be current."
mixedness(solver, buf) =
    integrate(solver, buf, I -> solver.Y[1][I] * solver.Y[2][I])

"The worst mass fraction over both species, accumulated into two `Ref`s."
function scan_mass_fractions!(lo, hi, solver, Q)
    n1, n2, n3 = solver.decomp.n_local
    for k in 1:n3, j in 1:n2, i in 1:n1
        I = gidx(solver, i, j, k)
        y = Q[I, 1] / (Q[I, 1] + Q[I, 2])
        lo[] = min(lo[], y, 1 - y)
        hi[] = max(hi[], y, 1 - y)
    end
    return nothing
end

"Interior points whose mass fractions leave [-BAND, 1 + BAND]. Primitives
must be current."
function count_outside_band(solver)
    n1, n2, n3 = solver.decomp.n_local
    n = 0
    for k in 1:n3, j in 1:n2, i in 1:n1
        I = gidx(solver, i, j, k)
        y = solver.Y[1][I]
        (y < -BAND || y > 1 + BAND) && (n += 1)
    end
    return MPI.Allreduce(n, +, solver.comm)
end

"Minimum mixture density over the interior."
function min_density(solver, Q)
    n1, n2, n3 = solver.decomp.n_local
    m = Inf
    for k in 1:n3, j in 1:n2, i in 1:n1
        m = min(m, mixture_density(solver, Q, gidx(solver, i, j, k)))
    end
    return MPI.Allreduce(m, min, solver.comm)
end

"""
The kinetic- and internal-energy rates of the artificial channels at the
current state, each from an exact difference of two right-hand sides.

`offs` and `alloffs` are `(solver, state, dQ)` triples for a solver with
`C_D = C_Y = 0` and one with every artificial constant zero. Writing the three
evaluations F, D and A, the species channel is F − D and the three viscous
channels together are D − A; mu*, beta* and kappa* are identical in F and D,
and the species coefficients are zero in both D and A, so each difference is
the divergence of its own fluxes alone (see the header note). All three
evaluations are collective.

The live solver's artificial coefficients and stage time are put back
afterwards: `max_rate` sizes the next step from those arrays, so an unbracketed
evaluation would perturb the run this is measuring.
"""
function channel_budget(solver, Q, offs, alloffs, dQ, buf)
    off, Qoff, dQoff = offs
    alloff, Qalloff, dQalloff = alloffs
    art_saved = CL.art_block(solver)
    tstage = solver.tstage
    compute_rhs!(solver, Q, dQ)
    for (s, Qs, dQs) in (offs, alloffs)
        copyto!(Qs, Q)
        s.tstage = tstage
        compute_rhs!(s, Qs, dQs)
    end
    m1, m2, m3 = solver.equations.i_mom
    i_energy = solver.equations.i_energy
    n_species = solver.equations.n_species
    Rspecies(I, c) = dQ[I, c] - dQoff[I, c]
    Rvisc(I, c) = dQoff[I, c] - dQalloff[I, c]
    # The primitives are current on the live solver: its own evaluation just
    # refreshed them from this Q. d(KE)/dt = u·R_mom − ½|u|² R_rho follows from
    # KE = ½|m|²/rho, and the internal-energy rate is the rest of ∫R_E.
    function rates(R)
        ke = integrate(solver, buf, I -> begin
            u, v, w = solver.u[I], solver.v[I], solver.w[I]
            Rrho = sum(R(I, sp) for sp in 1:n_species)
            u * R(I, m1) + v * R(I, m2) + w * R(I, m3) -
                0.5 * (u^2 + v^2 + w^2) * Rrho
        end)
        e = integrate(solver, buf, I -> R(I, i_energy))
        return ke, e - ke
    end
    ke_sp, internal_sp = rates(Rspecies)
    ke_visc, internal_visc = rates(Rvisc)
    # The two diffusivities the rates come from, read off the live solver's own
    # coefficients: D_art[1] is D_b under the bulk channel and D*_1 under the
    # Fickian one, and mu*/rho is the momentum diffusivity beside it.
    n1, n2, n3 = solver.decomp.n_local
    mom_max, rho_max, d_max, nu_max = 0.0, 0.0, 0.0, 0.0
    for k in 1:n3, j in 1:n2, i in 1:n1
        I = gidx(solver, i, j, k)
        mom_max = max(mom_max, abs(Rspecies(I, m1)), abs(Rspecies(I, m2)),
                      abs(Rspecies(I, m3)))
        rho_max = max(rho_max, abs(sum(Rspecies(I, sp) for sp in 1:n_species)))
        d_max = max(d_max, solver.D_art[1][I])
        nu_max = max(nu_max, solver.mu_art[I] / solver.rho[I])
    end
    v = MPI.Allreduce([mom_max, rho_max, d_max, nu_max], max, solver.comm)
    CL.set_art_block!(solver, art_saved)
    solver.tstage = tstage
    return (ke = ke_sp, internal = internal_sp, ke_visc = ke_visc,
            internal_visc = internal_visc, mom_max = v[1], rho_max = v[2],
            d_max = v[3], nu_max = v[4])
end

# --- the advected blob ------------------------------------------------------
#
# `brill_slab` in three dimensions: a sphere of the heavy gas in light gas at
# uniform p = 1, T = 1 and u, carried through the periodic unit cube for
# `periods` transits. The interface is the paper's tanh with 99% of its width
# in `Np` cells, so its physical thickness is Np/N and refining N sharpens it,
# where the one-dimensional case pins N = 20 Np and holds the thickness fixed.
# Both gases take gamma = `gamma_heavy` = 1.4 by default, the paper's choice;
# 1.09 turns the same run into the unequal-gamma contact.

function slab_case(channel, ratio)
    N = opt.N
    h = 1.0 / N
    w = 3 * opt.Np * h / 16
    u0 = opt.diagonal ? (SLAB_U, SLAB_U, SLAB_U) : (SLAB_U, 0.0, 0.0)
    unorm = sqrt(u0[1]^2 + u0[2]^2 + u0[3]^2)
    prob = Problem(eos = mixture(ratio, opt.gamma_heavy, "light"),
                   transport = Transport(mu0 = 0.0),
                   domain = ((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)), bcs = PER,
                   ic = (x, y, z) -> begin
                       r = sqrt((x - 0.5)^2 + (y - 0.5)^2 + (z - 0.5)^2)
                       V = (1 - tanh((r - SLAB_RADIUS) / w)) / 2
                       ρ = V * ratio + (1 - V)
                       Yh = V * ratio / ρ
                       Prim(Y = (1 - Yh, Yh), rho = ρ, u = u0, p = 1.0)
                   end)
    num = Numerics(n_global = (N, N, N), art = art_for(channel, 0.01, 100.0),
                   cfl = opt.cfl, filt = compact_filter(0.45),
                   filter_interval = 1, filter_cfl = 0.35,
                   control = StepControl(retries = 4, validity = :permissive))
    solver, Q = setup(prob, num)
    lo, hi = Ref(Inf), Ref(-Inf)
    steady_wall, steady_steps = 0.0, 0
    function diag(s, Q)
        scan_mass_fractions!(lo, hi, s, Q)
        if s.step > 20
            steady_wall += s.wall_step
            steady_steps += 1
        end
        return nothing
    end
    callbacks = opt.progress > 0 ?
        (diag, ProgressLog(every = opt.progress, tfinal = opt.periods / SLAB_U)) :
        (diag,)
    tfin = opt.periods / SLAB_U
    run!(solver, Q; tfinal = tfin, nmax = opt.nmax, callback = callbacks)
    refresh_primitives!(solver, Q)
    n1, n2, n3 = solver.decomp.n_local
    p_error, u_error = 0.0, 0.0
    for k in 1:n3, j in 1:n2, i in 1:n1
        I = gidx(solver, i, j, k)
        p_error = max(p_error, abs(solver.p[I] - 1))
        u_error = max(u_error, abs(solver.u[I] - u0[1]),
                      abs(solver.v[I] - u0[2]), abs(solver.w[I] - u0[3]))
    end
    v = MPI.Allreduce([p_error, u_error / unorm, -lo[], hi[]], max, solver.comm)
    return (p_error = v[1], u_error = v[2], worst_min = -v[3], worst_max = v[4],
            rho_min = min_density(solver, Q), steps = solver.step,
            ms_step = 1e3 * steady_wall / max(steady_steps, 1),
            completed = isfinite(solver.t) && solver.t >= tfin * (1 - 1e-9))
end

# --- the shocked bubble -----------------------------------------------------
#
# `shock_interface` in three dimensions: the same Mach 1.5 air shock and
# Rankine-Hugoniot post-shock state, running into a sphere of the heavy gas.
# Both x ends hold their initial state; the transverse directions are periodic.

function bubble_problem(ratio, nx, Ly)
    γa = 1.4
    M = BUBBLE_MACH
    p2 = 1 + 2γa / (γa + 1) * (M^2 - 1)
    r2 = (γa + 1) * M^2 / ((γa - 1) * M^2 + 2)
    u2 = M * sqrt(γa) * (1 - 1 / r2)
    h = 1.0 / (nx - 1)
    δ = opt.delta * h
    yc = Ly / 2
    bcs = (DirichletBC((x, y, z, t) -> Prim(Y = (1.0, 0.0), rho = r2,
                                            u = (u2, 0.0, 0.0), p = p2)),
           DirichletBC((x, y, z, t) -> Prim(Y = (1.0, 0.0), rho = 1.0,
                                            u = (0.0, 0.0, 0.0), p = 1.0)))
    prob = Problem(eos = mixture(ratio, BUBBLE_GAMMA_HEAVY, "air"),
                   transport = Transport(mu0 = 0.0),
                   domain = ((0.0, 1.0), (0.0, Ly), (0.0, Ly)),
                   bcs = (bcs, PER[2], PER[3]),
                   ic = (x, y, z) -> begin
                       r = sqrt((x - BUBBLE_X_CENTRE)^2 + (y - yc)^2 +
                                (z - yc)^2)
                       V = 1 - tanh_blend(r, BUBBLE_RADIUS, δ)  # 1 inside
                       ρa = V * ratio + (1 - V)                 # ahead of shock
                       Yh = V * ratio / ρa
                       s = tanh_blend(x, BUBBLE_X_SHOCK, δ)     # 0 post-shock
                       Prim(Y = (1 - Yh, Yh), rho = (1 - s) * r2 + s * ρa,
                            u = ((1 - s) * u2, 0.0, 0.0),
                            p = (1 - s) * p2 + s)
                   end)
    return prob
end

"""
One bubble run. `budget` builds the second solver and samples the species
channel's energy budget every `every` steps; otherwise the sample rows carry
the kinetic-energy history alone.
"""
function bubble_case(channel, ratio; nx = opt.nx, ny = opt.ny, C_D = 0.01,
                     C_Y = 100.0, budget::Bool = false)
    nz = ny
    prob = bubble_problem(ratio, nx, opt.Ly)
    control = StepControl(retries = 4, validity = :permissive)
    num = Numerics(n_global = (nx, ny, nz), art = art_for(channel, C_D, C_Y),
                   cfl = opt.cfl, filt = compact_filter(0.45),
                   filter_interval = 1, filter_cfl = 0.35, control = control)
    solver, Q = setup(prob, num)
    # The two comparison solvers carry the same problem, grid and process grid,
    # so their decompositions and boundary conditions match the live one point
    # for point; `off` differs in the species bracket alone and `alloff` in
    # every artificial constant. Both keep `enabled = true` with zeroed
    # constants rather than `enabled = false`, so that every sensor pass, line
    # solve and collective of `compute_artificial!` still runs and the two
    # evaluations differ in the coefficients and in nothing else.
    off = Qoff = alloff = Qalloff = dQ = dQoff = dQalloff = nothing
    if budget
        numerics(art) = Numerics(n_global = (nx, ny, nz), art = art,
                                 cfl = opt.cfl, filt = compact_filter(0.45),
                                 filter_interval = 1, filter_cfl = 0.35,
                                 control = control)
        off, Qoff = setup(prob, numerics(art_for(channel, 0.0, 0.0)))
        alloff, Qalloff = setup(prob, numerics(
            ArtParams(enabled = true, C_mu = 0.0, C_beta = 0.0, C_kappa = 0.0,
                      C_D = 0.0, C_Y = 0.0, species_flux = Symbol(channel))))
        dQ, dQoff, dQalloff = zero(Q), zero(Q), zero(Q)
    end
    buf = similar(solver.rho)
    fill!(buf, 0)
    e0 = total_energy_integral(solver, Q, buf)
    m0 = mass_integral(solver, Q, buf)
    lo, hi = Ref(Inf), Ref(-Inf)
    steady_wall, steady_steps = 0.0, 0
    ts, kes = Float64[], Float64[]
    rates = NamedTuple[]
    function diag(s, Q)
        scan_mass_fractions!(lo, hi, s, Q)
        if s.step > 20
            steady_wall += s.wall_step
            steady_steps += 1
        end
        # `step` is identical on every rank, so the collectives below are
        # entered by all of them or by none.
        s.step % opt.every == 0 || return nothing
        push!(ts, s.t)
        push!(kes, kinetic_energy(s, Q, buf))
        budget && push!(rates, channel_budget(s, Q, (off, Qoff, dQoff),
                                              (alloff, Qalloff, dQalloff),
                                              dQ, buf))
        return nothing
    end
    callbacks = opt.progress > 0 ?
        (diag, ProgressLog(every = opt.progress, tfinal = opt.tfin)) : (diag,)
    run!(solver, Q; tfinal = opt.tfin, nmax = opt.nmax, callback = callbacks)
    e1 = total_energy_integral(solver, Q, buf)
    m1 = mass_integral(solver, Q, buf)
    refresh_primitives!(solver, Q)
    mixed = mixedness(solver, buf)
    outside = count_outside_band(solver)
    ρmin = min_density(solver, Q)
    width = mix_width(solver, Q; dim = 1)
    theta = molecular_mixing(solver, Q; dim = 1)
    v = MPI.Allreduce([-lo[], hi[]], max, solver.comm)
    return (worst_min = -v[1], worst_max = v[2], outside = outside,
            mixedness = mixed, mix_width = width, theta = theta,
            rho_min = ρmin, energy_drift = (e1 - e0) / abs(e0),
            mass_drift = (m1 - m0) / abs(m0), ts = ts, kes = kes, rates = rates,
            steps = solver.step,
            ms_step = 1e3 * steady_wall / max(steady_steps, 1),
            completed = isfinite(solver.t) && solver.t >= opt.tfin * (1 - 1e-9))
end

# --- reporting --------------------------------------------------------------

function print_bubble_row(label, r)
    @printf("%-27s %8.4f %8.4f %7d %8.4f %10.3e %8.4f %9.3e",
            label, r.worst_min, r.worst_max, r.outside, r.rho_min, r.mixedness,
            r.theta, r.mix_width)
    @printf(" %9.2e %9.2e %6d %7.1f %s\n", r.energy_drift, r.mass_drift,
            r.steps, r.ms_step, r.completed ? "" : "INCOMPLETE")
end

function print_bubble_header()
    @printf("%-27s %8s %8s %7s %8s %10s %8s %9s", "case", "minY", "maxY",
            "outside", "rho_min", "mixedness", "theta", "W")
    @printf(" %9s %9s %6s %7s\n", "dE/E", "dM/M", "steps", "ms/step")
end

"A few evenly spaced instants of the kinetic-energy history."
function print_ke_history(r, rows = 8)
    isempty(r.ts) && return nothing
    n = length(r.ts)
    idx = unique(round.(Int, range(1, n; length = min(rows, n))))
    print("    KE(t):")
    for i in idx
        @printf("  %.3f:%.5e", r.ts[i], r.kes[i])
    end
    println()
    return nothing
end

function print_budget(r)
    println("       t   chan dKE/dt  visc dKE/dt   chan/(sum)   chan dU/dt" *
            "  visc dU/dt       max D  max mu*/rho   max|R_mom|  max|R_rho|")
    for i in eachindex(r.rates)
        b = r.rates[i]
        # The fraction of the artificial kinetic-energy sink the species channel
        # carries. Both rates are signed, so the denominator is their sum and
        # not the sum of magnitudes.
        frac = b.ke / (b.ke + b.ke_visc + 1e-300)
        @printf("  %6.4f %12.4e %12.4e %12.4f %12.4e %11.4e",
                r.ts[i], b.ke, b.ke_visc, frac, b.internal, b.internal_visc)
        @printf(" %11.3e %12.3e %12.3e %11.3e\n", b.d_max, b.nu_max,
                b.mom_max, b.rho_max)
    end
    return nothing
end

# --- parts ------------------------------------------------------------------

function run_slab(channels, ratios, rank)
    if rank == 0
        @printf("\n=== slab: %d^3, %.3g periods, Np = %d cells, ",
                opt.N, opt.periods, opt.Np)
        @printf("gamma_heavy = %.3g, %s advection\n", opt.gamma_heavy,
                opt.diagonal ? "diagonal" : "axis")
    end
    rank == 0 && @printf("%-22s %10s %10s %9s %9s %9s %6s %7s\n",
                         "case", "max|p-1|", "max du/u", "minY", "maxY",
                         "rho_min", "steps", "ms/step")
    for channel in channels, ratio in ratios
        r = slab_case(channel, ratio)
        rank == 0 || continue
        @printf("%-22s %10.3e %10.3e %9.4f %9.4f %9.4f %6d %7.1f %s\n",
                @sprintf("%s, R = %.4g", channel, ratio), r.p_error, r.u_error,
                r.worst_min, r.worst_max, r.rho_min, r.steps, r.ms_step,
                r.completed ? "" : "INCOMPLETE")
    end
    return nothing
end

function run_bubble(channels, ratios, rank)
    if rank == 0
        @printf("\n=== bubble: %d x %d x %d, Ly = %.3g, ", opt.nx, opt.ny,
                opt.ny, opt.Ly)
        @printf("delta = %.3g cells, t = %.3g\n", opt.delta, opt.tfin)
    end
    rank == 0 && print_bubble_header()
    for channel in channels, ratio in ratios
        r = bubble_case(channel, ratio)
        rank == 0 || continue
        print_bubble_row(@sprintf("%s, R = %.4g", channel, ratio), r)
        print_ke_history(r)
    end
    return nothing
end

function run_budget(channels, ratios, rank)
    if rank == 0
        @printf("\n=== budget: bubble %d x %d x %d, ", opt.nx, opt.ny, opt.ny)
        @printf("sampled every %d steps\n", opt.every)
    end
    for channel in channels, ratio in ratios
        r = bubble_case(channel, ratio; budget = true)
        rank == 0 || continue
        @printf("\n--- %s, R = %.4g: %d steps, %.1f ms/step\n",
                channel, ratio, r.steps, r.ms_step)
        print_budget(r)
    end
    return nothing
end

function run_constants(ratios, rank)
    C_Ds = parse_floats(opt.C_Ds, :C_Ds)
    C_Ys = parse_floats(opt.C_Ys, :C_Ys)
    rank == 0 && @printf("\n=== constants: bulk channel, bubble %d x %d x %d\n",
                         opt.nxc, opt.nyc, opt.nyc)
    rank == 0 && print_bubble_header()
    for ratio in ratios, C_D in C_Ds, C_Y in C_Ys
        r = bubble_case("bulk", ratio; nx = opt.nxc, ny = opt.nyc,
                        C_D = C_D, C_Y = C_Y)
        rank == 0 || continue
        print_bubble_row(@sprintf("R %.4g, C_D %.3g, C_Y %g", ratio, C_D, C_Y), r)
    end
    return nothing
end

function main()
    rank = MPI.Comm_rank(MPI.COMM_WORLD)
    parts = parse_names(opt.parts, :parts,
                        ("slab", "bubble", "budget", "constants"))
    channels = parse_names(opt.channels, :channels, ("fickian", "bulk"))
    ratios = parse_floats(opt.ratio, :ratio)
    if rank == 0
        @printf("=== bulk species channel, %d rank(s), %d thread(s); ",
                MPI.Comm_size(MPI.COMM_WORLD), Threads.nthreads())
        @printf("parts %s, channels %s, ratios %s\n", join(parts, ","),
                join(channels, ","), join(ratios, ","))
    end
    for part in parts
        part == "slab" && run_slab(channels, ratios, rank)
        part == "bubble" && run_bubble(channels, ratios, rank)
        part == "budget" && run_budget(channels, ratios, rank)
        part == "constants" && run_constants(ratios, rank)
    end
    return nothing
end

mpi_main(main)
