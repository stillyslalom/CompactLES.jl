# Step-resolved state probe for Noh, built to test one recorded hypothesis about
# the `cfl <= 0.15` ceiling: that beta* does not reach far enough AHEAD of the
# front to damp the dispersive undershoot that precedes the positivity loss.
#
# --- What this measured, so it is not rediscovered ---------------------------
#
# The hypothesis is wrong on three independent readings, and the write-up is
# in reference/CALIBRATION.md under "Where the restriction originates".
#
#   1. Reach is not short. beta* carries above a thousandth of its own maximum
#      out to 14.2-14.8 cells ahead of the front, held steady over a complete
#      nu = 1 run at the default cfl = 0.15, while the damage sits 3-5 cells
#      ahead. That is a margin of 3-5x rather than a deficit.
#
#   2. beta* is present where the damage is, at 14-21% of its maximum, and the
#      damage persists regardless. A failure occurring under a sixth of the peak
#      regularization is not explained by absent regularization.
#
#   3. Above the ceiling the run does not fail ahead of the front at all. At
#      nu = 1, cfl = 0.3 the first cell to degrade is the SYMMETRY cell i = 1,
#      within five steps, while the pre-shock field is still within 1% of unity.
#      The density hole that CALIBRATION.md recorded as a pre-shock undershoot
#      is at i = 3, between the wall and the front rather than ahead of it.
#
#   4. The symmetry cell fails through a STARTUP EXCURSION, not a gradual
#      degradation. Added later with the rho1/rho2, e1/e0 and b*1/max columns:
#      at nu = 3 the origin holds rho1/rho2 within 0.2% of unity for eighty-odd
#      steps carrying beta* at a thousandth of the line maximum, then goes
#      through one large excursion during which beta* there reaches the line
#      MAXIMUM. Surviving configurations pass through it; failing ones evacuate
#      the cell inside a single sampling interval. The excursion lands at
#      t ~ 0.394 at N = 128, 256 and 512 while its step number scales with N, so
#      it is a resolved feature of the warm start rather than a grid artifact.
#      This retired the two standing explanations at once: the fold closure is
#      sixth to seventh order (see bench/foldorder.jl), and the sensor is not
#      blind at the fold. Write-up under "The origin cell is a startup
#      transient".
#
# The probe also surfaced a condition no diagnostic then reported: runs that
# COMPLETE carry 6-8 cells of negative internal energy, travelling with the
# front, for their whole duration. `primitives!` floors T_ion at 1e-300 wherever
# e <= 0, so p goes to rho*R*1e-300 and the run continues, while the positivity
# check in `max_rate` guards rho, which stays positive throughout. See the
# `n_e<0` column.
#
# That condition is now measured rather than merely visible, and the positivity
# failsafe is the instrument. `floor=1e-8` counts it over a whole run without
# changing the trajectory, while `floor=1e-8 scope=internal_energy` repairs it,
# which on nu=1 at cfl 0.15 costs a 5% velocity damping on the worst cell and
# fails at step 18. The write-up is in reference/CALIBRATION.md under "The
# negative internal energy is not a rounding artifact". The converging
# geometries have not been measured through it; only the planar case has.
#
# Columns, one line per sampled step:
#
#   x_sh/h       exact shock position in cells, (gamma-1)/2 * t / h
#   rho_min i    smallest density over the line, and where
#   e/e0_min i   smallest internal energy as a multiple of the ambient
#                p0/(gamma-1), and where. Recovered from Q, not from T_ion,
#                which is floored where it goes negative.
#   n_e<0        interior cells carrying negative internal energy
#   b*@e/b*max   beta* at the worst-e cell, relative to its own maximum: the
#                direct test of whether regularization is present there
#   reach/h      furthest cell ahead of the front carrying beta* above a
#                thousandth of the maximum
#   rho1/rho2    symmetry cell density over its neighbour's. 1 is a healthy
#                converging profile; falling well below 1 is the symmetry cell
#                evacuating. Because beta* is proportional to rho, a thinning
#                cell suppresses its own regularization
#   e1/e0        internal energy at the symmetry cell, ambient multiples
#   b*1/max      beta* at the symmetry cell over the line maximum
#
# The last three are reported every line whether or not the symmetry cell is the
# worst one. The argmin columns track the front through most of a run, so a
# symmetry cell degrading underneath them stays invisible until it overtakes the
# front, by which point the run is a few steps from losing positivity.
#
# Usage: positional nu, then `key=value`.
#
#   julia --project=. bench/nohprobe.jl 1                    # planar, cfl 0.3
#   julia --project=. bench/nohprobe.jl 1 cfl=0.15 nmax=5000 every=500
#   julia --project=. bench/nohprobe.jl 2 cfl=0.2 sensor=gated_strain
#   julia --project=. bench/nohprobe.jl 3 cfl=0.3 detector=d8
#   julia --project=. bench/nohprobe.jl 1 cfl=0.3 dump=125    # full profile
#
# Scratch tooling, like everything else in bench/: it prints tables and asserts
# nothing. It runs serially because the cases are one-dimensional and a
# threaded region here falls far under THREAD_MIN_WORK, so `-t auto` buys
# nothing.

using MPI
MPI.Init(threadlevel=:funneled)
using CompactLES
using Printf

const CL = CompactLES
include(joinpath(@__DIR__, "..", "test", "references.jl"))
include(joinpath(@__DIR__, "..", "test", "cases.jl"))

# Sensor-shape defaults track `ArtParams()`, so a bare run probes the current
# defaults rather than whatever they were when this line was written.
const ART_DEFAULTS = ArtParams()
opt = CompactLES.script_args(ARGS, (nu = 1, cfl = 0.3, N = 0, nmax = 400, every = 25,
                         sensor = string(ART_DEFAULTS.beta_sensor),
                         smoother = string(ART_DEFAULTS.smoother),
                         detector = string(ART_DEFAULTS.detector),
                         dump = -1, profile = 32,
                         floor = 0.0, scope = "representable");
                  positional = (:nu,))

const ν = opt.nu
1 <= ν <= 3 || error("nu must be 1, 2 or 3")
const N = opt.N > 0 ? opt.N : Dict(NOH_N)[ν]
const T0 = Dict(NOH_T0)[ν]
const E0 = NOH_P0 / (NOH_G - 1)          # ambient internal energy

"""
    make_probe(solver, xs, h) -> effect!

The callback effect that prints one sampled line of the table. It reads
`beta_art` and `sensor` as the last RHS evaluation left them, so the reported
values are the ones the run stepped on.
"""
function make_probe(solver, xs, h)
    n = length(xs)
    o1, o2, o3 = solver.decomp.n_halo_d
    i_energy = solver.equations.i_energy
    grab(f) = [f[gidx(solver, i, 1, 1)] for i in 1:n]
    return function (solver, Q)
        rho = grab(solver.rho); u = grab(solver.u); b = grab(solver.beta_art)
        # From Q rather than T_ion: primitives! floors T_ion at 1e-300 wherever
        # the internal energy is negative, which is precisely what is wanted here.
        e = [Q[i + o1, 1 + o2, 1 + o3, i_energy] / rho[i] - 0.5 * u[i]^2
             for i in 1:n]
        t = solver.t + T0
        xsh = (NOH_G - 1) / 2 * t
        im = argmin(rho); ie = argmin(e)
        bmax = maximum(b)
        ahead = [i for i in 1:n if xs[i] > xsh && b[i] > 1e-3 * bmax]
        reach = isempty(ahead) ? 0.0 : (xs[maximum(ahead)] - xsh) / h
        # The symmetry cell reported unconditionally, not only when it happens to
        # be the worst. The argmin columns track the front for most of a run, so
        # a symmetry cell degrading underneath is invisible in them until it
        # overtakes the front, which is the question the ceiling poses.
        # rho_1/rho_2 is the evacuation measure: the cell thinning relative to
        # its neighbour suppresses its own beta*, which is proportional to rho.
        # Two calls because @printf needs a literal format string, and one
        # literal covering all thirteen columns runs past the line limit.
        @printf("%6d %8.5f %7.2f | %8.5f %5d | %10.1f %5d %6d | %10.3f %7.2f",
                solver.step, t, xsh / h + 1, rho[im], im,
                e[ie] / E0, ie, count(<=(0), e), b[ie] / bmax, reach)
        @printf(" | %8.4f %10.1f %8.3f\n",
                rho[1] / rho[2], e[1] / E0, b[1] / bmax)
        if solver.step == opt.dump
            s = grab(solver.sensor)
            println("\n  --- step $(solver.step), front at cell " *
                    "$(round(xsh / h + 1, digits = 2)) ---")
            @printf("  %4s %9s %10s %10s %11s %10s %10s\n",
                    "i", "x", "rho", "u", "e/e0", "sensor", "beta*")
            for i in 1:min(n, opt.profile)
                @printf("  %4d %9.5f %10.5f %10.5f %11.3e %10.3e %10.3e\n",
                        i, xs[i], rho[i], u[i], e[i] / E0, s[i], b[i])
            end
            println()
        end
        return false
    end
end

function main()
    art = ArtParams(beta_sensor = Symbol(opt.sensor),
                smoother = Symbol(opt.smoother),
                detector = Symbol(opt.detector))
    metric = ν == 1 ? CartesianMetric() :
             ν == 2 ? CylindricalMetric() : SphericalMetric()
    lobc = ν == 1 ? SlipWallBC() : ν == 2 ? AxisBC() : OriginBC()
    dom2 = ν == 1 ? (0.0, 1.0 / N) : ν == 2 ? (0.0, 1.0) : (π / 2, π / 2 + 1)
    dom3 = ν == 1 ? (0.0, 1.0 / N) : (0.0, 1.0)
    # The same setup test/cases.jl builds, reproduced rather than called because
    # noh_case takes no callback and the whole point here is per-step sampling.
    inflow = DirichletBC((x, y, z, t) -> begin
        ρ, uu, _ = noh_exact(x, isfinite(t) ? t + T0 : T0, ν, NOH_G)
        Prim(rho = ρ, u = (uu, 0.0, 0.0), p = NOH_P0)
    end)
    w = 4 / N
    ic = (x, y, z) -> begin
        T0 <= 0 && return Prim(rho = 1.0, u = (-1.0, 0.0, 0.0), p = NOH_P0)
        ρin, _, pin = noh_exact(0.0, T0, ν, NOH_G)
        ρout, _, _ = noh_exact(x, T0, ν, NOH_G)
        θ = tanh_blend(x, (NOH_G - 1) / 2 * T0, w)
        Prim(rho = (1 - θ) * ρin + θ * ρout, u = (-θ, 0.0, 0.0),
             p = (1 - θ) * pin + θ * NOH_P0)
    end
    prob = Problem(eos = IdealSpecies("gas"; gamma = NOH_G, R = 1.0),
                   transport = Transport(mu0 = 0.0), metric = metric,
                   domain = ((0.0, 1.0), dom2, dom3),
                   bcs = ((lobc, inflow), per3[2], per3[3]), ic = ic)
    control = StepControl(floor_ratio = opt.floor, floor_scope = Symbol(opt.scope))
    solver, Q = setup(prob, Numerics(n_global = (N, 1, 1), art = art,
                                     cfl = opt.cfl, control = control,
                                     filter_interval = 1))

    xs = Float64[xcoord(solver, 1, i) for i in 1:N]
    h = xs[2] - xs[1]
    println("nu = $ν, N = $N, cfl = $(opt.cfl), t0 = $T0, " *
            "beta_sensor = $(opt.sensor), smoother = $(opt.smoother), " *
            "detector = $(opt.detector)")
    @printf("%6s %8s %7s | %8s %5s | %10s %5s %6s | %10s %7s",
            "step", "t", "x_sh/h", "rho_min", "i", "e/e0_min", "i", "n_e<0",
            "b*@e/b*max", "reach/h")
    @printf(" | %8s %10s %8s\n", "rho1/rho2", "e1/e0", "b*1/max")
    println(repeat("-", 128))

    cb = Callback(EveryStep(opt.every), make_probe(solver, xs, h))
    try
        run!(solver, Q; tfinal = NOH_T - T0, nmax = opt.nmax, callback = cb)
        @printf("  reached t = %.5f after %d steps\n", solver.t + T0, solver.step)
    catch err
        err isa SolverFailure || rethrow()
        println("  FAILED: $(err.reason) at step $(solver.step)")
    end
end

main()
