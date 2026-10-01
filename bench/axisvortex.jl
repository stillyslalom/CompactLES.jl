# An isentropic vortex carried across the axis of a resolved (r, θ) grid, with
# and without azimuthal mode truncation and the artificial properties: whether
# each run completes, and its error against the translated vortex.
#
#   julia --project=. -t 1 bench/axisvortex.jl
#   julia --project=. -t 1 bench/axisvortex.jl configs=1:on:0.5,1:off:0.5 tfinal=0.4
#   julia --project=. -t 1 bench/axisvortex.jl N=32,64 configs=0:on:0.5
#
# --- Why this exists ---------------------------------------------------------
#
# The vortex is the one of the Axis-crossing vortex tutorial
# (docs/literate/axis_crossing_vortex.jl): Shu's isentropic vortex, β = 3,
# core radius 0.15, carried by a stream U = 0.5 from x = −0.4 through the axis
# at t = 0.8 to x = 0.4 at t = 1.6 on the unit disk, with a DirichletBC at
# r = 1 holding the exact translation. The solution at every time is known, so
# the error of a truncated run separates into what the projection deletes and
# what the scheme adds. Each entry of `configs` is `kappa:art:cfl`: the
# `polar_truncation` margin (0 for none), the artificial properties `on` (the
# defaults) or `off`, and the CFL number. Every run ends at `tfinal` or at a
# `SolverFailure`, whichever is first, and prints where it ended, its step
# count and mean step, and the largest |ρ − exact| and |u − exact| over the
# run, on the whole disk and on the first `inner` rings, with the same two at
# t = 0.4, 0.8 and at the end. The errors are sampled every 0.05 time units.
# The last columns are the run's wall time, the wall time per step (error
# sampling included), and the median of 200 `truncate_modes!` calls on the
# final state.
#
# `-t 1` and one process for every configuration: a truncated run at 48 × 96
# takes seconds, an untruncated one to t = 1.6 about a minute at CFL 0.5.
# The measurements are in reference/CALIBRATION_APPENDIX.md under
# "Azimuthal mode truncation".

using CompactLES
const CL = CompactLES
using Printf

const OPTS = CL.script_args(ARGS, (N="48,96", configs="0:on:0.5,0:off:0.5,1:on:0.3," *
                                   "1:on:0.5,1:off:0.5,2:on:0.5,2:off:0.5",
                                   tfinal=1.6, inner=2, nmax=20_000))

const GAMMA, BETA, RC, U, X0 = 1.4, 3.0, 0.15, 0.5, -0.4

function vortex(x, y, t)
    xi, eta = (x - X0 - U * t) / RC, y / RC
    f = exp(1 - xi^2 - eta^2)
    swirl = BETA / (2pi) * sqrt(f)
    T = 1 - (GAMMA - 1) * BETA^2 / (8GAMMA * pi^2) * f
    rho = T^(1 / (GAMMA - 1))
    return (; rho, p=rho * T, ux=U - swirl * eta, uy=swirl * xi)
end

function exact(r, theta, t)
    v = vortex(r * cos(theta), r * sin(theta), t)
    s, c = sincos(theta)
    return Prim(rho=v.rho, p=v.p, u=(c * v.ux + s * v.uy, -s * v.ux + c * v.uy, 0.0))
end

# The largest density and velocity differences on the disk and on the first
# `inner` rings.
function errors(solver, Q, inner)
    snap = field_snapshot(solver, Q; fields=(:rho, :u, :v))
    r, theta = snap.coords[1], snap.coords[2]
    e = zeros(4)
    for j in eachindex(theta), i in eachindex(r)
        x = exact(r[i], theta[j], solver.t)
        dr = abs(snap[:rho][i, j, 1] - x.rho)
        du = hypot(snap[:u][i, j, 1] - x.u[1], snap[:v][i, j, 1] - x.u[2])
        e[1] = max(e[1], dr); e[2] = max(e[2], du)
        if i <= inner
            e[3] = max(e[3], dr); e[4] = max(e[4], du)
        end
    end
    return e
end

function run_case(n, kappa, art, cfl, o)
    problem = Problem(
        name="vortex across the axis",
        eos=IdealSpecies("gas"; R=1.0, gamma=GAMMA),
        metric=CylindricalMetric(),
        domain=((0.0, 1.0), (0.0, 2pi), (0.0, 1.0)),
        bcs=((AxisBC(), DirichletBC((r, theta, z, t) -> exact(r, theta, t))),
             PeriodicBC(), PeriodicBC()),
        ic=(r, theta, z) -> exact(r, theta, 0.0))
    numerics = Numerics(n_global=(n[1], n[2], 1), cfl=cfl, polar_truncation=kappa,
                        art=ArtificialProperties(enabled=art))
    solver, Q = setup(problem, numerics)
    history = Tuple{Float64,Vector{Float64}}[]
    record = Callback(EveryTime(0.05), (s, Q) -> (push!(history,
                                                        (s.t, errors(s, Q, o.inner)));
                                                  nothing))
    status = "done"
    wall = @elapsed try
        run!(solver, Q; tfinal=o.tfinal, nmax=o.nmax, callback=record)
    catch err
        err isa SolverFailure || rethrow()
        status = "failed"
    end
    solver.t < o.tfinal * (1 - 1e-9) && status == "done" && (status = "nmax")
    proj = kappa > 0 ? sort([@elapsed(CL.truncate_modes!(solver, Q)) for _ in 1:200])[100] :
           NaN
    worst = isempty(history) ? fill(NaN, 4) : reduce((a, b) -> max.(a, b), last.(history))
    at(t) = (k = findfirst(h -> isapprox(h[1], t; atol=1e-9), history);
             k === nothing ? NaN : history[k][2][1])
    @printf("%5.1f %4s %5.2f  %-6s %6.3f %6d %9.2e  %9.2e %9.2e  %9.2e %9.2e  " *
            "%9.2e %9.2e %9.2e %6.1f %7.2f %7.1f\n",
            kappa, art ? "on" : "off", cfl, status, solver.t, solver.step,
            solver.t / max(solver.step, 1), worst[1], worst[2], worst[3], worst[4],
            at(0.4), at(0.8), isempty(history) ? NaN : history[end][2][1], wall,
            1e3 * wall / max(solver.step, 1), 1e6 * proj)
    flush(stdout)
end

function main(o)
    n = parse.(Int, split(o.N, ','))
    @printf("N = %d × %d, tfinal = %.2f, inner rings %d\n", n[1], n[2], o.tfinal, o.inner)
    println("kappa  art   cfl  status      t  steps   mean dt   max ρ err max u err" *
            "  inner ρ   inner u    ρ t=0.4   ρ t=0.8   ρ end    wall ms/step proj µs")
    for c in split(o.configs, ',')
        k, a, f = split(c, ':')
        a in ("on", "off") || error("art must be on or off, got $a")
        run_case(n, parse(Float64, k), a == "on", parse(Float64, f), o)
    end
end

main(OPTS)
