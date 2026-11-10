# An isentropic vortex carried across the axis of a resolved (r, θ) grid, with
# and without azimuthal mode truncation, the artificial properties and the
# filter, in the package's form of the radial pressure term or in ∂p/∂r, and
# the same vortex on two grids without an axis: whether each run completes, and
# its error against the translated vortex.
#
#   julia --project=. -t 1 bench/axisvortex.jl
#   julia --project=. -t 1 bench/axisvortex.jl configs=1:on:0.5,1:off:0.5 tfinal=0.4
#   julia --project=. -t 1 bench/axisvortex.jl N=32,64 configs=0:on:0.5
#   julia --project=. -t 1 bench/axisvortex.jl N=32,64 configs=0:on:0.5 \
#       variant=none,gradp filter=on,off history=true
#   julia --project=. -t 1 bench/axisvortex.jl geometry=box,annulus configs=0:on:0.5 \
#       filter=off,on history=true
#
# --- Why this exists ---------------------------------------------------------
#
# The vortex is the one of the Axis-crossing vortex tutorial
# (docs/literate/axis_crossing_vortex.jl): Shu's isentropic vortex, β = 3,
# core radius 0.15, carried by a stream U = 0.5 from x = −0.4 through the axis
# at t = 0.8 to x = 0.4 at t = 1.6 on the unit disk, with a DirichletBC at
# r = 1 holding the exact translation. The solution at every time is known, so
# the error of a run separates into what the scheme adds and, under
# `polar_truncation`, what the projection deletes.
#
# Each entry of `configs` is `kappa:art:cfl`: the `polar_truncation` margin
# (0 for none), the artificial properties `on` (the defaults) or `off`, and
# the CFL number. The lists `geometry`, `variant` and `filter` multiply it:
#   geometry  `disk`, the tutorial's grid on N = (n_r, n_θ) nodes; `box`, the
#             periodic square [−1, 1]² on n_box² nodes, whose spacing 2/n_box
#             matches the disk's Δr at the default; `annulus`, r ∈ [0.15, 1.5]
#             on n_annulus = (n_r, n_θ) nodes with a DirichletBC at both radii
#             and the vortex carried along y = 0.75, so that it passes the axis
#             no closer than 0.75 and its core sees azimuthal spacing close to
#             the disk's at r = 0.3 (the two controls without an axis)
#   variant   `none`, the package's radial pressure term, or `gradp`, ∂p/∂r on
#             the resolved θ through the source of bench/axisvariant.jl; the
#             two controls run `none` only
#   filter    `on`, the default state filter, or `off`
# Every run ends at `tfinal`, at a `SolverFailure` or at `nmax` steps, whichever
# is first (a run that grows without bound collapses its step and reaches
# `nmax`), and prints where it ended, its step count and mean step, and the
# largest |ρ − exact| and |u − exact| over the run, on the whole grid and
# within `inner` spacings of the point the vortex center crosses at t = 0.8
# (the first `inner` rings on the disk), with the whole-grid density error at
# t = 0.4, 0.8 and at the end. The errors are sampled every 0.05 time units,
# and `history=true` prints them. The last columns are the run's wall time,
# the wall time per step (error sampling included), and the median of 200
# `truncate_modes!` calls on the final state.
#
# `-t 1` and one process for every configuration: a truncated run at 48 × 96
# takes seconds, an untruncated one to t = 1.6 about a minute at CFL 0.5, and
# an unfiltered run on the disk that reaches `nmax` = 20000 under a minute.
# The measurements are in reference/CALIBRATION_APPENDIX.md under
# "Azimuthal mode truncation" and "Grid-scale growth at the r-z axis";
# bench/results/axisvortex.txt is the transcript of the second.

using CompactLES
using CompactLES: padded_index
const CL = CompactLES
using Printf

include(joinpath(@__DIR__, "axisvariant.jl"))

const OPTS = CL.script_args(ARGS, (N="48,96", configs="0:on:0.5,0:off:0.5,1:on:0.3," *
                                   "1:on:0.5,1:off:0.5,2:on:0.5,2:off:0.5",
                                   tfinal=1.6, inner=2, nmax=20_000, geometry="disk",
                                   variant="none", filter="on", history=false,
                                   n_box=64, n_annulus="44,160"))

const GAMMA, BETA, RC, U, X0 = 1.4, 3.0, 0.15, 0.5, -0.4
const Y_ANNULUS = 0.75

function vortex(x, y, t, y0)
    xi, eta = (x - X0 - U * t) / RC, (y - y0) / RC
    f = exp(1 - xi^2 - eta^2)
    swirl = BETA / (2pi) * sqrt(f)
    T = 1 - (GAMMA - 1) * BETA^2 / (8GAMMA * pi^2) * f
    rho = T^(1 / (GAMMA - 1))
    return (; rho, p=rho * T, ux=U - swirl * eta, uy=swirl * xi)
end

# The exact state in the grid's velocity components: polar on the disk and the
# annulus, Cartesian in the box.
function exact(a, b, t, polar, y0)
    polar || (v = vortex(a, b, t, y0); return Prim(rho=v.rho, p=v.p, u=(v.ux, v.uy, 0.0)))
    v = vortex(a * cos(b), a * sin(b), t, y0)
    s, c = sincos(b)
    return Prim(rho=v.rho, p=v.p, u=(c * v.ux + s * v.uy, -s * v.ux + c * v.uy, 0.0))
end

# The largest density and velocity differences on the grid and within
# `reach` of (0, y0), where the vortex center passes at t = 0.8.
function errors(solver, Q, polar, y0, reach)
    snap = field_snapshot(solver, Q; fields=(:rho, :u, :v))
    a, b = snap.coords[1], snap.coords[2]
    e = zeros(4)
    for j in eachindex(b), i in eachindex(a)
        x = exact(a[i], b[j], solver.t, polar, y0)
        dr = abs(snap[:rho][i, j, 1] - x.rho)
        du = hypot(snap[:u][i, j, 1] - x.u[1], snap[:v][i, j, 1] - x.u[2])
        e[1] = max(e[1], dr); e[2] = max(e[2], du)
        X, Y = polar ? (a[i] * cos(b[j]), a[i] * sin(b[j])) : (a[i], b[j])
        if hypot(X, Y - y0) < reach
            e[3] = max(e[3], dr); e[4] = max(e[4], du)
        end
    end
    return e
end

function build(geometry, n, kappa, art, cfl, variant, filt, o)
    polar = geometry != "box"
    y0 = geometry == "annulus" ? Y_ANNULUS : 0.0
    variant === :none || polar ||
        error("variant $variant is written for the radial pressure term")
    kappa > 0 && geometry != "disk" && error("polar_truncation is run on the disk only")
    held = DirichletBC((a, b, z, t) -> exact(a, b, t, polar, y0))
    per = (PeriodicBC(), PeriodicBC())
    domain, bcs, grid, metric = if geometry == "disk"
        ((0.0, 1.0), (0.0, 2pi), (0.0, 1.0)), ((AxisBC(), held), per, per),
            (n[1], n[2], 1), CylindricalMetric()
    elseif geometry == "annulus"
        m = parse.(Int, split(o.n_annulus, ','))
        ((0.15, 1.5), (0.0, 2pi), (0.0, 1.0)), ((held, held), per, per),
            (m[1], m[2], 1), CylindricalMetric()
    elseif geometry == "box"
        ((-1.0, 1.0), (-1.0, 1.0), (0.0, 1.0)), (per, per, per),
            (o.n_box, o.n_box, 1), CartesianMetric()
    else
        error("geometry must be disk, annulus or box, got $geometry")
    end
    problem = Problem(name="vortex", eos=IdealSpecies("gas"; R=1.0, gamma=GAMMA),
                      metric=metric, domain=domain, bcs=bcs,
                      ic=(a, b, z) -> exact(a, b, 0.0, polar, y0),
                      sources=variant === :none ? () : (AxisVariant(variant),))
    numerics = Numerics(n_global=grid, cfl=cfl, polar_truncation=kappa,
                        art=ArtificialProperties(enabled=art),
                        filter=filt ? StateFilter() : nothing)
    solver, Q = setup(problem, numerics)
    # The disk's first `inner` rings: r < inner·Δr, the same reach elsewhere.
    reach = o.inner * (geometry == "box" ? solver.h[1] : 1 / (n[1] - 0.5))
    return solver, Q, polar, y0, reach
end

function run_case(geometry, n, kappa, art, cfl, variant, filt, o)
    solver, Q, polar, y0, reach = build(geometry, n, kappa, art, cfl, variant, filt, o)
    history = Tuple{Float64,Vector{Float64}}[]
    record = Callback(EveryTime(0.05), (s, Q) -> (push!(history,
                                                        (s.t, errors(s, Q, polar, y0,
                                                                     reach)));
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
    @printf("%-7s %-5s %-3s %5.1f %4s %5.2f  %-6s %6.3f %6d %9.2e  %9.2e %9.2e  \
             %9.2e %9.2e  %9.2e %9.2e %9.2e %6.1f %7.2f %7.1f\n",
            geometry, variant, filt ? "on" : "off", kappa, art ? "on" : "off", cfl,
            status, solver.t, solver.step, solver.t / max(solver.step, 1), worst[1],
            worst[2], worst[3], worst[4], at(0.4), at(0.8),
            isempty(history) ? NaN : history[end][2][1], wall,
            1e3 * wall / max(solver.step, 1), 1e6 * proj)
    if o.history
        println("      t   max ρ err  max u err   near ρ     near u")
        for (t, e) in history
            @printf("   %5.2f  %9.3e  %9.3e  %9.3e  %9.3e\n", t, e...)
        end
    end
    flush(stdout)
end

function main(o)
    n = parse.(Int, split(o.N, ','))
    @printf("disk N = %d × %d, box %d², annulus %s, tfinal = %.2f, near reach %d spacings\n",
            n[1], n[2], o.n_box, o.n_annulus, o.tfinal, o.inner)
    println("geom    var   flt kappa  art   cfl  status      t  steps   mean dt" *
            "  max ρ err max u err   near ρ    near u    ρ t=0.4   ρ t=0.8   ρ end" *
            "    wall ms/step proj µs")
    for geometry in split(o.geometry, ','), f in split(o.filter, ','),
        v in split(o.variant, ','), c in split(o.configs, ',')
        f in ("on", "off") || error("filter must be on or off, got $f")
        k, a, cfl = split(c, ':')
        a in ("on", "off") || error("art must be on or off, got $a")
        geometry != "disk" && v != "none" && continue
        run_case(String(geometry), n, parse(Float64, k), a == "on", parse(Float64, cfl),
                 Symbol(v), f == "on", o)
    end
end

main(OPTS)
