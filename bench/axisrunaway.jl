# Whether grid-scale content grows at the r-z axis of an unrefined run, and
# when, against the node count, the filter's αf and the step.
#
#   julia --project=. -t 1 bench/axisrunaway.jl case=pulse cfl=0.177 N=2303,6908
#   julia --project=. -t 1 bench/axisrunaway.jl case=noh N=767 alpha=0.45,0.47
#   julia --project=. -t 1 bench/axisrunaway.jl case=pulse N=6908 cfl=0.177 near=0.45
#
# --- Why this exists ---------------------------------------------------------
#
# The linearized one-step map (`bench/axisspectrum.jl`) gives a growth rate per
# step; this script gives what a run makes of it. Each run carries a callback
# that reads, after every step, the grid-scale content of ρ, ρu_r and E over
# the first `W` interior nodes of the axis, as max |q_{i−1} − 2q_i + q_{i+1}|/4
# over i = 2 .. W relative to the largest |q| on the line (the momentum's
# floored at 1% of the density's), and stops the run once it exceeds `stop`
# (by default 0.05 for the pulse and never for Noh, whose axis layer holds a
# grid-scale dip of its own). It prints the first time the measure passes
# 1e-6 and 1e-3, its largest value, where the run ended, and the case's own
# reading: ρ − 1 at node 1 for the pulse, the plateau, the axis deficit and
# the front for Noh. The measurements are in reference/CALIBRATION_APPENDIX.md
# under "Grid-scale growth at the r-z axis".
#
# Cases:
#   pulse  the converging pulse of `axis_level_case` (test/smooth_cases.jl),
#          unrefined, on N radial nodes of (0, 2], filtered at every step
#          under `filter_cfl = 0.35`, to t = `tfinal` (0.4 by default)
#   noh    cylindrical Noh from the cold start as `noh_case(2)` (test/cases.jl)
#          on N nodes, to `NOH_T`, with the filter's αf replaced
#   plane  the pulse on a Cartesian line with a symmetry plane at x = 0, the
#          control: the same profile and grid without the axis metric
# Options: `N` and `alpha` are comma-separated lists; `interval` is the filter
# cadence (0 runs no filter); `cfl` > 0 fixes the pulse's step at cfl·h/c₀,
# c₀ = √1.4, in equal steps as `fixed_step_run!` takes them (the axis rows'
# step is cfl 0.177), and 0 takes the solver's own step at cfl 0.9; `tstart`
# is the time the measure starts (by default 0 for the pulse and 0.3 for Noh,
# whose shock forms at the axis and has left the window by then); `variant` is
# the patched divergence of `bench/axisspectrum.jl` (`none`, `areap`,
# `product`), and `near` > 0 filters the first `M` nodes at that αf, tapering
# to the run's αf at node 2M, as there.
#
# Cost: about 2.5 µs per node and step for the pulse, which takes 1.34 N steps
# to t = 0.4 at cfl 0.177 (a minute at N = 4607); Noh takes 8.7 N steps, 7 s at
# N = 767.
# Scratch tooling, like everything else in bench/: it prints and asserts
# nothing.

using MPI
MPI.Init(threadlevel=:funneled)
using CompactLES, Printf, LinearAlgebra
using CompactLES: padded_index, xcoord

const CL = CompactLES
MPI.Comm_size(MPI.COMM_WORLD) == 1 || error("run this study on one rank")
include(joinpath(@__DIR__, "..", "test", "smooth_cases.jl"))
include(joinpath(@__DIR__, "..", "test", "references.jl"))
include(joinpath(@__DIR__, "..", "test", "cases.jl"))
include(joinpath(@__DIR__, "axisvariant.jl"))

const OPTS = CL.script_args(ARGS, (case="pulse", N="767", alpha="0.47", interval=1,
                                   tfinal=0.4, cfl=0.0, tstart=-1.0, W=16, stop=-1.0,
                                   variant="none", near=0.0, M=8,
                                   nmax=200_000))

_list(T, s) = [parse(T, x) for x in split(s, ',')]

"The grid-scale measure and the times it first passed 1e-6 and 1e-3."
mutable struct SawtoothMonitor
    tstart::Float64
    W::Int
    stop::Float64
    peak::Float64
    t6::Float64
    t3::Float64
end

SawtoothMonitor(tstart, W, stop) = SawtoothMonitor(tstart, W, stop, 0.0, NaN, NaN)

function (m::SawtoothMonitor)(solver, Q)
    solver.t < m.tstart && return false
    eq = solver.equations
    n = solver.decomp.n_local[1]
    line(c) = (Q[padded_index(solver, i, 1, 1), c] for i in 1:n)
    # Each component relative to its largest magnitude on the line; the
    # momentum, zero at the start, relative to at least 1% of the density's.
    sρ = maximum(abs, line(1))
    s = 0.0
    for c in (1, eq.i_mom[1], eq.i_energy)
        scale = max(maximum(abs, line(c)), 0.01sρ)
        hp = 0.0
        for i in 2:m.W
            q = (Q[padded_index(solver, i - 1, 1, 1), c] -
                 2Q[padded_index(solver, i, 1, 1), c] +
                 Q[padded_index(solver, i + 1, 1, 1), c]) / 4
            hp = max(hp, abs(q))
        end
        s = max(s, hp / scale)
    end
    isfinite(s) || (s = Inf)
    m.peak = max(m.peak, s)
    s > 1e-6 && isnan(m.t6) && (m.t6 = solver.t)
    s > 1e-3 && isnan(m.t3) && (m.t3 = solver.t)
    return s > m.stop
end

CompactLES.rewind!(::SawtoothMonitor, t, step) = nothing

pulse_profile(r) = begin
    rho = 1 + 0.05 * exp(-((r - 0.5) / 0.1)^2)
    (rho, zero(r), zero(r), rho^1.4)
end

function build_pulse(N, α, interval, mode, axis, cfl; filter_cfl=0.35)
    lo = axis ? AxisBC() : SymmetryPlaneBC()
    # Under a fixed step the solver's own limit is set out of the way, so that
    # the endpoint of each call binds, as in `fixed_step_run!`.
    kw = (metric=axis ? CylindricalMetric() : CartesianMetric(),
          filt=compact_filter(α), filter_interval=interval, filter_cfl=filter_cfl,
          cfl=cfl > 0 ? 50.0 : 0.9)
    bcs = ((lo, SlipWallBC()), per3[2], per3[3])
    src = AxisVariant(mode)
    return _smooth_solver((N, 1, 1), 2.0, bcs, pulse_profile;
                          merge(SMOOTH_DEFAULTS, kw)..., sources=(src,))
end

function build_noh(N, α, interval, mode; filter_cfl=0.35)
    base = noh_problem(2; N, t0=0.0)
    # The same problem with the patched divergence as its source.
    prob = typeof(base)((f === :sources ? (AxisVariant(mode),) : getfield(base, f)
                         for f in fieldnames(typeof(base)))...)
    return setup(prob, Numerics(n_global=(N, 1, 1),
                                art=ArtificialProperties(enabled=true), cfl=NOH_CFL,
                                filter=StateFilter(compact_filter(α); cfl=filter_cfl,
                                                   interval=interval),
                                control=StepControl(validity=:permissive)))
end

function one_run(o, N, α, mode)
    noh = o.case == "noh"
    axis = o.case != "plane"
    interval = o.near > 0 ? 0 : o.interval
    solver, Q = noh ? build_noh(N, α, interval, mode) :
                build_pulse(N, α, interval, mode, axis, o.cfl)
    tfinal = noh ? NOH_T : o.tfinal
    mon = SawtoothMonitor(o.tstart < 0 ? (noh ? 0.3 : 0.0) : o.tstart, o.W,
                          o.stop < 0 ? (noh ? Inf : 0.05) : o.stop)
    cb = Callback(EveryStep(1), mon)
    if o.near > 0
        # The near-axis filter: two solvers of the same grid lend their passes.
        lend(a) = first(noh ? build_noh(N, a, 1, :none; filter_cfl=0.0) :
                        build_pulse(N, a, 1, :none, axis, o.cfl; filter_cfl=0.0))
        af = AxisFilter(lend(α), lend(o.near), o.M, 0.35, noh ? NOH_G : 1.4, Q)
        cb = (Callback(EveryStep(1), af), cb)
    end
    wall = @elapsed try
        if noh || o.cfl <= 0
            run!(solver, Q; tfinal=tfinal, nmax=o.nmax, callback=cb)
        else
            dt = o.cfl * solver.h[1] / sqrt(1.4)
            steps = ceil(Int, tfinal / dt)
            work = Workspace(Q)
            for k in 1:steps
                run!(solver, Q, work; tfinal=k == steps ? tfinal : k * dt, callback=cb)
                mon.peak > mon.stop && break
            end
        end
    catch err
        err isa SolverFailure || rethrow()
        println("  SolverFailure at t = ", solver.t)
    end
    o.near > 0 && @printf("near-axis αf %.3f over %d nodes, tapering to node %d\n",
                          o.near, o.M, 2o.M)
    @printf("%s N = %d, αf = %.2f every %d, variant %s: t = %.4f after %d steps (%.0f s)\n",
            o.case, N, α, o.interval, mode, solver.t, solver.step, wall)
    @printf("  grid-scale measure: peak %.2e, past 1e-6 at t = %.4f, past 1e-3 at t = %.4f\n",
            mon.peak, mon.t6, mon.t3)
    CL.primitives!(solver, Q)
    xs = [xcoord(solver, 1, i) for i in 1:N]
    ρ = [solver.rho[padded_index(solver, i, 1, 1)] for i in 1:N]
    if noh
        plat, deficit, front, epre = noh_metrics(xs, ρ, 2)
        @printf("  plateau %.4f  axis deficit %+.2f%%  front %.4f  L1 pre-shock %.2e\n",
                plat, 100deficit, front, epre)
    else
        @printf("  rho - 1 at node 1: %.4f\n", ρ[1] - 1)
    end
    flush(stdout)
end

function main(o)
    o.case in ("pulse", "noh", "plane") || error("case must be pulse, noh or plane")
    for mode in Symbol.(split(o.variant, ',')), α in _list(Float64, o.alpha),
        N in _list(Int, o.N)
        one_run(o, N, α, mode)
    end
end

main(OPTS)
