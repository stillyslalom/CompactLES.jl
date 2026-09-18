# The round-off-seeded transverse mode of the aligned planar Noh case.
#
#   julia --project=. -t 1 bench/noh_transverse.jl trace
#   julia --project=. -t 1 bench/noh_transverse.jl channels
#   julia --project=. -t 1 bench/noh_transverse.jl seeds
#   julia --project=. -t 1 bench/noh_transverse.jl widths
#   julia --project=. -t 1 bench/noh_transverse.jl coupling
#   julia --project=. -t 1 bench/noh_transverse.jl seed_channels seed=1e-10
#   julia --project=. -t 1 bench/noh_transverse.jl warm t0=0.1
#   julia --project=. -t 1 bench/noh_transverse.jl uniform seed=1e-10
#   julia --project=. -t 1 bench/noh_transverse.jl extended tfinal=1.2
#
# Settings (`key=value`): N, AR, nx, tfinal, nmax, sample (steps between
# non-mutating observations), seed, seed_mode, and t0. `trace` is the validation
# case, continuously integrated to its endpoint. `channels` removes the state
# filter or selected
# artificial-property channels. `seeds` injects one density Fourier mode at
# three amplitudes while preserving pressure and velocity. `widths` changes
# the periodic transverse width at fixed spacing. `extended` follows the
# default trajectory beyond the validation endpoint to test saturation.
#
# Every observation reads the conserved state in a post-step callback. It does
# not refresh primitives, land on a requested time, split run!, or mutate the
# state, so the unseeded trace has the validation case's time-step sequence.
# The density and transverse velocity are projected on every discrete Fourier
# mode at every longitudinal node. Besides the largest transverse spread, each
# sample records where the mode lives: the first four wall cells, the four-cell
# shock window, or neither. Scratch tooling, like everything else in bench/:
# it prints tables and asserts nothing.

using MPI
MPI.Init(threadlevel=:funneled)
using CompactLES
using Printf
using LinearAlgebra

const CL = CompactLES
include(joinpath(@__DIR__, "..", "test", "references.jl"))
include(joinpath(@__DIR__, "..", "test", "cases.jl"))

const OPTS = CL.script_args(filter(a -> occursin('=', a), ARGS),
    (N=100, AR=4, nx=12, tfinal=NOH_T, nmax=20000, sample=20, seed_mode=2,
     seed=1e-10, t0=0.1))
const NAMES = filter(a -> !occursin('=', a), ARGS)
const PARTS = isempty(NAMES) ? ["trace"] : NAMES
MPI.Comm_size(MPI.COMM_WORLD) == 1 || error("noh_transverse runs serially")

struct ModeSample
    step::Int
    t::Float64
    spread::Float64
    station_difference::Float64
    rho_modes::Vector{Float64}
    rho_y::Vector{Float64}
    velocity_modes::Vector{Float64}
    velocity_y::Vector{Float64}
end

mutable struct ModeTrace
    samples::Vector{ModeSample}
    interval::Int
end

# Read Q directly: observations run after the state filter and must not change
# the trajectory by refreshing the solver's primitive caches.
function mode_sample(solver, Q)
    nx, ny = solver.decomp.n_local[1:2]
    nm = fld(nx, 2)
    m1 = solver.equations.i_mom[1]
    rho_modes = zeros(nm); rho_y = zeros(nm)
    velocity_modes = zeros(nm); velocity_y = zeros(nm)
    spread = 0.0; station_difference = 0.0
    for j in 1:ny
        rho = [Float64(Q[gidx(solver, i, j, 1), 1]) for i in 1:nx]
        ux = [Float64(Q[gidx(solver, i, j, 1), m1] /
                      Q[gidx(solver, i, j, 1), 1]) for i in 1:nx]
        spread = max(spread, maximum(rho) - minimum(rho))
        station_difference = max(station_difference,
                                 maximum(abs(rho[i] - rho[1]) for i in 1:nx))
        rho_reference = rho[1]
        ux_reference = ux[1]
        for m in 1:nm
            rr = 0.0; ri = 0.0; ur = 0.0; ui = 0.0
            for i in 1:nx
                angle = 2pi * m * (i - 1) / nx
                c, sn = cos(angle), sin(angle)
                rr += (rho[i] - rho_reference) * c
                ri -= (rho[i] - rho_reference) * sn
                ur += (ux[i] - ux_reference) * c
                ui -= (ux[i] - ux_reference) * sn
            end
            scale = 2 / nx
            iseven(nx) && m == nm && (scale = 1 / nx)
            arho = scale * hypot(rr, ri)
            au = scale * hypot(ur, ui)
            if arho > rho_modes[m]
                rho_modes[m] = arho
                rho_y[m] = xcoord(solver, 2, j)
            end
            if au > velocity_modes[m]
                velocity_modes[m] = au
                velocity_y[m] = xcoord(solver, 2, j)
            end
        end
    end
    ModeSample(solver.step, Float64(solver.t), spread, station_difference,
               rho_modes, rho_y,
               velocity_modes, velocity_y)
end

function observe!(trace::ModeTrace, solver, Q)
    solver.step % trace.interval == 0 && push!(trace.samples, mode_sample(solver, Q))
    return false
end

function art_channels(name)
    name == :default && return ArtParams(enabled=true)
    name == :off && return ArtParams(enabled=false)
    name == :beta_only && return ArtParams(C_mu=0.0, C_beta=1.0, C_kappa=0.0,
                                           C_D=0.0)
    name == :beta_mu && return ArtParams(C_mu=0.002, C_beta=1.0, C_kappa=0.0,
                                         C_D=0.0)
    name == :beta_kappa && return ArtParams(C_mu=0.0, C_beta=1.0, C_kappa=0.01,
                                            C_D=0.0)
    name == :no_species && return ArtParams(C_mu=0.002, C_beta=1.0, C_kappa=0.01,
                                            C_D=0.0)
    name == :beta_off && return ArtParams(C_beta=0.0)
    name == :mu_only && return ArtParams(C_mu=0.002, C_beta=0.0, C_kappa=0.0,
                                         C_D=0.0)
    name == :kappa_only && return ArtParams(C_mu=0.0, C_beta=0.0, C_kappa=0.01,
                                            C_D=0.0)
    error("unknown artificial-property selection $name")
end

function aligned_problem(; N, AR, nx, seed, seed_mode, t0=0.0)
    h2 = 1.0 / (N - 1)
    h1 = h2 / AR
    Lx = nx * h1
    inflow = DirichletBC((x, y, z, t) -> Prim(rho=1.0, u=(0.0, -1.0, 0.0),
                                              p=NOH_P0))
    w = 4h2
    ic = (x, y, z) -> begin
        if t0 <= 0
            rho, uy, pressure = 1.0, -1.0, NOH_P0
        else
            rho_in, _, p_in = noh_exact(0.0, t0, 1, NOH_G)
            rho_out, _, _ = noh_exact(y, t0, 1, NOH_G)
            theta = tanh_blend(y, (NOH_G - 1) / 2 * t0, w)
            rho = (1 - theta) * rho_in + theta * rho_out
            uy = -theta
            pressure = (1 - theta) * p_in + theta * NOH_P0
        end
        rho *= 1 + seed * cos(2pi * seed_mode * x / Lx)
        Prim(rho=rho, u=(0.0, uy, 0.0), p=pressure)
    end
    per = (PeriodicBC(), PeriodicBC())
    problem = Problem(eos=IdealSpecies("gas"; gamma=NOH_G, R=1.0),
                      transport=Transport(mu0=0.0),
                      domain=((0.0, Lx), (0.0, 1.0), (0.0, h2)),
                      bcs=(per, (SlipWallBC(), inflow), per), ic=ic)
    return problem
end

function uniform_postshock_problem(; N, AR, nx, seed, seed_mode)
    h2 = 1.0 / (N - 1)
    h1 = h2 / AR
    Lx = nx * h1
    rho0, uy0, pressure0 = noh_exact(0.0, 1.0, 1, NOH_G)
    state = (x, y, z, t) -> Prim(rho=rho0, u=(0.0, uy0, 0.0), p=pressure0)
    ic = (x, y, z) -> Prim(rho=rho0 *
                                 (1 + seed * cos(2pi * seed_mode * x / Lx)),
                             u=(0.0, uy0, 0.0), p=pressure0)
    per = (PeriodicBC(), PeriodicBC())
    return Problem(eos=IdealSpecies("gas"; gamma=NOH_G, R=1.0),
                   transport=Transport(mu0=0.0),
                   domain=((0.0, Lx), (0.0, 1.0), (0.0, h2)),
                   bcs=(per, (SlipWallBC(), DirichletBC(state)), per), ic=ic)
end

function run_case(label; N=OPTS.N, AR=OPTS.AR, nx=OPTS.nx, tfinal=OPTS.tfinal,
                  seed=0.0, seed_mode=OPTS.seed_mode, t0=0.0,
                  channels=:default, filter_on=true, deriv=lele_d1_6(),
                  uniform=false)
    N >= 9 || throw(ArgumentError("N must be at least 9 for the C8 filter"))
    nx >= 9 || throw(ArgumentError("nx must be at least 9 for the C8 filter"))
    AR > 0 || throw(ArgumentError("AR must be positive"))
    OPTS.sample > 0 || throw(ArgumentError("sample must be positive"))
    OPTS.nmax > 0 || throw(ArgumentError("nmax must be positive"))
    t0 >= 0 || throw(ArgumentError("t0 must be nonnegative"))
    tfinal > t0 || throw(ArgumentError("tfinal must exceed t0"))
    seed == 0 || 1 <= seed_mode <= fld(nx, 2) ||
        throw(ArgumentError("a nonzero seed needs 1 <= seed_mode <= floor(nx/2)"))
    problem = uniform ? uniform_postshock_problem(; N, AR, nx, seed, seed_mode) :
                        aligned_problem(; N, AR, nx, seed, seed_mode, t0)
    numerics = Numerics(n_global=(nx, N, 1), art=art_channels(channels), cfl=NC_CFL,
                        deriv=deriv, filt=compact_filter(0.45),
                        filter_interval=filter_on ? 1 : 0, filter_cfl=0.35,
                        control=StepControl(validity=:permissive))
    solver, Q = setup(problem, numerics)
    trace = ModeTrace(ModeSample[mode_sample(solver, Q)], OPTS.sample)
    failure = nothing
    wall = @elapsed try
        run!(solver, Q; tfinal=tfinal - t0, nmax=OPTS.nmax,
             callback=(s, q) -> observe!(trace, s, q))
    catch err
        err isa SolverFailure || rethrow()
        failure = err
    end
    (isempty(trace.samples) || trace.samples[end].step != solver.step) &&
        push!(trace.samples, mode_sample(solver, Q))
    return (; label, solver, Q, trace, wall, failure, seed, seed_mode, t0,
            tfinal, N, AR, nx, uniform)
end

function growth_rate(samples, mode, ta, tb)
    rows = [(s.t, s.rho_modes[mode]) for s in samples if ta <= s.t <= tb &&
            isfinite(s.rho_modes[mode]) && s.rho_modes[mode] > 2e-16]
    length(rows) >= 3 || return NaN
    ts = first.(rows); ys = log.(last.(rows))
    tc = ts .- sum(ts) / length(ts)
    return dot(tc, ys) / dot(tc, tc)
end

function location(y, t, t0, N; uniform=false)
    h = 1 / (N - 1)
    y <= 4h && return "wall"
    uniform && return "bulk"
    Rs = (NOH_G - 1) / 2 * (t + t0)
    abs(y - Rs) <= 4h && return "shock"
    return "bulk"
end

function report(r; detail=false, tracked_mode=nothing)
    last = r.trace.samples[end]
    dominant = argmax(last.rho_modes)
    m = something(tracked_mode, dominant)
    half = max(last.t / 2, eps())
    q1 = growth_rate(r.trace.samples, m, 0.0, half)
    q2 = growth_rate(r.trace.samples, m, half, last.t)
    why = r.failure !== nothing ? "FAILED $(r.failure.reason)" :
          completed(r.solver, r.tfinal - r.t0) ? "complete" : "STOPPED"
    @printf("%-20s %s  steps %5d  t %.4f  wall %.1fs  range %.3e  station %.3e\n",
            r.label, why, r.solver.step, last.t, r.wall, last.spread,
            last.station_difference)
    @printf("  rho: m=%d (%.1f-cell wavelength) A=%.3e at y=%.4f (%s); ",
            m, r.nx / m, last.rho_modes[m], last.rho_y[m],
            location(last.rho_y[m], last.t, r.t0, r.N; uniform=r.uniform))
    @printf("rates first/second half %+.2f / %+.2f\n", q1, q2)
    mu = argmax(last.velocity_modes)
    @printf("  ux:  m=%d A=%.3e at y=%.4f (%s)\n", mu, last.velocity_modes[mu],
            last.velocity_y[mu],
            location(last.velocity_y[mu], last.t, r.t0, r.N; uniform=r.uniform))
    if tracked_mode !== nothing
        initial = r.trace.samples[1].rho_modes[m]
        gain = initial > 0 ? last.rho_modes[m] / initial : NaN
        average_rate = initial > 0 ? log(gain) / last.t : NaN
        @printf("  tracked seeded mode m=%d; final dominant m=%d; gain %.2e; ",
                m, dominant, gain)
        @printf("finite-horizon rate %+.2f\n", average_rate)
    end
    order = sortperm(last.rho_modes; rev=true)
    @printf("  spectrum: %s\n", join((@sprintf("m=%d %.2e", k, last.rho_modes[k])
                                           for k in order[1:min(4, end)]), ", "))
    if detail
        println("  step       t        range      station      A_m       y_m     region")
        for s in r.trace.samples
            showrow = s.step <= 200 || s.step % 500 == 0 || s.step == last.step
            showrow || continue
            @printf("  %5d  %.5f  %.3e  %.3e  %.3e  %.4f  %s\n", s.step, s.t,
                    s.spread, s.station_difference, s.rho_modes[m], s.rho_y[m],
                    location(s.rho_y[m], s.t, r.t0, r.N; uniform=r.uniform))
        end
    end
    flush(stdout)
end

function trace_part()
    println("\n=== aligned Noh transverse trace: N=$(OPTS.N), AR=$(OPTS.AR), " *
            "nx=$(OPTS.nx), sample=$(OPTS.sample) ===")
    report(run_case("default"); detail=true)
end

function channels_part()
    println("\n=== channel ablations: N=$(OPTS.N), AR=$(OPTS.AR), nx=$(OPTS.nx) ===")
    rows = (("default", :default, true), ("filter off", :default, false),
            ("art off", :off, true), ("beta only", :beta_only, true),
            ("beta + mu", :beta_mu, true), ("beta + kappa", :beta_kappa, true),
            ("beta off", :beta_off, true), ("mu only", :mu_only, true),
            ("kappa only", :kappa_only, true))
    for (label, channels, filter_on) in rows
        report(run_case(label; channels, filter_on))
    end
end

function seeds_part()
    mode = OPTS.seed_mode
    println("\n=== controlled density mode m=$mode: N=$(OPTS.N), AR=$(OPTS.AR), " *
            "nx=$(OPTS.nx) ===")
    for a in (1e-12, 1e-10, 1e-8)
        report(run_case(@sprintf("seed %.0e", a); seed=a, seed_mode=mode);
               detail=true, tracked_mode=mode)
    end
end

function widths_part()
    println("\n=== transverse-width sweep at fixed spacing: N=$(OPTS.N), AR=$(OPTS.AR) ===")
    # The default C8 state filter needs at least nine points on a line.
    for nx in (10, 12, 16, 24)
        report(run_case("nx=$nx"; nx))
    end
end

function coupling_part()
    println("\n=== coupled artificial channels, with and without species diffusion ===")
    report(run_case("default"))
    report(run_case("C_D = 0"; channels=:no_species))
end

function seed_channels_part()
    mode = OPTS.seed_mode
    println("\n=== matched m=$mode seed $(OPTS.seed) across artificial channels ===")
    rows = (("default", :default), ("beta only", :beta_only),
            ("beta + mu", :beta_mu), ("beta + kappa", :beta_kappa),
            ("C_D = 0", :no_species))
    for (label, channels) in rows
        report(run_case(label; channels, seed=OPTS.seed, seed_mode=mode);
               tracked_mode=mode)
    end
end

function warm_part()
    mode = OPTS.seed_mode
    println("\n=== exact-profile warm start t0=$(OPTS.t0), ending at t=$(OPTS.tfinal) ===")
    report(run_case("warm unseeded"; t0=OPTS.t0))
    report(run_case("warm seeded"; t0=OPTS.t0, seed=OPTS.seed, seed_mode=mode);
           tracked_mode=mode)
end

function uniform_part()
    mode = OPTS.seed_mode
    println("\n=== uniform exact postshock strip on the aligned grid ===")
    report(run_case("uniform unseeded"; uniform=true))
    report(run_case("uniform seeded"; uniform=true, seed=OPTS.seed, seed_mode=mode);
           tracked_mode=mode)
end

function extended_part()
    println("\n=== extended default trajectory: tfinal=$(OPTS.tfinal) ===")
    report(run_case("extended"); detail=true)
end

for part in PARTS
    part == "trace" ? trace_part() :
    part == "channels" ? channels_part() :
    part == "seeds" ? seeds_part() :
    part == "widths" ? widths_part() :
    part == "coupling" ? coupling_part() :
    part == "seed_channels" ? seed_channels_part() :
    part == "warm" ? warm_part() :
    part == "uniform" ? uniform_part() :
    part == "extended" ? extended_part() :
    error("unknown part '$part'; want trace, channels, seeds, widths, coupling, " *
          "seed_channels, warm, uniform, or extended")
end
