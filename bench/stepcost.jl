# Wall clock, allocation and garbage collection per time step of `run!` on
# configurations modeled on the documentation pages and examples, with a budget
# of the step by phase.
#
#   julia --project=. -t 1 bench/stepcost.jl                     # every case
#   julia --project=. -t 8 bench/stepcost.jl cases=bubble,tgv steps=20
#   julia --project=. -t 1 bench/stepcost.jl cases=ring affinity=0x5555
#
# Cases (`cases=`, comma-separated):
#
#   bubble   the helium cylinder of examples/shock_bubble.jl: two species,
#            NSCBC inflow and outflow, symmetry planes, planar, `ny` rows
#   ring     the meridional plane of examples/vortex_ring_shock.jl before the
#            shock: two species, r-z cylindrical metric, axis, slip walls and
#            a time-dependent Dirichlet jet face, `nr` x `nz`
#   tgv      the Taylor-Green vortex of examples/taylor_green.jl, one species,
#            molecular viscosity, triply periodic, `N`^3
#   bubbles  the tiled refinement of docs/literate/advected_bubbles.jl: four
#            species, periodic, a 96 x 96 root with 12-node tiles; whole-step
#            figures only
#
# Each case takes `warm` steps through `run!` first, which compiles every
# path, then two timed `run!` calls of `steps` and `2 steps` steps on the
# same solver; `steps = 0`, the default, sizes the first call to about three
# seconds, so that a garbage collection falling in one call and not the other
# is small against the difference. Wall time, bytes allocated and garbage-collection time per step
# are the differences between the two calls divided by `steps`, which removes
# the per-call setup of `run!` (the state validation, the positivity floors,
# the coefficient priming). `wall_step` is the median of the solver's own
# per-step timer over the second call.
#
# The budget replays the phases of one step on the state the timed run left,
# each `reps` times, and prints the median of each multiplied by the number of
# times a step takes it: six boundary enforcements (one before `max_rate`,
# four between stages and one after the last), one `max_rate`, five
# right-hand sides, five stage updates and the state filter every
# `filter_interval` steps. Four of the five right-hand sides exchange the
# state and recover the primitives themselves; the first reuses those of
# `max_rate`. A phase that writes the state is replayed on a copy restored
# between calls, outside the timed region. The flux divergence mirrors the
# branch `compute_rhs!` takes for the solver, so keep it in step with
# rhs.jl. The row `(rest of step)` is the measured step less the sum of the
# phases: run! bookkeeping, the hooks the budget does not list, and the
# difference between a phase replayed alone and the same phase inside the
# step, where the caches hold other arrays.
#
# Timings from a desktop with performance and efficiency cores spread with the
# core a process lands on; `affinity` sets the process affinity mask on
# Windows before anything is timed (0x5555 is one logical CPU on each of the
# eight performance cores of a 12900K). Compare two trees with bench/repeat.jl
# on the `step,` lines, whose fields are case, threads, ms/step, bytes/step and
# GC ms/step:
#
#   julia bench/repeat.jl runs=6 'pattern=^step,([^,]+),[^,]+,([^,]+)' \
#       -- ../before/bench/stepcost.jl cases=bubble -- bench/stepcost.jl cases=bubble

using MPI
MPI.Initialized() || MPI.Init(threadlevel=:funneled)
using CompactLES
using CompactLES.Regions
using Printf
using Statistics: median
const CL = CompactLES

opt = CL.script_args(ARGS, (cases = "bubble,ring,tgv,bubbles", steps = 0, warm = 3,
                            reps = 7, ny = 200, nr = 112, nz = 384, N = 64,
                            deriv = "c6", budget = true, affinity = 0))

# The derivative operator of every case, `deriv=c6` (the default), `c8` or `c10`.
derivative() = opt.deriv == "c6" ? lele_d1_6() : opt.deriv == "c8" ? lele_d1_8() :
               opt.deriv == "c10" ? lele_d1_10() :
               error("deriv must be c6, c8 or c10, got $(opt.deriv)")

if opt.affinity != 0 && Sys.iswindows()
    process = ccall((:GetCurrentProcess, "kernel32"), stdcall, Ptr{Cvoid}, ())
    ok = ccall((:SetProcessAffinityMask, "kernel32"), stdcall, Cint,
               (Ptr{Cvoid}, UInt), process, UInt(opt.affinity))
    ok == 0 && error("SetProcessAffinityMask failed")
end

# --- Cases ------------------------------------------------------------------

function bubble_case(ny)
    eos = IdealMixture(["Air", "He"])
    p0, T0 = 101_325.0, 295.0
    air = Prim(Y = mass_fractions(eos, "Air" => 1.0; basis = :mass), p = p0, T_ion = T0)
    helium = Prim(Y = mass_fractions(eos, "He" => 0.72, "Air" => 0.28; basis = :mass),
                  p = p0, T_ion = T0)
    incident = shock_jump(eos, air, 1.22)
    H, R, xc, x_shock, Lx = 0.0445, 0.025, 0.06, 0.025, 0.26
    problem = Problem(
        eos = eos,
        domain = ((0.0, Lx), (0.0, H), (0.0, 1.0)),
        bcs = ((NSCBCInflowBC(incident.post), NSCBCOutflowBC(pinf = p0)),
               (SymmetryPlaneBC(), SymmetryPlaneBC()), PeriodicBC()),
        ic = Layers(air, Slab(1, hi = x_shock) => incident.post,
                    Cylinder((xc, 0.0, 0.0), R) => helium))
    nx = round(Int, Lx / (H / ny)) + 1
    return setup(problem, Numerics(n_global = (nx, ny, 1), deriv = derivative()))
end

function ring_case(nr, nz)
    eos = IdealMixture(["Air", "SF6"])
    p0, T0 = 101_325.0, 295.0
    air = Prim(Y = (1.0, 0.0), p = p0, T_ion = T0)
    sf6 = Prim(Y = (0.0, 1.0), p = p0, T_ion = T0)
    R, D, z_interface, H = 0.127 / sqrt(π), 0.0127, 0.10, 0.25
    U, t_pulse = 60.0, 2 * 3.0 * 0.0127 / 60.0
    pulse(t) = t <= 0 || t >= t_pulse ? 0.0 : sin(π * t / t_pulse)^2
    jet(r, θ, z, t) = Prim(Y = (1.0, 0.0), p = p0, T_ion = T0,
                           u = (0.0, 0.0, -U * pulse(t) * (1 - tanh_blend(r, D / 2, 5e-4))))
    problem = Problem(
        eos = eos,
        metric = CylindricalMetric(),
        domain = ((0.0, R), (0.0, 2π), (0.0, H)),
        bcs = ((AxisBC(), SlipWallBC()), PeriodicBC(), (SlipWallBC(), DirichletBC(jet))),
        ic = Layers(air, Slab(3, hi = z_interface) => sf6; width = Cells(2)))
    solver, Q = setup(problem, Numerics(n_global = (nr, 1, nz), deriv = derivative()))
    # Into the pulse, so that the jet face carries a flow.
    solver.t = t_pulse / 4
    return solver, Q
end

function tgv_case(N)
    p0 = 10.0^2 / 1.4
    problem = Problem(
        eos = IdealSpecies("gas"; R = 1.0, gamma = 1.4),
        transport = ConstantTransport(mu0 = 1 / 1600),
        domain = ((0.0, 2pi), (0.0, 2pi), (0.0, 2pi)),
        bcs = (PeriodicBC(), PeriodicBC(), PeriodicBC()),
        ic = (x, y, z) -> Prim(
            u = (sin(x) * cos(y) * cos(z), -cos(x) * sin(y) * cos(z), 0.0),
            p = p0 + (cos(2x) + cos(2y)) * (cos(2z) + 2) / 16, rho = 1.0))
    return setup(problem, Numerics(n_global = (N, N, N), deriv = derivative()))
end

function bubbles_case()
    names = ["Air", "He", "SF6", "Kr"]
    eos = IdealMixture(names)
    p0, T0, L, U, R, w = 101_325.0, 300.0, 0.96, 300.0, 0.04, 0.01
    centers = ((0.30, 0.24), (0.92, 0.48), (0.92, 0.92))
    wrap(d) = d - L * round(d / L)
    bubble(b, x, y) = (1 - tanh((hypot(wrap(x - centers[b][1]),
                                       wrap(y - centers[b][2])) - R) / w)) / 2
    function initial_state(x, y, z)
        X = ntuple(b -> bubble(b, x, y), 3)
        return Prim(Y = mass_fractions(eos, "Air" => 1 - sum(X), "He" => X[1],
                                       "SF6" => X[2], "Kr" => X[3]; basis = :mole),
                    p = p0, T_ion = T0, u = (U, U, 0.0))
    end
    problem = Problem(eos = eos, domain = ((0.0, L), (0.0, L), (0.0, 1.0)),
                      bcs = (PeriodicBC(), PeriodicBC(), PeriodicBC()),
                      ic = initial_state)
    return setup(problem, Numerics(n_global = (96, 96, 1), amr = AMR(tile = 12),
                                       deriv = derivative()))
end

build(name) =
    name == "bubble" ? bubble_case(opt.ny) :
    name == "ring" ? ring_case(opt.nr, opt.nz) :
    name == "tgv" ? tgv_case(opt.N) :
    name == "bubbles" ? bubbles_case() :
    error("unknown case $name; the cases are bubble, ring, tgv and bubbles")

# --- Whole step ---------------------------------------------------------------

function timed_run!(solver, Q, work, n)
    wall_0 = solver.wall_total
    stats = @timed run!(solver, Q, work; tfinal = Inf, nmax = solver.step + n)
    return (time = stats.time, bytes = stats.bytes, gc = stats.gctime,
            wall = solver.wall_total - wall_0)
end

function whole_step(solver, Q, work, steps)
    a = timed_run!(solver, Q, work, steps)
    walls = Float64[]
    record = Callback(EveryStep(1), (s, _) -> (push!(walls, s.wall_step); nothing))
    wall_0 = solver.wall_total
    b = @timed run!(solver, Q, work; tfinal = Inf, nmax = solver.step + 2steps,
                    callback = record)
    b = (time = b.time, bytes = b.bytes, gc = b.gctime, wall = solver.wall_total - wall_0)
    return (ms = 1e3 * (b.time - a.time) / steps,
            bytes = (b.bytes - a.bytes) / steps,
            gc_ms = 1e3 * (b.gc - a.gc) / steps,
            wall_ms = 1e3 * median(walls))
end

# --- Budget -------------------------------------------------------------------

# Median of `reps` calls of `f`, each preceded (untimed) by `reset`.
function phase_time(f, reset, reps)
    reset(); f()
    t = Float64[]
    for _ in 1:reps
        reset()
        t0 = time_ns()
        f()
        push!(t, (time_ns() - t0) / 1e9)
    end
    return median(t)
end

function flux_divergence!(solver, Q, dQ)
    decomp = solver.decomp
    o1, o2, o3 = decomp.n_halo_d
    nx, ny, nz = decomp.n_local
    unitgeom = solver.metric isa CartesianMetric && all(isnothing, solver.stretch)
    for c in 1:solver.equations.n_cons
        CL.pointwise!(CL._zero_component_point!, dQ, nx, ny, nz, dQ, c, o1, o2, o3)
        for d in 1:3
            decomp.active[d] || continue
            Fdc = solver.flux[d, c]
            σ = solver.folds[d] === nothing ? 1 : solver.folds[d].sigflux[c]
            if unitgeom
                CL.div_subtract_along!(dQ, c, Fdc, solver, d, σ, nothing)
            else
                Ad = solver.area_d[d]
                n1f, n2f, n3f = CL.padded_extent(decomp)
                if CL._pressure_gradient(solver, d, c)
                    CL.pointwise!(CL._area_flux_less_point!, solver.tmp_b, n1f, n2f, n3f,
                                  solver.tmp_b, Ad, Fdc, solver.p)
                    CL.div_subtract_along!(dQ, c, solver.tmp_b, solver, d, σ,
                                           solver.inv_J)
                    CL.pressure_subtract_along!(dQ, c, solver, d)
                else
                    CL.pointwise!(CL._area_flux_point!, solver.tmp_b, n1f, n2f, n3f,
                                  solver.tmp_b, Ad, Fdc)
                    CL.div_subtract_along!(dQ, c, solver.tmp_b, solver, d, σ,
                                           solver.inv_J)
                end
            end
        end
    end
end

function budget(solver, Q, work, ms_step, reps)
    decomp = solver.decomp
    active = decomp.active
    dQ = work.dQ
    Q0 = copy(Q)
    restore() = copyto!(Q, Q0)
    none() = nothing
    vel = (solver.u, solver.v, solver.w)
    control = solver.control
    floors = CL.positivity_floors(solver, Q, control)
    floor_0 = (steps = 0, cells = 0, low_energy = 0, mass = 0.0, energy = 0.0,
               momentum = 0.0)
    rows = Tuple{String,Float64,Float64}[]      # name, seconds per call, calls per step
    phase(name, f, count; reset = none) =
        push!(rows, (name, phase_time(f, reset, reps), count))

    filter_count = solver.filter_interval > 0 ? 1 / solver.filter_interval : 0.0
    # Bring the primitives and gradients to the state, as a stage would find them.
    CL.exchange_state!(Q, decomp); CL.primitives!(solver, Q)
    CL.compute_primitives_and_gradients!(solver, Q, true, true)
    restore()

    phase("apply_bcs!", () -> CL.apply_bcs!(solver, Q), 6; reset = restore)
    phase("exchange_state!", () -> CL.exchange_state!(Q, decomp), 5)
    phase("primitives!", () -> CL.primitives!(solver, Q), 5)
    phase("max_rate sweep", () -> CL._local_max_rate(solver, Q), 1)
    phase("velocity grads", () -> for jj in 1:3, d in 1:3
        active[d] && CL.deriv_scaled_along!(solver.grad_u[d, jj], vel[jj], solver, d,
                                            CL.vel_parity(solver, d, jj))
    end, 5)
    solver.metric isa CartesianMetric ||
        phase("metric grad corr", () -> CL.metric_correct_gradients!(solver, solver.metric), 5)
    if solver.art.enabled
        phase("artificial", () -> CL.compute_artificial!(solver, Q), 5)
    end
    phase("T_ion grads", () -> for d in 1:3
        active[d] && CL.deriv_scaled_along!(solver.grad_T_ion[d], solver.T_ion, solver, d, 1)
    end, 5)
    CL._species_gradients_skipped(solver) ||
        phase("species grads", () -> CL._species_gradients!(solver), 5)
    CL._shared_species_diffusivity(solver) &&
        phase("channel grads", () -> CL._bulk_gradients!(solver, Q), 5)
    phase("assemble_fluxes!", () -> CL.assemble_fluxes!(solver, Q), 5)
    phase("correct_flux!", () -> for d in 1:3, side in 1:2
        active[d] && CL.correct_flux!(solver.bcs[d][side], solver, Q, d, side)
    end, 5)
    phase("flux exchange", () -> for d in 1:3
        CL.exchange_dim_batch!(view(solver.flux, d, :), decomp, d)
    end, 5)
    phase("flux divergence", () -> flux_divergence!(solver, Q, dQ), 5)
    solver.metric isa CartesianMetric ||
        phase("metric sources", () -> CL.add_metric_sources!(solver, dQ, Q, solver.metric), 5)
    phase("correct_rhs!", () -> for d in 1:3, side in 1:2
        active[d] && CL.correct_rhs!(solver.bcs[d][side], solver, Q, dQ, d, side)
    end, 5)
    phase("stage update", () -> CL._rk_update!(decomp, solver.equations.n_cons, Q, dQ,
                                               work.du, CL.RKA[2], CL.RKB[2], 1e-12), 5;
          reset = restore)
    filter_count > 0 &&
        phase("filter_state!", () -> CL.filter_state!(solver, Q), filter_count;
              reset = restore)
    # run! skips the failsafe where the floors are not positive.
    floors[1] > 0 &&
        phase("positivity failsafe", () -> CL._positivity_failsafe!(solver, Q, floors...,
                                                                     control, floor_0, 1),
              1; reset = restore)
    rhs = phase_time(() -> CL.compute_rhs!(solver, Q, dQ), none, reps)
    restore()

    total = sum(r[2] * r[3] for r in rows)
    @printf("  %-22s %9s %6s %9s %6s\n", "phase", "ms/call", "calls", "ms/step", "%step")
    for (name, t, n) in sort(rows, by = r -> -r[2] * r[3])
        @printf("  %-22s %9.3f %6.2f %9.3f %5.1f%%\n", name, 1e3t, n, 1e3t * n,
                100 * 1e3t * n / ms_step)
    end
    @printf("  %-22s %9s %6s %9.3f %5.1f%%\n", "(rest of step)", "", "",
            ms_step - 1e3total, 100 * (ms_step - 1e3total) / ms_step)
    @printf("  compute_rhs! alone %.3f ms; its phases above sum to %.3f ms\n", 1e3rhs,
            1e3 * sum(r[2] for r in rows if r[3] == 5 && r[1] != "stage update") -
            1e3 * sum(r[2] for r in rows if r[1] in ("exchange_state!", "primitives!")) / 5)
end

# --- Driver -------------------------------------------------------------------

function main()
    for name in split(opt.cases, ',')
        t_build = @elapsed (solver, Q) = build(name)
        work = CL.Workspace(Q)
        t_warm = @elapsed run!(solver, Q, work; tfinal = Inf, nmax = solver.step + opt.warm)
        multi = Q isa Vector
        npt = multi ? sum(prod(p.decomp.n_local) for p in solver.patches) :
              prod(solver.decomp.n_local)
        # `steps = 0` sizes each timed call to about three seconds.
        steps = opt.steps
        if steps <= 0
            t1 = @elapsed run!(solver, Q, work; tfinal = Inf, nmax = solver.step + 1)
            steps = clamp(round(Int, 3 / t1), 5, 400)
        end
        s = whole_step(solver, Q, work, steps)
        @printf("\n===== %s: %d points%s, %d species, -t %d =====\n", name, npt,
                multi ? " over $(length(solver.patches)) patches" : "",
                solver.equations.n_species, Threads.nthreads())
        @printf("  build %.1f s, %d warm steps %.1f s, timed over %d and %d steps\n",
                t_build, opt.warm, t_warm, steps, 2steps)
        @printf("  %.3f ms/step (solver timer %.3f ms), %.0f bytes/step, GC %.3f ms/step, \
                %.1f ns/point/step\n", s.ms, s.wall_ms, s.bytes, s.gc_ms, 1e6 * s.ms / npt)
        @printf("step,%s,%d,%.4f,%.0f,%.4f\n", name, Threads.nthreads(), s.ms, s.bytes,
                s.gc_ms)
        opt.budget && !multi && budget(solver, Q, work, s.ms, opt.reps)
        flush(stdout)
    end
end
# Run as a script; another script may include this file for the cases alone.
abspath(PROGRAM_FILE) == (@__FILE__) && main()
