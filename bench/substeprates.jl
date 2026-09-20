# Fresh-coefficient CFL growth inside Berger--Oliger substeps.
#
#   julia --project=. -t 1 bench/substeprates.jl steps=3 levels=3,4
#
# This calls the production subcycled driver and reads its private guard result.
using MPI
MPI.Init(threadlevel=:funneled)
using CompactLES
using Printf
const CL = CompactLES
const OPTS = CL.script_args(ARGS, (steps=3, levels="3,4", regrid_steps=6))
OPTS.steps > 0 && OPTS.regrid_steps > 0 || error("step counts must be positive")
printfmt(fmt, values...) = Printf.format(stdout, Printf.Format(fmt), values...)

wall2 = (SlipWallBC(), SlipWallBC())
per = (PeriodicBC(), PeriodicBC())
shock(x, x0) = x < x0 ? Prim(u=(0, 0, 0), p=1.0, rho=1.0) :
                         Prim(u=(0, 0, 0), p=0.1, rho=0.125)

function regions(depth)
    all = [BlockRegion((120, 0, 0), (41, 1, 1)),
           BlockRegion((390, 0, 0), (60, 1, 1)),
           BlockRegion((1200, 0, 0), (90, 1, 1))]
    return all[1:depth-1]
end

function make_solver(depth; dynamic=false, guarded=true)
    refine = dynamic ? BlockRegion((50, 0, 0), (31, 1, 1)) : regions(depth)
    Solver(n_global=(201, 1, 1), L_domain=(1.0, 1.0, 1.0),
           bcs=(wall2, per, per), cfl=0.2,
           subcycle=true, filter_interval=1, filter_cfl=0.35,
           control=StepControl(substep_cfl=guarded ? 1e99 : 0.0),
           regrid_interval=dynamic ? 1 : 0, refine=refine)
end

function measured_run(solver, nsteps, label;
                      initial=(x, y, z) -> shock(x, 0.69), record=true)
    states = allocate_state(solver)
    initialize!(solver, states, initial)
    workspace = CL.Workspace(states)
    CL._prime_coefficients!(solver, states, workspace)
    rows = NamedTuple[]
    wall = 0.0
    for _ in 1:nsteps
        before = CL.refined_region(solver, 1)
        CL._maybe_regrid!(solver, states, workspace, nothing)
        after = CL.refined_region(solver, 1)
        context = before == after ? label * "-steady" : label * "-changed"
        CL._presync!(solver, states)
        solver.tstage = solver.t
        apply_bcs!(solver, states)
        rate, _, direction_rate = max_rate(solver, states)
        dt = solver.cfl / rate
        started = time_ns()
        status, guard = CL._subcycled_step_status!(solver, states, workspace.dQ,
                                                   workspace.du, dt, true,
                                                   solver.control)
        wall += (time_ns() - started) / 1e9
        status == 0 || error("unexpected measured-driver status $status")
        record && push!(rows, (context=context, root_step=solver.step + 1,
                              cfl_fraction=guard.dt * guard.rate,
                              normalized_growth=guard.dt * guard.rate / solver.cfl,
                              level=guard.level, stage=guard.stage,
                              substep=guard.count, before=before, after=after))
        solver.t += dt
        solver.step += 1
        solver.dt_prev = dt
        solver.rate_prev = rate
        solver.filter_rate_prev = direction_rate
        solver.filter_interval > 0 && solver.step % solver.filter_interval == 0 &&
            filter_state!(solver, states)
        CL._post_step!(solver, states)
    end
    return rows, wall
end

smooth(x, y, z) = Prim(u=(0.05sin(2pi*x), 0, 0), p=1.0,
                       rho=1.0 + 0.05sin(2pi*x))

function guard_timing(depth, nsteps)
    # Two whole warm runs compile both branches before the paired samples.
    measured_run(make_solver(depth; guarded=false), nsteps, "warm";
                 initial=smooth, record=false)
    measured_run(make_solver(depth; guarded=true), nsteps, "warm";
                 initial=smooth, record=false)
    off = Float64[]; on = Float64[]
    for _ in 1:5
        _, a = measured_run(make_solver(depth; guarded=false), nsteps, "timing";
                            initial=smooth, record=false)
        _, b = measured_run(make_solver(depth; guarded=true), nsteps, "timing";
                            initial=smooth, record=false)
        push!(off, a / nsteps); push!(on, b / nsteps)
    end
    printfmt("L%d warmed guard timing: off %.3f ms, on %.3f ms/root-step, " *
             "overhead %.1f%% (minima of 5 paired runs x %d steps)\n",
            depth, 1e3minimum(off), 1e3minimum(on),
            100(minimum(on) / minimum(off) - 1), nsteps)
end

function endpoint_estimate(depth)
    solver = make_solver(depth)
    states = allocate_state(solver)
    initialize!(solver, states, smooth)
    workspace = CL.Workspace(states)
    CL._prime_coefficients!(solver, states, workspace)
    costs = Float64[]
    for lev in solver.levels
        for _ in 1:2
            CL._level_rhs!(solver, lev, states, workspace.dQ, false, true,
                           lev.level_comm.comm)
        end
        samples = [@elapsed CL._level_rhs!(solver, lev, states, workspace.dQ,
                                           false, true, lev.level_comm.comm)
                   for _ in 1:5]
        push!(costs, minimum(samples))
    end
    stage_counts = [5 * 3^(level - 1) for level in 1:depth]
    endpoint_counts = [level < depth ? 3^(level - 1) : 0 for level in 1:depth]
    endpoint = sum(endpoint_counts .* costs)
    total = sum((stage_counts .+ endpoint_counts) .* costs)
    @printf("L%d Hermite endpoint estimate: %.2f%% of RHS wall (level RHS us %s)\n",
            depth, 100endpoint / total,
            join((@sprintf("%.1f", 1e6c) for c in costs), ","))
end

function report(rows, wall)
    for context in unique(getfield.(rows, :context))
        selected = filter(r -> r.context == context, rows)
        peak = selected[argmax(getfield.(selected, :cfl_fraction))]
        printfmt("%-18s peak refreshed substep CFL %.6f (%.6fx root target) " *
                 "at root %d level %d substep %d stage %d\n",
                peak.context, peak.cfl_fraction, peak.normalized_growth,
                peak.root_step, peak.level, peak.substep, peak.stage)
        if endswith(context, "-changed")
            @printf("  layout %s -> %s (%d changed check%s)\n", peak.before,
                    peak.after, length(selected), length(selected) == 1 ? "" : "s")
        end
    end
    @printf("  driver including compilation %.3f ms/root-step\n", 1e3wall / length(rows))
end

MPI.Comm_size(MPI.COMM_WORLD) == 1 || error("substeprates.jl is a serial diagnostic")
for depth in parse.(Int, split(OPTS.levels, ','))
    3 <= depth <= 4 || error("levels must contain only 3 or 4")
    report(measured_run(make_solver(depth), OPTS.steps, "startup-L$depth")...)
end
report(measured_run(make_solver(2; dynamic=true), OPTS.regrid_steps, "regrid";
                    initial=(x, y, z) -> shock(x, 0.5))...)
for depth in parse.(Int, split(OPTS.levels, ','))
    guard_timing(depth, max(OPTS.steps, 3))
    endpoint_estimate(depth)
end
