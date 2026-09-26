# Cost of restricting a refined level into its parent before every RK stage
# of the global step, against the package's once-per-step restriction, and
# what the per-stage schedule changes in the error and the stable step.
#
#   julia --project=. -t 16 bench/restrictcost.jl [study=time] [case=2d] [sub=false]
#       [reps=5] [steps=10]
#   mpiexec -n 4 julia --project=. -t 1 bench/restrictcost.jl case=2d
#   julia --project=. -t 1 bench/restrictcost.jl study=error
#   julia --project=. -t 1 bench/restrictcost.jl study=cfl
#
# `time` builds one solver and alternates the two schedules within the
# process, `reps` times, `steps` steps each, and reports medians of the step
# wall time and of the time inside `restrict_level!` (max over ranks). The
# per-stage schedule is the `restrict=stage` override of
# bench/temporalorder.jl, restricted to the stage loop: `restrict_level!` runs
# before each stage's `prolong_level_ghosts!` but not inside `sync_levels!`,
# so a step carries 5 stage restrictions and the post-step one. The overrides
# are process-wide methods on package functions. Under `sub=true` only the
# package's schedule runs: the subcycled driver imposes the shell from the
# Hermite box, not through `prolong_level_ghosts!`, so the override is inert.
#
# `error` runs the two-level entropy wave of test/smooth_cases.jl to t = 0.5 at
# the smooth cases' default cfl and at the evolution studies' 0.25, both
# schedules, and prints the errors against the exact solution by region.
# `cfl` raises the cfl on the same case until the run fails, both schedules.

using MPI
MPI.Init(threadlevel=:funneled)
using CompactLES, Printf, Statistics
const CL = CompactLES
include(joinpath(@__DIR__, "..", "test", "smooth_cases.jl"))

const OPTS = CL.script_args(ARGS, (study="time", case="2d", sub=false, reps=5,
                                   steps=10, tile=16))

const STAGE = Ref(false)
const IN_SYNC = Ref(false)
const T_RESTRICT = Ref(0.0)
const N_RESTRICT = Ref(0)
const T_PROLONG = Ref(0.0)

function CL.restrict_level!(solver::CL.Solver, states::Vector)
    t0 = time_ns()
    invoke(CL.restrict_level!, Tuple{Any,Any}, solver, states)
    T_RESTRICT[] += (time_ns() - t0) / 1e9
    N_RESTRICT[] += 1
    return states
end

function CL.sync_levels!(solver::CL.Solver, states::Vector)
    IN_SYNC[] = true
    try
        invoke(CL.sync_levels!, Tuple{Any,Any}, solver, states)
    finally
        IN_SYNC[] = false
    end
    return states
end

function CL.prolong_level_ghosts!(solver::CL.Solver, states::Vector)
    STAGE[] && !IN_SYNC[] && CL.restrict_level!(solver, states)
    t0 = time_ns()
    invoke(CL.prolong_level_ghosts!, Tuple{Any,Any}, solver, states)
    T_PROLONG[] += (time_ns() - t0) / 1e9
    return states
end

const PER3 = ntuple(_ -> (PeriodicBC(), PeriodicBC()), 3)

# A smooth advected field on the periodic [0, 2π)^d, the package's default
# numerics (artificial properties and the per-step filter on).
function timing_case(case, sub, tile)
    if case == "1d"
        N = 960
        n, reg = (N, 1, 1), BlockRegion((5N ÷ 12, 0, 0), (N ÷ 6 + 1, 1, 1))
        tile = 0
    elseif case == "2d"
        n, reg = (256, 256, 1), BlockRegion((80, 80, 0), (97, 97, 1))
    elseif case == "3d"
        n, reg = (64, 64, 64), BlockRegion((17, 17, 17), (31, 31, 31))
    else
        error("case must be 1d, 2d or 3d")
    end
    solver = Solver(n_global=n, L_domain=(2pi, 2pi, 2pi), bcs=PER3, refine=reg,
                    tile=tile, subcycle=sub, cfl=0.5)
    states = allocate_state(solver)
    initialize!(solver, states, (x, y, z) -> Prim(
        rho=1 + 0.1 * sin(x) * (n[2] > 1 ? sin(y) : 1.0) * (n[3] > 1 ? sin(z) : 1.0),
        u=(0.5, n[2] > 1 ? 0.3 : 0.0, n[3] > 1 ? 0.2 : 0.0), p=1.0))
    return solver, states, reg
end

function timed_run!(solver, states, ws, steps, stage)
    STAGE[] = stage
    T_RESTRICT[] = 0.0; N_RESTRICT[] = 0; T_PROLONG[] = 0.0
    MPI.Barrier(MPI.COMM_WORLD)
    t = @elapsed run!(solver, states, ws; tfinal=1e9, nmax=solver.step + steps)
    comm = MPI.COMM_WORLD
    return (step=MPI.Allreduce(t, max, comm) / steps,
            restrict=MPI.Allreduce(T_RESTRICT[], max, comm) / steps,
            prolong=MPI.Allreduce(T_PROLONG[], max, comm) / steps,
            count=N_RESTRICT[] / steps)
end

function time_study()
    rank = MPI.Comm_rank(MPI.COMM_WORLD)
    np = MPI.Comm_size(MPI.COMM_WORLD)
    solver, states, reg = timing_case(OPTS.case, OPTS.sub, OPTS.tile)
    ws = CL.Workspace(states)
    modes = OPTS.sub ? (false,) : (false, true)
    for m in modes
        timed_run!(solver, states, ws, 2, m)     # compile both paths
    end
    res = Dict(m => NamedTuple[] for m in modes)
    for r in 1:OPTS.reps
        for m in (isodd(r) ? modes : reverse(modes))
            push!(res[m], timed_run!(solver, states, ws, OPTS.steps, m))
        end
    end
    rank == 0 || return
    @printf("case %s, root %s, region %s, tiles %d (on rank 0), sub=%s, np=%d, threads=%d\n",
            OPTS.case, string(solver.patches[1].decomp.n_global), string(reg.extent),
            length(level_regions(solver, 1)), OPTS.sub, np, Threads.nthreads())
    @printf("%-8s %10s %10s %8s %10s %9s\n", "sched", "step [ms]", "restr [ms]",
            "restr/st", "prolong", "restr %")
    base = 0.0
    for m in modes
        st = median(getfield.(res[m], :step)) * 1e3
        re = median(getfield.(res[m], :restrict)) * 1e3
        pr = median(getfield.(res[m], :prolong)) * 1e3
        m || (base = st)
        @printf("%-8s %10.3f %10.4f %8.2f %10.3f %8.2f%%", m ? "stage" : "step", st, re,
                res[m][1].count, pr, 100re / st)
        m && @printf("   overhead %+.2f%%", 100 * (st / base - 1))
        println()
    end
    # Paired per-rep ratios, the within-process comparison.
    if !OPTS.sub
        ratios = [b.step / a.step for (a, b) in zip(res[false], res[true])]
        @printf("paired stage/step ratios: %s, median %.4f\n",
                join((@sprintf("%.4f", x) for x in ratios), " "), median(ratios))
    end
end

entropy_ref(s) = analytic_reference(s.equations, entropy_profile(3, 0.37; t=s.t))

function error_study()
    println("two-level entropy wave, t = 0.5, errors in rho against the exact solution")
    @printf("%-5s %5s %-6s %6s %11s %11s %11s\n", "N", "cfl", "sched", "steps",
            "interface", "interior", "l2")
    for N in (96, 192), cfl in (0.25, 0.5), m in (false, true)
        STAGE[] = m
        s, q = entropy_case(N; levels=2, cfl=cfl)
        run!(s, q; tfinal=0.5)
        e = regional_errors(s, q, entropy_ref(s))
        @printf("%-5d %5.2f %-6s %6d %11.4e %11.4e %11.4e\n", N, cfl,
                m ? "stage" : "step", s.step, e.interface, e.interior, e.l2)
    end
end

function cfl_study()
    println("two-level entropy wave N = 96, t = 2: rho error by cfl (Inf = failed)")
    for cfl in (1.0, 1.4, 1.8, 1.85, 1.9, 1.95, 2.0), m in (false, true)
        STAGE[] = m
        s, q = entropy_case(96; levels=2, cfl=cfl)
        err = try
            run!(s, q; tfinal=2.0, nmax=4000)
            e = regional_errors(s, q, entropy_ref(s)).l2
            isfinite(e) ? e : Inf
        catch
            Inf
        end
        @printf("cfl %.2f %-6s %.4e\n", cfl, m ? "stage" : "step", err)
    end
end

function main()
    OPTS.study == "time" && return time_study()
    MPI.Comm_size(MPI.COMM_WORLD) == 1 || error("run $(OPTS.study) on one rank")
    OPTS.study == "error" && return error_study()
    OPTS.study == "cfl" && return cfl_study()
    error("study must be time, error or cfl")
end

main()
