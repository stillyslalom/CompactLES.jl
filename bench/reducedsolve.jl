# Cost of the three stages of a distributed compact line solve, timed
# separately: the local banded sweep, the Allgather of the interface values,
# and the dense solve of the replicated reduced system (`_reduced_solve!` in
# src/tridiag.jl and src/banded.jl). It is the instrument of ROADMAP item S12,
# which predicts from an operation count that the reduced stage, about
# 8 q^2 P^2 operations per line against 9 q n for the local work, overtakes
# the local work at a few hundred to a thousand ranks.
#
# For each derivative operator (C6 and C8 run the tridiagonal `LineSolver`,
# q = 1; C10 the pentadiagonal `BandLineSolver`, q = 2) and each direction,
# the derivative plan of a fully periodic solver is filled from a smooth field
# and its line solve is timed in parts:
#
#   sweep     the local elimination alone: the same `solve_lines!` or
#             `solve_lines_t!` call with the reduced stage switched off
#   total     the whole line solve: sweep, reduced stage, spike correction
#   allgather `MPI.Allgather!` of the 2q interface values per line over the
#             P ranks of the direction, with the buffers `_reduced_solve!` uses
#   ldiv      the triangular solves of the factorized reduced matrix, of order
#             2qP, for every line at once (the replicated O(P^2) term)
#   reduced   `_reduced_solve!` whole: allgather, pack, ldiv, unpack
#   local     total - reduced: the sweep and the correction, the work the
#             9qn count describes (derived, not timed)
#   apply     `apply_along!` whole: fill, line solve, scatter, for scale
#
# The line solves of the filter and the Gaussian smoother share this code, so
# the derivative plan stands for all of them.
#
# Every timed call is preceded by a barrier on the direction's sub-communicator,
# so the Allgather measures the collective and not the arrival skew of the
# ranks before it. Each rank takes the median over `reps` calls, and the table
# reports the maximum of those medians over all ranks, in one Allreduce: the
# slowest rank of a replicated collective is the one the step pays. Every rank
# runs every call the same number of times, so all timers are collective-safe.
# The inputs are restored before each call (outside the timer), so repeated
# solves neither grow nor decay into denormals.
#
# Columns of the table, per direction: P ranks along it, n local points per
# line, L lines per rank, q, and the per-line counts 9qn and 8q^2P^2 of the
# S12 model with their ratio; then the measured times in microseconds, the
# measured ratio ldiv / local, and two crossover estimates. On a uniform grid
# with P ranks per direction the per-rank local work is 9qN^3/P^3 and the
# reduced work 8q^2 N^2, which cross at
#
#   P_count = (9 N / (8 q))^(1/3)                    (operation count)
#   P_wall  = P_count * (rate_local / rate_ldiv)^(1/3)   (measured rates)
#
# where N is the global extent of the direction and each rate is time per
# counted operation in this run. P_wall assumes the per-operation rate of the
# dense solve stays what it is at this P. It does not: the reduced matrix
# (2qP)^2 doubles in size per doubling of P and leaves L1 then L2, so a
# P_wall taken far from the measured P is indicative only. Measure at the
# target P.
#
# Usage: positional global edge, then `key=value` options:
#
#   julia --project=. -t 1 bench/reducedsolve.jl 96
#
# and under MPI, which is the configuration the probe exists for:
#
#   MPIEXEC=$(julia --project=. -e 'using MPI; MPI.mpiexec(c -> print(c))')
#   "$MPIEXEC" -n 8 julia --project=. -t 1 bench/reducedsolve.jl 96 dims=2,2,2
#   "$MPIEXEC" -n 8 julia --project=. -t 1 bench/reducedsolve.jl 96 dims=8,1,1
#
# `bench/slurm/s12_reducedsolve.sbatch` is the cluster card. Precompile
# serially in the exact environment before any `mpiexec`: every checkout
# shares one depot pidfile, and a rank that hangs holding it blocks every
# Julia process on the machine without a symptom.
#
# Options:
#   N       global grid edge, or a comma-separated list of edges run in turn
#           (default 96). Every direction has N points; each block needs at
#           least 9 per direction for the C8 filter the solver plans, so
#           N >= 9 * dims[d].
#   dims    process grid `a,b,c` (default: `Dims_create`). A slab `P,1,1`
#           puts every rank on one direction and gives the largest P per
#           direction at a given rank count.
#   derivs  comma-separated list of c6, c8, c10 (default c6,c10: one
#           tridiagonal and one pentadiagonal solver).
#   reps    timed calls per stage per rank (default 50), after `warmup`
#           untimed calls (default 5).
#
# On a workstation under the JLL `mpiexec` nothing pins the ranks, and on a
# hybrid performance/efficiency-core CPU they migrate between core types, so
# a workstation table is a smoke test and not the crossover. Compare stages
# within one run, not across runs.
#
# Prints tables and asserts nothing. Rows prefixed `row,` are machine-readable
# for pooling several processes.

using MPI
MPI.Init(threadlevel=:funneled)
using CompactLES
using Printf
using Statistics
using LinearAlgebra: BLAS

const CL = CompactLES
const per3 = ntuple(_ -> (PeriodicBC(), PeriodicBC()), 3)

const DEFAULTS = (N = "96", dims = "", derivs = "c6,c10", reps = 50, warmup = 5)

function deriv_scheme(name)
    name == "c6" && return lele_d1_6(Float64)
    name == "c8" && return lele_d1_8(Float64)
    name == "c10" && return lele_d1_10(Float64)
    error("deriv must be c6, c8 or c10; got '$name'")
end

function parse_dims(spec)
    isempty(strip(spec)) && return nothing
    f = split(strip(spec), ',')
    length(f) == 3 || error("dims must be 'a,b,c'; got '$spec'")
    return ntuple(i -> parse(Int, strip(f[i])), 3)
end

bandwidth(::CL.LineSolver) = 1
bandwidth(ls::CL.BandLineSolver) = ls.q

"""
Median over `reps` calls of `f`, each preceded by `reset()` outside the timer
and by a barrier on `comm`, after `warmup` untimed calls. Collective: every
rank of `comm` must call it with the same counts.
"""
function stage_time(f, reset, comm, reps, warmup)
    for _ in 1:warmup
        reset(); MPI.Barrier(comm); f()
    end
    t = Vector{Float64}(undef, reps)
    for r in 1:reps
        reset()
        MPI.Barrier(comm)
        t0 = time_ns()
        f()
        t[r] = (time_ns() - t0) * 1e-9
    end
    return median(t)
end

"""
Time the stages of the line solve of `plan` on field `f`, with `out` as the
`apply_along!` output. Collective over the plan's sub-communicator; returns
per-rank medians in seconds.
"""
function time_plan(plan, out, f, decomp, reps, warmup)
    d = plan.dim
    ls = plan.line_solver
    comm = ls.comm
    q = bandwidth(ls)
    L = plan.lines
    B = plan.B
    solve! = plan.tr ? CL.solve_lines_t! : CL.solve_lines!
    fill!(B, 0)
    if plan.tr
        CL._fill_t!(B, plan, f, decomp, Val(d))
    else
        CL._fill_lines!(B, plan, f, decomp, Val(1))
    end
    B0 = copy(B)
    restore_B = () -> copyto!(B, B0)
    nothing_to_reset = () -> nothing

    total = stage_time(() -> solve!(B, ls), restore_B, comm, reps, warmup)
    # The same call with the reduced stage switched off is the sweep alone.
    # `hasred` is rank-independent here (every direction is periodic), so no
    # rank skips a collective another rank enters.
    hasred = ls.hasred
    ls.hasred = false
    sweep = try
        stage_time(() -> solve!(B, ls), restore_B, comm, reps, warmup)
    finally
        ls.hasred = hasred
    end
    hasred || return (; total, sweep, reduced = 0.0, allgather = 0.0,
                      ldiv = 0.0, apply = 0.0, q, L)

    # A full solve leaves the interface ends of the fill in `ls.ends`;
    # `_reduced_solve!` reads them and does not write them, so it repeats on
    # the same data without a reset.
    restore_B(); solve!(B, ls)
    reduced = stage_time(() -> CL._reduced_solve!(ls, L), nothing_to_reset,
                         comm, reps, warmup)
    allgather = if ls.P > 1
        buf = MPI.UBuffer(vec(ls.gath), 2q * L)
        stage_time(() -> MPI.Allgather!(ls.ends, buf, comm), nothing_to_reset,
                   comm, reps, warmup)
    else
        0.0
    end
    # `ldiv!` overwrites its right-hand side, so the reduced solution of the
    # last call is restored before each one: realistic magnitudes, no drift.
    z0 = copy(ls.z)
    red = ls.red
    ldiv = stage_time(() -> CL._reduced_ldiv!(red, ls.z, L, L),
                      () -> copyto!(ls.z, z0), comm, reps, warmup)
    CL._reduced_solve!(ls, L)
    apply = stage_time(() -> CL.apply_along!(out, plan, f, decomp),
                       nothing_to_reset, comm, reps, warmup)
    return (; total, sweep, reduced, allgather, ldiv, apply, q, L)
end

function environment(comm, opt)
    rank, nranks = MPI.Comm_rank(comm), MPI.Comm_size(comm)
    hosts = MPI.gather(gethostname(), comm; root = 0)
    rank == 0 || return nothing
    per_host = [count(==(h), hosts) for h in unique(hosts)]
    models = unique(strip(c.model) for c in Sys.cpu_info())
    cores = try string(CL.ThreadPinning.ncores()) catch; "?" end
    binary = try string(MPI.MPIPreferences.binary) catch; "unknown" end
    println("=== distributed line solve, stage by stage (ROADMAP S12)")
    println("    CPU             : ", join(models, " | "), "  (",
            Sys.CPU_THREADS, " logical, ", cores, " physical",
            length(models) > 1 ? ", NON-UNIFORM CORE MODELS" : "", ")")
    println("    nodes           : ", length(per_host), ", ranks per node ",
            minimum(per_host), "-", maximum(per_host), ", ", nranks, " rank(s)")
    println("    MPI library     : ", MPI.MPI_LIBRARY, " ", MPI.MPI_LIBRARY_VERSION,
            ", binary ", binary,
            endswith(binary, "_jll") && length(per_host) > 1 ?
            "   <-- BUNDLED JLL ON A MULTI-NODE RUN" : "")
    println("    Julia           : ", VERSION, ", ", Threads.nthreads(),
            " thread(s) per rank, ", BLAS.get_num_threads(), " BLAS thread(s)",
            BLAS.get_num_threads() > 1 ? "   <-- set OPENBLAS_NUM_THREADS=1" : "")
    Sys.iswindows() && nranks > 1 &&
        println("    affinity        : none (JLL mpiexec on Windows); ranks migrate")
    println("    timing          : median of ", opt.reps, " calls after ", opt.warmup,
            " warm-up, max over ranks; times in microseconds")
    flush(stdout)
    return nothing
end

function main(opt)
    comm = MPI.COMM_WORLD
    root = MPI.Comm_rank(comm) == 0
    say(fmt, args...) = root && print(Printf.format(Printf.Format(fmt), args...))
    environment(comm, opt)
    Ns = parse.(Int, strip.(split(opt.N, ',')))
    derivs = String.(strip.(split(opt.derivs, ',')))
    dims = parse_dims(opt.dims)
    for N in Ns, deriv in derivs
        solver = Solver(n_global = (N, N, N), L_domain = (2π, 2π, 2π), bcs = per3,
                        deriv = deriv_scheme(deriv), dims = dims)
        decomp = solver.decomp
        Q = allocate_state(solver)
        initialize!(solver, Q, (x, y, z) -> Prim(
            u = (sin(x) * cos(y) * cos(z), -cos(x) * sin(y) * cos(z), 0.0),
            p = 100.0 + (cos(2x) + cos(2y)) / 16, rho = 1.0))
        CL.exchange_state!(Q, decomp)
        CL.primitives!(solver, Q)
        f = solver.u
        say("\n  %s, %d^3, process grid %s\n", deriv, N, string(decomp.dims))
        say("  dir  P    n      L  q    9qn  8q2P2  count   sweep   total" *
            "  allgath    ldiv reduced   local   apply  ldiv/loc P_count P_wall\n")
        for d in 1:3
            plan = solver.deriv_plans[d]
            r = time_plan(plan, solver.tmp_a, f, decomp, opt.reps, opt.warmup)
            v = MPI.Allreduce([r.total, r.sweep, r.reduced, r.allgather,
                               r.ldiv, r.apply], MPI.MAX, comm)
            total, sweep, reduced, allgather, ldiv, apply = 1e6 .* v
            P, n, q, L = decomp.dims[d], plan.n, r.q, r.L
            c_local, c_red = 9q * n, 8q^2 * P^2
            loc = total - reduced
            p_count = cbrt(9N / (8q))
            # Measured time per counted operation, local over dense solve.
            p_wall = ldiv > 0 ? p_count * cbrt((loc / c_local) / (ldiv / c_red)) : NaN
            say("  %3d %2d %4d %6d %2d %6d %6d %6.3f %7.1f %7.1f %8.1f %7.1f " *
                "%7.1f %7.1f %7.1f %8.3f %7.2f %6.2f\n",
                d, P, n, L, q, c_local, c_red, c_red / c_local, sweep, total,
                allgather, ldiv, reduced, loc, apply, ldiv / loc, p_count, p_wall)
            say("row,%s,%d,%d,%d,%d,%d,%d,%d,%d,%.4e,%.4e,%.4e,%.4e,%.4e,%.4e\n",
                deriv, N, MPI.Comm_size(comm), Threads.nthreads(), d, P, n, L, q,
                sweep, total, allgather, ldiv, reduced, apply)
            root && flush(stdout)
        end
        CL.free_communicators!(decomp)
    end
    return nothing
end

const _opt = CompactLES.script_args(ARGS, DEFAULTS; positional = (:N,))
mpi_main(() -> main(_opt))
