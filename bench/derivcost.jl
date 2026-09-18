# Wall-clock cost per grid point per step of the three derivative operators,
# `lele_d1_6`, `lele_d1_8` and `lele_d1_10`, on one configuration where every
# line solve is periodic and one where two of them are closed.
#
# The operators differ in bandwidth (C6 and C8 are tridiagonal, C10 pentadiagonal)
# and in explicit right-hand-side width, so their cost separates in the compact
# line solves and nowhere else. Everything outside the derivative is held fixed:
# single species, ideal gas, `compact_filter(0.45)` every step, artificial
# properties at their defaults, `cfl = 0.5`, Float64.
#
# The timed quantity is `solver.wall_total` over a fixed number of steps, which
# is the run loop's own accounting: it spans `max_rate`, the boundary
# conditions, the stages and the filter pass, and excludes callbacks and setup.
# A warm-up `run!` of a few steps precedes every timed cell, so no compile time
# is inside the measurement and the artificial coefficients are primed.
#
# Under `mpiexec` the reported wall time is the maximum over ranks of that
# quantity, taken in one `Allreduce`, and the cost per point divides it by the
# global point count. A rank that finishes its share early waits inside the next
# collective, so the slowest rank is the step time the run actually pays.
# Rank 0 does all the printing. At one rank the reduction is the identity and
# the output is the serial output.
#
# Each cell is built, warmed and timed independently, so the operators may be
# batched into one process. Ordering bias is checked by running a process with
# `derivs=c10,c8,c6`: if the reversed order reproduces the forward order within
# the run-to-run spread, the batching is sound.
#
# Run-to-run spread on a desktop is 10-20% (CLAUDE.md, Timing noise), so take
# the median of at least three processes rather than one number.
#
# Usage: positional grid and step count, then `key=value` options:
#
#   julia --project=. -t 16 bench/derivcost.jl
#   julia --project=. -t 16 bench/derivcost.jl 64 30 derivs=c10,c8,c6
#   julia --project=. -t 1  bench/derivcost.jl 64 20 cases=periodic
#   julia --project=. -t 16 bench/derivcost.jl 64 30 phases=true cases=periodic
#
# and under MPI, which is the production configuration (one thread per rank):
#
#   MPIEXEC=$(julia --project=. -e 'using MPI; MPI.mpiexec(c -> print(c))')
#   "$MPIEXEC" -n 8 julia --project=. -t 1 bench/derivcost.jl 128 30
#   "$MPIEXEC" -n 8 julia --project=. -t 1 bench/derivcost.jl 128 30 dims=2,2,2
#
# Precompile serially in this exact environment before launching `mpiexec`: the
# precompile lock is one pidfile shared by every checkout, and a rank that hangs
# holding it blocks every other Julia process on the machine without a symptom.
#
# Options:
#   N         cubic grid edge (default 64). 64^3 is a few seconds per cell at
#             -t 16 and gives a ratio resolved well below the spread. Under MPI
#             this is the global edge, so the per-rank block is N^3 / nranks.
#   steps     timed steps per cell (default 30).
#   warmup    untimed steps per cell before timing (default 3). Three is enough
#             to compile the stepped path and prime the coefficient arrays; the
#             first `run!` of a process carries the compilation of every phase.
#   derivs    comma-separated list of c6, c8, c10 (default all three), in the
#             order they are run. Reverse it to test for ordering bias.
#   cases     comma-separated list of periodic (fully periodic box) and
#             slipwall (`SlipWallBC` at both ends of x, periodic in y and z,
#             so the closure rows and the closed line solve are exercised).
#             Default both.
#   dims      process grid as `a,b,c`, empty (default) to let MPI distribute the
#             ranks. Give it explicitly whenever two runs must decompose the
#             same way: `Dims_create` is free to return a different factorization
#             for a different rank count, and a 1 in a direction removes that
#             direction's distributed line solve altogether.
#   cfl       timestep multiplier (default 0.5). Cost per step is independent
#             of it; it only sets how far the run travels.
#   reps      repetitions of the whole matrix inside this process (default 1).
#             Separate processes are the better repetition, since a process
#             holds one heap and one set of compiled code.
#   phases    also time the phases of one right-hand-side evaluation per operator
#             (default false), so the cost difference can be attributed. The
#             phase timings are minima over repeated calls on a settled state,
#             reduced with a maximum over ranks like the step time, not a share
#             of the timed run, and the four phases inside `compute_rhs!` are
#             reported as a share of it. Every phase timed is collective and
#             every rank runs the same number of repeats, so the timers are
#             valid under MPI unchanged.
#
# The initial condition is the Taylor-Green field. It is mirror-symmetric about
# x = 0 and x = 2pi (the normal velocity vanishes there and the pressure is
# even), so the same state is admissible under both boundary configurations and
# the two cases differ only in the scheme's treatment of the x lines.
#
# Prints tables and asserts nothing. Rows prefixed `row,` are machine-readable
# for aggregating several processes.

using MPI
MPI.Init(threadlevel=:funneled)
using CompactLES
using Printf
using Statistics
using LinearAlgebra: BLAS

const CL = CompactLES
const per3 = ntuple(_ -> (PeriodicBC(), PeriodicBC()), 3)

const DEFAULTS = (N = 64, steps = 30, warmup = 3,
                  derivs = "c6,c8,c10", cases = "periodic,slipwall",
                  dims = "", cfl = 0.5, reps = 1, phases = false)

function deriv_scheme(name)
    name == "c6" && return lele_d1_6(Float64)
    name == "c8" && return lele_d1_8(Float64)
    name == "c10" && return lele_d1_10(Float64)
    error("deriv must be c6, c8 or c10; got '$name'")
end

function case_bcs(name)
    name == "periodic" && return per3
    name == "slipwall" &&
        return ((SlipWallBC(), SlipWallBC()), per3[2], per3[3])
    error("case must be periodic or slipwall; got '$name'")
end

function parse_dims(spec)
    isempty(strip(spec)) && return nothing
    f = split(strip(spec), ',')
    length(f) == 3 || error("dims must be 'a,b,c'; got '$spec'")
    return ntuple(i -> parse(Int, strip(f[i])), 3)
end

function build(case, deriv, N, cfl, dims)
    γ = 1.4
    c0 = 10.0                      # Ma ~ 0.1 at |u|max = 1
    p0 = c0^2 / γ
    prob = Problem(eos=IdealSpecies("gas"; R=1.0, gamma=γ),
                   transport=Transport(mu0=1 / 1600),
                   domain=((0.0, 2π), (0.0, 2π), (0.0, 2π)),
                   bcs=case_bcs(case),
                   ic=(x, y, z) -> Prim(
                       u=(sin(x) * cos(y) * cos(z),
                          -cos(x) * sin(y) * cos(z), 0.0),
                       p=p0 + (cos(2x) + cos(2y)) * (cos(2z) + 2) / 16,
                       rho=1.0))
    return setup(prob, Numerics(n_global=(N, N, N), cfl=cfl,
                                deriv=deriv_scheme(deriv),
                                filt=compact_filter(0.45),
                                art=ArtParams(), dims=dims))
end

# Minimum over repeated calls: the phase timings below compare stencils on a
# settled state, where the minimum is the estimator with the least noise. The
# maximum over ranks then reports the slowest rank's own minimum, the same
# estimator the step time uses.
function best(f, comm; reps=20)
    f(); f()
    t = minimum(@elapsed(f()) for _ in 1:reps)
    return MPI.Allreduce(t, MPI.MAX, comm)
end

"""
Phases of one right-hand-side evaluation, so a cost difference between operators
can be attributed to the derivative solves rather than read as a whole-step
number. The velocity and scalar gradient passes are where the scheme's bandwidth
enters; the artificial pass and the filter are timed beside them as the phases
the operator does not change.

Every call here is collective and the repeat counts are rank-independent, so
this runs under `mpiexec` unchanged.
"""
function phase_split(solver, Q)
    comm = solver.comm
    dQ = zero(Q)
    vel = (solver.u, solver.v, solver.w)
    CL.exchange_state!(Q, solver.decomp)
    CL.primitives!(solver, Q)
    t = Pair{String,Float64}[]
    push!(t, "velocity grads" => best(comm) do
        for jj in 1:3, d in 1:3
            CL.deriv_scaled_along!(solver.grad_u[d, jj], vel[jj], solver, d,
                                   CL.vel_parity(solver, d, jj))
        end
    end)
    push!(t, "scalar grads" => best(comm) do
        for d in 1:3
            CL.deriv_scaled_along!(solver.grad_T_ion[d], solver.T_ion, solver, d, 1)
            for sp in 1:solver.equations.n_species
                CL.deriv_scaled_along!(solver.grad_Y[d, sp], solver.Y[sp], solver, d, 1)
            end
        end
    end)
    push!(t, "artificial" => best(() -> CL.compute_artificial!(solver, Q), comm))
    push!(t, "assemble_fluxes!" => best(() -> CL.assemble_fluxes!(solver, Q), comm))
    push!(t, "compute_rhs!" => best(() -> compute_rhs!(solver, Q, dQ), comm))
    push!(t, "filter_state!" => best(() -> CL.filter_state!(solver, Q), comm,
                                     reps=10))
    return t
end

"""
Build, warm and time one (case, deriv) cell. The returned wall time, allocation
and footprint are maxima over the ranks; `ns_per_point` divides that wall time
by the global point count, so it is the cost the whole run pays per point.
"""
function time_cell(case, deriv, N, steps, warmup, cfl, dims, want_phases)
    solver = Q = nothing
    setup_s = @elapsed ((solver, Q) = build(case, deriv, N, cfl, dims))
    comm = solver.comm
    ws = Workspace(Q)
    # The warm-up compiles the stepped path and leaves the coefficient arrays
    # primed, so the timed call sizes its first step the way a long run does.
    run!(solver, Q, ws; tfinal=1e9, nmax=warmup)
    GC.gc()
    MPI.Barrier(comm)
    wall0, step0 = solver.wall_total, solver.step
    bytes = @allocated run!(solver, Q, ws; tfinal=1e9, nmax=step0 + steps)
    nstep = solver.step - step0
    footprint = Base.summarysize(solver) + Base.summarysize(Q) +
                Base.summarysize(ws)
    # One reduction for every per-rank quantity reported. The step time is the
    # slowest rank's, since the others are waiting inside the next collective.
    v = MPI.Allreduce(Float64[solver.wall_total - wall0, bytes, footprint,
                              Sys.maxrss()], MPI.MAX, comm)
    wall, bytes, footprint, maxrss = v[1], v[2], v[3], v[4]
    phases = want_phases ? phase_split(solver, Q) : Pair{String,Float64}[]
    return (; case, deriv, setup_s, nstep, wall,
            per_step = wall / max(nstep, 1),
            ns_per_point = 1e9 * wall / max(nstep, 1) / N^3,
            alloc_per_step = bytes / max(nstep, 1),
            footprint, maxrss, phases)
end

function main(opt)
    comm = MPI.COMM_WORLD
    rank, nranks = MPI.Comm_rank(comm), MPI.Comm_size(comm)
    root = rank == 0
    # A runtime format, not `@printf`: the macro takes a literal format string,
    # and every line below is printed only on rank 0.
    say(fmt, args...) = root && print(Printf.format(Printf.Format(fmt), args...))
    derivs = String.(strip.(split(opt.derivs, ',')))
    cases = String.(strip.(split(opt.cases, ',')))
    dims = parse_dims(opt.dims)
    N = opt.N
    say("=== derivative operator cost: %d^3, %d steps, %d warm-up, cfl %.3g\n",
        N, opt.steps, opt.warmup, opt.cfl)
    say("    %s, %d logical CPUs, %d rank(s), %d Julia thread(s), %d BLAS thread(s)\n",
        Sys.CPU_NAME, Sys.CPU_THREADS, nranks, Threads.nthreads(),
        BLAS.get_num_threads())
    say("    process grid %s, order %s, cases %s, %d rep(s) in this process\n",
        dims === nothing ? "auto" : string(dims), join(derivs, " "),
        join(cases, " "), opt.reps)
    root && println()
    root && println("  case       deriv  steps   s/step     ns/pt/step  " *
                    "alloc/step  footprint  setup")
    rows = Dict{Tuple{String,String},Vector{Float64}}()
    for rep in 1:opt.reps, case in cases, deriv in derivs
        r = time_cell(case, deriv, N, opt.steps, opt.warmup, opt.cfl, dims,
                      opt.phases && rep == 1)
        push!(get!(rows, (case, deriv), Float64[]), r.per_step)
        say("  %-10s %-6s %5d  %9.5f   %9.2f  %8.1f KiB  %6.1f MiB %5.1f s\n",
            r.case, r.deriv, r.nstep, r.per_step, r.ns_per_point,
            r.alloc_per_step / 1024, r.footprint / 2.0^20, r.setup_s)
        # Machine-readable, for pooling several processes.
        say("row,%s,%s,%d,%d,%d,%d,%.6e,%.4f,%.1f,%.1f\n",
            r.case, r.deriv, nranks, Threads.nthreads(), N, r.nstep,
            r.per_step, r.ns_per_point, r.alloc_per_step,
            r.footprint / 2.0^20)
        if root && !isempty(r.phases)
            # The first four phases are components of `compute_rhs!` and the
            # share is taken against it; `filter_state!` sits outside it. They
            # do not sum to the whole, so no total is printed.
            whole = last(r.phases[findfirst(x -> first(x) == "compute_rhs!",
                                            r.phases)])
            println("      phase (minimum over repeats)      ms   % of rhs   ns/pt")
            for (k, v) in r.phases
                @printf("      %-20s %10.3f %8.1f%% %8.2f\n",
                        k, 1e3v, 100v / whole, 1e9v / N^3)
            end
        end
        root && flush(stdout)
    end
    say("\n  peak resident set, largest rank: %.1f MiB\n",
        MPI.Allreduce(Float64(Sys.maxrss()), MPI.MAX, comm) / 2.0^20)
    opt.reps > 1 || return
    root && println("\n  medians over $(opt.reps) in-process rep(s), and ratio to c6")
    for case in cases
        base = haskey(rows, (case, "c6")) ? median(rows[(case, "c6")]) : NaN
        for deriv in derivs
            v = median(rows[(case, deriv)])
            say("  %-10s %-6s %9.5f s/step   %9.2f ns/pt   %.3fx\n",
                case, deriv, v, 1e9v / N^3, v / base)
        end
    end
end

const _opt = CompactLES.script_args(ARGS, DEFAULTS; positional = (:N, :steps))
mpi_main(() -> main(_opt))
