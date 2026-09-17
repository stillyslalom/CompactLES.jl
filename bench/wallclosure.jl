# The wall-closure qualification: the Brady–Livescu derivative rows as a
# bounded supported wall configuration under the default one-sided filter
# rows, beside the default cascade they would replace.
#
#   julia --project=. -t 1 bench/wallclosure.jl [parts=all]
#   julia --project=. -t 1 bench/wallclosure.jl parts=cfl,start
#   mpiexec -n 2 julia --project=. -t 1 bench/wallclosure.jl parts=extent
#
# Parts:
#
#   smooth   the smooth wall cases of test/smooth_cases.jl with the
#            artificial properties on, so the update at the wall is the
#            complete one, D(β* D) included: the inviscid and the viscous
#            adiabatic standing wave, the same wave with a tangential shear
#            between viscous slip walls, and the shear mode between adiabatic
#            and between isothermal no-slip walls, every closure under the
#            default filter every step, against the periodic mirror under
#            the same settings (the closure defect alone), with the
#            artificial properties off beside them; then the reflected
#            pulse of test/cases.jl, smooth and steepening, every closure
#   channels which artificial-property channel carries the wall defect the
#            smooth part shows once the properties are on: the inviscid
#            wall under C6 Brady–Livescu with one constant at a time, the
#            constants zeroed but the machinery enabled, the other smoother,
#            detector and sensor fields, with the default closure beside it
#   cfl      the stable CFL range: the inviscid wall with the artificial
#            properties on, the steepening pulse, Woodward–Colella and the
#            warm-started planar Noh, every closure, on a CFL ladder, with
#            the periodic mirror on the same ladder where one exists so a
#            failure is attributed to the rows or to the interior
#   start    how resolved a wall start must be: planar Noh from the exact
#            solution at t0 down to the singular start, with and without
#            retries
#   floor    the round-off floor: one derivative of a smooth field up to
#            N = 3073 in Float64, and the pulse against its mirror in
#            Float32, every closure
#   extent   the minimum block extent each closure set and the filter admit,
#            probed through plan_direction with both ends closed (one rank)
#            or one end closed (two ranks); the one part that runs under
#            mpiexec
#   plane    the two-dimensional Cartesian Noh plane, inflow on four faces
#            and corners, cold and warm-started, every closure
#
# Scratch tooling, like everything else in bench/: it prints tables and
# asserts nothing. The conclusions are written up in
# reference/CALIBRATION_APPENDIX.md and the decision in reference/CALIBRATION.md.

using MPI
MPI.Init(threadlevel=:funneled)
using CompactLES
using Printf

const CL = CompactLES
include(joinpath(@__DIR__, "..", "test", "references.jl"))
include(joinpath(@__DIR__, "..", "test", "cases.jl"))
include(joinpath(@__DIR__, "..", "test", "smooth_cases.jl"))

const OPTS = CL.script_args(ARGS, (parts="all",))
const PARTS = OPTS.parts == "all" ?
    ["smooth", "channels", "cfl", "start", "floor", "plane", "extent"] :
    split(OPTS.parts, ',')
const NP = MPI.Comm_size(MPI.COMM_WORLD)
const RANK = MPI.Comm_rank(MPI.COMM_WORLD)
NP == 1 || PARTS == ["extent"] || error("only parts=extent runs on more than one rank")

const CAP = 30_000
const REFDIR = joinpath(@__DIR__, "..", "test", "refs")
const NS = (49, 97, 193)
const TFINAL = 0.4
const CFL = 0.25

const CLOSURES = (("C6 neutral3", T -> lele_d1_6(T)),
                  ("C6 cascade3", T -> lele_d1_6(T; closures=:cascade3)),
                  ("C6 BL", T -> lele_d1_6(T; closures=:brady_livescu)),
                  ("C8 BL", T -> lele_d1_8(T; closures=:brady_livescu)))
const FILTERED = (filter_interval=1, filt=compact_filter(0.45), filter_cfl=0.35)
const ART_ON = ArtParams(enabled=true)
const ART_OFF = ArtParams(enabled=false)

sprintf(fmt::String, args...) = Printf.format(Printf.Format(fmt), args...)
printf(fmt::String, args...) = print(sprintf(fmt, args...))
pad(s, n) = rpad(s, n)
orders_string(hs, es) =
    join((@sprintf("%.2f", o) for o in successive_orders(hs, es)), " / ")

# A configuration that loses positivity raises `SolverFailure` from `run!`;
# the row then reads the failure and the study goes on. No other exception
# is caught.
function attempt(f)
    try
        return f()
    catch err
        err isa SolverFailure || rethrow()
        return sprintf("FAILED %s at step %d, t = %.4f", err.reason, err.step, err.t)
    end
end
failed(r) = r isa String

function read_ref(name)
    cols = [Float64[] for _ in 1:4]
    for line in eachline(joinpath(REFDIR, name))
        (isempty(line) || startswith(line, '#')) && continue
        for (c, tok) in enumerate(split(line, ','))
            push!(cols[c], parse(Float64, tok))
        end
    end
    Tuple(cols)
end

# --- part: smooth -------------------------------------------------------------
#
# A wall run against its periodic mirror under the same settings, at the
# same step: the closure defect, derivative and filter rows together, and
# nothing else, whether or not the artificial properties are on (their
# sensors and smoothers are symmetric about the walls, so the mirror is
# still the wall solution).

function wall_vs_mirror(N, deriv; viscous, art, slip=!viscous, c=0.0, cfl=CFL,
                        tfinal=TFINAL)
    attempt() do
        opts = (deriv=deriv, art=art, cfl=cfl, FILTERED...)
        solver, Q = wall_case(N; viscous=viscous, slip=slip, c=c, opts...)
        run!(solver, Q; tfinal=tfinal, nmax=CAP)
        mirror, Qm = mirror_case(N; viscous=viscous, c=c, opts...)
        run!(mirror, Qm; tfinal=tfinal, nmax=CAP)
        regional_errors(solver, Q, NodeReference(mirror, Qm))
    end
end

function shear_vs_mirror(N, deriv; art, Twall=NaN, cfl=CFL, tfinal=TFINAL)
    attempt() do
        opts = (deriv=deriv, art=art, cfl=cfl, FILTERED...)
        solver, Q = shear_case(N; Twall=Twall, opts...)
        run!(solver, Q; tfinal=tfinal, nmax=CAP)
        mirror, Qm = shear_mirror_case(N; opts...)
        run!(mirror, Qm; tfinal=tfinal, nmax=CAP)
        regional_errors(solver, Q, NodeReference(mirror, Qm); comp=3)
    end
end

function smooth_table(title, rowfn)
    println("\n--- $title ---")
    println("  closure       art   N     wall       interior   l2         orders (wall)")
    for (label, mk) in CLOSURES, (alab, art) in (("off", ART_OFF), ("on", ART_ON))
        hs = Float64[]; es = Float64[]
        for N in NS
            r = rowfn(N, mk(Float64), art)
            if failed(r)
                println("  ", pad(label, 14), pad(alab, 6), sprintf("%4d  ", N), r)
                continue
            end
            printf("  %-14s%-6s%4d  %.3e  %.3e  %.3e\n", label, alab, N, r.wall,
                   r.interior, r.l2)
            push!(hs, 1 / (N - 1)); push!(es, r.wall)
        end
        length(es) >= 2 && printf("  %-24s      orders %s\n", "", orders_string(hs, es))
        flush(stdout)
    end
end

function pulse_vs_mirror(::Type{T}, N, deriv; amp, art, tfinal=0.7, cfl=0.4) where {T}
    attempt() do
        kw = (amp=amp, art=art, deriv=deriv, cfl=cfl,
              filt=compact_filter(T(0.45), T))
        solver, Q = pulse_case(T, N; kw...)
        run!(solver, Q; tfinal=T(tfinal), nmax=CAP)
        mirror, Qm = pulse_case(T, N; mirror=true, kw...)
        run!(mirror, Qm; tfinal=T(tfinal), nmax=CAP)
        abs(solver.t - mirror.t) < 1e-6 ||
            return sprintf("clocks differ %.3e", solver.t - mirror.t)
        a = case_line_component(solver, Q, 1)
        b = case_line_component(mirror, Qm, 1)
        wall, interior, l2 = mirror_line_errors(a, b)
        rep = state_report(solver, Q)
        (wall=wall, interior=interior, l2=l2, steps=solver.step,
         inadmissible=rep.inadmissible, rho_min=rep.rho_min)
    end
end

function pulse_table(::Type{T}, ns; amp, art, tfinal=0.7, cfl=0.4) where {T}
    println("  closure       N     wall       interior   l2         steps  inadm  rho_min")
    for (label, mk) in CLOSURES
        hs = Float64[]; es = Float64[]
        for N in ns
            r = pulse_vs_mirror(T, N, mk(T); amp=amp, art=art, tfinal=tfinal, cfl=cfl)
            if failed(r)
                println("  ", pad(label, 14), sprintf("%4d  ", N), r)
                continue
            end
            printf("  %-14s%4d  %.3e  %.3e  %.3e  %5d  %5d  %.4f\n", label, N, r.wall,
                   r.interior, r.l2, r.steps, r.inadmissible, r.rho_min)
            push!(hs, 1 / (N - 1)); push!(es, r.interior)
        end
        length(es) >= 2 && printf("  %-14s      interior orders %s\n", "",
                                  orders_string(hs, es))
        flush(stdout)
    end
end

function smooth_part()
    println("\n=== smooth walls against the mirror, default filter every step, " *
            "cfl = $CFL, t = $TFINAL ===")
    smooth_table("inviscid slip walls, density",
                 (N, d, art) -> wall_vs_mirror(N, d; viscous=false, art=art))
    smooth_table("viscous adiabatic no-slip walls, density",
                 (N, d, art) -> wall_vs_mirror(N, d; viscous=true, art=art))
    smooth_table("viscous adiabatic slip walls with a tangential shear, density",
                 (N, d, art) -> wall_vs_mirror(N, d; viscous=true, slip=true,
                                               c=0.05, art=art))
    smooth_table("shear mode, adiabatic no-slip walls, rho v",
                 (N, d, art) -> shear_vs_mirror(N, d; art=art))
    smooth_table("shear mode, isothermal no-slip walls (Twall = 1), rho v",
                 (N, d, art) -> shear_vs_mirror(N, d; art=art, Twall=1.0))
    println("\n=== the reflected pulse against its mirror (density, t = 0.7, cfl = 0.4) ===")
    println("\n--- amp 0.01, artificial properties on ---")
    pulse_table(Float64, (49, 97, 193, 385); amp=0.01, art=true)
    println("\n--- amp 0.1, artificial properties on (shocked after the reflection) ---")
    pulse_table(Float64, (97, 193, 385, 769); amp=0.1, art=true)
end

# --- part: channels -----------------------------------------------------------
#
# The smooth part's rows with the properties on differ from the rows with
# them off only through the artificial terms, whose sensors are computed
# through the closed edge's own treatment (the detector clamps a closed
# edge, `delta4_sum!`) while the mirror sees the exact even extension. One
# constant at a time attributes the defect to a channel; the constants
# zeroed with the machinery enabled separate the coefficient arithmetic
# from the sensors.

function channels_part()
    variants = (("off", ART_OFF),
                ("all on", ART_ON),
                ("C_mu only", ArtParams(C_beta=0.0, C_kappa=0.0)),
                ("C_beta only", ArtParams(C_mu=0.0, C_kappa=0.0)),
                ("C_kappa only", ArtParams(C_mu=0.0, C_beta=0.0)),
                ("all zero, enabled", ArtParams(C_mu=0.0, C_beta=0.0, C_kappa=0.0)),
                ("all on, smoother=:compact", ArtParams(smoother=:compact)),
                ("all on, detector=:d8", ArtParams(detector=:d8)),
                ("all on, mu_sensor=:velocity", ArtParams(mu_sensor=:velocity)),
                ("all on, beta_sensor=:dilatation", ArtParams(beta_sensor=:dilatation)))
    println("\n=== the artificial-property channels at an inviscid wall, against the " *
            "mirror, cfl = $CFL, t = $TFINAL ===")
    for (label, mk) in (CLOSURES[3], CLOSURES[1], CLOSURES[2])
        println("\n--- $label ---")
        println("  variant                          N     wall       interior   l2")
        for (vlabel, art) in variants
            hs = Float64[]; es = Float64[]
            for N in NS
                r = wall_vs_mirror(N, mk(Float64); viscous=false, art=art)
                if failed(r)
                    println("  ", pad(vlabel, 33), sprintf("%4d  ", N), r)
                    continue
                end
                printf("  %-32s %4d  %.3e  %.3e  %.3e\n", vlabel, N, r.wall, r.interior,
                       r.l2)
                push!(hs, 1 / (N - 1)); push!(es, r.wall)
            end
            length(es) >= 2 && printf("  %-32s       orders %s\n", "",
                                      orders_string(hs, es))
            flush(stdout)
        end
    end
end

# --- part: cfl ----------------------------------------------------------------

function cfl_part()
    println("\n=== the stable CFL range ===")
    println("\n--- inviscid wall N = 97, artificial properties on, t = $TFINAL, " *
            "against the mirror at the same cfl ---")
    println("  closure       cfl    steps  wall       interior   l2         | mirror alone")
    for (label, mk) in CLOSURES, cfl in (0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0)
        deriv = mk(Float64)
        opts = (deriv=deriv, art=ART_ON, cfl=cfl, FILTERED...)
        m = attempt() do
            mirror, Qm = mirror_case(97; opts...)
            run!(mirror, Qm; tfinal=TFINAL, nmax=CAP)
            (mirror, Qm)
        end
        r = attempt() do
            solver, Q = wall_case(97; opts...)
            run!(solver, Q; tfinal=TFINAL, nmax=CAP)
            failed(m) && return (steps=solver.step,)
            (steps=solver.step,
             regional_errors(solver, Q, NodeReference(m[1], m[2]))...)
        end
        mtxt = failed(m) ? m : "completes"
        if failed(r)
            println("  ", pad(label, 14), sprintf("%.2f   ", cfl), r, " | ", mtxt)
        elseif failed(m)
            printf("  %-14s%.2f   %5d  completes                        | %s\n",
                   label, cfl, r.steps, mtxt)
        else
            printf("  %-14s%.2f   %5d  %.3e  %.3e  %.3e  | %s\n", label, cfl,
                   r.steps, r.wall, r.interior, r.l2, mtxt)
        end
        flush(stdout)
    end

    println("\n--- steepening pulse amp 0.1, N = 385, artificial properties on, " *
            "t = 0.7, against the mirror at the same cfl ---")
    println("  closure       cfl    steps  wall       interior   l2         inadm")
    for (label, mk) in CLOSURES, cfl in (0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.5)
        r = pulse_vs_mirror(Float64, 385, mk(Float64); amp=0.1, art=true, cfl=cfl)
        if failed(r)
            println("  ", pad(label, 14), sprintf("%.2f   ", cfl), r)
        else
            printf("  %-14s%.2f   %5d  %.3e  %.3e  %.3e  %d\n", label, cfl, r.steps,
                   r.wall, r.interior, r.l2, r.inadmissible)
        end
        flush(stdout)
    end

    println("\n--- Woodward–Colella N = $WC_N ---")
    xr, ρr, _, _ = read_ref("woodward_colella.csv")
    println("  closure       cfl    L1 rho     peak rho  at       rho_min")
    for (label, mk) in CLOSURES, cfl in (0.15, 0.3, 0.45, 0.6, 0.9, 1.2)
        r = attempt() do
            xs, ρ, _, _, ok = woodward(; deriv=mk(Float64), cfl=cfl, nmax=CAP)
            ok || return "step cap"
            imax = argmax(ρ)
            sprintf("%.3e  %.4f    %.4f   %.4f", l1(ρ, [interp1(xr, ρr, x) for x in xs]),
                    ρ[imax], xs[imax], minimum(ρ))
        end
        println("  ", pad(label, 14), sprintf("%.2f   ", cfl), r)
        flush(stdout)
    end

    println("\n--- planar Noh N = 400 warm-started at t0 = 0.3 ---")
    println("  closure       cfl    rho[1:4]                       plateau  deficit  shock")
    for (label, mk) in CLOSURES, cfl in (0.15, 0.3, 0.6, 0.9, 1.2)
        println("  ", pad(label, 14), sprintf("%.2f   ", cfl),
                noh_row(mk(Float64); t0=0.3, cfl=cfl))
        flush(stdout)
    end
end

# --- part: start --------------------------------------------------------------

function noh_row(deriv; t0, cfl=NOH_CFL, retries=0, N=400)
    attempt() do
        num = Numerics(n_global=(N, 1, 1), art=ART_ON, cfl=cfl, deriv=deriv,
                       filt=compact_filter(0.45), filter_interval=1, filter_cfl=0.35,
                       control=StepControl(validity=:permissive, retries=retries))
        solver, Q = setup(noh_problem(1; N=N, t0=t0), num)
        run!(solver, Q; tfinal=NOH_T - t0, nmax=CAP)
        completed(solver, NOH_T - t0) || return "step cap"
        xs, ρ, _, _ = case_line_profile(solver, Q)
        plat, deficit, Rs, _ = noh_metrics(xs, ρ, 1)
        sprintf("%.3f %.3f %.3f %.3f        %.4f   %+3.0f%%     %.4f  (%d steps)",
                ρ[1], ρ[2], ρ[3], ρ[4], plat / 4, 100deficit, Rs, solver.step)
    end
end

function start_part()
    println("\n=== planar Noh N = 400 from the exact solution at t0 (cfl = $NOH_CFL) ===")
    println("the front is at t0/3, the cell is 0.0025, and the blend spans four cells")
    println("  closure       t0      retries  rho[1:4]                       " *
            "plateau  deficit  shock")
    for (label, mk) in CLOSURES, t0 in (0.3, 0.1, 0.03, 0.01, 0.003, 0.0),
        retries in (0, 4)
        (t0 > 0.01 && retries > 0) && continue
        println("  ", pad(label, 14), sprintf("%.3f   %d        ", t0, retries),
                noh_row(mk(Float64); t0=t0, retries=retries))
        flush(stdout)
    end
end

# --- part: floor --------------------------------------------------------------

function floor_part()
    println("\n=== the round-off floor ===")
    println("\n--- one derivative of exp(sin(3x)) on the closed line, Float64, " *
            "actual spacing ---")
    println("  closure       N      wall       interior")
    for (label, mk) in CLOSURES
        deriv = mk(Float64)
        hs = Float64[]; ws = Float64[]
        for N in (193, 385, 769, 1537, 3073)
            e = closed_derivative_errors(N, deriv, x -> exp(sin(3x)),
                                         x -> 3cos(3x) * exp(sin(3x)))
            printf("  %-14s%4d   %.3e  %.3e\n", label, N, e.wall, e.interior)
            push!(hs, 1 / (N - 1)); push!(ws, e.wall)
        end
        printf("  %-14s       wall orders %s\n", "", orders_string(hs, ws))
        flush(stdout)
    end
    println("\n--- the reflected pulse against its mirror in Float32, " *
            "artificial properties on ---")
    println("\namp 0.01, t = 0.7")
    pulse_table(Float32, (49, 97, 193, 385); amp=0.01, art=true)
    println("\namp 0.1, t = 0.7 (shocked after the reflection)")
    pulse_table(Float32, (97, 193, 385); amp=0.1, art=true)
    println("\n--- the same in Float64, for the floor's ratio ---")
    println("\namp 0.01, t = 0.7")
    pulse_table(Float64, (49, 97, 193, 385); amp=0.01, art=true)
end

# --- part: extent -------------------------------------------------------------
#
# `plan_direction` raises when a block is too short for its closure rows,
# so the minimum is read by construction: the smallest local extent along
# a closed dimension that builds a plan. On one rank both ends of the
# dimension are closed; on two ranks split along it each rank holds one
# closed end, which is the decomposed case a wall rank sees.

function extent_part()
    schemes = (("C6 neutral3", lele_d1_6()),
               ("C6 cascade3", lele_d1_6(closures=:cascade3)),
               ("C6 cascade4", lele_d1_6(closures=:cascade4)),
               ("C6 BL", lele_d1_6(closures=:brady_livescu)),
               ("C8 cascade3", lele_d1_8()),
               ("C8 BL", lele_d1_8(closures=:brady_livescu)),
               ("C10", lele_d1_10()),
               ("filter onesided", compact_filter(0.45)),
               ("filter cascade", compact_filter(0.45; closures=:cascade)))
    if RANK == 0
        println("\n=== minimum local extent along a closed dimension, $NP rank(s) " *
                "($(NP == 1 ? "both ends" : "one end") closed per block) ===")
        println("  scheme            min extent   what the next-smaller extent raises")
    end
    for (label, scheme) in schemes
        minimum_n = 0; message = ""
        for n in 3:20
            ok = try
                decomp = CL.Decomp((NP * n, 12, 12), (false, true, true); dims=(NP, 1, 1))
                CL.plan_direction(decomp, scheme, 1, 1.0)
                NP > 1 && CL.free_communicators!(decomp)
                true
            catch err
                err isa ErrorException || rethrow()
                message = err.msg
                false
            end
            if ok
                minimum_n = n
                break
            end
        end
        RANK == 0 && printf("  %-18s%4d         %s\n", label, minimum_n, message)
    end
end

# --- part: plane --------------------------------------------------------------

function plane_part()
    println("\n=== Cartesian Noh plane N = 24, AR = 2 (inflow on four faces) ===")
    println("  closure       t0     plateau  center deficit  front x/y/diag           " *
            "L1 rho  steps")
    for (label, mk) in CLOSURES, t0 in (0.0, 0.3)
        r = attempt() do
            r = noh_cartesian(; N=24, AR=2, t0=t0, deriv=mk(Float64), nmax=CAP)
            r.completed || return "step cap"
            sprintf("%.3f    %+3.0f%%           %.4f %.4f %.4f    %.3f   %d",
                    r.plateau, 100r.deficit, r.cut_x.front, r.cut_y.front,
                    r.cut_diag.front, r.l1, r.steps)
        end
        println("  ", pad(label, 14), sprintf("%.1f    ", t0), r)
        flush(stdout)
    end
end

for part in PARTS
    part == "smooth" ? smooth_part() :
    part == "channels" ? channels_part() :
    part == "cfl" ? cfl_part() :
    part == "start" ? start_part() :
    part == "floor" ? floor_part() :
    part == "extent" ? extent_part() :
    part == "plane" ? plane_part() :
    error("unknown part '$part'; want smooth, channels, cfl, start, floor, extent " *
          "or plane")
end
