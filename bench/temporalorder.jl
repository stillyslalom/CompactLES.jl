# Temporal order of the complete time step: the LSRK(5,4) update with the
# boundary conditions enforced at every stage, time-dependent boundary data,
# the state filter, and the coarse-fine coupling of a two-level nest under
# global and subcycled stepping.
#
#   julia --project=. -t 1 bench/temporalorder.jl [study=all] [restrict=step]
#
# Every row integrates one case on one grid in a sequence of equal steps,
# through `run!` one step at a time (`fixed_step_run!` in
# test/smooth_cases.jl), and reports the maximum difference over the
# conserved components and the uncovered nodes against the same case on the
# same grid in many more steps. The spatial error cancels exactly, so the
# difference is the time integration's alone. Orders are between successive
# step counts. Artificial properties are off throughout.
#
# `study` is one of, or a comma-separated list of,
#
#   rk        the unfiltered update: a periodic standing wave (the interior
#             control), the same wave between slip walls, a supersonic
#             entropy wave entering through a `DirichletBC` that holds the
#             exact state at the stage time (`inflow_case`), and a subsonic
#             stream under an `NSCBCInflowBC` with an oscillating pointwise
#             target (`target_inflow_case`)
#   filter    the state filter once per step on the periodic and wall waves,
#             relaxed and at full strength
#   cfl       the temporal error at a fixed ratio of step to spacing under
#             grid refinement: the periodic entropy wave, the Dirichlet
#             inflow, the same inflow data imposed through their time
#             derivative instead (`RateDirichlet` below), the NSCBC target,
#             and the same NSCBC faces with constant targets and an
#             acoustic pulse in the interior
#   levels    the entropy wave through a two-level nest, global step and
#             subcycled, under the closure rows, the ghost fluxes (the
#             default) and the Brady–Livescu interface rows; the composite
#             error and the fine patch's
#   total     the same nest under the default ghost fluxes against the
#             exact solution, not against a finer step: N = 48, 96 and 192
#             at the default cfl 0.5 and at 0.125, the unrefined root, the
#             global step and subcycled, the fine patch's error and the
#             composite's. Where the two cfl columns agree, no time error
#             of any source is visible in the solution the run delivers.
#
# `restrict` changes the coupling schedule of the `levels` rows, as a
# diagnosis: `step` is the package's (the fine solution is injected into the
# covered parent nodes once per completed step), `stage` also injects before
# the shell imposition that follows every stage of the global step, and
# `none` never injects. The last two add methods to package functions for
# the whole process. Scratch tooling: it prints tables and asserts nothing;
# test/convergence.jl guards four of the rows. Serial, about a minute at -t 1.

using MPI
MPI.Init(threadlevel=:funneled)
using CompactLES, Printf
const CL = CompactLES
MPI.Comm_size(MPI.COMM_WORLD) == 1 || error("run this study on one rank")
include(joinpath(@__DIR__, "..", "test", "smooth_cases.jl"))

const OPTS = CL.script_args(ARGS, (study="all", restrict="step"))
OPTS.restrict in ("step", "stage", "none") ||
    error("restrict must be step, stage or none, got $(OPTS.restrict)")
selected(name) = OPTS.study == "all" || name in split(OPTS.study, ',')

# A cfl the endpoint clip always undercuts, so every step is the requested one.
const BIG_CFL = 50.0

if OPTS.restrict == "none"
    CL.restrict_level!(solver::CL.Solver, states::Vector) = states
elseif OPTS.restrict == "stage"
    function CL.prolong_level_ghosts!(solver::CL.Solver, states::Vector)
        CL.restrict_level!(solver, states)
        invoke(CL.prolong_level_ghosts!, Tuple{Any,Any}, solver, states)
    end
end

function row(name, steps, errs)
    @printf("%-46s", name)
    for (n, e) in zip(steps, errs)
        @printf(" %5d %.3e", n, e)
    end
    ords = successive_orders(1 ./ collect(steps), errs)
    println("   ", join((@sprintf("%.2f", o) for o in ords), " / "))
    flush(stdout)
end

temporal_row(name, build, tfinal, steps, ref_steps; patch=0) =
    row(name, steps, temporal_errors(build, tfinal, steps, ref_steps; patch=patch))

function rk_study()
    println("\nrk: steps and temporal error, orders between successive step counts")
    s = (20, 40, 80, 160)
    temporal_row("periodic standing wave, 96 nodes", () -> mirror_case(49; cfl=BIG_CFL),
                 0.4, s, 2560)
    temporal_row("slip walls, N = 49", () -> wall_case(49; cfl=BIG_CFL), 0.4, s, 2560)
    temporal_row("Dirichlet inflow g(t), N = 65", () -> inflow_case(65; cfl=BIG_CFL),
                 0.4, (80, 160, 320, 640), 10240)
    temporal_row("NSCBC inflow target(t), N = 65",
                 () -> target_inflow_case(65; cfl=BIG_CFL), 0.4, (40, 80, 160, 320), 5120)
end

function filter_study()
    println("\nfilter: the default filter every step unless named")
    s = (20, 40, 80, 160)
    temporal_row("periodic standing wave, relaxed",
                 () -> mirror_case(49; cfl=BIG_CFL, filter_interval=1), 0.4, s, 2560)
    temporal_row("periodic standing wave, full strength",
                 () -> mirror_case(49; cfl=BIG_CFL, filter_interval=1, filter_cfl=0.0),
                 0.4, s, 2560)
    temporal_row("slip walls N = 49, one-sided rows, relaxed",
                 () -> wall_case(49; cfl=BIG_CFL, filter_interval=1), 0.4, s, 2560)
    temporal_row("slip walls N = 49, cascade rows, relaxed",
                 () -> wall_case(49; cfl=BIG_CFL, filter_interval=1,
                                 filt=compact_filter(0.45; closures=:cascade)),
                 0.4, s, 2560)
end

# Boundary data imposed through their time derivative: the face node's
# right-hand side is dq/dt of the data at the stage time, so its stage values
# are the ones the integrator gives any node, where `DirichletBC` writes the
# data's own value at the stage time over them.
struct RateDirichlet{G} <: CL.BoundaryCondition
    q::G     # t -> conserved tuple at the face
end

function CL.correct_rhs!(bc::RateDirichlet, solver, Q, dQ, d, side)
    plane = CL.wallplane(solver.decomp, d, side)
    plane === nothing && return nothing
    for I in plane, c in 1:solver.equations.n_cons
        dQ[I, c] = derivative(τ -> bc.q(τ)[c], solver.tstage)
    end
    return nothing
end

function rate_inflow_case(N; k=2pi, phase=0.37, u0=2.0, opts...)
    prof(t) = entropy_profile(k, phase; u0=u0, t=t)
    probe, _ = inflow_case(N; opts...)
    eq = probe.equations
    inflow = RateDirichlet(t -> conserved(eq, prof(t)(zero(t))))
    _smooth_solver((N, 1, 1), 1.0, ((inflow, NSCBCOutflowBC(pinf=1.0)), per3[2], per3[3]),
                   prof(0.0); merge(SMOOTH_DEFAULTS, opts)...)
end

periodic_wave_case(N; k=2pi, phase=0.37, u0=2.0, opts...) =
    _smooth_solver((N, 1, 1), 1.0, per3, entropy_profile(k, phase; u0=u0);
                   merge(SMOOTH_DEFAULTS, opts)...)

# The faces of `target_inflow_case` with constant targets, the time
# dependence carried by an acoustic pulse in the interior instead.
function pulse_case(N; u0=0.3, eta=2.0, opts...)
    inflow = NSCBCInflowBC(u=(u0, 0.0, 0.0), T_ion=1.0, Y=[1.0], eta_u=eta, eta_T=eta)
    pulse(x) = exp(-100 * (x - 0.5)^2)
    _smooth_solver((N, 1, 1), 1.0, ((inflow, NSCBCOutflowBC(pinf=1.0)), per3[2], per3[3]),
                   x -> (1 + 0.01 * pulse(x), u0 + zero(x), zero(x), 1 + 0.014 * pulse(x));
                   merge(SMOOTH_DEFAULTS, opts)...)
end

function cfl_study()
    println("\ncfl: temporal error at dt = h/5 (cfl 0.64 on the inflow wave) and " *
            "h/10; the last column is the order in h at dt = h/5")
    cases = (("periodic entropy wave, u0 = 2", N -> periodic_wave_case(N - 1; cfl=BIG_CFL)),
             ("Dirichlet inflow, data at the stage time", N -> inflow_case(N; cfl=BIG_CFL)),
             ("Dirichlet inflow, data through dq/dt", N -> rate_inflow_case(N; cfl=BIG_CFL)),
             ("NSCBC inflow target(t)", N -> target_inflow_case(N; cfl=BIG_CFL)),
             ("NSCBC inflow, constant target, pulse", N -> pulse_case(N; cfl=BIG_CFL)))
    for (name, build) in cases
        println(name)
        prev = 0.0
        for N in (33, 65, 129, 257)
            # h = 1/(N − 1) on every case: the periodic line has N − 1 nodes.
            n = 2 * (N - 1)
            e = temporal_errors(() -> build(N), 0.4, (n, 2n), 32n)
            @printf("   N = %-4d  %.3e  %.3e   %.2f", N, e[1], e[2], log2(e[1] / e[2]))
            prev > 0 && @printf("   %.2f", log2(prev / e[1]))
            println()
            prev = e[1]
        end
    end
end

function levels_study()
    println("\nlevels: entropy wave, two levels, N = 96, t = 0.5, restrict = " *
            "$(OPTS.restrict); composite error, then the fine patch's")
    couplings = (("closure rows", (interface_flux=:closure,)),
                 ("ghost fluxes", (interface_flux=:ghost,)),
                 ("C6 BL", (deriv=lele_d1_6(closures=:brady_livescu),
                            interface_flux=:closure)))
    for (sub, steps, ref_steps) in ((false, (40, 80, 160, 320), 5120),
                                    (true, (14, 20, 28, 40, 56, 80, 160), 4480))
        for (name, kw) in couplings
            build() = entropy_case(96; levels=2, subcycle=sub, cfl=BIG_CFL, kw...)
            solver, ref = build()
            fixed_step_run!(solver, ref, 0.5, ref_steps)
            errs = zeros(length(steps), 2)
            for (i, n) in enumerate(steps)
                s, q = build()
                fixed_step_run!(s, q, 0.5, n)
                errs[i, 1] = state_difference(s, q, ref)
                errs[i, 2] = state_difference(s, q, ref; patch=2)
            end
            label = (sub ? "subcycled, " : "global step, ") * name
            row(label, steps, errs[:, 1])
            row("  fine patch", steps, errs[:, 2])
        end
    end
end

function total_study()
    println("\ntotal: entropy wave, two levels, ghost fluxes, t = 0.5, restrict = " *
            "$(OPTS.restrict); error in rho against the exact solution")
    @printf("%-10s %5s %4s %6s %11s %11s\n", "stepping", "cfl", "N", "steps",
            "fine patch", "composite")
    # The unrefined root is the control: its two cfl columns differ by the
    # integrator's own error at the root step.
    for mode in (:uniform, :global, :subcycled), cfl in (0.5, 0.125), N in (48, 96, 192)
        s, q = mode === :uniform ? entropy_case(N; cfl=cfl) :
               entropy_case(N; levels=2, subcycle=mode === :subcycled, cfl=cfl)
        run!(s, q; tfinal=0.5)
        exact = analytic_reference(s.equations, entropy_profile(3, 0.37; t=s.t))
        e = regional_errors(s, q, exact)
        if mode === :uniform
            @printf("%-10s %5.3f %4d %6d %11s %11.3e\n", mode, cfl, N, s.step, "-",
                    e.interior)
            continue
        end
        ps = CL.PatchSolver(s, getfield(s, :patches)[2])
        fine = maximum(1:ps.decomp.n_local[1]) do i
            abs(q[2][padded_index(ps, i, 1, 1), 1] - exact(xcoord(ps, 1, i))[1])
        end
        @printf("%-10s %5.3f %4d %6d %11.3e %11.3e\n", mode, cfl, N, s.step, fine,
                max(e.interface, e.interior))
        flush(stdout)
    end
end

function main()
    t0 = time()
    selected("rk") && rk_study()
    selected("filter") && filter_study()
    selected("cfl") && cfl_study()
    selected("levels") && levels_study()
    selected("total") && total_study()
    @printf("\ndone in %.1f s\n", time() - t0)
end

abspath(PROGRAM_FILE) == (@__FILE__) && main()
