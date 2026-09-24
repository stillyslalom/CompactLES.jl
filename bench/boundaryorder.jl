# The smooth-evolution accuracy matrix: walls, patch interfaces and
# refinement levels, by region, with the closure and filter options.
#
#   julia --project=. -t 1 bench/boundaryorder.jl [study=all] [ns=49,97,193]
#                                                  [cfl=0.25] [tfinal=0.4]
#
# `study` is one of, or a comma-separated list of,
#
#   truncation   one derivative of a polynomial of the closure's degree plus
#                one, wall window against interior, actual spacing
#   rhs          the instantaneous right-hand-side error of the assembled
#                Navier–Stokes operator on exact data, by region
#   walls        the standing wave between slip walls (inviscid) and
#                adiabatic no-slip walls (viscous), every derivative closure
#                under no filter, the cascade filter and the one-sided
#                filter, against the fine periodic reference (total error)
#                and against the periodic mirror at the same spacing (the
#                closure defect alone)
#   shear        the decaying shear mode between no-slip walls, exact
#   interfaces   the entropy wave and the viscous standing wave through a
#                same-level patch interface
#   levels       the same through two- and three-level nests, global step
#                and subcycled
#   fields       the wall and level rows' momentum and energy components
#   filter       the repeated-filter accumulation on the wall case: cadence,
#                relaxation and the wall rows, against the mirror
#   dt           the timestep floor of each family's finest grid
#   idiv         the flux divergence's rows at interface ends from each
#                `interface_divergence` source, walls on the default rows:
#                same-level and coarse-fine faces, both gradient treatments,
#                mixed wall and interface ends, Float32, acoustic reflection
#   gflux        the same rows under `interface_flux = :ghost` (the inviscid
#                and molecular fluxes differenced through interface ends from
#                ghost fluxes) beside the default and the Brady–Livescu rows,
#                with the polynomial right-hand side at both kinds of
#                interface end
#   transfer     the level rows at each `level_interpolation_order` in
#                `orders` for each derivative operator in `derivs` (c6, c6bl,
#                c8, c8bl, c10: the default rows or Brady–Livescu's), with
#                C10 under the default filter; global step, subcycled, three
#                levels, viscous, and the 2-D entropy wave through a square
#                level (`entropy2d_case`) for the C6 operators among them
#   transfertime the temporal order of the finest two-level grid, global step
#                and subcycled, at the highest of `orders`
#
# The last two run only when named; `all` leaves them out.
#
# `coupling` is a Julia expression for a NamedTuple of further `Solver`
# keywords applied to every row of the two transfer studies, e.g. an
# interface divergence choice: `coupling="(key=value,)"`. Empty by default,
# which measures the default interface coupling.
#
# Every evolution row is run at `cfl` and at `cfl/2`, and the `dt` column
# is the relative change of the primary error between the two; a row whose
# dt column is above 0.1 has its slope set by the time integrator, not by
# the spatial operator, and the `dt` study says where that floor is. The
# cases, references and regional norms are in test/smooth_cases.jl, shared
# with the guards in test/convergence.jl. Scratch tooling: it prints tables
# and asserts nothing.

using MPI
MPI.Init(threadlevel=:funneled)
using CompactLES, Printf
const CL = CompactLES
MPI.Comm_size(MPI.COMM_WORLD) == 1 || error("run this study on one rank")
include(joinpath(@__DIR__, "..", "test", "smooth_cases.jl"))

const OPTS = CL.script_args(ARGS, (study="all", ns="49,97,193", cfl=0.25, tfinal=0.4,
                                   orders="4,6,8", derivs="c6,c6bl,c8bl",
                                   coupling=""))
const TRANSFER_ORDERS = parse.(Int, split(OPTS.orders, ','))
const COUPLING = isempty(OPTS.coupling) ? (;) : Core.eval(Main, Meta.parse(OPTS.coupling))
const NS = parse.(Int, split(OPTS.ns, ','))
const NS_PERIODIC = NS .- 1          # 48, 96, 192: multiples of 24 for the nest
const CFL = OPTS.cfl
const TFINAL = OPTS.tfinal
const TFINAL_ENTROPY = 0.5           # as test/patch_tests.jl and level_tests.jl
const GAMMA = 1.4
const MU = 0.005

selected(name) = OPTS.study == "all" || name in split(OPTS.study, ',')
named(name) = name in split(OPTS.study, ',')

const DERIVS = (("C6 neutral3", lele_d1_6()),
                ("C6 cascade3", lele_d1_6(closures=:cascade3)),
                ("C6 cascade4", lele_d1_6(closures=:cascade4)),
                ("C6 BL", lele_d1_6(closures=:brady_livescu)),
                ("C8 BL", lele_d1_8(closures=:brady_livescu)))
const FILTERS = ((" unfiltered", (filter_interval=0,)),
                 (" cascade filter", (filter_interval=1,
                                      filt=compact_filter(0.45; closures=:cascade))),
                 (" onesided filter", (filter_interval=1,
                                       filt=compact_filter(0.45; closures=:onesided))))
const INTERIOR_DERIVS = (("C6", lele_d1_6()), ("C6 BL", lele_d1_6(closures=:brady_livescu)),
                         ("C10", lele_d1_10()))

# --- printing ---------------------------------------------------------------------

function header(title)
    println("\n", title)
    println("   N      h        wall       interface  covered    interior   " *
            "l2         dt      at")
end

function printrow(N, h, e, dtsens)
    @printf("%4d  %.3e  %.3e  %.3e  %.3e  %.3e  %.3e  %.3f   %s\n", N, h,
            e.wall, e.interface, e.covered, e.interior, e.l2, dtsens, string(e.at))
end

function printorders(hs, es, primary)
    p = successive_orders(hs, [getfield(e, primary) for e in es])
    l2 = successive_orders(hs, [e.l2 for e in es])
    @printf("      orders (%s): %s   l2: %s\n", primary,
            join((@sprintf("%.2f", x) for x in p), " / "),
            join((@sprintf("%.2f", x) for x in l2), " / "))
    flush(stdout)
end

# --- the evolution driver ---------------------------------------------------------
#
# `build(N; cfl)` returns (solver, states) at t = 0; `reference(solver, cfl)`
# returns the callable the final state is measured against, built after the
# run so that a mirror at the same spacing and step, or the exact solution
# at the solver's own clock, can be supplied.

# A configuration the closure table records as unstable (C8 Brady–Livescu
# under the cascade filter, `:cascade4` under the one-sided one) fails
# inside `run!`; the row reads FAILED and the study goes on.
function evolve!(build, N, cfl, tfinal)
    solver, states = build(N; cfl=cfl)
    try
        run!(solver, states; tfinal=tfinal, nmax=200_000)
    catch err
        err isa SolverFailure || rethrow()
        @printf("%4d  FAILED: %s at step %d, t = %.4f
", N, err.reason, err.step, solver.t)
        return nothing
    end
    return solver, states
end

const FAILED = (wall=NaN, interface=NaN, covered=NaN, interior=NaN, l2=NaN, at=(0, 0))

function evolution_study(title, build, reference, ns; primary=:wall, comp=1,
                         tfinal=TFINAL, cfl=CFL, errors=regional_errors)
    header(title)
    hs = Float64[]; es = []
    for N in ns
        run1 = evolve!(build, N, cfl, tfinal)
        run2 = run1 === nothing ? nothing : evolve!(build, N, cfl / 2, tfinal)
        if run1 === nothing || run2 === nothing
            push!(hs, NaN); push!(es, FAILED)
            continue
        end
        solver, states = run1
        e = errors(solver, states, reference(solver, cfl); comp=comp)
        solver2, states2 = run2
        e2 = errors(solver2, states2, reference(solver2, cfl / 2); comp=comp)
        dtsens = abs(getfield(e2, primary) - getfield(e, primary)) /
                 max(getfield(e, primary), 1e-300)
        push!(hs, root_spacing(solver)); push!(es, e)
        printrow(N, root_spacing(solver), e, dtsens)
    end
    any(isnan, hs) || printorders(hs, es, primary)
    return es
end

# --- references ---------------------------------------------------------------------

const REFERENCE_CFL = 0.125
const REF = Dict{Any,NodeReference}()

"The fine periodic reference of the standing wave, on nested nodes, run once."
function fine_mirror(viscous; opts...)
    get!(REF, (:mirror, viscous, values(opts))) do
        solver, states = mirror_case(4 * (maximum(NS) - 1) + 1; viscous=viscous,
                                     cfl=REFERENCE_CFL, opts...)
        run!(solver, states; tfinal=TFINAL)
        NodeReference(solver, states)
    end
end

"The fine periodic reference of the viscous standing wave on [0, 2)."
function fine_periodic(; opts...)
    get!(REF, (:periodic, values(opts))) do
        solver, states = viscous_periodic_case(9 * maximum(NS_PERIODIC);
                                               cfl=REFERENCE_CFL, opts...)
        run!(solver, states; tfinal=TFINAL)
        NodeReference(solver, states)
    end
end

"The periodic mirror of a wall run, at the same spacing, options and step."
function same_mirror(solver, cfl; viscous, opts...)
    N = solver.n_global[1]
    mirror, states = mirror_case(N; viscous=viscous, cfl=cfl, opts...)
    run!(mirror, states; tfinal=TFINAL)
    NodeReference(mirror, states)
end

entropy_reference(k, phase) =
    (solver, cfl) -> analytic_reference(solver.equations,
                                        entropy_profile(k, phase; t=solver.t))
entropy2d_reference() =
    (solver, cfl) -> begin
        prof = entropy2d_profile(2, 1, 0.37; t=solver.t)
        (x, y) -> conserved(solver.equations, prof(x, y))
    end
shear_reference(V) =
    (solver, cfl) -> analytic_reference(solver.equations, shear_profile(V, MU; t=solver.t))

# --- studies ---------------------------------------------------------------------------

function truncation_study()
    println("\n=== truncation: one derivative on the closed line, actual spacing ===")
    for (label, deriv, degree) in (("C6 neutral3", lele_d1_6(), 4),
                                   ("C6 cascade3", lele_d1_6(closures=:cascade3), 4),
                                   ("C6 cascade4", lele_d1_6(closures=:cascade4), 5),
                                   ("C6 BL", lele_d1_6(closures=:brady_livescu), 6),
                                   ("C8 BL", lele_d1_8(closures=:brady_livescu), 8))
        ns = (17, 33, 65, 129)
        println("\n$label, derivative of x^$degree")
        hs = Float64[]; ws = Float64[]; is = Float64[]
        for N in ns
            e = closed_derivative_errors(N, deriv, x -> x^degree,
                                         x -> degree * x^(degree - 1))
            push!(hs, 1 / (N - 1)); push!(ws, e.wall); push!(is, e.interior)
            @printf("%4d  wall %.3e  interior %.3e\n", N, e.wall, e.interior)
        end
        @printf("      wall orders: %s   interior: %s\n",
                join((@sprintf("%.2f", x) for x in successive_orders(hs, ws)), " / "),
                join((@sprintf("%.2f", x) for x in successive_orders(hs, is)), " / "))
    end
    println("\nexp(sin(3x)) on (24, 48, 96), the test/convergence.jl field, actual spacing")
    for (label, deriv) in DERIVS
        ns = (24, 48, 96)
        hs = Float64[]; ws = Float64[]; is = Float64[]
        for N in ns
            e = closed_derivative_errors(N, deriv, x -> exp(sin(3x)),
                                         x -> 3cos(3x) * exp(sin(3x)))
            push!(hs, 1 / (N - 1)); push!(ws, e.wall); push!(is, e.interior)
        end
        @printf("%-14s wall %.3e %.3e %.3e  orders %s   interior orders %s\n", label,
                ws..., join((@sprintf("%.2f", x) for x in successive_orders(hs, ws)), "/"),
                join((@sprintf("%.2f", x) for x in successive_orders(hs, is)), "/"))
    end
    flush(stdout)
end

function rhs_row(title, build, exact, ns; primary, comp=1)
    header(title)
    hs = Float64[]; es = []
    for N in ns
        solver, states = build(N)
        e = rhs_errors(solver, states, exact(solver); comp=comp)
        push!(hs, root_spacing(solver)); push!(es, e)
        printrow(N, root_spacing(solver), e, 0.0)
    end
    printorders(hs, es, primary)
end

function rhs_study()
    println("\n=== instantaneous right-hand-side error on exact data ===")
    prof = standing_profile(0.05, 0.05)
    for viscous in (false, true), (label, deriv) in DERIVS
        mu = viscous ? MU : 0.0
        rhs_row("RHS $(viscous ? "viscous" : "inviscid") wall, $label",
                N -> wall_case(N; viscous=viscous, deriv=deriv),
                s -> exact_rhs(s.equations, prof; mu=mu), NS; primary=:wall)
    end
    for (label, deriv) in DERIVS
        rhs_row("RHS shear mode, $label, rho v component",
                N -> shear_case(N; deriv=deriv),
                s -> exact_rhs(s.equations, shear_profile(0.1, MU); mu=MU,
                               source=x -> -MU * (0.1 * pi * cos(pi * x))^2),
                NS; primary=:wall, comp=3)
    end
    for (label, deriv) in INTERIOR_DERIVS, levels in (1, 2, 3)
        what = levels == 1 ? "two patches" : "$levels levels"
        rhs_row("RHS entropy wave k=3, $what, $label",
                N -> entropy_case(N; deriv=deriv, levels=levels,
                                  patch_grid=levels == 1 ? (2, 1, 1) : (1, 1, 1)),
                s -> exact_rhs(s.equations, entropy_profile(3, 0.37)), NS_PERIODIC;
                primary=:interface)
    end
end

function walls_study()
    println("\n=== walls: the standing wave, t = $TFINAL ===")
    for viscous in (false, true)
        fine = fine_mirror(viscous)
        for (label, deriv) in DERIVS, (flabel, fopts) in FILTERS
            what = "$(viscous ? "viscous" : "inviscid") wall, $label,$flabel"
            evolution_study("$what, against the fine reference",
                            (N; cfl) -> wall_case(N; viscous=viscous, deriv=deriv,
                                                  cfl=cfl, fopts...),
                            (s, cfl) -> fine, NS)
            evolution_study("$what, against the mirror (closure defect)",
                            (N; cfl) -> wall_case(N; viscous=viscous, deriv=deriv,
                                                  cfl=cfl, fopts...),
                            (s, cfl) -> same_mirror(s, cfl; viscous=viscous,
                                                    deriv=deriv, fopts...), NS)
        end
        evolution_study("$(viscous ? "viscous" : "inviscid") periodic mirror, C6, " *
                        "unfiltered, against the fine reference (interior only)",
                        (N; cfl) -> mirror_case(N; viscous=viscous, cfl=cfl),
                        (s, cfl) -> fine, NS; primary=:interior)
    end
end

function shear_study()
    println("\n=== shear mode between no-slip walls, exact, t = $TFINAL ===")
    for (label, deriv) in DERIVS, (flabel, fopts) in FILTERS
        evolution_study("shear, $label,$flabel, rho v component",
                        (N; cfl) -> shear_case(N; deriv=deriv, cfl=cfl, fopts...),
                        shear_reference(0.1), NS; comp=3)
    end
end

function interfaces_study()
    println("\n=== same-level patch interface, t = $TFINAL_ENTROPY (entropy) / $TFINAL ===")
    for (k, phase) in ((3, 0.37), (1, 0.0)), (label, deriv) in INTERIOR_DERIVS,
        (flabel, fopts) in FILTERS[1:2]
        evolution_study("entropy wave k=$k phase=$phase, two patches, $label,$flabel",
                        (N; cfl) -> entropy_case(N; k=k, phase=phase, deriv=deriv,
                                                 patch_grid=(2, 1, 1), cfl=cfl, fopts...),
                        entropy_reference(k, phase), NS_PERIODIC;
                        primary=:interface, tfinal=TFINAL_ENTROPY)
    end
    fine = fine_periodic()
    for (label, deriv) in INTERIOR_DERIVS
        evolution_study("viscous standing wave, two patches, $label, unfiltered",
                        (N; cfl) -> viscous_periodic_case(N; deriv=deriv,
                                                          patch_grid=(2, 1, 1), cfl=cfl),
                        (s, cfl) -> fine, NS_PERIODIC; primary=:interface)
    end
    evolution_study("viscous standing wave, one patch, C6, unfiltered (interior only)",
                    (N; cfl) -> viscous_periodic_case(N; cfl=cfl),
                    (s, cfl) -> fine, NS_PERIODIC; primary=:interior)
end

function levels_study()
    println("\n=== refinement levels, fixed physical endpoints ===")
    for levels in (2, 3), subcycle in (false, true), (label, deriv) in INTERIOR_DERIVS,
        (flabel, fopts) in FILTERS[1:2]
        evolution_study("entropy wave k=3, $levels levels, subcycle=$subcycle, " *
                        "$label,$flabel",
                        (N; cfl) -> entropy_case(N; deriv=deriv, levels=levels,
                                                 subcycle=subcycle, cfl=cfl, fopts...),
                        entropy_reference(3, 0.37), NS_PERIODIC;
                        primary=:interface, tfinal=TFINAL_ENTROPY)
    end
    fine = fine_periodic()
    for levels in (2, 3), (label, deriv) in INTERIOR_DERIVS
        evolution_study("viscous standing wave, $levels levels, $label, unfiltered",
                        (N; cfl) -> viscous_periodic_case(N; deriv=deriv, levels=levels,
                                                          cfl=cfl),
                        (s, cfl) -> fine, NS_PERIODIC; primary=:interface)
    end
end

function fields_study()
    println("\n=== the other components: rho u and E ===")
    fine = fine_mirror(false)
    for (label, deriv) in (DERIVS[1], DERIVS[4]), (cname, comp) in (("rho u", 2), ("E", 5))
        evolution_study("inviscid wall, $label, unfiltered, $cname",
                        (N; cfl) -> wall_case(N; deriv=deriv, cfl=cfl),
                        (s, cfl) -> fine, NS; comp=comp)
        evolution_study("entropy wave k=3, 2 levels, $label, unfiltered, $cname",
                        (N; cfl) -> entropy_case(N; deriv=deriv, levels=2, cfl=cfl),
                        entropy_reference(3, 0.37), NS_PERIODIC;
                        primary=:interface, tfinal=TFINAL_ENTROPY, comp=comp)
    end
end

function filter_study()
    println("\n=== repeated filtering on the inviscid wall, C6 cascade3 ===")
    println("closure defect against the mirror under the same filter, and the")
    println("total against the fine reference; passes = steps / filter_interval")
    fine = fine_mirror(false)
    CASCADE = compact_filter(0.45; closures=:cascade)
    rows = (("unfiltered", (filter_interval=0,)),
            ("cascade, every step, relaxed", (filter_interval=1, filt=CASCADE)),
            ("cascade, every 2nd step, relaxed", (filter_interval=2, filt=CASCADE)),
            ("cascade, every 4th step, relaxed", (filter_interval=4, filt=CASCADE)),
            ("cascade, every step, unrelaxed",
             (filter_interval=1, filter_cfl=0.0, filt=CASCADE)),
            ("onesided, every step, relaxed",
             (filter_interval=1, filt=compact_filter(0.45; closures=:onesided))),
            ("onesided, every step, unrelaxed",
             (filter_interval=1, filter_cfl=0.0,
              filt=compact_filter(0.45; closures=:onesided))))
    for N in NS
        println("\nN = $N")
        println("  configuration                          passes  vs mirror: wall      " *
                "interior   | vs fine: wall      interior   l2")
        for (label, fopts) in rows
            run1 = evolve!((N; cfl) -> wall_case(N; cfl=cfl, fopts...), N, CFL, TFINAL)
            run1 === nothing && continue
            solver, states = run1
            em = regional_errors(solver, states,
                                 same_mirror(solver, CFL; viscous=false, fopts...))
            ef = regional_errors(solver, states, fine)
            passes = fopts.filter_interval == 0 ? 0 : solver.step ÷ fopts.filter_interval
            @printf("  %-38s %6d  %.3e  %.3e  | %.3e  %.3e  %.3e\n", label, passes,
                    em.wall, em.interior, ef.wall, ef.interior, ef.l2)
        end
        # The mirror's own filter defect: the interior rows' eighth-order pass.
        for (label, fopts) in rows[1:2]
            run1 = evolve!((N; cfl) -> mirror_case(N; cfl=cfl, fopts...), N, CFL, TFINAL)
            run1 === nothing && continue
            solver, states = run1
            ef = regional_errors(solver, states, fine)
            @printf("  %-38s %6d  %10s  %10s  | %.3e  %.3e  %.3e   (periodic mirror)\n",
                    label, fopts.filter_interval == 0 ? 0 : solver.step, "", "",
                    ef.wall, ef.interior, ef.l2)
        end
        flush(stdout)
    end
end

function dt_study()
    println("\n=== the timestep floor at the finest grid ===")
    N = maximum(NS); Np = maximum(NS_PERIODIC)
    fine = fine_mirror(false)
    println("\ninviscid wall N=$N, C6 BL, unfiltered, against the fine reference")
    println("   cfl     steps  wall       interior   l2")
    for cfl in (0.5, 0.25, 0.125, 0.0625)
        run1 = evolve!((N; cfl) -> wall_case(N; deriv=DERIVS[4][2], cfl=cfl), N, cfl, TFINAL)
        run1 === nothing && continue
        solver, states = run1
        e = regional_errors(solver, states, fine)
        @printf("  %.4f  %5d  %.3e  %.3e  %.3e\n", cfl, solver.step, e.wall, e.interior, e.l2)
    end
    println("\nperiodic mirror N=$N, C6, unfiltered, against the fine reference")
    println("   cfl     steps  interior   l2")
    for cfl in (0.5, 0.25, 0.125, 0.0625)
        run1 = evolve!((N; cfl) -> mirror_case(N; cfl=cfl), N, cfl, TFINAL)
        run1 === nothing && continue
        solver, states = run1
        e = regional_errors(solver, states, fine)
        @printf("  %.4f  %5d  %.3e  %.3e\n", cfl, solver.step, e.interior, e.l2)
    end
    for subcycle in (false, true)
        println("\nentropy wave k=3 N=$Np, 3 levels, subcycle=$subcycle, C6 BL, unfiltered")
        println("   cfl     steps  interface  covered    interior   l2")
        for cfl in (0.5, 0.25, 0.125, 0.0625)
            run1 = evolve!((N; cfl) -> entropy_case(N; deriv=DERIVS[4][2], levels=3,
                                                    subcycle=subcycle, cfl=cfl),
                           Np, cfl, TFINAL_ENTROPY)
            run1 === nothing && continue
            solver, states = run1
            e = regional_errors(solver, states, entropy_reference(3, 0.37)(solver, cfl))
            @printf("  %.4f  %5d  %.3e  %.3e  %.3e  %.3e\n", cfl, solver.step,
                    e.interface, e.covered, e.interior, e.l2)
        end
    end
    flush(stdout)
end

# --- the interface divergence rows ---------------------------------------------------
#
# The derivative operator stays `lele_d1_6()` throughout, so every physical
# wall keeps the default rows, and only the flux divergence's rows at patch
# and level interface ends change with the `interface_divergence` source.

const IDIV_CLOSURES = (("default", nothing), ("cascade4", :cascade4),
                       ("BL", :brady_livescu))
"The `interface_divergence` source of each candidate, in element type `T`."
idiv_sources(T=Float64) = [(label, cl === nothing ? nothing : lele_d1_6(T; closures=cl))
                           for (label, cl) in IDIV_CLOSURES]
const IDIV = idiv_sources()
const IDIV_FILTERS = ((" unfiltered", (filter_interval=0,)),
                      (" default filter", (filter_interval=1,)))

# The acoustic pulse of test/patch_tests.jl and test/interface_reflection_tests.jl:
# a right-running pulse launched at π/2 on the periodic [0, 2π), run past
# the faces; the left-running characteristic left behind on uncovered root
# nodes upstream of the first face, against the run without the interface.
function pulse_leftgoing(N; source=nothing, patch_grid=(1, 1, 1), refine=nothing,
                         subcycle=false, opts...)
    amp = 1e-3; c0 = sqrt(GAMMA)
    pulse(x) = amp * exp(-40 * (x - pi / 2)^2)
    s = Solver(n_global=(N, 1, 1), L_domain=(2pi, 1.0, 1.0), bcs=per3,
               art=ArtParams(enabled=false), filter_interval=0,
               patch_grid=patch_grid, refine=refine, subcycle=subcycle;
               merge((interface_divergence=source,), opts)...)
    Q = allocate_state(s)
    initialize!(s, Q, (x, y, z) -> Prim(rho=(1 + pulse(x))^(1 / GAMMA),
                                        p=1 + pulse(x), u=(pulse(x) / c0, 0, 0)))
    run!(s, Q; tfinal=pi / c0)
    states = Q isa Vector ? Q : [Q]
    ps = CL.PatchSolver(s, getfield(s, :patches)[1])
    refresh_primitives!(ps, states[1])
    return [((ps.p[gidx(ps, i, 1, 1)] - 1) - c0 * ps.u[gidx(ps, i, 1, 1)]) / 2 / amp
            for i in 1:ps.decomp.n_local[1] if xcoord(ps, 1, i) < 2.1]
end

function idiv_study()
    println("\n=== interface divergence rows, deriv = lele_d1_6(), walls on its rows ===")
    per_rhs = (("extended", :extended), ("onesided", :onesided))
    # The instantaneous right-hand side on exact data.
    for (clabel, src) in IDIV, levels in (1, 2)
        what = levels == 1 ? "two patches" : "2 levels"
        rhs_row("RHS entropy wave k=3, $what, source $clabel",
                N -> entropy_case(N; levels=levels, interface_divergence=src,
                                  patch_grid=levels == 1 ? (2, 1, 1) : (1, 1, 1)),
                s -> exact_rhs(s.equations, entropy_profile(3, 0.37)), NS_PERIODIC;
                primary=:interface)
    end
    # Same-level interfaces: two entropy waves and the inviscid standing wave.
    fine_inviscid = fine_periodic(mu=0.0)
    for (clabel, src) in IDIV, (flabel, fopts) in IDIV_FILTERS
        for (k, phase) in ((3, 0.37), (1, 0.0))
            evolution_study("entropy wave k=$k, two patches, source $clabel,$flabel",
                            (N; cfl) -> entropy_case(N; k=k, phase=phase,
                                                     patch_grid=(2, 1, 1), cfl=cfl,
                                                     interface_divergence=src, fopts...),
                            entropy_reference(k, phase), NS_PERIODIC;
                            primary=:interface, tfinal=TFINAL_ENTROPY)
        end
        evolution_study("inviscid standing wave, two patches, source $clabel,$flabel",
                        (N; cfl) -> viscous_periodic_case(N; mu=0.0, patch_grid=(2, 1, 1),
                                                          cfl=cfl, interface_divergence=src,
                                                          fopts...),
                        (s, cfl) -> fine_inviscid, NS_PERIODIC; primary=:interface)
    end
    # Coarse-fine interfaces: nests, stepping modes, tiles.
    for (clabel, src) in IDIV
        for levels in (2, 3), subcycle in (false, true)
            evolution_study("entropy wave k=3, $levels levels, subcycle=$subcycle, " *
                            "source $clabel, unfiltered",
                            (N; cfl) -> entropy_case(N; levels=levels, subcycle=subcycle,
                                                     cfl=cfl, interface_divergence=src,
                                                     filter_interval=0),
                            entropy_reference(3, 0.37), NS_PERIODIC;
                            primary=:interface, tfinal=TFINAL_ENTROPY)
        end
        evolution_study("entropy wave k=3, 2 levels, source $clabel, default filter",
                        (N; cfl) -> entropy_case(N; levels=2, cfl=cfl, filter_interval=1,
                                                 interface_divergence=src),
                        entropy_reference(3, 0.37), NS_PERIODIC;
                        primary=:interface, tfinal=TFINAL_ENTROPY)
        evolution_study("entropy wave k=1, 2 levels, source $clabel, unfiltered",
                        (N; cfl) -> entropy_case(N; k=1, phase=0.0, levels=2, cfl=cfl,
                                                 interface_divergence=src),
                        entropy_reference(1, 0.0), NS_PERIODIC;
                        primary=:interface, tfinal=TFINAL_ENTROPY)
        evolution_study("inviscid standing wave, 2 levels, source $clabel, unfiltered",
                        (N; cfl) -> viscous_periodic_case(N; mu=0.0, levels=2, cfl=cfl,
                                                          interface_divergence=src),
                        (s, cfl) -> fine_inviscid, NS_PERIODIC; primary=:interface)
        evolution_study("entropy wave k=3, 2 levels tiled (tile 4), subcycled, " *
                        "source $clabel, unfiltered",
                        (N; cfl) -> entropy_case(N; levels=2, subcycle=true, tile=4,
                                                 cfl=cfl, interface_divergence=src),
                        entropy_reference(3, 0.37), NS_PERIODIC;
                        primary=:interface, tfinal=TFINAL_ENTROPY)
    end
    # Physical and interface ends on one line: the standing wave between slip
    # walls through a same-level interface at x = 1/2.
    fine = fine_mirror(false)
    for (clabel, src) in IDIV, (flabel, fopts) in IDIV_FILTERS
        evolution_study("inviscid wall + interface, two patches, source $clabel,$flabel",
                        (N; cfl) -> wall_case(N; patch_grid=(2, 1, 1), cfl=cfl,
                                              interface_divergence=src, fopts...),
                        (s, cfl) -> fine, NS; primary=:interface)
    end
    # Viscous: the gradient rows at the interface are `interface_rhs`'s.
    fine_viscous = fine_periodic()
    for (clabel, src) in IDIV, (rlabel, rhs) in per_rhs, levels in (1, 2)
        what = levels == 1 ? "two patches" : "2 levels"
        evolution_study("viscous standing wave, $what, interface_rhs $rlabel, " *
                        "source $clabel, unfiltered",
                        (N; cfl) -> viscous_periodic_case(N; levels=levels, cfl=cfl,
                                                          patch_grid=levels == 1 ?
                                                              (2, 1, 1) : (1, 1, 1),
                                                          interface_rhs=rhs,
                                                          interface_divergence=src),
                        (s, cfl) -> fine_viscous, NS_PERIODIC; primary=:interface)
    end
    # Float32: the entropy wave k=1 (the smallest signal-to-rounding ratio of
    # the resolved fields here) through each interface kind.
    for ((clabel, _), (_, src32)) in zip(IDIV, idiv_sources(Float32)),
        (what, kw) in (("two patches", (patch_grid=(2, 1, 1),)), ("2 levels", (levels=2,)))
        evolution_study("Float32 entropy wave k=1, $what, source $clabel, unfiltered",
                        (N; cfl) -> entropy_case(N; k=1, phase=0.0, cfl=cfl,
                                                 precision=Float32,
                                                 deriv=lele_d1_6(Float32),
                                                 filt=compact_filter(0.45, Float32),
                                                 interface_divergence=src32, kw...),
                        entropy_reference(1, 0.0), NS_PERIODIC;
                        primary=:interface, tfinal=TFINAL_ENTROPY)
    end
    # Acoustic reflection: the left-running characteristic upstream of the
    # first face over the pulse amplitude, against the run without it.
    println("\nacoustic pulse, left-running characteristic upstream of the faces / amplitude")
    println("   N    layout                    source      reflected")
    r1(N) = BlockRegion((80N ÷ 192, 0, 0), (32N ÷ 192 + 1, 1, 1))
    for N in (96, 192, 384)
        base = pulse_leftgoing(N)
        for (what, kw) in (("two patches", (patch_grid=(2, 1, 1),)),
                           ("2 levels", (refine=r1(N),)),
                           ("2 levels subcycled", (refine=r1(N), subcycle=true)))
            for (clabel, src) in IDIV
                r = pulse_leftgoing(N; source=src, kw...)
                @printf("%5d    %-24s  %-10s  %.3e\n", N, what, clabel,
                        maximum(abs.(r .- base)))
            end
        end
        flush(stdout)
    end
end

function transfer_study()
    println("\n=== level transfer order, fixed physical endpoints, coupling = " *
            "$(isempty(OPTS.coupling) ? "default" : OPTS.coupling) ===")
    fine = fine_periodic()
    table = Dict("c6" => ("C6", lele_d1_6()),
                 "c6bl" => ("C6 BL", lele_d1_6(closures=:brady_livescu)),
                 "c8" => ("C8", lele_d1_8()),
                 "c8bl" => ("C8 BL", lele_d1_8(closures=:brady_livescu)),
                 "c10" => ("C10", lele_d1_10()))
    derivs = [table[k] for k in split(OPTS.derivs, ',')]
    for (label, deriv) in derivs, p in TRANSFER_ORDERS
        o = (deriv=deriv, level_interpolation_order=p, COUPLING...)
        evolution_study("entropy wave k=3, 2 levels, $label, order $p, unfiltered",
                        (N; cfl) -> entropy_case(N; levels=2, cfl=cfl, o...),
                        entropy_reference(3, 0.37), NS_PERIODIC;
                        primary=:interface, tfinal=TFINAL_ENTROPY)
        evolution_study("entropy wave k=3, 2 levels, $label, order $p, filtered",
                        (N; cfl) -> entropy_case(N; levels=2, cfl=cfl, filter_interval=1,
                                                 o...),
                        entropy_reference(3, 0.37), NS_PERIODIC;
                        primary=:interface, tfinal=TFINAL_ENTROPY)
        evolution_study("entropy wave k=3, 2 levels subcycled, $label, order $p, unfiltered",
                        (N; cfl) -> entropy_case(N; levels=2, subcycle=true, cfl=cfl, o...),
                        entropy_reference(3, 0.37), NS_PERIODIC;
                        primary=:interface, tfinal=TFINAL_ENTROPY)
        evolution_study("entropy wave k=3, 3 levels subcycled, $label, order $p, unfiltered",
                        (N; cfl) -> entropy_case(N; levels=3, subcycle=true, cfl=cfl, o...),
                        entropy_reference(3, 0.37), NS_PERIODIC;
                        primary=:interface, tfinal=TFINAL_ENTROPY)
        evolution_study("viscous standing wave, 2 levels, $label, order $p, unfiltered",
                        (N; cfl) -> viscous_periodic_case(N; levels=2, cfl=cfl, o...),
                        (s, cfl) -> fine, NS_PERIODIC; primary=:interface)
    end
    # Two dimensions: the imposed planes carry interpolated values along the
    # other dimension, so the inviscid solution itself reads the order.
    ns2 = filter(n -> n % 12 == 0, NS_PERIODIC)
    derivs2 = filter(((label, _),) -> startswith(label, "C6"), derivs)
    for (label, deriv) in derivs2, p in TRANSFER_ORDERS, filtered in (false, true)
        evolution_study("2-D entropy wave (2, 1), 2 levels, $label, order $p, " *
                        (filtered ? "filtered" : "unfiltered"),
                        (N; cfl) -> entropy2d_case(N; cfl=cfl, deriv=deriv,
                                                   filter_interval=filtered ? 1 : 0,
                                                   level_interpolation_order=p,
                                                   COUPLING...),
                        entropy2d_reference(), ns2; primary=:interface,
                        tfinal=TFINAL_ENTROPY, errors=regional_errors2d)
    end
    # The C10 interior carries the C8 filter in any claim made for it.
    for p in TRANSFER_ORDERS
        evolution_study("entropy wave k=3, 2 levels, C10, order $p, filtered",
                        (N; cfl) -> entropy_case(N; levels=2, cfl=cfl, filter_interval=1,
                                                 deriv=lele_d1_10(),
                                                 level_interpolation_order=p,
                                                 COUPLING...),
                        entropy_reference(3, 0.37), NS_PERIODIC;
                        primary=:interface, tfinal=TFINAL_ENTROPY)
    end
end

# The time error alone: each run against the same configuration at a sixteenth
# of the largest step, node by node on the root and the fine patch, so the
# spatial error, identical in every run, cancels.
function transfer_time_study()
    Np = maximum(NS_PERIODIC)
    p = maximum(TRANSFER_ORDERS)
    println("
=== temporal order at a level interface, N = $Np, order $p ===")
    cfls = (0.8, 0.4, 0.2, 0.1)
    for (label, deriv) in (("C6", lele_d1_6()),
                           ("C6 BL", lele_d1_6(closures=:brady_livescu))),
        subcycle in (false, true)
        println("
entropy wave k=3, 2 levels, subcycle=$subcycle, $label, unfiltered")
        build = (N; cfl) -> entropy_case(N; deriv=deriv, levels=2, subcycle=subcycle,
                                         cfl=cfl, level_interpolation_order=p,
                                         COUPLING...)
        ref = evolve!(build, Np, cfls[end] / 2, TFINAL_ENTROPY)
        ref === nothing && continue
        _, ref_states = ref
        println("   cfl     steps  max |Q − Q(cfl = $(cfls[end] / 2))|  root / fine")
        ds = Float64[]
        for cfl in cfls
            run1 = evolve!(build, Np, cfl, TFINAL_ENTROPY)
            run1 === nothing && (push!(ds, NaN); continue)
            solver, states = run1
            d = [maximum(abs.(parent(states[k])[:, :, :, 1] .-
                              parent(ref_states[k])[:, :, :, 1])) for k in 1:2]
            push!(ds, maximum(d))
            @printf("  %.4f  %5d  %.3e / %.3e
", cfl, solver.step, d...)
        end
        @printf("      temporal orders: %s\n",
                join((@sprintf("%.2f", log(ds[k] / ds[k+1]) / log(2))
                      for k in 1:length(ds)-1), " / "))
        flush(stdout)
    end
end

# --- the divergence through interface ends from ghost fluxes -------------------------

const GF_BL = lele_d1_6(closures=:brady_livescu)
const GFLUX = (("default", (;)), ("BL", (interface_divergence=GF_BL,)),
               ("ghost", (interface_flux=:ghost,)))
# The viscous rows add the ghost path with the Brady–Livescu rows on its remainder.
const GFLUX_VISCOUS = (GFLUX..., ("ghost+BL", (interface_flux=:ghost,
                                              interface_divergence=GF_BL)))

function gflux_study()
    println("\n=== divergence through interface ends: ghost fluxes vs closure rows ===")
    # Polynomial consistency: every inviscid flux component of degree <= 5,
    # which the interior rows and the level interpolation reproduce.
    println("\npolynomial RHS on exact data, N = 96, interface window (rho, rho u, E)")
    for levels in (1, 2), (label, kw) in GFLUX
        s, st = polynomial_case(96; levels=levels,
                                patch_grid=levels == 1 ? (2, 1, 1) : (1, 1, 1), kw...)
        ex = exact_rhs(s.equations, polynomial_profile())
        es = [rhs_errors(s, st, ex; comp=c) for c in (1, s.equations.i_mom[1],
                                                        s.equations.i_energy)]
        @printf("  %-12s %-8s %s   interior %s\n",
                levels == 1 ? "two patches" : "2 levels", label,
                join((@sprintf("%.2e", e.interface) for e in es), " "),
                join((@sprintf("%.2e", e.interior) for e in es), " "))
    end
    for (label, kw) in GFLUX, levels in (1, 2)
        what = levels == 1 ? "two patches" : "2 levels"
        rhs_row("RHS entropy wave k=3, $what, $label",
                N -> entropy_case(N; levels=levels,
                                  patch_grid=levels == 1 ? (2, 1, 1) : (1, 1, 1), kw...),
                s -> exact_rhs(s.equations, entropy_profile(3, 0.37)), NS_PERIODIC;
                primary=:interface)
    end
    fine_inviscid = fine_periodic(mu=0.0)
    for (label, kw) in GFLUX, (flabel, fopts) in IDIV_FILTERS
        for (k, phase) in ((3, 0.37), (1, 0.0))
            evolution_study("entropy wave k=$k, two patches, $label,$flabel",
                            (N; cfl) -> entropy_case(N; k=k, phase=phase,
                                                     patch_grid=(2, 1, 1), cfl=cfl,
                                                     kw..., fopts...),
                            entropy_reference(k, phase), NS_PERIODIC;
                            primary=:interface, tfinal=TFINAL_ENTROPY)
        end
        evolution_study("inviscid standing wave, two patches, $label,$flabel",
                        (N; cfl) -> viscous_periodic_case(N; mu=0.0, patch_grid=(2, 1, 1),
                                                          cfl=cfl, kw..., fopts...),
                        (s, cfl) -> fine_inviscid, NS_PERIODIC; primary=:interface)
    end
    for (label, kw) in GFLUX
        for levels in (2, 3), subcycle in (false, true)
            evolution_study("entropy wave k=3, $levels levels, subcycle=$subcycle, " *
                            "$label, unfiltered",
                            (N; cfl) -> entropy_case(N; levels=levels, subcycle=subcycle,
                                                     cfl=cfl, filter_interval=0, kw...),
                            entropy_reference(3, 0.37), NS_PERIODIC;
                            primary=:interface, tfinal=TFINAL_ENTROPY)
        end
        evolution_study("entropy wave k=3, 2 levels, $label, default filter",
                        (N; cfl) -> entropy_case(N; levels=2, cfl=cfl, filter_interval=1,
                                                 kw...),
                        entropy_reference(3, 0.37), NS_PERIODIC;
                        primary=:interface, tfinal=TFINAL_ENTROPY)
        evolution_study("entropy wave k=1, 2 levels, $label, unfiltered",
                        (N; cfl) -> entropy_case(N; k=1, phase=0.0, levels=2, cfl=cfl,
                                                 kw...),
                        entropy_reference(1, 0.0), NS_PERIODIC;
                        primary=:interface, tfinal=TFINAL_ENTROPY)
        for subcycle in (false, true)
            evolution_study("inviscid standing wave, 2 levels, subcycle=$subcycle, " *
                            "$label, unfiltered",
                            (N; cfl) -> viscous_periodic_case(N; mu=0.0, levels=2,
                                                              subcycle=subcycle, cfl=cfl,
                                                              kw...),
                            (s, cfl) -> fine_inviscid, NS_PERIODIC; primary=:interface)
        end
        evolution_study("entropy wave k=3, 2 levels tiled (tile 4), subcycled, " *
                        "$label, unfiltered",
                        (N; cfl) -> entropy_case(N; levels=2, subcycle=true, tile=4,
                                                 cfl=cfl, kw...),
                        entropy_reference(3, 0.37), NS_PERIODIC;
                        primary=:interface, tfinal=TFINAL_ENTROPY)
    end
    fine = fine_mirror(false)
    for (label, kw) in GFLUX, (flabel, fopts) in IDIV_FILTERS
        evolution_study("inviscid wall + interface, two patches, $label,$flabel",
                        (N; cfl) -> wall_case(N; patch_grid=(2, 1, 1), cfl=cfl,
                                              kw..., fopts...),
                        (s, cfl) -> fine, NS; primary=:interface)
    end
    fine_viscous = fine_periodic()
    for (label, kw) in GFLUX_VISCOUS,
        (what, lkw) in (("two patches", (patch_grid=(2, 1, 1),)), ("2 levels", (levels=2,)),
                        ("2 levels subcycled", (levels=2, subcycle=true)))
        evolution_study("viscous standing wave, $what, $label, unfiltered",
                        (N; cfl) -> viscous_periodic_case(N; cfl=cfl, lkw..., kw...),
                        (s, cfl) -> fine_viscous, NS_PERIODIC; primary=:interface)
    end
    for (label, kw) in (("default", (;)), ("ghost", (interface_flux=:ghost,))),
        (what, lkw) in (("two patches", (patch_grid=(2, 1, 1),)), ("2 levels", (levels=2,)))
        evolution_study("Float32 entropy wave k=1, $what, $label, unfiltered",
                        (N; cfl) -> entropy_case(N; k=1, phase=0.0, cfl=cfl,
                                                 precision=Float32,
                                                 deriv=lele_d1_6(Float32),
                                                 filt=compact_filter(0.45, Float32),
                                                 kw..., lkw...),
                        entropy_reference(1, 0.0), NS_PERIODIC;
                        primary=:interface, tfinal=TFINAL_ENTROPY)
    end
    gflux_pulse()
end

function gflux_pulse()
    println("\nacoustic pulse, left-running characteristic upstream of the faces / amplitude")
    println("   N    layout                    variant     reflected")
    r1(N) = BlockRegion((80N ÷ 192, 0, 0), (32N ÷ 192 + 1, 1, 1))
    for N in (96, 192, 384)
        base = pulse_leftgoing(N)
        for (what, kw) in (("two patches", (patch_grid=(2, 1, 1),)),
                           ("2 levels", (refine=r1(N),)),
                           ("2 levels subcycled", (refine=r1(N), subcycle=true)))
            for (label, vkw) in GFLUX
                r = pulse_leftgoing(N; kw..., vkw...)
                @printf("%5d    %-24s  %-10s  %.3e\n", N, what, label,
                        maximum(abs.(r .- base)))
            end
        end
        flush(stdout)
    end
end

function main()
    t0 = time()
    selected("truncation") && truncation_study()
    selected("rhs") && rhs_study()
    selected("walls") && walls_study()
    selected("shear") && shear_study()
    selected("interfaces") && interfaces_study()
    selected("levels") && levels_study()
    selected("fields") && fields_study()
    selected("filter") && filter_study()
    selected("dt") && dt_study()
    selected("idiv") && idiv_study()
    # Run only when named: they repeat the level rows once per order.
    named("transfer") && transfer_study()
    named("transfertime") && transfer_time_study()
    selected("gflux") && gflux_study()
    @printf("\ndone in %.1f s\n", time() - t0)
end

abspath(PROGRAM_FILE) == (@__FILE__) && main()
