# The order of a slip wall on the axisymmetric r-z annulus, by region, and
# what sets it.
#
#   julia --project=. -t 1 bench/annuluswall.jl [study=all] [ns=25,49,97,193]
#                                               [refine=4] [cfl=0.4]
#
# `study` is one of, or a comma-separated list of,
#
#   evolution   the solution error at the final time against a nested fine
#               run of the same case in the same equal steps, so the error
#               is the spatial operator's alone: the wall node, the rest of
#               the wall window and the interior separately, for density
#               and radial momentum. The cases:
#                 standing     the standing wave of `standing_profile` on
#                              r ∈ [r0, r0 + 1], at r0 = 0.5, 5 and 50, and
#                              on the Cartesian line
#                 bent         the Cartesian line with u = b sin πx − πb
#                              x(1 − x), whose u_xx at the walls matches the
#                              curvature term u_r/r of the annulus standing
#                              wave at r0 = 0.5
#                 rest         ρ = 1 + a cos π(r − r0) at rest, isentropic
#                 pulse        `annulus_wall_case`: a Gaussian pulse at rest
#                              at the middle of the gap, before the walls at
#                              t = 0.2, on them at 0.45 and reflected by 0.7
#               then the reflected pulse under the default filter every step
#               and under the `:brady_livescu` derivative rows
#   gate        the convergence row of test/convergence.jl: the pulse on the
#               walls at t = 0.45, wall window, against the run on nested
#               nodes at a third of the spacing in 4(N − 1) equal steps, on
#               the annulus and on the Cartesian line
#   rhs         the right-hand side on exact data against the exact
#               axisymmetric Euler right-hand side, by region, on the
#               annulus and the Cartesian line: the derivative's wall rows,
#               the area weighting and the curvature source together, before
#               any evolution
#
# Why the cases. A slip wall holds u_r = 0 at every time, so the data must
# satisfy ∂ₜᵏ u_r = 0 at the wall for every k for the solution to be smooth
# up to the wall (the compatibility conditions of the initial-boundary value
# problem). For the standing wave ∂ₜ(ρu) = −p_r vanishes at the walls, but
# ∂ₜ²(ρu) = γp(u_rr + u_r/r − u/r²) = ±γpπb/r there: on the Cartesian line
# the data are even and odd about each wall and satisfy every condition, and
# on the annulus the curvature term u_r/r breaks the second one. The solution
# then carries a jump in its second derivatives along the characteristics
# from the corners (t = 0, r = wall), and every consistent scheme converges
# to it at a reduced order. The data at rest break only the third condition
# (γp(p_rrr + p_rr/r)/ρ ≠ 0), and the pulse none to the precision measured.
# `bent` breaks the second condition on the Cartesian line, where there is no
# curvature anywhere in the scheme.
#
# Every run of a row takes the same number of equal steps, set by `cfl` at
# the reference's spacing, so the coarse runs carry the reference's time
# error and the difference is spatial. `refine` is the reference's
# refinement of the finest grid. Scratch tooling: it prints tables and
# asserts nothing.

using MPI
MPI.Init(threadlevel=:funneled)
using CompactLES, Printf
using CompactLES: padded_index, xcoord
const CL = CompactLES
MPI.Comm_size(MPI.COMM_WORLD) == 1 || error("run this study on one rank")
include(joinpath(@__DIR__, "..", "test", "smooth_cases.jl"))

const OPTS = CL.script_args(ARGS, (study="all", ns="25,49,97,193", refine=4, cfl=0.4))
const NS = parse.(Int, split(OPTS.ns, ','))
const GAMMA = 1.4
const AMP = 0.05

selected(name) = OPTS.study == "all" || name in split(OPTS.study, ',')

# --- profiles in the wall-normal coordinate s = r − r0 ∈ [0, 1] ----------------

bent_profile(a, b) = s -> begin
    rho = 1 + a * cos(pi * s)
    (rho, b * sin(pi * s) - pi * b * s * (1 - s), zero(s), rho^GAMMA)
end

rest_profile(a) = s -> begin
    rho = 1 + a * cos(pi * s)
    (rho, zero(s), zero(s), rho^GAMMA)
end

const PROFILES = Dict("standing" => standing_profile(AMP, AMP),
                      "bent" => bent_profile(AMP, AMP),
                      "rest" => rest_profile(AMP))

# r0 = nothing is the Cartesian line on [0, 1]; the pulse is the shared case.
function annulus_case(N, prof, r0; opts...)
    prof == :pulse && return annulus_wall_case(N; cartesian=r0 === nothing,
                                               r0=something(r0, 0.5),
                                               merge((cfl=0.95,), values(opts))...)
    wall = SlipWallBC()
    shift = r0 === nothing ? 0.0 : r0
    geometry = r0 === nothing ? (;) : (metric=CylindricalMetric(), origin=(r0, 0.0, 0.0))
    _smooth_solver((N, 1, 1), 1.0, ((wall, wall), per3[2], per3[3]), r -> prof(r - shift);
                   merge(SMOOTH_DEFAULTS, (cfl=0.95,), geometry, values(opts))...)
end

# --- regional errors: the wall nodes, the rest of the wall windows, the interior -

function wall_regions(solver, Q, reference, comp)
    n = solver.decomp.n_local[1]
    node = window = interior = 0.0
    for i in 1:n
        I = padded_index(solver, i, 1, 1)
        e = abs(Q[I, comp] - reference(xcoord(solver, 1, i))[comp])
        if i == 1 || i == n
            node = max(node, e)
        elseif i <= SMOOTH_W || i > n - SMOOTH_W
            window = max(window, e)
        else
            interior = max(interior, e)
        end
    end
    return (node=node, window=window, interior=interior)
end

function print_table(title, hs, rows)
    println("\n", title)
    println("   N   ρ node     ρ window   ρ interior  ρu window  ρu interior")
    for (k, h) in enumerate(hs)
        r, m = rows[k]
        @printf("%4d  %.3e  %.3e  %.3e   %.3e  %.3e\n", round(Int, 1 / h) + 1,
                r.node, r.window, r.interior, m.window, m.interior)
    end
    length(hs) < 2 && return
    ord(f) = observed_order(hs, [f(rows[k]) for k in eachindex(hs)])
    @printf("order %.2f      %.2f       %.2f        %.2f       %.2f\n",
            ord(x -> x[1].node), ord(x -> x[1].window), ord(x -> x[1].interior),
            ord(x -> x[2].window), ord(x -> x[2].interior))
end

function evolution_row(title, prof, r0, tfinal; opts...)
    t0 = time()
    Nref = OPTS.refine * (maximum(NS) - 1) + 1
    href = 1 / (Nref - 1)
    steps = ceil(Int, tfinal * 1.3 / (OPTS.cfl * href))
    ref, rstates = annulus_case(Nref, prof, r0; opts...)
    fixed_step_run!(ref, rstates, tfinal, steps)
    reference = NodeReference(ref, rstates)
    m1 = ref.equations.i_mom[1]
    hs = Float64[]; rows = []
    for N in NS
        solver, states = annulus_case(N, prof, r0; opts...)
        fixed_step_run!(solver, states, tfinal, steps)
        push!(hs, solver.h[1])
        push!(rows, (wall_regions(solver, states, reference, 1),
                     wall_regions(solver, states, reference, m1)))
    end
    print_table(@sprintf("%s  (t = %.2f, %d steps, reference N = %d, %.0f s)",
                         title, tfinal, steps, Nref, time() - t0), hs, rows)
end

where(r0) = r0 === nothing ? "Cartesian line" : "annulus r0 = $r0"

function evolution_study()
    println("\n=== evolution: inviscid, unfiltered, no artificial properties ===")
    for r0 in (nothing, 0.5, 5.0, 50.0)
        evolution_row("standing wave, $(where(r0))", PROFILES["standing"], r0, 0.4)
    end
    evolution_row("bent standing wave, Cartesian line", PROFILES["bent"], nothing, 0.4)
    for r0 in (0.5, 5.0)
        evolution_row("cosine at rest, $(where(r0))", PROFILES["rest"], r0, 0.4)
    end
    for r0 in (nothing, 0.5)
        evolution_row("pulse before the walls, $(where(r0))", :pulse, r0, 0.2)
        evolution_row("pulse on the walls, $(where(r0))", :pulse, r0, 0.45)
        evolution_row("pulse reflected, $(where(r0))", :pulse, r0, 0.7)
    end
    println("\n=== evolution: the reflected pulse under the filter and other rows ===")
    for r0 in (nothing, 0.5)
        evolution_row("pulse reflected, default filter every step, $(where(r0))",
                      :pulse, r0, 0.7; filter_interval=1)
        evolution_row("pulse reflected, C6 :brady_livescu, $(where(r0))",
                      :pulse, r0, 0.7;
                      deriv=lele_d1_6(closures=:brady_livescu))
    end
end

# --- the right-hand side on exact data -------------------------------------------

"x → the exact inviscid axisymmetric right-hand side of `prof` at r = r0 + s
(r0 = nothing: the Cartesian one), in the solver's component order."
function exact_annulus_rhs(equations, prof, r0)
    r0 === nothing && return exact_rhs(equations, prof)
    function flux(r)
        rho, u, v, p = prof(r - r0)
        E = p / (GAMMA - 1) + rho * u * u / 2
        (r * rho * u, r * rho * u * u, r * (E + p) * u)
    end
    m1 = equations.i_mom[1]; ie = equations.i_energy
    return r -> begin
        f = ntuple(c -> -derivative(ρ -> flux(ρ)[c], r) / r, 3)
        dp = derivative(ρ -> prof(ρ - r0)[4], r)
        vals = zeros(typeof(r), equations.n_cons)
        vals[1] = f[1]; vals[m1] = f[2] - dp; vals[ie] = f[3]
        Tuple(vals)
    end
end

function rhs_study()
    println("\n=== right-hand side on exact data, inviscid ===")
    for (name, prof) in (("standing", PROFILES["standing"]), ("rest", PROFILES["rest"]))
        for r0 in (nothing, 0.5, 5.0)
            hs = Float64[]; rows = []
            for N in NS
                solver, states = annulus_case(N, prof, r0)
                exact = exact_annulus_rhs(solver.equations, prof, r0)
                dQ = zero(states)
                apply_bcs!(solver, states)
                compute_rhs!(solver, states, dQ)
                push!(hs, solver.h[1])
                m1 = solver.equations.i_mom[1]
                push!(rows, (wall_regions(solver, dQ, exact, 1),
                             wall_regions(solver, dQ, exact, m1)))
            end
            print_table("dQ/dt error, $name, $(where(r0))", hs, rows)
        end
    end
end

function gate_study()
    println("\n=== the convergence row: the pulse on the walls, t = 0.45 ===")
    for cartesian in (true, false)
        hs = Float64[]; walls = Float64[]; interiors = Float64[]
        for N in NS
            steps = 4 * (N - 1)
            solver, states = annulus_wall_case(N; cartesian=cartesian)
            fixed_step_run!(solver, states, 0.45, steps)
            fine, fstates = annulus_wall_case(3 * (N - 1) + 1; cartesian=cartesian)
            fixed_step_run!(fine, fstates, 0.45, steps)
            e = regional_errors(solver, states, NodeReference(fine, fstates))
            push!(hs, solver.h[1]); push!(walls, e.wall); push!(interiors, e.interior)
            @printf("%s N = %4d  wall %.3e  interior %.3e\n",
                    cartesian ? "Cartesian" : "annulus  ", N, e.wall, e.interior)
        end
        @printf("order: wall %.2f  interior %.2f\n", observed_order(hs, walls),
                observed_order(hs, interiors))
    end
end

selected("gate") && gate_study()
selected("rhs") && rhs_study()
selected("evolution") && evolution_study()
