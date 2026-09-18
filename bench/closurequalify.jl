# Independent production checks for the fifth-order closure search.
#
#   julia --project=. -t 1 bench/closurequalify.jl parts=polynomial,smooth
#   julia --project=. -t 1 bench/closurequalify.jl schemes=candidate parts=jacobian
#   julia --project=. -t 1 bench/closurequalify.jl schemes=candidate parts=stress
#   julia --project=. -t 1 bench/closurequalify.jl schemes=de parts=jacobian
#   julia --project=. -t 1 bench/closurequalify.jl parts=dilatation \
#       schemes=neutral3,brady_livescu,unfiltered,candidate,de \
#       beta_sensor=dilatation
#
# The candidate is supplied by the include-safe ClosureSearch module. Controls
# are the current neutral3 default and the published Brady-Livescu T6 rows.
# Polynomial errors use the production derivative; evolution errors use the
# shared smooth cases against a periodic mirror, isolating the wall defect.
# These are research measurements, not acceptance tests or default selection.

module ClosureQualification

using MPI, CompactLES, LinearAlgebra, Printf
const CL = CompactLES
include(joinpath(@__DIR__, "..", "test", "references.jl"))
include(joinpath(@__DIR__, "..", "test", "cases.jl"))
include(joinpath(@__DIR__, "..", "test", "smooth_cases.jl"))

function scheme_named(name)
    name == "neutral3" && return lele_d1_6()
    name == "brady_livescu" && return lele_d1_6(closures=:brady_livescu)
    name in ("candidate", "unfiltered", "de") || error("unknown scheme $name")
    isdefined(@__MODULE__, :ClosureSearch) ||
        error("candidate requires bench/closuresearch.jl")
    return name == "unfiltered" ? ClosureSearch.unfiltered_scheme() :
           name == "de" ? ClosureSearch.de_scheme() :
           ClosureSearch.candidate_scheme()
end

function polynomial_check(name, deriv, ns)
    # Normalize moments by their absolute term sum: large monomials near the
    # far end otherwise obscure floating-point cancellation in exact rows.
    defect = 0.0
    for (j, row) in enumerate(deriv.closures), degree in 0:5
        rhs_terms = [w * Float64(k - 1)^degree for (k, w) in enumerate(row.rhs)]
        lhs_terms = degree == 0 ? zeros(3) :
            [row.lhs[k] * degree * Float64(j + k - 3)^(degree - 1) for k in 1:3]
        scale = max(1.0, sum(abs, rhs_terms) + sum(abs, lhs_terms))
        defect = max(defect, abs(sum(rhs_terms) - sum(lhs_terms)) / scale)
    end
    @printf("moments %-15s normalized degree 0:5 defect %.3e\n", name, defect)
    errors = Float64[]
    for n in ns
        e = closed_derivative_errors(n, deriv, x -> x^6, x -> 6x^5)
        push!(errors, e.wall)
        @printf("polynomial %-15s N=%4d wall %.6e interior %.6e\n",
                name, n, e.wall, e.interior)
    end
    println("polynomial ", name, " orders ", successive_orders(1.0 ./ (ns .- 1), errors))
end

function smooth_check(name, deriv, opts)
    ns = parse.(Int, split(opts.ns, ','))
    for viscous in (false, true), art_on in (false, true), filtered in (false, true)
        for cfl in (opts.cfl, opts.cfl / 2)
            errors = Float64[]
            for n in ns
                settings = (deriv=deriv, viscous=viscous,
                            art=ArtParams(enabled=art_on,
                                          beta_sensor=Symbol(opts.beta_sensor)),
                            cfl=cfl,
                            filter_interval=filtered ? 1 : 0)
                solver, state = wall_case(n; settings...)
                mirror, mirrored = mirror_case(n; settings...)
                try
                    run!(solver, state; tfinal=opts.tfinal, nmax=opts.nmax)
                    run!(mirror, mirrored; tfinal=opts.tfinal, nmax=opts.nmax)
                    e = regional_errors(solver, state, NodeReference(mirror, mirrored))
                    push!(errors, e.wall)
                    Printf.format(stdout, Printf.Format(
                            "smooth %-15s viscous=%s art=%s filter=%s cfl=%.3f " *
                            "N=%4d wall %.6e interior %.6e l2 %.6e\n"),
                            name, viscous, art_on, filtered, cfl, n,
                            e.wall, e.interior, e.l2)
                catch err
                    err isa SolverFailure || rethrow()
                    println("smooth ", name, " FAILED N=", n, " ", err)
                    break
                end
                flush(stdout)
            end
            if length(errors) == length(ns)
                println("smooth ", name, " orders ",
                        successive_orders(1.0 ./ (ns .- 1), errors))
            end
            flush(stdout)
        end
    end
end

# --- N6j: wall matrix with the dilatation beta sensor ----------------------
#
# Keep `smooth` above as the original compact sweep: it remains useful for
# quickly comparing a new row to the legacy strain setup.  This part names all
# of the physical wall contracts explicitly and reports the actual mesh width,
# so a failed or capped run cannot be mistaken for an error measurement.

const DILATATION_CASES = (
    :inviscid_slip,
    :viscous_noslip,
    :viscous_slip_shear,
    :shear_adiabatic,
    :shear_isothermal,
)

function smooth_art(control::AbstractString, beta_sensor::AbstractString)
    control == "off" && return ArtParams(enabled=false)
    sensor = Symbol(control == "on" ? beta_sensor : control)
    sensor in (:strain, :dilatation) ||
        error("smooth control must be on, off, strain, or dilatation; got $control")
    return ArtParams(enabled=true, beta_sensor=sensor)
end

function smooth_filter(control::AbstractString)
    control == "on" && return 1
    control == "off" && return 0
    error("smooth filter must be on or off; got $control")
end

function smooth_pair(kind::Symbol, n, settings)
    if kind === :inviscid_slip
        solver, state = wall_case(n; viscous=false, slip=true, settings...)
        mirror, mirrored = mirror_case(n; viscous=false, settings...)
        return solver, state, mirror, mirrored, nothing
    elseif kind === :viscous_noslip
        solver, state = wall_case(n; viscous=true, slip=false, settings...)
        mirror, mirrored = mirror_case(n; viscous=true, settings...)
        return solver, state, mirror, mirrored, nothing
    elseif kind === :viscous_slip_shear
        solver, state = wall_case(n; viscous=true, slip=true, c=0.05, settings...)
        mirror, mirrored = mirror_case(n; viscous=true, c=0.05, settings...)
        return solver, state, mirror, mirrored, nothing
    elseif kind === :shear_adiabatic
        solver, state = shear_case(n; Twall=NaN, settings...)
        mirror, mirrored = shear_mirror_case(n; settings...)
        return solver, state, mirror, mirrored, 3
    elseif kind === :shear_isothermal
        solver, state = shear_case(n; Twall=1.0, settings...)
        mirror, mirrored = shear_mirror_case(n; settings...)
        return solver, state, mirror, mirrored, 3
    end
    error("unknown smooth wall case $kind")
end

function smooth_failure(name, kind, control, filter, cfl, n, h, stage, solver,
                        target, detail)
    Printf.format(stdout, Printf.Format(
        "dilatation %-12s case=%-19s control=%-10s filter=%s cfl=%.3f " *
        "N=%4d h=%.6e %s FAILED endpoint t=%.8f target=%.8f step=%d: %s\n"),
        name, String(kind), control, filter, cfl, n, h, stage, solver.t,
        target, solver.step, detail)
end

function run_smooth!(solver, state, target, nmax, callback)
    callback === nothing ?
        run!(solver, state; tfinal=target, nmax=nmax) :
        run!(solver, state; tfinal=target, nmax=nmax, callback=callback)
end

function dilatation_check(name, deriv, opts; wall_callback=nothing,
                          mirror_callback=nothing)
    ns = parse.(Int, split(opts.ns, ','))
    cases = Symbol.(split(opts.smooth_cases, ','))
    all(in(DILATATION_CASES), cases) || error("unknown smooth wall case")
    controls = split(opts.smooth_controls, ',')
    filters = split(opts.smooth_filters, ',')
    for kind in cases, control in controls, filter in filters,
        cfl in (opts.cfl, opts.cfl / 2)
        art = smooth_art(control, opts.beta_sensor)
        interval = smooth_filter(filter)
        errors = Float64[]
        hs = Float64[]
        for n in ns
            settings = (deriv=deriv, art=art, cfl=cfl,
                        filter_interval=interval)
            solver, state, mirror, mirrored, component =
                smooth_pair(kind, n, settings)
            h = xcoord(solver, 1, 2) - xcoord(solver, 1, 1)
            target = opts.tfinal
            try
                run_smooth!(solver, state, target, opts.nmax, wall_callback)
            catch err
                err isa SolverFailure || rethrow()
                smooth_failure(name, kind, control, filter, cfl, n, h, "wall",
                               solver, target, sprint(showerror, err))
                break
            end
            if !completed(solver, target)
                smooth_failure(name, kind, control, filter, cfl, n, h, "wall",
                               solver, target, "step cap or non-finite clock")
                break
            end
            try
                run_smooth!(mirror, mirrored, target, opts.nmax,
                            mirror_callback)
            catch err
                err isa SolverFailure || rethrow()
                smooth_failure(name, kind, control, filter, cfl, n, h, "mirror",
                               mirror, target, sprint(showerror, err))
                break
            end
            if !completed(mirror, target)
                smooth_failure(name, kind, control, filter, cfl, n, h, "mirror",
                               mirror, target, "step cap or non-finite clock")
                break
            end
            e = component === nothing ?
                regional_errors(solver, state, NodeReference(mirror, mirrored)) :
                regional_errors(solver, state, NodeReference(mirror, mirrored);
                                comp=component)
            push!(errors, e.wall)
            push!(hs, h)
            Printf.format(stdout, Printf.Format(
                "dilatation %-12s case=%-19s control=%-10s filter=%s cfl=%.3f " *
                "N=%4d h=%.6e wall %.6e interior %.6e l2 %.6e\n"),
                name, String(kind), control, filter, cfl, n, h,
                e.wall, e.interior, e.l2)
            flush(stdout)
        end
        if length(errors) == length(ns)
            println("dilatation ", name, " case=", kind, " control=", control,
                    " filter=", filter, " cfl=", cfl, " orders ",
                    successive_orders(hs, errors))
        end
        flush(stdout)
    end
end

function uniform_solver(deriv, n; filtered=false, wall=:slip)
    rho, pressure = 0.9, 1.1
    transverse = wall == :noslip ? 0.0 : 0.1
    profile = (x, y, z, t) -> Prim(rho=rho, u=(0.0, transverse, 0.0), p=pressure)
    bc = wall == :dirichlet ? DirichletBC(profile) :
         wall == :noslip ? NoSlipWallBC() : SlipWallBC()
    solver = Solver(n_global=(n, 1, 1), L_domain=(1.0, 1.0, 1.0),
                    bcs=((bc, bc), per3[2], per3[3]), deriv=deriv,
                    eos=IdealSpecies("gas"; R=1.0, gamma=1.4),
                    art=ArtParams(enabled=false),
                    transport=Transport(mu0=wall == :noslip ? 0.005 : 0.0),
                    filter_interval=filtered ? 1 : 0, filter_cfl=0.0, cfl=0.5)
    state = allocate_state(solver)
    initialize!(solver, state, (x, y, z) -> profile(x, y, z, 0.0))
    apply_bcs!(solver, state)
    return solver, state, sqrt(1.4 * pressure / rho)
end

function production_jacobian(deriv, n; filtered=false, delta=1e-5, wall=:slip,
                             poststep=nothing)
    solver, initial, sound = uniform_solver(deriv, n; filtered=filtered, wall=wall)
    dt = 0.5 / ((n - 1) * sound)
    indices = [(gidx(solver, i, 1, 1), c)
               for c in 1:solver.equations.n_cons for i in 1:n]
    function step_map(state)
        CL.step!(solver, state, zero(state), zero(state), dt)
        filtered && filter_state!(solver, state)
        poststep === nothing || poststep(solver,state)
        return state
    end
    matrix = zeros(length(indices), length(indices))
    for (j, (index, component)) in enumerate(indices)
        positive, negative = copy(initial), copy(initial)
        epsilon = delta * max(abs(initial[index, component]), 1.0)
        positive[index, component] += epsilon
        negative[index, component] -= epsilon
        step_map(positive)
        step_map(negative)
        for (i, (target, c)) in enumerate(indices)
            matrix[i, j] = (positive[target, c] - negative[target, c]) / (2epsilon)
        end
    end
    return matrix, dt
end

function jacobian_check(name, deriv, opts)
    walls = Symbol.(split(opts.jwalls, ','))
    all(w -> w in (:slip,:dirichlet,:noslip),walls) || error("unknown Jacobian wall")
    for n in parse.(Int, split(opts.jns, ',')), wall in walls
        for filtered in (false, true), delta in parse.(Float64,split(opts.deltas,','))
            matrix, dt = production_jacobian(deriv, n; filtered, delta, wall)
            radius = maximum(abs, eigvals(matrix))
            Printf.format(stdout, Printf.Format(
                    "jacobian %-15s N=%4d wall=%s filter=%s delta=%.0e " *
                    "radius %.12f rate %+.6e\n"),
                    name, n, wall, filtered, delta, radius, log(radius) / dt)
            flush(stdout)
        end
    end
end

function uniform_check(name, deriv, opts)
    for n in parse.(Int, split(opts.jns, ',')), filtered in (false, true)
        solver, state, _ = uniform_solver(deriv, n; filtered=filtered)
        initial = copy(state)
        try
            run!(solver, state; tfinal=opts.longtime, nmax=opts.nmax)
            velocity = maximum(abs(state[gidx(solver, i, 1, 1), 2] /
                                   state[gidx(solver, i, 1, 1), 1]) for i in 1:n)
            drift = maximum(abs(state[gidx(solver, i, 1, 1), c] -
                                initial[gidx(solver, i, 1, 1), c])
                            for i in 1:n for c in 1:solver.equations.n_cons)
            @printf("uniform %-15s N=%4d filter=%s t=%.1f |u| %.6e drift %.6e\n",
                    name, n, filtered, solver.t, velocity, drift)
        catch err
            err isa SolverFailure || rethrow()
            println("uniform ", name, " FAILED N=", n, " ", err)
        end
        flush(stdout)
    end
end

function stress_check(name, deriv, opts)
    sensor = Symbol(opts.beta_sensor)
    sensor in (:strain, :dilatation) ||
        error("beta_sensor must be strain or dilatation; got $(opts.beta_sensor)")
    art = ArtParams(enabled=true, beta_sensor=sensor)
    for start in (0.0, 0.1)
        try
            result = noh_case(1; N=opts.stressn, t0=start, deriv=deriv,
                              art=art, cfl=0.3, nmax=opts.nmax)
            println("stress ", name, " beta_sensor=", sensor,
                    " planar Noh start=", start, " completed=", result[5],
                    " ", result[6])
        catch err
            err isa SolverFailure || rethrow()
            println("stress ", name, " beta_sensor=", sensor,
                    " planar Noh start=", start, " FAILED ", err)
        end
        flush(stdout)
    end
    try
        result = woodward(; N=opts.stressn, deriv=deriv, art=art,
                          cfl=0.3, nmax=opts.nmax)
        println("stress ", name, " beta_sensor=", sensor,
                " Woodward-Colella completed=", result[5])
    catch err
        err isa SolverFailure || rethrow()
        println("stress ", name, " beta_sensor=", sensor,
                " Woodward-Colella FAILED ", err)
    end
end

function main(args=ARGS)
    MPI.Initialized() || MPI.Init(threadlevel=:funneled)
    MPI.Comm_size(MPI.COMM_WORLD) == 1 || error("run qualification on one rank")
    BLAS.set_num_threads(1)
    candidate_file = joinpath(@__DIR__, "closuresearch.jl")
    isfile(candidate_file) && include(candidate_file)
    opts = CL.script_args(args, (parts="polynomial,smooth", schemes="neutral3,brady_livescu",
        ns="49,97,193", pns="17,33,65,129", jns="51,101", cfl=0.25, tfinal=0.4,
        longtime=40.0, nmax=30000, stressn=200,
        jwalls="slip,dirichlet,noslip", deltas="3e-6,1e-5,3e-5",
        smooth_cases=("inviscid_slip,viscous_noslip,viscous_slip_shear," *
                      "shear_adiabatic,shear_isothermal"),
        smooth_controls="dilatation,strain,off", smooth_filters="on",
        beta_sensor="strain"))
    for name in split(opts.schemes, ',')
        deriv = Base.invokelatest(scheme_named, name)
        println("scheme ", name, " = ", deriv.name)
        for (j, row) in enumerate(deriv.closures)
            println("row ", j, " lhs=", row.lhs, " rhs=", row.rhs)
        end
        for part in split(opts.parts, ',')
            part == "polynomial" ? polynomial_check(name, deriv,
                parse.(Int, split(opts.pns, ','))) :
            part == "smooth" ? smooth_check(name, deriv, opts) :
            part == "dilatation" ? dilatation_check(name, deriv, opts) :
            part == "jacobian" ? jacobian_check(name, deriv, opts) :
            part == "uniform" ? uniform_check(name, deriv, opts) :
            part == "stress" ? stress_check(name, deriv, opts) :
            error("unknown qualification part $part")
        end
    end
end

end # module

abspath(PROGRAM_FILE) == abspath(@__FILE__) && ClosureQualification.main()
