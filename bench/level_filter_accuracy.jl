# Accuracy and reflection qualification for the experimental level-aware
# filter policies in `level_filter_policy.jl`. This file is included by
# `bench/levelfilter.jl` after that driver selects a policy.
# `ns=48,96,192` selects root sizes, `parts=smooth,reflection` the studies,
# and `nmax=20000` bounds each run. At eight ranks use `ns=96,192`: a
# 48-node root cannot provide nine filter nodes to each rank.

using CompactLES
using CompactLES: padded_index, xcoord
using MPI
using Printf

MPI.Initialized() || MPI.Init(threadlevel=:funneled)

include(joinpath(@__DIR__, "..", "test", "smooth_cases.jl"))

function _lf_options(args)
    opts = CompactLES.script_args(args,
        (parts="all", study="", ns="48,96,192", nmax=20000))
    opts.study == "" || opts.parts == "all" ||
        throw(ArgumentError("pass either parts= or study=, not both"))
    selection = opts.study == "" ? opts.parts : opts.study
    parts = Set(Symbol.(strip.(split(selection, ','))))
    all(p -> p in (:all, :smooth, :reflection), parts) ||
        throw(ArgumentError("parts must contain smooth, reflection, or all"))
    :all in parts && length(parts) != 1 &&
        throw(ArgumentError("all cannot be combined with another part"))
    ns = parse.(Int, strip.(split(opts.ns, ',')))
    isempty(ns) && error("ns must contain at least one root-grid size")
    all(n -> n > 0 && n % 24 == 0, ns) ||
        error("every ns entry must be a positive multiple of 24")
    issorted(ns) && allunique(ns) ||
        error("ns must be strictly increasing")
    all(diff(ns) .> 0) ||
        error("ns entries must be distinct and strictly increasing")
    opts.nmax > 0 || error("nmax must be positive")
    return parts, ns, opts.nmax
end

_lf_selected(parts, part) = :all in parts || part in parts
_lf_rank() = MPI.Comm_rank(MPI.COMM_WORLD)

function _lf_say(fmt, values...)
    _lf_rank() == 0 || return nothing
    Printf.format(stdout, Printf.Format(fmt), values...)
    flush(stdout)
    return nothing
end

function _lf_regional_errors(solver, states, reference; comp=1, W=SMOOTH_W)
    patches = getfield(solver, :patches)
    per_patch = states isa Vector ? states : [states]
    wall = interface = covered = interior = 0.0
    sq = Vector{Array{Float64,3}}(undef, length(patches))
    for (pi, patch) in enumerate(patches)
        ps = CompactLES.PatchSolver(solver, patch)
        Q = per_patch[pi]
        decomp = ps.decomp
        lo_wall = _is_wall(patch.bcs[1][1])
        hi_wall = _is_wall(patch.bcs[1][2])
        lo_interface = patch.bcs[1][1] isa CompactLES.InterfaceBC
        hi_interface = patch.bcs[1][2] isa CompactLES.InterfaceBC
        e2 = zeros(size(Q, 1), size(Q, 2), size(Q, 3))
        for i in 1:decomp.n_local[1]
            I = padded_index(ps, i, 1, 1)
            e = abs(Q[I, comp] - reference(xcoord(ps, 1, i))[comp])
            e2[I] = e * e
            # `i` is rank-local. Classify a window from the node's index in
            # the whole patch, or MPI block boundaries become false interfaces.
            node = decomp.offset[1] + i
            if ps.covered[I] != 0
                covered = max(covered, e)
            elseif (node <= W && lo_wall) ||
                   (node > decomp.n_global[1] - W && hi_wall)
                wall = max(wall, e)
            elseif (node <= W && lo_interface) ||
                   (node > decomp.n_global[1] - W && hi_interface)
                interface = max(interface, e)
            else
                interior = max(interior, e)
            end
        end
        sq[pi] = e2
    end
    maxima = MPI.Allreduce([wall, interface, covered, interior], max,
                           solver.comm)
    integral = states isa Vector ? volume_integral(solver, sq) :
                                  volume_integral(solver, sq[1])
    l2 = sqrt(integral / domain_volume(solver))
    return (wall=maxima[1], interface=maxima[2], covered=maxima[3],
            interior=maxima[4], l2=l2)
end

function _lf_smooth_run(N, depth, subcycle, cfl, nmax)
    solver, states = entropy_case(N; levels=depth, subcycle=subcycle,
                                  cfl=cfl, filter_interval=1,
                                  filter_cfl=0.35)
    run!(solver, states; tfinal=0.5, nmax=nmax)
    solver.t >= 0.5 - 16eps(0.5) || error("smooth evolution did not reach t=0.5")
    reference = analytic_reference(
        solver.equations, entropy_profile(3, 0.37; t=solver.t))
    errors = _lf_regional_errors(solver, states, reference; comp=1)
    return (; solver, errors, h=root_spacing(solver))
end

function _lf_smooth_study(ns, nmax)
    cfls = (0.25, 0.125)
    completed = 0
    expected = 2 * 2 * length(ns) * length(cfls)
    for depth in (2, 3), subcycle in (false, true)
        rows = Dict{Float64,Vector{Any}}(cfl => Any[] for cfl in cfls)
        for N in ns, cfl in cfls
            result = _lf_smooth_run(N, depth, subcycle, cfl, nmax)
            push!(rows[cfl], result)
            completed += 1
            e = result.errors
            _lf_say("LF_SMOOTH depth=%d subcycle=%s N=%d cfl=%.6g " *
                    "steps=%d h=%.9e interface=%.9e covered=%.9e " *
                    "interior=%.9e l2=%.9e complete=true\n",
                    depth, string(subcycle), N, cfl, result.solver.step,
                    result.h, e.interface, e.covered, e.interior, e.l2)
        end
        for cfl in cfls
            rs = rows[cfl]
            complete = length(rs) == length(ns)
            if complete && length(rs) >= 2
                hs = [r.h for r in rs]
                interface = [r.errors.interface for r in rs]
                l2 = [r.errors.l2 for r in rs]
                _lf_say("LF_SMOOTH_ORDER depth=%d subcycle=%s cfl=%.6g " *
                        "interface=%.6f l2=%.6f complete=true\n", depth,
                        string(subcycle), cfl, observed_order(hs, interface),
                        observed_order(hs, l2))
            else
                _lf_say("LF_SMOOTH_ORDER depth=%d subcycle=%s cfl=%.6g " *
                        "available=false complete=%s\n", depth,
                        string(subcycle), cfl, string(complete))
            end
        end
        full, half = rows[cfls[1]], rows[cfls[2]]
        if length(full) == length(ns) && length(half) == length(ns)
            for (N, a, b) in zip(ns, full, half)
                ei = abs(b.errors.interface - a.errors.interface) /
                     max(a.errors.interface, 1e-300)
                el2 = abs(b.errors.l2 - a.errors.l2) /
                      max(a.errors.l2, 1e-300)
                _lf_say("LF_SMOOTH_DT depth=%d subcycle=%s N=%d " *
                        "interface_rel=%.9e l2_rel=%.9e complete=true\n",
                        depth, string(subcycle), N, ei, el2)
            end
        else
            _lf_say("LF_SMOOTH_DT depth=%d subcycle=%s complete=false\n",
                    depth, string(subcycle))
        end
    end
    complete = completed == expected
    _lf_say("LF_SMOOTH_COMPLETE rows=%d expected=%d pass=%s\n",
            completed, expected, string(complete))
    complete || error("smooth study did not complete every requested row")
    return nothing
end

function _lf_reflection_solution(; refine=nothing, subcycle=false, nmax=20000)
    N = 192
    amp = 1e-3
    c0 = sqrt(1.4)
    bcs = ntuple(_ -> (PeriodicBC(), PeriodicBC()), 3)
    pulse(x) = amp * exp(-40 * (x - pi / 2)^2)
    ic(x, y, z) = Prim(rho=(1 + pulse(x))^(1 / 1.4),
                       p=1 + pulse(x), u=(pulse(x) / c0, 0, 0))
    solver = Solver(n_global=(N, 1, 1), L_domain=(2pi, 1.0, 1.0),
                    bcs=bcs, art=ArtificialProperties(enabled=false), filter_interval=1,
                    filter_cfl=0.35, refine=refine, subcycle=subcycle)
    state = allocate_state(solver)
    initialize!(solver, state, ic)
    run!(solver, state; tfinal=pi / c0, nmax=nmax)
    solver.t >= pi / c0 - 16eps(pi / c0) ||
        error("reflection evolution did not reach its target time")
    states = state isa Vector ? state : [state]
    ps = CompactLES.PatchSolver(solver, solver.patches[1])
    refresh_primitives!(ps, states[1])
    pressure = Float64[]
    leftgoing = Float64[]
    for i in 1:ps.decomp.n_local[1]
        xcoord(ps, 1, i) < 2.1 || continue
        I = padded_index(ps, i, 1, 1)
        dp = ps.p[I] - 1
        push!(pressure, dp)
        push!(leftgoing, (dp - c0 * ps.u[I]) / 2)
    end
    return (; solver, pressure, leftgoing)
end

function _lf_reflection_study(nmax)
    amp = 1e-3
    reference = _lf_reflection_solution(nmax=nmax)
    r1 = BlockRegion((80, 0, 0), (33, 1, 1))
    r2 = BlockRegion((264, 0, 0), (49, 1, 1))
    passed_all = true
    completed = 0
    for depth in (2, 3), subcycle in (false, true)
        reflected = _lf_reflection_solution(
            refine=depth == 2 ? r1 : [r1, r2], subcycle=subcycle, nmax=nmax)
        length(reflected.pressure) == length(reference.pressure) ||
            error("root comparison windows differ")
        local_wake = isempty(reference.pressure) ? 0.0 :
            maximum(abs.(reflected.pressure .- reference.pressure)) / amp
        local_left = isempty(reference.leftgoing) ? 0.0 :
            maximum(abs.(reflected.leftgoing .- reference.leftgoing)) / amp
        values = MPI.Allreduce([local_wake, local_left], max,
                               MPI.COMM_WORLD)
        passed = values[1] <= 0.01 && values[2] <= 0.01
        passed_all &= passed
        completed += 1
        _lf_say("LF_REFLECTION depth=%d subcycle=%s wake=%.9e " *
                "leftgoing=%.9e gate=%.9e pass=%s complete=true\n",
                depth, string(subcycle), values[1], values[2], 0.01,
                string(passed))
    end
    complete = completed == 4
    _lf_say("LF_REFLECTION_COMPLETE rows=%d expected=4 pass=%s gate_pass=%s\n",
            completed, string(complete), string(passed_all))
    complete || error("reflection study did not complete every row")
    passed_all || error("one or more reflection rows exceeded the 1% gate")
    return nothing
end

function _lf_main()
    parts, ns, nmax = _lf_options(ARGS)
    _lf_selected(parts, :smooth) && _lf_smooth_study(ns, nmax)
    _lf_selected(parts, :reflection) && _lf_reflection_study(nmax)
    return nothing
end

mpi_main(_lf_main)
