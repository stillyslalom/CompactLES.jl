# Negative internal energy ahead of a strong shock: where the points with
# ρe = E − ½|m|²/ρ ≤ 0 sit relative to the front, whether the step or the state
# filter puts them there, how deep they go against the jump, and what the
# repairs measured against them cost the validation metrics.
#
#   julia --project=. -t 1 bench/shockfoot.jl                          # every part
#   julia --project=. -t 1 bench/shockfoot.jl part=defaults cases=sedov
#   julia --project=. -t 1 bench/shockfoot.jl part=ablations,coefficients
#   julia --project=. -t 1 bench/shockfoot.jl part=ambient ambients=0.01,0.1,1
#
# The cases are Woodward–Colella, planar Noh (ν = 1) and Sedov from
# test/cases.jl, at the battery's resolutions and CFL numbers. Each run applies
# the state filter from an `EveryStep` callback instead of inside `run!`, so the
# state is read before and after every filter pass; `part=identity` checks that
# this reproduces the production case function's trajectory bit for bit.
# Woodward–Colella is rebuilt here with the ambient pressure as a parameter,
# and the identity part covers that copy too.
#
# Parts (`part=` takes a comma list, `cases=` restricts the cases):
#
#   identity      each case run as above against its function in test/cases.jl:
#                 the largest density difference at the end and both closing
#                 reports
#   defaults      the production settings to the case's end time: the tallies
#                 below, the first bad state with its neighbourhood, where the
#                 bad points of the closing state sit, and the validation metric
#   ablations     the filter off, then the artificial properties off
#   coefficients  C_beta = 2 and 4, C_kappa = 0.1, and
#                 beta_sensor = :ungated_dilatation, beside the default
#   ambient       Woodward–Colella at each ambient pressure in `ambients`
#                 (the battery's is 0.01); the front finder below needs the
#                 ambient pressure under the weaker reservoir's 100
#   repairs       a limit on the filter's correction (the internal energy of a
#                 point may fall to 1/2, then to 1%, of its value before the
#                 pass), then the positivity failsafe at floor_ratio = 0.01
#                 under each floor_scope, to the case's end time with the
#                 validation metric and the mass and energy each repair adds
#
# Woodward–Colella stops at the collision of the two blasts in the ablations,
# coefficients and ambient parts, when the undisturbed gas between the fronts
# is gone; the other parts run every case to its end time. `nmax` caps the
# shorter parts, since several of their runs lose positivity.
#
# The tallies, summed over the steps of a run:
#
#   bad pre / post    point-steps with ρe ≤ 0 before / after the filter pass
#   max               the most bad points after one pass
#   by step           points bad before a pass that were good after the last one
#   by filter, cured  points a pass turns bad, and turns good
#   min pre / post    the lowest ρe before / after a pass
#   ahead             the range of distances of the bad points from the nearest
#                     front, in cells, positive into the undisturbed gas;
#                     `other` counts those behind a front or in a step with none
#   depth             the lowest ρe of a bad point ahead of a front over ρe three
#                     cells behind that front, and the ambient ρe over the same
#   E<0               the fraction of bad points whose total energy is negative
#   rel               the median |ρe| / max(K, |E|) at a bad point, the relative
#                     cancellation a rounding error would have to reach
#
# The front is the outermost point above ρ = 2.5 for Noh (halfway up the jump
# to 4) and above ρ = 2 for Sedov (the level test/validation.jl reads R_s at),
# and for Woodward–Colella the last point from each wall whose pressure exceeds
# the geometric mean of the ambient pressure and 100.
#
# Scratch tooling, like everything else in bench/: it prints tables and asserts
# nothing. The conclusions drawn from a run of it are written up in
# reference/CALIBRATION_APPENDIX.md (negative internal energy ahead of a shock).

using MPI
MPI.Initialized() || MPI.Init(threadlevel=:funneled)
using CompactLES
using CompactLES: filter_state!, padded_index
using Printf

const CL = CompactLES
include(joinpath(@__DIR__, "..", "test", "references.jl"))
include(joinpath(@__DIR__, "..", "test", "cases.jl"))

const CASES = (:woodward, :noh, :sedov)
const PARTS = ("identity", "defaults", "ablations", "coefficients", "ambient", "repairs")
# A full run takes 3000 to 4500 steps; the cap ends a stalled one.
const FULL_NMAX = 100_000
const WC_AMBIENT = 0.01

# --- the cases ----------------------------------------------------------------

end_time(case) = case === :woodward ? WC_T : case === :noh ? NOH_T : SEDOV_T

# Internal energy of the undisturbed gas ahead of the fronts.
function ambient_energy(case, ambient)
    case === :woodward && return ambient / 0.4
    case === :noh && return NOH_P0 / (NOH_G - 1)
    return 1e-5 / 0.4
end

# `woodward` in test/cases.jl with the ambient pressure as a parameter.
function woodward_problem(ambient)
    h = 1.0 / (WC_N - 1)
    δ = 2h
    return Problem(eos=IdealSpecies("gas"; gamma=1.4, R=1.0),
                   transport=ConstantTransport(mu0=0.0),
                   domain=((0.0, 1.0), (0.0, h), (0.0, h)),
                   bcs=((SlipWallBC(), SlipWallBC()), per3[2], per3[3]),
                   ic=(x, y, z) -> Prim(rho=1.0, u=(0.0, 0.0, 0.0),
                        p = 1000 * (1 - tanh_blend(x, 0.1, δ)) +
                            ambient * (tanh_blend(x, 0.1, δ) - tanh_blend(x, 0.9, δ)) +
                            100 * tanh_blend(x, 0.9, δ)))
end

# The case's numerics, with the filter inside `run!` (`interval = 1`) or left to
# the callback (`interval = 0`). Every run is permissive, so a variant ending on
# an inadmissible state still reports its metric.
function build(case; art, interval, control, ambient)
    filter = StateFilter(compact_filter(); cfl=0.35, interval=interval)
    if case === :woodward
        return setup(woodward_problem(ambient),
                     Numerics(n_global=(WC_N, 1, 1), art=art, cfl=0.3, filter=filter,
                              control=control))
    elseif case === :noh
        return setup(noh_problem(1),
                     Numerics(n_global=(Dict(NOH_N)[1], 1, 1), art=art, cfl=NOH_CFL,
                              filter=filter, control=control))
    end
    return setup(sedov_problem(),
                 Numerics(n_global=(SEDOV_N, 1, 1), art=art, cfl=0.3, filter=filter,
                          control=control))
end

# --- the line state and the fronts --------------------------------------------

function line_state(solver, Q)
    eq = solver.equations
    nx = solver.decomp.n_local[1]
    im, ie, ns = eq.i_mom, eq.i_energy, eq.n_species
    ρ = zeros(nx); m = zeros(nx); E = zeros(nx); K = zeros(nx); ρe = zeros(nx)
    for i in 1:nx
        I = padded_index(solver, i, 1, 1)
        r = 0.0
        for s in 1:ns
            r += Q[I, s]
        end
        ρ[i] = r
        m[i] = Q[I, im[1]]
        E[i] = Q[I, ie]
        K[i] = 0.5 * (Q[I, im[1]]^2 + Q[I, im[2]]^2 + Q[I, im[3]]^2) / r
        ρe[i] = E[i] - K[i]
    end
    return (; ρ, m, E, K, ρe)
end

# Each front as (index, direction), the direction pointing into the undisturbed
# gas. Woodward–Colella has one front from each wall until the blasts collide,
# and none after.
function fronts(case, s, ambient)
    if case !== :woodward
        f = findlast(>(case === :noh ? 2.5 : 2.0), s.ρ)
        return f === nothing ? Tuple{Int,Int}[] : [(f, +1)]
    end
    threshold = sqrt(100 * ambient)
    low = findall(<(threshold), 0.4 .* s.ρe)
    isempty(low) && return Tuple{Int,Int}[]
    left, right = first(low), last(low)
    right - left < 4 && return Tuple{Int,Int}[]
    return [(left - 1, +1), (right + 1, -1)]
end

function nearest_front(fs, i)
    best = nothing
    for (f, dir) in fs
        offset = dir * (i - f)
        (best === nothing || abs(offset) < abs(best.offset)) &&
            (best = (; offset, front=f, dir))
    end
    return best
end

# --- the tallies --------------------------------------------------------------

mutable struct Tally
    steps::Int
    bad_pre::Int
    bad_post::Int
    max_bad::Int
    by_step::Int
    by_filter::Int
    cured::Int
    min_pre::Float64
    min_post::Float64
    offsets::Dict{Int,Int}
    other::Int
    depth::Float64
    ambient_depth::Float64
    negative_total::Int
    rel::Vector{Float64}
    limited::Int
    limit_mass::Float64
    limit_energy::Float64
end
Tally() = Tally(0, 0, 0, 0, 0, 0, 0, Inf, Inf, Dict{Int,Int}(), 0, Inf, NaN, 0,
                Float64[], 0, 0.0, 0.0)

function record!(T, prev, pre, post, fs, ambient_rhoe)
    T.steps += 1
    n = length(post.ρe)
    count_post = 0
    for i in 1:n
        bad_prev, bad_pre, bad_post = prev.ρe[i] <= 0, pre.ρe[i] <= 0, post.ρe[i] <= 0
        T.bad_pre += bad_pre
        T.by_step += bad_pre & !bad_prev
        T.by_filter += bad_post & !bad_pre
        T.cured += bad_pre & !bad_post
        bad_post || continue
        count_post += 1
        T.negative_total += post.E[i] < 0
        push!(T.rel, abs(post.ρe[i]) / max(post.K[i], abs(post.E[i])))
        near = nearest_front(fs, i)
        if near === nothing || near.offset <= 0
            T.other += 1
            continue
        end
        T.offsets[near.offset] = get(T.offsets, near.offset, 0) + 1
        jump = post.ρe[clamp(near.front - 3near.dir, 1, n)]
        if jump > 0 && post.ρe[i] / jump < T.depth
            T.depth = post.ρe[i] / jump
            T.ambient_depth = ambient_rhoe / jump
        end
    end
    T.bad_post += count_post
    T.max_bad = max(T.max_bad, count_post)
    T.min_pre = min(T.min_pre, minimum(pre.ρe))
    T.min_post = min(T.min_post, minimum(post.ρe))
    return nothing
end

# --- the filter pass and its limit --------------------------------------------

function filter_pass!(solver, Q)
    solver.filter_interval = 1
    filter_state!(solver, Q)
    solver.filter_interval = 0
    return nothing
end

internal_energy(q, ns, im, ie) =
    q[ie] - 0.5 * (q[im[1]]^2 + q[im[2]]^2 + q[im[3]]^2) / sum(view(q, 1:ns))

# The filter pass, then a limit on its correction at each point: the internal
# energy may fall to `fraction` of its value before the pass, and no lower than
# that value where it was already nonpositive. ρe is concave in the conserved
# state where ρ > 0, so the admissible fractions of the correction form an
# interval containing 0, and bisection finds its end.
function limited_pass!(T, solver, Q, fraction)
    eq = solver.equations
    nx = solver.decomp.n_local[1]
    im, ie, ns, nc = eq.i_mom, eq.i_energy, eq.n_species, eq.n_cons
    data = parent(Q)
    before = copy(data)
    filter_pass!(solver, Q)
    filtered = copy(data)
    q_old = zeros(nc); q_new = zeros(nc); q_try = zeros(nc)
    for i in 1:nx
        I = padded_index(solver, i, 1, 1)
        for c in 1:nc
            q_old[c] = before[I, c]
            q_new[c] = data[I, c]
        end
        e_old = internal_energy(q_old, ns, im, ie)
        target = e_old > 0 ? fraction * e_old : e_old
        internal_energy(q_new, ns, im, ie) >= target && continue
        lo, hi = 0.0, 1.0
        for _ in 1:60
            mid = 0.5 * (lo + hi)
            @. q_try = q_old + mid * (q_new - q_old)
            internal_energy(q_try, ns, im, ie) >= target ? (lo = mid) : (hi = mid)
        end
        for c in 1:nc
            data[I, c] = q_old[c] + lo * (q_new[c] - q_old[c])
        end
        T.limited += 1
    end
    T.limit_mass += sum(volume_integral(solver, data[:, :, :, s] .- filtered[:, :, :, s])
                        for s in 1:ns)
    T.limit_energy += volume_integral(solver, data[:, :, :, ie] .- filtered[:, :, :, ie])
    return nothing
end

# --- one run ------------------------------------------------------------------

function show_window(prev, pre, post, i0, window)
    @printf("    %5s %11s %11s %11s %11s %11s %11s %11s\n", "i", "rho", "u", "E", "K",
            "rhoe prev", "rhoe pre", "rhoe post")
    for i in max(1, i0 - window):min(length(pre.ρ), i0 + window)
        @printf("    %5d %11.4e %11.4e %11.4e %11.4e %11.4e %11.4e %11.4e\n", i,
                pre.ρ[i], pre.m[i] / pre.ρ[i], pre.E[i], pre.K[i], prev.ρe[i],
                pre.ρe[i], post.ρe[i])
    end
end

"""
    variant(label; art, filter, fraction, floor_scope, ambient)

One configuration. `filter` is `:pass` (the callback's pass), `:limit` (the
pass with its correction limited to `fraction`), `:off`, or `:inside` (the
filter inside `run!`, so the failsafe at `floor_scope` acts after it).
"""
variant(label; art=ArtificialProperties(enabled=true), filter=:pass, fraction=0.5,
        floor_scope=nothing, ambient=WC_AMBIENT) =
    (; label, art, filter, fraction, floor_scope, ambient)

function instrumented_run(case, v; full, nmax, window)
    control = v.floor_scope === nothing ? StepControl(validity=:permissive) :
              StepControl(validity=:permissive, floor_ratio=0.01, floor_scope=v.floor_scope)
    solver, Q = build(case; art=v.art, interval=v.filter === :inside ? 1 : 0,
                      control=control, ambient=v.ambient)
    data = parent(Q)
    ie, ns = solver.equations.i_energy, solver.equations.n_species
    energy0 = volume_integral(solver, data[:, :, :, ie])
    mass0 = sum(volume_integral(solver, data[:, :, :, s]) for s in 1:ns)
    ambient_rhoe = ambient_energy(case, v.ambient)
    stop_at_collision = case === :woodward && !full
    T = Tally()
    prev = Ref(line_state(solver, Q))
    shown = Ref(false)
    effect = function (solver, Q)
        pre = line_state(solver, Q)
        v.filter === :pass && filter_pass!(solver, Q)
        v.filter === :limit && limited_pass!(T, solver, Q, v.fraction)
        post = v.filter in (:pass, :limit) ? line_state(solver, Q) : pre
        fs = fronts(case, post, v.ambient)
        record!(T, prev[], pre, post, fs, ambient_rhoe)
        bad_pre, bad_post = findall(<=(0), pre.ρe), findall(<=(0), post.ρe)
        if !shown[] && !(isempty(bad_pre) && isempty(bad_post))
            shown[] = true
            i0 = isempty(bad_pre) ? bad_post[argmin(post.ρe[bad_post])] :
                                    bad_pre[argmin(pre.ρe[bad_pre])]
            @printf("  first bad state at step %d (t = %.4e): %d before the pass, %s\n",
                    solver.step, solver.t, length(bad_pre),
                    "$(length(bad_post)) after; fronts $(string(fs))")
            show_window(prev[], pre, post, i0, window)
        end
        prev[] = post
        stop_at_collision && isempty(fs) && solver.step > 10 && return true
        return nothing
    end
    status = "stopped"
    wall = @elapsed try
        run!(solver, Q; tfinal=end_time(case), nmax=full ? FULL_NMAX : nmax,
             callback=Callback(EveryStep(1), effect))
    catch err
        status = err isa CL.SolverFailure ? "failed: $(err.reason) @ $(err.step)" :
                 first("error: " * sprint(showerror, err), 120)
    end
    completed(solver, end_time(case)) && (status = "completed")
    stop_at_collision && status == "stopped" && (status = "collision")
    added = if v.filter === :limit
        (mass=T.limit_mass, energy=T.limit_energy)
    elseif v.floor_scope !== nothing
        (mass=solver.floor_tally.mass, energy=solver.floor_tally.energy)
    else
        nothing
    end
    added = added === nothing ? nothing :
            (mass=added.mass / mass0, energy=added.energy / energy0)
    return (; case, label=v.label, status, steps=T.steps, t=solver.t, wall, T, added,
            solver, Q, ambient=v.ambient)
end

# --- reports ------------------------------------------------------------------

const WC_REFERENCE = Ref{Any}(nothing)

function wc_reference()
    WC_REFERENCE[] === nothing || return WC_REFERENCE[]
    xs, ρs = Float64[], Float64[]
    for line in eachline(joinpath(@__DIR__, "..", "test", "refs", "woodward_colella.csv"))
        (isempty(line) || startswith(line, '#')) && continue
        tokens = split(line, ',')
        push!(xs, parse(Float64, tokens[1]))
        push!(ρs, parse(Float64, tokens[2]))
    end
    return WC_REFERENCE[] = (xs, ρs)
end

function metric(case, solver, Q)
    xs, ρ, _, _ = case_line_profile(solver, Q)
    if case === :woodward
        xr, ρr = wc_reference()
        imax = argmax(ρ)
        return @sprintf("L1 rho %.4e, peak %.4f at x = %.4f",
                        l1(ρ, [interp1(xr, ρr, x) for x in xs]), ρ[imax], xs[imax])
    elseif case === :noh
        plateau, deficit, shock, _ = noh_metrics(xs, ρ, 1)
        return @sprintf("plateau %.4f, wall deficit %.1f%%, shock %.4f", plateau,
                        100 * deficit, shock)
    end
    exact = sedov_shock_radius(SEDOV_E, SEDOV_T, 3, 1.4)
    radius = front_position(xs, ρ, 2.0)
    return @sprintf("R_s %.5f (%+.2f%%), peak rho %.4f", radius,
                    100 * (radius / exact - 1), maximum(ρ))
end

function offsets_text(T)
    isempty(T.offsets) && return "-"
    ks = sort(collect(keys(T.offsets)))
    return "$(first(ks))-$(last(ks))"
end

function median_of(v)
    isempty(v) && return NaN
    s = sort(v)
    return s[cld(length(s), 2)]
end

function print_detail(r)
    T = r.T
    @printf("  %s %s: %s, %d steps to t = %.4e in %.1f s\n", r.case, r.label,
            r.status, r.steps, r.t, r.wall)
    if !isempty(T.offsets)
        print("    bad points by distance ahead of the front: ")
        for k in sort(collect(keys(T.offsets)))
            print(k, ":", T.offsets[k], " ")
        end
        println("other:", T.other)
    end
    T.limited > 0 && @printf("    limited point-steps %d\n", T.limited)
    r.added === nothing ||
        @printf("    added by the repair: mass %+.3e, energy %+.3e of the initial totals\n",
                r.added.mass, r.added.energy)
    r.solver.floor_tally.steps > 0 &&
        println("    failsafe tally: ", r.solver.floor_tally)
    return nothing
end

const TABLE_HEAD = Printf.Format("\n  %-9s %-22s %-22s %6s %8s %8s %4s %7s %7s %6s " *
                                 "%11s %11s %6s %6s %10s %10s %5s %9s\n")
const TABLE_ROW = Printf.Format("  %-9s %-22s %-22s %6d %8d %8d %4d %7d %7d %6d " *
                                "%11.4e %11.4e %6s %6d %10.3e %10.3e %5.2f %9.2e\n")

function print_table(rows; metrics)
    Printf.format(stdout, TABLE_HEAD, "case", "variant", "end", "steps", "bad pre",
                  "bad post", "max", "by step", "by filt", "cured", "min pre", "min post",
                  "ahead", "other", "depth", "ambient", "E<0", "rel")
    for r in rows
        T = r.T
        nbad = max(T.bad_post, 1)
        Printf.format(stdout, TABLE_ROW, r.case, r.label, first(r.status, 22), r.steps,
                      T.bad_pre, T.bad_post, T.max_bad, T.by_step, T.by_filter, T.cured,
                      T.min_pre, T.min_post, offsets_text(T), T.other, T.depth,
                      T.ambient_depth, T.negative_total / nbad, median_of(T.rel))
    end
    metrics || return nothing
    println()
    for r in rows
        text = completed(r.solver, end_time(r.case)) ? metric(r.case, r.solver, r.Q) :
               "did not reach the end time"
        added = r.added === nothing ? "" :
                @sprintf(";  added mass %+.2e, energy %+.2e", r.added.mass, r.added.energy)
        @printf("  %-9s %-22s %s%s\n", r.case, r.label, text, added)
    end
    return nothing
end

# Where the bad points of the closing state sit.
function closing_cells(r)
    s = line_state(r.solver, r.Q)
    fs = fronts(r.case, s, r.ambient)
    report = state_report(r.solver, r.Q)
    @printf("  %s closing state: %d inadmissible, e_min %.4e, fronts %s\n", r.case,
            report.inadmissible, report.e_min, string(fs))
    for i in findall(<=(0), s.ρe)
        near = nearest_front(fs, i)
        @printf("    i %4d  x %.4f  ahead %s  rho %.4e  rhoe %.4e  E %.4e\n", i,
                xcoord(r.solver, 1, i), near === nothing ? "-" : string(near.offset),
                s.ρ[i], s.ρe[i], s.E[i])
    end
end

# --- the parts ----------------------------------------------------------------

function run_part(name, label_rows, cases; full, nmax, window, metrics)
    println("\n=== ", name, " ===")
    rows = []
    for case in cases, v in label_rows(case)
        println("== ", case, " ", v.label)
        r = instrumented_run(case, v; full, nmax, window)
        print_detail(r)
        push!(rows, r)
        flush(stdout)
    end
    print_table(rows; metrics)
    return rows
end

function identity_part(cases; window)
    println("\n=== identity: the callback's filter against run!'s ===")
    for case in cases
        r = instrumented_run(case, variant("default"); full=true, nmax=0, window=window)
        reference = case === :woodward ? woodward() :
                    case === :noh ? noh_case(1) : sedov()
        _, ρ, _, _ = case_line_profile(r.solver, r.Q)
        mine = state_report(r.solver, r.Q)
        @printf("  %-9s max |rho - rho_case| %.3e;  closing reports: %s / %s\n", case,
                maximum(abs.(reference[2] .- ρ)), string(mine),
                length(reference) >= 6 ? string(reference[6]) : "(not returned)")
        flush(stdout)
    end
end

function main(args)
    opt = CL.script_args(args, (part="all", cases="woodward,noh,sedov", nmax=4000,
                                window=5, ambients="0.01,0.1,1"))
    parts = opt.part == "all" ? collect(PARTS) : split(opt.part, ',')
    cases = Symbol.(split(opt.cases, ','))
    for p in parts
        p in PARTS ||
            throw(ArgumentError("unknown part '$p', want one of $(join(PARTS, ", "))"))
    end
    for c in cases
        c in CASES ||
            throw(ArgumentError("unknown case '$c', want one of $(join(CASES, ", "))"))
    end
    ambients = parse.(Float64, split(opt.ambients, ','))
    short = (; full=false, nmax=opt.nmax, window=opt.window)
    long = (; full=true, nmax=opt.nmax, window=opt.window)
    for p in parts
        if p == "identity"
            identity_part(cases; window=opt.window)
        elseif p == "defaults"
            rows = run_part(p, _ -> [variant("default")], cases; long..., metrics=true)
            foreach(closing_cells, rows)
        elseif p == "ablations"
            run_part(p, _ -> [variant("filter off"; filter=:off),
                              variant("art off"; art=ArtificialProperties(enabled=false))],
                     cases; short..., metrics=true)
        elseif p == "coefficients"
            run_part(p, _ -> [variant("default"),
                              variant("C_beta = 2"; art=ArtificialProperties(C_beta=2.0)),
                              variant("C_beta = 4"; art=ArtificialProperties(C_beta=4.0)),
                              variant("C_kappa = 0.1"; art=ArtificialProperties(C_kappa=0.1)),
                              variant("ungated dilatation";
                                      art=ArtificialProperties(
                                          beta_sensor=:ungated_dilatation))],
                     cases; short..., metrics=true)
        elseif p == "ambient"
            :woodward in cases || continue
            run_part(p, _ -> [variant("ambient p $a"; ambient=a) for a in ambients],
                     [:woodward]; short..., metrics=false)
        else
            run_part(p, _ -> [variant("default"),
                              variant("limit 1/2"; filter=:limit, fraction=0.5),
                              variant("limit 1%"; filter=:limit, fraction=0.01),
                              variant("floor representable"; filter=:inside,
                                      floor_scope=:representable),
                              variant("floor internal_energy"; filter=:inside,
                                      floor_scope=:internal_energy)],
                     cases; long..., metrics=true)
        end
    end
end

main(ARGS)
