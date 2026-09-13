# The wall-filter trial battery: the compact filter's cascade rows against
# its one-sided rows at closed ends, on the shock battery, a warm-started
# wall, a reflected acoustic pulse, the conservation and floor budgets, a
# species layer at a wall, and in Float32.
#
#   julia --project=. -t 1 bench/wallfilter.jl                     # everything
#   julia --project=. -t 1 bench/wallfilter.jl parts=shock,pulse
#   julia --project=. -t 1 bench/wallfilter.jl parts=closures alphaf=0.49
#
# Parts:
#
#   shock     the test/cases.jl battery under both filter closures with the
#             default derivative closure, at the relaxed weight
#             (filter_cfl = 0.35) and unrelaxed (0), so the two filter row
#             sets are compared within one time-scaling formulation at a
#             time. Every case has closed ends: the Dirichlet inflows of the
#             tubes and the Noh folds take the closure rows as a wall does.
#   closures  the closure-compatibility table: every derivative closure
#             under both filters on the wall-bounded shock cases, cold and
#             warm-started planar Noh and Woodward–Colella.
#   pulse     a simple-wave acoustic pulse reflected at a slip wall against
#             its periodic mirror at the same spacing, which is exact by
#             symmetry and carries the interior scheme and the filter but no
#             closure rows, so the difference is the closure defect alone;
#             artificial properties off and on, a smooth amplitude and one
#             that steepens into a shock after the reflection.
#   budget    the mass and energy the run creates or destroys between two
#             walls, split into the filter's own tally and the remainder,
#             the state the run ends on, the positivity floor's firings
#             under a nonzero floor, and a species layer against the wall
#             crossed twice by the steepening pulse.
#   float32   planar Noh, Woodward–Colella and the smooth pulse in Float32.
#   steepening  the steepening pulse separated into an amplitude that does
#             not shock and the shocking one read before its front forms;
#             not part of `all`.
#
# Background settings (`key=value`): alphaf (0.45 = the solver default) and
# filter_cfl for the parts that do not sweep it. Scratch tooling, like
# everything else in bench/: it prints tables and asserts nothing. The
# conclusions are written up in reference/CALIBRATION_APPENDIX.md and the
# decision in reference/CALIBRATION.md.

using MPI
MPI.Init(threadlevel=:funneled)
using CompactLES
using Printf

const CL = CompactLES
MPI.Comm_size(MPI.COMM_WORLD) == 1 || error("run this study on one rank")
include(joinpath(@__DIR__, "..", "test", "references.jl"))
include(joinpath(@__DIR__, "..", "test", "cases.jl"))

const OPTS = CL.script_args(ARGS, (parts="all", alphaf=0.45, filter_cfl=0.35))
const PARTS = OPTS.parts == "all" ?
    ["shock", "closures", "pulse", "budget", "float32"] : split(OPTS.parts, ',')
const CAP = 30_000
const REFDIR = joinpath(@__DIR__, "..", "test", "refs")

const FILTERS = (("cascade", :cascade), ("onesided", :onesided))
filt_of(cl, ::Type{T}=Float64) where {T} = compact_filter(T(OPTS.alphaf), T; closures=cl)
const DERIVS = (("C6 cascade3", T -> lele_d1_6(T)),
                ("C6 cascade4", T -> lele_d1_6(T; closures=:cascade4)),
                ("C6 BL", T -> lele_d1_6(T; closures=:brady_livescu)),
                ("C8 BL", T -> lele_d1_8(T; closures=:brady_livescu)))

# `@printf` takes a literal format string only; the formats below are built
# per table, so they go through the runtime form.
sprintf(fmt::String, args...) = Printf.format(Printf.Format(fmt), args...)
printf(fmt::String, args...) = print(sprintf(fmt, args...))
hr(n=100) = println(repeat("-", n))

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
pad(s, n) = rpad(s, n)

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

# --- the battery rows ---------------------------------------------------------

function row_noh(ν; kw...)
    attempt() do
        xs, ρ, _, _, ok, report = noh_case(ν; nmax=CAP, kw...)
        ok || return "step cap"
        plat, deficit, Rs, epre = noh_metrics(xs, ρ, ν)
        sprintf("plateau %.4f  deficit %+3.0f%%  shock %.4f  L1 pre %.2e  " *
                 "inadmissible %2d  e_min %+.4f", plat / 4.0^ν, 100deficit, Rs,
                 epre, report.inadmissible, report.e_min)
    end
end

function row_lax(; kw...)
    attempt() do
        xs, ρ, u, p, ok = lax(; nmax=CAP, kw...)
        ok || return "step cap"
        ex = [riemann_profile(x, LAX_T, 0.5, LAX_L, LAX_R, 1.4) for x in xs]
        sprintf("L1 rho %.3e  u %.3e  p %.3e  contact %.4f",
                 l1(ρ, [e[1] for e in ex]), l1(u, [e[2] for e in ex]),
                 l1(p, [e[3] for e in ex]), contact_width(xs, ρ, 0.5, 1.3))
    end
end

function row_shu(; kw...)
    attempt() do
        xs, ρ, _, _, ok = shu_osher(; nmax=CAP, kw...)
        ok || return "step cap"
        xr, ρr, _, _ = read_ref("shu_osher.csv")
        ref = [interp1(xr, ρr, x) for x in xs]
        band = so_band(xs)
        sprintf("L1 rho %.3e  train L1 %.3e  train peak %.4f  train amp %.4f",
                 l1(ρ, ref), l1(ρ[band], ref[band]), maximum(ρ[band]),
                 maximum(ρ[band]) - minimum(ρ[band]))
    end
end

function row_wc(; N=WC_N, kw...)
    attempt() do
        xs, ρ, _, _, ok = woodward(; N=N, nmax=CAP, kw...)
        ok || return "step cap"
        imax = argmax(ρ)
        s = sprintf("peak rho %.4f at %.4f  rho_min %.4f", ρ[imax], xs[imax], minimum(ρ))
        if N == WC_N
            xr, ρr, _, _ = read_ref("woodward_colella.csv")
            s = sprintf("L1 rho %.3e  ", l1(ρ, [interp1(xr, ρr, x) for x in xs])) * s
        end
        s
    end
end

function row_sedov(; kw...)
    attempt() do
        rs, ρ, _, _, ok, report = sedov(; nmax=CAP, kw...)
        ok || return "step cap"
        Rex = sedov_shock_radius(SEDOV_E, SEDOV_T, 3, 1.4)
        Rnum = front_position(rs, ρ, 2.0)
        sprintf("R_s %.4f (%+.2f%%)  peak rho %.3f  inadmissible %d  e_min %+.4f",
                 Rnum, 100 * (Rnum / Rex - 1), maximum(ρ), report.inadmissible,
                 report.e_min)
    end
end

function row_si(; kw...)
    attempt() do
        r = shock_interface(; nmax=CAP, kw...)
        r.completed || return "step cap"
        sprintf("worst Y %+.4f / %.4f  width %d cells  steps %d",
                 r.worst_min_Y, r.worst_max_Y, r.width_cells, r.steps)
    end
end

function row_warm(; kw...)
    attempt() do
        xs, ρ, _, _, ok, report = noh_case(1; t0=0.3, nmax=CAP, kw...)
        ok || return "step cap"
        plat, deficit, Rs, _ = noh_metrics(xs, ρ, 1)
        sprintf("rho[1:4] %.3f %.3f %.3f %.3f  plateau %.4f  deficit %+3.0f%%  shock %.4f",
                 ρ[1], ρ[2], ρ[3], ρ[4], plat / 4, 100deficit, Rs)
    end
end

# --- part: shock --------------------------------------------------------------

function shock_part()
    println("\n=== the battery under both filter closures, C6 cascade3 ===")
    rows = (("Noh nu=1 N=400", kw -> row_noh(1; kw...)),
            ("Noh nu=1 N=800", kw -> row_noh(1; N=800, kw...)),
            ("Noh nu=2", kw -> row_noh(2; kw...)),
            ("Noh nu=3", kw -> row_noh(3; kw...)),
            ("Lax", kw -> row_lax(; kw...)),
            ("Shu-Osher", kw -> row_shu(; kw...)),
            ("Woodward N=800", kw -> row_wc(; kw...)),
            ("Sedov", kw -> row_sedov(; kw...)),
            ("shock/SF6", kw -> row_si(; kw...)))
    for fc in (0.35, 0.0)
        println("\n--- filter_cfl = $fc ($(fc > 0 ? "relaxed" : "unrelaxed")) ---")
        for (name, f) in rows, (flab, cl) in FILTERS
            r = f((filt=filt_of(cl), filter_cfl=fc))
            println("  ", pad(name, 16), pad(flab, 10), r)
            flush(stdout)
        end
    end
end

# --- part: closures -----------------------------------------------------------

function closures_part()
    println("\n=== closure compatibility: derivative rows x filter rows (filter_cfl = $(OPTS.filter_cfl)) ===")
    cases = (("Woodward N=800", (d, f) -> row_wc(; deriv=d, filt=f, filter_cfl=OPTS.filter_cfl)),
             ("Noh cold N=400", (d, f) -> row_noh(1; deriv=d, filt=f, filter_cfl=OPTS.filter_cfl)),
             ("Noh warm t0=0.3", (d, f) -> row_warm(; deriv=d, filt=f, filter_cfl=OPTS.filter_cfl)))
    for (name, f) in cases
        println("\n--- $name ---")
        for (dlab, mk) in DERIVS, (flab, cl) in FILTERS
            r = f(mk(Float64), filt_of(cl))
            println("  ", pad(dlab, 14), pad(flab, 10), r)
            flush(stdout)
        end
    end
end

# --- the reflected pulse ------------------------------------------------------
#
# A left-moving simple wave of the ideal gas, p = 1 + amp exp(-((x - x0)/σ)²)
# with ρ = p^(1/γ) and u = -2 (c - c0)/(γ - 1), between slip walls on [0, 1].
# The periodic mirror on [0, 2) carries the pulse and its image about x = 1
# moving right, so its solution restricted to [0, 1] is the wall problem's
# at every time, reflections and steepening included, and it never evaluates
# a closure row.

const PULSE_G = 1.4
const PULSE_X0 = 0.5
const PULSE_S = 0.05

function pulse_prim(::Type{T}, x, amp, sgn) where {T}
    γ = PULSE_G
    p = 1 + amp * exp(-((x - PULSE_X0) / PULSE_S)^2)
    ρ = p^(1 / γ)
    c0 = sqrt(γ)
    c = sqrt(γ * p / ρ)
    u = -sgn * 2 * (c - c0) / (γ - 1)
    return Prim(rho=T(ρ), u=(T(u), zero(T), zero(T)), p=T(p))
end

function pulse_solver(::Type{T}, N; amp, art, mirror, cl, deriv=lele_d1_6(T),
                      cfl=0.4, filter_cfl=OPTS.filter_cfl, filter_interval=1,
                      control=StepControl(validity=:permissive)) where {T}
    per = (PeriodicBC(), PeriodicBC())
    h = one(T) / T(N - 1)
    n = mirror ? 2(N - 1) : N
    L = mirror ? T(2) : one(T)
    bcs = mirror ? per : (SlipWallBC(), SlipWallBC())
    solver = Solver(; n_global=(n, 1, 1), L_domain=(L, h, h),
                    bcs=(bcs, per, per),
                    eos=IdealSpecies(T, "gas"; R=one(T), gamma=T(PULSE_G)),
                    transport=Transport{T}(mu0=zero(T)),
                    art=ArtParams{T}(enabled=art), deriv=deriv,
                    filt=filt_of(cl, T), cfl=T(cfl), filter_interval=filter_interval,
                    filter_cfl=filter_cfl, control=control)
    Q = allocate_state(solver)
    initialize!(solver, Q, (x, y, z) ->
        x <= 1 ? pulse_prim(T, x, amp, 1) : pulse_prim(T, 2 - x, amp, -1))
    return solver, Q
end

function interior_line(solver, Q, comp)
    CL.exchange_state!(Q, solver.decomp)
    CL.primitives!(solver, Q)
    nx = solver.decomp.n_local[1]
    return [Float64(Q[gidx(solver, i, 1, 1), comp]) for i in 1:nx]
end

# Wall window (four nodes at each end), interior maximum and the root-mean-
# square over the line, of the wall run against its mirror, node by node.
function line_errors(a, b; W=4)
    N = length(a)
    e = abs.(a .- b[1:N])
    wall = maximum(e[[1:W; N-W+1:N]])
    interior = maximum(e[W+1:N-W])
    l2 = sqrt(sum(abs2, e) / N)
    return wall, interior, l2
end

function pulse_row(::Type{T}, N; amp, art, cl, tfinal, comp=1, kw...) where {T}
    attempt() do
        solver, Q = pulse_solver(T, N; amp, art, mirror=false, cl, kw...)
        run!(solver, Q; tfinal=T(tfinal), nmax=CAP)
        mirror, Qm = pulse_solver(T, N; amp, art, mirror=true, cl, kw...)
        run!(mirror, Qm; tfinal=T(tfinal), nmax=CAP)
        # The mirror lands on its own last step; compare at the same clock.
        abs(solver.t - mirror.t) < 1e-6 || return sprintf("clocks differ %.3e", solver.t - mirror.t)
        a = interior_line(solver, Q, comp)
        b = interior_line(mirror, Qm, comp)
        wall, interior, l2 = line_errors(a, b)
        rep = state_report(solver, Q)
        (wall=wall, interior=interior, l2=l2, steps=solver.step,
         inadmissible=rep.inadmissible, rho_min=rep.rho_min)
    end
end

function pulse_table(::Type{T}, ns; amp, art, tfinal, comp=1, kw...) where {T}
    for (flab, cl) in FILTERS
        es = Float64[]
        hs = Float64[]
        for N in ns
            r = pulse_row(T, N; amp, art, cl, tfinal, comp, kw...)
            if failed(r)
                println("  ", pad(flab, 10), sprintf("%4d  ", N), r)
                continue
            end
            printf("  %-10s%4d  wall %.3e  interior %.3e  l2 %.3e  steps %5d  " *
                   "inadmissible %d  rho_min %.4f\n", flab, N, r.wall, r.interior,
                   r.l2, r.steps, r.inadmissible, r.rho_min)
            push!(es, r.wall); push!(hs, 1 / (N - 1))
        end
        if length(es) >= 2
            orders = [log(es[i] / es[i+1]) / log(hs[i] / hs[i+1]) for i in 1:length(es)-1]
            printf("  %-10s      wall orders %s\n", "",
                   join((sprintf("%.2f", o) for o in orders), " / "))
        end
        flush(stdout)
    end
end

function pulse_part()
    # The pulse leaves x0 = 0.5 leftward at c0 = 1.18, reaches the wall near
    # t = 0.42 and is back at x0 near t = 0.85. At amp = 0.01 the simple wave
    # steepens over a distance of about seven domain lengths and stays
    # smooth; at amp = 0.1 it shocks about 0.7 into its path, after the
    # reflection.
    println("\n=== reflected acoustic pulse against its periodic mirror (density, t = 0.7) ===")
    println("\n--- amp 0.01, artificial properties off ---")
    pulse_table(Float64, (49, 97, 193, 385); amp=0.01, art=false, tfinal=0.7)
    println("\n--- amp 0.01, artificial properties on ---")
    pulse_table(Float64, (49, 97, 193, 385); amp=0.01, art=true, tfinal=0.7)
    println("\n--- amp 0.1, artificial properties on (shocked after the reflection) ---")
    pulse_table(Float64, (97, 193, 385, 769); amp=0.1, art=true, tfinal=0.7)
    println("\n--- amp 0.1, artificial properties on, unrelaxed (filter_cfl = 0) ---")
    pulse_table(Float64, (97, 193, 385, 769); amp=0.1, art=true, tfinal=0.7,
                filter_cfl=0.0)
end

# Where the steepening pulse's mid-resolution defect is made: an amplitude
# that steepens but does not shock within the run, and the shocking one read
# just after the reflection, before the front has formed.
function steepening_part()
    println("\n=== the steepening pulse, separated ===")
    println("\n--- amp 0.03, artificial properties on, t = 0.7 (steepens, no shock) ---")
    pulse_table(Float64, (97, 193, 385, 769); amp=0.03, art=true, tfinal=0.7)
    println("\n--- amp 0.1, artificial properties on, t = 0.5 (reflected, not yet shocked) ---")
    pulse_table(Float64, (97, 193, 385, 769); amp=0.1, art=true, tfinal=0.5)
    println("\n--- amp 0.1, artificial properties off, t = 0.5 ---")
    pulse_table(Float64, (97, 193, 385, 769); amp=0.1, art=false, tfinal=0.5)
end

# --- part: budget -------------------------------------------------------------

function totals(solver, Q)
    eq = solver.equations
    mass = sum(volume_integral(solver, view(Q, :, :, :, sp)) for sp in 1:eq.n_species)
    mom = volume_integral(solver, view(Q, :, :, :, eq.i_mom[1]))
    E = volume_integral(solver, view(Q, :, :, :, eq.i_energy))
    return [mass, mom, E]
end

# The run filters through this callback with the solver's own pass off, so
# the change of the totals across each pass is the filter's alone; the
# interval is raised to one for the pass so `filter_weight` reads the
# solver's cadence (as bench/filter_conservation.jl does).
function tallying_filter(acc, scale)
    return (solver, Q) -> begin
        before = totals(solver, Q)
        solver.filter_interval = 1
        filter_state!(solver, Q)
        solver.filter_interval = 0
        after = totals(solver, Q)
        acc .+= after .- before
        scale .= max.(scale, abs.(after))
        nothing
    end
end

function budget_run(build, tfinal)
    attempt() do
        solver, Q = build()
        t0 = totals(solver, Q)
        acc = zeros(3); scale = copy(abs.(t0))
        run!(solver, Q; tfinal=tfinal, nmax=CAP, callback=tallying_filter(acc, scale))
        t1 = totals(solver, Q)
        drift = (t1 .- t0) ./ max.(scale, 1e-300)
        filt = acc ./ max.(scale, 1e-300)
        rep = state_report(solver, Q)
        ft = solver.floor_tally
        (drift=drift, filt=filt, report=rep, tally=ft, steps=solver.step,
         solver=solver, Q=Q)
    end
end

function print_budget(label, r)
    if failed(r)
        println("  ", pad(label, 26), r)
        return
    end
    printf("  %-26s mass %+.2e (filter %+.2e)  energy %+.2e (filter %+.2e)  " *
           "mom %+.2e | inadmissible %d  e_min %+.4f  rho_min %.4f | floor steps %d " *
           "cells %d mass %+.1e energy %+.1e | steps %d\n", label, r.drift[1], r.filt[1],
           r.drift[3], r.filt[3], r.drift[2], r.report.inadmissible, r.report.e_min,
           r.report.rho_min, r.tally.steps, r.tally.cells, r.tally.mass, r.tally.energy,
           r.steps)
    flush(stdout)
end

function wc_build(cl; N=WC_N, control=StepControl(validity=:permissive), filter_cfl=OPTS.filter_cfl)
    h = 1.0 / (N - 1)
    δ = 2h
    prob = Problem(eos=IdealSpecies("gas"; gamma=1.4, R=1.0),
                   transport=Transport(mu0=0.0),
                   domain=((0.0, 1.0), (0.0, h), (0.0, h)),
                   bcs=((SlipWallBC(), SlipWallBC()), per3[2], per3[3]),
                   ic=(x, y, z) -> Prim(rho=1.0, u=(0.0, 0.0, 0.0),
                        p = 1000 * (1 - tanh_blend(x, 0.1, δ)) +
                            0.01 * (tanh_blend(x, 0.1, δ) - tanh_blend(x, 0.9, δ)) +
                            100 * tanh_blend(x, 0.9, δ)))
    () -> setup(prob, Numerics(n_global=(N, 1, 1), art=ArtParams(enabled=true),
                               cfl=0.3, filt=filt_of(cl), filter_interval=0,
                               filter_cfl=filter_cfl, control=control))
end

# Two gases of equal γ and gas constant, so the mixture is thermodynamically
# the single gas above and the species channel is the only thing the layer
# adds: `light` fills [0, 0.15] against the wall and `heavy` the rest, and
# the amp = 0.1 pulse crosses the layer inward, reflects, and crosses it
# again.
function layer_build(cl; N, amp=0.1, control=StepControl(validity=:permissive),
                     filter_cfl=OPTS.filter_cfl, ratio=1.0, gamma_heavy=PULSE_G)
    T = Float64
    h = 1.0 / (N - 1)
    eos = IdealMixture([IdealSpecies{T}("light", 1.0, PULSE_G),
                        IdealSpecies{T}("heavy", 1 / ratio, gamma_heavy)])
    per = (PeriodicBC(), PeriodicBC())
    () -> begin
        solver = Solver(; n_global=(N, 1, 1), L_domain=(1.0, h, h),
                        bcs=((SlipWallBC(), SlipWallBC()), per, per), eos=eos,
                        transport=Transport(mu0=0.0), art=ArtParams(enabled=true),
                        filt=filt_of(cl), cfl=0.4, filter_interval=0,
                        filter_cfl=filter_cfl, control=control)
        Q = allocate_state(solver)
        initialize!(solver, Q, (x, y, z) -> begin
            θ = tanh_blend(x, 0.15, 2h)
            pr = pulse_prim(T, x, amp, 1)
            Prim(Y=(1 - θ, θ), rho=pr.rho * ((1 - θ) + θ * ratio), u=pr.u, p=pr.p)
        end)
        solver, Q
    end
end

function budget_part()
    println("\n=== conservation and floor budgets between two walls (filter_cfl = $(OPTS.filter_cfl)) ===")
    println("drift: change of the total over the run relative to the largest total seen;")
    println("filter: the part of it the filter's own passes made")
    println("\n--- Woodward-Colella N=800 ---")
    for (flab, cl) in FILTERS
        print_budget("$flab, permissive", budget_run(wc_build(cl), WC_T))
    end
    for (flab, cl) in FILTERS
        print_budget("$flab, floor 1e-6",
                     budget_run(wc_build(cl; control=StepControl(validity=:permissive,
                                                                 floor_ratio=1e-6)), WC_T))
    end
    println("\n--- planar Noh N=400, cold start (inflow end open) ---")
    for (flab, cl) in FILTERS
        num = Numerics(n_global=(400, 1, 1), art=ArtParams(enabled=true), cfl=NOH_CFL,
                       filt=filt_of(cl), filter_interval=0, filter_cfl=OPTS.filter_cfl,
                       control=StepControl(validity=:permissive))
        r = budget_run(() -> setup(noh_problem(1; N=400, t0=0.0), num), NOH_T)
        print_budget("$flab, permissive", r)
        failed(r) && continue
        xs, ρ, _, _ = case_line_profile(r.solver, r.Q)
        plat, deficit, Rs, _ = noh_metrics(xs, ρ, 1)
        printf("  %-26s plateau %.4f  deficit %+3.0f%%  shock %.4f\n", "", plat / 4,
               100deficit, Rs)
    end
    for (flab, cl) in FILTERS
        num = Numerics(n_global=(400, 1, 1), art=ArtParams(enabled=true), cfl=NOH_CFL,
                       filt=filt_of(cl), filter_interval=0, filter_cfl=OPTS.filter_cfl,
                       control=StepControl(validity=:permissive, floor_ratio=1e-6))
        r = budget_run(() -> setup(noh_problem(1; N=400, t0=0.0), num), NOH_T)
        print_budget("$flab, floor 1e-6", r)
    end
    println("\n--- steepening pulse amp 0.1 N=385, artificial properties on ---")
    for (flab, cl) in FILTERS
        build = () -> pulse_solver(Float64, 385; amp=0.1, art=true, mirror=false, cl,
                                   filter_interval=0)
        print_budget("$flab, permissive", budget_run(build, 0.7))
    end
    println("\n--- species layer against the wall, steepening pulse, N=385 ---")
    for (ratio, gh, lab) in ((1.0, PULSE_G, "equal gases"), (5.04, 1.09, "air/SF6"))
        for (flab, cl) in FILTERS
            r = budget_run(layer_build(cl; N=385, ratio=ratio, gamma_heavy=gh), 0.7)
            print_budget("$flab, $lab", r)
            failed(r) && continue
            s, Q = r.solver, r.Q
            nx = s.decomp.n_local[1]
            Y1 = [Q[gidx(s, i, 1, 1), 1] / (Q[gidx(s, i, 1, 1), 1] + Q[gidx(s, i, 1, 1), 2])
                  for i in 1:nx]
            m1 = volume_integral(s, view(Q, :, :, :, 1))
            m2 = volume_integral(s, view(Q, :, :, :, 2))
            printf("  %-26s Y_light range %+.4f .. %.4f  width %d cells  " *
                   "species mass %.6f / %.6f  negative species %d\n", "",
                   minimum(Y1), maximum(Y1), count(y -> 0.05 < y < 0.95, Y1), m1, m2,
                   r.report.negative_species)
        end
    end
end

# --- part: float32 ------------------------------------------------------------

function noh_planar_T(::Type{T}, N; cl, cfl=NOH_CFL, filter_cfl=OPTS.filter_cfl) where {T}
    per = (PeriodicBC(), PeriodicBC())
    h = one(T) / T(N - 1)
    γ = NOH_G
    inflow = DirichletBC((x, y, z, t) -> begin
        ρ, uu, _ = noh_exact(Float64(x), isfinite(t) ? Float64(t) : 0.0, 1, γ)
        Prim(rho=T(ρ), u=(T(uu), zero(T), zero(T)), p=T(NOH_P0))
    end)
    solver = Solver(; n_global=(N, 1, 1), L_domain=(one(T), h, h),
                    bcs=((SlipWallBC(), inflow), per, per),
                    eos=IdealSpecies(T, "gas"; R=one(T), gamma=T(γ)),
                    transport=Transport{T}(mu0=zero(T)), art=ArtParams{T}(enabled=true),
                    deriv=lele_d1_6(T), filt=filt_of(cl, T), cfl=T(cfl),
                    filter_interval=1, filter_cfl=filter_cfl,
                    control=StepControl(validity=:permissive))
    Q = allocate_state(solver)
    initialize!(solver, Q, (x, y, z) -> Prim(rho=one(T), u=(-one(T), zero(T), zero(T)),
                                             p=T(NOH_P0)))
    return solver, Q
end

function woodward_T(::Type{T}, N; cl, cfl=0.3, filter_cfl=OPTS.filter_cfl) where {T}
    per = (PeriodicBC(), PeriodicBC())
    h = one(T) / T(N - 1)
    δ = 2 / (N - 1)
    solver = Solver(; n_global=(N, 1, 1), L_domain=(one(T), h, h),
                    bcs=((SlipWallBC(), SlipWallBC()), per, per),
                    eos=IdealSpecies(T, "gas"; R=one(T), gamma=T(1.4)),
                    transport=Transport{T}(mu0=zero(T)), art=ArtParams{T}(enabled=true),
                    deriv=lele_d1_6(T), filt=filt_of(cl, T), cfl=T(cfl),
                    filter_interval=1, filter_cfl=filter_cfl,
                    control=StepControl(validity=:permissive))
    Q = allocate_state(solver)
    initialize!(solver, Q, (x, y, z) -> begin
        xx = Float64(x)
        p = 1000 * (1 - tanh_blend(xx, 0.1, δ)) +
            0.01 * (tanh_blend(xx, 0.1, δ) - tanh_blend(xx, 0.9, δ)) +
            100 * tanh_blend(xx, 0.9, δ)
        Prim(rho=one(T), u=(zero(T), zero(T), zero(T)), p=T(p))
    end)
    return solver, Q
end

function float32_part()
    for T in (Float64, Float32)
        println("\n=== $T: planar Noh N=400 and Woodward-Colella N=800, C6 cascade3 ===")
        for (flab, cl) in FILTERS
            r = attempt() do
                solver, Q = noh_planar_T(T, 400; cl)
                run!(solver, Q; tfinal=T(NOH_T), nmax=CAP)
                completed(solver, NOH_T) || return "step cap"
                xs = [Float64(xcoord(solver, 1, i)) for i in 1:400]
                CL.exchange_state!(Q, solver.decomp); CL.primitives!(solver, Q)
                ρ = [Float64(solver.rho[gidx(solver, i, 1, 1)]) for i in 1:400]
                plat, deficit, Rs, epre = noh_metrics(xs, ρ, 1)
                rep = state_report(solver, Q)
                sprintf("plateau %.4f  deficit %+3.0f%%  shock %.4f  L1 pre %.2e  " *
                         "inadmissible %d  e_min %+.4f  steps %d", plat / 4, 100deficit,
                         Rs, epre, rep.inadmissible, rep.e_min, solver.step)
            end
            println("  ", pad("Noh", 12), pad(flab, 10), r)
            flush(stdout)
        end
        for (flab, cl) in FILTERS
            r = attempt() do
                solver, Q = woodward_T(T, WC_N; cl)
                run!(solver, Q; tfinal=T(WC_T), nmax=CAP)
                completed(solver, WC_T) || return "step cap"
                xs = [Float64(xcoord(solver, 1, i)) for i in 1:WC_N]
                CL.exchange_state!(Q, solver.decomp); CL.primitives!(solver, Q)
                ρ = [Float64(solver.rho[gidx(solver, i, 1, 1)]) for i in 1:WC_N]
                xr, ρr, _, _ = read_ref("woodward_colella.csv")
                imax = argmax(ρ)
                sprintf("L1 rho %.3e  peak rho %.4f at %.4f  rho_min %.4f  steps %d",
                         l1(ρ, [interp1(xr, ρr, x) for x in xs]), ρ[imax], xs[imax],
                         minimum(ρ), solver.step)
            end
            println("  ", pad("Woodward", 12), pad(flab, 10), r)
            flush(stdout)
        end
        println("\n--- $T: smooth pulse amp 0.01 against its mirror, artificial properties on ---")
        pulse_table(T, (49, 97, 193); amp=0.01, art=true, tfinal=0.7)
    end
end

for part in PARTS
    part == "shock" ? shock_part() :
    part == "closures" ? closures_part() :
    part == "pulse" ? pulse_part() :
    part == "budget" ? budget_part() :
    part == "float32" ? float32_part() :
    part == "steepening" ? steepening_part() :
    error("unknown part '$part'; want shock, closures, pulse, budget, float32 " *
          "or steepening")
end
