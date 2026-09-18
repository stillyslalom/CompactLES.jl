# Constant annihilation at a closed edge: the round-off a closure row leaves
# on a constant, separated into coefficient cancellation, summation
# round-off and solve conditioning, in both precisions, and what it costs
# a run.
#
#   julia --project=. -t 1 bench/constantfloor.jl [parts=all]
#   julia --project=. -t 1 bench/constantfloor.jl parts=freestream
#
# A periodic or interior row differences its taps before it multiplies
# (`coeffs[m] (f_{i+m} - f_{i-m})`), so a constant is annihilated exactly
# whatever its size. A closure row is a plain weighted sum over the first
# points of the line, so a constant c leaves (Σ w_j) c from the weights'
# own rounding plus the rounding of the products and their accumulation,
# of order eps · c · Σ|w_j| / h, before the line solve amplifies it. The
# anchored form Σ w_j (f_j - f_1) annihilates a constant exactly and is
# evaluated here beside the plain one.
#
# Parts:
#
#   rowsums    Σ w_j of every closure row as stored (Float64 and Float32)
#              and as prescaled by 1/h at two domain lengths, one that
#              makes 1/h an integer and one that does not; a filter row is
#              read as Σ rhs − Σ lhs, which is zero when the row passes a
#              constant
#   constant   one derivative (or one filter pass) of a constant on the
#              closed line: the closure rows' fill residual before the
#              solve and the solved residual at the wall window and in the
#              interior, both normalized to eps-scale (× h / c for a
#              derivative, / c for a filter), the anchored fill beside the
#              plain one, both precisions, two constants (a power of two
#              and a generic value), two lengths, three resolutions
#   offset     a small perturbation over a large constant, f = c + sin(3x),
#              c from 0 to 1e9: the derivative error against 3 cos(3x) at
#              the wall window and in the interior, plain and anchored,
#              both precisions
#   freestream the consequence through the solver: a uniform state between
#              slip walls (nondimensional, and SI-like at p = 1e5 Pa) in
#              both precisions, the right-hand side's residual at the wall
#              window and in the interior per component, then the wall-normal
#              velocity and pressure that 500, 1000 and 2000 steps grow, filter
#              on and off, with a periodic line as the control
#   mode       the growth of that velocity from its round-off seed under the
#              cascade rows, Float64, 1000 to 4000 steps: closure, filter,
#              artificial properties, resolution, tangential velocity, CFL
#   jacobian   one step linearized about the uniform state: the leading
#              eigenvalue of the amplification matrix as a growth rate, the
#              count of growing eigenvalues, and the leading eigenvector's
#              share within four nodes of a wall, per closure and filter
#
# Scratch tooling, like everything else in bench/: it prints tables and
# asserts nothing. The conclusions are in reference/CALIBRATION_APPENDIX.md.

using MPI
MPI.Init(threadlevel=:funneled)
using CompactLES
using Printf

const CL = CompactLES
MPI.Comm_size(MPI.COMM_WORLD) == 1 || error("run this study on one rank")

const OPTS = CL.script_args(ARGS, (parts="all",))
const PARTS = OPTS.parts == "all" ?
    ["rowsums", "constant", "offset", "freestream", "mode", "jacobian"] :
    split(OPTS.parts, ',')

sprintf(fmt::String, args...) = Printf.format(Printf.Format(fmt), args...)
printf(fmt::String, args...) = print(sprintf(fmt, args...))
pad(s, n) = rpad(s, n)

const DERIVS = (("C6 cascade3", T -> lele_d1_6(T; closures=:cascade3)),
                ("C6 cascade4", T -> lele_d1_6(T; closures=:cascade4)),
                ("C6 BL", T -> lele_d1_6(T; closures=:brady_livescu)),
                ("C8 cascade3", T -> lele_d1_8(T; closures=:cascade3)),
                ("C8 BL", T -> lele_d1_8(T; closures=:brady_livescu)),
                ("C10", T -> lele_d1_10(T)),
                ("C6 neutral3", T -> lele_d1_6(T)))
const FILTERS = (("filter onesided", T -> compact_filter(T(0.45), T)),
                 ("filter cascade", T -> compact_filter(T(0.45), T; closures=:cascade)),
                 ("gaussian", T -> gaussian_filter(T)),
                 ("pyranda filter", T -> pyranda_filter(T)))

# --- the line ---------------------------------------------------------------------
#
# One line of N nodes along dimension 1, both ends closed, the transverse
# dimensions collapsed. The plan is the solver's own (prescaled by the
# spacing, factorized once); the fill is re-done here in the plan's
# arithmetic order so the closure rows' residual can be read before the
# solve, and in the anchored form beside it.

struct Line{T,P}
    decomp::CL.Decomp{T}
    plan::P
    n::Int
    h::T
end

function Line(::Type{T}, N, scheme, L) where {T}
    decomp = CL.Decomp{T}((N, 1, 1), (false, true, true))
    h = T(L) / T(N - 1)
    plan = CL.plan_direction(decomp, scheme, 1, h)
    Line{T,typeof(plan)}(decomp, plan, N, h)
end

symmetric(line) = line.plan.scheme.symmetric
nclosure(line) = length(line.plan.clo)

"The line's values as a plain vector, from a function of x on [0, L]."
line_values(::Type{T}, N, L, fn) where {T} = T[fn(T(L) * T(i - 1) / T(N - 1)) for i in 1:N]

# The fill of one line, plain (the plan's own arithmetic, as `_fill_lines!`
# does it) or anchored (every tap taken relative to the row's first point,
# which annihilates a constant exactly; the interior rows already difference).
function fill_line(line::Line{T}, f::Vector{T}; anchored::Bool=false) where {T}
    plan = line.plan
    n = line.n
    B = zeros(T, n)
    ci = plan.ci
    M = length(ci)
    nclo = length(plan.clo); nchi = length(plan.chi)
    # An anchored filter row still has to pass the anchor itself: the row's
    # left-hand side applied to a constant a is (Σ lhs) a, and that is what
    # the subtracted taps took out. A derivative row's target is zero.
    # (Every preset stores a zero where a row's left-hand side reaches past
    # the edge, so the plain sum is the row's sum for tridiagonal and banded
    # rows alike.)
    lhs_sum(j) = sum(plan.scheme.closures[j].lhs)
    for jr in 1:nclo
        rhs = plan.clo[jr]
        i0 = plan.clo_first[jr] - 1
        anchor = anchored ? f[i0 + 1] : zero(T)
        acc = zero(T)
        for κ in eachindex(rhs)
            acc += rhs[κ] * (f[i0 + κ] - anchor)
        end
        B[jr] = symmetric(line) && anchored ? acc + anchor * lhs_sum(jr) : acc
    end
    for jr in 1:nchi
        rhs = plan.chi[jr]
        i0 = n + 2 - plan.chi_first[jr]
        anchor = anchored ? f[i0 - 1] : zero(T)
        acc = zero(T)
        for κ in eachindex(rhs)
            acc += rhs[κ] * (f[i0 - κ] - anchor)
        end
        B[n + 1 - jr] = symmetric(line) && anchored ? acc + anchor * lhs_sum(jr) : acc
    end
    for i in nclo+1:n-nchi
        if symmetric(line)
            acc = plan.a0 * f[i]
            for m in 1:M
                acc += ci[m] * (f[i + m] + f[i - m])
            end
        else
            acc = zero(T)
            for m in 1:M
                acc += ci[m] * (f[i + m] - f[i - m])
            end
        end
        B[i] = acc
    end
    return B
end

"The solved line from a fill vector, through the plan's own factorization."
function solve_line(line::Line{T}, B::Vector{T}) where {T}
    Bm = reshape(copy(B), line.n, 1)
    CL.solve_lines!(Bm, line.plan.line_solver)
    return vec(Bm)
end

const W = 4
wall_max(v) = maximum(abs, v[[1:W; end-W+1:end]])
interior_max(v) = maximum(abs, v[W+1:end-W])

# --- part: rowsums ----------------------------------------------------------------

function rowsums_part()
    println("\n=== closure-row weight sums ===")
    println("derivative rows: Σ w (zero for a constant); filter rows: Σ rhs − Σ lhs")
    println("stored as T, and prescaled by 1/h at N = 129 for L = 1 (1/h an " *
            "integer) and L = 2π")
    println("  scheme            row   Float64 stored  Float32 stored  " *
            "F64 L=1       F32 L=1       F64 L=2π      F32 L=2π")
    for (label, mk) in (DERIVS..., FILTERS...)
        s64 = mk(Float64); s32 = mk(Float32)
        rows64 = s64.closures; rows32 = s32.closures
        p = Dict((T, L) => Line(T, 129, mk(T), L).plan
                 for T in (Float64, Float32), L in (1.0, 2π))
        for j in eachindex(rows64)
            target(row) = sum(row.lhs)
            sym = s64.symmetric
            st64 = sum(rows64[j].rhs) - (sym ? target(rows64[j]) : 0)
            st32 = sum(rows32[j].rhs) - (sym ? target(rows32[j]) : 0)
            sc(T, L) = (pl = p[(T, L)];
                        sum(pl.clo[j]) - (sym ? target(pl.scheme.closures[j]) : 0))
            printf("  %-18s%2d   %+.3e      %+.3e      %+.3e    %+.3e    %+.3e    %+.3e\n",
                   label, j, st64, st32, sc(Float64, 1.0), sc(Float32, 1.0),
                   sc(Float64, 2π), sc(Float32, 2π))
        end
    end
end

# --- part: constant ---------------------------------------------------------------

function constant_part()
    println("\n=== a constant on the closed line ===")
    println("derivative: residual × h / c; filter: (F c − c) / c; eps is 2.2e-16 " *
            "(Float64) and 6.0e-8 (Float32)")
    println("fill: the closure rows' right-hand side before the solve; solved: " *
            "after it, wall window and interior; anchored: the fill with every " *
            "tap relative to the row's first point")
    for (label, mk) in (DERIVS..., FILTERS...)
        println("\n--- $label ---")
        println("  T        L     c          N     fill        anchored    " *
                "solved wall  solved int")
        for T in (Float64, Float32), L in (1.0, 2π), c in (1.0, 12345.678),
            N in (33, 129, 513)
            line = Line(T, N, mk(T), L)
            f = fill(T(c), N)
            Bp = fill_line(line, f)
            Ba = fill_line(line, f; anchored=true)
            sym = symmetric(line)
            nc = nclosure(line)
            if sym
                # A filter row should reproduce c: read the fill against the
                # row's left-hand side applied to c.
                lhs_c(j) = c * sum(line.plan.scheme.closures[j].lhs)
                fillres = maximum(abs(Float64(Bp[j]) - lhs_c(j)) for j in 1:nc) / c
                ancres = maximum(abs(Float64(Ba[j]) - lhs_c(j)) for j in 1:nc) / c
                out = solve_line(line, Bp) .- T(c)
                scale = 1 / c
            else
                fillres = maximum(abs, Float64.(Bp[1:nc])) * Float64(line.h) / c
                ancres = maximum(abs, Float64.(Ba[1:nc])) * Float64(line.h) / c
                out = solve_line(line, Bp)
                scale = Float64(line.h) / c
            end
            printf("  %-8s %-5s %-10.4g %4d  %.2e    %.2e    %.2e     %.2e\n",
                   T, L == 1.0 ? "1" : "2π", c, N, fillres, ancres,
                   wall_max(Float64.(out)) * scale, interior_max(Float64.(out)) * scale)
        end
        flush(stdout)
    end
end

# --- part: offset -----------------------------------------------------------------

function offset_part()
    println("\n=== a perturbation over a constant: f = c + sin(3x), derivative " *
            "against 3 cos(3x), N = 129, L = 1 ===")
    println("errors relative to the derivative's magnitude 3; the interior column " *
            "is the floor the stored field itself sets")
    for (label, mk) in DERIVS
        println("\n--- $label ---")
        println("  T        c          plain wall   anchored wall  interior")
        for T in (Float64, Float32), c in (0.0, 1.0, 1e3, 1e6, 1e9)
            N = 129
            line = Line(T, N, mk(T), 1.0)
            f = line_values(T, N, 1.0, x -> T(c) + sin(3x))
            df = line_values(Float64, N, 1.0, x -> 3cos(3x))
            ep = Float64.(solve_line(line, fill_line(line, f))) .- df
            ea = Float64.(solve_line(line, fill_line(line, f; anchored=true))) .- df
            printf("  %-8s %-10.1e %.2e     %.2e       %.2e\n", T, c,
                   wall_max(ep) / 3, wall_max(ea) / 3, interior_max(ep) / 3)
        end
        flush(stdout)
    end
end

# --- part: freestream -------------------------------------------------------------
#
# The state is uniform, with a tangential velocity so the x-momentum flux is
# the constant pressure through the wall rows and every other x flux is
# zero; a periodic line annihilates its constants exactly, so whatever the
# right-hand side reads is the closure rows'.

function uniform_solver(::Type{T}, N, deriv; rho, v, p, R, gamma, filter_on,
                        periodic=false, art_on=true, cfl=0.5,
                        control=StepControl(validity=:permissive)) where {T}
    per = (PeriodicBC(), PeriodicBC())
    h = one(T) / T(N - 1)
    solver = Solver(; n_global=(N, 1, 1), L_domain=(one(T), h, h),
                    bcs=(periodic ? per : (SlipWallBC(), SlipWallBC()), per, per),
                    eos=IdealSpecies(T, "gas"; R=T(R), gamma=T(gamma)),
                    transport=Transport{T}(mu0=zero(T)), art=ArtParams{T}(enabled=art_on),
                    deriv=deriv, filt=compact_filter(T(0.45), T), cfl=T(cfl),
                    filter_interval=filter_on ? 1 : 0, filter_cfl=T(0.35),
                    control=control)
    Q = allocate_state(solver)
    initialize!(solver, Q, (x, y, z) -> Prim(rho=T(rho), u=(zero(T), T(v), zero(T)),
                                             p=T(p)))
    return solver, Q
end

function drift(::Type{T}, st, deriv; filter_on, periodic, nmax) where {T}
    solver, Q = uniform_solver(T, 101, deriv; st..., filter_on=filter_on, periodic=periodic)
    run!(solver, Q; tfinal=T(Inf), nmax=nmax)
    CL.exchange_state!(Q, solver.decomp)
    CL.primitives!(solver, Q)
    n = solver.decomp.n_local[1]
    u = [Float64(solver.u[gidx(solver, i, 1, 1)]) for i in 1:n]
    dp = [Float64(solver.p[gidx(solver, i, 1, 1)]) / st.p - 1 for i in 1:n]
    return (u=u, dp=dp, t=Float64(solver.t))
end

function freestream_part()
    println("\n=== a uniform state on N = 101 nodes, between slip walls and periodic ===")
    states = (("nondimensional, p = 1", (rho=1.0, v=0.1, p=1.0, R=1.0, gamma=1.4)),
              ("nondimensional, generic", (rho=0.9, v=0.1, p=1.1, R=1.0, gamma=1.4)),
              ("SI-like, p = 1e5 Pa", (rho=1.2, v=50.0, p=1e5, R=287.0, gamma=1.4)))
    rows = (("walls, C6 cascade3", DERIVS[1][2], false),
            ("walls, C6 BL", DERIVS[3][2], false),
            ("periodic, C6", DERIVS[1][2], true))
    for (slabel, st) in states
        println("\n--- $slabel: rho = $(st.rho), v = $(st.v) (tangential), p = $(st.p) ---")
        println("right-hand side of the initial state, |dQ| max over the wall window / " *
                "the interior, per component (rho, rho u, rho v, rho w, E)")
        for T in (Float64, Float32), (rlabel, mk, periodic) in rows
            solver, Q = uniform_solver(T, 101, mk(T); st..., filter_on=false,
                                       periodic=periodic)
            apply_bcs!(solver, Q)
            dQ = zero(Q)
            compute_rhs!(solver, Q, dQ)
            n = solver.decomp.n_local[1]
            cols = String[]
            for c in 1:solver.equations.n_cons
                v = [Float64(dQ[gidx(solver, i, 1, 1), c]) for i in 1:n]
                push!(cols, sprintf("%.1e/%.1e", wall_max(v), interior_max(v)))
            end
            printf("  %-8s %-20s %s\n", T, rlabel, join(cols, "  "))
        end
        println("after 500 / 1000 / 2000 steps at cfl 0.5: the wall-normal velocity " *
                "|u| max and the pressure's relative drift max, wall window / interior")
        for T in (Float64, Float32), (rlabel, mk, periodic) in rows, filter_on in (false, true)
            cols = String[]
            t = 0.0
            for nmax in (500, 1000, 2000)
                r = drift(T, st, mk(T); filter_on=filter_on, periodic=periodic, nmax=nmax)
                push!(cols, sprintf("|u| %.1e/%.1e dp/p %.1e/%.1e", wall_max(r.u),
                                    interior_max(r.u), wall_max(r.dp), interior_max(r.dp)))
                t = r.t
            end
            printf("  %-8s %-20s filter %-4s %s  (t = %.2e)\n", T, rlabel,
                   filter_on ? "on" : "off", join(cols, " | "), t)
            flush(stdout)
        end
    end
end

# --- part: mode ---------------------------------------------------------------------
#
# The freestream ladder grows a wall-normal velocity from its round-off seed
# under the cascade rows, by a factor well above a random walk's √steps.
# This part reads the growth of |u| between slip walls on the generic
# nondimensional state in Float64, from 1000 to 4000 steps, against the
# closure, the filter, the artificial properties, the resolution and the
# tangential velocity, with the rate per unit time fitted between the
# last two readings.

function mode_part()
    println("\n=== growth of the wall-normal velocity from the round-off seed, " *
            "Float64, generic uniform state ===")
    println("  configuration                                       |u| at 1000 / 2000 / " *
            "4000 steps            rate per unit time")
    base = (rho=0.9, v=0.1, p=1.1, R=1.0, gamma=1.4)
    rows = []
    for (dlabel, mk) in (DERIVS[1], DERIVS[2], DERIVS[3], DERIVS[7]),
        filter_on in (false, true)
        push!(rows, ("N=101 $dlabel filter $(filter_on ? "on" : "off")", mk, 101,
                     filter_on, true, base))
    end
    push!(rows, ("N=101 C6 cascade3 filter off, art off", DERIVS[1][2], 101, false,
                 false, base))
    push!(rows, ("N=101 C6 cascade3 filter off, v = 0", DERIVS[1][2], 101, false,
                 true, merge(base, (v=0.0,))))
    push!(rows, ("N=101 C6 cascade3 filter off, v = 0.5", DERIVS[1][2], 101, false,
                 true, merge(base, (v=0.5,))))
    push!(rows, ("N=51 C6 cascade3 filter off", DERIVS[1][2], 51, false, true, base))
    push!(rows, ("N=201 C6 cascade3 filter off", DERIVS[1][2], 201, false, true, base))
    push!(rows, ("N=101 C6 cascade3 filter off, cfl 0.25", DERIVS[1][2], 101, false,
                 true, base, 0.25))
    for row in rows
        label, mk, N, filter_on, art_on, st = row[1:6]
        cfl = length(row) > 6 ? row[7] : 0.5
        us = Float64[]; ts = Float64[]
        failure = ""
        for nmax in (1000, 2000, 4000)
            solver, Q = uniform_solver(Float64, N, mk(Float64); st..., filter_on=filter_on,
                                       art_on=art_on, cfl=cfl)
            try
                run!(solver, Q; tfinal=Inf, nmax=nmax)
            catch err
                err isa SolverFailure || rethrow()
                failure = sprintf("FAILED %s at step %d, t = %.2f", err.reason, err.step,
                                  err.t)
                break
            end
            CL.exchange_state!(Q, solver.decomp)
            CL.primitives!(solver, Q)
            n = solver.decomp.n_local[1]
            push!(us, maximum(abs(solver.u[gidx(solver, i, 1, 1)]) for i in 1:n))
            push!(ts, solver.t)
        end
        if length(us) < 3
            printf("  %-52s %s %s\n", label,
                   join((sprintf("%.1e", u) for u in us), " / "), failure)
        else
            rate = us[3] > 0 && us[2] > 0 ? log(us[3] / us[2]) / (ts[3] - ts[2]) : NaN
            printf("  %-52s %.1e / %.1e / %.1e   %+.2f  (t = %.1f)\n", label, us...,
                   rate, ts[3])
        end
        flush(stdout)
    end
end

# --- part: jacobian ---------------------------------------------------------------
#
# One step linearized about the uniform state by finite differences: the
# amplification matrix G of the five conserved components on N nodes, its
# largest eigenvalue modulus as a growth rate per unit time, and where the
# leading eigenvector lives (the fraction of its norm within four nodes of
# either wall). The artificial properties are off, since their sensors are
# not differentiable at a uniform state; the mode part shows they do not
# set the rate. A filtered row applies the unrelaxed pass after the step.

using LinearAlgebra

function step_map(solver, Q0, dt, filter_on)
    Q = copy(Q0)
    dQ = zero(Q); du = zero(Q)
    solver.t = 0.0
    CL.step!(solver, Q, dQ, du, dt)
    filter_on && filter_state!(solver, Q)
    return Q
end

# The step map is differenced centrally with delta = 1e-5 · max(|Q|, 1); a
# neutral row then reads 1 + O(1e-9), against 1 + O(1e-7) for a one-sided
# 1e-7 difference, so a 1 + 1e-8 neutrality gate is resolved. `ladder`
# repeats the row at 3e-6 and 3e-5 so a
# reading can be told from its perturbation dependence.
function jacobian_row(label, deriv; N=51, filter_on=false, cl=:onesided, mu=0.0,
                      wall=:slip, alphaf=0.45, delta=1e-5, ladder=false)
    st = (rho=0.9, v=0.1, p=1.1, R=1.0, gamma=1.4)
    per = (PeriodicBC(), PeriodicBC())
    h = 1.0 / (N - 1)
    bc = wall === :noslip ? NoSlipWallBC() :
         wall === :dirichlet ? DirichletBC((x, y, z, t) -> Prim(rho=st.rho, u=(0.0, st.v, 0.0),
                                                                 p=st.p)) :
         SlipWallBC()
    solver = Solver(; n_global=(N, 1, 1), L_domain=(1.0, h, h), bcs=((bc, bc), per, per),
                    eos=IdealSpecies("gas"; R=st.R, gamma=st.gamma),
                    transport=Transport(mu0=mu), art=ArtParams(enabled=false),
                    deriv=deriv, filt=compact_filter(alphaf; closures=cl),
                    filter_interval=filter_on ? 1 : 0, filter_cfl=0.0, cfl=0.5)
    Q0 = allocate_state(solver)
    initialize!(solver, Q0, (x, y, z) -> Prim(rho=st.rho, u=(0.0, st.v, 0.0), p=st.p))
    apply_bcs!(solver, Q0)
    c = sqrt(st.gamma * st.p / st.rho)
    dt = 0.5 * h / c
    ncons = solver.equations.n_cons
    base = step_map(solver, Q0, dt, filter_on)
    idx = [(gidx(solver, i, 1, 1), comp) for comp in 1:ncons for i in 1:N]
    m = length(idx)
    function amplification(delta)
        G = zeros(m, m)
        for (j, (I, comp)) in enumerate(idx)
            Qp = copy(Q0); Qm = copy(Q0)
            eps = delta * max(abs(Q0[I, comp]), 1.0)
            Qp[I, comp] += eps; Qm[I, comp] -= eps
            Sp = step_map(solver, Qp, dt, filter_on)
            Sm = step_map(solver, Qm, dt, filter_on)
            for (i, (J, cc)) in enumerate(idx)
                G[i, j] = (Sp[J, cc] - Sm[J, cc]) / (2eps)
            end
        end
        G
    end
    if ladder
        for d in (3e-6, 3e-5)
            printf("  %-44s |λ|max %.10f   (delta %.0e)\n", label,
                   maximum(abs, eigvals(amplification(d))), d)
        end
    end
    vals, vecs = eigen(amplification(delta))
    k = argmax(abs.(vals))
    λ = vals[k]
    v = vecs[:, k]
    wallnorm = 0.0
    for (i, (_, _)) in enumerate(idx)
        node = (i - 1) % N + 1
        (node <= 4 || node > N - 4) && (wallnorm += abs2(v[i]))
    end
    rate = log(abs(λ)) / dt
    ngrow = count(x -> abs(x) > 1 + 1e-12, vals)
    printf("  %-44s |λ|max %.10f  rate %+.3f  growing %3d of %d  wall share %.2f\n",
           label, abs(λ), rate, ngrow, m, wallnorm / sum(abs2, v))
    flush(stdout)
end

function jacobian_part()
    println("\n=== one linearized step at the uniform state, N = 51, cfl 0.5, " *
            "artificial properties off ===")
    println("rate = ln|λ|max / dt per unit time (the mode part's units); growing = " *
            "eigenvalues outside the unit circle")
    c3 = lele_d1_6(closures=:cascade3)
    jacobian_row("C6 neutral3, unfiltered", lele_d1_6(); ladder=true)
    jacobian_row("C6 neutral3, onesided filter (unrelaxed)", lele_d1_6(); filter_on=true,
                 ladder=true)
    jacobian_row("C6 neutral3, no-slip, mu = 0.005, onesided", lele_d1_6(); mu=0.005,
                 wall=:noslip, filter_on=true)
    jacobian_row("C6 neutral3, Dirichlet ends, unfiltered", lele_d1_6(); wall=:dirichlet)
    jacobian_row("C6 neutral3, unfiltered, N = 101", lele_d1_6(); N=101)
    jacobian_row("C6 neutral3, onesided filter, N = 101", lele_d1_6(); N=101, filter_on=true)
    jacobian_row("C6 cascade3, unfiltered", c3)
    jacobian_row("C6 cascade3, onesided filter (unrelaxed)", c3; filter_on=true)
    jacobian_row("C6 cascade3, cascade filter (unrelaxed)", c3; filter_on=true,
                 cl=:cascade)
    jacobian_row("C6 cascade4, unfiltered", lele_d1_6(closures=:cascade4))
    jacobian_row("C6 BL, unfiltered", lele_d1_6(closures=:brady_livescu))
    jacobian_row("C6 BL, onesided filter (unrelaxed)", lele_d1_6(closures=:brady_livescu);
                 filter_on=true)
    jacobian_row("C8 neutral3, unfiltered", lele_d1_8())
    jacobian_row("C8 cascade3, unfiltered", lele_d1_8(closures=:cascade3))
    jacobian_row("C8 BL, unfiltered", lele_d1_8(closures=:brady_livescu))
    jacobian_row("C10 neutral3, unfiltered", lele_d1_10())
    jacobian_row("C10 cascade3, unfiltered", lele_d1_10(closures=:cascade3))
    jacobian_row("C6 cascade3, no-slip, mu = 0.005, unfiltered", c3; mu=0.005,
                 wall=:noslip)
    jacobian_row("C6 cascade3, slip, mu = 0.005, unfiltered", c3; mu=0.005)
    jacobian_row("C6 cascade3, Dirichlet ends, unfiltered", c3; wall=:dirichlet)
    jacobian_row("C6 cascade3, Dirichlet ends, onesided filter", c3;
                 wall=:dirichlet, filter_on=true)
    jacobian_row("C6 cascade3, onesided filter alphaf 0.40", c3; filter_on=true,
                 alphaf=0.40)
    jacobian_row("C6 cascade3, onesided filter alphaf 0.30", c3; filter_on=true,
                 alphaf=0.30)
    jacobian_row("C6 cascade3, unfiltered, N = 101", c3; N=101)
end

for part in PARTS
    part == "rowsums" ? rowsums_part() :
    part == "mode" ? mode_part() :
    part == "jacobian" ? jacobian_part() :
    part == "constant" ? constant_part() :
    part == "offset" ? offset_part() :
    part == "freestream" ? freestream_part() :
    error("unknown part '$part'; want rowsums, constant, offset, freestream, mode " *
          "or jacobian")
end
