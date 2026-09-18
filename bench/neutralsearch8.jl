# A neutral closure set for the C8 tridiagonal first derivative: the family,
# the search over it, the selection and the production gates.
#
#   julia --project=. -t 1 bench/neutralsearch8.jl [parts=all]
#   julia --project=. -t 1 bench/neutralsearch8.jl parts=scan grid=81
#   julia --project=. -t 1 bench/neutralsearch8.jl parts=select blas=12
#   julia --project=. -t 1 bench/neutralsearch8.jl parts=sweep blas=12 \
#       members=reuse_c6,band7_10 Ns=12:600,610:10:1200
#
# `parts=all` leaves out `select` and `sweep`, which are minutes and a few
# hundred seconds per member.
#
# `lele_d1_8` takes three closure rows at a closed edge. Under the
# `:cascade3` rows a uniform inviscid state between slip walls grows a
# wall-normal velocity at 1.4 per unit time, the mode that `:neutral3`
# removes for `lele_d1_6`. The C6 rows are derived for the C6 interior, so
# this study builds the three-row family, searches it for members whose
# injected slip-wall acoustic operator has a purely imaginary spectrum, and
# puts the survivors through the production gates.
#
# The family widens each cascade row by one point at fixed order, which is
# the construction of Carpenter, Gottlieb & Abarbanel (1993) that produced
# the C6 set:
#
#   g_1 + a g_2                = Σ_{k=1}^{4} w_k f_k   third order,  a free
#   b g_1 + g_2 + c g_3        = Σ_{k=1}^{5} w_k f_k   fourth order, b, c free
#   d g_2 + g_3 + e g_4        = Σ_{k=1}^{6} w_k f_k   fifth order,  d, e free
#
# with (a, b, c) = (2, 1/4, 1/4) and (d, e) = (1/3, 1/3) the cascade and
# (a, b, c) = (0, 3/5, 3/10) the C6 `:neutral3` rows. Row 3 is the one the
# C6 family does not have. Six points and a two-parameter left-hand side
# carry eight coefficients against seven conditions for sixth order, so the
# sixth-order members are a line, not a plane: the family part below prints
# it as 2d + e = 1, through the cascade's (1/3, 1/3). Every sixth-order row
# on that line is used here; the fifth-order plane off it is scanned only to
# show what dropping the order would buy.
#
# Parts:
#
#   family    the rows in exact rationals: the cascade and C6 `:neutral3`
#             members reproduced from the family, the monomial residual of
#             each row through its order and at the next degree, the
#             sixth-order line, the seventh-order row 3 on it, and the
#             assembled closed line against the production plan
#   scan      the neutral set: how many members of a (b, c) grid have an
#             injected acoustic spectrum on the imaginary axis at N = 51 and
#             101, against a and against row 3 along the sixth-order line,
#             then the band's extent at a = 0 and a fifth-order (d, e) slice
#   select    a pool of rational members filtered in three stages of
#             increasing line length, with the error constants of the
#             survivors; the stages are what the shortlist below came from
#   errors    one derivative of exp(sin 3x) on the closed line at N = 49, 97
#             and 193, wall window and interior, for the shortlist beside
#             the C8 cascade and C6 `:neutral3`, with the closed line's
#             condition number
#   sweep     every line length in `Ns`, the long leg: a member neutral at
#             two resolutions can still grow at a third, and this separates
#             the shortlist. Pass `blas` to give the large
#             eigensolves more than the one thread the package sets, and
#             `members` to run a subset of the shortlist
#   jacobian  the production gate: one step linearized about the uniform
#             state by centered differences, slip, no-slip and Dirichlet
#             ends, filtered and not, at N = 51 and 101
#   uniform   forty time units of a uniform state between slip walls under
#             the default relaxed filter, the time-domain reading of the
#             same mode
#
# Scratch tooling, like everything else in bench/: it prints tables and
# asserts nothing.

using MPI
MPI.Init(threadlevel=:funneled)
using CompactLES
using LinearAlgebra
using Printf

const CL = CompactLES
MPI.Comm_size(MPI.COMM_WORLD) == 1 || error("run this study on one rank")

include(joinpath(@__DIR__, "closuresearch.jl"))
using .ClosureSearch: derivative_matrix, acoustic_operator
include(joinpath(@__DIR__, "..", "test", "smooth_cases.jl"))

const OPTS = CL.script_args(ARGS, (parts="all", grid=41, blas=0, members="",
                                   Ns="12:600,610:10:1200"))
# `select` and `sweep` are the two long legs and are left out of `all`: the
# pool filter is minutes and the line-length sweep is a few hundred seconds
# per member. Ask for them by name.
const PARTS = OPTS.parts == "all" ?
    ["family", "scan", "errors", "jacobian", "uniform"] :
    split(OPTS.parts, ',')
OPTS.blas > 0 && BLAS.set_num_threads(OPTS.blas)

sprintf(fmt::String, args...) = Printf.format(Printf.Format(fmt), args...)
printf(fmt::String, args...) = print(sprintf(fmt, args...))

# --- the family -------------------------------------------------------------
#
# A row is fixed by its left-hand side once the order is set, so the weights
# are stored as an affine basis in the two left-hand-side coordinates:
# w = w_base + sub * w_sub + super * w_super, solved once in exact rationals
# against the Vandermonde of the row's points. A member then costs two axpys
# rather than a linear solve, which the scans below build tens of thousands
# of times.

const EXACT = Rational{BigInt}

monomial_derivative(p, x) = p == 0 ? zero(x) : EXACT(p) * x^(p - 1)

"Affine weight basis of the row at node `j` on points `1:npoints`."
function row_basis(j::Int, npoints::Int)
    order = npoints - 1
    points = EXACT[EXACT(k - 1) for k in 1:npoints]
    here = EXACT(j - 1)
    vandermonde = EXACT[points[k]^p for p in 0:order, k in 1:npoints]
    inverse = inv(vandermonde)
    left(sub, super) = EXACT[(j > 1 ? sub * monomial_derivative(p, here - 1) : EXACT(0)) +
                             monomial_derivative(p, here) +
                             super * monomial_derivative(p, here + 1) for p in 0:order]
    base = inverse * left(EXACT(0), EXACT(0))
    (base, inverse * left(EXACT(1), EXACT(0)) .- base,
     inverse * left(EXACT(0), EXACT(1)) .- base)
end

const ROW_POINTS = (4, 5, 6)
const ROW_BASIS = ntuple(j -> row_basis(j, ROW_POINTS[j]), 3)

row_weights(::Type{T}, j, sub, super) where {T} =
    T.(ROW_BASIS[j][1]) .+ T(sub) .* T.(ROW_BASIS[j][2]) .+ T(super) .* T.(ROW_BASIS[j][3])

"Residual of row `j` on the monomial of degree `p`; zero means exact."
function monomial_residual(j, sub, super, p)
    weights = row_weights(EXACT, j, sub, super)
    here = EXACT(j - 1)
    sum(weights[k] * EXACT(k - 1)^p for k in 1:ROW_POINTS[j]) -
        ((j > 1 ? sub * monomial_derivative(p, here - 1) : EXACT(0)) +
         monomial_derivative(p, here) + super * monomial_derivative(p, here + 1))
end

"The five left-hand-side coordinates of one member, in the header's letters."
member(a, b, c, d, e) = (a=a, b=b, c=c, d=d, e=e)

function closure_rows(::Type{T}, m) where {T}
    [CL.ClosureRow{T}((zero(T), one(T), T(m.a)), row_weights(T, 1, 0, m.a)),
     CL.ClosureRow{T}((T(m.b), one(T), T(m.c)), row_weights(T, 2, m.b, m.c)),
     CL.ClosureRow{T}((T(m.d), one(T), T(m.e)), row_weights(T, 3, m.d, m.e))]
end

"The C8 interior (Lele 1992) carrying one member's closure rows."
scheme_of(m, ::Type{T}=Float64) where {T} =
    CL.CompactScheme{T}("C8 trial", T(3//8), zero(T), T[25//32, 1//20, -1//480],
                        false, closure_rows(T, m))

"Largest real part of the injected slip-wall acoustic operator, in c/L units."
acoustic_growth(m::NamedTuple, N) = acoustic_growth(scheme_of(m), N)
acoustic_growth(scheme, N) = maximum(real, eigvals(first(acoustic_operator(scheme, N))))

# A neutral member's reading is the eigensolver's own floor, which rises with
# the operator norm and so with the line length: 1e-14 at N = 51 against
# 2e-12 at N = 1180 for the member selected below. The verdict tolerance
# therefore loosens with N, and a member that grows reads 1e-2 or more, far
# above either value.
const NEUTRAL_TOL = 1e-12
neutral_tolerance(N) = N <= 151 ? NEUTRAL_TOL : 1e-11
const GROWTH_REJECT = 1e-10

# The shortlist, in exact rationals. `cascade3` is the set this study
# replaced as the C8 default. `reuse_c6` is the C6 `:neutral3` rows over the
# C6 interior row, which is the same construction `:cascade3` uses;
# `band16_25`, `band33_50` and `band7_10` are the other rational members of
# the C6 neutral band over that row; the rest are what the select part's pool
# turned up, including `row3_7th`, the seventh-order row 3 that the family
# part isolates.
#
# Of these, `reuse_c6` and `band7_10` are the two the line-length sweep
# leaves standing; every other candidate grows at some line length between
# 12 and 1200, `row3_7th` from N = 18 up.
const MEMBERS = [
    ("cascade3",      member(2//1, 1//4, 1//4, 1//3, 1//3)),
    ("reuse_c6",      member(0//1, 3//5, 3//10, 1//3, 1//3)),
    ("band16_25",     member(0//1, 16//25, 9//50, 1//3, 1//3)),
    ("band33_50",     member(0//1, 33//50, 7//50, 1//3, 1//3)),
    ("band7_10",      member(0//1, 7//10, 1//25, 1//3, 1//3)),
    ("b3_5_c1_4",     member(0//1, 3//5, 1//4, 1//3, 1//3)),
    ("row3_7th",      member(0//1, 3//5, 3//10, 1//4, 1//2)),
    ("b3_4_c1_4",     member(0//1, 3//4, 1//4, 1//5, 3//5)),
    ("b4_5_c1_10",    member(0//1, 4//5, 1//10, 1//5, 3//5)),
    ("b3_4_c1_10",    member(0//1, 3//4, 1//10, 1//4, 1//2)),
    ("b5_9_c1_10",    member(0//1, 5//9, 1//10, 2//5, 1//5)),
]

named_member(name) = MEMBERS[findfirst(p -> p[1] == name, MEMBERS)][2]

"The members a part runs over: `members=` if given, otherwise `default`."
function selected_members(default)
    isempty(OPTS.members) && return default
    names = split(OPTS.members, ',')
    for name in names
        findfirst(p -> p[1] == name, MEMBERS) === nothing &&
            error("unknown member '$name'; want " * join(first.(MEMBERS), ", "))
    end
    Tuple(names)
end

member_string(m) = sprintf("(%s, %s, %s) + (%s, %s)", m.a, m.b, m.c, m.d, m.e)

# --- part: family -----------------------------------------------------------

function family_part()
    println("\n=== the three-row family ===")
    println("rows at the cascade point (a, b, c) = (2, 1/4, 1/4), (d, e) = (1/3, 1/3)")
    for (j, (sub, super)) in enumerate(((0//1, 2//1), (1//4, 1//4), (1//3, 1//3)))
        printf("  row %d  lhs (%s, 1, %s)  rhs %s\n", j, sub, super,
               row_weights(EXACT, j, sub, super))
    end
    println("rows at the C6 :neutral3 point (a, b, c) = (0, 3/5, 3/10)")
    for (j, (sub, super)) in enumerate(((0//1, 0//1), (3//5, 3//10)))
        printf("  row %d  lhs (%s, 1, %s)  rhs %s\n", j, sub, super,
               row_weights(EXACT, j, sub, super))
    end
    println("\nmonomial residuals (exact rationals) at the cascade point: rows 1 and 2 " *
            "are exact")
    println("through their order and not beyond, row 3 one degree further, since the " *
            "cascade's")
    println("(1/3, 1/3) is the five-point sixth-order row this six-point family " *
            "contains")
    for (j, (sub, super)) in enumerate(((0//1, 2//1), (1//4, 1//4), (1//3, 1//3)))
        order = ROW_POINTS[j] - 1
        through = maximum(abs(monomial_residual(j, sub, super, p)) for p in 0:order)
        printf("  row %d  degrees 0:%d residual %s   degree %d residual %s\n",
               j, order, through, order + 1, monomial_residual(j, sub, super, order + 1))
    end
    println("\nrow 3 carries eight coefficients against seven sixth-order conditions, " *
            "so the")
    println("sixth-order members are a line. Its degree-6 residual is affine in (d, e):")
    r0 = monomial_residual(3, 0//1, 0//1, 6)
    rd = monomial_residual(3, 1//1, 0//1, 6) - r0
    re = monomial_residual(3, 0//1, 1//1, 6) - r0
    printf("  residual(d, e) = %s + (%s) d + (%s) e, which vanishes iff 2d + e = 1\n",
           r0, rd, re)
    for (d, e) in ((1//3, 1//3), (1//4, 1//2), (1//5, 3//5), (2//5, 1//5))
        printf("  (d, e) = (%s, %s): degree-6 residual %s  degree-7 residual %s\n",
               d, e, monomial_residual(3, d, e, 6), monomial_residual(3, d, e, 7))
    end
    println("  (1/4, 1/2) annihilates degree 7 as well: a seventh-order row 3, the only")
    println("  one on the line, with weights ", row_weights(EXACT, 3, 1//4, 1//2))
    println("\nthe rows of the members that survive the line-length sweep")
    for name in GATE_MEMBERS
        m = named_member(name)
        println("  ", name, "  ", member_string(m))
        for (j, (sub, super)) in enumerate(((0//1, m.a), (m.b, m.c), (m.d, m.e)))
            printf("    row %d  lhs (%s, 1, %s)  rhs %s\n", j, sub, super,
                   row_weights(EXACT, j, sub, super))
        end
    end
    println("\nthe assembled closed line against the production plan, N = 41")
    production = first(derivative_matrix(CL.lele_d1_8(closures=:cascade3), 41))
    printf("  C8 :cascade3 from the family vs the production rows: %.3e\n",
           maximum(abs, first(derivative_matrix(scheme_of(named_member("cascade3")), 41)) -
                        production))
    c6 = first(derivative_matrix(CL.lele_d1_6(), 41))
    c6family = CL.CompactScheme{Float64}("C6 from the family", 1/3, 0.0, [7/9, 1/36],
        false, closure_rows(Float64, named_member("reuse_c6"))[1:2])
    printf("  C6 :neutral3 from the family vs lele_d1_6(): %.3e\n",
           maximum(abs, first(derivative_matrix(c6family, 41)) - c6))
end

# --- part: scan -------------------------------------------------------------

"Members of a (b, c) grid neutral at every `Ns`, as (b, c) pairs."
function neutral_grid(a, d, e; grid=OPTS.grid, span=(-1.0, 1.5), Ns=(51, 101))
    hits = NTuple{2,Float64}[]
    for b in range(span...; length=grid), c in range(span...; length=grid)
        scheme = scheme_of(member(a, b, c, d, e))
        ok = true
        for N in Ns
            growth = try acoustic_growth(scheme, N) catch; NaN end
            (isfinite(growth) && growth < NEUTRAL_TOL) || (ok = false; break)
        end
        ok && push!(hits, (b, c))
    end
    hits
end

function scan_part()
    println("\n=== the neutral set ===")
    printf("a %d x %d grid over b, c in [-1, 1.5], neutral at N = 51 and 101, " *
           "row 3 on the sixth-order line\n", OPTS.grid, OPTS.grid)
    println("      d:      0      1/5      1/4      1/3     9/20")
    for a in (0//1, 1//2, 1//1, 3//2, 2//1, 5//2)
        counts = [length(neutral_grid(Float64(a), Float64(d), 1 - 2Float64(d)))
                  for d in (0//1, 1//5, 1//4, 1//3, 9//20)]
        printf("  a = %-5s %s\n", a, join((sprintf("%7d", n) for n in counts), "  "))
        flush(stdout)
    end
    println("\nthe band at a = 0: the c extent at each occupied b")
    for (d, e) in ((1//5, 3//5), (1//4, 1//2), (1//3, 1//3))
        hits = neutral_grid(0.0, Float64(d), Float64(e))
        printf("  (d, e) = (%s, %s): %d neutral\n", d, e, length(hits))
        for b in sort(unique(first.(hits)))
            cs = sort([c for (bb, c) in hits if bb == b])
            printf("    b %+.4f  c %+.3f .. %+.3f  (%d)\n", b, cs[1], cs[end], length(cs))
        end
        flush(stdout)
    end
    println("\nthe fifth-order (d, e) plane at (a, b, c) = (0, 3/4, 1/4), neutral at " *
            "N = 51 and 101")
    hits = NTuple{2,Float64}[]
    for d in range(-0.5, 1.0; length=OPTS.grid), e in range(-0.5, 1.5; length=OPTS.grid)
        scheme = scheme_of(member(0.0, 0.75, 0.25, d, e))
        ok = all(begin
                     g = try acoustic_growth(scheme, N) catch; NaN end
                     isfinite(g) && g < NEUTRAL_TOL
                 end for N in (51, 101))
        ok && push!(hits, (d, e))
    end
    online = count(p -> abs(2p[1] + p[2] - 1) < 1e-9, hits)
    printf("  %d neutral of %d, of which %d on the sixth-order line 2d + e = 1\n",
           length(hits), OPTS.grid^2, online)
    if !isempty(hits)
        printf("  d in [%.3f, %.3f], e in [%.3f, %.3f]\n", extrema(first.(hits))...,
               extrema(last.(hits))...)
    end
end

# --- part: select -----------------------------------------------------------
#
# Neutrality at one line length is not neutrality at all of them, so a pool
# of rational members is filtered in three stages of increasing cost. Twenty
# line lengths are a prefilter and not a verdict: most of what survives all
# three stages still grows somewhere in the sweep part's six hundred, which
# is why the shortlist above is short.

const POOL_B = Rational{Int}[1//2, 5//9, 4//7, 3//5, 5//8, 2//3, 7//10, 3//4, 4//5, 9//10]
const POOL_C = Rational{Int}[-1//5, -1//10, 0//1, 1//20, 1//10, 3//20, 1//5, 1//4,
                             3//10, 1//3, 2//5, 1//2, 3//5, 7//10, 3//4]
const POOL_DE = [(1//3, 1//3), (1//4, 1//2), (3//10, 2//5), (1//5, 3//5), (2//5, 1//5)]
const STAGES = ((51, 101), (31, 71, 151, 203, 251, 301),
                (127, 181, 227, 277, 331, 371, 401, 415, 457, 503, 557, 601))

function select_part()
    println("\n=== the rational pool ===")
    pool = vec([member(0//1, b, c, d, e) for b in POOL_B, c in POOL_C, (d, e) in POOL_DE])
    printf("pool of %d members at a = 0\n", length(pool))
    survivors = pool
    for (k, Ns) in enumerate(STAGES)
        elapsed = @elapsed survivors = filter(survivors) do m
            scheme = scheme_of(m)
            all(acoustic_growth(scheme, N) < neutral_tolerance(N) for N in Ns)
        end
        printf("  stage %d over N = %s: %d survive  (%.1f s)\n", k, join(Ns, ","),
               length(survivors), elapsed)
        flush(stdout)
    end
    println("\nsurvivors by the interior error of one derivative of exp(sin 3x) at N = 97")
    rows = [(derivative_errors(m, 97), m) for m in survivors]
    sort!(rows; by=r -> r[1].interior)
    for (e, m) in rows
        printf("  %-28s wall %.4e  interior %.4e  cond(A) %.3f\n", member_string(m),
               e.wall, e.interior, line_condition(m, 51))
    end
end

# --- part: errors -----------------------------------------------------------

derivative_errors(m::NamedTuple, N) = derivative_errors(scheme_of(m), N)
derivative_errors(scheme, N) =
    closed_derivative_errors(N, scheme, x -> exp(sin(3x)), x -> 3cos(3x) * exp(sin(3x)))

line_condition(m::NamedTuple, N) = line_condition(scheme_of(m), N)
line_condition(scheme, N) = cond(last(derivative_matrix(scheme, N)))

function errors_part()
    println("\n=== one derivative of exp(sin 3x) on the closed line, Float64 ===")
    println("the wall window is the first and last four nodes; the interior is the rest")
    println("  member                        N     wall        interior    cond(A)")
    entries = [(name, scheme_of(m)) for (name, m) in MEMBERS]
    push!(entries, ("C8 :brady_livescu", CL.lele_d1_8(closures=:brady_livescu)))
    push!(entries, ("C6 :neutral3", CL.lele_d1_6()))
    push!(entries, ("C6 :cascade3", CL.lele_d1_6(closures=:cascade3)))
    for (name, scheme) in entries
        for N in (49, 97, 193)
            e = derivative_errors(scheme, N)
            printf("  %-26s %4d   %.4e  %.4e  %7.3f\n", name, N, e.wall, e.interior,
                   line_condition(scheme, N))
        end
        flush(stdout)
    end
end

# --- part: sweep ------------------------------------------------------------

function parse_Ns(spec)
    out = Int[]
    for piece in split(spec, ',')
        parts = parse.(Int, split(piece, ':'))
        append!(out, length(parts) == 1 ? (parts[1]:parts[1]) :
                     length(parts) == 2 ? (parts[1]:parts[2]) :
                     (parts[1]:parts[2]:parts[3]))
    end
    unique!(sort!(out))
end

function sweep_part()
    Ns = parse_Ns(OPTS.Ns)
    printf("\n=== growth over %d line lengths from %d to %d ===\n", length(Ns),
           first(Ns), last(Ns))
    println("a member neutral at two resolutions can still grow at a third")
    for name in selected_members(Tuple(n for (n, _) in MEMBERS if n != "cascade3"))
        m = named_member(name)
        scheme = scheme_of(m)
        worst = -Inf; worst_N = 0; failures = Int[]
        elapsed = @elapsed for N in Ns
            g = acoustic_growth(scheme, N)
            g > worst && (worst = g; worst_N = N)
            g > GROWTH_REJECT && push!(failures, N)
        end
        printf("  %-12s %-28s max %+.3e at N = %d  failures %s  (%.0f s)\n", name,
               member_string(m), worst, worst_N,
               isempty(failures) ? "none" :
                   join(failures[1:min(12, end)], ",") *
                   (length(failures) > 12 ? ",..." : ""), elapsed)
        flush(stdout)
    end
end

# --- part: jacobian ---------------------------------------------------------
#
# The production step map differenced centrally about the uniform state, from
# bench/constantfloor.jl: the amplification matrix of the five conserved
# components on N nodes, its largest eigenvalue modulus as a growth rate per
# unit time, the count outside the unit circle, and the leading eigenvector's
# share within four nodes of a wall. The artificial properties are off, since
# their sensors are not differentiable at a uniform state. A filtered row
# applies the unrelaxed pass after the step.

function step_map(solver, Q0, dt, filter_on)
    Q = copy(Q0)
    dQ = zero(Q); du = zero(Q)
    solver.t = 0.0
    CL.step!(solver, Q, dQ, du, dt)
    filter_on && filter_state!(solver, Q)
    return Q
end

function jacobian_row(label, deriv; N=51, filter_on=false, cl=:onesided, mu=0.0,
                      wall=:slip, alphaf=0.45, delta=1e-5, ladder=false)
    st = (rho=0.9, v=0.1, p=1.1, R=1.0, gamma=1.4)
    per = (PeriodicBC(), PeriodicBC())
    h = 1.0 / (N - 1)
    bc = wall === :noslip ? NoSlipWallBC() :
         wall === :dirichlet ?
             DirichletBC((x, y, z, t) -> Prim(rho=st.rho, u=(0.0, st.v, 0.0), p=st.p)) :
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

# The two members that survive the line-length sweep: `reuse_c6` is the one
# selected and `band7_10` its runner-up.
const GATE_MEMBERS = ("reuse_c6", "band7_10")

function jacobian_part()
    println("\n=== one linearized step at the uniform state, cfl 0.5, artificial " *
            "properties off ===")
    println("the gate is |λ|max <= 1 + 1e-8 on every row")
    for name in selected_members(GATE_MEMBERS)
        deriv = scheme_of(named_member(name))
        println("--- ", name, " ", member_string(named_member(name)))
        jacobian_row("slip, unfiltered", deriv; ladder=true)
        jacobian_row("slip, onesided filter (unrelaxed)", deriv; filter_on=true,
                     ladder=true)
        jacobian_row("slip, unfiltered, N = 101", deriv; N=101)
        jacobian_row("slip, onesided filter, N = 101", deriv; N=101, filter_on=true)
        jacobian_row("no-slip, mu = 0.005, onesided filter", deriv; mu=0.005,
                     wall=:noslip, filter_on=true)
        jacobian_row("no-slip, mu = 0.005, onesided, N = 101", deriv; N=101, mu=0.005,
                     wall=:noslip, filter_on=true)
        jacobian_row("Dirichlet ends, unfiltered", deriv; wall=:dirichlet)
        jacobian_row("Dirichlet ends, onesided filter", deriv; wall=:dirichlet,
                     filter_on=true)
    end
    println("--- C8 :cascade3, the set this one replaced as the default")
    c8c = CL.lele_d1_8(closures=:cascade3)
    jacobian_row("slip, unfiltered", c8c)
    jacobian_row("slip, onesided filter (unrelaxed)", c8c; filter_on=true)
    jacobian_row("slip, unfiltered, N = 101", c8c; N=101)
    jacobian_row("no-slip, mu = 0.005, onesided filter", c8c; mu=0.005,
                 wall=:noslip, filter_on=true)
    # A viscous no-slip wall reads 1 + 1e-8 to 2e-7 under every closure set,
    # the cascade included, so the C6 default is carried here as the scale
    # against which the no-slip rows above are read.
    println("--- C6 :neutral3, the accepted set at the C6 interior")
    jacobian_row("slip, unfiltered", CL.lele_d1_6())
    jacobian_row("no-slip, mu = 0.005, onesided filter", CL.lele_d1_6(); mu=0.005,
                 wall=:noslip, filter_on=true)
end

# --- part: uniform ----------------------------------------------------------
#
# The time-domain reading: a uniform state between slip walls holds its
# round-off seed under a neutral closure and grows it under the cascade.

function uniform_solver(N, deriv; cfl=0.5)
    per = (PeriodicBC(), PeriodicBC())
    h = 1.0 / (N - 1)
    solver = Solver(; n_global=(N, 1, 1), L_domain=(1.0, h, h),
                    bcs=((SlipWallBC(), SlipWallBC()), per, per),
                    eos=IdealSpecies("gas"; R=1.0, gamma=1.4),
                    transport=Transport(mu0=0.0), art=ArtParams(enabled=true),
                    deriv=deriv, filt=compact_filter(0.45), cfl=cfl,
                    filter_interval=1, filter_cfl=0.35,
                    control=StepControl(validity=:permissive))
    Q = allocate_state(solver)
    initialize!(solver, Q, (x, y, z) -> Prim(rho=0.9, u=(0.0, 0.1, 0.0), p=1.1))
    return solver, Q
end

function uniform_row(label, deriv, N)
    solver, Q = uniform_solver(N, deriv)
    cols = String[]
    for tfinal in (10.0, 20.0, 30.0, 40.0)
        try
            # A configuration that loses positivity does not stop: the
            # diffusive rate climbs until dt collapses. `nmax` is counted from
            # the run's start, so it caps the whole ladder, not each leg.
            run!(solver, Q; tfinal=tfinal, nmax=60_000)
        catch err
            err isa SolverFailure || rethrow()
            push!(cols, sprintf("%s at t = %.2f", err.reason, err.t))
            break
        end
        CL.exchange_state!(Q, solver.decomp)
        CL.primitives!(solver, Q)
        n = solver.decomp.n_local[1]
        push!(cols, sprintf("%.1e",
            maximum(abs(solver.u[gidx(solver, i, 1, 1)]) for i in 1:n)))
    end
    printf("  N = %3d  %-28s %s\n", N, label, join(cols, "  "))
    flush(stdout)
end

function uniform_part()
    println("\n=== max |u_n| of a uniform state between slip walls, default relaxed " *
            "filter ===")
    println("rho 0.9, tangential 0.1, p 1.1, cfl 0.5, Float64, at t = 10 / 20 / 30 / 40")
    for N in (51, 101)
        for name in selected_members(GATE_MEMBERS)
            uniform_row(name, scheme_of(named_member(name)), N)
        end
        uniform_row("C8 :cascade3", CL.lele_d1_8(closures=:cascade3), N)
    end
end

for part in PARTS
    part == "family" ? family_part() :
    part == "scan" ? scan_part() :
    part == "select" ? select_part() :
    part == "errors" ? errors_part() :
    part == "sweep" ? sweep_part() :
    part == "jacobian" ? jacobian_part() :
    part == "uniform" ? uniform_part() :
    error("unknown part '$part'; want family, scan, select, errors, sweep, jacobian " *
          "or uniform")
end
