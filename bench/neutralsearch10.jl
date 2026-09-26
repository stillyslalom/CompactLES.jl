# A neutral boundary closure set for the C10 pentadiagonal first derivative.
#
#   julia --project=. -t 1 bench/neutralsearch10.jl [parts=all]
#   julia --project=. -t 1 bench/neutralsearch10.jl parts=sweep Ns=long rows=1,7
#   julia --project=. -t 1 bench/neutralsearch10.jl parts=scan,wide grid=fine
#
# `lele_d1_10(closures = :cascade3)` closes a wall with the C6 cascade rows
# padded with zeros at ±2. Under them a uniform inviscid state between slip
# walls grows a wall-normal velocity, the mode the C6 `:neutral3` rows were
# built to remove. The same construction is carried out here for the
# pentadiagonal interior: the three closure rows are widened into a family at
# fixed order, and the family is searched for members whose injected
# slip-wall acoustic operator has a purely imaginary spectrum at every line
# length.
#
# The family, with the row's left-hand side on the left and its right-hand
# side reading from the edge node:
#
#   row 1   g_1 + a g_2 + a2 g_3                  = Σ_{k=1}^{4} w_k f_k
#   row 2   b g_1 + g_2 + c g_3 + c2 g_4          = Σ_{k=1}^{5} w_k f_k
#   row 3   dm2 g_1 + d g_2 + g_3 + e g_4 + e2 g_5 = Σ_{k=1}^{7} w_k f_k
#
# A row with M right-hand-side points and its left-hand side held fixed has
# M weights and M moment conditions, so it is exact through degree M − 1:
# third, fourth and sixth order. The cascade rows are the members
# (a, b, c, d, e) = (2, 1/4, 1/4, 1/3, 1/3) with the widening coefficients
# zero, at which row 2's weights on f_4, f_5 and row 3's on f_6, f_7 vanish
# and the rows reduce to the three- and five-point ones in `kernels_banded.jl`;
# (a, b, c) = (0, 3/5, 3/10) are the C6 `:neutral3` rows.
#
# Parts:
#
#   validate  the closed-line derivative matrix assembled from the rows
#             against the production plan's fill and solve, the family's
#             reproduction of the cascade and of the C6 neutral rows in
#             exact rationals, each row's order on monomials, and the
#             growth the C10 cascade rows carry
#   scan      the neutral set with row 3 held at the C6 interior row: the
#             count over a at coarse resolution, then the (b, c) band at
#             a = 0 pruned over a range of line lengths
#   wide      the same with row 3's left-hand side free, and a seeded
#             random search over all nine parameters, ranked by the
#             interior error constant among the members that pass a broad
#             line-length sweep
#   errors    one derivative of four smooth fields on the closed line at
#             N = 49 / 97 / 193 / 385, wall window and interior, for the
#             finalists beside the C10 cascade rows and C6 `:neutral3`,
#             with the closed line's condition number
#   sweep     the line-length sweep of the finalists; `Ns=long` runs every
#             N from 12 to 600 and every tenth to 1200
#   jacobian  the production step map linearized about the uniform state
#             between two walls, slip, no-slip and Dirichlet, filtered and
#             unfiltered, at N = 51 and 101
#   uniform   the wall-normal velocity a uniform state grows between slip
#             walls under the default relaxed filter, to t = 40
#
# Scratch tooling, like everything else in bench/: it prints tables and
# asserts nothing.

using MPI
MPI.Init(threadlevel=:funneled)
using CompactLES
using CompactLES: apply_bcs!, filter_state!, padded_index
using LinearAlgebra
using Printf
using Random

const CL = CompactLES
MPI.Comm_size(MPI.COMM_WORLD) == 1 || error("run this study on one rank")

const OPTS = CL.script_args(ARGS, (parts="all", Ns="12:200", grid="coarse",
                                   seed=20260918, draws=40000, rows="all"))
const PARTS = OPTS.parts == "all" ?
    ["validate", "scan", "wide", "errors", "sweep", "jacobian", "uniform"] :
    split(OPTS.parts, ',')

sprintf(fmt::String, args...) = Printf.format(Printf.Format(fmt), args...)
printf(fmt::String, args...) = print(sprintf(fmt, args...))

# --- the closure family -------------------------------------------------------
#
# One compact row at node `node`, with left-hand-side coefficients `values`
# on the nodes `node .+ offsets` and a right-hand side on nodes 1..npoints.
# Exactness on f(x) = x^p at unit spacing reads
#
#   Σ_k w_k (k−1)^p = p Σ_s c_s (node − 1 + s)^{p−1},
#
# the p = 0 line being Σ_k w_k = 0. Taking p = 0 .. npoints−1 determines the
# weights; the type parameter carries exact rationals when the parameters
# are rational and Float64 when they are not.

function row_weights(::Type{R}, node::Int, offsets, values, npoints::Int) where {R}
    vandermonde = R[R(k - 1)^p for p in 0:npoints-1, k in 1:npoints]
    target = zeros(R, npoints)
    for p in 1:npoints-1
        moment = zero(R)
        for (offset, coefficient) in zip(offsets, values)
            x = R(node - 1 + offset)
            moment += R(coefficient) * (p == 1 ? one(R) : x^(p - 1))
        end
        target[p+1] = R(p) * moment
    end
    vandermonde \ target
end

# The first degree at which the row is no longer exact, less one.
function row_order(node::Int, offsets, values, npoints::Int)
    R = Rational{BigInt}
    w = row_weights(R, node, offsets, R.(values), npoints)
    for p in npoints:20
        moment = zero(R)
        for (offset, coefficient) in zip(offsets, values)
            x = R(node - 1 + offset)
            moment += R(coefficient) * (p == 1 ? one(R) : x^(p - 1))
        end
        sum(w[k] * R(k - 1)^p for k in 1:npoints) != R(p) * moment && return p - 1
    end
    return 20
end

const ROW_NODES = (1, 2, 3)
const ROW_OFFSETS = ((0, 1, 2), (-1, 0, 1, 2), (-2, -1, 0, 1, 2))
const ROW_POINTS = (4, 5, 7)

# Every row's weights are affine in its left-hand-side coefficients, so the
# search evaluates them as a stored constant vector plus one basis vector
# per free coefficient rather than by solving a system per candidate.
function affine_basis(row::Int)
    offsets = ROW_OFFSETS[row]
    diagonal = findfirst(==(0), offsets)
    values = zeros(length(offsets)); values[diagonal] = 1
    constant = row_weights(Float64, ROW_NODES[row], offsets, values, ROW_POINTS[row])
    basis = Vector{Float64}[]
    for i in eachindex(offsets)
        i == diagonal && continue
        v = copy(values); v[i] = 1
        push!(basis, row_weights(Float64, ROW_NODES[row], offsets, v,
                                 ROW_POINTS[row]) .- constant)
    end
    (constant, basis)
end

const BASIS = (affine_basis(1), affine_basis(2), affine_basis(3))

"""The nine family coordinates, the four widening ones defaulting to zero."""
family(a, b, c, d, e; a2=0.0, c2=0.0, dm2=0.0, e2=0.0) =
    (Float64(a), Float64(a2), Float64(b), Float64(c), Float64(c2),
     Float64(dm2), Float64(d), Float64(e), Float64(e2))

const CASCADE = family(2, 1//4, 1//4, 1//3, 1//3)
const BandRow = CL.BandedClosureRow{Float64}

function trial_scheme(parameters)
    a, a2, b, c, c2, dm2, d, e, e2 = parameters
    w1 = BASIS[1][1] .+ a .* BASIS[1][2][1] .+ a2 .* BASIS[1][2][2]
    w2 = BASIS[2][1] .+ b .* BASIS[2][2][1] .+ c .* BASIS[2][2][2] .+
         c2 .* BASIS[2][2][3]
    w3 = BASIS[3][1] .+ dm2 .* BASIS[3][2][1] .+ d .* BASIS[3][2][2] .+
         e .* BASIS[3][2][3] .+ e2 .* BASIS[3][2][4]
    CL.BandedCompactScheme{Float64}("C10 trial closure", 2, [1/2, 1/20], 0.0,
        [17/24, 101/600, 1/600], false,
        [BandRow([0.0, 0.0, 1.0, a, a2], w1),
         BandRow([0.0, b, 1.0, c, c2], w2),
         BandRow([dm2, d, 1.0, e, e2], w3)])
end

# --- the closed-line operator -------------------------------------------------
#
# The banded counterpart of the assembly in `bench/closuresearch.jl`: the
# interior rows of `kernels_banded.jl`, the closure rows' full centered
# left-hand-side band with the diagonal at index q+1, and the high-side rows
# mirrored, the left-hand side reversed and the right-hand side reversed and
# negated. Validated against the production plan in the `validate` part.

function banded_derivative_matrix(scheme::CL.BandedCompactScheme, N::Integer)
    q = scheme.q
    nclosure = length(scheme.closures)
    N >= 2nclosure + 1 || throw(ArgumentError("N must be at least $(2nclosure + 1)"))
    T = eltype(scheme.coeffs)
    A = Matrix{T}(I, N, N)
    B = zeros(T, N, N)
    for j in nclosure+1:N-nclosure
        for s in 1:q
            A[j, j-s] = scheme.lhs[s]
            A[j, j+s] = scheme.lhs[s]
        end
        B[j, j] = scheme.a0
        for m in eachindex(scheme.coeffs)
            B[j, j+m] = scheme.coeffs[m]
            B[j, j-m] = -scheme.coeffs[m]
        end
    end
    for j in 1:nclosure
        row = scheme.closures[j]
        hi = N + 1 - j
        A[j, j] = zero(T)
        A[hi, hi] = zero(T)
        for s in 1:2q+1
            lo_col = j + (s - q - 1)
            1 <= lo_col <= N && (A[j, lo_col] += row.lhs[s])
            hi_col = hi - (s - q - 1)
            1 <= hi_col <= N && (A[hi, hi_col] += row.lhs[s])
        end
        len = length(row.rhs)
        B[j, row.first:row.first+len-1] .= row.rhs
        B[hi, N+1-row.first:-1:N+2-row.first-len] .= -row.rhs
    end
    return A \ B, A
end

"""The same operator through the production plan's fill and banded solve."""
function production_matrix(scheme, N::Integer)
    decomp = CL.Decomp{Float64}((N, 1, 1), (false, true, true))
    plan = CL.plan_direction(decomp, scheme, 1, 1.0)
    P = zeros(N, N)
    nlo = length(plan.clo)
    nhi = length(plan.chi)
    for k in 1:N
        f = zeros(N); f[k] = 1
        b = zeros(N)
        for j in 1:nlo
            first = plan.clo_first[j]
            b[j] = sum(plan.clo[j][i] * f[first+i-1] for i in eachindex(plan.clo[j]))
        end
        for j in 1:nhi
            first = N + 2 - plan.chi_first[j]
            b[N+1-j] = sum(plan.chi[j][i] * f[first-i] for i in eachindex(plan.chi[j]))
        end
        for j in nlo+1:N-nhi
            b[j] = plan.a0 * f[j]
            for m in eachindex(plan.ci)
                b[j] += plan.ci[m] * (scheme.symmetric ? f[j+m] + f[j-m] :
                                                         f[j+m] - f[j-m])
            end
        end
        work = reshape(b, N, 1)
        CL.solve_lines!(work, plan.line_solver)
        P[:, k] .= vec(work)
    end
    P
end

# The injected slip-wall acoustic operator on (p on N nodes, u on the N−2
# interior nodes), unit domain, rates in c/L units. The two prescribed
# endpoint velocities are eliminated rather than zeroed, which would leave
# defective zero modes.
function acoustic_operator(scheme, N::Integer)
    D, A = banded_derivative_matrix(scheme, N)
    D .*= N - 1
    L = [zeros(N, N) -D[:, 2:N-1]; -D[2:N-1, :] zeros(N - 2, N - 2)]
    return L, A
end

growth_rate(parameters, N) =
    maximum(real, eigvals(first(acoustic_operator(trial_scheme(parameters), N))))

function safe_growth(parameters, N)
    try
        growth_rate(parameters, N)
    catch
        NaN
    end
end

neutral(parameters, Ns; tol=1e-12) =
    all(((g = safe_growth(parameters, N)); !isnan(g) && g < tol) for N in Ns)

# --- accuracy -----------------------------------------------------------------
#
# One derivative of a smooth field on the closed line at its own spacing,
# read off the assembled operator, which the `validate` part shows equals
# the production one. The wall window is the first and last `W` nodes, the
# convention of `test/smooth_cases.jl`.

const FIELDS = (("exp(sin 3x)", x -> exp(sin(3x)), x -> 3cos(3x) * exp(sin(3x))),
                ("sin(5x + 0.7)", x -> sin(5x + 0.7), x -> 5cos(5x + 0.7)),
                ("1/(1.3 + x)", x -> 1 / (1.3 + x), x -> -1 / (1.3 + x)^2),
                ("exp(2x) cos 4x", x -> exp(2x) * cos(4x),
                 x -> exp(2x) * (2cos(4x) - 4sin(4x))))

function derivative_errors(parameters, N, field=FIELDS[1]; W=4)
    D, _ = banded_derivative_matrix(trial_scheme(parameters), N)
    x = range(0, 1; length=N)
    e = abs.((D * field[2].(x)) .* (N - 1) .- field[3].(x))
    (wall=maximum(e[[1:W; N-W+1:N]]), interior=maximum(e[W+1:N-W]))
end

line_condition(parameters, N) = cond(last(banded_derivative_matrix(
    trial_scheme(parameters), N)))

# --- the finalists ------------------------------------------------------------

const SELECTED = family(0, 3//5, 3//10, 1//3, 1//3)
const RUNNERS = (("(0, 3/5, 3/10)", SELECTED),
                 ("(0, 16/25, 9/50)", family(0, 16//25, 9//50, 1//3, 1//3)),
                 ("(0, 7/10, 1/25)", family(0, 7//10, 1//25, 1//3, 1//3)),
                 ("(0, 11/20, 1/2)", family(0, 11//20, 1//2, 1//3, 1//3)),
                 ("(0, 3/5, 0)", family(0, 3//5, 0, 1//3, 1//3)),
                 ("(0, 3/4, 3/4, 1/8, 1/4)", family(0, 3//4, 3//4, 1//8, 1//4)),
                 ("(0, 9/10, 3/5, 1/8, 1/4)", family(0, 9//10, 3//5, 1//8, 1//4)),
                 ("(0, 19/20, 2/5, 1/8, 1/4)", family(0, 19//20, 2//5, 1//8, 1//4)))

# --- part: validate -----------------------------------------------------------

function validate_part()
    println("\n=== operator assembly against the production plan ===")
    println("  scheme                          N     max |assembled − production|")
    for (label, scheme) in (("C10 cascade rows",
                             CL.lele_d1_10(closures=:cascade3)),
                            ("family at the cascade point", trial_scheme(CASCADE)),
                            ("family at the selected point", trial_scheme(SELECTED)))
        for N in (51, 101)
            mismatch = maximum(abs, first(banded_derivative_matrix(scheme, N)) -
                                    production_matrix(scheme, N))
            printf("  %-30s%5d   %.3e\n", label, N, mismatch)
        end
    end
    println("\n  family against the cascade rows of kernels_banded.jl, N = 51: ",
            sprintf("%.3e", maximum(abs,
                first(banded_derivative_matrix(trial_scheme(CASCADE), 51)) -
                first(banded_derivative_matrix(
                    CL.lele_d1_10(closures=:cascade3), 51)))))

    println("\n=== the family in exact rationals ===")
    R = Rational{BigInt}
    cases = (("row 1, a = 2 (cascade)", 1, (0, 1, 2), R[1, 2, 0]),
             ("row 1, a = 0 (selected)", 1, (0, 1, 2), R[1, 0, 0]),
             ("row 2, (b, c) = (1/4, 1/4)", 2, (-1, 0, 1, 2), R[1//4, 1, 1//4, 0]),
             ("row 2, (b, c) = (3/5, 3/10)", 2, (-1, 0, 1, 2), R[3//5, 1, 3//10, 0]),
             ("row 3, (d, e) = (1/3, 1/3)", 3, (-2, -1, 0, 1, 2),
              R[0, 1//3, 1, 1//3, 0]),
             ("row 3, (d, e) = (1/8, 1/4)", 3, (-2, -1, 0, 1, 2),
              R[0, 1//8, 1, 1//4, 0]))
    for (label, row, offsets, values) in cases
        w = row_weights(R, ROW_NODES[row], offsets, values, ROW_POINTS[row])
        printf("  %-30s order %d   weights %s\n", label,
               row_order(ROW_NODES[row], offsets, values, ROW_POINTS[row]),
               join(string.(w), ", "))
    end

    println("\n=== the growth the injected slip-wall operator carries ===")
    println("  rows                            N     max Re(lambda)   cond(A)")
    for (label, parameters) in (("C10 cascade", CASCADE), ("C10 selected", SELECTED))
        for N in (51, 101)
            L, A = acoustic_operator(trial_scheme(parameters), N)
            printf("  %-30s%5d   %+.6e   %8.3f\n", label, N,
                   maximum(real, eigvals(L)), cond(A))
        end
    end
end

# --- part: scan ---------------------------------------------------------------
#
# Row 3 held at the C6 interior row, which is what the cascade set already
# carries there. Neutrality is read on the 2N acoustic operator; the scan
# admits a point at 1e-12, and the growth of the members that fail is never
# between 1e-12 and 1e-8, so the threshold is not a knob.

function scan_part()
    fine = OPTS.grid == "fine"
    n = fine ? 81 : 41
    prune = (51, 31, 79, 101, 151)
    println("\n=== the neutral set over a, row 3 at the C6 interior row, " *
            "$(n)x$(n) in (b, c) over [-1, 1.5]^2 ===")
    println("  a       neutral at N = 51   surviving $(length(prune)) lengths   " *
            "b range           c range")
    bs = range(-1.0, 1.5; length=n)
    cs = range(-1.0, 1.5; length=n)
    for a in 0.0:0.5:6.0
        keep = Tuple{Float64,Float64}[]
        for b in bs, c in cs
            g = safe_growth(family(a, b, c, 1//3, 1//3), 51)
            !isnan(g) && g < 1e-12 && push!(keep, (b, c))
        end
        survive = filter(bc -> neutral(family(a, bc[1], bc[2], 1//3, 1//3),
                                       prune[2:end]), keep)
        brange = isempty(survive) ? "-" :
            sprintf("[%.3f, %.3f]", extrema(first.(survive))...)
        crange = isempty(survive) ? "-" :
            sprintf("[%.3f, %.3f]", extrema(last.(survive))...)
        printf("  %.2f    %5d of %5d        %5d                %-18s%s\n",
               a, length(keep), n * n, length(survive), brange, crange)
        flush(stdout)
    end

    nb = fine ? 130 : 65
    println("\n=== the band at a = 0: c range of the survivors, per b ===")
    bs = range(-1.0, 1.5; length=nb)
    cs = range(-1.0, 1.5; length=nb)
    band = Tuple{Float64,Float64}[]
    for b in bs, c in cs
        g = safe_growth(family(0, b, c, 1//3, 1//3), 51)
        !isnan(g) && g < 1e-12 && neutral(family(0, b, c, 1//3, 1//3),
                                          (31, 79, 101, 151)) && push!(band, (b, c))
    end
    printf("  %d of %d grid points neutral over five lengths\n", length(band), nb * nb)
    for b in sort(unique(first.(band)))
        v = sort([c for (bb, c) in band if bb == b])
        printf("    b = %.4f   %2d points   c in [%.4f, %.4f]\n",
               b, length(v), minimum(v), maximum(v))
    end
end

# --- part: wide ---------------------------------------------------------------
#
# Freeing row 3's left-hand side lowers the interior error constant by two
# orders of magnitude at N = 51 and 101, and the members that do it fail at
# most other line lengths, so the sweep and not the pair of resolutions is
# the filter here. The stages below run the cheap small lengths first.

const SWEEP_STAGES = ((51,), (16, 19, 24, 32, 37, 42, 48, 55, 64, 73),
                      12:60, 61:2:199)

function passes_stages(parameters)
    for stage in SWEEP_STAGES
        neutral(parameters, stage; tol=1e-10) || return false
    end
    true
end

function wide_part()
    fine = OPTS.grid == "fine"
    bs = fine ? (0.50:0.025:1.05) : (0.50:0.05:1.05)
    cs = fine ? (-0.10:0.05:0.90) : (-0.10:0.10:0.90)
    ds = fine ? (0.00:0.0125:0.35) : (0.00:0.025:0.35)
    es = fine ? (0.00:0.025:0.60) : (0.00:0.05:0.60)
    println("\n=== row 3's left-hand side free: $(length(bs))x$(length(cs))x" *
            "$(length(ds))x$(length(es)) over (b, c, d, e) ===")
    grid = [(b, c, d, e) for b in bs, c in cs, d in ds, e in es]
    ranked = sort(vec([(derivative_errors(family(0, p...), 97).interior, p)
                       for p in grid]), by=first)
    printf("  %d points; the ten lowest interior error constants at N = 97:\n",
           length(ranked))
    for (err, p) in ranked[1:10]
        printf("    b = %.4f c = %.4f d = %.4f e = %.4f   interior %.4e   %s\n",
               p..., err, passes_stages(family(0, p...)) ? "passes" :
               sprintf("fails at N = %d", first_failure(family(0, p...))))
    end
    survivors = Tuple{Float64,NTuple{4,Float64}}[]
    examined = 0
    for (err, p) in ranked
        (length(survivors) >= 8 || examined >= 4000) && break
        examined += 1
        passes_stages(family(0, p...)) && push!(survivors, (err, p))
    end
    printf("\n  the first %d members that pass every stage, out of the %d lowest:\n",
           length(survivors), examined)
    for (err, p) in survivors
        printf("    b = %.4f c = %.4f d = %.4f e = %.4f   interior %.4e\n", p..., err)
    end
    printf("  the selected member's interior error constant at N = 97: %.4e\n",
           derivative_errors(SELECTED, 97).interior)

    rng = MersenneTwister(OPTS.seed)
    println("\n=== a seeded random search over all nine coordinates ===")
    radii = (0.6, 0.4, 0.5, 0.5, 0.3, 0.3, 0.5, 0.5, 0.3)
    center = SELECTED
    best = Tuple{Float64,NTuple{9,Float64}}[]
    hits = 0
    for _ in 1:OPTS.draws
        parameters = ntuple(i -> center[i] + radii[i] * (2rand(rng) - 1), 9)
        safe_growth(parameters, 51) < 1e-12 || continue
        hits += 1
        passes_stages(parameters) || continue
        push!(best, (derivative_errors(parameters, 97).interior, parameters))
    end
    sort!(best, by=first)
    printf("  %d draws, %d neutral at N = 51, %d passing every stage\n",
           OPTS.draws, hits, length(best))
    for (err, parameters) in best[1:min(5, end)]
        printf("    interior %.4e   %s\n", err,
               join((sprintf("%.4f", v) for v in parameters), " "))
    end
end

function first_failure(parameters)
    for stage in SWEEP_STAGES, N in stage
        g = safe_growth(parameters, N)
        (isnan(g) || g > 1e-10) && return N
    end
    0
end

# --- part: errors -------------------------------------------------------------

function errors_part()
    Ns = (49, 97, 193, 385)
    rows = (("C10 cascade rows", CASCADE), RUNNERS...)
    for field in FIELDS
        println("\n=== one derivative of $(field[1]), wall window (W = 4) then " *
                "interior ===")
        printf("  %-26s%s\n", "rows",
               join((sprintf("%10d", N) for N in Ns), "  "))
        for (label, parameters) in rows
            w = Float64[]; i = Float64[]
            for N in Ns
                e = derivative_errors(parameters, N, field)
                push!(w, e.wall); push!(i, e.interior)
            end
            printf("  %-26s%s   wall  ord %s\n", label,
                   join((sprintf("%10.3e", v) for v in w), "  "),
                   join((sprintf("%.2f", log2(w[k] / w[k+1])) for k in 1:length(w)-1),
                        " "))
            printf("  %-26s%s   int   ord %s\n", "",
                   join((sprintf("%10.3e", v) for v in i), "  "),
                   join((sprintf("%.2f", log2(i[k] / i[k+1])) for k in 1:length(i)-1),
                        " "))
        end
    end
    println("\n=== the wall window at W = 6, exp(sin 3x), and cond(A) at N = 51 ===")
    printf("  %-26s%s   cond(A)\n", "rows",
           join((sprintf("%10d", N) for N in Ns), "  "))
    for (label, parameters) in rows
        w = [derivative_errors(parameters, N; W=6).wall for N in Ns]
        printf("  %-26s%s   %7.3f\n", label,
               join((sprintf("%10.3e", v) for v in w), "  "),
               line_condition(parameters, 51))
    end
    println("\n  C6 rows on the same field for reference, W = 4:")
    for (label, scheme) in (("C6 neutral3", CL.lele_d1_6()),
                            ("C6 cascade3", CL.lele_d1_6(closures=:cascade3)))
        w = Float64[]; i = Float64[]
        for N in Ns
            D, _ = tridiagonal_derivative_matrix(scheme, N)
            x = range(0, 1; length=N)
            e = abs.((D * FIELDS[1][2].(x)) .* (N - 1) .- FIELDS[1][3].(x))
            push!(w, maximum(e[[1:4; N-3:N]])); push!(i, maximum(e[5:N-4]))
        end
        printf("  %-26s%s   wall\n", label,
               join((sprintf("%10.3e", v) for v in w), "  "))
        printf("  %-26s%s   int\n", "",
               join((sprintf("%10.3e", v) for v in i), "  "))
    end
end

# The tridiagonal assembly, for the C6 reference rows only.
function tridiagonal_derivative_matrix(scheme::CL.CompactScheme, N::Integer)
    nr = length(scheme.closures)
    T = eltype(scheme.coeffs)
    A = Matrix{T}(I, N, N)
    B = zeros(T, N, N)
    for j in 1:nr
        row = scheme.closures[j]
        hi = N + 1 - j
        A[j, j] = row.lhs[2]
        j > 1 && (A[j, j-1] = row.lhs[1])
        j < N && (A[j, j+1] = row.lhs[3])
        B[j, row.first:row.first+length(row.rhs)-1] .= row.rhs
        A[hi, hi] = row.lhs[2]
        hi < N && (A[hi, hi+1] = row.lhs[1])
        hi > 1 && (A[hi, hi-1] = row.lhs[3])
        B[hi, N+1-row.first:-1:N+2-row.first-length(row.rhs)] .= -row.rhs
    end
    for j in nr+1:N-nr
        A[j, j-1] = scheme.alpha
        A[j, j+1] = scheme.alpha
        for m in eachindex(scheme.coeffs)
            B[j, j+m] = scheme.coeffs[m]
            B[j, j-m] = -scheme.coeffs[m]
        end
    end
    A \ B, A
end

# --- part: sweep --------------------------------------------------------------

function parse_lengths(spec)
    spec == "long" && return [12:600; 610:10:1200]
    out = Int[]
    for token in split(spec, ',')
        pieces = parse.(Int, split(token, ':'))
        length(pieces) == 1 && (push!(out, pieces[1]); continue)
        length(pieces) == 2 && (append!(out, pieces[1]:pieces[2]); continue)
        append!(out, pieces[1]:pieces[2]:pieces[3])
    end
    out
end

function sweep_part()
    Ns = parse_lengths(OPTS.Ns)
    printf("\n=== line-length sweep over %d lengths, %d to %d ===\n",
           length(Ns), minimum(Ns), maximum(Ns))
    println("  rows                        worst growth   at N    above 1e-10   " *
            "max cond(A)   seconds")
    chosen = OPTS.rows == "all" ? eachindex(RUNNERS) :
             parse.(Int, split(OPTS.rows, ','))
    for (label, parameters) in RUNNERS[chosen]
        t0 = time()
        worst = -Inf; where = 0; bad = Int[]; worst_cond = 0.0
        for N in Ns
            L, A = acoustic_operator(trial_scheme(parameters), N)
            g = maximum(real, eigvals(L))
            g > worst && (worst = g; where = N)
            g > 1e-10 && push!(bad, N)
            # cond is a singular-value decomposition and would outweigh the
            # eigensolve on the long lines; it is flat in N past a hundred.
            N <= 201 && (worst_cond = max(worst_cond, cond(A)))
        end
        printf("  %-26s%+.3e     %4d   %5d         %8.3f      %6.1f\n",
               label, worst, where, length(bad), worst_cond, time() - t0)
        isempty(bad) || printf("      first lengths above 1e-10: %s\n",
                               join(bad[1:min(16, end)], ", "))
        flush(stdout)
    end
end

# --- part: jacobian -----------------------------------------------------------
#
# The production step map linearized about the uniform state by central
# differences, copied from `bench/constantfloor.jl` so that the readings are
# comparable line for line: the amplification matrix of the five conserved
# components on N nodes and its largest eigenvalue modulus. The artificial
# properties are off, their sensors not being differentiable at a uniform
# state. A filtered row applies the unrelaxed pass after the step.

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
         wall === :dirichlet ? DirichletBC((x, y, z, t) -> Prim(rho=st.rho,
                                                                u=(0.0, st.v, 0.0),
                                                                p=st.p)) :
         SlipWallBC()
    solver = Solver(; n_global=(N, 1, 1), L_domain=(1.0, h, h), bcs=((bc, bc), per, per),
                    eos=IdealSpecies("gas"; R=st.R, gamma=st.gamma),
                    transport=ConstantTransport(mu0=mu),
                    art=ArtificialProperties(enabled=false),
                    deriv=deriv, filt=compact_filter(alphaf; closures=cl),
                    filter_interval=filter_on ? 1 : 0, filter_cfl=0.0, cfl=0.5)
    Q0 = allocate_state(solver)
    initialize!(solver, Q0, (x, y, z) -> Prim(rho=st.rho, u=(0.0, st.v, 0.0), p=st.p))
    apply_bcs!(solver, Q0)
    c = sqrt(st.gamma * st.p / st.rho)
    dt = 0.5 * h / c
    ncons = solver.equations.n_cons
    step_map(solver, Q0, dt, filter_on)
    idx = [(padded_index(solver, i, 1, 1), comp) for comp in 1:ncons for i in 1:N]
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
    for i in eachindex(idx)
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
    println("\n=== one linearized step at the uniform state, cfl 0.5, " *
            "artificial properties off ===")
    println("rate = ln|λ|max / dt per unit time; growing = eigenvalues outside " *
            "the unit circle")
    cases = (("C10 cascade", CL.lele_d1_10(closures=:cascade3)),
             ("C10 selected", trial_scheme(SELECTED)),
             ("C10 runner-up (0, 16/25, 9/50)",
              trial_scheme(family(0, 16//25, 9//50, 1//3, 1//3))),
             ("C10 widened (0, 9/10, 3/5, 1/8, 1/4)",
              trial_scheme(family(0, 9//10, 3//5, 1//8, 1//4))))
    for (label, deriv) in cases
        jacobian_row("$label, slip, unfiltered", deriv; ladder=true)
        jacobian_row("$label, slip, onesided filter", deriv; filter_on=true,
                     ladder=true)
        jacobian_row("$label, slip, unfiltered, N = 101", deriv; N=101)
        jacobian_row("$label, slip, onesided filter, N = 101", deriv; N=101,
                     filter_on=true)
        jacobian_row("$label, no-slip mu = 0.005, onesided", deriv; mu=0.005,
                     wall=:noslip, filter_on=true)
        jacobian_row("$label, Dirichlet ends, unfiltered", deriv; wall=:dirichlet)
        jacobian_row("$label, Dirichlet ends, onesided filter", deriv;
                     wall=:dirichlet, filter_on=true)
    end
end

# --- part: uniform ------------------------------------------------------------

function uniform_solver(N, deriv; rho, v, p, R, gamma, cfl=0.5)
    per = (PeriodicBC(), PeriodicBC())
    h = 1.0 / (N - 1)
    solver = Solver(; n_global=(N, 1, 1), L_domain=(1.0, h, h),
                    bcs=((SlipWallBC(), SlipWallBC()), per, per),
                    eos=IdealSpecies("gas"; R=R, gamma=gamma),
                    transport=ConstantTransport(mu0=0.0),
                    art=ArtificialProperties(enabled=true),
                    deriv=deriv, filt=compact_filter(0.45), cfl=cfl,
                    filter_interval=1, filter_cfl=0.35,
                    control=StepControl(validity=:permissive))
    Q = allocate_state(solver)
    initialize!(solver, Q, (x, y, z) -> Prim(rho=rho, u=(0.0, v, 0.0), p=p))
    return solver, Q
end

function uniform_part()
    println("\n=== a uniform state between slip walls, rho 0.9, tangential 0.1, " *
            "p 1.1 ===")
    println("the default relaxed filter every step, cfl 0.5: max |u_n| at " *
            "t = 10 / 20 / 30 / 40")
    cases = (("C10 cascade", CL.lele_d1_10(closures=:cascade3)),
             ("C10 selected", trial_scheme(SELECTED)),
             ("C10 runner-up (0, 16/25, 9/50)",
              trial_scheme(family(0, 16//25, 9//50, 1//3, 1//3))),
             ("C10 widened (0, 9/10, 3/5, 1/8, 1/4)",
              trial_scheme(family(0, 9//10, 3//5, 1//8, 1//4))))
    for N in (51, 101), (label, deriv) in cases
        solver, Q = uniform_solver(N, deriv; rho=0.9, v=0.1, p=1.1, R=1.0, gamma=1.4)
        readings = String[]
        for tfinal in (10.0, 20.0, 30.0, 40.0)
            failure = ""
            try
                run!(solver, Q; tfinal=tfinal, nmax=1_000_000)
            catch err
                err isa SolverFailure || rethrow()
                failure = sprintf("FAILED %s at t = %.2f", err.reason, err.t)
            end
            if isempty(failure)
                CL.exchange_state!(Q, solver.decomp)
                CL.primitives!(solver, Q)
                n = solver.decomp.n_local[1]
                push!(readings, sprintf("%.1e",
                    maximum(abs(solver.u[padded_index(solver, i, 1, 1)]) for i in 1:n)))
            else
                push!(readings, failure)
                break
            end
        end
        printf("  N = %3d  %-32s%s\n", N, label, join(readings, "  "))
        flush(stdout)
    end
end

for part in PARTS
    part == "validate" ? validate_part() :
    part == "scan" ? scan_part() :
    part == "wide" ? wide_part() :
    part == "errors" ? errors_part() :
    part == "sweep" ? sweep_part() :
    part == "jacobian" ? jacobian_part() :
    part == "uniform" ? uniform_part() :
    error("unknown part '$part'; want validate, scan, wide, errors, sweep, " *
          "jacobian or uniform")
end
