# Search for a symmetric closure of the staggered diffusion operator of
# src/staggered.jl at a fold whose area vanishes oddly (the cylindrical axis,
# the spherical poles), with dense matrices on one folded line in index units:
# nodes at i - 1/2, midpoints at m, the fold at 0 and a wall through node N.
# There the flux r K D_s T is continued smoothly through the plane, and
# L = (1/r) G r D_s is accurate but not symmetric in the node volumes V = r W.
#
# The family searched is every operator symmetric in a diagonal node norm,
#   L = -H⁻¹ Dᵀ M D,   H = V (1 + δw),   M = W_m r (1 + δμ),
# with δw and δμ nonzero on the first k nodes and midpoints and
# D = D_σ + Z P⁻ᵀ W_n, where D_σ is the mirror derivative of the data's
# parity σ, P the tridiagonal left-hand side of the adjoint G, and Z a k × s
# block. G = -H⁻¹ Dᵀ M then keeps the form P⁻¹(Q - Zᵀ(1 + δμ)), explicit edge
# rows ahead of the node solve, so its kink error at the fold is cancelled
# ahead of the solve, with no tail behind it. Such an operator is symmetric
# in H, negative semidefinite and conservative in H by construction; the
# design conditions, exactness of D on the data's monomials up to degree d_D
# and of G on the flux monomials up to degree d_G, are linear in (Z, δμ, δw).
# The first node of L is then of order min(d_D, d_G).
# A diagonal H is the only node norm compatible with the (1/r²) azimuthal and
# (1/sin²θ) polar operators; at a paired fold δw may differ between the even
# and odd combinations (the half-period shift commutes with the periodic
# operator), δμ may not, since it multiplies the coefficient κ.
#
# Printed:
#   check      the standalone smooth-continuation L against the package's
#              StaggeredDiffusion on a cylindrical-axis line (relative, h² L)
#   diagonal   the smallest asymmetry any diagonal norm gives the existing L
#   barrier    the least-squares residual of the design conditions against the
#              closure width k (s = k + 1); zero to round-off where a closure
#              exists
#   closures   exact closures at the reachable orders and constrained ones
#              (exact on the reachable set, least squares beyond it): asymmetry
#              in H, largest real eigenvalue, smallest weight, and the
#              max-norm error of L on exp(-4r²) (even) or r exp(-4r²) (odd)
#              over r < 1/2 at h = 1/16 ... 1/256, with its observed orders
#   defect     the contraction of defect correction x += A_s⁻¹(b - A x),
#              A = I - τL the accurate operator, A_s = I - τL_s a symmetric one,
#              against τ|λ_min|
#
# Usage (about fifteen seconds; single-threaded):
#   julia --project=. -t 1 bench/staggeredclosure.jl [n=200] [widths=6,10,14]

using CompactLES, LinearAlgebra, Printf

const OPT = CompactLES.script_args(ARGS, (n = 200, widths = "6,10,14"))
const N = OPT.n
const α = 9 / 62
const A_TAP = 63 / 62
const B_TAP = 17 / 62 / 3

node_ext(j, σf, σw) = j < 1 ? (1 - j, σf) : j > N ? (2N - j, σw) : (j, 1)
function mid_ext(m, σf, σw)
    m < 0 && return (-m, σf)
    m == 0 && return (0, σf > 0 ? 1 : 0)
    m > N - 1 && return (2N - 1 - m, σw)
    return (m, 1)
end

# Left- and right-hand sides of D (nodes to midpoints 0..N-1, row m + 1) and of
# G (midpoints to nodes), for input parity σf at the fold and σw at the wall.
function sides(op, σf, σw)
    P = zeros(N, N); Q = zeros(N, N)
    for r in 1:N
        if op === :D && r == 1 && σf > 0
            P[1, 1] = 1.0          # the plane midpoint of odd output is zero
            continue
        end
        for (o, c) in ((-1, α), (0, 1.0), (1, α))
            j, s = op === :D ? mid_ext(r - 1 + o, -σf, -σw) : node_ext(r + o, -σf, -σw)
            s == 0 && continue
            P[r, op === :D ? j + 1 : j] += s * c
        end
        taps = ((-1, -B_TAP), (0, -A_TAP), (1, A_TAP), (2, B_TAP))
        for (o, c) in taps
            if op === :D
                j, s = node_ext(r - 1 + o, σf, σw)
                Q[r, j] += s * c
            else
                j, s = mid_ext(r + o - 1, σf, σw)
                s == 0 || (Q[r, j+1] += s * c)
            end
        end
    end
    return P, Q
end

const XN = [i - 0.5 for i in 1:N]
const XM = [Float64(m) for m in 0:N-1]
const WM = [m == 0 ? 0.5 : 1.0 for m in 0:N-1]
const WN = [i == N ? 0.5 : 1.0 for i in 1:N]
# per parity of the data: mirror D, and the adjoint G's P and Q (odd flux for
# even data, even flux for odd data)
const OPS = Dict(
    :even => (D = (x -> x[1] \ x[2])(sides(:D, 1, 1)), PQ = sides(:G, -1, -1)),
    :odd => (D = (x -> x[1] \ x[2])(sides(:D, -1, 1)), PQ = sides(:G, 1, -1)))
const G_SMOOTH = Dict(:even => (x -> x[1] \ x[2])(sides(:G, 1, -1)),
                      :odd => (x -> x[1] \ x[2])(sides(:G, -1, -1)))
smooth_L(par) = Diagonal(1 ./ XN) * G_SMOOTH[par] * Diagonal(XM) * OPS[par].D

# ---------------------------------------------------------------- design --
struct Layout
    k::Int
    s::Int
    pars::Vector{Symbol}
    separate_w::Bool
end
nz(l) = l.k * l.s
ncol(l) = length(l.pars) * nz(l) + l.k + (l.separate_w ? length(l.pars) : 1) * l.k
tdeg(par, d) = par === :even ? (0:2:d) : (1:2:d)
fdeg(par, d) = par === :even ? (2:2:d) : (1:2:d)

# Rows of the conditions on parity index q: D exact on T = r^p for p ≤ dD and
# G exact on F = r^p for p ≤ dG, each row scaled to unit maximum.
function conditions(l::Layout, q, dD, dG)
    par = l.pars[q]
    P, Q = OPS[par].PQ
    k, s = l.k, l.s
    zi(m, j) = (q - 1) * nz(l) + (j - 1) * k + m
    mo = length(l.pars) * nz(l)
    wo = mo + k + (l.separate_w ? (q - 1) * k : 0)
    rows = Vector{Vector{Float64}}(); rhs = Float64[]
    for p in tdeg(par, dD)
        v = transpose(P) \ (WN .* XN .^ p)
        for m in 1:k
            a = zeros(ncol(l))
            for j in 1:s
                a[zi(m, j)] = v[j]
            end
            push!(rows, a); push!(rhs, 0.0)
        end
    end
    for p in fdeg(par, dG)
        F = WM .* XM .^ p
        dF = p .* XN .^ (p - 1)
        r0 = Q * F - P * dF
        for i in 1:(s+k+4)
            a = zeros(ncol(l))
            for m in 1:k
                i <= s && (a[zi(m, i)] -= F[m+1])
                a[mo+m] += Q[i, m+1] * F[m+1]
            end
            for j in 1:k
                a[wo+j] -= P[i, j] * dF[j]
            end
            push!(rows, a); push!(rhs, -r0[i])
        end
    end
    A = Matrix(reduce(hcat, rows)')
    scale = [maximum(abs, A[i, :]) for i in axes(A, 1)]
    keep = scale .> 0
    return A[keep, :] ./ scale[keep], rhs[keep] ./ scale[keep]
end

function stacked(l::Layout, degrees)
    parts = [conditions(l, q, degrees[q]...) for q in eachindex(l.pars)]
    return reduce(vcat, first.(parts)), reduce(vcat, last.(parts))
end

function lsq(A, b)
    F = svd(A)
    r = count(>(1e-12 * F.S[1]), F.S)
    x = F.V[:, 1:r] * ((F.U[:, 1:r]' * b) ./ F.S[1:r])
    return x, norm(A * x - b)
end

# Exact on `exact`, least squares on `extra` within that solution set.
function design(l::Layout, exact, extra=nothing)
    Ae, be = stacked(l, exact)
    x0, res = lsq(Ae, be)
    extra === nothing && return x0, res
    F = svd(Ae; full=true)
    r = count(>(1e-12 * F.S[1]), F.S)
    Z = F.V[:, r+1:end]
    Ax, bx = stacked(l, extra)
    reg = 1e-10
    y = [Ax * Z; sqrt(reg) * Z] \ [bx - Ax * x0; -sqrt(reg) * x0]
    return x0 + Z * y, norm(Ae * (x0 + Z * y) - be)
end

function operators(l::Layout, x)
    k = l.k
    mo = length(l.pars) * nz(l)
    μ = ones(N); μ[2:k+1] .+= x[mo+1:mo+k]
    map(eachindex(l.pars)) do q
        wo = mo + k + (l.separate_w ? (q - 1) * k : 0)
        w = ones(N); w[1:k] .+= x[wo+1:wo+k]
        par = l.pars[q]
        P, _ = OPS[par].PQ
        Z = zeros(N, N)
        Z[2:k+1, 1:l.s] = reshape(x[(q-1)*nz(l)+1:q*nz(l)], k, l.s) ./ μ[2:k+1]
        D = OPS[par].D + Z * (transpose(P) \ Matrix(Diagonal(WN)))
        H = Diagonal(WN .* XN .* w)
        (L = -(H \ (D' * Diagonal(WM .* XM .* μ) * D)), H = H, w = w, μ = μ)
    end
end

# ----------------------------------------------------------- measurement --
g(r) = exp(-4r^2)
dg(r) = -8r * g(r)
d2g(r) = (64r^2 - 8) * g(r)
const FIELDS = Dict(
    :even => (g, r -> d2g(r) + dg(r) / r),
    :odd => (r -> r * g(r), r -> 2dg(r) + r * d2g(r) + (g(r) + r * dg(r)) / r))
const HS = (1 / 16, 1 / 32, 1 / 64, 1 / 128, 1 / 256)

function errors(L, par)
    f, Lf = FIELDS[par]
    e = map(HS) do h
        r = XN .* h
        n = count(<(0.5), r)
        maximum(abs, ((L * f.(r)) ./ h^2 .- Lf.(r))[1:n])
    end
    return e, [log2(e[i] / e[i+1]) for i in 1:length(e)-1]
end

row(v, f) = join((Printf.format(Printf.Format(f), x) for x in v), " ")

function show(label, par, L, H)
    S = H * L
    e, p = errors(L, par)
    @printf("%-30s %-4s %8.1e %9.1e %6.3f  %s  %s\n", label, par, norm(S - S') / norm(S),
            maximum(real, eigvals(L)), minimum(diag(H) ./ (XN .* WN)), row(e, "%8.1e"),
            row(p, "%5.2f"))
end

function package_check()
    solver = Solver(n_global=(N, 1, 1), L_domain=(1.0, 1.0, 1.0),
                    metric=CylindricalMetric(),
                    bcs=((AxisBC(), SlipWallBC()), (PeriodicBC(), PeriodicBC()),
                         (PeriodicBC(), PeriodicBC())),
                    art=ArtificialProperties(enabled=false))
    decomp = solver.decomp
    pad = decomp.n_halo_d
    at(i) = CartesianIndex(pad[1] + i, pad[2] + 1, pad[3] + 1)
    op = CompactLES.StaggeredDiffusion(solver, 1)
    κ = CompactLES.field(decomp); κ .= 1
    M = zeros(N, N)
    for j in 1:N
        u = CompactLES.field(decomp); u[at(j)] = 1
        out = CompactLES.field(decomp)
        CompactLES.staggered_diffusion!(out, op, u, copy(κ), decomp)
        M[:, j] = [out[at(i)] for i in 1:N]
    end
    Ls = smooth_L(:even)
    return norm(solver.h[1]^2 * M - Ls) / norm(Ls)
end

function main()
    widths = parse.(Int, split(OPT.widths, ","))
    @printf("check    standalone against StaggeredDiffusion, cylindrical axis: %.1e\n",
            package_check())

    # A diagonal norm for the existing operator: the smallest singular vector
    # of w ↦ antisymmetric part of diag(w) L over the fold end.
    L = smooth_L(:even)[1:40, 1:40]
    A = reduce(hcat, [(a = zeros(40); a[i] = L[i, j]; a[j] = -L[j, i]; a)
                      for i in 1:40 for j in i+1:40])'
    F = svd(A)
    w = F.V[:, end]
    asym(S) = norm(S - S') / norm(S)
    @printf("diagonal best asymmetry %.1e (with V %.1e); smallest singular value %.1e\n\n",
            asym(Diagonal(w) * L), asym(Diagonal(XN[1:40]) * L), F.S[end])

    println("barrier  parity d_D d_G   residual at k = ", join(widths, ", "))
    for (par, dD, dG) in ((:even, 4, 4), (:even, 6, 4), (:even, 4, 6), (:even, 6, 6),
                          (:odd, 5, 3), (:odd, 3, 5), (:odd, 7, 3), (:odd, 5, 5))
        res = [design(Layout(k, k + 1, [par], false), [(dD, dG)])[2] for k in widths]
        @printf("         %-5s  %2d  %2d   %s\n", par, dD, dG, row(res, "%8.1e"))
    end
    for (label, sep) in (("pair, shared δw", false), ("pair, separate δw", true))
        res = [design(Layout(k, k + 1, [:even, :odd], sep), [(4, 4), (3, 3)])[2]
               for k in widths]
        @printf("         %-17s even 4/4 odd 3/3   %s\n", label, row(res, "%8.1e"))
    end

    println("\nclosures                       par      asym  max Re λ  min w  ",
            "error at h = 1/16 ... 1/256 and orders")
    for par in (:even, :odd)
        show("smooth continuation (current)", par, smooth_L(par), Diagonal(XN .* WN))
    end
    l = Layout(4, 5, [:even], false)
    x, _ = design(l, [(4, 4)])
    ops = operators(l, x)
    show("exact order 4, k = 4", :even, ops[1].L, ops[1].H)
    S = ops[1].H * ops[1].L
    @printf("    weights 1+δw %s  1+δμ %s\n", row(ops[1].w[1:4], "%.4f"),
            row(ops[1].μ[2:5], "%.4f"))
    @printf("    conservation |1ᵀ H L T| / |H L T|₁ %.1e; symmetric part max eigenvalue %.1e\n",
            (v = S * g.(XN ./ 32); abs(sum(v)) / sum(abs, v)),
            maximum(eigvals(Symmetric((S + S') / 2))) / norm(S))
    l = Layout(6, 7, [:even, :odd], true)
    x, _ = design(l, [(4, 4), (3, 3)])
    ops = operators(l, x)
    show("pair exact order 4/3, k = 6", :even, ops[1].L, ops[1].H)
    show("", :odd, ops[2].L, ops[2].H)
    for k in widths
        l = Layout(k, k + 1, [:even], false)
        x, _ = design(l, [(4, 4)], [(6, 6)])
        ops = operators(l, x)
        show("axis constrained, k = $k", :even, ops[1].L, ops[1].H)
    end
    for k in widths
        l = Layout(k, k + 1, [:even, :odd], true)
        x, _ = design(l, [(4, 4), (3, 3)], [(6, 6), (5, 5)])
        ops = operators(l, x)
        show("pair constrained, k = $k", :even, ops[1].L, ops[1].H)
        show("", :odd, ops[2].L, ops[2].H)
    end

    τs = [1e-2, 1e0, 1e2, 1e4, 1e6]
    λ = maximum(abs, eigvals(smooth_L(:even)))
    println("\ndefect   A_s                  τ|λ_min| = ", row(τs, "%8.0e"))
    le = Layout(4, 5, [:even], false)
    lo = Layout(4, 5, [:odd], false)
    adjoint(par) = Diagonal(1 ./ XN) * G_SMOOTH[par === :even ? :odd : :even] *
                   Diagonal(XM) * OPS[par].D
    for (label, par, Ls) in (
            ("adjoint mirror", :even, adjoint(:even)),
            ("exact closure order 4", :even, operators(le, design(le, [(4, 4)])[1])[1].L),
            ("adjoint mirror", :odd, adjoint(:odd)),
            ("exact closure order 3", :odd, operators(lo, design(lo, [(3, 3)])[1])[1].L))
        L = smooth_L(par)
        c = [maximum(abs, eigvals(I - (I - τ / λ * Ls) \ (I - τ / λ * L))) for τ in τs]
        @printf("         %-22s %-4s        %s\n", label, par, row(c, "%8.1e"))
    end
end

main()
