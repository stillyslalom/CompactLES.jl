# Stability certificates for the neutral C6 wall closures, beyond the spectrum.
#
#   julia --project=. -t 1 bench/closurecertify.jl
#   julia --project=. -t 1 bench/closurecertify.jl parts=spectrum,transient
#   julia --project=. -t 1 bench/closurecertify.jl parts=pseudo pseudo_ns=25,51,101
#   julia --project=. -t 1 bench/closurecertify.jl parts=resonance scan_lo=340 scan_hi=470
#   julia --project=. -t 1 bench/closurecertify.jl parts=norm norm_ns=21,31,51
#   julia --project=. -t 16 bench/closurecertify.jl wall=folded schemes=c6,c8,c10
#
# A neutral spectrum certifies nothing on its own. The injected slip-wall
# acoustic operator L is non-normal, so a perturbation of size eps can move an
# eigenvalue much further than eps, and a solution can grow by a large factor
# before the neutral spectrum takes over. Measured here, on the exact 2N model
# of `ClosureSearch.acoustic_operator`: the eps-pseudospectral abscissa, the
# Kreiss constant it implies, the eigenvector conditioning, and the transient
# amplification max_t ||exp(tL)||. Three members of the widened closure family
# are compared: the adopted (a,b,c) = (0, 3/5, 3/10), the neighbour
# (1/4, 3/5, 1/5) which loses neutrality at isolated node counts, and the
# cascade (2, 1/4, 1/4) as the unstable control. The resonance part locates the
# mechanism behind the neighbour's isolated node counts, and the norm part
# looks for structure in a numerically found energy norm.
#
# `wall=node`, the default, places the wall on a node and injects the endpoint
# velocities, which is what the closure rows above are for. `wall=folded` places
# it half a cell outside the end node and folds the interior stencil with the
# field's parity, so no closure row enters and `schemes` selects an interior
# instead: `c6`, `c8` and `c10` name the production presets. The parts
# `spectrum`, `pseudo`, `transient` and `resonance` follow the option.
#
# These are research measurements. The script prints tables and asserts nothing.

module ClosureCertify

using CompactLES, LinearAlgebra, Printf
const CL = CompactLES
include(joinpath(@__DIR__, "closuresearch.jl"))
const CS = ClosureSearch

# ---------------------------------------------------------------- closure family

"""Exact rows of the widened C6 closure family at `(a, b, c)`.

Row 1 is `g_1 + a g_2 = sum_{k=1}^{4} w_k f_k`, exact through degree three;
row 2 is `b g_1 + g_2 + c g_3 = sum_{k=1}^{5} w_k f_k`, exact through degree
four. Nodes are `x = 0, 1, ..` at unit spacing, so the derivative of `x^p` at
node `j` is `p (j-1)^(p-1)`, with `0^0 = 1`.
"""
function family_rows(a::Rational, b::Rational, c::Rational)
    V1 = Rational{BigInt}[Rational{BigInt}(k - 1)^p for p in 0:3, k in 1:4]
    r1 = Rational{BigInt}[p == 0 ? 0 : (p == 1 ? 1 : 0) + a * p for p in 0:3]
    V2 = Rational{BigInt}[Rational{BigInt}(k - 1)^p for p in 0:4, k in 1:5]
    r2 = Rational{BigInt}[p == 0 ? 0 :
         b * (p == 1 ? 1 : 0) + p + c * p * Rational{BigInt}(2)^(p - 1) for p in 0:4]
    (V1 \ r1, V2 \ r2)
end

"""A `CompactScheme` carrying the family member `(a, b, c)` on a C6 interior."""
function member_scheme(a::Rational, b::Rational, c::Rational, name)
    w1, w2 = family_rows(a, b, c)
    rows = [CL.ClosureRow{Float64}((0.0, 1.0, Float64(a)), Float64.(w1)),
            CL.ClosureRow{Float64}((Float64(b), 1.0, Float64(c)), Float64.(w2))]
    CL.CompactScheme{Float64}(name, 1/3, 0.0, [7/9, 1/36], false, rows)
end

const MEMBERS = Dict(
    "adopted"   => (0//1, 3//5, 3//10),
    "neighbour" => (1//4, 3//5, 1//5),
    "cascade"   => (2//1, 1//4, 1//4))

const PRESETS = Dict("c6" => lele_d1_6, "c8" => lele_d1_8, "c10" => lele_d1_10)

function scheme_of(name)
    haskey(MEMBERS, name) && return member_scheme(MEMBERS[name]..., name)
    haskey(PRESETS, name) && return PRESETS[name]()
    error("unknown scheme $name; use one of " *
          join(sort([collect(keys(MEMBERS)); collect(keys(PRESETS))]), ", "))
end

"""The acoustic operator of the requested wall placement."""
function wall_operator(scheme, N, wall)
    wall === :folded && return CS.folded_acoustic_operator(scheme, N)
    wall === :node || error("wall must be node or folded")
    # The node-centred model injects closure rows, which only the tridiagonal
    # assembly of `derivative_matrix` carries.
    scheme isa CL.CompactScheme ||
        error("the node-centred model is tridiagonal only; use wall=folded")
    CS.acoustic_operator(scheme, N)
end

# ------------------------------------------------------------------- small tools

_fields(s) = isempty(strip(s)) ? SubString{String}[] : split(s, ',')
parse_ints(s) = parse.(Int, _fields(s))
parse_floats(s) = parse.(Float64, _fields(s))
parse_names(s) = String.(_fields(s))

"""Deterministic power iteration for the spectral norm.

A full SVD costs more than the eigensolve it would annotate at the largest
line lengths here, and only three digits of the norm are ever printed.
"""
function two_norm(A; iters=200, tol=1e-10)
    T = eltype(A) <: Complex ? ComplexF64 : Float64
    n = size(A, 2)
    x = T[1 + (k - 1) / n for k in 1:n]
    x ./= norm(x)
    s = 0.0
    for _ in 1:iters
        z = A' * (A * x)
        snew = norm(z)
        snew == 0 && return 0.0
        x = z ./ snew
        abs(snew - s) <= tol * snew && (s = snew; break)
        s = snew
    end
    sqrt(s)
end

"""Spectral data of the acoustic operator: values, vectors, inverse, norms."""
function spectral(scheme, N, wall=:node)
    L, A = wall_operator(scheme, N, wall)
    E = eigen(L)
    V = E.vectors
    W = inv(V)
    # Bauer--Fike per eigenvalue: kappa_j = ||v_j|| ||w_j|| / |w_j' v_j|, which
    # is ||w_j|| for the unit-norm right vectors LAPACK returns.
    kappa = [norm(V[:, j]) * norm(W[j, :]) for j in 1:size(V, 2)]
    (L=L, A=A, values=E.values, V=V, W=W, kappa=kappa,
     normL=two_norm(L), condV=two_norm(V) * two_norm(W),
     abscissa=maximum(real, E.values))
end

# ------------------------------------------------------------------ part: verify

function part_verify(io)
    println(io, "\n== family construction against the production presets ==")
    println(io, " member       lhs defect     rhs defect")
    presets = (("adopted", lele_d1_6()), ("cascade", lele_d1_6(closures=:cascade3)))
    for (name, preset) in presets
        s = scheme_of(name)
        dl = 0.0
        dr = 0.0
        for j in 1:2
            dl = max(dl, maximum(abs, collect(s.closures[j].lhs) .-
                                      collect(preset.closures[j].lhs)))
            r1 = s.closures[j].rhs
            r2 = preset.closures[j].rhs
            m = max(length(r1), length(r2))
            pad(v) = [k <= length(v) ? v[k] : 0.0 for k in 1:m]
            dr = max(dr, maximum(abs, pad(r1) .- pad(r2)))
        end
        @printf(io, " %-10s   %.3e      %.3e\n", name, dl, dr)
    end
    s = scheme_of("neighbour")
    println(io, "\n neighbour (1/4, 3/5, 1/5) rows")
    for j in 1:2
        @printf(io, "   row %d lhs %s\n     rhs %s\n", j,
                string(s.closures[j].lhs),
                join([@sprintf("%.10f", v) for v in s.closures[j].rhs], " "))
    end
end

# ---------------------------------------------------------------- part: spectrum

function part_spectrum(io, names, ns, wall)
    println(io, "\n== spectrum, eigenvector conditioning and the all-time bound ==")
    println(io, " units: rates per unit time (c = L = 1); ||L||_2 scales as N")
    println(io, " closure      N    ||L||_2   max Re lam     cond(V)   max kappa_j",
                "   || |V| |V^-1| ||")
    for name in names, N in ns
        s = spectral(scheme_of(name), N, wall)
        bound = two_norm(abs.(s.V) * abs.(s.W))
        @printf(io, " %-10s %4d  %9.3e  %+.4e  %10.4f  %12.4f  %14.4f\n",
                name, N, s.normL, s.abscissa, s.condV, maximum(s.kappa), bound)
    end
end

# ------------------------------------------------------------------ part: pseudo

"""Does the vertical line `Re z = x` meet the `eps`-pseudospectrum of `L`?

`eps` is a singular value of `L - (x + iy) I` for some real `y` exactly when
the Hamiltonian matrix below has the eigenvalue `i y`, so one eigensolve
decides the whole line without a `y` grid (Byers 1988). `tol` is measured
against `||L||`, and the classification is exact except within a distance of
the pseudospectrum boundary that the accuracy sub-part prints.
"""
function meets(L, x, eps, tol)
    n = size(L, 1)
    Id = Matrix{Float64}(I, n, n)
    M = zeros(2n, 2n)
    M[1:n, 1:n] .= L .- x .* Id
    M[1:n, n+1:2n] .= (-eps) .* Id
    M[n+1:2n, 1:n] .= eps .* Id
    M[n+1:2n, n+1:2n] .= x .* Id .- transpose(L)
    ev = eigvals(M)
    minimum(z -> abs(real(z)), ev) <= tol
end

"""The eps-pseudospectral abscissa by bisection on the vertical-line test."""
function pseudo_abscissa(L, eps, abscissa, normL; bisect=14, rtol=1e-10, guess=1.0)
    tol = rtol * normL
    lo = abscissa                     # the spectrum lies in every pseudospectrum
    hi = lo + max(guess, 1.0) * eps
    calls = 1
    for _ in 1:40
        meets(L, hi, eps, tol) || break
        hi = lo + 2 * (hi - lo)
        calls += 1
    end
    for _ in 1:bisect
        mid = 0.5 * (lo + hi)
        calls += 1
        meets(L, mid, eps, tol) ? (lo = mid) : (hi = mid)
    end
    (0.5 * (lo + hi), 0.5 * (hi - lo), calls)
end

"""Smallest singular value of `(x + i y) I - L` on a `y` grid anchored on the
eigenvalues, used only to check the Hamiltonian test at small `N`."""
function line_sigma_min(L, x, values; refine=8)
    n = size(L, 1)
    best = Inf
    cand = sort(unique(round.(imag.(values), digits=10)))
    for y in cand
        s = minimum(svdvals((x + im * y) * Matrix{ComplexF64}(I, n, n) .- L))
        best = min(best, s)
    end
    # local refinement around the best anchor
    for y0 in cand
        for d in range(-0.5, 0.5, length=refine)
            y = y0 + d
            s = minimum(svdvals((x + im * y) * Matrix{ComplexF64}(I, n, n) .- L))
            best = min(best, s)
        end
    end
    best
end

function part_pseudo(io, names, ns, big_ns, epsilons, big_epsilons, bisect, rtol,
                     validate, wall)
    println(io, "\n== eps-pseudospectral abscissa and the Kreiss constant ==")
    println(io, " eps is absolute, in the units of L itself (rate per unit time);",
                " divide by ||L||_2")
    println(io, " for the relative reading. alpha_eps is Re z at the rightmost",
                " point of the")
    println(io, " eps-pseudospectrum; alpha_eps/eps is the Kreiss ratio at that",
                " eps, and K(L) is")
    println(io, " its supremum over eps.")
    for (nsel, esel) in ((ns, epsilons), (big_ns, big_epsilons))
        isempty(nsel) && continue
        for name in names, N in nsel
            s = spectral(scheme_of(name), N, wall)
            g = maximum(s.kappa)
            @printf(io, "\n %s  N=%d  ||L||_2 %.4e  max Re lam %+.3e  max kappa_j %.3f\n",
                    name, N, s.normL, s.abscissa, g)
            println(io, "      eps      alpha_eps     alpha_eps - absc   ",
                        "alpha_eps/eps   +/-        calls")
            for e in esel
                a, unc, calls = pseudo_abscissa(s.L, e, s.abscissa, s.normL;
                                                bisect, rtol, guess=g)
                @printf(io, "   %8.1e  %+.6e   %+.6e      %10.4f   %.1e  %5d\n",
                        e, a, a - s.abscissa, a / e, unc, calls)
            end
        end
    end
    validate || return
    vname = first(names)
    @printf(io, "\n-- accuracy of the vertical-line test (N = 25, %s) --\n", vname)
    s = spectral(scheme_of(vname), 25, wall)
    e = 1e-3
    a, _, _ = pseudo_abscissa(s.L, e, s.abscissa, s.normL; bisect=20, rtol,
                              guess=maximum(s.kappa))
    println(io, "   x/alpha_eps    min_y sigma_min((x+iy)I-L)      meets?")
    for f in (0.5, 0.9, 0.99, 1.0, 1.01, 1.1, 2.0)
        x = a * f
        sm = line_sigma_min(s.L, x, s.values)
        @printf(io, "   %10.4f     %.10e            %s\n", f, sm,
                meets(s.L, x, e, rtol * s.normL))
    end
    @printf(io, "   alpha_eps = %.8e at eps = %.1e; sigma_min there = %.8e\n",
            a, e, line_sigma_min(s.L, a, s.values))
end

# --------------------------------------------------------------- part: transient

"""Largest singular value of `V diag(exp(t lam)) V^-1` by warm-started subspace
iteration on the matrix-vector products alone.

The block is carried from the previous time so that the leading singular
subspace is already close; its last column is re-seeded from a fixed vector at
every time, because a warm block alone drifts and silently loses the maximum
where the subspace turns quickly.
"""
function expm_norm!(X, seed, V, W, lam, t; iters=25)
    e = exp.(t .* lam)
    X[:, end] .= seed
    for _ in 1:iters
        Y = V * (e .* (W * X))
        Z = W' * (conj.(e) .* (V' * Y))
        F = qr!(Z)
        X .= Matrix(F.Q)
    end
    opnorm(V * (e .* (W * X)), 2)
end

"""Two-scale time grid: the fast scale of the operator, then the unit scale."""
function time_grid(normL, tmax, coarse_dt)
    fast = collect(range(0.0, 50 / normL, length=301))
    mid = collect(range(0.0, min(2.0, tmax), step=coarse_dt))
    slow = collect(range(0.0, tmax, length=151))
    sort(unique(vcat(fast, mid, slow)))
end

function transient_max(L, values, V, W, normL, tmax, coarse_dt; block=3)
    ts = time_grid(normL, tmax, coarse_dt)
    n = size(V, 1)
    seed = ComplexF64[1 + (k - 1) / n for k in 1:n]
    seed ./= norm(seed)
    X = Matrix(qr!(ComplexF64[cos(pi * j * k / n) for k in 1:n, j in 1:block]).Q)
    best = 0.0
    tbest = 0.0
    bestfast = 0.0
    for t in ts
        g = expm_norm!(X, seed, V, W, values, t)
        if g > best
            best = g
            tbest = t
        end
        t <= 50 / normL && (bestfast = max(bestfast, g))
    end
    (best, tbest, bestfast, length(ts))
end

"""Diagonal quadrature of the (p, u) state, as an energy norm.

The node-centred state carries N pressures and N-2 interior velocities on a
grid of spacing 1/(N-1), whose end nodes take half a cell. A face-centred
mirror holds no node, so every one of the 2N unknowns takes a full cell of
1/N and the norm is the Euclidean one rescaled.
"""
function quadrature_scale(N, wall)
    wall === :folded && return fill(1 / N, 2N)
    h = 1 / (N - 1)
    wp = fill(h, N)
    wp[1] = h / 2
    wp[N] = h / 2
    vcat(wp, fill(h, N - 2))
end

function part_transient(io, names, ns, tmax, coarse_dt, validate, wall)
    println(io, "\n== transient amplification max_t ||exp(tL)|| ==")
    println(io, " Euclidean norm on (p, u); the energy column repeats it in the",
                " cell-measure")
    println(io, " quadrature norm. cond(V) bounds every t when the spectrum is",
                " on the axis.")
    println(io, " closure      N    max_2   at t      max fast   max energy",
                "   cond(V)   || |V||V^-1| ||   samples")
    for name in names, N in ns
        s = spectral(scheme_of(name), N, wall)
        best, tbest, bestfast, nt = transient_max(s.L, s.values, s.V, s.W,
                                                  s.normL, tmax, coarse_dt)
        w = quadrature_scale(N, wall)
        Lw = Diagonal(sqrt.(w)) * s.L * Diagonal(1 ./ sqrt.(w))
        Ew = eigen(Lw)
        Ww = inv(Ew.vectors)
        be, _, _, _ = transient_max(Lw, Ew.values, Ew.vectors, Ww, s.normL,
                                    tmax, coarse_dt)
        bound = two_norm(abs.(s.V) * abs.(s.W))
        @printf(io, " %-10s %4d %10.4g  %8.4f  %9.4f   %10.4g   %8.3f   %13.4f   %6d\n",
                name, N, best, tbest, bestfast, be, s.condV, bound, nt)
    end
    validate || return
    vname = first(names)
    @printf(io, "\n-- subspace iteration against a dense 2-norm (N = 51, %s) --\n",
            vname)
    s = spectral(scheme_of(vname), 51, wall)
    n = size(s.V, 1)
    seed = ComplexF64[1 + (k - 1) / n for k in 1:n]
    seed ./= norm(seed)
    X = Matrix(qr!(ComplexF64[cos(pi * j * k / n) for k in 1:n, j in 1:3]).Q)
    worst = 0.0
    for t in (0.01, 0.025, 0.1, 0.5, 1.0, 2.5, 7.0, 13.0, 20.0)
        a = expm_norm!(X, seed, s.V, s.W, s.values, t)
        b = opnorm(s.V * Diagonal(exp.(t .* s.values)) * s.W, 2)
        worst = max(worst, abs(a - b) / b)
        @printf(io, "   t %6.3f  power %.8f  dense %.8f  rel %.2e\n", t, a, b,
                abs(a - b) / b)
    end
    @printf(io, "   worst relative error %.2e\n", worst)
end

# --------------------------------------------------------------- part: resonance

"Modified wavenumber of the C6 interior row."
kprime(t) = (2 * (7 / 9) * sin(t) + 2 * (1 / 36) * sin(2t)) / (1 + (2 / 3) * cos(t))

function dkprime(t)
    num = 2 * (7 / 9) * sin(t) + 2 * (1 / 36) * sin(2t)
    den = 1 + (2 / 3) * cos(t)
    dnum = 2 * (7 / 9) * cos(t) + 4 * (1 / 36) * cos(2t)
    dden = -(2 / 3) * sin(t)
    (dnum * den - num * dden) / den^2
end

const THETA_PEAK = let
    g = range(1.0, 3.0, length=200001)
    g[argmax(kprime.(g))]
end

"""The two wavenumbers that the C6 interior carries at the same frequency."""
function branch_roots(v)
    f(t) = kprime(t) - v
    function bis(a, b)
        for _ in 1:200
            c = (a + b) / 2
            (f(a) * f(c) <= 0) ? (b = c) : (a = c)
        end
        (a + b) / 2
    end
    (bis(1e-9, THETA_PEAK), bis(THETA_PEAK, pi - 1e-12))
end

"""Effective wavenumber of a mode from the second difference of its pressure."""
function mode_theta(p, N)
    q = 10:N-9
    num = sum(abs2, p[q .+ 1] .- 2 .* p[q] .+ p[q .- 1])
    den = sum(abs2, p[q])
    acos(clamp(1 - sqrt(num / den) / 2, -1, 1))
end

"""Local maxima of the discrete Fourier amplitude of `p` on a theta grid."""
function theta_peaks(p, grid; floor_ratio=0.15)
    N = length(p)
    amp = [abs(sum(p[k + 1] * cis(-t * k) for k in 0:N-1)) for t in grid]
    m = maximum(amp)
    [(grid[k], amp[k] / m) for k in 2:length(grid)-1
     if amp[k] > amp[k-1] && amp[k] > amp[k+1] && amp[k] > floor_ratio * m]
end

function part_resonance(io, names, scan_lo, scan_hi, bubble_lo, bubble_hi,
                        detune_lo, detune_hi, wall, coarse_hi, coarse_step)
    println(io, wall === :folded ?
        "\n== node-count sweep of the folded wall ==" :
        "\n== the neighbour's node-count resonance ==")
    println(io, "\n-- node counts at which the spectrum leaves the axis --")
    scan = collect(scan_lo:scan_hi)
    coarse_hi > scan_hi &&
        append!(scan, (scan_hi + coarse_step):coarse_step:coarse_hi)
    for name in names
        name == "cascade" && continue
        bad = Tuple{Int,Float64}[]
        worst = 0.0
        for N in scan
            L, _ = wall_operator(scheme_of(name), N, wall)
            g = maximum(real, eigvals(L))
            worst = max(worst, g)
            g > 1e-8 && push!(bad, (N, g))
        end
        @printf(io, " %-10s scan %d:%d  max Re over all N %+.3e  unstable N: %s\n",
                name, first(scan), last(scan), worst,
                isempty(bad) ? "none" : join(string.(first.(bad)), ","))
        for (N, g) in bad
            L, _ = wall_operator(scheme_of(name), N, wall)
            ev = eigvals(L)
            k = argmax(real.(ev))
            M = N - 1
            v = abs(imag(ev[k])) / M
            t1, t2 = branch_roots(v)
            @printf(io, "    N=%4d  rate %+.5e  omega %.4f  omega h %.6f  ",
                    N, g, abs(imag(ev[k])), v)
            @printf(io, "theta1 %.6f theta2 %.6f  theta1 M/pi %.4f  theta2 M/pi %.4f\n",
                    t1, t2, t1 * M / pi, t2 * M / pi)
        end
    end
    if wall === :folded
        println(io, "\n (the collision sub-parts below dissect the node-centred",
                    " neighbour and are skipped)")
        return
    end
    println(io, "\n-- the colliding pair through the resonance (neighbour) --")
    println(io, " every mode within 0.4 percent of omega h = 1.3585, its effective")
    println(io, " wavenumber, and the detuning of the pair")
    println(io, " N      omega     theta_eff     Re          detuning")
    for N in bubble_lo:bubble_hi
        L, _ = CS.acoustic_operator(scheme_of("neighbour"), N)
        E = eigen(L)
        M = N - 1
        band = [j for j in eachindex(E.values)
                if imag(E.values[j]) > 0 && abs(imag(E.values[j]) / M - 1.3585) < 0.0055]
        sort!(band, by=j -> imag(E.values[j]))
        th = [mode_theta(E.vectors[1:N, j], N) for j in band]
        om = [imag(E.values[j]) for j in band]
        b1 = om[th .< 2.0]
        b2 = om[th .>= 2.0]
        det = if isempty(b2) || isempty(b1)
            # Either the pair has merged into one mixed mode, which the
            # wavenumber test then reads as a single branch, or the band holds
            # no partner at all; the ladder-separation table below tells them
            # apart on a wider band.
            length(band) == 2 && abs(om[2] - om[1]) < 1e-8 ? 0.0 : NaN
        else
            d = [y - x for x in b1, y in b2]
            d[argmin(abs.(d))]
        end
        for (k, j) in enumerate(band)
            @printf(io, " %4d %10.4f   %8.4f   %+.3e   %s\n", N,
                    om[k], th[k], real(E.values[j]),
                    k == 1 ? @sprintf("%+8.4f", det) : "")
        end
    end
    println(io, "\n-- separation of the two ladders across the resonance window --")
    println(io, " signed distance from the branch-2 mode to the nearest branch-1 mode,")
    println(io, " in the band |omega h - 1.3585| < 0.02")
    println(io, " N      adopted    neighbour")
    for N in detune_lo:detune_hi
        vals = Float64[]
        for name in ("adopted", "neighbour")
            L, _ = CS.acoustic_operator(scheme_of(name), N)
            E = eigen(L)
            M = N - 1
            b1 = Float64[]
            b2 = Float64[]
            for j in eachindex(E.values)
                om = imag(E.values[j])
                (om > 0 && abs(om / M - 1.3585) < 0.02) || continue
                push!(mode_theta(E.vectors[1:N, j], N) < 2.0 ? b1 : b2, om)
            end
            if isempty(b1) || isempty(b2)
                push!(vals, NaN)
            else
                d = [y - x for x in b1, y in b2]
                push!(vals, d[argmin(abs.(d))])
            end
        end
        @printf(io, " %4d  %+9.4f   %+9.4f\n", N, vals[1], vals[2])
    end
    println(io, "\n-- Fourier content of the colliding modes --")
    grid = collect(range(0.02, pi - 0.02, length=1200))
    for N in (370, 371, 372)
        L, _ = CS.acoustic_operator(scheme_of("neighbour"), N)
        E = eigen(L)
        M = N - 1
        band = [j for j in eachindex(E.values)
                if imag(E.values[j]) > 0 && abs(imag(E.values[j]) / M - 1.3585) < 0.004]
        sort!(band, by=j -> imag(E.values[j]))
        for j in band
            p = E.vectors[1:N, j]
            pk = theta_peaks(p, grid)
            wall = sqrt(sum(abs2, p[1:8]) + sum(abs2, p[N-7:N])) / norm(p)
            @printf(io, " N=%d omega %.4f Re %+.3e theta_eff %.4f wall share %.3f peaks %s\n",
                    N, imag(E.values[j]), real(E.values[j]), mode_theta(p, N), wall,
                    join([@sprintf("%.4f(%.2f)", t, w) for (t, w) in pk], " "))
        end
    end
    println(io, "\n-- the recurrence period from the interior dispersion alone --")
    t1, t2 = branch_roots(1.359675)
    r = dkprime(t1) / dkprime(t2)
    coef = (t2 - r * t1) / pi
    @printf(io, " at the resonant frequency omega h = 1.359675: theta1 %.7f theta2 %.7f\n",
            t1, t2)
    @printf(io, " dtheta2/dtheta1 = %.6f  (theta2 - r theta1)/pi = %.6f  theta1/pi = %.6f\n",
            r, coef, t1 / pi)
    println(io, " a resonance recurs at dM where dq = r dm + coef dM is an integer,")
    println(io, " with dm = round(theta1 dM / pi):")
    println(io, "   dM    dm       dq     residual")
    rows = [(dM, round(Int, t1 * dM / pi),
             r * round(Int, t1 * dM / pi) + coef * dM) for dM in 1:70]
    rows = sort(rows, by=x -> abs(x[3] - round(x[3])))
    for (dM, dm, dq) in rows[1:8]
        @printf(io, "  %4d  %4d  %9.4f   %.5f\n", dM, dm, dq, abs(dq - round(dq)))
    end
    println(io, "\n-- wall phases of the two branches --")
    println(io, " closure      N    branch 1 frac(theta M/pi)   branch 2 frac(theta M/pi)")
    for name in names, N in (370, 371, 372)
        L, _ = CS.acoustic_operator(scheme_of(name), N)
        E = eigen(L)
        M = N - 1
        f1 = Float64[]
        f2 = Float64[]
        for j in eachindex(E.values)
            om = imag(E.values[j])
            om > 0 || continue
            v = om / M
            abs(v - 1.3585) < 0.05 || continue
            th = mode_theta(E.vectors[1:N, j], N)
            t1b, t2b = branch_roots(v)
            push!(th < 2.0 ? f1 : f2, mod((th < 2.0 ? t1b : t2b) * M / pi, 1))
        end
        mean(v) = isempty(v) ? NaN : sum(v) / length(v)
        sd(v) = isempty(v) ? NaN : sqrt(sum((v .- mean(v)).^2) / length(v))
        @printf(io, " %-10s %4d       %.4f +/- %.4f (%2d)        %.4f +/- %.4f (%2d)\n",
                name, N, mean(f1), sd(f1), length(f1), mean(f2), sd(f2), length(f2))
    end
end

# -------------------------------------------------------------------- part: norm

"""Mask of the entries of `H D + D' H` that an energy estimate must annihilate.

`H D + D' H` is symmetric, so a support inside the wall columns is a support
inside the wall rows as well: every entry outside the two `w` by `w` corner
blocks and their cross terms has to vanish.
"""
function wall_mask(N, w)
    inwall(k) = k <= w || k > N - w
    Float64[(inwall(i) && inwall(j)) ? 0.0 : 1.0 for i in 1:N, j in 1:N]
end

constraint(H, D, mask) = mask .* (H * D + transpose(D) * H)

"""Exact basis of the symmetric `H` whose `H D + D' H` lives in the corners.

The constraint is linear, so the admissible set is the null space of a matrix
and the singular value decomposition hands it back outright. The least-squares
projection this replaces stalled at a relative residual of 1e-4, which made
every closure look certified.
"""
function kernel_basis(D, N, w)
    inwall(k) = k <= w || k > N - w
    cols = [(i, j) for i in 1:N for j in i:N]
    rows = [(k, l) for k in 1:N for l in k:N if !(inwall(k) && inwall(l))]
    A = zeros(length(rows), length(cols))
    BD = zeros(N, N)
    for (c, (i, j)) in enumerate(cols)
        fill!(BD, 0)
        BD[i, :] .+= view(D, j, :)
        i != j && (BD[j, :] .+= view(D, i, :))
        for (r, (k, l)) in enumerate(rows)
            A[r, c] = BD[k, l] + BD[l, k]
        end
    end
    Q = nullspace(A)
    basis = Matrix{Float64}[]
    for m in 1:size(Q, 2)
        H = zeros(N, N)
        for (c, (i, j)) in enumerate(cols)
            H[i, j] = Q[c, m]
            H[j, i] = Q[c, m]
        end
        push!(basis, H)
    end
    (basis, size(Q, 2))
end

"""Softmin of the eigenvalues and its gradient: a concave surrogate for lam_min."""
function softmin(H, beta)
    E = eigen(Symmetric(H))
    lam = E.values
    m = minimum(lam)
    e = exp.(-beta .* (lam .- m))
    s = sum(e)
    f = m - log(s) / beta
    (f, E.vectors * Diagonal(e ./ s) * transpose(E.vectors), m, maximum(lam))
end

"""Symmetric `H` in the kernel with the largest smallest eigenvalue found.

Ascent on a softmin surrogate over the kernel coordinates; `lam_min` itself is
not differentiable and a plain subgradient step chatters between the lowest
eigenvectors.
"""
function energy_norm(scheme, N, w; steps=200, rounds=4)
    D, _ = CS.derivative_matrix(scheme, N)
    basis, dim = kernel_basis(D, N, w)
    isempty(basis) && return (zeros(N, N), D, -Inf, 1.0, 0)
    G = [sum(Bi .* Bj) for Bi in basis, Bj in basis]
    x = G \ [tr(Bi) for Bi in basis]
    build(y) = begin
        H = sum(y[k] .* basis[k] for k in eachindex(basis))
        H .*= N / tr(H)
        H
    end
    H = build(x)
    beta = 50.0
    lo, hi = 0.0, 1.0
    for _ in 1:rounds
        step = 0.2
        for _ in 1:steps
            f, Gr, lo, hi = softmin(H, beta)
            g = [sum(Bi .* Gr) for Bi in basis]
            moved = false
            for _ in 1:40
                y = x .+ step .* g
                T = build(y)
                if first(softmin(T, beta)) > f
                    x, H = y, T
                    moved = true
                    break
                end
                step /= 2
            end
            moved || break
            step *= 1.5
        end
        beta *= 4
    end
    _, _, lo, hi = softmin(H, beta)
    (H, D, lo, hi, dim)
end

function part_norm(io, names, ns, w, steps)
    println(io, "\n== a symmetric H with (H D + D' H) supported on the wall corners ==")
    @printf(io, " wall depth %d; H normalised to trace N; a positive ratio is a\n", w)
    println(io, " Lyapunov certificate at that node count, a negative one says the",
                " ascent")
    println(io, " found none. lam are eigenvalues of H, not of the operator.")
    println(io, " closure      N   kernel dim   lam_min/lam_max   residual   interior H diag")
    corners = Dict{String,Matrix{Float64}}()
    for name in names, N in ns
        mask = wall_mask(N, w)
        H, D, lo, hi, dim = energy_norm(scheme_of(name), N, w; steps)
        res = sqrt(sum(abs2, constraint(H, D, mask))) /
              (sqrt(sum(abs2, H)) * sqrt(sum(abs2, D)))
        mid = N ÷ 2
        @printf(io, " %-10s %4d   %10d   %+15.6f   %9.2e   %.6f\n",
                name, N, dim, lo / hi, res, H[mid, mid])
        N == maximum(ns) && (corners[name] = copy(H))
    end
    println(io, "\n-- the exact Lyapunov certificate P = Re(V^-H V^-1) --")
    println(io, " P L + L' P vanishes when the spectrum is on the axis, so the",
                " certificate")
    println(io, " itself is never in doubt; the question is whether P has any",
                " structure.")
    println(io, " closure      N    cond(P)   residual   band width at 1e-6",
                "   interior diag spread")
    for name in names, N in ns
        s = spectral(scheme_of(name), N)
        P = real(s.W' * s.W)
        P .*= size(P, 1) / tr(P)
        res = two_norm(P * s.L .+ transpose(s.L) * P) / (two_norm(P) * s.normL)
        ev = eigvals(Symmetric(P))
        n = size(P, 1)
        scale = maximum(abs, P)
        band = maximum(abs(i - j) for i in 1:n, j in 1:n if abs(P[i, j]) > 1e-6 * scale)
        mid = (n ÷ 4):(3 * n ÷ 4)
        d = [P[k, k] for k in mid]
        @printf(io, " %-10s %4d  %9.3e  %9.2e  %19d   %.4f to %.4f\n",
                name, N, maximum(ev) / minimum(ev), res, band, minimum(d), maximum(d))
    end
    println(io, "\n-- leading 6x6 corner, scaled so the interior diagonal is one --")
    for name in names
        haskey(corners, name) || continue
        H = corners[name]
        mid = size(H, 1) ÷ 2
        S = H ./ H[mid, mid]
        println(io, " ", name)
        for i in 1:6
            println(io, "   ", join([@sprintf("%+9.5f", S[i, j]) for j in 1:6], " "))
        end
        println(io, "   interior diagonal near the wall: ",
                join([@sprintf("%.6f", S[k, k]) for k in 5:12], " "))
    end
    println(io, "\n-- does a fixed corner block certify other node counts? --")
    println(io, " closure      depth    N   interior residual   lam_min/lam_max")
    for name in names
        haskey(corners, name) || continue
        H = corners[name]
        n0 = size(H, 1)
        mid = n0 ÷ 2
        S = H ./ H[mid, mid]
        for depth in (6, 8, 10)
            for N in (51, 101, 201, 401)
                N <= 2depth + 4 && continue
                D, _ = CS.derivative_matrix(scheme_of(name), N)
                F = Matrix{Float64}(I, N, N)
                F[1:depth, 1:depth] .= S[1:depth, 1:depth]
                F[N-depth+1:N, N-depth+1:N] .= S[1:depth, 1:depth][end:-1:1, end:-1:1]
                mask = wall_mask(N, w)
                res = sqrt(sum(abs2, constraint(F, D, mask))) /
                      (sqrt(sum(abs2, F)) * sqrt(sum(abs2, D)))
                ev = eigvals(Symmetric(F))
                @printf(io, " %-10s %6d %5d   %17.3e   %+.6f\n",
                        name, depth, N, res, minimum(ev) / maximum(ev))
            end
        end
    end
end

# --------------------------------------------------------------------------- main

function main(args=ARGS)
    opts = CL.script_args(args, (
        parts="verify,spectrum,pseudo,transient,resonance,norm",
        schemes="adopted,neighbour,cascade", blas=12,
        spectrum_ns="25,51,101,201,401,801,370,371,372,415",
        pseudo_ns="25,51,101,201", pseudo_big_ns="371,415",
        eps="1e-2,1e-3,1e-4,1e-6,1e-8", big_eps="1e-3,1e-6",
        bisect=14, rtol=1e-10, validate=true,
        transient_ns="25,51,101,201", tmax=20.0, coarse_dt=0.01,
        scan_lo=340, scan_hi=470, scan_coarse_hi=0, scan_coarse_step=10,
        bubble_lo=366, bubble_hi=376,
        detune_lo=360, detune_hi=420,
        norm_ns="21,31,51", corner=4, ascent=200, wall="node"))
    wall = Symbol(opts.wall)
    wall in (:node, :folded) || error("wall must be node or folded")
    # The package pins BLAS to one thread for its tiny interface solve; this
    # script is dense LAPACK from end to end and wants the machine back.
    BLAS.set_num_threads(opts.blas)
    parts = parse_names(opts.parts)
    names = parse_names(opts.schemes)
    io = stdout
    @printf(io, "closurecertify: threads %d, BLAS %d\n", Threads.nthreads(),
            BLAS.get_num_threads())
    println(io, "parts = ", opts.parts, "; schemes = ", opts.schemes,
            "; wall = ", opts.wall)
    for part in parts
        t = @elapsed begin
            if part == "verify"
                part_verify(io)
            elseif part == "spectrum"
                part_spectrum(io, names, parse_ints(opts.spectrum_ns), wall)
            elseif part == "pseudo"
                part_pseudo(io, names, parse_ints(opts.pseudo_ns),
                            parse_ints(opts.pseudo_big_ns), parse_floats(opts.eps),
                            parse_floats(opts.big_eps), opts.bisect, opts.rtol,
                            opts.validate, wall)
            elseif part == "transient"
                part_transient(io, names, parse_ints(opts.transient_ns), opts.tmax,
                               opts.coarse_dt, opts.validate, wall)
            elseif part == "resonance"
                part_resonance(io, names, opts.scan_lo, opts.scan_hi, opts.bubble_lo,
                               opts.bubble_hi, opts.detune_lo, opts.detune_hi, wall,
                               opts.scan_coarse_hi, opts.scan_coarse_step)
            elseif part == "norm"
                part_norm(io, names, parse_ints(opts.norm_ns), opts.corner, opts.ascent)
            else
                error("unknown part $part")
            end
        end
        @printf(io, "\n[%s took %.1f s]\n", part, t)
        flush(io)
    end
end

end # module

if abspath(PROGRAM_FILE) == @__FILE__
    ClosureCertify.main()
end
