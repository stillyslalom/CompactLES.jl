# Operator and integrator choices for implicit diffusion (ROADMAP H1/H2),
# measured on 1-D model problems with dense matrices, periodic but for the
# wall part. reference/IMPLICIT.md is the design this informs.
#
# Part `spectra`: the implicit system (I - τL)x = b for a variable
# conductivity κ = 1 + 1e3·exp(-((x - 1/2)/0.05)²), solved by conjugate
# gradients preconditioned with P2 = I - τL2(κ), the second-order conservative
# three-point operator that a multigrid cycle would invert in 3-D. L is one of
#   wide       D κ D, the C6 first derivative applied twice: the form the
#              explicit right-hand side uses for the molecular fluxes
#   staggered  -Dsᵀ κ_mid Ds, the C6 staggered derivative (Lele 1992, α = 9/62)
#              from nodes to midpoints and back, κ_mid by sixth-order explicit
#              interpolation: conservative, symmetric, and nonzero at Nyquist
# It prints the truncation error of each on a manufactured field and the CG
# iteration count against the step in units of the explicit diffusive limit.
# The wide form's symbol vanishes at the grid Nyquist mode, so P2 there is
# not spectrally equivalent to it, and its count grows with N and the step.
#
# Part `imex`: T_t + u T_x = (κ0 T^{5/2} T_x)_x, u = 1, cfl 0.5 advective steps,
# the diffusion on the staggered operator. Compares
#   ARK        ARK4(3)6L[2]SA (Kennedy & Carpenter 2003, the SUNDIALS tables):
#              explicit advection, ESDIRK diffusion, each implicit stage a
#              Picard iteration on κ with a P2-preconditioned CG inner solve
#   STS        Strang splitting, RKL2 super-time-stepping (Meyer, Balsara and
#              Aslam 2014) for half steps of diffusion around RK4 advection
# against an ARK run at `refine`× the steps, with the operator applications
# per step (for ARK also the preconditioner solves, each a V-cycle in 3-D).
# R is the step over the forward-Euler diffusive limit at the peak T.
# The profile relaxes toward uniform within t = 0.25 once κ0 ≥ 10, so the
# stiff rows measure cost, not accuracy.
#
# Part `walls`: the staggered operator L = G K D_s on a closed line x ∈ [0, 1]
# with nodes on the walls, under two wall treatments:
#   rows    one-sided explicit closure rows on K points: D_s at the first
#           midpoint, G at the first two nodes with the wall flux as data,
#           and the interpolation at the first two midpoints (sixth order)
#   mirror  the node-centred mirror of src/staggered.jl: T even through the
#           wall node (the adiabatic wall), midpoint data odd, which is the
#           periodic operator on the doubled line folded back
# It prints the max-norm truncation error of L, and of D_s and G for the rows;
# the relative asymmetry of W_n L with the trapezoidal W_n; the conservation
# defect |Σ W_n L T - (F(1) - F(0))|; and the largest real part of an
# eigenvalue of the zero-flux operator. The rows run on exp(sin 3x) with
# κ = 1 + sin(2x)/2; the mirror on a field and coefficient of its own parity,
# then on T'(0) = 0 with T'''(0) ≠ 0, and on κ'(0) ≠ 0: the price of the
# mirror when the data are not of its parity.
#
# Usage (seconds for spectra and walls, a few minutes for imex;
# single-threaded):
#   julia --project=. -t 1 bench/stiffdiffusion.jl [parts=spectra,imex,walls]
#       [n=128,256,512,1024] [imex_n=128] [kappa0=1e-4,1e-3,1e-2,1e-1,1,10]
#       [refine=16] [wall_n=25,49,97,193]

using CompactLES, LinearAlgebra, Printf

const OPT = CompactLES.script_args(ARGS, (parts = "spectra,imex,walls",
    n = "128,256,512,1024", imex_n = 128, kappa0 = "1e-4,1e-3,1e-2,1e-1,1,10",
    refine = 16, wall_n = "25,49,97,193"))

circulant(n, taps) = (M = zeros(n, n);
    for i in 1:n, (o, v) in taps; M[i, mod1(i + o, n)] += v; end; M)

# The interior rows of `lele_d1_6`: α = 1/3, a = 14/9, b = 1/9.
wide_d1(n, h) = circulant(n, [(0, 1.0), (-1, 1/3), (1, 1/3)]) \
    circulant(n, [(1, 14/9/2h), (-1, -14/9/2h), (2, 1/9/4h), (-2, -1/9/4h)])

# Node i to midpoint i + 1/2: α = 9/62, a = 63/62, b = 17/62 over 3h.
staggered_d1(n, h) = circulant(n, [(0, 1.0), (-1, 9/62), (1, 9/62)]) \
    circulant(n, [(1, 63/62/h), (0, -63/62/h), (2, 17/62/3h), (-1, -17/62/3h)])

midpoint6(κ) = (150 .* (κ .+ circshift(κ, -1)) .-
                25 .* (circshift(κ, 1) .+ circshift(κ, -2)) .+
                3 .* (circshift(κ, 2) .+ circshift(κ, -3))) ./ 256

function second_order(n, h, κ)
    L = zeros(n, n)
    for i in 1:n
        kp = (κ[i] + κ[mod1(i + 1, n)]) / 2
        km = (κ[i] + κ[mod1(i - 1, n)]) / 2
        L[i, mod1(i + 1, n)] += kp / h^2
        L[i, mod1(i - 1, n)] += km / h^2
        L[i, i] -= (kp + km) / h^2
    end
    return L
end

mutable struct Counts
    apply::Int
    precondition::Int
end
const COUNTS = Counts(0, 0)

function pcg!(x, A, P, b; tol = 1e-12, maxit = 5000)
    r = b - A * x
    COUNTS.apply += 1
    z = P \ r
    COUNTS.precondition += 1
    q = copy(z); rz = dot(r, z); nb = norm(b)
    iterations = 0
    while norm(r) > tol * nb && iterations < maxit
        Aq = A * q
        COUNTS.apply += 1
        α = rz / dot(q, Aq)
        x .+= α .* q
        r .-= α .* Aq
        z = P \ r
        COUNTS.precondition += 1
        rz_next = dot(r, z)
        q .= z .+ (rz_next / rz) .* q
        rz = rz_next
        iterations += 1
    end
    return iterations
end

function spectra(ns)
    println("== spectra: CG iterations for (I - τL)x = b, preconditioned by P2")
    @printf("%6s  %11s %11s | %s\n", "n", "err wide", "err stag",
            "iterations wide / staggered at dt/dt_explicit = 1e2, 1e4, 1e6")
    κf(x) = 1 + 1e3 * exp(-((x - 0.5) / 0.05)^2)
    dκ(x) = 1e3 * exp(-((x - 0.5) / 0.05)^2) * (-2 * (x - 0.5) / 0.05^2)
    T(x) = sin(2π * x) + 0.2 * cos(6π * x)
    dT(x) = 2π * cos(2π * x) - 1.2π * sin(6π * x)
    d2T(x) = -(2π)^2 * sin(2π * x) - 0.2 * (6π)^2 * cos(6π * x)
    for n in ns
        h = 1.0 / n
        x = (0:n-1) .* h
        κ = κf.(x)
        D = wide_d1(n, h)
        Ds = staggered_d1(n, h)
        wide = D * Diagonal(κ) * D
        staggered = -Ds' * Diagonal(midpoint6(κ)) * Ds
        exact = dκ.(x) .* dT.(x) .+ κ .* d2T.(x)
        err_w = maximum(abs.(wide * T.(x) .- exact))
        err_s = maximum(abs.(staggered * T.(x) .- exact))
        b = sin.(2π .* x) .+ 0.1 .* (-1) .^ (0:n-1)
        counts = String[]
        for ratio in (1e2, 1e4, 1e6)
            τ = ratio * h^2 / maximum(κ)
            P = cholesky(Symmetric(I - τ * second_order(n, h, κ)))
            kw = pcg!(zero(b), Symmetric(I - τ * wide), P, b)
            ks = pcg!(zero(b), Symmetric(I - τ * staggered), P, b)
            push!(counts, @sprintf("%4d / %3d", kw, ks))
        end
        @printf("%6d  %11.3e %11.3e | %s\n", n, err_w, err_s, join(counts, "   "))
    end
end

struct Problem
    n::Int
    h::Float64
    κ0::Float64
    D::Matrix{Float64}
    Ds::Matrix{Float64}
end
Problem(n, κ0) = (h = 1 / n; Problem(n, h, κ0, wide_d1(n, h), staggered_d1(n, h)))
conductivity(p, T) = p.κ0 .* max.(T, 1e-3) .^ 2.5
diffusion(p, T) = -p.Ds' * Diagonal(midpoint6(conductivity(p, T))) * p.Ds
advection(p, T) = -(p.D * T)
conduction(p, T) = diffusion(p, T) * T

# Y - γ dt L(Y) Y = rhs, by Picard iteration on κ.
function implicit_stage(p, rhs, guess, γdt; tol = 1e-11)
    Y = copy(guess)
    for _ in 1:50
        A = Symmetric(I - γdt .* diffusion(p, Y))
        P = cholesky(Symmetric(I - γdt .* second_order(p.n, p.h, conductivity(p, Y))))
        Y_next = copy(Y)
        pcg!(Y_next, A, P, rhs)
        change = norm(Y_next - Y) / norm(Y_next)
        Y = Y_next
        change < tol && break
    end
    return Y
end

const ARK_E = let A = zeros(6, 6)
    A[2,1] = 1/2
    A[3,1] = 13861/62500; A[3,2] = 6889/62500
    A[4,1] = -116923316275/2393684061468; A[4,2] = -2731218467317/15368042101831
    A[4,3] = 9408046702089/11113171139209
    A[5,1] = -451086348788/2902428689909; A[5,2] = -2682348792572/7519795681897
    A[5,3] = 12662868775082/11960479115383; A[5,4] = 3355817975965/11060851509271
    A[6,1] = 647845179188/3216320057751; A[6,2] = 73281519250/8382639484533
    A[6,3] = 552539513391/3454668386233; A[6,4] = 3354512671639/8306763924573
    A[6,5] = 4040/17871
    A
end
const ARK_I = let A = zeros(6, 6)
    A[2,1] = 1/4
    A[3,1] = 8611/62500; A[3,2] = -1743/31250
    A[4,1] = 5012029/34652500; A[4,2] = -654441/2922500; A[4,3] = 174375/388108
    A[5,1] = 15267082809/155376265600; A[5,2] = -71443401/120774400
    A[5,3] = 730878875/902184768; A[5,4] = 2285395/8070912
    A[6,1] = 82889/524892; A[6,3] = 15625/83664; A[6,4] = 69875/102672
    A[6,5] = -2260/8211
    for i in 2:6; A[i,i] = 1/4; end
    A
end
const ARK_B = [82889/524892, 0, 15625/83664, 69875/102672, -2260/8211, 1/4]

function ark_step(p, T, dt)
    kE = Vector{Vector{Float64}}(undef, 6)
    kI = similar(kE)
    kE[1] = advection(p, T)
    kI[1] = conduction(p, T)
    COUNTS.apply += 1
    Y = T
    for i in 2:6
        rhs = copy(T)
        for j in 1:i-1
            rhs .+= dt * ARK_E[i,j] .* kE[j] .+ dt * ARK_I[i,j] .* kI[j]
        end
        Y = implicit_stage(p, rhs, Y, dt * ARK_I[i,i])
        kE[i] = advection(p, Y)
        kI[i] = (Y .- rhs) ./ (dt * ARK_I[i,i])
    end
    return T .+ dt .* sum(ARK_B[i] .* (kE[i] .+ kI[i]) for i in 1:6)
end

# RKL2 over dt, the stage count from dt ≤ dt_FE (s² + s - 2)/4, s odd.
function rkl2!(p, T, dt)
    ρ = 1.1 * maximum(abs, eigvals(Symmetric(diffusion(p, T))))
    dt_fe = 2 / ρ
    s = 3
    while dt_fe * (s^2 + s - 2) / 4 < dt
        s += 2
    end
    b(j) = j <= 2 ? 1/3 : (j^2 + j - 2) / (2j * (j + 1))
    w1 = 4 / (s^2 + s - 2)
    Y0 = copy(T)
    L0 = conduction(p, Y0)
    COUNTS.apply += 1
    Y_prev, Y_cur = Y0, Y0 .+ b(1) * w1 * dt .* L0
    for j in 2:s
        μ = (2j - 1) / j * b(j) / b(j - 1)
        ν = -(j - 1) / j * b(j) / b(j - 2)
        μt = μ * w1
        γt = -(1 - b(j - 1)) * μt
        Lj = conduction(p, Y_cur)
        COUNTS.apply += 1
        Y_next = μ .* Y_cur .+ ν .* Y_prev .+ (1 - μ - ν) .* Y0 .+
                 μt * dt .* Lj .+ γt * dt .* L0
        Y_prev, Y_cur = Y_cur, Y_next
    end
    T .= Y_cur
    return s
end

function rk4_advection(p, T, dt)
    k1 = advection(p, T)
    k2 = advection(p, T .+ dt / 2 .* k1)
    k3 = advection(p, T .+ dt / 2 .* k2)
    k4 = advection(p, T .+ dt .* k3)
    return T .+ dt / 6 .* (k1 .+ 2k2 .+ 2k3 .+ k4)
end

function integrate(p, T0, tfinal, nsteps, method)
    T = copy(T0)
    dt = tfinal / nsteps
    for _ in 1:nsteps
        if method === :ark
            T = ark_step(p, T, dt)
        else
            rkl2!(p, T, dt / 2)
            T = rk4_advection(p, T, dt)
            rkl2!(p, T, dt / 2)
        end
    end
    return T
end

function imex(n, κ0s, refine)
    println("== imex: n = $n, u = 1, t = 0.25, cfl 0.5; reference ARK at $(refine)x the steps")
    @printf("%8s %8s %6s | %11s %6s %6s %6s | %11s %6s\n", "κ0", "R", "steps",
            "ARK err", "order", "apply", "prec", "STS err", "apply")
    x = (0:n-1) ./ n
    T0 = 1 .+ 0.5 .* sin.(2π .* x) .+ 0.2 .* cos.(4π .* x)
    tfinal = 0.25
    base = ceil(Int, tfinal / (0.5 / n))
    for κ0 in κ0s
        p = Problem(n, κ0)
        ρ = maximum(abs, eigvals(Symmetric(diffusion(p, fill(1.7, n)))))
        reference = integrate(p, T0, tfinal, refine * base, :ark)
        previous = NaN
        for m in (1, 2)
            steps = m * base
            COUNTS.apply = 0; COUNTS.precondition = 0
            err_a = maximum(abs.(integrate(p, T0, tfinal, steps, :ark) .- reference))
            apply_a = COUNTS.apply ÷ steps
            prec_a = COUNTS.precondition ÷ steps
            COUNTS.apply = 0
            err_s = maximum(abs.(integrate(p, T0, tfinal, steps, :sts) .- reference))
            apply_s = COUNTS.apply ÷ steps
            order = isnan(previous) ? NaN : log2(previous / err_a)
            previous = err_a
            R = (tfinal / steps) / (2 / ρ)
            @printf("%8.0e %8.1f %6d | %11.3e %6.2f %6d %6d | %11.3e %6d\n",
                    κ0, R, steps, err_a, order, apply_a, prec_a, err_s, apply_s)
        end
    end
end

# Node i + 1/2 from node i: the midpoint derivative mid → node and the
# sixth-order interpolation, periodic.
staggered_back(n, h) = circulant(n, [(0, 1.0), (-1, 9/62), (1, 9/62)]) \
    circulant(n, [(0, 63/62/h), (-1, -63/62/h), (1, 17/62/3h), (-2, -17/62/3h)])
interpolation6(n) = circulant(n, [(0, 150/256), (1, 150/256), (-1, -25/256),
                                  (2, -25/256), (-2, 3/256), (3, 3/256)])

# Explicit weights on the points `pts` (in units of h) for the m-th derivative
# at `x`, exact to degree length(pts) - 1, solved in rationals.
function onesided(pts, x, m)
    K = length(pts)
    M = [Rational{BigInt}(p)^k for k in 0:K-1, p in pts]
    r = [k < m ? big(0) // 1 :
         factorial(big(k)) // factorial(big(k - m)) * Rational{BigInt}(x)^(k - m)
         for k in 0:K-1]
    Float64.(M \ r)
end

# D_s (mids × nodes), G (nodes × [wall, mids, wall]) and the interpolation on
# N nodes of a closed line, the ends closed by one-sided rows on K points.
function closure_rows(N, K)
    h = 1 / (N - 1); M = N - 1
    P = Matrix(1.0I, M, M); Q = zeros(M, N)
    for j in 2:M-1
        P[j, j-1] = P[j, j+1] = 9/62
        Q[j, j-1:j+2] .= [-17/186, -63/62, 63/62, 17/186]
    end
    c = onesided(0:K-1, 1//2, 1)
    Q[1, 1:K] .= c; Q[M, N:-1:N-K+1] .= -c
    Pn = Matrix(1.0I, N, N); Qn = zeros(N, M + 2)
    for i in 3:N-2
        Pn[i, i-1] = Pn[i, i+1] = 9/62
        Qn[i, i-1:i+2] .= [-17/186, -63/62, 63/62, 17/186]
    end
    pts = vcat(0//1, [j - 1//2 for j in 1:K-1])
    for (r, x) in ((1, 0), (2, 1))
        w = onesided(pts, x, 1)
        Qn[r, 1:K] .= w; Qn[N+1-r, M+2:-1:M+3-K] .= -w
    end
    I6 = zeros(M, N)
    for j in 3:M-2
        I6[j, j-2:j+3] .= [3, -25, 150, 150, -25, 3] ./ 256
    end
    for r in 1:2
        w = onesided(0:5, r - 1//2, 0)
        I6[r, 1:6] .= w; I6[M+1-r, N:-1:N-5] .= w
    end
    (P \ Q) ./ h, (Pn \ Qn) ./ h, I6
end

# The node-centred mirror: the periodic operators on the doubled line of
# 2(N - 1) nodes, restricted to T even through both walls.
function mirror_operator(N, κ)
    h = 1 / (N - 1); n2 = 2(N - 1)
    E = zeros(n2, N)
    for i in 1:n2
        E[i, i <= N ? i : n2 + 2 - i] = 1
    end
    L = staggered_back(n2, h) * Diagonal(interpolation6(n2) * (E * κ)) * staggered_d1(n2, h)
    return (L * E)[1:N, :]
end

function wall_properties(L, LT, h, flux_jump)
    N = size(L, 1)
    Wn = fill(h, N); Wn[1] = Wn[end] = h / 2
    WL = Diagonal(Wn) * L
    norm(WL - WL') / norm(WL), abs(sum(Wn .* LT) - flux_jump), maximum(real.(eigvals(L)))
end

function walls(Ns)
    println("== walls: L = G K D_s on [0, 1], nodes on the walls")
    @printf("%-14s %5s  %9s %9s %9s | %9s %9s %9s\n", "", "N", "err D_s", "err G",
            "err L", "asym W L", "cons", "max Re λ")
    T(x) = exp(sin(3x)); dT(x) = 3cos(3x) * T(x)
    d2T(x) = (9cos(3x)^2 - 9sin(3x)) * T(x)
    κf(x) = 1 + sin(2x) / 2; dκ(x) = cos(2x)
    F(x) = κf(x) * dT(x); dF(x) = dκ(x) * dT(x) + κf(x) * d2T(x)
    for K in (4, 5)
        for N in Ns
            h = 1 / (N - 1); x = collect(0:N-1) .* h; y = x[1:end-1] .+ h / 2
            Ds, Gs, I6 = closure_rows(N, K)
            LT = Gs * vcat(F(0.0), (I6 * κf.(x)) .* (Ds * T.(x)), F(1.0))
            L0 = Gs[:, 2:end-1] * Diagonal(I6 * κf.(x)) * Ds
            asym, cons, λ = wall_properties(L0, LT, h, F(1.0) - F(0.0))
            @printf("%-14s %5d  %9.2e %9.2e %9.2e | %9.2e %9.2e %9.1e\n", "rows K = $K", N,
                    maximum(abs.(Ds * T.(x) .- dT.(y))),
                    maximum(abs.(Gs * vcat(F(0.0), F.(y), F(1.0)) .- dF.(x))),
                    maximum(abs.(LT .- dF.(x))), asym, cons, λ)
        end
    end
    # A field of the mirror's parity, and one with T'(0) = 0 but T'''(0) ≠ 0.
    cases = (("mirror, even",
              x -> cos(2π * x) + 0.3cos(3π * x), x -> 1 + 0.5cos(π * x),
              x -> -2π * sin(2π * x) - 0.9π * sin(3π * x),
              x -> -(2π)^2 * cos(2π * x) - 2.7π^2 * cos(3π * x), x -> -0.5π * sin(π * x)),
             ("mirror, T'''",
              x -> cos(π * x) + x^3 * (1 - x)^3, x -> 1 + 0.5cos(π * x),
              x -> -π * sin(π * x) + 3x^2 * (1 - x)^2 * (1 - 2x),
              x -> -π^2 * cos(π * x) + 6x * (1 - x) * (1 - 5x + 5x^2),
              x -> -0.5π * sin(π * x)),
             ("mirror, κ'",
              x -> cos(2π * x) + 0.3cos(3π * x), x -> 1 + x / 2,
              x -> -2π * sin(2π * x) - 0.9π * sin(3π * x),
              x -> -(2π)^2 * cos(2π * x) - 2.7π^2 * cos(3π * x), x -> 0.5))
    for (name, Tc, κc, dTc, d2Tc, dκc) in cases
        for N in Ns
            h = 1 / (N - 1); x = collect(0:N-1) .* h
            L = mirror_operator(N, κc.(x))
            LT = L * Tc.(x)
            exact = dκc.(x) .* dTc.(x) .+ κc.(x) .* d2Tc.(x)
            asym, cons, λ = wall_properties(L, LT, h, 0.0)
            @printf("%-14s %5d  %9s %9s %9.2e | %9.2e %9.2e %9.1e\n", name, N, "", "",
                    maximum(abs.(LT .- exact)), asym, cons, λ)
        end
    end
end

function main()
    parts = split(OPT.parts, ',')
    "spectra" in parts && spectra(parse.(Int, split(OPT.n, ',')))
    "imex" in parts && imex(OPT.imex_n, parse.(Float64, split(OPT.kappa0, ',')), OPT.refine)
    "walls" in parts && walls(parse.(Int, split(OPT.wall_n, ',')))
end

main()
