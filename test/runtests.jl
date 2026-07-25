# Serial test suite (run with: julia --project=. test/runtests.jl
# or under one MPI rank). Ordered so failures localize: solvers → operators →
# closures/folds → metric → full RHS. Multi-rank checks live in mpi_tests.jl.

using MPI
MPI.Init(threadlevel=:funneled)
using CompactLES
using Test, LinearAlgebra, Random

const CL = CompactLES
Random.seed!(7)

per3 = ntuple(_ -> (PeriodicBC(), PeriodicBC()), 3)
mkslv(; kw...) = Solver(; bcs=per3, L_domain=(2π, 2π, 2π), art=ArtParams(enabled=false), kw...)

"Max interior error of a scalar field against an analytic function."
function ferr(s, f, fn)
    e = 0.0
    for k in 1:s.decomp.n_local[3], j in 1:s.decomp.n_local[2], i in 1:s.decomp.n_local[1]
        e = max(e, abs(f[gidx(s, i, j, k)] -
                       fn(xcoord(s, 1, i), xcoord(s, 2, j), xcoord(s, 3, k))))
    end
    e
end

fillf!(s, f, fn) = (for k in 1:s.decomp.n_local[3], j in 1:s.decomp.n_local[2],
                        i in 1:s.decomp.n_local[1]
    f[gidx(s, i, j, k)] = fn(xcoord(s, 1, i), xcoord(s, 2, j), xcoord(s, 3, k))
end; f)

# --- Exact Riemann solver for the ideal-gas Euler equations (Toro, Ch. 4) ---
# Used by the Sod shock-tube integration test to build the analytic reference.

"Star-region (p*, u*) by Newton iteration on the pressure function."
function exact_riemann_star(ρL, uL, pL, ρR, uR, pR, γ)
    cL = sqrt(γ * pL / ρL); cR = sqrt(γ * pR / ρR)
    G1 = (γ - 1) / (2γ)
    f(p, ρk, pk, ck) = p > pk ?
        (p - pk) * sqrt((2 / ((γ + 1) * ρk)) / (p + (γ - 1) / (γ + 1) * pk)) :
        (2ck / (γ - 1)) * ((p / pk)^G1 - 1)
    fder(p, ρk, pk, ck) = p > pk ?
        sqrt((2 / ((γ + 1) * ρk)) / (p + (γ - 1) / (γ + 1) * pk)) *
            (1 - (p - pk) / (2 * (p + (γ - 1) / (γ + 1) * pk))) :
        (1 / (ρk * ck)) * (p / pk)^(-(γ + 1) / (2γ))
    p = max(0.5 * (pL + pR), 1e-8)
    for _ in 1:100
        F  = f(p, ρL, pL, cL) + f(p, ρR, pR, cR) + (uR - uL)
        Fd = fder(p, ρL, pL, cL) + fder(p, ρR, pR, cR)
        pnew = p - F / Fd
        abs(pnew - p) / (0.5 * (p + pnew)) < 1e-12 && (p = pnew; break)
        p = max(pnew, 1e-9)
    end
    ustar = 0.5 * (uL + uR) +
            0.5 * (f(p, ρR, pR, cR) - f(p, ρL, pL, cL))
    (p, ustar, cL, cR)
end

"Sample (ρ,u,p) of the exact solution at self-similar speed S = (x−x0)/t."
function exact_riemann_sample(S, ρL, uL, pL, ρR, uR, pR, γ, pstar, ustar, cL, cR)
    G1 = (γ - 1) / (2γ); G6 = (γ - 1) / (γ + 1); G7 = (γ - 1) / 2
    if S <= ustar                                   # left of contact
        if pstar > pL                               # left shock
            SL = uL - cL * sqrt((γ + 1) / (2γ) * pstar / pL + G1)
            S <= SL && return (ρL, uL, pL)
            return (ρL * ((pstar / pL + G6) / (G6 * pstar / pL + 1)), ustar, pstar)
        else                                        # left rarefaction
            SHL = uL - cL; STL = ustar - cL * (pstar / pL)^G1
            S <= SHL && return (ρL, uL, pL)
            S >= STL && return (ρL * (pstar / pL)^(1 / γ), ustar, pstar)
            u = (2 / (γ + 1)) * (cL + G7 * uL + S)
            c = (2 / (γ + 1)) * (cL + G7 * (uL - S))
            return (ρL * (c / cL)^(2 / (γ - 1)), u, pL * (c / cL)^(2γ / (γ - 1)))
        end
    else                                            # right of contact
        if pstar > pR                               # right shock
            SR = uR + cR * sqrt((γ + 1) / (2γ) * pstar / pR + G1)
            S >= SR && return (ρR, uR, pR)
            return (ρR * ((pstar / pR + G6) / (G6 * pstar / pR + 1)), ustar, pstar)
        else                                        # right rarefaction
            SHR = uR + cR; STR = ustar + cR * (pstar / pR)^G1
            S >= SHR && return (ρR, uR, pR)
            S <= STR && return (ρR * (pstar / pR)^(1 / γ), ustar, pstar)
            u = (2 / (γ + 1)) * (-cR + G7 * uR + S)
            c = (2 / (γ + 1)) * (cR - G7 * (uR - S))
            return (ρR * (c / cR)^(2 / (γ - 1)), u, pR * (c / cR)^(2γ / (γ - 1)))
        end
    end
end

@testset "banded LU vs dense" begin
    for q in (1, 2), n in (9, 17)
        A = zeros(n, n)
        for i in 1:n, jj in max(1, i-q):min(n, i+q)
            A[i, jj] = (i == jj ? 3.0 : 0.0) + randn()
        end
        Ab = zeros(2q + 1, n)
        for i in 1:n, ss in -q:q
            1 <= i + ss <= n && (Ab[q+1+ss, i] = A[i, i+ss])
        end
        F = CL.BandFactor(Ab, q)
        x = randn(n); b = A * x
        y = copy(b); CL.solve_col!(y, F)
        @test y ≈ x atol = 1e-9 rtol = 1e-9
    end
end

@testset "periodic C6 derivative: spectral accuracy" begin
    s = mkslv(n_global=(32, 32, 32))
    f = CL.field(s.decomp); df = CL.field(s.decomp)
    fillf!(s, f, (x, y, z) -> sin(3x) * cos(2y))
    CL.exchange_halos!(f, s.decomp)
    CL.deriv_along!(df, f, s, 1, 1); CL._scale_grad!(df, s, 1)
    @test ferr(s, df, (x, y, z) -> 3cos(3x) * cos(2y)) < 1e-4  # C6 at k=3 on 32³
    CL.deriv_along!(df, f, s, 2, 1); CL._scale_grad!(df, s, 2)
    @test ferr(s, df, (x, y, z) -> -2sin(3x) * sin(2y)) < 2e-5
    CL.deriv_along!(df, f, s, 3, 1); CL._scale_grad!(df, s, 3)
    @test ferr(s, df, (x, y, z) -> 0.0) < 1e-10
end

@testset "pentadiagonal C10 derivative: accuracy vs C6" begin
    # Exercises the q=2 band path end to end: BandPlan assembly, the banded
    # LU + solve_col!, and the periodic self-coupling reduced-interface solve
    # (BandLineSolver) in all three dimensions — plus the closed-domain
    # closure rows. The only other C10 coverage is a finiteness smoke test.
    s = mkslv(n_global=(32, 32, 32), deriv=lele_d1_10())
    f = CL.field(s.decomp); df = CL.field(s.decomp)
    fillf!(s, f, (x, y, z) -> sin(3x) * cos(2y))
    CL.exchange_halos!(f, s.decomp)
    CL.deriv_along!(df, f, s, 1, 1); CL._scale_grad!(df, s, 1)
    e10 = ferr(s, df, (x, y, z) -> 3cos(3x) * cos(2y))
    @test e10 < 1e-7     # C10 at k=3 on 32³ (measured ≈ 2.8e-8)
    CL.deriv_along!(df, f, s, 2, 1); CL._scale_grad!(df, s, 2)
    @test ferr(s, df, (x, y, z) -> -2sin(3x) * sin(2y)) < 1e-8   # transposed path
    CL.deriv_along!(df, f, s, 3, 1); CL._scale_grad!(df, s, 3)
    @test ferr(s, df, (x, y, z) -> 0.0) < 1e-12
    # C10 must decisively beat the tridiagonal C6 on the same field (≈ 2000× here).
    s6 = mkslv(n_global=(32, 32, 32), deriv=lele_d1_6())
    f6 = CL.field(s6.decomp); df6 = CL.field(s6.decomp)
    fillf!(s6, f6, (x, y, z) -> sin(3x) * cos(2y))
    CL.exchange_halos!(f6, s6.decomp)
    CL.deriv_along!(df6, f6, s6, 1, 1); CL._scale_grad!(df6, s6, 1)
    @test e10 < ferr(s6, df6, (x, y, z) -> 3cos(3x) * cos(2y)) / 100
    # Closed domain: deg-3 polynomial is exact through the C10 closure rows
    # (a distinct band path: closure substitution, V = W = 0, no reduced stage).
    sc = Solver(n_global=(32, 12, 12), L_domain=(1.0, 1.0, 1.0),
                bcs=((SlipWallBC(), SlipWallBC()), per3[2], per3[3]),
                deriv=lele_d1_10(), art=ArtParams(enabled=false))
    fc = CL.field(sc.decomp); dfc = CL.field(sc.decomp)
    fillf!(sc, fc, (x, y, z) -> 1 + 2x + 3x^2 - x^3)
    CL.exchange_halos!(fc, sc.decomp)
    CL.deriv_along!(dfc, fc, sc, 1, 1); CL._scale_grad!(dfc, sc, 1)
    @test ferr(sc, dfc, (x, y, z) -> 2 + 6x - 3x^2) < 1e-10
end

@testset "pade_d1_4: fourth-order interior convergence" begin
    # The only coverage of the 4th-order tridiagonal scheme; a wrong
    # coefficient shows as a wrong slope, not a wrong level.
    errs = Float64[]
    for N in (16, 32, 64)
        s = mkslv(n_global=(N, 12, 12), deriv=pade_d1_4())
        f = CL.field(s.decomp); df = CL.field(s.decomp)
        fillf!(s, f, (x, y, z) -> sin(x))
        CL.exchange_halos!(f, s.decomp)
        CL.deriv_along!(df, f, s, 1, 1); CL._scale_grad!(df, s, 1)
        push!(errs, ferr(s, df, (x, y, z) -> cos(x)))
    end
    p = log(errs[1] / errs[3]) / log(4)          # observed order over 16 -> 64
    @test 3.5 < p < 4.5
    @test errs[3] < errs[2] < errs[1]
end

@testset "closures on the transposed path: y and z walls" begin
    # operators.jl applies boundary closure rows through a DIFFERENT code path
    # for y/z (the transposed line gather) than for x. Every other closed-domain
    # test in this suite walls off x only, so those rows were never executed.
    # Deg-3 polynomial exactness is the same assertion the x test makes.
    for d in (2, 3)
        ng = ntuple(k -> k == d ? 32 : 12, 3)
        bcs = ntuple(k -> k == d ? (SlipWallBC(), SlipWallBC()) : per3[k], 3)
        s = Solver(n_global=ng, L_domain=(1.0, 1.0, 1.0), bcs=bcs,
                   art=ArtParams(enabled=false))
        f = CL.field(s.decomp); df = CL.field(s.decomp)
        poly = t -> 1 + 2t + 3t^2 - t^3
        dpoly = t -> 2 + 6t - 3t^2
        fillf!(s, f, (x, y, z) -> poly(d == 2 ? y : z))
        CL.exchange_halos!(f, s.decomp)
        CL.deriv_along!(df, f, s, d, 1); CL._scale_grad!(df, s, d)
        @test ferr(s, df, (x, y, z) -> dpoly(d == 2 ? y : z)) < 1e-10
    end
end

@testset "C10 through a coordinate-singularity fold" begin
    # operators_banded.jl folds the ghost-unknown coupling onto the diagonal
    # for the pentadiagonal scheme (lo_fold/hi_fold). The C6 axis tests never
    # reach it and the C10 tests are all periodic or plain-walled.
    s = Solver(n_global=(64, 1, 12), L_domain=(1.0, 1.0, 0.5),
               metric=CylindricalMetric(), deriv=lele_d1_10(),
               bcs=((AxisBC(), SlipWallBC()), per3[2], per3[3]),
               art=ArtParams(enabled=false))
    f = CL.field(s.decomp); df = CL.field(s.decomp)
    fillf!(s, f, (r, θ, z) -> r * exp(-4r^2))            # odd across the axis
    CL.exchange_halos!(f, s.decomp)
    CL.deriv_along!(df, f, s, 1, -1); CL._scale_grad!(df, s, 1)
    @test ferr(s, df, (r, θ, z) -> (1 - 8r^2) * exp(-4r^2)) < 5e-6
    fillf!(s, f, (r, θ, z) -> exp(-4r^2))                # even across the axis
    CL.exchange_halos!(f, s.decomp)
    CL.deriv_along!(df, f, s, 1, 1); CL._scale_grad!(df, s, 1)
    @test ferr(s, df, (r, θ, z) -> -8r * exp(-4r^2)) < 2e-5
end

@testset "transposed y/z path ≡ x path on permuted data" begin
    s = mkslv(n_global=(24, 24, 24))
    f = CL.field(s.decomp); g = CL.field(s.decomp)
    d1 = CL.field(s.decomp); d2 = CL.field(s.decomp)
    fn = (x, y, z) -> sin(2x + 0.3) * cos(y) + 0.1z^0   # z-independent
    fillf!(s, f, fn)                                     # varies in x
    fillf!(s, g, (x, y, z) -> fn(y, x, z))               # same profile along y
    CL.exchange_halos!(f, s.decomp); CL.exchange_halos!(g, s.decomp)
    CL.deriv_along!(d1, f, s, 1, 1)
    CL.deriv_along!(d2, g, s, 2, 1)
    e = maximum(abs(d1[gidx(s, i, j, k)] - d2[gidx(s, j, i, k)])
                for i in 1:24, j in 1:24, k in 1:24)
    @test e < 1e-11
end

@testset "closed-domain closures: polynomial exactness (deg ≤ 3)" begin
    s = Solver(n_global=(32, 12, 12), L_domain=(1.0, 1.0, 1.0),
               bcs=((SlipWallBC(), SlipWallBC()), per3[2], per3[3]),
               art=ArtParams(enabled=false))
    f = CL.field(s.decomp); df = CL.field(s.decomp)
    fillf!(s, f, (x, y, z) -> 1 + 2x + 3x^2 - x^3)
    CL.exchange_halos!(f, s.decomp)
    CL.deriv_along!(df, f, s, 1, 1); CL._scale_grad!(df, s, 1)
    @test ferr(s, df, (x, y, z) -> 2 + 6x - 3x^2) < 1e-10
end

@testset "filter: constants exact, Nyquist damped, parity of closures" begin
    s = mkslv(n_global=(32, 12, 12))
    f = CL.field(s.decomp)
    fillf!(s, f, (x, y, z) -> 1.0)
    filter_field!(f, s)
    @test ferr(s, f, (x, y, z) -> 1.0) < 1e-12
    nx = s.decomp.n_local[1]
    for k in 1:s.decomp.n_local[3], j in 1:s.decomp.n_local[2], i in 1:nx
        f[gidx(s, i, j, k)] = 1.0 + 0.5 * (-1)^i        # constant + Nyquist
    end
    filter_field!(f, s)
    dev = maximum(abs(f[gidx(s, i, 1, 1)] - 1.0) for i in 1:nx)
    @test dev < 0.35                                      # sawtooth strongly damped
end

@testset "axisymmetric axis fold: manufactured smooth solution" begin
    # u_r = r·g(r) is an odd smooth function; d/dr through the fold must
    # match analytics at the first half-offset nodes — the sharpest probe of
    # the folded row and mirror fill.
    s = Solver(n_global=(64, 1, 12), L_domain=(1.0, 1.0, 0.5),
               metric=CylindricalMetric(),
               bcs=((AxisBC(), SlipWallBC()), per3[2], per3[3]),
               art=ArtParams(enabled=false))
    f = CL.field(s.decomp); df = CL.field(s.decomp)
    fillf!(s, f, (r, θ, z) -> r * exp(-4r^2))            # odd across the axis
    CL.exchange_halos!(f, s.decomp)
    CL.deriv_along!(df, f, s, 1, -1); CL._scale_grad!(df, s, 1)
    @test ferr(s, df, (r, θ, z) -> (1 - 8r^2) * exp(-4r^2)) < 5e-6
    fillf!(s, f, (r, θ, z) -> exp(-4r^2))                # even across the axis
    CL.exchange_halos!(f, s.decomp)
    CL.deriv_along!(df, f, s, 1, 1); CL._scale_grad!(df, s, 1)
    @test ferr(s, df, (r, θ, z) -> -8r * exp(-4r^2)) < 2e-5  # even fold: 3rd-order, larger const
end

@testset "resolved-θ axis: antipodal pairing (local)" begin
    # f = r cosθ · e^{−4r²} = x·g is globally smooth through the axis and is
    # ODD under (r,θ)→(−r,θ) with the pairing (θ+π picks up the cos sign).
    s = Solver(n_global=(48, 16, 1), L_domain=(1.0, 2π, 1.0),
               metric=CylindricalMetric(),
               bcs=((AxisBC(), SlipWallBC()), per3[2], per3[3]),
               art=ArtParams(enabled=false))
    f = CL.field(s.decomp); df = CL.field(s.decomp)
    fillf!(s, f, (r, θ, z) -> r * cos(θ) * exp(-4r^2))
    CL.exchange_halos!(f, s.decomp)
    CL.deriv_along!(df, f, s, 1, -1); CL._scale_grad!(df, s, 1)
    @test ferr(s, df, (r, θ, z) -> cos(θ) * (1 - 8r^2) * exp(-4r^2)) < 1e-5
    # A scalar even case: f = e^{−4r²}·(1 + ½cos 2θ) maps to itself at θ+π.
    fillf!(s, f, (r, θ, z) -> exp(-4r^2) * (1 + 0.5cos(2θ)))
    CL.exchange_halos!(f, s.decomp)
    CL.deriv_along!(df, f, s, 1, 1); CL._scale_grad!(df, s, 1)
    @test ferr(s, df, (r, θ, z) -> -8r * exp(-4r^2) * (1 + 0.5cos(2θ))) < 1e-4  # 3rd-order, larger const
end

@testset "spherical poles + origin: derivative of a smooth 3-D Gaussian" begin
    # f = e^{−4r²} is smooth at origin and poles; ∂f/∂r and (1/r)∂f/∂θ = 0.
    s = Solver(n_global=(40, 16, 12), L_domain=(1.0, π, 2π),
               metric=SphericalMetric(),
               bcs=((OriginBC(), SlipWallBC()),
                    (PoleBC(), PoleBC()), per3[3]),
               art=ArtParams(enabled=false))
    f = CL.field(s.decomp); df = CL.field(s.decomp)
    fillf!(s, f, (r, θ, φ) -> exp(-4r^2))
    CL.exchange_halos!(f, s.decomp)
    CL.deriv_along!(df, f, s, 1, 1); CL._scale_grad!(df, s, 1)
    @test ferr(s, df, (r, θ, φ) -> -8r * exp(-4r^2)) < 1e-4  # 3rd-order, larger const
    CL.deriv_along!(df, f, s, 2, 1); CL._scale_grad!(df, s, 2)
    @test ferr(s, df, (r, θ, φ) -> 0.0) < 1e-8
end

@testset "rigid rotation in cylindrical: zero strain" begin
    # u_θ = Ω r ⇒ S_ij = 0 identically; probes the curvature corrections.
    s = Solver(n_global=(32, 16, 12), L_domain=(1.0, 2π, 0.5),
               metric=CylindricalMetric(), origin=(0.2, 0.0, 0.0),
               bcs=((SlipWallBC(), SlipWallBC()), per3[2], per3[3]),
               art=ArtParams(enabled=false))
    Q = allocate_state(s)
    initialize!(s, Q, (r, θ, z) -> Prim(u=(0.0, 0.3r, 0.0), p=1.0, rho=1.0))
    dQ = zero(Q)
    compute_rhs!(s, Q, dQ)
    smax = 0.0
    for k in 1:s.decomp.n_local[3], j in 1:s.decomp.n_local[2], i in 1:s.decomp.n_local[1]
        I = gidx(s, i, j, k)
        for b in 1:3, a in 1:3
            smax = max(smax, abs(0.5 * (s.grad_u[a, b][I] + s.grad_u[b, a][I])))
        end
    end
    @test smax < 1e-8
end

@testset "freestream preservation (uniform state ⇒ dQ ≈ 0)" begin
    # Uniform ρ, p, u = 0 must give zero RHS in every metric, with stretch,
    # and with folds: divergence/source/geometry consistency in one number.
    cases = [
        (; n_global=(24, 12, 12), L_domain=(1.0, 1.0, 1.0), metric=CartesianMetric(),
           bcs=per3, kw=(;)),
        (; n_global=(24, 12, 12), L_domain=(1.0, 1.0, 1.0), metric=CartesianMetric(),
           bcs=per3, kw=(; stretch=(sine_cluster(0.0, 1.0, 0.5, 0.4),
                                    nothing, nothing),
                         bcs=((SlipWallBC(), SlipWallBC()), per3[2], per3[3]))),
        (; n_global=(24, 12, 12), L_domain=(1.0, 2π, 0.5), metric=CylindricalMetric(),
           bcs=((SlipWallBC(), SlipWallBC()), per3[2], per3[3]),
           kw=(; origin=(0.2, 0.0, 0.0))),
        (; n_global=(32, 1, 12), L_domain=(1.0, 1.0, 0.5), metric=CylindricalMetric(),
           bcs=((AxisBC(), SlipWallBC()), per3[2], per3[3]), kw=(;)),
        (; n_global=(24, 16, 1), L_domain=(1.0, 2π, 1.0), metric=CylindricalMetric(),
           bcs=((AxisBC(), SlipWallBC()), per3[2], per3[3]), kw=(;)),
        (; n_global=(24, 12, 12), L_domain=(1.0, π, 2π), metric=SphericalMetric(),
           bcs=((OriginBC(), SlipWallBC()), (PoleBC(), PoleBC()), per3[3]),
           kw=(;)),
    ]
    for cs in cases
        kw = merge((; n_global=cs.n_global, L_domain=cs.L_domain, metric=cs.metric,
                     bcs=cs.bcs, art=ArtParams(enabled=false)), cs.kw)
        s = Solver(; kw...)
        Q = allocate_state(s)
        initialize!(s, Q, (x, y, z) -> Prim(u=(0, 0, 0), p=1.0, rho=1.0))
        apply_bcs!(s, Q)
        dQ = zero(Q)
        compute_rhs!(s, Q, dQ)
        m = maximum(abs(dQ[gidx(s, i, j, k), c])
                    for c in 1:s.n_cons, i in 1:s.decomp.n_local[1],
                        j in 1:s.decomp.n_local[2], k in 1:s.decomp.n_local[3])
        @test m < 1e-8
    end
end

@testset "EOS: conserved ↔ primitive round trip (two species)" begin
    eos = IdealMixture([IdealSpecies{Float64}("a", 1.0, 1.4),
                        IdealSpecies{Float64}("b", 0.2, 1.09)])
    s = mkslv(n_global=(12, 12, 12), eos=eos)
    Q = allocate_state(s)
    pr = Prim(u=(0.3, -0.1, 0.2), p=0.8, T_ion=1.7, Y=(0.35, 0.65))
    initialize!(s, Q, (x, y, z) -> pr)
    CL.exchange_state!(Q, s.decomp)
    CL.primitives!(s, Q)
    I = gidx(s, 3, 4, 5)
    @test s.p[I] ≈ 0.8 atol = 1e-12
    @test s.T_ion[I] ≈ 1.7 atol = 1e-12
    @test s.u[I] ≈ 0.3 atol = 1e-12
    @test s.Y[2][I] ≈ 0.65 atol = 1e-12
end

@testset "conservation: periodic RHS integrates to zero" begin
    s = mkslv(n_global=(16, 16, 16), transport=Transport(mu0=1e-3))
    Q = allocate_state(s)
    initialize!(s, Q, (x, y, z) ->
        Prim(u=(0.1sin(x)cos(y), -0.1cos(x)sin(y), 0.05sin(z)),
             p=1 + 0.05cos(x)cos(z), rho=1 + 0.1sin(y)))
    dQ = zero(Q)
    compute_rhs!(s, Q, dQ)
    for c in 1:s.n_cons
        tot = sum(dQ[gidx(s, i, j, k), c] for i in 1:16, j in 1:16, k in 1:16)
        @test abs(tot) < 1e-8 * 16^3
    end
end

@testset "NSCBC outflow: matched uniform stream ⇒ no correction" begin
    s = Solver(n_global=(32, 12, 12), L_domain=(1.0, 0.4, 0.4),
               bcs=((DirichletBC((x, y, z, t) -> Prim(u=(0.3, 0, 0), p=1.0, rho=1.0)),
                     NSCBCOutflowBC(pinf=1.0)), per3[2], per3[3]),
               art=ArtParams(enabled=false))
    Q = allocate_state(s)
    initialize!(s, Q, (x, y, z) -> Prim(u=(0.3, 0, 0), p=1.0, rho=1.0))
    apply_bcs!(s, Q)
    dQ = zero(Q)
    compute_rhs!(s, Q, dQ)
    nx = s.decomp.n_local[1]
    m = maximum(abs(dQ[gidx(s, nx, j, k), c])
                for c in 1:s.n_cons, j in 1:12, k in 1:12)
    @test m < 1e-8
end

@testset "NoSlipWallBC: adiabatic zeroes velocity, isothermal sets T_ion" begin
    for Twall in (NaN, 2.5)
        s = Solver(n_global=(24, 12, 12), L_domain=(1.0, 0.4, 0.4),
                   bcs=((NoSlipWallBC(Twall=Twall), NoSlipWallBC(Twall=Twall)),
                        per3[2], per3[3]),
                   art=ArtParams(enabled=false))
        Q = allocate_state(s)
        initialize!(s, Q, (x, y, z) -> Prim(u=(0.3, -0.2, 0.1), p=1.0, T_ion=1.0))
        apply_bcs!(s, Q)
        CL.exchange_state!(Q, s.decomp)
        CL.primitives!(s, Q)
        nx = s.decomp.n_local[1]
        uw = maximum(abs(s.u[gidx(s, i, j, k)]) + abs(s.v[gidx(s, i, j, k)]) +
                     abs(s.w[gidx(s, i, j, k)])
                     for i in (1, nx), j in 1:12, k in 1:12)   # both walls
        @test uw < 1e-12
        if !isnan(Twall)                       # isothermal wall holds Twall
            e = maximum(abs(s.T_ion[gidx(s, i, j, k)] - Twall)
                        for i in (1, nx), j in 1:12, k in 1:12)
            @test e < 1e-12
        end
    end
end

@testset "ExtrapolationBC: copies the adjacent interior plane" begin
    s = Solver(n_global=(24, 12, 12), L_domain=(1.0, 0.4, 0.4),
               bcs=((ExtrapolationBC(), ExtrapolationBC()), per3[2], per3[3]),
               art=ArtParams(enabled=false))
    Q = allocate_state(s)
    initialize!(s, Q, (x, y, z) -> Prim(u=(0.2 + x, 0, 0), p=1 + 0.1x, rho=1 + 0.3x))
    apply_bcs!(s, Q)
    nx = s.decomp.n_local[1]
    d = 0.0
    for c in 1:s.n_cons, j in 1:12, k in 1:12
        d = max(d, abs(Q[gidx(s, 1, j, k), c]  - Q[gidx(s, 2, j, k), c]))
        d = max(d, abs(Q[gidx(s, nx, j, k), c] - Q[gidx(s, nx-1, j, k), c]))
    end
    @test d == 0.0
    # a uniform state must be untouched by the extrapolation
    Q2 = allocate_state(s)
    initialize!(s, Q2, (x, y, z) -> Prim(u=(0.2, 0, 0), p=1.0, rho=1.0))
    ref = copy(Q2)
    apply_bcs!(s, Q2)
    @test Q2 == ref
end

@testset "NSCBC inflow: matched uniform stream ⇒ no correction" begin
    # Mirrors the outflow test. This whole method was previously never
    # compiled, so nothing in it — including the transverse terms — had ever
    # been executed, let alone checked.
    uin = (0.3, 0.0, 0.0)
    s = Solver(n_global=(32, 12, 12), L_domain=(1.0, 0.4, 0.4),
               bcs=((NSCBCInflowBC(u=uin, T_ion=1.0), NSCBCOutflowBC(pinf=1.0)),
                    per3[2], per3[3]),
               eos=single_species(gamma=1.4, R=1.0),
               art=ArtParams(enabled=false))
    Q = allocate_state(s)
    initialize!(s, Q, (x, y, z) -> Prim(u=uin, p=1.0, T_ion=1.0))
    apply_bcs!(s, Q)
    dQ = zero(Q)
    compute_rhs!(s, Q, dQ)
    m = maximum(abs(dQ[gidx(s, 1, j, k), c]) for c in 1:s.n_cons, j in 1:12, k in 1:12)
    @test m < 1e-8
    # and a mismatched stream must produce a non-trivial correction
    Q2 = allocate_state(s)
    initialize!(s, Q2, (x, y, z) -> Prim(u=(0.15, 0, 0), p=1.0, T_ion=1.0))
    apply_bcs!(s, Q2)
    dQ2 = zero(Q2)
    compute_rhs!(s, Q2, dQ2)
    m2 = maximum(abs(dQ2[gidx(s, 1, j, k), c]) for c in 1:s.n_cons, j in 1:12, k in 1:12)
    @test m2 > 1e-3
end

@testset "multi-species artificial diffusivity responds to Y gradients" begin
    # artificial.jl computes per-species D* only when n_species > 1; that branch was
    # never executed, since the only multi-species test disables art.
    eos = IdealMixture([IdealSpecies{Float64}("a", 1.0, 1.4),
                        IdealSpecies{Float64}("b", 0.2, 1.09)])
    s = Solver(n_global=(64, 12, 12), L_domain=(1.0, 0.2, 0.2), bcs=per3, eos=eos,
               art=ArtParams(enabled=true))
    Q = allocate_state(s)
    dQ = zero(Q)
    # smooth composition: D* must stay tiny
    initialize!(s, Q, (x, y, z) -> Prim(Y=(0.5 + 0.1sin(2π * x), 0.5 - 0.1sin(2π * x)),
                                        p=1.0, rho=1.0))
    compute_rhs!(s, Q, dQ)
    smooth_max = max(maximum(s.D_art[1]), maximum(s.D_art[2]))
    # sharp interface: D* must switch on
    initialize!(s, Q, (x, y, z) -> begin
        θ = tanh_blend(x, 0.5, 0.01)
        Prim(Y=(1 - θ, θ), p=1.0, rho=1.0)
    end)
    compute_rhs!(s, Q, dQ)
    sharp_max = max(maximum(s.D_art[1]), maximum(s.D_art[2]))
    @test sharp_max > 100 * max(smooth_max, 1e-14)
    @test all(isfinite, s.D_art[1]) && all(isfinite, s.D_art[2])
end

@testset "dt_report agrees with compute_dt and names the limiter" begin
    s = mkslv(n_global=(16, 16, 16), transport=Transport(mu0=1e-3))
    Q = allocate_state(s)
    initialize!(s, Q, (x, y, z) -> Prim(u=(0.2sin(x), 0, 0), p=1 + 0.1cos(y), rho=1.0))
    r = dt_report(s, Q)
    @test r.dt ≈ compute_dt(s, Q) rtol = 1e-12
    @test r.kind in (:acoustic, :diffusive, :curvature)
    @test r.dim in 1:3
    @test all(1 .<= r.index .<= 16)
end

@testset "curvature_rate: collapsed angular dims restrict dt" begin
    # The spherical branch of curvature_rate had no coverage at all, and the
    # whole point of the term is that a COLLAPSED angular dimension carries a
    # stiff geometric source the advective CFL loop never sees. Swirl must
    # therefore shorten dt even though nothing varies in θ or φ.
    mk(metric, uang) = begin
        s = Solver(n_global=(64, 1, 1), L_domain=(1.0, 1.0, 1.0), metric=metric,
                   origin=(0.5, π / 2, 0.0),
                   bcs=((SlipWallBC(), SlipWallBC()), per3[2], per3[3]),
                   art=ArtParams(enabled=false))
        Q = allocate_state(s)
        initialize!(s, Q, (r, θ, φ) -> Prim(u=(0.0, uang, uang), p=1.0, rho=1.0))
        s, Q
    end
    for metric in (CylindricalMetric(), SphericalMetric())
        s0, Q0 = mk(metric, 0.0)
        s1, Q1 = mk(metric, 2.0)
        @test compute_dt(s1, Q1) < compute_dt(s0, Q0)
        @test CL.curvature_rate(s1, metric, gidx(s1, 5, 1, 1), (0.0, 2.0, 2.0)) > 0
        @test CL.curvature_rate(s0, metric, gidx(s0, 5, 1, 1), (0.0, 0.0, 0.0)) == 0
    end
    # Cartesian has no curvature term at all
    sc = mkslv(n_global=(16, 16, 16))
    @test CL.curvature_rate(sc, CartesianMetric(), gidx(sc, 2, 2, 2), (1.0, 1.0, 1.0)) == 0
end

@testset "save_vtk writes a readable .pvtr/.vtr pair" begin
    s = mkslv(n_global=(12, 12, 12))
    Q = allocate_state(s)
    initialize!(s, Q, (x, y, z) -> Prim(u=(sin(x), 0, 0), p=1 + 0.1cos(y), rho=1.0))
    save_vtk(s, Q, "test_vtk")
    files = filter(startswith("test_vtk"), readdir())
    @test any(endswith(".pvtr"), files)
    @test any(endswith(".vtr"), files)
    @test all(f -> filesize(f) > 0, files)
    txt = read(first(filter(endswith(".pvtr"), files)), String)
    @test occursin("PRectilinearGrid", txt)
    foreach(rm, files)
end

@testset "conserved_from_prim / nspecies round trip" begin
    eos = IdealMixture([IdealSpecies{Float64}("a", 1.0, 1.4),
                        IdealSpecies{Float64}("b", 0.2, 1.09)])
    @test nspecies(eos) == 2
    @test nspecies(single_species()) == 1
    pr = Prim(u=(0.3, -0.1, 0.2), p=0.8, T_ion=1.7, Y=(0.35, 0.65))
    q = conserved_from_prim(eos, pr)
    ρ = q[1] + q[2]
    @test q[1] / ρ ≈ 0.35 atol = 1e-12
    @test q[3] / ρ ≈ 0.3 atol = 1e-12          # ρu / ρ
    Rm = 0.35 * eos.Rk[1] + 0.65 * eos.Rk[2]
    @test ρ * Rm * 1.7 ≈ 0.8 atol = 1e-12      # p = ρ R_m T_ion
end

@testset "checkpoint round trip" begin
    s = mkslv(n_global=(12, 12, 12))
    Q = allocate_state(s)
    initialize!(s, Q, (x, y, z) -> Prim(u=(sin(x), 0, 0), p=1 + 0.1cos(y), rho=1.0))
    s.t = 0.37; s.step = 42
    save_checkpoint(s, Q, "test_ckpt")
    Q2 = allocate_state(s); s.t = 0.0; s.step = 0
    load_checkpoint!(s, Q2, "test_ckpt")
    @test s.t == 0.37 && s.step == 42
    @test all(Q2[gidx(s, i, j, k), c] == Q[gidx(s, i, j, k), c]
              for c in 1:5, i in 1:12, j in 1:12, k in 1:12)
    foreach(rm, filter(startswith("test_ckpt"), readdir()))
end

@testset "smoke: three RK steps of every headline configuration" begin
    for build in (
        () -> setup(Problem(domain=((0.0, 2π), (0.0, 2π), (0.0, 2π)), bcs=per3,
                            ic=(x, y, z) -> Prim(u=(0.1sin(x), 0, 0), p=1.0, rho=1.0)),
                    Numerics(n_global=(16, 16, 16))),
        () -> setup(Problem(domain=((0.0, 2π), (0.0, 2π), (0.0, 2π)), bcs=per3,
                            ic=(x, y, z) -> Prim(u=(0.1sin(x), 0, 0), p=1.0, rho=1.0)),
                    Numerics(n_global=(16, 16, 16), deriv=lele_d1_10())),
        () -> setup(Problem(domain=((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)),
                            metric=CylindricalMetric(),
                            bcs=((AxisBC(), SlipWallBC()), per3[2], per3[3]),
                            ic=(r, θ, z) -> Prim(u=(0, 0, 0), p=1 + exp(-40(r - 0.4)^2), rho=1.0)),
                    Numerics(n_global=(48, 1, 1))),
    )
        s, Q = build()
        run!(s, Q; tfinal=1e9, nmax=3)
        bad = any(!isfinite(Q[gidx(s, i, j, k), c])
                  for c in 1:s.n_cons, i in 1:s.decomp.n_local[1],
                      j in 1:s.decomp.n_local[2], k in 1:s.decomp.n_local[3])
        @test !bad
    end
end

@testset "Sod shock tube: profile matches exact Riemann solution" begin
    # Classic single-gas Sod (γ = 1.4) on [0,1], diaphragm at x = 0.5:
    #   left (ρ,u,p) = (1, 0, 1), right (ρ,u,p) = (0.125, 0, 0.1).
    # Integrate to t = 0.2 in 1-D (collapsed transverse dims) with artificial
    # fluid properties capturing the shock, and compare the ρ/u/p line profile
    # against the analytic Riemann solution sampled at each node. Errors are
    # L1 over the profile — shock/contact smearing over a few cells is expected,
    # so the tolerance is looser than the smooth-operator tests but still tight
    # enough that a wrong wave speed, plateau, or star state fails it.
    γ = 1.4
    ρL, uL, pL = 1.0, 0.0, 1.0
    ρR, uR, pR = 0.125, 0.0, 0.1
    x0 = 0.5; tfin = 0.2
    Nx = 400; Lx = 1.0; hx = Lx / (Nx - 1); δ = 2hx
    prob = Problem(name="Sod", eos=single_species(gamma=γ, R=1.0),
                   transport=Transport(mu0=0.0),
                   domain=((0.0, Lx), (0.0, hx), (0.0, hx)),
                   bcs=((SlipWallBC(), SlipWallBC()), per3[2], per3[3]),
                   ic=(x, y, z) -> begin
                       θ = tanh_blend(x, x0, δ)
                       Prim(rho=(1 - θ) * ρL + θ * ρR,
                            u=((1 - θ) * uL + θ * uR, 0.0, 0.0),
                            p=(1 - θ) * pL + θ * pR)
                   end)
    s, Q = setup(prob, Numerics(n_global=(Nx, 1, 1), art=ArtParams(enabled=true),
                                cfl=0.4, filter_interval=1))
    run!(s, Q; tfinal=tfin, nmax=100_000)
    CL.exchange_state!(Q, s.decomp)
    CL.primitives!(s, Q)

    pstar, ustar, cL, cR = exact_riemann_star(ρL, uL, pL, ρR, uR, pR, γ)
    # Star state is the canonical Sod result (Toro): p* ≈ 0.30313, u* ≈ 0.92745.
    @test pstar ≈ 0.30313 atol = 1e-4
    @test ustar ≈ 0.92745 atol = 1e-4

    nx = s.decomp.n_local[1]
    eρ = eu = ep = 0.0
    for i in 1:nx
        I = gidx(s, i, 1, 1); x = xcoord(s, 1, i)
        r, u, p = exact_riemann_sample((x - x0) / tfin, ρL, uL, pL, ρR, uR, pR,
                                       γ, pstar, ustar, cL, cR)
        eρ += abs(s.rho[I] - r); eu += abs(s.u[I] - u); ep += abs(s.p[I] - p)
    end
    eρ /= nx; eu /= nx; ep /= nx
    @test eρ < 2e-2      # measured ≈ 2.9e-3
    @test eu < 2e-2      # measured ≈ 4.6e-3
    @test ep < 1e-2      # measured ≈ 2.5e-3
end

println("serial tests complete")
