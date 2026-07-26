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
function ferr(solver, f, fn)
    e = 0.0
    for k in 1:solver.decomp.n_local[3], j in 1:solver.decomp.n_local[2], i in 1:solver.decomp.n_local[1]
        e = max(e, abs(f[gidx(solver, i, j, k)] -
                       fn(xcoord(solver, 1, i), xcoord(solver, 2, j), xcoord(solver, 3, k))))
    end
    e
end

fillf!(solver, f, fn) = (for k in 1:solver.decomp.n_local[3], j in 1:solver.decomp.n_local[2],
                        i in 1:solver.decomp.n_local[1]
    f[gidx(solver, i, j, k)] = fn(xcoord(solver, 1, i), xcoord(solver, 2, j), xcoord(solver, 3, k))
end; f)

# Analytic references (exact Riemann solver, Noh, Sedov) live in one place so
# the serial suite and test/validation.jl measure against the same solution.
include("references.jl")

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
    solver = mkslv(n_global=(32, 32, 32))
    f = CL.field(solver.decomp); df = CL.field(solver.decomp)
    fillf!(solver, f, (x, y, z) -> sin(3x) * cos(2y))
    CL.exchange_halos!(f, solver.decomp)
    CL.deriv_along!(df, f, solver, 1, 1); CL._scale_grad!(df, solver, 1)
    @test ferr(solver, df, (x, y, z) -> 3cos(3x) * cos(2y)) < 1e-4  # C6 at k=3 on 32³
    CL.deriv_along!(df, f, solver, 2, 1); CL._scale_grad!(df, solver, 2)
    @test ferr(solver, df, (x, y, z) -> -2sin(3x) * sin(2y)) < 2e-5
    CL.deriv_along!(df, f, solver, 3, 1); CL._scale_grad!(df, solver, 3)
    @test ferr(solver, df, (x, y, z) -> 0.0) < 1e-10
end

@testset "pentadiagonal C10 derivative: accuracy vs C6" begin
    # Exercises the q=2 band path end to end: BandPlan assembly, the banded
    # LU + solve_col!, and the periodic self-coupling reduced-interface solve
    # (BandLineSolver) in all three dimensions — plus the closed-domain
    # closure rows. The only other C10 coverage is a finiteness smoke test.
    solver = mkslv(n_global=(32, 32, 32), deriv=lele_d1_10())
    f = CL.field(solver.decomp); df = CL.field(solver.decomp)
    fillf!(solver, f, (x, y, z) -> sin(3x) * cos(2y))
    CL.exchange_halos!(f, solver.decomp)
    CL.deriv_along!(df, f, solver, 1, 1); CL._scale_grad!(df, solver, 1)
    e10 = ferr(solver, df, (x, y, z) -> 3cos(3x) * cos(2y))
    @test e10 < 1e-7     # C10 at k=3 on 32³ (measured ≈ 2.8e-8)
    CL.deriv_along!(df, f, solver, 2, 1); CL._scale_grad!(df, solver, 2)
    @test ferr(solver, df, (x, y, z) -> -2sin(3x) * sin(2y)) < 1e-8   # transposed path
    CL.deriv_along!(df, f, solver, 3, 1); CL._scale_grad!(df, solver, 3)
    @test ferr(solver, df, (x, y, z) -> 0.0) < 1e-12
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
        solver = mkslv(n_global=(N, 12, 12), deriv=pade_d1_4())
        f = CL.field(solver.decomp); df = CL.field(solver.decomp)
        fillf!(solver, f, (x, y, z) -> sin(x))
        CL.exchange_halos!(f, solver.decomp)
        CL.deriv_along!(df, f, solver, 1, 1); CL._scale_grad!(df, solver, 1)
        push!(errs, ferr(solver, df, (x, y, z) -> cos(x)))
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
        solver = Solver(n_global=ng, L_domain=(1.0, 1.0, 1.0), bcs=bcs,
                   art=ArtParams(enabled=false))
        f = CL.field(solver.decomp); df = CL.field(solver.decomp)
        poly = t -> 1 + 2t + 3t^2 - t^3
        dpoly = t -> 2 + 6t - 3t^2
        fillf!(solver, f, (x, y, z) -> poly(d == 2 ? y : z))
        CL.exchange_halos!(f, solver.decomp)
        CL.deriv_along!(df, f, solver, d, 1); CL._scale_grad!(df, solver, d)
        @test ferr(solver, df, (x, y, z) -> dpoly(d == 2 ? y : z)) < 1e-10
    end
end

@testset "C10 through a coordinate-singularity fold" begin
    # operators_banded.jl folds the ghost-unknown coupling onto the diagonal
    # for the pentadiagonal scheme (lo_fold/hi_fold). The C6 axis tests never
    # reach it and the C10 tests are all periodic or plain-walled.
    solver = Solver(n_global=(64, 1, 12), L_domain=(1.0, 1.0, 0.5),
               metric=CylindricalMetric(), deriv=lele_d1_10(),
               bcs=((AxisBC(), SlipWallBC()), per3[2], per3[3]),
               art=ArtParams(enabled=false))
    f = CL.field(solver.decomp); df = CL.field(solver.decomp)
    fillf!(solver, f, (r, θ, z) -> r * exp(-4r^2))            # odd across the axis
    CL.exchange_halos!(f, solver.decomp)
    CL.deriv_along!(df, f, solver, 1, -1); CL._scale_grad!(df, solver, 1)
    @test ferr(solver, df, (r, θ, z) -> (1 - 8r^2) * exp(-4r^2)) < 5e-6
    fillf!(solver, f, (r, θ, z) -> exp(-4r^2))                # even across the axis
    CL.exchange_halos!(f, solver.decomp)
    CL.deriv_along!(df, f, solver, 1, 1); CL._scale_grad!(df, solver, 1)
    @test ferr(solver, df, (r, θ, z) -> -8r * exp(-4r^2)) < 2e-5
end

@testset "transposed y/z path ≡ x path on permuted data" begin
    solver = mkslv(n_global=(24, 24, 24))
    f = CL.field(solver.decomp); g = CL.field(solver.decomp)
    d1 = CL.field(solver.decomp); d2 = CL.field(solver.decomp)
    fn = (x, y, z) -> sin(2x + 0.3) * cos(y) + 0.1z^0   # z-independent
    fillf!(solver, f, fn)                                     # varies in x
    fillf!(solver, g, (x, y, z) -> fn(y, x, z))               # same profile along y
    CL.exchange_halos!(f, solver.decomp); CL.exchange_halos!(g, solver.decomp)
    CL.deriv_along!(d1, f, solver, 1, 1)
    CL.deriv_along!(d2, g, solver, 2, 1)
    e = maximum(abs(d1[gidx(solver, i, j, k)] - d2[gidx(solver, j, i, k)])
                for i in 1:24, j in 1:24, k in 1:24)
    @test e < 1e-11
end

@testset "closed-domain closures: polynomial exactness (deg ≤ 3)" begin
    solver = Solver(n_global=(32, 12, 12), L_domain=(1.0, 1.0, 1.0),
               bcs=((SlipWallBC(), SlipWallBC()), per3[2], per3[3]),
               art=ArtParams(enabled=false))
    f = CL.field(solver.decomp); df = CL.field(solver.decomp)
    fillf!(solver, f, (x, y, z) -> 1 + 2x + 3x^2 - x^3)
    CL.exchange_halos!(f, solver.decomp)
    CL.deriv_along!(df, f, solver, 1, 1); CL._scale_grad!(df, solver, 1)
    @test ferr(solver, df, (x, y, z) -> 2 + 6x - 3x^2) < 1e-10
end

@testset "filter: constants exact, Nyquist damped, parity of closures" begin
    solver = mkslv(n_global=(32, 12, 12))
    f = CL.field(solver.decomp)
    fillf!(solver, f, (x, y, z) -> 1.0)
    filter_field!(f, solver)
    @test ferr(solver, f, (x, y, z) -> 1.0) < 1e-12
    nx = solver.decomp.n_local[1]
    for k in 1:solver.decomp.n_local[3], j in 1:solver.decomp.n_local[2], i in 1:nx
        f[gidx(solver, i, j, k)] = 1.0 + 0.5 * (-1)^i        # constant + Nyquist
    end
    filter_field!(f, solver)
    dev = maximum(abs(f[gidx(solver, i, 1, 1)] - 1.0) for i in 1:nx)
    @test dev < 0.35                                      # sawtooth strongly damped
end

@testset "axisymmetric axis fold: manufactured smooth solution" begin
    # u_r = r·g(r) is an odd smooth function; d/dr through the fold must
    # match analytics at the first half-offset nodes — the sharpest probe of
    # the folded row and mirror fill.
    solver = Solver(n_global=(64, 1, 12), L_domain=(1.0, 1.0, 0.5),
               metric=CylindricalMetric(),
               bcs=((AxisBC(), SlipWallBC()), per3[2], per3[3]),
               art=ArtParams(enabled=false))
    f = CL.field(solver.decomp); df = CL.field(solver.decomp)
    fillf!(solver, f, (r, θ, z) -> r * exp(-4r^2))            # odd across the axis
    CL.exchange_halos!(f, solver.decomp)
    CL.deriv_along!(df, f, solver, 1, -1); CL._scale_grad!(df, solver, 1)
    @test ferr(solver, df, (r, θ, z) -> (1 - 8r^2) * exp(-4r^2)) < 5e-6
    fillf!(solver, f, (r, θ, z) -> exp(-4r^2))                # even across the axis
    CL.exchange_halos!(f, solver.decomp)
    CL.deriv_along!(df, f, solver, 1, 1); CL._scale_grad!(df, solver, 1)
    @test ferr(solver, df, (r, θ, z) -> -8r * exp(-4r^2)) < 2e-5  # even fold: 3rd-order, larger const
end

@testset "resolved-θ axis: antipodal pairing (local)" begin
    # f = r cosθ · e^{−4r²} = x·g is globally smooth through the axis and is
    # ODD under (r,θ)→(−r,θ) with the pairing (θ+π picks up the cos sign).
    solver = Solver(n_global=(48, 16, 1), L_domain=(1.0, 2π, 1.0),
               metric=CylindricalMetric(),
               bcs=((AxisBC(), SlipWallBC()), per3[2], per3[3]),
               art=ArtParams(enabled=false))
    f = CL.field(solver.decomp); df = CL.field(solver.decomp)
    fillf!(solver, f, (r, θ, z) -> r * cos(θ) * exp(-4r^2))
    CL.exchange_halos!(f, solver.decomp)
    CL.deriv_along!(df, f, solver, 1, -1); CL._scale_grad!(df, solver, 1)
    @test ferr(solver, df, (r, θ, z) -> cos(θ) * (1 - 8r^2) * exp(-4r^2)) < 1e-5
    # A scalar even case: f = e^{−4r²}·(1 + ½cos 2θ) maps to itself at θ+π.
    fillf!(solver, f, (r, θ, z) -> exp(-4r^2) * (1 + 0.5cos(2θ)))
    CL.exchange_halos!(f, solver.decomp)
    CL.deriv_along!(df, f, solver, 1, 1); CL._scale_grad!(df, solver, 1)
    @test ferr(solver, df, (r, θ, z) -> -8r * exp(-4r^2) * (1 + 0.5cos(2θ))) < 1e-4  # 3rd-order, larger const
end

@testset "spherical poles + origin: derivative of a smooth 3-D Gaussian" begin
    # f = e^{−4r²} is smooth at origin and poles; ∂f/∂r and (1/r)∂f/∂θ = 0.
    solver = Solver(n_global=(40, 16, 12), L_domain=(1.0, π, 2π),
               metric=SphericalMetric(),
               bcs=((OriginBC(), SlipWallBC()),
                    (PoleBC(), PoleBC()), per3[3]),
               art=ArtParams(enabled=false))
    f = CL.field(solver.decomp); df = CL.field(solver.decomp)
    fillf!(solver, f, (r, θ, φ) -> exp(-4r^2))
    CL.exchange_halos!(f, solver.decomp)
    CL.deriv_along!(df, f, solver, 1, 1); CL._scale_grad!(df, solver, 1)
    @test ferr(solver, df, (r, θ, φ) -> -8r * exp(-4r^2)) < 1e-4  # 3rd-order, larger const
    CL.deriv_along!(df, f, solver, 2, 1); CL._scale_grad!(df, solver, 2)
    @test ferr(solver, df, (r, θ, φ) -> 0.0) < 1e-8
end

@testset "rigid rotation in cylindrical: zero strain" begin
    # u_θ = Ω r ⇒ S_ij = 0 identically; probes the curvature corrections.
    solver = Solver(n_global=(32, 16, 12), L_domain=(1.0, 2π, 0.5),
               metric=CylindricalMetric(), origin=(0.2, 0.0, 0.0),
               bcs=((SlipWallBC(), SlipWallBC()), per3[2], per3[3]),
               art=ArtParams(enabled=false))
    Q = allocate_state(solver)
    initialize!(solver, Q, (r, θ, z) -> Prim(u=(0.0, 0.3r, 0.0), p=1.0, rho=1.0))
    dQ = zero(Q)
    compute_rhs!(solver, Q, dQ)
    smax = 0.0
    for k in 1:solver.decomp.n_local[3], j in 1:solver.decomp.n_local[2], i in 1:solver.decomp.n_local[1]
        I = gidx(solver, i, j, k)
        for b in 1:3, a in 1:3
            smax = max(smax, abs(0.5 * (solver.grad_u[a, b][I] + solver.grad_u[b, a][I])))
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
        solver = Solver(; kw...)
        Q = allocate_state(solver)
        initialize!(solver, Q, (x, y, z) -> Prim(u=(0, 0, 0), p=1.0, rho=1.0))
        apply_bcs!(solver, Q)
        dQ = zero(Q)
        compute_rhs!(solver, Q, dQ)
        m = maximum(abs(dQ[gidx(solver, i, j, k), c])
                    for c in 1:solver.equations.n_cons, i in 1:solver.decomp.n_local[1],
                        j in 1:solver.decomp.n_local[2], k in 1:solver.decomp.n_local[3])
        @test m < 1e-8
    end
end

@testset "EOS: conserved ↔ primitive round trip (two species)" begin
    eos = IdealMixture([IdealSpecies{Float64}("a", 1.0, 1.4),
                        IdealSpecies{Float64}("b", 0.2, 1.09)])
    solver = mkslv(n_global=(12, 12, 12), eos=eos)
    Q = allocate_state(solver)
    pr = Prim(u=(0.3, -0.1, 0.2), p=0.8, T_ion=1.7, Y=(0.35, 0.65))
    initialize!(solver, Q, (x, y, z) -> pr)
    CL.exchange_state!(Q, solver.decomp)
    CL.primitives!(solver, Q)
    I = gidx(solver, 3, 4, 5)
    @test solver.p[I] ≈ 0.8 atol = 1e-12
    @test solver.T_ion[I] ≈ 1.7 atol = 1e-12
    @test solver.u[I] ≈ 0.3 atol = 1e-12
    @test solver.Y[2][I] ≈ 0.65 atol = 1e-12
end

@testset "EquationSet owns the conserved layout" begin
    eos = IdealMixture([IdealSpecies{Float64}("a", 1.0, 1.4),
                        IdealSpecies{Float64}("b", 0.2, 1.09)])
    solver = mkslv(n_global=(12, 12, 12), eos=eos)
    @test solver.equations isa NavierStokes1T
    @test solver.equations.n_species == 2
    @test solver.equations.n_cons == 6
    @test solver.equations.i_mom == (3, 4, 5)
    @test solver.equations.i_energy == 6
    @test solver.equations.component_names ==
          ["rho_a", "rho_b", "rho_u1", "rho_u2", "rho_u3", "rho_E"]
    @test fieldtype(typeof(solver), :eos) === typeof(solver.eos)
    @test fieldtype(typeof(solver), :metric) === typeof(solver.metric)
    @test fieldtype(typeof(solver), :folds) === typeof(solver.folds)
end

@testset "tuple source terms add momentum and energy work" begin
    force = ConstantBodyForce((1.0, 2.0, -3.0))
    solver = mkslv(n_global=(12, 12, 12), sources=(force,))
    Q = allocate_state(solver)
    initialize!(solver, Q, (x, y, z) ->
        Prim(u=(0.5, 0.25, -0.1), p=1.0, rho=2.0))
    dQ = zero(Q)
    compute_rhs!(solver, Q, dQ)
    I = gidx(solver, 3, 4, 5)
    @test dQ[I, solver.equations.i_mom[1]] ≈ 2.0 atol = 1e-10
    @test dQ[I, solver.equations.i_mom[2]] ≈ 4.0 atol = 1e-10
    @test dQ[I, solver.equations.i_mom[3]] ≈ -6.0 atol = 1e-10
    @test dQ[I, solver.equations.i_energy] ≈ 2.6 atol = 1e-10
    @test typeof(solver.sources) === Tuple{typeof(force)}
end

@testset "Workspace reproduces explicit stage arrays" begin
    solver1 = mkslv(n_global=(12, 12, 12))
    solver2 = mkslv(n_global=(12, 12, 12))
    Q1 = allocate_state(solver1)
    Q2 = allocate_state(solver2)
    ic = (x, y, z) -> Prim(u=(0.1sin(x), -0.1cos(y), 0.05sin(z)),
                            p=1 + 0.02cos(x), rho=1 + 0.03sin(y))
    initialize!(solver1, Q1, ic)
    initialize!(solver2, Q2, ic)
    dQ = zero(Q1)
    du = zero(Q1)
    workspace = Workspace(Q2)
    step!(solver1, Q1, dQ, du, 1e-4)
    step!(solver2, Q2, workspace, 1e-4)
    @test Q1 == Q2
    @test dQ == workspace.dQ
    @test du == workspace.du
end

@testset "conservation: periodic RHS integrates to zero" begin
    solver = mkslv(n_global=(16, 16, 16), transport=Transport(mu0=1e-3))
    Q = allocate_state(solver)
    initialize!(solver, Q, (x, y, z) ->
        Prim(u=(0.1sin(x)cos(y), -0.1cos(x)sin(y), 0.05sin(z)),
             p=1 + 0.05cos(x)cos(z), rho=1 + 0.1sin(y)))
    dQ = zero(Q)
    compute_rhs!(solver, Q, dQ)
    for c in 1:solver.equations.n_cons
        tot = sum(dQ[gidx(solver, i, j, k), c] for i in 1:16, j in 1:16, k in 1:16)
        @test abs(tot) < 1e-8 * 16^3
    end
end

@testset "NSCBC outflow: matched uniform stream ⇒ no correction" begin
    solver = Solver(n_global=(32, 12, 12), L_domain=(1.0, 0.4, 0.4),
               bcs=((DirichletBC((x, y, z, t) -> Prim(u=(0.3, 0, 0), p=1.0, rho=1.0)),
                     NSCBCOutflowBC(pinf=1.0)), per3[2], per3[3]),
               art=ArtParams(enabled=false))
    Q = allocate_state(solver)
    initialize!(solver, Q, (x, y, z) -> Prim(u=(0.3, 0, 0), p=1.0, rho=1.0))
    apply_bcs!(solver, Q)
    dQ = zero(Q)
    compute_rhs!(solver, Q, dQ)
    nx = solver.decomp.n_local[1]
    m = maximum(abs(dQ[gidx(solver, nx, j, k), c])
                for c in 1:solver.equations.n_cons, j in 1:12, k in 1:12)
    @test m < 1e-8
end

@testset "NoSlipWallBC: adiabatic zeroes velocity, isothermal sets T_ion" begin
    for Twall in (NaN, 2.5)
        solver = Solver(n_global=(24, 12, 12), L_domain=(1.0, 0.4, 0.4),
                   bcs=((NoSlipWallBC(Twall=Twall), NoSlipWallBC(Twall=Twall)),
                        per3[2], per3[3]),
                   art=ArtParams(enabled=false))
        Q = allocate_state(solver)
        initialize!(solver, Q, (x, y, z) -> Prim(u=(0.3, -0.2, 0.1), p=1.0, T_ion=1.0))
        apply_bcs!(solver, Q)
        CL.exchange_state!(Q, solver.decomp)
        CL.primitives!(solver, Q)
        nx = solver.decomp.n_local[1]
        uw = maximum(abs(solver.u[gidx(solver, i, j, k)]) + abs(solver.v[gidx(solver, i, j, k)]) +
                     abs(solver.w[gidx(solver, i, j, k)])
                     for i in (1, nx), j in 1:12, k in 1:12)   # both walls
        @test uw < 1e-12
        if !isnan(Twall)                       # isothermal wall holds Twall
            e = maximum(abs(solver.T_ion[gidx(solver, i, j, k)] - Twall)
                        for i in (1, nx), j in 1:12, k in 1:12)
            @test e < 1e-12
        end
    end
end

@testset "ExtrapolationBC: copies the adjacent interior plane" begin
    solver = Solver(n_global=(24, 12, 12), L_domain=(1.0, 0.4, 0.4),
               bcs=((ExtrapolationBC(), ExtrapolationBC()), per3[2], per3[3]),
               art=ArtParams(enabled=false))
    Q = allocate_state(solver)
    initialize!(solver, Q, (x, y, z) -> Prim(u=(0.2 + x, 0, 0), p=1 + 0.1x, rho=1 + 0.3x))
    apply_bcs!(solver, Q)
    nx = solver.decomp.n_local[1]
    d = 0.0
    for c in 1:solver.equations.n_cons, j in 1:12, k in 1:12
        d = max(d, abs(Q[gidx(solver, 1, j, k), c]  - Q[gidx(solver, 2, j, k), c]))
        d = max(d, abs(Q[gidx(solver, nx, j, k), c] - Q[gidx(solver, nx-1, j, k), c]))
    end
    @test d == 0.0
    # a uniform state must be untouched by the extrapolation
    Q2 = allocate_state(solver)
    initialize!(solver, Q2, (x, y, z) -> Prim(u=(0.2, 0, 0), p=1.0, rho=1.0))
    ref = copy(Q2)
    apply_bcs!(solver, Q2)
    @test Q2 == ref
end

@testset "NSCBC inflow: matched uniform stream ⇒ no correction" begin
    # Mirrors the outflow test. This whole method was previously never
    # compiled, so nothing in it — including the transverse terms — had ever
    # been executed, let alone checked.
    uin = (0.3, 0.0, 0.0)
    solver = Solver(n_global=(32, 12, 12), L_domain=(1.0, 0.4, 0.4),
               bcs=((NSCBCInflowBC(u=uin, T_ion=1.0), NSCBCOutflowBC(pinf=1.0)),
                    per3[2], per3[3]),
               eos=single_species(gamma=1.4, R=1.0),
               art=ArtParams(enabled=false))
    Q = allocate_state(solver)
    initialize!(solver, Q, (x, y, z) -> Prim(u=uin, p=1.0, T_ion=1.0))
    apply_bcs!(solver, Q)
    dQ = zero(Q)
    compute_rhs!(solver, Q, dQ)
    m = maximum(abs(dQ[gidx(solver, 1, j, k), c])
                for c in 1:solver.equations.n_cons, j in 1:12, k in 1:12)
    @test m < 1e-8
    # and a mismatched stream must produce a non-trivial correction
    Q2 = allocate_state(solver)
    initialize!(solver, Q2, (x, y, z) -> Prim(u=(0.15, 0, 0), p=1.0, T_ion=1.0))
    apply_bcs!(solver, Q2)
    dQ2 = zero(Q2)
    compute_rhs!(solver, Q2, dQ2)
    m2 = maximum(abs(dQ2[gidx(solver, 1, j, k), c])
                 for c in 1:solver.equations.n_cons, j in 1:12, k in 1:12)
    @test m2 > 1e-3
end

@testset "multi-species artificial diffusivity responds to Y gradients" begin
    # artificial.jl computes per-species D* only when n_species > 1; that branch was
    # never executed, since the only multi-species test disables art.
    eos = IdealMixture([IdealSpecies{Float64}("a", 1.0, 1.4),
                        IdealSpecies{Float64}("b", 0.2, 1.09)])
    solver = Solver(n_global=(64, 12, 12), L_domain=(1.0, 0.2, 0.2), bcs=per3, eos=eos,
               art=ArtParams(enabled=true))
    Q = allocate_state(solver)
    dQ = zero(Q)
    # smooth composition: D* must stay tiny
    initialize!(solver, Q, (x, y, z) -> Prim(Y=(0.5 + 0.1sin(2π * x), 0.5 - 0.1sin(2π * x)),
                                        p=1.0, rho=1.0))
    compute_rhs!(solver, Q, dQ)
    smooth_max = max(maximum(solver.D_art[1]), maximum(solver.D_art[2]))
    # sharp interface: D* must switch on
    initialize!(solver, Q, (x, y, z) -> begin
        θ = tanh_blend(x, 0.5, 0.01)
        Prim(Y=(1 - θ, θ), p=1.0, rho=1.0)
    end)
    compute_rhs!(solver, Q, dQ)
    sharp_max = max(maximum(solver.D_art[1]), maximum(solver.D_art[2]))
    @test sharp_max > 100 * max(smooth_max, 1e-14)
    @test all(isfinite, solver.D_art[1]) && all(isfinite, solver.D_art[2])
end

@testset "dt_report agrees with compute_dt and names the limiter" begin
    solver = mkslv(n_global=(16, 16, 16), transport=Transport(mu0=1e-3))
    Q = allocate_state(solver)
    initialize!(solver, Q, (x, y, z) -> Prim(u=(0.2sin(x), 0, 0), p=1 + 0.1cos(y), rho=1.0))
    r = dt_report(solver, Q)
    @test r.dt ≈ compute_dt(solver, Q) rtol = 1e-12
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
        solver = Solver(n_global=(64, 1, 1), L_domain=(1.0, 1.0, 1.0), metric=metric,
                   origin=(0.5, π / 2, 0.0),
                   bcs=((SlipWallBC(), SlipWallBC()), per3[2], per3[3]),
                   art=ArtParams(enabled=false))
        Q = allocate_state(solver)
        initialize!(solver, Q, (r, θ, φ) -> Prim(u=(0.0, uang, uang), p=1.0, rho=1.0))
        solver, Q
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

@testset "StepControl: floors, positivity, and the default no-op" begin
    # The floor logic is pure, so it is checked directly rather than by
    # constructing a run that fails in each of five ways.
    c = StepControl()
    ok(dt, ρ; seen=1.0, ctl=c) = CL.check_step(ctl, dt, ρ, seen, 1, 0.0, 0.5)
    @test ok(1e-3, 1.0) === nothing
    @test ok(NaN, 1.0).reason === :nonfinite
    @test ok(Inf, 1.0).reason === :nonfinite
    @test ok(1e-3, -1e-9).reason === :negative_density
    @test ok(1e-3, 0.0).reason === :negative_density
    @test ok(1e-50, 1.0).reason === :planck            # below the Planck failsafe
    @test ok(1e-40, 1.0).reason === :dt_collapse       # above it, but collapsed
    # The Planck floor is unconditional: it fires even with every user floor off.
    bare = StepControl(dt_min=0.0, dt_min_ratio=0.0)
    @test ok(1e-50, 1.0; ctl=bare).reason === :planck
    @test ok(1e-40, 1.0; ctl=bare) === nothing         # relative floor disabled
    # An explicit absolute floor.
    @test ok(1e-9, 1.0; ctl=StepControl(dt_min=1e-6)).reason === :dt_min
    # Ordering: a non-finite dt is reported as such, not as a floor breach.
    @test ok(NaN, -1.0).reason === :nonfinite
    # Message carries the state that localizes the failure.
    e = ok(1e-50, 1.0)
    msg = sprint(showerror, e)
    @test occursin("SolverFailure(:planck)", msg)
    @test occursin("StepControl(retries", msg)

    # With prediction off (the default) the chosen step is exactly compute_dt,
    # so adding all of this changed nothing for a healthy run.
    solver = mkslv(n_global=(16, 16, 16))
    Q = allocate_state(solver)
    initialize!(solver, Q, (x, y, z) -> Prim(u=(0.2sin(x), 0, 0), p=1.0, rho=1.0))
    rate, ρmin = max_rate(solver, Q)
    @test ρmin ≈ 1.0 atol = 1e-12
    @test CL.predicted_dt(solver, StepControl(), rate) == compute_dt(solver, Q)
    # Prediction only ever shortens the step, and only when the rate is rising.
    solver.rate_prev = 0.5 * rate
    @test CL.predicted_dt(solver, StepControl(predict=1.0), rate) < compute_dt(solver, Q)
    solver.rate_prev = 2.0 * rate              # falling rate: no extrapolation
    @test CL.predicted_dt(solver, StepControl(predict=1.0), rate) == compute_dt(solver, Q)
    # Growth capping is relative to the previous accepted step.
    solver.rate_prev = 0.0
    solver.dt_prev = 1e-9
    @test CL.predicted_dt(solver, StepControl(max_growth=1.5), rate) ≈ 1.5e-9 rtol = 1e-14
end

@testset "run!: failure is raised, and recoverable with retries" begin
    # Noh at nu=1. Two distinct behaviours, and the distinction is the whole
    # point of the rollback:
    #
    #   cfl = 0.3 degrades GRADUALLY — the density undershoot grows for ~150
    #   steps before positivity goes — so the run must fail loudly rather than
    #   grind, and rollback cannot save it because the savepoint is already
    #   damaged by the time anything notices.
    #
    #   cfl = 0.9 fails ABRUPTLY in the startup transient, which is what a user
    #   who guessed a CFL actually hits, and rolling back past it with a halved
    #   CFL recovers the correct answer.
    γ = 5 / 3; p0 = 1e-4; tfin = 0.6; N = 400
    build(cfl, control) = begin
        inflow = DirichletBC((x, y, z, t) -> begin
            ρ, u, _ = noh_exact(x, isfinite(t) ? t : 0.0, 1, γ)
            Prim(rho=ρ, u=(u, 0.0, 0.0), p=p0)
        end)
        prob = Problem(eos=single_species(gamma=γ, R=1.0), transport=Transport(mu0=0.0),
                       domain=((0.0, 1.0), (0.0, 1 / N), (0.0, 1 / N)),
                       bcs=((SlipWallBC(), inflow), per3[2], per3[3]),
                       ic=(x, y, z) -> Prim(rho=1.0, u=(-1.0, 0.0, 0.0), p=p0))
        setup(prob, Numerics(n_global=(N, 1, 1), art=ArtParams(enabled=true),
                             cfl=cfl, control=control, filter_interval=1))
    end
    s1, Q1 = build(0.3, StepControl())
    err = nothing
    try
        run!(s1, Q1; tfinal=tfin, nmax=20_000)
    catch e
        err = e
    end
    @test err isa SolverFailure
    @test err.reason in (:negative_density, :dt_collapse)
    @test s1.step < 20_000                     # it stopped early, it did not grind

    s2, Q2 = build(0.9, StepControl(retries=5, savepoint_interval=20))
    run!(s2, Q2; tfinal=tfin, nmax=20_000)
    @test s2.t ≈ tfin rtol = 1e-9
    @test s2.cfl < 0.9                         # it backed off, and says by how much
    CL.exchange_state!(Q2, s2.decomp); CL.primitives!(s2, Q2)
    # Post-shock plateau, sampled between the wall-heating layer and the shock
    # at x = (γ−1)t/2 = 0.2 — a window that straddles the shock would average
    # the answer with the undisturbed inflow and pass for the wrong reason.
    core = [i for i in 1:N if 0.06 <= xcoord(s2, 1, i) <= 0.14]
    plateau = sum(s2.rho[gidx(s2, i, 1, 1)] for i in core) / length(core)
    @test plateau ≈ 4.0 rtol = 0.05            # exact Noh plateau for nu = 1
    # The backoff compounds: successive retries must keep halving, not keep
    # re-applying one factor to the same starting CFL.
    @test s2.cfl <= 0.9 * 0.5 + 1e-12
end

@testset "NASA-9 mixture reduces exactly to the ideal mixture" begin
    # The strongest available check on the polynomial machinery: with only the
    # constant term a3 populated, cp is temperature-independent and every
    # quantity — including the Newton inversion of e(T) — must reproduce the
    # closed-form ideal-gas answer to round-off. A transcription error in the
    # enthalpy integral, the cv, or the sound speed shows up here.
    γ, R = 1.4, 1.0
    cp = γ * R / (γ - 1)
    ideal = single_species(gamma=γ, R=R)
    poly = Nasa9Mixture([nasa9_constant_cp("gas", R, cp)])
    @test nspecies(poly) == 1
    for T in (0.3, 1.0, 7.5, 300.0)
        @test CL.species_cp(poly, 1, T) ≈ cp rtol = 1e-14
        @test CL.species_enthalpy(poly, 1, T) ≈ cp * T rtol = 1e-14
        @test CL.species_energy(poly, 1, T) ≈ (cp - R) * T rtol = 1e-14
    end
    pr = Prim(u=(0.3, -0.1, 0.2), p=0.8, T_ion=1.7)
    qi = conserved_from_prim(ideal, pr)
    qp = conserved_from_prim(poly, pr)
    @test all(qi .≈ qp)
    # ... and through the full primitives path, which is where the Newton
    # inversion actually runs.
    for eos in (ideal, poly)
        solver = mkslv(n_global=(12, 12, 12), eos=eos)
        Q = allocate_state(solver)
        initialize!(solver, Q, (x, y, z) -> Prim(u=(0.3, 0, 0), p=0.8, T_ion=1.7))
        CL.exchange_state!(Q, solver.decomp)
        CL.primitives!(solver, Q)
        I = gidx(solver, 3, 4, 5)
        @test solver.p[I] ≈ 0.8 rtol = 1e-12
        @test solver.T_ion[I] ≈ 1.7 rtol = 1e-12
        @test solver.c[I] ≈ sqrt(γ * R * 1.7) rtol = 1e-12
        @test solver.cp_mix[I] ≈ cp rtol = 1e-12
    end
end

@testset "NASA-9: thermodynamic consistency of a varying cp" begin
    # A genuinely temperature-dependent coefficient set, checked against the
    # two identities that must hold whatever the coefficients are:
    # dh/dT = cp, and the Newton inversion is the inverse of e(T).
    R = 287.0
    sp = Nasa9Species{Float64}(name="fake", R=R,
                               a=(1.2e4, -50.0, 3.6, 6.0e-4, -1.0e-7, 1.0e-11,
                                  -4.0e-16), b1=-1.0e3)
    eos = Nasa9Mixture([sp, nasa9_constant_cp("inert", 200.0, 900.0)];
                       T_guess=500.0)
    for T in (250.0, 800.0, 2500.0)
        δ = 1e-4 * T
        dh = (CL.species_enthalpy(eos, 1, T + δ) -
              CL.species_enthalpy(eos, 1, T - δ)) / 2δ
        @test dh ≈ CL.species_cp(eos, 1, T) rtol = 1e-7
    end
    # cp really does vary — otherwise the identity above is vacuous — and cv
    # stays positive across the range, which the Newton solve relies on.
    @test CL.species_cp(eos, 1, 2500.0) / CL.species_cp(eos, 1, 300.0) > 1.2
    @test all(CL.species_cp(eos, 1, T) > R for T in 250.0:50.0:2500.0)
    # e(T) round trip through the Newton solve, at three compositions.
    for Y in ((1.0, 0.0), (0.5, 0.5), (0.2, 0.8))
        for T in (250.0, 800.0, 2500.0)
            e = sum(Y[k] * CL.species_energy(eos, k, T) for k in 1:2)
            @test CL.mixture_temperature(eos, e, k -> Y[k]) ≈ T rtol = 1e-12
        end
    end
    # And end to end: a state initialized from (p, T_ion) recovers both.
    solver = mkslv(n_global=(12, 12, 12), eos=eos)
    Q = allocate_state(solver)
    initialize!(solver, Q, (x, y, z) -> Prim(Y=(0.35, 0.65), u=(120.0, 0, 0),
                                             p=2.5e5, T_ion=1400.0))
    CL.exchange_state!(Q, solver.decomp)
    CL.primitives!(solver, Q)
    I = gidx(solver, 3, 4, 5)
    @test solver.p[I] ≈ 2.5e5 rtol = 1e-10
    @test solver.T_ion[I] ≈ 1400.0 rtol = 1e-10
    @test solver.Y[2][I] ≈ 0.65 rtol = 1e-12
    # A step must stay finite: the Newton solve runs inside every RK stage.
    run!(solver, Q; tfinal=1e9, nmax=3)
    @test all(isfinite, Q)
end

@testset "NASA CEA reader: intervals, molar mass, and energy reference" begin
    he, co2 = read_nasa9(["He", "CO2"])
    @test (he.name, co2.name) == ("He", "CO2")
    @test co2.R ≈ 8.31446261815324 / 44.0095e-3 rtol = 1e-14
    @test length(co2.intervals) == 3
    @test [(item.Tmin, item.Tmax) for item in co2.intervals] ==
          [(200.0, 1000.0), (1000.0, 6000.0), (6000.0, 20000.0)]

    sensible = Nasa9Mixture([co2])
    @test CL.species_enthalpy(sensible, 1, 298.15) ≈ 0.0 atol = 1e-6
    @test CL.species_cp(sensible, 1, 300.0) ≈ 845.7241586606974 rtol = 1e-13
    for i in 1:2
        T_join = co2.intervals[i].Tmax
        left, right = co2.intervals[i], co2.intervals[i + 1]
        @test CL._nasa9_cp_over_R(left, T_join) ≈
              CL._nasa9_cp_over_R(right, T_join) rtol = 5e-7
        @test CL._nasa9_h_over_R(left, T_join) ≈
              CL._nasa9_h_over_R(right, T_join) rtol = 5e-7
    end

    co2_formation = read_nasa9("CO2"; reference=:formation)
    formation = Nasa9Mixture([co2_formation])
    h298_molar = CL.species_enthalpy(formation, 1, 298.15) * 44.0095e-3
    @test h298_molar ≈ -393510.0 atol = 5.0
    @test CL.species_cp(formation, 1, 300.0) == CL.species_cp(sensible, 1, 300.0)
    for eos in (sensible, formation), T in (220.0, 300.0, 999.0, 1001.0,
                                                    1500.0, 5999.0, 6001.0, 10000.0)
        e = CL.species_energy(eos, 1, T)
        @test CL.mixture_temperature(eos, e, _ -> 1.0) ≈ T rtol = 2e-13
    end

    Y = (0.3, 0.7)
    for reference in (:sensible, :formation)
        mixture = Nasa9Mixture(read_nasa9(["He", "CO2"]; reference))
        for T in (300.0, 1400.0, 7000.0)
            e = sum(Y[k] * CL.species_energy(mixture, k, T) for k in 1:2)
            @test CL.mixture_temperature(mixture, e, k -> Y[k]) ≈ T rtol = 2e-13
        end
    end
    # The late Air record exercises the CEA file's product/reactant separators.
    @test read_nasa9("Air").name == "Air"
    @test CL.species_names(Nasa9Mixture(read_nasa9(["CO2", "He"]))) == ["CO2", "He"]
    @test_throws ArgumentError read_nasa9("CO2"; reference=:unknown)

    # Zero-interval records list a heat of formation and no fit, and are three
    # lines rather than two. Miscounting them desynchronizes the scan against
    # the file instead of failing locally, so pin both the record that has to be
    # rejected and a real species that only parses if the skip is right.
    @test_throws "no temperature intervals" read_nasa9("n-Butanol")
    @test length(read_nasa9("Jet-A(g)").intervals) == 2
    # The message matters: a desynchronized scan also throws ArgumentError, from
    # misreading a coefficient line as a species header.
    @test_throws "NASA CEA species not found" read_nasa9("not-a-CEA-species")

    # The two CEA records whose interval joins are loosest; the tolerance is
    # sized to admit them and still reject a mistranscribed coefficient.
    @test all(nspecies(Nasa9Mixture(read_nasa9([name]))) == 1
              for name in ("ALOCL", "SnCL2"))
    good = Nasa9Interval{Float64}(200.0, 1000.0,
                                  (0.0, 0.0, 3.5, 0.0, 0.0, 0.0, 0.0), 0.0)
    bad = Nasa9Interval{Float64}(1000.0, 6000.0,
                                 (0.0, 0.0, 4.5, 0.0, 0.0, 0.0, 0.0), 0.0)
    @test_throws ArgumentError Nasa9Species{Float64}("broken", 287.0, [good, bad])
end

@testset "StiffenedGas: perfect-gas limit, and a real liquid" begin
    # p_inf = 0 must reproduce a perfect gas exactly, which pins the algebra;
    # then a water-like parameter set exercises the branch that matters.
    γ = 1.4; R = 1.0; cv = R / (γ - 1)
    sg = StiffenedGas(gamma=γ, p_inf=0.0, cv=cv, name="gas")
    @test nspecies(sg) == 1
    pr = Prim(u=(0.3, -0.1, 0.2), p=0.8, T_ion=1.7)
    @test all(conserved_from_prim(single_species(gamma=γ, R=R), pr) .≈
              conserved_from_prim(sg, pr))
    solver = mkslv(n_global=(12, 12, 12), eos=sg)
    Q = allocate_state(solver)
    initialize!(solver, Q, (x, y, z) -> pr)
    CL.exchange_state!(Q, solver.decomp); CL.primitives!(solver, Q)
    I = gidx(solver, 3, 4, 5)
    @test solver.p[I] ≈ 0.8 rtol = 1e-12
    @test solver.T_ion[I] ≈ 1.7 rtol = 1e-12
    @test solver.c[I] ≈ sqrt(γ * R * 1.7) rtol = 1e-12
    @test CL.eos_phi(sg, 1.0, 0.8, 1.7, γ * cv) ≈ cv / R rtol = 1e-13

    # Water: γ = 4.4, p∞ = 6e8 Pa. The point of the model is that the sound
    # speed is set by p∞, not by p, so it stays near 1500 m/s at 1 atm where a
    # perfect gas would give a few hundred.
    water = StiffenedGas(gamma=4.4, p_inf=6.0e8, cv=1816.0, name="water")
    s2 = Solver(n_global=(12, 12, 12), L_domain=(1.0, 1.0, 1.0), bcs=per3,
                eos=water, art=ArtParams(enabled=false))
    Q2 = allocate_state(s2)
    initialize!(s2, Q2, (x, y, z) -> Prim(u=(0, 0, 0), p=101325.0, rho=1000.0))
    CL.exchange_state!(Q2, s2.decomp); CL.primitives!(s2, Q2)
    J = gidx(s2, 3, 4, 5)
    @test s2.p[J] ≈ 101325.0 rtol = 1e-9
    @test 1400 < s2.c[J] < 1700                      # c = sqrt(γ(p+p∞)/ρ)
    @test s2.c[J] ≈ sqrt(4.4 * (101325.0 + 6.0e8) / 1000.0) rtol = 1e-12
    # Uniform state ⇒ zero RHS, the same freestream statement made for every
    # other configuration in this suite.
    apply_bcs!(s2, Q2)
    dQ2 = zero(Q2)
    compute_rhs!(s2, Q2, dQ2)
    @test maximum(abs, dQ2) < 1e-8 * 101325.0
    # An acoustic pulse stays finite and does not leave the stiffened branch.
    initialize!(s2, Q2, (x, y, z) ->
        Prim(u=(0, 0, 0), p=101325.0 * (1 + 0.01sin(2π * x)), rho=1000.0))
    run!(s2, Q2; tfinal=1e9, nmax=5)
    CL.primitives!(s2, Q2)
    @test all(isfinite, Q2)
    @test minimum(s2.rho[gidx(s2, i, j, k)] for i in 1:12, j in 1:12, k in 1:12) > 0
end

@testset "diagnostics: quadrature, plane averages, mixing measures" begin
    # The quadrature is the load-bearing part: every mixing number is a ratio of
    # two of these integrals, so a wrong edge weight biases θ and W silently
    # rather than failing. Cartesian must be exact; curvilinear is O(h²) at a
    # node-centered edge by construction (see the note in diagnostics.jl).
    solver = Solver(n_global=(24, 16, 12), L_domain=(2.0, 1.0, 0.5),
                    bcs=((SlipWallBC(), SlipWallBC()), per3[2], per3[3]),
                    art=ArtParams(enabled=false))
    @test domain_volume(solver) ≈ 1.0 atol = 1e-12          # 2.0 × 1.0 × 0.5
    ones_f = CL.field(solver.decomp); fill!(ones_f, 1.0)
    @test volume_integral(solver, ones_f) ≈ 1.0 atol = 1e-12
    @test volume_average(solver, ones_f) ≈ 1.0 atol = 1e-12
    # ∫x dV over x ∈ [0,2] with unit transverse area = 1.0 (trapezoid is exact
    # for a linear integrand, which is what the half edge weight buys).
    lin = CL.field(solver.decomp)
    fillf!(solver, lin, (x, y, z) -> x)
    @test volume_integral(solver, lin) ≈ 1.0 atol = 1e-12
    # A plane average of a function of x alone returns that function, and the
    # spacing profile sums to the extent.
    prof = plane_profile(solver, lin, 1)
    xs = profile_coordinate(solver, 1)
    @test length(prof) == 24
    @test maximum(abs, prof .- xs) < 1e-12
    @test sum(profile_spacing(solver, 1)) ≈ 2.0 atol = 1e-12

    # Cylindrical with the axis fold: half-offset cells carry full weight, so
    # the error is the O(h²) edge term at the outer wall only.
    cyl = Solver(n_global=(64, 1, 12), L_domain=(1.0, 1.0, 1.0),
                 metric=CylindricalMetric(),
                 bcs=((AxisBC(), SlipWallBC()), per3[2], per3[3]),
                 art=ArtParams(enabled=false))
    ones_c = CL.field(cyl.decomp); fill!(ones_c, 1.0)
    h = cyl.h[1]
    @test volume_integral(cyl, ones_c) ≈ 0.5 rtol = 1e-3    # ∫r dr dθ dz, θ collapsed
    @test volume_integral(cyl, ones_c) - 0.5 < h^2          # the edge term, not more

    # Mixing measures against states whose answers are definitional.
    eos = IdealMixture([IdealSpecies{Float64}("a", 1.0, 1.4),
                        IdealSpecies{Float64}("b", 1.0, 1.4)])
    mk(ic) = begin
        s = Solver(n_global=(64, 12, 12), L_domain=(1.0, 0.2, 0.2), bcs=per3,
                   eos=eos, art=ArtParams(enabled=false))
        Q = allocate_state(s)
        initialize!(s, Q, ic)
        CL.exchange_state!(Q, s.decomp); CL.primitives!(s, Q)
        s, Q
    end
    # The pair W and θ exists precisely to separate stirring from mixing, and
    # these three states are the demonstration: the first two have IDENTICAL
    # mix width and opposite θ.
    #
    # (a) uniformly mixed everywhere: W is the full extent, θ = 1.
    s1, Q1 = mk((x, y, z) -> Prim(Y=(0.5, 0.5), p=1.0, rho=1.0))
    @test mix_width(s1, Q1) ≈ 1.0 atol = 1e-12
    @test molecular_mixing(s1, Q1) ≈ 1.0 atol = 1e-12
    # (b) stirred but not mixed: each plane is half pure a and half pure b, so
    # ⟨Y_a⟩⟨Y_b⟩ is unchanged from (a) but ⟨Y_a Y_b⟩ vanishes pointwise.
    s2, Q2 = mk((x, y, z) -> Prim(Y=(y < 0.1 ? 1.0 : 0.0, y < 0.1 ? 0.0 : 1.0),
                                  p=1.0, rho=1.0))
    @test mix_width(s2, Q2) ≈ 1.0 atol = 1e-12
    @test molecular_mixing(s2, Q2) < 1e-12
    # (c) an x-only interface: nothing varies within a plane, so θ is 1 by
    # construction and W collapses onto the interface.
    s3, Q3 = mk((x, y, z) -> begin
        θ = tanh_blend(x, 0.5, 1 / 64)
        Prim(Y=(1 - θ, θ), p=1.0, rho=1.0)
    end)
    @test mix_width(s3, Q3) < 0.05
    @test molecular_mixing(s3, Q3) ≈ 1.0 atol = 1e-12
    # The PDF of a segregated field piles up at 0 and 1 and integrates to 1.
    centers, pdf = species_pdf(s2, 1; nbins=20)
    @test sum(pdf) * (centers[2] - centers[1]) ≈ 1.0 atol = 1e-10
    @test pdf[1] + pdf[end] > 0.99 * sum(pdf)
    _, pdf1 = species_pdf(s1, 1; nbins=20)
    @test pdf1[11] > 0.99 * sum(pdf1)                # all mass in the Y=0.5 bin

    # TKE removes the plane mean, so a uniform stream carries none; a
    # transverse-varying velocity does.
    s4, Q4 = mk((x, y, z) -> Prim(Y=(1.0, 0.0), u=(0.7, 0, 0), p=1.0, rho=1.0))
    @test turbulent_kinetic_energy(s4, Q4) < 1e-20
    @test maximum(abs, tke_profile(s4, Q4)) < 1e-20
    s5, Q5 = mk((x, y, z) -> Prim(Y=(1.0, 0.0), u=(0.7, 0.3sin(2π * y / 0.2), 0),
                                  p=1.0, rho=1.0))
    @test turbulent_kinetic_energy(s5, Q5) > 1e-3
    # Dissipation is a sink: zero for a uniform state, positive under shear.
    @test abs(dissipation_rate(s4, Q4)) < 1e-20
    s6, Q6 = mk((x, y, z) -> Prim(Y=(1.0, 0.0), u=(0.0, 0.3sin(2π * x), 0),
                                  p=1.0, rho=1.0))
    s6.transport = Transport(mu0=1e-2)
    @test dissipation_rate(s6, Q6) > 0
end

@testset "save_vtk writes a readable .pvtr/.vtr pair" begin
    solver = mkslv(n_global=(12, 12, 12))
    Q = allocate_state(solver)
    initialize!(solver, Q, (x, y, z) -> Prim(u=(sin(x), 0, 0), p=1 + 0.1cos(y), rho=1.0))
    save_vtk(solver, Q, "test_vtk")
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

@testset "callbacks: triggers, dt landing, composition, termination" begin
    wall3 = ((SlipWallBC(), SlipWallBC()), (PeriodicBC(), PeriodicBC()),
             (PeriodicBC(), PeriodicBC()))
    mkrun() = begin
        solver = Solver(bcs=wall3, n_global=(16, 12, 12), L_domain=(1.0, 1.0, 1.0),
                        art=ArtParams(enabled=false), cfl=0.4)
        Q = allocate_state(solver)
        initialize!(solver, Q, (x, y, z) -> Prim(u=(0.2, 0, 0),
                                                 p=1 + 0.1exp(-40(x - 0.5)^2), rho=1.0))
        solver, Q
    end

    # AtTime must be landed on exactly, not overshot: dt is clipped inside run!,
    # which is the whole reason the trigger cannot be written by a caller.
    solver, Q = mkrun()
    hits = Float64[]
    targets = [0.02, 0.05, 0.11]
    run!(solver, Q; tfinal=0.2,
         callback=Callback(AtTime(targets), (s, _) -> (push!(hits, s.t); nothing)))
    @test length(hits) == 3
    @test hits ≈ targets rtol = 1e-14
    @test solver.t ≈ 0.2 rtol = 1e-14

    # Unsorted input is ordered, and a scalar is accepted.
    @test AtTime([0.3, 0.1, 0.2]).times == [0.1, 0.2, 0.3]
    @test AtTime(0.5).times == [0.5]

    # A time already behind the solver must not drive dt to zero and stall.
    solver, Q = mkrun()
    late = Float64[]
    run!(solver, Q; tfinal=0.01, nmax=20,
         callback=Callback(AtTime(-1.0), (s, _) -> (push!(late, s.t); nothing)))
    @test length(late) == 1 && solver.step > 0

    # EveryStep, and a bare function still runs every step with its return
    # value ignored — returning true must NOT stop the run.
    solver, Q = mkrun()
    steps = Int[]
    every = 0
    run!(solver, Q; tfinal=1e9, nmax=10,
         callback=(Callback(EveryStep(3), (s, _) -> (push!(steps, s.step); nothing)),
                   (s, _) -> (every += 1; true)))
    @test steps == [3, 6, 9]
    @test every == 10 && solver.step == 10

    # WhenState fires once by default and repeatedly with once=false.
    for (once, want) in ((true, 1), (false, 3))
        solver, Q = mkrun()
        fires = 0
        run!(solver, Q; tfinal=1e9, nmax=5,
             callback=Callback(WhenState((s, _) -> s.step >= 3; once=once),
                               (_, _) -> (fires += 1; nothing)))
        @test fires == want
    end

    # An effect returning true ends the run after that step.
    solver, Q = mkrun()
    run!(solver, Q; tfinal=1e9, nmax=1000,
         callback=Callback(WhenState((s, _) -> s.step >= 4), (_, _) -> true))
    @test solver.step == 4
end

@testset "SwitchableBC: transparent before the switch, forwards after" begin
    wall3 = ((SlipWallBC(), SlipWallBC()), (PeriodicBC(), PeriodicBC()),
             (PeriodicBC(), PeriodicBC()))
    @test_throws ArgumentError SwitchableBC(SlipWallBC(), PeriodicBC())
    @test_throws ArgumentError SwitchableBC(AxisBC(), SlipWallBC())
    @test_throws ArgumentError SwitchableBC(SlipWallBC(), OriginBC())
    @test CL.isperiodic(SwitchableBC(PeriodicBC(), PeriodicBC()))
    @test !CL.isperiodic(SwitchableBC(SlipWallBC(), ExtrapolationBC()))

    mkrun(xbc) = begin
        solver = Solver(bcs=(xbc, (PeriodicBC(), PeriodicBC()),
                             (PeriodicBC(), PeriodicBC())),
                        n_global=(16, 12, 12), L_domain=(1.0, 1.0, 1.0),
                        art=ArtParams(enabled=false), cfl=0.4)
        Q = allocate_state(solver)
        initialize!(solver, Q, (x, y, z) -> Prim(u=(0.2, 0, 0),
                                                 p=1 + 0.1exp(-40(x - 0.5)^2), rho=1.0))
        solver, Q
    end

    # Unswitched, the wrapper must be bit-identical to the condition it wraps —
    # that is what makes it safe to leave in a problem specification.
    s_ref, Q_ref = mkrun((SlipWallBC(), SlipWallBC()))
    run!(s_ref, Q_ref; tfinal=1e9, nmax=12)
    sw = (SwitchableBC(SlipWallBC(), ExtrapolationBC()),
          SwitchableBC(SlipWallBC(), ExtrapolationBC()))
    s_wrap, Q_wrap = mkrun(sw)
    run!(s_wrap, Q_wrap; tfinal=1e9, nmax=12)
    @test Q_wrap == Q_ref
    @test !switched(sw[1])

    # Switched by a callback, it must actually take the other branch: the state
    # diverges from the unswitched reference and stays finite.
    sw2 = (SwitchableBC(SlipWallBC(), ExtrapolationBC()),
           SwitchableBC(SlipWallBC(), ExtrapolationBC()))
    s_sw, Q_sw = mkrun(sw2)
    run!(s_sw, Q_sw; tfinal=1e9, nmax=12,
         callback=Callback(WhenState((s, _) -> s.step >= 4),
                           (_, _) -> (switch!.(sw2); nothing)))
    @test all(switched, sw2)
    @test all(isfinite, Q_sw)
    @test Q_sw != Q_ref
    @test switch!(sw2[1]) === sw2[1]      # idempotent
end

@testset "checkpoint round trip" begin
    solver = mkslv(n_global=(12, 12, 12))
    Q = allocate_state(solver)
    initialize!(solver, Q, (x, y, z) -> Prim(u=(sin(x), 0, 0), p=1 + 0.1cos(y), rho=1.0))
    solver.t = 0.37; solver.step = 42
    save_checkpoint(solver, Q, "test_ckpt")
    Q2 = allocate_state(solver); solver.t = 0.0; solver.step = 0
    load_checkpoint!(solver, Q2, "test_ckpt")
    @test solver.t == 0.37 && solver.step == 42
    @test all(Q2[gidx(solver, i, j, k), c] == Q[gidx(solver, i, j, k), c]
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
        solver, Q = build()
        run!(solver, Q; tfinal=1e9, nmax=3)
        bad = any(!isfinite(Q[gidx(solver, i, j, k), c])
                  for c in 1:solver.equations.n_cons, i in 1:solver.decomp.n_local[1],
                      j in 1:solver.decomp.n_local[2], k in 1:solver.decomp.n_local[3])
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
    solver, Q = setup(prob, Numerics(n_global=(Nx, 1, 1), art=ArtParams(enabled=true),
                                cfl=0.4, filter_interval=1))
    run!(solver, Q; tfinal=tfin, nmax=100_000)
    CL.exchange_state!(Q, solver.decomp)
    CL.primitives!(solver, Q)

    pstar, ustar, cL, cR = exact_riemann_star(ρL, uL, pL, ρR, uR, pR, γ)
    # Star state is the canonical Sod result (Toro): p* ≈ 0.30313, u* ≈ 0.92745.
    @test pstar ≈ 0.30313 atol = 1e-4
    @test ustar ≈ 0.92745 atol = 1e-4

    nx = solver.decomp.n_local[1]
    eρ = eu = ep = 0.0
    for i in 1:nx
        I = gidx(solver, i, 1, 1); x = xcoord(solver, 1, i)
        r, u, p = exact_riemann_sample((x - x0) / tfin, ρL, uL, pL, ρR, uR, pR,
                                       γ, pstar, ustar, cL, cR)
        eρ += abs(solver.rho[I] - r); eu += abs(solver.u[I] - u); ep += abs(solver.p[I] - p)
    end
    eρ /= nx; eu /= nx; ep /= nx
    @test eρ < 2e-2      # measured ≈ 2.9e-3
    @test eu < 2e-2      # measured ≈ 4.6e-3
    @test ep < 1e-2      # measured ≈ 2.5e-3
end

println("serial tests complete")
