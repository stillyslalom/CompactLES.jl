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

@testset "concise frontend displays" begin
    prob = Problem(name="display test",
                   domain=((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)), bcs=per3,
                   ic=(x, y, z) -> Prim(u=(x, y, z), p=1.0, rho=1.0))
    num = Numerics(n_global=(12, 12, 12), art=ArtParams(enabled=false))
    solver, Q = setup(prob, num)

    @test Q isa ConservedState
    @test parent(Q) isa Array{Float64,4}
    @test size(Q) == (20, 20, 20, 5)
    Q[1, 1, 1, 1] = 2.0
    @test parent(Q)[1, 1, 1, 1] == 2.0
    @test parent(view(Q, :, :, :, 1)) === parent(Q)
    @test copy(Q) isa ConservedState
    @test zero(Q) isa ConservedState

    q_display = sprint(show, MIME("text/plain"), Q)
    pair_display = sprint(show, MIME("text/plain"), (solver, Q))
    @test q_display == "ConservedState{Float64}(20 × 20 × 20 × 5)"
    @test length(pair_display) < 200
    @test occursin("Solver(grid=12 × 12 × 12", pair_display)
    @test occursin(q_display, pair_display)
    @test run!(solver, Q; tfinal=1.0, nmax=0) === Q

    solver_display = sprint(show, MIME("text/plain"), solver)
    problem_display = sprint(show, MIME("text/plain"), prob)
    numerics_display = sprint(show, MIME("text/plain"), num)
    @test occursin("5 conserved variables", solver_display)
    @test occursin(prob.name, problem_display)
    @test !occursin("#", problem_display)       # do not dump the IC closure type
    @test occursin("artificial properties: disabled", numerics_display)
    @test all(length.((solver_display, problem_display, numerics_display)) .< 500)
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

@testset "validate_bc: NSCBC restrictions are setup errors" begin
    # Both restrictions used to be documented and unenforced. On an angular face
    # nothing failed: the wave analysis dropped the curvature terms carried by
    # grad_u[d,d] and the run completed with a wrong answer.
    two = IdealMixture([IdealSpecies{Float64}("a", 1.0, 1.4),
                        IdealSpecies{Float64}("b", 2.0, 1.6)])
    wall = (SlipWallBC(), SlipWallBC())
    cyl(θbc) = Solver(n_global=(12, 12, 12), L_domain=(1.0, 1.0, 1.0),
                      bcs=(wall, θbc, per3[3]), metric=CylindricalMetric(),
                      origin=(1.0, 0.0, 0.0), art=ArtParams(enabled=false))
    # r is a length under CylindricalMetric, θ is not.
    @test cyl(wall) isa Solver
    @test_throws ErrorException cyl((NSCBCOutflowBC(pinf=1.0), SlipWallBC()))
    @test_throws ErrorException cyl((SlipWallBC(),
                                     NSCBCInflowBC(u=(0.0, 0.1, 0.0), T_ion=1.0)))
    # Wrapping does not evade it: the `after` arm never passes through setup again.
    @test_throws ErrorException cyl((SwitchableBC(SlipWallBC(),
                                                 NSCBCOutflowBC(pinf=1.0)),
                                     SlipWallBC()))
    # Composition length, previously re-checked on every RHS call of the run.
    cart(Y) = Solver(n_global=(12, 12, 12), L_domain=(1.0, 1.0, 1.0),
                     bcs=((NSCBCInflowBC(u=(0.3, 0.0, 0.0), T_ion=1.0, Y=Y),
                           NSCBCOutflowBC(pinf=1.0)), per3[2], per3[3]),
                     eos=two, art=ArtParams(enabled=false))
    @test cart([0.4, 0.6]) isa Solver
    @test_throws ErrorException cart([1.0])
    @test_throws ErrorException cart([0.2, 0.3, 0.5])
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

@testset "compression-keyed beta sensors: gated_strain and dilatation" begin
    # The point of both non-default sensors is that bulk viscosity stops firing
    # on vortical structures and on expansions. Both halves are checked here,
    # along with the requirement that mu* — which keeps the strain sensor in
    # every case — is bit-identical across the three settings.
    tgv(sensor) = begin
        s = Solver(n_global=(32, 32, 32), L_domain=(2π, 2π, 2π), bcs=per3,
                   art=ArtParams(enabled=true, beta_sensor=sensor))
        Q = allocate_state(s)
        initialize!(s, Q, (x, y, z) -> Prim(rho=1.0, p=100.0,
            u=(sin(x)cos(y)cos(z), -cos(x)sin(y)cos(z), 0.0)))
        compute_rhs!(s, Q, zero(Q))
        s
    end
    ss, sg, sd = tgv(:strain), tgv(:gated_strain), tgv(:dilatation)
    # Taylor–Green is solenoidal, so neither compression-keyed sensor has
    # anything to fire on; the strain sensor sees the vortical structure and
    # fires. :dilatation collapses to round-off because its sensor is built
    # from ∇·u. :gated_strain keeps the firing sensor and leans on the switch
    # alone, which removes 99.4% of the total but not the maximum: the strain
    # sensor peaks at the cusps of |S|, and the switch degenerates at exactly
    # those points, where the vorticity vanishes along with |S| itself. The
    # test is therefore on the total, with the surviving peak recorded here
    # rather than asserted away.
    bsum(s) = sum(s.beta_art[gidx(s, i, j, k)] for i in 1:32, j in 1:32, k in 1:32)
    @test maximum(sd.beta_art) < 1e-12 * maximum(ss.beta_art)
    @test bsum(sg) < 1e-2 * bsum(ss)
    @test sd.mu_art == ss.mu_art
    @test sg.mu_art == ss.mu_art
    @test all(isfinite, sd.beta_art) && all(isfinite, sg.beta_art)

    # A velocity ramp: beta* must be exactly zero wherever the flow expands,
    # and must still switch on where it compresses.
    ramp(sensor) = begin
        s = Solver(n_global=(64, 12, 12), L_domain=(1.0, 0.2, 0.2), bcs=per3,
                   art=ArtParams(enabled=true, beta_sensor=sensor))
        Q = allocate_state(s)
        initialize!(s, Q, (x, y, z) -> Prim(rho=1.0, p=1.0,
                                            u=(0.5tanh((x - 0.5) / 0.02), 0.0, 0.0)))
        compute_rhs!(s, Q, zero(Q))
        s
    end
    div1(s, I) = s.grad_u[1, 1][I] + s.grad_u[2, 2][I] + s.grad_u[3, 3][I]
    rstrain = ramp(:strain)
    for sensor in (:gated_strain, :dilatation)
        s = ramp(sensor)
        idx = [gidx(s, i, 1, 1) for i in 1:64]
        expanding = [I for I in idx if div1(s, I) > 0]
        @test !isempty(expanding)
        @test all(I -> s.beta_art[I] == 0.0, expanding)
        @test maximum(s.beta_art) > 0.1 * maximum(rstrain.beta_art)
    end
    # gated_strain multiplies the strain sensor by a factor in [0, 1], so it can
    # only ever reduce beta*, never move it somewhere new.
    sg2 = ramp(:gated_strain)
    @test all(i -> sg2.beta_art[gidx(sg2, i, 1, 1)] <=
                   rstrain.beta_art[gidx(rstrain, i, 1, 1)] + 1e-300, 1:64)

    @test_throws ErrorException Solver(n_global=(16, 16, 16), L_domain=(1.0, 1.0, 1.0),
                                       bcs=per3, art=ArtParams(beta_sensor=:bogus))
end

@testset "compact_d8 ring detector: symbol, closures, and selectivity" begin
    # The whole point of the d8 detector is what it does BELOW the Nyquist,
    # so the operator is pinned against its analytic symbol rather than
    # against a convergence order. It is also the only symmetric banded
    # scheme in the package, hence the only exercise of BandPlan's filter-side
    # sign conventions (RHS added rather than subtracted, high-edge closure
    # rows mirrored rather than negated).
    N = 64
    art8 = ArtParams(detector=:d8)
    s = Solver(n_global=(N, 1, 1), L_domain=(1.0, 1.0, 1.0), art=art8, bcs=per3)
    f = CL.field(s.decomp); out = CL.field(s.decomp)
    pad = s.decomp.n_halo_d[1]
    A(k) = 1 + 2 * (14 / 29) * cos(k) + 2 * (3 / 58) * cos(2k)
    # RHS is 60/(240·29) times the undivided eighth difference (2 − 2cos k)⁴.
    symbol(k) = (60 / (240 * 29)) * (2 - 2cos(k))^4 / A(k)
    for kf in (0.25, 0.5, 0.75, 1.0)
        k = kf * π
        for i in 1:N
            f[i+pad, 1, 1] = cos(k * i)
        end
        CL.exchange_halos!(f, s.decomp)
        CL.apply_along!(out, s.ring_plans[1], f, s.decomp)
        i0 = N ÷ 2
        # rtol is 1e-9, not machine epsilon: the detector is a high-pass, so a
        # well-resolved wave is the difference of coefficients of order 1
        # producing an answer of order 1e-4, and three digits go to
        # cancellation. That is a property of the operator, not slack here.
        @test out[i0+pad, 1, 1] / cos(k * i0) ≈ symbol(k) rtol = 1e-9
    end
    # Normalization: the response at two points per wavelength is 16, which is
    # what undivided δ⁴ gives there. That is what lets the four artificial
    # constants transfer between detectors.
    @test symbol(π) ≈ 16.0 rtol = 1e-12
    # Selectivity against undivided δ⁴ on the same normalization: measured
    # 569× at eight points per wavelength and 26× at four.
    d4(k) = (2 - 2cos(k))^2
    @test symbol(0.25π) < d4(0.25π) / 500
    @test symbol(0.5π) < d4(0.5π) / 25

    # Constants must be annihilated exactly through the four closure rows, not
    # by cancellation: every row's weights sum to zero by construction.
    sw = Solver(n_global=(N, 1, 1), L_domain=(1.0, 1.0, 1.0), art=art8,
                bcs=((SlipWallBC(), SlipWallBC()), per3[2], per3[3]))
    fw = CL.field(sw.decomp); ow = CL.field(sw.decomp)
    fill!(fw, 1.0)
    CL.exchange_halos!(fw, sw.decomp)
    CL.apply_along!(ow, sw.ring_plans[1], fw, sw.decomp)
    @test maximum(abs, ow[(pad+1):(pad+N), 1, 1]) < 1e-14

    @test_throws ErrorException Solver(n_global=(16, 16, 16), L_domain=(1.0, 1.0, 1.0),
                                       bcs=per3, art=ArtParams(detector=:bogus))
end

@testset "detector = :d8 through compute_artificial!" begin
    # End to end on all three dimensions and both sensor weights, against the
    # δ⁴ default on identical data. A near-grid-scale ramp must still fire; a
    # resolved wave must fire far less than δ⁴ does, which is the reason the
    # detector exists.
    #
    # The selectivity assertion is on κ\*, not β\*, and that is the measurement
    # rather than a convenience. κ\*'s input is the internal energy, which is
    # smooth when the flow is; β\*'s is |S|, which has a cusp wherever the
    # strain passes through zero, and a cusp is a grid-scale feature no
    # detector can decline to see. On a resolved sine the two detectors
    # therefore differ by 2.6e6 on κ\* and by a factor of 1.8 on β\*. See
    # `reference/CALIBRATION.md`.
    sensors(detector, fn) = begin
        s = Solver(n_global=(64, 12, 12), L_domain=(1.0, 0.2, 0.2), bcs=per3,
                   art=ArtParams(enabled=true, detector=detector))
        Q = allocate_state(s)
        initialize!(s, Q, fn)
        compute_rhs!(s, Q, zero(Q))
        s
    end
    ramp = (x, y, z) -> Prim(rho=1.0, p=1.0, u=(0.5tanh((x - 0.5) / 0.02), 0.0, 0.0))
    wave = (x, y, z) -> Prim(rho=1.0, p=1.0 + 0.1sinpi(2x), u=(0.0, 0.0, 0.0))
    r4, r8 = sensors(:delta4, ramp), sensors(:d8, ramp)
    w4, w8 = sensors(:delta4, wave), sensors(:d8, wave)
    for s in (r8, w8)
        @test all(isfinite, s.beta_art) && all(isfinite, s.mu_art)
        @test all(isfinite, s.kappa_art)
        @test all(>=(0.0), s.beta_art)
    end
    # A ramp two to three cells wide is close enough to the grid scale that
    # the shared normalization holds; an order of magnitude either way would
    # mean the scaling is wrong. Measured 0.23.
    @test 0.1 < maximum(r8.beta_art) / maximum(r4.beta_art) < 10
    # A pressure wave resolved over 64 points is where they must not agree.
    # Measured 3.9e-7.
    @test maximum(w8.kappa_art) < 1e-4 * maximum(w4.kappa_art)
    @test maximum(w4.kappa_art) > 0
end

@testset "detector = :d8 through a coordinate-singularity fold" begin
    # BandPlan's fold assembly with a SYMMETRIC scheme, plus the :ring role in
    # fold_apply!. The C10 fold test covers the antisymmetric case only.
    s = Solver(n_global=(64, 1, 12), L_domain=(1.0, 1.0, 0.5),
               metric=CylindricalMetric(),
               bcs=((AxisBC(), SlipWallBC()), per3[2], per3[3]),
               art=ArtParams(enabled=true, detector=:d8))
    f = CL.field(s.decomp); out = CL.field(s.decomp)
    # An even function of r: d8 across the axis fold must stay smooth and
    # small, and in particular must not blow up in the first cells.
    fillf!(s, f, (r, θ, z) -> exp(-4r^2))
    CL.exchange_halos!(f, s.decomp)
    CL.ring_along!(out, f, s, 1, 1)
    @test all(isfinite, out)
    @test maximum(abs, out[gidx(s, i, 1, k)] for i in 1:64, k in 1:12) < 1e-3
    # And the full sensor path runs through the fold.
    Q = allocate_state(s)
    initialize!(s, Q, (r, θ, z) -> Prim(rho=1.0, p=1.0, u=(-1.0, 0.0, 0.0)))
    compute_rhs!(s, Q, zero(Q))
    @test all(isfinite, s.beta_art) && maximum(s.beta_art) > 0
end

@testset "sensor fields: mu* from the velocity, beta* from the dilatation" begin
    # Cook builds both sensors from |S|; the reference implementation builds
    # them from the velocity components and from ∇·u. The difference between
    # the two is the absolute value, and these tests pin its two consequences.
    oned(art, fn; N=64) = begin
        s = Solver(n_global=(N, 1, 1), L_domain=(1.0, 1.0, 1.0), bcs=per3, art=art)
        Q = allocate_state(s)
        initialize!(s, Q, fn)
        compute_rhs!(s, Q, zero(Q))
        s
    end
    # 1. The two-point wave. A centered derivative annihilates it exactly, so
    # |S| and ∇·u are identically zero on the shortest wave the grid carries
    # and every sensor built from them returns zero there. The velocity sensor
    # returns the full undivided response, C_mu·ρ·16h.
    nyq = (x, y, z) -> Prim(rho=1.0, p=1.0, u=(cospi(64x), 0.0, 0.0))
    @test maximum(oned(ArtParams(mu_sensor=:strain), nyq).mu_art) == 0.0
    @test maximum(oned(ArtParams(mu_sensor=:velocity), nyq).mu_art) ≈
          0.002 * 16 / 64 rtol = 1e-12
    @test maximum(oned(ArtParams(beta_sensor=:ungated_dilatation), nyq).beta_art) == 0.0

    # 2. The detector's selectivity survives the field change, which is the
    # reason for making it. On a wave resolved over eight points the two
    # detectors are built to differ by 569×, and they do so when applied to a
    # smooth field. Applied to |S| the same pair differs by 1.28×, the cusps
    # where the strain passes through zero being grid-scale structure at every
    # wavelength. Measured 569 and 1.28.
    wave = (x, y, z) -> Prim(rho=1.0, p=1.0, u=(cospi(16x), 0.0, 0.0))
    peak(ms, det) = maximum(oned(ArtParams(mu_sensor=ms, detector=det), wave).mu_art)
    @test peak(:velocity, :d8) < peak(:velocity, :delta4) / 100
    @test peak(:strain, :d8) > peak(:strain, :delta4) / 10

    # 3. Σ_d against MAX. They are the same operation in one dimension, and the
    # reduction is a per-direction one, so MAX can never exceed Σ_d anywhere.
    ramp1 = (x, y, z) -> Prim(rho=1.0, p=1.0, u=(0.5tanh((x - 0.5) / 0.02), 0.0, 0.0))
    @test oned(ArtParams(reduction=:sum), ramp1).mu_art ==
          oned(ArtParams(reduction=:max), ramp1).mu_art
    tgv3(red) = begin
        s = Solver(n_global=(32, 32, 32), L_domain=(2π, 2π, 2π), bcs=per3,
                   art=ArtParams(reduction=red))
        Q = allocate_state(s)
        initialize!(s, Q, (x, y, z) -> Prim(rho=1.0, p=100.0,
            u=(sin(x)cos(y)cos(z), -cos(x)sin(y)cos(z), 0.0)))
        compute_rhs!(s, Q, zero(Q))
        s
    end
    tsum, tmax = tgv3(:sum), tgv3(:max)
    @test all(tmax.mu_art .<= tsum.mu_art)
    @test maximum(tmax.mu_art) < maximum(tsum.mu_art)

    # 4. Parity across a fold. u_r is odd there, so the detector requires the
    # sign. With it, u_r = r produces no sensor at all, that being the regular
    # behaviour of a radial velocity at the axis, while a uniform u_r produces
    # one on the first cell, the folded field having a genuine kink there.
    axial(art, fn) = begin
        s = Solver(n_global=(32, 1, 1), L_domain=(1.0, 1.0, 1.0),
                   metric=CylindricalMetric(),
                   bcs=((AxisBC(), SlipWallBC()), per3[2], per3[3]), art=art)
        Q = allocate_state(s)
        initialize!(s, Q, fn)
        compute_rhs!(s, Q, zero(Q))
        s
    end
    lin = (r, θ, z) -> Prim(rho=1.0, p=1.0, u=(r, 0.0, 0.0))
    uni = (r, θ, z) -> Prim(rho=1.0, p=1.0, u=(-1.0, 0.0, 0.0))
    # δ⁴ across the fold reads the half-offset mirror for an odd field, so this
    # is exact; the d8 closure reaches it through a line solve, hence round-off.
    sl = axial(ArtParams(mu_sensor=:velocity), lin)
    @test sl.mu_art[gidx(sl, 1, 1, 1)] == 0.0
    s8 = axial(ArtParams(mu_sensor=:velocity, detector=:d8), lin)
    @test s8.mu_art[gidx(s8, 1, 1, 1)] < 1e-14
    su = axial(ArtParams(mu_sensor=:velocity), uni)
    @test su.mu_art[gidx(su, 1, 1, 1)] > 0

    # 5. The ungated dilatation sensor is the form the reference ships. The
    # gated one is that sensor multiplied by the Ducros switch, so it is never
    # larger, and is exactly zero wherever the flow expands.
    ug = oned(ArtParams(beta_sensor=:ungated_dilatation), ramp1)
    g = oned(ArtParams(beta_sensor=:dilatation), ramp1)
    @test all(g.beta_art .<= ug.beta_art .+ 1e-300)
    div1(s, I) = s.grad_u[1, 1][I] + s.grad_u[2, 2][I] + s.grad_u[3, 3][I]
    expanding = [gidx(ug, i, 1, 1) for i in 1:64 if div1(ug, gidx(ug, i, 1, 1)) > 0]
    @test !isempty(expanding)
    @test all(I -> g.beta_art[I] == 0.0, expanding)
    @test any(I -> ug.beta_art[I] > 0.0, expanding)

    # 6. The channels are independent: rebuilding one leaves the other's
    # numbers bit-identical to the shipped configuration's.
    base = oned(ArtParams(), ramp1)
    @test oned(ArtParams(mu_sensor=:velocity), ramp1).beta_art == base.beta_art
    @test oned(ArtParams(beta_sensor=:ungated_dilatation), ramp1).mu_art == base.mu_art
    @test oned(ArtParams(mu_sensor=:velocity), ramp1).mu_art != base.mu_art

    @test_throws ErrorException Solver(n_global=(16, 16, 16), L_domain=(1.0, 1.0, 1.0),
                                       bcs=per3, art=ArtParams(mu_sensor=:bogus))
    @test_throws ErrorException Solver(n_global=(16, 16, 16), L_domain=(1.0, 1.0, 1.0),
                                       bcs=per3, art=ArtParams(reduction=:bogus))
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

# Walk the appended-data blocks of a .vtr/.vts piece and return the Float32
# point-data arrays by name. Each block is a UInt64 byte count followed by its
# payload, so this also verifies that the offsets and lengths are consistent. A
# writer that miscounted desynchronizes here rather than producing a plausible
# file that only ParaView rejects.
function read_vtk_pointdata(path)
    raw = read(path)
    marker = Vector{UInt8}("<AppendedData encoding=\"raw\">\n_")
    i0 = findfirst(marker, raw)[end]
    header = String(copy(raw[1:i0]))
    decls = [(m.captures[1], parse(Int, m.captures[2])) for m in eachmatch(
        r"<DataArray type=\"Float32\" Name=\"([^\"]+)\" NumberOfComponents=\"(\d+)\"",
        header)]
    blocks = Vector{Vector{UInt8}}()
    pos = i0 + 1
    stop = length(raw) - length("\n</AppendedData>\n</VTKFile>\n")
    while pos <= stop
        n = only(reinterpret(UInt64, raw[pos:pos+7]))
        push!(blocks, raw[pos+8:pos+7+n])
        pos += 8 + n
    end
    @assert length(blocks) >= length(decls)
    geom = blocks[1:end-length(decls)]           # coordinates or points
    out = Dict{String,Vector{Float32}}()
    for (m, (name, nc)) in enumerate(decls)
        arr = collect(reinterpret(Float32, blocks[end-length(decls)+m]))
        out[name] = arr
    end
    return out, geom, decls
end

@testset "save_vtk: field selection and derived fields" begin
    dir = mktempdir()
    solver = Solver(bcs=per3, n_global=(32, 32, 12), L_domain=(1.0, 1.0, 1.0),
                    art=ArtParams(enabled=true))
    Q = allocate_state(solver)

    # (a) A uniform stream, for which every derived field is zero by
    # definition. This detects a sign error or a transposed index in the curl.
    initialize!(solver, Q, (x, y, z) -> Prim(u=(0.3, 0, 0), p=1.0, rho=1.0))
    derived = (:vorticity, :vorticity_magnitude, :qcriterion, :divergence,
               :schlieren, :mach)
    save_vtk(solver, Q, joinpath(dir, "uniform"); fields=derived)
    got, _, decls = read_vtk_pointdata(joinpath(dir, "uniform.r0000.vtr"))
    @test [n for (n, _) in decls] == ["vorticity", "vorticity_magnitude",
                                      "qcriterion", "divergence", "schlieren", "mach"]
    for name in ("vorticity", "vorticity_magnitude", "qcriterion", "divergence",
                 "schlieren")
        @test maximum(abs, got[name]) < 1e-5
    end
    @test all(≈(0.3 / sqrt(1.4)), got["mach"])        # |u|/c, γRT = 1.4 here

    # (b) A shear layer u₁ = sin(2πy), for which ω₃ = −∂u₁/∂x₂ = −2π cos(2πy)
    # and the divergence is zero. The interior scheme is C6, so the tolerance
    # is tight.
    initialize!(solver, Q, (x, y, z) -> Prim(u=(sin(2π * y), 0, 0), p=1.0, rho=1.0))
    save_vtk(solver, Q, joinpath(dir, "shear"); fields=(:vorticity, :divergence))
    got, _, _ = read_vtk_pointdata(joinpath(dir, "shear.r0000.vtr"))
    ω = got["vorticity"]
    nx, ny = 32, 32
    werr = 0.0
    for j in 1:ny, i in 1:nx
        m = (j - 1) * nx + i                          # k = 1 plane, i fastest
        want = -2π * cos(2π * xcoord(solver, 2, j))
        werr = max(werr, abs(ω[3m] - want))           # third component
    end
    @test werr < 1e-4
    @test maximum(abs, got["divergence"]) < 1e-4

    # (c) The default set is unchanged from before field selection was added.
    save_vtk(solver, Q, joinpath(dir, "default"))
    _, _, decls = read_vtk_pointdata(joinpath(dir, "default.r0000.vtr"))
    @test [n for (n, _) in decls] == ["rho", "velocity", "p", "T_ion", "Y1"]
    @test [nc for (_, nc) in decls] == [1, 3, 1, 1, 1]

    # (d) The artificial coefficients are readable, which is the reason for
    # exposing them: they report the local action of the regularization.
    save_vtk(solver, Q, joinpath(dir, "art"); fields=(:sensor, :mu_art, :D_art))
    _, _, decls = read_vtk_pointdata(joinpath(dir, "art.r0000.vtr"))
    @test [n for (n, _) in decls] == ["sensor", "mu_art", "D_art1"]

    @test_throws ArgumentError save_vtk(solver, Q, joinpath(dir, "bad");
                                        fields=(:not_a_field,))
    rm(dir; recursive=true)
end

@testset "save_vtk: strided subsampling" begin
    dir = mktempdir()
    N = (32, 16, 16)
    solver = Solver(bcs=per3, n_global=N, L_domain=(1.0, 1.0, 1.0),
                    art=ArtParams(enabled=false))
    Q = allocate_state(solver)
    initialize!(solver, Q, (x, y, z) -> Prim(u=(0.1, 0, 0), p=1.0,
                                             rho=1 + x + 2y + 4z))

    # Coordinates and payload for a given stride, read back from the piece.
    strided(stem, stride) = begin
        save_vtk(solver, Q, stem; fields=(:rho,), stride=stride)
        got, geom, _ = read_vtk_pointdata(stem * ".r0000.vtr")
        coords = [collect(reinterpret(Float64, g)) for g in geom]
        header = String(copy(read(stem * ".pvtr")))
        whole = match(r"WholeExtent=\"([^\"]+)\"", header).captures[1]
        (coords, got["rho"], whole)
    end

    # Stride 1 reproduces the unstrided grid, confirming that the general path
    # introduces no index shift.
    coords, rho1, whole = strided(joinpath(dir, "s1"), 1)
    @test length.(coords) == [32, 16, 16]
    @test whole == "0 31 0 15 0 15"
    for d in 1:3
        @test coords[d] ≈ [xcoord(solver, d, i) for i in 1:N[d]] atol = 1e-15
    end

    # Stride 2 gives half the points per dimension and extents in the coarse
    # index space. The coordinates are the ODD global stations 1, 3, 5, ...,
    # not a uniformly re-spaced grid.
    coords, rho2, whole = strided(joinpath(dir, "s2"), 2)
    @test length.(coords) == [16, 8, 8]
    @test whole == "0 15 0 7 0 7"
    for d in 1:3
        @test coords[d] ≈ [xcoord(solver, d, i) for i in 1:2:N[d]] atol = 1e-15
    end
    @test length(rho2) == 16 * 8 * 8

    # The payload holds the same values, sampled rather than averaged or
    # shifted.
    want = Float32[]
    for k in 1:2:16, j in 1:2:16, i in 1:2:32
        push!(want, Float32(solver.rho[gidx(solver, i, j, k)]))
    end
    @test rho2 == want

    # Per-dimension strides are independent.
    coords, _, whole = strided(joinpath(dir, "s421"), (4, 2, 1))
    @test length.(coords) == [8, 8, 16]
    @test whole == "0 7 0 7 0 15"

    # An odd extent keeps the ceiling: 1, 3, ..., 15 out of 15 is 8 stations.
    odd = Solver(bcs=per3, n_global=(15, 16, 16), L_domain=(1.0, 1.0, 1.0),
                 art=ArtParams(enabled=false))
    Qo = allocate_state(odd)
    initialize!(odd, Qo, (x, y, z) -> Prim(u=(0, 0, 0), p=1.0, rho=1.0))
    save_vtk(odd, Qo, joinpath(dir, "odd"); fields=(:rho,), stride=2)
    @test occursin("WholeExtent=\"0 7 0 7 0 7\"",
                   String(copy(read(joinpath(dir, "odd.pvtr")))))

    @test_throws ArgumentError save_vtk(solver, Q, joinpath(dir, "bad"); stride=0)
    @test_throws ArgumentError save_vtk(solver, Q, joinpath(dir, "bad");
                                        stride=(2, 0, 1))
    rm(dir; recursive=true)
end

@testset "save_vtk: slicing" begin
    dir = mktempdir()
    N = (24, 16, 12)
    solver = Solver(bcs=per3, n_global=N, L_domain=(1.0, 1.0, 1.0),
                    art=ArtParams(enabled=false))
    Q = allocate_state(solver)
    initialize!(solver, Q, (x, y, z) -> Prim(u=(0.1, 0, 0), p=1.0,
                                             rho=1 + x + 100y + 10000z))
    refresh_primitives!(solver, Q)

    # A slice is one point thick in the sliced dimension and full in the others.
    for (d, g) in ((1, 7), (2, 5), (3, 9))
        stem = joinpath(dir, "s$(d)_$(g)")
        save_vtk(solver, Q, stem; fields=(:rho,), slice=(d, g))
        want = ntuple(t -> t == d ? 1 : N[t], 3)
        header = String(copy(read(stem * ".pvtr")))
        @test occursin("WholeExtent=\"0 $(want[1]-1) 0 $(want[2]-1) " *
                       "0 $(want[3]-1)\"", header)
        got, geom, _ = read_vtk_pointdata(stem * ".r0000.vtr")
        coords = [collect(reinterpret(Float64, b)) for b in geom]
        @test length.(coords) == [want...]
        # The retained station is the requested one, not the first.
        @test coords[d] ≈ [xcoord(solver, d, g)] atol = 1e-15
        @test length(got["rho"]) == prod(want)
        # And the values are the plane at g, not some other plane.
        idx = ntuple(t -> t == d ? (g:g) : (1:N[t]), 3)
        expect = Float32[solver.rho[gidx(solver, i, j, k)]
                         for i in idx[1], j in idx[2], k in idx[3]][:]
        @test got["rho"] == expect
    end

    # A slice composes with a stride on the two dimensions still resolved.
    stem = joinpath(dir, "both")
    save_vtk(solver, Q, stem; fields=(:rho,), stride=2, slice=(3, 5))
    header = String(copy(read(stem * ".pvtr")))
    @test occursin("WholeExtent=\"0 11 0 7 0 0\"", header)
    _, geom, _ = read_vtk_pointdata(stem * ".r0000.vtr")
    @test length.([collect(reinterpret(Float64, b)) for b in geom]) == [12, 8, 1]

    @test_throws ArgumentError save_vtk(solver, Q, joinpath(dir, "bad");
                                        slice=(4, 1))
    @test_throws ArgumentError save_vtk(solver, Q, joinpath(dir, "bad");
                                        slice=(2, 0))
    @test_throws ArgumentError save_vtk(solver, Q, joinpath(dir, "bad");
                                        slice=(2, 17))
    rm(dir; recursive=true)
end

@testset "save_vtk: a resolved angle writes a curvilinear grid" begin
    dir = mktempdir()
    # A cylindrical annulus with θ resolved. Written as a rectilinear grid it
    # would render unwrapped, as a box in (r, θ) rather than as an annulus.
    bcs = ((SlipWallBC(), SlipWallBC()), (PeriodicBC(), PeriodicBC()),
           (PeriodicBC(), PeriodicBC()))
    nr, nθ, nz = 12, 24, 10
    solver = Solver(bcs=bcs, n_global=(nr, nθ, nz), L_domain=(1.0, 2π, 1.0),
                    origin=(0.5, 0.0, 0.0), metric=CylindricalMetric(),
                    art=ArtParams(enabled=false))
    Q = allocate_state(solver)
    # Solid-body swirl: purely azimuthal, constant magnitude.
    initialize!(solver, Q, (r, θ, z) -> Prim(u=(0.0, 0.5, 0.0), p=1.0, rho=1.0))

    @test container_extension(solver) == ".pvts"
    save_vtk(solver, Q, joinpath(dir, "annulus"))
    @test isfile(joinpath(dir, "annulus.pvts"))
    @test isfile(joinpath(dir, "annulus.r0000.vts"))
    @test occursin("PStructuredGrid", read(joinpath(dir, "annulus.pvts"), String))

    got, geom, _ = read_vtk_pointdata(joinpath(dir, "annulus.r0000.vts"))
    pts = collect(reinterpret(Float64, only(geom)))
    @test length(pts) == 3 * nr * nθ * nz
    radii = [hypot(pts[3m-2], pts[3m-1]) for m in 1:(nr*nθ*nz)]
    @test minimum(radii) ≈ 0.5 rtol = 1e-12
    @test maximum(radii) ≈ 1.5 rtol = 1e-12

    # The velocity must be rotated into the frame in which the points are
    # written. Unrotated, (u_r, u_θ, u_z) = (0, 0.5, 0) would appear as a
    # uniform y-directed stream; rotated, it is tangent to every circle.
    vel = got["velocity"]
    radial = 0.0; magnitude = 0.0
    for m in 1:(nr*nθ*nz)
        x, y = pts[3m-2], pts[3m-1]
        ux, uy = Float64(vel[3m-2]), Float64(vel[3m-1])
        radial = max(radial, abs(ux * x + uy * y) / hypot(x, y))
        magnitude = max(magnitude, abs(hypot(ux, uy) - 0.5))
    end
    @test radial < 1e-6                    # no radial component: purely tangential
    @test magnitude < 1e-6                 # and the swirl speed is preserved

    # An axisymmetric (θ-collapsed) polar run remains rectilinear, since the
    # meridional half-plane is the computed domain and a rectangle represents
    # it correctly.
    axi = Solver(bcs=(bcs[1], (PeriodicBC(), PeriodicBC()), bcs[3]),
                 n_global=(nr, 1, nz), L_domain=(1.0, 2π, 1.0),
                 origin=(0.5, 0.0, 0.0), metric=CylindricalMetric(),
                 art=ArtParams(enabled=false))
    @test container_extension(axi) == ".pvtr"
    rm(dir; recursive=true)
end

@testset "FieldWriter: numbered frames and a .pvd carrying physical time" begin
    dir = mktempdir()
    solver = Solver(bcs=((SlipWallBC(), SlipWallBC()), (PeriodicBC(), PeriodicBC()),
                         (PeriodicBC(), PeriodicBC())),
                    n_global=(16, 12, 12), L_domain=(1.0, 1.0, 1.0),
                    art=ArtParams(enabled=false), cfl=0.4)
    Q = allocate_state(solver)
    initialize!(solver, Q, (x, y, z) -> Prim(u=(0.2, 0, 0),
                                             p=1 + 0.1exp(-40(x - 0.5)^2), rho=1.0))
    writer = FieldWriter(joinpath(dir, "frames", "field"))
    run!(solver, Q; tfinal=0.03,
         callback=Callback(EveryTime(0.01), writer))

    # One frame per instant, numbered from zero, with the directory created.
    @test writer.index == 3
    @test writer.times ≈ [0.01, 0.02, 0.03] rtol = 1e-14
    for m in 0:2
        @test isfile(CL.frame_prefix(writer, m) * ".pvtr")
        @test filesize(CL.frame_prefix(writer, m) * ".pvtr") > 0
    end

    # The collection is what allows the sequence to animate against physical
    # time rather than frame index, so its timesteps are the assertion.
    pvd = read(joinpath(dir, "frames", "field.pvd"), String)
    @test occursin("type=\"Collection\"", pvd)
    @test count("<DataSet", pvd) == 3
    stamps = [parse(Float64, m.captures[1])
              for m in eachmatch(r"timestep=\"([^\"]+)\"", pvd)]
    @test stamps ≈ [0.01, 0.02, 0.03] rtol = 1e-14
    # Pieces are named relative to the .pvd's own directory rather than by
    # absolute path, so the collection remains readable on another machine.
    @test occursin("file=\"field_0000.pvtr\"", pvd)
    @test !occursin(dir, pvd)

    @test writer.wall_io > 0            # I/O cost is not in solver.wall_step
    rm(dir; recursive=true)
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

    # Any two of (p, rho, T_ion) determine the state; all three spellings of the
    # same point must give the same conserved vector.
    ρ0 = 0.8 / (Rm * 1.7)
    from_rho_T = conserved_from_prim(eos, Prim(u=(0.3, -0.1, 0.2), rho=ρ0,
                                              T_ion=1.7, Y=(0.35, 0.65)))
    from_p_rho = conserved_from_prim(eos, Prim(u=(0.3, -0.1, 0.2), p=0.8,
                                              rho=ρ0, Y=(0.35, 0.65)))
    @test all(isapprox.(from_rho_T, q; rtol=1e-14))
    @test all(isapprox.(from_p_rho, q; rtol=1e-14))
    # The other two EOS models take the same three spellings. Each one's thermal
    # relation supplies the pressure that (rho, T_ion) implies.
    liquid = StiffenedGas(gamma=4.4, p_inf=6.0e8, cv=1816.0)
    p_liquid = 1000.0 * CL.gas_constant(liquid) * 300.0 - liquid.p_inf
    ideal = single_species(gamma=1.4, R=287.0)
    for (E, p_of) in ((liquid, p_liquid), (ideal, 1000.0 * 287.0 * 300.0))
        @test all(isapprox.(conserved_from_prim(E, Prim(rho=1000.0, T_ion=300.0)),
                            conserved_from_prim(E, Prim(p=p_of, T_ion=300.0));
                            rtol=1e-12))
    end
    # Exactly two, no more and no fewer.
    @test_throws ErrorException Prim(p=1.0)
    @test_throws ErrorException Prim(rho=1.0)
    @test_throws ErrorException Prim(p=1.0, rho=1.0, T_ion=1.0)
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

@testset "EveryTime: evenly spaced instants, restart anchoring, soft landing" begin
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

    # Every instant is landed on exactly, and none is skipped.
    solver, Q = mkrun()
    hits = Float64[]
    run!(solver, Q; tfinal=0.05,
         callback=Callback(EveryTime(0.01), (s, _) -> (push!(hits, s.t); nothing)))
    @test hits ≈ collect(0.01:0.01:0.05) rtol = 1e-14

    # Anchored to solver.t on first use, not to zero: a restarted run picks up at
    # the next instant rather than replaying the schedule.
    solver, Q = mkrun()
    solver.t = 0.055
    hits = Float64[]
    run!(solver, Q; tfinal=0.08,
         callback=Callback(EveryTime(0.01), (s, _) -> (push!(hits, s.t); nothing)))
    @test hits ≈ [0.06, 0.07, 0.08] rtol = 1e-14

    # `start` offsets the schedule, and instants before it are not visited.
    solver, Q = mkrun()
    hits = Float64[]
    run!(solver, Q; tfinal=0.05,
         callback=Callback(EveryTime(0.02; start=0.015),
                           (s, _) -> (push!(hits, s.t); nothing)))
    @test hits ≈ [0.015, 0.035] rtol = 1e-14

    @test_throws ArgumentError EveryTime(0.0)
    @test_throws ArgumentError StepControl(landing_steps=0)

    # The soft landing is the purpose of `landing_steps`. Under a hard clip the
    # same schedule leaves one very small step before every instant, so compare
    # the spread of accepted steps.
    spread(control) = begin
        s, q = mkrun()
        dts = Float64[]
        run!(s, q; tfinal=0.05, control=control,
             callback=(Callback(EveryTime(0.01), (_, _) -> nothing),
                       (x, _) -> push!(dts, x.dt_prev)))
        minimum(dts) / maximum(dts)
    end
    @test spread(StepControl(landing_steps=1)) < 0.35     # sliver: 2.2e-3 vs 7.8e-3
    @test spread(StepControl()) > 0.99                    # split evenly instead

    # An instant BEYOND tfinal must not be landed on. Instants are computed as
    # `start + n*interval`, so the third one here is `3 * 0.05`, which is the next
    # float above the 0.15 literal — the schedule ends one ULP past the endpoint.
    # The soft landing used to aim at it anyway, and since `dt` was already
    # clipped to `tfinal - t` the gap stayed a shade above `dt` and `ceil(gap/dt)`
    # stayed at 2, halving the step against a target the run never reaches. The
    # run still finished and every instant still fired, so the only symptom was
    # step count: 48 steps against 9 in the case this was found in.
    #
    # Measured against the same run with no trigger rather than a fixed number,
    # since the step count is a property of the CFL here and not of the landing.
    @test 3 * 0.05 > 0.15                          # the ULP the case rests on
    nsteps(cb) = begin
        s, q = mkrun()
        run!(s, q; tfinal=0.15, callback=cb)
        s.step
    end
    bare = nsteps(nothing)
    @test nsteps(Callback(EveryTime(0.05), (_, _) -> nothing)) <= bare + 4
end

@testset "rewind!: a rollback re-arms the instants it abandoned" begin
    # Unit behaviour first. AtTime rewinds to the first instant ahead of the
    # restored time, EveryTime re-anchors, and a step-counting trigger has no
    # schedule to move and takes the no-op default.
    trigger = AtTime([0.1, 0.2, 0.3])
    trigger.next = 3
    CL.rewind!(trigger, 0.15, 7)
    @test trigger.next == 2
    CL.rewind!(trigger, 0.35, 9)
    @test trigger.next == 4                              # all behind: none left
    CL.rewind!(trigger, 0.0, 0)
    @test trigger.next == 1

    et = EveryTime(0.01)
    solver = Solver(bcs=per3, n_global=(16, 12, 12), L_domain=(1.0, 1.0, 1.0),
                    art=ArtParams(enabled=false))
    solver.t = 0.045
    @test CL.next_time(et, solver) ≈ 0.05
    CL.rewind!(et, 0.021, 3)
    solver.t = 0.021
    @test CL.next_time(et, solver) ≈ 0.03
    @test CL.rewind!(EveryStep(4), 0.1, 3) === nothing

    # Then the wiring, against a rollback placed where it can be reasoned about.
    # Physical failures occur in the startup transient, before a savepoint has
    # advanced past step 0, which demonstrates nothing here. Corrupting the
    # state once at a chosen time and letting the positivity check find it
    # places the rollback across instants that have already fired.
    walls = ((SlipWallBC(), SlipWallBC()), (PeriodicBC(), PeriodicBC()),
             (PeriodicBC(), PeriodicBC()))
    s2 = Solver(bcs=walls, n_global=(16, 12, 12), L_domain=(1.0, 1.0, 1.0),
                art=ArtParams(enabled=false), cfl=0.4)
    Q2 = allocate_state(s2)
    initialize!(s2, Q2, (x, y, z) -> Prim(u=(0.2, 0, 0),
                                          p=1 + 0.1exp(-40(x - 0.5)^2), rho=1.0))
    instants = collect(0.005:0.005:0.06)
    seen = Float64[]
    spoil = Callback(WhenState((x, _) -> x.t >= 0.037),
                     (x, q) -> (q[gidx(x, 3, 3, 3), 1] = -1.0; nothing))
    run!(s2, Q2; tfinal=0.06, nmax=5000,
         control=StepControl(retries=2, savepoint_interval=2),
         callback=(Callback(AtTime(instants), (x, _) -> (push!(seen, x.t); nothing)),
                   spoil))
    @test s2.cfl < 0.4                                    # it did roll back
    @test s2.t ≈ 0.06 rtol = 1e-12                        # and then finished
    # Re-arming has a positive signature: an instant crossed on the abandoned
    # trajectory is visited a second time on the replacement trajectory. Without
    # the rewind it is visited once and never revisited, and the `unique` check
    # below would fall short.
    @test length(seen) > length(instants)
    @test sort(unique(round.(seen, digits=10))) ≈ instants rtol = 1e-9
end

@testset "state queries and refresh_primitives! in a callback" begin
    eos = IdealMixture([IdealSpecies{Float64}("a", 1.0, 1.4),
                        IdealSpecies{Float64}("b", 2.0, 1.6)])
    solver = Solver(bcs=per3, n_global=(16, 12, 12), L_domain=(1.0, 1.0, 1.0),
                    eos=eos, art=ArtParams(enabled=false), cfl=0.4)
    Q = allocate_state(solver)
    initialize!(solver, Q, (x, y, z) -> Prim(Y=(0.3, 0.7), u=(0.2, -0.1, 0.05),
                                             p=1 + 0.1exp(-40(x - 0.5)^2), rho=1.0))
    refresh_primitives!(solver, Q)

    # The layout-free spellings agree with the primitives they stand in for, and
    # do not assume a species count.
    I = gidx(solver, 5, 4, 3)
    @test mixture_density(solver, Q, I) ≈ solver.rho[I] rtol = 1e-14
    @test all(velocity(solver, Q, I) .≈ (solver.u[I], solver.v[I], solver.w[I]))
    @test mass_fraction(solver, Q, I, 2) ≈ solver.Y[2][I] rtol = 1e-14
    @test total_energy(solver, Q, I) == Q[I, solver.equations.i_energy]
    @test mixture_density(solver, Q, I) ≈ Q[I, 1] + Q[I, 2] rtol = 1e-14

    # In serial every rank holds every edge, so the `nothing` branch is
    # exercised only under MPI.
    @test boundary_plane(solver, 2, 1) == CL.wallplane(solver.decomp, 2, 1)

    # interior_index is the documented conversion between the two index bases:
    # boundary_plane yields padded indices, xcoord takes interior ones.
    @test interior_index(solver, I) == (5, 4, 3)
    @test gidx(solver, interior_index(solver, I)...) == I
    for J in boundary_plane(solver, 1, 2)
        i, j, k = interior_index(solver, J)
        @test gidx(solver, i, j, k) == J
        @test i == solver.decomp.n_local[1]              # the high-x face
        @test xcoord(solver, 1, i) ==
              global_xcoord(solver, 1, solver.decomp.offset[1] + i)
    end

    # The contract stated by refresh_primitives!: inside a callback the
    # primitives belong to the fifth RK stage's INPUT state, so they predate the
    # final stage update and the filter pass. The assertion is that refreshing
    # changes the answer, and that the refreshed value is the one Q holds.
    stale = Ref(0.0); fresh = Ref(0.0); fromQ = Ref(0.0)
    run!(solver, Q; tfinal=1e9, nmax=6, callback=(s, q) -> begin
        s.step == 6 || return
        stale[] = s.rho[I]
        fromQ[] = mixture_density(s, q, I)
        refresh_primitives!(s, q)
        fresh[] = s.rho[I]
    end)
    @test isapprox(fresh[], fromQ[]; rtol=1e-14)         # refreshed agrees with Q
    @test !isapprox(stale[], fromQ[]; rtol=1e-12)        # unrefreshed does not
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

    # Switched by a callback, it must actually take the other branch. That the
    # branch is *physically* the right one is the next testset's job; this one
    # only pins the wrapper's mechanics.
    sw2 = (SwitchableBC(SlipWallBC(), ExtrapolationBC()),
           SwitchableBC(SlipWallBC(), ExtrapolationBC()))
    s_sw, Q_sw = mkrun(sw2)
    run!(s_sw, Q_sw; tfinal=1e9, nmax=12,
         callback=Callback(WhenState((s, _) -> s.step >= 4),
                           (_, _) -> (switch!.(sw2); nothing)))
    @test all(switched, sw2)
    @test Q_sw != Q_ref
    @test switch!(sw2[1]) === sw2[1]      # idempotent
end

@testset "SwitchableBC: switching upstream lets the reflected wave leave" begin
    # A shock/interface interaction in a translating frame, with the upstream
    # boundary switching from inflow to outflow when the interface-reflected
    # wave reaches it. This is the case SwitchableBC was built for, and the one
    # a finiteness check says nothing about: whether the switched boundary
    # actually lets the wave out.
    #
    #        shocked driven | driven |    heavy
    #      ---------------->|        |
    #        (region 2)     ^shock   ^interface
    #      0               0.20     0.40              1
    #
    # The state initialized upstream of the shock is the SHOCKED driven gas —
    # region 2 of the normal-shock relations, not a driver — so the incident
    # shock is already formed and the run starts at the physics of interest.
    # Sustaining that state is what makes the upstream boundary an INFLOW
    # (u2 + U > 0 into the domain) for as long as the incident shock is being
    # driven. NSCBCInflowBC is the condition that holds it: it replaces every
    # incoming characteristic with a relaxation toward (u, T_ion, Y), which is
    # exactly right while region 2 is uniform there.
    #
    # It becomes exactly wrong the moment the interface-reflected shock arrives.
    # Behind that shock the state is no longer region 2, so an inflow condition
    # still relaxing toward region-2 targets is over-constraining the boundary,
    # and the mismatch radiates back inward onto the interface. The fix is to
    # stop imposing and start absorbing — switch to NSCBCOutflowBC — which is
    # the whole point of the wrapper.
    #
    # The frame translation U is chosen so the SHOCKED interface is nearly at
    # rest (it drifts ~0.04 once the shock has passed it), the standard frame for a
    # shock/interface calculation: it keeps the interface in the box, and it
    # gives both faces a mean flow so nothing is being tested at the degenerate
    # u = 0 point where NSCBC's Mach-dependent terms drop out.
    γ, RL, RH, MS, U = 1.4, 1.0, 0.25, 1.5, -0.5
    N, XS, XI, TEND = 160, 0.20, 0.40, 1.0
    Hx = 1 / (N - 1)

    # Region 1 (unshocked driven gas) and the normal-shock relations for the
    # region 2 that sits behind a Mach MS shock running into it.
    p1 = ρ1 = T1 = 1.0
    c1 = sqrt(γ)
    p2 = p1 * (2γ * MS^2 - (γ - 1)) / (γ + 1)
    ρ2 = ρ1 * ((γ + 1) * MS^2) / ((γ - 1) * MS^2 + 2)
    u2 = c1 * 2 / (γ + 1) * (MS^2 - 1) / MS
    T2 = p2 / (ρ2 * RL)
    eos2 = IdealMixture([IdealSpecies{Float64}("light", RL, γ),
                         IdealSpecies{Float64}("heavy", RH, γ)])

    # `xlo < 0` extends the domain upstream at the same spacing, which is how
    # the reference below is built. The heavy gas is at the driven gas's p and
    # T_ion, so RH sets the density ratio: Atwood 0.6 here.
    build(xbc; xlo=0.0) = begin
        n = round(Int, (1.0 - xlo) / Hx) + 1
        δ = 2Hx
        setup(Problem(eos=eos2, transport=Transport(mu0=0.0),
                      domain=((xlo, xlo + (n - 1) * Hx), (0.0, Hx), (0.0, Hx)),
                      bcs=(xbc, per3[2], per3[3]),
                      ic=(x, y, z) -> begin
                          s = tanh_blend(x, XS, δ)     # 0 shocked, 1 unshocked
                          θ = tanh_blend(x, XI, δ)     # 0 light,   1 heavy
                          Prim(Y=(1 - θ, θ), u=((1 - s) * u2 + U, 0.0, 0.0),
                               p=(1 - s) * p2 + s * p1,
                               T_ion=(1 - s) * T2 + s * T1)
                      end),
              Numerics(n_global=(n, 1, 1), art=ArtParams(enabled=true),
                       cfl=0.4, filter_interval=1))
    end

    inflow() = NSCBCInflowBC(u=(u2 + U, 0.0, 0.0), T_ion=T2, Y=[1.0, 0.0])
    # sigma small on purpose. The pressure this boundary will face once the
    # reflected shock has passed is the post-reflected-shock pressure (~3.23),
    # which is the answer to the interaction and not something the test may
    # assume; relaxing hard toward `pinf` would impose a pressure known to be
    # wrong and drag the driven section with it. A weak relaxation is the
    # honest reading of "let the wave out and do not hold a pressure I do not
    # know" — measured: sigma 0.25 costs a factor 3.5 in the agreement below.
    outflow() = NSCBCOutflowBC(pinf=p2, sigma=0.05)
    # Nothing reaches the downstream end within TEND (the transmitted shock
    # gets to x ~ 0.8), so a Dirichlet on the unshocked heavy state is exact
    # there and keeps this test about the upstream boundary alone.
    downstream() = DirichletBC((x, y, z, t) -> Prim(Y=(0.0, 1.0),
                                                    u=(U, 0.0, 0.0), p=p1,
                                                    T_ion=T1))

    # "The reflected shock has reached the upstream plane." A rank-local verdict
    # on the plane this rank owns, left to WhenState to reduce — the shape the
    # WhenState docstring asks for, and the reason a switch must never be driven
    # from an unreduced rank-local test.
    arrived(solver, Q) = begin
        plane = CL.wallplane(solver.decomp, 1, 1)
        plane === nothing && return false
        any(I -> solver.p[I] > 1.05 * p2, plane)
    end

    xline(s, Q) = begin
        CL.exchange_state!(Q, s.decomp)
        CL.primitives!(s, Q)
        nx = s.decomp.n_local[1]
        ([xcoord(s, 1, i) for i in 1:nx],
         [s.p[gidx(s, i, 1, 1)] for i in 1:nx],
         [s.u[gidx(s, i, 1, 1)] for i in 1:nx],
         [s.Y[2][gidx(s, i, 1, 1)] for i in 1:nx])
    end
    # Interface position, interpolated across the cell where Y_heavy crosses
    # 1/2 — a sub-cell measure, since a re-shock moves the interface by a
    # fraction of a cell over this run.
    ipos(x, Y) = begin
        i = findfirst(>=(0.5), Y)
        x[i-1] + (0.5 - Y[i-1]) / (Y[i] - Y[i-1]) * (x[i] - x[i-1])
    end

    # (1) Switched on arrival.
    sw = SwitchableBC(inflow(), outflow())
    s_sw, Q_sw = build((sw, downstream()))
    fired_step = Ref(0)
    Q_fired = Ref{Any}(nothing)
    run!(s_sw, Q_sw; tfinal=TEND, nmax=100_000,
         callback=Callback(WhenState(arrived),
                           (s, Q) -> (switch!(sw); fired_step[] = s.step;
                                      Q_fired[] = copy(Q); nothing)))
    @test switched(sw)
    @test 0 < fired_step[] < 100_000          # it really did fire

    # (2) Inflow held throughout — the control, and the same run bit-for-bit up
    #     to the switch. WhenState does not clip dt, so the step sequences agree
    #     and this is an equality, not an approximation.
    s_h, Q_h = build((inflow(), downstream()))
    run!(s_h, Q_h; tfinal=TEND, nmax=fired_step[])
    @test s_h.step == fired_step[]
    @test Q_h == Q_fired[]
    run!(s_h, Q_h; tfinal=TEND, nmax=100_000)

    # (3) The reference: the same problem with the upstream boundary one domain
    #     length further away, so the reflected shock never reaches it and no
    #     boundary treatment can matter. This is what makes the comparison a
    #     measurement rather than a difference of two guesses — a non-reflecting
    #     condition is only ever tested against a domain big enough not to need
    #     one. Same spacing, so the grids coincide on x >= 0 to round-off.
    s_r, Q_r = build((inflow(), downstream()); xlo=-1.0)
    run!(s_r, Q_r; tfinal=TEND, nmax=100_000)

    xr, pr, ur, Yr = xline(s_r, Q_r)
    xh, ph, uh, Yh = xline(s_h, Q_h)
    xs, ps, us, Ys = xline(s_sw, Q_sw)
    keep = [i for i in eachindex(xr) if xr[i] >= -1e-9]
    @test maximum(abs, xr[keep] .- xh) < 1e-12

    win = [i for i in eachindex(xh) if 0.05 <= xh[i] <= 0.95]
    errp(p) = maximum(abs.(p[win] .- pr[keep][win]))
    erru(u) = maximum(abs.(u[win] .- ur[keep][win]))
    ref_i = ipos(xr, Yr)

    # The measurement. Held, the inflow condition keeps imposing region 2 after
    # the reflected shock has arrived and the error radiates back over the
    # interface; switched, the wave leaves. Measured against the reference:
    # pressure 0.112 held vs 0.017 switched, velocity 0.0289 vs 0.0027,
    # interface displacement 0.0031 vs 0.0006 (about a tenth of a cell).
    # Guards are ratios plus loose absolute bounds: this is a nonlinear run
    # through the artificial-property sensor, so it is not bit-reproducible and
    # only the order of magnitude is being asserted.
    @test errp(ps) < 0.04
    @test errp(ph) > 0.08
    @test errp(ph) > 3 * errp(ps)
    @test erru(us) < 0.008
    @test erru(uh) > 0.02
    @test erru(uh) > 3 * erru(us)
    @test abs(ipos(xs, Ys) - ref_i) < 0.0015
    @test abs(ipos(xh, Yh) - ref_i) > 2.5 * abs(ipos(xs, Ys) - ref_i)

    # After the switch the wrapper must not merely resemble `after` but *be* it.
    # Compared at the RHS rather than by restarting a run, because compute_dt
    # reads the previous step's artificial coefficients, so a fresh solver takes
    # a different first step from the same state and the two step sequences part
    # company for reasons that have nothing to do with the boundary. One
    # evaluation on a shared, richly non-uniform state isolates the forwarding
    # itself — both enforce! and correct_rhs!.
    s_p, Q_p = build((outflow(), downstream()))
    copyto!(Q_p, Q_sw)
    s_p.t = s_sw.t
    s_p.step = s_sw.step
    s_p.tstage = s_sw.tstage
    apply_bcs!(s_sw, Q_sw)
    apply_bcs!(s_p, Q_p)
    @test Q_sw == Q_p                          # enforce! forwards
    dQ_sw, dQ_p = zero(Q_sw), zero(Q_p)
    compute_rhs!(s_sw, Q_sw, dQ_sw)
    compute_rhs!(s_p, Q_p, dQ_p)
    @test maximum(abs, dQ_sw) > 1              # a non-trivial state
    @test dQ_sw == dQ_p                        # correct_rhs! forwards
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

# Not solver code, but `bench/` and the cluster scripts are not imported by this
# suite, so if these are not tested here they are not tested anywhere — and the
# whole reason they exist is to turn a silent wrong-default run into an error.
@testset "script argument parsing" begin
    defaults = (N=32, tfinal=10.0, configs="off:1", progress=0, quiet=false)
    pos = (:N, :tfinal)

    @test script_args(String[], defaults; positional=pos) == defaults
    opt = script_args(["128", "2.5", "progress=200", "configs=on:1:0.008"],
                      defaults; positional=pos)
    @test (opt.N, opt.tfinal, opt.progress) == (128, 2.5, 200)
    @test opt.configs == "on:1:0.008"      # String defaults pass through verbatim
    # Order is free, and a positional may be supplied by name instead.
    @test script_args(["progress=5", "64"], defaults; positional=pos).N == 64
    @test script_args(["N=64"], defaults; positional=pos).N == 64
    @test script_args(["nmax=1"], (nmax=typemax(Int),)).nmax == 1

    for text in ("true", "yes", "on", "1")
        @test script_args(["quiet=$text"], defaults; positional=pos).quiet
    end
    for text in ("false", "no", "off", "0")
        @test !script_args(["quiet=$text"], defaults; positional=pos).quiet
    end

    # The point of the exercise: every one of these was a silent default under
    # the environment-variable form this replaced.
    @test_throws ArgumentError script_args(["progerss=1"], defaults; positional=pos)
    @test_throws ArgumentError script_args(["progress=lots"], defaults; positional=pos)
    @test_throws ArgumentError script_args(["quiet=maybe"], defaults; positional=pos)
    @test_throws ArgumentError script_args(["1", "2", "3"], defaults; positional=pos)
    @test_throws ArgumentError script_args(["64"], defaults)
    @test_throws ArgumentError script_args(String[], defaults; positional=(:nope,))

    @test script_grid("128") == (128, 128, 128)
    @test script_grid("256,256,512") == (256, 256, 512)
    @test_throws ArgumentError script_grid("128,256")
    @test_throws ArgumentError script_grid("128x128")
end

# HDF5 is a weak dependency and is not loadable from the package environment
# alone, so the extension tests run only where it is present. The skip is
# printed rather than silent.
if (try
        @eval using HDF5
        true
    catch
        false
    end)
    include("hdf5_tests.jl")
else
    println("HDF5 not loadable in this environment — extension tests SKIPPED. " *
            "Run test/hdf5_tests.jl from an environment carrying both, and " *
            "under mpiexec for the decomposition-independent restart.")
end

# A Makie backend is likewise a weak dependency, absent from the package
# environment. The extraction API it exercises is exported by the core, so this
# skip only foregoes the extension's plotting methods and the collective-profile
# check under decomposition.
if (try
        @eval using CairoMakie
        true
    catch
        false
    end)
    include("makie_tests.jl")
else
    println("Makie backend not loadable in this environment — extension tests " *
            "SKIPPED. Run test/makie_tests.jl from the docs environment, and " *
            "under mpiexec for the decomposition-independent profile.")
end

println("serial tests complete")
