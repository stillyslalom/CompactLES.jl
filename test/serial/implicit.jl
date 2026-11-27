# The serial suite's tests of the implicit diffusion stage and the additive
# Runge–Kutta pair. Included by test/serial_suite.jl, which loads the package and
# defines the imports and helpers used here; not runnable on its own.

@testset "implicit diffusion stage" begin
    # Tridiagonal lines with their own coefficients against a dense solve,
    # closed and periodic, on one rank (the MPI suite splits them).
    for periodic in (false, true)
        n, L = 7, 3
        vl = CL.VariableLines{Float64}(n, L, MPI.COMM_SELF, 1, 0, periodic)
        vl.lower .= -rand(n, L); vl.upper .= -rand(n, L)
        periodic || (vl.lower[1, :] .= 0; vl.upper[n, :] .= 0)
        vl.diag .= 0.1 .+ rand(n, L) .- vl.lower .- vl.upper
        CL.factor_lines!(vl)
        B = randn(n, L); X = CL.solve_variable_lines!(copy(B), vl)
        for l in 1:L
            A = diagm(vl.diag[:, l])
            for i in 1:n
                A[i, mod1(i - 1, n)] += vl.lower[i, l]
                A[i, mod1(i + 1, n)] += vl.upper[i, l]
            end
            @test norm(A * X[:, l] - B[:, l]) < 1e-12 * norm(B[:, l])
        end
    end

    # ∇·(κ∇T) in the metric's coordinates, the inner derivative by a complex
    # step and the outer by a sixth-order central difference.
    function divergence(metric, active, Tfn, κfn, x)
        J(y) = prod(CL.scalefactors(metric, y...))
        total = 0.0
        for d in 1:3
            active[d] || continue
            e = ntuple(k -> k == d ? 1.0 : 0.0, 3)
            flux(s) = (y = x .+ s .* e;
                       J(y) / CL.scalefactors(metric, y...)[d]^2 * κfn(y...) *
                       imag(Tfn((y .+ 1e-30im .* e)...)) / 1e-30)
            δ = 1e-4
            total += sum(c * (flux(m * δ) - flux(-m * δ))
                         for (m, c) in enumerate((3 / 4, -3 / 20, 1 / 60))) / δ
        end
        return total / J(x)
    end
    noart = ArtificialProperties(enabled=false)
    walls = (SlipWallBC(), SlipWallBC())
    axis = ((AxisBC(), SlipWallBC()), per3[2], per3[3])
    gauss(r) = exp(-16r^2)
    # name, solver of scale N, wall parity, whether V L is symmetric, the
    # manufactured order and its tolerance, T and κ. The shell's curved walls
    # are first order at the wall node, which the solve carries into a
    # second-order solution; on the resolved axis T is negligible at the wall.
    cases = (
        ("walls", N -> build_solver(
            Problem(domain=((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)),
                    bcs=(walls, walls, per3[3]), ic=nothing),
            Numerics(n_global=(N + 1, N + 1, 1), art=noart)), 1, true, 6.0,
         (x, y, z) -> cospi(x) * cospi(y) + 0.3cospi(2x),
         (x, y, z) -> 1 + 0.5cospi(x) * cospi(2y)),
        ("isothermal walls", N -> build_solver(
            Problem(domain=((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)),
                    bcs=(walls, walls, per3[3]), ic=nothing),
            Numerics(n_global=(N + 1, N + 1, 1), art=noart)),
         -1, true, 6.0, (x, y, z) -> sinpi(x) * sinpi(y) * (1 + 0.3cospi(x)),
         (x, y, z) -> 1 + 0.5cospi(x) * cospi(2y)),
        ("symmetry planes", N -> build_solver(
            Problem(domain=((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)),
                    bcs=((SymmetryPlaneBC(), SymmetryPlaneBC()), walls, per3[3]), ic=nothing),
            Numerics(n_global=(N, N + 1, 1), art=noart)), 1, true, 6.0,
         (x, y, z) -> cospi(x) * cospi(y) + 0.3cospi(2x),
         (x, y, z) -> 1 + 0.5cospi(x) * cospi(2y)),
        ("stretched walls", N -> build_solver(
            Problem(domain=((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)),
                    bcs=(walls, walls, per3[3]), ic=nothing),
            Numerics(n_global=(N + 1, N + 1, 1), art=noart,
                     stretch=(sine_cluster(0.0, 1.0, 0.5, 0.4), nothing,
                                                          nothing))), 1, true, 6.0,
         (x, y, z) -> cospi(x) * cospi(y) + 0.3cospi(2x),
         (x, y, z) -> 1 + 0.5cospi(x) * cospi(2y)),
        ("cylindrical shell", N -> build_solver(
            Problem(domain=((0.5, 1.5), (0.0, 2π), (0.0, 1.0)),
                    metric=CylindricalMetric(), bcs=(walls, per3[2], per3[3]), ic=nothing),
            Numerics(n_global=(N + 1, N, 1), art=noart)),
         1, true, 2.0, (r, θ, z) -> cospi(r - 0.5) * (1 + 0.3cos(θ)),
         (r, θ, z) -> 1 + 0.3r * sin(θ)),
        ("resolved axis", N -> build_solver(
            Problem(domain=((0.0, 1.5), (0.0, 2π), (0.0, 1.0)), metric=CylindricalMetric(),
                    bcs=axis, ic=nothing),
            Numerics(n_global=(N, N, 1), art=noart)),
         1, false, 6.0, (r, θ, z) -> gauss(r) * (1 + r * cos(θ)),
         (r, θ, z) -> 1 + 0.5r^2 + 0.2r * sin(θ)),
    )
    γΔt = 0.01
    for (label, build, parity, symmetric, order, Tfn, κfn) in cases
        errs = Float64[]; hs = Float64[]
        for N in (16, 32)
            solver = build(N)
            decomp = solver.decomp
            stage = CL.DiffusionStage(solver; parity=parity)
            @test stage.symmetric == symmetric
            κ = field(decomp); exact = field(decomp); rhs = field(decomp)
            fillf!(solver, κ, κfn); fillf!(solver, exact, Tfn)
            fillf!(solver, rhs, (x...) -> Tfn(x...) - γΔt * divergence(
                solver.metric, decomp.active, Tfn, κfn, Float64.(x)))
            pad = CartesianIndex(decomp.n_halo_d)
            for I in CartesianIndices(decomp.n_local)
                stage.dirichlet[I] && (rhs[I+pad] = exact[I+pad] = 0)
            end
            T = copy(rhs)
            result = CL.solve_stage!(T, stage, rhs, κ, γΔt)
            @test result.converged
            inner = interior(decomp)
            push!(errs, maximum(abs, T[inner] .- exact[inner]))
            push!(hs, solver.h[1])
            # An odd temperature stays zero on the wall nodes.
            parity < 0 && @test all(T[I+pad] == 0 for I in CartesianIndices(decomp.n_local)
                                    if stage.dirichlet[I])
            N == 16 || continue
            # The multigrid cycle is a symmetric positive definite operator.
            u = field(decomp); v = field(decomp); Bu = field(decomp); Bv = field(decomp)
            u[inner] .= randn(size(inner)); v[inner] .= randn(size(inner))
            CL._precondition!(Bu, stage, u); CL._precondition!(Bv, stage, v)
            @test abs(dot(u[inner], Bv[inner]) - dot(v[inner], Bu[inner])) <
                  1e-12 * abs(dot(u[inner], Bu[inner]))
            @test dot(u[inner], Bu[inner]) > 0
            # A uniform right-hand side returns unchanged without iterating.
            parity < 0 && continue
            rhs .= 1.3; T .= 1.3
            uniform = CL.solve_stage!(T, stage, rhs, κ, 10.0)
            @test uniform.iterations == 0
            @test maximum(abs, T[inner] .- 1.3) < 1e-13
        end
        @test log(errs[1] / errs[2]) / log(hs[1] / hs[2]) > order - 0.3
    end

    # Iteration counts flat in the grid and in the step, from the explicit
    # limit to four orders beyond it, for conjugate gradients on the walls and
    # GMRES on the resolved axis.
    for (build, parity) in ((cases[1][2], 1), (cases[6][2], 1))
        counts = Int[]
        for N in (16, 32), γ in (1e-3, 1e-1, 1e1)
            solver = build(N)
            stage = CL.DiffusionStage(solver; parity=parity)
            κ = fillf!(solver, field(solver.decomp), (x, y, z) -> 1 + 0.5x * sin(3y))
            rhs = fillf!(solver, field(solver.decomp),
                         (x, y, z) -> exp(-3(x - 0.3)^2) * cos(y))
            result = CL.solve_stage!(copy(rhs), stage, rhs, κ, γ)
            @test result.converged
            push!(counts, result.iterations)
        end
        @test maximum(counts) <= 20
        @test all(abs.(counts[4:6] .- counts[1:3]) .<= 3)
    end

    # A spherical ball with its origin and poles, on the GMRES path: the
    # residual of the solve and a uniform right-hand side.
    solver = build_solver(
        Problem(domain=((0.0, 1.5), (0.0, π), (0.0, 2π)), metric=SphericalMetric(),
                bcs=((OriginBC(), SlipWallBC()), (PoleBC(), PoleBC()), per3[3]), ic=nothing),
        Numerics(n_global=(12, 12, 12), art=noart))
    decomp = solver.decomp
    stage = CL.DiffusionStage(solver)
    κ = fillf!(solver, field(decomp), (r, θ, φ) -> 1 + 0.5r^2 + 0.2r * sin(θ) * cos(φ))
    rhs = fillf!(solver, field(decomp), (r, θ, φ) -> gauss(r) * (1 + 0.5r * cos(θ)))
    T = copy(rhs)
    result = CL.solve_stage!(T, stage, rhs, κ, 0.1)
    out = CL.stage_operator!(field(decomp), stage, T, κ, 0.1)
    inner = interior(decomp)
    V = stage.volume[inner]
    @test result.converged
    @test sqrt(sum(V .* (out[inner] .- rhs[inner]) .^ 2) / sum(V .* rhs[inner] .^ 2)) < 1e-9
    rhs .= 2.0; T .= 2.0
    @test CL.solve_stage!(T, stage, rhs, κ, 10.0).iterations == 0
    @test maximum(abs, T[inner] .- 2) < 1e-13

    @test_throws ArgumentError CL.DiffusionStage(build_solver(
        Problem(domain=((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)), bcs=per3, ic=nothing),
        Numerics(n_global=(16, 16, 1), art=noart, execution=Execution(patch_grid=(2, 1, 1)))))
end

@testset "additive Runge–Kutta conduction" begin
    # The tableau: row sums, and every order condition of a two-part
    # additive pair to fourth order, the embedded weights to third. a^I is
    # exact; a^E is published to thirteen digits.
    big(A) = Rational{BigInt}.(A)
    AE, AI = big(CL.ARK436_EXPLICIT), big(CL.ARK436_IMPLICIT)
    b, bh, c = big(CL.ARK436_WEIGHTS), big(CL.ARK436_EMBEDDED), big(CL.ARK436_NODES)
    defect(x) = abs(Float64(x))
    @test all(i -> AE[i, i] == 0 && all(j -> AE[i, j] == 0 && AI[i, j] == 0, i+1:6), 1:6)
    @test all(i -> AI[i, i] == (i == 1 ? 0 : 1 // 4), 1:6)
    @test AI[6, :] == b                               # stiffly accurate
    one6 = ones(Rational{BigInt}, 6)
    @test maximum(defect, AE * one6 .- c) < 1e-24
    @test AI * one6 == c
    for (w, order) in ((b, 4), (bh, 3))
        conditions = Any[sum(w) - 1, w' * c - 1 // 2, w' * c .^ 2 - 1 // 3]
        for X in (AE, AI)
            push!(conditions, w' * (X * c) - 1 // 6)
        end
        if order == 4
            push!(conditions, w' * c .^ 3 - 1 // 4)
            for X in (AE, AI)
                push!(conditions, w' * (c .* (X * c)) - 1 // 8,
                      w' * (X * c .^ 2) - 1 // 12)
                for Y in (AE, AI)
                    push!(conditions, w' * (X * (Y * c)) - 1 // 24)
                end
            end
        end
        @test maximum(defect, conditions) < 1e-24
    end
    @test all(==(0), (b' * (AI * c) - 1 // 6, b' * (AI * (AI * c)) - 1 // 24,
                      bh' * (AI * c) - 1 // 6))

    # The conduction pulse on [-1, 1): c_v = 1, κ = α, and a gas constant small
    # enough that the flow the pulse drives stays below the time error
    # (bench/archive/imexconduction.jl, which also measures the step rules).
    α, σ, A = 1.0, 0.2, 0.5
    t0 = σ^2 / (2α)
    noart = ArtificialProperties(enabled=false)
    pulse(x, t) = 1 + A * sqrt(t0 / (t + t0)) *
                  sum(exp(-(x + 2m)^2 / (4α * (t + t0))) for m in -4:4)
    function line(n; implicit=ImplicitConduction(step_rule=:none), gas=1e-10,
                  control=StepControl())
        h = 2 / n
        mu = 1e-3 * α * h^2
        build_solver(
            Problem(domain=((-1.0, -1.0 + 2.0), (0.0, 0.0 + 1.0), (0.0, 0.0 + 1.0)),
                    bcs=per3, eos=IdealSpecies("gas"; R=gas, gamma=1 + gas),
                    transport=ConstantTransport(mu0=mu, Pr=mu * (1 + gas) / α, Sc=1.0),
                    ic=nothing),
            Numerics(n_global=(n, 1, 1), art=noart, cfl=1e6, implicit=implicit,
                     control=control, filter=StateFilter(interval=0)))
    end
    function start(solver, T0)
        Q = allocate_state(solver)
        initialize!(solver, Q, (x, y, z) -> Prim(rho=1.0, u=(0.0, 0.0, 0.0),
                                                 p=1e-10 * T0(x)))
        Q
    end
    function steps!(solver, Q, tf, n)
        for k in 1:n
            run!(solver, Q; tfinal=k == n ? tf : k * tf / n)
        end
        Q
    end
    temperature(solver, Q, i) =
        (refresh_primitives!(solver, Q); solver.T_ion[padded_index(solver, i, 1, 1)])

    n = 256
    limit = (2 / n)^2 / (2α)                          # forward-Euler diffusive limit
    tf = 4t0
    solver = line(n)
    @test solver.transport isa CL.WithoutConduction
    @test solver.implicit isa CL.ImexIntegrator
    Q = start(solver, x -> pulse(x, 0.0))
    nsteps = round(Int, tf / (100limit))             # R = 100
    steps!(solver, Q, tf, nsteps)
    refresh_primitives!(solver, Q)
    err = maximum(abs(solver.T_ion[padded_index(solver, i, 1, 1)] -
                      pulse(xcoord(solver, 1, i), tf)) for i in 1:n) / A
    @test err < 2e-5
    @test solver.implicit.accepted == nsteps && solver.implicit.rejected == 0
    @test solver.implicit.solves == 5nsteps          # one per implicit stage

    # Stiff stability: one step at R = 1e5 removes a grid-Nyquist mode.
    solver = line(64)
    h = 2 / 64
    Q = start(solver, x -> 1 + 0.01 * (-1)^round(Int, (x + 1) / h))
    nyquist(solver, Q) = abs(sum((-1)^(i - 1) * temperature(solver, Q, i)
                                 for i in 1:64) / 64)
    @test nyquist(solver, Q) ≈ 0.01
    run!(solver, Q; tfinal=1e5 * h^2 / (2α))
    @test nyquist(solver, Q) < 1e-6

    # A stage that does not converge within its Krylov budget ends the step
    # with SolverFailure, which takes StepControl's rollback: each retry
    # lowers the CFL and caps the step below the one that failed, and the
    # failure is raised once the retries are spent.
    tight = ImplicitConduction(step_rule=:none, max_iterations=2, rtol=1e-10)
    tf = 1e4 * h^2 / (2α)
    solver = line(64; implicit=tight)
    Q = start(solver, x -> pulse(x, 0.0))
    failure = try
        run!(solver, Q; tfinal=tf)
        nothing
    catch e
        e
    end
    @test failure isa SolverFailure && failure.reason === :implicit_solve
    @test failure.dt == tf
    solver = line(64; implicit=tight, control=StepControl(retries=2))
    Q = start(solver, x -> pulse(x, 0.0))
    failure = try
        run!(solver, Q; tfinal=tf)
        nothing
    catch e
        e
    end
    @test failure isa SolverFailure && failure.reason === :implicit_solve
    @test solver.cfl == 1e6 / 4 && failure.dt == tf / 4
    @test solver.implicit.dt_cap == tf / 8

    # The step rules: `:error` rejects and subdivides an attempt beyond its
    # tolerance, and the state lands at the requested endpoint either way.
    for rule in (ImplicitConduction(tolerance=1e-4),
                 ImplicitConduction(step_rule=:temperature, target_change=0.02))
        solver = line(128; implicit=rule)
        Q = start(solver, x -> pulse(x, 0.0))
        run!(solver, Q; tfinal=t0)
        @test solver.t == t0
        refresh_primitives!(solver, Q)
        @test maximum(abs(solver.T_ion[padded_index(solver, i, 1, 1)] -
                          pulse(xcoord(solver, 1, i), t0)) for i in 1:128) / A < 1e-2
    end

    # The default integrator is untouched, and the unsupported ends are
    # refused.
    plain = build_solver(
        Problem(domain=((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)), bcs=per3, ic=nothing),
        Numerics(n_global=(16, 1, 1), art=noart))
    @test plain.implicit === nothing && plain.transport isa ConstantTransport
    iso = NoSlipWallBC(Twall=1.0)
    @test_throws ArgumentError build_solver(
        Problem(domain=((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)),
                bcs=((iso, iso), per3[2], per3[3]), ic=nothing),
        Numerics(n_global=(16, 1, 1), art=noart, implicit=ImplicitConduction()))
    @test_throws ArgumentError build_solver(
        Problem(domain=((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)), bcs=per3, ic=nothing),
        Numerics(n_global=(32, 1, 1), art=noart, implicit=ImplicitConduction(),
                 execution=Execution(patch_grid=(2, 1, 1))))
    @test_throws ArgumentError ImplicitConduction(step_rule=:splitting)
end
