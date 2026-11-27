# The serial suite's tests of the line solves, the compact derivatives and the
# staggered operators. Included by test/serial_suite.jl, which loads the package and
# defines the imports and helpers used here; not runnable on its own.

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

@testset "reduced interface band LU vs dense" begin
    # The reduced stage of P ranks from random spike corner blocks: every
    # rank's band solve against a dense solve of the same matrix, closed and
    # periodic, the second with spikes large enough that the band LU pivots.
    rng = MersenneTwister(12)
    for q in (1, 2), periodic in (false, true), P in (2, 3, 4, 7), scale in (0.4, 1.5)
        m2, L = 2q, CL.REDUCED_BLOCK + 22   # two blocks of lines
        allb = zeros(4q * q * P)
        for rk in 0:(P-1), block in 0:3, t in 1:q, r in 1:q
            # V toward the previous rank is absent at a closed line's low end,
            # W toward the next at its high end.
            absent = !periodic && ((rk == 0 && block <= 1) || (rk == P - 1 && block >= 2))
            allb[4q*q*rk + block*q*q + (t-1)*q + r] =
                absent ? 0.0 : scale * (2rand(rng) - 1)
        end
        R = zeros(m2 * P, m2 * P)
        CL._reduced_entries!((i, j, v) -> (R[i, j] += v), allb, q, P)
        gath = randn(rng, m2, L, P)
        z = reduce(vcat, [gath[:, :, rk+1] for rk in 0:(P-1)])
        exact = R \ z
        for p in 0:(P-1)
            red, band = CL._reduced_factor(allb, q, P, p, periodic, L)
            @test red === nothing
            @test CL._band_matrix(band) ≈ R rtol = 1e-14
            zbp, zbn = zeros(L, q), zeros(L, q)
            CL._band_reduced!(band, gath, zbp, zbn, q, L)
            cprev, cnext = m2 * mod(p - 1, P) + q, m2 * mod(p + 1, P)
            tol = 1e-13 * cond(R)
            @test maximum(abs, zbp' .- exact[cprev+1:cprev+q, :]) < tol
            @test maximum(abs, zbn' .- exact[cnext+1:cnext+q, :]) < tol
            # Bitwise independent of the number of lines per call.
            zbp2, zbn2 = zeros(L, q), zeros(L, q)
            CL._band_reduced!(band, gath, zbp2, zbn2, q, 5)
            @test zbp2[1:5, :] == zbp[1:5, :] && zbn2[1:5, :] == zbn[1:5, :]
        end
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
    # C10 error on this field is ≈2000× below C6; the guard requires 100×.
    s6 = mkslv(n_global=(32, 32, 32), deriv=lele_d1_6())
    f6 = CL.field(s6.decomp); df6 = CL.field(s6.decomp)
    fillf!(s6, f6, (x, y, z) -> sin(3x) * cos(2y))
    CL.exchange_halos!(f6, s6.decomp)
    CL.deriv_along!(df6, f6, s6, 1, 1); CL._scale_grad!(df6, s6, 1)
    @test e10 < ferr(s6, df6, (x, y, z) -> 3cos(3x) * cos(2y)) / 100
    # Closed domain: deg-3 polynomial is exact through the C10 closure rows
    # (a distinct band path: closure substitution, V = W = 0, no reduced stage).
    sc = build_solver(
        Problem(domain=((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)),
                bcs=((SlipWallBC(), SlipWallBC()), per3[2], per3[3]), ic=nothing),
        Numerics(n_global=(32, 12, 12), deriv=lele_d1_10(),
                 art=ArtificialProperties(enabled=false)))
    fc = CL.field(sc.decomp); dfc = CL.field(sc.decomp)
    fillf!(sc, fc, (x, y, z) -> 1 + 2x + 3x^2 - x^3)
    CL.exchange_halos!(fc, sc.decomp)
    CL.deriv_along!(dfc, fc, sc, 1, 1); CL._scale_grad!(dfc, sc, 1)
    @test ferr(sc, dfc, (x, y, z) -> 2 + 6x - 3x^2) < 1e-10
end

@testset "C10 interface closures: two rows per end, exact to degree 9" begin
    # The pentadiagonal scheme's patch-interface rows: the two edge rows
    # whose left-hand side would couple a ghost unknown become compact rows
    # Taylor-matched to order 10 on a 9-point right-hand side, row 1
    # one-sided on the left-hand side (nodes 1, 2, 3) and reading four ghost
    # layers, row 2 a centered tridiagonal row. A closed line whose ghosts
    # carry the analytic function must therefore differentiate a degree-9
    # polynomial to round-off through both ends, where the scheme's own C6
    # cascade would leave a third-order error.
    rows = CL.interface_closures(lele_d1_10())
    @test length(rows) == 2
    @test [r.first for r in rows] == [-3, -2]
    @test rows[1].lhs == [0, 0, 1, 8 / 5, 2 / 5]
    @test rows[2].lhs == [0, 2 / 5, 1, 2 / 5, 0]
    @test all(r -> abs(sum(r.rhs)) < 1e-15, rows)
    @test all(r -> length(r.rhs) == 9, rows)
    ring = CL.interface_closures(compact_d8())
    @test [(r.first, r.rhs) for r in ring] == [(1, [1.0]), (2, [1.0])]
    # The detector's own interface rows keep the undivided eighth difference:
    # degree-7 polynomials are annihilated, and the response at k = π is
    # the interior's 16, row 2 also matching its k → 0 ratio 1/232.
    d8rows = CL._ring_interface_rows(Float64)
    @test [r.first for r in d8rows] == [-3, -2]
    @test all(r -> all(n -> abs(sum(r.rhs .* (-4:4) .^ n)) < 1e-12, 0:7), d8rows)
    nyquist(r) = sum(r.rhs .* (-1) .^ (-4:4)) / (1 - 2r.lhs[2])
    @test all(r -> nyquist(r) ≈ 16, d8rows)
    @test d8rows[2].rhs[5] / 70 / (1 + 2d8rows[2].lhs[2]) ≈ 1 / 232
    n = 40
    h = 0.1
    d = Decomp((n, 4, 4), (false, true, true); dims=(1, 1, 1))
    plan = CL.plan_direction(d, lele_d1_10(), 1, h; lo_closures=rows, hi_closures=rows)
    p9(x) = 1 + x - 0.5x^2 + 0.3x^3 - 0.2x^4 + 0.1x^5 - 0.05x^6 + 0.02x^7 -
            0.01x^8 + 0.004x^9
    dp9(x) = 1 - x + 0.9x^2 - 0.8x^3 + 0.5x^4 - 0.3x^5 + 0.14x^6 - 0.08x^7 +
             0.036x^8
    f = CL.field(d); df = CL.field(d)
    pad = d.n_halo_d
    for k in axes(f, 3), j in axes(f, 2), i in axes(f, 1)
        f[i, j, k] = p9((i - pad[1] - 1) * h)
    end
    CL.apply_along!(df, plan, f, d)
    e = maximum(abs(df[i+pad[1], pad[2]+1, pad[3]+1] - dp9((i - 1) * h)) for i in 1:n)
    @test e < 1e-10                     # measured 2.0e-12
    # The same line under the scheme's own closures is third order at the
    # edge rows, so the interface rows are what carry the exactness.
    own = CL.plan_direction(d, lele_d1_10(), 1, h)
    CL.apply_along!(df, own, f, d)
    eo = maximum(abs(df[i+pad[1], pad[2]+1, pad[3]+1] - dp9((i - 1) * h)) for i in 1:n)
    @test eo > 1e-4
    # A narrower halo than the rows read is refused with the width named.
    d3 = Decomp((n, 4, 4), (false, true, true); dims=(1, 1, 1), n_halo=3)
    @test_throws "n_halo ≥ 4" CL.plan_direction(d3, lele_d1_10(), 1, h;
                                                lo_closures=rows, hi_closures=rows)
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
        solver = build_solver(
            Problem(domain=((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)), bcs=bcs, ic=nothing),
            Numerics(n_global=ng, art=ArtificialProperties(enabled=false)))
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
    solver = build_solver(
        Problem(domain=((0.0, 1.0), (0.0, 1.0), (0.0, 0.5)), metric=CylindricalMetric(),
                bcs=((AxisBC(), SlipWallBC()), per3[2], per3[3]), ic=nothing),
        Numerics(n_global=(64, 1, 12), deriv=lele_d1_10(),
                 art=ArtificialProperties(enabled=false)))
    f = CL.field(solver.decomp); df = CL.field(solver.decomp)
    fillf!(solver, f, (r, θ, z) -> r * exp(-4r^2))            # odd across the axis
    CL.exchange_halos!(f, solver.decomp)
    CL.deriv_along!(df, f, solver, 1, -1); CL._scale_grad!(df, solver, 1)
    # Both errors sit at the closed outer end, so they carry the default
    # closure's wall constant: measured 3.6e-6 and 2.1e-5 under `:neutral3`,
    # 2.4e-6 and 1.6e-5 under `:cascade3`.
    @test ferr(solver, df, (r, θ, z) -> (1 - 8r^2) * exp(-4r^2)) < 6e-6
    fillf!(solver, f, (r, θ, z) -> exp(-4r^2))                # even across the axis
    CL.exchange_halos!(f, solver.decomp)
    CL.deriv_along!(df, f, solver, 1, 1); CL._scale_grad!(df, solver, 1)
    @test ferr(solver, df, (r, θ, z) -> -8r * exp(-4r^2)) < 3e-5
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
    e = maximum(abs(d1[padded_index(solver, i, j, k)] - d2[padded_index(solver, j, i, k)])
                for i in 1:24, j in 1:24, k in 1:24)
    @test e < 1e-11
end

@testset "staggered operators: adjoint pair, telescoping, symmetry, sweep paths" begin
    # Dense matrices of the three staggered operators on one line of N nodes,
    # columns from unit vectors. The midpoint rows and columns are the first
    # N - 1 on a closed line, where slot N carries no midpoint.
    function staggered_matrices(N, periodic, parity)
        decomp = Decomp((N, 1, 1), (periodic, false, false))
        h = periodic ? 1 / N : 1 / (N - 1)
        plans = (CL.plan_staggered(decomp, :to_mid, 1, h; parity=parity),
                 CL.plan_staggered(decomp, :to_node, 1, h; parity=-parity),
                 CL.plan_staggered(decomp, :interpolate, 1, h; parity=1))
        f = field(decomp); out = field(decomp); pad = decomp.n_halo
        mats = map(plans) do plan
            M = zeros(N, N)
            for j in 1:N
                f .= 0; f[j+pad, 1, 1] = 1
                CL.exchange_dim!(f, decomp, 1)
                out .= NaN
                apply_along!(out, plan, f, decomp)
                M[:, j] = out[pad+1:pad+N, 1, 1]
            end
            M
        end
        return mats..., h
    end
    for N in (16, 17)
        D, G, Ip, h = staggered_matrices(N, true, 1)
        # Periodic: G is exactly the negative transpose of D, so L = -Dᵀ K D.
        @test norm(G + D') < 1e-12 * norm(D)
        @test norm(D * ones(N)) < 1e-12 * norm(D)
        @test norm(Ip * ones(N) .- 1) < 1e-14
        κ = 1 .+ 0.5 .* sin.(2π .* (0:N-1) ./ N)
        L = G * Diagonal(Ip * κ) * D
        @test norm(L - L') < 1e-12 * norm(L)
        @test maximum(eigvals(Symmetric((L + L') / 2))) < 1e-10 * norm(L)
        @test abs(sum(L * sin.(1:N))) < 1e-11 * norm(L)
    end
    for N in (16, 17), parity in (1, -1)
        D, G, Ip, h = staggered_matrices(N, false, parity)
        M = N - 1
        # The slot of midpoint N is zero whatever the input.
        @test all(iszero, D[N, :]) && all(iszero, Ip[N, :])
        D = D[1:M, :]; G = G[:, 1:M]; Ip = Ip[1:M, :]
        # Under the wall mirror the pair stays adjoint in the trapezoidal node
        # weights: W_n G = -Dᵀ W_m with W_m = h, on the nodes the parity
        # leaves free (all of them for the even temperature of an adiabatic
        # wall; the interior for the odd one of an isothermal wall).
        Wn = fill(h, N); Wn[1] = Wn[N] = h / 2
        free = parity == 1 ? (1:N) : (2:N-1)
        @test norm((Diagonal(Wn) * G .+ h .* D')[free, :]) < 1e-12 * norm(D) * h
        @test norm(Ip * ones(N) .- 1) < 1e-14
        κ = 1 .+ 0.5 .* cos.(π .* (0:N-1) ./ (N - 1))
        WL = (Diagonal(Wn) * G * Diagonal(Ip * κ) * D)[free, free]
        @test norm(WL - WL') < 1e-12 * norm(WL)
        @test maximum(eigvals(Symmetric((WL + WL') / 2))) < 1e-10 * norm(WL)
        if parity == 1
            # The even mirror is the adiabatic wall: a constant carries no
            # flux, and Σ W_n L T telescopes to the zero wall flux. (The odd
            # mirror passes flux through the wall and conserves nothing.)
            @test norm(D * ones(N)) < 1e-12 * norm(D)
            @test abs(sum(WL * sin.(1:N))) < 1e-11 * norm(WL)
        end
    end
    # The transposed y and z sweeps against the x sweep on permuted data, on a
    # decomposition with every dimension active.
    for periodic in (true, false), op in (:to_mid, :to_node, :interpolate)
        decomp = Decomp((12, 12, 12), (periodic, periodic, periodic))
        h = 1 / 12
        f = field(decomp); pad = decomp.n_halo
        for k in 1:12, j in 1:12, i in 1:12
            f[i+pad, j+pad, k+pad] = sin(0.7i + 0.2) * cos(0.3j) + 0.1k
        end
        CL.exchange_halos!(f, decomp)
        outs = map(1:3) do d
            out = field(decomp)
            g = permutedims(f, d == 1 ? (1, 2, 3) : d == 2 ? (2, 1, 3) : (3, 2, 1))
            apply_along!(out, CL.plan_staggered(decomp, op, d, h), g, decomp)
            permutedims(out, d == 1 ? (1, 2, 3) : d == 2 ? (2, 1, 3) : (3, 2, 1))
        end
        inner = ntuple(_ -> pad+1:pad+12, 3)
        @test maximum(abs.(outs[2][inner...] .- outs[1][inner...])) < 1e-12
        @test maximum(abs.(outs[3][inner...] .- outs[1][inner...])) < 1e-12
    end
end

@testset "staggered operators: half-offset folds and curvilinear metric" begin
    # A line of N nodes folded half a cell beyond an end is half of a longer
    # line carrying the mirrored data: between two planes, a periodic line of
    # 2N nodes; with a plane at one end and a wall at the other, a closed
    # line of 2N nodes with walls of the same parity at both ends. Every
    # operator on the folded line, the plane midpoints included (slot 0 at a
    # low plane, slot N at a high one), must equal the doubled line's.
    N = 12; h = 1 / N
    smooth(x) = exp(sin(3x) + 0.3cos(7x)) + 0.2x
    for (lo, hi) in ((true, true), (true, false), (false, true)), σ in (1, -1),
        p in (1, -1), op in (:to_mid, :to_node, :interpolate)
        (lo && hi) && p == -1 && continue          # no wall to take p
        periodic = lo && hi
        df = Decomp((N, 1, 1), (false, false, false))
        dd = Decomp((2N, 1, 1), (periodic, false, false))
        mid_in = op === :to_node
        # Node j of the doubled line sits at (j - N - 1/2) h and its midpoint
        # j at (j - N) h, with a plane at x = 0 (and between two planes a
        # second one at x = N h, the periodic wrap). The folded line is its
        # upper half below a low plane (index i + N) and its lower half above
        # a high one (index i).
        shift = lo ? N : 0
        onplane(x) = abs(x) < h / 4 || (periodic && abs(abs(x) - N * h) < h / 4)
        value(x) = onplane(x) && mid_in && σ < 0 ? 0.0 :
                   x >= 0 ? smooth(x) : σ * smooth(-x)
        pf = CL.plan_staggered(df, op, 1, h; parity=p, lo_fold=lo ? σ : nothing,
                               hi_fold=hi ? σ : nothing)
        pd = CL.plan_staggered(dd, op, 1, h; parity=p)
        pad = df.n_halo
        ff = field(df); fd = field(dd)
        for j in 1:2N
            fd[j+pad, 1, 1] = value((j - N - (mid_in ? 0.0 : 0.5)) * h)
        end
        for i in (mid_in && lo ? 0 : 1):N
            ff[i+pad, 1, 1] = fd[i+shift+pad, 1, 1]
        end
        CL.exchange_dim!(fd, dd, 1)
        outf = fill!(field(df), NaN); outd = field(dd)
        apply_along!(outf, pf, ff, df)
        apply_along!(outd, pd, fd, dd)
        rows = mid_in ? (1:N) : ((lo ? 0 : 1):N)
        e = maximum(abs(outf[i+pad, 1, 1] - outd[i+shift+pad, 1, 1]) for i in rows)
        @test e < 1e-12 * maximum(abs, outd[pad+1:pad+2N, 1, 1])
    end

    # The adjoint identity between two planes, with weight h on every node
    # and h/2 on the plane midpoints (slots 0 and N), holds for both parities,
    # and W_n L is symmetric, negative semidefinite and, for the even
    # temperature, conservative.
    for σ in (1, -1)
        df = Decomp((N, 1, 1), (false, false, false))
        plans = (CL.plan_staggered(df, :to_mid, 1, h; lo_fold=σ, hi_fold=σ),
                 CL.plan_staggered(df, :to_node, 1, h; lo_fold=-σ, hi_fold=-σ),
                 CL.plan_staggered(df, :interpolate, 1, h; lo_fold=1, hi_fold=1))
        pad = df.n_halo
        mids = 0:N
        dense(plan, rows, cols) = begin
            M = zeros(length(rows), length(cols)); f = field(df); out = field(df)
            for (c, j) in enumerate(cols)
                f .= 0; f[j+pad, 1, 1] = 1
                out .= NaN
                apply_along!(out, plan, f, df)
                M[:, c] = [out[i+pad, 1, 1] for i in rows]
            end
            M
        end
        D = dense(plans[1], mids, 1:N)
        G = dense(plans[2], 1:N, mids)
        Ip = dense(plans[3], mids, 1:N)
        Wm = fill(h, N + 1); Wm[1] = Wm[end] = h / 2
        @test norm(h .* G .+ D' * Diagonal(Wm)) < 1e-12 * norm(D) * h
        κ = 1 .+ 0.5 .* cospi.(((1:N) .- 0.5) ./ N)
        WL = h .* G * Diagonal(Ip * κ) * D
        @test norm(WL - WL') < 1e-12 * norm(WL)
        @test maximum(eigvals(Symmetric((WL + WL') / 2))) < 1e-10 * norm(WL)
        σ == 1 && @test abs(sum(WL * sin.(1:N))) < 1e-11 * norm(WL)
    end

    # The metric: J W_n L with J W_n the node volumes is symmetric, negative
    # semidefinite and conservative, tested on random vectors over the block,
    # on every geometry the solver builds but the folds whose area vanishes
    # oddly (the axis, the poles). There the flux continues smoothly through
    # the singular set, which is not the mirror adjoint to D_s, and the
    # defect is bounded instead. A constant carries no flux anywhere. W_n is
    # h, and h/2 on a wall node.
    noart = ArtificialProperties(enabled=false)
    walls = (SlipWallBC(), SlipWallBC())
    geometries = (
        ("stretched line", 1, true, build_solver(
            Problem(domain=((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)),
                    bcs=(walls, per3[2], per3[3]), ic=nothing),
            Numerics(n_global=(17, 1, 1), art=noart,
                     stretch=(sine_cluster(0.0, 1.0, 0.3, 0.4), nothing, nothing)))),
        ("cylindrical shell", 1, true, build_solver(
            Problem(domain=((0.5, 0.5 + 1.0), (0.0, 0.0 + 1.0), (0.0, 0.0 + 1.0)),
                    metric=CylindricalMetric(), bcs=(walls, per3[2], per3[3]), ic=nothing),
            Numerics(n_global=(17, 1, 1), art=noart))),
        ("cylindrical axis", 1, false, build_solver(
            Problem(domain=((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)), metric=CylindricalMetric(),
                    bcs=((AxisBC(), SlipWallBC()), per3[2], per3[3]), ic=nothing),
            Numerics(n_global=(16, 1, 1), art=noart))),
        ("resolved-θ axis", 1, false, build_solver(
            Problem(domain=((0.0, 1.0), (0.0, 2π), (0.0, 1.0)), metric=CylindricalMetric(),
                    bcs=((AxisBC(), SlipWallBC()), per3[2], per3[3]), ic=nothing),
            Numerics(n_global=(12, 12, 1), art=noart))),
        ("cylindrical θ", 2, true, build_solver(
            Problem(domain=((0.5, 1.5), (0.0, 2π), (0.0, 1.0)),
                    metric=CylindricalMetric(), bcs=(walls, per3[2], per3[3]), ic=nothing),
            Numerics(n_global=(12, 12, 1), art=noart))),
        ("spherical origin", 1, true, build_solver(
            Problem(domain=((0.0, 1.0), (0.0, π), (0.0, 2π)), metric=SphericalMetric(),
                    bcs=((OriginBC(), SlipWallBC()), (PoleBC(), PoleBC()), per3[3]),
                    ic=nothing),
            Numerics(n_global=(12, 12, 12), art=noart))),
        ("spherical poles", 2, false, build_solver(
            Problem(domain=((0.0, 1.0), (0.0, π), (0.0, 2π)), metric=SphericalMetric(),
                    bcs=((OriginBC(), SlipWallBC()), (PoleBC(), PoleBC()), per3[3]),
                    ic=nothing),
            Numerics(n_global=(12, 12, 12),
                                                            art=noart))),
    )
    for (label, d, symmetric, solver) in geometries
        decomp = solver.decomp
        op = CL.StaggeredDiffusion(solver, d)
        pad = decomp.n_halo_d
        inner = CartesianIndices(ntuple(k -> pad[k]+1:pad[k]+decomp.n_local[k], 3))
        n = decomp.n_local[d]
        closed = !decomp.periodic[d]
        wall_lo = closed && (solver.folds[d] === nothing || !solver.folds[d].lo)
        wall_hi = closed && (solver.folds[d] === nothing || !solver.folds[d].hi)
        vol = zeros(size(solver.inv_J))
        for I in inner
            i = I[d] - pad[d]
            w = solver.h[d] * ((i == 1 && wall_lo) || (i == n && wall_hi) ? 0.5 : 1.0)
            vol[I] = w / solver.inv_J[I]
        end
        κ = field(decomp)
        for I in inner
            x = Tuple(I) .* 0.37
            κ[I] = 1 + 0.3sin(x[1] + 2x[2]) * cos(x[3])
        end
        apply(u) = (out = field(decomp); t = copy(u);
                    CL.staggered_diffusion!(out, op, t, copy(κ), decomp); out)
        inprod(u, v) = sum(vol[I] * u[I] * v[I] for I in inner)
        one_f = field(decomp); one_f[inner] .= 1
        L1 = apply(one_f)
        @test maximum(abs, L1[inner]) < 1e-9
        for trial in 1:4
            u = field(decomp); v = field(decomp)
            u[inner] .= randn(size(inner)); v[inner] .= randn(size(inner))
            Lu, Lv = apply(u), apply(v)
            tol = symmetric ? 1e-11 : 1e-2
            @test abs(inprod(u, Lv) - inprod(v, Lu)) < tol * abs(inprod(u, Lu))
            @test inprod(u, Lu) < 0
            @test abs(inprod(one_f, Lu)) < tol * abs(inprod(u, Lu))
        end
    end
end
