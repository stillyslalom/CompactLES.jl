# The serial suite's tests of the boundary closures and the compact filter. Included
# by test/serial_suite.jl, which loads the package and defines the imports and
# helpers used here; not runnable on its own.

@testset "closed-domain closures: polynomial exactness (deg ≤ 3)" begin
    solver = build_solver(
        Problem(domain=((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)),
                bcs=((SlipWallBC(), SlipWallBC()), per3[2], per3[3]), ic=nothing),
        Numerics(n_global=(32, 12, 12), art=ArtificialProperties(enabled=false)))
    f = CL.field(solver.decomp); df = CL.field(solver.decomp)
    fillf!(solver, f, (x, y, z) -> 1 + 2x + 3x^2 - x^3)
    CL.exchange_halos!(f, solver.decomp)
    CL.deriv_along!(df, f, solver, 1, 1); CL._scale_grad!(df, solver, 1)
    @test ferr(solver, df, (x, y, z) -> 2 + 6x - 3x^2) < 1e-10
end

@testset "closure sets: exact through the closure order, not beyond" begin
    # Each closure set is exact for polynomials up to the order of its lowest
    # row and visibly inexact one degree higher, which pins both the
    # coefficients and the row count: a dropped row would expose an interior
    # stencil reading past the edge, and a wrong coefficient breaks exactness
    # at some degree ≤ 3. The Brady–Livescu rows are held to the same
    # tolerance despite their ~1e3 closed-line condition number.
    for (deriv, deg) in ((lele_d1_6(), 3),
                         (lele_d1_6(closures=:cascade3), 3),
                         (lele_d1_6(closures=:cascade4), 4),
                         (lele_d1_6(closures=:brady_livescu), 5),
                         (lele_d1_8(), 3),
                         (lele_d1_8(closures=:cascade3), 3),
                         (lele_d1_8(closures=:cascade4), 4),
                         (lele_d1_8(closures=:brady_livescu), 7),
                         (lele_d1_10(), 3),
                         (lele_d1_10(closures=:cascade3), 3))
        solver = build_solver(
            Problem(domain=((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)),
                    bcs=((SlipWallBC(), SlipWallBC()), per3[2], per3[3]), ic=nothing),
            Numerics(n_global=(32, 16, 16), deriv=deriv,
                     art=ArtificialProperties(enabled=false)))
        f = CL.field(solver.decomp); df = CL.field(solver.decomp)
        fillf!(solver, f, (x, y, z) -> sum(x^m for m in 0:deg))
        CL.exchange_halos!(f, solver.decomp)
        CL.deriv_along!(df, f, solver, 1, 1); CL._scale_grad!(df, solver, 1)
        @test ferr(solver, df, (x, y, z) -> sum(m * x^(m - 1) for m in 1:deg)) < 1e-10
        fillf!(solver, f, (x, y, z) -> x^(deg + 1))
        CL.exchange_halos!(f, solver.decomp)
        CL.deriv_along!(df, f, solver, 1, 1); CL._scale_grad!(df, solver, 1)
        @test ferr(solver, df, (x, y, z) -> (deg + 1) * x^deg) > 1e-7
    end
    @test_throws ErrorException lele_d1_6(closures=:unknown)
    # The C8 neutral set is the C6 neutral rows over the C6 interior row,
    # which is also the third cascade row.
    @test lele_d1_8().closures[1:2] == lele_d1_6().closures
    @test lele_d1_8().closures[3] == lele_d1_8(closures=:cascade3).closures[3]
    @test_throws ErrorException CL.neutral_closures(Float64, 4)
    # The C10 rows are the C8 rows with a zero outer band, whichever set is
    # selected, and the cascade set reproduces the rows the scheme carried
    # before it took a keyword.
    for j in 1:3
        @test lele_d1_10().closures[j].lhs == [0; lele_d1_8().closures[j].lhs...; 0]
        @test lele_d1_10().closures[j].rhs == lele_d1_8().closures[j].rhs
        @test lele_d1_10(closures=:cascade3).closures[j].lhs ==
            [0; lele_d1_8(closures=:cascade3).closures[j].lhs...; 0]
        @test lele_d1_10(closures=:cascade3).closures[j].rhs ==
            lele_d1_8(closures=:cascade3).closures[j].rhs
    end
    let rows = lele_d1_10(closures=:cascade3).closures
        @test rows[1].lhs == [0, 0, 1, 2, 0]
        @test rows[1].rhs == [-5//2, 2, 1//2]
        @test rows[2].lhs == [0, 1//4, 1, 1//4, 0]
        @test rows[2].rhs == [-3//4, 0, 3//4]
        @test rows[3].lhs == Float64[0, 1//3, 1, 1//3, 0]
        @test rows[3].rhs == Float64[-1//36, -7//9, 0, 7//9, 1//36]
        @test all(row -> row.first == 1, rows)
    end
    @test_throws ErrorException lele_d1_10(closures=:cascade4)
    @test_throws ErrorException lele_d1_10(closures=:brady_livescu)
    # An interface imposes no wall condition, so a neutral scheme falls back
    # to the cascade rows of its own width there, and a cascade scheme keeps
    # what it has.
    @test CL.interface_divergence_closures(lele_d1_8()) ==
        lele_d1_8(closures=:cascade3).closures
    @test length(CL.interface_divergence_closures(lele_d1_8())) == 3
    @test CL.interface_divergence_closures(lele_d1_8(closures=:cascade3)) ==
        lele_d1_8(closures=:cascade3).closures
    @test CL.interface_divergence_closures(lele_d1_6()) ==
        lele_d1_6(closures=:cascade3).closures
    @test CL.interface_divergence_closures(lele_d1_10()) ==
        lele_d1_10(closures=:cascade3).closures
    @test length(CL.interface_divergence_closures(lele_d1_10())) == 3
    @test CL.interface_divergence_closures(lele_d1_10(closures=:cascade3)) ==
        lele_d1_10(closures=:cascade3).closures
    @test CL.interface_divergence_closures(compact_d8()) == compact_d8().closures
    @test CL.interface_divergence_closures(pade_d1_4()) == pade_d1_4().closures
end

@testset "filter closures: one-sided rows, published row, wall exactness" begin
    af = 0.45
    # The derivation reproduces the interior stencil at the centered point
    # and Gaitonde–Visbal's tabulated row 2 (their (1 + 254αf)/256 and
    # (31 + 2αf)/32 leading entries), and follows the element type.
    base = compact_filter(af; closures=:cascade)
    r5 = CL.onesided_filter_row(af, 5, 4)
    @test maximum(abs.(r5 .- [reverse(base.coeffs); base.a0; base.coeffs])) < 1e-14
    os = compact_filter(af; closures=:onesided)
    @test abs(os.closures[2].rhs[1] - (1 + 254af) / 256) < 1e-14
    @test abs(os.closures[2].rhs[2] - (31 + 2af) / 32) < 1e-14
    @test os.closures[1].rhs == [1.0]
    os32 = compact_filter(0.45f0, Float32; closures=:onesided)
    @test eltype(os32.closures[2].rhs) === Float32
    @test_throws ErrorException compact_filter(af; closures=:unknown)
    # One closed-domain pass: the one-sided rows leave a degree-7 polynomial
    # unchanged; the cascade's F2 row alters anything above degree 1.
    for (filt, deg, exact) in ((os, 7, true), (base, 3, false))
        solver = build_solver(
            Problem(domain=((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)),
                    bcs=((SlipWallBC(), SlipWallBC()), per3[2], per3[3]), ic=nothing),
            Numerics(n_global=(32, 12, 12), art=ArtificialProperties(enabled=false),
                     filter=filt))
        f = CL.field(solver.decomp); g = CL.field(solver.decomp)
        fillf!(solver, f, (x, y, z) -> sum(x^m for m in 0:deg))
        CL.exchange_halos!(f, solver.decomp)
        CL.filt_along!(g, f, solver, 1, 1)
        e = ferr(solver, g, (x, y, z) -> sum(x^m for m in 0:deg))
        @test exact ? e < 1e-12 : e > 1e-6
    end
end

@testset "filter: constants exact, Nyquist damped, parity of closures" begin
    solver = mkslv(n_global=(32, 12, 12))
    f = CL.field(solver.decomp)
    fillf!(solver, f, (x, y, z) -> 1.0)
    filter_field!(f, solver)
    @test ferr(solver, f, (x, y, z) -> 1.0) < 1e-12
    nx = solver.decomp.n_local[1]
    for k in 1:solver.decomp.n_local[3], j in 1:solver.decomp.n_local[2], i in 1:nx
        f[padded_index(solver, i, j, k)] = 1.0 + 0.5 * (-1)^i        # constant + Nyquist
    end
    filter_field!(f, solver)
    dev = maximum(abs(f[padded_index(solver, i, 1, 1)] - 1.0) for i in 1:nx)
    @test dev < 0.35                                      # sawtooth strongly damped
end

@testset "filter relaxation: weight, exact blend, full strength at the reference" begin
    # The relaxed filter must be an exact linear interpolation between the state
    # and its filtered image, and must reduce to the unrelaxed pass at w = 1 —
    # bit-identically, since every guarded number in the suite was measured on
    # that path.
    ic = (x, y, z) -> Prim(rho=1.0 + 0.2sin(x) * cos(2y), p=1.0 + 0.1sin(3z),
                           u=(0.3sin(2x), 0.2cos(y), 0.1sin(z) * cos(x)))
    build(fc) = setup(Problem(eos=IdealSpecies("gas"; gamma=1.4, R=1.0),
                              transport=ConstantTransport(mu0=0.0),
                              domain=((0.0, 2π), (0.0, 2π), (0.0, 2π)),
                              bcs=per3, ic=ic),
                      Numerics(n_global=(16, 16, 16), art=ArtificialProperties(enabled=false),
                               cfl=0.4, filter=StateFilter(; cfl=fc)))

    solver0, Q0 = build(0.0)
    @test CL.filter_weight(solver0, 1) == 1.0       # disabled: full strength

    # dt_prev = 0 before any step, where the unrelaxed meaning is the only one
    # available, so the weight must be 1 even with the relaxation configured.
    solverW, _ = build(0.4)
    @test CL.filter_weight(solverW, 1) == 1.0
    # The weight of a pass along d is dt · r_d · √n / filter_cfl on n active
    # dimensions, three here: r_d = 1/√3 makes dt · r_d · √3 the plain dt.
    r3 = 1 / sqrt(3.0)
    solverW.dt_prev = 0.2; solverW.filter_rate_prev = (r3, r3, r3)  # 0.2 / 0.4
    @test CL.filter_weight(solverW, 1) ≈ 0.5
    @test CL.filter_weight(solverW, 3) ≈ 0.5
    # Each direction reads its own rate: a slower direction filters less.
    solverW.filter_rate_prev = (r3, r3 / 2, r3 / 4)
    @test CL.filter_weight(solverW, 2) ≈ 0.25
    @test CL.filter_weight(solverW, 3) ≈ 0.125
    solverW.dt_prev = 2.0                            # dt · r_1 · √3 = 2, well over
    @test CL.filter_weight(solverW, 1) == 1.0        # capped, never over-filters

    # Interior only. `filter_state!` exchanges halos, so comparing whole padded
    # arrays measures halo initialization rather than the filter — a difference
    # of 2.83 against a filter effect three orders smaller.
    function interior(solver, Q)
        nx, ny, nz = solver.decomp.n_local
        [Q[padded_index(solver, i, j, k), c]
         for i in 1:nx, j in 1:ny, k in 1:nz, c in 1:solver.equations.n_cons]
    end
    dev(A, B) = maximum(abs, A .- B)

    ref = interior(solver0, Q0)                      # unfiltered state
    filter_state!(solver0, Q0)
    full = dev(interior(solver0, Q0), ref)           # fully filtered image
    @test full > 0

    solverB, QB = build(0.8)                         # w = dt · r_d · √3 / 0.8
    solverB.dt_prev = 0.4; solverB.filter_rate_prev = (r3, r3, r3)   # w = 0.5
    @test CL.filter_weight(solverB, 1) ≈ 0.5
    filter_state!(solverB, QB)
    # Filtering is applied dimension by dimension, so the composite is not
    # (1-w)Q + w·F(Q) exactly; it is that per direction. Check the property that
    # does hold globally and motivates the relaxation: the blended state
    # lies strictly between the two, and approaches each end as w does.
    @test 0 < dev(interior(solverB, QB), ref) < full

    solverS, QS = build(1000.0)                      # w = 4e-4, nearly no filter
    solverS.dt_prev = 0.4; solverS.filter_rate_prev = (r3, r3, r3)
    filter_state!(solverS, QS)
    @test dev(interior(solverS, QS), ref) < 0.01 * full

    # w == 1 must take the copy path, not the blend, so a run at or above the
    # reference CFL is the unrelaxed solver exactly.
    solverE, QE = build(0.4)
    solverE.dt_prev = 0.4                            # dt · r_d · √3 = 0.69 ≥ 0.4
    solverE.filter_rate_prev = (1.0, 1.0, 1.0)
    @test all(d -> CL.filter_weight(solverE, d) == 1.0, 1:3)
    filter_state!(solverE, QE)
    @test QE == Q0                                   # bit-identical, not approx
end

@testset "volume-weighted filter: identity, constants, and the measured defects" begin
    # `filter_weighting = :volume` is the form of the public Pyranda
    # implementation, F(J q) / F(J). bench/archive/filter_conservation.jl measures it
    # against the unweighted operator on every geometry; the facts pinned here
    # are the ones the default rests on (reference/CALIBRATION_APPENDIX.md, the
    # filter on non-uniform volumes). One directional pass on a line is the
    # matrix M assembled from unit impulses, and Mᵀ V − V is its conservation
    # defect on the quadrature volumes V: the mass a pass creates on q is that
    # row vector applied to q.
    # 64 nodes, as the bench measures: on a shorter line the wall rows' leak
    # reaches the fold rows and the origin figures below are not round-off.
    # The figures were measured under the filter's cascade rows, so the lines
    # pin them; the one-sided rows, which are the default, move
    # the wall leak inward and are checked separately below.
    N = 64
    walls = ((SlipWallBC(), SlipWallBC()), per3[2], per3[3])
    line(wt; filt=compact_filter(0.45; closures=:cascade), bcs, metric=CartesianMetric(),
         stretch=(nothing, nothing, nothing),
         domain=((0.0, 1.0), (0.0, 1.0), (0.0, 1.0))) =
        build_solver(Problem(; domain, bcs, metric, ic=nothing),
                     Numerics(n_global=(N, 1, 1), art=ArtificialProperties(enabled=false),
                              stretch=stretch, filter=StateFilter(filt; weighting=wt)))
    function line_operator(s)
        Q = allocate_state(s)
        M = zeros(N, N)
        for n in 1:N
            fill!(parent(Q), 0.0)
            Q[padded_index(s, n, 1, 1), 1] = 1.0
            filter_state!(s, Q)
            for k in 1:N
                M[k, n] = Q[padded_index(s, k, 1, 1), 1]
            end
        end
        V = [CL.quad_weight(s, 1, n) / s.inv_J[padded_index(s, n, 1, 1)] for n in 1:N]
        return M, V
    end
    defect(s) = ((M, V) = line_operator(s); (M' * V .- V) ./ V)

    # A periodic line has unit column sums, so it conserves; a wall line does
    # not: the cascade closure rows carry a defect of a few percent of a
    # node's content that decays into the interior at the tridiagonal root,
    # 0.627 per row at α = 0.45, and is still above 1e-4 twelve rows in. On a
    # uniform volume the weighting is skipped and the operator is the same
    # matrix bit for bit.
    Mp, _ = line_operator(line(:none; bcs=per3))
    @test maximum(abs, sum(Mp; dims=1) .- 1) < 1e-13
    Mw, _ = line_operator(line(:none; bcs=walls))
    Mv, _ = line_operator(line(:volume; bcs=walls))
    @test Mw == Mv
    cs = vec(sum(Mw; dims=1)) .- 1
    @test 0.01 < abs(cs[2]) < 0.08                  # measured -4.0e-2
    @test 1e-4 < abs(cs[12]) < 1e-2                 # measured -3.8e-4
    # The one-sided rows leave row 2 nearly conservative and move the peak
    # of the leak to rows 3–7 (2.9e-2 at row 5) without reducing its total.
    Mo, _ = line_operator(line(:none; bcs=walls, filt=compact_filter(0.45)))
    co = vec(sum(Mo; dims=1)) .- 1
    @test abs(co[2]) < 0.01                         # measured 2.6e-3
    @test 0.01 < maximum(abs, co[3:7]) < 0.08
    @test 0.5 < sum(abs, co) / sum(abs, cs) < 2

    # On a clustered wall line the closure leak dominates either way; the
    # weighting moves the interior defect by less than a tenth of it.
    st = (sine_cluster(0.0, 1.0, 0.5, 0.5), nothing, nothing)
    dn = defect(line(:none; bcs=walls, stretch=st))
    dv = defect(line(:volume; bcs=walls, stretch=st))
    @test maximum(abs, dn[9:N-8]) < 5e-3            # measured 1.7e-3
    @test maximum(abs, dv[9:N-8]) < 5e-3            # measured 1.6e-3
    @test abs(maximum(abs, dn[9:N-8]) - maximum(abs, dv[9:N-8])) < 5e-4

    # At the cylindrical axis the odd-parity fold the weighted form needs
    # (J = r) does not have unit column sums: its first-row defect is 0.149
    # against 8.6e-3 unweighted. At the spherical origin (J = r² even) both
    # conserve to round-off.
    axis = ((AxisBC(), SlipWallBC()), per3[2], per3[3])
    dn = defect(line(:none; bcs=axis, metric=CylindricalMetric()))
    dv = defect(line(:volume; bcs=axis, metric=CylindricalMetric()))
    @test abs(dn[1]) < 2e-2
    @test abs(dv[1]) > 0.1
    @test abs(dv[1]) > 10 * abs(dn[1])
    orig = ((OriginBC(), SlipWallBC()), per3[2], per3[3])
    sph = (; metric=SphericalMetric(), domain=((0.0, 1.0), (π / 2, π / 2 + 1.0), (0.0, 1.0)))
    dn = defect(line(:none; bcs=orig, sph...))
    dv = defect(line(:volume; bcs=orig, sph...))
    @test maximum(abs, dn[1:4]) < 1e-9                # measured 6.5e-11
    @test maximum(abs, dv[1:4]) < 1e-9                # measured 2e-14

    # Constants: the unweighted operator never reads the volume, and the
    # weighted one divides F(J) by itself, so both hold a uniform state on
    # every metric, folds and stretching included.
    function interior_change(s, Q, Q0)
        nx, ny, nz = s.decomp.n_local
        maximum(abs(Q[padded_index(s, i, j, k), c] - Q0[padded_index(s, i, j, k), c])
                for c in 1:s.equations.n_cons, i in 1:nx, j in 1:ny, k in 1:nz)
    end
    none = (nothing, nothing, nothing)
    metrics = [
        (; n_global=(24, 12, 12), domain=((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)),
           metric=CartesianMetric(), bcs=walls,
           stretch=(sine_cluster(0.0, 1.0, 0.5, 0.4), nothing, nothing)),
        (; n_global=(32, 1, 12), domain=((0.0, 1.0), (0.0, 1.0), (0.0, 0.5)),
           metric=CylindricalMetric(), bcs=axis, stretch=none),
        (; n_global=(24, 16, 1), domain=((0.0, 1.0), (0.0, 2π), (0.0, 1.0)),
           metric=CylindricalMetric(), bcs=axis, stretch=none),
        (; n_global=(24, 12, 12), domain=((0.0, 1.0), (0.0, π), (0.0, 2π)),
           metric=SphericalMetric(),
           bcs=((OriginBC(), SlipWallBC()), (PoleBC(), PoleBC()), per3[3]), stretch=none),
    ]
    for cs in metrics, wt in (:none, :volume)
        s = build_solver(Problem(; domain=cs.domain, metric=cs.metric, bcs=cs.bcs,
                                 ic=nothing),
                         Numerics(; n_global=cs.n_global, stretch=cs.stretch,
                                  art=ArtificialProperties(enabled=false),
                                  filter=StateFilter(weighting=wt)))
        Q = allocate_state(s)
        initialize!(s, Q, (x, y, z) -> Prim(u=(0, 0, 0), p=1.0, rho=1.0))
        apply_bcs!(s, Q)
        Q0 = copy(Q)
        filter_state!(s, Q)
        @test interior_change(s, Q, Q0) < 1e-12
    end

    # Uniform Cartesian in three dimensions: the weighting is skipped, so a
    # pass on a non-uniform state is the unweighted pass bit for bit.
    ic = (x, y, z) -> Prim(u=(0.3sin(x)cos(y), -0.3cos(x)sin(y), 0.1sin(z)),
                            p=1 + 0.1cos(x)cos(z), rho=1 + 0.2sin(y))
    states = map((:none, :volume)) do wt
        s = build_solver(
            Problem(domain=((0.0, 2π), (0.0, 2π), (0.0, 2π)), bcs=per3, ic=nothing),
            Numerics(n_global=(16, 16, 16), art=ArtificialProperties(enabled=false),
                     filter=StateFilter(weighting=wt)))
        Q = allocate_state(s)
        initialize!(s, Q, ic)
        filter_state!(s, Q)
        Q
    end
    @test states[1] == states[2]
    @test_throws ErrorException build_solver(
        Problem(domain=((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)), bcs=per3, ic=nothing),
        Numerics(n_global=(16, 1, 1), filter=StateFilter(weighting=:mass)))
end
