# The serial suite's tests of the coordinate folds, mode truncation, the symmetry
# plane and curvilinear metrics. Included by test/serial_suite.jl, which loads the
# package and defines the imports and helpers used here; not runnable on its own.

@testset "axisymmetric axis fold: manufactured smooth solution" begin
    # u_r = r·g(r) is an odd smooth function; d/dr through the fold must
    # match analytics at the first half-offset nodes — the sharpest probe of
    # the folded row and mirror fill.
    solver = build_solver(
        Problem(domain=((0.0, 1.0), (0.0, 1.0), (0.0, 0.5)), metric=CylindricalMetric(),
                bcs=((AxisBC(), SlipWallBC()), per3[2], per3[3]), ic=nothing),
        Numerics(n_global=(64, 1, 12), art=ArtificialProperties(enabled=false)))
    f = CL.field(solver.decomp); df = CL.field(solver.decomp)
    fillf!(solver, f, (r, θ, z) -> r * exp(-4r^2))            # odd across the axis
    CL.exchange_halos!(f, solver.decomp)
    CL.deriv_along!(df, f, solver, 1, -1); CL._scale_grad!(df, solver, 1)
    @test ferr(solver, df, (r, θ, z) -> (1 - 8r^2) * exp(-4r^2)) < 5e-6
    fillf!(solver, f, (r, θ, z) -> exp(-4r^2))                # even across the axis
    CL.exchange_halos!(f, solver.decomp)
    CL.deriv_along!(df, f, solver, 1, 1); CL._scale_grad!(df, solver, 1)
    # Even fold: third order with a larger constant. The maximum sits at the
    # outer slip wall: 2.14e-5 under the `:neutral3` rows, 1.4e-5 under
    # `:cascade3`.
    @test ferr(solver, df, (r, θ, z) -> -8r * exp(-4r^2)) < 3e-5
end

@testset "resolved-θ axis: antipodal pairing (local)" begin
    # f = r cosθ · e^{−4r²} = x·g is a scalar that is globally smooth through
    # the axis; its antipodal image is −f, so the odd combination carries all
    # of it. It is not a velocity component (a u_r of that form has no smooth
    # Cartesian preimage), so σ = +1.
    solver = build_solver(
        Problem(domain=((0.0, 1.0), (0.0, 2π), (0.0, 1.0)), metric=CylindricalMetric(),
                bcs=((AxisBC(), SlipWallBC()), per3[2], per3[3]), ic=nothing),
        Numerics(n_global=(48, 16, 1), art=ArtificialProperties(enabled=false)))
    f = CL.field(solver.decomp); df = CL.field(solver.decomp)
    fillf!(solver, f, (r, θ, z) -> r * cos(θ) * exp(-4r^2))
    CL.exchange_halos!(f, solver.decomp)
    CL.deriv_along!(df, f, solver, 1, 1); CL._scale_grad!(df, solver, 1)
    # The maximum sits at the outer slip wall: 1.06e-5 under the `:neutral3`
    # rows, below 1e-5 under `:cascade3`.
    @test ferr(solver, df, (r, θ, z) -> cos(θ) * (1 - 8r^2) * exp(-4r^2)) < 1.5e-5
    # A scalar even case: f = e^{−4r²}·(1 + ½cos 2θ) maps to itself at θ+π.
    fillf!(solver, f, (r, θ, z) -> exp(-4r^2) * (1 + 0.5cos(2θ)))
    CL.exchange_halos!(f, solver.decomp)
    CL.deriv_along!(df, f, solver, 1, 1); CL._scale_grad!(df, solver, 1)
    @test ferr(solver, df, (r, θ, z) -> -8r * exp(-4r^2) * (1 + 0.5cos(2θ))) < 1e-4  # 3rd-order, larger const
end

@testset "azimuthal mode truncation: table, projection, rate cap" begin
    # N_θ = 32 over 24 radial nodes at κ = 1: mode_limit(r_i) =
    # max(2, ⌊π(i − ½)⌋) is 2, 4, 7, 10, 14 on the first five rings, and
    # ring 6 (17 ≥ N_θ/2) is the first to keep every mode. The radial and
    # azimuthal momenta keep one mode more, and the θ rate is capped at it.
    mk(κ; n_global=(24, 32, 1), L_domain=(1.0, 2π, 1.0)) = build_solver(
        Problem(domain=((0.0, L_domain[1]), (0.0, L_domain[2]), (0.0, L_domain[3])),
                metric=CylindricalMetric(),
                bcs=((AxisBC(), SlipWallBC()), per3[2], per3[3]), ic=nothing),
        Numerics(n_global=n_global, art=ArtificialProperties(enabled=false),
                 polar_truncation=κ))
    solver = mk(1.0)
    tr = solver.truncation
    o1 = solver.decomp.n_halo_d[1]
    @test tr.mode_limit == [2, 4, 7, 10, 14]
    @test tr.rings == (o1 + 1):(o1 + 5)
    @test tr.vector == solver.equations.i_mom[1]:solver.equations.i_mom[2]
    @test tr.theta_cap ≈ [(m + 1) / (π * xcoord(solver, 1, i))
                          for (i, m) in enumerate(tr.mode_limit)] rtol = 1e-14
    @test isempty(mk(0.0).truncation.rings)
    @test mk(2.0).truncation.mode_limit == [2, 2, 3, 5, 7, 8, 10, 11, 13, 14]
    @test_throws ArgumentError mk(0.5)
    @test_throws ErrorException mk(1.0; n_global=(24, 1, 1), L_domain=(1.0, 1.0, 1.0))
    @test_throws ErrorException build_solver(
        Problem(domain=((0.0, 1.0), (0.0, π), (0.0, 1.0)), bcs=per3, ic=nothing),
        Numerics(n_global=(24, 32, 1), polar_truncation=1.0))
    @test_throws ErrorException build_solver(
        Problem(domain=((0.0, 1.0), (0.0, π), (0.0, 2π)), metric=SphericalMetric(),
                bcs=((OriginBC(), SlipWallBC()), (PoleBC(), PoleBC()), per3[3]), ic=nothing),
                                             Numerics(n_global=(24, 32, 1),
                                                      polar_truncation=1.0))

    nθ = 32
    ring(Q, i, c) = [Q[padded_index(solver, i, j, 1), c] for j in 1:nθ]
    # Amplitude of mode m on ring i of component c.
    amp(Q, i, c, m) = (v = ring(Q, i, c); θ = [xcoord(solver, 2, j) for j in 1:nθ];
                       hypot(sum(v .* cos.(m .* θ)), sum(v .* sin.(m .* θ))) * 2 / nθ)
    maxdiff(A, B) = maximum(abs, parent(A) .- parent(B))
    im = solver.equations.i_mom
    U = 0.3
    stream(r, θ, z) = Prim(u=(U * cos(θ), -U * sin(θ), 0.0), p=1.0, rho=1.0)

    # A uniform freestream is m = 1 in the physical velocity components.
    Q = allocate_state(solver)
    initialize!(solver, Q, stream)
    Q0 = deepcopy(Q)
    CL.truncate_modes!(solver, Q)
    @test maxdiff(Q, Q0) < 1e-14

    # A quadratic field is Cartesian mode 2 at O(r²): mode 2 of a density
    # and, through the rotation to u_r and u_θ, mode 3 of the physical
    # momenta. Ring 1 (limit 2) keeps both. The density varies in one field
    # and the velocity in the other, so that neither momentum carries their
    # product; the energy, quartic in r through the kinetic energy, is not
    # compared.
    r1 = xcoord(solver, 1, 1)
    for (a, b) in ((0.3, 0.0), (0.0, 1.0))
        function quadratic(r, θ, z)
            s, c = sincos(θ)
            x, y = r * c, r * s
            ux = U + b * (0.2x + 0.1y + 0.5(x^2 - y^2))
            uy = b * (0.3x - 0.2y + 0.4x * y)
            return Prim(u=(c * ux + s * uy, -s * ux + c * uy, 0.0), p=1.0,
                        rho=1.0 + a * x * y)
        end
        initialize!(solver, Q, quadratic)
        Q0 = deepcopy(Q)
        @test amp(Q0, 1, a > 0 ? 1 : im[1], a > 0 ? 2 : 3) > 0.1r1^2
        CL.truncate_modes!(solver, Q)
        @test all(c -> maximum(abs, ring(Q, 1, c) .- ring(Q0, 1, c)) < 1e-14,
                  (1, im...))
    end

    # Rigid rotation over a radially varying state is m = 0 on every ring.
    initialize!(solver, Q, (r, θ, z) -> Prim(u=(0.0, 0.4r, 0.0), p=1 + r^2,
                                             rho=1 + 0.5r^2))
    Q0 = deepcopy(Q)
    CL.truncate_modes!(solver, Q)
    @test maxdiff(Q, Q0) < 1e-14

    # m = 6 seeded on ring 2 (limit 4) is removed and m = 3 there is kept; the
    # same m = 6 on ring 10, outside the table, is untouched bit for bit.
    seeded(r, θ, z) = Prim(u=(U * cos(θ), -U * sin(θ), 0.0), p=1.0,
                           rho=1 + (abs(r - xcoord(solver, 1, 2)) < 1e-12 ?
                                    0.1cos(6θ) + 0.05sin(3θ) : 0.0) +
                               (abs(r - xcoord(solver, 1, 10)) < 1e-12 ?
                                    0.1cos(6θ) : 0.0))
    initialize!(solver, Q, seeded)
    Q0 = deepcopy(Q)
    CL.truncate_modes!(solver, Q)
    @test amp(Q0, 2, 1, 6) ≈ 0.1
    @test amp(Q, 2, 1, 6) < 1e-14
    @test amp(Q, 2, 1, 3) ≈ 0.05 rtol = 1e-13
    @test ring(Q, 10, 1) == ring(Q0, 10, 1)
    @test ring(Q, 10, 5) == ring(Q0, 10, 5)
    # Idempotent: a second projection changes nothing beyond round-off.
    Q1 = deepcopy(Q)
    CL.truncate_modes!(solver, Q)
    @test maxdiff(Q, Q1) < 1e-14

    # Ring sums of mass, energy and Cartesian momentum, over a state with
    # content in every mode on every ring.
    rng = MersenneTwister(11)
    for k in axes(parent(Q), 4), I in CartesianIndices(solver.rho)
        parent(Q)[I, k] = (k == 1 ? 1.0 : k == 5 ? 2.5 : 0.0) + 0.1 * randn(rng)
    end
    Q0 = deepcopy(Q)
    CL.truncate_modes!(solver, Q)
    @test maxdiff(Q, Q0) > 1e-2
    for i in 1:6
        θ = [xcoord(solver, 2, j) for j in 1:nθ]
        for S in (Q -> sum(ring(Q, i, 1)), Q -> sum(ring(Q, i, 5)),
                  Q -> sum(ring(Q, i, im[1]) .* cos.(θ) .- ring(Q, i, im[2]) .* sin.(θ)),
                  Q -> sum(ring(Q, i, im[1]) .* sin.(θ) .+ ring(Q, i, im[2]) .* cos.(θ)))
            @test S(Q) ≈ S(Q0) atol = 1e-13
        end
    end

    # The rate cap: the same state steps longer under the table, dt_report
    # agrees with compute_dt, and at κ = 0 the loop is the untruncated one.
    off = mk(0.0)
    Qoff = allocate_state(off)
    initialize!(off, Qoff, stream)
    initialize!(solver, Q, stream)
    @test compute_dt(solver, Q) > 4 * compute_dt(off, Qoff)
    @test dt_report(solver, Q).dt ≈ compute_dt(solver, Q) rtol = 1e-12
    @test dt_report(off, Qoff).dt ≈ compute_dt(off, Qoff) rtol = 1e-12

    # run! applies the projection after every step: on ring 1 (limit 2) a
    # seeded pressure in modes 3 and 4 leaves no mode above 2 in the scalars
    # and none above 3 in the radial and azimuthal momenta after one step,
    # while the momenta's mode 3, which the pressure gradient drives, stays.
    initialize!(solver, Q, (r, θ, z) -> Prim(u=(U * cos(θ), -U * sin(θ), 0.0),
                                             p=1.0 + 0.01(cos(3θ) + sin(4θ)) *
                                                     exp(-20r^2),
                                             rho=1.0))
    run!(solver, Q; tfinal=1.0, nmax=1)
    scalars = setdiff(1:solver.equations.n_cons, tr.vector)
    @test all(c -> amp(Q, 1, c, 3) < 1e-14 && amp(Q, 1, c, 4) < 1e-14, scalars)
    @test all(c -> amp(Q, 1, c, 4) < 1e-14, tr.vector)
    @test all(c -> amp(Q, 1, c, 3) > 1e-6, tr.vector)
end

@testset "spherical poles + origin: derivative of a smooth 3-D Gaussian" begin
    # f = e^{−4r²} is smooth at origin and poles; ∂f/∂r and (1/r)∂f/∂θ = 0.
    solver = build_solver(Problem(domain=((0.0, 1.0), (0.0, π), (0.0, 2π)),
                                  metric=SphericalMetric(),
                                  bcs=((OriginBC(), SlipWallBC()),
                                            (PoleBC(), PoleBC()), per3[3]),
                                  ic=nothing),
                          Numerics(n_global=(40, 16, 12),
                                   art=ArtificialProperties(enabled=false)))
    f = CL.field(solver.decomp); df = CL.field(solver.decomp)
    fillf!(solver, f, (r, θ, φ) -> exp(-4r^2))
    CL.exchange_halos!(f, solver.decomp)
    CL.deriv_along!(df, f, solver, 1, 1); CL._scale_grad!(df, solver, 1)
    @test ferr(solver, df, (r, θ, φ) -> -8r * exp(-4r^2)) < 1e-4  # 3rd-order, larger const
    CL.deriv_along!(df, f, solver, 2, 1); CL._scale_grad!(df, solver, 2)
    @test ferr(solver, df, (r, θ, φ) -> 0.0) < 1e-8
end

@testset "symmetry plane: construction, spacing and the self-paired fold" begin
    # The plane sits half a cell outside the end it is applied to, so each
    # folded end takes half a cell off the line: one fold gives L/(N − ½) and
    # two give L/N, with node 1 at h/2 wherever the low end is folded.
    sym = (SymmetryPlaneBC(), SymmetryPlaneBC())
    wall = SlipWallBC()
    cart(bcs1) = build_solver(
        Problem(domain=((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)), bcs=(bcs1, per3[2], per3[3]),
                ic=nothing),
        Numerics(n_global=(16, 12, 12), art=ArtificialProperties(enabled=false)))
    both = cart(sym)
    @test both.h[1] ≈ 1 / 16
    @test both.coord_shift[1] ≈ both.h[1] / 2
    @test xcoord(both, 1, 1) ≈ 1 / 32
    @test xcoord(both, 1, 16) ≈ 31 / 32
    lo = cart((SymmetryPlaneBC(), wall))
    @test lo.h[1] ≈ 1 / 15.5
    @test xcoord(lo, 1, 1) ≈ lo.h[1] / 2
    @test xcoord(lo, 1, 16) ≈ 1.0
    hi = cart((wall, SymmetryPlaneBC()))
    @test hi.h[1] ≈ 1 / 15.5
    @test hi.coord_shift[1] == 0.0
    @test xcoord(hi, 1, 1) ≈ 0.0
    @test xcoord(hi, 1, 16) ≈ 1 - hi.h[1] / 2
    @test (hi.folds[1].lo, hi.folds[1].hi) == (false, true)
    # Self-paired: each line continues into itself, so there is no pairing
    # dimension and no partner block to buffer.
    @test both.folds[1].pair === nothing
    @test both.folds[1].sigvel == (-1, 1, 1)
    @test isempty(both.pairbuf) && isempty(both.pairout)
    # Only the normal velocity is odd, so the mass, energy and tangential
    # momentum fluxes are odd and the normal momentum flux is even.
    eq = both.equations
    @test both.folds[1].sigflux[1] == -1
    @test both.folds[1].sigflux[eq.i_energy] == -1
    @test both.folds[1].sigflux[eq.i_mom[1]] == 1
    @test both.folds[1].sigflux[eq.i_mom[2]] == -1
    # The sensor operators' node-centred wall rows must not be planned here:
    # the fold takes its own half-offset mirror inside the same routines.
    @test sensor_mirror(SymmetryPlaneBC()) === false
    @test CL._sensor_wall_faces(((SymmetryPlaneBC(), SlipWallBC()),
                                 (PeriodicBC(), PeriodicBC()),
                                 (PeriodicBC(), PeriodicBC()))) ==
          ((false, true), (false, false), (false, false))
    @test isperiodic(SymmetryPlaneBC()) === false
    @test CL._is_fold_bc(SymmetryPlaneBC())
    # A plane on dimension 3 beside a cylindrical axis on dimension 1: the
    # pairing is per dimension, and the axis' resolved-θ fold still buffers.
    cyl = build_solver(
        Problem(domain=((0.0, 1.0), (0.0, 2π), (0.0, 0.5)), metric=CylindricalMetric(),
                bcs=((AxisBC(), SlipWallBC()), per3[2], sym), ic=nothing),
        Numerics(n_global=(16, 16, 12), art=ArtificialProperties(enabled=false)))
    @test cyl.h[3] ≈ 0.5 / 12
    @test cyl.coord_shift[3] ≈ cyl.h[3] / 2
    @test cyl.folds[3].pair === nothing
    @test cyl.folds[3].sigvel == (1, 1, -1)
    @test cyl.folds[1] !== nothing && cyl.folds[1].pair !== nothing
    @test !isempty(cyl.pairbuf)
end

@testset "symmetry plane: a uniform state is preserved" begin
    # No node sits on a plane and nothing is enforced there, so a uniform
    # state has to come out of the folded operators unchanged on its own: the
    # mirror halo is the interior value and every flux divergence cancels.
    sym = (SymmetryPlaneBC(), SymmetryPlaneBC())
    eos = IdealMixture([IdealSpecies{Float64}("light", 1.0, 1.4),
                        IdealSpecies{Float64}("heavy", 0.5, 1.3)])
    solver = build_solver(
        Problem(domain=((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)), eos=eos, bcs=(sym, sym, sym),
                ic=nothing),
        Numerics(n_global=(12, 12, 12), art=ArtificialProperties(enabled=true)))
    Q = allocate_state(solver)
    initialize!(solver, Q, (x, y, z) ->
        Prim(Y=(0.3, 0.7), rho=1.2, p=0.8, u=(0.0, 0.0, 0.0)))
    dQ = zero(Q)
    apply_bcs!(solver, Q)
    compute_rhs!(solver, Q, dQ)
    @test maximum(abs, dQ) < 1e-14
    Q0 = copy(Q)
    run!(solver, Q; tfinal=1e9, nmax=5)
    spread, drift = 0.0, 0.0
    for c in 1:solver.equations.n_cons
        v0 = Q0[padded_index(solver, 1, 1, 1), c]
        for k in 1:12, j in 1:12, i in 1:12
            v = Q[padded_index(solver, i, j, k), c]
            spread = max(spread, abs(v - Q[padded_index(solver, 1, 1, 1), c]))
            drift = max(drift, abs(v - v0))
        end
    end
    @test spread < 1e-12
    @test drift < 1e-12
end

@testset "symmetry plane: the mirror of the periodic run on the doubled line" begin
    # Data carrying the slip-wall parity about both ends (rho, p and the mass
    # fractions even, the normal velocity odd, the tangential velocity even)
    # make the run between symmetry planes on [0, 1] at N nodes the
    # restriction of the periodic run on [0, 2) at 2N nodes, node for node.
    # The two agree to round-off because the folded operator is the periodic
    # operator restricted by parity: no closure row enters either, so there
    # is no closure defect to separate them. The node-centred wall is the
    # contrast — its defect is what test/convergence.jl measures.
    sym = (SymmetryPlaneBC(), SymmetryPlaneBC())
    two = IdealMixture([IdealSpecies{Float64}("light", 1.0, 1.4),
                        IdealSpecies{Float64}("heavy", 0.5, 1.3)])
    a, b, N, nsteps = 0.05, 0.05, 32, 30
    function mirror_error(; deriv=lele_d1_6(), art=ArtificialProperties(enabled=true),
                          mu=0.0, dim=1, eos=nothing, n=N)
        h = 1 / n
        nt = 12                                    # transverse extent, 2-D form
        tang = dim == 2
        ic = (x, y, z) -> begin
            s = dim == 1 ? x : y
            rho = 1 + a * cos(pi * s)
            un = b * sin(pi * s)
            ut = tang ? 0.07 * cos(pi * s) * sin(2pi * x) : 0.0
            u = dim == 1 ? (un, ut, 0.0) : (ut, un, 0.0)
            eos === nothing ? Prim(rho=rho, u=u, p=rho^1.4) :
                Prim(Y=(0.5 + 0.2cos(pi * s), 0.5 - 0.2cos(pi * s)),
                     rho=rho, u=u, p=rho^1.4)
        end
        gas = eos === nothing ? IdealSpecies("gas"; R=1.0, gamma=1.4) : eos
        function advance(ng, L, bcs, origin)
            domain = ntuple(d -> (origin[d], origin[d] + L[d]), 3)
            s = build_solver(Problem(; domain, bcs, eos=gas,
                                     transport=ConstantTransport(mu0=mu, Pr=0.7),
                                     ic=nothing),
                             Numerics(n_global=ng, deriv=deriv, art=art, cfl=0.4,
                                      filter=StateFilter(interval=1, cfl=0.0)))
            Q = allocate_state(s)
            initialize!(s, Q, ic)
            run!(s, Q; tfinal=1e9, nmax=nsteps)
            (s, Q)
        end
        sf, Qf = dim == 1 ?
            advance((n, 1, 1), (1.0, 1.0, 1.0), (sym, per3[2], per3[3]),
                    (0.0, 0.0, 0.0)) :
            advance((nt, n, 1), (1.0, 1.0, 1.0), (per3[1], sym, per3[3]),
                    (0.0, 0.0, 0.0))
        # The doubled periodic line samples the same nodes: its origin is
        # shifted half a cell so that node j sits at (j − ½)h, as the folded
        # grid's node i does.
        sp, Qp = dim == 1 ?
            advance((2n, 1, 1), (2.0, 1.0, 1.0), per3, (h / 2, 0.0, 0.0)) :
            advance((nt, 2n, 1), (1.0, 2.0, 1.0), per3, (0.0, h / 2, 0.0))
        err, scale = 0.0, 0.0
        ni, nj = dim == 1 ? (n, 1) : (nt, n)
        for c in 1:sf.equations.n_cons, j in 1:nj, i in 1:ni
            vf = Qf[padded_index(sf, i, j, 1), c]
            vp = Qp[padded_index(sp, i, j, 1), c]
            err = max(err, abs(vf - vp))
            scale = max(scale, abs(vp))
        end
        return err / scale
    end
    # Measured on the workstation: 2.4e-15 to 3.6e-15 over the twelve
    # one-dimensional combinations below, 3.1e-15 viscous, and 4.3e-15 to
    # 5.5e-15 for the two-dimensional plane on dimension 2.
    for deriv in (lele_d1_6(), lele_d1_8(), lele_d1_10()),
        detector in (:delta4, :d8), smoother in (:gaussian, :compact)
        art = ArtificialProperties(enabled=true, detector=detector, smoother=smoother)
        @test mirror_error(deriv=deriv, art=art) < 2e-14
    end
    # Two species, the mass fractions even about both planes.
    @test mirror_error(eos=two) < 2e-14
    # Physical viscosity: the tangential shear traction and the normal heat
    # flux at the plane come from the fold's parities and nothing else.
    @test mirror_error(mu=0.005) < 2e-14
    # The plane on dimension 2, with the transverse direction periodic and a
    # tangential velocity, inviscid and viscous.
    @test mirror_error(dim=2, n=24) < 2e-14
    @test mirror_error(dim=2, n=24, mu=0.005) < 2e-14
end

@testset "symmetry plane: setup rejections" begin
    sym = (SymmetryPlaneBC(), SymmetryPlaneBC())
    wall = (SlipWallBC(), SlipWallBC())
    off = ArtificialProperties(enabled=false)
    cart(; stretch=(nothing, nothing, nothing), patch_grid=(1, 1, 1)) = build_solver(
        Problem(domain=((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)), bcs=(sym, per3[2], per3[3]),
                ic=nothing),
        Numerics(n_global=(48, 12, 12), art=off, stretch=stretch,
                 execution=Execution(patch_grid=patch_grid)))
    @test cart() isa Solver
    # A stretched dimension has no mirror: the fold reflects the computational
    # coordinate and the map does not commute with it.
    @test_throws ErrorException cart(
        stretch=(sine_cluster(0.0, 1.0, 0.5, 0.4), nothing, nothing))
    # The scale factors of the plane's own dimension must not depend on its
    # coordinate, which excludes the cylindrical radius and every spherical
    # dimension (validate_bc). The same rule makes a shared dimension with a
    # coordinate fold unreachable, since every dimension carrying one fails it.
    @test_throws ErrorException build_solver(
        Problem(domain=((0.0, 1.0), (0.0, π), (0.0, 2π)), metric=SphericalMetric(),
                bcs=(sym, (PoleBC(), PoleBC()), per3[3]), ic=nothing),
        Numerics(n_global=(16, 16, 12), art=off))
    @test_throws ErrorException build_solver(
        Problem(domain=((0.2, 0.2 + 1.0), (0.0, 0.0 + 2π), (0.0, 0.0 + 0.5)),
                metric=CylindricalMetric(), bcs=(sym, per3[2], per3[3]), ic=nothing),
        Numerics(n_global=(16, 16, 12), art=off))
    @test_throws ErrorException build_solver(
        Problem(domain=((0.0, 1.0), (0.0, 2π), (0.0, 0.5)), metric=CylindricalMetric(),
                bcs=((AxisBC(), SymmetryPlaneBC()), per3[2], per3[3]), ic=nothing),
        Numerics(n_global=(16, 16, 12), art=off))
    # A patched run takes SlipWallBC at that face instead. A refined run
    # keeps the plane, whether its level stays off it or reaches it (the
    # level tests cover the second).
    @test_throws ErrorException cart(patch_grid=(2, 1, 1))
    @test length(build_solver(
        Problem(domain=((0.0, 2π), (0.0, 1.0), (0.0, 1.0)), bcs=(sym, per3[2], per3[3]),
                ic=nothing),
        Numerics(n_global=(48, 1, 1), art=off, filter=StateFilter(interval=0),
                 amr=AMR(initial=BlockRegion((20, 0, 0), (8, 1, 1)),
                         regrid_interval=0))).patches) == 2
end

@testset "rigid rotation in cylindrical: zero strain" begin
    # u_θ = Ω r ⇒ S_ij = 0 identically; probes the curvature corrections.
    solver = build_solver(
        Problem(domain=((0.2, 0.2 + 1.0), (0.0, 0.0 + 2π), (0.0, 0.0 + 0.5)),
                metric=CylindricalMetric(),
                bcs=((SlipWallBC(), SlipWallBC()), per3[2], per3[3]), ic=nothing),
        Numerics(n_global=(32, 16, 12), art=ArtificialProperties(enabled=false)))
    Q = allocate_state(solver)
    initialize!(solver, Q, (r, θ, z) -> Prim(u=(0.0, 0.3r, 0.0), p=1.0, rho=1.0))
    dQ = zero(Q)
    compute_rhs!(solver, Q, dQ)
    smax = 0.0
    for k in 1:solver.decomp.n_local[3], j in 1:solver.decomp.n_local[2], i in 1:solver.decomp.n_local[1]
        I = padded_index(solver, i, j, k)
        for b in 1:3, a in 1:3
            smax = max(smax, abs(0.5 * (solver.grad_u[a, b][I] + solver.grad_u[b, a][I])))
        end
    end
    @test smax < 1e-8
end
