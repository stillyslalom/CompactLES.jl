# Patch-abstraction tests: layout arithmetic, and the two-conforming-patch
# verification gates of reference/AMR_GPU.md (manufactured smooth
# solution across the interface, acoustic pulse reflection, conservation
# drift against the single-patch baseline). Serial; the rank-partitioned
# multi-patch path is exercised by test/mpi_tests.jl.
#
# Guards follow the convergence-suite convention: measured values are baked in
# with headroom, and a moved digit means the interface treatment changed.

@testset "patch layout arithmetic" begin
    # Non-periodic split shares the interface plane: extents sum to N + P - 1.
    regions = CL.patch_slabs((97, 12, 1), (false, true, true), (2, 1, 1))
    @test length(regions) == 2
    @test regions[1].offset == (0, 0, 0)
    @test regions[1].extent[1] + regions[2].extent[1] == 98
    @test regions[2].offset[1] == regions[1].extent[1] - 1
    # Periodic split wraps: extents sum to N + P.
    regions = CL.patch_slabs((96, 1, 1), (true, true, true), (3, 1, 1))
    @test length(regions) == 3
    @test sum(r.extent[1] for r in regions) == 99
    # Rank counts: proportional, at least one each, summing to np.
    counts = CL.patch_rank_counts(regions, 7)
    @test sum(counts) == 7 && all(>=(1), counts)
    big = CL.patch_slabs((300, 1, 1), (true, true, true), (2, 1, 1))
    @test CL.patch_rank_counts([CL.BlockRegion((0, 0, 0), (30, 1, 1)),
                                CL.BlockRegion((0, 0, 0), (90, 1, 1))], 4) == [1, 3]
    @test_throws ErrorException CL.patch_slabs((96, 12, 1), (true, true, true),
                                               (2, 2, 1))
end

# Entropy-wave advection: ρ(x, t) = 1 + a sin(x − u₀t), constant u and p, is an
# exact Euler solution, so the max-norm error against it measures the full
# spatial discretization including the interface treatment.
function _entropy_wave_error(N::Int, patch_grid; tfinal=0.5, interface_rhs=:extended)
    per3 = ntuple(_ -> (PeriodicBC(), PeriodicBC()), 3)
    u0 = 0.5
    ic(x, y, z) = Prim(u=(u0, 0, 0), p=1.0, rho=1.0 + 0.2 * sin(x))
    solver = Solver(n_global=(N, 1, 1), L_domain=(2π, 1.0, 1.0), bcs=per3,
                    art=ArtParams(enabled=false), filter_interval=0,
                    patch_grid=patch_grid, interface_rhs=interface_rhs)
    Q = allocate_state(solver)
    initialize!(solver, Q, ic)
    run!(solver, Q; tfinal=tfinal)
    err = 0.0
    states = Q isa Vector ? Q : [Q]
    for (ps, Qp) in CL.eachpatch(solver, states)
        nx = ps.decomp.n_local[1]
        for i in 1:nx
            I = gidx(ps, i, 1, 1)
            exact = 1.0 + 0.2 * sin(xcoord(ps, 1, i) - u0 * solver.t)
            err = max(err, abs(Qp[I, 1] - exact))
        end
    end
    return err
end

@testset "two patches: manufactured smooth solution across the interface" begin
    errs = [_entropy_wave_error(N, (2, 1, 1)) for N in (48, 96, 192)]
    orders = [log2(errs[i] / errs[i+1]) for i in 1:2]
    @info "two-patch entropy wave" errs orders
    # The interface treatment must not stall convergence: the expectation
    # is at least the closure-cascade order ≈ 3.
    @test all(>(2.5), orders)
    # And it must stay a small perturbation on the single-patch answer.
    ref = _entropy_wave_error(192, (1, 1, 1))
    @test errs[3] < max(10 * ref, 1e-8)
end

@testset "two patches: acoustic pulse reflection at the interface" begin
    # Right-moving acoustic pulse (u′ = p′/ρc) launched in patch 1, crossing
    # the interface at x = π. After it passes, anything left behind it on the
    # patch-1 side is reflection.
    per3 = ntuple(_ -> (PeriodicBC(), PeriodicBC()), 3)
    amp = 1e-3
    pulse(x) = amp * exp(-40.0 * (x - π / 2)^2)
    ic(x, y, z) = Prim(u=(pulse(x) / sqrt(1.4), 0, 0), p=1.0 + pulse(x),
                       rho=(1.0 + pulse(x))^(1 / 1.4))
    solver = Solver(n_global=(192, 1, 1), L_domain=(2π, 1.0, 1.0), bcs=per3,
                    art=ArtParams(enabled=false), filter_interval=0,
                    patch_grid=(2, 1, 1))
    states = allocate_state(solver)
    initialize!(solver, states, ic)
    # c ≈ √1.4; run the pulse from π/2 to ≈ 3π/2, well past the interface.
    run!(solver, states; tfinal=π / sqrt(1.4))
    reflected = 0.0
    ps = PatchSolver(solver, solver.patches[1])
    CL.refresh_primitives!(ps, states[1])
    nx = ps.decomp.n_local[1]
    for i in 1:nx
        x = xcoord(ps, 1, i)
        x < π - 1.0 || continue    # behind the pulse, clear of the interface
        I = gidx(ps, i, 1, 1)
        reflected = max(reflected, abs(ps.p[I] - 1.0))
    end
    @info "two-patch pulse reflection" reflected reflected / amp
    # Measured 2.34e-3 at N = 192, converging at ≈ 5th order (6.5e-2 at 96,
    # 4.9e-5 at 384); the single-patch wake in the same window is 2.4e-10.
    @test reflected / amp < 5e-3
end

@testset "two patches: conservation drift vs single patch" begin
    per3 = ntuple(_ -> (PeriodicBC(), PeriodicBC()), 3)
    ic(x, y, z) = Prim(u=(0.3 + 0.1 * sin(x), 0, 0), p=1.0 + 0.05 * cos(x),
                       rho=1.0 + 0.2 * sin(x))
    drift = map(((1, 1, 1), (2, 1, 1))) do pg
        solver = Solver(n_global=(96, 1, 1), L_domain=(2π, 1.0, 1.0), bcs=per3,
                        art=ArtParams(enabled=false), filter_interval=1,
                        patch_grid=pg)
        Q = allocate_state(solver)
        initialize!(solver, Q, ic)
        states = Q isa Vector ? Q : [Q]
        rho0 = volume_integral(solver, [Array(view(s, :, :, :, 1)) for s in states])
        run!(solver, Q; tfinal=2.0)
        rho1 = volume_integral(solver, [Array(view(s, :, :, :, 1)) for s in states])
        abs(rho1 - rho0) / abs(rho0)
    end
    @info "conservation drift (single, two-patch)" drift
    @test drift[1] < 1e-13                  # periodic single patch: round-off
    @test drift[2] < 1e-6                   # interface drift term, small
end

@testset "patched solver rejects unsupported configurations" begin
    per3 = ntuple(_ -> (PeriodicBC(), PeriodicBC()), 3)
    @test_throws ErrorException Solver(n_global=(96, 1, 1), L_domain=(1.0, 1.0, 1.0),
                                       bcs=per3, patch_grid=(2, 1, 1),
                                       deriv=lele_d1_10())
    @test_throws ErrorException Solver(n_global=(96, 1, 1), L_domain=(1.0, 1.0, 1.0),
                                       bcs=per3, patch_grid=(2, 1, 1),
                                       art=ArtParams(detector=:d8))
    @test_throws ErrorException Solver(n_global=(96, 1, 1), L_domain=(1.0, 1.0, 1.0),
                                       metric=CylindricalMetric(),
                                       bcs=((AxisBC(), SlipWallBC()), per3[2], per3[3]),
                                       patch_grid=(2, 1, 1))
end
