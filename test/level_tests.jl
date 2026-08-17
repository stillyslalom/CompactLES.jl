# Stage 3 static-refinement tests: the two-level gates of reference/AMR_GPU.md
# (manufactured smooth solution spanning the coarse-fine boundary, a Sod shock
# through the refinement boundary in both directions with Cook sensors live,
# and conservation drift), plus the configuration guards. Serial, like the
# stage itself.
#
# Guards are measured values with headroom, per the convergence-suite
# convention. The interpolation/injection coupling and the reasons the
# invertible filter pair is not the default are documented at the top of
# src/levels.jl.

@testset "refined solver rejects unsupported configurations" begin
    per3l = ntuple(_ -> (PeriodicBC(), PeriodicBC()), 3)
    mk(; kw...) = Solver(; n_global=(96, 1, 1), L_domain=(2π, 1.0, 1.0),
                         bcs=per3l, refine=BlockRegion((40, 0, 0), (16, 1, 1)),
                         kw...)
    @test_throws ErrorException mk(deriv=lele_d1_10())
    @test_throws ErrorException mk(art=ArtParams(detector=:d8))
    @test_throws ErrorException mk(metric=CylindricalMetric())
    @test_throws ErrorException mk(patch_grid=(2, 1, 1))
    # Nesting margin: a region reaching the boundary is refused.
    @test_throws ErrorException Solver(n_global=(96, 1, 1), L_domain=(2π, 1, 1),
                                       bcs=per3l,
                                       refine=BlockRegion((0, 0, 0), (16, 1, 1)))
    # A refined solver reports two patches, level 1 second, at h/3.
    solver = mk()
    @test npatches(solver) == 2
    ps = PatchSolver(solver, solver.patches[2])
    @test ps.patch.level == 1
    @test ps.h[1] ≈ (2π / 96) / 3
    @test ps.decomp.n_local[1] == 3 * 16 - 2
    # Coincident nodes agree in physical coordinates.
    pc = PatchSolver(solver, solver.patches[1])
    @test xcoord(ps, 1, 1) ≈ xcoord(pc, 1, 41) atol = 1e-14
    @test xcoord(ps, 1, 46) ≈ xcoord(pc, 1, 56) atol = 1e-13
end

function _level_wave_error(N; mode=:inject, tfinal=0.5)
    per3l = ntuple(_ -> (PeriodicBC(), PeriodicBC()), 3)
    u0 = 0.5
    solver = Solver(n_global=(N, 1, 1), L_domain=(2π, 1.0, 1.0), bcs=per3l,
                    art=ArtParams(enabled=false), filter_interval=0,
                    level_restriction=mode,
                    refine=BlockRegion((N ÷ 2 - N ÷ 12, 0, 0), (N ÷ 6, 1, 1)))
    states = allocate_state(solver)
    initialize!(solver, states, (x, y, z) ->
        Prim(u=(u0, 0, 0), p=1.0, rho=1.0 + 0.2 * sin(x)))
    run!(solver, states; tfinal=tfinal)
    e = 0.0
    for (ps, Q) in CL.eachpatch(solver, states)
        for i in 1:ps.decomp.n_local[1]
            I = gidx(ps, i, 1, 1)
            e = max(e, abs(Q[I, 1] - (1.0 + 0.2 * sin(xcoord(ps, 1, i) -
                                                      u0 * solver.t))))
        end
    end
    return e
end

@testset "two levels: manufactured solution across the coarse-fine boundary" begin
    errs = [_level_wave_error(N) for N in (48, 96, 192)]
    orders = [log2(errs[i] / errs[i+1]) for i in 1:2]
    @info "two-level entropy wave" errs orders
    # Measured 8.5e-8 / 7.7e-9 / 6.2e-10, orders 3.46 / 3.64: the one-sided
    # divergence closures bind, as at a Stage 2 interface.
    @test all(>(3.0), orders)
    @test errs[2] < 3e-8
    # The deconvolution/filter coupling measures order 1.3-1.7 (see the
    # src/levels.jl header); pin that it stays selectable and stable.
    ef = _level_wave_error(96; mode=:filter)
    @test 1e-5 < ef < 1e-4
end

# Trapezoid mass over the two-level composite: the coarse level outside the
# covered region plus the fine level inside it, the shared boundary plane
# taking half weight from each.
function _two_level_mass(solver, states, N)
    lt = solver.level_transfer
    ps = PatchSolver(solver, solver.patches[1])
    pad = ps.decomp.n_halo_d[1]
    h = ps.h[1]
    lo = lt.region.offset[1] + 1
    hi = lt.region.offset[1] + lt.region.extent[1]
    m = 0.0
    for i in 1:N
        w = (i == 1 || i == N) ? 0.5 : 1.0
        (lo < i < hi) && continue
        (i == lo || i == hi) && (w = 0.5)
        m += w * states[1][i + pad, 1, 1, 1] * h
    end
    pf = PatchSolver(solver, solver.patches[2])
    padf = pf.decomp.n_halo_d[1]
    hf = pf.h[1]
    nf = pf.decomp.n_local[1]
    for i in 1:nf
        w = (i == 1 || i == nf) ? 0.5 : 1.0
        m += w * states[2][i + padf, 1, 1, 1] * hf
    end
    return m
end

@testset "two levels: Sod shock through the refinement boundary" begin
    wall2 = (SlipWallBC(), SlipWallBC())
    per = (PeriodicBC(), PeriodicBC())
    ic(x, y, z) = x < 0.5 ? Prim(u=(0, 0, 0), p=1.0, rho=1.0) :
                            Prim(u=(0, 0, 0), p=0.1, rho=0.125)
    N = 201
    # Region x ∈ [0.6, 0.8]: the shock (speed ≈ 1.75) enters through the low
    # face at t ≈ 0.057 and leaves through the high face at t ≈ 0.17, so one
    # run crosses the boundary in both directions, with the Cook sensors and
    # the state filter live throughout — the in-situ form of the Stage 1
    # sensor-injection measurement.
    solver = Solver(n_global=(N, 1, 1), L_domain=(1.0, 1.0, 1.0),
                    bcs=(wall2, per, per), cfl=0.4,
                    refine=BlockRegion((120, 0, 0), (41, 1, 1)))
    states = allocate_state(solver)
    initialize!(solver, states, ic)
    m0 = _two_level_mass(solver, states, N)
    run!(solver, states; tfinal=0.1, nmax=20000)
    # Ahead of the shock (x > 0.85) the exact solution is still quiescent, so
    # any momentum there beyond round-off is interface-generated noise.
    # Measured 6.4e-10 with the unrefined run at 1.9e-10.
    ps = PatchSolver(solver, solver.patches[1])
    pad = ps.decomp.n_halo_d[1]
    m1 = solver.equations.i_mom[1]
    noise = maximum(abs(states[1][i + pad, 1, 1, m1]) for i in 172:N)
    @info "Sod through refinement boundary" noise
    @test noise < 1e-8
    # Positivity holds through both crossings (interior only; physical-edge
    # halos are never written and stay zero).
    for (psq, Q) in CL.eachpatch(solver, states)
        n = psq.decomp.n_local[1]
        padq = psq.decomp.n_halo_d[1]
        @test minimum(Q[i + padq, 1, 1, 1] for i in 1:n) > 0.05
    end
    # Run on to t = 0.2 (shock exits the region) and measure conservation.
    run!(solver, states; tfinal=0.2, nmax=40000)
    drift = abs(_two_level_mass(solver, states, N) - m0) / m0
    @info "two-level Sod mass drift" drift
    # Measured 1.36e-4 (:inject; :filter halves it at three decades of smooth
    # accuracy — src/levels.jl header). The unrefined run drifts 5e-11.
    @test drift < 5e-4
end


@testset "two levels: 2-D refined region advects a smooth vortex field" begin
    # A genuinely multi-dimensional check that the tensor-product transfer
    # chain and the 2-D shell imposition hold together: entropy wave advected
    # diagonally through a 2-D refined region.
    per3l = ntuple(_ -> (PeriodicBC(), PeriodicBC()), 3)
    u0, v0 = 0.4, 0.3
    solver = Solver(n_global=(48, 48, 1), L_domain=(2π, 2π, 1.0), bcs=per3l,
                    art=ArtParams(enabled=false), filter_interval=0,
                    refine=BlockRegion((18, 18, 0), (12, 12, 1)))
    states = allocate_state(solver)
    initialize!(solver, states, (x, y, z) ->
        Prim(u=(u0, v0, 0), p=1.0, rho=1.0 + 0.1 * sin(x) * sin(y)))
    run!(solver, states; tfinal=0.5)
    e = 0.0
    for (ps, Q) in CL.eachpatch(solver, states)
        n = ps.decomp.n_local
        for j in 1:n[2], i in 1:n[1]
            I = gidx(ps, i, j, 1)
            exact = 1.0 + 0.1 * sin(xcoord(ps, 1, i) - u0 * solver.t) *
                          sin(xcoord(ps, 2, j) - v0 * solver.t)
            e = max(e, abs(Q[I, 1] - exact))
        end
    end
    @info "2-D refined advection" e
    # Measured 4.3e-8.
    @test e < 1e-6
end
