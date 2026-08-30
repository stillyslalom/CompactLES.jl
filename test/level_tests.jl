# Two-level refinement tests: manufactured smooth solution spanning the
# coarse-fine boundary, a Sod shock through the refinement boundary in both
# directions with Cook sensors live, conservation drift, and the
# configuration guards; then the subcycled, tagged, and regridded variants.
# Serial; the decomposed forms live in mpi_tests.jl.
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

function _level_wave_error(N; mode=:inject, tfinal=0.5, subcycle=false,
                           levels=2)
    per3l = ntuple(_ -> (PeriodicBC(), PeriodicBC()), 3)
    u0 = 0.5
    r1 = BlockRegion((N ÷ 2 - N ÷ 12, 0, 0), (N ÷ 6, 1, 1))
    # A third level over the middle half of the level-1 patch, in that
    # patch's node space, so the nest scales with N.
    e1 = 3 * (N ÷ 6) - 2
    r2 = BlockRegion((e1 ÷ 4, 0, 0), (e1 ÷ 2, 1, 1))
    solver = Solver(n_global=(N, 1, 1), L_domain=(2π, 1.0, 1.0), bcs=per3l,
                    art=ArtParams(enabled=false), filter_interval=0,
                    level_restriction=mode, subcycle=subcycle,
                    refine=levels == 3 ? [r1, r2] : r1)
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
    # divergence closures bind, as at a same-level patch interface.
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
    region = CL.refined_region(solver)
    ps = PatchSolver(solver, solver.patches[1])
    pad = ps.decomp.n_halo_d[1]
    h = ps.h[1]
    lo = region.offset[1] + 1
    hi = region.offset[1] + region.extent[1]
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
    # the state filter live throughout — the in-situ form of the
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

# --- Subcycling, tagging, regridding -----------------------------------------

@testset "subcycle and regrid configuration guards" begin
    per3l = ntuple(_ -> (PeriodicBC(), PeriodicBC()), 3)
    mk(; kw...) = Solver(; n_global=(96, 1, 1), L_domain=(2π, 1.0, 1.0),
                         bcs=per3l, kw...)
    # Both features require an initial refined region.
    @test_throws ErrorException mk(subcycle=true)
    @test_throws ErrorException mk(regrid_interval=10)
    @test_throws ErrorException mk(refine=BlockRegion((40, 0, 0), (16, 1, 1)),
                                   regrid_interval=-1)
end

@testset "subcycled two levels: manufactured solution across the boundary" begin
    errs = [_level_wave_error(N; subcycle=true) for N in (48, 96, 192)]
    orders = [log2(errs[i] / errs[i+1]) for i in 1:2]
    @info "subcycled two-level entropy wave" errs orders
    # Measured 8.4e-8 / 7.5e-9 / 5.9e-10, orders 3.49 / 3.66 — within a few
    # percent of the global-dt coupling's figures, so the cubic Hermite
    # boundary data does not bind. The step count drops threefold: dt is now
    # coarse-limited (the fine rate enters the reduction divided by 3).
    @test all(>(3.0), orders)
    @test errs[2] < 3e-8
end

@testset "subcycled Sod through the refinement boundary" begin
    wall2 = (SlipWallBC(), SlipWallBC())
    per = (PeriodicBC(), PeriodicBC())
    ic(x, y, z) = x < 0.5 ? Prim(u=(0, 0, 0), p=1.0, rho=1.0) :
                            Prim(u=(0, 0, 0), p=0.1, rho=0.125)
    N = 201
    # The static gate's configuration with subcycling on: the same region,
    # CFL, and crossing schedule, so the guards compare directly.
    solver = Solver(n_global=(N, 1, 1), L_domain=(1.0, 1.0, 1.0),
                    bcs=(wall2, per, per), cfl=0.4, subcycle=true,
                    refine=BlockRegion((120, 0, 0), (41, 1, 1)))
    states = allocate_state(solver)
    initialize!(solver, states, ic)
    m0 = _two_level_mass(solver, states, N)
    run!(solver, states; tfinal=0.1, nmax=20000)
    ps = PatchSolver(solver, solver.patches[1])
    pad = ps.decomp.n_halo_d[1]
    m1 = solver.equations.i_mom[1]
    noise = maximum(abs(states[1][i + pad, 1, 1, m1]) for i in 172:N)
    @info "subcycled Sod through refinement boundary" noise
    # Measured 1.3e-10 against the global-dt gate's 6.4e-10 (5.7e-11 under the
    # former κ/(ρ cp) diffusive limit; the cv form takes different steps).
    @test noise < 1e-8
    for (psq, Q) in CL.eachpatch(solver, states)
        n = psq.decomp.n_local[1]
        padq = psq.decomp.n_halo_d[1]
        @test minimum(Q[i + padq, 1, 1, 1] for i in 1:n) > 0.05
    end
    run!(solver, states; tfinal=0.2, nmax=40000)
    drift = abs(_two_level_mass(solver, states, N) - m0) / m0
    @info "subcycled two-level Sod mass drift" drift
    # Measured 9.8e-5 against the global-dt gate's 1.36e-4.
    @test drift < 5e-4
end

@testset "tagging and regridding track a Sod shock" begin
    wall2 = (SlipWallBC(), SlipWallBC())
    per = (PeriodicBC(), PeriodicBC())
    ic(x, y, z) = x < 0.5 ? Prim(u=(0, 0, 0), p=1.0, rho=1.0) :
                            Prim(u=(0, 0, 0), p=0.1, rho=0.125)
    N = 201
    Nf = 3N - 2
    tf = 0.15
    # Uniform-fine reference and uniform-coarse baseline. CFL 0.2
    # throughout: the initial discontinuity sits inside the refined region, and a subcycled fine
    # level runs three substeps on one rate measurement, which tightens the
    # documented startup restriction (reference/AMR_GPU.md, two-level
    # refinement).
    sf = Solver(n_global=(Nf, 1, 1), L_domain=(1.0, 1.0, 1.0),
                bcs=(wall2, per, per), cfl=0.2)
    Qf = allocate_state(sf)
    initialize!(sf, Qf, ic)
    run!(sf, Qf; tfinal=tf, nmax=40000)
    padF = sf.decomp.n_halo_d[1]
    rho_ref = [Qf[i + padF, 1, 1, 1] for i in 1:Nf]
    sc = Solver(n_global=(N, 1, 1), L_domain=(1.0, 1.0, 1.0),
                bcs=(wall2, per, per), cfl=0.2)
    Qc = allocate_state(sc)
    initialize!(sc, Qc, ic)
    run!(sc, Qc; tfinal=tf, nmax=40000)
    padC = sc.decomp.n_halo_d[1]
    e_base = maximum(abs(Qc[i + padC, 1, 1, 1] - rho_ref[3i - 2]) for i in 1:N)
    # AMR: subcycled with the region retagged every 5 coarse steps.
    sa = Solver(n_global=(N, 1, 1), L_domain=(1.0, 1.0, 1.0),
                bcs=(wall2, per, per), cfl=0.2, subcycle=true,
                regrid_interval=5, refine=BlockRegion((85, 0, 0), (31, 1, 1)))
    states = allocate_state(sa)
    initialize!(sa, states, ic)
    run!(sa, states; tfinal=tf, nmax=40000)
    region = CL.refined_region(sa)
    lo = region.offset[1] + 1
    hi = region.offset[1] + region.extent[1]
    # The region moved off its initial site and holds the shock (x ≈ 0.76).
    shock_node = round(Int, (0.5 + 1.75 * sa.t) * (N - 1)) + 1
    @info "regrid tracking" region=(lo, hi) shock_node
    @test region.offset[1] != 85
    @test lo < shock_node < hi
    padc = sa.patches[1].decomp.n_halo_d[1]
    padf = sa.patches[2].decomp.n_halo_d[1]
    e_amr = 0.0
    for i in 1:N
        v = lo <= i <= hi ? states[2][3 * (i - lo) + 1 + padf, 1, 1, 1] :
                            states[1][i + padc, 1, 1, 1]
        e_amr = max(e_amr, abs(v - rho_ref[3i - 2]))
    end
    @test nlevels(sa) == 2
    @info "moving-region Sod vs uniform fine" e_amr e_base
    # Measured: composite 2.7e-3 against the uniform-coarse baseline's
    # 7.3e-2 (3.9e-3 under the former κ/(ρ cp) diffusive limit, whose steps
    # differ); the refinement recovers most of the uniform-fine answer at a
    # third of the fine points.
    @test e_amr < 1.5e-2
    @test e_base > 5e-2
    @test e_amr < e_base / 3
    for (psq, Q) in CL.eachpatch(sa, states)
        n = psq.decomp.n_local
        padq = psq.decomp.n_halo_d[1]
        @test minimum(Q[i + padq, 1, 1, 1] for i in 1:n[1]) > 0.05
    end
    # The global-dt coupling regrids through the same machinery.
    sg = Solver(n_global=(N, 1, 1), L_domain=(1.0, 1.0, 1.0),
                bcs=(wall2, per, per), cfl=0.2, regrid_interval=5,
                refine=BlockRegion((85, 0, 0), (31, 1, 1)))
    states_g = allocate_state(sg)
    initialize!(sg, states_g, ic)
    run!(sg, states_g; tfinal=0.05, nmax=20000)
    @test CL.refined_region(sg).offset[1] != 85
end

# --- Three levels ------------------------------------------------------------

@testset "three levels: configuration and geometry" begin
    per3l = ntuple(_ -> (PeriodicBC(), PeriodicBC()), 3)
    r1 = BlockRegion((40, 0, 0), (16, 1, 1))        # level-1 extent 46
    mk(r2; kw...) = Solver(; n_global=(96, 1, 1), L_domain=(2π, 1.0, 1.0),
                           bcs=per3l, refine=[r1, r2], kw...)
    # Nesting is checked against the parent patch, not the root grid.
    @test_throws ErrorException mk(BlockRegion((2, 0, 0), (10, 1, 1)))
    @test_throws ErrorException mk(BlockRegion((30, 0, 0), (14, 1, 1)))
    @test_throws ErrorException mk(BlockRegion((10, 0, 0), (3, 1, 1)))
    # Regridding stays two-level.
    @test_throws ErrorException mk(BlockRegion((10, 0, 0), (20, 1, 1)),
                                   regrid_interval=5)
    solver = mk(BlockRegion((10, 0, 0), (20, 1, 1)))
    @test nlevels(solver) == 3
    @test npatches(solver) == 3
    @test [p.level for p in solver.patches] == [0, 1, 2]
    @test refined_region(solver, 2).offset == (10, 0, 0)
    p0 = PatchSolver(solver, solver.patches[1])
    p1 = PatchSolver(solver, solver.patches[2])
    p2 = PatchSolver(solver, solver.patches[3])
    @test p2.h[1] ≈ p0.h[1] / 9
    @test p2.decomp.n_local[1] == 3 * 20 - 2
    # Level-2 node 1 coincides with level-1 node 11 and root node 40 + 11/3
    # does not exist; level-2 node 4 ↔ level-1 node 12, and level-1 node 13
    # ↔ root node 45.
    @test xcoord(p2, 1, 1) ≈ xcoord(p1, 1, 11) atol = 1e-14
    @test xcoord(p2, 1, 7) ≈ xcoord(p1, 1, 13) atol = 1e-14
    @test xcoord(p1, 1, 13) ≈ xcoord(p0, 1, 45) atol = 1e-14
end

@testset "three levels: manufactured solution across nested boundaries" begin
    for subcycle in (false, true)
        errs = [_level_wave_error(N; levels=3, subcycle=subcycle)
                for N in (48, 96, 192)]
        orders = [log2(errs[i] / errs[i+1]) for i in 1:2]
        @info "three-level entropy wave" subcycle errs orders
        # Measured 9.0e-8 / 9.1e-9 / 6.8e-10, orders 3.31 / 3.74 at the
        # global dt and 9.0e-8 / 8.8e-9 / 6.3e-10, orders 3.35 / 3.79
        # subcycled: the two-level figures (3.46 / 3.64) with a second
        # coarse-fine boundary pair inside the first.
        @test all(>(3.0), orders)
        @test errs[2] < 3e-8
    end
end

@testset "three levels: Sod through nested refinement boundaries" begin
    wall2 = (SlipWallBC(), SlipWallBC())
    per = (PeriodicBC(), PeriodicBC())
    ic(x, y, z) = x < 0.5 ? Prim(u=(0, 0, 0), p=1.0, rho=1.0) :
                            Prim(u=(0, 0, 0), p=0.1, rho=0.125)
    N = 201
    # The two-level Sod gate's region with a level-2 patch over the middle
    # of the level-1 patch (extent 121): the shock crosses four coarse-fine
    # boundaries with the sensors and filter live.
    solver = Solver(n_global=(N, 1, 1), L_domain=(1.0, 1.0, 1.0),
                    bcs=(wall2, per, per), cfl=0.4, subcycle=true,
                    refine=[BlockRegion((120, 0, 0), (41, 1, 1)),
                            BlockRegion((30, 0, 0), (60, 1, 1))])
    states = allocate_state(solver)
    initialize!(solver, states, ic)
    run!(solver, states; tfinal=0.1, nmax=20000)
    ps = PatchSolver(solver, solver.patches[1])
    pad = ps.decomp.n_halo_d[1]
    m1 = solver.equations.i_mom[1]
    # Momentum ahead of the shock on the two-level gate's schedule (t = 0.1,
    # x > 0.85) is refinement-boundary noise; measured 6.4e-10, the
    # two-level figure, so the inner boundary pair adds nothing visible.
    noise = maximum(abs(states[1][i + pad, 1, 1, m1]) for i in 172:N)
    @info "three-level Sod noise ahead of the shock" noise
    @test noise < 1e-8
    run!(solver, states; tfinal=0.2, nmax=40000)
    @test all(all(isfinite, parent(Q)) for Q in states)
    for (psq, Q) in CL.eachpatch(solver, states)
        n = psq.decomp.n_local[1]
        @test minimum(Q[gidx(psq, i, 1, 1), 1] for i in 1:n) > 0.05
    end
end
