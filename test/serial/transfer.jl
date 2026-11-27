# The serial suite's tests of the level-transfer operators. Included by
# test/serial_suite.jl, which loads the package and defines the imports and helpers
# used here; not runnable on its own.

# --- AMR level-transfer operators (reference/AMR_GPU.md) -------------------

"Fill the interior of a padded field with fn(x), x = (i-1)h along `dim`."
function transfer_fill!(f, decomp, dim, fn, h)
    pad = decomp.n_halo_d
    for k in 1:decomp.n_local[3], j in 1:decomp.n_local[2], i in 1:decomp.n_local[1]
        f[i+pad[1], j+pad[2], k+pad[3]] = fn(((i, j, k)[dim] - 1) * h)
    end
    return f
end

"Max interior difference of two padded fields on the same decomposition."
function transfer_dev(f, g, decomp)
    pad = decomp.n_halo_d
    e = 0.0
    for k in 1:decomp.n_local[3], j in 1:decomp.n_local[2], i in 1:decomp.n_local[1]
        e = max(e, abs(f[i+pad[1], j+pad[2], k+pad[3]] - g[i+pad[1], j+pad[2], k+pad[3]]))
    end
    return e
end

@testset "AMR transfer pair: DC gain, closures, invertibility" begin
    # Conservation is by unit zero-wavenumber gain: the two sides of the
    # transfer relation agree on it to the last bit, row 1 of the closure set
    # is exact, and row 2 is within one ULP of the interior gain.
    @test CL.AMR_A + 2 * CL.AMR_B + 2 * CL.AMR_C == 1 + 2 * CL.AMR_ALPHA
    @test sum(CL.AMR_EDGE_ROW1) == 1.0
    @test abs(sum(CL.AMR_EDGE_ROW2) - (1 + 2 * CL.AMR_ALPHA)) <= eps()

    restriction, prolongation = amr_transfer_schemes()
    @test restriction isa CompactScheme{Float64}
    @test prolongation isa BandedCompactScheme{Float64}

    # prolong(restrict(f)) == f with no sampling, through the distributed
    # solvers and the one-sided closure rows (measured 1.6e-15 closed,
    # 2.7e-15 periodic at n = 96).
    for periodic in (false, true)
        d = Decomp((96, 4, 4), (periodic, true, true); dims=(1, 1, 1))
        rplan = CL.plan_direction(d, restriction, 1, 1)
        pplan = CL.plan_direction(d, prolongation, 1, 1)
        f = CL.field(d); fbar = CL.field(d); back = CL.field(d)
        transfer_fill!(f, d, 1, x -> sin(5x) + 0.4cos(11x) + 0.1sin(29x), 2π / 96)
        CL.exchange_dim!(f, d, 1)
        apply_along!(fbar, rplan, f, d)
        CL.exchange_dim!(fbar, d, 1)
        apply_along!(back, pplan, fbar, d)
        @test transfer_dev(f, back, d) < 1e-14
    end

    # Constants through the full transfer, closures included: restriction
    # reproduces a constant to the last bit (measured 0.0), prolongation to a
    # few ULPs (measured 13 eps).
    nc = 24; nf = 3nc - 2
    df = Decomp((nf, 4, 4), (false, true, true); dims=(1, 1, 1))
    dc = Decomp((nc, 4, 4), (false, true, true); dims=(1, 1, 1))
    tp = plan_transfer(df, dc, 1)
    fine = CL.field(df); coarse = CL.field(dc)
    unit_f = CL.field(df); unit_c = CL.field(dc)
    transfer_fill!(unit_f, df, 1, x -> 1.0, 1.0)
    transfer_fill!(unit_c, dc, 1, x -> 1.0, 1.0)
    restrict!(coarse, tp, unit_f)
    @test transfer_dev(coarse, unit_c, dc) <= eps()
    prolong!(fine, tp, unit_c)
    @test transfer_dev(fine, unit_f, df) <= 32 * eps()

    # Each Lagrange weight row is computed in exact rational arithmetic and
    # sums to one within rounding of the converted weights.
    for order in (4, 6, 8)
        W = CL.amr_interpolation_weights(Float64, order)
        for r in 1:order-1, sub in 1:2
            @test abs(sum(W[:, sub, r]) - 1) <= 4 * eps()
        end
    end

    # Setup-time validation.
    @test_throws ErrorException plan_transfer(df, df, 1)          # wrong extents
    @test_throws ErrorException plan_transfer(df, dc, 1; interp_order=5)
    d1 = Decomp((nf, 1, 1), (false, true, true); dims=(1, 1, 1))
    @test_throws ErrorException plan_transfer(d1, d1, 2)          # collapsed dim
end

@testset "AMR transfer: 3:1 sampling convention" begin
    # The convention (reference/AMR_GPU.md risk 1, resolved by measurement):
    # restriction filters the fine line and takes the coincident nodes;
    # prolongation interpolates the coarse field onto the intermediate fine
    # nodes and deconvolves. Restriction is then a LEFT INVERSE of
    # prolongation, so coarse → fine → coarse is exact for arbitrary data —
    # the property that keeps levels consistent without refluxing.
    for (periodic, dim) in ((true, 1), (false, 2))
        nc = 32
        nf = periodic ? 3nc : 3nc - 2
        shape(n) = ntuple(d -> d == dim ? n : 4, 3)
        pers = ntuple(d -> d == dim ? periodic : true, 3)
        df = Decomp(shape(nf), pers; dims=(1, 1, 1))
        dc = Decomp(shape(nc), pers; dims=(1, 1, 1))
        tp = plan_transfer(df, dc, dim)
        hc = periodic ? 2π / nc : 1.0 / (nc - 1)
        coarse = CL.field(dc); coarse2 = CL.field(dc); fine = CL.field(df)
        transfer_fill!(coarse, dc, dim, x -> sin(5x) + exp(sin(3x)) + x, hc)
        prolong!(fine, tp, coarse)
        restrict!(coarse2, tp, fine)
        @test transfer_dev(coarse, coarse2, dc) < 1e-14
    end

    # Fine → coarse → fine loses subsampled content and converges at the
    # interpolation order: measured 5.93 between n_coarse = 32 and 64 at the
    # default order 6 (3.97 at order 4, 7.97 at order 8; bench/archive/amr_transfer.jl).
    errs = Float64[]
    for nc in (32, 64)
        nf = 3nc
        df = Decomp((nf, 4, 4), (true, true, true); dims=(1, 1, 1))
        dc = Decomp((nc, 4, 4), (true, true, true); dims=(1, 1, 1))
        tp = plan_transfer(df, dc, 1)
        fine = CL.field(df); fine2 = CL.field(df); coarse = CL.field(dc)
        transfer_fill!(fine, df, 1, x -> sin(2x) + cos(3x), 2π / nf)
        restrict!(coarse, tp, fine)
        prolong!(fine2, tp, coarse)
        push!(errs, transfer_dev(fine, fine2, df))
    end
    @test errs[2] < 2e-5
    @test log2(errs[1] / errs[2]) > 5.5
end

@testset "AMR transfer through a fold: symmetric closure variants" begin
    # Pyranda tabulates ±1 symmetric closure variants via ghost folding —
    # the same algebra `plan_direction`'s lo_fold performs. Pin the mapping:
    # a half line under a parity fold with mirror-filled halos must reproduce
    # the closed full line on parity-extended data, row for row.
    m = 12
    n = 2m
    dfull = Decomp((n, 4, 4), (false, true, true); dims=(1, 1, 1))
    dhalf = Decomp((m, 4, 4), (false, true, true); dims=(1, 1, 1))
    data = [exp(sin(0.4j)) + 0.05j for j in 1:m]
    for scheme in amr_transfer_schemes(), σ in (1, -1)
        plan_full = CL.plan_direction(dfull, scheme, 1, 1)
        plan_half = CL.plan_direction(dhalf, scheme, 1, 1; lo_fold=σ)
        pad = dfull.n_halo_d
        ffull = CL.field(dfull); ofull = CL.field(dfull)
        fhalf = CL.field(dhalf); ohalf = CL.field(dhalf)
        for k in 1:4, j in 1:4, i in 1:m
            ffull[pad[1]+m+i, pad[2]+j, pad[3]+k] = data[i]
            ffull[pad[1]+m+1-i, pad[2]+j, pad[3]+k] = σ * data[i]
            fhalf[pad[1]+i, pad[2]+j, pad[3]+k] = data[i]
        end
        for k in 1:4, j in 1:4, i in 1:pad[1]     # half-offset mirror halo
            fhalf[pad[1]+1-i, pad[2]+j, pad[3]+k] = σ * data[i]
        end
        apply_along!(ofull, plan_full, ffull, dfull)
        apply_along!(ohalf, plan_half, fhalf, dhalf)
        e = maximum(abs(ohalf[pad[1]+i, pad[2]+j, pad[3]+k] -
                        ofull[pad[1]+m+i, pad[2]+j, pad[3]+k])
                    for k in 1:4, j in 1:4, i in 1:m)
        @test e < 1e-13
    end
end

@testset "AMR transfer: sensor injection near a transfer end" begin
    # The deconvolution amplifies fine-Nyquist content by ≈ 20
    # (bench/archive/amr_transfer.jl), and the risk named in the plan is that content
    # injected by the Cook sensor chain near a transfer boundary rides through
    # prolongation. Measured: the smoothed δ⁴ sensor of a 2h shock profile
    # round-trips with amplification ≤ 1.13 at any distance from the end, the
    # state field undershoots by ≤ 3% of ambient, and the round-trip error
    # decays ≈ 3× per point once outside the shock footprint (≤ 1.6e-4 by 12
    # cells). The bounds here carry factor 2–6 headroom.
    nc = 33
    nf = 3nc - 2
    solver = build_solver(
        Problem(domain=((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)),
                bcs=((ExtrapolationBC(), ExtrapolationBC()), per3[2], per3[3]), ic=nothing),
        Numerics(n_global=(nf, 1, 1)))
    df = solver.decomp
    dc = Decomp((nc, 1, 1), (false, true, true); dims=(1, 1, 1))
    tp = plan_transfer(df, dc, 1)
    hf = 1.0 / (nf - 1)
    pad = df.n_halo_d
    rho = CL.field(df); rho2 = CL.field(df); coarse = CL.field(dc)
    sensor2 = CL.field(df)
    for dist in (6, 12)
        xs = 1.0 - dist * hf
        transfer_fill!(rho, df, 1, x -> 1.0 + 0.5 * (1 + tanh((x - xs) / (2hf))), hf)
        restrict!(coarse, tp, rho)
        prolong!(rho2, tp, coarse)
        @test minimum(rho2[pad[1]+1:pad[1]+nf, pad[2]+1, pad[3]+1]) > 1.0 - 0.06
        CL.exchange_halos!(rho, df)
        CL.delta4_sum!(solver.sensor, rho, solver, 1)
        CL.smooth!(solver.sensor, solver)
        smax = maximum(abs, solver.sensor)
        restrict!(coarse, tp, solver.sensor)
        prolong!(sensor2, tp, coarse)
        s2max = maximum(abs, sensor2[pad[1]+1:pad[1]+nf, pad[2]+1, pad[3]+1])
        @test s2max <= 2 * smax
    end
    # Localization: with the shock at mid-domain, the round-trip error 12 or
    # more cells away is below 1e-3 of the O(6e-2) error at the shock.
    transfer_fill!(rho, df, 1, x -> 1.0 + 0.5 * (1 + tanh((x - 0.5) / (2hf))), hf)
    restrict!(coarse, tp, rho)
    prolong!(rho2, tp, coarse)
    ic = (nf + 1) ÷ 2
    efar = maximum(abs(rho[pad[1]+i, pad[2]+1, pad[3]+1] -
                       rho2[pad[1]+i, pad[2]+1, pad[3]+1])
                   for i in 1:nf if abs(i - ic) >= 12)
    @test efar < 1e-3
end
