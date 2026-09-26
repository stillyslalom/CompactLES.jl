# Patch-abstraction tests: layout arithmetic, and the two-conforming-patch
# verification gates of reference/AMR_GPU.md (manufactured smooth
# solution across the interface, acoustic pulse reflection, conservation
# drift against the single-patch baseline). Serial; the rank-partitioned
# multi-patch path is exercised by test/mpi_tests.jl.
#
# Guards follow the convergence-suite convention: measured values are baked in
# with headroom, and a moved digit means the interface treatment changed.

using CompactLES: compute_rhs!, npatches, padded_index, xcoord

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
function _entropy_wave_error(N::Int, patch_grid; tfinal=0.5, interface_rhs=:extended,
                             deriv=lele_d1_6(), n_halo=4)
    per3 = ntuple(_ -> (PeriodicBC(), PeriodicBC()), 3)
    u0 = 0.5
    ic(x, y, z) = Prim(u=(u0, 0, 0), p=1.0, rho=1.0 + 0.2 * sin(x))
    solver = Solver(n_global=(N, 1, 1), L_domain=(2π, 1.0, 1.0), bcs=per3,
                    art=ArtificialProperties(enabled=false), filter_interval=0,
                    patch_grid=patch_grid, interface_rhs=interface_rhs,
                    deriv=deriv, n_halo=n_halo)
    Q = allocate_state(solver)
    initialize!(solver, Q, ic)
    run!(solver, Q; tfinal=tfinal)
    err = 0.0
    states = Q isa Vector ? Q : [Q]
    for (ps, Qp) in CL.eachpatch(solver, states)
        nx = ps.decomp.n_local[1]
        for i in 1:nx
            I = padded_index(ps, i, 1, 1)
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
    # The pentadiagonal C10 closes its interfaces with two compact rows per
    # end. Measured 8.2e-7 / 9.8e-8 / 8.5e-9, orders
    # 3.06 / 3.52: the same one-sided divergence rows bind.
    errs10 = [_entropy_wave_error(N, (2, 1, 1); deriv=lele_d1_10())
              for N in (48, 96, 192)]
    orders10 = [log2(errs10[i] / errs10[i+1]) for i in 1:2]
    @info "two-patch entropy wave, C10" errs10 orders10
    @test all(>(2.5), orders10)
    @test errs10[3] < 2e-8
end

# The inviscid gates above run the flux divergence alone, whose plans keep
# the one-sided rows at an interface, so they cannot tell `:extended` from
# `:onesided`. The interface rows serve the gradients, so a viscous wave
# exercises them: against the single-patch answer at the same N, the
# extended-data rows converge at order ≈ 4 and the one-sided ones at ≈ 2.
function _viscous_wave(N::Int, patch_grid; deriv, n_halo, interface_rhs=:extended)
    per3 = ntuple(_ -> (PeriodicBC(), PeriodicBC()), 3)
    ic(x, y, z) = Prim(u=(0.5 + 0.1 * sin(2x), 0, 0), p=1.0 + 0.05 * cos(x),
                       rho=1.0 + 0.2 * sin(x))
    solver = Solver(n_global=(N, 1, 1), L_domain=(2π, 1.0, 1.0), bcs=per3,
                    art=ArtificialProperties(enabled=false), filter_interval=0,
                    transport=ConstantTransport(mu0=2e-2), patch_grid=patch_grid,
                    deriv=deriv, n_halo=n_halo, interface_rhs=interface_rhs)
    Q = allocate_state(solver)
    initialize!(solver, Q, ic)
    run!(solver, Q; tfinal=0.5)
    states = Q isa Vector ? Q : [Q]
    rho = Dict{Int,Float64}()          # by root node
    for (ps, Qp) in CL.eachpatch(solver, states)
        for i in 1:ps.decomp.n_local[1]
            rho[ps.patch.region.offset[1] + i] = Qp[padded_index(ps, i, 1, 1), 1]
        end
    end
    return rho
end

@testset "two patches: viscous wave through the interface rows" begin
    for (label, deriv, n_halo) in (("C6", lele_d1_6(), 4), ("C10", lele_d1_10(), 4))
        errs = Dict{Symbol,Vector{Float64}}()
        for rhs in (:extended, :onesided)
            errs[rhs] = map([48, 96, 192]) do N
                single = _viscous_wave(N, (1, 1, 1); deriv=deriv, n_halo=n_halo)
                two = _viscous_wave(N, (2, 1, 1); deriv=deriv, n_halo=n_halo,
                                    interface_rhs=rhs)
                maximum(abs(two[g] - single[g]) for g in keys(single))
            end
        end
        ext, osd = errs[:extended], errs[:onesided]
        orders = [log2(ext[i] / ext[i+1]) for i in 1:2]
        @info "two-patch viscous wave, $label" ext osd orders
        # Measured C6 6.1e-5 / 3.0e-6 / 1.9e-7 (orders 4.34 / 4.00) against
        # one-sided 8.4e-5 / 2.0e-5 / 5.9e-6 (2.09 / 1.73); C10 6.0e-5 /
        # 4.0e-6 / 2.6e-7 (3.93 / 3.92) against 7.5e-5 / 1.9e-5 / 5.9e-6.
        @test all(>(3.0), orders)
        @test ext[3] < 1e-6
        @test osd[3] > 10 * ext[3]
    end
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
                    art=ArtificialProperties(enabled=false), filter_interval=0,
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
        I = padded_index(ps, i, 1, 1)
        reflected = max(reflected, abs(ps.p[I] - 1.0))
    end
    @info "two-patch pulse reflection" reflected reflected / amp
    # Measured 2.34e-3 at N = 192, converging at ≈ 5th order (6.5e-2 at 96,
    # 4.9e-5 at 384); the single-patch wake in the same window is 2.4e-10.
    @test reflected / amp < 5e-3
    # C10: 4.1e-3 at 192 (7.5e-2 at 96, 7.7e-5 at 384), the larger mismatch
    # between the interior rows and the divergence's C6 closure cascade.
    solver10 = Solver(n_global=(192, 1, 1), L_domain=(2π, 1.0, 1.0), bcs=per3,
                      art=ArtificialProperties(enabled=false), filter_interval=0,
                      patch_grid=(2, 1, 1), deriv=lele_d1_10())
    states10 = allocate_state(solver10)
    initialize!(solver10, states10, ic)
    run!(solver10, states10; tfinal=π / sqrt(1.4))
    reflected10 = 0.0
    ps10 = PatchSolver(solver10, solver10.patches[1])
    CL.refresh_primitives!(ps10, states10[1])
    for i in 1:ps10.decomp.n_local[1]
        xcoord(ps10, 1, i) < π - 1.0 || continue
        reflected10 = max(reflected10, abs(ps10.p[padded_index(ps10, i, 1, 1)] - 1.0))
    end
    @info "two-patch pulse reflection, C10" reflected10 reflected10 / amp
    @test reflected10 / amp < 1e-2
end

@testset "two patches: conservation drift vs single patch" begin
    per3 = ntuple(_ -> (PeriodicBC(), PeriodicBC()), 3)
    ic(x, y, z) = Prim(u=(0.3 + 0.1 * sin(x), 0, 0), p=1.0 + 0.05 * cos(x),
                       rho=1.0 + 0.2 * sin(x))
    drift = map(((1, 1, 1), (2, 1, 1))) do pg
        solver = Solver(n_global=(96, 1, 1), L_domain=(2π, 1.0, 1.0), bcs=per3,
                        art=ArtificialProperties(enabled=false), filter_interval=1,
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
    # C10's interface rows read four ghost layers, the default halo; a
    # narrower one is refused with the width to construct with.
    @test_throws "n_halo ≥ 4" Solver(n_global=(96, 1, 1), L_domain=(1.0, 1.0, 1.0),
                                     bcs=per3, patch_grid=(2, 1, 1),
                                     deriv=lele_d1_10(), n_halo=3)
    @test npatches(Solver(n_global=(96, 1, 1), L_domain=(1.0, 1.0, 1.0), bcs=per3,
                          patch_grid=(2, 1, 1), deriv=lele_d1_10())) == 2
    @test npatches(Solver(n_global=(96, 1, 1), L_domain=(1.0, 1.0, 1.0), bcs=per3,
                          patch_grid=(2, 1, 1), deriv=lele_d1_10(),
                          interface_rhs=:onesided)) == 2
    @test_throws ErrorException Solver(n_global=(96, 1, 1), L_domain=(1.0, 1.0, 1.0),
                                       bcs=per3, patch_grid=(2, 1, 1),
                                       art=ArtificialProperties(detector=:d8))
    @test_throws ErrorException Solver(n_global=(96, 1, 1), L_domain=(1.0, 1.0, 1.0),
                                       metric=CylindricalMetric(),
                                       bcs=((AxisBC(), SlipWallBC()), per3[2], per3[3]),
                                       patch_grid=(2, 1, 1))
end

# The flux divergence's rows at an interface end come from the
# `interface_divergence` scheme when one is given, independently of the
# gradient rows (`interface_rhs`) and of a physical end's rows.
_div_rows_match(plan, decomp, deriv, h, lo, hi) =
    (ref = CL.plan_direction(decomp, deriv, 1, h; lo_closures=lo, hi_closures=hi);
     plan.clo == ref.clo && plan.chi == ref.chi && plan.clo_first == ref.clo_first)

function _patched_rhs(; kw...)
    per3 = ntuple(_ -> (PeriodicBC(), PeriodicBC()), 3)
    s = Solver(n_global=(48, 1, 1), L_domain=(2π, 1.0, 1.0), bcs=per3,
               art=ArtificialProperties(enabled=false), filter_interval=0,
               transport=ConstantTransport(mu0=1e-2), patch_grid=(2, 1, 1); kw...)
    Q = allocate_state(s)
    initialize!(s, Q, (x, y, z) -> Prim(u=(0.5 + 0.1 * sin(2x), 0, 0),
                                        p=1.0 + 0.05 * cos(x), rho=1.0 + 0.2 * sin(x)))
    dQ = [zero(q) for q in Q]
    CL._presync!(s, Q)
    for lev in getfield(s, :levels)
        CL._level_rhs!(s, lev, Q, dQ, false)
    end
    return [parent(d) for d in dQ]
end

@testset "interface divergence rows follow their source scheme" begin
    per3 = ntuple(_ -> (PeriodicBC(), PeriodicBC()), 3)
    walls = ((SlipWallBC(), SlipWallBC()), per3[2], per3[3])
    c6 = lele_d1_6()
    bl = lele_d1_6(closures=:brady_livescu)
    # A wall at the low end of patch 1 keeps the derivative's own rows, and
    # the interface at its high end takes the source's, under either
    # gradient treatment.
    for rhs in (:extended, :onesided)
        s = Solver(n_global=(48, 1, 1), L_domain=(1.0, 1.0, 1.0), bcs=walls,
                   patch_grid=(2, 1, 1), interface_rhs=rhs, interface_divergence=bl)
        p1 = getfield(s, :patches)[1]
        @test _div_rows_match(p1.div_plans[1], p1.decomp, c6, p1.h[1],
                              nothing, bl.closures)
    end
    # The default, spelled out: an interface takes the cascade3 rows in place
    # of the neutral set, so naming them changes no bit of the right-hand side.
    @test _patched_rhs() == _patched_rhs(interface_divergence=lele_d1_6(closures=:cascade3))
    # With extended gradients a periodic pair reads the derivative's closure
    # rows only in the divergence, so a source scheme is the whole of the
    # difference between the two derivative operators there.
    @test _patched_rhs(interface_divergence=bl) == _patched_rhs(deriv=bl)
    @test _patched_rhs(interface_divergence=bl) != _patched_rhs()
    # Refused at setup: another interior, another element type, a filter, a
    # run without an interface, and more rows than a regridded patch holds.
    mk(; kw...) = Solver(n_global=(48, 1, 1), L_domain=(1.0, 1.0, 1.0), bcs=per3; kw...)
    @test_throws "interior coefficients" mk(patch_grid=(2, 1, 1),
                                            interface_divergence=lele_d1_8())
    @test_throws "floating-point types" mk(patch_grid=(2, 1, 1), deriv=lele_d1_6(),
                                           interface_divergence=lele_d1_6(Float32))
    @test_throws "interior coefficients" mk(patch_grid=(2, 1, 1), deriv=lele_d1_10(),
                                            interface_divergence=c6)
    @test_throws "interior coefficients" mk(patch_grid=(2, 1, 1),
                                            interface_divergence=compact_filter(0.45))
    @test_throws "has neither" mk(interface_divergence=bl)
    @test_throws "raise tile" mk(deriv=lele_d1_8(),
                                 interface_divergence=lele_d1_8(closures=:brady_livescu),
                                 refine=BlockRegion((18, 0, 0), (8, 1, 1)),
                                 regrid_interval=5)
    @test_throws "unknown closure set" lele_d1_10(closures=:brady_livescu)
    @test npatches(mk(patch_grid=(2, 1, 1), deriv=lele_d1_10(),
                      interface_divergence=lele_d1_10(closures=:cascade3))) == 2
end

# `interface_flux = :ghost` differences the inviscid flux through an
# interface end from fluxes evaluated on the ghost state. On data whose
# conserved variables are polynomials of degree 4 and whose inviscid fluxes
# are of degree 5, the interior rows, the gradient plans' interface rows and
# the order-6 level interpolation are all exact, so the right-hand side is
# exact to round-off at a same-level interface and at both ends of a refined
# patch; the one-sided rows of the default are not.
function _polynomial_rhs_error(; kw...)
    γ = 1.4
    ρ(x) = 1 + 0.2x - 0.1x^2;  dρ(x) = 0.2 - 0.2x
    u(x) = 0.3 + 0.2x;         du = 0.2
    p(x) = 1 + 0.1x^2;         dp(x) = 0.2x
    E(x) = p(x) / (γ - 1) + ρ(x) * u(x)^2 / 2
    dE(x) = dp(x) / (γ - 1) + (dρ(x) * u(x)^2 + 2ρ(x) * u(x) * du) / 2
    exact(x) = (-(dρ(x) * u(x) + ρ(x) * du),
                -(dρ(x) * u(x)^2 + 2ρ(x) * u(x) * du + dp(x)),
                -((dE(x) + dp(x)) * u(x) + (E(x) + p(x)) * du))
    per = (PeriodicBC(), PeriodicBC())
    ext = (ExtrapolationBC(), ExtrapolationBC())
    s = Solver(n_global=(96, 1, 1), L_domain=(1.0, 1.0, 1.0), bcs=(ext, per, per),
               art=ArtificialProperties(enabled=false), filter_interval=0; kw...)
    Q = allocate_state(s)
    initialize!(s, Q, (x, y, z) -> Prim(rho=ρ(x), u=(u(x), 0, 0), p=p(x)))
    dQ = [zero(q) for q in Q]
    CL._presync!(s, Q)
    for lev in getfield(s, :levels)
        CL._level_rhs!(s, lev, Q, dQ, false)
    end
    eq = s.equations
    err = 0.0
    # Within four nodes of an interface end: of the shared plane between the
    # two slabs, or of either end of the refined patch.
    for (ps, d) in CL.eachpatch(s, dQ)
        n = ps.decomp.n_local[1]
        lo = ps.bcs[1][1] isa CL.InterfaceBC
        hi = ps.bcs[1][2] isa CL.InterfaceBC
        for i in 1:n
            (lo && i <= 4) || (hi && i > n - 4) || continue
            I = padded_index(ps, i, 1, 1)
            e = exact(xcoord(ps, 1, i))
            err = max(err, abs(d[I, 1] - e[1]), abs(d[I, eq.i_mom[1]] - e[2]),
                      abs(d[I, eq.i_energy] - e[3]))
        end
    end
    return err
end

@testset "interface flux: ghost fluxes through interface ends" begin
    for layout in ((patch_grid=(2, 1, 1),), (refine=BlockRegion((40, 0, 0), (17, 1, 1)),))
        @test _polynomial_rhs_error(; layout..., interface_flux=:ghost) < 1e-11
        @test _polynomial_rhs_error(; layout...) > 1e-10
    end
    # The default spelled out changes no bit; on a viscous pair the ghost
    # path splits the flux and changes the interface rows only.
    @test _patched_rhs() == _patched_rhs(interface_flux=:closure)
    @test _patched_rhs(interface_flux=:ghost) != _patched_rhs()
    per3 = ntuple(_ -> (PeriodicBC(), PeriodicBC()), 3)
    mk(; kw...) = Solver(n_global=(48, 1, 1), L_domain=(1.0, 1.0, 1.0), bcs=per3; kw...)
    @test_throws "must be :closure or :ghost" mk(patch_grid=(2, 1, 1),
                                                 interface_flux=:extended)
    @test_throws "has neither" mk(interface_flux=:ghost)
    @test_throws "interface_rhs = :extended" mk(patch_grid=(2, 1, 1),
                                                interface_rhs=:onesided,
                                                interface_flux=:ghost)
    @test npatches(mk(patch_grid=(2, 1, 1), interface_flux=:ghost,
                      interface_divergence=lele_d1_6(closures=:brady_livescu))) == 2
end

# Under `interface_flux = :ghost` with molecular transport the viscous,
# conductive and diffusive flux joins the ghost-differenced part: at a
# same-level face its ghost values are the neighbour's own interior flux,
# exchanged after every patch has evaluated, and at a coarse-fine face they
# are evaluated from the gradients of the interpolated shell. The right-hand
# side on smooth exact data within four nodes of an interface end is compared
# with the uniform periodic operator at the patch's own spacing, sampled at
# the same points, over N = 48 and 96.
function _viscous_interface_error(N; kw...)
    per3 = ntuple(_ -> (PeriodicBC(), PeriodicBC()), 3)
    ic(x, y, z) = Prim(rho=1 + 0.2sin(x + 0.3), u=(0.4 + 0.1cos(2x), 0, 0),
                       p=1 + 0.1cos(x - 0.2), Y=(0.6 + 0.2sin(x), 0.4 - 0.2sin(x)))
    eos = IdealMixture([IdealSpecies{Float64}("a", 1.0, 1.4),
                        IdealSpecies{Float64}("b", 0.5, 1.3)])
    mk(; k...) = Solver(n_global=(N, 1, 1), L_domain=(2π, 1.0, 1.0), bcs=per3, eos=eos,
                        art=ArtificialProperties(enabled=false), filter_interval=0,
                        transport=ConstantTransport(mu0=2e-2); k...)
    s = mk(; kw...)
    Q = allocate_state(s)
    initialize!(s, Q, ic)
    dQ = [zero(q) for q in Q]
    CL._presync!(s, Q)
    for lev in getfield(s, :levels)
        CL._level_rhs!(s, lev, Q, dQ, false)
    end
    err = 0.0
    for (ps, d) in CL.eachpatch(s, dQ)
        h = ps.h[1]
        Nu = round(Int, 2π / h)
        su = mk(; n_global=(Nu, 1, 1))
        Qu = allocate_state(su)
        initialize!(su, Qu, ic)
        dQu = zero(Qu)
        compute_rhs!(su, Qu, dQu)
        n = ps.decomp.n_local[1]
        lo = ps.bcs[1][1] isa CL.InterfaceBC
        hi = ps.bcs[1][2] isa CL.InterfaceBC
        for i in 1:n
            (lo && i <= 4) || (hi && i > n - 4) || continue
            iu = mod1(round(Int, xcoord(ps, 1, i) / h) + 1, Nu)
            I = padded_index(ps, i, 1, 1)
            Iu = padded_index(su, iu, 1, 1)
            err = max(err, maximum(abs(d[I, c] - dQu[Iu, c]) for c in 1:size(d, 4)))
        end
    end
    return err
end

@testset "interface flux: molecular ghost fluxes" begin
    level(N) = BlockRegion((5N ÷ 12, 0, 0), (N ÷ 6 + 1, 1, 1))
    for (layout, ghost_max, order_min) in (((patch_grid=(2, 1, 1),), 1e-6, 5.5),
                                           ((refine=level,), 1e-8, 6.0))
        kw(N) = map(v -> v isa Function ? v(N) : v, layout)
        e48 = _viscous_interface_error(48; kw(48)..., interface_flux=:ghost)
        e96 = _viscous_interface_error(96; kw(96)..., interface_flux=:ghost)
        closure96 = _viscous_interface_error(96; kw(96)...)
        @test e96 < ghost_max
        @test log2(e48 / e96) > order_min
        @test closure96 > 100 * e96
    end
    # The ghost flux arrays exist only where they are read: under the ghost
    # path with molecular transport, along each interface dimension.
    per3 = ntuple(_ -> (PeriodicBC(), PeriodicBC()), 3)
    mk(; kw...) = Solver(n_global=(48, 1, 1), L_domain=(1.0, 1.0, 1.0), bcs=per3,
                         patch_grid=(2, 1, 1); kw...)
    extents(s) = [size(p.ghost_flux[d], 4) for p in getfield(s, :patches), d in 1:3]
    @test all(==(0), extents(mk(interface_flux=:ghost)))
    @test all(==(0), extents(mk(transport=ConstantTransport(mu0=1e-2))))
    viscous = extents(mk(interface_flux=:ghost, transport=ConstantTransport(mu0=1e-2)))
    @test all(==(5), viscous[:, 1]) && all(==(0), viscous[:, 2:3])
end
