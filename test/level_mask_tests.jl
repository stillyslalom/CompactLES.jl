# The level tests of what a parent does at the nodes a child covers: its filter
# residual, derivatives and sensors, its step, and the composite quadrature and
# samples. Included by test/level_tests.jl, which defines the helpers used here.

@testset "a parent's filter pass drops the residual of the nodes a child covers" begin
    # The pass on the root of a refined run is f + A⁻¹ M (B − A) f along each
    # line, M zero at a fully covered node where the density is not resolved;
    # checked against dense periodic operators on a field of noise, for the
    # tridiagonal and the banded filter, with the resolution threshold and its
    # reach at zero so that every covered node of the noise takes the mask.
    threshold, reach = CL.CHILD_MASK_THRESHOLD[], CL.CHILD_MASK_REACH[]
    CL.CHILD_MASK_THRESHOLD[], CL.CHILD_MASK_REACH[] = 0.0, 0
    per3l = ntuple(_ -> (PeriodicBC(), PeriodicBC()), 3)
    for filt in (compact_filter(0.47), pyranda_filter())
        s = build_solver(
            Problem(domain=((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)), bcs=per3l, ic=nothing),
            Numerics(n_global=(24, 20, 1), filter=filt,
                     amr=AMR(initial=BlockRegion((6, 5, 0), (9, 8, 1)), regrid_interval=0)))
        states = allocate_state(s)
        ps = CL.PatchSolver(s, s.patches[1])
        Q = states.patches[1]
        n = ps.decomp.n_local
        o = ps.decomp.n_halo_d
        rng = Random.MersenneTwister(7)
        interior = ntuple(d -> o[d]+1:o[d]+n[d], 3)
        view(parent(Q), interior..., :) .= 1 .+ 0.1 .* rand(rng, n..., size(Q, 4))
        expected = Array(parent(Q)[interior..., :])
        keep = ps.patch.amr.covered[interior...] .!= 0xff
        @test count(!, keep) > 0
        for d in 1:2
            scheme = CL._plan_at(ps.plans.filter_plans, d).scheme
            lhs = scheme isa CL.CompactScheme ? [scheme.alpha] : scheme.lhs
            m = n[d]
            circ(c0, c) = [i == j ? c0 : (k = mod(i - j, m); k = min(k, m - k);
                           1 <= k <= length(c) ? c[k] : 0.0) for i in 1:m, j in 1:m]
            A = circ(1.0, lhs)
            B = circ(scheme.a0, scheme.coeffs)
            for c in axes(expected, 4), line in CartesianIndices(d == 1 ? (n[2],) : (n[1],))
                sel = d == 1 ? (:, line[1], 1, c) : (line[1], :, 1, c)
                mk = d == 1 ? keep[:, line[1], 1] : keep[line[1], :, 1]
                f = expected[sel...]
                expected[sel...] = f + A \ (mk .* ((B - A) * f))
            end
        end
        filter_state!(ps, Q)
        @test maximum(abs, parent(Q)[interior..., :] .- expected) < 1e-12
    end
    # On the radial line of an r-z run, a level at the axis: the root's line
    # folds at the axis and closes at the wall, and A and B are probed from
    # the plan of each parity, so the mirrored rows and the closure rows the
    # mask leaves alone are checked as the plan solves them.
    per = (PeriodicBC(), PeriodicBC())
    s = build_solver(
        Problem(domain=((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)),
                bcs=((AxisBC(), SlipWallBC()), per, per), metric=CylindricalMetric(),
                ic=nothing),
        Numerics(n_global=(40, 1, 1),
                 amr=AMR(initial=BlockRegion((0, 0, 0), (13, 1, 1)), regrid_interval=0)))
    states = allocate_state(s)
    ps = CL.PatchSolver(s, s.patches[1])
    Q = states.patches[1]
    o = ps.decomp.n_halo_d
    n = ps.decomp.n_local[1]
    rows = o[1]+1:o[1]+n
    rng = Random.MersenneTwister(11)
    view(parent(Q), rows, 1 + o[2], 1 + o[3], :) .= 1 .+ 0.1 .* rand(rng, n, size(Q, 4))
    f0 = Array(parent(Q)[rows, 1 + o[2], 1 + o[3], :])
    covered = ps.patch.amr.covered[rows, 1 + o[2], 1 + o[3]] .== 0xff
    @test count(covered) > 0
    fold = ps.folds[1]
    expected = similar(f0)
    probe = similar(parent(Q), size(parent(Q))[1:3])
    out = similar(probe)
    for c in axes(f0, 2)
        σ = CL.cons_parity(ps, 1, c)
        plan = CL._fold_plan(fold, σ, Val(:filter), 1, false)
        F = zeros(n, n)
        Ainv = zeros(n, n)
        for j in 1:n
            fill!(probe, 0)
            probe[o[1] + j, 1 + o[2], 1 + o[3]] = 1
            CL.fold_fill!(probe, ps.decomp, 1, fold.lo, fold.hi, σ)
            CL.apply_along!(out, plan, probe, ps.decomp)
            F[:, j] = out[rows, 1 + o[2], 1 + o[3]]
            CL.solve_along!(out, plan, probe, ps.decomp)
            Ainv[:, j] = out[rows, 1 + o[2], 1 + o[3]]
        end
        A = inv(Ainv)
        mk = [covered[i] && i <= n - length(plan.chi) ? 0.0 : 1.0 for i in 1:n]
        f = f0[:, c]
        expected[:, c] = f + Ainv * (mk .* ((A * F - A) * f))
    end
    filter_state!(ps, Q)
    @test maximum(abs, parent(Q)[rows, 1 + o[2], 1 + o[3], :] .- expected) < 1e-12
    CL.CHILD_MASK_THRESHOLD[], CL.CHILD_MASK_REACH[] = threshold, reach
    # A resolved field under the default threshold takes the plain pass, bit
    # for bit.
    smooth(r, θ, z) = Prim(rho=1 + 0.1 * cospi(r), u=(0.0, 0.0, 0.0), p=1.0)
    initialize!(s, states, smooth)
    plain = copy(parent(Q))
    CL.with_knob(CL.MASK_CHILD_RESIDUAL => false) do
        filter_state!(ps, ConservedState(plain))
    end
    filter_state!(ps, Q)
    @test parent(Q)[rows, :, :, :] == plain[rows, :, :, :]
end

@testset "a parent's derivatives drop the source of what a child resolves" begin
    # An advected slab, a density twice the ambient between two jumps one node
    # wide, at uniform velocity and pressure, deep inside a level. Away from
    # the jumps every rate vanishes. The root's plain compact derivatives carry
    # an alternating tail of the jumps to its uncovered nodes; the masked ones
    # drop the jumps' sources, so the uncovered rates are zero.
    per = (PeriodicBC(), PeriodicBC())
    s = build_solver(
        Problem(domain=((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)), bcs=(per, per, per),
                ic=nothing),
        Numerics(n_global=(64, 1, 1),
                 amr=AMR(initial=BlockRegion((16, 0, 0), (33, 1, 1)), regrid_interval=0)))
    states = allocate_state(s)
    slab(x, y, z) = Prim(rho=0.36 < x < 0.64 ? 2.0 : 1.0, u=(1.0, 0.0, 0.0), p=1.0)
    initialize!(s, states, slab)
    ps = CL.PatchSolver(s, s.patches[1])
    Q = states.patches[1]
    o = ps.decomp.n_halo_d
    rows = o[1]+1:o[1]+ps.decomp.n_local[1]
    line(A) = A[rows, 1 + o[2], 1 + o[3], :]
    uncovered = ps.patch.amr.covered[rows, 1 + o[2], 1 + o[3]] .== 0
    @test count(uncovered) > 20
    rates(masked) = begin
        dQ = ConservedState(zero(parent(Q)))
        CL.with_knob(CL.MASK_CHILD_DERIVATIVE => masked) do
            CL.compute_rhs!(ps, Q, dQ)
        end
        line(parent(dQ))
    end
    plain = rates(false)
    masked = rates(true)
    @test ps.patch.amr.child_masked == (true, false, false)
    scale = maximum(abs, plain)
    @test maximum(abs, plain[uncovered, :]) > 1e-4 * scale
    @test maximum(abs, masked[uncovered, :]) < 1e-13 * scale
    # A resolved state arms nothing and takes the plain derivatives, bit for
    # bit.
    initialize!(s, states, (x, y, z) -> Prim(rho=1 + 0.1 * sinpi(2x), u=(1.0, 0.0, 0.0),
                                             p=1.0))
    plain = rates(false)
    masked = rates(true)
    @test ps.patch.amr.child_masked == (false, false, false)
    @test masked == plain
end

@testset "the derivative mask and a parent's species gradients" begin
    # A slab of the heavier gas in the same place: the density and the
    # temperature jump across it, and the mask arms along x. Under
    # `MASK_CHILD_SPECIES_GRADIENTS = false` the gradients the species channel
    # differences (the mass fractions and the partial densities) are the plain
    # ones, bit for bit, and the temperature gradient takes the mask; under
    # `true` the first take it as well.
    per = (PeriodicBC(), PeriodicBC())
    eos = IdealMixture([IdealSpecies{Float64}("light", 1.0, 1.4),
                        IdealSpecies{Float64}("heavy", 0.25, 1.09)])
    s = build_solver(
        Problem(domain=((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)), bcs=(per, per, per), eos=eos,
                transport=ConstantTransport{Float64}(mu0=1e-3), ic=nothing),
        Numerics(n_global=(64, 1, 1), art=ArtificialProperties(enabled=true),
                 amr=AMR(initial=BlockRegion((16, 0, 0), (33, 1, 1)), regrid_interval=0)))
    states = allocate_state(s)
    initialize!(s, states, (x, y, z) -> begin
        θ = 0.36 < x < 0.64 ? 1.0 : 0.0
        Prim(Y=(1 - θ, θ), rho=1.0 + θ, u=(1.0, 0.0, 0.0), p=1.0)
    end)
    ps = CL.PatchSolver(s, s.patches[1])
    Q = states.patches[1]
    o = ps.decomp.n_halo_d
    rows = o[1]+1:o[1]+ps.decomp.n_local[1]
    line(A) = A[rows, 1 + o[2], 1 + o[3]]
    gradients(derivative, species) = begin
        CL.with_knob(CL.MASK_CHILD_DERIVATIVE => derivative,
                     CL.MASK_CHILD_SPECIES_GRADIENTS => species) do
            CL.compute_rhs!(ps, Q, ConservedState(zero(parent(Q))))
        end
        (Y=line(ps.grad_Y[1, 2]), rhoY=line(ps.grad_Q[1, 2]), T=line(ps.grad_T_ion[1]))
    end
    plain = gradients(false, false)
    species_plain = gradients(true, false)
    @test ps.patch.amr.child_masked == (true, false, false)
    masked = gradients(true, true)
    @test species_plain.Y == plain.Y
    @test species_plain.rhoY == plain.rhoY
    @test species_plain.T != plain.T
    @test masked.T == species_plain.T
    @test masked.Y != plain.Y
    @test masked.rhoY != plain.rhoY
end

@testset "a parent's sensors drop the nodes a child's restriction overwrites" begin
    # The slab of the heavier gas deep inside a level, and a second one the
    # root alone holds. The level's region spans root nodes 17 to 49: the
    # restriction writes 19 to 47 (grade 1 of `Patch.overwritten`), and the
    # nodes five and more inside a face take grade 2.
    per = (PeriodicBC(), PeriodicBC())
    eos = IdealMixture([IdealSpecies{Float64}("light", 1.0, 1.4),
                        IdealSpecies{Float64}("heavy", 0.25, 1.09)])
    slabs(x, y, z) = begin
        θ = (0.36 < x < 0.64 || 0.86 < x < 0.92) ? 1.0 : 0.0
        Prim(Y=(1 - θ, θ), rho=1.0 + θ, u=(1.0, 0.0, 0.0), p=1.0)
    end
    mk(; regrid_interval=0, tag_sensor_threshold=0) = build_solver(
        Problem(domain=((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)), bcs=(per, per, per), eos=eos,
                ic=nothing),
        Numerics(n_global=(64, 1, 1), art=ArtificialProperties(enabled=true),
                 amr=AMR(initial=BlockRegion((16, 0, 0), (33, 1, 1)),
                         regrid_interval=regrid_interval,
                         tag_sensor_threshold=tag_sensor_threshold)))
    s = mk()
    states = allocate_state(s)
    initialize!(s, states, slabs)
    ps = CL.PatchSolver(s, s.patches[1])
    Q = states.patches[1]
    o = ps.decomp.n_halo_d
    rows = o[1]+1:o[1]+ps.decomp.n_local[1]
    line(A) = A[rows, 1 + o[2], 1 + o[3]]
    grade = line(ps.patch.amr.overwritten)
    @test findall(>(0), grade) == 19:47
    @test findall(==(2), grade) == 22:44
    coefficients(masked) = begin
        CL.with_knob(CL.MASK_CHILD_SENSORS => masked) do
            CL.compute_rhs!(ps, Q, ConservedState(zero(parent(Q))))
        end
        (kappa=line(ps.kappa_art), D=line(ps.D_art[1]))
    end
    whole = coefficients(false)
    masked = coefficients(true)
    # The smoother reaches four nodes, so the coefficients vanish from the
    # fifth overwritten node in, where the whole sensors hold the slab's
    # edges, and are the whole ones, bit for bit, five nodes outside.
    inside, outside = 23:43, 52:64
    @test minimum(whole.kappa[[24, 42]]) > 0 && minimum(whole.D[[24, 42]]) > 0
    @test all(iszero, masked.kappa[inside]) && all(iszero, masked.D[inside])
    @test maximum(whole.D[outside]) > 0
    @test masked.kappa[outside] == whole.kappa[outside]
    @test masked.D[outside] == whole.D[outside]
    # A regrid check with the sensor criterion on tags from coefficients that
    # keep every sensor, and its priming leaves the masked ones for the step.
    r = mk(regrid_interval=4, tag_sensor_threshold=0.05)
    rstates = allocate_state(r)
    initialize!(r, rstates, slabs)
    rline(A) = A[rows, 1 + o[2], 1 + o[3]]
    CL._tag_coefficients!(r, rstates.patches)
    @test rline(r.patches[1].D_art[1])[inside] == whole.D[inside]
    CL._regrid_prime!(r, rstates.patches, CL._patch_workspace(CL.Workspace(rstates)),
                      nothing, false)
    @test all(iszero, rline(r.patches[1].D_art[1])[inside])
end

@testset "a level that resolves a feature sets its parent's step" begin
    # A disc of the heavier gas advected inside a subcycled level. The root's
    # nodes under the level hold the restricted disc, whose edge is narrower
    # than a root spacing. With the root's sensors dropped where the
    # restriction overwrites it, the level's rate sets the step: a third of
    # it exceeds the root's. With every sensor kept the root's rate does, and
    # the same number of steps covers less time.
    per = (PeriodicBC(), PeriodicBC())
    eos = IdealMixture([IdealSpecies{Float64}("light", 1.0, 1.4),
                        IdealSpecies{Float64}("heavy", 0.25, 1.09)])
    disc(x, y, z) = begin
        θ = hypot(x - 0.5, y - 0.5) < 0.1 ? 1.0 : 0.0
        Prim(Y=(1 - θ, θ), rho=1.0 + θ, u=(1.0, 0.5, 0.0), p=1.0)
    end
    rates(masked) = begin
        CL.with_knob(CL.MASK_CHILD_SENSORS => masked) do
            s = build_solver(
                Problem(domain=((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)), bcs=(per, per, per),
                        eos=eos, ic=nothing),
                Numerics(n_global=(64, 64, 1), cfl=0.5,
                         art=ArtificialProperties(enabled=true),
                         amr=AMR(initial=BlockRegion((16, 16, 0), (33, 33, 1)),
                                 regrid_interval=0, subcycle=true)))
            states = allocate_state(s)
            initialize!(s, states, disc)
            run!(s, states; tfinal=1e6, nmax=20)
            r = [CL._patch_rate!(s, p, states.patches[i])[1] for (i, p) in enumerate(s.patches)]
            (root=r[1], level=r[2], t=s.t)
        end
    end
    masked, whole = rates(true), rates(false)
    @test 3 * masked.root < masked.level
    @test 3 * whole.root > whole.level
    @test masked.root < 0.8 * whole.root
    @test masked.t > whole.t
end

@testset "a parent's gas beside a dense layer its child holds" begin
    # A layer three thousand times denser than the gas around it, of another
    # species, at rest at uniform pressure, its edges half a root spacing wide,
    # deep inside a level. The root's uncovered nodes hold the light gas alone,
    # and nothing should move. With the parent's species gradients masked, the
    # default, the gas on those nodes keeps its density and composition and
    # the root keeps its hyperbolic rate. With them plain the gas's density
    # there moves by tens of percent within thirty steps, its heavy fraction
    # leaves [0, 1] and the root's rate rises a hundredfold: the failure of a
    # capsule's shell under a refined band, whose gas fill is four orders
    # lighter than its ablator, without the capsule.
    per = (PeriodicBC(), PeriodicBC())
    eos = IdealMixture([IdealSpecies{Float64}("light", 1.0, 5 / 3),
                        IdealSpecies{Float64}("heavy", 0.25, 5 / 3)])
    n = 64
    layer(x, y, z) = begin
        θ = 1 - tanh_blend(abs(x - 0.5), 0.1, 0.5 / n)
        ρl, ρh = (1 - θ) + 1e-8, 3000θ + 1e-8
        Prim(Y=(ρl / (ρl + ρh), ρh / (ρl + ρh)), rho=ρl + ρh, u=(0.0, 0.0, 0.0), p=1.0)
    end
    gas(masked) = begin
        CL.with_knob(CL.MASK_CHILD_SPECIES_GRADIENTS => masked) do
            s = build_solver(
                Problem(domain=((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)), bcs=(per, per, per),
                        eos=eos, ic=nothing),
                Numerics(n_global=(n, 1, 1), cfl=0.25,
                         art=ArtificialProperties(enabled=true),
                         amr=AMR(initial=BlockRegion((16, 0, 0), (33, 1, 1)),
                                 regrid_interval=0)))
            states = allocate_state(s)
            initialize!(s, states, layer)
            root, Q = s.patches[1], states.patches[1]
            o = root.decomp.n_halo_d
            nodes = [CartesianIndex(i + o[1], 1 + o[2], 1 + o[3]) for i in 1:n
                     if root.amr.covered[i + o[1], 1 + o[2], 1 + o[3]] == 0x00]
            density() = [Q[I, 1] + Q[I, 2] for I in nodes]
            ρ0 = density()
            try
                run!(s, states; tfinal=1e9, nmax=30)
            catch e
                # The plain run ends in a state the validity check rejects.
                e isa CL.SolverFailure || rethrow()
            end
            ratio = density() ./ ρ0
            heavy = [Q[I, 2] / (Q[I, 1] + Q[I, 2]) for I in nodes]
            (low=minimum(ratio), high=maximum(ratio),
             excursion=maximum(max.(-heavy, heavy .- 1)),
             rate=CL._patch_rate!(s, root, Q)[1], nodes=length(nodes))
        end
    end
    @test CL.MASK_CHILD_SPECIES_GRADIENTS[]
    masked, plain = gas(true), gas(false)
    @test masked.nodes > 25
    @test 0.999 < masked.low && masked.high < 1.001
    @test masked.excursion < 1e-3
    @test plain.low < 0.9 && plain.high > 1.1
    @test plain.excursion > 0.1
    @test plain.rate > 50 * masked.rate
end

@testset "covered masks: composite quadrature counts every node once" begin
    # Node-centered trapezoid quadrature is exact for a linear field on a
    # wall-bounded box, so the composite integral, volume and plane
    # profiles of x + 2y + 3z are known to round-off on every layout: one
    # refined patch, a 3×3×3 tile nest sharing planes (the lattice reaches
    # past the requested box), and a three-level nest. The same sums
    # without the masks carry the covered volume twice, which is what the
    # masks exist to remove.
    wall = (SlipWallBC(), SlipWallBC())
    walls3 = (wall, wall, wall)
    N = 24
    two = IdealMixture([IdealSpecies{Float64}("a", 1.0, 1.4),
                        IdealSpecies{Float64}("b", 1.0, 1.4)])
    lin(x, y, z) = x + 2y + 3z
    function fields(s, states, f)
        map(CL.eachpatch(s, states)) do (ps, Q)
            a = similar(ps.rho)
            d = ps.decomp
            for k in 1:d.n_local[3], j in 1:d.n_local[2], i in 1:d.n_local[1]
                a[padded_index(ps, i, j, k)] = f(xcoord(ps, 1, i), xcoord(ps, 2, j),
                                                 xcoord(ps, 3, k))
            end
            a
        end
    end
    mk(; tile=0, refine=nothing) = build_solver(
        Problem(domain=((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)), bcs=walls3, eos=two,
                transport=ConstantTransport(mu0=0.01), ic=nothing),
        Numerics(n_global=(N, N, N), amr=AMR(initial=refine, regrid_interval=0, tile=tile)))
    # A composition linear in x and a shear u = y: the plane averages the
    # mixing measures need are exact at every station, and the shear's
    # gradient is exact for every compact derivative.
    ic(x, y, z) = Prim(Y=(x, 1 - x), u=(y, 0, 0), p=1.0, rho=1.0)
    s0 = mk()
    Q0 = allocate_state(s0)
    initialize!(s0, Q0, ic)
    W0 = mix_width(s0, Q0; dim=1)
    θ0 = molecular_mixing(s0, Q0; dim=1)
    tke0 = tke_profile(s0, Q0; dim=1)
    k0 = turbulent_kinetic_energy(s0, Q0; dim=1)
    diss0 = dissipation_rate(s0, Q0)
    refresh_primitives!(s0, Q0)
    pdf0 = species_pdf(s0, 1; nbins=4)[2]
    @test abs(W0 - 2 / 3) < 2e-3 && abs(θ0 - 1) < 1e-12
    r1 = BlockRegion((6, 6, 6), (8, 8, 8))
    layouts = ((tile=0, refine=r1), (tile=4, refine=r1),
               (tile=0, refine=[r1, BlockRegion((24, 24, 24), (8, 8, 8))]))
    for lay in layouts
        s = mk(tile=lay.tile, refine=lay.refine)
        states = allocate_state(s)
        initialize!(s, states, ic)
        # The state as `run!` leaves it: shells imposed and shared planes
        # exchanged, which the gradient diagnostic reads.
        CL._presync!(s, states)
        fs = fields(s, states, lin)
        # The level-1 cover in root nodes: the box itself, or the lattice
        # tiles meeting it, which reach past it.
        regs = level_regions(s, 1)
        lo = minimum(r.offset[1] for r in regs) + 1
        hi = maximum(r.offset[1] + r.extent[1] for r in regs)
        outside = vcat(1:lo-1, hi+1:N)
        @test abs(domain_volume(s) - 1) < 1e-13
        @test abs(volume_integral(s, fs) - 3) < 1e-13
        @test abs(volume_average(s, fs) - 3) < 1e-13
        unmasked = sum(CL._local_volume_integral(PatchSolver(s, p), fs[i], false)
                       for (i, p) in enumerate(s.patches))
        @test unmasked - 3 > 0.05
        c = (1, 2, 3)
        for d in 1:3
            prof = plane_profile(s, fs, d)
            xs = profile_coordinate(s, d)
            rest = sum(c[t] * 0.5 for t in 1:3 if t != d)
            @test length(prof) == N
            @test maximum(abs.(prof .- (c[d] .* xs .+ rest))) < 1e-12
            @test abs(sum(profile_spacing(s, d)) - 1) < 1e-13
        end
        # The root's mask: an interior node of the region is fully covered,
        # a face node keeps half its cell, an edge node three quarters, a
        # corner seven eighths, and a node outside all of it. In the plane
        # normal to x a face node at x = x_lo is fully covered, in the
        # plane normal to y it is half covered.
        root = PatchSolver(s, s.patches[1])
        m(g1, g2, g3) = root.amr.covered[padded_index(root, g1, g2, g3)]
        @test CL.uncovered_fraction(m(10, 10, 10)) == 0
        @test CL.uncovered_fraction(m(lo, 10, 10)) == 0.5
        @test CL.uncovered_fraction(m(hi, 10, 10)) == 0.5
        @test CL.uncovered_fraction(m(lo, lo, 10)) == 0.75
        @test CL.uncovered_fraction(m(lo, lo, lo)) == 0.875
        @test m(lo - 1, 10, 10) == 0 && m(hi + 1, 10, 10) == 0
        @test CL.uncovered_plane_fraction(m(lo, 10, 10), 1) == 0
        @test CL.uncovered_plane_fraction(m(lo, 10, 10), 2) == 0.5
        @test CL.uncovered_plane_fraction(m(10, 10, 10), 3) == 0
        # The mixing measures agree with the single level to round-off;
        # the turbulence profile agrees exactly on planes the region does
        # not meet and to trapezoid accuracy on those it does.
        @test abs(mix_width(s, states; dim=1) - W0) < 1e-12
        @test abs(molecular_mixing(s, states; dim=1) - θ0) < 1e-12
        tke = tke_profile(s, states; dim=1)
        @test maximum(abs.(tke[outside] .- tke0[outside])) < 1e-14
        @test maximum(abs.(tke .- tke0)) < 1e-2 * maximum(tke0)
        @test abs(turbulent_kinetic_energy(s, states; dim=1) - k0) < 1e-2 * k0
        @test abs(dissipation_rate(s, states) - diss0) < 1e-2 * diss0
        refresh_primitives!(s, states)
        pdf = species_pdf(s, 1; nbins=4)[2]
        @test abs(sum(pdf) * 0.25 - 1) < 1e-12
        @test maximum(abs.(pdf .- pdf0)) < 0.1
    end
    # A regrid rewrites the root's mask from the new tile set, and the
    # composite quadrature stays exact across it.
    per = (PeriodicBC(), PeriodicBC())
    sa = build_solver(
        Problem(domain=((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)), bcs=(wall, per, per),
                ic=nothing),
        Numerics(n_global=(201, 1, 1), cfl=0.2,
                 amr=AMR(initial=BlockRegion((85, 0, 0), (31, 1, 1)), subcycle=true,
                         regrid_interval=5, tile=8)))
    states = allocate_state(sa)
    initialize!(sa, states, (x, y, z) -> x < 0.5 ?
        Prim(u=(0, 0, 0), p=1.0, rho=1.0) : Prim(u=(0, 0, 0), p=0.1, rho=0.125))
    mask0 = copy(sa.patches[1].amr.covered)
    run!(sa, states; tfinal=1.0, nmax=20)
    root = sa.patches[1]
    @test root.amr.covered != mask0
    kept = copy(root.amr.covered)
    CL._fill_covered!(root, level_regions(sa, 1))
    @test root.amr.covered == kept
    @test abs(domain_volume(sa) - 1) < 1e-13
    @test abs(volume_integral(sa, fields(sa, states, (x, y, z) -> x)) - 0.5) < 1e-13
end

@testset "composite samples: line_sample and field_slice take the finest level" begin
    # A root node takes the value of the finest level holding it. The
    # reference below finds the holders by coordinate, independently of the
    # node-index arithmetic the implementation uses: per root node on the
    # line, the mean over the patches of the highest level with a node at
    # that position.
    wall = (SlipWallBC(), SlipWallBC())
    N = 24
    function explicit_line(s, fs, dim, ga, gb)
        a, b = CL._plane_dims(dim)
        xs = ntuple(d -> profile_coordinate(s, d), 3)
        n = s.n_global[dim]
        best, sums, cnt = fill(-1, n), zeros(n), zeros(n)
        for (li, p) in enumerate(s.patches)
            ps = PatchSolver(s, p)
            d = ps.decomp
            for k in 1:d.n_local[3], j in 1:d.n_local[2], i in 1:d.n_local[1]
                x = (xcoord(ps, 1, i), xcoord(ps, 2, j), xcoord(ps, 3, k))
                abs(x[a] - xs[a][ga]) < 1e-9 && abs(x[b] - xs[b][gb]) < 1e-9 ||
                    continue
                g = findfirst(xg -> abs(xg - x[dim]) < 1e-9, xs[dim])
                g === nothing && continue
                if p.level > best[g]
                    best[g], sums[g], cnt[g] = p.level, 0.0, 0.0
                end
                p.level == best[g] || continue
                sums[g] += fs[li][padded_index(ps, i, j, k)]
                cnt[g] += 1
            end
        end
        return sums ./ cnt
    end
    # Scaling a patch's whole state by 1 + level makes the density a marker
    # of the level a value came from.
    function mark_levels!(s, states)
        for (ps, Q) in CL.eachpatch(s, states)
            parent(Q) .*= 1 + ps.patch.level
        end
    end
    lin(x, y, z) = Prim(u=(0.1x^2, 0.1y, 0), p=1.0, rho=1 + x + 2y + 3z)
    r1 = BlockRegion((6, 6, 6), (8, 8, 8))
    layouts = ((refine=r1, tile=0, patch_grid=(1, 1, 1)),
               (refine=r1, tile=4, patch_grid=(1, 1, 1)),
               (refine=[r1, BlockRegion((24, 24, 24), (8, 8, 8))], tile=0,
                patch_grid=(1, 1, 1)),
               (refine=nothing, tile=0, patch_grid=(2, 1, 1)))
    for lay in layouts
        s = build_solver(
            Problem(domain=((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)), bcs=(wall, wall, wall),
                    ic=nothing),
            Numerics(n_global=(N, N, N), execution=Execution(patch_grid=lay.patch_grid),
                     amr=AMR(initial=lay.refine, regrid_interval=0, tile=lay.tile)))
        states = allocate_state(s)
        initialize!(s, states, lin)
        # A linear field is the same number on every level.
        x, v = line_sample(s, states, :rho; dim=1, index=(10, 12))
        xs = profile_coordinate(s, 2)
        @test x == profile_coordinate(s, 1)
        @test maximum(abs.(v .- (1 .+ x .+ 2xs[10] .+ 3xs[12]))) < 1e-13
        x1, x2, plane = field_slice(s, states, :rho; normal=2, index=12)
        @test size(plane) == (N, N)
        @test maximum(abs(plane[i, k] - (1 + x1[i] + 2xs[12] + 3x2[k]))
                      for i in 1:N, k in 1:N) < 1e-13
        # `at` snaps to the nearest root node.
        _, snapped = line_sample(s, states, :rho; dim=3,
                                 at=(x[10] + 0.01, xs[11] - 0.01))
        @test snapped == line_sample(s, states, :rho; dim=3, index=(10, 11))[2]
        # A derived field agrees with the explicit per-patch arrays.
        fs = field_array(s, states, :divergence)
        _, dv = line_sample(s, states, :divergence; dim=2, index=(11, 10))
        @test dv == explicit_line(s, fs, 2, 11, 10)
        _, v0 = line_sample(s, states, :rho; dim=1, index=(10, 10))
        mark_levels!(s, states)
        fs = field_array(s, states, :rho)
        for (dim, ga, gb) in ((1, 10, 12), (2, 11, 10), (3, 8, 13))
            _, v = line_sample(s, states, :rho; dim=dim, index=(ga, gb))
            @test v == explicit_line(s, fs, dim, ga, gb)
        end
        # The line through (10, 10) meets the finest level of every layout.
        top = maximum(p.level for p in s.patches)
        _, v = line_sample(s, states, :rho; dim=1, index=(10, 10))
        ratio = v ./ v0
        @test maximum(abs.(ratio .- round.(ratio))) < 1e-14
        @test maximum(ratio) ≈ 1 + top && minimum(ratio) ≈ 1
        @test_throws ArgumentError line_sample(s, states, :rho; index=(1, N + 1))
        @test_throws ArgumentError field_slice(s, states, :rho; index=0)
    end
    # After regrids: the tiles have moved, and a marked state still reads the
    # finest level at every root node the level-1 tiles hold.
    per = (PeriodicBC(), PeriodicBC())
    sa = build_solver(
        Problem(domain=((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)), bcs=(wall, per, per),
                ic=nothing),
        Numerics(n_global=(201, 1, 1), cfl=0.2,
                 amr=AMR(initial=BlockRegion((85, 0, 0), (31, 1, 1)), subcycle=true,
                         regrid_interval=5, tile=8)))
    states = allocate_state(sa)
    initialize!(sa, states, (x, y, z) -> x < 0.5 ?
        Prim(u=(0, 0, 0), p=1.0, rho=1.0) : Prim(u=(0, 0, 0), p=0.1, rho=0.125))
    before = level_regions(sa, 1)
    run!(sa, states; tfinal=1.0, nmax=20)
    @test level_regions(sa, 1) != before
    fs = field_array(sa, states, :rho)
    x, v = line_sample(sa, states, :rho)
    @test v == explicit_line(sa, fs, 1, 1, 1)
    held = falses(201)
    for r in level_regions(sa, 1)
        held[r.offset[1]+1:r.offset[1]+r.extent[1]] .= true
    end
    mark_levels!(sa, states)
    _, v2 = line_sample(sa, states, :rho)
    @test v2[held] == 2 .* v[held] && v2[.!held] == v[.!held]
    _, _, plane = field_slice(sa, states, :rho; normal=3)
    @test vec(plane) == v2
end

@testset "covered masks: refined TGV energy history matches the single level" begin
    # Taylor–Green at 24³ with an 8³ region refined off-center, where the
    # kinetic energy density is well above its mean, so a coarse node under
    # the level counted twice moves the composite energy by several percent
    # while the refinement itself moves it by parts in ten thousand over the
    # smooth early history. The refined run's history is compared with the
    # single-level one at the refined run's own instants, the single-level
    # history interpolated linearly between its steps.
    N = 24
    per3l = ntuple(_ -> (PeriodicBC(), PeriodicBC()), 3)
    c0 = 10.0
    p0 = c0^2 / 1.4
    tgv(x, y, z) = Prim(u=(sin(x) * cos(y) * cos(z), -cos(x) * sin(y) * cos(z), 0.0),
                        p=p0 + (cos(2x) + cos(2y)) * (cos(2z) + 2) / 16, rho=1.0)
    mk(; refine=nothing, subcycle=false) = build_solver(
        Problem(domain=((0.0, 2π), (0.0, 2π), (0.0, 2π)), bcs=per3l,
                transport=ConstantTransport(mu0=1 / 1600), ic=nothing),
        Numerics(n_global=(N, N, N), cfl=0.6,
                 amr=AMR(initial=refine, regrid_interval=0, subcycle=subcycle)))
    # Kinetic energy per unit volume: the composite quadrature of ½ρ|u|²
    # over the held patches, and the same sums without the masks.
    function energies(s, states)
        fs = map(CL.eachpatch(s, states.patches)) do (ps, Q)
            f = zeros(size(ps.rho))
            m = s.equations.i_mom
            d = ps.decomp
            for k in 1:d.n_local[3], j in 1:d.n_local[2], i in 1:d.n_local[1]
                I = padded_index(ps, i, j, k)
                f[I] = 0.5 * (Q[I, m[1]]^2 + Q[I, m[2]]^2 + Q[I, m[3]]^2) / Q[I, 1]
            end
            f
        end
        masked = volume_integral(s, fs) / (2π)^3
        unmasked = sum(CL._local_volume_integral(PatchSolver(s, p), fs[i], false)
                       for (i, p) in enumerate(s.patches)) / (2π)^3
        return masked, unmasked
    end
    function history(s, states, tfinal)
        ts = Float64[]
        kes = Float64[]
        raw = Float64[]
        record = Callback(EveryStep(1), (sv, Q) -> begin
            e, u = energies(sv, Q)
            push!(ts, sv.t)
            push!(kes, e)
            push!(raw, u)
            nothing
        end)
        run!(s, states; tfinal=tfinal, nmax=400, callback=record)
        return ts, kes, raw
    end
    # Both extrema are reached inside the first thirty steps: at t = 0.3
    # the drift maximum and the double-count minimum come out the same
    # numbers to every digit, so the second half of that run measured
    # nothing and cost half the testset.
    tfinal = 0.15
    s0 = mk()
    Q0 = allocate_state(s0)
    initialize!(s0, Q0, tgv)
    e0, u0 = energies(s0, Q0)
    @test e0 == u0
    ts0, kes0, _ = history(s0, Q0, tfinal)
    s1 = mk(refine=BlockRegion((4, 4, 4), (8, 8, 8)), subcycle=true)
    states = allocate_state(s1)
    initialize!(s1, states, tgv)
    e1, u1 = energies(s1, states)
    # At t = 0 both grids sample the same field; the fine sampling moves
    # the quadrature by its own truncation (measured 2.5e-4 relative), the
    # double count by the covered energy (measured 1.4e-2).
    @test abs(e1 - e0) < 5e-4 * e0
    @test u1 - e1 > 1e-2 * e0
    ts1, kes1, raw1 = history(s1, states, tfinal)
    interp(t) = (i = searchsortedlast(ts0, t); i == length(ts0) ? kes0[end] :
                 kes0[i] + (kes0[i+1] - kes0[i]) * (t - ts0[i]) / (ts0[i+1] - ts0[i]))
    ref = [t <= ts0[1] ? kes0[1] : interp(t) for t in ts1]
    drift = maximum(abs.(kes1 .- ref) ./ ref)
    double = minimum((raw1 .- kes1) ./ kes1)
    # Measured: the two histories take the same 31 steps to t = 0.15 and
    # stay within 2.5e-4 of each other, the sampling difference above;
    # the unmasked sum sits 1.4e-2 above throughout.
    @info "refined TGV energy history" steps=(length(ts0), length(ts1)) drift double
    @test drift < 1e-3
    @test double > 1e-2
end
