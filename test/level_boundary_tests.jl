# The level tests of levels at the domain's boundaries: walls, symmetry planes, the
# r-z axis, NSCBC faces, folds and periodic seams. Included by test/level_tests.jl,
# which defines the helpers used here.

# Measured 4.1e-8 at the low wall and 9.0e-9 at the high one, on the three
# conserved components of the 1-D standing wave at N = 49, t = 0.2.
const WALL_LEVEL_TOL = 1e-7

@testset "a level reaching a wall carries the wall on that face" begin
    per = (PeriodicBC(), PeriodicBC())
    function wall_level(N; refined=true, side=:lo, bc=SlipWallBC())
        m = (N - 1) ÷ 6 + 1
        region = BlockRegion((side == :lo ? 0 : N - m, 0, 0), (m, 1, 1))
        n = refined ? N : 3 * (N - 1) + 1
        solver = build_solver(
            Problem(domain=((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)), bcs=((bc, bc), per, per),
                    ic=nothing),
            Numerics(n_global=(n, 1, 1), cfl=0.9, art=ArtificialProperties(enabled=false),
                     filter=StateFilter(interval=0),
                     amr=AMR(initial=refined ? region : nothing, regrid_interval=0)))
        states = allocate_state(solver)
        initialize!(solver, states, (x, y, z) -> begin
            rho = 1 + 0.05 * cos(pi * x)
            Prim(rho=rho, u=(0.05 * sin(pi * x), 0.0, 0.0), p=rho^1.4)
        end)
        return solver, states
    end
    solver, states = wall_level(49)
    fine = solver.patches[2]
    lt = getfield(solver, :levels)[2].transfers[1]
    # The wall face carries the root's own condition object; the other face
    # stays parent-fed, and only it takes a shell, a box buffer and the
    # restriction margin.
    @test fine.bcs[1][1] === solver.patches[1].bcs[1][1]
    @test CL.parent_fed(fine.bcs[1][2])
    @test lt.boundary[1] == (true, false) && lt.imposed[1] == (false, true)
    @test CL._box_buffer(lt)[1] == (0, CL.LEVEL_BUFFER)
    @test first(CL._restrict_window(lt)[1]) == 1
    # The composite quadrature counts the covered wall node once, the wall
    # node's outer half-cell covered with its inner. The cells meet without
    # overlap at the coarse-fine face, so a linear field leaves the two
    # walls' half cells, which no longer cancel: one at the tile's spacing
    # and one at the root's, f'(H² − h²)/8 together for f = 1 + 2x. The
    # junction's Ω on the tile's high face adds −f'(H² − h²)/24.
    @test domain_volume(solver) ≈ 1.0 atol = 1e-14
    lin = map(getfield(solver, :patches)) do p
        ps = PatchSolver(solver, p)
        a = zeros(size(p.rho))
        for i in 1:ps.decomp.n_local[1]
            a[padded_index(ps, i, 1, 1)] = 1 + 2 * xcoord(ps, 1, i)
        end
        a
    end
    H = solver.patches[1].h[1]
    @test volume_integral(solver, lin) ≈ 2.0 + 2 * (H^2 - (H / 3)^2) / 12 atol = 1e-13
    root = PatchSolver(solver, solver.patches[1])
    @test root.amr.covered[padded_index(root, 1, 1, 1)] == 0xff
    # Against the uniform run at the level's spacing in the same equal steps,
    # the refined patch differs by the refinement's error alone, at the low
    # wall and, mirrored, at the high one.
    for side in (:lo, :hi)
        s, q = wall_level(49; side=side)
        f, fq = wall_level(49; refined=false)
        for k in 1:96
            run!(s, q; tfinal=0.2k / 96)
            run!(f, fq; tfinal=0.2k / 96)
        end
        @test s.step == f.step == 96
        pf = PatchSolver(f, f.patches[1])
        ps = PatchSolver(s, s.patches[2])
        off = ps.patch.region.offset[1]
        e = maximum(abs(q.patches[2][padded_index(ps, i, 1, 1), c] -
                        fq[padded_index(pf, off + i, 1, 1), c])
                    for i in 1:ps.decomp.n_local[1], c in 1:3)
        @info "wall level against the uniform run" side e
        @test e < WALL_LEVEL_TOL
    end
    # A face whose condition a level cannot carry is refused by name.
    @test_throws "ExtrapolationBC" wall_level(49; bc=ExtrapolationBC())
end

# Measured 5.3e-12 at the low plane and 1.0e-11 at the high one, on the three
# conserved components of the 1-D standing wave at N = 48, t = 0.2.
const PLANE_LEVEL_TOL = 1e-10

@testset "a level reaching a symmetry plane folds on that face" begin
    per = (PeriodicBC(), PeriodicBC())
    plane = SymmetryPlaneBC()
    function plane_level(N; refined=true, side=:lo, bcs=((plane, plane), per, per),
                         nest=nothing,)
        m = N ÷ 6 + 1
        region = BlockRegion((side == :lo ? 0 : N - m, 0, 0), (m, 1, 1))
        refine = !refined ? nothing : nest === nothing ? region : [region, nest]
        solver = build_solver(
            Problem(domain=((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)), bcs=bcs, ic=nothing),
            Numerics(n_global=(refined ? N : 3N, 1, 1), cfl=0.9,
                     art=ArtificialProperties(enabled=false),
                     filter=StateFilter(interval=0),
                     amr=AMR(initial=refine, regrid_interval=0)))
        states = allocate_state(solver)
        initialize!(solver, states, (x, y, z) -> begin
            rho = 1 + 0.05 * cos(pi * x)
            Prim(rho=rho, u=(0.05 * sin(pi * x), 0.0, 0.0), p=rho^1.4)
        end)
        return solver, states
    end
    N = 48
    for side in (:lo, :hi)
        s, q = plane_level(N; side=side)
        root = PatchSolver(s, s.patches[1])
        fine = PatchSolver(s, s.patches[2])
        lt = getfield(s, :levels)[2].transfers[1]
        k = side == :lo ? 1 : 2
        # The plane's face carries the root's condition and its fold; the
        # other face stays parent-fed.
        @test fine.bcs[1][k] === root.bcs[1][k]
        @test CL.parent_fed(fine.bcs[1][3 - k])
        @test fine.folds[1] isa CL.FoldSpec && fine.folds[1].pair === nothing
        @test (fine.folds[1].lo, fine.folds[1].hi) == (k == 1, k == 2)
        @test fine.folds[1].sigvel == root.folds[1].sigvel
        @test lt.boundary[1][k] && lt.folded[1][k] && !lt.imposed[1][k]
        @test lt.imposed[1][3 - k]
        # One fine node beyond the coincident lattice, half a fine cell from
        # the plane: the tile has 3m − 1 nodes and its node next to the plane
        # sits at h/6 from it.
        m = lt.region.extent[1]
        @test fine.decomp.n_global[1] == 3m - 1
        edge = k == 1 ? xcoord(fine, 1, 1) : 1 - xcoord(fine, 1, 3m - 1)
        @test edge ≈ root.h[1] / 6
        @test CL._box_buffer(lt)[1] == (CL.LEVEL_BUFFER, CL.LEVEL_BUFFER)
        # The restriction reaches the region's node on the plane side, the
        # coincident fine node one past the extra node there.
        win = CL._restrict_window(lt)[1]
        @test k == 1 ? first(win) == 1 : last(win) == m
        # The composite quadrature counts the covered cell at the plane once,
        # the tile's node there holding the whole fine cell [0, h/3].
        @test domain_volume(s) ≈ 1.0 atol = 1e-14
        lin = map(getfield(s, :patches)) do p
            ps = PatchSolver(s, p)
            a = zeros(size(p.rho))
            for i in 1:ps.decomp.n_local[1]
                a[padded_index(ps, i, 1, 1)] = 1 + 2 * xcoord(ps, 1, i)
            end
            a
        end
        # At a plane every cell is whole, and at the coarse-fine face the
        # root's face node takes its whole cell and the tile's cells start
        # beyond it, so every cell is centred on its node and the cells
        # integrate a linear field exactly. The junction's Ω adds
        # ±f'(h² − (h/3)²)/24 with the face on the tile's low or high side.
        h = root.h[1]
        face = (k == 1 ? -1 : 1) * 2 * (h^2 - (h / 3)^2) / 24
        @test volume_integral(s, lin) ≈ 2.0 + face atol = 1e-13
        # Against the uniform run at the level's spacing in the same equal
        # steps, the tile differs by the refinement's error alone.
        f, fq = plane_level(N; refined=false)
        for n in 1:96
            run!(s, q; tfinal=0.2n / 96)
            run!(f, fq; tfinal=0.2n / 96)
        end
        @test s.step == f.step == 96
        # The box keeps its buffer across the plane, the mirror of the
        # parent's nodes, with the component's parity.
        box = lt.box_gather
        pad = lt.pdecomps[1].n_halo_d[1]
        nb = lt.pdecomps[1].n_global[1]
        B = CL.LEVEL_BUFFER
        @test any(!iszero, view(box, :, 1, 1, 2))
        for j in 1:B
            out, in_ = k == 1 ? (B + 1 - j, B + j) : (nb - B + j, nb - B + 1 - j)
            @test box[out + pad, 1, 1, 1] == box[in_ + pad, 1, 1, 1]
            @test box[out + pad, 1, 1, 2] == -box[in_ + pad, 1, 1, 2]
        end
        pf = PatchSolver(f, f.patches[1])
        # The uniform run's node 1 lies at h/6, the root's at h/2, so the
        # level's node g is the uniform run's node g + 1.
        off = fine.patch.region.offset[1] + 1
        @test maximum(abs(xcoord(fine, 1, i) - xcoord(pf, 1, off + i))
                      for i in 1:fine.decomp.n_local[1]) < 1e-15
        e = maximum(abs(q.patches[2][padded_index(fine, i, 1, 1), c] -
                        fq[padded_index(pf, off + i, 1, 1), c])
                    for i in 1:fine.decomp.n_local[1], c in 1:3)
        @info "plane level against the uniform run" side e
        @test e < PLANE_LEVEL_TOL
    end
    # A corner tile between planes at x = 0 and y = 0 against the periodic
    # image of the run: [0, 2)² on 2N × 2N nodes, the root's node i at
    # (i − ½)h along both as in the folded run, and the level across both
    # seams over the image of the folded region and the region itself, so
    # the tile's node (i, j) is the image's node (3m − 1 + i, 3m − 1 + j).
    # The image's restriction filters the tile's lines through each plane
    # with the interior stencil, so a restriction that folds its filter at
    # both planes, stage by stage, reproduces it to round-off, the odd
    # momenta included; one that closed or mis-signed the filter at a plane
    # differs by the filter's own defect, measured 1.0e-4 on the root with
    # every component taken even.
    ic2(x, y, z) = begin
        rho = 1 + 0.05 * cos(pi * x) * cos(pi * y)
        Prim(rho=rho, u=(0.05 * sin(pi * x) * cos(pi * y),
                         0.03 * cos(pi * x) * sin(pi * y), 0.0), p=rho^1.4)
    end
    function corner_level(N; image=false, level_restriction=:inject)
        m = N ÷ 6 + 1
        solver = image ?
            build_solver(
                Problem(domain=((0.5 / N, 0.5 / N + 2.0), (0.5 / N, 0.5 / N + 2.0),
                                (0.0, 1.0)),
                        bcs=(per, per, per), ic=nothing),
                Numerics(n_global=(2N, 2N, 1), cfl=0.9,
                         art=ArtificialProperties(enabled=false),
                         filter=StateFilter(interval=0),
                         amr=AMR(initial=BlockRegion((2N - m, 2N - m, 0), (2m, 2m, 1)),
                                 regrid_interval=0, level_restriction=level_restriction))) :
            build_solver(
                Problem(domain=((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)),
                        bcs=((plane, plane), (plane, plane), per), ic=nothing),
                Numerics(n_global=(N, N, 1), cfl=0.9,
                         art=ArtificialProperties(enabled=false),
                         filter=StateFilter(interval=0),
                         amr=AMR(initial=BlockRegion((0, 0, 0), (m, m, 1)),
                                 regrid_interval=0, level_restriction=level_restriction)))
        states = allocate_state(solver)
        initialize!(solver, states, ic2)
        return solver, states
    end
    N = 24
    m = N ÷ 6 + 1
    for mode in (:inject, :filter)
        s, q = corner_level(N; level_restriction=mode)
        r, rq = corner_level(N; image=true, level_restriction=mode)
        for n in 1:2N
            run!(s, q; tfinal=0.2n / 2N)
            run!(r, rq; tfinal=0.2n / 2N)
        end
        @test s.step == r.step == 2N
        fine, image = PatchSolver(s, s.patches[2]), PatchSolver(r, r.patches[2])
        nl = fine.decomp.n_local
        e = maximum(abs(q.patches[2][padded_index(fine, i, j, 1), c] -
                        rq.patches[2][padded_index(image, 3m - 1 + i, 3m - 1 + j, 1), c])
                    for i in 1:nl[1], j in 1:nl[2], c in 1:4)
        root, rimage = PatchSolver(s, s.patches[1]), PatchSolver(r, r.patches[1])
        e = max(e, maximum(abs(q.patches[1][padded_index(root, i, j, 1), c] -
                               rq.patches[1][padded_index(rimage, i, j, 1), c])
                           for i in 1:N, j in 1:N, c in 1:4))
        @info "corner level against its periodic image" mode e
        @test e < 1e-13
    end
    # A second level stopping short of the plane is told the offset reaching
    # it.
    @test_throws "offset -1 reaches the SymmetryPlaneBC" plane_level(N;
        nest=BlockRegion((0, 0, 0), (10, 1, 1)))
end

# The r-z level's fine patch against the uniform run at its spacing, a
# cylindrical pulse leaving it through both faces, viscous, N = 49: measured
# 5.3e-7 over the four components; the Cartesian case of the same shape
# measures 4.0e-7.
const RZ_LEVEL_TOL = 1e-6

@testset "an axisymmetric level: per-tile metric, freestream and r-weighted budgets" begin
    wall = (SlipWallBC(), SlipWallBC())
    per = (PeriodicBC(), PeriodicBC())
    axis = ((AxisBC(), SlipWallBC()), per, per)
    rz(n; origin=0.5, bcs=(wall, per, per), L_domain=(1.0, 1.0, 1.0), refine=nothing,
       transport=ConstantTransport(), cfl=0.5, filter_interval=1,
       art=ArtificialProperties()) =
        build_solver(
            Problem(domain=((origin, origin + L_domain[1]), (0.0, L_domain[2]),
                            (0.0, L_domain[3])),
                    bcs=bcs, metric=CylindricalMetric(), transport=transport, ic=nothing),
            Numerics(n_global=n, cfl=cfl, art=art,
                     filter=StateFilter(interval=filter_interval),
                     amr=AMR(initial=refine, regrid_interval=0)))
    # Each tile evaluates its geometry at its own nodes: the refined patch's
    # arrays equal the uniform run's at the level's spacing node for node,
    # halo layers included.
    solver = rz((25, 1, 24); refine=BlockRegion((8, 0, 6), (9, 1, 10)))
    uniform = rz((73, 1, 72))
    ps = PatchSolver(solver, solver.patches[2])
    pu = PatchSolver(uniform, uniform.patches[1])
    shift = CartesianIndex(ps.patch.region.offset[1], 0, ps.patch.region.offset[3])
    geometry(p) = (p.inv_J, p.area_d[1], p.area_d[3], p.inv_h[1], p.inv_h[3], p.inv_r)
    e = 0.0
    for (a, b) in zip(geometry(ps), geometry(pu)), I in CartesianIndices(a)
        J = I + shift
        checkbounds(Bool, b, J) && (e = max(e, abs(a[I] - b[J]) / abs(b[J])))
    end
    @test e < 1e-14
    # The composite quadrature weighs each node by its own r: exact for the
    # linear J = r, the annulus holding (3/2)²/2 − (1/2)²/2 = 1 per radian
    # per unit length, and for a density linear in z between z walls.
    @test domain_volume(solver) ≈ 1.0 atol = 1e-14
    closed = rz((25, 1, 25); bcs=(wall, per, wall),
                refine=BlockRegion((8, 0, 6), (9, 1, 10)))
    states = allocate_state(closed)
    initialize!(closed, states,
                (r, θ, z) -> Prim(rho=1 + 0.5z, u=(0.0, 0.0, 0.0), p=1.0))
    budget = CL._conserved_budget(closed, states)
    @test budget.total_mass ≈ 1.25 atol = 1e-13
    @test budget.total_energy ≈ 2.5 atol = 1e-13
    # A root on the axis keeps its fold, and a level held off it by the
    # nesting margin changes nothing of the root's quadrature.
    onaxis = rz((37, 1, 24); origin=0.0, bcs=axis,
                refine=BlockRegion((12, 0, 6), (10, 1, 8)))
    @test domain_volume(onaxis) ≈
          domain_volume(rz((37, 1, 24); origin=0.0, bcs=axis)) atol = 1e-14
    @test_throws "CylindricalMetric with θ collapsed" rz((24, 16, 1);
        L_domain=(1.0, 2π, 1.0), refine=BlockRegion((8, 0, 0), (6, 16, 1)))
    # Freestream: a uniform state translating along z stays uniform to
    # round-off on either root, through the viscous ghost fluxes and their
    # curvature terms.
    for (origin, bcs) in ((0.5, (wall, per, per)), (0.0, axis))
        s = rz((37, 1, 24); origin=origin, bcs=bcs,
               refine=BlockRegion((12, 0, 6), (10, 1, 8)),
               transport=ConstantTransport(mu0=1e-3))
        q = allocate_state(s)
        initialize!(s, q, (r, θ, z) -> Prim(rho=1.3, u=(0.0, 0.0, 0.3), p=0.9))
        q0 = [copy(parent(x)) for x in q.patches]
        run!(s, q; tfinal=1.0, nmax=10)
        drift = 0.0
        for (i, p) in enumerate(getfield(s, :patches))
            pp = PatchSolver(s, p)
            nl = pp.decomp.n_local
            for c in 1:5, k in 1:nl[3], j in 1:nl[2], ii in 1:nl[1]
                I = padded_index(pp, ii, j, k)
                drift = max(drift, abs(q.patches[i][I, c] - q0[i][I, c]))
            end
        end
        @test s.step == 10
        @test drift < 1e-12
    end
    # Against the uniform run at the level's spacing in the same equal steps,
    # the refined patch differs by the coupling's error alone.
    pulse(n; refine=nothing) = begin
        local sv = rz((n, 1, 1); bcs=(wall, per, per), refine=refine, cfl=0.9,
                      filter_interval=0, art=ArtificialProperties(enabled=false),
                      transport=ConstantTransport(mu0=0.002))
        local sq = allocate_state(sv)
        initialize!(sv, sq, (r, θ, z) -> begin
            rho = 1 + 0.05 * exp(-((r - 1) / 0.1)^2)
            Prim(rho=rho, u=(0.0, 0.0, 0.0), p=rho^1.4)
        end)
        sv, sq
    end
    s, q = pulse(49; refine=BlockRegion((20, 0, 0), (9, 1, 1)))
    f, fq = pulse(145)
    for k in 1:576
        run!(s, q; tfinal=0.2k / 576)
        run!(f, fq; tfinal=0.2k / 576)
    end
    @test s.step == f.step == 576
    ps = PatchSolver(s, s.patches[2])
    pf = PatchSolver(f, f.patches[1])
    off = ps.patch.region.offset[1]
    e = maximum(abs(q.patches[2][padded_index(ps, i, 1, 1), c] -
                    fq[padded_index(pf, off + i, 1, 1), c])
                for i in 1:ps.decomp.n_local[1], c in 1:4)
    @info "r-z level against the uniform run" e
    @test e < RZ_LEVEL_TOL
    # Composite conservation on an annulus periodic in z, a pulse crossing
    # the level: the budget ledger's pieces telescope to the r-weighted
    # drift, the level's coarse-fine flux and the parent's flux into the
    # covered region cancel to the drift's order, and the drift measured
    # 8.7e-7.
    s = rz((37, 1, 36); bcs=(wall, per, per),
           refine=BlockRegion((12, 0, 12), (12, 1, 12)),
           art=ArtificialProperties(enabled=false), cfl=0.5, filter_interval=0)
    q = allocate_state(s)
    initialize!(s, q, (r, θ, z) -> begin
        g = exp(-40((r - 0.9)^2 + (z - 0.45)^2))
        Prim(rho=1.0 + 0.1g, u=(0.0, 0.0, 0.2), p=1.0 + 0.1g)
    end)
    b0 = CL._conserved_budget(s, q)
    ledger = CL._ledger_begin!(s, q)
    run!(s, q; tfinal=0.4, observer=ledger)
    r = CL._ledger_end!(ledger, s, q)
    b1 = CL._conserved_budget(s, q)
    drift = (b1.total_mass - b0.total_mass) / b0.total_mass
    @info "r-z composite mass drift" drift
    @test maximum(abs, r.residual) < 1e-13
    @test abs(r.pieces[(:coarse_fine_flux, 1)][1] +
              r.pieces[(:covered_face_flux, 0)][1]) < 1e-5 * b0.total_mass
    @test abs(drift) < 2e-5
end

# The refined patch at an NSCBC face against the uniform run at its spacing in
# equal steps, an acoustic pulse of amplitude 0.01 halfway out of the face,
# N = 49, over the conserved components: measured 1.4e-6 at the outflow and
# 2.7e-6 at the inflow under a moving target. The reflection is the largest
# |p − 1| left in the domain after the pulse has gone: 1.6053e-4 refined
# against 1.6056e-4 uniform at the fine spacing, the uniform coarse run
# reflecting 1.6652e-4.
const NSCBC_LEVEL_TOL = 5e-6

const NSCBC_REFLECTION_TOL = 1e-6

@testset "a level reaching an NSCBC face carries the characteristic correction" begin
    per = (PeriodicBC(), PeriodicBC())
    c = sqrt(1.4)
    # A pulse running toward the refined face on a stream at 0.3 with
    # ρ = p = 1: out through the outflow at x = 1, or, running left at
    # c − 0.3, out through the inflow at x = 0, whose target moves in time.
    function nscbc_level(N; refined=true, face=:outflow, uniform_coarse=false)
        m = (N - 1) ÷ 6 + 1
        region = BlockRegion((face === :outflow ? N - m : 0, 0, 0), (m, 1, 1))
        n = refined || uniform_coarse ? N : 3 * (N - 1) + 1
        target(x, y, z, t) = Prim(u=(0.3 * (1 + 0.05 * sinpi(5t)), 0.0, 0.0),
                                  T_ion=1.0, p=1.0)
        inflow = NSCBCInflowBC(u=(0.3, 0.0, 0.0), T_ion=1.0,
                               target=face === :inflow ? target : nothing)
        solver = build_solver(
            Problem(domain=((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)),
                    bcs=((inflow, NSCBCOutflowBC(pinf=1.0)), per, per), ic=nothing),
            Numerics(n_global=(n, 1, 1), cfl=0.9, art=ArtificialProperties(enabled=false),
                     filter=StateFilter(interval=0),
                     amr=AMR(initial=refined ? region : nothing, regrid_interval=0)))
        states = allocate_state(solver)
        dir = face === :outflow ? 1.0 : -1.0
        initialize!(solver, states, (x, y, z) -> begin
            a = 0.01 * exp(-((x - 0.5) / 0.1)^2)
            Prim(rho=1 + a / c^2, u=(0.3 + dir * a / c, 0.0, 0.0), p=1 + a)
        end)
        return solver, states
    end
    run_to!(s, q, t, steps) = for n in 1:steps
        run!(s, q; tfinal=t * n / steps)
    end
    # The largest |p − 1| over the nodes no child covers.
    function reflection(s, q)
        r = 0.0
        refresh_primitives!(s, q)
        for (ps, Q) in CL.eachpatch(s, q), i in 1:ps.decomp.n_local[1]
            I = padded_index(ps, i, 1, 1)
            ps.amr.covered[I] == 0 && (r = max(r, abs(ps.p[I] - 1)))
        end
        r
    end
    solver, _ = nscbc_level(49)
    fine = solver.patches[2]
    lt = getfield(solver, :levels)[2].transfers[1]
    # The outflow face carries the root's own condition object, with its
    # relaxation length read from the domain, not from the tile.
    @test fine.bcs[1][2] === solver.patches[1].bcs[1][2]
    @test CL.parent_fed(fine.bcs[1][1])
    @test lt.boundary[1] == (false, true) && lt.imposed[1] == (true, false)
    @test PatchSolver(solver, fine).L_domain == solver.L_domain
    # Against the uniform run at the level's spacing in the same equal steps,
    # halfway through the pulse's exit (the outflow's pulse moves at 1.48,
    # the inflow's at 0.88).
    for (face, t) in ((:outflow, 0.35), (:inflow, 0.6))
        s, q = nscbc_level(49; face=face)
        f, fq = nscbc_level(49; face=face, refined=false)
        steps = round(Int, 288 * t / 0.35)
        run_to!(s, q, t, steps)
        run_to!(f, fq, t, steps)
        @test s.step == f.step == steps
        pf = PatchSolver(f, f.patches[1])
        ps = PatchSolver(s, s.patches[2])
        off = ps.patch.region.offset[1]
        e = maximum(abs(q.patches[2][padded_index(ps, i, 1, 1), k] -
                        fq[padded_index(pf, off + i, 1, 1), k])
                    for i in 1:ps.decomp.n_local[1], k in 1:5)
        @info "NSCBC level against the uniform run" face e
        @test e < NSCBC_LEVEL_TOL
    end
    # The wave the outflow reflects, once the pulse has left, is the uniform
    # fine run's: the refined run and the uniform coarse one differ from it
    # by the refinement's error and by the coarse spacing's.
    s, q = nscbc_level(49)
    f, fq = nscbc_level(49; refined=false)
    cs, cq = nscbc_level(49; refined=false, uniform_coarse=true)
    run_to!(s, q, 0.8, 576)
    run_to!(f, fq, 0.8, 576)
    run_to!(cs, cq, 0.8, 192)
    r_level, r_fine, r_coarse = reflection(s, q), reflection(f, fq), reflection(cs, cq)
    @info "NSCBC reflection" r_level r_fine r_coarse
    @test abs(r_level - r_fine) < NSCBC_REFLECTION_TOL
    @test abs(r_coarse - r_fine) > 3 * NSCBC_REFLECTION_TOL
    # A face no level carries is refused by name.
    @test_throws "ExtrapolationBC" build_solver(
        Problem(domain=((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)),
                bcs=((SlipWallBC(), ExtrapolationBC()), per, per), ic=nothing),
        Numerics(n_global=(49, 1, 1),
                 amr=AMR(initial=BlockRegion((40, 0, 0), (9, 1, 1)), regrid_interval=0)))
end

# The tile at the outer face of the r-z annulus against the uniform run at its
# spacing, N = 49, t = 0.5, halfway through the pulse's exit: measured 6.6e-7
# over the five components and the whole tile. The same uniform run under the
# plane form of the outflow differs from it by 1.0e-3, so a tile that took
# the face's curvature term at a wrong radius, or not at all, fails here.
const RADIAL_NSCBC_LEVEL_TOL = 5e-6

@testset "a level at a radial NSCBC face carries the curvature term" begin
    per = (PeriodicBC(), PeriodicBC())
    c = sqrt(1.4)
    # An outgoing cylindrical pulse on the axisymmetric annulus r ∈ [1/2, 3/2],
    # gas at rest, leaving through the outflow at r = 3/2, with a level over
    # the outer N ÷ 6 + 1 root nodes; `refined = false` gives the uniform run
    # at the level's spacing, 3(N − 1) + 1 nodes.
    function radial_level(N; refined=true)
        m = (N - 1) ÷ 6 + 1
        solver = build_solver(
            Problem(domain=((0.5, 0.5 + 1.0), (0.0, 0.0 + 1.0), (0.0, 0.0 + 1.0)),
                    metric=CylindricalMetric(),
                    bcs=((SlipWallBC(), NSCBCOutflowBC(pinf=1.0)), per, per), ic=nothing),
            Numerics(n_global=(refined ? N : 3 * (N - 1) + 1, 1, 1), cfl=0.9,
                     art=ArtificialProperties(enabled=false),
                     filter=StateFilter(interval=0),
                     amr=AMR(initial=refined ? BlockRegion((N - m, 0, 0), (m, 1, 1)) : nothing,
                             regrid_interval=0)))
        states = allocate_state(solver)
        initialize!(solver, states, (x, y, z) -> begin
            a = 0.01 * exp(-((x - 1) / 0.1)^2)
            Prim(rho=1 + a / c^2, u=(a / c, 0.0, 0.0), p=1 + a)
        end)
        return solver, states
    end
    run_to!(s, q, t, steps) = for n in 1:steps
        run!(s, q; tfinal=t * n / steps)
    end
    s, q = radial_level(49)
    f, fq = radial_level(49; refined=false)
    run_to!(s, q, 0.5, 411)
    run_to!(f, fq, 0.5, 411)
    @test s.step == f.step == 411
    pf = PatchSolver(f, f.patches[1])
    ps = PatchSolver(s, s.patches[2])
    off = ps.patch.region.offset[1]
    e = maximum(abs(q.patches[2][padded_index(ps, i, 1, 1), k] -
                    fq[padded_index(pf, off + i, 1, 1), k])
                for i in 1:ps.decomp.n_local[1], k in 1:5)
    @info "radial NSCBC level against the uniform run" e
    @test e < RADIAL_NSCBC_LEVEL_TOL
end

# The axis tile against the uniform run at its spacing, a pulse converging on
# the axis, N = 96, t = 0.3: measured 1.6e-7 over the five components and the
# whole tile. The corner tile of the axis and a plane at z = 0, a spherical
# pulse, N = 36, t = 0.3: measured 2.0e-5. The axis tile under the :filter
# restriction: measured 7.3e-6, the attenuation the filter writes into the
# covered nodes at every step.
const AXIS_LEVEL_TOL = 5e-7

const CORNER_LEVEL_TOL = 5e-5

const AXIS_FILTER_TOL = 2e-5

@testset "a level reaching the r-z axis folds on that face" begin
    per = (PeriodicBC(), PeriodicBC())
    axis = (AxisBC(), SlipWallBC())
    zplane = (SymmetryPlaneBC(), SlipWallBC())
    # A pulse converging on the axis of the radial line r ∈ (0, 1], with a
    # level over the first N ÷ 6 + 1 root nodes; `refined = false` gives the
    # uniform run at the level's spacing, 3N − 1 nodes, whose node j is the
    # tile's node j. Filtered at every step: the axis converts the grid-scale
    # waves the coarse-fine face emits into smooth ones
    # (`axis_level_case` in smooth_cases.jl).
    function axis_level(N; refined=true, nest=nothing,level_restriction=:inject)
        region = BlockRegion((0, 0, 0), (N ÷ 6 + 1, 1, 1))
        refine = !refined ? nothing : nest === nothing ? region : [region, nest]
        solver = build_solver(
            Problem(domain=((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)), bcs=(axis, per, per),
                    metric=CylindricalMetric(), ic=nothing),
            Numerics(n_global=(refined ? N : 3N - 1, 1, 1), cfl=0.9,
                     art=ArtificialProperties(enabled=false),
                     filter=StateFilter(interval=1),
                     amr=AMR(initial=refine, regrid_interval=0,
                             level_restriction=level_restriction)))
        states = allocate_state(solver)
        initialize!(solver, states, (r, θ, z) -> begin
            rho = 1 + 0.05 * exp(-((r - 0.3) / 0.1)^2)
            Prim(rho=rho, u=(0.0, 0.0, 0.0), p=rho^1.4)
        end)
        return solver, states
    end
    N = 96
    s, q = axis_level(N)
    root = PatchSolver(s, s.patches[1])
    fine = PatchSolver(s, s.patches[2])
    lt = getfield(s, :levels)[2].transfers[1]
    # The axis face carries the root's condition and its fold, self-paired,
    # with the axis parities; the other face stays parent-fed.
    @test fine.bcs[1][1] === root.bcs[1][1]
    @test CL.parent_fed(fine.bcs[1][2])
    @test fine.folds[1] isa CL.FoldSpec && fine.folds[1].pair === nothing
    @test (fine.folds[1].lo, fine.folds[1].hi) == (true, false)
    @test fine.folds[1].sigvel == root.folds[1].sigvel == (-1, -1, 1)
    @test fine.folds[1].sigflux == root.folds[1].sigflux
    @test lt.boundary[1][1] && lt.folded[1][1] && !lt.imposed[1][1] && lt.imposed[1][2]
    # One fine node beyond the coincident lattice: 3m − 1 nodes, the first at
    # h/6, and the geometry of the uniform run at h/3 node for node.
    m = lt.region.extent[1]
    h = root.h[1]
    @test fine.decomp.n_global[1] == 3m - 1
    @test xcoord(fine, 1, 1) ≈ h / 6
    f, fq = axis_level(N; refined=false)
    pf = PatchSolver(f, f.patches[1])
    e = 0.0
    for (a, b) in zip((fine.inv_J, fine.area_d[1], fine.inv_r),
                      (pf.inv_J, pf.area_d[1], pf.inv_r)), i in 1:fine.decomp.n_local[1]
        I, J = padded_index(fine, i, 1, 1), padded_index(pf, i, 1, 1)
        e = max(e, abs(a[I] - b[J]) / abs(b[J]))
    end
    @test e < 1e-14
    @test CL._box_buffer(lt)[1] == (CL.LEVEL_BUFFER, CL.LEVEL_BUFFER)
    @test first(CL._restrict_window(lt)[1]) == 1
    # The composite quadrature: the root's node at h/2 is covered on both
    # halves of [0, h], and the tile's node at h/6 carries the axis edge
    # factor of its own spacing, as the root alone carries its own. The
    # cells meet without overlap at the coarse-fine face, so they depart
    # from the root alone by the axis's edge term on the linear r, −h²/24
    # at the root's spacing, taken at the tile's instead:
    # (h² − (h/3)²)/24 = h²/27. The junction's Ω on the tile's high face
    # takes −(h² − (h/3)²)/24 of the volume's integrand r, which cancels it.
    @test CL.uncovered_fraction(root.amr.covered[padded_index(root, 1, 1, 1)]) == 0
    @test CL.quad_weight(fine, 1, 1) == 1.0
    @test CL._edge_factor(fine, 1, 1, padded_index(fine, 1, 1, 1)) ≈ 11 / 12
    alone = build_solver(
        Problem(domain=((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)), bcs=(axis, per, per),
                metric=CylindricalMetric(), ic=nothing),
        Numerics(n_global=(N, 1, 1)))
    @test domain_volume(s) ≈ domain_volume(alone) atol = 1e-15
    # Against the uniform run at the level's spacing in the same equal steps,
    # the tile differs by the refinement's error alone.
    n = 3N
    for k in 1:n
        run!(s, q; tfinal=0.3k / n)
        run!(f, fq; tfinal=0.3k / n)
    end
    @test s.step == f.step == n
    # The box keeps its buffer across the axis, the mirror of the parent's
    # nodes: density even, radial momentum odd.
    box = lt.box_gather
    pad = lt.pdecomps[1].n_halo_d[1]
    B = CL.LEVEL_BUFFER
    @test any(!iszero, view(box, :, 1, 1, 2))
    for j in 1:B
        @test box[B + 1 - j + pad, 1, 1, 1] == box[B + j + pad, 1, 1, 1]
        @test box[B + 1 - j + pad, 1, 1, 2] == -box[B + j + pad, 1, 1, 2]
    end
    e = maximum(abs(q.patches[2][padded_index(fine, i, 1, 1), c] -
                    fq[padded_index(pf, i, 1, 1), c])
                for i in 1:fine.decomp.n_local[1], c in 1:5)
    @info "axis level against the uniform run" e
    @test e < AXIS_LEVEL_TOL
    # The :filter restriction folds its filter at the axis with each
    # component's parity; its attenuation of the covered nodes, not the axis,
    # sets the difference from the uniform run.
    sf, qf = axis_level(N; level_restriction=:filter)
    for k in 1:n
        run!(sf, qf; tfinal=0.3k / n)
    end
    @test sf.step == n
    finef = PatchSolver(sf, sf.patches[2])
    ef = maximum(abs(qf.patches[2][padded_index(finef, i, 1, 1), c] -
                     fq[padded_index(pf, i, 1, 1), c])
                 for i in 1:finef.decomp.n_local[1], c in 1:5)
    @info "axis level under :filter against the uniform run" ef
    @test ef < AXIS_FILTER_TOL
    # A second level stopping short of the axis is told the offset reaching
    # it.
    @test_throws "offset -1 reaches the AxisBC" axis_level(N;
        nest=BlockRegion((0, 0, 0), (10, 1, 1)))

    # The corner of the axis and a symmetry plane at z = 0: the tile folds on
    # both dimensions, with each root fold's parities.
    function corner(N; refined=true,transport=ConstantTransport(), filter_interval=1,
                                    art=ArtificialProperties(), cfl=0.5)
        n = refined ? N : 3N - 1
        solver = build_solver(
            Problem(domain=((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)), bcs=(axis, per, zplane),
                    metric=CylindricalMetric(), transport=transport, ic=nothing),
            Numerics(n_global=(n, 1, n), art=art, cfl=cfl,
                     filter=StateFilter(interval=filter_interval),
                     amr=AMR(initial=refined ?
                                     BlockRegion((0, 0, 0), (N ÷ 6 + 1, 1, N ÷ 6 + 1)) :
                                     nothing,
                             regrid_interval=0)))
        states = allocate_state(solver)
        initialize!(solver, states, (r, θ, z) -> begin
            rho = 1 + 0.05 * exp(-((sqrt(r^2 + z^2) - 0.4) / 0.15)^2)
            Prim(rho=rho, u=(0.0, 0.0, 0.0), p=rho^1.4)
        end)
        return solver, states
    end
    N = 36
    s, q = corner(N)
    root = PatchSolver(s, s.patches[1])
    fine = PatchSolver(s, s.patches[2])
    lt = getfield(s, :levels)[2].transfers[1]
    @test lt.folded == ((true, false), (false, false), (true, false))
    @test fine.folds[3].sigvel == root.folds[3].sigvel == (1, 1, -1)
    @test fine.folds[1].sigvel == root.folds[1].sigvel
    @test xcoord(fine, 1, 1) ≈ root.h[1] / 6 && xcoord(fine, 3, 1) ≈ root.h[3] / 6
    # The axis's edge term at the tile's spacing and the junction's Ω on the
    # tile's r face cancel per unit of z, as on the axis tile above, and the
    # z face's Ω takes nothing from an integrand r constant along z.
    alone = build_solver(
        Problem(domain=((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)), bcs=(axis, per, zplane),
                metric=CylindricalMetric(), ic=nothing),
        Numerics(n_global=(N, 1, N)))
    @test domain_volume(s) ≈ domain_volume(alone) atol = 1e-15
    # A uniform state at rest stays at rest to round-off through both folds,
    # the viscous ghost fluxes, the artificial properties, the d8 detector and
    # the filter.
    rest, rq = corner(N; transport=ConstantTransport(mu0=1e-3), filter_interval=1,
                      art=ArtificialProperties(enabled=true, detector=:d8))
    initialize!(rest, rq, (r, θ, z) -> Prim(rho=1.3, u=(0.0, 0.0, 0.0), p=0.9))
    q0 = [copy(parent(x)) for x in rq.patches]
    run!(rest, rq; tfinal=1.0, nmax=10)
    drift = 0.0
    for (i, p) in enumerate(getfield(rest, :patches))
        pp = PatchSolver(rest, p)
        nl = pp.decomp.n_local
        for c in 1:5, k in 1:nl[3], ii in 1:nl[1]
            I = padded_index(pp, ii, 1, k)
            drift = max(drift, abs(rq.patches[i][I, c] - q0[i][I, c]))
        end
    end
    @test rest.step == 10
    @test drift < 1e-12
    # Against the uniform run at the level's spacing in the same equal steps.
    s, q = corner(N; filter_interval=1, art=ArtificialProperties(enabled=false), cfl=0.9)
    f, fq = corner(N; refined=false, filter_interval=1,
                   art=ArtificialProperties(enabled=false), cfl=0.9)
    n = 4N
    for k in 1:n
        run!(s, q; tfinal=0.3k / n)
        run!(f, fq; tfinal=0.3k / n)
    end
    @test s.step == f.step == n
    fine = PatchSolver(s, s.patches[2])
    pf = PatchSolver(f, f.patches[1])
    nl = fine.decomp.n_local
    e = maximum(abs(q.patches[2][padded_index(fine, i, 1, k), c] -
                    fq[padded_index(pf, i, 1, k), c])
                for i in 1:nl[1], k in 1:nl[3], c in 1:5)
    @info "corner level against the uniform run" e
    @test e < CORNER_LEVEL_TOL
    # The composite budget of the closed corner run: the ledger's pieces
    # telescope to the r-weighted drift of the mass.
    s, q = corner(N; art=ArtificialProperties(enabled=false))
    b0 = CL._conserved_budget(s, q)
    ledger = CL._ledger_begin!(s, q)
    run!(s, q; tfinal=0.2, observer=ledger)
    r = CL._ledger_end!(ledger, s, q)
    b1 = CL._conserved_budget(s, q)
    drift = (b1.total_mass - b0.total_mass) / b0.total_mass
    @info "corner composite mass drift" drift
    @test maximum(abs, r.residual) < 1e-13
    @test abs(drift) < 5e-6
end

# A level placed on the high wall of the 1-D standing wave by a regrid at
# step 0, against the uniform run at its spacing in equal steps at N = 49,
# t = 0.2, over the conserved components: measured 4.0e-8 for one box and
# 1.5e-7 for two tiles (the tile face between them adds its error). A pulse
# followed by tiles out through an NSCBC outflow, against the uniform run at
# t = 0.35, halfway through the exit: 5.8e-7.
const PLACED_WALL_TOL = 5e-7

const PLACED_NSCBC_TOL = 5e-6

@testset "level_boundaries places a regridded level on a wall or NSCBC face" begin
    per = (PeriodicBC(), PeriodicBC())
    walls = (SlipWallBC(), SlipWallBC())
    N = 49
    wave(x, y, z) = (rho = 1 + 0.05 * cos(pi * x);
                     Prim(rho=rho, u=(0.05 * sin(pi * x), 0.0, 0.0), p=rho^1.4))
    # The window the predicate tags, moved between regrid checks.
    cut = Ref(0.85)
    upper = Ref(2.0)
    near(p, I) = cut[] < xcoord(p, 1, interior_index(p, I)[1]) < upper[]
    function placed(; refined=true, tile=0, lb=true, bcs=(walls, per, per),
                    region=BlockRegion((16, 0, 0), (9, 1, 1)), ic=wave,
                    level_interpolation_order=nothing, max_levels=nothing)
        local n = refined ? N : 3 * (N - 1) + 1
        local s = build_solver(
            Problem(domain=((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)), bcs=bcs, ic=nothing),
            Numerics(n_global=(n, 1, 1), cfl=0.9, art=ArtificialProperties(enabled=false),
                     filter=StateFilter(interval=0),
                     amr=AMR(initial=refined ? region : nothing,
                             level_interpolation_order=level_interpolation_order,
                             regrid_interval=refined ? 10^6 : 0, tag_threshold=Inf,
                             tag_buffer=2, tile=tile, max_levels=max_levels,
                             level_boundaries=refined && lb));
            tag_predicate=refined ? near : nothing)
        local q = allocate_state(s)
        initialize!(s, q, ic)
        return s, q
    end
    function regrid_now!(s, q)
        getfield(s, :regrid).checks += 1
        return CL.regrid!(s, q, CL.Workspace(q), nothing)
    end
    # The largest difference of the level's patches from the uniform run at
    # their spacing, over the conserved components.
    function against_uniform(s, q, f, fq)
        local pf = PatchSolver(f, f.patches[1])
        local e = 0.0
        for k in 2:length(s.patches)
            local ps = PatchSolver(s, s.patches[k])
            local off = ps.patch.region.offset[1]
            e = max(e, maximum(abs(q.patches[k][padded_index(ps, i, 1, 1), c] -
                                   fq[padded_index(pf, off + i, 1, 1), c])
                               for i in 1:ps.decomp.n_local[1], c in axes(q.patches[k], 4)))
        end
        return e
    end
    steps!(s, q, k0, k1) = for k in k0:k1
        run!(s, q; tfinal=0.2k / 96)
    end
    margin = CL.LEVEL_BUFFER
    reaches(r) = r.offset[1] + r.extent[1] == N
    # The placement is on by default; without regridding the keyword
    # leaves an explicit region as it is.
    s = build_solver(
        Problem(domain=((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)), bcs=(walls, per, per),
                ic=nothing),
        Numerics(n_global=(N, 1, 1),
                 amr=AMR(initial=BlockRegion((16, 0, 0), (9, 1, 1)), regrid_interval=1)))
    @test getfield(s, :regrid).boundaries
    s = build_solver(
        Problem(domain=((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)), bcs=(walls, per, per),
                ic=nothing),
        Numerics(n_global=(N, 1, 1),
                 amr=AMR(initial=BlockRegion((16, 0, 0), (9, 1, 1)), regrid_interval=0,
                         level_boundaries=true)))
    @test only(level_regions(s, 1)) == BlockRegion((16, 0, 0), (9, 1, 1))

    # --- The tag clamp of a box, off and on ----------------------------------
    cut[] = 0.85
    s, q = placed(lb=false)
    r = @test_logs (:warn, r"does not reach") CL.tagged_region(s, q.patches[1])
    @test r.offset[1] + r.extent[1] == N - margin
    s, q = placed()
    r = @test_logs CL.tagged_region(s, q.patches[1])
    @test r == BlockRegion((39, 0, 0), (10, 1, 1))
    # A narrow tag at the face widens until the box, which takes no buffer
    # beyond the face, spans the interpolation order.
    cut[] = 0.99
    s10, q10 = placed(level_interpolation_order=10)
    @test CL.tagged_region(s10, q10.patches[1]) == BlockRegion((N - 6, 0, 0), (6, 1, 1))
    # A buffered end inside the margin band moves onto the face, where a
    # parent-fed face would have no room for its box.
    cut[], upper[] = 0.8, 0.93
    @test CL.tagged_region(s, q.patches[1]) == BlockRegion((37, 0, 0), (12, 1, 1))
    upper[] = 2.0

    # --- A box regridded onto the wall, against the uniform fine run ---------
    cut[] = 0.85
    s, q = placed()
    @test regrid_now!(s, q)
    lt = getfield(s, :levels)[2].transfers[1]
    @test reaches(lt.region) && lt.boundary[1] == (false, true)
    @test s.patches[2].bcs[1][2] === s.patches[1].bcs[1][2]
    f, fq = placed(refined=false)
    steps!(s, q, 1, 96)
    steps!(f, fq, 1, 96)
    @test s.step == f.step == 96
    e = against_uniform(s, q, f, fq)
    @info "box regridded onto the wall against the uniform run" e
    @test e < PLACED_WALL_TOL
    # The box grows along the wall: the carry keeps the wall plane, the
    # level's own solution, bit for bit with the rest of the overlap.
    old = copy(parent(q.patches[2]))
    pold = PatchSolver(s, s.patches[2])
    nold = pold.decomp.n_local[1]
    cut[] = 0.7
    @test regrid_now!(s, q)
    pnew = PatchSolver(s, s.patches[2])
    shift = pnew.decomp.n_local[1] - nold
    @test shift > 0 && reaches(level_regions(s, 1)[1])
    @test all(q.patches[2][padded_index(pnew, i + shift, 1, 1), c] ==
              old[padded_index(pold, i, 1, 1), c] for i in 2:nold, c in axes(old, 4))

    # --- Tiles, off and on, and a restart --------------------------------------
    cut[] = 0.85
    s, q = placed(tile=8, lb=false)
    @test_logs (:warn, r"does not reach") match_mode=:any regrid_now!(s, q)
    @test level_regions(s, 1)[end].offset[1] + level_regions(s, 1)[end].extent[1] ==
          N - margin
    s, q = placed(tile=8)
    @test regrid_now!(s, q)
    regs = level_regions(s, 1)
    @test regs == [BlockRegion((32, 0, 0), (9, 1, 1)), BlockRegion((40, 0, 0), (9, 1, 1))]
    @test [lt.boundary[1] for lt in getfield(s, :levels)[2].transfers] ==
          [(false, false), (false, true)]
    f, fq = placed(refined=false)
    steps!(s, q, 1, 48)
    dir = mktempdir()
    save_checkpoint(s, q, joinpath(dir, "walltiles"))
    steps!(s, q, 49, 96)
    steps!(f, fq, 1, 96)
    e = against_uniform(s, q, f, fq)
    @info "tiles regridded onto the wall against the uniform run" e
    @test e < PLACED_WALL_TOL
    r, rq = placed(tile=8)
    load_checkpoint!(r, rq, joinpath(dir, "walltiles"))
    @test level_regions(r, 1) == regs
    steps!(r, rq, 49, 96)
    @test r.t == s.t && length(rq.patches) == length(q.patches) &&
          all(parent(rq.patches[k])[CL.interior(s.patches[k].decomp), :] ==
              parent(q.patches[k])[CL.interior(s.patches[k].decomp), :] for k in eachindex(q.patches))
    rm(dir; recursive=true)

    # --- Three regridded levels reach the wall, and the tag sweep of a tile
    # clamps at its wall face -------------------------------------------------
    for lb in (false, true)
        cut[] = 0.85
        s, q = placed(tile=8, lb=lb, max_levels=3, region=nothing)
        regrid_now!(s, q)
        N2 = 3 * (N - 1) + 1
        r2 = level_regions(s, 2)
        @test any(r -> r.offset[1] + r.extent[1] == N2, r2) == lb
        lb && @test all(lt.boundary[1] == (false, lt.region.offset[1] +
                                                   lt.region.extent[1] == N2)
                        for lt in getfield(s, :levels)[3].transfers)
        steps!(s, q, 1, 4)
        @test all(all(isfinite, parent(Q)) for Q in q.patches)
    end
    # A uniform state tags nothing on a wall tile: the δ⁴ taps stop at the
    # wall, where the tile's ghost layers hold no data of its own.
    cut[] = 0.85
    s, q = placed(tile=8, ic=(x, y, z) -> Prim(rho=1.0, u=(0.0, 0.0, 0.0), p=1.0))
    regrid_now!(s, q)
    getfield(s, :regrid).threshold = 0.02
    cut[] = 2.0
    k = only(k for (k, lt) in zip(getfield(s, :levels)[2].patches,
                                  getfield(s, :levels)[2].transfers) if lt.boundary[1][2])
    p = s.patches[k]
    marks = Ref(0)
    CL._tag_sweep!((g1, g2, g3, level) -> (marks[] += 1), s, p, q.patches[k],
                   zeros(Int8, size(p.amr.covered)), false)
    @test marks[] == 0

    # --- A pulse leaving through an NSCBC outflow ---------------------------------
    c = sqrt(1.4)
    xc(t) = 0.5 + (0.3 + c) * t
    follow(p, I) = abs(xcoord(p, 1, interior_index(p, I)[1]) - xc(p.t)) < 0.12
    open_bcs = ((NSCBCInflowBC(u=(0.3, 0.0, 0.0), T_ion=1.0), NSCBCOutflowBC(pinf=1.0)),
                per, per)
    pulse(x, y, z) = (a = 0.01 * exp(-((x - 0.5) / 0.1)^2);
                      Prim(rho=1 + a / c^2, u=(0.3 + a / c, 0.0, 0.0), p=1 + a))
    function pulse_run(; refined=true, lb=true, coarse=false)
        local n = refined || coarse ? N : 3 * (N - 1) + 1
        local s = build_solver(
            Problem(domain=((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)), bcs=open_bcs, ic=nothing),
            Numerics(n_global=(n, 1, 1), cfl=0.9, art=ArtificialProperties(enabled=false),
                     filter=StateFilter(interval=0),
                     amr=AMR(initial=refined ? BlockRegion((16, 0, 0), (17, 1, 1)) : nothing,
                             regrid_interval=refined ? 4 : 0, tag_threshold=Inf,
                             tag_buffer=2, tile=8, level_boundaries=refined && lb));
            tag_predicate=refined ? follow : nothing)
        local q = allocate_state(s)
        initialize!(s, q, pulse)
        return s, q
    end
    function reflection(s, q)
        local r = 0.0
        refresh_primitives!(s, q)
        for (ps, Q) in CL.eachpatch(s, q), i in 1:ps.decomp.n_local[1]
            local I = padded_index(ps, i, 1, 1)
            ps.amr.covered[I] == 0 && (r = max(r, abs(ps.p[I] - 1)))
        end
        return r
    end
    run_to!(s, q, t, steps) = for n in 1:steps
        run!(s, q; tfinal=t * n / steps)
    end
    # Off, the tiles following the pulse stop at the margin as it nears the
    # face (t = 0.25, the pulse centred at 0.87).
    s, q = pulse_run(lb=false)
    @test_logs (:warn, r"does not reach") match_mode=:any run_to!(s, q, 0.25, 180)
    @test all(!reaches(r) for r in level_regions(s, 1))
    # On, a tile reaches the face and carries the outflow. Against the
    # uniform run at the fine spacing in equal steps, halfway through the
    # exit (t = 0.35, step 252), and in the reflection once the pulse has
    # gone (t = 0.8), which the uniform coarse run misses.
    s, q = pulse_run()
    f, fq = pulse_run(refined=false)
    cs, cq = pulse_run(refined=false, coarse=true)
    e = Inf
    for n in 1:576
        run!(s, q; tfinal=0.8n / 576)
        run!(f, fq; tfinal=0.8n / 576)
        n == 252 || continue
        @test any(reaches, level_regions(s, 1))
        @test s.patches[end].bcs[1][2] === s.patches[1].bcs[1][2]
        e = against_uniform(s, q, f, fq)
    end
    run_to!(cs, cq, 0.8, 192)
    r_level, r_fine, r_coarse = reflection(s, q), reflection(f, fq), reflection(cs, cq)
    @info "pulse through a placed outflow level" e r_level r_fine r_coarse
    @test e < PLACED_NSCBC_TOL
    @test abs(r_level - r_fine) < NSCBC_REFLECTION_TOL
    @test abs(r_coarse - r_fine) > 3 * NSCBC_REFLECTION_TOL

    # --- The frontend's shapes -------------------------------------------------
    domain = ((0.0, 1.0), (0.0, 1.0), (0.0, 1.0))
    shape_run(bcs, shape; metric=CartesianMetric(), kw...) =
        setup(Problem(domain=domain, bcs=bcs, metric=metric,
                      ic=(x, y, z) -> wave(x, y, z)),
              Numerics(n_global=(N, 1, 1), art=ArtificialProperties(enabled=false),
                       amr=AMR(; initial=shape, kw...)))
    toward_wall = CL.Regions.Box((0.8, 0.0, 0.0), (1.0, 1.0, 1.0))
    s, _ = @test_logs (:warn, r"domain boundary") match_mode=:any shape_run(
        (walls, per, per), toward_wall; level_boundaries=false)
    @test refined_region(s).offset[1] + refined_region(s).extent[1] == N - margin
    s, _ = @test_logs min_level=Base.CoreLogging.Warn shape_run(
        (walls, per, per), toward_wall)
    @test reaches(refined_region(s))
    # A static shape's first level reaches a symmetry plane or the r-z axis,
    # which it folds; with the keyword off it keeps the margin and says so.
    toward_fold = CL.Regions.Box((0.0, 0.0, 0.0), (0.2, 1.0, 1.0))
    for (bcs, metric) in ((((SymmetryPlaneBC(), SlipWallBC()), per, per), CartesianMetric()),
                          (((AxisBC(), SlipWallBC()), per, per), CylindricalMetric()))
        s, _ = shape_run(bcs, toward_fold; metric)
        @test refined_region(s).offset[1] == 0 &&
              getfield(s, :levels)[2].transfers[1].folded[1] == (true, false)
        s, _ = @test_logs (:warn, r"domain boundary") match_mode=:any shape_run(
            bcs, toward_fold; metric, level_boundaries=false)
        @test refined_region(s).offset[1] == margin
    end
end

@testset "level_boundaries keeps the margin at a face a level cannot carry" begin
    per = (PeriodicBC(), PeriodicBC())
    N = 49
    margin = CL.LEVEL_BUFFER
    wave(x, y, z) = (rho = 1 + 0.05 * cos(pi * x);
                     Prim(rho=rho, u=(0.05 * sin(pi * x), 0.0, 0.0), p=rho^1.4))
    high(p, I) = xcoord(p, 1, interior_index(p, I)[1]) > 0.85
    low(p, I) = xcoord(p, 1, interior_index(p, I)[1]) < 0.15
    function tagged(bcs, tag; tile=0, level_restriction=:inject)
        local s = build_solver(
            Problem(domain=((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)), bcs=bcs, ic=nothing),
            Numerics(n_global=(N, 1, 1), art=ArtificialProperties(enabled=false),
                     filter=StateFilter(interval=0),
                     amr=AMR(initial=BlockRegion((16, 0, 0), (9, 1, 1)),
                             level_restriction=level_restriction, regrid_interval=10^6,
                             tag_threshold=Inf, tag_buffer=2, tile=tile)); tag_predicate=tag)
        local q = allocate_state(s)
        initialize!(s, q, wave)
        return s, q
    end
    # A face no level carries: the box, the tiles and a shape stop at the
    # margin, and each path reports it by one warning instead of failing setup.
    kept = ((SlipWallBC(), ExtrapolationBC()), per, per)
    s, q = tagged(kept, high)
    r = @test_logs (:warn, r"does not reach") CL.tagged_region(s, q.patches[1])
    @test r.offset[1] + r.extent[1] == N - margin
    s, q = tagged(kept, high; tile=8)
    getfield(s, :regrid).checks += 1
    @test_logs (:warn, r"does not reach") match_mode=:any CL.regrid!(s, q, CL.Workspace(q),
                                                                      nothing)
    @test !isempty(level_regions(s, 1))
    @test all(r -> r.offset[1] + r.extent[1] <= N - margin, level_regions(s, 1))
    s, _ = @test_logs (:warn, r"domain boundary") match_mode=:any setup(
        Problem(domain=((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)), bcs=kept, ic=wave),
        Numerics(n_global=(N, 1, 1), art=ArtificialProperties(enabled=false),
                 amr=AMR(initial=CL.Regions.Box((0.8, 0.0, 0.0), (1.0, 1.0, 1.0)))))
    @test refined_region(s).offset[1] + refined_region(s).extent[1] == N - margin
    # A symmetry plane is reached under either restriction.
    plane = ((SymmetryPlaneBC(), SlipWallBC()), per, per)
    for mode in (:inject, :filter)
        s, q = tagged(plane, low; level_restriction=mode)
        @test CL.tagged_region(s, q.patches[1]).offset[1] == 0
    end
end

# A level regridded onto the fold of the 1-D standing wave at step 0, against
# the uniform run at its spacing in equal steps at N = 49, t = 0.2, over the
# conserved components: at a symmetry plane, measured 2.9e-8 for one box and
# 1.8e-7 for two tiles (the tile face between them adds its error); at the
# r-z axis, filtered at every step, 3.6e-11 and 5.7e-11. A second level at
# either plane of the standing wave, N = 48, t = 0.1, against the uniform run
# at its spacing: 4.5e-13 and 2.3e-13.
const FOLD_REGRID_TOL = 5e-7

const FOLD_NEST_TOL = 5e-12

@testset "levels regridded onto a fold and nested at one" begin
    per = (PeriodicBC(), PeriodicBC())
    plane = (SymmetryPlaneBC(), SlipWallBC())
    axis = (AxisBC(), SlipWallBC())
    wave(x, y, z) = (rho = 1 + 0.05 * cos(pi * x);
                     Prim(rho=rho, u=(0.05 * sin(pi * x), 0.0, 0.0), p=rho^1.4))
    N = 49
    # The window the predicate tags, moved between regrid checks.
    cut = Ref(0.15)
    near(p, I) = xcoord(p, 1, interior_index(p, I)[1]) < cut[]
    # `refined = false` gives the uniform run at the level's spacing: a fold at
    # the low face and a wall at the high one put the root's node 1 at h/2
    # with h = 1/(N − ½), so the level's spacing is that of 3N − 1 nodes.
    function placed(fold; refined=true, tile=0, lb=true,
                    region=BlockRegion((16, 0, 0), (9, 1, 1)), max_levels=nothing)
        cyl = fold === axis
        local s = build_solver(
            Problem(domain=((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)), bcs=(fold, per, per),
                    metric=cyl ? CylindricalMetric() : CartesianMetric(), ic=nothing),
            Numerics(n_global=(refined ? N : 3N - 1, 1, 1), cfl=0.9,
                     art=ArtificialProperties(enabled=false),
                     filter=StateFilter(interval=cyl ? 1 : 0),
                     amr=AMR(initial=refined ? region : nothing,
                             regrid_interval=refined ? 10^6 : 0, tag_threshold=Inf,
                             tag_buffer=2, tile=tile, max_levels=max_levels,
                             level_boundaries=refined && lb));
            tag_predicate=refined ? near : nothing)
        local q = allocate_state(s)
        initialize!(s, q, wave)
        return s, q
    end
    function regrid_now!(s, q)
        getfield(s, :regrid).checks += 1
        return CL.regrid!(s, q, CL.Workspace(q), nothing)
    end
    # The largest difference of the patches of level ℓ from the uniform run
    # at their spacing: level-ℓ patch node i is the uniform run's node
    # i + offset + (3^ℓ − 1)/2, the uniform run starting half its spacing
    # from the fold.
    function against_uniform(s, q, f, fq, ℓ=1)
        local pf = PatchSolver(f, f.patches[1])
        local e = 0.0
        for k in getfield(s, :levels)[ℓ + 1].patches
            local ps = PatchSolver(s, s.patches[k])
            local off = ps.patch.region.offset[1] + (3^ℓ - 1) ÷ 2
            e = max(e, maximum(abs(q.patches[k][padded_index(ps, i, 1, 1), c] -
                                   fq[padded_index(pf, off + i, 1, 1), c])
                               for i in 1:ps.decomp.n_local[1], c in axes(q.patches[k], 4)))
        end
        return e
    end
    steps!(s, q, k0, k1) = for k in k0:k1
        run!(s, q; tfinal=0.2k / 96)
    end
    interior_equal(r, rq, s, q) =
        r.t == s.t && length(rq.patches) == length(q.patches) &&
        all(parent(rq.patches[k])[CL.interior(s.patches[k].decomp), :] ==
            parent(q.patches[k])[CL.interior(s.patches[k].decomp), :] for k in eachindex(q.patches))
    margin = CL.LEVEL_BUFFER

    for fold in (plane, axis)
        name = fold === axis ? "axis" : "plane"
        # --- A box follows the tags onto the fold, off and on ------------------
        cut[] = 0.15
        s, q = placed(fold; lb=false)
        r = @test_logs (:warn, r"does not reach") CL.tagged_region(s, q.patches[1])
        @test r.offset[1] == margin
        s, q = placed(fold)
        @test CL.tagged_region(s, q.patches[1]).offset[1] == 0
        @test regrid_now!(s, q)
        lt = getfield(s, :levels)[2].transfers[1]
        @test lt.region.offset[1] == 0 && lt.folded[1] == (true, false)
        fine = PatchSolver(s, s.patches[2])
        @test fine.bcs[1][1] === s.patches[1].bcs[1][1] && fine.folds[1] isa CL.FoldSpec
        @test fine.decomp.n_global[1] == 3 * lt.region.extent[1] - 1
        @test xcoord(fine, 1, 1) ≈ s.patches[1].h[1] / 6
        f, fq = placed(fold; refined=false)
        steps!(s, q, 1, 96)
        steps!(f, fq, 1, 96)
        @test s.step == f.step == 96
        e = against_uniform(s, q, f, fq)
        @info "box regridded onto the $name against the uniform run" e
        @test e < FOLD_REGRID_TOL
        # The box grows away from the fold: the carry keeps the node beyond
        # the coincident lattice and the fold's plane, bit for bit with the
        # rest of the overlap.
        old = copy(parent(q.patches[2]))
        pold = PatchSolver(s, s.patches[2])
        nold = pold.decomp.n_local[1]
        cut[] = 0.3
        @test regrid_now!(s, q)
        pnew = PatchSolver(s, s.patches[2])
        @test pnew.decomp.n_local[1] > nold && level_regions(s, 1)[1].offset[1] == 0
        @test all(q.patches[2][padded_index(pnew, i, 1, 1), c] == old[padded_index(pold, i, 1, 1), c]
                  for i in 1:nold-1, c in axes(old, 4))

        # --- Tiles on the fold, and a restart ------------------------------------
        cut[] = 0.3
        s, q = placed(fold; tile=8)
        @test regrid_now!(s, q)
        regs = level_regions(s, 1)
        @test length(regs) == 2 && regs[1].offset[1] == 0
        @test [lt.folded[1] for lt in getfield(s, :levels)[2].transfers] ==
              [(r.offset[1] == 0, false) for r in regs]
        f, fq = placed(fold; refined=false)
        steps!(s, q, 1, 48)
        dir = mktempdir()
        save_checkpoint(s, q, joinpath(dir, "foldtiles"))
        steps!(s, q, 49, 96)
        steps!(f, fq, 1, 96)
        e = against_uniform(s, q, f, fq)
        @info "tiles regridded onto the $name against the uniform run" e
        @test e < FOLD_REGRID_TOL
        r, rq = placed(fold; tile=8)
        load_checkpoint!(r, rq, joinpath(dir, "foldtiles"))
        @test level_regions(r, 1) == regs
        steps!(r, rq, 49, 96)
        @test interior_equal(r, rq, s, q)
        rm(dir; recursive=true)

        # --- Three regridded levels, the second reaching the fold at offset −1,
        # and a restart ---------------------------------------------------------
        cut[] = 0.3
        s, q = placed(fold; tile=8, max_levels=3, region=nothing)
        regrid_now!(s, q)
        r2 = level_regions(s, 2)
        @test any(r -> r.offset[1] == -1, r2)
        lts = getfield(s, :levels)[3].transfers
        @test all(lt.folded[1] == (lt.region.offset[1] == -1, false) for lt in lts)
        @test all(p -> p.level != 2 ||
                       CL._fold_active(p.folds[1]) == (p.region.offset[1] == -4),
                  s.patches)
        steps!(s, q, 1, 4)
        dir = mktempdir()
        save_checkpoint(s, q, joinpath(dir, "foldnest"))
        steps!(s, q, 5, 8)
        @test all(all(isfinite, parent(Q)) for Q in q.patches)
        r, rq = placed(fold; tile=8, max_levels=3, region=nothing)
        load_checkpoint!(r, rq, joinpath(dir, "foldnest"))
        @test level_regions(r, 2) == r2
        steps!(r, rq, 5, 8)
        @test interior_equal(r, rq, s, q)
        rm(dir; recursive=true)
    end

    # --- The frontend's shapes: a regridded shape's level reaches the fold,
    # and nested static shapes reach it on both levels ------------------------
    domain = ((0.0, 1.0), (0.0, 1.0), (0.0, 1.0))
    shape_run(bcs, shape; metric=CartesianMetric(), kw...) =
        setup(Problem(domain=domain, bcs=bcs, metric=metric,
                      ic=(x, y, z) -> wave(x, y, z)),
              Numerics(n_global=(N, 1, 1), art=ArtificialProperties(enabled=false),
                       amr=AMR(; initial=shape, kw...)))
    toward_fold = CL.Regions.Box((0.0, 0.0, 0.0), (0.2, 1.0, 1.0))
    inner = CL.Regions.Box((0.0, 0.0, 0.0), (0.08, 1.0, 1.0))
    for (fold, metric) in ((plane, CartesianMetric()), (axis, CylindricalMetric()))
        s, _ = shape_run((fold, per, per), toward_fold; metric, level_boundaries=true,
                         regrid_interval=5)
        @test refined_region(s).offset[1] == 0 &&
              getfield(s, :levels)[2].transfers[1].folded[1] == (true, false)
        s, _ = shape_run((fold, per, per), [toward_fold, inner]; metric,
                         level_boundaries=true)
        lt = getfield(s, :levels)[3].transfers[1]
        @test lt.region.offset[1] == -1 && lt.folded[1] == (true, false)
        @test xcoord(PatchSolver(s, s.patches[3]), 1, 1) ≈ s.patches[1].h[1] / 18
    end
end

@testset "a second level nested at a fold" begin
    per = (PeriodicBC(), PeriodicBC())
    plane = SymmetryPlaneBC()
    wave(x, y, z) = (rho = 1 + 0.05 * cos(pi * x);
                     Prim(rho=rho, u=(0.05 * sin(pi * x), 0.0, 0.0), p=rho^1.4))
    # A plane at both ends of the standing wave, N = 48: h = 1/N, root node i
    # at (i − ½)h, and the uniform run at the second level's spacing has 9N
    # nodes, the first at h/18. Level-1 node n lies at h/2 + (n − 1)h/3, so
    # its node 0 lies at h/6, and a second level reaching the low plane starts
    # there, at offset −1 of level 1's node space, and at the high plane ends
    # one node past the last node coincident with the root's, 3(N − 1) + 1.
    N = 48
    function nested(side; refined=true,)
        E1 = 3 * (N - 1) + 1
        r1 = side == :lo ? BlockRegion((0, 0, 0), (9, 1, 1)) :
                           BlockRegion((N - 9, 0, 0), (9, 1, 1))
        r2 = side == :lo ? BlockRegion((-1, 0, 0), (12, 1, 1)) :
                           BlockRegion((E1 + 1 - 12, 0, 0), (12, 1, 1))
        local s = build_solver(
            Problem(domain=((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)),
                    bcs=((plane, plane), per, per), ic=nothing),
            Numerics(n_global=(refined ? N : 9N, 1, 1), cfl=0.9,
                     art=ArtificialProperties(enabled=false),
                     filter=StateFilter(interval=0),
                     amr=AMR(initial=refined ? [r1, r2] : nothing, regrid_interval=0)))
        local q = allocate_state(s)
        initialize!(s, q, wave)
        return s, q
    end
    f, fq = nested(:lo; refined=false)
    pf = PatchSolver(f, f.patches[1])
    for n in 1:96
        run!(f, fq; tfinal=0.1n / 96)
    end
    for side in (:lo, :hi)
        k = side == :lo ? 1 : 2
        s, q = nested(side)
        lt = getfield(s, :levels)[3].transfers[1]
        p2 = PatchSolver(s, s.patches[3])
        @test lt.folded[1] == (k == 1, k == 2) && lt.boundary[1][k] && !lt.imposed[1][k]
        @test p2.folds[1] isa CL.FoldSpec && p2.bcs[1][k] === s.patches[1].bcs[1][k]
        # The second level's region starts at −4 of its own node space, its
        # first node half its spacing from the plane.
        @test p2.decomp.n_global[1] == 3 * 12 - 1
        k == 1 && @test p2.patch.region.offset[1] == -4
        edge = k == 1 ? xcoord(p2, 1, 1) : 1 - xcoord(p2, 1, p2.decomp.n_global[1])
        @test edge ≈ s.patches[1].h[1] / 18
        # The box of the second level mirrors level 1's nodes across the plane.
        @test CL._box_buffer(lt)[1] == (CL.LEVEL_BUFFER, CL.LEVEL_BUFFER)
        @test domain_volume(s) ≈ 1.0 atol = 1e-14
        for n in 1:96
            run!(s, q; tfinal=0.1n / 96)
        end
        @test s.step == f.step
        box = lt.box_gather
        pad = lt.pdecomps[1].n_halo_d[1]
        nb = lt.pdecomps[1].n_global[1]
        B = CL.LEVEL_BUFFER
        for j in 1:B
            out, in_ = k == 1 ? (B + 1 - j, B + j) : (nb - B + j, nb - B + 1 - j)
            @test box[out + pad, 1, 1, 1] == box[in_ + pad, 1, 1, 1]
            @test box[out + pad, 1, 1, 2] == -box[in_ + pad, 1, 1, 2]
        end
        # Against the uniform run at the second level's spacing, level by
        # level: level-ℓ node i is the uniform run's node i + offset + (3^ℓ − 1)/2
        # at 3^ℓ its spacing, the second level's node for node.
        e = 0.0
        for kk in 2:3
            ps = PatchSolver(s, s.patches[kk])
            ℓ = ps.patch.level
            ℓ == 2 || continue
            off = ps.patch.region.offset[1] + (3^ℓ - 1) ÷ 2
            @test maximum(abs(xcoord(ps, 1, i) - xcoord(pf, 1, off + i))
                          for i in 1:ps.decomp.n_local[1]) < 1e-14
            e = max(e, maximum(abs(q.patches[kk][padded_index(ps, i, 1, 1), c] -
                                   fq[padded_index(pf, off + i, 1, 1), c])
                               for i in 1:ps.decomp.n_local[1], c in 1:3))
        end
        @info "second level at a plane against the uniform run" side e
        @test e < FOLD_NEST_TOL
    end
    # A second level stopping short of the plane inside the margin is
    # refused, with the offset that reaches the plane named.
    @test_throws "offset -1 reaches the SymmetryPlaneBC" build_solver(
        Problem(domain=((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)), bcs=((plane, plane), per, per),
                ic=nothing),
        Numerics(n_global=(N, 1, 1),
                 amr=AMR(initial=[BlockRegion((0, 0, 0), (9, 1, 1)),
                                  BlockRegion((0, 0, 0), (12, 1, 1))],
                         regrid_interval=0)))
end

# A level across a periodic seam against the same level placed away from the
# seam, with the problem translated by the same number of root nodes: a
# periodic run is invariant under the translation, so the two differ by the
# round-off of the root's cyclic line solves alone. The largest difference
# over the conserved components and every patch: measured 1.1e-14 for a box
# in 1-D, either stepping mode, and on a corner box in 2-D, 8.4e-15 for tiles,
# 2.2e-14 (box) and 1.9e-14 (tiles) for a level at the r-z axis across the
# seam in z, static or regridded, and 4.4e-14 for a density step through a
# box under the coarse-fine correction on every junction line, whose
# composite mass holds to 4e-15.
const SEAM_SHIFT_TOL = 1e-12

# A box and tiles regridded across the seam, against the uniform run at the
# fine spacing in equal steps, the largest difference over the run: measured
# 4.8e-10 (box) and 4.3e-10 (tiles).
const SEAM_REGRID_TOL = 5e-9

@testset "a level across a periodic seam" begin
    per = (PeriodicBC(), PeriodicBC())
    N = 48
    # The entropy wave on the periodic unit line or square, translated by `s`
    # root nodes along each resolved dimension.
    wave(s, dims) = (x, y, z) -> begin
        phase = 2pi * (x - s / N) + (dims == 2 ? 2pi * (y - s / N) : 0.0)
        Prim(rho=1 + 0.2 * sin(phase + 0.37), u=(0.5, dims == 2 ? 0.25 : 0.0, 0.0),
             p=1.0)
    end
    function seam(; dims=1, s=0, region=nothing,subcycle=false, tile=0)
        local sv = build_solver(
            Problem(domain=((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)), bcs=(per, per, per),
                    ic=nothing),
            Numerics(n_global=dims == 2 ? (N, N, 1) : (N, 1, 1), cfl=0.9,
                     art=ArtificialProperties(enabled=false),
                     filter=StateFilter(interval=0),
                     amr=AMR(initial=region, regrid_interval=0, subcycle=subcycle,
                             tile=tile)))
        local q = allocate_state(sv)
        initialize!(sv, q, wave(s, dims))
        return sv, q
    end
    steps!(s, q, k0, k1, n, T, observer=nothing) = for k in k0:k1
        run!(s, q; tfinal=T * k / n, observer=observer)
    end
    # The largest difference between run `a` and run `b`, its translation by
    # `s` root nodes along every resolved dimension (or by `s[d]` along each):
    # root node g of `a` is node g + s of `b`, and each refined patch of `a`
    # is the patch of `b` whose region is its own shifted by 3^ℓ s, modulo the
    # level's period.
    function shifted_difference(a, qa, b, qb, s)
        local e = 0.0
        local S = s isa Int ? (s, s, s) : s
        for (k, p) in enumerate(a.patches)
            local act = p.decomp.active
            local P = ntuple(d -> act[d] ? 3^p.level * N : 1, 3)
            local sh = ntuple(d -> act[d] ? 3^p.level * S[d] : 0, 3)
            local pad = p.decomp.n_halo_d
            if p.level == 0
                for I in CL.interior(p.decomp), c in axes(qa.patches[1], 4)
                    local g = Tuple(I) .- pad
                    local J = CartesianIndex(mod1.(g .+ sh, P) .+ pad)
                    e = max(e, abs(qa.patches[1][I, c] - qb.patches[1][J, c]))
                end
                continue
            end
            local kb = findfirst(b.patches) do pb
                pb.level == p.level && pb.region.extent == p.region.extent &&
                    all(mod.(pb.region.offset .- p.region.offset .- sh, P) .== 0)
            end
            local ia = CL.interior(p.decomp)
            e = max(e, maximum(abs.(parent(qa.patches[k])[ia, :] .- parent(qb.patches[kb])[ia, :])))
        end
        return e
    end
    margin = CL.LEVEL_BUFFER

    # --- One box across the seam, 1-D, both stepping modes -------------------
    for subcycle in (false, true)
        a, qa = seam(region=BlockRegion((10, 0, 0), (9, 1, 1)); subcycle)
        # A region may start before the seam; it is stored from its first
        # node's place in the period.
        b, qb = seam(s=36, region=BlockRegion((-2, 0, 0), (9, 1, 1)); subcycle)
        lt = getfield(b, :levels)[2].transfers[1]
        @test level_regions(b, 1) == [BlockRegion((46, 0, 0), (9, 1, 1))]
        @test lt.imposed[1] == (true, true) && lt.boundary[1] == (false, false)
        @test lt.period == (N, 0, 0) && lt.coarse_regions == [b.patches[1].region]
        # The root's covered mask is the translated one, the covered nodes on
        # both sides of the seam, and the composite quadrature is exact.
        ca, cb = a.patches[1].amr.covered[:, 1, 1], b.patches[1].amr.covered[:, 1, 1]
        pad = a.patches[1].decomp.n_halo_d[1]
        @test all(cb[mod1(g + 36, N) + pad] == ca[g + pad] for g in 1:N)
        @test domain_volume(b) ≈ 1.0 atol = 1e-14
        steps!(a, qa, 1, 40, 40, 0.2)
        steps!(b, qb, 1, 40, 40, 0.2)
        @test a.step == b.step
        e = shifted_difference(a, qa, b, qb, 36)
        @info "box across the seam against its translation" subcycle e
        @test e < SEAM_SHIFT_TOL
    end
    # One box does not wrap around onto itself.
    @test_throws "periodic dimension 1" seam(region=BlockRegion((0, 0, 0), (N - 7, 1, 1)))

    # --- Density steps through a box across the seam, the correction on ---
    # A single step crosses the box's low face as the run goes, so the gate
    # flags the junction and the coarse-fine correction runs; across the seam
    # the junction's parent face node, its correction node and its stencil
    # lie in different images of the box. The gate tests the uncovered nodes
    # alone, so the steps after the step has entered the box go uncorrected
    # and the composite mass drifts, by 4.3e-5 in both images. A square wave
    # of six-node plateaus keeps a step among the gate's nodes at both faces
    # in every step of the run, so every junction line is corrected in every
    # step and the mass holds to round-off, measured 4.6e-15. Its edges lie
    # between nodes, so both translations sample the same values.
    density = (single=xs -> xs < 0.15 ? 2.0 : 1.0,
               wave=xs -> iseven(floor(Int, (xs * N + 0.5) / 6)) ? 2.0 : 1.0)
    for (profile, bound) in ((:single, 1e-4), (:wave, 1e-12))
        step_ic(s) = (x, y, z) ->
            Prim(rho=density[profile](mod(x - s / N, 1.0)), u=(0.5, 0.0, 0.0), p=1.0)
        contact = map(((0, BlockRegion((10, 0, 0), (9, 1, 1))),
                       (36, BlockRegion((-2, 0, 0), (9, 1, 1))))) do (s, r)
            local sv = build_solver(
                Problem(domain=((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)), bcs=(per, per, per),
                        ic=nothing),
                Numerics(n_global=(N, 1, 1), cfl=0.4, amr=AMR(initial=r, regrid_interval=0)))
            local q = allocate_state(sv)
            initialize!(sv, q, step_ic(s))
            local m0 = volume_integral(sv, q, :rho)
            local ledger = CL._ledger_begin!(sv, q)
            steps!(sv, q, 1, 30, 30, 0.3, ledger)
            local led = CL._ledger_end!(ledger, sv, q)
            (s=sv, q=q, drift=(volume_integral(sv, q, :rho) - m0) / m0,
             reflux=get(led.pieces, (:reflux, 0), [0.0])[1])
        end
        @test all(c -> c.reflux != 0, contact)
        e = shifted_difference(contact[1].s, contact[1].q, contact[2].s, contact[2].q,
                               36)
        @info("density steps through a box across the seam", profile, e,
              contact[1].drift, contact[2].drift)
        @test e < SEAM_SHIFT_TOL
        @test abs(contact[1].drift - contact[2].drift) < 1e-12
        @test all(c -> abs(c.drift) < bound, contact)
    end

    # --- A corner box across both seams, and a tiled level, 2-D -----------------
    a, qa = seam(dims=2, region=BlockRegion((10, 10, 0), (9, 9, 1)))
    b, qb = seam(dims=2, s=36, region=BlockRegion((46, 46, 0), (9, 9, 1)))
    @test domain_volume(b) ≈ 1.0 atol = 1e-14
    # The conservation ledger takes the parent's flux into the covered region
    # across the region's faces on both sides of the seam: its pieces
    # telescope and the two coarse-fine columns cancel as they do away from
    # the seam.
    ledgers = map(((a, qa), (b, qb))) do (s, q)
        local b0 = CL._conserved_budget(s, q)
        local ledger = CL._ledger_begin!(s, q)
        steps!(s, q, 1, 12, 12, 0.1, ledger)
        local r = CL._ledger_end!(ledger, s, q)
        local b1 = CL._conserved_budget(s, q)
        (drift=(b1.total_mass - b0.total_mass) / b0.total_mass, r=r,
         mass=b0.total_mass)
    end
    for l in ledgers
        @test maximum(abs, l.r.residual) < 1e-13
    end
    cf(l) = l.r.pieces[(:coarse_fine_flux, 1)][1] + l.r.pieces[(:covered_face_flux, 0)][1]
    @info "corner box across both seams" ledgers[1].drift ledgers[2].drift cf.(ledgers)
    @test abs(ledgers[2].drift - ledgers[1].drift) < 1e-13
    @test abs(cf(ledgers[2]) - cf(ledgers[1])) < 1e-13
    e = shifted_difference(a, qa, b, qb, 36)
    @info "corner box across both seams against its translation" e
    @test e < SEAM_SHIFT_TOL
    # Tiles on a lattice of edge 6, which divides N: the region's cells on
    # either side of the seam abut there as same-level neighbors, and the
    # tile whose last cell ends on the seam along y takes a parent-fed face.
    a, qa = seam(dims=2, tile=6, region=BlockRegion((12, 12, 0), (13, 7, 1)))
    b, qb = seam(dims=2, tile=6, s=30, region=BlockRegion((42, 42, 0), (13, 7, 1)))
    regs = level_regions(b, 1)
    @test regs == [BlockRegion((0, 42, 0), (7, 7, 1)), BlockRegion((42, 42, 0), (7, 7, 1))]
    @test [p.faces[1] for p in b.patches[2:end]] == [(2, 0), (0, 1)]
    @test [lt.imposed[2] for lt in getfield(b, :levels)[2].transfers] ==
          [(true, true), (true, true)]
    steps!(a, qa, 1, 12, 12, 0.1)
    steps!(b, qb, 1, 12, 12, 0.1)
    e = shifted_difference(a, qa, b, qb, 30)
    @info "tiles across the seam against their translation" e
    @test e < SEAM_SHIFT_TOL

    # A fold on one dimension and the seam on another: a level at the axis of
    # an r-z run across the periodic seam in z, static and regridded onto the
    # axis under level_boundaries, box and tiles, against its translation in z.
    function rz(; s=0, region=nothing,tile=0, regrid_interval=0,
                                      tag_threshold=CompactLES.DEFAULT_TAG_THRESHOLD,
                                      tag_predicate=nothing, tag_buffer=4,
                                      level_boundaries=true)
        local sv = build_solver(
            Problem(domain=((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)),
                    bcs=((AxisBC(), SlipWallBC()), per, per), metric=CylindricalMetric(),
                    ic=nothing),
            Numerics(n_global=(24, 1, N), cfl=0.9, art=ArtificialProperties(enabled=false),
                     filter=StateFilter(interval=1),
                     amr=AMR(initial=region, regrid_interval=regrid_interval,
                             tag_threshold=tag_threshold, tag_buffer=tag_buffer, tile=tile,
                             level_boundaries=level_boundaries)); tag_predicate=tag_predicate)
        local q = allocate_state(sv)
        initialize!(sv, q, (r, θ, z) ->
            Prim(rho=1 + 0.1 * exp(-10r^2) * sin(2pi * (z - s / N) + 0.37),
                 u=(0.0, 0.0, 0.3), p=1.0))
        return sv, q
    end
    near_axis(s) = (p, I) -> begin
        local i, _, k = interior_index(p, I)
        CL._domain_coordinate(p, 1, i) < 0.2 &&
            abs(mod(CL._domain_coordinate(p, 3, k) - 0.97 - s / N + 0.5, 1) - 0.5) < 0.06
    end
    for (tile, ra, rb, s) in ((0, BlockRegion((0, 0, 10), (9, 1, 9)),
                               BlockRegion((0, 0, 46), (9, 1, 9)), 36),
                              (6, BlockRegion((0, 0, 12), (7, 1, 13)),
                               BlockRegion((0, 0, 42), (7, 1, 13)), 30))
        a, qa = rz(region=ra, tile=tile)
        b, qb = rz(s=s, region=rb, tile=tile)
        @test getfield(b, :levels)[2].transfers[1].folded[1] == (true, false)
        steps!(a, qa, 1, 8, 8, 0.05)
        steps!(b, qb, 1, 8, 8, 0.05)
        e = shifted_difference(a, qa, b, qb, (0, 0, s))
        @info "a level at the axis across the seam in z against its translation" tile e
        @test e < SEAM_SHIFT_TOL
        # Regridded: the window at the axis lies across the seam in `b` and
        # clear of it in `a`.
        a, qa = rz(region=ra, tile=tile, regrid_interval=2, tag_threshold=Inf,
                   tag_predicate=near_axis(-s), tag_buffer=2, level_boundaries=true)
        b, qb = rz(s=s, region=rb, tile=tile, regrid_interval=2, tag_threshold=Inf,
                   tag_predicate=near_axis(0), tag_buffer=2, level_boundaries=true)
        steps!(a, qa, 1, 8, 8, 0.05)
        steps!(b, qb, 1, 8, 8, 0.05)
        ta, tb = level_regions(a, 1), level_regions(b, 1)
        @test length(ta) == length(tb) && all(r -> r.offset[1] == 0, tb) &&
              any(r -> r.offset[3] + r.extent[3] > N, tb)
        # The box moves with the window; the tiles already cover it.
        tile == 0 && @test tb != [rb]
        e = shifted_difference(a, qa, b, qb, (0, 0, s))
        @info "a level regridded onto the axis across the seam" tile e
        @test e < SEAM_SHIFT_TOL
    end

    # --- Regridded across the seam under level_boundaries ---------------------
    # A window moving right at unit speed, from x = 0.75 across the seam.
    xc(t) = 0.75 + t
    window(p, I) = abs(mod(xcoord(p, 1, interior_index(p, I)[1]) - xc(p.t) + 0.5, 1) -
                       0.5) < 0.08
    function moving(; refined=true, tile=0, lb=true)
        local n = refined ? N : 3N
        local s = build_solver(
            Problem(domain=((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)), bcs=(per, per, per),
                    ic=nothing),
            Numerics(n_global=(n, 1, 1), cfl=0.9, art=ArtificialProperties(enabled=false),
                     filter=StateFilter(interval=0),
                     amr=AMR(initial=refined ? BlockRegion((32, 0, 0), (9, 1, 1)) : nothing,
                             regrid_interval=refined ? 4 : 0, tag_threshold=Inf,
                             tag_buffer=2, tile=tile, level_boundaries=refined && lb));
            tag_predicate=refined ? window : nothing)
        local q = allocate_state(s)
        initialize!(s, q, wave(0, 1))
        return s, q
    end
    # The level's patches against the uniform run at their spacing: fine node
    # m of the level's node space is node m of the uniform run, modulo 3N.
    function against_uniform(s, q, f, fq)
        local pf = PatchSolver(f, f.patches[1])
        local e = 0.0
        for k in 2:length(s.patches)
            local ps = PatchSolver(s, s.patches[k])
            local off = ps.patch.region.offset[1]
            e = max(e, maximum(abs(q.patches[k][padded_index(ps, i, 1, 1), c] -
                                   fq[padded_index(pf, mod1(off + i, 3N), 1, 1), c])
                               for i in 1:ps.decomp.n_local[1], c in axes(q.patches[k], 4)))
        end
        return e
    end
    crosses(r) = r.offset[1] + r.extent[1] > N
    # Off, the level stays the margin clear of the seam and says so.
    s, q = moving(lb=false)
    @test_logs (:warn, r"does not reach") match_mode=:any steps!(s, q, 1, 48, 96, 0.4)
    @test !any(crosses, level_regions(s, 1))
    for tile in (0, 6)
        s, q = moving(; tile)
        f, fq = moving(refined=false)
        crossed = false
        e = 0.0
        dir = mktempdir()
        for k in 1:96
            run!(s, q; tfinal=0.4k / 96)
            run!(f, fq; tfinal=0.4k / 96)
            crossed |= any(crosses, level_regions(s, 1))
            e = max(e, against_uniform(s, q, f, fq))
            k == 48 && save_checkpoint(s, q, joinpath(dir, "seam"))
        end
        @info "level regridded across the seam against the uniform run" tile e
        @test crossed
        @test e < SEAM_REGRID_TOL
        # A restart past the crossing continues bit for bit.
        r, rq = moving(; tile)
        load_checkpoint!(r, rq, joinpath(dir, "seam"))
        for k in 49:96
            run!(r, rq; tfinal=0.4k / 96)
        end
        @test level_regions(r, 1) == level_regions(s, 1)
        @test r.t == s.t && length(rq.patches) == length(q.patches) &&
              all(parent(rq.patches[k])[CL.interior(s.patches[k].decomp), :] ==
                  parent(q.patches[k])[CL.interior(s.patches[k].decomp), :] for k in eachindex(q.patches))
        rm(dir; recursive=true)
    end
    # Three regridded levels: the second is tagged on the first's tiles, on
    # the lattice of the first level's node space, which wraps at its own
    # period, 3N.
    s3 = build_solver(
        Problem(domain=((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)), bcs=(per, per, per),
                ic=nothing),
        Numerics(n_global=(N, 1, 1), cfl=0.9, art=ArtificialProperties(enabled=false),
                 filter=StateFilter(interval=0),
                 amr=AMR(initial=nothing, regrid_interval=4, tag_threshold=Inf,
                         tag_buffer=2, tile=6, max_levels=3, level_boundaries=true));
        tag_predicate=window)
    q3 = allocate_state(s3)
    initialize!(s3, q3, wave(0, 1))
    crossed = false
    for k in 1:48
        run!(s3, q3; tfinal=0.4k / 96)
        crossed |= any(r -> r.offset[1] + r.extent[1] > 3N, level_regions(s3, 2))
    end
    @test crossed && all(all(isfinite, parent(Q)) for Q in q3.patches)

    # --- The frontend's shapes ---------------------------------------------------
    # A shape at the seam, given in the domain's coordinates as its two
    # parts: without level_boundaries its first level stays the margin clear
    # and says so, with it the level crosses the seam.
    at_seam = CL.Regions.Box((0.0, 0.0, 0.0), (0.05, 1.0, 1.0)) ∪
              CL.Regions.Box((0.95, 0.0, 0.0), (1.0, 1.0, 1.0))
    shape_run(lb) = setup(Problem(domain=((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)),
                                  bcs=(per, per, per), ic=wave(0, 1)),
                          Numerics(n_global=(N, 1, 1),
                                   art=ArtificialProperties(enabled=false),
                                   amr=AMR(initial=at_seam, level_boundaries=lb)))
    s, _ = @test_logs (:warn, r"domain boundary") match_mode=:any shape_run(false)
    @test refined_region(s).offset[1] == margin
    s, _ = @test_logs min_level=Base.CoreLogging.Warn shape_run(true)
    r = refined_region(s)
    @test r.offset[1] + r.extent[1] > N && r.offset[1] + r.extent[1] - N >= 3
end
