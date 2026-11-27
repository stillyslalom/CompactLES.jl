# The MPI suite's phases of inflow and outflow faces, the turbulent inflow, the
# hydrostatic state, composite faces, and levels at planes, the axis and folds.
# Included by test/mpi_tests.jl, which loads the package, defines `check`, `section`
# and the other helpers used here, and runs the phases; run that file, not this one.

# ---------------------------------------------------------------------------
# 4c. NSCBC inflow with its plane decomposed. The transverse terms read
#     compact derivatives of p and ρ along the transverse dimensions,
#     distributed solves every rank must enter ahead of the plane-ownership
#     test, and the plane then spans several ranks, each holding a block of
#     the transverse variation. The right-hand side and a few steps are
#     measured against the serial rebuild on COMM_SELF, split along the
#     transverse dimension and along the face normal in turn; the second
#     split leaves the plane on one rank while the normal solves stay
#     distributed. With the artificial properties on, the default species
#     channel leaves `grad_Y` to this condition, whose own solves then join
#     the ones ahead of the ownership test.
# ---------------------------------------------------------------------------
function test_nscbc_inflow()
    section("NSCBC inflow: transverse terms on a decomposed plane")
    function blockdiff(s, a, ref, b)
        e = 0.0
        for I in CL.interior(s.decomp), c in 1:s.equations.n_cons
            loc = Tuple(I) .- s.decomp.n_halo_d
            J = padded_index(ref, (loc .+ s.decomp.offset)...)
            e = max(e, abs(Float64(a[I, c] - b[J, c])))
        end
        e
    end
    ic = (x, y, z) -> begin
        θ = 0.5 + 0.2 * sin(2π * y)
        Prim(u=(0.3 + 0.05 * sin(2π * y), 0.1 * cos(2π * y), 0.0),
             p=1 + 0.05 * sin(2π * y + 1), T_ion=1 + 0.1 * cos(2π * y),
             Y=(θ, 1 - θ))
    end
    function build(comm_here, dims_here, art_on)
        bc = NSCBCInflowBC(u=(0.3, 0.0, 0.0), T_ion=1.0, Y=[0.6, 0.4])
        sol = build_solver(Problem(domain=((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)),
                                   bcs=((bc, NSCBCOutflowBC(pinf=1.0)), per3[2], per3[3]),
                                   eos=IdealMixture([IdealSpecies{Float64}("a", 1.0, 1.4),
                                                IdealSpecies{Float64}("b", 0.5, 1.4)]),
                                   transport=ConstantTransport(mu0=0.0), ic=nothing),
                           Numerics(n_global=(SPLITN, SPLITN, 1),
                                    art=ArtificialProperties(enabled=art_on), cfl=0.4,
                                    execution=Execution(comm=comm_here, dims=dims_here)))
        Q = allocate_state(sol)
        initialize!(sol, Q, ic)
        apply_bcs!(sol, Q)
        dQ = zero(Q)
        compute_rhs!(sol, Q, dQ)
        return sol, Q, dQ
    end
    for art_on in (false, true), ax in (2, 1)
        s, Q, dQ = build(comm, splitdims(ax), art_on)
        ref, Qref, dQref = build(MPI.COMM_SELF, (1, 1, 1), art_on)
        tag = art_on ? ", artificial properties on" : ""
        check("inflow RHS matches serial, split along dim $ax$tag",
              gmax(blockdiff(s, dQ, ref, dQref)), 1e-10)
        run!(s, Q; tfinal=1e9, nmax=4)
        run!(ref, Qref; tfinal=1e9, nmax=4)
        check("inflow run matches serial, split along dim $ax$tag",
              gmax(blockdiff(s, Q, ref, Qref)), 1e-10)
    end
end

# ---------------------------------------------------------------------------
# 4c'. An NSCBC face carried by a refined level. The level's patch is
#     decomposed along the face normal over its own communicator, so its
#     ranks that own no piece of the face plane enter the characteristic
#     correction's distributed solves on that communicator and return, while
#     the root's correction runs on the root's. Every block the rank holds is
#     measured against the serial rebuild on COMM_SELF after a few steps, in
#     1-D with the level at the outflow and at the inflow, global step and
#     subcycled, and in 2-D with the transverse terms, the plane then divided
#     along y as well at np = 8.
# ---------------------------------------------------------------------------
function test_nscbc_level()
    section("NSCBC level face: the correction on a level's communicator")
    # Max |distributed − serial| over this rank's interior block of patch
    # `p`, against the serial patch `rp`, which holds the whole block.
    function blockdiff(p, Q, rp, Qref)
        e = 0.0
        for I in CL.interior(p.decomp), c in 1:size(Q, 4)
            loc = Tuple(I) .- p.decomp.n_halo_d
            J = CartesianIndex(loc .+ p.decomp.offset .+ rp.decomp.n_halo_d)
            e = max(e, abs(Float64(Q[I, c] - Qref[J, c])))
        end
        e
    end
    # The root block and each level block the rank holds, the serial patch
    # of a level block found by its tile index.
    function leveldiff(s, qs, ref, qref)
        e = blockdiff(s.patches[1], qs.patches[1], ref.patches[1], qref.patches[1])
        lev, rlev = s.levels[2], ref.levels[2]
        for (k, ti) in zip(lev.patches, lev.tiles)
            rk = rlev.patches[findfirst(==(ti), rlev.tiles)]
            e = max(e, blockdiff(s.patches[k], qs.patches[k], ref.patches[rk], qref.patches[rk]))
        end
        e
    end
    # A right-running acoustic pulse on a stream at u0 = 0.3 (ρ = p = 1, so
    # c = √1.4), centred on the refined face; in 2-D its amplitude and a
    # transverse velocity vary along y.
    function pulse(x0, y2d)
        (x, y, z) -> begin
            f = 0.01 * exp(-((x - x0) / 0.03)^2) *
                (y2d ? 1 + 0.3 * sin(2π * y) : 1.0)
            Prim(rho=1 + f / 1.4, u=(0.3 + f / sqrt(1.4),
                                      y2d ? 0.05 * cos(2π * y) : 0.0, 0.0),
                 p=1 + f)
        end
    end
    bcs = ((NSCBCInflowBC(u=(0.3, 0.0, 0.0), T_ion=1.0), NSCBCOutflowBC(pinf=1.0)),
           per3[2], per3[3])
    function build(comm_here, ng, region, ic, subcycle)
        s = build_solver(
            Problem(domain=((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)), bcs=bcs, ic=nothing),
            Numerics(n_global=ng, art=ArtificialProperties(enabled=false), cfl=0.5,
                     filter=StateFilter(interval=1), execution=Execution(comm=comm_here),
                     amr=AMR(initial=region, regrid_interval=0, subcycle=subcycle)))
        q = allocate_state(s)
        initialize!(s, q, ic)
        return s, q
    end
    function compare(label, ng, region, ic; subcycle=false, nmax=6)
        s, q = build(comm, ng, region, ic, subcycle)
        ref, qref = build(MPI.COMM_SELF, ng, region, ic, subcycle)
        run!(s, q; tfinal=1e9, nmax=nmax)
        run!(ref, qref; tfinal=1e9, nmax=nmax)
        check("NSCBC level $label matches serial", gmax(leveldiff(s, q, ref, qref)),
              1e-10)
        return s
    end
    # 1-D: the level over the last or the first 25 of 97 root nodes, 73 fine
    # nodes, which divide over every rank up to np = 8 along the normal.
    ng1 = (97, 1, 1)
    for (face, region, x0) in ((:outflow, BlockRegion((72, 0, 0), (25, 1, 1)), 0.97),
                               (:inflow, BlockRegion((0, 0, 0), (25, 1, 1)), 0.03)),
        subcycle in (false, true)
        tag = "$face$(subcycle ? ", subcycled" : "")"
        s = compare("1-D at the $tag", ng1, region,
                    pulse(x0, false); subcycle=subcycle)
        side = face === :outflow ? 2 : 1
        held = [s.patches[k].decomp for k in s.levels[2].patches]
        split = maximum((dc.sub_size[1] for dc in held); init=1)
        without = count(dc -> CL.wallplane(dc, 1, side) === nothing, held)
        check("NSCBC level 1-D $tag: split along the normal",
              gmax(split) == np ? 0.0 : 1.0, 0.5)
        check("NSCBC level 1-D $tag: ranks off the plane enter",
              abs(gsum(without) - (np - 1)), 0.5)
    end
    # 2-D: a 25 × 12 root-node level at the outflow, inside the periodic y
    # extent by the nesting margin, the fine patch 73 × 34, split along x at
    # np = 2 and 4 and along both dimensions at np = 8.
    compare("2-D at the outflow, transverse terms", (72, 24, 1),
            BlockRegion((47, 6, 0), (25, 12, 1)), pulse(0.97, true); nmax=4)
end

# ---------------------------------------------------------------------------
# Synthetic turbulent inflow as an NSCBC target on a face divided among the
# ranks. The field is a function of position and time alone, so the decomposed
# run matches the serial one to round-off and a restart continues bit for bit
# with no generator state in the checkpoint; a different seed is a boundary
# change the configuration record refuses.
# ---------------------------------------------------------------------------
function test_turbulent_inflow()
    section("turbulent inflow: decomposed face and restart")
    function blockdiff(s, a, ref, b)
        e = 0.0
        for I in CL.interior(s.decomp), c in 1:s.equations.n_cons
            loc = Tuple(I) .- s.decomp.n_halo_d
            J = padded_index(ref, (loc .+ s.decomp.offset)...)
            e = max(e, abs(Float64(a[I, c] - b[J, c])))
        end
        e
    end
    mean = Prim(u=(0.3, 0.0, 0.0), p=1.0, T_ion=1.0)
    function build(comm_here, dims_here; seed=1)
        turb = TurbulentInflow(mean; length_scale=0.1, intensity=0.1, seed=seed)
        inlet = NSCBCInflowBC(mean; target=turb, eta_u=5.0, eta_T=5.0)
        sol = build_solver(
            Problem(domain=((0.0, 0.5), (0.0, 1.0), (0.0, 0.4)),
                    bcs=((inlet, NSCBCOutflowBC(pinf=1.0)), per3[2], per3[3]), ic=nothing),
            Numerics(n_global=(16, SPLITN, 18), art=ArtificialProperties(enabled=false),
                     cfl=0.4, execution=Execution(comm=comm_here, dims=dims_here)))
        Q = allocate_state(sol)
        initialize!(sol, Q, (x, y, z) -> mean)
        return sol, Q
    end
    ref, Qref = build(MPI.COMM_SELF, (1, 1, 1))
    run!(ref, Qref; tfinal=1e9, nmax=4)
    grids = np >= 4 && iseven(np) ? (splitdims(2), (1, np ÷ 2, 2)) : (splitdims(2),)
    for dims_here in grids
        s, Q = build(comm, dims_here)
        run!(s, Q; tfinal=1e9, nmax=2)
        save_checkpoint(s, Q, "mpi_turb_ckpt")
        run!(s, Q; tfinal=1e9, nmax=4)
        check("turbulent inflow run matches serial, dims $dims_here",
              gmax(blockdiff(s, Q, ref, Qref)), 1e-10)
        r, Qr = build(comm, dims_here)
        load_checkpoint!(r, Qr, "mpi_turb_ckpt")
        inner = CL.interior(s.decomp)
        run!(r, Qr; tfinal=1e9, nmax=4)
        check("restart continues bit for bit, dims $dims_here",
              gmax(maximum(abs, parent(Qr)[inner, :] - parent(Q)[inner, :])) +
              abs(r.t - s.t), 1e-300)
        other, Qo = build(comm, dims_here; seed=2)
        threw = try
            load_checkpoint!(other, Qo, "mpi_turb_ckpt")
            0
        catch e
            occursin("allow = (:boundaries,)", sprint(showerror, e)) ? 1 : 0
        end
        check("another seed is refused on every rank", abs(gsum(threw) - np), 0.5)
        MPI.Barrier(comm)
        rank == 0 && foreach(rm, filter(startswith("mpi_turb_ckpt"), readdir()))
        MPI.Barrier(comm)
    end
end

# ---------------------------------------------------------------------------
# Hydrostatic initial state with the acceleration along a decomposed
# dimension. Every rank solves its own lines whole, so the state must match the
# serial one exactly and stay at rest through the distributed divergence.
# ---------------------------------------------------------------------------
function test_hydrostatic()
    section("hydrostatic state: balance along a decomposed dimension")
    function blockdiff(s, a, ref, b)
        e = 0.0
        for I in CL.interior(s.decomp), c in 1:s.equations.n_cons
            loc = Tuple(I) .- s.decomp.n_halo_d
            J = padded_index(ref, (loc .+ s.decomp.offset)...)
            e = max(e, abs(Float64(a[I, c] - b[J, c])))
        end
        e
    end
    eos = IdealMixture([IdealSpecies("heavy"; R=1.0, gamma=1.4),
                        IdealSpecies("light"; R=3.0, gamma=1.4)])
    heavy = Prim(Y=(1.0, 0.0), p=10.0, rho=3.0)
    light = Prim(Y=(0.0, 1.0), p=10.0, rho=1.0)
    for ax in (1, 2)
        g = ntuple(d -> d == ax ? -1.0 : 0.0, 3)
        bcs = ntuple(d -> d == ax ? (SlipWallBC(), SlipWallBC()) : per3[d], 3)
        ic = Hydrostatic(Layers(light, Slab(ax, lo=0.5) => heavy); p_ref=10.0, at=1.0)
        prob = Problem(eos=eos, domain=((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)), bcs=bcs,
                       ic=ic, sources=(ConstantBodyForce(g),))
        ng = ntuple(d -> d == ax ? SPLITN : d == 3 ? 1 : 12, 3)
        num(c, dims) = Numerics(n_global=ng, filter=nothing,
                                execution=Execution(comm=c, dims=dims),
                                art=ArtificialProperties(enabled=false))
        s, Q = setup(prob, num(comm, splitdims(ax)))
        ref, Qref = setup(prob, num(MPI.COMM_SELF, (1, 1, 1)))
        check("hydrostatic state matches serial, split along dim $ax",
              gmax(blockdiff(s, Q, ref, Qref)), 1e-13)
        run!(s, Q; tfinal=1e9, nmax=50)
        m = 0.0
        for I in CL.interior(s.decomp), c in s.equations.i_mom
            m = max(m, abs(Q[I, c]))
        end
        check("hydrostatic column at rest after 50 steps, split along dim $ax",
              gmax(m), 1e-13)
    end
end

# ---------------------------------------------------------------------------
# Composite face with its mask divided among the ranks. Four members share
# the low-x face: a no-slip wall, a Dirichlet stream, a characteristic outflow
# and a characteristic inflow, over bands of y. Split along y, the face spans
# every rank and most ranks hold one or two of the members, so the other
# members' distributed solves run on ranks whose part of the face holds none
# of their points. Split along x, one rank holds the whole face and the rest
# none of it. The right-hand side and a few steps are measured against the
# serial rebuild on COMM_SELF, with and without the artificial properties.
# ---------------------------------------------------------------------------
function test_composite_face()
    section("composite face: members' collectives on ranks without their points")
    function blockdiff(s, a, ref, b)
        e = 0.0
        for I in CL.interior(s.decomp), c in 1:s.equations.n_cons
            loc = Tuple(I) .- s.decomp.n_halo_d
            J = padded_index(ref, (loc .+ s.decomp.offset)...)
            e = max(e, abs(Float64(a[I, c] - b[J, c])))
        end
        e
    end
    ic = (x, y, z) -> begin
        θ = 0.5 + 0.2 * sin(2π * y)
        Prim(u=(0.2 + 0.05 * sin(2π * y), 0.05 * cos(2π * y), 0.0),
             p=1 + 0.02 * sin(2π * y + 1), T_ion=1 + 0.05 * cos(2π * y),
             Y=(θ, 1 - θ))
    end
    stream = (x, y, z, t) -> Prim(u=(0.2, 0.0, 0.0), T_ion=1.0, rho=1.0, Y=(0.6, 0.4))
    # Bands at the eighths, the y blocks of an eight-rank split.
    band(y) = y < 0.25 ? 1 : y < 0.375 ? 2 : y < 0.5 ? 3 : y < 0.75 ? 4 : 1
    function build(comm_here, dims_here, art_on)
        face = CompositeBC((NoSlipWallBC(), DirichletBC(stream),
                            NSCBCOutflowBC(pinf=1.0),
                            NSCBCInflowBC(u=(0.2, 0.0, 0.0), T_ion=1.0, Y=[0.6, 0.4])),
                           (x, y, z) -> band(y))
        sol = build_solver(Problem(domain=((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)),
                                   bcs=((face, NSCBCOutflowBC(pinf=1.0)), per3[2], per3[3]),
                                   eos=IdealMixture([IdealSpecies{Float64}("a", 1.0, 1.4),
                                                IdealSpecies{Float64}("b", 0.5, 1.4)]),
                                   transport=ConstantTransport(mu0=1e-3), ic=nothing),
                           Numerics(n_global=(SPLITN, SPLITN, 1),
                                    art=ArtificialProperties(enabled=art_on), cfl=0.4,
                                    execution=Execution(comm=comm_here, dims=dims_here)))
        Q = allocate_state(sol)
        initialize!(sol, Q, ic)
        apply_bcs!(sol, Q)
        dQ = zero(Q)
        compute_rhs!(sol, Q, dQ)
        return sol, Q, dQ
    end
    for art_on in (false, true), ax in (2, 1)
        s, Q, dQ = build(comm, splitdims(ax), art_on)
        ref, Qref, dQref = build(MPI.COMM_SELF, (1, 1, 1), art_on)
        tag = art_on ? ", artificial properties on" : ""
        check("composite RHS matches serial, split along dim $ax$tag",
              gmax(blockdiff(s, dQ, ref, dQref)), 1e-10)
        run!(s, Q; tfinal=1e9, nmax=4)
        run!(ref, Qref; tfinal=1e9, nmax=4)
        check("composite run matches serial, split along dim $ax$tag",
              gmax(blockdiff(s, Q, ref, Qref)), 1e-10)
    end
end

# ---------------------------------------------------------------------------
# A refined level reaching a symmetry plane, its tile folding there at the
# fine spacing, with the tile decomposed over the level's ranks: the fold's
# mirror fill on the edge rank, the box's mirror across the plane on every
# owner, and the restriction's coincident samples shifted by the node beyond
# the lattice. A few steps with the artificial properties and the `:d8`
# detector on two species and molecular ghost fluxes, against the serial
# rebuild on COMM_SELF, tile by tile.
# ---------------------------------------------------------------------------
function test_plane_level()
    section("level at a symmetry plane: decomposed tile against serial")
    sym = (SymmetryPlaneBC(), SymmetryPlaneBC())
    per = (PeriodicBC(), PeriodicBC())
    # Max |distributed − serial| over this rank's block of one patch.
    function blockdiff(ps, a, pr, b)
        e = 0.0
        for I in CL.interior(ps.decomp), c in 1:size(a, 4)
            loc = Tuple(I) .- ps.decomp.n_halo_d
            J = padded_index(pr, (loc .+ ps.decomp.offset)...)
            e = max(e, abs(a[I, c] - b[J, c]))
        end
        e
    end
    function build(comm_here, tile)
        s = build_solver(Problem(domain=((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)),
                                 bcs=(sym, per, per),
                                 eos=IdealMixture([IdealSpecies{Float64}("a", 1.0, 1.4),
                                              IdealSpecies{Float64}("b", 0.7, 1.3)]),
                                 transport=ConstantTransport{Float64}(mu0=0.01), ic=nothing),
                         Numerics(n_global=(48, 24, 1), cfl=0.4,
                                  art=ArtificialProperties(enabled=true, detector=:d8),
                                  filter=StateFilter(interval=1),
                                  execution=Execution(comm=comm_here),
                                  amr=AMR(initial=BlockRegion((0, 4, 0), (10, 16, 1)),
                                          regrid_interval=0, tile=tile)))
        Q = allocate_state(s)
        initialize!(s, Q, (x, y, z) -> begin
            Y1 = 0.5 + 0.2 * cos(pi * x) * sin(2pi * y)
            Prim(rho=1 + 0.1 * cos(pi * x), u=(0.05 * sin(pi * x), 0.02, 0.0),
                 p=1.0 + 0.05 * cos(2pi * y), Y=(Y1, 1 - Y1))
        end)
        run!(s, Q; tfinal=1e9, nmax=3)
        return s, Q
    end
    for tile in (0, 5)
        s, Q = build(comm, tile)
        ref, Qref = build(MPI.COMM_SELF, tile)
        e = 0.0
        for (k, p) in enumerate(s.patches)
            p.level == 1 || continue
            r = findfirst(q -> q.level == 1 && q.region == p.region, ref.patches)
            ps, pr = PatchSolver(s, p), PatchSolver(ref, ref.patches[r])
            e = max(e, blockdiff(ps, Q.patches[k], pr, Qref.patches[r]))
        end
        # A tile folds exactly where it reaches the plane, which puts its
        # first node, one beyond the coincident lattice, at offset −1.
        check("tile $tile: the tiles fold where they reach the plane",
              gmax(count(p -> p.level == 1 &&
                              CL._fold_active(p.folds[1]) !=
                              (p.region.offset[1] == -1),
                         s.patches)), 0.5)
        check("tile $tile: fine state against serial", gmax(e), 1e-11)
        check("tile $tile: root state against serial",
              gmax(blockdiff(PatchSolver(s, s.patches[1]), Q.patches[1],
                             PatchSolver(ref, ref.patches[1]), Qref.patches[1])), 1e-11)
        check("tile $tile: composite volume exact", abs(domain_volume(s) - 1), 1e-13)
    end
end

# ---------------------------------------------------------------------------
# A refined level reaching the axis of an r-z run and a symmetry plane at
# z = 0, the corner tile folding on both at the fine spacing, with the tile
# decomposed over the level's ranks: the axis fold's mirror fill on the edge
# rank with its odd radial velocity and area factor, the box mirror across
# both folds on every owner, and the restriction's coincident samples. A few
# steps with the artificial properties and the `:d8` detector on two species
# and molecular ghost fluxes, against the serial rebuild on COMM_SELF, tile
# by tile.
# ---------------------------------------------------------------------------
function test_axis_level()
    section("level on the r-z axis: decomposed corner tile against serial")
    axis = (AxisBC(), SlipWallBC())
    per = (PeriodicBC(), PeriodicBC())
    plane = (SymmetryPlaneBC(), SlipWallBC())
    function blockdiff(ps, a, pr, b)
        e = 0.0
        for I in CL.interior(ps.decomp), c in 1:size(a, 4)
            loc = Tuple(I) .- ps.decomp.n_halo_d
            J = padded_index(pr, (loc .+ ps.decomp.offset)...)
            e = max(e, abs(a[I, c] - b[J, c]))
        end
        e
    end
    function build(comm_here, tile)
        s = build_solver(Problem(domain=((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)),
                                 bcs=(axis, per, plane), metric=CylindricalMetric(),
                                 eos=IdealMixture([IdealSpecies{Float64}("a", 1.0, 1.4),
                                              IdealSpecies{Float64}("b", 0.7, 1.3)]),
                                 transport=ConstantTransport{Float64}(mu0=0.01), ic=nothing),
                         Numerics(n_global=(48, 1, 36), cfl=0.4,
                                  art=ArtificialProperties(enabled=true, detector=:d8),
                                  filter=StateFilter(interval=1),
                                  execution=Execution(comm=comm_here),
                                  amr=AMR(initial=BlockRegion((0, 0, 0), (10, 1, 10)),
                                          regrid_interval=0, tile=tile)))
        Q = allocate_state(s)
        initialize!(s, Q, (r, θ, z) -> begin
            Y1 = 0.5 + 0.2 * cos(pi * r) * cos(pi * z)
            Prim(rho=1 + 0.1 * cos(pi * r) * cos(2pi * z),
                 u=(0.05 * sin(pi * r), 0.0, 0.03 * sin(pi * z)),
                 p=1.0 + 0.05 * cos(pi * r), Y=(Y1, 1 - Y1))
        end)
        run!(s, Q; tfinal=1e9, nmax=3)
        return s, Q
    end
    for tile in (0, 5)
        s, Q = build(comm, tile)
        ref, Qref = build(MPI.COMM_SELF, tile)
        e = 0.0
        for (k, p) in enumerate(s.patches)
            p.level == 1 || continue
            r = findfirst(q -> q.level == 1 && q.region == p.region, ref.patches)
            ps, pr = PatchSolver(s, p), PatchSolver(ref, ref.patches[r])
            e = max(e, blockdiff(ps, Q.patches[k], pr, Qref.patches[r]))
        end
        # A tile folds on a dimension exactly where it reaches the axis or
        # the plane, which puts its first node there at offset −1.
        check("tile $tile: the tiles fold where they reach the axis and the plane",
              gmax(count(p -> p.level == 1 &&
                              any(d -> CL._fold_active(p.folds[d]) !=
                                       (p.region.offset[d] == -1), (1, 3)),
                         s.patches)), 0.5)
        check("tile $tile: fine state against serial", gmax(e), 1e-11)
        check("tile $tile: root state against serial",
              gmax(blockdiff(PatchSolver(s, s.patches[1]), Q.patches[1],
                             PatchSolver(ref, ref.patches[1]), Qref.patches[1])), 1e-11)
        check("tile $tile: composite volume against serial",
              abs(domain_volume(s) - domain_volume(ref)), 1e-14)
    end
end

# ---------------------------------------------------------------------------
# Levels regridded onto a fold (`level_boundaries`), decomposed: a tagged
# window moving onto the symmetry plane of a 1-D tube, and onto the axis of
# an r-z radial line, followed by one box, by lattice tiles and by three
# regridded levels whose second level reaches the fold at offset −1, against
# the serial rebuild on COMM_SELF every three steps through the regrids: the
# same regions on every level, the fold on the tiles that reach it, and every
# block this rank holds equal to the serial patch's. A rebalanced tiled run
# with the migration audit on moves folded tiles block to block and compares
# them with the replicated carry.
# ---------------------------------------------------------------------------
function test_fold_regrid()
    section("folded regrid: tags place levels on a symmetry plane and the axis")
    per = (PeriodicBC(), PeriodicBC())
    function blockdiff(p, Q, rp, Qref)
        e = 0.0
        for I in CL.interior(p.decomp), c in 1:size(Q, 4)
            loc = Tuple(I) .- p.decomp.n_halo_d
            J = CartesianIndex(loc .+ p.decomp.offset .+ rp.decomp.n_halo_d)
            e = max(e, abs(Float64(Q[I, c] - Qref[J, c])))
        end
        e
    end
    function leveldiff(s, qs, ref, qref)
        e = blockdiff(s.patches[1], qs.patches[1], ref.patches[1], qref.patches[1])
        for ℓ in 2:length(s.levels)
            lev, rlev = s.levels[ℓ], ref.levels[ℓ]
            for (k, ti) in zip(lev.patches, lev.tiles)
                region = lev.transfers[ti].region
                rt = findfirst(lt -> lt.region == region, rlev.transfers)
                rk = rlev.patches[findfirst(==(rt), rlev.tiles)]
                e = max(e, blockdiff(s.patches[k], qs.patches[k], ref.patches[rk], qref.patches[rk]))
            end
        end
        e
    end
    regions(s) = [level_regions(s, ℓ) for ℓ in 1:length(s.levels)-1]
    # A window of half-width 0.08 moving left at 4 onto the fold at x = 0.
    window(p, I) = abs(CL.xcoord(p, 1, CL.interior_index(p, I)[1]) - (0.2 - 4 * p.t)) <
                   0.08
    wave = (x, y, z) -> (rho = 1 + 0.05 * cos(pi * x);
                         Prim(rho=rho, u=(0.05 * sin(pi * x), 0.0, 0.0), p=rho^1.4))
    function build(comm_here, fold, tile; levels=2,rebalance=0, rebalance_persist=2)
        cyl = fold[1] isa AxisBC
        local s = build_solver(
            Problem(domain=((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)),
                    bcs=(fold, per3[2], per3[3]),
                    metric=cyl ? CylindricalMetric() : CartesianMetric(), ic=nothing),
            Numerics(n_global=(97, 1, 1), art=ArtificialProperties(enabled=false), cfl=0.5,
                     filter=StateFilter(interval=1), execution=Execution(comm=comm_here),
                     amr=AMR(initial=levels > 2 ? nothing :
                                BlockRegion((24, 0, 0), (17, 1, 1)),
                             regrid_interval=3, tag_threshold=Inf, tag_buffer=2, tile=tile,
                             rebalance=rebalance, rebalance_persist=rebalance_persist,
                             max_levels=levels, level_boundaries=true,
                             subdivide_lattice=false)); tag_predicate=window)
        local q = allocate_state(s)
        initialize!(s, q, wave)
        return s, q
    end
    for (name, fold) in (("plane", (SymmetryPlaneBC(), SlipWallBC())),
                         ("axis", (AxisBC(), SlipWallBC()))),
        (label, tile, levels) in (("box", 0, 2), ("tiles", 8, 2), ("three levels", 8, 3))
        s, q = build(comm, fold, tile; levels)
        ref, qref = build(MPI.COMM_SELF, fold, tile; levels)
        differ = 0
        folded = falses(levels - 1)
        e = 0.0
        # A rank outside a level's parent holds none of its layout; rank 0
        # holds every level's, so its reading is the run's. The second level
        # of the three reaches the fold at the check after step 21.
        for n in 3:3:24
            run!(s, q; tfinal=1e9, nmax=n)
            run!(ref, qref; tfinal=1e9, nmax=n)
            differ += MPI.bcast(regions(s) != regions(ref), comm; root=0)
            for ℓ in 1:levels-1, lt in s.levels[ℓ + 1].transfers
                lt.folded[1][1] && lt.region.offset[1] == (1 - 3^(ℓ - 1)) ÷ 2 &&
                    (folded[ℓ] = true)
            end
            differ == 0 && (e = max(e, leveldiff(s, q, ref, qref)))
        end
        check("folded regrid, $name $label: the regions regrid as serial",
              differ, 0.5)
        check("folded regrid, $name $label: every level reaches the fold",
              MPI.bcast(count(!, folded), comm; root=0), 0.5)
        check("folded regrid, $name $label: every block matches serial", gmax(e), 1e-10)
    end
    # Folded tiles moved by a rebalance migrate block to block; the audit
    # carries each through the replicated gather as well.
    CL.MIGRATION_AUDIT[] = true
    CL.MIGRATION_AUDIT_RESULT[] = (tiles=0, mismatches=0)
    s, q = build(comm, (SymmetryPlaneBC(), SlipWallBC()), 4; rebalance=1.0,
                 rebalance_persist=1)
    run!(s, q; tfinal=1e9, nmax=30)
    CL.MIGRATION_AUDIT[] = false
    audit = CL.MIGRATION_AUDIT_RESULT[]
    audited = gsum(audit.tiles)
    rank == 0 && println("  folded regrid: $audited migrated tile(s) audited")
    check("folded regrid: migrated state equals the gathered carry bitwise",
          gsum(audit.mismatches), 0.5)
end
