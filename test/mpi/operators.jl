# The MPI suite's phases of the decomposed line solves, transfers, folds, halos,
# freestream, conservation and synchronization. Included by test/mpi_tests.jl, which
# loads the package, defines `check`, `section` and the other helpers used here, and
# runs the phases; run that file, not this one.

# ---------------------------------------------------------------------------
# 1. Periodic C6 derivative split across ranks (default tridiagonal scheme).
#    Split along x (dim-1 path) and along y and z (the transposed path) — each
#    forces the LineSolver reduced-interface stage in that dimension.
# ---------------------------------------------------------------------------
function test_periodic_c6()
    section("periodic C6 derivative: reduced-interface solve across ranks")
    fx  = (x, y, z) -> sin(3x) * cos(2y)
    dfx = (x, y, z) -> 3cos(3x) * cos(2y)   # ∂/∂x
    dfy = (x, y, z) -> -2sin(3x) * sin(2y)  # ∂/∂y
    # Split each dimension in turn.
    for ax in 1:3
        ng = ntuple(d -> d == ax ? SPLITN : 16, 3)
        solver = build_solver(
            Problem(domain=((0.0, 2π), (0.0, 2π), (0.0, 2π)), bcs=per3, ic=nothing),
            Numerics(n_global=ng, art=ArtificialProperties(enabled=false),
                     execution=Execution(dims=splitdims(ax))))
        f = CL.field(solver.decomp); df = CL.field(solver.decomp)
        fillf!(solver, f, fx)
        CL.exchange_halos!(f, solver.decomp)
        CL.deriv_along!(df, f, solver, ax, 1); CL._scale_grad!(df, solver, ax)
        ref = ax == 1 ? dfx : ax == 2 ? dfy : ((x, y, z) -> 0.0)
        # x/y derivatives of sin(3x)cos(2y) are nontrivial; the z derivative is 0.
        tol = ax == 3 ? 1e-10 : 1e-4
        check("∂/∂x$(ax) split $(np)-way (dims=$(splitdims(ax)))",
              gmax(ferr(solver, df, ref)), tol)
    end
end

# ---------------------------------------------------------------------------
# 2. Pentadiagonal C10 derivative split across ranks — the headline case: the
#    q = 2 BandLineSolver reduced solve (2q·P dense system, spike correction)
#    is reachable only with P > 1 in the split dimension. 64 points keep ≥ 7
#    per rank up to np = 8 (C10 needs local extent ≥ 7).
# ---------------------------------------------------------------------------
function test_pentadiagonal_c10()
    section("pentadiagonal C10 derivative: banded reduced-interface solve across ranks")
    fx  = (x, y, z) -> sin(3x) * cos(2y)
    dfx = (x, y, z) -> 3cos(3x) * cos(2y)
    dfy = (x, y, z) -> -2sin(3x) * sin(2y)
    for ax in 1:2   # x (direct) and y (transposed banded path)
        ng = ntuple(d -> d == ax ? SPLITN : 16, 3)
        solver = build_solver(
            Problem(domain=((0.0, 2π), (0.0, 2π), (0.0, 2π)), bcs=per3, ic=nothing),
            Numerics(n_global=ng, deriv=lele_d1_10(),
                     art=ArtificialProperties(enabled=false),
                     execution=Execution(dims=splitdims(ax))))
        f = CL.field(solver.decomp); df = CL.field(solver.decomp)
        fillf!(solver, f, fx)
        CL.exchange_halos!(f, solver.decomp)
        CL.deriv_along!(df, f, solver, ax, 1); CL._scale_grad!(df, solver, ax)
        ref = ax == 1 ? dfx : dfy
        check("C10 ∂/∂x$(ax) split $(np)-way", gmax(ferr(solver, df, ref)), 1e-8)
    end
    # Closed domain (SlipWallBC) with the derivative dimension split: closure
    # rows live on the two edge ranks, interior ranks carry V/W spikes. A deg-3
    # polynomial is exact through the C10 closure rows, under either set.
    sc = build_solver(
        Problem(domain=((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)),
                bcs=((SlipWallBC(), SlipWallBC()), per3[2], per3[3]), ic=nothing),
        Numerics(n_global=(SPLITN, 12, 12), deriv=lele_d1_10(),
                 art=ArtificialProperties(enabled=false),
                 execution=Execution(dims=splitdims(1))))
    fc = CL.field(sc.decomp); dfc = CL.field(sc.decomp)
    fillf!(sc, fc, (x, y, z) -> 1 + 2x + 3x^2 - x^3)
    CL.exchange_halos!(fc, sc.decomp)
    CL.deriv_along!(dfc, fc, sc, 1, 1); CL._scale_grad!(dfc, sc, 1)
    check("C10 closed-domain deg-3 exact, x split", gmax(ferr(sc, dfc,
          (x, y, z) -> 2 + 6x - 3x^2)), 1e-9)
end

# ---------------------------------------------------------------------------
# 3. Closed-domain (SlipWallBC) C6 polynomial exactness with the derivative
#    dimension split — closure rows on the edge ranks plus interior spikes.
# ---------------------------------------------------------------------------
function test_closed_c6()
    section("closed-domain C6 closures: polynomial exactness, derivative dim split")
    solver = build_solver(
        Problem(domain=((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)),
                bcs=((SlipWallBC(), SlipWallBC()), per3[2], per3[3]), ic=nothing),
        Numerics(n_global=(SPLITN, 12, 12), art=ArtificialProperties(enabled=false),
                 execution=Execution(dims=splitdims(1))))
    f = CL.field(solver.decomp); df = CL.field(solver.decomp)
    fillf!(solver, f, (x, y, z) -> 1 + 2x + 3x^2 - x^3)
    CL.exchange_halos!(f, solver.decomp)
    CL.deriv_along!(df, f, solver, 1, 1); CL._scale_grad!(df, solver, 1)
    check("C6 closed-domain deg-3 exact, x split", gmax(ferr(solver, df,
          (x, y, z) -> 2 + 6x - 3x^2)), 1e-9)
    # The Brady–Livescu rows on the edge ranks of the same split: their
    # polynomial exactness must survive the distributed solve, whose reduced
    # interface stage the rows' 1e3 closed-line condition number passes
    # through, at the same tolerance as the cascade above. Every rank builds
    # the plan, so the check is collective, as the one above is.
    for (label, deriv, deg) in
        (("C6 :brady_livescu", lele_d1_6(closures=:brady_livescu), 5),
         ("C8 :brady_livescu", lele_d1_8(closures=:brady_livescu), 7))
        solver = build_solver(
            Problem(domain=((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)),
                    bcs=((SlipWallBC(), SlipWallBC()), per3[2], per3[3]), ic=nothing),
            Numerics(n_global=(SPLITN, 12, 12), deriv=deriv,
                     art=ArtificialProperties(enabled=false),
                     execution=Execution(dims=splitdims(1))))
        f = CL.field(solver.decomp); df = CL.field(solver.decomp)
        fillf!(solver, f, (x, y, z) -> sum(x^m for m in 0:deg))
        CL.exchange_halos!(f, solver.decomp)
        CL.deriv_along!(df, f, solver, 1, 1); CL._scale_grad!(df, solver, 1)
        check("$label closed-domain deg-$deg exact, x split", gmax(ferr(solver, df,
              (x, y, z) -> sum(m * x^(m - 1) for m in 1:deg))), 1e-9)
    end
end

# ---------------------------------------------------------------------------
# 3b. AMR transfer pair (reference/AMR_GPU.md). The invertibility
#     identity prolong(restrict(f)) == f holds at any rank count, so applying
#     the pair along a split dimension pins the distributed tridiagonal solve
#     of the restriction filter and the pentadiagonal spike solve of the
#     deconvolution, edge-rank closure rows included, at machine precision —
#     any interface bug is an O(1) error. The TransferPlan check decomposes a
#     transverse dimension instead (the transfer dimension itself must not
#     be split; see transfer.jl) and asserts the coarse → fine → coarse left inverse.
# ---------------------------------------------------------------------------
function test_transfer_pair()
    section("AMR transfer operators: distributed pair and left inverse")
    restriction, prolongation = amr_transfer_schemes()
    for ax in (1, 2)
        n_global = ntuple(d -> d == ax ? SPLITN : 6, 3)
        decomp = Decomp(n_global, (false, false, false); dims=splitdims(ax))
        rplan = CL.plan_direction(decomp, restriction, ax, 1)
        pplan = CL.plan_direction(decomp, prolongation, ax, 1)
        f = CL.field(decomp); fbar = CL.field(decomp); back = CL.field(decomp)
        pad = decomp.n_halo_d; off = decomp.offset
        for k in 1:decomp.n_local[3], j in 1:decomp.n_local[2], i in 1:decomp.n_local[1]
            gx, gy, gz = i + off[1], j + off[2], k + off[3]
            f[i+pad[1], j+pad[2], k+pad[3]] =
                sin(0.37 * (gx, gy, gz)[ax]) + 0.5cos(0.11gx + 0.23gy + 0.05gz)
        end
        CL.exchange_dim!(f, decomp, ax)
        CL.apply_along!(fbar, rplan, f, decomp)
        CL.exchange_dim!(fbar, decomp, ax)
        CL.apply_along!(back, pplan, fbar, decomp)
        e = 0.0
        for k in 1:decomp.n_local[3], j in 1:decomp.n_local[2], i in 1:decomp.n_local[1]
            e = max(e, abs(back[i+pad[1], j+pad[2], k+pad[3]] -
                           f[i+pad[1], j+pad[2], k+pad[3]]))
        end
        check("pair round trip, dim $ax split", gmax(e), 1e-13)
    end

    nc = 17
    nf = 3nc - 2
    df = Decomp((nf, SPLITN, 1), (false, false, true); dims=splitdims(2))
    dc = Decomp((nc, SPLITN, 1), (false, false, true); dims=splitdims(2))
    tp = plan_transfer(df, dc, 1)
    coarse = CL.field(dc); coarse2 = CL.field(dc); fine = CL.field(df)
    pad = dc.n_halo_d; off = dc.offset
    for k in 1:dc.n_local[3], j in 1:dc.n_local[2], i in 1:dc.n_local[1]
        coarse[i+pad[1], j+pad[2], k+pad[3]] =
            exp(sin(0.5 * (i + off[1]))) + 0.1 * (j + off[2])
    end
    prolong!(fine, tp, coarse)
    restrict!(coarse2, tp, fine)
    e = 0.0
    for k in 1:dc.n_local[3], j in 1:dc.n_local[2], i in 1:dc.n_local[1]
        e = max(e, abs(coarse2[i+pad[1], j+pad[2], k+pad[3]] -
                       coarse[i+pad[1], j+pad[2], k+pad[3]]))
    end
    check("coarse->fine->coarse, transverse split", gmax(e), 1e-13)
end

# ---------------------------------------------------------------------------
# 4. Off-rank folds. Splitting the pairing (or reversed) dimension across ranks
#    forces the pair_forward!/pair_backward! block-Sendrecv butterfly that the
#    serial suite (always local_pair) never reaches.
# ---------------------------------------------------------------------------
function test_offrank_folds()
    section("off-rank folds: antipodal pairing butterfly across ranks")

    # (a) Resolved-θ cylindrical axis, θ split. The fold is on r (dim 1, one
    # rank); the antipodal partner is the θ+π slot, so splitting θ makes the
    # partner live on rank +np/2 → off-rank shift butterfly. f = r cosθ e^{−4r²}
    # is smooth through the axis and ODD under (r,θ)→(−r,θ) with the pairing.
    # θ count 48 is even and divisible by np∈{2,4,8}, and keeps ≥ 5 θ-points
    # per rank for the θ filter/derivative plans (≥ 9 at np = 8).
    solver = build_solver(
        Problem(domain=((0.0, 1.0), (0.0, 2π), (0.0, 1.0)), metric=CylindricalMetric(),
                bcs=((AxisBC(), SlipWallBC()), per3[2], per3[3]), ic=nothing),
        Numerics(n_global=(40, SPLITN, 1), art=ArtificialProperties(enabled=false),
                 execution=Execution(dims=splitdims(2))))
    f = CL.field(solver.decomp); df = CL.field(solver.decomp)
    # A scalar whose antipodal image is −f: σ = +1, the odd combination
    # carries all of it (see the fold_apply! comment on the mirror signs).
    fillf!(solver, f, (r, θ, z) -> r * cos(θ) * exp(-4r^2))
    CL.exchange_halos!(f, solver.decomp)
    CL.deriv_along!(df, f, solver, 1, 1); CL._scale_grad!(df, solver, 1)
    check("cyl axis ∂/∂r, θ split (off-rank shift)", gmax(ferr(solver, df,
          (r, θ, z) -> cos(θ) * (1 - 8r^2) * exp(-4r^2))), 1e-4)

    # (b) Spherical origin + poles, θ split. The origin fold reverses θ in its
    # partner map (revdim = 2), so splitting θ triggers the off-rank REVERSED
    # butterfly. f = e^{−4r²} is smooth at the origin and both poles.
    ssθ = build_solver(
        Problem(domain=((0.0, 1.0), (0.0, π), (0.0, 2π)), metric=SphericalMetric(),
                bcs=((OriginBC(), SlipWallBC()), (PoleBC(), PoleBC()), per3[3]), ic=nothing),
        Numerics(n_global=(40, SPLITN, 12), art=ArtificialProperties(enabled=false),
                 execution=Execution(dims=splitdims(2))))
    f = CL.field(ssθ.decomp); df = CL.field(ssθ.decomp)
    fillf!(ssθ, f, (r, θ, φ) -> exp(-4r^2))
    CL.exchange_halos!(f, ssθ.decomp)
    CL.deriv_along!(df, f, ssθ, 1, 1); CL._scale_grad!(df, ssθ, 1)
    check("sph origin ∂/∂r, θ split (off-rank reverse)", gmax(ferr(ssθ, df,
          (r, θ, φ) -> -8r * exp(-4r^2))), 1e-4)
    CL.deriv_along!(df, f, ssθ, 2, 1); CL._scale_grad!(df, ssθ, 2)
    check("sph poles (1/r)∂/∂θ = 0, θ split", gmax(ferr(ssθ, df,
          (r, θ, φ) -> 0.0)), 1e-8)

    # (c) The volume-weighted state filter through both off-rank folds. It
    # filters J·ρ with the parity of the product, odd at the axis (J = r) and
    # even at the origin (J = r²), and divides by the filtered J; a wrong
    # parity leaves an O(1) kink at the singular point, where a right one
    # changes the smooth density below by the filter's own truncation. The
    # wall closure rows at r = 1 are excluded from the measure.
    for (label, mk) in (
        ("cyl axis, θ split",
         () -> build_solver(
             Problem(domain=((0.0, 1.0), (0.0, 2π), (0.0, 1.0)), metric=CylindricalMetric(),
                     bcs=((AxisBC(), SlipWallBC()), per3[2], per3[3]), ic=nothing),
             Numerics(n_global=(40, SPLITN, 1), art=ArtificialProperties(enabled=false),
                      filter=StateFilter(weighting=:volume),
                      execution=Execution(dims=splitdims(2))))),
        ("sph origin+poles, θ split",
         () -> build_solver(Problem(domain=((0.0, 1.0), (0.0, π), (0.0, 2π)),
                                    metric=SphericalMetric(),
                                    bcs=((OriginBC(), SlipWallBC()), (PoleBC(), PoleBC()),
                                    per3[3]),
                                    ic=nothing),
                            Numerics(n_global=(40, SPLITN, 12),
                                     art=ArtificialProperties(enabled=false),
                                     filter=StateFilter(weighting=:volume),
                                     execution=Execution(dims=splitdims(2))))))
        sv = mk()
        Q = allocate_state(sv)
        initialize!(sv, Q, (r, θ, φ) -> Prim(u=(0, 0, 0), p=1.0,
                                             rho=1 + 0.5 * exp(-4r^2)))
        apply_bcs!(sv, Q)
        Q0 = copy(Q)
        filter_state!(sv, Q)
        m = 0.0
        for k in 1:sv.decomp.n_local[3], j in 1:sv.decomp.n_local[2],
            i in 1:sv.decomp.n_local[1]
            xcoord(sv, 1, i) < 0.5 || continue
            I = padded_index(sv, i, j, k)
            m = max(m, abs(Q[I, 1] - Q0[I, 1]))
        end
        check("volume-weighted filter through the fold: " * label, gmax(m), 1e-6)
    end

    # (d) The :delta4 detector through the same butterfly. `delta4_sum!` takes
    # every sensed field across a paired fold as the even and odd combinations
    # of the line and its antipodal partner, so the off-rank exchange carries
    # the sensors as well as the derivatives. Each case is compared with the
    # same rank's serial rebuild on COMM_SELF, first the detector alone on an
    # even and an odd field, then the artificial coefficients, the right-hand
    # side and a few steps of a flow crossing the axis. The second process
    # grid splits r as well, so the ranks away from the axis take the
    # butterfly without holding the fold (at np = 2 with the pair local).
    rdims = (2, np ÷ 2, 1)
    function detector_diff(s, ref, a, b, ncomp)
        e = 0.0
        scale = 0.0
        for I in CL.interior(s.decomp), c in 1:ncomp
            loc = Tuple(I) .- s.decomp.n_halo_d
            J = padded_index(ref, (loc .+ s.decomp.offset)...)
            e = max(e, abs(a[I, c] - b[J, c]))
            scale = max(scale, abs(b[J, c]))
        end
        return gmax(e) / gmax(scale)
    end
    cyl(comm_here, dims_here; art=ArtificialProperties(enabled=false)) =
        build_solver(
            Problem(domain=((0.0, 1.0), (0.0, 2π), (0.0, 1.0)), metric=CylindricalMetric(),
                    bcs=((AxisBC(), SlipWallBC()), per3[2], per3[3]),
                    transport=ConstantTransport(mu0=1e-3), ic=nothing),
            Numerics(n_global=(40, SPLITN, 1), art=art,
                     execution=Execution(comm=comm_here, dims=dims_here)))
    sph(comm_here, dims_here) =
        build_solver(
            Problem(domain=((0.0, 1.0), (0.0, π), (0.0, 2π)), metric=SphericalMetric(),
                    bcs=((OriginBC(), SlipWallBC()), (PoleBC(), PoleBC()), per3[3]),
                    ic=nothing),
            Numerics(n_global=(40, SPLITN, 12), art=ArtificialProperties(enabled=false),
                     execution=Execution(comm=comm_here, dims=dims_here)))
    # A smooth scalar (the Cartesian coordinate along the pairing, σ = +1)
    # and the radial component of a uniform Cartesian flow, which changes
    # sign across the fold (σ = −1).
    fields = (("even", (r, θ, z) -> exp(-4r^2) * (1 + r * cos(θ)), (1, 1, 1)),
              ("odd", (r, θ, z) -> cos(θ) * (1 + r^2), (-1, 1, 1)))
    for (label, mk, dims_here) in (("cyl axis, θ split", cyl, splitdims(2)),
                                   ("cyl axis, r and θ split", cyl, rdims),
                                   ("sph origin+poles, θ split", sph, splitdims(2)))
        s = mk(comm, dims_here)
        ref = mk(MPI.COMM_SELF, (1, 1, 1))
        for (parity_label, fn, parity) in fields
            out = map((s, ref)) do sv
                f = CL.field(sv.decomp)
                sensed = CL.field(sv.decomp)
                fillf!(sv, f, fn)
                CL.exchange_halos!(f, sv.decomp)
                CL.delta4_sum!(sensed, f, sv, 2; parity=parity)
            end
            check("δ⁴ detector, $parity_label field, $label",
                  detector_diff(s, ref, out[1], out[2], 1), 1e-12)
        end
    end
    # A uniform Cartesian stream through the axis with a radial compression
    # on it. The `:velocity` μ* sensor takes each velocity component across
    # the fold with its own sign, the `:strain` one the strain magnitude.
    flow = (r, θ, z) -> Prim(rho=1 + 0.5 * exp(-4r^2), p=1 + 0.2 * exp(-8r^2),
                             u=(0.3 * cos(θ) - 0.2 * r * exp(-4r^2),
                                -0.3 * sin(θ), 0.0))
    for (label, dims_here) in (("θ split", splitdims(2)), ("r and θ split", rdims)),
        mu_sensor in (:strain, :velocity)
        art = ArtificialProperties(enabled=true, mu_sensor=mu_sensor)
        label = "$label, $mu_sensor"
        s = cyl(comm, dims_here; art=art)
        ref = cyl(MPI.COMM_SELF, (1, 1, 1); art=art)
        states = map((s, ref)) do sv
            Q = allocate_state(sv)
            initialize!(sv, Q, flow)
            apply_bcs!(sv, Q)
            dQ = zero(Q)
            compute_rhs!(sv, Q, dQ)
            (Q, dQ)
        end
        for name in (:mu_art, :beta_art, :kappa_art)
            check("$name through the axis, $label",
                  detector_diff(s, ref, getproperty(s, name),
                                getproperty(ref, name), 1), 1e-10)
        end
        ncons = s.equations.n_cons
        check("RHS with the detector through the axis, $label",
              detector_diff(s, ref, states[1][2], states[2][2], ncons), 1e-10)
        run!(s, states[1][1]; tfinal=1e9, nmax=3)
        run!(ref, states[2][1]; tfinal=1e9, nmax=3)
        check("three steps through the axis, $label",
              detector_diff(s, ref, states[1][1], states[2][1], ncons), 1e-10)
    end
end

# ---------------------------------------------------------------------------
# 4a. Face-centred symmetry planes on a SPLIT dimension. The plane's fold is
#     self-paired, so section 4's butterfly never runs here; the new path is a
#     fold whose OWN dimension is decomposed. The mirror halo then exists on
#     the two end ranks alone while every rank runs the folded line solve, the
#     folded filter and the folded sensor operators, so a rank that skipped
#     them for holding no plane would hang its peers rather than answer
#     wrongly. Each case is measured against the same rank's serial rebuild
#     on COMM_SELF.
# ---------------------------------------------------------------------------
function test_symmetry_plane()
    section("symmetry plane: folded operators on a decomposed dimension")
    sym = (SymmetryPlaneBC(), SymmetryPlaneBC())
    # Max |distributed − serial| over this rank's interior block, for a state
    # or a right-hand side. The serial rebuild spans the global grid, so the
    # local index reaches it through this rank's offset.
    function blockdiff(s, a, ref, b)
        e = 0.0
        for I in CL.interior(s.decomp), c in 1:s.equations.n_cons
            loc = Tuple(I) .- s.decomp.n_halo_d
            J = padded_index(ref, (loc .+ s.decomp.offset)...)
            e = max(e, abs(Float64(a[I, c] - b[J, c])))
        end
        e
    end
    # The same case twice: over a real process grid on COMM_WORLD, and whole
    # on COMM_SELF. Every rank builds both, so no rank can skip a distributed
    # solve its peers are waiting in.
    function build(T, ng, bcs, ic, detector, channel, comm_here, dims_here)
        sol = build_solver(Problem(domain=((0.0, one(T)), (0.0, one(T)), (0.0, one(T))),
                                   bcs=bcs,
                                   eos=IdealMixture([IdealSpecies{T}("a", T(1), T(1.4)),
                                                IdealSpecies{T}("b", T(0.7), T(1.3))]),
                                   transport=ConstantTransport{T}(mu0=T(0.01)), ic=nothing),
                           Numerics(n_global=ng,
                                    art=ArtificialProperties{T}(enabled=true,
                                                                detector=detector,
                                                                species_flux=channel),
                                    deriv=lele_d1_6(T), cfl=T(0.4),
                                    filter=StateFilter(compact_filter(T(0.45), T),
                                                       interval=1, cfl=zero(T)),
                                    execution=Execution(comm=comm_here, dims=dims_here)))
        Q = allocate_state(sol)
        initialize!(sol, Q, ic)
        apply_bcs!(sol, Q)
        dQ = zero(Q)
        compute_rhs!(sol, Q, dQ)
        return sol, Q, dQ
    end
    # One right-hand side and then a few filtered steps, each against the
    # serial rebuild: the divergence alone would not catch a fold that the
    # update or the state filter reaches differently on a split dimension.
    function compare(label, T, ng, bcs, ic, dims_here;
                     detector=:delta4, channel=:fickian, nmax=4)
        s, Q, dQ = build(T, ng, bcs, ic, detector, channel, comm, dims_here)
        ref, Qref, dQref = build(T, ng, bcs, ic, detector, channel,
                                 MPI.COMM_SELF, (1, 1, 1))
        tol = T === Float32 ? 5e-4 : 1e-8
        check("folded distributed RHS $label",
              gmax(blockdiff(s, dQ, ref, dQref)), tol)
        run!(s, Q; tfinal=1e9, nmax=nmax)
        run!(ref, Qref; tfinal=1e9, nmax=nmax)
        check("folded distributed run $label",
              gmax(blockdiff(s, Q, ref, Qref)), tol)
        return s
    end

    # A plane at both ends of the split dimension, one axis at a time. The
    # data carries the parity the plane imposes: cos(πr) and cos(2πr) are even
    # about r = 0 and r = 1, sin(πr) is odd about both, and the tangential
    # velocity rides across the fold as an even component.
    for T in (Float64, Float32), ax in 1:3
        ic = (x, y, z) -> begin
            r = (x, y, z)[ax]
            a = T(0.4) + T(0.1) * cos(T(2pi) * r)
            vel = ntuple(d -> d == ax ? T(0.1) * sin(T(pi) * r) :
                                        T(0.08) * cos(T(pi) * r), 3)
            Prim(rho=one(T) + T(0.05) * cos(T(pi) * r),
                 T_ion=T(2) + T(0.1) * cos(T(pi) * r), Y=(a, 1 - a), u=vel)
        end
        ng = ntuple(d -> d == ax ? SPLITN : 1, 3)
        bcs = ntuple(d -> d == ax ? sym : per3[d], 3)
        s = compare("$T dim=$ax", T, ng, bcs, ic, splitdims(ax);
                    detector=(ax == 3 ? :d8 : :delta4),
                    channel=(:partial_density, :bulk, :fickian)[ax])
        # At np > 2 the middle ranks of the split hold neither plane and reach
        # the folded solve only because nothing on the path returns early.
        owns_neither = all(sd -> CL.wallplane(s.decomp, ax, sd) === nothing, 1:2)
        check("interior ranks participate $T dim=$ax",
              abs(gsum(Int(owns_neither)) - max(np - 2, 0)), 0.5)
    end

    # A corner: planes at both ends of two dimensions with the process grid
    # split along both, so each fold's own dimension is decomposed and the two
    # mirror fills meet in the halo corner no message fills.
    cdims = iseven(np) && np >= 4 ? (2, np ÷ 2, 1) : (np, 1, 1)
    cic = (x, y, z) -> begin
        a = 0.4 + 0.1 * cos(2pi * x) * cos(2pi * y)
        Prim(rho=1.05 + 0.05 * cos(pi * x) * cos(pi * y),
             T_ion=2.0 + 0.1 * cos(pi * x), Y=(a, 1 - a),
             u=(0.1 * sin(pi * x) * cos(pi * y),
                0.08 * cos(pi * x) * sin(pi * y), 0.0))
    end
    compare("corner dims=$cdims", Float64, (SPLITN, 40, 1),
            (sym, sym, per3[3]), cic, cdims; nmax=3)

    # The layout a planar Noh run takes: the plane at the high end alone, a
    # Dirichlet end at the low end, and the split between the two. The folded
    # end takes no closure row while the Dirichlet end keeps its own.
    inflow = Prim(rho=1.0, T_ion=2.0, Y=(0.4, 0.6), u=(-0.2, 0.0, 0.0))
    nbcs = ((DirichletBC((x, y, z, t) -> inflow), SymmetryPlaneBC()),
            per3[2], per3[3])
    nic = (x, y, z) -> Prim(rho=1.0 + 0.05 * x, T_ion=2.0, Y=(0.4, 0.6),
                            u=(-0.2, 0.0, 0.0))
    compare("Dirichlet-to-plane dim=1", Float64, (SPLITN, 1, 1), nbcs, nic,
            splitdims(1))
end

# ---------------------------------------------------------------------------
# 4b. Halo-exchange consistency across rank boundaries. exchange_dim_batch! and
#     exchange_halos! share the per-dimension buffers (the batched path grows
#     them), so the two must agree slab-for-slab; and because dimensions are
#     exchanged sequentially with previously-filled halos included in each
#     slab, the diagonal CORNER halos must come out right with no dedicated
#     diagonal messages. Both are no-ops at np == 1.
# ---------------------------------------------------------------------------
function test_halo_consistency()
    section("halo exchange: batched ≡ per-array, and corner halos")
    fn = (x, y, z) -> sin(x) + 2cos(y) - z^0 * 0.3

    # (a) Batched vs per-array over all three dimensions, with the batch run
    # FIRST so its (larger, multi-component) buffer is the one the scalar path
    # then reuses — exchange_halos! must still transfer exactly one slab.
    # Comparison covers the full padded arrays, halos included.
    for ax in 1:3
        ng = ntuple(d -> d == ax ? SPLITN : 16, 3)
        solver = build_solver(
            Problem(domain=((0.0, 2π), (0.0, 2π), (0.0, 2π)), bcs=per3, ic=nothing),
            Numerics(n_global=ng, art=ArtificialProperties(enabled=false),
                     execution=Execution(dims=splitdims(ax))))
        a = CL.field(solver.decomp); b = CL.field(solver.decomp); c = CL.field(solver.decomp)
        fillf!(solver, a, fn); fillf!(solver, b, fn); fillf!(solver, c, fn)
        for d in 1:3
            CL.exchange_dim_batch!([b, c], solver.decomp, d)   # grows the shared buffer
        end
        CL.exchange_halos!(a, solver.decomp)                   # must still send one slab
        e = maximum(abs, a .- b)
        check("batched ≡ per-array, dim $ax split $(np)-way", gmax(e), 1e-14)
    end

    # (b) Corner halo: after the three sequential exchanges, the diagonal halo
    # points must hold the analytic value at their own physical coordinates —
    # only true if the y and z exchanges carried the already-filled x halo
    # layers. Uses the DEFAULT process grid (dims=nothing) so the automatic
    # Dims_create path is covered too; at np ≥ 4 that is a genuine 2-D/3-D
    # grid with real diagonal neighbours.
    solver = build_solver(
        Problem(domain=((0.0, 2π), (0.0, 2π), (0.0, 2π)), bcs=per3, ic=nothing),
        Numerics(n_global=(24, 24, 24), art=ArtificialProperties(enabled=false)))
    f = CL.field(solver.decomp)
    fillf!(solver, f, fn)
    CL.exchange_halos!(f, solver.decomp)
    e = 0.0
    for kk in (0, solver.decomp.n_local[3] + 1), jj in (0, solver.decomp.n_local[2] + 1),
        ii in (0, solver.decomp.n_local[1] + 1)
        e = max(e, abs(f[padded_index(solver, ii, jj, kk)] -
                       fn(xcoord(solver, 1, ii), xcoord(solver, 2, jj), xcoord(solver, 3, kk))))
    end
    check("corner halos filled (all 8 corners)", gmax(e), 1e-12)
end

# ---------------------------------------------------------------------------
# 5. Freestream preservation (uniform state ⇒ dQ ≈ 0), decomposed. Divergence,
#    metric sources, and the fold-aware geometry must stay consistent to
#    machine zero even when a fold's antipodal partner, or the θ line through
#    the poles, lies on another rank.
# ---------------------------------------------------------------------------
function test_freestream()
    section("freestream preservation: uniform state ⇒ dQ ≈ 0, decomposed")
    cases = [
        (; name="cartesian, x split",
           n_global=(SPLITN, 12, 12), L_domain=(1.0, 1.0, 1.0), metric=CartesianMetric(),
           bcs=per3, dims=splitdims(1), kw=(;)),
        (; name="cartesian stretched, x split",
           n_global=(SPLITN, 12, 12), L_domain=(1.0, 1.0, 1.0), metric=CartesianMetric(),
           bcs=((SlipWallBC(), SlipWallBC()), per3[2], per3[3]), dims=splitdims(1),
           kw=(; stretch=(sine_cluster(0.0, 1.0, 0.5, 0.4), nothing, nothing))),
        (; name="cylindrical off-axis, θ split",
           n_global=(12, SPLITN, 12), L_domain=(1.0, 2π, 0.5), metric=CylindricalMetric(),
           bcs=((SlipWallBC(), SlipWallBC()), per3[2], per3[3]), dims=splitdims(2),
           kw=(; origin=(0.2, 0.0, 0.0))),
        (; name="cylindrical resolved-θ axis, θ split",
           n_global=(40, SPLITN, 1), L_domain=(1.0, 2π, 1.0), metric=CylindricalMetric(),
           bcs=((AxisBC(), SlipWallBC()), per3[2], per3[3]), dims=splitdims(2),
           kw=(;)),
        (; name="spherical origin+poles, θ split (revdim off-rank)",
           n_global=(40, SPLITN, 12), L_domain=(1.0, π, 2π), metric=SphericalMetric(),
           bcs=((OriginBC(), SlipWallBC()), (PoleBC(), PoleBC()), per3[3]),
           dims=splitdims(2), kw=(;)),
        (; name="spherical origin+poles, φ split (pdim off-rank)",
           n_global=(40, 16, SPLITN), L_domain=(1.0, π, 2π), metric=SphericalMetric(),
           bcs=((OriginBC(), SlipWallBC()), (PoleBC(), PoleBC()), per3[3]),
           dims=splitdims(3), kw=(;)),
        # The radial pressure gradient of a stretched annulus solved across
        # ranks, and the sources of the collapsed dimensions, which carry no
        # pressure.
        (; name="cylindrical stretched annulus, r split",
           n_global=(SPLITN, 12, 1), L_domain=(1.0, 2π, 1.0), metric=CylindricalMetric(),
           bcs=((SlipWallBC(), SlipWallBC()), per3[2], per3[3]), dims=splitdims(1),
           kw=(; origin=(0.2, 0.0, 0.0),
                 stretch=(sine_cluster(0.2, 1.2, 0.5, 0.4), nothing, nothing))),
        (; name="cylindrical r collapsed, θ split",
           n_global=(1, SPLITN, 1), L_domain=(1.0, 2π, 1.0), metric=CylindricalMetric(),
           bcs=per3, dims=splitdims(2), kw=(; origin=(1.0, 0.0, 0.0))),
        (; name="spherical r collapsed, θ split",
           n_global=(1, SPLITN, 1), L_domain=(1.0, π/2, 1.0), metric=SphericalMetric(),
           bcs=(per3[1], (SlipWallBC(), SlipWallBC()), per3[3]), dims=splitdims(2),
           kw=(; origin=(1.0, π/4, 0.0))),
        (; name="spherical θ collapsed at π/2, r split",
           n_global=(SPLITN, 1, 1), L_domain=(1.0, 1.0, 1.0), metric=SphericalMetric(),
           bcs=((SlipWallBC(), SlipWallBC()), per3[2], per3[3]), dims=splitdims(1),
           kw=(; origin=(0.2, π/2, 0.0))),
    ]
    for cs in cases
        o = merge((; origin=(0.0, 0.0, 0.0), stretch=(nothing, nothing, nothing)), cs.kw)
        domain = ntuple(d -> (o.origin[d], o.origin[d] + cs.L_domain[d]), 3)
        build() = build_solver(Problem(; domain, metric=cs.metric, bcs=cs.bcs, ic=nothing),
                               Numerics(; n_global=cs.n_global, stretch=o.stretch,
                                        art=ArtificialProperties(enabled=false),
                                        filter=StateFilter(weighting=:volume),
                                        execution=Execution(dims=cs.dims)))
        # A collapsed r warns once, from rank 0, and nothing else warns.
        solver, warnings = recorded_warnings(build)
        if cs.n_global[1] == 1
            expected = rank == 0 ? 1 : 0
            collapsed = count(m -> occursin("r is collapsed", m), warnings)
            check(cs.name * ": one setup warning, from rank 0",
                  gmax(abs(length(warnings) - expected) + abs(collapsed - expected)), 0.5)
        end
        Q = allocate_state(solver)
        initialize!(solver, Q, (x, y, z) -> Prim(u=(0, 0, 0), p=1.0, rho=1.0))
        apply_bcs!(solver, Q)
        dQ = zero(Q)
        compute_rhs!(solver, Q, dQ)
        m = 0.0
        for c in 1:solver.equations.n_cons, k in 1:solver.decomp.n_local[3],
            j in 1:solver.decomp.n_local[2], i in 1:solver.decomp.n_local[1]
            m = max(m, abs(dQ[padded_index(solver, i, j, k), c]))
        end
        check(cs.name, gmax(m), 1e-8)
        # The volume-weighted filter divides F(J) by itself, so a uniform
        # state survives a pass on every metric; across a fold split over
        # ranks that also exercises the parity the product J·q folds with.
        Q0 = copy(Q)
        filter_state!(solver, Q)
        m = 0.0
        for c in 1:solver.equations.n_cons, k in 1:solver.decomp.n_local[3],
            j in 1:solver.decomp.n_local[2], i in 1:solver.decomp.n_local[1]
            I = padded_index(solver, i, j, k)
            m = max(m, abs(Q[I, c] - Q0[I, c]))
        end
        check(cs.name * ": volume-weighted filter holds a uniform state",
              gmax(m), 1e-12)
    end
end

# ---------------------------------------------------------------------------
# 6. Conservation: a periodic RHS must integrate to zero globally — a genuine
#    telescoping-flux check across rank boundaries (global MPI.Reduce sum).
# ---------------------------------------------------------------------------
function test_conservation()
    section("conservation: periodic RHS integrates to zero (global sum)")
    N = SPLITN                   # ≥ 9 per rank up to np = 8 (filter closure)
    solver = build_solver(
        Problem(domain=((0.0, 2π), (0.0, 2π), (0.0, 2π)), bcs=per3,
                transport=ConstantTransport(mu0=1e-3), ic=nothing),
        Numerics(n_global=(N, 16, 16), art=ArtificialProperties(enabled=false),
                 execution=Execution(dims=splitdims(1))))
    Q = allocate_state(solver)
    initialize!(solver, Q, (x, y, z) ->
        Prim(u=(0.1sin(x)cos(y), -0.1cos(x)sin(y), 0.05sin(z)),
             p=1 + 0.05cos(x)cos(z), rho=1 + 0.1sin(y)))
    dQ = zero(Q)
    compute_rhs!(solver, Q, dQ)
    worst = 0.0
    npts = N * 16 * 16
    for c in 1:solver.equations.n_cons
        loc = 0.0
        for k in 1:solver.decomp.n_local[3], j in 1:solver.decomp.n_local[2], i in 1:solver.decomp.n_local[1]
            loc += dQ[padded_index(solver, i, j, k), c]
        end
        worst = max(worst, abs(gsum(loc)))
    end
    check("Σ dQ over global periodic domain", worst, 1e-8 * npts)
end

# ---------------------------------------------------------------------------
# 7. Global synchronization. Everything above checks a spatial answer; these
#    check that the ranks stay in lockstep. compute_dt and the RK loop both
#    Allreduce, so a missing or mis-ordered reduction shows up as ranks
#    disagreeing about dt or drifting apart in time — which no single-rank
#    test can see, and which corrupts a long run silently.
# ---------------------------------------------------------------------------
function test_sync()
    section("global synchronization: dt and time identical on every rank")
    solver = build_solver(
        Problem(domain=((0.0, 2π), (0.0, 2π), (0.0, 2π)), bcs=per3, ic=nothing),
        Numerics(n_global=(SPLITN, 16, 16), art=ArtificialProperties(enabled=false),
                 execution=Execution(dims=splitdims(1))))
    Q = allocate_state(solver)
    initialize!(solver, Q, (x, y, z) ->
        Prim(u=(0.2sin(x), 0, 0), p=1 + 0.1cos(y), rho=1 + 0.2sin(z)))
    dt = compute_dt(solver, Q)
    spread = MPI.Allreduce(dt, max, comm) - MPI.Allreduce(dt, min, comm)
    check("compute_dt spread across ranks", spread, 1e-15 * dt)

    # Ten RK steps of a real transient: every rank must finish at the same t
    # and nothing may go non-finite.
    s2, Q2 = setup(Problem(domain=((0.0, 1.0), (0.0, 0.25), (0.0, 0.25)),
                           bcs=((SlipWallBC(), NSCBCOutflowBC(pinf=1.0)),
                                per3[2], per3[3]),
                           ic=(x, y, z) -> Prim(u=(0, 0, 0),
                                                p=1 + 4exp(-200(x - 0.3)^2),
                                                rho=1 + exp(-200(x - 0.3)^2))),
                   Numerics(n_global=(SPLITN, 16, 16), execution=Execution(dims=splitdims(1))))
    run!(s2, Q2; tfinal=1e9, nmax=10)
    nbad = 0.0
    for c in 1:s2.equations.n_cons, k in 1:s2.decomp.n_local[3], j in 1:s2.decomp.n_local[2],
        i in 1:s2.decomp.n_local[1]
        isfinite(Q2[padded_index(s2, i, j, k), c]) || (nbad += 1)
    end
    check("10 RK steps stay finite (count of non-finite)", gsum(nbad), 0.5)
    tspread = MPI.Allreduce(s2.t, max, comm) - MPI.Allreduce(s2.t, min, comm)
    check("simulation time spread after 10 steps", tspread, 1e-14 * max(s2.t, 1e-30))
end
