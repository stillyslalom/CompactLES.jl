# Included by mpi_tests.jl after its MPI harness definitions.

# The positivity limiter on a decomposed line: the running sums meet across
# rank boundaries through the gathered line totals, one closed direction
# (anchored at the wall flux) and one periodic (anchored through the face
# relation), split in turn. A blast in a box limited from its first steps,
# against the same run on one rank, which every rank computes for itself.
function test_positivity_limiter()
    section("positivity limiter: decomposed against one rank")
    gas = IdealSpecies("gas"; gamma=1.4, R=1.0)
    ic(x, y, z) = Prim(rho=1.0, u=(0.0, 0.0, 0.0),
                       p=1e-3 + exp(-((x - 0.45)^2 + (y - 0.5)^2) / 0.004))
    function make(comm_here, dims_here)
        s = Solver(n_global=(SPLITN, SPLITN, 1), L_domain=(1.0, 1.0, 1.0),
                   bcs=((SlipWallBC(), SlipWallBC()), per3[2], per3[3]), eos=gas,
                   art=ArtificialProperties(enabled=true), cfl=0.4,
                   control=StepControl(validity=:permissive), positivity_limiter=true,
                   comm=comm_here, dims=dims_here)
        Q = allocate_state(s)
        initialize!(s, Q, ic)
        run!(s, Q; tfinal=1.0, nmax=40)
        return s, Q
    end
    ref, Qref = make(MPI.COMM_SELF, (1, 1, 1))
    cref = CL.positivity_counts(ref)
    for ax in 1:2
        s, Q = make(comm, splitdims(ax))
        c = CL.positivity_counts(s)
        off = s.decomp.offset
        worst = 0.0
        for j in 1:s.decomp.n_local[2], i in 1:s.decomp.n_local[1], k in 1:s.equations.n_cons
            a = Q[padded_index(s, i, j, 1), k]
            b = Qref[padded_index(ref, i + off[1], j + off[2], 1), k]
            worst = max(worst, abs(a - b))
        end
        scale = maximum(abs, parent(Qref))
        # The running sums gathered across ranks differ from the serial ones in
        # the last bits, and θ, which falls to zero where a side's first-order
        # half state leaves its bound, is not continuous in them: the state
        # differs by 1.6e-10 after 40 steps at np = 2 and by 5e-15 after 20.
        check("split axis $ax: state against one rank (relative)", gmax(worst) / scale,
              1e-8)
        check("split axis $ax: clock against one rank", abs(s.t - ref.t) / ref.t, 1e-8)
        check("split axis $ax: stage faces limited (relative difference)",
              abs(c.stage_limited - cref.stage_limited) / cref.stage_limited, 1e-2)
        check("split axis $ax: faces tested, counted once", abs(c.stage_faces -
              cref.stage_faces) + abs(c.filter_faces - cref.filter_faces), 0.5)
        report = state_report(s, Q)
        check("split axis $ax: inadmissible points at the end",
              report.inadmissible + report.negative_density, 0.5)
    end
    check("one rank: the limiter acted (expect > 0)",
          cref.stage_limited > 0 && cref.filter_limited > 0 ? 0.0 : 1.0, 0.5)

    # A blast through the spherical origin, the radial line split across every
    # rank: the fold's rank holds the face register at r = 0, and the face two
    # ranks share takes the same θ on both from the exchanged cell rates.
    section("positivity limiter: a radial line split across ranks against one rank")
    N = 96
    blast(r, θ, φ) = Prim(rho=1.0, u=(0.0, 0.0, 0.0), p=1e-4 + exp(-(r / 0.08)^2))
    function sphere(comm_here, dims_here)
        s = Solver(n_global=(N, 1, 1), L_domain=(1.0, 1.0, 1.0),
                   origin=(0.0, π / 2, 0.0), metric=SphericalMetric(),
                   bcs=((OriginBC(), SlipWallBC()), per3[2], per3[3]), eos=gas,
                   art=ArtificialProperties(enabled=true), cfl=0.3,
                   control=StepControl(validity=:permissive), positivity_limiter=true,
                   comm=comm_here, dims=dims_here)
        Q = allocate_state(s)
        initialize!(s, Q, blast)
        run!(s, Q; tfinal=1.0, nmax=60)
        return s, Q
    end
    sref, Qsref = sphere(MPI.COMM_SELF, (1, 1, 1))
    csref = CL.positivity_counts(sref)
    s, Q = sphere(comm, splitdims(1))
    c = CL.positivity_counts(s)
    worst = 0.0
    for i in 1:s.decomp.n_local[1], k in 1:s.equations.n_cons
        a = Q[padded_index(s, i, 1, 1), k]
        b = Qsref[padded_index(sref, i + s.decomp.offset[1], 1, 1), k]
        worst = max(worst, abs(a - b))
    end
    check("radial split: state against one rank (relative)",
          gmax(worst) / maximum(abs, parent(Qsref)), 1e-12)
    check("radial split: clock against one rank", abs(s.t - sref.t) / sref.t, 1e-12)
    check("radial split: stage faces limited (relative difference)",
          abs(c.stage_limited - csref.stage_limited) / max(csref.stage_limited, 1), 1e-2)
    check("radial split: faces tested, counted once",
          abs(c.stage_faces - csref.stage_faces) +
          abs(c.filter_faces - csref.filter_faces), 0.5)
    report = state_report(s, Q)
    check("radial split: inadmissible points at the end",
          report.inadmissible + report.negative_density, 0.5)
    check("radial, one rank: the limiter acted (expect > 0)",
          csref.stage_limited > 0 && csref.filter_limited > 0 ? 0.0 : 1.0, 0.5)

    # A blast at the corner of the r-z axis and a symmetry plane at z = 0,
    # split along r and along z in turn: the rates the z faces read cross the
    # rank boundary with the radial pass's exchange, and the plane's face
    # register sits on the rank holding z = 0.
    section("positivity limiter: the r-z plane split across ranks against one rank")
    corner(r, θ, z) = Prim(rho=1.0, u=(0.0, 0.0, 0.0),
                           p=1e-4 + exp(-(r^2 + z^2) / 0.06^2))
    function plane(comm_here, dims_here)
        s = Solver(n_global=(SPLITN, 1, SPLITN), L_domain=(1.0, 1.0, 1.0),
                   metric=CylindricalMetric(),
                   bcs=((AxisBC(), SlipWallBC()), per3[2], (SymmetryPlaneBC(), SlipWallBC())),
                   eos=gas, art=ArtificialProperties(enabled=true), cfl=0.3,
                   control=StepControl(validity=:permissive), positivity_limiter=true,
                   comm=comm_here, dims=dims_here)
        Q = allocate_state(s)
        initialize!(s, Q, corner)
        run!(s, Q; tfinal=1.0, nmax=40)
        return s, Q
    end
    pref, Qpref = plane(MPI.COMM_SELF, (1, 1, 1))
    cpref = CL.positivity_counts(pref)
    for ax in (1, 3)
        s, Q = plane(comm, splitdims(ax))
        c = CL.positivity_counts(s)
        off = s.decomp.offset
        worst = 0.0
        for k in 1:s.decomp.n_local[3], i in 1:s.decomp.n_local[1], q in 1:s.equations.n_cons
            a = Q[padded_index(s, i, 1, k), q]
            b = Qpref[padded_index(pref, i + off[1], 1, k + off[3]), q]
            worst = max(worst, abs(a - b))
        end
        # θ is not continuous in the gathered running sums' last bits, as in
        # the Cartesian case above: 1.9e-12 at np = 2 split along r.
        check("r-z split axis $ax: state against one rank (relative)",
              gmax(worst) / maximum(abs, parent(Qpref)), 1e-8)
        check("r-z split axis $ax: stage faces limited (relative difference)",
              abs(c.stage_limited - cpref.stage_limited) / max(cpref.stage_limited, 1),
              1e-2)
        check("r-z split axis $ax: faces tested, counted once",
              abs(c.stage_faces - cpref.stage_faces) +
              abs(c.filter_faces - cpref.filter_faces), 0.5)
        report = state_report(s, Q)
        check("r-z split axis $ax: inadmissible points at the end",
              report.inadmissible + report.negative_density, 0.5)
    end
    check("r-z, one rank: the limiter acted (expect > 0)",
          cpref.stage_limited > 0 ? 0.0 : 1.0, 0.5)

    # Two same-level patches split along x, each on its own share of the
    # ranks and decomposed within it at np > 2, against both patches on one
    # rank: a patch's lines end at the interface as at a wall, and nothing of
    # the limiter crosses it.
    section("positivity limiter: same-level patches across ranks against one rank")
    function patched(comm_here)
        s = Solver(n_global=(96, 32, 1), L_domain=(1.0, 1.0, 1.0),
                   bcs=((SlipWallBC(), SlipWallBC()), per3[2], per3[3]), eos=gas,
                   art=ArtificialProperties(enabled=true), cfl=0.4,
                   control=StepControl(validity=:permissive), positivity_limiter=true,
                   patch_grid=(2, 1, 1), interface_flux=:closure, comm=comm_here)
        Q = allocate_state(s)
        initialize!(s, Q, (x, y, z) -> Prim(rho=1.0, u=(0.0, 0.0, 0.0),
                                            p=1e-3 + exp(-((x - 0.5)^2 + (y - 0.5)^2) /
                                                         0.004)))
        run!(s, Q; tfinal=1.0, nmax=40)
        return s, Q
    end
    tref, Qtref = patched(MPI.COMM_SELF)
    ctref = CL.positivity_counts(tref)
    s, Q = patched(comm)
    c = CL.positivity_counts(s)
    worst = 0.0
    for (ps, q) in CL.eachpatch(s, Q)
        pref = getfield(tref, :patches)[ps.patch.id]
        qref = Qtref[ps.patch.id]
        off = ps.decomp.offset
        for j in 1:ps.decomp.n_local[2], i in 1:ps.decomp.n_local[1],
            k in 1:s.equations.n_cons
            a = q[padded_index(ps, i, j, 1), k]
            b = qref[padded_index(CL.PatchSolver(tref, pref), i + off[1], j + off[2], 1), k]
            worst = max(worst, abs(a - b))
        end
    end
    scale = maximum(q -> maximum(abs, parent(q)), Qtref)
    check("patches: state against one rank (relative)", gmax(worst) / scale, 1e-8)
    check("patches: clock against one rank", abs(s.t - tref.t) / tref.t, 1e-8)
    check("patches: stage faces limited (relative difference)",
          abs(c.stage_limited - ctref.stage_limited) / max(ctref.stage_limited, 1), 1e-2)
    check("patches: faces tested, counted once",
          abs(c.stage_faces - ctref.stage_faces) +
          abs(c.filter_faces - ctref.filter_faces), 0.5)
    report = state_report(s, Q)
    check("patches: inadmissible points at the end",
          report.inadmissible + report.negative_density, 0.5)
    check("patches, one rank: the limiter acted (expect > 0)",
          ctref.stage_limited > 0 && ctref.filter_limited > 0 ? 0.0 : 1.0, 0.5)

    # The same decomposed runs on device storage, under FORCE_KA and
    # FORCE_DEVICE_EXCHANGE, against the host runs on the same ranks: the
    # line planes cross to the host for the gathered offsets and back, and at
    # the interface the θ planes stage through the host messages.
    section("positivity limiter: device storage against the host, decomposed")
    cpu = KernelAbstractions.CPU()
    function staged(build)
        CL.FORCE_KA[] = true
        CL.FORCE_DEVICE_EXCHANGE[] = true
        try
            return build()
        finally
            CL.FORCE_KA[] = false
            CL.FORCE_DEVICE_EXCHANGE[] = false
        end
    end
    function blast2(backend)
        s = Solver(n_global=(SPLITN, SPLITN, 1), L_domain=(1.0, 1.0, 1.0),
                   bcs=((SlipWallBC(), SlipWallBC()), per3[2], per3[3]), eos=gas,
                   art=ArtificialProperties(enabled=true), cfl=0.4,
                   control=StepControl(validity=:permissive), positivity_limiter=true,
                   dims=splitdims(1), backend=backend)
        Q = allocate_state(s)
        initialize!(s, Q, ic)
        run!(s, Q; tfinal=1.0, nmax=20)
        return s, Q
    end
    s1, Q1 = blast2(CPUBackend())
    s2, Q2 = staged(() -> blast2(DeviceBackend(cpu)))
    check("device split x: bitwise against the host",
          gmax(maximum(abs.(parent(Q1) .- parent(Q2)))), 1e-300)
    check("device split x: counts and steps agree",
          CL.positivity_counts(s1) == CL.positivity_counts(s2) && s1.step == s2.step ?
          0.0 : 1.0, 0.5)
    # A pressure jump beside the interface, whose shock limits the faces about
    # the shared node from one side first, so that the exchange acts.
    function patched2(backend)
        s = Solver(n_global=(96, 1, 1), L_domain=(1.0, 1.0, 1.0),
                   bcs=((SlipWallBC(), SlipWallBC()), per3[2], per3[3]), eos=gas,
                   art=ArtificialProperties(enabled=true), cfl=0.3, filter_cfl=0.35,
                   control=StepControl(validity=:permissive), positivity_limiter=true,
                   patch_grid=(2, 1, 1), interface_flux=:closure, backend=backend)
        Q = allocate_state(s)
        initialize!(s, Q, (x, y, z) -> Prim(rho=1.0, u=(0.0, 0.0, 0.0),
                                            p=1000 * (1 - tanh_blend(x, 0.45, 2 / 95)) +
                                              0.01))
        run!(s, Q; tfinal=1.0, nmax=30)
        return s, Q
    end
    s1, Q1 = patched2(CPUBackend())
    s2, Q2 = staged(() -> patched2(DeviceBackend(cpu)))
    worst = isempty(Q1) ? 0.0 :
            maximum(maximum(abs.(parent(Q1[i]) .- parent(Q2[i]))) for i in eachindex(Q1))
    check("device patches: bitwise against the host", gmax(worst), 1e-300)
    check("device patches: counts and steps agree",
          CL.positivity_counts(s1) == CL.positivity_counts(s2) && s1.step == s2.step ?
          0.0 : 1.0, 0.5)

    # A refined level of tiles, which the level's rank subset holds, a tile
    # decomposed within its owners where they are several, at the global step
    # and subcycled, against every patch on one rank: the tiles' shared faces
    # take one θ through the level's records and communicator. The subcycled
    # run's root step is three of the global run's, and its limiter acts within
    # the steps taken; at a lower ambient pressure, where it acts sooner, a
    # last-bit difference in the root's decomposed running sums flips a θ in
    # the cold gas and the states differ by 3e-5 after twelve steps.
    for sub in (false, true)
        section("positivity limiter: refined tiles across ranks against one rank" *
                (sub ? ", subcycled" : ""))
        function refined(comm_here)
            s = Solver(n_global=(48, 48, 1), L_domain=(1.0, 1.0, 1.0),
                       bcs=((SlipWallBC(), SlipWallBC()), (SlipWallBC(), SlipWallBC()),
                            per3[3]), eos=gas,
                       art=ArtificialProperties(enabled=true), cfl=0.4,
                       control=StepControl(validity=:permissive), positivity_limiter=true,
                       refine=BlockRegion((12, 12, 0), (24, 24, 1)), tile=8,
                       subcycle=sub, interface_flux=:closure, comm=comm_here)
            Q = allocate_state(s)
            initialize!(s, Q, (x, y, z) -> Prim(rho=1.0, u=(0.0, 0.0, 0.0),
                                                p=1e-3 + exp(-((x - 0.5)^2 +
                                                               (y - 0.45)^2) / 0.004)))
            run!(s, Q; tfinal=1.0, nmax=20)
            return s, Q
        end
        rref, Qrref = refined(MPI.COMM_SELF)
        crref = CL.positivity_counts(rref)
        s, Q = refined(comm)
        c = CL.positivity_counts(s)
        rpatches = getfield(rref, :patches)
        worst = 0.0
        for (ps, q) in CL.eachpatch(s, Q)
            k = findfirst(p -> p.level == ps.patch.level && p.region == ps.patch.region,
                          rpatches)
            pr = CL.PatchSolver(rref, rpatches[k])
            off = ps.decomp.offset
            for j in 1:ps.decomp.n_local[2], i in 1:ps.decomp.n_local[1],
                n in 1:s.equations.n_cons
                a = q[padded_index(ps, i, j, 1), n]
                b = Qrref[k][padded_index(pr, i + off[1], j + off[2], 1), n]
                worst = max(worst, abs(a - b))
            end
        end
        scale = maximum(q -> maximum(abs, parent(q)), Qrref)
        tag = sub ? "subcycled tiles" : "tiles"
        check("$tag: state against one rank (relative)", gmax(worst) / scale, 1e-8)
        check("$tag: clock against one rank", abs(s.t - rref.t) / rref.t, 1e-8)
        check("$tag: stage faces limited (relative difference)",
              abs(c.stage_limited - crref.stage_limited) / max(crref.stage_limited, 1),
              1e-2)
        report = state_report(s, Q)
        check("$tag: inadmissible points at the end",
              report.inadmissible + report.negative_density, 0.5)
        sub && check("$tag, one rank: the limiter acted (expect > 0)",
                     crref.stage_limited > 0 ? 0.0 : 1.0, 0.5)
    end

    # The shell's admissible fallback on a box decomposed over its level's
    # ranks, limiter off, subcycled: each rank gathers the multilinear rings of
    # the others' components beside the chain's, against one rank, which reads
    # the parent's boxes directly. The shell takes the fallback at 346 nodes
    # by the last step of the one-rank run.
    section("shell fallback: a decomposed box against one rank")
    function boxed(comm_here)
        s = Solver(n_global=(48, 48, 1), L_domain=(1.0, 1.0, 1.0),
                   bcs=((SlipWallBC(), SlipWallBC()), (SlipWallBC(), SlipWallBC()),
                        per3[3]), eos=gas, transport=ConstantTransport(mu0=0.0),
                   art=ArtificialProperties(enabled=true), cfl=0.4,
                   control=StepControl(validity=:permissive),
                   refine=BlockRegion((20, 18, 0), (10, 10, 1)), subcycle=true,
                   comm=comm_here)
        Q = allocate_state(s)
        initialize!(s, Q, (x, y, z) -> Prim(rho=1.0, u=(0.0, 0.0, 0.0),
                                            p=1e-3 + exp(-((x - 0.5)^2 +
                                                           (y - 0.45)^2) / 0.004)))
        run!(s, Q; tfinal=1.0, nmax=20)
        return s, Q
    end
    bref, Qbref = boxed(MPI.COMM_SELF)
    s, Q = boxed(comm)
    boxes = [p for p in getfield(s, :patches) if p.level == 1]
    span = isempty(boxes) ? 0 : MPI.Comm_size(boxes[1].decomp.comm)
    check("decomposed box: the box spans several ranks (expect > 1)",
          gmax(Float64(span)) > 1 ? 0.0 : 1.0, 0.5)
    bpatches = getfield(bref, :patches)
    worst = 0.0
    for (ps, q) in CL.eachpatch(s, Q)
        k = findfirst(p -> p.level == ps.patch.level, bpatches)
        pr = CL.PatchSolver(bref, bpatches[k])
        off = ps.decomp.offset
        for j in 1:ps.decomp.n_local[2], i in 1:ps.decomp.n_local[1],
            n in 1:s.equations.n_cons
            a = q[padded_index(ps, i, j, 1), n]
            b = Qbref[k][padded_index(pr, i + off[1], j + off[2], 1), n]
            worst = max(worst, abs(a - b))
        end
    end
    scale = maximum(q -> maximum(abs, parent(q)), Qbref)
    check("decomposed box: state against one rank (relative)", gmax(worst) / scale, 1e-8)
    check("decomposed box: clock against one rank", abs(s.t - bref.t) / bref.t, 1e-8)
end
