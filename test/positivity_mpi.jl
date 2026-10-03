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
end
