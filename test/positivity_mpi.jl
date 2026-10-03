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
        check("split axis $ax: state against one rank (relative)", gmax(worst) / scale,
              1e-12)
        check("split axis $ax: clock against one rank", abs(s.t - ref.t) / ref.t, 1e-12)
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
end
