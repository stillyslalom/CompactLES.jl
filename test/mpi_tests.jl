# Multi-rank test suite — the companion to the serial runtests.jl.
#
# Run with (bundled MS-MPI mpiexec from MPI.jl):
#   mpiexec -n 2 julia --project=. -t 1 test/mpi_tests.jl
#   mpiexec -n 4 julia --project=. -t 1 test/mpi_tests.jl
#   mpiexec -n 8 julia --project=. -t 1 test/mpi_tests.jl
#
# This suite targets the code paths that ONLY execute when a dimension is
# split across more than one rank and are therefore UNREACHABLE from the serial
# suite (which runs at np == 1):
#
#   1. The cross-rank reduced-interface (spike) line solve — for both the
#      tridiagonal C6 solver (LineSolver) and the pentadiagonal C10 solver
#      (BandLineSolver): Allgather of the interface blocks, the dense reduced
#      LU, and the threaded spike correction. Serial runs skip this entirely
#      for non-periodic lines and only self-couple for periodic ones.
#   2. Halo exchange across real rank boundaries (exchange_halos! /
#      exchange_dim_batch!), which the np == 1 case exercises only as a
#      periodic self-wrap.
#   3. Off-rank coordinate-singularity folds — the pair_forward!/pair_backward!
#      butterfly that does MPI.Sendrecv! of whole blocks (the e-keeper/o-keeper
#      split), reached only when the pairing/reversed dimension is split.
#   4. The discrete-GCL freestream identity with the θ-derivative of the area
#      factor crossing a rank boundary.
#   5. Telescoping flux conservation across rank boundaries.
#   6. Global synchronization — that every rank agrees on dt and on the
#      simulation time after a real transient. A missing Allreduce leaves the
#      ranks integrating different equations and cannot be seen serially.
#   7. Per-rank checkpoint round trips under a real decomposition.
#
# Correctness strategy (see the module note in banded.jl / tridiag.jl): the
# reduced-interface method is designed to reproduce the *exact* single-domain
# solution regardless of the rank count, so any interface bug shows up as an
# O(1) error, not a small one. Every test therefore reduces a max-abs error
# (or a global sum) to a single scalar with MPI.Allreduce — so ALL ranks assert
# the SAME value and fail together — and compares it to the analytic result at
# the tolerance the serial suite achieves on the same operator.

using MPI
MPI.Init(threadlevel=:funneled)
using CompactLES
using Printf, LinearAlgebra

const CL = CompactLES
const comm = MPI.COMM_WORLD
const np = MPI.Comm_size(comm)
const rank = MPI.Comm_rank(comm)

if np == 1
    rank == 0 && println("mpi_tests.jl is the MULTI-RANK suite; run under " *
                         "mpiexec -n ≥2 (serial checks live in runtests.jl). " *
                         "Nothing to do at np=1.")
    MPI.Finalize()
    exit(0)
end

# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------

const per3 = ntuple(_ -> (PeriodicBC(), PeriodicBC()), 3)

"Max interior error of a scalar field against an analytic function (this rank)."
function ferr(solver, f, fn)
    e = 0.0
    for k in 1:solver.decomp.n_local[3], j in 1:solver.decomp.n_local[2], i in 1:solver.decomp.n_local[1]
        e = max(e, abs(f[gidx(solver, i, j, k)] -
                       fn(xcoord(solver, 1, i), xcoord(solver, 2, j), xcoord(solver, 3, k))))
    end
    e
end

fillf!(solver, f, fn) = (for k in 1:solver.decomp.n_local[3], j in 1:solver.decomp.n_local[2],
                        i in 1:solver.decomp.n_local[1]
    f[gidx(solver, i, j, k)] = fn(xcoord(solver, 1, i), xcoord(solver, 2, j), xcoord(solver, 3, k))
end; f)

gmax(x) = MPI.Allreduce(Float64(x), MPI.MAX, comm)
gsum(x) = MPI.Allreduce(Float64(x), MPI.SUM, comm)

const _npass = Ref(0)
const _nfail = Ref(0)

"Assert a globally-reduced scalar `val` is below `tol`. All ranks see the same
`val`, so all ranks agree on pass/fail. Rank 0 prints one line."
function check(name, val, tol)
    ok = isfinite(val) && val < tol
    ok ? (_npass[] += 1) : (_nfail[] += 1)
    rank == 0 && @printf("    %-52s  %10.3e < %-9.1e  %s\n",
                         name, val, tol, ok ? "ok" : "FAIL")
    return ok
end

section(name) = rank == 0 && println("\n$name")

"Process grid that puts all np ranks on dimension `ax`, one elsewhere."
splitdims(ax) = ntuple(d -> d == ax ? np : 1, 3)

# Global extent for any decomposed dimension. Every active dimension carries a
# C8 filter plan whose closure needs ≥ 9 points per rank, so the split dim must
# give ≥ 9 locally at the largest np we run (8): 72 / 8 = 9. 72 is even and
# divisible by 2, 4, 8, satisfying the folds' uniform-even-block requirement.
const SPLITN = 72

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
        solver = Solver(n_global=ng, L_domain=(2π, 2π, 2π), bcs=per3,
                   art=ArtParams(enabled=false), dims=splitdims(ax))
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
        solver = Solver(n_global=ng, L_domain=(2π, 2π, 2π), bcs=per3,
                   deriv=lele_d1_10(), art=ArtParams(enabled=false),
                   dims=splitdims(ax))
        f = CL.field(solver.decomp); df = CL.field(solver.decomp)
        fillf!(solver, f, fx)
        CL.exchange_halos!(f, solver.decomp)
        CL.deriv_along!(df, f, solver, ax, 1); CL._scale_grad!(df, solver, ax)
        ref = ax == 1 ? dfx : dfy
        check("C10 ∂/∂x$(ax) split $(np)-way", gmax(ferr(solver, df, ref)), 1e-8)
    end
    # Closed domain (SlipWallBC) with the derivative dimension split: closure
    # rows live on the two edge ranks, interior ranks carry V/W spikes. A deg-3
    # polynomial is exact through the C10 closure cascade.
    sc = Solver(n_global=(SPLITN, 12, 12), L_domain=(1.0, 1.0, 1.0),
                bcs=((SlipWallBC(), SlipWallBC()), per3[2], per3[3]),
                deriv=lele_d1_10(), art=ArtParams(enabled=false),
                dims=splitdims(1))
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
    solver = Solver(n_global=(SPLITN, 12, 12), L_domain=(1.0, 1.0, 1.0),
               bcs=((SlipWallBC(), SlipWallBC()), per3[2], per3[3]),
               art=ArtParams(enabled=false), dims=splitdims(1))
    f = CL.field(solver.decomp); df = CL.field(solver.decomp)
    fillf!(solver, f, (x, y, z) -> 1 + 2x + 3x^2 - x^3)
    CL.exchange_halos!(f, solver.decomp)
    CL.deriv_along!(df, f, solver, 1, 1); CL._scale_grad!(df, solver, 1)
    check("C6 closed-domain deg-3 exact, x split", gmax(ferr(solver, df,
          (x, y, z) -> 2 + 6x - 3x^2)), 1e-9)
end

# ---------------------------------------------------------------------------
# 4. Off-rank folds. Splitting the pairing (or reversed) dimension across ranks
#    forces the pair_forward!/pair_backward! block-Sendrecv butterfly that the
#    serial suite (always local_pair) never reaches.
# ---------------------------------------------------------------------------
function test_offrank_folds()
    section("off-rank folds: antipodal pairing butterfly across ranks")

    # (a) Resolved-θ cylindrical axis, θ SPLIT. The fold is on r (dim 1, one
    # rank); the antipodal partner is the θ+π slot, so splitting θ makes the
    # partner live on rank +np/2 → off-rank shift butterfly. f = r cosθ e^{−4r²}
    # is smooth through the axis and ODD under (r,θ)→(−r,θ) with the pairing.
    # θ count 48 is even and divisible by np∈{2,4,8}, and keeps ≥ 5 θ-points
    # per rank for the θ filter/derivative plans (≥ 9 at np = 8).
    solver = Solver(n_global=(40, SPLITN, 1), L_domain=(1.0, 2π, 1.0),
               metric=CylindricalMetric(),
               bcs=((AxisBC(), SlipWallBC()), per3[2], per3[3]),
               art=ArtParams(enabled=false), dims=splitdims(2))
    f = CL.field(solver.decomp); df = CL.field(solver.decomp)
    fillf!(solver, f, (r, θ, z) -> r * cos(θ) * exp(-4r^2))
    CL.exchange_halos!(f, solver.decomp)
    CL.deriv_along!(df, f, solver, 1, -1); CL._scale_grad!(df, solver, 1)
    check("cyl axis ∂/∂r, θ split (off-rank shift)", gmax(ferr(solver, df,
          (r, θ, z) -> cos(θ) * (1 - 8r^2) * exp(-4r^2))), 1e-4)

    # (b) Spherical origin + poles, θ SPLIT. The origin fold reverses θ in its
    # partner map (revdim = 2), so splitting θ triggers the off-rank REVERSED
    # butterfly. f = e^{−4r²} is smooth at the origin and both poles.
    ssθ = Solver(n_global=(40, SPLITN, 12), L_domain=(1.0, π, 2π),
                 metric=SphericalMetric(),
                 bcs=((OriginBC(), SlipWallBC()), (PoleBC(), PoleBC()), per3[3]),
                 art=ArtParams(enabled=false), dims=splitdims(2))
    f = CL.field(ssθ.decomp); df = CL.field(ssθ.decomp)
    fillf!(ssθ, f, (r, θ, φ) -> exp(-4r^2))
    CL.exchange_halos!(f, ssθ.decomp)
    CL.deriv_along!(df, f, ssθ, 1, 1); CL._scale_grad!(df, ssθ, 1)
    check("sph origin ∂/∂r, θ split (off-rank reverse)", gmax(ferr(ssθ, df,
          (r, θ, φ) -> -8r * exp(-4r^2))), 1e-4)
    CL.deriv_along!(df, f, ssθ, 2, 1); CL._scale_grad!(df, ssθ, 2)
    check("sph poles (1/r)∂/∂θ = 0, θ split", gmax(ferr(ssθ, df,
          (r, θ, φ) -> 0.0)), 1e-8)
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
        solver = Solver(n_global=ng, L_domain=(2π, 2π, 2π), bcs=per3,
                   art=ArtParams(enabled=false), dims=splitdims(ax))
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
    solver = Solver(n_global=(24, 24, 24), L_domain=(2π, 2π, 2π), bcs=per3,
               art=ArtParams(enabled=false))
    f = CL.field(solver.decomp)
    fillf!(solver, f, fn)
    CL.exchange_halos!(f, solver.decomp)
    e = 0.0
    for kk in (0, solver.decomp.n_local[3] + 1), jj in (0, solver.decomp.n_local[2] + 1),
        ii in (0, solver.decomp.n_local[1] + 1)
        e = max(e, abs(f[gidx(solver, ii, jj, kk)] -
                       fn(xcoord(solver, 1, ii), xcoord(solver, 2, jj), xcoord(solver, 3, kk))))
    end
    check("corner halos filled (all 8 corners)", gmax(e), 1e-12)
end

# ---------------------------------------------------------------------------
# 5. Freestream preservation (uniform state ⇒ dQ ≈ 0), decomposed. Divergence,
#    metric sources, and the (fold-aware, discrete-GCL) geometry must stay
#    consistent to machine zero even when the singular-axis θ-derivative of the
#    area factor crosses a rank boundary.
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
    ]
    for cs in cases
        kw = merge((; n_global=cs.n_global, L_domain=cs.L_domain, metric=cs.metric,
                    bcs=cs.bcs, dims=cs.dims, art=ArtParams(enabled=false)), cs.kw)
        solver = Solver(; kw...)
        Q = allocate_state(solver)
        initialize!(solver, Q, (x, y, z) -> Prim(u=(0, 0, 0), p=1.0, rho=1.0))
        apply_bcs!(solver, Q)
        dQ = zero(Q)
        compute_rhs!(solver, Q, dQ)
        m = 0.0
        for c in 1:solver.equations.n_cons, k in 1:solver.decomp.n_local[3],
            j in 1:solver.decomp.n_local[2], i in 1:solver.decomp.n_local[1]
            m = max(m, abs(dQ[gidx(solver, i, j, k), c]))
        end
        check(cs.name, gmax(m), 1e-8)
    end
end

# ---------------------------------------------------------------------------
# 6. Conservation: a periodic RHS must integrate to zero globally — a genuine
#    telescoping-flux check across rank boundaries (global MPI.Reduce sum).
# ---------------------------------------------------------------------------
function test_conservation()
    section("conservation: periodic RHS integrates to zero (global sum)")
    N = SPLITN                   # ≥ 9 per rank up to np = 8 (filter closure)
    solver = Solver(n_global=(N, 16, 16), L_domain=(2π, 2π, 2π), bcs=per3,
               transport=Transport(mu0=1e-3), art=ArtParams(enabled=false),
               dims=splitdims(1))
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
            loc += dQ[gidx(solver, i, j, k), c]
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
    solver = Solver(n_global=(SPLITN, 16, 16), L_domain=(2π, 2π, 2π), bcs=per3,
               art=ArtParams(enabled=false), dims=splitdims(1))
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
                   Numerics(n_global=(SPLITN, 16, 16), dims=splitdims(1)))
    run!(s2, Q2; tfinal=1e9, nmax=10)
    nbad = 0.0
    for c in 1:s2.equations.n_cons, k in 1:s2.decomp.n_local[3], j in 1:s2.decomp.n_local[2],
        i in 1:s2.decomp.n_local[1]
        isfinite(Q2[gidx(s2, i, j, k), c]) || (nbad += 1)
    end
    check("10 RK steps stay finite (count of non-finite)", gsum(nbad), 0.5)
    tspread = MPI.Allreduce(s2.t, max, comm) - MPI.Allreduce(s2.t, min, comm)
    check("simulation time spread after 10 steps", tspread, 1e-14 * max(s2.t, 1e-30))
end

# ---------------------------------------------------------------------------
# 7b. Callback triggers must be globally consistent, and a SwitchableBC must
#     switch on the same step everywhere. This is the deadlock case, not an
#     accuracy case: NSCBCOutflowBC runs collectives that SlipWallBC does not,
#     so if the ranks disagree about whether the switch has happened, some enter
#     a collective the others never reach and the run hangs at zero CPU rather
#     than failing. WhenState reduces its condition for exactly this reason.
#
#     The condition below is deliberately the worst case: true on ONE rank only
#     — the one owning the high-x boundary plane, which is where NSCBC would
#     run. Without the reduction inside WhenState this test hangs. With it, the
#     rank-local condition must produce bit-identical state to a condition every
#     rank can evaluate for itself, since both fire on the same step.
# ---------------------------------------------------------------------------
function test_callback_consistency()
    section("callback triggers and SwitchableBC agree across ranks")
    build() = begin
        xbc = (SlipWallBC(), SwitchableBC(SlipWallBC(), NSCBCOutflowBC(pinf=1.0)))
        s, Q = setup(Problem(domain=((0.0, 1.0), (0.0, 0.25), (0.0, 0.25)),
                             bcs=(xbc, per3[2], per3[3]),
                             ic=(x, y, z) -> Prim(u=(0, 0, 0),
                                                  p=1 + 4exp(-200(x - 0.3)^2),
                                                  rho=1 + exp(-200(x - 0.3)^2))),
                     Numerics(n_global=(SPLITN, 16, 16), dims=splitdims(1)))
        s, Q, xbc[2]
    end

    # (a) rank-local condition: only the last rank along x can see it.
    s_local, Q_local, bc_local = build()
    run!(s_local, Q_local; tfinal=1e9, nmax=12,
         callback=Callback(WhenState((s, _) -> rank == np - 1 && s.step >= 5),
                           (_, _) -> (switch!(bc_local); nothing)))

    # (b) the same switch from a condition every rank evaluates identically.
    s_glob, Q_glob, bc_glob = build()
    run!(s_glob, Q_glob; tfinal=1e9, nmax=12,
         callback=Callback(WhenState((s, _) -> s.step >= 5),
                           (_, _) -> (switch!(bc_glob); nothing)))

    check("switched flag agrees on every rank",
          MPI.Allreduce(Int(switched(bc_local)), max, comm) -
          MPI.Allreduce(Int(switched(bc_local)), min, comm), 0.5)
    err = 0.0
    for c in 1:s_local.equations.n_cons, k in 1:s_local.decomp.n_local[3],
        j in 1:s_local.decomp.n_local[2], i in 1:s_local.decomp.n_local[1]
        I = gidx(s_local, i, j, k)
        err = max(err, abs(Q_local[I, c] - Q_glob[I, c]))
    end
    check("rank-local vs global trigger: state difference",
          MPI.Allreduce(err, max, comm), 1e-15)
    tspread = MPI.Allreduce(s_local.t, max, comm) -
              MPI.Allreduce(s_local.t, min, comm)
    check("time spread after a switched run", tspread,
          1e-14 * max(s_local.t, 1e-30))

    # An AtTime trigger clips dt, which must stay a global decision: every rank
    # has to land on the same instant or they integrate different equations.
    s_t, Q_t, _ = build()
    landed = Float64[]
    run!(s_t, Q_t; tfinal=0.05,
         callback=Callback(AtTime([0.01, 0.02]),
                           (s, _) -> (push!(landed, s.t); nothing)))
    hit_spread = 0.0
    for (m, want) in enumerate((0.01, 0.02))
        m <= length(landed) || (hit_spread = Inf; break)
        hit_spread = max(hit_spread, abs(landed[m] - want),
                         MPI.Allreduce(landed[m], max, comm) -
                         MPI.Allreduce(landed[m], min, comm))
    end
    check("AtTime landed exactly, identically on all ranks", hit_spread, 1e-14)

    # EveryTime shortens dt in the same way and additionally re-anchors from
    # solver.t. That anchoring must also be a global decision: a rank computing
    # a different next instant would shorten to a different dt, and the ranks
    # would diverge on the following step.
    s_e, Q_e, _ = build()
    ticks = Float64[]
    run!(s_e, Q_e; tfinal=0.045,
         callback=Callback(EveryTime(0.015), (s, _) -> (push!(ticks, s.t); nothing)))
    tick_spread = length(ticks) == 3 ? 0.0 : Inf
    for (m, want) in enumerate((0.015, 0.03, 0.045))
        m <= length(ticks) || break
        tick_spread = max(tick_spread, abs(ticks[m] - want),
                          MPI.Allreduce(ticks[m], max, comm) -
                          MPI.Allreduce(ticks[m], min, comm))
    end
    check("EveryTime landed exactly, identically on all ranks", tick_spread, 1e-14)
end

# ---------------------------------------------------------------------------
# 8. State queries under decomposition. `boundary_plane` is rank-local by
#    design, returning `nothing` on every rank that does not hold the face, and
#    that branch does not arise in serial where one rank holds every edge. A
#    callback condition reading a boundary is written in this form, so the
#    branch is more significant than its size suggests.
# ---------------------------------------------------------------------------
function test_state_queries()
    section("state queries across ranks")
    s, Q = setup(Problem(domain=((0.0, 1.0), (0.0, 0.25), (0.0, 0.25)),
                         bcs=((SlipWallBC(), SlipWallBC()), per3[2], per3[3]),
                         ic=(x, y, z) -> Prim(u=(0.2, 0, 0),
                                              p=1 + 0.5exp(-200(x - 0.3)^2),
                                              rho=1 + exp(-200(x - 0.3)^2))),
                 Numerics(n_global=(SPLITN, 16, 16), dims=splitdims(1)))

    # Split along x, so exactly one rank holds the low-x face and exactly one
    # holds the high-x face, for any number of ranks.
    owners(side) = gsum(boundary_plane(s, 1, side) === nothing ? 0 : 1)
    check("exactly one rank owns the low-x face",  abs(owners(1) - 1), 0.5)
    check("exactly one rank owns the high-x face", abs(owners(2) - 1), 0.5)
    # Transverse dimensions are undivided here, so every rank owns those faces.
    check("every rank owns an undivided face",
          abs(gsum(boundary_plane(s, 2, 1) === nothing ? 0 : 1) - np), 0.5)

    # The `upstream_density` pattern from the tutorial: a rank-local maximum
    # over the face held by this rank, reduced by the caller. It must agree with
    # the same quantity taken from the refreshed primitives.
    refresh_primitives!(s, Q)
    face_max = let plane = boundary_plane(s, 1, 1)
        plane === nothing ? -Inf : maximum(I -> mixture_density(s, Q, I), plane)
    end
    face_rho = let plane = boundary_plane(s, 1, 1)
        plane === nothing ? -Inf : maximum(I -> s.rho[I], plane)
    end
    check("face maximum from Q matches the refreshed primitives",
          abs(gmax(face_max) - gmax(face_rho)), 1e-14)

    # refresh_primitives! is collective, so every rank must reach it, and
    # afterwards each rank's primitives agree with its own Q.
    err = 0.0
    for k in 1:s.decomp.n_local[3], j in 1:s.decomp.n_local[2], i in 1:s.decomp.n_local[1]
        I = gidx(s, i, j, k)
        err = max(err, abs(mixture_density(s, Q, I) - s.rho[I]))
        u, v, w = velocity(s, Q, I)
        err = max(err, abs(u - s.u[I]), abs(v - s.v[I]), abs(w - s.w[I]))
    end
    check("primitives agree with Q on every rank after a refresh", gmax(err), 1e-14)
end

# ---------------------------------------------------------------------------
# 9. FieldWriter under decomposition. save_vtk is collective, since it
#    Allgathers the piece extents so that rank 0 can write the container, and
#    the serial suite does not exercise this path: one rank writes one piece
#    there, leaving the per-rank naming and the container's piece list untested.
#    The writer adds directory creation and a rank-0 collection file, both of
#    which can succeed in serial and deadlock at high rank counts.
# ---------------------------------------------------------------------------
function test_field_writer()
    section("FieldWriter under decomposition")
    solver, Q = setup(Problem(domain=((0.0, 1.0), (0.0, 0.25), (0.0, 0.25)),
                              bcs=((SlipWallBC(), SlipWallBC()), per3[2], per3[3]),
                              ic=(x, y, z) -> Prim(u=(0.2, 0, 0),
                                                   p=1 + 0.1exp(-40(x - 0.5)^2),
                                                   rho=1.0)),
                      Numerics(n_global=(SPLITN, 16, 16), dims=splitdims(1)))
    writer = FieldWriter(joinpath("mpi_frames", "field"))
    run!(solver, Q; tfinal=0.03, callback=Callback(EveryTime(0.01), writer))

    # Every rank must have written the same number of frames at the same times.
    # A rank that skipped one would have desynchronized the internal Allgather.
    check("frame count agrees on every rank",
          MPI.Allreduce(writer.index, max, comm) -
          MPI.Allreduce(writer.index, min, comm), 0.5)
    tspread = 0.0
    for m in 1:min(writer.index, 3)
        tspread = max(tspread, abs(writer.times[m] - 0.01m),
                      MPI.Allreduce(writer.times[m], max, comm) -
                      MPI.Allreduce(writer.times[m], min, comm))
    end
    check("frame times exact and identical on every rank", tspread, 1e-14)

    # One piece per rank per frame, one container per frame, one collection.
    MPI.Barrier(comm)
    counted = rank == 0 ? length(filter(endswith(".vtr"), readdir("mpi_frames"))) : 0
    check("one .vtr piece per rank per frame", abs(gsum(counted) - 3 * np), 0.5)
    containers = rank == 0 ? length(filter(endswith(".pvtr"), readdir("mpi_frames"))) : 0
    check("one .pvtr container per frame", abs(gsum(containers) - 3), 0.5)
    pieces = rank == 0 ?
        count("<Piece", read(joinpath("mpi_frames", "field_0000.pvtr"), String)) : 0
    check("container lists every rank's piece", abs(gsum(pieces) - np), 0.5)
    sets = rank == 0 ?
        count("<DataSet", read(joinpath("mpi_frames", "field.pvd"), String)) : 0
    check("collection written once, by rank 0", abs(gsum(sets) - 3), 0.5)

    MPI.Barrier(comm)
    rank == 0 && rm("mpi_frames"; recursive=true)
    MPI.Barrier(comm)

    # Strided output decomposed. The retained points are defined on the GLOBAL
    # index, so the pieces sample one common lattice and tile the coarse grid
    # exactly. Striding each rank's block from its own first point would produce
    # files that appear correct but tile incorrectly, with a doubled or missing
    # plane at a rank boundary. Serial runs cannot detect this, having only one
    # piece.
    strided, Qs = setup(Problem(domain=((0.0, 1.0), (0.0, 0.25), (0.0, 0.25)),
                                bcs=(per3[1], per3[2], per3[3]),
                                ic=(x, y, z) -> Prim(u=(0.1, 0, 0), p=1.0,
                                                     rho=1 + x)),
                        Numerics(n_global=(SPLITN, 16, 16), dims=splitdims(1)))
    for sd in (2, 3)
        save_vtk(strided, Qs, "mpi_stride"; fields=(:rho,), stride=sd)
        MPI.Barrier(comm)
        nkeep = (length(1:sd:SPLITN), length(1:sd:16), length(1:sd:16))
        total, inside, whole_ok = 0, 0, 0
        if rank == 0
            txt = read("mpi_stride.pvtr", String)
            whole_ok = occursin("WholeExtent=\"0 $(nkeep[1]-1) 0 $(nkeep[2]-1) " *
                                "0 $(nkeep[3]-1)\"", txt) ? 1 : 0
            inside = 1
            for m in eachmatch(r"<Piece Extent=\"([^\"]+)\"", txt)
                e = parse.(Int, split(m.captures[1]))
                total += prod(ntuple(d -> e[2d] - e[2d-1] + 1, 3))
                for d in 1:3
                    (0 <= e[2d-1] <= e[2d] <= nkeep[d] - 1) || (inside = 0)
                end
            end
        end
        # Piece point counts summing to the coarse grid, with every piece
        # inside it, admits neither an overlap nor a gap.
        check("stride $sd: pieces tile the coarse grid",
              abs(gsum(total) - prod(nkeep)), 0.5)
        check("stride $sd: every piece within the whole extent",
              abs(gsum(inside) - 1), 0.5)
        check("stride $sd: whole extent is the coarse grid",
              abs(gsum(whole_ok) - 1), 0.5)
        MPI.Barrier(comm)
        rank == 0 && foreach(rm, filter(startswith("mpi_stride"), readdir()))
        MPI.Barrier(comm)
    end

    # A stride coarse enough to leave a rank with no points gives that rank no
    # piece to write, and must fail on EVERY rank. One rank raising while the
    # others proceed into the Allgather produces a hang rather than an error.
    too_coarse = 2 * (SPLITN ÷ np) + 1
    threw = try
        save_vtk(strided, Qs, "mpi_toocoarse"; fields=(:rho,), stride=too_coarse)
        0
    catch e
        e isa ArgumentError ? 1 : 0
    end
    check("an over-coarse stride throws on every rank", abs(gsum(threw) - np), 0.5)
    MPI.Barrier(comm)
    rank == 0 && foreach(rm, filter(startswith("mpi_toocoarse"), readdir()))
    MPI.Barrier(comm)

    # The curvilinear path decomposed, split along the resolved angle. It writes
    # a different piece type and container, so no check above covers it, and
    # splitting the angle is the case in which the piece extents must agree
    # around a periodic wrap.
    cyl, Qc = setup(Problem(domain=((0.5, 1.5), (0.0, 2π), (0.0, 1.0)),
                            metric=CylindricalMetric(),
                            bcs=((SlipWallBC(), SlipWallBC()), per3[2], per3[3]),
                            ic=(r, θ, z) -> Prim(u=(0.0, 0.5, 0.0), p=1.0, rho=1.0)),
                    Numerics(n_global=(12, SPLITN, 12), dims=splitdims(2)))
    check("resolved angle selects the structured container",
          container_extension(cyl) == ".pvts" ? 0.0 : 1.0, 0.5)
    save_vtk(cyl, Qc, "mpi_annulus")
    MPI.Barrier(comm)
    mine = isfile("mpi_annulus.r" * lpad(rank, 4, '0') * ".vts") ? 1 : 0
    check("every rank wrote its .vts piece", abs(gsum(mine) - np), 0.5)
    vts = rank == 0 ? read("mpi_annulus.pvts", String) : ""
    kind = rank == 0 ? (occursin("PStructuredGrid", vts) ? 1 : 0) : 0
    check("container is a PStructuredGrid", abs(gsum(kind) - 1), 0.5)
    check("container lists every angular piece",
          abs(gsum(rank == 0 ? count("<Piece", vts) : 0) - np), 0.5)
    MPI.Barrier(comm)
    rank == 0 && foreach(rm, filter(startswith("mpi_annulus"), readdir()))
    MPI.Barrier(comm)
end

# ---------------------------------------------------------------------------
# 10. Checkpoint round trip with a real decomposition — one file per rank, so
#    the per-rank offsets and extents must line up on write and read back.
# ---------------------------------------------------------------------------
function test_checkpoint()
    section("checkpoint round trip, decomposed")
    solver = Solver(n_global=(SPLITN, 16, 16), L_domain=(2π, 2π, 2π), bcs=per3,
               art=ArtParams(enabled=false), dims=splitdims(1))
    Q = allocate_state(solver)
    initialize!(solver, Q, (x, y, z) -> Prim(u=(sin(x), 0, 0), p=1 + 0.1cos(y), rho=1.0))
    solver.t = 1.25; solver.step = 17
    save_checkpoint(solver, Q, "mpi_ckpt")
    MPI.Barrier(comm)
    Q2 = allocate_state(solver); solver.t = 0.0; solver.step = 0
    load_checkpoint!(solver, Q2, "mpi_ckpt")
    check("restored t and step", abs(solver.t - 1.25) + abs(solver.step - 17), 1e-15)
    d = 0.0
    for c in 1:solver.equations.n_cons, k in 1:solver.decomp.n_local[3],
        j in 1:solver.decomp.n_local[2], i in 1:solver.decomp.n_local[1]
        d = max(d, abs(Q2[gidx(solver, i, j, k), c] - Q[gidx(solver, i, j, k), c]))
    end
    check("state bit-identical after round trip", gmax(d), 1e-300)
    # One rank does the cleanup: every rank wrote its own file, but all of them
    # racing to rm the whole glob just makes them delete each other's entries
    # and throw ENOENT.
    MPI.Barrier(comm)
    rank == 0 && foreach(rm, filter(startswith("mpi_ckpt"), readdir()))
    MPI.Barrier(comm)
end

# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
rank == 0 && println("=== CompactLES multi-rank test suite (np = $np) ===")

try
    test_periodic_c6()
    test_pentadiagonal_c10()
    test_closed_c6()
    test_halo_consistency()
    test_offrank_folds()
    test_freestream()
    test_conservation()
    test_sync()
    test_callback_consistency()
    test_state_queries()
    test_field_writer()
    test_checkpoint()
catch e
    # A throw on any rank (e.g. a setup error) must not deadlock the others.
    println("rank $rank: uncaught exception: ", e)
    flush(stdout)
    MPI.Abort(comm, 1)
end

# Every check reduced its metric across all ranks, so _nfail is identical
# everywhere: all ranks exit together with the same status.
MPI.Barrier(comm)
if rank == 0
    total = _npass[] + _nfail[]
    println("\n=== $(_npass[])/$total checks passed" *
            (_nfail[] == 0 ? "" : "  ($(_nfail[]) FAILED)") * " ===")
end
MPI.Barrier(comm)
if _nfail[] > 0
    exit(1)
end
rank == 0 && println("multi-rank tests complete")
MPI.Finalize()
