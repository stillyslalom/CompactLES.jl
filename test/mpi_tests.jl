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
function ferr(s, f, fn)
    e = 0.0
    for k in 1:s.dec.nloc[3], j in 1:s.dec.nloc[2], i in 1:s.dec.nloc[1]
        e = max(e, abs(f[gidx(s, i, j, k)] -
                       fn(xcoord(s, 1, i), xcoord(s, 2, j), xcoord(s, 3, k))))
    end
    e
end

fillf!(s, f, fn) = (for k in 1:s.dec.nloc[3], j in 1:s.dec.nloc[2], i in 1:s.dec.nloc[1]
    f[gidx(s, i, j, k)] = fn(xcoord(s, 1, i), xcoord(s, 2, j), xcoord(s, 3, k))
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
        s = Solver(nglob=ng, Ldom=(2π, 2π, 2π), bcs=per3,
                   art=ArtParams(enabled=false), dims=splitdims(ax))
        f = CL.field(s.dec); df = CL.field(s.dec)
        fillf!(s, f, fx)
        CL.exchange_halos!(f, s.dec)
        CL.deriv_along!(df, f, s, ax, 1); CL._scale_grad!(df, s, ax)
        ref = ax == 1 ? dfx : ax == 2 ? dfy : ((x, y, z) -> 0.0)
        # x/y derivatives of sin(3x)cos(2y) are nontrivial; the z derivative is 0.
        tol = ax == 3 ? 1e-10 : 1e-4
        check("∂/∂x$(ax) split $(np)-way (dims=$(splitdims(ax)))",
              gmax(ferr(s, df, ref)), tol)
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
        s = Solver(nglob=ng, Ldom=(2π, 2π, 2π), bcs=per3,
                   deriv=lele_d1_10(), art=ArtParams(enabled=false),
                   dims=splitdims(ax))
        f = CL.field(s.dec); df = CL.field(s.dec)
        fillf!(s, f, fx)
        CL.exchange_halos!(f, s.dec)
        CL.deriv_along!(df, f, s, ax, 1); CL._scale_grad!(df, s, ax)
        ref = ax == 1 ? dfx : dfy
        check("C10 ∂/∂x$(ax) split $(np)-way", gmax(ferr(s, df, ref)), 1e-8)
    end
    # Closed domain (SlipWallBC) with the derivative dimension split: closure
    # rows live on the two edge ranks, interior ranks carry V/W spikes. A deg-3
    # polynomial is exact through the C10 closure cascade.
    sc = Solver(nglob=(SPLITN, 12, 12), Ldom=(1.0, 1.0, 1.0),
                bcs=((SlipWallBC(), SlipWallBC()), per3[2], per3[3]),
                deriv=lele_d1_10(), art=ArtParams(enabled=false),
                dims=splitdims(1))
    fc = CL.field(sc.dec); dfc = CL.field(sc.dec)
    fillf!(sc, fc, (x, y, z) -> 1 + 2x + 3x^2 - x^3)
    CL.exchange_halos!(fc, sc.dec)
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
    s = Solver(nglob=(SPLITN, 12, 12), Ldom=(1.0, 1.0, 1.0),
               bcs=((SlipWallBC(), SlipWallBC()), per3[2], per3[3]),
               art=ArtParams(enabled=false), dims=splitdims(1))
    f = CL.field(s.dec); df = CL.field(s.dec)
    fillf!(s, f, (x, y, z) -> 1 + 2x + 3x^2 - x^3)
    CL.exchange_halos!(f, s.dec)
    CL.deriv_along!(df, f, s, 1, 1); CL._scale_grad!(df, s, 1)
    check("C6 closed-domain deg-3 exact, x split", gmax(ferr(s, df,
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
    s = Solver(nglob=(40, SPLITN, 1), Ldom=(1.0, 2π, 1.0),
               metric=CylindricalMetric(),
               bcs=((AxisBC(), SlipWallBC()), per3[2], per3[3]),
               art=ArtParams(enabled=false), dims=splitdims(2))
    f = CL.field(s.dec); df = CL.field(s.dec)
    fillf!(s, f, (r, θ, z) -> r * cos(θ) * exp(-4r^2))
    CL.exchange_halos!(f, s.dec)
    CL.deriv_along!(df, f, s, 1, -1); CL._scale_grad!(df, s, 1)
    check("cyl axis ∂/∂r, θ split (off-rank shift)", gmax(ferr(s, df,
          (r, θ, z) -> cos(θ) * (1 - 8r^2) * exp(-4r^2))), 1e-4)

    # (b) Spherical origin + poles, θ SPLIT. The origin fold reverses θ in its
    # partner map (revdim = 2), so splitting θ triggers the off-rank REVERSED
    # butterfly. f = e^{−4r²} is smooth at the origin and both poles.
    ssθ = Solver(nglob=(40, SPLITN, 12), Ldom=(1.0, π, 2π),
                 metric=SphericalMetric(),
                 bcs=((OriginBC(), SlipWallBC()), (PoleBC(), PoleBC()), per3[3]),
                 art=ArtParams(enabled=false), dims=splitdims(2))
    f = CL.field(ssθ.dec); df = CL.field(ssθ.dec)
    fillf!(ssθ, f, (r, θ, φ) -> exp(-4r^2))
    CL.exchange_halos!(f, ssθ.dec)
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
        s = Solver(nglob=ng, Ldom=(2π, 2π, 2π), bcs=per3,
                   art=ArtParams(enabled=false), dims=splitdims(ax))
        a = CL.field(s.dec); b = CL.field(s.dec); c = CL.field(s.dec)
        fillf!(s, a, fn); fillf!(s, b, fn); fillf!(s, c, fn)
        for d in 1:3
            CL.exchange_dim_batch!([b, c], s.dec, d)   # grows the shared buffer
        end
        CL.exchange_halos!(a, s.dec)                   # must still send one slab
        e = maximum(abs, a .- b)
        check("batched ≡ per-array, dim $ax split $(np)-way", gmax(e), 1e-14)
    end

    # (b) Corner halo: after the three sequential exchanges, the diagonal halo
    # points must hold the analytic value at their own physical coordinates —
    # only true if the y and z exchanges carried the already-filled x halo
    # layers. Uses the DEFAULT process grid (dims=nothing) so the automatic
    # Dims_create path is covered too; at np ≥ 4 that is a genuine 2-D/3-D
    # grid with real diagonal neighbours.
    s = Solver(nglob=(24, 24, 24), Ldom=(2π, 2π, 2π), bcs=per3,
               art=ArtParams(enabled=false))
    f = CL.field(s.dec)
    fillf!(s, f, fn)
    CL.exchange_halos!(f, s.dec)
    e = 0.0
    for kk in (0, s.dec.nloc[3] + 1), jj in (0, s.dec.nloc[2] + 1),
        ii in (0, s.dec.nloc[1] + 1)
        e = max(e, abs(f[gidx(s, ii, jj, kk)] -
                       fn(xcoord(s, 1, ii), xcoord(s, 2, jj), xcoord(s, 3, kk))))
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
           nglob=(SPLITN, 12, 12), Ldom=(1.0, 1.0, 1.0), metric=CartesianMetric(),
           bcs=per3, dims=splitdims(1), kw=(;)),
        (; name="cartesian stretched, x split",
           nglob=(SPLITN, 12, 12), Ldom=(1.0, 1.0, 1.0), metric=CartesianMetric(),
           bcs=((SlipWallBC(), SlipWallBC()), per3[2], per3[3]), dims=splitdims(1),
           kw=(; stretch=(sine_cluster(0.0, 1.0, 0.5, 0.4), nothing, nothing))),
        (; name="cylindrical off-axis, θ split",
           nglob=(12, SPLITN, 12), Ldom=(1.0, 2π, 0.5), metric=CylindricalMetric(),
           bcs=((SlipWallBC(), SlipWallBC()), per3[2], per3[3]), dims=splitdims(2),
           kw=(; origin=(0.2, 0.0, 0.0))),
        (; name="cylindrical resolved-θ axis, θ split",
           nglob=(40, SPLITN, 1), Ldom=(1.0, 2π, 1.0), metric=CylindricalMetric(),
           bcs=((AxisBC(), SlipWallBC()), per3[2], per3[3]), dims=splitdims(2),
           kw=(;)),
        (; name="spherical origin+poles, θ split (revdim off-rank)",
           nglob=(40, SPLITN, 12), Ldom=(1.0, π, 2π), metric=SphericalMetric(),
           bcs=((OriginBC(), SlipWallBC()), (PoleBC(), PoleBC()), per3[3]),
           dims=splitdims(2), kw=(;)),
        (; name="spherical origin+poles, φ split (pdim off-rank)",
           nglob=(40, 16, SPLITN), Ldom=(1.0, π, 2π), metric=SphericalMetric(),
           bcs=((OriginBC(), SlipWallBC()), (PoleBC(), PoleBC()), per3[3]),
           dims=splitdims(3), kw=(;)),
    ]
    for cs in cases
        kw = merge((; nglob=cs.nglob, Ldom=cs.Ldom, metric=cs.metric,
                    bcs=cs.bcs, dims=cs.dims, art=ArtParams(enabled=false)), cs.kw)
        s = Solver(; kw...)
        Q = allocate_state(s)
        initialize!(s, Q, (x, y, z) -> Prim(u=(0, 0, 0), p=1.0, rho=1.0))
        apply_bcs!(s, Q)
        dQ = zero(Q)
        compute_rhs!(s, Q, dQ)
        m = 0.0
        for c in 1:s.ncons, k in 1:s.dec.nloc[3], j in 1:s.dec.nloc[2], i in 1:s.dec.nloc[1]
            m = max(m, abs(dQ[gidx(s, i, j, k), c]))
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
    s = Solver(nglob=(N, 16, 16), Ldom=(2π, 2π, 2π), bcs=per3,
               transport=Transport(mu0=1e-3), art=ArtParams(enabled=false),
               dims=splitdims(1))
    Q = allocate_state(s)
    initialize!(s, Q, (x, y, z) ->
        Prim(u=(0.1sin(x)cos(y), -0.1cos(x)sin(y), 0.05sin(z)),
             p=1 + 0.05cos(x)cos(z), rho=1 + 0.1sin(y)))
    dQ = zero(Q)
    compute_rhs!(s, Q, dQ)
    worst = 0.0
    npts = N * 16 * 16
    for c in 1:s.ncons
        loc = 0.0
        for k in 1:s.dec.nloc[3], j in 1:s.dec.nloc[2], i in 1:s.dec.nloc[1]
            loc += dQ[gidx(s, i, j, k), c]
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
    s = Solver(nglob=(SPLITN, 16, 16), Ldom=(2π, 2π, 2π), bcs=per3,
               art=ArtParams(enabled=false), dims=splitdims(1))
    Q = allocate_state(s)
    initialize!(s, Q, (x, y, z) ->
        Prim(u=(0.2sin(x), 0, 0), p=1 + 0.1cos(y), rho=1 + 0.2sin(z)))
    dt = compute_dt(s, Q)
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
                   Numerics(nglob=(SPLITN, 16, 16), dims=splitdims(1)))
    run!(s2, Q2; tfinal=1e9, nmax=10)
    nbad = 0.0
    for c in 1:s2.ncons, k in 1:s2.dec.nloc[3], j in 1:s2.dec.nloc[2],
        i in 1:s2.dec.nloc[1]
        isfinite(Q2[gidx(s2, i, j, k), c]) || (nbad += 1)
    end
    check("10 RK steps stay finite (count of non-finite)", gsum(nbad), 0.5)
    tspread = MPI.Allreduce(s2.t, max, comm) - MPI.Allreduce(s2.t, min, comm)
    check("simulation time spread after 10 steps", tspread, 1e-14 * max(s2.t, 1e-30))
end

# ---------------------------------------------------------------------------
# 8. Checkpoint round trip with a real decomposition — one file per rank, so
#    the per-rank offsets and extents must line up on write and read back.
# ---------------------------------------------------------------------------
function test_checkpoint()
    section("checkpoint round trip, decomposed")
    s = Solver(nglob=(SPLITN, 16, 16), Ldom=(2π, 2π, 2π), bcs=per3,
               art=ArtParams(enabled=false), dims=splitdims(1))
    Q = allocate_state(s)
    initialize!(s, Q, (x, y, z) -> Prim(u=(sin(x), 0, 0), p=1 + 0.1cos(y), rho=1.0))
    s.t = 1.25; s.step = 17
    save_checkpoint(s, Q, "mpi_ckpt")
    MPI.Barrier(comm)
    Q2 = allocate_state(s); s.t = 0.0; s.step = 0
    load_checkpoint!(s, Q2, "mpi_ckpt")
    check("restored t and step", abs(s.t - 1.25) + abs(s.step - 17), 1e-15)
    d = 0.0
    for c in 1:s.ncons, k in 1:s.dec.nloc[3], j in 1:s.dec.nloc[2], i in 1:s.dec.nloc[1]
        d = max(d, abs(Q2[gidx(s, i, j, k), c] - Q[gidx(s, i, j, k), c]))
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
