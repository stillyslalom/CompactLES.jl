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

# Timing first, so that package load is measured rather than assumed. Every
# rank compiles this suite independently, so the compile column here is
# per-rank work that the rank count multiplies. See test/timing.jl.
include("timing.jl")

@phase "package load" begin
    using MPI
    MPI.Init(threadlevel=:funneled)
    using CompactLES
    using LinearAlgebra
    import KernelAbstractions
end

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
# 3b. AMR transfer pair (reference/AMR_GPU.md, Stage 1). The invertibility
#     identity prolong(restrict(f)) == f holds at any rank count, so applying
#     the pair along a split dimension pins the distributed tridiagonal solve
#     of the restriction filter and the pentadiagonal spike solve of the
#     deconvolution, edge-rank closure rows included, at machine precision —
#     any interface bug is an O(1) error. The TransferPlan check decomposes a
#     transverse dimension instead (the transfer dimension itself must not be
#     split in Stage 1) and asserts the coarse → fine → coarse left inverse.
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
    #
    # The two instants sit a little over two CFL steps apart (dt ≈ 1.3e-3 here),
    # which is inside `landing_steps`, so the soft landing subdivides the gap on
    # the approach rather than clipping straight to it. Both branches of the clip
    # are therefore exercised in the eight steps this takes. See the note on run
    # length below.
    s_t, Q_t, _ = build()
    landed = Float64[]
    run!(s_t, Q_t; tfinal=0.008,
         callback=Callback(AtTime([0.003, 0.006]),
                           (s, _) -> (push!(landed, s.t); nothing)))
    hit_spread = 0.0
    for (m, want) in enumerate((0.003, 0.006))
        m <= length(landed) || (hit_spread = Inf; break)
        hit_spread = max(hit_spread, abs(landed[m] - want),
                         MPI.Allreduce(landed[m], max, comm) -
                         MPI.Allreduce(landed[m], min, comm))
    end
    check("AtTime landed exactly, identically on all ranks", hit_spread, 1e-14)

    # EveryTime shortens dt in the same way and additionally re-anchors from
    # solver.t. That anchoring must also be a global decision: a rank computing
    # a different next instant would shorten to a different dt, and the ranks
    # would diverge on the following step. Three ticks, so the re-anchoring is
    # exercised twice rather than once.
    #
    # Run length. Everything this function checks is agreement between ranks on a
    # per-step decision, so it is the *step count* that has to be enough to catch
    # a divergence, and every step costs the same. The four runs here take 12, 12,
    # 8 and 9 steps, against 12, 12, 44 and 40 before the lengths were cut. That
    # matters at high rank counts on a runner with fewer cores than ranks, where
    # each step is a fixed toll of small collectives and the phase cost is linear
    # in steps: at np = 8 on a four-core CI runner these runs cost ~7 s per step
    # against ~0.04 s at np = 2, so the phase was 13 minutes of a 20-minute job.
    # Lengthen a run here only for a check that needs the extra steps.
    #
    # The four runs also take the same number of steps at every rank count, which
    # is what makes the step count the thing to watch: measured 12, 12, 44 and 40
    # at both np = 2 and np = 8 before the cut.
    #
    # tfinal here is one ULP BELOW the third instant, since `0.006 + 0.003` is
    # not the 0.009 literal. That combination used to cost 48 steps rather than 9:
    # the soft landing in `run!` aimed at an instant past the endpoint and halved
    # dt against it forever. Nothing in the checks below moved, so the step count
    # is the only evidence — leave the two literals as they are, and see the note
    # on the landing in `run!`.
    s_e, Q_e, _ = build()
    ticks = Float64[]
    run!(s_e, Q_e; tfinal=0.009,
         callback=Callback(EveryTime(0.003), (s, _) -> (push!(ticks, s.t); nothing)))
    tick_spread = length(ticks) == 3 ? 0.0 : Inf
    for (m, want) in enumerate((0.003, 0.006, 0.009))
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
    # Three frames is what the piece and container counts below are written
    # against; the run is then only as long as it takes to produce them, since
    # every check here is on the files rather than on the state in them. The
    # interval is a little over one CFL step, so the frames still land on
    # separate steps. See the run-length note in test_callback_consistency.
    run!(solver, Q; tfinal=0.006, callback=Callback(EveryTime(0.002), writer))

    # Every rank must have written the same number of frames at the same times.
    # A rank that skipped one would have desynchronized the internal Allgather.
    check("frame count agrees on every rank",
          MPI.Allreduce(writer.index, max, comm) -
          MPI.Allreduce(writer.index, min, comm), 0.5)
    tspread = 0.0
    for m in 1:min(writer.index, 3)
        tspread = max(tspread, abs(writer.times[m] - 0.002m),
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
# 11. Slicing under decomposition. A slice is the one case in which a rank
#     legitimately has nothing to write: only the ranks whose block spans the
#     requested plane hold any of it. Those ranks must write no piece file and
#     must not be listed in the container, and the pieces that do exist must
#     still tile the plane. Serial runs cannot show any of this.
# ---------------------------------------------------------------------------
function test_slicing()
    section("slicing under decomposition")
    solver, Q = setup(Problem(domain=((0.0, 1.0), (0.0, 0.25), (0.0, 0.25)),
                              bcs=(per3[1], per3[2], per3[3]),
                              ic=(x, y, z) -> Prim(u=(0.1, 0, 0), p=1.0,
                                                   rho=1 + x)),
                      Numerics(n_global=(SPLITN, 16, 16), dims=splitdims(1)))

    # (a) Slice across the SPLIT dimension: exactly one rank spans the plane.
    save_vtk(solver, Q, "mpi_slice_x"; fields=(:rho,), slice=(1, SPLITN ÷ 2 + 1))
    MPI.Barrier(comm)
    pieces = rank == 0 ? length(filter(f -> startswith(f, "mpi_slice_x") &&
                                            endswith(f, ".vtr"), readdir())) : 0
    check("split-dimension slice writes one piece", abs(gsum(pieces) - 1), 0.5)
    listed = rank == 0 ? count("<Piece", read("mpi_slice_x.pvtr", String)) : 0
    check("container lists only the rank holding the plane",
          abs(gsum(listed) - 1), 0.5)
    whole = rank == 0 ?
        (occursin("WholeExtent=\"0 0 0 15 0 15\"",
                  read("mpi_slice_x.pvtr", String)) ? 1 : 0) : 0
    check("sliced dimension is one point thick", abs(gsum(whole) - 1), 0.5)
    MPI.Barrier(comm)
    rank == 0 && foreach(rm, filter(startswith("mpi_slice_x"), readdir()))
    MPI.Barrier(comm)

    # (b) Slice across an UNDIVIDED dimension: every rank spans the plane, and
    #     their pieces must tile it.
    save_vtk(solver, Q, "mpi_slice_z"; fields=(:rho,), slice=(3, 8))
    MPI.Barrier(comm)
    pieces = rank == 0 ? length(filter(f -> startswith(f, "mpi_slice_z") &&
                                            endswith(f, ".vtr"), readdir())) : 0
    check("undivided-dimension slice writes every piece",
          abs(gsum(pieces) - np), 0.5)
    total, inside = 0, 0
    if rank == 0
        txt = read("mpi_slice_z.pvtr", String)
        inside = 1
        for m in eachmatch(r"<Piece Extent=\"([^\"]+)\"", txt)
            e = parse.(Int, split(m.captures[1]))
            total += prod(ntuple(d -> e[2d] - e[2d-1] + 1, 3))
            (e[5] == 0 && e[6] == 0) || (inside = 0)
        end
    end
    check("sliced pieces tile the plane", abs(gsum(total) - SPLITN * 16), 0.5)
    check("every piece is flat in the sliced dimension",
          abs(gsum(inside) - 1), 0.5)
    MPI.Barrier(comm)
    rank == 0 && foreach(rm, filter(startswith("mpi_slice_z"), readdir()))
    MPI.Barrier(comm)

    # An out-of-range plane is rejected on every rank, not just the one that
    # would have held it.
    threw = try
        save_vtk(solver, Q, "mpi_slice_bad"; fields=(:rho,), slice=(1, SPLITN + 1))
        0
    catch e
        e isa ArgumentError ? 1 : 0
    end
    check("an out-of-range slice throws on every rank", abs(gsum(threw) - np), 0.5)
    MPI.Barrier(comm)
    rank == 0 && foreach(rm, filter(startswith("mpi_slice_bad"), readdir()))
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
    # The header describes the state, not only its shape. Each rank writes the
    # global coordinates rather than its own slice of them, so a solver holding
    # the same extents over a longer domain is refused on every rank at once
    # rather than on whichever ranks happen to notice.
    stretched = Solver(n_global=(SPLITN, 16, 16), L_domain=(4π, 2π, 2π), bcs=per3,
                       art=ArtParams(enabled=false), dims=splitdims(1))
    threw = try
        load_checkpoint!(stretched, allocate_state(stretched), "mpi_ckpt")
        0
    catch e
        occursin("grid coordinate mismatch", sprint(showerror, e)) ? 1 : 0
    end
    check("a different domain is refused on every rank", abs(gsum(threw) - np), 0.5)
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

# ---------------------------------------------------------------------------
# 7c. Artificial properties under decomposition. Every other test here disables
#     them, so the sensor path had no multi-rank coverage at all — and
#     `beta_sensor = :dilatation` adds a second distributed filter solve inside
#     `compute_artificial!`, on the far side of no early return. Splitting the
#     same field along each axis in turn must give the same integrated
#     coefficients: a halo the new path forgot to exchange changes the sensor
#     only near a rank boundary, which is exactly what a whole-domain sum of
#     three different decompositions catches.
# ---------------------------------------------------------------------------
function test_artificial_decomposition()
    section("artificial properties: coefficients independent of the split axis")
    N = SPLITN
    totals = Dict{Symbol,Vector{Float64}}()
    # The strain sensor reproduces to round-off. The dilatation sensor does not,
    # and the reason is in the switch rather than in any halo: H(−Δ) is
    # discontinuous at Δ = 0, and with the literature ε = 1e-32 the ratio is
    # still ≈ 1 there, so a point whose dilatation is zero to round-off toggles
    # its whole β* between 0 and C_β·ρ·sensor depending on which way the last bit
    # fell. Measured at 2e-6 relative for :gated_strain and 2e-7 for
    # :dilatation, against 1e-14 for the strain sensor — :gated_strain is the
    # worse of the two because the sensor it gates is O(1) at the points that
    # toggle. Each tolerance leaves one to two orders on top of the measured
    # value and stays two or more below the ~1e-2 an unexchanged halo would
    # produce. See reference/CALIBRATION.md.
    # `:ungated_dilatation` is the same sensor field with no switch on it, so it
    # goes back into the round-off class the strain sensor is in: the field
    # itself always reproduced, and only the discontinuity spoiled it.
    tol_of = Dict(:strain => 1e-10, :gated_strain => 1e-4, :dilatation => 1e-5,
                  :ungated_dilatation => 1e-10)
    for sensor in (:strain, :gated_strain, :dilatation, :ungated_dilatation)
        sums = Float64[]
        for ax in 1:3
            solver = Solver(n_global=(N, N, N), L_domain=(2π, 2π, 2π), bcs=per3,
                       art=ArtParams(enabled=true, beta_sensor=sensor),
                       dims=splitdims(ax))
            Q = allocate_state(solver)
            # Compressible and vortical at once, so both the dilatation gate and
            # the vorticity discriminator are away from their trivial values.
            initialize!(solver, Q, (x, y, z) ->
                Prim(u=(sin(x)cos(y)cos(z), -cos(x)sin(y)cos(z), 0.3sin(2z)),
                     p=1 + 0.2cos(x), rho=1 + 0.3sin(y)))
            compute_rhs!(solver, Q, zero(Q))
            loc = 0.0
            for k in 1:solver.decomp.n_local[3], j in 1:solver.decomp.n_local[2],
                i in 1:solver.decomp.n_local[1]
                loc += solver.beta_art[gidx(solver, i, j, k)]
            end
            push!(sums, gsum(loc))
        end
        totals[sensor] = sums
        spread = maximum(sums) - minimum(sums)
        check("$sensor: Σ β* spread over the three split axes",
              spread, tol_of[sensor] * max(maximum(sums), 1e-30))
    end
    # No two sensors may agree here, or the checks above would pass just as
    # happily on a `beta_sensor` that was being ignored. `check` tests val < tol,
    # so each relative difference is inverted into one: this passes when the
    # totals differ by more than 0.1%.
    for sensor in (:gated_strain, :dilatation, :ungated_dilatation)
        rel = abs(totals[:strain][1] - totals[sensor][1]) /
              max(totals[:strain][1], 1e-300)
        check("β* differs between strain and $sensor", 1e-3 / max(rel, 1e-300), 1.0)
    end
    # The μ* channel, whose two settings are the field it reads and how the
    # directions combine. Both are plain reductions with no switch, so both
    # reproduce to round-off; `:max` is the one setting in `ArtParams` that a
    # one-dimensional test cannot distinguish from the default at all, which
    # leaves this its only coverage outside Taylor–Green.
    musums = Dict{Symbol,Vector{Float64}}()
    for (label, art) in ((:strain_sum, ArtParams(enabled=true)),
                         (:velocity_sum, ArtParams(enabled=true, mu_sensor=:velocity)),
                         (:strain_max, ArtParams(enabled=true, reduction=:max)))
        sums = Float64[]
        for ax in 1:3
            solver = Solver(n_global=(N, N, N), L_domain=(2π, 2π, 2π), bcs=per3,
                       art=art, dims=splitdims(ax))
            Q = allocate_state(solver)
            initialize!(solver, Q, (x, y, z) ->
                Prim(u=(sin(x)cos(y)cos(z), -cos(x)sin(y)cos(z), 0.3sin(2z)),
                     p=1 + 0.2cos(x), rho=1 + 0.3sin(y)))
            compute_rhs!(solver, Q, zero(Q))
            loc = 0.0
            for k in 1:solver.decomp.n_local[3], j in 1:solver.decomp.n_local[2],
                i in 1:solver.decomp.n_local[1]
                loc += solver.mu_art[gidx(solver, i, j, k)]
            end
            push!(sums, gsum(loc))
        end
        musums[label] = sums
        check("$label: Σ μ* spread over the three split axes",
              maximum(sums) - minimum(sums), 1e-10 * max(maximum(sums), 1e-30))
    end
    for label in (:velocity_sum, :strain_max)
        rel = abs(musums[:strain_sum][1] - musums[label][1]) /
              max(musums[:strain_sum][1], 1e-300)
        check("μ* differs between strain_sum and $label", 1e-3 / max(rel, 1e-300), 1.0)
    end
end

# ---------------------------------------------------------------------------
# 7d. The d8 ringing detector under decomposition. `detector = :d8` puts a
#     pentadiagonal distributed line solve inside `compute_artificial!`, which
#     is the first time the banded reduced-interface stage — and its
#     `Allgather` — runs anywhere except the derivative path. It sits below no
#     early return and must be reached by every rank, so a mistake here hangs
#     rather than fails. Unlike the compression-keyed sensors this one has no
#     discontinuous switch in it, so it must reproduce to round-off.
# ---------------------------------------------------------------------------
function test_ring_detector_decomposition()
    section("d8 detector: sensors independent of the split axis")
    N = SPLITN
    totals = Dict{Symbol,NTuple{2,Vector{Float64}}}()
    for detector in (:delta4, :d8)
        bsums, ksums = Float64[], Float64[]
        for ax in 1:3
            solver = Solver(n_global=(N, N, N), L_domain=(2π, 2π, 2π), bcs=per3,
                       art=ArtParams(enabled=true, detector=detector),
                       dims=splitdims(ax))
            Q = allocate_state(solver)
            initialize!(solver, Q, (x, y, z) ->
                Prim(u=(sin(x)cos(y)cos(z), -cos(x)sin(y)cos(z), 0.3sin(2z)),
                     p=1 + 0.2cos(x), rho=1 + 0.3sin(y)))
            compute_rhs!(solver, Q, zero(Q))
            bloc, kloc = 0.0, 0.0
            for k in 1:solver.decomp.n_local[3], j in 1:solver.decomp.n_local[2],
                i in 1:solver.decomp.n_local[1]
                I = gidx(solver, i, j, k)
                bloc += solver.beta_art[I]
                kloc += solver.kappa_art[I]
            end
            push!(bsums, gsum(bloc)); push!(ksums, gsum(kloc))
        end
        totals[detector] = (bsums, ksums)
        for (label, sums) in (("β*", bsums), ("κ*", ksums))
            check("$detector: Σ $label spread over the three split axes",
                  maximum(sums) - minimum(sums), 1e-10 * max(maximum(sums), 1e-30))
        end
    end
    # As in 7c: the checks above would pass just as happily on a `detector`
    # that was being ignored, so require the two to disagree. κ* is the
    # channel where they separate by orders rather than by a factor, its input
    # being smooth where β*'s is not.
    rel = abs(totals[:delta4][2][1] - totals[:d8][2][1]) /
          max(totals[:delta4][2][1], 1e-300)
    check("κ* differs between delta4 and d8", 1e-3 / max(rel, 1e-300), 1.0)
end

function test_positivity_floor()
    section("positivity failsafe: same tally and same repair under any split")
    # Six cells damaged by hand at known global indices, so that whichever rank
    # owns each one, the reduced tally has to come out the same. Two carry a
    # negative partial density with the mixture density still healthy, two an
    # internal energy below the floor with the total energy still positive, and
    # two a negative total energy. Two species, so that the first pair exercises
    # the clip-and-rescale path rather than the density floor.
    #
    # The first entry of each triple is the index along whichever axis is split,
    # which is the only one carrying SPLITN points; the other two follow in
    # ascending dimension order. Writing them the other way round would put a
    # 66 on a 16-point dimension and silently damage nothing.
    damaged = ((7, 3, 3), (41, 9, 5), (13, 5, 11), (58, 2, 8), (22, 7, 2),
               (66, 12, 14))
    eos = IdealMixture([IdealSpecies{Float64}("a", 1.0, 1.4),
                        IdealSpecies{Float64}("b", 2.0, 1.6)])
    counts = Float64[]
    masses = Float64[]
    energies = Float64[]
    for ax in 1:3
        n_global = ntuple(d -> d == ax ? SPLITN : 16, 3)
        solver = Solver(n_global=n_global, L_domain=(2π, 2π, 2π), bcs=per3,
                        eos=eos, art=ArtParams(enabled=false), dims=splitdims(ax))
        others = filter(d -> d != ax, 1:3)
        Q = allocate_state(solver)
        initialize!(solver, Q, (x, y, z) -> Prim(rho=1.5, u=(0.4, 0.0, 0.0), p=1.0,
                                                 Y=(0.5, 0.5)))
        rho_floor, e_floor =
            CL.positivity_floors(solver, Q, StepControl(floor_ratio=1e-6))
        ie = solver.equations.i_energy
        m1 = solver.equations.i_mom[1]
        off = solver.decomp.offset
        nloc = solver.decomp.n_local
        for (n, g) in enumerate(damaged)
            gi = ntuple(d -> d == ax ? g[1] : (d == others[1] ? g[2] : g[3]), 3)
            # Global index to this rank's local one; skip what it does not own.
            loc = ntuple(d -> gi[d] - off[d], 3)
            all(d -> 1 <= loc[d] <= nloc[d], 1:3) || continue
            I = gidx(solver, loc...)
            if n <= 2
                ρ = mixture_density(solver, Q, I)
                Q[I, 1] = -0.5
                Q[I, 2] = ρ + 0.5                  # mixture density unchanged
            elseif n <= 4
                ke = 0.5 * Q[I, m1]^2 / 1.5
                Q[I, ie] = 0.95 * ke               # e < 0, E > 0
            else
                Q[I, ie] = -2.0                    # unrepresentable
            end
        end
        t = CL.apply_positivity_floor!(solver, Q, rho_floor, e_floor,
                                       :internal_energy)
        push!(counts, t.cells); push!(masses, t.mass); push!(energies, t.energy)
        check("split axis $ax: repaired cells (expect 6)", abs(t.cells - 6), 0.5)
        check("split axis $ax: low-energy cells (expect 4)",
              abs(t.low_energy - 4), 0.5)
        # Every damaged cell is now representable on the rank that owns it.
        nbad = 0.0
        for k in 1:nloc[3], j in 1:nloc[2], i in 1:nloc[1]
            I = gidx(solver, i, j, k)
            ρ = mixture_density(solver, Q, I)
            ρ >= rho_floor || (nbad += 1)
            ke = 0.5 * (Q[I, m1]^2 + Q[I, m1 + 1]^2 + Q[I, m1 + 2]^2) / ρ
            (Q[I, ie] - ke) / ρ >= e_floor * (1 - 1e-9) || (nbad += 1)
        end
        check("split axis $ax: cells left below a floor", gsum(nbad), 0.5)
    end
    # The tally is a reduced quantity, so it must not depend on who owns what.
    check("repaired-cell count independent of the split",
          maximum(counts) - minimum(counts), 0.5)
    check("mass added independent of the split",
          maximum(masses) - minimum(masses), 1e-12 * max(maximum(masses), 1e-30))
    check("energy added independent of the split",
          maximum(energies) - minimum(energies),
          1e-12 * max(maximum(energies), 1e-30))
end

# ---------------------------------------------------------------------------
# 8. Two-patch layout under rank partitioning. The patch machinery splits the
#    rank set with MPI.Comm_split, each patch running its own decomposition and
#    plans; the interface ghost exchange, shared-plane averaging and the
#    world-hoisted reductions in run! are all np > 1-only paths. Within a patch
#    the spike solve reproduces the single-domain answer bit for bit at any
#    rank count, and every cross-patch transfer moves values unchanged, so the
#    max-norm error of the run — an associativity-free reduction — must equal
#    the serial two-patch value EXACTLY at every np. The reference below was
#    measured with test/patch_tests.jl's entropy-wave case at np = 1.
# ---------------------------------------------------------------------------
function test_two_patch_layout()
    section("two-patch layout: rank-partitioned run matches the serial answer")
    u0 = 0.5
    ic(x, y, z) = Prim(u=(u0, 0, 0), p=1.0, rho=1.0 + 0.2 * sin(x))
    solver = Solver(n_global=(96, 1, 1), L_domain=(2π, 1.0, 1.0), bcs=per3,
                    art=ArtParams(enabled=false), filter_interval=0,
                    patch_grid=(2, 1, 1))
    states = allocate_state(solver)
    initialize!(solver, states, ic)
    run!(solver, states; tfinal=0.5)
    err = 0.0
    for (ps, Q) in CL.eachpatch(solver, states)
        for i in 1:ps.decomp.n_local[1]
            I = gidx(ps, i, 1, 1)
            exact = 1.0 + 0.2 * sin(xcoord(ps, 1, i) - u0 * solver.t)
            err = max(err, abs(Q[I, 1] - exact))
        end
    end
    gerr = gmax(err)
    # Serial (np = 1) value of the identical run. At np = 2 each patch sits on
    # one rank, the arithmetic is identical to the serial patch-sequential
    # order, and the reproduction is bitwise. At np ≥ 4 the patch itself is
    # decomposed and its closed lines take the spike/reduced path where the
    # serial patch ran a plain Thomas sweep, so the answer agrees to round-off
    # accumulation instead (measured 3.1e-15 at np = 4, 6.7e-16 at np = 8,
    # against a 9.5e-8 signal).
    ref = 9.5442217462604617e-8
    tol = np <= 2 ? 1e-20 : 1e-13
    check("two-patch entropy wave: max error matches serial", abs(gerr - ref), tol)
    check("two-patch run: step count matches serial", abs(solver.step - 28), 0.5)
end

# ---------------------------------------------------------------------------
# Device line solves (reference/AMR_GPU.md, G2). A DevicePlan runs the fill,
# sweep, spike correction and scatter as KernelAbstractions kernels and keeps
# the reduced interface stage on the host, mirroring the host arithmetic per
# line operation for operation — so on the KA CPU backend the decomposed
# comparison is BITWISE, not to round-off. Splitting each dimension in turn
# forces the cross-rank Allgather of the reduced stage through the device
# path's pack/unpack staging, for both the tridiagonal C6 (periodic wrap) and
# the pentadiagonal C10 (closed edge ranks carrying closure rows).
# ---------------------------------------------------------------------------
function test_device_lines()
    section("device line solves: DevicePlan bitwise vs host under decomposition")
    cpu = KernelAbstractions.CPU()
    for (label, scheme, periodic) in (("C6 periodic", CL.lele_d1_6(Float64), true),
                                      ("C10 closed", CL.lele_d1_10(Float64), false))
        for ax in 1:3
            n_global = ntuple(d -> d == ax ? SPLITN : 12, 3)
            decomp = Decomp(n_global, (periodic, periodic, periodic);
                            dims=splitdims(ax))
            plan = CL.plan_direction(decomp, scheme, ax, 0.05)
            dplan = device_plan(plan, cpu)
            f = CL.field(decomp)
            pad = decomp.n_halo_d; off = decomp.offset
            for k in 1:decomp.n_local[3], j in 1:decomp.n_local[2],
                    i in 1:decomp.n_local[1]
                gx, gy, gz = i + off[1], j + off[2], k + off[3]
                f[i+pad[1], j+pad[2], k+pad[3]] =
                    sin(0.31gx) + cos(0.17gy) + 0.5sin(0.23gz + 0.4gx)
            end
            CL.exchange_dim!(f, decomp, ax)
            out_h = CL.field(decomp); out_d = CL.field(decomp)
            CL.apply_along!(out_h, plan, f, decomp)
            CL.apply_along!(out_d, dplan, f, decomp)
            e = 0.0
            for k in 1:decomp.n_local[3], j in 1:decomp.n_local[2],
                    i in 1:decomp.n_local[1]
                e = max(e, abs(out_h[i+pad[1], j+pad[2], k+pad[3]] -
                               out_d[i+pad[1], j+pad[2], k+pad[3]]))
            end
            check("$label device path bitwise, dim $ax split", gmax(e), 1e-300)
        end
    end
end

# ---------------------------------------------------------------------------
# Staged device exchange (reference/AMR_GPU.md, G3b). A device-resident field
# packs each halo slab into a contiguous backend stage by broadcast, crosses
# ranks through one contiguous copy each way around the same Sendrecv, and the
# fold pairing stages its whole blocks the same way. On host arrays those are
# the identical element copies, so FORCE_DEVICE_EXCHANGE pins the staged path
# bitwise here — combined with FORCE_KA and a DeviceBackend(CPU()) solver,
# a full decomposed run takes the closest CPU analog of a distributed device
# run: DevicePlans, kernel bodies, and staged communication together.
# ---------------------------------------------------------------------------
function test_staged_exchange()
    section("staged device exchange: decomposed runs bitwise (G3b)")
    cpu = KernelAbstractions.CPU()
    function tube(backend)
        eos = IdealMixture([IdealSpecies{Float64}("light", 1.0, 1.4),
                            IdealSpecies{Float64}("heavy", 0.2, 1.09)])
        s = Solver(n_global=(SPLITN, 16, 12), L_domain=(1.0, 0.6, 0.3),
                   eos=eos, bcs=((SlipWallBC(), SlipWallBC()), per3[2], per3[3]),
                   dims=splitdims(1), backend=backend)
        Q = allocate_state(s)
        initialize!(s, Q, (x, y, z) -> begin
            θ = 0.5 * (1 + tanh((x - 0.5) / 0.05))
            Prim(Y=(1 - θ, θ), rho=(1 - θ) + 0.625θ, p=(1 - θ) + 0.1θ,
                 u=(0.1 * sin(2π * y / 0.6), 0.0, 0.05 * cos(2π * z / 0.3)))
        end)
        return s, Q
    end
    s1, Q1 = tube(CPUBackend())
    run!(s1, Q1; tfinal=0.02, nmax=6)
    CL.FORCE_KA[] = true
    CL.FORCE_DEVICE_EXCHANGE[] = true
    s2, Q2 = tube(DeviceBackend(cpu))
    try
        run!(s2, Q2; tfinal=0.02, nmax=6)
    finally
        CL.FORCE_KA[] = false
        CL.FORCE_DEVICE_EXCHANGE[] = false
    end
    check("staged multispecies tube, split x: bitwise",
          gmax(maximum(abs.(parent(Q1) .- parent(Q2)))), 1e-300)

    # Off-rank fold pairing: θ-split resolved axis, whole-block Sendrecv
    # through sendrecv_block!'s staging.
    function axis(backend)
        s = Solver(n_global=(20, SPLITN, 12), L_domain=(1.0, 2π, 0.5),
                   bcs=((AxisBC(), SlipWallBC()), per3[2], per3[3]),
                   metric=CylindricalMetric(), dims=splitdims(2),
                   backend=backend)
        Q = allocate_state(s)
        initialize!(s, Q, (r, θ, z) ->
            Prim(u=(0.05r * cos(θ), 0.2r, 0.05 * sin(2π * z / 0.5)),
                 p=1.0 + 0.02r^2, rho=1.0))
        return s, Q
    end
    s3, Q3 = axis(CPUBackend())
    run!(s3, Q3; tfinal=0.02, nmax=4)
    CL.FORCE_KA[] = true
    CL.FORCE_DEVICE_EXCHANGE[] = true
    s4, Q4 = axis(DeviceBackend(cpu))
    try
        run!(s4, Q4; tfinal=0.02, nmax=4)
    finally
        CL.FORCE_KA[] = false
        CL.FORCE_DEVICE_EXCHANGE[] = false
    end
    check("staged axis fold pair, split θ: bitwise",
          gmax(maximum(abs.(parent(Q3) .- parent(Q4)))), 1e-300)
end

const SUITE = (
    ("periodic C6", test_periodic_c6),
    ("pentadiagonal C10", test_pentadiagonal_c10),
    ("closed C6", test_closed_c6),
    ("device line solves", test_device_lines),
    ("staged device exchange", test_staged_exchange),
    ("AMR transfer pair", test_transfer_pair),
    ("halo consistency", test_halo_consistency),
    ("off-rank folds", test_offrank_folds),
    ("freestream", test_freestream),
    ("conservation", test_conservation),
    ("sync", test_sync),
    ("artificial decomposition", test_artificial_decomposition),
    ("d8 detector decomposition", test_ring_detector_decomposition),
    ("positivity floor", test_positivity_floor),
    ("callback consistency", test_callback_consistency),
    ("state queries", test_state_queries),
    ("field writer", test_field_writer),
    ("slicing", test_slicing),
    ("checkpoint", test_checkpoint),
    ("two-patch layout", test_two_patch_layout),
)

try
    # Barrier before each timing so that a slow rank in one test is charged to
    # that test rather than to the next one it holds up.
    for (name, testfn) in SUITE
        MPI.Barrier(comm)
        @phase name testfn()
    end
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

# Rank 0's own breakdown, plus two whole-communicator numbers: the slowest
# rank's total, and the compile time summed over ranks. The second is the one
# that scales with np — every rank compiles this suite from scratch, so the job
# pays np times for the same code generation while the runner has far fewer
# cores than ranks.
let total = time() - T_SCRIPT_START,
    compile = sum(p -> p[3], PHASE_LOG; init=0.0)
    slowest = MPI.Allreduce(total, max, comm)
    compile_all = MPI.Allreduce(compile, +, comm)
    rank == 0 && timing_report(; title="mpi phase timing (np = $np, rank 0)",
                               extra=("slowest rank (script)" => slowest,
                                      "compiling, summed over ranks" => compile_all))
end

if _nfail[] > 0
    exit(1)
end
rank == 0 && println("multi-rank tests complete")
MPI.Finalize()
