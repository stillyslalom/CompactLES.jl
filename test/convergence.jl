# Convergence and validation studies. These are slower than the unit tests
# and print tables rather than asserting hard tolerances (except where the
# expected order is unambiguous) — they are the second line of defence, run
# once the fast tests in runtests.jl and mpi_tests.jl are green.
#
#   julia --project=. -t auto test/convergence.jl
#
# What each study pins down:
#   1. Interior order — 6 for lele_d1_6, 10 for lele_d1_10 on periodic grids.
#      A wrong interior coefficient shows as a wrong slope, not a wrong level.
#   2. Closed-domain order — the global max-norm order drops to the closure
#      order near walls. A slope of 1 means the boundary rows are wrong; a
#      slope of 6 means the closures are never being hit.
#   3. Axis/origin/pole order — the sharpest scalar diagnostic of the fold
#      signs: a sign error usually gives O(1) error at the first node, so the
#      slope collapses to ~0 rather than degrading gracefully.
#   4. Taylor–Green kinetic-energy decay — an end-to-end physics check
#      against published Re = 1600 data.
#
# The `expect` values below are REGRESSION GUARDS set from measured behaviour,
# not from the formal interior order. Measured on this code (max norm):
#
#   C6 interior 6.01 | C10 interior 10.04 | C6 wall closures 3.17
#   cyl axis odd 3.71 | cyl axis even 3.00 | resolved-θ axis 3.71
#   spherical origin 2.99
#
# These are GLOBAL max norms, and every fold study closes its outer end with a
# SlipWallBC. The orders near 3 therefore belong to the WALL, not to the fold:
# the global max is attained at the last interior cell in all four fold studies,
# and splitting the norm by region shows the fold's own error converging at
# 6.05-7.01 and sitting three to five orders of magnitude below the interior.
# `bench/foldorder.jl` does that split and carries the numbers; the write-up is
# in reference/CALIBRATION.md under "The fold closure is not third order".
#
# So a fold study here guards two things at once, and only the weaker of them is
# about the fold. The slope confirms the outer wall's closure cascade, which is
# a known property rather than a defect. What the fold contributes is the LEVEL:
# a fold sign error gives O(1) error at the first node and collapses the slope
# to ~0, which is what item 3 above is really watching for.

# Timing first, so that package load is measured rather than assumed. See
# test/timing.jl for what the compile column means.
include("timing.jl")

@phase "package load" begin
    using MPI
    MPI.Init(threadlevel=:funneled)
    using CompactLES
    using Printf, Test
end

const CL = CompactLES
per3 = ntuple(_ -> (PeriodicBC(), PeriodicBC()), 3)

fillf!(solver, f, fn) = (for k in 1:solver.decomp.n_local[3], j in 1:solver.decomp.n_local[2],
                        i in 1:solver.decomp.n_local[1]
    f[gidx(solver, i, j, k)] = fn(xcoord(solver, 1, i), xcoord(solver, 2, j), xcoord(solver, 3, k))
end; f)

function ferr(solver, f, fn)
    e = 0.0
    for k in 1:solver.decomp.n_local[3], j in 1:solver.decomp.n_local[2], i in 1:solver.decomp.n_local[1]
        e = max(e, abs(f[gidx(solver, i, j, k)] -
                       fn(xcoord(solver, 1, i), xcoord(solver, 2, j), xcoord(solver, 3, k))))
    end
    e
end

"Least-squares slope of log(err) vs log(N), i.e. the observed order."
function order(Ns, errs)
    x = log.(Float64.(Ns)); y = log.(max.(errs, 1e-16))
    n = length(x)
    sx = sum(x); sy = sum(y)
    -(n * sum(x .* y) - sx * sy) / (n * sum(x .^ 2) - sx^2)
end

# Each study spends its time in two places: constructing a Solver per
# resolution, which plans every direction and builds the metric, and the single
# derivative that the order is measured from. Splitting the two across all
# studies says whether a cheaper study means fewer resolutions or smaller ones.
const T_BUILD = Ref(0.0)
const T_DERIV = Ref(0.0)

function study(name, Ns, build, fld, ref; expect=nothing, tol=1.0)
    t0 = time(); c0 = compile_ns()
    errs = Float64[]
    for N in Ns
        tb = time()
        solver = build(N)
        T_BUILD[] += time() - tb
        td = time()
        f = CL.field(solver.decomp); df = CL.field(solver.decomp)
        fillf!(solver, f, fld)
        CL.exchange_halos!(f, solver.decomp)
        CL.deriv_along!(df, f, solver, 1, ref.parity)
        CL._scale_grad!(df, solver, 1)
        push!(errs, ferr(solver, df, ref.fn))
        T_DERIV[] += time() - td
    end
    p = order(Ns, errs)
    @printf("%-38s  ", name)
    for (N, e) in zip(Ns, errs)
        @printf("N=%-4d %.3e  ", N, e)
    end
    @printf("order ≈ %.2f\n", p)
    push!(PHASE_LOG, (name, time() - t0, (compile_ns() - c0) / 1e9))
    expect === nothing || @test abs(p - expect) < tol
    p
end

println("\n=== interior order (periodic) ===")
study("C6 periodic derivative", (16, 32, 64),
      N -> Solver(n_global=(N, 12, 12), L_domain=(2π, 2π, 2π), bcs=per3,
                  art=ArtParams(enabled=false)),
      (x, y, z) -> sin(x),
      (fn=(x, y, z) -> cos(x), parity=1); expect=6.0, tol=1.2)

study("C10 periodic derivative", (16, 24, 32),
      N -> Solver(n_global=(N, 12, 12), L_domain=(2π, 2π, 2π), bcs=per3,
                  deriv=lele_d1_10(), art=ArtParams(enabled=false)),
      (x, y, z) -> sin(x),
      (fn=(x, y, z) -> cos(x), parity=1); expect=10.0, tol=2.5)

println("\n=== closed-domain order (boundary closures active) ===")
study("C6 with wall closures", (24, 48, 96),
      N -> Solver(n_global=(N, 12, 12), L_domain=(1.0, 1.0, 1.0),
                  bcs=((SlipWallBC(), SlipWallBC()), per3[2], per3[3]),
                  art=ArtParams(enabled=false)),
      (x, y, z) -> exp(sin(3x)),
      (fn=(x, y, z) -> 3cos(3x) * exp(sin(3x)), parity=1); expect=3.2, tol=0.8)

println("\n=== coordinate-singularity folds ===")
study("cylindrical axis, odd field (u_r-like)", (32, 64, 128),
      N -> Solver(n_global=(N, 1, 12), L_domain=(1.0, 1.0, 0.5),
                  metric=CylindricalMetric(),
                  bcs=((AxisBC(), SlipWallBC()), per3[2], per3[3]),
                  art=ArtParams(enabled=false)),
      (r, θ, z) -> r * exp(-4r^2),
      (fn=(r, θ, z) -> (1 - 8r^2) * exp(-4r^2), parity=-1); expect=3.7, tol=0.8)

study("cylindrical axis, even field (scalar)", (32, 64, 128),
      N -> Solver(n_global=(N, 1, 12), L_domain=(1.0, 1.0, 0.5),
                  metric=CylindricalMetric(),
                  bcs=((AxisBC(), SlipWallBC()), per3[2], per3[3]),
                  art=ArtParams(enabled=false)),
      (r, θ, z) -> exp(-4r^2),
      (fn=(r, θ, z) -> -8r * exp(-4r^2), parity=1); expect=3.0, tol=0.8)

study("resolved-θ axis, x-like field", (32, 64, 128),
      N -> Solver(n_global=(N, 16, 1), L_domain=(1.0, 2π, 1.0),
                  metric=CylindricalMetric(),
                  bcs=((AxisBC(), SlipWallBC()), per3[2], per3[3]),
                  art=ArtParams(enabled=false)),
      (r, θ, z) -> r * cos(θ) * exp(-4r^2),
      (fn=(r, θ, z) -> cos(θ) * (1 - 8r^2) * exp(-4r^2), parity=-1);
      expect=3.7, tol=0.8)

study("spherical origin, radial Gaussian", (24, 48, 96),
      N -> Solver(n_global=(N, 12, 12), L_domain=(1.0, π, 2π),
                  metric=SphericalMetric(),
                  bcs=((OriginBC(), SlipWallBC()),
                       (PoleBC(), PoleBC()), per3[3]),
                  art=ArtParams(enabled=false)),
      (r, θ, φ) -> exp(-4r^2),
      (fn=(r, θ, φ) -> -8r * exp(-4r^2), parity=1); expect=3.0, tol=0.8)

# ---------------------------------------------------------------------------
# Taylor–Green vortex: dissipation-rate history at Re = 1600. Reference peak
# dissipation occurs near t ≈ 9 with ε ≈ 1.2e-2 (van Rees et al. 2011);
# at 64³ a well-behaved code lands within a few percent, and a broken
# viscous term or filter shows up as a badly misplaced or damped peak.
# This is the long one — comment it out for quick iterations.

function taylor_green_ke(N; tfinal=10.0, Re=1600.0)
    γ = 1.4; c0 = 10.0; p0 = c0^2 / γ
    prob = Problem(eos=single_species(gamma=γ),
                   transport=Transport(mu0=1 / Re),
                   domain=((0.0, 2π), (0.0, 2π), (0.0, 2π)), bcs=per3,
                   ic=(x, y, z) -> Prim(
                       u=(sin(x) * cos(y) * cos(z), -cos(x) * sin(y) * cos(z), 0.0),
                       p=p0 + (1 / 16) * (cos(2x) + cos(2y)) * (cos(2z) + 2),
                       rho=1.0))
    solver, Q = setup(prob, Numerics(n_global=(N, N, N), art=ArtParams(enabled=false),
                                cfl=0.6))
    cellvol = prod(solver.h)
    ts = Float64[]; kes = Float64[]
    cb = (solver, Q) -> begin
        ke = 0.0
        for k in 1:solver.decomp.n_local[3], j in 1:solver.decomp.n_local[2], i in 1:solver.decomp.n_local[1]
            I = gidx(solver, i, j, k)
            ρ = Q[I, 1]
            m1, m2, m3 = solver.equations.i_mom
            ke += 0.5 * (Q[I, m1]^2 + Q[I, m2]^2 + Q[I, m3]^2) / ρ
        end
        ke = MPI.Allreduce(ke * cellvol, +, solver.decomp.comm) / (2π)^3
        push!(ts, solver.t); push!(kes, ke)
    end
    run!(solver, Q; tfinal=tfinal, callback=cb)
    ts, kes
end

if get(ENV, "CL_RUN_TG", "0") == "1"
    @phase "Taylor–Green 64³" begin
        println("\n=== Taylor–Green Re=1600, 64³ (set CL_RUN_TG=1 to enable) ===")
        ts, kes = taylor_green_ke(64)
        eps = -diff(kes) ./ diff(ts)
        imax = argmax(eps)
        @printf("peak dissipation %.4e at t = %.2f (reference ≈ 1.2e-2 at t ≈ 9)\n",
                eps[imax], ts[imax+1])
    end
end

println("\nconvergence studies complete")
timing_report(; title="convergence phase timing",
              extra=("of which Solver construction" => T_BUILD[],
                     "of which derivative and error" => T_DERIV[]))
