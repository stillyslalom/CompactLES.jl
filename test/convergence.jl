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
# The boundary closures and the singular-axis folds are third-order in the max
# norm — the error is dominated by the first node or two off the wall/axis,
# which is why the serial suite's fold tolerances carry the same note. That is
# a known property of the closure cascade, not a defect these studies found;
# what they guard is that it does not get *worse*.

using MPI
MPI.Init(threadlevel=:funneled)
using CompactLES
using Printf, Test

const CL = CompactLES
per3 = ntuple(_ -> (PeriodicBC(), PeriodicBC()), 3)

fillf!(s, f, fn) = (for k in 1:s.decomp.n_local[3], j in 1:s.decomp.n_local[2],
                        i in 1:s.decomp.n_local[1]
    f[gidx(s, i, j, k)] = fn(xcoord(s, 1, i), xcoord(s, 2, j), xcoord(s, 3, k))
end; f)

function ferr(s, f, fn)
    e = 0.0
    for k in 1:s.decomp.n_local[3], j in 1:s.decomp.n_local[2], i in 1:s.decomp.n_local[1]
        e = max(e, abs(f[gidx(s, i, j, k)] -
                       fn(xcoord(s, 1, i), xcoord(s, 2, j), xcoord(s, 3, k))))
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

function study(name, Ns, build, fld, ref; expect=nothing, tol=1.0)
    errs = Float64[]
    for N in Ns
        s = build(N)
        f = CL.field(s.decomp); df = CL.field(s.decomp)
        fillf!(s, f, fld)
        CL.exchange_halos!(f, s.decomp)
        CL.deriv_along!(df, f, s, 1, ref.parity)
        CL._scale_grad!(df, s, 1)
        push!(errs, ferr(s, df, ref.fn))
    end
    p = order(Ns, errs)
    @printf("%-38s  ", name)
    for (N, e) in zip(Ns, errs)
        @printf("N=%-4d %.3e  ", N, e)
    end
    @printf("order ≈ %.2f\n", p)
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
    s, Q = setup(prob, Numerics(n_global=(N, N, N), art=ArtParams(enabled=false),
                                cfl=0.6))
    cellvol = prod(s.h)
    ts = Float64[]; kes = Float64[]
    cb = (s, Q) -> begin
        ke = 0.0
        for k in 1:s.decomp.n_local[3], j in 1:s.decomp.n_local[2], i in 1:s.decomp.n_local[1]
            I = gidx(s, i, j, k)
            ρ = Q[I, 1]
            ke += 0.5 * (Q[I, s.i_mom[1]]^2 + Q[I, s.i_mom[2]]^2 + Q[I, s.i_mom[3]]^2) / ρ
        end
        ke = MPI.Allreduce(ke * cellvol, +, s.decomp.comm) / (2π)^3
        push!(ts, s.t); push!(kes, ke)
    end
    run!(s, Q; tfinal=tfinal, callback=cb)
    ts, kes
end

if get(ENV, "CL_RUN_TG", "0") == "1"
    println("\n=== Taylor–Green Re=1600, 64³ (set CL_RUN_TG=1 to enable) ===")
    ts, kes = taylor_green_ke(64)
    eps = -diff(kes) ./ diff(ts)
    imax = argmax(eps)
    @printf("peak dissipation %.4e at t = %.2f (reference ≈ 1.2e-2 at t ≈ 9)\n",
            eps[imax], ts[imax+1])
end

println("\nconvergence studies complete")
