# Where the fold studies' error actually lives, by region of the line.
#
#   julia --project=. -t auto bench/foldorder.jl
#
# --- Why this exists ---------------------------------------------------------
#
# test/convergence.jl measures a max norm over the whole line, and every one of
# its fold studies closes the outer end with a SlipWallBC whose closure rows
# measure 3.17 on their own. A reported fold order near 3 is
# therefore consistent with two different worlds: the fold is third order, or
# the fold is accurate and the outer wall dominates the norm. Splitting the norm
# by region separates them, and the answer is the second one — see
# reference/CALIBRATION.md under "The fold closure is not third order".
#
# The first study is the control: both ends are walls, so the fold window is a
# wall window and must report the wall order. It does, at 3.23.
#
# Columns are max norms over disjoint index windows of the radial line:
#
#   fold(1:W)   the first W interior cells, i.e. the singular end
#   mid         everything between the two windows
#   outer(W)    the last W interior cells, i.e. the SlipWall end
#   argmax i/n  where the global max norm sits, which is the whole point
#
# Scratch tooling, like everything else in bench/: it prints a table and asserts
# nothing. It runs the same fields and resolutions as test/convergence.jl so the
# regional numbers compose back to the global orders that file guards.

using MPI
MPI.Init(threadlevel=:funneled)
using CompactLES
using Printf

const CL = CompactLES
const W = 3
const per3 = ntuple(_ -> (PeriodicBC(), PeriodicBC()), 3)

"Max norm of `f - fn` over the fold window, the middle, and the outer window."
function regional_errs(solver, f, fn)
    n1, n2, n3 = solver.decomp.n_local
    fold = 0.0; mid = 0.0; outer = 0.0; imax = 0; emax = 0.0
    for k in 1:n3, j in 1:n2, i in 1:n1
        e = abs(f[gidx(solver, i, j, k)] -
                fn(xcoord(solver, 1, i), xcoord(solver, 2, j),
                   xcoord(solver, 3, k)))
        if e > emax
            emax = e; imax = i
        end
        if i <= W
            fold = max(fold, e)
        elseif i > n1 - W
            outer = max(outer, e)
        else
            mid = max(mid, e)
        end
    end
    return (fold = fold, mid = mid, outer = outer, imax = imax, n = n1)
end

"Least-squares slope of log(err) vs log(N), i.e. the observed order."
function order(Ns, errs)
    x = log.(Float64.(Ns)); y = log.(max.(errs, 1e-16))
    n = length(x); sx = sum(x); sy = sum(y)
    return -(n * sum(x .* y) - sx * sy) / (n * sum(x .^ 2) - sx^2)
end

function study(name, Ns, build, fld, ref)
    rows = map(Ns) do N
        solver = build(N)
        f = CL.field(solver.decomp); df = CL.field(solver.decomp)
        for k in 1:solver.decomp.n_local[3], j in 1:solver.decomp.n_local[2],
            i in 1:solver.decomp.n_local[1]
            f[gidx(solver, i, j, k)] = fld(xcoord(solver, 1, i),
                                           xcoord(solver, 2, j),
                                           xcoord(solver, 3, k))
        end
        CL.exchange_halos!(f, solver.decomp)
        CL.deriv_along!(df, f, solver, 1, ref.parity)
        CL._scale_grad!(df, solver, 1)
        regional_errs(solver, df, ref.fn)
    end
    println("\n", name)
    @printf("  %-6s %12s %12s %12s   %s\n",
            "N", "fold(1:$W)", "mid", "outer($W)", "argmax i / n")
    for (N, r) in zip(Ns, rows)
        @printf("  %-6d %12.3e %12.3e %12.3e   %d / %d\n",
                N, r.fold, r.mid, r.outer, r.imax, r.n)
    end
    @printf("  order  %12.2f %12.2f %12.2f\n",
            order(Ns, [r.fold for r in rows]),
            order(Ns, [r.mid for r in rows]),
            order(Ns, [r.outer for r in rows]))
end

# The control. Both ends are walls, so "fold" is a second wall window here and
# is expected to report the wall order rather than a fold order.
study("C6, both ends walls (control)", (24, 48, 96),
      N -> Solver(n_global=(N, 12, 12), L_domain=(1.0, 1.0, 1.0),
                  bcs=((SlipWallBC(), SlipWallBC()), per3[2], per3[3]),
                  art=ArtParams(enabled=false)),
      (x, y, z) -> exp(sin(3x)),
      (fn=(x, y, z) -> 3cos(3x) * exp(sin(3x)), parity=1))

study("cylindrical axis, odd field (u_r-like)", (32, 64, 128),
      N -> Solver(n_global=(N, 1, 12), L_domain=(1.0, 1.0, 0.5),
                  metric=CylindricalMetric(),
                  bcs=((AxisBC(), SlipWallBC()), per3[2], per3[3]),
                  art=ArtParams(enabled=false)),
      (r, θ, z) -> r * exp(-4r^2),
      (fn=(r, θ, z) -> (1 - 8r^2) * exp(-4r^2), parity=-1))

study("cylindrical axis, even field (scalar)", (32, 64, 128),
      N -> Solver(n_global=(N, 1, 12), L_domain=(1.0, 1.0, 0.5),
                  metric=CylindricalMetric(),
                  bcs=((AxisBC(), SlipWallBC()), per3[2], per3[3]),
                  art=ArtParams(enabled=false)),
      (r, θ, z) -> exp(-4r^2),
      (fn=(r, θ, z) -> -8r * exp(-4r^2), parity=1))

study("spherical origin, even field (scalar)", (24, 48, 96),
      N -> Solver(n_global=(N, 12, 12), L_domain=(1.0, π, 2π),
                  metric=SphericalMetric(),
                  bcs=((OriginBC(), SlipWallBC()), (PoleBC(), PoleBC()), per3[3]),
                  art=ArtParams(enabled=false)),
      (r, θ, φ) -> exp(-4r^2),
      (fn=(r, θ, φ) -> -8r * exp(-4r^2), parity=1))

# Not in test/convergence.jl. u_r is odd across the origin, so this is the
# parity a converging Noh calculation actually differentiates there.
study("spherical origin, odd field (u_r-like)", (24, 48, 96),
      N -> Solver(n_global=(N, 12, 12), L_domain=(1.0, π, 2π),
                  metric=SphericalMetric(),
                  bcs=((OriginBC(), SlipWallBC()), (PoleBC(), PoleBC()), per3[3]),
                  art=ArtParams(enabled=false)),
      (r, θ, φ) -> r * exp(-4r^2),
      (fn=(r, θ, φ) -> (1 - 8r^2) * exp(-4r^2), parity=-1))

println("\nfoldorder complete")
