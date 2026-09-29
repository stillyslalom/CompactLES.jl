# Convergence and validation studies. These are slower than the unit tests and
# print a table of measured orders as well as asserting them; they are the
# second line of defence, run once the fast tests in runtests.jl and
# mpi_tests.jl are green.
#
#   julia --project=. -t auto test/convergence.jl
#
# What each study pins down:
#   1. Interior order — 6 for lele_d1_6, 8 for lele_d1_8, 10 for lele_d1_10
#      on periodic grids. A wrong interior coefficient shows as a wrong slope,
#      not a wrong level.
#   2. Closed-domain order — the global max-norm order drops to the closure
#      order near walls. A slope of 1 means the boundary rows are wrong; a
#      slope of 6 means the closures are never being hit. The default
#      `:neutral3` closures are measured alongside the `:cascade3`,
#      `:cascade4` and `:brady_livescu` alternatives, whose whole purpose is
#      this slope, and one pass of the state filter is measured the same way
#      under its two wall closures, since a filtered run cannot exceed the
#      filter's order.
#   3. Axis/origin/pole order — the sharpest scalar diagnostic of the fold
#      signs: a sign error usually gives O(1) error at the first node, so the
#      slope collapses to ~0 rather than degrading gracefully. The symmetry
#      plane is the same fold on a Cartesian dimension, and its rows read the
#      interior order on both parities because no closure row is planned at a
#      folded end.
#   4. Closure truncation on a polynomial — one derivative of x^(q+1), q the
#      closure rows' exactness degree, against the actual spacing: the rows'
#      own pointwise order, 3/4/5/7 for :cascade3/:cascade4/C6 and C8
#      :brady_livescu, which the smooth-field slopes of item 2 sit above by
#      the favorable phase of exp(sin(3x)).
#   4b. Staggered operators — the implicit diffusion operator's derivatives
#      between nodes and midpoints, the midpoint interpolation and their
#      composition, periodic, under the wall mirror and between two
#      symmetry planes on fields of its parity: all six, since neither plans
#      a closure row. The composition on the solver's metrics reads six on
#      a stretched line, five and four through the folds of the axis, the
#      poles and the origin, where J⁻¹ is singular, and one at the node of a
#      curved wall.
#   5. Smooth evolution — the final-time solution error of the wall window,
#      or of the patch/level interface window, on cases with a reference
#      free of closure error (test/smooth_cases.jl). These are the orders a
#      run sees: the closure's, plus what the solution norm gains over the
#      pointwise truncation; the cascade filter caps every wall row near 1.8
#      and the one-sided rows lift the cap. An interface under the default
#      ghost fluxes reads the interior order, 6.0 through a level and 6.8
#      through two patches; under the closure rows a level interface reads
#      the C6 closure cascade at the fine spacing, 3.6, and :brady_livescu
#      6.0.
#   6. Temporal order — one case on one grid in a sequence of equal steps
#      against the same case in many more steps, so the spatial error cancels
#      and the slope is the time integration's: time-dependent boundary data
#      evaluated at the stage times read the integrator's fourth order, and a
#      level interface reads the coupling schedule's order.
#   7. Taylor–Green kinetic-energy decay — an end-to-end physics check
#      against published Re = 1600 data.
#
# The `expect` values below are REGRESSION GUARDS set from measured behaviour,
# not from the formal interior order. Measured on this code (max norm):
#
#   C6 interior 6.01 | C8 interior 8.00 | C10 interior 10.04
#   C6 wall closures 3.18 | C6 wall closures :cascade3 3.17 | :cascade4 4.02
#   C6 wall closures :brady_livescu 5.88 | C8 wall closures 3.18 |
#   C8 wall closures :brady_livescu 7.91 | C10 wall closures 3.18
#   filter pass :cascade 1.88 | filter pass :onesided 8.07
#   symmetry planes C6 even 6.00 / odd 6.00 | C8 even 8.05 / odd 7.95 |
#   C10 even 10.23 / odd 10.17 | filter pass between planes 7.88
#   cyl axis odd 3.76 | cyl axis even 2.99 | resolved-θ axis 3.76
#   spherical origin 2.97
#   polynomial rows: C6 :neutral3 3.00 | :cascade3 3.00 | :cascade4 4.00 |
#   C6 :brady_livescu 5.00 | C8 :neutral3 3.00 | C8 :brady_livescu 7.00 |
#   C10 :neutral3 3.00
#   staggered operators: periodic D_s 6.04 | G 6.07 | interpolation 5.97 |
#   L 6.06 | wall mirror D_s 6.00 | G 6.02 | interpolation 5.99 | L 6.02 |
#   symmetry planes D_s 6.01 | G 5.98 | interpolation 6.00 | L 5.98 | L on an
#   odd T 6.01 | L stretched 6.06 | cylindrical axis 5.96 | resolved-θ axis
#   5.07 | spherical origin 4.03 | spherical poles 5.33 | curved wall 1.00
#   wall evolution (window max norm, t = 0.4): inviscid C6 4.01 | inviscid C6
#   :cascade3 3.93 | cascade filter 1.94 | onesided filter 3.90 |
#   C6 :brady_livescu 5.73 | viscous no-slip C6 4.00 | viscous slip C6 4.00 |
#   shear mode 4.67
#   symmetry-plane evolution (window max norm, t = 0.4, against the fine
#   folded mirror): inviscid C6 4.46 | C6 onesided filter 4.69 | C8 4.00 |
#   C10 4.00 | viscous slip with shear C6 6.04
#   interface evolution (entropy wave, t = 0.5): two patches C6 (k = 1) 6.79 |
#   two levels C6 (k = 3) 6.01 | two levels, closure rows 3.62 | two levels
#   :brady_livescu 6.01 | three levels subcycled 6.00 | two levels, cascade
#   filter 6.87 | two levels :brady_livescu, d8 detector 5.93 | two levels,
#   pentadiagonal filter 6.01
#   a level at a wall (fine wall window against the uniform run at the fine
#   spacing, equal steps, t = 0.4): two levels at a slip wall C6 4.68 |
#   two levels at a symmetry plane C6 6.33
#   a level on an r-z annulus (fine interface window against the uniform run
#   at the fine spacing, equal steps, t = 0.2): C6 5.83 | viscous 5.40
#   a level at an NSCBC outflow (fine face window against the uniform run at
#   the fine spacing, equal steps, t = 0.35): C6 6.45
#   temporal order (fixed grid, equal steps): Dirichlet inflow g(t) 3.99 |
#   NSCBC inflow target(t) 4.09 | two levels, global step 1.00 | two levels
#   subcycled, ghost fluxes 3.85 | additive pair, viscous periodic 4.02 |
#   additive pair, no-slip walls at Pr = 0.007 3.63
#
# The default of all three derivative presets is `:neutral3`; the `:cascade3`
# rows are measured beside it wherever the two closures differ. The
# coordinate-singularity studies close their outer end with a wall and take
# the default rows; a symmetry plane plans no closure row at all. The
# interface studies take the default ghost fluxes, which difference the flux
# through an interface end with the interior rows; the closure-rows row and
# the two :brady_livescu rows take `interface_flux = :closure`, under which
# the flux divergence at an interface end selects the cascade rows
# (`interface_divergence_closures`) or the source scheme's.
#
# Those eighty-one numbers are also passed to each study as `recorded` and
# guarded to ±0.02, separately from the wide `expect`/`tol` pair. See the
# comment on `study` for which failure each guard reports. Each study also
# prints the order of the L2 norm over the interior, unguarded: the max norm
# is set by the wall rows alone, and Gustafsson's theorem allows the solution
# norm one order more than a boundary closure delivers pointwise, so the two
# together say what a closure set buys in a solution rather than at the wall.
#
# These are GLOBAL max norms, and every coordinate-singularity study closes
# its outer end with a SlipWallBC. The orders near 3 therefore belong to the
# WALL, not to the fold: the global max is attained at the last interior cell
# in all four of them, and splitting the norm by region shows the fold's own
# error converging at 6.05-7.01 and sitting three to five orders of magnitude
# below the interior. `bench/foldorder.jl` does that split and carries the
# numbers; the write-up is in reference/CALIBRATION_APPENDIX.md under "The fold
# closure is not third order". The symmetry-plane studies close neither end
# with a wall, which is why they report the fold's own order directly.
#
# So a fold study here guards two things at once, and only the weaker of them is
# about the fold. The slope confirms the outer wall's closure cascade, a known
# property, not a defect. The fold contributes the error level: a fold sign error
# gives O(1) error at the first node and collapses the slope to ~0. Item 3 guards
# against that collapse.

# Timing first, ensuring package load is measured rather than assumed. See
# test/timing.jl for what the compile column means.
include("timing.jl")

@phase "package load" begin
    using MPI
    MPI.Init(threadlevel=:funneled)
    using CompactLES
    using Printf, Test
end
using CompactLES: padded_index, xcoord

const CL = CompactLES
# The smooth-evolution cases, their references and regional norms; shared
# with bench/boundaryorder.jl, which runs the full matrix. Defines `per3`.
include("smooth_cases.jl")

fillf!(solver, f, fn) = (for k in 1:solver.decomp.n_local[3], j in 1:solver.decomp.n_local[2],
                        i in 1:solver.decomp.n_local[1]
    f[padded_index(solver, i, j, k)] = fn(xcoord(solver, 1, i), xcoord(solver, 2, j),
                                          xcoord(solver, 3, k))
end; f)

function ferr(solver, f, fn)
    e = 0.0
    for k in 1:solver.decomp.n_local[3], j in 1:solver.decomp.n_local[2], i in 1:solver.decomp.n_local[1]
        e = max(e, abs(f[padded_index(solver, i, j, k)] -
                       fn(xcoord(solver, 1, i), xcoord(solver, 2, j), xcoord(solver, 3, k))))
    end
    e
end

"Root-mean-square error over the interior, the discrete L2 norm per point."
function ferr2(solver, f, fn)
    e = 0.0; n = 0
    nl = solver.decomp.n_local
    for k in 1:nl[3], j in 1:nl[2], i in 1:nl[1]
        e += (f[padded_index(solver, i, j, k)] -
              fn(xcoord(solver, 1, i), xcoord(solver, 2, j), xcoord(solver, 3, k)))^2
        n += 1
    end
    sqrt(e / n)
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

# Each study carries two guards, which fail for different reasons.
#
#   `expect` / `tol` is wide. It detects a regressed order: a wrong
#   interior coefficient, a wrong closure row or a fold sign error moves the
#   slope by whole integers, and the width leaves room for the scatter of a
#   three-point least-squares fit.
#
#   `recorded` is the order this code measures today, listed in the header
#   above, guarded to DRIFT_TOL. A change not intended to affect numerics
#   reproduces it to the printed precision, so a moved digit says the change
#   reached the numerics — which is information, not necessarily a defect.
#
# Update `recorded` only together with the header table, and only once the
# cause of the move is understood.
const DRIFT_TOL = 0.02

function study(name, Ns, build, fld, ref; expect=nothing, tol=1.0, recorded=nothing,
               op=:deriv)
    t0 = time(); c0 = compile_ns()
    errs = Float64[]; errs2 = Float64[]
    for N in Ns
        tb = time()
        solver = build(N)
        T_BUILD[] += time() - tb
        td = time()
        f = CL.field(solver.decomp); df = CL.field(solver.decomp)
        fillf!(solver, f, fld)
        CL.exchange_halos!(f, solver.decomp)
        if op === :deriv
            CL.deriv_along!(df, f, solver, 1, ref.parity)
            CL._scale_grad!(df, solver, 1)
        else
            CL.filt_along!(df, f, solver, 1, ref.parity)
        end
        push!(errs, ferr(solver, df, ref.fn))
        push!(errs2, ferr2(solver, df, ref.fn))
        T_DERIV[] += time() - td
    end
    p = order(Ns, errs)
    @printf("%-38s  ", name)
    for (N, e) in zip(Ns, errs)
        @printf("N=%-4d %.3e  ", N, e)
    end
    @printf("order ≈ %.2f  (L2 %.2f)\n", p, order(Ns, errs2))
    push!(PHASE_LOG, (name, time() - t0, (compile_ns() - c0) / 1e9))
    if expect !== nothing
        abs(p - expect) < tol || println(
            "  ORDER REGRESSED: $(round(p, digits=2)) is outside $expect ± $tol. " *
            "That is a wrong coefficient, closure row or fold sign, not a drift.")
        @test abs(p - expect) < tol
    end
    if recorded !== nothing
        abs(p - recorded) < DRIFT_TOL || println(
            "  ORDER DRIFTED: $(round(p, digits=2)) against the recorded $recorded. " *
            "The scheme is intact and something reached the numerics. Find the " *
            "cause before updating the recorded value here and in the header.")
        @test abs(p - recorded) < DRIFT_TOL
    end
    p
end

println("\n=== interior order (periodic) ===")
study("C6 periodic derivative", (16, 32, 64),
      N -> Solver(n_global=(N, 12, 12), L_domain=(2π, 2π, 2π), bcs=per3,
                  art=ArtificialProperties(enabled=false)),
      (x, y, z) -> sin(x),
      (fn=(x, y, z) -> cos(x), parity=1); expect=6.0, tol=1.2, recorded=6.01)

study("C8 periodic derivative", (16, 32, 64),
      N -> Solver(n_global=(N, 12, 12), L_domain=(2π, 2π, 2π), bcs=per3,
                  deriv=lele_d1_8(), art=ArtificialProperties(enabled=false)),
      (x, y, z) -> sin(x),
      (fn=(x, y, z) -> cos(x), parity=1); expect=8.0, tol=1.5, recorded=8.00)

study("C10 periodic derivative", (16, 24, 32),
      N -> Solver(n_global=(N, 12, 12), L_domain=(2π, 2π, 2π), bcs=per3,
                  deriv=lele_d1_10(), art=ArtificialProperties(enabled=false)),
      (x, y, z) -> sin(x),
      (fn=(x, y, z) -> cos(x), parity=1); expect=10.0, tol=2.5, recorded=10.04)

println("\n=== closed-domain order (boundary closures active) ===")
study("C6 with wall closures", (24, 48, 96),
      N -> Solver(n_global=(N, 12, 12), L_domain=(1.0, 1.0, 1.0),
                  bcs=((SlipWallBC(), SlipWallBC()), per3[2], per3[3]),
                  art=ArtificialProperties(enabled=false)),
      (x, y, z) -> exp(sin(3x)),
      (fn=(x, y, z) -> 3cos(3x) * exp(sin(3x)), parity=1);
      expect=3.2, tol=0.8, recorded=3.18)

study("C6 wall closures, :cascade3", (24, 48, 96),
      N -> Solver(n_global=(N, 12, 12), L_domain=(1.0, 1.0, 1.0),
                  bcs=((SlipWallBC(), SlipWallBC()), per3[2], per3[3]),
                  deriv=lele_d1_6(closures=:cascade3),
                  art=ArtificialProperties(enabled=false)),
      (x, y, z) -> exp(sin(3x)),
      (fn=(x, y, z) -> 3cos(3x) * exp(sin(3x)), parity=1);
      expect=3.2, tol=0.8, recorded=3.17)

study("C6 wall closures, :cascade4", (24, 48, 96),
      N -> Solver(n_global=(N, 12, 12), L_domain=(1.0, 1.0, 1.0),
                  bcs=((SlipWallBC(), SlipWallBC()), per3[2], per3[3]),
                  deriv=lele_d1_6(closures=:cascade4),
                  art=ArtificialProperties(enabled=false)),
      (x, y, z) -> exp(sin(3x)),
      (fn=(x, y, z) -> 3cos(3x) * exp(sin(3x)), parity=1);
      expect=4.0, tol=0.8, recorded=4.02)

study("C6 wall closures, :brady_livescu", (24, 48, 96),
      N -> Solver(n_global=(N, 12, 12), L_domain=(1.0, 1.0, 1.0),
                  bcs=((SlipWallBC(), SlipWallBC()), per3[2], per3[3]),
                  deriv=lele_d1_6(closures=:brady_livescu),
                  art=ArtificialProperties(enabled=false)),
      (x, y, z) -> exp(sin(3x)),
      (fn=(x, y, z) -> 3cos(3x) * exp(sin(3x)), parity=1);
      expect=6.0, tol=1.0, recorded=5.88)

study("C8 with wall closures", (24, 48, 96),
      N -> Solver(n_global=(N, 12, 12), L_domain=(1.0, 1.0, 1.0),
                  bcs=((SlipWallBC(), SlipWallBC()), per3[2], per3[3]),
                  deriv=lele_d1_8(), art=ArtificialProperties(enabled=false)),
      (x, y, z) -> exp(sin(3x)),
      (fn=(x, y, z) -> 3cos(3x) * exp(sin(3x)), parity=1);
      expect=3.2, tol=0.8, recorded=3.18)

study("C8 wall closures, :brady_livescu", (24, 48, 96),
      N -> Solver(n_global=(N, 12, 12), L_domain=(1.0, 1.0, 1.0),
                  bcs=((SlipWallBC(), SlipWallBC()), per3[2], per3[3]),
                  deriv=lele_d1_8(closures=:brady_livescu),
                  art=ArtificialProperties(enabled=false)),
      (x, y, z) -> exp(sin(3x)),
      (fn=(x, y, z) -> 3cos(3x) * exp(sin(3x)), parity=1);
      expect=8.0, tol=1.2, recorded=7.91)

study("C10 with wall closures", (24, 48, 96),
      N -> Solver(n_global=(N, 12, 12), L_domain=(1.0, 1.0, 1.0),
                  bcs=((SlipWallBC(), SlipWallBC()), per3[2], per3[3]),
                  deriv=lele_d1_10(), art=ArtificialProperties(enabled=false)),
      (x, y, z) -> exp(sin(3x)),
      (fn=(x, y, z) -> 3cos(3x) * exp(sin(3x)), parity=1);
      expect=3.2, tol=0.8, recorded=3.18)

# One pass of the state filter on a closed line, measured as |F f − f|. The
# wall cascade (identity, F2, F4, F6) is second order along the whole
# line, not only at the wall, because the compact solve carries the row-2
# error inward; the one-sided Gaitonde–Visbal rows, which are the default,
# restore the interior order. Resolutions are lower because the
# eighth-order pass reaches round-off by N = 48 on this field.
study("C8 filter pass, :cascade", (12, 16, 24, 32),
      N -> Solver(n_global=(N, 12, 12), L_domain=(1.0, 1.0, 1.0),
                  bcs=((SlipWallBC(), SlipWallBC()), per3[2], per3[3]),
                  filt=compact_filter(0.45; closures=:cascade),
                  art=ArtificialProperties(enabled=false)),
      (x, y, z) -> exp(sin(3x)),
      (fn=(x, y, z) -> exp(sin(3x)), parity=1);
      expect=2.0, tol=0.6, recorded=1.88, op=:filter)

study("C8 filter pass, :onesided", (12, 16, 24, 32),
      N -> Solver(n_global=(N, 12, 12), L_domain=(1.0, 1.0, 1.0),
                  bcs=((SlipWallBC(), SlipWallBC()), per3[2], per3[3]),
                  filt=compact_filter(0.45; closures=:onesided),
                  art=ArtificialProperties(enabled=false)),
      (x, y, z) -> exp(sin(3x)),
      (fn=(x, y, z) -> exp(sin(3x)), parity=1);
      expect=8.0, tol=1.5, recorded=8.07, op=:filter)

# A symmetry plane sits half a cell outside the end node, so the folded
# operator runs the interior stencil to the edge over the mirror halo and no
# closure row exists to lose order at: these rows read the interior order,
# unlike the wall rows above. The fields carry the plane's parity about
# x = 0 and x = 1, which is what `ref.parity` tells the fold. The
# resolutions fall with the order because the error reaches round-off
# quickly on a field this smooth.
const SYM = (SymmetryPlaneBC(), SymmetryPlaneBC())
sym_solver(N, deriv) =
    Solver(n_global=(N, 12, 12), L_domain=(1.0, 1.0, 1.0),
           bcs=(SYM, per3[2], per3[3]), deriv=deriv,
           filt=compact_filter(0.45), art=ArtificialProperties(enabled=false))
sym_even(x, y, z) = exp(cospi(x))
sym_deven(x, y, z) = -pi * sinpi(x) * exp(cospi(x))
sym_odd(x, y, z) = sinpi(x) * exp(cospi(x))
sym_dodd(x, y, z) = pi * exp(cospi(x)) * (cospi(x) - sinpi(x)^2)

println("\n=== symmetry planes (folded operators, no closure row) ===")
study("C6 symmetry planes, even field", (24, 48, 96),
      N -> sym_solver(N, lele_d1_6()), sym_even, (fn=sym_deven, parity=1);
      expect=6.0, tol=1.0, recorded=6.00)
study("C6 symmetry planes, odd field", (24, 48, 96),
      N -> sym_solver(N, lele_d1_6()), sym_odd, (fn=sym_dodd, parity=-1);
      expect=6.0, tol=1.0, recorded=6.00)
study("C8 symmetry planes, even field", (16, 24, 32),
      N -> sym_solver(N, lele_d1_8()), sym_even, (fn=sym_deven, parity=1);
      expect=8.0, tol=1.2, recorded=8.05)
study("C8 symmetry planes, odd field", (16, 24, 32),
      N -> sym_solver(N, lele_d1_8()), sym_odd, (fn=sym_dodd, parity=-1);
      expect=8.0, tol=1.2, recorded=7.95)
study("C10 symmetry planes, even field", (12, 16, 24),
      N -> sym_solver(N, lele_d1_10()), sym_even, (fn=sym_deven, parity=1);
      expect=10.0, tol=2.0, recorded=10.23)
study("C10 symmetry planes, odd field", (12, 16, 24),
      N -> sym_solver(N, lele_d1_10()), sym_odd, (fn=sym_dodd, parity=-1);
      expect=10.0, tol=2.0, recorded=10.17)

# One filter pass between the planes, as the closed-line pass above. The
# eighth-order interior rows run to the edge here, so the pass keeps its own
# order instead of the cap a wall closure puts on it.
study("C8 filter pass, symmetry planes", (16, 24, 32, 48),
      N -> sym_solver(N, lele_d1_6()), sym_even, (fn=sym_even, parity=1);
      expect=8.0, tol=1.5, recorded=7.88, op=:filter)

println("\n=== coordinate-singularity folds ===")
study("cylindrical axis, odd field (u_r-like)", (32, 64, 128),
      N -> Solver(n_global=(N, 1, 12), L_domain=(1.0, 1.0, 0.5),
                  metric=CylindricalMetric(),
                  bcs=((AxisBC(), SlipWallBC()), per3[2], per3[3]),
                  art=ArtificialProperties(enabled=false)),
      (r, θ, z) -> r * exp(-4r^2),
      (fn=(r, θ, z) -> (1 - 8r^2) * exp(-4r^2), parity=-1);
      expect=3.7, tol=0.8, recorded=3.76)

study("cylindrical axis, even field (scalar)", (32, 64, 128),
      N -> Solver(n_global=(N, 1, 12), L_domain=(1.0, 1.0, 0.5),
                  metric=CylindricalMetric(),
                  bcs=((AxisBC(), SlipWallBC()), per3[2], per3[3]),
                  art=ArtificialProperties(enabled=false)),
      (r, θ, z) -> exp(-4r^2),
      (fn=(r, θ, z) -> -8r * exp(-4r^2), parity=1);
      expect=3.0, tol=0.8, recorded=2.99)

study("resolved-θ axis, x-like field", (32, 64, 128),
      N -> Solver(n_global=(N, 16, 1), L_domain=(1.0, 2π, 1.0),
                  metric=CylindricalMetric(),
                  bcs=((AxisBC(), SlipWallBC()), per3[2], per3[3]),
                  art=ArtificialProperties(enabled=false)),
      (r, θ, z) -> r * cos(θ) * exp(-4r^2),
      (fn=(r, θ, z) -> cos(θ) * (1 - 8r^2) * exp(-4r^2), parity=1);
      expect=3.7, tol=0.8, recorded=3.76)

study("spherical origin, radial Gaussian", (24, 48, 96),
      N -> Solver(n_global=(N, 12, 12), L_domain=(1.0, π, 2π),
                  metric=SphericalMetric(),
                  bcs=((OriginBC(), SlipWallBC()),
                       (PoleBC(), PoleBC()), per3[3]),
                  art=ArtificialProperties(enabled=false)),
      (r, θ, φ) -> exp(-4r^2),
      (fn=(r, θ, φ) -> -8r * exp(-4r^2), parity=1);
      expect=3.0, tol=0.8, recorded=2.97)

# ---------------------------------------------------------------------------
# Closure truncation and smooth evolution, the accuracy matrix's gated rows.
#
# The polynomial rows measure the closure rows' own pointwise order: one
# derivative of x^(q+1), q the closure's exactness degree, on 17 to 129
# nodes against the actual spacing 1/(N − 1). The evolution rows integrate
# a smooth case to a fixed time and read the error of the region the row
# names, in the maximum norm, against a reference that carries no closure
# error of its own: the exact solution, or the periodic mirror of a wall
# problem at the same spacing and step (`test/smooth_cases.jl` explains
# both). The fitted order is over the three spacings; the l2 order printed
# beside it is the composite volume-weighted norm's, covered parents
# excluded, half an order above the windowed maximum for a defect
# confined to a window of fixed node count. Every row runs at cfl = 0.25,
# where bench/boundaryorder.jl measured the halved step to move the
# windowed error by under 1% on the rows gated here (the C6 BL wall rows
# are within 3%); the C8 BL rows and every unfiltered interior are time-
# or roundoff-limited at the finest grid and are measured, not gated.

function truncation_study(name, Ns, deriv, degree; expect, tol, recorded)
    t0 = time(); c0 = compile_ns()
    hs = Float64[]; errs = Float64[]; errs_in = Float64[]
    for N in Ns
        e = closed_derivative_errors(N, deriv, x -> x^degree,
                                     x -> degree * x^(degree - 1))
        push!(hs, 1 / (N - 1)); push!(errs, e.wall); push!(errs_in, e.interior)
    end
    p = observed_order(hs, errs)
    @printf("%-38s  ", name)
    for (N, e) in zip(Ns, errs)
        @printf("N=%-4d %.3e  ", N, e)
    end
    @printf("order ≈ %.2f  (interior %.2f)\n", p, observed_order(hs, errs_in))
    push!(PHASE_LOG, (name, time() - t0, (compile_ns() - c0) / 1e9))
    _guard(name, p, expect, tol, recorded)
    p
end

function _guard(name, p, expect, tol, recorded)
    abs(p - expect) < tol || println(
        "  ORDER REGRESSED: $(round(p, digits=2)) is outside $expect ± $tol. " *
        "That is a wrong coefficient, closure row or interface rule, not a drift.")
    @test abs(p - expect) < tol
    if recorded !== nothing
        abs(p - recorded) < DRIFT_TOL || println(
            "  ORDER DRIFTED: $(round(p, digits=2)) against the recorded $recorded. " *
            "The scheme is intact and something reached the numerics. Find the " *
            "cause before updating the recorded value here and in the header.")
        @test abs(p - recorded) < DRIFT_TOL
    end
end

function evolution_study(name, Ns, build, reference; primary, comp=1, tfinal,
                         expect, tol, recorded, steps=nothing, patches=nothing)
    t0 = time(); c0 = compile_ns()
    hs = Float64[]; errs = Float64[]; errs2 = Float64[]
    for N in Ns
        solver, states = build(N)
        # `steps(N)` equal steps where given, so that a reference taking the
        # same steps shares the time error; the CFL-limited step otherwise.
        steps === nothing ? run!(solver, states; tfinal=tfinal) :
                            fixed_step_run!(solver, states, tfinal, steps(N))
        e = regional_errors(solver, states, reference(solver); comp=comp,
                            patches=patches)
        push!(hs, root_spacing(solver))
        push!(errs, getfield(e, primary)); push!(errs2, e.l2)
    end
    p = observed_order(hs, errs)
    @printf("%-38s  ", name)
    for (N, e) in zip(Ns, errs)
        @printf("N=%-4d %.3e  ", N, e)
    end
    @printf("order ≈ %.2f  (l2 %.2f)\n", p, observed_order(hs, errs2))
    push!(PHASE_LOG, (name, time() - t0, (compile_ns() - c0) / 1e9))
    _guard(name, p, expect, tol, recorded)
    p
end

const EVOLUTION_CFL = 0.25
const WALL_NS = (49, 97, 193)          # closed lines, h = 1/(N − 1)
const PLANE_NS = (25, 49, 97)          # folded lines, h = 1/N; see the plane rows
const PERIODIC_NS = (48, 96, 192)      # periodic roots, multiples of 24 for the nest

# The periodic mirror of the wall run that has just finished, at the same
# spacing and step: the wall problem without its closure rows.
function mirror_reference(solver; viscous=false, opts...)
    mirror, states = mirror_case(solver.n_global[1]; viscous=viscous,
                                 cfl=EVOLUTION_CFL, opts...)
    run!(mirror, states; tfinal=0.4)
    NodeReference(mirror, states)
end

# A run between symmetry planes is the periodic run on the doubled line
# restricted by parity, so its mirror at the same spacing reproduces it to
# round-off (test/runtests.jl measures that) and would measure nothing here.
# The reference is a five-times-finer folded mirror instead: an odd refinement
# so that every node of the study grid is a node of the reference, and fine
# enough that the reference carries 1/625 of the study's step error. What the
# wall window reads against it is then the run's own total error, there being
# no closure defect to separate out.
const FOLD_REFINE = 5
function folded_reference(solver; viscous=false, opts...)
    mirror, states = mirror_case(FOLD_REFINE * solver.n_global[1]; folded=true,
                                 viscous=viscous, cfl=EVOLUTION_CFL, opts...)
    run!(mirror, states; tfinal=0.4)
    NodeReference(mirror, states)
end

println("\n=== closure truncation on a polynomial (actual spacing) ===")
truncation_study("C6 :neutral3 rows, d/dx x^4", (17, 33, 65, 129), lele_d1_6(), 4;
                 expect=3.0, tol=0.5, recorded=3.00)
truncation_study("C6 :cascade3 rows, d/dx x^4", (17, 33, 65, 129),
                 lele_d1_6(closures=:cascade3), 4; expect=3.0, tol=0.5, recorded=3.00)
truncation_study("C6 :cascade4 rows, d/dx x^5", (17, 33, 65, 129),
                 lele_d1_6(closures=:cascade4), 5; expect=4.0, tol=0.5, recorded=4.00)
truncation_study("C6 :brady_livescu rows, d/dx x^6", (17, 33, 65, 129),
                 lele_d1_6(closures=:brady_livescu), 6; expect=5.0, tol=0.5, recorded=5.00)
truncation_study("C8 :neutral3 rows, d/dx x^4", (17, 33, 65, 129), lele_d1_8(), 4;
                 expect=3.0, tol=0.5, recorded=3.00)
truncation_study("C8 :brady_livescu rows, d/dx x^8", (17, 33, 65, 129),
                 lele_d1_8(closures=:brady_livescu), 8; expect=7.0, tol=0.5, recorded=7.00)
truncation_study("C10 :neutral3 rows, d/dx x^4", (17, 33, 65, 129), lele_d1_10(), 4;
                 expect=3.0, tol=0.5, recorded=3.00)

# ---------------------------------------------------------------------------
# The staggered operators of the implicit diffusion operator
# (src/staggered.jl): D_s from nodes to midpoints, G back to nodes, the
# midpoint interpolation of the coefficient, and their composition
# L = G K D_s against (κ T')'. Each is measured on a line of N nodes along x,
# periodic, and closed with the wall mirror on fields of its parity (T and κ
# even through both walls, the flux odd), where the interior rows reach the
# wall and the order is the interior's.

function staggered_errors(N, periodic, which)
    decomp = CL.Decomp((N, 1, 1), (periodic, false, false))
    h = periodic ? 1 / N : 1 / (N - 1)
    op = CL.StaggeredDiffusion(decomp, 1, h; parity=1)
    k1, k2 = periodic ? (2π, 6π) : (2π, 3π)
    c1, c2 = periodic ? (sin, cos) : (cos, cos)
    T(x) = c1(k1 * x) + 0.2c2(k2 * x)
    dT(x) = periodic ? k1 * cos(k1 * x) - 0.2k2 * sin(k2 * x) :
                       -k1 * sin(k1 * x) - 0.2k2 * sin(k2 * x)
    d2T(x) = -k1^2 * c1(k1 * x) - 0.2k2^2 * c2(k2 * x)
    κ(x) = periodic ? 1 + 0.5sin(2π * x) : 1 + 0.5cos(π * x)
    dκ(x) = periodic ? π * cos(2π * x) : -0.5π * sin(π * x)
    pad = decomp.n_halo_d[1]
    load!(f, fn, shift) = (for i in 1:N; f[i+pad, 1, 1] = fn((i - 1) * h + shift); end; f)
    Tf = load!(CL.field(decomp), T, 0.0)
    κf = load!(CL.field(decomp), κ, 0.0)
    out = CL.field(decomp)
    nmid = periodic ? N : N - 1
    if which === :to_mid
        CL.exchange_dim!(Tf, decomp, 1)
        CL.apply_along!(out, op.to_mid, Tf, decomp)
        fn, npts, shift = dT, nmid, h / 2
    elseif which === :interpolate
        CL.exchange_dim!(κf, decomp, 1)
        CL.apply_along!(out, op.interpolate, κf, decomp)
        fn, npts, shift = κ, nmid, h / 2
    elseif which === :to_node
        g = load!(CL.field(decomp), x -> κ(x) * dT(x), h / 2)
        CL.exchange_dim!(g, decomp, 1)
        CL.apply_along!(out, op.to_node, g, decomp)
        fn, npts, shift = x -> dκ(x) * dT(x) + κ(x) * d2T(x), N, 0.0
    else
        CL.staggered_diffusion!(out, op, Tf, κf, decomp)
        fn, npts, shift = x -> dκ(x) * dT(x) + κ(x) * d2T(x), N, 0.0
    end
    e = [out[i+pad, 1, 1] - fn((i - 1) * h + shift) for i in 1:npts]
    return h, maximum(abs, e), sqrt(sum(abs2, e) / npts)
end

function staggered_study(name, Ns, periodic, which; expect, tol, recorded)
    t0 = time(); c0 = compile_ns()
    hs = Float64[]; errs = Float64[]; errs2 = Float64[]
    for N in Ns
        h, e, e2 = staggered_errors(N, periodic, which)
        push!(hs, h); push!(errs, e); push!(errs2, e2)
    end
    p = observed_order(hs, errs)
    @printf("%-38s  ", name)
    for (N, e) in zip(Ns, errs)
        @printf("N=%-4d %.3e  ", N, e)
    end
    @printf("order ≈ %.2f  (L2 %.2f)\n", p, observed_order(hs, errs2))
    push!(PHASE_LOG, (name, time() - t0, (compile_ns() - c0) / 1e9))
    _guard(name, p, expect, tol, recorded)
    p
end

println("\n=== staggered operators (implicit diffusion) ===")
staggered_study("staggered D_s, periodic", (16, 32, 64), true, :to_mid;
                expect=6.0, tol=0.5, recorded=6.04)
staggered_study("staggered G, periodic", (16, 32, 64), true, :to_node;
                expect=6.0, tol=0.5, recorded=6.07)
staggered_study("staggered interpolation, periodic", (16, 32, 64), true, :interpolate;
                expect=6.0, tol=0.5, recorded=5.97)
staggered_study("staggered L = G K D_s, periodic", (16, 32, 64), true, :diffusion;
                expect=6.0, tol=0.5, recorded=6.06)
staggered_study("staggered D_s, wall mirror", (17, 33, 65), false, :to_mid;
                expect=6.0, tol=0.5, recorded=6.00)
staggered_study("staggered G, wall mirror", (17, 33, 65), false, :to_node;
                expect=6.0, tol=0.5, recorded=6.02)
staggered_study("staggered interpolation, wall mirror", (17, 33, 65), false, :interpolate;
                expect=6.0, tol=0.5, recorded=5.99)
staggered_study("staggered L = G K D_s, wall mirror", (17, 33, 65), false, :diffusion;
                expect=6.0, tol=0.5, recorded=6.02)

# The same operators between two symmetry planes half a cell beyond the end
# nodes, where a midpoint lies on each plane: T and κ even through both, and
# once an odd T, whose even flux is an unknown on the planes (slot N, and at
# the low plane slot 0, eliminated from the row of midpoint 1). The folded
# line is half of a periodic line, so the order is the interior's.
function staggered_fold_errors(N, which; odd=false)
    decomp = CL.Decomp((N, 1, 1), (false, false, false))
    h = 1 / N
    σ = odd ? -1 : 1
    op = CL.StaggeredDiffusion(decomp, 1, h; parity=1, lo_fold=σ, hi_fold=σ)
    T(x) = odd ? sinpi(x) * (1 + 0.3cospi(2x)) : cospi(2x) + 0.2cospi(3x)
    dT(x) = odd ? pi * cospi(x) * (1 + 0.3cospi(2x)) - 0.6pi * sinpi(x) * sinpi(2x) :
                  -2pi * sinpi(2x) - 0.6pi * sinpi(3x)
    d2T(x) = odd ? -pi^2 * sinpi(x) * (1 + 0.3cospi(2x)) -
                   1.2pi^2 * cospi(x) * sinpi(2x) - 1.2pi^2 * sinpi(x) * cospi(2x) :
                   -4pi^2 * cospi(2x) - 1.8pi^2 * cospi(3x)
    κ(x) = 1 + 0.5cospi(x)
    dκ(x) = -0.5pi * sinpi(x)
    pad = decomp.n_halo_d[1]
    load!(f, fn, shift, r) = (for i in r; f[i+pad, 1, 1] = fn((i - 0.5) * h + shift); end; f)
    Tf = load!(CL.field(decomp), T, 0.0, 1:N)
    κf = load!(CL.field(decomp), κ, 0.0, 1:N)
    out = CL.field(decomp)
    if which === :to_mid
        CL.apply_along!(out, op.to_mid, Tf, decomp)
        fn, r, shift = dT, 0:N, h / 2
    elseif which === :interpolate
        CL.apply_along!(out, op.interpolate, κf, decomp)
        fn, r, shift = κ, 0:N, h / 2
    elseif which === :to_node
        g = load!(CL.field(decomp), x -> κ(x) * dT(x), h / 2, 0:N)
        CL.apply_along!(out, op.to_node, g, decomp)
        fn, r, shift = x -> dκ(x) * dT(x) + κ(x) * d2T(x), 1:N, 0.0
    else
        CL.staggered_diffusion!(out, op, Tf, κf, decomp)
        fn, r, shift = x -> dκ(x) * dT(x) + κ(x) * d2T(x), 1:N, 0.0
    end
    e = [out[i+pad, 1, 1] - fn((i - 0.5) * h + shift) for i in r]
    return h, maximum(abs, e), sqrt(sum(abs2, e) / length(r))
end

# L along one dimension of a solver's metric, J⁻¹ ∂(C κ ∂T) with C = J/h_d²,
# against the analytic operator, over the nodes `window` admits. The
# coordinate-singularity rows keep to r < 1/2: the outer wall is curved, so
# its node is first order (the last row measures it), and the compact solve
# carries that error inward only over a few nodes.
function staggered_metric_errors(N, case)
    solver = case.build(N)
    op = CL.StaggeredDiffusion(solver, case.dim)
    decomp = solver.decomp
    Tf = CL.field(decomp); κf = CL.field(decomp); out = CL.field(decomp)
    fillf!(solver, Tf, case.T)
    fillf!(solver, κf, case.κ)
    CL.staggered_diffusion!(out, op, Tf, κf, decomp)
    e = Float64[]
    for k in 1:decomp.n_local[3], j in 1:decomp.n_local[2], i in 1:decomp.n_local[1]
        x = (xcoord(solver, 1, i), xcoord(solver, 2, j), xcoord(solver, 3, k))
        case.window(x...) || continue
        push!(e, out[padded_index(solver, i, j, k)] - case.L(x...))
    end
    return solver.h[case.dim], maximum(abs, e), sqrt(sum(abs2, e) / length(e))
end

function staggered_study(name, Ns, errors::Function; expect, tol, recorded)
    t0 = time(); c0 = compile_ns()
    hs = Float64[]; errs = Float64[]; errs2 = Float64[]
    for N in Ns
        h, e, e2 = errors(N)
        push!(hs, h); push!(errs, e); push!(errs2, e2)
    end
    p = observed_order(hs, errs)
    @printf("%-38s  ", name)
    for (N, e) in zip(Ns, errs)
        @printf("N=%-4d %.3e  ", N, e)
    end
    @printf("order ≈ %.2f  (L2 %.2f)\n", p, observed_order(hs, errs2))
    push!(PHASE_LOG, (name, time() - t0, (compile_ns() - c0) / 1e9))
    _guard(name, p, expect, tol, recorded)
    p
end

for (label, which, rec) in (("D_s", :to_mid, 6.01), ("G", :to_node, 5.98),
                            ("interpolation", :interpolate, 6.00),
                            ("L = G K D_s", :diffusion, 5.98))
    staggered_study("staggered $label, symmetry planes", (16, 32, 64),
                    N -> staggered_fold_errors(N, which);
                    expect=6.0, tol=0.5, recorded=rec)
end
staggered_study("staggered L, symmetry planes, odd T", (16, 32, 64),
                N -> staggered_fold_errors(N, :diffusion; odd=true);
                expect=6.0, tol=0.5, recorded=6.01)

# The metric rows. Radial fields are smooth through the singular point: the
# Gaussian and its product with the Cartesian coordinate along the pairing,
# which gives both the even and the odd combination of a paired fold. On the
# sphere the polar field is z plus a multiple of x. The stretched line
# clusters symmetrically (ξc = 1/2), so its map and the fields are even about
# both walls and the mirror is exact there. At the first node of a fold J⁻¹
# grows as 1/r (axis, poles) or 1/r² (origin) and multiplies the sixth-order
# error of G, so those rows read five and four; the axisymmetric Gaussian
# keeps six, since its flux r κ T' is even and the odd derivative that sets
# the error vanishes at the axis.
cluster = sine_cluster(0.0, 1.0, 0.5, 0.4)
gauss(r) = exp(-4r^2)
dgauss(r) = -8r * exp(-4r^2)
d2gauss(r) = (64r^2 - 8) * exp(-4r^2)
# 1/r^m ∂_r(r^m κ ∂_r T) for T = g(r) (1 + a r c) and κ = 1 + 0.5 r² + b r s,
# with c and s angular factors held fixed along the line.
function radial_L(r, m, a, c, b, s)
    T1 = dgauss(r) * (1 + a * r * c) + gauss(r) * a * c
    T2 = d2gauss(r) * (1 + a * r * c) + 2dgauss(r) * a * c
    κ = 1 + 0.5r^2 + b * r * s
    dκ = r + b * s
    return dκ * T1 + κ * (T2 + m * T1 / r)
end
const STAGGERED_METRIC_CASES = (
    (name="staggered L, stretched line", Ns=(17, 33, 65), expect=6.0, tol=0.5,
     recorded=6.06, dim=1,
     build=N -> Solver(n_global=(N, 1, 1), L_domain=(1.0, 1.0, 1.0),
                       bcs=((SlipWallBC(), SlipWallBC()), per3[2], per3[3]),
                       stretch=(cluster, nothing, nothing),
                       art=ArtificialProperties(enabled=false)),
     T=(x, y, z) -> cospi(2x) + 0.2cospi(3x), κ=(x, y, z) -> 1 + 0.5cospi(x),
     L=(x, y, z) -> -0.5pi * sinpi(x) * (-2pi * sinpi(2x) - 0.6pi * sinpi(3x)) +
                    (1 + 0.5cospi(x)) * (-4pi^2 * cospi(2x) - 1.8pi^2 * cospi(3x)),
     window=(x, y, z) -> true),
    (name="staggered L, cylindrical axis", Ns=(32, 64, 128), expect=6.0, tol=0.5,
     recorded=5.96, dim=1,
     build=N -> Solver(n_global=(N, 1, 12), L_domain=(1.0, 1.0, 0.5),
                       metric=CylindricalMetric(),
                       bcs=((AxisBC(), SlipWallBC()), per3[2], per3[3]),
                       art=ArtificialProperties(enabled=false)),
     T=(r, θ, z) -> gauss(r), κ=(r, θ, z) -> 1 + 0.5r^2,
     L=(r, θ, z) -> radial_L(r, 1, 0.0, 0.0, 0.0, 0.0),
     window=(r, θ, z) -> r < 0.5),
    (name="staggered L, resolved-θ axis", Ns=(32, 64, 128), expect=5.0, tol=0.5,
     recorded=5.07, dim=1,
     build=N -> Solver(n_global=(N, 16, 1), L_domain=(1.0, 2π, 1.0),
                       metric=CylindricalMetric(),
                       bcs=((AxisBC(), SlipWallBC()), per3[2], per3[3]),
                       art=ArtificialProperties(enabled=false)),
     T=(r, θ, z) -> gauss(r) * (1 + r * cos(θ)),
     κ=(r, θ, z) -> 1 + 0.5r^2 + 0.2r * sin(θ),
     L=(r, θ, z) -> radial_L(r, 1, 1.0, cos(θ), 0.2, sin(θ)),
     window=(r, θ, z) -> r < 0.5),
    (name="staggered L, spherical origin", Ns=(24, 48, 96), expect=4.0, tol=0.5,
     recorded=4.03, dim=1,
     build=N -> Solver(n_global=(N, 12, 12), L_domain=(1.0, π, 2π),
                       metric=SphericalMetric(),
                       bcs=((OriginBC(), SlipWallBC()), (PoleBC(), PoleBC()), per3[3]),
                       art=ArtificialProperties(enabled=false)),
     T=(r, θ, φ) -> gauss(r) * (1 + 0.5r * cos(θ)),
     κ=(r, θ, φ) -> 1 + 0.5r^2 + 0.2r * sin(θ) * cos(φ),
     L=(r, θ, φ) -> radial_L(r, 2, 0.5, cos(θ), 0.2, sin(θ) * cos(φ)),
     window=(r, θ, φ) -> r < 0.5),
    (name="staggered L, spherical poles", Ns=(16, 32, 64), expect=5.0, tol=0.5,
     recorded=5.33, dim=2,
     build=N -> Solver(n_global=(12, N, 12), L_domain=(1.0, π, 2π),
                       metric=SphericalMetric(), origin=(0.5, 0.0, 0.0),
                       bcs=((SlipWallBC(), SlipWallBC()), (PoleBC(), PoleBC()),
                            per3[3]),
                       art=ArtificialProperties(enabled=false)),
     T=(r, θ, φ) -> cos(θ) + 0.3sin(θ) * cos(φ),
     κ=(r, θ, φ) -> 1 + 0.3cos(θ)^2,
     # (1/(r² sinθ)) ∂_θ(sinθ κ ∂_θ T)
     L=(r, θ, φ) -> begin
         κ = 1 + 0.3cos(θ)^2; dκ = -0.6cos(θ) * sin(θ)
         T1 = -sin(θ) + 0.3cos(θ) * cos(φ); T2 = -cos(θ) - 0.3sin(θ) * cos(φ)
         (cos(θ) * κ * T1 + sin(θ) * (dκ * T1 + κ * T2)) / (r^2 * sin(θ))
     end,
     window=(r, θ, φ) -> true),
    (name="staggered L, curved wall", Ns=(17, 33, 65), expect=1.0, tol=0.5,
     recorded=1.00, dim=1,
     build=N -> Solver(n_global=(N, 1, 12), L_domain=(1.0, 1.0, 0.5),
                       metric=CylindricalMetric(), origin=(0.5, 0.0, 0.0),
                       bcs=((SlipWallBC(), SlipWallBC()), per3[2], per3[3]),
                       art=ArtificialProperties(enabled=false)),
     T=(r, θ, z) -> cospi(r - 0.5), κ=(r, θ, z) -> 1.0,
     L=(r, θ, z) -> -pi^2 * cospi(r - 0.5) - pi * sinpi(r - 0.5) / r,
     window=(r, θ, z) -> true),
)
for case in STAGGERED_METRIC_CASES
    staggered_study(case.name, case.Ns, N -> staggered_metric_errors(N, case);
                    expect=case.expect, tol=case.tol, recorded=case.recorded)
end

println("\n=== smooth evolution: wall window, t = 0.4 ===")
evolution_study("inviscid wall, C6, unfiltered", WALL_NS,
                N -> wall_case(N; cfl=EVOLUTION_CFL), mirror_reference;
                primary=:wall, tfinal=0.4, expect=3.9, tol=0.8, recorded=4.01)
evolution_study("inviscid wall, C6 :cascade3, unfiltered", WALL_NS,
                N -> wall_case(N; cfl=EVOLUTION_CFL, deriv=lele_d1_6(closures=:cascade3)),
                s -> mirror_reference(s; deriv=lele_d1_6(closures=:cascade3));
                primary=:wall, tfinal=0.4, expect=3.9, tol=0.8, recorded=3.93)
evolution_study("inviscid wall, C6, cascade filter", WALL_NS,
                N -> wall_case(N; cfl=EVOLUTION_CFL, filter_interval=1,
                               filt=compact_filter(0.45; closures=:cascade)),
                s -> mirror_reference(s; filter_interval=1,
                                      filt=compact_filter(0.45; closures=:cascade));
                primary=:wall, tfinal=0.4, expect=1.8, tol=0.6, recorded=1.94)
evolution_study("inviscid wall, C6, onesided filter", WALL_NS,
                N -> wall_case(N; cfl=EVOLUTION_CFL, filter_interval=1,
                               filt=compact_filter(0.45; closures=:onesided)),
                s -> mirror_reference(s; filter_interval=1,
                                      filt=compact_filter(0.45; closures=:onesided));
                primary=:wall, tfinal=0.4, expect=3.8, tol=0.8, recorded=3.90)
evolution_study("inviscid wall, C6 :brady_livescu, unfiltered", WALL_NS,
                N -> wall_case(N; cfl=EVOLUTION_CFL, deriv=lele_d1_6(closures=:brady_livescu)),
                s -> mirror_reference(s; deriv=lele_d1_6(closures=:brady_livescu));
                primary=:wall, tfinal=0.4, expect=5.7, tol=0.8, recorded=5.73)
evolution_study("viscous no-slip wall, C6, unfiltered", WALL_NS,
                N -> wall_case(N; cfl=EVOLUTION_CFL, viscous=true),
                s -> mirror_reference(s; viscous=true);
                primary=:wall, tfinal=0.4, expect=3.9, tol=0.8, recorded=4.00)
# The tangential amplitude puts a shear traction on the symmetry plane, which
# the slip wall's flux contract removes along with the conductive heat flux.
evolution_study("viscous slip wall, C6, unfiltered", WALL_NS,
                N -> wall_case(N; cfl=EVOLUTION_CFL, viscous=true, slip=true, c=0.05),
                s -> mirror_reference(s; viscous=true, c=0.05);
                primary=:wall, tfinal=0.4, expect=3.9, tol=0.8, recorded=4.00)
evolution_study("shear mode, no-slip wall, C6, unfiltered", WALL_NS,
                N -> shear_case(N; cfl=EVOLUTION_CFL),
                s -> analytic_reference(s.equations, shear_profile(0.1, 0.005; t=s.t));
                primary=:wall, comp=3, tfinal=0.4, expect=4.7, tol=0.8,
                recorded=4.67)

# The same wave between symmetry planes, against the fine folded mirror. The
# wall window has no closure row and reads what the interior reads: on the
# inviscid rows at cfl = 0.25 that is the time integrator's own order, which
# the C8 and C10 rows read as 4.00; the C6 rows sit above it because the
# coarsest grid still shows the sixth-order spatial part on top. Quartering
# the step isolates that part at 5.93, so nothing at the plane caps it, and
# the viscous row's smaller step shows the same thing directly.
#
# The plane rows run one ladder coarser than the wall rows. On WALL_NS the
# finest grid's error is 2.5e-13, where the step's accumulated round-off is a
# few percent of the difference and moves with the host's arithmetic: the
# same code measured the C10 row at 3.96 on one machine and 4.00 on another,
# outside DRIFT_TOL, with the two coarser grids agreeing to four digits.
# PLANE_NS keeps every difference above 1e-12, where that spread is under
# 0.2% and the fitted order moves by 0.001.
println("\n=== smooth evolution: symmetry plane, wall window, t = 0.4 ===")
evolution_study("inviscid planes, C6, unfiltered", PLANE_NS,
                N -> wall_case(N; folded=true, cfl=EVOLUTION_CFL),
                folded_reference;
                primary=:wall, tfinal=0.4, expect=4.4, tol=0.8, recorded=4.46)
evolution_study("inviscid planes, C6, onesided filter", PLANE_NS,
                N -> wall_case(N; folded=true, cfl=EVOLUTION_CFL,
                               filter_interval=1),
                s -> folded_reference(s; filter_interval=1);
                primary=:wall, tfinal=0.4, expect=4.6, tol=0.8, recorded=4.69)
evolution_study("inviscid planes, C8, unfiltered", PLANE_NS,
                N -> wall_case(N; folded=true, cfl=EVOLUTION_CFL,
                               deriv=lele_d1_8()),
                s -> folded_reference(s; deriv=lele_d1_8());
                primary=:wall, tfinal=0.4, expect=4.0, tol=0.8, recorded=4.00)
evolution_study("inviscid planes, C10, unfiltered", PLANE_NS,
                N -> wall_case(N; folded=true, cfl=EVOLUTION_CFL,
                               deriv=lele_d1_10()),
                s -> folded_reference(s; deriv=lele_d1_10());
                primary=:wall, tfinal=0.4, expect=4.0, tol=0.8, recorded=4.00)
# The viscous row's step is diffusion-limited, so the step error is far below
# the spatial one and the row reads the interior order instead of the
# integrator's. On WALL_NS that would put the finest grid's difference at
# round-off, and the reference, whose step falls with h², costs the cube of
# its refinement there.
evolution_study("viscous slip planes, C6, unfiltered", PLANE_NS,
                N -> wall_case(N; folded=true, cfl=EVOLUTION_CFL, viscous=true,
                               slip=true, c=0.05),
                s -> folded_reference(s; viscous=true, c=0.05);
                primary=:wall, tfinal=0.4, expect=6.0, tol=1.0, recorded=6.04)

println("\n=== smooth evolution: interface window, entropy wave, t = 0.5 ===")
entropy_ref(s) = analytic_reference(s.equations, entropy_profile(3, 0.37; t=s.t))
# The same-level interface is measured on the k = 1 wave: at the root
# spacing the k = 3 wave is pre-asymptotic below N = 192 under the closure
# rows (bench/boundaryorder.jl reads 5.1 / 4.7 there against 3.1 / 3.5 at
# k = 1), while a level interface sits at the fine spacing and reads its
# asymptotic order on either wave.
evolution_study("two patches, C6, k = 1", PERIODIC_NS,
                N -> entropy_case(N; cfl=EVOLUTION_CFL, patch_grid=(2, 1, 1), k=1, phase=0.0),
                s -> analytic_reference(s.equations, entropy_profile(1, 0.0; t=s.t));
                primary=:interface, tfinal=0.5, expect=6.8, tol=1.0, recorded=6.79)
evolution_study("two levels, C6", PERIODIC_NS,
                N -> entropy_case(N; cfl=EVOLUTION_CFL, levels=2), entropy_ref;
                primary=:interface, tfinal=0.5, expect=6.0, tol=0.8, recorded=6.01)
# The closure rows, which the rows below them compare with their alternatives.
evolution_study("two levels, C6, closure rows", PERIODIC_NS,
                N -> entropy_case(N; cfl=EVOLUTION_CFL, levels=2, interface_flux=:closure),
                entropy_ref;
                primary=:interface, tfinal=0.5, expect=3.6, tol=0.8, recorded=3.62)
evolution_study("two levels, C6 :brady_livescu", PERIODIC_NS,
                N -> entropy_case(N; cfl=EVOLUTION_CFL, levels=2, interface_flux=:closure,
                                  deriv=lele_d1_6(closures=:brady_livescu)), entropy_ref;
                primary=:interface, tfinal=0.5, expect=6.0, tol=0.8, recorded=6.01)
evolution_study("three levels, C6, subcycled", PERIODIC_NS,
                N -> entropy_case(N; cfl=EVOLUTION_CFL, levels=3, subcycle=true), entropy_ref;
                primary=:interface, tfinal=0.5, expect=6.0, tol=0.8, recorded=6.00)
evolution_study("two levels, C6, cascade filter", PERIODIC_NS,
                N -> entropy_case(N; cfl=EVOLUTION_CFL, levels=2, filter_interval=1),
                entropy_ref;
                primary=:interface, tfinal=0.5, expect=6.9, tol=1.0, recorded=6.87)
# The artificial properties on under the :d8 detector, whose interface rows
# read the imposed ghosts. The :brady_livescu row is the one whose error
# level exposes the sensor: closing the detector on its own rows at the
# coarse-fine faces instead reads order 2.
evolution_study("two levels, C6 :brady_livescu, d8 detector", PERIODIC_NS,
                N -> entropy_case(N; cfl=EVOLUTION_CFL, levels=2, interface_flux=:closure,
                                  deriv=lele_d1_6(closures=:brady_livescu),
                                  art=ArtificialProperties(enabled=true, detector=:d8)),
                entropy_ref;
                primary=:interface, tfinal=0.5, expect=5.9, tol=0.8, recorded=5.93)
evolution_study("two levels, C6, pentadiagonal filter", PERIODIC_NS,
                N -> entropy_case(N; cfl=EVOLUTION_CFL, levels=2, filter_interval=1,
                                  filt=pyranda_filter()), entropy_ref;
                primary=:interface, tfinal=0.5, expect=6.0, tol=0.8, recorded=6.01)

# A level against a slip wall, its tile's face there carrying the wall: the
# refined patch's wall window against the uniform run at its spacing, both in
# the same equal steps (cfl ≈ 0.25 at the fine spacing), so the fine
# closure's own defect and the time error cancel and the row reads what the
# coarse region and the coarse-fine face bring to the wall.
println("\n=== smooth evolution: a level at a wall, fine wall window, t = 0.4 ===")
const LEVEL_WALL_NS = (25, 49, 97)
wall_level_reference(s) = begin
    fine, states = wall_level_case(s.n_global[1]; refined=false)
    fixed_step_run!(fine, states, 0.4, 6 * (s.n_global[1] - 1))
    NodeReference(fine, states)
end
evolution_study("two levels at a slip wall, C6", LEVEL_WALL_NS,
                N -> wall_level_case(N), wall_level_reference;
                primary=:wall, tfinal=0.4, steps=N -> 6 * (N - 1), patches=(2,),
                expect=4.6, tol=0.8, recorded=4.68)
# The same against a symmetry plane: the tile's first node lies half a fine
# cell from the plane, one node beyond the coincident lattice, and the tile
# folds there as the uniform run does, so the window reads what the coarse
# region and the coarse-fine face bring to the plane: the interior order of
# the coarse region, the plane adding no defect of its own. At N = 96 the
# difference is 3e-15, round-off; these grids keep it above 1e-12.
const LEVEL_PLANE_NS = (18, 24, 36)
plane_level_reference(s) = begin
    fine, states = plane_level_case(s.n_global[1]; refined=false)
    fixed_step_run!(fine, states, 0.4, 6 * s.n_global[1])
    NodeReference(fine, states)
end
evolution_study("two levels at a symmetry plane, C6", LEVEL_PLANE_NS,
                N -> plane_level_case(N), plane_level_reference;
                primary=:wall, tfinal=0.4, steps=N -> 6N, patches=(2,),
                expect=6.3, tol=0.8, recorded=6.33)

# A level on the axisymmetric annulus: a cylindrical pulse leaves the refined
# patch through both of its coarse-fine faces, and the patch's interface
# window is read against the uniform run at its spacing in the same equal
# steps. The viscous row reaches the molecular ghost fluxes from the gradient
# ring and the curvature terms they carry.
println("\n=== smooth evolution: a level on an r-z annulus, fine window, t = 0.2 ===")
const ANNULUS_LEVEL_NS = (49, 97, 193)
annulus_level_reference(viscous, steps) = s -> begin
    fine, states = annulus_level_case(s.n_global[1]; refined=false, viscous=viscous)
    fixed_step_run!(fine, states, 0.2, steps(s.n_global[1]))
    NodeReference(fine, states)
end
for (name, viscous, steps, recorded) in
        (("two levels on an r-z annulus, C6", false, N -> 3 * (N - 1), 5.83),
         ("two levels on an r-z annulus, viscous", true, N -> 12 * (N - 1), 5.40))
    evolution_study(name, ANNULUS_LEVEL_NS,
                    N -> annulus_level_case(N; viscous=viscous),
                    annulus_level_reference(viscous, steps);
                    primary=:interface, tfinal=0.2, steps=steps, patches=(2,),
                    expect=5.5, tol=0.8, recorded=recorded)
end

# A level at an NSCBC outflow: an acoustic pulse leaves through the tile's
# face at x = 1, which carries the outflow's characteristic correction at the
# fine spacing, and the refined patch's window at that face is read against
# the uniform run at its spacing in the same equal steps (cfl ≈ 0.26 at the
# fine spacing), halfway through the pulse's exit. The coarse region is under
# a node per pulse width at N = 25, hence the finer sequence.
println("\n=== smooth evolution: a level at an NSCBC outflow, fine face window, " *
        "t = 0.35 ===")
const OUTFLOW_LEVEL_NS = (49, 97, 193)
outflow_level_reference(s) = begin
    fine, states = outflow_level_case(s.n_global[1]; refined=false)
    fixed_step_run!(fine, states, 0.35, 6 * (s.n_global[1] - 1))
    NodeReference(fine, states)
end
evolution_study("two levels at an NSCBC outflow, C6", OUTFLOW_LEVEL_NS,
                N -> outflow_level_case(N), outflow_level_reference;
                primary=:wall, tfinal=0.35, steps=N -> 6 * (N - 1), patches=(2,),
                expect=6.0, tol=0.8, recorded=6.45)

# ---------------------------------------------------------------------------
# Temporal order. Each row integrates one case on one grid in a sequence of
# equal steps, through `run!` one step at a time, and reads the maximum
# difference over the conserved components against the same case on the
# same grid in many more steps (`temporal_errors` in test/smooth_cases.jl).
# The spatial error cancels exactly, so the slope against the step is the
# time integration's; bench/temporalorder.jl runs the full set.
#
# The inflow rows carry their time dependence in the boundary data alone, a
# `DirichletBC` state and an `NSCBCInflowBC` target, both evaluated at the
# stage time, and read the integrator's fourth order. The level rows read
# the coupling schedule instead: the fine solution is injected into the
# covered parent nodes once per completed step, which makes the global step
# first order in the step, and the subcycled row, under the ghost fluxes,
# where that term is smallest, reads the Hermite shell's fourth order above
# it at the largest steps. With the injection removed both read four.

function temporal_study(name, steps, build, tfinal, ref_steps; expect, tol, recorded)
    t0 = time(); c0 = compile_ns()
    errs = temporal_errors(build, tfinal, steps, ref_steps)
    dts = tfinal ./ collect(steps)
    p = observed_order(dts, errs)
    @printf("%-38s  ", name)
    for (n, e) in zip(steps, errs)
        @printf("n=%-4d %.3e  ", n, e)
    end
    @printf("order ≈ %.2f  (%s)\n", p,
            join((@sprintf("%.2f", o) for o in successive_orders(dts, errs)), " / "))
    push!(PHASE_LOG, (name, time() - t0, (compile_ns() - c0) / 1e9))
    _guard(name, p, expect, tol, recorded)
    p
end

# A cfl the endpoint clip always undercuts, so every step is the requested one.
const TEMPORAL_CFL = 50.0

println("\n=== temporal order: fixed grid, equal steps ===")
temporal_study("Dirichlet inflow g(t), N = 33", (48, 96, 192),
               () -> inflow_case(33; cfl=TEMPORAL_CFL), 0.4, 1536;
               expect=4.0, tol=0.5, recorded=3.99)
temporal_study("NSCBC inflow target(t), N = 33", (24, 48, 96),
               () -> target_inflow_case(33; cfl=TEMPORAL_CFL), 0.4, 768;
               expect=4.0, tol=0.5, recorded=4.09)
temporal_study("two levels, global step, N = 48", (24, 48, 96),
               () -> entropy_case(48; levels=2, cfl=TEMPORAL_CFL), 0.5, 1536;
               expect=1.0, tol=0.5, recorded=1.00)
temporal_study("two levels, subcycled, ghost fluxes", (14, 20, 28),
               () -> entropy_case(96; levels=2, subcycle=true, interface_flux=:ghost,
                                  cfl=TEMPORAL_CFL), 0.5, 2240;
               expect=3.8, tol=0.6, recorded=3.85)

# The additive pair of src/imex.jl, the molecular conduction in its implicit
# half and fixed steps under `step_rule = :none`: the viscous standing wave
# on the periodic line at Pr = 0.7, where the conductive rate is below the
# acoustic one and the pair reads four, and between adiabatic no-slip walls
# at Pr = 0.007, where the step is 13 to 51 times the explicit diffusive
# limit, the wall mirror of the staggered stage is in the implicit solve,
# and the order falls toward the implicit half's stage order. Each stage is
# solved to 1e-14 of the thermal energy, below the reference's time error.
const IMEX_TEMPORAL = ImplicitConduction(step_rule=:none, rtol=1e-14)
temporal_study("pair, viscous periodic wave, N = 32", (12, 24, 48),
               () -> viscous_periodic_case(32; cfl=TEMPORAL_CFL, implicit=IMEX_TEMPORAL),
               0.4, 384; expect=4.0, tol=0.5, recorded=4.02)
temporal_study("pair, no-slip walls, Pr = 0.007, N = 33", (16, 32, 64),
               () -> wall_case(33; viscous=true, Pr=0.007, cfl=TEMPORAL_CFL,
                               implicit=IMEX_TEMPORAL),
               0.4, 512; expect=3.6, tol=0.6, recorded=3.63)

# ---------------------------------------------------------------------------
# Taylor–Green vortex: dissipation-rate history at Re = 1600. Reference peak
# dissipation occurs near t ≈ 9 with ε ≈ 1.2e-2 (van Rees et al. 2011);
# at 64³ a well-behaved code lands within a few percent, and a broken
# viscous term or filter shows up as a badly misplaced or damped peak.
# This is the long one — comment it out for quick iterations.

function taylor_green_ke(N; tfinal=10.0, Re=1600.0)
    γ = 1.4; c0 = 10.0; p0 = c0^2 / γ
    prob = Problem(eos=IdealSpecies("gas"; R=1.0, gamma=γ),
                   transport=ConstantTransport(mu0=1 / Re),
                   domain=((0.0, 2π), (0.0, 2π), (0.0, 2π)), bcs=per3,
                   ic=(x, y, z) -> Prim(
                       u=(sin(x) * cos(y) * cos(z), -cos(x) * sin(y) * cos(z), 0.0),
                       p=p0 + (1 / 16) * (cos(2x) + cos(2y)) * (cos(2z) + 2),
                       rho=1.0))
    solver, Q = setup(prob,
                      Numerics(n_global=(N, N, N), art=ArtificialProperties(enabled=false),
                                cfl=0.6))
    cellvol = prod(solver.h)
    ts = Float64[]; kes = Float64[]
    cb = (solver, Q) -> begin
        ke = 0.0
        for k in 1:solver.decomp.n_local[3], j in 1:solver.decomp.n_local[2], i in 1:solver.decomp.n_local[1]
            I = padded_index(solver, i, j, k)
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
