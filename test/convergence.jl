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
#      `:cascade3` closures are measured alongside the `:cascade4` and
#      `:brady_livescu` alternatives, whose whole purpose is this slope, and
#      one pass of the state filter is measured the same way under its two
#      wall closures, since a filtered run cannot exceed the filter's order.
#   3. Axis/origin/pole order — the sharpest scalar diagnostic of the fold
#      signs: a sign error usually gives O(1) error at the first node, so the
#      slope collapses to ~0 rather than degrading gracefully.
#   4. Closure truncation on a polynomial — one derivative of x^(q+1), q the
#      closure rows' exactness degree, against the actual spacing: the rows'
#      own pointwise order, 3/4/5/7 for :cascade3/:cascade4/C6 and C8
#      :brady_livescu, which the smooth-field slopes of item 2 sit above by
#      the favorable phase of exp(sin(3x)).
#   5. Smooth evolution — the final-time solution error of the wall window,
#      or of the patch/level interface window, on cases with a reference
#      free of closure error (test/smooth_cases.jl). These are the orders a
#      run sees: the closure's, plus what the solution norm gains over the
#      pointwise truncation; the cascade filter caps every wall row near 1.8
#      and the one-sided rows lift the cap; a level interface reads the C6
#      closure cascade at the fine spacing, 3.6, and :brady_livescu 6.0.
#   6. Taylor–Green kinetic-energy decay — an end-to-end physics check
#      against published Re = 1600 data.
#
# The `expect` values below are REGRESSION GUARDS set from measured behaviour,
# not from the formal interior order. Measured on this code (max norm):
#
#   C6 interior 6.01 | C8 interior 8.00 | C10 interior 10.04
#   C6 wall closures 3.17 | C6 wall closures :cascade4 4.02
#   C6 wall closures :brady_livescu 5.88 | C8 wall closures :brady_livescu 7.91
#   filter pass :cascade 1.88 | filter pass :onesided 8.07
#   cyl axis odd 3.71 | cyl axis even 3.00 | resolved-θ axis 3.71
#   spherical origin 2.99
#   polynomial rows: C6 :cascade3 3.00 | :cascade4 4.00 | C6 :brady_livescu 5.00
#   C8 :brady_livescu 7.00
#   wall evolution (window max norm, t = 0.4): inviscid C6 3.93 | cascade
#   filter 1.81 | onesided filter 3.84 | C6 :brady_livescu 5.73 | viscous
#   no-slip C6 3.92 | shear mode 4.71
#   interface evolution (entropy wave, t = 0.5): two patches C6 (k = 1) 3.31 |
#   two levels C6 (k = 3) 3.62 | two levels :brady_livescu 6.01 | three levels
#   subcycled 3.72 | two levels, cascade filter 4.12
#
# Those twenty-eight numbers are also passed to each study as `recorded` and
# guarded to ±0.02, separately from the wide `expect`/`tol` pair. See the
# comment on `study` for which failure each guard reports. Each study also
# prints the order of the L2 norm over the interior, unguarded: the max norm
# is set by the wall rows alone, and Gustafsson's theorem allows the solution
# norm one order more than a boundary closure delivers pointwise, so the two
# together say what a closure set buys in a solution rather than at the wall.
#
# These are GLOBAL max norms, and every fold study closes its outer end with a
# SlipWallBC. The orders near 3 therefore belong to the WALL, not to the fold:
# the global max is attained at the last interior cell in all four fold studies,
# and splitting the norm by region shows the fold's own error converging at
# 6.05-7.01 and sitting three to five orders of magnitude below the interior.
# `bench/foldorder.jl` does that split and carries the numbers; the write-up is
# in reference/CALIBRATION_APPENDIX.md under "The fold closure is not third order".
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

const CL = CompactLES
# The smooth-evolution cases, their references and regional norms; shared
# with bench/boundaryorder.jl, which runs the full matrix. Defines `per3`.
include("smooth_cases.jl")

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

"Root-mean-square error over the interior, the discrete L2 norm per point."
function ferr2(solver, f, fn)
    e = 0.0; n = 0
    nl = solver.decomp.n_local
    for k in 1:nl[3], j in 1:nl[2], i in 1:nl[1]
        e += (f[gidx(solver, i, j, k)] -
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
                  art=ArtParams(enabled=false)),
      (x, y, z) -> sin(x),
      (fn=(x, y, z) -> cos(x), parity=1); expect=6.0, tol=1.2, recorded=6.01)

study("C8 periodic derivative", (16, 32, 64),
      N -> Solver(n_global=(N, 12, 12), L_domain=(2π, 2π, 2π), bcs=per3,
                  deriv=lele_d1_8(), art=ArtParams(enabled=false)),
      (x, y, z) -> sin(x),
      (fn=(x, y, z) -> cos(x), parity=1); expect=8.0, tol=1.5, recorded=8.00)

study("C10 periodic derivative", (16, 24, 32),
      N -> Solver(n_global=(N, 12, 12), L_domain=(2π, 2π, 2π), bcs=per3,
                  deriv=lele_d1_10(), art=ArtParams(enabled=false)),
      (x, y, z) -> sin(x),
      (fn=(x, y, z) -> cos(x), parity=1); expect=10.0, tol=2.5, recorded=10.04)

println("\n=== closed-domain order (boundary closures active) ===")
study("C6 with wall closures", (24, 48, 96),
      N -> Solver(n_global=(N, 12, 12), L_domain=(1.0, 1.0, 1.0),
                  bcs=((SlipWallBC(), SlipWallBC()), per3[2], per3[3]),
                  art=ArtParams(enabled=false)),
      (x, y, z) -> exp(sin(3x)),
      (fn=(x, y, z) -> 3cos(3x) * exp(sin(3x)), parity=1);
      expect=3.2, tol=0.8, recorded=3.17)

study("C6 wall closures, :cascade4", (24, 48, 96),
      N -> Solver(n_global=(N, 12, 12), L_domain=(1.0, 1.0, 1.0),
                  bcs=((SlipWallBC(), SlipWallBC()), per3[2], per3[3]),
                  deriv=lele_d1_6(closures=:cascade4),
                  art=ArtParams(enabled=false)),
      (x, y, z) -> exp(sin(3x)),
      (fn=(x, y, z) -> 3cos(3x) * exp(sin(3x)), parity=1);
      expect=4.0, tol=0.8, recorded=4.02)

study("C6 wall closures, :brady_livescu", (24, 48, 96),
      N -> Solver(n_global=(N, 12, 12), L_domain=(1.0, 1.0, 1.0),
                  bcs=((SlipWallBC(), SlipWallBC()), per3[2], per3[3]),
                  deriv=lele_d1_6(closures=:brady_livescu),
                  art=ArtParams(enabled=false)),
      (x, y, z) -> exp(sin(3x)),
      (fn=(x, y, z) -> 3cos(3x) * exp(sin(3x)), parity=1);
      expect=6.0, tol=1.0, recorded=5.88)

study("C8 wall closures, :brady_livescu", (24, 48, 96),
      N -> Solver(n_global=(N, 12, 12), L_domain=(1.0, 1.0, 1.0),
                  bcs=((SlipWallBC(), SlipWallBC()), per3[2], per3[3]),
                  deriv=lele_d1_8(closures=:brady_livescu),
                  art=ArtParams(enabled=false)),
      (x, y, z) -> exp(sin(3x)),
      (fn=(x, y, z) -> 3cos(3x) * exp(sin(3x)), parity=1);
      expect=8.0, tol=1.2, recorded=7.91)

# One pass of the state filter on a closed line, measured as |F f − f|. The
# default wall cascade (identity, F2, F4, F6) is second order along the whole
# line, not only at the wall, because the compact solve carries the row-2
# error inward; the one-sided Gaitonde–Visbal rows restore the interior
# order. Resolutions are lower because the eighth-order pass reaches
# round-off by N = 48 on this field.
study("C8 filter pass, :cascade", (12, 16, 24, 32),
      N -> Solver(n_global=(N, 12, 12), L_domain=(1.0, 1.0, 1.0),
                  bcs=((SlipWallBC(), SlipWallBC()), per3[2], per3[3]),
                  art=ArtParams(enabled=false)),
      (x, y, z) -> exp(sin(3x)),
      (fn=(x, y, z) -> exp(sin(3x)), parity=1);
      expect=2.0, tol=0.6, recorded=1.88, op=:filter)

study("C8 filter pass, :onesided", (12, 16, 24, 32),
      N -> Solver(n_global=(N, 12, 12), L_domain=(1.0, 1.0, 1.0),
                  bcs=((SlipWallBC(), SlipWallBC()), per3[2], per3[3]),
                  filt=compact_filter(0.45; closures=:onesided),
                  art=ArtParams(enabled=false)),
      (x, y, z) -> exp(sin(3x)),
      (fn=(x, y, z) -> exp(sin(3x)), parity=1);
      expect=8.0, tol=1.5, recorded=8.07, op=:filter)

println("\n=== coordinate-singularity folds ===")
study("cylindrical axis, odd field (u_r-like)", (32, 64, 128),
      N -> Solver(n_global=(N, 1, 12), L_domain=(1.0, 1.0, 0.5),
                  metric=CylindricalMetric(),
                  bcs=((AxisBC(), SlipWallBC()), per3[2], per3[3]),
                  art=ArtParams(enabled=false)),
      (r, θ, z) -> r * exp(-4r^2),
      (fn=(r, θ, z) -> (1 - 8r^2) * exp(-4r^2), parity=-1);
      expect=3.7, tol=0.8, recorded=3.71)

study("cylindrical axis, even field (scalar)", (32, 64, 128),
      N -> Solver(n_global=(N, 1, 12), L_domain=(1.0, 1.0, 0.5),
                  metric=CylindricalMetric(),
                  bcs=((AxisBC(), SlipWallBC()), per3[2], per3[3]),
                  art=ArtParams(enabled=false)),
      (r, θ, z) -> exp(-4r^2),
      (fn=(r, θ, z) -> -8r * exp(-4r^2), parity=1);
      expect=3.0, tol=0.8, recorded=3.00)

study("resolved-θ axis, x-like field", (32, 64, 128),
      N -> Solver(n_global=(N, 16, 1), L_domain=(1.0, 2π, 1.0),
                  metric=CylindricalMetric(),
                  bcs=((AxisBC(), SlipWallBC()), per3[2], per3[3]),
                  art=ArtParams(enabled=false)),
      (r, θ, z) -> r * cos(θ) * exp(-4r^2),
      (fn=(r, θ, z) -> cos(θ) * (1 - 8r^2) * exp(-4r^2), parity=1);
      expect=3.7, tol=0.8, recorded=3.71)

study("spherical origin, radial Gaussian", (24, 48, 96),
      N -> Solver(n_global=(N, 12, 12), L_domain=(1.0, π, 2π),
                  metric=SphericalMetric(),
                  bcs=((OriginBC(), SlipWallBC()),
                       (PoleBC(), PoleBC()), per3[3]),
                  art=ArtParams(enabled=false)),
      (r, θ, φ) -> exp(-4r^2),
      (fn=(r, θ, φ) -> -8r * exp(-4r^2), parity=1);
      expect=3.0, tol=0.8, recorded=2.99)

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
                         expect, tol, recorded)
    t0 = time(); c0 = compile_ns()
    hs = Float64[]; errs = Float64[]; errs2 = Float64[]
    for N in Ns
        solver, states = build(N)
        run!(solver, states; tfinal=tfinal)
        e = regional_errors(solver, states, reference(solver); comp=comp)
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
const PERIODIC_NS = (48, 96, 192)      # periodic roots, multiples of 24 for the nest

# The periodic mirror of the wall run that has just finished, at the same
# spacing and step: the wall problem without its closure rows.
function mirror_reference(solver; viscous=false, opts...)
    mirror, states = mirror_case(solver.n_global[1]; viscous=viscous,
                                 cfl=EVOLUTION_CFL, opts...)
    run!(mirror, states; tfinal=0.4)
    NodeReference(mirror, states)
end

println("\n=== closure truncation on a polynomial (actual spacing) ===")
truncation_study("C6 :cascade3 rows, d/dx x^4", (17, 33, 65, 129), lele_d1_6(), 4;
                 expect=3.0, tol=0.5, recorded=3.00)
truncation_study("C6 :cascade4 rows, d/dx x^5", (17, 33, 65, 129),
                 lele_d1_6(closures=:cascade4), 5; expect=4.0, tol=0.5, recorded=4.00)
truncation_study("C6 :brady_livescu rows, d/dx x^6", (17, 33, 65, 129),
                 lele_d1_6(closures=:brady_livescu), 6; expect=5.0, tol=0.5, recorded=5.00)
truncation_study("C8 :brady_livescu rows, d/dx x^8", (17, 33, 65, 129),
                 lele_d1_8(closures=:brady_livescu), 8; expect=7.0, tol=0.5, recorded=7.00)

println("\n=== smooth evolution: wall window, t = 0.4 ===")
evolution_study("inviscid wall, C6, unfiltered", WALL_NS,
                N -> wall_case(N; cfl=EVOLUTION_CFL), mirror_reference;
                primary=:wall, tfinal=0.4, expect=3.9, tol=0.8, recorded=3.93)
evolution_study("inviscid wall, C6, cascade filter", WALL_NS,
                N -> wall_case(N; cfl=EVOLUTION_CFL, filter_interval=1),
                s -> mirror_reference(s; filter_interval=1);
                primary=:wall, tfinal=0.4, expect=1.8, tol=0.6, recorded=1.81)
evolution_study("inviscid wall, C6, onesided filter", WALL_NS,
                N -> wall_case(N; cfl=EVOLUTION_CFL, filter_interval=1,
                               filt=compact_filter(0.45; closures=:onesided)),
                s -> mirror_reference(s; filter_interval=1,
                                      filt=compact_filter(0.45; closures=:onesided));
                primary=:wall, tfinal=0.4, expect=3.8, tol=0.8, recorded=3.84)
evolution_study("inviscid wall, C6 :brady_livescu, unfiltered", WALL_NS,
                N -> wall_case(N; cfl=EVOLUTION_CFL, deriv=lele_d1_6(closures=:brady_livescu)),
                s -> mirror_reference(s; deriv=lele_d1_6(closures=:brady_livescu));
                primary=:wall, tfinal=0.4, expect=5.7, tol=0.8, recorded=5.73)
evolution_study("viscous no-slip wall, C6, unfiltered", WALL_NS,
                N -> wall_case(N; cfl=EVOLUTION_CFL, viscous=true),
                s -> mirror_reference(s; viscous=true);
                primary=:wall, tfinal=0.4, expect=3.9, tol=0.8, recorded=3.92)
evolution_study("shear mode, no-slip wall, C6, unfiltered", WALL_NS,
                N -> shear_case(N; cfl=EVOLUTION_CFL),
                s -> analytic_reference(s.equations, shear_profile(0.1, 0.005; t=s.t));
                primary=:wall, comp=3, tfinal=0.4, expect=4.7, tol=0.8,
                recorded=4.71)

println("\n=== smooth evolution: interface window, entropy wave, t = 0.5 ===")
entropy_ref(s) = analytic_reference(s.equations, entropy_profile(3, 0.37; t=s.t))
# The same-level interface is measured on the k = 1 wave: at the root
# spacing the k = 3 wave is pre-asymptotic below N = 192 (bench/boundaryorder.jl
# reads 5.1 / 4.7 there against 3.1 / 3.5 at k = 1), while a level interface
# sits at the fine spacing and reads its asymptotic order on either wave.
evolution_study("two patches, C6, k = 1", PERIODIC_NS,
                N -> entropy_case(N; cfl=EVOLUTION_CFL, patch_grid=(2, 1, 1), k=1, phase=0.0),
                s -> analytic_reference(s.equations, entropy_profile(1, 0.0; t=s.t));
                primary=:interface, tfinal=0.5, expect=3.3, tol=0.8, recorded=3.31)
evolution_study("two levels, C6", PERIODIC_NS,
                N -> entropy_case(N; cfl=EVOLUTION_CFL, levels=2), entropy_ref;
                primary=:interface, tfinal=0.5, expect=3.6, tol=0.8, recorded=3.62)
evolution_study("two levels, C6 :brady_livescu", PERIODIC_NS,
                N -> entropy_case(N; cfl=EVOLUTION_CFL, levels=2,
                                  deriv=lele_d1_6(closures=:brady_livescu)), entropy_ref;
                primary=:interface, tfinal=0.5, expect=6.0, tol=0.8, recorded=6.01)
evolution_study("three levels, C6, subcycled", PERIODIC_NS,
                N -> entropy_case(N; cfl=EVOLUTION_CFL, levels=3, subcycle=true), entropy_ref;
                primary=:interface, tfinal=0.5, expect=3.7, tol=0.8, recorded=3.72)
evolution_study("two levels, C6, cascade filter", PERIODIC_NS,
                N -> entropy_case(N; cfl=EVOLUTION_CFL, levels=2, filter_interval=1),
                entropy_ref;
                primary=:interface, tfinal=0.5, expect=4.1, tol=0.8, recorded=4.12)

# ---------------------------------------------------------------------------
# Taylor–Green vortex: dissipation-rate history at Re = 1600. Reference peak
# dissipation occurs near t ≈ 9 with ε ≈ 1.2e-2 (van Rees et al. 2011);
# at 64³ a well-behaved code lands within a few percent, and a broken
# viscous term or filter shows up as a badly misplaced or damped peak.
# This is the long one — comment it out for quick iterations.

function taylor_green_ke(N; tfinal=10.0, Re=1600.0)
    γ = 1.4; c0 = 10.0; p0 = c0^2 / γ
    prob = Problem(eos=IdealSpecies("gas"; R=1.0, gamma=γ),
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
