# Shock-capturing validation battery. Slower than the unit tests; run after
# runtests.jl and mpi_tests.jl are green, alongside test/convergence.jl.
#
#   julia --project=. -t auto test/validation.jl
#   julia --project=. test/validation.jl --generate   # rebuild stored references
#
# The cases themselves live in test/cases.jl, shared with bench/artcal.jl so the
# calibration study and this battery cannot drift apart. This file holds the
# reference and guard for each case.
#
# The battery is split by what each case can be measured against, because the
# two kinds of reference carry very different weight:
#
#   ANALYTIC — independent of this code, so a failure means the code is wrong.
#     Lax    — exact Riemann solution; a stiffer star state than Sod, with a
#              contact the artificial diffusivity has to hold together.
#     Sedov  — self-similar blast trajectory in spherical geometry: a real
#              strong shock driven out through the origin fold, rather than the
#              manufactured smooth fields test/convergence.jl folds.
#     Noh    — exact at all times, in all three geometries. The honest test of
#              an artificial-viscosity code: plateau, shock speed and pre-shock
#              compression are all fixed numbers, so wall heating at the
#              symmetry point shows up as a density deficit with nowhere to
#              hide. This is the case that constrains C_beta and the CFL.
#
#   Stored — this code at 4x resolution. A regression guard, not validation.
#     Shu–Osher        — shock/entropy-wave interaction, the problem the
#                        high-order interior scheme exists for. The criterion is
#                        survival of the post-shock wave train, and there is
#                        no closed form for its amplitude.
#     Woodward–Colella — blast-wave collision; strong-shock robustness plus the
#                        collided contact position after two reflections.
#
# Guards are set from measured behaviour, as in test/convergence.jl, at roughly
# 1.5–2x the measured value. The measured numbers print every run; a moved digit
# after a change not intended to touch numerics indicates a numerical effect.
#
# Measured on this code (serial, C6, ArtParams defaults; each case's CFL is in
# test/cases.jl):
#
#   Lax        L1 rho 4.99e-3, u 7.47e-3, p 7.56e-3
#   Shu-Osher  L1 rho 6.91e-3, wave train 2.09e-2, train peak 4.680
#   Woodward   L1 rho 3.25e-2, peak rho 6.608 at x = 0.7785
#   Sedov      R_s 0.8086 vs 0.8000 analytic (+1.07%), peak rho 5.12 (jump 6)
#   Noh nu=1   plateau 3.9971/4    shock 0.2044/0.2   wall deficit 64%
#   Noh nu=2   plateau 14.988/16   shock 0.2093/0.2   wall deficit 56%
#   Noh nu=3   plateau 62.406/64   shock 0.2091/0.2   wall deficit 27%
#
# These were re-measured in August 2026 when ArtParams.smoother moved to
# :gaussian; reference/CALIBRATION.md carries why, and the previous set under
# :compact for comparison. Every plateau and every pre-shock L1 improved, wall
# heating worsened at nu = 1 and nu = 2 and improved at nu = 3, and the two
# STORED cases drifted by about 1% and 3% of their L1. None of that is a bug:
# it is a different regularization, held to the same guards.
#
# Four significant figures is as far as these are reproducible. A shock-capturing
# run integrates thousands of steps through a nonlinear sensor, so an arithmetic
# reassociation anywhere in the artificial-property path moves the last digit or
# two. The smooth-field orders in test/convergence.jl ARE bit-reproducible and
# are the place to look when a change is supposed to be numerics-neutral.
#
# Two operating limits found while building this, both real; the numbers behind
# them are in reference/CALIBRATION.md.
#
#   * Strong shocks need cfl <= 0.15, not the 0.5 default: above ~0.2 the Noh
#     cases lose positivity within a few hundred steps. The cause is a dispersive
#     undershoot at the shock that the artificial viscosity does not damp, not
#     the one-step lag in compute_dt — that hypothesis was tested with a rate
#     predictor and rejected. StepControl(retries = 4) is the practical answer;
#     reference/CALIBRATION.md has the trace.
#   * The spherical origin will not take a discontinuity that is not resolved
#     over at least ~3 cells, nor the singular t = 0 start of spherical Noh.
#     The cylindrical axis takes both.
#
# --- What has been checked against something other than this code ------------
#
# The two STORED cases are regression guards by construction, so they were spot
# checked separately against quantities that do not come from here:
#
#   Shu-Osher shock position at t = 1.8. The front travels at the Mach-3 speed
#   into the (rho=1, p=1) mean state, S = 3*sqrt(1.4), putting it at
#   x = -4 + 1.8 S = 2.3894. Measured 2.3905 in the N=3200 reference and 2.3890
#   at N=1600 — 0.05%, and not a number this code could get right by accident.
#   The post-shock mean density is 3.847 against the exact post-shock (= left)
#   state 3.8571, 0.25% low, which is the captured-shock smearing.
#
#   Woodward-Colella peak density. Converged here to 6.52 at x = 0.767 (6.57 at
#   N=400 falling monotonically to 6.52 at N=6400). Published reference
#   solutions put the peak near 6.4; schemes that under-resolve the collided
#   shell report ~5.4 at 400 points. So this code is ~2% high against the
#   literature figure and on the correct side of the under-resolution error,
#   which is the expected signature of a low-dissipation scheme on a thin
#   contact. It is a spot check against a number read from a paper, not a
#   digitized profile, and should be treated as such.

using MPI
MPI.Init(threadlevel=:funneled)
using CompactLES
using Printf, Test

const CL = CompactLES
include("references.jl")
include("cases.jl")

const REFDIR = joinpath(@__DIR__, "refs")
const GENERATE = "--generate" in ARGS

isroot() = MPI.Comm_rank(MPI.COMM_WORLD) == 0
say(args...) = isroot() && println(args...)
# Runtime format string: @printf demands a literal, and one call site below
# builds its format by concatenation.
sayf(fmt, args...) = isroot() && print(Printf.format(Printf.Format(fmt), args...))

"Read a stored reference profile as (x, rho, u, p)."
function read_ref(name)
    path = joinpath(REFDIR, name)
    isfile(path) || error("missing reference $path — run with --generate")
    cols = [Float64[] for _ in 1:4]
    for line in eachline(path)
        (isempty(line) || startswith(line, '#')) && continue
        for (c, tok) in enumerate(split(line, ','))
            push!(cols[c], parse(Float64, tok))
        end
    end
    Tuple(cols)
end

function write_ref(name, prof; note)
    mkpath(REFDIR)
    open(joinpath(REFDIR, name), "w") do io
        println(io, "# ", note)
        println(io, "# x,rho,u,p")
        for i in eachindex(prof[1])
            @printf(io, "%.10e,%.10e,%.10e,%.10e\n",
                    prof[1][i], prof[2][i], prof[3][i], prof[4][i])
        end
    end
    println("wrote $name (", length(prof[1]), " points)")
end

"""
    regenerate(name, case, N, xt; delta, note)

Run `case` at 4x resolution and store it on the test grid as `name`. `delta` is
the test resolution's initial smearing width, passed explicitly so the reference
solves the same continuous problem the test does rather than a sharper one — see
the note on `tube` in cases.jl.
"""
function regenerate(name, case, N, xt; delta, note)
    xr, ρr, ur, pr, _ = case(N=4N, delta=delta)
    isroot() && write_ref(name,
        (xt, [interp1(xr, ρr, x) for x in xt], [interp1(xr, ur, x) for x in xt],
             [interp1(xr, pr, x) for x in xt]);
        note="$note, this code at N=$(4N) interpolated to N=$N")
end

# ===========================================================================
say("\n=== Lax shock tube (analytic: exact Riemann) ===")

let (xs, ρ, u, p, ok) = lax()
    @test ok
    ex = [riemann_profile(x, LAX_T, 0.5, LAX_L, LAX_R, 1.4) for x in xs]
    eρ = l1(ρ, [e[1] for e in ex])
    eu = l1(u, [e[2] for e in ex])
    ep = l1(p, [e[3] for e in ex])
    sayf("  N=%d  L1: rho %.3e  u %.3e  p %.3e\n", LAX_N, eρ, eu, ep)
    @test eρ < 1.0e-2
    @test eu < 1.5e-2
    @test ep < 1.5e-2
end

# ===========================================================================
say("\n=== Shu–Osher shock/entropy-wave interaction (stored reference) ===")

GENERATE && regenerate("shu_osher.csv", shu_osher, SO_N,
                       [-5.0 + (i - 1) * 10 / (SO_N - 1) for i in 1:SO_N];
                       delta=2 * 10 / (SO_N - 1), note="Shu-Osher t=1.8")

let (xs, ρ, u, p, ok) = shu_osher()
    @test ok
    xr, ρr, _, _ = read_ref("shu_osher.csv")
    ref = [interp1(xr, ρr, x) for x in xs]
    band = so_band(xs)
    sayf("  N=%d   L1 rho %.3e   L1 rho in wave train %.3e   train peak %.4f\n",
         SO_N, l1(ρ, ref), l1(ρ[band], ref[band]), maximum(ρ[band]))
    @test l1(ρ, ref) < 1.5e-2
    @test l1(ρ[band], ref[band]) < 4e-2
    @test maximum(ρ[band]) > 4.3
end

# ===========================================================================
say("\n=== Woodward–Colella blast-wave interaction (stored reference) ===")

GENERATE && regenerate("woodward_colella.csv", woodward, WC_N,
                       [(i - 1) / (WC_N - 1) for i in 1:WC_N];
                       delta=2 / (WC_N - 1), note="Woodward-Colella t=0.038")

let (xs, ρ, u, p, ok) = woodward()
    @test ok
    xr, ρr, _, _ = read_ref("woodward_colella.csv")
    eρ = l1(ρ, [interp1(xr, ρr, x) for x in xs])
    imax = argmax(ρ)
    sayf("  N=%d   L1 rho %.3e   peak rho %.4f at x = %.4f\n",
         WC_N, eρ, ρ[imax], xs[imax])
    @test all(isfinite, ρ) && minimum(ρ) > 0
    @test eρ < 6e-2
    @test 0.75 < xs[imax] < 0.80      # collided contact position
end

# ===========================================================================
say("\n=== Sedov–Taylor blast through the spherical origin (analytic) ===")

let (rs, ρ, u, p, ok) = sedov()
    @test ok
    Rex = sedov_shock_radius(SEDOV_E, SEDOV_T, 3, 1.4)
    Rnum = front_position(rs, ρ, 2.0)          # outermost crossing of 2·ρ₀
    sayf("  N=%d  R_s %.4f vs %.4f analytic (%+.2f%%)  peak rho %.3f (jump 6)\n",
         SEDOV_N, Rnum, Rex, 100 * (Rnum / Rex - 1), maximum(ρ))
    @test abs(Rnum / Rex - 1) < 0.03
    @test maximum(ρ) > 4.5                     # jump is 6; capture smears it
    @test minimum(ρ) > 0
end

# ===========================================================================
say("\n=== Noh implosion, three geometries (analytic at all times) ===")

for (ν, ptol) in ((1, 0.01), (2, 0.10), (3, 0.15))
    xs, ρ, u, p, ok = noh_case(ν)
    @test ok
    plat, deficit, Rnum, epre = noh_metrics(xs, ρ, ν)
    exact = 4.0^ν
    sayf("  nu=%d N=%d  plateau %.4f (exact %.1f)  wall deficit %+.0f%%  " *
         "shock %.4f/%.4f  L1 pre-shock rho %.2e\n",
         ν, Dict(NOH_N)[ν], plat, exact, 100deficit, Rnum,
         (NOH_G - 1) / 2 * NOH_T, epre)
    @test abs(plat / exact - 1) < ptol
    @test 0 < deficit < 0.7                   # wall heating budget
    @test abs(Rnum - (NOH_G - 1) / 2 * NOH_T) < 0.025
    @test epre < 5e-2
end

say("\nvalidation battery complete")
