# Calibration sweep for the Cook artificial-property constants.
#
#   julia --project=. -t auto bench/artcal.jl            # everything (~6 min)
#   julia --project=. -t auto bench/artcal.jl beta cfl   # named sweeps only
#
# Sweeps available: mu beta kappa D cfl resolution sensor smoother detector
#
# Scratch tooling, like everything else in bench/: it prints tables, asserts
# nothing, and is not part of the gate. The conclusions drawn from a run of it
# are written up in reference/CALIBRATION.md — update that file when this one
# is re-run with different cases, or the write-up silently goes stale.
#
# The cases come from test/cases.jl, the same file test/validation.jl guards,
# so a constant swept here is measured against exactly the run the regression
# guard is set from.
#
# What each column responds to, and why these cases:
#
#   Noh   plateau, wall deficit, robustness — the shock-thickness constants
#         (C_mu, C_beta) and the entropy error at the symmetry point. The only
#         case where the right answer is a fixed number in three geometries.
#   Lax   contact and star-state error — what an over-damping constant costs on
#         an ordinary shock tube.
#   Shu   wave-train amplitude — what over-damping costs on the smooth structure
#         the high-order scheme is actually for. This is the constraint pulling
#         the other way from Noh.
#   WC    survival at a 10^5 pressure ratio — a pass/fail robustness floor.
#   Mix   interface width — the only case that isolates C_D.

using MPI
MPI.Init(threadlevel=:funneled)
using CompactLES
using Printf

const CL = CompactLES
include(joinpath(@__DIR__, "..", "test", "references.jl"))
include(joinpath(@__DIR__, "..", "test", "cases.jl"))

const DEFAULTS = ArtParams()
const ALL = ["mu", "beta", "kappa", "D", "cfl", "resolution", "sensor", "smoother",
             "detector"]
# Sweep names are bare words; `key=value` sets the background configuration that
# every sweep then runs against. Refitting a constant under a changed smoother
# is exactly `artcal.jl kappa smoother=gaussian`, and keeping the two forms in
# one argument list is what stops that turning into a second script.
#
# The background defaults are read off `ArtParams()` rather than written out, so
# a bare run measures against the shipped configuration and follows it when the
# shipped one moves. They used to be spelled out, and went stale the day the
# smoother default changed: every sweep then silently measured on top of
# `:compact` while the solver shipped `:gaussian`.
const OPTS = script_args(filter(a -> occursin('=', a), ARGS),
                         (smoother = DEFAULTS.smoother,
                          detector = DEFAULTS.detector))
const NAMES = filter(a -> !occursin('=', a), ARGS)
const WHICH = isempty(NAMES) ? ALL : NAMES
want(name) = name in WHICH

# A sweep deliberately visits settings that do not work, and a setting that does
# not work costs unbounded time rather than failing (see NMAX in cases.jl). Every
# case here is capped at a few times the healthy step count; anything that hits
# the cap is reported as NaN, which is the answer the sweep wants anyway.
const CAP = 30_000

art(; kw...) = ArtParams(; enabled=true,
                         C_mu=DEFAULTS.C_mu, C_beta=DEFAULTS.C_beta,
                         C_kappa=DEFAULTS.C_kappa, C_D=DEFAULTS.C_D,
                         beta_sensor=DEFAULTS.beta_sensor,
                         smoother=OPTS.smoother, detector=OPTS.detector, kw...)

mark(v, d) = v == d ? "*" : " "     # flags the shipped default in a sweep

"""
    attempt(f, blank)

Run one swept configuration, returning `blank` if it raises `SolverFailure`.

`StepControl` detects positivity loss and `run!` raises, which is what a sweep
wants for the ~150 steps of wall time it saves — but an uncaught raise ends the
whole sweep at its first bad point, which is not. The failure is raised off a
reduced quantity, so under `mpiexec` every rank lands here at the same step;
nothing else may be caught, on pain of ranks disagreeing about how to continue.
"""
function attempt(f, blank)
    try
        return f()
    catch err
        err isa SolverFailure || rethrow()
        return blank
    end
end

# --- per-case measurements, one line each -----------------------------------

m_noh(ν; kw...) = attempt((NaN, NaN, NaN)) do
    xs, ρ, _, _, ok = noh_case(ν; nmax=CAP, kw...)
    ok || return (NaN, NaN, NaN)
    plat, deficit, Rs, _ = noh_metrics(xs, ρ, ν)
    (plat / 4.0^ν, deficit, Rs)
end

m_lax(; kw...) = attempt((NaN, NaN)) do
    xs, ρ, u, p, ok = lax(; nmax=CAP, kw...)
    ok || return (NaN, NaN)
    ex = [riemann_profile(x, LAX_T, 0.5, LAX_L, LAX_R, 1.4) for x in xs]
    # Contact width across the star-region density jump, which is where an
    # over-large C_kappa or C_beta shows up first.
    (l1(ρ, [e[1] for e in ex]), contact_width(xs, ρ, 0.5, 1.3))
end

m_shu(; kw...) = attempt((NaN, NaN)) do
    xs, ρ, _, _, ok = shu_osher(; N=400, nmax=CAP, kw...)
    ok || return (NaN, NaN)
    band = so_band(xs)
    (maximum(ρ[band]) - minimum(ρ[band]), maximum(ρ[band]))
end

m_wc(; kw...) = attempt((NaN, NaN)) do
    xs, ρ, _, _, ok = woodward(; N=400, nmax=CAP, kw...)
    ok || return (NaN, NaN)
    (maximum(ρ), xs[argmax(ρ)])
end

m_mix(; kw...) = attempt(NaN) do
    xs, Y, _, _, ok = species_advection(; nmax=CAP, kw...)
    ok || return NaN
    contact_width(xs, Y, 0.0, 1.0)
end

hr() = println(repeat("-", 96))

# ===========================================================================
if want("mu")
    println("\n=== C_mu sweep (shear artificial viscosity) ===")
    println("C_mu      | Noh1 plat/exact  deficit | Noh3 plat/exact | Lax L1  contact | Shu train amp")
    hr()
    for c in (0.0, 0.0005, 0.002, 0.008, 0.032)
        a = art(C_mu=c)
        n1 = m_noh(1; art=a); n3 = m_noh(3; art=a)
        lx = m_lax(art=a);    sh = m_shu(art=a)
        @printf("%-8.4g%s | %14.4f  %+6.0f%% | %15.4f | %7.1e %7.4f | %.4f\n",
                c, mark(c, DEFAULTS.C_mu), n1[1], 100n1[2], n3[1], lx[1], lx[2], sh[1])
    end
end

# ===========================================================================
if want("beta")
    println("\n=== C_beta sweep (bulk artificial viscosity — the shock constant) ===")
    println("C_beta    | Noh1 plat/exact  deficit | Noh3 plat/exact | Lax L1  contact | Shu train amp | WC peak")
    hr()
    for c in (0.0, 0.25, 0.5, 1.0, 2.0, 4.0)
        a = art(C_beta=c)
        n1 = m_noh(1; art=a); n3 = m_noh(3; art=a)
        lx = m_lax(art=a);    sh = m_shu(art=a); wc = m_wc(art=a)
        @printf("%-8.4g%s | %14.4f  %+6.0f%% | %15.4f | %7.1e %7.4f | %13.4f | %.4f\n",
                c, mark(c, DEFAULTS.C_beta), n1[1], 100n1[2], n3[1],
                lx[1], lx[2], sh[1], wc[1])
    end
end

# ===========================================================================
if want("kappa")
    println("\n=== C_kappa sweep (artificial conductivity) ===")
    println("C_kappa   | Noh1 plat/exact  deficit | Noh3 plat/exact | Lax L1  contact | WC peak")
    hr()
    for c in (0.0, 0.0025, 0.01, 0.04, 0.16)
        a = art(C_kappa=c)
        n1 = m_noh(1; art=a); n3 = m_noh(3; art=a)
        lx = m_lax(art=a);    wc = m_wc(art=a)
        @printf("%-8.4g%s | %14.4f  %+6.0f%% | %15.4f | %7.1e %7.4f | %.4f\n",
                c, mark(c, DEFAULTS.C_kappa), n1[1], 100n1[2], n3[1],
                lx[1], lx[2], wc[1])
    end
end

# ===========================================================================
if want("D")
    println("\n=== C_D sweep (artificial species diffusivity) ===")
    println("C_D       | interface width after t=$(MIX_T) at u=$(MIX_U) (initial 2h = $(round(2/MIX_N, digits=5)))")
    hr()
    for c in (0.0, 0.0025, 0.01, 0.04, 0.16)
        @printf("%-8.4g%s | %.5f\n", c, mark(c, DEFAULTS.C_D), m_mix(art=art(C_D=c)))
    end
end

# ===========================================================================
if want("cfl")
    println("\n=== CFL sweep (the lagged-coefficient limit, not a constant) ===")
    println("cfl       | Noh1 plat/exact  deficit | Noh2 plat/exact | Noh3 plat/exact | WC peak")
    hr()
    for c in (0.4, 0.3, 0.2, 0.15, 0.1)
        n1 = m_noh(1; cfl=c); n2 = m_noh(2; cfl=c); n3 = m_noh(3; cfl=c)
        wc = m_wc(cfl=c)
        @printf("%-9.3g | %14.4f  %+6.0f%% | %15.4f | %15.4f | %.4f\n",
                c, n1[1], 100n1[2], n2[1], n3[1], wc[1])
    end
    println("  (NaN = the run lost positivity or stalled before reaching t_final)")
end

# ===========================================================================
if want("resolution")
    println("\n=== resolution (are the defaults grid-converged?) ===")
    println("N         | Noh1 plat/exact  deficit | Lax L1   | mix width")
    hr()
    for N in (128, 256, 512, 1024)
        n1 = m_noh(1; N=N)
        lx = m_lax(N=N)
        @printf("%-9d | %14.4f  %+6.0f%% | %8.1e | %.5f\n",
                N, n1[1], 100n1[2], lx[1], m_mix(N=N))
    end
end

# ===========================================================================
if want("sensor")
    println("\n=== beta_sensor (strain vs compression-keyed bulk viscosity) ===")
    println("sensor     | Noh1 plat/exact  deficit | Noh3 plat/exact | Lax L1  contact | Shu train amp | WC peak")
    hr()
    for s in (:strain, :gated_strain, :dilatation)
        a = art(beta_sensor=s)
        n1 = m_noh(1; art=a); n3 = m_noh(3; art=a)
        lx = m_lax(art=a);    sh = m_shu(art=a); wc = m_wc(art=a)
        @printf("%-10s%s | %14.4f  %+6.0f%% | %15.4f | %7.1e %7.4f | %13.4f | %.4f\n",
                s, mark(s, DEFAULTS.beta_sensor), n1[1], 100n1[2], n3[1],
                lx[1], lx[2], sh[1], wc[1])
    end
    # The question the sensor exists to answer: keying β* on compression is the
    # literature's response to a front that is not damped early enough, which is
    # the diagnosis reference/CALIBRATION.md gives for the cfl ≤ 0.15 ceiling.
    # So the ladder is run per sensor, not just the shipped CFL.
    println("\n--- the Noh CFL ceiling, per sensor ---")
    println("sensor      cfl  | Noh1 plat/exact | Noh2 plat/exact | Noh3 plat/exact")
    hr()
    for s in (:strain, :gated_strain, :dilatation), c in (0.4, 0.3, 0.2, 0.15)
        a = art(beta_sensor=s)
        n1 = m_noh(1; art=a, cfl=c); n2 = m_noh(2; art=a, cfl=c)
        n3 = m_noh(3; art=a, cfl=c)
        @printf("%-10s %-5.3g | %15.4f | %15.4f | %15.4f\n", s, c, n1[1], n2[1], n3[1])
    end
    println("  (NaN = the run lost positivity or stalled before reaching t_final)")
end

# The sensor smoother stands in for Cook's Gaussian test filter. `:compact`
# reuses the conserved-state filter, which at alphaf = 0.45 retains 99% of the
# amplitude at four points per wavelength; `:gaussian` is the explicit
# nine-point stencil the reference implementation applies, which retains 19%
# and carries no line solve. The two therefore differ in cost and in answer,
# and the four constants above are calibrated per setting — so a row that
# improves here is not yet an improvement until those are refitted.
# reference/CALIBRATION.md carries the transfer functions.
if want("smoother")
    println("\n=== smoother (the Cook test filter) ===")
    println("smoother   | Noh1 plat/exact  deficit | Noh3 plat/exact | Lax L1  contact | Shu train amp | WC peak")
    hr()
    for s in (:compact, :gaussian)
        a = art(smoother=s)
        n1 = m_noh(1; art=a); n3 = m_noh(3; art=a)
        lx = m_lax(art=a);    sh = m_shu(art=a); wc = m_wc(art=a)
        @printf("%-10s%s | %14.4f  %+6.0f%% | %15.4f | %7.1e %7.4f | %13.4f | %.4f\n",
                s, mark(s, DEFAULTS.smoother), n1[1], 100n1[2], n3[1],
                lx[1], lx[2], sh[1], wc[1])
    end
    println("\n--- the Noh CFL ceiling, per smoother ---")
    println("smoother    cfl  | Noh1 plat/exact | Noh2 plat/exact | Noh3 plat/exact")
    hr()
    for s in (:compact, :gaussian), c in (0.4, 0.3, 0.2, 0.15)
        a = art(smoother=s)
        n1 = m_noh(1; art=a, cfl=c); n2 = m_noh(2; art=a, cfl=c)
        n3 = m_noh(3; art=a, cfl=c)
        @printf("%-10s %-5.3g | %15.4f | %15.4f | %15.4f\n", s, c, n1[1], n2[1], n3[1])
    end
    println("  (NaN = the run lost positivity or stalled before reaching t_final)")
end

# The detector is the high-pass every sensor is built from: Cook's undivided
# δ⁴, or the reference implementation's compact eighth derivative. The two are
# normalized to the same response at two points per wavelength and diverge
# below it — by 26x at four points and 569x at eight — so this sweep asks what
# the solver does with a sensor that stops seeing resolved structure. As with
# the smoother, the four constants are calibrated per setting, so a row that
# improves is not yet an improvement.
if want("detector")
    println("\n=== detector (the sensor high-pass) ===")
    println("detector   | Noh1 plat/exact  deficit | Noh3 plat/exact | Lax L1  contact | Shu train amp | WC peak")
    hr()
    for s in (:delta4, :d8)
        a = art(detector=s)
        n1 = m_noh(1; art=a); n3 = m_noh(3; art=a)
        lx = m_lax(art=a);    sh = m_shu(art=a); wc = m_wc(art=a)
        @printf("%-10s%s | %14.4f  %+6.0f%% | %15.4f | %7.1e %7.4f | %13.4f | %.4f\n",
                s, mark(s, DEFAULTS.detector), n1[1], 100n1[2], n3[1],
                lx[1], lx[2], sh[1], wc[1])
    end
    println("\n--- the Noh CFL ceiling, per detector ---")
    println("detector    cfl  | Noh1 plat/exact | Noh2 plat/exact | Noh3 plat/exact")
    hr()
    for s in (:delta4, :d8), c in (0.5, 0.4, 0.3, 0.2, 0.15)
        a = art(detector=s)
        n1 = m_noh(1; art=a, cfl=c); n2 = m_noh(2; art=a, cfl=c)
        n3 = m_noh(3; art=a, cfl=c)
        @printf("%-10s %-5.3g | %15.4f | %15.4f | %15.4f\n", s, c, n1[1], n2[1], n3[1])
    end
    println("  (NaN = the run lost positivity or stalled before reaching t_final)")
end

println("\nartcal complete")
