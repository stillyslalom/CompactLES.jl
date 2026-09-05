# Calibration sweep for the Cook artificial-property constants.
#
#   julia --project=. -t auto bench/artcal.jl            # everything (~6 min)
#   julia --project=. -t auto bench/artcal.jl beta cfl   # named sweeps only
#
# Sweeps available: mu beta kappa D Y cfl resolution sensor smoother detector
# field response miranda bulk
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
#   SI    mass-fraction excursion and interface width at a shocked air/SF6
#         interface — what C_D does where the interface is forced, which Mix
#         never is; the width it costs against the excursion it removes.

using MPI
MPI.Init(threadlevel=:funneled)
using CompactLES
using Printf

const CL = CompactLES
include(joinpath(@__DIR__, "..", "test", "references.jl"))
include(joinpath(@__DIR__, "..", "test", "cases.jl"))

const DEFAULTS = ArtParams()
const ALL = ["mu", "beta", "kappa", "D", "Y", "cfl", "resolution", "sensor",
             "smoother", "detector", "field", "response", "miranda", "bulk"]
# Sweep names are bare words; `key=value` sets the background configuration that
# every sweep then runs against. Refitting a constant under a changed smoother
# is exactly `artcal.jl kappa smoother=gaussian`, and keeping the two forms in
# using one argument list avoids a second script.
#
# The background defaults are read off `ArtParams()` rather than written out, so
# a bare run measures against the default configuration and follows it when the
# default moves. Spelled-out values became stale when the smoother default
# changed: every sweep then silently measured on top of
# `:compact` while the solver defaulted to `:gaussian`.
const OPTS = CompactLES.script_args(filter(a -> occursin('=', a), ARGS),
                         (smoother = DEFAULTS.smoother,
                          detector = DEFAULTS.detector))
const NAMES = filter(a -> !occursin('=', a), ARGS)
const WHICH = isempty(NAMES) ? ALL : NAMES
want(name) = name in WHICH

# A sweep deliberately visits settings that do not work, and a setting that does
# not work costs unbounded time rather than failing (see NMAX in cases.jl). Every
# case here is capped at a few times the healthy step count.
#
# The two ways of not working print differently, and the distinction is not
# cosmetic. **NaN is positivity loss** — a `SolverFailure`, which is a real limit
# of the configuration. **Inf is the step cap** — the run was still healthy and
# had not reached the target time, which is an artifact of CAP and of the
# timestep, not a result. Reporting both as NaN reads as a ceiling that moves
# with the CFL, since
# a smaller timestep needs more steps to cover the same interval; separating them
# exposes a failure that gets *worse* as the timestep falls.
const CAP = 30_000

art(; kw...) = ArtParams(; enabled=true,
                         C_mu=DEFAULTS.C_mu, C_beta=DEFAULTS.C_beta,
                         C_kappa=DEFAULTS.C_kappa, C_D=DEFAULTS.C_D,
                         mu_sensor=DEFAULTS.mu_sensor,
                         beta_sensor=DEFAULTS.beta_sensor,
                         reduction=DEFAULTS.reduction,
                         smoother=OPTS.smoother, detector=OPTS.detector, kw...)

mark(v, d) = v == d ? "*" : " "     # flags the default in a sweep

"""
    attempt(f, blank)

Run one swept configuration, returning `blank` if it raises `SolverFailure`.

`StepControl` detects positivity loss and `run!` raises, saving a sweep ~150
steps of wall time. An uncaught exception would end the whole sweep at its first
bad point. The failure is raised off a
reduced quantity, so under `mpiexec` every rank lands here at the same step;
no other exception may be caught because ranks could disagree about how to continue.
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
    ok || return (Inf, Inf, Inf)
    plat, deficit, Rs, _ = noh_metrics(xs, ρ, ν)
    (plat / 4.0^ν, deficit, Rs)
end

m_lax(; kw...) = attempt((NaN, NaN)) do
    xs, ρ, u, p, ok = lax(; nmax=CAP, kw...)
    ok || return (Inf, Inf)
    ex = [riemann_profile(x, LAX_T, 0.5, LAX_L, LAX_R, 1.4) for x in xs]
    # Contact width across the star-region density jump, which is where an
    # over-large C_kappa or C_beta shows up first.
    (l1(ρ, [e[1] for e in ex]), contact_width(xs, ρ, 0.5, 1.3))
end

m_shu(; kw...) = attempt((NaN, NaN)) do
    xs, ρ, _, _, ok = shu_osher(; N=400, nmax=CAP, kw...)
    ok || return (Inf, Inf)
    band = so_band(xs)
    (maximum(ρ[band]) - minimum(ρ[band]), maximum(ρ[band]))
end

m_wc(; kw...) = attempt((NaN, NaN)) do
    xs, ρ, _, _, ok = woodward(; N=400, nmax=CAP, kw...)
    ok || return (Inf, Inf)
    (maximum(ρ), xs[argmax(ρ)])
end

m_mix(; kw...) = attempt(NaN) do
    xs, Y, _, _, ok = species_advection(; nmax=CAP, kw...)
    ok || return Inf
    contact_width(xs, Y, 0.0, 1.0)
end

# Width and steps come back as floats so a failed row prints as NaN or Inf
# through the same format as a healthy one.
m_si(; kw...) = attempt((NaN, NaN, NaN, NaN)) do
    r = shock_interface(; nmax=CAP, kw...)
    r.completed || return (Inf, Inf, Inf, Inf)
    (r.worst_min_Y, r.worst_max_Y, Float64(r.width_cells), Float64(r.steps))
end

# The shocked interface fails at density ratio 100 under the default channel
# with a DomainError out of the sound speed rather than a SolverFailure, so
# this form also reports that as a failed row instead of ending the sweep.
m_si_ratio(; kw...) = try
    m_si(; kw...)
catch err
    err isa DomainError || rethrow()
    (NaN, NaN, NaN, NaN)
end

m_slab(; kw...) = attempt((NaN, NaN, NaN, NaN)) do
    r = brill_slab(; nmax=CAP, kw...)
    r.completed || return (Inf, Inf, Inf, Inf)
    (r.p_error, r.worst_min_Y, minimum(r.rho), Float64(r.steps))
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
    # C_beta and the CFL ceiling are not separable. The accuracy table above is
    # read at NOH_CFL, and a constant that looks worse there can still be the one
    # that holds a geometry at a larger timestep. That is the whole question
    # under `detector = :d8`, where the planar and cylindrical ceilings are gone
    # and the spherical one falls to 0.25. Run the ladder per constant for the
    # same reason the sensor and detector sweeps run it per setting.
    println("\n--- the Noh CFL ceiling, per C_beta ---")
    println("C_beta      cfl  | Noh1 plat/exact | Noh2 plat/exact | Noh3 plat/exact")
    hr()
    for c in (0.25, 0.5, 1.0, 2.0), cf in (1.0, 0.4, 0.3, 0.25, 0.2, 0.15)
        a = art(C_beta=c)
        n1 = m_noh(1; art=a, cfl=cf); n2 = m_noh(2; art=a, cfl=cf)
        n3 = m_noh(3; art=a, cfl=cf)
        @printf("%-10.4g %-5.3g | %15.4f | %15.4f | %15.4f\n",
                c, cf, n1[1], n2[1], n3[1])
    end
    println("  (NaN = lost positivity; Inf = still healthy at the step cap)")
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
if want("Y")
    println("\n=== C_D at a shocked air/SF6 interface (mass-fraction excursion) ===")
    println("C_D       delta/h | worst min Y   max Y | width cells | steps")
    hr()
    for c in (0.01, 0.1, 1.0), d in (2, 4, 8)
        si = m_si(art=art(C_D=c), delta=d)
        @printf("%-8.4g%s %-7.3g | %+11.4f %7.4f | %11g | %5g\n",
                c, mark(c, DEFAULTS.C_D), d, si[1], si[2], si[3], si[4])
    end
    println("  (NaN = lost positivity; Inf = still healthy at the step cap)")
    println("
=== C_Y at the same interface, delta = 2h (the mass-fraction bound) ===")
    println("C_Y               | worst min Y   max Y | width cells | steps")
    hr()
    for c in (0.0, 50.0, 100.0, 200.0, 1000.0)
        si = m_si(art=art(C_Y=c), delta=2)
        @printf("%-8.4g%s         | %+11.4f %7.4f | %11g | %5g
",
                c, mark(c, DEFAULTS.C_Y), si[1], si[2], si[3], si[4])
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
    println("  (NaN = lost positivity; Inf = still healthy at the step cap)")
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
    # The ladder therefore runs per sensor, not only for the default CFL.
    println("\n--- the Noh CFL ceiling, per sensor ---")
    println("sensor      cfl  | Noh1 plat/exact | Noh2 plat/exact | Noh3 plat/exact")
    hr()
    for s in (:strain, :gated_strain, :dilatation), c in (0.4, 0.3, 0.2, 0.15)
        a = art(beta_sensor=s)
        n1 = m_noh(1; art=a, cfl=c); n2 = m_noh(2; art=a, cfl=c)
        n3 = m_noh(3; art=a, cfl=c)
        @printf("%-10s %-5.3g | %15.4f | %15.4f | %15.4f\n", s, c, n1[1], n2[1], n3[1])
    end
    println("  (NaN = lost positivity; Inf = still healthy at the step cap)")
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
    println("  (NaN = lost positivity; Inf = still healthy at the step cap)")
end

# The detector is the high-pass every sensor is built from: Cook's undivided
# δ⁴, or the reference implementation's compact eighth derivative. The two are
# normalized to the same response at two points per wavelength and diverge
# below it — by 26x at four points and 569x at eight — so this sweep measures the
# solver response to a sensor that stops responding to resolved structure. As with
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
    println("  (NaN = lost positivity; Inf = still healthy at the step cap)")
end

# The field each channel's detector reads. Cook takes μ* and β* from the strain
# magnitude |S|; the reference implementation takes μ* from the velocity
# components and β* from the dilatation, and neither of those carries an
# absolute value. |S| has a cusp wherever the strain passes through zero, and a
# cusp is grid-scale structure at any resolution, so a selective detector and an
# unselective one return the same magnitude through that field. The `response`
# sweep below measures that. The field and the detector are therefore not
# independent changes, and this sweep runs the fields against both detectors.
if want("field")
    println("\n=== sensor fields (which field each channel's detector reads) ===")
    println("detector mu*/beta*             | Noh1 plat/exact  deficit | Noh3 plat/exact | Lax L1  contact | Shu train amp | WC peak")
    hr()
    for det in (:delta4, :d8),
        (ms, bs) in ((:strain, :strain), (:velocity, :strain),
                     (:strain, :ungated_dilatation),
                     (:velocity, :ungated_dilatation))
        a = art(mu_sensor=ms, beta_sensor=bs, detector=det)
        n1 = m_noh(1; art=a); n3 = m_noh(3; art=a)
        lx = m_lax(art=a);    sh = m_shu(art=a); wc = m_wc(art=a)
        @printf("%-8s %-9s %-9s%s | %14.4f  %+6.0f%% | %15.4f | %7.1e %7.4f | %13.4f | %.4f\n",
                det, ms, bs,
                mark((det, ms, bs), (DEFAULTS.detector, DEFAULTS.mu_sensor,
                                     DEFAULTS.beta_sensor)),
                n1[1], 100n1[2], n3[1], lx[1], lx[2], sh[1], wc[1])
    end
    println("\n--- the Noh CFL ceiling, per field pairing ---")
    println("detector mu*/beta*              cfl  | Noh1 plat/exact | Noh2 plat/exact | Noh3 plat/exact")
    hr()
    for det in (:delta4, :d8),
        (ms, bs) in ((:strain, :strain), (:velocity, :strain),
                     (:strain, :ungated_dilatation),
                     (:velocity, :ungated_dilatation)),
        c in (1.0, 0.4, 0.2, 0.15)
        a = art(mu_sensor=ms, beta_sensor=bs, detector=det)
        n1 = m_noh(1; art=a, cfl=c); n2 = m_noh(2; art=a, cfl=c)
        n3 = m_noh(3; art=a, cfl=c)
        @printf("%-8s %-9s %-10s %-5.3g | %15.4f | %15.4f | %15.4f\n",
                det, ms, bs, c, n1[1], n2[1], n3[1])
    end
    println("  (NaN = lost positivity; Inf = still healthy at the step cap)")
end

# Sensor response against wavelength, on one velocity sine and with no time
# integration: what each (field, detector) pair reports before any case is run.
# The comparison is against the detector's own designed response, a factor of
# 569 at eight points per wavelength, 26 at four and 1 at the Nyquist
# (reference/CALIBRATION.md). A pair applied to a smooth field recovers that
# separation; applied to |S| it recovers almost none of it.
if want("response")
    N = 64
    function sine_solver(k, a)
        prob = Problem(eos=IdealSpecies("gas"; gamma=1.4, R=1.0),
                       transport=Transport(mu0=0.0),
                       domain=((0.0, 2π), (0.0, 0.1), (0.0, 0.1)), bcs=per3,
                       ic=(x, y, z) -> Prim(rho=1.0, p=1.0,
                                            u=(cos(k * x), 0.0, 0.0)))
        solver, Q = setup(prob, Numerics(n_global=(N, 1, 1), art=a))
        CL.compute_primitives_and_gradients!(solver, Q)
        CL.compute_artificial!(solver, Q)
        return solver
    end
    peak_mu(k, ms, det) = maximum(sine_solver(k, art(mu_sensor=ms, detector=det)).mu_art)
    peak_beta(k, bs, det) = maximum(sine_solver(k, art(beta_sensor=bs, detector=det)).beta_art)

    println("\n=== sensor response on u = cos(kx), N = $N periodic ===")
    println("k/pi    ppw  | mu* strain: d4     d8     ratio | mu* velocity: d4     d8     ratio")
    hr()
    for k in (2, 4, 8, 16, 24, 32)
        s4 = peak_mu(k, :strain, :delta4);   s8 = peak_mu(k, :strain, :d8)
        v4 = peak_mu(k, :velocity, :delta4); v8 = peak_mu(k, :velocity, :d8)
        @printf("%-6.3f %5.1f | %9.3e %9.3e %7.3g | %9.3e %9.3e %7.3g\n",
                2k / N, N / k, s4, s8, s4 / s8, v4, v8, v4 / v8)
    end
    println("k/pi    ppw  | beta* strain: d4    d8     ratio | beta* dilatation: d4  d8     ratio")
    hr()
    for k in (2, 4, 8, 16, 24, 32)
        s4 = peak_beta(k, :strain, :delta4)
        s8 = peak_beta(k, :strain, :d8)
        v4 = peak_beta(k, :ungated_dilatation, :delta4)
        v8 = peak_beta(k, :ungated_dilatation, :d8)
        @printf("%-6.3f %5.1f | %9.3e %9.3e %7.3g | %9.3e %9.3e %7.3g\n",
                2k / N, N / k, s4, s8, s4 / s8, v4, v8, v4 / v8)
    end
    println("  (k/pi = 1 is the two-point wave; a centered derivative annihilates it,")
    println("   so every sensor built from |S| or from div u reports exactly zero there)")
end

# The reference implementation's current set (Brill, Olson & Bokman 2025, eqs.
# 22–27): the eighth-derivative detector, the directional maximum, μ* from the
# velocity components, β* from the dilatation with the compression switch, and
# constants an order of magnitude or more below Cook 2007's (C_mu = 1e-4,
# C_beta = 7e-2, C_kappa = 1e-3, C_D = 2e-4). Theirs scale with Δ²/Δt where this
# package scales with cΔ, a ratio of roughly 1/CFL ≈ 2.5, so the comparable
# values here are 2.5× theirs. Each option is measured alone in the sweeps
# above; this is the combination, at the package's constants, at the rescaled
# ones, at a midpoint, and at the rescaled ones with C_beta held at 1.0, the
# lower edge of the `:d8` window, since `:dilatation` loses the converging Noh
# geometries at the fold and the question is whether the constant alone
# recovers them.
if want("miranda")
    println("\n=== Miranda's set (sensors, reduction, and rescaled constants) ===")
    println("config             | Noh1 plat   def | Noh2 plat   def | Noh3 plat   def" *
            " | Lax L1  | Shu tr | WC peak | mix wid | SI minY  wid")
    hr()
    sensors = (detector=:d8, reduction=:max, mu_sensor=:velocity,
               beta_sensor=:dilatation)
    scaled = (C_mu=2.5e-4, C_beta=0.175, C_kappa=2.5e-3, C_D=5e-4)
    rows = (("default", art()),
            ("sensors only", art(; sensors...)),
            ("sensors, x1", art(; sensors..., scaled...)),
            ("sensors, x4", art(; sensors..., C_mu=1e-3, C_beta=0.7,
                                C_kappa=1e-2, C_D=2e-3)),
            ("sensors, x1, Cb=1", art(; sensors..., scaled..., C_beta=1.0)))
    for (name, a) in rows
        n1 = m_noh(1; art=a); n2 = m_noh(2; art=a); n3 = m_noh(3; art=a)
        lx = m_lax(art=a);    sh = m_shu(art=a);    wc = m_wc(art=a)
        mx = m_mix(art=a);    si = m_si(art=a, delta=2)
        @printf("%-18s | %9.4f %+5.0f%% | %9.4f %+5.0f%% | %9.4f %+5.0f%%",
                name, n1[1], 100n1[2], n2[1], 100n2[2], n3[1], 100n3[2])
        @printf(" | %7.1e | %6.4f | %7.4f | %7.5f | %+7.4f %4g\n",
                lx[1], sh[1], wc[1], mx, si[1], si[3])
    end
    println("  (NaN = lost positivity; Inf = still healthy at the step cap)")
end

# The bulk species channel (`species_flux = :bulk`) against the default Fickian
# one on the battery rows that carry more than one species, which are the only
# rows the option touches: the advected interface (width), the shocked
# air/SF6 interface at SF6's density ratio and at 100, and the Brill slab at
# density ratio 100 with 7 cells per interface (pressure error at ten
# periods, the paper's stability metric). The single-species rows are
# bit-identical between the two and are not repeated here. The measurements
# behind the option are in reference/CALIBRATION.md, "The bulk species
# channel".
if want("bulk")
    println("\n=== The bulk species channel against the Fickian one ===")
    println("channel  | mix wid | SI 5.04: minY  wid steps | " *
            "SI 100: minY  wid steps | slab 100/7: max|p-1|  minY  min rho  steps")
    hr()
    for flux in (:fickian, :bulk)
        a = art(species_flux=flux)
        mx = m_mix(art=a)
        s1 = m_si_ratio(art=a, delta=2)
        s2 = m_si_ratio(art=a, delta=2, rho_heavy=100.0)
        sl = m_slab(art=a)
        @printf("%-8s | %7.5f | %+7.4f %4g %5g | %+7.4f %4g %5g | %9.2e %+8.4f %7.4f %5g\n",
                flux, mx, s1[1], s1[3], s1[4], s2[1], s2[3], s2[4],
                sl[1], sl[2], sl[3], sl[4])
    end
    println("  (NaN = lost positivity; Inf = still healthy at the step cap)")
end

println("\nartcal complete")
