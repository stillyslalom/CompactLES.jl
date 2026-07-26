# Calibration sweep for the Cook artificial-property constants.
#
#   julia --project=. -t auto bench/artcal.jl            # everything (~6 min)
#   julia --project=. -t auto bench/artcal.jl beta cfl   # named sweeps only
#
# Sweeps available: mu beta kappa D cfl resolution
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
const WHICH = isempty(ARGS) ? ["mu", "beta", "kappa", "D", "cfl", "resolution"] : ARGS
want(name) = name in WHICH

# A sweep deliberately visits settings that do not work, and a setting that does
# not work costs unbounded time rather than failing (see NMAX in cases.jl). Every
# case here is capped at a few times the healthy step count; anything that hits
# the cap is reported as NaN, which is the answer the sweep wants anyway.
const CAP = 30_000

art(; kw...) = ArtParams(; enabled=true,
                         C_mu=DEFAULTS.C_mu, C_beta=DEFAULTS.C_beta,
                         C_kappa=DEFAULTS.C_kappa, C_D=DEFAULTS.C_D, kw...)

mark(v, d) = v == d ? "*" : " "     # flags the shipped default in a sweep

# --- per-case measurements, one line each -----------------------------------

function m_noh(ν; kw...)
    xs, ρ, _, _, ok = noh_case(ν; nmax=CAP, kw...)
    ok || return (NaN, NaN, NaN)
    plat, deficit, Rs, _ = noh_metrics(xs, ρ, ν)
    (plat / 4.0^ν, deficit, Rs)
end

function m_lax(; kw...)
    xs, ρ, u, p, ok = lax(; nmax=CAP, kw...)
    ok || return (NaN, NaN)
    ex = [riemann_profile(x, LAX_T, 0.5, LAX_L, LAX_R, 1.4) for x in xs]
    # Contact width across the star-region density jump, which is where an
    # over-large C_kappa or C_beta shows up first.
    (l1(ρ, [e[1] for e in ex]), contact_width(xs, ρ, 0.5, 1.3))
end

function m_shu(; kw...)
    xs, ρ, _, _, ok = shu_osher(; N=400, nmax=CAP, kw...)
    ok || return (NaN, NaN)
    band = so_band(xs)
    (maximum(ρ[band]) - minimum(ρ[band]), maximum(ρ[band]))
end

function m_wc(; kw...)
    xs, ρ, _, _, ok = woodward(; N=400, nmax=CAP, kw...)
    ok || return (NaN, NaN)
    (maximum(ρ), xs[argmax(ρ)])
end

function m_mix(; kw...)
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

println("\nartcal complete")
