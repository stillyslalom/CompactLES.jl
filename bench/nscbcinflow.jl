# The transverse terms of the characteristic boundary conditions: the
# reflection of an obliquely incident acoustic pulse and the entry or exit of
# a vortex, under the weights `beta_t` of the transverse terms at an inflow
# and at an outflow face.
#
#   julia --project=. -t 1 bench/nscbcinflow.jl [pulse vortex outflow]
#   julia --project=. -t 1 bench/nscbcinflow.jl pulse M=0.1
#   julia --project=. -t 1 bench/nscbcinflow.jl vortex etas=0.28,1,4 strength=0.5
#   julia --project=. -t 1 bench/nscbcinflow.jl outflow betas=0,-1,0.7,1
#
# Parts:
#
#   pulse    a Gaussian pressure pulse released half a unit inside the
#            inflow face, whose upstream half meets the face at every
#            incidence angle; the reflection under each `beta_t` of
#            NSCBCInflowBC, at the default relaxation rates
#   vortex   an isentropic vortex entering through the inflow face from a
#            time-dependent target, the analytic vortex at the plane; the
#            error of its imposition under each `beta_t` and each
#            relaxation rate `eta`
#   outflow  the same pulse and vortex leaving through an NSCBCOutflowBC
#            face, under the weights 0, M (`beta_t = -1`, the default),
#            1 - M and 1 unless `betas=` says otherwise
#
# Settings (`key=value`): N (points per unit length), M (stream Mach
# number), eps (pulse amplitude), r0 (pulse radius), strength (vortex peak
# velocity over the stream velocity), rv (vortex radius), betas and etas
# (comma-separated sweeps for the inflow parts), art (artificial properties
# on), cfl, and Ly (the periodic transverse width).
#
# Every case is measured against the same run on a domain extended by two
# units past the face under test, where the disturbance meets no boundary
# within the run. The difference between the two is the reflection of the
# face, or the error of its imposition, and is reported as the maximum and
# the root-mean-square deviation over the shared region, scaled by the
# disturbance's amplitude: for the pulse, the largest excursion the extended
# run carries across the plane of the face; for the vortex, its peak
# velocity and its pressure depression. Scratch tooling, like everything
# else in bench/: it prints tables and asserts nothing.

using MPI
MPI.Init(threadlevel=:funneled)
using CompactLES
using CompactLES: padded_index
using Printf

const CL = CompactLES

const OPTS = CL.script_args(filter(a -> occursin('=', a), ARGS),
    (N=64, M=0.3, eps=1e-3, r0=0.1, strength=0.3, rv=0.1, betas="0,-1,1",
     etas="0.28", art=false, cfl=0.5, Ly=1.0))
const PARTS = let names = filter(a -> !occursin('=', a), ARGS)
    isempty(names) ? ["pulse", "vortex", "outflow"] : names
end
MPI.Comm_size(MPI.COMM_WORLD) == 1 || error("nscbcinflow.jl runs serially")

const γ = 1.4
const c0 = sqrt(γ)                 # p = ρ = 1, R = 1
const u0 = OPTS.M * c0
parselist(s) = parse.(Float64, split(s, ','))
const BETAS = parselist(OPTS.betas)
const ETAS = parselist(OPTS.etas)

betaname(β) = β < 0 ? "M (local)" : @sprintf("%.2f", β)

# A closed x range at spacing 1/N (N L + 1 nodes) and a periodic y width
# Ly at the same spacing.
function build(xlo, xhi, xbcs, ic)
    nx = round(Int, (xhi - xlo) * OPTS.N) + 1
    ny = round(Int, OPTS.Ly * OPTS.N)
    per = (PeriodicBC(), PeriodicBC())
    solver = Solver(n_global=(nx, ny, 1), L_domain=(xhi - xlo, OPTS.Ly, 1.0),
                    origin=(xlo, 0.0, 0.0), bcs=(xbcs, per, per),
                    eos=IdealSpecies("gas"; gamma=γ, R=1.0),
                    transport=ConstantTransport(mu0=0.0),
                    art=ArtificialProperties(enabled=OPTS.art), cfl=OPTS.cfl)
    Q = allocate_state(solver)
    initialize!(solver, Q, ic)
    return solver, Q
end

inflow(; beta_t=1.0, eta=0.28, target=nothing) =
    NSCBCInflowBC(u=(u0, 0.0, 0.0), T_ion=1.0, eta_u=eta, eta_T=eta, eta_t=eta,
                  eta_Y=eta, beta_t=beta_t, target=target)
outflow(; beta_t=-1.0) = NSCBCOutflowBC(pinf=1.0, beta_t=beta_t)

function interior(solver, Q, name)
    CL.refresh_primitives!(solver, Q)
    f = CL.scalar_field(solver, name)
    nx, ny, _ = solver.decomp.n_local
    return [f[padded_index(solver, i, j, 1)] for i in 1:nx, j in 1:ny]
end

# Maximum and root-mean-square of run − reference over x ∈ [xa, xb], the
# reference being the extended domain, whose nodes coincide with the run's.
function deviation(solver, Q, ref, Qref, name, xa, xb)
    a = interior(solver, Q, name)
    b = interior(ref, Qref, name)
    off = round(Int, (solver.origin[1] - ref.origin[1]) * OPTS.N)
    ia = round(Int, (xa - solver.origin[1]) * OPTS.N) + 1
    ib = round(Int, (xb - solver.origin[1]) * OPTS.N) + 1
    emax = 0.0
    e2 = 0.0
    n = 0
    for j in axes(a, 2), i in ia:ib
        d = a[i, j] - b[i + off, j]
        emax = max(emax, abs(d))
        e2 += d * d
        n += 1
    end
    return emax, sqrt(e2 / n)
end

# The largest pressure excursion the extended run carries across the plane
# x = xf over its whole course: the incident amplitude at the face, read
# after every step from the reference's own state.
function plane_excursion!(ref, xf)
    i = round(Int, (xf - ref.origin[1]) * OPTS.N) + 1
    ny = ref.decomp.n_local[2]
    peak = Ref(0.0)
    cb = (s, Q) -> begin
        CL.refresh_primitives!(s, Q)
        for j in 1:ny
            peak[] = max(peak[], abs(s.p[padded_index(s, i, j, 1)] - 1))
        end
        nothing
    end
    return peak, cb
end

# --- the disturbances -------------------------------------------------------

# An isentropic Gaussian pressure pulse of amplitude eps and radius r0.
pulse_ic(xc, yc) = (x, y, z) -> begin
    g = OPTS.eps * exp(-((x - xc)^2 + (y - yc)^2) / OPTS.r0^2)
    Prim(u=(u0, 0.0, 0.0), p=1 + g, rho=(1 + g)^(1 / γ))
end

# The isentropic vortex of peak velocity A = strength · u0 at radius rv,
# centred at (xc, yc); Yee's form with the radius scaled in.
const A_VORTEX = OPTS.strength * u0
function vortex(x, y, xc, yc)
    rv = OPTS.rv
    q = ((x - xc)^2 + (y - yc)^2) / rv^2
    e = exp((1 - q) / 2)
    du = -A_VORTEX * (y - yc) / rv * e
    dv = A_VORTEX * (x - xc) / rv * e
    T = 1 - (γ - 1) * A_VORTEX^2 / (2γ) * e * e
    return Prim(u=(u0 + du, dv, 0.0), T_ion=T, rho=T^(1 / (γ - 1)))
end
vortex_ic(xc, yc) = (x, y, z) -> vortex(x, y, xc, yc)
vortex_target(xc0, yc) = (x, y, z, t) -> vortex(x, y, xc0 + u0 * t, yc)
# Its pressure depression at the centre, the pressure scale of the errors.
const DP_VORTEX = 1 - (1 - (γ - 1) * A_VORTEX^2 / (2γ) * exp(1))^(γ / (γ - 1))

# --- parts ------------------------------------------------------------------

function pulse_part()
    yc = OPTS.Ly / 2
    xc = 0.5
    # The upstream half reaches the face at c0(1 − M), the reflection
    # returns at c0(1 + M): measured once the reflection is half a unit back
    # inside, over the region it has crossed.
    tend = xc / (c0 * (1 - OPTS.M)) + 0.5 / (c0 * (1 + OPTS.M))
    ref, Qref = build(-2.0, 2.0, (inflow(), outflow()), pulse_ic(xc, yc))
    peak, cb = plane_excursion!(ref, 0.0)
    run!(ref, Qref; tfinal=tend, callback=cb)
    @printf("\npulse reflection at the inflow: M = %.2f, eps = %.1e, incident amplitude at the plane %.3e, t = %.3f\n",
            OPTS.M, OPTS.eps, peak[], tend)
    @printf("  %-10s  %12s  %12s\n", "beta_t", "max|dp|/inc", "rms|dp|/inc")
    for β in BETAS
        s, Q = build(0.0, 2.0, (inflow(; beta_t=β), outflow()), pulse_ic(xc, yc))
        run!(s, Q; tfinal=tend)
        emax, erms = deviation(s, Q, ref, Qref, :p, 0.0, 1.5)
        @printf("  %-10s  %12.3e  %12.3e\n", betaname(β), emax / peak[],
                erms / peak[])
    end
end

function vortex_part()
    yc = OPTS.Ly / 2
    xc0 = -0.5
    tend = 1.5 / u0                       # centre at x = 1
    ref, Qref = build(-1.5, 2.0, (inflow(), outflow()), vortex_ic(xc0, yc))
    run!(ref, Qref; tfinal=tend)
    @printf("\nvortex entry through the inflow: M = %.2f, peak velocity %.3f (%.2f of the stream), pressure depression %.3e, t = %.3f\n",
            OPTS.M, A_VORTEX, OPTS.strength, DP_VORTEX, tend)
    @printf("  %-6s  %-10s  %12s  %12s  %12s  %12s\n", "eta", "beta_t",
            "max|du|/A", "rms|du|/A", "max|dv|/A", "max|dp|/dp0")
    for η in ETAS, β in BETAS
        bc = inflow(; beta_t=β, eta=η, target=vortex_target(xc0, yc))
        s, Q = build(0.0, 2.0, (bc, outflow()), vortex_ic(xc0, yc))
        run!(s, Q; tfinal=tend)
        umax, urms = deviation(s, Q, ref, Qref, :u, 0.0, 1.6)
        vmax, _ = deviation(s, Q, ref, Qref, :v, 0.0, 1.6)
        pmax, _ = deviation(s, Q, ref, Qref, :p, 0.0, 1.6)
        @printf("  %-6.2f  %-10s  %12.3e  %12.3e  %12.3e  %12.3e\n", η,
                betaname(β), umax / A_VORTEX, urms / A_VORTEX, vmax / A_VORTEX,
                pmax / DP_VORTEX)
    end
end

function outflow_part()
    yc = OPTS.Ly / 2
    betas = OPTS.betas == "0,-1,1" ? [0.0, -1.0, 1 - OPTS.M, 1.0] : BETAS
    # The pulse: released half a unit inside the outflow face, the reflection
    # returning upstream at c0(1 − M).
    xc = 1.5
    tend = 0.5 / (c0 * (1 + OPTS.M)) + 0.5 / (c0 * (1 - OPTS.M))
    ref, Qref = build(0.0, 4.0, (inflow(), outflow()), pulse_ic(xc, yc))
    peak, cb = plane_excursion!(ref, 2.0)
    run!(ref, Qref; tfinal=tend, callback=cb)
    @printf("\npulse reflection at the outflow: M = %.2f, eps = %.1e, incident amplitude at the plane %.3e, t = %.3f\n",
            OPTS.M, OPTS.eps, peak[], tend)
    @printf("  %-10s  %12s  %12s\n", "beta_t", "max|dp|/inc", "rms|dp|/inc")
    for β in betas
        s, Q = build(0.0, 2.0, (inflow(), outflow(; beta_t=β)), pulse_ic(xc, yc))
        run!(s, Q; tfinal=tend)
        emax, erms = deviation(s, Q, ref, Qref, :p, 0.5, 2.0)
        @printf("  %-10s  %12.3e  %12.3e\n", betaname(β), emax / peak[],
                erms / peak[])
    end
    # The vortex: released a unit inside the face, measured once its centre
    # is half a unit past it.
    xc0 = 1.0
    tend = 1.5 / u0
    ref, Qref = build(0.0, 4.0, (inflow(), outflow()), vortex_ic(xc0, yc))
    run!(ref, Qref; tfinal=tend)
    @printf("\nvortex exit through the outflow: M = %.2f, peak velocity %.3f, pressure depression %.3e, t = %.3f\n",
            OPTS.M, A_VORTEX, DP_VORTEX, tend)
    @printf("  %-10s  %12s  %12s  %12s  %12s\n", "beta_t", "max|du|/A",
            "rms|du|/A", "max|dv|/A", "max|dp|/dp0")
    for β in betas
        s, Q = build(0.0, 2.0, (inflow(), outflow(; beta_t=β)), vortex_ic(xc0, yc))
        run!(s, Q; tfinal=tend)
        umax, urms = deviation(s, Q, ref, Qref, :u, 0.4, 2.0)
        vmax, _ = deviation(s, Q, ref, Qref, :v, 0.4, 2.0)
        pmax, _ = deviation(s, Q, ref, Qref, :p, 0.4, 2.0)
        @printf("  %-10s  %12.3e  %12.3e  %12.3e  %12.3e\n", betaname(β),
                umax / A_VORTEX, urms / A_VORTEX, vmax / A_VORTEX,
                pmax / DP_VORTEX)
    end
end

const PART_FNS = Dict("pulse" => pulse_part, "vortex" => vortex_part,
                      "outflow" => outflow_part)
for part in PARTS
    haskey(PART_FNS, part) || error("unknown part $part; parts are " *
                                    join(sort(collect(keys(PART_FNS))), ", "))
    PART_FNS[part]()
end
