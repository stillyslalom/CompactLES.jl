# False activation of the artificial species diffusivity D*, the bulk viscosity
# β* and the conductivity κ* on smooth fields, at the production constants and
# the default filter.
#
#   julia --project=. -t 1 bench/falseactivation.jl                     # everything
#   julia --project=. -t 1 bench/falseactivation.jl composition probe   # named parts only
#   julia --project=. -t 1 bench/falseactivation.jl acoustic Ns=64,128 Nvar=64
#
# Parts: `composition` (a two-gas slab of molecular-weight ratio 5 at uniform
# p, T and u, advected one period), `thermal` (a one-gas slab of temperature
# ratio 1.5 at uniform p and u, advected one period), `acoustic` (a right-running
# Gaussian pressure pulse of amplitude `amp` crossing the resting slab of
# `composition`, to t = 0.45, after it has split into its reflected and
# transmitted parts), `probe` (a contact localization evaluated on the final
# states of the three cases and on two battery states, without running it),
# and `split` (the `:d8` detector on the fields of κ* and D* only, run on the
# three cases and on the one-dimensional battery of bench/artcal.jl, which
# takes most of the time), and `splitcd` (C_D re-swept under that split on the
# species rows of the battery and on the composition slab). `species` is the
# species-only split, `:d8` on the mass and mole fractions and δ⁴ on every
# other field, with C_D swept over `cds` on the species rows of the battery,
# the two density-ratio-100 rows and the composition slab at N = 64 and 128;
# `speciescost` times `compute_artificial!` and `compute_rhs!` under it
# against δ⁴ within one process, and is the one part to run at `-t 16`.
# Every slab edge is a tanh of width `W` in physical units, so the ladder `Ns`
# resolves it over N·W cells and the activation falls at the rate the sensor's
# truncation sets.
#
# Each case prints a ladder over `Ns` and a table of single-factor variations
# at `Nvar`. The columns are the time maximum over the run of
#
#   D*/(c h)          the species diffusivity against the grid diffusivity,
#   β*/(ρ c h)        the kinematic bulk viscosity against it,
#   κ*/(ρ c_p c h)    the thermal diffusivity against it,
#
# and the L1 errors at the end: against the exact translation in the two
# advected cases (Y of the heavy gas, and T), and against an art-off run at
# `Nref` points in the acoustic case (the pressure perturbation over `amp`).
# `on − off` is the L1 difference from the same run with the artificial
# properties disabled and the same filter, which is the error the three
# channels deposit. Every variation row prints the state filter's transfer
# function at kh = π/4, π/2 and 3π/4, its cadence in steps and the mean
# relaxation weight a pass took over the run (`filter_weight`), since a
# changed filter or CFL changes the state the sensors read.
#
# Settings (`key=value`): Ns (comma list), Nvar, Nref, W, amp, cfl, nmax, cds
# (comma list of the C_D values `species` sweeps), reps (timing repetitions).
#
# Scratch tooling, like everything else in bench/: it prints tables, asserts
# nothing, and is not part of the gate. The results are written up in
# reference/CALIBRATION_APPENDIX.md, "False activation on smooth fields".

using MPI
MPI.Initialized() || MPI.Init()
using CompactLES
using CompactLES: padded_index, xcoord
using Printf
const CL = CompactLES
include(joinpath(@__DIR__, "..", "test", "references.jl"))
include(joinpath(@__DIR__, "..", "test", "cases.jl"))

const OPTS = CompactLES.script_args(filter(a -> occursin('=', a), ARGS),
                                    (Ns = "64,128,256", Nvar = 128, Nref = 1024,
                                     W = 1 / 16, amp = 1e-2, cfl = 0.5,
                                     nmax = 20_000, cds = "0.1,0.3,1,3", reps = 30))
const NAMES = filter(a -> !occursin('=', a), ARGS)
const PARTS = isempty(NAMES) ? ["composition", "thermal", "acoustic", "probe", "split",
                                "splitcd", "species", "speciescost"] : NAMES
const NS = parse.(Int, split(OPTS.Ns, ','))
want(name) = name in PARTS

const R_HEAVY = 0.2        # gas constant of the heavy gas: density ratio 5 at equal p, T
const T_HOT = 1.5          # temperature ratio of the thermal slab
const U_ADV = 1.0          # advection speed; one period of the unit line is t = 1
const T_ACOUSTIC = 0.45
const SIGMA = 1 / 20       # Gaussian pulse width

# Slab on [0.25, 0.75] with edges of physical width W: near x = 0.25 the profile
# is (1 + tanh((x − 0.25)/W))/2 to leading order. It is built on cos(2πx) so that
# it is analytic across the periodic seam. A difference of two tanh functions is
# not: its slope jumps there by 2e-2 at W = 1/16, and the detector output of
# that kink falls as h and set the finest rows of the ladder when it was tried.
slab(x) = (1 - tanh(cos(2π * x) / (2π * OPTS.W)) / tanh(1 / (2π * OPTS.W))) / 2
wrap(d) = mod(d + 0.5, 1.0) - 0.5

two_gas() = IdealMixture([IdealSpecies{Float64}("light", 1.0, 1.4),
                          IdealSpecies{Float64}("heavy", R_HEAVY, 1.4)])

function problem(case::Symbol, amp)
    dom(h) = ((0.0, 1.0), (0.0, h), (0.0, h))
    if case === :composition
        eos = two_gas()
        ic = (x, y, z) -> (Yh = slab(x);
                           Prim(Y=(1 - Yh, Yh), u=(U_ADV, 0.0, 0.0), p=1.0, T_ion=1.0))
        return eos, ic, 1.0 / U_ADV
    elseif case === :thermal
        eos = IdealSpecies("gas"; gamma=1.4, R=1.0)
        ic = (x, y, z) -> Prim(u=(U_ADV, 0.0, 0.0), p=1.0,
                               T_ion=1.0 + (T_HOT - 1.0) * slab(x))
        return eos, ic, 1.0 / U_ADV
    else
        eos = two_gas()
        c_l = sqrt(1.4)
        ic = (x, y, z) -> begin
            Yh = slab(x)
            ρ0 = 1.0 / ((1 - Yh) + R_HEAVY * Yh)
            pp = amp * exp(-(wrap(x) / SIGMA)^2)
            Prim(Y=(1 - Yh, Yh), u=(pp / c_l, 0.0, 0.0), p=1.0 + pp,
                 rho=ρ0 + pp / c_l^2)
        end
        return eos, ic, T_ACOUSTIC
    end
end

# Transfer function of a symmetric compact filter at kh, from its interior rows.
function transfer(filt, kh)
    num = filt.a0 + sum(2 * a * cos(m * kh) for (m, a) in enumerate(filt.coeffs))
    return num / (1 + 2 * filt.alpha * cos(kh))
end

# The tables of the appendix section were measured at the defaults of the
# time, δ⁴ on every sensor and `C_D = 0.1`; `art_at` pins them, so that the
# rows labelled `default` rerun unchanged under the later `:species_d8`
# default. Keywords override either.
const MEASURED_DEFAULTS = (detector=:delta4, C_D=0.1)
art_at(; kw...) = ArtificialProperties(; merge(MEASURED_DEFAULTS, values(kw))...)

"""
    run_case(case, N; art, filt, filter_interval, filter_cfl, cfl, amp)

One run, returning the time maxima of the three normalized coefficients, the
final line of (x, Y_heavy, T, p), the mean filter weight and the step count.
"""
function run_case(case::Symbol, N; art=art_at(),
                  filt=compact_filter(0.45), filter_interval=1, filter_cfl=0.35,
                  cfl=OPTS.cfl, amp=OPTS.amp)
    eos, ic, tfin = problem(case, amp)
    h = 1.0 / N
    prob = Problem(eos=eos, transport=ConstantTransport(mu0=0.0),
                   domain=((0.0, 1.0), (0.0, h), (0.0, h)), bcs=per3, ic=ic)
    solver, Q = setup(prob, Numerics(n_global=(N, 1, 1), art=art, cfl=cfl, filt=filt,
                                     filter_interval=filter_interval,
                                     filter_cfl=filter_cfl))
    peak = zeros(3)
    wsum = Ref(0.0); nw = Ref(0)
    nx = solver.decomp.n_local[1]
    sample = (s, Q) -> begin
        hs = s.h[1]
        for i in 1:nx
            I = padded_index(s, i, 1, 1)
            ch = s.c[I] * hs
            D = maximum(D_k[I] for D_k in s.D_art)
            peak[1] = max(peak[1], D / ch)
            peak[2] = max(peak[2], s.beta_art[I] / (s.rho[I] * ch))
            peak[3] = max(peak[3], s.kappa_art[I] / (s.rho[I] * s.cp_mix[I] * ch))
        end
        if s.dt_prev > 0
            wsum[] += CL.filter_weight(s, 1); nw[] += 1
        end
    end
    run!(solver, Q; tfinal=tfin, nmax=OPTS.nmax, callback=sample)
    CL.exchange_state!(Q, solver.decomp)
    CL.primitives!(solver, Q)
    at(f) = [f[padded_index(solver, i, 1, 1)] for i in 1:nx]
    xs = Float64[xcoord(solver, 1, i) for i in 1:nx]
    Yh = length(solver.Y) > 1 ? at(solver.Y[2]) : zeros(nx)
    return (; peak, x=xs, Y=Yh, T=at(solver.T_ion), p=at(solver.p),
            w=nw[] > 0 ? wsum[] / nw[] : NaN, steps=solver.step,
            ok=completed(solver, tfin), filt, filter_interval)
end

# The field each case is judged on, and its reference at the same nodes.
function judged(case, r, ref)
    case === :composition && return r.Y, slab.(r.x)
    case === :thermal && return r.T, 1.0 .+ (T_HOT - 1.0) .* slab.(r.x)
    stride = length(ref.x) ÷ length(r.x)
    pref = ref.p[1:stride:end]
    @assert maximum(abs, ref.x[1:stride:end] .- r.x) < 1e-12
    return (r.p .- 1) ./ OPTS.amp, (pref .- 1) ./ OPTS.amp
end

function errors(case, on, off, ref)
    q_on, q_ex = judged(case, on, ref)
    q_off, _ = judged(case, off, ref)
    return l1(q_on, q_ex), l1(q_off, q_ex), l1(q_on, q_off)
end

rate(a, b) = a > 0 && b > 0 ? log2(a / b) : NaN

const CASE_LABEL = Dict(
    :composition => "composition slab, R ratio 5, uniform p T u = 1, t = 1 (judged on Y_heavy)",
    :thermal => "thermal slab, T ratio 1.5, uniform p u = 1, t = 1 (judged on T)",
    :acoustic => "pulse amp $(OPTS.amp) crossing the resting two-gas slab, t = $(T_ACOUSTIC) " *
                 "(judged on p'/amp against art off at N = $(OPTS.Nref))")

reference(case) = case === :acoustic ?
    run_case(case, OPTS.Nref; art=ArtificialProperties(enabled=false)) : nothing

function ladder(case, ref)
    println("\n=== $(CASE_LABEL[case]): ladder, W = $(OPTS.W) ===")
    println("   N  N·W |   D*/ch  beta*/rho ch  kappa*/rho cp ch |    E on     E off   on-off |" *
            " rates: D*  beta*  kappa*  on-off")
    prev = nothing
    finals = Dict{Int,Any}()
    for N in NS
        on = run_case(case, N)
        off = run_case(case, N; art=ArtificialProperties(enabled=false))
        e_on, e_off, d = errors(case, on, off, ref)
        @printf("%4d %4.0f | %9.2e %9.2e %9.2e | %8.2e %8.2e %8.2e |", N, N * OPTS.W,
                on.peak..., e_on, e_off, d)
        if prev !== nothing
            @printf("  %5.2f %5.2f %5.2f %5.2f", rate(prev[1][1], on.peak[1]),
                    rate(prev[1][2], on.peak[2]), rate(prev[1][3], on.peak[3]),
                    rate(prev[2], d))
        end
        (on.ok && off.ok) || print("  INCOMPLETE")
        println()
        prev = (on.peak, d)
        finals[N] = on
    end
    return finals
end

function variations(case, ref)
    N = OPTS.Nvar
    a = art_at
    rows = Any[("default", a(), (;)),
               ("C_D = 0", a(C_D=0.0), (;)),
               ("C_beta = 0", a(C_beta=0.0), (;)),
               ("C_kappa = 0", a(C_kappa=0.0), (;)),
               ("C_mu = 0", a(C_mu=0.0), (;)),
               ("beta :gated_strain", a(beta_sensor=:gated_strain), (;)),
               ("beta :dilatation", a(beta_sensor=:dilatation), (;)),
               ("beta :ungated_dil", a(beta_sensor=:ungated_dilatation), (;)),
               ("detector :d8", a(detector=:d8), (;)),
               ("smoother :compact", a(smoother=:compact), (;))]
    if case !== :thermal
        push!(rows, ("flux :bulk", a(species_flux=:bulk), (;)),
                    ("flux :fickian", a(species_flux=:fickian), (;)))
    end
    push!(rows, ("filter alpha 0.49", a(), (filt=compact_filter(0.49),)),
                ("filter off", a(), (filter_interval=0,)),
                ("cfl 0.25", a(), (cfl=0.25,)),
                ("cfl 0.25, filter_cfl 0", a(), (cfl=0.25, filter_cfl=0.0)))
    case === :acoustic && push!(rows, ("amp 1e-3", a(), (amp=1e-3,)))
    println("\n=== $(CASE_LABEL[case]): single-factor variations at N = $N ===")
    println("row                     | T(pi/4) T(pi/2) T(3pi/4) every     w |" *
            "   D*/ch  beta*/rho ch  kappa*/rho cp ch |    E on     E off   on-off | steps")
    offs = Dict{Any,Any}()
    for (label, art, num) in rows
        on = run_case(case, N; art=art, num...)
        off = get!(offs, num) do
            run_case(case, N; art=ArtificialProperties(enabled=false), num...)
        end
        if haskey(num, :amp)
            # The reference at the smaller amplitude, so p'/amp stays comparable.
            ref_a = run_case(case, OPTS.Nref; art=ArtificialProperties(enabled=false),
                             amp=num.amp)
            q_on = (on.p .- 1) ./ num.amp
            q_off = (off.p .- 1) ./ num.amp
            q_ref = (ref_a.p[1:(OPTS.Nref ÷ N):end] .- 1) ./ num.amp
            e_on, e_off, d = l1(q_on, q_ref), l1(q_off, q_ref), l1(q_on, q_off)
        else
            e_on, e_off, d = errors(case, on, off, ref)
        end
        tf = on.filter_interval == 0 ? (1.0, 1.0, 1.0) :
             Tuple(transfer(on.filt, k) for k in (π / 4, π / 2, 3π / 4))
        @printf("%-23s | %7.4f %7.4f %8.4f %5d %5.2f | %9.2e %9.2e %9.2e | %8.2e %8.2e %8.2e | %5d%s\n",
                label, tf..., on.filter_interval, on.w, on.peak..., e_on, e_off, d,
                on.steps, (on.ok && off.ok) ? "" : " INCOMPLETE")
    end
end

# --- the contact-localization probe ------------------------------------------
#
# A candidate gate for the D* and κ* sensors, separable from the compression
# sensing of β*: the undivided fourth difference normalized by the total
# variation over the same five points,
#
#   R_i = |δ⁴f_i| / Σ_{m=-2}^{1} |f_{i+m+1} − f_{i+m}|,
#
# which is (kh)³/4 on a resolved sine, 2 on a grid-to-grid oscillation and 3
# beside a one-cell step, and the gated sensor |δ⁴f| · min(1, R/R0). The probe
# reports the fraction of the peak ungated sensor the gate keeps, on the final
# smooth states and on two battery states that the channels exist for: the
# shocked air/SF6 interface of `shock_interface` (Y) and the Lax shock tube
# (the internal energy at its shock and contact).
function gated_retention(f, R0; periodic, skip=4)
    n = length(f)
    idx(i) = periodic ? mod1(i, n) : clamp(i, 1, n)
    rng = periodic ? (1:n) : (1 + skip):(n - skip)
    s = zeros(length(rng)); g = zeros(length(rng))
    for (j, i) in enumerate(rng)
        d4 = f[idx(i-2)] - 4f[idx(i-1)] + 6f[idx(i)] - 4f[idx(i+1)] + f[idx(i+2)]
        tv = sum(abs(f[idx(i+m+1)] - f[idx(i+m)]) for m in -2:1)
        R = abs(d4) / (tv + 1e-300)
        s[j] = abs(d4)
        g[j] = abs(d4) * min(1.0, R / R0)
    end
    return maximum(g) / maximum(s), maximum(s)
end

function probe(finals)
    R0s = (0.5, 1.0, 2.0)
    println("\n=== contact localization: peak |d4 f| kept by min(1, R/R0) ===")
    println("state                                 field   max|d4 f| | kept at R0 = " *
            join(R0s, ", "))
    function line(label, field, f; periodic)
        k = [gated_retention(f, R0; periodic)[1] for R0 in R0s]
        m = gated_retention(f, 1.0; periodic)[2]
        @printf("%-37s %-6s %10.2e | %s\n", label, field, m,
                join((@sprintf("%.3g", v) for v in k), "  "))
    end
    e_ideal(r, R) = r.p ./ (0.4 .* (r.p ./ (R .* r.T)))   # e = p/((γ−1)ρ)
    for (case, fin) in finals
        for N in sort(collect(keys(fin)))
            r = fin[N]
            Rmix = case === :thermal ? ones(length(r.x)) :
                   (1 .- r.Y) .+ R_HEAVY .* r.Y
            e = e_ideal(r, Rmix)
            case !== :thermal && line("$case N=$N", "Y", r.Y; periodic=true)
            line("$case N=$N", "e", e; periodic=true)
        end
    end
    si = shock_interface()
    line("air/SF6 shocked interface N=$(SI_N), t=$(SI_T)", "Y", si[2]; periodic=false)
    x, ρ, u, p, _ = lax()
    line("Lax N=$(LAX_N), t=$(LAX_T)", "e", p ./ (0.4 .* ρ); periodic=false)
end

# --- the split detector ------------------------------------------------------
#
# The second candidate: `:d8` on the fields of the κ* and D* channels (the
# internal energy, the mass and the mole fractions) and `:delta4` on |S|, the
# field of μ* and β*. β* is then unchanged, so whatever the spherical-origin
# ceiling owes to β* under `:delta4` is kept. The solver has one detector for
# every field, so bench/detector_split.jl redefines the dispatch of
# `detect_sum!` for a run built under `:d8`; `SPLIT[]` selects this split and
# `SPECIES_ONLY[]` the species-only one of the `species` part.
include(joinpath(@__DIR__, "detector_split.jl"))

function attempt(f, blank)
    try
        return f()
    catch err
        err isa SolverFailure || rethrow()
        return blank
    end
end

function battery_row(label, a)
    cap = 30_000
    noh(ν, cfl) = attempt((NaN, NaN)) do
        xs, ρ, _, _, ok, _ = noh_case(ν; art=a, cfl=cfl, nmax=cap)
        ok || return (Inf, Inf)
        plat, deficit, _, _ = noh_metrics(xs, ρ, ν)
        (plat / 4.0^ν, deficit)
    end
    n1 = noh(1, NOH_CFL); n2 = noh(2, NOH_CFL); n3 = noh(3, NOH_CFL)
    n3c = noh(3, 0.3)   # the documented origin ceiling under :delta4
    lx = attempt((NaN, NaN)) do
        xs, ρ, u, p, ok = lax(; art=a, nmax=cap)
        ex = [riemann_profile(x, LAX_T, 0.5, LAX_L, LAX_R, 1.4)[1] for x in xs]
        ok ? (l1(ρ, ex), contact_width(xs, ρ, 0.5, 1.3)) : (Inf, Inf)
    end
    sh = attempt(NaN) do
        xs, ρ, _, _, ok = shu_osher(; N=400, art=a, nmax=cap)
        band = so_band(xs)
        ok ? maximum(ρ[band]) - minimum(ρ[band]) : Inf
    end
    wc = attempt(NaN) do
        xs, ρ, _, _, ok = woodward(; N=400, art=a, nmax=cap)
        ok ? maximum(ρ) : Inf
    end
    si = attempt((NaN, NaN)) do
        r = shock_interface(; art=a, nmax=cap)
        r.completed ? (sum(abs, diff(r.Y_air)) - 1, r.worst_min_Y) : (Inf, Inf)
    end
    mx = attempt(NaN) do
        xs, Y, _, _, ok = species_advection(; art=a, nmax=cap)
        ok ? contact_width(xs, Y, 0.0, 1.0) : Inf
    end
    @printf("%-8s | %6.4f %+4.0f%% | %7.4f %+4.0f%% | %6.4f %+4.0f%% | %6.4f | ",
            label, n1[1], 100n1[2], n2[1], 100n2[2], n3[1], 100n3[2], n3c[1])
    @printf("%7.1e %6.4f | %6.4f | %6.4f | %6.4f %+7.4f | %7.5f\n", lx..., sh, wc,
            si..., mx)
end

function split_detector()
    println("\n=== the split detector: smooth cases at N = $(OPTS.Nvar) ===")
    println("config   case         |   D*/ch  beta*/rho ch  kappa*/rho cp ch |   on-off")
    N = OPTS.Nvar
    for case in (:composition, :thermal, :acoustic)
        ref = reference(case)
        off = run_case(case, N; art=ArtificialProperties(enabled=false))
        for (label, det, split) in (("delta4", :delta4, false), ("d8", :d8, false),
                                    ("split", :d8, true))
            SPLIT[] = split
            on = run_case(case, N; art=art_at(detector=det))
            SPLIT[] = false
            _, _, d = errors(case, on, off, ref)
            @printf("%-8s %-12s | %9.2e %9.2e %9.2e | %8.2e\n", label, case, on.peak..., d)
        end
    end
    println("\n=== the split detector on the battery ===")
    println("config   | Noh1 plat def | Noh2 plat  def | Noh3 plat def | Noh3@0.3 |" *
            " Lax L1  contact | Shu tr | WC peak | SI TV-1  min Y | mix wid")
    for (label, det, split) in (("delta4", :delta4, false), ("d8", :d8, false),
                                ("split", :d8, true))
        SPLIT[] = split
        battery_row(label, art_at(detector=det))
        SPLIT[] = false
    end
    println("  (NaN = lost positivity; Inf = still healthy at the step cap)")
end

# C_D under the split detector: `:d8` passes less of the period-four trail
# behind a shocked interface than δ⁴ does, so the species constant is re-swept on
# the two species rows of the battery and on the smooth composition slab.
function split_cd()
    println("\n=== C_D under the split detector ===")
    println("config  C_D   | SI TV-1   min Y  | mix wid | composition N=$(OPTS.Nvar): D*/ch  on-off")
    N = OPTS.Nvar
    off = run_case(:composition, N; art=ArtificialProperties(enabled=false))
    for (label, split, cds) in (("delta4", false, (0.1,)), ("split", true, (0.1, 0.3, 1.0, 3.0)))
        for cd in cds
            SPLIT[] = split
            a = ArtificialProperties(detector=split ? :d8 : :delta4, C_D=cd)
            si = attempt((NaN, NaN)) do
                r = shock_interface(; art=a, nmax=30_000)
                r.completed ? (sum(abs, diff(r.Y_air)) - 1, r.worst_min_Y) : (Inf, Inf)
            end
            mx = attempt(NaN) do
                xs, Y, _, _, ok = species_advection(; art=a, nmax=30_000)
                ok ? contact_width(xs, Y, 0.0, 1.0) : Inf
            end
            on = run_case(:composition, N; art=a)
            SPLIT[] = false
            _, _, d = errors(:composition, on, off, nothing)
            @printf("%-7s %-5.2g | %7.4f %+8.4f | %7.5f | %9.2e %9.2e\n", label, cd,
                    si..., mx, on.peak[1], d)
        end
    end
end

# --- the species-only split ----------------------------------------------------
#
# `:d8` on the mass and mole fractions and δ⁴ everywhere else, so the D* sensor
# alone changes detector. C_D is swept on the species rows: the shocked air/SF6
# interface, the same at density ratio 100 (the case the default channel exists
# to carry), the Brill slab at ratio 100, the passive advected interface, and
# the smooth composition slab at N·W = 4 and 8. The Lax row checks that a
# single-species run is the δ⁴ run bit for bit.
const SPECIES_ROW = Printf.Format("%-8s %-4.2g | %7.4f %+8.4f %3d | %7.4f %+8.4f %3d %5d |" *
                                  " %+8.4f %7.1e | %7.5f | %9.2e %9.2e | %9.2e %9.2e\n")

function species_split()
    cds = parse.(Float64, split(OPTS.cds, ','))
    cap = 30_000
    println("\n=== the species-only split ===")
    SPECIES_ONLY[] = true
    _, ρs, _, _, _ = lax(; art=ArtificialProperties(detector=:d8), nmax=cap)
    SPECIES_ONLY[] = false
    _, ρd, _, _, _ = lax(; art=art_at(), nmax=cap)
    @printf("Lax (one species): split identical to δ4: %s, max |Δρ| %.1e\n",
            ρs == ρd, maximum(abs, ρs .- ρd))
    Ncomp = (64, 128)
    offs = Dict(N => run_case(:composition, N; art=ArtificialProperties(enabled=false))
                for N in Ncomp)
    println("config   C_D  | SF6: TV-1   min Y  wid | R100: TV-1   min Y  wid steps |" *
            " slab100: min Y   |p-1| | mix wid | comp N=64 D*/ch on-off | N=128 D*/ch on-off")
    configs = Any[("delta4", false, :delta4, 0.1)]
    append!(configs, [("species", true, :d8, cd) for cd in cds])
    for (label, species, det, cd) in configs
        SPECIES_ONLY[] = species
        a = ArtificialProperties(detector=det, C_D=cd)
        si(ρh) = attempt((NaN, NaN, -1, -1)) do
            r = shock_interface(; art=a, rho_heavy=ρh, nmax=cap)
            r.completed || return (Inf, Inf, -1, r.steps)
            (sum(abs, diff(r.Y_air)) - 1, r.worst_min_Y, r.width_cells, r.steps)
        end
        s5 = si(SI_RHO_HEAVY)
        s100 = si(100.0)
        br = attempt((NaN, NaN)) do
            r = brill_slab(; art=a, nmax=cap)
            r.completed ? (r.worst_min_Y, r.p_error) : (Inf, Inf)
        end
        mx = attempt(NaN) do
            xs, Y, _, _, ok = species_advection(; art=a, nmax=cap)
            ok ? contact_width(xs, Y, 0.0, 1.0) : Inf
        end
        comp = map(Ncomp) do N
            on = run_case(:composition, N; art=a)
            _, _, d = errors(:composition, on, offs[N], nothing)
            (on.peak[1], d)
        end
        SPECIES_ONLY[] = false
        Printf.format(stdout, SPECIES_ROW, label, cd, s5[1], s5[2], s5[3], s100..., br...,
                      mx, comp[1]..., comp[2]...)
    end
    println("  (NaN = lost positivity; Inf = incomplete at the step cap)")
end

# The cost of the species-only split: `compute_artificial!` and `compute_rhs!`
# under δ⁴, under the split and under `:d8` on every field, interleaved
# repetition by repetition in one process and reported as the minimum over
# `reps`, on the two-species tube of bench/phases.jl and on a 64³ periodic box.
function species_cost()
    eos = IdealMixture([IdealSpecies{Float64}("light", 1.0, 1.4),
                        IdealSpecies{Float64}("heavy", 0.2, 1.09)])
    tube(det) = begin
        s = Solver(n_global=(512, 32, 1), L_domain=(1.0, 0.06, 1.0), eos=eos,
                   bcs=((SlipWallBC(), SlipWallBC()), per3[2], per3[3]),
                   art=ArtificialProperties(detector=det))
        Q = allocate_state(s)
        initialize!(s, Q, (x, y, z) -> begin
            θ = tanh_blend(x, 0.5, 0.02)
            Prim(Y=(1 - θ, θ), rho=(1 - θ) + 0.625θ, p=(1 - θ) + 0.1θ)
        end)
        (s, Q)
    end
    box(det) = begin
        s = Solver(n_global=(64, 64, 64), L_domain=(1.0, 1.0, 1.0), eos=eos, bcs=per3,
                   art=ArtificialProperties(detector=det))
        Q = allocate_state(s)
        initialize!(s, Q, (x, y, z) -> begin
            θ = 0.5 + 0.45 * sin(2π * x) * sin(2π * y) * cos(2π * z)
            Prim(Y=(1 - θ, θ), u=(0.1 * sin(2π * y), 0.0, 0.0), p=1.0, T_ion=1.0)
        end)
        (s, Q)
    end
    println("\n=== cost of the species-only split, $(Threads.nthreads()) threads, " *
            "min over $(OPTS.reps) interleaved repetitions ===")
    println("case            config   | artificial ms  rhs ms | ratio to delta4: art  rhs")
    for (name, build) in (("tube 512x32", tube), ("box 64^3", box))
        (sd, Qd), (s8, Q8) = build(:delta4), build(:d8)
        dQd, dQ8 = zero(Qd), zero(Q8)
        runs = (("delta4", sd, Qd, dQd, false), ("species", s8, Q8, dQ8, true),
                ("d8", s8, Q8, dQ8, false))
        best = Dict(l => [Inf, Inf] for (l, _...) in runs)
        for rep in 0:OPTS.reps, (l, s, Q, dQ, sp) in runs
            SPECIES_ONLY[] = sp
            ta = @elapsed CL.compute_rhs!(s, Q, dQ)
            tb = @elapsed CL.compute_artificial!(s, Q)
            SPECIES_ONLY[] = false
            rep == 0 && continue   # compilation
            best[l][1] = min(best[l][1], tb)
            best[l][2] = min(best[l][2], ta)
        end
        for (l, _...) in runs
            b = best[l]; r = best["delta4"]
            @printf("%-15s %-8s | %10.3f %9.3f | %22.3f %5.3f\n", name, l, 1e3b[1],
                    1e3b[2], b[1] / r[1], b[2] / r[2])
        end
    end
end

function main()
    finals = Dict{Symbol,Any}()
    for case in (:composition, :thermal, :acoustic)
        (want(String(case)) || want("probe")) || continue
        ref = reference(case)
        finals[case] = ladder(case, ref)
        want(String(case)) && variations(case, ref)
    end
    want("probe") && probe(finals)
    want("split") && split_detector()
    want("splitcd") && split_cd()
    want("species") && species_split()
    want("speciescost") && species_cost()
    println("\nfalseactivation complete")
end

main()
