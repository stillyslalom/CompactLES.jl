# The species channels at a contact that carries a temperature jump.
#
# `reference/DESIGN.md` ("The species channel") proves that the default channel
# and `:bulk` hold a uniform (u, p, T) state invariant, and `brill_slab` measures
# it. The argument needs a uniform temperature. At a contact with uniform p and u
# and a temperature jump, the partial densities diffuse across a gradient of T,
# and Σ_k R_k J_k = −D_b ∂(p/T) no longer vanishes: the channel carries a volume
# flux, and so does the Fickian channel and the artificial conductivity. No
# channel is then required to leave p unchanged, and the question becomes whether
# the discrete channel moves p and u by what its own continuous model calls for.
# Three parts:
#
#   rates       The initial right-hand side of the periodic slab below, one
#               configuration at a time, mapped onto ∂_t p and ∂_t u through the
#               exact linearization of the EOS. Rows: the scheme with the
#               artificial properties off, whose continuous pressure rate is
#               zero (convection at uniform u and p), so what it prints is the
#               discrete error; κ*, the increment from off to the artificial
#               properties with C_D = C_Y = 0; and each species channel, the
#               increment from C_D = C_Y = 0 to the production constants. Each
#               increment is printed beside its continuous model's rate,
#               evaluated with the same coefficient field and the scheme's
#               derivative in the temperature-equation form (below), so the
#               difference between the two is the discrete identity the
#               channel's flux form fails. The last column is the velocity the
#               continuous source sets up once acoustics has relaxed it,
#               U_LM = amplitude of ∫ (S_p − P')/(ρc²) dx (below).
#   drift       The slab advected for `periods` domain transits: the largest
#               spatial pressure deviation max |p − p̄|/p0 and velocity deviation
#               max |u − U| over the run, the mean pressure change at the end,
#               and (p_eq − p0)/p0, the mean pressure of the completely mixed
#               state with the same mass, momentum and energy, which is where
#               every conservative channel ends.
#   identities  The discrete product identities of the convective and the
#               partial-density channel terms, on a shocked form of the same
#               contact (nonuniform u), sampled along the run (below).
#
#   julia --project=. -t 1 bench/thermalcontact.jl
#   julia --project=. -t 1 bench/thermalcontact.jl rates N=128,256
#   julia --project=. -t 1 bench/thermalcontact.jl drift eos=ideal T_heavy=900 \
#       channels=partial_density periods=0.2                         # smoke run
#
# Positional: parts. Keys: eos (`nasa9`, the NASA-9 mixture, and `ideal`, the
# `IdealMixture` of the same species with constant heat capacities sampled at
# 298.15 K, the control that separates the temperature dependence of c_p from the
# unequal heat capacities); channels (`off`, artificial properties disabled;
# `none`, artificial properties on with C_D = C_Y = 0; `fickian`, `bulk`,
# `partial_density`); N (the grids, a comma list; the interface has a fixed
# physical width, so a pair is a refinement study); light, heavy (NASA-9 species
# names of the ambient gas and the slab); T_light, T_heavy (the slab's
# temperature, a comma list, so that equal temperatures reproduce the uniform-T
# invariant as a control); width (the 1–99% thickness of each interface, in
# units of the unit domain); U, p0 (advection speed in m/s and pressure in Pa);
# periods, cfl, nmax (the drift runs); Ni, mach, tfin_i, sample (the identities
# part: grid, shock Mach number in the light gas, end time in s and sampling
# interval in steps); progress (`ProgressLog` cadence in steps, 0 off).
#
# The slab. A layer of the heavy gas at T_heavy occupies the middle half of a
# periodic unit domain of the light gas at T_light, with uniform p0 and velocity
# U. The heavy gas's volume fraction V is a tanh profile, X_heavy = V, and
# T = (1 − V) T_light + V T_heavy, so ρ_k = X_k p0/(R_k T) is at uniform p. Under
# He/CO2 the two gases have unequal heat capacities (γ = 5/3 against about 1.29
# at 300 K), and under NASA-9 the CO2 heat capacity also rises by about 40%
# between 300 and 900 K. Physical transport is off.
#
# The continuous rates. In the frame moving with the uniform velocity, write
# r_k = ∂_t ρ_k and q_T = ρ c_v ∂_t T; then
#     ∂_t p = T Σ_k R_k r_k + (Σ_k ρ_k R_k) q_T/(ρ c_v),
# and for each term, with J_k its species flux and ∂ the derivative,
#     partial density   J_k = −D_b ∂ρ_k,  q_T = −Σ_k c_v,k J_k ∂T
#     bulk              the same plus q_T += ∂(D_b ρ c_v ∂T)
#     Fickian           J_k = −ρD_k ∂Y_k + ρY_k Σ_j D_j ∂Y_j,
#                       q_T = −T Σ_k R_k ∂J_k − ∂T Σ_k c_p,k J_k
#     κ*                J_k = 0, q_T = ∂(κ* ∂T)
# with r_k = −∂J_k throughout. These follow from the flux forms by the product
# rule and e_k' = c_v,k; the scheme differences the flux forms themselves, so the
# discrete and the continuous rate agree to the truncation error of the discrete
# product rule, which is what the `rel diff` column measures. At uniform T every
# q_T above but the Fickian vanishes and the partial-density and bulk ∂_t p
# reduce to T ∂(D_b ∂(Σ_k R_k ρ_k)) = 0: the invariant. With a temperature jump
# S_p = ∂_t p is nonzero pointwise for every row. In the low-Mach limit the
# pressure stays uniform, P(t), and the source instead sets a velocity
# divergence ∂u = (S_p − P')/(ρc²), with P' the value that makes the integral of
# ∂u over the periodic domain vanish; U_LM is half the range of the velocity
# that divergence integrates to. That velocity, and a pressure disturbance of
# order ρcU_LM radiated while the compressible system sets it up, are the drift
# the continuous model calls for.
#
# The identities. At uniform u every convective flux and every partial-density
# flux is u times a linear function of the conserved components, and the
# derivative is linear, so ∂_t u = 0 holds exactly; the `rates` part prints
# max |∂_t u| as the check. What remains are the identities that need the
# product rule. With F = Σ_k J_k and D the scheme's derivative, the `identities`
# part integrates over the domain
#     rK_ch  = −u D(uF) + (u²/2) D(F) + D(F u²/2)    channel kinetic energy
#     rK_sp  = the same with D(uF) replaced by the split form
#              [D(uF) + u DF + F Du]/2
#     rT_ch  = −D(Σ e_k J_k) + Σ e_k D(J_k) + Σ c_v,k J_k DT   channel heating
#     rK_cv  = −u D(ρu²) + (u²/2) D(ρu) + D(ρu³/2)   convective kinetic energy
#     rP     = −D(pu) + u Dp + p Du                  pressure work
# each the discrete rate a derived equation receives minus what its continuous
# counterpart says, and zero in the continuous limit. Total energy is
# conservative, so a kinetic-energy residual is an exchange with the internal
# energy. Each is printed twice: its integral, and its L1 norm over the L1
# norm of the term it corrupts (u D(uF) − (u²/2) DF, Σ c_v,k J_k DT,
# u D(ρu²) − (u²/2) D(ρu) and p Du), which is the local relative error. The
# integrals are compared with the artificial viscosity's dissipation
# Φ = ∫ (4μ*/3 + β*)(Du)² dx and the pressure dilatation Π = ∫ p Du dx. Where D
# is skew-symmetric (the periodic operator; here the interior, the fields being
# uniform beside the two ends) ∫ u Dv = −∫ v Du, so the integrals of rT_ch
# (under constant c_v,k), rP and rK_sp vanish to round-off, and those columns
# check the implementation rather than measure anything.
#
# Serial only: a one-dimensional case, run at -t 1 (each threaded region is far
# below the threading threshold). The rates part costs seconds; the drift part at
# the defaults is 40 runs, about three minutes, most of it the 256-point runs.
#
# Scratch tooling, like everything else in bench/: it prints tables and asserts
# nothing.

using MPI
MPI.Init(threadlevel=:funneled)
using CompactLES
using CompactLES: padded_index
using Printf

const CL = CompactLES
const PER = ntuple(_ -> (PeriodicBC(), PeriodicBC()), 3)

const DEFAULTS = (parts = "rates,drift,identities", eos = "nasa9,ideal",
                  channels = "off,none,fickian,bulk,partial_density",
                  N = "128,256", light = "He", heavy = "CO2",
                  T_light = 300.0, T_heavy = "300,900", width = 0.05,
                  U = 100.0, p0 = 1e5, periods = 1.0, cfl = 0.4,
                  nmax = typemax(Int), Ni = 400, mach = 1.5, tfin_i = 4e-4,
                  sample = 40, progress = 0)

const opt = CL.script_args(ARGS, DEFAULTS; positional = (:parts,))

MPI.Comm_size(MPI.COMM_WORLD) == 1 ||
    error("thermalcontact.jl is serial; run it without mpiexec")

const CHANNELS = ("off", "none", "fickian", "bulk", "partial_density")
const SPECIES_CHANNELS = (:fickian, :bulk, :partial_density)

function parse_list(spec, key, allowed=nothing)
    out = String[String(strip(s)) for s in split(String(spec), ',')
                 if !isempty(strip(s))]
    isempty(out) && error("$key list must not be empty")
    if allowed !== nothing
        for s in out
            s in allowed || error("$key entry '$s' is not one of " *
                                  join(allowed, ", "))
        end
    end
    return out
end

parse_floats(spec, key) = parse.(Float64, parse_list(spec, key))
parse_ints(spec, key) = parse.(Int, parse_list(spec, key))

# --- thermodynamics ---------------------------------------------------------

make_eos(name) = name == "nasa9" ? Nasa9Mixture([opt.light, opt.heavy]) :
    IdealMixture([opt.light, opt.heavy])

e_species(eos::Nasa9Mixture, k, T) = CL.species_energy(eos, k, T)
cv_species(eos::Nasa9Mixture, k, T) = CL.species_cp(eos, k, T) - eos.Rk[k]
e_species(eos::IdealMixture, k, T) = eos.cvk[k] * T
cv_species(eos::IdealMixture, k, T) = eos.cvk[k]

function art_for(channel)
    channel == "off" && return ArtificialProperties(enabled = false)
    channel == "none" && return ArtificialProperties(C_D = 0.0, C_Y = 0.0)
    return ArtificialProperties(species_flux = Symbol(channel))
end

# --- the slab ---------------------------------------------------------------

function slab_setup(eos, N, T_heavy, art)
    h = 1.0 / N
    w = 3 * opt.width / 16
    Rl, Rh = eos.Rk[1], eos.Rk[2]
    Tl = opt.T_light
    prob = Problem(eos = eos, transport = ConstantTransport(mu0 = 0.0),
                   domain = ((0.0, 1.0), (0.0, h), (0.0, h)), bcs = PER,
                   ic = (x, y, z) -> begin
                       V = (1 - tanh((abs(x - 0.5) - 0.25) / w)) / 2
                       T = (1 - V) * Tl + V * T_heavy
                       ρl = (1 - V) * opt.p0 / (Rl * T)
                       ρh = V * opt.p0 / (Rh * T)
                       ρ = ρl + ρh
                       Prim(Y = (ρl / ρ, ρh / ρ), rho = ρ,
                            u = (opt.U, 0.0, 0.0), p = opt.p0)
                   end)
    return setup(prob, Numerics(n_global = (N, 1, 1), art = art, cfl = opt.cfl,
                                filt = compact_filter(0.45), filter_interval = 1,
                                filter_cfl = 0.35,
                                control = StepControl(retries = 4)))
end

# --- line helpers -----------------------------------------------------------

line(solver, f) = [f[padded_index(solver, i, 1, 1)]
                   for i in 1:solver.decomp.n_local[1]]
line(solver, Q, c) = [Q[padded_index(solver, i, 1, 1), c]
                      for i in 1:solver.decomp.n_local[1]]

"The scheme's derivative along x of a line of interior values."
function ddx(solver, v::Vector{Float64})
    buf = solver.tmp_a
    out = solver.tmp_b
    for i in eachindex(v)
        buf[padded_index(solver, i, 1, 1)] = v[i]
    end
    CL.exchange_halos!(buf, solver.decomp)
    CL.deriv_scaled_along!(out, buf, solver, 1, 1)
    return line(solver, out)
end

"Primitive lines of the state the solver's primitives currently describe."
function state_lines(solver, Q)
    eos = solver.eos
    ns = solver.equations.n_species
    ρk = [line(solver, Q, k) for k in 1:ns]
    ρ = sum(ρk)
    T = line(solver, solver.T_ion)
    ek = [e_species.(Ref(eos), k, T) for k in 1:ns]
    cvk = [cv_species.(Ref(eos), k, T) for k in 1:ns]
    cv = sum(ρk[k] .* cvk[k] for k in 1:ns) ./ ρ
    return (ρk = ρk, ρ = ρ, T = T, u = line(solver, solver.u),
            p = line(solver, solver.p), c = line(solver, solver.c),
            ek = ek, cvk = cvk, cv = cv, Rk = collect(eos.Rk), ns = ns)
end

"∂_t p and ∂_t u of a right-hand side increment ΔQ, by the EOS linearization."
function pressure_rate(solver, s, ΔQ)
    m1 = solver.equations.i_mom[1]
    ie = solver.equations.i_energy
    dρk = [line(solver, ΔQ, k) for k in 1:s.ns]
    dρ = sum(dρk)
    dm = line(solver, ΔQ, m1)
    dE = line(solver, ΔQ, ie)
    dρe = dE .- s.u .* dm .+ 0.5 .* s.u .^ 2 .* dρ
    dT = (dρe .- sum(s.ek[k] .* dρk[k] for k in 1:s.ns)) ./ (s.ρ .* s.cv)
    ρR = sum(s.ρk[k] .* s.Rk[k] for k in 1:s.ns)
    dp = sum(s.Rk[k] .* s.T .* dρk[k] for k in 1:s.ns) .+ ρR .* dT
    du = (dm .- s.u .* dρ) ./ s.ρ
    return dp, du
end

"∂_t p of the continuous model of one term, in the temperature-equation form."
function continuous_rate(solver, s, term, coef)
    ns = s.ns
    dT = ddx(solver, s.T)
    ρR = sum(s.ρk[k] .* s.Rk[k] for k in 1:ns)
    if term === :kappa
        qT = ddx(solver, coef[1] .* dT)
        return ρR .* qT ./ (s.ρ .* s.cv)
    end
    if term === :fickian
        Yk = [s.ρk[k] ./ s.ρ for k in 1:ns]
        dY = [ddx(solver, Yk[k]) for k in 1:ns]
        Vc = sum(coef[j] .* dY[j] for j in 1:ns)
        Jk = [s.ρ .* (-coef[k] .* dY[k] .+ Yk[k] .* Vc) for k in 1:ns]
    else
        Jk = [-coef[1] .* ddx(solver, s.ρk[k]) for k in 1:ns]
    end
    dJk = [ddx(solver, Jk[k]) for k in 1:ns]
    rk = [-dJk[k] for k in 1:ns]
    if term === :fickian
        cpk = [s.cvk[k] .+ s.Rk[k] for k in 1:ns]
        qT = -s.T .* sum(s.Rk[k] .* dJk[k] for k in 1:ns) .-
             dT .* sum(cpk[k] .* Jk[k] for k in 1:ns)
    else
        qT = -dT .* sum(s.cvk[k] .* Jk[k] for k in 1:ns)
        term === :bulk && (qT = qT .+ ddx(solver, coef[1] .* s.ρ .* s.cv .* dT))
    end
    return s.T .* sum(s.Rk[k] .* rk[k] for k in 1:ns) .+ ρR .* qT ./ (s.ρ .* s.cv)
end

"Half the range of the low-Mach velocity a pressure source S_p sets up."
function low_mach_velocity(solver, s, Sp)
    h = solver.h[1]
    w = 1 ./ (s.ρ .* s.c .^ 2)
    Pdot = sum(Sp .* w) / sum(w)
    Ucum = cumsum((Sp .- Pdot) .* w) .* h
    return (maximum(Ucum) - minimum(Ucum)) / 2
end

rhs(solver, Q) = (dQ = similar(Q); CL.compute_rhs!(solver, Q, dQ); dQ)

# --- part 1: the initial rates ----------------------------------------------

function part_rates()
    println("\n=== the initial pressure and velocity rates ===")
    println("tau = h/(U + c_light); rates are p-changes per tau over p0 and " *
            "u-changes per tau in m/s")
    for eosname in parse_list(opt.eos, "eos", ("nasa9", "ideal"))
        eos = make_eos(eosname)
        for Th in parse_floats(opt.T_heavy, "T_heavy"), N in parse_ints(opt.N, "N")
            off, Qoff = slab_setup(eos, N, Th, art_for("off"))
            dQoff = rhs(off, Qoff)
            s = state_lines(off, Qoff)
            τ = off.h[1] / (opt.U + s.c[1])
            scale = τ / opt.p0
            @printf("\n--- %s, %s/%s, T %g/%g K, N = %d\n", eosname, opt.light,
                    opt.heavy, opt.T_light, Th, N)
            println("term             max|dp|disc  max|dp|cont  rel diff    " *
                    "mean dp disc  mean dp cont  max|du|      U_LM (m/s)")
            dp, du = pressure_rate(off, s, dQoff)
            @printf("%-16s %11.3e  %11.3e  %9s  %+12.3e  %+12.3e  %10.3e  %10s\n",
                    "convection", maximum(abs, dp) * scale, 0.0, "-",
                    sum(dp) / N * scale, 0.0, maximum(abs, du) * τ, "-")
            none_pd = nothing
            for ch in SPECIES_CHANNELS
                none, Qn = slab_setup(eos, N, Th, ArtificialProperties(
                    C_D = 0.0, C_Y = 0.0, species_flux = ch))
                dQn = rhs(none, Qn)
                if ch === :partial_density
                    # κ*: the increment from off to the artificial properties
                    # with the species constants zero.
                    dpk, duk = pressure_rate(none, s, dQn .- dQoff)
                    cont = continuous_rate(none, s, :kappa,
                                           (line(none, none.kappa_art),))
                    report_row("kappa*", dpk, duk, cont, scale, τ, N,
                               low_mach_velocity(none, s, cont))
                end
                full, Qf = slab_setup(eos, N, Th, art_for(String(ch)))
                dQf = rhs(full, Qf)
                dpc, duc = pressure_rate(full, s, dQf .- dQn)
                coef = ch === :fickian ?
                       Tuple(line(full, full.D_art[k]) for k in 1:s.ns) :
                       (line(full, full.D_art[1]),)
                cont = continuous_rate(full, s, ch, coef)
                report_row(String(ch), dpc, duc, cont, scale, τ, N,
                           low_mach_velocity(full, s, cont))
            end
        end
    end
end

function report_row(name, dp, du, cont, scale, τ, N, U_LM)
    nc = sqrt(sum(abs2, cont))
    rel = nc > 0 ? sqrt(sum(abs2, dp .- cont)) / nc : NaN
    @printf("%-16s %11.3e  %11.3e  %9.2e  %+12.3e  %+12.3e  %10.3e  %10.3e\n",
            name, maximum(abs, dp) * scale, maximum(abs, cont) * scale, rel,
            sum(dp) / N * scale, sum(cont) / N * scale, maximum(abs, du) * τ,
            U_LM)
end

# --- part 2: the advected slab ----------------------------------------------

"Mean pressure of the completely mixed state with the solver's integrals."
function mixed_pressure(solver, Q)
    eos = solver.eos
    ns = solver.equations.n_species
    m1 = solver.equations.i_mom[1]
    ie = solver.equations.i_energy
    N = solver.decomp.n_local[1]
    ρk = [sum(line(solver, Q, k)) / N for k in 1:ns]
    ρ = sum(ρk)
    m = sum(line(solver, Q, m1)) / N
    ρe = sum(line(solver, Q, ie)) / N - 0.5 * m^2 / ρ
    T = sum(line(solver, solver.T_ion)) / N
    for _ in 1:50
        f = sum(ρk[k] * e_species(eos, k, T) for k in 1:ns) - ρe
        T -= f / sum(ρk[k] * cv_species(eos, k, T) for k in 1:ns)
    end
    return sum(ρk[k] * eos.Rk[k] for k in 1:ns) * T
end

function drift_run(eos, N, Th, channel)
    solver, Q = slab_setup(eos, N, Th, art_for(channel))
    CL.refresh_primitives!(solver, Q)
    p_eq = mixed_pressure(solver, Q)
    pdev = Ref(0.0)
    udev = Ref(0.0)
    pmean = Ref(opt.p0)
    record = Callback(EveryStep(1), (s, Qs) -> begin
        CL.refresh_primitives!(s, Qs)
        p = line(s, s.p)
        p̄ = sum(p) / length(p)
        pdev[] = max(pdev[], maximum(x -> abs(x - p̄), p) / opt.p0)
        udev[] = max(udev[], maximum(x -> abs(x - opt.U), line(s, s.u)))
        pmean[] = p̄
        nothing
    end)
    tfin = opt.periods / opt.U
    callbacks = opt.progress > 0 ?
                (record, ProgressLog(every = opt.progress, tfinal = tfin)) :
                (record,)
    elapsed = @elapsed begin
        try
            run!(solver, Q; tfinal = tfin, nmax = opt.nmax, callback = callbacks)
        catch err
            err isa SolverFailure || rethrow()
        end
    end
    done = isfinite(solver.t) && solver.t >= tfin * (1 - 1e-9)
    return (pdev = pdev[], udev = udev[], dmean = (pmean[] - opt.p0) / opt.p0,
            eq = (p_eq - opt.p0) / opt.p0, steps = solver.step, done = done,
            elapsed = elapsed)
end

function part_drift()
    println("\n=== the advected slab: drift over $(opt.periods) period(s) ===")
    channels = parse_list(opt.channels, "channels", CHANNELS)
    for eosname in parse_list(opt.eos, "eos", ("nasa9", "ideal"))
        eos = make_eos(eosname)
        for Th in parse_floats(opt.T_heavy, "T_heavy")
            @printf("\n--- %s, %s/%s, T %g/%g K, U = %g m/s\n", eosname,
                    opt.light, opt.heavy, opt.T_light, Th, opt.U)
            println("channel            N  max|p-pbar|/p0  max|u-U| m/s  " *
                    "(pbar-p0)/p0  (p_eq-p0)/p0  steps   s")
            for N in parse_ints(opt.N, "N"), ch in channels
                r = drift_run(eos, N, Th, ch)
                @printf("%-16s %4d  %14.3e  %12.3e  %+12.3e  %+12.3e  %5d%s %5.1f\n",
                        ch, N, r.pdev, r.udev, r.dmean, r.eq, r.steps,
                        r.done ? " " : "*", r.elapsed)
            end
        end
    end
    println("* = did not reach the end time")
end

# --- part 3: the discrete identities on a shocked contact -------------------

function shocked_setup(eos, N, Th)
    γ = 5 / 3                     # the light gas, He, is monatomic
    Rl, Rh = eos.Rk[1], eos.Rk[2]
    Tl = opt.T_light
    M = opt.mach
    ρ1 = opt.p0 / (Rl * Tl)
    c1 = sqrt(γ * Rl * Tl)
    p2 = opt.p0 * (1 + 2γ / (γ + 1) * (M^2 - 1))
    r2 = (γ + 1) * M^2 / ((γ - 1) * M^2 + 2)
    u2 = M * c1 * (1 - 1 / r2)
    ρh = opt.p0 / (Rh * Th)
    h = 1.0 / (N - 1)
    δ = 2h
    blend(x, x0) = (1 + tanh((x - x0) / δ)) / 2
    bcs = (DirichletBC((x, y, z, t) -> Prim(Y = (1.0, 0.0), rho = r2 * ρ1,
                                            u = (u2, 0.0, 0.0), p = p2)),
           DirichletBC((x, y, z, t) -> Prim(Y = (0.0, 1.0), rho = ρh,
                                            u = (0.0, 0.0, 0.0), p = opt.p0)))
    prob = Problem(eos = eos, transport = ConstantTransport(mu0 = 0.0),
                   domain = ((0.0, 1.0), (0.0, h), (0.0, h)),
                   bcs = (bcs, PER[2], PER[3]),
                   ic = (x, y, z) -> begin
                       s = blend(x, 0.15)          # 0 post-shock, 1 ahead
                       V = blend(x, 0.35)          # heavy volume fraction
                       T = (1 - V) * Tl + V * Th
                       ρl = (1 - V) * opt.p0 / (Rl * T)
                       ρhv = V * opt.p0 / (Rh * T)
                       ρa = ρl + ρhv
                       ρ = (1 - s) * r2 * ρ1 + s * ρa
                       Yh = s * ρhv / ρ
                       Prim(Y = (1 - Yh, Yh), rho = ρ,
                            u = ((1 - s) * u2, 0.0, 0.0),
                            p = (1 - s) * p2 + s * opt.p0)
                   end)
    return setup(prob, Numerics(n_global = (N, 1, 1),
                                art = ArtificialProperties(), cfl = opt.cfl,
                                filt = compact_filter(0.45), filter_interval = 1,
                                filter_cfl = 0.35,
                                control = StepControl(retries = 4)))
end

const ID_ROW = Printf.Format("%5d %7.1f  %+.2e  %.2e  %+.2e  %+.2e  %.2e  " *
                             "%+.2e  %.2e  %+.2e  %.2e  %.2e  %+.2e\n")

"One sample of the identity residuals on the state `Q` the run holds."
function identity_sample(solver, Q, dQ)
    saved = CL.art_block(solver)
    tstage = solver.tstage
    CL.compute_rhs!(solver, Q, dQ)      # coefficients and primitives of Q
    s = state_lines(solver, Q)
    D_b = line(solver, solver.D_art[1])
    μ = line(solver, solver.mu_art)
    β = line(solver, solver.beta_art)
    CL.set_art_block!(solver, saved)
    solver.tstage = tstage
    D(v) = ddx(solver, v)
    u, p, ρ = s.u, s.p, s.ρ
    ux = D(u)
    Jk = [-D_b .* D(s.ρk[k]) for k in 1:s.ns]
    F = sum(Jk)
    DF = D(F)
    DuF = D(u .* F)
    DKF = D(0.5 .* u .^ 2 .* F)
    rK_ch = -u .* DuF .+ 0.5 .* u .^ 2 .* DF .+ DKF
    rK_sp = -u .* 0.5 .* (DuF .+ u .* DF .+ F .* ux) .+ 0.5 .* u .^ 2 .* DF .+ DKF
    dT = D(s.T)
    heat = sum(s.cvk[k] .* Jk[k] for k in 1:s.ns) .* dT
    rT_ch = -D(sum(s.ek[k] .* Jk[k] for k in 1:s.ns)) .+
            sum(s.ek[k] .* D(Jk[k]) for k in 1:s.ns) .+ heat
    m = ρ .* u
    rK_cv = -u .* D(m .* u) .+ 0.5 .* u .^ 2 .* D(m) .+ D(0.5 .* m .* u .^ 2)
    rP = -D(p .* u) .+ u .* D(p) .+ p .* ux
    h = solver.h[1]
    # The two end rows hold Dirichlet data and the fields are uniform beside
    # them; the integrals run over the interior points.
    ∫(v) = h * sum(@view v[3:end-2])
    return (step = solver.step, t = solver.t,
            rK_ch = ∫(rK_ch), rK_ch_abs = ∫(abs.(rK_ch)), rK_sp = ∫(rK_sp),
            rT_ch = ∫(rT_ch), rT_ch_abs = ∫(abs.(rT_ch)), heat = ∫(abs.(heat)),
            ke_ch = ∫(abs.(u .* DuF .- 0.5 .* u .^ 2 .* DF)),
            rK_cv = ∫(rK_cv), rK_cv_abs = ∫(abs.(rK_cv)),
            ke_cv = ∫(abs.(u .* D(m .* u) .- 0.5 .* u .^ 2 .* D(m))),
            rP = ∫(rP), rP_abs = ∫(abs.(rP)), pw = ∫(abs.(p .* ux)),
            Φ = ∫((4 / 3 .* μ .+ β) .* ux .^ 2), Π = ∫(p .* ux))
end

function part_identities()
    println("\n=== discrete identities on the shocked contact ===")
    for eosname in parse_list(opt.eos, "eos", ("nasa9", "ideal"))
        eos = make_eos(eosname)
        Th = maximum(parse_floats(opt.T_heavy, "T_heavy"))
        solver, Q = shocked_setup(eos, opt.Ni, Th)
        dQ = similar(Q)
        rows = NamedTuple[]
        record = Callback(EveryStep(opt.sample), (s, Qs) -> begin
            push!(rows, identity_sample(s, Qs, dQ))
            nothing
        end)
        callbacks = opt.progress > 0 ?
                    (record, ProgressLog(every = opt.progress,
                                         tfinal = opt.tfin_i)) : (record,)
        run!(solver, Q; tfinal = opt.tfin_i, nmax = opt.nmax,
             callback = callbacks)
        @printf("\n--- %s, Mach %g in %s at %g K into %s at %g K, N = %d, %d steps\n",
                eosname, opt.mach, opt.light, opt.T_light, opt.heavy, Th,
                opt.Ni, solver.step)
        println("integrals in W/m^2 (per unit cross-section)")
        println("relative columns: a residual's L1 norm over the L1 norm of " *
                "the term it corrupts")
        println(" step   t (us)    ∫rK_ch   rel|rK_ch|  ∫rK_sp     ∫rT_ch  " *
                " rel|rT_ch|  ∫rK_cv   rel|rK_cv|   ∫rP      rel|rP|     Φ" *
                "          Π")
        for r in rows
            print(Printf.format(ID_ROW, r.step, 1e6 * r.t, r.rK_ch,
                                r.rK_ch_abs / r.ke_ch, r.rK_sp, r.rT_ch,
                                r.rT_ch_abs / r.heat, r.rK_cv,
                                r.rK_cv_abs / r.ke_cv, r.rP, r.rP_abs / r.pw,
                                r.Φ, r.Π))
        end
    end
end

function main()
    parts = parse_list(opt.parts, "parts", ("rates", "drift", "identities"))
    "rates" in parts && part_rates()
    "drift" in parts && part_drift()
    "identities" in parts && part_identities()
    return nothing
end

main()
