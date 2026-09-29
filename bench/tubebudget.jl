# The kinetic- and internal-energy budget of the shocked He/CO2 tube, term by
# term, and the widths of its species, velocity and temperature layers.
#
# The tube is the deck of bench/he_co2_shock_tube.jl on `Nasa9Mixture` (the
# deck of the partial-density channel's two-dimensional comparison): a 4 m
# tube of square 0.2 m section, helium driven 10:1 from a diaphragm at 2 m
# into a He/CO2 contact at 3 m perturbed by one cosine mode of 10% of the span,
# slip walls at both ends, Euler fluxes plus the artificial properties, the
# compact filter every step, CFL 0.5.
#
#   julia --project=. -t 16 bench/tubebudget.jl                         # budget
#   julia --project=. -t 16 bench/tubebudget.jl widths nx=1536 ny=96    # finer
#   julia --project=. -t 16 bench/tubebudget.jl budget nx=192 ny=12 \
#       tfinal=3e-4 channels=partial_density                            # smoke
#
# Positional: part (`budget`, the attribution and the widths; `widths`, the
# widths alone, for a finer grid). Keys: nx ny tfinal; eos (`nasa9`, or
# `ideal`, the calorically perfect mixture of the Pyranda deck); channels (a
# comma list of `fickian`, `bulk`, `partial_density`); transport (`euler`,
# the deck, or `cea`, `CeaTransport` with unity Lewis number, which gives the
# molecular row a nonzero entry); snapshots (instants in ms at which the widths
# and the cumulative budget are printed); band (the half-width in metres of
# the interface band the widths are measured over, a comma list); cfl; nmax;
# progress (`ProgressLog` cadence in steps, 0 off); sharpen (the
# `ArtificialProperties.C_sharpen` of the partial-density channel, 0 off).
#
# The attribution. A callback after every accepted step replays that step from
# the state the previous step returned: the five low-storage Runge-Kutta
# stages of `step!`, each with the boundary enforcement and `compute_rhs!`
# that `step!` applies, then `filter_state!`. The replayed state is compared
# with the one `run!` holds; they agree bitwise unless something after the
# filter (the positivity failsafe) wrote the state, and that difference is the
# `repairs` row. At every stage the right-hand side is split by re-assembling
# the flux from the stage's primitives, gradients and artificial coefficients
# with one coefficient field zeroed at a time and differencing it with the
# scheme's divergence:
#
#   pressure       the flux p in the normal momentum and p u in the energy
#   convection     ρ_k u, ρ u u and E u
#   mu*, beta*,    the increments the coefficient field makes to the assembled
#   kappa*,        flux (D*: every species coefficient, which carries the
#   species        channel's consistency fluxes and the mass-fraction bound)
#   sharpening     the interface sharpening flux under `sharpen`, with its
#                  consistency fluxes: the columns of `grad_Q` that hold it
#                  are zeroed; zero when `sharpen = 0`
#   molecular      the rest of the assembled flux without the boundary hooks:
#                  zero to round-off under the Euler deck, which checks the
#                  split
#   wall flux      what `correct_flux!` changes (the slip wall zeroes the wall
#                  plane's mass, tangential momentum and energy flux)
#   wall state     what `enforce!` changes at each stage (the slip wall removes
#                  the wall node's normal momentum and its kinetic energy)
#
# The flux is linear in each coefficient field at fixed primitives, so the
# terms sum to the right-hand side to round-off. The low-storage update is
# linear in the stage right-hand sides, Q^{n+1} = Q^n + dt Σ_s b_s k_s plus the
# enforcement increments, with b_s from the 2N coefficients, so each term's
# state increment over the step is exact. Its kinetic-energy change is the
# line integral of the gradient of KE = ∫ |m|²/(2ρ) along the straight path
# from the step's initial to its final state, dotted with the increment, by
# five Gauss points in the path parameter. The terms' changes then sum to the
# step's KE change up to that quadrature (a KE gradient at the midpoint state
# alone left an error the size of the filter row, from the steps that move a
# shock by a fraction of a cell). The split of the nonlinear part among the
# terms is the straight-path convention. With the filter and the repairs
# measured as state differences, the budget closes against the measured
# change; the closure residual is printed. IE = ∫E − KE, so each term's
# internal-energy change is its total-energy change less its kinetic one.
# Integrals are per unit depth, with the trapezoidal weights at the walls.
#
# Where each term puts heat. A conservative term that carries no net mass
# flux (κ*, the Fickian channel) changes neither KE nor ∫E, so the budget
# shows it only at the walls. The last two columns split each term's sensible
# heating, δ(ρe) − Σ_k e_k δρ_k = ρ c_v δT, between the gases by their shares
# of the local heat capacity, ρ_k c_v,k/(ρ c_v), since they share one
# temperature: a transfer from the He to the CO2 is a pair of equal and
# opposite entries, a source a pair of one sign. The inviscid rows carry the
# compression heating there, split between their two rows by the
# conservative form.
#
# The widths, on the interface band: the points within `band` of a point whose
# CO2 mass fraction lies in (0.01, 0.99). With L = ∫|∇Y| dA the interface
# length (ΔY = 1), the thickness of a field φ is (∫|∇φ| dA)² / (L ∫|∇φ|² dA),
# with |∇φ| replaced by the vorticity |ω| for the velocity layer. A tanh
# profile tanh(n/w) has thickness 3w, its 10–90% width 2.2w. Widths are printed
# in cells of the x spacing, beside ∫|ω| dA / L and ∫|∇T| dA / L, the
# velocity jump across the layer and the temperature variation across it.
#
# Serial: the replay holds a second state and the term arrays. At the default
# grid the budget costs about fifteen plain runs, nine minutes per channel at
# -t 16 on the development workstation.
#
# Scratch tooling, like everything else in bench/: it prints tables and
# asserts nothing.

using MPI
MPI.Init(threadlevel=:funneled)
using CompactLES
using Printf

const CL = CompactLES

const DEFAULTS = (part = "budget", nx = 768, ny = 48, tfinal = 2.5e-3,
                  eos = "nasa9", channels = "fickian,bulk,partial_density",
                  transport = "euler", snapshots = "1.0,1.5,2.0,2.5",
                  band = "0.04,0.08", cfl = 0.5, nmax = 1_000_000, progress = 0,
                  sharpen = 0.0)
const opt = CL.script_args(ARGS, DEFAULTS; positional = (:part,))

MPI.Comm_size(MPI.COMM_WORLD) == 1 ||
    error("tubebudget.jl is serial; run it without mpiexec")
opt.part in ("budget", "widths") || error("part must be budget or widths")

const Lx = 4.0
const Lyz = 0.2
const x_diaphragm = 2.0
const x_iface = 3.0
const atm = 101325.0
const p_driver, p_driven = 10atm, 1atm
const T0 = 300.0
const A_pert = 0.10Lyz

parse_list(s) = [String(strip(x)) for x in split(String(s), ',') if !isempty(strip(x))]

function make_eos(name)
    name == "nasa9" && return Nasa9Mixture(["He", "CO2"])
    name == "ideal" || error("eos must be nasa9 or ideal")
    Ru = 8.314462618
    return IdealMixture([IdealSpecies("He"; R = 1e3 * Ru / 4.0026, gamma = 5 / 3),
                         IdealSpecies("CO2"; R = 1e3 * Ru / 44.0095, gamma = 1.289)])
end

function tube(channel)
    nx, ny = opt.nx, opt.ny
    hx = Lx / (nx - 1)
    δ = 3hx
    eos = make_eos(opt.eos)
    transport = opt.transport == "euler" ? ConstantTransport(mu0 = 0.0) :
                opt.transport == "cea" ? CeaTransport(eos) :
                error("transport must be euler or cea")
    prob = Problem(eos = eos, transport = transport,
                   domain = ((0.0, Lx), (0.0, Lyz), (0.0, Lyz)),
                   bcs = ((SlipWallBC(), SlipWallBC()),
                          (PeriodicBC(), PeriodicBC()), (PeriodicBC(), PeriodicBC())),
                   ic = (x, y, z) -> begin
                       xi = x_iface + A_pert * cos(2π * y / Lyz)
                       θ = tanh_blend(x, xi, δ)
                       p = p_driven + (p_driver - p_driven) *
                           (1 - tanh_blend(x, x_diaphragm, δ))
                       Prim(Y = (1 - θ, θ), p = p, T_ion = T0)
                   end)
    art = ArtificialProperties(enabled = true, species_flux = Symbol(channel),
                               C_sharpen = channel == "partial_density" ? opt.sharpen : 0.0)
    num = Numerics(n_global = (nx, ny, 1), art = art, cfl = opt.cfl,
                   control = StepControl(retries = 4, validity = :permissive))
    return setup(prob, num)
end

# --- integrals ----------------------------------------------------------------

"Quadrature weights over the padded array (zero on halos), per unit depth."
function weights(solver)
    o1, o2, o3 = solver.decomp.n_halo_d
    nx, ny, nz = solver.decomp.n_local
    W = zeros(size(solver.rho))
    a = solver.h[1] * solver.h[2]
    for j in 1:ny, i in 1:nx
        W[i+o1, j+o2, 1+o3] = a * CL.quad_weight(solver, 1, i) *
                              CL.quad_weight(solver, 2, j)
    end
    return W
end

struct Layout
    ns::Int
    m::NTuple{3,Int}
    ie::Int
end
layout(solver) = Layout(solver.equations.n_species, Tuple(solver.equations.i_mom),
                        solver.equations.i_energy)

function ke_ie(W, Q, L::Layout)
    ke = 0.0
    et = 0.0
    @inbounds for I in CartesianIndices(W)
        w = W[I]
        w == 0 && continue
        ρ = 0.0
        for k in 1:L.ns
            ρ += Q[I, k]
        end
        m2 = Q[I, L.m[1]]^2 + Q[I, L.m[2]]^2 + Q[I, L.m[3]]^2
        ke += w * m2 / (2ρ)
        et += w * Q[I, L.ie]
    end
    return ke, et - ke
end

# Five-point Gauss-Legendre nodes and weights on [0, 1].
const GAUSS_S = (0.5 - 0.4530899229693320, 0.5 - 0.2692346550528416, 0.5,
                 0.5 + 0.2692346550528416, 0.5 + 0.4530899229693320)
const GAUSS_W = (0.1184634425280945, 0.2393143352496832, 0.2844444444444444,
                 0.2393143352496832, 0.1184634425280945)

"""
The KE and IE change of each increment `accs[n]`, as the line integral of the
KE gradient along the straight path from `Q0` to `Q0 + ΔQ`, ΔQ = Σ accs. The
integrals sum to KE(Q0 + ΔQ) − KE(Q0) up to the quadrature error of five Gauss
points in the path parameter (KE is rational along the path).
"""
function ke_ie_increments(W, Q0, ΔQ, accs, L::Layout, ek, φ)
    nt = length(accs)
    dke = zeros(nt)
    de = zeros(nt)
    dh = zeros(nt)
    dsens = zeros(nt)
    @inbounds for I in CartesianIndices(W)
        w = W[I]
        w == 0 && continue
        ρ0 = 0.0
        dρ = 0.0
        for k in 1:L.ns
            ρ0 += Q0[I, k]
            dρ += ΔQ[I, k]
        end
        # The path averages of u and |u|²/2.
        ū1 = ū2 = ū3 = k̄ = 0.0
        for g in 1:5
            s = GAUSS_S[g]
            ρ = ρ0 + s * dρ
            u1 = (Q0[I, L.m[1]] + s * ΔQ[I, L.m[1]]) / ρ
            u2 = (Q0[I, L.m[2]] + s * ΔQ[I, L.m[2]]) / ρ
            u3 = (Q0[I, L.m[3]] + s * ΔQ[I, L.m[3]]) / ρ
            ū1 += GAUSS_W[g] * u1
            ū2 += GAUSS_W[g] * u2
            ū3 += GAUSS_W[g] * u3
            k̄ += GAUSS_W[g] * (u1^2 + u2^2 + u3^2) / 2
        end
        for n in 1:nt
            a = accs[n]
            aρ = 0.0
            for k in 1:L.ns
                aρ += a[I, k]
            end
            dk = w * (ū1 * a[I, L.m[1]] + ū2 * a[I, L.m[2]] + ū3 * a[I, L.m[3]] -
                      k̄ * aρ)
            dke[n] += dk
            de[n] += w * a[I, L.ie] - dk
            sensible = w * a[I, L.ie] - dk
            for k in 1:L.ns
                sensible -= w * ek[k][I] * a[I, k]
            end
            dh[n] += φ[I] * sensible
            dsens[n] += sensible
        end
    end
    return dke, de, dh, dsens
end

"The heating of the heavy gas from Qa to Qb (see `heavy_heating_weights!`)."
function ie_heavy_change(W, Qa, Qb, L::Layout, ek, φ)
    dh = 0.0
    dsens = 0.0
    @inbounds for I in CartesianIndices(W)
        w = W[I]
        w == 0 && continue
        ρa = 0.0
        ρb = 0.0
        for k in 1:L.ns
            ρa += Qa[I, k]
            ρb += Qb[I, k]
        end
        iea = Qa[I, L.ie] - (Qa[I, L.m[1]]^2 + Qa[I, L.m[2]]^2 + Qa[I, L.m[3]]^2) / (2ρa)
        ieb = Qb[I, L.ie] - (Qb[I, L.m[1]]^2 + Qb[I, L.m[2]]^2 + Qb[I, L.m[3]]^2) / (2ρb)
        sensible = ieb - iea
        for k in 1:L.ns
            sensible -= ek[k][I] * (Qb[I, k] - Qa[I, k])
        end
        dh += w * φ[I] * sensible
        dsens += w * sensible
    end
    return dh, dsens
end

e_species(eos::Nasa9Mixture, k, T) = CL.species_energy(eos, k, T)
cv_species(eos::Nasa9Mixture, k, T) = CL.species_cp(eos, k, T) - eos.Rk[k]
e_species(eos::IdealMixture, k, T) = eos.cvk[k] * T
cv_species(eos::IdealMixture, k, T) = eos.cvk[k]

"""
The weights of the heavy gas's heating. An increment δQ changes the internal
energy per volume by δ(ρe) = Σ_k e_k δρ_k + ρ c_v δT; the second part is the
sensible heating, and the heavy gas takes the fraction φ = ρ_h c_v,h/(ρ c_v)
of it, since the two gases share one temperature. e_k and φ are evaluated at
the temperature of the step's last stage, with the densities of the state
midway through the step.
"""
function heavy_heating_weights!(bud, solver)
    eos = solver.eos
    L = bud.L
    T = solver.T_ion
    @inbounds for I in CartesianIndices(bud.W)
        bud.W[I] == 0 && continue
        num = 0.0
        den = 0.0
        for k in 1:L.ns
            ρk = (bud.Q0[I, k] + bud.Qr[I, k]) / 2
            bud.ek[k][I] = e_species(eos, k, T[I])
            c = ρk * cv_species(eos, k, T[I])
            den += c
            k == L.ns && (num = c)
        end
        bud.φ[I] = num / den
    end
    return bud
end

# --- the split right-hand side --------------------------------------------------

const TERMS = (:pressure, :convection, :mu, :beta, :kappa, :species, :sharpening,
               :molecular, :wall_flux, :wall_state)
const LABELS = Dict(:pressure => "pressure", :convection => "convection",
                    :mu => "mu*", :beta => "beta*", :kappa => "kappa*",
                    :species => "species channel", :sharpening => "sharpening",
                    :molecular => "molecular",
                    :wall_flux => "wall flux", :wall_state => "wall state",
                    :filter => "filter", :repairs => "repairs",
                    :nonlinearity => "path quadrature")

"The divergence of `solver.flux` into dQ, with or without the wall hooks."
function flux_divergence!(dQ, solver, Q, hooks::Bool)
    decomp = solver.decomp
    if hooks
        for d in 1:3, side in 1:2
            decomp.active[d] || continue
            CL.correct_flux!(solver.bcs[d][side], solver, Q, d, side)
        end
    end
    for d in 1:3
        decomp.active[d] && CL.exchange_dim_batch!(view(solver.flux, d, :), decomp, d)
    end
    fill!(parent(dQ), 0)
    for c in 1:solver.equations.n_cons, d in 1:3
        decomp.active[d] || continue
        CL.div_subtract_along!(dQ, c, solver.flux[d, c], solver, d, 1, nothing)
    end
    return dQ
end

"The flux of the pressure (`:pressure`) or of convection (`:convection`) alone."
function partial_flux!(solver, Q, which::Symbol)
    L = layout(solver)
    u = (solver.u, solver.v, solver.w)
    p = solver.p
    ρ = solver.rho
    for d in 1:3
        solver.decomp.active[d] || continue
        ud = u[d]
        for c in 1:solver.equations.n_cons
            F = solver.flux[d, c]
            if which === :pressure
                if c == L.m[d]
                    F .= p
                elseif c == L.ie
                    F .= p .* ud
                else
                    fill!(F, 0)
                end
            else
                if c <= L.ns
                    F .= ρ .* solver.Y[c] .* ud
                elseif c == L.ie
                    @views F .= parent(Q)[:, :, :, c] .* ud
                else
                    j = findfirst(==(c), L.m)
                    F .= ρ .* ud .* u[j]
                end
            end
        end
    end
    return solver
end

mutable struct Budget
    W::Array{Float64,3}
    L::Layout
    Q0::Array{Float64,4}     # the state the previous step returned
    Qr::Array{Float64,4}
    Qb::Array{Float64,4}
    Qm::Array{Float64,4}
    dQ::Array{Float64,4}
    dv::Array{Float64,4}
    du::Array{Float64,4}
    acc::Dict{Symbol,Array{Float64,4}}
    ke::Dict{Symbol,Float64}
    ie::Dict{Symbol,Float64}
    ih::Dict{Symbol,Float64}   # the heating of the heavy gas
    is::Dict{Symbol,Float64}   # the heating of both
    ek::Vector{Array{Float64,3}}
    φ::Array{Float64,3}
    b::NTuple{5,Float64}
    t_prev::Float64
    step_prev::Int
    valid::Bool
    hook_check::Float64      # max |recomputed full RHS − compute_rhs!|
    split_check::Float64     # max |Σ terms − full RHS| relative
    repaired_steps::Int
    ke0::Float64
    ie0::Float64
end

"Weights b_s of the stage right-hand sides in the low-storage update."
function stage_weights()
    A, B = CL.RKA, CL.RKB
    return ntuple(j -> sum(B[s] * prod((A[r] for r in j+1:s); init = 1.0)
                           for s in j:5), 5)
end

function Budget(solver, Q)
    P = parent(Q)
    W = weights(solver)
    L = layout(solver)
    acc = Dict(t => zero(P) for t in TERMS)
    ke0, ie0 = ke_ie(W, P, L)
    keys_all = (TERMS..., :filter, :repairs, :nonlinearity)
    return Budget(W, L, copy(P), zero(P), zero(P), zero(P), zero(P), zero(P),
                  zero(P), acc, Dict(k => 0.0 for k in keys_all),
                  Dict(k => 0.0 for k in keys_all), Dict(k => 0.0 for k in keys_all),
                  Dict(k => 0.0 for k in keys_all),
                  [zero(W) for _ in 1:L.ns], zero(W), stage_weights(), solver.t,
                  solver.step, true, 0.0, 0.0, 0, ke0, ie0)
end

"Replay the step `run!` just accepted and add its terms to the budget."
function replay!(bud::Budget, solver, Q)
    P = parent(Q)
    if solver.step != bud.step_prev + 1
        bud.valid = false       # a rollback: the chain of states is broken
    end
    dt = solver.dt_prev
    t_run, tstage_run = solver.t, solver.tstage
    art_saved = CL.art_block(solver)
    Qr = bud.Qr
    Q0 = bud.Q0
    copyto!(Qr, Q0)
    fill!(bud.du, 0)
    foreach(a -> fill!(a, 0), values(bud.acc))
    Qs = CL.ConservedState(Qr)
    dQs = CL.ConservedState(bud.dQ)
    dvs = CL.ConservedState(bud.dv)
    arrays = (mu = solver.mu_art, beta = solver.beta_art, kappa = solver.kappa_art)
    solver.t = bud.t_prev
    for s in 1:5
        solver.tstage = bud.t_prev + CL.RKC[s] * dt
        copyto!(bud.Qb, Qr)
        CL.apply_bcs!(solver, Qs)
        bud.acc[:wall_state] .+= Qr .- bud.Qb
        CL.compute_rhs!(solver, Qs, dQs)
        w = bud.b[s] * dt
        split_stage!(bud, solver, Qs, dQs, dvs, arrays, w)
        CL._rk_update!(solver.decomp, solver.equations.n_cons, Qs, dQs,
                       CL.ConservedState(bud.du), CL.RKA[s], CL.RKB[s], dt)
    end
    solver.tstage = bud.t_prev + dt
    copyto!(bud.Qb, Qr)
    CL.apply_bcs!(solver, Qs)
    bud.acc[:wall_state] .+= Qr .- bud.Qb
    solver.t = t_run
    # The step's increment, attributed along the straight path Q0 -> Qr.
    bud.Qm .= Qr .- Q0
    ke_a, ie_a = ke_ie(bud.W, Q0, bud.L)
    ke_b, ie_b = ke_ie(bud.W, Qr, bud.L)
    heavy_heating_weights!(bud, solver)
    dks, dis, dhs, dss = ke_ie_increments(bud.W, Q0, bud.Qm, [bud.acc[t] for t in TERMS],
                                     bud.L, bud.ek, bud.φ)
    sk, si = sum(dks), sum(dis)
    for (n, t) in enumerate(TERMS)
        bud.ke[t] += dks[n]
        bud.ie[t] += dis[n]
        bud.ih[t] += dhs[n]
        bud.is[t] += dss[n]
    end
    bud.ke[:nonlinearity] += (ke_b - ke_a) - sk
    bud.ie[:nonlinearity] += (ie_b - ie_a) - si
    copyto!(bud.Qb, Qr)
    CL.filter_state!(solver, Qs)
    for (key, a, b) in ((:filter, bud.Qb, Qr), (:repairs, Qr, P))
        h, sens = ie_heavy_change(bud.W, a, b, bud.L, bud.ek, bud.φ)
        bud.ih[key] += h
        bud.is[key] += sens
    end
    ke_c, ie_c = ke_ie(bud.W, Qr, bud.L)
    bud.ke[:filter] += ke_c - ke_b
    bud.ie[:filter] += ie_c - ie_b
    ke_d, ie_d = ke_ie(bud.W, P, bud.L)
    bud.ke[:repairs] += ke_d - ke_c
    bud.ie[:repairs] += ie_d - ie_c
    if interior_differs(solver, Qr, P)
        bud.repaired_steps += 1
    end
    CL.set_art_block!(solver, art_saved)
    solver.tstage = tstage_run
    copyto!(Q0, P)
    bud.t_prev = solver.t
    bud.step_prev = solver.step
    return nothing
end

function interior_differs(solver, A, B)
    o1, o2, o3 = solver.decomp.n_halo_d
    nx, ny, nz = solver.decomp.n_local
    r = (o1+1:o1+nx, o2+1:o2+ny, o3+1:o3+nz)
    return @views A[r..., :] != B[r..., :]
end

function split_stage!(bud, solver, Qs, dQs, dvs, arrays, w)
    full = bud.dQ
    dv = bud.dv
    # The sharpening flux, where on, is held in the `grad_Q` columns past the
    # partial densities, which `compute_rhs!` filled for this stage.
    n_sp = solver.equations.n_species
    sharp = CL._sharpening(solver) ?
        [solver.grad_Q[d, n_sp + k] for d in 1:3 for k in 1:n_sp] : typeof(solver.rho)[]
    saved = (mu = copy(arrays.mu), beta = copy(arrays.beta),
             kappa = copy(arrays.kappa), D = [copy(a) for a in solver.D_art],
             S = [copy(a) for a in sharp])
    # The assembled flux with the hooks: must reproduce compute_rhs!.
    CL.assemble_fluxes!(solver, Qs)
    flux_divergence!(dvs, solver, Qs, true)
    bud.hook_check = max(bud.hook_check, maximum(abs, dv .- full))
    CL.assemble_fluxes!(solver, Qs)
    flux_divergence!(dvs, solver, Qs, false)
    nohook = copy(dv)
    bud.acc[:wall_flux] .+= w .* (full .- nohook)
    rest = copy(nohook)
    for (t, zero_it!) in ((:mu, () -> fill!(arrays.mu, 0)),
                          (:beta, () -> fill!(arrays.beta, 0)),
                          (:kappa, () -> fill!(arrays.kappa, 0)),
                          (:species, () -> foreach(a -> fill!(a, 0), solver.D_art)),
                          (:sharpening, () -> foreach(a -> fill!(a, 0), sharp)))
        zero_it!()
        CL.assemble_fluxes!(solver, Qs)
        flux_divergence!(dvs, solver, Qs, false)
        term = nohook .- dv
        bud.acc[t] .+= w .* term
        rest .-= term
        copyto!(arrays.mu, saved.mu)
        copyto!(arrays.beta, saved.beta)
        copyto!(arrays.kappa, saved.kappa)
        foreach((a, b) -> copyto!(a, b), solver.D_art, saved.D)
        foreach((a, b) -> copyto!(a, b), sharp, saved.S)
    end
    for t in (:pressure, :convection)
        partial_flux!(solver, Qs, t)
        flux_divergence!(dvs, solver, Qs, false)
        bud.acc[t] .+= w .* dv
        rest .-= dv
    end
    bud.acc[:molecular] .+= w .* rest
    scale = maximum(abs, full)
    bud.split_check = max(bud.split_check, maximum(abs, rest) / scale)
    return nothing
end

# --- widths ---------------------------------------------------------------------

function widths(solver, Q, bands)
    CL.refresh_primitives!(solver, Q)
    o1, o2, o3 = solver.decomp.n_halo_d
    nx, ny, _ = solver.decomp.n_local
    hx, hy = solver.h[1], solver.h[2]
    Y = solver.Y[2]
    fields = (Y, solver.T_ion, solver.u, solver.v)
    grads = [similar(solver.rho) for _ in 1:4, _ in 1:2]
    for (n, f) in enumerate(fields)
        CL.exchange_halos!(f, solver.decomp)
        for d in 1:2
            CL.deriv_scaled_along!(grads[n, d], f, solver, d, 1)
        end
    end
    I(i, j) = CartesianIndex(i + o1, j + o2, 1 + o3)
    inlayer = [0.01 < Y[I(i, j)] < 0.99 for i in 1:nx, j in 1:ny]
    out = NamedTuple[]
    for band in bands
        ri = ceil(Int, band / hx)
        rj = ceil(Int, band / hy)
        mask = falses(nx, ny)
        for j in 1:ny, i in 1:nx
            inlayer[i, j] || continue
            for dj in -rj:rj, di in -ri:ri
                ii = i + di
                1 <= ii <= nx || continue
                jj = mod1(j + dj, ny)
                (di * hx)^2 + (dj * hy)^2 <= band^2 && (mask[ii, jj] = true)
            end
        end
        s1 = zeros(3)
        s2 = zeros(3)
        dA = hx * hy
        for j in 1:ny, i in 1:nx
            mask[i, j] || continue
            K = I(i, j)
            gY = hypot(grads[1, 1][K], grads[1, 2][K])
            gT = hypot(grads[2, 1][K], grads[2, 2][K])
            ω = abs(grads[4, 1][K] - grads[3, 2][K])
            for (n, g) in enumerate((gY, ω, gT))
                s1[n] += g * dA
                s2[n] += g^2 * dA
            end
        end
        Lint = s1[1]
        th = [s1[n]^2 / (Lint * s2[n]) / hx for n in 1:3]
        push!(out, (band = band, L = Lint, Y = th[1], u = th[2], T = th[3],
                    ΔU = s1[2] / Lint, ΔT = s1[3] / Lint))
    end
    return out
end

function print_widths(solver, rows, channel)
    for r in rows
        @printf("  %-16s t %.2f ms  band %.2f m | L %.3f m | ", channel, 1e3 * solver.t,
                r.band, r.L)
        @printf("Y %5.2f  u %5.2f  T %5.2f cells | ΔU %6.1f m/s  ΔT %6.1f K\n",
                r.Y, r.u, r.T, r.ΔU, r.ΔT)
    end
end

function print_budget(bud, solver, Q, channel)
    ke1, ie1 = ke_ie(bud.W, parent(Q), bud.L)
    dke, die = ke1 - bud.ke0, ie1 - bud.ie0
    @printf("\n  budget of %s to t = %.3f ms, %d steps%s; J/m (per unit depth)\n",
            channel, 1e3 * solver.t, solver.step,
            bud.valid ? "" : "  [INVALID: a rollback broke the chain]")
    @printf("  %-16s %13s %13s %13s %13s %13s\n", "term", "ΔKE", "ΔIE", "ΔE",
            "He heating", "CO2 heating")
    sk, si = 0.0, 0.0
    for t in (TERMS..., :filter, :repairs)
        @printf("  %-16s %+13.5e %+13.5e %+13.5e %+13.5e %+13.5e\n", LABELS[t], bud.ke[t],
                bud.ie[t], bud.ke[t] + bud.ie[t], bud.is[t] - bud.ih[t], bud.ih[t])
        sk += bud.ke[t]
        si += bud.ie[t]
    end
    @printf("  %-16s %+13.5e %+13.5e %+13.5e\n", "sum", sk, si, sk + si)
    @printf("  %-16s %+13.5e %+13.5e %+13.5e\n", "measured", dke, die, dke + die)
    @printf("  %-16s %+13.5e %+13.5e   (KE %.2e, IE %.2e of the measured change)\n",
            "closure", dke - sk, die - si, abs(dke - sk) / abs(dke),
            abs(die - si) / abs(die))
    @printf("  closure from the path quadrature alone: KE %+.2e, IE %+.2e\n",
            bud.ke[:nonlinearity], bud.ie[:nonlinearity])
    @printf("  checks: hooked flux vs compute_rhs! %.2e; molecular remainder / |k| %.2e; ",
            bud.hook_check, bud.split_check)
    @printf("steps with repairs %d; KE0 %.5e IE0 %.5e\n", bud.repaired_steps, bud.ke0,
            bud.ie0)
end

function run_channel(channel)
    solver, Q = tube(channel)
    solver.decomp.periodic[1] && error("unexpected periodic x")
    bands = parse.(Float64, parse_list(opt.band))
    instants = [1e-3 * parse(Float64, s) for s in parse_list(opt.snapshots)]
    filter!(t -> 0 < t <= opt.tfinal, instants)
    bud = opt.part == "budget" ? Budget(solver, Q) : nothing
    cbs = Any[]
    bud === nothing ||
        push!(cbs, Callback(EveryStep(1), (s, Qs) -> (replay!(bud, s, Qs); nothing)))
    push!(cbs, Callback(AtTime(instants), (s, Qs) -> begin
        print_widths(s, widths(s, Qs, bands), channel)
        bud === nothing || print_budget(bud, s, Qs, channel)
        nothing
    end))
    opt.progress > 0 && push!(cbs, ProgressLog(every = opt.progress, tfinal = opt.tfinal))
    @printf("\n=== %s, %s, transport %s, %d x %d, h = %.3e m ===\n", channel, opt.eos,
            opt.transport, opt.nx, opt.ny, solver.h[1])
    elapsed = @elapsed run!(solver, Q; tfinal = opt.tfinal, nmax = opt.nmax,
                            callback = Tuple(cbs))
    @printf("  done: %d steps to t = %.3f ms in %.0f s\n", solver.step, 1e3 * solver.t,
            elapsed)
    flush(stdout)
end

function main()
    for ch in parse_list(opt.channels)
        ch in ("fickian", "bulk", "partial_density") || error("unknown channel $ch")
        run_channel(ch)
    end
end

main()
