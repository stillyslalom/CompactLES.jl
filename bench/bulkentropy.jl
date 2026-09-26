# Do the species channels' entropy inequalities survive the discretization?
#
# `reference/DESIGN.md` derives the inequality for the continuous model: for a
# convex entropy pair (η, ψ) of the hyperbolic system, adding the flux
# F_q = −D_b ∇q to every conserved variable gives
# ∂_t η + ∇·(ψ − D_b ∇η) = −D_b ∇qᵀ η''(q) ∇q ≤ 0 pointwise for D_b ≥ 0. Nothing
# in that argument is discrete. The compact derivative is not summation-by-parts
# against the trapezoid quadrature that `volume_integral` applies, the compact
# filter and the Runge–Kutta update are separate operators applied in sequence,
# and a semi-discrete flux divergence need not produce entropy pointwise. So the
# continuous property says nothing yet about what the complete update does, and
# the three parts below measure it with the scheme's own operators.
#
# Entropy here is ρs = Σ_k ρ_k (c_v,k ln T − R_k ln ρ_k) with the per-species
# constants dropped, for an `IdealMixture` whose species carry constant
# c_v,k = R_k/(γ_k − 1). Signs follow η = −ρs, so a positive number is entropy
# produced and the continuous model's channel contribution is nonnegative.
#
# Both flow cases are inviscid (`ConstantTransport(mu0 = 0)`), so no physical
# dissipation enters any number reported: what is measured is the scheme's own
# entropy budget, split into the artificial species channel, the rest of the
# right-hand side, and the filter.
#
#   julia --project=. bench/bulkentropy.jl [N] [tfin] [key=value ...]
#   julia --project=. bench/bulkentropy.jl parts=variables
#   julia --project=. bench/bulkentropy.jl 32 2.0 parts=production
#   julia --project=. bench/bulkentropy.jl 32 2.0 parts=step
#
# Parts, comma-separated in `parts=` (default all three):
#
#   variables   The closed-form entropy variables w = ∂(ρs)/∂q, and the
#               chain-rule form of ∂_d qᵀ (ρs)''(q) ∂_d q that avoids
#               assembling the Hessian, each checked against central
#               differences at random admissible states. The two closed forms
#               feed the other parts, so nothing else here is trustworthy
#               until this prints a small discrepancy.
#
#   production  The semi-discrete entropy production of the species channel
#               alone. Two solvers differ only in `C_D` and `C_Y`, so the
#               difference R = dQ_full − dQ_off of their right-hand sides on
#               the same state is the channel's contribution; the run prints
#               the max difference of μ*, β* and κ* between the two and the max
#               D* of the off solver beside every sample, which is the check
#               that the isolation holds. Reported per sample: the channel's
#               production ∫ w·R dV, the whole right-hand side's ∫ w·dQ dV so
#               the channel's share is visible, and the continuous model's
#               production evaluated with the discrete conserved gradients the
#               solver already holds in `grad_Q`: under `:bulk` the quadratic
#               form ∫ D_b Σ_d ∂_d qᵀ (−(ρs)'') ∂_d q dV, under
#               `:partial_density` ∫ D_b Σ_d Σ_k R_k (∂_d ρ_k)²/ρ_k dV. The
#               difference between that form and ∫ w·R dV is the discrete
#               non-summation-by-parts defect, which is the number this part exists for. Both
#               integrals exclude the points where a partial density has gone
#               nonpositive, since the entropy Hessian carries R_k/ρ_k and is
#               not defined there, and the count of excluded points is printed
#               beside them: at a resolution that leaves many of them the
#               defect is a statement about a fraction of the domain.
#
#   step        The fully discrete update: S = ∫ρs dV recorded after every
#               Runge–Kutta step and again after the filter pass, for the
#               partial-density, bulk and Fickian channels and the artificial
#               properties off.
#               The off configuration is what makes the filter's own
#               contribution readable. S is the integral over the whole domain,
#               so a point outside the entropy's domain enters it at a floored
#               logarithm; those point-visits are counted in the last column,
#               and a large count means S itself rests on the floor.
#
# Options:
#   N         cubic grid of the three-dimensional case (default 32)
#   tfin      end time of the three-dimensional case (default 2.0)
#   nmax      step cap, applied to every run here (default none). A sweep that
#             may visit a configuration that loses positivity must set one; see
#             CLAUDE.md, Conventions.
#   ratio     density ratio of the interface: the heavy gas takes R = 1/ratio
#             and γ = 1.09 against air's R = 1, γ = 1.4 (default 5.04, SF6)
#   cfl       timestep multiplier of the three-dimensional case (default 0.5)
#   sample    steps between `production` samples (default 10)
#   states    random states the `variables` checks use (default 8)
#   seed      seed for those states (default 20260919)
#   slab      also run the one-dimensional `brill_slab` under `step`
#             (default true). It is the case the invariance claim of
#             `reference/DESIGN.md` rests on and it is cheap.
#   progress  ProgressLog interval in steps, 0 (default) to disable
#
# Scratch tooling, like everything else in bench/: it prints tables, asserts
# nothing, and is not part of the gate.
#
# Runs under mpiexec unchanged. Every diagnostic here is a `volume_integral`,
# `filter_state!` or `compute_rhs!`, all collective, and every rank enters
# every one of them.

using MPI
MPI.Init(threadlevel=:funneled)
using CompactLES
using CompactLES: padded_index
using Printf
using Random

const CL = CompactLES
# `cases.jl` supplies the Brill slab's constants and the shared `per3`; it
# reads `references.jl` for the cases this script does not use.
include(joinpath(@__DIR__, "..", "test", "references.jl"))
include(joinpath(@__DIR__, "..", "test", "cases.jl"))

const DEFAULTS = (N = 32, tfin = 2.0, nmax = typemax(Int), ratio = 5.04,
                  cfl = 0.5, sample = 10, states = 8, seed = 20260919,
                  slab = true, progress = 0,
                  parts = "variables,production,step")

# A partial density at or below zero is outside the entropy's domain, and the
# species bound of `ArtificialProperties` reduces such an excursion without removing it, so
# a two-species interface produces them and the run continues. The logarithm is
# floored rather than allowed to raise, at this fraction of the local mixture
# density: relative, so a floored point contributes a bounded ~28 R_k ρ_k
# whatever the density ratio of the case, where an absolute floor near the
# smallest representable number would contribute ~691 R_k ρ_k and swamp the
# integral. Every consumer counts the floored points and reports the count, so a
# number resting on them is visible as such.
const RHO_REL = 1e-12

# Absolute floor for the temperature's logarithm. A nonpositive temperature
# means the state is not one this entropy is defined at; the run is over by
# then and the floor exists to keep the diagnostic from raising first.
const T_FLOOR = 1e-300

# --- the entropy, its gradient and its directional quadratic form ------------

"""
ρ, Σ_k ρ_k c_v,k, |u|²/2 and T at one conserved state `q`, which is a vector of
the `n_cons` components in the solver's layout (partial densities, three
momenta, total energy).
"""
@inline function state_scalars(cvk, q, ns, mom, ie)
    rho = 0.0
    C = 0.0
    for k in 1:ns
        rho += q[k]
        C += cvk[k] * q[k]
    end
    m2 = q[mom[1]]^2 + q[mom[2]]^2 + q[mom[3]]^2
    T = (q[ie] - m2 / (2rho)) / C
    return rho, C, m2 / (2rho^2), T
end

"""
    entropy_density(Rk, cvk, q, ns, mom, ie) -> ρs

ρs = Σ_k ρ_k (c_v,k ln T − R_k ln ρ_k) at one conserved state, the per-species
integration constants dropped. Dropping them is legitimate here because every
quantity reported is a difference of this over time or its pairing with a
right-hand side, and a constant times ρ_k contributes Σ_k a_k ∂_t ρ_k, which is
a conserved quantity's time derivative and cancels from both.
"""
@inline function entropy_density(Rk, cvk, q, ns, mom, ie)
    rho, _, _, T = state_scalars(cvk, q, ns, mom, ie)
    s = 0.0
    lnT = log(max(T, T_FLOOR))
    for k in 1:ns
        s += q[k] * (cvk[k] * lnT - Rk[k] * log(max(q[k], RHO_REL * abs(rho), T_FLOOR)))
    end
    return s
end

"""
    entropy_variables!(w, Rk, cvk, q, ns, mom, ie) -> w

w = ∂(ρs)/∂q at one conserved state:

    w_{ρ_k} = c_v,k ln T − R_k ln ρ_k − c_p,k + |u|²/(2T)
    w_{m_j} = −u_j / T
    w_{ρE}  = 1 / T

with c_p,k = c_v,k + R_k. The `variables` part checks this against central
differences of [`entropy_density`](@ref); the derivation is not trusted without
that check.
"""
function entropy_variables!(w, Rk, cvk, q, ns, mom, ie)
    rho, C, half_u2, T = state_scalars(cvk, q, ns, mom, ie)
    lnT = log(max(T, T_FLOOR))
    for k in 1:ns
        s_k = cvk[k] * lnT - Rk[k] * log(max(q[k], RHO_REL * abs(rho), T_FLOOR))
        w[k] = s_k - (cvk[k] + Rk[k]) + half_u2 / T
    end
    for j in 1:3
        w[mom[j]] = -(q[mom[j]] / rho) / T
    end
    w[ie] = 1 / T
    return w
end

"""
    admissible(cvk, q, ns, mom, ie) -> Bool

Whether one conserved state lies in the entropy's domain: every partial density
and the temperature strictly positive.

The floors above keep ρs and w finite at a point that has left it, but the
Hessian cannot be salvaged the same way: its species block carries R_k/ρ_k, so
one point with a negative partial density and a nonzero gradient contributes an
arbitrary amount set by the floor rather than by the state. Measured at 16³,
where 278 of 4096 points end outside the band: the unmasked quadratic form reads
1.6e7 against a channel production of 0.24. The semi-discrete integrals of part
`production` therefore exclude such points, both of them under the same mask, and
report how many were excluded.
"""
@inline function admissible(cvk, q, ns, mom, ie)
    for k in 1:ns
        q[k] > 0 || return false
    end
    _, _, _, T = state_scalars(cvk, q, ns, mom, ie)
    return T > 0
end

"""
    entropy_form(Rk, cvk, q, g, ns, mom, ie) -> −gᵀ(ρs)''(q)g

The quadratic form of the entropy Hessian in the direction `g`, as the chain
rule of [`entropy_variables!`](@ref) along `g` rather than an assembled
Hessian: gᵀσ''g = g·(d/dε)σ'(q + εg), and every derivative below is that
directional derivative of one closed-form expression. The sign is flipped on
return, so the value is nonnegative wherever ρs is concave in q, which is the
sense in which the continuous channel produces entropy.
"""
function entropy_form(Rk, cvk, q, g, ns, mom, ie)
    rho, C, half_u2, T = state_scalars(cvk, q, ns, mom, ie)
    drho = 0.0
    dC = 0.0
    for k in 1:ns
        drho += g[k]
        dC += cvk[k] * g[k]
    end
    # K = |m|²/2ρ, so ∂K = Σ_j u_j ∂m_j − (|u|²/2) ∂ρ.
    dK = -half_u2 * drho
    for j in 1:3
        dK += (q[mom[j]] / rho) * g[mom[j]]
    end
    dT = ((g[ie] - dK) - T * dC) / C
    form = (-dT / T^2) * g[ie]
    dudu = 0.0
    for j in 1:3
        u = q[mom[j]] / rho
        du = (g[mom[j]] - u * drho) / rho
        dudu += u * du
        form += (-du / T + u * dT / T^2) * g[mom[j]]
    end
    for k in 1:ns
        dw = cvk[k] * dT / T - Rk[k] * g[k] / max(q[k], RHO_REL * abs(rho), T_FLOOR) +
             dudu / T - half_u2 * dT / T^2
        form += dw * g[k]
    end
    return -form
end

# --- the buffers every field pass shares -------------------------------------

"""
Per-solver scratch for the entropy diagnostics: the mixture's constants, the
conserved layout, three `n_cons` work vectors, one padded field the
`volume_integral`s read, and the running count of points whose partial density
was floored out of the logarithm's domain.
"""
function entropy_buffers(solver)
    eq = solver.equations
    nc = eq.n_cons
    return (Rk = collect(Float64, solver.eos.Rk),
            cvk = collect(Float64, solver.eos.cvk),
            ns = eq.n_species, n_cons = nc, mom = eq.i_mom, ie = eq.i_energy,
            q = zeros(Float64, nc), w = zeros(Float64, nc),
            g = zeros(Float64, nc),
            field = zeros(Float64, CL.padded_extent(solver.decomp)),
            floored = Ref(0))
end

"S = ∫ρs dV, collective. Read from `Q` alone, so no primitives pass and no
halo exchange perturbs the run this is measuring."
function entropy_total(solver, Q, buf)
    nx, ny, nz = solver.decomp.n_local
    q, f = buf.q, buf.field
    bad = 0
    @inbounds for k in 1:nz, j in 1:ny, i in 1:nx
        I = padded_index(solver, i, j, k)
        for c in 1:buf.n_cons
            q[c] = Q[I, c]
        end
        for sp in 1:buf.ns
            q[sp] <= 0 && (bad += 1)
        end
        f[I] = entropy_density(buf.Rk, buf.cvk, q, buf.ns, buf.mom, buf.ie)
    end
    buf.floored[] += bad
    return volume_integral(solver, f)
end

"""
∫ w·dQ dV over the points in the entropy's domain, the entropy production of
whatever right-hand side `dQ` holds. With `dQ_off` given, the pairing is with the
difference dQ − dQ_off, which is the one channel the two solvers differ in.
Returns the integral and the number of excluded points; see [`admissible`](@ref).
"""
function production_integral(solver, Q, dQ, dQ_off, buf)
    nx, ny, nz = solver.decomp.n_local
    q, w, f = buf.q, buf.w, buf.field
    bad = 0
    @inbounds for k in 1:nz, j in 1:ny, i in 1:nx
        I = padded_index(solver, i, j, k)
        for c in 1:buf.n_cons
            q[c] = Q[I, c]
        end
        if !admissible(buf.cvk, q, buf.ns, buf.mom, buf.ie)
            f[I] = 0.0
            bad += 1
            continue
        end
        entropy_variables!(w, buf.Rk, buf.cvk, q, buf.ns, buf.mom, buf.ie)
        acc = 0.0
        for c in 1:buf.n_cons
            r = dQ_off === nothing ? dQ[I, c] : dQ[I, c] - dQ_off[I, c]
            acc += w[c] * r
        end
        f[I] = acc
    end
    return volume_integral(solver, f), MPI.Allreduce(bad, +, solver.comm)
end

"""
∫ D_b Σ_d ∂_d qᵀ (−(ρs)''(q)) ∂_d q dV, the continuous model's entropy
production of the bulk channel evaluated with the scheme's own gradients:
`solver.grad_Q[d, c]` holds ∂_d Q_c from `_bulk_gradients!` and every
`D_art[k]` holds D_b, both left in place by the `compute_rhs!` that precedes
this. Only meaningful under `species_flux = :bulk`, which is the only setting
that fills `grad_Q`. Masked to the entropy's domain exactly as
[`production_integral`](@ref) is, so the difference of the two is a difference
over one set of points.
"""
function quadratic_integral(solver, Q, buf)
    nx, ny, nz = solver.decomp.n_local
    q, g, f = buf.q, buf.g, buf.field
    active = solver.decomp.active
    D_b = solver.D_art[1]
    gQ = solver.grad_Q
    @inbounds for k in 1:nz, j in 1:ny, i in 1:nx
        I = padded_index(solver, i, j, k)
        for c in 1:buf.n_cons
            q[c] = Q[I, c]
        end
        if !admissible(buf.cvk, q, buf.ns, buf.mom, buf.ie)
            f[I] = 0.0
            continue
        end
        acc = 0.0
        for d in 1:3
            active[d] || continue
            for c in 1:buf.n_cons
                g[c] = gQ[d, c][I]
            end
            acc += entropy_form(buf.Rk, buf.cvk, q, g, buf.ns, buf.mom, buf.ie)
        end
        f[I] = D_b[I] * acc
    end
    return volume_integral(solver, f)
end

"""
∫ D_b Σ_d Σ_k R_k (∂_d ρ_k)² / ρ_k dV, the continuous model's entropy production
of the partial-density channel (`reference/DESIGN.md`, "The species channel"),
evaluated with the partial-density gradients `_bulk_gradients!` leaves in the
first n_species columns of `solver.grad_Q` under `species_flux =
:partial_density`. Masked to the entropy's domain as the other two integrals
are.
"""
function partial_density_integral(solver, Q, buf)
    nx, ny, nz = solver.decomp.n_local
    q, f = buf.q, buf.field
    active = solver.decomp.active
    D_b = solver.D_art[1]
    gQ = solver.grad_Q
    @inbounds for k in 1:nz, j in 1:ny, i in 1:nx
        I = padded_index(solver, i, j, k)
        for c in 1:buf.n_cons
            q[c] = Q[I, c]
        end
        if !admissible(buf.cvk, q, buf.ns, buf.mom, buf.ie)
            f[I] = 0.0
            continue
        end
        acc = 0.0
        for d in 1:3
            active[d] || continue
            for sp in 1:buf.ns
                acc += buf.Rk[sp] * gQ[d, sp][I]^2 / q[sp]
            end
        end
        f[I] = D_b[I] * acc
    end
    return volume_integral(solver, f)
end

# --- the two flow cases ------------------------------------------------------

# Radius and half-thickness of the interface. The thickness is physical rather
# than a cell count, so refining the grid resolves the same interface instead
# of measuring a different one; it is about two cells at N = 32.
const SPHERE_R = π / 2
const SPHERE_W = 0.4

"""
Taylor–Green velocity at Mach 0.1 over a tanh sphere of heavy gas in air, in a
2π-periodic cube at uniform p = 1 and T = 1. The composition alone varies at
t = 0, so the density ratio sits on a smooth interface a few cells wide, which
is what makes the species sensor fire; the velocity field supplies the strain
that the rest of the artificial properties key on.
"""
function interface_problem(ratio)
    eos = IdealMixture([IdealSpecies{Float64}("air", 1.0, 1.4),
                        IdealSpecies{Float64}("heavy", 1 / ratio, 1.09)])
    amp = 0.1 * sqrt(1.4)          # Mach 0.1 on air's sound speed at p = T = 1
    return Problem(eos=eos, transport=ConstantTransport(mu0=0.0),
                   domain=((0.0, 2π), (0.0, 2π), (0.0, 2π)), bcs=per3,
                   ic=(x, y, z) -> begin
                       r = sqrt((x - π)^2 + (y - π)^2 + (z - π)^2)
                       Yh = (1 - tanh((r - SPHERE_R) / SPHERE_W)) / 2
                       Rmix = (1 - Yh) + Yh / ratio
                       Prim(Y=(1 - Yh, Yh), rho=1 / Rmix, p=1.0,
                            u=(amp * sin(x) * cos(y) * cos(z),
                               -amp * cos(x) * sin(y) * cos(z), 0.0))
                   end)
end

# `validity = :permissive`: the mass-fraction excursion is what the entropy
# floor above counts, so a run that leaves the species band is measured here
# rather than rejected.
interface_setup(opt, art) =
    setup(interface_problem(opt.ratio),
          Numerics(n_global=(opt.N, opt.N, opt.N), cfl=opt.cfl, art=art,
                   control=StepControl(validity=:permissive)))

# `brill_slab` in test/cases.jl drives its own `run!` and takes no callback, so
# the per-step entropy of that case cannot be recorded through it and the setup
# is rebuilt here. Both read BR_R, BR_NP, BR_U and BR_PERIODS from cases.jl, so
# the two cannot come to describe different slabs.
function slab_setup(art)
    N = 20 * BR_NP
    h = 1.0 / N
    w = 3 * BR_NP * h / 16
    eos = IdealMixture([IdealSpecies{Float64}("light", 1.0, 1.4),
                        IdealSpecies{Float64}("heavy", 1 / BR_R, 1.4)])
    prob = Problem(eos=eos, transport=ConstantTransport(mu0=0.0),
                   domain=((0.0, 1.0), (0.0, h), (0.0, h)), bcs=per3,
                   ic=(x, y, z) -> begin
                       V = (1 - tanh((abs(x - 0.5) - 0.25) / w)) / 2
                       rho = V * BR_R + (1 - V)
                       Yh = V * BR_R / rho
                       Prim(Y=(1 - Yh, Yh), rho=rho, u=(BR_U, 0.0, 0.0), p=1.0)
                   end)
    return setup(prob, Numerics(n_global=(N, 1, 1), art=art, cfl=0.4,
                                control=StepControl(retries=4,
                                                    validity=:permissive)))
end

# --- part 1: the entropy variables -------------------------------------------

"""
Random admissible conserved states: partial densities well inside (0, 1], a
subsonic velocity and a temperature of order one, so every logarithm and every
division below is far from its limit and the finite differences measure the
derivation rather than a floor.
"""
function random_states(rng, mix, n)
    eq = CL.NavierStokes1T(mix)
    out = Vector{Vector{Float64}}(undef, n)
    for s in 1:n
        q = zeros(Float64, eq.n_cons)
        rho = 0.0
        C = 0.0
        for k in 1:eq.n_species
            q[k] = 0.1 + 0.9 * rand(rng)
            rho += q[k]
            C += mix.cvk[k] * q[k]
        end
        u = ntuple(_ -> 0.6 * (rand(rng) - 0.5), 3)
        for j in 1:3
            q[eq.i_mom[j]] = rho * u[j]
        end
        T = 0.5 + 1.5 * rand(rng)
        q[eq.i_energy] = C * T + rho * (u[1]^2 + u[2]^2 + u[3]^2) / 2
        out[s] = q
    end
    return out, eq
end

function part_variables(opt, rank)
    mix = IdealMixture([IdealSpecies{Float64}("air", 1.0, 1.4),
                        IdealSpecies{Float64}("heavy", 1 / opt.ratio, 1.09),
                        IdealSpecies{Float64}("third", 0.4, 1.25)])
    rng = MersenneTwister(opt.seed)
    states, eq = random_states(rng, mix, opt.states)
    Rk, cvk = mix.Rk, mix.cvk
    ns, mom, ie = eq.n_species, eq.i_mom, eq.i_energy
    w = zeros(Float64, eq.n_cons)
    wp = zeros(Float64, eq.n_cons)
    wm = zeros(Float64, eq.n_cons)
    ent(q) = entropy_density(Rk, cvk, q, ns, mom, ie)
    worst_w = 0.0
    worst_form = 0.0
    for q in states
        entropy_variables!(w, Rk, cvk, q, ns, mom, ie)
        # Central differences of ρs, component by component, at a step scaled
        # to the component so a momentum near zero is differenced on the same
        # relative footing as a partial density.
        for c in 1:eq.n_cons
            d = 1e-6 * max(abs(q[c]), 1.0)
            qp = copy(q); qp[c] += d
            qm = copy(q); qm[c] -= d
            fd = (ent(qp) - ent(qm)) / 2d
            worst_w = max(worst_w, abs(w[c] - fd) / max(abs(fd), 1.0))
        end
        # The directional form against a central difference of w itself, which
        # is the same identity `entropy_form` differentiates by hand.
        g = [0.4 * (rand(rng) - 0.5) * max(abs(x), 1.0) for x in q]
        eps = 1e-6
        qp = q .+ eps .* g
        qm = q .- eps .* g
        entropy_variables!(wp, Rk, cvk, qp, ns, mom, ie)
        entropy_variables!(wm, Rk, cvk, qm, ns, mom, ie)
        fd = 0.0
        for c in 1:eq.n_cons
            fd += g[c] * (wp[c] - wm[c]) / 2eps
        end
        closed = -entropy_form(Rk, cvk, q, g, ns, mom, ie)
        worst_form = max(worst_form, abs(closed - fd) / max(abs(fd), 1.0))
    end
    rank == 0 || return nothing
    println("\n=== entropy variables w = d(rho s)/dq ===")
    println("  w_rho_k = c_v,k ln T - R_k ln rho_k - c_p,k + |u|^2/(2T)")
    println("  w_m_j   = -u_j / T")
    println("  w_rhoE  = 1 / T")
    @printf("  %d species, %d random admissible states, central differences ",
            ns, opt.states)
    @printf("at a relative step of 1e-6\n")
    @printf("  max relative discrepancy: w %.3e, quadratic form %.3e\n",
            worst_w, worst_form)
    return nothing
end

# --- part 2: the semi-discrete production of the channel ----------------------

"""
One sample of the channel's entropy production on the state `Q` the run holds.

`full` and `off` differ only in `C_D` and `C_Y`, and neither constant reaches
μ*, β* or κ*: `compute_artificial!` builds those from the strain magnitude and
the internal energy before the species sweep, and only `bulk_diffusivity!` (or
the Fickian sweep it replaces) reads the two. The difference of the two
right-hand sides is therefore the species channel alone, up to the round-off of
the compact divergence, and the three coefficient differences returned beside
it are the check that this holds.

The full solver's coefficient arrays are put back afterwards: `max_rate` sizes
the next step from them and this sample has overwritten them, so without the
restore the instrument would perturb the run it is measuring.
"""
function production_sample(full, off, Q, Q_off, dQf, dQo, buf, channel)
    art_saved = CL.art_block(full)
    tstage_saved = full.tstage
    CL.compute_rhs!(full, Q, dQf)
    # Before anything else runs on this solver: `grad_Q` and `D_art` are
    # workspace and coefficient arrays the next evaluation overwrites.
    form = channel === :bulk ? quadratic_integral(full, Q, buf) :
           channel === :partial_density ? partial_density_integral(full, Q, buf) : NaN
    blk_full = CL.art_block(full)
    copyto!(parent(Q_off), parent(Q))
    CL.compute_rhs!(off, Q_off, dQo)
    blk_off = CL.art_block(off)
    p_channel, excluded = production_integral(full, Q, dQf, dQo, buf)
    p_total, _ = production_integral(full, Q, dQf, nothing, buf)
    CL.set_art_block!(full, art_saved)
    full.tstage = tstage_saved
    # μ*, β*, κ* are the first three planes of the block; the D* follow.
    d_coef = maximum(abs, view(blk_full, :, :, :, 1:3) .-
                          view(blk_off, :, :, :, 1:3); init=0.0)
    d_off = maximum(abs, view(blk_off, :, :, :, 4:size(blk_off, 4)); init=0.0)
    red = MPI.Allreduce([d_coef, d_off], max, full.comm)
    return (t=full.t, step=full.step, p_channel=p_channel, p_total=p_total,
            form=form, excluded=excluded, d_coef=red[1], d_off=red[2])
end

function part_production(opt, rank)
    rank == 0 && println("\n=== semi-discrete entropy production of the " *
                         "species channel ===")
    for channel in (:partial_density, :bulk, :fickian)
        art_on = ArtificialProperties(species_flux=channel)
        art_off = ArtificialProperties(species_flux=channel, C_D=0.0, C_Y=0.0)
        full, Q = interface_setup(opt, art_on)
        off, Q_off = interface_setup(opt, art_off)
        workspace = Workspace(Q)
        buf = entropy_buffers(full)
        dQf = copy(Q)
        dQo = copy(Q_off)
        rows = NamedTuple[]
        record = Callback(EveryStep(opt.sample), (s, Qs) -> begin
            push!(rows, production_sample(s, off, Qs, Q_off, dQf, dQo, buf,
                                          channel))
            nothing
        end)
        callbacks = opt.progress > 0 ?
                    (record, ProgressLog(every=opt.progress, tfinal=opt.tfin)) :
                    (record,)
        elapsed = @elapsed run!(full, Q, workspace; tfinal=opt.tfin,
                                nmax=opt.nmax, callback=callbacks)
        rank == 0 || continue
        @printf("\n--- species_flux = :%s, %d^3, cfl %.3g, %d steps, %.1f s\n",
                channel, opt.N, opt.cfl, full.step, elapsed)
        println(" step      t     P_channel      P_total    share   " *
                "quad form     defect   defect/P  excl")
        for r in rows
            share = 100 * r.p_channel / (abs(r.p_total) > 0 ? r.p_total : NaN)
            defect = r.p_channel - r.form
            rel = defect / (abs(r.p_channel) > 0 ? r.p_channel : NaN)
            @printf("%5d %6.3f %+12.4e %+12.4e %6.1f%%", r.step, r.t,
                    r.p_channel, r.p_total, share)
            if isnan(r.form)
                @printf("          n/a          n/a        n/a")
            else
                @printf(" %+11.4e %+10.3e %8.2f%%", r.form, defect, 100rel)
            end
            @printf(" %5d\n", r.excluded)
        end
        isempty(rows) && println("  (no sample: raise nmax or lower sample=)")
        if !isempty(rows)
            @printf("isolation: max |d(mu*,beta*,kappa*)| %.3e, ",
                    maximum(r -> r.d_coef, rows))
            @printf("max D* with C_D = C_Y = 0 %.3e\n",
                    maximum(r -> r.d_off, rows))
            println("excl = points outside the entropy's domain, excluded " *
                    "from both integrals")
        end
    end
    return nothing
end

# --- part 3: the fully discrete update ---------------------------------------

"""
One configuration's per-step entropy record: S after the Runge–Kutta update and
S after the filter pass, for every accepted step.

`run!` filters between the update and the callbacks, so no callback can see the
state between them. The filter is therefore taken out of the driver by setting
`filter_interval` to zero after setup and applied from the callback instead,
which is the same operator at the same point of the step: `filter_weight` reads
`dt_prev` and `filter_rate_prev`, both recorded before the callbacks run, and
the interval it also reads is restored around the call. The state entering the
next step's checks is the filtered one either way.

A `StepControl` rollback rewinds the run but not the vectors below, so the
record count is printed beside the step count; a mismatch means a retry
happened and the sums include the abandoned steps.
"""
function step_record(solver, Q, tfinal, opt, buf)
    interval = solver.filter_interval
    solver.filter_interval = 0
    workspace = Workspace(Q)
    S0 = entropy_total(solver, Q, buf)
    d_rk = Float64[]
    d_filt = Float64[]
    prev = Ref(S0)
    record = Callback(EveryStep(1), (s, Qs) -> begin
        S_step = entropy_total(s, Qs, buf)
        if interval > 0 && s.step % interval == 0
            s.filter_interval = interval
            CL.filter_state!(s, Qs)
            s.filter_interval = 0
        end
        S_filt = entropy_total(s, Qs, buf)
        push!(d_rk, S_step - prev[])
        push!(d_filt, S_filt - S_step)
        prev[] = S_filt
        nothing
    end)
    callbacks = opt.progress > 0 ?
                (record, ProgressLog(every=opt.progress, tfinal=tfinal)) :
                (record,)
    elapsed = @elapsed run!(solver, Q, workspace; tfinal=tfinal,
                            nmax=opt.nmax, callback=callbacks)
    return (S0=S0, S1=prev[], d_rk=d_rk, d_filt=d_filt, elapsed=elapsed,
            steps=solver.step,
            floored=MPI.Allreduce(buf.floored[], +, solver.comm))
end

"Decreases, largest decrease and sum of one series of entropy increments."
function decrease_summary(d)
    isempty(d) && return (n=0, worst=0.0, sum=0.0)
    return (n=count(<(0), d), worst=minimum(d), sum=sum(d))
end

function print_step_row(label, r)
    rk = decrease_summary(r.d_rk)
    fl = decrease_summary(r.d_filt)
    @printf("%-10s %5d/%-5d %+11.4e %+11.4e", label, r.steps, length(r.d_rk),
            r.S0, r.S1 - r.S0)
    @printf(" | %5d %+10.3e %+10.3e", rk.n, rk.worst, rk.sum)
    @printf(" | %5d %+10.3e %+10.3e", fl.n, fl.worst, fl.sum)
    @printf(" | %8d %6.1f\n", r.floored, r.elapsed)
end

# A configuration that loses positivity raises `SolverFailure` from `run!`;
# the row records the failure and the remaining configurations still run.
# One rank only: an exception on one rank of several would leave the others
# waiting in a collective, so under `mpiexec` the failure is left to abort.
function attempt_step(f, label, rank)
    try
        return f()
    catch e
        MPI.Comm_size(MPI.COMM_WORLD) == 1 || rethrow()
        rank == 0 && @printf("%-10s FAIL: %s
", label,
                             first(split(sprint(showerror, e), '
')))
        return nothing
    end
end

function part_step(opt, rank)
    configs = (("partial", ArtificialProperties(species_flux=:partial_density)),
               ("bulk", ArtificialProperties(species_flux=:bulk)),
               ("fickian", ArtificialProperties(species_flux=:fickian)),
               ("art off", ArtificialProperties(enabled=false)))
    header = "config     steps/rec  S(0)         dS total    " *
             "|  RK step: dec      worst        sum" *
             " |  filter: dec      worst        sum |  floored  wall"
    rank == 0 && println("\n=== fully discrete entropy, " *
                         "three-dimensional interface ===")
    if rank == 0
        @printf("%d^3, cfl %.3g, tfin %.3g, filter every step\n",
                opt.N, opt.cfl, opt.tfin)
        println(header)
    end
    for (label, art) in configs
        solver, Q = interface_setup(opt, art)
        buf = entropy_buffers(solver)
        r = attempt_step(label, rank) do
            step_record(solver, Q, opt.tfin, opt, buf)
        end
        r === nothing || rank == 0 && print_step_row(label, r)
    end
    opt.slab || return nothing
    rank == 0 && println("\n=== fully discrete entropy, Brill slab " *
                         "(1-D, ratio $(BR_R), $(BR_NP) cells per interface) ===")
    rank == 0 && println(header)
    for (label, art) in configs
        solver, Q = slab_setup(art)
        buf = entropy_buffers(solver)
        r = attempt_step(label, rank) do
            step_record(solver, Q, BR_PERIODS / BR_U, opt, buf)
        end
        r === nothing || rank == 0 && print_step_row(label, r)
    end
    return nothing
end

function main(opt)
    rank = MPI.Comm_rank(MPI.COMM_WORLD)
    parts = [strip(p) for p in split(opt.parts, ',') if !isempty(strip(p))]
    known = ("variables", "production", "step")
    for p in parts
        p in known ||
            error("unknown part '$p', want one of: " * join(known, ", "))
    end
    if rank == 0
        @printf("=== bulk species channel entropy, %d rank(s), %d thread(s)\n",
                MPI.Comm_size(MPI.COMM_WORLD), Threads.nthreads())
    end
    "variables" in parts && part_variables(opt, rank)
    "production" in parts && part_production(opt, rank)
    "step" in parts && part_step(opt, rank)
    rank == 0 && println("\nbulkentropy complete")
    return nothing
end

const _opt = CompactLES.script_args(ARGS, DEFAULTS; positional=(:N, :tfin))
mpi_main(() -> main(_opt))
