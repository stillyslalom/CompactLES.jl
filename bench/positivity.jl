# A conservative positivity limiter in face-flux form on the one-dimensional
# Woodward–Colella and planar Noh (ν = 1) cases of test/cases.jl: whether it
# removes the points with ρ ≤ 0 or ρe = E − ½|m|²/ρ ≤ 0, what it does to the
# validation metrics, and how often it acts.
#
#   julia --project=. -t 1 bench/positivity.jl                     # every part
#   julia --project=. -t 1 bench/positivity.jl part=weights,pulse
#   julia --project=. -t 1 bench/positivity.jl part=variants cases=noh variants=none,A+B
#
# The collocated compact divergence on a closed line is a difference of face
# fluxes under node weights W with Σ_i W_i (D f)_i = f_N − f_1: the face flux is
# the running sum F̂_{i+½} = f_1 + Σ_{k≤i} W_k (D f)_k, and (D f)_i is
# (F̂_{i+½} − F̂_{i−½}) / W_i. In one Cartesian dimension the right-hand side is
# −D F exactly, so the face form of any linear combination of right-hand sides
# is the running sum of that combination, anchored at the point flux of node 1.
# The script computes W from the solver's own divergence plan, column by column,
# and drives the solver's pieces (`apply_bcs!`, `compute_rhs!`, the low-storage
# update, `filter_state!`) through its own copy of the `run!` loop.
#
# Two limiters, after Hu, Adams & Shu (J. Comput. Phys. 242, 2013), each a
# correction applied only at the faces it limits, so a run in which no face is
# limited is the unlimited run bit for bit:
#
#   A  the increment of each Runge–Kutta stage, B_k du_k, in face form
#      B_k Φ_k, blended per face toward the Lax–Friedrichs flux of the stage's
#      base state times the stage's own time advance τ_k = (c_{k+1} − c_k) dt.
#      θ per face is the smaller of the two cells' limits, each the largest θ
#      for which the half state Q_i ∓ 2 G_{i±½} / W_i keeps ρ and ρe above ε
#      (linear interpolation in ρ, then in ρe, which is concave). The limited
#      increment is written back into the register du, so the next stage's
#      A_{k+1} du sees it; `A-nostore` limits Q only and leaves du unlimited.
#      `A-rhs` blends only the stage's own term B_k dt F̂_k toward τ_k F_LF and
#      leaves the history B_k A_k Φ_{k−1} as it is, the form that needs no face
#      register in several dimensions and carries no guarantee.
#   B  the correction of each filter pass, whose running sum under the same
#      weights is a face flux once the constant a closed line leaves in it is
#      removed (measured from the interior face relation of the filter). θ per
#      face scales that flux toward zero, the admissible state before the pass.
#
# Boundary faces are never limited, and a node a `DirichletBC` overwrites
# (Noh's inflow node) sets no limit. A cell whose half state at θ = 0 is not
# admissible (an inadmissible base state, or the first-order CFL bound broken
# at a small weight) takes θ = 0 from that side, and is counted as unguaranteed.
#
# Parts (`part=` takes a comma list):
#
#   weights   W for each case's line, against the polynomial and random-field
#             identity and the interior face relation of the derivative
#   identity  this loop with no limiter against `run!` with the filter applied
#             from a callback (bench/shockfoot.jl's form), bitwise, over
#             `identity_steps` steps of each case
#   pulse     a smooth acoustic pulse between slip walls, unlimited and A + B:
#             no face limited and the states bitwise equal
#   variants  each case to its end time under each variant in `variants`:
#             none, A, A-nostore, A-rhs, B, A+B or A-rhs+B, each optionally
#             followed by `@r` for bounds of r times the initial minimum ρ and
#             ρe (A+B@0.01)
#
# The tallies of `variants`, summed over the steps of a run:
#
#   bad pre / post    point-steps with ρ ≤ 0 or ρe ≤ 0 before / after the
#                     filter pass (as bench/shockfoot.jl, which counts ρe ≤ 0)
#   by step, by filt  points bad before a pass that were good after the last
#                     one, and points a pass turns bad
#   stage bad         point-stages bad after a Runge–Kutta stage
#   min pre / post    the lowest ρe before / after a pass
#   A faces, B faces  the fraction of interior face-stages (face-passes)
#                     limited, and the number of unguaranteed cell-sides
#   ahead             the range of distances, in cells, of the faces A limits
#                     ahead of the nearest front, and the count behind one
#   mass, energy      the change of Σ W Q over the run, and the part of it the
#                     corrections made, relative to the initial totals
#   rate              the mean over steps of the rate that sized the step over
#                     its hyperbolic part (|u| + c)/h, and on the steps where the
#                     diffusive part binds, which artificial diffusivity is largest
#
# Scratch tooling: it prints tables and asserts nothing.

using MPI
MPI.Initialized() || MPI.Init(threadlevel=:funneled)
using CompactLES
using CompactLES: filter_state!, padded_index, xcoord
using Printf
using Statistics: median

const CL = CompactLES
include(joinpath(@__DIR__, "..", "test", "references.jl"))
include(joinpath(@__DIR__, "..", "test", "cases.jl"))

const CASES = (:woodward, :noh)
const PARTS = ("weights", "identity", "pulse", "variants")
const VARIANTS = ("none", "A", "A-nostore", "A-rhs", "B", "A+B", "A-rhs+B")
const WC_AMBIENT = 0.01
# Hu, Adams & Shu take ε = min(1e-13, the initial minimum) for ρ and for ρe.
# A variant named `name@r` takes ε = r times the initial minimum instead.
const EPS = 1e-13
# The time advance of each stage, c_{k+1} − c_k with c_6 = 1. Its sum is 1, so a
# step limited at every stage is a sequence of first-order steps of total dt.
const STAGE_ADVANCE = ntuple(k -> (k < 5 ? CL.RKC[k+1] : 1.0) - CL.RKC[k], 5)

# --- the cases ----------------------------------------------------------------

end_time(case) = case === :woodward ? WC_T : case === :noh ? NOH_T : PULSE_T
case_gamma(case) = case === :noh ? NOH_G : 1.4

function woodward_problem()
    h = 1.0 / (WC_N - 1)
    δ = 2h
    return Problem(eos=IdealSpecies("gas"; gamma=1.4, R=1.0),
                   transport=ConstantTransport(mu0=0.0),
                   domain=((0.0, 1.0), (0.0, h), (0.0, h)),
                   bcs=((SlipWallBC(), SlipWallBC()), per3[2], per3[3]),
                   ic=(x, y, z) -> Prim(rho=1.0, u=(0.0, 0.0, 0.0),
                        p = 1000 * (1 - tanh_blend(x, 0.1, δ)) +
                            WC_AMBIENT * (tanh_blend(x, 0.1, δ) - tanh_blend(x, 0.9, δ)) +
                            100 * tanh_blend(x, 0.9, δ)))
end

# A smooth pressure pulse of amplitude 1e-3 between slip walls, reflected once.
const PULSE_N = 201
const PULSE_T = 0.6
function pulse_problem()
    h = 1.0 / (PULSE_N - 1)
    return Problem(eos=IdealSpecies("gas"; gamma=1.4, R=1.0),
                   transport=ConstantTransport(mu0=0.0),
                   domain=((0.0, 1.0), (0.0, h), (0.0, h)),
                   bcs=((SlipWallBC(), SlipWallBC()), per3[2], per3[3]),
                   ic=(x, y, z) -> Prim(rho=1.0, u=(0.0, 0.0, 0.0),
                                        p=1 + 1e-3 * exp(-((x - 0.3) / 0.05)^2)))
end

# The case's numerics with the filter left to the loop (`interval = 0`), under
# `validity = :permissive` as in bench/shockfoot.jl.
function build(case)
    filter = StateFilter(compact_filter(); cfl=0.35, interval=0)
    control = StepControl(validity=:permissive)
    art = ArtificialProperties(enabled=true)
    case === :woodward &&
        return setup(woodward_problem(),
                     Numerics(n_global=(WC_N, 1, 1), art=art, cfl=0.3, filter=filter,
                              control=control))
    case === :noh &&
        return setup(noh_problem(1),
                     Numerics(n_global=(Dict(NOH_N)[1], 1, 1), art=art, cfl=NOH_CFL,
                              filter=filter, control=control))
    return setup(pulse_problem(),
                 Numerics(n_global=(PULSE_N, 1, 1), art=art, cfl=0.3, filter=filter,
                          control=control))
end

# --- the face weights ---------------------------------------------------------

# The divergence plan along dimension 1 as a dense matrix, one column per unit
# flux, through the same call `compute_rhs!` makes.
function divergence_matrix(solver, Q, idx)
    N = length(idx)
    F = zero(solver.flux[1, 1])
    dQ = zero(Q)
    D = zeros(N, N)
    for j in 1:N
        fill!(F, 0)
        F[idx[j]] = 1
        fill!(parent(dQ), 0)
        CL.div_subtract_along!(dQ, 1, F, solver, 1, 1, nothing)
        for i in 1:N
            D[i, j] = -parent(dQ)[idx[i], 1]
        end
    end
    return D
end

# W with Wᵀ D = (e_N − e_1)ᵀ, free on `edge` nodes at each end and one shared
# value inside, by least squares; the residual is the test of the face form.
function closure_weights(D; edge=16)
    N = size(D, 1)
    free = [1:edge; N-edge+1:N]
    inner = edge+1:N-edge
    M = zeros(N, length(free) + 1)
    for j in 1:N
        for (u, i) in enumerate(free)
            M[j, u] = D[i, j]
        end
        M[j, end] = sum(D[i, j] for i in inner)
    end
    target = zeros(N)
    target[1], target[N] = -1, 1
    x = M \ target
    W = fill(x[end], N)
    W[free] = x[1:end-1]
    return W, maximum(abs, M * x - target)
end

# The interior face relation α F̂_{j−½} + F̂_{j+½} + α F̂_{j+3/2} = Ĝ_{j+½} of the
# compact derivative, with Ĝ_{j+½} = Σ_m c_m Σ_{l=1−m}^{m} f_{j+l}, as the largest
# residual over faces `margin` or more from either end.
function face_relation_residual(scheme, f, Df, W; margin=24)
    N = length(f)
    Fh = similar(f, N + 1)
    Fh[1] = f[1]
    for i in 1:N
        Fh[i+1] = Fh[i] + W[i] * Df[i]
    end
    α, c = scheme.alpha, scheme.coeffs
    worst = 0.0
    for j in margin:N-margin
        g = sum(c[m] * sum(f[j+l] for l in 1-m:m) for m in eachindex(c))
        worst = max(worst, abs(α * Fh[j] + Fh[j+1] + α * Fh[j+2] - g))
    end
    return worst, abs(Fh[N+1] - f[N])
end

# --- the line geometry and scratch ---------------------------------------------

struct Line
    idx::Vector{CartesianIndex{3}}
    W::Vector{Float64}          # node weights of the divergence, summing to the length
    ω::Vector{Float64}          # the same normalized to 1 inside, for the filter
    constrained::BitVector      # nodes whose half states set a limit
    γ::Float64
    ns::Int
    im::NTuple{3,Int}
    ie::Int
    nc::Int
    psi::Vector{Float64}        # explicit face stencil of the filter correction
    alpha_f::Float64
    eps_rho::Float64            # the bounds the limiters keep ρ and ρe above
    eps_e::Float64
end

function filter_face_stencil(scheme)
    M = length(scheme.coeffs)
    s = zeros(2M + 1)               # s[l + M + 1], l = −M..M, of (B − A) f
    s[M+1] = scheme.a0 - 1
    for m in 1:M
        v = scheme.coeffs[m] - (m == 1 ? scheme.alpha : 0.0)
        s[M+1+m] = v
        s[M+1-m] = v
    end
    # t_l = Σ_{l' ≥ l} s_{l'} for l = 1 − M .. M, so that Σ_l s_l f_{i+l} is
    # ψ_{i+½} − ψ_{i−½} with ψ_{i+½} = Σ_l t_l f_{i+l}.
    return [sum(s[k+M+1] for k in l:M) for l in 1-M:M]
end

function line_geometry(solver, Q, case; weights=nothing, eps_ratio=0.0)
    eq = solver.equations
    N = solver.decomp.n_local[1]
    idx = [padded_index(solver, i, 1, 1) for i in 1:N]
    W = weights === nothing ? closure_weights(divergence_matrix(solver, Q, idx))[1] :
        weights
    h = W[N÷2]
    constrained = trues(N)
    # Noh's inflow node is overwritten by its DirichletBC before every stage.
    case === :noh && (constrained[N] = false)
    fscheme = compact_filter()
    L = Line(idx, W, W ./ h, constrained, case_gamma(case), eq.n_species, eq.i_mom,
             eq.i_energy, eq.n_cons, filter_face_stencil(fscheme), fscheme.alpha, EPS, EPS)
    s = line_energy(parent(Q), L)
    eps_rho = eps_ratio > 0 ? eps_ratio * minimum(s.ρ) : min(EPS, minimum(s.ρ))
    eps_e = eps_ratio > 0 ? eps_ratio * minimum(s.ρe) : min(EPS, minimum(s.ρe))
    return Line(idx, W, W ./ h, constrained, L.γ, L.ns, L.im, L.ie, L.nc, L.psi,
                L.alpha_f, eps_rho, eps_e)
end

function read_line!(out, data, L::Line)
    for c in 1:L.nc, i in eachindex(L.idx)
        out[i, c] = data[L.idx[i], c]
    end
    return out
end

struct Scratch
    base::Matrix{Float64}
    flux::Matrix{Float64}       # Euler flux at each node of the base state
    speed::Vector{Float64}
    GH::Matrix{Float64}         # high-order face increments, faces 0..N
    GL::Matrix{Float64}         # first-order face increments
    θ::Vector{Float64}
    limited::BitVector          # faces limited at some stage of this step
    register::Vector{Float64}   # the boundary face register at node 1
    register_far::Vector{Float64}
    qH::Vector{Float64}
    qL::Vector{Float64}
    q::Vector{Float64}
end
function Scratch(L::Line)
    N, nc = length(L.idx), L.nc
    return Scratch(zeros(N, nc), zeros(N, nc), zeros(N), zeros(N + 1, nc),
                   zeros(N + 1, nc), ones(N + 1), falses(N + 1), zeros(nc), zeros(nc),
                   zeros(nc), zeros(nc), zeros(nc))
end

# --- admissibility --------------------------------------------------------------

function density(q, L::Line)
    ρ = 0.0
    for s in 1:L.ns
        ρ += q[s]
    end
    return ρ
end

internal_energy(q, L::Line, ρ) =
    q[L.ie] - 0.5 * (q[L.im[1]]^2 + q[L.im[2]]^2 + q[L.im[3]]^2) / ρ

function admissible(q, L::Line)
    ρ = density(q, L)
    return ρ >= L.eps_rho && internal_energy(q, L, ρ) >= L.eps_e
end

# The largest θ ∈ [0, 1] at which the half state q0 + s (θ GH + (1 − θ) GL)
# keeps ρ and ρe at or above ε, row `j` of GH and GL: linear interpolation in
# ρ, then in ρe along the shortened segment, which is sufficient because ρe is
# concave in the conserved state. Returns (θ, guaranteed).
function side_theta(q0, GH, GL, j, s, S::Scratch, L::Line)
    qH, qL, q = S.qH, S.qL, S.q
    for c in 1:L.nc
        qH[c] = q0[c] + s * GH[j, c]
        qL[c] = q0[c] + s * GL[j, c]
    end
    admissible(qH, L) && return 1.0, true
    ρL = density(qL, L)
    (ρL >= L.eps_rho && internal_energy(qL, L, ρL) >= L.eps_e) || return 0.0, false
    ρH = density(qH, L)
    θ = ρH < L.eps_rho ? (ρL - L.eps_rho) / (ρL - ρH) : 1.0
    for c in 1:L.nc
        q[c] = qL[c] + θ * (qH[c] - qL[c])
    end
    eL = internal_energy(qL, L, ρL)
    e = internal_energy(q, L, density(q, L))
    e < L.eps_e && (θ *= (eL - L.eps_e) / (eL - e))
    return clamp(θ, 0.0, 1.0), true
end

# --- the tallies ----------------------------------------------------------------

mutable struct Tally
    steps::Int
    bad_pre::Int
    bad_post::Int
    max_bad::Int
    by_step::Int
    by_filter::Int
    cured::Int
    stage_bad::Int
    min_pre::Float64
    min_post::Float64
    stage_faces::Int
    stage_limited::Int
    stage_unguaranteed::Int
    filter_faces::Int
    filter_limited::Int
    filter_unguaranteed::Int
    ahead_min::Float64
    ahead_max::Float64
    behind::Int
    correction_mass::Float64
    correction_energy::Float64
    closure_mismatch::Float64
    offset_spread::Float64
    lf_cfl_edge::Float64        # largest 2 τ α / W at an end node
    lf_cfl_inner::Float64       # and inside
    limiter_wall::Float64
    rate_ratio::Float64         # Σ over steps of the step's rate over (|u| + c)/h
    diffusive::Int              # steps whose rate exceeds (|u| + c)/h by 5%
    dominant::Vector{Int}       # their largest diffusivity: μ*/ρ, β*/ρ, κ*/(ρ c_p)
end
Tally() = Tally(0, 0, 0, 0, 0, 0, 0, 0, Inf, Inf, 0, 0, 0, 0, 0, 0, Inf, -Inf, 0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, zeros(Int, 3))

# What sized the step: the rate of `max_rate` against its hyperbolic part, and
# where the diffusive part binds, which artificial diffusivity is the largest.
function record_rate!(T, solver, rate, hyperbolic, L::Line)
    T.rate_ratio += rate / hyperbolic
    rate > 1.05 * hyperbolic || return nothing
    T.diffusive += 1
    cp = L.γ / (L.γ - 1)
    largest = zeros(3)
    for I in L.idx
        ρ = solver.rho[I]
        largest[1] = max(largest[1], solver.mu_art[I] / ρ)
        largest[2] = max(largest[2], solver.beta_art[I] / ρ)
        largest[3] = max(largest[3], solver.kappa_art[I] / (ρ * cp))
    end
    T.dominant[argmax(largest)] += 1
    return nothing
end

function line_energy(data, L::Line)
    N = length(L.idx)
    ρ = zeros(N)
    ρe = zeros(N)
    q = zeros(L.nc)
    for i in 1:N
        for c in 1:L.nc
            q[c] = data[L.idx[i], c]
        end
        ρ[i] = density(q, L)
        ρe[i] = internal_energy(q, L, ρ[i])
    end
    return (; ρ, ρe)
end

bad(s, i) = s.ρ[i] <= 0 || s.ρe[i] <= 0

function record!(T, prev, pre, post)
    T.steps += 1
    count_post = 0
    for i in eachindex(post.ρe)
        bad_prev, bad_pre, bad_post = bad(prev, i), bad(pre, i), bad(post, i)
        T.bad_pre += bad_pre
        T.by_step += bad_pre & !bad_prev
        T.by_filter += bad_post & !bad_pre
        T.cured += bad_pre & !bad_post
        count_post += bad_post
    end
    T.bad_post += count_post
    T.max_bad = max(T.max_bad, count_post)
    T.min_pre = min(T.min_pre, minimum(pre.ρe))
    T.min_post = min(T.min_post, minimum(post.ρe))
    return nothing
end

# The fronts of bench/shockfoot.jl, each as (index, direction into the
# undisturbed gas).
function fronts(case, s)
    if case === :noh
        f = findlast(>(2.5), s.ρ)
        return f === nothing ? Tuple{Int,Int}[] : [(f, +1)]
    end
    threshold = sqrt(100 * WC_AMBIENT)
    low = findall(<(threshold), 0.4 .* s.ρe)
    isempty(low) && return Tuple{Int,Int}[]
    left, right = first(low), last(low)
    right - left < 4 && return Tuple{Int,Int}[]
    return [(left - 1, +1), (right + 1, -1)]
end

# Where the faces limited in this step sit: face slot j + 1 lies between
# nodes j and j + 1, at j + ½.
function record_faces!(T, case, limited, s)
    fs = fronts(case, s)
    for slot in eachindex(limited)
        limited[slot] || continue
        x = slot - 0.5
        best = nothing
        for (f, dir) in fs
            offset = dir * (x - f)
            (best === nothing || abs(offset) < abs(best)) && (best = offset)
        end
        if best === nothing || best <= 0
            T.behind += 1
        else
            T.ahead_min = min(T.ahead_min, best)
            T.ahead_max = max(T.ahead_max, best)
        end
    end
    return nothing
end

# --- the limited stage ----------------------------------------------------------

# The Euler flux along dimension 1 and |u| + c at every node of `q`.
function euler_fluxes!(S::Scratch, q, L::Line)
    for i in axes(q, 1)
        ρ = 0.0
        for s in 1:L.ns
            ρ += q[i, s]
        end
        ρ = max(ρ, EPS)
        u = q[i, L.im[1]] / ρ
        ρe = q[i, L.ie] - 0.5 * (q[i, L.im[1]]^2 + q[i, L.im[2]]^2 +
                                 q[i, L.im[3]]^2) / ρ
        p = (L.γ - 1) * ρe
        for s in 1:L.ns
            S.flux[i, s] = q[i, s] * u
        end
        for d in 1:3
            S.flux[i, L.im[d]] = q[i, L.im[d]] * u
        end
        S.flux[i, L.im[1]] += p
        S.flux[i, L.ie] = (q[i, L.ie] + p) * u
        S.speed[i] = abs(u) + sqrt(L.γ * max(p, 0.0) / ρ)
    end
    return S
end

# Apply the face corrections δ = (θ − 1)(GH − GL) to the line of `data` (and of
# `du` scaled by 1/B when the register is kept), cell i taking δ / weight[i],
# and return the `weight`-weighted mass and energy they changed.
function correct!(data, du, B, S::Scratch, L::Line, weight)
    N = length(L.idx)
    mass, energy = 0.0, 0.0
    for j in 1:N-1
        θ = S.θ[j+1]
        θ < 1 || continue
        a, b = L.idx[j], L.idx[j+1]
        for c in 1:L.nc
            δ = (θ - 1) * (S.GH[j+1, c] - S.GL[j+1, c])
            old_a, old_b = data[a, c], data[b, c]
            data[a, c] -= δ / weight[j]
            data[b, c] += δ / weight[j+1]
            if du !== nothing
                du[a, c] -= δ / (weight[j] * B)
                du[b, c] += δ / (weight[j+1] * B)
            end
            change = weight[j] * (data[a, c] - old_a) +
                     weight[j+1] * (data[b, c] - old_b)
            c <= L.ns && (mass += change)
            c == L.ie && (energy += change)
        end
    end
    return mass, energy
end

# Per-face θ for increments whose half states are base ∓ 2 G / weight, from the
# rows of S.GH and S.GL. Ends: the boundary faces are not limited; a cell whose
# half state across one is inadmissible is counted unguaranteed.
function face_thetas!(T, S::Scratch, base, weight, L::Line, filter::Bool)
    N = length(L.idx)
    limited = 0
    unguaranteed = 0
    S.θ[1] = 1.0
    S.θ[N+1] = 1.0
    for j in 1:N-1
        θ = 1.0
        if L.constrained[j]
            t, ok = side_theta(view(base, j, :), S.GH, S.GL, j + 1, -2 / weight[j], S, L)
            θ = min(θ, t)
            unguaranteed += !ok
        end
        if L.constrained[j+1]
            t, ok = side_theta(view(base, j + 1, :), S.GH, S.GL, j + 1, 2 / weight[j+1],
                               S, L)
            θ = min(θ, t)
            unguaranteed += !ok
        end
        S.θ[j+1] = θ
        if θ < 1
            limited += 1
            filter || (S.limited[j+1] = true)
        end
    end
    for (i, slot, s) in ((1, 1, 2.0), (N, N + 1, -2.0))
        L.constrained[i] || continue
        for c in 1:L.nc
            S.q[c] = base[i, c] + s * S.GH[slot, c] / weight[i]
        end
        unguaranteed += !admissible(S.q, L)
    end
    if filter
        T.filter_faces += N - 1
        T.filter_limited += limited
        T.filter_unguaranteed += unguaranteed
    else
        T.stage_faces += N - 1
        T.stage_limited += limited
        T.stage_unguaranteed += unguaranteed
    end
    return limited
end

function limit_stage!(T, solver, data, du, dQ, stage, dt, S::Scratch, L::Line, store,
                      own)
    N = length(L.idx)
    A, B = CL.RKA[stage], CL.RKB[stage]
    τ = STAGE_ADVANCE[stage] * dt
    # The face register Φ_k = A_k Φ_{k−1} + dt F̂_k at both boundary faces, whose
    # flux is the point flux of the end node; the running sum of B_k du_k from
    # node 1 is then B_k Φ_k at every face, and its value at the far face
    # against the far register measures the face form.
    for c in 1:L.nc
        S.register[c] = A * S.register[c] + dt * solver.flux[1, c][L.idx[1]]
        S.register_far[c] = A * S.register_far[c] + dt * solver.flux[1, c][L.idx[N]]
        S.GH[1, c] = B * S.register[c]
        for j in 1:N
            S.GH[j+1, c] = S.GH[j, c] - L.W[j] * B * du[L.idx[j], c]
        end
        scale = max(maximum(abs, view(S.GH, :, c)), 1e-300)
        T.closure_mismatch = max(T.closure_mismatch,
                                 abs(S.GH[N+1, c] - B * S.register_far[c]) / scale)
    end
    euler_fluxes!(S, S.base, L)
    for j in 1:N-1
        α = max(S.speed[j], S.speed[j+1])
        for c in 1:L.nc
            S.GL[j+1, c] = τ * (0.5 * (S.flux[j, c] + S.flux[j+1, c]) -
                                0.5 * α * (S.base[j+1, c] - S.base[j, c]))
        end
        cfl_a, cfl_b = 2τ * α / L.W[j], 2τ * α / L.W[j+1]
        if j == 1 || j + 1 == N
            T.lf_cfl_edge = max(T.lf_cfl_edge, j == 1 ? cfl_a : cfl_b)
            T.lf_cfl_inner = max(T.lf_cfl_inner, j == 1 ? cfl_b : cfl_a)
        else
            T.lf_cfl_inner = max(T.lf_cfl_inner, cfl_a, cfl_b)
        end
    end
    if own
        # The first-order target keeps the history: GL ← GH − B dt F̂_k + τ F_LF,
        # with F̂_k the running sum of −dQ from the flux of node 1.
        for c in 1:L.nc
            F = solver.flux[1, c][L.idx[1]]
            for j in 1:N-1
                F -= L.W[j] * parent(dQ)[L.idx[j], c]
                S.GL[j+1, c] += S.GH[j+1, c] - B * dt * F
            end
        end
    end
    face_thetas!(T, S, S.base, L.W, L, false) > 0 || return nothing
    mass, energy = correct!(data, store ? du : nothing, B, S, L, L.W)
    T.correction_mass += mass
    T.correction_energy += energy
    return nothing
end

# `step!` with the stage limiter between the low-storage update and the next
# stage; with `stage = false` the same calls as `step!` in the same order.
function limited_step!(T, solver, Q, ws, dt, prepared, S::Scratch, L::Line, v)
    data, du = parent(Q), parent(ws.du)
    nc = L.nc
    for stage in 1:5
        solver.tstage = solver.t + oftype(solver.t, CL.RKC[stage]) * dt
        first_prepared = prepared && stage == 1
        first_prepared || CL.apply_bcs!(solver, Q)
        CL.compute_rhs!(solver, Q, ws.dQ, first_prepared)
        v.stage && read_line!(S.base, data, L)
        CL._rk_update!(solver.decomp, nc, Q, ws.dQ, ws.du, CL.RKA[stage], CL.RKB[stage],
                       dt)
        if v.stage
            T.limiter_wall += @elapsed limit_stage!(T, solver, data, du, ws.dQ, stage,
                                                    dt, S, L, v.store, v.own)
            s = line_energy(data, L)
            T.stage_bad += count(i -> bad(s, i), eachindex(s.ρ))
        end
    end
    solver.tstage = solver.t + dt
    CL.apply_bcs!(solver, Q)
    CL._validate_transport_state!(solver, Q)
    return Q
end

# --- the limited filter pass ----------------------------------------------------

function limited_filter!(T, solver, Q, S::Scratch, L::Line, v)
    data = parent(Q)
    v.filter && read_line!(S.base, data, L)
    solver.filter_interval = 1
    wf = CL.filter_weight(solver, 1)
    filter_state!(solver, Q)
    solver.filter_interval = 0
    v.filter || return nothing
    T.limiter_wall += @elapsed begin
        N = length(L.idx)
        M = length(L.psi) ÷ 2
        α = L.alpha_f
        margin = 24
        for c in 1:L.nc
            # Φ_{j+½} = −Σ_{i≤j} ω_i Δ_i, so that Δ_i = Φ_{i−½} − Φ_{i+½}.
            S.GH[1, c] = 0.0
            for j in 1:N
                S.GH[j+1, c] = S.GH[j, c] - L.ω[j] * (data[L.idx[j], c] - S.base[j, c])
            end
            # Inside the line A Φ + w_f ψ is one constant C; Φ − C/(1 + 2α) is
            # the face flux of the pass, and the closed ends' defect is left on
            # the boundary faces.
            C = [α * S.GH[j, c] + S.GH[j+1, c] + α * S.GH[j+2, c] +
                 wf * sum(L.psi[l+M] * S.base[j+l, c] for l in 1-M:M)
                 for j in margin:N-margin]
            mid = median(C)
            scale = max(maximum(abs, view(S.base, :, c)), 1e-300)
            T.offset_spread = max(T.offset_spread, maximum(x -> abs(x - mid), C) / scale)
            offset = mid / (1 + 2α)
            for j in 1:N+1
                S.GH[j, c] -= offset
                S.GL[j, c] = 0.0
            end
        end
        if face_thetas!(T, S, S.base, L.ω, L, true) > 0
            # Conservative under ω = W / h, reported under W.
            mass, energy = correct!(data, nothing, 1.0, S, L, L.ω)
            T.correction_mass += mass * L.W[N÷2]
            T.correction_energy += energy * L.W[N÷2]
        end
    end
    return nothing
end

# --- one run --------------------------------------------------------------------

function variant_flags(label)
    parts = split(label, '@')
    name = parts[1]
    eps_ratio = length(parts) > 1 ? parse(Float64, parts[2]) : 0.0
    v = flags(name)
    return (; v..., name=label, eps_ratio)
end

function flags(name)
    name == "none" && return (; stage=false, store=true, own=false, filter=false)
    name == "A" && return (; stage=true, store=true, own=false, filter=false)
    name == "A-nostore" && return (; stage=true, store=false, own=false, filter=false)
    name == "A-rhs" && return (; stage=true, store=true, own=true, filter=false)
    name == "B" && return (; stage=false, store=true, own=false, filter=true)
    name == "A+B" && return (; stage=true, store=true, own=false, filter=true)
    name == "A-rhs+B" && return (; stage=true, store=true, own=true, filter=true)
    throw(ArgumentError("unknown variant '$name', want one of " *
                        "$(join(VARIANTS, ", ")), each optionally followed by @ratio"))
end

totals(data, L::Line) =
    (sum(L.W[i] * sum(data[L.idx[i], s] for s in 1:L.ns) for i in eachindex(L.idx)),
     sum(L.W[i] * data[L.idx[i], L.ie] for i in eachindex(L.idx)))

# The `run!` loop of a single-patch solver with no regrid, rollback, failsafe,
# callback or landing, with the filter pass after the step.
function limited_run(case, v; nmax, weights=nothing)
    solver, Q = build(case)
    L = line_geometry(solver, Q, case; weights, eps_ratio=v.eps_ratio)
    S = Scratch(L)
    ws = CL.Workspace(Q)
    control = solver.control
    T = Tally()
    data = parent(Q)
    mass0, energy0 = totals(data, L)
    tfin = oftype(solver.t, end_time(case))
    status = "completed"
    wall = @elapsed begin
        CL._prime_coefficients!(solver, Q, ws)
        dt_seen = 0.0
        prev = line_energy(data, L)
        while solver.t < tfin && solver.step < nmax
            solver.tstage = solver.t
            CL.apply_bcs!(solver, Q)
            rate, rho_min, filter_rate = CL.max_rate(solver, Q)
            record_rate!(T, solver, rate, filter_rate[1], L)
            dt = CL.predicted_dt(solver, control, rate)
            failure = CL.check_step(control, dt, rho_min, dt_seen, solver.step, solver.t,
                                    solver.cfl)
            if failure !== nothing
                status = "failed: $(failure.reason) @ $(failure.step)"
                break
            end
            dt_seen = max(dt_seen, dt)
            dt = min(dt, tfin - solver.t)
            CL._advances(solver.t, dt) || break
            fill!(S.register, 0.0)
            fill!(S.register_far, 0.0)
            fill!(S.limited, false)
            limited_step!(T, solver, Q, ws, dt, true, S, L, v)
            solver.t += dt
            solver.step += 1
            solver.dt_prev = dt
            solver.rate_prev = rate
            solver.filter_rate_prev = filter_rate
            pre = line_energy(data, L)
            any(S.limited) && record_faces!(T, case, S.limited, pre)
            limited_filter!(T, solver, Q, S, L, v)
            post = line_energy(data, L)
            record!(T, prev, pre, post)
            prev = post
        end
    end
    solver.t < tfin && status == "completed" && (status = "stopped @ $(solver.step)")
    mass1, energy1 = totals(data, L)
    return (; case, v, status, wall, T, solver, Q, L,
            mass=(mass1 - mass0) / mass0, energy=(energy1 - energy0) / energy0,
            corr_mass=T.correction_mass / mass0, corr_energy=T.correction_energy / energy0)
end

# --- reports --------------------------------------------------------------------

const WC_REFERENCE = Ref{Any}(nothing)

function wc_reference()
    WC_REFERENCE[] === nothing || return WC_REFERENCE[]
    xs, ρs = Float64[], Float64[]
    for line in eachline(joinpath(@__DIR__, "..", "test", "refs", "woodward_colella.csv"))
        (isempty(line) || startswith(line, '#')) && continue
        tokens = split(line, ',')
        push!(xs, parse(Float64, tokens[1]))
        push!(ρs, parse(Float64, tokens[2]))
    end
    return WC_REFERENCE[] = (xs, ρs)
end

function metric(case, solver, Q)
    xs, ρ, _, _ = case_line_profile(solver, Q)
    if case === :woodward
        xr, ρr = wc_reference()
        imax = argmax(ρ)
        return @sprintf("L1 rho %.4e, peak %.4f at x = %.4f",
                        l1(ρ, [interp1(xr, ρr, x) for x in xs]), ρ[imax], xs[imax])
    end
    plateau, deficit, shock, _ = noh_metrics(xs, ρ, 1)
    return @sprintf("plateau %.4f, wall deficit %.1f%%, shock %.4f", plateau,
                    100 * deficit, shock)
end

fraction(a, b) = b == 0 ? 0.0 : a / b
range_text(T) = T.ahead_max < T.ahead_min ? "-" :
                @sprintf("%.1f-%.1f", T.ahead_min, T.ahead_max)

const TABLE_HEAD = Printf.Format("\n  %-9s %-10s %-14s %5s %7s %7s %4s %6s %6s %6s " *
                                 "%10s %10s %8s %5s %8s %5s %10s %5s\n")
const TABLE_ROW = Printf.Format("  %-9s %-10s %-14s %5d %7d %7d %4d %6d %6d %6d " *
                                "%10.3e %10.3e %8.2e %5d %8.2e %5d %10s %5d\n")

function print_table(rows)
    Printf.format(stdout, TABLE_HEAD, "case", "variant", "end", "steps", "bad pre",
                  "bd post", "max", "bystep", "byfilt", "stage", "min pre", "min post",
                  "A faces", "unguA", "B faces", "unguB", "A ahead", "behnd")
    for r in rows
        T = r.T
        Printf.format(stdout, TABLE_ROW, r.case, r.v.name, first(r.status, 14), T.steps,
                      T.bad_pre, T.bad_post, T.max_bad, T.by_step, T.by_filter,
                      T.stage_bad, T.min_pre, T.min_post,
                      fraction(T.stage_limited, T.stage_faces), T.stage_unguaranteed,
                      fraction(T.filter_limited, T.filter_faces), T.filter_unguaranteed,
                      range_text(T), T.behind)
    end
    println()
    for r in rows
        T = r.T
        text = completed(r.solver, end_time(r.case)) ? metric(r.case, r.solver, r.Q) :
               "did not reach the end time"
        @printf("  %-9s %-10s %s\n", r.case, r.v.name, text)
        @printf("  %-9s %-10s mass %+.3e (corrections %+.1e), energy %+.3e \
                 (corrections %+.1e); %.1f s, limiter %.1f s\n", "", "", r.mass,
                r.corr_mass, r.energy, r.corr_energy, r.wall, T.limiter_wall)
        @printf("  %-9s %-10s far-face mismatch %.1e, filter offset spread %.1e, \
                 first-order CFL 2τα/W end %.3f inside %.3f\n", "", "",
                T.closure_mismatch, T.offset_spread, T.lf_cfl_edge, T.lf_cfl_inner)
        @printf("  %-9s %-10s mean rate / (|u| + c)/h %.3f; diffusive-bound steps %d \
                 (largest μ*/ρ %d, β*/ρ %d, κ*/(ρ c_p) %d)\n", "", "",
                T.rate_ratio / max(T.steps, 1), T.diffusive, T.dominant...)
    end
    return nothing
end

# --- the parts ------------------------------------------------------------------

function weights_part(cases)
    println("\n=== weights: the divergence plan's face form ===")
    scheme = lele_d1_6()
    out = Dict{Symbol,Vector{Float64}}()
    for case in cases
        solver, Q = build(case)
        N = solver.decomp.n_local[1]
        idx = [padded_index(solver, i, 1, 1) for i in 1:N]
        D = divergence_matrix(solver, Q, idx)
        W, residual = closure_weights(D)
        out[case] = W
        h = xcoord(solver, 1, 2) - xcoord(solver, 1, 1)
        x = [xcoord(solver, 1, i) for i in 1:N]
        @printf("  %s, N = %d: least-squares residual %.2e, inner W/h - 1 = %.1e\n",
                case, N, residual, W[N÷2] / h - 1)
        low = (@sprintf("%.4f", W[i] / h) for i in 1:8)
        high = (@sprintf("%.4f", W[i] / h) for i in N:-1:N-7)
        println("    W/h at the low end : ", join(low, "  "))
        println("    W/h at the high end: ", join(high, "  "))
        @printf("    min W/h %.4f, max W/h %.4f, max |W/h - 1| beyond 12 nodes %.1e\n",
                minimum(W) / h, maximum(W) / h,
                maximum(abs(W[i] / h - 1) for i in 13:N-12))
        for (name, f) in (("x^3 - 2x^2 + x", @. x^3 - 2x^2 + x),
                          ("sin(7x) + x^5", @. sin(7x) + x^5),
                          ("random", pseudo_random(N)))
            Df = D * f
            rel, far = face_relation_residual(scheme, f, Df, W)
            @printf("    %-16s |Σ W Df - (f_N - f_1)| %.1e, face relation %.1e, \
                     far face %.1e\n", name, abs(sum(W .* Df) - (f[end] - f[1])), rel, far)
        end
    end
    return out
end

# A deterministic pseudo-random field in [−½, ½), from a linear congruential
# generator, so the script needs no Random.
function pseudo_random(N::Int)
    s = UInt64(0x9e3779b97f4a7c15)
    out = zeros(N)
    for i in 1:N
        s = s * 0x5851f42d4c957f2d + 0x14057b7ef767814f
        out[i] = (s >> 11) / 2.0^53 - 0.5
    end
    return out
end

function filter_pass!(solver, Q)
    solver.filter_interval = 1
    filter_state!(solver, Q)
    solver.filter_interval = 0
    return nothing
end

function interior_equal(a, b, L::Line)
    worst = 0.0
    for c in 1:L.nc, I in L.idx
        worst = max(worst, abs(a[I, c] - b[I, c]))
    end
    return worst
end

function identity_part(cases, steps, weights)
    println("\n=== identity: this loop, unlimited, against run! ===")
    for case in cases
        r = limited_run(case, variant_flags("none"); nmax=steps,
                        weights=get(weights, case, nothing))
        solver, Q = build(case)
        run!(solver, Q; tfinal=end_time(case), nmax=steps,
             callback=Callback(EveryStep(1), (s, q) -> (filter_pass!(s, q); nothing)))
        @printf("  %-9s %d steps: max |Q - Q_run!| %.3e, t %.17g / %.17g\n", case,
                solver.step, interior_equal(parent(r.Q), parent(Q), r.L), r.solver.t,
                solver.t)
    end
end

function pulse_part()
    println("\n=== pulse: a smooth acoustic pulse, unlimited and A + B ===")
    a = limited_run(:pulse, variant_flags("none"); nmax=typemax(Int))
    b = limited_run(:pulse, variant_flags("A+B"); nmax=typemax(Int), weights=a.L.W)
    @printf("  %d / %d steps to t = %.3f; faces limited: stage %d, filter %d; \
             max |Q_A+B - Q| %.3e; first-order CFL end %.3f inside %.3f\n",
            a.T.steps, b.T.steps, b.solver.t, b.T.stage_limited, b.T.filter_limited,
            interior_equal(parent(a.Q), parent(b.Q), a.L), b.T.lf_cfl_edge,
            b.T.lf_cfl_inner)
    @printf("  far-face mismatch %.1e, filter offset spread %.1e\n",
            b.T.closure_mismatch, b.T.offset_spread)
end

function main(args)
    opt = CL.script_args(args, (part="all", cases="woodward,noh",
                                variants=join(VARIANTS, ","), nmax=100_000,
                                identity_steps=300))
    parts = opt.part == "all" ? collect(PARTS) : split(opt.part, ',')
    cases = Symbol.(split(opt.cases, ','))
    for p in parts
        p in PARTS ||
            throw(ArgumentError("unknown part '$p', want one of $(join(PARTS, ", "))"))
    end
    for c in cases
        c in CASES ||
            throw(ArgumentError("unknown case '$c', want one of $(join(CASES, ", "))"))
    end
    variants = variant_flags.(split(opt.variants, ','))
    weights = "weights" in parts ? weights_part(cases) : Dict{Symbol,Vector{Float64}}()
    "identity" in parts && identity_part(cases, opt.identity_steps, weights)
    "pulse" in parts && pulse_part()
    "variants" in parts || return nothing
    println("\n=== variants ===")
    rows = []
    for case in cases, v in variants
        r = limited_run(case, v; nmax=opt.nmax, weights=get(weights, case, nothing))
        @printf("  %s %s: %s, %d steps in %.1f s\n", case, v.name, r.status, r.T.steps,
                r.wall)
        flush(stdout)
        push!(rows, r)
    end
    print_table(rows)
    return nothing
end

main(ARGS)
