# A conservative positivity limiter in face-flux form on the one-dimensional
# Woodward–Colella and planar Noh (ν = 1) cases of test/cases.jl: whether it
# removes the points with ρ ≤ 0 or ρe = E − ½|m|²/ρ ≤ 0, what it does to the
# validation metrics, and how often it acts.
#
#   julia --project=. -t 1 bench/positivity.jl                     # every part
#   julia --project=. -t 1 bench/positivity.jl part=weights,pulse
#   julia --project=. -t 1 bench/positivity.jl part=variants cases=noh variants=none,A+B
#   julia --project=. -t 1 bench/positivity.jl part=src
#   julia --project=. -t 1 bench/positivity.jl part=src cases=sedov,noh2,noh3
#   julia --project=. -t 1 bench/positivity.jl part=rz N=96
#   julia --project=. -t 1 bench/positivity.jl part=cost cases=woodward,noh,sedov,rz,blast N=64
#   julia --project=. -t 1 bench/positivity.jl part=patches patches=2,3
#   julia --project=. -t 1 bench/positivity.jl part=cost cases=woodward,blast N=80 patches=2
#   julia --project=. -t 1 bench/positivity.jl part=levels levels=wc,wcsub,noh43
#   julia --project=. -t 1 bench/positivity.jl part=levelcost levels=wc,wcsub,noh85,tiles,a19
#   julia --project=. -t 1 bench/positivity.jl part=levels levels=a19,a19tiles a19n=33
#   julia --project=. -t 1 bench/positivity.jl part=levels levels=pnoh81tiles,pnoh81regrid
#   julia --project=. -t 1 bench/positivity.jl part=tilecost levels=a19,a19tiles a19n=33
#   julia --project=. -t 1 bench/positivity.jl part=tiledrift
#   julia --project=. -t 1 bench/positivity.jl part=tiledrift ledger=true conservative=false
#   julia --project=. -t 1 bench/positivity.jl part=levels levels=wc,noh81tiles flux=ghost
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
#   src       each case through `run!` with `Numerics(positivity_limiter)`
#             off and on, the package's form of A+B@0.01 with each cell's
#             bound at most 1% of its own value: the points with ρ ≤ 0 or
#             ρe ≤ 0 after each step, the metric, the steps, what the limiter
#             did and the wall time. It also takes the radial cases of
#             test/cases.jl, Sedov and Noh ν = 2, 3, folded at r = 0, whose
#             metric is the validation battery's
#   rz        Sedov of test/cases.jl on the r-z plane, N × N nodes over the
#             quarter plane folded at the axis and by a symmetry plane at z = 0,
#             limiter off and on, against the spherical line on N nodes with
#             the limiter on: the shock radius (the outermost crossing of
#             ρ = 2) along the axis, along the plane and along the diagonal,
#             the peak density, the inadmissible point-steps, and the stage
#             faces limited at the axis. Not in `part=all`; about two minutes
#             at N = 96
#   cost      the wall time per step with the limiter off and on over the
#             first `steps` steps of each case in `cases`, from fresh solvers,
#             `reps` off/on pairs alternated in one process: the medians and
#             the median of the pairs' ratios. `cases` takes the radial cases
#             here too, `rz` (the Sedov of part=rz on N × N nodes) and `blast`
#             (the same deposit at the centre of a periodic square of N × N
#             nodes). Not in `part=all`. `patches` > 1 splits the Cartesian
#             cases into that many same-level patches along x
#   patches   Woodward–Colella and planar Noh on one patch and on each count
#             of same-level patches in `patches`, along x under
#             `patch_interfaces = flux` (`:closure` unless `flux=ghost`, which
#             sets `LIMITED_GHOST_INTERFACES` so that a limited run keeps it;
#             the same for `cost` and the level parts below), limiter off and
#             on: the steps, the
#             inadmissible point-steps, the unguaranteed sides and the L1
#             density difference from the one-patch run in the same state, and
#             the one-patch runs' own difference after a relative change of
#             1e-15 to the initial energy. Not in `part=all`
#   levels    refined levels under `patch_interfaces = flux`, limiter off
#             and on, each case of `levels` to its end time: the steps, the
#             inadmissible point-steps after every step, split into the nodes
#             of a refined patch's coarse-fine plane (the parent's shell) and
#             the rest, the limiter's counts and the case's metrics. `wc` and
#             `wcsub`: Woodward–Colella on 200 nodes with a box over root nodes
#             111:170, at the global step and subcycled; `wcregrid`: a
#             regridded box and regridded tiles of edge 8 following the density
#             tags, the tiles also subcycled; `noh85`, `noh43` (and `noh85sub`,
#             `noh43sub`): the cylindrical Noh of test/validation.jl's level
#             rows on a box at the axis over that many root nodes, and
#             `nohuniform`, the uniform run at the level's spacing; `noh81`
#             and `noh81tiles`, `noh81regrid` (each also with `sub`): a box
#             over 81 nodes, ten lattice cells of edge 8, and the same as
#             static tiles of edge 8 and as tiles regridded every 10 steps;
#             `pnoh81tiles`, `pnoh81regrid` (each also with `sub`): the planar
#             Noh (ν = 1) on the same 256 root nodes and the same tiles;
#             `a19`: the converging shock of the r-z plane, `a19n`² nodes
#             (64), with a box at the corner of the axis and the plane, to
#             t = 0.165, and `a19tiles`, `a19regrid` (each also with `sub`)
#             with tiles of edge 8 over the lattice cells within that box,
#             static and regridded (where a19n − 1 is not a multiple of 8 the
#             lattice cells at the outer faces are partial, too short for the
#             limiter, and a limited regrid leaves them to the root); `tiles`
#             and `tilessub`: a blast in a 64²
#             square with tiles of edge 8 over its centre, to t = 0.25. Not in
#             `part=all`
#   levelcost the wall time per step of the same cases' first steps, as
#             `cost` takes it. Not in `part=all`
#   tilecost  the same for two cases of `levels`, a box and then tiles, the
#             four runs alternated in one process: the limited tiled step over
#             the unlimited one and over the limited box. Not in `part=all`
#   tiledrift the cylindrical Noh of `noh81` under each variant of `drift`: a
#             layout (`box`, `tiles8`, `tiles16`, `tiles80`, `regrid8`,
#             `uniform`), optionally followed after a hyphen by one change
#             (`sub`, `ghost`, `noreflux`, `clamp`, `nofilter`, `lim`,
#             `ghostlim`; see `drift_variant`), unlimited and under
#             `:closure` unless the change says otherwise; `ledger=true` adds
#             each cylindrical run's mass drift attributed to the mechanism
#             that made it (`:rhs`, `:filter`, ...) by the budget ledger of
#             src/ledger.jl. Each run prints the metrics of `levels`, the
#             composite mass drift beside two forms that use no mask, the L1
#             density error over the level's region against the exact solution
#             and the uniform run, and the drift's history; then the planar
#             Noh (ν = 1) on the same layouts under each variant of `planar`,
#             and a smooth pulse at rest on the cylindrical line under each
#             variant of `pulse`. Not in `part=all`
#
# `conservative=false` clears `CONSERVATIVE_INTERFACE_FILTER` for every part:
# the filter closes each same-level interface with the identity row in place
# of its conservative rows, the reference those rows are measured against.
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

const CASES = (:woodward, :noh, :sedov, :noh2, :noh3)
const RADIAL = (:sedov, :noh2, :noh3)
const PLANE_CASES = (:rz, :blast)
const PARTS = ("weights", "identity", "pulse", "variants", "src")
const VARIANTS = ("none", "A", "A-nostore", "A-rhs", "B", "A+B", "A-rhs+B")
const WC_AMBIENT = 0.01
# The patch and level interfaces' flux (`flux=`) of the parts `patches`, `cost`,
# `levels`, `levelcost` and `tilecost`.
const FLUX = Ref(:closure)
# Hu, Adams & Shu take ε = min(1e-13, the initial minimum) for ρ and for ρe.
# A variant named `name@r` takes ε = r times the initial minimum instead.
const EPS = 1e-13
# The time advance of each stage, c_{k+1} − c_k with c_6 = 1. Its sum is 1, so a
# step limited at every stage is a sequence of first-order steps of total dt.
const STAGE_ADVANCE = ntuple(k -> (k < 5 ? CL.RKC[k+1] : 1.0) - CL.RKC[k], 5)

# --- the cases ----------------------------------------------------------------

end_time(case) = case === :woodward ? WC_T : case === :noh ? NOH_T :
                 case === :sedov ? SEDOV_T : case === :noh2 ? NOH_T :
                 case === :noh3 ? NOH_T - Dict(NOH_T0)[3] : PULSE_T
case_gamma(case) = case === :noh ? NOH_G : 1.4

function woodward_problem(N=WC_N)
    h = 1.0 / (N - 1)
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
# `validity = :permissive` as in bench/shockfoot.jl. `interval = 1` gives the
# filter back to `run!`, and `limiter` sets `Numerics(positivity_limiter)`.
function build(case; interval=0, limiter=false, patches=1)
    filter = StateFilter(compact_filter(); cfl=0.35, interval)
    control = StepControl(validity=:permissive)
    art = ArtificialProperties(enabled=true)
    execution = Execution(patch_grid=(patches, 1, 1))
    case === :woodward &&
        return setup(woodward_problem(),
                     Numerics(n_global=(WC_N, 1, 1), art=art, cfl=0.3, filter=filter,
                              control=control, positivity_limiter=limiter,
                              execution=execution,
                              patch_interfaces=FLUX[]))
    case === :sedov &&
        return setup(sedov_problem(),
                     Numerics(n_global=(SEDOV_N, 1, 1), art=art, cfl=0.3, filter=filter,
                              control=control, positivity_limiter=limiter))
    if case in (:noh2, :noh3)
        ν = case === :noh2 ? 2 : 3
        return setup(noh_problem(ν),
                     Numerics(n_global=(Dict(NOH_N)[ν], 1, 1), art=art, cfl=NOH_CFL,
                              filter=filter, control=control, positivity_limiter=limiter))
    end
    case === :noh &&
        return setup(noh_problem(1),
                     Numerics(n_global=(Dict(NOH_N)[1], 1, 1), art=art, cfl=NOH_CFL,
                              filter=filter, control=control,
                              positivity_limiter=limiter, execution=execution,
                              patch_interfaces=FLUX[]))
    return setup(pulse_problem(),
                 Numerics(n_global=(PULSE_N, 1, 1), art=art, cfl=0.3, filter=filter,
                          control=control, positivity_limiter=limiter))
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
    if case === :sedov
        R = front_position(xs, ρ, 2.0)
        exact = sedov_shock_radius(SEDOV_E, SEDOV_T, 3, 1.4)
        return @sprintf("R_s %.4f (%+.2f%%), peak %.3f", R, 100 * (R / exact - 1),
                        maximum(ρ))
    end
    ν = case === :noh2 ? 2 : case === :noh3 ? 3 : 1
    plateau, deficit, shock, _ = noh_metrics(xs, ρ, ν)
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

function src_part(cases)
    println("\n=== src: run! with Numerics(positivity_limiter) ===")
    for case in cases, limiter in (false, true)
        solver, Q = build(case; interval=1, limiter)
        bad = Ref(0)
        count_bad = (s, q) -> (r = state_report(s, q);
                               bad[] += r.negative_density + r.inadmissible; nothing)
        wall = @elapsed run!(solver, Q; tfinal=end_time(case), callback=count_bad)
        text = completed(solver, end_time(case)) ? metric(case, solver, Q) :
               "did not reach the end time"
        @printf("  %-9s %-4s %5d steps, bad post %6d, %s, %.1f s\n", case,
                limiter ? "on" : "off", solver.step, bad[], text, wall)
        if limiter
            c = CL.positivity_counts(solver)
            @printf("  %-9s %-4s faces limited: stage %.2e, pass %.2e; unguaranteed \
                     sides %d\n", "", "", c.stage_limited / c.stage_faces,
                    c.filter_limited / c.filter_faces, c.unguaranteed)
        end
        flush(stdout)
    end
end

# The Sedov deposit of test/cases.jl in a plane: on the r-z quarter plane
# folded at the axis and by a symmetry plane at z = 0 (`:rz`), or at the
# centre of a periodic square (`:blast`), which the limiter acts on in two
# Cartesian directions from the first steps.
function plane_problem(case)
    R = 1.2
    γ = 1.4
    pin = SEDOV_E * (γ - 1) / (π^1.5 * SEDOV_S^3)
    gas = IdealSpecies("gas"; gamma=γ, R=1.0)
    case === :blast &&
        return Problem(eos=gas, transport=ConstantTransport(mu0=0.0),
                       domain=((0.0, R), (0.0, R), (0.0, 1.0)), bcs=per3,
                       ic=(x, y, z) -> Prim(rho=1.0, u=(0.0, 0.0, 0.0),
                                            p=1e-5 + pin * exp(-((x - R / 2)^2 +
                                                                 (y - R / 2)^2) /
                                                               SEDOV_S^2)))
    return Problem(eos=gas, transport=ConstantTransport(mu0=0.0),
                   metric=CylindricalMetric(), domain=((0.0, R), (0.0, 1.0), (0.0, R)),
                   bcs=((AxisBC(), SlipWallBC()), per3[2], (SymmetryPlaneBC(), SlipWallBC())),
                   ic=(r, θ, z) -> Prim(rho=1.0, u=(0.0, 0.0, 0.0),
                                        p=1e-5 + pin * exp(-(r^2 + z^2) / SEDOV_S^2)))
end

plane_numerics(on, n; patches=1) =
    Numerics(n_global=n, art=ArtificialProperties(enabled=true), cfl=0.3,
             filter=StateFilter(compact_filter(); cfl=0.35),
             control=StepControl(validity=:permissive), positivity_limiter=on,
             execution=Execution(patch_grid=(patches, 1, 1)), patch_interfaces=FLUX[])

# Sedov on the r-z quarter plane and on the spherical line at the same spacing.
function rz_part(N)
    println("\n=== rz: Sedov on the r-z plane against the spherical line ===")
    prob = plane_problem(:rz)
    exact = sedov_shock_radius(SEDOV_E, SEDOV_T, 3, 1.4)
    for on in (false, true)
        solver, Q = setup(prob, plane_numerics(on, (N, 1, N)))
        bad = Ref(0)
        count_bad = (s, q) -> (r = state_report(s, q);
                               bad[] += r.negative_density + r.inadmissible; nothing)
        wall = @elapsed run!(solver, Q; tfinal=SEDOV_T, callback=count_bad)
        ρ(i, k) = Q[padded_index(solver, i, 1, k), 1]
        x(i) = xcoord(solver, 1, i)
        z(k) = xcoord(solver, 3, k)
        along_r = front_position([x(i) for i in 1:N], [ρ(i, 1) for i in 1:N], 2.0)
        along_z = front_position([z(k) for k in 1:N], [ρ(1, k) for k in 1:N], 2.0)
        diagonal = front_position([hypot(x(i), z(i)) for i in 1:N],
                                  [ρ(i, i) for i in 1:N], 2.0)
        @printf("  r-z %d², %-3s %5d steps, bad %6d, R_s along r %.4f, along z %.4f, \
                 diagonal %.4f (exact %.4f), peak ρ %.3f, %.1f s\n", N,
                on ? "on" : "off", solver.step, bad[], along_r, along_z, diagonal, exact,
                maximum(ρ(i, k) for i in 1:N, k in 1:N), wall)
        if on
            c = CL.positivity_counts(solver)
            @printf("  %24s faces limited: stage %.2e, pass %.2e; unguaranteed \
                     sides %d; axis faces limited at %d of %d line-stages\n", "",
                    c.stage_limited / c.stage_faces, c.filter_limited / c.filter_faces,
                    c.unguaranteed, c.axis_limited, 5 * solver.step * N)
        end
        flush(stdout)
    end
    solver, Q = setup(sedov_problem(), plane_numerics(true, (N, 1, 1)))
    run!(solver, Q; tfinal=SEDOV_T)
    xs, ρs, _, _ = case_line_profile(solver, Q)
    @printf("  sphere %d, on: %5d steps, R_s %.4f, peak ρ %.3f\n", N, solver.step,
            front_position(xs, ρs, 2.0), maximum(ρs))
end

# The wall time per step off and on, over the first `steps` steps from fresh
# solvers, `reps` pairs alternated after a compiling run of each.
function cost_part(cases, N, steps, reps, patches)
    println("\n=== cost: wall time per step, limiter off and on, $patches patch(es) ===")
    for case in cases
        plane = case in PLANE_CASES
        make(on) = !plane ? build(case; interval=1, limiter=on, patches) :
                   setup(plane_problem(case),
                         plane_numerics(on, case === :rz ? (N, 1, N) : (N, N, 1);
                                        patches))
        tfinal = plane ? SEDOV_T : end_time(case)
        for on in (false, true)
            solver, Q = make(on)
            run!(solver, Q; tfinal, nmax=3)
        end
        per_step = Dict(false => Float64[], true => Float64[])
        for _ in 1:reps, on in (false, true)
            solver, Q = make(on)
            wall = @elapsed run!(solver, Q; tfinal, nmax=steps)
            push!(per_step[on], wall / solver.step)
        end
        ratios = per_step[true] ./ per_step[false]
        @printf("  %-9s %s: off %.3f ms, on %.3f ms per step; on/off %.2f (pairs %s)\n",
                case, plane ? "$(N)²" : "line", 1e3 * median(per_step[false]),
                1e3 * median(per_step[true]), median(ratios),
                join((@sprintf("%.2f", r) for r in ratios), ", "))
        flush(stdout)
    end
end

# Each line of the density along x, the shared node of two patches once.
function line_density(solver, Q)
    points = Dict{Float64,Float64}()
    for (ps, q) in (Q isa Vector ? CL.eachpatch(solver, Q) : ((solver, Q),)),
        i in 1:ps.decomp.n_local[1]
        points[xcoord(ps, 1, i)] = q[padded_index(ps, i, 1, 1), 1]
    end
    return [points[x] for x in sort(collect(keys(points)))]
end

function patches_part(counts)
    println("\n=== patches: same-level patches along x, interface_flux = :$(FLUX[]) ===")
    for case in (:woodward, :noh), limiter in (false, true)
        function run_case(patches; perturb=false)
            solver, Q = build(case; interval=1, limiter, patches)
            if perturb
                for i in 1:solver.decomp.n_local[1]
                    Q[padded_index(solver, i, 1, 1), 5] *= 1 + 1e-15
                end
            end
            bad = Ref(0)
            count_bad = (s, q) -> (r = state_report(s, q);
                                   bad[] += r.negative_density + r.inadmissible; nothing)
            run!(solver, Q; tfinal=end_time(case), callback=count_bad)
            return solver, line_density(solver, Q), bad[]
        end
        ref, ρref, _ = run_case(1)
        L1(ρ) = sum(abs, ρ .- ρref) / sum(abs, ρref)
        _, ρp, _ = run_case(1; perturb=true)
        @printf("  %-9s %-3s one patch: %5d steps; after a 1e-15 change, L1 %.2e\n", case,
                limiter ? "on" : "off", ref.step, L1(ρp))
        for P in counts
            solver, ρ, bad = run_case(P)
            @printf("  %-9s %-3s %d patches: %5d steps, bad %6d, L1 against one patch \
                     %.2e", case, limiter ? "on" : "off", P, solver.step, bad, L1(ρ))
            limiter && @printf(", unguaranteed sides %d",
                               CL.positivity_counts(solver).unguaranteed)
            println()
        end
        flush(stdout)
    end
end

# --- refined levels --------------------------------------------------------

units(s, Q) = Q isa Vector ? CL.eachpatch(s, Q) : ((s, Q),)
level_of(ps) = ps isa CL.PatchSolver ? ps.patch.level : 0

# The inadmissible points of the composite, those on a refined patch's
# coarse-fine plane apart.
function bad_split(s, Q)
    shell = 0
    rest = 0
    eq = s.equations
    m1, m2, m3 = eq.i_mom
    for (ps, q) in units(s, Q)
        n = ps.decomp.n_local
        o = ps.decomp.offset
        N = ps.decomp.n_global
        for k in 1:n[3], j in 1:n[2], i in 1:n[1]
            I = padded_index(ps, i, j, k)
            ρ = sum(q[I, c] for c in 1:eq.n_species)
            e = q[I, eq.i_energy] - (q[I, m1]^2 + q[I, m2]^2 + q[I, m3]^2) / (2ρ)
            (ρ > 0 && e > 0) && continue
            g = (i + o[1], j + o[2], k + o[3])
            on = any(d -> n[d] > 1 && ((g[d] == 1 && CL.parent_fed(ps.bcs[d][1])) ||
                                       (g[d] == N[d] && CL.parent_fed(ps.bcs[d][2]))), 1:3)
            on ? (shell += 1) : (rest += 1)
        end
    end
    return shell, rest
end

# (x, ρ) along the first line, the finest level's value at each x.
function level_profile(s, Q)
    pts = Dict{Float64,Tuple{Int,Float64}}()
    for (ps, q) in units(s, Q), i in 1:ps.decomp.n_local[1]
        x = round(xcoord(ps, 1, i), digits=12)
        lv = level_of(ps)
        haskey(pts, x) && pts[x][1] >= lv && continue
        pts[x] = (lv, q[padded_index(ps, i, 1, 1), 1])
    end
    x = sort(collect(keys(pts)))
    return x, [pts[k][2] for k in x]
end

level_numerics(n, amr, on; kw...) =
    Numerics(; n_global=n, art=ArtificialProperties(enabled=true),
             filter=StateFilter(compact_filter(); cfl=0.35),
             control=StepControl(validity=:permissive), patch_interfaces=FLUX[], amr,
             positivity_limiter=on, kw...)

const LEVEL_CASES = ("wc", "wcsub", "wcregrid", "noh85", "noh43", "noh85sub", "noh43sub",
                     "nohuniform", "noh81", "noh81tiles", "noh81tilessub", "noh81regrid",
                     "noh81regridsub", "pnoh81tiles", "pnoh81tilessub", "pnoh81regrid",
                     "pnoh81regridsub", "a19", "a19tiles", "a19tilessub", "a19regrid",
                     "a19regridsub", "tiles", "tilessub")

# The refinement of a Noh level case named noh<m>, then `tiles` for static tiles
# of edge 8 and `regrid` for tiles of edge 8 regridded every 10 steps from the
# box, and `sub` for subcycling.
function noh_level(name)
    rest = name[4:end]
    m = parse(Int, rest[1:2])
    rest = rest[3:end]
    sub = endswith(rest, "sub")
    tiles = startswith(rest, "tiles")
    regrid = startswith(rest, "regrid")
    box = BlockRegion((0, 0, 0), (m, 1, 1))
    amr = tiles ? AMR(initial=box, tile=8, subcycle=sub) :
          regrid ? AMR(initial=box, tile=8, regrid_interval=10, subcycle=sub) :
                   AMR(initial=box, subcycle=sub)
    label = (tiles ? "tiles of 8" : regrid ? "regridded tiles" : "box") * " over $m nodes" *
            (sub ? ", sub" : "")
    return amr, label
end

# The runs of a level case: (label, problem, numerics from the limiter switch,
# end time, metric of the final state or `nothing`). `a19n` is the root's nodes
# per direction of the r-z converging shock.
function level_runs(name; a19n=64)
    per = (PeriodicBC(), PeriodicBC())
    if name in ("wc", "wcsub", "wcregrid")
        wc = woodward_problem(200)
        num(a) = on -> level_numerics((200, 1, 1), a, on; cfl=0.3)
        name == "wcregrid" || return [("box", wc,
                                       num(AMR(initial=BlockRegion((110, 0, 0), (60, 1, 1)),
                                               subcycle=name == "wcsub")), WC_T, nothing)]
        jumps = (x, y, z, t) -> t == 0 && (abs(x - 0.1) < 0.03 || abs(x - 0.9) < 0.03)
        amr(; kw...) = AMR(; initial=jumps, regrid_interval=10, kw...)
        return [(label, wc, num(a), WC_T, nothing)
                for (label, a) in (("regridded box", amr()),
                                   ("regridded tiles", amr(tile=8)),
                                   ("regridded tiles, subcycled", amr(tile=8, subcycle=true)))]
    elseif startswith(name, "noh") || startswith(name, "pnoh")
        # The planar rows take the cylindrical rows' 256 root nodes.
        ν = startswith(name, "p") ? 1 : 2
        N = Dict(NOH_N)[2]
        uniform = name == "nohuniform"
        amr, label = uniform ? (nothing, "uniform, $(3N - 1) nodes") :
                     noh_level(ν == 1 ? name[2:end] : name)
        metric = (s, Q) -> begin
            x, ρ = level_profile(s, Q)
            plat, deficit, shock, epre = noh_metrics(x, ρ, ν)
            ν == 1 && return @sprintf("plateau %.4f, deficit %.1f%%, shock %.4f, \
                                       pre-shock L1 %.2e", plat, 100deficit, shock, epre)
            drift = volume_integral(s, Q, :rho) / noh_cylinder_mass(NOH_T) - 1
            @sprintf("plateau %.4f, deficit %.1f%%, shock %.4f, mass %+.2e", plat,
                     100deficit, shock, drift)
        end
        noh_num(on) = Numerics(n_global=(uniform ? 3N - 1 : N, 1, 1),
                               art=ArtificialProperties(enabled=true), cfl=NOH_CFL,
                               positivity_limiter=on, patch_interfaces=FLUX[],
                               control=StepControl(validity=:permissive), amr=amr)
        return [(label, noh_problem(ν; N, t0=0.0), noh_num, NOH_T, metric)]
    elseif startswith(name, "a19")
        rz = Problem(eos=IdealSpecies("gas"; R=1.0, gamma=1.4), metric=CylindricalMetric(),
                     domain=((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)),
                     bcs=((AxisBC(), SlipWallBC()), per, (SymmetryPlaneBC(), SlipWallBC())),
                     ic=(r, θ, z) -> (d = tanh_blend(hypot(r, z), 0.7, 0.024);
                                      Prim(rho=1.0 + 3.0 * d, p=0.1 + 19.9 * d)))
        corner = CL.Regions.Box((0.0, 0.0, 0.0), (0.8, 1.0, 0.8))
        # The tiles cover whole lattice cells of edge 8 within the box's 0.8.
        m = 8 * floor(Int, 0.8 * (a19n - 1) / 8) + 1
        cells = BlockRegion((0, 0, 0), (m, 1, m))
        sub = endswith(name, "sub")
        tiles = startswith(name, "a19tiles")
        regrid = startswith(name, "a19regrid")
        amr = tiles ? AMR(initial=cells, tile=8, subcycle=sub) :
              regrid ? AMR(initial=cells, tile=8, regrid_interval=10, subcycle=sub) :
                       AMR(initial=corner)
        label = (tiles ? "tiles of 8 at the corner" :
                 regrid ? "regridded tiles of 8" : "box at the corner") * (sub ? ", sub" : "")
        return [(label, rz,
                 on -> Numerics(n_global=(a19n, 1, a19n), amr=amr,
                                positivity_limiter=on, patch_interfaces=FLUX[],
                                control=StepControl(validity=:permissive)), 0.165, nothing)]
    end
    blast = Problem(eos=IdealSpecies("gas"; R=1.0, gamma=1.4),
                    transport=ConstantTransport(mu0=0.0),
                    domain=((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)),
                    bcs=((SlipWallBC(), SlipWallBC()), (SlipWallBC(), SlipWallBC()), per),
                    ic=(x, y, z) -> Prim(rho=1.0, u=(0.0, 0.0, 0.0),
                                         p=1e-3 + exp(-((x - 0.5)^2 + (y - 0.45)^2) /
                                                      0.004)))
    return [("tiles of edge 8", blast,
             on -> level_numerics((64, 64, 1),
                                  AMR(initial=BlockRegion((16, 16, 0), (32, 32, 1)), tile=8,
                                      subcycle=name == "tilessub"), on; cfl=0.4),
             0.25, nothing)]
end

function levels_part(names, a19n)
    println("\n=== levels: refined levels, interface_flux = :$(FLUX[]), " *
            "limiter off and on ===")
    for name in names, (label, prob, num, tfinal, metric) in level_runs(name; a19n),
        on in (false, true)
        s, Q = setup(prob, num(on))
        bad = [0, 0]
        count_bad = (s, q) -> (bad .+= bad_split(s, q); nothing)
        run!(s, Q; tfinal, callback=count_bad)
        @printf("  %-10s %-26s %-3s %6d steps, bad shell %5d, rest %6d", name, label,
                on ? "on" : "off", s.step, bad[1], bad[2])
        if on
            c = CL.positivity_counts(s)
            @printf(", limited %.2f%% / %.2f%%, unguaranteed %d",
                    100c.stage_limited / c.stage_faces, 100c.filter_limited / c.filter_faces,
                    c.unguaranteed)
        end
        metric === nothing || print(", ", metric(s, Q))
        println()
        flush(stdout)
    end
end

function level_cost_part(names, reps, a19n)
    println("\n=== levelcost: wall time per step of refined runs, interface_flux = " *
            ":$(FLUX[]), limiter off and on ===")
    steps = Dict("wc" => 600, "wcsub" => 300, "noh85" => 1000, "noh43" => 1000,
                 "noh85sub" => 400, "noh43sub" => 400, "a19" => 150, "a19tiles" => 150,
                 "tiles" => 60, "tilessub" => 30)
    for name in names, (label, prob, num, tfinal, _) in level_runs(name; a19n)
        nmax = get(steps, name, 300)
        for on in (false, true)
            s, Q = setup(prob, num(on))
            run!(s, Q; tfinal, nmax=2)
        end
        per_step = Dict(false => Float64[], true => Float64[])
        for _ in 1:reps, on in (false, true)
            s, Q = setup(prob, num(on))
            wall = @elapsed run!(s, Q; tfinal, nmax)
            push!(per_step[on], wall / s.step)
        end
        ratios = per_step[true] ./ per_step[false]
        @printf("  %-10s %-26s off %.3f ms, on %.3f ms per step; on/off %.2f\n", name,
                label, 1e3 * median(per_step[false]), 1e3 * median(per_step[true]),
                median(ratios))
        flush(stdout)
    end
end

# The step of the second level case of `names` (tiles) against the first (a
# box), limiter off and on, the four runs alternated `reps` times in one
# process: the medians per step and the medians of the ratios of tiles on to
# tiles off and to the box on.
function tile_cost_part(names, reps, a19n)
    println("\n=== tilecost: the limited tiled step against the unlimited one and the box, " *
            "interface_flux = :$(FLUX[]) ===")
    length(names) == 2 || throw(ArgumentError("part=tilecost takes two level cases"))
    nmax = 150
    runs = [only(level_runs(name; a19n)) for name in names]
    for (_, prob, num, tfinal, _) in runs, on in (false, true)
        s, Q = setup(prob, num(on))
        run!(s, Q; tfinal, nmax=2)
    end
    per_step = Dict((k, on) => Float64[] for k in 1:2, on in (false, true))
    for _ in 1:reps, (k, (_, prob, num, tfinal, _)) in enumerate(runs), on in (false, true)
        s, Q = setup(prob, num(on))
        wall = @elapsed run!(s, Q; tfinal, nmax)
        push!(per_step[(k, on)], wall / s.step)
    end
    for (k, (label, _, _, _, _)) in enumerate(runs)
        @printf("  %-28s off %.3f ms, on %.3f ms per step\n", label,
                1e3 * median(per_step[(k, false)]), 1e3 * median(per_step[(k, true)]))
    end
    @printf("  tiles on / tiles off %.2f, tiles on / box on %.2f\n",
            median(per_step[(2, true)] ./ per_step[(2, false)]),
            median(per_step[(2, true)] ./ per_step[(1, true)]))
    return nothing
end

# --- the tiled level's mass drift ---------------------------------------------

# A drift variant is a layout and an optional change after a hyphen
# (`tiles8-ghost`): `box`, the level over root nodes 1:DRIFT_M as one patch;
# `tilesE`, static tiles of edge E (80 is one tile); `regrid8`, tiles of edge 8
# regridded every 10 steps; `regridbox`, one box regridded every 10 steps;
# `uniform`, the run at the level's spacing. The
# changes are `sub` (subcycled), `ghost` (`interface_flux = :ghost`),
# `noreflux` (`REFLUX` off), `clamp` (`SENSOR_INTERFACE_GHOSTS` off),
# `nofilter` (no filter pass), `lim` (the positivity limiter on) and
# `ghostlim` (the limiter on under `:ghost`, `LIMITED_GHOST_INTERFACES` set).
# The limited runs take no
# conservative coarse-fine coupling, as `REFLUX` off.
const DRIFT_LAYOUTS = ("box", "tiles8", "tiles16", "tiles80", "regrid8", "regridbox",
                       "uniform")
const DRIFT_CHANGES = ("", "sub", "ghost", "noreflux", "clamp", "nofilter", "lim",
                       "ghostlim")
const DRIFT_M = 81
# The drift history's sample times.
const DRIFT_MARKS = collect(0.02:0.02:NOH_T)

function drift_variant(name)
    layout, change = occursin('-', name) ? split(name, '-'; limit=2) : (name, "")
    layout in DRIFT_LAYOUTS && change in DRIFT_CHANGES ||
        throw(ArgumentError("unknown drift variant '$name', want a layout of " *
                            "$(join(DRIFT_LAYOUTS, ", ")) and a change of " *
                            join(DRIFT_CHANGES[2:end], ", ")))
    return String(layout), String(change)
end

function drift_numerics(layout, change, N, ν=2)
    sub = change == "sub"
    box = BlockRegion((0, 0, 0), (DRIFT_M, 1, 1))
    amr = layout == "uniform" ? nothing :
          layout == "box" ? AMR(initial=box, subcycle=sub) :
          layout == "regrid8" ? AMR(initial=box, tile=8, regrid_interval=10, subcycle=sub) :
          layout == "regridbox" ? AMR(initial=box, regrid_interval=10, subcycle=sub) :
          AMR(initial=box, tile=parse(Int, layout[6:end]), subcycle=sub)
    # The level's spacing on the cylinder's half-offset axis is R/(3N − 3/2),
    # on the planar wall's node-centred line R/(3N − 3).
    n_uniform = ν == 2 ? 3N - 1 : 3N - 2
    return Numerics(n_global=(layout == "uniform" ? n_uniform : N, 1, 1),
                    art=ArtificialProperties(enabled=true), cfl=NOH_CFL,
                    filter=change == "nofilter" ? nothing : StateFilter(),
                    patch_interfaces=change in ("ghost", "ghostlim") ? :ghost : :closure,
                    positivity_limiter=change in ("lim", "ghostlim"),
                    control=StepControl(validity=:permissive), amr=amr)
end

# Run `f` with the toggles of `change` set, and restore them.
function with_drift_toggles(f, change)
    saved = (CL.REFLUX[], CL.SENSOR_INTERFACE_GHOSTS[], CL.LIMITED_GHOST_INTERFACES[])
    CL.REFLUX[] = change != "noreflux"
    CL.SENSOR_INTERFACE_GHOSTS[] = change != "clamp"
    CL.LIMITED_GHOST_INTERFACES[] = change == "ghostlim"
    try
        return f()
    finally
        CL.REFLUX[], CL.SENSOR_INTERFACE_GHOSTS[], CL.LIMITED_GHOST_INTERFACES[] = saved
    end
end

function density_array(ps, q)
    f = zeros(size(ps.rho))
    for I in CartesianIndices(f)
        f[I] = sum(q[I, c] for c in 1:ps.equations.n_species)
    end
    return f
end

# The mass two ways that use neither the covered masks nor the junction
# weights: the single-patch quadrature of every refined patch plus the root's
# over the nodes from DRIFT_M out, half weight at DRIFT_M (`split`); and the
# root's over the whole line, holding the restriction of the level (`root`).
# Also the largest difference between a covered root node and the level's
# node at the same position, relative to the density there.
function unmasked_masses(s, Q)
    split_mass = 0.0
    root = nothing
    fine = Dict{Float64,Float64}()
    for (ps, q) in units(s, Q)
        f = density_array(ps, q)
        if level_of(ps) == 0
            root = (ps, f)
            continue
        end
        split_mass += CL._local_volume_integral(ps, f, false)
        for i in 1:ps.decomp.n_local[1]
            fine[round(xcoord(ps, 1, i), digits=12)] = f[padded_index(ps, i, 1, 1)]
        end
    end
    ps, f = root
    whole = CL._local_volume_integral(ps, f, false)
    restricted = 0.0
    for i in 1:ps.decomp.n_local[1]
        I = padded_index(ps, i, 1, 1)
        g = ps.decomp.offset[1] + i
        x = round(xcoord(ps, 1, i), digits=12)
        haskey(fine, x) && (restricted = max(restricted, abs(f[I] - fine[x]) / f[I]))
        g < DRIFT_M ? (f[I] = 0.0) : g == DRIFT_M ? (f[I] *= 0.5) : nothing
    end
    return split_mass + CL._local_volume_integral(ps, f, false), whole, restricted
end

const DRIFT_UNIFORM = Ref{Any}(nothing)
# `ledger=true`: each cylindrical run's mass drift attributed to the mechanism
# that made it by the private budget ledger of src/ledger.jl.
const DRIFT_LEDGER = Ref(false)

# The ledger's mass pieces, summed over the levels, relative to `mass`.
function print_drift_ledger(r, mass)
    pieces = Dict{Symbol,Float64}()
    for ((phase, _), v) in r.pieces
        phase in CL.LEDGER_PHASES || continue
        pieces[phase] = get(pieces, phase, 0.0) + v[1]
    end
    @printf("  %-17s ledger %s\n", "",
            join((@sprintf("%s %+.2e", p, pieces[p] / mass)
                  for p in CL.LEDGER_PHASES if haskey(pieces, p)), "  "))
end

# The Noh of the level rows (`noh81`, unlimited; ν = 1 the planar line of
# test/cases.jl, whose exact mass is 1 + t) under one drift variant: the
# steps, the metrics of the level rows, the composite mass drift and the two
# unmasked ones, the density L1 error ∫|ρ − ρ_ref| r^(ν−1) dr over the
# level's region [0, x_M] against the exact solution and against the uniform
# run at the level's spacing (when `uniform` ran first), the mean |ρ − 4^ν|
# over the plateau core, and the composite drift at the times of DRIFT_MARKS.
# The unmasked masses assume the level over root nodes 1:DRIFT_M, so a
# regridded layout prints none.
function noh_drift_run(name, ν=2)
    layout, change = drift_variant(name)
    N = Dict(NOH_N)[ν]
    exact_mass = ν == 2 ? noh_cylinder_mass : t -> 1 + t
    history = Float64[]
    ledger = Ref{Any}(nothing)
    s, Q, wall = with_drift_toggles(change) do
        solver, states = setup(noh_problem(ν; N, t0=0.0),
                               drift_numerics(layout, change, N, ν))
        track = DRIFT_LEDGER[] && ν == 2 && states isa Vector
        track && CL._ledger_begin!(solver, states)
        record = (sv, q) -> begin
            while length(history) < length(DRIFT_MARKS) &&
                  sv.t >= DRIFT_MARKS[length(history)+1] - 1e-12
                push!(history, volume_integral(sv, q, :rho) / exact_mass(sv.t) - 1)
            end
            nothing
        end
        elapsed = @elapsed try
            run!(solver, states; tfinal=NOH_T, callback=record)
        catch err
            err isa SolverFailure || rethrow()
        end
        track && (ledger[] = CL._ledger_end!(solver, states))
        solver, states, elapsed
    end
    if s.t < NOH_T - 1e-12
        @printf("  %-17s %5d steps %6.1f s  stopped at t = %.4f\n", name, s.step, wall, s.t)
        return nothing
    end
    x, ρ = level_profile(s, Q)
    plat, deficit, shock, _ = noh_metrics(x, ρ, ν)
    composite = volume_integral(s, Q, :rho) / exact_mass(s.t) - 1
    @printf("  %-17s %5d steps %6.1f s  plateau %.4f  deficit %4.1f%%  shock %.4f\n", name,
            s.step, wall, plat, 100deficit, shock)
    if layout in ("uniform", "regrid8", "regridbox")
        @printf("  %-17s mass %+.2e\n", "", composite)
    else
        split_mass, root, restricted = unmasked_masses(s, Q)
        @printf("  %-17s mass %+.2e  split %+.2e  root %+.2e  restricted %.1e\n", "",
                composite, split_mass / exact_mass(s.t) - 1, root / exact_mass(s.t) - 1,
                restricted)
    end
    ledger[] === nothing || print_drift_ledger(ledger[], exact_mass(s.t))
    # The level's spacing and the station of root node DRIFT_M.
    hf, xM = ν == 2 ? (1 / (3 * (N - 0.5)), (DRIFT_M - 0.5) / (N - 0.5)) :
                      (1 / (3 * (N - 1)), (DRIFT_M - 1) / (N - 1))
    level = [i for i in eachindex(x) if x[i] <= xM + 1e-9]
    l1_exact = sum(abs(ρ[i] - noh_exact(x[i], NOH_T, ν, NOH_G)[1]) * x[i]^(ν - 1) * hf
                   for i in level)
    Rs = (NOH_G - 1) / 2 * NOH_T
    core = [i for i in eachindex(x) if 0.3Rs <= x[i] <= 0.7Rs]
    l1_plateau = sum(abs(ρ[i] - 4.0^ν) for i in core) / length(core)
    layout == "uniform" && change == "" &&
        (DRIFT_UNIFORM[] = Dict(zip(round.(x, digits=12), ρ)))
    ref = DRIFT_UNIFORM[]
    l1_uniform = ref === nothing ? NaN :
                 sum(abs(ρ[i] - ref[round(x[i], digits=12)]) * x[i]^(ν - 1) * hf
                     for i in level)
    @printf("  %-17s L1 exact %.3e  L1 uniform %.3e  plateau |ρ - 4^ν| %.4f\n", "", l1_exact,
            l1_uniform, l1_plateau)
    @printf("  %-17s history %s\n", "", join((@sprintf("%+.1e", d) for d in history), " "))
    flush(stdout)
    return nothing
end

# A smooth density pulse at rest, ρ = 1 + 0.05 exp(−((r − 0.12)/0.03)²) under
# p ∝ ρ^γ, on the same line closed by a slip wall at r = 1: the composite mass
# at t = 0 against the first variant's and the drift of the composite mass
# over t = 0.3, in which the outgoing half crosses every tile face beyond
# r = 0.12 and leaves the level at r = 0.313.
function pulse_drift_run(name, mass0)
    layout, change = drift_variant(name)
    N = Dict(NOH_N)[2]
    prob = Problem(eos=IdealSpecies("gas"; gamma=NOH_G, R=1.0),
                   transport=ConstantTransport(mu0=0.0), metric=CylindricalMetric(),
                   domain=((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)),
                   bcs=((AxisBC(), SlipWallBC()), per3[2], per3[3]),
                   ic=(r, y, z) -> (ρ = 1 + 0.05 * exp(-((r - 0.12) / 0.03)^2);
                                    Prim(rho=ρ, u=(0.0, 0.0, 0.0), p=ρ^NOH_G / NOH_G)))
    s, m0, m1 = with_drift_toggles(change) do
        solver, states = setup(prob, drift_numerics(layout, change, N))
        before = volume_integral(solver, states, :rho)
        run!(solver, states; tfinal=0.3)
        solver, before, volume_integral(solver, states, :rho)
    end
    mass0[] === nothing && (mass0[] = m0)
    @printf("  %-17s %5d steps  mass at t = 0 against the first %+.2e, drift %+.2e\n", name,
            s.step, m0 / mass0[] - 1, m1 / m0 - 1)
    flush(stdout)
    return nothing
end

function tile_drift_part(names, pulse_names, planar_names)
    println("\n=== tiledrift: the cylindrical Noh level over 81 nodes ===")
    DRIFT_UNIFORM[] = nothing
    for name in names
        noh_drift_run(name)
    end
    if !isempty(planar_names)
        println("\n=== tiledrift: the planar Noh level over 81 nodes ===")
        DRIFT_UNIFORM[] = nothing
        for name in planar_names
            noh_drift_run(name, 1)
        end
    end
    isempty(pulse_names) && return nothing
    println("\n=== tiledrift: a smooth pulse at rest on the same line, to t = 0.3 ===")
    mass0 = Ref{Any}(nothing)
    for name in pulse_names
        pulse_drift_run(name, mass0)
    end
    return nothing
end

function main(args)
    opt = CL.script_args(args, (part="all", cases="woodward,noh",
                                variants=join(VARIANTS, ","), nmax=100_000,
                                identity_steps=300, N=96, steps=500, reps=3,
                                patches="1", levels="wc,wcsub,noh43", a19n=64,
                                flux="closure", ledger=false, conservative=true,
                                drift="uniform,box,tiles8,tiles16,tiles80,box-sub," *
                                      "tiles8-sub,regrid8,tiles8-ghost,box-ghost," *
                                      "tiles8-noreflux,tiles8-clamp,tiles8-nofilter," *
                                      "box-nofilter,box-lim,tiles8-lim,box-ghostlim," *
                                      "tiles8-ghostlim,tiles16-ghostlim",
                                pulse="box,tiles8,tiles8-ghost,box-nofilter,tiles8-nofilter",
                                planar="uniform,box,tiles8,tiles8-ghost,tiles8-lim," *
                                       "tiles8-ghostlim"))
    opt.flux in ("closure", "ghost") ||
        throw(ArgumentError("unknown flux '$(opt.flux)', want closure or ghost"))
    FLUX[] = Symbol(opt.flux)
    DRIFT_LEDGER[] = opt.ledger
    parts = opt.part == "all" ? collect(PARTS) : split(opt.part, ',')
    cases = Symbol.(split(opt.cases, ','))
    counts = parse.(Int, split(opt.patches, ','))
    for p in parts
        p in PARTS ||
            p in ("rz", "cost", "patches", "levels", "levelcost", "tilecost", "tiledrift") ||
            throw(ArgumentError("unknown part '$p', want one of $(join(PARTS, ", ")), \
                                 rz, cost, patches, levels, levelcost, tilecost or tiledrift"))
    end
    drift_names = split(opt.drift, ',')
    pulse_names = filter(!isempty, split(opt.pulse, ','))
    planar_names = filter(!isempty, split(opt.planar, ','))
    foreach(drift_variant, [drift_names; pulse_names; planar_names])
    level_names = split(opt.levels, ',')
    for l in level_names
        l in LEVEL_CASES ||
            throw(ArgumentError("unknown level case '$l', want one of " *
                                join(LEVEL_CASES, ", ")))
    end
    for c in cases
        c in CASES || (c in PLANE_CASES && parts == ["cost"]) ||
            throw(ArgumentError("unknown case '$c', want one of $(join(CASES, ", "))"))
        c in RADIAL && parts != ["src"] && parts != ["cost"] &&
            throw(ArgumentError("the radial case '$c' runs in part=src or part=cost alone"))
    end
    variants = variant_flags.(split(opt.variants, ','))
    # A limited run keeps :ghost at its interfaces only under the toggle;
    # `conservative=false` closes the filter at a same-level interface with the
    # identity row (`CONSERVATIVE_INTERFACE_FILTER`).
    saved = (CL.LIMITED_GHOST_INTERFACES[], CL.CONSERVATIVE_INTERFACE_FILTER[])
    CL.LIMITED_GHOST_INTERFACES[] = FLUX[] === :ghost
    CL.CONSERVATIVE_INTERFACE_FILTER[] = opt.conservative
    try
        run_parts(parts, cases, counts, variants, level_names, drift_names, pulse_names,
                  planar_names, opt)
    finally
        CL.LIMITED_GHOST_INTERFACES[], CL.CONSERVATIVE_INTERFACE_FILTER[] = saved
    end
    return nothing
end

function run_parts(parts, cases, counts, variants, level_names, drift_names, pulse_names,
                   planar_names, opt)
    weights = "weights" in parts ? weights_part(cases) : Dict{Symbol,Vector{Float64}}()
    "identity" in parts && identity_part(cases, opt.identity_steps, weights)
    "pulse" in parts && pulse_part()
    "src" in parts && src_part(cases)
    "rz" in parts && rz_part(opt.N)
    "cost" in parts && cost_part(cases, opt.N, opt.steps, opt.reps, first(counts))
    "patches" in parts && patches_part(counts)
    "levels" in parts && levels_part(level_names, opt.a19n)
    "levelcost" in parts && level_cost_part(level_names, opt.reps, opt.a19n)
    "tilecost" in parts && tile_cost_part(level_names, opt.reps, opt.a19n)
    "tiledrift" in parts && tile_drift_part(drift_names, pulse_names, planar_names)
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
