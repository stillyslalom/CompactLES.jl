# The positivity limiter selected by `Numerics(positivity_limiter = true)`:
# a conservative correction, after Hu, Adams & Shu (J. Comput. Phys. 242,
# 2013), that keeps the mixture density and the internal energy per volume
# ρe = E − ½|m|²/ρ above a bound ε at every stage and every filter pass.
#
# The face form. On a closed line the compact divergence is a difference of
# face fluxes under node weights W with Σ_i W_i (D f)_i = f_N − f_1: the
# closure rows telescope under positive weights (Lele 1992, §4.2), so the face
# flux is the running sum F̂_{i+½} = f_1 + Σ_{k≤i} W_k (D f)_k and needs no
# line solve. On a periodic line W = h and the running sum leaves one constant
# per line, fixed by the interior face relation
# Σ_s l_s F̂_{j+½+s} = Σ_m c_m Σ_{l=1−m}^{m} f_{j+l} of the scheme's own rows.
# The weights are computed from the divergence plan's scheme on a model line
# (`_limiter_weights`), so another closure set supplies its own, and a set
# whose rows do not telescope is rejected at setup.
#
# The stage limiter (A). The low-storage integrator advances
# du ← A_k du + dt dQ, Q ← Q + B_k du, and A_2..A_5 are negative, so a stage
# state is not a convex combination of forward-Euler steps. The limiter
# therefore acts on the whole stage increment B_k du_k. Per active direction d
# a node register r_d = A_k r_d + dt (D_d F_d)_k carries that direction's part
# of du (du = −Σ_d r_d + the node terms), and its running sum is the face
# register Φ_d. Each face of B_k Φ_d is blended toward τ_k times the
# Lax–Friedrichs flux of the stage's base state, τ_k = (c_{k+1} − c_k) dt the
# stage's own time advance, with θ ∈ [0, 1] per face the smaller of the two
# cells' limits. Cell i splits its increment over the directions in
# proportion α_d = λ_d / Σ λ, λ_d = (|u_d| + c)/W_d, so its half state across
# a face is Q_i ∓ 2 G / (α_d W_d) and the first-order one is admissible when
# 2 τ a Σ λ / (|u_d| + c) ≤ 1 with a the face's wave speed. The limited
# increment enters dQ before the low-storage update and the register r_d, so
# the next stage's A_{k+1} sees it. A cell where the first-order bound fails
# or whose first-order half state is inadmissible is counted as unguaranteed;
# the latter takes θ = 0 from that side. At the default `:neutral3` wall node
# (W/h = 0.215) the bound fails above a CFL of about 0.32. A cell on a closed
# end takes its boundary face, which is never limited, in one part with its
# interior face rather than in two halves: split in two, the wall's pressure
# flux alone leaves a half state whose kinetic energy exceeds E behind a strong
# shock reaching the wall, which the real update, with both faces, does not.
#
# The filter limiter (B). The correction of a filter pass along d is a
# difference of a face flux, the running sum of −ω Δ with ω = W/h, once the
# constant a closed line leaves in it is removed through the filter's interior
# face relation. θ per face scales it toward zero, the admissible state before
# the pass. The closure rows of the filter do not conserve, and their defect
# stays on the boundary faces, which neither limiter touches. The filter's row
# at a closed end is the identity, and the limiter leaves that node alone as
# well: a limited face beside it corrects its interior neighbour only, so the
# line's total moves by that correction, as it does under the closure rows.
#
# Both are corrections at the faces they limit: a run in which no face is
# limited is the unlimited run bit for bit. The node terms (sources, the
# NSCBC corrections) are outside the face form and outside the guarantee. The
# partial densities are not bounded: their interface undershoots are inside
# the species band by design, and a bound there would act at every captured
# interface.
#
# Per-point work runs through `pointwise!` bodies over three index boxes: the
# lines of a direction (one sequential scan per line), its faces, and its
# nodes. The running sum along a decomposed line takes one `Allgather` of the
# line totals over the direction's sub-communicator per direction per stage
# and per filter pass, entered by every rank of it.

# ε is this fraction of the minimum ρ and ρe of the state entering `run!`.
const LIMITER_FRACTION = 0.01

"""
    PositivityLimiter

The state of the positivity limiter on a solver built with
`positivity_limiter = true` (see `Numerics`): the face weights, the
per-direction node registers, the line planes of the running sums, and the
bounds of the current `run!`. Built by setup.
"""
mutable struct PositivityLimiter{T,A<:AbstractArray{T,3},V<:AbstractVector{T}}
    weights::NTuple{3,V}            # W_d at each padded position along d
    inv_weights::NTuple{3,V}        # 1 / W_d, which the face pass multiplies by
    free::NTuple{3,Vector{Bool}}    # positions on a face whose condition
                                    # overwrites the state (DirichletBC)
    registers::Matrix{A}            # r[d, c]
    register_fields::Vector{FieldVector{A,Vector{A}}}   # r[d, :] per d
    faces::FieldVector{A,Vector{A}} # face values, in the workspace's flux[1, :]
    base::FieldVector{A,Vector{A}}  # the filter's pre-pass state, flux[2, :]
    anchor::Vector{Array{T,3}}      # per d: (n_cons, n_a, n_b) Φ at the low face
    totals::Vector{Array{T,3}}      # line totals of the local running sums
    aux::Vector{Array{T,3}}         # anchor (stage) or measurement (filter)
    offset::Vector{Array{T,3}}      # this rank's running-sum offset
    wrap::Vector{Array{T,3}}        # the value at the global low face
    cstar::Vector{Array{T,3}}       # the constant removed from a filter flux
    zero_plane::Vector{Array{T,3}}
    send::Vector{Vector{T}}
    recv::Vector{Vector{T}}
    deriv_lhs::Vector{T}            # the derivative's interior face relation
    deriv_rhs::Vector{T}
    filter_lhs::Vector{T}           # the filter's, and its explicit face stencil
    filter_psi::Vector{T}
    periodic::NTuple{3,Bool}
    measure::NTuple{3,Int}          # local face of the filter relation, 0 none
    anchor_face::NTuple{3,Int}      # local face of a periodic anchor
    eps_rho::T
    eps_e::T
    active::Bool
    # Stage faces, stage faces limited, filter faces, filter faces limited,
    # and unguaranteed cell sides, rank-local; `positivity_counts` reduces.
    counts::Vector{Int}
end

# --- Setup ---------------------------------------------------------------

_band_lhs(s::CompactScheme) = [s.alpha]
_band_lhs(s::BandedCompactScheme) = copy(s.lhs)

# W with Wᵀ D = (e_N − e_1)ᵀ for the divergence D of `scheme` on a closed line
# of N nodes at spacing h: free on `edge` nodes at either end and one shared
# value inside, by least squares, whose residual tests the face form. D has a
# one-dimensional left null space (an odd-even mode near the closures), and
# the shared interior value excludes it. A line longer than the model takes
# the model's end weights and its interior value between them; the weights
# approach the interior value geometrically (by 0.036 per node for C6).
function _limiter_weights(scheme, N::Int, h, n_halo::Int, ::Type{T}) where {T}
    M = min(N, 128)
    D = _line_operator(scheme, M, h, n_halo, T)
    edge = min(24, (M - 1) ÷ 2 - 2)
    edge >= 4 || throw(ArgumentError(
        "positivity_limiter: a closed line of $N nodes is too short for the " *
        "face form of the divergence"))
    free = [1:edge; M-edge+1:M]
    inner = edge+1:M-edge
    A = zeros(M, length(free) + 1)
    for j in 1:M
        for (u, i) in enumerate(free)
            A[j, u] = D[i, j]
        end
        A[j, end] = sum(D[i, j] for i in inner)
    end
    target = zeros(M)
    target[1], target[M] = -1, 1
    x = A \ target
    residual = maximum(abs, A * x - target)
    Wm = fill(x[end], M)
    Wm[free] = x[1:end-1]
    W = M == N ? Wm : [Wm[1:48]; fill(x[end], N - 96); Wm[M-47:M]]
    return W, residual
end

# The explicit face stencil of a filter's correction: (B − A) q, a symmetric
# zero-sum stencil s_l, is ψ_{i+½} − ψ_{i−½} with ψ_{i+½} = Σ_l t_l q_{i+l},
# t_l = Σ_{l' ≥ l} s_{l'}, l = 1 − M .. M; stored at t[l + M].
function _filter_face_stencil(scheme, ::Type{T}) where {T}
    lhs = _band_lhs(scheme)
    M = max(length(scheme.coeffs), length(lhs))
    s = zeros(2M + 1)
    s[M+1] = scheme.a0 - 1
    for m in 1:M
        v = (m <= length(scheme.coeffs) ? scheme.coeffs[m] : 0.0) -
            (m <= length(lhs) ? lhs[m] : 0.0)
        s[M+1+m] = v
        s[M+1-m] = v
    end
    return T[sum(s[k+M+1] for k in l:M) for l in 1-M:M]
end

# The checks of the configurations the limiter covers, before anything is
# built: a single host patch of an unstretched Cartesian grid without folds,
# levels or the implicit integrator, an ideal-gas mixture, and closed lines
# long enough for the face relation of the filter.
function _validate_positivity(bcs, metric, stretch, patch_grid, nlev, backend, eos,
                              implicit, equations, deriv, filt, n_global, n_halo,
                              L_domain, ::Type{T}) where {T}
    fail(what) = throw(ArgumentError("positivity_limiter: $what"))
    metric isa CartesianMetric && all(isnothing, stretch) ||
        fail("supports an unstretched CartesianMetric only")
    prod(patch_grid) == 1 && nlev == 1 ||
        fail("supports a single patch without refinement (patch_grid, refine, amr)")
    backend isa CPUBackend || fail("runs on the host backend only")
    eos isa IdealMixture ||
        fail("supports the ideal-gas EOS (IdealSpecies, IdealMixture), whose " *
             "admissible states are ρ > 0 and ρe > 0; got $(typeof(eos).name.name)")
    implicit === nothing || fail("does not combine with implicit conduction")
    equations.n_cons == equations.n_species + 4 ||
        fail("supports the single-temperature equation set")
    for d in 1:3, side in 1:2
        bc = bcs[d][side]
        bc isa Union{SymmetryPlaneBC,AxisBC,OriginBC,PoleBC} &&
            fail("does not support folded ends; face $d/$side carries " *
                 "$(nameof(typeof(bc)))")
    end
    for d in 1:3
        n_global[d] > 1 && !isperiodic(bcs[d][1]) || continue
        h = T(L_domain[d] / (n_global[d] - 1))
        W, res = _limiter_weights(deriv, n_global[d], h, n_halo, T)
        res <= sqrt(eps(T)) * 1e-2 ||
            fail("the closure rows of $(deriv.name) do not take a face-flux form " *
                 "(residual $res)")
        all(>(0), W) || fail("the face weights of $(deriv.name) are not positive")
        least = 2 * _limiter_margin(W, h, filt, deriv, T) + 4
        n_global[d] >= least ||
            fail("dimension $d has $(n_global[d]) nodes; the face relation of the " *
                 "filter needs at least $least on a closed line")
    end
    return nothing
end

# Nodes from a closed end beyond which both the derivative's weights and the
# filter's rows are interior, with the stencils' reach.
function _limiter_margin(W, h, filt, deriv, ::Type{T}) where {T}
    half = length(W) ÷ 2
    tol = sqrt(eps(T)) / 10
    tail = something(findfirst(i -> all(abs(W[k] / h - 1) <= tol for k in i:half),
                               1:half), half)
    reach = max(length(filt.coeffs), length(_band_lhs(filt))) + length(_band_lhs(filt))
    return max(tail, nclosure(filt), nclosure(deriv)) + reach + 1
end

# The limiter of a single-patch solver, after its geometry is filled: the
# weights at every padded position this rank holds, the registers, and the
# local faces where a periodic anchor and the filter relation are measured.
function PositivityLimiter(solver)
    decomp = solver.decomp
    T = eltype(solver.h)
    n_cons = solver.equations.n_cons
    plan_scheme(d) = (p = _plan_at(solver.div_plans, d); p.scheme)
    deriv = getfield(solver, :schemes).deriv
    filt = getfield(solver, :schemes).filt
    # The weights along each global line, and the nodes from a closed end
    # beyond which the filter's face relation is interior.
    global_weights = ntuple(3) do d
        decomp.active[d] || return Float64[1.0]
        N = decomp.n_global[d]
        decomp.periodic[d] && return fill(Float64(solver.h[d]), N)
        return _limiter_weights(plan_scheme(d), N, solver.h[d], decomp.n_halo, T)[1]
    end
    weights = ntuple(3) do d
        decomp.active[d] || return T[one(T)]
        n = decomp.n_local[d]
        o = decomp.n_halo_d[d]
        N = decomp.n_global[d]
        out = fill(solver.h[d], n + 2o)
        for p in 1:n+2o
            g = decomp.offset[d] + p - o
            decomp.periodic[d] && (g = mod1(g, N))
            1 <= g <= N && (out[p] = T(global_weights[d][g]))
        end
        out
    end
    free = ntuple(3) do d
        decomp.active[d] || return [false]
        n = decomp.n_local[d]
        o = decomp.n_halo_d[d]
        N = decomp.n_global[d]
        [(decomp.offset[d] + p - o == 1 && solver.bcs[d][1] isa DirichletBC) ||
         (decomp.offset[d] + p - o == N && solver.bcs[d][2] isa DirichletBC)
         for p in 1:n+2o]
    end
    empty = similar(solver.tmp_a, T, 0, 0, 0)
    registers = [decomp.active[d] ? zero(solver.tmp_a) : empty
                 for d in 1:3, _ in 1:n_cons]
    register_fields = [FieldVector([registers[d, c] for c in 1:n_cons]) for d in 1:3]
    planes = [zeros(T, n_cons, _transverse(decomp.n_local, d)...) for d in 1:3]
    copies() = [copy(p) for p in planes]
    send = [zeros(T, 2 * length(planes[d]) + 1) for d in 1:3]
    recv = [zeros(T, decomp.dims[d] * length(send[d])) for d in 1:3]
    filter_lhs = _band_lhs(filt)
    q = length(filter_lhs)
    qd = length(_band_lhs(deriv))
    measure = ntuple(3) do d
        decomp.active[d] || return 0
        n = decomp.n_local[d]
        N = decomp.n_global[d]
        j = clamp(N ÷ 2 - decomp.offset[d], q, n - q)
        decomp.periodic[d] && return j
        margin = _limiter_margin(global_weights[d], solver.h[d], filt, deriv, T)
        margin <= decomp.offset[d] + j <= N - margin ? j : 0
    end
    anchor_face = ntuple(d -> decomp.active[d] ? clamp(decomp.n_local[d] ÷ 2, qd,
                                                       decomp.n_local[d] - qd) : 0, 3)
    ws = solver.flux
    return PositivityLimiter{T,typeof(solver.tmp_a),typeof(weights[1])}(
        weights, map(w -> one(T) ./ w, weights), free, registers, register_fields,
        FieldVector([ws[1, c] for c in 1:n_cons]),
        FieldVector([ws[2, c] for c in 1:n_cons]),
        copies(), copies(), copies(), copies(), copies(), copies(), copies(),
        send, recv, _band_lhs(deriv), copy(deriv.coeffs), filter_lhs,
        _filter_face_stencil(filt, T), decomp.periodic, measure, anchor_face,
        zero(T), zero(T), false, zeros(Int, 5))
end

# The two transverse extents of a direction's lines, in index order.
_transverse(n, d) = d == 1 ? (n[2], n[3]) : d == 2 ? (n[1], n[3]) : (n[1], n[2])

# The padded node at position p along d on line (a, b), a and b the
# transverse interior indices in index order.
@inline _line_node(d, p, a, b, o1, o2, o3) =
    d == 1 ? CartesianIndex(p + o1, a + o2, b + o3) :
    d == 2 ? CartesianIndex(a + o1, p + o2, b + o3) :
             CartesianIndex(a + o1, b + o2, p + o3)

# --- The bounds of a run ---------------------------------------------------

# ε from the state entering `run!`, the same rule at a restart and after a
# phase change: `LIMITER_FRACTION` of the global minimum of ρ and of ρe over
# the interior. Collective over the solver's communicator. Returns whether the
# limiter acts in this run; a state whose minimum is not positive gives no
# scale, and the run proceeds unlimited with a warning, as the failsafe does.
_positivity_setup!(solver, Q) = false
function _positivity_setup!(solver, Q::ConservedState)
    lim = getfield(solver, :positivity)
    lim === nothing && return false
    return _positivity_bounds!(lim, solver, Q)
end

function _positivity_bounds!(lim::PositivityLimiter, solver, Q)
    decomp = solver.decomp
    o1, o2, o3 = decomp.n_halo_d
    nx, ny, nz = decomp.n_local
    ns = solver.equations.n_species
    m1, m2, m3 = solver.equations.i_mom
    ie = solver.equations.i_energy
    ρmin = Inf
    emin = Inf
    @inbounds for k in 1:nz, j in 1:ny, i in 1:nx
        I = CartesianIndex(i + o1, j + o2, k + o3)
        ρ = zero(eltype(Q))
        for sp in 1:ns
            ρ += Q[I, sp]
        end
        ρmin = min(ρmin, ρ)
        ρ > 0 || continue
        emin = min(emin, Q[I, ie] - (Q[I, m1]^2 + Q[I, m2]^2 + Q[I, m3]^2) / (2ρ))
    end
    red = MPI.Allreduce([ρmin, emin], min, solver.comm)
    T = eltype(lim.eps_rho)
    lim.active = red[1] > 0 && red[2] > 0
    if lim.active
        lim.eps_rho = T(LIMITER_FRACTION * red[1])
        lim.eps_e = T(LIMITER_FRACTION * red[2])
    elseif MPI.Comm_rank(solver.comm) == 0
        @warn "run!: the positivity limiter is inactive in this run. The minimum " *
              "density or internal energy of the state entering it is not " *
              "positive, so there is no bound to scale."
    end
    return lim.active
end

"""
    positivity_counts(solver) -> NamedTuple

What the positivity limiter of `solver` has done since it was built, summed
over the ranks: the interior faces tested at the Runge–Kutta stages and at the
filter passes, the faces limited in each, and the unguaranteed cell sides,
where the first-order bound failed or the first-order half state was not
admissible. Collective over the solver's communicator.
"""
function positivity_counts(solver)
    lim = getfield(solver, :positivity)
    lim === nothing && throw(ArgumentError(
        "positivity_counts: the solver was built without positivity_limiter"))
    c = MPI.Allreduce(lim.counts, +, solver.comm)
    return (stage_faces=c[1], stage_limited=c[2], filter_faces=c[3],
            filter_limited=c[4], unguaranteed=c[5])
end

# --- The limited step ------------------------------------------------------

# `step!` with the stage limiter between each stage's right-hand side and its
# low-storage update; with no face limited it makes the calls of `step!` in
# the same order and the same arithmetic.
function _limited_run_step!(solver, Q, workspace, dt, prepared::Bool)
    lim = getfield(solver, :positivity)
    dQ, du = workspace.dQ, workspace.du
    decomp = solver.decomp
    for stage in 1:5
        solver.tstage = solver.t + oftype(solver.t, RKC[stage]) * dt
        first_prepared = prepared && stage == 1
        if !first_prepared
            _ledger_open!(solver, Q)
            apply_bcs!(solver, Q)
            _ledger!(solver, Q, :wall_enforce)
        end
        T = eltype(lim.eps_rho)
        compute_rhs!(solver, Q, _LimiterRHS(dQ, lim, T(RKA[stage]), T(dt)),
                     first_prepared)
        _ledger_faces!(solver, 1)
        _limit_stage!(lim, solver, Q, dQ, stage, dt)
        _ledger_open!(solver, Q)
        _rk_update!(decomp, solver.equations.n_cons, Q, dQ, du,
                    RKA[stage], RKB[stage], dt)
        _ledger_update!(solver, Q, dQ, RKA[stage], RKB[stage], dt)
    end
    solver.tstage = solver.t + dt
    _ledger_open!(solver, Q)
    apply_bcs!(solver, Q)
    _ledger!(solver, Q, :wall_enforce)
    _validate_transport_state!(solver, Q)
    return nothing
end

# The time advance of stage k, c_{k+1} − c_k with c_6 = 1.
_stage_advance(stage::Int) = (stage < 5 ? RKC[stage+1] : 1.0) - RKC[stage]

# The right-hand side array a limited stage hands `compute_rhs!`: the
# workspace's dQ, indexed through, with the divergence of each flux taken
# once more out of the fused subtraction. `div_subtract_along!` on it runs the
# derivative into scratch and subtracts it, the two-pass form the folds take,
# bit for bit the fused one, and updates the direction's register
# r_d ← A r_d + dt D_d F_d and the face register at the global low face from
# the same derivative, so the registers cost no line solve of their own. Every
# other phase of the right-hand side reads and writes it as dQ.
struct _LimiterRHS{T,D<:AbstractArray{T,4},L} <: AbstractArray{T,4}
    dQ::D
    lim::L
    A::T
    dt::T
end
Base.parent(x::_LimiterRHS) = parent(x.dQ)
Base.size(x::_LimiterRHS) = size(x.dQ)
Base.axes(x::_LimiterRHS) = axes(x.dQ)
Base.IndexStyle(::Type{<:_LimiterRHS{T,D}}) where {T,D} = IndexStyle(D)
Base.@propagate_inbounds Base.getindex(x::_LimiterRHS, I...) = getindex(x.dQ, I...)
Base.@propagate_inbounds Base.setindex!(x::_LimiterRHS, v, I...) =
    setindex!(x.dQ, v, I...)
Base.view(x::_LimiterRHS, I...) = view(x.dQ, I...)
@inline _cpu_storage(x::_LimiterRHS) = _cpu_storage(x.dQ)
@inline _kernel_arg(x::_LimiterRHS) = _kernel_arg(x.dQ)

function div_subtract_along!(dQ::_LimiterRHS, c::Int, f, solver::SolverLike, d::Int,
                             σf::Int, inv_J)
    lim = dQ.lim
    decomp = solver.decomp
    nx, ny, nz = decomp.n_local
    o1, o2, o3 = decomp.n_halo_d
    div_along!(solver.tmp_a, f, solver, d, σf)
    if inv_J === nothing
        pointwise!(_subtract_div_point!, dQ.dQ, nx, ny, nz,
                   dQ.dQ, solver.tmp_a, c, o1, o2, o3)
    else
        pointwise!(_subtract_jac_div_point!, dQ.dQ, nx, ny, nz,
                   dQ.dQ, solver.tmp_a, inv_J, c, o1, o2, o3)
    end
    pointwise!(_limiter_register_point!, solver.tmp_a, nx, ny, nz,
               lim.registers[d, c], solver.tmp_a, dQ.A, dQ.dt, o1, o2, o3)
    if decomp.coords[d] == 0
        nA, nB = _transverse(decomp.n_local, d)
        pointwise!(_limiter_anchor_point!, solver.tmp_a, nA, nB, 1,
                   lim.anchor[d], f, solver.tmp_a, lim.weights[d], dQ.A, dQ.dt, c, d,
                   lim.periodic[d], lim.anchor_face[d], lim.deriv_lhs, lim.deriv_rhs,
                   o1, o2, o3)
    end
    return dQ
end

function _limit_stage!(lim::PositivityLimiter, solver, Q, dQ, stage::Int, dt)
    decomp = solver.decomp
    T = eltype(lim.eps_rho)
    B = T(RKB[stage])
    dtt = T(dt)
    τ = T(_stage_advance(stage) * dt)
    o1, o2, o3 = decomp.n_halo_d
    lim.active || return nothing
    inv_B = one(T) / B
    inv_Bdt = one(T) / (B * dtt)
    for d in 1:3
        decomp.active[d] || continue
        _line_faces!(lim, solver, Q, d, 1, B, zero(T))
        limited, shared = _limit_faces!(lim, solver, Q, d, τ, 1)
        lim.counts[2] += limited
        limited + shared > 0 || continue
        n = decomp.n_local[d]
        nA, nB = _transverse(decomp.n_local, d)
        pointwise!(_limiter_correct_point!, solver.tmp_a, n, nA, nB,
                   dQ, lim.register_fields[d], lim.faces, solver.tmp_b, Q,
                   (solver.u, solver.v, solver.w), solver.c, solver.p,
                   lim.weights[d], d, one(T), τ, 1, (inv_B, inv_Bdt),
                   _limiter_layout(solver), _limiter_closed(lim, decomp, d), n,
                   o1, o2, o3)
    end
    return nothing
end

_limiter_layout(solver) =
    (solver.equations.n_species, solver.equations.i_mom...,
     solver.equations.i_energy, solver.equations.n_cons)

# The face values of direction d into `lim.faces`: the local running sums of
# each line, the line totals gathered over the direction's sub-communicator,
# and the offsets that make them one running sum along the global line,
# anchored at the global low face. `mode` 1 is the stage register, scaled by
# `scale` = B_k and anchored at the face register; `mode` 2 a filter pass of
# weight `wf`, its constant removed through the filter's face relation. An
# interface or a level coupling would anchor a line here as a wall does.
function _line_faces!(lim::PositivityLimiter, solver, Q, d::Int, mode::Int, scale,
                      wf)
    decomp = solver.decomp
    o1, o2, o3 = decomp.n_halo_d
    n = decomp.n_local[d]
    nA, nB = _transverse(decomp.n_local, d)
    n_cons = solver.equations.n_cons
    pointwise!(_limiter_scan_point!, solver.tmp_a, nA, nB, 1,
               lim.faces, lim.register_fields[d], Q, lim.base, lim.weights[d],
               solver.h[d], lim.totals[d], lim.aux[d], n_cons, n, d, mode,
               mode == 2 ? lim.measure[d] : 0, lim.filter_lhs, lim.filter_psi, wf,
               o1, o2, o3)
    _line_offsets!(lim, decomp, d, mode)
    last_rank = decomp.coords[d] == decomp.dims[d] - 1
    pointwise!(_limiter_offset_point!, solver.tmp_a, n + 1, nA, nB,
               lim.faces, lim.offset[d], lim.wrap[d],
               mode == 1 ? lim.zero_plane[d] : lim.cstar[d], scale,
               lim.periodic[d] && last_rank, n, d, n_cons, o1, o2, o3)
    return nothing
end

# The offsets of this rank's running sums. One `Allgather` of the line totals
# (and of the low face's anchor or the filter relation's measurement) over
# the direction's sub-communicator, entered by every rank of it; the prefix
# sums are formed in rank order on every rank, so the face two ranks share
# carries the same value on both. Without a decomposition along d nothing is
# exchanged.
function _line_offsets!(lim::PositivityLimiter, decomp, d::Int, mode::Int)
    tot = lim.totals[d]
    # The stage's anchor is the face register at the global low face.
    aux = mode == 1 ? lim.anchor[d] : lim.aux[d]
    off, wrap, cstar = lim.offset[d], lim.wrap[d], lim.cstar[d]
    L = length(tot)
    P = decomp.dims[d]
    if P == 1
        for e in 1:L
            base = mode == 1 ? aux[e] : zero(eltype(aux))
            off[e] = base
            wrap[e] = base
            cstar[e] = mode == 1 ? zero(eltype(aux)) : aux[e]
        end
        mode == 2 && lim.measure[d] == 0 && _no_measure_error(d)
        return nothing
    end
    send, recv = lim.send[d], lim.recv[d]
    copyto!(send, 1, vec(tot), 1, L)
    copyto!(send, L + 1, vec(aux), 1, L)
    send[end] = mode == 2 && lim.measure[d] > 0 ? 1 : 0
    MPI.Allgather!(send, MPI.UBuffer(recv, length(send)), decomp.sub[d])
    stride = length(send)
    me = decomp.coords[d]
    # The lowest rank holding a face where the filter relation is interior.
    v = -1
    if mode == 2
        for r in 0:P-1
            recv[r*stride+stride] > 0 && (v = r; break)
        end
        v < 0 && _no_measure_error(d)
    end
    for e in 1:L
        O = mode == 1 ? recv[L+e] : zero(eltype(recv))
        wrap[e] = O
        Ov = O
        for r in 0:P-1
            r == me && (off[e] = O)
            r == v && (Ov = O)
            O += recv[r*stride+e]
        end
        cstar[e] = mode == 2 ? Ov + recv[v*stride+L+e] : zero(eltype(recv))
    end
    return nothing
end

@noinline _no_measure_error(d) =
    error("positivity limiter: no rank holds an interior face of dimension $d " *
          "for the filter's face relation")

# Whether this rank holds the global low and high ends of a closed line of d.
_limiter_closed(lim, decomp, d) =
    (!lim.periodic[d] && decomp.offset[d] == 0,
     !lim.periodic[d] && decomp.offset[d] + decomp.n_local[d] == decomp.n_global[d])

# θ per face of direction d into `tmp_b` (and the unguaranteed sides into
# `tmp_a`), then the faces and sides of this rank counted. Returns the number
# of faces limited that this rank owns, and whether its low face, which the
# rank below owns and counts, is limited anywhere: its node 1 takes that
# face's correction too.
function _limit_faces!(lim::PositivityLimiter, solver, Q, d::Int, τ, mode::Int)
    decomp = solver.decomp
    o1, o2, o3 = decomp.n_halo_d
    n = decomp.n_local[d]
    nA, nB = _transverse(decomp.n_local, d)
    closed_lo, closed_hi = _limiter_closed(lim, decomp, d)
    pointwise!(_limiter_theta_point!, solver.tmp_a, n + 1, nA, nB,
               solver.tmp_b, solver.tmp_a, Q, lim.faces,
               (solver.u, solver.v, solver.w), solver.c, solver.p, lim.inv_weights,
               lim.free, decomp.active, d, n, (closed_lo, closed_hi), solver.h[d],
               τ, mode, (lim.eps_rho, lim.eps_e), _limiter_layout(solver),
               o1, o2, o3)
    return _limiter_count!(lim, solver.tmp_b, solver.tmp_a, decomp, d, closed_hi, mode)
end

function _limiter_count!(lim, theta, flags, decomp, d, closed_hi, mode)
    o1, o2, o3 = decomp.n_halo_d
    n = decomp.n_local[d]
    nA, nB = _transverse(decomp.n_local, d)
    limited = 0
    shared = 0
    sides = 0
    @inbounds for b in 1:nB, a in 1:nA
        for f in 0:n
            I = _line_node(d, f, a, b, o1, o2, o3)
            if theta[I] < 1
                f >= 1 ? (limited += 1) : (shared += 1)
            end
            flag = Int(flags[I])
            f >= 1 && (flag & 1) != 0 && (sides += 1)
            f <= n - 1 && (flag & 2) != 0 && (sides += 1)
        end
    end
    faces = (closed_hi ? n - 1 : n) * nA * nB
    lim.counts[mode == 1 ? 1 : 3] += faces
    lim.counts[5] += sides
    return limited, shared
end

# --- The limited filter pass ---------------------------------------------

# `filter_state!` on an unweighted Cartesian grid with the filter limiter
# after each directional pass; with no face limited the same passes and the
# same arithmetic.
function _limited_filter_state!(solver, Q)
    lim = getfield(solver, :positivity)
    decomp = solver.decomp
    n_cons = solver.equations.n_cons
    comps = [view(Q, :, :, :, c) for c in 1:n_cons]
    n1, n2, n3 = padded_extent(decomp)
    o1, o2, o3 = decomp.n_halo_d
    for d in 1:3
        decomp.active[d] || continue
        w = filter_weight(solver, d)
        exchange_dim_batch!(comps, decomp, d)
        for c in 1:n_cons
            pointwise!(_copy_component_point!, solver.tmp_a, n1, n2, n3,
                       lim.base[c], Q, c)
        end
        for c in 1:n_cons
            filt_along!(solver.tmp_a, comps[c], solver, d, 1)
            if w == 1
                copy_interior!(comps[c], solver.tmp_a, decomp)
            else
                blend_interior!(comps[c], solver.tmp_a, w, decomp)
            end
        end
        _line_faces!(lim, solver, Q, d, 2, one(w), w)
        # The half states are read from the state before the pass.
        limited, shared = _limit_faces!(lim, solver, _BaseState(lim.base), d,
                                        zero(w), 2)
        lim.counts[4] += limited
        limited + shared > 0 || continue
        n = decomp.n_local[d]
        nA, nB = _transverse(decomp.n_local, d)
        pointwise!(_limiter_correct_point!, solver.tmp_a, n, nA, nB,
                   Q, lim.register_fields[d], lim.faces, solver.tmp_b, Q,
                   (solver.u, solver.v, solver.w), solver.c, solver.p,
                   lim.weights[d], d, solver.h[d], zero(w), 2, (one(w), one(w)),
                   _limiter_layout(solver), _limiter_closed(lim, decomp, d), n,
                   o1, o2, o3)
    end
    return Q
end

# The pre-pass state of a filter pass, held per component in the workspace,
# indexed as a conserved state is.
struct _BaseState{F}
    fields::F
end
Base.@propagate_inbounds Base.getindex(s::_BaseState, I::CartesianIndex{3}, c::Int) =
    s.fields[c][I]

# --- Per-point bodies --------------------------------------------------------

# r ← A r + dt div over the interior; RKA[1] = 0, and the first stage assigns,
# so a register left non-finite by an abandoned step is forgotten.
@inline function _limiter_register_point!(r, div, A, dt, o1, o2, o3, i, j, k)
    @inbounds begin
        I = CartesianIndex(i + o1, j + o2, k + o3)
        r[I] = ifelse(iszero(A), dt * div[I], A * r[I] + dt * div[I])
    end
    return nothing
end

# The face register at the global low face of each line, Φ ← A Φ + dt F̂:
# the point flux of node 1 on a closed line (the wall-corrected flux), and on
# a periodic line the constant the interior face relation gives the running
# sum of this stage's divergence, measured at local face `j` of the rank
# holding the global low face.
@inline function _limiter_anchor_point!(anchor, F, div, W, A, dt, c, d, periodic,
                                        j, lhs, rhs, o1, o2, o3, a, b, _)
    @inbounds begin
        o = d == 1 ? o1 : d == 2 ? o2 : o3
        if periodic
            q = length(lhs)
            T = eltype(anchor)
            S = zero(T)
            lo = j - q
            acc = zero(T)
            sum_l = one(T)
            for s in 1:q
                sum_l += 2 * lhs[s]
            end
            for p in 1:j+q
                S += W[p+o] * div[_line_node(d, p, a, b, o1, o2, o3)]
                if p >= lo
                    s = p - j
                    acc += (s == 0 ? one(T) : lhs[abs(s)]) * S
                end
            end
            # The face j − q may be face 0, whose running sum is zero.
            G = zero(T)
            for m in eachindex(rhs)
                for l in 1-m:m
                    G += rhs[m] * F[_line_node(d, j + l, a, b, o1, o2, o3)]
                end
            end
            value = (G - acc) / sum_l
        else
            value = F[_line_node(d, 1, a, b, o1, o2, o3)]
        end
        anchor[c, a, b] = ifelse(iszero(A), dt * value, A * anchor[c, a, b] + dt * value)
    end
    return nothing
end

# One line's local running sums into the face slots: slot p holds face p + ½,
# slot 0 (a halo node) the low face of the block, at zero. Mode 1 sums W r,
# mode 2 the filter correction −ω Δ with ω = W/h, and in mode 2 the filter's
# face relation is measured at local face `j` when j > 0:
# (Σ_s l_s S_{j+s} + w ψ_j) / Σ_s l_s.
@inline function _limiter_scan_point!(faces, regs, Q, base, W, hd, totals, aux,
                                      n_cons, n, d, mode, j, lhs, psi, wf,
                                      o1, o2, o3, a, b, _)
    @inbounds begin
        o = d == 1 ? o1 : d == 2 ? o2 : o3
        T = eltype(totals)
        for c in 1:n_cons
            fc = faces[c]
            S = zero(T)
            fc[_line_node(d, 0, a, b, o1, o2, o3)] = S
            for p in 1:n
                I = _line_node(d, p, a, b, o1, o2, o3)
                if mode == 1
                    S += W[p+o] * regs[c][I]
                else
                    S -= (W[p+o] / hd) * (Q[I, c] - base[c][I])
                end
                fc[I] = S
            end
            totals[c, a, b] = S
            if mode == 2 && j > 0
                q = length(lhs)
                acc = fc[_line_node(d, j, a, b, o1, o2, o3)]
                sum_l = one(T)
                for s in 1:q
                    acc += lhs[s] * (fc[_line_node(d, j - s, a, b, o1, o2, o3)] +
                                     fc[_line_node(d, j + s, a, b, o1, o2, o3)])
                    sum_l += 2 * lhs[s]
                end
                M = length(psi) ÷ 2
                ψ = zero(T)
                for l in 1-M:M
                    ψ += psi[l+M] * base[c][_line_node(d, j + l, a, b, o1, o2, o3)]
                end
                aux[c, a, b] = (acc + wf * ψ) / sum_l
            end
        end
    end
    return nothing
end

# Face f = f1 − 1 of a line: scale · ((O + S_f) − c*). On the last rank of a
# periodic line the high face is the global low face, whose value the first
# rank holds.
@inline function _limiter_offset_point!(faces, offset, wrap, cstar, scale, wrap_high,
                                        n, d, n_cons, o1, o2, o3, f1, a, b)
    @inbounds begin
        f = f1 - 1
        I = _line_node(d, f, a, b, o1, o2, o3)
        for c in 1:n_cons
            v = wrap_high && f == n ? wrap[c, a, b] : offset[c, a, b] + faces[c][I]
            faces[c][I] = scale * (v - cstar[c, a, b])
        end
    end
    return nothing
end

# Mixture density, momentum and total energy of a state.
@inline function _limiter_state(Q, I, lay)
    ns, m1, m2, m3, ie, _ = lay
    ρ = zero(eltype(Q[I, 1]))
    for sp in 1:ns
        ρ += Q[I, sp]
    end
    return (ρ, Q[I, m1], Q[I, m2], Q[I, m3], Q[I, ie])
end

@inline function _limiter_face_state(faces, I, lay)
    ns, m1, m2, m3, ie, _ = lay
    ρ = zero(eltype(faces[1]))
    for sp in 1:ns
        ρ += faces[sp][I]
    end
    return (ρ, faces[m1][I], faces[m2][I], faces[m3][I], faces[ie][I])
end

# q + s G, componentwise over the five aggregated values.
@inline _limiter_axpy(q, s, G) = map((x, y) -> x + s * y, q, G)

@inline _limiter_internal(q) = q[5] - (q[2] * q[2] + q[3] * q[3] + q[4] * q[4]) / (2 * q[1])
# ρ ≥ ε_ρ and ρe ≥ ε_e, the second multiplied through by 2ρ > 0.
@inline _limiter_admissible(q, ε) =
    q[1] >= ε[1] &&
    2 * q[1] * q[5] - (q[2] * q[2] + q[3] * q[3] + q[4] * q[4]) >= 2 * q[1] * ε[2]

# τ times the Lax–Friedrichs flux of the aggregated state along d between
# nodes L and R, whose wave speed is the larger of the two.
@inline function _limiter_lf_state(qL, qR, uL, uR, pL, pR, a, τ, d)
    fL = (qL[1] * uL, qL[2] * uL + (d == 1) * pL, qL[3] * uL + (d == 2) * pL,
          qL[4] * uL + (d == 3) * pL, (qL[5] + pL) * uL)
    fR = (qR[1] * uR, qR[2] * uR + (d == 1) * pR, qR[3] * uR + (d == 2) * pR,
          qR[4] * uR + (d == 3) * pR, (qR[5] + pR) * uR)
    return map((x, y, u, v) -> τ * ((x + y) / 2 - a * (v - u) / 2), fL, fR, qL, qR)
end

# The limit of one cell side: the largest θ for which q + s (θ G + (1 − θ) G_L)
# keeps ρ and ρe at or above ε, by linear interpolation in ρ and then in ρe
# along the shortened segment, sufficient since ρe is concave. Returns θ and
# whether the side is guaranteed.
@inline function _limiter_side(q, G, GL, s, ε)
    T = typeof(s)
    qH = _limiter_axpy(q, s, G)
    _limiter_admissible(qH, ε) && return (one(T), true)
    qL = _limiter_axpy(q, s, GL)
    _limiter_admissible(qL, ε) || return (zero(T), false)
    θ = qH[1] < ε[1] ? (qL[1] - ε[1]) / (qL[1] - qH[1]) : one(T)
    qθ = _limiter_axpy(qL, θ, map(-, qH, qL))
    eL = _limiter_internal(qL)
    e = _limiter_internal(qθ)
    e < ε[2] && (θ *= (eL - ε[2]) / (eL - e))
    return (clamp(θ, zero(T), one(T)), true)
end

# Σ_d (|u_d| + c)/W_d at node I, the cell's total first-order rate, from the
# inverse weights.
@inline function _limiter_rate(uvw, c, inv_W, act, I)
    Λ = zero(eltype(c))
    for e in 1:3
        act[e] || continue
        Λ += (abs(uvw[e][I]) + c[I]) * inv_W[e][I[e]]
    end
    return Λ
end

@inline _limiter_constrained(free, I) = !(free[1][I[1]] | free[2][I[2]] | free[3][I[3]])

# θ at face f = f1 − 1 of a line of direction d into `theta`, and its
# unguaranteed sides into `flags` (1 the low cell, 2 the high one). A global
# closed end is not limited. Mode 1 is a stage (half states scaled by
# 2Λ/(|u_d| + c) and the first-order flux τ F_LF); mode 2 a filter pass (2/ω,
# toward zero).
@inline function _limiter_theta_point!(theta, flags, Q, faces, uvw, c, p, inv_W, free,
                                       act, d, n, closed, hd, τ, mode, ε, lay,
                                       o1, o2, o3, f1, a, b)
    @inbounds begin
        T = eltype(theta)
        f = f1 - 1
        Il = _line_node(d, f, a, b, o1, o2, o3)
        Ir = _line_node(d, f + 1, a, b, o1, o2, o3)
        G = _limiter_face_state(faces, Il, lay)
        flag = 0
        θ = one(T)
        lo_bnd = closed[1] && f == 0
        hi_bnd = closed[2] && f == n
        ud = uvw[d]
        sl = zero(T); sr = zero(T)
        if mode == 1
            spl = abs(ud[Il]) + c[Il]
            spr = abs(ud[Ir]) + c[Ir]
            !lo_bnd && (sl = spl > 0 ? -2 * _limiter_rate(uvw, c, inv_W, act, Il) / spl :
                             T(-Inf))
            !hi_bnd && (sr = spr > 0 ? 2 * _limiter_rate(uvw, c, inv_W, act, Ir) / spr :
                             T(Inf))
        else
            !lo_bnd && (sl = -2 * hd * inv_W[d][Il[d]])
            !hi_bnd && (sr = 2 * hd * inv_W[d][Ir[d]])
        end
        if !(lo_bnd || hi_bnd)
            # The filter's row at a closed end is the identity, and its
            # correction leaves the end node alone; so does the limiter.
            wall_l = closed[1] && f == 1
            wall_r = closed[2] && f == n - 1
            cl = _limiter_constrained(free, Il) && !(mode == 2 && wall_l)
            cr = _limiter_constrained(free, Ir) && !(mode == 2 && wall_r)
            ql = _limiter_state(Q, Il, lay)
            qr = _limiter_state(Q, Ir, lay)
            # At a stage, a cell on a closed end keeps its boundary face, which
            # is never limited, in the same part as its interior face: one part
            # of weight α_d, q + (G_boundary − G_interior)/(α_d W), in place of
            # two halves, so the wall's flux enters at its own size, not twice.
            bl = ql; br = qr
            if mode == 1 && wall_l
                sl /= 2
                bl = _limiter_axpy(ql, -sl, _limiter_face_state(faces,
                                   _line_node(d, 0, a, b, o1, o2, o3), lay))
            end
            if mode == 1 && wall_r
                sr /= 2
                br = _limiter_axpy(qr, -sr, _limiter_face_state(faces,
                                   _line_node(d, n, a, b, o1, o2, o3), lay))
            end
            need = (cl && !_limiter_admissible(_limiter_axpy(bl, sl, G), ε)) ||
                   (cr && !_limiter_admissible(_limiter_axpy(br, sr, G), ε))
            if need
                GL = (zero(T), zero(T), zero(T), zero(T), zero(T))
                a_face = zero(T)
                if mode == 1
                    a_face = max(abs(ud[Il]) + c[Il], abs(ud[Ir]) + c[Ir])
                    GL = _limiter_lf_state(ql, qr, ud[Il], ud[Ir], p[Il], p[Ir], a_face,
                                           τ, d)
                end
                # A side whose first-order half state is not admissible, or
                # whose first-order bound τ a |s| ≤ 1 fails, is unguaranteed.
                if cl
                    t, ok = _limiter_side(bl, G, GL, sl, ε)
                    θ = min(θ, t)
                    (ok && τ * a_face * abs(sl) <= 1) || (flag |= 1)
                end
                if cr
                    t, ok = _limiter_side(br, G, GL, sr, ε)
                    θ = min(θ, t)
                    (ok && τ * a_face * abs(sr) <= 1) || (flag |= 2)
                end
            end
        end
        theta[Il] = θ
        flags[Il] = flag
    end
    return nothing
end

# τ times component `cc` of the Lax–Friedrichs flux along d between L and R.
@inline function _limiter_lf_component(Q, uvw, c, p, cc, L, R, τ, d, lay)
    ns, m1, m2, m3, ie, _ = lay
    ud = uvw[d]
    uL, uR = ud[L], ud[R]
    a = max(abs(uL) + c[L], abs(uR) + c[R])
    qL, qR = Q[L, cc], Q[R, cc]
    fL = qL * uL
    fR = qR * uR
    if cc == ie
        fL += p[L] * uL
        fR += p[R] * uR
    elseif (cc == m1 && d == 1) || (cc == m2 && d == 2) || (cc == m3 && d == 3)
        fL += p[L]
        fR += p[R]
    end
    return τ * ((fL + fR) / 2 - a * (qR - qL) / 2)
end

# The corrections at node p of a line from its two faces, where either is
# limited: δ = (θ − 1)(G − G_L) per face, node change (δ_{p−½} − δ_{p+½})/W_p.
# Mode 1 adds it to dQ / (B dt) and takes it out of the register, r ← r −
# change / B; mode 2 adds it to the state, with ω = W/h in place of W.
@inline function _limiter_correct_point!(target, regs, faces, theta, Q, uvw, c, pr,
                                         W, d, hd, τ, mode, scales, lay, closed, n,
                                         o1, o2, o3, pp, a, b)
    @inbounds begin
        T = eltype(theta)
        # A filter pass leaves a closed end's node alone (see the face pass).
        mode == 2 && ((closed[1] && pp == 1) || (closed[2] && pp == n)) &&
            return nothing
        Im = _line_node(d, pp - 1, a, b, o1, o2, o3)
        I = _line_node(d, pp, a, b, o1, o2, o3)
        Ip = _line_node(d, pp + 1, a, b, o1, o2, o3)
        θm = theta[Im]
        θp = theta[I]
        (θm < 1 || θp < 1) || return nothing
        o = d == 1 ? o1 : d == 2 ? o2 : o3
        Wp = mode == 1 ? W[pp+o] : W[pp+o] / hd
        n_cons = lay[6]
        inv_B, inv_Bdt = scales
        for cc in 1:n_cons
            change = zero(T)
            if θm < 1
                gl = mode == 1 ? _limiter_lf_component(Q, uvw, c, pr, cc, Im, I, τ, d,
                                                       lay) : zero(T)
                change += (θm - 1) * (faces[cc][Im] - gl)
            end
            if θp < 1
                gl = mode == 1 ? _limiter_lf_component(Q, uvw, c, pr, cc, I, Ip, τ, d,
                                                       lay) : zero(T)
                change -= (θp - 1) * (faces[cc][I] - gl)
            end
            change /= Wp
            if mode == 1
                target[I, cc] += change * inv_Bdt
                regs[cc][I] -= change * inv_B
            else
                target[I, cc] += change
            end
        end
    end
    return nothing
end
