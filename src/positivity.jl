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
# limited is the unlimited run bit for bit. On a Cartesian line the node terms
# (sources, the NSCBC corrections) are outside the face form and outside the
# guarantee. The partial densities are not bounded: their interface
# undershoots are inside the species band by design, and a bound there would
# act at every captured interface.
#
# The bound. ε is a fraction of the minimum ρ and ρe of the state entering
# `run!`, and each cell keeps the smaller of ε and `LIMITER_CELL_FRACTION` of
# its own base value (the stage's base state, or the state before a filter
# pass). The first-order update keeps a state positive, not above a fixed ε,
# so a cell far below the initial minimum (the evacuated core of a blast)
# would otherwise have first-order half states under the bound and no
# guarantee. The fraction is the same 1%: a larger one (0.1, 0.5) let more
# cells settle below the run's ε, where the artificial conductivity sizes the
# step, and added up to half the steps on Woodward–Colella and planar Noh.
#
# Radial lines. On the radial lines of `CylindricalMetric` (θ collapsed) and
# `SphericalMetric` (θ and φ collapsed), folded at r = 0 by `AxisBC` or
# `OriginBC`, the folded divergence is the interior scheme on the mirrored
# data. Its weights are h on the fold side and the closed line's tail at the
# far end, for either parity, and the face value at the fold is zero for an
# odd area-weighted flux and, for an even one, the value the interior face
# relation gives at face 0 with the mirrored point values. Mass and energy
# take the area-weighted fluxes, D(A F), and the radial momentum takes its
# pressure as a gradient with a register of its own, so a node changes by
# −inv_J (Φ_{i+½} − Φ_{i−½})/W_i − inv_h (Π_{i+½} − Π_{i−½})/W_i. The
# first-order counterpart is the Rusanov flux of the area-weighted fluxes on
# the dual face areas Ā_k = A(Σ_{j≤k} W_j), Ā = 0 at the fold, with the
# pressure face value {p} (p_1 at the fold). Each cell's half states subtract
# its own τ Ā G(Q_i) and τ p_i, so that a cell splits into a part per face, a
# geometric part −τ (Ā_{i+½} − Ā_{i−½}) G(Q_i)/V_i that no face changes, and
# a part for the node terms. Each face part is a Lax–Friedrichs half state of
# G + ζ p e_r, ζ = J_i/Ā the cell's pressure fraction at that face, and is
# admissible at the speed `_radial_speed` gives, which exceeds |u| + c where
# ζ ≠ 1. The shares are proportional to Ā a/V per face (the fold face takes
# `AXIS_SHARE` of the outer face's), γ u⁺ ΔĀ/V for the geometric part, and
# the node part takes the smallest share that keeps it admissible, computed
# in closed form; a cell is guaranteed when τ times its rate over the share
# left is below one. A cell whose node part alone would leave the bound (the
# hoop stress of the artificial bulk viscosity at a shock, accelerating the
# cold gas) is counted unguaranteed and takes the first-order flux at both
# its faces: on Sedov the alternative, the shares without the node part,
# left inadmissible points behind the front. Spherical mass and energy fluxes
# are odd at the origin, so the limited scheme conserves Σ W J Q there; the
# cylindrical ones are even and the axis face carries a flux of the scheme
# itself.
#
# The r-z plane. Along z the divergence is D(r F) with r constant on a line,
# so the z faces take the Cartesian form with the face values divided by J = r
# and the first-order flux multiplied by it, and each cell's rate is the one
# the radial pass computed, which already holds 2(|u_z| + c)/W_z. A symmetry
# plane at z = 0 is a fold whose face is limited from node 1 alone, its
# first-order flux the Rusanov flux against node 1's mirror, which carries the
# reflected pressure in the normal momentum and nothing in the rest; the
# corner cell takes the axis share, the plane face and its two interior faces.
#
# Per-point work runs through `pointwise!` bodies over three index boxes: the
# lines of a direction (one sequential scan per line), its faces, and its
# nodes. The running sum along a decomposed line takes one `Allgather` of the
# line totals over the direction's sub-communicator per direction per stage
# and per filter pass, entered by every rank of it.

# ε is this fraction of the minimum ρ and ρe of the state entering `run!`.
const LIMITER_FRACTION = 0.01
# A cell's bound is at most this fraction of its own base ρ and ρe.
const LIMITER_CELL_FRACTION = LIMITER_FRACTION
# The share of the face at r = 0 on a radial line, against the cell's outer
# face. It bounds the first-order step at the fold's node: at the spherical
# origin τ (|u| + c)/h < 1/(4 (1 + AXIS_SHARE)). The cylindrical axis face
# carries the scheme's own O(h²) flux, up to a twelfth of node 1's rate, which
# a limited axis face removes; on cylindrical Noh that happens at a handful of
# stages.
const AXIS_SHARE = 0.25

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
    registers::Matrix{A}            # r[d, c]; c = n_cons + 1 the radial pressure
    register_fields::Vector{FieldVector{A,Vector{A}}}   # r[d, :] per d
    faces::FieldVector{A,Vector{A}} # face values, in the workspace's flux[1, :],
                                    # and the pressure's in flux[3, 1]
    base::FieldVector{A,Vector{A}}  # the filter's pre-pass state, flux[2, :]
    rates::A                        # a radial cell's rate, flux[2, 1] at a stage
    areas::NTuple{3,V}              # Ā at the face after each padded position
    radial::NTuple{3,Bool}          # d is a radial line folded at r = 0
    fold_lo::NTuple{3,Bool}         # the global low end of d is a fold
    n_comp::NTuple{3,Int}           # registers of d: n_cons, and the pressure
    volume::Vector{Bool}            # per component (and a trailing false): a
                                    # filter pass along a radial line weights
                                    # it by J
    volume_state::NTuple{5,Bool}    # the same for ρ, the momenta and E
    areal::NTuple{3,Bool}           # d takes area-weighted fluxes, D(A_d F), on
                                    # lines of constant A_d (z on the r-z plane)
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
    # unguaranteed cell sides, and stage faces limited at r = 0, rank-local;
    # `positivity_counts` reduces.
    counts::Vector{Int}
end

# The radial grids the limiter covers: dimension 1 folded at r = 0, on the r-z
# plane of a `CylindricalMetric` (θ collapsed, z resolved or not) with
# `AxisBC`, or on the radial line of a `SphericalMetric` (θ and φ collapsed)
# with `OriginBC`.
_limiter_radial_line(bcs, metric, n_global) =
    n_global[1] > 1 && n_global[2] == 1 &&
    (metric isa CylindricalMetric && bcs[1][1] isa AxisBC ||
     metric isa SphericalMetric && bcs[1][1] isa OriginBC && n_global[3] == 1)

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

# The weights of a line of N nodes folded at its low end: h on the fold side
# and the far half of a closed line of 2N nodes, whose tail holds the closure
# rows' weights. The folded divergence is the interior scheme on the mirrored
# data, so these serve either parity of the folded field.
function _limiter_fold_weights(scheme, N::Int, h, n_halo::Int, ::Type{T}) where {T}
    W = _limiter_weights(scheme, 2N, h, n_halo, T)[1][N+1:2N]
    W[1:N÷2] .= h
    return W
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
# or a radial grid folded at r = 0 (`_limiter_radial_line`), on the r-z plane
# with a symmetry plane allowed at the low end of z, without levels or the
# implicit integrator, an ideal-gas mixture, and closed lines long enough for
# the face relation of the filter.
function _validate_positivity(bcs, metric, stretch, patch_grid, nlev, backend, eos,
                              implicit, equations, deriv, filt, filter_weighting,
                              n_global, n_halo, L_domain, ::Type{T}) where {T}
    fail(what) = throw(ArgumentError("positivity_limiter: $what"))
    radial = _limiter_radial_line(bcs, metric, n_global)
    metric isa CartesianMetric || radial ||
        fail("supports the CartesianMetric, the r-z plane of a CylindricalMetric " *
             "with θ collapsed and AxisBC at r = 0, and the radial line of a " *
             "SphericalMetric with θ and φ collapsed and OriginBC at r = 0")
    all(isnothing, stretch) || fail("supports an unstretched grid only")
    radial && filter_weighting !== :none &&
        fail("on a radial grid takes the filter weighting :none")
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
        radial && d == 1 && side == 1 && continue
        radial && d == 3 && side == 1 && bc isa SymmetryPlaneBC && continue
        bc isa Union{SymmetryPlaneBC,AxisBC,OriginBC,PoleBC} &&
            fail("does not support folded ends other than r = 0 of a radial " *
                 "grid and z = 0 of an r-z plane; face $d/$side carries " *
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
    fold_lo = ntuple(d -> decomp.active[d] && solver.folds[d] !== nothing &&
                          solver.folds[d].lo, 3)
    curved = !(solver.metric isa CartesianMetric)
    radial = (fold_lo[1] && curved, false, false)
    areal = ntuple(d -> decomp.active[d] && curved && !radial[d], 3)
    # The weights along each global line, and the nodes from a closed end
    # beyond which the filter's face relation is interior. A folded line's
    # far end closes with the derivative's rows.
    global_weights = ntuple(3) do d
        decomp.active[d] || return Float64[1.0]
        N = decomp.n_global[d]
        decomp.periodic[d] && return fill(Float64(solver.h[d]), N)
        fold_lo[d] && return _limiter_fold_weights(deriv, N, solver.h[d],
                                                   decomp.n_halo, T)
        return _limiter_weights(plan_scheme(d), N, solver.h[d], decomp.n_halo, T)[1]
    end
    # The dual face areas of a radial line, Ā_k = A(Σ_{j≤k} W_j) at face k + ½,
    # 0 at the fold, where A is the metric's radial area factor J/h_1.
    areas = ntuple(3) do d
        radial[d] || return T[]
        n = decomp.n_local[d]
        o = decomp.n_halo_d[d]
        N = decomp.n_global[d]
        position = [0.0; cumsum(global_weights[d])]
        x = ntuple(e -> _phys_and_jac(solver, e, decomp.n_halo_d[e] + 1)[1], 3)
        out = zeros(T, n + 2o)
        for p in 1:n+2o
            g = decomp.offset[d] + p - o
            0 <= g <= N || continue
            x1, x2, x3 = ntuple(e -> e == d ? position[g+1] : x[e], 3)
            h1, h2, h3 = scalefactors(solver.metric, x1, x2, x3)
            out[p] = T(h1 * h2 * h3 / (d == 1 ? h1 : d == 2 ? h2 : h3))
        end
        out
    end
    n_comp = ntuple(d -> radial[d] ? n_cons + 1 : n_cons, 3)
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
    registers = [decomp.active[d] && c <= n_comp[d] ? zero(solver.tmp_a) : empty
                 for d in 1:3, c in 1:n_cons+1]
    register_fields = [FieldVector([registers[d, c] for c in 1:n_comp[d]])
                       for d in 1:3]
    planes = [zeros(T, n_comp[d], _transverse(decomp.n_local, d)...) for d in 1:3]
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
    faces = [ws[1, c] for c in 1:n_cons]
    any(radial) && push!(faces, ws[3, 1])
    # On a radial line a limited filter pass weights the components even at
    # the fold by J, so that its corrections telescope in Σ W J q, the sum the
    # spherical filter conserves (its correction annihilates the even
    # polynomial r² on the mirrored line), and their running sum starts from
    # zero at the fold. An odd one, the radial momentum, has no conservation
    # law and keeps W and the constant of the filter's face relation.
    dr = findfirst(radial)
    volume = [dr !== nothing && c <= n_cons && cons_parity(solver, dr, c) == 1
              for c in 1:n_cons+1]
    lay = _limiter_layout(solver)
    volume_state = (volume[1], volume[lay[2]], volume[lay[3]], volume[lay[4]],
                    volume[lay[5]])
    return PositivityLimiter{T,typeof(solver.tmp_a),typeof(weights[1])}(
        weights, map(w -> one(T) ./ w, weights), free, registers, register_fields,
        FieldVector(faces), FieldVector([ws[2, c] for c in 1:n_cons]), ws[2, 1],
        areas, radial, fold_lo, n_comp, volume, volume_state, areal,
        copies(), copies(), copies(), copies(), copies(), copies(), copies(),
        send, recv, _band_lhs(deriv), copy(deriv.coeffs), filter_lhs,
        _filter_face_stencil(filt, T), decomp.periodic, measure, anchor_face,
        zero(T), zero(T), false, zeros(Int, 6))
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
filter passes, the faces limited in each, the unguaranteed cell sides, where
the first-order bound failed or the first-order half state was not
admissible, on a radial grid also those of a cell whose source and boundary
terms alone would leave the bound, and on a radial grid the stage faces
limited at r = 0 (`axis_limited`, counted in `stage_limited` too). Collective
over the solver's communicator.
"""
function positivity_counts(solver)
    lim = getfield(solver, :positivity)
    lim === nothing && throw(ArgumentError(
        "positivity_counts: the solver was built without positivity_limiter"))
    c = MPI.Allreduce(lim.counts, +, solver.comm)
    return (stage_faces=c[1], stage_limited=c[2], filter_faces=c[3],
            filter_limited=c[4], unguaranteed=c[5], axis_limited=c[6])
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
        _limit_stage!(lim, solver, Q, dQ, du, stage, dt)
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

div_subtract_along!(dQ::_LimiterRHS, c::Int, f, solver::SolverLike, d::Int, σf::Int,
                    inv_J) = _limited_subtract!(dQ, c, c, f, solver, d, σf, inv_J)

# The radial pressure gradient, kept in register n_cons + 1 of its direction.
pressure_subtract_along!(dQ::_LimiterRHS, c::Int, solver::SolverLike, d::Int) =
    _limited_subtract!(dQ, c, solver.equations.n_cons + 1, solver.p, solver, d, 1,
                       solver.inv_h[d])

# The subtraction from component `c` and the update of register `slot`.
function _limited_subtract!(dQ::_LimiterRHS, c::Int, slot::Int, f, solver, d::Int,
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
               lim.registers[d, slot], solver.tmp_a, dQ.A, dQ.dt, o1, o2, o3)
    if decomp.coords[d] == 0
        nA, nB = _transverse(decomp.n_local, d)
        pointwise!(_limiter_anchor_point!, solver.tmp_a, nA, nB, 1,
                   lim.anchor[d], f, solver.tmp_a, lim.weights[d], dQ.A, dQ.dt, slot, d,
                   lim.periodic[d], lim.fold_lo[d], σf, lim.anchor_face[d],
                   lim.deriv_lhs, lim.deriv_rhs, o1, o2, o3)
    end
    return dQ
end

function _limit_stage!(lim::PositivityLimiter, solver, Q, dQ, du, stage::Int, dt)
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
        if lim.radial[d]
            _limit_radial!(lim, solver, Q, dQ, du, d, T(RKA[stage]), B, dtt, τ)
            continue
        end
        limited, shared = _limit_faces!(lim, solver, Q, d, τ, 1)
        lim.counts[2] += limited
        limited + shared > 0 || continue
        n = decomp.n_local[d]
        nA, nB = _transverse(decomp.n_local, d)
        pointwise!(_limiter_correct_point!, solver.tmp_a, n, nA, nB,
                   dQ, lim.register_fields[d], lim.faces, solver.tmp_b, Q,
                   (solver.u, solver.v, solver.w), solver.c, solver.p,
                   lim.weights[d], d, one(T), τ, 1, (inv_B, inv_Bdt),
                   _limiter_layout(solver),
                   (_limiter_closed(lim, decomp, d)..., _limiter_fold(lim, decomp, d)), n,
                   false, lim.volume, solver.inv_J, lim.areal[d], lim.anchor[d],
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
    # A stage takes every register of d, the radial pressure's included; a
    # filter pass the conserved components.
    n_comp = mode == 1 ? lim.n_comp[d] : solver.equations.n_cons
    volume = mode == 2 && lim.radial[d]
    pointwise!(_limiter_scan_point!, solver.tmp_a, nA, nB, 1,
               lim.faces, lim.register_fields[d], Q, lim.base, lim.weights[d],
               solver.h[d], lim.totals[d], lim.aux[d], n_comp, n, d, mode,
               mode == 2 ? lim.measure[d] : 0, lim.filter_lhs, lim.filter_psi, wf,
               volume, lim.volume, solver.inv_J, o1, o2, o3)
    _line_offsets!(lim, decomp, d, mode)
    last_rank = decomp.coords[d] == decomp.dims[d] - 1
    pointwise!(_limiter_offset_point!, solver.tmp_a, n + 1, nA, nB,
               lim.faces, lim.offset[d], lim.wrap[d],
               mode == 1 ? lim.zero_plane[d] : lim.cstar[d], scale,
               lim.periodic[d] && last_rank, n, d, n_comp, volume, lim.volume,
               o1, o2, o3)
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

# Whether this rank holds the global low and high ends of a closed line of d;
# a fold is not a closed end, and its face is limited.
_limiter_closed(lim, decomp, d) =
    (!lim.periodic[d] && !lim.fold_lo[d] && decomp.offset[d] == 0,
     !lim.periodic[d] && decomp.offset[d] + decomp.n_local[d] == decomp.n_global[d])

# Whether this rank holds the fold at the global low end of d.
_limiter_fold(lim, decomp, d) = lim.fold_lo[d] && decomp.offset[d] == 0

# The bound of a cell whose base state aggregates to q: the smaller of the run's
# ε and the fraction ε[3] of the cell's own ρ and ρe, at least zero.
@inline function _limiter_cell_bound(q, ε)
    ρ = q[1]
    e = ρ > 0 ? _limiter_internal(q) : zero(ρ)
    return (min(ε[1], ε[3] * max(ρ, zero(ρ))), min(ε[2], ε[3] * max(e, zero(e))))
end

# The run's bounds and the cell fraction, as the per-point bodies take them.
_limiter_bounds(lim) = (lim.eps_rho, lim.eps_e, oftype(lim.eps_rho, LIMITER_CELL_FRACTION))

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
    fold = _limiter_fold(lim, decomp, d)
    pointwise!(_limiter_theta_point!, solver.tmp_a, n + 1, nA, nB,
               solver.tmp_b, solver.tmp_a, Q, lim.faces,
               (solver.u, solver.v, solver.w), solver.c, solver.p, lim.inv_weights,
               lim.free, decomp.active, d, n, (closed_lo, closed_hi, fold), solver.h[d],
               τ, mode, _limiter_bounds(lim), _limiter_layout(solver),
               mode == 2 ? lim.radial[d] : lim.areal[d],
               mode == 2 ? lim.volume_state : (true, true, true, true, true), solver.inv_J,
               mode == 1 && lim.areal[d], lim.rates, o1, o2, o3)
    return _limiter_count!(lim, solver.tmp_b, solver.tmp_a, decomp, d, closed_hi, mode,
                           fold)
end

# A fold's face is this rank's own and counted with its faces.
function _limiter_count!(lim, theta, flags, decomp, d, closed_hi, mode, fold)
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
                (f >= 1 || fold) ? (limited += 1) : (shared += 1)
                mode == 1 && fold && f == 0 && lim.radial[d] && (lim.counts[6] += 1)
            end
            flag = Int(flags[I])
            f >= 1 && (flag & 1) != 0 && (sides += 1)
            f <= n - 1 && (flag & 2) != 0 && (sides += 1)
        end
    end
    faces = ((closed_hi ? n - 1 : n) + (fold ? 1 : 0)) * nA * nB
    lim.counts[mode == 1 ? 1 : 3] += faces
    lim.counts[5] += sides
    return limited, shared
end

# --- The stage on a radial line ------------------------------------------

# The stage limiter along a radial line d: each node's rate, θ per face, and
# the corrections, which a limited fold face also takes into the face
# register at the fold. `A` and `B` are the stage's low-storage coefficients.
function _limit_radial!(lim::PositivityLimiter, solver, Q, dQ, du, d::Int, A, B, dt, τ)
    decomp = solver.decomp
    o1, o2, o3 = decomp.n_halo_d
    n = decomp.n_local[d]
    nA, nB = _transverse(decomp.n_local, d)
    closed_lo, closed_hi = _limiter_closed(lim, decomp, d)
    fold = _limiter_fold(lim, decomp, d)
    uvw = (solver.u, solver.v, solver.w)
    lay = _limiter_layout(solver)
    bounds = _limiter_bounds(lim)
    regs = (lim.register_fields[1], lim.register_fields[2], lim.register_fields[3])
    pointwise!(_limiter_radial_rate_point!, lim.rates, n, nA, nB,
               lim.rates, Q, dQ, du, regs, uvw, solver.c, solver.p, solver.rho,
               solver.inv_J, solver.inv_h[d], lim.weights, lim.areas[d], decomp.active,
               d, (fold, closed_hi, n), (A, B, dt), bounds, lay, o1, o2, o3)
    # The face two ranks share takes the same θ on both, from both cells'
    # rates, along r here and along z in the pass that follows.
    for e in 1:3
        decomp.active[e] && decomp.dims[e] > 1 && exchange_dim!(lim.rates, decomp, e)
    end
    pointwise!(_limiter_radial_theta_point!, solver.tmp_a, n + 1, nA, nB,
               solver.tmp_b, solver.tmp_a, Q, lim.faces, lim.rates, uvw, solver.c,
               solver.p, solver.rho, solver.inv_J, solver.inv_h[d], lim.weights[d],
               lim.areas[d], lim.free, d, (fold, closed_lo, closed_hi, n), τ, bounds,
               lay, o1, o2, o3)
    limited, shared = _limiter_count!(lim, solver.tmp_b, solver.tmp_a, decomp, d,
                                      closed_hi, 1, fold)
    lim.counts[2] += limited
    limited + shared > 0 || return nothing
    pointwise!(_limiter_radial_correct_point!, solver.tmp_a, n, nA, nB,
               dQ, lim.register_fields[d], lim.faces, solver.tmp_b, Q, uvw, solver.c,
               solver.p, solver.rho, solver.inv_J, solver.inv_h[d], lim.weights[d],
               lim.areas[d], lim.anchor[d], d, τ, (one(B) / B, one(B) / (B * dt)), lay,
               fold, o1, o2, o3)
    return nothing
end

@inline _limiter_gamma(ρ, c, p) = p > 0 ? max(ρ * c * c / p, one(p)) : one(p)

# The Lax–Friedrichs speed at which Q ∓ F̃(Q)/a is admissible for the flux
# F̃ = G + ζ p e_r of a cell whose pressure fraction at the face is ζ:
# (1 − λv)(1 − λv(1 + (γ − 1)(1 − ζ))) > λ² ζ² c² (γ − 1)/(2γ), λ = 1/a,
# v = ±u, holds for a ≥ max(1, |1 + (γ − 1)(1 − ζ)|)|u| + ζ c √((γ − 1)/(2γ)),
# and the second term is kept at least c so that ζ = 1 gives |u| + c.
@inline function _radial_speed(u, c, γ, ζ)
    k = abs(1 + (γ - 1) * (1 - ζ))
    return max(one(k), k) * abs(u) + max(one(k), ζ * sqrt((γ - 1) / (2γ))) * c
end

# The speed at a face of area Ā between nodes L and R: the largest over both
# states and both cells' ζ = (inv_h / inv_J) / Ā, since each cell's half state
# holds both states. With R === L, the speed of one state at a closed end.
@inline function _radial_face_speed(ud, c, p, ρ, inv_J, inv_h, Ā, L, R)
    γL = _limiter_gamma(ρ[L], c[L], p[L])
    γR = _limiter_gamma(ρ[R], c[R], p[R])
    ζL = inv_h[L] / (inv_J[L] * Ā)
    ζR = inv_h[R] / (inv_J[R] * Ā)
    return max(_radial_speed(ud[L], c[L], γL, ζL), _radial_speed(ud[L], c[L], γL, ζR),
               _radial_speed(ud[R], c[R], γR, ζL), _radial_speed(ud[R], c[R], γR, ζR))
end

# The area-weighted flux of a state per unit area, G = (ρu, m u, (E + p)u)
# along d, without the pressure, aggregated.
@inline _radial_flux(q, u, p) = (q[1] * u, q[2] * u, q[3] * u, q[4] * u, (q[5] + p) * u)

# The node change per unit time of an area face value G and a pressure face
# value Π, as a cell with inv_J/W = sJ and inv_h/W = sh takes them.
@inline _radial_vector(G, Π, sJ, sh, d) =
    (sJ * G[1], sJ * G[2] + (d == 1) * sh * Π, sJ * G[3] + (d == 2) * sh * Π,
     sJ * G[4] + (d == 3) * sh * Π, sJ * G[5])

# The node terms' part of the stage increment at I, aggregated: B (A du + dt dQ)
# less the directions' face parts −B inv_J r_e (and −B inv_h r_p for the
# radial pressure of d), which leaves B times the node terms' history.
@inline function _limiter_node_increment(Q, dQ, du, regs, inv_J, inv_h, act, d, A, B,
                                         dt, lay, I)
    ns, m1, m2, m3, ie, nc = lay
    md = d == 1 ? m1 : d == 2 ? m2 : m3
    T = typeof(B)
    out = (zero(T), zero(T), zero(T), zero(T), zero(T))
    for cc in 1:nc
        x = iszero(A) ? dt * dQ[I, cc] : A * du[I, cc] + dt * dQ[I, cc]
        for e in 1:3
            act[e] || continue
            x += inv_J[I] * regs[e][cc][I]
        end
        cc == md && (x += inv_h[I] * regs[d][nc+1][I])
        x *= B
        slot = cc <= ns ? 1 : cc == m1 ? 2 : cc == m2 ? 3 : cc == m3 ? 4 : cc == ie ? 5 : 0
        slot > 0 && (out = Base.setindex(out, out[slot] + x, slot))
    end
    return out
end

# The largest s ≥ 0 for which q + s δ keeps ρ ≥ ε[1] and ρe ≥ ε[2]: the
# smaller of the density's root and the first positive root of
# 2ρ(s)(E(s) − ρ(s) ε[2]) − |m(s)|², a quadratic in s; Inf when neither
# bound is reached. The admissible set is convex, so [0, s] is admissible.
@inline function _limiter_reach(q, δ, ε)
    T = typeof(q[1])
    s = T(Inf)
    δ[1] < 0 && (s = min(s, (q[1] - ε[1]) / -δ[1]))
    a0 = 2 * q[1] * (q[5] - ε[2]) - (q[2] * q[2] + q[3] * q[3] + q[4] * q[4])
    a0 > 0 || return zero(T)
    a1 = 2 * (q[1] * δ[5] + δ[1] * (q[5] - ε[2])) -
         2 * (q[2] * δ[2] + q[3] * δ[3] + q[4] * δ[4])
    a2 = 2 * δ[1] * δ[5] - (δ[2] * δ[2] + δ[3] * δ[3] + δ[4] * δ[4])
    disc = a1 * a1 - 4 * a2 * a0
    if iszero(a2)
        a1 < 0 && (s = min(s, a0 / -a1))
    elseif disc >= 0
        # The two roots, formed without cancellation.
        w = -(a1 + copysign(sqrt(disc), a1)) / 2
        r1 = w / a2
        r2 = iszero(w) ? T(Inf) : a0 / w
        r1 > 0 && (s = min(s, r1))
        r2 > 0 && (s = min(s, r2))
    end
    return max(s, zero(T))
end

# The rate of the cell at node pp of a radial line d over the share its face
# and geometric parts keep, into `rates`: Σ R / (1 − β_n), with R the per-volume
# rates Ā a inv_J/W of its two faces (`AXIS_SHARE` of the outer face's for
# the face at the fold), γ u⁺ ΔĀ inv_J/W of its geometric part and
# 2(|u_e| + c)/W_e of every other active direction, and β_n the node part's
# share; Inf where the node part alone needs the whole cell.
@inline function _limiter_radial_rate_point!(rates, Q, dQ, du, regs, uvw, c, p, ρ,
                                             inv_J, inv_h, W, areas, act, d, ends,
                                             coeffs, bounds, lay, o1, o2, o3, pp, a, b)
    @inbounds begin
        T = eltype(rates)
        fold, closed_hi, n = ends
        A, B, dt = coeffs
        o = d == 1 ? o1 : d == 2 ? o2 : o3
        Im = _line_node(d, pp - 1, a, b, o1, o2, o3)
        I = _line_node(d, pp, a, b, o1, o2, o3)
        Ip = _line_node(d, pp + 1, a, b, o1, o2, o3)
        ud = uvw[d]
        Āp = areas[pp+o]
        Ām = areas[pp-1+o]
        ap = closed_hi && pp == n ?
             _radial_face_speed(ud, c, p, ρ, inv_J, inv_h, Āp, I, I) :
             _radial_face_speed(ud, c, p, ρ, inv_J, inv_h, Āp, I, Ip)
        Rp = Āp * ap
        Rm = fold && pp == 1 ? T(AXIS_SHARE) * Rp :
             Ām * _radial_face_speed(ud, c, p, ρ, inv_J, inv_h, Ām, Im, I)
        γ = _limiter_gamma(ρ[I], c[I], p[I])
        R = inv_J[I] / W[d][pp+o] * (Rp + Rm + γ * max(ud[I], zero(T)) * (Āp - Ām))
        for e in 1:3
            (e == d || !act[e]) && continue
            R += 2 * (abs(uvw[e][I]) + c[I]) / W[e][I[e]]
        end
        q = _limiter_state(Q, I, lay)
        δ = _limiter_node_increment(Q, dQ, du, regs, inv_J, inv_h, act, d, A, B, dt,
                                    lay, I)
        βn = one(T) / _limiter_reach(q, δ, _limiter_cell_bound(q, bounds))
        rates[I] = βn < 1 ? R / (1 - βn) : T(Inf)
    end
    return nothing
end

# θ at face f = f1 − 1 of a radial line d, and its unguaranteed sides, as
# `_limiter_theta_point!` takes them at a stage. A cell's part for the face
# is q ∓ (1/β)[H − τ (Ā G(q), p) as a node change], 1/β its rate over the
# face's, and its first-order counterpart the Lax–Friedrichs half state; the
# cell at the fold takes the fold face with the axis share, whose first-order
# values are 0 and τ p_1, and the cell on a closed far end takes its boundary
# face, never limited, in one part with its interior face, as on a Cartesian
# line. A side is guaranteed when τ times its cell's rate is below one; a
# cell whose node part needs the whole cell takes the first-order flux.
@inline function _limiter_radial_theta_point!(theta, flags, Q, faces, rates, uvw, c, p,
                                              ρ, inv_J, inv_h, W, areas, free, d, ends,
                                              τ, bounds, lay, o1, o2, o3, f1, a, b)
    @inbounds begin
        T = eltype(theta)
        fold, closed_lo, closed_hi, n = ends
        f = f1 - 1
        Il = _line_node(d, f, a, b, o1, o2, o3)
        Ir = _line_node(d, f + 1, a, b, o1, o2, o3)
        flag = 0
        θ = one(T)
        if !((closed_lo && f == 0) || (closed_hi && f == n))
            o = d == 1 ? o1 : d == 2 ? o2 : o3
            ud = uvw[d]
            fold_face = fold && f == 0
            wall_r = closed_hi && f == n - 1
            Ā = areas[f+o]
            np = lay[6] + 1
            G = _limiter_face_state(faces, Il, lay)
            Π = faces[np][Il]
            cl = !fold_face && _limiter_constrained(free, Il)
            cr = _limiter_constrained(free, Ir)
            a_face = fold_face ? zero(T) :
                     _radial_face_speed(ud, c, p, ρ, inv_J, inv_h, Ā, Il, Ir)
            ql = _limiter_state(Q, Il, lay)
            qr = _limiter_state(Q, Ir, lay)
            εl = _limiter_cell_bound(ql, bounds)
            εr = _limiter_cell_bound(qr, bounds)
            sJl = inv_J[Il] / W[f+o]
            shl = inv_h[Il] / W[f+o]
            sJr = inv_J[Ir] / W[f+1+o]
            shr = inv_h[Ir] / W[f+1+o]
            # The low cell's part for its outer face.
            kl = rates[Il] / (sJl * Ā * a_face)
            bl = _limiter_axpy(ql, kl * τ,
                               _radial_vector(map(x -> Ā * x,
                                                  _radial_flux(ql, ud[Il], p[Il])),
                                              p[Il], sJl, shl, d))
            Hl = _radial_vector(G, Π, sJl, shl, d)
            # The high cell's part for its inner face.
            if fold_face
                Ā1 = areas[1+o]
                I2 = _line_node(d, 2, a, b, o1, o2, o3)
                kr = rates[Ir] /
                     (sJr * T(AXIS_SHARE) * Ā1 *
                      _radial_face_speed(ud, c, p, ρ, inv_J, inv_h, Ā1, Ir, I2))
                br = _limiter_axpy(qr, -kr * τ, _radial_vector(
                                   (zero(T), zero(T), zero(T), zero(T), zero(T)), p[Ir],
                                   sJr, shr, d))
            elseif wall_r
                Āw = areas[f+1+o]
                aw = _radial_face_speed(ud, c, p, ρ, inv_J, inv_h, Āw, Ir, Ir)
                γ = _limiter_gamma(ρ[Ir], c[Ir], p[Ir])
                kr = rates[Ir] / (sJr * (Ā * a_face + Āw * aw +
                                         γ * max(ud[Ir], zero(T)) * (Āw - Ā)))
                br = _limiter_axpy(qr, -kr, _radial_vector(
                                   _limiter_face_state(faces, Ir, lay), faces[np][Ir],
                                   sJr, shr, d))
            else
                kr = rates[Ir] / (sJr * Ā * a_face)
                br = _limiter_axpy(qr, -kr * τ,
                                   _radial_vector(map(x -> Ā * x,
                                                      _radial_flux(qr, ud[Ir], p[Ir])),
                                                  p[Ir], sJr, shr, d))
            end
            Hr = _radial_vector(G, Π, sJr, shr, d)
            okl = isfinite(kl)
            okr = isfinite(kr)
            need = (cl && !(okl && _limiter_admissible(_limiter_axpy(bl, -kl, Hl), εl))) ||
                   (cr && !(okr && _limiter_admissible(_limiter_axpy(br, kr, Hr), εr)))
            if need
                # First order: τ Ā times the Rusanov flux, and τ {p}; at the
                # fold 0 and τ p_1.
                if fold_face
                    GL = (zero(T), zero(T), zero(T), zero(T), zero(T))
                    ΠL = τ * p[Ir]
                else
                    Fl = _radial_flux(ql, ud[Il], p[Il])
                    Fr = _radial_flux(qr, ud[Ir], p[Ir])
                    GL = map((x, y, u, v) -> τ * Ā * ((x + y) / 2 - a_face * (v - u) / 2),
                             Fl, Fr, ql, qr)
                    ΠL = τ * (p[Il] + p[Ir]) / 2
                end
                if cl
                    t, ok = okl ? _limiter_side(bl, Hl, _radial_vector(GL, ΠL, sJl, shl, d),
                                                -kl, εl) : (zero(T), false)
                    θ = min(θ, t)
                    (ok && τ * rates[Il] < 1) || (flag |= 1)
                end
                if cr
                    t, ok = okr ? _limiter_side(br, Hr, _radial_vector(GL, ΠL, sJr, shr, d),
                                                kr, εr) : (zero(T), false)
                    θ = min(θ, t)
                    (ok && τ * rates[Ir] < 1) || (flag |= 2)
                end
            end
        end
        theta[Il] = θ
        flags[Il] = flag
    end
    return nothing
end

# The corrections at node pp of a radial line from its two faces, where either
# is limited: δ = (θ − 1)(Φ − Φ_L) per face for each conserved component and
# for the pressure, the node changing by inv_J (δ_{p−½} − δ_{p+½})/W_p and,
# on the radial momentum, by inv_h times the pressure's. Each enters dQ / (B dt)
# and leaves its register, and a limited fold face enters the face register at
# the fold, whose running sum the next stage starts from.
@inline function _limiter_radial_correct_point!(dQ, regs, faces, theta, Q, uvw, c, p, ρ,
                                                inv_J, inv_h, W, areas, anchor, d, τ,
                                                scales, lay, fold, o1, o2, o3, pp, a, b)
    @inbounds begin
        T = eltype(theta)
        Im = _line_node(d, pp - 1, a, b, o1, o2, o3)
        I = _line_node(d, pp, a, b, o1, o2, o3)
        Ip = _line_node(d, pp + 1, a, b, o1, o2, o3)
        θm = theta[Im]
        θp = theta[I]
        (θm < 1 || θp < 1) || return nothing
        o = d == 1 ? o1 : d == 2 ? o2 : o3
        ns, m1, m2, m3, ie, nc = lay
        md = d == 1 ? m1 : d == 2 ? m2 : m3
        inv_B, inv_Bdt = scales
        fold_face = fold && pp == 1
        ud = uvw[d]
        Ām = areas[pp-1+o]
        Āp = areas[pp+o]
        am = θm < 1 && !fold_face ?
             _radial_face_speed(ud, c, p, ρ, inv_J, inv_h, Ām, Im, I) : zero(T)
        ap = θp < 1 ? _radial_face_speed(ud, c, p, ρ, inv_J, inv_h, Āp, I, Ip) : zero(T)
        inv_W = one(T) / W[pp+o]
        for cc in 0:nc
            # cc = 0 is the pressure, kept in register nc + 1.
            slot = cc == 0 ? nc + 1 : cc
            change = zero(T)
            if θm < 1
                gl = cc == 0 ? τ * (fold_face ? p[I] : (p[Im] + p[I]) / 2) :
                     fold_face ? zero(T) :
                     τ * Ām * _radial_lf_component(Q, ud, p, cc, Im, I, am, ie)
                δ = (θm - 1) * (faces[slot][Im] - gl)
                change += δ
                fold_face && (anchor[slot, a, b] += δ * inv_B)
            end
            if θp < 1
                gl = cc == 0 ? τ * (p[I] + p[Ip]) / 2 :
                     τ * Āp * _radial_lf_component(Q, ud, p, cc, I, Ip, ap, ie)
                change -= (θp - 1) * (faces[slot][I] - gl)
            end
            change *= inv_W
            if cc == 0
                dQ[I, md] += inv_h[I] * change * inv_Bdt
            else
                dQ[I, cc] += inv_J[I] * change * inv_Bdt
            end
            regs[slot][I] -= change * inv_B
        end
    end
    return nothing
end

# Component `cc` of the Rusanov flux per unit area between L and R at speed a,
# the area-weighted flux without the pressure.
@inline function _radial_lf_component(Q, ud, p, cc, L, R, a, ie)
    uL, uR = ud[L], ud[R]
    qL, qR = Q[L, cc], Q[R, cc]
    fL = qL * uL
    fR = qR * uR
    if cc == ie
        fL += p[L] * uL
        fR += p[R] * uR
    end
    return (fL + fR) / 2 - a * (qR - qL) / 2
end

# --- The limited filter pass ---------------------------------------------

# `filter_state!` under the weighting `:none` with the filter limiter after
# each directional pass; with no face limited the same passes and the same
# arithmetic. A fold's face takes its limit from node 1 alone: the filter's
# face flux there vanishes for an even field and not for an odd one.
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
            filt_along!(solver.tmp_a, comps[c], solver, d, cons_parity(solver, d, c))
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
                   _limiter_layout(solver),
                   (_limiter_closed(lim, decomp, d)..., _limiter_fold(lim, decomp, d)), n,
                   lim.radial[d], lim.volume, solver.inv_J, false, lim.anchor[d],
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
# holding the global low face. At a fold the mirrored face relation at face 0,
# Σ_{|s|≤q} l_s F̂_s = Σ_m c_m Σ_{l=1−m}^{m} f_l with F̂_{−s} = σf F̂_s and the
# halo holding the mirror f_{1−l} = σf f_l, gives zero for an odd flux and,
# for an even one, (Σ_m c_m Σ_l f_l − 2 Σ_s l_s S_s) / (1 + 2 Σ_s l_s).
@inline function _limiter_anchor_point!(anchor, F, div, W, A, dt, c, d, periodic,
                                        fold, σf, j, lhs, rhs, o1, o2, o3, a, b, _)
    @inbounds begin
        o = d == 1 ? o1 : d == 2 ? o2 : o3
        if fold
            T = eltype(anchor)
            value = zero(T)
            if σf > 0
                S = zero(T)
                acc = zero(T)
                sum_l = one(T)
                for s in eachindex(lhs)
                    S += W[s+o] * div[_line_node(d, s, a, b, o1, o2, o3)]
                    acc += 2 * lhs[s] * S
                    sum_l += 2 * lhs[s]
                end
                G = zero(T)
                for m in eachindex(rhs)
                    for l in 1-m:m
                        G += rhs[m] * F[_line_node(d, l, a, b, o1, o2, o3)]
                    end
                end
                value = (G - acc) / sum_l
            end
        elseif periodic
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
# (Σ_s l_s S_{j+s} + w ψ_j) / Σ_s l_s. Where `volume` is set, a component
# marked in `mask` sums −ω J Δ instead.
@inline function _limiter_scan_point!(faces, regs, Q, base, W, hd, totals, aux,
                                      n_cons, n, d, mode, j, lhs, psi, wf, volume, mask,
                                      inv_J, o1, o2, o3, a, b, _)
    @inbounds begin
        o = d == 1 ? o1 : d == 2 ? o2 : o3
        T = eltype(totals)
        for c in 1:n_cons
            fc = faces[c]
            S = zero(T)
            fc[_line_node(d, 0, a, b, o1, o2, o3)] = S
            weighted = volume && mask[c]
            for p in 1:n
                I = _line_node(d, p, a, b, o1, o2, o3)
                if mode == 1
                    S += W[p+o] * regs[c][I]
                elseif weighted
                    S -= (W[p+o] / hd) / inv_J[I] * (Q[I, c] - base[c][I])
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
# rank holds. A component the scan weighted by J starts from zero at the fold
# and takes no constant.
@inline function _limiter_offset_point!(faces, offset, wrap, cstar, scale, wrap_high,
                                        n, d, n_cons, volume, mask, o1, o2, o3, f1, a, b)
    @inbounds begin
        f = f1 - 1
        I = _line_node(d, f, a, b, o1, o2, o3)
        for c in 1:n_cons
            v = wrap_high && f == n ? wrap[c, a, b] : offset[c, a, b] + faces[c][I]
            faces[c][I] = volume && mask[c] ? scale * v : scale * (v - cstar[c, a, b])
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
# closed end is not limited; a fold's face is, from node 1's side alone, its
# other cell node 1's mirror. Mode 1 is a stage (half states scaled by
# 2Λ/(|u_d| + c) and the first-order flux τ F_LF); mode 2 a filter pass (2/ω,
# toward zero). Each cell keeps its own bound (`_limiter_cell_bound`). Under
# `volume` a cell takes the components marked in `vmask` divided by its J: at a
# stage along an area-weighted direction (`areal`) every component, whose
# face values are A_d times a flux on a line of constant A_d = J, and in a
# filter pass on a radial grid the components weighted by J. An areal stage
# reads each cell's rate from `rates`, which the radial pass fills.
@inline function _limiter_theta_point!(theta, flags, Q, faces, uvw, c, p, inv_W, free,
                                       act, d, n, closed, hd, τ, mode, bounds, lay,
                                       volume, vmask, inv_J, areal, rates,
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
        fold_face = closed[3] && f == 0
        ud = uvw[d]
        sl = zero(T); sr = zero(T)
        if mode == 1
            spl = abs(ud[Il]) + c[Il]
            spr = abs(ud[Ir]) + c[Ir]
            if areal
                !lo_bnd && !fold_face && (sl = spl > 0 ? -rates[Il] / spl : T(-Inf))
                !hi_bnd && (sr = spr > 0 ? rates[Ir] / spr : T(Inf))
            else
                !lo_bnd && (sl = spl > 0 ?
                                 -2 * _limiter_rate(uvw, c, inv_W, act, Il) / spl :
                                 T(-Inf))
                !hi_bnd && (sr = spr > 0 ?
                                 2 * _limiter_rate(uvw, c, inv_W, act, Ir) / spr :
                                 T(Inf))
            end
        else
            !lo_bnd && (sl = -2 * hd * inv_W[d][Il[d]])
            !hi_bnd && (sr = 2 * hd * inv_W[d][Ir[d]])
        end
        if !(lo_bnd || hi_bnd)
            # The filter's row at a closed end is the identity, and its
            # correction leaves the end node alone; so does the limiter.
            wall_l = closed[1] && f == 1
            wall_r = closed[2] && f == n - 1
            cl = _limiter_constrained(free, Il) && !(mode == 2 && wall_l) && !fold_face
            cr = _limiter_constrained(free, Ir) && !(mode == 2 && wall_r)
            qr = _limiter_state(Q, Ir, lay)
            # The state across a fold is node 1's mirror, its normal momentum
            # reversed.
            ql = fold_face ? _limiter_mirror(qr, d) : _limiter_state(Q, Il, lay)
            uL = fold_face ? -ud[Ir] : ud[Il]
            cL = fold_face ? c[Ir] : c[Il]
            pL = fold_face ? p[Ir] : p[Il]
            εl = _limiter_cell_bound(ql, bounds)
            εr = _limiter_cell_bound(qr, bounds)
            Gl = volume ? _limiter_volume(G, vmask, inv_J[Il]) : G
            Gr = volume ? _limiter_volume(G, vmask, inv_J[Ir]) : G
            # At a stage, a cell on a closed end keeps its boundary face, which
            # is never limited, in the same part as its interior face: one part
            # of weight α_d, q + (G_boundary − G_interior)/(α_d W), in place of
            # two halves, so the wall's flux enters at its own size, not twice.
            bl = ql; br = qr
            if mode == 1 && wall_l
                sl /= 2
                Gb = _limiter_face_state(faces, _line_node(d, 0, a, b, o1, o2, o3), lay)
                bl = _limiter_axpy(ql, -sl,
                                   volume ? _limiter_volume(Gb, vmask, inv_J[Il]) : Gb)
            end
            if mode == 1 && wall_r
                sr /= 2
                Gb = _limiter_face_state(faces, _line_node(d, n, a, b, o1, o2, o3), lay)
                br = _limiter_axpy(qr, -sr,
                                   volume ? _limiter_volume(Gb, vmask, inv_J[Ir]) : Gb)
            end
            need = (cl && !_limiter_admissible(_limiter_axpy(bl, sl, Gl), εl)) ||
                   (cr && !_limiter_admissible(_limiter_axpy(br, sr, Gr), εr))
            if need
                GL = (zero(T), zero(T), zero(T), zero(T), zero(T))
                a_face = zero(T)
                if mode == 1
                    a_face = max(abs(uL) + cL, abs(ud[Ir]) + c[Ir])
                    GL = _limiter_lf_state(ql, qr, uL, ud[Ir], pL, p[Ir], a_face, τ, d)
                end
                # A side whose first-order half state is not admissible, or
                # whose first-order bound τ a |s| ≤ 1 fails, is unguaranteed.
                if cl
                    t, ok = _limiter_side(bl, Gl, GL, sl, εl)
                    θ = min(θ, t)
                    (ok && τ * a_face * abs(sl) <= 1) || (flag |= 1)
                end
                if cr
                    t, ok = _limiter_side(br, Gr, GL, sr, εr)
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

# A state reflected across a plane normal to d.
@inline _limiter_mirror(q, d) =
    (q[1], d == 1 ? -q[2] : q[2], d == 2 ? -q[3] : q[3], d == 3 ? -q[4] : q[4], q[5])

# The aggregated face values a cell divides by its J where `vmask` is set.
@inline _limiter_volume(G, vmask, inv_J) = map((g, m) -> m ? g * inv_J : g, G, vmask)

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
# change / B, and a limited fold face enters the face register `anchor`; mode 2
# adds it to the state, with ω = W/h in place of W, and under `volume` with ω J
# for the components marked in `mask`. Along an `areal` direction the
# first-order face value is J G_L and the node change inv_J times the face form's.
@inline function _limiter_correct_point!(target, regs, faces, theta, Q, uvw, c, pr,
                                         W, d, hd, τ, mode, scales, lay, closed, n,
                                         volume, mask, inv_J, areal, anchor,
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
        fold_face = closed[3] && pp == 1
        for cc in 1:n_cons
            change = zero(T)
            if θm < 1
                gl = mode == 2 ? zero(T) :
                     fold_face ? _limiter_lf_mirror(Q, uvw, c, pr, cc, I, τ, d, lay) :
                     _limiter_lf_component(Q, uvw, c, pr, cc, Im, I, τ, d, lay)
                areal && (gl /= inv_J[I])
                δ = (θm - 1) * (faces[cc][Im] - gl)
                change += δ
                mode == 1 && fold_face && (anchor[cc, a, b] += δ * inv_B)
            end
            if θp < 1
                gl = mode == 1 ? _limiter_lf_component(Q, uvw, c, pr, cc, I, Ip, τ, d,
                                                       lay) : zero(T)
                areal && (gl /= inv_J[I])
                change -= (θp - 1) * (faces[cc][I] - gl)
            end
            change /= Wp
            if mode == 1
                target[I, cc] += (areal ? inv_J[I] * change : change) * inv_Bdt
                regs[cc][I] -= change * inv_B
            else
                volume && mask[cc] && (change *= inv_J[I])
                target[I, cc] += change
            end
        end
    end
    return nothing
end

# τ times component `cc` of the Lax–Friedrichs flux along d between node I's
# mirror across a fold and node I.
@inline function _limiter_lf_mirror(Q, uvw, c, p, cc, I, τ, d, lay)
    ns, m1, m2, m3, ie, _ = lay
    md = d == 1 ? m1 : d == 2 ? m2 : m3
    u = uvw[d][I]
    a = abs(u) + c[I]
    q = Q[I, cc]
    qL = cc == md ? -q : q
    fR = q * u
    fL = -qL * u
    if cc == ie
        fL -= p[I] * u
        fR += p[I] * u
    elseif cc == md
        fL += p[I]
        fR += p[I]
    end
    return τ * ((fL + fR) / 2 - a * (q - qL) / 2)
end
