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
# itself. The radial line of a refined box clear of r = 0 closes at its
# coarse-fine faces: its dual areas start at its first node's radius, and its
# filter passes keep W and the face relation's constant for every component.
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
# Same-level patches. Under `interface_flux = :closure` a patch's line closes
# the divergence at an interface with the interface rows, so the line is a
# closed one whose end weights are those rows' (0.369 h for the `:cascade3`
# rows the default C6 takes there, against 0.215 h at a `:neutral3` wall), and
# the shared node is the end node of both patches. Its end face, which carries
# the point flux of the node, is limited as well, toward the point flux of the
# node's state, by one θ on both patches (`_exchange_interface_theta!`). Under
# `:ghost` the inviscid flux and the rest close with different rows, and the
# setup rejects the limiter.
#
# Refined levels. A refined patch closes its lines at a coarse-fine face with
# the same interface rows, so its weights there are a same-level patch's. The
# end node at that face is not the patch's own: the parent's shell overwrites
# it after each stage (before each stage under subcycling). The limiter leaves
# it alone as it does a Dirichlet node: the face beside it is limited from the
# interior cell alone and the end face not at all, and composite conservation
# there is the coarse-fine coupling's own. Tiles share their faces as
# same-level patches do, through the level's records and communicator. The
# parent's nodes a child covers are limited as any other node and then take
# the child's coincident values from the injected restriction. Under
# subcycling every level limits the stages of its own steps with its own
# registers, and a refined level's filter passes, which it takes at its own
# cadence, are limited there.
#
# Per-point work runs through `pointwise!` bodies over three index boxes: the
# lines of a direction (the running sums and the θ passes, one sequential sweep
# per line), its faces, and its nodes. The running sum along a decomposed line
# takes one `Allgather` of the line totals over the direction's
# sub-communicator per direction per stage and per filter pass, entered by
# every rank of it; on a line a rank holds whole, the sweep of the running sum
# finishes the face values.
#
# On device storage the bodies launch as kernels over the same boxes, and
# every array they read lives in the fields' storage: the weights, the face
# areas, the flags of the overwritten positions and the line planes are
# uploaded or allocated there at construction. Four steps stay on the host,
# each with an explicit copy: the tallies of a θ pass, reduced on the device
# to four sums that cross to the host (one copy per pass); the offsets of a
# decomposed line, whose planes cross to the host for the `Allgather` and
# back (one copy each way per direction, stage and filter pass); the θ
# exchange at a same-level interface, whose two planes per record cross and
# return; and the minima that set ε, from a host copy of the state once per
# `run!`. A device level's tiles share stacked storage, and the limiter, whose
# registers are per patch, is not carried on a tiled device level.

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
mutable struct PositivityLimiter{T,A<:AbstractArray{T,3},V<:AbstractVector{T},
                                 B<:AbstractVector{Bool},P<:AbstractArray{T,3},
                                 C<:AbstractArray{Int,3}}
    weights::NTuple{3,V}            # W_d at each padded position along d
    inv_weights::NTuple{3,V}        # 1 / W_d, which the face pass multiplies by
    free::Tuple{B,B,B,A}            # positions on a face whose condition
                                    # overwrites the state (DirichletBC, a
                                    # coarse-fine face), and the patch's
                                    # `overwritten` nodes
    registers::Matrix{A}            # r[d, c]; c = n_cons + 1 the radial pressure
    register_fields::Vector{FieldVector{A,Vector{A}}}   # r[d, :] per d
    faces::FieldVector{A,Vector{A}} # face values, in the workspace's flux[1, :],
                                    # and the pressure's in flux[3, 1]
    base::FieldVector{A,Vector{A}}  # the filter's pre-pass state, flux[2, :]
    rates::A                        # a radial cell's rate, flux[2, 1] at a stage
    speeds::A                       # a radial face's speed, flux[2, 2] at a stage
    theta::A                        # θ per face; the workspace's tmp_b on one patch
    areas::NTuple{3,V}              # Ā at the face after each padded position
    radial::NTuple{3,Bool}          # d is a radial line folded at r = 0
    fold_lo::NTuple{3,Bool}         # the global low end of d is a fold
    n_comp::NTuple{3,Int}           # registers of d: n_cons, and the pressure
    volume::B                       # per component (and a trailing false): a
                                    # filter pass along a radial line weights
                                    # it by J
    volume_state::NTuple{5,Bool}    # the same for ρ, the momenta and E
    areal::NTuple{3,Bool}           # d takes area-weighted fluxes, D(A_d F), on
                                    # lines of constant A_d (z on the r-z plane)
    interface::NTuple{3,NTuple{2,Bool}} # the ends of d at a patch interface
    anchor::Vector{P}               # per d: (n_cons, n_a, n_b) Φ at the low face
    anchor_hi::Vector{P}            # Φ at the high face, at a patch interface
    totals::Vector{P}               # line totals of the local running sums
    aux::Vector{P}                  # anchor (stage) or measurement (filter)
    offset::Vector{P}               # this rank's running-sum offset
    wrap::Vector{P}                 # the value at the global low face
    cstar::Vector{P}                # the constant removed from a filter flux
    zero_plane::Vector{P}
    line_counts::Vector{C}          # per d: (4, n_a, n_b) each line's tallies
                                    # of a θ pass, summed into `counts`
    # Host copies of `offset`, `wrap` and `cstar` per d, which `_line_offsets!`
    # forms and uploads on device storage; empty until a device path needs them.
    staged::Vector{NTuple{3,Array{T,3}}}
    send::Vector{Vector{T}}
    recv::Vector{Vector{T}}
    deriv_lhs::V                    # the derivative's interior face relation
    deriv_rhs::V
    filter_lhs::V                   # the filter's, and its explicit face stencil
    filter_psi::V
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

"""
    PatchLimiters

The positivity limiters of a solver with several patches, one per patch this
rank holds, in the order of `solver.patches`. On a refined solver they are
built as `run!` starts and rebuilt for the patches a regrid replaces
(`_follow_patches!`), since a patch's limiter holds the weights of its own
lines; the counts of a dropped patch's limiter are kept in `retired`.
`least` is the fewest parent nodes along a dimension a regridded box spans,
the shortest fine line between two coarse-fine faces that takes the face
form and the filter's face relation.
"""
mutable struct PatchLimiters
    limiters::Vector{Any}
    patches::Vector{Any}
    retired::Vector{Int}
    least::Int
end

PatchLimiters(limiters::Vector{Any}, patches::Vector{Any}, least::Int=4) =
    PatchLimiters(limiters, patches, zeros(Int, 6), least)
Base.getindex(pl::PatchLimiters, i::Int) = pl.limiters[i]

# The fewest parent nodes along a dimension `tagged_region` gives a box: four,
# or the limiter's least.
_limiter_least_extent(solver) = (pl = getfield(solver, :positivity);
                                 pl isa PatchLimiters ? pl.least : 4)

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
# `lo_closures` and `hi_closures` are an end's rows where the plan replaces the
# scheme's own, at a patch interface.
function _limiter_weights(scheme, N::Int, h, n_halo::Int, ::Type{T};
                          lo_closures=nothing, hi_closures=nothing) where {T}
    M = min(N, 128)
    D = _line_operator(scheme, M, h, n_halo, T; lo_closures, hi_closures)
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
# rows' weights (`hi_closures` where the far end closes with rows other than
# the scheme's own, at a coarse-fine face). The folded divergence is the
# interior scheme on the mirrored data, so these serve either parity of the
# folded field.
function _limiter_fold_weights(scheme, N::Int, h, n_halo::Int, ::Type{T};
                               hi_closures=nothing) where {T}
    W = _limiter_weights(scheme, 2N, h, n_halo, T; lo_closures=hi_closures,
                         hi_closures)[1][N+1:2N]
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
# built: patches of an unstretched Cartesian grid without folds, one or
# several along a slab layout whose interfaces close the divergence with their
# own rows (`interface_flux = :closure`), or a single radial grid folded at
# r = 0 (`_limiter_radial_line`), on the r-z plane with a symmetry plane
# allowed at the low end of z; refined levels on either, under the same
# interface rows (the constructor switches `:ghost` to them, with a warning)
# and the injected restriction, tiled on the Cartesian grid of the host
# backend only; without the implicit integrator, an ideal-gas mixture, and
# closed lines long enough for the face relation of the filter.
function _validate_positivity(bcs, metric, stretch, patch_grid, nlev, backend, eos,
                              implicit, equations, deriv, filt, filter_weighting,
                              n_global, n_halo, L_domain, ::Type{T};
                              interface_flux::Symbol=:closure,
                              interface_rhs::Symbol=:extended,
                              interface_divergence=nothing, tile::Int=0,
                              level_restriction::Symbol=:inject) where {T}
    fail(what) = throw(ArgumentError("positivity_limiter: $what"))
    radial = _limiter_radial_line(bcs, metric, n_global)
    metric isa CartesianMetric || radial ||
        fail("supports the CartesianMetric, the r-z plane of a CylindricalMetric " *
             "with θ collapsed and AxisBC at r = 0, and the radial line of a " *
             "SphericalMetric with θ and φ collapsed and OriginBC at r = 0")
    all(isnothing, stretch) || fail("supports an unstretched grid only")
    radial && filter_weighting !== :none &&
        fail("on a radial grid takes the filter weighting :none")
    if nlev > 1
        level_restriction === :inject ||
            fail("on refined levels takes level_restriction = :inject: the " *
                 "filtered restriction writes values the limiter does not bound " *
                 "onto the parent")
        radial && tile > 0 &&
            fail("on a radial grid supports a refined level as one box (tile = 0): " *
                 "a tile's radial line closes at a shared face, which the radial " *
                 "limiter does not take")
    end
    npatch = prod(patch_grid)
    # Under `:ghost` the inviscid flux of an interface dimension takes the
    # gradient's interface rows, whose composite weight at the shared node is
    # h, and the rest of the flux the divergence's one-sided rows, whose
    # composite weight there is twice the end weight: no one set of node
    # weights writes both as differences of face fluxes. The constructor has
    # switched such interfaces to `:closure` before this check.
    (npatch == 1 && nlev == 1) || interface_flux === :closure ||
        error("positivity limiter: interfaces under :ghost reached the check")
    # A device level's tiles share stacked storage, whose batched right-hand
    # side has no per-tile registers.
    backend isa DeviceBackend && nlev > 1 && tile > 0 &&
        fail("on a DeviceBackend supports a refined level as one box (tile = 0)")
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
    # Every closed line, a patch's line along the split dimension with the
    # interface rows at its interface ends.
    ds = npatch > 1 ? findfirst(>(1), patch_grid) : 0
    periodic = ntuple(d -> n_global[d] > 1 ? isperiodic(bcs[d][1]) : true, 3)
    regions = npatch > 1 ? patch_slabs(n_global, periodic, patch_grid) : BlockRegion[]
    rows = _limiter_interface_rows(deriv, interface_rhs, interface_divergence)
    for d in 1:3
        n_global[d] > 1 || continue
        if d == ds
            h = T(L_domain[d] / (periodic[d] ? n_global[d] : n_global[d] - 1))
            for (pid, r) in enumerate(regions)
                lo = pid > 1 || periodic[d] ? rows : nothing
                hi = pid < length(regions) || periodic[d] ? rows : nothing
                _check_limiter_line(fail, deriv, filt, d, r.extent[d], h, n_halo, T,
                                    lo, hi)
            end
        elseif !periodic[d]
            h = T(L_domain[d] / (n_global[d] - 1))
            _check_limiter_line(fail, deriv, filt, d, n_global[d], h, n_halo, T,
                                nothing, nothing)
        end
        # A tile's line on a refined level: 3 tile + 1 nodes between two
        # interface ends at the level's spacing. A box's extent changes at a
        # regrid; its patch is checked as its limiter is built.
        if nlev > 1 && tile > 0
            hf = T(L_domain[d] / (periodic[d] ? n_global[d] : n_global[d] - 1) / 3)
            _check_limiter_line(fail, deriv, filt, d, 3tile + 1, hf, n_halo, T,
                                rows, rows)
        end
    end
    return nothing
end

# The fewest parent nodes a regridded box spans along a dimension under the
# limiter: its fine line of 3(n − 1) + 1 nodes, between two coarse-fine faces,
# takes the face form and the filter's face relation (`_check_limiter_line`).
function _limiter_least_box(deriv, filt, interface_rhs, interface_divergence,
                            n_halo::Int, ::Type{T}) where {T}
    rows = _limiter_interface_rows(deriv, interface_rhs, interface_divergence)
    N = _limiter_least_line(deriv, filt, n_halo, T, rows, rows)
    return N == typemax(Int) ? 4 : max(4, cld(N - 1, 3) + 1)
end

# The rows that close the divergence at a patch interface end, as the patched
# solver plans them: `nothing` where they are the scheme's own.
_limiter_interface_rows(deriv, interface_rhs::Symbol, interface_divergence) =
    interface_rhs === :extended || interface_divergence !== nothing ?
    interface_divergence_rows(deriv, interface_divergence) : nothing

# A closed line of N nodes whose ends close with `lo` and `hi` (`nothing` for
# the scheme's own rows): the face form exists, its weights are positive, and
# the line holds an interior face for the filter's face relation.
function _check_limiter_line(fail, deriv, filt, d, N, h, n_halo, ::Type{T}, lo,
                             hi) where {T}
    _limiter_line_ok(deriv, filt, N, n_halo, T, lo, hi) && return nothing
    least = _limiter_least_line(deriv, filt, n_halo, T, lo, hi)
    N < least < typemax(Int) &&
        fail("a line along dimension $d has $N nodes; the face relation of the " *
             "filter needs at least $least on a closed line")
    W, res = _limiter_weights(deriv, max(N, 16), h, n_halo, T; lo_closures=lo,
                              hi_closures=hi)
    res <= sqrt(eps(T)) * 1e-2 ||
        fail("the closure rows of $(deriv.name) do not take a face-flux form " *
             "(residual $res)")
    fail("the face weights of $(deriv.name) are not positive")
end

# Whether a closed line of N nodes whose ends close with `lo` and `hi` takes the
# face form with positive weights and holds a face for the filter's relation:
# beyond the margin of `_limiter_margin` on a long line, at its middle face
# (`_limiter_relation_face`) on a shorter one.
function _limiter_line_ok(deriv, filt, N::Int, n_halo::Int, ::Type{T}, lo,
                          hi) where {T}
    (min(N, 128) - 1) ÷ 2 - 2 >= 4 || return false
    W, res = _limiter_weights(deriv, N, one(T), n_halo, T; lo_closures=lo, hi_closures=hi)
    (res <= sqrt(eps(T)) * 1e-2 && all(>(0), W)) || return false
    least = 2 * _limiter_margin(W, one(T), filt, deriv, T; rows=_row_count(lo, hi)) + 4
    return N >= least || _limiter_relation_face(W, one(T), filt, N ÷ 2, T)
end

# The fewest nodes of a closed line `_limiter_line_ok` accepts; `typemax(Int)`
# where none up to 256 is.
function _limiter_least_line(deriv, filt, n_halo::Int, ::Type{T}, lo, hi) where {T}
    for N in 12:256
        _limiter_line_ok(deriv, filt, N, n_halo, T, lo, hi) && return N
    end
    return typemax(Int)
end

# Whether the filter's interior face relation holds at global face g of a
# closed line of weights W: the rows at the nodes it differences, g − q + 1 to
# g + q, are interior filter rows under unit ω = W/h, and the explicit face
# stencil reads nodes of the line. A line shorter than the margin above takes
# its relation where this holds; a longer one keeps the margin's face, where
# it holds too.
function _limiter_relation_face(W, h, filt, g::Int, ::Type{T}) where {T}
    N = length(W)
    q = length(_band_lhs(filt))
    M = max(length(filt.coeffs), q)
    nc = nclosure(filt)
    tol = sqrt(eps(T)) / 10
    (g + 1 - M >= 1 && g + M <= N) || return false
    for i in g-q+1:g+q
        (nc < i <= N - nc && abs(W[i] / h - 1) <= tol) || return false
    end
    return true
end

_row_count(lo, hi) = max(lo === nothing ? 0 : length(lo), hi === nothing ? 0 : length(hi))

# Nodes from a closed end beyond which both the derivative's weights and the
# filter's rows are interior, with the stencils' reach: the larger over the two
# ends, which differ where one closes a patch interface, whose divergence rows
# number `rows`.
function _limiter_margin(W, h, filt, deriv, ::Type{T}; rows::Int=0) where {T}
    half = length(W) ÷ 2
    tol = sqrt(eps(T)) / 10
    interior(k) = abs(W[k] / h - 1) <= tol
    tail_lo = something(findfirst(i -> all(interior(k) for k in i:half), 1:half), half)
    n = length(W)
    tail_hi = something(findfirst(i -> all(interior(n + 1 - k) for k in i:half), 1:half),
                        half)
    reach = max(length(filt.coeffs), length(_band_lhs(filt))) + length(_band_lhs(filt))
    return max(tail_lo, tail_hi, nclosure(filt), nclosure(deriv), rows) + reach + 1
end

# The limiter of one patch, after its geometry is filled: the weights at every
# padded position this rank holds, the registers, and the local faces where a
# periodic anchor and the filter relation are measured. A patch interface is a
# closed end of the patch's lines, whose weights come from the interface rows.
function PositivityLimiter(solver)
    decomp = solver.decomp
    T = eltype(solver.h)
    n_cons = solver.equations.n_cons
    plan_scheme(d) = (p = _plan_at(solver.div_plans, d);
                      (p isa DevicePlan ? p.host : p).scheme)
    schemes = solver.schemes
    deriv = schemes.deriv
    filt = schemes.filt
    irows = _limiter_interface_rows(deriv, schemes.interface_rhs,
                                    schemes.interface_divergence)
    # A same-level interface and a refined patch's coarse-fine face both close
    # the divergence with the interface rows.
    end_rows(d, side) = solver.bcs[d][side] isa InterfaceBC ? irows : nothing
    fold_lo = ntuple(d -> decomp.active[d] && solver.folds[d] !== nothing &&
                          solver.folds[d].lo, 3)
    curved = !(solver.metric isa CartesianMetric)
    # A line ending at an interface, a refined patch's included, whose extent
    # the setup does not see where a regrid gave it.
    fail(what) = throw(ArgumentError("positivity_limiter: $what"))
    for d in 1:3
        (decomp.active[d] && !decomp.periodic[d] && !fold_lo[d] &&
         any(bc -> bc isa InterfaceBC, solver.bcs[d])) || continue
        _check_limiter_line(fail, plan_scheme(d), filt, d, decomp.n_global[d],
                            solver.h[d], decomp.n_halo, T, end_rows(d, 1), end_rows(d, 2))
    end
    # The radial lines: the root's, folded at r = 0, and a refined patch's,
    # folded there where the patch reaches the axis or the origin and closed
    # at a coarse-fine face otherwise.
    radial = (curved && decomp.active[1], false, false)
    areal = ntuple(d -> decomp.active[d] && curved && !radial[d], 3)
    # The weights along each global line, and the nodes from a closed end
    # beyond which the filter's face relation is interior. A folded line's
    # far end closes with the derivative's rows, or the interface rows at a
    # coarse-fine face.
    global_weights = ntuple(3) do d
        decomp.active[d] || return Float64[1.0]
        N = decomp.n_global[d]
        decomp.periodic[d] && return fill(Float64(solver.h[d]), N)
        fold_lo[d] && return _limiter_fold_weights(deriv, N, solver.h[d],
                                                   decomp.n_halo, T;
                                                   hi_closures=end_rows(d, 2))
        return _limiter_weights(plan_scheme(d), N, solver.h[d], decomp.n_halo, T;
                                lo_closures=end_rows(d, 1), hi_closures=end_rows(d, 2))[1]
    end
    # The dual face areas of a radial line, Ā_k = A(r_0 + Σ_{j≤k} W_j) at face
    # k + ½, where A is the metric's radial area factor J/h_1 and r_0 the low
    # face's radius: 0 at the fold, the first node's at a closed low end.
    areas = ntuple(3) do d
        radial[d] || return T[]
        n = decomp.n_local[d]
        o = decomp.n_halo_d[d]
        N = decomp.n_global[d]
        position = [0.0; cumsum(global_weights[d])]
        fold_lo[d] ||
            (position .+= _phys_and_jac(solver, d, o + 1 - decomp.offset[d])[1])
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
    # The nodes a face condition overwrites after the update: a Dirichlet
    # plane, and the end plane of a refined patch at a coarse-fine face, which
    # the parent's shell imposes.
    overwrites(bc) = bc isa DirichletBC || parent_fed(bc)
    free = ntuple(3) do d
        decomp.active[d] || return [false]
        n = decomp.n_local[d]
        o = decomp.n_halo_d[d]
        N = decomp.n_global[d]
        [(decomp.offset[d] + p - o == 1 && overwrites(solver.bcs[d][1])) ||
         (decomp.offset[d] + p - o == N && overwrites(solver.bcs[d][2]))
         for p in 1:n+2o]
    end
    empty = similar(solver.tmp_a, T, 0, 0, 0)
    registers = [decomp.active[d] && c <= n_comp[d] ? zero(solver.tmp_a) : empty
                 for d in 1:3, c in 1:n_cons+1]
    register_fields = [FieldVector([registers[d, c] for c in 1:n_comp[d]])
                       for d in 1:3]
    # The line planes and the tallies live in the fields' storage, where the
    # bodies read and write them.
    like = solver.tmp_a
    planes = [_limiter_zeros(like, T, (n_comp[d], _transverse(decomp.n_local, d)...))
              for d in 1:3]
    copies() = [_limiter_zeros(like, T, size(p)) for p in planes]
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
        W = global_weights[d]
        margin = _limiter_margin(W, solver.h[d], filt, deriv, T;
                                 rows=_row_count(end_rows(d, 1), end_rows(d, 2)))
        N >= 2 * margin + 4 || return _limiter_relation_face(W, solver.h[d], filt,
                                                             decomp.offset[d] + j, T) ? j : 0
        margin <= decomp.offset[d] + j <= N - margin ? j : 0
    end
    anchor_face = ntuple(d -> decomp.active[d] ? clamp(decomp.n_local[d] ÷ 2, qd,
                                                       decomp.n_local[d] - qd) : 0, 3)
    ws = solver.flux
    # The ends at a same-level interface, whose node another patch holds too.
    # A coarse-fine end node is the parent's, not shared, and its face is
    # left alone as a Dirichlet node's is.
    shared(bc) = bc isa InterfaceBC && !parent_fed(bc)
    interface = ntuple(d -> decomp.active[d] ?
                            (shared(solver.bcs[d][1]), shared(solver.bcs[d][2])) :
                            (false, false), 3)
    # The patches of a rank share the workspace by extent, and a stage takes
    # every patch's θ before any patch's correction (`_limit_patched_stage!`),
    # so a patch holds its face values and θ itself.
    own = any(any, interface)
    faces = own ? [zero(solver.tmp_a) for c in 1:n_cons] : [ws[1, c] for c in 1:n_cons]
    theta = own ? zero(solver.tmp_a) : solver.tmp_b
    any(radial) && push!(faces, ws[3, 1])
    # On a radial line folded at r = 0 a limited filter pass weights the
    # components even at the fold by J, so that its corrections telescope in
    # Σ W J q, the sum the spherical filter conserves (its correction
    # annihilates the even polynomial r² on the mirrored line), and their
    # running sum starts from zero at the fold. An odd one, the radial
    # momentum, has no conservation law and keeps W and the constant of the
    # filter's face relation, as every component of a radial line without the
    # fold does.
    dr = findfirst(radial)
    volume = [dr !== nothing && fold_lo[dr] && c <= n_cons &&
              cons_parity(solver, dr, c) == 1 for c in 1:n_cons+1]
    lay = _limiter_layout(solver)
    volume_state = (volume[1], volume[lay[2]], volume[lay[3]], volume[lay[4]],
                    volume[lay[5]])
    # The bodies index every field by the linear index of a padded node.
    shape = size(solver.tmp_a)
    fields = (solver.tmp_b, solver.rho, solver.u, solver.v, solver.w, solver.p, solver.c,
              solver.inv_J, solver.inv_h..., faces..., ws[2, 1], ws[2, 2])
    all(f -> size(f) == shape, fields) ||
        error("positivity limiter: the solver's fields do not share one padded extent")
    # The vectors the bodies read, formed on the host and uploaded once.
    up(x) = _limiter_upload(like, x)
    weights_s = map(up, weights)
    free_s = map(up, free)
    line_counts = [_limiter_zeros(like, Int, (4, _transverse(decomp.n_local, d)...))
                   for d in 1:3]
    return PositivityLimiter{T,typeof(solver.tmp_a),typeof(weights_s[1]),
                             typeof(free_s[1]),typeof(planes[1]),eltype(line_counts)}(
        weights_s, map(w -> up(one(T) ./ w), weights), (free_s..., _limiter_exempt(solver)),
        registers, register_fields,
        FieldVector(faces), FieldVector([ws[2, c] for c in 1:n_cons]), ws[2, 1], ws[2, 2],
        theta, map(up, areas), radial, fold_lo, n_comp, up(volume), volume_state, areal,
        interface,
        copies(), copies(), copies(), copies(), copies(), copies(), copies(), copies(),
        line_counts, NTuple{3,Array{T,3}}[], send, recv, up(T.(_band_lhs(deriv))),
        up(T.(deriv.coeffs)), up(T.(filter_lhs)), up(_filter_face_stencil(filt, T)),
        decomp.periodic, measure, anchor_face, zero(T), zero(T), false, zeros(Int, 6))
end

# A host array in the storage of `like`: the array itself on host storage, a
# device copy otherwise.
function _limiter_upload(like, x::Array)
    _cpu_storage(like) && return x
    y = similar(like, eltype(x), size(x))
    copyto!(y, x)
    return y
end

# A zero-filled array of `dims` in the storage of `like`.
_limiter_zeros(like, ::Type{S}, dims::Dims) where {S} = fill!(similar(like, S, dims), zero(S))

# The two transverse extents of a direction's lines, in index order.
_transverse(n, d) = d == 1 ? (n[2], n[3]) : d == 2 ? (n[1], n[3]) : (n[1], n[2])

# The padded node at position p along d on line (a, b), a and b the
# transverse interior indices in index order.
@inline _line_node(d, p, a, b, o1, o2, o3) =
    d == 1 ? CartesianIndex(p + o1, a + o2, b + o3) :
    d == 2 ? CartesianIndex(a + o1, p + o2, b + o3) :
             CartesianIndex(a + o1, b + o2, p + o3)

# The same node's linear index in an array of the padded extent, whose first
# two sizes are n1 and n2, and the linear stride along d. The bodies index the
# fields, which share that extent, this way: a Cartesian index costs each
# array its own stride arithmetic, measured at a third of a θ pass.
@inline function _line_linear(d, p, a, b, o1, o2, o3, n1, n2)
    I = _line_node(d, p, a, b, o1, o2, o3)
    return I[1] + (I[2] - 1) * n1 + (I[3] - 1) * n1 * n2
end
@inline _line_stride(d, n1, n2) = d == 1 ? 1 : d == 2 ? n1 : n1 * n2

# A conserved state or an increment of one, indexed by a node's linear index
# and a component; `_BaseState` reads the same way.
struct _LinearState{A}
    data::A
    stride::Int
end
_LinearState(Q::ConservedState) = (q = parent(Q); _LinearState(q, stride(q, 4)))
Base.@propagate_inbounds Base.getindex(s::_LinearState, l::Int, c::Int) =
    s.data[l+(c-1)*s.stride]
Base.@propagate_inbounds Base.setindex!(s::_LinearState, v, l::Int, c::Int) =
    setindex!(s.data, v, l + (c - 1) * s.stride)
Adapt.adapt_structure(to, s::_LinearState) = _LinearState(adapt(to, s.data), s.stride)

# --- The bounds of a run ---------------------------------------------------

# ε from the state entering `run!`, the same rule at a restart and after a
# phase change: `LIMITER_FRACTION` of the global minimum of ρ and of ρe over
# the interior points where each is positive. Collective over the solver's
# communicator. Returns whether the limiter acts in this run. A limited run
# can end with a point marginally below zero, a cell whose node terms alone
# took it there, and the next run starts from that state; the point's bound is
# zero (`_limiter_cell_bound`), so its faces take the first-order flux where
# its half states need it and it is counted unguaranteed, and the rest of the
# grid keeps the scale of its own state. Only a state without a positive
# density or internal energy anywhere gives no scale, and the run then
# proceeds unlimited with a warning, as the failsafe does.
_positivity_setup!(solver, Q) = false
function _positivity_setup!(solver, Q::Union{ConservedState,Vector{<:ConservedState}})
    lim = getfield(solver, :positivity)
    lim === nothing && return false
    return _positivity_bounds!(lim, solver, Q)
end

_positivity_bounds!(lim::PositivityLimiter, solver, Q) =
    _set_limiter_bounds!((lim,), solver, _limiter_minima(solver, Q))

# The patches' minima, over every patch this rank holds, reduced once.
function _positivity_bounds!(pl::PatchLimiters, solver, states)
    lims = _follow_patches!(solver, pl)
    ρmin = Inf
    emin = Inf
    for (ps, Q) in eachpatch(solver, states)
        r, e = _limiter_minima(ps, Q)
        ρmin = min(ρmin, r)
        emin = min(emin, e)
    end
    return _set_limiter_bounds!(lims, solver, (ρmin, emin))
end

# The limiters of the patches this rank holds now, built for a patch that has
# none (every patch of a refined solver as its first `run!` starts, and the
# patches a regrid replaced) with the run's bounds, and dropped with a patch a
# regrid dropped, its counts kept. Rank-local: a patch's limiter reads only
# the patch.
function _follow_patches!(solver, pl::PatchLimiters)
    patches = getfield(solver, :patches)
    if !(length(pl.patches) == length(patches) &&
         all(i -> pl.patches[i] === patches[i], eachindex(patches)))
        _rebuild_limiters!(solver, pl)
    end
    # The exempt nodes are written over the interior by a regrid; a θ pass
    # reads them on the rank halos as well, as the rank holding them does.
    for (lim, p) in zip(pl.limiters, patches)
        isempty(lim.free[4]) || exchange_halos!(lim.free[4], p.decomp)
    end
    return pl.limiters
end

function _rebuild_limiters!(solver, pl::PatchLimiters)
    patches = getfield(solver, :patches)
    held = IdDict{Any,Any}(zip(pl.patches, pl.limiters))
    template = isempty(pl.limiters) ? nothing : first(pl.limiters)
    limiters = Any[]
    for p in patches
        lim = pop!(held, p, nothing)
        if lim === nothing
            lim = PositivityLimiter(PatchSolver(solver, p))
            if template !== nothing
                lim.eps_rho = template.eps_rho
                lim.eps_e = template.eps_e
                lim.active = template.active
            end
        end
        push!(limiters, lim)
    end
    for lim in values(held)
        pl.retired .+= lim.counts
    end
    pl.limiters = limiters
    pl.patches = Any[p for p in patches]
    return pl
end

# This rank's minimum ρ over the interior points where it is positive, and of
# ρe where both are. A state in device storage is copied to the host for the
# sweep, once per `run!`.
_limiter_minima(solver, Q) =
    _device_path(Q) ?
    _limiter_minima_host(solver, ConservedState(Array(_dense_copy(parent(Q))))) :
    _limiter_minima_host(solver, Q)

function _limiter_minima_host(solver, Q)
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
        ρ > 0 || continue
        ρmin = min(ρmin, ρ)
        e = Q[I, ie] - (Q[I, m1]^2 + Q[I, m2]^2 + Q[I, m3]^2) / (2ρ)
        e > 0 && (emin = min(emin, e))
    end
    return ρmin, emin
end

# The run's ε on every limiter of `lims` from this rank's minima, reduced over
# the solver's communicator.
function _set_limiter_bounds!(lims, solver, minima)
    red = MPI.Allreduce([minima[1], minima[2]], min, solver.comm)
    active = isfinite(red[1]) && isfinite(red[2])
    for lim in lims
        T = eltype(lim.eps_rho)
        lim.active = active
        if active
            lim.eps_rho = T(LIMITER_FRACTION * red[1])
            lim.eps_e = T(LIMITER_FRACTION * red[2])
        end
    end
    if !active && MPI.Comm_rank(solver.comm) == 0
        @warn "run!: the positivity limiter is inactive in this run. No point " *
              "of the state entering it has a positive density and internal " *
              "energy, so there is no bound to scale."
    end
    return active
end

"""
    positivity_counts(solver) -> NamedTuple

What the positivity limiter of `solver` has done since it was built, summed
over the ranks and the patches: the interior faces tested at the Runge–Kutta
stages (a shared node's end face once) and at the filter passes, the faces
limited in each, the unguaranteed cell sides, where the first-order bound
failed or the first-order half state was not admissible, on a radial grid
also those of a cell whose source and boundary terms alone would leave the
bound, and on a radial grid the stage faces limited at r = 0 (`axis_limited`, counted in `stage_limited` too). Collective
over the solver's communicator.
"""
function positivity_counts(solver)
    lim = getfield(solver, :positivity)
    lim === nothing && throw(ArgumentError(
        "positivity_counts: the solver was built without positivity_limiter"))
    local_counts = lim isa PositivityLimiter ? lim.counts :
                   foldl((s, l) -> s .+ l.counts, lim.limiters; init=copy(lim.retired))
    c = MPI.Allreduce(local_counts, +, solver.comm)
    return (stage_faces=c[1], stage_limited=c[2], filter_faces=c[3],
            filter_limited=c[4], unguaranteed=c[5], axis_limited=c[6])
end

# --- The limited step ------------------------------------------------------

# `step!` with the stage limiter between each stage's right-hand side and its
# low-storage update; with no face limited it makes the calls of `step!` in
# the same order and the same arithmetic.
function _limited_run_step!(solver, Q, workspace, dt, prepared::Bool, control)
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

# The multi-patch `step!`, with the stage limiter of every patch
# (`_limit_level_stage!`) between the right-hand sides and the stage updates,
# level by level as `step!` takes them; a subcycled solver takes
# `subcycled_step!`'s schedule with the limiter (`_advance_level!`). A patch's
# lines end at a same-level interface as at a wall, with the weights of the
# interface rows, and `_exchange_interface_theta!` has what the two patches
# share there. A refined patch's lines end at a coarse-fine face with the same
# rows, and the end node there, which the parent's shell overwrites after the
# stage, is left alone as a Dirichlet node is. Returns a `SolverFailure` where
# the subcycled schedule rejects a substep, and `nothing` otherwise.
function _limited_run_step!(solver, states::Vector{<:ConservedState}, workspace, dt,
                            prepared::Bool, control)
    lims = _follow_patches!(solver, getfield(solver, :positivity))
    getfield(solver, :subcycle) &&
        return _limited_subcycled_step!(solver, states, workspace, dt, prepared, control,
                                        lims)
    patches = getfield(solver, :patches)
    levels = getfield(solver, :levels)
    dQs, dus = workspace.dQ, workspace.du
    for stage in 1:5
        solver.tstage = solver.t + oftype(solver.t, RKC[stage]) * dt
        first_prepared = prepared && stage == 1
        limited = _limiter_stage_rhs(lims, dQs, stage, dt)
        for lev in levels
            status = _level_rhs!(solver, lev, states, limited, first_prepared)
            _check_transport_status(solver, status)
        end
        for lev in levels
            _limit_level_stage!(lims, solver, lev, states, dQs, dus, stage, dt)
        end
        for lev in levels
            _ledger_open!(solver, states, lev)
            _level_update!(solver, lev, states, dQs, dus, RKA[stage], RKB[stage], dt)
            _ledger_update!(solver, states, dQs, RKA[stage], RKB[stage], dt, lev)
        end
        _ledger_open!(solver, states)
        sync_patches!(solver, states)
        _ledger!(solver, states, :same_level)
        if length(levels) > 1
            prolong_level_ghosts!(solver, states)
            _ledger!(solver, states, :shell)
        end
    end
    solver.tstage = solver.t + dt
    _ledger_open!(solver, states)
    for (i, p) in enumerate(patches)
        apply_bcs!(PatchSolver(solver, p), states[i])
    end
    _ledger!(solver, states, :wall_enforce)
    _validate_transport_state!(solver, states)
    return nothing
end

# `_subcycled_run_step!` with the stage limiter on every level's stages and the
# filter limiter on every refined level's passes (`_advance_level!`).
function _limited_subcycled_step!(solver, states, workspace, dt, prepared::Bool,
                                  control, lims)
    status, guard = _subcycled_step_status!(solver, states, workspace.dQ, workspace.du,
                                             dt, prepared, control, lims)
    status == 0 && return nothing
    status == SUBSTEP_CFL && return _substep_cfl_failure(solver, guard, dt, control)
    status == SUBSTEP_INVALID && return _substep_invalid_failure(solver, guard, dt)
    _check_transport_status(solver, status)
    return nothing
end

# The right-hand side arrays a limited stage hands `_level_rhs!`, one per
# patch (`_LimiterRHS`); the plain arrays where the step is not limited.
_limiter_stage_rhs(::Nothing, dQs, stage::Int, dt) = dQs
function _limiter_stage_rhs(lims::Vector{Any}, dQs, stage::Int, dt)
    T = eltype(first(lims).eps_rho)
    return [_LimiterRHS(dQs[i], lims[i], T(RKA[stage]), T(dt)) for i in eachindex(dQs)]
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
    pointwise!(_limiter_subtract_point!, dQ.dQ, nx, ny, nz,
               dQ.dQ, lim.registers[d, slot], solver.tmp_a, inv_J, c, dQ.A, dQ.dt,
               o1, o2, o3)
    if decomp.coords[d] == 0
        nA, nB = _transverse(decomp.n_local, d)
        pointwise!(_limiter_anchor_point!, solver.tmp_a, nA, nB, 1,
                   lim.anchor[d], f, solver.tmp_a, lim.weights[d], dQ.A, dQ.dt, slot, d,
                   lim.periodic[d], lim.fold_lo[d], σf, lim.anchor_face[d],
                   lim.deriv_lhs, lim.deriv_rhs, o1, o2, o3)
    end
    if lim.interface[d][2] && decomp.coords[d] == decomp.dims[d] - 1
        nA, nB = _transverse(decomp.n_local, d)
        pointwise!(_limiter_end_anchor_point!, solver.tmp_a, nA, nB, 1,
                   lim.anchor_hi[d], f, dQ.A, dQ.dt, slot, d, decomp.n_local[d],
                   o1, o2, o3)
    end
    return dQ
end

function _limit_stage!(lim::PositivityLimiter, solver, Q, dQ, du, stage::Int, dt)
    lim.active || return nothing
    for d in 1:3
        solver.decomp.active[d] || continue
        _stage_theta!(lim, solver, Q, dQ, du, stage, dt, d) &&
            _stage_correct!(lim, solver, Q, dQ, stage, dt, d)
    end
    return nothing
end

# The stage limiter of every patch this rank holds on a level, a direction at
# a time: each patch's θ, the exchange that gives the faces about a shared node
# the same θ on both patches, then each patch's corrections. A patch without a
# same-level interface shares its face values and θ with the other patches of
# its extent (the workspace) and takes its corrections at once. The level's
# shared planes are the root's records at the root and the level's records of
# the direction, in the level's communicator, below it.
_limit_level_stage!(::Nothing, solver, lev, states, dQs, dus, stage::Int, dt) = nothing
function _limit_level_stage!(lims::Vector{Any}, solver, lev, states, dQs, dus,
                             stage::Int, dt)
    first(lims).active || return nothing
    isempty(lev.patches) && return nothing
    patches = getfield(solver, :patches)
    root = lev.index == 0
    for d in 1:3
        PatchSolver(solver, patches[first(lev.patches)]).decomp.active[d] || continue
        due = falses(length(patches))
        for i in lev.patches
            ps = PatchSolver(solver, patches[i])
            lim = lims[i]
            due[i] = _stage_theta!(lim, ps, states[i], dQs[i], dus[i], stage, dt, d)
            if due[i] && !any(any, lim.interface)
                _stage_correct!(lim, ps, states[i], dQs[i], stage, dt, d)
                due[i] = false
            end
        end
        records = root ? solver.plane_pairs : lev.plane_pairs[d]
        _exchange_interface_theta!(solver, lims, d, due, records,
                                   root ? solver.comm : lev.level_comm.comm)
        for i in lev.patches
            due[i] && _stage_correct!(lims[i], PatchSolver(solver, patches[i]), states[i],
                                      dQs[i], stage, dt, d)
        end
    end
    return nothing
end

# The face values and θ of direction d at a stage, and whether a face is
# limited where this rank's nodes take its correction; a radial line takes its
# whole pass here.
function _stage_theta!(lim::PositivityLimiter, solver, Q, dQ, du, stage::Int, dt, d::Int)
    T = eltype(lim.eps_rho)
    B = T(RKB[stage])
    τ = T(_stage_advance(stage) * dt)
    _line_faces!(lim, solver, Q, d, 1, B, zero(T))
    decomp = solver.decomp
    if _limiter_ends(lim, decomp, d)[5]
        nA, nB = _transverse(decomp.n_local, d)
        pointwise!(_limiter_end_face_point!, solver.tmp_a, nA, nB, 1,
                   lim.faces, lim.anchor_hi[d], B, lim.n_comp[d], decomp.n_local[d], d,
                   decomp.n_halo_d...)
    end
    if lim.radial[d]
        _limit_radial!(lim, solver, Q, dQ, du, d, T(RKA[stage]), B, T(dt), τ)
        return false
    end
    limited, shared = _limit_faces!(lim, solver, Q, d, τ, 1)
    lim.counts[2] += limited
    return limited + shared > 0
end

function _stage_correct!(lim::PositivityLimiter, solver, Q, dQ, stage::Int, dt, d::Int)
    decomp = solver.decomp
    T = eltype(lim.eps_rho)
    B = T(RKB[stage])
    τ = T(_stage_advance(stage) * dt)
    o1, o2, o3 = decomp.n_halo_d
    inv_B = one(T) / B
    inv_Bdt = one(T) / (B * T(dt))
    n = decomp.n_local[d]
    nA, nB = _transverse(decomp.n_local, d)
    pointwise!(_limiter_correct_point!, solver.tmp_a, n, nA, nB,
               _LinearState(dQ), lim.register_fields[d], lim.faces, lim.theta,
               _LinearState(Q), (solver.u, solver.v, solver.w), solver.c, solver.p,
               lim.weights[d], d, one(T), τ, 1, (inv_B, inv_Bdt),
               _limiter_layout(solver), _limiter_ends(lim, decomp, d), n,
               false, lim.volume, solver.inv_J, lim.areal[d],
               (lim.anchor[d], lim.anchor_hi[d]), lim.free, o1, o2, o3)
    return nothing
end

# A shared interface node is the last node of one patch and the first of the
# other, and its two copies are averaged after every stage, so the node's
# update is the mean of the two patches' updates, each of the node's end
# weight. Each patch keeps its own update of the node within the node's bound,
# and their mean, the admissible set being convex, is within it as well. The
# update takes the face beside the node and the face at the node itself, the
# patch's end face, which carries the point flux there. The first-order flux of
# that face is the point flux of the node's state, the same on both sides, so
# where the two patches' fluxes there agree (the inviscid flux of the averaged
# state) limiting it on both sides by one θ conserves, and where they differ
# (the artificial fluxes, from sensors each patch smooths with its own closure
# rows) it moves the scheme's own mismatch toward zero. The node's limit is
# taken along the segment that blends both its faces by one θ, so it holds for
# any smaller θ on both, and the θ pass gives the end face its neighbour's θ.
# The exchange below gives both patches the smaller of their two, at the end
# face and at each patch's face beside the node, from the shared-plane records
# the averaging uses. Collective among the ranks holding the shared planes.
function _exchange_interface_theta!(solver, lims, d::Int, due, records, comm)
    isempty(records) && return nothing
    _device_path(lims[first(records).patch].theta) &&
        return _exchange_interface_theta_staged!(solver, lims, d, due, records, comm)
    patches = getfield(solver, :patches)
    me = MPI.Comm_rank(comm)
    normal(pl) = any(lims[pl.patch].interface[d])
    reqs = MPI.Request[]
    for pl in records
        normal(pl) && pl.partner != me || continue
        push!(reqs, MPI.Irecv!(pl.buf, comm; source=pl.partner, tag=pl.tag))
    end
    for pl in records
        normal(pl) && pl.partner != me || continue
        lim = lims[pl.patch]
        s = _end_shift(patches[pl.patch].decomp, pl.mine, d)
        idx = 1
        for I in CartesianIndices(pl.mine)
            pl.sbuf[idx] = lim.theta[_shift_index(I, d, s)]
            idx += 1
        end
        push!(reqs, MPI.Isend(pl.sbuf, comm; dest=pl.partner, tag=pl.sendtag))
    end
    MPI.Waitall(reqs)
    for pl in records
        normal(pl) || continue
        lim = lims[pl.patch]
        s = _end_shift(patches[pl.patch].decomp, pl.mine, d)
        local_pair = pl.partner == me
        other = local_pair ? lims[pl.partner_patch].theta : nothing
        so = local_pair ? _end_shift(patches[pl.partner_patch].decomp, pl.theirs, d) : 0
        theirs = local_pair ? CartesianIndices(pl.theirs) : nothing
        decomp = patches[pl.patch].decomp
        idx = 1
        for I in CartesianIndices(pl.mine)
            Ie = _shift_index(I, d, s)
            mine = lim.theta[Ie]
            partner = local_pair ? other[_shift_index(theirs[idx], d, so)] : pl.buf[idx]
            idx += 1
            # A level's records reach into the transverse halos, where no θ
            # pass runs.
            _interior_transverse(I, decomp, d) || continue
            m = min(mine, partner)
            m < mine || continue
            # The end face and the face beside the node: the end face is the
            # node's high face at a patch's high end (shift 0) and its low face
            # at a low end (shift −1).
            Ib = _shift_index(Ie, d, s == 0 ? -1 : 1)
            lim.theta[Ie] = m
            lim.theta[Ib] = m
            due[pl.patch] = true
            # Faces this patch counts that the exchange limits: at a high end
            # both, at a low end the face beside the node.
            mine == 1 && (lim.counts[2] += s == 0 ? 2 : 1)
        end
    end
    return nothing
end

# The exchange above on device storage. Each record's planes of θ cross to the
# host as `_pack_fields!` stages a block: the end faces for the message and
# the comparison, the faces beside the node, and on a local pairing the
# partner's end faces, read after the records before it have written them, as
# the host loop reads them. The minima are taken on the host in the same order
# and both planes are written back where a face changed.
function _exchange_interface_theta_staged!(solver, lims, d::Int, due, records, comm)
    patches = getfield(solver, :patches)
    me = MPI.Comm_rank(comm)
    normal(pl) = any(lims[pl.patch].interface[d])
    end_planes(pl) = _shift_ranges(pl.mine, d,
                                   _end_shift(patches[pl.patch].decomp, pl.mine, d))
    reqs = MPI.Request[]
    for pl in records
        normal(pl) && pl.partner != me || continue
        push!(reqs, MPI.Irecv!(pl.buf, comm; source=pl.partner, tag=pl.tag))
    end
    for pl in records
        normal(pl) && pl.partner != me || continue
        _pack_fields!(pl.sbuf, (lims[pl.patch].theta,), end_planes(pl))
        push!(reqs, MPI.Isend(pl.sbuf, comm; dest=pl.partner, tag=pl.sendtag))
    end
    MPI.Waitall(reqs)
    for pl in records
        normal(pl) || continue
        lim = lims[pl.patch]
        T = eltype(lim.theta)
        s = _end_shift(patches[pl.patch].decomp, pl.mine, d)
        re = _shift_ranges(pl.mine, d, s)
        rb = _shift_ranges(re, d, s == 0 ? -1 : 1)
        m = prod(length.(re))
        own = Vector{T}(undef, m)
        beside = Vector{T}(undef, m)
        _pack_fields!(own, (lim.theta,), re)
        _pack_fields!(beside, (lim.theta,), rb)
        partner = if pl.partner == me
            so = _end_shift(patches[pl.partner_patch].decomp, pl.theirs, d)
            v = Vector{T}(undef, m)
            _pack_fields!(v, (lims[pl.partner_patch].theta,), _shift_ranges(pl.theirs, d, so))
            v
        else
            pl.buf
        end
        decomp = patches[pl.patch].decomp
        changed = false
        for (idx, I) in enumerate(CartesianIndices(pl.mine))
            _interior_transverse(I, decomp, d) || continue
            mine = own[idx]
            v = min(mine, partner[idx])
            v < mine || continue
            own[idx] = v
            beside[idx] = v
            changed = true
            due[pl.patch] = true
            mine == 1 && (lim.counts[2] += s == 0 ? 2 : 1)
        end
        if changed
            _unpack_fields!((lim.theta,), own, re)
            _unpack_fields!((lim.theta,), beside, rb)
        end
    end
    return nothing
end

# Padded ranges `r` moved by `s` along d.
_shift_ranges(r::NTuple{3,UnitRange{Int}}, d::Int, s::Int) =
    ntuple(e -> e == d ? (first(r[e])+s:last(r[e])+s) : r[e], 3)

_interior_transverse(I, decomp, d) =
    all(e -> e == d || decomp.n_halo_d[e] < I[e] <= decomp.n_halo_d[e] + decomp.n_local[e],
        1:3)

# The shift from a shared plane node to the slot of the patch's end face: 0 at
# the patch's high end, where face f sits at node f, and −1 at its low end.
_end_shift(decomp, region, d) = first(region[d]) == decomp.n_halo_d[d] + 1 ? -1 : 0
_shift_index(I::CartesianIndex{3}, d, s) =
    CartesianIndex(ntuple(e -> e == d ? I[e] + s : I[e], 3))

_limiter_layout(solver) =
    (solver.equations.n_species, solver.equations.i_mom...,
     solver.equations.i_energy, solver.equations.n_cons)

# The face values of direction d into `lim.faces`: the local running sums of
# each line, the line totals gathered over the direction's sub-communicator,
# and the offsets that make them one running sum along the global line,
# anchored at the global low face. `mode` 1 is the stage register, scaled by
# `scale` = B_k and anchored at the face register; `mode` 2 a filter pass of
# weight `wf`, its constant removed through the filter's face relation. A
# patch interface anchors a line here as a wall does.
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
    # A line this rank holds whole takes its offsets from its own anchor or
    # measurement, and the scan finishes its face values.
    whole = decomp.dims[d] == 1
    whole && mode == 2 && lim.measure[d] == 0 && _no_measure_error(d)
    pointwise!(_limiter_scan_point!, solver.tmp_a, 1, nA, nB,
               lim.faces, lim.register_fields[d], _LinearState(Q), lim.base, lim.weights[d],
               solver.h[d], lim.totals[d], lim.aux[d], lim.anchor[d], n_comp, n, d, mode,
               mode == 2 ? lim.measure[d] : 0, lim.filter_lhs, lim.filter_psi, wf,
               volume, lim.volume, solver.inv_J, (whole, lim.periodic[d], scale),
               o1, o2, o3)
    whole && return nothing
    _line_offsets!(lim, decomp, d, mode)
    last_rank = decomp.coords[d] == decomp.dims[d] - 1
    pointwise!(_limiter_offset_point!, solver.tmp_a, n + 1, nA, nB,
               lim.faces, lim.offset[d], lim.wrap[d],
               mode == 1 ? lim.zero_plane[d] : lim.cstar[d], scale,
               lim.periodic[d] && last_rank, n, d, n_comp, volume, lim.volume,
               o1, o2, o3)
    return nothing
end

# The offsets of this rank's running sums along a decomposed line. One
# `Allgather` of the line totals (and of the low face's anchor or the filter
# relation's measurement) over the direction's sub-communicator, entered by
# every rank of it; the prefix sums are formed in rank order on every rank, so
# the face two ranks share carries the same value on both.
function _line_offsets!(lim::PositivityLimiter, decomp, d::Int, mode::Int)
    tot = lim.totals[d]
    # The stage's anchor is the face register at the global low face.
    aux = mode == 1 ? lim.anchor[d] : lim.aux[d]
    L = length(tot)
    send = lim.send[d]
    # Device storage crosses to the host message here, and the offsets formed
    # on the host are uploaded after: one copy of each plane per pass.
    copyto!(send, 1, vec(tot), 1, L)
    copyto!(send, L + 1, vec(aux), 1, L)
    _device_path(tot) ? _staged_offsets!(lim, decomp, d, mode) :
                        _form_offsets!(lim.offset[d], lim.wrap[d], lim.cstar[d], lim, decomp,
                                       d, mode)
    return nothing
end

# The offsets formed in the host copies of the planes and uploaded.
function _staged_offsets!(lim::PositivityLimiter, decomp, d::Int, mode::Int)
    planes = _staged_planes!(lim, d)
    _form_offsets!(planes..., lim, decomp, d, mode)
    copyto!(lim.offset[d], planes[1])
    copyto!(lim.wrap[d], planes[2])
    copyto!(lim.cstar[d], planes[3])
    return nothing
end

# The host copies of the offset planes of d, allocated at the first device pass.
function _staged_planes!(lim::PositivityLimiter{T}, d::Int) where {T}
    if isempty(lim.staged)
        for e in 1:3
            dims = size(lim.offset[e])
            push!(lim.staged, (zeros(T, dims), zeros(T, dims), zeros(T, dims)))
        end
    end
    return lim.staged[d]
end

# The Allgather of the message `send` holds, and the offsets of this rank's
# running sums, the global low face's value and the filter's constant formed
# from it into `off`, `wrap` and `cstar`, host arrays.
function _form_offsets!(off, wrap, cstar, lim::PositivityLimiter, decomp, d::Int,
                        mode::Int)
    L = length(off)
    P = decomp.dims[d]
    send, recv = lim.send[d], lim.recv[d]
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

# The ends of d a θ pass and a correction take: the closed ends and the fold,
# as above, and which closed end is a patch interface.
function _limiter_ends(lim, decomp, d)
    closed_lo, closed_hi = _limiter_closed(lim, decomp, d)
    return (closed_lo, closed_hi, _limiter_fold(lim, decomp, d),
            closed_lo && lim.interface[d][1], closed_hi && lim.interface[d][2])
end

# The bound of a cell whose base state aggregates to q: the smaller of the run's
# ε and the fraction ε[3] of the cell's own ρ and ρe, at least zero.
@inline function _limiter_cell_bound(q, ε)
    ρ = q[1]
    e = ρ > 0 ? _limiter_internal(q) : zero(ρ)
    return (min(ε[1], ε[3] * max(ρ, zero(ρ))), min(ε[2], ε[3] * max(e, zero(e))))
end

# The run's bounds and the cell fraction, as the per-point bodies take them.
_limiter_bounds(lim) = (lim.eps_rho, lim.eps_e, oftype(lim.eps_rho, LIMITER_CELL_FRACTION))

# θ per face of direction d into `lim.theta`, a line per body, then the faces and
# the unguaranteed sides of this rank counted. Returns the number of faces
# limited that this rank owns, and whether its low face, which the rank below
# owns and counts, is limited anywhere: its node 1 takes that face's
# correction too.
function _limit_faces!(lim::PositivityLimiter, solver, Q, d::Int, τ, mode::Int)
    decomp = solver.decomp
    o1, o2, o3 = decomp.n_halo_d
    n = decomp.n_local[d]
    nA, nB = _transverse(decomp.n_local, d)
    ends = _limiter_ends(lim, decomp, d)
    closed_hi, fold = ends[2], ends[3]
    pointwise!(_limiter_theta_point!, solver.tmp_a, 1, nA, nB,
               lim.theta, lim.line_counts[d], _linear(Q), lim.faces,
               (solver.u, solver.v, solver.w), solver.c, solver.p, lim.inv_weights,
               lim.free, decomp.active, d, n, ends, solver.h[d],
               τ, mode, _limiter_bounds(lim), _limiter_layout(solver),
               mode == 2 ? lim.radial[d] : lim.areal[d],
               mode == 2 ? lim.volume_state : (true, true, true, true, true), solver.inv_J,
               mode == 1 && lim.areal[d], lim.rates, o1, o2, o3)
    return _limiter_tally!(lim, decomp, d, closed_hi, mode, fold,
                           mode == 1 && ends[5])
end

# The tallies a θ pass left per line in `line_counts[d]`: the faces limited,
# the low face limited where the rank below owns it, the unguaranteed sides,
# and the faces limited at r = 0. A fold's face is this rank's own and counted
# with its faces.
function _limiter_tally!(lim, decomp, d, closed_hi, mode, fold, end_face=false)
    n = decomp.n_local[d]
    tallies = lim.line_counts[d]
    nA, nB = size(tallies, 2), size(tallies, 3)
    limited, shared, sides, axis = _device_path(tallies) ? _tally_reduced(tallies) :
                                   _tally_sum(tallies)
    faces = ((closed_hi ? n - 1 : n) + (fold ? 1 : 0) + (end_face ? 1 : 0)) * nA * nB
    lim.counts[mode == 1 ? 1 : 3] += faces
    lim.counts[5] += sides
    lim.counts[6] += axis
    return limited, shared
end

# The four tallies summed over the lines: a host loop, and on device storage a
# reduction over the lines there with one copy of its four sums to the host.
function _tally_sum(tallies)
    limited = 0
    shared = 0
    sides = 0
    axis = 0
    @inbounds for b in 1:size(tallies, 3), a in 1:size(tallies, 2)
        limited += tallies[1, a, b]
        shared += tallies[2, a, b]
        sides += tallies[3, a, b]
        axis += tallies[4, a, b]
    end
    return limited, shared, sides, axis
end

function _tally_reduced(tallies)
    s = Array(sum(tallies; dims=(2, 3)))
    return s[1], s[2], s[3], s[4]
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
    speeds = lim.speeds
    Ql = _LinearState(Q)
    dQl = _LinearState(dQ)
    pointwise!(_limiter_radial_rate_point!, lim.rates, 1, nA, nB,
               lim.rates, speeds, Ql, dQl, _LinearState(du), regs, uvw, solver.c,
               solver.p, solver.rho, solver.inv_J, solver.inv_h[d], lim.weights,
               lim.areas[d], decomp.active, d, (fold, closed_hi, n), (A, B, dt), bounds,
               lay, lim.free, o1, o2, o3)
    # The face two ranks share takes the same θ on both, from both cells'
    # rates, along r here and along z in the pass that follows.
    for e in 1:3
        decomp.active[e] && decomp.dims[e] > 1 && exchange_dim!(lim.rates, decomp, e)
    end
    pointwise!(_limiter_radial_theta_point!, solver.tmp_a, 1, nA, nB,
               solver.tmp_b, lim.line_counts[d], Ql, lim.faces, lim.rates, uvw, solver.c,
               solver.p, solver.rho, solver.inv_J, solver.inv_h[d], lim.weights[d],
               lim.areas[d], speeds, lim.free, d, (fold, closed_lo, closed_hi, n), τ,
               bounds, lay, o1, o2, o3)
    limited, shared = _limiter_tally!(lim, decomp, d, closed_hi, 1, fold)
    lim.counts[2] += limited
    limited + shared > 0 || return nothing
    pointwise!(_limiter_radial_correct_point!, solver.tmp_a, n, nA, nB,
               dQl, lim.register_fields[d], lim.faces, solver.tmp_b, Ql, uvw, solver.p,
               solver.inv_J, solver.inv_h[d], lim.weights[d], lim.areas[d], speeds,
               lim.anchor[d], d, τ, (one(B) / B, one(B) / (B * dt)), lay, fold, lim.free,
               o1, o2, o3)
    return nothing
end

@inline _limiter_gamma(ρ, c, p) = p > 0 ? max(ρ * c * c / p, one(p)) : one(p)

# The Lax–Friedrichs speed at which Q ∓ F̃(Q)/a is admissible for the flux
# F̃ = G + ζ p e_r of a cell whose pressure fraction at the face is ζ:
# (1 − λv)(1 − λv(1 + (γ − 1)(1 − ζ))) > λ² ζ² c² (γ − 1)/(2γ), λ = 1/a,
# v = ±u, holds for a ≥ max(1, |1 + (γ − 1)(1 − ζ)|)|u| + ζ c √((γ − 1)/(2γ)),
# and the second term is kept at least c so that ζ = 1 gives |u| + c. The
# node's `_radial_node` holds s = √((γ − 1)/(2γ)).
@inline function _radial_speed(x, ζ)
    k = abs(1 + (x.γ - 1) * (1 - ζ))
    return max(one(k), k) * abs(x.u) + max(one(k), ζ * x.s) * x.c
end

# What a node contributes to the speeds of its two faces.
Base.@propagate_inbounds function _radial_node(ud, c, p, ρ, inv_J, inv_h, L)
    γ = _limiter_gamma(ρ[L], c[L], p[L])
    return (u=ud[L], c=c[L], γ=γ, s=sqrt((γ - 1) / (2γ)), ih=inv_h[L], iJ=inv_J[L])
end

# The speed at a face of area Ā between nodes l and r (`_radial_node`): the
# largest over both states and both cells' ζ = (inv_h / inv_J) / Ā, since each
# cell's half state holds both states. With r = l, the speed of one state at a
# closed end.
@inline function _radial_face_speed(l, r, Ā)
    ζL = l.ih / (l.iJ * Ā)
    ζR = r.ih / (r.iJ * Ā)
    return max(_radial_speed(l, ζL), _radial_speed(l, ζR), _radial_speed(r, ζL),
               _radial_speed(r, ζR))
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
Base.@propagate_inbounds function _limiter_node_increment(Q, dQ, du, regs, inv_J, inv_h,
                                                          act, d, A, B, dt, lay, I)
    ns, m1, m2, m3, ie, nc = lay
    md = d == 1 ? m1 : d == 2 ? m2 : m3
    T = typeof(B)
    out = (zero(T), zero(T), zero(T), zero(T), zero(T))
    # The directions' register sets differ in length (the radial pressure's
    # is one longer), so the tuple is read at literal positions: a runtime
    # index into it does not compile for a device.
    for cc in 1:nc
        x = iszero(A) ? dt * dQ[I, cc] : A * du[I, cc] + dt * dQ[I, cc]
        act[1] && (x += inv_J[I] * regs[1][cc][I])
        act[2] && (x += inv_J[I] * regs[2][cc][I])
        act[3] && (x += inv_J[I] * regs[3][cc][I])
        if cc == md
            r = d == 1 ? regs[1][nc+1][I] : d == 2 ? regs[2][nc+1][I] : regs[3][nc+1][I]
            x += inv_h[I] * r
        end
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

# The speed of each face of a radial line d into `speeds`, the face f at node
# f, and the rate of each cell over the share its face and geometric parts
# keep into `rates`: Σ R / (1 − β_n), with R the per-volume rates
# Ā a inv_J/W of its two faces (`AXIS_SHARE` of the outer face's for the face
# at the fold), γ u⁺ ΔĀ inv_J/W of its geometric part and 2(|u_e| + c)/W_e of
# every other active direction, and β_n the node part's share; Inf where the
# node part alone needs the whole cell. The face at a closed far end takes its
# one state; the fold's face takes none. The θ and correction passes read the
# speeds, and the sweep carries a node's part of them to its next face.
@inline function _limiter_radial_rate_point!(rates, speeds, Q, dQ, du, regs, uvw, c, p, ρ,
                                             inv_J, inv_h, W, areas, act, d, ends, coeffs,
                                             bounds, lay, free, o1, o2, o3, _, a, b)
    @inbounds begin
        T = eltype(rates)
        fold, closed_hi, n = ends
        A, B, dt = coeffs
        o = d == 1 ? o1 : d == 2 ? o2 : o3
        n1, n2 = size(rates, 1), size(rates, 2)
        sd = _line_stride(d, n1, n2)
        L0 = _line_linear(d, 0, a, b, o1, o2, o3, n1, n2)
        ud = uvw[d]
        lo = _radial_node(ud, c, p, ρ, inv_J, inv_h, L0)
        lo_bad = _limiter_unheld_bad(Q, free, L0, _line_node(d, 0, a, b, o1, o2, o3), lay)
        a_lo = zero(T)
        for pp in 0:n
            L = L0 + pp * sd
            hi = _radial_node(ud, c, p, ρ, inv_J, inv_h, L + sd)
            hi_bad = _limiter_unheld_bad(Q, free, L + sd,
                                         _line_node(d, pp + 1, a, b, o1, o2, o3), lay)
            Āp = areas[pp+o]
            # A node the limiter does not hold whose state is not admissible
            # leaves the face the speed of the other.
            a_hi = fold && pp == 0 ? zero(T) :
                   closed_hi && pp == n ? _radial_face_speed(lo, lo, Āp) :
                   lo_bad ? _radial_face_speed(hi, hi, Āp) :
                   hi_bad ? _radial_face_speed(lo, lo, Āp) :
                   _radial_face_speed(lo, hi, Āp)
            speeds[L] = a_hi
            if pp >= 1
                # The cell at node pp, between faces pp − ½ and pp + ½.
                I = _line_node(d, pp, a, b, o1, o2, o3)
                Ām = areas[pp-1+o]
                Rp = Āp * a_hi
                Rm = fold && pp == 1 ? T(AXIS_SHARE) * Rp : Ām * a_lo
                R = inv_J[L] / W[d][pp+o] *
                    (Rp + Rm + lo.γ * max(ud[L], zero(T)) * (Āp - Ām))
                for e in 1:3
                    (e == d || !act[e]) && continue
                    R += 2 * (abs(uvw[e][L]) + c[L]) / W[e][I[e]]
                end
                q = _limiter_state(Q, L, lay)
                δ = _limiter_node_increment(Q, dQ, du, regs, inv_J, inv_h, act, d, A, B,
                                            dt, lay, L)
                βn = one(T) / _limiter_reach(q, δ, _limiter_cell_bound(q, bounds))
                rates[L] = βn < 1 ? R / (1 - βn) : T(Inf)
            end
            lo = hi
            lo_bad = hi_bad
            a_lo = a_hi
        end
    end
    return nothing
end

# Whether the node at linear index L (Cartesian I) is one the limiter does not
# hold and its state is not admissible: a shell node interpolated across a
# shock, or an exempt parent node.
Base.@propagate_inbounds _limiter_unheld_bad(Q, free, L, I, lay) =
    !_limiter_constrained(free, I) && !_limiter_positive(_limiter_state(Q, L, lay))

# θ at each face of a radial line d, the face f in place of node f, and the
# line's tallies into `counts` (see `_limiter_tally!`), as the Cartesian θ
# pass takes them at a stage. A cell's part for the face is
# q ∓ (1/β)[H − τ (Ā G(q), p) as a node change], 1/β its rate over the face's,
# and its first-order counterpart the Lax–Friedrichs half state; the cell at
# the fold takes the fold face with the axis share, whose first-order values
# are 0 and τ p_1, and the cell on a closed far end takes its boundary face,
# never limited, in one part with its interior face, as on a Cartesian line.
# A side is guaranteed when τ times its cell's rate is below one; a cell
# whose node part needs the whole cell takes the first-order flux.
@inline function _limiter_radial_theta_point!(theta, counts, Q, faces, rates, uvw, c, p,
                                              ρ, inv_J, inv_h, W, areas, speeds, free, d,
                                              ends, τ, bounds, lay, o1, o2, o3, _, a, b)
    @inbounds begin
        T = eltype(theta)
        fold, closed_lo, closed_hi, n = ends
        o = d == 1 ? o1 : d == 2 ? o2 : o3
        n1, n2 = size(theta, 1), size(theta, 2)
        sd = _line_stride(d, n1, n2)
        L0 = _line_linear(d, 0, a, b, o1, o2, o3, n1, n2)
        ud = uvw[d]
        np = lay[6] + 1
        limited = 0
        shared = 0
        sides = 0
        axis = 0
        lo = _radial_cell(Q, free, inv_J, inv_h, W, bounds, lay, L0,
                          _line_node(d, 0, a, b, o1, o2, o3), o)
        for f in 0:n
            Ll = L0 + f * sd
            Lr = Ll + sd
            hi = _radial_cell(Q, free, inv_J, inv_h, W, bounds, lay, Lr,
                              _line_node(d, f + 1, a, b, o1, o2, o3), f + 1 + o)
            flag = 0
            θ = one(T)
            if !((closed_lo && f == 0) || (closed_hi && f == n))
                fold_face = fold && f == 0
                wall_r = closed_hi && f == n - 1
                Ā = areas[f+o]
                G = _limiter_face_state(faces, Ll, lay)
                Π = faces[np][Ll]
                cl = !fold_face && lo.constrained
                cr = hi.constrained
                a_face = fold_face ? zero(T) : speeds[Ll]
                ql, εl, sJl, shl = lo.q, lo.ε, lo.sJ, lo.sh
                qr, εr, sJr, shr = hi.q, hi.ε, hi.sJ, hi.sh
                # The low cell's part for its outer face.
                kl = rates[Ll] / (sJl * Ā * a_face)
                bl = _limiter_axpy(ql, kl * τ,
                                   _radial_vector(map(x -> Ā * x,
                                                      _radial_flux(ql, ud[Ll], p[Ll])),
                                                  p[Ll], sJl, shl, d))
                Hl = _radial_vector(G, Π, sJl, shl, d)
                # The high cell's part for its inner face.
                if fold_face
                    Ā1 = areas[1+o]
                    kr = rates[Lr] / (sJr * T(AXIS_SHARE) * Ā1 * speeds[Lr])
                    br = _limiter_axpy(qr, -kr * τ, _radial_vector(
                                       (zero(T), zero(T), zero(T), zero(T), zero(T)),
                                       p[Lr], sJr, shr, d))
                elseif wall_r
                    Āw = areas[f+1+o]
                    aw = speeds[Lr]
                    γ = _limiter_gamma(ρ[Lr], c[Lr], p[Lr])
                    kr = rates[Lr] / (sJr * (Ā * a_face + Āw * aw +
                                             γ * max(ud[Lr], zero(T)) * (Āw - Ā)))
                    br = _limiter_axpy(qr, -kr, _radial_vector(
                                       _limiter_face_state(faces, Lr, lay), faces[np][Lr],
                                       sJr, shr, d))
                else
                    kr = rates[Lr] / (sJr * Ā * a_face)
                    br = _limiter_axpy(qr, -kr * τ,
                                       _radial_vector(map(x -> Ā * x,
                                                          _radial_flux(qr, ud[Lr], p[Lr])),
                                                      p[Lr], sJr, shr, d))
                end
                Hr = _radial_vector(G, Π, sJr, shr, d)
                okl = isfinite(kl)
                okr = isfinite(kr)
                need_l = cl && !(okl && _limiter_admissible(_limiter_axpy(bl, -kl, Hl), εl))
                need = need_l ||
                       (cr && !(okr && _limiter_admissible(_limiter_axpy(br, kr, Hr), εr)))
                if need
                    # First order: τ Ā times the Rusanov flux, and τ {p}; at the
                    # fold 0 and τ p_1.
                    if fold_face
                        GL = (zero(T), zero(T), zero(T), zero(T), zero(T))
                        ΠL = τ * p[Lr]
                    else
                        # A node the limiter does not hold whose state is not
                        # admissible takes the held cell's state.
                        own_l = !lo.constrained && !_limiter_positive(ql)
                        own_r = !hi.constrained && !_limiter_positive(qr)
                        qa, ua, pa = own_l ? (qr, ud[Lr], p[Lr]) : (ql, ud[Ll], p[Ll])
                        qb, ub, pb = own_r ? (ql, ud[Ll], p[Ll]) : (qr, ud[Lr], p[Lr])
                        Fl = _radial_flux(qa, ua, pa)
                        Fr = _radial_flux(qb, ub, pb)
                        GL = map((x, y, u, v) ->
                                     τ * Ā * ((x + y) / 2 - a_face * (v - u) / 2),
                                 Fl, Fr, qa, qb)
                        ΠL = τ * (pa + pb) / 2
                    end
                    if cl
                        t, ok = okl ?
                                _limiter_side(bl, Hl, _radial_vector(GL, ΠL, sJl, shl, d),
                                              -kl, εl) : (zero(T), false)
                        θ = min(θ, t)
                        (ok && τ * rates[Ll] < 1) || (flag |= 1)
                    end
                    if cr
                        t, ok = okr ?
                                _limiter_side(br, Hr, _radial_vector(GL, ΠL, sJr, shr, d),
                                              kr, εr) : (zero(T), false)
                        θ = min(θ, t)
                        (ok && τ * rates[Lr] < 1) || (flag |= 2)
                    end
                end
            end
            theta[Ll] = θ
            if θ < 1
                (f >= 1 || fold) ? (limited += 1) : (shared += 1)
                fold && f == 0 && (axis += 1)
            end
            f >= 1 && (flag & 1) != 0 && (sides += 1)
            f <= n - 1 && (flag & 2) != 0 && (sides += 1)
            lo = hi
        end
        counts[1, a, b] = limited
        counts[2, a, b] = shared
        counts[3, a, b] = sides
        counts[4, a, b] = axis
    end
    return nothing
end

# A radial cell's state and bound, its inverse Jacobian and inverse spacing
# over its weight W[pw], and whether no face condition overwrites it: what a
# θ pass reads of it at both its faces.
Base.@propagate_inbounds function _radial_cell(Q, free, inv_J, inv_h, W, bounds, lay, L,
                                               I, pw)
    q = _limiter_state(Q, L, lay)
    return (q=q, ε=_limiter_cell_bound(q, bounds), sJ=inv_J[L] / W[pw],
            sh=inv_h[L] / W[pw], constrained=_limiter_constrained(free, I))
end

# The corrections at node pp of a radial line from its two faces, where either
# is limited: δ = (θ − 1)(Φ − Φ_L) per face for each conserved component and
# for the pressure, the node changing by inv_J (δ_{p−½} − δ_{p+½})/W_p and,
# on the radial momentum, by inv_h times the pressure's. Each enters dQ / (B dt)
# and leaves its register, and a limited fold face enters the face register at
# the fold, whose running sum the next stage starts from.
@inline function _limiter_radial_correct_point!(dQ, regs, faces, theta, Q, uvw, p,
                                                inv_J, inv_h, W, areas, speeds, anchor, d,
                                                τ, scales, lay, fold, free, o1, o2, o3, pp,
                                                a, b)
    @inbounds begin
        T = eltype(theta)
        n1, n2 = size(theta, 1), size(theta, 2)
        I = _line_linear(d, pp, a, b, o1, o2, o3, n1, n2)
        sd = _line_stride(d, n1, n2)
        Im = I - sd
        Ip = I + sd
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
        am = θm < 1 && !fold_face ? speeds[Im] : zero(T)
        ap = θp < 1 ? speeds[I] : zero(T)
        # The nodes replaced in the first-order flux, as the face pass replaced
        # them: a neighbour the limiter does not hold whose state is not
        # admissible by this node, and this node, if it is one, by the
        # neighbour, so both nodes of a face take one flux (see
        # `_limiter_correct_point!`).
        Jm = !fold_face && _limiter_unheld_bad(Q, free, Im,
                                               _line_node(d, pp - 1, a, b, o1, o2, o3),
                                               lay) ? I : Im
        Jp = _limiter_unheld_bad(Q, free, Ip, _line_node(d, pp + 1, a, b, o1, o2, o3),
                                 lay) ? I : Ip
        own = _limiter_unheld_bad(Q, free, I, _line_node(d, pp, a, b, o1, o2, o3), lay)
        Km = own ? Im : I
        Kp = own ? Ip : I
        inv_W = one(T) / W[pp+o]
        for cc in 0:nc
            # cc = 0 is the pressure, kept in register nc + 1.
            slot = cc == 0 ? nc + 1 : cc
            change = zero(T)
            if θm < 1
                gl = cc == 0 ? τ * (fold_face ? p[I] : (p[Jm] + p[Km]) / 2) :
                     fold_face ? zero(T) :
                     τ * Ām * _radial_lf_component(Q, ud, p, cc, Jm, Km, am, ie)
                δ = (θm - 1) * (faces[slot][Im] - gl)
                change += δ
                fold_face && (anchor[slot, a, b] += δ * inv_B)
            end
            if θp < 1
                gl = cc == 0 ? τ * (p[Kp] + p[Jp]) / 2 :
                     τ * Āp * _radial_lf_component(Q, ud, p, cc, Kp, Jp, ap, ie)
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
Base.@propagate_inbounds function _radial_lf_component(Q, ud, p, cc, L, R, a, ie)
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
_limited_filter_state!(solver, Q::ConservedState) =
    _limited_filter_pass!(getfield(solver, :positivity), solver, Q)

# The multi-patch `filter_state!`, each patch's passes limited. The filter's
# row at an interface node is the identity, as at a wall, and the averaging
# that follows leaves the node as it was; the shell imposition overwrites a
# refined patch's node at a coarse-fine face. A subcycled refined level
# filters inside `_advance_level!` (`_limited_level_filter!`).
function _limited_filter_state!(solver, states::Vector{<:ConservedState})
    lims = _follow_patches!(solver, getfield(solver, :positivity))
    subcycle = getfield(solver, :subcycle)
    for lev in getfield(solver, :levels)
        subcycle && lev.index > 0 && continue
        _limited_level_filter!(lims, solver, lev, states)
    end
    _ledger!(solver, states, :filter)
    sync_patches!(solver, states)
    _ledger!(solver, states, :same_level)
    return states
end

# `_level_filter!` with each patch's passes limited, and `_level_filter!`
# itself where the step is not limited.
_limited_level_filter!(::Nothing, solver, lev, states) =
    _level_filter!(solver, lev, states)
function _limited_level_filter!(lims::Vector{Any}, solver, lev, states)
    patches = getfield(solver, :patches)
    for i in lev.patches
        _limited_filter_pass!(lims[i], PatchSolver(solver, patches[i]), states[i])
    end
    return states
end

# The passes of `filter_state!` under the weighting `:none`, the residual a
# child level covers dropped as there (`_child_mask!`), each followed by the
# filter limiter.
function _limited_filter_pass!(lim::PositivityLimiter, solver, Q)
    decomp = solver.decomp
    n_cons = solver.equations.n_cons
    comps = [view(Q, :, :, :, c) for c in 1:n_cons]
    n1, n2, n3 = padded_extent(decomp)
    o1, o2, o3 = decomp.n_halo_d
    masked = _masked_filter(solver, Q, false)
    for d in 1:3
        decomp.active[d] || continue
        w = filter_weight(solver, d)
        exchange_dim_batch!(comps, decomp, d)
        mask = _child_mask!(solver, Q, comps, d, masked)
        for c in 1:n_cons
            pointwise!(_copy_component_point!, solver.tmp_a, n1, n2, n3,
                       lim.base[c], Q, c)
        end
        for c in 1:n_cons
            _filter_line!(solver.tmp_a, comps[c], solver, d, cons_parity(solver, d, c),
                          mask)
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
        Ql = _LinearState(Q)
        pointwise!(_limiter_correct_point!, solver.tmp_a, n, nA, nB,
                   Ql, lim.register_fields[d], lim.faces, lim.theta, Ql,
                   (solver.u, solver.v, solver.w), solver.c, solver.p,
                   lim.weights[d], d, solver.h[d], zero(w), 2, (one(w), one(w)),
                   _limiter_layout(solver),
                   _limiter_ends(lim, decomp, d), n,
                   lim.radial[d], lim.volume, solver.inv_J, false,
                   (lim.anchor[d], lim.anchor_hi[d]), lim.free,
                   o1, o2, o3)
    end
    return Q
end

# The pre-pass state of a filter pass, held per component in the workspace,
# indexed as a `_LinearState` is.
struct _BaseState{F}
    fields::F
end
Base.@propagate_inbounds Base.getindex(s::_BaseState, l::Int, c::Int) = s.fields[c][l]
Adapt.adapt_structure(to, s::_BaseState) = _BaseState(adapt(to, s.fields))

# The state a θ pass reads, in the linear form.
_linear(Q::ConservedState) = _LinearState(Q)
_linear(s::_BaseState) = s

# --- Per-point bodies --------------------------------------------------------

# The subtraction of a divergence from component `c` of dQ, as
# `_subtract_div_point!` and `_subtract_jac_div_point!` make it, and the
# register update r ← A r + dt div from the same read over the interior.
# RKA[1] = 0, and the first stage assigns, so a register left non-finite by an
# abandoned step is forgotten.
@inline function _limiter_subtract_point!(dQ, r, div, inv_J, c, A, dt, o1, o2, o3,
                                          i, j, k)
    @inbounds begin
        I = CartesianIndex(i + o1, j + o2, k + o3)
        x = div[I]
        if inv_J === nothing
            dQ[I, c] -= x
        else
            dQ[I, c] -= inv_J[I] * x
        end
        r[I] = ifelse(iszero(A), dt * x, A * r[I] + dt * x)
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

# The face register at the high face of a line ending at a patch interface,
# Φ ← A Φ + dt F_n, the point flux of its last node, as the low face's is of
# node 1: the patch across the interface holds the same node as its node 1.
@inline function _limiter_end_anchor_point!(anchor, F, A, dt, c, d, n, o1, o2, o3,
                                            a, b, _)
    @inbounds begin
        value = F[_line_node(d, n, a, b, o1, o2, o3)]
        anchor[c, a, b] = ifelse(iszero(A), dt * value, A * anchor[c, a, b] + dt * value)
    end
    return nothing
end

# The stage's face value at the high face of a line ending at a patch
# interface: B_k times its register, in place of the running sum's end, which
# meets it to round-off. Both patches then take one value at the shared node's
# face, bit for bit where their point fluxes agree.
@inline function _limiter_end_face_point!(faces, anchor, scale, n_comp, n, d, o1, o2,
                                          o3, a, b, _)
    @inbounds begin
        n1, n2 = size(faces[1], 1), size(faces[1], 2)
        L = _line_linear(d, n, a, b, o1, o2, o3, n1, n2)
        for c in 1:n_comp
            faces[c][L] = scale * anchor[c, a, b]
        end
    end
    return nothing
end

# One line's local running sums into the face slots: slot p holds face p + ½,
# slot 0 (a halo node) the low face of the block, at zero. Mode 1 sums W r,
# mode 2 the filter correction −ω Δ with ω = W/h, and in mode 2 the filter's
# face relation is measured at local face `j` when j > 0:
# (Σ_s l_s S_{j+s} + w ψ_j) / Σ_s l_s. Where `volume` is set, a component
# marked in `mask` sums −ω J Δ instead. On a line this rank holds whole
# (`ends[1]`) the offsets are the line's own, the stage's face register
# `anchor` or the filter's measured constant, and the face values are
# finished here as `_limiter_offset_point!` finishes them on a decomposed one,
# scaled by `ends[3]`, the global low face's value at the high face of a
# periodic line (`ends[2]`): a stage's in the same sweep as the sums, a
# filter pass's after its measurement.
@inline function _limiter_scan_point!(faces, regs, Q, base, W, hd, totals, aux, anchor,
                                      n_cons, n, d, mode, j, lhs, psi, wf, volume, mask,
                                      inv_J, ends, o1, o2, o3, _, a, b)
    @inbounds begin
        o = d == 1 ? o1 : d == 2 ? o2 : o3
        T = eltype(totals)
        whole, wrap_high, scale = ends
        n1, n2 = size(inv_J, 1), size(inv_J, 2)
        sd = _line_stride(d, n1, n2)
        L0 = _line_linear(d, 0, a, b, o1, o2, o3, n1, n2)
        for c in 1:n_cons
            fc = faces[c]
            S = zero(T)
            weighted = volume && mask[c]
            if mode == 1
                rc = regs[c]
                if whole
                    off = anchor[c, a, b]
                    fc[L0] = scale * ((off + S) - zero(T))
                    for p in 1:n
                        L = L0 + p * sd
                        S += W[p+o] * rc[L]
                        fc[L] = scale * ((off + S) - zero(T))
                    end
                    wrap_high && (fc[L0+n*sd] = scale * (off - zero(T)))
                    totals[c, a, b] = S
                    continue
                end
                fc[L0] = S
                for p in 1:n
                    L = L0 + p * sd
                    S += W[p+o] * rc[L]
                    fc[L] = S
                end
            else
                bc = base[c]
                fc[L0] = S
                for p in 1:n
                    L = L0 + p * sd
                    if weighted
                        S -= (W[p+o] / hd) / inv_J[L] * (Q[L, c] - bc[L])
                    else
                        S -= (W[p+o] / hd) * (Q[L, c] - bc[L])
                    end
                    fc[L] = S
                end
            end
            totals[c, a, b] = S
            if mode == 2 && j > 0
                q = length(lhs)
                Lj = L0 + j * sd
                acc = fc[Lj]
                sum_l = one(T)
                for s in 1:q
                    acc += lhs[s] * (fc[Lj-s*sd] + fc[Lj+s*sd])
                    sum_l += 2 * lhs[s]
                end
                M = length(psi) ÷ 2
                ψ = zero(T)
                bc = base[c]
                for l in 1-M:M
                    ψ += psi[l+M] * bc[Lj+l*sd]
                end
                aux[c, a, b] = (acc + wf * ψ) / sum_l
            end
            whole || continue
            # A filter pass: no offset, and the measured constant removed.
            off = zero(T)
            cstar = aux[c, a, b]
            for f in 0:n
                L = L0 + f * sd
                v = wrap_high && f == n ? off : off + fc[L]
                fc[L] = weighted ? scale * v : scale * (v - cstar)
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
Base.@propagate_inbounds function _limiter_state(Q, I, lay)
    ns, m1, m2, m3, ie, _ = lay
    ρ = zero(eltype(Q[I, 1]))
    for sp in 1:ns
        ρ += Q[I, sp]
    end
    return (ρ, Q[I, m1], Q[I, m2], Q[I, m3], Q[I, ie])
end

Base.@propagate_inbounds function _limiter_face_state(faces, I, lay)
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
# ρ > 0 and ρe > 0.
@inline _limiter_positive(q) =
    q[1] > 0 && 2 * q[1] * q[5] - (q[2] * q[2] + q[3] * q[3] + q[4] * q[4]) > 0
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

# Σ_d (|u_d| + c)/W_d at the node of linear index L and Cartesian index I, the
# cell's total first-order rate, from the inverse weights.
Base.@propagate_inbounds function _limiter_rate(uvw, c, inv_W, act, L, I)
    Λ = zero(eltype(c))
    for e in 1:3
        act[e] || continue
        Λ += (abs(uvw[e][L]) + c[L]) * inv_W[e][I[e]]
    end
    return Λ
end

# The nodes of a parent patch a child's restriction overwrites after every
# step (`Patch.overwritten`), which the limiter does not hold. Under subcycling
# the parent's step runs there at up to `OVERWRITTEN_CFL`, beyond the
# first-order bound, and held to their cell bounds stage after stage they fell
# to temperatures of 1e-12, at which κ* = ρc/T set the root step. Their
# neighbours' first-order fluxes do not read them where they are not
# admissible. Empty where no node is exempt.
_limiter_exempt(solver) = similar(solver.tmp_a, eltype(solver.tmp_a), 0, 0, 0)
_limiter_exempt(ps::PatchSolver) =
    isempty(ps.patch.overwritten) ?
    similar(ps.tmp_a, eltype(ps.tmp_a), 0, 0, 0) : ps.patch.overwritten

# A node is held unless a face condition overwrites it or it is exempt
# (`_limiter_exempt`).
Base.@propagate_inbounds _limiter_constrained(free, I) =
    !(free[1][I[1]] | free[2][I[2]] | free[3][I[3]]) &&
    (isempty(free[4]) || iszero(free[4][I]))

# The half-state scale of a cell along d: 2Λ/(|u_d| + c) at a stage, its rate
# over |u_d| + c along an area-weighted direction, Inf where that speed
# vanishes, and 2/ω = 2h/W in a filter pass.
Base.@propagate_inbounds function _limiter_scale(uvw, c, inv_W, act, rates, d, hd, mode,
                                                 areal, L, I)
    T = eltype(c)
    if mode == 1
        sp = abs(uvw[d][L]) + c[L]
        sp > 0 || return T(Inf)
        return areal ? rates[L] / sp : 2 * _limiter_rate(uvw, c, inv_W, act, L, I) / sp
    end
    return 2 * hd * inv_W[d][I[d]]
end

# θ at each face of a line of direction d into `theta`, the face f in place of
# node f, and the line's tallies into `counts` (see `_limiter_tally!`), an
# unguaranteed side flagged 1 for the low cell and 2 for the high one. A global
# closed end is not limited; a fold's face is, from node 1's side alone, its
# other cell node 1's mirror. Mode 1 is a stage (half states scaled by
# 2Λ/(|u_d| + c) and the first-order flux τ F_LF); mode 2 a filter pass (2/ω,
# toward zero). Each cell keeps its own bound (`_limiter_cell_bound`). Under
# `volume` a cell takes the components marked in `vmask` divided by its J: at a
# stage along an area-weighted direction (`areal`) every component, whose
# face values are A_d times a flux on a line of constant A_d = J, and in a
# filter pass on a radial grid the components weighted by J. An areal stage
# reads each cell's rate from `rates`, which the radial pass fills. A node's
# state, bound, half-state scale s (s_d = 2Λ/(|u_d| + c) at a stage, the low
# cell taking −s) and constraint serve both its faces: the sweep carries the
# high cell's to the next face.
@inline function _limiter_theta_point!(theta, counts, Q, faces, uvw, c, p, inv_W, free,
                                       act, d, n, closed, hd, τ, mode, bounds, lay,
                                       volume, vmask, inv_J, areal, rates,
                                       o1, o2, o3, _, a, b)
    @inbounds begin
        T = eltype(theta)
        ud = uvw[d]
        n1, n2 = size(theta, 1), size(theta, 2)
        sd = _line_stride(d, n1, n2)
        L0 = _line_linear(d, 0, a, b, o1, o2, o3, n1, n2)
        limited = 0
        shared = 0
        sides = 0
        I0 = _line_node(d, 0, a, b, o1, o2, o3)
        q_lo = _limiter_state(Q, L0, lay)
        ε_lo = _limiter_cell_bound(q_lo, bounds)
        s_lo = _limiter_scale(uvw, c, inv_W, act, rates, d, hd, mode, areal, L0, I0)
        free_lo = _limiter_constrained(free, I0)
        for f in 0:n
            Ll = L0 + f * sd
            Lr = Ll + sd
            Ir = _line_node(d, f + 1, a, b, o1, o2, o3)
            q_hi = _limiter_state(Q, Lr, lay)
            ε_hi = _limiter_cell_bound(q_hi, bounds)
            s_hi = _limiter_scale(uvw, c, inv_W, act, rates, d, hd, mode, areal, Lr, Ir)
            free_hi = _limiter_constrained(free, Ir)
            G = _limiter_face_state(faces, Ll, lay)
            flag = 0
            θ = one(T)
            lo_bnd = closed[1] && f == 0
            hi_bnd = closed[2] && f == n
            fold_face = closed[3] && f == 0
            # The low cell takes −s; the sign change is exact, so −s is the
            # low cell's own expression.
            sl = lo_bnd || (mode == 1 && areal && fold_face) ? zero(T) : -s_lo
            sr = hi_bnd ? zero(T) : s_hi
            if !(lo_bnd || hi_bnd)
                # The filter's row at a closed end is the identity, and its
                # correction leaves the end node alone; so does the limiter.
                wall_l = closed[1] && f == 1
                wall_r = closed[2] && f == n - 1
                cl = free_lo && !(mode == 2 && wall_l) && !fold_face
                cr = free_hi && !(mode == 2 && wall_r)
                qr = q_hi
                # The state across a fold is node 1's mirror, its normal momentum
                # reversed.
                ql = fold_face ? _limiter_mirror(qr, d) : q_lo
                uL = fold_face ? -ud[Lr] : ud[Ll]
                cL = fold_face ? c[Lr] : c[Ll]
                pL = fold_face ? p[Lr] : p[Ll]
                εl = fold_face ? _limiter_cell_bound(ql, bounds) : ε_lo
                εr = ε_hi
                Gl = volume ? _limiter_volume(G, vmask, inv_J[Ll]) : G
                Gr = volume ? _limiter_volume(G, vmask, inv_J[Lr]) : G
                # At a stage, a cell on a closed end keeps its boundary face, which
                # is never limited, in the same part as its interior face: one part
                # of weight α_d, q + (G_boundary − G_interior)/(α_d W), in place of
                # two halves, so the wall's flux enters at its own size, not twice.
                bl = ql; br = qr
                Gbl = G; Gbr = G
                if mode == 1 && wall_l
                    sl /= 2
                    Gb = _limiter_face_state(faces, L0, lay)
                    Gbl = volume ? _limiter_volume(Gb, vmask, inv_J[Ll]) : Gb
                    bl = _limiter_axpy(ql, -sl, Gbl)
                end
                if mode == 1 && wall_r
                    sr /= 2
                    Gb = _limiter_face_state(faces, L0 + n * sd, lay)
                    Gbr = volume ? _limiter_volume(Gb, vmask, inv_J[Lr]) : Gb
                    br = _limiter_axpy(qr, -sr, Gbr)
                end
                need = (cl && !_limiter_admissible(_limiter_axpy(bl, sl, Gl), εl)) ||
                       (cr && !_limiter_admissible(_limiter_axpy(br, sr, Gr), εr))
                if need
                    GL = (zero(T), zero(T), zero(T), zero(T), zero(T))
                    a_face = zero(T)
                    if mode == 1
                        # A node the limiter does not hold whose state is not
                        # admissible (a shell node interpolated across a shock)
                        # takes the held cell's state in the first-order flux.
                        own_l = !free_lo && !fold_face && !_limiter_positive(ql)
                        own_r = !free_hi && !_limiter_positive(qr)
                        qa, ua, ca, pa = own_l ? (qr, ud[Lr], c[Lr], p[Lr]) :
                                                 (ql, uL, cL, pL)
                        qb, ub, cb, pb = own_r ? (ql, uL, cL, pL) :
                                                 (qr, ud[Lr], c[Lr], p[Lr])
                        a_face = max(abs(ua) + ca, abs(ub) + cb)
                        GL = _limiter_lf_state(qa, qb, ua, ub, pa, pb, a_face, τ, d)
                    end
                    # The end cell at a patch interface blends its end face with
                    # the face beside it, toward the point flux of its state.
                    GLl = mode == 1 && wall_l && closed[4] ?
                          _limiter_end_flux(GL, Gbl, ql, uL, pL, τ, d) : GL
                    GLr = mode == 1 && wall_r && closed[5] ?
                          _limiter_end_flux(GL, Gbr, qr, ud[Lr], p[Lr], τ, d) : GL
                    # A side whose first-order half state is not admissible, or
                    # whose first-order bound τ a |s| ≤ 1 fails, is unguaranteed.
                    if cl
                        t, ok = _limiter_side(bl, Gl, GLl, sl, εl)
                        θ = min(θ, t)
                        (ok && τ * a_face * abs(sl) <= 1) || (flag |= 1)
                    end
                    if cr
                        t, ok = _limiter_side(br, Gr, GLr, sr, εr)
                        θ = min(θ, t)
                        (ok && τ * a_face * abs(sr) <= 1) || (flag |= 2)
                    end
                end
            end
            theta[Ll] = θ
            θ < 1 && ((f >= 1 || closed[3]) ? (limited += 1) : (shared += 1))
            f >= 1 && (flag & 1) != 0 && (sides += 1)
            f <= n - 1 && (flag & 2) != 0 && (sides += 1)
            q_lo, ε_lo, s_lo, free_lo = q_hi, ε_hi, s_hi, free_hi
        end
        # A patch's end face takes the θ of the face beside it.
        if mode == 1 && closed[5]
            θe = theta[L0+(n-1)*sd]
            theta[L0+n*sd] = θe
            θe < 1 && (limited += 1)
        end
        if mode == 1 && closed[4]
            θe = theta[L0+sd]
            theta[L0] = θe
            θe < 1 && (shared += 1)
        end
        counts[1, a, b] = limited
        counts[2, a, b] = shared
        counts[3, a, b] = sides
        counts[4, a, b] = 0
    end
    return nothing
end

# The first-order flux an end cell at a patch interface blends toward, as the
# face beside it sees it: that face's Lax–Friedrichs flux GL, less τ times the
# point flux of the cell's state f(q) at the end face, plus the end face's own
# value Gb, which the cell's half state has already taken out.
@inline _limiter_end_flux(GL, Gb, q, u, p, τ, d) =
    map((x, b, f) -> x + b - f, GL, Gb, _limiter_lf_state(q, q, u, u, p, p, zero(u), τ, d))

# A state reflected across a plane normal to d.
@inline _limiter_mirror(q, d) =
    (q[1], d == 1 ? -q[2] : q[2], d == 2 ? -q[3] : q[3], d == 3 ? -q[4] : q[4], q[5])

# The aggregated face values a cell divides by its J where `vmask` is set.
@inline _limiter_volume(G, vmask, inv_J) = map((g, m) -> m ? g * inv_J : g, G, vmask)

# τ times component `cc` of the Lax–Friedrichs flux along d between L and R.
Base.@propagate_inbounds function _limiter_lf_component(Q, uvw, c, p, cc, L, R, τ, d, lay)
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
                                         volume, mask, inv_J, areal, anchors, free,
                                         o1, o2, o3, pp, a, b)
    @inbounds begin
        T = eltype(theta)
        # A filter pass leaves a closed end's node alone (see the face pass).
        mode == 2 && ((closed[1] && pp == 1) || (closed[2] && pp == n)) &&
            return nothing
        n1, n2 = size(theta, 1), size(theta, 2)
        I = _line_linear(d, pp, a, b, o1, o2, o3, n1, n2)
        Im = I - _line_stride(d, n1, n2)
        Ip = I + _line_stride(d, n1, n2)
        θm = theta[Im]
        θp = theta[I]
        (θm < 1 || θp < 1) || return nothing
        o = d == 1 ? o1 : d == 2 ? o2 : o3
        Wp = mode == 1 ? W[pp+o] : W[pp+o] / hd
        n_cons = lay[6]
        inv_B, inv_Bdt = scales
        fold_face = closed[3] && pp == 1
        # A patch's end face takes the point flux of its node, and its
        # correction enters the face register at that end, as a fold's does.
        end_lo = closed[4] && pp == 1
        end_hi = closed[5] && pp == n
        # A node the limiter does not hold whose state is not admissible is
        # replaced by the other node of the face in the first-order flux, as
        # the face pass replaced it: a neighbour by this node, and this node
        # by the neighbour. Both nodes of a face then take one flux, and the
        # corrections telescope; a face whose two corrections differed would
        # leave their difference in the face register, carried by every later
        # stage's running sum to the far end of the line.
        own_m = mode == 1 && !fold_face && !end_lo &&
                !_limiter_constrained(free, _line_node(d, pp - 1, a, b, o1, o2, o3)) &&
                !_limiter_positive(_limiter_state(Q, Im, lay))
        own_p = mode == 1 && !end_hi &&
                !_limiter_constrained(free, _line_node(d, pp + 1, a, b, o1, o2, o3)) &&
                !_limiter_positive(_limiter_state(Q, Ip, lay))
        own = mode == 1 && !_limiter_constrained(free, _line_node(d, pp, a, b, o1, o2, o3)) &&
              !_limiter_positive(_limiter_state(Q, I, lay))
        Jm = end_lo || own_m ? I : Im
        Km = own && !end_lo ? Im : I
        Jp = end_hi || own_p ? I : Ip
        Kp = own && !end_hi ? Ip : I
        anchor, anchor_hi = anchors
        for cc in 1:n_cons
            change = zero(T)
            if θm < 1
                gl = mode == 2 ? zero(T) :
                     fold_face ? _limiter_lf_mirror(Q, uvw, c, pr, cc, I, τ, d, lay) :
                     _limiter_lf_component(Q, uvw, c, pr, cc, Jm, Km, τ, d, lay)
                areal && (gl /= inv_J[I])
                δ = (θm - 1) * (faces[cc][Im] - gl)
                change += δ
                mode == 1 && (fold_face || end_lo) && (anchor[cc, a, b] += δ * inv_B)
            end
            if θp < 1
                gl = mode == 1 ? _limiter_lf_component(Q, uvw, c, pr, cc, Kp, Jp, τ, d,
                                                       lay) : zero(T)
                areal && (gl /= inv_J[I])
                δp = (θp - 1) * (faces[cc][I] - gl)
                change -= δp
                mode == 1 && end_hi && (anchor_hi[cc, a, b] += δp * inv_B)
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
Base.@propagate_inbounds function _limiter_lf_mirror(Q, uvw, c, p, cc, I, τ, d, lay)
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
