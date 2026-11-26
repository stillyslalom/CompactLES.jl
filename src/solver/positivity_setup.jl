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
# at a wall or a coarse-fine end is the identity, and the limiter leaves that
# node alone as well: a limited face beside it corrects its interior neighbour
# only, so the line's total moves by that correction, as it does under the
# closure rows. At a node two same-level patches share the filter's rows
# conserve and the node takes a correction, which the limiter holds: the node
# takes its end face and the face beside it in one part, by one θ that the
# exchange of a same-level interface makes the same on both patches
# (`_filter_ends`).
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
# The bound on ρe is at least `LIMITER_ENERGY_ULPS` eps(T) |E|. At a node the
# flow keeps pushing down, as in the cold gas ahead of the planar Noh shock
# (ρe/E = 3e-4), every limited write lands on the cell's bound, and a bound of
# a fraction of the cell's own ρe falls a hundredfold per write until it is
# below the rounding step of E; the last write then leaves ρe at zero or a
# unit or two in the last place below it. The floor keeps the bound 16 to 32
# units in the last place of E above zero: on the planar Noh on patches and
# tiles, under one-ulp perturbations of the initial state, the multiple 4 left
# no inadmissible point and the multiples 1 and 2 did. In Float32 the cold
# gas there holds about 2500 eps(T) |E|, so the floor stays below its 1%
# bound.
#
# A write aimed at the floor rounds to a few units in the last place of E on
# either side of it. Where the floor sets a cell's bound, a state therefore
# holds the bound if its ρe is at least the floor less
# `LIMITER_ENERGY_TOLERANCE` eps(T) |E|, or the bound without the floor where
# that is larger (`_limiter_cell_bound`). Every test of a state against the
# bound takes that threshold (the high-order and first-order half states, and
# a radial cell's node part), while θ is still interpolated to the bound itself
# (`_limiter_side`). What the limiter guarantees for such a cell is ρe at least
# the bound less the tolerance, at least (`LIMITER_ENERGY_ULPS` −
# `LIMITER_ENERGY_TOLERANCE`) eps(T) |E| > 0, so a held cell stays admissible
# to `state_report`, whose test is ρe > 0. Where a fraction of the cell's own
# state or ε sets the bound, the threshold is the bound, and a run in which the
# floor never sets one is bitwise the run without the tolerance. Without it a
# cell rounded under the floor is below its bound, and the unguaranteed sides
# of the planar Noh on patches and tiles, and in Float32, rise well above their
# count under a bound without the floor; under one-ulp perturbations of the
# initial state, 3 is the smallest tolerance that leaves no more of them than
# that. A cell whose ρe lies under the threshold is below its own bound: its
# faces take the first-order flux where its half states need it, and it is
# counted unguaranteed.
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
# A tile's radial line closes at a face it shares with another tile as a
# Cartesian line does (see Same-level patches), and folds at r = 0 only on the
# tiles holding the axis. Its dual areas start at the first
# node's radius and add the weights, whose sum over a closed line is its
# length, so at the end face they reach the shared node's own area A_N on
# both tiles. The end face carries the node's point flux, A_N G(q_N) and p_N,
# so its first-order counterpart is τ A_N G(q_N) and τ p_N, and the end cell's
# first-order update is the Lax–Friedrichs part of its interior face and the
# geometric part −τ (A_N − Ā) G(q_N)/V: the end face adds nothing at first
# order. The end cell takes both its faces and its geometric part in one part,
# as a cell on a closed far end does, its end face blended toward that point
# flux by the θ of the face beside it, and the exchange of θ gives both tiles
# the smaller of their two. The end faces' corrections enter the face
# registers at both ends, so the two tiles keep one value there. Their θ is
# exchanged between the θ pass and the correction, so a tile of a radial grid
# holds its rates, face speeds, face values and θ itself, as a patch with a
# same-level interface holds its face values and θ.
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
# node's state, by one θ on both patches (`_exchange_interface_theta!`).
#
# The constructor closes the interfaces of a limited solver with these rows
# in place of `:ghost`, with a warning, unless `interface_limited_ghost`
# (`PatchInterfaces(limited_ghost = true)`) is set: on a tiled level a
# round-off perturbation of a limited `:ghost` run grows to a visible
# difference within a few hundred steps, so its solution depends on the rank
# count; under the closure rows the perturbation stays at round-off.
#
# Under `:ghost` with that setting and the artificial properties on, the
# gradient plans difference the whole flux of an interface dimension, ghost
# values included (`_remainder_ghosted`), and their rows at the interface read
# the ghosts. The
# line is open there (`_limiter_line_weights`): its weights are h up to the
# end node and h/2 there, h in the composite of the two patches, the end face
# lies at the shared node, and the face values are anchored by the interior
# face relation measured near the low end (`_limiter_open_face`), as on a
# periodic line. The end face's first-order counterpart is again the node's
# point flux, and the two patches limit it by one θ as above. Their end face
# values differ by the scheme's own mismatch at the node, the explicit row
# there in place of a compact one, which limiting moves toward zero. Where a
# part of the flux took the divergence's one-sided rows, whose face form
# differs (`_limiter_ghost`), the constructor closes the interfaces with the
# closure rows instead, with a warning.
#
# Refined levels. A refined patch closes its lines at a coarse-fine face with
# the same interface rows, or leaves them open there, so its weights there are
# a same-level patch's. The
# end node at that face is not the patch's own: the parent's shell overwrites
# it after each stage (before each stage under subcycling). The limiter leaves
# it alone as it does a Dirichlet node: the face beside it is limited from the
# interior cell alone and the end face not at all, and composite conservation
# there is the coarse-fine coupling's own. Tiles share their faces as
# same-level patches do, through the level's records and communicator, on the
# radial grids as well (above). The parent's nodes a child covers are limited
# as any other node and then take the child's coincident values from the
# injected restriction. Under
# subcycling every level limits the stages of its own steps with its own
# registers, and a refined level's filter passes, which it takes at its own
# cadence, are limited there. A tile over a partial lattice cell, at a domain
# face or at the nesting margin, can be too short for the face form: setup
# refuses such a patch, and a regrid does not create it and leaves the cell
# to the parent (`_limiter_tile_short`). A node the fill of a new patch
# leaves inadmissible takes the multilinear state of its parent nodes
# (`_fill_fallback!`), as the shell does.
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
# A cell's bound on ρe is at least this many times eps(T) |E|, between this
# many and twice as many units in the last place of its total energy E.
const LIMITER_ENERGY_ULPS = 16
# A state whose ρe lies under that floor by at most this many times eps(T) |E|
# counts as holding a bound the floor sets.
const LIMITER_ENERGY_TOLERANCE = 3
# The share of the face at r = 0 on a radial line, against the cell's outer
# face. It bounds the first-order step at the fold's node: at the spherical
# origin τ (|u| + c)/h < 1/(4 (1 + AXIS_SHARE)). The cylindrical axis face
# carries the scheme's own O(h²) flux, up to a twelfth of node 1's rate, which
# a limited axis face removes; on cylindrical Noh that happens at a handful of
# stages.
const AXIS_SHARE = 0.25

# The fewest parent nodes along a dimension `tagged_region` gives a box: four,
# or the limiter's least.
_limiter_least_extent(solver) = (pl = getfield(solver, :positivity);
                                 pl === nothing ? 4 : pl.least)

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
# folded field. A far end that is open (`:open`, see `_limiter_line_weights`)
# takes h up to it.
function _limiter_fold_weights(scheme, N::Int, h, n_halo::Int, ::Type{T};
                               hi_closures=nothing) where {T}
    hi_closures === :open && return _limiter_open_ends!(fill(Float64(h), N), h, false,
                                                        true)
    W = _limiter_weights(scheme, 2N, h, n_halo, T; lo_closures=hi_closures,
                         hi_closures)[1][N+1:2N]
    W[1:N÷2] .= h
    return W
end

# The weights of a closed line of N nodes whose ends close with `lo` and `hi`:
# a set of closure rows, `nothing` for the scheme's own, or `:open`, an end at
# a patch or level interface where the gradient plans difference the flux from
# its ghost values (`interface_flux = :ghost`). Those rows read the ghosts, so
# no node weights write the line's divergence as a difference of face fluxes
# with the point flux at that end. The line takes weight h up to the end
# instead, the interior scheme's, and h/2 at the end node, whose end face then
# lies at the node; the face values satisfy the interior face relation, which
# anchors them (`_limiter_open_face`). An end node shared with another patch
# has weight h/2 in each, h in the composite. A line closed at its other end
# takes the weights of a closed line of 2N nodes there. Returns the weights and
# the least-squares residual of the closed end's face form, zero where both
# ends are open.
function _limiter_line_weights(scheme, N::Int, h, n_halo::Int, ::Type{T}, lo,
                               hi) where {T}
    open_lo, open_hi = lo === :open, hi === :open
    (open_lo || open_hi) ||
        return _limiter_weights(scheme, N, h, n_halo, T; lo_closures=lo, hi_closures=hi)
    open_lo && open_hi && return _limiter_open_ends!(fill(Float64(h), N), h, true, true),
                                 0.0
    rows = open_lo ? hi : lo
    W, residual = _limiter_weights(scheme, 2N, h, n_halo, T; lo_closures=rows,
                                   hi_closures=rows)
    return _limiter_open_ends!(open_lo ? W[N+1:2N] : W[1:N], h, open_lo, open_hi),
           residual
end

function _limiter_open_ends!(W, h, lo::Bool, hi::Bool)
    lo && (W[1] = h / 2)
    hi && (W[end] = h / 2)
    return W
end

# The local face of a line's open low end at which the stage register's anchor
# is measured from the interior face relation of the derivative, as on a
# periodic line. The relation at face j holds at every face past the rows of
# the interface closure, which the running sum's differences reproduce, and
# reads the faces j − q to j + q, q the bands of the left-hand side: j is past
# the closure by q + 1, so that none of them is the end face at node 1.
_limiter_open_face(deriv) = length(interface_closures(deriv)) + length(_band_lhs(deriv)) + 1

# The face relation's rows reach `length(_band_lhs(deriv))` faces past the
# measured one, which the block of the rank holding the open end must hold.
_limiter_open_reach(deriv) = _limiter_open_face(deriv) + length(_band_lhs(deriv))

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
# several along a slab layout, or a single radial grid folded at r = 0
# (`_limiter_radial_line`), on the r-z plane with a symmetry plane allowed at
# the low end of z; refined levels on either under the injected restriction,
# tiled on the host backend only; without the implicit integrator, an
# ideal-gas mixture, and closed lines long enough for the face relation of
# the filter. The interfaces are open ends of the lines under `:ghost`
# (`_limiter_line_weights`), where the constructor has kept it
# (`_limiter_ghost`), and close with their own rows under `:closure`.
function _validate_positivity(bcs, metric, stretch, patch_grid, nlev, backend, eos,
                              implicit, equations, deriv, filt, filter_weighting,
                              n_global, n_halo, L_domain, ::Type{T};
                              interface_flux::Symbol=:closure,
                              interface_rhs::Symbol=:extended,
                              interface_divergence=nothing,
                              tile::NTuple{3,Int}=(0, 0, 0),
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
    nlev > 1 && level_restriction !== :inject &&
        fail("on refined levels takes level_restriction = :inject: the " *
             "filtered restriction writes values the limiter does not bound " *
             "onto the parent")
    npatch = prod(patch_grid)
    # A device level's tiles share stacked storage, whose batched right-hand
    # side has no per-tile registers.
    backend isa DeviceBackend && nlev > 1 && _tiled(tile) &&
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
    # interface rows, or an open end, at its interface ends.
    ds = npatch > 1 ? findfirst(>(1), patch_grid) : 0
    periodic = ntuple(d -> n_global[d] > 1 ? isperiodic(bcs[d][1]) : true, 3)
    regions = npatch > 1 ? patch_slabs(n_global, periodic, patch_grid) : BlockRegion[]
    rows = _limiter_end_rows(deriv, interface_flux, interface_rhs, interface_divergence)
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
        # regrid, and a tile over a partial lattice cell is shorter: the
        # constructed hierarchy is checked patch by patch, and a regrid
        # creates no patch that fails the check (`_limiter_tile_short`). A
        # regrid that subdivides the lattice stops at the edge this line
        # needs (`_limiter_least_extent`).
        if nlev > 1 && _tiled(tile)
            hf = T(L_domain[d] / (periodic[d] ? n_global[d] : n_global[d] - 1) / 3)
            _check_limiter_line(fail, deriv, filt, d, 3tile[d] + 1, hf, n_halo, T,
                                rows, rows)
        end
    end
    return nothing
end

# The fewest parent nodes a regridded box spans along a dimension under the
# limiter: its fine line of 3(n − 1) + 1 nodes, between two coarse-fine faces,
# takes the face form and the filter's face relation (`_check_limiter_line`).
function _limiter_least_box(deriv, filt, interface_flux, interface_rhs,
                            interface_divergence, n_halo::Int, ::Type{T}) where {T}
    rows = _limiter_end_rows(deriv, interface_flux, interface_rhs, interface_divergence)
    N = _limiter_least_line(deriv, filt, n_halo, T, rows, rows)
    return N == typemax(Int) ? 4 : max(4, cld(N - 1, 3) + 1)
end

# The first line of a level-ℓ patch over `region` (level ℓ − 1's node space)
# that the limiter's construction would refuse, as `(d, N, least)`: the
# dimension, the line's node count and the fewest nodes a line with the same
# ends takes; `nothing` where every line passes or the solver has no limiter.
# A line is tested as `PositivityLimiter` tests it: where an end lies at an
# interface, closed there by the interface rows or open, and at a domain face
# closed by that face's own rows. A line folded at r = 0 and a line spanning
# the domain are not tested. A partial lattice cell at a domain face or at the
# nesting margin gives a tile shorter than the whole cell `_validate_positivity`
# checks.
function _limiter_tile_short(solver, region::BlockRegion, ℓ::Int)
    pl = getfield(solver, :positivity)
    pl === nothing && return nothing
    boundary = only(_region_boundaries(solver, [region], ℓ))
    folded = only(_region_folds(solver, [boundary]))
    active = ntuple(d -> solver.n_global[d] > 1, 3)
    extent = fine_extent(region, active, folded)
    schemes = getfield(solver, :schemes)
    deriv, filt = schemes.deriv, schemes.filt
    rows = _limiter_end_rows(deriv, getfield(solver, :interface_flux),
                             schemes.interface_rhs, schemes.interface_divergence)
    n_halo = getfield(solver, :patches)[1].decomp.n_halo
    T = eltype(_root_spacing(solver))
    for d in 1:3
        (active[d] && !folded[d][1] && !all(boundary[d])) || continue
        lo_rows = boundary[d][1] ? nothing : rows
        hi_rows = boundary[d][2] ? nothing : rows
        ok = get!(pl.lines, (extent[d], !boundary[d][1], !boundary[d][2])) do
            _limiter_line_ok(deriv, filt, extent[d], n_halo, T, lo_rows, hi_rows)
        end
        ok && continue
        return (d, extent[d], _limiter_least_line(deriv, filt, n_halo, T, lo_rows, hi_rows))
    end
    return nothing
end

# Whether a regrid may create a level-ℓ tile over `region`: always without the
# limiter, and under it where every line of the tile passes
# `_limiter_tile_short`. The test reads the region and setup constants alone,
# so every rank takes the same tiles.
_limiter_tile_ok(solver, region::BlockRegion, ℓ::Int) =
    _limiter_tile_short(solver, region, ℓ) === nothing

# The setup check of a refined hierarchy under the limiter, whose patches'
# limiters are built as `run!` starts: every patch of every level passes
# `_limiter_tile_short`.
function _check_limiter_tiles(solver)
    getfield(solver, :positivity) === nothing && return nothing
    for ℓ in 1:length(getfield(solver, :levels))-1, region in level_regions(solver, ℓ)
        short = _limiter_tile_short(solver, region, ℓ)
        short === nothing && continue
        d, N, least = short
        throw(ArgumentError(
            "positivity_limiter: a refined patch is too short: the level-$ℓ patch " *
            "over $region has $N nodes along dimension $d, and a line between its " *
            "faces takes at least $least for the face form of the divergence and the " *
            "face relation of the filter; widen the region, or cover whole cells of " *
            "the tile lattice and leave a partial cell at the edge of the domain or " *
            "of the nesting margin unrefined"))
    end
    return nothing
end

# The rows that close the divergence at a patch interface end, as the patched
# solver plans them: `nothing` where they are the scheme's own.
_limiter_interface_rows(deriv, interface_rhs::Symbol, interface_divergence) =
    interface_rhs === :extended || interface_divergence !== nothing ?
    interface_divergence_rows(deriv, interface_divergence) : nothing

# A line's end at a patch or level interface as the limiter takes it: open
# under `:ghost` (`_limiter_line_weights`), closed by the divergence's
# interface rows under `:closure`.
_limiter_end_rows(deriv, interface_flux::Symbol, interface_rhs::Symbol,
                  interface_divergence) =
    interface_flux === :ghost ? :open :
    _limiter_interface_rows(deriv, interface_rhs, interface_divergence)

# Whether a limited solver may keep `interface_flux = :ghost` at its
# interfaces, under `interface_limited_ghost`: where the gradient plans
# difference the whole flux of every interface dimension, the remainder F − f
# with its ghost values included (`_remainder_ghosted`). Elsewhere a part of the
# flux takes the divergence's one-sided rows, whose face form differs, and the
# constructor closes the interfaces with the closure rows.
_limiter_ghost(art) = _ghost_remainder(:ghost, art)

# A closed line of N nodes whose ends close with `lo` and `hi` (`nothing` for
# the scheme's own rows, `:open` at an open end): the face form exists, its
# weights are positive, and the line holds an interior face for the filter's
# face relation.
function _check_limiter_line(fail, deriv, filt, d, N, h, n_halo, ::Type{T}, lo,
                             hi) where {T}
    _limiter_line_ok(deriv, filt, N, n_halo, T, lo, hi) && return nothing
    least = _limiter_least_line(deriv, filt, n_halo, T, lo, hi)
    N < least < typemax(Int) &&
        fail("a line along dimension $d has $N nodes; the face relation of the " *
             "filter needs at least $least on a closed line")
    W, res = _limiter_line_weights(deriv, max(N, 16), h, n_halo, T, lo, hi)
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
    lo === :open && N < _limiter_open_reach(deriv) && return false
    W, res = _limiter_line_weights(deriv, N, one(T), n_halo, T, lo, hi)
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

_row_count(lo, hi) = max(_row_count(lo), _row_count(hi))
_row_count(::Nothing) = 0
_row_count(::Symbol) = 0
_row_count(rows) = length(rows)

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
# periodic or open-end anchor and the filter relation are measured. A patch
# interface is a closed end of the patch's lines, open under `:ghost` and
# weighted by the interface rows under `:closure` (`_limiter_line_weights`).
# A tile of a radial grid (`separate`) holds the radial pass's rates, face
# speeds and pressure face values: the z faces of every tile read the rates
# after the radial passes of all of them, and a radial correction waits for the
# θ exchange.
function PositivityLimiter(solver; separate::Bool=false)
    decomp = solver.decomp
    T = eltype(solver.h)
    n_cons = solver.equations.n_cons
    # A refined patch's inactive fold holds the plans along its dimension.
    plan_scheme(d) = (fold = _fold_at(solver, d);
                      p = fold === nothing ? _plan_at(solver.plans.div_plans, d) :
                          fold_vplan(fold, 1);
                      (p isa DevicePlan ? p.host : p).scheme)
    schemes = solver.schemes
    deriv = schemes.deriv
    filt = schemes.filt
    irows = _limiter_end_rows(deriv, solver.interface_flux, schemes.interface_rhs,
                              schemes.interface_divergence)
    # A same-level interface and a refined patch's coarse-fine face both close
    # the divergence with the interface rows, or both are open.
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
    # An open end takes the face form of the gradient plans, which difference
    # the whole flux there (`_limiter_ghost`).
    open = ntuple(d -> decomp.active[d] ?
                       (end_rows(d, 1) === :open, end_rows(d, 2) === :open) :
                       (false, false), 3)
    for d in 1:3
        any(open[d]) && !_remainder_ghosted(solver, d) &&
            error("positivity limiter: an interface of dimension $d under :ghost " *
                  "takes a part of its flux through the divergence plans")
        open[d][1] && decomp.offset[d] == 0 &&
            decomp.n_local[d] < _limiter_open_reach(deriv) &&
            fail("the block holding the low interface end along dimension $d has " *
                 "$(decomp.n_local[d]) nodes, fewer than the " *
                 "$(_limiter_open_reach(deriv)) the anchor of its face values reads")
    end
    # The radial lines: the root's, folded at r = 0, and a refined patch's,
    # folded there where the patch reaches the axis or the origin and closed
    # at a coarse-fine face otherwise.
    radial = (curved && decomp.active[1], false, false)
    areal = ntuple(d -> decomp.active[d] && curved && !radial[d], 3)
    # The weights along each global line, and the nodes from a closed end
    # beyond which the filter's face relation is interior. A folded line's
    # far end closes with the derivative's rows, or at a coarse-fine face
    # with the interface rows or open.
    global_weights = ntuple(3) do d
        decomp.active[d] || return Float64[1.0]
        N = decomp.n_global[d]
        decomp.periodic[d] && return fill(Float64(solver.h[d]), N)
        fold_lo[d] && return _limiter_fold_weights(deriv, N, solver.h[d],
                                                   decomp.n_halo, T;
                                                   hi_closures=end_rows(d, 2))
        return _limiter_line_weights(plan_scheme(d), N, solver.h[d], decomp.n_halo, T,
                                     end_rows(d, 1), end_rows(d, 2))[1]
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
    send = [zeros(T, 4 * length(planes[d]) + 1) for d in 1:3]
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
    anchor_face = ntuple(d -> !decomp.active[d] ? 0 :
                              open[d][1] ? _limiter_open_face(deriv) :
                              clamp(decomp.n_local[d] ÷ 2, qd, decomp.n_local[d] - qd), 3)
    ws = _limiter_flux_rows(solver, decomp.active, radial)
    # The ends at a same-level interface, whose node another patch holds too.
    # A coarse-fine end node is the parent's, not shared, and its face is
    # left alone as a Dirichlet node's is.
    shared(bc) = bc isa InterfaceBC && !parent_fed(bc)
    interface = ntuple(d -> decomp.active[d] ?
                            (shared(solver.bcs[d][1]), shared(solver.bcs[d][2])) :
                            (false, false), 3)
    # The filter's conservative rows close the faces the patch shares with a
    # patch of its own level, and a filter pass holds the end node there
    # (`_filter_ends`).
    conservative = interface
    # The patches of a rank share the workspace by extent, and a stage takes
    # every patch's θ before any patch's correction (`_limit_level_stage!`),
    # so a patch with a same-level interface, and a tile of a radial grid,
    # holds its face values and θ itself.
    separate = separate && any(radial)
    own = any(any, interface) || separate
    faces = own ? [zero(solver.tmp_a) for c in 1:n_cons] : [ws[1, c] for c in 1:n_cons]
    theta = own ? zero(solver.tmp_a) : solver.tmp_b
    any(radial) && push!(faces, separate ? zero(solver.tmp_a) : ws[3, 1])
    rates = separate ? zero(solver.tmp_a) : ws[2, 1]
    speeds = separate ? zero(solver.tmp_a) : ws[2, 2]
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
              solver.inv_J, solver.inv_h..., faces..., rates, speeds)
    all(f -> size(f) == shape, fields) ||
        error("positivity limiter: the solver's fields do not share one padded extent")
    # The vectors the bodies read, formed on the host and uploaded once.
    up(x) = _on_storage(like, x)
    weights_s = map(up, weights)
    free_s = map(up, free)
    line_counts = [_limiter_zeros(like, Int, (4, _transverse(decomp.n_local, d)...))
                   for d in 1:3]
    A = typeof(solver.tmp_a)
    weight_set = LimiterWeights{T,A,typeof(weights_s[1]),typeof(free_s[1])}(
        weights_s, map(w -> up(one(T) ./ w), weights), (free_s..., _limiter_exempt(solver)),
        map(up, areas), up(volume), up(T.(_band_lhs(deriv))), up(T.(deriv.coeffs)),
        up(T.(filter_lhs)), up(_filter_face_stencil(filt, T)))
    line_buffers = LimiterLineBuffers{T,typeof(planes[1]),eltype(line_counts)}(
        copies(), copies(), copies(), copies(), copies(), copies(), copies(), copies(),
        copies(), copies(), line_counts, NTuple{3,Array{T,3}}[], send, recv)
    scratch = LimiterScratch{A}(FieldVector(faces),
                                FieldVector([ws[2, c] for c in 1:n_cons]), rates,
                                speeds, theta, _scratch_lease(solver))
    return PositivityLimiter{T,A,typeof(weights_s[1]),typeof(free_s[1]),
                             typeof(planes[1]),eltype(line_counts)}(
        weight_set, line_buffers, scratch, LimiterCounters(zeros(Int, 6)),
        registers, register_fields, radial, fold_lo, n_comp, volume_state, areal,
        interface, open, conservative, decomp.periodic, measure, anchor_face,
        zero(T), zero(T), false)
end

# The workspace's flux rows the limiter borrows as scratch: the face values in
# row 1 (and the pressure's in [3, 1] on a radial line), the filter's pre-pass
# state, the rates and the speeds in row 2. A collapsed dimension's row holds
# one shared array in every slot (`_rhs_workspace`), so the slots of such a row
# that the limiter reads are replaced by arrays of its own.
function _limiter_flux_rows(solver, active::NTuple{3,Bool}, radial::NTuple{3,Bool})
    ws = copy(solver.flux)
    for d in 1:3
        active[d] && continue
        slots = d < 3 ? axes(ws, 2) : (any(radial) ? (1:1) : (1:0))
        for c in slots
            ws[d, c] = zero(solver.tmp_a)
        end
    end
    return ws
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
