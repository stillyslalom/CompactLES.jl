# Static two-level refinement at a global timestep (Stage 3 of
# reference/AMR_GPU.md).
#
# One user-specified region of the root grid is covered by a single level-1
# patch at refinement ratio 3, node-centered: a region of m coarse nodes per
# refined dimension carries 3m − 2 fine nodes, coarse node a + k − 1 coinciding
# with fine node 3k − 2. By default both levels advance every RK stage with
# the same dt (the global minimum, which the shared `max_rate` reduction
# already supplies once the fine patch joins `solver.patches`), so no temporal
# interpolation arises anywhere. Under `subcycle = true` (Stage 4) the fine
# level instead takes three steps of dt/3 per coarse step, and the temporal
# interpolation this needs is the Hermite box at the end of this file; the
# region can also move under `regrid_interval` (src/regrid.jl).
#
# The two levels are coupled on two schedules, both by default through the
# POINT-SAMPLE halves of the Stage 1 transfer machinery:
#
#   - After every RK stage update, `prolong_level_ghosts!` interpolates the
#     coarse state (order 6) over a box extending `LEVEL_BUFFER` coarse nodes
#     beyond the refined region and overwrites the fine patch's ghost ring AND
#     its boundary-plane nodes from the result. The plane nodes are included
#     deliberately: the fine solve is then boundary-forced by the coarse
#     solution exactly as a `DirichletBC` face would be, which removes the
#     drift mode between the levels without an averaging step.
#
#   - After every completed step (post-filter), `restrict_level!` copies the
#     fine field's coincident-node values onto the covered coarse region,
#     holding `RESTRICT_MARGIN` coarse nodes back from the boundary (see that
#     constant for the measured amplifying loop the margin breaks).
#
# The invertible filter pair itself (`prolong!`/`restrict!`, deconvolution
# against Gaussian filtering) is NOT the default coupling, on a Stage 3
# measurement: the pair's contract is that prolongation input is samples of
# the FILTERED field, while the live coarse solution is point samples of the
# field itself. Deconvolving point samples "sharpens" data that was never
# smoothed and filtering on the way down attenuates resolved content, both
# O(h²) against the solution — the entropy-wave gate measured order 1.3–1.7
# and errors three decades above the interpolation/injection coupling's,
# which measures order ≈ 3.5 (the one-sided divergence closures binding, as
# at a Stage 2 interface). `level_restriction = :filter` keeps the filtered
# path selectable — its anti-alias smoothing is the tool to reach for if
# injection restriction proves positivity-limited on captured shocks — and
# the full pair remains what Stage 4 regridding will initialize NEW fine
# regions from, where the coarse data is all there is.
#
# Multi-dimensional transfer is a tensor product realized as a chain: each
# active dimension is refined in turn, so a chain of K = (number of active
# dimensions) `TransferPlan`s connects the coarse box to the fully fine box
# through K − 1 intermediate grids, each stage carrying its own scratch array.
#
# Scope of this first cut, enforced by `Solver`: serial only (the coarse
# sub-box copy and the restriction write-back are direct array copies; the
# gather/scatter that a decomposed level needs is the follow-up that also
# lifts `plan_transfer`'s alignment restriction), Cartesian metric, no
# stretching, no folds, tridiagonal schemes, the `:delta4` detector, and no
# same-level patch decomposition alongside refinement. The fine patch's line
# solves close at the coarse–fine boundary with the Stage 2 interface rows
# (extended-data gradients and filters, one-sided divergence).

"Coarse nodes of prolongation buffer beyond the refined region, per side."
const LEVEL_BUFFER = 4

# Coarse nodes per side of the covered region excluded from the restriction
# write-back. Restricting all the way to the coarse-fine boundary closes an
# amplifying loop: the fine solution is least accurate at its imposed
# boundary, the restriction segment's one-sided closure rows read exactly
# those values, and the polluted coarse boundary nodes feed the next fine
# shell — measured gain ≈ 2 per step on the entropy-wave test, against a flat
# error with the write-back held off the boundary. Two coarse nodes clear
# both the closure rows' footprint and the imposed plane's neighborhood.
const RESTRICT_MARGIN = 2

"""
    LevelTransfer

Bound form of the two-level coupling: the refined region, the prolongation
chain over the buffered box, and the restriction chain over the fine patch's
extent, each a sequence of [`TransferPlan`](@ref)s refining one active
dimension at a time with per-stage scratch. Constructed at setup by the
[`Solver`](@ref) constructor's `refine` keyword; consumed by
`prolong_level_ghosts!` and `restrict_level!`.
"""
struct LevelTransfer{T}
    region::BlockRegion              # refined region, root (coarse) node space
    fine_index::Int                  # index of the level-1 patch in solver.patches
    restriction::Symbol              # :inject (coincident-node copy, default)
                                     # or :filter (the invertible Stage 1 pair)
    active_dims::Vector{Int}
    pdecomps::Vector{Decomp{T}}      # prolongation chain, stage 0 (coarse box) .. K
    pplans::Vector{TransferPlan{T}}
    pstage::Vector{Array{T,3}}
    rdecomps::Vector{Decomp{T}}      # restriction chain, stage 0 (region) .. K
    rplans::Vector{TransferPlan{T}}
    rstage::Vector{Array{T,3}}       # scratch for stages 0 .. K-1
    # Subcycling storage (Stage 4): the coarse solution on the buffered box at
    # the two ends of the current coarse step, values and RHS rates, per
    # conserved component — the data of the cubic Hermite interpolant that
    # supplies the fine shell at fine stage times. Shaped like pstage[1] with a
    # trailing component index; empty when the solver does not subcycle.
    box_Q0::Array{T,4}               # coarse box state at t^n
    box_dQ0::Array{T,4}              # coarse box RHS at t^n
    box_Q1::Array{T,4}               # coarse box state at t^n + dt
    box_dQ1::Array{T,4}              # coarse box RHS at t^n + dt
end

# Decomp chain refining `dims_to_refine` one at a time, starting from the
# coarse extents. Every stage is serial (this file's scope) and non-periodic
# along active dimensions; collapsed dimensions stay collapsed.
function _refine_chain(::Type{T}, ext0::NTuple{3,Int}, active::NTuple{3,Bool},
                       dims_to_refine::Vector{Int}, n_halo::Int,
                       interp_order::Int) where {T}
    pper = ntuple(d -> !active[d], 3)
    exts = [ext0]
    for dk in dims_to_refine
        prev = exts[end]
        push!(exts, ntuple(d -> d == dk ? 3 * prev[d] - 2 : prev[d], 3))
    end
    decomps = [Decomp{T}(e, pper; dims=(1, 1, 1), n_halo=n_halo) for e in exts]
    plans = [plan_transfer(decomps[k+1], decomps[k], dims_to_refine[k], T;
                           interp_order=interp_order)
             for k in eachindex(dims_to_refine)]
    stages = [zeros(T, ntuple(d -> e[d] + 2 * (active[d] ? n_halo : 0), 3))
              for e in exts]
    return decomps, plans, stages
end

function build_level_transfer(::Type{T}, region::BlockRegion,
                              active::NTuple{3,Bool}, n_halo::Int,
                              fine_index::Int, restriction::Symbol,
                              n_cons::Int=0, subcycle::Bool=false) where {T}
    dims_to_refine = [d for d in 1:3 if active[d]]
    boxext = ntuple(d -> active[d] ? region.extent[d] + 2 * LEVEL_BUFFER :
                                     region.extent[d], 3)
    pdecomps, pplans, pstage = _refine_chain(T, boxext, active, dims_to_refine,
                                             n_halo, 6)
    # The restriction chain never interpolates (that half of each TransferPlan
    # goes unused), so it is built at interpolation order 2, which admits the
    # smallest legal regions.
    rdecomps, rplans, rstage = _refine_chain(T, region.extent, active,
                                             dims_to_refine, n_halo, 2)
    boxsize = subcycle ? (size(pstage[1])..., n_cons) : (0, 0, 0, 0)
    return LevelTransfer{T}(region, fine_index, restriction, dims_to_refine,
                            pdecomps, pplans, pstage,
                            rdecomps, rplans, rstage,
                            zeros(T, boxsize), zeros(T, boxsize),
                            zeros(T, boxsize), zeros(T, boxsize))
end

"Fine extent of a refined region: 3m − 2 nodes per active dimension."
fine_extent(region::BlockRegion, active::NTuple{3,Bool}) =
    ntuple(d -> active[d] ? 3 * region.extent[d] - 2 : region.extent[d], 3)

# --- Prolongation: coarse state → fine ghost ring and boundary planes -------

function _copy_coarse_box!(dst, coarse_Q, c::Int, lt::LevelTransfer,
                           dp::Decomp, box::Decomp)
    # Coarse box node n ↔ root node region.offset − LEVEL_BUFFER + n (active
    # dims). The box lies strictly inside the root patch by the nesting
    # margin, so every read is an interior value.
    padc = dp.n_halo_d
    padb = box.n_halo_d
    off = ntuple(d -> lt.region.offset[d] - (dp.active[d] ? LEVEL_BUFFER : 0), 3)
    nb = box.n_local
    @inbounds for k in 1:nb[3], j in 1:nb[2], i in 1:nb[1]
        dst[i + padb[1], j + padb[2], k + padb[3]] =
            coarse_Q[off[1] + i + padc[1], off[2] + j + padc[2],
                     off[3] + k + padc[3], c]
    end
    return dst
end

function _write_fine_shell!(fine_Q, c::Int, box_field, lt::LevelTransfer,
                            df::Decomp, boxf::Decomp, shell_only::Bool=true)
    # Fine patch node i ↔ fine box node i + 3·LEVEL_BUFFER (active dims). The
    # shell is every padded slot outside the strict interior [2, n−1] of each
    # active dimension: the ghost ring plus the boundary planes, both imposed
    # from the prolonged coarse state. `shell_only = false` writes every slot
    # instead — the whole-patch initialization a regrid performs on a freshly
    # created fine region.
    padf = df.n_halo_d
    padb = boxf.n_halo_d
    nf = df.n_local
    shift = ntuple(d -> df.active[d] ? 3 * LEVEL_BUFFER : 0, 3)
    r = ntuple(d -> (1 - padf[d]):(nf[d] + padf[d]), 3)
    @inbounds for k in r[3], j in r[2], i in r[1]
        interior = (!df.active[1] || 2 <= i <= nf[1] - 1) &&
                   (!df.active[2] || 2 <= j <= nf[2] - 1) &&
                   (!df.active[3] || 2 <= k <= nf[3] - 1)
        shell_only && interior && continue
        fine_Q[i + padf[1], j + padf[2], k + padf[3], c] =
            box_field[i + shift[1] + padb[1], j + shift[2] + padb[2],
                      k + shift[3] + padb[3]]
    end
    return fine_Q
end

"""
    prolong_level_ghosts!(solver, states)

Impose the fine patch's ghost ring and boundary-plane nodes from the order-6
interpolation of the current coarse state over the buffered box, per conserved
component, and return `states`. Runs after every RK stage update and inside
the pre-step synchronization; a solver without refinement returns immediately.
Serial by this stage's scope, so nothing here communicates. The name says
"prolong" for the operation's role; the operator is `interpolate!`, per the
header note on why the deconvolving `prolong!` is not used here.
"""
function prolong_level_ghosts!(solver, states)
    lt = solver.level_transfer
    lt === nothing && return states
    patches = getfield(solver, :patches)
    coarse = patches[1]
    fine = patches[lt.fine_index]
    Qc = states[1]
    Qf = states[lt.fine_index]
    K = length(lt.pplans)
    for c in 1:solver.equations.n_cons
        _copy_coarse_box!(lt.pstage[1], Qc, c, lt, coarse.decomp, lt.pdecomps[1])
        for k in 1:K
            # Interpolation, not deconvolution: the coarse solution is point
            # samples, and `prolong!`'s deconvolution is exact only on data a
            # `restrict!` produced. See the `interpolate!` docstring; the full
            # pair remains the tool for initializing a NEW fine region from
            # restricted coarse data (Stage 4 regridding).
            interpolate!(lt.pstage[k+1], lt.pplans[k], lt.pstage[k])
        end
        _write_fine_shell!(Qf, c, lt.pstage[K+1], lt, fine.decomp,
                           lt.pdecomps[K+1])
    end
    # The imposed shell replaces the halo values the previous exchange left,
    # so the fine state is self-consistent without a further exchange: the
    # patch is serial and its ghost ring is exactly the shell just written.
    return states
end

# --- Restriction: fine state → covered coarse region ------------------------

function _write_covered_region!(coarse_Q, c::Int, restricted, lt::LevelTransfer,
                                dp::Decomp, box::Decomp)
    padc = dp.n_halo_d
    padb = box.n_halo_d
    off = lt.region.offset
    nb = box.n_local
    r = ntuple(d -> dp.active[d] ?
               ((1 + RESTRICT_MARGIN):(nb[d] - RESTRICT_MARGIN)) : (1:nb[d]), 3)
    @inbounds for k in r[3], j in r[2], i in r[1]
        coarse_Q[off[1] + i + padc[1], off[2] + j + padc[2],
                 off[3] + k + padc[3], c] =
            restricted[i + padb[1], j + padb[2], k + padb[3]]
    end
    return coarse_Q
end

"""
    restrict_level!(solver, states)

Restrict the fine state onto the covered coarse region, per conserved
component, and return `states`. Under the default `:inject` mode this copies
the coincident-node values; under `:filter` it applies the invertible pair's
Gaussian filter over the fine patch's extent (tabulated one-sided closure
rows) before subsampling. Either way the write-back stops `RESTRICT_MARGIN`
coarse nodes short of the coarse-fine boundary. Runs once per completed step,
after the state filter; a solver without refinement returns immediately.
"""
function restrict_level!(solver, states)
    lt = solver.level_transfer
    lt === nothing && return states
    patches = getfield(solver, :patches)
    coarse = patches[1]
    Qc = states[1]
    Qf = states[lt.fine_index]
    K = length(lt.rplans)
    for c in 1:solver.equations.n_cons
        src = view(Qf, :, :, :, c)
        for k in K:-1:1
            dst = lt.rstage[k]
            input = k == K ? src : lt.rstage[k+1]
            if lt.restriction === :filter
                restrict!(dst, lt.rplans[k], input)
            else
                inject!(dst, lt.rplans[k], input)
            end
        end
        _write_covered_region!(Qc, c, lt.rstage[1], lt, coarse.decomp,
                               lt.rdecomps[1])
    end
    return states
end

"""
    sync_levels!(solver, states)

Bring the two levels to mutual consistency: restrict the fine state onto the
covered coarse region, then re-impose the fine shell from the (updated) coarse
state. This is the pre-step form; within a step only the prolongation half
runs, since restriction is a per-step operation by the plan's schedule.
"""
function sync_levels!(solver, states)
    restrict_level!(solver, states)
    prolong_level_ghosts!(solver, states)
    return states
end

# --- Subcycling support (Stage 4): the Hermite box -------------------------
#
# Under subcycling (three fine steps of dt/3 per coarse step, Berger–Oliger
# order: coarse first, fine after), the fine shell needs coarse values at fine
# stage times BETWEEN t^n and t^{n+1}. The coarse solution over the step is
# reconstructed on the buffered box by cubic Hermite interpolation from its
# endpoint values and endpoint RHS rates, O(dt⁴), matching the integrator's
# order; LSRK54 has no free dense output and this is the standard substitute.
# The t^n data falls out of the coarse step's first stage; the t^{n+1} data
# costs one extra coarse RHS evaluation per step, taken before the fine
# subcycles so it samples the coarse trajectory rather than the restricted
# composite (the restriction write-back would perturb the box values read
# here).

"""
    save_level_box!(lt, decomp, Q, dQ, at_end)

Copy the coarse state `Q` and its RHS `dQ` over the buffered prolongation box
into the [`LevelTransfer`](@ref)'s Hermite storage — the `t^n` slots when
`at_end` is false, the `t^n + dt` slots when true. `decomp` is the coarse
patch's decomposition. Serial, per the level-transfer scope.
"""
function save_level_box!(lt::LevelTransfer, decomp::Decomp, Q, dQ, at_end::Bool)
    boxQ = at_end ? lt.box_Q1 : lt.box_Q0
    boxdQ = at_end ? lt.box_dQ1 : lt.box_dQ0
    box = lt.pdecomps[1]
    for c in 1:size(boxQ, 4)
        _copy_coarse_box!(view(boxQ, :, :, :, c), Q, c, lt, decomp, box)
        _copy_coarse_box!(view(boxdQ, :, :, :, c), dQ, c, lt, decomp, box)
    end
    return lt
end

# Cubic Hermite blend of the stored box data at fraction θ ∈ [0, 1] of the
# coarse step, written into `dst` (the chain's stage-0 scratch) for component
# `c`. `dt` is the coarse step, which scales the stored rates.
function _hermite_box!(dst, lt::LevelTransfer, c::Int, θ, dt)
    θT = eltype(dst)(θ)
    dtT = eltype(dst)(dt)
    oneT = one(θT)
    h00 = (oneT + eltype(dst)(2) * θT) * (oneT - θT)^2
    h10 = θT * (oneT - θT)^2
    h01 = θT^2 * (eltype(dst)(3) - eltype(dst)(2) * θT)
    h11 = θT^2 * (θT - oneT)
    box = lt.pdecomps[1]
    pad = box.n_halo_d
    nb = box.n_local
    Q0, dQ0 = lt.box_Q0, lt.box_dQ0
    Q1, dQ1 = lt.box_Q1, lt.box_dQ1
    @inbounds for k in 1:nb[3], j in 1:nb[2], i in 1:nb[1]
        I = CartesianIndex(i + pad[1], j + pad[2], k + pad[3])
        dst[I] = h00 * Q0[I, c] + h01 * Q1[I, c] +
                 dtT * (h10 * dQ0[I, c] + h11 * dQ1[I, c])
    end
    return dst
end

"""
    RegridSpec

Configuration and rebuild inputs for tagging-driven regridding (Stage 4,
`src/regrid.jl`): the regrid cadence in coarse steps, the tagging threshold on
the relative undivided fourth difference of the mixture density, the buffer of
coarse cells added around tagged cells, the nesting margin, and everything a
fine-patch rebuild needs that the `Solver` does not itself retain — the
schemes, halo width, interface treatment, and backend. `last_step` records the
step of the most recent regrid check so a run resumed on the same solver keeps
the cadence. Constructed by the [`Solver`](@ref) constructor's
`regrid_interval` keyword; consumed by `regrid!`.
"""
mutable struct RegridSpec{T}
    interval::Int
    threshold::T
    buffer::Int
    margin::Int
    n_halo::Int
    interface_rhs::Symbol
    deriv::CompactScheme{T}
    filt::CompactScheme{T}
    smoo::CompactScheme{T}
    backend::AbstractBackend
    last_step::Int
end

"""
    hermite_level_shell!(solver, states, θ, dt)

Impose the fine patch's shell (ghost ring plus boundary planes) from the cubic
Hermite reconstruction of the coarse solution at fraction `θ` of the coarse
step of size `dt`, through the same order-6 interpolation chain
[`prolong_level_ghosts!`](@ref) uses. Requires both endpoint slots filled by
[`save_level_box!`](@ref); at `θ = 0` the result is exactly the `t^n` coarse
state and the imposition reduces to the unsubcycled one.
"""
function hermite_level_shell!(solver, states, θ, dt)
    lt = solver.level_transfer
    patches = getfield(solver, :patches)
    fine = patches[lt.fine_index]
    Qf = states[lt.fine_index]
    K = length(lt.pplans)
    for c in 1:solver.equations.n_cons
        _hermite_box!(lt.pstage[1], lt, c, θ, dt)
        for k in 1:K
            interpolate!(lt.pstage[k+1], lt.pplans[k], lt.pstage[k])
        end
        _write_fine_shell!(Qf, c, lt.pstage[K+1], lt, fine.decomp,
                           lt.pdecomps[K+1])
    end
    return states
end
