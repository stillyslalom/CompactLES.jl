# The `Solver` container and its accessors: the run configuration and clock
# over a vector of patches (patches.jl), the forwarding of patch-owned
# property names to the sole patch, the index and coordinate maps, the
# layout-independent readers of the conserved state, and `allocate_state`.
# Construction is in construction.jl and the right-hand side in rhs.jl.

# The numerical choices a solver is built with that its plans and levels do
# not keep in a comparable form: a plan holds its scheme only where the
# dimension is active and unfolded, a static level keeps no scheme at all, and
# `interface_rhs`, the interpolation order and the restriction live only in
# the transfers and the `RegridSpec`. The checkpoint's configuration record
# (io.jl) reads them from here, and `run!` reads the restriction once per call.
# Never read on the step path.
struct SchemeSettings
    deriv::AbstractCompactScheme
    filt::AbstractCompactScheme
    interface_divergence::Union{Nothing,AbstractCompactScheme}
    interface_rhs::Symbol
    level_interpolation_order::Int
    level_restriction::Symbol
end

# The patch parameter `P` is unconstrained, not `P <: Patch{T}`: a refined
# solver stores its root and level-1 patches in one vector, and where the two
# differ in a `Patch` parameter `P` is their typejoin. The per-patch loops
# then cost one dynamic dispatch per patch per call, behind the PatchSolver
# function barrier so the bodies stay concrete, while the single-patch and
# same-level multi-patch cases keep a concrete `P`. Boundary conditions are
# not such a difference: they are stored abstractly on the `Patch`
# (patches.jl), so a level-1 patch under an interface often shares its
# parent's type outright.
mutable struct Solver{T,Eq<:EquationSet,E<:EOS,Tr<:AbstractTransport{T},M<:Metric,St,Src,P}
    equations::Eq
    eos::E
    transport::Tr
    art::ArtParams{T}
    metric::M
    stretch::St
    sources::Src
    L_domain::NTuple{3,T}
    origin::NTuple{3,T}
    coord_shift::NTuple{3,T}                # half-cell offset (axis grids)
    h::NTuple{3,T}
    cfl::T
    filter_interval::Int
    filter_cfl::T                           # 0 = unrelaxed; see filter_weight
    filter_weighting::Symbol                # :none or :volume; see filter_state!
    control::StepControl                    # timestep floors, prediction, retry
    n_global::NTuple{3,Int}                 # whole-grid extents (all patches)
    # Per-patch state: everything from the decomposition and the operator plans
    # through the field arrays lives on `Patch` (patches.jl). With one patch,
    # `Base.getproperty` below forwards the patch-owned names to it, so
    # `solver.rho`, `solver.decomp` and the rest read as they always did.
    patches::Vector{P}                      # this rank's patches: the root, then
                                            # its tiles of each level in that
                                            # level's order (Level.tiles)
    patch_regions::Vector{BlockRegion}      # every patch, id order, global nodes
    comm::MPI.Comm                          # reduction communicator spanning all
                                            # ranks (the sole patch's Cartesian
                                            # communicator when there is one patch)
    ghost_sends::Vector{GhostRecord{T}}     # interface exchange records; empty
    ghost_recvs::Vector{GhostRecord{T}}     # with one patch
    plane_pairs::Vector{PlaneRecord{T}}
    levels::Vector{Level{T}}                # the refinement hierarchy, root
                                            # first; one entry without refinement
                                            # (levels.jl)
    subcycle::Bool                          # Berger–Oliger subcycling; timestep.jl
    regrid::Union{Nothing,RegridSpec{T}}    # tagging + regridding; regrid.jl
    t::T
    tstage::T
    step::Int
    dt_prev::T                              # last accepted step, for growth capping
    rate_prev::T                            # last CFL rate, for the predictor
    filter_rate_prev::NTuple{3,T}           # its per-direction hyperbolic rates,
                                            # for filter_weight
    # Wall-clock accounting, filled in by `run!`. Rank-local: a reduction here
    # would be a collective on every step, paid by every run whether or not
    # anything reads it. Callers wanting load imbalance reduce these
    # themselves at their own (much lower) reporting frequency; see
    # `ProgressLog`. Seconds, and Float64 regardless of T.
    wall_step::Float64                      # last completed step, excl. callbacks
    wall_total::Float64                     # cumulative over the run
    # Of those, the seconds spent inside the collectives every rank of a run
    # enters (the rate and floor reductions, the level box and restriction
    # gathers) and in a refined level's record exchange, where a lightly
    # loaded rank blocks on a heavier one. `wall_step - wall_wait` is the
    # rank's own work, the quantity a rebalance compares across ranks
    # (`_rebalance_due!`); the step wall alone is nearly uniform under any
    # load, since every rank waits at the same reductions.
    wall_wait::Float64                      # in the last completed step
    wait_total::Float64                     # cumulative over the run
    # What the positivity failsafe has seen and repaired. Global, unlike the
    # rank-local wall-clock fields above: the repair reduces its own
    # tally on every step it runs, and it runs only when
    # `StepControl.floor_ratio` is set, so a run that leaves it off pays nothing.
    floor_tally::FloorTally
    # How the flux divergence treats an interface end: `:closure` closes it
    # with one-sided rows (`div_plans`); `:ghost` differences the inviscid
    # flux through the end with the gradient plans' interface rows, reading
    # inviscid fluxes evaluated on the exchanged or imposed ghost state, and
    # the remainder of the flux through `div_plans`.
    interface_flux::Symbol
    schemes::SchemeSettings                 # construction record; see above
end

# Patch-owned property names forward to the sole patch, which keeps every
# existing consumer of the pre-patch field layout (tests, benches, examples,
# and the compute path itself in the single-patch case) reading unchanged.
# On a multi-patch solver those names have no single answer, so the forward
# throws; the multi-patch drivers hand each routine a `PatchSolver` instead.
@inline function Base.getproperty(s::Solver, name::Symbol)
    if _is_patch_prop(name)
        ps = getfield(s, :patches)
        length(ps) == 1 || _patch_prop_error(name)
        return _patch_property(@inbounds(ps[1]), name)
    end
    return getfield(s, name)
end

@noinline function _patch_prop_error(name::Symbol)
    diagnostics = "Pass the state vector to a diagnostic that takes it " *
                  "(line_profile, line_sample, field_slice, field_array, " *
                  "mix_width, save_vtk)"
    _is_workspace_prop(name) &&
        error("property `$name` is right-hand-side scratch and this solver holds " *
              "several patches (a refined or patched run), which share it by " *
              "extent, so it holds values from the patch evaluated last. " *
              diagnostics * ", which recompute it for each patch")
    error("property `$name` is per-patch state and this solver holds several " *
          "patches (a refined or patched run). " * diagnostics * ", or iterate " *
          "`for (ps, Q) in eachpatch(solver, states)`, where `ps.$name` is one " *
          "patch's array")
end

# The sole patch of a single-patch solver, for the scratch-writer records
# (patches.jl); the compute path never reaches this with several patches.
@inline _patch_of(s::Solver) = @inbounds getfield(s, :patches)[1]

"""
Union of the two objects the compute path accepts: a (single-patch) `Solver`,
whose property forwarding exposes its sole patch, and a [`PatchSolver`](@ref)
binding one patch of a multi-patch solver. Every routine between the
step drivers and the arrays is written against this type.
"""
const SolverLike{T} = Union{Solver{T},PatchSolver{T}}

"Number of patches this rank's solver holds."
npatches(s::Solver) = length(getfield(s, :patches))

"Iterate `(patch_solver, state)` pairs over this rank's patches, in the global
patch order, with `states` aligned with `solver.patches`."
eachpatch(solver::Solver, states) =
    ((PatchSolver(solver, p), states[i]) for (i, p) in
     enumerate(getfield(solver, :patches)))

"""
    xcoord(solver, d, i)

Physical coordinate of rank-local, one-based interior index `i` in direction
`d`. The index does not include halo padding. This is equivalent to
`global_xcoord(solver, d, solver.region.offset[d] + solver.decomp.offset[d] + i)`,
the patch region offset placing a patch's block in the whole grid (zero for a
single-patch solver).
"""
xcoord(solver::SolverLike, d::Int, i::Int) =
    global_xcoord(solver, d,
                  solver.region.offset[d] + solver.decomp.offset[d] + i)

"""
    global_xcoord(solver, d, g)

Physical coordinate of global, one-based index `g` in direction `d`, including
any half-cell fold offset and [`Stretch`](@ref) mapping. `g` indexes the whole
grid, not a patch. Every rank returns the same value for the same `(d, g)`.
Use this form when assembling a global coordinate vector and [`xcoord`](@ref)
for a local interior index.
"""
function global_xcoord(solver::SolverLike, d::Int, g::Int)
    ξ = solver.origin[d] + solver.coord_shift[d] + (g - 1) * solver.h[d]
    st = solver.stretch[d]
    return st === nothing ? ξ : st.x(ξ)
end

"""
    gidx(solver, i, j, k) -> CartesianIndex

Convert rank-local, one-based interior indices to a `CartesianIndex` for the
halo-padded state and solver fields. Collapsed directions have zero padding.
The result is rank-local; it does not locate a global point owned by another
rank.
"""
function gidx(solver::SolverLike, i::Int, j::Int, k::Int)
    pad = solver.decomp.n_halo_d
    return CartesianIndex(i + pad[1], j + pad[2], k + pad[3])
end

"""
    interior_index(solver, I) -> (i, j, k)

Rank-local, one-based interior indices of the padded `CartesianIndex` `I`; the
inverse of [`gidx`](@ref). Use it wherever a padded index has to be handed to
something that takes interior ones. [`boundary_plane`](@ref) yields padded
indices, while [`xcoord`](@ref) expects an interior index:

```julia
for I in boundary_plane(solver, 1, 1)
    i, j, k = interior_index(solver, I)
    x1 = xcoord(solver, 1, i)
end
```

Indices of halo cells come back outside `1:n_local[d]`, and are meaningful only
as an offset from this rank's block.
"""
@inline function interior_index(solver::SolverLike, I::CartesianIndex{3})
    pad = solver.decomp.n_halo_d
    return (I[1] - pad[1], I[2] - pad[2], I[3] - pad[3])
end

# --- Reading the in-flight conserved state -----------------------------------
#
# `Q` is a 4-D array in the layout defined by the equation set, so a caller
# wanting the density writes `Q[I, 1] + Q[I, 2]` and has silently hardcoded a
# two-species run. The functions below are the layout-independent equivalents,
# for callback conditions, custom diagnostics, and any other code reading `Q`
# between steps, when the primitives are stale; see
# `refresh_primitives!`.
#
# All of them index the PADDED arrays, following the convention of `gidx` and
# `boundary_plane`, and all are rank-local: they report nothing about points
# this rank does not hold. A predicate built from them is reduced by
# `WhenState`, or must be reduced by the caller.

"""
    mixture_density(solver, Q, I) -> ρ

Mixture density at padded index `I`, the sum of the partial densities. Prefer
this method to explicit component indices; the expression then remains valid
when the species count changes.
"""
@inline function mixture_density(solver::SolverLike, Q, I)
    ρ = zero(eltype(Q))
    @inbounds for sp in 1:solver.equations.n_species
        ρ += Q[I, sp]
    end
    return ρ
end

"""
    velocity(solver, Q, I) -> (u, v, w)

Physical, coordinate-aligned velocity components at padded index `I`, recovered
from the momenta and the mixture density.
"""
@inline function velocity(solver::SolverLike, Q, I)
    ri = one(eltype(Q)) / mixture_density(solver, Q, I)
    m1, m2, m3 = solver.equations.i_mom
    @inbounds return (Q[I, m1] * ri, Q[I, m2] * ri, Q[I, m3] * ri)
end

"Total energy per unit volume at padded index `I`."
@inline total_energy(solver::SolverLike, Q, I) =
    @inbounds Q[I, solver.equations.i_energy]

"Mass fraction of species `sp` at padded index `I`."
@inline mass_fraction(solver::SolverLike, Q, I, sp::Int) =
    @inbounds Q[I, sp] / mixture_density(solver, Q, I)

"""
    boundary_plane(solver, d, side) -> CartesianIndices or nothing

Padded indices of this rank's plane on the global `side` (1 low, 2 high) of
dimension `d`, or `nothing` when this rank does not own that edge. Ownership
along `d` is the only test applied, so a periodic or collapsed dimension yields
a plane as well. The plane spans the full local extent of the other two
dimensions and is one point thick in `d`. For example, a boundary diagnostic may
be written as

```julia
plane = boundary_plane(solver, 1, 1)
plane === nothing && return -Inf
return maximum(I -> mixture_density(solver, Q, I), plane)
```

The result is rank-local. [`WhenState`](@ref) reduces a predicate constructed
from it; other uses require an explicit reduction. An unreduced value is valid
globally only in a serial calculation.

These are padded indices. Convert one with [`interior_index`](@ref) before
passing it to [`xcoord`](@ref) or anything else expecting an interior index.
"""
boundary_plane(solver::SolverLike, d::Int, side::Int) = wallplane(solver.decomp, d, side)

"""
    allocate_state(solver)

Zero-filled conserved storage for `solver`: a [`ConservedState`](@ref) matching
its sole patch, or, for a patch-decomposed solver, a `Vector` of them aligned
with `solver.patches`. The vector form is selected by the *global* layout, not
by the local patch count: a rank of a partitioned slab run holds exactly one
patch, and a rank outside a refined level's rank subset holds only the root,
but the multi-patch drivers dispatch on the vector form, and a single-array
state on such a rank silently skips the interface and level synchronization.
"""
function allocate_state(solver::Solver)
    n_cons = solver.equations.n_cons
    _multipatch(solver) || return _state_like(solver.rho, n_cons)
    patches = getfield(solver, :patches)
    levels = getfield(solver, :levels)
    spec = getfield(solver, :regrid)
    # A stacked level's tiles share one stacked state, each tile holding the
    # view over its own block, as its fields are; the step drivers advance
    # the stack through `_stack_state`. The vector's element type then covers
    # the root's dense state and the tiles' views, and is widened as soon as
    # a regrid could stack a tile on this rank, so a rank joining the level
    # later can take one.
    stackable = any(lev -> !isempty(lev.stacks), levels) ||
                (spec !== nothing && _stacked_level(spec.backend, spec.tile))
    stackable || return [_state_like(p.rho, n_cons) for p in patches]
    states = Vector{ConservedState{typeof(solver.cfl)}}(undef, length(patches))
    for lev in levels
        for st in lev.stacks
            _stacked_states!(states, st, n_cons)
        end
        stacked = Set(li for st in lev.stacks for li in st.members)
        for li in lev.patches
            li in stacked && continue
            states[li] = _state_like(patches[li].rho, n_cons)
        end
    end
    return states
end

# Fill the members' slots of `states` with views of one fresh zero state over
# the stack's storage; `shift` moves the slot index (a regrid assembles the
# level's vectors without the root, at index − 1).
function _stacked_states!(states, st::TileStack, n_cons::Int, shift::Int=0)
    raw = _state_like(parent(st.patch.rho), n_cons)
    for (slot, li) in enumerate(st.members)
        states[li + shift] = _tile_state(raw, slot, st.patch.rho.stride)
    end
    return states
end

# Tile `slot`'s view of a stacked state.
_tile_state(raw::ConservedState, slot::Int, stride::Int) =
    ConservedState(view(parent(raw), :, :, ((slot - 1) * stride + 1):(slot * stride), :))

# Whether the solver's layout is a multi-patch one, asked of the layout rather
# than of `npatches`: both a slab partition and a refined level's rank subset
# leave some rank holding one patch of a layout that has several.
_multipatch(solver::Solver) =
    length(getfield(solver, :patch_regions)) > 1 ||
    length(getfield(solver, :levels)) > 1
allocate_state(ps::PatchSolver) = _state_like(ps.rho, ps.equations.n_cons)

# Zero-filled conserved storage matching one patch's field storage, so a
# device-resident patch gets device state without the solver carrying its
# backend object around.
_state_like(rho::AbstractArray{T,3}, n_cons::Int) where {T} =
    ConservedState(fill!(similar(rho, size(rho)..., n_cons), zero(T)))
