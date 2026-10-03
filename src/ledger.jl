# Budget ledger: the composite conserved integrals attributed to the
# mechanism that changed them.
#
# A diagnostic for `bench/interfaceconservation.jl` and its tests, private
# like `_conserved_budget`, whose quadrature it shares: trapezoid weights,
# one half at a node-centered edge and at a patch's interface end, the metric
# Jacobian with its edge factor (the quadrature note in `diagnostics.jl`), and
# on a composite solver the covered fraction of each coarse node's cell.
# Every driver that writes the state brackets the write with two hooks:
# `_ledger_open!` before it and `_ledger!(…, phase)` after it. A hook
# recomputes the budget of the patches it names and adds the change since
# that patch's previous hook to `(phase, patch level)`; the opening hook
# names the phase `:unattributed`, so a write no bracket covers lands there,
# and the telescoping sum of every piece is the total drift. The right-hand
# side is also integrated along the low-storage recurrence from the stage
# right-hand sides and from the fluxes through each kind of patch face
# (`LEDGER_DERIVED`), which gives the boundary-flux closure: on a node-centered
# physical face the flux entering through it, against the `:rhs` piece.
#
# Everything a hook does is rank-local: no hook reduces, so a hook inside
# the subcycled recursion, which only a level's owners enter, cannot strand
# a rank, and a patch moving between ranks at a regrid is summed correctly
# at the end. `_ledger_end!` is the one collective. The hooks cost a field
# load and a branch when the ledger is off; `_ledger_mark!` is compiled once,
# unspecialized, so it adds no per-solver-type code to the drivers.

"""The mechanisms a ledger piece is attributed to, in report order."""
const LEDGER_PHASES = (:rhs, :wall_enforce, :filter, :same_level, :shell, :restrict,
                       :regrid, :repair, :truncation, :callback, :rollback,
                       :unattributed)

# The right-hand side integrated through the stage recurrence: its own
# integral, and the fluxes entering through physical, same-level and
# coarse-fine patch faces, and, on a parent patch, through the boundary of
# the region its child level covers. Not state writes, so outside the
# telescoping sum. On a conservative coupling the last two cancel.
const LEDGER_DERIVED = (:rhs_integral, :wall_flux, :same_level_flux,
                        :coarse_fine_flux, :covered_face_flux)

mutable struct BudgetLedger
    on::Bool
    n::Int                          # n_species + 4 budget channels
    version::Int                    # regrid checks at the last rebase
    levels::Vector{Int}             # per held patch, its level
    last::Vector{Vector{Float64}}   # per held patch, the budget at its last hook
    initial::Vector{Float64}        # this rank's total at `_ledger_begin!`
    pieces::Dict{Tuple{Symbol,Int},Vector{Float64}}
    du_rhs::Vector{Vector{Float64}} # per patch, the recurrence of ∫ dQ
    du_face::Vector{Matrix{Float64}}    # per patch, n × 4 face-flux recurrences
    stage_face::Vector{Matrix{Float64}} # per patch, this stage's face fluxes
end

BudgetLedger() = BudgetLedger(false, 0, 0, Int[], Vector{Float64}[], Float64[],
                              Dict{Tuple{Symbol,Int},Vector{Float64}}(),
                              Vector{Float64}[], Matrix{Float64}[], Matrix{Float64}[])

const BUDGET_LEDGER = BudgetLedger()

# The hooks. `scope` is a `Level` (its patches), a held patch index, or
# `nothing` (every held patch). The test is a Bool field of a concrete
# constant, so the drivers' inferred code gains no non-concrete value and no
# dispatch site.
@inline function _ledger!(solver, states, phase::Symbol, scope=nothing)
    BUDGET_LEDGER.on && _ledger_mark!(BUDGET_LEDGER, solver, states, phase, scope)
    return nothing
end

@inline _ledger_open!(solver, states, scope=nothing) =
    _ledger!(solver, states, :unattributed, scope)

# After the stage update of `scope`'s patches: the recurrence, then the mark.
@inline function _ledger_update!(solver, states, dQs, A, B, dt, scope=nothing)
    BUDGET_LEDGER.on &&
        _ledger_stage!(BUDGET_LEDGER, solver, states, dQs, A, B, dt, scope)
    return nothing
end

# After `compute_rhs!` of held patch `pi`, while the pooled flux is its own.
@inline function _ledger_faces!(solver, pi::Int)
    BUDGET_LEDGER.on && _ledger_capture!(BUDGET_LEDGER, solver, pi)
    return nothing
end

# A regrid, a rollback or anything else that replaces the layout: every held
# patch rebased, the change per level attributed to `phase`.
@inline function _ledger_rebase!(solver, states, phase::Symbol)
    BUDGET_LEDGER.on && _ledger_rebase_all!(BUDGET_LEDGER, solver, states, phase)
    return nothing
end

_ledger_held(solver) = length(getfield(solver, :patches))
_ledger_state(states::Vector, pi) = states[pi]
_ledger_state(Q, pi) = Q
_ledger_version(solver) = (spec = getfield(solver, :regrid);
                           spec === nothing ? 0 : spec.checks)
_ledger_indices(solver, ::Nothing) = 1:_ledger_held(solver)
_ledger_indices(solver, pi::Int) = pi:pi
_ledger_indices(solver, lev) = lev.patches

function _ledger_patch_budget(solver, states, pi::Int)
    ps = PatchSolver(solver, getfield(solver, :patches)[pi])
    return _local_conserved_budget(ps, _ledger_state(states, pi), _composite(solver))
end

function _ledger_piece!(L::BudgetLedger, phase::Symbol, level::Int)
    return get!(() -> zeros(L.n), L.pieces, (phase, level))
end

"""
    _ledger_begin!(solver, Q)

Start attributing the composite budget of `Q` (a state or a state vector) to
the mechanisms that change it, from this state. Rank-local; every rank of
`solver.comm` calls it before the run and `_ledger_end!` after. Host storage
only: the budget is a host sweep.
"""
function _ledger_begin!(solver::Solver, Q)
    _cpu_storage(_ledger_state(Q, 1)) ||
        error("the budget ledger is a host sweep; a DeviceBackend is not supported")
    L = BUDGET_LEDGER
    L.n = solver.equations.n_species + 4
    empty!(L.pieces)
    _ledger_reset!(L, solver, Q)
    L.initial = reduce(+, L.last; init=zeros(L.n))
    L.on = true
    return L
end

function _ledger_reset!(L::BudgetLedger, solver, states)
    np = _ledger_held(solver)
    patches = getfield(solver, :patches)
    L.version = _ledger_version(solver)
    L.levels = [patches[pi].level for pi in 1:np]
    L.last = [_ledger_patch_budget(solver, states, pi) for pi in 1:np]
    L.du_rhs = [zeros(L.n) for _ in 1:np]
    L.du_face = [zeros(L.n, 4) for _ in 1:np]
    L.stage_face = [fill(NaN, L.n, 4) for _ in 1:np]
    return L
end

@noinline function _ledger_rebase_all!(L::BudgetLedger, @nospecialize(solver),
                                       @nospecialize(states), phase::Symbol)
    old = Dict{Int,Vector{Float64}}()
    for (pi, b) in enumerate(L.last)
        acc = get!(() -> zeros(L.n), old, L.levels[pi])
        acc .+= b
    end
    _ledger_reset!(L, solver, states)
    new = Dict{Int,Vector{Float64}}()
    for (pi, b) in enumerate(L.last)
        acc = get!(() -> zeros(L.n), new, L.levels[pi])
        acc .+= b
    end
    for ℓ in union(keys(old), keys(new))
        _ledger_piece!(L, phase, ℓ) .+= get(() -> zeros(L.n), new, ℓ) .-
                                        get(() -> zeros(L.n), old, ℓ)
    end
    return nothing
end

@noinline function _ledger_mark!(L::BudgetLedger, @nospecialize(solver),
                                 @nospecialize(states), phase::Symbol,
                                 @nospecialize(scope))
    # A regrid check replaces the layout and rewrites the covered masks in
    # place; the first hook after it rebases every patch.
    if _ledger_version(solver) != L.version || _ledger_held(solver) != length(L.last)
        _ledger_rebase_all!(L, solver, states, :regrid)
    end
    for pi in _ledger_indices(solver, scope)
        b = _ledger_patch_budget(solver, states, pi)
        _ledger_piece!(L, phase, L.levels[pi]) .+= b .- L.last[pi]
        L.last[pi] = b
    end
    return nothing
end

@noinline function _ledger_stage!(L::BudgetLedger, @nospecialize(solver),
                                  @nospecialize(states), @nospecialize(dQs),
                                  A, B, dt, @nospecialize(scope))
    if _ledger_version(solver) != L.version || _ledger_held(solver) != length(L.last)
        _ledger_rebase_all!(L, solver, states, :regrid)
    end
    for pi in _ledger_indices(solver, scope)
        ℓ = L.levels[pi]
        # The low-storage update adds B (A du + dt dQ) to the state, so the
        # same recurrence on the integrals gives each stage's contribution.
        R = _ledger_patch_budget(solver, dQs, pi)
        du = L.du_rhs[pi]
        @. du = A * du + dt * R
        _ledger_piece!(L, :rhs_integral, ℓ) .+= B .* du
        duf = L.du_face[pi]
        @. duf = A * duf + dt * L.stage_face[pi]
        for (k, name) in enumerate(LEDGER_DERIVED[2:5])
            _ledger_piece!(L, name, ℓ) .+= B .* view(duf, :, k)
        end
        fill!(L.stage_face[pi], NaN)
    end
    _ledger_mark!(L, solver, states, :rhs, scope)
    return nothing
end

# The flux entering held patch `pi` through each kind of face, from the flux
# `compute_rhs!` differenced (after `correct_flux!`): column 1 physical, 2
# same-level, 3 coarse-fine, and 4 the parent's flux into each region its
# child level covers, across the region's parent-fed faces, which the
# covered mask removes from the composite. A face counts on a node-centered
# end only: a
# periodic dimension has none, and a folded end is half a cell from its
# plane. The face node's flux times the transverse weights of the budget,
# edge factors included, the in-plane uncovered fraction and the
# transverse cell measure is the term the flux divergence integrates to at
# that end; an interface end under `interface_flux = :ghost` differences
# more than this flux, so there its column is the closure flux only.
@noinline function _ledger_capture!(L::BudgetLedger, @nospecialize(solver), pi::Int)
    patch = getfield(solver, :patches)[pi]
    ps = PatchSolver(solver, patch)
    J = zeros(L.n, 4)
    decomp = ps.decomp
    eq = ps.equations
    nsp = eq.n_species
    comps = Int[1:nsp; collect(eq.i_mom); eq.i_energy]
    o = decomp.n_halo_d
    nl = decomp.n_local
    masked = _composite(solver)
    measure = Float64(cell_measure(ps))
    for d in 1:3, side in 1:2
        decomp.active[d] && !decomp.periodic[d] || continue
        fold = ps.folds[d]
        fold !== nothing && (side == 1 ? fold.lo : fold.hi) && continue
        (side == 1 ? decomp.offset[d] == 0 :
                     decomp.offset[d] + nl[d] == decomp.n_global[d]) || continue
        bc = ps.bcs[d][side]
        kind = bc isa InterfaceBC ? (bc.neighbor == 0 ? 3 : 2) : 1
        sgn = side == 1 ? 1.0 : -1.0
        il = side == 1 ? 1 : nl[d]
        area = measure / Float64(ps.h[d])
        Ad = ps.area_d[d]
        rng = ntuple(t -> t == d ? (il:il) : (1:nl[t]), 3)
        for k in rng[3], j in rng[2], i in rng[1]
            I = CartesianIndex(i + o[1], j + o[2], k + o[3])
            w = 1.0
            d == 1 || (w *= quad_weight(ps, 1, i) * _edge_factor(ps, 1, i, I))
            d == 2 || (w *= quad_weight(ps, 2, j) * _edge_factor(ps, 2, j, I))
            d == 3 || (w *= quad_weight(ps, 3, k) * _edge_factor(ps, 3, k, I))
            if masked
                m = ps.covered[I]
                m == 0 || (w *= uncovered_plane_fraction(m, d))
            end
            w *= sgn * area * Float64(Ad[I])
            for (b, c) in enumerate(comps)
                J[b, kind] += w * Float64(ps.flux[d, c][I])
            end
        end
    end
    levels = getfield(solver, :levels)
    if patch.level + 2 <= length(levels)
        # A region across a periodic seam meets this patch in its images.
        images = _images(_level_period(solver, patch.level))
        for lt in levels[patch.level + 2].transfers, σ in images
            _ledger_covered_faces!(J, ps, patch, _shifted(lt.region::BlockRegion, σ),
                                   lt.imposed::NTuple{3,NTuple{2,Bool}}, comps)
        end
    end
    L.stage_face[pi] = J
    return nothing
end

# Column 4: the parent's flux across the parent-fed faces of child region
# `r` (parent-level node space), with the trapezoid weights of the region's
# own face and the parent's transverse cell measure. Leaving the parent's
# uncovered part into the region counts negative, so on a conservative
# coupling this column and the child's column 3 sum to zero.
function _ledger_covered_faces!(J, ps, patch, r::BlockRegion, imposed, comps)
    decomp = ps.decomp
    o = decomp.n_halo_d
    nl = decomp.n_local
    base = ntuple(d -> patch.region.offset[d] + decomp.offset[d], 3)
    measure = Float64(cell_measure(ps))
    for d in 1:3, side in 1:2
        decomp.active[d] && imposed[d][side] || continue
        g = side == 1 ? r.offset[d] + 1 : r.offset[d] + r.extent[d]
        il = g - base[d]
        1 <= il <= nl[d] || continue
        sgn = side == 1 ? -1.0 : 1.0
        area = measure / Float64(ps.h[d])
        rng = ntuple(3) do t
            t == d && return il:il
            decomp.active[t] || return 1:1
            lo = max(r.offset[t] + 1 - base[t], 1)
            lo:min(r.offset[t] + r.extent[t] - base[t], nl[t])
        end
        for k in rng[3], j in rng[2], i in rng[1]
            I = CartesianIndex(i + o[1], j + o[2], k + o[3])
            w = sgn * area * Float64(ps.area_d[d][I])
            for (t, it) in ((1, i), (2, j), (3, k))
                (t == d || !decomp.active[t]) && continue
                gt = base[t] + it
                (gt == r.offset[t] + 1 || gt == r.offset[t] + r.extent[t]) && (w *= 0.5)
            end
            for (b, c) in enumerate(comps)
                J[b, 4] += w * Float64(ps.flux[d, c][I])
            end
        end
    end
    return J
end

"""
    _ledger_end!(solver, Q) -> NamedTuple

Close the ledger on the state `Q` and reduce it over `solver.comm`: every
rank calls it. Returns `pieces`, a `Dict` from `(phase, level)` to the change
of each budget channel (species masses, then the three momenta, then the
energy) that phase made on that level; `initial` and `final`, the composite
budgets; and `residual`, the final less the initial less the sum of the
`LEDGER_PHASES` pieces, the round-off of the telescoping sum.
`close = false` reports the ledger so far and leaves it open.
"""
function _ledger_end!(solver::Solver, Q; close::Bool=true)
    L = BUDGET_LEDGER
    L.on || error("_ledger_end!: no ledger is open")
    _ledger_mark!(L, solver, Q, :unattributed, nothing)
    L.on = !close
    final = reduce(+, L.last; init=zeros(L.n))
    # The levels a rank has seen differ (a rank holding no tile of a level
    # records nothing there), so the dense layout takes the largest.
    top = max(maximum(last, keys(L.pieces); init=0), maximum(L.levels; init=0))
    nlev = MPI.Allreduce(top + 1, max, solver.comm)
    names = (LEDGER_PHASES..., LEDGER_DERIVED...)
    dense = zeros(L.n, nlev, length(names))
    for ((phase, ℓ), v) in L.pieces
        dense[:, ℓ + 1, findfirst(==(phase), names)] .= v
    end
    red = MPI.Allreduce(vcat(vec(dense), L.initial, final), +, solver.comm)
    nd = length(dense)
    dense = reshape(red[1:nd], size(dense))
    initial = red[nd+1:nd+L.n]
    final = red[nd+L.n+1:end]
    pieces = Dict{Tuple{Symbol,Int},Vector{Float64}}()
    for (p, name) in enumerate(names), ℓ in 0:nlev-1
        v = dense[:, ℓ + 1, p]
        any(!iszero, v) && (pieces[(name, ℓ)] = v)
    end
    attributed = zeros(L.n)
    for ((phase, _), v) in pieces
        phase in LEDGER_PHASES && (attributed .+= v)
    end
    return (; pieces, initial, final, residual=final .- initial .- attributed,
            levels=nlev)
end
