# Checkpointing the refinement hierarchy: the record a checkpoint carries
# beyond the root's state, and the restart that rebuilds the level-1 tile set
# from it. The two checkpoint formats (the per-rank files of src/io.jl and the
# shared HDF5 file of the extension) store the record differently and share
# everything else here.
#
# --- What the record holds ----------------------------------------------------
#
# Per refined level, the tile layout: the tile regions in the parent level's
# node space, which is what `_regrid_tiles!` derives a level from, and the
# stored owner ranges, which are the authority across regrids
# (`Level.owners`). Then the regrid state: the step of the last check
# (`RegridSpec.last_step`, so the cadence continues), the check count and the
# per-tile creation record (`checks`, `created`, which the lifetime and the
# hold band read), and the rebalance streak. The tag thresholds, the buffer and
# the interval are configuration the solver constructor supplies, recorded for
# provenance and not restored; the tile edge is recorded and checked, since a
# layout on one lattice cannot be placed on another.
#
# --- Restart on any rank count ------------------------------------------------
#
# A restart rebuilds level 1 from the layout when it differs from the solver's
# (or when the ownership does, on the writing rank count), all tiles fresh and
# their state read from the file afterwards, so no carry-over or migration
# arises. On the rank count that wrote the file the stored ownership is
# restored: the same tiles on the same rank ranges are the same
# decompositions, and the run then continues bit for bit. On another rank
# count the level is partitioned afresh by `_tile_owners`, a different
# decomposition of the same tiles, and the continuation agrees to round-off,
# the tier a subset-owned tile already holds against the serial answer. A
# level below the first is static (regridding is two-level), so its layout
# must be the solver's own; the constructor's `refine` keyword supplies it.
#
# Every decision here is derived from the record, which is identical on every
# rank (broadcast from rank 0 at the write, read by every rank at the load),
# so the collective communicator splits inside `_replace_level!` are reached
# by every rank with the same arguments.

"""
    LevelRecord

One refined level's layout as a checkpoint records it: the tile regions in
the parent level's node space and the owner rank range of each, in the
level's communicator.
"""
struct LevelRecord
    regions::Vector{BlockRegion}
    owners::Vector{UnitRange{Int}}
end

"""
    HierarchyRecord

What a checkpoint carries of the refinement hierarchy beyond the root's
state: the rank count of the writing run, one [`LevelRecord`](@ref) per
refined level, and the regrid state (`tile`, `interval`, `threshold` and
`buffer` for provenance and the tile check; `last_step`, `checks`, the
per-tile `created` record aligned with level 1's regions, `streak` and
`imbalance` restored onto the `RegridSpec`). `tile` is −1 when the writing
run had no `RegridSpec`. Built by [`hierarchy_record`](@ref) and consumed by
[`restore_hierarchy!`](@ref).
"""
struct HierarchyRecord
    np::Int
    levels::Vector{LevelRecord}
    tile::Int
    interval::Int
    threshold::Float64
    buffer::Int
    last_step::Int
    checks::Int
    created::Vector{Int}
    streak::Int
    imbalance::Float64
end

"""
    hierarchy_record(solver) -> HierarchyRecord

The [`HierarchyRecord`](@ref) of `solver`'s current hierarchy, identical on
every rank: rank 0 assembles it, since the level subsets are prefixes of the
rank list and rank 0 therefore holds every level's layout, and broadcasts
it over the run's communicator. Collective.
"""
function hierarchy_record(solver::Solver)
    comm = getfield(solver, :comm)
    rec = nothing
    if MPI.Comm_rank(comm) == 0
        levels = getfield(solver, :levels)
        spec = getfield(solver, :regrid)
        recs = [LevelRecord([lt.region for lt in lev.transfers], copy(lev.owners))
                for lev in levels[2:end]]
        created = spec === nothing || isempty(recs) ? Int[] :
                  [get(spec.created, r, 0) for r in recs[1].regions]
        rec = spec === nothing ?
              HierarchyRecord(MPI.Comm_size(comm), recs, -1, 0, 0.0, 0, 0, 0,
                              created, 0, 1.0) :
              HierarchyRecord(MPI.Comm_size(comm), recs, spec.tile, spec.interval,
                              Float64(spec.threshold), spec.buffer, spec.last_step,
                              spec.checks, created, spec.streak, spec.imbalance)
    end
    return MPI.bcast(rec, comm; root=0)
end

# Flat integer and float images of a record, the form the per-rank checkpoint
# stores; the HDF5 checkpoint stores the same fields as named datasets.
function _record_image(rec::HierarchyRecord)
    ints = Int64[rec.np, length(rec.levels), rec.tile, rec.interval, rec.buffer,
                 rec.last_step, rec.checks, rec.streak, length(rec.created)]
    append!(ints, rec.created)
    for lev in rec.levels
        push!(ints, length(lev.regions))
        for r in lev.regions
            append!(ints, r.offset)
            append!(ints, r.extent)
        end
        for o in lev.owners
            push!(ints, first(o), last(o))
        end
    end
    return ints, Float64[rec.threshold, rec.imbalance]
end

function _record_from_image(ints::AbstractVector{<:Integer},
                            floats::AbstractVector{<:Real})
    pos = Ref(0)
    next!() = (pos[] += 1; Int(ints[pos[]]))
    np = next!()
    nlev = next!()
    tile = next!()
    interval = next!()
    buffer = next!()
    last_step = next!()
    checks = next!()
    streak = next!()
    created = [next!() for _ in 1:next!()]
    levels = LevelRecord[]
    for _ in 1:nlev
        n = next!()
        regions = [BlockRegion((next!(), next!(), next!()),
                               (next!(), next!(), next!())) for _ in 1:n]
        owners = [(lo = next!(); hi = next!(); lo:hi) for _ in 1:n]
        push!(levels, LevelRecord(regions, owners))
    end
    pos[] == length(ints) ||
        error("hierarchy record: $(length(ints)) words where $(pos[]) were read")
    length(floats) == 2 ||
        error("hierarchy record: $(length(floats)) real values where 2 were expected")
    return HierarchyRecord(np, levels, tile, interval, Float64(floats[1]), buffer,
                           last_step, checks, created, streak, Float64(floats[2]))
end

# A checkpoint of a patch layout is written and read through the state vector
# form, which serves the refinement hierarchy: one root patch and its levels.
# The same-level slab layout (`patch_grid`) has no checkpoint yet.
function _check_hierarchy_layout(solver::Solver, name::AbstractString)
    length(getfield(solver, :patch_regions)) == 1 ||
        error("$name: a same-level patch layout (patch_grid) has no checkpoint; " *
              "the state vector form serves the refinement hierarchy")
    return solver
end

"""
    restore_hierarchy!(solver, states, rec, source)

Bring `solver`'s hierarchy to the one `rec` records, resizing `states` (the
vector aligned with `solver.patches`) with it, and restore the regrid state
onto the solver's `RegridSpec`. `source` names the file in error messages.

Level 1 is rebuilt through `_replace_level!` when its tile regions differ
from the solver's, or when the rank count is the one that wrote `rec` and
the stored ownership differs; the rebuilt tiles are fresh, and the caller
reads their state from the checkpoint afterwards. Rebuilding requires a
`RegridSpec` (the schemes a fresh tile is planned with live there) with the
recorded tile edge, and a hierarchy of two levels. Every level below the
first must already match, since it is static. Collective over the run.
"""
function restore_hierarchy!(solver::Solver, states::Vector{<:ConservedState},
                            rec::HierarchyRecord, source::AbstractString)
    levels = getfield(solver, :levels)
    spec = getfield(solver, :regrid)
    length(rec.levels) == length(levels) - 1 ||
        error("refinement mismatch: $source records $(length(rec.levels)) refined " *
              "level(s) and this solver has $(length(levels) - 1)")
    np = MPI.Comm_size(getfield(solver, :comm))
    same_np = rec.np == np
    # A static level: nothing can rebuild it, so the solver must have been
    # built with its regions. A rank outside the parent's subset holds no
    # transfer to compare, and the ranks that do hold them raise together.
    for ℓ in 2:length(rec.levels)
        lev = levels[ℓ + 1]
        isempty(lev.transfers) && continue
        current = [lt.region for lt in lev.transfers]
        current == rec.levels[ℓ].regions ||
            error("level $ℓ layout mismatch: $source records the regions " *
                  "$(rec.levels[ℓ].regions) and this solver was built with " *
                  "$current; a level below the first is static, so build the " *
                  "solver with the regions the checkpoint records")
    end
    if !isempty(rec.levels)
        lev = levels[2]
        want = rec.levels[1]
        current = [lt.region for lt in lev.transfers]
        if current != want.regions || (same_np && lev.owners != want.owners)
            length(levels) == 2 ||
                error("level 1 layout mismatch: $source records the regions " *
                      "$(want.regions) and this solver was built with $current; " *
                      "a hierarchy of more than two levels is static, so build " *
                      "the solver with the regions the checkpoint records")
            spec === nothing &&
                error("level 1 layout mismatch: $source records the regions " *
                      "$(want.regions) and this solver was built with $current; " *
                      "a solver without regridding cannot rebuild its level, so " *
                      "build it with refine set to the recorded region, or with " *
                      "regrid_interval > 0 and the recorded tile")
            spec.tile == rec.tile ||
                error("tile mismatch: $source was written with tile = $(rec.tile) " *
                      "and this solver has tile = $(spec.tile); a layout on one " *
                      "lattice cannot be placed on another")
            _replace_level!(solver, states, want.regions,
                            same_np ? want.owners : nothing)
        end
    end
    if spec !== nothing
        spec.last_step = rec.last_step
        spec.checks = rec.checks
        spec.streak = rec.streak
        spec.imbalance = rec.imbalance
        # The busy-time marks are wall-clock readings of the writing process;
        # the interval measured at the next check starts here.
        spec.wall_mark = solver.wall_total
        spec.wait_mark = solver.wait_total
        spec.wall_regrid = 0.0
        empty!(spec.created)
        if !isempty(rec.levels)
            regions = rec.levels[1].regions
            # A file written without regridding records no creation check;
            # every tile then dates from check 0, as at setup.
            created = isempty(rec.created) ? zeros(Int, length(regions)) : rec.created
            length(created) == length(regions) ||
                error("hierarchy record: $(length(created)) creation checks for " *
                      "$(length(regions)) tiles")
            for (r, c) in zip(regions, created)
                spec.created[r] = c
            end
        end
    end
    return solver
end

# Rebuild level 1 over `regions` (root node space) with the owner ranges
# `owners`, or with a fresh partition when `owners` is `nothing`, every tile
# fresh and its state zero. The shape of `_regrid_tiles!` without the tag
# decision, the carry and the migration: the current tiles are dropped
# (their last restriction is not taken, since the root's state is about to be
# replaced as well), the level and tile communicators are split afresh, and
# the covered mask and the geometry follow. `states` is resized with the
# patch vector. Collective over the run: `regions` and `owners` are the same
# on every rank.
function _replace_level!(solver::Solver{T}, states::Vector{<:ConservedState},
                         regions::Vector{BlockRegion},
                         stored_owners::Union{Nothing,Vector{UnitRange{Int}}}) where {T}
    spec = getfield(solver, :regrid)
    levels = getfield(solver, :levels)
    patches = getfield(solver, :patches)
    lev = levels[2]
    root = patches[1]
    n_global = solver.n_global
    active = ntuple(d -> n_global[d] > 1, 3)
    root_lc = levels[1].level_comm
    old_lc = lev.level_comm
    n_cons = solver.equations.n_cons
    spec.tile == 0 && length(regions) != 1 &&
        error("restart: the checkpoint records $(length(regions)) tiles on level " *
              "1 and this solver refines one box (tile = 0)")
    for r in regions
        _covered_by(_buffered(r, active, spec.margin), [root.region]) ||
            error("restart: the recorded region $r is not nested $(spec.margin) " *
                  "root nodes inside the domain")
    end
    if stored_owners === nothing
        owners, np_new = _tile_owners(regions, active, root_lc.size)
    else
        length(stored_owners) == length(regions) ||
            error("restart: $(length(stored_owners)) owner ranges for " *
                  "$(length(regions)) tiles")
        owners = stored_owners
        np_new = maximum(last, owners) + 1
        np_new <= root_lc.size ||
            error("restart: the recorded ownership reaches rank $(np_new - 1) " *
                  "of $(root_lc.size)")
    end
    # The dropped tiles' decompositions, freed after the swap; the old
    # transfers' chains go with them. Every rank drops the whole level.
    dropped_decomps = [patches[li].decomp for li in lev.patches]
    free_tile_group!(lev.group)
    resized = np_new != old_lc.size
    resized && free_level_comm!(old_lc)
    new_lc = resized ? split_level_comm(root_lc, np_new) : old_lc
    group = new_lc.owned ? split_tile_comm(new_lc, owners) : absent_tile_group()
    held = [ti for ti in eachindex(regions) if owners[ti] == group.ranks]
    faces = _tile_faces(regions)
    ws_pool = [p.rhs_workspace for p in patches]
    local_of = zeros(Int, length(regions))
    indices = [k + 1 for k in eachindex(held)]
    for (k, ti) in enumerate(held)
        local_of[ti] = k + 1
    end
    # Every tile is fresh: on the host through `_build_fine_patch`, on a
    # device backend as stacked storage (rhs.jl).
    new_patches, stacks = _build_level_patches(T, regions, held, faces, active,
                                               root.h, spec.n_halo, group.comm,
                                               spec.deriv, spec.filt, spec.smoo,
                                               solver.art.smoother,
                                               spec.interface_rhs, spec.backend,
                                               ws_pool, solver.equations.n_species,
                                               n_cons, 1, 1, spec.tile)
    restriction = lev.transfers[1].restriction
    transfers = [build_level_transfer(
        T, tr, active, spec.n_halo, [root.region], [1],
        Union{Nothing,Decomp{T}}[root.decomp], local_of[ti], restriction,
        n_cons, getfield(solver, :subcycle),
        local_of[ti] == 0 ? nothing : new_patches[local_of[ti] - 1].decomp,
        root_lc.comm, length(owners[ti]), faces[ti])
        for (ti, tr) in enumerate(regions)]
    resize!(patches, 1 + length(held))
    resize!(states, 1 + length(held))
    stacked = Set(li for st in stacks for li in st.members)
    for (k, p) in enumerate(new_patches)
        patches[k + 1] = p
        k + 1 in stacked || (states[k + 1] = _state_like(p.rho, n_cons))
    end
    for st in stacks
        _stacked_states!(states, st, n_cons)
    end
    if new_lc.owned
        fine_regions = [BlockRegion(ntuple(d -> active[d] ? 3 * tr.offset[d] : 0, 3),
                                    fine_extent(tr, active)) for tr in regions]
        records = _level_records(T, new_lc.comm, fine_regions, held, indices,
                                 [p.decomp for p in new_patches], n_cons)
        levels[2] = Level{T}(1, new_lc, owners, group, held, indices, transfers,
                             records; stacks)
    else
        levels[2] = Level{T}(1, new_lc, owners, group, held, indices, transfers;
                             stacks)
    end
    _fill_covered!(root, regions)
    for li in indices
        init_geometry!(PatchSolver(solver, patches[li]))
    end
    for old_lt in lev.transfers
        free_transfer_decomps!(old_lt)
    end
    for decomp in dropped_decomps
        free_communicators!(decomp)
    end
    return solver
end

# This rank's tiles of the refined levels, as `(level, tile, patch index)`
# triples in the order the patch vector holds them: what a checkpoint writes
# blocks for beyond the root, and what a restart reads them back into.
function _held_tiles(solver::Solver)
    levels = getfield(solver, :levels)
    return [(ℓ - 1, ti, li) for ℓ in 2:length(levels)
            for (li, ti) in zip(levels[ℓ].patches, levels[ℓ].tiles)]
end
