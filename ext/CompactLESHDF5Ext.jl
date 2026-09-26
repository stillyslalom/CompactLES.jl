module CompactLESHDF5Ext

# HDF5 implementation of the shared-file writes declared in src/hdf5.jl. Read
# the note at the top of that file first: it explains why this is an extension,
# and why there are two write backends.

using CompactLES
using CompactLES: BlockRegion, Decomp, PatchSolver, Solver, owned_region, region_ranges
using CompactLES: axis_matches, global_axis, type_name
using CompactLES: ensure_output_dir, restore_switches!, switch_codes
using CompactLES: n_art_fields, art_block, set_art_block!, nlevels
using CompactLES: _check_level_count, _check_art_count, refresh_primitives!
using CompactLES: HierarchyRecord, LevelRecord, hierarchy_record, restore_hierarchy!
using CompactLES: ConfigurationRecord, configuration_record, configuration_digests
using CompactLES: _verify_configuration
using MPI
using HDF5

has_parallel() = HDF5.has_parallel()

# Format 2 added the species set, the metric and the grid coordinates to the
# checkpoint header, which format 1 has no record of. Format 3 added the element
# type of the state and the mutable run state (`cfl`, `dt_prev`, `rate_prev`,
# and the `switched` flag of each boundary face); format 4 the level count and
# the artificial coefficient arrays; format 5 the per-direction rates
# `filter_weight` reads; format 6 the configuration record under `config`.
# The reasoning is at the top of `src/io.jl`. The reader accepts format 5,
# which is format 6 without the record, and loads it unchecked with a
# warning; an older file is refused, since a restart those fields do not
# cover cannot be validated at all.
const CKPT_FORMAT = 6
const CKPT_FORMAT_UNRECORDED = 5

# --- Opening a shared file --------------------------------------------------
#
# `with_shared_file(body, path, mode, comm)` calls `body(file)` on every rank
# with a writable handle, under whichever backend the libhdf5 build supports.
#
# The serialized backend is a token relay. Rank 0 runs `body` first, because it
# is the rank that must create the file and its datasets; every other rank then
# waits for its predecessor, opens the existing file, runs `body`, and closes.
# Closing before passing the token makes this safe: a serial libhdf5 cannot
# have the file open in two processes at once, and the result is corruption,
# not a failure.
#
# An exception inside `body` is a collective problem, not a local one. A
# rank that threw before passing its token leaves its successor blocked in
# `Recv!` and its predecessors blocked in the closing collective, so the token
# moves from a `finally` and the failure is then reduced across the
# communicator. The ranks that succeeded also raise, avoiding a return into a
# communicator whose next collective they no longer agree on.
#
# The parallel backend cannot be made safe to the same degree. `h5open` on the
# MPI-IO driver and every dataset creation under it are themselves collective, so
# a rank throwing partway through `body` has diverged from the others and
# blocks them where they stand. The reduction below still covers a failure raised
# on every rank, or one raised after the last collective call in `body`.

# --- Transfer mode under the parallel backend --------------------------------
#
# Nothing here sets `dxpl_mpio = :collective`, so every hyperslab write below
# goes out under HDF5's default independent transfer mode. That is correct but
# not fast: a collective transfer lets the MPI-IO layer aggregate the per-rank
# hyperslabs into a few large contiguous writes, accounting for most of a
# shared write scale at high rank counts.
#
# A collective transfer requires every rank of the file's communicator to call
# H5Dwrite on the same dataset in the same order. `write_block!` below already
# does that: a rank holding no block of a dataset (no part of a sliced plane, or
# no piece of a refined tile) issues a write with an empty selection instead of
# skipping it. The empty write is issued under the serialized backend too, where
# it is a no-op, so the workstation tests exercise the call sequence the parallel
# backend depends on. Setting `dxpl_mpio = :collective` on the block datasets is
# what remains, and belongs on a machine with a parallel libhdf5 built against
# the run's MPI, the only place it can be exercised: `hdf5_parallel()` is false
# on a workstation, where no transfer property applies.
#
# Measured on that machine: the change does not pay.
# A 128^3 Taylor-Green run over 224 ranks on two rzhound nodes (system
# MVAPICH2 2.3.7, hdf5-parallel 1.14.0, Lustre at stripe count 8) wrote two
# 151 MB checkpoints during 11,504 steps. Everything outside the solver came to
# 17.75 s of a 1176.34 s run, 1.5%, and that figure also includes a per-step
# globally reduced kinetic energy from the caller's own callback. Both
# independent-mode writes together therefore cost at most 1.5% of the run, and
# in practice far less. At this rank count and file size the aggregation a
# collective transfer buys cannot be large.

# The relay token's tag. Nothing else uses 0 on these communicators; the halo
# families start at 10 (see the tag note in src/halo.jl).
const TOKEN_TAG = 0

function with_shared_file(body, path::AbstractString, mode::AbstractString,
                          comm::MPI.Comm)
    if has_parallel()
        failure = nothing
        try
            # One collective open.
            h5open(path, mode, comm) do file
                body(file)
            end
        catch e
            failure = e
        end
        _raise_shared_failure(failure, path, comm)
        return path
    end
    rank = MPI.Comm_rank(comm)
    nranks = MPI.Comm_size(comm)
    token = Ref(0)
    rank == 0 || MPI.Recv!(token, comm; source=rank - 1, tag=TOKEN_TAG)
    # Rank 0 creates ("w" / "cw"), everyone after appends to what exists.
    local_mode = rank == 0 ? mode : (mode == "r" ? "r" : "r+")
    failure = nothing
    try
        h5open(path, local_mode) do file
            body(file)
        end
    catch e
        failure = e
    finally
        rank == nranks - 1 || MPI.Send(token, comm; dest=rank + 1, tag=TOKEN_TAG)
    end
    _raise_shared_failure(failure, path, comm)
    return path
end

# The reduction doubles as the barrier closing `with_shared_file`:
# every rank has finished with the file by the time it returns.
function _raise_shared_failure(failure, path::AbstractString, comm::MPI.Comm)
    anyfail = MPI.Allreduce(failure === nothing ? 0 : 1, max, comm)
    failure === nothing || throw(failure)
    anyfail == 0 ||
        error("with_shared_file: another rank failed on $path and raised the " *
              "cause; this rank's own write completed")
    return nothing
end

# A dataset covering the whole global array, created once and written in
# per-rank pieces. Under the serialized backend only rank 0 creates it.
function shared_dataset(file, name::AbstractString, ::Type{T}, dims,
                        comm::MPI.Comm) where {T}
    if has_parallel() || MPI.Comm_rank(comm) == 0
        return create_dataset(file, name, datatype(T), dataspace(dims))
    end
    return file[name]
end

# --- Rank-independent metadata ----------------------------------------------
#
# Metadata carries the same value on every rank, but it cannot be written only
# from rank 0 under the parallel backend: that file is open on the MPI-IO
# driver, where creating a group or a dataset is collective and a rank skipping
# one leaves the ranks with divergent file structure. So every rank reaches
# every creation call, and the write that follows is independent and done by
# rank 0 alone. Under the serialized backend rank 0 holds the file by itself and
# both halves are its own.

function write_meta!(g, name::AbstractString, value, rank::Int)
    dtype = datatype(value)
    dset = create_dataset(g, name, dtype, dataspace(value))
    rank == 0 && write_dataset(dset, dtype, value)
    close(dset)
    return nothing
end

# A Julia String maps to a variable-length HDF5 datatype, and parallel HDF5
# refuses to write one ("Parallel IO does not support writing VL or region
# reference datatypes yet"), so string metadata goes out as fixed-length
# null-padded records. HDF5.jl strips the padding on read and returns String.
function write_strings!(g, name::AbstractString, strs, rank::Int)
    width = maximum(ncodeunits, strs; init=1)
    dtype = HDF5.Datatype(HDF5.API.h5t_copy(HDF5.API.H5T_C_S1))
    HDF5.API.h5t_set_size(dtype, width)
    HDF5.API.h5t_set_strpad(dtype, HDF5.API.H5T_STR_NULLPAD)
    HDF5.API.h5t_set_cset(dtype, HDF5.API.H5T_CSET_UTF8)
    dset = create_dataset(g, name, dtype, dataspace((length(strs),)))
    if rank == 0
        buf = zeros(UInt8, width, length(strs))
        for (i, s) in enumerate(strs)
            copyto!(view(buf, 1:ncodeunits(s), i), codeunits(s))
        end
        write_dataset(dset, dtype, buf)
    end
    close(dset)
    return nothing
end


# One rank's block of a shared dataset: `dset[ranges...] = data`, or, with
# `data === nothing` on a rank holding no block, a write selecting no element in
# either dataspace. Every rank calls this for every block dataset, in the same
# order, which a collective transfer requires; see the transfer-mode note above.
function write_block!(dset, ranges, data)
    data === nothing || return (dset[ranges...] = data)
    memtype = datatype(dset)
    fspace = dataspace(dset)
    mspace = dataspace((0,))
    try
        lock(HDF5.API.liblock)
        status = try
            ccall((:H5Sselect_none, HDF5.API.libhdf5), HDF5.API.herr_t,
                  (HDF5.API.hid_t,), fspace)
        finally
            unlock(HDF5.API.liblock)
        end
        status < 0 && error("write_block!: H5Sselect_none failed")
        HDF5.API.h5d_write(dset, memtype, mspace, fspace, dset.xfer, UInt8[0])
    finally
        close(mspace)
        close(fspace)
        close(memtype)
    end
    return nothing
end
read_region3(dset, region::BlockRegion) = dset[region_ranges(region)...]
read_region4(dset, region::BlockRegion, ncomp::Int) =
    dset[region_ranges(region)..., 1:ncomp]

# --- Checkpoint / restart ---------------------------------------------------
#
# WHAT THE HEADER HAS TO PIN DOWN. The state is one flat global array of
# conserved components, and nothing in `state/Q` records what those components
# mean or where the points are. A restart onto a solver that disagrees is
# therefore not detectable from the array: it reads cleanly and decodes to a
# different state. The header records everything the interpretation depends on, and
# `load_checkpoint_hdf5!` checks all of it.
#
# The reasoning for each field is at the top of `src/io.jl`, whose per-rank
# checkpoint records the same set for the same reasons, and `type_name`,
# `global_axis` and `axis_matches` are taken from there, not restated here, so
# that the two paths cannot drift apart.

function CompactLES.save_checkpoint_hdf5(solver::Solver, Q, prefix::AbstractString)
    CompactLES._multipatch(solver) &&
        error("save_checkpoint_hdf5: this solver holds a patch layout; pass the " *
              "state vector allocate_state returned")
    comm = solver.comm
    rank = MPI.Comm_rank(comm)
    path = string(prefix, ".h5")
    ensure_output_dir(prefix, comm)
    with_shared_file(path, "w", comm) do file
        (has_parallel() || rank == 0) && _write_ckpt_meta!(file, solver, solver, Q, rank)
        _write_state!(file, "state", solver, Q, solver.decomp.n_global, comm,
                      solver.equations.n_cons, n_art_fields(solver), eltype(Q))
    end
    return prefix
end

# The refined-hierarchy form: the root's state under `state`, the hierarchy
# record under `hierarchy`, and every tile's state under
# `levels/<level>/tiles/<tile>`, each dataset spanning the tile's own node
# space and written in per-rank hyperslabs. Every rank walks every tile of
# the record, since under the parallel backend a dataset's creation is
# collective, and writes a block only where it holds one.
function CompactLES.save_checkpoint_hdf5(solver::Solver,
                                         states::Vector{<:ConservedState},
                                         prefix::AbstractString)
    CompactLES._check_hierarchy_layout(solver, "save_checkpoint_hdf5")
    patches = getfield(solver, :patches)
    root = PatchSolver(solver, patches[1])
    comm = solver.comm
    rank = MPI.Comm_rank(comm)
    rec = hierarchy_record(solver)
    active = ntuple(d -> solver.n_global[d] > 1, 3)
    held = Dict((ℓ, ti) => li for (ℓ, ti, li) in CompactLES._held_tiles(solver))
    n_cons = solver.equations.n_cons
    n_art = n_art_fields(root)
    T = eltype(states[1])
    path = string(prefix, ".h5")
    ensure_output_dir(prefix, comm)
    with_shared_file(path, "w", comm) do file
        if has_parallel() || rank == 0
            _write_ckpt_meta!(file, solver, root, states[1], rank)
            _write_record!(file, rec, rank)
        end
        _write_state!(file, "state", root, states[1], root.decomp.n_global, comm,
                      n_cons, n_art, T)
        for (ℓ, lev) in enumerate(rec.levels), (ti, region) in enumerate(lev.regions)
            li = get(held, (ℓ, ti), 0)
            _write_state!(file, "levels/$ℓ/tiles/$ti",
                          li == 0 ? nothing : PatchSolver(solver, patches[li]),
                          li == 0 ? nothing : states[li],
                          CompactLES.fine_extent(region, active), comm,
                          n_cons, n_art, T)
        end
    end
    return prefix
end

# The header, describing the root patch `root`'s state (the solver itself
# when it holds one patch) and the mutable run state; the reasoning per field
# is at the top of src/io.jl.
function _write_ckpt_meta!(file, solver::Solver, root, Q, rank::Int)
    decomp = root.decomp
    g = create_group(file, "meta")
    write_meta!(g, "format", CKPT_FORMAT, rank)
    write_meta!(g, "t", Float64(solver.t), rank)
    write_meta!(g, "step", Int64(solver.step), rank)
    write_meta!(g, "n_global", Int64[decomp.n_global...], rank)
    write_meta!(g, "n_cons", Int64(solver.equations.n_cons), rank)
    write_meta!(g, "n_species", Int64(solver.equations.n_species), rank)
    write_meta!(g, "n_levels", Int64(nlevels(solver)), rank)
    write_meta!(g, "n_art", Int64(n_art_fields(root)), rank)
    write_strings!(g, "component_names", solver.equations.component_names, rank)
    write_strings!(g, "metric", [type_name(solver.metric)], rank)
    write_strings!(g, "eos", [type_name(solver.eos)], rank)
    write_strings!(g, "eltype", [string(eltype(Q))], rank)
    # The mutable run state, for the reasons src/io.jl gives: a retry
    # lowers `cfl`, the growth cap and `filter_weight` read `dt_prev`
    # and `rate_prev`, and a boundary face that has switched must not
    # come back unswitched on any rank.
    write_meta!(g, "cfl", Float64(solver.cfl), rank)
    write_meta!(g, "dt_prev", Float64(solver.dt_prev), rank)
    write_meta!(g, "rate_prev", Float64(solver.rate_prev), rank)
    write_meta!(g, "filter_rate_prev", collect(Float64, solver.filter_rate_prev),
                rank)
    write_meta!(g, "switched", switch_codes(root), rank)
    cg = create_group(file, "grid")
    for d in 1:3
        write_meta!(cg, "xyz"[d:d], global_axis(root, d), rank)
    end
    _write_configuration!(file, configuration_record(solver), rank)
    return file
end

# The configuration record as three parallel string datasets, with one
# digest per group as a readable summary (`group fnv1a64 <hex>`); the reader
# compares the entries, not the digests.
function _write_configuration!(file, rec::ConfigurationRecord, rank::Int)
    g = create_group(file, "config")
    write_meta!(g, "version", Int64(rec.version), rank)
    write_strings!(g, "groups", rec.groups, rank)
    write_strings!(g, "paths", rec.paths, rank)
    write_strings!(g, "values", rec.values, rank)
    write_strings!(g, "digests",
                   [group * " fnv1a64 " * digest
                    for (group, digest) in configuration_digests(rec)], rank)
    return file
end

_read_configuration(file) =
    ConfigurationRecord(Int(read(file["config/version"])),
                        String.(read(file["config/groups"])),
                        String.(read(file["config/paths"])),
                        String.(read(file["config/values"])))

# One patch's state under group `name`: `Q` over the patch's node space
# `extent`, and the artificial coefficients beside it when they are
# computed, for the reason src/io.jl gives (the next step is sized from
# them). `ps === nothing` on a rank holding no block of the patch, which
# reaches the (collective) creation and writes nothing.
function _write_state!(file, name::AbstractString, ps, Q, extent, comm::MPI.Comm,
                       n_cons::Int, n_art::Int, ::Type{T}) where {T}
    region = ps === nothing ? nothing : owned_region(ps.decomp)
    # `T` is the state's element type, not Float64: a Float32 solver would
    # otherwise write a widened copy that no longer round-trips bit for bit.
    ranges = ps === nothing ? () : (region_ranges(region)..., 1:n_cons)
    dset = shared_dataset(file, name * "/Q", T, (extent..., n_cons), comm)
    try
        data = nothing
        if ps !== nothing
            o1, o2, o3 = ps.decomp.n_halo_d
            nx, ny, nz = ps.decomp.n_local
            data = Q[o1+1:o1+nx, o2+1:o2+ny, o3+1:o3+nz, :]
        end
        write_block!(dset, ranges, data)
    finally
        close(dset)
    end
    n_art > 0 || return file
    dset = shared_dataset(file, name * "/art", T, (extent..., n_art), comm)
    try
        write_block!(dset, ps === nothing ? () : (region_ranges(region)..., 1:n_art),
                     ps === nothing ? nothing : art_block(ps))
    finally
        close(dset)
    end
    return file
end

# The hierarchy record as named datasets: per level a 6 x n table of tile
# regions (offset, extent) and a 2 x n table of owner ranges, then the
# regrid state. Readable with h5dump, unlike the per-rank checkpoint's flat
# image of the same record.
function _write_record!(file, rec::HierarchyRecord, rank::Int)
    g = create_group(file, "hierarchy")
    write_meta!(g, "np", Int64(rec.np), rank)
    write_meta!(g, "n_levels", Int64(length(rec.levels)), rank)
    for (ℓ, lev) in enumerate(rec.levels)
        lg = create_group(g, "level$ℓ")
        n = length(lev.regions)
        regions = reshape(Int64[v for r in lev.regions for v in (r.offset..., r.extent...)],
                          6, n)
        owners = reshape(Int64[v for o in lev.owners for v in (first(o), last(o))], 2, n)
        # A level with no tiles writes no tables; the reader takes their
        # absence as the empty set.
        n == 0 && continue
        write_meta!(lg, "regions", regions, rank)
        write_meta!(lg, "owners", owners, rank)
    end
    rg = create_group(g, "regrid")
    write_meta!(rg, "tile", Int64(rec.tile), rank)
    write_meta!(rg, "interval", Int64(rec.interval), rank)
    write_meta!(rg, "threshold", Float64(rec.threshold), rank)
    write_meta!(rg, "buffer", Int64(rec.buffer), rank)
    write_meta!(rg, "last_step", Int64(rec.last_step), rank)
    write_meta!(rg, "checks", Int64(rec.checks), rank)
    write_meta!(rg, "streak", Int64(rec.streak), rank)
    write_meta!(rg, "imbalance", Float64(rec.imbalance), rank)
    isempty(rec.created) || write_meta!(rg, "created", Int64[rec.created...], rank)
    return file
end

function _read_record(file)
    g = file["hierarchy"]
    levels = LevelRecord[]
    for ℓ in 1:Int(read(g["n_levels"]))
        if !haskey(g["level$ℓ"], "regions")
            push!(levels, LevelRecord(BlockRegion[], UnitRange{Int}[]))
            continue
        end
        R = read(g["level$ℓ/regions"])
        O = read(g["level$ℓ/owners"])
        push!(levels, LevelRecord(
            [BlockRegion((Int(R[1, k]), Int(R[2, k]), Int(R[3, k])),
                         (Int(R[4, k]), Int(R[5, k]), Int(R[6, k]))) for k in axes(R, 2)],
            [Int(O[1, k]):Int(O[2, k]) for k in axes(O, 2)]))
    end
    rg = g["regrid"]
    created = haskey(rg, "created") ? Int.(read(rg["created"])) : Int[]
    return HierarchyRecord(Int(read(g["np"])), levels, Int(read(rg["tile"])),
                           Int(read(rg["interval"])), Float64(read(rg["threshold"])),
                           Int(read(rg["buffer"])), Int(read(rg["last_step"])),
                           Int(read(rg["checks"])), created, Int(read(rg["streak"])),
                           Float64(read(rg["imbalance"])))
end

function CompactLES.load_checkpoint_hdf5!(solver::Solver, Q, prefix::AbstractString;
                                          allow=())
    CompactLES._multipatch(solver) &&
        error("load_checkpoint_hdf5!: this solver holds a patch layout; pass the " *
              "state vector allocate_state returned")
    path = string(prefix, ".h5")
    # Read is not a write, so every rank may open the file at once even without
    # a parallel build.
    h5open(path, "r") do file
        n_art = _read_ckpt_meta!(file, solver, solver, Q, path, allow)
        _read_state!(file, "state", solver, Q, n_art)
    end
    refresh_primitives!(solver, Q)
    return Q
end

# The refined-hierarchy form: the header checks and the root's block, then
# the hierarchy is brought to the recorded one (`restore_hierarchy!`, which
# rebuilds level 1 on this run's rank count) and every tile this rank then
# holds reads its block from the tile's dataset.
function CompactLES.load_checkpoint_hdf5!(solver::Solver,
                                          states::Vector{<:ConservedState},
                                          prefix::AbstractString; allow=())
    CompactLES._check_hierarchy_layout(solver, "load_checkpoint_hdf5!")
    patches = getfield(solver, :patches)
    path = string(prefix, ".h5")
    h5open(path, "r") do file
        root = PatchSolver(solver, patches[1])
        n_art = _read_ckpt_meta!(file, solver, root, states[1], path, allow)
        _read_state!(file, "state", root, states[1], n_art)
        restore_hierarchy!(solver, states, _read_record(file), path)
        for (ℓ, ti, li) in CompactLES._held_tiles(solver)
            _read_state!(file, "levels/$ℓ/tiles/$ti", PatchSolver(solver, patches[li]),
                         states[li], n_art)
        end
    end
    refresh_primitives!(solver, states)
    return states
end

# The header checks against the root patch `root`, then the mutable run
# state onto `solver`; returns the coefficient field count the blocks carry.
function _read_ckpt_meta!(file, solver::Solver, root, Q, path::AbstractString, allow)
    decomp = root.decomp
    n_cons = solver.equations.n_cons
    fmt = Int(read(file["meta/format"]))
    fmt in (CKPT_FORMAT, CKPT_FORMAT_UNRECORDED) ||
        error("checkpoint format mismatch in $path: file is format $fmt, " *
              "this version writes format $CKPT_FORMAT and reads formats " *
              "$CKPT_FORMAT_UNRECORDED and $CKPT_FORMAT. A file written " *
              "before format $CKPT_FORMAT_UNRECORDED lacks header fields a " *
              "restart depends on and cannot be validated; rerun to " *
              "regenerate it.")
    ng = read(file["meta/n_global"])
    Tuple(Int.(ng)) == Tuple(decomp.n_global) ||
        error("global grid mismatch: file has $(Tuple(Int.(ng))), " *
              "solver has $(Tuple(decomp.n_global))")
    Int(read(file["meta/n_cons"])) == n_cons ||
        error("conserved layout mismatch: file has " *
              "$(Int(read(file["meta/n_cons"]))), solver has $n_cons")
    # `n_cons` alone does not identify the conserved layout: two species
    # sets of the same size agree on it and mean different things.
    nsp = Int(read(file["meta/n_species"]))
    nsp == solver.equations.n_species ||
        error("species count mismatch: file has $nsp, solver has " *
              "$(solver.equations.n_species)")
    _check_level_count(Int(read(file["meta/n_levels"])), solver, path)
    n_art = _check_art_count(Int(read(file["meta/n_art"])), root, path)
    names = String.(read(file["meta/component_names"]))
    names == solver.equations.component_names ||
        error("conserved component mismatch: file has $names, solver has " *
              "$(solver.equations.component_names)")
    stored_metric = String(first(read(file["meta/metric"])))
    stored_metric == type_name(solver.metric) ||
        error("metric mismatch: file has $stored_metric, solver has " *
              "$(type_name(solver.metric))")
    stored_eos = String(first(read(file["meta/eos"])))
    stored_eos == type_name(solver.eos) ||
        error("equation of state mismatch: file has $stored_eos, solver " *
              "has $(type_name(solver.eos))")
    stored_eltype = String(first(read(file["meta/eltype"])))
    stored_eltype == string(eltype(Q)) ||
        error("element type mismatch: file holds $stored_eltype, this " *
              "state array holds $(eltype(Q))")
    # The coordinates carry the domain extent, the origin and any `Stretch`,
    # none of which `n_global` constrains.
    for d in 1:3
        axis_matches(read(file["grid/" * "xyz"[d:d]]), global_axis(root, d)) ||
            error("grid coordinate mismatch on dimension $d: the stored " *
                  "coordinates differ from this solver's, so the domain " *
                  "extent, the origin or a Stretch mapping is not the one " *
                  "the checkpoint was written on")
    end
    # The comparison precedes every write to the solver.
    stored = fmt == CKPT_FORMAT ? _read_configuration(file) : nothing
    _verify_configuration(stored, solver, allow, path)
    solver.t = read(file["meta/t"])
    solver.step = Int(read(file["meta/step"]))
    solver.cfl = read(file["meta/cfl"])
    solver.dt_prev = read(file["meta/dt_prev"])
    solver.rate_prev = read(file["meta/rate_prev"])
    fr = read(file["meta/filter_rate_prev"])
    solver.filter_rate_prev = ntuple(d -> oftype(solver.dt_prev, fr[d]), 3)
    restore_switches!(root, read(file["meta/switched"]), path)
    return n_art
end

# One patch's blocks under group `name`, as `_write_state!` laid them out.
function _read_state!(file, name::AbstractString, ps, Q, n_art::Int)
    decomp = ps.decomp
    region = owned_region(decomp)
    o1, o2, o3 = decomp.n_halo_d
    nx, ny, nz = decomp.n_local
    n_cons = ps.equations.n_cons
    Q[o1+1:o1+nx, o2+1:o2+ny, o3+1:o3+nz, :] .=
        read_region4(file[name * "/Q"], region, n_cons)
    n_art > 0 && set_art_block!(ps, read_region4(file[name * "/art"], region, n_art))
    return Q
end

# --- Field dump, with an XDMF3 sidecar --------------------------------------
#
# Array layout is the whole of the care needed here. Julia is column-major and
# HDF5 is row-major, and HDF5.jl bridges them by reversing the dimension list on
# disk: a Julia (nx, ny, nz) dataset is reported by h5dump as nz x ny x nx.
# XDMF reads in the row-major convention, so a scalar's Dimensions are written
# "NZ NY NX".
#
# Vectors follow from the same rule. XDMF requires the component to vary fastest,
# so the Julia array must be (3, nx, ny, nz), component first, which lands on
# disk as nz x ny x nx x 3 and is declared "NZ NY NX 3". That is the
# interleaved layout `_interior_vector` produces, so the payload is reshaped,
# not permuted.

const XDMF_SCALAR = "Scalar"
const XDMF_VECTOR = "Vector"

_xdmf_dims(n) = string(n[3], " ", n[2], " ", n[1])

# Global coordinate vectors, identical on every rank, so no communication is
# needed to write them. A sliced dimension contributes its single coordinate.
function _coarse_axis(solver::Solver, d, stride, slice)
    n = solver.decomp.n_global[d]
    if slice !== nothing && Int(slice[1]) == d
        return Float64[CompactLES.global_xcoord(solver, d, Int(slice[2]))]
    end
    return Float64[CompactLES.global_xcoord(solver, d, g) for g in 1:stride[d]:n]
end

# What the XDMF description of one frame needs: the `.h5` file name relative to
# the description, the time, the written extent, each field's name and component
# count, and the grid type. The frame's own sidecar and the temporal collection
# are both written from it, so the two cannot describe the same frame
# differently.
struct FrameGrid
    h5name::String
    t::Float64
    nglobal::NTuple{3,Int}
    fields::Vector{Tuple{String,Int}}
    curvilinear::Bool
end

_patch_layout_error(name) =
    error("$name: a patch layout has no shared-file field dump; save_vtk and " *
          "FieldWriter(format = :vtk) write it as one piece per patch under a " *
          "multiblock index")

CompactLES.save_hdf5(solver::Solver, states::Vector{<:ConservedState},
                     prefix::AbstractString; kwargs...) = _patch_layout_error("save_hdf5")

function CompactLES.save_hdf5(solver::Solver, Q, prefix::AbstractString;
                              fields=CompactLES.DEFAULT_VTK_FIELDS, stride=1,
                              slice=nothing)
    write_field_frame(solver, Q, prefix; fields=fields, stride=stride, slice=slice)
    return prefix
end

# The frame writer behind `save_hdf5` and `FieldWriter(format = :hdf5)`, which
# returns the frame's `FrameGrid` for the collection.
write_field_frame(solver::Solver, states::Vector{<:ConservedState},
                  prefix::AbstractString; kwargs...) =
    _patch_layout_error("FieldWriter(format = :hdf5)")

function write_field_frame(solver::Solver, Q, prefix::AbstractString;
                           fields=CompactLES.DEFAULT_VTK_FIELDS, stride=1,
                           slice=nothing)
    decomp = solver.decomp
    comm = decomp.comm
    rank = MPI.Comm_rank(comm)
    st = CompactLES._normalize_stride(stride)
    ranges = CompactLES._output_ranges(solver, st, slice)
    CompactLES._check_output(solver, st, slice, ranges)
    curvilinear = CompactLES._curvilinear(solver)

    # The artificial coefficients are restored after the extraction, so a dump
    # requesting them leaves the integrator's next timestep unchanged.
    entries = CompactLES.preserving_artificial(solver,
            any(CompactLES._wants_artificial, fields)) do
        es = Tuple{String,Int,Vector{Float32}}[]
        CompactLES._prepare_fields!(solver, Q, fields)
        for name in fields
            append!(es, CompactLES.vtk_field_entries(solver, Q, name,
                                                     curvilinear, ranges))
        end
        return es
    end

    nglobal = CompactLES._output_global(solver, st, slice)
    nlocal = ntuple(d -> length(ranges[d]), 3)
    # Coarse-index offset of this rank's block, the same arithmetic the VTK
    # piece extents use. Empty on a rank holding no part of a slice.
    off, _ = CompactLES._output_piece_extent(solver, st, slice, ranges)
    mine = CompactLES._has_output(ranges)
    region = BlockRegion(off, nlocal)
    path = string(prefix, ".h5")
    ensure_output_dir(prefix, comm)

    with_shared_file(path, "w", comm) do file
        if has_parallel() || rank == 0
            g = create_group(file, "meta")
            write_meta!(g, "t", Float64(solver.t), rank)
            write_meta!(g, "step", Int64(solver.step), rank)
            write_meta!(g, "n_global", Int64[nglobal...], rank)
            write_meta!(g, "stride", Int64[st...], rank)
            write_meta!(g, "curvilinear", Int64(curvilinear), rank)
            if !curvilinear
                # The coordinate vectors are global, so every rank holds
                # the same values and none of this needs communication.
                cg = create_group(file, "grid")
                for d in 1:3
                    write_meta!(cg, "xyz"[d:d],
                                _coarse_axis(solver, d, st, slice), rank)
                end
            end
        end
        if curvilinear
            # The dataset is created whether or not this rank holds part of the
            # plane, for the reason the field loop below gives: rank 0 may be one
            # of the ranks with nothing to write, and every rank must reach the
            # creation call under the parallel backend.
            dset = shared_dataset(file, "grid/points", Float64, (3, nglobal...), comm)
            try
                pts = nothing
                if mine
                    # One position per point, component first, so the sidecar
                    # can point XDMF's XYZ geometry straight at it.
                    pts = Array{Float64}(undef, 3, nlocal...)
                    for (kk, k) in enumerate(ranges[3]),
                        (jj, j) in enumerate(ranges[2]),
                        (ii, i) in enumerate(ranges[1])
                        x, y, z = CompactLES._cartesian_position(solver.metric,
                            xcoord(solver, 1, i), xcoord(solver, 2, j),
                            xcoord(solver, 3, k))
                        pts[1, ii, jj, kk] = x
                        pts[2, ii, jj, kk] = y
                        pts[3, ii, jj, kk] = z
                    end
                end
                write_block!(dset, (1:3, region_ranges(region)...), pts)
            finally
                close(dset)
            end
        end
        for (name, ncomp, data) in entries
            dims = ncomp == 1 ? nglobal : (ncomp, nglobal...)
            dset = shared_dataset(file, "fields/" * name, Float32, dims, comm)
            try
                # A rank holding no part of the plane creates the dataset and
                # issues an empty write into it; see the note on `dxpl_mpio`
                # at the head of this file.
                r = region_ranges(region)
                if ncomp == 1
                    write_block!(dset, r, mine ? reshape(data, nlocal) : nothing)
                else
                    write_block!(dset, (1:ncomp, r...),
                                 mine ? reshape(data, ncomp, nlocal...) : nothing)
                end
            finally
                close(dset)
            end
        end
    end

    grid = FrameGrid(basename(path), Float64(solver.t), nglobal,
                     Tuple{String,Int}[(name, ncomp) for (name, ncomp, _) in entries],
                     curvilinear)
    rank == 0 && _write_xdmf(string(prefix, ".xmf"), grid)
    MPI.Barrier(comm)
    return grid
end

const XDMF_HEADER = "<?xml version=\"1.0\" ?>\n<!DOCTYPE Xdmf SYSTEM \"Xdmf.dtd\" []>\n" *
                    "<Xdmf Version=\"3.0\">\n <Domain>\n"
const XDMF_FOOTER = " </Domain>\n</Xdmf>\n"

# One frame's uniform grid, every line prefixed with `indent`.
function _write_grid(io, g::FrameGrid, indent::AbstractString)
    dims = _xdmf_dims(g.nglobal)
    h5name = g.h5name
    line(parts...) = write(io, indent, parts..., "\n")
    line("<Grid Name=\"mesh\" GridType=\"Uniform\">")
    line(" <Time Value=\"", string(g.t), "\"/>")
    if g.curvilinear
        line(" <Topology TopologyType=\"3DSMesh\" Dimensions=\"", dims, "\"/>")
        line(" <Geometry GeometryType=\"XYZ\">")
        line("  <DataItem Dimensions=\"", dims, " 3\" NumberType=\"Float\" ",
             "Precision=\"8\" Format=\"HDF\">", h5name, ":/grid/points</DataItem>")
        line(" </Geometry>")
    else
        line(" <Topology TopologyType=\"3DRectMesh\" Dimensions=\"", dims, "\"/>")
        line(" <Geometry GeometryType=\"VXVYVZ\">")
        for d in 1:3
            line("  <DataItem Dimensions=\"", string(g.nglobal[d]),
                 "\" NumberType=\"Float\" Precision=\"8\" Format=\"HDF\">",
                 h5name, ":/grid/", "xyz"[d:d], "</DataItem>")
        end
        line(" </Geometry>")
    end
    for (name, ncomp) in g.fields
        kind = ncomp == 1 ? XDMF_SCALAR : XDMF_VECTOR
        shape = ncomp == 1 ? dims : string(dims, " ", ncomp)
        line(" <Attribute Name=\"", name, "\" AttributeType=\"", kind,
             "\" Center=\"Node\">")
        line("  <DataItem Dimensions=\"", shape, "\" NumberType=\"Float\" ",
             "Precision=\"4\" Format=\"HDF\">", h5name, ":/fields/", name,
             "</DataItem>")
        line(" </Attribute>")
    end
    line("</Grid>")
    return io
end

function _write_xdmf(path, g::FrameGrid)
    open(path, "w") do io
        write(io, XDMF_HEADER)
        _write_grid(io, g, "  ")
        write(io, XDMF_FOOTER)
    end
    return path
end

# The temporal collection of a `FieldWriter(format = :hdf5)`, rewritten in full
# after every frame, for the reason `_write_pvd` gives, and replaced in one
# rename so that an interruption leaves the previous complete file. Each frame's
# grid is written inline rather than through XInclude of its sidecar, which not
# every XDMF reader resolves. Called on rank 0 only.
function write_xdmf_collection(writer)
    path = string(writer.prefix, ".xmf")
    CompactLES._replace_file(path) do io
        write(io, XDMF_HEADER)
        write(io, "  <Grid Name=\"", basename(writer.prefix), "\" ",
                  "GridType=\"Collection\" CollectionType=\"Temporal\">\n")
        for g in writer.grids
            _write_grid(io, g::FrameGrid, "   ")
        end
        write(io, "  </Grid>\n", XDMF_FOOTER)
    end
    return path
end

end # module
