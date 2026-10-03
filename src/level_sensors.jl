# The artificial coefficients of a tiled level, computed over the level as one
# field rather than tile by tile.
#
# Inside one tile `compute_artificial!` builds each sensor from fields whose
# ghost layers at a face shared with another tile it cannot fill. The strain
# magnitude and the dilatation are computed over the interior, so the δ⁴
# detector clamps its taps at the face, and every sensor is smoothed with the
# closure rows of a field without ghosts. μ* and β* then differ from those of
# one patch spanning both tiles over the last few nodes before the face, by up
# to half of their peak on a converging shock, and the solution carries the
# difference as a checkerboard along the tile lattice. The coefficients of the
# spanning patch need the strain magnitude six nodes into the neighbor, two for
# the detector and four for the smoother, past the halo of four.
#
# The pass below reaches them in stages separated by exchanges over the level's
# same-level records, which no per-tile right-hand side may run (patches.jl):
# the sensed fields without interface ghosts after the gradients, and each
# sensor before every directional smoothing pass, so that the second direction
# reads the neighbor's sensor smoothed along the first, as inside one patch.
# The sensors are held in the coefficient arrays themselves until the last
# stage writes the coefficients over them, and the sensed fields in the tile's
# `sensed_fields`. Each tile's right-hand side then runs with the coefficients
# this leaves on the patch (`compute_rhs!` with `coefficients_current`), and
# computes its gradients again, since the workspace the tiles share holds one
# tile's at a time.
#
# A coarse-fine face keeps the treatment a tile has without the pass, up to
# the order of a sum, under the default detector and smoother. The sensed
# fields repeat their edge value into the ghost layers there, which reproduces
# the δ⁴ detector's clamp to the bit. The sensors take the half-offset mirror,
# which the `:gaussian` smoother's closure rows fold onto, read through the
# smoother's interface rows (`InterfaceSmoothPlans`), so the two differ in the
# order of the sum alone. Under `detector = :d8` the sensed fields take the
# half-offset mirror and the detector's ghost-reading interface rows, and under
# `smoother = :compact` the smoother takes the state filter's interface rows:
# both continue the field as the scheme's own closure rows do but are other
# rows. A face on the domain boundary keeps its own treatment throughout.

# Whether `lev` takes the level-wide pass: a refined level on which some face
# is shared by two tiles, with the artificial properties on. `phases` follows
# from every tile's faces, so every rank holding the level's records agrees,
# and the pass runs no collective a rank without records would miss.
_level_sensors(solver, lev::Level) =
    lev.index > 0 && any(lev.phases) && solver.art.enabled

_strain_sensed(art::ArtificialProperties) =
    art.mu_sensor === :strain || art.beta_sensor === :strain ||
    art.beta_sensor === :gated_strain
_dilatation_sensed(art::ArtificialProperties) =
    art.beta_sensor === :dilatation || art.beta_sensor === :ungated_dilatation

"""
    _sensed_field_count(art, tile) -> Int

The number of sensed fields a tile of a level of lattice edge `tile` holds for
the level's artificial-property pass (`Patch.sensed_fields`): the strain
magnitude, which `scalar_field` reads whichever sensors are selected, and the
dilatation where β* is built from it, in that order. Zero on a level of one
patch (`tile = 0`), which never takes the pass, and with the artificial
properties off.
"""
function _sensed_field_count(art::ArtificialProperties, tile::Int)
    (tile > 0 && art.enabled) || return 0
    return 1 + Int(_dilatation_sensed(art))
end

# The coefficient arrays of patch `p` that hold the level's sensors between
# detection and the coefficients: the μ* sensor in `mu_art` unless β* shares
# the strain sensor, the β* sensor in `beta_art`, the internal-energy sensor in
# `kappa_art`, and the species sensor in `D_art[1]` where one diffusivity
# serves every species or in each `D_art[k]` under `:fickian`.
function _sensor_arrays(p, art::ArtificialProperties, n_species::Int)
    out = empty(p.Y)
    shared_strain = art.mu_sensor === :strain && _strain_beta(art)
    shared_strain || push!(out, p.mu_art)
    push!(out, p.beta_art)
    push!(out, p.kappa_art)
    if n_species > 1
        _shared_species_diffusivity(art, n_species) ? push!(out, p.D_art[1]) :
                                                      append!(out, p.D_art)
    end
    return out
end

_strain_beta(art::ArtificialProperties) =
    art.beta_sensor === :strain || art.beta_sensor === :gated_strain

# The right-hand sides of a level that takes the pass (`_level_rhs!`): the
# boundary conditions on every tile, the pass, then each unit's right-hand side
# with the coefficients and the primitives the pass leaves. A stacked level is
# device storage, which the ledger does not sweep, so it takes no hooks.
function _sensor_level_rhs!(solver::Solver, lev::Level, states, dQs, prepared::Bool,
                            enforce::Bool)
    patches = getfield(solver, :patches)
    stacked = !isempty(lev.stacks)
    if enforce
        for pi in lev.patches
            stacked || _ledger_open!(solver, states, pi)
            apply_bcs!(PatchSolver(solver, patches[pi]), states[pi])
            stacked || _ledger!(solver, states, :wall_enforce, pi)
        end
    end
    _level_artificial!(solver, lev, states, prepared)
    if stacked
        for st in lev.stacks
            compute_rhs!(PatchSolver(solver, st.patch), _stack_state(st, states),
                         _stack_state(st, dQs), true, true)
        end
    else
        for pi in lev.patches
            compute_rhs!(PatchSolver(solver, patches[pi]), states[pi], dQs[pi],
                         true, true)
            _ledger_faces!(solver, pi)
        end
    end
    return nothing
end

# One call of `f(ps, Q)` per evaluation unit of the level: each tile on the
# host, each stack of tiles on a device backend (`TileStack`), as the
# right-hand side runs.
function _each_unit(f::F, solver::Solver, lev::Level, states) where {F}
    patches = getfield(solver, :patches)
    if isempty(lev.stacks)
        for pi in lev.patches
            f(PatchSolver(solver, patches[pi]), states[pi])
        end
    else
        for st in lev.stacks
            f(PatchSolver(solver, st.patch), _stack_state(st, states))
        end
    end
    return nothing
end

"""
    _level_artificial!(solver, lev, states, prepared, held = nothing)

The artificial coefficients of every tile of `lev` that this rank holds,
computed as one patch spanning the level would compute them, and written into
each tile's `mu_art`, `beta_art`, `kappa_art` and `D_art`. The primitives and
the velocity gradients are computed here, the primitives skipped when
`prepared` says the caller has refreshed them; on return the primitives are
current for the tiles' right-hand sides, and the compression switch of a gated
β* is left to them (`compute_rhs!`). A `Dict` `held` receives, by patch index,
a copy of each tile's smoothed internal-energy sensor, which the shared
workspace holds for one unit only (`_output_level_artificial!`).

Entered by every rank holding a tile of `lev`, at the same point: the
exchanges are point-to-point over the level's records, and each tile's line
solves and halo exchanges are collective over its own communicator.
"""
function _level_artificial!(solver::Solver, lev::Level, states, prepared::Bool,
                            held=nothing)
    art = solver.art
    n_species = solver.equations.n_species
    _each_unit(solver, lev, states) do ps, Q
        compute_primitives_and_gradients!(ps, Q, prepared)
        _sensed_fields!(ps)
    end
    _sync_sensor_fields!(solver, lev, p -> p.sensed_fields, art.detector !== :d8, 1:3)
    _each_unit(solver, lev, states) do ps, Q
        _level_sensor_detect!(ps, Q)
    end
    sensors(p) = _sensor_arrays(p, art, n_species)
    n_global = solver.n_global
    for d in 1:3
        n_global[d] > 1 || continue
        _sync_sensor_fields!(solver, lev, sensors, false, d:d)
        _each_unit(solver, lev, states) do ps, Q
            for f in sensors(ps)
                smooth_along!(ps.tmp_a, f, ps, d, 1, true)
                copy_interior!(f, ps.tmp_a, ps.decomp)
            end
        end
    end
    if held !== nothing
        patches = getfield(solver, :patches)
        for pi in lev.patches
            held[pi] = _dense_copy(patches[pi].kappa_art)
        end
    end
    _each_unit(solver, lev, states) do ps, Q
        _level_coefficients!(ps)
    end
    return nothing
end

# The sensed fields without interface ghosts, over the interior, from the
# velocity gradients the shared workspace holds for this unit.
function _sensed_fields!(ps)
    art = ps.art
    decomp = ps.decomp
    o1, o2, o3 = decomp.n_halo_d
    nx, ny, nz = decomp.n_local
    sf = ps.sensed_fields
    grad_u = ps.field_tuples.grad_u
    pointwise!(_strain_mag_point!, sf[1], nx, ny, nz, sf[1], grad_u, o1, o2, o3)
    _dilatation_sensed(art) &&
        pointwise!(_dilatation_point!, sf[2], nx, ny, nz, sf[2], grad_u, o1, o2, o3)
    return ps
end

# Every sensor of the unit before smoothing, each into the coefficient array
# `_sensor_arrays` names, through the detectors `compute_artificial!` applies
# but reading the ghost layers of every field at every interface face.
function _level_sensor_detect!(ps, Q)
    art = ps.art
    decomp = ps.decomp
    o1, o2, o3 = decomp.n_halo_d
    nx, ny, nz = decomp.n_local
    n_species, _, i_energy, (m1, m2, m3) = equation_layout(ps.equations)
    sf = ps.sensed_fields
    _strain_sensed(art) &&
        _detect!(_strain_beta(art) ? ps.beta_art : ps.mu_art, sf[1], ps, 2, true)
    if art.mu_sensor === :velocity
        vel = (ps.u, ps.v, ps.w)
        fill!(ps.mu_art, 0)
        for j in 1:3
            detect_sum!(ps.mu_art, vel[j], ps, 1; accumulate=true,
                        parity=ntuple(d -> vel_parity(ps, d, j), 3),
                        wall_parity=ntuple(d -> d == j ? -1 : 1, 3), ghosts=true)
        end
    end
    _dilatation_sensed(art) && _detect!(ps.beta_art, sf[2], ps, 2, true)
    nxf, nyf, nzf = padded_extent(decomp)
    pointwise!(_internal_energy_point!, ps.tmp_a, nxf, nyf, nzf,
               ps.tmp_a, Q, ps.rho, m1, m2, m3, i_energy)
    exchange_halos!(ps.tmp_a, decomp)
    _detect!(ps.kappa_art, ps.tmp_a, ps, 1, true)
    n_species > 1 || return ps
    h_bound, inv_n = _species_bound_length(decomp, ps.h, art.C_D)
    a1, a2, a3 = decomp.active
    ih1, ih2, ih3 = ps.inv_h
    if _shared_species_diffusivity(art, n_species)
        _bulk_species_sensor!(ps.D_art[1], ps, art.C_D, art.C_Y, h_bound, inv_n,
                              ih1, ih2, ih3, a1, a2, a3, art.Y_tolerance)
        return ps
    end
    for sp in 1:n_species
        D = ps.D_art[sp]
        species_detect_sum!(D, ps.Y[sp], ps)
        pointwise!(_species_bound_point!, D, nx, ny, nz, D, ps.Y[sp], art.C_D,
                   art.C_Y, h_bound, inv_n, ih1, ih2, ih3, a1, a2, a3,
                   art.Y_tolerance, o1, o2, o3)
    end
    return ps
end

# The coefficients from the smoothed sensors, in place, by the per-point bodies
# `compute_artificial!` uses. Each body reads its sensor at a point before it
# writes that point, so a coefficient array may hold its own sensor; the
# shared species sensor in `D_art[1]` is read into the other species first.
# The workspace's `strain_mag` and `sensor` take what `compute_artificial!`
# leaves in them, the strain magnitude and the smoothed internal-energy
# sensor, for `scalar_field`, and name the unit as their writer; the last
# unit's stay.
function _level_coefficients!(ps)
    art = ps.art
    decomp = ps.decomp
    o1, o2, o3 = decomp.n_halo_d
    nx, ny, nz = decomp.n_local
    copy_interior!(ps.strain_mag, ps.sensed_fields[1], decomp)
    copy_interior!(ps.sensor, ps.kappa_art, decomp)
    _mark_sensors!(ps)
    mu, beta, rho = ps.mu_art, ps.beta_art, ps.rho
    if art.mu_sensor === :strain && _strain_beta(art)
        pointwise!(_mu_beta_point!, mu, nx, ny, nz, mu, beta, rho, beta,
                   art.C_mu, art.C_beta, o1, o2, o3)
    else
        pointwise!(_rho_sensor_point!, mu, nx, ny, nz, mu, rho, mu, art.C_mu,
                   o1, o2, o3)
        pointwise!(_rho_sensor_point!, beta, nx, ny, nz, beta, rho, beta,
                   art.C_beta, o1, o2, o3)
    end
    kappa = ps.kappa_art
    pointwise!(_kappa_point!, kappa, nx, ny, nz, kappa, ps.eos, rho, ps.c,
               ps.T_ion, ps.cp_mix, kappa, art.C_kappa, o1, o2, o3)
    n_species = ps.equations.n_species
    n_species > 1 || return ps
    D = ps.D_art
    shared = _shared_species_diffusivity(art, n_species)
    for sp in n_species:-1:1
        src = shared ? D[1] : D[sp]
        pointwise!(_species_diffusivity_point!, src, nx, ny, nz, D[sp], ps.c, src,
                   o1, o2, o3)
    end
    return ps
end

# --- Output --------------------------------------------------------------------
#
# A diagnostic or a field writer that reports the coefficients from a state
# vector recomputes them from that state, as the single-patch form does, and
# restores the integrator's afterwards (`preserving_artificial`). On a level
# that takes the pass the recomputation is the pass itself, so the output
# carries the coefficients the level's next right-hand side computes from the
# same state, not those of each tile alone.

"""
    _output_level_artificial!(solver, states) -> Dict{Int,Any}

Run the level-wide pass on `states` for every level that takes it, before an
output walks the patches, and return the held sensors by patch index (see
`_level_artificial!`); a patch absent from the result recomputes its
coefficients alone (`_output_artificial!`). The coefficient arrays are
overwritten, so the caller holds this inside `preserving_artificial`. The
exchange time is not charged to the step's `wall_wait`. Every rank must call
it at the same point, before any patch's own derived-field pass.
"""
function _output_level_artificial!(solver::Solver, states)
    held = Dict{Int,Any}()
    solver.art.enabled || return held
    wait = solver.wall_wait
    for lev in getfield(solver, :levels)
        _level_sensors(solver, lev) &&
            _level_artificial!(solver, lev, states, false, held)
    end
    solver.wall_wait = wait
    return held
end

"""
    _output_artificial!(ps, Q, sensor)

The coefficients and the sensor scratch of one patch for an output, after the
caller's gradient pass on it: `compute_artificial!` where `sensor` is
`nothing`, and otherwise what the patch's right-hand side adds to the level
pass, the compression switch of a gated β*, with the strain magnitude and the
held `sensor` copied into the shared workspace for `scalar_field`.
"""
function _output_artificial!(ps, Q, sensor)
    sensor === nothing && return compute_artificial!(ps, Q)
    _gated(ps.art) && gate_beta!(ps)
    copy_interior!(ps.strain_mag, ps.sensed_fields[1], ps.decomp)
    copy_interior!(ps.sensor, sensor, ps.decomp)
    _mark_sensors!(ps)
    return ps
end

# --- Exchange ------------------------------------------------------------------

"""
    _sync_sensor_fields!(solver, lev, fields_of, replicate, dims)

Fill the ghost layers along each dimension of `dims` of the fields
`fields_of(patch)` of every tile of `lev` this rank holds: the rank halos;
the neighbor's interior at a shared face, over the level's same-level records
of that dimension, taken in dimension order as the state's are
(`_sync_level_records!`); and at a coarse-fine face the tile's own edge,
repeated when `replicate` and mirrored about the half-offset point otherwise.
Every reader of these layers is a line operator along one dimension at the
tile's interior transverse nodes, so no edge or corner ghost is filled.
Point-to-point over the level's communicator and collective over each tile's.
"""
function _sync_sensor_fields!(solver, lev::Level, fields_of::F, replicate::Bool,
                              dims) where {F}
    patches = getfield(solver, :patches)
    comm = lev.level_comm.comm
    t0 = time_ns()
    for d in dims
        lev.phases[d] && _exchange_field_ghosts!(patches, fields_of, comm,
                                                 lev.ghost_sends[d], lev.ghost_recvs[d])
        for pi in lev.patches
            p = patches[pi]
            fields = fields_of(p)
            isempty(fields) || !p.decomp.active[d] ||
                exchange_dim_batch!(fields, p.decomp, d)
        end
    end
    _wait!(solver, t0)
    for pi in lev.patches
        p = patches[pi]
        _fill_coarse_fine!(fields_of(p), p, replicate, dims)
    end
    return nothing
end

# The coarse-fine faces of patch `p` along `dims`, at the ranks owning them,
# filled for every field.
function _fill_coarse_fine!(fields, p, replicate::Bool, dims)
    decomp = p.decomp
    for d in dims
        decomp.active[d] || continue
        lo = parent_fed(p.bcs[d][1])
        hi = parent_fed(p.bcs[d][2])
        (lo || hi) || continue
        for f in fields
            replicate ? _edge_fill!(f, decomp, d, lo, hi) :
                        fold_fill!(f, decomp, d, lo, hi, 1)
        end
    end
    return fields
end

"""
    _edge_fill!(f, decomp, d, lo, hi)

Repeat the edge node of `f` along `d` into every halo layer beyond the ends
`lo` and `hi` select, at the ranks owning them: the continuation the δ⁴
detector's clamp reads. The layout and the stacked form are `fold_fill!`'s.
"""
function _edge_fill!(f, decomp::Decomp, d::Int, lo::Bool, hi::Bool)
    pad = decomp.n_halo_d[d]
    n = decomp.n_local[d]
    np = _stack_of(f) === nothing ? size(f) : padded_extent(decomp)
    for (end_lo, owns) in ((true, lo && decomp.sub_rank[d] == 0),
                           (false, hi && decomp.sub_rank[d] == decomp.sub_size[d] - 1))
        owns || continue
        if d == 3
            pointwise!(_edge_fill_z_point!, f, np[1], np[2], pad,
                       f, pad, n, np[3], end_lo)
        else
            pointwise!(_edge_fill_point!, f, pad, np[d == 1 ? 2 : 1], np[3],
                       f, d, pad, n, end_lo)
        end
    end
    return f
end

@inline function _edge_fill_point!(f, d, pad, n, lo, i, j, k)
    @inbounds if lo
        if d == 1
            f[pad-i+1, j, k] = f[pad+1, j, k]
        else
            f[j, pad-i+1, k] = f[j, pad+1, k]
        end
    else
        if d == 1
            f[pad+n+i, j, k] = f[pad+n, j, k]
        else
            f[j, pad+n+i, k] = f[j, pad+n, k]
        end
    end
    return nothing
end

@inline function _edge_fill_z_point!(f, pad, n, n3, lo, i, j, k)
    layer = (k - 1) % n3 + 1
    base = k - layer
    @inbounds if lo
        f[i, j, base+pad-layer+1] = f[i, j, base+pad+1]
    else
        f[i, j, base+pad+n+layer] = f[i, j, base+pad+n]
    end
    return nothing
end

# The ghost refill of `_exchange_ghosts!` (patches.jl) for a list of scalar
# fields per patch, every field of a record in one message. The record buffers
# are sized for the conserved components, at least as many as the fields any
# caller passes, and only their leading part is sent.
function _exchange_field_ghosts!(patches, fields_of::F, comm::MPI.Comm, sends,
                                 recvs) where {F}
    (isempty(recvs) && isempty(sends)) && return nothing
    me = MPI.Comm_rank(comm)
    reqs = MPI.Request[]
    for r in recvs
        r.partner == me && continue
        n = length(fields_of(patches[r.patch])) * prod(length.(r.mine))
        push!(reqs, MPI.Irecv!(view(r.buf, 1:n), comm; source=r.partner, tag=r.tag))
    end
    for s in sends
        src = fields_of(patches[s.patch])
        if s.partner == me
            dst = fields_of(patches[s.partner_patch])
            for c in eachindex(src)
                _copy_field_block!(dst[c], s.theirs, src[c], s.mine)
            end
        else
            n = _pack_fields!(s.buf, src, s.mine)
            push!(reqs, MPI.Isend(view(s.buf, 1:n), comm; dest=s.partner, tag=s.tag))
        end
    end
    MPI.Waitall(reqs)
    for r in recvs
        r.partner == me && continue
        _unpack_fields!(fields_of(patches[r.patch]), r.buf, r.mine)
    end
    return nothing
end

# The scalar forms of `_pack!`, `_unpack!` and `_copy_block!`: host storage
# loops, device storage stages each block through one contiguous device array,
# with the same element order.
function _pack_fields!(buf::AbstractVector, fields, r::NTuple{3,UnitRange{Int}})
    m = prod(length.(r))
    n = m * length(fields)
    if !_device_path(first(fields))
        idx = 1
        @inbounds for f in fields, k in r[3], j in r[2], i in r[1]
            buf[idx] = f[i, j, k]
            idx += 1
        end
        return n
    end
    dsend = _device_send_stage(parent(first(fields)), n)
    for (c, f) in enumerate(fields)
        reshape(view(dsend, (c - 1) * m + 1:c * m), length.(r)) .= view(f, r...)
    end
    _tracked_copy!(buf, 1, dsend, 1, n)
    return n
end

function _unpack_fields!(fields, buf::AbstractVector, r::NTuple{3,UnitRange{Int}})
    m = prod(length.(r))
    n = m * length(fields)
    if !_device_path(first(fields))
        idx = 1
        @inbounds for f in fields, k in r[3], j in r[2], i in r[1]
            f[i, j, k] = buf[idx]
            idx += 1
        end
        return fields
    end
    drecv = _device_send_stage(parent(first(fields)), n)
    _tracked_copy!(drecv, 1, buf, 1, n)
    for (c, f) in enumerate(fields)
        view(f, r...) .= reshape(view(drecv, (c - 1) * m + 1:c * m), length.(r))
    end
    return fields
end

function _copy_field_block!(dst, rdst::NTuple{3,UnitRange{Int}}, src,
                            rsrc::NTuple{3,UnitRange{Int}})
    if !_device_path(dst)
        @inbounds for (kd, ks) in zip(rdst[3], rsrc[3]), (jd, js) in zip(rdst[2], rsrc[2])
            for (id, is) in zip(rdst[1], rsrc[1])
                dst[id, jd, kd] = src[is, js, ks]
            end
        end
        return dst
    end
    view(dst, rdst...) .= view(src, rsrc...)
    return dst
end
