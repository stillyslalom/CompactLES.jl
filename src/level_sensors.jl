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
# this leaves on the patch (`compute_rhs!` with `coefficients_current`) and
# with the velocity gradients the pass computed, which each tile holds in
# arrays of its own (`_own_gradients`) or in its block of a stack's. The
# smoothed volume-fraction gradients of the sharpening flux take the same
# stages after the coefficients (`_level_sharpening_gradients!`).
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
    _sensed_field_count(art, tile, n_species) -> Int
    _sensed_field_count(solver, tile) -> Int

The number of sensed fields a tile of a level of lattice edge `tile` holds for
the level's artificial-property pass (`Patch.sensed_fields`): the strain
magnitude, which `scalar_field` reads whichever sensors are selected, and the
dilatation where β* is built from it, in that order; then, where the
sharpening flux is on, the smoothed volume-fraction gradients of its first
n_species − 1 species, three per species (`_sharpen_gradient`). Zero on a
level of one patch (`tile = 0`), which never takes the pass, and with the
artificial properties off.
"""
function _sensed_field_count(art::ArtificialProperties, tile::Int, n_species::Int)
    (tile > 0 && art.enabled) || return 0
    sharpened = _sharpening(art, n_species) ? 3 * (n_species - 1) : 0
    return _sensor_field_count(art) + sharpened
end
_sensed_field_count(solver, tile::Int) =
    _sensed_field_count(solver.art, tile, solver.equations.n_species)

# The sensed fields the sensors are detected from, which lead the list.
_sensor_field_count(art::ArtificialProperties) = 1 + Int(_dilatation_sensed(art))

# The smoothed gradient along `d` of the volume fraction of species `sp` that
# a tile of a level taking the pass holds for its sharpening flux, the slot
# kept along a collapsed dimension as well.
_sharpen_gradient(p, art::ArtificialProperties, sp::Int, d::Int) =
    p.sensed_fields[_sensor_field_count(art) + 3 * (sp - 1) + d]

# The coefficient arrays of patch `p` that hold the level's sensors between
# detection and the coefficients: the μ* sensor in `mu_art` unless β* shares
# the strain sensor, the β* sensor in `beta_art`, the internal-energy sensor in
# `kappa_art`, and the species sensor in `D_art[1]` where one diffusivity
# serves every species or in each `D_art[k]` under `:fickian`.
function _sensor_arrays(p, art::ArtificialProperties, n_species::Int)
    out = empty(p.Y)
    _foreach_sensor_array(f -> push!(out, f), p, art, n_species)
    return out
end

# `g(f)` for each array of `_sensor_arrays`, in its order, without collecting
# them.
@inline function _foreach_sensor_array(g::G, p, art::ArtificialProperties,
                                       n_species::Int) where {G}
    shared_strain = art.mu_sensor === :strain && _strain_beta(art)
    shared_strain || g(p.mu_art)
    g(p.beta_art)
    g(p.kappa_art)
    if n_species > 1
        if _shared_species_diffusivity(art, n_species)
            g(p.D_art[1])
        else
            foreach(g, p.D_art)
        end
    end
    return nothing
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
        if stacked
            for pi in lev.patches
                _tile_bcs!(solver, _cold(patches[pi]), states, pi)
            end
        else
            _foreach_tile(_enforced_tile!, solver, lev, states)
        end
    end
    _level_artificial!(solver, lev, states, prepared)
    # A setup constant, the same on every rank holding the level.
    _sharpening(solver) && _level_sharpening_gradients!(solver, lev, states)
    if stacked
        for st in lev.stacks
            compute_rhs!(PatchSolver(solver, st.patch), _stack_state(st, states),
                         _stack_state(st, dQs), true, true)
        end
    else
        _foreach_tile(_current_tile_rhs!, solver, lev, states, dQs)
    end
    return nothing
end

# One tile's parts of `_sensor_level_rhs!`, with their ledger hooks.
function _enforced_tile!(solver, pi::Int, states)
    _ledger_open!(solver, states, pi)
    _tile_bcs!(solver, _cold(getfield(solver, :patches)[pi]), states, pi)
    _ledger!(solver, states, :wall_enforce, pi)
    return nothing
end
function _current_tile_rhs!(solver, pi::Int, states, dQs)
    _tile_rhs!(solver, _cold(getfield(solver, :patches)[pi]), states, dQs, pi, true,
               true)
    _ledger_faces!(solver, pi)
    return nothing
end

# One call of `f(ps, Q, arg)` per evaluation unit of the level: each tile on
# the host, each stack of tiles on a device backend (`TileStack`), as the
# right-hand side runs. `f` is a function, not a closure, and `arg` a value
# already on the heap or one Julia keeps boxed (a `Bool`, a small `Int`, a
# vector), so that the call on each tile, dynamic since `solver.patches` holds
# patches of several types, allocates nothing: a capturing closure is boxed at
# every such call.
function _each_unit(f::F, solver::Solver, lev::Level, states, arg=nothing) where {F}
    if isempty(lev.stacks)
        _foreach_tile(_tile_unit!, solver, lev, f, states, arg)
    else
        for st in lev.stacks
            f(PatchSolver(solver, st.patch), _stack_state(st, states), arg)
        end
    end
    return nothing
end

# `f(ps, Q, arg)` for tile `p`, behind a barrier on its concrete type: built
# here, the `PatchSolver` and the state wrapper are not boxed as arguments of
# the dynamic call. Such a barrier leaves `p` undeclared and is reached through
# `_cold`. An argument inferred as the abstract `Patch` is covered by the one
# method, so Julia invokes its specialization on `Patch` statically, and every
# field read of `p` in it is dynamic and boxes what it reads; a declared
# `p::Patch` has the same effect at a dynamic call, whose specialization Julia
# then widens to the declared type.
_unit_call(f::F, solver, p, states, pi::Int, arg) where {F} =
    f(PatchSolver(solver, p), states[pi], arg)
_tile_unit!(solver, pi::Int, f::F, states, arg) where {F} =
    _unit_call(f, solver, _cold(getfield(solver, :patches)[pi]), states, pi, arg)

# The boundary conditions and the right-hand side of tile `p`, `states[pi]`,
# behind the same barrier.
_tile_bcs!(solver, p, states, pi::Int) =
    apply_bcs!(PatchSolver(solver, p), states[pi])
_tile_rhs!(solver, p, states, dQs, pi::Int, prepared::Bool, current::Bool) =
    compute_rhs!(PatchSolver(solver, p), states[pi], dQs[pi], prepared, current)

# The fields `select(p, solver)` of each tile of `lev` this rank holds, indexed
# by patch index and unassigned at the indices of other patches. Gathered once
# per pass, so that the exchanges and fills read each list at its concrete
# type.
function _level_field_lists(solver::Solver, lev::Level, select::F) where {F}
    patches = getfield(solver, :patches)
    held = [select(patches[pi], solver) for pi in lev.patches]
    lists = similar(held, length(patches))
    for (k, pi) in enumerate(lev.patches)
        lists[pi] = held[k]
    end
    return lists
end

_sensed_list(p, solver) = p.sensed_fields
_sensor_list(p, solver) = _sensor_arrays(p, solver.art, solver.equations.n_species)
_sharpen_list(p, solver) =
    [_sharpen_gradient(p, solver.art, sp, d)
     for sp in 1:(solver.equations.n_species - 1) for d in 1:3 if p.decomp.active[d]]

# The units of `_level_artificial!` and `_level_sharpening_gradients!`.
function _unit_sensed!(ps, Q, prepared::Bool)
    compute_primitives_and_gradients!(ps, Q, prepared, true)
    _sensed_fields!(ps)
    return nothing
end
_unit_detect!(ps, Q, ::Nothing) = (_level_sensor_detect!(ps, Q); nothing)
_unit_coefficients!(ps, Q, ::Nothing) = (_level_coefficients!(ps); nothing)
function _unit_smooth_sensors!(ps, Q, d::Int)
    _foreach_sensor_array(ps, ps.art, ps.equations.n_species) do f
        smooth_along!(ps.tmp_a, f, ps, d, 1, true)
        copy_interior!(f, ps.tmp_a, ps.decomp)
    end
    return nothing
end
function _unit_sharpen_gradients!(ps, Q, ::Nothing)
    art = ps.art
    for sp in 1:(ps.equations.n_species - 1)
        _volume_fraction_gradients!(ntuple(d -> _sharpen_gradient(ps, art, sp, d), 3),
                                    ps, sp)
    end
    return nothing
end
function _unit_smooth_gradients!(ps, Q, d::Int)
    for f in _sharpen_list(ps, ps)
        smooth_along!(ps.tmp_a, f, ps, d, 1, true)
        copy_interior!(f, ps.tmp_a, ps.decomp)
    end
    return nothing
end

"""
    _level_artificial!(solver, lev, states, prepared, held = nothing)

The artificial coefficients of every tile of `lev` that this rank holds,
computed as one patch spanning the level would compute them, and written into
each tile's `mu_art`, `beta_art`, `kappa_art` and `D_art`. The primitives and
the velocity gradients are computed here, the primitives skipped when
`prepared` says the caller has refreshed them; on return the primitives and
each tile's velocity gradients are current for the tiles' right-hand sides,
which read them (`_gradient_step!`), and the compression switch of a gated
β* is left to them (`compute_rhs!`). A `Dict` `held` receives, by patch index,
a copy of each tile's smoothed internal-energy sensor, which the shared
workspace holds for one unit only (`_output_level_artificial!`).

Entered by every rank holding a tile of `lev`, at the same point: the
exchanges are point-to-point over the level's records, and each tile's line
solves and halo exchanges are collective over its own communicator.
"""
function _level_artificial!(solver::Solver, lev::Level, states, prepared::Bool,
                            held=nothing)
    _each_unit(_unit_sensed!, solver, lev, states, prepared)
    _sync_sensor_fields!(solver, lev, _level_field_lists(solver, lev, _sensed_list),
                         solver.art.detector !== :d8, 1:3)
    _each_unit(_unit_detect!, solver, lev, states)
    sensors = _level_field_lists(solver, lev, _sensor_list)
    n_global = solver.n_global
    for d in 1:3
        n_global[d] > 1 || continue
        _sync_sensor_fields!(solver, lev, sensors, false, d:d)
        _each_unit(_unit_smooth_sensors!, solver, lev, states, d)
    end
    if held !== nothing
        patches = getfield(solver, :patches)
        for pi in lev.patches
            held[pi] = _dense_copy(patches[pi].kappa_art)
        end
    end
    _each_unit(_unit_coefficients!, solver, lev, states)
    return nothing
end

"""
    _level_sharpening_gradients!(solver, lev, states)

The smoothed volume-fraction gradients the sharpening flux builds its pair
normals and its gate from (`_sharpening_fluxes!`), computed over the whole of
`lev` as the sensors are: each tile differences the fractions with its own
interface ghosts, and each directional smoothing pass reads the neighbor's
gradients, smoothed along the previous directions, across a shared face. The
results stay in each tile's `sensed_fields` (`_sharpen_gradient`), and the
tile's right-hand side uses them instead of smoothing its own when its
coefficients are current. Runs after `_level_artificial!`, whose primitives it
reads, on the same ranks at the same point.
"""
function _level_sharpening_gradients!(solver::Solver, lev::Level, states)
    _each_unit(_unit_sharpen_gradients!, solver, lev, states)
    # The active gradients of every species in one exchange per direction
    # (`_sharpen_list`): at most 3(N − 1) fields, within the n_cons = N + 4 a
    # record's buffer holds for N ≤ `SHARPEN_MAX_SPECIES`.
    gradients = _level_field_lists(solver, lev, _sharpen_list)
    n_global = solver.n_global
    for d in 1:3
        n_global[d] > 1 || continue
        _sync_sensor_fields!(solver, lev, gradients, false, d:d)
        _each_unit(_unit_smooth_gradients!, solver, lev, states, d)
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
    _sync_sensor_fields!(solver, lev, lists, replicate, dims)

Fill the ghost layers along each dimension of `dims` of the fields `lists[i]`
of every tile `i` of `lev` this rank holds (`_level_field_lists`): the rank
halos; the neighbor's interior at a shared face, over the level's same-level records
of that dimension, taken in dimension order as the state's are
(`_sync_level_records!`); and at a coarse-fine face the tile's own edge,
repeated when `replicate` and mirrored about the half-offset point otherwise.
Every reader of these layers is a line operator along one dimension at the
tile's interior transverse nodes, so no edge or corner ghost is filled.
Point-to-point over the level's communicator and collective over each tile's.
"""
function _sync_sensor_fields!(solver, lev::Level, lists, replicate::Bool, dims)
    patches = getfield(solver, :patches)
    comm = lev.level_comm.comm
    t0 = time_ns()
    for d in dims
        lev.phases[d] && _exchange_field_ghosts!(lists, comm, lev.ghost_sends[d],
                                                 lev.ghost_recvs[d])
        for pi in lev.patches
            _exchange_fields_along!(lists[pi], _cold(patches[pi]), d)
        end
    end
    _wait!(solver, t0)
    for pi in lev.patches
        _fill_coarse_fine!(lists[pi], patches[pi], replicate, first(dims), last(dims))
    end
    return nothing
end

# The rank halos of `fields` along `d` on patch `p`, behind a barrier on the
# patch's type (`_unit_call`): its decomposition, read from an abstractly typed
# patch, would be boxed.
function _exchange_fields_along!(fields, p, d::Int)
    isempty(fields) || !p.decomp.active[d] || exchange_dim_batch!(fields, p.decomp, d)
    return nothing
end

# The coarse-fine faces of patch `p` along dimensions `dlo:dhi`, at the ranks
# owning them, filled for every field.
function _fill_coarse_fine!(fields, p, replicate::Bool, dlo::Int, dhi::Int)
    decomp = p.decomp
    for d in dlo:dhi
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
# fields per patch, `lists[i]` for patch `i`, every field of a record in one
# message. The record buffers are sized for the conserved components, at least
# as many as the fields any caller passes, and only their leading part is sent.
function _exchange_field_ghosts!(lists, comm::MPI.Comm, sends, recvs)
    (isempty(recvs) && isempty(sends)) && return nothing
    me = MPI.Comm_rank(comm)
    reqs = MPI.Request[]
    for r in recvs
        r.partner == me && continue
        n = length(lists[r.patch]) * prod(length.(r.mine))
        push!(reqs, MPI.Irecv!(view(r.buf, 1:n), comm; source=r.partner, tag=r.tag))
    end
    for s in sends
        src = lists[s.patch]
        if s.partner == me
            dst = lists[s.partner_patch]
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
        _unpack_fields!(lists[r.patch], r.buf, r.mine)
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
