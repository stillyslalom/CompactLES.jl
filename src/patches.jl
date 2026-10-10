# Patch abstraction: per-patch state split out of the solver configuration.
#
# A Patch is a logically rectangular block of uniform resolution carrying its
# own communicator, decomposition, operator plans, folds, and every persistent
# field array, typed A <: AbstractArray{T,3} against the storage backend. The
# scratch of a right-hand-side evaluation is not per patch: it lives on an
# RHSWorkspace shared by the rank's patches of equal padded extent, since a
# rank advances them in sequence (see "Ownership and freshness" below for the
# three fields that are read after the evaluation that filled them). The
# Solver in solver.jl keeps the physics configuration and the run clock,
# plus the vector of this rank's patches in global order. In the common case of
# one patch spanning all ranks, that patch's decomposition is built over
# MPI.COMM_WORLD, and `Base.getproperty` on Solver forwards the patch-owned
# names to the sole patch, so `solver.rho`, `solver.decomp` and the rest of
# the solver surface resolve to that patch.
#
# With several patches, the rank set is partitioned: MPI.Comm_split assigns
# each rank to exactly one patch's communicator, ranks in proportion to patch
# volume, except in a serial run where the one rank advances every patch in
# sequence. Patch interfaces are node-coincident (abutting patches share the
# interface plane) and are coupled by three mechanisms, all between stages and
# never inside a per-patch RHS evaluation (a shared rank advances its patches
# sequentially, so cross-patch communication inside compute_rhs! would wait on
# work that has not run yet):
#
#   1. `exchange_patch_ghosts!` copies each patch's interface ghost layers from
#      the abutting patch's interior. The ghosts carry conserved state; every
#      derived quantity in them (primitives, internal energy, mass fractions)
#      is recomputed locally by the existing full-padded-array passes.
#   2. The line solves are closed at the face with rows whose left-hand side
#      couples no ghost unknown (`interface_closures` in kernels.jl); whether
#      their right-hand sides read the exchanged ghosts (`:extended`) or stay
#      one-sided (`:onesided`) is the Solver's `interface_rhs` setting. The
#      flux divergence always keeps the scheme's own one-sided closures: ghost
#      FLUXES would require a second communication phase inside the RHS, which
#      the sequential-patch constraint above forbids.
#   3. `average_shared_planes!` replaces both patches' copies of each shared
#      interface-plane node by their mean after every RK stage, removing the
#      drift mode between the two solves.
#
# The current layout generator tiles the domain into slabs along a single
# dimension (`patch_grid` with one entry > 1). That keeps every interface's
# transverse extent equal to the full domain width, so diagonal patch
# adjacency does not arise; general patch grids extend the same record
# machinery with corner exchanges and are future work, as is patching across a
# coordinate fold (constraint 4 of reference/AMR_GPU.md; `Solver` rejects
# both).

# --- Ownership and freshness --------------------------------------------------
#
# Every array a run keeps, its owner, its writers, and what makes it stale.
# "Writes to Q" are the stage update, `filter_state!`, the positivity repairs,
# `sync_patches!`/`sync_levels!`, `regrid!`, `load_checkpoint!`, a rollback,
# and a callback.
#
# On the Patch, one set per patch:
#   rho u v w p T_ion c cp_mix Y   `primitives!` (through `refresh_primitives!`,
#       `max_rate`, `compute_rhs!`). Valid for the Q they were computed from and
#       stale after any write to Q; in a callback they hold the input of the
#       last RK stage. `prepared` (timestep.jl) asserts them current, and
#       `run!` passes it only after its own `max_rate` with no write to Q since.
#   mu_art beta_art kappa_art D_art   `compute_artificial!`. Read by the next
#       `max_rate` and the sensor tag criterion, so they outlive the step.
#       `preserving_artificial` restores them after an observation; a rollback
#       and `load_checkpoint!` restore them from the banked or recorded block;
#       `regrid!` recomputes them on the new hierarchy. On a tiled level with
#       shared faces `_level_artificial!` writes them instead, holding each
#       sensor in them before smoothing.
#   sensed_fields   within one level's artificial-property pass.
#   inv_J area_d inv_h inv_r cot_over_r cot_over_r_gcl   geometry, written once
#       by `init_geometry!`.
#   covered overwritten child_deep   `_fill_covered!` at setup and at every
#       regrid.
#   child_masked   `_arm_child_mask!` at every right-hand side.
#   reflux_deferred   `compute_rhs!`, around its component loop.
#   ghost_flux   written and consumed within one level's right-hand side.
#   level_scratch   within one parent step.
#   pairbuf pairout   within one line operation.
#
# On the RHSWorkspace, one set per padded extent per rank:
#   grad_T_ion grad_Y grad_Q flux sensor_sp ring_buf   valid only inside the
#       evaluation that wrote them; `max_rate` borrows grad_T_ion.
#   tmp_a tmp_b   free scratch of any single call (the RHS, `max_rate`,
#       `filter_state!`, the diagnostics, the tag sweep).
#   child_mask child_solve   the derivative mask of a parent patch, held from
#       the arming of one right-hand side to its end (`ChildMaskRecord`), and
#       the scratch of one masked derivative (`_mask_child_derivative!`);
#       empty on a solver of one level.
#   grad_u, strain_mag sensor   `compute_primitives_and_gradients!` and
#       `compute_artificial!`. They keep the last pass, which may belong to
#       another patch of the extent; `gradients_filled_by`/`sensors_filled_by`
#       name the patch and `scalar_field` checks them.
#
# On the Solver: the clock and step history (t, step, tstage, dt_prev,
# rate_prev, filter_rate_prev, cfl), which a rollback restores or zeroes and a
# checkpoint records. Held by the caller or by `run!`: the `Workspace` pair
# dQ/du, whose accumulator stage 1 forgets (RKA[1] = 0) except for a NaN, which
# a rollback after a non-finite failure clears; and the savepoint (Q, the
# coefficients, the clock, the switch flags), banked at a vetted step entry.
#
# Material caches of the planned material interface (a per-point equilibrium
# or table state) would sit on the patch beside the primitives and go stale
# with them on any write to Q. A nonlinear trial state is scratch of the solve
# that owns it and is discarded with the stage on a rollback.

# The patch whose parent-level derivative mask `child_mask` holds, as that
# patch's `covered` array (the mark of `gradients_filled_by`), and per
# dimension the range of the plan's lines holding a masked node. Set where a
# right-hand side arms the mask and cleared where it ends
# (`_release_child_mask!`), so that every derivative of one evaluation reads
# the mask its primitives gave and no later state reads it.
mutable struct ChildMaskRecord
    patch::Array{UInt8,3}
    lines::NTuple{3,UnitRange{Int}}
end

"""
    RHSWorkspace

The scratch fields of one right-hand-side evaluation, shared by every patch on
a rank whose padded local extent matches. A rank advances its patches in
sequence and every value in these arrays is consumed before the next patch's
`compute_rhs!` begins, so one set serves them all; only the persistent state
and geometry on [`Patch`](@ref) are per patch. A tiled level's lattice cells
carry equal extents by construction, so its tiles hold one set between them,
a cell clipped at the domain edge excepted. The exception is `grad_u` on a
tile of a level that computes its artificial coefficients level-wide
(`_level_artificial!`): that pass computes every tile's velocity gradients
before any tile's right-hand side reads them, so each such tile holds its own
(`_own_gradients`), and its right-hand side reads them rather than computing
them again.

Storage placement follows field lifetime. `mu_art`, `beta_art`, `kappa_art`,
and `D_art` survive into the next step's [`max_rate`](@ref). The primitives
must be current on every patch for a `prepared` first stage, and the geometry
never changes. Those groups therefore stay on the patch. Gradients, sensor
fields, sensor scratch, and assembled fluxes are written and consumed within
one patch's RHS and remain in the shared workspace.

Three fields are read after the evaluation that wrote them. `grad_u` holds the
last gradient pass and `strain_mag` and `sensor` the last artificial-property
pass, and [`scalar_field`](@ref) returns them for `:strain_mag`, `:sensor`, and
the names derived from the velocity gradients. On a multi-patch solver that
pass may belong to another patch of the same extent. `gradients_filled_by` and
`sensors_filled_by` therefore record the patch that wrote each group last, and
`scalar_field` raises an error for a patch that did not. [`field_array`](@ref),
`save_vtk` and `field_snapshot` recompute the fields of each patch before
reading them. A direct property read such as `ps.sensor` is not checked.

`pairbuf`/`pairout` stay on the patch instead: their allocation follows that
patch's coordinate folds, and a fold is rejected on every patched and refined
run, so they are never duplicated across a rank's patches.
"""
struct RHSWorkspace{T,A<:AbstractArray{T,3}}
    grad_u::Matrix{A}
    grad_T_ion::NTuple{3,A}
    grad_Y::Matrix{A}
    strain_mag::A
    sensor::A
    sensor_sp::A
    tmp_a::A
    tmp_b::A
    ring_buf::A                    # detector = :d8 only; empty otherwise
    flux::Matrix{A}                # flux[d, c]
    grad_Q::Matrix{A}              # shared-D_b species channels; 0 × 0 otherwise
    # The parent-level derivative mask and its line solve (rhs.jl); a refined
    # solver only, empty otherwise.
    child_mask::A
    child_solve::A
    # The patch whose gradient pass last wrote `grad_u` and whose artificial
    # pass last wrote `strain_mag` and `sensor`, recorded as that patch's
    # `covered` array: a host array every patch allocates for itself and
    # `_repatch` carries over, so `===` on it identifies the patch across the
    # `Patch` a regrid replaces it with. `SCRATCH_UNFILLED` before the first
    # pass.
    gradients_filled_by::Base.RefValue{Array{UInt8,3}}
    sensors_filled_by::Base.RefValue{Array{UInt8,3}}
    # The patch whose derivative mask `child_mask` holds for reuse, and its
    # masked lines (`ChildMaskRecord`, rhs.jl).
    child_mask_record::ChildMaskRecord
end

# The mark of a workspace no pass has written yet.
const SCRATCH_UNFILLED = zeros(UInt8, 0, 0, 0)

ChildMaskRecord() = ChildMaskRecord(SCRATCH_UNFILLED, (1:0, 1:0, 1:0))

# `ws` with velocity-gradient arrays of its own from the allocator `g()`, and
# their own mark, sharing every other field and the sensor mark with `ws`:
# the set of a tile whose gradients outlive the evaluation that wrote them
# (`RHSWorkspace`).
_own_gradients(ws::RHSWorkspace, g::F) where {F} =
    RHSWorkspace([g() for _ in 1:3, _ in 1:3], ws.grad_T_ion, ws.grad_Y,
                 ws.strain_mag, ws.sensor, ws.sensor_sp, ws.tmp_a, ws.tmp_b,
                 ws.ring_buf, ws.flux, ws.grad_Q, ws.child_mask, ws.child_solve,
                 Ref(SCRATCH_UNFILLED), ws.sensors_filled_by, ws.child_mask_record)

RHSWorkspace(grad_u, grad_T_ion, grad_Y, strain_mag, sensor, sensor_sp, tmp_a,
             tmp_b, ring_buf, flux, grad_Q, child_mask, child_solve) =
    RHSWorkspace(grad_u, grad_T_ion, grad_Y, strain_mag, sensor, sensor_sp, tmp_a,
                 tmp_b, ring_buf, flux, grad_Q, child_mask, child_solve,
                 Ref(SCRATCH_UNFILLED), Ref(SCRATCH_UNFILLED), ChildMaskRecord())

"""
    RHSWorkspace(backend, decomp, n_species, n_cons, ring, grad_Q_columns, child = false)

Allocate one scratch set on `backend` for a patch decomposed as `decomp`.
`ring` selects the `detector = :d8` ringing buffer, which is a zero-extent
placeholder under the default `:delta4`; a positive `grad_Q_columns` selects
the `grad_Q` gradients of the conserved components, `grad_Q[d, c]`, which the
species channels with one shared diffusivity (`species_flux =
:partial_density` and `:bulk`) difference into their fluxes, the first
`grad_Q_columns` of them (`_grad_Q_columns`); it is a 0 × 0 matrix of the
same array type otherwise, so the types do not depend on the option. `child`
selects the two arrays of the parent-level derivative mask, which a solver
with more than one level allocates (`_mask_child_derivative!`).
"""
function RHSWorkspace(backend::AbstractBackend, decomp::Decomp{T},
                      n_species::Int, n_cons::Int, ring::Bool,
                      grad_Q_columns::Int, child::Bool=false) where {T}
    f() = field(backend, decomp)
    return _rhs_workspace(f, empty_field(backend, T), n_species, n_cons, ring,
                          grad_Q_columns, child; active=decomp.active)
end

# The set from an allocator `f()` and the zero-extent placeholder `empty` of
# the same storage type: `field` on a backend above, or the stacked arrays
# of a device level's spanning patch (construction.jl).
#
# The fluxes, the species gradients and the conserved gradients of a collapsed
# dimension are neither computed nor read: every pass over them, the flux
# assembly, the species channels, the sharpening flux, the boundary hooks, the
# exchanges and the ledger, takes only the active dimensions. Their slots
# therefore hold one array between them, `sink`, which keeps every slot a
# padded field of the workspace's storage (a view, a device tuple and an
# extent check see the shape they expect) at the cost of one field instead of
# 2 n_cons + n_species on a planar or r-z run. The positivity limiter, which
# borrows flux rows as scratch, takes arrays of its own for a collapsed row
# (`_limiter_flux_rows`). The columns of `grad_Q` past `grad_Q_columns`,
# which the channel leaves alone, share the same array.
function _rhs_workspace(f::F, empty, n_species::Int, n_cons::Int,
                        ring::Bool, grad_Q_columns::Int, child::Bool=false;
                        active::NTuple{3,Bool}=(true, true, true)) where {F}
    bulk = grad_Q_columns > 0
    shared = !all(active) || (bulk && grad_Q_columns < n_cons)
    sink = shared ? f() : empty
    g(d) = active[d] ? f() : sink
    gq(d, c) = c <= grad_Q_columns ? g(d) : sink
    return RHSWorkspace([f() for _ in 1:3, _ in 1:3],
                        (f(), f(), f()),
                        [g(d) for d in 1:3, _ in 1:n_species],
                        f(), f(), f(), f(), f(),
                        ring ? f() : empty,
                        [g(d) for d in 1:3, _ in 1:n_cons],
                        bulk ? [gq(d, c) for d in 1:3, c in 1:n_cons] :
                               Matrix{typeof(empty)}(undef, 0, 0),
                        child ? f() : empty, child ? f() : empty)
end

# A tile's workspace on a stacked level: views of the spanning patch's set
# over the tile's block `kr` along the third dimension, so a tile evaluated on
# its own (a diagnostic's gradients, say) writes the slots the batched
# evaluation writes. The placeholder keeps the view type at zero extent.
function _view_workspace(ws::RHSWorkspace, kr::UnitRange{Int})
    v(a) = view(a, :, :, kr)
    # A placeholder array keeps its zero extent.
    vp(a) = size(a, 3) == 0 ? view(a, :, :, 1:0) : v(a)
    return RHSWorkspace(map(v, ws.grad_u), map(v, ws.grad_T_ion), map(v, ws.grad_Y),
                        v(ws.strain_mag), v(ws.sensor), v(ws.sensor_sp),
                        v(ws.tmp_a), v(ws.tmp_b), vp(ws.ring_buf),
                        map(v, ws.flux), map(v, ws.grad_Q), vp(ws.child_mask),
                        vp(ws.child_solve))
end

"An empty, concretely typed pool of [`RHSWorkspace`](@ref) sets for `backend`."
rhs_workspace_pool(backend::AbstractBackend, ::Type{T}) where {T} =
    RHSWorkspace{T,typeof(empty_field(backend, T))}[]

"""
    rhs_workspace!(pool, backend, decomp, n_species, n_cons, ring, grad_Q_columns,
                   child = false)

The [`RHSWorkspace`](@ref) serving a patch decomposed as `decomp`: an existing
set of `pool` whose arrays already carry the padded extent this patch needs,
or a fresh one appended to `pool`. Sharing is keyed on the padded extent alone,
so every array a patch is handed is the one it would have allocated for
itself, with its own storage type and strides; a level's equal-extent tiles
therefore collapse onto one set without any patch seeing a view or a reshape.
Callers seed `pool` with the workspaces already in use on the rank, which is
how a regrid grows it: a new tile whose extent is unlike anything held gets
its own set, and a departing tile's is available to a replacement of the same
size.

`slot` spreads the tiles of one extent over several sets where a level's
tiles are evaluated concurrently (`_tile_groups`): the tile is handed the
`mod1(slot, _workspace_slots(backend, decomp))`-th set of its extent, so
consecutive tiles of a level take different sets up to that count. With one
slot, the default, every tile of the extent shares one set.
"""
function rhs_workspace!(pool::AbstractVector, backend::AbstractBackend,
                        decomp::Decomp{T}, n_species::Int, n_cons::Int,
                        ring::Bool, grad_Q_columns::Int, child::Bool=false,
                        slot::Int=1) where {T}
    n = ntuple(d -> decomp.n_local[d] + 2 * decomp.n_halo_d[d], 3)
    wanted = mod1(slot, _workspace_slots(backend, decomp))
    seen = 0
    for (k, w) in enumerate(pool)
        size(w.tmp_a) == n && (!isempty(w.ring_buf) == ring) &&
            (!isempty(w.grad_Q) == (grad_Q_columns > 0)) &&
            (!isempty(w.child_mask) == child) || continue
        # A tile holding gradients of its own (`_own_gradients`) shares the
        # rest of its set, so the set is counted at its first entry only.
        _shares_scratch(w, pool, k) && continue
        seen += 1
        seen == wanted && return w
    end
    w = RHSWorkspace(backend, decomp, n_species, n_cons, ring, grad_Q_columns, child)
    push!(pool, w)
    return w
end

_shares_scratch(w, pool, k::Int) = any(i -> pool[i].tmp_a === w.tmp_a, 1:(k - 1))

"""
    _workspace_slots(backend, decomp) -> Int

The number of [`RHSWorkspace`](@ref) sets the tiles of one padded extent are
spread over: the session's thread count where a tile is held whole by one
rank on host storage and its interior is below [`THREAD_MIN_WORK`](@ref),
so that every threaded region inside it runs serially and the level's tiles
are evaluated concurrently instead (`_tile_groups`); one otherwise. A set
costs a few dozen fields of the tile's padded extent, which is under
`THREAD_MIN_WORK` points wherever more than one is made.
"""
function _workspace_slots(backend::AbstractBackend, decomp::Decomp)
    (backend isa CPUBackend && prod(decomp.dims) == 1) || return 1
    TILE_SLOTS[] > 0 && return TILE_SLOTS[]
    return Threads.nthreads() > 1 && prod(decomp.n_local) < THREAD_MIN_WORK[] ?
           Threads.nthreads() : 1
end

"""
Test and benchmark toggle: a positive value spreads the tiles of one extent
held whole on host storage over that many [`RHSWorkspace`](@ref) sets and
evaluates a level's tiles as concurrent tasks (`_tile_groups`) whatever their
size and the session's thread count, so that one thread exercises the
concurrent path; 0, the default, takes the rule of `_workspace_slots`. Read
when the tiles are built and at every concurrent phase.
"""
const TILE_SLOTS = Ref(0)

"""
    RefluxCapture

One side of a coarse-fine junction as one patch holds it: the parent's lines
through the face node beside a child tile, or the child's lines through its
first counted nodes (src/reflux.jl). `nodes` are the padded indices along `d`
whose increments enter the junction's face flux, with the coefficients of the
derivative (`dcoef`) and of the filter (`fcoef`); `dr` and `fr` are the
padded index `j` of the anchor face `j + ½` whose explicit stencil this rank
evaluates, 0 where another rank holds it. Every per-line array is indexed by
the line's position in `lines`, the padded ranges of the two transverse
dimensions in ascending order. `stage` collects one right-hand side, `du` and
`reg` follow it through the low-storage recurrence, and `reg` takes each
filter pass directly; `om0` is scratch for the child's correction of the
quadrature. `window` lists the parent's padded indices along `d` of the
gate's nodes this rank holds, `wslots` their places among the junction's
`GATE_WIDTH`, and `rho0` the density there at the step's start.
"""
struct RefluxCapture{T,M<:AbstractMatrix{T},V<:AbstractVector{T}}
    level::Int                  # index in `solver.levels` of the parent level
    junction::Int               # the junction's index under that level
    child::Bool
    d::Int
    sgn::T
    nodes::NTuple{6,Int}
    dcoef::NTuple{6,T}
    dr::Int
    drs::NTuple{8,T}            # derivative face stencil, l = 1 - dM .. dM
    dM::Int
    drscale::T
    fcoef::NTuple{6,T}
    fr::Int
    frs::NTuple{8,T}            # filter face stencil, l = 1 - fM .. fM
    fM::Int
    frscale::T
    hd::T
    lines::NTuple{2,UnitRange{Int}}
    kap::V                      # per line: the measure of a derivative's flux
    kapf::V                     # per line: that of a filter pass, 0 if not taken
    bnode::Int                  # parent: padded index of the correction's node, or 0
    wb::V                       # parent: that node's composite weight
    band::NTuple{6,Int}         # child: the quadrature correction's nodes
    omega::NTuple{6,T}
    entry::Vector{Int}          # per line, its line of the junction
    window::Vector{Int}         # parent: the gate's nodes along d held here
    wslots::Vector{Int}         # parent: their places among the gate's slots
    rho0::Matrix{Float64}       # parent: the density there at the step's start
    snap::M
    stage::M
    du::M
    reg::M
    om0::M
end

"""
    Patch

Per-patch state of a [`Solver`](@ref): the patch's place in the global grid
(`region`, in global node index space), its communicator and [`Decomp`](@ref),
its operator plans and folds, and every persistent field array, typed
`A <: AbstractArray{T,3}` by the storage backend. The scratch of a
right-hand-side evaluation is not per patch; it lives on the
[`RHSWorkspace`](@ref) this patch shares with the rank's other patches of the
same extent, and the property forwarding below reaches it by the same names as
before. Constructed by `Solver`; user code normally reaches these fields
through the solver (which forwards them for a single-patch run) without
holding a `Patch` directly.

The face conditions are deliberately not in the type. They are stored under
the abstract [`FaceConditions`](@ref), because they are configuration the
compute path below `step!` never reads: with `bcs` as a type parameter, every
combination of boundary conditions recompiled the whole right-hand-side tree,
and combinations are what a user varies. Measured on the precompile workload,
erasing them takes the number of distinct `Patch` types from 24 to 16, the
package precompile from 108 s to 97 s, and the compile of one previously
unseen boundary-condition combination from 1.85 s to 0.58 s. The cost is one
dynamic dispatch per face in each of `apply_bcs!`, `correct_flux!`, and
`correct_rhs!`, up to six calls to each hook per stage,
each in front of a whole-plane sweep: `apply_bcs!` measured 66.4 µs to
71.8 µs at 64³, against an 87 ms step. The artificial-property detector pays a
fourth, `sensor_mirror` through `_face_mirror` (artificial.jl), once per closed
face per sensor field in front of a whole-array sweep.

The plan tuples are the case where this does not pay, and the comment above
`_plan_at` in rhs.jl carries the measurement. `folds` and `ring_plans` keep
parameters of their own for a third reason: `nothing` in either holds a whole
operator path (the fold closures, the `:d8` detector) off the default
configuration's inference path entirely.

The struct is mutable, although no field but the flags `child_masked` and
`reflux_deferred` is reassigned after construction, because of those dynamic
calls and the `_cold` barriers: an argument passed
through one is boxed, and an immutable `Patch`, held inline in its
[`PatchSolver`](@ref) with the plans and the workspace it carries, would be
copied whole into each box, and into every `PatchSolver` built from an entry
of the solver's abstractly typed patch list. A mutable one is boxed as a
reference.
"""
mutable struct Patch{T,A<:AbstractArray{T,3},Fo,DP,VP,FP,SP,RP,W,LS,GF,TF,RC}
    id::Int
    level::Int
    region::BlockRegion                     # offset + extent, in this LEVEL's node
                                            # index space (level ℓ spacing h/3^ℓ,
                                            # so physical coordinates follow from
                                            # region.offset and h alone)
    comm::MPI.Comm                          # ranks owning pieces of this patch
    decomp::Decomp{T}                       # decomposition of the patch over comm
    h::NTuple{3,T}                          # this level's grid spacing (h/3 per
                                            # refinement level below the root)
    faces::NTuple{3,NTuple{2,Int}}          # 0 = physical/periodic; else neighbor patch id
    bcs::FaceConditions                     # per-face conditions (InterfaceBC at interfaces)
    folds::Fo
    deriv_plans::DP
    div_plans::VP                           # divergence plans: deriv_plans unless an
                                            # interface end forces one-sided closures
    filter_plans::FP
    smooth_plans::SP
    ring_plans::RP
    pairbuf::A
    pairout::A
    # primitives (full padded arrays)
    rho::A; u::A; v::A; w::A
    p::A; T_ion::A; c::A; cp_mix::A
    Y::Vector{A}
    # artificial properties: written by one RHS and read by the next step's
    # max_rate, so they outlive the evaluation that fills them
    mu_art::A; beta_art::A; kappa_art::A
    D_art::Vector{A}
    # geometry
    inv_J::A
    area_d::NTuple{3,A}
    inv_h::NTuple{3,A}
    inv_r::A
    cot_over_r::A
    # discrete-GCL cotθ/r (metric.jl gcl_cotr!): read by the momentum sources only
    cot_over_r_gcl::A
    # The RHS scratch set (RHSWorkspace), shared with the rank's other patches
    # of the same padded extent. Its names reach callers through the property
    # forwarding below, so `solver.grad_u` and `solver.tmp_a` read as before.
    rhs_workspace::W
    # Which orthants of each node's quadrature cell a child level covers
    # (`_fill_covered!`): bit (s1 > 0) + 2(s2 > 0) + 4(s3 > 0) for the
    # orthant of signs s_d, so 0 is uncovered, 0xFF fully covered, and a
    # node on a child's face carries half its bits. Host storage over the
    # padded extent; only the interior is written. Filled at setup and at
    # every regrid from the child regions every rank of this level's subset
    # holds, so a diagnostic on this level's ranks alone reads it.
    covered::Array{UInt8,3}
    # Nonzero at the interior nodes deep inside a child level, which its
    # restriction overwrites after every step and `max_rate` holds to
    # `OVERWRITTEN_CFL` (`_fill_overwritten!`). Zero-extent on a patch of a
    # solver with one level.
    overwritten::A
    # The device scratch of this patch's level transfer (`LevelScratch`,
    # levels.jl): the Hermite boxes and the interpolation chain's stages on
    # the backend, so a device-resident refined patch's shell is built
    # without host arithmetic. Empty on the host backend, whose chain runs
    # on the transfer's own host stages, and on a patch that is nobody's
    # child.
    level_scratch::LS
    # The molecular flux of each dimension with an interface end, all
    # conserved components, under `interface_flux = :ghost` with molecular
    # transport (`_ghost_viscous`): written over the interior by the
    # right-hand side, over a coarse-fine face's ghost layers from the
    # shell's gradients, and over a same-level face's ghost layers by the
    # level's flux records, which read the neighbour's interior (the
    # `ghost_flux` section of rhs.jl). A zero-extent array in every other
    # dimension and configuration.
    ghost_flux::GF
    # The sensed fields of a tile on a tiled level that are computed over the
    # interior and carry no interface ghosts of their own, the strain
    # magnitude and the dilatation: held per tile between the phases of the
    # level's artificial-property pass (`_level_artificial!`), whose records
    # fill their ghost layers at a shared face. Empty on every other patch.
    sensed_fields::Vector{A}
    # Per node, the dimensions along which the parent-level derivative mask
    # may take it, as bits (`_fill_child_deep!`); written with `covered`.
    # Zero-extent where the workspace carries no mask scratch.
    child_deep::A
    # Per dimension, whether this patch's last right-hand side found a node
    # for the parent-level derivative mask (`_arm_child_mask!`): reduced over
    # the patch's communicator, so every rank of a line takes the extra solve
    # or none does. All false on a patch with no child level.
    child_masked::NTuple{3,Bool}
    # The same collections as isbits-adaptable tuples (FieldVector /
    # FieldMatrix, pointwise.jl): what the pointwise per-point bodies index,
    # since a Vector or Matrix kernel argument hangs a device launch. Same
    # array objects, no copies (`Y` and `D_art` off the patch, `grad_u`,
    # `grad_Y` and `flux` off the workspace); the convenience constructor
    # below derives them.
    field_tuples::TF
    # The coarse-fine junctions this patch takes part in, as a parent or as
    # a child (`RefluxCapture`, src/reflux.jl); rebuilt whenever the layout
    # changes, empty on a solver without refinement.
    reflux_captures::RC
    # Whether the right-hand side running on this patch defers the child's
    # rate of Ω to its component loop (`_reflux_defer!`): set around that
    # loop and clear outside it. Per patch, so that the tiles of a level
    # evaluated concurrently each hold their own.
    reflux_deferred::Bool
end

# The positional argument list every construction site uses; `field_tuples`
# is derived here, not at the sites, so the sites never fall out of
# step with the wrapper set.
function Patch(id, level, region, comm, decomp, h, faces, bcs, folds,
               deriv_plans, div_plans, filter_plans, smooth_plans, ring_plans,
               pairbuf, pairout, rho, u, v, w, p, T_ion, c, cp_mix, Y,
               mu_art, beta_art, kappa_art, D_art,
               inv_J, area_d, inv_h, inv_r, cot_over_r, cot_over_r_gcl,
               rhs_workspace, covered, overwritten, level_scratch, ghost_flux,
               sensed_fields=empty(Y),
               child_deep=isempty(rhs_workspace.child_mask) ?
                          rhs_workspace.child_mask : zero(overwritten),
               child_masked=(false, false, false))
    field_tuples = (Y=FieldVector(Y), D_art=FieldVector(D_art),
                    grad_u=FieldMatrix(rhs_workspace.grad_u),
                    grad_Y=FieldMatrix(rhs_workspace.grad_Y),
                    flux=FieldMatrix(rhs_workspace.flux))
    return Patch(id, level, region, comm, decomp, h, faces, bcs, folds,
                 deriv_plans, div_plans, filter_plans, smooth_plans,
                 ring_plans, pairbuf, pairout, rho, u, v, w, p, T_ion, c,
                 cp_mix, Y, mu_art, beta_art, kappa_art, D_art,
                 inv_J, area_d, inv_h, inv_r, cot_over_r, cot_over_r_gcl,
                 rhs_workspace, covered, overwritten, level_scratch, ghost_flux,
                 sensed_fields, child_deep, child_masked, field_tuples,
                 RefluxCapture{eltype(rho),typeof(similar(rho, 0, 0)),
                               typeof(similar(rho, 0))}[], false)
end

# --- Covered masks ----------------------------------------------------------
#
# A diagnostic over a refined run integrates the composite grid: the coarse
# nodes a child level covers must be excluded, and excluded exactly once.
# With node-centered quadrature a node's cell is the box of half-cells
# around it, and a child tile, whose region spans parent nodes lo .. hi
# along each dimension, covers the orthant of signs s_d exactly when the
# half-cell on each side s_d lies inside [lo, hi]: the + side for
# lo ≤ g ≤ hi − 1, the − side for lo + 1 ≤ g ≤ hi. The mask records the
# covered orthants as bits, one per orthant, so a node on a child's face
# keeps the half of its cell outside the child, a corner a quarter, and a
# node shared by two abutting tiles is covered by both halves. The fine
# patch's own quadrature gives its boundary planes half weights, so the
# two sides of a coarse-fine face sum to one and the plane is counted
# once, which is the same rule `volume_integral` applies to a same-level
# interface plane. A union of lattice tiles is covered by the union of
# their bits, which is why the mask is per orthant and not a per-dimension
# factor.

"The covered mask of a patch decomposed as `decomp`: uncovered everywhere."
_covered_mask(decomp::Decomp) =
    zeros(UInt8, ntuple(d -> decomp.n_local[d] + 2 * decomp.n_halo_d[d], 3))

"""
    _fill_covered!(patch, regions, period=(0, 0, 0))

Rewrite `patch.covered` from the child regions `regions`, given in this
patch's level node space (a `LevelTransfer.region` of the level below), a
region across a periodic seam covering through its images under `period`
(that node space's period, `_level_period`), and with it `patch.overwritten`
(`_fill_overwritten!`). Rank-local: it reads the regions and the patch's own
block placement, both of which every rank of the patch's level holds.
"""
function _fill_covered!(patch::Patch, regions::Vector{BlockRegion},
                        period::NTuple{3,Int}=(0, 0, 0))
    regions = [_shifted(r, σ) for r in regions for σ in _images(period)]
    covered = patch.covered
    overwritten = patch.overwritten
    decomp = patch.decomp
    o = decomp.n_halo_d
    n = decomp.n_local
    base = ntuple(d -> patch.region.offset[d] + decomp.offset[d], 3)
    # The mask bytes over the interior and, where `overwritten` or `child_deep`
    # is formed, `LEVEL_BUFFER` nodes beyond it on every active side, which
    # their erosions read: those nodes may lie on another rank or another
    # tile, so they are evaluated from the regions, not exchanged.
    eroded = !isempty(overwritten) || !isempty(patch.child_deep)
    pad = ntuple(d -> decomp.active[d] && eroded ? LEVEL_BUFFER : 0, 3)
    bits_ext = zeros(UInt8, ntuple(d -> n[d] + 2 * pad[d], 3))
    # The patch's own edge nodes along each dimension, and whether the face
    # there closes the domain. A node on such a face has no cell beyond it,
    # so a child reaching the face covers the outer half as well, and the
    # node, covered on the inner side, keeps nothing.
    edge_lo = ntuple(d -> patch.region.offset[d] + 1, 3)
    edge_hi = ntuple(d -> patch.region.offset[d] + decomp.n_global[d], 3)
    closed = ntuple(d -> ntuple(s -> decomp.active[d] && !decomp.periodic[d] &&
                                     !(patch.bcs[d][s] isa InterfaceBC), 2), 3)
    for r in regions
        lo = ntuple(d -> r.offset[d] + 1, 3)
        hi = ntuple(d -> r.offset[d] + r.extent[d], 3)
        # Local indices of the nodes the region meets.
        rng = ntuple(d -> decomp.active[d] ?
                     (max(lo[d] - base[d], 1 - pad[d]):min(hi[d] - base[d],
                                                          n[d] + pad[d])) :
                     (1:1), 3)
        any(isempty, rng) && continue
        @inbounds for k in rng[3], j in rng[2], i in rng[1]
            g = (base[1] + i, base[2] + j, base[3] + k)
            # Per dimension, whether the − and + half-cells are covered; a
            # collapsed dimension is spanned by every region.
            minus = ntuple(d -> !decomp.active[d] || lo[d] + 1 <= g[d] <= hi[d] ||
                                (g[d] == lo[d] == edge_lo[d] && closed[d][1]), 3)
            plus = ntuple(d -> !decomp.active[d] || lo[d] <= g[d] <= hi[d] - 1 ||
                               (g[d] == hi[d] == edge_hi[d] && closed[d][2]), 3)
            bits = zero(UInt8)
            for b in 0:7
                s1, s2, s3 = b & 1, (b >> 1) & 1, (b >> 2) & 1
                (s1 == 1 ? plus[1] : minus[1]) &&
                    (s2 == 1 ? plus[2] : minus[2]) &&
                    (s3 == 1 ? plus[3] : minus[3]) || continue
                bits |= UInt8(1) << b
            end
            bits_ext[i + pad[1], j + pad[2], k + pad[3]] |= bits
        end
    end
    fill!(covered, zero(UInt8))
    interior = CartesianIndices(n)
    @inbounds for J in interior
        covered[J + CartesianIndex(o)] = bits_ext[J + CartesianIndex(pad)]
    end
    isempty(overwritten) ||
        _fill_overwritten!(overwritten, bits_ext, pad, base, edge_lo, edge_hi,
                           closed, decomp)
    isempty(patch.child_deep) ||
        _fill_child_deep!(patch.child_deep, bits_ext, pad, base, edge_lo, edge_hi,
                          closed, decomp)
    return patch
end

# Whether each node of `bits_ext` is covered over its whole cell or lies
# beyond a face closing the domain.
function _fully_covered_ext(bits_ext::Array{UInt8,3}, pad, base, edge_lo, edge_hi,
                            closed, decomp::Decomp)
    active = decomp.active
    full = falses(size(bits_ext))
    @inbounds for J in CartesianIndices(full)
        g = ntuple(d -> base[d] + J[d] - pad[d], 3)
        beyond = any(d -> active[d] && ((g[d] < edge_lo[d] && closed[d][1]) ||
                                        (g[d] > edge_hi[d] && closed[d][2])), 1:3)
        full[J] = beyond || bits_ext[J] == 0xff
    end
    return full
end

# The parent nodes the derivative mask may take along each dimension
# (`_mask_child_derivative!`, rhs.jl), as bit d − 1 of a small integer stored
# in the field's element type. A node qualifies along `d` when the restriction
# writes it, which holds `RESTRICT_MARGIN` = 2 nodes off a face, so that it and
# its neighbours along every active dimension are covered over their whole
# cells, and when it and `CHILD_DERIVATIVE_DEPTH` nodes on either side along
# `d` are.
function _fill_child_deep!(deep, bits_ext::Array{UInt8,3}, pad::NTuple{3,Int}, base,
                           edge_lo, edge_hi, closed, decomp::Decomp)
    n = decomp.n_local
    o = decomp.n_halo_d
    active = decomp.active
    full = _fully_covered_ext(bits_ext, pad, base, edge_lo, edge_hi, closed, decomp)
    span(K, d, w) = all(t -> full[K + t * CartesianIndex(ntuple(e -> Int(e == d), 3))],
                        -w:w)
    T = eltype(deep)
    host = zeros(T, size(deep))
    @inbounds for J in CartesianIndices(n)
        K = J + CartesianIndex(pad)
        written = all(d -> !active[d] || span(K, d, 1), 1:3)
        written || continue
        bits = 0
        for d in 1:3
            active[d] && span(K, d, CHILD_DERIVATIVE_DEPTH) && (bits |= 1 << (d - 1))
        end
        host[J + CartesianIndex(o)] = T(bits)
    end
    _upload!(deep, host)
    return deep
end

# The parent nodes `max_rate` holds to `OVERWRITTEN_CFL`: those a child
# level's restriction overwrites after every step, less those whose values
# within the step reach the child's ghost layers or outlive the step. A
# covered node carries the restricted fine solution, at which the parent's
# artificial coefficients, sensed at the parent spacing, take the diffusivity
# of a feature thinner than that spacing; at the solver's CFL number the
# rate there bounded most root steps of the refined runs measured and made
# them about 1.5 times as short as the uncovered solution needs. A node
# qualifies when every node within `LEVEL_BUFFER` of it along each active
# dimension is covered over its whole cell (mask byte 0xFF) or lies beyond a
# face closing the domain. The set then starts `LEVEL_BUFFER + 1` nodes inside
# every face a child shares with its parent: past the `RESTRICT_MARGIN`
# nodes the restriction does not write, which carry the parent's solution
# into the next step, and past the Lagrange stencils of the shell fill, which
# reach `interp_order ÷ 2` ≤ 5 nodes in, so the child's ghost layers read
# only nodes advanced at the solver's CFL number.
function _fill_overwritten!(overwritten, bits_ext::Array{UInt8,3},
                            pad::NTuple{3,Int}, base, edge_lo, edge_hi, closed,
                            decomp::Decomp)
    n = decomp.n_local
    o = decomp.n_halo_d
    active = decomp.active
    full = _fully_covered_ext(bits_ext, pad, base, edge_lo, edge_hi, closed, decomp)
    # Erosion by a box of half-width `LEVEL_BUFFER`, one active dimension at a
    # time. A node in the pad takes a truncated window, and no interior node's
    # result reads such a node along a dimension already eroded.
    for d in 1:3
        active[d] || continue
        src = copy(full)
        m = size(full, d)
        @inbounds for J in CartesianIndices(full)
            ok = src[J]
            if ok
                for s in -LEVEL_BUFFER:LEVEL_BUFFER
                    q = J[d] + s
                    1 <= q <= m || continue
                    K = CartesianIndex(ntuple(e -> e == d ? q : J[e], 3))
                    src[K] || (ok = false; break)
                end
            end
            full[J] = ok
        end
    end
    T = eltype(overwritten)
    host = zeros(T, size(overwritten))
    @inbounds for J in CartesianIndices(n)
        full[J + CartesianIndex(pad)] && (host[J + CartesianIndex(o)] = one(T))
    end
    _upload!(overwritten, host)
    return overwritten
end

"Fraction of a node's quadrature cell no child covers, from its mask byte."
@inline uncovered_fraction(m::UInt8) = (8 - count_ones(m)) / 8

"""
    uncovered_plane_fraction(m, d)

Fraction of a node's cell within the plane normal to `d` that no child
covers: the plane's sub-cells pair the orthants across `d`, and a sub-cell
is covered when either orthant of its pair is (a child covering a half-cell
on one side of the plane covers the plane itself).
"""
@inline function uncovered_plane_fraction(m::UInt8, d::Int)
    bit = UInt8(1) << (d - 1)
    free = 0
    for b in 0:7
        (b & Int(bit)) == 0 || continue
        lo = (m >> b) & 0x01
        hi = (m >> (b | Int(bit))) & 0x01
        (lo | hi) == 0 && (free += 1)
    end
    return free / 4
end

# Property names owned by the patch's shared [`RHSWorkspace`](@ref) rather than
# by the patch itself. Split out of the test below so that one `===` chain
# still decides both questions at a literal call site.
@inline _is_workspace_prop(n::Symbol) =
    n === :grad_u || n === :grad_T_ion || n === :grad_Y ||
    n === :strain_mag || n === :sensor || n === :sensor_sp ||
    n === :tmp_a || n === :tmp_b || n === :ring_buf || n === :flux ||
    n === :grad_Q || n === :child_mask || n === :child_solve

# Property names owned by the patch or its workspace, not by the solver
# configuration. `Base.getproperty(::Solver, name)` forwards these to the sole
# patch of a single-patch solver, and `PatchSolver` routes them per patch. The
# test is a `===` chain, allowing a literal property name to constant-fold to a
# plain `getfield` at every call site.
@inline _is_patch_prop(n::Symbol) =
    n === :decomp || n === :bcs || n === :folds || n === :faces ||
    n === :region || n === :h || n === :deriv_plans || n === :div_plans ||
    n === :filter_plans || n === :smooth_plans || n === :ring_plans ||
    n === :pairbuf || n === :pairout || n === :ghost_flux || n === :sensed_fields ||
    n === :rho || n === :u || n === :v || n === :w ||
    n === :p || n === :T_ion || n === :c || n === :cp_mix || n === :Y ||
    n === :mu_art || n === :beta_art || n === :kappa_art || n === :D_art ||
    n === :inv_J || n === :area_d || n === :inv_h || n === :inv_r ||
    n === :cot_over_r || n === :cot_over_r_gcl || n === :covered ||
    n === :overwritten || n === :child_deep || n === :child_masked ||
    n === :field_tuples || _is_workspace_prop(n)

# The read behind both forwards: a workspace name takes one further hop.
@inline _patch_property(p::Patch, n::Symbol) =
    _is_workspace_prop(n) ? getfield(getfield(p, :rhs_workspace), n) :
                            getfield(p, n)

"""
    PatchSolver(solver, patch)

Access adapter binding one [`Patch`](@ref) to its solver configuration.
Property reads forward patch-owned names to the patch and everything else to
the solver, so every routine written against the solver's property surface
(`s.rho`, `s.decomp`, `s.equations`, ...) runs unchanged per patch. The
multi-patch drivers in timestep.jl construct these; a single-patch `Solver`
plays the role itself through its own property forwarding.
"""
struct PatchSolver{T,S,P<:Patch{T}}
    solver::S
    patch::P
    PatchSolver(solver, patch::Patch{T}) where {T} =
        new{T,typeof(solver),typeof(patch)}(solver, patch)
end

@inline function Base.getproperty(ps::PatchSolver, name::Symbol)
    _is_patch_prop(name) && return _patch_property(getfield(ps, :patch), name)
    name === :solver && return getfield(ps, :solver)
    name === :patch && return getfield(ps, :patch)
    return getproperty(getfield(ps, :solver), name)
end

# Solver clock updates written through a PatchSolver land on the solver.
@inline Base.setproperty!(ps::PatchSolver, name::Symbol, value) =
    setproperty!(getfield(ps, :solver), name, value)

# The patch a `SolverLike` evaluates; the `Solver` method is in solver.jl.
@inline _patch_of(ps::PatchSolver) = getfield(ps, :patch)

# Record the patch of `solver` as the last writer of its workspace's
# `grad_u` (`_mark_gradients!`) or `strain_mag` and `sensor`
# (`_mark_sensors!`). One reference store per pass.
@inline function _mark_gradients!(solver)
    p = _patch_of(solver)
    getfield(p, :rhs_workspace).gradients_filled_by[] = getfield(p, :covered)
    return nothing
end
@inline function _mark_sensors!(solver)
    p = _patch_of(solver)
    getfield(p, :rhs_workspace).sensors_filled_by[] = getfield(p, :covered)
    return nothing
end

# The guard of `scalar_field` on a name it reads from the shared workspace:
# the group (`:gradients` or `:sensors`) must have been written last by this
# patch, or by no pass at all, in which case it holds its allocation zeros.
function _check_scratch_writer(solver, group::Symbol, name::Symbol)
    p = _patch_of(solver)
    ws = getfield(p, :rhs_workspace)
    mark = group === :gradients ? ws.gradients_filled_by[] : ws.sensors_filled_by[]
    (mark === getfield(p, :covered) || mark === SCRATCH_UNFILLED) && return nothing
    throw(ArgumentError(
        "scalar_field: `$name` is read from RHS scratch that this patch shares " *
        "with other patches of the same extent, and another patch wrote it last. " *
        "Use field_array(solver, states, :$name), which recomputes it for each " *
        "patch."))
end

"""
    PatchFields

The storage one right-hand-side phase reads and writes, gathered off a
`Solver` or `PatchSolver` by [`patch_fields`](@ref): the decomposition, the
spacings, and the field arrays. Its type depends on the element type and the
array types alone, never on the operator plans, folds, stretch, sources, or
the wrapper, so an orchestration body keyed on it compiles once per
`(T, array type)` and is shared by every scheme, detector, dimensionality
and patch wrapper. The plan-dependent operators are reached through the
solver passed alongside as an unspecialized handle; see
[`compute_artificial!`](@ref).

Built on the stack at each call: the same array objects, no copies.
"""
struct PatchFields{T,A<:AbstractArray{T,3},YV,W,GQ,H,FT}
    decomp::Decomp{T}
    h::NTuple{3,T}
    inv_h::H
    rho::A; u::A; v::A; w::A
    p::A; T_ion::A; c::A; cp_mix::A
    Y::YV
    mu_art::A; beta_art::A; kappa_art::A
    D_art::YV
    strain_mag::W; sensor::W; sensor_sp::W; tmp_a::W; tmp_b::W
    grad_T_ion::NTuple{3,W}
    grad_Q::GQ
    ft::FT
end

@inline patch_fields(s) =
    PatchFields(s.decomp, s.h, s.inv_h, s.rho, s.u, s.v, s.w, s.p, s.T_ion,
                s.c, s.cp_mix, s.Y, s.mu_art, s.beta_art, s.kappa_art,
                s.D_art, s.strain_mag, s.sensor, s.sensor_sp, s.tmp_a,
                s.tmp_b, s.grad_T_ion, s.grad_Q, s.field_tuples)

# The equation set's layout as plain integers, so a body taking it is not
# keyed on the equation-set type: (n_species, n_cons, i_energy, i_mom).
@inline equation_layout(eq) = (eq.n_species, eq.n_cons, eq.i_energy, eq.i_mom)

# --- Patch layout -----------------------------------------------------------

"""
    patch_slabs(n_global, periodic, patch_grid) -> Vector{BlockRegion}

Tile the global node lattice into `prod(patch_grid)` slab patches. Exactly one
entry of `patch_grid` may exceed one (see the header). Abutting patches share
their interface plane node; along a non-periodic split dimension of N nodes,
P patches therefore carry N + P − 1 nodes in total, and along a periodic one
N + P, the last patch's high plane being the first patch's low plane. Regions
are returned in patch id order with global node offsets.
"""
function patch_slabs(n_global::NTuple{3,Int}, periodic::NTuple{3,Bool},
                     patch_grid::NTuple{3,Int})
    all(>=(1), patch_grid) || error("patch_grid entries must be positive")
    nsplit = count(>(1), patch_grid)
    nsplit <= 1 || error("patch_grid may split one dimension only " *
                         "(slab layout); got $patch_grid")
    npatch = prod(patch_grid)
    npatch == 1 && return [BlockRegion((0, 0, 0), n_global)]
    ds = findfirst(>(1), patch_grid)
    P = patch_grid[ds]
    N = n_global[ds]
    # Interval boundaries in global node index space: patch j spans nodes
    # b[j] .. b[j+1] inclusive (the shared planes). Periodic wraps the last
    # boundary onto node N + 1 ≡ 1.
    span = periodic[ds] ? N : N - 1
    bounds = [1 + round(Int, span * (j / P)) for j in 0:P]
    length(unique(bounds)) == P + 1 ||
        error("n_global[$ds] = $N is too small for $P patches")
    regions = BlockRegion[]
    for j in 1:P
        lo, hi = bounds[j], bounds[j+1]
        offset = ntuple(d -> d == ds ? lo - 1 : 0, 3)
        extent = ntuple(d -> d == ds ? hi - lo + 1 : n_global[d], 3)
        push!(regions, BlockRegion(offset, extent))
    end
    return regions
end

"""
    patch_rank_counts(regions, np) -> Vector{Int}

Ranks assigned to each patch: proportional to patch volume by largest
remainder, at least one per patch, summing to `np`. Deterministic, so every
rank computes the same partition without communication.
"""
function patch_rank_counts(regions::Vector{BlockRegion}, np::Int)
    npatch = length(regions)
    np >= npatch || error("$(np) ranks cannot partition over $npatch patches; " *
                          "run with at least one rank per patch (or serially)")
    vol = [prod(r.extent) for r in regions]
    total = sum(vol)
    share = [np * v / total for v in vol]
    counts = max.(floor.(Int, share), 1)
    # Largest-remainder distribution after flooring shares and enforcing one
    # rank per patch. If the minimum constraint over-allocates ranks, remove
    # them from the largest counts first.
    while sum(counts) < np
        i = argmax(share .- counts)
        counts[i] += 1
    end
    while sum(counts) > np
        order = sortperm(counts; rev=true)
        done = false
        for i in order
            counts[i] > 1 || continue
            counts[i] -= 1
            done = true
            break
        end
        done || error("cannot assign at least one rank per patch")
    end
    return counts
end

# --- Interface exchange records ---------------------------------------------
#
# Built once at setup from a world-Allgathered table of every rank's
# (patch id, block offset, block extent). Every record carries PADDED local
# index ranges into this rank's own patch arrays, plus the partner world rank
# and a tag both sides compute identically (`_pair_tags`). `partner == myrank`
# marks the serial case, executed as a direct copy between two local patches.

struct GhostRecord{T}
    patch::Int                        # local patch index on this rank
    partner::Int                      # world rank (== my rank for a local copy)
    partner_patch::Int                # local patch index of the partner (local case)
    tag::Int
    mine::NTuple{3,UnitRange{Int}}    # padded ranges in my patch's arrays
    theirs::NTuple{3,UnitRange{Int}}  # padded ranges in the partner patch (local case)
    buf::Vector{T}
end

struct PlaneRecord{T}
    patch::Int
    partner::Int                      # world rank (== my rank for a local pairing)
    partner_patch::Int
    partner_pid::Int                  # the partner's patch id (every case)
    tag::Int                          # tag this side receives on
    sendtag::Int                      # tag the partner receives on
    mine::NTuple{3,UnitRange{Int}}
    theirs::NTuple{3,UnitRange{Int}}  # local case only
    buf::Vector{T}                    # receive buffer
    sbuf::Vector{T}                   # send buffer
end

# Tags. A message is named by (source patch, destination patch, dimension,
# destination face side): a periodic two-slab layout pairs the same two
# patches across both faces, so the side is part of the name. Between two
# ranks each name occurs once per record kind, and both ranks hold the names
# of every message between them, so each numbers them in sorted order and
# the index is the tag, offset by the kind's base so ghosts and plane
# averages do not collide. The tag is then bounded by the messages between
# one pair of ranks, a few per tile face they share, rather than by the
# patch count: a tag computed from the patch pair grows with its square,
# passing the 32767 the MPI standard guarantees at 52 patches and 2^31 − 1 at
# 18918.
const _GHOST_TAG_BASE = 2000
const _PLANE_TAG_BASE = 16000
const _TAG_SPAN = 14000

# One tag per message from the base of its kind: `partners[i]` is the other
# rank of message `i` and `names[i]` its name; messages to or from the calling
# rank itself (`partners[i] == me`) move without MPI and take no tag.
function _pair_tags(partners::Vector{Int}, names::Vector{NTuple{4,Int}}, base::Int,
                    me::Int)
    tags = zeros(Int, length(names))
    count = Dict{Int,Int}()
    for i in sortperm(collect(zip(partners, names)))
        partners[i] == me && continue
        k = get(count, partners[i], 0)
        k < _TAG_SPAN || error("more than $_TAG_SPAN interface messages between " *
                               "rank $me and rank $(partners[i])")
        tags[i] = base + k
        count[partners[i]] = k + 1
    end
    return tags
end

# Intersection of 1-based node ranges, empty allowed.
_isect(a::UnitRange{Int}, b::UnitRange{Int}) =
    max(first(a), first(b)):min(last(a), last(b))

# Padded local range of patch-node range `r` for a rank whose block starts at
# `off` with pad `pad`: node g ↦ padded index g - off + pad.
_padded(r::UnitRange{Int}, off::Int, pad::Int) = (first(r)-off+pad):(last(r)-off+pad)

# The record copies below take two forms. Host storage runs the scalar loops
# the root path was measured under. Device storage stages every message the
# way the halo exchange does (halo.jl): a strided block packs by broadcast
# into a contiguous device stage, one copy crosses to the host MPI buffer,
# and the receive side reverses it; a local copy or combination between two
# device patches is a broadcast between the two views. Each form applies the
# same arithmetic per element, so the two are bitwise.

# The block of `Q` over the padded ranges `r`, every component.
@inline _block_view(Q, r::NTuple{3,UnitRange{Int}}) =
    view(parent(Q), r[1], r[2], r[3], 1:size(Q, 4))

function _pack!(buf::AbstractVector, Q, r::NTuple{3,UnitRange{Int}})
    if !_device_path(Q)
        idx = 1
        @inbounds for c in 1:size(Q, 4), k in r[3], j in r[2], i in r[1]
            buf[idx] = Q[i, j, k, c]
            idx += 1
        end
        return buf
    end
    v = _block_view(Q, r)
    n = length(v)
    dsend = _device_send_stage(parent(Q), n)
    reshape(view(dsend, 1:n), size(v)) .= v
    _tracked_copy!(buf, 1, dsend, 1, n)
    return buf
end

function _unpack!(Q, buf::AbstractVector, r::NTuple{3,UnitRange{Int}})
    if !_device_path(Q)
        idx = 1
        @inbounds for c in 1:size(Q, 4), k in r[3], j in r[2], i in r[1]
            Q[i, j, k, c] = buf[idx]
            idx += 1
        end
        return Q
    end
    v = _block_view(Q, r)
    n = length(v)
    drecv = _device_send_stage(parent(Q), n)
    _tracked_copy!(drecv, 1, buf, 1, n)
    v .= reshape(view(drecv, 1:n), size(v))
    return Q
end

# Q ← w·Q + (1 − w)·buf over `r`; the equal-weight form keeps the mean's
# original arithmetic, since the root path is held bitwise against it. The half
# is taken in the state's type: a bare 0.5 widens each Float32 sum to Float64
# for the multiply, which rounds back to the same value (0.5 scales exactly)
# but emits a double-precision instruction per point.
function _combine_from!(Q, buf::AbstractVector, r::NTuple{3,UnitRange{Int}}, w)
    half = eltype(Q)(0.5)
    wT = eltype(Q)(w)
    vT = one(wT) - wT
    if !_device_path(Q)
        idx = 1
        if w == 0.5
            @inbounds for c in 1:size(Q, 4), k in r[3], j in r[2], i in r[1]
                Q[i, j, k, c] = half * (Q[i, j, k, c] + buf[idx])
                idx += 1
            end
        else
            @inbounds for c in 1:size(Q, 4), k in r[3], j in r[2], i in r[1]
                Q[i, j, k, c] = wT * Q[i, j, k, c] + vT * buf[idx]
                idx += 1
            end
        end
        return Q
    end
    v = _block_view(Q, r)
    n = length(v)
    drecv = _device_send_stage(parent(Q), n)
    _tracked_copy!(drecv, 1, buf, 1, n)
    b = reshape(view(drecv, 1:n), size(v))
    if w == 0.5
        v .= half .* (v .+ b)
    else
        v .= wT .* v .+ vT .* b
    end
    return Q
end

function _copy_block!(Qdst, rdst::NTuple{3,UnitRange{Int}},
                      Qsrc, rsrc::NTuple{3,UnitRange{Int}})
    if !_device_path(Qdst)
        @inbounds for c in 1:size(Qdst, 4)
            for (kd, ks) in zip(rdst[3], rsrc[3]), (jd, js) in zip(rdst[2], rsrc[2])
                for (id, is) in zip(rdst[1], rsrc[1])
                    Qdst[id, jd, kd, c] = Qsrc[is, js, ks, c]
                end
            end
        end
        return Qdst
    end
    _block_view(Qdst, rdst) .= _block_view(Qsrc, rsrc)
    return Qdst
end

# Both copies ← wa·Qa + (1 − wa)·Qb, from the old values, so a pairing that
# appears once from each side is idempotent for any weight.
function _combine_blocks!(Qa, ra::NTuple{3,UnitRange{Int}},
                          Qb, rb::NTuple{3,UnitRange{Int}}, wa)
    wT = eltype(Qa)(wa)
    vT = one(wT) - wT
    half = eltype(Qa)(0.5)     # in the state's type; see `_combine_from!`
    if !_device_path(Qa)
        @inbounds for c in 1:size(Qa, 4)
            for (ka, kb) in zip(ra[3], rb[3]), (ja, jb) in zip(ra[2], rb[2])
                for (ia, ib) in zip(ra[1], rb[1])
                    m = wa == 0.5 ? half * (Qa[ia, ja, ka, c] + Qb[ib, jb, kb, c]) :
                                    wT * Qa[ia, ja, ka, c] + vT * Qb[ib, jb, kb, c]
                    Qa[ia, ja, ka, c] = m
                    Qb[ib, jb, kb, c] = m
                end
            end
        end
        return Qa
    end
    # The a-side takes the combination first, from both old values, and the
    # b-side then copies it: the same value in both, as the scalar form.
    va = _block_view(Qa, ra)
    vb = _block_view(Qb, rb)
    if wa == 0.5
        va .= half .* (va .+ vb)
    else
        va .= wT .* va .+ vT .* vb
    end
    vb .= va
    return Qa
end

# --- Record construction ----------------------------------------------------

# One row per (world rank, patch) pair: who owns which block of which patch.
struct BlockEntry
    rank::Int
    pid::Int
    offset::NTuple{3,Int}     # patch-local node offset of the rank's block
    n_local::NTuple{3,Int}
end

# Every rank's block, known everywhere. Each rank contributes its blocks
# through a single Allgatherv: one block with the rank set partitioned over
# root slabs, a block of each tile it holds on a refined level, any number
# per rank. A serial run holds every patch and communicates nothing.
function _block_table(comm::MPI.Comm, my_pids::Vector{Int}, my_decomps)
    np = MPI.Comm_size(comm)
    if np == 1
        return [BlockEntry(0, my_pids[i], my_decomps[i].offset, my_decomps[i].n_local)
                for i in eachindex(my_pids)]
    end
    mine = Int64[]
    for (p, d) in zip(my_pids, my_decomps)
        append!(mine, Int64[p, d.offset..., d.n_local...])
    end
    counts = MPI.Allgather(Cint(length(mine)), comm)
    flat = Vector{Int64}(undef, sum(counts))
    MPI.Allgatherv!(mine, MPI.VBuffer(flat, counts), comm)
    table = BlockEntry[]
    at = 0
    for r in 0:np-1, _ in 1:(counts[r + 1] ÷ 7)
        e = flat[at .+ (1:7)]
        push!(table, BlockEntry(r, Int(e[1]), (Int(e[2]), Int(e[3]), Int(e[4])),
                                (Int(e[5]), Int(e[6]), Int(e[7]))))
        at += 7
    end
    return table
end

"""
    build_interface_records(T, comm, regions, faces_all, my_pids, my_decomps,
                            n_cons) -> (ghost_sends, ghost_recvs, plane_pairs)

Construct the interface exchange records for this rank's patches: ghost sends,
ghost receives, and shared-plane averaging pairs, each carrying padded local
index ranges, the partner world rank, a deterministic tag, and its buffer.
Collective over `comm` (one Allgatherv of the per-rank blocks); every rank
derives every record it participates in from the same table, so senders and
receivers agree without negotiation.
"""
function build_interface_records(::Type{T}, comm::MPI.Comm,
                                 regions::Vector{BlockRegion}, faces_all,
                                 my_pids::Vector{Int}, my_decomps,
                                 n_cons::Int;
                                 padded_transverse::NTuple{3,Bool}=(false, false, false)
                                 ) where {T}
    npatch = length(regions)
    ghost_sends = GhostRecord{T}[]
    ghost_recvs = GhostRecord{T}[]
    plane_pairs = PlaneRecord{T}[]
    npatch == 1 && return ghost_sends, ghost_recvs, plane_pairs
    # The name of each record's message, received and sent, for `_pair_tags`.
    recv_names = NTuple{4,Int}[]
    send_names = NTuple{4,Int}[]
    plane_rnames = NTuple{4,Int}[]
    plane_snames = NTuple{4,Int}[]
    table = _block_table(comm, my_pids, my_decomps)
    me = MPI.Comm_rank(comm)
    local_of = Dict(p => i for (i, p) in enumerate(my_pids))

    # Padded ranges of patch-node ranges for a given block.
    padded3(rs, off, pad) = ntuple(d -> _padded(rs[d], off[d], pad[d]), 3)

    for (li, p) in enumerate(my_pids)
        dp = my_decomps[li]
        pad = dp.n_halo_d
        n_halo = dp.n_halo
        myint = ntuple(d -> dp.offset[d]+1:dp.offset[d]+dp.n_local[d], 3)
        # Transverse ranges: the block interior, or the block padded by the
        # halo along dimensions whose ghosts an earlier phase has already
        # filled (the dimension-phased sync of a tiled level, levels.jl),
        # which is how a record reaches the edge and corner ghosts.
        widen(r, d) = padded_transverse[d] ? ((first(r) - pad[d]):(last(r) + pad[d])) : r
        myrng = ntuple(d -> widen(myint[d], d), 3)
        for ds in 1:3, side in 1:2
            q = faces_all[p][ds][side]
            q == 0 && continue
            n_ext = regions[p].extent[ds]
            nq_ext = regions[q].extent[ds]
            otherside = side == 1 ? 2 : 1
            # This rank participates only if it owns the face plane.
            owns = side == 1 ? dp.offset[ds] == 0 :
                               dp.offset[ds] + dp.n_local[ds] == n_ext
            owns || continue
            # Map between my patch-node index m and the partner's g along ds:
            # hi face: m = n_ext + (g - 1); lo face: m = 1 - (nq_ext - g).
            tomine(g) = side == 2 ? n_ext + g - 1 : g - nq_ext + 1
            # My ghost layers correspond to partner interior nodes:
            src_q = side == 2 ? (2:n_halo+1) : (nq_ext-n_halo:nq_ext-1)
            # The partner's ghost layers correspond to my interior nodes:
            src_p = side == 2 ? (n_ext-n_halo:n_ext-1) : (2:n_halo+1)
            # Their ghost node for my source node m, in their patch-node space:
            totheirs(m) = side == 2 ? m - n_ext + 1 : m + nq_ext - 1
            plane_p = side == 2 ? n_ext : 1
            plane_q = side == 2 ? 1 : nq_ext
            rname = (q, p, ds, side)
            sname = (p, q, ds, otherside)
            for e in table
                e.pid == q || continue
                eint = ntuple(d -> e.offset[d]+1:e.offset[d]+e.n_local[d], 3)
                tover = ntuple(d -> d == ds ? (1:1) :
                               _isect(myrng[d], widen(eint[d], d)), 3)
                any(d -> d != ds && isempty(tover[d]), 1:3) && continue
                # Only the partner block that owns its own face plane holds the
                # ghost-source nodes (its extent is at least the scheme minimum,
                # which exceeds n_halo + 1) and carries interface ghosts of its
                # own to fill.
                owns_e = side == 1 ? e.offset[ds] + e.n_local[ds] == nq_ext :
                                     e.offset[ds] == 0
                # --- ghost receive: partner nodes src_q ∩ e's block → my ghosts
                dsq = owns_e ? _isect(src_q, eint[ds]) : (1:0)
                if !isempty(dsq) && e.rank != me
                    # (the e.rank == me case is realized as a local copy on the
                    # send side, so no receive record is built for it)
                    mine_r = ntuple(d -> d == ds ?
                        (tomine(first(dsq)):tomine(last(dsq))) : tover[d], 3)
                    minep = padded3(mine_r, dp.offset, pad)
                    buf = Vector{T}(undef, n_cons * prod(length.(minep)))
                    push!(ghost_recvs, GhostRecord{T}(li, e.rank, 0, 0,
                          minep, minep, buf))
                    push!(recv_names, rname)
                end
                # --- ghost send: my nodes src_p ∩ my block, needed by e
                dsp = owns_e ? _isect(src_p, myint[ds]) : (1:0)
                if !isempty(dsp)
                    push!(send_names, sname)
                    src = ntuple(d -> d == ds ? dsp : tover[d], 3)
                    srcp = padded3(src, dp.offset, pad)
                    if e.rank == me
                        lq = local_of[q]
                        dq = my_decomps[lq]
                        theirs_ds = totheirs(first(dsp)):totheirs(last(dsp))
                        dst = ntuple(d -> d == ds ? theirs_ds : tover[d], 3)
                        dstp = padded3(dst, dq.offset, dq.n_halo_d)
                        push!(ghost_sends, GhostRecord{T}(li, me, lq, 0,
                              srcp, dstp, Vector{T}()))
                    else
                        buf = Vector{T}(undef, n_cons * prod(length.(srcp)))
                        push!(ghost_sends, GhostRecord{T}(li, e.rank, 0, 0,
                              srcp, srcp, buf))
                    end
                end
                # --- shared plane: my node plane_p ↔ partner node plane_q
                if plane_q in eint[ds]
                    push!(plane_rnames, rname)
                    push!(plane_snames, sname)
                    mine_r = ntuple(d -> d == ds ? (plane_p:plane_p) : tover[d], 3)
                    minep = padded3(mine_r, dp.offset, pad)
                    if e.rank == me
                        lq = local_of[q]
                        dq = my_decomps[lq]
                        theirs_r = ntuple(d -> d == ds ? (plane_q:plane_q) : tover[d], 3)
                        theirsp = padded3(theirs_r, dq.offset, dq.n_halo_d)
                        push!(plane_pairs, PlaneRecord{T}(li, me, lq, q, 0, 0,
                              minep, theirsp, Vector{T}(), Vector{T}()))
                    else
                        nvals = n_cons * prod(length.(minep))
                        push!(plane_pairs, PlaneRecord{T}(li, e.rank, 0, q, 0,
                              0, minep, minep, Vector{T}(undef, nvals),
                              Vector{T}(undef, nvals)))
                    end
                end
            end
        end
    end
    rtags = _pair_tags([r.partner for r in ghost_recvs], recv_names,
                       _GHOST_TAG_BASE, me)
    stags = _pair_tags([r.partner for r in ghost_sends], send_names,
                       _GHOST_TAG_BASE, me)
    prtags = _pair_tags([r.partner for r in plane_pairs], plane_rnames,
                        _PLANE_TAG_BASE, me)
    pstags = _pair_tags([r.partner for r in plane_pairs], plane_snames,
                        _PLANE_TAG_BASE, me)
    retag(r::GhostRecord{T}, tag) =
        GhostRecord{T}(r.patch, r.partner, r.partner_patch, tag, r.mine, r.theirs, r.buf)
    retag(r::PlaneRecord{T}, tag, sendtag) =
        PlaneRecord{T}(r.patch, r.partner, r.partner_patch, r.partner_pid, tag, sendtag,
                       r.mine, r.theirs, r.buf, r.sbuf)
    return GhostRecord{T}[retag(r, t) for (r, t) in zip(ghost_sends, stags)],
           GhostRecord{T}[retag(r, t) for (r, t) in zip(ghost_recvs, rtags)],
           PlaneRecord{T}[retag(r, t, u) for (r, t, u) in zip(plane_pairs, prtags, pstags)]
end

# --- Runtime exchange -------------------------------------------------------

"""
    exchange_patch_ghosts!(solver, states)

Fill every patch's interface ghost layers from the abutting patch's interior
values of the conserved state, and return `states` (the per-patch state vector
aligned with `solver.patches`). Ghost strips span each rank's own transverse
interior; the following within-patch [`exchange_state!`](@ref) carries them
into edge and corner halos using the sequential-dimension halo exchange that
fills corners. Collective over the ranks owning either side of any
interface; a run with one patch returns immediately.
"""
exchange_patch_ghosts!(solver, states) =
    _exchange_ghosts!(solver, states, solver.comm, solver.ghost_sends,
                      solver.ghost_recvs)

# `comm` must be the communicator the records were built over, because each
# record's `partner` is a rank number in that communicator: `solver.comm` for
# the root's records, the owning level's communicator for a refined level's
# (levels.jl). It is a parameter rather than a read of `solver.comm` so that
# a refined level's exchange stays among its subset's ranks.
function _exchange_ghosts!(solver, states, comm::MPI.Comm, sends, recvs)
    (isempty(recvs) && isempty(sends)) && return states
    me = MPI.Comm_rank(comm)
    reqs = MPI.Request[]
    for r in recvs
        r.partner == me && continue
        push!(reqs, MPI.Irecv!(r.buf, comm; source=r.partner, tag=r.tag))
    end
    for s in sends
        if s.partner == me
            _copy_block!(states[s.partner_patch], s.theirs, states[s.patch], s.mine)
        else
            _pack!(s.buf, states[s.patch], s.mine)
            push!(reqs, MPI.Isend(s.buf, comm; dest=s.partner, tag=s.tag))
        end
    end
    MPI.Waitall(reqs)
    for r in recvs
        r.partner == me && continue
        _unpack!(states[r.patch], r.buf, r.mine)
    end
    return states
end

"""
    average_shared_planes!(solver, states)

Make each shared interface-plane node consistent by replacing both patches'
copies with their mean, over each rank's transverse interior, and return
`states`. Both sides compute the identical mean of the identical pair, so the
result does not depend on which side is asked. Collective as
[`exchange_patch_ghosts!`](@ref) is; no-op with one patch.
"""
average_shared_planes!(solver, states) =
    _average_planes!(solver, states, solver.comm, solver.plane_pairs)

_average_planes!(solver, states, comm::MPI.Comm, planes) =
    _combine_planes!(solver, states, comm, planes, _ -> 0.5)

# Each shared-plane node ← w·(own copy) + (1 − w)·(partner's copy), with
# `wself(record)` the weight of this side; the two sides' weights must sum
# to one, which the mean (0.5 everywhere) and a one-way seeding (0 on the
# taking side, 1 on the giving side) both satisfy.
function _combine_planes!(solver, states, comm::MPI.Comm, planes,
                          wself::F) where {F}
    isempty(planes) && return states
    me = MPI.Comm_rank(comm)
    reqs = MPI.Request[]
    for pl in planes
        if pl.partner == me
            # A local pairing appears once from each side; the combination
            # writes both copies from the old values and is idempotent, so
            # the second application is a no-op, not a double count.
            _combine_blocks!(states[pl.patch], pl.mine,
                             states[pl.partner_patch], pl.theirs, wself(pl))
        else
            push!(reqs, MPI.Irecv!(pl.buf, comm; source=pl.partner, tag=pl.tag))
        end
    end
    for pl in planes
        pl.partner == me && continue
        _pack!(pl.sbuf, states[pl.patch], pl.mine)
        # The send uses the RECEIVER's tag, computed at record build time.
        push!(reqs, MPI.Isend(pl.sbuf, comm; dest=pl.partner, tag=pl.sendtag))
    end
    MPI.Waitall(reqs)
    for pl in planes
        pl.partner == me && continue
        _combine_from!(states[pl.patch], pl.buf, pl.mine, wself(pl))
    end
    return states
end

"""
    sync_patches!(solver, states)

Bring the whole multi-patch state to mutual consistency: average the shared
interface planes, refill the interface ghost layers from the (averaged)
neighboring interiors, at the root and on every refined level through that
level's own records, and exchange each patch's own rank-boundary halos so
edge and corner cells agree with both. Called by the multi-patch drivers
after every RK stage update and at the head of each `run!` iteration; a
single-patch solver never reaches it. Collective over `solver.comm` for the
root's records and over each refined level's own communicator for that
level's, which only the level's owners enter.
"""
function sync_patches!(solver, states)
    average_shared_planes!(solver, states)
    exchange_patch_ghosts!(solver, states)
    levels = getfield(solver, :levels)
    for ℓ in 2:length(levels)
        # The records of a refined level are addressed in that level's own
        # communicator, so only its owners may enter them; a rank without
        # state on the level holds no record to run either.
        levels[ℓ].level_comm.owned || continue
        _sync_level_records!(solver, states, levels[ℓ])
    end
    for (i, patch) in enumerate(solver.patches)
        _exchange_patch_state!(states, i, _cold(patch))
    end
    return states
end

# The rank halos of `states[i]` on `patch`, behind a barrier on the patch's
# concrete type (`_unit_call`): its decomposition, read from an abstractly typed
# patch, would be boxed.
_exchange_patch_state!(states, i::Int, patch) =
    exchange_state!(states[i], patch.decomp)
