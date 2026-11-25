# The process-global switches and counters that tests, benchmarks and probes
# set from outside the package, gathered in one file and included ahead of
# every file that reads one. None is a solver setting: a run that sets none of
# them takes the defaults, and the suites restore each after use, through
# `with_knob` where the setting brackets a block. `THREAD_MIN_WORK` stays in
# threading.jl, since it is a tunable of the session read from the
# environment, and `FIXED_LATTICE_COSTS` beside the cost model in regrid_lattice.jl.

"""
    with_knob(f, knob => value, ...)

Run `f()` with each `knob`, a `Ref` of this file, set to its `value`, and
restore the values the knobs held on the way out, whether `f` returns or
throws. The knobs are process-global, so a test that sets one and fails
before resetting it changes every testset after it; a block under
`with_knob` cannot.
"""
function with_knob(f, settings::Pair{<:Base.RefValue}...)
    saved = map(s -> s.first[], settings)
    for s in settings
        s.first[] = s.second
    end
    try
        return f()
    finally
        for (s, old) in zip(settings, saved)
            s.first[] = old
        end
    end
end

# --- Device paths on host storage ------------------------------------------------

"""
Test and benchmark toggle: `true` routes every [`pointwise!`](@ref) launch
through the KernelAbstractions path even on `Array` storage, where
`get_backend` supplies the KA CPU backend. The default `false` keeps `Array`
storage on the `@threaded` path, which is the configuration every guard was
measured under. The device testsets flip this around the runs they compare
against the `CPUBackend` bitwise; nothing else should.
"""
const FORCE_KA = Ref(false)

"""
Test toggle: `true` routes ordinary `Array` fields through every
device-storage path: the halo and fold-pair exchange staging (pack buffers,
offset copies, and all), the interface-record staging of a patch layout,
the level transfer's device chain on the fine patch's `LevelScratch`, the
regrid's block copies and the device tag sweep. On host storage each of
those performs the identical element copies and the identical per-point
arithmetic, so the suites flip this, with `FORCE_KA` for the kernel path,
to pin the device paths bitwise on machines without a GPU. The default
`false` keeps host fields on the direct forms.
"""
const FORCE_DEVICE_EXCHANGE = Ref(false)

"""
Launch synchronization policy: `false` (the default) defers. Kernels queue
on the backend's one in-order stream and the host synchronizes only at its
interaction points (the reduced-solve staging fence, the synchronous
device↔host copies of the halo/ring staging and the reductions); this
removes the per-launch round trip measured at 28–32% of a device step.
`true` restores the conservative mode, a host synchronize after every
pointwise launch and between every line-solve subkernel, and is the
correctness fallback: flip it when
bisecting a device discrepancy, so ordering bugs separate from arithmetic
ones.
"""
const DEVICE_SYNC = Ref(false)

# --- Transfer accounting -----------------------------------------------------------

# Transfer accounting for the distributed-device measurements: bytes staged
# through the device
# buffers and the wall time of the synchronous copies, accumulated only while
# a bench turns the toggle on (the timer synchronizes nothing extra, since the
# copies are synchronous, but the clock calls are not free).
const TRACK_DEVICE_TRANSFERS = Ref(false)
const DEVICE_TRANSFER_BYTES = Ref(0)
const DEVICE_TRANSFER_TIME = Ref(0.0)

# Reduced-interface transfer accounting: the per-apply device/host
# copies of the interface values and correction factors, tallied apart from
# the halo staging so the two shares are readable separately. Active only
# under `TRACK_DEVICE_TRANSFERS`, like the halo counters.
const REDUCED_TRANSFER_BYTES = Ref(0)
const REDUCED_TRANSFER_TIME = Ref(0.0)

# --- Tiles, lattices and migration -------------------------------------------------

"""
Test and benchmark toggle: a positive value spreads the tiles of one extent
held whole on host storage over that many [`RHSWorkspace`](@ref) sets and
evaluates a level's tiles as concurrent tasks (`_tile_groups`) whatever their
size and the session's thread count, so that one thread exercises the
concurrent path; 0, the default, takes the rule of `_workspace_slots`. Read
when the tiles are built and at every concurrent phase.
"""
const TILE_SLOTS = Ref(0)

# Test and bench toggle: a positive count groups the ranks into nodes of that
# many consecutive ranks in place of the nodes MPI reports (`_node_map`), so
# a run on one machine takes the paths of a run across nodes.
const NODE_SIZE = Ref(0)

# Bench toggle: rank 0 prints each lattice choice, the model and the
# predicted seconds per root step of every lattice.
const LATTICE_TRACE = Ref(false)

# Test hook. With the flag on, `_regrid_tiles!` also carries each moved tile
# through the replicated gather and `_carry_over!`, and adds to the result
# the tiles it audited on this rank and the slots at which the migrated
# state differs from that reference; the MPI suite holds the count at zero.
const MIGRATION_AUDIT = Ref(false)
const MIGRATION_AUDIT_RESULT = Ref((tiles=0, mismatches=0))

# Test hook: the number of `restrict_level!` calls in this process, from
# which a test reads the restrictions a run takes per step.
const RESTRICTION_COUNT = Ref(0)

# --- Parent-level masks ------------------------------------------------------------
#
# The switches of the masks a parent level applies where a child covers it.
# Each defaults to the production setting; the derivations and measurements
# are beside the code that reads them, in rhs.jl (the derivative, species
# gradient and sensor masks) and timestep_rk.jl (the filter residual mask).

const MASK_CHILD_DERIVATIVE = Ref(true)
const MASK_CHILD_SPECIES_GRADIENTS = Ref(true)
const MASK_CHILD_SENSORS = Ref(true)
const MASK_CHILD_RESIDUAL = Ref(true)
const CHILD_MASK_THRESHOLD = Ref(0.01)
const CHILD_MASK_REACH = Ref(2)
