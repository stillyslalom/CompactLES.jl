# Multi-rank test suite — the companion to the serial runtests.jl.
#
# Run with (bundled MS-MPI mpiexec from MPI.jl):
#   mpiexec -n 2 julia --project=. -t 1 test/mpi_tests.jl
#   mpiexec -n 4 julia --project=. -t 1 test/mpi_tests.jl
#   mpiexec -n 8 julia --project=. -t 1 test/mpi_tests.jl
#
# The phases are functions defined in the files of test/mpi/, one per
# subsystem, which this file includes after its helpers. `phases=` runs a
# subset of the suite by phase name, comma-separated (the names are the first
# elements of SUITE at the bottom of this file):
#   mpiexec -n 8 julia --project=. -t 1 test/mpi_tests.jl "phases=halo consistency"
#
# `rank_shape=true` runs the phases whose code path depends on the rank count
# (RANK_SHAPE_PHASES, beside SUITE), the selection CI and the core gate run at
# eight ranks:
#   mpiexec -n 8 julia --project=. -t 1 test/mpi_tests.jl rank_shape=true
#
# This suite targets the code paths that ONLY execute when a dimension is
# split across more than one rank and are therefore UNREACHABLE from the serial
# suite (which runs at np == 1):
#
#   1. The cross-rank reduced-interface (spike) line solve — for both the
#      tridiagonal C6 solver (LineSolver) and the pentadiagonal C10 solver
#      (BandLineSolver): Allgather of the interface blocks, the dense reduced
#      LU, and the threaded spike correction. Serial runs skip this entirely
#      for non-periodic lines and only self-couple for periodic ones.
#   2. Halo exchange across real rank boundaries (exchange_halos! /
#      exchange_dim_batch!), which the np == 1 case exercises only as a
#      periodic self-wrap.
#   3. Off-rank coordinate-singularity folds — the pair_forward!/pair_backward!
#      butterfly that does MPI.Sendrecv! of whole blocks (the e-keeper/o-keeper
#      split), reached only when the pairing/reversed dimension is split, for
#      the derivatives, the filters and the :delta4 detector.
#   4. Freestream preservation with a fold's antipodal partner, or the θ line
#      through the poles, split across ranks.
#   5. Telescoping flux conservation across rank boundaries.
#   6. Global synchronization — that every rank agrees on dt and on the
#      simulation time after a real transient. A missing Allreduce leaves the
#      ranks integrating different equations and cannot be seen serially.
#   7. Per-rank checkpoint round trips under a real decomposition.
#
# Correctness strategy (see the module note in banded.jl / tridiag.jl): the
# reduced-interface method is designed to reproduce the *exact* single-domain
# solution regardless of the rank count, so any interface bug shows up as an
# O(1) error, not a small one. Every test therefore reduces a max-abs error
# (or a global sum) to a single scalar with MPI.Allreduce — so ALL ranks assert
# the SAME value and fail together — and compares it to the analytic result at
# the tolerance the serial suite achieves on the same operator.

# Timing first, ensuring package load is measured rather than assumed. Every
# rank compiles this suite independently, so the compile column here is
# per-rank work that the rank count multiplies. See test/timing.jl.
include("timing.jl")

@phase "package load" begin
    using MPI
    MPI.Init(threadlevel=:funneled)
    @isdefined(build_solver) || include(joinpath(@__DIR__, "build_solver.jl"))
    @isdefined(use_test_image) || include(joinpath(@__DIR__, "test_image.jl"))
    using CompactLES
    import CompactLES: Decomp, exchange_halos!, interior, field, DirPlan, BandPlan,
                       DevicePlan, device_plan, apply_along!, filter_field!,
                       amr_transfer_schemes, amr_restriction_scheme,
                       amr_prolongation_scheme, amr_interpolation_weights,
                       TransferPlan, plan_transfer, restrict!, prolong!,
                       plan_direction, Patch, PatchSolver, InterfaceBC, CoarseFineBC,
                       LevelTransfer, Level, LevelComm, exchange_patch_ghosts!,
                       average_shared_planes!, prolong_level_ghosts!, restrict_level!,
                       scalar_field, container_extension, PLANCK_TIME,
                       THREAD_MIN_WORK, script_args, script_grid
    using LinearAlgebra
    import KernelAbstractions
end
using CompactLES: compute_rhs!, apply_bcs!, filter_state!, CPUBackend, padded_index, xcoord,
                  global_xcoord
using CompactLES.Regions

const CL = CompactLES
const comm = MPI.COMM_WORLD
const np = MPI.Comm_size(comm)
const rank = MPI.Comm_rank(comm)

if np == 1
    rank == 0 && println("mpi_tests.jl is the MULTI-RANK suite; run under " *
                         "mpiexec -n ≥2 (serial checks live in runtests.jl). " *
                         "Nothing to do at np=1.")
    MPI.Finalize()
    exit(0)
end

# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------

const per3 = ntuple(_ -> (PeriodicBC(), PeriodicBC()), 3)

# A regridded level with fewer tiles than ranks takes a finer lattice, so a
# decomposed run tiles its levels differently from the serial run the
# phases compare it with. The regridding solvers of the suite keep every
# level on its configured lattice (`subdivide_lattice = false`); the
# "lattice subdivision" and "lattice over nodes" phases leave it on.

"Max interior error of a scalar field against an analytic function (this rank)."
function ferr(solver, f, fn)
    e = 0.0
    for k in 1:solver.decomp.n_local[3], j in 1:solver.decomp.n_local[2], i in 1:solver.decomp.n_local[1]
        e = max(e, abs(f[padded_index(solver, i, j, k)] -
                       fn(xcoord(solver, 1, i), xcoord(solver, 2, j), xcoord(solver, 3, k))))
    end
    e
end

fillf!(solver, f, fn) = (for k in 1:solver.decomp.n_local[3], j in 1:solver.decomp.n_local[2],
                        i in 1:solver.decomp.n_local[1]
    f[padded_index(solver, i, j, k)] = fn(xcoord(solver, 1, i), xcoord(solver, 2, j),
                                          xcoord(solver, 3, k))
end; f)

gmax(x) = MPI.Allreduce(Float64(x), MPI.MAX, comm)
gsum(x) = MPI.Allreduce(Float64(x), MPI.SUM, comm)

const _npass = Ref(0)
const _nfail = Ref(0)

"Assert a globally-reduced scalar `val` is below `tol`. All ranks see the same
`val`, so all ranks agree on pass/fail. Rank 0 prints one line."
function check(name, val, tol)
    ok = isfinite(val) && val < tol
    ok ? (_npass[] += 1) : (_nfail[] += 1)
    rank == 0 && @printf("    %-52s  %10.3e < %-9.1e  %s\n",
                         name, val, tol, ok ? "ok" : "FAIL")
    return ok
end

section(name) = rank == 0 && println("\n$name")

# A logger keeping the messages of this rank's records at Warn and above, so a
# phase can count the warnings a call raises on each rank.
struct WarnRecorder <: Base.CoreLogging.AbstractLogger
    messages::Vector{String}
end
Base.CoreLogging.min_enabled_level(::WarnRecorder) = Base.CoreLogging.Warn
Base.CoreLogging.shouldlog(::WarnRecorder, args...) = true
Base.CoreLogging.catch_exceptions(::WarnRecorder) = false
Base.CoreLogging.handle_message(r::WarnRecorder, level, message, args...; kwargs...) =
    (push!(r.messages, string(message)); nothing)

"The value of `f()` and the warning messages it raised on this rank."
function recorded_warnings(f)
    recorder = WarnRecorder(String[])
    value = Base.CoreLogging.with_logger(f, recorder)
    return value, recorder.messages
end

"Process grid that puts all np ranks on dimension `ax`, one elsewhere."
splitdims(ax) = ntuple(d -> d == ax ? np : 1, 3)

# Global extent for any decomposed dimension. Every active dimension carries a
# C8 filter plan whose closure needs ≥ 9 points per rank, so the split dim must
# give ≥ 9 locally at the largest np we run (8): 72 / 8 = 9. 72 is even and
# divisible by 2, 4, 8, satisfying the folds' uniform-even-block requirement.
const SPLITN = 72

rank == 0 && println("=== CompactLES multi-rank test suite (np = $np) ===")

# The phases, by subsystem.
include("mpi/operators.jl")
include("mpi/runtime.jl")
include("mpi/output.jl")
include("mpi/hierarchy.jl")
include("mpi/artificial.jl")
include("mpi/patches.jl")
include("mpi/devices.jl")
include("mpi/levels.jl")
include("mpi/boundaries.jl")
include("mpi/implicit.jl")

include("wall_flux_mpi.jl")
include("conservation_mpi.jl")
include("positivity_mpi.jl")

const SUITE = (
    ("periodic C6", test_periodic_c6),
    ("pentadiagonal C10", test_pentadiagonal_c10),
    ("closed C6", test_closed_c6),
    ("no-slip wall flux", test_no_slip_wall_flux),
    ("slip wall flux", test_slip_wall_flux),
    ("device line solves", test_device_lines),
    ("staged device exchange", test_staged_exchange),
    ("distributed refinement", test_refined_decomposed),
    ("level rank subsets", test_level_subset),
    ("tiled refinement", test_tiled_level),
    ("partitioned coupling", test_partitioned_coupling),
    ("covered masks", test_covered_masks),
    ("AMR transfer pair", test_transfer_pair),
    ("staggered operators", test_staggered),
    ("implicit conduction", test_imex_pair),
    ("incompressible", test_incompressible),
    ("halo consistency", test_halo_consistency),
    ("off-rank folds", test_offrank_folds),
    ("mode truncation", test_mode_truncation),
    ("symmetry plane", test_symmetry_plane),
    ("level at a symmetry plane", test_plane_level),
    ("level on the axis", test_axis_level),
    ("folded regrid", test_fold_regrid),
    ("NSCBC inflow", test_nscbc_inflow),
    ("NSCBC level face", test_nscbc_level),
    ("turbulent inflow", test_turbulent_inflow),
    ("composite face", test_composite_face),
    ("hydrostatic state", test_hydrostatic),
    ("freestream", test_freestream),
    ("conservation", test_conservation),
    ("composite budgets", test_composite_budgets),
    ("sync", test_sync),
    ("artificial decomposition", test_artificial_decomposition),
    ("bulk species channel", test_bulk_decomposition),
    ("d8 detector decomposition", test_ring_detector_decomposition),
    ("positivity floor", test_positivity_floor),
    ("positivity limiter", test_positivity_limiter),
    ("state validity", test_state_validity),
    ("callback consistency", test_callback_consistency),
    ("observation and clock", test_observation_clock),
    ("state queries", test_state_queries),
    ("field writer", test_field_writer),
    ("shared piece planes", test_shared_planes),
    ("slicing", test_slicing),
    ("line sample", test_line_sample),
    ("checkpoint", test_checkpoint),
    ("wall limit", test_wall_limit),
    ("hierarchy checkpoint", test_hierarchy_checkpoint),
    ("phase change", test_phase_change),
    ("hierarchy phase change", test_hierarchy_phase_change),
    ("unrefined start", test_unrefined_start),
    ("deep regrid", test_deep_regrid),
    ("deep regrid subsets", test_deep_regrid_subsets),
    ("lattice subdivision", test_lattice_subdivision),
    ("lattice over nodes", test_lattice_nodes),
    ("placed levels", test_placed_levels),
    ("seam levels", test_seam_levels),
    ("two-patch layout", test_two_patch_layout),
    ("bulk patched layout", test_bulk_patched),
)

# The phases whose code path depends on the rank count, which `rank_shape=true`
# selects. CI and the core gate in CLAUDE.md run them at eight ranks, a count
# that oversubscribes the CI runner (see .github/workflows/CI.yml), where the
# full suite costs minutes per rank. Most tests put all ranks on one dimension
# of SPLITN = 72 points via `splitdims`, and at np = 8 that is 9 points per
# rank, the C8 filter's minimum block and the smallest the reduced-interface
# solves, the transfer operators, the fold butterfly, the positivity repair
# and the wall closures of "closed C6" ever see: those phases run. The
# corner-halo check's default Dims_create grid changes shape outright (2x2x2
# at np = 8, 2x1x1 at np = 2), a tiled level of four tiles leaves four of
# eight ranks without one, and the lattice subdivision takes a different
# lattice at each count, over nodes of a different size. A composite face's
# member mask splits across more ranks,
# and the deep-regrid levels fall onto rank subsets and move a tile only at
# np = 8. The level coupling's point-to-point messages pair different ranks at
# each count, and so do the owner ranges a hierarchy's phase change carries.
# The mode-truncation phase splits θ over np / 2 or np ranks and r in two. The
# slicing phase derives its over-coarse stride from np, and the shared-plane
# phase lays its ranks on a 2 x (np / 2) grid. The remaining phases
# (the timestepping runs, the writers, the checkpoint, the sensor comparisons,
# the phase change of a uniform grid) count ranks or change block sizes only,
# and they are the step-bound ones.
# The incompressible phase's pressure multigrid aggregates within the rank
# blocks, so its hierarchy changes with the count.
# The wall-flux phases are absent: a collective below an early return
# deadlocks once a rank lacks a wall, which every rank does at two ranks.
const RANK_SHAPE_PHASES = (
    "periodic C6", "pentadiagonal C10", "closed C6", "device line solves",
    "tiled refinement", "partitioned coupling", "AMR transfer pair",
    "staggered operators", "implicit conduction", "incompressible",
    "halo consistency",
    "off-rank folds", "mode truncation", "symmetry plane",
    "level at a symmetry plane", "level on the axis", "folded regrid",
    "NSCBC inflow", "NSCBC level face", "composite face", "freestream",
    "composite budgets", "positivity floor", "positivity limiter", "slicing",
    "shared piece planes", "hierarchy phase change", "deep regrid subsets",
    "lattice subdivision", "lattice over nodes", "placed levels", "seam levels",
)

# `phases=` selects phases by name from SUITE, comma-separated and in SUITE's
# order, and `rank_shape=true` selects RANK_SHAPE_PHASES; with neither given,
# every phase runs. An unknown name is an error on every rank, not a silently
# empty run.
const OPTS = script_args(ARGS, (phases = "", rank_shape = false))
const SELECTED = let names = first.(SUITE),
    wanted = OPTS.rank_shape ? collect(RANK_SHAPE_PHASES) :
             isempty(OPTS.phases) ? collect(names) :
             strip.(split(OPTS.phases, ','))
    OPTS.rank_shape && !isempty(OPTS.phases) &&
        throw(ArgumentError("phases= and rank_shape=true select a phase set each; " *
                            "give one"))
    unknown = setdiff(wanted, names)
    isempty(unknown) ||
        throw(ArgumentError("phases= names no phase of SUITE: " *
                            join(unknown, ", ")))
    Tuple(p for p in SUITE if first(p) in wanted)
end

try
    # Barrier before each timing, ensuring a slow rank in one test is charged to
    # that test rather than to the next one it holds up.
    for (name, testfn) in SELECTED
        MPI.Barrier(comm)
        @phase name testfn()
    end
catch e
    # A throw on any rank (e.g. a setup error) must not deadlock the others.
    println("rank $rank: uncaught exception: ", e)
    flush(stdout)
    MPI.Abort(comm, 1)
end

# Every check should reduce its metric across all ranks, so that _nfail is
# identical everywhere. One that tests a rank-local value can fail on a rank
# other than rank 0, which prints it as passing; the count is therefore
# reduced, so that such a check fails the run on every rank instead of one
# rank exiting alone and aborting the job.
const _nfail_local = _nfail[]
_nfail[] = MPI.Allreduce(_nfail_local, MPI.MAX, comm)
if rank == 0
    total = _npass[] + _nfail[]
    println("\n=== $(_npass[])/$total checks passed" *
            (_nfail[] == 0 ? "" : "  ($(_nfail[]) FAILED)") * " ===")
    _nfail[] > _nfail_local &&
        println("    a check failed on another rank only; its line above reads ok")
end
MPI.Barrier(comm)

# Rank 0's own breakdown, plus two whole-communicator numbers: the slowest
# rank's total, and the compile time summed over ranks. The second is the one
# that scales with np — every rank compiles this suite from scratch, so the job
# pays np times for the same code generation while the runner has far fewer
# cores than ranks.
let total = time() - T_SCRIPT_START,
    compile = sum(p -> p[3], PHASE_LOG; init=0.0)
    slowest = MPI.Allreduce(total, max, comm)
    compile_all = MPI.Allreduce(compile, +, comm)
    rank == 0 && timing_report(; title="mpi phase timing (np = $np, rank 0)",
                               extra=("slowest rank (script)" => slowest,
                                      "compiling, summed over ranks" => compile_all))
end

if _nfail[] > 0
    exit(1)
end
rank == 0 && println("multi-rank tests complete")
MPI.Finalize()
