# Cost of a refined Richtmyer–Meshkov run against uniform grids, and where a
# refined run's time goes. The case is the air/acetone/SF6 single-mode
# interface of docs/literate/richtmyer_meshkov.jl (Collins & Jacobs 2002,
# Mach 1.21, λ = 59.33 mm, a0 = 1.83 mm, a 5 mm diffuse layer): NSCBC inflow
# and outflow along x, whose inflow target switches to the gas behind the
# reflected shock when that shock reaches the inflow face, and symmetry planes
# at the crest and trough lines, half a wavelength apart.
#
#   dim=2  the page's channel, 25 cm by λ/2, the interface at x_i + a0 cos ky.
#   dim=3  the same channel with z resolved over λ/2 as well, symmetry planes
#          in y and z, the interface at x_i + a0 cos ky cos kz, and 18 cm
#          long (below).
#
# Configurations (`config=`), all at `nres` root nodes per wavelength
# (Δx = λ/nres; `nres/2` nodes across the half wavelength):
#
#   coarse  uniform at the root spacing
#   fine2   uniform at a third of it, the finest spacing of amr2: the root's
#           n_global (nx, ny, nz) becomes (3(nx − 1) + 1, 3ny, 3nz), which
#           puts its nodes on the refined level's lattice exactly (the x ends
#           are NSCBC faces, a node on each; the y and z ends are symmetry
#           planes half a cell beyond the last node)
#   fine3   uniform at a ninth, the finest spacing of amr3
#   amr2    the coarse root and one refined level
#   amr3    the coarse root and two refined levels
#
# Refinement is the production path: `AMR(initial = :sensor, tile, max_levels,
# subcycle)` with the default tag criterion (relative δ⁴ of the mixture
# density, threshold 0.02, buffer 4 root cells), regridding at the default
# interval (the steps a feature at the CFL limit takes to cross half the
# buffer: 4 at CFL 0.5), the default level interpolation order (8 under C6),
# injection restriction, the default `:ghost` interface flux and the
# conservative coupling at coarse-fine faces. Choices that can bias the
# comparison, and why:
#
# - `tile = 12` root nodes. The shock, the interface and later the reflected
#   and transmitted shocks are separated planar features, which a single box
#   would join into one; 12 is the edge of docs/literate/advected_bubbles.jl,
#   and at nres = 24 (12 root nodes across λ/2) one level-1 tile spans the
#   transverse extent, so a planar feature costs one row of tiles. The edge
#   applies on every level, in the parent's nodes, so a level-2 tile spans a
#   third of λ/2 and a planar feature costs 3 level-2 tiles per row in 2-D and
#   9 in 3-D. A smaller edge follows the features more tightly at a per-tile
#   cost (plan construction, coupling records) that grows quickly in 3-D.
#   `tile=` sets it, `regrid=` the regrid interval.
# - `subcycle = true`, the Berger–Oliger step a production run takes, against
#   the AMR default of false; `subcycle=false` runs the global step.
# - Rebalancing off, the AMR default: a surviving tile keeps its owner ranks
#   and a fresh one takes free ranks or joins a neighbor's group, so the
#   refined nodes per rank can be uneven. It is not available with two
#   regridded levels; `rebalance=` sets the threshold for amr2.
# - `nres = 24` is the smallest root the sensor-tagged start admits: AMR
#   seeds its tagging with a box that needs 12 root nodes across each
#   resolved dimension, here the half wavelength. 2-D fine2 is then the docs
#   page's resolution (72 per λ). The same root serves 3-D, where fine3
#   (216 per λ) would hold 901 × 108 × 108 nodes in the 25 cm channel, about
#   8 GB of solver state at the measured 0.8 kB per node of a uniform grid.
#   The 3-D channel is 18 cm long instead: the transmitted shock is at
#   x ≈ 14.4 cm at the end of the window, and 73 root nodes are the fewest
#   that 8 ranks can split (below). fine3 then holds 649 × 108 × 108 nodes,
#   6.9 GB of solver state and 15 GB resident over 8 ranks.
#   `Lx=` sets the length for every configuration alike, and `nmax=` stops a
#   run early with a projection of the full window.
# - The process grid: the MPI default where `setup` accepts it, otherwise
#   the accepted factorization with the smallest block surface. Every block
#   needs 9 nodes along each resolved dimension (the filter's stencil), so
#   the 12 root nodes across λ/2 cannot be split and 8 ranks lay the coarse
#   grid and a refined run's root out as slabs along x, which needs at least
#   72 root nodes there. `dims=` overrides. A refined level's tiles take the
#   rank subsets the solver assigns them.
#
# Window: `tfinal` = 0.45 ms. The shock reaches the interface at 0.085 ms,
# the reflected shock leaves through the inflow face at 0.442 ms, where the
# inflow target switches, and the transmitted shock is at x ≈ 14 cm at the
# end. Every configuration thus sees the impact, the ripple's compression and
# the start of its growth, and the refined levels follow four features
# (incident, transmitted and reflected shocks, interface) through regrids,
# the reflected shock out through the NSCBC face. The full page runs to 1 ms;
# this window is 45% of it.
#
# Report, from rank 0: one `rmicost,key=value,...` line for parsing, then a
# human block with the steps (root, and per level under subcycling, counting
# 3^ℓ substeps per root step while level ℓ holds tiles), the wall split into
# the first `warm` root steps (compilation) and the steady rest, steady ms
# per root step (mean and median), the steady wall per microsecond of
# simulated time (the number that compares configurations; a run stopped by
# `nmax` also projects the full window from it), the nodes per level at the
# end and their time average, the tiles per level and their sizes, nodes per
# rank per level, regrid checks and the excess wall of the steps that held
# one, busy and waiting time per rank (`solver.wall_wait`, the time inside
# the run-wide collectives and the level record exchange), allocation and GC
# time per step, peak resident memory, and the physics check: the integral
# mix width W = ∫ 4 X̄(1 − X̄) dx and the equivalent sharp-interface position
# x_c = ∫ (1 − X̄) dx of the plane-averaged SF6 mole fraction X̄, composite
# (each root station averaged over the finest data covering it) on a refined
# run. Agreement of amr2 with fine2 and of amr3 with fine3 in W and x_c is the
# check that the refined run computes the same flow.
#
# Per-step data comes from a trigger that records and never fires, so it
# perturbs nothing; a callback that fires makes `run!` repeat the level
# restriction before the next step, so `progress` (ProgressLog) is kept
# sparse.
#
# `profile=true` adds a sampling profile of the steady steps on every rank,
# binned by phase (the first phase in the list below whose function is on a
# sample's stack) and by level (the depth of `_advance_level!`; the global
# step and everything outside a subcycled level is "outer"), with the share of
# each phase's samples that sits inside an MPI.jl call (waiting or moving
# data). It prints ms per root step per phase: the mean and max over ranks,
# rank 0, and the busiest rank. The profiler perturbs the wall (within the
# run-to-run spread on 2-D amr3), so time with `profile=false` and read the
# phases from a second run.
#
#   mpiexec -n 8 julia --project=. -t 1 bench/rmicost.jl config=amr2 dim=2
#   mpiexec -n 8 julia --project=. -t 1 bench/rmicost.jl config=fine3 dim=3 nmax=200
#   mpiexec -n 8 julia --project=. -t 1 bench/rmicost.jl config=amr3 profile=true
#
# `affinity=0x5555` restricts every rank to one logical CPU per performance
# core of the 12900K workstation; 0 leaves the affinity alone.

using MPI
MPI.Init(threadlevel=:funneled)
using CompactLES
using CompactLES.Regions
using Printf
using Profile
using Statistics

const opt = CompactLES.script_args(ARGS, (
    config = "amr2", dim = 2, nres = 24, tfinal = 4.5e-4, nmax = typemax(Int),
    Lx = 0.0, tile = 12, subcycle = true, regrid = 0, warm = 10, progress = 50,
    profile = false, delay = 0.005, dims = "", affinity = 0, rebalance = 0.0))

const CONFIGS = ("coarse", "fine2", "fine3", "amr2", "amr3")
opt.config in CONFIGS || error("config must be one of $(join(CONFIGS, ", "))")
opt.dim in (2, 3) || error("dim must be 2 or 3")
iseven(opt.nres) || error("nres must be even (nres/2 nodes across λ/2)")

if opt.affinity != 0 && Sys.iswindows()
    ccall((:SetProcessAffinityMask, "kernel32"), stdcall, Cint, (Ptr{Cvoid}, Csize_t),
          ccall((:GetCurrentProcess, "kernel32"), stdcall, Ptr{Cvoid}, ()), opt.affinity)
end

# A format assembled from literal pieces, which `@printf` does not accept.
say(fmt, args...) = print(Printf.format(Printf.Format(fmt), args...))

# --- Problem (docs/literate/richtmyer_meshkov.jl) ---------------------------

const eos = IdealMixture(["Air", "C3H6O,acetone", "SF6"])
const p0, T0 = 92_555.0, 296.0
const air = Prim(Y = mass_fractions(eos, "Air" => 0.75, "C3H6O,acetone" => 0.25;
                                    basis = :mole), p = p0, T_ion = T0)
const sf6 = Prim(Y = mass_fractions(eos, "SF6" => 1.0; basis = :mole), p = p0,
                 T_ion = T0)
const incident = shock_jump(eos, air, 1.21)
const impact = riemann_interface(eos, incident.post, sf6)
const lambda = 0.05933
const kwave = 2pi / lambda
const a0 = 1.83e-3
const delta = 5e-3
const x_shock, x_interface = 0.05, 0.08
const t_impact = (x_interface - x_shock) / incident.shock_speed
const t_reflected = t_impact + x_interface / abs(impact.left_speed)

inflow_state(x, y, z, t) = t < t_reflected ? incident.post : impact.left

function problem(dim, Lx)
    ripple = dim == 2 ? (y, z) -> x_interface + a0 * cos(kwave * y) :
                        (y, z) -> x_interface + a0 * cos(kwave * y) * cos(kwave * z)
    return Problem(
        name = "air/SF6 Richtmyer–Meshkov",
        eos = eos,
        domain = ((0.0, Lx), (0.0, lambda / 2), (0.0, dim == 2 ? 1.0 : lambda / 2)),
        bcs = ((NSCBCInflowBC(incident.post; target = inflow_state),
                NSCBCOutflowBC(pinf = p0)),
               (SymmetryPlaneBC(), SymmetryPlaneBC()),
               dim == 2 ? PeriodicBC() : (SymmetryPlaneBC(), SymmetryPlaneBC())),
        ic = Layers(air,
                    Layer(Slab(1, lo = ripple), sf6; width = delta / sqrt(pi)),
                    Slab(1, hi = x_shock) => incident.post; profile = :erf),
    )
end

# The root grid at `nres` nodes per wavelength, as the page builds it, and the
# uniform grid whose spacing is that of refined level `level`.
function grid(dim, nres, Lx, level)
    ny = nres ÷ 2
    nx = round(Int, Lx / (lambda / 2 / ny))
    r = 3^level
    return (r * (nx - 1) + 1, r * ny, dim == 3 ? r * ny : 1)
end

# The process grids to try, in order: `dims=` alone when given, else MPI's
# default and then every other factorization of the ranks over the resolved
# dimensions by increasing block surface. `setup` rejects a grid that leaves
# some block too few nodes for a scheme (`check_block_extents`), on every rank
# alike, and the next is tried.
function process_grids(n_global, np)
    isempty(opt.dims) || return [Tuple(parse.(Int, split(opt.dims, ',')))]
    active = ntuple(d -> n_global[d] > 1, 3)
    auto = Int.(MPI.Dims_create(np, zeros(Cint, count(active))))
    grids = [ntuple(d -> active[d] ? popfirst!(auto) : 1, 3)]
    surface(g) = (n = ntuple(d -> n_global[d] / g[d], 3);
                  n[1] * n[2] + n[2] * n[3] + n[1] * n[3])
    others = NTuple{3,Int}[]
    for a in 1:np, b in 1:np
        np % (a * b) == 0 || continue
        g = (a, b, np ÷ (a * b))
        all(d -> active[d] || g[d] == 1, 1:3) && g != grids[1] && push!(others, g)
    end
    return [grids; sort(others; by = surface)]
end

function setup_on_some_grid(prob, num, np)
    for dims in process_grids(num.n_global, np)
        try
            solver, Q = setup(prob, Numerics(num; execution = Execution(dims = dims)))
            return solver, Q, dims
        catch err
            err isa ArgumentError && occursin("points per block", err.msg) || rethrow()
        end
    end
    error("no process grid of $np ranks fits n_global = $(num.n_global)")
end

# --- Per-step record -------------------------------------------------------

# A trigger that records every completed step and never fires: `run!` asks a
# trigger after each step, and a callback that runs makes the next step
# repeat the level restriction, which a recording must not cause. Everything
# it reads is rank-local.
mutable struct StepRecorder <: CompactLES.Trigger
    warm::Int
    profile::Bool
    nlev::Int
    wall::Vector{Float64}
    wait::Vector{Float64}
    dt::Vector{Float64}
    checks::Vector{Int}
    changed::Vector{Bool}
    tiles::Vector{Vector{Int}}        # per step, tiles per refined level (rank 0)
    points_dt::Vector{Float64}        # ∫ local nodes dt, per level
    point_steps::Vector{Float64}      # local nodes × substeps, per level
    regions::Vector{Vector{BlockRegion}}
    t_warm::Float64
    gc_warm::Tuple{Int64,UInt64}
    wait_warm::Float64
end

StepRecorder(warm, profile, nlev) =
    StepRecorder(warm, profile, nlev, Float64[], Float64[], Float64[], Int[], Bool[],
                 Vector{Int}[], zeros(nlev), zeros(nlev),
                 [BlockRegion[] for _ in 1:nlev], 0.0, (0, 0), 0.0)

gc_counters() = (Base.gc_bytes(), Base.gc_num().total_time)

function level_points(solver)
    levels = getfield(solver, :levels)
    patches = getfield(solver, :patches)
    return [sum((prod(patches[i].decomp.n_local) for i in lev.patches); init = 0)
            for lev in levels]
end

function CompactLES.fired!(r::StepRecorder, solver, Q)
    push!(r.wall, solver.wall_step)
    push!(r.wait, solver.wall_wait)
    push!(r.dt, solver.dt_prev)
    spec = getfield(solver, :regrid)
    push!(r.checks, spec === nothing ? 0 : spec.checks)
    nlev = nlevels(solver)
    tiles = [length(level_regions(solver, l)) for l in 1:nlev-1]
    push!(r.tiles, tiles)
    changed = false
    for l in 1:nlev-1
        regs = level_regions(solver, l)
        if regs != r.regions[l+1]
            changed = true
            r.regions[l+1] = regs
        end
    end
    push!(r.changed, changed && solver.step > 1)
    points = level_points(solver)
    sub = getfield(solver, :subcycle)
    for l in eachindex(points)
        r.points_dt[l] += points[l] * solver.dt_prev
        r.point_steps[l] += points[l] * (sub ? 3^(l - 1) : 1)
    end
    if solver.step == r.warm
        r.t_warm = solver.t
        r.gc_warm = gc_counters()
        r.wait_warm = solver.wait_total
        if r.profile
            Profile.clear()
            Profile.start_timer()
        end
    end
    return false
end

# --- Phase profile ---------------------------------------------------------

# A sample counts toward the first phase whose function is on its stack, so a
# phase nested in another (the coefficient pass of a regrid inside the
# regrid, the reflux hooks inside the right-hand side) is listed first.
const PHASES = [
    ("regrid: tagging", (:_tag_sweep!, :tagged_region, :_tag_tiles, :_tag_bounds)),
    ("regrid: coefficients", (:_regrid_prime!,)),
    ("regrid: placement", (:_place_tiles, :_rebalance_due!)),
    ("regrid: fill/migrate", (:_fill_tiles_from_parent!, :_fill_levels_from_parents!,
                              :_fill_tile_from_box!, :_carry_over!, :_migrate_tile!,
                              :_copy_tile!, :_gather_tile)),
    ("regrid: rebuild", (:_replace_level!, :_build_level_patches, :build_level_coupling,
                         :_fill_covered!, :_build_reflux!)),
    ("regrid: rest", (:_maybe_regrid!, :_regrid_impl!)),
    ("reflux", (:_reflux_apply!, :_reflux_gather!, :_reflux_guard!, :_reflux_change!,
                :_reflux_begin_step!, :_reflux_fold!, :_reflux_filter!, :_reflux_open!,
                :_reflux_close!, :_reflux_component!)),
    # The coefficient pass alone: `_sensor_level_rhs!`, which calls it, also
    # holds the tiles' boundary conditions and right-hand sides.
    ("level artificial", (:_level_artificial!,)),
    ("folded box", (:_mirror_folded_box!,)),
    ("Hermite shell fill", (:hermite_level_shell!,)),
    ("box saves", (:save_level_boxes!, :_exchange_boxes!)),
    ("restriction", (:restrict_level!, :_restrict_tiles!, :_exchange_restriction!)),
    ("post-step shell", (:prolong_level_ghosts!,)),
    ("ghost-flux divergence", (:_level_ghost_fluxes!, :_cold_ghost_flux_divergence!,
                               :_ghost_flux_divergence!, :_ghost_flux_solves!,
                               :_coarse_fine_ghost_fluxes!)),
    ("same-level sync", (:_sync_level!, :sync_patches!, :_sync_level_records!)),
    ("max_rate", (:max_rate,)),
    ("validity checks", (:_refreshed_substep_status!, :_substep_validity_status!,
                         :_apply_validity!, :state_report, :_positivity_failsafe!,
                         :_validate_transport_state!, :check_step)),
    ("state filter", (:filter_state!, :_level_filter!)),
    ("RHS", (:compute_rhs!,)),
    ("stage update", (:_rk_update!, :_level_update!)),
    ("boundary conditions", (:apply_bcs!,)),
    ("callbacks", (:run_callbacks!, :_step_callbacks)),
    ("other", ()),
]
const NPHASE = length(PHASES)
const NDEPTH = 4                     # outer, root, level 1, level 2 and deeper

function frame_info(ip)
    return map(Profile.lookup(convert(Ptr{Cvoid}, ip))) do fr
        name = String(fr.func)
        m = match(r"^#+([^#]+)#", name)
        file = String(fr.file)
        mpi = occursin(r"[\\/]MPI[\\/]", file) || startswith(name, "MPI_") ||
              startswith(name, "PMPI_")
        (m === nothing ? fr.func : Symbol(m.captures[1]), mpi)
    end
end

# Samples inside `run!`, as a flat vector: [phase, depth] counts, then the
# same for the samples inside an MPI.jl call.
function phase_counts()
    data = Profile.fetch(include_meta = false)
    stacks = Dict{Vector{UInt64},Int}()
    current = UInt64[]
    for ip in data
        if ip == 0
            isempty(current) || (stacks[current] = get(stacks, current, 0) + 1)
            current = UInt64[]
        else
            push!(current, ip)
        end
    end
    frames = Dict{UInt64,Vector{Tuple{Symbol,Bool}}}()
    counts = zeros(Int, NPHASE, NDEPTH, 2)
    for (stack, count) in stacks
        funcs = Symbol[]
        depth = 0
        mpi = false
        for ip in stack
            for (func, inmpi) in get!(() -> frame_info(ip), frames, ip)
                push!(funcs, func)
                mpi |= inmpi
                func === :_advance_level! && (depth += 1)
            end
        end
        :run! in funcs || continue
        phase = NPHASE
        for (q, (_, names)) in enumerate(PHASES)
            any(in(names), funcs) && (phase = q; break)
        end
        d = min(depth, NDEPTH - 1) + 1
        counts[phase, d, 1] += count
        mpi && (counts[phase, d, 2] += count)
    end
    return vec(counts)
end

# --- Run -------------------------------------------------------------------

function main()
    comm = MPI.COMM_WORLD
    rank, np = MPI.Comm_rank(comm), MPI.Comm_size(comm)
    root = rank == 0
    refined = startswith(opt.config, "amr")
    level = opt.config == "fine2" ? 1 : opt.config == "fine3" ? 2 : 0
    nlev = opt.config == "amr2" ? 2 : opt.config == "amr3" ? 3 : 1
    Lx = opt.Lx > 0 ? opt.Lx : opt.dim == 2 ? 0.25 : 0.18
    n_global = grid(opt.dim, opt.nres, Lx, level)
    amr = refined ?
          AMR(initial = :sensor, tile = opt.tile, max_levels = nlev,
              subcycle = opt.subcycle,
              regrid_interval = opt.regrid > 0 ? opt.regrid : nothing,
              rebalance = opt.rebalance) : nothing
    num = Numerics(n_global = n_global, amr = amr)
    if root
        say("rmicost: %s, %d-D, nres %d (Δx root %.3f mm), n_global %s, np %d, " *
            "threads %d\n", opt.config, opt.dim, opt.nres, 1e3 * lambda / opt.nres,
            n_global, np, Threads.nthreads())
        @printf("  Lx %.3f m; impact %.1f µs, reflected shock out %.1f µs, ", Lx,
                1e6t_impact, 1e6t_reflected)
        @printf("tfinal %.1f µs\n", 1e6opt.tfinal)
        flush(stdout)
    end
    opt.profile && Profile.init(n = 10_000_000, delay = opt.delay)
    MPI.Barrier(comm)
    prob = problem(opt.dim, Lx)
    t_setup = @elapsed solver, Q, dims = setup_on_some_grid(prob, num, np)
    t_setup = MPI.Allreduce(t_setup, max, comm)
    root && @printf("  process grid %s, setup %.1f s\n", dims, t_setup)
    if root && refined
        say("  interface flux %s, level interpolation order %d, " *
                "subcycle %s, regrid interval %d, tiles %s\n",
                solver.interface_flux, getfield(solver, :schemes).level_interpolation_order,
                getfield(solver, :subcycle), getfield(solver, :regrid).interval,
                [length(level_regions(solver, l)) for l in 1:nlevels(solver)-1])
        flush(stdout)
    end
    recorder = StepRecorder(opt.warm, opt.profile, nlevels(solver))
    callbacks = (Callback(recorder, (solver, Q) -> nothing),)
    if opt.progress > 0
        progress = ProgressLog(every = opt.progress, tfinal = opt.tfinal)
        # One report of the initial state, so that its compilation on rank 0
        # is not waited out by the other ranks inside a timed step.
        progress.effect!(solver, Q)
        callbacks = (callbacks..., progress)
    end
    gc0 = gc_counters()
    MPI.Barrier(comm)
    t_run = @elapsed run!(solver, Q; tfinal = opt.tfinal, nmax = opt.nmax,
                          callback = callbacks)
    opt.profile && Profile.stop_timer()
    gc1 = gc_counters()
    report(solver, Q, recorder, comm, (; t_setup, t_run, gc0, gc1, n_global, dims,
                                        refined, nlev))
end

# --- Report ----------------------------------------------------------------

function physics(solver, Q)
    x, X = line_profile(solver, Q, :X; dim = 1, species = 3)
    dx = profile_spacing(solver, 1)
    return (W = sum(4 .* X .* (1 .- X) .* dx), xc = sum((1 .- X) .* dx))
end

function report(solver, Q, r::StepRecorder, comm, run)
    rank, np = MPI.Comm_rank(comm), MPI.Comm_size(comm)
    root = rank == 0
    phys = physics(solver, Q)
    nsteps = length(r.wall)
    warm = min(r.warm, nsteps)
    steady = warm+1:nsteps
    nsteady = length(steady)
    nsteady == 0 && (r.t_warm = solver.t)
    wall_startup = sum(r.wall[1:warm]; init = 0.0)
    wall_steady = sum(r.wall[steady]; init = 0.0)
    wait_steady = solver.wait_total - (nsteady > 0 ? r.wait_warm : 0.0)
    gc_steady = nsteady > 0 ? (run.gc1[1] - r.gc_warm[1], run.gc1[2] - r.gc_warm[2]) :
                (0, UInt64(0))
    nlev = nlevels(solver)
    points_now = level_points(solver)
    # Per rank: steady wall and wait, steady bytes and GC seconds, peak
    # resident set, nodes per level now.
    mine = Float64[wall_steady, wait_steady, gc_steady[1], gc_steady[2] / 1e9,
                   Sys.maxrss(), points_now...]
    gathered = MPI.Gather(mine, comm; root = 0)
    points_dt = MPI.Reduce(r.points_dt, +, comm; root = 0)
    point_steps = MPI.Reduce(r.point_steps, +, comm; root = 0)
    footprint = MPI.Reduce(Base.summarysize(solver) + Base.summarysize(Q), +, comm;
                           root = 0)
    prof = opt.profile ? MPI.Gather(phase_counts(), comm; root = 0) : Int[]
    root || return
    table = reshape(gathered, length(mine), np)

    sim_steady = solver.t - r.t_warm
    ms_step = nsteady > 0 ? 1e3 * wall_steady / nsteady : NaN
    ms_median = nsteady > 0 ? 1e3 * median(r.wall[steady]) : NaN
    per_us = sim_steady > 0 ? wall_steady / (1e6 * sim_steady) : NaN
    finished = solver.t >= opt.tfinal * (1 - 1e-12)
    projected = wall_startup + per_us * 1e6 * (opt.tfinal - r.t_warm)
    projected_steps = nsteady > 0 ? warm + nsteady * (opt.tfinal - r.t_warm) /
                                            sim_steady : NaN
    sub = getfield(solver, :subcycle)
    level_steps = [nsteps; [sum(t -> (length(t) >= l && t[l] > 0 ?
                                      (sub ? 3^l : 1) : 0), r.tiles; init = 0)
                            for l in 1:nlev-1]]
    points_total = round.(Int, vec(sum(table[6:end, :]; dims = 2)))
    points_avg = solver.t > 0 ? points_dt ./ solver.t : points_total
    rss = table[5, :]
    busy = table[1, :] .- table[2, :]
    checks = r.checks
    check_steps = [i for i in 2:nsteps if checks[i] != checks[i-1]]
    changed_steps = findall(r.changed)
    quiet = setdiff(steady, check_steps)
    base = isempty(quiet) ? NaN : median(r.wall[quiet])
    regrid_excess = sum((r.wall[i] - base for i in check_steps if i in steady); init = 0.0)
    nchecks_steady = count(in(steady), check_steps)
    peak = nsteady > 0 ? steady[argmax(r.wall[steady])] : 0

    fields = Pair{String,Any}[
        "config" => opt.config, "dim" => opt.dim, "nres" => opt.nres,
        "n_global" => join(run.n_global, "x"), "np" => np,
        "dims" => join(run.dims, "x"), "subcycle" => sub, "tile" => opt.tile,
        "t_end_us" => round(1e6 * solver.t; digits = 2), "finished" => finished,
        "steps" => nsteps, "level_steps" => join(level_steps, "/"),
        "setup_s" => round(run.t_setup; digits = 2),
        "run_s" => round(run.t_run; digits = 2),
        "startup_steps" => warm, "startup_s" => round(wall_startup; digits = 2),
        "steady_steps" => nsteady, "steady_s" => round(wall_steady; digits = 3),
        "ms_per_step" => round(ms_step; sigdigits = 4),
        "ms_per_step_median" => round(ms_median; sigdigits = 4),
        "s_per_us" => round(per_us; sigdigits = 4),
        "projected_s" => round(projected; sigdigits = 4),
        "projected_steps" => round(projected_steps; digits = 0),
        "points" => join(points_total, "/"),
        "points_avg" => join(round.(Int, points_avg), "/"),
        "point_steps" => join(round.(Int, point_steps), "/"),
        "regrid_checks" => length(check_steps), "regrid_changes" => length(changed_steps),
        "regrid_excess_s" => round(regrid_excess; sigdigits = 3),
        "busy_max_over_mean" => round(maximum(busy) / mean(busy); digits = 3),
        "wait_frac" => round(sum(table[2, :]) / max(sum(table[1, :]), eps()); digits = 3),
        "alloc_MB_per_step" => round(mean(table[3, :]) / 2^20 / max(nsteady, 1);
                                     sigdigits = 3),
        "gc_ms_per_step" => round(1e3 * mean(table[4, :]) / max(nsteady, 1);
                                  sigdigits = 3),
        "maxrss_GB_max" => round(maximum(rss) / 2^30; digits = 3),
        "maxrss_GB_sum" => round(sum(rss) / 2^30; digits = 3),
        "footprint_GB" => round(footprint / 2^30; digits = 3),
        "W_mm" => round(1e3 * phys.W; digits = 4),
        "xc_mm" => round(1e3 * phys.xc; digits = 4),
    ]
    println("rmicost," * join((k * "=" * string(v) for (k, v) in fields), ","))

    @printf("\n%s %d-D, nres %d, n_global %s, %d ranks as %s\n", opt.config, opt.dim,
            opt.nres, run.n_global, np, run.dims)
    @printf("  simulated %.2f µs of %.2f%s\n", 1e6 * solver.t, 1e6 * opt.tfinal,
            finished ? "" : " (stopped by nmax)")
    @printf("  root steps %d; steps per level %s%s\n", nsteps, join(level_steps, ", "),
            sub && run.refined ? " (3^ℓ substeps per root step while ℓ holds tiles)" : "")
    @printf("  setup %.2f s; run %.2f s: first %d steps %.2f s, steady %d steps %.2f s,\n",
            run.t_setup, run.t_run, warm, wall_startup, nsteady, wall_steady)
    say("    outside the steps %.2f s (compilation and priming before the first, " *
        "callbacks)\n", run.t_run - wall_startup - wall_steady)
    @printf("  steady %.2f ms per root step (median %.2f; slowest %.2f at step %d)\n",
            ms_step, ms_median, 1e3 * (peak > 0 ? r.wall[peak] : NaN), peak)
    @printf("  steady %.4f s per µs simulated; full window %.1f s over %d steps%s\n",
            per_us, projected, round(Int, projected_steps),
            finished ? "" : " (projected)")
    for l in 1:nlev
        @printf("  level %d: %d nodes at the end, %.0f time-averaged, %.3g node-steps\n",
                l - 1, points_total[l], points_avg[l], point_steps[l])
    end
    for l in 1:nlev-1
        lev = getfield(solver, :levels)[l+1]
        active = ntuple(d -> run.n_global[d] > 1, 3)
        sizes = [prod(CompactLES.fine_extent(lt.region, active, lt.folded))
                 for lt in lev.transfers]
        counts = [length(t) >= l ? t[l] : 0 for t in r.tiles]
        say("  level %d tiles: %d now (min %d, max %d over the run); nodes per tile " *
                "%s\n", l, length(sizes), isempty(counts) ? 0 : minimum(counts),
                maximum(counts; init = 0),
                isempty(sizes) ? "-" : @sprintf("%d–%d, median %d", minimum(sizes),
                                                maximum(sizes), median(sizes)))
    end
    if run.refined
        say("  regrid checks %d (tile set changed at %d); steady checks %d cost " *
                "%.3f s over the median step (%.1f%% of the steady wall)\n",
                length(check_steps), length(changed_steps), nchecks_steady,
                regrid_excess, 100 * regrid_excess / max(wall_steady, eps()))
    end
    println("  nodes per rank per level (end):")
    for l in 1:nlev
        @printf("    level %d: %s\n", l - 1, join(round.(Int, table[5+l, :]), " "))
    end
    @printf("  steady busy per rank (s): %s\n", join((@sprintf("%.2f", b) for b in busy), " "))
    @printf("  steady wait per rank (s): %s\n",
            join((@sprintf("%.2f", w) for w in table[2, :]), " "))
    @printf("  busy max/mean %.3f; waiting %.1f%% of the steady wall summed over ranks\n",
            maximum(busy) / mean(busy), 100 * sum(table[2, :]) / max(sum(table[1, :]), eps()))
    say("  allocation %.3g MB and GC %.3g ms per steady step (rank mean; max %.3g MB, " *
            "%.3g ms)\n", mean(table[3, :]) / 2^20 / max(nsteady, 1),
            1e3 * mean(table[4, :]) / max(nsteady, 1),
            maximum(table[3, :]) / 2^20 / max(nsteady, 1),
            1e3 * maximum(table[4, :]) / max(nsteady, 1))
    say("  peak resident %.2f GB max per rank, %.2f GB summed; solver and state " *
            "%.2f GB summed\n", maximum(rss) / 2^30, sum(rss) / 2^30, footprint / 2^30)
    @printf("  mix width W %.4f mm, interface position x_c %.4f mm\n", 1e3 * phys.W,
            1e3 * phys.xc)
    opt.profile && print_profile(reshape(prof, NPHASE, NDEPTH, 2, np), table, busy,
                                 nsteady)
    flush(stdout)
end

function print_profile(counts, table, busy, nsteady)
    np = size(counts, 4)
    totals = [sum(counts[:, :, 1, r]) for r in 1:np]
    any(iszero, totals) && println("  (a rank recorded no samples)")
    # ms per root step of each phase on each rank: its share of the rank's
    # samples times the rank's steady wall.
    ms = [totals[r] == 0 ? 0.0 :
          1e3 * sum(counts[q, :, 1, r]) / totals[r] * table[1, r] / max(nsteady, 1)
          for q in 1:NPHASE, r in 1:np]
    busiest = argmax(busy)
    say("\n  phase profile of the steady steps, ms per root step (%d–%d samples " *
            "per rank; busiest rank %d)\n", minimum(totals), maximum(totals), busiest - 1)
    @printf("  %-22s %7s %7s %7s %7s  %5s   %6s %6s %6s %6s\n", "phase", "mean", "max",
            "rank0", "busiest", "MPI%", "outer", "root", "L1", "L2+")
    order = sortperm(vec(sum(ms; dims = 2)); rev = true)
    for q in order
        sum(counts[q, :, 1, :]) == 0 && continue
        all_q = sum(counts[q, :, 1, :])
        depth = [sum(counts[q, d, 1, :]) / all_q for d in 1:NDEPTH]
        say("  %-22s %7.2f %7.2f %7.2f %7.2f  %5.1f   %5.1f%% %5.1f%% %5.1f%% " *
                "%5.1f%%\n", PHASES[q][1], mean(ms[q, :]), maximum(ms[q, :]), ms[q, 1],
                ms[q, busiest], 100 * sum(counts[q, :, 2, :]) / all_q,
                (100 .* depth)...)
    end
    @printf("  %-22s %7.2f %7.2f %7.2f %7.2f  %5.1f\n", "total", mean(sum(ms; dims = 1)),
            maximum(sum(ms; dims = 1)), sum(ms[:, 1]), sum(ms[:, busiest]),
            100 * sum(counts[:, :, 2, :]) / sum(counts[:, :, 1, :]))
    for q in 1:NPHASE
        println("rmicost_phase,", PHASES[q][1], ",",
                join((@sprintf("%.3f", ms[q, r]) for r in 1:np), ","))
    end
end

mpi_main(main)
