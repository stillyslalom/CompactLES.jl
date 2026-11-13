# Time to solution of a refined run against the uniform grid at the refined
# spacing. Each problem runs, in one process and warm (a short untimed run of
# every configuration first), as
#
#   coarse   the root grid alone
#   box      the root grid with one subcycled, regridded box (AMR, tile = 0)
#   tiles    the same with lattice tiles (AMR, tile > 0)
#   fine     the uniform grid at the refined spacing, the reference answer
#
# (`box_global` and `tiles_global` add the unsubcycled forms on request), and
# the script reports per configuration:
#
# - the run wall (`solver.wall_total`, setup excluded and printed beside it),
#   root steps, point-steps (points times steps, summed over levels and
#   substeps) and the wall per point-step, which separates the cost per point
#   from the number of points;
# - the composite error against the fine run: every node of every patch is
#   compared with the fine run's node at the same coordinates, and the L1 norm
#   is the composite quadrature of `volume_integral`, so a root node a child
#   level covers is weighted by the uncovered fraction of its cell
#   (`Patch.covered`) and counted once; Linf runs over nodes with any
#   uncovered fraction; species problems add the mixedness ∫Y(1−Y)dV. A
#   refined run's L1 is also split into the refined level's nodes and the
#   root's uncovered nodes, since a whole-domain norm mixes the error the
#   cover reaches with the root's own;
# - for a refined run, a sampling profile of `run!` binned by phase and by
#   level: the Hermite fill of the shell, the box saves, the folded box at an
#   axis or a plane, the extra right-hand side of the Hermite endpoint, the
#   ghost-flux divergence at coarse-fine faces, restriction, the post-step
#   shell, the same-level synchronization of a tiled level, regridding
#   (tagging, coefficients, fill, the rest), the state filter, the right-hand
#   side, the level-wide artificial coefficients of a tiled level and the rest
#   of the step; a sample counts toward the first phase in that order whose
#   function is on its stack;
# - for a refined run, the step-size account: root steps against the coarse
#   run's, which rate class bounded each root step (root nodes uncovered,
#   partly covered or covered by the child level, a refined level at its
#   3^level-scaled rate, or the nodes of any level `max_rate` holds to
#   `OVERWRITTEN_CFL`, `Patch.overwritten`) and whether the artificial
#   diffusivity or the hyperbolic rate dominated there, the root step count
#   predicted if the covered root nodes were left out of `max_rate`, or kept
#   with their artificial diffusivity left out, and the CFL number the
#   overwritten nodes took against the solver's. The rate is recomputed from
#   the state after each step, the state the next step's `max_rate` sizes from, and is
#   compared with the rate `run!` records for that step; the two differ only
#   across a regrid check, which refreshes the coefficients. The prediction
#   from the refined levels' rate alone, with and without the artificial
#   diffusivity, is the root step count the level's own rate would allow;
# - for a refined run against the fine run, the wall ratio as the product of
#   the ideal (the point-steps had the finest level taken the fine run's steps
#   on its time-averaged points), the measured point-steps over the ideal and
#   the cost per point-step over the fine run's (`attribution`), with the
#   root's share of the points and the padded-over-interior points of the
#   levels; the profile also gives each phase's time per point-step of the
#   whole run and of the level its samples fall on, the uniform runs' alike.
#
# The profile and the step-size account share one more run of every
# configuration, since a profiler perturbs the wall and a per-step callback
# makes `run!` repeat the restriction before the next step. The profile leaves
# out the callback's samples and those of that repeated restriction, and the
# milliseconds per step it prints scale its shares by the timed run's wall.
# The callback's run is checked against the timed run's step count and final
# root state; on a tiled 3-D level the two have differed in the last digits,
# a cause not yet traced.
#
#   julia --project=. -t 1 bench/amrwin.jl quick=true
#   julia --project=. -t 1 bench/amrwin.jl problems=sod
#   julia --project=. -t 1 bench/amrwin.jl problems=blob3d blob3d_n=36
#
# Problems (`problems=`, comma-separated):
#
#   sod     planar Sod tube on a 2-D slip-walled channel, cfl 0.2
#   blobs   three two-gas blobs advected diagonally through a periodic box
#   shock   1-D cylindrical converging shock toward the axis (the imploding
#           shock tutorial's case; the fine grid is 3N − 1 points, which puts
#           its nodes on the refined level's under the half-cell axis offset)
#   blob3d  bench/amr_cost.jl's 3-D heavy blob and its regrid settings, the
#           level's first layout taken from the tags (`blob3d_start=box`
#           gives that script's box), against the uniform 3N³ grid at the
#           refined spacing
#
# `configs=` selects the configurations, `profile=false` and `diagnose=false`
# skip the extra run, `art=false` turns off the artificial properties
# everywhere, `validity=permissive` reports an invalid final state (a mass
# fraction outside the species band, a negative internal energy) instead of
# dropping the configuration, `quick=true` takes small grids and short end
# times for a smoke run. Each problem's grid, tile edge and end time
# are options (`sod_n`, `blob3d_t`, ...), and the blobs' edge width `blobs_w`.
# The full set takes about half an hour at -t 1 on the workstation, the 3-D
# problem a third of it; run one problem per process. The measurements are in
# reference/CALIBRATION_APPENDIX.md under this script's name.
#
# Under `mpiexec -n N` every configuration runs decomposed and rank 0 prints:
#
#   mpiexec -n 8 julia --project=. -t 1 bench/amrwin.jl problems=blobs
#
# The Sod channel lays its ranks along the tube, since its 17 root nodes across
# cannot be split under the filter's nine-node minimum. The fine run's compared
# fields are gathered to rank 0 by `field_snapshot` and broadcast, so each rank
# compares its own nodes and the composite error is the same quadrature at any
# rank count. The wall is the slowest rank's. The waiting (`solver.wait_total`:
# time inside the run-wide collectives and the level record exchanges) as a
# share of a rank's wall, and the busy time (the wall less the waiting), are
# given as their largest and mean over the ranks. Points, point-steps and the
# census are summed over the ranks, and the wall per point-step is core time,
# the wall times the rank count, so that a run scaling perfectly keeps it. The
# step-size account takes each class's largest rate over the ranks. The phase
# profile is rank 0's sampler alone, and its last column is the share of each
# phase's samples inside an MPI call, so the time a collective or an exchange
# waits shows under the phase that holds it. Allocation and GC are rank 0's.
#
# The summary numbers, the phase shares and the step-size account also appear
# on `row,<problem>,<config>,<key>,<value>` lines, so bench/repeat.jl can take
# medians over processes:
#
#   julia bench/repeat.jl runs=5 'pattern=^row,([^,]+),([^,]+),wall,([^,]+)' \
#       -- bench/amrwin.jl problems=sod profile=false diagnose=false

using MPI
MPI.Init(threadlevel=:funneled)
using CompactLES
using Printf
using Profile
const CL = CompactLES
const COMM = MPI.COMM_WORLD
nranks() = MPI.Comm_size(COMM)

const opt = CompactLES.script_args(ARGS, (
    problems = "sod,blobs,shock,blob3d", configs = "coarse,box,tiles,fine",
    quick = false, warm = 12, nmax = 100_000, profile = true, diagnose = true,
    art = true, validity = "strict", delay = 0.002,
    sod_n = 201, sod_ny = 17, sod_tile = 8, sod_t = 0.1,
    blobs_n = 64, blobs_tile = 16, blobs_t = 0.2, blobs_w = 0.01,
    shock_n = 256, shock_tile = 16, shock_t = 0.221,
    blob3d_n = 24, blob3d_tile = 8, blob3d_t = 1.0, blob3d_start = "sensor"))

const wall2 = (SlipWallBC(), SlipWallBC())

# ---------------------------------------------------------------------------
# Problems. Each gives the root and fine grids, the end time, the CFL number,
# the two refined configurations and the fields compared.

function sod_spec(nx, ny, tile, tfinal)
    W = (ny - 1) / (nx - 1)          # square root cells
    problem = Problem(name="sod 2-D", eos=IdealSpecies("gas"; R=1.0, gamma=1.4),
        domain=((0.0, 1.0), (0.0, W), (0.0, 1.0)),
        bcs=(wall2, wall2, PeriodicBC()),
        ic=(x, y, z) -> x < 0.5 ? Prim(u=(0, 0, 0), p=1.0, rho=1.0) :
                                  Prim(u=(0, 0, 0), p=0.1, rho=0.125))
    # The jump sits inside the refined region at t = 0, and a subcycled level
    # takes three substeps on one rate measurement: cfl 0.2 (AMR_GPU.md,
    # startup traps), for every configuration alike.
    box = AMR(initial=:sensor, subcycle=true)
    # Decomposed, every rank along the tube: the root nodes across it are too
    # few to split under the filter's nine-node minimum.
    return (; name="sod", title="planar Sod tube in 2-D", problem,
            n_root=(nx, ny, 1), n_fine=(3nx - 2, 3ny - 2, 1), tfinal, cfl=0.2,
            box, tiles=AMR(box; tile=tile), compare=[(:rho, 1)], mixture=false,
            dims=(nranks(), 1, 1))
end

function blobs_spec(n, tile, tfinal, w)
    eos = IdealMixture([IdealSpecies{Float64}("light", 1.0, 1.4),
                        IdealSpecies{Float64}("heavy", 0.25, 1.09)])
    centers = ((0.25, 0.30), (0.70, 0.45), (0.45, 0.80))
    R, U = 0.08, 0.5
    wrap(d) = d - round(d)
    function ic(x, y, z)
        θ = 0.0
        for (cx, cy) in centers
            r = hypot(wrap(x - cx), wrap(y - cy))
            θ += 0.5 * (1 - tanh((r - R) / w))
        end
        θ = min(θ, 1.0)
        return Prim(Y=(1 - θ, θ), rho=1.0 + 3.0θ, p=1.0, u=(U, U, 0.0))
    end
    problem = Problem(name="three blobs", eos=eos,
        domain=((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)),
        bcs=(PeriodicBC(), PeriodicBC(), PeriodicBC()), ic=ic)
    box = AMR(initial=:sensor, subcycle=true)
    return (; name="blobs", title="three blobs advected through a periodic box",
            problem, n_root=(n, n, 1), n_fine=(3n, 3n, 1), tfinal, cfl=0.5,
            box, tiles=AMR(box; tile=tile), compare=[(:rho, 1), (:Y, 2)],
            mixture=true, dims=nothing)
end

function shock_spec(n, tile, tfinal)
    problem = Problem(name="imploding shock", eos=IdealSpecies("gas"; R=1.0, gamma=1.4),
        metric=CylindricalMetric(),
        domain=((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)),
        bcs=((AxisBC(), SlipWallBC()), PeriodicBC(), PeriodicBC()),
        ic=(r, θ, z) -> begin
            drive = tanh_blend(r, 0.7, 0.012)
            Prim(rho=1.0 + 3.0 * drive, p=0.1 + 19.9 * drive)
        end)
    box = AMR(initial=:sensor, subcycle=true)
    # h = 1/(N − 1/2) with node 1 at h/2, so the grid at h/3 has 3N − 1 nodes.
    return (; name="shock", title="1-D converging shock at the axis", problem,
            n_root=(n, 1, 1), n_fine=(3n - 1, 1, 1), tfinal, cfl=0.5,
            box, tiles=AMR(box; tile=tile), compare=[(:rho, 1)], mixture=false,
            dims=nothing)
end

function blob3d_spec(n, tile, tfinal)
    eos = IdealMixture([IdealSpecies{Float64}("light", 1.0, 1.4),
                        IdealSpecies{Float64}("heavy", 0.25, 1.09)])
    u0 = 0.5
    ic(x, y, z) = begin
        r2 = (x - π / 2)^2 + (y - π / 2)^2 + (z - π / 2)^2
        θ = 0.5 * (1 - tanh((sqrt(r2) - 0.6) / 0.15))
        Prim(Y=(1 - θ, θ), rho=1.0 + 1.5θ, p=2.0, u=(u0, u0, u0))
    end
    problem = Problem(name="3-D blob", eos=eos,
        domain=((0.0, 2π), (0.0, 2π), (0.0, 2π)),
        bcs=(PeriodicBC(), PeriodicBC(), PeriodicBC()), ic=ic)
    # The level's first layout comes from the tags. bench/amr_cost.jl's given
    # box (`blob3d_start=box`) puts a coarse-fine face a few edge widths
    # outside the blob's edge, which the root does not resolve: the parent's
    # nodes beside that face leave the species band before the first regrid
    # check, the check copies the excursion into new tiles, and the
    # mass-fraction bound of D* then sets the level's step for the rest of the
    # run, at about twice the fine run's step count on a 48³ root.
    opt.blob3d_start in ("sensor", "box") ||
        error("blob3d_start = $(opt.blob3d_start), want sensor or box")
    initial = opt.blob3d_start == "box" ?
              BlockRegion((n ÷ 8, n ÷ 8, n ÷ 8), (n ÷ 3, n ÷ 3, n ÷ 3)) : :sensor
    box = AMR(initial=initial,
              subcycle=true, regrid_interval=10, tag_buffer=4, tag_threshold=0.02)
    return (; name="blob3d", title="bench/amr_cost.jl's 3-D blob", problem,
            n_root=(n, n, n), n_fine=(3n, 3n, 3n), tfinal, cfl=0.5,
            box, tiles=AMR(box; tile=tile), compare=[(:rho, 1), (:Y, 2)],
            mixture=true, dims=nothing)
end

function problem_spec(name)
    q = opt.quick
    name == "sod" && return q ? sod_spec(61, 13, 8, 0.02) :
                                sod_spec(opt.sod_n, opt.sod_ny, opt.sod_tile, opt.sod_t)
    # The quick blobs take an edge of one root spacing: at a third of one the
    # box run leaves the species band, which is a finding for the full size,
    # not for a smoke run.
    name == "blobs" && return q ? blobs_spec(32, 8, 0.03, 0.03) :
                                  blobs_spec(opt.blobs_n, opt.blobs_tile, opt.blobs_t,
                                             opt.blobs_w)
    name == "shock" && return q ? shock_spec(96, 16, 0.02) :
                                  shock_spec(opt.shock_n, opt.shock_tile, opt.shock_t)
    name == "blob3d" && return q ? blob3d_spec(18, 6, 0.15) :
                                   blob3d_spec(opt.blob3d_n, opt.blob3d_tile, opt.blob3d_t)
    error("unknown problem '$name', want sod, blobs, shock or blob3d")
end

function numerics(spec, config)
    art = ArtificialProperties(enabled=opt.art)
    control = StepControl(validity=Symbol(opt.validity))
    # One rank takes the default execution, as before the decomposed form.
    extra = nranks() > 1 ? (; execution=Execution(dims=spec.dims)) : (;)
    config == "coarse" && return Numerics(n_global=spec.n_root, cfl=spec.cfl, art=art,
                                          control=control; extra...)
    config == "fine" && return Numerics(n_global=spec.n_fine, cfl=spec.cfl, art=art,
                                        control=control; extra...)
    amr = config == "box" ? spec.box :
          config == "tiles" ? spec.tiles :
          config == "box_global" ? AMR(spec.box; subcycle=false) :
          config == "tiles_global" ? AMR(spec.tiles; subcycle=false) :
          error("unknown config '$config', want coarse, box, tiles, fine, " *
                "box_global or tiles_global")
    return Numerics(n_global=spec.n_root, cfl=spec.cfl, art=art, amr=amr,
                    control=control; extra...)
end

refined(config) = config in ("box", "tiles", "box_global", "tiles_global")

# ---------------------------------------------------------------------------
# Runs

statevec(states) = states isa Vector ? states : [states]
patches_of(solver) = getfield(solver, :patches)

function fresh(spec, config)
    t0 = time_ns()
    solver, states = setup(spec.problem, numerics(spec, config))
    return solver, states, Workspace(states), (time_ns() - t0) / 1e9
end

function timed_run(spec, config; nmax=opt.nmax, callback=nothing, profiled=false)
    GC.gc()
    solver, states, workspace, t_setup = fresh(spec, config)
    run() = run!(solver, states, workspace; tfinal=spec.tfinal, nmax=nmax,
                 callback=callback)
    stats = profiled ? (Profile.@profile @timed run()) : @timed run()
    return (; solver, states, t_setup, stats)
end

# The sum of the root patch's interior state: a fingerprint comparing two runs
# of one configuration.
function root_fingerprint(solver, states)
    ps = CL.PatchSolver(solver, patches_of(solver)[1])
    Q = parent(statevec(states)[1])
    o = ps.decomp.n_halo_d
    n = ps.decomp.n_local
    return MPI.Allreduce(sum(view(Q, o[1]+1:o[1]+n[1], o[2]+1:o[2]+n[2],
                                  o[3]+1:o[3]+n[3], :)), +, COMM)
end

level_factor(solver, level) = getfield(solver, :subcycle) ? 3^level : 1

# The interior points of every patch, summed over the ranks.
global_points(solver) =
    MPI.Allreduce(sum(prod(p.decomp.n_local) for p in patches_of(solver)), +, COMM)

# The padded points (halos included) over the interior points, of the root's
# pieces and of the refined levels' pieces in the final layout, over the ranks.
function padded_ratios(solver)
    acc = zeros(4)
    for p in patches_of(solver)
        i = p.level == 0 ? 1 : 3
        acc[i] += prod(CL.padded_extent(p.decomp))
        acc[i+1] += prod(p.decomp.n_local)
    end
    MPI.Allreduce!(acc, +, COMM)
    return acc[1] / acc[2], acc[4] > 0 ? acc[3] / acc[4] : NaN
end

# The slowest rank's wall, and over the ranks the largest and the mean of the
# waiting share of a rank's wall and of its busy time (wall less waiting).
function wall_stats(solver)
    wall, wait = solver.wall_total, solver.wait_total
    share = wall > 0 ? wait / wall : 0.0
    busy = wall - wait
    np = nranks()
    return (; wall=MPI.Allreduce(wall, max, COMM),
            wait_share_max=MPI.Allreduce(share, max, COMM),
            wait_share_mean=MPI.Allreduce(share, +, COMM) / np,
            busy_max=MPI.Allreduce(busy, max, COMM),
            busy_mean=MPI.Allreduce(busy, +, COMM) / np)
end

# ---------------------------------------------------------------------------
# Composite error against the fine run

function fields_of(solver, states, name, species)
    states isa Vector && return field_array(solver, states, name; species=species)
    return [field_array(solver, states, name; species=species)]
end

# The fine run's node index coinciding with each local node of `ps` along `d`,
# 0 where none does.
function fine_index_map(ps, d, fine)
    n = ps.decomp.n_local[d]
    fine.decomp.active[d] || return ones(Int, n)
    hf = fine.h[d]
    x1 = CL.global_xcoord(fine, d, 1)
    nf = fine.decomp.n_global[d]
    return map(1:n) do i
        r = (CL.xcoord(ps, d, i) - x1) / hf
        g = round(Int, r)
        abs(r - g) < 1e-6 || return 0
        g += 1
        fine.decomp.periodic[d] ? mod1(g, nf) : (1 <= g <= nf ? g : 0)
    end
end

function composite_errors(solver, states, ref, spec)
    patches = patches_of(solver)
    maps = [ntuple(d -> fine_index_map(CL.PatchSolver(solver, p), d, ref.solver), 3)
            for p in patches]
    # Decided over the ranks: a rank may hold the root's piece alone while
    # others hold tiles, and the split below runs collectives.
    several = MPI.Allreduce(length(patches), max, COMM) > 1
    out = Dict{String,Float64}()
    missing_nodes = 0
    for (name, sp) in spec.compare
        fs = fields_of(solver, states, name, sp)
        F = ref.fields[(name, sp)]
        errs = [zeros(size(f)) for f in fs]
        ones_ = [zeros(size(f)) for f in fs]
        # The same sums split by region: `inside` on the refined levels'
        # nodes, `outside` on the root's nodes weighted by their uncovered
        # fraction, so that an error the cover cannot reach is read apart.
        inside = [zeros(size(f)) for f in fs]
        inside_ones = [zeros(size(f)) for f in fs]
        linf = 0.0
        missing_nodes = 0
        for (li, p) in enumerate(patches)
            ps = CL.PatchSolver(solver, p)
            o = ps.decomp.n_halo_d
            n = ps.decomp.n_local
            m1, m2, m3 = maps[li]
            covered = ps.covered
            for k in 1:n[3], j in 1:n[2], i in 1:n[1]
                g1, g2, g3 = m1[i], m2[j], m3[k]
                if g1 == 0 || g2 == 0 || g3 == 0
                    missing_nodes += 1
                    continue
                end
                I = CartesianIndex(i + o[1], j + o[2], k + o[3])
                e = abs(fs[li][I] - F[g1, g2, g3])
                errs[li][I] = e
                ones_[li][I] = 1.0
                if p.level > 0
                    inside[li][I] = e
                    inside_ones[li][I] = 1.0
                end
                covered[I] == 0xff || (linf = max(linf, e))
            end
        end
        label = name === :Y ? "Y$sp" : String(name)
        total, volume = volume_integral(solver, errs), volume_integral(solver, ones_)
        out["L1_$label"] = total / volume
        out["Linf_$label"] = MPI.Allreduce(linf, max, COMM)
        missing_nodes = MPI.Allreduce(missing_nodes, +, COMM)
        if several
            e_in, v_in = volume_integral(solver, inside), volume_integral(solver, inside_ones)
            out["L1in_$label"] = e_in / v_in
            out["L1out_$label"] = (total - e_in) / (volume - v_in)
            out["cover_volume"] = v_in / volume
        end
    end
    if spec.mixture
        Ys = fields_of(solver, states, :Y, 2)
        mixed = volume_integral(solver, [Y .* (1 .- Y) for Y in Ys])
        out["mixedness_rel"] = mixed / ref.mixedness - 1
    end
    out["noncoincident"] = missing_nodes
    return out
end

# The fine run's field over its whole grid without halos, on every rank: the
# interior of the one rank's array, or decomposed, gathered to rank 0 by
# `field_snapshot` and broadcast.
function global_field(run, name, sp)
    s = run.solver
    if nranks() == 1
        f = fields_of(s, run.states, name, sp)[1]
        o = s.decomp.n_halo_d
        n = s.decomp.n_local
        return f[o[1]+1:o[1]+n[1], o[2]+1:o[2]+n[2], o[3]+1:o[3]+n[3]]
    end
    snap = field_snapshot(s, statevec(run.states)[1]; fields=[name])
    F = zeros(s.decomp.n_global)
    if snap !== nothing
        A = snap[name]
        F .= ndims(A) == 4 ? view(A, :, :, :, sp) : A
    end
    return MPI.Bcast!(F, COMM; root=0)
end

function fine_reference(run, spec)
    fields = Dict{Tuple{Symbol,Int},Array{Float64,3}}()
    for (name, sp) in spec.compare
        fields[(name, sp)] = global_field(run, name, sp)
    end
    mixedness = NaN
    if spec.mixture
        Y = fields_of(run.solver, run.states, :Y, 2)[1]
        mixedness = volume_integral(run.solver, [Y .* (1 .- Y)])
    end
    return (; solver=run.solver, fields, mixedness)
end

# ---------------------------------------------------------------------------
# Step-size account and census, from a callback after every step

const CLASSES = ("root uncovered", "root partly covered", "root covered",
                 "level 1", "level 2", "level 3", "overwritten")
# The classes whose rate enters `max_rate` unscaled; the last holds the nodes
# of any level it holds to `OVERWRITTEN_CFL` (`Patch.overwritten`), and "root
# covered" the covered root nodes outside it.
const COUNTED = 1:6

mutable struct ClassMax
    rate::Float64          # largest rate in the class (scaled for a level)
    rate_noart::Float64    # largest rate with the artificial diffusivity left out
    art_at::Float64        # artificial share of the rate at the largest point
end
ClassMax() = ClassMax(0.0, 0.0, 0.0)

function record!(cm::ClassMax, r, r0, scale)
    r /= scale
    r0 /= scale
    if r > cm.rate
        cm.rate = r
        cm.art_at = r > 0 ? (r - r0) / r : 0.0
    end
    cm.rate_noart = max(cm.rate_noart, r0)
    return cm
end

# The rate `_local_max_rate_loop` forms at every interior point of one patch,
# with and without the artificial coefficients, accumulated by class.
function patch_rates!(classes, ps, Q, scale, zero_cache)
    CL.refresh_primitives!(ps, Q)
    decomp = ps.decomp
    o = decomp.n_halo_d
    n = decomp.n_local
    act = decomp.active
    nsp = ps.equations.n_species
    Z = get!(() -> zeros(size(ps.mu_art)), zero_cache, size(ps.mu_art))
    Dz = [Z for _ in 1:nsp]
    sharp = CL._sharpening_constants(ps)
    level = ps.patch.level
    covered = ps.covered
    overwritten = ps.overwritten
    for k in 1:n[3], j in 1:n[2], i in 1:n[1]
        I = CartesianIndex(i + o[1], j + o[2], k + o[3])
        ρ = ps.rho[I]
        c = ps.c[I]
        cp = ps.cp_mix[I]
        uv = (ps.u[I], ps.v[I], ps.w[I])
        acc = 0.0
        dsum = 0.0
        for d in 1:3
            act[d] || continue
            idx = ps.inv_h[d][I] / ps.h[d]
            acc += abs(uv[d]) * idx
            dsum += idx * idx
        end
        acc += c * sqrt(dsum)
        acc += CL.curvature_rate(ps, ps.metric, I, uv)
        molecular = CL.transport_at(ps.transport, ps.eos, ps.T_ion, ps.rho, ps.cp_mix,
                                    ps.field_tuples.Y, I)
        ν = CL._diffusive_rate(ps.eos, ρ, ps.p[I], ps.T_ion[I], cp, molecular,
                               ps.mu_art, ps.beta_art, ps.kappa_art, ps.D_art, I, nsp)
        ν0 = CL._diffusive_rate(ps.eos, ρ, ps.p[I], ps.T_ion[I], cp, molecular,
                                Z, Z, Z, Dz, I, nsp)
        acc += CL._sharpening_rate(sharp, c, ps.inv_h, ps.h, act, I)
        cls = !isempty(overwritten) && overwritten[I] != 0 ? 7 :
              level > 0 ? 3 + min(level, 3) :
              covered[I] == 0x00 ? 1 : covered[I] == 0xff ? 3 : 2
        record!(classes[cls], acc + 2 * ν * dsum, acc + 2 * ν0 * dsum, scale)
    end
    return classes
end

# The callback accumulates each rank's own pieces, with no collective, and
# `reduce_census!` combines the ranks once after the run.
mutable struct Census
    level_steps::Vector{Int}         # steps taken by each level, substeps counted
    level_points::Vector{Float64}    # point-steps of each level
    level_size::Vector{Float64}      # points of each level, summed over root steps
    level_padded::Vector{Float64}    # the same with the halos, summed over root steps
    level_present::Vector{NTuple{4,Int}} # per root step, each level's steps, 0 absent
    tiles::Vector{Int}               # level-1 patches after each root step
    steps::Vector{Vector{ClassMax}}  # per root step, the class maxima after it
    rate_next::Vector{Float64}       # the rate that sized each step, from run!
    regrid_checks::Vector{Int}
    zero_cache::Dict{Any,Array{Float64,3}}
end
Census() = Census(zeros(Int, 4), zeros(4), zeros(4), zeros(4), NTuple{4,Int}[], Int[],
                  Vector{ClassMax}[], Float64[], Int[], Dict{Any,Array{Float64,3}}())

function census_callback(census::Census, rates::Bool)
    return function (solver, states)
        push!(census.rate_next, solver.rate_prev)
        spec = getfield(solver, :regrid)
        push!(census.regrid_checks, spec === nothing ? 0 : spec.checks)
        present = falses(4)
        classes = [ClassMax() for _ in CLASSES]
        ntiles = 0
        for (li, p) in enumerate(patches_of(solver))
            ℓ = min(p.level, 3)
            npts = prod(p.decomp.n_local)
            f = level_factor(solver, p.level)
            census.level_points[ℓ+1] += npts * f
            census.level_size[ℓ+1] += npts
            census.level_padded[ℓ+1] += prod(CL.padded_extent(p.decomp))
            present[ℓ+1] = true
            # A tile decomposed over several ranks is counted by the rank
            # holding its first node.
            p.level == 1 && all(==(0), p.decomp.offset) && (ntiles += 1)
            rates && patch_rates!(classes, CL.PatchSolver(solver, p),
                                  statevec(states)[li], f, census.zero_cache)
        end
        push!(census.level_present,
              ntuple(ℓ -> present[ℓ] ? level_factor(solver, ℓ - 1) : 0, 4))
        push!(census.tiles, ntiles)
        rates && push!(census.steps, classes)
        return nothing
    end
end

# Every rank calls this once after the census run: sums over the ranks, a
# level present on any rank, and per step and class the largest rate over the
# ranks with the artificial share at the rank holding it.
function reduce_census!(c::Census)
    for v in (c.level_points, c.level_size, c.level_padded)
        MPI.Allreduce!(v, +, COMM)
    end
    MPI.Allreduce!(c.tiles, +, COMM)
    n = length(c.level_present)
    present = [c.level_present[i][ℓ] for ℓ in 1:4, i in 1:n]
    n > 0 && MPI.Allreduce!(present, max, COMM)
    c.level_steps .= vec(sum(present; dims=2))
    nranks() == 1 && return c
    nsteps = length(c.steps)
    nsteps == 0 && return c
    nc = length(CLASSES)
    packed = [getfield(c.steps[i][q], f) for f in (:rate, :rate_noart, :art_at),
              q in 1:nc, i in 1:nsteps]
    gathered = reshape(MPI.Allgather(vec(packed), COMM), 3, nc, nsteps, nranks())
    for i in 1:nsteps, q in 1:nc
        r = argmax(view(gathered, 1, q, i, :))
        cm = c.steps[i][q]
        cm.rate = gathered[1, q, i, r]
        cm.art_at = gathered[3, q, i, r]
        cm.rate_noart = maximum(view(gathered, 2, q, i, :))
    end
    return c
end

# ---------------------------------------------------------------------------
# Phase profile

const PHASES = [
    ("regrid: tagging", (:_tag_sweep!, :tagged_region, :_tag_tiles, :_tag_bounds)),
    ("regrid: coefficients", (:_regrid_prime!,)),
    ("regrid: fill", (:_fill_tiles_from_parent!, :_fill_levels_from_parents!,
                      :_fill_tile_from_box!, :_carry_over!, :_migrate_tile!,
                      :_copy_tile!, :_gather_tile)),
    ("regrid: rest", (:_maybe_regrid!,)),
    ("Hermite endpoint RHS", ()),
    ("folded box", (:_mirror_folded_box!,)),
    ("Hermite shell fill", (:hermite_level_shell!,)),
    ("box saves", (:save_level_boxes!, :_exchange_boxes!)),
    ("restriction", (:restrict_level!, :_restrict_tiles!, :_exchange_restriction!)),
    ("post-step shell", (:prolong_level_ghosts!,)),
    ("ghost-flux divergence", (:_level_ghost_fluxes!, :_cold_ghost_flux_divergence!,
                               :_ghost_flux_divergence!, :_ghost_flux_solves!,
                               :_coarse_fine_ghost_fluxes!)),
    ("same-level sync", (:_sync_level!, :sync_patches!)),
    ("max_rate", (:max_rate,)),
    ("substep checks", (:_refreshed_substep_status!, :_substep_validity_status!)),
    ("state filter", (:filter_state!, :_level_filter!)),
    ("RHS", (:compute_rhs!,)),
    ("level sensors", (:_sensor_level_rhs!,)),
    ("stage update", (:_rk_update!, :_level_update!)),
    ("boundary conditions", (:apply_bcs!,)),
    ("reflux apply", (:_reflux_apply!,)),
    ("state validation", (:validate_state!, :state_report, :_reduce_state_report)),
    ("positivity floors", (:positivity_floors,)),
    ("other", ()),
]
const ENDPOINT = 5
const OTHER = length(PHASES)

# The line of `_advance_level!`'s endpoint right-hand side in the loaded
# source: the one `_level_rhs!` call there with both flags false. A stack
# whose innermost `_advance_level!` frame stands on it is that evaluation.
function endpoint_lines()
    file = joinpath(dirname(pathof(CompactLES)), "timestep.jl")
    isfile(file) || return Int[]
    lines = readlines(file)
    start = findfirst(l -> occursin("function _advance_level!(", l), lines)
    start === nothing && return Int[]
    hits = Int[]
    for i in start+1:length(lines)
        startswith(lines[i], "end") && break
        occursin("_level_rhs!(solver, lev, states, dQs, false, false", lines[i]) &&
            push!(hits, i)
    end
    return hits
end

# A frame's function name without the decoration of a keyword body or a
# closure (`#run!#12` is `run!`), with its line.
function frame_names(ip)
    src = dirname(pathof(CompactLES))
    return map(Profile.lookup(convert(Ptr{Cvoid}, ip))) do fr
        name = String(fr.func)
        m = match(r"^#+([^#]+)#", name)
        (m === nothing ? fr.func : Symbol(m.captures[1]), fr.line,
         startswith(String(fr.file), src))
    end
end

# Whether an instruction pointer stands in an MPI call: a frame of MPI.jl's
# source, or a library function named MPI_ or PMPI_.
function in_mpi(ip)
    return any(Profile.lookup(convert(Ptr{Cvoid}, ip))) do fr
        occursin(r"[\\/]MPI[\\/]", String(fr.file)) ||
            occursin(r"^P?MPI[DI]?_", String(fr.func))
    end
end

# Samples of the main thread inside `run!`, binned by phase and by the depth of
# `_advance_level!` (0 outside it, 1 at the root, ℓ + 1 on level ℓ). Samples in
# a callback, and in the restriction `run!` repeats before a step because a
# callback ran, are counted apart and left out of the shares. The lines of
# `_advance_level!` under the right-hand-side phases are tallied, so that the
# endpoint's attribution can be checked against the source.
function phase_profile()
    data = Profile.fetch(include_meta=false)
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
    frames = Dict{UInt64,Vector{Tuple{Symbol,Int,Bool}}}()
    mpi_frames = Dict{UInt64,Bool}()
    endpoint = endpoint_lines()
    counts = zeros(Int, length(PHASES), 5)
    mpi_counts = zeros(Int, length(PHASES))
    total = 0
    excluded = 0
    rhs_lines = Dict{Int,Int}()
    other_funcs = Dict{Symbol,Int}()
    rhs_phases = (ENDPOINT, findfirst(p -> p[1] == "RHS", PHASES),
                  findfirst(p -> p[1] == "level sensors", PHASES),
                  findfirst(p -> p[1] == "ghost-flux divergence", PHASES))
    for (stack, count) in stacks
        funcs = Symbol[]
        depth = 0
        innermost_line = 0
        innermost_own = Symbol("")
        for ip in stack
            for (func, line, own) in get!(() -> frame_names(ip), frames, ip)
                push!(funcs, func)
                own && innermost_own === Symbol("") && (innermost_own = func)
                if func === :_advance_level!
                    depth == 0 && (innermost_line = line)
                    depth += 1
                end
            end
        end
        :run! in funcs || continue
        if :_step_callbacks in funcs || :_start_callbacks in funcs ||
           (:_presync! in funcs && :restrict_level! in funcs)
            excluded += count
            continue
        end
        phase = OTHER
        for (q, (label, names)) in enumerate(PHASES)
            if q == ENDPOINT
                depth > 0 && innermost_line in endpoint && (phase = q; break)
            elseif any(in(names), funcs)
                phase = q
                break
            end
        end
        counts[phase, min(depth, 4)+1] += count
        phase == OTHER &&
            (other_funcs[innermost_own] = get(other_funcs, innermost_own, 0) + count)
        any(ip -> get!(() -> in_mpi(ip), mpi_frames, ip), stack) &&
            (mpi_counts[phase] += count)
        total += count
        depth > 0 && phase in rhs_phases &&
            (rhs_lines[innermost_line] = get(rhs_lines, innermost_line, 0) + count)
    end
    return (; counts, mpi_counts, total, excluded, rhs_lines, other_funcs, endpoint,
            endpoint_found=!isempty(endpoint))
end

# ---------------------------------------------------------------------------
# Reports

row(problem, config, key, value) = println("row,", problem, ",", config, ",", key, ",",
                                           value)

# `points` holds the point-steps a column's time is divided by: every level's
# (outside the level advance), the root's, then each refined level's.
function print_profile(spec, config, prof, wall, steps, points)
    np = nranks()
    @printf("\nphase profile, %s %s%s: %d samples of run!, timed wall %.2f s, %d root steps\n",
            spec.name, config, np > 1 ? " (rank 0's sampler, $np ranks)" : "",
            prof.total, wall, steps)
    @printf("  (%d samples in the callback and the repeated restriction left out)\n",
            prof.excluded)
    prof.endpoint_found ||
        println("  (endpoint line not found in the loaded timestep.jl; the endpoint " *
                "RHS is counted under RHS)")
    prof.total == 0 && return
    @printf("  %-24s %7s %9s   %7s %7s %7s %7s %7s %7s\n", "phase", "share", "ms/step",
            "outside", "root", "L1", "L2", "L3", "in MPI")
    for (q, (label, _)) in enumerate(PHASES)
        c = sum(prof.counts[q, :])
        c == 0 && continue
        share = c / prof.total
        @printf("  %-24s %6.1f%% %9.3f  ", label, 100share, 1e3 * share * wall / steps)
        for col in 1:5
            @printf(" %6.1f%%", 100 * prof.counts[q, col] / prof.total)
        end
        @printf(" %6.1f%%\n", 100 * prof.mpi_counts[q] / prof.total)
        row(spec.name, config, "phase:" * label, round(share; sigdigits=4))
        np > 1 && row(spec.name, config, "phase_mpi:" * label,
                      round(prof.mpi_counts[q] / prof.total; sigdigits=4))
    end
    # Core time per point-step: a phase's time times the rank count, over the
    # point-steps of the whole run (all), or of the level its column counts.
    ns(c, ps) = ps > 0 ? 1e9 * np * c / prof.total * wall / ps : NaN
    @printf("  core-ns per point-step:  %9s %9s %9s %9s %9s %9s\n", "all", "outside",
            "per root", "per L1", "per L2", "per L3")
    for (q, (label, _)) in enumerate(PHASES)
        c = sum(prof.counts[q, :])
        c == 0 && continue
        @printf("  %-24s %9.1f", label, ns(c, points[1]))
        for col in 1:5
            @printf(" %9.1f", prof.counts[q, col] == 0 ? 0.0 :
                              ns(prof.counts[q, col], points[col]))
        end
        println()
        row(spec.name, config, "phase_ns:" * label, round(ns(c, points[1]); sigdigits=4))
        for (col, key) in ((2, "phase_ns_root:"), (3, "phase_ns_L1:"))
            prof.counts[q, col] > 0 &&
                row(spec.name, config, key * label,
                    round(ns(prof.counts[q, col], points[col]); sigdigits=4))
        end
    end
    if !isempty(prof.other_funcs)
        println("  other, by the innermost package function on the stack:")
        for (func, c) in first(sort(collect(prof.other_funcs), by=x -> -x[2]), 6)
            @printf("    %6.1f%%  %s\n", 100c / prof.total,
                    func === Symbol("") ? "(none)" : func)
        end
    end
    isempty(prof.rhs_lines) && return
    @printf("  right-hand-side samples by line of _advance_level! (endpoint at %s):\n",
            join(prof.endpoint, ", "))
    file = joinpath(dirname(pathof(CompactLES)), "timestep.jl")
    source = isfile(file) ? readlines(file) : String[]
    for (line, c) in first(sort(collect(prof.rhs_lines), by=x -> -x[2]), 6)
        text = 1 <= line <= length(source) ? strip(source[line]) : ""
        @printf("    %5d %6.1f%%  %s\n", line, 100c / prof.total, first(text, 60))
    end
end

function print_census(spec, config, census, steps, coarse_steps)
    nfine = prod(spec.n_fine)
    @printf("\n%s %s: steps per level", spec.name, config)
    for ℓ in 0:3
        census.level_steps[ℓ+1] > 0 && @printf("  L%d %d", ℓ, census.level_steps[ℓ+1])
    end
    n = max(length(census.tiles), 1)
    mean_l1 = census.level_size[2] / n
    @printf("\n  level 1: mean %.0f points (%.3f of the fine grid), %.1f patches (max %d)\n",
            mean_l1, mean_l1 / nfine, sum(census.tiles) / n,
            isempty(census.tiles) ? 0 : maximum(census.tiles))
    row(spec.name, config, "cover", round(mean_l1 / nfine; sigdigits=4))
    for ℓ in 0:3
        census.level_steps[ℓ+1] > 0 &&
            row(spec.name, config, "level_steps:L$ℓ", census.level_steps[ℓ+1])
    end
    @printf("  root: mean %.0f points (%.3f of the fine grid), %.3f of the point-steps\n",
            census.level_size[1] / n, census.level_size[1] / n / nfine,
            census.level_points[1] / sum(census.level_points))
    @printf("  padded over interior points, time-averaged: root %.3f, level 1 %.3f\n",
            census.level_padded[1] / census.level_size[1],
            census.level_padded[2] / census.level_size[2])
    row(spec.name, config, "root_cover", round(census.level_size[1] / n / nfine;
                                               sigdigits=4))
    row(spec.name, config, "root_share", round(census.level_points[1] /
                                               sum(census.level_points); sigdigits=4))
    row(spec.name, config, "padded_ratio_L1",
        round(census.level_padded[2] / census.level_size[2]; sigdigits=4))
    isempty(census.steps) && return
    # The step-size account. Step i's rate after it sizes step i + 1, which
    # `run!` records as rate_prev at the next callback.
    nsteps = length(census.steps)
    bind = zeros(Int, length(CLASSES))
    bind_art = zeros(Int, length(CLASSES))
    pred = Dict("covered root nodes left out" => 0.0,
                "partly covered ones too" => 0.0,
                "covered root nodes without artificial diffusivity" => 0.0,
                "no artificial diffusivity anywhere" => 0.0,
                "the refined levels' rate alone" => 0.0,
                "the refined levels' rate without artificial diffusivity" => 0.0)
    mismatch = Float64[]
    held = Float64[]
    weight = CL._overwritten_weight(spec.cfl)
    for (i, cl) in enumerate(census.steps)
        rates = [[c.rate for c in cl[COUNTED]]; weight * cl[7].rate]
        R = maximum(rates)
        R > 0 || continue
        b = argmax(rates)
        cl[7].rate > 0 && push!(held, cl[7].rate / R)
        bind[b] += 1
        cl[b].art_at > 0.5 && (bind_art[b] += 1)
        unc, part = cl[1].rate, cl[2].rate
        levels = max(cl[4].rate, cl[5].rate, cl[6].rate)
        pred["covered root nodes left out"] += max(unc, part, levels) / R
        pred["partly covered ones too"] += max(unc, levels) / R
        pred["covered root nodes without artificial diffusivity"] +=
            max(unc, part, cl[3].rate_noart, levels) / R
        pred["no artificial diffusivity anywhere"] +=
            maximum(c.rate_noart for c in cl[COUNTED]) / R
        pred["the refined levels' rate alone"] += levels / R
        pred["the refined levels' rate without artificial diffusivity"] +=
            max(cl[4].rate_noart, cl[5].rate_noart, cl[6].rate_noart) / R
        if i < nsteps && census.regrid_checks[i+1] == census.regrid_checks[i]
            push!(mismatch, abs(census.rate_next[i+1] / R - 1))
        end
    end
    @printf("  root steps %d, root-only run %d: ratio %.3f\n", steps, coarse_steps,
            steps / max(coarse_steps, 1))
    row(spec.name, config, "root_steps_over_coarse", round(steps / max(coarse_steps, 1);
                                                          sigdigits=4))
    println("  rate class bounding the next root step (share of steps; artificial " *
            "diffusivity above half the rate there):")
    for (q, label) in enumerate(CLASSES)
        bind[q] == 0 && continue
        @printf("    %-22s %6.1f%%   (artificial %5.1f%% of these)\n", label,
                100bind[q] / nsteps, 100bind_art[q] / bind[q])
    end
    println("  root steps predicted from the recomputed rates (sum of rate ratios):")
    for (label, s) in sort(collect(pred), by=x -> -x[2])
        @printf("    %-50s %8.1f  (%.3f of root-only)\n", label, s,
                s / max(coarse_steps, 1))
        row(spec.name, config, "predicted_steps:" * label, round(s; digits=1))
    end
    if !isempty(held)
        sort!(held)
        @printf("  overwritten nodes, CFL number over the solver's (at most %.2f): %s\n",
                1 / weight, @sprintf("median %.2f, max %.2f",
                                     held[(length(held) + 1) ÷ 2], held[end]))
        row(spec.name, config, "overwritten_cfl_ratio_max", round(held[end]; sigdigits=4))
    end
    if !isempty(mismatch)
        sort!(mismatch)
        println("  instrument check, recomputed rate against the rate that sized the " *
                "next step:")
        @printf("    %d pairs without a regrid check between: median %.1e, max %.1e\n",
                length(mismatch), mismatch[(length(mismatch) + 1) ÷ 2], mismatch[end])
    end
end

# ---------------------------------------------------------------------------

function run_problem(spec, configs)
    @printf("\n=== %s: %s, root %s, fine %s, t = %g, cfl %g, art %s ===\n",
            spec.name, spec.title, join(spec.n_root, "x"), join(spec.n_fine, "x"),
            spec.tfinal, spec.cfl, opt.art)
    failed = Dict{String,String}()
    attempt(f, config, what) = try
        f()
    catch err
        msg = first(sprint(showerror, err), 400)
        failed[config] = "$what: $msg"
        @printf("  %s %s failed: %s\n", config, what, msg)
        nothing
    end
    # Warm every configuration first, so each timed run below reuses its code.
    for config in configs
        t = @elapsed attempt(config, "warm-up") do
            timed_run(spec, config; nmax=opt.warm)
        end
        haskey(failed, config) || @printf("  warm-up %-12s %6.1f s\n", config, t)
    end
    live = filter(c -> !haskey(failed, c), configs)
    # The fine reference first, so each later run is compared and dropped.
    order = sort(live, by=c -> c == "fine" ? 0 : c == "coarse" ? 1 : 2)
    results = Dict{String,Any}()
    ref = nothing
    for config in order
        r = attempt(config, "timed run") do
            timed_run(spec, config)
        end
        r === nothing && continue
        s = r.solver
        reached = isapprox(s.t, spec.tfinal; rtol=1e-12)
        reached || @printf("  %s stopped at t = %g after %d steps (nmax)\n", config,
                           s.t, s.step)
        npoints = global_points(s)
        ws = wall_stats(s)
        res = Dict{String,Any}("wall" => ws.wall,
                               "setup" => MPI.Allreduce(r.t_setup, max, COMM),
                               "steps" => s.step, "points" => npoints,
                               "alloc" => r.stats.bytes, "gc" => r.stats.gctime,
                               "elapsed" => r.stats.time, "reached" => reached,
                               "fingerprint" => root_fingerprint(s, r.states))
        if nranks() > 1
            res["wait_share_max"] = ws.wait_share_max
            res["wait_share_mean"] = ws.wait_share_mean
            res["busy_max"] = ws.busy_max
            res["busy_mean"] = ws.busy_mean
        end
        padded_root, padded_level = padded_ratios(s)
        res["padded_end_root"] = padded_root
        isfinite(padded_level) && (res["padded_end_L1"] = padded_level)
        if config == "fine"
            ref = fine_reference(r, spec)
            for d in 1:3
                ref.solver.decomp.active[d] || continue
                h_root = spec.n_root[d] > 1 ? numerics_h(spec, d) : 1.0
                isapprox(ref.solver.h[d] * 3, h_root; rtol=1e-12) ||
                    @printf("  warning: fine spacing %g along %d, root spacing %g\n",
                            ref.solver.h[d], d, h_root)
            end
        elseif ref !== nothing
            merge!(res, composite_errors(s, r.states, ref, spec))
        end
        results[config] = res
    end
    coarse_steps = haskey(results, "coarse") ? results["coarse"]["steps"] : 0
    # The census, the step-size account and the profile of each configuration,
    # in one more run; the uniform runs' own bound and phases are the
    # reference the refined ones are read against. Without `diagnose`, the
    # profiled run carries a census without the rates, for its point-steps.
    censuses = Dict{String,Census}()
    profiles = Dict{String,Any}()
    opt.profile && Profile.init(n=10_000_000, delay=opt.delay)
    for config in filter(c -> haskey(results, c), order)
        (opt.diagnose || opt.profile) || break
        census = Census()
        opt.profile && Profile.clear()
        r = attempt(config, opt.diagnose ? "census run" : "profiled run") do
            timed_run(spec, config; callback=census_callback(census, opt.diagnose),
                      profiled=opt.profile)
        end
        r === nothing && continue
        # Rank 0's samples alone.
        opt.profile && (profiles[config] = MPI.Comm_rank(COMM) == 0 ? phase_profile() :
                                                                     nothing)
        reduce_census!(census)
        censuses[config] = census
        res = results[config]
        sum_census = root_fingerprint(r.solver, r.states)
        same = r.solver.step == res["steps"] && sum_census == res["fingerprint"]
        same || @printf("  %s census run: steps %d / %d, root sum differs by %.1e relative\n",
                        config, r.solver.step, res["steps"],
                        abs(sum_census / res["fingerprint"] - 1))
        refined(config) && (res["point_steps"] = sum(census.level_points))
    end
    for config in ("coarse", "fine")
        haskey(results, config) || continue
        res = results[config]
        res["point_steps"] = Float64(res["points"] * res["steps"])
    end
    # Summary table.
    println()
    keys_err = String[]
    for (name, sp) in spec.compare
        label = name === :Y ? "Y$sp" : String(name)
        push!(keys_err, "L1_$label", "Linf_$label")
    end
    spec.mixture && push!(keys_err, "mixedness_rel")
    fine_wall = haskey(results, "fine") ? results["fine"]["wall"] : NaN
    np = nranks()
    # Core time per point-step, the wall times the rank count.
    per_point(res) = 1e9 * np * res["wall"] / get(res, "point_steps", NaN)
    fine_per_point = haskey(results, "fine") ? per_point(results["fine"]) : NaN
    @printf("%-13s %7s %8s %7s %6s %11s %8s %7s %8s %5s", "config", "setup s", "wall s",
            "/fine", "steps", "point-steps", "ns/pt-st", "/fine", "alloc GB", "gc%")
    for k in keys_err
        @printf(" %13s", k)
    end
    println()
    for config in order
        haskey(results, config) || continue
        res = results[config]
        ps = get(res, "point_steps", NaN)
        @printf("%-13s %7.2f %8.2f %7.3f %6d %11.4g %8.1f %7.3f %8.2f %5.1f", config,
                res["setup"], res["wall"], res["wall"] / fine_wall, res["steps"], ps,
                per_point(res), per_point(res) / fine_per_point, res["alloc"] / 1e9,
                100 * res["gc"] / res["elapsed"])
        for k in keys_err
            @printf(" %13.4e", get(res, k, NaN))
        end
        println()
        if np > 1
            @printf("  %d ranks: waiting %.1f%% of the wall at most, %.1f%% on average;",
                    np, 100res["wait_share_max"], 100res["wait_share_mean"])
            @printf(" busy %.2f s at most, %.2f s on average (%.3f)\n", res["busy_max"],
                    res["busy_mean"], res["busy_max"] / res["busy_mean"])
        end
        for (name, sp) in spec.compare
            label = name === :Y ? "Y$sp" : String(name)
            haskey(res, "L1in_$label") || continue
            @printf("  %s: L1 %.4e on the level (%.3f of the volume), %.4e on the root\n",
                    label, res["L1in_$label"], res["cover_volume"], res["L1out_$label"])
        end
        get(res, "noncoincident", 0) > 0 &&
            @printf("  (%d nodes of %s have no coincident fine node and are left out)\n",
                    Int(res["noncoincident"]), config)
        for (k, v) in res
            k in ("fingerprint", "reached") && continue
            row(spec.name, config, k, v)
        end
    end
    for config in order
        haskey(censuses, config) || continue
        refined(config) || continue
        print_census(spec, config, censuses[config], results[config]["steps"],
                     coarse_steps)
    end
    for config in ("coarse", "fine")
        (haskey(censuses, config) && !isempty(censuses[config].steps)) || continue
        c = censuses[config]
        art = count(cl -> cl[1].art_at > 0.5, c.steps)
        @printf("\n%s %s: bounding rate over half artificial on %.1f%% of steps\n",
                spec.name, config, 100art / length(c.steps))
    end
    for config in order
        (haskey(profiles, config) && profiles[config] !== nothing) || continue
        res = results[config]
        total = get(res, "point_steps", NaN)
        lp = haskey(censuses, config) && refined(config) ? censuses[config].level_points :
                                                           [total, 0.0, 0.0, 0.0]
        print_profile(spec, config, profiles[config], res["wall"], res["steps"],
                      (total, lp...))
    end
    println()
    for config in order
        (refined(config) && haskey(results, config) && isfinite(fine_wall)) || continue
        res, fine = results[config], results["fine"]
        ratio = res["wall"] / fine_wall
        with_setup = (res["wall"] + res["setup"]) / (fine["wall"] + fine["setup"])
        @printf("time to solution, %s %s: %.3f of the fine wall, %.3f with setup (%s)\n",
                spec.name, config, ratio, with_setup, ratio < 1 ? "below" : "not below")
    end
    for config in order
        (refined(config) && haskey(censuses, config) && haskey(results, "fine")) ||
            continue
        attribution(spec, config, results[config], results["fine"], censuses[config])
    end
    for (config, why) in failed
        @printf("%s %s not measured, %s\n", spec.name, config, why)
    end
    return results
end

# The wall over the fine run's as a product of three factors. The ideal is the
# refined run's point-steps over the fine run's had its finest level taken
# exactly the fine run's steps (a coarser level a third of them per level when
# subcycled, all of them otherwise) on its time-averaged points; the step
# factor is the measured point-steps over that; the per-point factor is the
# core time per point-step over the fine run's.
function attribution(spec, config, res, fine, c::Census)
    n = max(length(c.tiles), 1)
    finest = findlast(>(0), c.level_steps) - 1
    subcycled = config in ("box", "tiles")
    nf, pf = fine["steps"], prod(spec.n_fine)
    ideal_ps = sum(c.level_size[ℓ+1] / n * nf * (subcycled ? 3.0^(ℓ - finest) : 1.0)
                   for ℓ in 0:finest)
    ideal = ideal_ps / (pf * nf)
    steps = res["point_steps"] / ideal_ps
    cost = (res["wall"] / res["point_steps"]) / (fine["wall"] / fine["point_steps"])
    @printf("attribution, %s %s: wall/fine %.3f = ideal %.3f x steps %.3f", spec.name,
            config, res["wall"] / fine["wall"], ideal, steps)
    @printf(" x per point-step %.3f (level %d steps / fine %.3f)\n", cost, finest,
            c.level_steps[finest+1] / nf)
    row(spec.name, config, "ideal", round(ideal; sigdigits=4))
    row(spec.name, config, "step_factor", round(steps; sigdigits=4))
    row(spec.name, config, "point_cost_factor", round(cost; sigdigits=4))
    row(spec.name, config, "level_steps_over_fine",
        round(c.level_steps[finest+1] / nf; sigdigits=4))
end

# The root spacing along `d`, from the domain and the grid counts by the rule
# each problem's fine grid was sized by, so that a fine grid whose nodes do not
# fall on the refined level's is reported.
function numerics_h(spec, d)
    lo, hi = spec.problem.domain[d]
    n = spec.n_root[d]
    nf = spec.n_fine[d]
    # periodic: L/n with nf = 3n; walls: L/(n − 1) with nf = 3n − 2; one
    # folded end: L/(n − 1/2) with nf = 3n − 1.
    nf == 3n && return (hi - lo) / n
    nf == 3n - 2 && return (hi - lo) / (n - 1)
    nf == 3n - 1 && return (hi - lo) / (n - 0.5)
    return NaN
end

function main()
    # Every rank runs every collective; rank 0 alone prints.
    MPI.Comm_rank(COMM) == 0 || redirect_stdout(devnull)
    @printf("Julia %s, %d ranks, %d threads, %d hardware threads, %s\n", VERSION,
            nranks(), Threads.nthreads(), Sys.CPU_THREADS,
            strip(Sys.cpu_info()[1].model))
    @printf("options: %s\n", join(("$k=$v" for (k, v) in pairs(opt)), " "))
    configs = String.(strip.(split(opt.configs, ',')))
    for name in strip.(split(opt.problems, ','))
        run_problem(problem_spec(name), configs)
    end
end

mpi_main(main)
