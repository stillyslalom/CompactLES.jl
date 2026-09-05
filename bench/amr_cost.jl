# The AMR cost demonstration (reference/AMR_GPU.md): equal accuracy
# at reduced cost on a 3-D mixing case, now that the level transfer is
# distributed. A heavy-gas blob in a light background advects diagonally; the
# interface is the refinement target, and three configurations run the
# identical physics:
#
#   coarse     — uniform N³
#   composite  — N³ root grid with a subcycled refined region tracking the
#                blob through tagging-driven regridding (ratio 3)
#   fine       — uniform (3N−2)³, the resolution the refined region reaches
#
# The metric is the two-run difference of the mixture composition on the
# shared coarse lattice against the fine run (fine node 3k−2 coincides with
# coarse node k), reduced globally; the claim under test is that the
# composite sits near the fine answer at a fraction of the fine cost. Wall
# time is `solver.wall_total` (slowest rank) and memory the rank-summed
# solver+state+workspace footprint.
#
#   mpiexec -n 8 julia --project=. -t 1 bench/amr_cost.jl 48 1.0
#
# Runs serially too (slowly at the fine resolution). `nmax=` caps each run.
# `tag=delta4` (default) tracks the blob on the relative δ⁴ρ criterion;
# `tag=sensor` parks that criterion and tracks it on the artificial
# diffusivity number instead (`tag_sensor_threshold`, the `sensor=` value),
# the scheme's own statement of where it is under-resolved. `cache=` keeps
# the coarse and fine references across the two tag modes.

using MPI
MPI.Init(threadlevel=:funneled)
using CompactLES
using Printf
using Serialization
const CL = CompactLES

const per3 = ntuple(_ -> (PeriodicBC(), PeriodicBC()), 3)

const opt = CompactLES.script_args(ARGS, (N = 48, tfinal = 1.0, nmax = typemax(Int),
                               progress = 0, cache = "", tag = "delta4",
                               sensor = 0.1);
                        positional = (:N, :tfinal))
opt.tag in ("delta4", "sensor") || error("tag must be delta4 or sensor, got '$(opt.tag)'")

blob_ic(u0) = (x, y, z) -> begin
    r2 = (x - π / 2)^2 + (y - π / 2)^2 + (z - π / 2)^2
    θ = 0.5 * (1 - tanh((sqrt(r2) - 0.6) / 0.15))
    Prim(Y=(1 - θ, θ), rho=1.0 + 1.5θ, p=2.0, u=(u0, u0, u0))
end

function run_config(mode, N, tfinal, nmax, progress)
    u0 = 0.5
    eos = IdealMixture([IdealSpecies{Float64}("light", 1.0, 1.4),
                        IdealSpecies{Float64}("heavy", 0.25, 1.09)])
    Nf = mode === :fine ? 3N - 2 : N
    # The sensor mode parks the δ⁴ρ criterion at a threshold nothing
    # reaches and tags on the artificial diffusivity number alone.
    tagkw = opt.tag == "sensor" ?
            (tag_threshold=1e6, tag_sensor_threshold=opt.sensor) :
            (tag_threshold=0.02,)
    kw = mode === :composite ?
         (refine=BlockRegion((N ÷ 8, N ÷ 8, N ÷ 8),
                             (N ÷ 3, N ÷ 3, N ÷ 3)),
          subcycle=true, regrid_interval=10, tag_buffer=4, tagkw...) : (;)
    solver = Solver(n_global=(Nf, Nf, Nf), L_domain=(2π, 2π, 2π), bcs=per3,
                    eos=eos, cfl=0.5; kw...)
    states = allocate_state(solver)
    initialize!(solver, states, blob_ic(u0))
    workspace = Workspace(states)
    mem = Base.summarysize(solver) + Base.summarysize(states) +
          Base.summarysize(workspace)
    mem = MPI.Allreduce(mem, +, MPI.COMM_WORLD)
    cb = progress > 0 ? ProgressLog(every=progress, tfinal=tfinal) : nothing
    run!(solver, states, workspace; tfinal=tfinal, nmax=nmax, callback=cb)
    # Composition on the shared coarse lattice, replicated: the root/coarse
    # patch's own nodes, the fine run sampled at its coincident nodes.
    sample = zeros(N, N, N, 1)
    Q = states isa Vector ? states[1] : states
    patch = getfield(solver, :patches)[1]
    dcp = patch.decomp
    blocks = CL._owned_blocks(dcp, dcp.comm)
    st = mode === :fine ? (3, 3, 3) : (1, 1, 1)
    # Fraction of the heavy species: pack ρY₂/ρ pointwise into a scratch and
    # gather it. Host copy first so the same code serves a future device run.
    Qh = parent(Q) isa Array ? parent(Q) : Array(parent(Q))
    frac = zeros(size(Qh, 1), size(Qh, 2), size(Qh, 3), 1)
    @inbounds for k in axes(Qh, 3), j in axes(Qh, 2), i in axes(Qh, 1)
        ρ = Qh[i, j, k, 1] + Qh[i, j, k, 2]
        frac[i, j, k, 1] = Qh[i, j, k, 2] / ρ
    end
    CL.gather_region!(sample, ntuple(d -> 1:dcp.n_global[d], 3), (0, 0, 0),
                      (0, 0, 0), frac, dcp.comm, blocks, dcp.offset,
                      dcp.n_halo_d, st)
    # The refined cover's bounding box in root nodes (the box itself, or the
    # tiles' extent), for the split metrics below.
    region = BlockRegion((0, 0, 0), (0, 0, 0))
    if mode === :composite
        regs = level_regions(solver, 1)
        lo = ntuple(d -> minimum(r.offset[d] for r in regs), 3)
        hi = ntuple(d -> maximum(r.offset[d] + r.extent[d] for r in regs), 3)
        region = BlockRegion(lo, ntuple(d -> hi[d] - lo[d], 3))
    end
    # Mixedness ∫ Y(1−Y) dV on each run's own grid(s): the physical mixing
    # metric, insensitive to the sub-cell interface displacement that
    # dominates a pointwise comparison. The composite is the masked
    # quadrature of `volume_integral`: coarse nodes under the refined
    # level are excluded by the covered masks, exactly once.
    fs = map(CL.eachpatch(solver, states isa Vector ? states : [states])) do (ps, Qp)
        d = ps.decomp
        p = d.n_halo_d
        Qa = parent(Qp) isa Array ? parent(Qp) : Array(parent(Qp))
        f = zeros(size(Qa, 1), size(Qa, 2), size(Qa, 3))
        for k in 1:d.n_local[3], j in 1:d.n_local[2], i in 1:d.n_local[1]
            ρ1 = Qa[i + p[1], j + p[2], k + p[3], 1]
            ρ2 = Qa[i + p[1], j + p[2], k + p[3], 2]
            Y2 = ρ2 / (ρ1 + ρ2)
            f[i + p[1], j + p[2], k + p[3]] = Y2 * (1 - Y2)
        end
        f
    end
    mixed = volume_integral(solver, fs)
    return (; sample, wall=MPI.Allreduce(solver.wall_total, max, MPI.COMM_WORLD),
            steps=solver.step, mem, region, mixed)
end

# Error metrics against the fine sample, split by whether the coarse node
# lies inside the composite's final refined region — the composite can only
# improve where it refined, and lumping the two regimes together is how a
# cost case gets misread.
function split_metrics(sample, fine, region)
    inr(i, d) = region.offset[d] < i <= region.offset[d] + region.extent[d]
    s_in = 0.0; n_in = 0; m_in = 0.0
    s_out = 0.0; n_out = 0; m_out = 0.0
    for k in axes(sample, 3), j in axes(sample, 2), i in axes(sample, 1)
        d = abs(sample[i, j, k, 1] - fine[i, j, k, 1])
        if inr(i, 1) && inr(j, 2) && inr(k, 3)
            s_in += d^2; n_in += 1; m_in = max(m_in, d)
        else
            s_out += d^2; n_out += 1; m_out = max(m_out, d)
        end
    end
    return (l2_in=sqrt(s_in / max(n_in, 1)), max_in=m_in,
            l2_out=sqrt(s_out / max(n_out, 1)), max_out=m_out)
end

function main()
    rank = MPI.Comm_rank(MPI.COMM_WORLD)
    np = MPI.Comm_size(MPI.COMM_WORLD)
    N, tfinal = opt.N, opt.tfinal
    if rank == 0
        @printf("=== AMR cost case: blob mixing, N=%d vs fine %d, ", N, 3N - 2)
        @printf("t=%.2f, np=%d\n", tfinal, np)
    end
    results = Dict{Symbol,Any}()
    for mode in (:coarse, :composite, :fine)
        suffix = mode === :composite ? "_$(opt.tag)" : ""
        cachefile = isempty(opt.cache) ? "" :
                    joinpath(opt.cache, "amr_cost_$(mode)$(suffix)_$(N)_$(tfinal)_$(np).jls")
        if !isempty(cachefile) && isfile(cachefile)
            results[mode] = Serialization.deserialize(cachefile)
            rank == 0 && @printf("%-10s (cached) steps %5d  wall %8.2f s\n",
                                 mode, results[mode].steps, results[mode].wall)
            continue
        end
        elapsed = @elapsed begin
            results[mode] = run_config(mode, N, tfinal, opt.nmax, opt.progress)
        end
        r = results[mode]
        if rank == 0
            @printf("%-10s steps %5d  solver wall %8.2f s  ",
                    mode, r.steps, r.wall)
            @printf("(script %8.2f s)  memory %7.1f MiB\n",
                    elapsed, r.mem / 2.0^20)
            isempty(cachefile) || Serialization.serialize(cachefile, r)
        end
    end
    if rank == 0
        fine = results[:fine].sample
        region = results[:composite].region
        @printf("composite final region: offset %s extent %s\n",
                region.offset, region.extent)
        for mode in (:coarse, :composite)
            m = split_metrics(results[mode].sample, fine, region)
            @printf("%-10s vs fine:  in-region L2 %.4e max %.4e   ",
                    mode, m.l2_in, m.max_in)
            @printf("outside L2 %.4e max %.4e\n", m.l2_out, m.max_out)
        end
        Mf = results[:fine].mixed
        @printf("mixedness ∫Y(1-Y)dV: coarse %.6e  composite %.6e  fine %.6e\n",
                results[:coarse].mixed, results[:composite].mixed, Mf)
        @printf("mixedness error vs fine: coarse %+.3e  composite %+.3e\n",
                results[:coarse].mixed - Mf, results[:composite].mixed - Mf)
        wc, wm, wf = (results[m].wall for m in (:coarse, :composite, :fine))
        @printf("cost: composite %.2fx coarse wall, %.1f%% of fine wall\n",
                wm / wc, 100wm / wf)
    end
end

mpi_main(main)
