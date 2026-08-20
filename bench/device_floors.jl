# Cross-machine device performance survey. One machine's timings are not a
# property of the code: the rzadams MI300A log (bench/logs/rzadams_20260819.txt)
# shows C6 Float64 line solves at 3.0/6.0 ms in dims 1-2 against ~0.3 ms for
# every other scheme x dim, and a 26x Float64/Float32 whole-step ratio on
# hardware whose FP64 vector rate is half its FP32 rate — neither visible on
# the RDNA2 workstation. This script is the instrument for such questions: run
# it unchanged on each machine and compare tables, rather than reading any one
# machine's numbers as a property of the implementation.
#
# Sections:
#   host sanity — a threaded-speedup probe. A core-starved allocation (a
#       launcher default of one core per task under several Julia threads)
#       inflates every host-relative ratio 10-30x and is invisible in the
#       timings themselves; this probe makes it visible up front.
#   floors — kernel submission, submission+synchronize round trip, and an
#       idle-queue synchronize. These bound what any launch policy can recover
#       and price the per-apply reduced-solve fence.
#   stall watch — one fixed apply hammered for `watch` seconds with every
#       call timed. rzadams (MI300A, ROCm 6.4.3, Julia 1.12.7) shows sustained
#       episodes, seconds long, in which each device wait costs an integer
#       number of milliseconds instead of ~0.15 ms; they strike arbitrary
#       scheme/dim/precision cells (min-over-50 readings of 2/3/5/13 ms and
#       one whole TGV leg at 27x) and are invisible in a table that samples
#       each cell once. This section measures their rate and dwell time and
#       reports GC time alongside, since the ms quantization implicates a
#       sleeping wait path (OS timer granularity) or the runtime's scheduler.
#   line solves — scheme (C6/C10/C8 filter) x dim x precision x size, device
#       and host apply walls. A per-apply cost flat across sizes is a
#       latency/serialization floor; one that scales with n is throughput
#       (occupancy, register spill). Methodology matches
#       bench/device_bringup.jl so rows compare with existing logs.
#   TGV steps — warm device s/step at scheme x precision, single species,
#       artificial properties off. If a Float64 step anomaly collapses when
#       C6 is swapped for C10, the line solve owns it; if it persists, it is
#       precision-wide. CPU-vs-device TGV lives in bench/device_solver.jl and
#       is not repeated here.
#
# Multi-rank launches are supported and are the natural shape on a multi-APU
# node (one rank per accelerator): the per-device sections run independently
# on every rank over MPI.COMM_SELF and rank 0 prints the gathered reports, so
# per-device variance on one node comes for free. The TGV section instead
# decomposes the one case across all ranks when np > 1 — one device per rank
# plus staged halo exchange is the production shape, and the printed s/step
# is labeled accordingly. When the launcher exposes several devices to one
# rank, ranks pick devices round-robin by node-local rank, so ranks cannot
# silently share one accelerator.
#
# Device packages are not CompactLES dependencies, so run from an environment
# carrying CompactLES AND the device package:
#
#   julia --project=<env-with-AMDGPU> -t 8 bench/device_floors.jl backend=amdgpu
#   flux run -N1 -n4 --exclusive julia --project -t 8 \
#       bench/device_floors.jl backend=amdgpu
#
# First-launch kernel compilation dominates the wall time of a fresh process;
# every printed number is a minimum over reps after warm-up.

using MPI
MPI.Init(threadlevel=:funneled)
using CompactLES
using Printf
const CL = CompactLES
# KernelAbstractions through CompactLES's own import, so the script runs from
# any environment carrying CompactLES plus a device package and nothing else.
const KA = CL.KernelAbstractions

opt = script_args(ARGS, (backend = "amdgpu", sizes = "32,64,96", n = 64,
                         steps = 10, reps = 50, watch = 30))

# select_device! takes the node-local rank and is a no-op when the launcher
# already restricts each rank to one visible device.
device_array, ka_backend, device_description, select_device! =
    if opt.backend == "amdgpu"
        @eval using AMDGPU
        @eval AMDGPU.functional() || error("AMDGPU is not functional here")
        @eval (ROCArray, ROCBackend(), () -> string(AMDGPU.device()),
               lr -> begin
                   devs = AMDGPU.devices()
                   AMDGPU.device!(devs[mod(lr, length(devs)) + 1])
               end)
    elseif opt.backend == "cuda"
        @eval using CUDA
        @eval CUDA.functional() || error("CUDA is not functional here")
        @eval (CuArray, CUDABackend(), () -> string(CUDA.device()),
               lr -> CUDA.device!(mod(lr, length(CUDA.devices()))))
    else
        error("backend must be amdgpu or cuda, got $(opt.backend)")
    end

best(f; reps) = (f(); f(); minimum(@elapsed(f()) for _ in 1:reps))

# A dependent multiply-add chain: it measures fair core time per thread, not
# peak FLOPs, which is exactly what a starved allocation distorts.
function _spin(seed::Float64, iters::Int)
    s = seed
    for _ in 1:iters
        s = muladd(s, 0.999999, 1e-9)
    end
    return s
end

function host_sanity(io)
    nt = Threads.nthreads()
    @printf(io, "julia %s on %s (%s), %d threads of %d hardware threads\n",
            string(VERSION), Base.Libc.gethostname(), Sys.MACHINE, nt,
            Sys.CPU_THREADS)
    iters = 100_000_000
    sink = zeros(nt)
    _spin(0.5, 1000)
    Threads.@threads :static for t in 1:nt
        sink[t] = _spin(0.5, 1000)
    end
    # Keep every result in `sink`: a discarded _spin call is a pure loop the
    # compiler deletes, and t1 then measures nothing.
    t1 = @elapsed (sink[1] = _spin(0.5, iters))
    tn = @elapsed Threads.@threads :static for t in 1:nt
        sink[t] = _spin(0.5, iters)
    end
    @printf(io, "host threading: %d identical compute-bound tasks take ", nt)
    @printf(io, "%.2fx one task's wall (ideal 1.0)\n", tn / t1)
    if nt > 1 && tn / t1 > 2.0
        println(io, "  WARNING: threads appear core-starved; host-relative",
                " ratios below are not meaningful.")
        println(io, "  Check the launcher's cores-per-task and binding",
                " before reading any host number.")
    end
    return nothing
end

function floors(io, gpu, device_array, reps)
    g = device_array(randn(8, 8, 8))
    ih = device_array(ones(8, 8, 8))
    launch() = CL.pointwise_ka!(CL._scale_grad_point!, gpu, 8, 8, 8, g, ih)
    launch()
    KA.synchronize(gpu)
    t_sub = best(launch; reps=reps)
    KA.synchronize(gpu)
    t_rt = best(() -> (launch(); KA.synchronize(gpu)); reps=reps)
    t_idle = best(() -> KA.synchronize(gpu); reps=reps)
    @printf(io, "submission %.1f us   launch+sync %.1f us   idle sync %.1f us\n",
            1e6t_sub, 1e6t_rt, 1e6t_idle)
    return nothing
end

function _median_sorted(s::Vector{Float64})
    n = length(s)
    return isodd(n) ? s[(n + 1) >> 1] : 0.5 * (s[n >> 1] + s[(n >> 1) + 1])
end

function stall_watch(io, gpu, device_array, seconds)
    T = Float64
    nn = 32
    decomp = Decomp{T}((nn, nn, nn), (true, true, true);
                       dims=(1, 1, 1), comm=MPI.COMM_SELF)
    fh = CL.field(decomp)
    copyto!(fh, randn(T, size(fh)...))
    CL.exchange_halos!(fh, decomp)
    fg = device_array(fh)
    og = device_array(CL.field(decomp))
    plan = CL.plan_direction(decomp, lele_d1_6(T), 1, T(1) / nn)
    dplan = device_plan(plan, gpu)
    CL.apply_along!(og, dplan, fg, decomp)
    CL.apply_along!(og, dplan, fg, decomp)
    times = Float64[]
    sizehint!(times, ceil(Int, 1e4 * seconds))
    gc0 = Base.gc_num()
    t0 = time()
    while time() - t0 < seconds
        push!(times, @elapsed CL.apply_along!(og, dplan, fg, decomp))
    end
    gc_ms = Base.GC_Diff(Base.gc_num(), gc0).total_time / 1e6
    s = sort(times)
    med = _median_sorted(s)
    p99 = s[ceil(Int, 0.99 * length(s))]
    @printf(io, "%d calls: median %.3f ms  p99 %.3f ms  max %.3f ms  ",
            length(s), 1e3med, 1e3p99, 1e3s[end])
    @printf(io, "gc %.0f ms\n", gc_ms)
    # A slow call is one above 1 ms absolute. The fast baseline for this
    # apply is 0.13-0.45 ms on every machine measured and stalled calls sit
    # at 2 ms and above; a relative (multiple-of-median) threshold inverts
    # when a window is stalled throughout, since the median itself is then
    # milliseconds (measured on rzadams: median 13.000 ms for a full watch).
    # An episode is a maximal run of consecutive slow calls; dwell time
    # separates sustained stalls from one-off hiccups (GC, preemption).
    thr = 1e-3
    n_ep = 0
    n_slow = 0
    stall_t = 0.0
    cur = 0.0
    longest = 0.0
    offset = 0.0
    ep_start = 0.0
    episodes = NTuple{2,Float64}[]
    for t in times
        if t > thr
            cur == 0.0 && (n_ep += 1; ep_start = offset)
            n_slow += 1
            cur += t
            stall_t += t
            longest = max(longest, cur)
        else
            cur >= 0.1 && length(episodes) < 8 && push!(episodes, (ep_start, cur))
            cur = 0.0
        end
        offset += t
    end
    cur >= 0.1 && length(episodes) < 8 && push!(episodes, (ep_start, cur))
    @printf(io, "slow calls >1 ms: %d, %.2f s total (%.1f%% of watch); ",
            n_slow, stall_t, 100stall_t / sum(times))
    @printf(io, "episodes: %d, longest %.2f s\n", n_ep, longest)
    for (st, du) in episodes
        @printf(io, "    sustained episode at %6.2f s, %.3f s long\n", st, du)
    end
    edges = [0.25e-3, 0.5e-3, 1e-3, 2e-3, 5e-3, 10e-3]
    counts = zeros(Int, length(edges) + 1)
    for t in times
        counts[searchsortedfirst(edges, t)] += 1
    end
    labels = ("<=0.25ms", "0.25-0.5", "0.5-1", "1-2", "2-5", "5-10", ">10ms")
    for (lab, c) in zip(labels, counts)
        c > 0 && @printf(io, "  %s: %d", lab, c)
    end
    println(io)
    return nothing
end

function line_matrix(io, gpu, device_array, sizes, reps)
    @printf(io, "%-16s", "")
    for nn in sizes
        @printf(io, "  %10s (host)  ", "n=$nn")
    end
    println(io)
    for T in (Float64, Float32)
        fields = map(sizes) do nn
            decomp = Decomp{T}((nn, nn, nn), (true, true, true);
                               dims=(1, 1, 1), comm=MPI.COMM_SELF)
            fh = CL.field(decomp)
            copyto!(fh, randn(T, size(fh)...))
            CL.exchange_halos!(fh, decomp)
            (decomp=decomp, fh=fh, fg=device_array(fh),
             oh=CL.field(decomp), og=device_array(CL.field(decomp)))
        end
        schemes = (("C6", lele_d1_6(T)), ("C10", lele_d1_10(T)),
                   ("filt", compact_filter(T(0.45), T)))
        for (label, scheme) in schemes, dim in 1:3
            @printf(io, "%s %-5s dim %d ", T === Float64 ? "F64" : "F32",
                    label, dim)
            mismatch = false
            for w in fields
                nn = w.decomp.n_global[1]
                plan = CL.plan_direction(w.decomp, scheme, dim, T(1) / nn)
                dplan = device_plan(plan, gpu)
                CL.apply_along!(w.oh, plan, w.fh, w.decomp)
                CL.apply_along!(w.og, dplan, w.fg, w.decomp)
                dmax = maximum(abs.(view(Array(w.og), CL.interior(w.decomp))
                                    .- view(w.oh, CL.interior(w.decomp))))
                mismatch |= dmax != 0
                t_dev = best(() -> CL.apply_along!(w.og, dplan, w.fg,
                                                   w.decomp); reps=reps)
                t_host = best(() -> CL.apply_along!(w.oh, plan, w.fh,
                                                    w.decomp); reps=reps)
                @printf(io, "  %8.3f (%8.3f)", 1e3t_dev, 1e3t_host)
            end
            println(io, mismatch ? "   |dev-host| != 0 !" : "")
        end
    end
    return nothing
end

function tgv_build(T, deriv, n, ka_backend)
    per = (PeriodicBC(), PeriodicBC())
    γ = T(1.4)
    c0 = T(10)
    p0 = c0^2 / γ
    s = Solver(n_global=(n, n, n), L_domain=(T(2π), T(2π), T(2π)),
               bcs=(per, per, per), eos=single_species(T),
               transport=Transport{T}(mu0=T(1 / 1600)),
               art=ArtParams{T}(enabled=false),
               deriv=deriv, filt=compact_filter(T(0.45), T),
               cfl=T(0.6), backend=DeviceBackend(ka_backend))
    Q = allocate_state(s)
    initialize!(s, Q, (x, y, z) -> Prim(
        u=(sin(x) * cos(y) * cos(z), -cos(x) * sin(y) * cos(z), zero(T)),
        p=p0 + one(T) / T(16) * (cos(T(2)x) + cos(T(2)y)) * (cos(T(2)z) + T(2)),
        rho=one(T)))
    return s, Q
end

function tgv_steps(ka_backend, n, steps, comm)
    rank = MPI.Comm_rank(comm)
    for T in (Float64, Float32)
        for (label, deriv) in (("C6 ", lele_d1_6(T)), ("C10", lele_d1_10(T)))
            s, Q = tgv_build(T, deriv, n, ka_backend)
            run!(s, Q; tfinal=1e6, nmax=steps)
            s, Q = tgv_build(T, deriv, n, ka_backend)
            run!(s, Q; tfinal=1e6, nmax=steps)
            wall = MPI.Allreduce(s.wall_total, max, comm)
            rank == 0 && @printf("TGV %s %s %d^3: device %.4f s/step ",
                                 T === Float64 ? "F64" : "F32", label, n,
                                 wall / s.step)
            rank == 0 && @printf("(warm, %d steps)\n", s.step)
        end
    end
    return nothing
end

function main(opt, device_array, ka_backend, device_description, select_device!)
    comm = MPI.COMM_WORLD
    np = MPI.Comm_size(comm)
    rank = MPI.Comm_rank(comm)
    local_comm = MPI.Comm_split_type(comm, MPI.COMM_TYPE_SHARED, rank)
    select_device!(MPI.Comm_rank(local_comm))
    sizes = parse.(Int, split(opt.sizes, ','))
    io = IOBuffer()
    println(io, "device: ", device_description())
    println(io, "-- host sanity")
    host_sanity(io)
    println(io, "-- launch floors (min over $(opt.reps))")
    floors(io, ka_backend, device_array, opt.reps)
    println(io, "-- stall watch ($(opt.watch) s of C6 F64 dim-1 n=32 applies)")
    stall_watch(io, ka_backend, device_array, opt.watch)
    println(io, "-- line solves, ms/apply (device, host in parens; ",
            "min over $(opt.reps))")
    line_matrix(io, ka_backend, device_array, sizes, opt.reps)
    reports = MPI.gather(String(take!(io)), comm)
    if rank == 0
        for (r, report) in enumerate(reports)
            np > 1 && println("======== rank $(r - 1)")
            print(report)
        end
        println(np > 1 ? "== TGV device steps (decomposed over $np ranks)" :
                "== TGV device steps")
    end
    tgv_steps(ka_backend, opt.n, opt.steps, comm)
    return nothing
end

mpi_main(() -> main(opt, device_array, ka_backend, device_description,
                    select_device!))
