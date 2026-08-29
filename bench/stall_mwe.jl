# Minimal reproducer ladder for the MI300A wait-stall mode
# (reference/rocm_wait_stall_report.md). Deliberately depends on AMDGPU.jl
# alone — no CompactLES, no KernelAbstractions, no MPI — so the file can be
# attached to a ticket as-is. Run from any environment carrying AMDGPU:
#
#   julia --project=<env-with-AMDGPU> -t 8 stall_mwe.jl watch=120
#   julia --project=<env-with-AMDGPU> -t 1 stall_mwe.jl watch=120
#
# The full survey (bench/device_floors.jl) measures the stall at every
# thread count above 1: episodes of seconds to minutes in which every
# device wait costs ~13 or ~26 ms — ROCR's polling-wait sleep cadence,
# reproduced deterministically by HSA_ENABLE_INTERRUPT=0 — instead of the
# microseconds the interrupt path delivers. This script asks how little it
# takes to trigger the same degradation:
#
#   mode=kernel   one trivial kernel launch + stream synchronize per call
#   mode=kernel2  two launches, one synchronize
#   mode=multi    eight launches of eight DISTINCT kernel objects, one
#                 synchronize (the real call's kernel diversity)
#   mode=copy     kernel + device-to-host copy (adds the Managed sync path)
#   mode=full     kernel + D2H + host arithmetic + H2D (the shape of one
#                 compact-solve apply, with no solver code)
#   mode=ka       the kernel launched through KernelAbstractions on
#                 ROCBackend instead of @roc — the one layer between the
#                 clean rungs and the stalling survey watch. Needs
#                 KernelAbstractions in the environment
#                 (import Pkg; Pkg.add("KernelAbstractions")).
# Orthogonal additions: mpi=1 (initialize MPI, communicate nothing),
# alloc=N (N heap bytes per call, drives GC), burst=1 (compile and launch
# 40 distinct kernels before the watch — the startup storm).
#
# Climb only as far as needed: if a rung stalls at -t 8 and not at -t 1,
# that rung is the reproducer and the next discrimination is
# stall_mwe.cpp (does a plain HIP loop with dormant extra threads stall —
# Julia involved or not?). Measured so far on rzadams: mode=kernel is
# clean at -n1 and -n4 (120 s windows), while the full survey's watch
# stalls even at -n1 — the ingredient is call content or process history,
# and mpi=1 (initialize MPI, communicate nothing) is the highest-prior
# single addition since every stalling run had MPI initialized.
# sync= selects the wait:
#   blocking  AMDGPU.synchronize(blocking=true)  (default; no Julia tasks)
#   default   AMDGPU.synchronize()               (spin + task fallback)
#   direct    raw hipStreamSynchronize ccall     (bypasses AMDGPU.jl logic)
#
# `work` tunes kernel duration; the calibration line reports it. The
# solver's stalled waits had ~0.15 ms of device work per call, so the
# default targets tens of microseconds — long enough that the wait
# genuinely waits, short enough that a 13 ms sleep quantum is a 100x hit.

using AMDGPU
using Printf

# Settings come from ARGS as key=value, the shape `script_args` gives every
# other script here. This file carries no CompactLES dependency, so the
# parse is spelled out, including the two failure modes that matter: an
# unknown key is an error rather than a default silently run for the length
# of a job, and an argument without '=' is named rather than a BoundsError.
const KNOWN_KEYS = ("mode", "watch", "work", "sync", "mpi", "alloc", "burst")
const opts = Dict{String,String}()
for a in ARGS
    parts = split(a, '='; limit=2)
    length(parts) == 2 ||
        error("argument \"$a\" is not key=value; keys: " *
              join(KNOWN_KEYS, ", "))
    key = String(parts[1])
    key in KNOWN_KEYS ||
        error("unknown key \"$key\"; keys: " * join(KNOWN_KEYS, ", "))
    opts[key] = String(parts[2])
end
getopt(k, d) = get(opts, k, d)

const mode = getopt("mode", "kernel")
const watch = parse(Float64, getopt("watch", "120"))
const work = parse(Int, getopt("work", "20000"))
const syncmode = getopt("sync", "blocking")
const use_mpi = getopt("mpi", "0") == "1"
const allocbytes = parse(Int, getopt("alloc", "0"))
const use_burst = getopt("burst", "0") == "1"

# mpi=1 initializes MPI and nothing else — no communication follows. Every
# run observed to stall had MPI initialized (Cray MPICH: progress threads,
# memory-registration hooks, GPU-aware transport plumbing) and every clean
# MWE run did not, so MPI initialization is the single highest-prior
# ingredient. The default stays 0 to keep the file AMDGPU-only.
if use_mpi
    @eval using MPI
    @eval MPI.Init(threadlevel=:funneled)
end

if mode == "ka"
    @eval using KernelAbstractions
    @eval KernelAbstractions.@kernel function ka_spin!(x, iters)
        i = KernelAbstractions.@index(Global)
        acc = x[i]
        for _ in 1:iters
            acc = muladd(acc, 0.9999f0, 1f-7)
        end
        x[i] = acc
    end
end

function spin_kernel!(x, iters)
    i = workitemIdx().x
    acc = x[i]
    for _ in 1:iters
        acc = muladd(acc, 0.9999f0, 1f-7)
    end
    x[i] = acc
    return
end

# Eight structurally distinct kernels for mode=multi: the real solver call
# launches ~6-8 different kernel objects per apply (fill, sweeps, spike
# correction, pack, scatter), cycling that many completion signals, where
# the clean rungs reuse one kernel object. The perturbation constant makes
# each function unique so each compiles to its own code object.
for k in 1:8
    @eval function $(Symbol(:spin_kernel_, k, :!))(x, iters)
        i = workitemIdx().x
        acc = x[i] + $(Float32(k)) * 1.0f-6
        for _ in 1:iters
            acc = muladd(acc, 0.9999f0, 1f-7)
        end
        x[i] = acc
        return
    end
end
const multi_kernels = (spin_kernel_1!, spin_kernel_2!, spin_kernel_3!,
                       spin_kernel_4!, spin_kernel_5!, spin_kernel_6!,
                       spin_kernel_7!, spin_kernel_8!)

# Defined at top level (not inside main) so the fresh methods are visible
# when launched — an @eval inside the running function trips world age.
for k in 1:40
    @eval function $(Symbol(:burst_kernel_, k, :!))(x)
        i = workitemIdx().x
        x[i] += $(Float32(k)) * 1.0f-9
        return
    end
end
const burst_kernels = Tuple(eval(Symbol(:burst_kernel_, k, :!)) for k in 1:40)

sync_default() = AMDGPU.synchronize()
sync_blocking() = AMDGPU.synchronize(; blocking=true)
sync_direct() = AMDGPU.HIP.hipStreamSynchronize(AMDGPU.stream())

function main()
    dosync = syncmode == "default" ? sync_default :
             syncmode == "direct" ? sync_direct : sync_blocking
    n = 256
    x = ROCArray(fill(1.0f0, n))
    host = zeros(Float32, n)
    # burst=1 compiles and launches 40 additional distinct kernels before
    # the watch: several stalled survey windows were stalled from t = 0,
    # implicating the startup compile/module-load storm as the event that
    # leaves the process in the degraded state.
    if use_burst
        for f in burst_kernels
            @roc groupsize=n f(x)
        end
        AMDGPU.synchronize(; blocking=true)
    end
    call! = if mode == "kernel"
        () -> (@roc groupsize=n spin_kernel!(x, work); dosync())
    elseif mode == "ka"
        kab = KernelAbstractions.get_backend(x)
        k! = ka_spin!(kab)
        () -> (k!(x, work; ndrange=n); dosync())
    elseif mode == "multi"
        wk = max(1, work ÷ 8)
        () -> begin
            for f in multi_kernels
                @roc groupsize=n f(x, wk)
            end
            dosync()
        end
    elseif mode == "kernel2"
        () -> (@roc groupsize=n spin_kernel!(x, work);
               @roc groupsize=n spin_kernel!(x, work); dosync())
    elseif mode == "copy"
        () -> (@roc groupsize=n spin_kernel!(x, work); dosync();
               copyto!(host, x))
    elseif mode == "full"
        () -> (@roc groupsize=n spin_kernel!(x, work); dosync();
               copyto!(host, x); host .= host .* 0.5f0 .+ 0.25f0;
               copyto!(x, host))
    else
        error("mode must be kernel, kernel2, multi, copy, full or ka, got $mode")
    end
    # alloc=N allocates N heap bytes per call, driving the garbage
    # collector the way the real solver call does (its device-to-host
    # staging materializes a host array every apply; ~1 s of GC per 120 s
    # window). GC does not sustain the stalled state — a fully stalled
    # window measured zero GC time — but a collection's inter-thread stop
    # signals are a candidate for the *entry* event, and would explain the
    # more-than-one-default-thread gate: stopping the world only signals
    # other threads when other threads exist.
    if allocbytes > 0
        inner! = call!
        call! = () -> (Base.donotdelete(Vector{UInt8}(undef, allocbytes));
                       inner!())
    end
    call!(); call!()
    t_call = minimum(@elapsed(call!()) for _ in 1:50)

    @printf("julia %s on %s, %d default + %d interactive threads, pid %d\n",
            string(VERSION), Base.Libc.gethostname(), Threads.nthreads(),
            Threads.nthreads(:interactive), getpid())
    println("device: ", AMDGPU.device())
    hsaint = get(ENV, "HSA_ENABLE_INTERRUPT", "(unset)")
    @printf("mode=%s sync=%s work=%d mpi=%d alloc=%d burst=%d  ",
            mode, syncmode, work, use_mpi ? 1 : 0, allocbytes,
            use_burst ? 1 : 0)
    @printf("HSA_ENABLE_INTERRUPT=%s  calibration %.3f ms/call\n",
            hsaint, 1e3t_call)

    times = Float64[]
    sizehint!(times, ceil(Int, 2e4 * watch))
    gc0 = Base.gc_num()
    t0 = time()
    while time() - t0 < watch
        push!(times, @elapsed call!())
    end
    gc_ms = Base.GC_Diff(Base.gc_num(), gc0).total_time / 1e6

    s = sort(times)
    med = s[(length(s) + 1) >> 1]
    p99 = s[ceil(Int, 0.99 * length(s))]
    @printf("%d calls: median %.3f ms  p99 %.3f ms  max %.3f ms  gc %.0f ms\n",
            length(s), 1e3med, 1e3p99, 1e3s[end], gc_ms)
    # Slow threshold 1 ms absolute: baseline calls sit well under 0.5 ms
    # and ROCR's polling sleep quantum is ~13 ms; a relative threshold
    # inverts when the whole window is stalled.
    thr = 1e-3
    n_slow = 0
    stall_t = 0.0
    cur = 0.0
    longest = 0.0
    offset = 0.0
    ep_start = 0.0
    episodes = NTuple{2,Float64}[]
    for t in times
        if t > thr
            cur == 0.0 && (ep_start = offset)
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
    @printf("slow calls >1 ms: %d, %.2f s total (%.1f%% of watch); ",
            n_slow, stall_t, 100stall_t / sum(times))
    @printf("longest episode %.2f s\n", longest)
    for (st, du) in episodes
        @printf("    sustained episode at %6.2f s, %.3f s long\n", st, du)
    end
    edges = [0.25e-3, 0.5e-3, 1e-3, 2e-3, 5e-3, 10e-3]
    counts = zeros(Int, length(edges) + 1)
    for t in times
        counts[searchsortedfirst(edges, t)] += 1
    end
    labels = ("<=0.25ms", "0.25-0.5", "0.5-1", "1-2", "2-5", "5-10", ">10ms")
    for (lab, c) in zip(labels, counts)
        c > 0 && @printf("  %s: %d", lab, c)
    end
    println()
    return nothing
end

main()
