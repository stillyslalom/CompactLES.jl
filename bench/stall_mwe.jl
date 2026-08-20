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
#   mode=copy     kernel + device-to-host copy (adds the Managed sync path)
#   mode=full     kernel + D2H + host arithmetic + H2D (the shape of one
#                 compact-solve apply, with no solver code)
#
# Climb only as far as needed: if mode=kernel stalls at -t 8 and not at
# -t 1, the reproducer is ~40 lines of AMDGPU.jl and the next
# discrimination is stall_mwe.cpp (does a plain HIP loop with dormant
# extra threads stall — Julia involved or not?). sync= selects the wait:
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

const opts = Dict{String,String}()
for a in ARGS
    k, v = split(a, '='; limit=2)
    opts[String(k)] = String(v)
end
getopt(k, d) = get(opts, k, d)

const mode = getopt("mode", "kernel")
const watch = parse(Float64, getopt("watch", "120"))
const work = parse(Int, getopt("work", "20000"))
const syncmode = getopt("sync", "blocking")

function spin_kernel!(x, iters)
    i = workitemIdx().x
    acc = x[i]
    for _ in 1:iters
        acc = muladd(acc, 0.9999f0, 1f-7)
    end
    x[i] = acc
    return
end

sync_default() = AMDGPU.synchronize()
sync_blocking() = AMDGPU.synchronize(; blocking=true)
sync_direct() = AMDGPU.HIP.hipStreamSynchronize(AMDGPU.stream())

function main()
    dosync = syncmode == "default" ? sync_default :
             syncmode == "direct" ? sync_direct : sync_blocking
    n = 256
    x = ROCArray(fill(1.0f0, n))
    host = zeros(Float32, n)
    call! = if mode == "kernel"
        () -> (@roc groupsize=n spin_kernel!(x, work); dosync())
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
        error("mode must be kernel, kernel2, copy or full, got $mode")
    end
    call!(); call!()
    t_call = minimum(@elapsed(call!()) for _ in 1:50)

    @printf("julia %s on %s, %d default + %d interactive threads, pid %d\n",
            string(VERSION), Base.Libc.gethostname(), Threads.nthreads(),
            Threads.nthreads(:interactive), getpid())
    println("device: ", AMDGPU.device())
    hsaint = get(ENV, "HSA_ENABLE_INTERRUPT", "(unset)")
    @printf("mode=%s sync=%s work=%d  HSA_ENABLE_INTERRUPT=%s  ",
            mode, syncmode, work, hsaint)
    @printf("calibration %.3f ms/call\n", 1e3t_call)

    times = Float64[]
    sizehint!(times, ceil(Int, 2e4 * watch))
    t0 = time()
    while time() - t0 < watch
        push!(times, @elapsed call!())
    end

    s = sort(times)
    med = s[(length(s) + 1) >> 1]
    p99 = s[ceil(Int, 0.99 * length(s))]
    @printf("%d calls: median %.3f ms  p99 %.3f ms  max %.3f ms\n",
            length(s), 1e3med, 1e3p99, 1e3s[end])
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
