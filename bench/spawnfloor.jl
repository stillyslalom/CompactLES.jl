# Per-region spawn/join floor of `Threads.@threads` at this session's thread
# count. One threaded region over a trivial body is timed many times; the
# minimum is the cost before any region work happens and the quantity
# `THREAD_MIN_WORK` amortizes (see the `@threaded` docstring). The
# floor grows with the thread count — 0.6 µs at 1 thread to 14 µs at 24 on a
# 24-thread desktop — so the curve comes from running this script at several
# `-t` values, one process each, not from one process varying anything:
#
#     for t in 1 2 4 8 16; do julia --project=. -t $t bench/spawnfloor.jl; done
#     srun -N 1 -n 1 --cpu-bind=threads julia --project=. -t 8 bench/spawnfloor.jl
#
# Run-to-run spread applies here as everywhere (10–20%); the minimum within a
# process is stable, but compare processes by repeating them.
#
# The derived line multiplies the floor by `regions`, the approximate number of
# threaded regions one `compute_rhs!` evaluation enters (~120 after the fused
# scatters; the count is in the note at the top of `src/threading.jl`). That
# product is the per-RHS overhead a threaded rank pays before threads help at
# all, and is the quantity to weigh against the halo-exchange savings of a
# coarser decomposition.

using CompactLES
using Printf

function measure(trips::Int, reps::Int)
    out = Vector{Int}(undef, trips)
    threaded_once() = (Threads.@threads for i in 1:trips
        out[i] = i
    end)
    serial_once() = (for i in 1:trips
        out[i] = i
    end)
    sample(f) = begin
        f(); f()                       # compile + warm
        t = Vector{Float64}(undef, reps)
        for r in 1:reps
            t0 = time_ns()
            f()
            t[r] = (time_ns() - t0) / 1e3   # µs
        end
        sort!(t)
        (min = t[1], med = t[cld(reps, 2)])
    end
    return sample(threaded_once), sample(serial_once),
           Threads.nthreads() > 1 ? @allocated(threaded_once()) : 0
end

function main(args)
    opt = script_args(args, (trips = 0, reps = 10_000, regions = 120))
    trips = opt.trips > 0 ? opt.trips : Threads.nthreads()
    threaded, serial, bytes = measure(trips, opt.reps)
    @printf("-t %-3d trips=%-4d  spawn floor %7.2f µs (median %7.2f)",
            Threads.nthreads(), trips, threaded.min, threaded.med)
    @printf("  serial %6.3f µs  %6d B/region\n", serial.min, bytes)
    @printf("       x %d regions/RHS = %.1f ms per RHS call before any speedup\n",
            opt.regions, threaded.min * opt.regions / 1e3)
end

main(ARGS)
