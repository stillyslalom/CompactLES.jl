# Threading policy.
#
# `Threads.@threads` spawns one Task per thread per region. That is free when
# the loop body is large and ruinous when it is not, and this solver is full of
# small regions: one compute_rhs! evaluation runs ~150 of them (one per
# direction per velocity component, per species, per flux, plus every pointwise
# rescale), so at 24 threads a single RHS call spawns ~3600 tasks and allocates
# ~150 kB *per thread*, independent of grid size.
#
# Measured before this policy existed (24 logical cores, compute_rhs!):
#
#     grid          -t 1        -t 24
#     1-D, 512      0.14 ms     1.16 ms     8.3x SLOWER
#     24^3          7.2  ms    11.8  ms     1.6x SLOWER
#     48^3         45.6  ms    21.3  ms     2.1x faster
#     96^3        340.7  ms    87.5  ms     3.9x faster
#
# So the axisymmetric and 1-D radial runs — converging_shock.jl, the Sod tube,
# anything with a collapsed dimension — were paying a large penalty for asking
# for threads, which the README tells users to do.
#
# `@threaded work for ... end` runs the loop inline, with no task spawn and no
# allocation, when `work` is below THREAD_MIN_WORK, and defers to
# Threads.@threads otherwise. `work` is the number of grid points the loop will
# touch in total — NOT its trip count — so a `for k in 1:nz` loop carrying an
# inner `j, i` nest passes `nx * ny * nz`.

"""
Minimum total work, in grid points, required to use the thread pool. The default
follows the measurements above: 32^3 is approximately the break-even point and
48^3 benefits from threading.
Override with the `CL_THREAD_MIN_WORK` environment variable.
"""
const THREAD_MIN_WORK = Ref(1 << 15)

function __init_threading__()
    v = get(ENV, "CL_THREAD_MIN_WORK", nothing)
    v === nothing || (THREAD_MIN_WORK[] = parse(Int, v))
    return nothing
end

# Takes Int, not Integer: an abstract parameter type leaves the `>=` below as a
# runtime dispatch, once per threaded region (~150 per RHS call).
@inline _use_threads(work::Int) =
    Threads.nthreads() > 1 && work >= THREAD_MIN_WORK[]

"""
    @threaded work for ... end

Thread the loop only when `work` (total grid points touched) justifies the
parallel-dispatch cost; otherwise run it inline. See the note at the top of
this file.

## Threading backend

`@batch` reduced allocation in `compute_rhs!` at 48^3 on 24 threads from
3,638,096 B to 37,024 B, but increased total runtime. The following measurements
are for the 64^3 Taylor–Green case with `-t 8`:

    phase                Threads.@threads   Polyester.@batch
    primitives!                 1.11 ms          0.43 ms
    assemble_fluxes!            1.79 ms          1.84 ms
    velocity grads              3.95 ms         31.78 ms
    scalar grads                2.72 ms         24.12 ms
    flux divergence             7.00 ms         60.30 ms
    compute_rhs! (total)       22.26 ms        133.29 ms

`@batch` is faster for the two phases consisting of a single large pointwise
loop and approximately 8–9 times slower for phases that invoke compact
operators. The latter execute about 36 small gather, solve, and scatter regions
sequentially per gradient phase. Synchronizing the `@batch` pool for each region
costs more than spawning tasks in this workload.

Combining the backends by call depth was also slower. Routing operator work to
tasks while retaining `@batch` for leaf loops measured 1279.95 ms, 58 times the
all-task result, because Polyester's spin-waiting workers occupy the Julia
threads used by `Threads.@threads`.

The implementation therefore uses `Threads.@threads` throughout and applies the
work threshold above to avoid task overhead on small cases. The backend choice
should be reevaluated if the operator path is reorganized into a small number of
large regions.
"""
macro threaded(work, loop)
    (loop isa Expr && loop.head === :for) ||
        error("@threaded expects `@threaded <work> for ... end`")
    return esc(quote
        if $CompactLES._use_threads(Int($work))
            Threads.@threads $loop
        else
            $loop
        end
    end)
end
