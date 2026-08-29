# Threading policy.
#
# `Threads.@threads` spawns one Task per thread per region. That is free when
# the loop body is large and ruinous when it is not, and this solver is full of
# small regions: one compute_rhs! evaluation runs ~120 of them (one per
# direction per velocity component, per species, per flux, plus the remaining
# pointwise passes; the fused scatters in operators.jl removed ~30), so at 24
# threads a single RHS call spawns ~2900 tasks and allocates ~110 kB *per
# thread*, independent of grid size.
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
# touch in total — NOT its trip count — so a loop over (j, k) carrying an inner
# `i` loop passes `nx * ny * nz`, not `ny * nz`.
#
# Total work alone is not sufficient, because only the loop `@threaded` wraps is
# divided: a nest threaded on its outermost index over a collapsed dimension has
# one trip and so cannot be divided at all, whatever its total work. Planar-2-D
# runs, `n_global = (nx, ny, 1)`, hit this on every pointwise loop in the RHS,
# running serially at any thread count while still paying the region-entry cost.
# The policy therefore has two parts:
#
#   1. Pointwise nests iterate their two outer indices through `outer_indices`,
#      one flattened `CartesianIndices` whose trip count is n2*n3. A collapsed
#      third dimension then still divides over the second.
#   2. `_use_threads` takes the trip count and refuses to spawn for a loop with
#      one trip. That covers the 1-D case, which no flattening can help.

"""
Minimum total work, in grid points, required to use the thread pool, stored as
the effective total for this session. The default is
[`THREAD_MIN_WORK_PER_THREAD`](@ref) `* Threads.nthreads()`, set when the
module initializes: the spawn/join floor of one threaded region grows with the
thread count (measured 0.6 µs at 1 thread to 14 µs at 24 on a 24-core
desktop), so the work needed to amortize it grows with it. A fixed total of
32768 was measured to leave 25-30% on the table for planar-2-D RHS evaluations
of 16384 points at 8 and 16 threads, while at 24 threads that same work is
better off serial — no constant total serves both. Override the total with
the `CL_THREAD_MIN_WORK` environment variable, which is read once, when the
module initializes.
"""
const THREAD_MIN_WORK = Ref(1 << 15)

"""
Per-thread work floor, in grid points per thread, from which the
[`THREAD_MIN_WORK`](@ref) default is built. The break-even sits near 1024
points per thread across the measured anchors (24-core desktop, minimum over
30 calls, several processes): threading 16384-point regions is 25-30% faster
at 8 and 16 threads (2048 and 1024 points per thread) and slightly slower at
24 (683 per thread); threading 4096-point regions at 16 threads (256 per
thread) is 1.8x slower; and the previous fixed total of 32768, tuned at 24
threads, is 1365 per thread — the same figure from an independent
measurement.
"""
const THREAD_MIN_WORK_PER_THREAD = 1024

function __init_threading__()
    THREAD_MIN_WORK[] = THREAD_MIN_WORK_PER_THREAD * Threads.nthreads()
    v = get(ENV, "CL_THREAD_MIN_WORK", nothing)
    if v !== nothing
        # Named, because this runs in `__init__`: a bare `parse` failure there
        # surfaces as an initialization error naming neither the variable nor
        # its value, and the environment form has no `script_args` to check it.
        n = tryparse(Int, v)
        n === nothing &&
            error("CL_THREAD_MIN_WORK must be an integer number of grid " *
                  "points, got \"$v\"")
        THREAD_MIN_WORK[] = n
    end
    return nothing
end

# Takes Int, not Integer: an abstract parameter type leaves the `>=` below as a
# runtime dispatch, once per threaded region (~150 per RHS call).
@inline _use_threads(work::Int, trips::Int) =
    Threads.nthreads() > 1 && trips > 1 && work >= THREAD_MIN_WORK[]

"""
    outer_indices(n2, n3)

Flattened iteration space for the two outer indices of a pointwise loop nest.
Both arguments may be a count or an index range, and the first varies fastest,
so

```julia
@threaded nx*ny*nz for jk in outer_indices(ny, nz)
    j, k = Tuple(jk)
    for i in 1:nx
```

visits the same points in the same order as the `for k in 1:nz`, `for j in 1:ny`
nest it replaces, while giving [`@threaded`](@ref) `ny*nz` trips to divide
instead of `nz`. The note at the top of `threading.jl` has the reasoning.
"""
@inline outer_indices(n2, n3) = CartesianIndices((n2, n3))

"""
    @threaded work for ... end

Thread the loop only when the session has more than one thread, `work` (total
grid points touched, converted with `Int`) reaches [`THREAD_MIN_WORK`](@ref),
*and* the loop has more than one trip to divide; otherwise run it inline. The
iteration expression is evaluated once either way. A pointwise nest should
iterate [`outer_indices`](@ref) so that the trip-count condition does not fail
on a collapsed dimension. See the note at the top of this file.

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
    spec = loop.args[1]
    (spec isa Expr && spec.head === :(=)) ||
        error("@threaded expects a single-variable `for` loop")
    var, iter, body = spec.args[1], spec.args[2], loop.args[2]
    # The iteration space is bound to a local first, so that reading its trip
    # count does not evaluate the expression a second time.
    rng = gensym(:threaded_range)
    return esc(quote
        local $rng = $iter
        if $CompactLES._use_threads(Int($work), length($rng))
            Threads.@threads for $var in $rng
                $body
            end
        else
            for $var in $rng
                $body
            end
        end
    end)
end
