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
Minimum total work (in grid points) before a loop is worth handing to the
thread pool. Tunable at runtime; the default was chosen from the table above,
where 32^3 is roughly break-even and 48^3 is a clear win.
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

## Why Threads.@threads and not Polyester.@batch

`@batch` was tried here and is a large net loss for this solver, despite doing
exactly what it advertises on allocation (compute_rhs! at 48^3 / 24 threads:
3,638,096 B -> 37,024 B, ~98x less). Measured on tgv 64^3 at -t 8:

    phase                Threads.@threads   Polyester.@batch
    primitives!                 1.11 ms          0.43 ms
    assemble_fluxes!            1.79 ms          1.84 ms
    velocity grads              3.95 ms         31.78 ms
    scalar grads                2.72 ms         24.12 ms
    flux divergence             7.00 ms         60.30 ms
    compute_rhs! (total)       22.26 ms        133.29 ms

`@batch` wins on the two phases that are a single large leaf loop over grid
points, and loses by ~8-9x on every phase that drives the compact operators.
Those phases are not nested — `apply_along!` runs its gather, solve and scatter
regions sequentially — they simply run MANY small regions back to back (about
36 per gradient phase), and `@batch`'s pool synchronisation per region costs
more there than a task spawn does.

Mixing the two backends by call depth is worse still: routing the operator path
to tasks while leaf loops stayed on `@batch` measured 1279.95 ms, 58x the
all-tasks figure, because Polyester's spin-waiting workers occupy the very
Julia threads `Threads.@threads` then tries to schedule onto.

So: one backend, and it is tasks. The threshold above is what keeps the task
cost off the small cases. Revisit `@batch` only if the operator path is ever
restructured into few large regions instead of many small ones.
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
