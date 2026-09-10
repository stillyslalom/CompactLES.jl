# The parallel compact solve

A compact derivative is globally coupled along a grid line. Splitting that line
over MPI ranks cannot turn it into independent local derivatives without
changing the numerical operator. CompactLES instead solves the original global
banded system through a reduced interface problem.

## Decomposition

`Decomp` owns a three-dimensional Cartesian process grid. Each rank stores a
rectangular interior block and halo layers. For every resolved direction it
also constructs a one-dimensional subcommunicator containing the ranks along a
grid line in process space.

Collapsed dimensions have one global point, no halos, and process-grid extent
one.

## Local factorization and spikes

For the tridiagonal case, write the equations on one rank as

```math
T x = d-a_L x^-_n e_1-c_R x^+_1 e_n,
```

where ``T`` is the rank's local tridiagonal block, ``x`` its ``n`` unknowns,
``d`` the local right-hand side, and ``e_1``, ``e_n`` the first and last
standard basis vectors of length ``n``. The coefficients ``a_L`` and ``c_R``
couple the first and last local rows to their neighbors, and ``x^-_n`` and
``x^+_1`` are the two exterior interface values those neighbors supply: the
previous rank's last unknown and the next rank's first. After a one-time local
factorization,

```math
x = T^{-1}d-x^-_n T^{-1}(a_Le_1)-x^+_1T^{-1}(c_Re_n).
```

The latter two vectors are the left and right spikes. They depend on the
operator and local extent, not on the differentiated field, and are therefore
computed during planning.

Evaluating this expression at the first and last local points yields a dense
system in only two interface unknowns per rank. Its matrix is assembled and
factorized once. Each operator application then performs:

1. batched local banded solves for all grid lines;
2. one collective exchange of interface values;
3. a solve with the prefactorized reduced matrix; and
4. local spike corrections.

The pentadiagonal path generalizes the interface to the first and last two
values. It reproduces the serial compact solution exactly; the rank interface
is not approximated with an explicit stencil.

## Memory layout

The first array dimension is contiguous in Julia. The `x` solve naturally packs
lines in that order. The `y` and `z` paths use a transposed line matrix so fill,
elimination, spike correction, and scatter still traverse contiguous blocks.
This is a rank-local layout choice; the global field is never transposed among
ranks.

## Halo exchange

Explicit right-hand-side stencils still need neighbor values. Halo slabs are
exchanged one dimension at a time, with later slabs including the halos
filled in earlier dimensions. This populates edges and corners without separate
diagonal messages.

Physical-edge halos remain stale because one-sided closures do not read them.
Coordinate-fold halos are populated by parity or antipodal mappings instead.

## Collective discipline

Every rank in a directional subcommunicator must call an operator in identical
order. Rank-local branches may occur only after collective operator work is
complete. Violating this rule produces a communication hang, commonly with all
ranks at zero CPU utilization.

Collective diagnostics obey the same rule. It is correct for only rank zero to
print a result, but every rank must participate in the reduction that produces
it.

## Scaling implications

Local banded work decreases with decomposition, while the reduced interface
system and collective latency grow with ranks along a direction. Process grids
should therefore avoid decomposing a short dimension more finely than needed.
The nine-point local filter minimum is an algebraic constraint; useful strong
scaling generally stops before reaching it. The two subsections below give
the measurements behind the launch advice in [Run in parallel](@ref) and the
cost model behind the process-grid advice.

### Threads and ranks

The solver threads its point loops and line solves, so a rank may be given
several threads. At a fixed core count that has lost to one-thread ranks on
every machine measured, by about 2x:

| case | configuration | per step |
|---|---|---|
| 256³ Taylor--Green, two 112-core nodes | 224 ranks × 1 thread | 0.64 s |
| same | 112 ranks × 2 threads, verified two-core masks | 1.2 s |
| 768×48 two-species tube, eight-core workstation | 8 ranks × 1 thread | 12 ms |
| same | 1 rank × 8 threads, pinned to the eight cores | 24 ms |

The workstation figures are net of compilation and were taken with the
process confined to its performance cores; unpinned, the eight-thread run
was slower still and varied by 20% between launches.

The loss is structural rather than a tuning problem, and it is not memory
bandwidth at the workstation size, where the whole state fits in cache. A
right-hand-side evaluation enters some 120 to 200 threaded regions, one per
directional sweep of each field and one per pointwise pass, each with a
spawn-and-join floor of a few microseconds and a barrier at its end. On the
768×48 case those regions carry about 54 µs of serial work each, the floor
alone is about a quarter of the threaded evaluation, and each line solve is
three regions (fill, solve, scatter) between which the solved block migrates
between cores. A rank, by contrast, keeps its block in one core's cache
across every region of the step, and its only synchronization is the
collective inside each line solve. A threaded design reaches the same
locality only by giving each thread a fixed sub-block with shared-memory
halos, which is what MPI's on-node transport already does. A prototype that
fuses each line solve into one region ran 1.86x faster on that solve, which
narrows the gap without closing it.

Threads do have two openings. On a node whose rank count is set by its
accelerators, the interface stage of a device plan runs on the host, whose
remaining cores are otherwise idle. The other is one rank per NUMA domain
with pinned threads, the configuration in which the halo-exchange savings are largest
and the thread losses smallest; it has not been measured with clean masks.
Neither changes the default: launch with `-t 1` under `mpiexec`, and on a
workstation prefer `mpiexec -n 8 julia -t 1` to `julia -t 8` for anything
above 1-D. One-dimensional cases run below the threading threshold either
way.

### Many ranks along a direction

Per-step communication is not what limits rank counts. The Allgather of
interface values spans only the ranks along one direction, whose count grows
as the cube root of the total, and on four 112-core nodes the measured
scaling was 93% per node doubling. What grows is the reduced interface solve
itself. With ``P`` ranks along a direction and half-bandwidth ``q``, the
reduced matrix is dense of order ``2qP`` and every rank applies its
factorization to each of its own lines, at about ``8q^2P^2`` operations per
line, against roughly ``9qn`` for the local sweep and spike correction on
``n = N/P`` points. Counted per rank on a uniform three-dimensional process
grid, the local work shrinks as ``N^3/P^3`` while the reduced stage costs
about ``8q^2N^2`` per solve whatever ``P`` is, since the growth in the matrix
cancels the shrinking number of local lines. It is a fixed per-rank cost and
so an Amdahl term. The two are equal in operation count at

| ``N`` | ``q = 1`` (C6) | ``q = 2`` (C10) |
|---|---|---|
| 256 | ``P \approx 6.6`` | ``P \approx 5.2`` |
| 512 | 8.3 | 6.6 |
| 1024 | 10.4 | 8.3 |

which correspond to a few hundred to a thousand ranks in total. The reduced solve runs cache-resident at dense-linear-algebra rates
while the local sweep streams memory, so the wall-clock crossover is later,
by perhaps a factor of two to three in ``P``; it is nonetheless inside the
range of a production run. The reduced matrix is block-tridiagonal in ``P``,
since a rank's interface unknowns couple only to its two neighbors, and a
banded factorization brings the per-line cost to order ``q^2P`` and the
per-rank cost back to a shrinking ``N^2/P``. That change, and the
distributed reduced solve that removes the Allgather volume after it, are
planned work rather than current behavior.

The same count sets the process-grid rule. For a fixed total rank count the
per-rank reduced cost summed over the three directions is proportional to
the sum of the cubes of the per-direction rank counts, which is smallest for
a grid as close to uniform as the extents allow and largest for a slab
decomposition. Halo traffic prefers the same shape.
