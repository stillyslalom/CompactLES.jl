# Parallel compact solve

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

Evaluating this expression at the first and last local points yields a system
in only two interface unknowns per rank. A rank's interface unknowns couple
only to those of its two neighbors, so the matrix is block-tridiagonal across
the ranks, with a block in each corner on a periodic line. Every rank assembles
the whole matrix and factorizes it once as a band matrix with partial pivoting,
the periodic ranks taken in the order 0, P−1, 1, P−2, … so that the corner
blocks fall inside the band. Each operator application then performs:

1. batched local banded solves for all grid lines;
2. one collective exchange of interface values;
3. a solve with the prefactorized reduced matrix; and
4. local spike corrections.

The pentadiagonal path generalizes the interface to the first and last two
values. It solves the same algebraic system as the serial compact operator,
with floating-point agreement to roundoff; the rank interface is not
approximated with an explicit stencil.

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
reduced matrix has order ``2qP``, and every rank applies its band
factorization to each of its own lines. Without row interchanges, which the
well-conditioned operators do not need, that costs about ``2n_r(k_l+k_u+1)``
operations per line for ``n_r = 2qP`` unknowns and half-bandwidths ``k_l``,
``k_u``: ``20P`` and ``36P`` for ``q = 1`` on a closed and a periodic line,
``88P`` and ``152P`` for ``q = 2``. The local sweep and spike correction cost
roughly ``9qn`` on ``n = N/P`` points. Counted per rank on a uniform
three-dimensional process grid, the local work shrinks as ``N^3/P^3`` and the
reduced stage as ``N^2/P``, so the ratio of the two grows as ``P^2/N`` and
they are equal in operation count at

| ``N`` | ``q = 1`` closed | ``q = 1`` periodic | ``q = 2`` closed | ``q = 2`` periodic |
|---|---|---|---|---|
| 256 | ``P \approx 10.7`` | ``8.0`` | ``7.2`` | ``5.5`` |
| 512 | 15.2 | 11.3 | 10.2 | 7.8 |
| 1024 | 21.5 | 16.0 | 14.5 | 11.0 |

which correspond to a few hundred to ten thousand ranks in total. A dense factorization,
``8q^2P^2`` operations per line, would give a fixed per-rank cost
``8q^2N^2`` and crossovers of ``P \approx 6.6``, 8.3 and 10.4 for ``q = 1``
and 5.2, 6.6 and 8.3 for ``q = 2`` at the same extents. At the rank counts a
workstation reaches the band solve runs several times faster than the dense
one did, well ahead of the operation count, since it streams every line
through one short loop per band entry. The solve is still replicated: every
rank solves the whole reduced system for its own lines. Removing that
replication, by a distributed reduced solve or by dividing the lines among
the ranks of a direction, is planned work rather than current behavior.

The same count sets the process-grid rule. For a fixed total rank count the
per-rank reduced cost summed over the three directions is proportional to the
sum of the squares of the per-direction rank counts, which is smallest for a grid as close to
uniform as the extents allow and largest for a slab decomposition. Halo
traffic prefers the same shape.
