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
values. It reproduces the serial compact solution rather than approximating the
rank interface with an explicit stencil.

## Memory layout

The first array dimension is contiguous in Julia. The `x` solve naturally packs
lines in that order. The `y` and `z` paths use a transposed line matrix so fill,
elimination, spike correction, and scatter still traverse contiguous blocks.
This is a rank-local layout choice; the global field is never transposed among
ranks.

## Halo exchange

Explicit right-hand-side stencils still need neighbor values. Halo slabs are
exchanged one dimension at a time, with later slabs including halos already
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
scaling generally stops before reaching it.
