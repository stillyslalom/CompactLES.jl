# Validation

The numerical gate is layered so failures point toward a specific part of the
implementation.

## Serial suite

```sh
julia --project=. test/runtests.jl
```

This covers compact operators, boundary closures, transposed sweeps, filters,
coordinate folds, freestream preservation, conservation, EOS round-trips,
boundary conditions, checkpoint/VTK I/O, and short end-to-end configurations.

## Convergence guards

```sh
julia --project=. test/convergence.jl
```

The suite measures observed order rather than checking only a tolerance at one
resolution. Current regression targets are approximately:

| Case | Observed order |
|:--|--:|
| periodic C6 derivative | 6.01 |
| periodic C10 derivative | 10.04 |
| C6 closed-wall closures | 3.17 |
| cylindrical-axis odd/even folds | 3.71 / 3.00 |
| resolved-angle axis fold | 3.71 |
| spherical origin | 2.99 |

For a change that is not intended to alter the numerics, compare the printed
error magnitudes as well as the fitted order. A change in the reported digits
can identify a discretization change before an order guard fails.

## Distributed suite

```sh
mpiexec -n 2 julia --project=. test/mpi_tests.jl
mpiexec -n 4 julia --project=. test/mpi_tests.jl
mpiexec -n 8 julia --project=. test/mpi_tests.jl
```

These runs exercise distributed tridiagonal and pentadiagonal solves,
cross-rank halo exchange, off-rank folds, the discrete geometric conservation
law, and telescoping flux conservation. Running several rank counts changes
the ownership and interface patterns and is therefore materially stronger than
one MPI smoke test.

CI runs the serial suite on the minimum supported Julia, current Julia, and
pre-release Julia. A separate job gates the convergence suite and all three MPI
rank counts on current Julia. The documentation build is also a required job.

## Performance diagnostics

The scripts in `bench/` are comparison tools rather than pass/fail tests.
`bench/jetcheck.jl` and `bench/audit.jl` report inference and allocation sites;
compare their counts before and after a performance-sensitive change. Do not
sum overlapping entry-point counts.
