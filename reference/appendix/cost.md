# Calibration appendix: Cost and output

This file holds the sections of the measurement record on the cost of the operators and
the step, the shared-file write and azimuthal mode truncation.
[CALIBRATION_APPENDIX.md](../CALIBRATION_APPENDIX.md) indexes the record and states its
conventions.

## Operator and step cost

```text
MPIEXEC=$(julia --project=. -e 'using MPI; MPI.mpiexec(c -> print(c))')
"$MPIEXEC" -n 8 julia --project=. -t 1 bench/derivcost.jl 128 30 dims=2,2,2
"$MPIEXEC" -n 8 julia --project=. -t 1 bench/derivcost.jl 128 10 dims=2,2,2 phases=true cases=periodic
julia --project=. -t 1 bench/phases.jl
julia --project=. -t 1 bench/archive/reducedsolve.jl 48 derivs=c6,c10
"$MPIEXEC" -n 2 julia --project=. -t 1 bench/archive/reducedsolve.jl 96 derivs=c6,c10
"$MPIEXEC" -n 8 julia --project=. -t 1 bench/archive/reducedsolve.jl 96 dims=8,1,1   # also 2, 4
"$MPIEXEC" -n 16 julia --project=. -t 1 bench/archive/reducedsolve.jl 192 dims=16,1,1
"$MPIEXEC" -n 8 julia --project=. -t 1 bench/archive/reducedsolve.jl 96 mode=accuracy
"$MPIEXEC" -n 8 julia --project=. -t 1 bench/archive/reducedsolve.jl 96,192 mode=accuracy operators=filter,pyranda rhs=random
```

Run-to-run spread on this workstation is 10–20%, so every ratio below is formed within a
process and every absolute figure is a median of three.

### The derivative operators

In the production configuration C8 costs a few percent of a step over C6, and C10 16%. At
32³ per rank every operator's cost per point rises 13% from the communication share. The C8
figure stays where it is there and the C10 figure rises to about 20%. The single-rank
single-thread ratios, 10% and 29%, are the undiluted arithmetic. The derivative solves are a
fifth to a quarter of the right-hand side, the artificial-property pass and the filter are
flat across the operators to within 3%, and a decomposed line solve adds a
reduced-interface stage and a halo exchange per direction that no operator changes. That
stage dilutes C8's extra multiply-adds more than C10's extra band, since the pentadiagonal
interface stage is itself wider.

The table gives the wall-clock cost per grid point per step of `lele_d1_6`, `lele_d1_8` and
`lele_d1_10` on a single-species ideal-gas box, the Taylor–Green field as the initial state,
the default filter every step, `ArtificialProperties()` defaults, in double precision. Each
cell is warmed over three steps and timed over thirty through `solver.wall_total`, spanning
`max_rate`, `apply_bcs!`, the stages and the filter pass, reduced as the maximum over ranks
after a barrier; a fourth process with the operators in reverse order reproduces the forward
ratios inside the spread. The production configuration is eight ranks at one thread each.

```
cfl 0.5, Float64; 12th-gen Core i9-12900K (8 performance and 8 efficiency cores, 24 threads),
Microsoft MPI 10.1 from the JLL

                                            C6        C8        C10      C8/C6          C10/C6
128^3 on 8 ranks (2,2,2), periodic, -t 1   444.1     461.5     517.3    1.035–1.054    1.157–1.170
128^3 on 8 ranks, slip walls in x          447.3     459.2     518.1    1.010–1.041    1.090–1.167
64^3 on 8 ranks (32^3 per rank), periodic  503.3     520.7     604.8    0.979–1.087    1.202–1.231
64^3 on 8 ranks, slip walls in x           500.4     531.3     608.5    1.042–1.065    1.203–1.239
64^3 on 1 rank, -t 1, periodic            1353.5    1486.1    1748.8    1.072–1.106    1.266–1.308
64^3 on 1 rank, -t 16, periodic            402.6     416.9     498.5    1.030–1.042    1.220–1.251
64^3 on 1 rank, -t 16, slip walls in x     371.3     385.1     426.6    1.034–1.039    1.107–1.199
```

ns per point per step, global points over the slowest rank's wall; the ratio columns span
the processes. The phases of one right-hand-side evaluation at 128³ on eight ranks,
periodic, the maximum over ranks of each rank's minimum over repeated calls on a settled
state, ns per point (the first four inside `compute_rhs!`, `filter_state!` outside it):

```
                           8 ranks, -t 1                 1 rank, -t 16
                            C6      C8      C10          C6      C8      C10
velocity gradients         9.38   10.86   13.20         8.69    9.87   14.01
scalar gradients           5.23    6.28    8.01         6.03    6.60    9.28
artificial properties     18.77   20.14   17.63        16.06   16.27   15.55
assemble_fluxes!           7.32    7.32    6.86         8.42    9.19    9.28
compute_rhs! (whole)      73.64   82.75   90.26        71.32   73.27   92.20
filter_state!             24.35   21.98   21.63        19.19   18.25   19.31
```

The closed configuration costs the same as the periodic one on eight ranks and less on one,
where the closed line solve carries no cyclic correction. The pentadiagonal plans add under
half a percent to the per-rank footprint, and the allocation per step is the same across
the operators.

```
at 64³ per rank: plans +1.1 MiB on a 234 MiB footprint; allocation 7.7 KiB per rank per step
```

Rank placement is not controllable with this launcher. Microsoft MPI's `-affinity` and
`-affinity_layout seq:P` options have no effect through the JLL `mpiexec`, which runs
without the `smpd` service and sets no affinity mask: sampled over two seconds of work,
every rank migrated over the whole machine including the efficiency cores, and two ranks
were seen on one logical CPU at the same instant. The process-to-process spread is the size
of the whole C8 effect, so the C8 figure is quoted as a few percent. Pinning would take a
`SetProcessAffinityMask` call inside the run, not a launcher flag.

```
process-to-process spread
64³ per rank     2.7–4.2% on five of six cells (one 15% outlier)
32³ per rank     19–30% in the periodic cells
```

### The x-direction pentadiagonal solve

`bench/archive/reducedsolve.jl` (`48 derivs=c6,c10` serially, `96` at np = 2, `-t 1`, fully
periodic) splits a line solve into the local sweep, the reduced interface stage, and the spike
correction (total − sweep − reduced). On the x layout, where a line is a column of the n ×
lines block, the cost of the C10 spike correction is set by whether its interface values
stay in registers. Its point loop is an inner sum over the q coupling columns. With a
runtime trip count the loop reloads the interface values for every point, since the
compiler cannot rule out aliasing between them and the stores into the block, and
`code_llvm` shows no vector instruction; the back substitution of the band sweep reloads
its U coefficients once per column for the same reason. The solver takes q = 2 as a
compile-time constant and reads those values once per line and once per row, which gives
bitwise-identical results, a correction an order of magnitude cheaper and half the whole
x solve. Times in µs:

```
                                sweep   correction    total
48^3, 1 rank, x  runtime q      260.4      251.2      544.0
                 constant q     150.0       37.6      222.8
48^3, 1 rank, y                  43.0       37.5      114.7
96^3, 2 ranks, x runtime q     1237.4     1084.2     2637.4
(P = 2)          constant q     905.6       68.5     1286.5
96^3, 2 ranks, y (P = 1)        383.2      126.1      639.0
```

Timed in one process against a copy of the runtime-q loops, the whole x solve of the C10
derivative and the `pyranda_filter` plans is about two to three times faster on a periodic
line and less on a closed one, which carries no correction, and the C10/C6 step ratio falls
by about a tenth. The remaining gap is the x sweep, which reads a column block with a stride
of n. It is 3.5x the y sweep, where C6's is 2.8x.

```
one process, serial at 48³ and 96³, constant q against runtime q
whole x solve, periodic line       2.3–2.9x faster
whole x solve, closed line         1.6–1.8x faster
bench/derivcost.jl 64 20 cases=periodic, one rank, -t 1, one process each
C10/C6 step ratio                  1.29 runtime q, 1.19 constant q
```

A scratch prototype that copies 64 lines into a lines × n buffer, sweeps it as the
transposed path does, and copies back was bitwise equal and 1.4x faster than the hoisted
sweep; it needs a scratch buffer per task and was not adopted.

### The decomposed line solve against serial

A decomposed solve of a system of condition number κ reproduces serial to between a
twentieth and a half of κ·eps, and 3e-15 is that bar only for the operators with cond(A)
near 20. The departure from serial is the same fraction of cond(A)·eps for every operator,
including `pyranda_filter`, whose left-hand-side symbol nearly vanishes near k = π and
sets a condition number two orders above the others'. `bench/archive/reducedsolve.jl
mode=accuracy` applies the x line solver once with the line split over P ranks, against the
P = 1 solve and against a 256-bit dense solve, max-norm relative to the largest exact value.

```
np = 8, 96 points, 64 smooth right-hand sides per operator
departure / (cond(A)·eps)   0.13–0.44 for the four well-conditioned operators,
                            0.13–0.21 for pyranda_filter
pyranda_filter symbol       1 + 2α cos k + 2β cos 2k, falling to 1.3e-3 near k = 0.98π

                     cond(A)  cond(red)    serial error   departure, P = 2..8   pivoted dense LU
c6                     5.0      2.24        2.4e-16        3.7e-16 – 4.9e-16       2.4e-16
c10                   21.0      5.51        8.4e-16        9.2e-16 – 1.4e-15       1.1e-15
compact_filter()      19.0      4.36        4.2e-16        5.6e-16 – 1.0e-15       1.0e-15
compact_d8()          15.0      4.49        6.8e-16        8.3e-16 – 1.5e-15       8.3e-16
pyranda_filter()    2087       394–440      4.8e-14        5.9e-14 – 9.5e-14       1.0e-13
```

The departure does not grow with P, the reduced matrix is better conditioned than A at
every P, and the local block's condition number falls with n, so neither the interface
stage nor the unpivoted local factorization amplifies round-off. For `pyranda_filter` the
serial solve and a pivoted dense LU of the same system sit as far from the exact solution as
the decomposed solve sits from serial, so the decomposed solve departs from serial by the
error any double-precision solve of this system carries. The per-solve ratio between the two
filters is of the order of the ratio between the decomposed runs' departures from serial in
the MPI suite's two-level wave rows under the two filters, so those departures are this
floor accumulated over the steps, not a defect of the distributed solve.

```
local block, pyranda_filter              cond 2083 at P = 1, 766 at P = 8
uniform-noise right-hand sides, 96 and 192 points
  pyranda_filter                         2.4e-14 – 5.2e-14 (0.05–0.11 cond(A)·eps)
  default filter                         1.1e-15 – 1.8e-15
pyranda_filter against the default filter
  per solve, smooth data                 60–170 times
  MPI two-level wave rows                1e-13 to 1.4e-12 against 7e-15
```

The table above is the band solve of the reduced matrix. A dense LU of the reduced matrix,
measured the same way, gives departures within the same range and the same maximum over P.
On the real reduced matrices of the four well-conditioned operators the pivoted band LU
interchanges no rows; on `pyranda_filter()`'s it interchanges some.

```
dense LU against band, largest change   pyranda_filter() at P = 6, 7.1e-14 to 9.5e-14
                                        (0.153 to 0.206 cond(A)·eps); maximum over P 9.5e-14
rows interchanged, pyranda_filter()     8–46 of 2qP at P = 4–16
```

The same instrument in `mode=timing`, x direction of a fully periodic solver split P ways
(`dims=P,1,1`), `-t 1`, µs, the maximum over ranks of each rank's median; "reduced" is
`_reduced_solve!` whole (Allgather and solve), "solve" everything after the Allgather, the
dense column the `ldiv!` alone. Dense and band are two processes. The band solve is
2.3–5.6x faster than the dense triangular solves at every P measured. At P = 2 the band
covers nearly the whole matrix, and the gain is the loop structure, each band entry one
contiguous pass over a block of 128 lines, rather than the operation count.

```
                 lines    dense     band             reduced          local sweep
                          ldiv      solve       dense    band      dense / band process
C6,  P = 2,  96³  9216    102.8      18.2       141.4     36.0          646 / 936
C10, P = 2,  96³  9216    220.8      63.1       316.0    100.1         1116 / 922
C6,  P = 4,  96³  9216    231.8      50.3       511.2    116.1          296 / 357
C10, P = 4,  96³  9216    517.2     299.2       766.4    427.4          637 / 823
C6,  P = 8,  96³  9216    925.6     207.6      1589.9    379.7          263 / 287
C10, P = 8,  96³  9216   2789.6     919.9      2989.4   1349.4         1072 / 650
C6,  P = 16, 192³ 36864  19868    6149        45356    10131           4915 / 4257
C10, P = 16, 192³ 36864  41601   12093        73127    21146           6708 / 5650
```

Operations per line of the band solve, counted as the table's `red` column (twice the
multiply-adds of the forward sweep and the back substitution down to the first unknown the
rank reads, maximum over ranks), are linear in P where the dense count grows as P², and
their ratio to P approaches a constant as the fixed end terms dilute. At P = 16 the ranks
fill the workstation's sixteen cores, eight of them efficiency cores, and the Allgather
rises to milliseconds, so that row measures the machine as much as the solve. A serial line
keeps the dense 2q × 2q solve. The crossover against the local work is a cluster
measurement and remains open.

```
operations per line     P = 4    8      16
C6, band                104      248    536
C6, dense               128      512    2048
C10, band               428      1036   2252
C10, dense              512      2048   8192
band / P on a periodic line: toward 34 (q = 1) and 141 (q = 2)
Allgather in the last two rows of the timing table: 4.5–9.1 ms
```

### The sensor phase

The velocity sensor and the `:d8` detector each make the sensor phase dearer, and together
they make it costlier than the rest of the evaluation combined. The default `:gaussian`
smoother makes it cheaper than `:compact`. `bench/phases.jl` runs the two-species tube,
back to back on one machine. The `artificial` phase is `compute_artificial!`, and
`line solves` counts the directional solves per right-hand-side evaluation as (filter and
gradient) + (`:d8` detector):

```
setting                          artificial   % of RHS   compute_rhs!   line solves
strain / strain / delta4*         1.050 ms     26.0%       4.178 ms      24 + 0
mu_sensor = velocity              1.535 ms     33.7%       4.690 ms      24 + 0
beta_sensor = ungated_dilatation  1.287 ms     29.8%       4.349 ms      24 + 0
detector = d8                     1.733 ms     35.9%       4.798 ms      24 + 8
mu_sensor = velocity, d8          2.678 ms     47.1%       5.775 ms      24 + 14
```

The velocity sensor detects three fields where the strain sensor detects one, which is +46%
on the sensor phase under δ⁴. Paired with `:d8` it adds six more pentadiagonal solves per
right-hand side, and the sensor phase then costs more than everything else in the evaluation
combined. `:d8` alone is eight pentadiagonal solves per right-hand side, one per active
dimension per sensor, for +80% on the sensor phase and +19% on the whole evaluation in a
separate back-to-back pair. Against a 10–20% run-to-run spread the phase figure is resolved
and the total is marginal.

```
separate pair, artificial phase    without :d8: 0.946 ms, 23.5% of the RHS
                                   with :d8:    1.707 ms, 34.6%
```

The smoother moves the same phase the other way: from `:compact` to the default `:gaussian`
the `artificial` phase falls by more than a quarter, against a run-to-run spread of about
1.3% over three `:compact` readings.

```
artificial phase      :compact 1.360 ms (31.8% of the RHS), :gaussian 0.971 ms (24.8%)
```

The Gaussian is an explicit nine-point stencil with no line solve and no interface reduction,
where the compact pass costs a distributed line solve per active dimension per sensor,
`n_species` of them per evaluation for the species sensors.

`compute_artificial!` is therefore 24.8–26.0% of the multicomponent right-hand side under
the default smoother, most of it in the smoothing of the sensors, one sweep per species. At
`n_species == 2` that per-species machinery is a measurable no-op, and it earns its cost
only at three or more species; cutting it is a numerics decision, a shared against a
per-species sensor, and not a code tweak.

### bench/archive/nasa9_inversion.jl: the NASA-9 temperature inversion

Under `:linear` the sweep finds no failure in either precision. Every remaining failure is
the four-species mixture above 6000 K under `:polynomial`, where the H2O fit ends and its
extrapolated energy is not monotone: states with no bracket, and silent ones whose energy
has a second root inside the fitted range, which the inversion returns. `:linear` is
therefore the default of `Nasa9Mixture`, and `:polynomial` an opt-in.

The first command below measures the cost and the second the failures, in both precisions
under `:polynomial` and `:linear`. "Wrong" is a recovery more than 1e-3 from the temperature
the energy was formed at; "silent" is a wrong one whose status carries no flag.

```
julia --project=. -t 1 bench/archive/nasa9_inversion.jl
    N2/O2/CO2/H2O, 300–3000 K, three processes per checkout, alternating
... 200000 part=sweep
    200,000 states per case over 200–20,000 K, for N2, CO2, He and N2/O2/CO2/H2O,
    in Float64 and Float32
```

| inversion | not converged | no bracket | wrong | silent |
|---|---|---|---|---|
| iterates free to leave the fitted range, 32 eps | 387,465 | 335,896 | 362,331 | — |
| iterates held on the fitted range's edge, 32 eps | 117,757 | 61,215 | 150,998 | 20,536 |
| the same, criterion eps^(2/3) floored at 1e-10 | 61,460 | 61,211 | 150,999 | 20,537 |

| measurement, four species at 300–3000 K | 32 eps | eps^(2/3) |
|---|---|---|
| Newton iterations per point | 4.744 | 3.934 |
| `mixture_temperature`, ns per point | 187.4–189.8 | 157.4–160.1 |
| `recover_primitives!`, ns per point, 32³ | 235.4–237.6 | 197.4–198.4 |

Holding the iterate on the edge of the fitted range removes the failures of pure CO2 at
high temperature, where the seed lands past the fitted range and the fit's cv is negative.
The 32 eps criterion fails on pure N2 at intermediate temperatures although the recovered
temperature is accurate, its error set by the residual's round-off and by the fits'
continuity at the 6000 K join. The criterion eps^(2/3) is met on all of those states.

| case | states | result |
|---|---|---|
| pure CO2 at 13–20 kK, `:polynomial`, iterates free | 70,708 | 68,231 unbracketed and wrong in either precision; the seed, linearized about 300 K, lands past 20,000 K |
| the same, iterates held on the edge | the same | all recovered, to 5.2e-14 in Float64 |
| pure N2 at 6–13 kK, 32 eps | 70,706 | unmet on 5,967 (Float64) and 12,172 (Float32); largest error 1.6e-13 and 7.7e-5 |
| the same in Float32 at 100,000 states, 32 eps, eps^(2/3) and 1e-4 | | 6,021, 0 and 0 of those misses |

## The shared-file write

```text
"$MPIEXEC" -n 4 julia --project=<env with HDF5> -t 1 bench/archive/hdf5xfer.jl 64 5
"$MPIEXEC" -n 4 julia --project=<env with HDF5> -t 1 bench/archive/hdf5xfer.jl 128 5
"$MPIEXEC" -n 8 julia --project=<env with HDF5> -t 1 bench/archive/hdf5xfer.jl 128 5
```

The MPI transfer mode of the block datasets, `:collective` (the default) against
`:independent`, on a parallel libhdf5 on the workstation, writing to the WSL virtual disk.
Each cell is the maximum over ranks of the wall time of the whole collective call, file
creation and close included, the median and the minimum of five after one untimed write.
The checkpoint is the `Float64` state (5 N³ values), the frame the `Float32` density and
velocity (4 N³), and the slice one plane across the split dimension, which one rank holds
entirely while the others issue empty-selection writes.

```
conda-forge hdf5 2.2.0 over OpenMPI 5.0.11 under WSL2, 12th-gen Core i9-12900K

                                   independent          collective       collective /
                                   median    min        median    min    independent
64^3,  4 ranks (2,2,1) checkpoint  0.0273  0.0263       0.0201  0.0195   0.74
                       frame       0.0126  0.0121       0.0100  0.0095   0.79
                       slice       0.0024  0.0021       0.0022  0.0020   0.92
128^3, 4 ranks (2,2,1) checkpoint  0.2300  0.1261       0.0870  0.0761   0.38
                       frame       0.0656  0.0604       0.0490  0.0468   0.75
                       slice       0.0093  0.0092       0.0092  0.0092   0.99
128^3, 8 ranks (2,2,2) checkpoint  0.1366  0.1226       0.0985  0.0826   0.72
                       frame       0.0652  0.0618       0.0457  0.0401   0.70
                       slice       0.0099  0.0091       0.0099  0.0088   1.00
```

The collective mode is faster wherever more than one rank holds data and equal on the
slice, so it is the default. Seconds per write; the ratio column is of medians. The 80 MiB
checkpoint completes in a tenth of a second, which is the page cache and not the disk, so
the ratios measure the aggregation of the per-rank hyperslabs against uncoordinated writes
into the kernel and say nothing about a parallel filesystem at high rank counts; the same
script gives that comparison from a cluster environment. The independent-mode cost on
rzhound's Lustre is in `reference/CLUSTER.md`, and either mode is affordable there.

`test/hdf5_tests.jl` passes on this stack at one, two and four ranks under both modes,
including the sliced plane and the tiled hierarchy checkpoint, whose ranks without a tile
issue the empty write; the same file passes on the Windows HDF5_jll, whose MPI build over
Microsoft MPI also reports `hdf5_parallel() == true`, at one and two ranks, and on a
conda-forge `nompi` libhdf5, which selects the serialized relay, at one and two ranks.

## Azimuthal mode truncation

```text
julia --project=. -t 1 bench/archive/axisvortex.jl
julia --project=. -t 1 bench/archive/axisvortex.jl N=32,64 configs=1:on:0.5,1:off:0.5,0:on:0.5
```

The isentropic vortex of the Axis-crossing vortex tutorial is carried straight through the
axis of the unit disk, with θ over 2π and z collapsed, a `DirichletBC` at r = 1 holding the
exact translation, and the default numerics but for `polar_truncation`, the artificial
properties and the CFL number. The errors are the largest |ρ − exact| and |u − exact| at any
node over samples every 0.05 time units to the end; on every truncated run the largest
difference lies on the first two rings. Measured at `-t 1` on the workstation.

```text
vortex    β = 3, core radius 0.15, U = 0.5, from x = −0.4 at t = 0 to x = 0.4 at t = 1.6
grid      48 × 96 nodes
```

### The mode-count rule

Three rules are compared, each with one limit table:

- the original, mode_limit = max(1, ⌊πr/(κΔr)⌋) for every component, with the θ rate capped
  at that mode;
- the momenta's limit raised by one and the cap at the momenta's mode, since u_r and u_θ
  carry Cartesian mode m ± 1 in their mode m, so their mode m is O(r^(m−1));
- that rule with the floor raised to 2, which the package takes.

Under each, `failed` is a `SolverFailure` on positivity.

```text
rule                 κ  art  CFL  ends at   steps   max ρ err   max u err
one limit, floor 1   1  on   0.3  t = 0.29    155     1.12e-2     3.75e-2
                     1  on   0.5  t = 0.21     64     9.56e-3     2.73e-2
                     1  off  0.3  1.6         756     2.13e-2     5.46e-2
                     1  off  0.5  1.6         465     1.89e-2     4.96e-2
                     2  on   0.3  t = 0.34    148     1.57e-2     6.44e-2
                     2  on   0.5  t = 0.29     84     1.27e-2     4.97e-2
                     2  off  0.3  1.6         596     2.20e-2     7.79e-2
                     2  off  0.5  1.6         370     1.88e-2     6.95e-2
momenta +1, floor 1  1  on   0.3  1.6         768     1.04e-3     2.20e-3
                     1  on   0.5  1.6         468     9.07e-4     1.84e-3
                     1  off  0.3  1.6         766     1.09e-3     2.35e-3
                     1  off  0.5  1.6         468     9.24e-4     1.92e-3
                     2  on   0.3  1.6         762     2.07e-3     3.86e-3
                     2  on   0.5  1.6         465     1.82e-3     3.31e-3
                     2  off  0.3  1.6         744     2.18e-3     4.10e-3
                     2  off  0.5  1.6         456     1.83e-3     3.43e-3
momenta +1, floor 2  1  on   0.3  1.6         988     7.41e-5     1.03e-4
                     1  on   0.5  1.6         601     4.60e-5     6.30e-5
                     1  off  0.3  1.6         985     7.59e-5     1.05e-4
                     1  off  0.5  1.6         601     4.62e-5     6.38e-5
                     2  on   0.3  1.6        1047     1.84e-3     4.16e-3
                     2  on   0.5  1.6         626     1.57e-3     3.28e-3
                     2  off  0.3  1.6         986     1.91e-3     4.47e-3
                     2  off  0.5  1.6         601     1.56e-3     3.36e-3
untruncated          0  on   0.5  1.6        8292     3.45e-5     1.49e-4
                     0  off  0.5  1.6        8264     3.42e-5     1.50e-4
```

Under one limit the first ring (limit 1) deletes the m = 2 content of u_r and u_θ, which the
vortex's strain carries at O(r), and of every scalar, which its curvature carries at O(r²).
Without the artificial properties the run completes with the density error 550 times the
untruncated run's; with them it loses positivity. The extra momentum mode alone makes every
run complete, with or without the artificial properties, and leaves the density error 26 times
the untruncated run's. That remainder is the scalars' m = 2 content on the first ring. A run
with every limit raised by one at κ = 1 reaches the error of the floor of 2 in as many
steps.

The floor of 2 brings the lower κ within a factor 1.33 of the untruncated error in a fourteenth
of the steps. The θ rate of the first ring then exceeds the radial one, and the step is 22%
shorter than at floor 1. Raising the momenta by a second mode instead changes the density
error by under 7%. At κ = 2 the second ring also has limit 2. Its O(r³) deletion holds the
error at the same level under every rule with the momenta raised, and raising every limit
by one removes most of it.

```text
rule                          κ   max ρ err
every limit raised by one     1   4.52e-5 in 601 steps
every limit raised by one     2   2.21e-4
momenta +2, floor 1           1   within 7% of momenta +1, floor 1
```

The deleted content is regenerated by the flow and removed again every step, so the error of a
truncated run grows with the number of projections. Under the package's rule at the lower κ,
the smaller CFL number takes 1.64 times the steps and has 1.61 times the density error. Under
the same rule the artificial properties leave the density error at or below its value
without them in every row but one, the higher κ at the larger CFL number, where it is 0.6%
above in 4% more steps.

### Cost and the coarser grid

One `truncate_modes!` call costs a few percent of a step, in proportion to the number of
active rings:

```text
48 × 96, artificial properties on, step of 13 ms
κ   active rings   one truncate_modes! call   share of a step
1   15             409 µs                     3%
2   30             813 µs                     6%
```

On the tutorial's coarser grid:

```text
32 × 64, CFL 0.5
κ  art  steps   max ρ err   max u err
1  on     402     1.80e-4     3.34e-4
1  off    402     1.83e-4     3.41e-4
0  on    3747     3.13e-4     1.34e-3
```

The truncated run is the more accurate there: the untruncated error is set by the azimuthal
resolution of the core away from the axis, which the tutorial describes, and not by the axis.
