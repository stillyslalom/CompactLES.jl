# Calibration appendix: Boundaries, closures and folds

This file holds the sections of the measurement record on the accuracy of the boundary
closures and boundary conditions: the smooth-evolution matrix, the wall rows and their
certificates, the flux contracts, the symmetry plane, the inflow and outflow faces, and
the coordinate folds. [CALIBRATION_APPENDIX.md](../CALIBRATION_APPENDIX.md) indexes the
record and states its conventions.

## The smooth-evolution accuracy matrix

```text
julia --project=. -t 1 bench/boundaryorder.jl
julia --project=. -t 16 test/convergence.jl
```

The matrix records the order of the solution at a wall or an interface after a smooth
evolution, separated from the interior, with the time error controlled, unfiltered and
filtered. The cases are `test/smooth_cases.jl`, shared with `test/convergence.jl`, which
guards fifteen of the rows. Serial Float64, every evolution at `cfl = 0.25` with the
artificial properties off; the same walls with the properties on are in [wall closures in
production](#wall-closures-in-production).

**The cases and their references.** Every case is one-dimensional along x. The standing wave
`rho = 1 + 0.05 cos(πx)`, `u = 0.05 sin(πx)`, `p = rho^γ` on the unit interval between slip
walls, or between adiabatic no-slip walls with `mu0 = 0.005` (u vanishes and T is even at
each wall, so the data satisfy both conditions), integrated to t = 0.4, a third of an
acoustic transit. Its density and pressure are even and its velocity odd about both walls,
so it is the restriction of a periodic problem of twice the length, and a periodic run on
2(N − 1) nodes at the same spacing is the wall run without its closure rows. The difference
between the two, at the same step, is the closure defect alone, derivative and filter rows
together; the difference from a periodic run at four times the resolution is the total
error, and both are tabulated. The decaying shear mode `v = 0.1 sin(πx) exp(−μπ²t)` at
uniform rho and p between no-slip walls is exact once a source removes the viscous heating
`μ v_x²`; its tangential stress `μ v_x` is nonzero at the wall, so the wall rows
differentiate a nontrivial flux. The entropy wave `rho = 1 + 0.2 sin(k(x − u₀t) + φ)` on the
periodic [0, 2π), the wave of the patch and level tests, is exact. The viscous standing
wave, periodic over twice the wall problem's length, has its fine reference on 1728 nodes,
which nests every level's nodes.
Refinement regions are fixed in physical space at every N.

Errors are maximum norms over regions: the wall window (the first and last four nodes of a
physical boundary), the interface window (the same at a patch or level end), the covered
parent nodes under a child level, and the interior; `l2` is the composite volume-weighted
root-mean-square through the package's masked quadrature, covered parents excluded. Orders
are between successive resolutions against the actual spacing. The `dt` column is the
relative change of the windowed error when the step is halved, below 0.01 on every row
cited as an order unless said otherwise.

### Closure truncation and the instantaneous right-hand side

Every closure's rows converge at their formal order on one derivative of x^(q+1), q the
rows' exactness degree, and the interior converges at the same rate, the wall defect carried
inward by the compact solve. Wall window:

```
                     N=17       33         65         129        orders
C6 cascade3, x^4     1.092e-3   1.365e-4   1.706e-5   2.132e-6   3.00 / 3.00 / 3.00
C6 cascade4, x^5     2.397e-4   1.498e-5   9.363e-7   5.852e-8   4.00 / 4.00 / 4.00
C6 BL, x^6           1.540e-4   4.812e-6   1.504e-7   4.700e-9   5.00 / 5.00 / 5.00
C8 BL, x^8           2.872e-4   2.252e-6   1.759e-8   1.382e-10  6.99 / 7.00 / 6.99
```

The `exp(sin(3x))` slopes of the convergence suite sit above these by the field's phase and
are regression guards, not the rows' order.

The instantaneous error of the assembled Navier–Stokes right-hand side on the exact initial
data, density component, wall window / interior:

```
                          N=49                    97                      193                     wall orders
inviscid wall, cascade3   9.848e-7 / 1.779e-8     6.166e-8 / 1.116e-9     3.856e-9 / 6.978e-11    4.00 / 4.00
inviscid wall, cascade4   6.724e-7 / 7.731e-9     4.234e-8 / 4.863e-10    2.651e-9 / 3.044e-11    3.99 / 4.00
inviscid wall, C6 BL      3.588e-8 / 3.878e-10    5.702e-10 / 6.134e-12   8.954e-12 / 9.606e-14   5.98 / 5.99
inviscid wall, C8 BL      5.441e-9 / 3.199e-10    2.211e-11 / 1.299e-12   1.356e-13 / 5.829e-15   7.94 / 7.35
shear mode (rho v), cascade3   6.217e-7 / 3.165e-8   7.775e-8 / 3.961e-9   9.719e-9 / 4.953e-10  3.00 / 3.00
shear mode (rho v), cascade4   6.383e-7 / 1.081e-8   7.996e-8 / 1.357e-9   1.000e-8 / 1.699e-10  3.00 / 3.00
shear mode (rho v), C6 BL      2.896e-9 / 5.562e-11  9.098e-11 / 1.743e-12 2.964e-12 / 5.344e-14 4.99 / 4.94
shear mode (rho v), C8 BL      8.903e-10 / 1.889e-11 7.597e-12 / 1.624e-13 2.286e-12 / 5.300e-14 6.87 / roundoff
```

The viscous wall reads the same density rows as the inviscid one, the mass flux having no
viscous term, and its energy rows two orders lower at the wall: the conductive flux is a
second derivative of an even temperature, so the rows' third-order defect in `T_x` is second
order once differentiated again.

```
energy component, :cascade3   wall orders
inviscid wall                 4.0 / 4.0
viscous wall                  2.6 / 2.0
```

The standing wave's mass flux is odd about the wall, so the leading error term of
every closure, proportional to a derivative of the flux that vanishes there, drops out and
the rows read one order above their formal one, while the shear mode's tangential stress is
even and reads the formal 3 and 5. The viscous term is two derivatives: `:cascade4`'s
fourth-order defect in `v_x` becomes third order once differentiated again, no better than
`:cascade3` on the shear mode, while the Brady–Livescu rows' leading term again vanishes on
the odd `v` and they read 5 after the second derivative.

### Walls

The standing wave's wall window at t = 0.4, density, against the fine reference; the row
against the mirror agrees to three digits everywhere the closure defect is above 1e-12:

```
                                     N=49        97          193         orders          l2 orders
inviscid, C6 cascade3, unfiltered    2.843e-7    1.949e-8    1.217e-9    3.87 / 4.00     4.25 / 4.62
inviscid, C6 cascade3, cascade filter 2.891e-5   8.736e-6    2.359e-6    1.73 / 1.89     2.16 / 2.11
inviscid, C6 cascade3, onesided      2.580e-7    1.353e-8    1.254e-9    4.25 / 3.43     4.70 / 4.12
inviscid, C6 cascade4, unfiltered    3.137e-8    6.000e-9    2.110e-9    2.39 / 1.51     3.16 / 0.64
inviscid, C6 cascade4, cascade filter 4.858e-5   1.285e-5    3.306e-6    1.92 / 1.96     2.27 / 2.15
inviscid, C6 cascade4, onesided      4.168e-8    6.826e-8    7.814e-7    −0.71 / −3.52   0.49 / −3.78
inviscid, C6 BL, unfiltered          2.537e-9    4.503e-11   8.802e-13   5.82 / 5.68     6.44 / 5.53
inviscid, C6 BL, cascade filter      4.039e-5    1.415e-5    4.028e-6    1.51 / 1.81     2.25 / 2.19
inviscid, C6 BL, onesided            2.517e-9    4.485e-11   8.737e-13   5.81 / 5.68     6.32 / 5.53
inviscid, C8 BL, unfiltered          1.818e-10   4.603e-12   4.134e-13   5.30 / 3.48     time-limited
inviscid, C8 BL, cascade filter      negative density at step 15 / 17 / 19
inviscid, C8 BL, onesided            1.519e-10   4.527e-12   2.855e-13   5.07 / 3.99     time-limited
viscous, C6 cascade3, unfiltered     2.336e-7    1.583e-8    1.019e-9    3.88 / 3.96     4.56 / 4.53
viscous, C6 cascade3, cascade filter 2.712e-5    7.371e-6    2.132e-6    1.88 / 1.79     2.11 / 2.09
viscous, C6 cascade3, onesided       1.538e-7    1.247e-8    8.703e-10   3.62 / 3.84     4.28 / 4.39
viscous, C6 cascade4, unfiltered     3.387e-8    1.065e-9    3.421e-11   4.99 / 4.96     5.37 / 5.22
viscous, C6 cascade4, onesided       3.707e-8    1.306e-9    5.270e-11   4.83 / 4.63     5.21 / 5.07
viscous, C6 BL, unfiltered           1.941e-9    4.784e-11   8.411e-13   5.34 / 5.83     6.26 / 6.41
viscous, C6 BL, onesided             1.976e-9    4.376e-11   7.807e-13   5.50 / 5.81     6.16 / 6.32
viscous, C8 BL, unfiltered           1.112e-10   9.266e-13   2.853e-14   6.91 / roundoff
viscous, C8 BL, cascade filter       negative density at step 63 / 190 / 1055
viscous, C8 BL, onesided             2.126e-10   1.019e-12   3.864e-14   7.71 / roundoff
```

The shear mode's tangential momentum, exact:

```
                              N=49        97          193         orders
C6 cascade3, unfiltered       7.924e-9    3.293e-10   1.164e-11   4.59 / 4.82
C6 cascade3, cascade filter   6.452e-6    5.969e-7    4.976e-8    3.43 / 3.58
C6 cascade3, onesided         1.586e-9    7.802e-11   3.574e-12   4.35 / 4.45
C6 cascade4, unfiltered       9.437e-10   5.930e-11   2.358e-12   3.99 / 4.65
C6 cascade4, onesided         2.114e-9    7.152e-11   2.172e-12   4.89 / 5.04
C6 BL, unfiltered             1.692e-10   1.359e-12   1.052e-14   6.96 / 7.01
C6 BL, cascade filter         7.507e-6    1.128e-6    1.427e-7    2.73 / 2.98
C6 BL, onesided               1.943e-10   1.692e-12   1.334e-14   6.84 / 6.99
C8 BL, unfiltered             8.431e-13   2.160e-15   roundoff    8.61
C8 BL, cascade filter         negative density at step 86 / 270, dt collapse at 5309
C8 BL, onesided               1.063e-12   4.356e-15   roundoff    7.93
```

1. **The cascade filter is the accuracy of every filtered wall.** Under it every derivative
   closure converges at 1.5–2.3, with a hundred to a thousand times the unfiltered error on
   the finest grid. The one-sided rows return each closure to within a few percent of its
   unfiltered error. The filtered rows' `dt` measure is ten to fifty times the unfiltered
   rows' 0.001, so the filter's defect alone depends on the step. The F2 row's defect is
   `O(h²) f''` at the wall, and the shear mode shows it: its filtered field is odd about the
   wall, `f''` vanishes there, and the cap is 3.5 instead of 1.8. The default filter rows
   moved for this reason ([the filter's wall rows](#the-filters-wall-rows)).

2. **The unfiltered `:cascade3` wall is fourth order in evolution**, on the standing wave
   with and without viscosity and on the shear mode, whose right-hand side is third order at
   the wall: the solution norm gains at least an order over the pointwise truncation, as
   Gustafsson's theorem allows, and the `l2` order is half an order above the windowed one
   because the defect occupies a window of fixed node count.

3. **C6 Brady–Livescu reads 5.7–5.8 at a wall**, unfiltered or under the one-sided rows,
   inviscid and viscous, and more than an order higher on the shear mode. C8 Brady–Livescu
   converges at 7–8 until the time error or round-off floors it from the middle grid on. It
   fails on smooth data under the cascade filter in every configuration measured, so the
   recorded shock failures of that pair are not shock failures.

4. **`:cascade4` carries an undamped mode at an inviscid wall.** Without a filter its error
   stops converging by N = 193; under the one-sided filter it grows with N; under the
   cascade filter it is capped at 1.9 as every closure is. Viscosity damps the mode, and the
   viscous wall reads 5.0 unfiltered. `:cascade4` needs the F2 row.

5. **The mirror and the fine reference agree**, so the closure defect is the whole of the
   wall error at every C6 row. The interior error is two to three orders below it and
   converges at 4.0–4.2 at `cfl = 0.25`, the time integrator's order:

   ```
                  N=49       97         193
   interior       8.1e-11    4.3e-12    2.6e-13
   ```

### Patch interfaces and refinement levels

The entropy wave through a same-level interface (two root patches, periodic, so both patch
ends are interfaces), interface window, t = 0.5:

```
                                   N=48        96          192         orders          l2 orders
k=3, C6, unfiltered                5.682e-4    1.653e-5    6.315e-7    5.10 / 4.71     5.21 / 4.61
k=3, C6, cascade filter            2.677e-4    1.109e-5    4.201e-7    4.59 / 4.72     4.99 / 4.75
k=3, C6 BL, unfiltered             1.381e-4    1.394e-6    1.119e-8    6.63 / 6.96     6.92 / 7.34
k=3, C6 BL, cascade filter         1.472e-4    1.883e-6    1.945e-8    6.29 / 6.60     6.74 / 7.13
k=3, C10, unfiltered               5.800e-4    1.639e-5    5.941e-7    5.14 / 4.79     5.13 / 4.65
k=1, C6, unfiltered                8.105e-7    9.544e-8    8.262e-9    3.09 / 3.53     4.00 / 4.32
k=1, C6, cascade filter            8.731e-7    4.501e-8    3.799e-9    4.28 / 3.57     4.52 / 4.50
k=1, C6 BL, unfiltered             7.105e-8    7.909e-10   8.611e-12   6.49 / 6.52     6.95 / 7.03
k=1, C6 BL, cascade filter         9.362e-8    1.343e-9    1.502e-11   6.12 / 6.48     6.80 / 7.02
k=1, C10, unfiltered               8.173e-7    9.800e-8    8.523e-9    3.06 / 3.52     4.04 / 4.35
```

A same-level interface reads the order of its closure rows, not of the interior scheme: the
C10 rows are the C6 rows to three digits at every interface, the divergence's closure rows
there being the C6 cascade whichever interior is chosen. The k = 3 wave is pre-asymptotic at
the root spacing, where kh = 0.39 on the coarsest grid. Its higher orders are the approach
to the asymptotic ones that the k = 1 wave reads over the same grids, so
`test/convergence.jl` gates this row on that wave. The same-level interface's
right-hand-side error is 27 times the two-level one's at the same root spacing, 3³ for the
fine level's spacing.

The entropy wave, k = 3, through the nests, interface window; the C10 rows are the C6 rows
to two digits and are omitted:

```
                                                 N=48        96          192         orders          l2 orders
2 levels, C6, unfiltered                         1.420e-5    1.335e-6    9.408e-8    3.41 / 3.83     3.62 / 3.53
2 levels, C6, cascade filter                     4.766e-6    2.433e-7    1.577e-8    4.29 / 3.95     4.59 / 4.08
2 levels, C6 BL, unfiltered                      2.667e-7    4.111e-9    6.401e-11   6.02 / 6.01     6.02 / 5.99
2 levels, C6 BL, cascade filter                  1.567e-6    4.131e-8    7.378e-10   5.25 / 5.81     6.82 / 6.57
2 levels subcycled, C6, unfiltered               1.420e-5    1.321e-6    9.198e-8    3.43 / 3.84     3.64 / 3.56
2 levels subcycled, C6 BL, unfiltered            2.670e-7    4.128e-9    6.480e-11   6.02 / 5.99     6.02 / 5.98
3 levels, C6, unfiltered                         1.244e-5    1.101e-6    7.407e-8    3.50 / 3.89     3.58 / 3.52
3 levels, C6, cascade filter                     4.998e-6    1.579e-7    1.097e-8    4.98 / 3.85     5.99 / 4.27
3 levels, C6 BL, unfiltered                      2.667e-7    4.111e-9    6.391e-11   6.02 / 6.01     6.02 / 5.99
3 levels subcycled, C6, unfiltered               1.244e-5    1.086e-6    7.192e-8    3.52 / 3.92     3.61 / 3.55
3 levels subcycled, C6 BL, unfiltered            2.670e-7    4.127e-9    6.491e-11   6.02 / 5.99     6.01 / 5.98
viscous standing wave, 2 levels, C6              1.318e-7    1.079e-8    8.597e-10   3.61 / 3.65     3.69 / 3.79
viscous standing wave, 2 levels, C6 BL           6.755e-10   1.418e-11   3.511e-13   5.57 / 5.34     6.09 / 5.63
viscous standing wave, 3 levels, C6              9.841e-8    9.216e-9    8.284e-10   3.42 / 3.48     3.38 / 3.57
```

A level interface converges at 3.4–3.9 under the C6 closure cascade at the fine spacing, on
every nest, subcycled or not, inviscid or viscous. C6 Brady–Livescu reads 6.0 on the
inviscid nests and less on the viscous one. The covered parent nodes and the interior sit at
or below the interface window throughout, and the composite `l2` follows the window. The
subcycled rows differ from the global-step ones in the third digit, and on the finest grid
their `dt` measure is several times the global step's, the Hermite shell's time error
beginning to show under the sixth-order closure:

```
dt measure at N = 192    global step     subcycled
                         0.001–0.006     0.01–0.02
```

The filter lowers the interface error of the cascade rows and raises Brady–Livescu's, up to
twofold at a same-level interface and six to twelve times at a level interface; at a level
interface the filter's closed rows act at the fine spacing, and the 1.8 cap of the wall rows
does not appear within these grids.

**The other components.** The momentum and energy of the inviscid wall and the two-level
entropy wave, unfiltered:

```
                                     N=49/48     97/96       193/192     orders
inviscid wall, cascade3, rho u       4.801e-8    2.282e-9    9.429e-11   4.39 / 4.60
inviscid wall, cascade3, E           1.043e-6    7.114e-8    4.436e-9    3.87 / 4.00
inviscid wall, C6 BL, rho u          6.084e-10   3.863e-12   5.091e-14   7.30 / 6.25 (time-limited at 193)
inviscid wall, C6 BL, E              8.409e-9    1.427e-10   3.051e-12   5.88 / 5.55
2 levels, cascade3, rho u            7.099e-6    6.675e-7    4.704e-8    3.41 / 3.83
2 levels, cascade3, E                1.775e-6    1.669e-7    1.176e-8    3.41 / 3.83
2 levels, C6 BL, rho u               1.334e-7    2.056e-9    3.201e-11   6.02 / 6.01
2 levels, C6 BL, E                   3.334e-8    5.140e-10   8.022e-12   6.02 / 6.00
```

The energy reads the density's order at the wall and the momentum half an order more; at a
level interface the three components read one order to two digits, so the density rows the
guards are set from are representative.

### Level transfer order

`julia --project=. -t 1 bench/boundaryorder.jl study=transfer,transfertime`
(about thirteen minutes; `coupling="(key=value,)"` reruns every row under
another interface coupling). The interpolation order of the level transfer
enters only where a run reads interpolated values. The level rows at each
`level_interpolation_order`, default interface coupling, interface window:

```
interface window at N = 192, orders over N = 48 / 96 / 192;
"same" is identical to the last digit across orders 4, 6 and 8;
2-D: the oblique entropy wave k = (2, 1), u = (0.5, 0.25) through a square level
                                            order 4             order 6             order 8
1-D, 2 levels, C6, unfiltered               9.408e-8 (same)     same                same         3.41 / 3.83
1-D, 2 levels, C6, filtered                 8.968e-8 4.31/3.99  1.577e-8 4.29/3.95  1.561e-8 4.10/3.91
1-D, 2 levels subcycled / 3 levels, C6      9.198e-8 / 7.192e-8, same at every order
1-D viscous, 2 levels, C6                   4.575e-10 3.80/4.17 8.597e-10 3.61/3.65 8.616e-10 3.61/3.65
1-D, 2 levels, C6 BL, unfiltered            6.401e-11 (same)    same                same         6.02 / 6.01
1-D, 2 levels, C6 BL, filtered              3.309e-7 3.67/3.96  7.378e-10 5.25/5.81 1.049e-10 6.98/6.78
1-D, 2 levels subcycled / 3 levels, C6 BL   6.480e-11 / 6.491e-11, same at every order
1-D viscous, 2 levels, C6 BL                1.239e-9 2.90/2.87  3.511e-13 5.57/5.34 1.080e-13 6.20/5.98
1-D, 2 levels, C10, filtered                9.047e-8 4.30/3.99  1.592e-8 4.31/3.95  1.575e-8 4.13/3.91
2-D, 2 levels, C6, unfiltered               1.019e-7 3.98/3.57  1.840e-8 3.35/3.82  1.840e-8 3.35/3.82
2-D, 2 levels, C6, filtered                 8.095e-8 3.90/4.00  3.720e-9 4.05/4.04  3.697e-9 3.92/4.01
2-D, 2 levels, C6 BL, unfiltered            1.024e-7 3.66/3.89  9.156e-11 5.72/5.81 7.739e-12 5.63/5.71
2-D, 2 levels, C6 BL, filtered              6.894e-7 2.61/2.61  6.043e-10 4.87/4.69 1.279e-11 6.69/6.58
```

In one dimension the imposed boundary plane is a single node coincident with a
parent node, so an inviscid, unfiltered run never reads an interpolated value
and its rows cannot depend on the order; the transfer enters through the
ghosts a gradient (viscous) or a filter reads, and in two dimensions through
the planes, interpolated along the other dimension. Under the default coupling
order 6 is saturated everywhere but the viscous row, where order 4 happens to
measure lower. The 2-D case is `entropy2d_case`.

With Brady–Livescu interface rows, which stand in here for a sixth-order
interface divergence, the transfer order sets the interface error: under the
sixth-order transfer the filtered, viscous and two-dimensional rows fall short
of the closure's order, and the eighth-order transfer lowers their errors 3 to
47 times on the finest grid. There the 2-D filtered rows read a `dt` measure of
0.066, the time error beginning to show.
C8 Brady–Livescu at a level interface is unstable at every order: the inviscid
error grows between the two finest grids and the filtered runs lose positivity
by t = 0.4. The C10 interior under the C8 filter reads the C6 rows.

Under the cascade rows the C8 and C10 levels read the same at every transfer
order to the second digit. The closure rows bind, so matching the order to the
operator costs nothing there:

```
derivs=c8,c10 orders=6,8,10, any order      interface window   orders
C8, unfiltered                              1.288e-7           3.27 / 3.85
C10, unfiltered                             1.432e-7           3.61 / 3.76
C8 and C10, filtered                        1.57–1.59e-8
viscous, C8 and C10                         9.2e-10, 9.8e-10   3.5–3.6
```

Under the ghost fluxes (`coupling="(interface_flux=:ghost,)" orders=8,10
derivs=c6,c8,c10`) the shell binds instead:

```
                                            order 8              order 10
2 levels, C6, unfiltered                    6.380e-11 6.02/6.01  the same
2 levels, C6, filtered                      1.049e-10 6.96/6.78  the same
2 or 3 levels subcycled, C6                 6.481e-11 6.02/5.99  the same
viscous, 2 levels, C6                       1.060e-13 6.29/6.06  1.059e-13 6.31/6.06
2-D, 2 levels, C6, unfiltered / filtered    4.071e-12 / 6.636e-12, 6.08/5.89 / 6.87/6.91; the same
2 levels, C8, unfiltered                    1.873e-12 7.27/7.46  8.715e-14 8.00/7.77
viscous, 2 levels, C8                       9.497e-13 at N = 96  1.354e-14 at N = 96
2 levels, C10, unfiltered                   1.392e-12 7.70/7.55  1.643e-14 9.80/7.23
viscous, 2 levels, C10                      3.646e-13 at N = 96  1.932e-14 at N = 96
2 levels, C8 or C10, filtered               8.13e-11 7.14/7.04   the same
2 levels subcycled, C8 or C10               orders 7.2–7.8 over N = 48 / 96; dt 0.4–0.9 at N = 192
```

C6 is saturated at order 8; C8 reaches its eighth order and C10 its tenth only
at order 10, below which the shell sets the constant. The viscous rows reach the
round-off floor at N = 96. The filtered rows read the C8 filter. The subcycled
C8 and C10 rows turn to the Hermite shell's time error on the finest grid.

**Time.** The temporal order of these rows is in [temporal order](#temporal-order).

### Repeated filtering

The inviscid wall under `:cascade3`, the total against the fine reference, wall window /
interior; `passes` is the number of filter applications:

```
N = 49                              passes   wall        interior     l2
unfiltered                             0     2.843e-7    5.135e-8     3.571e-8
cascade, every step, relaxed          97     2.891e-5    1.860e-5     8.905e-6
cascade, every 2nd step, relaxed      48     2.734e-5    1.488e-5     7.293e-6
cascade, every 4th step, relaxed      24     2.005e-5    1.071e-5     4.876e-6
cascade, every step, unrelaxed        97     3.565e-5    2.247e-5     1.120e-5
onesided, every step, relaxed         97     2.580e-7    4.400e-8     3.297e-8
onesided, every step, unrelaxed       97     2.865e-7    5.202e-8     3.651e-8
periodic mirror, cascade, every step  97     –           8.798e-11    4.030e-11  (unfiltered 8.131e-11)

N = 193
unfiltered                             0     1.217e-9    1.342e-10    7.650e-11
cascade, every step, relaxed         386     2.359e-6    1.137e-6     4.616e-7
cascade, every 2nd step, relaxed     193     2.189e-6    9.290e-7     3.611e-7
cascade, every 4th step, relaxed      96     1.592e-6    6.091e-7     2.245e-7
cascade, every step, unrelaxed       386     2.687e-6    1.378e-6     5.841e-7
onesided, every step, relaxed        386     1.254e-9    1.272e-10    7.277e-11
onesided, every step, unrelaxed      386     1.575e-9    2.924e-10    1.162e-10
periodic mirror, cascade, every step 386     –           2.827e-13    1.112e-13  (unfiltered 2.642e-13)
```

The cascade filter's wall error is far from proportional to the number of passes: a quarter
of the passes removes only 30% of it, and no reduction of the cadence or of the relaxation
at a fixed spacing buys a factor of two. The defect is the deposit of an `O(h²)` disturbance
two cells from the wall on every pass, propagated and damped by the wall rows and the
interior operator rather than set by the deposit rate alone. On the finest grid the error
under a pass every step is 1900 times the unfiltered one. Under the same passes the
one-sided rows, relaxed, are within 3% of the unfiltered error, and on the periodic mirror
the eighth-order interior pass moves the interior error by under 10% at every N, so the
whole of the filter's wall defect is the F2 row.

### The timestep floor

The finest grid of each family under the sixth-order closure, whose spatial error is small
enough for the time error to show, against the fine reference:

```
inviscid wall N=193, C6 BL     cfl 0.5      0.25        0.125       0.0625
  wall                          4.641e-12   8.802e-13   9.137e-13   8.751e-13
  interior                      3.976e-12   2.420e-13   4.396e-14   6.217e-14
periodic mirror N=193, C6
  interior                      4.018e-12   2.642e-13   3.564e-14   3.397e-14
entropy wave N=192, 3 levels, C6 BL, global step
  interface                     6.396e-11   6.391e-11   6.384e-11   6.394e-11
entropy wave N=192, 3 levels, C6 BL, subcycled
  interface                     8.007e-11   6.491e-11   6.397e-11   6.402e-11
```

The interior's time error is fourth order in the step, falling by a factor of 15 over the
first halving against the integrator's 16, until it floors at the reference's own error. The
wall window's closure defect is reached by `cfl = 0.25`, which is why the walls run there
and why the C8 rows, whose closure defect is below 1e-12 at N = 97, are measured and not
gated. At a level interface the global-step run is spatially limited at every step tested,
and the subcycled run's Hermite shell adds 25% at the largest step and 1.5% at the next.

### Temporal order

```text
julia --project=. -t 1 bench/archive/temporalorder.jl [study=all] [restrict=step|stage|none]
```

Each row integrates one case on one grid in equal steps through `run!` and reads the
maximum difference over the conserved components against the same case on the same grid
in at least 16 times as many steps, so the spatial error cancels. The inflow cases carry
their time dependence in the boundary data alone: a supersonic entropy wave through a
`DirichletBC` holding the exact state, and a uniform stream under an `NSCBCInflowBC` whose
target velocity and temperature oscillate, both with an `NSCBCOutflowBC`. `test/convergence.jl`
guards four rows on smaller grids. Steps, error and the orders between successive step
counts:

```
periodic standing wave, 96 nodes     20 1.248e-7   40 7.710e-9   80 4.787e-10  160 2.982e-11   4.02 / 4.01 / 4.00
slip walls, N = 49                   20 2.468e-7   40 1.700e-8   80 1.055e-9   160 6.794e-11   3.86 / 4.01 / 3.96
Dirichlet inflow g(t), N = 65        80 3.361e-7  160 2.280e-8  320 1.474e-9   640 9.367e-11   3.88 / 3.95 / 3.98
NSCBC inflow target(t), N = 65       40 5.517e-6   80 4.016e-7  160 2.422e-8   320 1.466e-9    3.78 / 4.05 / 4.05
filter, periodic, relaxed            20 1.248e-7   40 7.702e-9   80 4.790e-10  160 2.999e-11   4.02 / 4.01 / 4.00
filter, periodic, full strength      20 1.240e-7   40 6.949e-9   80 4.415e-10  160 6.947e-10   4.16 / 3.98 / -0.65
filter, walls, one-sided, relaxed    20 2.301e-7   40 3.953e-8   80 3.521e-8   160 1.683e-8    2.54 / 0.17 / 1.06
filter, walls, cascade, relaxed      20 4.774e-5   40 2.389e-5   80 9.042e-6   160 4.386e-6    1.00 / 1.40 / 1.04

dt = h/5 and h/10 by N (h/5 is cfl 0.64 on the inflow wave), order in h at h/5
                                     N = 33     65         129        257        orders in h         in dt at 257
periodic entropy wave, u0 = 2        2.539e-7   1.594e-8   9.960e-10  6.255e-11  3.99 / 4.00 / 3.99  3.84
Dirichlet data at the stage time     4.906e-7   5.470e-8   1.232e-8   2.929e-9   3.16 / 2.15 / 2.07  3.92
Dirichlet data through dq/dt         4.422e-7   3.119e-8   2.160e-9   1.702e-10  3.83 / 3.85 / 3.67  4.08
NSCBC target(t)                      1.347e-7   5.991e-8   2.575e-8   1.105e-8   1.17 / 1.22 / 1.22  4.04
NSCBC constant target, pulse         4.108e-7   2.296e-8   1.444e-9   9.074e-11  4.16 / 3.99 / 3.99  4.00

entropy wave, two levels, N = 96, t = 0.5, composite
global step, steps                   40         80         160        320
  closure rows                       3.016e-8   1.535e-8   7.651e-9   3.726e-9    0.97 / 1.00 / 1.04
  ghost fluxes (default)             9.659e-11  2.236e-11  9.823e-12  4.836e-12   2.11 / 1.19 / 1.02
  C6 BL                              8.269e-11  2.805e-11  1.323e-11  6.320e-12   1.56 / 1.08 / 1.07
  closure rows, restrict=stage       1.455e-10  9.680e-12  6.020e-13  9.326e-14   3.91 / 4.01 / 2.69
  ghost fluxes, restrict=stage       6.181e-11  3.873e-12  2.520e-13  4.574e-14   4.00 / 3.94 / 2.46
  ghost fluxes, restrict=none        6.180e-11  3.876e-12  2.545e-13  4.086e-14   3.99 / 3.93 / 2.64
subcycled, steps                     14         20         28         40         56         80         160
  closure rows                       8.139e-8   5.789e-8   4.223e-8   3.014e-8   2.176e-8   1.532e-8   7.616e-9    0.96 / 0.94 / 0.95 / 0.97 / 0.98 / 1.01
  ghost fluxes (default)             5.730e-9   1.405e-9   3.973e-10  1.202e-10  4.887e-11  2.455e-11  1.002e-11   3.94 / 3.75 / 3.35 / 2.68 / 1.93 / 1.29
  C6 BL                              4.596e-9   1.144e-9   3.282e-10  1.016e-10  4.633e-11  2.692e-11  1.305e-11   3.90 / 3.71 / 3.29 / 2.33 / 1.52 / 1.04
  closure rows, restrict=none        4.115e-9   9.882e-10  2.573e-10  6.178e-11  1.608e-11  3.863e-12  2.426e-13   4.00 / 4.00 / 4.00 / 4.00 / 4.00 / 3.99
  ghost fluxes, restrict=none        6.007e-9   1.427e-9   3.689e-10  8.807e-11  2.282e-11  5.436e-12  3.015e-13   4.03 / 4.02 / 4.02 / 4.01 / 4.02 / 4.17
```

The script's `total` study, below, measures the same nest under the default ghost fluxes
against the exact solution, so the spatial and temporal errors add, and sets the unrefined
root beside it. The maximum density error, the fine patch's and the composite's:

```
                 cfl 0.5                                cfl 0.125
                 N = 48      96         192             N = 48      96         192
uniform root     2.708e-7    4.370e-9   8.030e-11       2.667e-7    4.112e-9   6.409e-11
global, fine     1.359e-7    1.721e-9   2.419e-11       1.376e-7    1.733e-9   2.419e-11
global           2.703e-7    4.414e-9   6.624e-11       2.716e-7    4.424e-9   6.616e-11
subcycled, fine  1.342e-7    1.812e-9   9.613e-11       1.365e-7    1.725e-9   2.419e-11
subcycled        2.716e-7    4.652e-9   9.613e-11       2.707e-7    4.416e-9   6.616e-11

restrict=stage, global, fine          1.382e-7   1.738e-9   2.430e-11
restrict=none, global, fine           9.896e-8   1.526e-9   2.382e-11
restrict=none, subcycled, fine        1.003e-7   1.617e-9   8.751e-11   (all at cfl 0.5)
```

On a fixed grid the update is fourth order with the boundary data evaluated at the
stage time, Dirichlet and NSCBC alike. At a fixed ratio of step to spacing the temporal
error of time-dependent boundary data falls more slowly than the interior's h⁴: as h² for
the Dirichlet state, largest at the first nodes of the inflow, and as h^1.2 for the NSCBC
target, whose faces keep h⁴ with constant targets and a pulse in the interior. The same
Dirichlet data imposed through their time derivative on the face node's right-hand side
(a bench-local condition) keep 3.7–3.9. This is the boundary-data order reduction of
Runge–Kutta methods of stage order one (Carpenter, Gottlieb, Abarbanel and Don, SIAM J.
Sci. Comput. 16, 1995).

The relaxed filter is a splitting first order in the step: where a pass changes the
solution, as the cascade rows do at a wall, the error is first order, and on the smooth
periodic wave the change is below the integrator's error. At full strength the filter's
error grows with the number of passes.

The level interface is first order in the step under the package's schedule, global step
and subcycled alike. Injecting the fine solution into the covered parent nodes before every
stage of the global step (`restrict=stage`), or never (`restrict=none`), gives fourth order,
so the term is the once-per-step injection, after which the coarse operator advances the
covered nodes for a whole step. It scales with the difference between the coarse and fine
right-hand sides: the ghost-flux and Brady–Livescu rows are 770 and 590 times smaller than
the closure rows' at the smallest step. The subcycled Hermite shell's fourth order shows
above it at the largest steps.

Against the exact solution the term does not show. Under the global step every composite
error at the larger step equals its value at the smaller to 0.5%, and the fine patch's to
1.2% on the coarsest grid and to the printed digits on the finest. Injection before every
stage raises no entry by more than 2%. The subcycled level at cfl 0.5 and N = 192 carries
four times its spatial error. The unrefined root at the same step also exceeds its error at
the smaller step, and removing the injection altogether lowers the fine patch's error by 9%,
so most of that excess is the fourth-order integration at the root step and the Hermite
shell's reconstruction of it, not the injection. Under subcycling the parent's step is
complete before its children start, so there is no stage to restrict at. `restrict=none`
lowers the fine patch's error by 27% on the coarsest grid and 1.5% on the finest, a spatial
effect on a wave the root resolves; without restriction the coarse state under the patch,
which the shell is interpolated from, never receives what the fine level resolves and the
root does not. The
injection is kept once per step in both modes
([bench/archive/restrictcost.jl](amr.md#bencharchiverestrictcostjl-restriction-before-every-stage)).

### What the matrix settles

The wall order a run sees is the filter's, 1.8 under the cascade rows for any derivative
closure and the closure's own under the one-sided rows; every interface, same-level or
coarse-fine, reads the C6 cascade's 3.5 at its own spacing whatever the interior scheme. The
guards in `test/convergence.jl` are set from these runs and are bit-reproducible on this
workstation at one thread; they measure the walls against the mirror, so the matrix's fine
references are not needed by them. Two-dimensional walls with a tangential inviscid flow,
isothermal walls, moving refinement and Float32 are not in the matrix. A wall placed on the
half-offset mirror carries no closure row at all and reads round-off ([the face-centred
symmetry plane](#the-face-centred-symmetry-plane)).

### Grid convergence of the shock battery

```
N         | Noh1 plat  deficit | Lax L1  | mix width
128       |  0.9945       +60% | 1.3e-2  |  0.03581
256       |  0.9994       +59% | 7.1e-3  |  0.01820
512       |  0.9991       +58% | 3.9e-3  |  0.00931
1024      |  0.9992       +56% | 1.9e-3  |  0.00481
```

**Lax L1 halves per doubling**, giving first-order L1 convergence for the captured
discontinuity regardless of interior order; the sixth- and tenth-order convergence lives in
`test/convergence.jl`, on smooth fields. **Interface width halves per doubling**, so the
regularization follows the mesh and does not settle at a fixed physical scale, as required
for the Cook artificial properties to act as a subgrid model. **Wall heating does not
converge away**, the deficit falling only from 60% to 56% across the table, the known
character of the Noh problem: the entropy error is deposited once, in the first cell at
shock formation, and remains there. Its spatial extent decreases with the cell size, so the
integrated error vanishes while the pointwise error does not.

## The filter's wall rows

```text
julia --project=. -t 1 bench/archive/wallfilter.jl
```

The cascade rows of `compact_filter` leave the end row unfiltered and apply centered
F2/F4/F6 rows at rows 2–4. One pass |F f − f| on a smooth closed line is second order along
the whole line, since the compact solve carries the second row's error inward (1.88 in the
max norm, 2.21 in L2), so the cascade rows cap the wall order of every filtered run with an
O(h²) deposit two cells from the wall each step.

`compact_filter(closures = :onesided)` replaces the same three rows by the one-sided
eighth-order rows of Gaitonde and Visbal, derived from polynomial exactness through degree 7
plus a Nyquist zero as the interior stencil is. They reproduce the centered stencil at the
centered point to 1e-16 and the published second row to 1e-14. One pass is eighth order
everywhere (8.07 in the max norm, 8.75 in L2). The second and third rows alone exceed unit
gain at some wavenumbers, yet the spectral norm of a hundred passes stays below the
cascade's at αf = 0.45 and exceeds it only at αf = 0:

```
N = 64 for the norms           αf = 0.45   αf = 0
largest gain, row 2            1.10        1.39
largest gain, row 3            1.03        1.32
‖F¹⁰⁰‖₂, one-sided             1.05        1.42
‖F¹⁰⁰‖₂, cascade               1.14        1.35
```

Both keep every eigenvalue inside the unit disk apart from the two exact ones at 1, the
constant and the unfiltered end rows.

### The battery under both row sets

The two row sets differ only where a wall carries a gradient: the one-sided rows lower the
planar Noh wall deficit, and the nine other rows agree under both at the folds, the tubes,
the blast and the shocked interface. Every case of the battery has closed ends: the
Dirichlet inflows of the tubes and the outer boundaries of the folded Noh cases take the
closure rows as a wall does. At the relaxed default, `filter_cfl = 0.35`, C6 `:cascade3`:

```
case               rows       plateau   deficit   shock    inadmissible  e_min
Noh nu=1 N=400     cascade    0.9989    60%       0.2044   8             -0.0218
                   onesided   0.9975    50%       0.2043   7             -0.0176
Noh nu=1 N=800     cascade    0.9992    61%       0.2027   6             -0.0271
                   onesided   0.9977    43%       0.2026   6             -0.0356
Woodward N=800     cascade    L1 rho 3.217e-2  peak 6.6162 at 0.7785  rho_min 0.1479
                   onesided   L1 rho 3.222e-2  peak 6.6166 at 0.7785  rho_min 0.1478
Noh nu=2, nu=3, Lax, Shu-Osher, Sedov, shock/SF6: both row sets agree to every
                   digit printed
```

Unrelaxed, the gain at the shocked wall is larger, and the one-sided rows also bring the
plateau 0.03% nearer the exact value and the front 0.002 nearer. Relaxation weakens every
pass near the diffusion-limited front under `C_beta = 1`, shrinking the cascade's F2
deposit there.

```
Noh nu=1, unrelaxed   cascade deficit   onesided deficit
N = 400               64%               29%
N = 800               66%               30%
```

### The closure-compatibility table

Every derivative closure under both sets on the wall-bounded shock cases, relaxed default:

```
case               derivative      cascade rows                            one-sided rows
Woodward N=800     C6 :cascade3    L1 3.217e-2, peak 6.6162                L1 3.222e-2, peak 6.6166
                   C6 :cascade4    L1 3.217e-2, peak 6.6161                negative density, step 2470, t = 0.019
                   C6 BL           negative density, step 851, t = 0.006   L1 3.218e-2, peak 6.6166
                   C8 BL           negative density, step 56               L1 3.229e-2, peak 6.6248
Noh cold N=400     C6 :cascade3    plateau 0.9989, deficit 60%             plateau 0.9975, deficit 50%
                   C6 :cascade4    plateau 0.9989, deficit 66%             plateau 0.9973, wall density 5.04 (overshoot 26%)
                   C6 BL           negative density, step 65               negative density, step 201
                   C8 BL           negative density, step 41               negative density, step 21
Noh warm t0=0.3    C6 :cascade3    rho[1:4] 3.990 3.994 3.998 4.001       4.025 3.988 3.993 4.007
                   C6 :cascade4    rho[1:4] 3.983 3.990 3.997 4.001       4.086 4.075 3.916 3.928
                   C6 BL           rho[1:4] 3.937 3.964 3.990 4.006       3.992 3.996 3.996 4.000
                   C8 BL           negative density, step 143, t = 0.024  3.895 3.995 4.001 3.997
```

Without the F2 row `:cascade4` fails Woodward–Colella and overshoots the cold wall; C6
Brady–Livescu completes Woodward–Colella only under the one-sided rows; C8 Brady–Livescu
fails the cold start under both. The cold rows predate the detector's wall mirror and the
slip-wall flux contract, under which C6 Brady–Livescu completes that start ([how resolved a
start must be](#how-resolved-a-start-must-be)).

The C6 Brady–Livescu failure under the cascade rows is a wall mode driven by the artificial
bulk viscosity, not by the discontinuity. A planar Noh warm-started from the exact solution
holds a uniform ρ = 4, u = 0 plateau at the wall and nothing reaches the wall before the run
ends. With an unprimed first step and an unrelaxed weight the wall density under
`:brady_livescu` nonetheless grows exponentially from t ≈ 0.06, doubling every ≈ 0.01 time
units:

```
warm planar Noh, unprimed, unrelaxed   final wall density
exact                                  4
:cascade3                              3.99
:brady_livescu                         20.3
```

Zeroing `C_kappa` and `C_mu` together leaves that growth unchanged, so β\* is the driver;
zeroing `C_beta` loses the shock instead. Where β\* is not active at a wall the rows are
innocuous: a 1% Gaussian pressure pulse between two slip walls holds `:cascade3`'s wall
density to four digits over three acoustic transits, and the Noh problem mirrored about the
wall, with inflow at both ends and no wall, completes with `:cascade3`'s profile.

The scalar spectra do not predict the failure. The sets are separated by the wall mode of
D(β D), the diffusion operator assembled for the bulk term from the first-derivative rows:
with the end rows free its largest real eigenvalue is positive and grows with the order of
the closure, and with both end rows injected it is negative. Only the momentum is injected
at a slip wall, so the density and energy rows take the free-end spectrum.

```
D(β D), N = 64, h = 1                  largest real eigenvalue
:cascade3, end rows free               1.3e-6
:cascade4, end rows free               1.7e-5
C6 :brady_livescu, end rows free       1.4e-4
C8 :brady_livescu, end rows free       3.7e-3
all five, both end rows injected       −2.5e-3
```

### The reflected pulse

A left-moving simple wave of the ideal gas, `p = 1 + amp exp(-((x - 0.5)/0.05)^2)` with `rho
= p^(1/γ)` and `u = -2(c - c0)/(γ - 1)`, between slip walls on the unit interval at
`cfl = 0.4`, against its periodic mirror of twice the length at the same spacing and step.
The mirror carries the pulse and its image about x = 1, so its restriction to the unit
interval is the wall problem at every time and evaluates no closure row. The pulse reaches
the wall near t = 0.42 and returns to its origin near t = 0.85. The density is read at
t = 0.7, with the reflected pulse in the interior. At amp = 0.01 the wave steepens over
about seven domain lengths and stays smooth. At amp = 0.1 it shocks about 0.7 into its path.

```
amp 0.01, artificial properties on         N=49        97          193         385         orders (interior)
cascade    wall                            4.5e-6      1.2e-6      2.1e-7      2.9e-8
           interior                        2.70e-4     9.13e-5     2.00e-5     4.28e-6     1.56 / 2.19 / 2.22
           l2                              6.83e-5     1.90e-5     3.98e-6     8.77e-7
onesided   wall                            2.1e-4      7.8e-6      2.1e-7      8.1e-9
           interior                        3.19e-4     2.12e-5     8.93e-7     3.08e-8     3.91 / 4.57 / 4.86
           l2                              1.15e-4     5.25e-6     1.76e-7     6.09e-9

amp 0.1, t = 0.7 (shocked after the reflection)
                                           N=97        193         385         769
cascade    interior                        9.81e-4     5.86e-4     2.20e-4     8.07e-5     0.74 / 1.41 / 1.45
           l2                              1.63e-4     5.64e-5     1.68e-5     4.17e-6
onesided   interior                        1.99e-3     7.14e-4     7.71e-5     5.98e-7     1.48 / 3.21 / 7.01
           l2                              7.96e-4     2.28e-4     1.81e-5     1.16e-7
```

On the resolved pulse the one-sided rows converge faster than the cascade rows, which cap at
2.2, and are a hundred times more accurate on the finest grid. On the coarsest grid, where σ
is 2.4 cells, they are worse. On the steepening pulse they are worse on the two coarser
grids and about a hundred times better on the finest, the crossover falling near N = 385.
The artificial properties move nothing here, the amp = 0.01 rows with them off agreeing to
two digits. The mid-resolution penalty is larger with the artificial properties off, so it
is not a sensor or bulk-viscosity coupling but the pre-asymptotic
behaviour of rows 2 and 3, whose gain exceeds unity at high wavenumber, on a reflection
resolved over fewer than about ten cells. The crossover moves to coarser grids as the
amplitude falls (N = 193 at amp = 0.03).

### Conservation and floor budgets

The runs filter through a callback with the solver's own pass off, so the change of the
totals across each pass is the filter's alone. `drift` is the change over the run relative to
the largest total seen and `filter` the passes' share; the remainder is the derivative
closure's, which is not summation-by-parts, and the floor's.

```
                                     mass drift (filter)        energy drift (filter)      floor
Woodward N=800, cascade              -3.81e-6 (-3.21e-6)        -1.93e-5 (-1.57e-5)        off
Woodward N=800, onesided             -1.77e-7 (+5.71e-8)        -1.32e-7 (+3.74e-8)        off
Woodward N=800, cascade, floor 1e-6  -3.80e-6 (-3.20e-6)        +2.35e-2 (-1.53e-5)        3257 steps, 15186 cells
Woodward N=800, onesided, floor 1e-6 +5.61e-7 (+2.11e-8)        +2.35e-2 (+2.61e-8)        3254 steps, 15169 cells
steepening pulse N=385, cascade      -2.85e-7 (-2.27e-7)        -4.68e-7 (-3.66e-7)        off
steepening pulse N=385, onesided     -2.91e-8 (+5.11e-10)       -4.26e-8 (+1.01e-9)        off
planar Noh N=400, cascade            filter -1.09e-3            filter +1.61e-4            0 cells under floor 1e-6
planar Noh N=400, onesided           filter -3.31e-5            filter -1.12e-5            0 cells under floor 1e-6
```

Between two walls the one-sided rows cut the run's mass and energy drift by twenty to a
hundred times, and the filter's own share of it by two orders. The floor, where on, fires on
the same steps and repairs the same cells to a tenth of a percent under both sets, and a
species layer against the wall reads the same mass-fraction range, interface width and
species masses to six digits.

### Float32

In Float32 the wider one-sided rows carry more round-off than the cascade rows but no floor
of the Brady–Livescu kind. Planar Noh at N = 400 in Float32 reads the Float64 rows to every
digit printed under both sets. The smooth pulse against its mirror floors near 1e-5 from
N = 97 under both sets. At N = 193 the one-sided wall window is three times the cascade's,
and its interior and `l2` below the cascade's. Woodward–Colella at N = 800 completes in
Float32 under neither set, losing positivity in the first blast, a limitation of the 10⁵
pressure ratio, so Noh alone carries the comparison.

### The decision

`compact_filter` takes `closures = :onesided` by default. The wall window of a filtered run
then converges at the derivative closure's own order instead of 1.8, and a resolved
wall-bounded evolution carries a hundred to a thousand times less error. The filter creates
two orders less mass and energy on any closed line, the planar Noh wall deficit is smaller
by 10–18 points under the relaxed default and by more unrelaxed, and the C6 and C8
Brady–Livescu rows run at a shocked wall. The cost is a two- to threefold penalty on a
reflection resolved over fewer than about ten cells, a plateau 0.15% lower on the relaxed
Noh wall, and pairing `:cascade4`, which fails without the F2 row, with
`compact_filter(closures = :cascade)`. The validation guards and stored references are taken
under the one-sided rows; regression rows asserting a cascade measurement pin it, as they
pin `filter_cfl = 0`.

The cascade's F2 row damps a linear slip-wall instability of the derivative closure exactly.
Without it the mode grows; it is measured under [constant
annihilation](#constant-annihilation) and removed by the neutral closure rows.

## Wall closures in production

```text
julia --project=. -t 1 bench/archive/wallclosure.jl
julia --project=. -t 1 bench/archive/closurequalify.jl parts=dilatation schemes=neutral3,brady_livescu
julia --project=. -t 1 bench/archive/closurequalify.jl parts=jacobian schemes=neutral3 jns=51,101 jwalls=slip
```

Unless a table says otherwise the case is the standing wave or the shear mode of
`test/smooth_cases.jl` at `cfl = 0.25` to t = 0.4, artificial properties on, under
`compact_filter(0.45, closures = :onesided)` every step with `filter_cfl = 0.35`, and
the error is the four-node wall window against the periodic mirror at the same spacing, which
carries no closure rows. The closure sets are C6 `:neutral3` (the default), `:cascade3`, C6
and C8 `:brady_livescu`, and the archived fifth-order rows of [the fifth-order closure
search](#the-fifth-order-closure-search).

### Smooth walls with the artificial properties on

With the artificial properties on under `beta_sensor = :dilatation`, the neutral rows
converge at about fourth order at every smooth wall and the Brady–Livescu rows at fifth to
seventh, and halving the step moves no order by more than a tenth. These are
field-specific evolution orders; the derivative rows retain their formal boundary moments.
All 450 derivative-only paired comparisons reach t = 0.4.

| rows | case | errors at CFL 0.25, N = 49 / 97 / 193 | orders, CFL 0.25 | orders, CFL 0.125 |
|---|---|---|---|---|
| neutral3 | inviscid slip | 5.802e-07 / 3.987e-08 / 2.588e-09 | 3.86 / 3.95 | 3.87 / 3.95 |
| neutral3 | viscous no-slip | 4.159e-07 / 2.999e-08 / 2.006e-09 | 3.79 / 3.90 | 3.80 / 3.90 |
| neutral3 | adiabatic shear | 1.439e-08 / 6.176e-10 / 2.442e-11 | 4.54 / 4.66 | 4.55 / 4.66 |
| neutral3 | isothermal shear | 1.437e-08 / 6.173e-10 / 2.441e-11 | 4.54 / 4.66 | 4.55 / 4.66 |
| Brady–Livescu | inviscid slip | 2.459e-09 / 4.851e-11 / 6.630e-13 | 5.66 / 6.19 | 5.67 / 6.11 |
| Brady–Livescu | viscous no-slip | 1.952e-09 / 4.409e-11 / 8.038e-13 | 5.47 / 5.78 | 5.47 / 5.77 |
| Brady–Livescu | adiabatic shear | 1.942e-10 / 1.692e-12 / 1.342e-14 | 6.84 / 6.98 | 6.84 / 6.97 |
| Brady–Livescu | isothermal shear | 1.942e-10 / 1.692e-12 / 1.354e-14 | 6.84 / 6.97 | 6.84 / 6.97 |

The isothermal rows read the adiabatic ones to three digits under every closure, so the
isothermal flux path adds no closure defect on a compatible case; the viscous slip wall with a
tangential shear reads the viscous no-slip row to three digits, tabulated under [the wall flux
contracts](#the-wall-flux-contracts). The fine shear errors lie near 1e-14, so the last pair's
near-seventh-order slopes touch the floating-point floor.

**The inviscid slip wall isolates the strain sensor's cusp.** Wall errors at N = 193 and CFL
0.25 under the three sensor settings, archived fifth-order rows beside the production sets:

| rows | strain | dilatation | properties off |
|---|---|---|---|
| neutral3 | 2.582e-09 | 2.588e-09 | 2.588e-09 |
| Brady–Livescu | 5.441e-11 | 6.630e-13 | 7.976e-13 |
| unfiltered-search | 4.762e-11 | 7.934e-13 | 9.137e-13 |
| filtered-objective | 7.462e-11 | 1.369e-12 | 1.403e-12 |
| DE | 1.135e-10 | 1.479e-12 | 1.499e-12 |

For Brady–Livescu the reduction is a factor of 82, and the same sensor change removes the
inviscid cap for all four fifth-order coefficient sets; the neutral closure's larger
truncation error hides it. Dilatation is not a general improvement: the viscous standing waves
and the two shear contracts already agree with their properties-off controls under both
sensors. Unfiltered, the waves under dilatation read the same orders at both CFL numbers:

| rows | inviscid slip | viscous no-slip |
|---|---|---|
| neutral3 | 3.99 / 4.04 | 4.01 / 4.01 |
| Brady–Livescu | 5.86 / 5.77 | 5.32 / 5.91 |

### Which channel carries the residual

The residual of the inviscid Brady–Livescu wall is β\*'s alone, entering through the
strain sensor's cusp. The inviscid slip wall under C6 Brady–Livescu with one constant at a
time, the constants zeroed with the machinery enabled, and the other smoother, detector and
sensor fields, node-centred sensor rows throughout:

```
variant                       N=49        97          193         orders
off                           2.430e-9    4.919e-11   7.976e-13   5.63 / 5.95
all on                        2.260e-8    4.866e-10   5.441e-11   5.54 / 3.16
C_mu only                     2.461e-9    4.854e-11   6.681e-13
C_beta only                   2.255e-8    4.853e-10   5.425e-11
C_kappa only                  2.430e-9    4.916e-11   8.524e-13
all zero, enabled             2.430e-9    4.919e-11   7.976e-13
smoother = :compact           3.088e-8    3.969e-10   5.033e-11   6.28 / 2.98
detector = :d8                3.698e-9    9.936e-11   1.305e-11   5.22 / 2.93
mu_sensor = :velocity         2.255e-8    4.853e-10   5.428e-11
beta_sensor = :dilatation     2.459e-9    4.851e-11   6.630e-13   5.66 / 6.19
```

The constants zeroed with the machinery enabled reproduce the properties-off row bitwise, so
the coefficient arithmetic adds nothing. **β\* carries the whole residual**: the `C_beta` row
equals the all-on row to three digits, and μ\* and κ\* sit where their constants put them.
The carrier is the strain sensor's cusp, not an operator's wall closure. The detector sum
`detect_sum!` under `:delta4`, on a field exactly even about both walls, reproduces the
periodic mirror bitwise at the first six nodes. The dilatation crosses zero in the interior,
where β\* stands far above its wall-region background, identically in the wall run and the
mirror run. The strain run's wall-region
β\* is noisy node to node where the dilatation run's is smooth and matches its own mirror to
between 1e-5 and 1e-2. The two sensor runs agree early and separate late, with a
wall-localized difference profile:

```
N = 193                                        strain sensor    dilatation sensor
β* at the dilatation's zero, node 98           5.505e-8, wall run and mirror run alike
β* background in the wall region               3e-13
wall error at step 1 and at step 143           equal, 2.5e-13 at the later
wall error at t = 0.4                          5.4e-11          8.2e-13
```

Under `:dilatation` the two detectors are indistinguishable at the wall, so `:d8` carries
the same cusp limit as `:delta4`.

Under the default closures every variant reads its own closure error, a thousand times the
residual, so the effect is invisible there.

### The reflected pulse under each closure

The pulse of [the filter's wall rows](#the-filters-wall-rows), density against its mirror at
t = 0.7, `cfl = 0.4`, artificial properties on:

```
amp 0.01                 N=49        97          193         385         interior orders
C6 cascade3  wall        2.087e-4    7.791e-6    2.070e-7    8.060e-9
             interior    3.186e-4    2.116e-5    8.929e-7    3.083e-8    3.91 / 4.57 / 4.86
             l2          1.149e-4    5.252e-6    1.762e-7    6.090e-9
C6 BL        wall        1.953e-4    4.389e-6    4.569e-8    1.532e-10
             interior    2.343e-4    1.402e-5    3.137e-7    1.586e-8    4.06 / 5.48 / 4.31
             l2          1.037e-4    3.542e-6    7.785e-8    3.207e-9
C8 BL        wall        1.418e-4    5.951e-7    1.493e-9    2.293e-11
             interior    3.347e-4    1.927e-5    4.248e-7    2.894e-8    4.12 / 5.50 / 3.88
             l2          1.345e-4    4.486e-6    8.424e-8    4.254e-9
C6 neutral3                                                  1.192e-9

amp 0.1 (shocked after the reflection)
                         N=97        193         385         769
C6 cascade3  wall        3.323e-3    7.459e-4    6.037e-5    1.424e-7
             interior    1.987e-3    7.143e-4    7.711e-5    5.980e-7    1.48 / 3.21 / 7.01
C6 BL        wall        4.538e-4    2.702e-5    2.957e-7    1.281e-8
             interior    1.247e-3    5.425e-4    6.126e-5    6.435e-7    1.20 / 3.15 / 6.57
C8 BL        wall        3.415e-3    4.148e-4    1.700e-6    2.100e-8
             interior    2.547e-3    7.812e-4    1.162e-4    2.470e-6    1.71 / 2.75 / 5.56
C6 neutral3                                                              1.906e-7
```

At the wall window the C6 Brady–Livescu rows are more accurate than the cascade at N = 385 by
a factor of fifty on the smooth pulse and two hundred on the steepening one. In the
interior, where the reflected pulse sits at the end, the closures agree within a factor of
three above the coarsest grid, the error there being the filter's and the time integrator's;
the exception is C8 Brady–Livescu on the finest steepening grid, at four times the cascade's
error. On the coarsest grid, where the pulse is 2.4 cells wide, every closure reads the same
1e-4. The `C6 neutral3` lines give the default closure at the finest grid of each amplitude,
with the window not recorded.

### The stable CFL range

The inviscid wall at N = 97 with the artificial properties on, against the mirror at the
same CFL number, which carries the interior scheme and the filter but no closure row:

```
cfl     C6 cascade3   C6 BL         C8 BL                        mirror
0.25    1.379e-8      1.134e-9      1.354e-9                     completes
0.50    1.542e-8      1.135e-9      1.330e-9                     completes
0.75    1.672e-8      1.565e-9      2.608e-9                     completes
1.00    1.742e-8      1.435e-9      2.342e-9                     completes
1.25    1.782e-8      1.097e-9      negative density, step 30    completes
1.50    1.805e-8      8.453e-10     negative density, step 12    completes
1.75    1.809e-8      1.224e-9      negative density, step 8     completes
2.00    8.542e-5      5.458e-3      negative density, step 8     completes
```

C6 Brady–Livescu holds its wall error flat to `cfl = 1.75`, as the cascade does; at 2.0 both
complete but their difference from the mirror jumps four orders. C8 Brady–Livescu fails from
1.25 with its mirror completing, so the T8 rows carry a stability limit of their own at a
smooth wall. On the shocked cases C6 Brady–Livescu completes every case at every CFL number
the cascade does and matches its accuracy or improves on it, while C8 Brady–Livescu
departs from the warm Noh wall well before it fails it at the top of the ladder:

```
shocked cases     C6 BL                                  C8 BL
Woodward–Colella  L1 within 0.1% of the cascade's,      completes to cfl 1.2, within 1%
                  cfl 0.15 to 1.2
warm Noh wall     within 0.2% of 4 to cfl 1.2, where     4.15 at cfl 0.6, 4.34 at 0.9,
                  the cascade's first node reads         fails at 1.2
                  4.01–4.03
steepening pulse  completes where the cascade does       fails from cfl 1.2
```

The singular start of the same case (N = 400, t0 = 0) carries a ceiling of its own under the
C6 Brady–Livescu rows. `rho[1:4]`, the wall deficit and the shock position at `retries = 0`:

```
cfl     C6 neutral3                            C6 cascade3                            C6 BL                                  C8 BL
0.15    3.043 3.370 3.712 3.941  24%  0.2021    3.115 3.490 3.717 3.923  22%  0.2020    3.126 3.576 4.229 4.516  22%  0.2019    negative density, step 23
0.30    3.049 3.376 3.717 3.942  24%  0.2021    3.173 3.508 3.722 3.934  21%  0.2019    3.279 3.640 4.295 4.547  18%  0.2017    negative density, step 13
0.45    3.056 3.384 3.721 3.942  24%  0.2021    3.236 3.523 3.729 3.949  19%  0.2019    4.086 4.298 4.696 4.722  -2%  0.1999    negative density, step 7
0.60    3.063 3.390 3.724 3.943  23%  0.2021    3.301 3.534 3.741 3.967  17%  0.2018    negative density, step 57, t = 0.0197  negative density, step 6
0.90    3.072 3.398 3.730 3.949  23%  0.2021    3.313 3.557 3.772 3.978  17%  0.2017    negative density, step 38, t = 0.0195  negative density, step 4

retries = 4                                                                                 C6 BL
0.6                                                                                         3.360 3.679 4.337 4.564
0.9                                                                                         4.015 4.319 4.727 4.734
```

The third- and fourth-order rows complete the singular start at every CFL number of the
ladder with a wall profile that moves in the second digit. The C6 Brady–Livescu rows complete
it to `cfl = 0.45` and lose positivity within the first sixty steps from 0.6, a startup event
of the kind [the CFL section](shocks.md#the-cfl-restriction-and-the-symmetry-cell) describes:
`StepControl(retries = 4)` completes both larger steps from a rollback. Their profile is not
the default's: the first node sits nearer the plateau while the third and fourth overshoot
it, and at the largest step they complete without retries the whole wall window sits above
the plateau with the shock 1% inside the exact position. C8 Brady–Livescu fails the singular
start at every CFL number, with or without retries.

### How resolved a start must be

C6 Brady–Livescu takes every start of the ladder, the singular one included, and C8
Brady–Livescu only the two best resolved. Planar Noh at N = 400 from the exact solution at
t0, `cfl = 0.15`. The front is at t0/3 and the cell is 0.0025. The warm start blends the
plateau into the inflow over four cells about the front, so a start whose front lies within
four cells of the wall is not the exact solution under any closure:

```
t0      front (cells)   C6 neutral3                C6 cascade3                C6 BL                      C8 BL
0.3     40              3.999 3.994 3.998 4.002    4.014 3.990 3.992 4.003    3.989 3.993 3.995 4.000    3.944 4.000 4.000 3.996
0.1     13              4.018 3.973 4.025 4.005    4.033 3.967 4.020 4.007    3.960 3.974 4.029 4.005    3.834 4.012 4.084 4.055
0.03    4               4.283 4.668 4.762 4.351    4.465 4.639 4.775 4.363    6.074 4.739 4.447 4.135    negative density, step 4
0.01    1.3             4.979 4.415 3.859 3.752    4.906 4.363 3.826 3.733    4.729 4.445 3.934 3.976    negative density, step 27
0.003   0.4             3.941 3.491 3.584 3.729    4.086 3.536 3.562 3.713    3.556 3.505 3.765 4.109    negative density, step 83
0.0     singular        3.043 3.370 3.712 3.941    3.115 3.490 3.717 3.923    3.126 3.576 4.229 4.516    negative density, step 23
```

`StepControl(retries = 4)` changes no completing row and completes no C8 row, so this is not
the startup restriction of [the CFL section](shocks.md#the-cfl-restriction-and-the-symmetry-cell). C6
Brady–Livescu completes these starts with the detector's wall mirror and the slip-wall flux
contract; without them it lost positivity on the two least resolved starts within a few
steps. Further starts complete as well:

```
further C6 Brady–Livescu starts, all completing
N = 400          t0 = 0.05, 0.02, 0.006, 0.001
N = 200, 800     t0 = 0, 0.003, 0.01
```

On the singular start the rows read the default closure's plateau, 1.0003 against 0.9997
of the exact value, with a slightly smaller wall deficit.

No measure of the initial data on the closure-row footprint orders the C6 failures that
remain. The largest one-cell jump of the normal velocity over the sound speed on the six
nodes the rows read, after the wall value is injected, spans nearly four orders across the
starts, and the one failure under the dilatation sensor falls at an intermediate start:

```
start          largest jump of u_n / c   strain sensor   beta_sensor = :dilatation
singular       77                        completes       completes
t0 = 0.003     1.0                       completes       completes
t0 = 0.02      0.49                      completes       negative density at step 136
t0 = 0.1       0.013                     completes
```

The remaining C6 failure on this case is a CFL ceiling of the singular start, tabulated
under [the stable CFL range](#the-stable-cfl-range). A setup-time initial-data check for
these rows would therefore reject starts they take or admit starts they fail, and none was
added (roadmap N6m).

### The round-off floor

In Float64 the Brady–Livescu rows floor where the cascade does not, but remain the more
accurate set at any resolution a run uses. One derivative of `exp(sin(3x))` on the closed
line, Float64, wall window:

```
N        C6 cascade3   C6 BL        C8 BL
193      9.075e-6      7.219e-9     2.872e-10
385      1.118e-6      1.689e-10    6.901e-12
769      1.387e-7      7.837e-12    1.063e-11
1537     1.728e-8      5.329e-11    2.491e-11
3073     2.155e-9      6.106e-11    7.566e-11
orders   3.0 throughout  5.4 / 4.4 / floor  5.4 / floor, rising
```

The cascade converges at 3.0 to the last grid with no floor in sight. The Brady–Livescu rows
floor near 1e-11 on a derivative of magnitude 8, C8 on a coarser grid than C6, and the C8
floor rises with N as the condition number does. That floor is four orders below the
cascade's error there, and the cascade would reach it only near N = 10⁴. The interior sits
20–100 times below the wall.

The conditioning does show in Float32 on the same derivative, where the C6 Brady–Livescu
rows floor between 1e-3 and 5e-3 absolute and rise with N. From N = 48 up the cascade is
therefore the more accurate closure in Float32, the precision the device path runs at,
which `test/float32_validation.jl` pins:

```
Float32, wall window            N=24       48         96         192
C6 BL                           1.02e-3    1.21e-3    2.80e-3    4.52e-3
C6 cascade3                     6.24e-3    6.88e-4    8.99e-5    9.82e-5
```

That floor does not reach the solution: a reflected pulse in Float32 floors near 3e-5 from
N = 97 under every closure, and on the steepening pulse the C6 Brady–Livescu wall window is
half the cascade's. Float32 is neither a reason to avoid the rows nor a reason to prefer
them.

### The minimum extent

`plan_direction` raises when a block is too short for its rows, so the minimum is read by
construction, on one rank (both ends of the dimension closed) and on two ranks split along it:

```
scheme            both ends closed   one end closed
C6 cascade3       5                  5
C6 cascade4       5                  5
C6 BL             9                  5
C8 cascade3       7                  7
C8 BL             13                 7
C10               7                  7
filter, either    9                  9
```

The filter's nine points bind every configuration but C8 Brady–Livescu with both ends closed,
which needs thirteen; C6 Brady–Livescu changes no minimum extent. The decomposed solve
reproduces the rows' polynomial exactness across an eight-way split, through degree 5 under
C6 and 7 under C8. The agreement is to 1e-9, and `test/mpi_tests.jl` checks it beside the
cascade's degree 3.

### A two-dimensional wall

The Cartesian Noh plane at N = 24, AR = 2 (`noh_cartesian`, the exact inflow on four faces,
so the closure rows see an oblique inflow and four corners):

```
                   plateau   center deficit   front x / y / diag        L1 rho   steps
C6 cascade3 cold   11.862    44%              0.2357 0.2328 0.2343      0.907    759
C6 BL cold         11.883    44%              0.2357 0.2314 0.2344      0.940    817
C6 cascade3 warm   16.574    13%              0.2213 0.2244 0.2252      0.462    281
C6 BL warm         16.574    13%              0.2214 0.2244 0.2252      0.482    279
C8 BL cold         timestep collapse at t = 0.25
C8 BL warm         negative density at t = 0.20
```

C6 Brady–Livescu reads the cascade's plateau, deficit and fronts to three digits on both
starts, the cold one included; this case's singular start is at the centre of the plane, where
no closure row sees it.

### The production Jacobian and the uniform state

Under the neutral rows the production step about a uniform state has a spectral radius of
one to the reading's precision at a slip wall, filtered or not, at every interior order;
under the cascade rows it exceeds one by up to a few percent. The table gives the spectral
radius of the centered Jacobian of the production step, artificial properties off, the
one-sided filter unrelaxed where on:

```
uniform state ρ = 0.9, tangential velocity 0.1, p = 1.1; cfl 0.5; N = 51 unless stated;
perturbations 3e-6 and 3e-5 move the neutral readings by ±5e-9 and the cascade's by 1e-10
                                      C6 :neutral3   C6 :cascade3   C8 :neutral3   C8 :cascade3   C10 :neutral3  C10 :cascade3
slip, unfiltered                      1.0000000000   1.0175743678   1.0000000001   1.0104697113   1.0000000001   1.0198057979
slip, one-sided filter                1.0000000011   1.0094669437   1.0000000033   1.0085087244   1.0000000035   1.0081446666
slip, unfiltered, N = 101             1.0000000001   1.0090626412   1.0000000001   1.0082617272   1.0000000001   1.0119295843
slip, one-sided filter, N = 101       1.0000000010   1.0040111557   1.0000000032                  1.0000000034   1.0029634131
Dirichlet ends, unfiltered            1.0000000000   1.0000000000   1.0000000001                  1.0000000001   1.0000000001
Dirichlet ends, one-sided filter      1.0000000018   1.0000000006   1.0000000023                  1.0000000008   1.0000000027
no-slip μ = 0.005, one-sided filter   1.0000000186   1.0000000070   1.0000000216   1.0000000079   1.0000000256   1.0000000093
2-D slip box 13 × 13, unfiltered      1.0000000001   1.0369419374
```

The 2-D row is the full production Jacobian of a square between four slip walls, corners
included. An unfiltered viscous no-slip wall exceeds one by 1e-7 to 3e-7 for every closure
set, the cascade included, and is not a closure effect.

The neutral rows hold the uniform state at round-off for forty time units at every interior
and both line lengths, where the cascade rows grow the slip-wall mode of [constant
annihilation](#constant-annihilation) and two of the runs lose positivity. Under the default
relaxed filter every step, max |u_n|:

```
cfl 0.5, double, t =      10       20       30       40
N = 51    C6 :neutral3    2.2e-14  3.9e-14  3.1e-14  3.4e-14
          C6 :cascade3    4.0e-10  7.0e-05  negative density at t = 29.75
          C8 :neutral3    2.7e-14  3.5e-14  4.1e-14  3.2e-14
          C8 :cascade3    7.1e-11  3.9e-06  2.0e-03  3.7e-03
          C10 :neutral3   1.9e-14  3.6e-14  3.7e-14  3.4e-14
          C10 :cascade3   3.6e-11  1.2e-06  2.9e-03  1.8e-03 (saturated)
N = 101   C6 :neutral3    2.5e-14  2.9e-14  4.9e-14  8.0e-14
          C6 :cascade3    9.2e-11  2.9e-06  9.1e-02  negative density at t = 36.09
          C8 :neutral3    2.2e-14  2.1e-14  5.1e-14  5.3e-14
          C8 :cascade3    1.7e-11  8.4e-08  3.5e-04  1.2e-03
          C10 :neutral3   2.8e-14  4.1e-14  6.7e-14  1.2e-13
          C10 :cascade3   3.1e-12  4.5e-09  7.1e-06  8.0e-04
```

The neutral rows cost a constant factor in the wall error at unchanged orders: the
wall-window error after an evolution is 2.5 times the cascade's at the same order, and 1.35
times on one derivative, where only the first row enters. Neutral rows against the cascade:

```
                                               neutral3, N = 49 / 97 / 193     cascade3
one derivative of exp(sin 3x), wall window     8.57e-4  1.01e-4  1.22e-5   vs  6.29e-4  7.46e-5  9.07e-6
standing wave, slip walls, t = 0.4, wall       7.22e-7  4.57e-8  2.78e-9   vs  2.84e-7  1.95e-8  1.22e-9
  orders                                       3.98  4.04                  vs  3.87  4.00
standing wave, no-slip μ = 0.005, wall         5.76e-7  3.60e-8  2.24e-9   vs  2.34e-7  1.58e-8  1.02e-9
reflected pulse, art off, t = 0.7, wall        3.04e-5  7.58e-7  3.49e-8   vs  2.63e-4  4.26e-6  1.08e-8
  interior                                     2.97e-4  2.78e-5  1.47e-6   vs  3.79e-4  2.14e-5  9.17e-7
  l2                                           8.32e-5  5.27e-6  2.31e-7   vs  1.39e-4  4.59e-6  1.42e-7
```

The pulse's wall-window error is eight times smaller on the coarsest grid and three times
larger on the finest, and its interior error 1.5 times larger from N = 97. On the battery
the neutral rows read the cascade's ν = 2 and ν = 3 rows in every printed digit and the
planar wall in the fourth digit.

### The decisions

**C6 `:neutral3` is the default.** Every long inviscid run between slip walls or symmetry
planes grew the cascade's mode, and the alternatives were a knob (`compact_filter(closures =
:cascade)` with its second-order wall defect, viscosity, or Brady–Livescu, which failed the
cold start on the solver of that decision and carries a line-length resonance under the
filter); the cost is a factor 2.5 in the wall error constant at unchanged orders and a
fourth-digit move of the planar Noh battery. `:cascade3` stays available for comparison with
earlier results, and the same two neutral rows are the C8 and C10 defaults ([closure
certificates](#closure-certificates)).

Two consequences follow from the rows' weights rather than their stability. A Float32
freestream at a wall is no longer exact: the cascade's dyadic weights annihilated a constant
in floating point, and the neutral rows' thirds leave the round-off of their products, 2.2e-6
on the walled Cartesian case of `test/float32_validation.jl`. And the flux divergence at a
patch or level interface end keeps one-sided rows, a flux array carrying no ghosts; under the
neutral rows the entropy-wave interface-window errors were two to five times larger, so
`interface_divergence_closures` keeps the cascade rows there, an interface imposing no
injected condition, and the interface baselines are unchanged to every printed digit.

**C6 `:brady_livescu` with the default `compact_filter` is a supported wall configuration**
within these limits: any CFL number the cascade rows complete a resolved case at, a
lower ceiling on the singular planar Noh start ([the stable CFL range](#the-stable-cfl-range)),
which the rows complete at that case's CFL number, Float64 or Float32, block extents no
smaller than the filter requires, serial or decomposed. Its wall solution is sixth order with
the artificial properties off and fourth order with them on under the strain sensor, at an
error fifteen times below the cascade's, and under `beta_sensor = :dilatation` it recovers
the closure's own order at an error eighty times smaller again. `test/validation.jl` guards
it on Woodward–Colella, the warm Noh wall and the cold one.

**C8 `:brady_livescu` is not supported at a wall**: the T8 rows fail a smooth wall from
`cfl = 1.25` with the mirror completing, depart from the warm Noh wall at half the CFL
number at which they fail it, and fail the two-dimensional plane on both starts and every
start of the planar case but the 40- and 13-cell ones. They remain available for a periodic
or interior block.

## The wall flux contracts

```text
julia --project=. -t 1 bench/boundaryorder.jl wall_only=true
julia --project=. -t 16 test/wall_flux_tests.jl
```

A wall correction sets the assembled normal flux before halo exchange and compact divergence
rather than subtracting terms from it, so it also removes the normal `:bulk` component flux
and is independent of the EOS energy gauge. It changes no derivative or filter coefficient.

### The no-slip wall

Each species flux is zero at an impermeable noncatalytic wall. Adiabatic total-energy flux is
zero; isothermal energy flux is `-(mu0 * cp_mix / Pr + kappa_art) * grad_T_ion[d]`, removing
species enthalpy transport while retaining pressure and viscous traction. On the incompatible
linear-temperature probe the two energy fluxes change from `[-0.005, -0.005]` to exactly
`[0.0, 0.0]`; the regression also compares the second compact right-hand-side row with the
uncorrected slip-wall case, so an endpoint-only patch cannot satisfy it. The suite
`test/wall_flux_tests.jl` runs in Float64 with the default C6 closure, unit length
and density, ideal gas R = 1 and gamma = 1.4, and no filtering or artificial transport:

| Check | Measurement | Regression guard |
|---|---|---|
| Insulated conduction, N=33 / 65, temperature max error | 1.3972e-6 / 3.0304e-7 | fine error <5e-7 and reduction >4 |
| Same, absolute trapezoidal domain-energy drift | 4.0894e-8 / 4.3029e-9 | fine drift <1e-8 |
| Species cosine diffusion, N=65, mass-fraction max error | 2.4444e-7 | <3e-7 |
| Isothermal, initial integrated RHS minus boundary heat rate | 2.7576e-6 | absolute defect <4e-6 |
| Isothermal, evolved energy rate minus time-averaged boundary heat rate | 1.8716e-6 | absolute defect <3e-6 |
| Mixed temperature/composition probe, N=65, interior RHS max error, species / energy | 3.6932e-6 / 7.7558e-6 | <1e-5 each |
| Filter-only probe, N=33, trapezoidal species-1 mass / total energy change | +4.5253e-4 / +1.1230e-3 | – |

The insulated temperature is `1 + 0.08 exp(-alpha*4pi^2*t) cos(2pi*x)` with `mu0=0.015`,
`Pr=0.8` and final time 0.002, an analytic momentum source balancing its pressure gradient so
that conduction evolves through the computed energy right-hand side; the species case uses
`Y1=0.5+0.1 cos(2pi*x)`, `mu0=0.012` and `Sc=0.75`, pressure and temperature uniform to
roundoff; the mixed temperature/composition probe checks the right-hand side at one
instant. The isothermal case starts at `T=1+0.05 sin(pi*x)` with `Twall=1`, `mu0=0.01` and
final time 2e-5; the residual between its energy-change rate and the trapezoidal integral of
the endpoint heat rates includes spatial quadrature and hard temperature enforcement, and is
neither a residual normal species flux nor a claim of exact discrete conservation. The filter-only
probe, with high-frequency cosine fields, changes the trapezoidal species mass and total
energy while the adiabatic wall fluxes stay exactly zero, so no global filter-conservation
claim follows from this hook.

Coverage: the direct face tests seed artificial conductivity and species diffusivity
independently of the detector, `:fickian` and `:bulk` both, over both precisions, all six
physical faces and their corners, ideal/NASA-9 and stiffened-gas EOS, nonsingular
cylindrical/spherical metrics, `SwitchableBC` and KernelAbstractions CPU execution. The
24-case `bench/archive/wallflux.jl` matrix also passes on a Radeon RX 6800 XT at zero maximum CPU/GPU
evolved-state difference for every precision, normal, thermal condition and species channel.
The abstract face dispatch costs a fixed 16 B per active face and nothing proportional to wall
area, and adds the expected `correct_flux!` dispatch to the JET reports:

| Allocation or report | Without face dispatch | With face dispatch |
|---|---|---|
| 48³ right-hand side | 208 B | 304 B |
| 48³ five-stage step | 1552 B | 2032 B |
| isothermal wall hook, every size | | 0 B |
| JET | 1 / 2 | 2 / 3 |

### The slip wall

Without a `correct_flux!` method the fluxes assembled at a slip-wall node reach the compact
divergence as the interior formulas produce them, which under a physical shear viscosity
leaves a conductive heat flux across an adiabatic symmetry plane and a shear traction on it,
and the near-wall solution error stops converging. The case is the standing wave between slip
walls at `cfl = 0.25` to t = 0.4, C6 `:neutral3`, artificial properties off, the density in
the four-node wall window against the periodic run at the same spacing. Filtered, at
`filter_cfl = 0.35`:

| μ | path | N = 49 | N = 97 | N = 193 | orders |
|---|---|---|---|---|---|
| 5e-3 | no hook | 5.733e-6 | 4.336e-6 | 3.991e-6 | 0.40 / 0.12 |
| 5e-3 | contract | 4.140e-7 | 2.996e-8 | 2.005e-9 | 3.79 / 3.90 |
| 5e-4 | no hook | 6.567e-7 | 9.289e-8 | 5.167e-8 | 2.82 / 0.85 |
| 5e-4 | contract | 5.309e-7 | 3.549e-8 | 2.304e-9 | 3.90 / 3.95 |

Unfiltered, as `test/convergence.jl` runs it, the uncorrected wall loses its order under
both closures, and more so at the larger viscosity:

| μ | closure | uncorrected orders, unfiltered |
|---|---|---|
| 5e-3 | `:neutral3` | 0.91 / 0.24 |
| | `:cascade3` | 0.59 / 0.12 |
| 5e-4 | `:neutral3` | 3.06 / 1.25 |
| | `:cascade3` | 2.48 / 1.01 |

**Which flux carries it.** In one dimension the conductive energy flux carries the whole
defect. A slip-wall node in a one-dimensional standing wave carries an exactly zero species
flux, ΣY = 1 making the correction velocity cancel the single species' diffusive flux, and an
exactly zero tangential momentum flux, the transverse dimensions being collapsed. Imposing
either alone reproduces the uncorrected row digit for digit, while imposing the energy flux
alone reproduces the full contract. The energy flux at the wall node is the conductive term
alone, equal to −(μ c_p/Pr) ∂T/∂n to every digit, where the mirror run's coincident node
carries round-off:

| wall-node flux, N = 97, μ = 5e-3, t = 0.4 | wall run | mirror run |
|---|---|---|
| species | 0 | |
| normal momentum | 1.070 | |
| tangential momentum | 0 | |
| energy | −2.1378e-6 | −7.2e-15 |

The convective, viscous work, enthalpy and `:bulk` terms vanish because u_n is enforced to
zero on the plane.

**Why the order collapses.** The closure rows themselves are not at fault: on the exactly
even initial data they return ∂T/∂n at their own order, third for `:neutral3`. That value
against time, filtered, μ = 5e-3:

| path | N | t = 0 | t = 0.01 | t = 0.05 | t = 0.1 | t = 0.2 | t = 0.4 |
|---|---|---|---|---|---|---|---|
| no hook | 49 | 3.901e-6 | 1.353e-5 | 4.258e-5 | 6.050e-5 | 8.973e-5 | 1.133e-4 |
| no hook | 97 | 4.885e-7 | 4.237e-6 | 1.624e-5 | 3.134e-5 | 5.704e-5 | 8.551e-5 |
| no hook | 193 | 6.109e-8 | 1.833e-6 | 1.038e-5 | 2.290e-5 | 4.608e-5 | 7.429e-5 |
| contract | 49 | 3.901e-6 | 1.050e-5 | 1.628e-5 | 1.060e-5 | 3.822e-6 | 1.404e-6 |
| contract | 97 | 4.885e-7 | 1.970e-6 | 1.663e-6 | 1.139e-6 | 4.298e-7 | 1.225e-7 |
| contract | 193 | 6.109e-8 | 2.608e-7 | 1.957e-7 | 1.353e-7 | 5.210e-8 | 1.313e-8 |
| both | orders | 3.00 / 3.00 | | | | | |
| no hook | orders | | | | | | 0.41 / 0.20 |
| contract | orders | | | | | | 3.52 / 3.22 |

Without the contract the wall gradient grows in time to a level that hardly falls with N;
with it the gradient decays and keeps converging. The closure rows' truncation error is a
heat flux across a plane that conducts none; the temperature defect it leaves in the first
few cells regenerates the gradient, and the two settle at a level that no longer follows h.

**The tangential traction.** The one-dimensional case cannot separate the tangential momentum
flux, zero there. In two dimensions the energy flux carries most of the defect and the
tangential traction the rest. The case below is measured against the periodic run on the
doubled x domain, the error being the maximum density difference over the plane:

```
slip walls on both ends of x, periodic in y over 16 nodes, μ = 5e-3, t = 0.2
ρ = 1 + 0.05 cos(πx)(1 + 0.3 cos y)
u = 0.05 sin(πx) cos y,  v = 0.05 cos(πx) sin y,  p = ρ^1.4
```

| imposed at the wall node | N = 25 | N = 49 | N = 97 | orders |
|---|---|---|---|---|
| nothing | 1.066e-5 | 4.155e-6 | 3.109e-6 | 1.36 / 0.42 |
| energy flux | 1.017e-5 | 7.699e-7 | 1.080e-7 | 3.72 / 2.83 |
| the contract | 1.035e-5 | 6.701e-7 | 4.355e-8 | 3.95 / 3.94 |

A one-dimensional case with a tangential velocity reproduces the contract's orders with the
artificial properties on or off, and its uncorrected row is within 2% of the same case without
a tangential component, so the traction contributes little without wall-parallel variation.

**The contract.** `correct_flux!(::SlipWallBC, ...)` writes zero on the owned wall plane for
every species flux, every tangential momentum flux and the total normal energy flux, leaving
the normal momentum flux untouched; that flux carries the pressure, the normal viscous stress
and the β\* dilatational term, all even about the plane. The wall stays adiabatic:
`SlipWallBC` carries no wall temperature and a symmetry plane admits no conductive exchange.
With the artificial properties on, the same three components lose their κ\* ∂T/∂n, D\* ∂Y/∂n
and tangential μ\* contributions at every slip wall; on a smooth single-species field at `mu0
= 0` only κ\* ∂T/∂n survives and κ\* is negligible there, so the inviscid smooth rows do not
move and the battery's shocked slip walls do.

**Effect on the battery and the guards.** The contract changes the runs whose slip wall
conducts heat. The two battery cases that reflect a Noh implosion off a slip wall lose wall
heating, the κ\* flux the wall was conducting being one of its sources:

| battery case | plateau without / with the contract | wall deficit without / with |
|---|---|---|
| planar Noh | 3.9883 / 3.9988 | 50% / 24% |
| aligned Noh | 3.9792 / 4.0035 | 54% / 33% |

The aligned case's transverse round-off grows by a factor of 27 ([the aligned Noh
transverse mode](shocks.md#the-aligned-noh-transverse-mode)); every other battery row holds to the
digits printed. No row of `test/convergence.jl` moves: with `mu0 = 0` and the artificial
properties off the wall-plane fluxes the contract writes are already exactly zero, so it is a
no-op there, and the file's viscous slip-wall row reads 4.00 beside the viscous no-slip
wall's 4.00. The inviscid slip-wall rows of `bench/archive/wallclosure.jl` hold to four digits
and the smooth reflected pulse is unchanged; the shocked pulse moves in the third digit, where
the reflection leaves a nonzero κ\* at the wall. The viscous slip-wall table matches the
viscous no-slip table to three digits at every closure:

| closure | properties | N = 49 | N = 97 | N = 193 | orders |
|---|---|---|---|---|---|
| C6 `:neutral3` | off | 4.145e-7 | 3.000e-8 | 2.009e-9 | 3.79 / 3.90 |
| | on | 4.165e-7 | 3.003e-8 | 2.010e-9 | 3.79 / 3.90 |
| C6 `:cascade3` | off | 1.544e-7 | 1.251e-8 | 8.727e-10 | 3.63 / 3.84 |
| | on | 1.554e-7 | 1.253e-8 | 8.730e-10 | 3.63 / 3.84 |
| C6 Brady–Livescu | off | 1.972e-9 | 4.406e-11 | 8.094e-13 | 5.48 / 5.77 |
| | on | 1.972e-9 | 4.407e-11 | 8.074e-13 | 5.48 / 5.77 |
| C8 Brady–Livescu | off | 1.996e-10 | 9.719e-13 | 1.843e-14 | 7.68 / 5.72 |
| | on | 2.052e-10 | 1.007e-12 | 1.554e-14 | 7.67 / 6.02 |

## Constant annihilation

```text
julia --project=. -t 1 bench/archive/constantfloor.jl
```

A periodic or interior row differences its taps before it multiplies, so a constant is
annihilated exactly whatever its size. A closure row is a weighted sum over the first points
of the line, so a constant c leaves (Σ w_j) c from the weights' rounding plus the rounding of
the products and their accumulation, of order eps · c · Σ|w_j| / h before the line solve
amplifies it. At issue is whether that residual affects an evolution or a useful precision
range, and whether the anchored form Σ w_j (f_j − f_1), exact on a constant, buys anything.
Every derivative and filter preset is measured in both precisions with the plan's prescaled
coefficients; the normalized residuals do not depend on N between 33 and 513 nodes.

**The weight sums.** The cascade rows' weights are dyadic rationals and sum to zero exactly
in both precisions, stored and prescaled, when L = 1 makes the inverse spacing an integer;
at L = 2π the prescaling leaves a rounding residual. The Brady–Livescu rows' sums are
rounding residuals as stored and scale with the inverse spacing when prescaled. The filter
rows pass a constant to round-off, as do the Gaussian and the Pyranda filter.

```
sum of the weights                      Float64             Float32
cascade row 1, prescaled at L = 2π      −1.8e-15            −9.5e-7
Brady–Livescu rows, stored              1e-16 to 4e-15      1e-7 to 2e-6
filter rows, Σ rhs − Σ lhs              4e-16 at worst      1e-7 to 4e-7
```

**A constant on the closed line.** The residual a closure row leaves on a constant is
round-off, and its three sources separate. The weights' own sums are at most a tenth of the
fill residual (half on the second row of C8 Brady–Livescu), and the products and their
accumulation make up the rest of the fill, 2–40 eps. The line solve amplifies the fill by
the closed-line condition number of the rows, a factor of two for the cascade and 5–40 for
Brady–Livescu. The interior carries a tenth to a hundredth of the wall residual. The
table gives the fill residual before the solve and the solved residual at the wall window
and in the interior, normalized to eps-scale (× h / c for a derivative, / c for a filter).
The anchored fill reads exactly zero on every derivative row and 2.5e-8 at most on a filter
row.

```
N = 129, c = 12345.678, L = 2π
                     Float64: fill      solved wall   interior  |  Float32: fill    solved wall   interior
C6 cascade3          1.7e-16   3.9e-16       6.0e-18  |  3.1e-8   7.0e-8        1.1e-9
C6 cascade4          4.3e-16   2.5e-15       3.9e-17  |  1.1e-7   6.4e-7        9.8e-9
C6 BL                5.2e-16   1.4e-14       4.3e-16  |  3.6e-7   2.3e-6        4.7e-8
C8 cascade3          3.2e-17   4.7e-17       9.6e-18  |  (as C10)
C8 BL                9.6e-15   1.6e-14       9.8e-16  |  2.3e-6   1.1e-5        5.7e-7
C10                  1.7e-16   3.8e-16       7.5e-18  |  3.1e-8   1.1e-7        1.6e-8
filter onesided      5.9e-16   1.2e-15       1.0e-15  |  5.0e-7   8.7e-7        4.8e-7
filter cascade       3.0e-16   4.4e-16       1.0e-15  |  1.8e-7   4.0e-7        2.4e-7
gaussian             1.5e-16   1.5e-16       0        |  1.0e-7   7.9e-8        7.9e-8
pyranda filter       0         9.0e-15       1.5e-14  |  9.4e-8   6.9e-6        2.9e-5
```

In absolute terms the derivative of a constant c reads 2–40 times eps · c / h at the wall:

```
derivative of a constant at the wall, c = 1e5, h = 0.01
Float64     1e-9
Float32     1–20
```

**A perturbation over a constant.** Over a large offset the derivative error at the wall
has a floor set by the stored field's quantization, eps · c, differentiated, which no row
can remove. The anchored rows recover a factor of two to four for the cascade and ten to
thirty for Brady–Livescu at the wall, within a factor of two of the interior's floor. The
difference appears only at offsets where the interior error is already large, c = 1e3 in
single precision and c = 1e9 in double. In Float64 an offset below 1e6 moves nothing. The
table gives the error at the wall window under the plain and the anchored rows, with the
interior as the floor.

```
f = c + sin(3x), error of f' against 3 cos(3x) relative to 3, N = 129
                  Float64: c=0      1e3       1e6       1e9     |  Float32: c=0      1         1e3       1e6
C6 cascade3 plain      3.9e-7    3.9e-7    4.2e-7    2.4e-5   |  2.3e-6    1.2e-5    8.8e-3    1.9e+1
            anchored   3.9e-7    3.9e-7    4.0e-7    1.2e-5   |  5.8e-7    1.2e-5    2.4e-3    4.6e+0
            interior   6.2e-9    6.2e-9    7.7e-9    4.0e-6   |  2.4e-6    6.8e-6    2.5e-3    2.4e+0
C6 BL       plain      3.4e-10   2.1e-9    5.9e-7    4.6e-4   |  4.7e-5    4.4e-4    3.8e-1    3.6e+2
            anchored   3.4e-10   3.4e-10   7.2e-8    3.3e-5   |  3.5e-5    1.0e-4    1.5e-2    1.4e+1
            interior   3.9e-12   4.8e-11   1.5e-8    1.4e-5   |  2.4e-6    1.3e-5    1.2e-2    1.1e+1
```

### The consequence in a run

The residual on a uniform state grows with the pressure, so it reaches order one at the
wall only for an SI-unit pressure in Float32. The case is a uniform state on 101 nodes
between slip walls, with a tangential velocity so the x-momentum flux is the constant
pressure through the wall rows and every other x flux is zero, the artificial properties on,
and a periodic line as the control. The table gives the right-hand side of the initial
state, x-momentum, wall window / interior:

```
                       Float64 cascade3    Float64 BL         Float32 cascade3   Float32 BL
p = 1 (exact)          0 / 0               3.5e-12 / 9.4e-14  0 / 0              4.6e-4 / 1.1e-5
p = 1.1, rho = 0.9     3.2e-14 / 4.9e-16   5.3e-12 / 1.4e-13  2.6e-5 / 3.9e-7    2.9e-4 / 1.3e-5
p = 1e5, rho = 1.2     0 / 0               3.0e-7 / 8.2e-9    1.1 / 1.7e-2       17 / 0.25
periodic               0 / 0               0 / 0              0 / 0              0 / 0
```

Under the cascade rows the wall-normal velocity that grows from a nonzero round-off seed is
a mode of the closed line. It grows neither linearly, as a constant residual would, nor as
√steps, as a random walk would, but by a factor of 70 per thousand steps in either
precision, while the Brady–Livescu rows hold their seed and the periodic control holds its
own. The seed only sets when the mode becomes visible. With the filter on, the periodic
control also drifts in single precision, interior rows included, from the filter's own
constant-passing round-off; that drift is of the order of the closure rows' residual, and
anchoring the derivative rows would not touch it. The table gives the wall-normal velocity
|u| and the pressure's relative drift at the wall window:

```
steps 500 / 1000 / 2000, cfl 0.5   Float64                              Float32
p = 1.1, cascade3, filter off      1.3e-14 / 9.2e-13 / 9.5e-9          1.0e-5 / 7.2e-4 / 6.2e-3
p = 1.1, cascade3, filter on       4.8e-14 / 2.9e-13 / 1.2e-11         2.5e-5 / 1.4e-4 / 1.2e-3
p = 1.1, BL, filter off            1.4e-14 / 3.8e-14 / 3.5e-14         1.0e-5 / 2.9e-5 / 1.1e-5
p = 1.1, BL, filter on             2.8e-14 / 4.5e-14 / 3.9e-13         1.5e-5 / 5.7e-5 / 1.5e-4
p = 1.1, periodic, filter on       7.4e-15 / 5.6e-15 / 3.1e-15         2.7e-6 / 4.3e-6 / 1.5e-5
p = 1e5, cascade3, filter off      0 / 0 / 0                           1.6e-3 / 1.3e-1 / 1.9
p = 1e5, cascade3, filter on       4.9e-12 / 2.3e-11 / 8.0e-10         3.0e-3 / 1.5e-2 / 2.6e-1
p = 1e5, BL, filter off            1.8e-12 / 7.4e-12 / 4.5e-12         1.1e-3 / 6.5e-3 / 1.1e-2
p = 1e5, periodic, filter on       8.2e-13 / 1.4e-12 / 1.3e-12         3.5e-3 / 6.4e-3 / 1.1e-2
```

### The slip-wall mode

The growth is a linear instability of the discrete step at an inviscid slip wall under the
cascade closures. Its rate is the same at three resolutions and at half the step, so it is
an O(c/L) mode and not a grid mode, and about half of its eigenvector lies within four
nodes of the walls. The cascade filter's F2 row damps it exactly; the one-sided filter rows
halve it, and strengthening them makes it worse, because rows 2 and 3 of that set exceed
unit gain at high wavenumber. Physical viscosity removes it at a no-slip wall and leaves a
small rate at a slip wall. Dirichlet ends are neutral, so the shock tubes and the Noh
inflows are outside it.

Under the defaults a seed reaches |u| = 1e-2 in about 30 time units in double precision and
in half that in single precision, and the artificial properties then hold it there. The
battery's wall cases end at t = 0.6 or earlier, before the mode is visible; a long inviscid
run between slip walls or symmetry planes reaches it. The neutral closure rows remove the
mode ([closure certificates](#closure-certificates)), and it is absent from a face-centred
symmetry plane.

The first table gives the wall-normal velocity grown from the round-off seed of the generic
uniform state in double precision, with the growth rate per unit time fitted between the
last two readings. The sound speed is 1.31 on a unit domain, so a unit of time is 0.76
acoustic transits. The cascade4 rows lose positivity before the last reading.

```
steps                                         1000        2000        4000       rate
N=101 C6 cascade3, filter off                 1.1e-12     9.5e-9      6.4e-3     +1.79
N=101 C6 cascade3, filter on (relaxed)        3.1e-13     1.2e-11     1.8e-8     +0.96
N=101 C6 cascade4, filter off                 2.2e-2      negative density at step 1680
N=101 C6 cascade4, filter on                  2.1e-2      negative density at step 1362
N=101 C6 BL, filter off                       4.4e-14     5.1e-14     1.0e-13    +0.09
N=101 C6 BL, filter on                        7.0e-14     3.9e-13     6.3e-12    +0.36
N=101 C6 cascade3, filter off, art off        1.1e-12     9.5e-9      8.4e-2     +2.12
N=101 C6 cascade3, filter off, v = 0          3.5e-13     2.2e-9      6.6e-3     +1.98
N=101 C6 cascade3, filter off, v = 0.5        2.2e-12     2.2e-8      6.6e-3     +1.69
N=51  C6 cascade3, filter off                 1.7e-9      9.6e-3      1.1e-2     saturated
N=201 C6 cascade3, filter off                 4.0e-14     1.7e-12     3.3e-9     +1.99
N=101 C6 cascade3, filter off, cfl 0.25       2.2e-14     1.3e-12     1.3e-8     +2.41
```

The second table linearizes one step about the uniform state by centered differences, with
the artificial properties off: their sensors are not differentiable there, and the rows
above show that they set the saturation near |u| = 1e-2 and not the rate. It gives the
largest eigenvalue modulus of the amplification matrix as a rate per unit time, the count
of eigenvalues outside the unit circle, and the share of the leading eigenvector's norm
within four nodes of a wall. A filtered row applies the unrelaxed pass after the step. The
linearized rates reproduce the time-domain ones.

```
N = 51 unless noted, cfl 0.5                  |λ|max        rate      growing   wall share
C6 cascade3, unfiltered                       1.01757437    +2.28     54 / 255  0.48
C6 cascade3, onesided filter                  1.00946695    +1.23     56 / 255  0.43
C6 cascade3, cascade filter                   1.00000004    +0.00      6 / 255  0.03
C6 cascade3, onesided filter, αf = 0.40       1.01075467    +1.40     59 / 255  0.42
C6 cascade3, onesided filter, αf = 0.30       1.01397722    +1.82     56 / 255  0.54
C6 cascade4, unfiltered                       1.05524354    +7.03     55 / 255  0.80
C6 BL, unfiltered                             1.00000002    +0.00     48 / 255  0.98
C6 BL, onesided filter                        1.00443450    +0.58     14 / 255  0.13
C8 cascade3, unfiltered                       1.01046971    +1.36     60 / 255  0.33
C8 BL, unfiltered                             1.00100131    +0.13     53 / 255  0.14
C10 (cascade rows), unfiltered                1.01980580    +2.57     51 / 255  0.47
C6 cascade3, no-slip, μ = 0.005               1.00000031    +0.00      1 / 255  0.59
C6 cascade3, slip, μ = 0.005                  1.00028758    +0.04     10 / 255  0.16
C6 cascade3, Dirichlet ends, unfiltered       1.00000001    +0.00     49 / 255  0.01
C6 cascade3, Dirichlet ends, onesided filter  1.00000004    +0.00      8 / 255  0.01
C6 cascade3, unfiltered, N = 101              1.00906264    +2.36    110 / 505  0.27
```

The centered difference at `delta = 1e-5 · max(|Q|, 1)` resolves |λ| to about 1e-9, so a
row whose |λ|max departs from one by less than that is neutral and its growing count is
round-off. The `ladder` keyword repeats a row at deltas of 3e-6 and 3e-5, and the readings
move by that resolution.

### The decision on anchored rows

No change to the closure rows for constant annihilation. The residual a closure row leaves on
a constant is round-off in every precision: 2–40 eps relative to c/h at the wall, a tenth to a
hundredth of that in the interior, from the products and their accumulation, the weights' own
sums below that and the solve's amplification above it for the Brady–Livescu sets. Anchoring
would make a uniform state's residual exactly zero and buy a factor of two to thirty on a
perturbation over an offset, but only where the stored field's quantization already floors the
interior at the same level, c/f' above 1e6 in double precision and above 1e3 in single. In a
filtered single-precision run the filter's own constant-passing round-off drifts the
interior by as much. The practical consequence of the seed is the slip-wall mode, and no
anchoring changes its growth. In single precision with SI units, p = 1e5 Pa leaves a
spurious gradient of 1 Pa/m at a wall. Through the mode it becomes a wall-normal velocity of
1.9 m/s by t = 0.03 s, which is a precision-policy matter and not a closure one.

## Closure certificates

```text
julia --project=. -t 16 bench/archive/closurecertify.jl
julia --project=. -t 1 bench/archive/neutralsearch8.jl parts=scan grid=81
julia --project=. -t 1 bench/archive/neutralsearch10.jl parts=validate,scan,wide,errors
julia --project=. -t 16 bench/archive/closurecertify.jl wall=folded schemes=c6,c8,c10 parts=spectrum,pseudo,transient
```

The instability of [constant annihilation](#constant-annihilation) is removed by new closure
rows: `:neutral3` is the default of `lele_d1_6`, `lele_d1_8` and `lele_d1_10`, the same two
rows over each interior. The production checks are in [wall closures in
production](#wall-closures-in-production).

### The instruments

`closurecertify.jl` builds the closure rows from the exact rationals of the (a, b, c) family,
checks them against `lele_d1_6()` and `lele_d1_6(closures = :cascade3)`, and measures the
injected acoustic operator L on the 2N − 2 unknowns, p on every node and u on the interior
nodes. It takes c = L = 1, so rates are per unit time and ‖L‖₂ grows in proportion to N.
Byers' test gives the pseudospectral abscissa in fourteen bisections per α_ε, and a dense
minimum of σ_min((x + iy)I − L) over y reproduces ε to 1e-6 relative. The floor on α_ε,
`rtol · ‖L‖₂`, is 7e-8 at N = 201 under the script's default. Readings at ε ≤ 1e-7
therefore use `rtol = 1e-13`. The norm ‖exp(tL)‖₂ comes from one warm-started subspace
eigendecomposition, verified against a dense operator norm at nine times to 2.3e-12 relative
over 646 times, so the last window is a lower bound and the eigenvector condition number the
rigorous one.

The instability is already in the semi-discrete operator: the injected operator's largest
real part is 2.31 per unit time against the 2.28 measured on the step. The step is checked
against a linear model built of the closed-line derivative matrix D (read off the production
operator, or assembled from the closure rows; the two agree to 6e-14), the Euler flux
Jacobian at the uniform state, the five-stage low-storage Runge–Kutta tableau, and the
wall's normal momentum zeroed at each stage as `enforce!` does, its energy correction
quadratic and dropping out of the linearization. Its amplification matrix agrees with the
production Jacobian to 8e-11 in every entry at N = 51. The acoustic pair drives the entropy
and tangential components and not the reverse, so the reduced system (p, u) on 2N unknowns
carries the whole spectrum and evaluates in a few milliseconds. Its unstable eigenvalue has
imaginary part 89 at N = 51. That is 4.6 points per wavelength, so the mode is near the grid
scale and grows at a rate of order one. It is removed exactly by the second-order F2 row
and not by the one-sided filter rows, whose gain exceeds one at high wavenumber.

`neutralsearch8.jl` and `neutralsearch10.jl` define the three-row families below, verify them
against the production plans by differentiating unit vectors, banded assembly included, scan
them, sweep the finalists over line lengths, and run the production Jacobian and the
forty-time-unit uniform state:

```
agreement with the production plans    4e-16 to 9e-16
scan                                   N = 51 and 101
sweep (649 lengths)                    every N from 12 to 600 and every tenth to 1200
one growth evaluation at N = 51        2 to 5 ms
one sweep                              220 to 300 s per member
```

### The C6 family

The cascade's rows are the unique third-order three-point row 1 and the unique fourth-order
three-point Padé row 2; widening each by one point frees three coefficients at fixed order:

```
g_1 + a g_2 = Σ_{k=1}^{4} w_k f_k          third order, a free
b g_1 + g_2 + c g_3 = Σ_{k=1}^{5} w_k f_k   fourth order, b and c free

(a, b, c) = (2, 1/4, 1/4)                   the cascade
a = 3                                       the :cascade4 row 1
```

Neutral members are rare, and most of those neutral at one line length grow at another.
At the scan's length every eigenvalue of a neutral member's injected acoustic operator lies
on the imaginary axis to 1e-14, and every other member grows at a clearly separated rate.
The fourth-order five-point row-1 family with the same row 2 has no neutral member.

```
scan over a ∈ [0, 6], b, c ∈ [−1, 1.5] at N = 51
grid points                                       16,900
neutral                                           98
growth rate between 1e-6 and 1e-2                 none
neutral over twelve line lengths from 51 to 296   4 of the 98
```

At a = 0, mapped in the (b, c) plane over eight line lengths, the neutral set is a curved
band with a hole:

```
band width in c       about 0.1
ends (b, c)           ≈ (0.56, 0.45) and (0.8, −0.1)
hole                  near b = 0.68
```

A sweep over longer lines separates two rational members of the band. One grows at a
periodic set of node counts and the other stays neutral at every length measured:

```
sweep: every N from 301 to 600 and every tenth to 1200
(a, b, c)            growth rate                      at
(1/4, 3/5, 1/5)      up to 0.042 per unit time        N = 371 and every 44th node count after it
(0, 3/5, 3/10)       below 3e-12                      every N from 12 to 1200
```

Within the a = 0 band the wall error constants rise with b, so the adopted member is the one
at the band's low-b edge that passes the sweep, the neutral member with the smallest
truncation constants:

```
row 1  (0, 1, 0)        [-11/6, 3, -3/2, 1/3]                 explicit, third order
row 2  (3/5, 1, 3/10)   [-59/40, 41/30, -3/10, 1/2, -11/120]  compact, fourth order
```

The closed line's condition number is 5.0 against the cascade's 16. The other members of
the a = 0 band also pass the production Jacobians and complete cold Noh, with wall errors
rising with b. Members neutral at two resolutions can still fail the N sweep, the Dirichlet
ends or the filtered step, so two resolutions do not select a member.

```
other band members at a = 0, (b, c) =
    (16/25, 9/50), (33/50, 7/50), (7/10, 1/25), (19/25, −3/50)
    pass the production Jacobians; cold Noh deficits 52.9–53.1%
neutral at N = 51 and 101 only, (a, b, c) =
    (1/4, 3/5, 1/5), (0, 4/5, 0), (1/2, 1/2, 1/2)
    fail the N sweep, the Dirichlet ends or the filtered step
```

**Two alternatives to new rows are closed negative.** A filter mirroring only the normal
acoustic pair, replacing p and u_n on rows 2–4 of each wall by their node-centered
parity-mirror filtered values, is neutral and holds a uniform state, but lowers every
measured order of accuracy, by more than one in the interior. One pass of either filter on
the smooth solution agrees with the exact mirror pass to 1e-14, so the loss comes from the
two row sets' different response to the wall truncation content the cascade rows inject
every step.

```
                                          standard    normal-pair mirror
|λ|max at N = 51                                      1.0000000021
uniform state to t = 40                               2e-14
standing wave, wall order                 3.84        3.40
standing wave, interior order             4.22        3.03
reflected pulse, wall order               5.9         3.1
```

Brady–Livescu rows on the flux divergence only, with the gradient, sensor and filter plans
left on the cascade, have a Jacobian equal to full Brady–Livescu's row for row but fail the
singular cold planar Noh start earlier, and cfl 0.05 moves the failure without removing it.
The converse swap completes the case, so the divergence rows are necessary and sufficient
for the failure, and the hybrid forfeits the rows' accuracy at every gradient-bearing wall.

```
singular cold planar Noh start              fails at step
Brady–Livescu on the divergence only        61
full Brady–Livescu                          201
```

### The pseudospectral measurement

A neutral member shows no non-normal amplification. For it α_ε equals max κ · ε to four or
five digits at every ε over six decades, so the ε-pseudospectrum is the first-order
eigenvalue perturbation. The Kreiss constant and the eigenvector condition number of the
adopted rows show no trend in N, the first measured to N = 415 and the second to 801. For
the cascade, and for the neighbour at its resonant node counts, α_ε tends to the positive
spectral abscissa as ε → 0 and the Kreiss constant is unbounded; the cascade's α_ε above its
abscissa is again κ · ε, so its instability is spectral and not pseudospectral. The bound
‖|V||V⁻¹|‖₂ grows linearly with N and is of no use.

The table compares the adopted rows, the neighbour and the cascade as the unstable control.
ε is absolute, in the units of L's entries; `max κ` is the largest Bauer–Fike eigenvalue
condition number ‖v‖‖w‖, and K(L) is the Kreiss constant sup α_ε/ε over the range of ε.

```
adopted (0, 3/5, 3/10), neighbour (1/4, 3/5, 1/5); ε from 1e-2 to 1e-8
                 N    ‖L‖₂     max Re λ    cond(V)   max κ    α_ε/ε             K(L)
adopted         25   8.55e1   +3.6e-15     3.08     1.2478   1.2477–1.2478     1.25
adopted         51   1.78e2   +2.1e-14     5.24     1.9728   1.9661–1.9662     1.97
adopted        201   7.13e2   +1.1e-13     5.05     1.9281   1.9281–1.9283     1.93
adopted        371   1.32e3   +3.4e-13     6.34     2.4085   2.4084–2.4122     2.41
adopted        801   2.85e3   +6.3e-13     8.75     3.3020
neighbour      201   6.86e2   +9.6e-14     9.40     3.8852   3.8850–3.8854     3.89
neighbour      371   1.27e3   +2.011e-2    7.33     3.0771   22 … 2.0e4        ∞
neighbour      415   1.42e3   +2.869e-2    5.28     2.2405   30 … 2.9e4        ∞
neighbour      801   2.74e3   +1.1e-12     8.96     3.7703
cascade         51   2.22e2   +1.767       34.2     9.1372   180 … 1.8e8       ∞
cascade        201   8.89e2   +1.775      150.5    37.133    181 … 1.8e8       ∞
cascade        801   3.56e3   +2.073      124.6    29.277
```

The adopted rows amplify by at most 3.8 over twenty time units at every N measured, the
Euclidean and quadrature-norm figures agree to three digits, and the initial fast transient
is independent of N. The spectral abscissa is below 1e-12 at every node count measured, so
cond(V), at most 8.75, bounds ‖exp(tL)‖ for all time. The table gives the maximum of
‖exp(tL)‖₂, the maximum over the fast window, and the same maximum in the trapezoid
quadrature norm of (p, u).

```
t ∈ [0, 20], fast window [0, 50/‖L‖₂]
                 N    max ‖e^{tL}‖₂   at t     fast window   energy norm
adopted         25       2.677        1.03       2.6108        2.020
adopted         51       3.783       18.13       2.6102        3.756
adopted        201       3.677        8.00       2.6102        3.617
adopted        401       3.138       16.13       2.6102        3.110
neighbour      201       7.809       13.87       2.5578        7.788
neighbour      401       5.163        7.33       2.5578        5.157
cascade         51     8.1e15        20.00       8.2996       7.3e15
cascade        401     5.3e17        20.00       8.2666       5.2e17
```

### The resonance

The neighbour's instability is a collision of the two wavenumber branches of the C6
interior at isolated line lengths. Below the peak frequency of the modified wavenumber the
interior carries two wavenumbers for one frequency, so each closed line holds two ladders of
modes, one per branch, and at a resonant length a mode of each branch merges into a growing
pair.

```
k′(θ) = (2 · (7/9) sin θ + 2 · (1/36) sin 2θ)/(1 + (2/3) cos θ)
peak 1.9894 at θ = 2.2671
```

Over N = 340 to 470 the neighbour leaves the axis only at the first three node counts of the
second table below. The adopted rows read at most 8.5e-13 over the same range. The colliding
pair at N = 371 is one mode from each branch:

```
   N     ω          Re λ        peaks θ (weight)              branch     detuning
  370   500.0375   −1.2e-14    1.3600 (1.00)                  1
  370   502.6665   −8.5e-14    2.7801 (1.00)                  2         −0.3746
  371   503.0798   ±2.011e-2   1.3651 (1.00), 2.7801 (0.40)   merged     0
  372   503.1503   +7.7e-14    1.3626 (1.00)                  1
  372   503.4540   +3.9e-14    2.7827 (1.00)                  2         +0.3037
```

The two eigenvalues merge into a quartet at N = 371 and separate again at N = 372, so the
bubble is narrower than one node count and the unstable N are isolated points. The modes
are propagating, not evanescent wall modes: the share of their norm within eight nodes of a
wall is only modestly above that of a uniform profile. Every resonance sits at the same
frequency and the same wall phases, and the integer parts of the phases advance by fixed
steps between consecutive resonances (M = N − 1):

```
norm within eight nodes of a wall: the modes 0.27 to 0.38, a uniform profile 0.21

   N     rate        ωh        θ₁        θ₂        θ₁M/π      θ₂M/π
  371   +2.011e-2   1.35968   1.36495   2.78161   160.7569   327.6032
  415   +2.869e-2   1.35882   1.36407   2.78192   179.7581   366.6015
  459   +3.182e-2   1.35813   1.36336   2.78216   198.7587   405.6001
  503   +3.236e-2   1.35755                       217.7589   444.5988
  547   +3.112e-2   1.35707                       236.7588   483.5976
  591   +2.829e-2   1.35666                       255.7584   522.5964

ωh                                       1.3585 ± 0.0015 (4.62 points per wavelength)
fractional parts of θ₁M/π and θ₂M/π      0.758 and 0.600
integer parts advance by                 19 and 39 per 44 nodes
```

The period follows from the interior dispersion alone. Along k′(θ₁) = k′(θ₂), a resonance
recurs at a node-count step ΔM when Δq below is an integer, and at the measured frequency
ΔM = 44 is the only step under seventy that comes close:

```
r = dθ₂/dθ₁ = k″(θ₁)/k″(θ₂),  Δm = round(θ₁ΔM/π),  Δq = r Δm + ((θ₂ − r θ₁)/π) ΔM
at the measured frequency     r = −0.346815,  (θ₂ − r θ₁)/π = 1.036097
|Δq − round Δq|               0.0012 at that step; 0.025 or more for every other step under seventy
```

It predicts the six node counts of the table, which the scans found, and nothing else
between N = 300 and 620. The period is a property of the C6 interior row and not of the
closure, which sets the two wall phases and with them whether the coincidence family lands
on integer node counts. Frequency coincidence alone is not sufficient: a mode of the second
branch can come close to a partner on the first and both stay neutral.

```
near coincidence that stays neutral    N      comes within
adopted rows                           379    0.019
neighbour                              394    0.008
```

### The structural attempt

Two conditions were solved exactly, at a four-node wall depth. First, a symmetric H with H D
+ Dᵀ H supported on the two wall corners, the admissible set the null space of the support
constraint taken by SVD, and the minimum eigenvalue maximized over it at tr H = N. A
well-conditioned positive-definite H exists for all three closures, the cascade included,
with its extreme eigenvalues within a factor of two of each other for the cascade and within
20% for the two neutral members, so the support condition does not separate a neutral
closure from an unstable one. A least-squares projection of the same problem stalls at a
relative residual of 1e-4 and reads as a separation that the exact solution does not show.

```
N = 51, constraint residual 4e-16    λ_min/λ_max of H
adopted rows                         +0.848
neighbour                            +0.834
cascade                              +0.517
```

Second, the exact Lyapunov certificate: for a real L with imaginary spectrum, P = Re(V⁻ᴴ
V⁻¹) is symmetric positive definite with P L + Lᵀ P = 0 to round-off, and ‖exp(tL)‖ ≤
√cond(P) for all time. It exists for the adopted rows, and for the cascade only with a
residual, since the cascade's spectrum is not imaginary. It has no structure. P is block
diagonal in (p, u) with dense blocks, and its interior diagonal varies by a factor of seven
without settling. A corner of the N = 51 solution padded with the identity interior leaves
an off-corner residual, relative to the norms of the padded matrix and of D, that neither
falls with N nor vanishes and is nearly independent of the corner's depth, so the solution
gives no fixed-depth corner block of the Sharan, Brady and Livescu form.

```
Lyapunov certificate      cond(P)                         residual of P L + Lᵀ P
adopted rows              10.5 (N = 21), 27.5 (N = 51)    1e-15
cascade                   1170                            7.1e-4

corner of the N = 51 solution at depth 6, 8 or 10, identity interior
off-corner residual at N = 51 to 401      2.5e-4 to 2.7e-4
change from depth 6 to 10                 under 4%
```

**Prior art.** The recipe is published; the result is not. Carpenter, Gottlieb and Abarbanel
(ICASE 91-71; J. Comput. Phys. 108, 1993, §7) widened compact closure rows into three- and
four-parameter families and searched for a left-half-plane spectrum on scalar advection with
a Dirichlet inflow. Zingg and Lederle widened the two boundary rows of explicit schemes by
one point each and fixed the two-parameter family by the spectrum. Brady and Livescu
(Computers & Fluids 183, 2019) fixed free boundary coefficients by the stability of the
injected nonlinear Euler problem, with conservation constraints, at one order below the
interior. Row 1 of the neutral set is Lele's (1992, eq. 4.1.3) third-order one-sided family
at α = 0, the explicit difference.

Sharan, Brady and Livescu (SIAM J. Numer. Anal. 60, 2022) treat strongly imposed boundary
conditions by the energy method. Their test system, below, is the slip-wall acoustic pair in
characteristic variables, and at τ = 1 their Figure 2 shows the CGA closures, Strand's SBP
stencils and the Cook–Riley C6 compact scheme with eigenvalues in the right half-plane, the
mode measured here. Their remedy, for explicit interiors, is a nonsquare operator with
full-norm corner blocks and a skew Q = HD outside them, and their Theorem 1 is the Lyapunov
certificate above.

Not found in any of these: a compact interior with exactly neutral closures, the reflecting
two-wall injected operator as the selection problem, a two-parameter neutral set, or an
instability at particular line lengths only. The literature expects the boundary-dependent
spectrum to be independent of N (Beam and Warming 1993); the nearest frameworks are CGA's
N-parity-dependent neutral example (eqs. 78–80) and Bonnet-Eymard, Coulombel and Faye
(arXiv:2504.00667) on wave packets coupling two boundaries. A neutral spectrum of a
non-normal operator is necessary and not sufficient.

```
test system   U_t + U_x = 0, V_t − V_x = 0, U(0) = τ₁ V(0), V(1) = τ₂ U(1)
their remedy  explicit interiors up to 3-6-3
```

### The C8 family

The C8 interior reaches ±3, so a closed edge takes three rows. Widening each cascade row by
one point at fixed order:

```
g_1 + a g_2 = Σ_{k=1}^{4} w_k f_k                third order,  a free
b g_1 + g_2 + c g_3 = Σ_{k=1}^{5} w_k f_k         fourth order, b and c free
d g_2 + g_3 + e g_4 = Σ_{k=1}^{6} w_k f_k         sixth order on 2d + e = 1

degree-6 residual of row 3: 12 − 24d − 12e, zero on a line through the cascade's (1/3, 1/3)
```

Six weights and two left-hand-side coordinates against seven conditions leave the sixth-order
members of row 3 on a line, not a plane. Off the line the row is fifth order. The family
reproduces the C8 cascade rows and the C6 `:neutral3` rows exactly, and the assembled line
agrees with the production plan to 4.4e-16.

As for C6, the neutral set sits at a = 0, with an explicit third-order row 1, and is empty
at large a. Here it is a two-dimensional region in (b, c) rather than a thin band. At a = 0
the wall-window error is set by row 1 alone, so every member there has the same wall error
and the discriminators are the interior error and the sweep. The table counts the neutral
members against a and against the position of row 3 on the sixth-order line.

```
neutral count, 41 × 41 grid in (b, c) ∈ [−1, 1.5]², N = 51 and 101
  a \ d      0     1/5    1/4    1/3    9/20
  0        132     73     76     60     71
  1/2       50     22     22     28     40
  1         12      6      7      7     17
  3/2        3      0      1      4      1
  2          1      0      0      0      1
  5/2        0      0      0      0      0
```

Only two members pass the sweep over 649 line lengths, and both keep the C6 interior row as
row 3 over C6-neutral first and second rows. The unique seventh-order row 3 on the line is
neutral at the scan's two lengths and cuts the interior error by 20%, but fails the sweep at
short lines (the last row of the table). A member is rejected above 1e-10.

```
  (a, b, c) + (d, e)                max Re λ     at N    verdict
  (0, 3/5, 3/10) + (1/3, 1/3)       +1.9e-12     1180    passes every N
  (0, 7/10, 1/25) + (1/3, 1/3)      +2.5e-12      740    passes every N
  (0, 3/5, 1/4) + (1/3, 1/3)        +9.8e-2      1040    fails from N = 104
  (0, 3/4, 1/4) + (1/5, 3/5)        +2.2e-1       580    fails from N = 97
  (0, 4/5, 1/10) + (1/5, 3/5)       +2.3e-1       780    fails from N = 118
  (0, 3/4, 1/10) + (1/4, 1/2)       +1.3e-1       478    fails from N = 130
  (0, 5/9, 1/10) + (2/5, 1/5)       +1.8e-1       436    fails from N = 142
  (0, 16/25, 9/50) + (1/3, 1/3)     +2.9e-2        92    fails at N = 92 only (to 400)
  (0, 33/50, 7/50) + (1/3, 1/3)     +2.8e-2       345    fails at N = 37 and 345 (to 400)
  (0, 3/5, 3/10) + (1/4, 1/2)       +6.3e-1       384    fails from N = 18 (to 400)
```

Twenty line lengths are a prefilter and not a verdict. Of 750 rational members such a
prefilter keeps 79, all with the same wall error. The two of lowest interior error among
them fail the full sweep, growing at 0.22 per unit time from N = 97. The adopted set is the
first row of the table, preferred over the runner-up for its denominators, its interior
error (in the table of error constants below) and its viscous Jacobian reading.

### The C10 family

The pentadiagonal interior also reaches ±3 and takes three rows, whose left-hand sides carry
the full band:

```
g_1 + a g_2 + a₂ g_3 = Σ_{k=1}^{4} w_k f_k                    third order
b g_1 + g_2 + c g_3 + c₂ g_4 = Σ_{k=1}^{5} w_k f_k             fourth order
d₋₂ g_1 + d g_2 + g_3 + e g_4 + e₂ g_5 = Σ_{k=1}^{7} w_k f_k   sixth order

(d, e) = (1/3, 1/3): centred; the seven-point row collapses to the cascade's, w₆ = w₇ = 0
```

Row 3 takes seven points because a row with M right-hand-side points and a fixed left-hand
side is exact through degree M − 1. The cascade's five-point row is sixth order only because
its left-hand side makes it centred, and at that left-hand side the seven-point row
collapses to the cascade's exactly. The banded assembly agrees with the production
`BandPlan` to 5.6e-16 to 8.9e-16.

With row 3 held at the C6 interior row, a = 0 is forced as before. No growth rate between
1e-12 and 1e-8 occurs, so the threshold is not a knob.

```
row 3 the C6 interior row; (b, c) ∈ [−1, 1.5]²; pruned over N = 31, 79, 101 and 151
  a       neutral at N = 51   surviving five lengths   b range          c range
  0            137 of 1681            24               [0.56, 0.88]     [−0.13, 0.63]
  1/2           56 of 1681            18               [0.50, 0.63]     [0.13, 0.75]
  1             16 of 1681             2               [0.50, 0.50]     [0.25, 0.31]
  3/2            4 of 1681             1               [0.44, 0.44]     [0.25, 0.25]
  2              2 of 1681             0
  5/2 to 6     0 to 1 of 1681          0
```

Only the first member below, with the cascade's row 3, stays below 1e-10 over the sweep of
every line length. Two rational members of the C6 band that are neutral for C6 at every N
fail on the pentadiagonal interior, at N = 315 and N = 127, so the resonance is specific to
the interior.

```
  (a, b, c [, d, e])                 max Re λ     at N    N above 1e-10
  (0, 3/5, 3/10)                     +1.7e-12     1090        0
  (0, 16/25, 9/50)                   +5.7e-2       516        6
  (0, 7/10, 1/25)                    +3.5e-2       127        2
  (0, 11/20, 1/2)                    +1.1e-1      1130       29
  (0, 3/5, 0)                        +6.1e-1      1100      285
  (0, 3/4, 3/4, 1/8, 1/4)            +8.2e-1      1120      475
  (0, 9/10, 3/5, 1/8, 1/4)           +7.8e-2      1160        8
  (0, 19/20, 2/5, 1/8, 1/4)          +6.9e-2       990        3
```

Widening row 3 buys accuracy and no neutrality. The lowest interior error constants sit at
the (d, e) of the last three rows of the table, where the interior error at N = 97 falls to
3.5e-8, two orders below the selected rows'. Every one of the ten lowest fails a staged
sweep, the best at N = 226 and seven further node counts. A seeded random search over all
nine coordinates, ranked by the interior error at N = 97 among the members passing every
stage, found none below 1.2e-7:

```
seeded random search      draws 40,000; neutral at N = 51: 1956; passing every stage: 121
```

The adopted set is the C6 `:neutral3` rows padded with zeros at ±2 over the unchanged row
3, so the minimum extent and the halo reach do not move.

### Error constants and the decision

The neutral rows cost accuracy at unchanged order. Their wall-window error is the same on
every interior, 1.35 times the cascade's at third order, and their interior-window error is
3.4 times the cascade's on every interior. On C8 the closed line's condition number falls by
more than half; on the pentadiagonal interior it is set by the interior and reads 20.7 for
every band member against 24.6 for the cascade. The table gives one derivative of
exp(sin 3x) on the closed line, the wall window of four nodes then the interior, with the
closed line's condition number.

```
N = 49 / 97 / 193
                          wall                              interior                      cond
C8 :neutral3     8.573e-4  1.011e-4  1.224e-5     4.878e-5  5.648e-6  6.751e-7     6.95
C8 :cascade3     6.309e-4  7.492e-5  9.107e-6     1.422e-5  1.676e-6  2.029e-7    16.25
C8 runner-up     8.573e-4  1.011e-4  1.224e-5     5.097e-5  5.903e-6  7.058e-7     6.94
C8 :brady_livescu 2.150e-6 3.277e-8  2.872e-10    1.307e-7  1.951e-9  1.697e-11 4348.9
C10 :neutral3    8.574e-4  1.011e-4  1.224e-5     6.710e-5  7.769e-6  9.287e-7    20.69
C10 :cascade3    6.328e-4  7.514e-5  9.134e-6     1.961e-5  2.311e-6  2.797e-7    24.59
C6 :neutral3     8.573e-4  1.011e-4  1.224e-5     3.387e-5  3.922e-6  4.689e-7     4.98
C6 :cascade3     6.287e-4  7.465e-5  9.075e-6     9.841e-6  1.161e-6  1.405e-7    15.97
```

On C8 and C10 the two neutral rows are followed by the C6 interior row the cascade already
used there, so `neutral_closures` builds the three-row set from the C6 table and that row;
`:cascade3` is kept for comparison. The closed C8 and C10 studies of `test/convergence.jl`
read the C6 study's errors to the printed digits. A patch or level interface keeps the
cascade rows for every neutral set. The certificate is the measured one, with no trend in N:

```
Kreiss constant                     below 2.5
eigenvector condition number        below 9 to N = 801
transient amplification             below 4 over twenty time units
```

No N-independent structural certificate has been found. The line-length resonance of the
neighbouring members is a collision of the two interior wavenumber branches of a compact
scheme, whose period is set by the interior and whose occurrence is decided by the
closure's wall phases, so the selection sweeps over every line length and not over a sample.

### The same model on a face-centred symmetry plane

`bench/archive/closuresearch.jl` holds `folded_derivative_matrix` (the interior stencil on every row,
the taps outside the line folded back with the field's sign, the ghost coupling of the result
folded onto the diagonal with the opposite sign) and `folded_acoustic_operator`,
`L = [0 −D_odd; −D_even 0]` on the 2N unknowns (p even, u odd) with no endpoint
elimination; `bench/archive/closurecertify.jl` takes `wall=folded`. The folded matrix reproduces the
package's fold plan, equals the periodic operator on 2N nodes restricted by parity, and
differentiates at the interior order:

```
fold plan on unit vectors              0.0 (C6, C8), 7.8e-16 (C10)
periodic operator restricted by parity 1.3e-15
order on cos(πx) and sin(πx)           6.00 (C6), 7.97 (C8), round-off floor from N = 20 (C10)
```

The folded L is antisymmetric and normal to round-off for all three schemes, and the
node-centred operator is neither. Every unknown has the same cell measure and the mirror
holds no node, so the restriction argument gives skew-adjointness in the Euclidean inner
product itself, not similarity to it.

```
N = 25 to 801, all three schemes    ‖L + Lᵀ‖/‖L‖        ‖LᵀL − LLᵀ‖/‖L‖²
folded                              3e-16 to 8e-16      6e-16 to 1e-15
node-centred                        0.84                0.97
```

The consequences follow without a search. The folded operator has no transient growth and
no unstable line length, and its Kreiss ratio is one to three digits:

```
                    spectral abscissa    Kreiss ratio α_ε/ε      max_t ‖exp(tL)‖, t ≤ 20
folded, N = 25      +3.2e-15             1.0000                  1
folded, N = 51                           1.0000–1.0001           1
folded, N = 101                          1.0000–1.0003           1
folded, N = 201                          1.0000–1.0012           1
folded, N = 801     +7.1e-13
node :neutral3      +3.6e-15 … +6.3e-13  1.25 / 1.97 / 1.26 / 1.93   2.68 / 3.78 / 2.61 / 3.68
```

The Kreiss ratios hold over ε = 1e-2 to 1e-6 for all three schemes. At ε = 1e-8 both
operators rise, because the classification tolerance sits on top of ε. The folded semigroup
is an isometry: the subspace iteration and the dense 2-norm agree to 3.8e-13 at every
sampled time, in the Euclidean and in the cell-measure norm. The eigenvector condition
numbers of 1.1 to 7.1 come from the general eigensolver's treatment of the doubly
degenerate ± pairs of a normal operator. The node-count scan finds no unstable length.

```
scan over every N from 12 to 600 and every tenth to 1200
largest real part        +1.137e-12 (C6), +1.592e-12 (C8), +1.577e-12 (C10)

wall times at -t 16, the three schemes
spectrum 19 s, pseudo 118 s, transient 106 s, scan 999 s
```

## The fifth-order closure search

```text
julia --project=. -t 1 bench/archive/closuresearch.jl mode=report
julia --project=. -t 1 bench/archive/closurequalify.jl schemes=de parts=polynomial,jacobian,smooth,stress
julia --project=. -t 1 bench/archive/closureenergy.jl
julia --project=. -t 1 bench/archive/closuredamping.jl parts=jacobian,uniform schemes=candidate strength=0.1 components=acoustic
```

A dead end kept for its instruments and its constraints. The search constructs fifth-order
boundary rows while retaining the C6 interior and its tridiagonal solve, and measures them
through the production derivative and timestep. Nothing established here is an energy bound
or an impossibility result. The archived vectors, seeds, populations, generation counts and
training grids are in `bench/archive/closuresearch_results.md`.

**The family.** Each row reads the first six field values. With a unit diagonal on the
implicit side, the first row has one free superdiagonal and rows 2–4 have two free
off-diagonals each; the seven parameters determine the explicit weights by exactness on
monomials of degrees zero through five, in rational arithmetic. Brady–Livescu T6 belongs to
this family, with additional conservation constraints the search does not impose. A
seven-point extension adds a multiple of `[1,-6,15,-20,15,-6,1]`, which annihilates degrees
zero through five, for eleven free parameters. The acoustic screening operator evolves
pressure on all N points and normal velocity on the N−2 interior points; finite-grid
spectral neutrality is not an energy estimate uniform in N, and eigenvector conditioning and
sampled resolvents are diagnostics, not proofs.

**Four candidate sets and what killed each.**

- `ClosureSearch.unfiltered_scheme()`, six points, selected on unfiltered acoustic growth and
  conditioning. It is fifth order in production at about a tenth of Brady–Livescu's implicit
  condition number, but its filtered production Jacobian grows, persisting across the
  perturbation ladder, and although neutral without the filter at its training sizes it
  grows at a longer line of the sweep.
- `ClosureSearch.candidate_scheme()`, six points, selected against both the unfiltered
  growth and the filtered step. Its filtered radii improve but still exceed the model gate,
  and the held-out unfiltered sweep finds growth at a longer line.
- `ClosureSearch.de_scheme()`, seven points, differential evolution optimizing stability
  before conditioning. It retains fifth-order accuracy and its slip and no-slip radii sit at
  the Jacobian's differencing floor, but its Dirichlet radius exceeds one by the same amount
  at every line length, unfiltered and filtered, and the one-sided filter does not remove
  that growth. The `:neutral3` control at the same settings is neutral, so the growth
  belongs to these rows. The filter does remove its one unfiltered slip resonance, at
  N = 171.
- The joint derivative/filter treatment of `bench/archive/closuredamping.jl`: the filtered-objective
  candidate with an extra rank-one wall pass `Fwall = I − σ v vᵀ/(vᵀv)`, `v =
  [1,-6,15,-20,15,-6,1]`, σ = 0.1, which preserves degree-five polynomials and is an O(h^5)
  boundary perturbation at fixed hyperbolic CFL. Applied to every conserved component it is
  rejected: the acoustic block alone appears neutral but the full production Jacobian grows,
  the missing modes the scalar entropy and tangential branches governed by the composite
  scalar filter. Applied only to pressure and normal velocity, with total energy adjusted
  for both the pressure and kinetic-energy changes, it passes the full Jacobian, holds a
  uniform state, reads smooth-wall orders near six with the artificial properties off, and
  fails only at the two shortest lines of its sweep. The prototype is a serial
  one-dimensional calorically perfect gas.

```
accuracy                     points   implicit cond.   moment residual   order on x^6
unfiltered_scheme()          six      141              7.0e-17           5.00000
de_scheme()                  seven                     9.995e-17         4.99964 to 5.00000
Brady–Livescu                         1192

failures                     test                                           N             reading
unfiltered_scheme()          filtered production Jacobian                   51 / 101      1.006528 / 1.002726
                               growth per unit time                                         0.85 / 0.71
                             unfiltered sweep, every N from 12 through 200  171           growth 0.662 c/L
candidate_scheme()           filtered radii against the gate 1 + 1e-10                    1.0000148 and 1.0000059
                             held-out unfiltered sweep                      415           growth 1.041 c/L
de_scheme()                  Dirichlet radius, unfiltered / filtered        79, 101, 171  1.121 / 1.0746
                               growth per unit time                         the same      +14.67, +18.81, +31.98
:neutral3 control            same settings, filtered                                      1.000000001380
joint, every component       full production Jacobian                       51 and 101    1.00087
joint, acoustic components   full production Jacobian                                     1 + 3.4e-9
                             uniform state to t = 40                                      below 2.5e-13
                             smooth-wall orders, artificial properties off                5.94 / 6.06
                             every N from 14 through 200, five larger
                             probes, three CFL numbers                      fails at 14 and 15
```

**An exact energy-norm feasibility probe.** A simple norm ansatz is ruled out, though not
other compact SBP norms. `bench/archive/closureenergy.jl` tests a restricted construction:
symmetric implicit A, `B + B' = (5/3) diag(-1,0,...,1)`, fifth-order boundary
moments, and the unchanged C6 interior. Positive A would give an SBP norm `H = (3/5)A`, and
decoupling its endpoint would make strong injection orthogonal in that norm. Exact rational
elimination finds inconsistent moment systems for tridiagonal boundary blocks with 4–12
closure rows, and full boundary blocks also fail in the tested widths; each rejection carries
a checked exact left-null certificate `y'M = 0, y'b = 1`. The standard explicit sixth-order
control admits the expected fifth-order restricted-full-norm moment families at 7 and 8 rows,
verifying the machinery.

**Brady–Livescu also has a line-length resonance.** The published T6 control is not
unconditionally neutral, with or without the filter: it grows at N = 171 in both, persisting
across the perturbation ladder. Its neutral readings at N = 51 and 101 do not generalize.

```
Brady–Livescu T6 at that N       production Jacobian    growth per unit time
unfiltered                       1.000751               0.334
unrelaxed default filter         1.000239               0.106
```

**The verdict.** A fifth-order closure is useful only where the artificial properties are
off or the dilatation sensor is on. Under the strain sensor the sensor's own cusp caps an
inviscid wall at fourth order under any closure ([wall closures in
production](#wall-closures-in-production)), and the production workloads of this solver are
shocked and cold-started. With the sensor mirrors and the slip-wall flux contract in place,
only Brady–Livescu and the unfiltered-search rows complete cold planar Noh; the other three
fail within a few tens of steps under both sensors. All five derivative choices complete
Woodward–Colella and the t0 = 0.1 warm start. Completion under a permissive policy is not an
admissibility result, and one extent, CFL and end time do not establish a cold-start
envelope.

```
cold planar Noh, N = 200, CFL 0.3, both sensors
Brady–Livescu                     completes, seven inadmissible cells
unfiltered-search rows            completes, seven inadmissible cells
filtered-objective candidate      fails at step 30
seven-point DE rows               fails at step 28
joint treatment                   fails at step 31
```

No fifth-order route is a production candidate, so the thread is closed. The independent
limitations remain: Brady–Livescu has a filtered slip-wall resonance, the unfiltered-search
set has the N = 171 resonance, the filtered-objective candidate has a held-out line-length
failure, and DE has a filtered Dirichlet radius above one. A resumption would measure
closures under `beta_sensor = :dilatation`, where the wall is no longer capped, rather than
begin at the derivative rows. The instruments are kept: `bench/archive/closuresearch.jl` holds the
family, the reduced models and the searches, `bench/archive/closurequalify.jl` the production
polynomial, Jacobian, evolution and shock checks, `bench/archive/closureenergy.jl` the exact
energy-norm probe, and `bench/archive/closuredamping.jl` the joint trial. Search assembly agrees
with production basis-vector applications to 1.066e-14 for the derivative and exactly for
the filter.

## The sensor operators at walls

```text
julia --project=. -t 1 bench/archive/sensorwall.jl
```

Three operators around the artificial-property sensors take node-centred rows at a wall and
the half-offset mirror at a fold: `delta4_sum!`, the `:gaussian` smoother and `ring_sum!`,
the `:d8` detector. The section measures these rows against closures that do not reproduce
a reflecting wall: taps read past a closed edge by clamping the index, and smoother rows
folded onto the half-offset mirror, which sit half a cell out at a node-centred wall. Their
effect on the wall solution is in [wall closures in
production](#wall-closures-in-production).

### The extensions

`delta4_sum!` reads its taps at a reflecting wall from the node-centred mirror of the
interior, ghost `2-q` low and `2n-q` high, with the field's sign: +1 for every scalar sensed
and for the tangential velocities, −1 for the wall-normal component, through
`velocity_mu!`'s `wall_parity` keyword. `sensor_mirror(bc)` names the reflecting faces,
`true` for `SlipWallBC` and `NoSlipWallBC` and `false` by default, so Dirichlet,
extrapolation and NSCBC faces and interface ends keep the clamp; the `:delta4` path queries
it per call, and the compact sensor operators' wall rows are planned from it once, when the
solver is built.

`wall_closures(scheme, σ)`, in `kernels.jl` and `kernels_banded.jl`, folds a symmetric
scheme's interior stencil, taps and left-hand-side unknowns alike, onto the node-centred
mirror, q < 1 onto 2 − q, with the sign σ, inheriting a filter's unit row sum and the eighth
derivative's zero row sum from the interior weights. The `:gaussian` smoother takes the
σ = +1 rows at every face `sensor_mirror` names, its input a detector output past an
absolute value and so even; the `:d8` detector takes both signs, a pair per dimension
indexed by the wall sign that aliases to one plan where neither face is a wall, leaving the
default configuration's plans and memory unchanged. A fold's closed far end takes the same
rows, and `FoldSpec.ring_plans` is 2×2, ghost parity by wall sign. Patched runs apply
the smoother rows at physical wall faces only. The `:compact` smoother shares the state
filter's plans, carries no half-cell shift and is unchanged.

At a coordinate fold `delta4_sum!` takes the half-offset mirror with the sign `parity[d]`:
at a self-paired fold the mirror is the signed line itself; at a paired fold every field
goes through the even/odd butterfly of `folds.jl`, unchanged, since the per-half mirror
signs derive from e(−r, θ) = ½[σ f(Mx) + σ² f(x)] and are independent of σ. The paired path
keeps the wall mirror too, the pairing map acting on the angular coordinates alone and
commuting with the reflection about the wall node. The clamp remains at closed edges that
are neither wall nor fold; on a half-offset grid mirror and clamp differ on one tap, so the
change reaches one cell per folded end.

### What the clamp cost

At a fold, for an even field the clamp misplaces one δ⁴ tap by a term that the vanishing
edge derivative makes O(h²); for an odd field the edge derivative is the largest quantity
there and the same tap is wrong at O(h). On u_r = r at the cylindrical axis, which should
produce no sensor at all, the clamp gives μ\* = 1.2e-6 on the axis cell at N = 32, a
spurious viscosity of C_mu·ρ·h² on the cell where every converging case fails; the mirror
gives none.

On the first cell off the axis the clamp's error relative to the exact δ⁴ grows as h⁻², as
an O(h²) relative error predicts, and the mirror reproduces the exact value to every printed
digit. A scratch probe applies `detect_sum!` under `:delta4` at weight power 2 to the two
even fields of the table, comparing the first interior cell against the analytic δ⁴ of the
smooth even extension through r < 0. The self-paired axisymmetric cylinder and the paired
resolved-θ cylinder and spherical origin give identical numbers:

```
N    field        cell 1 clamp   cell 1 mirror   exact          clamp / exact
32   1 + r²       2.031364e-06   2.237789e-19    2.237789e-19
32   exp(−4r²)    8.278876e-06   1.942393e-07    1.942393e-07   42.62
64   1 + r²       1.230085e-07   5.506717e-20    5.506717e-20
64   exp(−4r²)    4.943444e-07   2.920133e-09    2.920133e-09   169.29

cells 2, 3, 4 and 8 read 1.0000 of the exact value under both extensions
```

### The operator probe

At a node-centred wall the half-offset rows make the `:d8` detector second order on an even
field and first order on an odd one, and the `:gaussian` smoother second order; the
node-centred rows reproduce the periodic result to round-off. `bench/archive/sensorwall.jl` prints
|wall − periodic| at a node over the input amplitude, on an even and an odd field that
reflect about the wall nodes. The first table is the low wall under the half-offset rows.

```
even field   cos(πx) + 0.5cos(5πx) + 0.1cos(13πx)
odd field    sin(πx) + 0.1sin(13πx)
both of period 2, reflecting about the nodes at x = 0 and x = 1

operator / field        node   N=49        97          193         orders
:d8, even                  1   4.637e-03   1.075e-03   2.629e-04   2.11 / 2.03
                           2   1.014e-02   2.334e-03   5.725e-04
                           6   1.290e-03   3.221e-04   7.967e-05
:d8, odd                   1   1.535e-02   7.152e-03   3.527e-03   1.10 / 1.02
                           2   3.260e-02   1.553e-02   7.680e-03
                           6   4.180e-03   2.147e-03   1.069e-03
:gaussian, even            1   1.871e-02   5.225e-03   1.345e-03   1.84 / 1.96
                           2   4.315e-03   1.158e-03   2.949e-04
                           3   5.554e-04   1.460e-04   3.696e-05
                           4   4.047e-05   1.040e-05   2.619e-06
:compact, even             1   3.310e-06   1.358e-08   5.375e-11   7.93 / 7.98
```

The high wall reads the low one in every row. The other operators and nodes, on both
parities and at both walls, read:

```
:gaussian    exactly 0 at nodes 5 and 6, its rows reaching four nodes
:compact     between 1.3e-05 and 1.4e-10 at nodes 2 to 6
:delta4      the 1e-15 floor at nodes 1 and 2, exactly 0 at nodes 3 to 6
```

Under the node-centred rows every operator reads round-off at every node of either wall at
every resolution, and `:delta4` and `:compact` are unchanged in every digit. A
one-dimensional cross-check reads the same way, `ring_along!` at the high-pass's own
cancellation floor. At a symmetry plane the same probe reads round-off for every operator on
both parities with no wall rows at all ([the face-centred symmetry
plane](#the-face-centred-symmetry-plane)).

```
node-centred rows, the six nodes of either wall      at most
:d8, even field                                      1.994e-15
:d8, odd field                                       2.545e-15
:gaussian, nodes 1 to 4                              2.776e-16 (exactly 0 at nodes 5 and 6)
one-dimensional check at N = 49
    smoother                                         1.5e-16 to 2.9e-16 relative
    ring_along!                                      1e-12 to 3e-11
```

### What moved

Every smooth-wall row with the artificial properties off is bit-identical, as is
`test/convergence.jl` study by study (every study there runs with the properties off). With
them on, the wall errors of the Brady–Livescu closures fall by one to two orders at the
viscous and shear walls, to within about 1% of their properties-off values. The
`detector = :d8` channel row falls by factors 2.7 to 5.3. The inviscid slip wall under the
strain sensor is then limited by that sensor's cusp rather than by any operator's edge, and
the two default closures move in the third digit.

On the battery the mirror moved the planar Noh wall deficit and the aligned Noh case and
nothing else past the printed digits. In the aligned case the `:delta4` wall mirror alone
raised the transverse round-off by two orders over the clamp's, and the full set brought it
below the clamp's ([the aligned Noh transverse mode](shocks.md#the-aligned-noh-transverse-mode)). The
fold rows moved in the fourth or fifth digit and no guard failed, and spherical Noh moved by
one step at either CFL.

```
aligned Noh, transverse round-off      clamp 1.4e-8, :delta4 wall mirror alone 2.8e-6, full set 7.8e-9

spherical Noh                          clamp at the fold    mirror at the fold
CFL 0.30, steps completed              544                  543
CFL 0.40, negative density at step     105                  106
```

At the cylindrical axis the internal-energy sensor carried most of the clamp's spurious
contribution: between clamp and mirror Σμ\* and Σβ\* move in the eleventh digit while Σκ\*
falls by a factor 5.9. A scratch MPI check with θ split over two ranks agrees with one rank
to round-off on every sum, with no deadlock. No MPI phase runs the detector across a paired
fold, and the butterfly exchange every scalar sensor carries there is untimed.

```
np = 1 against np = 2, relative      Σμ* 1.2e-15     Σβ* 6e-16     Σκ* 5e-16
```

The device path reproduces the host bitwise. `_device_plan` copies the closure rows
generically. On the device below, `ring_plans[1]` holds two distinct plans and
`ring_plans[2]` one aliased plan as on the host, and `mu_art`, `beta_art`, `kappa_art` and
`dQ` after one `compute_rhs!` agree bitwise with the `CPUBackend` solver from the same
standing-wave data, over the interior and over the six nodes nearest each wall, as does the
state after five steps. The comparison repeats under `mu_sensor = :velocity`, which
exercises the odd wall sign, and under the `detector = :delta4` control.

```
AMD RX 6800 XT (gfx1030) through AMDGPU.jl, (48, 16, 16) slab, slip walls on dimension 1
```

In the default configuration the extra rows cost nothing, the pair aliasing one plan. Under
`:d8` with a wall they add one plan per walled dimension and about a percent of the solver's
memory. Construction does not move, and per call the walled configuration is 4 to 8%
cheaper in both runs, inside the run-to-run spread.

```
:d8 at (96, 64, 64)          with a wall    without
distinct ring plans          four           three
packed-line buffer           12.00 MiB      9.00 MiB
Base.summarysize             289.01 MiB     285.50 MiB
```

`bench/jetcheck.jl` reports one dispatch site for the hook, `sensor_mirror` through
`_face_mirror`. Two rejected spellings measured five sites (a bare call whose `Any` return
destroyed the pointwise body's specialization) and one extra `convert` site (a `::Bool`
annotation); the adopted `@noinline` `@nospecialize` form compared with `=== true` holds it
to one. `bench/audit.jl` reads a constant allocation per call in `compute_artificial!`,
`compute_rhs!` and `step!`, three extra scalars captured per threaded region rather than a
cost per point, with every inference row unchanged.

```
jetcheck dispatch sites      compute_rhs! 3, step! 4, every other entry point 0
audit allocation at 48³      +1536 B per call; 0 B at -t 1
```

Two test guards moved with the `:d8` wall rows. The `:d8 through a coordinate-singularity
fold` window narrowed to the inner half, because the test field has slope −0.147 at the
outer slip wall and is not the reflection the wall rows continue it as, which reads 3.6e-3
at the wall node. The `u_r = r` axis check under `:d8` moved from 1e-14 to 1e-12 because
that field is not odd about the outer wall, the slip condition leaves a kink there, and the
pentadiagonal inverse carries a decaying tail of that mismatch to the axis.

## The face-centred symmetry plane

```text
julia --project=. -t 1 bench/archive/wallclosure.jl parts=smooth wall=folded
julia --project=. -t 1 bench/archive/sensorwall.jl wall=folded
julia --project=. -t 1 bench/archive/closurequalify.jl parts=jacobian schemes=neutral3 jns=51,101 jwalls=symmetry,slip
mpiexec -n 4 julia --project=. -t 1 test/mpi_tests.jl "phases=symmetry plane"
```

An inviscid slip wall is a symmetry plane: density, pressure, energy, species and the
tangential velocity are even about it and the normal velocity odd. `SymmetryPlaneBC` places
the plane half a cell outside the end node, on the half-offset grid the coordinate folds
use, every operator running its interior stencil over the mirrored halo with no closure row.
It is the self-paired fold of the axisymmetric axis with the slip wall's parities, on any
dimension whose scale factors do not depend on the folded coordinate; `DESIGN.md` has the
mechanism and the setup rules.

### Mirror equivalence

The folded operator is the periodic operator on the doubled line restricted by parity, so a
run between symmetry planes and the periodic run on the doubled line, offset by half a cell,
differ by round-off for every operator in the step. The `symmetry plane: the mirror of the
periodic run on the doubled line` testset measures that at N = 32 over thirty to forty
steps, compact filter every step at full strength and artificial properties on, relative
maximum norm over every conserved component on the coincident nodes:

```text
[0, 1] at N nodes against [0, 2) at 2N nodes with origin h/2
                  :delta4/:gaussian  :delta4/:compact  :d8/:gaussian  :d8/:compact
lele_d1_6              2.52e-15          3.36e-15        3.36e-15       3.53e-15
lele_d1_8              2.35e-15          3.02e-15        3.19e-15       3.53e-15
lele_d1_10             3.19e-15          3.36e-15        3.02e-15       3.53e-15
viscous, mu0 = 0.005   3.15e-15
2-D, plane on dim 2, tangential velocity   5.54e-15    viscous 4.33e-15
```

Two species with mass fractions even about both planes pass the same guard, set at 2e-14. A
uniform multispecies state between six planes gives a `compute_rhs!` output of exactly zero,
and five steps leave a spread and a drift of 1.9e-14.

### One derivative, one filter pass, and the sensors

Between two planes one derivative converges at the interior order on even and odd fields
alike, and so does a filter pass. The table is `test/convergence.jl`, global maximum norm of
one derivative between two planes on [0, 1], the field even (exp(cos πx)) or odd
(sin(πx) exp(cos πx)) about both planes:

```text
                                 Ns            errors                              order
C6 symmetry planes, even       24/48/96    1.903e-6  2.980e-8  4.656e-10          6.00
C6 symmetry planes, odd        24/48/96    7.732e-6  1.211e-7  1.893e-9           6.00
C8 symmetry planes, even       16/24/32    1.562e-6  5.974e-8  5.914e-9           8.05
C8 symmetry planes, odd        16/24/32    6.724e-6  2.686e-7  2.711e-8           7.95
C10 symmetry planes, even      12/16/24    7.977e-7  4.414e-8  6.667e-10         10.23
C10 symmetry planes, odd       12/16/24    3.919e-6  2.066e-7  3.400e-9          10.17
C8 filter pass, planes         16/24/32/48 4.051e-7  1.711e-8  1.762e-9  7.016e-11  7.88
```

The resolutions fall with the order because a field this smooth reaches round-off quickly.
With a plane at the low end and a node-centred `SlipWallBC` at the high end,
`bench/archive/foldorder.jl` splits the norm: the fold window converges at or above the interior
order, and the maximum sits at the last node of the wall window, which converges at third to
fourth order. Both rows are bitwise identical to the cylindrical-axis rows of the same file.

```text
bench/archive/foldorder.jl     even    odd
fold window            7.01    6.05
wall window            2.99    3.76
```

At a plane every sensor operator reproduces the periodic mirror to round-off, at the level
of the node-centred wall rows of [the sensor operators at
walls](#the-sensor-operators-at-walls). `bench/archive/sensorwall.jl wall=folded` runs the `:delta4`
and `:d8` detectors on an even and an odd field and the `:gaussian` and `:compact` smoothers
on the even field, at a plane against the periodic mirror, on both faces.

```text
N = 49 / 97 / 193, relative departure from the periodic mirror
plane, maximum over every combination and node     9.1e-15
plane, typical entry                               1e-16 to 3e-15, many entries exactly zero
node-centred wall rows                             2.8e-16 (smoother), 2.5e-15 (detector)
```

### The smooth wall matrix

The node-centred rows carry a closure defect that converges near fourth order, and the
folded rows are round-off. The table is `bench/archive/wallclosure.jl parts=smooth wall=folded`, the
wall-window maximum norm at t = 0.4 against the mirror at the same spacing, the default
filter every step, cfl 0.25:

```text
                                        node-centred                          folded
inviscid slip, C6, art off     5.783e-7  3.984e-8  2.588e-9 (3.86/3.94)   4.4e-15  4.2e-15  7.1e-15
inviscid slip, C6, art on      5.666e-7  3.939e-8  2.582e-9 (3.85/3.93)   3.7e-15  3.8e-15  1.0e-14
inviscid slip, C8, art on      5.669e-7  3.936e-8  2.585e-9 (3.85/3.93)   4.7e-15  6.7e-15  7.3e-15
inviscid slip, C10, art on     5.661e-7  3.930e-8  2.584e-9 (3.85/3.93)   5.1e-15  8.9e-15  5.1e-15
viscous slip + shear, C6, on   4.165e-7  3.003e-8  2.010e-9 (3.79/3.90)   3.3e-15  3.2e-15  7.3e-15
viscous slip + shear, C8, on   4.152e-7  3.007e-8  2.016e-9 (3.79/3.90)   4.8e-15  4.4e-15  3.9e-15
viscous slip + shear, C10, on  4.120e-7  2.998e-8  2.014e-9 (3.78/3.90)   3.6e-15  1.3e-15  2.0e-15
```

The largest folded entry over the thirty-six rows is 1.5e-14. A mirror at the same spacing
therefore measures nothing about a plane, and the evolution rows of `test/convergence.jl`
take a five-times-finer folded mirror as their reference, an odd refinement so every study
node is a reference node, carrying 1/625 of the study's step error. The wall window then
reads the run's own total error and equals the interior column in every row:

```text
                                        Ns          errors                        order
inviscid planes, C6, unfiltered        25/49/97   1.719e-9   7.399e-11  4.084e-12  4.46
inviscid planes, C6, one-sided filter  25/49/97   2.353e-9   7.970e-11  4.092e-12  4.69
inviscid planes, C8, unfiltered        25/49/97   8.689e-10  5.900e-11  3.838e-12  4.00
inviscid planes, C10, unfiltered       25/49/97   8.693e-10  5.901e-11  3.844e-12  4.00
inviscid planes, C6, cfl/4             25/49/97   8.587e-10  1.526e-11  2.782e-13  5.93
viscous slip planes, C6, unfiltered    25/49/97   8.751e-10  1.528e-11  2.445e-13  6.04
```

The 4.00 of the C8 and C10 rows is the time integrator. Halving and quartering the step on
the middle grid show the C6 error to be a fourth-order time error over a fixed spatial floor
of 1.5e-11, and the cfl/4 row is that floor's sixth-order sequence. The C6 rows read above
fourth order because that floor is still visible on the coarsest grid, where they sit
twice as high as C8 and C10, and the gap closes with refinement. The viscous row's step is
diffusion-limited and shows the spatial order directly.

```text
C6, middle grid, half the step         1.87e-11
C6, middle grid, a quarter of the step 1.53e-11
C6 above C8 and C10                    N = 25: twice    N = 49: 25%    N = 97: 6%
```

The ladder stops at its present finest grid because one level finer the step's accumulated
round-off differs between hosts. Two Julia versions on the workstation agree to every
printed digit, so the spread is the hardware's arithmetic and not the compiler's. At the
finest grid of the ladder in use the two hosts differ by 0.16% and the fitted order by
0.001.

```text
C10 row on the ladder 49/97/193     middle      finest      order
workstation                         3.844e-12   2.580e-13   3.96
GitHub runner                       3.838e-12   2.456e-13   4.00
finest-grid difference 2.5e-13; the N = 49 column identical to four digits
Julia 1.11.4 and 1.13.0 on the workstation agree to every printed digit
```

### The production Jacobian

The plane's radius is one to the finite-difference noise, and its filtered rate is six to
thirty times below the node-centred slip wall's, the one configuration in which the neutral
rows show a measurable positive rate. The table is `bench/archive/closurequalify.jl parts=jacobian
jwalls=symmetry,slip`, the finite-differenced `step!` about a uniform state, radius and rate
per unit time at three differencing steps:

```text
δ = 3e-6 / 1e-5 / 3e-5
 N   wall      filter   radius − 1                    rate
 51  symmetry  off      2.5e-10  7.6e-11  1.9e-11     +3.2e-8  +1.0e-8  +2.5e-9
 51  symmetry  on       1.3e-10  9.0e-11  5.0e-12     +1.6e-8  +1.2e-8  +6.5e-10
 51  slip      on       3.9e-9   1.1e-9   3.2e-10     +5.1e-7  +1.5e-7  +4.2e-8
101  symmetry  on       1.3e-10  1.7e-10  3.2e-11     +3.4e-8  +4.4e-8  +8.4e-9
101  slip      on       3.8e-9   9.7e-10  2.1e-10     +9.9e-7  +2.5e-7  +5.6e-8
```

C8 and C10 measured through `production_jacobian` directly read the same way, against
filtered slip-wall counterparts one to two orders higher. The exact linear model of the
folded operator, antisymmetric with an isometric semigroup, is in [closure
certificates](#closure-certificates).

```text
production_jacobian at δ = 1e-5, radius − 1     unfiltered    filtered
C8, N = 51                                      7.6e-11       1.0e-11
C8, N = 101                                     1.2e-10       1.7e-10
C10                                             8.1e-11       3.9e-11
C10                                             1.1e-10       1.6e-11
filtered slip-wall counterparts                               2.7e-9 at both N
```

### MPI and device parity

The `symmetry plane` phase builds every case twice, on the process grid and whole on
`MPI.COMM_SELF`, comparing the rank's interior block against the serial rebuild for the
`compute_rhs!` output and the state after filtered steps: planes at both ends of the split
dimension on each axis in Float64 and Float32 with two species, physical viscosity,
`detector = :d8` on one axis and `species_flux = :bulk` on another; a corner with planes at
both ends of two split dimensions; and the Noh layout, `DirichletBC` low and the plane high.
Every check passes far inside its tolerance, with two ranks at np = 4 owning neither plane
and still reaching the folded solve.

```text
22 checks at np = 2 and np = 4     residual              tolerance
Float64                            7.1e-15 to 2.3e-14    1e-8
Float32                            2.4e-7 to 6.7e-6      5e-4
```

Device plans inherit the fold from the wrapped host plan's factorization, and the two device
runs of `test/device_tests.jl` are bitwise against the CPU solver.

### The battery at a plane

Against the node-centred wall the plane improves Woodward–Colella by 6% in L1 with the
contact where it was, reads the planar Noh wall deficit one point higher at six times less
negative a wall-layer internal energy, cuts the aligned Noh deficit by five points, and
holds the aligned case's transverse round-off four hundred times smaller ([the shock
battery](shocks.md#the-shock-battery)). That row's uniformity guard is therefore 5e-9 where the
node-centred row's is 5e-7. Every pre-existing row reads its recorded value to four digits.
The planar deficit repeats the earlier finding: the wall heating of Noh is a property of the
captured shock's start at the wall, not of the closure rows.

`test/cases.jl` gives `woodward`, `noh_case` (ν = 1) and `noh_aligned` a `folded` keyword
that places `SymmetryPlaneBC()` where the `SlipWallBC` was, the spacing following the grid
(h = L/N between two planes, L/(N − ½) from a plane to the Dirichlet inflow) and feeding the
collapsed extents and the blend width, so the two runs differ in the wall placement alone.
`test/validation.jl` runs each folded case beside its original under the same bounds. The
wall deficit is sampled at the first node on both grids, the wall itself node-centred and
half a cell inside the plane folded; `noh_plane_density`, the even continuation
(9ρ₁ − ρ₂)/8 onto the plane, prints beside it.

## The inflow transverse terms

```text
julia --project=. -t 1 bench/archive/nscbcinflow.jl pulse vortex outflow betas=0,0.15,0.3,0.5,0.7,0.85,1 etas=4,16,64
julia --project=. -t 1 bench/archive/nscbcinflow.jl pulse vortex outflow M=0.1 betas=0,-1,0.9,1 etas=16
julia --project=. -t 1 bench/archive/nscbcinflow.jl pulse vortex outflow M=0.5 betas=0,-1,0.5,1 etas=16
julia --project=. -t 1 bench/archive/nscbcinflow.jl vortex betas=0,0.5,1 etas=256,1024
```

The transcript of the four commands is `bench/archive/results/nscbcinflow.txt`.

Every imposed amplitude of `NSCBCInflowBC` carries `−beta_t·𝒯`, the
weighted transverse contribution of its characteristic, except the transverse
pressure gradient ∇_t p/ρ in the transverse-velocity amplitudes, which takes
min(`beta_t`, M) (the derivation is in `src/physics/nscbc.jl`); `NSCBCOutflowBC` carries
the same weight on its one incoming wave. The instrument measures both faces on a
two-dimensional stream, artificial properties off and the default filter every
step, against the same run on a domain extended two units past the face under
test, where the disturbance meets no boundary in the run: the difference is the
face's reflection or the error of its imposition. A pressure pulse released half a
unit inside the face meets it at every incidence angle. Its pressure deviation is
scaled by the largest excursion the extended run carries across the plane of the
face, and the vorticity entering through the face, taken by second-order
differences, by the pulse's acoustic velocity over its radius. An isentropic vortex
enters through the inflow from a time-dependent target, the analytic vortex at the
plane, or leaves through the outflow; its deviation is scaled by its peak velocity
and by its pressure depression. The vortex rows run to the time the centre reaches
one unit inside the face; the pulse rows to the time the reflection is half a unit
back inside.

```
stream             Mach 0.3, γ = 1.4, p = ρ = 1, 64 points per unit length, periodic width 1
pressure pulse     amplitude 1e-3, radius 0.1
  scale of p       the extended run's largest excursion across the face's plane:
                   2.1e-4 at the inflow, 2.6e-4 at the outflow
isentropic vortex  peak velocity 0.3 of the stream, radius 0.1
  scale of p       its pressure depression, 1.5e-2
```

### The pulse at the inflow

The pressure reflection is least at an intermediate weight and the entering vorticity
least at the full share. The reflection is least at a weight of 0.5, a factor 9 under the
LODI form. It rises from there to the full share, still 2.4 times under the LODI form. The
vorticity falls monotonically above the Mach number, to a tenth of the LODI form's and a
fifth of the Mach number's at the full share; below the Mach number it is not monotone in
the weight. Above the Mach number the pressure part keeps the weight M and only the rest of
𝒯 follows `beta_t`. The table gives the maximum and root-mean-square deviation of p over
the region the reflection has crossed, over the incident amplitude, and of the vorticity,
over the pulse's acoustic velocity over its radius, at the default relaxation rates:

| `beta_t` | max p | rms p | max ω | rms ω |
|---|---|---|---|---|
| 0 (LODI) | 0.220 | 0.045 | 0.345 | 0.042 |
| 0.15 | 0.126 | 0.025 | 0.105 | 0.016 |
| 0.3 (= M) | 0.047 | 0.0093 | 0.178 | 0.023 |
| 0.5 | 0.025 | 0.0057 | 0.123 | 0.016 |
| 0.7 | 0.043 | 0.0089 | 0.075 | 0.0094 |
| 0.85 | 0.069 | 0.013 | 0.038 | 0.0051 |
| 1 | 0.093 | 0.016 | 0.035 | 0.0030 |

### The vortex at the inflow

The vortex enters only under a fast enough relaxation, and at every rate the full share
admits it best. At the default rates, η = 0.28, the vortex does not enter at all under any
weight: the error is the vortex itself. A relaxation admits a structure only when its time
L_ref/(η c) is short against the passage time r_v/u_0, and the error then falls as 1/η up
to the largest rate run, with no stiffness at η = 1024. At every rate the full share is the
best weight, by a factor 3 to 4 over the LODI form at the three highest rates, with the
error in u falling monotonically in the weight and the pressure error falling fastest.

```
passage time r_v/u_0        0.28
relaxation time L_ref/(η c) 6 at the default rate, 0.03 at η = 64
```

The table gives the maximum and root-mean-square deviation of u over the peak velocity,
the maximum of v likewise, and the maximum of p over the pressure depression, once the
centre is one unit inside:

| `eta` | `beta_t` | max u | rms u | max v | max p |
|---|---|---|---|---|---|
| 4 | 0 | 0.93 | 0.168 | 1.21 | 0.97 |
| 4 | 0.5 | 0.83 | 0.152 | 1.22 | 0.95 |
| 4 | 1 | 0.64 | 0.107 | 1.09 | 0.83 |
| 16 | 0 | 0.52 | 0.107 | 1.22 | 0.67 |
| 16 | 0.5 | 0.38 | 0.078 | 0.97 | 0.55 |
| 16 | 1 | 0.28 | 0.044 | 0.59 | 0.36 |
| 64 | 0 | 0.22 | 0.036 | 0.44 | 0.40 |
| 64 | 0.3 | 0.17 | 0.028 | 0.35 | 0.31 |
| 64 | 0.5 | 0.14 | 0.023 | 0.30 | 0.25 |
| 64 | 0.7 | 0.10 | 0.018 | 0.25 | 0.18 |
| 64 | 1 | 0.068 | 0.012 | 0.16 | 0.091 |
| 256 | 0 | 0.061 | 0.0093 | 0.11 | 0.11 |
| 256 | 0.5 | 0.038 | 0.0058 | 0.074 | 0.065 |
| 256 | 1 | 0.017 | 0.0030 | 0.042 | 0.024 |
| 1024 | 0 | 0.016 | 0.0024 | 0.027 | 0.028 |
| 1024 | 0.5 | 0.011 | 0.0015 | 0.018 | 0.017 |
| 1024 | 1 | 0.0052 | 0.0008 | 0.010 | 0.0069 |

### Across the Mach number

The ordering is the same at every Mach number: the Mach-number weight reflects the pulse
least, and the full share admits the vortex best, by a factor 6 over the LODI form at
Mach 0.1, where the Mach-number weight is nearly the LODI form. Under the full share the
pulse carries in between a half and a tenth of the LODI form's vorticity, and the
reflection is less than the LODI form's at every Mach number. The table repeats the two
disturbances at three Mach numbers, the domain and the disturbances unchanged and η = 16
for the vortex: the pulse's maximum deviation over its incident amplitude and maximum
vorticity over its acoustic velocity over its radius, and the vortex's maximum u over its
peak velocity and maximum p over its pressure depression.

| M | `beta_t` | pulse max p | pulse max ω | vortex max u | vortex max p |
|---|---|---|---|---|---|
| 0.1 | 0 | 0.158 | 0.110 | 0.55 | 0.94 |
| 0.1 | M | 0.093 | 0.132 | 0.52 | 0.86 |
| 0.1 | 1 | 0.104 | 0.054 | 0.092 | 0.121 |
| 0.3 | 0 | 0.220 | 0.345 | 0.52 | 0.67 |
| 0.3 | M | 0.047 | 0.178 | 0.45 | 0.60 |
| 0.3 | 1 | 0.093 | 0.035 | 0.28 | 0.36 |
| 0.5 | 0 | 0.376 | 0.319 | 0.66 | 0.92 |
| 0.5 | M | 0.025 | 0.097 | 0.58 | 0.79 |
| 0.5 | 1 | 0.081 | 0.039 | 0.45 | 0.61 |

### The same disturbances at the outflow

The outflow's reflection of the pulse is least at the Mach number in the root-mean-square
and at half of it in the maximum, and the vortex leaves most cleanly at 1 − M, where the
pressure error is a fourteenth of the LODI form's and a fifth of the default's:

| `beta_t` | pulse max | pulse rms | vortex max u | vortex rms u | vortex max v | vortex max p |
|---|---|---|---|---|---|---|
| 0 (LODI) | 0.106 | 0.015 | 0.033 | 0.013 | 0.045 | 0.69 |
| 0.15 | 0.060 | 0.0079 | 0.021 | 0.0088 | 0.034 | 0.50 |
| 0.3 (= M, default) | 0.097 | 0.0067 | 0.015 | 0.0065 | 0.024 | 0.27 |
| 0.5 | 0.128 | 0.014 | 0.0093 | 0.0043 | 0.010 | 0.10 |
| 0.7 (= 1 − M) | 0.194 | 0.022 | 0.0061 | 0.0026 | 0.0012 | 0.050 |
| 0.85 | 0.242 | 0.028 | 0.057 | 0.0071 | 0.067 | 0.098 |
| 1 | 0.284 | 0.033 | 0.022 | 0.0034 | 0.019 | 0.10 |

At Mach 0.1 the Mach number is the best weight for both disturbances. At Mach 0.5 the
constant 1 − M equals the stream's Mach number but not the local Mach number under the
vortex. There the LODI form reflects the pulse least and 1 − M carries the vortex out best.
The outflow default stays at the Mach number, since the outflow's task is the pulse's: the
weight 1 − M, which carries a vortex out best at the two higher Mach numbers, reflects the
pulse twice as strongly as the Mach number at Mach 0.3.

| weight | Mach 0.1: pulse reflection | vortex pressure error | Mach 0.5: pulse reflection | vortex pressure error |
|---|---|---|---|---|
| LODI form | 0.187 | 0.032 | 0.052 | 0.48 |
| the Mach number | 0.117 | 0.0060 | 0.131 | 0.19 |
| 1 − M | 0.247 | 0.020 | 0.131 | 0.064 |
| the full share | 0.273 | 0.11 | 0.279 | 0.37 |

### The decision

`NSCBCInflowBC` defaults to `beta_t = 1`, the full share, with the transverse
pressure gradient of the transverse-velocity amplitudes at min(`beta_t`, M).
The full share is the form the literature restates for an inflow. It admits
a vortex best at every relaxation rate and Mach number measured, and at
Mach 0.3 a pulse leaving through the face carries in the least vorticity
under it. Under it the incoming characteristic variables also follow their
targets through a transverse flow at the face, the transverse velocity up to
the rate −(1 − M)∇_t p/ρ that an acoustic wave leaving obliquely gives it
there; `test/serial/rhs_boundaries.jl` asserts this at both faces. The weight M on the
pressure part leaves exactly that rate. Under the full weight on it, u_t is held
against an obliquely leaving acoustic wave at the face and a vorticity wave enters
the domain, as where a pressure pulse converging on the axis crosses the top face
of `examples/vortex_ring_shock.jl`. On this instrument the weight M lowers the
default's pulse reflection by about a quarter and raises the entering vortex's
error by a few percent.

```
vortex_ring_shock.jl, azimuthal vorticity carried into the tube
full weight on the pressure part     2500 1/s
weight M                             at most 100 1/s

inflow default, pressure part at     full weight   weight M
pulse reflection                     0.128         0.093
vortex error at η = 16, u            0.262         0.280   (7%)
vortex error at η = 16, p            0.355         0.362   (2%)
```

The default's pulse reflection is still twice that of `beta_t = -1` and almost four times
that of a weight of 0.5. For an inflow that admits no structure and faces outgoing waves,
`beta_t = -1` is the better weight. `NSCBCOutflowBC` keeps the Mach number.

## Fold order and geometry limits

```text
julia --project=. -t 1 bench/archive/foldorder.jl
```

`test/convergence.jl` reports a global max norm, and every one of its fold studies closes
the outer end with a `SlipWallBC` whose closure rows measure about 3 on their own. Splitting
that norm by region, on the same fields and resolutions, separates the two ends:

```
study                              fold(1:3)   mid    outer(3)   global argmax
C6, both ends walls (control)         3.23     3.19     3.17       i = n
cylindrical axis, odd  (u_r-like)     6.05     3.76     3.71       i = n
cylindrical axis, even (scalar)       7.01     2.99     3.00       i = n
spherical origin, even (scalar)       7.00     2.97     2.99       i = n
spherical origin, odd  (u_r-like)     6.07     3.86     3.81       i = n
```

**The global maximum sits at the outer wall in every fold study.** The fold's own error
converges at the interior order or above and is three to five orders of magnitude below the
interior error:

```
spherical origin, N = 96     fold window 7.0e-12, middle of the line 1.9e-7
```

Every global error `test/convergence.jl` prints equals the outer-window norm to every digit
printed, so the guarded numbers in that file are measurements of the outer wall taken
through a norm insensitive to the fold. With walls at both ends the same window reports
3.23, so the split does detect a third-order closure where one is present. The middle of the
line converges at 3 as well, the compact scheme's line-global coupling carrying the wall's
closure error inward rather than a property of the fold. Both parities were measured,
including the odd one that `test/convergence.jl` does not cover and that a converging
calculation differentiates at the origin. A face-centred symmetry plane reads the same
fold-window orders bitwise ([the face-centred symmetry
plane](#the-face-centred-symmetry-plane)).

**A blast at the spherical origin carries a lower CFL ceiling than at the axis.** Under
∂p/∂r the ceiling is 0.2 to 0.3 on the larger blast, which fails at the same time in either
form of the pressure term, well after the start. On the smaller blast it is 0.3 to 0.5.
The area form of the radial pressure term requires initial data resolved over three cells
or more ([grid-scale growth](#grid-scale-growth-at-the-r-z-axis)): in that form the small
top hat fails within tens of steps at every edge width to three cells, and under ∂p/∂r a
sharp edge completes. `test/cases.jl` initializes Sedov with a Gaussian deposit.

The case is a top hat of the Sedov deposit's energy with a tanh edge of w cells, run without
retries; "area form" restores the area form of the radial pressure term through
`bench/archive/axisvariant.jl`, and the axis and the plane take the same p_in over the same line.
With retries both blasts complete, at half the starting CFL.

```
p = 1e-5 + p_in inside r₀; N = 256 over (0, 1.2]; t = 0.2
r0     w        cfl    origin                    area form               axis, plane
0.03   0        0.3    completes                 fails at step 3
0.03   0        0.15                             fails at step 144
0.03   1, 2, 3  0.3    completes (w = 1)         fails at 42, 32, 46
0.03   1        0.5    fails at step 18          fails at step 11        complete
0.1    1        0.15   completes                 completes
0.1    1        0.2    completes
0.1    1, 4     0.3    fails at t = 0.046, 0.050 fails at t = 0.046      complete
0.1    8        0.3    completes
0.1    1        0.5    fails at t = 0.045        fails at t = 0.045      complete

with retries = 4     from cfl    completes at
larger blast         0.3         0.15
smaller blast        0.5         0.25
```

**The singular t = 0 start of Noh completes at the origin at cfl 0.15**. It returns a
plateau 18% low ([the singular start](shocks.md#the-singular-start-and-the-warm-start)), since the
exact solution requires 64× compression to appear at the origin instantaneously. In the area
form it completes only at cfl 0.05. A warm start from the exact solution at t = 0.3
integrates to 0.6, testing maintenance of the solution through the origin without the
initialization singularity. The cylindrical axis accepts the cold start at 16× compression.

### Mass through the cylindrical axis face

```text
julia --project=. -t 1 bench/archive/axismass.jl
```

The transcript is `bench/archive/results/axismass.txt`.

Under the node weights W of the positivity limiter (h on the fold's half of the line,
the closed line's tail at the far end) the folded divergence telescopes to
Σ W (D g) = g_N − Ĝ_0, Ĝ_0 the face value at the fold, so the right-hand side changes
Σ W J ρ by Ĝ_0 alone on a line closed by a slip wall. Ĝ_0 is round-off (under 5e-16) for
the odd area-weighted fluxes at the spherical origin and at a face-centred symmetry plane.
At the cylindrical axis the even mass and energy fluxes leave a face that matches its
prediction to the third or fourth digit. The table measures them from one right-hand side
on a converging profile, per radian, with "corrected" the mass face less (h²/24) dρ_1/dt:

```
profile ρ(0) = 1.5, u_r'(0) = −1; prediction −h² ρ u_r'(0)/12, (E + p) in place of ρ for energy
N      mass face    /prediction   /node-1 rate   corrected    energy face /prediction
64     +3.103e-05   1.0008        0.0835         +6.456e-08   1.0007
128    +7.691e-06   1.0002        0.0834         +3.995e-09   1.0002
256    +1.915e-06   1.0001        0.0833         +2.481e-10   1.0000
```

The face carries one twelfth of node 1's mass rate W_1 J_1 dρ_1/dt at every resolution,
which is the ratio below for any smooth flow; on this profile it is one sixth of node 1's
mass per unit time. The even folded divergence annihilates a constant to round-off, where
conservation would need Σ W (D 1) = 1, so no node weights, positive or not, conserve the r-z
mass exactly. With θ resolved the faces of all lines sum to the collapsed value times 2π,
within a percent on the axis-crossing vortex. The sum is round-off with the centre on the
axis, where ∇·(ρu)(0) = 0.

```
ratio of the face to node 1's rate     −h² ρ u_r'(0)/12 to (h²/2)(−2ρ u_r'(0))
constant through the folded divergence max |D 1| ≤ 2.8e-14
θ resolved, sum of the faces           −(π h²/12) ∇·(ρu)(0)
vortex centred at x = −0.15            ratio 1.0072, 1.0020 and 1.0005 at N = 32, 64 and 128
```

**The face is the rate of change of the quadrature's own error at the axis.** The fold side
of W J is the midpoint rule on f = r q, and at q = ρ the rate of change of its
Euler–Maclaurin error is the face. Subtracting (h²/24) q_1 from the W J total leaves the
fourth-order remainder of the corrected column above.

```
Euler–Maclaurin error      (h²/24) f'(0) = (h²/24) q(0)
its rate at q = ρ          −(h²/24) ∇·(ρu)(0) = −h² ρ u_r'(0)/12, the face
corrected node-1 weight    11h²/24 per radian in place of h²/2
```

The mass drift of the closed r-z runs is the axis face. In every run, filter and artificial
properties each on or off, the right-hand side's part under W J equals the time integral of
the face to the accuracy of the trapezoid rule in time, and the corrected part falls by a
factor of 14 to 44 over the last doubling; the energy totals follow the same pattern. On
the spherical origin, the symmetry plane and the line between two slip walls the same mass
column is round-off (under 2e-14), and the drift under W J is the filter's alone. The table
gives closed runs to t = 1, the relative change of the total mass: the first column is
`volume_integral`'s total under the edge-corrected weights described below, the others the
right-hand side's part under W J (the filter's share removed), the time integral of Ĝ_0
over the steps, and the corrected total's right-hand-side part.

```
slip wall at r = 1, cfl 0.5, the default filter and artificial properties
case     N     vol. int.   W J rhs     ∫ Ĝ_0 dt    corrected rhs
pulse    64    +4.81e-07   -6.27e-06   -6.26e-06   -2.36e-08
pulse    128   +1.47e-08   -1.55e-06   -1.55e-06   -1.46e-09
pulse    256   +5.89e-10   -3.87e-07   -3.87e-07   -9.10e-11
implode  64    -2.21e-07   -8.40e-06   -8.40e-06   -5.72e-10
implode  128   -1.31e-08   -2.09e-06   -2.09e-06   -4.28e-11
implode  256   -7.24e-10   -5.19e-07   -5.19e-07   -3.03e-12

all 24 r-z runs: W J rhs against ∫ Ĝ_0 dt, 1.6e-3 relative at N = 64 and 4.8e-5 at 256;
the last doubling is N = 128 to 256
```

**The diagnostics' edge correction.** `volume_integral`, `domain_volume`, the conserved
budget and its ledger, and the positivity floor's tally take the same correction in a form
that reads only the metric (the quadrature note in `src/solver/diagnostics.jl`). Their rule is the
trapezoid rule on f = J q in ξ, the midpoint rule along a folded dimension, whose
Euler–Maclaurin edge terms are −(h²/12) ∂ₙf and +(h²/24) ∂ₙf with ∂ₙf = q ∂ₙJ + J ∂ₙq;
the edge node's weight removes q ∂ₙJ, with ∂ₙJ from the node and its two inward neighbours.
At the axis that is the corrected node weight above, 11/12 of the midpoint's. At the
spherical origin ∂ₙJ = 0 and nothing changes. At a curved or stretched node-centred face
the half weight becomes ½(1 + h ∂ₙJ/(6J)). The term left is the Cartesian trapezoid's
−(h²/12) J ∂ₙq, zero for a field even about the face.

The corrected rule is fourth order on every geometry for a field even about its faces, and a
field with a normal derivative at a node-centred face keeps the second order of the
Cartesian rule. On exp(−r²) the uncorrected rule errs less, its axis and wall terms having
opposite signs. The table gives errors of smooth fields against their exact integrals
(`parts=quadrature`) along r, and along θ between the poles:

```
case                                 N = 64      N = 128     N = 256     order
axis, q = 2 + cos πr                 +2.845e-09  +1.751e-10  +1.086e-11  4.01
axis, q = exp(−r²)                   -1.520e-05  -3.772e-06  -9.392e-07  2.01
annulus, q = 2 + cos π(r − ½)        -5.222e-09  -3.162e-10  -1.945e-11  4.02
origin, q = 2 + cos πr               -5.059e-09  -3.112e-10  -1.930e-11  4.01
poles, q = 2 + cos π(r − ½) cos 2θ   -9.505e-07  -5.947e-08  -3.718e-09  4.00
stretched, q = 2 + cos πx            -2.996e-07  -1.770e-08  -1.075e-09  4.04
stretched, q = exp(x)                +4.581e-05  +1.122e-05  +2.781e-06  2.01

exp(−r²) at the axis: uncorrected error h²/24 − h²/(12e) against h²/(6e)
```

In the closed runs the corrected totals at the spherical origin change by no more than the
trapezoid total of a pulse between two Cartesian slip walls, whose drift is the trapezoid
weights' departure from W at the walls:

```
closed runs, change of the total                    three resolutions, coarse to fine
spherical implosion, corrected                      +3.00e-8    −3.01e-9    −2.54e-10
spherical pulse, corrected                          +7.96e-7    +3.70e-8    +1.48e-9
pulse between two Cartesian slip walls, trapezoid   +4.31e-6    +3.98e-7    +2.49e-8
```

**Decision.** The options and their size:

- Carry the axis face explicitly: subtracting Ĝ_0 at node 1 conserves Σ W J ρ exactly but
  changes node 1's rate by one twelfth at every resolution, an order-one error at the node
  whose fold windows above converge at high order; spreading it over a fixed region instead
  adds an O(h²) error at every node of the region. The face value itself is local to the
  rank holding the fold, through the mirrored face relation there.
- Change the quadrature: the end correction (h²/24) q_1 conserves the r-z totals to fourth
  order with no change to the solution.
- Leave it: the second-order drift is the change of the reporting quadrature's leading
  error term, and the corrected total shows the solution's mass conserved to fourth order.

The scheme is left as it is: the measured drift is a property of the quadrature, and
carrying the face would replace it with an order-one error in node 1's rate. The
diagnostics take the end correction at every edge whose Jacobian varies along the line, as
above.

## Grid-scale growth at the r-z axis

The instruments are `bench/archive/axisspectrum.jl` and `bench/archive/axisrunaway.jl`. The
first is the one-step map of an unrefined line from the axis to r = 2,
linearized by central differences about ρ = p = 1 in ρ, ρu_r (and ρu_θ on
a resolved θ) and E at every node. The second runs the converging pulse of
`axis_level_case` and cylindrical Noh with the grid-scale content of the first
16 nodes read after every step. The package's form of the radial pressure
term is ∂p/∂r on θ-collapsed r-z and under the spherical metric
(`_radial_pressure_gradient`), and the area form on a resolved cylindrical θ; the spherical θ-momentum takes (1/r)∂p/∂θ
(`_polar_pressure_gradient`). The other forms are source terms and the
near-axis filter a callback (`bench/archive/axisvariant.jl`): `areap` restores the
area form on θ-collapsed r-z and at the spherical origin, `gradp` takes ∂p/∂r
on a resolved θ, `areatheta` restores the area form of the θ-momentum at the
poles, `product` writes every radial divergence of the mass, radial momentum
and energy fluxes as D(F) + F/r.

**The mechanism.** In the area form the radial momentum carries the pressure
as (1/r)D(r p) − p/r. On a mode of wavenumber ω under a slowly varying
envelope it differs from D(p) by (1 − k'(ω)) p/r, k' being the derivative of
the C6 modified wavenumber. The factor vanishes on smooth modes and is large
near the grid scale. The leading eigenvector of the area form's map peaks at
node 1, is acoustic and repeats every three nodes. The Cartesian line with a
symmetry plane at its end has no eigenvalue above 1 + 1e-8 in any
configuration tried.

```
ω                     1      1.5    2π/3   2.5    π
1 − k'                0.00   0.05   0.56   2.0    16/3

leading eigenvector at αf = 0.47: δE/δρ = 3.5, E +1.00, −0.33, −0.24, +0.35, smooth share 0.47
```

**The rate.** log|λ| per step of the leading eigenvalue; "neutral" is below
1e-8. Under the area form the rate per step is independent of N, so the rate
per unit time is proportional to N:

```
area form, αf = 0.47, cfl 0.18, filter_cfl = 0.35: log|λ| = 7.032e-3 per step at every N
N                 64      128     256
per unit time     1.47    2.95    5.90
```

At the smaller steps the growth sets in between αf = 0.46 and 0.465, and it
rises with αf and with the step:

| area form, αf, `filter_cfl` 0.35 | cfl 0.1 | 0.15 | 0.18 | 0.3 | 0.5 |
|---|---|---|---|---|---|
| 0.45 | neutral | | neutral | neutral | 0.0104 |
| 0.46 | neutral | neutral | neutral | neutral | 0.0292 |
| 0.465 | | 0.0021 | | | |
| 0.47 | 0.0040 | 0.0059 | 0.0070 | 0.0106 | 0.0511 |
| 0.475 | | 0.0101 | | | |
| 0.48 | 0.0098 | 0.0149 | 0.0180 | 0.0301 | 0.0775 |
| 0.49 | | 0.0267 | | | |

At full strength and the solver's own step the area form grows faster still,
and unfiltered it grows at a rate per unit time that rises only slowly with N:

```
area form                                     αf     rate
full strength (filter_cfl = 0), cfl 0.9       0.45   0.089 per step
                                              0.47   0.142 per step; a rest state seeded at
                                                     1e-10 fails in 146 steps
unfiltered, N = 64                                   1.34 per unit time
unfiltered, N = 128                                  1.51 per unit time
```

Under ∂p/∂r the filtered map is neutral in every configuration tried, and
unfiltered it grows from the axis node at about a quarter of the area form's
rate. `product` is neutral at 0.47, cfl 0.18. With viscosity 0.01 the map in
that configuration is neutral in both forms, so the viscous hoop term τ_θθ/r was
left in the area form without a measurement that separates it.

```
∂p/∂r
neutral       αf 0.47 and 0.48, cfl 0.18 and 0.5, N = 64 and 128;
              full strength at cfl 0.9 under αf 0.45 and 0.47
unfiltered    0.34 and 0.40 per unit time at N = 64 and 128, cfl 0.18; 0.24 at cfl 0.9;
              smooth share 0.51 to 0.74
```

**The runs.** The area form at αf of 0.47 and above stops on the finer grids,
the sooner the finer the grid, where ∂p/∂r completes. The pulse runs at the axis
rows' step, cfl 0.177, and stops where the grid-scale content reaches 5%. On a
completed run the density at node 1 exceeds the ambient by 0.1161.

| form | αf | nodes | end, run to t = 0.4 |
|---|---|---|---|
| area | 0.45, 0.46 | 6908 | complete, content 2.1e-6 |
| area | 0.47 | 2303 | complete, content 5.3e-5 |
| area | 0.47 | 3455, 4607, 6908 | stopped at t = 0.392, 0.300, 0.204 |
| area | 0.48 | 1151, 1535, 2303, 3455 | stopped at t = 0.387, 0.308, 0.202, 0.139 |
| ∂p/∂r | 0.47 | 6908 | complete, content 2.1e-6 |
| ∂p/∂r | 0.48 | 3455 | complete, content 1.6e-5 |
| symmetry-plane line | 0.48 | 3455 | complete, content 8.6e-6 |
| area, cfl 0.9 | 0.47 | 767 | stopped at t = 0.355 |
| area, cfl 0.9 | 0.45 | 767 | complete |
| ∂p/∂r, cfl 0.9 | 0.47 | 767 | complete, content 9.5e-4 |

The symmetry-plane line completes, and at the solver's own step the area form
stops under the larger αf where ∂p/∂r completes. Cylindrical Noh completes at
every setting, with the plateau and the front unchanged by αf and by the form.
The axis deficit 1 − ρ₁/16 moves: under the area form it shrinks as αf rises,
most on the finer grids, while under ∂p/∂r it barely depends on αf.

| form, αf (cylindrical Noh, cfl 0.15) | 767 | 1535 | 3071 |
|---|---|---|---|
| plateau, every setting | 15.6523 | 15.8163 | 15.8985 |
| area, 0.45 | 50.1% | 48.9% | 46.0% |
| area, 0.47 | 46.6% | 41.0% | 40.1% |
| area, 0.48 | 46.5% | 33.0% | 25.1% |
| ∂p/∂r, 0.45 | 51.3% | 48.9% | 46.5% |
| ∂p/∂r, 0.47 | 50.9% | 48.4% | 46.0% |
| ∂p/∂r, 0.48 | 50.5% | 48.0% | 45.7% |

**The remedies.** Measured as source terms before the package took ∂p/∂r.
The near-axis filter takes αf = 0.45 over the first eight nodes, tapering to
the run's αf at node 16.

| | map, filtered | map, unfiltered | pulse at cfl 0.177 |
|---|---|---|---|
| `product` | neutral at 0.47 and 0.48, cfl 0.18 | neutral at cfl 0.9, 0.19 per unit time at 0.18 | not run |
| near-axis 0.45 | neutral under 0.47 and 0.48 at cfl 0.18; 0.0100 at cfl 0.5 | | 0.47 on 6908 nodes completes, 0.1161 |

The axis convergence rows (errors on three grids and the order); the annulus
and corner rows moved with them, and `test/convergence.jl` carries the ∂p/∂r
values:

| errors at N = 96, 192, 384 and the order | inviscid | viscous |
|---|---|---|
| area form | 4.594e-7, 2.986e-9, 3.012e-11, 6.93 | 1.078e-6, 1.164e-8, 1.429e-10, 6.42 |
| ∂p/∂r | 3.693e-7, 1.555e-9, 4.821e-11, 6.43 | 9.873e-7, 1.034e-8, 1.229e-10, 6.47 |
| `product` | 3.966e-7, 2.032e-9, 3.905e-11, 6.64 | 8.762e-7, 8.621e-9, 9.599e-11, 6.56 |

The rows filter at αf = 0.45, so the near-axis filter leaves them unchanged.
The Noh axis rows of `test/validation.jl` moved from the area form's deficits
below to the values its header carries, and the fronts moved by at most 1e-4.

```
Noh axis rows, area form                        deficit
256 nodes                                       57.1%
uniform 767, and the level over N/3 root nodes  46.6%
level over N/6 + 1                              59.9% at plateau 14.8044
near-axis filter, 256 nodes                     55.1%
near-axis filter, 767                           50.2%
```

**The resolved θ, the spherical origin and the poles.** The same map on three
lines, each in its area form and under ∂p/∂r:

```
resolved θ   12 nodes over the full circle, N = 32 and 64; the area form in the package,
             ∂p/∂r through gradp
origin       the spherical line with θ and φ collapsed; area form (1/r²)D(r² p) − 2p/r,
             which areap restores
poles        a θ line through both poles at r = 1 with r and φ collapsed, N = 32 and 64;
             area form (1/(r sinθ))D(sinθ p) − (cotθ/r) p, which areatheta restores
```

| | area form | ∂p/∂r |
|---|---|---|
| resolved θ, unfiltered, cfl 0.18 | 1.41 per unit time | 0.34 per unit time |
| resolved θ, 0.47, cfl 0.18 and 0.25 | neutral | neutral |
| origin, unfiltered, cfl 0.18 | 2.2e-3 per step (N = 64), 2.0e-3 (128) | neutral |
| origin, unfiltered, cfl 0.5 | 4.0e-3, 2.3e-3 | neutral |
| origin, 0.45 / 0.47, cfl 0.18 | 0.144 / 0.174 per step | neutral |
| origin, 0.45 / 0.47, cfl 0.5 | 0.501 / 0.571 | neutral |
| origin, 0.45 / 0.47, cfl 0.9, full strength | 1.87 / 1.91 | neutral |
| poles, unfiltered, cfl 0.18 | 4.1e-7 per step (N = 32), 9.1e-7 (64) | 3.8e-10, 2.5e-8 |
| poles, unfiltered, cfl 0.5 | 2.8e-7, 1.2e-6 | 4.6e-8, 4.6e-8 |
| poles, 0.45, cfl 0.18 | neutral | neutral |
| poles, 0.47, cfl 0.18 | 6.90e-3 / 6.92e-3 per step (N = 32 / 64) | neutral |
| poles, 0.45 / 0.47, cfl 0.5 | 9.7e-3 / 4.7e-2 (N = 32), 9.9e-3 / 5.0e-2 (64) | neutral |
| poles, 0.45 / 0.47, cfl 0.9, full strength | 0.078 / 0.13 (N = 32), 0.086 / 0.14 (64) | neutral |

The filtered rates at the origin are independent of N and exceed the r-z
axis's by a factor of about 25. At the poles they are within 4% of the r-z
axis's at αf = 0.47. The origin's mode peaks at node 1 and the poles' at the
first or last node. Under the package's forms the largest filtered eigenvalue
at the origin and the poles is below 6.2e-9 per step, the smooth mode at 1. On
the resolved θ the pass along θ runs at full strength at these steps (its rate
reads the ring's spacing at the axis), and the filtered map is neutral in
both forms; at cfl 0.5 the ring's θ step is beyond the scheme's limit and
both forms grow at 0.8 per step. In the area form a seeded rest state at the
origin fails without the artificial properties, the sooner at the larger step,
and with them the density deviation at the origin saturates:

```
smooth share of the leading mode, filtered    origin 0.41, poles 0.46 to 0.48

seeded rest state at the origin, area form: 1e-10 in ρ, N = 64, αf = 0.47, the solver's own step
artificial properties off, cfl 0.18    fails between 160 and 200 steps
artificial properties off, cfl 0.5     fails between 40 and 80 steps
artificial properties on               density deviation saturates at 1e-2 to 4e-2
```

The consequences for runs at the origin are under [the CFL
restriction](shocks.md#the-cfl-restriction-and-the-symmetry-cell) and [fold order and geometry
limits](#fold-order-and-geometry-limits).

**The vortex across the resolved-θ axis.** `bench/archive/axisvortex.jl`
(`bench/archive/results/axisvortex.txt`) runs the isentropic vortex of the
Axis-crossing vortex tutorial on its disk with the artificial properties, in
the package's area form and in ∂p/∂r (`gradp`), with the filter off and at the
default. The controls carry the same vortex in the area form on two grids
without the axis, a periodic square and an annulus. "Near" is within two
spacings of the point the center crosses at t = 0.8, on the disk its first two
rings.

```
disk       32 × 64 at cfl 0.5, to t = 1.6 or 20000 steps
square     periodic [−1, 1]² on 64² nodes, spacing 0.031 against the disk's Δr of 0.032
annulus    r ∈ [0.15, 1.5] on 44 × 160 nodes, the vortex carried along y = 0.75
```

| grid, form, filter | end | near ρ error at t = 0.3 / 0.5 / 0.8 | largest ρ / u error |
|---|---|---|---|
| disk, area, off | 20000 steps, t = 0.864 | 1.07e-4 / 1.16e-3 / 0.117 | 0.193 / 0.084 |
| disk, ∂p/∂r, off | 20000 steps, t = 0.896 | 3.54e-5 / 4.02e-4 / 0.019 | 0.064 / 0.102 |
| disk, area, on | complete, 3747 steps | 3.18e-5 / 1.63e-4 / 2.09e-4 | 3.130e-4 / 1.335e-3 |
| disk, ∂p/∂r, on | complete, 3747 steps | 3.04e-5 / 1.58e-4 / 2.12e-4 | 3.189e-4 / 1.341e-3 |
| square, off | complete, 320 steps | 2.2e-6 / 8.0e-6 / 4.9e-6 | 4.50e-5 / 1.02e-3 |
| square, on | complete, 320 steps | 6.6e-6 / 1.5e-5 / 1.7e-5 | 5.07e-5 / 1.06e-3 |
| annulus, off | complete, 960 steps | 1.9e-6 / 7.8e-6 / 8.3e-6 | 3.68e-5 / 2.11e-4 |
| annulus, on | complete, 960 steps | 9.9e-6 / 2.1e-5 / 4.2e-5 | 8.41e-5 / 4.74e-4 |

Unfiltered, the disk's error grows from its first rings in both forms and the
step collapses before t = 0.9. ∂p/∂r lowers the amplitude at t = 0.8 sixfold
and does not remove the growth. The same vortex without the axis completes
unfiltered on both controls with a density error below 5e-5, so the runaway
belongs to the resolved-θ axis, not to the cylindrical metric or to the
unfiltered scheme; the disk's step, set by its first ring, is a twelfth of the
square's. Filtered, ∂p/∂r raises the largest density error by 1.9% and the
velocity error by 0.4%. It lowers the density error at t = 1.4 by 1.0%.

```
disk, unfiltered, t = 0.3 to 0.8    growth of the near error
area form                           14.0 per unit time
∂p/∂r                               12.6 per unit time
mean step                           4.3e-5, against 4.3e-4 filtered
```

## The outflow face on a radial grid

```text
julia --project=. -t 1 bench/archive/sphereoutflow.jl radii=2.1,4.1 sigmas=0,0.25,1
julia --project=. -t 1 bench/archive/sphereoutflow.jl radii=2.1 sigmas=0 ns=64,128,256
julia --project=. -t 1 bench/archive/sphereoutflow.jl radii=2.1,4.1 sigmas=0.25 betas=1
```

The dipole of the Oscillating sphere tutorial leaves through
`NSCBCOutflowBC(pinf = 1/γ)`, and the reflection coefficient is the ratio of the
incoming to the outgoing spherical Hankel coefficient fitted to the pressure
amplitude over the last period (`test/sphere_dipole.jl`). A linear model of the
boundary relation for the exact field (the l = 1 mode, gas at rest, the angular
divergence left in the right-hand side as `beta_t` ≈ M ≈ 1e-3 leaves it) gives
the reflections of the first table:

```
dipole    sphere radius 0.1, k = ω = 4π, U = 1e-3, 16 polar nodes,
          the radial spacing of 128 nodes over [0.1, 2.1] at every outer radius R

linear model                                         R = 2.1    R = 4.1
plane form, sigma = 0                                0.01903
plane form, sigma = 0.25                             0.02399
curvature term                                       7.21e-4    1.89e-4
curvature term written on u_r, −n_c ρc² u_r/2r,      1.44e-3
  instead of on p′
```

| R | kR | `sigma` | plane form | curvature term | 1/(2kR) | 1/(2(kR)²) |
|---|---|---|---|---|---|---|
| 2.1 | 26.4 | 0 | 1.901e-2 | 1.002e-3 | 1.895e-2 | 7.18e-4 |
| 2.1 | 26.4 | 0.25 (default) | 2.395e-2 | 1.002e-3 | | |
| 2.1 | 26.4 | 1 | | 1.327e-3 | | |
| 4.1 | 51.5 | 0 | 9.718e-3 | 3.489e-4 | 9.705e-3 | 1.88e-4 |
| 4.1 | 51.5 | 0.25 (default) | 1.220e-2 | 3.489e-4 | | |
| 4.1 | 51.5 | 1 | | 4.135e-4 | | |

The curvature term's measured reflection converges to the linear model's under
radial refinement, and about a quarter of the remainder at 128 nodes is
discretization. The fitted outgoing amplitude is within 0.3% of the exact one at
128 nodes under either form. With 64 nodes it is 2% high.

```
R = 2.1, sigma = 0, radial nodes     64          128         256
curvature term                       3.601e-3    1.002e-3    7.54e-4
plane form                           1.905e-2
```

Under the curvature term the default `sigma` changes nothing, since the term's
relaxation rate c/R exceeds σc/L_ref at both radii. At `sigma` = 1 the excess
adds the reflection (K − κ)/(2ω) in some phase. The plane form's relaxation adds
σc/(2ωL_ref) in phase.

```
radius                                2.1        4.1
c/R                                   0.476      0.244
σc/L_ref, default sigma               0.125      0.0625
excess of σc/L_ref at sigma = 1       0.024      0.006
plane form, σc/(2ωL_ref)              4.97e-3    2.49e-3
```

The transverse term carries the u_θ cot θ/r of the collapsed φ, which the
φφ velocity gradient holds once u_r/r is dropped. Under the default weight
`beta_t` = M ≈ 1e-3 it moves each reflection above by at most 1.2e-5 of its
value. Under `beta_t` = 1, where the transverse term enters at first order in
the amplitude, the reflection falls far faster with the radius when the term is
kept than when it is dropped:

```
beta_t = M, R = 2.1                curvature term from 1.0021156e-3 to 1.0021067e-3

beta_t = 1                         R = 2.1    R = 4.1    falls as
with the transverse term           4.55e-4    4.13e-5    (kR)^−3.6
without it                         2.96e-4    1.62e-4    (kR)^−0.9
```

On the tutorial's own grid the package's form matches the exact amplitude several
times more closely than the plane form, and the largest pressure error settles
at the value the tutorial reports with the exact solution held at the outer
boundary:

```
tutorial's grid (32 polar nodes), default relaxation, to t = 2
amplitude along the line nearest the axis / exact    0.999 to 1.004; plane form 0.98 to 1.03
phase                                                within −0.003 to +0.001 rad
largest pressure error                               1.0e-3 U from t = 0.5 on
```

The same term carries over to a refined level at the outer face of the r-z annulus. An
outgoing pulse on the tile differs from the uniform run at the tile's spacing by 6.6e-7,
while the uniform run under the plane form differs from it by 1.0e-3
(`test/level_tests.jl`):

```
annulus r ∈ [1/2, 3/2], n_c = 1, a level over the outer sixth; pulse amplitude 1e-2;
49 root nodes, 411 equal steps to t = 0.5, halfway through the exit
```

## The slip wall of the r-z annulus

```text
julia --project=. -t 1 bench/archive/annuluswall.jl
```

A standing wave between the slip walls of the axisymmetric annulus converged
near second order where the Cartesian line converges near fourth. The
instrument runs the inviscid case on r ∈ [r0, r0 + 1] (θ and z collapsed),
unfiltered and without artificial properties unless a row says otherwise.
Every grid of a row takes the same equal steps as a reference on 769 nested
nodes, so the difference is spatial. The four grids run from N = 25 to 193 by
doubling. Errors are maximum norms over the wall nodes, the rest of the
four-node wall windows and the interior; orders are fitted over the four grids.

**The right-hand side on exact data.** Truncation of the assembled
right-hand side on the standing wave, against the exact axisymmetric Euler
right-hand side, at N = 193:

| | ρ at the wall node | order | ρu in the wall window | order |
|---|---|---|---|---|
| Cartesian line | 6.08e-9 | 3.99 | 1.91e-7 | 2.98 |
| annulus, r0 = 0.5 | 5.19e-7 | 2.94 | 1.89e-7 | 2.97 |
| annulus, r0 = 5 | 4.65e-8 | 2.84 | 1.90e-7 | 2.98 |

The wall rows, the area weighting and the curvature source together
truncate at the closure rows' third order on the annulus as on the line. The
line's fourth order in the density is the parity of ρu about the wall, which
r ρu does not have.

**Evolution.** Density at N = 193 and its fitted order:

| case | wall window | order | interior | order |
|---|---|---|---|---|
| standing wave, Cartesian line | 1.83e-10 | 4.66 | 2.99e-10 | 4.42 |
| standing wave, annulus r0 = 0.5 | 8.19e-8 | 2.78 | 5.41e-7 | 2.00 |
| standing wave, annulus r0 = 5 | 1.37e-8 | 2.83 | 7.17e-8 | 2.17 |
| standing wave, annulus r0 = 50 | 1.51e-9 | 3.68 | 7.22e-9 | 2.85 |
| standing wave with u_xx ≠ 0 at the walls, Cartesian line | 1.21e-7 | 2.67 | 8.35e-7 | 1.96 |
| cosine at rest, annulus r0 = 0.5 | 5.81e-9 | 3.46 | 9.59e-9 | 3.30 |
| cosine at rest, annulus r0 = 5 | 8.48e-10 | 3.95 | 1.76e-9 | 3.55 |
| pulse on the walls (t = 0.45), Cartesian line | 1.52e-7 | 4.64 | 6.33e-7 | 3.73 |
| pulse on the walls (t = 0.45), annulus r0 = 0.5 | 1.59e-7 | 4.82 | 6.09e-7 | 3.77 |
| pulse reflected (t = 0.7), Cartesian line | 2.36e-7 | 4.44 | 6.32e-7 | 4.27 |
| pulse reflected (t = 0.7), annulus r0 = 0.5 | 2.76e-7 | 4.67 | 6.63e-7 | 4.25 |
| reflected, default filter every step, Cartesian line | 2.34e-9 | 5.61 | 3.03e-7 | 4.21 |
| reflected, default filter every step, annulus | 4.37e-9 | 5.37 | 2.95e-7 | 4.29 |
| reflected, C6 `:brady_livescu`, Cartesian line | 2.23e-9 | 6.08 | 2.77e-8 | 5.20 |
| reflected, C6 `:brady_livescu`, annulus | 2.50e-9 | 6.06 | 3.09e-8 | 5.18 |

The pulse (`annulus_wall_case`, width 0.1 at the middle of the gap) spans
2.4 nodes per width on the coarsest grid, so its fitted orders include a
pre-asymptotic point.

The order is set by the data, not by the scheme. A slip wall holds u_r = 0
at every time, so a solution smooth up to the wall requires ∂ₜᵏ(ρu) = 0
there for every k. For the standing wave ∂ₜ(ρu) = −p_r vanishes at the
wall, but ∂ₜ²(ρu) = γp(u_rr + u_r/r − u/r²) = ±γpπb/r does not: the
curvature term u_r/r breaks the second condition, and the solution carries a
jump in its second derivatives along the characteristics from the walls at
t = 0. The data at rest break only the third condition,
γp(p_rrr + p_rr/r)/ρ ≠ 0, and converge one order faster. The Cartesian line
with u = b sin πx − πb x(1 − x), whose u_xx at the walls equals the
annulus's u_r/r at r0 = 0.5, loses the same order with no curvature anywhere
in the scheme. The loss on the annulus recedes as r0 grows, since the defect
scales with 1/r0. On the pulse, whose data vanish at the walls, the
annulus reproduces the line's error to within a factor of two at every grid
from N = 97, unfiltered, filtered and under the `:brady_livescu` rows. No
candidate at the curved wall (the wall rows, the area weighting, the
curvature source at the wall node, the filter's wall rows) therefore adds to
the Cartesian wall's own error, and none was switched off separately.

**The convergence row.** `test/convergence.jl` reads the pulse's wall window
against the same case on nested nodes at a third of the spacing. The annulus
converges at nearly the order of the Cartesian line:

```
wall window at t = 0.45, 4(N − 1) equal steps    N = 49     97        193       order
Cartesian line                                   2.51e-4    8.60e-6   2.91e-7   4.88
annulus                                          2.77e-4    1.01e-5   4.60e-7   4.62
```
