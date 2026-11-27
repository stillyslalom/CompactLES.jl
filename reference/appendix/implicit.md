# Calibration appendix: Implicit and incompressible solves

This file holds the sections of the measurement record on the implicit diffusion stage
and the incompressible prototype built on its operators.
[CALIBRATION_APPENDIX.md](../CALIBRATION_APPENDIX.md) indexes the record and states its
conventions.

## Stiff diffusion

```text
julia --project=. -t 1 bench/archive/stiffdiffusion.jl
```

One-dimensional model problems with dense matrices, periodic but for the wall treatment,
measured before any implicit code exists, for the operator and integrator choices of
[IMPLICIT.md](../IMPLICIT.md).

### The implicit operator and its preconditioner

`(I − τL)x = b` with a narrow peak in κ, solved by conjugate gradients preconditioned with
the second-order conservative operator. The wide form is the C6 first derivative applied
twice, as in the explicit molecular fluxes; the staggered form is the C6 staggered
derivative from nodes to midpoints and back with κ interpolated to the midpoints at sixth
order. Errors are the truncation errors of each operator on a manufactured field;
iterations are at steps that are multiples of the explicit diffusive limit:

```
κ = 1 + 1e3·exp(−((x − 1/2)/0.05)²)

     n     err wide    err stag | iterations wide / staggered at dt/dt_explicit = 1e2, 1e4, 1e6
   128    7.647e-01   4.256e+00 |   19 /  13     42 /  14     46 /  15
   256    1.147e-02   6.918e-02 |   21 /  13     53 /  12     76 /  13
   512    1.785e-04   1.099e-03 |   26 /  13     74 /  10    114 /  10
  1024    2.973e-06   1.781e-05 |   37 /  12    120 /   8    201 /   8
```

Both forms converge at sixth order, the staggered one with an error constant six times larger.
The wide form's count grows with the grid and with the step, because its symbol vanishes at
the Nyquist mode where the preconditioner's does not; the staggered form's count is flat
across a ten-thousandfold range in the step and an eightfold range in the grid.

### ARK against RKL2

`T_t + T_x = (κ₀ T^{5/2} T_x)_x` on 128 nodes at two step sizes, the diffusion on the
staggered operator. ARK is `ARK4(3)6L[2]SA` with a Picard iteration on κ in each implicit
stage; STS is Strang splitting with RKL2 half steps around RK4 advection. Errors are against
ARK at sixteen times the steps; `apply` counts applications of the diffusion operator per
step and `prec` the preconditioner solves, each a V-cycle in 3-D. R is the step over the
forward-Euler diffusive limit at the peak temperature:

```
to t = 0.25 at advective CFL 0.5 and at twice as many steps

      κ0        R  steps |     ARK err  order  apply   prec |     STS err  apply
   1e-04      0.1     64 |   3.864e-09    NaN     31     30 |   3.390e-08      6
   1e-04      0.0    128 |   2.493e-10   3.95     31     30 |   2.120e-09      6
   1e-03      0.8     64 |   7.671e-09    NaN     41     40 |   5.648e-08      6
   1e-03      0.4    128 |   4.881e-10   3.97     37     36 |   3.713e-09      6
   1e-02      8.2     64 |   7.243e-08    NaN     54     53 |   6.545e-07      6
   1e-02      4.1    128 |   4.872e-09   3.89     46     45 |   9.092e-08      6
   1e-01     81.9     64 |   6.878e-08    NaN    115    114 |   5.722e-07     19
   1e-01     40.9    128 |   4.386e-09   3.97     85     84 |   1.475e-07     14
   1e+00    818.9     64 |   2.180e-10    NaN     71     70 |   9.553e-08     47
   1e+00    409.5    128 |   2.456e-11   3.15     58     57 |   2.358e-08     35
   1e+01   8189.2     64 |   1.572e-11    NaN     21     20 |   4.091e-11    139
   1e+01   4094.6    128 |   1.761e-11  -0.16     18     17 |   4.113e-11     98
```

ARK is fourth order through R ≈ 80. At R ≈ 800 its observed order falls to 3.15, the
reduction expected of stage order two. At κ₀ = 10 the profile has relaxed toward uniform and
both errors sit near the Picard tolerance, so that row measures cost only. At equal steps
the split RKL2 is second order and from 9 to over 400 times less accurate, the gap widest
where the diffusion is stiffest while the solution still varies. Its diffusion work grows as
√R while the implicit count falls once the solution smooths. The two cross in operator
applications between R ≈ 800 and R ≈ 8e3, and later once a V-cycle is costed. A Picard
stage converged in every row.

### Wall treatment of the staggered operator

```text
julia --project=. -t 1 bench/archive/stiffdiffusion.jl parts=walls
```

`L = G K D_s` on a closed line with nodes on the walls. The rows close each end with
one-sided explicit rows on K points (D_s at the first midpoint, G at the first two nodes with
the wall flux as data, the interpolation at sixth order); the mirror is the node-centred wall
of `src/core/staggered.jl`, the periodic operator on the doubled line folded back. Errors are max-norm
truncation errors; `asym` is ‖W L − (W L)ᵀ‖/‖W L‖ with trapezoidal node weights W, `cons` the
defect of Σ W L T against the wall-flux difference, and the last column the largest real part
of an eigenvalue of the zero-flux operator. The rows run on a smooth field and coefficient;
"even" is a field and a coefficient of the mirror's parity, "T'''" a field of zero slope at the
wall whose third derivative does not vanish there, under an even κ, and "κ'" an even field
under a κ of nonzero slope at the wall:

```
rows     field exp(sin 3x), κ = 1 + sin(2x)/2
T'''     T'(0) = 0 but T'''(0) ≠ 0
κ'       κ = 1 + x/2

                   N    err D_s     err G     err L |  asym W L      cons  max Re λ
rows K = 4        25   1.24e-03  1.90e-02  1.54e-01 |  9.77e-02  1.64e-03  -4.3e-13
rows K = 4        49   1.42e-04  2.30e-03  3.64e-02 |  6.79e-02  2.30e-04   4.0e-13
rows K = 4        97   1.68e-05  2.79e-04  8.82e-03 |  4.76e-02  3.31e-05  -4.3e-12
rows K = 4       193   2.03e-06  3.46e-05  2.17e-03 |  3.35e-02  5.12e-06  -3.3e-11
rows K = 5        25   1.87e-04  5.29e-03  2.31e-02 |  1.38e-01  3.99e-04  -5.9e-13
rows K = 5        49   1.34e-05  2.13e-04  3.76e-03 |  9.58e-02  5.18e-06  -4.0e-13
rows K = 5        97   8.53e-07  9.84e-06  5.05e-04 |  6.71e-02  5.25e-06   1.8e-12
rows K = 5       193   5.33e-08  5.11e-07  6.47e-05 |  4.73e-02  1.73e-06   2.5e-13
mirror, even      25                       8.94e-05 |  8.61e-17  2.65e-14  -2.1e-14
mirror, even      49                       1.39e-06 |  9.80e-17  1.71e-13   4.1e-13
mirror, even      97                       2.16e-08 |  1.14e-16  1.17e-13  -4.9e-12
mirror, even     193                       4.34e-10 |  1.09e-16  1.42e-13  -7.6e-12
mirror, T'''      25                       6.99e-02 |  8.61e-17  1.81e-14  -2.1e-14
mirror, T'''      49                       3.50e-02 |  9.80e-17  3.79e-13   4.1e-13
mirror, T'''      97                       1.75e-02 |  1.14e-16  2.42e-13  -4.9e-12
mirror, T'''     193                       8.76e-03 |  1.09e-16  3.52e-12  -7.6e-12
mirror, κ'        25                       2.18e-01 |  9.43e-17  8.08e-14   3.3e-13
mirror, κ'        49                       1.07e-01 |  8.91e-17  1.35e-13  -3.9e-14
mirror, κ'        97                       5.35e-02 |  1.01e-16  5.48e-13  -2.2e-12
mirror, κ'       193                       2.67e-02 |  9.41e-17  1.09e-12  -1.2e-11
```

The rows converge at their own order in D_s and G but leave L at second order on four
points and near third on five. Their asymmetry of 3 to 14% decays only as the square root
of h and their conservation defect stays well above round-off, although their spectrum
stays real and nonpositive. The mirror is symmetric and conservative to round-off at every
N and sixth order on data of its parity. On data that are not, the jump of an odd
derivative in the mirrored extension gives a first-order truncation at the wall node, from
T''' and from κ' alike. For conduction behind an
adiabatic wall with κ = κ(T) the solution has that parity (T', κ' = κ'(T) T' and T''' vanish
at the wall), so the loss is confined to incompatible initial data.

### The staggered operator at an odd-area fold

```text
julia --project=. -t 1 bench/archive/staggeredfolds.jl
```

`L = J⁻¹ G C K D_s` along one line through a fold whose area vanishes oddly, the flux
continued smoothly through the singular set as the explicit divergence continues it, with
κ = 1. `asym` is ‖V L − (V L)ᵀ‖/‖V L‖ in the node volumes V = J W, `rows` the rows of the
antisymmetric part above 1e-3 of the largest, `cons` the defect |Σ V L T|/Σ V|L T| on a
smooth zero-flux T, then the largest real part of an eigenvalue of L and the largest
eigenvalue of the symmetric part of V L over its norm. The symmetry plane is the control:

```
geometry             N       asym  rows       cons       Re λ      sym λ
symmetry plane      16   9.92e-17     0   2.06e-15   1.46e-13   1.66e-18
cylindrical axis    16   2.70e-03     6   9.40e-04   7.72e-14   2.42e-06
spherical poles     16   9.21e-03    12   4.17e-03  -1.71e-15   1.18e-05
symmetry plane      32   7.86e-17     0   5.22e-15   2.96e-14   9.67e-18
cylindrical axis    32   9.30e-04     6   2.28e-04  -2.40e-13   4.17e-07
spherical poles     32   3.28e-03    12   1.04e-03   3.19e-15   2.08e-06
symmetry plane      64   6.45e-17     0   2.70e-14   9.93e-13  -4.05e-18
cylindrical axis    64   3.25e-04     6   5.60e-05   1.00e-13   7.28e-08
spherical poles     64   1.16e-03    12   2.61e-04  -2.57e-13   3.68e-07
symmetry plane     128   9.44e-17     0   7.69e-14   6.85e-12  -2.97e-18
cylindrical axis   128   1.14e-04     6   1.39e-05  -6.51e-12   1.28e-08
spherical poles    128   4.11e-04    12   6.52e-05  -1.08e-13   6.52e-08
```

The asymmetry sits in the six rows nearest each fold and falls as h^1.5, the conservation
defect falls at second order, and the spectrum of L stays real and nonpositive, but the
symmetric part of V L carries a small positive eigenvalue, so V is not an energy norm there.
The mirror adjoint to D_s, which continues the flux as |r| K T', is symmetric to round-off
but inconsistent: its error at the first node does not fall with h. The spherical origin's
area r² sinθ is even, and there the smooth continuation is the adjoint one.

### Symmetric closures at an odd-area fold

```text
julia --project=. -t 1 bench/archive/staggeredclosure.jl
```

The operators searched are symmetric in a diagonal node norm by construction:

```
L = -H⁻¹ Dᵀ M D,    H = V (1 + δw),    M = W_m r (1 + δμ)
```

H and M are modified on the first k nodes and midpoints, and D is the mirror derivative of
the data's parity plus a correction Z P⁻ᵀ W_n with Z a k × (k + 1) block and P the left-hand
side of the adjoint G, which keeps G in the form of explicit edge rows ahead of its node
solve. Such an L is negative semidefinite and conservative in H. Exactness of D on the
data's monomials to degree d_D and of G on the flux monomials to degree d_G is linear in
(Z, δμ, δw), and the first node is then of order min(d_D, d_G). A diagonal node norm is the
only one compatible with the 1/r² azimuthal and 1/sin²θ polar operators, and a diagonal face
norm the only one exact for a variable κ; at a paired fold δw may still differ between the
even and odd combinations. The dense matrices of the script reproduce `StaggeredDiffusion`
on a cylindrical-axis line to 2.2e-16. The best diagonal norm for the present operator
leaves an asymmetry of 8.2e-5, against 6.5e-4 in V. The least-squares residual of the
design conditions against the closure width, zero to round-off where a closure exists:

```
parity             d_D d_G      residual at k = 6, 10, 14
even                4   4       7.6e-16  2.8e-15  1.1e-15
even                6   4       7.6e-16  2.8e-15  1.1e-15
even                4   6       5.7e-07  2.3e-08  2.8e-09
even                6   6       5.8e-07  2.4e-08  2.9e-09
odd                 5   3       1.7e-15  7.3e-16  1.8e-15
odd                 3   5       2.0e-15  2.4e-15  2.7e-15
odd                 7   3       1.7e-15  7.3e-16  1.8e-15
odd                 5   5       1.4e-07  6.1e-09  7.5e-10
pair, shared δw     4/4 and 3/3 2.0e-07  8.9e-09  1.1e-09
pair, separate δw   4/4 and 3/3 2.6e-15  3.5e-15  3.8e-15
```

No width makes D exact to degree 4 and G exact to degree 5 together: the residual falls with k
but does not vanish, and a nonlinear search that also modifies the tridiagonal left-hand side of
G on the first rows does not lower it. The first node is therefore of order 4 on data even across
the fold and 3 on odd data. The unresolved axis carries even data only and takes an exact order-4
closure at k = 4. A paired fold carries both and reaches orders 4 and 3 only with a separate δw
per parity. The closures, with the asymmetry in H, the largest real eigenvalue of L, the smallest
weight 1 + δw, and the max-norm error of L with its observed orders:

```
error over r < 1/2 on exp(-4r²) (even) and r exp(-4r²) (odd) at h = 1/16 ... 1/256

closure                   par   asym  max Re λ  min w  error                                        orders
smooth continuation       even 5.8e-05  2.4e-16 1.000  4.4e-05 6.8e-07 1.1e-08 1.7e-10 7.4e-11  6.00 6.00 5.95 1.23
smooth continuation       odd  5.8e-05 -5.3e-16 1.000  1.3e-04 4.3e-06 1.4e-07 4.3e-09 1.3e-10  4.96 4.99 5.00 5.00
exact order 4, k = 4      even 3.9e-17 -3.1e-16 0.902  2.4e-03 1.7e-04 1.1e-05 7.0e-07 4.4e-08  3.80 3.95 3.99 4.00
pair exact 4/3, k = 6     even 3.9e-17 -3.5e-16 0.975  7.0e-02 5.4e-03 3.6e-04 2.2e-05 1.4e-06  3.71 3.92 3.98 4.00
                          odd  3.9e-17  1.5e-16 0.996  6.1e-01 1.0e-01 1.3e-02 1.7e-03 2.1e-04  2.61 2.90 2.97 2.99
axis constrained, k = 6   even 3.9e-17 -1.1e-16 0.889  1.1e-03 2.3e-05 5.1e-07 1.7e-08 7.6e-10  5.53 5.49 4.96 4.44
axis constrained, k = 10  even 3.9e-17 -3.5e-16 0.980  7.3e-02 2.0e-03 3.6e-05 5.7e-07 8.8e-09  5.20 5.79 5.96 6.03
axis constrained, k = 14  even 3.9e-17  4.1e-17 0.991  3.0e-01 1.3e-02 2.7e-04 4.5e-06 7.2e-08  4.53 5.59 5.90 5.98
pair constrained, k = 6   even 3.9e-17 -3.7e-16 0.890  1.1e-03 2.3e-05 5.5e-07 1.9e-08 9.3e-10  5.51 5.42 4.87 4.33
                          odd  3.9e-17 -1.9e-16 0.988  4.6e-02 1.8e-03 6.2e-05 2.0e-06 6.1e-08  4.63 4.91 4.98 5.00
pair constrained, k = 10  even 3.9e-17 -1.1e-15 0.981  7.3e-02 2.0e-03 3.6e-05 5.7e-07 8.8e-09  5.20 5.79 5.96 6.02
                          odd  3.9e-17  1.9e-16 0.997  6.6e-01 4.1e-02 1.5e-03 5.0e-05 1.6e-06  4.02 4.74 4.94 4.98
```

Every symmetric closure is far less accurate than the smooth continuation on these grids.
The exact order-4 closure conserves to 1.8e-14 in H, and its symmetric part has no positive
eigenvalue above round-off. Its first-node error is nevertheless 55 times the smooth
continuation's on the coarsest grid and 1000 times two refinements later. The constrained
closures are exact on the reachable set and least squares on the next degree. Their residual
decays with k and their observed orders reach 5 to 6 over this range. Their error
nevertheless stays 30 to 500 times the smooth continuation's and grows with k at coarse h.
The last order of the smooth continuation's even row is round-off.

```
exact order-4 closure, first four nodes    1 + δw = 0.902, 1.003, 1.001, 1.000
                                           1 + δμ = 1.028, 0.992, 1.001, 1.000
```

Defect correction x ← x + A_s⁻¹(b - A x), with A = I - τL the smooth continuation and A_s =
I - τL_s a symmetric operator solved by conjugate gradients, contracts at every step size:

```
A_s                     par   τ|λ_min| = 1e-02    1e+00    1e+02    1e+04    1e+06
adjoint mirror          even             6.1e-04  3.9e-02  1.3e-01  1.4e-01  1.4e-01
exact closure order 4   even             1.0e-04  5.0e-03  6.6e-03  6.9e-03  6.9e-03
adjoint mirror          odd              7.0e-04  4.1e-02  1.2e-01  1.2e-01  1.2e-01
exact closure order 3   odd              5.5e-04  3.2e-02  8.4e-02  1.0e-01  1.5e-01
```

### Staggered conduction against the wide form

```text
julia --project=. -t 1 bench/archive/staggeredconduction.jl
```

`T_t = (κ₀ T^{5/2} T_x)_x + S` with a manufactured T, integrated by RK4 on the package plans:
the staggered `StaggeredDiffusion` and, periodic only, the wide form D κ D on `lele_d1_6`.
The last column is the grid-Nyquist amplitude at the end of a second run started with a
Nyquist perturbation of 1e-3:

```
T = 1 + 0.3e^(−t) f(x), κ₀ = 0.02, to t = 0.1 at dt = 0.2 h²/κmax

line      form           N  steps       error  order  Σ W T drift     Nyquist
periodic  staggered     32     20   5.436e-08    NaN    2.22e-16   4.478e-07
periodic  staggered     64     79   8.640e-10   5.98    4.44e-16   4.946e-15
periodic  staggered    128    316   1.363e-11   5.99    6.66e-16   8.674e-19
periodic  wide          32     20   5.565e-08    NaN    4.44e-16   1.000e-03
periodic  wide          64     79   8.480e-10   6.04    7.77e-16   1.000e-03
periodic  wide         128    316   1.317e-11   6.01    0.00e+00   1.000e-03
wall      staggered     32     20   3.049e-10    NaN    3.33e-16   4.374e-07
wall      staggered     64     79   4.777e-12   6.00    6.66e-16   4.115e-13
wall      staggered    128    316   7.483e-14   6.00    2.22e-16   5.040e-14
```

On this single-mode solution the two forms agree to within 4% at sixth order, and both
conserve Σ W T to round-off. The wide form carries the Nyquist perturbation unchanged; the
staggered form removes it. Behind the adiabatic wall mirror the staggered run keeps sixth order.

### The implicit stage

```text
julia --project=. -t 16 bench/archive/implicitstage.jl part=iterations
julia --project=. -t 16 bench/archive/implicitstage.jl part=manufactured
julia --project=. -t 16 bench/archive/implicitstage.jl part=preconditioner
```

`(I − γΔt L) T = r` by `solve_stage!` of `src/solver/implicit.jl` to rtol = 1e-10 in the L2 norm of
the residual, with a smooth variable κ (κ_max/κ_min ≈ 3) and the manufactured T as r. The
curved outer wall of the axis and origin grids lies at r = 1.5, where T = e^(−16 r²) is
negligible. Iterations of conjugate gradients (CG) or GMRES, each one V-cycle, at three
steps:

```
R = γΔt κ_max/h_min²: 0.2–10 at the smallest step, 2e3–6e4 at the largest

geometry                 method   N          γΔt = 1e-3  1e-1       10
periodic 2-D                 CG   16/32/64   8/9/9    12/13/13   14/15/15
walls 2-D                    CG   16/32/64   7/9/11   14/15/15   16/17/18
isothermal walls 2-D         CG   16/32/64   7/9/10   14/16/18   17/19/22
symmetry planes 2-D          CG   16/32/64   7/9/11   14/14/15   16/17/18
stretched walls 2-D          CG   16/32/64   8/9/11   14/15/15   16/17/18
cylindrical shell r-θ        CG   16/32/64   7/7/8    11/12/14   13/14/15
cylindrical axis r-z      GMRES   16/32/64   7/9/11   14/15/15   17/18/18
resolved axis r-θ         GMRES   16/32/64   8/10/11  13/14/15   16/17/18
walls 3-D                    CG   12/24/48   6/8/9    13/13/14   15/16/16
cylindrical axis r-θ-z    GMRES   12/24/48   8/10/11  13/15/17   17/20/23
spherical shell, poles    GMRES   12/24/48   7/8/9    12/14/18   15/17/25
spherical origin, poles   GMRES   12/24/48   12/19/53 17/33/79   20/38/97
```

Every grid but the spherical ball stays within a few iterations while R grows by four orders.
The GMRES cases converge in the counts of the CG ones, so the asymmetry beside the axis costs
nothing measurable. Manufactured solutions at γΔt = 0.01, variable κ, maximum error and its
observed order (the constant-κ rows agree to two digits):

```
geometry                  N          max errors                orders   iterations
periodic 2-D              16/32/64   2.38e-05 3.56e-07 5.50e-09  6.06 6.02   12/12/12
walls 2-D                 16/32/64   1.19e-06 1.88e-08 2.94e-10  5.98 6.00   12/13/13
isothermal walls 2-D      16/32/64   7.00e-07 1.10e-08 1.73e-10  5.99 5.99   11/13/14
symmetry planes 2-D       16/32/64   1.18e-06 1.88e-08 2.92e-10  5.97 6.00   11/12/13
stretched walls 2-D       16/32/64   1.07e-05 1.60e-07 2.46e-09  6.07 6.02   12/13/14
cylindrical shell r-θ     16/32/64   1.42e-03 3.81e-04 9.82e-05  1.90 1.95   10/11/12
cylindrical axis r-z      16/32/64   6.16e-04 8.74e-06 1.29e-07  6.14 6.08   12/13/14
resolved axis r-θ         16/32/64   5.19e-04 7.52e-06 1.09e-07  5.97 6.03   12/13/14
walls 3-D                 12/24/48   6.17e-06 9.82e-08 1.54e-09  5.97 5.99   10/11/12
cylindrical axis r-θ-z    12/24/48   3.58e-03 5.52e-05 7.97e-07  6.02 6.11   12/13/14
spherical origin, poles   12/24/48   3.82e-03 1.82e-04 4.65e-06  4.26 5.21   15/28/76
spherical shell, poles    12/24/48   4.80e-03 1.29e-03 3.35e-04  1.89 1.94   10/11/14
```

The stage solution keeps the operator's sixth order through the walls of either parity, the
symmetry planes, the stretching and the axis. A curved wall's first-order node gives a
second-order solution (L2 order 2.00). The ball's maximum error sits at the origin node,
whose operator row is of fourth and fifth order (L2 order 6.01 at 48).

GMRES iterations on the same systems at γΔt = 10, preconditioned by the cycle and by the exact
inverse of the second-order operator:

```
geometry                  cycle | exact S⁻¹          cycle | exact S⁻¹
periodic 2-D              N=32    15 |   10   N=64    15 |    6
walls 2-D                 N=32    17 |    8   N=64    17 |    5
isothermal walls 2-D      N=32    19 |    8   N=64    21 |    5
cylindrical shell r-θ     N=32    14 |   12   N=64    15 |   12
resolved axis r-θ         N=32    17 |   13   N=64    17 |   12
walls 3-D                 N=16    15 |   11   N=32    16 |    8
cylindrical axis r-θ-z    N=16    18 |   14   N=32    21 |   13
spherical shell, poles    N=16    16 |   12   N=32    19 |   13
spherical origin, poles   N=16    23 |   14   N=32    41 |   14
```

The pairing of the staggered operator with `L₂` is flat everywhere, the ball included; the
growth on the ball is the cycle's. Near the origin both angular couplings dominate the radial
one, and line relaxation does not smooth an error that varies slowly in both angles and
alternates in r; plane relaxation would. Three alternatives to the cycle, measured by
editing `src/solver/implicit.jl`, each take more iterations than the cycle kept. Coarsening the
angular dimensions with the others leaves the ball unchanged, and a W-cycle on the halved
operator takes no fewer iterations than the V-cycle.

```
CG iterations at γΔt = 10, N = 16/32/64       grid                       alternative   cycle kept
Galerkin coarse operator, W-cycle             walls                      22/27/32      18/20/21, halved operator, V-cycle
every line of a direction relaxed at once     cylindrical shell          23/32/47      15/16/17, zebra colouring
angular dimensions coarsened with the others  axis grid, 3-D (12/24/48)  29/38/52      17/20/23, angular dimensions last
                                              spherical shell            18/25/36      15/17/25
```

### The additive pair on the conduction

```text
julia --project=. -t 1 bench/archive/imexconduction.jl part=pulse explicit=true
julia --project=. -t 1 bench/archive/imexconduction.jl part=pulse rtol=1e-12
julia --project=. -t 1 bench/archive/imexconduction.jl part=rules
julia --project=. -t 1 bench/archive/imexconduction.jl part=stiff
julia --project=. -t 1 bench/archive/imexconduction.jl part=cfl
```

`ARK4(3)6L[2]SA` with the molecular conduction implicit (`ImplicitConduction`,
`src/solver/imex.jl`), one patch, one rank, workstation. A Gaussian pulse diffuses on a periodic
line in a gas chosen so that the flow the pulse drives is negligible; the error is the
maximum over the amplitude against the image sum of the heat kernel. R is the step over the
forward-Euler limit h²/2α. Fixed steps, `step_rule = :none`:

```
pulse    T = 1 + 0.5 e^(−x²/2σ²), σ = 0.2, α = 1, on [−1, 1) with N = 1024, to t = 4σ²/2α = 0.08
gas      R = 1e-10, γ = 1 + R
limit    h²/2α = 1.91e-6

                     rtol = 1e-8                      rtol = 1e-12
R       steps   max err/A  Krylov/solve   wall s   max err/A  Krylov/solve
10       4194   1.299e-10     1.00        24.2     1.299e-10     1.00
100       419   1.302e-09     1.00         2.17    1.935e-10     1.56
1000       42   1.244e-08     1.00         0.22    1.196e-09     2.00
default  83887  1.642e-12       -        109.2     (five-stage integrator, cfl 0.5)
```

One linear solve per implicit stage at every R (five per step): the conduction is linear in
T and the energy linear in T, so each stage's outer iteration stops after one Newton step. At
rtol = 1e-8 the error grows in proportion to R: the stage residual left by a solve that meets
its tolerance in one or two iterations enters every step, and 1e-8 of the thermal energy is
the error level it sets. At rtol = 1e-12 the error at the smallest R is the staggered
operator's spatial error, which the wide operator of the default integrator does not carry,
and the error at the largest R is the time error. In one dimension the line relaxation of
the cycle is an exact solve of the second-order operator, so the counts are lower than the
multidimensional ones of the implicit stage above.

Step rules on the same pulse from t = 0, the step otherwise unlimited (the acoustic limit is
far above the run), rtol = 1e-8. Accepted and rejected are the attempts the step rule
counted; Krylov is the total over the run:

```
rule          target   accepted  rejected   max err/A   Krylov
error          1e-2        1         0      1.646e-02      10
error          1e-3        2         1      5.929e-04      30
error          1e-4        5         2      7.643e-06      55
temperature    0.2         1         0      1.646e-02      10
temperature    0.05        7         1      4.667e-06      53
temperature    0.01       64         3      8.435e-09     347
```

At equal work the two rules reach comparable errors, as the `:error` row at 1e-4 and the
`:temperature` row at 0.05 show. Neither target is the final error: the embedded estimate
bounds a local error per step, the fractional change a step size. The run is one call of
`run!` whose step the endpoint clips, so every attempt after the first is the step rule's
subdivision of that interval.

Stiff stability is measured on a smooth mode plus a grid-Nyquist mode of amplitude 1e-2,
with five fixed steps at each R:

```
N = 256, rtol = 1e-8
R       dt          Nyquist after 1 step   after 5   energy drift   Krylov/solve
1e3     3.05e-2     2.73e-05               5.9e-15    6.7e-16        1.80
1e4     3.05e-1     2.75e-06               1.3e-17    1.1e-16        1.60
1e5     3.05e+0     2.75e-07               2.4e-17   -2.3e-14        1.60
1e6     3.05e+1     2.75e-08               1.3e-17    3.1e-14        1.64
```

The Nyquist amplification of one step falls as 1/R, the L-stable limit of the stiffly
accurate implicit half, and the total energy is conserved to round-off at every R.

A `cfl` chosen for the default integrator is stable under the pair, whose explicit half has
the larger acoustic limit both on the imaginary axis and on an acoustic wave:

```
acoustic limit                                              pair    default five-stage   ratio
imaginary-axis limit of the stability function              4.001   3.341                1.198
largest bounded cfl, inviscid acoustic wave on 64 periodic  2.305   2.195                1.05
  nodes over 400 steps, no filter and no artificial
  properties, bisected to 1%
```

## The incompressible prototype

```text
julia --project=. -t 1 bench/archive/incompressible_checks.jl part=order
julia --project=. -t 1 bench/archive/incompressible_checks.jl part=energy
julia --project=. -t 1 bench/archive/incompressible_tgv.jl 32 10.0
mpiexec -n 8 julia --project=. -t 1 bench/archive/incompressible_tgv.jl 64 10.0
```

`convection=divergence` on either script gives the divergence-form rows.

`src/solver/incompressible.jl` under its defaults: the skew-symmetric convective term, the pressure
solve to rtol = 1e-12, and the projection that also removes the state's divergence over one
step. The design is [INCOMPRESSIBLE.md](../INCOMPRESSIBLE.md).

**Order.** Velocity and pressure converge at sixth order, with and without walls. The
two-dimensional Taylor–Green vortex, maximum errors of the velocity on its staggered points
and of the kinematic pressure at the nodes, orders against the previous row:

```
ν = 0.1, to t = 1 at cfl 0.2
domain                N    velocity            pressure            CG/stage  divergence
periodic [0, 2π)²     16   2.022e-7            8.900e-5            6.0       1.1e-15
                      32   3.186e-9   (5.99)   1.498e-6   (5.89)   7.5       2.1e-14
                      64   4.987e-11  (6.00)   2.385e-8   (5.97)   7.8       9.5e-15
slip walls [0, π]²    16   3.186e-9            1.498e-6            10.8      3.9e-14
                      32   4.987e-11  (6.00)   2.385e-8   (5.97)   10.8      9.4e-15
                      64   7.797e-13  (6.00)   3.745e-10  (5.99)   10.0      1.5e-14
```

The slip box at N intervals reproduces the periodic run at 2N points to the printed digits,
as the mirror is an exact symmetry of the vortex. The divergence form gives the same velocity
errors and a pressure error eight times smaller. Without the divergence term in the
projection the divergence grows and the velocity error stalls; with it the divergence stays
at round-off at any tolerance.

```
divergence form, pressure error, periodic, N as above    1.155e-5   1.838e-7   2.885e-9
without the divergence term, rtol = 1e-10, N = 64 slip box: divergence 5.5e-9 after
  1223 steps, velocity error stalled at 8.5e-11 (order 0.35)
```

**Energy.** The work of the projected inviscid right-hand side on the kinetic energy relative
to ‖u‖ ‖P N‖ at the initial field, and the relative change of K over 50 steps at cfl 0.5:

```
case                  skew: work   K/K0 − 1     divergence: work   K/K0 − 1
periodic 32²          -1.3e-17     +4.1e-9      -3.0e-17           +1.8e-4
slip box 33²          -7.8e-14     -1.3e-11     +4.2e-7            -1.1e-5
no-slip box 33²       -2.6e-4      -4.0e-3      -2.3e-4            -5.3e-3
3-D slip z 16²×17     -5.0e-18     -2.0e-8      -4.2e-18           +9.8e-3
```

The skew form conserves the energy to the time integration's error on the periodic and
slip-walled grids. The divergence form's work is at round-off on the two periodic initial
fields and not on the evolved ones, whose energy it changes by 1e-4 to 1e-2 over the run. The
no-slip mirror dissipates under both forms: the tangential velocity is not odd about the
wall, which the first-order wall node turns into a loss rather than a gain.

**Decomposed runs.** The decomposed runs reproduce the run on one rank to round-off. The
larger differences occur where the walls or the finer blocks make the pressure solves stop at
different iteration counts on the two grids.

```
mpi_tests.jl phase "incompressible", four steps at Δt = 0.02, np = 2, 4 and 8, every case
maximum velocity difference against one rank / maximum velocity    5.7e-16 to 1.7e-13
divergence                                                         4.0e-14 to 1.4e-12
```

**Three-dimensional Taylor–Green, Re = 1600**, with no filter and no model, against the
spectral reference peak of −dK/dt. `scheme` is −dK/dt less the discrete viscous dissipation
at the printed samples, the energy the convective term and the time integration remove:

```
to t = 10 at cfl 0.8; reference peak 1.286e-2 at t = 8.97
grid  form        peak −dK/dt     at t    scheme at the samples      steps  CG/stage  wall
32³   divergence  1.349e-2        8.73    −9.7e-3 to −2.8e-5         182    12.6      66 s, 1 rank
32³   skew        1.090e-2        7.85    −7.6e-5 to +3.0e-6         175    12.6      63 s, 1 rank
64³   divergence  1.586e-2        8.50    −2.2e-3 to −1.7e-6         355    12.4      310 s, 8 ranks
64³   skew        1.540e-2        8.68    −5.5e-5 to +3.9e-5         354    12.4      279 s, 8 ranks
```

The divergence form returns energy to the resolved scales, up to two thirds of the viscous
rate on the coarser grid near t = 7, and its −dK/dt oscillates between samples; its peak on
the coarser grid lies near the reference only through that error. Under the skew form
−dK/dt follows the viscous dissipation at both grids, and the unmodelled peak lies below the
reference at 32³ and above it at 64³, where the resolved gradients overshoot before the
cascade is complete. Both forms run to t = 10 without a filter at both grids; the
compressible solver at 32³ needs its filter (see [Taylor-Green](filter.md#taylor-green)). The
conjugate-gradient count per stage is flat in the grid in every configuration:

```
CG per stage    periodic plane 6–8, between slip walls 10–11, three dimensions 11–14
```
