# CompactLES — Incompressible solver: design

This document is the design of the incompressible solver, roadmap item I1,
and of the stages that take it from the delivered prototype to the
production configuration. The prototype is `src/incompressible.jl`; its
measurements are in the appendix under
[the incompressible prototype](CALIBRATION_APPENDIX.md#the-incompressible-prototype).

## Contents

1. [Purpose](#purpose)
2. [Prior implementations](#prior-implementations)
3. [Arrangement and operators](#arrangement-and-operators)
4. [Projection and time integration](#projection-and-time-integration)
5. [Boundaries](#boundaries)
6. [Parallel, refined and device execution](#parallel-refined-and-device-execution)
7. [Rejected alternatives](#rejected-alternatives)
8. [Stages](#stages)
9. [Open questions](#open-questions)

## Purpose

The compressible solver steps at the acoustic rate `(|u| + c)/h`. At a Mach
number of 0.01 the step is a hundredth of the advective one, and the
low-Mach mixing problems the package targets (Rayleigh–Taylor growth at
late time, the incompressible limit of Richtmyer–Meshkov mixing, decaying
and forced turbulence for SGS calibration) are then dominated by acoustic
steps that carry no information. The incompressible equations remove the
acoustic rate. The variable-density form (Livescu and Ristorcelli, JFM 591,
2007) is the target, since the mixing problems are at
Atwood numbers well above the Boussinesq range; the constant-density solver
is its first stage, because the variable-density pressure equation is the
same operator with a coefficient.

## Prior implementations

- **Nagarajan, Lele and Ferziger** (JCP 191, 2003): sixth-order compact
  staggered derivatives and interpolations on the marker-and-cell
  arrangement, an exact projection, and robustness at high Reynolds number
  without filtering. The arrangement and operators below follow it.
- **PadeOps** (Ghate and Lele): Fourier pseudo-spectral in the horizontal
  directions and sixth-order compact staggered differences in the vertical,
  the pressure solved by transforms horizontally and a banded solve
  vertically. Its vertical treatment is the one-dimensional case of the
  operators here.
- **AMReX incflo and IAMR**: second-order finite volume with an exact MAC
  projection of the face velocities used for advection and an approximate
  nodal projection of the cell-centred velocity, both by geometric
  multigrid over the AMR hierarchy, with composite solves across levels.
  The source of the refined-level design below.
- **Livescu and Ristorcelli** (JFM 591, 2007, and later CFDNS studies):
  the variable-density incompressible equations of miscible mixing, whose
  pressure equation carries the coefficient 1/ρ and whose divergence
  constraint is set by the molecular mixing.

## Arrangement and operators

The kinematic pressure φ lives at the nodes and velocity component i at the
midpoints along dimension i and the nodes along the others. The operators
are those of `src/staggered.jl`: the compact derivatives `D_s` (nodes to
midpoints) and `G` (midpoints to nodes), α = 9/62, sixth order, and the
explicit sixth-order interpolations between nodes and midpoints in both
directions. The divergence is `Σ_i G_i u_i`, the gradient `D_s,i φ`, and
their composition `L = Σ_i G_i D_s,i` is the operator of the implicit
diffusion stage at unit coefficient ([IMPLICIT.md](IMPLICIT.md)).

The convective term is the mean of its divergence and advective forms, the
skew-symmetric form. For `∂_i(u_i u_i)` the component is interpolated to the
nodes, squared, and differentiated back by `D_s,i`; for `∂_j(u_j u_i)`,
j ≠ i, both components are interpolated to the points that are midpoints
along i and along j, multiplied there, and the product differentiated along
j by `G_j` onto u_i's points. The advective form takes the same transporting
velocity at the same points, multiplies it by the staggered derivative of
u_i there, and interpolates the product back. The midpoint-to-node
interpolation is the transpose of the node-to-midpoint one, since the taps
are symmetric, and `G_j` is minus the adjoint of `D_s,j` in the node and
midpoint weights, so the contributions of the two forms to the kinetic
energy cancel term by term: the convective term does no work on the
discrete kinetic energy, as Morinishi, Lund, Vasilyev and Moin (JCP 143,
1998) construct at second order. The divergence form alone, kept as an
option, conserves momentum instead and returns energy to the resolved scales
where the grid under-resolves the cascade. The viscous term along each
direction is the composition of the two staggered first derivatives.

## Projection and time integration

The right-hand side `N(u)` of each Runge–Kutta stage is projected,
`P N = N − D_s φ` with `L φ = Σ_i G_i N_i`, rather than the state. P is
linear and constant, so `du/dt = P N(u)` is an ordinary differential
equation on the divergence-free fields and the five-stage low-storage
scheme keeps its fourth order without a pressure extrapolation. φ at a
stage is the kinematic pressure at the stage time, to the order of the
scheme. Since the projection is exact, the divergence of the state is held
at the tolerance of the solve and does not accumulate truncation error.

The pressure equation is singular, its null space the constants, on a
periodic or wall-bounded domain. The right-hand side has zero weighted sum
by the adjoint identity `W_n G = −D_sᵀ W_m` of the staggered operators,
which holds on periodic lines and through the wall mirror; its round-off
mean is removed before the solve. The solve is the stage's conjugate
gradient iteration at zero capacity, preconditioned by the multigrid cycle
on the second-order operator, whose coarsest level carries a rank-one term
that makes it definite and returns the solution of zero sum. The tolerance is
relative to the larger of the divergence and the divergence scale `‖N‖/h`
of the field being projected, so a field already divergence free is not
iterated against its own round-off. Each stage's projection also removes
the divergence the state carries, at the rate of the reciprocal step: the
residual of every earlier solve then decays instead of accumulating over
the steps, and the term vanishes on a divergence-free state. The multigrid
hierarchy of the singular operator keeps two nodes per dimension on its
coarsest levels, since a line with one transverse aggregate and no mass
couples to nothing and its relaxation is singular.

## Boundaries

A closed dimension is bounded by walls through its end nodes, treated by the
staggered plans as mirror planes. The normal component is odd, the pressure
even, and a tangential component even at a slip wall and odd at a no-slip
wall. A slip wall is a symmetry of the equations, so every field has the
mirror's parity and the operators keep the interior order. A no-slip
tangential velocity is not odd about the wall: its second normal derivative
there balances the tangential pressure gradient and the viscous stress
divergence, so the mirror is first order at the wall node. The candidate
remedy is closure rows on the tangential components only, keeping the
mirror for the normal component and the pressure so that the projection
stays exact and symmetric; [IMPLICIT.md](IMPLICIT.md#open-questions) records
why a symmetric closure was not found for the conduction operator, and the
same search applies here.

Inflow and outflow boundaries are not in the prototype. Both change the
pressure equation's null space and the compatibility condition on its
right-hand side.

## Parallel, refined and device execution

The design targets the production configuration from the start; the
prototype covers its first part.

- **Ranks.** Every operator is a directional plan on the distributed
  tridiagonal solve, and the projection is the collective conjugate-gradient
  iteration of the implicit stage, so the solver runs decomposed over any
  process grid with at least four points per rank and direction (the
  staggered plans' minimum). Each Krylov iteration is two reductions and one
  halo exchange per dimension, and every rank iterates to the same count.
- **Refined levels.** The velocity is a face field, so coarse–fine transfer
  conserves face fluxes: a coarse face is the area average of the fine
  faces on it, and prolongation along a coarse–fine face interpolates the
  coarse face velocity divergence-consistently (Balsara 2001; Martin and
  Colella 2000). The projection becomes a composite solve over the
  hierarchy, with the coarse–fine face fluxes of `D_s φ` matched so that
  the composite divergence vanishes; the multigrid hierarchy of the stage
  extends across levels as AMReX's composite MAC projection does. The
  patch-interface machinery of the compressible solver (ghost exchange,
  restriction, reflux registers) carries over in form, with the face as the
  unit of the register in place of the node.
- **Device.** The explicit interpolations are pointwise kernels; the
  derivatives are tridiagonal line solves of the kind `device_plan` already
  mirrors for the compact plans. The Krylov vectors are broadcasts and
  reductions. The multigrid smoother's per-line tridiagonal solves and the
  replicated coarsest factorization are the parts with no device form yet,
  shared with the implicit stage's own device item.

## Rejected alternatives

- **Collocated velocity with a wide-derivative projection.** The
  composition of two collocated compact derivatives vanishes at the Nyquist
  mode, so the pressure equation decouples odd and even points and the
  projection leaves the checkerboard mode in the velocity, which the filter
  must then remove. An exact projection on a collocated grid needs a face
  velocity carried beside the node velocity (the Rhie–Chow and Kim–Choi
  forms, or incflo's MAC velocity), which drifts from the node velocity by
  the interpolation error and doubles the pressure solves.
- **Artificial compressibility or a low-Mach preconditioned compressible
  solve.** Either keeps the compressible solver's machinery, but the first
  carries a pseudo-acoustic rate that must be converged away each step, and
  the second changes the dissipation of the compressible scheme, against
  which the artificial-property calibration was made.
- **Projecting the state after each stage.** Equivalent to projecting the
  right-hand side for this linear, constant projection, at the same cost;
  the right-hand-side form names φ as the pressure directly.

## Stages

1. **Constant density on one patch, host storage, decomposed.**
   `src/incompressible.jl`: the arrangement, the operators, the projection
   and the projected low-storage step, the skew-symmetric and divergence
   forms, periodic and slip-wall dimensions, and the no-slip mirror at first
   order. Delivered with the Taylor–Green vortex at sixth order periodic and
   between slip walls, the divergence at the solve tolerance, the convective
   work on the kinetic energy at round-off, the decomposed run against the
   undivided one at two, four and eight ranks, and the three-dimensional
   vortex at Re = 1600 without a filter or a model
   (`bench/incompressible_tgv.jl`).
2. **Scalars and the variable-density form.** Passive scalars in
   divergence form transported by the projected face velocity, then the
   variable-density equations with the pressure equation
   `Σ_i G_i (1/ρ) D_s,i φ = r`, which is the implicit stage's operator at the
   coefficient 1/ρ, and the divergence constraint of the mixing model.
   Gate: a Rayleigh–Taylor single-mode growth rate against linear theory,
   and the decomposed run against the undivided one.
3. **No-slip closure.** Closure rows for the tangential components at a
   no-slip wall. Gate: plane Poiseuille and the start-up of Couette flow at
   the interior order or a stated wall order.
4. **Refined levels.** Face-centred coarse–fine transfer and the composite
   projection. Gate: the two-level Taylor–Green and single-mode
   Rayleigh–Taylor against the uniform run at the fine spacing.
5. **Device storage.** Device mirrors of the staggered plans and of the
   smoother. Gate: the device step against the host step, and timings on the
   target machine.

## Open questions

- Whether the projection should also hold the state's divergence. The
  right-hand side is projected with the term that removes the state's
  divergence over one step, so the solve's residual decays rather than
  accumulating; a fixed number of iterations per stage, instead of a
  tolerance, would rest on that term and is not yet measured.
- Whether an LES model is needed beyond the staggered arrangement's
  robustness, and which: the compact filter of the compressible solver
  applies along the node directions of each component unchanged, and along
  its own direction on a periodic dimension; at a wall it needs a
  midpoint mirror.
