# CompactLES — Implicit diffusion and IMEX integration: design

This document is the design for roadmap items H1 (implicit diffusion
infrastructure) and H2 (compatible IMEX time integration). The operator and
integrator choices below rest on the
1-D model studies of `bench/stiffdiffusion.jl`, whose tables are in the
appendix under [stiff diffusion](CALIBRATION_APPENDIX.md#stiff-diffusion).
The material, closure and energy contracts that H3–H6 bring to the implicit
system are specified in [DESIGN.md](DESIGN.md#material-and-physics-interfaces);
this document covers the numerical machinery they share.

## Contents

1. [The stiffness to be removed](#the-stiffness-to-be-removed)
2. [Prior implementations](#prior-implementations)
3. [The implicit operator](#the-implicit-operator)
4. [The linear solve](#the-linear-solve)
5. [The integrator](#the-integrator)
6. [Rejected alternatives](#rejected-alternatives)
7. [Stages](#stages)
8. [Open questions](#open-questions)

## The stiffness to be removed

The explicit step is bounded by the diffusive rate `κ/(ρ c_v h²)` summed over
directions, beside the acoustic rate `(|u| + c)/h`. For the molecular
transport of the present target problems the acoustic rate dominates. For
Spitzer–Härm electron conduction, `κ ∝ T_ele^{5/2}`, the diffusive rate in a
hot corona or a heated shell exceeds the acoustic rate by three to six
orders of magnitude, and it grows as the grid is refined, because it scales
as `h⁻²` against the acoustic `h⁻¹`. Flux-limited radiation diffusion (H6) is
stiffer again, and electron–ion and matter–radiation exchange (H3, H6) add
local rates with no spatial coupling that can exceed both. The ratio R of the
advective step to the forward-Euler diffusive limit is the measure used
below. R of order 1e3 to 1e6 is the design range.

## Prior implementations

- **FLASH** solves conduction and multigroup radiation diffusion through a
  general implicit diffusion unit: a θ-scheme (backward Euler or
  Crank–Nicolson) with coefficients lagged to the start of the step,
  operator-split from the hydrodynamics, solved by HYPRE Krylov methods with
  multigrid preconditioning, with harmonic, min/max,
  Larsen and Levermore–Pomraning flux limiters applied to the coefficient.
  It is first order in time where the coefficients vary.
- **CRASH** (Van der Holst et al., ApJS 194, 2011) splits a step into
  hydrodynamics, radiation advection in frequency, and one implicit solve of
  radiation diffusion, electron conduction and energy exchange together, with
  a Krylov solver on the coupled system.
- **Riot** (LANL; [documentation](https://lanl.github.io/riot/main/index.html)),
  a Parthenon/Kokkos finite-volume code, splits its stiff physics from the
  explicit stages and applies it in a fixed order: electron–ion exchange by
  an exact exponential update, then Spitzer–Härm electron and Braginskii ion
  conduction, both solved by BiCGSTAB without a multigrid preconditioner and
  without a flux limiter. Its multigroup P1 radiation diffusion holds the
  group energy at cell centres and the flux at faces, solves every group
  together in one Newton iteration under BiCGSTAB preconditioned by
  geometric multigrid, and uses the P1 relaxation factor in place of a flux
  limiter. Its implicit packages size the step by a target fractional change
  of temperature, not by a stability bound.
- **Athena++, PLUTO** and several astrophysical codes treat anisotropic
  conduction explicitly by RKL2 super-time-stepping (Meyer, Balsara and
  Aslam, MNRAS 422, 2012 and JCP 257, 2014), Strang-split from the
  hydrodynamics, second order in time, with no linear solve.
- **SUNDIALS ARKODE** supplies additive Runge–Kutta pairs, among them
  ARK4(3)6L[2]SA (Kennedy and Carpenter, Appl. Numer. Math. 44, 2003):
  six stages, fourth order, an L-stable stiffly accurate ESDIRK implicit half
  of stage order two, and an explicit half sharing its abscissae.
- **Low-storage IMEX schemes** (Cavaglieri and Bewley, JCP 286, 2015) keep
  the register count of a low-storage explicit method at third order.

None of the codes above discretizes with compact finite differences, so the
operator question below has no precedent among them.

## The implicit operator

The explicit right-hand side forms a molecular flux divergence as `D κ D`:
the compact first derivative, a multiplication by the coefficient, and the
compact first derivative again. The symbol of `D` vanishes at the grid
Nyquist mode, so `D κ D` does not damp the odd–even mode; the explicit solver
relies on the filter for it. An implicit solve with this operator inherits
the defect. A Krylov method preconditioned by the second-order conservative
three-point operator `L₂`, the operator a multigrid cycle inverts cheaply,
is then not spectrally equivalent to the system: the preconditioned
condition number grows in proportion to the step, and the iteration count
grows with both the step and the grid.

The implicit operator is therefore the conservative staggered compact form

    L = −D_sᵀ K D_s,     K = diag(κ at midpoints),

with `D_s` the sixth-order compact staggered derivative from nodes to
midpoints (Lele 1992, α = 9/62) and κ interpolated to the midpoints at sixth
order. The form is conservative, because the flux lives at the midpoints and
telescopes; symmetric negative semidefinite, so the implicit stage is
unconditionally energy-stable; nonzero at the Nyquist mode; and sixth order.
Its truncation error is several times the wide form's at equal resolution,
and it costs the same: one tridiagonal solve per direction per derivative.
Preconditioned by `L₂` its conjugate-gradient count is flat in both the grid
and the step over the whole design range; the wide form's is not.

At a closed wall the operators do not use closure rows. Each wall node is a
mirror plane: the temperature is even through it (the adiabatic wall) or,
written on `T − T_wall`, odd (the isothermal wall), the flux has the opposite
parity, and the interior rows run to the wall on the mirrored data. The
adjoint identity of the periodic pair then holds on the closed line with
trapezoidal node weights, so the discrete operator stays symmetric, negative
semidefinite and conservative. One-sided closure rows keep a second- or third-order
truncation but make the operator nonsymmetric and leave a conservation
defect, which rules out conjugate gradients; the mirror is exact to the
interior order on data of its parity and first order at the wall node on
data that are not. The measurements are in the appendix under
[stiff diffusion](CALIBRATION_APPENDIX.md#wall-treatment-of-the-staggered-operator).

A symmetry plane or a coordinate singularity lies half a cell beyond the end
node, so a midpoint lies on the plane. The mirror is taken about that
midpoint; an even plane value is an unknown of the line, eliminated from the
row of the next midpoint at a low end and recovered into the halo slot
beneath it. The folded line is half of a longer unfolded one, so the
identities and the interior order carry over, and a paired fold runs the
even and odd combinations through the two parity plans as the explicit
operators do. On a curvilinear or stretched line the flux is `C K D_s T`
with `C = J/h_d²` evaluated at the midpoints and the divergence takes
`J⁻¹`, so `J W L` is symmetric in the node volumes. A curved wall's `C` is
not even about the wall, which makes its node first order as a
nonconstant κ does. Where the area vanishes oddly, at the cylindrical axis
and the spherical poles, the flux `C K D_s T` continues evenly through the
plane while `D_s T` is odd, so the smooth continuation, which the explicit
divergence also takes, is not the mirror adjoint to `D_s`: `J W L` is
symmetric there only up to a defect confined to the rows near the fold and
falling with h, and conserves to second order
([appendix](CALIBRATION_APPENDIX.md#the-staggered-operator-at-an-odd-area-fold)).
The adjoint mirror would
continue the flux as `|r| K T'`, which is inconsistent at the first node.
The spherical origin's area is even and keeps both. At the axis and the
poles no closure symmetric in a diagonal node norm keeps the order: the
first node reaches order four on data even across the fold and three on
odd data, at errors far above the smooth continuation's
([appendix](CALIBRATION_APPENDIX.md#symmetric-closures-at-an-odd-area-fold)).

The staggered form enters only the implicit part. The explicit molecular
fluxes keep `D κ D`, so no current baseline moves. A conduction channel moved
into the implicit part changes discretization with it, which is a numerics
change of that configuration and is qualified as one.

## The linear solve

Each implicit stage solves `(I − γΔt L) Y = r`, nonlinear in `Y` through
κ(T) and, under H3 and H6, through the exchange terms.

- **Outer iteration.** Picard iteration on the coefficient converges for the
  measured conduction cases. Coupled exchange needs Jacobian-free
  Newton–Krylov, whose Jacobian action is a finite difference of the
  residual. The residual is the stage equation evaluated with the high-order
  operator, so the converged stage carries the operator's accuracy whatever
  preconditioner is used.
- **Krylov method.** Conjugate gradients while the system is symmetric
  (conduction alone); GMRES once exchange or anisotropy breaks the symmetry,
  and at the cylindrical axis and the spherical poles, where the staggered
  operator is not symmetric in the rows next to the fold. Both are written
  in the package on the volume-weighted system over the padded fields, so
  that each reduction is one collective and no dependency is added.
- **Preconditioner.** `I − γΔt L₂`, assembled pointwise from the lagged
  coefficient over the patch's metric, inverted approximately by one
  multigrid V-cycle. On the structured patch grid this is the case HYPRE's
  PFMG and SMG solve, but HYPRE.jl wraps only the ParCSR interface
  (BoomerAMG and the Krylov methods), so the routes are BoomerAMG on the
  assembled `L₂`, a binding of HYPRE's Struct interface, or an in-house
  geometric cycle. The in-house cycle's line-relaxation smoother would
  reuse the distributed tridiagonal solver, since `L₂` along one grid line
  is tridiagonal. Stage 2 takes the in-house cycle: pairwise aggregation
  with halved coarse conductances, zebra line relaxation on per-line
  tridiagonal solves, and a dense coarsest solve replicated on every rank.
  Exchange
  terms are local, so they join the preconditioner as a pointwise block
  (physics-based preconditioning in the sense of Knoll and Keyes, JCP 193,
  2004) without changing its sparsity.
- **Collectives.** Every Krylov iteration is a global dot product and a halo
  exchange; every rank iterates to the same count from the same reductions.
  Convergence, failure and retry are collective decisions, as the
  `StepControl` rollback already is.

## The integrator

ARK4(3)6L[2]SA is the H2 integrator. The advective, acoustic and
artificial-property terms form its explicit half; conduction, radiation
diffusion and exchange its implicit half. It reaches fourth order on the model
problem at moderate stiffness and falls toward third at large R, the order
reduction expected of stage order two.

It is not low-storage. Six explicit stage derivatives are held for every
conserved component, and six implicit ones for the implicit components only.
At 128³ with eight components that is on the order of a gigabyte beside the
state, acceptable on a node and on an MI300A, but the reason the explicit
RK45 stays the default: a run with no implicit term never allocates the ARK
workspace, as DESIGN.md's analytic execution contract requires. The explicit
half's stability region differs from RK45's; its acoustic limit is slightly
higher, so a `cfl` chosen for RK45 is stable under the pair
([the additive pair](CALIBRATION_APPENDIX.md#the-additive-pair-on-the-conduction)).

The stage equations share one right-hand side, so components contribute to
one residual and are not updated in sequence, as H2 requires.

## Rejected alternatives

- **RKL2 super-time-stepping.** Explicit, with no linear solve, and it reuses
  the explicit operators unchanged. Its stage count grows as `√R`, while the
  implicit step's iteration count does not grow with R, so the two cross
  near R of order 1e3 on the model problem in operator applications, and
  beyond it once the preconditioner's V-cycles are counted too. At the upper
  design range RKL2 costs more by an order of magnitude or more. It is second order in time and requires splitting from the
  hydrodynamics, which conflicts with the joint residual that H2 specifies.
- **A θ-scheme with lagged coefficients** (the FLASH form). First order in
  time where the coefficients vary, against a solver whose temporal order
  is guarded at four.
- **ADI on a narrow compact second derivative.** The narrow sixth-order
  compact second derivative gives banded line systems that the existing
  distributed solver could factor directly. It is not conservative for a
  variable coefficient, and approximate factorization adds a splitting error
  that grows with the step. As a smoother inside the preconditioner it
  remains available.
- **Low-storage IMEX pairs.** Third order at the register count of a
  low-storage method. Revisit if the ARK workspace is the binding constraint
  on the target machine.

## Stages

1. **Staggered operator.** `src/staggered.jl`: the plans from nodes to
   midpoints and back and the midpoint interpolation of the coefficient,
   periodic, with the wall mirror and across the half-offset folds, on the
   distributed tridiagonal solve of `plan_direction`, and on the metrics and
   stretched grids the solver supports; not reached from the right-hand
   side. Delivered with convergence rows (sixth order, fifth and fourth at
   the first node of the singular folds, first at a curved wall node), the
   adjoint and symmetry identities in the serial suite, the decomposed solve against the undivided one in the MPI suite,
   off-rank folds included, and an explicit conduction run against the
   wide form (`bench/staggeredconduction.jl`).
2. **Implicit solve on one patch.** `src/implicit.jl`: the matrix-free
   stage operator on the volume-weighted system, `L₂` assembled on the patch
   metric, conjugate gradients (GMRES at the axis and poles) preconditioned
   by one multigrid V-cycle on `L₂`; host storage only, not reached from the
   right-hand side. Delivered with manufactured solutions at sixth order on
   every geometry but the curved walls (second, from the first-order wall
   node), iteration counts flat in grid and step on every two-dimensional
   geometry and on three-dimensional Cartesian and cylindrical grids, a
   uniform right-hand side returned without iterating, and the decomposed
   solve against the undivided one in the MPI suite
   ([the implicit stage](CALIBRATION_APPENDIX.md#the-implicit-stage)).
   Remaining: the smoother on a spherical grid with its origin, where the
   two angular directions dominate the radial one and the count grows with
   the grid (plane relaxation is the candidate), and agglomeration of the
   coarsest level, which holds one node per rank and is factorized densely
   on every rank.
3. **ARK integration of the existing conduction.** The 1T molecular
   conduction of the present solver as the first implicit component, so H1
   and H2 are verified before H3 exists. Gate: temporal order on the
   smooth-evolution cases, stiff stability at large R, the acoustic CFL limit
   of the explicit half, and a failed implicit solve recovered by collective
   rollback.
4. **Refined levels and devices.** A level's implicit stage with Hermite
   coarse–fine data, and the preconditioner on device storage. Designed after
   stage 3 has measured costs.

H3 then adds `T_ele` and exchange as implicit components, H4 the
Spitzer–Härm coefficient with its flux limiter, and H6 the radiation energy.

## Open questions

- Walls whose data are not of the mirror's parity: an isothermal wall with
  κ(T), where the odd mirror is inexact in `T''`, and a prescribed nonzero
  wall flux, and every curved wall, whose face factor is not even about the
  wall. A summation-by-parts closure with diagonal norms and one
  modified row did not reach first-degree accuracy in a brief numerical
  search; wider closures and a non-diagonal node norm remain untried.
- The solve at the cylindrical axis and the spherical poles: GMRES on the
  smoothly continued operator, or a defect correction whose inner solve
  applies conjugate gradients to the symmetric adjoint mirror, under which
  the correction contracts at every step size
  ([appendix](CALIBRATION_APPENDIX.md#symmetric-closures-at-an-odd-area-fold)).
  The choice rests on the measured cost of each. Stage 2 delivers GMRES,
  which reaches 1e-10 in as many preconditioned iterations as conjugate
  gradients on the symmetric geometries
  ([appendix](CALIBRATION_APPENDIX.md#the-implicit-stage)); a correction
  contracting by 0.14 per outer step needs about twelve outer steps to the
  same tolerance, each an inner solve, and competes only if loose inner
  solves keep that contraction.
- The step size of the implicit half. The L-stable implicit half imposes no
  stability limit, so its step is set by accuracy: either the embedded error estimate of the ARK
  pair, or a target fractional change of each implicit temperature per step,
  Riot's rule, which needs no error norm across components of different
  units. Stage 3 measures both on the existing conduction.
- A flux limiter makes the coefficient depend on the gradient, so the
  Picard iteration may stall; JFNK with the limited flux is the fallback.
- The V-cycle's counts are measured on smooth coefficients; a coefficient
  contrast beyond the 1e3 of the 1-D study remains to be measured with it.
- Whether the explicit molecular fluxes should also move to the staggered
  form, which would remove the filter's role at the Nyquist mode for the
  viscous terms, is a separate numerics question outside H1.
