# CompactLES — Design and Mechanics

This document explains how CompactLES is put together and how the solver works,
for readers who want to modify it, extend it, or understand its numerics. For
usage, see [README.md](README.md).

## Contents

1. [Overview](#overview)
2. [Source map](#source-map)
3. [The frontend/backend split](#the-frontendbackend-split)
4. [State representation and decomposition](#state-representation-and-decomposition)
5. [Compact schemes](#compact-schemes)
6. [The distributed compact solve](#the-distributed-compact-solve)
7. [Halo exchange](#halo-exchange)
8. [Time integration](#time-integration)
9. [The RHS: a walkthrough](#the-rhs-a-walkthrough)
10. [Artificial fluid properties](#artificial-fluid-properties)
11. [Curvilinear metrics and the discrete GCL](#curvilinear-metrics-and-the-discrete-gcl)
12. [Coordinate-singularity folds](#coordinate-singularity-folds)
13. [Multicomponent thermodynamics](#multicomponent-thermodynamics)
14. [Characteristic boundary conditions](#characteristic-boundary-conditions)
15. [Threading and MPI discipline](#threading-and-mpi-discipline)
16. [Extension points](#extension-points)

## Overview

CompactLES solves the compressible, multicomponent Navier–Stokes equations in
conservative form on structured, orthogonal-curvilinear grids. The conserved
vector per grid point is

    Q = (ρY₁, …, ρY_Ns, ρu, ρv, ρw, E)

— partial densities for each of `Ns` species, the three momentum components, and
total energy. Spatial derivatives use Lele-style **compact (Padé) finite
differences**: implicit schemes that couple an entire grid line through a
banded left-hand-side matrix, giving spectral-like resolution at modest stencil
width. Because the operator is line-global, the central parallel challenge is a
**distributed banded solve**, which the code handles with the spike /
reduced-interface method.

There is no Riemann solver and no flux limiter. Shocks and under-resolved
gradients are regularized instead by **Cook-style artificial fluid properties** —
localized artificial viscosity, bulk viscosity, conductivity, and species
diffusivity keyed to high-order derivative sensors — together with a compact
low-pass **filter** applied to the conserved state. Time advances with a
low-storage five-stage RK45.

The guiding architectural idea is a strict separation between the physical
problem specification and the numerical discretization, so that the same problem
can be run at any resolution, scheme order, or process count without change.

## Source map

| File | Responsibility |
|------|----------------|
| `src/CompactLES.jl`        | Module: include order and exports |
| `src/decomposition.jl`     | 3-D MPI Cartesian decomposition, sub-communicators, halo buffers |
| `src/halo.jl`              | Sequential-dimension `Sendrecv!` halo exchange; batched multi-field exchange; axis parity fill |
| `src/tridiag.jl`           | Thomas factorization + spike/reduced-interface distributed tridiagonal solve |
| `src/banded.jl`            | Generalization of the distributed solve to half-bandwidth q (pentadiagonal and up) |
| `src/lines_transposed.jl`  | Cache-friendly (lines × n) fill/solve/scatter for the y/z sweeps |
| `src/kernels.jl`           | `CompactScheme`, `ClosureRow`, and tridiagonal presets (C6, C4, C8 filter) |
| `src/kernels_banded.jl`    | `BandedCompactScheme` and the C10 pentadiagonal preset |
| `src/operators.jl`         | `DirPlan`: bind a scheme to a dimension; line fill, distributed solve, scatter |
| `src/operators_banded.jl`  | `BandPlan`: the banded counterpart |
| `src/physics.jl`           | EOS abstraction, `IdealMixture`, `Transport`, `primitives!` |
| `src/boundary.jl`          | `BoundaryCondition` types, wall enforcement, `apply_bcs!` |
| `src/nscbc.jl`             | Characteristic NSCBC subsonic outflow and inflow |
| `src/metric.jl`            | Cartesian/cylindrical/spherical metrics, stretch mappings, curvature corrections, momentum sources, discrete GCL |
| `src/folds.jl`             | Coordinate-singularity (axis/origin/pole) parity and antipodal folds |
| `src/artificial.jl`        | Cook artificial μ\*, β\*, κ\*, D\* from δ⁴ sensors |
| `src/rhs.jl`               | `Solver` container, flux assembly, the conservative NS RHS |
| `src/timestep.jl`          | RK45, CFL timestep, the `run!` loop, per-step filtering |
| `src/io.jl`                | Per-rank checkpoint/restart, parallel VTK output |
| `src/problem.jl`           | Frontend: `Prim`, `Problem`, `Numerics`, `setup`, initialization |

The include order in `CompactLES.jl` is bottom-up: low-level solvers first, then
operators, then physics, then the RHS and frontend that compose them.

## The frontend/backend split

The user-facing vocabulary lives in `problem.jl` and is deliberately ignorant of
the backend:

- **`Prim`** — a pointwise primitive state (velocity, pressure, composition,
  and one of T_ion or ρ). Construction validates that mass fractions sum to one and
  that exactly one of temperature or density is given.
- **`Problem`** — physics and geometry only: EOS, transport model, metric,
  coordinate `domain` as three `(lo, hi)` pairs, boundary conditions, and the
  initial-condition function. No grid, scheme, or rank information.
- **`Numerics`** — the discretization: resolution `n_global`, derivative and filter
  schemes, artificial-property constants, CFL, filter interval, process grid,
  halo width, and stretch mappings.

`setup(problem, numerics)` constructs the `Solver` and returns it with the
initialized conserved state `Q`. Initial conditions and Dirichlet forcing are
plain functions of physical coordinates `(x₁, x₂, x₃)` (and time `t` for
forcing) evaluated through `xcoord`, so they never see halos, offsets, or the
conserved layout. The `conserved_from_prim` step is EOS business, so a new
equation of state slots in without touching the IC path.

The payoff: the same `Problem` re-runs at a different resolution or scheme order
by changing only the `Numerics`, and a convergence study is a loop over
`Numerics` values. A future backend (GPU, permuted storage) would consume the
identical `Problem`.

## State representation and decomposition

`Decomp` (in `decomposition.jl`) owns the parallel layout. The global grid of
`n_global` points per dimension is split across a process grid `dims` (chosen
automatically via `MPI.Dims_create` or supplied explicitly). Each rank stores
its interior block padded by `n_halo` halo layers on every active side; the halo
width defaults to 4, matching the widest stencil reach of the default schemes.

Key derived quantities on `Decomp`:

- `active[d]` — whether dimension `d` is resolved (`n_global[d] > 1`). **Collapsed**
  dimensions (`n_global[d] == 1`) carry no halos, derivatives, filters, or
  exchanges; `n_halo_d[d]` (per-dimension halo pad) is zero for them. This is what
  makes a 1-D or 2-D or axisymmetric run cost 1-D/2-D memory and work rather
  than a thin 3-D slab.
- `neighbors[d]` — the `(lo, hi)` neighbor ranks along `d`, with `MPI.PROC_NULL` at
  open (non-periodic) global edges.
- `sub[d]` — a sub-communicator containing exactly the ranks along dimension
  `d`, used by the distributed line solves and line-wise reductions.

Fields are allocated by `field(decomp)` (a scalar with halos) and `allocate_state`
(the 4-D conserved array `Q[x, y, z, 1:n_cons]`). Two index helpers recur
throughout: `gidx(s, i, j, k)` maps a local interior index to the halo-offset
`CartesianIndex`, and `xcoord(s, d, i)` maps a local index to a physical
coordinate (including any stretch mapping and half-cell offset).

The `Solver` struct (in `rhs.jl`) is the backend container: it holds the
`Decomp`, the EOS and transport, the artificial-property parameters, the metric
and any fold specs, the per-dimension operator plans (`deriv_plans`, `filter_plans`),
pre-allocated primitive/gradient/flux/geometry scratch arrays, and the current
time and step. All hot-loop scratch is allocated once at setup; the RHS
allocates nothing.

## Compact schemes

A `CompactScheme` (in `kernels.jl`) describes a tridiagonal-LHS compact operator

    α gᵢ₋₁ + gᵢ + α gᵢ₊₁ = a₀ fᵢ + Σₘ coeffs[m] (fᵢ₊ₘ ± fᵢ₋ₘ)

with `−` (antisymmetric) for **derivatives** and `+` (symmetric, plus the center
weight `a₀`) for **filters**. Derivative right-hand sides are additionally scaled
by 1/h at plan time. Non-periodic edges are closed by explicit `ClosureRow`s —
each carries its own LHS `(sub, diag, super)` triple and an RHS stencil on the
first few points from the edge. High-edge rows are generated automatically by
mirroring (LHS sub/super swapped; RHS reversed, and negated for derivatives).

Presets:

- `lele_d1_6()` — sixth-order tridiagonal first derivative (α = 1/3, a = 14/9,
  b = 1/9) with third/fourth-order one-sided closures on the first two rows.
- `pade_d1_4()` — fourth-order Padé first derivative.
- `compact_filter(αf)` — the eighth-order Gaitonde–Visbal filter. `αf ∈ (−½, ½)`
  sets the strength (larger → weaker). Near closed edges the first row is left
  unfiltered and rows 2–4 apply centered compact filters of order 2/4/6 at the
  same αf — the standard reduced-order boundary cascade.

`kernels_banded.jl` generalizes this to a **banded LHS** of half-bandwidth `q`
via `BandedCompactScheme` (q = 1 is tridiagonal, q = 2 is pentadiagonal, which
admits schemes to tenth order). `lele_d1_10()` is the classical tenth-order
pentadiagonal derivative (β = 1/20, α = 1/2) with a C6-cascade closure on the
first three rows; its RHS reaches ±3, so the default `n_halo = 4` suffices.

Users supply their own operators by constructing these types directly and
passing them as `deriv=` or `filt=`.

## The distributed compact solve

Because a compact operator couples an entire grid line, evaluating it is solving
a banded linear system per line — and lines cross rank boundaries. This is the
heart of the parallel design. Consider the tridiagonal case (`tridiag.jl`).

**Local factorization.** Each rank factorizes its local tridiagonal block `T`
once at plan time with a Thomas factorization (`TriFactor`). Solving one RHS is
then a forward/back substitution (`solve_col!`).

**Cross-rank coupling via spikes.** On rank *p*, the global rows along a line
read

    T x = d − aL·x_prev_last·e₁ − cR·x_next_first·eₙ

where `aL` couples local row 1 to the previous rank's last unknown and `cR`
couples local row n to the next rank's first unknown. Writing `y = T⁻¹d`,
`v = T⁻¹(aL e₁)`, `w = T⁻¹(cR eₙ)`, the local solution is

    x = y − x_prev_last·v − x_next_first·w.

The vectors `v` and `w` — the **spikes** — depend only on the scheme, so they
are computed once at plan time. Evaluating `x` at the first and last row of
every rank yields a dense **2P × 2P reduced system** in the interface unknowns
`(x₁⁽⁰⁾, xₙ⁽⁰⁾, x₁⁽¹⁾, xₙ⁽¹⁾, …)`. That reduced matrix also depends only on the
scheme; it is assembled from a single `Allgather` of the spike corners
`(v₁, vₙ, w₁, wₙ)` and LU-factorized **once at plan time**.

**Per application**, then, costs: batched local Thomas solves (threaded over
lines), one `Allgather` of the two interface values per line, one dense
triangular solve for *all lines simultaneously* (`ldiv!` on the pre-factorized
reduced LU), and a threaded rank-local correction `x ← x − v·xl − w·xr`. This
reproduces the *exact* single-domain solution regardless of rank count — an
interface bug shows up as an O(1) error, not a small one, which is precisely how
the MPI test suite catches it.

Periodic single-rank directions reuse the same path through self-coupling (no
communication); non-periodic single-rank directions skip the reduced stage
entirely (`v = w = 0`).

`banded.jl` extends all of this to half-bandwidth `q`: the cross-rank coupling
runs through triangular `q×q` blocks, the spikes become `n×q` blocks, the
interface unknowns per rank become its first and last `q` values, and the
reduced system grows to `(2qP)²` — still one `Allgather` of four `q×q` corner
blocks per rank, still factorized once. The banded LU is unpivoted, which is
standard practice for the well-conditioned (if not strictly diagonally dominant)
compact LHS matrices.

**Memory layout.** `operators.jl` binds a scheme to a dimension as a `DirPlan`,
which packs each grid line into a scratch matrix `B`, runs the distributed
solve, and scatters back. The x-sweep uses an `n × lines` layout that is
contiguous both ways. The y- and z-sweeps would otherwise stride badly, so
`lines_transposed.jl` provides a `(lines × n)` layout in which fills, scatters,
eliminations, and spike corrections all iterate in the field's memory order
(innermost index along x), letting the Thomas/banded solves run as `@simd`
row-outer sweeps over contiguous line blocks, threaded by chunking lines. No
extra field copies and no cross-rank transposes are needed.

## Halo exchange

`halo.jl` fills rank-boundary halos with `MPI.Sendrecv!`. The three dimensions
are exchanged **sequentially**, and each dimension's exchange includes the
halo layers already filled by previous dimensions in its slabs — so edge and
corner halos are populated correctly without dedicated diagonal messages. At
non-periodic global edges the neighbor is `MPI.PROC_NULL`, making the Sendrecv a
no-op on that side; those physical-edge halos are left stale and **never read**,
because derivative closures are one-sided and filter closures are identities
there. This is what keeps the halo logic uniform.

Two batching optimizations matter for performance:

- `exchange_dim_batch!` exchanges many same-sized fields along one dimension in a
  single message per neighbor per phase. Flux arrays are differentiated only
  along their own direction, so they exchange halos only in that dimension; the
  conserved state is batched the same way per dimension.
- `exchange_state!` applies that to all conserved components at once — six
  messages per state exchange instead of `6 × (Ns + 4) × 3`.

`axis_fill!` handles the parity mirror fill used by the coordinate-singularity
folds (see below).

## Time integration

`timestep.jl` implements the five-stage fourth-order low-storage
Carpenter–Kennedy RK45. `step!` runs the five stages, each of which enforces
boundary conditions, evaluates the RHS, and applies the low-storage update

    du ← A·du + dt·dQ;   Q ← Q + B·du

with the stage tables `RKA`, `RKB`, `RKC`. The stage time `s.tstage` is set so
that time-dependent boundary forcing is sampled at the correct sub-step.

`compute_dt` forms a CFL-limited step from an acoustic rate `(|u_d| + c)/h_d`
(per active dimension, with the physical spacing including metric and stretch
factors) plus a diffusive rate built from the current molecular and artificial
transport coefficients, reduced across all ranks with `MPI.Allreduce`. The
artificial coefficients lag by one step (they were computed in the previous
RHS), which is standard and harmless at the usual CFL.

`curvature_rate` supplies the one term the per-dimension loop cannot see. A
*resolved* angular dimension already bounds its own geometric source, because
the advective rate |u_ang|/(r·Δang) dominates the source rate |u_ang|/r. A
*collapsed* one is skipped by the loop entirely, yet ρu_θ²/r keeps driving u_r
as a stiff source at small r — axisymmetric-with-swirl being the case that
matters. `curvature_rate` is dispatched on the metric and returns zero for
Cartesian and for every resolved angular dimension. `dt_report` is the
diagnostic companion: it repeats the `compute_dt` sweep while tracking *which*
node and which term set the limit, and returns `(dt, rank, index, coords, dim,
kind)` with `kind ∈ (:acoustic, :diffusive, :curvature)`. See the CFL section of
the README for what the answers mean on polar grids.

`run!` is the outer loop: step, advance time, and every `filter_interval` steps
call `filter_state!`, which applies the compact filter to every conserved
component along every active dimension (with batched halo exchange and the
correct axis parity per component). An optional `callback(s, Q)` runs after each
step for diagnostics or output.

## The RHS: a walkthrough

`compute_rhs!` (in `rhs.jl`) evaluates dQ/dt into the interior of `dQ`,
assuming boundary conditions are already enforced on `Q`. Step by step:

1. **Exchange state halos** (`exchange_state!`) and **recover primitives**
   (`primitives!`) — ρ, u, v, w, p, T_ion, sound speed, mixture cₚ, and the mass
   fractions — over the full padded arrays.
2. **Velocity gradients.** For each direction `d` and velocity component `j`,
   apply the compact derivative and scale by 1/h_d (which carries the stretch
   Jacobian). `grad_u[d, j]` holds the d-derivative of the j-th component. Collapsed
   dimensions contribute zero.
3. **Curvature corrections** (`metric_correct_gradients!`) add the algebraic
   terms from the rotating unit vectors so that `grad_u` holds *physical* tensor
   components (e.g. in cylindrical, `grad_u[2,2] += u_r/r`).
4. **Artificial properties** (`compute_artificial!`) fill μ\*, β\*, κ\*, and the
   per-species D\* from the current gradients and primitives.
5. **Scalar gradients.** Temperature and each mass fraction, same derivative +
   scaling, for the heat and species-diffusion fluxes.
6. **Flux assembly** (`assemble_fluxes!`) builds the conservative fluxes
   `flux[d, c]` pointwise: species convection plus diffusion with a correction
   velocity, momentum convection plus pressure and the full viscous stress
   (including the artificial bulk term), and the energy flux (enthalpy
   convection, viscous work, Fourier heat flux, and enthalpy diffusion
   Σ h_k J_k). This is *the single place fluxes are built.*
7. **Flux halo exchange**, per dimension, batched over conserved components.
8. **Metric divergence.** For each component `c` and direction `d`, form the
   area-weighted flux `A_d F_d`, take its compact derivative, and accumulate
   `dQ[c] −= inv_J · ∂(A_d F_d)`. `J = h₁h₂h₃` and `A_d = J/h_d`, so the compact
   derivatives act on area-weighted fluxes and the divergence is exact for the
   metric.
9. **Momentum sources** (`add_metric_sources!`) add the algebraic ∇·Π terms that
   appear in curvilinear coordinates (e.g. +Π_θθ/r in cylindrical).
10. **Boundary corrections.** For every face, `correct_rhs!` applies any
    characteristic (NSCBC) correction to the RHS.

Every loop over the interior is threaded over the outermost index; every MPI
call sits in a serial section between threaded regions.

## Artificial fluid properties

`artificial.jl` implements the Cook (2007) model. The sensors are **undivided**
explicit fourth differences δ⁴ = (1, −4, 6, −4, 1) in the computational indices,
so the formal grid-spacing powers reduce to per-dimension weights: h² for the
shear/bulk viscosities (from |∇⁴S|·Δ⁶) and h for conductivity and species
diffusivity (from |∇⁴e|·Δ⁵ and |∇⁴Y|·Δ⁵). The Gaussian test filter of the
original model is approximated by one compact-filter pass (`smooth!`).

Concretely, `compute_artificial!`:

- Builds the strain-rate magnitude |S| from the metric-corrected `grad_u`, computes
  its δ⁴ sensor summed over directions, smooths it, and sets
  μ\* = C_μ·ρ·sensor and β\* = C_β·ρ·sensor.
- Computes the internal energy directly from `Q` (EOS-agnostic), takes its δ⁴
  sensor, smooths, and sets κ\* = C_κ·(ρc/T_ion)·sensor.
- For each species, takes the δ⁴ sensor of Y_k, smooths, and sets
  D\*_k = C_D·c·sensor_k.

Constants live in `ArtParams` (defaults C_μ = 0.002, C_β = 1.0, C_κ = 0.01,
C_D = 0.01) and should be revisited per configuration. `enabled=false` skips the
whole computation and leaves the coefficient arrays zero. On curvilinear grids
the sensors act in computational index space — a grid-based regularization,
consistent with resolving power following the mesh.

## Curvilinear metrics and the discrete GCL

`metric.jl` supports Cartesian, cylindrical (r, θ, z), and spherical (r, θ, φ)
coordinates, uniformly spaced in the computational directions. Geometry enters
through orthogonal scale factors `h_d(x)` in three places:

1. **Divergence** — `∇·F = (1/J) Σ_d ∂(A_d F_d)/∂x_d`, applied per flux
   component with the compact derivative (step 8 above).
2. **Gradients** — `(∇f)_d = (1/h_d) ∂f/∂x_d`, plus the curvature corrections to
   the physical velocity-gradient components.
3. **Momentum sources** — the algebraic terms of ∇·Π from the rotating basis.

The geometric arrays (J⁻¹, A_d, 1/h_d, 1/r, cotθ/r) are evaluated analytically
over the full padded arrays, so they need no halo exchange; scale factors are
clamped away from zero so that stale physical-edge halo values stay finite.

**Stretched meshes** are a `Stretch(x, dxds)` mapping per dimension from a
uniform computational coordinate s ∈ [0, 1]. The mapping Jacobian simply
multiplies that dimension's scale factor, so stretching composes with
cylindrical/spherical geometry with no new operator or parallel machinery — the
curvature corrections were refactored to be *additive* on top of a generic 1/h
scaling. `sine_cluster` provides a closed-form interior-clustering map. Stretched
dimensions must be non-periodic.

**Discrete GCL.** A subtlety: for the spherical θ-momentum source, the flux
divergence `−inv_J·D_ξ2(A₂·p)` and the analytic source `+(cotθ/r)·Π_φφ` cancel
*analytically* for a uniform state, but `D_ξ2(sinθ) ≠ cosθ` *discretely*, leaving
an O(h⁴) freestream residual. `gcl_cotr!` restores exact discrete freestream
preservation by redefining `cotθ/r` as `inv_J·D_ξ2(A₂)` — the identical operator
(same scheme, same fold, same antipodal sign) applied to the identical area
factor — so the source cancels the divergence node-by-node by construction. The
freestream-preservation test verifies this to machine zero in every metric.

## Coordinate-singularity folds

Cylindrical and spherical grids have coordinate singularities (r = 0, the
spherical poles) where scale factors vanish. Rather than excluding them,
CompactLES **regularizes** them on a half-offset grid — `r_i = (i − ½)h`, so no
node sits on the singularity and no scale factor is zero — plus parity/antipodal
folds implemented in `folds.jl`.

**The axisymmetric axis** (`AxisBC`, cylindrical, θ collapsed) is the simplest
case: halos below the axis are mirror-filled with a per-field sign (scalars and
u_z even, u_r and u_θ odd; radial fluxes pick up the A₁ = r area weight, which
is odd and flips parity), interior stencils run to the first node, and the
implicit LHS coupling to the ghost unknown folds analytically onto the matrix
diagonal — per solution parity, since derivatives flip field parity and filters
preserve it. `operators.jl` implements the diagonal fold (`b[1] += σg·α`); the
scheme is planned in both parities and the RHS reads the mirror-filled halo.

**The resolved cases** — resolved-θ cylindrical axis, spherical origin, spherical
poles — all reduce to one structure: a fold whose partner is *antipodal* rather
than the line itself:

    cylindrical axis:  (−r, θ)      ≡ (r, θ+π)
    spherical origin:  (−r, θ, φ)   ≡ (r, π−θ, φ+π)
    spherical poles:   (−θ, φ)      ≡ (θ, φ+π)

The even/odd decomposition `e = ½(f + σ f∘M)`, `o = ½(f − σ f∘M)` — with M the
pairing map and σ the component's antipodal basis sign — turns each into two
problems the parity-fold machinery already solves exactly: e is even across the
singular point, o is odd. The folded compact plans apply unchanged and
reconstruction is the inverse butterfly. Component signs are tabulated in
`folds.jl`: cylindrical axis (u_r, u_θ, u_z) → (−1, −1, +1); spherical origin
(u_r, u_θ, u_φ) → (−1, +1, −1); poles (+1, −1, −1). Radial/polar flux parities
fold in the area weights.

**Parallel layout.** When the shifted or reversed dimension is split across
ranks, the pairing is realized by whole-block `MPI.Sendrecv!` exchanges between
partner ranks (the e-keeper/o-keeper butterfly). Setup validates the required
layout restrictions: the shifted dimension must be on one rank or split into an
even number of uniform blocks (partner at +P/2), and the reversed dimension
likewise (reflected partner). On-rank pairing runs both parity plans and selects
per half; off-rank pairing costs two full-block exchanges per application.

This is the newest and least-exercised machinery in the package. The antipodal
sign tables and off-rank e/o bookkeeping are exactly where bugs hide — the MPI
test suite has already found and fixed two (an odd-combo mirror sign and a
reversed-dimension slot flip that only appeared when the reversed dimension was
split). The full-ball origin+poles combination has had the least scrutiny.

## Multicomponent thermodynamics

`physics.jl` defines the EOS contract the solver talks to:

- `nspecies(eos)` → number of species Ns
- a `(ρ, e, Y) → (p, T_ion, c, cₚ_mix)` state evaluation (via `primitives!`)
- `species_enthalpy(eos, k, T_ion)` → partial specific enthalpy h_k(T_ion)

`IdealMixture` (per-species γ_k and R_k, linear mixture rules) is the provided
implementation. Because hot loops reach the EOS through a function barrier
(`primitives!` dispatches on `typeof(s.eos)`), an abstractly typed `eos` field
costs one dynamic dispatch per array pass, not per point — so cubic
(Peng–Robinson), polynomial-cₚ, or tabular models can be dropped in behind the
same interface without touching the flow solver or paying a per-point penalty.

Species diffusion uses a common molecular diffusivity (μ₀/Sc) plus the
per-species Cook artificial D\*_k, with a **correction velocity** in the flux
assembly:

    J_k = −ρ D_k ∇Y_k + ρ Y_k Σ_j D_j ∇Y_j

which enforces Σ_k J_k = 0 exactly. Enthalpy diffusion Σ h_k J_k enters the
energy flux.

## Characteristic boundary conditions

`nscbc.jl` implements Poinsot–Lele NSCBC subsonic outflow and inflow as **LODI
wave-amplitude corrections of the RHS**, not hard state enforcement. The
interior scheme's one-sided closure rows already produce the physically
evaluated incoming wave in `dQ`; the correction replaces only that wave's
amplitude and propagates the difference analytically to the conserved
components.

- **Outflow** (`NSCBCOutflowBC(pinf=..., sigma=...)`) imposes the incoming
  acoustic amplitude `L* = K(p − p∞)` with `K = σ(1 − M²)c/L_ref`, mapping the
  delta onto ρ, momentum, and energy (species scale with Y_k, energy through
  φ = cv_m/R_m). It optionally carries a β_t-weighted share of the transverse
  terms (Yoo & Im 2007), with the default using the local Mach number. Supersonic
  points get no correction (all waves leave).
- **Inflow** (`NSCBCInflowBC(u=..., T_ion=..., Y=...)`) replaces every incoming wave
  (acoustic, entropy, transverse, species) with a relaxation toward the targets
  while keeping the single outgoing acoustic wave as computed. Targets may be
  constant or a pointwise `(x, y, z, t) -> Prim` function evaluated at the stage
  time. The LODI derivation of each relaxation form is spelled out in the source
  so the signs can be audited.

The full LODI algebra — the mapping from wave-amplitude deltas to conserved
components, including the species contribution to the energy through ∂φ/∂Y_k — is
written out in the file's header comments.

Time-dependent full-state forcing is `DirichletBC((x,y,z,t) -> Prim)`, suitable
for pistons, oscillating drivers, and supersonic inflow. Note that a full-state
Dirichlet over-constrains a *subsonic* boundary and will reflect — characteristic
inflow is the right tool there.

## Threading and MPI discipline

The solver runs with `MPI.Init(threadlevel=:funneled)`: **all MPI calls are made
from serial sections**, never from inside a `Threads.@threads` region. The
pattern throughout is threaded compute → serial communication → threaded
compute. Shared-memory parallelism is over grid lines (in the operators) and over
the outermost array index (in the pointwise loops). Halo buffers are typed
concretely as `Float64`, so the code is currently Float64-only.

## Extension points

**New compact kernels.** Construct a `CompactScheme` (tridiagonal) or
`BandedCompactScheme` (banded) with the interior LHS coefficient(s), the RHS
half-stencil, the symmetric flag (filters) or not (derivatives, auto-scaled by
1/h), and closure rows for non-periodic edges. Pass it as `deriv=` or `filt=`.
High-side closures are mirrored automatically. Watch the halo width: a scheme
whose RHS reaches ±m needs `n_halo ≥ m`.

**New boundary conditions.** Subtype `BoundaryCondition` and implement
`enforce!(bc, Q, s, dim, side)` for hard state enforcement on the wall plane
(index sets come from `wallplane`), and/or `correct_rhs!(bc, s, Q, dQ, dim,
side)` for a characteristic RHS correction. Declare periodicity via `isperiodic`.

**New physics.** `assemble_fluxes!` is the single place fluxes are built and
`compute_artificial!` the single place regularization is built — extend the model
there.

**New equations of state.** Implement the three-function EOS contract
(`nspecies`, the state evaluation in `primitives!`, and `species_enthalpy`) plus
`conserved_from_prim`. The function-barrier design keeps this cheap.

**New output formats.** `io.jl` shows the pattern: per-rank writes plus a rank-0
container, using `MPI.Allgather` only to collect piece extents. HDF5/XDMF for
very large runs would follow the same shape.
