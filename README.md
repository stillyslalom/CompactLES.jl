# CompactLES

A draft compressible LES solver in Julia: Lele-style compact finite
differences, Cook (2007) artificial fluid properties for shock and subgrid
regularization, a Gaitonde–Visbal eighth-order compact filter, low-storage
RK45 time integration, shared-memory threading over grid lines, and MPI domain
decomposition with halo exchange plus a genuinely distributed tridiagonal
solve for the globally coupled compact schemes.

This is untested scaffolding intended as a starting point, not a validated
production code. It targets MPI.jl v0.20; expect to shake out small API and
indexing issues on first run.

## Layout

    src/decomposition.jl   3-D MPI Cartesian decomposition, sub-communicators, buffers
    src/halo.jl            sequential-dimension Sendrecv halo exchange (fills corners)
    src/tridiag.jl         Thomas factorization + spike/reduced-interface distributed solve
    src/kernels.jl         CompactScheme abstraction and presets (C6 derivative, C8 filter)
    src/boundary.jl        BoundaryCondition types and wall-plane enforcement
    src/operators.jl       DirPlan: per-dimension operator plans, threaded application
    src/physics.jl         EOS abstraction, multicomponent ideal mixture, primitives
    src/metric.jl          Cartesian / cylindrical / spherical orthogonal metrics
    src/artificial.jl      Cook artificial μ*, β*, κ*, D* from undivided δ⁴ sensors
    src/rhs.jl             Solver container, flux assembly, conservative NS RHS
    src/timestep.jl        RK45 (Carpenter–Kennedy), CFL timestep, run loop
    src/nscbc.jl           characteristic (NSCBC) subsonic outflow
    src/io.jl              per-rank checkpoint / restart
    src/problem.jl         frontend: Prim, function ICs/BCs, Problem/Numerics/setup
    examples/              Taylor–Green, two-gas shock tube, oscillating driver

## Numerics

Interior spatial derivatives use the sixth-order tridiagonal Lele scheme
(α = 1/3, a = 14/9, b = 1/9) with third/fourth-order one-sided closures on the
first two rows at non-periodic edges. Dealiasing and shock-adjacent smoothing
come from the eighth-order Gaitonde–Visbal compact filter applied to the
conserved variables once per step (identity rows near closed edges). The gas
is calorically perfect; the equations are solved in conservative form with the
full viscous stress (including the artificial bulk term) and Fourier heat flux.

Cook's artificial fluid properties supply localized shear viscosity, bulk
viscosity, and conductivity. Fourth derivatives of the strain magnitude and
internal energy are approximated by undivided explicit fourth differences, so
the formal Δ⁶ and Δ⁵ spacing powers reduce to per-dimension weights h² and h;
the Gaussian test filter is approximated by one compact-filter pass. Constants
(Cμ = 0.002, Cβ = 1.0, Cκ = 0.01) follow the usual Cook-family calibrations
and should be revisited for your configurations.

Time integration is the five-stage fourth-order low-storage Carpenter–Kennedy
RK45; the timestep combines acoustic and diffusive CFL constraints with an
all-rank reduction.

## Parallel design

Each rank owns an interior block padded by H = 4 halo layers. Halo exchange
runs the three dimensions sequentially with `MPI.Sendrecv!`, including
previously exchanged halos in each slab so corners fill without diagonal
messages; open (non-periodic) edges use `MPI.PROC_NULL`. Physical-edge halos
are deliberately never read: derivative closures are one-sided and filter
closures are identities, which is what makes the halo logic uniform.

Compact schemes couple entire grid lines, so the tridiagonal solve is
distributed with the spike (reduced-interface) method. On rank p the global
rows along a line read

    T x = d − aL·x_prev_last·e₁ − cR·x_next_first·eₙ

with T the local tridiagonal block. Writing y = T⁻¹d, v = T⁻¹(aL e₁),
w = T⁻¹(cR eₙ), the local solution is x = y − x_prev_last·v − x_next_first·w,
and evaluating at the first and last rows of every rank yields a dense 2P×2P
system in the interface unknowns per line. Because that matrix depends only on
the scheme, it is assembled from a single Allgather of (v₁, vₙ, w₁, wₙ) and
LU-factorized once at plan time. Each application then costs: batched local
Thomas solves threaded over lines, one Allgather of two interface values per
line, one dense triangular solve for all lines simultaneously, and a threaded
local correction. Periodic single-rank directions reuse the same path through
self-coupling (no communication); non-periodic single-rank directions skip the
reduced stage. All MPI calls are made from serial sections
(`MPI.Init(threadlevel=:funneled)`).

## Extension points

New compact kernels: construct a `CompactScheme` with the interior LHS
coefficient α, the RHS half-stencil weights, the symmetric flag (filters) or
not (derivatives, auto-scaled by 1/h), and a vector of `ClosureRow`s for
non-periodic edges; pass it as `deriv=` or `filt=` to `Solver`. High-side
closures are mirrored automatically.

New boundary conditions: subtype `BoundaryCondition` and implement
`enforce!(bc, Q, s, dim, side)`; declare periodicity via `isperiodic` if
applicable. Wall-plane index sets come from `wallplane`.

New physics: `assemble_fluxes!` is the single place fluxes are built;
`compute_artificial!` is the single place the regularization is built.

### Pentadiagonal and higher-order kernels

`src/banded.jl` generalizes the distributed solve to LHS half-bandwidth q
(tridiagonal is q = 1, pentadiagonal q = 2), which admits compact schemes up
to 10th order. Cross-rank coupling now runs through triangular q×q blocks AL
and CR built from the interior LHS coefficients; the spikes become n×q blocks
V and W, the interface unknowns per rank become its first and last q values,
and the reduced system grows to (2qP)² — still assembled from a single
Allgather of the four q×q spike corner blocks per rank and LU-factorized once
at plan time. Per application the extra cost over the tridiagonal path is
mild: the banded local solve does ~q² work per row, the interface Allgather
carries 2q values per line, and the correction sums q spike pairs per point.
The banded LU is unpivoted; compact LHS matrices (Lele C10: α = 1/2,
β = 1/20) are not strictly diagonally dominant but are well-conditioned, and
unpivoted solves are standard practice for them.

Banded schemes are described by `BandedCompactScheme` (interior LHS
coefficient vector, RHS half-stencil, symmetric flag, closure rows carrying a
full centered 2q+1 LHS vector); `lele_d1_10()` provides the classical
10th-order pentadiagonal derivative (β = 1/20, α = 1/2, a = 17/12,
b = 101/150, c = 1/100) with a C6-cascade boundary closure on the first three
rows. Pass it directly: `Solver(...; deriv=lele_d1_10())`. Its RHS reaches ±3
so the default H = 4 halo suffices; a 10th-order pentadiagonal *filter*
(RHS ±5) would need `H=5`.

### Multicomponent flow

The conserved layout is (ρY₁ … ρY_Ns, ρu, ρv, ρw, E); mixture density is the
sum of partial densities and every species carries its own transport equation.
Thermodynamics goes through an EOS contract — `nspecies`, a (ρ, e, Y) →
(p, T, c, cp) state evaluation, and `species_enthalpy(eos, k, T)` — with
`IdealMixture` (per-species γ_k and R_k, linear mixture rules) as the provided
implementation; cubic, polynomial-cp, or tabular models slot in behind the
same interface without touching the flow solver, since hot loops reach the
EOS through a function barrier. Species diffusion uses a single common
diffusivity (molecular μ₀/Sc plus Cook's artificial D* built from summed
|δ⁴Y_k| sensors), which with ΣY_k = 1 keeps the diffusive fluxes mass
consistent identically; per-species D*_k with a correction velocity is a
TODO. Enthalpy diffusion Σ h_k J_k is included in the energy flux.

### Cylindrical and spherical coordinates

The `metric` keyword selects Cartesian (default), cylindrical (r, θ, z), or
spherical (r, θ_polar, φ) coordinates on a uniformly spaced coordinate grid.
Geometry enters in three places: flux divergences become
J⁻¹ Σ_d ∂(A_d F_d)/∂x_d with J = h₁h₂h₃ and A_d = J/h_d (the compact
derivatives act on area-weighted fluxes); scalar gradients pick up 1/h_d and
velocity gradients pick up the standard curvature corrections so that τ and
|S| are built from physical tensor components; and the momentum equations
gain the algebraic source terms of ∇·Π in curvilinear coordinates
(e.g. +Π_θθ/r and −Π_θr/r in cylindrical). Geometric arrays are evaluated
analytically over the padded arrays, so they need no halo exchange.
Coordinate singularities are not treated: exclude the axis/origin (r > 0) and
the spherical poles (0 < θ < π) via the `origin` keyword, e.g.
`metric=CylindricalMetric(), origin=(0.2, 0.0, 0.0)` for an annular sector.
The artificial-property sensors act in computational index space — a
grid-based regularization, consistent with resolution following the mesh.

## Frontend

Problems are specified in primitive, pointwise terms, decoupled from the
backend. `Prim(; u, p, T or rho, Y)` describes a state; the EOS supplies the
missing thermodynamic variable and the primitive→conserved conversion, so
future cubic/tabular models plug into the same path. Initial conditions are
plain functions (x₁, x₂, x₃) → Prim evaluated at physical coordinates —
they never see ranks, halos, offsets, or component layouts (and should be
pure, since initialization is threaded). Time-dependent boundary forcing is
`DirichletBC((x₁, x₂, x₃, t) -> Prim)`, evaluated at the RK stage time —
suitable for pistons, oscillating drivers, and supersonic inflow; note that
full-state Dirichlet over-constrains a subsonic boundary and will reflect
(characteristic inflow is the right tool there and remains a TODO).

`Problem` bundles physics and geometry only (EOS, transport, metric,
coordinate domain as (lo, hi) pairs, BCs, IC); `Numerics` bundles resolution,
schemes, artificial-property constants, CFL, and the process grid.
`setup(problem, numerics)` returns `(solver, Q)` with the state initialized.
The split is deliberate: the same Problem re-runs at different resolutions or
orders unchanged, and a future backend (GPU, permuted storage) consumes the
identical specification. A convergence study is a loop over `Numerics`.

```julia
prob = Problem(
    eos = IdealMixture([IdealSpecies{Float64}("light", 1.0, 1.4),
                        IdealSpecies{Float64}("heavy", 0.2, 1.09)]),
    domain = ((0.0, 1.0), (0.0, 0.05), (0.0, 0.05)),
    bcs = ((SlipWallBC(), NSCBCOutflowBC(pinf=0.1)),
           (PeriodicBC(), PeriodicBC()), (PeriodicBC(), PeriodicBC())),
    ic = (x, y, z) -> begin
        θ = tanh_blend(x, 0.5, 0.01)
        Prim(Y=(1-θ, θ), rho=1 - 0.375θ, p=1 - 0.9θ)
    end)
s, Q = setup(prob, Numerics(nglob=(512, 16, 16)))
run!(s, Q; tfinal=0.25)
```

### Reduced dimensionality and the cylindrical axis

Collapsed dimensions (`nglob[d] = 1` with a periodic BC pair) carry no
derivatives, filters, halo padding, or exchanges — a 2-D axisymmetric run
costs 2-D memory and 2-D work, not a thin 3-D slab — while keeping their
velocity component and all metric source terms. That combination is exactly
axisymmetric (r, z) flow with optional swirl: ∂/∂θ ≡ 0, but u_θ evolves and
the centrifugal/Coriolis-type Π-sources act. 1-D radial problems
(`nglob = (Nr, 1, 1)`) work the same way.

The cylindrical axis is regularized rather than excluded: `AxisBC()` at low r
(with `CylindricalMetric`, collapsed θ, unstretched r) switches that
dimension to a half-offset grid r_i = (i − ½)h, so no node sits at r = 0 and
no scale factor vanishes. Axis closure is by parity: halos below the axis are
mirror-filled with per-field signs (scalars and u_z even; u_r and u_θ odd;
radial fluxes as derived per component, with the A₁ = r area weight flipping
parity), interior stencils run to the first node, and the implicit LHS
coupling to the ghost unknown folds analytically onto the matrix diagonal —
per solution parity, since derivatives flip field parity and filters preserve
it. This works for both tridiagonal and pentadiagonal kernels. The full disk
with resolved θ (which needs θ+π pairing across the axis) and the spherical
origin/poles remain open. See `examples/converging_shock.jl` for a
Guderley-flavored axis stress test.

NSCBC inflow targets may now be time- and space-dependent:
`NSCBCInflowBC(..., target=(x, y, z, t) -> Prim(...))` overrides the constant
targets pointwise at the RK stage time.

### Resolved-axis coupling and the spherical origin/poles

All three remaining coordinate singularities reduce to one structure: a fold
whose partner is antipodal rather than the line itself —
(−r, θ) ≡ (r, θ+π) at the cylindrical axis, (−r, θ, φ) ≡ (r, π−θ, φ+π) at
the spherical origin, and (−θ, φ) ≡ (θ, φ+π) at the poles. The even/odd
decomposition e = ½(f + σ f∘M), o = ½(f − σ f∘M) (σ the component's
antipodal basis sign, M the pairing map) turns each into two problems the
existing parity-fold machinery already solves exactly: e is even across the
singular point and o is odd, so the folded compact plans apply unchanged,
and reconstruction is the inverse butterfly with the single sign σ.
Component signs: cylindrical axis (u_r, u_θ, u_z) → (−1, −1, +1); spherical
origin (u_r, u_θ, u_φ) → (−1, +1, −1) (ê_θ is invariant at the antipode);
poles (+1, −1, −1). Radial/polar flux parities include the area weights
(A₁ = r odd, A₁ = r²sinθ even, A₂ = r sinθ odd in θ); tables are derived in
the source.

Usage: `AxisBC()` now supports resolved θ (even Nθ over 2π); `OriginBC()` at
low r with `SphericalMetric` (θ range symmetric about π/2, φ collapsed or
even over 2π); `PoleBC()` at BOTH ends of θ over (0, π), half-offset grids
throughout. Degenerate reductions come free: 1-D spherical blast problems
(`nglob = (Nr, 1, 1)` with `OriginBC`), axisymmetric spherical (φ collapsed,
pairing collapses to a θ-reversal), and the axisymmetric cylindrical axis
(pairing collapses to the plain parity fill).

Parallel-layout restrictions (validated at setup): the shifted dimension is
on one rank or split into an even number of uniform blocks (partner rank at
+P/2, same local slot); the reversed dimension of the origin likewise
(reflected partner). Costs: on-rank pairing runs both parity plans and
selects per half (2× that operator); off-rank pairing costs two full-block
pairwise exchanges per application. Poles fold both ends of θ, implemented
for tridiagonal and pentadiagonal kernels alike.

Honest flags: this is the least-tested machinery in the package — the
antipodal sign tables and the off-rank e/o bookkeeping are exactly where a
first run will find bugs, and the derivations are written into folds.jl to
be audited line by line. The origin+poles combination (full ball) composes
in principle but has had the least design scrutiny.

## Running

    julia --project=. -t auto examples/taylor_green.jl
    mpiexec -n 8 julia --project=. -t 2 examples/taylor_green.jl
    mpiexec -n 4 julia --project=. -t 2 examples/shock_tube.jl
    mpiexec -n 2 julia --project=. -t 2 examples/piston_driver.jl

Instantiate first with `julia --project=. -e 'using Pkg; Pkg.instantiate()'`.
Each rank's local extent in every dimension must be at least 9 (filter closure
plus stencil width); choose `dims` accordingly for thin dimensions.

## TODO triage (updated)

Implemented this round, in priority order:

1. **NSCBC subsonic outflow** (`NSCBCOutflowBC(pinf=..., sigma=0.25)`) — the
   biggest physics gap for open-domain shock work. Implemented as a LODI
   wave-amplitude correction of the RHS: the incoming acoustic wave computed
   by the one-sided closure is replaced by K(p − p∞), K = σ(1 − M²)c/L_ref,
   with the delta mapped analytically onto the conserved components (species
   scale with Y_k; energy through φ = cv_m/R_m). Supersonic points are left
   untouched. Transverse-term damping (Yoo & Im) and a matching NSCBC inflow
   remain open.
2. **Batched, direction-aware halo exchange** — flux arrays are
   differentiated only along their own direction, so they now exchange halos
   only in that dimension, with the whole flux block packed into one message
   per neighbor per phase: six flux messages per RHS instead of
   6 × (Ns + 4) × 3. The conserved state is batched the same way.
3. **Checkpoint / restart** (`save_checkpoint`, `load_checkpoint!`) —
   per-rank raw binary with a validating header (grid, decomposition,
   conserved layout, t, step). Dependency-free; same-decomposition restarts
   only.
4. **Filter boundary closures** — the identity rows near closed edges are
   replaced by the standard reduced-order cascade (row 1 unfiltered, rows
   2–4 centered compact filters of order 2/4/6 at the same αf), restoring
   near-wall dissipation control.
5. **Isothermal no-slip walls** — `NoSlipWallBC(Twall=...)` now enforces wall
   energy via the EOS (adiabatic remains the NaN default).

Second round, in priority order:

6. **NSCBC subsonic inflow** (`NSCBCInflowBC(u=..., T=..., Y=...)`) — every
   incoming wave (acoustic, entropy, transverse, species) is replaced by a
   relaxation toward the targets while the outgoing acoustic wave is kept as
   computed; the LODI derivation of each relaxation form is spelled out in
   the source so the signs can be audited. Constant targets for now.
7. **Stretched meshes** (`Stretch(x, dxds)`, canned `sine_cluster`) —
   per-dimension monotone mappings from a uniform computational coordinate,
   set in `Numerics` (clustering is a resolution decision, so it lives with
   the discretization, not the Problem). The mapping Jacobian simply
   multiplies that dimension's metric scale factor, so stretching composes
   with cylindrical/spherical geometry and required no new operator or
   parallel machinery — only a refactor making curvature corrections additive
   on top of a generic 1/h scaling. Stretched dims must be non-periodic.
8. **Per-species artificial diffusivities** — D*_k from per-species |δ⁴Y_k|
   sensors, with a correction velocity in the flux assembly
   (J_k = −ρD_k∇Y_k + ρY_k Σ_j D_j∇Y_j) enforcing Σ_k J_k = 0 exactly.
   Costs Ns filter sweeps per RHS.

Third round, in priority order:

9.  **Efficient 1-D/2-D domains** — collapsed dimensions with per-dimension
    halo padding (zero for collapsed dims), skipped operators/exchanges, and
    retained metric sources; axisymmetric-with-swirl falls out naturally.
10. **Cylindrical axis regularization** (`AxisBC`) — half-offset grid plus
    parity-folded compact closures; details above.
11. **Time-dependent NSCBC inflow targets** — pointwise Prim-function
    override at the RK stage time.

Fourth round, in priority order:

12. **NSCBC transverse-term accounting** (Yoo & Im 2007) — the imposed
    outflow wave now carries a β_t-weighted share of the transverse
    contribution 𝒯_in = u_t·∇_t p + ρc²∇_t·u_t ∓ ρc u_t·∇_t u_n (derived in
    the source from the p and u_n equations projected on the incoming
    characteristic; ρc² stands in for γp, keeping it EOS-agnostic). β_t = 0
    recovers plain LODI, 1 is full accounting, and the default uses the
    local Mach number per the Yoo–Im damping recommendation. Terms along
    collapsed dimensions vanish automatically.
13. **Transposed line path for y/z sweeps** — the dominant cache cost was
    strided reads in the y/z line fills. Dims 2 and 3 now use a
    (lines × n) scratch layout with fills, scatters, eliminations, and spike
    corrections all iterating in the field's memory order: the Thomas and
    banded solves become row-outer sweeps vectorized (@simd) across
    contiguous line blocks, threaded by chunking lines. The x sweep keeps
    the original layout, which was already contiguous both ways. No extra
    field copies, no cross-rank transposes.
14. **VTK output** (`save_vtk`) — dependency-free parallel rectilinear
    output: per-rank .vtr with appended raw Float32 payloads plus a .pvtr
    container, physical coordinates including stretch mappings; open in
    ParaView/VisIt. Pieces abut without ghost overlap (cell-based filters
    may show seams).

Fifth round: resolved-θ axis coupling, the spherical origin, and the
spherical poles, via the antipodal even/odd fold described above.

Still open: NSCBC inflow transverse terms; HDF5/XDMF for very large runs;
GPU.

## Caveats and TODO

- Entirely untested: written without a Julia/MPI environment available. Expect
  minor MPI.jl API drift (Dims_create/Cart_sub signatures), typos, and
  off-by-one issues to surface on first compile/run.
- NSCBC outflow (with Yoo–Im transverse accounting) and inflow are
  implemented; inflow transverse terms are not.
- VTK output only; no HDF5/XDMF, no GPU path.
- The artificial-property sensors clamp indices at physical edges, degrading
  the δ⁴ estimate there; acceptable for slip-wall shock tubes, worth revisiting
  for wall-bounded turbulence.
- Float64 only (halo buffers are typed concretely).
- Multicomponent: per-species artificial diffusivities with correction
  velocity are in; Soret/Dufour and reacting-source hooks are TODO.
- Curvilinear: all coordinate singularities (cylindrical axis, spherical
  origin and poles) are regularized via parity/antipodal folds; wall BCs
  assume coordinate-surface walls.
