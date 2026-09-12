Wall and AMR accuracy investigation, 2026-09-05

Roadmap N6 (September 2026) promoted the probes below into durable studies:
the cases, references and regional norms are `test/smooth_cases.jl`, the
gated rows are the closure-truncation and smooth-evolution sections of
`test/convergence.jl`, and `bench/boundaryorder.jl` is now the full accuracy
matrix rather than the probe script this file's reproduction line names.
The measurements are in CALIBRATION_APPENDIX.md under "The smooth-evolution
accuracy matrix"; this file is kept as the record of the audit that
motivated them, and its numbers are the audit's.

The present default boundary treatments do limit global accuracy on smooth
problems. The immediate targets are the wall filter and the interface flux
divergence. Raising the interior derivative order does not remove either
limiter. Existing closure options already demonstrate substantial improvement;
a wholesale interface rewrite is not the first experiment to undertake.

Reproduction: `julia --project=. -t 1 test/convergence.jl` and
`julia --project=. -t 1 bench/boundaryorder.jl`. Measurements below are serial
CPU Float64. The diagnostic changes no production operator or default. Shock
and Float32 outcomes cited below are prior measurements in CALIBRATION_APPENDIX.md,
not new runs of those batteries.

**What the wall measurements mean.** The existing convergence suite passed
and reproduced every recorded slope. The sub-second-order result is
`|F f - f|` for ONE application of the default C8 state filter, measured on
`exp(sin(3x))`: 1.88 in maximum norm. Its near-wall rows are identity/F2/F4/F6;
the F2 row dominates. `compact_filter(closures=:onesided)` raises that same
measurement to 8.07. The default C6 derivative separately measures 3.17,
or 4.02 with `:cascade4`.

The measured 5.88/7.91 with C6/C8 `:brady_livescu` are not their formal
pointwise boundary orders. Their rows are fifth/seventh order. The new
diagnostic differentiates x^6/x^8 on N=17,33,65,129 wall grids, using actual
h=1/(N-1) in the slope. It measures 5.000 for C6 and approximately 7.000 for
C8 before roundoff. Stable PDE solution convergence can gain an order over
boundary truncation error under suitable assumptions; these scalar derivative
tests do not establish that solution order. Existing fits also use N instead
of N-1 on closed grids and only a short resolution range.

A one-pass filter slope is not a measured final-time wall-bounded solution
slope. Every-step filtering injects its defect repeatedly. For a smooth mode,
the equivalent perturbation rate scales as `(F-I)/dt`; with dt proportional
to h, an O(h^p) pass can contribute O(h^(p-1)) per unit time. Boundary error
localization, propagation and damping determine the actual solution norm.
`filter_cfl` removes the dependence on pass count as dt changes at fixed h
below the relaxation threshold; its rate contains a factor proportional to
1/h and does not automatically restore a spatial order. A C8 filter also
needs consideration before claiming tenth-order accuracy for a filtered C10
calculation.

**Direct AMR evidence.** The existing entropy wave has rho=1+0.2sin(x),
u=0.5, constant pressure, t=0.5, no state filter or artificial properties,
and N=48,96,192 on the periodic parent. The maximum includes all patch
nodes, including covered parent nodes, matching the existing regression gate.

| Derivative / closed rows | Errors at N=48 / 96 / 192 | Successive orders |
|---|---|---|
| C6 cascade3 | 8.446e-8 / 7.679e-9 / 6.178e-10 | 3.459 / 3.636 |
| C10 cascade3 | 8.224e-8 / 5.726e-9 / 4.915e-10 | 3.844 / 3.542 |
| C6 cascade4 | 5.147e-9 / 1.505e-10 / 6.536e-12 | 5.095 / 4.526 |
| C6 Brady-Livescu | 1.507e-10 / 3.709e-12 / 7.927e-14 | 5.344 / 5.548 |

These runs use CFL=0.5. Reducing CFL to 0.125 leaves the default C6
finest-grid error at 6.287e-10 and the slopes at 3.445/3.624. Time error
does not explain the present default order. The closure change alone is
about 95 times better with cascade4 at N=192; the Brady-Livescu result there
is close enough to roundoff that its improvement ratio should not be used
as an asymptotic prediction.

A second wave, rho=1+0.2sin(3x+0.37), avoids relying on one favorable phase
and gives larger, more resolved errors. At CFL=0.125 without subcycling:

| Closed rows | Errors at N=48 / 96 / 192 | Successive orders |
|---|---|---|
| C6 cascade3 | 1.322e-5 / 1.005e-6 / 8.199e-8 | 3.718 / 3.615 |
| C6 Brady-Livescu | 2.752e-7 / 4.317e-9 / 8.018e-11 | 5.994 / 5.751 |

That is about 1,000 times less error at N=192. Subcycled Brady-Livescu gives
5.990/5.755 at CFL=0.125, versus 5.915/5.543 at CFL=0.5: temporal boundary
error starts to be visible once the spatial closure is improved. Neither
these cases nor the original wave certify viscous, shock, multidimensional,
moving-refinement, MPI, GPU or long-time stability.

The mechanism is explicit in `src/rhs.jl::_fine_plans`: extended interface
rows serve gradients, while `vplans_f` calls `mkf(deriv,d)` with the
derivative's original closed rows. Same-level patch interfaces follow the
same policy. C10 still uses the low-order C6 closure cascade for divergence.
`interface_rhs=:extended` therefore cannot improve the inviscid wave by
itself. Flux exchange inside `compute_rhs!` is within each patch's
decomposition, not across patch or refinement interfaces.

The live transfer is sixth-order Lagrange interpolation plus coincident-node
injection. Its construction hardcodes 6 in `src/levels.jl`; the standalone
transfer API supports even orders only through 8. The optional filtered
restriction/deconvolution pair has a different representation contract and
already measured 1.3-1.7 order on live point samples. It is not an accuracy
upgrade. Cubic Hermite supplies subcycled boundary states with O(dt^4)
accuracy; the time integrator is also fourth order. Neither prevents high
spatial accuracy at sufficiently small dt, but both must be separated in
an order study.

**Wall correctness probe (before R5).** `NoSlipWallBC()` documented an
adiabatic wall, but its implementation set momentum to zero and removed
kinetic energy without imposing zero normal heat flux. `compute_rhs!`
differentiated temperature normally and assembled `-kappa*grad(T)` at the
wall; there was no specialized no-slip flux correction. A diagnostic state
with rho=1, u=0, p=1+0.1x and mu0=0.01 gave energy flux -0.005 at both
wall endpoints after boundary enforcement and RHS evaluation. The linear
temperature deliberately violates adiabatic compatibility: this is a test
of missing flux imposition, not a manufactured convergence solution. A
physically correct thermal boundary treatment and its compatibility tests
are prerequisites for claiming high-order adiabatic viscous walls.
R5 now imposes the total normal species and thermal flux through `correct_flux!`
after complete assembly and before divergence. The diagnostic's two energy
fluxes are zero after this change. This fixes the physical flux contract; it
does not promote a wall accuracy order or make the compact operator/filter
globally conservative. The compatible evolution and budget checks are in
`test/wall_flux_tests.jl`, with measurements recorded in `CALIBRATION_APPENDIX.md`.

**Recommended sequence.**

1. Recalibrate the wall filter with the existing `:onesided` option and
   C6 `:cascade3` as the first default candidate. CALIBRATION_APPENDIX.md already
   records substantially improved planar Noh wall heating (64% to 27% at
   N=400) and a comparable Woodward-Colella profile. This removes the
   largest filter defect while retaining the more robust derivative.
   Validate thermal/species flux boundary conditions alongside this work.
2. Separate physical-wall and interface divergence closure selection.
   Trial C6 Brady-Livescu specifically at interfaces, using the existing
   implementation, and validate smooth, viscous, acoustic-reflection,
   shock-crossing, conservation, moving/tiled refinement and Float32
   cases. The smooth evidence above justifies this bounded experiment.
   Do not switch all physical walls with the same global keyword.
3. Offer a validated smooth Float64 wall configuration with Brady-Livescu
   and the one-sided filter. The recorded shock failures preclude a
   blanket replacement: cold Noh still fails with Brady-Livescu; cascade4
   loses stability with the one-sided filter even on a recorded smooth
   pulse. C6/C8 Brady-Livescu closed matrices also have conditioning around
   1e3-4e3, with recorded Float32 derivative errors around 1e-3 rather than
   continuing high-order convergence. Treat derivative, filter and
   D(beta D) diffusion stability together.
4. For a general 6-10th-order interface target, evaluate extended flux
   divergence with current-stage ghost fluxes. Inviscid flux can potentially
   be evaluated locally from exchanged state; viscous/artificial flux
   requires valid gradients and coefficients too. A general implementation
   needs phased work across patches (assemble, exchange, diverge), or a
   justified overlap computation. The current shared RHS workspace means
   retaining fluxes across phases has a real memory/lifetime cost. Merely
   assigning gradient plans to divergence would read invalid flux ghosts.
5. Measure and then raise transfer order as needed. Differentiating an
   O(h^r) ghost error can produce O(h^(r-1)) in a first derivative and
   O(h^(r-2)) in a second derivative before solution-level gains. Sixth-order
   value interpolation is not proof of sixth-order viscous interface
   consistency, and it cannot justify eighth/tenth-order AMR claims.
   Couple improved transfer to conservation and stability checks; injection
   is exact for coincident point samples but does not guarantee a conserved
   composite integral.

A summation-by-parts operator with penalty coupling is a longer-term option
if a provable energy estimate and conservation are required. It is not
inherently limited to sub-fourth-order interfaces: Almquist, Wang and Werpers
construct order-preserving nonconforming interpolation for diagonal-norm SBP
operators with second derivatives. Their result is a design reference, not
a proof for the present compact operators or a drop-in interpolation table.
See [the authors' paper](https://arxiv.org/abs/1806.01931).
The current high-order closure family comes from
[Brady and Livescu's boundary schemes](https://www.sciencedirect.com/science/article/abs/pii/S0045793018309356);
their analyzed stability properties should not be assumed for this package's
filter, artificial diffusion, enforcement and AMR coupling combination.

Acceptance should distinguish polynomial exactness, instantaneous RHS
truncation error, one-pass filtering and final-time solution error. Use
actual h, several fields/phases, separate wall/interface/interior maxima,
composite volume-weighted norms excluding covered parent nodes, fixed
physical refinement geometry, and a dt sweep. Check the error magnitude
and roundoff floor as well as fitted slopes. Existing regression values
should remain historical baselines until a numerical change is deliberately
validated; they are not specifications of achievable order.
